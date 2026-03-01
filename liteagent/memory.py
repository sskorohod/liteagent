"""4-layer memory system: conversation → scoped state → semantic → knowledge extractor."""

import json
import logging
import math
import os
import pickle
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Contradiction indicator words (EN + RU)
_CONTRADICTION_WORDS = {
    "not", "no", "never", "isn't", "aren't", "wasn't", "weren't", "don't",
    "doesn't", "didn't", "can't", "won't", "shouldn't", "couldn't",
    "не", "нет", "никогда", "ни", "без",
}


class MemorySystem:
    """Persistent memory with semantic recall and auto-learning."""

    def __init__(self, config: dict, client=None, provider=None):
        db_path = Path(config.get("memory", {}).get("db_path", "~/.liteagent/memory.db")).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.provider = provider or client  # backward compat
        self.config = config.get("memory", {})
        self._session_state: dict[str, Any] = {}
        self._conversations: dict[str, list] = {}  # user_id → messages
        self._features_config = config.get("features", {})
        self._embedder = self._init_embedder()
        self._init_tables()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                type TEXT DEFAULT 'fact',
                importance REAL DEFAULT 0.5,
                hash TEXT UNIQUE,
                created_at TEXT,
                accessed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                updated_at TEXT,
                PRIMARY KEY (user_id, key)
            );
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS session_summaries (
                user_id TEXT NOT NULL,
                summary TEXT,
                updated_at TEXT,
                PRIMARY KEY (user_id)
            );
            CREATE TABLE IF NOT EXISTS usage_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0,
                timestamp TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_stats(timestamp);

            -- Feature tables (metacognition, evolution, synthesis)
            CREATE TABLE IF NOT EXISTS interaction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                user_input TEXT,
                agent_response TEXT,
                tool_calls_json TEXT,
                success INTEGER DEFAULT 1,
                confidence REAL,
                model_used TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_interaction_user
                ON interaction_log(user_id, created_at);

            CREATE TABLE IF NOT EXISTS style_profiles (
                user_id TEXT PRIMARY KEY,
                formality REAL DEFAULT 0.5,
                verbosity REAL DEFAULT 0.5,
                technical_level REAL DEFAULT 0.5,
                emoji_usage REAL DEFAULT 0.0,
                language TEXT DEFAULT 'en',
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS friction_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                user_input TEXT,
                agent_response TEXT,
                extracted_lesson TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS prompt_patches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patch_text TEXT NOT NULL,
                reason TEXT,
                applied INTEGER DEFAULT 0,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS synthesized_tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                source_code TEXT NOT NULL,
                parameters_json TEXT,
                approved INTEGER DEFAULT 0,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                steps_json TEXT NOT NULL,
                trigger_pattern TEXT,
                use_count INTEGER DEFAULT 0,
                created_at TEXT
            );
        """)
        # Add embedding column if missing (migration for existing DBs)
        try:
            self.db.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Add archived_at column if missing (memory conflict resolution)
        try:
            self.db.execute("ALTER TABLE memories ADD COLUMN archived_at TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        self.db.commit()

    def _init_embedder(self):
        """Try to load sentence-transformers for semantic search, fallback to None."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
            return model
        except ImportError:
            logger.debug("sentence-transformers not installed, using keyword matching")
            return None

    def _embed(self, text: str) -> bytes | None:
        """Generate embedding as pickle bytes, or None if embedder unavailable."""
        if self._embedder is None:
            return None
        vec = self._embedder.encode(text)
        return pickle.dumps(vec)

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity between two numpy arrays."""
        dot = float(a @ b)
        norm_a = float(math.sqrt(a @ a))
        norm_b = float(math.sqrt(b @ b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    # ══════════════════════════════════════════
    # L1: CONVERSATION MEMORY
    # ══════════════════════════════════════════

    def get_history(self, user_id: str) -> list[dict]:
        """Get current conversation history."""
        return self._conversations.get(user_id, [])

    def add_message(self, user_id: str, role: str, content):
        """Add message to conversation buffer."""
        if user_id not in self._conversations:
            self._conversations[user_id] = []
        self._conversations[user_id].append({"role": role, "content": content})

    def get_compressed_history(self, user_id: str) -> list[dict]:
        """Return conversation history with compression for older messages."""
        keep_recent = self.config.get("keep_recent_messages", 6)
        messages = self.get_history(user_id)

        if len(messages) <= keep_recent:
            # Add session summary as context if available
            summary = self._get_session_summary(user_id)
            prefix = []
            if summary:
                prefix = [{"role": "user", "content": [
                    {"type": "text", "text": f"[Previous context]: {summary}",
                     "cache_control": {"type": "ephemeral"}}
                ]},
                          {"role": "assistant", "content": "Understood, I'll keep that context in mind."}]
            return prefix + messages

        # Split: old messages get summarized, recent stay exact
        recent = messages[-keep_recent:]
        summary = self._get_session_summary(user_id)
        prefix = []
        if summary:
            prefix = [{"role": "user", "content": [
                {"type": "text", "text": f"[Previous context]: {summary}",
                 "cache_control": {"type": "ephemeral"}}
            ]},
                      {"role": "assistant", "content": "Understood."}]
        return prefix + recent

    def clear_conversation(self, user_id: str):
        """Clear conversation buffer (on session end)."""
        self._conversations.pop(user_id, None)

    # ══════════════════════════════════════════
    # L2: SCOPED STATE (SQLite)
    # ══════════════════════════════════════════

    def get_state(self, key: str, user_id: str | None = None) -> Any:
        """Get scoped state. Prefix convention: user: / app: / no prefix = session."""
        if key.startswith("user:") and user_id:
            row = self.db.execute(
                "SELECT value FROM user_state WHERE user_id=? AND key=?",
                (user_id, key)).fetchone()
            return json.loads(row[0]) if row else None
        elif key.startswith("app:"):
            row = self.db.execute(
                "SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None
        else:
            return self._session_state.get(key)

    def set_state(self, key: str, value: Any, user_id: str | None = None):
        """Set scoped state."""
        now = datetime.now().isoformat()
        if key.startswith("user:") and user_id:
            self.db.execute(
                "INSERT OR REPLACE INTO user_state VALUES (?, ?, ?, ?)",
                (user_id, key, json.dumps(value), now))
            self.db.commit()
        elif key.startswith("app:"):
            self.db.execute(
                "INSERT OR REPLACE INTO app_state VALUES (?, ?, ?)",
                (key, json.dumps(value), now))
            self.db.commit()
        else:
            self._session_state[key] = value

    # ══════════════════════════════════════════
    # L3: SEMANTIC MEMORY
    # ══════════════════════════════════════════

    def recall(self, query: str, user_id: str, top_k: int = 5) -> list[dict]:
        """Find relevant memories using embeddings (if available) or keyword matching."""
        rows = self.db.execute(
            """SELECT content, type, importance, created_at, embedding
               FROM memories WHERE user_id = ? AND archived_at IS NULL
               ORDER BY importance DESC, created_at DESC
               LIMIT 50""",
            (user_id,)).fetchall()

        if not rows:
            return []

        # Try embedding-based scoring
        query_embedding = None
        if self._embedder is not None:
            query_embedding = self._embedder.encode(query)

        query_words = set(query.lower().split())
        scored = []
        for content, mtype, importance, created_at, emb_blob in rows:
            # Semantic score (embedding cosine similarity)
            semantic_score = 0.0
            if query_embedding is not None and emb_blob:
                try:
                    content_emb = pickle.loads(emb_blob)
                    semantic_score = self._cosine_similarity(query_embedding, content_emb)
                except Exception:
                    pass

            # Keyword score (fallback / supplement)
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            keyword_score = overlap / max(len(query_words), 1)

            # Combined: prefer embedding when available
            if query_embedding is not None and emb_blob:
                relevance = semantic_score * 0.7 + keyword_score * 0.3
            else:
                relevance = keyword_score

            recency = self._recency_score(created_at)
            score = relevance * 0.6 + importance * 0.3 + recency * 0.1
            scored.append({
                "content": content,
                "type": mtype,
                "score": score,
                "importance": importance,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def remember(self, content: str, user_id: str,
                       memory_type: str = "fact", importance: float = 0.5):
        """Store a new memory with deduplication, conflict detection, and optional embedding."""
        content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()

        # Dedup by hash
        existing = self.db.execute(
            "SELECT id FROM memories WHERE hash = ?", (content_hash,)).fetchone()
        if existing:
            # Update access time and maybe bump importance
            self.db.execute(
                "UPDATE memories SET accessed_at=?, importance=MIN(importance+0.1, 1.0) WHERE id=?",
                (datetime.now().isoformat(), existing[0]))
            self.db.commit()
            return

        # Conflict detection (if enabled)
        mcd_cfg = self._features_config.get("memory_conflict_detection", {})
        if mcd_cfg.get("enabled", False):
            conflicts = self.detect_memory_conflicts(
                content, user_id, memory_type,
                threshold=mcd_cfg.get("similarity_threshold", 0.75))
            if conflicts and mcd_cfg.get("auto_resolve", True):
                for conflict in conflicts:
                    action = await self.resolve_memory_conflict(
                        content, conflict["existing"], user_id)
                    self._apply_conflict_resolution(
                        action, content, conflict["existing"],
                        user_id, memory_type, importance)
                return  # Resolution handles storage

        embedding = self._embed(content)
        self.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash, created_at, accessed_at, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, content, memory_type, importance, content_hash,
             datetime.now().isoformat(), datetime.now().isoformat(), embedding))
        self.db.commit()

    def get_all_memories(self, user_id: str) -> list[dict]:
        """Get all memories for a user (for debugging/display)."""
        rows = self.db.execute(
            """SELECT content, type, importance, created_at
               FROM memories WHERE user_id = ?
               ORDER BY importance DESC, created_at DESC""",
            (user_id,)).fetchall()
        return [{"content": r[0], "type": r[1], "importance": r[2], "created_at": r[3]}
                for r in rows]

    def forget(self, user_id: str, content_fragment: str):
        """Delete memories matching a fragment."""
        self.db.execute(
            "DELETE FROM memories WHERE user_id=? AND content LIKE ?",
            (user_id, f"%{content_fragment}%"))
        self.db.commit()

    # ══════════════════════════════════════════
    # MEMORY CONFLICT DETECTION & RESOLUTION
    # ══════════════════════════════════════════

    def detect_memory_conflicts(self, content: str, user_id: str,
                                memory_type: str = "fact",
                                threshold: float = 0.75) -> list[dict]:
        """Find existing memories that may conflict with new content.

        Returns list of {"existing": {id, content, type}, "similarity": float, "conflict_type": str}.
        """
        rows = self.db.execute(
            """SELECT id, content, type, embedding
               FROM memories WHERE user_id = ? AND archived_at IS NULL
               ORDER BY created_at DESC LIMIT 100""",
            (user_id,)).fetchall()

        if not rows:
            return []

        new_embedding = None
        if self._embedder is not None:
            new_embedding = self._embedder.encode(content)

        new_words = set(content.lower().split())
        conflicts = []

        for mem_id, mem_content, mem_type, emb_blob in rows:
            # Compute semantic similarity
            similarity = 0.0
            if new_embedding is not None and emb_blob:
                try:
                    existing_emb = pickle.loads(emb_blob)
                    similarity = self._cosine_similarity(new_embedding, existing_emb)
                except Exception:
                    pass
            else:
                # Keyword fallback
                mem_words = set(mem_content.lower().split())
                overlap = len(new_words & mem_words)
                similarity = overlap / max(len(new_words | mem_words), 1)

            if similarity < threshold:
                continue

            # Check for contradiction indicators
            conflict_type = self._detect_contradiction_type(content, mem_content)
            if conflict_type:
                conflicts.append({
                    "existing": {"id": mem_id, "content": mem_content, "type": mem_type},
                    "similarity": similarity,
                    "conflict_type": conflict_type,
                })

        return conflicts

    @staticmethod
    def _detect_contradiction_type(new_content: str, old_content: str) -> str | None:
        """Detect if two similar memories contradict each other.

        Returns conflict type string or None if no contradiction detected.
        """
        new_lower = new_content.lower()
        old_lower = old_content.lower()

        # Check for negation words in either
        new_has_neg = any(w in new_lower.split() for w in _CONTRADICTION_WORDS)
        old_has_neg = any(w in old_lower.split() for w in _CONTRADICTION_WORDS)

        if new_has_neg != old_has_neg:
            return "negation"

        # Check for value replacement patterns
        # e.g., "works at Google" vs "works at Apple"
        new_words = set(new_lower.split())
        old_words = set(old_lower.split())
        common = new_words & old_words
        diff_new = new_words - old_words
        diff_old = old_words - new_words

        # If they share many words but differ in key content → replacement
        if len(common) >= 2 and diff_new and diff_old:
            common_ratio = len(common) / max(len(new_words | old_words), 1)
            if common_ratio >= 0.4:
                return "replacement"

        return None

    async def resolve_memory_conflict(self, new_content: str,
                                       existing: dict,
                                       user_id: str) -> str:
        """Use LLM to decide how to resolve a memory conflict.

        Returns action: "replace" | "merge" | "archive_old" | "keep_both".
        """
        if not self.provider:
            return "keep_both"

        prompt = (
            "Two memories about the same user may conflict. Decide the best action.\n\n"
            f"EXISTING memory: {existing['content']}\n"
            f"NEW memory: {new_content}\n\n"
            "Choose ONE action:\n"
            "- replace: the new memory replaces the old (old is outdated)\n"
            "- merge: combine both into a single updated memory\n"
            "- archive_old: keep both but mark old as archived\n"
            "- keep_both: they don't actually conflict, keep both active\n\n"
            "Return ONLY the action word, nothing else."
        )

        try:
            model = self.config.get("extraction_model", "claude-haiku-4-5-20251001")
            result = await self.provider.complete(
                model=model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            action = result.content[0].text.strip().lower()
            if action in ("replace", "merge", "archive_old", "keep_both"):
                logger.info("Memory conflict resolved: %s (old: '%s...', new: '%s...')",
                            action, existing['content'][:40], new_content[:40])
                return action
        except Exception as e:
            logger.warning("Memory conflict resolution failed: %s", e)

        return "keep_both"  # Safe default

    def _apply_conflict_resolution(self, action: str, new_content: str,
                                    existing: dict, user_id: str,
                                    memory_type: str = "fact",
                                    importance: float = 0.5):
        """Apply the chosen conflict resolution action."""
        now = datetime.now().isoformat()
        new_hash = hashlib.md5(new_content.lower().strip().encode()).hexdigest()
        embedding = self._embed(new_content)

        if action == "replace":
            # Update existing memory with new content
            self.db.execute(
                """UPDATE memories SET content=?, hash=?, embedding=?,
                   accessed_at=?, importance=? WHERE id=?""",
                (new_content, new_hash, embedding, now, importance, existing["id"]))

        elif action == "archive_old":
            # Archive old memory, insert new
            self.db.execute(
                "UPDATE memories SET archived_at=? WHERE id=?",
                (now, existing["id"]))
            self.db.execute(
                """INSERT INTO memories (user_id, content, type, importance, hash,
                   created_at, accessed_at, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, new_content, memory_type, importance, new_hash,
                 now, now, embedding))

        elif action == "merge":
            # Merge: combine both contents
            merged = f"{existing['content']} [updated: {new_content}]"
            merged_hash = hashlib.md5(merged.lower().strip().encode()).hexdigest()
            merged_emb = self._embed(merged)
            self.db.execute(
                """UPDATE memories SET content=?, hash=?, embedding=?,
                   accessed_at=?, importance=MIN(importance+0.1, 1.0) WHERE id=?""",
                (merged, merged_hash, merged_emb, now, existing["id"]))

        else:  # keep_both
            self.db.execute(
                """INSERT INTO memories (user_id, content, type, importance, hash,
                   created_at, accessed_at, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, new_content, memory_type, importance, new_hash,
                 now, now, embedding))

        self.db.commit()

    def get_archived_memories(self, user_id: str, limit: int = 20) -> list[dict]:
        """Get recently archived memories (for /conflicts command)."""
        rows = self.db.execute(
            """SELECT id, content, type, archived_at
               FROM memories WHERE user_id = ? AND archived_at IS NOT NULL
               ORDER BY archived_at DESC LIMIT ?""",
            (user_id, limit)).fetchall()
        return [{"id": r[0], "content": r[1], "type": r[2], "archived_at": r[3]}
                for r in rows]

    # ══════════════════════════════════════════
    # L4: KNOWLEDGE EXTRACTOR
    # ══════════════════════════════════════════

    async def extract_and_learn(self, user_input: str, agent_response: str, user_id: str):
        """Extract knowledge from conversation turn using cheap model."""
        if not self.provider or not self.config.get("auto_learn", True):
            return

        extraction_prompt = (
            "Analyze this conversation exchange. Extract ONLY genuinely new, "
            "specific facts — not opinions or generic statements.\n\n"
            f"User: {user_input}\n"
            f"Assistant: {agent_response}\n\n"
            'Return JSON only:\n'
            '{"facts":["..."],"preferences":["..."],"corrections":["..."],'
            '"session_summary":"one-line context update or empty string"}\n\n'
            "Rules:\n"
            "- facts: concrete info about the user (name, job, projects, etc.)\n"
            "- preferences: how user likes things done (language, format, style)\n"
            "- corrections: if user corrected a previous assumption\n"
            "- session_summary: brief note of what was discussed (for context compression)\n"
            "- Empty arrays/strings if nothing new. Don't invent facts."
        )

        try:
            model = self.config.get("extraction_model", "claude-haiku-4-5-20251001")
            result = await self.provider.complete(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": extraction_prompt}],
            )
            text = result.content[0].text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(text)

            for fact in data.get("facts", []):
                if len(fact) > 10:  # Skip trivial
                    await self.remember(fact, user_id, "fact", 0.6)
            for pref in data.get("preferences", []):
                if len(pref) > 10:
                    await self.remember(pref, user_id, "preference", 0.8)
            for correction in data.get("corrections", []):
                if len(correction) > 10:
                    await self.remember(correction, user_id, "correction", 0.9)

            # Update session summary for context compression
            summary_update = data.get("session_summary", "")
            if summary_update and len(summary_update) > 5:
                self._update_session_summary(user_id, summary_update)

            extracted = (len(data.get("facts", [])) + len(data.get("preferences", []))
                         + len(data.get("corrections", [])))
            if extracted:
                logger.debug("Extracted %d items from conversation", extracted)

        except Exception as e:
            logger.warning("Knowledge extraction failed: %s", e)

        # Friction detection for self-evolving prompt
        if self._features_config.get("self_evolving_prompt", {}).get("enabled"):
            try:
                from .evolution import detect_friction, store_friction
                signal = detect_friction(user_input)
                if signal:
                    store_friction(self.db, user_id, signal, user_input, agent_response)
            except Exception as e:
                logger.debug("Friction detection failed: %s", e)

    def _update_session_summary(self, user_id: str, summary_update: str):
        """Append to session summary."""
        existing = self._get_session_summary(user_id) or ""
        # Keep summary under ~500 chars
        new_summary = (existing + " " + summary_update).strip()
        if len(new_summary) > 500:
            new_summary = new_summary[-500:]
        self.db.execute(
            "INSERT OR REPLACE INTO session_summaries VALUES (?, ?, ?)",
            (user_id, new_summary, datetime.now().isoformat()))
        self.db.commit()

    def _get_session_summary(self, user_id: str) -> str | None:
        row = self.db.execute(
            "SELECT summary FROM session_summaries WHERE user_id=?",
            (user_id,)).fetchone()
        return row[0] if row else None

    # ══════════════════════════════════════════
    # USAGE TRACKING
    # ══════════════════════════════════════════

    def track_usage(self, user_id: str, model: str, usage, cost_usd: float = 0):
        """Log token usage for cost tracking."""
        self.db.execute(
            """INSERT INTO usage_stats
               (user_id, model, input_tokens, output_tokens, cache_read_tokens, cost_usd, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, model,
             getattr(usage, 'input_tokens', 0),
             getattr(usage, 'output_tokens', 0),
             getattr(usage, 'cache_read_input_tokens', 0),
             cost_usd,
             datetime.now().isoformat()))
        self.db.commit()

    def get_today_cost(self) -> float:
        """Get total cost for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM usage_stats WHERE timestamp LIKE ?",
            (f"{today}%",)).fetchone()
        return row[0] if row else 0.0

    def get_usage_summary(self, days: int = 7) -> dict:
        """Get usage summary for last N days."""
        rows = self.db.execute(
            """SELECT model,
                      SUM(input_tokens) as inp,
                      SUM(output_tokens) as out,
                      SUM(cache_read_tokens) as cached,
                      SUM(cost_usd) as cost,
                      COUNT(*) as calls
               FROM usage_stats
               WHERE timestamp >= datetime('now', ?)
               GROUP BY model""",
            (f"-{days} days",)).fetchall()
        return [{"model": r[0], "input_tokens": r[1], "output_tokens": r[2],
                 "cache_read_tokens": r[3], "cost_usd": r[4], "calls": r[5]}
                for r in rows]

    def get_daily_usage(self, days: int = 14) -> list[dict]:
        """Get daily aggregated usage for charting."""
        rows = self.db.execute(
            """SELECT DATE(timestamp) as day,
                      SUM(input_tokens) as inp,
                      SUM(output_tokens) as out,
                      SUM(cost_usd) as cost,
                      COUNT(*) as calls
               FROM usage_stats
               WHERE timestamp >= datetime('now', ?)
               GROUP BY DATE(timestamp)
               ORDER BY day""",
            (f"-{days} days",)).fetchall()
        return [{"date": r[0], "input_tokens": r[1], "output_tokens": r[2],
                 "cost_usd": round(r[3], 4), "calls": r[4]} for r in rows]

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by its row id. Returns True if deleted."""
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()
        return cur.rowcount > 0

    def get_all_memories(self, user_id: str = None) -> list[dict]:
        """Get all memories, optionally filtered by user_id."""
        if user_id:
            rows = self.db.execute(
                "SELECT id, user_id, content, type, importance, created_at FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT id, user_id, content, type, importance, created_at FROM memories ORDER BY created_at DESC"
            ).fetchall()
        return [{"id": r[0], "user_id": r[1], "content": r[2], "type": r[3],
                 "importance": r[4], "created_at": r[5]} for r in rows]

    def get_memory_count(self, user_id: str = None) -> int:
        """Count memories, optionally filtered by user_id."""
        if user_id:
            row = self.db.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,)).fetchone()
        else:
            row = self.db.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def get_total_usage_stats(self) -> dict:
        """Get aggregate usage stats for overview KPI."""
        row = self.db.execute(
            """SELECT COUNT(*) as calls,
                      COALESCE(SUM(input_tokens), 0) as inp,
                      COALESCE(SUM(output_tokens), 0) as out,
                      COALESCE(SUM(cost_usd), 0) as cost
               FROM usage_stats""").fetchone()
        return {"total_calls": row[0], "total_input_tokens": row[1],
                "total_output_tokens": row[2], "total_cost_usd": round(row[3], 4)}

    def prune_old_memories(self, days: int = 90, min_importance: float = 0.3) -> int:
        """Delete memories older than `days` with importance below threshold.
        Returns number of deleted rows."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cur = self.db.execute(
            "DELETE FROM memories WHERE accessed_at < ? AND importance < ?",
            (cutoff, min_importance))
        self.db.commit()
        deleted = cur.rowcount
        if deleted:
            logger.info("Pruned %d old memories (days=%d, min_importance=%.1f)",
                        deleted, days, min_importance)
        return deleted

    # ══════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════

    @staticmethod
    def _recency_score(created_at: str) -> float:
        """Score 0-1 based on how recent the memory is."""
        try:
            dt = datetime.fromisoformat(created_at)
            days_ago = (datetime.now() - dt).days
            return max(0, 1 - days_ago / 365)
        except (ValueError, TypeError):
            return 0.5

    def close(self):
        self.db.close()
