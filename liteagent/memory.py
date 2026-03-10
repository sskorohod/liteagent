"""4-layer memory system: conversation → scoped state → semantic → knowledge extractor.

Phase 1 improvements (2026-03):
- FTS5 full-text search for BM25-ranked keyword retrieval
- Hybrid recall: vector + BM25 + temporal decay via RRF fusion
- LLM-based session summary compression
- Agent memory tools (remember, forget, list)

Phase 2: Episodic memory — records interaction episodes (multi-turn conversations)
with topic shift detection, episode summaries, and episode-level recall.

Phase 3: Graph memory — entity + relation graph extracted from conversations.
Entities (person, project, tool, concept, location) with relations between them.
Graph-enhanced recall: entity neighborhood search merged into hybrid results.

Phase 4: Procedural memory — learned workflows from recurring tool patterns.
Crystallized from episodes, matched by keyword/embedding, injected into context.

Phase 5: Memory quality metrics — query logging, hit rate, latency tracking,
health check, stale memory detection.
"""

import json
import logging
import math
import os
import pickle
import sqlite3
import hashlib
import uuid
import copy
import asyncio
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

# Default RRF k parameter (from the original RRF paper)
_RRF_K = 60
_CANONICAL_PROFILE_SLOTS = frozenset({"name", "language", "role"})
_AUTO_ALIAS_IDS = frozenset({"dashboard-user", "api-user", "tg-user"})
_RESERVED_USER_IDS = frozenset({"", "default", "dashboard-user", "api-user", "tg-user", "system"})


# Backward-compat: import OllamaEmbedder from new module
from .embedders import OllamaEmbedder, create_embedder as _create_embedder
from .providers import create_provider


def _safe_parse_llm_json(text: str, fallback):
    """Parse JSON from LLM output tolerating control chars, truncation, wrong root type.

    1. Strip control chars.
    2. Try direct json.loads.
    3. Find outermost balanced { } or [ ] and retry.
    4. Return fallback on failure.
    """
    import re as _re
    text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    try:
        return json.loads(text)
    except Exception:
        pass
    for opener, closer in (('{', '}'), ('[', ']')):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:
                        break
    return fallback


class MemorySystem:
    """Persistent memory with semantic recall and auto-learning."""

    def __init__(self, config: dict, client=None, provider=None):
        db_path = Path(config.get("memory", {}).get("db_path", "~/.liteagent/memory.db")).expanduser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        # Restrict DB file permissions (owner read/write only)
        if db_path.exists():
            try:
                import stat
                os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR)  # 600
            except OSError:
                pass  # Windows doesn't support chmod
        self.db.execute("PRAGMA journal_mode=WAL")
        self.provider = provider or client  # backward compat
        self._config = config  # full config (for temporal_decay settings etc.)
        self.config = config.get("memory", {})
        # Pass through default_model for extraction (so Ollama uses its own model)
        self.config["_default_model"] = config.get("agent", {}).get("default_model", "")
        self._session_state: dict[str, Any] = {}
        self._conversations: dict[str, list] = {}  # user_id → messages
        self._features_config = config.get("features", {})
        self._active_episodes: dict[str, str] = {}  # user_id → episode_id
        max_extract_concurrency = int(self.config.get("extraction_max_concurrency", 1) or 1)
        self._extraction_semaphore = asyncio.Semaphore(max(1, max_extract_concurrency))
        self._extraction_provider = self._init_extraction_provider()
        self._extraction_provider_name = str(self.config.get("extraction_provider", "")).strip().lower()
        self._embedder = self._init_embedder()
        self._mx_daemon_task: asyncio.Task | None = None
        self._mx_daemon_running = False
        self._mx_daemon_worker_id = f"mx-{uuid.uuid4().hex[:8]}"
        self._mx_daemon_last_pause_reason = ""
        self._mx_daemon_last_pause_at = 0.0
        self._mx_local_worker_last_run = 0.0
        self._mx_local_worker_last_stats: dict[str, Any] = {}
        self._mx_shadow_cleanup_last_run = 0.0
        self._mx_shadow_cleanup_last_stats: dict[str, Any] = {}
        self._last_recall_trace: dict[str, dict[str, Any]] = {}
        self._fts_available = False
        self._init_tables()
        self._init_fts()

    def _init_extraction_provider(self):
        """Optional dedicated provider for memory extraction tasks."""
        requested = str(self.config.get("extraction_provider", "")).strip().lower()
        if not requested:
            return None
        current = str(self._config.get("agent", {}).get("provider", "")).strip().lower()
        if requested == current:
            return None
        try:
            cfg = copy.deepcopy(self._config)
            cfg.setdefault("agent", {})
            cfg["agent"]["provider"] = requested
            extraction_model = str(self.config.get("extraction_model", "")).strip()
            if extraction_model:
                cfg["agent"]["default_model"] = extraction_model
            provider = create_provider(cfg)
            logger.info("Memory extraction provider initialized: %s", requested)
            return provider
        except Exception as e:
            logger.warning("Failed to init extraction provider '%s': %s", requested, e)
            return None

    def _get_extraction_provider(self):
        """Provider used for memory-side LLM tasks."""
        return self._extraction_provider or self.provider

    def _get_extraction_model(self, fallback: str) -> str:
        """Model used for memory-side LLM tasks."""
        explicit = str(self.config.get("extraction_model", "")).strip()
        if explicit:
            return explicit
        return self.config.get("_default_model", fallback)

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
            CREATE TABLE IF NOT EXISTS user_identity_map (
                alias_user_id TEXT PRIMARY KEY,
                person_id TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_user_identity_person
                ON user_identity_map(person_id);
            CREATE TABLE IF NOT EXISTS canonical_profile_slots (
                person_id TEXT NOT NULL,
                slot_key TEXT NOT NULL,
                slot_value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                version INTEGER DEFAULT 1,
                source TEXT DEFAULT 'unknown',
                updated_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (person_id, slot_key)
            );
            CREATE INDEX IF NOT EXISTS idx_cps_person
                ON canonical_profile_slots(person_id, updated_at DESC);
            CREATE TABLE IF NOT EXISTS canonical_profile_slot_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                slot_key TEXT NOT NULL,
                slot_value TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                source TEXT DEFAULT 'unknown',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_cpsh_person
                ON canonical_profile_slot_history(person_id, slot_key, created_at DESC);
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_chat_history_user
                ON chat_history(user_id, created_at);

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
                rating INTEGER,
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

            CREATE TABLE IF NOT EXISTS thinking_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                note_type TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                normalized_content TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                novelty REAL DEFAULT 0.5,
                recurrence REAL DEFAULT 1.0,
                strategic_importance REAL DEFAULT 0.5,
                score REAL DEFAULT 0.5,
                status TEXT DEFAULT 'active',
                meta_json TEXT,
                first_seen_at TEXT,
                last_seen_at TEXT,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(user_id, note_type, normalized_content)
            );
            CREATE INDEX IF NOT EXISTS idx_thinking_notes_user
                ON thinking_notes(user_id, score DESC, last_seen_at DESC);
            CREATE INDEX IF NOT EXISTS idx_thinking_notes_type
                ON thinking_notes(user_id, note_type, score DESC, last_seen_at DESC);

            CREATE TABLE IF NOT EXISTS thinking_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source_note_id INTEGER NOT NULL,
                target_note_id INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 0.5,
                created_at TEXT,
                updated_at TEXT,
                UNIQUE(user_id, source_note_id, target_note_id, relation_type)
            );
            CREATE INDEX IF NOT EXISTS idx_thinking_edges_source
                ON thinking_edges(user_id, source_note_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_thinking_edges_target
                ON thinking_edges(user_id, target_note_id, relation_type);

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

            CREATE TABLE IF NOT EXISTS file_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                storage_key TEXT UNIQUE NOT NULL,
                original_name TEXT NOT NULL,
                mime_type TEXT DEFAULT 'application/octet-stream',
                size_bytes INTEGER DEFAULT 0,
                source TEXT DEFAULT 'unknown',
                user_id TEXT DEFAULT 'system',
                description TEXT DEFAULT '',
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now')),
                accessed_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_file_index_user
                ON file_index(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_file_index_source
                ON file_index(source);

            -- Tool analytics: per-call tracking for performance and reliability insights
            CREATE TABLE IF NOT EXISTS tool_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                duration_ms INTEGER DEFAULT 0,
                success INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_tool_analytics_name
                ON tool_analytics(tool_name, created_at);
            CREATE INDEX IF NOT EXISTS idx_tool_analytics_user
                ON tool_analytics(user_id, created_at);

            -- Phase 2: Episodic memory
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                summary TEXT,
                outcome TEXT DEFAULT 'unknown',
                tool_sequence TEXT DEFAULT '[]',
                topics TEXT DEFAULT '[]',
                embedding BLOB,
                turn_count INTEGER DEFAULT 0,
                created_at TEXT,
                closed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_user
                ON episodes(user_id, created_at);

            CREATE TABLE IF NOT EXISTS episode_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                turn_index INTEGER,
                user_input TEXT,
                agent_response TEXT,
                tool_calls TEXT DEFAULT '[]',
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episode_turns_ep
                ON episode_turns(episode_id, turn_index);

            -- Phase 3: Graph memory (entities + relations)
            CREATE TABLE IF NOT EXISTS memory_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                embedding BLOB,
                first_seen TEXT,
                last_seen TEXT,
                mention_count INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_entities_user
                ON memory_entities(user_id, name);
            CREATE INDEX IF NOT EXISTS idx_entities_type
                ON memory_entities(user_id, entity_type);

            CREATE TABLE IF NOT EXISTS memory_relations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                evidence TEXT DEFAULT '',
                created_at TEXT,
                updated_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_relations_source
                ON memory_relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_relations_target
                ON memory_relations(target_id);
            CREATE INDEX IF NOT EXISTS idx_relations_user
                ON memory_relations(user_id);

            CREATE TABLE IF NOT EXISTS memory_entity_mentions (
                entity_id TEXT NOT NULL,
                memory_id INTEGER NOT NULL,
                PRIMARY KEY (entity_id, memory_id)
            );

            -- Phase 4: Procedural memory (learned workflows)
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                trigger_patterns TEXT DEFAULT '[]',
                steps TEXT NOT NULL DEFAULT '[]',
                preconditions TEXT DEFAULT '',
                success_rate REAL DEFAULT 1.0,
                use_count INTEGER DEFAULT 0,
                user_id TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT,
                last_used TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_procedures_user
                ON procedures(user_id);

            -- Phase 5: Memory quality metrics
            CREATE TABLE IF NOT EXISTS memory_query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                search_mode TEXT DEFAULT 'hybrid',
                result_count INTEGER DEFAULT 0,
                top_score REAL DEFAULT 0.0,
                latency_ms REAL DEFAULT 0.0,
                was_used INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_query_log_user
                ON memory_query_log(user_id, created_at);
            CREATE TABLE IF NOT EXISTS memory_query_affinity (
                user_id TEXT NOT NULL,
                query_norm TEXT NOT NULL,
                memory_id INTEGER NOT NULL,
                hit_count INTEGER DEFAULT 1,
                total_strength REAL DEFAULT 1.0,
                last_source TEXT DEFAULT '',
                last_used TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, query_norm, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_mqa_user_query
                ON memory_query_affinity(user_id, query_norm, last_used DESC);
            CREATE INDEX IF NOT EXISTS idx_mqa_user_memory
                ON memory_query_affinity(user_id, memory_id, last_used DESC);
            CREATE TABLE IF NOT EXISTS memory_query_penalty (
                user_id TEXT NOT NULL,
                query_norm TEXT NOT NULL,
                memory_id INTEGER NOT NULL,
                miss_count INTEGER DEFAULT 1,
                total_penalty REAL DEFAULT 1.0,
                last_source TEXT DEFAULT '',
                last_used TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, query_norm, memory_id)
            );
            CREATE INDEX IF NOT EXISTS idx_mqp_user_query
                ON memory_query_penalty(user_id, query_norm, last_used DESC);
            CREATE INDEX IF NOT EXISTS idx_mqp_user_memory
                ON memory_query_penalty(user_id, memory_id, last_used DESC);
            CREATE TABLE IF NOT EXISTS memory_recall_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                strategy TEXT DEFAULT 'hybrid',
                intent_slot TEXT DEFAULT '',
                profile_expected TEXT DEFAULT '',
                profile_hit INTEGER DEFAULT 0,
                top_memories_json TEXT DEFAULT '[]',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_mrt_user_created
                ON memory_recall_traces(user_id, created_at DESC);
            CREATE TABLE IF NOT EXISTS memory_extraction_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                total_candidates INTEGER DEFAULT 0,
                saved_count INTEGER DEFAULT 0,
                dropped_pollution INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_mer_user_created
                ON memory_extraction_runs(user_id, created_at DESC);

            -- Phase 6: Memory exchange (token-efficient context packs)
            CREATE TABLE IF NOT EXISTS memory_context_packs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                query_hint TEXT DEFAULT '',
                title TEXT DEFAULT '',
                content TEXT NOT NULL,
                source_memory_ids TEXT DEFAULT '[]',
                relevance REAL DEFAULT 0.5,
                trust REAL DEFAULT 0.5,
                novelty REAL DEFAULT 0.5,
                token_estimate INTEGER DEFAULT 0,
                score REAL DEFAULT 0.0,
                hit_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                last_used TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_mxp_user_score
                ON memory_context_packs(user_id, score DESC, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_mxp_user_hint
                ON memory_context_packs(user_id, query_hint);

            CREATE TABLE IF NOT EXISTS memory_shadow_predictions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                anchor_query TEXT NOT NULL,
                predicted_query TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                pack_id TEXT,
                hit_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'ready',
                created_at TEXT,
                used_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_msp_user_pred
                ON memory_shadow_predictions(user_id, predicted_query, created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_exchange_intents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source TEXT DEFAULT 'turn',
                anchor_query TEXT DEFAULT '',
                payload_json TEXT DEFAULT '{}',
                processed INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_mxi_user_created
                ON memory_exchange_intents(user_id, created_at DESC);
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
        # Add file_meta column (JSON: {filename, mime_type, size_bytes})
        try:
            self.db.execute("ALTER TABLE memories ADD COLUMN file_meta TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        # Memory exchange queue lifecycle columns (for daemon processing)
        try:
            self.db.execute("ALTER TABLE memory_exchange_intents ADD COLUMN priority INTEGER DEFAULT 5")
        except sqlite3.OperationalError:
            pass
        try:
            self.db.execute("ALTER TABLE memory_exchange_intents ADD COLUMN status TEXT DEFAULT 'queued'")
        except sqlite3.OperationalError:
            pass
        try:
            self.db.execute("ALTER TABLE memory_exchange_intents ADD COLUMN attempts INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            self.db.execute("ALTER TABLE memory_exchange_intents ADD COLUMN last_error TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            self.db.execute("ALTER TABLE memory_exchange_intents ADD COLUMN locked_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            self.db.execute("ALTER TABLE memory_exchange_intents ADD COLUMN locked_by TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            self.db.execute("ALTER TABLE memory_exchange_intents ADD COLUMN processed_at TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            self.db.execute(
                """UPDATE memory_exchange_intents
                   SET status = CASE
                       WHEN processed = 1 THEN 'done'
                       WHEN status IS NULL OR status = '' THEN 'queued'
                       ELSE status
                   END"""
            )
        except sqlite3.OperationalError:
            pass
        try:
            self.db.execute(
                """CREATE INDEX IF NOT EXISTS idx_mxi_status_priority
                   ON memory_exchange_intents(status, priority, created_at)"""
            )
        except sqlite3.OperationalError:
            pass
        # Migration: add rating column if it doesn't exist yet
        try:
            self.db.execute("ALTER TABLE interaction_log ADD COLUMN rating INTEGER")
        except sqlite3.OperationalError:
            pass
        # Migration: add tool_analytics table if it doesn't exist yet
        try:
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS tool_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool_name TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    duration_ms INTEGER DEFAULT 0,
                    success INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT (datetime('now'))
                )"""
            )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_analytics_name"
                " ON tool_analytics(tool_name, created_at)"
            )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_analytics_user"
                " ON tool_analytics(user_id, created_at)"
            )
        except sqlite3.OperationalError:
            pass
        self.db.commit()

    def _init_fts(self):
        """Initialize FTS5 virtual table for BM25 keyword search."""
        if not self.config.get("fts_enabled", True):
            return
        try:
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    content,
                    memory_id UNINDEXED,
                    user_id UNINDEXED,
                    tokenize='unicode61'
                )
            """)
            self.db.commit()
            self._fts_available = True
            # Backfill: sync existing memories that aren't in FTS yet
            self._backfill_fts()
            logger.debug("Memory FTS5 index initialized")
        except Exception as e:
            logger.debug("FTS5 not available for memory: %s", e)

    def _backfill_fts(self):
        """Backfill FTS5 index with existing memories not yet indexed."""
        try:
            # Find memories not in FTS
            rows = self.db.execute("""
                SELECT m.id, m.content, m.user_id
                FROM memories m
                WHERE m.id NOT IN (SELECT CAST(memory_id AS INTEGER) FROM memory_fts)
                  AND m.archived_at IS NULL
            """).fetchall()
            if rows:
                for mem_id, content, uid in rows:
                    self.db.execute(
                        "INSERT INTO memory_fts (content, memory_id, user_id) VALUES (?, ?, ?)",
                        (content, str(mem_id), uid))
                self.db.commit()
                logger.debug("Backfilled %d memories into FTS5 index", len(rows))
        except Exception as e:
            logger.debug("FTS5 backfill skipped: %s", e)

    def _fts_insert(self, memory_id: int, content: str, user_id: str):
        """Insert a memory into FTS5 index."""
        if not self._fts_available:
            return
        try:
            self.db.execute(
                "INSERT INTO memory_fts (content, memory_id, user_id) VALUES (?, ?, ?)",
                (content, str(memory_id), user_id))
        except Exception as e:
            logger.debug("FTS5 insert failed: %s", e)

    def _fts_delete(self, memory_id: int):
        """Delete a memory from FTS5 index."""
        if not self._fts_available:
            return
        try:
            self.db.execute(
                "DELETE FROM memory_fts WHERE memory_id = ?", (str(memory_id),))
        except Exception as e:
            logger.debug("FTS5 delete failed: %s", e)

    def _fts_search(self, query: str, user_id: str, top_k: int = 20) -> list[dict]:
        """BM25 keyword search via FTS5. Returns [{id, content, score}, ...]."""
        if not self._fts_available:
            return []
        try:
            # Escape FTS5 special characters
            fts_query = " ".join(
                word for word in query.split()
                if len(word) >= 2 and not any(c in word for c in '"*(){}[]')
            )
            if not fts_query:
                return []
            rows = self.db.execute("""
                SELECT f.memory_id, f.content, bm25(memory_fts) as score
                FROM memory_fts f
                WHERE memory_fts MATCH ? AND f.user_id = ?
                ORDER BY score
                LIMIT ?
            """, (fts_query, user_id, top_k)).fetchall()
            return [{"id": int(r[0]), "content": r[1], "score": -r[2]}  # bm25() returns negative
                    for r in rows]
        except Exception as e:
            logger.debug("FTS5 search failed: %s", e)
            return []

    def _init_embedder(self):
        """Initialize embedder via unified embedders module."""
        return _create_embedder(self._config)

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
        """Get current conversation history (RAM buffer)."""
        user_id = self.get_canonical_person_id(user_id)
        return self._conversations.get(user_id, [])

    @staticmethod
    def _content_for_model(content):
        """Normalize persisted UI-rich chat content into plain model-friendly text."""
        if isinstance(content, dict) and "text" in content:
            return content.get("text", "")
        return content

    def add_message(self, user_id: str, role: str, content):
        """Add message to conversation buffer AND persist to SQLite."""
        user_id = self.get_canonical_person_id(user_id)
        if user_id not in self._conversations:
            self._conversations[user_id] = []
        self._conversations[user_id].append({"role": role, "content": self._content_for_model(content)})
        # Persist
        content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
        try:
            self.db.execute(
                "INSERT INTO chat_history (user_id, role, content) VALUES (?, ?, ?)",
                (user_id, role, content_str))
            self.db.commit()
        except Exception as e:
            logger.warning("Failed to persist chat message: %s", e)

    def load_history(self, user_id: str, limit: int = 2000) -> list[dict]:
        """Load persisted chat history from SQLite into RAM buffer."""
        user_id = self.get_canonical_person_id(user_id)
        rows = self.db.execute(
            "SELECT role, content FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)).fetchall()
        rows.reverse()  # oldest first
        messages = []
        for role, content_str in rows:
            # Try to parse as JSON (for structured content), fallback to plain string
            try:
                content = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                content = content_str
            messages.append({"role": role, "content": self._content_for_model(content)})
        self._conversations[user_id] = messages
        return messages

    def get_chat_history_for_display(self, user_id: str, limit: int = 100) -> list[dict]:
        """Get chat history with timestamps for dashboard display."""
        user_id = self.get_canonical_person_id(user_id)
        rows = self.db.execute(
            "SELECT role, content, created_at FROM chat_history WHERE user_id=? ORDER BY id DESC LIMIT ?",
            (user_id, limit)).fetchall()
        rows.reverse()
        result = []
        for role, content_str, created_at in rows:
            try:
                content = json.loads(content_str)
            except (json.JSONDecodeError, TypeError):
                content = content_str
            # For display, flatten structured content to string
            if isinstance(content, list):
                text = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content)
                display_content = text
            elif isinstance(content, dict) and "text" in content:
                display_content = content
            elif isinstance(content, str):
                display_content = content
            else:
                display_content = str(content)
            result.append({"role": role, "content": display_content, "created_at": created_at})
        return result

    def clear_chat_history(self, user_id: str):
        """Clear persisted chat history for a user."""
        user_id = self.get_canonical_person_id(user_id)
        self.db.execute("DELETE FROM chat_history WHERE user_id=?", (user_id,))
        self.db.commit()

    @staticmethod
    def _estimate_tokens(messages: list) -> int:
        """Token estimate using UTF-8 byte length for multi-language accuracy.

        UTF-8 Cyrillic: 2 bytes/char → ~2 chars/token.
        UTF-8 ASCII/Latin: 1 byte/char → ~4 chars/token.
        Using bytes//4 gives a balanced, slightly conservative estimate for
        mixed Russian/English content (errs on the side of caution).
        """
        total = 0
        for m in messages:
            c = m.get("content", "")
            text = c if isinstance(c, str) else json.dumps(c, ensure_ascii=False)
            total += len(text.encode("utf-8")) // 4 + 10  # +10 msg overhead
        return total

    # Tokens reserved for system prompt + tools + memory context + response.
    # soul.md ≈ 2K, tools ≈ 3K, memory context ≈ 1K, response headroom ≈ 2K → 8K total.
    _SYSTEM_OVERHEAD_TOKENS: int = 8_000

    def get_compressed_history(self, user_id: str) -> list[dict]:
        """Return full conversation history trimmed only by token budget.

        Mirrors ChatGPT / Claude.ai / Gemini behaviour: the entire conversation
        is sent to the model on every turn; oldest messages are dropped first
        when the effective token budget is exceeded.

        Budget = max_history_tokens − system overhead (soul + tools + memory).
        """
        user_id = self.get_canonical_person_id(user_id)
        messages = list(self.get_history(user_id))  # copy — don't mutate RAM buffer

        # Effective budget for history alone (system prompt overhead reserved separately).
        max_tokens = self.config.get("max_history_tokens", 100_000)
        effective = max(2_000, max_tokens - self._SYSTEM_OVERHEAD_TOKENS)

        total = self._estimate_tokens(messages)
        if total <= effective:
            return messages  # fast path — everything fits

        # Binary search for the cut index: find the smallest prefix to drop
        # so the remaining messages fit within the budget. O(n log n) vs O(n²).
        lo, hi = 0, len(messages) - 2  # always keep at least 2 messages
        while lo < hi:
            mid = (lo + hi) // 2
            if self._estimate_tokens(messages[mid:]) <= effective:
                hi = mid
            else:
                lo = mid + 1

        trimmed = messages[lo:]
        logger.debug(
            "History trimmed: dropped %d messages (budget %d tokens, had %d)",
            lo, effective, total,
        )
        return trimmed

    def clear_conversation(self, user_id: str):
        """Clear conversation buffer (on session end)."""
        user_id = self.get_canonical_person_id(user_id)
        self._conversations.pop(user_id, None)

    # ══════════════════════════════════════════
    # L2: SCOPED STATE (SQLite)
    # ══════════════════════════════════════════

    def get_state(self, key: str, user_id: str | None = None) -> Any:
        """Get scoped state. Prefix convention: user: / app: / no prefix = session."""
        if key.startswith("user:") and user_id:
            user_id = self.get_canonical_person_id(user_id)
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
            user_id = self.get_canonical_person_id(user_id)
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
    # USER PROFILE SNAPSHOT (PINNED MEMORY)
    # ══════════════════════════════════════════

    @staticmethod
    def _normalize_user_id(user_id: str | None) -> str:
        return str(user_id or "").strip() or "default"

    def get_canonical_person_id(self, user_id: str | None) -> str:
        """Resolve user/channel alias to canonical person_id."""
        raw = self._normalize_user_id(user_id)
        try:
            row = self.db.execute(
                "SELECT person_id FROM user_identity_map WHERE alias_user_id = ?",
                (raw,),
            ).fetchone()
            if row and row[0]:
                return str(row[0])
        except Exception:
            pass
        return raw

    def set_user_alias(self, alias_user_id: str, person_id: str, *,
                       source: str = "manual", confidence: float = 1.0) -> str:
        """Persist alias -> canonical person mapping."""
        alias = self._normalize_user_id(alias_user_id)
        person = self._normalize_user_id(person_id)
        now = datetime.now().isoformat()
        conf = max(0.0, min(float(confidence), 1.0))
        self.db.execute(
            """INSERT INTO user_identity_map
               (alias_user_id, person_id, source, confidence, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(alias_user_id) DO UPDATE SET
                   person_id = excluded.person_id,
                   source = excluded.source,
                   confidence = excluded.confidence,
                   updated_at = excluded.updated_at""",
            (alias, person, source[:40], conf, now, now),
        )
        self.db.commit()
        if alias != person:
            self._merge_identity_data(alias, person)
        return person

    def _merge_identity_data(self, alias_user_id: str, person_id: str) -> None:
        """Best-effort merge of persisted data when alias is mapped to canonical person."""
        alias = self._normalize_user_id(alias_user_id)
        person = self._normalize_user_id(person_id)
        if alias == person:
            return

        # Generic tables where user_id can be rewritten without key conflicts.
        generic_tables = (
            "memories",
            "chat_history",
            "interaction_log",
            "friction_signals",
            "usage_stats",
            "episodes",
            "memory_entities",
            "memory_relations",
            "procedures",
            "memory_query_log",
            "memory_context_packs",
            "memory_shadow_predictions",
            "memory_exchange_intents",
            "memory_recall_traces",
            "memory_extraction_runs",
            "file_index",
        )
        for table in generic_tables:
            try:
                self.db.execute(
                    f"UPDATE {table} SET user_id = ? WHERE user_id = ?",
                    (person, alias),
                )
            except Exception:
                pass

        try:
            # Merge user_state rows conservatively (existing person values win).
            self.db.execute(
                """INSERT OR IGNORE INTO user_state (user_id, key, value, updated_at)
                   SELECT ?, key, value, updated_at
                   FROM user_state
                   WHERE user_id = ?""",
                (person, alias),
            )
            self.db.execute("DELETE FROM user_state WHERE user_id = ?", (alias,))
        except Exception:
            pass

        try:
            # Merge session summaries into canonical row.
            alias_row = self.db.execute(
                "SELECT summary FROM session_summaries WHERE user_id = ?",
                (alias,),
            ).fetchone()
            person_row = self.db.execute(
                "SELECT summary FROM session_summaries WHERE user_id = ?",
                (person,),
            ).fetchone()
            alias_summary = str(alias_row[0] or "") if alias_row else ""
            person_summary = str(person_row[0] or "") if person_row else ""
            merged = " ".join([person_summary.strip(), alias_summary.strip()]).strip()
            if merged:
                self.db.execute(
                    "INSERT OR REPLACE INTO session_summaries VALUES (?, ?, ?)",
                    (person, merged[:1200], datetime.now().isoformat()),
                )
            self.db.execute("DELETE FROM session_summaries WHERE user_id = ?", (alias,))
        except Exception:
            pass

        try:
            self.db.execute(
                """INSERT OR IGNORE INTO style_profiles
                   (user_id, formality, verbosity, technical_level, emoji_usage, language, updated_at)
                   SELECT ?, formality, verbosity, technical_level, emoji_usage, language, updated_at
                   FROM style_profiles
                   WHERE user_id = ?""",
                (person, alias),
            )
            self.db.execute("DELETE FROM style_profiles WHERE user_id = ?", (alias,))
        except Exception:
            pass

        # Merge canonical slots manually due composite PK.
        try:
            rows = self.db.execute(
                """SELECT slot_key, slot_value, confidence, version, source, updated_at
                   FROM canonical_profile_slots
                   WHERE person_id = ?""",
                (alias,),
            ).fetchall()
            for slot_key, slot_value, confidence, version, source, updated_at in rows:
                cur = self.db.execute(
                    """SELECT confidence FROM canonical_profile_slots
                       WHERE person_id = ? AND slot_key = ?""",
                    (person, slot_key),
                ).fetchone()
                if not cur:
                    self.db.execute(
                        """INSERT INTO canonical_profile_slots
                           (person_id, slot_key, slot_value, confidence, version, source, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (person, slot_key, slot_value, confidence, version, source, updated_at),
                    )
                elif float(confidence or 0.0) > float(cur[0] or 0.0):
                    self.db.execute(
                        """UPDATE canonical_profile_slots
                           SET slot_value = ?, confidence = ?, version = ?, source = ?, updated_at = ?
                           WHERE person_id = ? AND slot_key = ?""",
                        (slot_value, confidence, version, source, updated_at, person, slot_key),
                    )
            self.db.execute(
                """INSERT INTO canonical_profile_slot_history
                   (person_id, slot_key, slot_value, confidence, source, created_at)
                   SELECT ?, slot_key, slot_value, confidence, source, created_at
                   FROM canonical_profile_slot_history
                   WHERE person_id = ?""",
                (person, alias),
            )
            self.db.execute("DELETE FROM canonical_profile_slots WHERE person_id = ?", (alias,))
            self.db.execute("DELETE FROM canonical_profile_slot_history WHERE person_id = ?", (alias,))
        except Exception:
            pass

        # Merge in-memory conversation buffers.
        if alias in self._conversations:
            current = self._conversations.get(person, [])
            current.extend(self._conversations.get(alias, []))
            self._conversations[person] = current
            self._conversations.pop(alias, None)

        self.db.commit()

    def get_aliases_for_person(self, person_id: str | None) -> list[str]:
        person = self._normalize_user_id(person_id)
        rows = self.db.execute(
            "SELECT alias_user_id FROM user_identity_map WHERE person_id = ? ORDER BY alias_user_id ASC",
            (person,),
        ).fetchall()
        aliases = [str(r[0]) for r in rows if r and r[0]]
        if person not in aliases:
            aliases.insert(0, person)
        return aliases

    def get_identity_snapshot(self, user_id: str | None) -> dict[str, Any]:
        raw = self._normalize_user_id(user_id)
        person = self.get_canonical_person_id(raw)
        aliases = self.get_aliases_for_person(person)
        return {
            "user_id": raw,
            "person_id": person,
            "aliases": aliases,
            "is_alias": raw != person,
        }

    @staticmethod
    def _canonical_slot_key(slot_key: str) -> str:
        return str(slot_key or "").strip().lower()

    def get_canonical_slot(self, user_id: str, slot_key: str) -> dict[str, Any] | None:
        person_id = self.get_canonical_person_id(user_id)
        key = self._canonical_slot_key(slot_key)
        if key not in _CANONICAL_PROFILE_SLOTS:
            return None
        row = self.db.execute(
            """SELECT slot_value, confidence, version, source, updated_at
               FROM canonical_profile_slots
               WHERE person_id = ? AND slot_key = ?""",
            (person_id, key),
        ).fetchone()
        if not row:
            return None
        return {
            "person_id": person_id,
            "slot_key": key,
            "slot_value": str(row[0] or ""),
            "confidence": float(row[1] or 0.0),
            "version": int(row[2] or 1),
            "source": str(row[3] or ""),
            "updated_at": row[4],
        }

    def get_canonical_profile(self, user_id: str) -> dict[str, dict[str, Any]]:
        person_id = self.get_canonical_person_id(user_id)
        rows = self.db.execute(
            """SELECT slot_key, slot_value, confidence, version, source, updated_at
               FROM canonical_profile_slots
               WHERE person_id = ?""",
            (person_id,),
        ).fetchall()
        profile: dict[str, dict[str, Any]] = {}
        for slot_key, slot_value, confidence, version, source, updated_at in rows:
            key = str(slot_key or "")
            profile[key] = {
                "value": str(slot_value or ""),
                "confidence": float(confidence or 0.0),
                "version": int(version or 1),
                "source": str(source or ""),
                "updated_at": updated_at,
            }
        return profile

    def get_canonical_profile_history(self, user_id: str, slot_key: str,
                                      limit: int = 20) -> list[dict[str, Any]]:
        person_id = self.get_canonical_person_id(user_id)
        key = self._canonical_slot_key(slot_key)
        rows = self.db.execute(
            """SELECT slot_value, confidence, source, created_at
               FROM canonical_profile_slot_history
               WHERE person_id = ? AND slot_key = ?
               ORDER BY id DESC LIMIT ?""",
            (person_id, key, max(1, min(int(limit), 200))),
        ).fetchall()
        return [{
            "slot_value": str(r[0] or ""),
            "confidence": float(r[1] or 0.0),
            "source": str(r[2] or ""),
            "created_at": r[3],
        } for r in rows]

    @staticmethod
    def _normalize_slot_value(slot_key: str, value: str) -> str:
        """Normalize profile slot value for deterministic comparison."""
        v = " ".join(str(value or "").strip().split())
        if not v:
            return ""
        v = v.strip(" \t\r\n.,!?;:\"'`")
        if slot_key == "name":
            return v[:60].strip()
        return v[:120].strip()

    @staticmethod
    def _extract_slot_signals(slot_key: str, text: str) -> tuple[list[str], list[str]]:
        """Extract positive/negative slot signals from free text."""
        import re

        s = " ".join(str(text or "").strip().split())
        if not s:
            return [], []

        if slot_key != "name":
            extracted = MemorySystem._extract_profile_facts(s)
            val = MemorySystem._normalize_slot_value(slot_key, extracted.get(slot_key, ""))
            return ([val] if val else []), []

        token = r"([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{1,40}(?:\s+[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{1,40})?)"
        positives: list[str] = []
        negatives: list[str] = []
        blocked = {
            "user", "a user", "the user", "assistant", "memory",
            "пользователь", "ассистент", "память", "имя",
        }

        def _push(dst: list[str], raw: str):
            val = MemorySystem._normalize_slot_value("name", str(raw or "").replace("_", " "))
            if not val:
                return
            low = val.lower()
            if low in blocked or len(low) < 2 or any(ch.isdigit() for ch in low):
                return
            dst.append(val)

        positive_patterns = (
            rf"\bменя\s+зовут\s+{token}",
            rf"\bмо[её]\s+имя\s+{token}",
            rf"\bпользовател[ья]\s+зовут\s+{token}",
            rf"\bимя\s+пользователя\s*[:\-—]\s*{token}",
            rf"\bзови\s+меня\s+{token}",
            rf"\bmy\s+name\s+is\s+{token}",
            rf"\bcall\s+me\s+{token}",
            rf"\buser(?:'s)?\s+name\s+is\s+{token}",
            rf"\buser\s+name\s*[:\-]\s*{token}",
            rf"^\s*{token}\s+запиши(?:\s+в\s+свою\s+память)?\s*$",
            rf"^\s*{token}\s+запомни\s*$",
        )
        for pat in positive_patterns:
            for m in re.finditer(pat, s, flags=re.IGNORECASE):
                _push(positives, m.group(1))

        negative_patterns = (
            r"\bname_is_not_([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё_\-]{1,50})\b",
            rf"\bname[_\s]*is[_\s]*not[_\s]+{token}\b",
            rf"\b(мо[её]\s+)?имя\s+не\s+{token}\b",
            rf"\b(меня|тебя)\s+зовут\s+не\s+{token}\b",
            rf"\bа\s+не\s+{token}\b",
        )
        for pat in negative_patterns:
            for m in re.finditer(pat, s, flags=re.IGNORECASE):
                grp = m.group(1)
                if grp and grp.lower() in {"меня", "тебя"} and m.lastindex and m.lastindex >= 2:
                    grp = m.group(2)
                _push(negatives, grp)

        # Stable order + dedupe
        positives = list(dict.fromkeys(positives))
        negatives = list(dict.fromkeys(negatives))
        return positives, negatives

    def resolve_profile_slot(self, user_id: str, slot_key: str, *,
                             lookback: int = 300, auto_heal: bool = True) -> dict[str, Any]:
        """Resolve slot value from canonical/profile/memory/chat evidence."""
        uid = self.get_canonical_person_id(user_id)
        key = self._canonical_slot_key(slot_key)
        result: dict[str, Any] = {
            "slot_key": key,
            "value": "",
            "confidence": 0.0,
            "source": "none",
            "evidence": [],
            "auto_healed": False,
        }
        if key not in _CANONICAL_PROFILE_SLOTS:
            return result

        candidates: dict[str, dict[str, Any]] = {}
        negatives: set[str] = set()

        def _add_negative(raw_value: str):
            val = self._normalize_slot_value(key, raw_value)
            if not val:
                return
            low = val.lower()
            negatives.add(low)
            candidates.pop(low, None)

        def _add_candidate(raw_value: str, score: float, source: str, evidence: str = ""):
            val = self._normalize_slot_value(key, raw_value)
            if not val:
                return
            low = val.lower()
            if low in negatives:
                return
            if key == "name":
                try:
                    if self.is_slot_value_contradicted(uid, key, val):
                        return
                except Exception:
                    pass
            item = candidates.get(low)
            if not item:
                item = {
                    "value": val,
                    "score": 0.0,
                    "max_component": 0.0,
                    "source": "",
                    "evidence": [],
                }
                candidates[low] = item
            comp = max(0.0, float(score))
            item["score"] += comp
            if comp >= float(item.get("max_component", 0.0)):
                item["max_component"] = comp
                item["source"] = source[:40]
            if evidence:
                item["evidence"].append(str(evidence)[:160])

        canonical = self.get_canonical_slot(uid, key)
        cur_value = ""
        cur_conf = 0.0
        cur_contradicted = False
        if canonical and canonical.get("slot_value"):
            cur_value = self._normalize_slot_value(key, canonical.get("slot_value", ""))
            cur_conf = float(canonical.get("confidence", 0.0) or 0.0)
            if cur_value:
                cur_contradicted = self.is_slot_value_contradicted(uid, key, cur_value)
                if cur_contradicted:
                    _add_negative(cur_value)
                else:
                    _add_candidate(cur_value, 2.0 + min(max(cur_conf, 0.0), 1.0) * 1.2,
                                   "canonical", f"canonical:{cur_conf:.2f}")

        raw_profile = self.get_state("user:profile_facts", user_id=uid)
        if isinstance(raw_profile, dict):
            pval = self._normalize_slot_value(key, raw_profile.get(key, ""))
            if pval:
                if key == "name" and self.is_slot_value_contradicted(uid, key, pval):
                    _add_negative(pval)
                else:
                    _add_candidate(pval, 1.35, "profile_state", "profile_state")

        lim = max(20, min(int(lookback), 1000))

        mem_rows = self.db.execute(
            """SELECT content, type, importance
               FROM memories
               WHERE user_id = ? AND archived_at IS NULL
                 AND type IN ('fact', 'preference', 'correction')
               ORDER BY id DESC LIMIT ?""",
            (uid, lim),
        ).fetchall()
        mem_total = max(len(mem_rows), 1)
        mem_type_weight = {"correction": 1.15, "fact": 0.82, "preference": 0.38}
        for idx, row in enumerate(mem_rows):
            content = str((row or [""])[0] or "")
            mtype = str((row or ["", "fact"])[1] or "fact")
            importance = float((row or ["", "", 0.5])[2] or 0.5)
            positives, negs = self._extract_slot_signals(key, content)
            for n in negs:
                _add_negative(n)
            if not positives:
                continue
            recency = max(0.1, 1.0 - (idx / mem_total))
            score = 0.42 + mem_type_weight.get(mtype, 0.45) + min(max(importance, 0.0), 1.0) * 0.35 + recency * 0.45
            for p in positives:
                _add_candidate(p, score, f"memory:{mtype}", content)

        chat_rows = self.db.execute(
            """SELECT content
               FROM chat_history
               WHERE user_id = ? AND role = 'user'
               ORDER BY id DESC LIMIT ?""",
            (uid, lim),
        ).fetchall()
        chat_total = max(len(chat_rows), 1)
        for idx, row in enumerate(chat_rows):
            content = str((row or [""])[0] or "")
            positives, negs = self._extract_slot_signals(key, content)
            for n in negs:
                _add_negative(n)
            if not positives:
                continue
            recency = max(0.15, 1.0 - (idx / chat_total))
            score = 1.35 + recency * 0.8
            for p in positives:
                _add_candidate(p, score, "chat:user", content)

        if not candidates:
            profile = self.get_user_profile(uid) or {}
            fallback = self._normalize_slot_value(key, profile.get(key, ""))
            if fallback:
                _add_candidate(fallback, 1.0, "profile_fallback", "get_user_profile")

        if not candidates:
            return result

        ordered = sorted(candidates.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)
        best = ordered[0]
        best_score = float(best.get("score", 0.0))
        second_score = float(ordered[1].get("score", 0.0)) if len(ordered) > 1 else 0.0
        margin = max(0.0, best_score - second_score)
        conf = min(0.99, max(0.45, 0.42 + min(best_score, 6.0) * 0.085 + min(margin, 2.5) * 0.07))

        evidence: list[str] = []
        seen_ev: set[str] = set()
        for ev in best.get("evidence", []):
            low = str(ev).strip().lower()
            if not low or low in seen_ev:
                continue
            seen_ev.add(low)
            evidence.append(str(ev))
            if len(evidence) >= 4:
                break

        result.update({
            "value": str(best.get("value", "")).strip(),
            "confidence": conf,
            "source": str(best.get("source", "") or "inferred"),
            "evidence": evidence,
        })

        if auto_heal and result["value"]:
            should_upsert = False
            if not canonical:
                should_upsert = True
            elif cur_contradicted:
                should_upsert = True
            elif cur_value.lower() != str(result["value"]).lower():
                should_upsert = conf >= max(0.62, cur_conf + 0.03)
            if should_upsert:
                healed = self.upsert_canonical_slot(
                    uid,
                    key,
                    str(result["value"]),
                    confidence=max(0.72, conf),
                    source=f"resolver:{str(result['source'])[:24]}",
                )
                result["auto_healed"] = bool(healed)

        return result

    def is_slot_value_contradicted(self, user_id: str, slot_key: str, slot_value: str,
                                   lookback: int = 300) -> bool:
        """Check whether a canonical/profile slot value is contradicted by memory."""
        import re

        person_id = self.get_canonical_person_id(user_id)
        key = self._canonical_slot_key(slot_key)
        value = " ".join(str(slot_value or "").strip().lower().split())
        if key not in _CANONICAL_PROFILE_SLOTS or not value:
            return False

        rows = self.db.execute(
            """SELECT content
               FROM memories
               WHERE user_id = ? AND archived_at IS NULL
                 AND type IN ('fact', 'correction')
               ORDER BY id DESC LIMIT ?""",
            (person_id, max(10, int(lookback))),
        ).fetchall()

        value_re = re.escape(value)
        value_us = re.escape(value.replace(" ", "_"))
        for row in rows:
            s = " ".join(str((row or [""])[0] or "").lower().split())
            if not s:
                continue
            if key == "name":
                # Machine-readable corrections emitted by extractor.
                if re.search(rf"\bname_is_not_{value_us}\b", s):
                    return True
                # Natural language negations.
                if re.search(rf"\bname[_\s]*is[_\s]*not[_\s]+{value_re}\b", s):
                    return True
                if re.search(rf"\b(мо[её]\s+)?имя\s+не\s+{value_re}\b", s):
                    return True
                if re.search(rf"\b(меня|тебя)\s+зовут\s+не\s+{value_re}\b", s):
                    return True
        return False

    def upsert_canonical_slot(self, user_id: str, slot_key: str, slot_value: str,
                              *, confidence: float = 0.7, source: str = "extractor",
                              force: bool = False) -> dict[str, Any] | None:
        """Store canonical profile slot with confidence and immutable history."""
        person_id = self.get_canonical_person_id(user_id)
        key = self._canonical_slot_key(slot_key)
        value = str(slot_value or "").strip()
        if key not in _CANONICAL_PROFILE_SLOTS or not value:
            return None

        conf = max(0.0, min(float(confidence), 1.0))
        now = datetime.now().isoformat()
        cur = self.db.execute(
            """SELECT slot_value, confidence, version
               FROM canonical_profile_slots
               WHERE person_id = ? AND slot_key = ?""",
            (person_id, key),
        ).fetchone()
        if cur:
            cur_value = str(cur[0] or "").strip()
            cur_conf = float(cur[1] or 0.0)
            cur_version = int(cur[2] or 1)
            cur_contradicted = self.is_slot_value_contradicted(person_id, key, cur_value) if cur_value else False
            # Keep stronger value unless new evidence is significantly better or identical.
            if value.lower() != cur_value.lower() and (not force) and (not cur_contradicted) and conf < (cur_conf + 0.12):
                return {
                    "person_id": person_id,
                    "slot_key": key,
                    "slot_value": cur_value,
                    "confidence": cur_conf,
                    "version": cur_version,
                    "source": "kept_existing",
                    "updated_at": now,
                }
            if value.lower() != cur_value.lower() and cur_contradicted:
                # Prefer replacing stale contradicted values even with moderate confidence.
                conf = max(conf, min(0.78, cur_conf * 0.82))
            if value.lower() != cur_value.lower() and force:
                # Explicit user instruction should override previous slot value.
                conf = max(conf, min(0.99, cur_conf + 0.04))
            new_version = cur_version + (1 if value.lower() != cur_value.lower() else 0)
            new_conf = max(cur_conf * 0.85, conf) if value.lower() != cur_value.lower() else max(cur_conf, conf)
            self.db.execute(
                """UPDATE canonical_profile_slots
                   SET slot_value = ?, confidence = ?, version = ?, source = ?, updated_at = ?
                   WHERE person_id = ? AND slot_key = ?""",
                (value[:120], new_conf, new_version, source[:40], now, person_id, key),
            )
            self.db.execute(
                """INSERT INTO canonical_profile_slot_history
                   (person_id, slot_key, slot_value, confidence, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (person_id, key, value[:120], new_conf, source[:40], now),
            )
        else:
            self.db.execute(
                """INSERT INTO canonical_profile_slots
                   (person_id, slot_key, slot_value, confidence, version, source, updated_at)
                   VALUES (?, ?, ?, ?, 1, ?, ?)""",
                (person_id, key, value[:120], conf, source[:40], now),
            )
            self.db.execute(
                """INSERT INTO canonical_profile_slot_history
                   (person_id, slot_key, slot_value, confidence, source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (person_id, key, value[:120], conf, source[:40], now),
            )
        self.db.commit()
        return self.get_canonical_slot(person_id, key)

    def get_user_profile(self, user_id: str) -> dict[str, str]:
        """Get pinned user profile facts extracted from conversation."""
        person_id = self.get_canonical_person_id(user_id)
        profile = self.get_state("user:profile_facts", user_id=person_id)
        data = dict(profile) if isinstance(profile, dict) else {}
        canonical = self.get_canonical_profile(person_id)
        for slot, meta in canonical.items():
            val = str(meta.get("value") or "").strip()
            if val and not self.is_slot_value_contradicted(person_id, slot, val):
                data[slot] = val
        # Also hide contradicted slot values from legacy pinned profile state.
        for slot in _CANONICAL_PROFILE_SLOTS:
            val = str(data.get(slot) or "").strip()
            if val and self.is_slot_value_contradicted(person_id, slot, val):
                data.pop(slot, None)
        return data

    def _set_user_profile(self, user_id: str, profile: dict[str, str]) -> None:
        """Persist pinned profile facts with conservative field limits."""
        person_id = self.get_canonical_person_id(user_id)
        clean = {}
        for k, v in profile.items():
            key = str(k).strip().lower()
            val = str(v).strip()
            if not key or not val:
                continue
            clean[key] = val[:120]
            if key in _CANONICAL_PROFILE_SLOTS:
                self.upsert_canonical_slot(person_id, key, val, confidence=0.72, source="profile")
        self.set_state("user:profile_facts", clean, user_id=person_id)

    @staticmethod
    def _is_self_referential_memory_limit(text: str) -> bool:
        """Detect low-signal meta statements about assistant memory limits."""
        s = " ".join(str(text or "").lower().split())
        if not s:
            return False

        direct_patterns = (
            "i can't remember previous conversations",
            "i cannot remember previous conversations",
            "i don't remember previous conversations",
            "i do not remember previous conversations",
            "i only remember this chat",
            "i only have access to this conversation",
            "as an ai, i don't have memory",
            "у меня нет доступа к прошлым сообщениям",
            "я не помню прошлые разговоры",
            "я не могу помнить прошлые разговоры",
            "я помню только этот чат",
            "я помню только текущую сессию",
        )
        if any(p in s for p in direct_patterns):
            return True

        memory_terms = ("remember", "recall", "memory", "помн", "памят")
        if not any(t in s for t in memory_terms):
            return False

        first_person = ("i ", "i'm ", "i am ", "я ", "у меня ", "мне ")
        limiting_terms = (
            "can't", "cannot", "don't", "do not", "нет", "не могу",
            "не имею", "only this chat", "this session",
            "только этот чат", "только в этом чате", "текущую сессию",
            "previous conversation", "прошлые разговоры",
            "past conversation", "прошлых сообщений",
        )
        return any(p in s for p in first_person) and any(t in s for t in limiting_terms)

    @staticmethod
    def _is_memory_pollution_text(text: str) -> bool:
        """Detect low-signal assistant-disclaimer text that should not become memory."""
        s = " ".join(str(text or "").lower().split())
        if not s:
            return True
        denylist = (
            "i can't remember",
            "i cannot remember",
            "i don't remember",
            "i do not remember",
            "i only remember this chat",
            "i only have access to this conversation",
            "i only have access to the current session",
            "i only work within this session",
            "no long-term memory",
            "no long term memory",
            "does not have long-term memory",
            "does not have long term memory",
            "does_not_have_long_term_memory",
            "system_does_not_store_data_between_sessions",
            "works only in current context",
            "without saving data between sessions",
            "as an ai",
            "i cannot browse",
            "i can't browse",
            "я не помню",
            "я не могу помнить",
            "я помню только этот чат",
            "у меня нет доступа к прошлым сообщениям",
            "нет долгосрочной памяти",
            "не имеет долгосрочной памяти",
            "между сессиями информация не сохраняется",
            "без сохранения данных между сессиями",
            "система работает только в текущем контексте",
            "не может запоминать или воспроизводить предыдущие сессии",
            "только в текущей сессии",
            "как ии",
            "я не могу проверить",
            "не могу выйти в интернет",
        )
        if any(p in s for p in denylist):
            return True
        # Catch paraphrases like "memory only in current session / no persistence between sessions".
        memory_terms = ("memory", "long-term", "remember", "recall", "памят", "долгосроч", "запомин")
        scope_terms = ("session", "between sessions", "current context", "текущ", "сесси", "между сесс")
        neg_terms = ("no ", "not ", "cannot", "can't", "без ", "нет ", "не ", "только ")
        if (
            any(t in s for t in memory_terms)
            and any(t in s for t in scope_terms)
            and any(t in s for t in neg_terms)
        ):
            return True
        # Excessively generic metadata sentences add noise and pollute recall.
        generic_prefixes = (
            "assistant says",
            "assistant response",
            "the assistant said",
            "ответ ассистента",
            "ассистент сказал",
            "assistant:",
            "ассистент:",
            "ai:",
            "model:",
        )
        return any(s.startswith(p) for p in generic_prefixes)

    @staticmethod
    def _is_assistant_meta_statement(text: str) -> bool:
        """Detect assistant/system self-referential statements (low-value user memory)."""
        s = " ".join(str(text or "").lower().split())
        if not s:
            return False
        role_terms = ("assistant", "ассистент", "ai", "model", "модель", "chatgpt", "llm")
        meta_terms = (
            "said", "says", "answer", "responded", "remember", "memory",
            "сказал", "ответил", "помнит", "памят", "не помнит", "не может",
        )
        if any(s.startswith(p) for p in ("assistant:", "ассистент:", "model:", "ai:")):
            return True
        return any(t in s for t in role_terms) and any(t in s for t in meta_terms)

    @staticmethod
    def _is_operational_memory_noise(text: str) -> bool:
        """Detect ephemeral execution/file-system facts that should not become user memory."""
        s = " ".join(str(text or "").lower().split())
        if not s:
            return True
        command_markers = (
            "`", "curl ", "docker ", "docker-compose", "npm ", "pip ", "pytest ",
            "http/1.1", "http 200", "http 404", "nginx", "uvicorn", "sqlite",
            "ls -", "cd ", "mkdir ", "cp ", "mv ", "rm ", "cat ", "echo ",
        )
        fs_markers = (
            "folder ", "directory ", "file ", "path ", "project tree", "storage",
            "not found", "missing", "does not exist", "exists in", "created", "executed",
            "reload", "build has been", "command execution",
            "папк", "директор", "файл", "путь", "не найден", "отсутствует",
            "не существует", "создан", "выполнен", "перезагруз", "сборк",
        )
        return any(marker in s for marker in command_markers) or any(marker in s for marker in fs_markers)

    @staticmethod
    def _primary_script_family(text: str) -> str:
        """Return the dominant writing system for coarse language-compat checks."""
        counts = {"latin": 0, "cyrillic": 0, "hangul": 0, "cjk": 0, "arabic": 0}
        for ch in str(text or ""):
            code = ord(ch)
            if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
                counts["latin"] += 1
            elif 0x0400 <= code <= 0x04FF:
                counts["cyrillic"] += 1
            elif 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
                counts["hangul"] += 1
            elif 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
                counts["cjk"] += 1
            elif 0x0600 <= code <= 0x06FF:
                counts["arabic"] += 1
        family, count = max(counts.items(), key=lambda item: item[1])
        return family if count > 0 else "other"

    @classmethod
    def _is_cross_script_noise(cls, query: str, content: str) -> bool:
        """Reject strongly incompatible script matches unless lexical overlap exists."""
        q_script = cls._primary_script_family(query)
        c_script = cls._primary_script_family(content)
        if q_script == "other" or c_script == "other" or q_script == c_script:
            return False
        if c_script == "latin":
            return False
        overlap = cls._query_overlap(query, content)
        if q_script in {"cyrillic", "latin"} and c_script in {"hangul", "cjk", "arabic"}:
            return overlap < 0.04
        if q_script in {"hangul", "cjk", "arabic"} and c_script in {"cyrillic"}:
            return overlap < 0.04
        return False

    @staticmethod
    def _coerce_string_list(value: Any) -> list[str]:
        """Coerce mixed LLM JSON shapes into a clean list of non-empty strings."""
        if value is None:
            return []
        if isinstance(value, str):
            s = value.strip()
            return [s] if s else []
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                candidate = ""
                for key in ("text", "content", "value", "fact", "preference", "correction", "summary"):
                    raw = item.get(key)
                    if isinstance(raw, str) and raw.strip():
                        candidate = raw.strip()
                        break
                if not candidate:
                    continue
                cleaned.append(candidate)
                continue
            s = str(item).strip()
            if s:
                cleaned.append(s)
        return cleaned

    @classmethod
    def _coerce_summary_text(cls, value: Any) -> str:
        """Normalize session_summary into a plain string."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("text", "content", "summary", "value"):
                raw = value.get(key)
                if isinstance(raw, str) and raw.strip():
                    return raw.strip()
        return ""

    def _thinking_cloud_enabled(self) -> bool:
        return bool(self.config.get("thinking_cloud_enabled", True))

    @staticmethod
    def _safe_unit_float(value: Any, default: float = 0.5) -> float:
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _normalize_thinking_text(text: str) -> str:
        import re

        s = " ".join(str(text or "").strip().split())
        if not s:
            return ""
        s = s.strip(" .,!?:;\"'`()[]{}")
        s = re.sub(r"\s+", " ", s.lower())
        s = re.sub(r"[\"'`]+", "", s)
        return s[:500]

    @staticmethod
    def _summarize_thinking_title(text: str, note_type: str) -> str:
        words = [w for w in str(text or "").strip().split() if w]
        if not words:
            return note_type.replace("_", " ").title()
        title = " ".join(words[:8]).strip(" .,!?:;\"'")
        if len(words) > 8:
            title += "..."
        return title

    @classmethod
    def _sanitize_theme_labels(cls, labels: Any) -> list[str]:
        out: list[str] = []
        if isinstance(labels, str):
            labels = [labels]
        if not isinstance(labels, list):
            return out
        for item in labels[:6]:
            label = " ".join(str(item or "").strip().split())
            if not label:
                continue
            label = label.strip(" .,!?:;\"'")
            if len(label) < 2:
                continue
            if len(label) > 60:
                label = label[:60].rstrip()
            if cls._is_memory_pollution_text(label) or cls._is_assistant_meta_statement(label):
                continue
            if label.lower() not in {x.lower() for x in out}:
                out.append(label)
        return out

    @classmethod
    def _coerce_thinking_items(cls, value: Any, default_type: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, str):
            s = value.strip()
            return [{"content": s, "note_type": default_type}] if s else []
        if not isinstance(value, list):
            return []
        items: list[dict[str, Any]] = []
        for raw in value:
            if raw is None:
                continue
            if isinstance(raw, dict):
                content = ""
                for key in ("content", "text", "value", "idea", "question", "constraint", "signal", "title"):
                    maybe = raw.get(key)
                    if isinstance(maybe, str) and maybe.strip():
                        content = " ".join(maybe.strip().split())
                        break
                if not content:
                    continue
                items.append({
                    "note_type": str(raw.get("note_type") or default_type or "").strip() or default_type,
                    "content": content,
                    "title": str(raw.get("title") or "").strip(),
                    "themes": cls._sanitize_theme_labels(raw.get("themes")),
                    "confidence": cls._safe_unit_float(raw.get("confidence"), 0.62),
                    "importance": cls._safe_unit_float(raw.get("importance"), 0.6),
                    "novelty": cls._safe_unit_float(raw.get("novelty"), 0.55),
                })
                continue
            content = " ".join(str(raw).strip().split())
            if content:
                items.append({
                    "note_type": default_type,
                    "content": content,
                    "title": "",
                    "themes": [],
                    "confidence": 0.62,
                    "importance": 0.6,
                    "novelty": 0.55,
                })
        return items

    @staticmethod
    def _thinking_score(confidence: float, novelty: float, recurrence: float,
                        strategic_importance: float) -> float:
        rec_norm = max(0.0, min(float(recurrence or 0.0) / 5.0, 1.0))
        score = (
            float(strategic_importance or 0.0) * 0.45
            + float(confidence or 0.0) * 0.25
            + float(novelty or 0.0) * 0.10
            + rec_norm * 0.20
        )
        return round(max(0.0, min(score, 1.0)), 4)

    def _upsert_thinking_edge(self, user_id: str, source_note_id: int,
                              target_note_id: int, relation_type: str,
                              weight: float = 0.6) -> None:
        uid = self.get_canonical_person_id(user_id)
        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO thinking_edges
               (user_id, source_note_id, target_note_id, relation_type, weight, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(user_id, source_note_id, target_note_id, relation_type)
               DO UPDATE SET
                   weight = MAX(thinking_edges.weight, excluded.weight),
                   updated_at = excluded.updated_at""",
            (uid, int(source_note_id), int(target_note_id), str(relation_type or "related_to"),
             self._safe_unit_float(weight, 0.6), now, now),
        )

    def upsert_thinking_note(self, user_id: str, note_type: str, content: str,
                             *, title: str = "", confidence: float = 0.62,
                             novelty: float = 0.55, strategic_importance: float = 0.6,
                             themes: list[str] | None = None,
                             meta: dict[str, Any] | None = None) -> int | None:
        if not self._thinking_cloud_enabled():
            return None
        uid = self.get_canonical_person_id(user_id)
        ntype = str(note_type or "idea").strip().lower() or "idea"
        text = " ".join(str(content or "").strip().split())
        normalized = self._normalize_thinking_text(text)
        if not text or not normalized:
            return None
        if self._is_memory_pollution_text(text) or self._is_assistant_meta_statement(text):
            return None
        if self._is_operational_memory_noise(text) and ntype != "theme":
            return None

        confidence = self._safe_unit_float(confidence, 0.62)
        novelty = self._safe_unit_float(novelty, 0.55)
        strategic_importance = self._safe_unit_float(strategic_importance, 0.6)
        cleaned_themes = self._sanitize_theme_labels(themes or [])
        meta_json = json.dumps(meta or {}, ensure_ascii=False) if meta else None
        now = datetime.now().isoformat()

        row = self.db.execute(
            """SELECT id, recurrence, confidence, novelty, strategic_importance
               FROM thinking_notes
               WHERE user_id = ? AND note_type = ? AND normalized_content = ?""",
            (uid, ntype, normalized),
        ).fetchone()
        if row:
            note_id = int(row[0])
            recurrence = float(row[1] or 1.0) + 1.0
            merged_conf = max(float(row[2] or 0.0), confidence)
            merged_novelty = max(float(row[3] or 0.0), novelty)
            merged_importance = max(float(row[4] or 0.0), strategic_importance)
            score = self._thinking_score(merged_conf, merged_novelty, recurrence, merged_importance)
            self.db.execute(
                """UPDATE thinking_notes
                   SET title = ?, content = ?, confidence = ?, novelty = ?,
                       recurrence = ?, strategic_importance = ?, score = ?,
                       meta_json = COALESCE(?, meta_json),
                       last_seen_at = ?, updated_at = ?, status = 'active'
                   WHERE id = ?""",
                (
                    title.strip() or self._summarize_thinking_title(text, ntype),
                    text,
                    merged_conf,
                    merged_novelty,
                    recurrence,
                    merged_importance,
                    score,
                    meta_json,
                    now,
                    now,
                    note_id,
                ),
            )
        else:
            recurrence = 1.0
            score = self._thinking_score(confidence, novelty, recurrence, strategic_importance)
            cur = self.db.execute(
                """INSERT INTO thinking_notes
                   (user_id, note_type, title, content, normalized_content, confidence, novelty,
                    recurrence, strategic_importance, score, status, meta_json,
                    first_seen_at, last_seen_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)""",
                (
                    uid,
                    ntype,
                    title.strip() or self._summarize_thinking_title(text, ntype),
                    text,
                    normalized,
                    confidence,
                    novelty,
                    recurrence,
                    strategic_importance,
                    score,
                    meta_json,
                    now,
                    now,
                    now,
                    now,
                ),
            )
            note_id = int(cur.lastrowid)

        if cleaned_themes and ntype != "theme":
            for theme in cleaned_themes:
                theme_id = self.upsert_thinking_note(
                    uid,
                    "theme",
                    theme,
                    title=theme,
                    confidence=max(confidence, 0.72),
                    novelty=max(novelty * 0.85, 0.35),
                    strategic_importance=max(strategic_importance * 0.9, 0.4),
                    themes=[],
                    meta={"source": "thinking-theme"},
                )
                if theme_id:
                    self._upsert_thinking_edge(uid, note_id, theme_id, "about", weight=max(strategic_importance, 0.5))

        self.db.commit()
        return note_id

    def store_thinking_cloud_items(self, user_id: str, buckets: dict[str, Any]) -> dict[str, int]:
        if not self._thinking_cloud_enabled():
            return {"saved": 0}
        uid = self.get_canonical_person_id(user_id)
        mapping = {
            "ideas": "idea",
            "constraints": "constraint",
            "open_questions": "open_question",
            "decision_signals": "decision_signal",
            "directions": "direction",
        }
        saved = 0
        per_type: dict[str, int] = {}
        for key, note_type in mapping.items():
            items = self._coerce_thinking_items(buckets.get(key), note_type)
            for item in items:
                content = str(item.get("content", "") or "").strip()
                if len(content) < 12:
                    continue
                quality = (
                    self._safe_unit_float(item.get("importance"), 0.6) * 0.55
                    + self._safe_unit_float(item.get("confidence"), 0.62) * 0.35
                    + self._safe_unit_float(item.get("novelty"), 0.55) * 0.10
                )
                if quality < 0.48:
                    continue
                note_id = self.upsert_thinking_note(
                    uid,
                    note_type,
                    content,
                    title=str(item.get("title", "") or "").strip(),
                    confidence=self._safe_unit_float(item.get("confidence"), 0.62),
                    novelty=self._safe_unit_float(item.get("novelty"), 0.55),
                    strategic_importance=self._safe_unit_float(item.get("importance"), 0.6),
                    themes=item.get("themes") or [],
                    meta={"source": "thinking-cloud", "bucket": key},
                )
                if note_id:
                    saved += 1
                    per_type[note_type] = per_type.get(note_type, 0) + 1
        return {"saved": saved, **per_type}

    def _get_thinking_note_themes(self, note_ids: list[int]) -> dict[int, list[str]]:
        if not note_ids:
            return {}
        placeholders = ",".join("?" for _ in note_ids)
        rows = self.db.execute(
            f"""SELECT e.source_note_id, t.content
                FROM thinking_edges e
                JOIN thinking_notes t ON t.id = e.target_note_id
                WHERE e.source_note_id IN ({placeholders})
                  AND e.relation_type = 'about'
                  AND t.status = 'active'
                ORDER BY t.score DESC, t.last_seen_at DESC""",
            tuple(note_ids),
        ).fetchall()
        out: dict[int, list[str]] = {}
        for source_id, theme in rows:
            bucket = out.setdefault(int(source_id), [])
            if theme and theme not in bucket:
                bucket.append(str(theme))
        return out

    def recall_thinking_cloud(self, query: str, user_id: str, top_k: int = 5) -> list[dict]:
        if not self._thinking_cloud_enabled():
            return []
        uid = self.get_canonical_person_id(user_id)
        rows = self.db.execute(
            """SELECT id, note_type, title, content, confidence, novelty, recurrence,
                      strategic_importance, score, last_seen_at, created_at
               FROM thinking_notes
               WHERE user_id = ? AND status = 'active'
               ORDER BY score DESC, last_seen_at DESC
               LIMIT 120""",
            (uid,),
        ).fetchall()
        if not rows:
            return []

        q = str(query or "").strip()
        results: list[dict[str, Any]] = []
        for row in rows:
            note_id = int(row[0])
            note_type = str(row[1] or "")
            content = str(row[3] or "")
            overlap = self._query_overlap(q, content) if q else 0.0
            if q and overlap <= 0 and note_type != "theme":
                continue
            recency = self._recency_score(str(row[9] or row[10] or ""))
            base = float(row[8] or 0.0)
            type_bonus = {
                "direction": 0.08,
                "constraint": 0.07,
                "decision_signal": 0.06,
                "idea": 0.05,
                "open_question": 0.04,
                "theme": -0.05,
            }.get(note_type, 0.0)
            final = base * 0.58 + overlap * 0.26 + recency * 0.12 + type_bonus
            results.append({
                "id": note_id,
                "type": note_type,
                "title": str(row[2] or ""),
                "content": content,
                "score": round(final, 4),
                "base_score": round(base, 4),
                "overlap": round(overlap, 4),
                "last_seen_at": str(row[9] or row[10] or ""),
            })
        results.sort(key=lambda item: item["score"], reverse=True)
        out = results[:max(1, top_k)]
        themes_by_id = self._get_thinking_note_themes([item["id"] for item in out])
        for item in out:
            item["themes"] = themes_by_id.get(item["id"], [])
        return out

    def get_thinking_cloud_summary(self, user_id: str, limit: int = 8) -> dict[str, Any]:
        if not self._thinking_cloud_enabled():
            return {
                "enabled": False,
                "overview": {"total_notes": 0, "themes": 0, "open_questions": 0, "active_directions": 0},
                "themes": [],
                "directions": [],
                "constraints": [],
                "open_questions": [],
                "recent": [],
                "updated_at": "",
            }
        uid = self.get_canonical_person_id(user_id)
        total_notes = int(self.db.execute(
            "SELECT COUNT(*) FROM thinking_notes WHERE user_id = ? AND status = 'active'",
            (uid,),
        ).fetchone()[0] or 0)
        theme_total = int(self.db.execute(
            "SELECT COUNT(*) FROM thinking_notes WHERE user_id = ? AND status = 'active' AND note_type = 'theme'",
            (uid,),
        ).fetchone()[0] or 0)
        open_total = int(self.db.execute(
            "SELECT COUNT(*) FROM thinking_notes WHERE user_id = ? AND status = 'active' AND note_type = 'open_question'",
            (uid,),
        ).fetchone()[0] or 0)
        direction_total = int(self.db.execute(
            """SELECT COUNT(*) FROM thinking_notes
               WHERE user_id = ? AND status = 'active'
                 AND note_type IN ('direction', 'idea', 'decision_signal')""",
            (uid,),
        ).fetchone()[0] or 0)

        top_theme_rows = self.db.execute(
            """SELECT t.id, t.content, t.score, COUNT(e.id) AS linked
               FROM thinking_notes t
               LEFT JOIN thinking_edges e
                 ON e.target_note_id = t.id AND e.relation_type = 'about' AND e.user_id = t.user_id
               WHERE t.user_id = ? AND t.status = 'active' AND t.note_type = 'theme'
               GROUP BY t.id, t.content, t.score
               ORDER BY linked DESC, t.score DESC, t.last_seen_at DESC
               LIMIT ?""",
            (uid, max(4, limit)),
        ).fetchall()
        directions_rows = self.db.execute(
            """SELECT id, note_type, title, content, score, recurrence, last_seen_at
               FROM thinking_notes
               WHERE user_id = ? AND status = 'active'
                 AND note_type IN ('direction', 'idea', 'decision_signal')
               ORDER BY score DESC, last_seen_at DESC
               LIMIT ?""",
            (uid, limit),
        ).fetchall()
        constraints_rows = self.db.execute(
            """SELECT id, title, content, score, recurrence, last_seen_at
               FROM thinking_notes
               WHERE user_id = ? AND status = 'active' AND note_type = 'constraint'
               ORDER BY score DESC, last_seen_at DESC
               LIMIT ?""",
            (uid, max(3, limit // 2)),
        ).fetchall()
        questions_rows = self.db.execute(
            """SELECT id, title, content, score, recurrence, last_seen_at
               FROM thinking_notes
               WHERE user_id = ? AND status = 'active' AND note_type = 'open_question'
               ORDER BY score DESC, last_seen_at DESC
               LIMIT ?""",
            (uid, max(3, limit // 2)),
        ).fetchall()
        recent_rows = self.db.execute(
            """SELECT id, note_type, title, content, score, last_seen_at
               FROM thinking_notes
               WHERE user_id = ? AND status = 'active'
               ORDER BY last_seen_at DESC, score DESC
               LIMIT ?""",
            (uid, max(4, limit)),
        ).fetchall()

        recent_ids = [int(r[0]) for r in recent_rows]
        direction_ids = [int(r[0]) for r in directions_rows]
        constraint_ids = [int(r[0]) for r in constraints_rows]
        question_ids = [int(r[0]) for r in questions_rows]
        theme_map = self._get_thinking_note_themes(recent_ids + direction_ids + constraint_ids + question_ids)

        def _fmt_note(row, *, note_type_idx=1, title_idx=2, content_idx=3,
                      score_idx=4, recurrence_idx=5, last_idx=6):
            note_id = int(row[0])
            return {
                "id": note_id,
                "type": str(row[note_type_idx] or ""),
                "title": str(row[title_idx] or ""),
                "content": str(row[content_idx] or ""),
                "score": round(float(row[score_idx] or 0.0), 4),
                "recurrence": round(float(row[recurrence_idx] or 0.0), 2),
                "last_seen_at": str(row[last_idx] or ""),
                "themes": theme_map.get(note_id, []),
            }

        updated_at = ""
        if recent_rows:
            updated_at = str(recent_rows[0][5] or "")
        return {
            "enabled": True,
            "overview": {
                "total_notes": total_notes,
                "themes": theme_total,
                "open_questions": open_total,
                "active_directions": direction_total,
            },
            "themes": [
                {
                    "id": int(r[0]),
                    "label": str(r[1] or ""),
                    "score": round(float(r[2] or 0.0), 4),
                    "linked_notes": int(r[3] or 0),
                }
                for r in top_theme_rows
            ],
            "directions": [_fmt_note(r) for r in directions_rows],
            "constraints": [
                {
                    "id": int(r[0]),
                    "type": "constraint",
                    "title": str(r[1] or ""),
                    "content": str(r[2] or ""),
                    "score": round(float(r[3] or 0.0), 4),
                    "recurrence": round(float(r[4] or 0.0), 2),
                    "last_seen_at": str(r[5] or ""),
                    "themes": theme_map.get(int(r[0]), []),
                }
                for r in constraints_rows
            ],
            "open_questions": [
                {
                    "id": int(r[0]),
                    "type": "open_question",
                    "title": str(r[1] or ""),
                    "content": str(r[2] or ""),
                    "score": round(float(r[3] or 0.0), 4),
                    "recurrence": round(float(r[4] or 0.0), 2),
                    "last_seen_at": str(r[5] or ""),
                    "themes": theme_map.get(int(r[0]), []),
                }
                for r in questions_rows
            ],
            "recent": [
                {
                    "id": int(r[0]),
                    "type": str(r[1] or ""),
                    "title": str(r[2] or ""),
                    "content": str(r[3] or ""),
                    "score": round(float(r[4] or 0.0), 4),
                    "last_seen_at": str(r[5] or ""),
                    "themes": theme_map.get(int(r[0]), []),
                }
                for r in recent_rows
            ],
            "updated_at": updated_at,
        }

    def get_thinking_cloud_context(self, user_id: str, query: str = "", top_k: int = 5) -> str:
        if not self._thinking_cloud_enabled():
            return ""
        summary = self.get_thinking_cloud_summary(user_id, limit=max(4, top_k))
        recalled = self.recall_thinking_cloud(query, user_id, top_k=top_k)
        if not summary.get("overview", {}).get("total_notes") and not recalled:
            return ""

        lines = ["\n\n## User thinking cloud:"]
        top_themes = [item.get("label", "") for item in summary.get("themes", [])[:4] if item.get("label")]
        if top_themes:
            lines.append("- Dominant themes: " + ", ".join(top_themes))

        directions = [item.get("content", "") for item in summary.get("directions", [])[:3] if item.get("content")]
        if directions:
            lines.append("- Active directions: " + "; ".join(directions))

        constraints = [item.get("content", "") for item in summary.get("constraints", [])[:2] if item.get("content")]
        if constraints:
            lines.append("- Stable constraints: " + "; ".join(constraints))

        questions = [item.get("content", "") for item in summary.get("open_questions", [])[:2] if item.get("content")]
        if questions:
            lines.append("- Open questions: " + "; ".join(questions))

        if recalled:
            focus_lines = []
            for item in recalled[:3]:
                label = item.get("type", "note").replace("_", " ")
                content = item.get("content", "")
                if content:
                    focus_lines.append(f"{label}: {content}")
            if focus_lines:
                lines.append("- Most relevant to this request: " + " | ".join(focus_lines))

        return "\n".join(lines)

    @staticmethod
    def _thinking_obsidian_folder(note_type: str) -> str:
        mapping = {
            "theme": "Themes",
            "direction": "Directions",
            "constraint": "Constraints",
            "open_question": "Open Questions",
            "idea": "Ideas",
            "decision_signal": "Decision Signals",
        }
        return mapping.get(str(note_type or "").strip().lower(), "Notes")

    @staticmethod
    def _thinking_obsidian_slug(text: str, note_id: int) -> str:
        import re as _re

        base = _re.sub(r"[^a-z0-9]+", "-", str(text or "").strip().lower()).strip("-")
        if not base:
            base = f"note-{int(note_id)}"
        if len(base) > 56:
            base = base[:56].rstrip("-")
        return f"{base}-{int(note_id)}"

    @staticmethod
    def _thinking_yaml_scalar(value: Any) -> str:
        text = str(value or "")
        if not text:
            return '""'
        if "\n" in text:
            text = text.replace("\n", " ")
        if any(ch in text for ch in [":", "[", "]", "{", "}", "#", '"', "'"]):
            escaped = text.replace('"', '\\"')
            return f'"{escaped}"'
        return text

    @classmethod
    def _thinking_frontmatter(cls, data: dict[str, Any]) -> str:
        lines: list[str] = ["---"]
        for key, value in data.items():
            if value is None or value == "":
                continue
            if isinstance(value, list):
                if not value:
                    continue
                lines.append(f"{key}:")
                for item in value:
                    lines.append(f"  - {cls._thinking_yaml_scalar(item)}")
                continue
            if isinstance(value, float):
                lines.append(f"{key}: {value:.4f}".rstrip("0").rstrip("."))
                continue
            lines.append(f"{key}: {cls._thinking_yaml_scalar(value)}")
        lines.append("---")
        return "\n".join(lines)

    def export_thinking_cloud_obsidian(self, user_id: str, limit: int = 200) -> dict[str, Any]:
        """Build an Obsidian-compatible vault export for Thinking Cloud."""
        uid = self.get_canonical_person_id(user_id)
        lim = max(20, min(int(limit or 200), 500))
        summary = self.get_thinking_cloud_summary(uid, limit=12)

        rows = self.db.execute(
            """SELECT id, note_type, title, content, score, recurrence, confidence,
                      novelty, strategic_importance, first_seen_at, last_seen_at
               FROM thinking_notes
               WHERE user_id = ? AND status = 'active'
               ORDER BY score DESC, last_seen_at DESC
               LIMIT ?""",
            (uid, lim),
        ).fetchall()

        ids = [int(r[0]) for r in rows]
        theme_map = self._get_thinking_note_themes(ids)
        edge_map: dict[int, list[tuple[int, str]]] = {}
        if ids:
            placeholders = ",".join("?" for _ in ids)
            edge_rows = self.db.execute(
                f"""SELECT source_note_id, target_note_id, relation_type
                    FROM thinking_edges
                    WHERE user_id = ? AND source_note_id IN ({placeholders})""",
                [uid, *ids],
            ).fetchall()
            for source_id, target_id, relation_type in edge_rows:
                edge_map.setdefault(int(source_id), []).append((int(target_id), str(relation_type or "")))

        files: dict[str, str] = {}
        lookup: dict[int, dict[str, Any]] = {}
        theme_titles: dict[str, str] = {}

        for row in rows:
            note_id = int(row[0])
            note_type = str(row[1] or "")
            title = str(row[2] or "").strip() or self._summarize_thinking_title(str(row[3] or ""), note_type)
            content = str(row[3] or "").strip()
            folder = self._thinking_obsidian_folder(note_type)
            rel_path = f"{folder}/{self._thinking_obsidian_slug(title, note_id)}.md"
            lookup[note_id] = {
                "id": note_id,
                "type": note_type,
                "title": title,
                "content": content,
                "path": rel_path,
                "score": float(row[4] or 0.0),
                "recurrence": float(row[5] or 0.0),
                "confidence": float(row[6] or 0.0),
                "novelty": float(row[7] or 0.0),
                "importance": float(row[8] or 0.0),
                "first_seen_at": str(row[9] or ""),
                "last_seen_at": str(row[10] or ""),
                "themes": list(theme_map.get(note_id, [])),
            }
            if note_type == "theme":
                theme_titles[title.lower()] = title

        for note in lookup.values():
            theme_links = []
            for theme in note["themes"]:
                theme_title = theme_titles.get(str(theme).lower(), str(theme))
                theme_links.append(f"[[{theme_title}]]")

            related_links = []
            for target_id, relation_type in edge_map.get(int(note["id"]), []):
                target = lookup.get(int(target_id))
                if not target:
                    continue
                related_links.append(f"- `{relation_type or 'related_to'}` [[{target['title']}]]")

            frontmatter = self._thinking_frontmatter({
                "title": note["title"],
                "type": note["type"],
                "source": "liteagent-thinking-cloud",
                "user_id": uid,
                "score": note["score"],
                "recurrence": note["recurrence"],
                "confidence": note["confidence"],
                "novelty": note["novelty"],
                "strategic_importance": note["importance"],
                "first_seen_at": note["first_seen_at"],
                "last_seen_at": note["last_seen_at"],
                "themes": note["themes"],
                "tags": [
                    "liteagent",
                    "thinking-cloud",
                    f"thinking/{note['type']}",
                ],
            })

            body = [
                frontmatter,
                "",
                f"# {note['title']}",
                "",
                note["content"],
                "",
            ]
            if theme_links:
                body.extend([
                    "## Themes",
                    "",
                    "- " + "\n- ".join(theme_links),
                    "",
                ])
            if related_links:
                body.extend([
                    "## Related Notes",
                    "",
                    *related_links,
                    "",
                ])
            files[note["path"]] = "\n".join(body).strip() + "\n"

        theme_section = [f"- [[{item.get('label')}]]" for item in summary.get("themes", []) if item.get("label")]
        direction_section = [f"- [[{item.get('title')}]]" for item in summary.get("directions", []) if item.get("title")]
        constraint_section = [f"- [[{item.get('title')}]]" for item in summary.get("constraints", []) if item.get("title")]
        question_section = [f"- [[{item.get('title')}]]" for item in summary.get("open_questions", []) if item.get("title")]

        index_frontmatter = self._thinking_frontmatter({
            "title": "Thinking Cloud",
            "type": "thinking_cloud_index",
            "source": "liteagent-thinking-cloud",
            "user_id": uid,
            "exported_at": datetime.now().isoformat(),
            "tags": ["liteagent", "thinking-cloud", "dashboard-export"],
        })
        index_lines = [
            index_frontmatter,
            "",
            "# Thinking Cloud",
            "",
            "This vault is an Obsidian-compatible export of LiteAgent's strategic memory layer.",
            "",
            "## Overview",
            "",
            f"- Strategic notes: {int(summary.get('overview', {}).get('total_notes', 0) or 0)}",
            f"- Themes: {int(summary.get('overview', {}).get('themes', 0) or 0)}",
            f"- Active directions: {int(summary.get('overview', {}).get('active_directions', 0) or 0)}",
            f"- Open questions: {int(summary.get('overview', {}).get('open_questions', 0) or 0)}",
            "",
        ]
        if theme_section:
            index_lines.extend(["## Dominant Themes", "", *theme_section, ""])
        if direction_section:
            index_lines.extend(["## Active Directions", "", *direction_section, ""])
        if constraint_section:
            index_lines.extend(["## Stable Constraints", "", *constraint_section, ""])
        if question_section:
            index_lines.extend(["## Open Questions", "", *question_section, ""])
        index_lines.extend([
            "## Structure",
            "",
            "- `Themes/`",
            "- `Directions/`",
            "- `Constraints/`",
            "- `Open Questions/`",
            "- `Ideas/`",
            "- `Decision Signals/`",
            "",
        ])
        files["Thinking Cloud.md"] = "\n".join(index_lines).strip() + "\n"

        canvas_nodes: list[dict[str, Any]] = []
        canvas_edges: list[dict[str, Any]] = []
        column_x = {
            "theme": 0,
            "direction": 380,
            "constraint": 760,
            "open_question": 1140,
            "idea": 1520,
            "decision_signal": 1900,
        }
        row_y: dict[str, int] = {}
        for note_id, note in lookup.items():
            ntype = str(note["type"] or "")
            row = row_y.get(ntype, 0)
            row_y[ntype] = row + 1
            canvas_nodes.append({
                "id": f"note-{note_id}",
                "type": "file",
                "file": note["path"],
                "x": column_x.get(ntype, 2280),
                "y": row * 220,
                "width": 320,
                "height": 140,
            })
        for source_id, rels in edge_map.items():
            for target_id, relation_type in rels:
                if source_id not in lookup or target_id not in lookup:
                    continue
                canvas_edges.append({
                    "id": f"edge-{source_id}-{target_id}-{relation_type or 'related'}",
                    "fromNode": f"note-{source_id}",
                    "toNode": f"note-{target_id}",
                    "label": relation_type or "related_to",
                })
        files["Thinking Cloud.canvas"] = json.dumps({
            "nodes": canvas_nodes,
            "edges": canvas_edges,
        }, ensure_ascii=False, indent=2)

        return {
            "vault_name": f"LiteAgent Thinking Cloud - {uid}",
            "files": files,
            "overview": summary.get("overview", {}),
            "user_id": uid,
        }
        if isinstance(value, list):
            parts = cls._coerce_string_list(value)
            return " ".join(parts).strip()
        return str(value or "").strip()

    @staticmethod
    def _coerce_object_list(value: Any) -> list[dict]:
        """Normalize entities/relations payload into a list of dicts."""
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, dict)]

    @staticmethod
    def _extract_profile_facts(text: str) -> dict[str, str]:
        """Rule-based extraction of stable personal facts from text."""
        import re

        s = (text or "").strip()
        if not s:
            return {}
        low = s.lower()
        out: dict[str, str] = {}

        # Name in RU/EN forms
        patterns = (
            (r"\bменя зовут\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-\s]{0,40})", "name"),
            (r"\bмо[её]\s+имя\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-\s]{0,40})", "name"),
            (r"\bпользовател[ья]\s+зовут\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-\s]{0,40})", "name"),
            (r"\bимя\s+пользователя\s*[:\-—]\s*([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-\s]{0,40})", "name"),
            (r"\bзови\s+меня\s+([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-\s]{0,40})", "name"),
            (r"^\s*([A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{1,40})\s+запиши(?:\s+в\s+свою\s+память)?\s*$", "name"),
            (r"\bmy name is\s+([A-Za-z][A-Za-z\-\s]{0,40})", "name"),
            (r"\bcall me\s+([A-Za-z][A-Za-z\-\s]{0,40})", "name"),
            (r"\buser(?:'s)?\s+name\s+is\s+([A-Za-z][A-Za-z\-\s]{0,40})", "name"),
            (r"\buser\s+name\s*[:\-]\s*([A-Za-z][A-Za-z\-\s]{0,40})", "name"),
            (r"\bi am\s+([A-Za-z][A-Za-z\-\s]{0,40})", "name"),
        )
        for pat, field in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                val = m.group(1).strip(" .,!?:;\"'")
                if len(val) >= 2:
                    out[field] = val

        # Role / occupation
        role_patterns = (
            r"\bя\s+(?:работаю|являюсь)\s+([A-Za-zА-Яа-яЁё0-9\-\s]{2,60})",
            r"\bi work as\s+([A-Za-z0-9\-\s]{2,60})",
            r"\bi am a[n]?\s+([A-Za-z0-9\-\s]{2,60})",
        )
        for pat in role_patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                out["role"] = m.group(1).strip(" .,!?:;\"'")
                break

        # Location
        loc_patterns = (
            r"\bя живу в\s+([A-Za-zА-Яа-яЁё0-9\-\s]{2,60})",
            r"\bi live in\s+([A-Za-z0-9\-\s]{2,60})",
        )
        for pat in loc_patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                out["location"] = m.group(1).strip(" .,!?:;\"'")
                break

        # Language preference
        if "на русском" in low or "русском языке" in low:
            out["language"] = "ru"
        elif "in english" in low or "на английском" in low:
            out["language"] = "en"

        return out

    def apply_explicit_profile_update(self, user_id: str, text: str) -> dict[str, dict[str, Any]]:
        """Immediately apply explicit user profile statements (name/language/role)."""
        import re

        person_id = self.get_canonical_person_id(user_id)
        s = " ".join(str(text or "").strip().split())
        if not s:
            return {}
        low = s.lower()
        if self._is_memory_pollution_text(s):
            return {}

        # Only treat clear user instructions/statements as immediate updates.
        explicit_markers = (
            "запомни", "запиши", "remember", "save this",
            "меня зовут", "моё имя", "мое имя", "зови меня",
            "my name is", "call me", "my role", "i work as", "i live in",
            "я работаю", "я живу в", "пиши на русском", "отвечай на русском",
        )
        if not any(m in low for m in explicit_markers):
            # Allow strict "<Name> запиши..." intent even if marker list misses.
            if not re.search(r"^\s*[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё\-]{1,40}\s+запиши", s, flags=re.IGNORECASE):
                return {}

        extracted = self._extract_profile_facts(s)
        # Extra robust name parsing for imperative forms.
        name_pos, _ = self._extract_slot_signals("name", s)
        if name_pos:
            extracted["name"] = name_pos[0]

        if not extracted:
            return {}

        profile = self.get_user_profile(person_id) or {}
        updates: dict[str, dict[str, Any]] = {}

        for slot_key, raw_value in extracted.items():
            key = self._canonical_slot_key(slot_key)
            val = self._normalize_slot_value(key, raw_value)
            if not val:
                continue
            prev = ""
            prev_slot = self.get_canonical_slot(person_id, key) if key in _CANONICAL_PROFILE_SLOTS else None
            if prev_slot and prev_slot.get("slot_value"):
                prev = self._normalize_slot_value(key, prev_slot.get("slot_value", ""))
            elif profile.get(key):
                prev = self._normalize_slot_value(key, profile.get(key, ""))

            if key in _CANONICAL_PROFILE_SLOTS:
                base_conf = 0.98 if key == "name" else 0.94
                self.upsert_canonical_slot(
                    person_id,
                    key,
                    val,
                    confidence=base_conf,
                    source="explicit-user",
                    force=True,
                )
            profile[key] = val
            updates[key] = {"previous": prev, "value": val}

        if updates:
            self._set_user_profile(person_id, profile)
        return updates

    def update_user_profile_from_texts(self, user_id: str, texts: list[str]) -> dict[str, str]:
        """Merge extracted profile facts from multiple texts into pinned profile."""
        person_id = self.get_canonical_person_id(user_id)
        profile = self.get_user_profile(person_id)
        changed = False
        for t in texts:
            if self._is_memory_pollution_text(t):
                continue
            extracted = self._extract_profile_facts(t)
            for k, v in extracted.items():
                if profile.get(k) != v:
                    profile[k] = v
                    changed = True
                if k in _CANONICAL_PROFILE_SLOTS:
                    self.upsert_canonical_slot(person_id, k, v, confidence=0.74, source="profile-extract")
        if changed:
            self._set_user_profile(person_id, profile)
        return profile

    def ensure_user_profile(self, user_id: str, history_limit: int = 250) -> dict[str, str]:
        """Backfill pinned profile from existing memories/chat when profile is empty."""
        person_id = self.get_canonical_person_id(user_id)
        existing = self.get_user_profile(person_id)
        if existing:
            return existing

        texts: list[str] = []
        try:
            mem_rows = self.db.execute(
                """SELECT content FROM memories
                   WHERE user_id = ? AND archived_at IS NULL
                   ORDER BY id DESC LIMIT ?""",
                (person_id, history_limit),
            ).fetchall()
            texts.extend([r[0] for r in mem_rows if r and r[0]])
        except Exception:
            pass

        try:
            chat_rows = self.db.execute(
                """SELECT content FROM chat_history
                   WHERE user_id = ? AND role = 'user'
                   ORDER BY id DESC LIMIT ?""",
                (person_id, history_limit),
            ).fetchall()
            texts.extend([r[0] for r in chat_rows if r and r[0]])
        except Exception:
            pass

        if not texts:
            profile = self.get_canonical_profile(person_id)
            return {k: str(v.get("value") or "") for k, v in profile.items() if v.get("value")}
        return self.update_user_profile_from_texts(person_id, texts)

    # ══════════════════════════════════════════
    # MEMORY EXCHANGE + SHADOW TWIN
    # ══════════════════════════════════════════

    def _memory_exchange_enabled(self) -> bool:
        return bool(self.config.get("memory_exchange_enabled", True))

    def _shadow_twin_enabled(self) -> bool:
        return self._memory_exchange_enabled() and bool(
            self.config.get("shadow_twin_enabled", True))

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, (len(text or "") + 3) // 4)

    @staticmethod
    def _query_overlap(a: str, b: str) -> float:
        import re

        def words(s: str) -> set[str]:
            return {
                w for w in re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", (s or "").lower())
                if len(w) >= 3
            }

        wa = words(a)
        wb = words(b)
        if not wa or not wb:
            return 0.0
        inter = len(wa & wb)
        return inter / max(len(wa), len(wb))

    @staticmethod
    def _normalize_query_pattern(query: str) -> str:
        import re

        tokens = [
            token for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", (query or "").lower())
            if len(token) >= 3
        ]
        if not tokens:
            return ""
        # Deterministic token set improves recall for paraphrased queries.
        return " ".join(sorted(set(tokens)))

    @staticmethod
    def _score_context_pack(relevance: float, trust: float, novelty: float,
                            token_estimate: int) -> float:
        rel = max(0.0, min(1.0, relevance))
        tr = max(0.0, min(1.0, trust))
        nov = max(0.0, min(1.0, novelty))
        tok = max(1, int(token_estimate))
        return round((rel * tr * nov) * (1000.0 / tok), 6)

    def _memory_exchange_daemon_enabled(self) -> bool:
        return self._memory_exchange_enabled() and bool(
            self.config.get("memory_exchange_daemon_enabled", True))

    def _memory_exchange_daemon_interval_sec(self) -> float:
        try:
            raw = float(self.config.get("memory_exchange_daemon_interval_sec", 1.0))
        except (TypeError, ValueError):
            raw = 1.0
        return max(0.1, min(raw, 30.0))

    def _memory_exchange_daemon_batch_size(self) -> int:
        try:
            raw = int(self.config.get("memory_exchange_daemon_batch_size", 3))
        except (TypeError, ValueError):
            raw = 3
        return max(1, min(raw, 20))

    def _memory_exchange_max_attempts(self) -> int:
        try:
            raw = int(self.config.get("memory_exchange_max_attempts", 3))
        except (TypeError, ValueError):
            raw = 3
        return max(1, min(raw, 10))

    def _memory_exchange_queue_max_pending(self) -> int:
        try:
            raw = int(self.config.get("memory_exchange_queue_max_pending", 5000))
        except (TypeError, ValueError):
            raw = 5000
        return max(100, min(raw, 200000))

    def _memory_local_worker_enabled(self) -> bool:
        return self._memory_exchange_daemon_enabled() and bool(
            self.config.get("memory_local_worker_enabled", True))

    def _memory_local_worker_interval_sec(self) -> float:
        try:
            raw = float(self.config.get("memory_local_worker_interval_sec", 12.0))
        except (TypeError, ValueError):
            raw = 12.0
        return max(2.0, min(raw, 300.0))

    def _memory_local_worker_batch_size(self) -> int:
        try:
            raw = int(self.config.get("memory_local_worker_batch_size", 24))
        except (TypeError, ValueError):
            raw = 24
        return max(4, min(raw, 500))

    def _shadow_queue_cleanup_enabled(self) -> bool:
        return self._memory_exchange_enabled() and bool(
            self.config.get("shadow_queue_cleanup_enabled", True))

    def _shadow_queue_cleanup_interval_sec(self) -> float:
        try:
            raw = float(self.config.get("shadow_queue_cleanup_interval_sec", 60.0))
        except (TypeError, ValueError):
            raw = 60.0
        return max(5.0, min(raw, 3600.0))

    def _shadow_ready_ttl_hours(self) -> int:
        try:
            raw = int(self.config.get("shadow_ready_ttl_hours", 24))
        except (TypeError, ValueError):
            raw = 24
        return max(1, min(raw, 720))

    def _shadow_used_ttl_hours(self) -> int:
        try:
            raw = int(self.config.get("shadow_used_ttl_hours", 72))
        except (TypeError, ValueError):
            raw = 72
        return max(1, min(raw, 1440))

    def _shadow_max_ready_per_user(self) -> int:
        try:
            raw = int(self.config.get("shadow_max_ready_per_user", 120))
        except (TypeError, ValueError):
            raw = 120
        return max(20, min(raw, 5000))

    def _classify_memory_exchange_priority(self, anchor_query: str) -> int:
        """Lower value means higher priority in queue."""
        q = (anchor_query or "").strip().lower()
        if not q:
            return 5
        high_markers = (
            "как меня зовут", "кто я", "помнишь", "что ты помнишь",
            "my name", "who am i", "remember", "what do you remember",
        )
        if any(m in q for m in high_markers):
            return 1
        if len(q) <= 80:
            return 3
        if len(q) >= 350:
            return 7
        return 5

    def _store_memory_exchange_intent(self, user_id: str, anchor_query: str,
                                      payload: dict, source: str = "turn",
                                      *, priority: int = 5, status: str = "queued",
                                      processed: int = 0) -> int | None:
        user_id = self.get_canonical_person_id(user_id)
        try:
            cur = self.db.execute(
                """INSERT INTO memory_exchange_intents
                   (user_id, source, anchor_query, payload_json, processed,
                    priority, status, attempts, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (user_id, source, anchor_query[:300],
                 json.dumps(payload, ensure_ascii=False),
                 int(processed),
                 max(1, min(int(priority), 9)),
                 status,
                 datetime.now().isoformat()))
            self.db.commit()
            return int(cur.lastrowid)
        except Exception:
            return None

    async def enqueue_memory_exchange_intent(self, user_input: str, user_id: str,
                                             assistant_response: str = "",
                                             source: str = "turn",
                                             priority: int | None = None) -> dict:
        """Queue memory-exchange work for daemon execution."""
        user_id = self.get_canonical_person_id(user_id)
        if not self._memory_exchange_enabled():
            return {"status": "disabled"}
        anchor = (user_input or "").strip()
        if len(anchor) < 3:
            return {"status": "skipped"}

        # Lazy-start daemon for non-API channels (CLI/Telegram) on first enqueue.
        if self._memory_exchange_daemon_enabled():
            try:
                await self.start_memory_exchange_daemon()
            except Exception:
                pass

        queue_max = self._memory_exchange_queue_max_pending()
        pending = self.db.execute(
            """SELECT COUNT(*) FROM memory_exchange_intents
               WHERE status IN ('queued', 'running')"""
        ).fetchone()[0]
        if int(pending or 0) >= queue_max:
            return {"status": "throttled", "pending": int(pending or 0)}

        prio = int(priority) if priority is not None else self._classify_memory_exchange_priority(anchor)
        intent_id = self._store_memory_exchange_intent(
            user_id=user_id,
            anchor_query=anchor,
            payload={"user_input": anchor[:400], "assistant_response": assistant_response[:400]},
            source=source,
            priority=prio,
            status="queued",
            processed=0,
        )
        if not intent_id:
            return {"status": "error"}
        return {"status": "queued", "intent_id": intent_id, "priority": prio}

    def _select_memories_for_pack(self, memories: list[dict], budget_tokens: int) -> list[dict]:
        selected = []
        used_tokens = 0
        seen = set()
        for m in memories:
            mid = m.get("id")
            content = str(m.get("content", "")).strip()
            if not content or mid in seen:
                continue
            text = content[:360]
            tok = self._estimate_tokens(text)
            if used_tokens + tok > budget_tokens:
                continue
            selected.append({
                "id": int(mid) if mid is not None else None,
                "content": text,
                "score": float(m.get("score", 0.0)),
                "importance": float(m.get("importance", 0.5)),
                "tokens": tok,
            })
            used_tokens += tok
            seen.add(mid)
        return selected

    def _upsert_context_pack(self, user_id: str, query_hint: str,
                             selected: list[dict]) -> str | None:
        if not selected:
            return None
        now = datetime.now().isoformat()
        source_ids = [s["id"] for s in selected if s.get("id") is not None]
        lines = [f"- {s['content']}" for s in selected]
        content = "\n".join(lines)
        token_estimate = self._estimate_tokens(content)
        relevance = sum(s.get("score", 0.0) for s in selected) / max(len(selected), 1)
        trust = sum(s.get("importance", 0.5) for s in selected) / max(len(selected), 1)

        prev = self.db.execute(
            """SELECT id, source_memory_ids FROM memory_context_packs
               WHERE user_id = ? AND LOWER(query_hint) = LOWER(?)
               ORDER BY updated_at DESC LIMIT 1""",
            (user_id, query_hint[:300]),
        ).fetchone()
        novelty = 1.0
        if prev:
            try:
                prev_ids = set(json.loads(prev[1] or "[]"))
                cur_ids = set(source_ids)
                if prev_ids or cur_ids:
                    novelty = 1.0 - (len(prev_ids & cur_ids) / max(len(prev_ids | cur_ids), 1))
            except Exception:
                novelty = 1.0
        score = self._score_context_pack(relevance, trust, novelty, token_estimate)

        key = hashlib.md5(
            f"{user_id}|{query_hint.lower()}|{content.lower()}".encode("utf-8")
        ).hexdigest()
        pack_id = f"mxp_{key[:24]}"
        title = (query_hint or "memory pack")[:120]
        exists = self.db.execute(
            "SELECT id FROM memory_context_packs WHERE id = ?",
            (pack_id,),
        ).fetchone()
        if exists:
            self.db.execute(
                """UPDATE memory_context_packs
                   SET title = ?, source_memory_ids = ?, relevance = ?, trust = ?,
                       novelty = ?, token_estimate = ?, score = ?, updated_at = ?
                   WHERE id = ?""",
                (title, json.dumps(source_ids), relevance, trust,
                 novelty, token_estimate, score, now, pack_id))
        else:
            self.db.execute(
                """INSERT INTO memory_context_packs
                   (id, user_id, query_hint, title, content, source_memory_ids,
                    relevance, trust, novelty, token_estimate, score,
                    hit_count, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                (pack_id, user_id, query_hint[:300], title, content,
                 json.dumps(source_ids), relevance, trust, novelty,
                 token_estimate, score, now, now))
        self.db.commit()
        return pack_id

    def _create_pack_for_query(self, user_id: str, query: str) -> str | None:
        q = (query or "").strip()
        if len(q) < 3:
            return None
        top_k = int(self.config.get("memory_exchange_top_k", 8) or 8)
        memories = self.recall(q, user_id, top_k=max(3, min(top_k, 20)))
        if not memories:
            memories = self._chat_history_candidates(user_id, q, limit=30)
        if not memories:
            return None
        budget = int(self.config.get("memory_exchange_pack_budget_tokens", 450) or 450)
        selected = self._select_memories_for_pack(memories, max(100, min(budget, 3000)))
        return self._upsert_context_pack(user_id, q, selected)

    def _chat_history_candidates(self, user_id: str, query: str,
                                 limit: int = 30) -> list[dict]:
        """Fallback candidates from recent chat history when semantic memory is sparse."""
        rows = self.db.execute(
            """SELECT id, role, content
               FROM chat_history
               WHERE user_id = ?
               ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        out = []
        for row in rows:
            row_id, role, content_raw = row
            content = content_raw
            try:
                parsed = json.loads(content_raw)
                if isinstance(parsed, list):
                    parts = []
                    for b in parsed:
                        if isinstance(b, dict) and b.get("type") == "text":
                            parts.append(str(b.get("text", "")))
                    content = " ".join(parts).strip() or content_raw
            except Exception:
                pass
            overlap = self._query_overlap(query, content)
            if overlap < 0.15:
                continue
            out.append({
                "id": int(-row_id),
                "content": content[:360],
                "score": min(0.8, 0.3 + overlap),
                "importance": 0.55 if role == "user" else 0.45,
            })
        return out

    def _predict_shadow_queries_heuristic(self, user_input: str, user_id: str,
                                          assistant_response: str = "") -> list[tuple[str, float]]:
        base = (user_input or "").strip()
        out: list[tuple[str, float]] = []
        if base:
            out.append((base, 0.9))

        if self._graph_enabled():
            entities = self.entity_search(base, user_id, top_k=3)
            for ent in entities:
                name = str(ent.get("name", "")).strip()
                if not name:
                    continue
                score = float(ent.get("score", 0.5))
                out.append((f"{name} details", min(0.85, max(0.45, score))))

        if assistant_response:
            first = str(assistant_response).split("\n", 1)[0].strip()
            if len(first) > 8:
                out.append((first[:120], 0.4))

        uniq: list[tuple[str, float]] = []
        seen = set()
        for q, c in out:
            key = q.lower().strip()
            if key and key not in seen:
                uniq.append((q.strip(), float(c)))
                seen.add(key)
        max_n = int(self.config.get("shadow_twin_predictions", 3) or 3)
        return uniq[:max(1, min(max_n, 8))]

    async def _predict_shadow_queries_llm(self, user_input: str) -> list[str]:
        provider = self._get_extraction_provider()
        if not provider:
            return []
        prompt = (
            "Generate up to 3 likely follow-up user queries for this message. "
            "Return JSON array of strings only.\n\n"
            f"Message: {user_input}"
        )
        model = self._get_extraction_model("claude-haiku-4-5-20251001")
        try:
            result = await provider.complete(
                model=model,
                max_tokens=120,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(result, "usage") and result.usage:
                self.track_internal_cost(model, result.usage)
            text = result.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            arr = _safe_parse_llm_json(text, [])
            if isinstance(arr, list):
                out = []
                for item in arr:
                    q = str(item).strip()
                    if len(q) >= 3:
                        out.append(q[:180])
                return out[:3]
        except Exception:
            return []
        return []

    async def _run_memory_exchange_cycle_core(self, anchor: str, user_id: str,
                                              assistant_response: str = "") -> dict:
        """Core pack/prediction builder logic used by sync call and daemon."""
        queries: list[tuple[str, float]] = []
        if self._shadow_twin_enabled() and self.config.get("shadow_twin_use_llm", False):
            async with self._extraction_semaphore:
                llm_q = await self._predict_shadow_queries_llm(anchor)
            if llm_q:
                queries.extend([(q, 0.75 - i * 0.1) for i, q in enumerate(llm_q)])

        if not queries:
            queries = self._predict_shadow_queries_heuristic(anchor, user_id, assistant_response)

        packs_created = 0
        predictions_created = 0
        for query, confidence in queries:
            pack_id = self._create_pack_for_query(user_id, query)
            if not pack_id:
                continue
            packs_created += 1
            if self._shadow_twin_enabled():
                now = datetime.now().isoformat()
                existing = self.db.execute(
                    """SELECT id FROM memory_shadow_predictions
                       WHERE user_id = ? AND LOWER(predicted_query) = LOWER(?)
                       ORDER BY created_at DESC LIMIT 1""",
                    (user_id, query[:300]),
                ).fetchone()
                if existing:
                    self.db.execute(
                        """UPDATE memory_shadow_predictions
                           SET anchor_query = ?, confidence = ?, pack_id = ?,
                               status = 'ready', created_at = ?, used_at = NULL
                           WHERE id = ?""",
                        (anchor[:300], float(confidence), pack_id, now, str(existing[0])),
                    )
                else:
                    self.db.execute(
                        """INSERT INTO memory_shadow_predictions
                           (id, user_id, anchor_query, predicted_query, confidence,
                            pack_id, hit_count, status, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, 0, 'ready', ?)""",
                        (str(uuid.uuid4()), user_id, anchor[:300], query[:300],
                         float(confidence), pack_id, now),
                    )
                predictions_created += 1
        self.db.commit()
        return {
            "status": "ok",
            "packs_created": packs_created,
            "predictions_created": predictions_created,
        }

    def _mark_memory_exchange_intent_done(self, intent_id: int):
        now = datetime.now().isoformat()
        self.db.execute(
            """UPDATE memory_exchange_intents
               SET status = 'done', processed = 1, processed_at = ?, locked_at = NULL,
                   locked_by = NULL, last_error = NULL
               WHERE id = ?""",
            (now, intent_id),
        )
        self.db.commit()

    def _mark_memory_exchange_intent_failed(self, intent_id: int, error: str,
                                            *, requeue: bool):
        now = datetime.now().isoformat()
        if requeue:
            self.db.execute(
                """UPDATE memory_exchange_intents
                   SET status = 'queued', processed = 0, locked_at = NULL, locked_by = NULL,
                       last_error = ?, processed_at = NULL
                   WHERE id = ?""",
                (str(error)[:400], intent_id),
            )
        else:
            self.db.execute(
                """UPDATE memory_exchange_intents
                   SET status = 'failed', processed = 1, locked_at = NULL, locked_by = NULL,
                       last_error = ?, processed_at = ?
                   WHERE id = ?""",
                (str(error)[:400], now, intent_id),
            )
        self.db.commit()

    def _memory_exchange_is_high_load(self) -> tuple[bool, str]:
        """Pause background worker when foreground request pressure is high."""
        if not bool(self.config.get("memory_exchange_daemon_auto_pause", True)):
            return False, ""
        active_threshold = int(self.config.get("memory_exchange_daemon_pause_active_requests", 1) or 1)
        queued_threshold = int(self.config.get("memory_exchange_daemon_pause_queued_requests", 2) or 2)
        try:
            from .agent import LiteAgent
            active = len(LiteAgent.get_active_requests())
            queued = len(LiteAgent.get_queued_requests())
        except Exception:
            return False, ""
        if active >= max(0, active_threshold):
            return True, f"active_requests={active}"
        if queued >= max(0, queued_threshold):
            return True, f"queued_requests={queued}"
        return False, ""

    async def process_memory_exchange_queue_once(self, max_items: int | None = None) -> dict:
        """Process queued intents once, respecting priority and load auto-pause."""
        if not self._memory_exchange_enabled():
            return {"status": "disabled", "processed": 0, "failed": 0}

        shadow_cleanup = self.cleanup_shadow_prediction_queue()

        paused, reason = self._memory_exchange_is_high_load()
        if paused:
            self._mx_daemon_last_pause_reason = reason
            self._mx_daemon_last_pause_at = datetime.now().timestamp()
            return {
                "status": "paused",
                "reason": reason,
                "processed": 0,
                "failed": 0,
                "shadow_cleanup": shadow_cleanup,
            }

        batch_size = max_items or self._memory_exchange_daemon_batch_size()
        rows = self.db.execute(
            """SELECT id, user_id, anchor_query, payload_json, attempts
               FROM memory_exchange_intents
               WHERE status = 'queued'
               ORDER BY priority ASC, created_at ASC
               LIMIT ?""",
            (batch_size * 3,),
        ).fetchall()
        if not rows:
            return {"status": "idle", "processed": 0, "failed": 0, "shadow_cleanup": shadow_cleanup}

        claimed = []
        now = datetime.now().isoformat()
        for row in rows:
            if len(claimed) >= batch_size:
                break
            intent_id = int(row[0])
            cur = self.db.execute(
                """UPDATE memory_exchange_intents
                   SET status = 'running', locked_at = ?, locked_by = ?, attempts = attempts + 1
                   WHERE id = ? AND status = 'queued'""",
                (now, self._mx_daemon_worker_id, intent_id),
            )
            if cur.rowcount:
                claimed.append({
                    "id": intent_id,
                    "user_id": str(row[1] or ""),
                    "anchor_query": str(row[2] or ""),
                    "payload_json": str(row[3] or "{}"),
                    "attempts_before": int(row[4] or 0),
                })
        self.db.commit()

        if not claimed:
            return {"status": "idle", "processed": 0, "failed": 0, "shadow_cleanup": shadow_cleanup}

        processed = 0
        failed = 0
        max_attempts = self._memory_exchange_max_attempts()
        for intent in claimed:
            try:
                payload = json.loads(intent["payload_json"] or "{}")
                assistant_response = str(payload.get("assistant_response", ""))
            except Exception:
                assistant_response = ""
            try:
                await self._run_memory_exchange_cycle_core(
                    intent["anchor_query"],
                    intent["user_id"],
                    assistant_response,
                )
                self._mark_memory_exchange_intent_done(intent["id"])
                processed += 1
            except Exception as e:
                attempts = intent["attempts_before"] + 1
                self._mark_memory_exchange_intent_failed(
                    intent["id"],
                    str(e),
                    requeue=attempts < max_attempts,
                )
                failed += 1

        status = "ok" if processed else "failed"
        return {
            "status": status,
            "processed": processed,
            "failed": failed,
            "shadow_cleanup": shadow_cleanup,
        }

    def cleanup_shadow_prediction_queue(self, *, force: bool = False) -> dict:
        """Prune stale/duplicate shadow predictions to keep queue healthy."""
        if not self._shadow_queue_cleanup_enabled():
            return {"status": "disabled", "removed_total": 0}

        now_ts = datetime.now().timestamp()
        interval = self._shadow_queue_cleanup_interval_sec()
        if (not force) and (now_ts - float(self._mx_shadow_cleanup_last_run or 0.0) < interval):
            return {"status": "skipped", "removed_total": 0}

        removed_orphan = removed_stale_ready = removed_stale_used = removed_dup = removed_capped = 0

        # Remove predictions pointing to missing context packs.
        cur = self.db.execute(
            """DELETE FROM memory_shadow_predictions
               WHERE pack_id IS NOT NULL
                 AND NOT EXISTS (
                    SELECT 1 FROM memory_context_packs p
                    WHERE p.id = memory_shadow_predictions.pack_id
                 )"""
        )
        removed_orphan += max(0, int(cur.rowcount or 0))

        ready_ttl = f"-{self._shadow_ready_ttl_hours()} hours"
        used_ttl = f"-{self._shadow_used_ttl_hours()} hours"

        cur = self.db.execute(
            """DELETE FROM memory_shadow_predictions
               WHERE status = 'ready'
                 AND COALESCE(datetime(created_at), datetime(replace(created_at, 'T', ' ')))
                     < datetime('now', ?)""",
            (ready_ttl,),
        )
        removed_stale_ready += max(0, int(cur.rowcount or 0))

        cur = self.db.execute(
            """DELETE FROM memory_shadow_predictions
               WHERE status = 'used'
                 AND COALESCE(datetime(used_at), COALESCE(datetime(created_at), datetime(replace(created_at, 'T', ' '))))
                     < datetime('now', ?)""",
            (used_ttl,),
        )
        removed_stale_used += max(0, int(cur.rowcount or 0))

        # Keep one latest ready prediction per (user, predicted_query).
        rows = self.db.execute(
            """SELECT id, user_id, predicted_query, confidence, created_at
               FROM memory_shadow_predictions
               WHERE status = 'ready'
               ORDER BY user_id ASC, LOWER(predicted_query) ASC,
                        COALESCE(datetime(created_at), datetime(replace(created_at, 'T', ' '))) DESC,
                        confidence DESC"""
        ).fetchall()
        seen: set[tuple[str, str]] = set()
        to_delete: list[str] = []
        for row in rows:
            pid = str(row[0] or "")
            uid = str(row[1] or "")
            pq = str(row[2] or "").strip().lower()
            if not pid or not uid or not pq:
                continue
            key = (uid, pq)
            if key in seen:
                to_delete.append(pid)
                continue
            seen.add(key)
        if to_delete:
            self.db.executemany(
                "DELETE FROM memory_shadow_predictions WHERE id = ?",
                [(pid,) for pid in to_delete],
            )
            removed_dup += len(to_delete)

        # Cap ready predictions per user.
        cap = self._shadow_max_ready_per_user()
        users = self.db.execute(
            "SELECT DISTINCT user_id FROM memory_shadow_predictions WHERE status = 'ready'"
        ).fetchall()
        for row in users:
            uid = str(row[0] or "")
            if not uid:
                continue
            ids = [
                str(r[0]) for r in self.db.execute(
                    """SELECT id
                       FROM memory_shadow_predictions
                       WHERE status = 'ready' AND user_id = ?
                       ORDER BY confidence DESC,
                                COALESCE(datetime(created_at), datetime(replace(created_at, 'T', ' '))) DESC""",
                    (uid,),
                ).fetchall()
            ]
            if len(ids) <= cap:
                continue
            overflow = ids[cap:]
            self.db.executemany(
                "DELETE FROM memory_shadow_predictions WHERE id = ?",
                [(pid,) for pid in overflow],
            )
            removed_capped += len(overflow)

        self.db.commit()
        removed_total = removed_orphan + removed_stale_ready + removed_stale_used + removed_dup + removed_capped
        stats = {
            "status": "ok",
            "removed_total": int(removed_total),
            "removed_orphan": int(removed_orphan),
            "removed_stale_ready": int(removed_stale_ready),
            "removed_stale_used": int(removed_stale_used),
            "removed_duplicates": int(removed_dup),
            "removed_over_cap": int(removed_capped),
            "updated_at": datetime.now().isoformat(),
        }
        self._mx_shadow_cleanup_last_run = now_ts
        self._mx_shadow_cleanup_last_stats = stats
        return stats

    async def run_local_memory_worker_once(self, max_items: int | None = None) -> dict:
        """Background local worker: profile slot refresh + graph back-linking."""
        if not self._memory_local_worker_enabled():
            return {"status": "disabled", "processed": 0}

        paused, reason = self._memory_exchange_is_high_load()
        if paused:
            return {"status": "paused", "reason": reason, "processed": 0}

        batch = max_items or self._memory_local_worker_batch_size()
        try:
            last_id = int(self.get_state("app:memory_local_worker_last_id") or 0)
        except Exception:
            last_id = 0

        rows = self.db.execute(
            """SELECT id, user_id, content
               FROM memories
               WHERE id > ? AND archived_at IS NULL
               ORDER BY id ASC LIMIT ?""",
            (last_id, batch),
        ).fetchall()
        if not rows:
            return {"status": "idle", "processed": 0}

        processed = 0
        slots_updated = 0
        links_added = 0
        entity_cache: dict[str, dict[str, str]] = {}
        last_seen_id = last_id
        graph_on = self._graph_enabled()

        for row in rows:
            mem_id = int(row[0])
            uid = self.get_canonical_person_id(str(row[1] or "default"))
            text = str(row[2] or "")
            last_seen_id = max(last_seen_id, mem_id)
            if not text:
                continue
            if self._is_memory_pollution_text(text) or self._is_assistant_meta_statement(text):
                processed += 1
                continue

            # Slow worker continuously refreshes canonical slots from fresh memories.
            extracted = self._extract_profile_facts(text)
            for slot_key, slot_value in extracted.items():
                if slot_key not in _CANONICAL_PROFILE_SLOTS:
                    continue
                res = self.upsert_canonical_slot(
                    uid, slot_key, slot_value,
                    confidence=0.56, source="local-worker",
                )
                if res:
                    slots_updated += 1

            # Optional graph back-linking for memories that were ingested without graph extraction.
            if graph_on:
                if uid not in entity_cache:
                    ent_rows = self.db.execute(
                        """SELECT id, name FROM memory_entities
                           WHERE user_id = ?
                           ORDER BY mention_count DESC
                           LIMIT 300""",
                        (uid,),
                    ).fetchall()
                    entity_cache[uid] = {
                        str(name): str(eid) for eid, name in ent_rows if eid and name
                    }
                if entity_cache.get(uid):
                    links_added += self._link_memory_to_entities(mem_id, text, entity_cache[uid])

            processed += 1

        self.set_state("app:memory_local_worker_last_id", last_seen_id)
        self._mx_local_worker_last_run = datetime.now().timestamp()
        self._mx_local_worker_last_stats = {
            "processed": processed,
            "slots_updated": slots_updated,
            "links_added": links_added,
            "last_memory_id": last_seen_id,
            "updated_at": datetime.now().isoformat(),
        }
        return {"status": "ok", **self._mx_local_worker_last_stats}

    async def _memory_exchange_daemon_loop(self):
        interval = self._memory_exchange_daemon_interval_sec()
        while self._mx_daemon_running:
            try:
                res = await self.process_memory_exchange_queue_once()
                now_ts = datetime.now().timestamp()
                if self._memory_local_worker_enabled() and (
                    now_ts - float(self._mx_local_worker_last_run or 0.0)
                ) >= self._memory_local_worker_interval_sec():
                    await self.run_local_memory_worker_once()
                if res.get("status") == "ok" and res.get("processed", 0) > 0:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Memory exchange daemon error: %s", e)
                await asyncio.sleep(interval)

    async def start_memory_exchange_daemon(self) -> dict:
        """Start always-on memory exchange worker loop."""
        if not self._memory_exchange_daemon_enabled():
            return {"status": "disabled"}
        if self._mx_daemon_task and not self._mx_daemon_task.done():
            return {"status": "already_running"}
        self._mx_daemon_running = True
        self._mx_daemon_task = asyncio.create_task(self._memory_exchange_daemon_loop())
        logger.info("Memory exchange daemon started (worker=%s)", self._mx_daemon_worker_id)
        return {"status": "started", "worker_id": self._mx_daemon_worker_id}

    async def stop_memory_exchange_daemon(self) -> dict:
        """Stop always-on memory exchange worker loop."""
        self._mx_daemon_running = False
        task = self._mx_daemon_task
        self._mx_daemon_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return {"status": "stopped"}

    def memory_exchange_daemon_state(self) -> dict:
        """Current daemon state for dashboard/API telemetry."""
        running = bool(self._mx_daemon_task and not self._mx_daemon_task.done())
        return {
            "enabled": self._memory_exchange_daemon_enabled(),
            "running": running,
            "worker_id": self._mx_daemon_worker_id,
            "last_pause_reason": self._mx_daemon_last_pause_reason,
            "last_pause_at": self._mx_daemon_last_pause_at,
            "local_worker_enabled": self._memory_local_worker_enabled(),
            "local_worker_interval_sec": self._memory_local_worker_interval_sec(),
            "local_worker_last_run": self._mx_local_worker_last_run,
            "local_worker_last_stats": self._mx_local_worker_last_stats,
            "shadow_cleanup_enabled": self._shadow_queue_cleanup_enabled(),
            "shadow_cleanup_interval_sec": self._shadow_queue_cleanup_interval_sec(),
            "shadow_cleanup_last_run": self._mx_shadow_cleanup_last_run,
            "shadow_cleanup_last_stats": self._mx_shadow_cleanup_last_stats,
        }

    async def run_memory_exchange_cycle(self, user_input: str, user_id: str,
                                        assistant_response: str = "") -> dict:
        """Build/update token-efficient context packs and shadow predictions."""
        user_id = self.get_canonical_person_id(user_id)
        if not self._memory_exchange_enabled():
            return {"status": "disabled"}
        anchor = (user_input or "").strip()
        if len(anchor) < 3:
            return {"status": "skipped"}

        intent_id = self._store_memory_exchange_intent(
            user_id=user_id,
            anchor_query=anchor,
            payload={"user_input": anchor[:400], "assistant_response": assistant_response[:400]},
            priority=self._classify_memory_exchange_priority(anchor),
            status="running",
            processed=0,
        )
        try:
            result = await self._run_memory_exchange_cycle_core(anchor, user_id, assistant_response)
            if intent_id:
                self._mark_memory_exchange_intent_done(intent_id)
            return result
        except Exception as e:
            if intent_id:
                self._mark_memory_exchange_intent_failed(intent_id, str(e), requeue=False)
            raise

    def get_memory_exchange_context(self, query: str, user_id: str,
                                    max_packs: int | None = None,
                                    token_budget: int | None = None) -> str:
        """Return precomputed memory packs relevant to current query."""
        user_id = self.get_canonical_person_id(user_id)
        if not self._memory_exchange_enabled():
            return ""
        self.cleanup_shadow_prediction_queue()
        q = (query or "").strip()
        if len(q) < 3:
            return ""

        max_packs = int(max_packs or self.config.get("memory_exchange_max_packs", 2) or 2)
        max_packs = max(1, min(max_packs, 5))
        token_budget = int(token_budget or self.config.get("memory_exchange_context_budget_tokens", 700) or 700)
        token_budget = max(120, min(token_budget, 4000))

        candidates = []
        pred_rows = self.db.execute(
            """SELECT s.id as pred_id, s.predicted_query, s.confidence,
                      p.id, p.title, p.content, p.token_estimate, p.score, p.query_hint
               FROM memory_shadow_predictions s
               JOIN memory_context_packs p ON p.id = s.pack_id
               WHERE s.user_id = ? AND s.status = 'ready'
               ORDER BY s.created_at DESC LIMIT 40""",
            (user_id,),
        ).fetchall()
        for row in pred_rows:
            overlap = self._query_overlap(q, row[1] or "")
            if overlap < 0.35 and (q.lower() != (row[1] or "").lower()):
                continue
            weight = float(row[7] or 0.0) * (1.0 + float(row[2] or 0.0) + overlap)
            candidates.append({
                "pack_id": row[3],
                "title": row[4],
                "content": row[5] or "",
                "tokens": int(row[6] or self._estimate_tokens(row[5] or "")),
                "score": float(row[7] or 0.0),
                "query_hint": row[8] or "",
                "weight": weight,
                "pred_id": row[0],
            })

        pack_rows = self.db.execute(
            """SELECT id, title, content, token_estimate, score, query_hint
               FROM memory_context_packs
               WHERE user_id = ?
               ORDER BY score DESC, updated_at DESC LIMIT 40""",
            (user_id,),
        ).fetchall()
        for row in pack_rows:
            overlap = self._query_overlap(q, row[5] or "")
            if overlap < 0.2 and q.lower() not in (row[5] or "").lower():
                continue
            weight = float(row[4] or 0.0) * (1.0 + overlap)
            candidates.append({
                "pack_id": row[0],
                "title": row[1],
                "content": row[2] or "",
                "tokens": int(row[3] or self._estimate_tokens(row[2] or "")),
                "score": float(row[4] or 0.0),
                "query_hint": row[5] or "",
                "weight": weight,
                "pred_id": None,
            })

        if not candidates:
            return ""

        candidates.sort(key=lambda x: x["weight"], reverse=True)
        selected = []
        seen = set()
        used_tokens = 0
        for item in candidates:
            pid = item["pack_id"]
            if pid in seen:
                continue
            tok = max(1, int(item["tokens"]))
            if used_tokens + tok > token_budget:
                continue
            selected.append(item)
            seen.add(pid)
            used_tokens += tok
            if len(selected) >= max_packs:
                break

        if not selected:
            return ""

        now = datetime.now().isoformat()
        for item in selected:
            self.db.execute(
                "UPDATE memory_context_packs SET hit_count = hit_count + 1, last_used = ? WHERE id = ?",
                (now, item["pack_id"]),
            )
            if item.get("pred_id"):
                self.db.execute(
                    """UPDATE memory_shadow_predictions
                       SET hit_count = hit_count + 1, used_at = ?, status = 'used'
                       WHERE id = ?""",
                    (now, item["pred_id"]),
                )
        self.db.commit()

        lines = ["## Memory Exchange (precomputed):"]
        for idx, item in enumerate(selected, 1):
            title = item["title"] or item["query_hint"] or f"pack {idx}"
            lines.append(f"[Pack {idx}] {title} | score={item['score']:.3f}")
            lines.append(item["content"])
        return "\n".join(lines)

    @staticmethod
    def _classify_query_intent(query: str) -> dict[str, Any]:
        q = " ".join(str(query or "").strip().lower().split())
        if not q:
            return {"personal": False, "slot": "", "query": q}
        slot_markers = {
            "name": (
                "как меня зовут", "мое имя", "моё имя", "my name", "what is my name",
                "remember my name", "кто я",
            ),
            "language": (
                "на каком языке", "мой язык", "какой язык", "which language",
                "language do i prefer",
            ),
            "role": (
                "кто я по роли", "моя роль", "кем я работаю", "my role", "what is my role",
                "what do i do",
            ),
        }
        slot = ""
        for k, markers in slot_markers.items():
            if any(m in q for m in markers):
                slot = k
                break
        personal_markers = (
            "что ты знаешь обо мне", "помнишь меня", "about me",
            "who am i", "remember me", "what do you remember about me",
        )
        personal = bool(slot or any(m in q for m in personal_markers))
        return {"personal": personal, "slot": slot, "query": q}

    @staticmethod
    def _slot_memory_match(slot_key: str, content: str, expected_value: str = "") -> bool:
        text = str(content or "").lower()
        if not text:
            return False
        if expected_value and expected_value.lower() in text:
            return True
        slot_markers = {
            "name": ("меня зовут", "мое имя", "моё имя", "my name", "user name"),
            "language": ("на русском", "на английском", "language", "язык"),
            "role": ("работаю", "являюсь", "i work as", "i am a", "role"),
        }
        return any(m in text for m in slot_markers.get(slot_key, ()))

    def _build_profile_slot_memory(self, user_id: str, slot_key: str) -> tuple[dict | None, str]:
        if slot_key not in _CANONICAL_PROFILE_SLOTS:
            return None, ""
        resolved = self.resolve_profile_slot(user_id, slot_key, lookback=220, auto_heal=False)
        value = str(resolved.get("value") or "").strip()
        conf = float(resolved.get("confidence", 0.0) or 0.0)
        if not value:
            return None, ""

        labels = {
            "name": "name",
            "language": "preferred language",
            "role": "role",
        }
        content = f"User {labels.get(slot_key, slot_key)} is {value}."
        return ({
            "id": -1,
            "content": content,
            "type": "profile_slot",
            "score": 1.2 + min(conf, 1.0) * 0.3,
            "importance": conf,
            "slot_key": slot_key,
            "slot_value": value,
        }, value)

    def _store_recall_trace(self, user_id: str, query: str, strategy: str,
                            results: list[dict], intent: dict | None = None,
                            profile_expected: str = "") -> None:
        uid = self.get_canonical_person_id(user_id)
        top = []
        for r in (results or [])[:5]:
            top.append({
                "id": int(r.get("id", -1)) if str(r.get("id", "")).lstrip("-").isdigit() else -1,
                "type": str(r.get("type", "fact")),
                "score": round(float(r.get("score", 0.0)), 4),
                "content": str(r.get("content", ""))[:240],
            })
        slot = str((intent or {}).get("slot") or "")
        expected = str(profile_expected or "")
        hit = 0
        if slot and expected:
            low = expected.lower()
            hit = 1 if any(low in str(t.get("content", "")).lower() for t in top) else 0
        trace = {
            "user_id": uid,
            "query": str(query or ""),
            "strategy": strategy,
            "intent_slot": slot,
            "profile_expected": expected,
            "profile_hit": int(hit),
            "top_memories": top,
            "created_at": datetime.now().isoformat(),
        }
        self._last_recall_trace[uid] = trace
        try:
            self.db.execute(
                """INSERT INTO memory_recall_traces
                   (user_id, query, strategy, intent_slot, profile_expected, profile_hit, top_memories_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uid,
                    str(query or "")[:400],
                    strategy[:40],
                    slot[:40],
                    expected[:160],
                    int(hit),
                    json.dumps(top, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            self.db.commit()
        except Exception:
            pass

    def get_last_recall_trace(self, user_id: str | None = None, limit: int = 1) -> list[dict]:
        lim = max(1, min(int(limit), 20))
        if user_id:
            uid = self.get_canonical_person_id(user_id)
            rows = self.db.execute(
                """SELECT user_id, query, strategy, intent_slot, profile_expected, profile_hit,
                          top_memories_json, created_at
                   FROM memory_recall_traces
                   WHERE user_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (uid, lim),
            ).fetchall()
        else:
            rows = self.db.execute(
                """SELECT user_id, query, strategy, intent_slot, profile_expected, profile_hit,
                          top_memories_json, created_at
                   FROM memory_recall_traces
                   ORDER BY id DESC LIMIT ?""",
                (lim,),
            ).fetchall()
        out = []
        for row in rows:
            try:
                top = json.loads(row[6] or "[]")
            except Exception:
                top = []
            out.append({
                "user_id": str(row[0] or ""),
                "query": str(row[1] or ""),
                "strategy": str(row[2] or ""),
                "intent_slot": str(row[3] or ""),
                "profile_expected": str(row[4] or ""),
                "profile_hit": int(row[5] or 0),
                "top_memories": top,
                "created_at": row[7],
            })
        return out

    def recall_type_aware(self, query: str, user_id: str, top_k: int = 5) -> list[dict]:
        """Type-aware retrieval for personal/profile questions."""
        uid = self.get_canonical_person_id(user_id)
        intent = self._classify_query_intent(query)
        # Deep pool for post-ranking; keep normal recall behavior for non-personal queries.
        pool = self.recall(query, uid, top_k=max(top_k * 3, 12))
        if not intent.get("personal"):
            return pool[:top_k]

        expected = ""
        injected: list[dict] = []
        slot_key = str(intent.get("slot") or "")
        if slot_key:
            slot_mem, expected = self._build_profile_slot_memory(uid, slot_key)
            if slot_mem:
                injected.append(slot_mem)

        def type_bonus(mem_type: str) -> float:
            bonus = {
                "fact": 0.35,
                "preference": 0.22,
                "correction": 0.12,
                "profile_slot": 0.45,
            }
            return bonus.get(str(mem_type or ""), 0.0)

        primary: list[dict] = []
        secondary: list[dict] = []
        for item in pool:
            content = str(item.get("content", ""))
            if slot_key and self._slot_memory_match(slot_key, content, expected):
                primary.append(item)
            else:
                secondary.append(item)

        primary.sort(key=lambda m: float(m.get("score", 0.0)) + type_bonus(m.get("type", "")), reverse=True)
        secondary.sort(key=lambda m: float(m.get("score", 0.0)) + type_bonus(m.get("type", "")), reverse=True)

        merged: list[dict] = []
        seen: set[tuple[int, str]] = set()
        for item in injected + primary + secondary:
            key = (int(item.get("id", -1)), str(item.get("content", ""))[:220].lower())
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= top_k:
                break

        self._store_recall_trace(uid, query, "type_aware", merged, intent, expected)
        return merged

    # ══════════════════════════════════════════
    # L3: SEMANTIC MEMORY
    # ══════════════════════════════════════════

    def recall(self, query: str, user_id: str, top_k: int = 5) -> list[dict]:
        """Find relevant memories using hybrid search: vector + BM25 + temporal decay.

        Search modes (config.memory.search_mode):
        - "hybrid" (default): RRF fusion of vector + BM25 keyword results
        - "vector": embedding cosine similarity only
        - "keyword": BM25 via FTS5 only (or word-overlap fallback)
        """
        import time as _time
        _recall_start = _time.monotonic()
        user_id = self.get_canonical_person_id(user_id)
        search_mode = self.config.get("search_mode", "hybrid")

        # Temporal decay config
        mem_cfg = self._config.get("memory", {}) if hasattr(self, '_config') else {}
        use_temporal_decay = mem_cfg.get("temporal_decay_enabled", True)
        decay_rate = mem_cfg.get("temporal_decay_rate", 0.01)

        # --- BM25 keyword search via FTS5 ---
        # Apply query expansion (from OpenClaw) to improve FTS recall
        fts_query = query
        if search_mode in ("hybrid", "keyword"):
            try:
                from .query_expansion import maybe_expand
                fts_query = maybe_expand(query)
                if fts_query != query:
                    logger.debug("Memory FTS query expanded: %r → %r", query, fts_query)
            except ImportError:
                pass

        bm25_results = []
        if search_mode in ("hybrid", "keyword"):
            bm25_results = self._fts_search(fts_query, user_id, top_k=20)

        # --- Vector search (brute-force over embeddings) ---
        vector_results = []
        query_embedding = None
        if search_mode in ("hybrid", "vector") and self._embedder is not None:
            query_embedding = self._embedder.encode(query)
            vector_results = self._vector_search(query_embedding, user_id, top_k=20)

        # --- Graph memory search ---
        graph_results = self._graph_recall(query, user_id, top_k=10)

        # --- Merge results ---
        if search_mode == "hybrid" and bm25_results and vector_results:
            merged = self._rrf_fusion(vector_results, bm25_results)
        elif vector_results:
            merged = vector_results
        elif bm25_results:
            merged = bm25_results
        else:
            # Fallback: legacy word-overlap search (no embedder, no FTS5)
            merged = self._keyword_fallback_search(query, user_id, top_k=20)

        # Merge graph results into main results via RRF
        if graph_results and merged:
            merged = self._rrf_fusion(merged, graph_results)
        elif graph_results:
            merged = graph_results

        if not merged:
            return []

        # --- Enrich with importance + temporal decay ---
        affinity_boosts = self._query_affinity_boosts(query, user_id)
        penalty_boosts = self._query_penalty_scores(query, user_id)
        memory_meta = self._get_memory_metadata(user_id, [r["id"] for r in merged])
        scored = []
        for item in merged:
            mid = item["id"]
            if mid not in memory_meta:
                continue
            meta = memory_meta[mid]
            importance = meta.get("importance", 0.5)
            created_at = meta.get("created_at", "")
            accessed_at = meta.get("accessed_at", "")
            mtype = meta.get("type", "fact")
            content = str(item.get("content", meta.get("content", "")))

            # Drop known low-signal memory capability disclaimers from all memory types.
            # This prevents old polluted facts/corrections from dominating recall.
            if self._is_memory_pollution_text(content) or self._is_self_referential_memory_limit(content):
                continue
            if self._is_cross_script_noise(query, content):
                continue

            relevance = item.get("score", 0.0)
            # Normalize relevance to 0-1 range
            relevance = min(1.0, max(0.0, relevance))

            if use_temporal_decay:
                decay = self._temporal_decay_score(created_at, accessed_at, decay_rate)
                score = relevance * 0.5 + importance * 0.25 + decay * 0.25
            else:
                recency = self._recency_score(created_at)
                score = relevance * 0.6 + importance * 0.3 + recency * 0.1
            affinity_boost = float(affinity_boosts.get(mid, 0.0) or 0.0)
            penalty_boost = float(penalty_boosts.get(mid, 0.0) or 0.0)
            score += affinity_boost
            score -= penalty_boost

            scored.append({
                "id": mid,
                "content": content,
                "type": mtype,
                "score": score,
                "importance": importance,
                "affinity_boost": affinity_boost,
                "penalty_boost": penalty_boost,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        # MMR re-ranking (from OpenClaw) — diversify when enabled in config
        mem_mmr_cfg = mem_cfg.get("mmr", {})
        if mem_mmr_cfg.get("enabled", False) and len(scored) > 1:
            try:
                from .mmr import mmr_rerank, MMRConfig, MMRItem
                mmr_conf = MMRConfig(
                    enabled=True,
                    lambda_=mem_mmr_cfg.get("lambda", 0.7),
                )
                mmr_items: list[MMRItem] = [
                    {"id": str(r["id"]), "score": r["score"], "content": r["content"]}
                    for r in scored
                ]
                reranked = mmr_rerank(mmr_items, mmr_conf, top_k=top_k)
                id_to_item = {str(r["id"]): r for r in scored}
                top_results = [id_to_item[item["id"]]
                               for item in reranked if item["id"] in id_to_item]
            except Exception as mmr_err:
                logger.debug("MMR re-ranking failed: %s", mmr_err)
                top_results = scored[:top_k]
        else:
            top_results = scored[:top_k]

        # Touch-on-access: update accessed_at for retrieved memories
        accessed_ids = [r["id"] for r in top_results if r.get("id")]
        if accessed_ids:
            now = datetime.now().isoformat()
            for mid in accessed_ids:
                self.db.execute(
                    "UPDATE memories SET accessed_at = ? WHERE id = ?",
                    (now, mid))
            self.db.commit()

        # Log query metrics
        if self.config.get("metrics_enabled", False):
            latency = (_time.monotonic() - _recall_start) * 1000
            top_score = top_results[0]["score"] if top_results else 0.0
            self._log_query(user_id, query, search_mode,
                            len(top_results), top_score, latency)

        # Explainability trace (latest retrieval rationale).
        self._store_recall_trace(user_id, query, search_mode, top_results)

        return top_results

    def _log_query(self, user_id: str, query: str, search_mode: str,
                   result_count: int, top_score: float, latency_ms: float):
        """Log a memory query for metrics tracking."""
        user_id = self.get_canonical_person_id(user_id)
        try:
            self.db.execute(
                """INSERT INTO memory_query_log
                   (user_id, query, search_mode, result_count, top_score, latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, query[:200], search_mode, result_count,
                 top_score, latency_ms))
            self.db.commit()
        except Exception:
            pass  # Don't fail recall if logging fails

    def mark_query_used(self, query_log_id: int):
        """Mark a logged query as actually used (results were helpful)."""
        self.db.execute(
            "UPDATE memory_query_log SET was_used = 1 WHERE id = ?",
            (query_log_id,))
        self.db.commit()

    def reinforce_recall(self, query: str, user_id: str, memory_ids: list[int],
                         *, strength: float = 1.0, source: str = "recall") -> int:
        """Reinforce memories that were actually used for a query.

        This creates a lightweight feedback loop:
        - marks matching query logs as used
        - slightly boosts importance of the memories
        - stores query→memory affinity for future similar recalls
        """
        uid = self.get_canonical_person_id(user_id)
        qnorm = self._normalize_query_pattern(query)
        if not uid or not qnorm:
            return 0

        unique_ids = []
        seen = set()
        for raw in memory_ids or []:
            try:
                mid = int(raw)
            except (TypeError, ValueError):
                continue
            if mid <= 0 or mid in seen:
                continue
            seen.add(mid)
            unique_ids.append(mid)
        if not unique_ids:
            return 0

        strength = max(0.1, min(float(strength or 1.0), 3.0))
        now = datetime.now().isoformat()
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self.db.execute(
            f"""SELECT id FROM memories
                WHERE user_id = ? AND archived_at IS NULL
                  AND id IN ({placeholders})""",
            (uid, *unique_ids),
        ).fetchall()
        valid_ids = [int(row[0]) for row in rows]
        if not valid_ids:
            return 0

        importance_bump = min(0.08, 0.02 + (strength * 0.02))
        self.db.executemany(
            """UPDATE memories
               SET accessed_at = ?,
                   importance = MIN(importance + ?, 1.0)
               WHERE id = ?""",
            [(now, importance_bump, mid) for mid in valid_ids],
        )

        self.db.execute(
            """UPDATE memory_query_log
               SET was_used = 1
               WHERE id IN (
                   SELECT id FROM memory_query_log
                   WHERE user_id = ? AND LOWER(query) = LOWER(?)
                   ORDER BY id DESC LIMIT 5
               )""",
            (uid, str(query or "").strip()),
        )

        for mid in valid_ids:
            self.db.execute(
                """INSERT INTO memory_query_affinity
                   (user_id, query_norm, memory_id, hit_count, total_strength, last_source, last_used)
                   VALUES (?, ?, ?, 1, ?, ?, ?)
                   ON CONFLICT(user_id, query_norm, memory_id) DO UPDATE SET
                       hit_count = hit_count + 1,
                       total_strength = total_strength + excluded.total_strength,
                       last_source = excluded.last_source,
                       last_used = excluded.last_used""",
                (uid, qnorm, mid, strength, str(source or "")[:40], now),
            )

        self.db.commit()
        return len(valid_ids)

    def penalize_recall(self, query: str, user_id: str, memory_ids: list[int],
                        *, strength: float = 1.0, source: str = "recall") -> int:
        """Slightly down-rank memories repeatedly shown but not selected for a query."""
        uid = self.get_canonical_person_id(user_id)
        qnorm = self._normalize_query_pattern(query)
        if not uid or not qnorm:
            return 0

        unique_ids = []
        seen = set()
        for raw in memory_ids or []:
            try:
                mid = int(raw)
            except (TypeError, ValueError):
                continue
            if mid <= 0 or mid in seen:
                continue
            seen.add(mid)
            unique_ids.append(mid)
        if not unique_ids:
            return 0

        strength = max(0.1, min(float(strength or 1.0), 3.0))
        now = datetime.now().isoformat()
        placeholders = ",".join("?" for _ in unique_ids)
        rows = self.db.execute(
            f"""SELECT id FROM memories
                WHERE user_id = ? AND archived_at IS NULL
                  AND id IN ({placeholders})""",
            (uid, *unique_ids),
        ).fetchall()
        valid_ids = [int(row[0]) for row in rows]
        if not valid_ids:
            return 0

        importance_drop = min(0.04, 0.008 + (strength * 0.008))
        floor = 0.08
        self.db.executemany(
            """UPDATE memories
               SET importance = MAX(importance - ?, ?)
               WHERE id = ?""",
            [(importance_drop, floor, mid) for mid in valid_ids],
        )

        for mid in valid_ids:
            self.db.execute(
                """INSERT INTO memory_query_penalty
                   (user_id, query_norm, memory_id, miss_count, total_penalty, last_source, last_used)
                   VALUES (?, ?, ?, 1, ?, ?, ?)
                   ON CONFLICT(user_id, query_norm, memory_id) DO UPDATE SET
                       miss_count = miss_count + 1,
                       total_penalty = total_penalty + excluded.total_penalty,
                       last_source = excluded.last_source,
                       last_used = excluded.last_used""",
                (uid, qnorm, mid, strength, str(source or "")[:40], now),
            )

        self.db.commit()
        return len(valid_ids)

    def register_recall_feedback(self, query: str, user_id: str, shown_ids: list[int],
                                 used_ids: list[int], *, strength: float = 1.0,
                                 source: str = "recall") -> dict[str, int]:
        """Apply balanced recall feedback for one retrieval event.

        `used_ids` are reinforced. `shown_ids - used_ids` get a small penalty so the
        same query gradually prefers the memories that actually make it into context.
        """
        shown_unique = []
        shown_seen = set()
        for raw in shown_ids or []:
            try:
                mid = int(raw)
            except (TypeError, ValueError):
                continue
            if mid <= 0 or mid in shown_seen:
                continue
            shown_seen.add(mid)
            shown_unique.append(mid)

        used_unique = []
        used_seen = set()
        for raw in used_ids or []:
            try:
                mid = int(raw)
            except (TypeError, ValueError):
                continue
            if mid <= 0 or mid in used_seen:
                continue
            used_seen.add(mid)
            used_unique.append(mid)

        unused = [mid for mid in shown_unique if mid not in used_seen]
        reinforced = self.reinforce_recall(
            query, user_id, used_unique, strength=strength, source=source)
        penalized = 0
        if unused:
            penalty_strength = max(0.1, min(strength * 0.6, 1.8))
            penalized = self.penalize_recall(
                query, user_id, unused, strength=penalty_strength, source=source)
        return {
            "shown": len(shown_unique),
            "used": len(used_unique),
            "reinforced": reinforced,
            "penalized": penalized,
        }

    def _query_affinity_boosts(self, query: str, user_id: str, limit: int = 120) -> dict[int, float]:
        """Load small query→memory boosts learned from prior successful recalls."""
        uid = self.get_canonical_person_id(user_id)
        qnorm = self._normalize_query_pattern(query)
        if not uid or not qnorm:
            return {}

        rows = self.db.execute(
            """SELECT query_norm, memory_id, hit_count, total_strength
               FROM memory_query_affinity
               WHERE user_id = ?
               ORDER BY last_used DESC
               LIMIT ?""",
            (uid, max(10, min(int(limit or 120), 400))),
        ).fetchall()
        boosts: dict[int, float] = {}
        for row_query, raw_mid, raw_hits, raw_strength in rows:
            overlap = 1.0 if str(row_query or "") == qnorm else self._query_overlap(qnorm, str(row_query or ""))
            if overlap < 0.45:
                continue
            try:
                mid = int(raw_mid)
            except (TypeError, ValueError):
                continue
            hits = max(1, int(raw_hits or 1))
            total_strength = max(0.1, float(raw_strength or 1.0))
            boost = min(0.22, 0.04 * overlap + 0.025 * min(hits, 4) + 0.02 * min(total_strength, 3.0))
            if overlap >= 0.99:
                boost = min(0.28, boost + 0.05)
            if boost > boosts.get(mid, 0.0):
                boosts[mid] = boost
        return boosts

    def _query_penalty_scores(self, query: str, user_id: str, limit: int = 120) -> dict[int, float]:
        """Load small query→memory penalties learned from repeated non-selection."""
        uid = self.get_canonical_person_id(user_id)
        qnorm = self._normalize_query_pattern(query)
        if not uid or not qnorm:
            return {}

        rows = self.db.execute(
            """SELECT query_norm, memory_id, miss_count, total_penalty
               FROM memory_query_penalty
               WHERE user_id = ?
               ORDER BY last_used DESC
               LIMIT ?""",
            (uid, max(10, min(int(limit or 120), 400))),
        ).fetchall()
        penalties: dict[int, float] = {}
        for row_query, raw_mid, raw_misses, raw_penalty in rows:
            overlap = 1.0 if str(row_query or "") == qnorm else self._query_overlap(qnorm, str(row_query or ""))
            if overlap < 0.45:
                continue
            try:
                mid = int(raw_mid)
            except (TypeError, ValueError):
                continue
            misses = max(1, int(raw_misses or 1))
            total_penalty = max(0.1, float(raw_penalty or 1.0))
            penalty = min(0.18, 0.018 * overlap + 0.02 * min(misses, 4) + 0.015 * min(total_penalty, 3.0))
            if overlap >= 0.99:
                penalty = min(0.22, penalty + 0.04)
            if penalty > penalties.get(mid, 0.0):
                penalties[mid] = penalty
        return penalties

    def get_memory_quality_metrics(self, user_id: str | None = None,
                                   days: int = 30, k: int = 5) -> dict[str, Any]:
        """SaaS-friendly memory quality KPIs."""
        cutoff = (datetime.now() - timedelta(days=max(1, int(days)))).isoformat()
        uid = self.get_canonical_person_id(user_id) if user_id else ""

        q_params: list[Any] = [cutoff]
        q_filter = ""
        if uid:
            q_filter = "AND user_id = ?"
            q_params.append(uid)
        q_row = self.db.execute(
            f"""SELECT COUNT(*),
                       SUM(CASE WHEN result_count > 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN top_score >= 0.2 THEN 1 ELSE 0 END)
                FROM memory_query_log
                WHERE created_at >= ? {q_filter}""",
            q_params,
        ).fetchone() or (0, 0, 0)
        total_queries = int(q_row[0] or 0)
        queries_with_results = int(q_row[1] or 0)
        confident_hits = int(q_row[2] or 0)

        recall_at_k = round(queries_with_results / max(total_queries, 1), 3)
        recall_confident_at_k = round(confident_hits / max(total_queries, 1), 3)

        t_params: list[Any] = [cutoff]
        t_filter = ""
        if uid:
            t_filter = "AND user_id = ?"
            t_params.append(uid)
        trace_row = self.db.execute(
            f"""SELECT COUNT(*), SUM(profile_hit)
                FROM memory_recall_traces
                WHERE created_at >= ? AND intent_slot != '' {t_filter}""",
            t_params,
        ).fetchone() or (0, 0)
        profile_checks = int(trace_row[0] or 0)
        profile_hits = int(trace_row[1] or 0)
        profile_accuracy = round(profile_hits / max(profile_checks, 1), 3)

        if uid:
            c_params = [uid, cutoff]
            c_where = "WHERE person_id = ? AND created_at >= ?"
        else:
            c_params = [cutoff]
            c_where = "WHERE created_at >= ?"

        slot_total_row = self.db.execute(
            f"""SELECT COUNT(*) FROM (
                    SELECT person_id, slot_key
                    FROM canonical_profile_slot_history
                    {c_where}
                    GROUP BY person_id, slot_key
                )""",
            c_params,
        ).fetchone()
        slots_with_history = int((slot_total_row or [0])[0] or 0)
        contradiction_row = self.db.execute(
            f"""SELECT COUNT(*) FROM (
                    SELECT person_id, slot_key
                    FROM canonical_profile_slot_history
                    {c_where}
                    GROUP BY person_id, slot_key
                    HAVING COUNT(DISTINCT LOWER(slot_value)) > 1
                )""",
            c_params,
        ).fetchone()
        contradictory_slots = int((contradiction_row or [0])[0] or 0)
        contradiction_rate = round(contradictory_slots / max(slots_with_history, 1), 3)

        e_params: list[Any] = [cutoff]
        e_filter = ""
        if uid:
            e_filter = "AND user_id = ?"
            e_params.append(uid)
        ext_row = self.db.execute(
            f"""SELECT COALESCE(SUM(total_candidates), 0),
                       COALESCE(SUM(saved_count), 0),
                       COALESCE(SUM(dropped_pollution), 0)
                FROM memory_extraction_runs
                WHERE created_at >= ? {e_filter}""",
            e_params,
        ).fetchone() or (0, 0, 0)
        extraction_candidates = int(ext_row[0] or 0)
        extraction_saved = int(ext_row[1] or 0)
        extraction_dropped = int(ext_row[2] or 0)
        memory_poison_rate = round(extraction_dropped / max(extraction_candidates, 1), 3)

        return {
            "window_days": max(1, int(days)),
            "k": max(1, int(k)),
            "recall_at_k": recall_at_k,
            "recall_confident_at_k": recall_confident_at_k,
            "profile_accuracy": profile_accuracy,
            "contradiction_rate": contradiction_rate,
            "memory_poison_rate": memory_poison_rate,
            "counts": {
                "queries_total": total_queries,
                "profile_checks": profile_checks,
                "slots_with_history": slots_with_history,
                "contradictory_slots": contradictory_slots,
                "extraction_candidates": extraction_candidates,
                "extraction_saved": extraction_saved,
                "extraction_dropped": extraction_dropped,
            },
        }

    def get_memory_metrics(self, user_id: str | None = None,
                           days: int = 30) -> dict:
        """Compute memory health metrics for a user (or all users).

        Returns:
            dict with hit_rate, avg_latency_ms, avg_score, total_queries,
            usage_rate, staleness stats, total_memories, entity_count, etc.
        """
        if user_id:
            user_id = self.get_canonical_person_id(user_id)
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        params: list = [cutoff]
        user_filter = ""
        if user_id:
            user_filter = "AND user_id = ?"
            params.append(user_id)

        # Query stats
        stats = self.db.execute(
            f"""SELECT COUNT(*), AVG(result_count), AVG(top_score),
                       AVG(latency_ms), SUM(was_used)
                FROM memory_query_log WHERE created_at > ? {user_filter}""",
            params).fetchone()
        total_queries = stats[0] or 0
        avg_results = stats[1] or 0
        avg_score = stats[2] or 0
        avg_latency = stats[3] or 0
        used_count = stats[4] or 0

        hit_rate = (total_queries - (self.db.execute(
            f"SELECT COUNT(*) FROM memory_query_log WHERE created_at > ? "
            f"AND result_count = 0 {user_filter}", params
        ).fetchone()[0] or 0)) / max(total_queries, 1)

        usage_rate = used_count / max(total_queries, 1)

        # Memory stats
        mem_params = [user_id] if user_id else []
        mem_filter = "WHERE user_id = ?" if user_id else ""
        total_memories = self.db.execute(
            f"SELECT COUNT(*) FROM memories {mem_filter}",
            mem_params).fetchone()[0]

        # Stale memories (not accessed in 30 days)
        stale_cutoff = (datetime.now() - timedelta(days=30)).isoformat()
        stale = self.db.execute(
            f"""SELECT COUNT(*) FROM memories {mem_filter}
                {'AND' if mem_filter else 'WHERE'} accessed_at < ?""",
            mem_params + [stale_cutoff]).fetchone()[0]

        # Entity count
        entity_count = self.db.execute(
            f"SELECT COUNT(*) FROM memory_entities {mem_filter}",
            mem_params).fetchone()[0]

        # Episode count
        episode_count = self.db.execute(
            f"SELECT COUNT(*) FROM episodes {mem_filter}",
            mem_params).fetchone()[0]

        # Procedure count
        procedure_count = self.db.execute(
            f"SELECT COUNT(*) FROM procedures {mem_filter}",
            mem_params).fetchone()[0]
        thinking_filter = "WHERE user_id = ? AND status = 'active'" if user_id else "WHERE status = 'active'"
        thinking_total = self.db.execute(
            f"SELECT COUNT(*) FROM thinking_notes {thinking_filter}",
            mem_params,
        ).fetchone()[0]
        thinking_theme_count = self.db.execute(
            f"SELECT COUNT(*) FROM thinking_notes {thinking_filter} AND note_type = 'theme'",
            mem_params,
        ).fetchone()[0]
        thinking_open_questions = self.db.execute(
            f"SELECT COUNT(*) FROM thinking_notes {thinking_filter} AND note_type = 'open_question'",
            mem_params,
        ).fetchone()[0]

        return {
            "total_queries": total_queries,
            "hit_rate": round(hit_rate, 3),
            "usage_rate": round(usage_rate, 3),
            "avg_results": round(avg_results, 2),
            "avg_score": round(avg_score, 3),
            "avg_latency_ms": round(avg_latency, 1),
            "total_memories": total_memories,
            "stale_memories": stale,
            "entity_count": entity_count,
            "episode_count": episode_count,
            "procedure_count": procedure_count,
            "thinking_notes": thinking_total,
            "thinking_themes": thinking_theme_count,
            "thinking_open_questions": thinking_open_questions,
            "quality": self.get_memory_quality_metrics(user_id=user_id, days=days),
        }

    def memory_health_check(self, user_id: str | None = None) -> dict:
        """Run a quick health check on memory system. Returns status + issues."""
        metrics = self.get_memory_metrics(user_id)
        issues = []
        status = "healthy"

        if metrics["total_queries"] > 10 and metrics["hit_rate"] < 0.5:
            issues.append("Low hit rate — many queries return no results")
            status = "warning"
        if metrics["avg_score"] > 0 and metrics["avg_score"] < 0.2:
            issues.append("Low average relevance scores — embeddings may need re-indexing")
            status = "warning"
        if metrics["total_memories"] > 0:
            stale_ratio = metrics["stale_memories"] / metrics["total_memories"]
            if stale_ratio > 0.7:
                issues.append(f"{metrics['stale_memories']}/{metrics['total_memories']} "
                               "memories are stale (not accessed in 30 days)")
                status = "warning"
        if metrics["avg_latency_ms"] > 500:
            issues.append(f"High recall latency ({metrics['avg_latency_ms']:.0f}ms)")
            status = "degraded"

        return {
            "status": status,
            "issues": issues,
            "metrics": metrics,
        }

    def _vector_search(self, query_embedding, user_id: str, top_k: int = 20) -> list[dict]:
        """Vector search over memory embeddings (brute-force cosine similarity)."""
        rows = self.db.execute(
            """SELECT id, content, embedding
               FROM memories WHERE user_id = ? AND archived_at IS NULL
                 AND embedding IS NOT NULL
               ORDER BY importance DESC
               LIMIT 200""",
            (user_id,)).fetchall()
        if not rows:
            return []
        scored = []
        for row_id, content, emb_blob in rows:
            try:
                content_emb = pickle.loads(emb_blob)
                sim = self._cosine_similarity(query_embedding, content_emb)
                scored.append({"id": row_id, "content": content, "score": sim})
            except Exception:
                continue
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _keyword_fallback_search(self, query: str, user_id: str, top_k: int = 20) -> list[dict]:
        """Legacy word-overlap keyword search (used when FTS5 and embedder unavailable)."""
        rows = self.db.execute(
            """SELECT id, content FROM memories
               WHERE user_id = ? AND archived_at IS NULL
               ORDER BY importance DESC, created_at DESC
               LIMIT 50""",
            (user_id,)).fetchall()
        if not rows:
            return []
        query_words = set(query.lower().split())
        scored = []
        for row_id, content in rows:
            content_words = set(content.lower().split())
            overlap = len(query_words & content_words)
            score = overlap / max(len(query_words), 1)
            if score > 0:
                scored.append({"id": row_id, "content": content, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _rrf_fusion(vector_results: list[dict], keyword_results: list[dict],
                    k: int = _RRF_K) -> list[dict]:
        """Reciprocal Rank Fusion — merge two ranked lists.

        RRF score = Σ 1/(k + rank_i) for each list the document appears in.
        """
        scores: dict[int, float] = {}
        data: dict[int, dict] = {}

        for rank, item in enumerate(vector_results):
            mid = item["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
            data[mid] = item

        for rank, item in enumerate(keyword_results):
            mid = item["id"]
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
            if mid not in data:
                data[mid] = item

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for mid, score in ranked:
            item = data[mid].copy()
            item["score"] = round(score, 6)
            results.append(item)
        return results

    def _get_memory_metadata(self, user_id: str, memory_ids: list[int]) -> dict[int, dict]:
        """Fetch metadata (importance, type, timestamps) for a list of memory IDs."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        rows = self.db.execute(
            f"""SELECT id, content, type, importance, created_at, accessed_at
                FROM memories
                WHERE id IN ({placeholders}) AND user_id = ? AND archived_at IS NULL""",
            (*memory_ids, user_id)).fetchall()
        return {r[0]: {"content": r[1], "type": r[2], "importance": r[3],
                        "created_at": r[4] or "", "accessed_at": r[5] or ""}
                for r in rows}

    async def remember(self, content: str, user_id: str,
                       memory_type: str = "fact", importance: float = 0.5,
                       file_meta: str | None = None) -> int | None:
        """Store a new memory with deduplication, conflict detection, and optional embedding."""
        user_id = self.get_canonical_person_id(user_id)
        if memory_type in {"fact", "preference", "correction"}:
            if self._is_memory_pollution_text(content) or self._is_assistant_meta_statement(content):
                logger.debug("Skip polluted %s memory: %s", memory_type, str(content)[:120])
                return None
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
            return int(existing[0])

        # Conflict detection (if enabled)
        mcd_cfg = self._features_config.get("memory_conflict_detection", {})
        if mcd_cfg.get("enabled", False):
            conflicts = self.detect_memory_conflicts(
                content, user_id, memory_type,
                threshold=mcd_cfg.get("similarity_threshold", 0.75))
            if conflicts and mcd_cfg.get("auto_resolve", True):
                last_id = None
                for conflict in conflicts:
                    action = await self.resolve_memory_conflict(
                        content, conflict["existing"], user_id)
                    last_id = self._apply_conflict_resolution(
                        action, content, conflict["existing"],
                        user_id, memory_type, importance)
                return last_id  # Resolution handles storage

        embedding = self._embed(content)
        now = datetime.now().isoformat()
        cur = self.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash, created_at, accessed_at, embedding, file_meta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, content, memory_type, importance, content_hash,
             now, now, embedding, file_meta))
        # Sync FTS5 index
        self._fts_insert(cur.lastrowid, content, user_id)
        self.db.commit()
        extracted = self._extract_profile_facts(content)
        for slot_key, slot_value in extracted.items():
            if slot_key in _CANONICAL_PROFILE_SLOTS:
                base_conf = 0.67 if memory_type == "fact" else 0.58
                self.upsert_canonical_slot(
                    user_id, slot_key, slot_value,
                    confidence=base_conf + min(max(float(importance), 0.0), 1.0) * 0.2,
                    source=f"memory:{memory_type}",
                )
        return int(cur.lastrowid)

    def forget(self, user_id: str, content_fragment: str):
        """Delete memories matching a fragment."""
        user_id = self.get_canonical_person_id(user_id)
        # Get IDs for FTS5 cleanup before deleting
        rows = self.db.execute(
            "SELECT id FROM memories WHERE user_id=? AND content LIKE ?",
            (user_id, f"%{content_fragment}%")).fetchall()
        for (mid,) in rows:
            self._fts_delete(mid)
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
        provider = self._get_extraction_provider()
        if not provider:
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
            model = self._get_extraction_model("claude-haiku-4-5-20251001")
            result = await provider.complete(
                model=model,
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(result, "usage") and result.usage:
                self.track_internal_cost(model, result.usage)
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
                                    importance: float = 0.5) -> int | None:
        """Apply the chosen conflict resolution action and return active memory ID."""
        now = datetime.now().isoformat()
        new_hash = hashlib.md5(new_content.lower().strip().encode()).hexdigest()
        embedding = self._embed(new_content)

        if action == "replace":
            # Update existing memory with new content
            self.db.execute(
                """UPDATE memories SET content=?, hash=?, embedding=?,
                   accessed_at=?, importance=? WHERE id=?""",
                (new_content, new_hash, embedding, now, importance, existing["id"]))
            # Update FTS5
            self._fts_delete(existing["id"])
            self._fts_insert(existing["id"], new_content, user_id)
            active_id = int(existing["id"])

        elif action == "archive_old":
            # Archive old memory, insert new
            self.db.execute(
                "UPDATE memories SET archived_at=? WHERE id=?",
                (now, existing["id"]))
            self._fts_delete(existing["id"])
            cur = self.db.execute(
                """INSERT INTO memories (user_id, content, type, importance, hash,
                   created_at, accessed_at, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, new_content, memory_type, importance, new_hash,
                 now, now, embedding))
            self._fts_insert(cur.lastrowid, new_content, user_id)
            active_id = int(cur.lastrowid)

        elif action == "merge":
            # Merge: combine both contents
            merged = f"{existing['content']} [updated: {new_content}]"
            merged_hash = hashlib.md5(merged.lower().strip().encode()).hexdigest()
            merged_emb = self._embed(merged)
            self.db.execute(
                """UPDATE memories SET content=?, hash=?, embedding=?,
                   accessed_at=?, importance=MIN(importance+0.1, 1.0) WHERE id=?""",
                (merged, merged_hash, merged_emb, now, existing["id"]))
            # Update FTS5
            self._fts_delete(existing["id"])
            self._fts_insert(existing["id"], merged, user_id)
            active_id = int(existing["id"])

        else:  # keep_both
            cur = self.db.execute(
                """INSERT INTO memories (user_id, content, type, importance, hash,
                   created_at, accessed_at, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, new_content, memory_type, importance, new_hash,
                 now, now, embedding))
            self._fts_insert(cur.lastrowid, new_content, user_id)
            active_id = int(cur.lastrowid)

        self.db.commit()
        return active_id

    def get_archived_memories(self, user_id: str, limit: int = 20) -> list[dict]:
        """Get recently archived memories (for /conflicts command)."""
        user_id = self.get_canonical_person_id(user_id)
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

    async def extract_and_learn(self, user_input: str, agent_response: str, user_id: str,
                                file_meta: list | None = None):
        """Extract knowledge from conversation turn using cheap model."""
        user_id = self.get_canonical_person_id(user_id)
        # Keep a pinned profile snapshot even if LLM extraction is unavailable.
        # Use user utterance only: assistant self-descriptions must not override user profile.
        self.update_user_profile_from_texts(user_id, [user_input])

        provider = self._get_extraction_provider()
        if not provider or not self.config.get("auto_learn", True):
            return

        graph_on = self._graph_enabled()
        graph_schema = ""
        if graph_on:
            graph_schema = (
                ',"entities":[{"name":"...","type":"person|project|tool|concept|location"}],'
                '"relations":[{"source":"...","target":"...","type":"works_on|uses|prefers|knows|created|located_in|part_of"}]'
            )
        extraction_prompt = (
            "Analyze this conversation exchange. Extract ONLY genuinely new, "
            "specific facts — not opinions or generic statements.\n\n"
            f"User: {user_input}\n"
            f"Assistant: {agent_response}\n\n"
            'Return JSON only:\n'
            '{"facts":["..."],"preferences":["..."],"corrections":["..."],'
            '"session_summary":"one-line context update or empty string",'
            '"ideas":[{"content":"...","themes":["..."],"importance":0.0,"confidence":0.0,"novelty":0.0}],'
            '"constraints":[{"content":"...","themes":["..."],"importance":0.0,"confidence":0.0,"novelty":0.0}],'
            '"open_questions":[{"content":"...","themes":["..."],"importance":0.0,"confidence":0.0,"novelty":0.0}],'
            '"decision_signals":[{"content":"...","themes":["..."],"importance":0.0,"confidence":0.0,"novelty":0.0}],'
            '"directions":[{"content":"...","themes":["..."],"importance":0.0,"confidence":0.0,"novelty":0.0}]'
            + graph_schema + '}\n\n'
            "Rules:\n"
            "- Treat USER message as source of truth for facts/preferences/corrections.\n"
            "- Never save assistant self-descriptions, capability disclaimers, or model limitations as user memory.\n"
            "- facts: concrete info about the user (name, job, projects, etc.)\n"
            "- preferences: how user likes things done (language, format, style)\n"
            "- corrections: if user corrected a previous assumption\n"
            "- session_summary: brief note of what was discussed (for context compression)\n"
            "- ideas: durable product/technical/business ideas the user is exploring\n"
            "- constraints: stable boundaries or tradeoffs the user cares about\n"
            "- open_questions: unresolved questions the user is actively thinking through\n"
            "- decision_signals: explicit criteria such as quality, speed, local-first, cost sensitivity\n"
            "- directions: recurring long-term themes or areas the user is moving toward\n"
            "- For ideas/constraints/open_questions/decision_signals/directions, extract only if they are strategic or durable enough to improve future collaboration.\n"
            "- importance/confidence/novelty must be 0.0-1.0 floats if present.\n"
            + ("- entities: named things mentioned (people, projects, tools, places, concepts)\n"
               "- relations: connections between entities (who works on what, uses what tool, etc.)\n"
               if graph_on else "")
            + "- Empty arrays/strings if nothing new. Don't invent facts."
        )

        try:
            # Throttle background memory extraction calls to keep local models responsive.
            async with self._extraction_semaphore:
                model = self._get_extraction_model("claude-haiku-4-5-20251001")
                result = await provider.complete(
                    model=model,
                    max_tokens=500 if graph_on else 300,
                    messages=[{"role": "user", "content": extraction_prompt}],
                )
                if hasattr(result, "usage") and result.usage:
                    self.track_internal_cost(model, result.usage, user_id)
                raw_text = result.content[0].text if result.content else None
                if not raw_text:
                    return
                text = raw_text.strip()
                # Handle markdown code blocks
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

                _fallback_data = {
                    "facts": [],
                    "preferences": [],
                    "corrections": [],
                    "ideas": [],
                    "constraints": [],
                    "open_questions": [],
                    "decision_signals": [],
                    "directions": [],
                }
                data = _safe_parse_llm_json(text, _fallback_data)
                if not isinstance(data, dict):
                    data = _fallback_data
                raw_facts = self._coerce_string_list(data.get("facts"))
                raw_preferences = self._coerce_string_list(data.get("preferences"))
                raw_corrections = self._coerce_string_list(data.get("corrections"))
                dropped_pollution = 0
                user_slot_values = {
                    str(v).strip().lower()
                    for v in self._extract_profile_facts(user_input).values()
                    if str(v).strip()
                }

                def _clean_items(items: list[str], *, correction: bool = False) -> list[str]:
                    nonlocal dropped_pollution
                    cleaned: list[str] = []
                    for item in items:
                        low_item = item.lower()
                        if self._is_memory_pollution_text(item) or self._is_assistant_meta_statement(item):
                            dropped_pollution += 1
                            continue
                        if correction and self._is_self_referential_memory_limit(item):
                            dropped_pollution += 1
                            continue
                        overlap_user = self._query_overlap(item, user_input)
                        overlap_assistant = self._query_overlap(item, agent_response)
                        grounded_by_slot = any(v and v in low_item for v in user_slot_values)
                        correction_semantic = False
                        if correction:
                            has_contrast = any(tok in low_item for tok in (
                                " not ", " isn't ", " is not ", " instead ", " rather than ",
                                " не ", " а не ", "неверно", "ошибка", "правильно",
                            ))
                            mentions_identity = any(tok in low_item for tok in (
                                "name", "my name", "имя", "зовут",
                                "language", "язык", "role", "роль",
                                "location", "локац", "i am", "я ",
                            ))
                            correction_semantic = has_contrast and mentions_identity
                        if correction and not (overlap_user >= 0.09 or grounded_by_slot or correction_semantic):
                            dropped_pollution += 1
                            continue
                        if overlap_user < 0.06 and overlap_assistant >= 0.22 and not grounded_by_slot:
                            dropped_pollution += 1
                            continue
                        if overlap_assistant >= 0.18 and self._is_operational_memory_noise(item):
                            dropped_pollution += 1
                            continue
                        cleaned.append(item)
                    return cleaned

                facts = _clean_items(raw_facts)
                preferences = _clean_items(raw_preferences)
                corrections = _clean_items(raw_corrections, correction=True)

                # Graph memory: store entities and relations
                entity_map: dict[str, str] = {}  # entity name -> entity_id
                if graph_on:
                    graph_entities = self._coerce_object_list(data.get("entities"))
                    graph_relations = self._coerce_object_list(data.get("relations"))
                    for ent in graph_entities:
                        ename = str(ent.get("name", "") or "").strip()
                        etype = str(ent.get("type", "concept") or "concept").strip()
                        if ename and len(ename) > 1:
                            eid = self.upsert_entity(ename, etype, user_id)
                            if eid:
                                entity_map[ename] = eid
                    for rel in graph_relations:
                        src = str(rel.get("source", "") or "").strip()
                        tgt = str(rel.get("target", "") or "").strip()
                        rtype = str(rel.get("type", "related_to") or "related_to").strip()
                        if src and tgt and src != tgt:
                            # Auto-create entities if not yet created
                            if src not in entity_map:
                                eid = self.upsert_entity(src, "concept", user_id)
                                if eid:
                                    entity_map[src] = eid
                            if tgt not in entity_map:
                                eid = self.upsert_entity(tgt, "concept", user_id)
                                if eid:
                                    entity_map[tgt] = eid
                            self.upsert_relation(src, tgt, rtype, user_id)
                    if entity_map:
                        logger.debug("Graph: %d entities, %d relations",
                                     len(entity_map), len(graph_relations))

                # Serialize file_meta once for all memories from this turn
                _fm_json = json.dumps(file_meta, ensure_ascii=False) if file_meta else None

                # Persist memories and attach graph links when entity names match.
                saved_count = 0
                for fact in facts:
                    if len(fact) > 10:  # Skip trivial
                        mid = await self.remember(fact, user_id, "fact", 0.6, file_meta=_fm_json)
                        if graph_on and mid:
                            self._link_memory_to_entities(mid, fact, entity_map)
                        if mid:
                            saved_count += 1
                for pref in preferences:
                    if len(pref) > 10:
                        mid = await self.remember(pref, user_id, "preference", 0.8, file_meta=_fm_json)
                        if graph_on and mid:
                            self._link_memory_to_entities(mid, pref, entity_map)
                        if mid:
                            saved_count += 1
                for correction in corrections:
                    if len(correction) > 10:
                        mid = await self.remember(correction, user_id, "correction", 0.9, file_meta=_fm_json)
                        if graph_on and mid:
                            self._link_memory_to_entities(mid, correction, entity_map)
                        if mid:
                            saved_count += 1

                thinking_saved = self.store_thinking_cloud_items(user_id, data)

                # Update pinned profile from extracted statements as well.
                profile_sources = []
                profile_sources.extend(facts)
                profile_sources.extend(preferences)
                profile_sources.extend(corrections)
                if profile_sources:
                    self.update_user_profile_from_texts(user_id, profile_sources)

                # Update session summary for context compression (async-aware)
                summary_update = self._coerce_summary_text(data.get("session_summary", ""))
                if summary_update and len(summary_update) > 5:
                    await self._update_session_summary_async(user_id, summary_update)

                extracted = len(facts) + len(preferences) + len(corrections)
                total_candidates = len(raw_facts) + len(raw_preferences) + len(raw_corrections)
                self.db.execute(
                    """INSERT INTO memory_extraction_runs
                       (user_id, total_candidates, saved_count, dropped_pollution, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (user_id, int(total_candidates), int(saved_count), int(dropped_pollution),
                     datetime.now().isoformat()),
                )
                self.db.commit()
                if extracted or thinking_saved.get("saved", 0):
                    logger.debug(
                        "Extracted %d memory items and %d thinking notes from conversation",
                        extracted,
                        int(thinking_saved.get("saved", 0) or 0),
                    )

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
        """Append to session summary, with LLM compression when it grows too long."""
        existing = self._get_session_summary(user_id) or ""
        new_summary = (existing + " " + summary_update).strip()
        # Compress with LLM when summary exceeds threshold
        if len(new_summary) > 600:
            new_summary = self._compress_summary_sync(new_summary)
        # Hard cap fallback
        if len(new_summary) > 800:
            new_summary = new_summary[-800:]
        self.db.execute(
            "INSERT OR REPLACE INTO session_summaries VALUES (?, ?, ?)",
            (user_id, new_summary, datetime.now().isoformat()))
        self.db.commit()

    def _compress_summary_sync(self, summary: str) -> str:
        """Compress a session summary using LLM (synchronous wrapper for async context)."""
        if not self._get_extraction_provider():
            # No LLM available — truncate from the beginning
            return summary[-500:]
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — schedule as a task
                # Fall back to simple truncation; async version called separately
                return summary[-500:]
            return loop.run_until_complete(self._compress_summary(summary))
        except RuntimeError:
            return summary[-500:]

    async def _compress_summary(self, summary: str) -> str:
        """Compress a session summary using LLM to preserve semantic content."""
        provider = self._get_extraction_provider()
        if not provider:
            return summary[-500:]
        prompt = (
            "Compress this conversation summary into a concise version (max 400 chars). "
            "Keep key facts, decisions, and context. Remove redundancy. "
            "Return ONLY the compressed summary, nothing else.\n\n"
            f"Summary to compress:\n{summary}"
        )
        try:
            model = self.config.get("compression_model") or self._get_extraction_model(
                "claude-haiku-4-5-20251001")
            result = await provider.complete(
                model=model, max_tokens=200,
                messages=[{"role": "user", "content": prompt}])
            if hasattr(result, "usage") and result.usage:
                self.track_internal_cost(model, result.usage)
            compressed = result.content[0].text.strip()
            if 20 < len(compressed) < 600:
                return compressed
        except Exception as e:
            logger.debug("Summary compression failed: %s", e)
        return summary[-500:]

    async def _update_session_summary_async(self, user_id: str, summary_update: str):
        """Async version of _update_session_summary — uses LLM compression."""
        existing = self._get_session_summary(user_id) or ""
        new_summary = (existing + " " + summary_update).strip()
        if len(new_summary) > 600:
            new_summary = await self._compress_summary(new_summary)
        if len(new_summary) > 800:
            new_summary = new_summary[-800:]
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
    # EPISODIC MEMORY (Phase 2)
    # ══════════════════════════════════════════

    def _episodic_enabled(self) -> bool:
        return self.config.get("episodic_memory", False)

    def start_episode(self, user_id: str) -> str:
        """Start a new episode. Returns episode_id."""
        episode_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        self.db.execute(
            """INSERT INTO episodes (id, user_id, created_at, turn_count)
               VALUES (?, ?, ?, 0)""",
            (episode_id, user_id, now))
        self.db.commit()
        self._active_episodes[user_id] = episode_id
        logger.debug("Started episode %s for user %s", episode_id[:8], user_id)
        return episode_id

    def get_active_episode(self, user_id: str) -> str | None:
        """Get the active episode ID for a user, or None."""
        episode_id = self._active_episodes.get(user_id)
        if episode_id:
            # Verify it's still open
            row = self.db.execute(
                "SELECT id FROM episodes WHERE id = ? AND closed_at IS NULL",
                (episode_id,)).fetchone()
            if row:
                return episode_id
            self._active_episodes.pop(user_id, None)
        return None

    def ensure_episode(self, user_id: str) -> str:
        """Get active episode or start a new one."""
        episode_id = self.get_active_episode(user_id)
        if not episode_id:
            episode_id = self.start_episode(user_id)
        return episode_id

    def add_episode_turn(self, episode_id: str, user_input: str,
                          agent_response: str, tool_calls: list | None = None):
        """Add a turn to an episode."""
        now = datetime.now().isoformat()
        # Get current turn count
        row = self.db.execute(
            "SELECT turn_count FROM episodes WHERE id = ?",
            (episode_id,)).fetchone()
        if not row:
            logger.debug("Episode %s not found, skipping turn", episode_id[:8])
            return
        turn_index = row[0]

        tool_calls_json = json.dumps(tool_calls or [], ensure_ascii=False, default=str)
        self.db.execute(
            """INSERT INTO episode_turns (episode_id, turn_index, user_input,
               agent_response, tool_calls, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (episode_id, turn_index, user_input,
             agent_response[:2000],  # Cap response length for storage
             tool_calls_json, now))

        # Update turn count and aggregate tool sequence
        existing_tools = self.db.execute(
            "SELECT tool_sequence FROM episodes WHERE id = ?",
            (episode_id,)).fetchone()
        try:
            all_tools = json.loads(existing_tools[0]) if existing_tools and existing_tools[0] else []
        except (json.JSONDecodeError, TypeError):
            all_tools = []
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name", tc) if isinstance(tc, dict) else str(tc)
                if name not in all_tools:
                    all_tools.append(name)

        self.db.execute(
            """UPDATE episodes SET turn_count = turn_count + 1,
               tool_sequence = ? WHERE id = ?""",
            (json.dumps(all_tools, ensure_ascii=False), episode_id))
        self.db.commit()

    async def close_episode(self, episode_id: str, outcome: str = "completed"):
        """Close an episode, generate summary if enough turns."""
        row = self.db.execute(
            "SELECT user_id, turn_count, closed_at FROM episodes WHERE id = ?",
            (episode_id,)).fetchone()
        if not row or row[2]:  # Not found or already closed
            return
        user_id, turn_count = row[0], row[1]
        now = datetime.now().isoformat()

        summary = None
        if turn_count >= 2 and self._get_extraction_provider():
            summary = await self._generate_episode_summary(episode_id)
        if not summary:
            # Fallback: use first user message as summary
            first = self.db.execute(
                "SELECT user_input FROM episode_turns WHERE episode_id = ? ORDER BY turn_index LIMIT 1",
                (episode_id,)).fetchone()
            if first and first[0]:
                summary = first[0][:200]

        embedding = self._embed(summary) if summary else None

        self.db.execute(
            """UPDATE episodes SET closed_at = ?, outcome = ?, summary = ?,
               embedding = ? WHERE id = ?""",
            (now, outcome, summary, embedding, episode_id))
        self.db.commit()
        self._active_episodes.pop(user_id, None)
        logger.debug("Closed episode %s (%d turns, outcome=%s)",
                      episode_id[:8], turn_count, outcome)

    async def _generate_episode_summary(self, episode_id: str) -> str | None:
        """Generate a summary of an episode using LLM."""
        turns = self.db.execute(
            """SELECT user_input, agent_response FROM episode_turns
               WHERE episode_id = ? ORDER BY turn_index LIMIT 10""",
            (episode_id,)).fetchall()
        if not turns:
            return None

        dialog = "\n".join(
            f"User: {u[:200]}\nAssistant: {a[:200]}" for u, a in turns)

        prompt = (
            "Summarize this conversation in 1-2 sentences. "
            "Focus on: what the user wanted, what was done, and the outcome. "
            "Return ONLY the summary.\n\n" + dialog
        )
        try:
            provider = self._get_extraction_provider()
            if not provider:
                return turns[0][0][:200] if turns else None
            model = self.config.get("compression_model") or self._get_extraction_model(
                "claude-haiku-4-5-20251001")
            result = await provider.complete(
                model=model, max_tokens=150,
                messages=[{"role": "user", "content": prompt}])
            if hasattr(result, "usage") and result.usage:
                self.track_internal_cost(model, result.usage)
            summary = result.content[0].text.strip()
            if 10 < len(summary) < 500:
                return summary
        except Exception as e:
            logger.debug("Episode summary generation failed: %s", e)
        # Fallback: first user message as title
        return turns[0][0][:200] if turns else None

    def detect_topic_shift(self, user_id: str, new_message: str,
                            threshold: float | None = None) -> bool:
        """Detect if the new message represents a topic shift from current episode.

        Compares embedding of new message with average of last N turns.
        Returns True if similarity drops below threshold.
        """
        if threshold is None:
            threshold = self.config.get("episode_topic_shift_threshold", 0.3)

        episode_id = self.get_active_episode(user_id)
        if not episode_id:
            return False  # No active episode — not a shift, just start new

        if self._embedder is None:
            return False  # Can't detect without embedder

        # Get recent turns
        rows = self.db.execute(
            """SELECT user_input FROM episode_turns
               WHERE episode_id = ? ORDER BY turn_index DESC LIMIT 3""",
            (episode_id,)).fetchall()
        if not rows:
            return False

        # Combine recent turn texts
        recent_text = " ".join(r[0][:200] for r in rows if r[0])
        if not recent_text:
            return False

        try:
            new_emb = self._embedder.encode(new_message)
            recent_emb = self._embedder.encode(recent_text)
            similarity = self._cosine_similarity(new_emb, recent_emb)
            return similarity < threshold
        except Exception:
            return False

    def recall_episodes(self, query: str, user_id: str, top_k: int = 3) -> list[dict]:
        """Recall relevant past episodes using hybrid search over summaries."""
        # BM25 search over episode summaries via FTS5 (if available)
        # For simplicity, do brute-force over summaries with embeddings + keywords
        rows = self.db.execute(
            """SELECT id, title, summary, outcome, tool_sequence, topics,
                      turn_count, created_at, closed_at, embedding
               FROM episodes
               WHERE user_id = ? AND closed_at IS NOT NULL AND summary IS NOT NULL
               ORDER BY created_at DESC LIMIT 50""",
            (user_id,)).fetchall()
        if not rows:
            return []

        query_embedding = None
        if self._embedder is not None:
            try:
                query_embedding = self._embedder.encode(query)
            except Exception:
                pass

        query_words = set(query.lower().split())
        scored = []

        for (ep_id, title, summary, outcome, tool_seq, topics,
             turn_count, created_at, closed_at, emb_blob) in rows:

            # Semantic score
            semantic_score = 0.0
            if query_embedding is not None and emb_blob:
                try:
                    ep_emb = pickle.loads(emb_blob)
                    semantic_score = self._cosine_similarity(query_embedding, ep_emb)
                except Exception:
                    pass

            # Keyword score on summary
            summary_words = set((summary or "").lower().split())
            overlap = len(query_words & summary_words)
            keyword_score = overlap / max(len(query_words), 1)

            if query_embedding is not None and emb_blob:
                relevance = semantic_score * 0.7 + keyword_score * 0.3
            else:
                relevance = keyword_score

            if relevance < 0.05:
                continue

            # Recency bonus
            recency = self._recency_score(created_at)
            score = relevance * 0.7 + recency * 0.3

            try:
                tools = json.loads(tool_seq) if tool_seq else []
            except (json.JSONDecodeError, TypeError):
                tools = []

            scored.append({
                "id": ep_id,
                "title": title,
                "summary": summary,
                "outcome": outcome,
                "tools": tools,
                "turn_count": turn_count,
                "created_at": created_at,
                "score": score,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_episode_turns(self, episode_id: str, limit: int = 20) -> list[dict]:
        """Get turns for an episode."""
        rows = self.db.execute(
            """SELECT turn_index, user_input, agent_response, tool_calls, created_at
               FROM episode_turns WHERE episode_id = ?
               ORDER BY turn_index LIMIT ?""",
            (episode_id, limit)).fetchall()
        return [{"turn_index": r[0], "user_input": r[1], "agent_response": r[2],
                 "tool_calls": r[3], "created_at": r[4]} for r in rows]

    def get_episode(self, episode_id: str) -> dict | None:
        """Get a single episode by ID."""
        row = self.db.execute(
            """SELECT id, user_id, title, summary, outcome, tool_sequence,
                      topics, turn_count, created_at, closed_at
               FROM episodes WHERE id = ?""",
            (episode_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "user_id": row[1], "title": row[2],
            "summary": row[3], "outcome": row[4],
            "tools": json.loads(row[5]) if row[5] else [],
            "topics": json.loads(row[6]) if row[6] else [],
            "turn_count": row[7], "created_at": row[8], "closed_at": row[9],
        }

    def get_recent_episodes(self, user_id: str, limit: int = 10) -> list[dict]:
        """Get recent episodes for a user."""
        rows = self.db.execute(
            """SELECT id, title, summary, outcome, turn_count, created_at, closed_at
               FROM episodes WHERE user_id = ?
               ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit)).fetchall()
        return [{"id": r[0], "title": r[1], "summary": r[2], "outcome": r[3],
                 "turn_count": r[4], "created_at": r[5], "closed_at": r[6]}
                for r in rows]

    def prune_episode_turns(self, days: int = 7):
        """Remove individual turns from old closed episodes, keep summaries."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cur = self.db.execute(
            """DELETE FROM episode_turns WHERE episode_id IN (
                SELECT id FROM episodes WHERE closed_at IS NOT NULL
                AND closed_at < ?)""",
            (cutoff,))
        if cur.rowcount:
            self.db.commit()
            logger.debug("Pruned %d old episode turns (older than %d days)",
                          cur.rowcount, days)

    # ══════════════════════════════════════════
    # PHASE 3: GRAPH MEMORY
    # ══════════════════════════════════════════

    def _graph_enabled(self) -> bool:
        return bool(self.config.get("graph_memory", False))

    def upsert_entity(self, name: str, entity_type: str, user_id: str,
                      properties: dict | None = None) -> str:
        """Create or update an entity node. Returns entity ID."""
        name_norm = name.strip()
        if not name_norm:
            return ""
        # Check for existing entity (case-insensitive)
        row = self.db.execute(
            "SELECT id, mention_count, properties FROM memory_entities "
            "WHERE user_id = ? AND LOWER(name) = LOWER(?)",
            (user_id, name_norm)).fetchone()
        now = datetime.now().isoformat()
        if row:
            eid, count, old_props_json = row
            # Merge properties
            old_props = json.loads(old_props_json) if old_props_json else {}
            if properties:
                old_props.update(properties)
            self.db.execute(
                """UPDATE memory_entities SET mention_count = ?, last_seen = ?,
                   properties = ? WHERE id = ?""",
                (count + 1, now, json.dumps(old_props, ensure_ascii=False), eid))
            self.db.commit()
            return eid
        else:
            eid = str(uuid.uuid4())
            embedding = self._embed(f"{name_norm} ({entity_type})")
            self.db.execute(
                """INSERT INTO memory_entities
                   (id, name, entity_type, user_id, properties, embedding,
                    first_seen, last_seen, mention_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (eid, name_norm, entity_type, user_id,
                 json.dumps(properties or {}, ensure_ascii=False),
                 embedding, now, now))
            self.db.commit()
            return eid

    def upsert_relation(self, source_name: str, target_name: str,
                        relation_type: str, user_id: str,
                        evidence: str = "") -> str | None:
        """Create or strengthen a relation between two entities. Returns relation ID."""
        # Find source and target entities
        src = self.db.execute(
            "SELECT id FROM memory_entities WHERE user_id = ? AND LOWER(name) = LOWER(?)",
            (user_id, source_name.strip())).fetchone()
        tgt = self.db.execute(
            "SELECT id FROM memory_entities WHERE user_id = ? AND LOWER(name) = LOWER(?)",
            (user_id, target_name.strip())).fetchone()
        if not src or not tgt:
            return None
        source_id, target_id = src[0], tgt[0]
        now = datetime.now().isoformat()
        # Check for existing relation
        row = self.db.execute(
            """SELECT id, weight, evidence FROM memory_relations
               WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
            (source_id, target_id, relation_type)).fetchone()
        if row:
            rid, weight, old_evidence = row
            new_evidence = old_evidence
            if evidence and evidence not in (old_evidence or ""):
                new_evidence = ((old_evidence + "; ") if old_evidence else "") + evidence
            self.db.execute(
                "UPDATE memory_relations SET weight = ?, evidence = ?, updated_at = ? WHERE id = ?",
                (weight + 0.5, new_evidence, now, rid))
            self.db.commit()
            return rid
        else:
            rid = str(uuid.uuid4())
            self.db.execute(
                """INSERT INTO memory_relations
                   (id, source_id, target_id, relation_type, user_id, weight,
                    evidence, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, ?)""",
                (rid, source_id, target_id, relation_type, user_id,
                 evidence, now, now))
            self.db.commit()
            return rid

    def link_entity_to_memory(self, entity_id: str, memory_id: int):
        """Link an entity to a memory record."""
        try:
            self.db.execute(
                "INSERT OR IGNORE INTO memory_entity_mentions (entity_id, memory_id) VALUES (?, ?)",
                (entity_id, memory_id))
            self.db.commit()
        except Exception:
            pass

    @staticmethod
    def _entity_matches_text(entity_name: str, text: str) -> bool:
        """True if entity name appears in text (word-aware where possible)."""
        import re

        name = (entity_name or "").strip()
        body = (text or "").strip()
        if not name or not body:
            return False
        # Try strict-ish word boundary first; fallback to substring for mixed scripts.
        pat = r"\b" + re.escape(name) + r"\b"
        if re.search(pat, body, flags=re.IGNORECASE):
            return True
        return name.lower() in body.lower()

    def _link_memory_to_entities(self, memory_id: int, memory_text: str,
                                 entity_map: dict[str, str]) -> int:
        """Link a memory to entities whose names appear in the memory text."""
        linked = 0
        seen = set()
        for name, eid in (entity_map or {}).items():
            if eid in seen:
                continue
            if self._entity_matches_text(name, memory_text):
                self.link_entity_to_memory(eid, memory_id)
                seen.add(eid)
                linked += 1
        return linked

    def backfill_entity_mentions(self, user_id: str, memory_limit: int = 2000,
                                 entity_limit: int = 1000) -> dict[str, int]:
        """Backfill entity->memory links for existing memories by name matching."""
        entities = self.db.execute(
            """SELECT id, name FROM memory_entities
               WHERE user_id = ? ORDER BY mention_count DESC LIMIT ?""",
            (user_id, entity_limit),
        ).fetchall()
        if not entities:
            return {"entities": 0, "memories": 0, "links_added": 0}

        memories = self.db.execute(
            """SELECT id, content FROM memories
               WHERE user_id = ? AND archived_at IS NULL
               ORDER BY id DESC LIMIT ?""",
            (user_id, memory_limit),
        ).fetchall()
        links_added = 0
        for mem_id, content in memories:
            text = content or ""
            if not text:
                continue
            for eid, name in entities:
                if not self._entity_matches_text(name, text):
                    continue
                before = self.db.total_changes
                self.db.execute(
                    "INSERT OR IGNORE INTO memory_entity_mentions (entity_id, memory_id) VALUES (?, ?)",
                    (eid, mem_id),
                )
                if self.db.total_changes > before:
                    links_added += 1
        self.db.commit()
        return {
            "entities": len(entities),
            "memories": len(memories),
            "links_added": links_added,
        }

    def get_entity(self, entity_id: str) -> dict | None:
        """Get a single entity by ID."""
        row = self.db.execute(
            """SELECT id, name, entity_type, user_id, properties,
                      first_seen, last_seen, mention_count
               FROM memory_entities WHERE id = ?""",
            (entity_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "name": row[1], "entity_type": row[2],
            "user_id": row[3], "properties": json.loads(row[4] or "{}"),
            "first_seen": row[5], "last_seen": row[6], "mention_count": row[7],
        }

    def find_entity(self, name: str, user_id: str) -> dict | None:
        """Find entity by name (case-insensitive)."""
        row = self.db.execute(
            """SELECT id, name, entity_type, user_id, properties,
                      first_seen, last_seen, mention_count
               FROM memory_entities WHERE user_id = ? AND LOWER(name) = LOWER(?)""",
            (user_id, name.strip())).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "name": row[1], "entity_type": row[2],
            "user_id": row[3], "properties": json.loads(row[4] or "{}"),
            "first_seen": row[5], "last_seen": row[6], "mention_count": row[7],
        }

    def get_entities(self, user_id: str, entity_type: str | None = None,
                     limit: int = 50) -> list[dict]:
        """List entities for a user, optionally filtered by type."""
        if entity_type:
            rows = self.db.execute(
                """SELECT id, name, entity_type, properties, mention_count,
                          first_seen, last_seen
                   FROM memory_entities WHERE user_id = ? AND entity_type = ?
                   ORDER BY mention_count DESC LIMIT ?""",
                (user_id, entity_type, limit)).fetchall()
        else:
            rows = self.db.execute(
                """SELECT id, name, entity_type, properties, mention_count,
                          first_seen, last_seen
                   FROM memory_entities WHERE user_id = ?
                   ORDER BY mention_count DESC LIMIT ?""",
                (user_id, limit)).fetchall()
        return [{"id": r[0], "name": r[1], "entity_type": r[2],
                 "properties": json.loads(r[3] or "{}"),
                 "mention_count": r[4], "first_seen": r[5], "last_seen": r[6]}
                for r in rows]

    def get_entity_relations(self, entity_id: str) -> list[dict]:
        """Get all relations involving an entity (as source or target)."""
        rows = self.db.execute(
            """SELECT r.id, r.relation_type, r.weight, r.evidence,
                      s.name as source_name, s.entity_type as source_type,
                      t.name as target_name, t.entity_type as target_type
               FROM memory_relations r
               JOIN memory_entities s ON r.source_id = s.id
               JOIN memory_entities t ON r.target_id = t.id
               WHERE r.source_id = ? OR r.target_id = ?
               ORDER BY r.weight DESC""",
            (entity_id, entity_id)).fetchall()
        return [{"id": r[0], "relation_type": r[1], "weight": r[2],
                 "evidence": r[3], "source_name": r[4], "source_type": r[5],
                 "target_name": r[6], "target_type": r[7]}
                for r in rows]

    def get_entity_neighborhood(self, entity_name: str, user_id: str,
                                hops: int = 2) -> dict:
        """BFS traversal from an entity: returns entities + relations within N hops."""
        root = self.find_entity(entity_name, user_id)
        if not root:
            return {"entities": [], "relations": []}
        visited = {root["id"]}
        entities = [root]
        all_relations = []
        frontier = {root["id"]}
        for _ in range(hops):
            if not frontier:
                break
            next_frontier = set()
            placeholders = ",".join("?" * len(frontier))
            rows = self.db.execute(
                f"""SELECT r.id, r.source_id, r.target_id, r.relation_type, r.weight, r.evidence,
                           s.name, s.entity_type, t.name, t.entity_type
                    FROM memory_relations r
                    JOIN memory_entities s ON r.source_id = s.id
                    JOIN memory_entities t ON r.target_id = t.id
                    WHERE r.user_id = ? AND (r.source_id IN ({placeholders})
                       OR r.target_id IN ({placeholders}))""",
                (user_id, *frontier, *frontier)).fetchall()
            for r in rows:
                rel = {"id": r[0], "source_id": r[1], "target_id": r[2],
                       "relation_type": r[3], "weight": r[4], "evidence": r[5],
                       "source_name": r[6], "source_type": r[7],
                       "target_name": r[8], "target_type": r[9]}
                all_relations.append(rel)
                for neighbor_id in (r[1], r[2]):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_frontier.add(neighbor_id)
                        ent = self.get_entity(neighbor_id)
                        if ent:
                            entities.append(ent)
            frontier = next_frontier
        # Deduplicate relations
        seen_rels = set()
        unique_rels = []
        for r in all_relations:
            if r["id"] not in seen_rels:
                seen_rels.add(r["id"])
                unique_rels.append(r)
        return {"entities": entities, "relations": unique_rels}

    def entity_search(self, query: str, user_id: str, top_k: int = 10) -> list[dict]:
        """Search entities by name (substring) and embedding similarity."""
        results = []
        # Substring match on name
        rows = self.db.execute(
            """SELECT id, name, entity_type, mention_count
               FROM memory_entities WHERE user_id = ?
               AND LOWER(name) LIKE ?
               ORDER BY mention_count DESC LIMIT ?""",
            (user_id, f"%{query.lower()}%", top_k)).fetchall()
        seen = set()
        for r in rows:
            results.append({"id": r[0], "name": r[1], "entity_type": r[2],
                            "mention_count": r[3], "score": 1.0})
            seen.add(r[0])
        # Embedding search if available
        if self._embedder and len(results) < top_k:
            query_emb = self._embedder.encode(query)
            emb_rows = self.db.execute(
                """SELECT id, name, entity_type, mention_count, embedding
                   FROM memory_entities WHERE user_id = ?
                   AND embedding IS NOT NULL LIMIT 200""",
                (user_id,)).fetchall()
            scored = []
            for r in emb_rows:
                if r[0] in seen:
                    continue
                try:
                    emb = pickle.loads(r[4])
                    sim = self._cosine_sim(query_emb, emb)
                    if sim > 0.3:
                        scored.append({"id": r[0], "name": r[1], "entity_type": r[2],
                                       "mention_count": r[3], "score": float(sim)})
                except Exception:
                    continue
            scored.sort(key=lambda x: x["score"], reverse=True)
            results.extend(scored[:top_k - len(results)])
        return results[:top_k]

    def _graph_recall(self, query: str, user_id: str, top_k: int = 10) -> list[dict]:
        """Search memories through entity graph: find entities in query, then
        return memories linked to those entities and their neighbors."""
        if not self._graph_enabled():
            return []
        # Find entities mentioned in query
        matched_entities = self.entity_search(query, user_id, top_k=5)
        if not matched_entities:
            return []
        # Gather entity IDs (including 1-hop neighbors for high-score matches)
        entity_ids = set()
        for ent in matched_entities:
            entity_ids.add(ent["id"])
            if ent.get("score", 0) > 0.7:
                # Add 1-hop neighbors for strong matches
                rels = self.get_entity_relations(ent["id"])
                for rel in rels[:5]:  # Limit neighbor expansion
                    entity_ids.add(rel.get("source_name", ""))
                    entity_ids.add(rel.get("target_name", ""))
        # Get memory IDs linked to these entities
        if not entity_ids:
            return []
        entity_id_list = [eid for eid in entity_ids if len(eid) == 36]  # UUID only
        if not entity_id_list:
            return []
        placeholders = ",".join("?" * len(entity_id_list))
        rows = self.db.execute(
            f"""SELECT DISTINCT m.id, m.content, m.importance
                FROM memories m
                JOIN memory_entity_mentions em ON em.memory_id = m.id
                WHERE em.entity_id IN ({placeholders})
                AND m.user_id = ? AND m.archived_at IS NULL
                ORDER BY m.importance DESC LIMIT ?""",
            (*entity_id_list, user_id, top_k)).fetchall()
        return [{"id": r[0], "content": r[1], "score": float(r[2] or 0.5)}
                for r in rows]

    def merge_entities(self, entity_ids: list[str], primary_name: str | None = None):
        """Merge duplicate entities into one. First entity is kept as primary."""
        if len(entity_ids) < 2:
            return
        primary_id = entity_ids[0]
        if primary_name:
            self.db.execute("UPDATE memory_entities SET name = ? WHERE id = ?",
                            (primary_name, primary_id))
        # Sum mention_counts
        total_mentions = 0
        for eid in entity_ids:
            row = self.db.execute(
                "SELECT mention_count FROM memory_entities WHERE id = ?",
                (eid,)).fetchone()
            if row:
                total_mentions += row[0]
        self.db.execute("UPDATE memory_entities SET mention_count = ? WHERE id = ?",
                        (total_mentions, primary_id))
        # Redirect relations and mentions from secondary entities
        for eid in entity_ids[1:]:
            self.db.execute(
                "UPDATE memory_relations SET source_id = ? WHERE source_id = ?",
                (primary_id, eid))
            self.db.execute(
                "UPDATE memory_relations SET target_id = ? WHERE target_id = ?",
                (primary_id, eid))
            self.db.execute(
                "UPDATE OR IGNORE memory_entity_mentions SET entity_id = ? WHERE entity_id = ?",
                (primary_id, eid))
            self.db.execute("DELETE FROM memory_entity_mentions WHERE entity_id = ?", (eid,))
            self.db.execute("DELETE FROM memory_entities WHERE id = ?", (eid,))
        # Remove self-referencing relations
        self.db.execute(
            "DELETE FROM memory_relations WHERE source_id = target_id")
        self.db.commit()

    # ══════════════════════════════════════════
    # PHASE 4: PROCEDURAL MEMORY
    # ══════════════════════════════════════════

    def _procedural_enabled(self) -> bool:
        feat = self._features_config.get("procedural_memory", {})
        if isinstance(feat, dict):
            return bool(feat.get("enabled", False))
        return bool(feat)

    def save_procedure(self, name: str, description: str, steps: list[dict],
                       user_id: str, trigger_patterns: list[str] | None = None,
                       preconditions: str = "") -> str:
        """Save or update a procedure (learned workflow)."""
        name_norm = name.strip()
        if not name_norm:
            return ""
        # Check for existing procedure with same name
        row = self.db.execute(
            "SELECT id FROM procedures WHERE user_id = ? AND LOWER(name) = LOWER(?)",
            (user_id, name_norm)).fetchone()
        now = datetime.now().isoformat()
        if row:
            pid = row[0]
            embedding = self._embed(f"{name_norm}: {description}")
            self.db.execute(
                """UPDATE procedures SET description = ?, steps = ?,
                   trigger_patterns = ?, preconditions = ?, embedding = ?,
                   last_used = ? WHERE id = ?""",
                (description, json.dumps(steps, ensure_ascii=False),
                 json.dumps(trigger_patterns or [], ensure_ascii=False),
                 preconditions, embedding, now, pid))
            self.db.commit()
            return pid
        else:
            pid = str(uuid.uuid4())
            embedding = self._embed(f"{name_norm}: {description}")
            self.db.execute(
                """INSERT INTO procedures
                   (id, name, description, trigger_patterns, steps, preconditions,
                    success_rate, use_count, user_id, embedding, created_at, last_used)
                   VALUES (?, ?, ?, ?, ?, ?, 1.0, 0, ?, ?, ?, ?)""",
                (pid, name_norm, description,
                 json.dumps(trigger_patterns or [], ensure_ascii=False),
                 json.dumps(steps, ensure_ascii=False),
                 preconditions, user_id, embedding, now, now))
            self.db.commit()
            return pid

    def get_procedure(self, procedure_id: str) -> dict | None:
        """Get a single procedure by ID."""
        row = self.db.execute(
            """SELECT id, name, description, trigger_patterns, steps, preconditions,
                      success_rate, use_count, user_id, created_at, last_used
               FROM procedures WHERE id = ?""",
            (procedure_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row[0], "name": row[1], "description": row[2],
            "trigger_patterns": json.loads(row[3] or "[]"),
            "steps": json.loads(row[4] or "[]"),
            "preconditions": row[5], "success_rate": row[6],
            "use_count": row[7], "user_id": row[8],
            "created_at": row[9], "last_used": row[10],
        }

    def get_procedures(self, user_id: str, limit: int = 20) -> list[dict]:
        """List procedures for a user, ordered by use_count."""
        rows = self.db.execute(
            """SELECT id, name, description, success_rate, use_count,
                      trigger_patterns, created_at, last_used
               FROM procedures WHERE user_id = ?
               ORDER BY use_count DESC LIMIT ?""",
            (user_id, limit)).fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2],
                 "success_rate": r[3], "use_count": r[4],
                 "trigger_patterns": json.loads(r[5] or "[]"),
                 "created_at": r[6], "last_used": r[7]}
                for r in rows]

    def recall_procedures(self, query: str, user_id: str,
                          top_k: int = 3) -> list[dict]:
        """Find relevant procedures by query (keyword + embedding)."""
        if not self._procedural_enabled():
            return []
        results = []
        seen = set()
        # Keyword match on name/description
        rows = self.db.execute(
            """SELECT id, name, description, steps, trigger_patterns,
                      success_rate, use_count
               FROM procedures WHERE user_id = ?
               AND (LOWER(name) LIKE ? OR LOWER(description) LIKE ?)
               ORDER BY use_count DESC LIMIT ?""",
            (user_id, f"%{query.lower()}%", f"%{query.lower()}%", top_k)).fetchall()
        for r in rows:
            results.append({
                "id": r[0], "name": r[1], "description": r[2],
                "steps": json.loads(r[3] or "[]"),
                "trigger_patterns": json.loads(r[4] or "[]"),
                "success_rate": r[5], "use_count": r[6], "score": 1.0,
            })
            seen.add(r[0])
        # Trigger pattern match
        all_procs = self.db.execute(
            """SELECT id, name, description, steps, trigger_patterns,
                      success_rate, use_count
               FROM procedures WHERE user_id = ?""",
            (user_id,)).fetchall()
        for r in all_procs:
            if r[0] in seen:
                continue
            triggers = json.loads(r[4] or "[]")
            q_lower = query.lower()
            if any(t.lower() in q_lower or q_lower in t.lower() for t in triggers):
                results.append({
                    "id": r[0], "name": r[1], "description": r[2],
                    "steps": json.loads(r[3] or "[]"),
                    "trigger_patterns": triggers,
                    "success_rate": r[5], "use_count": r[6], "score": 0.8,
                })
                seen.add(r[0])
        # Embedding search
        if self._embedder and len(results) < top_k:
            query_emb = self._embedder.encode(query)
            emb_rows = self.db.execute(
                """SELECT id, name, description, steps, trigger_patterns,
                          success_rate, use_count, embedding
                   FROM procedures WHERE user_id = ?
                   AND embedding IS NOT NULL""",
                (user_id,)).fetchall()
            scored = []
            for r in emb_rows:
                if r[0] in seen:
                    continue
                try:
                    emb = pickle.loads(r[7])
                    sim = self._cosine_sim(query_emb, emb)
                    if sim > 0.3:
                        scored.append({
                            "id": r[0], "name": r[1], "description": r[2],
                            "steps": json.loads(r[3] or "[]"),
                            "trigger_patterns": json.loads(r[4] or "[]"),
                            "success_rate": r[5], "use_count": r[6],
                            "score": float(sim),
                        })
                except Exception:
                    continue
            scored.sort(key=lambda x: x["score"], reverse=True)
            results.extend(scored[:top_k - len(results)])
        return results[:top_k]

    def record_procedure_use(self, procedure_id: str, success: bool = True):
        """Update use_count and success_rate after a procedure is used."""
        row = self.db.execute(
            "SELECT use_count, success_rate FROM procedures WHERE id = ?",
            (procedure_id,)).fetchone()
        if not row:
            return
        use_count, old_rate = row
        new_count = use_count + 1
        # Exponential moving average for success_rate
        alpha = 0.3
        new_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * old_rate
        self.db.execute(
            """UPDATE procedures SET use_count = ?, success_rate = ?,
               last_used = ? WHERE id = ?""",
            (new_count, new_rate, datetime.now().isoformat(), procedure_id))
        self.db.commit()

    def delete_procedure(self, procedure_id: str):
        """Delete a procedure by ID."""
        self.db.execute("DELETE FROM procedures WHERE id = ?", (procedure_id,))
        self.db.commit()

    async def crystallize_procedures(self, user_id: str, min_occurrences: int = 2):
        """Analyze closed episodes to find repeating tool patterns and create procedures.

        Scans episodes with similar tool sequences that appear >= min_occurrences times.
        Uses LLM to generate a generalized procedure from the pattern.
        """
        if not self._procedural_enabled() or not self._episodic_enabled():
            return []
        # Get closed episodes with tools
        rows = self.db.execute(
            """SELECT id, summary, tool_sequence FROM episodes
               WHERE user_id = ? AND closed_at IS NOT NULL
               AND tool_sequence != '[]' AND summary IS NOT NULL
               ORDER BY closed_at DESC LIMIT 50""",
            (user_id,)).fetchall()
        if len(rows) < min_occurrences:
            return []
        # Group by normalized tool sequence
        seq_groups: dict[str, list] = {}
        for ep_id, summary, tool_seq_json in rows:
            tools = json.loads(tool_seq_json or "[]")
            if not tools:
                continue
            # Normalize: just tool names in order
            key = ",".join(sorted(set(tools)))
            if key not in seq_groups:
                seq_groups[key] = []
            seq_groups[key].append({"id": ep_id, "summary": summary, "tools": tools})
        # Find patterns that appear >= min_occurrences
        created = []
        for tool_key, episodes in seq_groups.items():
            if len(episodes) < min_occurrences:
                continue
            tool_names = tool_key.split(",")
            # Check if procedure already exists for this tool combo
            existing = self.db.execute(
                "SELECT id FROM procedures WHERE user_id = ? AND name LIKE ?",
                (user_id, f"%{'_'.join(tool_names[:3])}%")).fetchone()
            if existing:
                continue
            # Generate procedure name and description
            summaries = [ep["summary"] for ep in episodes[:3]]
            name = f"workflow_{'_'.join(tool_names[:3])}"
            description = f"Common pattern using {', '.join(tool_names)}: " + "; ".join(summaries[:2])
            if len(description) > 300:
                description = description[:300]
            steps = [{"tool": t, "reasoning": f"Use {t}"} for t in tool_names]
            triggers = [s[:50] for s in summaries[:3]]
            pid = self.save_procedure(
                name=name, description=description, steps=steps,
                user_id=user_id, trigger_patterns=triggers)
            if pid:
                created.append(pid)
                logger.debug("Crystallized procedure '%s' from %d episodes",
                             name, len(episodes))
        return created

    # ══════════════════════════════════════════
    # USAGE TRACKING
    # ══════════════════════════════════════════

    def track_usage(self, user_id: str, model: str, usage, cost_usd: float = 0):
        """Log token usage for cost tracking."""
        user_id = self.get_canonical_person_id(user_id)
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

    def track_internal_cost(self, model: str, usage, user_id: str = "system") -> float:
        """Calculate and track cost for internal LLM calls (extraction, compression, vision).
        Returns the cost in USD."""
        try:
            from .providers import get_pricing
            pricing = get_pricing(model)
            cost = (
                getattr(usage, 'input_tokens', 0) * pricing["input"] / 1_000_000
                + getattr(usage, 'output_tokens', 0) * pricing["output"] / 1_000_000
                + getattr(usage, 'cache_read_input_tokens', 0) * pricing["cache_read"] / 1_000_000
            )
            self.track_usage(user_id, model, usage, cost)
            return cost
        except Exception:
            return 0.0

    def get_today_cost(self) -> float:
        """Get total cost for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM usage_stats WHERE timestamp LIKE ?",
            (f"{today}%",)).fetchone()
        return row[0] if row else 0.0

    def get_hour_cost(self) -> dict:
        """Get cost and calls for the last hour."""
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM usage_stats "
            "WHERE timestamp >= datetime('now', '-1 hour')").fetchone()
        return {"cost": row[0] if row else 0.0, "calls": row[1] if row else 0}

    def get_today_stats(self) -> dict:
        """Get cost and calls for today."""
        today = datetime.now().strftime("%Y-%m-%d")
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM usage_stats "
            "WHERE timestamp LIKE ?", (f"{today}%",)).fetchone()
        return {"cost": row[0] if row else 0.0, "calls": row[1] if row else 0}

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
        self._fts_delete(memory_id)
        cur = self.db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.db.commit()
        return cur.rowcount > 0

    def get_all_memories(self, user_id: str = None) -> list[dict]:
        """Get all memories, optionally filtered by user_id."""
        if user_id:
            user_id = self.get_canonical_person_id(user_id)
            rows = self.db.execute(
                "SELECT id, user_id, content, type, importance, created_at, file_meta FROM memories WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT id, user_id, content, type, importance, created_at, file_meta FROM memories ORDER BY created_at DESC"
            ).fetchall()
        return [{"id": r[0], "user_id": r[1], "content": r[2], "type": r[3],
                 "importance": r[4], "created_at": r[5], "file_meta": r[6]} for r in rows]

    def get_memory_count(self, user_id: str = None) -> int:
        """Count memories, optionally filtered by user_id."""
        if user_id:
            user_id = self.get_canonical_person_id(user_id)
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

    def get_success_rate(self, hours: int = 24) -> float:
        """Success rate from interaction_log for last N hours."""
        row = self.db.execute(
            """SELECT COUNT(*) as total,
                      SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as ok
               FROM interaction_log
               WHERE created_at >= datetime('now', ?)""",
            (f"-{hours} hours",)).fetchone()
        if not row or row[0] == 0:
            return 100.0
        return round(row[1] / row[0] * 100, 1)

    def get_avg_confidence(self, hours: int = 24) -> float:
        """Average confidence from interaction_log for last N hours."""
        row = self.db.execute(
            """SELECT AVG(confidence) FROM interaction_log
               WHERE confidence IS NOT NULL
                 AND created_at >= datetime('now', ?)""",
            (f"-{hours} hours",)).fetchone()
        return round(row[0], 1) if row and row[0] is not None else 0.0

    def get_cache_efficiency(self) -> float:
        """Ratio of cache_read_tokens to total input_tokens (percentage)."""
        row = self.db.execute(
            """SELECT COALESCE(SUM(cache_read_tokens), 0),
                      COALESCE(SUM(input_tokens), 0)
               FROM usage_stats""").fetchone()
        if not row or row[1] == 0:
            return 0.0
        return round(row[0] / row[1] * 100, 1)

    def get_yesterday_stats(self) -> dict:
        """Get cost and calls for yesterday (for delta calculations)."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        row = self.db.execute(
            "SELECT COALESCE(SUM(cost_usd), 0), COUNT(*) FROM usage_stats "
            "WHERE timestamp LIKE ?", (f"{yesterday}%",)).fetchone()
        return {"cost": row[0] if row else 0.0, "calls": row[1] if row else 0}

    def get_model_distribution_today(self) -> list[dict]:
        """Model usage breakdown for today (for mini donut chart)."""
        today = datetime.now().strftime("%Y-%m-%d")
        rows = self.db.execute(
            """SELECT model, COUNT(*) as calls, COALESCE(SUM(cost_usd), 0) as cost
               FROM usage_stats WHERE timestamp LIKE ?
               GROUP BY model ORDER BY calls DESC""",
            (f"{today}%",)).fetchall()
        return [{"model": r[0], "calls": r[1], "cost_usd": round(r[2], 4)} for r in rows]

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
        """Score 0-1 based on how recent the memory is (linear decay)."""
        try:
            dt = datetime.fromisoformat(created_at)
            days_ago = (datetime.now() - dt).days
            return max(0, 1 - days_ago / 365)
        except (ValueError, TypeError):
            return 0.5

    @staticmethod
    def _temporal_decay_score(created_at: str, accessed_at: str | None = None,
                              decay_rate: float = 0.01) -> float:
        """Exponential temporal decay: score = exp(-decay_rate * days_since_access).

        Unlike linear recency:
        - Uses accessed_at (not just created_at), so re-accessed memories stay fresh
        - Exponential curve preserves recent memories while gracefully forgetting old ones
        - Default decay_rate=0.01 gives half-life of ~69 days
        """
        import math
        try:
            last_seen = datetime.fromisoformat(accessed_at or created_at)
            days_since = max(0, (datetime.now() - last_seen).total_seconds() / 86400)
            return math.exp(-decay_rate * days_since)
        except (ValueError, TypeError):
            return 0.5


    # ══════════════════════════════════════════
    # USER RATINGS / FEEDBACK
    # ══════════════════════════════════════════

    def rate_last_response(self, user_id: str, rating: int) -> bool:
        """Set rating (1-5) on the most recent interaction_log row for this user.

        Returns True if a row was updated.
        """
        rating = max(1, min(5, int(rating)))
        uid = self._normalize_user_id(user_id)
        row = self.db.execute(
            "SELECT id FROM interaction_log WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (uid,),
        ).fetchone()
        if not row:
            return False
        self.db.execute(
            "UPDATE interaction_log SET rating = ? WHERE id = ?",
            (rating, row[0]),
        )
        self.db.commit()
        return True

    def get_rating_stats(self, days: int = 30, user_id: str | None = None) -> dict:
        """Return rating statistics: avg_rating, rated_count, positive_rate."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        params: list = [cutoff]
        user_filter = ""
        if user_id:
            user_filter = "AND user_id = ?"
            params.append(self._normalize_user_id(user_id))

        row = self.db.execute(
            f"""SELECT COUNT(*), AVG(rating),
                       SUM(CASE WHEN rating >= 4 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN rating IS NOT NULL THEN 1 ELSE 0 END)
                FROM interaction_log
                WHERE created_at > ? {user_filter}""",
            params,
        ).fetchone()
        total = row[0] or 0
        avg = round(float(row[1] or 0), 2)
        positive = row[2] or 0
        rated = row[3] or 0
        return {
            "avg_rating": avg,
            "rated_count": rated,
            "total_interactions": total,
            "positive_rate": round(positive / max(rated, 1) * 100, 1),
            "coverage": round(rated / max(total, 1) * 100, 1),
        }

    # ══════════════════════════════════════════
    # TOOL ANALYTICS
    # ══════════════════════════════════════════

    def record_tool_calls(self, tool_calls: list, user_id: str) -> None:
        """Persist per-call analytics for every tool in the list."""
        if not tool_calls:
            return
        uid = self._normalize_user_id(user_id)
        now = datetime.now().isoformat()
        rows = [
            (tc.get("name", "unknown"), uid,
             int(tc.get("duration_ms") or 0),
             0 if tc.get("error") else 1,
             now)
            for tc in tool_calls
        ]
        self.db.executemany(
            "INSERT INTO tool_analytics (tool_name, user_id, duration_ms, success, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        self.db.commit()

    def get_tool_analytics(self, days: int = 30, user_id: str | None = None) -> list:
        """Return aggregated tool stats: calls, success_rate, avg_duration_ms, max_duration_ms."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        params: list = [cutoff]
        user_filter = ""
        if user_id:
            user_filter = "AND user_id = ?"
            params.append(self._normalize_user_id(user_id))

        rows = self.db.execute(
            f"""SELECT tool_name,
                       COUNT(*) AS calls,
                       ROUND(AVG(success) * 100, 1) AS success_rate,
                       ROUND(AVG(duration_ms), 0) AS avg_ms,
                       MAX(duration_ms) AS max_ms
                FROM tool_analytics
                WHERE created_at > ? {user_filter}
                GROUP BY tool_name
                ORDER BY calls DESC""",
            params,
        ).fetchall()
        return [
            {
                "tool_name": r[0],
                "calls": r[1],
                "success_rate": r[2],
                "avg_duration_ms": int(r[3] or 0),
                "max_duration_ms": int(r[4] or 0),
            }
            for r in rows
        ]

    def close(self):
        self._mx_daemon_running = False
        if self._mx_daemon_task and not self._mx_daemon_task.done():
            self._mx_daemon_task.cancel()
        self._mx_daemon_task = None
        self.db.close()
