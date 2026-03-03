"""Night Worker — background batch processor for KB enrichment.

Processes knowledge base chunks overnight using free local models (Ollama).
Tasks: contextual enrichment, embedding generation, entity extraction.

Storage: SQLite table night_tasks in the KB database.
Scheduling: via scheduler.py (cron: "0 22 * * *")
"""

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    CONTEXTUAL_ENRICHMENT = "contextual_enrichment"
    EMBEDDING_GENERATION = "embedding_generation"
    ENTITY_EXTRACTION = "entity_extraction"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class NightTask:
    id: str
    task_type: str
    target_id: str  # chunk_id
    status: str = "pending"
    error: str | None = None
    retry_count: int = 0
    created_at: str = ""
    completed_at: str | None = None


class NightWorker:
    """Background batch processor for knowledge base enrichment."""

    def __init__(self, config: dict, kb_db, provider=None, embedder=None):
        """
        Args:
            config: night_worker config section
            kb_db: sqlite3 connection to knowledge_base.db
            provider: LLM provider (OllamaProvider for free processing)
            embedder: Embedding provider
        """
        self._config = config
        self.db = kb_db
        self._provider = provider
        self._embedder = embedder
        self._batch_size = config.get("batch_size", 20)
        self._max_tasks = config.get("max_tasks_per_run", 200)
        self._max_runtime = config.get("max_runtime_sec", 3600)
        self._running = False
        self._init_schema()

    def _init_schema(self):
        """Create night_tasks table."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS night_tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                error TEXT,
                retry_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT
            )
        """)
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_nt_status ON night_tasks(status)")
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_nt_type ON night_tasks(task_type)")
        self.db.commit()

    # ═══════════════════════════════════════
    # QUEUE API
    # ═══════════════════════════════════════

    def enqueue(self, task_type: str, target_id: str) -> str:
        """Add a single task to the queue. Returns task ID."""
        # Skip if already pending for this target+type
        existing = self.db.execute(
            "SELECT id FROM night_tasks WHERE target_id = ? AND task_type = ? "
            "AND status IN ('pending', 'running')",
            (target_id, task_type)).fetchone()
        if existing:
            return existing[0] if isinstance(existing, tuple) else existing["id"]

        task_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO night_tasks (id, task_type, target_id) VALUES (?, ?, ?)",
            (task_id, task_type, target_id))
        self.db.commit()
        return task_id

    def enqueue_batch(self, task_type: str, target_ids: list[str]) -> int:
        """Add multiple tasks. Returns count of actually enqueued."""
        count = 0
        for tid in target_ids:
            existing = self.db.execute(
                "SELECT 1 FROM night_tasks WHERE target_id = ? AND task_type = ? "
                "AND status IN ('pending', 'running')",
                (tid, task_type)).fetchone()
            if not existing:
                self.db.execute(
                    "INSERT INTO night_tasks (id, task_type, target_id) VALUES (?, ?, ?)",
                    (str(uuid.uuid4()), task_type, tid))
                count += 1
        self.db.commit()
        return count

    def enqueue_unenriched(self) -> dict:
        """Find all chunks without context_prefix and enqueue them."""
        counts = {}

        # Contextual enrichment — chunks without context_prefix
        rows = self.db.execute(
            "SELECT id FROM kb_chunks WHERE (context_prefix IS NULL OR context_prefix = '') "
            "AND chunk_type != 'parent'"
        ).fetchall()
        chunk_ids = [r[0] if isinstance(r, tuple) else r["id"] for r in rows]
        if chunk_ids:
            counts["contextual_enrichment"] = self.enqueue_batch(
                TaskType.CONTEXTUAL_ENRICHMENT, chunk_ids)

        # Embedding generation — chunks without embedding
        rows = self.db.execute(
            "SELECT id FROM kb_chunks WHERE embedding IS NULL "
            "AND chunk_type != 'parent'"
        ).fetchall()
        chunk_ids = [r[0] if isinstance(r, tuple) else r["id"] for r in rows]
        if chunk_ids:
            counts["embedding_generation"] = self.enqueue_batch(
                TaskType.EMBEDDING_GENERATION, chunk_ids)

        # Entity extraction — chunks not yet processed (check night_tasks for done)
        rows = self.db.execute(
            "SELECT c.id FROM kb_chunks c "
            "WHERE c.chunk_type != 'parent' AND NOT EXISTS ("
            "  SELECT 1 FROM night_tasks t WHERE t.target_id = c.id "
            "  AND t.task_type = 'entity_extraction' AND t.status = 'done')"
        ).fetchall()
        chunk_ids = [r[0] if isinstance(r, tuple) else r["id"] for r in rows]
        if chunk_ids:
            counts["entity_extraction"] = self.enqueue_batch(
                TaskType.ENTITY_EXTRACTION, chunk_ids)

        return counts

    def get_queue_stats(self) -> dict:
        """Get task queue statistics."""
        stats = {"total": 0, "pending": 0, "running": 0, "done": 0, "failed": 0}
        rows = self.db.execute(
            "SELECT status, COUNT(*) as cnt FROM night_tasks GROUP BY status"
        ).fetchall()
        for row in rows:
            status = row[0] if isinstance(row, tuple) else row["status"]
            cnt = row[1] if isinstance(row, tuple) else row["cnt"]
            stats[status] = cnt
            stats["total"] += cnt

        # Per-type stats
        type_stats = {}
        rows = self.db.execute(
            "SELECT task_type, status, COUNT(*) as cnt FROM night_tasks "
            "GROUP BY task_type, status"
        ).fetchall()
        for row in rows:
            ttype = row[0] if isinstance(row, tuple) else row["task_type"]
            status = row[1] if isinstance(row, tuple) else row["status"]
            cnt = row[2] if isinstance(row, tuple) else row["cnt"]
            if ttype not in type_stats:
                type_stats[ttype] = {}
            type_stats[ttype][status] = cnt
        stats["by_type"] = type_stats

        return stats

    # ═══════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════

    async def run(self) -> dict:
        """Process all pending tasks. Returns summary."""
        if self._running:
            return {"status": "already_running"}

        self._running = True
        t0 = time.monotonic()
        results = {"processed": 0, "failed": 0, "skipped": 0, "by_type": {}}

        try:
            for task_type in [TaskType.CONTEXTUAL_ENRICHMENT,
                              TaskType.EMBEDDING_GENERATION,
                              TaskType.ENTITY_EXTRACTION]:
                type_result = await self._process_type(task_type.value, t0)
                results["by_type"][task_type.value] = type_result
                results["processed"] += type_result.get("processed", 0)
                results["failed"] += type_result.get("failed", 0)

                # Check runtime limit
                elapsed = time.monotonic() - t0
                if elapsed > self._max_runtime:
                    logger.info("Night worker: runtime limit reached (%.0fs)", elapsed)
                    break
        finally:
            self._running = False
            results["runtime_sec"] = round(time.monotonic() - t0, 1)
            results["status"] = "completed"

        logger.info("Night worker finished: %d processed, %d failed in %.1fs",
                     results["processed"], results["failed"], results["runtime_sec"])
        return results

    async def _process_type(self, task_type: str, start_time: float) -> dict:
        """Process all pending tasks of a given type."""
        result = {"processed": 0, "failed": 0}
        processed_total = 0

        while processed_total < self._max_tasks:
            # Check runtime
            if time.monotonic() - start_time > self._max_runtime:
                break

            # Fetch batch
            tasks = self.db.execute(
                "SELECT id, target_id, retry_count FROM night_tasks "
                "WHERE task_type = ? AND status = 'pending' "
                "ORDER BY created_at LIMIT ?",
                (task_type, self._batch_size)).fetchall()

            if not tasks:
                break

            for task_row in tasks:
                task_id = task_row[0] if isinstance(task_row, tuple) else task_row["id"]
                target_id = task_row[1] if isinstance(task_row, tuple) else task_row["target_id"]
                retry_count = task_row[2] if isinstance(task_row, tuple) else task_row["retry_count"]

                # Mark as running
                self.db.execute(
                    "UPDATE night_tasks SET status = 'running' WHERE id = ?",
                    (task_id,))
                self.db.commit()

                try:
                    if task_type == TaskType.CONTEXTUAL_ENRICHMENT:
                        await self._enrich_chunk(target_id)
                    elif task_type == TaskType.EMBEDDING_GENERATION:
                        await self._embed_chunk(target_id)
                    elif task_type == TaskType.ENTITY_EXTRACTION:
                        await self._extract_entities(target_id)

                    # Mark done
                    self.db.execute(
                        "UPDATE night_tasks SET status = 'done', "
                        "completed_at = datetime('now') WHERE id = ?",
                        (task_id,))
                    self.db.commit()
                    result["processed"] += 1
                except Exception as e:
                    error_msg = str(e)[:500]
                    new_retry = retry_count + 1
                    new_status = "failed" if new_retry >= 3 else "pending"
                    self.db.execute(
                        "UPDATE night_tasks SET status = ?, error = ?, "
                        "retry_count = ? WHERE id = ?",
                        (new_status, error_msg, new_retry, task_id))
                    self.db.commit()
                    result["failed"] += 1
                    logger.debug("Night task %s failed: %s", task_id, error_msg)

                processed_total += 1
                if processed_total >= self._max_tasks:
                    break

        return result

    # ═══════════════════════════════════════
    # TASK HANDLERS
    # ═══════════════════════════════════════

    async def _enrich_chunk(self, chunk_id: str):
        """Generate contextual prefix for a chunk."""
        if not self._provider:
            raise RuntimeError("No LLM provider for contextual enrichment")

        row = self.db.execute(
            "SELECT c.content, c.section_path, d.name as doc_name "
            "FROM kb_chunks c JOIN kb_documents d ON d.id = c.doc_id "
            "WHERE c.id = ?", (chunk_id,)).fetchone()
        if not row:
            return

        content = row["content"] if not isinstance(row, tuple) else row[0]
        section = row["section_path"] if not isinstance(row, tuple) else row[1]
        doc_name = row["doc_name"] if not isinstance(row, tuple) else row[2]

        prompt = (
            f"Document: \"{doc_name}\"\n"
            f"Section: \"{section}\"\n\n"
            f"Chunk content:\n{content[:1500]}\n\n"
            "Please give a short succinct context (2-3 sentences) to situate this chunk "
            "within the overall document for the purposes of improving search retrieval. "
            "Answer ONLY with the context, nothing else."
        )

        resp = await self._provider.complete(
            messages=[{"role": "user", "content": prompt}], max_tokens=200)
        context_prefix = resp.text.strip()

        if context_prefix:
            self.db.execute(
                "UPDATE kb_chunks SET context_prefix = ? WHERE id = ?",
                (context_prefix, chunk_id))

            # Re-index FTS5 if available
            try:
                self.db.execute(
                    "DELETE FROM kb_fts WHERE chunk_id = ?", (chunk_id,))
                fts_content = f"{context_prefix} {content}"
                self.db.execute(
                    "INSERT INTO kb_fts (content, chunk_id, doc_name, section_path) "
                    "VALUES (?, ?, ?, ?)",
                    (fts_content, chunk_id, doc_name, section))
            except Exception:
                pass

            self.db.commit()

    async def _embed_chunk(self, chunk_id: str):
        """Generate embedding for a chunk."""
        import asyncio
        import pickle

        if not self._embedder:
            raise RuntimeError("No embedder for embedding generation")

        row = self.db.execute(
            "SELECT content, context_prefix FROM kb_chunks WHERE id = ?",
            (chunk_id,)).fetchone()
        if not row:
            return

        content = row["content"] if not isinstance(row, tuple) else row[0]
        context_prefix = row["context_prefix"] if not isinstance(row, tuple) else row[1]

        embed_text = content[:2000]
        if context_prefix:
            embed_text = f"{context_prefix}\n\n{embed_text}"

        vec = await asyncio.to_thread(self._embedder.encode, embed_text[:2000])
        embedding_blob = pickle.dumps(vec)

        self.db.execute(
            "UPDATE kb_chunks SET embedding = ? WHERE id = ?",
            (embedding_blob, chunk_id))
        self.db.commit()

    async def _extract_entities(self, chunk_id: str):
        """Extract entities from a chunk (regex + optional LLM)."""
        row = self.db.execute(
            "SELECT c.content, c.doc_id FROM kb_chunks c WHERE c.id = ?",
            (chunk_id,)).fetchone()
        if not row:
            return

        content = row["content"] if not isinstance(row, tuple) else row[0]
        doc_id = row["doc_id"] if not isinstance(row, tuple) else row[1]

        # Regex extraction
        entities = self._regex_extract(content)

        # LLM extraction if provider available
        if self._provider:
            try:
                llm_entities = await self._llm_extract_entities(content)
                # Merge, avoid duplicates
                existing_names = {e[0].lower() for e in entities}
                for name, etype in llm_entities:
                    if name.lower() not in existing_names:
                        entities.append((name, etype))
                        existing_names.add(name.lower())
            except Exception as e:
                logger.debug("LLM entity extraction failed: %s", e)

        # Upsert entities and mentions
        for name, entity_type in entities:
            self._upsert_entity(name, entity_type, doc_id, chunk_id)

        self.db.commit()

    # ═══════════════════════════════════════
    # ENTITY EXTRACTION HELPERS
    # ═══════════════════════════════════════

    @staticmethod
    def _regex_extract(text: str) -> list[tuple[str, str]]:
        """Extract entities using regex patterns."""
        entities = []
        seen = set()

        # Russian legal: ст. 123, статья 456
        for m in re.finditer(r'(?:ст(?:атья|\.)\s*(\d+(?:\.\d+)*))', text, re.I):
            name = f"ст. {m.group(1)}"
            if name.lower() not in seen:
                entities.append((name, "article"))
                seen.add(name.lower())

        # Federal laws: N 44-ФЗ, №123-ФЗ, Федеральный закон
        for m in re.finditer(
                r'(?:[NН№]\s*(\d+[-–]?(?:ФЗ|фз)))', text, re.I):
            name = f"№{m.group(1)}"
            if name.lower() not in seen:
                entities.append((name, "law"))
                seen.add(name.lower())

        # Dates: DD.MM.YYYY
        for m in re.finditer(r'\b(\d{2}\.\d{2}\.\d{4})\b', text):
            name = m.group(1)
            if name not in seen:
                entities.append((name, "date"))
                seen.add(name)

        # Monetary amounts: 1 000 000 руб, $1,234.56
        for m in re.finditer(
                r'(\d[\d\s,.]*\d)\s*(?:руб|₽|р\.|тыс|млн|млрд)', text, re.I):
            name = m.group(0).strip()
            if name not in seen:
                entities.append((name, "amount"))
                seen.add(name)

        # Percentages: 13%, 18.5%
        for m in re.finditer(r'(\d+(?:[.,]\d+)?)\s*%', text):
            name = f"{m.group(1)}%"
            if name not in seen:
                entities.append((name, "percentage"))
                seen.add(name)

        return entities

    async def _llm_extract_entities(self, text: str) -> list[tuple[str, str]]:
        """Extract entities using LLM."""
        if not self._provider:
            return []

        prompt = (
            "Extract key entities (people, organizations, laws, dates, terms) "
            "from the following text. Return a JSON array of objects with "
            "\"name\" and \"type\" fields.\n\n"
            f"Text:\n{text[:1500]}\n\n"
            "Entities (JSON array):"
        )

        resp = await self._provider.complete(
            messages=[{"role": "user", "content": prompt}], max_tokens=500)

        match = re.search(r'\[.*\]', resp.text, re.S)
        if match:
            try:
                items = json.loads(match.group(0))
                return [
                    (item["name"], item.get("type", "unknown"))
                    for item in items
                    if isinstance(item, dict) and "name" in item
                ]
            except (json.JSONDecodeError, KeyError):
                pass
        return []

    def _upsert_entity(self, name: str, entity_type: str, doc_id: str, chunk_id: str):
        """Insert or update entity and create mention link."""
        # Check kb_entities table exists (created by KnowledgeBase Phase 5)
        try:
            existing = self.db.execute(
                "SELECT id, count FROM kb_entities WHERE name = ? AND doc_id = ?",
                (name, doc_id)).fetchone()

            if existing:
                eid = existing[0] if isinstance(existing, tuple) else existing["id"]
                cnt = existing[1] if isinstance(existing, tuple) else existing["count"]
                self.db.execute(
                    "UPDATE kb_entities SET count = ? WHERE id = ?",
                    (cnt + 1, eid))
            else:
                eid = str(uuid.uuid4())
                self.db.execute(
                    "INSERT INTO kb_entities (id, name, entity_type, doc_id, count) "
                    "VALUES (?, ?, ?, ?, 1)",
                    (eid, name, entity_type, doc_id))

            # Create mention if not exists
            existing_mention = self.db.execute(
                "SELECT 1 FROM kb_entity_mentions WHERE entity_id = ? AND chunk_id = ?",
                (eid, chunk_id)).fetchone()
            if not existing_mention:
                self.db.execute(
                    "INSERT INTO kb_entity_mentions (entity_id, chunk_id) VALUES (?, ?)",
                    (eid, chunk_id))
        except Exception as e:
            # Tables might not exist yet (Phase 5 not applied)
            logger.debug("Entity upsert skipped (tables may not exist): %s", e)

    def stop(self):
        """Signal the worker to stop."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
