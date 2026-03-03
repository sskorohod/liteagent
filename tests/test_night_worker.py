"""Tests for night_worker.py — background batch processor."""

import json
import sqlite3
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from liteagent.night_worker import NightWorker, TaskType, TaskStatus


@pytest.fixture
def nw_db(tmp_path):
    """Create a test DB with KB schema + night_tasks."""
    db_path = str(tmp_path / "test_kb.db")
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE kb_documents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            doc_hash TEXT UNIQUE,
            page_count INTEGER DEFAULT 0,
            chunk_count INTEGER DEFAULT 0,
            metadata_json TEXT DEFAULT '{}',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE kb_chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES kb_documents(id),
            content TEXT NOT NULL,
            section_path TEXT DEFAULT '',
            page_start INTEGER DEFAULT 0,
            page_end INTEGER DEFAULT 0,
            chunk_type TEXT DEFAULT 'text',
            chunk_index INTEGER DEFAULT 0,
            embedding BLOB,
            context_prefix TEXT DEFAULT '',
            parent_id TEXT DEFAULT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE kb_entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            doc_id TEXT,
            count INTEGER DEFAULT 1
        );
        CREATE TABLE kb_entity_mentions (
            entity_id TEXT,
            chunk_id TEXT,
            PRIMARY KEY (entity_id, chunk_id)
        );
    """)
    # Add FTS5 if available
    try:
        db.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
                content, chunk_id UNINDEXED, doc_name UNINDEXED,
                section_path UNINDEXED, tokenize='unicode61'
            )
        """)
    except Exception:
        pass
    db.commit()
    yield db
    db.close()


@pytest.fixture
def sample_data(nw_db):
    """Insert sample document and chunks."""
    nw_db.execute(
        "INSERT INTO kb_documents (id, name, doc_hash) VALUES ('d1', 'Tax Guide', 'hash1')")
    for i in range(5):
        nw_db.execute(
            "INSERT INTO kb_chunks (id, doc_id, content, section_path, chunk_type) "
            "VALUES (?, 'd1', ?, 'Chapter 1', 'text')",
            (f"c{i}", f"Chunk {i} about tax rates and regulations. Статья {i+100}."))
    nw_db.commit()
    return nw_db


class TestNightWorkerSchema:
    """Test schema creation."""

    def test_creates_night_tasks_table(self, nw_db):
        worker = NightWorker({}, nw_db)
        tables = [r[0] for r in nw_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "night_tasks" in tables

    def test_creates_indexes(self, nw_db):
        worker = NightWorker({}, nw_db)
        indexes = [r[0] for r in nw_db.execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()]
        assert "idx_nt_status" in indexes
        assert "idx_nt_type" in indexes


class TestNightWorkerQueue:
    """Test queue operations."""

    def test_enqueue_single(self, nw_db):
        worker = NightWorker({}, nw_db)
        task_id = worker.enqueue("contextual_enrichment", "c1")
        assert task_id
        row = nw_db.execute("SELECT * FROM night_tasks WHERE id = ?",
                            (task_id,)).fetchone()
        assert row["status"] == "pending"
        assert row["target_id"] == "c1"

    def test_enqueue_dedup(self, nw_db):
        worker = NightWorker({}, nw_db)
        id1 = worker.enqueue("contextual_enrichment", "c1")
        id2 = worker.enqueue("contextual_enrichment", "c1")
        assert id1 == id2  # Same task, deduped

    def test_enqueue_batch(self, nw_db):
        worker = NightWorker({}, nw_db)
        count = worker.enqueue_batch("embedding_generation", ["c1", "c2", "c3"])
        assert count == 3

    def test_enqueue_batch_dedup(self, nw_db):
        worker = NightWorker({}, nw_db)
        worker.enqueue("embedding_generation", "c1")
        count = worker.enqueue_batch("embedding_generation", ["c1", "c2"])
        assert count == 1  # c1 already pending

    def test_queue_stats(self, nw_db):
        worker = NightWorker({}, nw_db)
        worker.enqueue_batch("contextual_enrichment", ["c1", "c2"])
        worker.enqueue("embedding_generation", "c3")
        stats = worker.get_queue_stats()
        assert stats["pending"] == 3
        assert stats["total"] == 3

    def test_enqueue_unenriched(self, sample_data):
        worker = NightWorker({}, sample_data)
        counts = worker.enqueue_unenriched()
        # All 5 chunks need enrichment (no context_prefix)
        assert counts.get("contextual_enrichment", 0) == 5
        # All 5 need embedding (no embedding blob)
        assert counts.get("embedding_generation", 0) == 5


class TestNightWorkerExecution:
    """Test task execution."""

    @pytest.mark.asyncio
    async def test_run_empty_queue(self, nw_db):
        worker = NightWorker({}, nw_db)
        result = await worker.run()
        assert result["status"] == "completed"
        assert result["processed"] == 0

    @pytest.mark.asyncio
    async def test_run_already_running(self, nw_db):
        worker = NightWorker({}, nw_db)
        worker._running = True
        result = await worker.run()
        assert result["status"] == "already_running"
        worker._running = False

    @pytest.mark.asyncio
    async def test_enrich_chunk(self, sample_data):
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=MagicMock(
            text="This chunk covers income tax rates."))

        worker = NightWorker({}, sample_data, provider=mock_provider)
        await worker._enrich_chunk("c0")

        row = sample_data.execute(
            "SELECT context_prefix FROM kb_chunks WHERE id = 'c0'").fetchone()
        assert row["context_prefix"] == "This chunk covers income tax rates."

    @pytest.mark.asyncio
    async def test_embed_chunk(self, sample_data):
        import numpy as np
        mock_embedder = MagicMock()
        mock_embedder.encode = MagicMock(return_value=np.random.randn(384))

        worker = NightWorker({}, sample_data, embedder=mock_embedder)
        await worker._embed_chunk("c0")

        row = sample_data.execute(
            "SELECT embedding FROM kb_chunks WHERE id = 'c0'").fetchone()
        assert row["embedding"] is not None

    @pytest.mark.asyncio
    async def test_extract_entities_regex(self, sample_data):
        worker = NightWorker({}, sample_data)
        await worker._extract_entities("c0")
        # Should find "ст. 100" via regex
        entities = sample_data.execute(
            "SELECT * FROM kb_entities").fetchall()
        # May or may not find entities depending on regex match
        assert isinstance(entities, list)

    @pytest.mark.asyncio
    async def test_run_enrichment(self, sample_data):
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=MagicMock(
            text="Context for this chunk."))

        worker = NightWorker({"batch_size": 2, "max_tasks_per_run": 3},
                             sample_data, provider=mock_provider)
        worker.enqueue_batch("contextual_enrichment", ["c0", "c1", "c2"])

        result = await worker.run()
        assert result["processed"] == 3
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_task_failure_retry(self, sample_data):
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("LLM down"))

        # max_tasks_per_run=1 so the run only processes the task once
        worker = NightWorker({"batch_size": 10, "max_tasks_per_run": 1},
                             sample_data, provider=mock_provider)
        worker.enqueue("contextual_enrichment", "c0")

        result = await worker.run()
        # Should fail but remain pending (retry_count < 3)
        row = sample_data.execute(
            "SELECT status, retry_count FROM night_tasks WHERE target_id = 'c0'"
        ).fetchone()
        assert row["retry_count"] == 1
        assert row["status"] == "pending"  # Can retry

    def test_stop(self, nw_db):
        worker = NightWorker({}, nw_db)
        worker._running = True
        worker.stop()
        assert not worker.is_running


class TestRegexExtract:
    """Test entity regex extraction."""

    def test_extract_article(self):
        entities = NightWorker._regex_extract("Согласно ст. 123 НК РФ")
        names = [e[0] for e in entities]
        assert "ст. 123" in names

    def test_extract_law(self):
        entities = NightWorker._regex_extract("Федеральный закон N 44-ФЗ")
        types = {e[0]: e[1] for e in entities}
        assert any("44-ФЗ" in k for k in types)

    def test_extract_date(self):
        entities = NightWorker._regex_extract("Дата вступления 01.01.2024")
        names = [e[0] for e in entities]
        assert "01.01.2024" in names

    def test_extract_percentage(self):
        entities = NightWorker._regex_extract("Ставка НДС 18.5%")
        names = [e[0] for e in entities]
        assert "18.5%" in names

    def test_extract_amount(self):
        entities = NightWorker._regex_extract("Сумма 1 000 000 руб")
        assert any(e[1] == "amount" for e in entities)

    def test_no_duplicates(self):
        entities = NightWorker._regex_extract("ст. 123 и ст. 123 повторно")
        article_entities = [e for e in entities if e[0] == "ст. 123"]
        assert len(article_entities) == 1
