"""Tests for liteagent.file_manager — centralized file storage + index."""

import asyncio
import pickle
import sqlite3
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from liteagent.file_manager import FileManager, create_file_manager


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def db():
    """In-memory SQLite with file_index table."""
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
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
    """)
    conn.commit()
    return conn


@pytest.fixture
def mock_storage():
    """Mock StorageBackend with async methods."""
    s = MagicMock()
    s.async_upload = AsyncMock(return_value="files/test/abc_hello.txt")
    s.async_download = AsyncMock(return_value=b"hello world")
    s.async_get_url = AsyncMock(return_value="https://s3.example.com/presigned")
    s.async_delete = AsyncMock(return_value=True)
    s.async_delete_many = AsyncMock(return_value={"deleted": ["key1"], "errors": []})
    s.async_list_files = AsyncMock(return_value=[])
    return s


@pytest.fixture
def fm(mock_storage, db):
    """FileManager with mock storage and in-memory DB."""
    return FileManager(mock_storage, db, embedder=None)


# ── Ingest ───────────────────────────────────────────────────

async def test_ingest_basic(fm, mock_storage, db):
    """Test basic file ingestion."""
    result = await fm.ingest(
        b"hello world", "hello.txt",
        source="telegram", user_id="tg-123")

    assert result["original_name"] == "hello.txt"
    assert result["size_bytes"] == 11
    assert result["source"] == "telegram"
    assert result["mime_type"] == "text/plain"
    mock_storage.async_upload.assert_called_once()

    # Check DB
    row = db.execute(
        "SELECT * FROM file_index WHERE original_name = 'hello.txt'").fetchone()
    assert row is not None


async def test_ingest_auto_describe_text(fm):
    """Text files get auto-description from content."""
    result = await fm.ingest(
        b"First line\nSecond line\nThird", "readme.txt",
        source="api", user_id="user-1")

    assert "First line" in result["description"]


async def test_ingest_deduplication(fm, mock_storage, db):
    """Same data + name + source → upsert (no duplicate)."""
    data = b"same content"
    await fm.ingest(data, "file.txt", source="api")
    await fm.ingest(data, "file.txt", source="api")  # same source → same key

    count = db.execute("SELECT COUNT(*) FROM file_index").fetchone()[0]
    assert count == 1  # deduplicated by storage_key


async def test_ingest_different_sources(fm, db):
    """Same data but different source → different storage keys."""
    data = b"same content"
    await fm.ingest(data, "file.txt", source="api")
    await fm.ingest(data, "file.txt", source="telegram")

    count = db.execute("SELECT COUNT(*) FROM file_index").fetchone()[0]
    assert count == 2  # different source prefixes → different keys


async def test_ingest_different_data(fm, db):
    """Different data → different entries."""
    await fm.ingest(b"data1", "a.txt", source="api")
    await fm.ingest(b"data2", "b.txt", source="api")

    count = db.execute("SELECT COUNT(*) FROM file_index").fetchone()[0]
    assert count == 2


async def test_ingest_with_embedder(mock_storage, db):
    """When embedder is available, embedding is stored."""
    import numpy as np
    embedder = MagicMock()
    embedder.encode = MagicMock(return_value=np.array([0.1, 0.2, 0.3]))

    fm = FileManager(mock_storage, db, embedder=embedder)
    await fm.ingest(b"hello", "test.txt", source="api")

    row = db.execute(
        "SELECT embedding FROM file_index WHERE original_name = 'test.txt'"
    ).fetchone()
    assert row[0] is not None  # embedding blob stored
    vec = pickle.loads(row[0])
    assert len(vec) == 3


async def test_ingest_local(fm, tmp_path):
    """Test ingesting a local file."""
    f = tmp_path / "local.txt"
    f.write_text("local content")

    result = await fm.ingest_local(str(f), source="agent", user_id="u1")
    assert result["original_name"] == "local.txt"
    assert result["size_bytes"] == len("local content")


async def test_ingest_local_not_found(fm):
    """Ingesting non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        await fm.ingest_local("/nonexistent/file.txt")


# ── Search ───────────────────────────────────────────────────

async def test_search_keyword(fm):
    """Search by keyword in filename/description."""
    await fm.ingest(b"data", "invoice_2024.pdf", source="telegram",
                    description="Invoice from Acme Corp")
    await fm.ingest(b"data2", "photo.jpg", source="telegram",
                    description="Beach photo")

    results = fm.search("invoice")
    assert len(results) >= 1
    assert results[0]["original_name"] == "invoice_2024.pdf"


async def test_search_no_results(fm):
    """Search with no matching files returns empty list."""
    results = fm.search("nonexistent")
    assert results == []


async def test_search_with_embedder(mock_storage, db):
    """Semantic search with embedder."""
    import numpy as np
    embedder = MagicMock()
    # File embedding
    embedder.encode = MagicMock(side_effect=[
        np.array([1.0, 0.0, 0.0]),  # for ingest
        np.array([0.9, 0.1, 0.0]),  # for search query
    ])

    fm = FileManager(mock_storage, db, embedder=embedder)
    await fm.ingest(b"data", "report.pdf", source="api",
                    description="Annual report 2024")

    results = fm.search("annual report")
    assert len(results) >= 1


# ── List ─────────────────────────────────────────────────────

async def test_list_files(fm):
    """List all files."""
    await fm.ingest(b"a", "a.txt", source="api")
    await fm.ingest(b"b", "b.txt", source="telegram")

    files = fm.list_files()
    assert len(files) == 2


async def test_list_files_by_source(fm):
    """Filter by source."""
    await fm.ingest(b"a", "a.txt", source="api")
    await fm.ingest(b"b", "b.txt", source="telegram")

    files = fm.list_files(source="telegram")
    assert len(files) == 1
    assert files[0]["source"] == "telegram"


async def test_list_files_by_user(fm):
    """Filter by user_id."""
    await fm.ingest(b"a", "a.txt", source="api", user_id="u1")
    await fm.ingest(b"b", "b.txt", source="api", user_id="u2")

    files = fm.list_files(user_id="u1")
    assert len(files) == 1


async def test_count_files(fm):
    """Count total files."""
    assert fm.count_files() == 0
    await fm.ingest(b"a", "a.txt", source="api")
    assert fm.count_files() == 1


# ── Download URL ─────────────────────────────────────────────

async def test_get_download_url(fm, mock_storage):
    """Get presigned URL."""
    await fm.ingest(b"data", "test.txt", source="api")

    url = await fm.get_download_url("files/api/abc_test.txt")
    assert url == "https://s3.example.com/presigned"
    mock_storage.async_get_url.assert_called_once()


# ── Cleanup ──────────────────────────────────────────────────

async def test_propose_cleanup_empty(fm):
    """No old files → empty candidates."""
    await fm.ingest(b"fresh", "new.txt", source="api")
    candidates = fm.propose_cleanup(days_unused=1)
    assert len(candidates) == 0


async def test_propose_cleanup_finds_old(fm, db):
    """Files with old accessed_at should be proposed."""
    # Insert manually with old date
    db.execute("""
        INSERT INTO file_index (storage_key, original_name, size_bytes, source, user_id,
                                created_at, accessed_at)
        VALUES ('old/key', 'old_file.txt', 100, 'api', 'u1',
                '2020-01-01 00:00:00', '2020-01-01 00:00:00')
    """)
    db.commit()

    candidates = fm.propose_cleanup(days_unused=30)
    assert len(candidates) == 1
    assert candidates[0]["original_name"] == "old_file.txt"


async def test_confirm_cleanup(fm, mock_storage, db):
    """Confirm cleanup deletes from S3 and index."""
    db.execute("""
        INSERT INTO file_index (storage_key, original_name, size_bytes, source, user_id,
                                created_at, accessed_at)
        VALUES ('key1', 'delete_me.txt', 100, 'api', 'u1',
                '2020-01-01', '2020-01-01')
    """)
    db.commit()

    result = await fm.confirm_cleanup(["key1"])
    assert result["deleted"] == ["key1"]

    row = db.execute(
        "SELECT * FROM file_index WHERE storage_key = 'key1'").fetchone()
    assert row is None  # removed from index


# ── Auto-describe ────────────────────────────────────────────

def test_auto_describe_text():
    """Text files get description from content."""
    desc = FileManager._auto_describe(
        b"Hello world\nSecond line", "text/plain", "readme.txt")
    assert "Hello world" in desc


def test_auto_describe_image():
    """Image files get a basic description."""
    desc = FileManager._auto_describe(
        b"\x89PNG\r\n\x1a\n", "image/png", "photo.png")
    assert "photo.png" in desc


def test_auto_describe_unknown():
    """Unknown mime type still gets filename."""
    desc = FileManager._auto_describe(
        b"\x00\x01", "application/octet-stream", "data.bin")
    assert "data.bin" in desc


# ── create_file_manager ─────────────────────────────────────

def test_create_file_manager_no_storage():
    """Returns None when storage is not enabled."""
    agent = MagicMock()
    agent._storage = None
    assert create_file_manager(agent) is None


def test_create_file_manager_with_storage():
    """Returns FileManager when storage is available."""
    agent = MagicMock()
    agent._storage = MagicMock()
    agent.memory.db = sqlite3.connect(":memory:")
    agent.memory.db.executescript("""
        CREATE TABLE IF NOT EXISTS file_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            storage_key TEXT UNIQUE NOT NULL,
            original_name TEXT NOT NULL,
            mime_type TEXT DEFAULT '',
            size_bytes INTEGER DEFAULT 0,
            source TEXT DEFAULT 'unknown',
            user_id TEXT DEFAULT 'system',
            description TEXT DEFAULT '',
            embedding BLOB,
            created_at TEXT DEFAULT (datetime('now')),
            accessed_at TEXT DEFAULT (datetime('now'))
        );
    """)
    agent.memory._embedder = None

    fm = create_file_manager(agent)
    assert fm is not None
    assert isinstance(fm, FileManager)


# ── Integration: agent.ingest_file ───────────────────────────

async def test_agent_ingest_file_no_storage():
    """agent.ingest_file returns None when storage is off."""
    from liteagent.agent import LiteAgent

    # Minimal mock agent
    agent = MagicMock(spec=LiteAgent)
    agent._file_manager = None

    # Call the real method
    result = await LiteAgent.ingest_file(agent, b"data", "test.txt")
    assert result is None
