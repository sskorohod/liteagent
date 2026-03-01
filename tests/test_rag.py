"""Tests for RAG pipeline — chunking, ingestion, search, management."""

import os
import sqlite3
import tempfile

import pytest

from liteagent.rag import RAGPipeline


@pytest.fixture
def db():
    """In-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def rag(db):
    """RAG pipeline with keyword-only search (no embedder)."""
    return RAGPipeline(db, embedder=None, config={
        "chunk_size": 100,
        "overlap": 20,
        "top_k": 3,
    })


@pytest.fixture
def tmp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield d


# ═══════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════

class TestChunking:
    def test_empty_text(self, rag):
        assert rag.chunk_text("") == []
        assert rag.chunk_text("   ") == []

    def test_short_text_no_split(self, rag):
        text = "Hello, world!"
        chunks = rag.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits(self, rag):
        # Create text longer than chunk_size (100)
        text = "This is sentence one. " * 20  # ~440 chars
        chunks = rag.chunk_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            # Chunks should be reasonable size (may exceed slightly due to overlap)
            assert len(chunk) > 0

    def test_paragraph_splitting(self, rag):
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird paragraph content here."
        chunks = rag.chunk_text(text)
        assert len(chunks) >= 1
        # All content should be preserved
        combined = " ".join(chunks)
        assert "First" in combined
        assert "Third" in combined

    def test_newline_splitting(self, rag):
        lines = ["Line number " + str(i) + " with some extra text" for i in range(20)]
        text = "\n".join(lines)
        chunks = rag.chunk_text(text)
        assert len(chunks) > 1


# ═══════════════════════════════════════
# FILE LOADING
# ═══════════════════════════════════════

class TestFileLoading:
    def test_load_txt_file(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Hello, this is a test file.")
        content = rag.load_file(path)
        assert "Hello" in content

    def test_load_md_file(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "test.md")
        with open(path, "w") as f:
            f.write("# Header\n\nSome **bold** text.")
        content = rag.load_file(path)
        assert "Header" in content

    def test_load_html_file(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "test.html")
        with open(path, "w") as f:
            f.write("<html><body><p>Hello &amp; World</p></body></html>")
        content = rag.load_file(path)
        assert "Hello" in content
        assert "& World" in content
        assert "<p>" not in content  # Tags stripped

    def test_load_py_file(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "test.py")
        with open(path, "w") as f:
            f.write("def hello():\n    return 'world'\n")
        content = rag.load_file(path)
        assert "def hello" in content

    def test_load_nonexistent_file(self, rag):
        with pytest.raises(FileNotFoundError):
            rag.load_file("/nonexistent/file.txt")

    def test_strip_html(self):
        html = "<html><script>alert('x')</script><p>Hello &amp; World</p></html>"
        text = RAGPipeline._strip_html(html)
        assert "alert" not in text
        assert "Hello & World" in text


# ═══════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════

class TestIngestion:
    def test_ingest_single_file(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "doc.txt")
        with open(path, "w") as f:
            f.write("This is a test document with some content for ingestion.")
        result = rag.ingest(path)
        assert result["status"] == "ingested"
        assert result["chunks"] >= 1

    def test_ingest_deduplication(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "doc.txt")
        with open(path, "w") as f:
            f.write("Same content for dedup test.")
        r1 = rag.ingest(path)
        assert r1["status"] == "ingested"
        r2 = rag.ingest(path)
        assert r2["status"] == "unchanged"

    def test_ingest_update_on_change(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "doc.txt")
        with open(path, "w") as f:
            f.write("Version 1 content.")
        r1 = rag.ingest(path)
        assert r1["status"] == "ingested"

        with open(path, "w") as f:
            f.write("Version 2 content is different.")
        r2 = rag.ingest(path)
        assert r2["status"] == "ingested"

    def test_ingest_directory(self, rag, tmp_dir):
        for name in ["a.txt", "b.md", "c.py"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write(f"Content of {name} for testing.")
        # Also create a non-supported file
        with open(os.path.join(tmp_dir, "skip.bin"), "wb") as f:
            f.write(b"\x00\x01\x02")
        result = rag.ingest(tmp_dir)
        assert result["files"] == 3
        assert result["chunks"] >= 3


# ═══════════════════════════════════════
# SEARCH
# ═══════════════════════════════════════

class TestSearch:
    def test_keyword_search(self, rag, tmp_dir):
        # Ingest a file with known content
        path = os.path.join(tmp_dir, "doc.txt")
        with open(path, "w") as f:
            f.write("Python is a programming language. It supports many paradigms.")
        rag.ingest(path)
        results = rag.search("Python programming")
        assert len(results) > 0
        assert "Python" in results[0]["content"]

    def test_search_empty_db(self, rag):
        results = rag.search("anything")
        assert results == []

    def test_search_returns_source(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "myfile.txt")
        with open(path, "w") as f:
            f.write("Unique content about quantum mechanics and physics.")
        rag.ingest(path)
        results = rag.search("quantum physics")
        assert len(results) > 0
        assert results[0]["source"] == "myfile.txt"

    def test_search_respects_top_k(self, rag, tmp_dir):
        # Create multiple files
        for i in range(5):
            path = os.path.join(tmp_dir, f"doc{i}.txt")
            with open(path, "w") as f:
                f.write(f"Document {i} about testing search functionality.")
        rag.ingest(tmp_dir)
        results = rag.search("testing search", top_k=2)
        assert len(results) <= 2


# ═══════════════════════════════════════
# MANAGEMENT
# ═══════════════════════════════════════

class TestManagement:
    def test_list_documents(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "doc.txt")
        with open(path, "w") as f:
            f.write("Test document content.")
        rag.ingest(path)
        docs = rag.list_documents()
        assert len(docs) == 1
        assert docs[0]["name"] == "doc.txt"
        assert docs[0]["chunks"] >= 1

    def test_delete_document(self, rag, tmp_dir):
        path = os.path.join(tmp_dir, "doc.txt")
        with open(path, "w") as f:
            f.write("Document to delete.")
        rag.ingest(path)
        docs = rag.list_documents()
        assert len(docs) == 1
        ok = rag.delete_document(docs[0]["id"])
        assert ok is True
        assert len(rag.list_documents()) == 0

    def test_delete_nonexistent_document(self, rag):
        ok = rag.delete_document(999)
        assert ok is False

    def test_get_stats(self, rag, tmp_dir):
        stats = rag.get_stats()
        assert stats["documents"] == 0
        assert stats["chunks"] == 0

        path = os.path.join(tmp_dir, "doc.txt")
        with open(path, "w") as f:
            f.write("Some content for stats test.")
        rag.ingest(path)
        stats = rag.get_stats()
        assert stats["documents"] == 1
        assert stats["chunks"] >= 1
