"""Tests for RAG pipeline — chunking, ingestion, search, management."""

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock

import pytest

from liteagent.rag import (
    RAGPipeline,
    VectorBackend,
    SqliteBruteForceBackend,
    _cosine_similarity,
    create_vector_backend,
)


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

    def test_get_stats_extended_fields(self, rag):
        """Stats include backend, search mode, chunk settings, embedder info."""
        stats = rag.get_stats()
        assert "backend" in stats
        assert "search_mode" in stats
        assert "chunk_size" in stats
        assert "chunk_overlap" in stats
        assert stats["search_mode"] == "hybrid"  # default from config


# ═══════════════════════════════════════
# VECTOR BACKENDS
# ═══════════════════════════════════════

class TestVectorBackend:
    def test_abstract_interface(self):
        """VectorBackend raises NotImplementedError."""
        backend = VectorBackend()
        with pytest.raises(NotImplementedError):
            backend.upsert([], [], [])
        with pytest.raises(NotImplementedError):
            backend.search(None)
        with pytest.raises(NotImplementedError):
            backend.delete([])
        with pytest.raises(NotImplementedError):
            backend.count()

    def test_sqlite_brute_force_empty(self, db):
        """SqliteBruteForceBackend returns empty on no data."""
        # Need rag_chunks table
        RAGPipeline(db, embedder=None, config={})
        backend = SqliteBruteForceBackend(db)
        assert backend.count() == 0
        assert backend.search(None) == []

    def test_sqlite_brute_force_search(self, db):
        """SqliteBruteForceBackend performs cosine similarity search."""
        import numpy as np
        import pickle

        rag = RAGPipeline(db, embedder=None, config={})
        backend = SqliteBruteForceBackend(db)

        # Insert test data manually
        db.execute(
            "INSERT INTO rag_documents (path, name, doc_hash, chunk_count, created_at) "
            "VALUES ('test.txt', 'test.txt', 'abc', 1, '2024-01-01')")
        doc_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        vec = np.array([1.0, 0.0, 0.0], dtype="float32")
        db.execute(
            "INSERT INTO rag_chunks (doc_id, chunk_index, content, embedding) "
            "VALUES (?, 0, 'test content', ?)",
            (doc_id, pickle.dumps(vec)))
        db.commit()

        query_vec = np.array([0.9, 0.1, 0.0], dtype="float32")
        results = backend.search(query_vec, top_k=5)
        assert len(results) == 1
        assert results[0]["payload"]["content"] == "test content"
        assert results[0]["score"] > 0.9

    def test_create_vector_backend_keyword(self, db):
        """vector_backend=keyword returns None."""
        config = {"rag": {"vector_backend": "keyword"}}
        backend = create_vector_backend(config, db, embedder=None)
        assert backend is None

    def test_create_vector_backend_auto_fallback(self, db):
        """Auto mode falls back through unavailable backends."""
        config = {"rag": {"vector_backend": "auto"}}
        # Without sqlite-vec or Qdrant, should try brute-force
        # But brute-force needs embedder, so with None → returns None
        backend = create_vector_backend(config, db, embedder=None)
        # May be None (no embedder) or brute-force
        # Should not crash
        assert backend is None or isinstance(backend, SqliteBruteForceBackend)


# ═══════════════════════════════════════
# COSINE SIMILARITY
# ═══════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-6

    def test_zero_vector(self):
        import numpy as np
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 0.0])
        assert _cosine_similarity(a, b) == 0.0


# ═══════════════════════════════════════
# HYBRID SEARCH / RRF FUSION
# ═══════════════════════════════════════

class TestHybridSearch:
    def test_rrf_fusion_basic(self):
        """RRF fusion merges two ranked lists."""
        vec_results = [
            {"content": "doc1", "source": "a.txt", "chunk_index": 0, "score": 0.9},
            {"content": "doc2", "source": "b.txt", "chunk_index": 0, "score": 0.7},
        ]
        kw_results = [
            {"content": "doc2", "source": "b.txt", "chunk_index": 0, "score": 0.8},
            {"content": "doc3", "source": "c.txt", "chunk_index": 0, "score": 0.5},
        ]
        fused = RAGPipeline._rrf_fusion(vec_results, kw_results, k=60)
        assert len(fused) == 3
        # doc2 appears in both → should be ranked higher
        doc2_items = [r for r in fused if r["source"] == "b.txt"]
        assert len(doc2_items) == 1
        # doc2 should have higher RRF score than doc3
        assert doc2_items[0]["score"] > [r for r in fused if r["source"] == "c.txt"][0]["score"]

    def test_rrf_fusion_empty_lists(self):
        """RRF fusion with empty lists returns empty."""
        assert RAGPipeline._rrf_fusion([], []) == []

    def test_rrf_fusion_single_list(self):
        """RRF fusion with one empty list returns the other."""
        vec = [{"content": "a", "source": "x.txt", "chunk_index": 0, "score": 0.9}]
        result = RAGPipeline._rrf_fusion(vec, [])
        assert len(result) == 1

    def test_search_mode_keyword_only(self, db, tmp_dir):
        """keyword mode uses only keyword search."""
        rag = RAGPipeline(db, embedder=None, config={
            "chunk_size": 100, "overlap": 20, "top_k": 3,
            "search": {"mode": "keyword"},
        })
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Python is a great programming language for data science.")
        rag.ingest(path)
        results = rag.search("Python")
        assert len(results) > 0

    def test_search_mode_hybrid_no_embedder(self, db, tmp_dir):
        """hybrid mode without embedder falls back to keyword."""
        rag = RAGPipeline(db, embedder=None, config={
            "chunk_size": 100, "overlap": 20, "top_k": 3,
            "search": {"mode": "hybrid"},
        })
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Machine learning algorithms for classification tasks.")
        rag.ingest(path)
        results = rag.search("machine learning")
        assert len(results) > 0


# ═══════════════════════════════════════
# INDEX_CONTENT (for FileManager integration)
# ═══════════════════════════════════════

class TestIndexContent:
    def test_index_content_basic(self, db):
        """index_content stores text without reading from filesystem."""
        rag = RAGPipeline(db, embedder=None, config={"chunk_size": 100, "overlap": 20})
        result = rag.index_content(
            "This is the full content of a document stored in S3.",
            source_key="files/api/abc_test.txt",
            source_name="test.txt",
            file_type=".txt")
        assert result["status"] == "indexed"
        assert result["chunks"] >= 1

        # Verify in database
        docs = rag.list_documents()
        assert len(docs) == 1
        assert docs[0]["path"] == "files/api/abc_test.txt"
        assert docs[0]["name"] == "test.txt"

    def test_index_content_deduplication(self, db):
        """Same content + key = unchanged on second call."""
        rag = RAGPipeline(db, embedder=None, config={"chunk_size": 100, "overlap": 20})
        r1 = rag.index_content("Same text", source_key="key1", source_name="doc.txt")
        assert r1["status"] == "indexed"
        r2 = rag.index_content("Same text", source_key="key1", source_name="doc.txt")
        assert r2["status"] == "unchanged"

    def test_index_content_update(self, db):
        """Changed content for same key → re-indexed."""
        rag = RAGPipeline(db, embedder=None, config={"chunk_size": 100, "overlap": 20})
        rag.index_content("Version 1", source_key="key1", source_name="doc.txt")
        r2 = rag.index_content("Version 2 is different", source_key="key1", source_name="doc.txt")
        assert r2["status"] == "indexed"

    def test_index_content_searchable(self, db):
        """Indexed content is searchable via keyword search."""
        rag = RAGPipeline(db, embedder=None, config={
            "chunk_size": 500, "overlap": 50, "top_k": 5,
            "search": {"mode": "keyword"},
        })
        rag.index_content(
            "Quantum computing uses qubits instead of classical bits.",
            source_key="files/api/quantum.pdf",
            source_name="quantum.pdf",
            file_type=".txt")
        results = rag.search("quantum qubits")
        assert len(results) > 0
        assert "quantum" in results[0]["content"].lower()


# ═══════════════════════════════════════
# IMPROVED CHUNKING (markdown/code-aware)
# ═══════════════════════════════════════

class TestImprovedChunking:
    def test_markdown_aware_splitting(self, db):
        """Markdown files split by heading hierarchy."""
        rag = RAGPipeline(db, embedder=None, config={"chunk_size": 80, "overlap": 10})
        text = (
            "# Chapter 1\n\nFirst chapter content with enough words to fill the chunk size.\n\n"
            "# Chapter 2\n\nSecond chapter content with enough words to fill the chunk size too."
        )
        chunks = rag.chunk_text(text, file_type="md")
        assert len(chunks) >= 2
        combined = " ".join(chunks)
        assert "Chapter 1" in combined
        assert "Chapter 2" in combined

    def test_python_aware_splitting(self, db):
        """Python files split by class/def boundaries."""
        rag = RAGPipeline(db, embedder=None, config={"chunk_size": 80, "overlap": 10})
        text = "class Foo:\n    pass\n\ndef bar():\n    return 1\n\ndef baz():\n    return 2"
        chunks = rag.chunk_text(text, file_type="py")
        assert len(chunks) >= 1
        combined = " ".join(chunks)
        assert "Foo" in combined
        assert "bar" in combined

    def test_code_block_protection(self, db):
        """Fenced code blocks are not split in the middle."""
        rag = RAGPipeline(db, embedder=None, config={"chunk_size": 80, "overlap": 10})
        text = "Some intro.\n\n```python\ndef hello():\n    print('world')\n```\n\nSome outro."
        chunks = rag.chunk_text(text, file_type="md")
        # The code block should be intact in one chunk
        found_code = False
        for c in chunks:
            if "def hello" in c:
                assert "print('world')" in c
                found_code = True
        assert found_code

    def test_protect_and_restore_code_blocks(self):
        """_protect_code_blocks and _restore_code_blocks are inverse."""
        text = "before ```code here``` after"
        protected, blocks = RAGPipeline._protect_code_blocks(text)
        assert "```" not in protected
        assert len(blocks) == 1
        restored = RAGPipeline._restore_code_blocks(protected, blocks)
        assert restored == text


# ═══════════════════════════════════════
# FTS5 BM25
# ═══════════════════════════════════════

class TestFTS5BM25:
    def test_fts5_initialization(self, db):
        """FTS5 table created if available, _fts_available flag set."""
        rag = RAGPipeline(db, embedder=None, config={"chunk_size": 100, "overlap": 20})
        assert hasattr(rag, "_fts_available")
        # In standard Python sqlite3, FTS5 should be available
        assert isinstance(rag._fts_available, bool)

    def test_bm25_search_with_fts5(self, db, tmp_dir):
        """BM25 search returns results via FTS5."""
        rag = RAGPipeline(db, embedder=None, config={
            "chunk_size": 200, "overlap": 20, "top_k": 5,
            "search": {"mode": "keyword"},
        })
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Python is excellent for machine learning and data science applications.")
        rag.ingest(path)

        if not rag._fts_available:
            pytest.skip("FTS5 not available in this SQLite build")

        results = rag._bm25_search("Python machine learning", top_k=5)
        assert len(results) > 0
        assert "Python" in results[0]["content"]
        assert results[0]["score"] >= 0

    def test_word_overlap_fallback(self, db, tmp_dir):
        """Word overlap search works when FTS5 is disabled."""
        rag = RAGPipeline(db, embedder=None, config={
            "chunk_size": 200, "overlap": 20, "top_k": 5,
        })
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Artificial intelligence is transforming healthcare and education.")
        rag.ingest(path)

        # Force FTS unavailable
        rag._fts_available = False
        results = rag._keyword_search("intelligence healthcare", top_k=5)
        assert len(results) > 0
        assert "intelligence" in results[0]["content"].lower()

    def test_fts5_cleanup_on_delete(self, db, tmp_dir):
        """FTS entries are cleaned when document is deleted."""
        rag = RAGPipeline(db, embedder=None, config={
            "chunk_size": 200, "overlap": 20,
        })
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Unique searchable content for deletion test.")
        rag.ingest(path)

        if not rag._fts_available:
            pytest.skip("FTS5 not available in this SQLite build")

        # Should find it
        results = rag._bm25_search("unique searchable deletion", top_k=5)
        assert len(results) > 0

        # Get doc_id and delete
        docs = rag.list_documents()
        assert len(docs) > 0
        rag.delete_document(docs[0]["id"])

        # Should not find it
        results = rag._bm25_search("unique searchable deletion", top_k=5)
        assert len(results) == 0

    def test_keyword_search_prefers_bm25(self, db, tmp_dir):
        """_keyword_search uses BM25 when FTS5 is available."""
        rag = RAGPipeline(db, embedder=None, config={
            "chunk_size": 200, "overlap": 20, "top_k": 5,
        })
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, "w") as f:
            f.write("Database optimization techniques for SQL performance tuning.")
        rag.ingest(path)

        if not rag._fts_available:
            pytest.skip("FTS5 not available in this SQLite build")

        results = rag._keyword_search("database optimization SQL", top_k=5)
        assert len(results) > 0
