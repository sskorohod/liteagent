"""Tests for Knowledge Base — Advanced RAG pipeline."""

import asyncio
import json
import os
import pickle
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from liteagent.knowledge_base import KBChunk, KBDocument, KBSearchResult, KnowledgeBase


# ═══════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════

@pytest.fixture
def kb_config(tmp_path):
    """Minimal KB config with temp DB."""
    return {
        "enabled": True,
        "db_path": str(tmp_path / "test_kb.db"),
        "chunk_size": 200,
        "chunk_overlap": 30,
        "search_mode": "hybrid",
        "rerank": False,
        "query_rewrite": False,
        "max_file_size_mb": 10,
    }


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns random vectors."""
    emb = MagicMock()
    emb.name = "mock/test"
    emb.dim = 8
    emb.encode = MagicMock(side_effect=lambda text: np.random.rand(8).astype("float32"))
    emb.encode_batch = MagicMock(
        side_effect=lambda texts: [np.random.rand(8).astype("float32") for _ in texts])
    return emb


@pytest.fixture
def mock_provider():
    """Mock LLM provider."""
    provider = AsyncMock()
    resp = MagicMock()
    resp.text = '["sub query 1", "sub query 2"]'
    provider.complete = AsyncMock(return_value=resp)
    return provider


@pytest.fixture
def kb(kb_config, mock_embedder):
    """Knowledge base with mock embedder, no provider."""
    kb_inst = KnowledgeBase(config=kb_config, embedder=mock_embedder)
    yield kb_inst
    kb_inst.close()


@pytest.fixture
def kb_with_provider(kb_config, mock_embedder, mock_provider):
    """Knowledge base with both mock embedder and provider."""
    kb_inst = KnowledgeBase(
        config=kb_config, embedder=mock_embedder, provider=mock_provider)
    yield kb_inst
    kb_inst.close()


@pytest.fixture
def sample_md(tmp_path):
    """Sample markdown file with headings."""
    p = tmp_path / "sample.md"
    p.write_text(
        "# Chapter 1\n\nIntroduction paragraph.\n\n"
        "## Section 1.1\n\nFirst section content here with enough text.\n\n"
        "## Section 1.2\n\nSecond section content with details.\n\n"
        "# Chapter 2\n\nAnother chapter content.\n\n"
        "## Section 2.1\n\nFinal section.\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def sample_txt(tmp_path):
    """Sample plain text file."""
    p = tmp_path / "sample.txt"
    p.write_text(
        "First paragraph about accounting.\n\n"
        "Second paragraph about taxes.\n\n"
        "Third paragraph about deductions.\n",
        encoding="utf-8",
    )
    return str(p)


@pytest.fixture
def sample_html(tmp_path):
    """Sample HTML file."""
    p = tmp_path / "sample.html"
    p.write_text(
        "<html><body>"
        "<h1>Tax Guide</h1><p>Welcome to the guide.</p>"
        "<h2>Income Tax</h2><p>Income tax details.</p>"
        "<table><tr><td>Rate</td><td>13%</td></tr></table>"
        "</body></html>",
        encoding="utf-8",
    )
    return str(p)


# ═══════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════

class TestSchema:
    def test_schema_creates_tables(self, kb):
        tables = [row[0] for row in kb.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "kb_documents" in tables
        assert "kb_chunks" in tables
        assert "kb_query_log" in tables

    def test_fts_available(self, kb):
        assert kb._fts_available is True

    def test_schema_idempotent(self, kb):
        kb._init_schema()  # second call should not fail


# ═══════════════════════════════════════
# PARSING
# ═══════════════════════════════════════

class TestParsing:
    def test_parse_markdown_sections(self, kb, sample_md):
        parsed = kb._parse_file(sample_md)
        sections = parsed["pages"][0]["sections"]
        assert len(sections) >= 4
        # Check heading levels detected
        titles = [s["title"] for s in sections if s["title"]]
        assert "Chapter 1" in titles
        assert "Section 1.1" in titles

    def test_parse_plain_text(self, kb, sample_txt):
        parsed = kb._parse_file(sample_txt)
        sections = parsed["pages"][0]["sections"]
        assert len(sections) == 3
        assert "accounting" in sections[0]["content"]

    def test_parse_html_headings(self, kb, sample_html):
        parsed = kb._parse_file(sample_html)
        sections = parsed["pages"][0]["sections"]
        titles = [s["title"] for s in sections if s["title"]]
        assert "Tax Guide" in titles
        assert "Income Tax" in titles

    def test_parse_html_tables(self, kb, sample_html):
        parsed = kb._parse_file(sample_html)
        tables = parsed["pages"][0]["tables"]
        assert len(tables) >= 1

    def test_parse_file_not_found(self, kb):
        with pytest.raises(FileNotFoundError):
            kb._parse_file("/nonexistent/file.txt")

    def test_parse_file_too_large(self, kb, tmp_path):
        kb._max_file_size_mb = 0.001  # very small limit
        p = tmp_path / "big.txt"
        p.write_text("x" * 2000, encoding="utf-8")
        with pytest.raises(ValueError, match="too large"):
            kb._parse_file(str(p))

    def test_strip_html(self, kb):
        result = KnowledgeBase._strip_html(
            "<p>Hello &amp; <b>world</b></p>")
        assert "Hello & world" in result


# ═══════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════

class TestChunking:
    def test_semantic_chunking(self, kb, sample_md):
        parsed = kb._parse_file(sample_md)
        chunks = kb._chunk_structured(parsed, "test")
        assert len(chunks) >= 4
        assert all(isinstance(c, KBChunk) for c in chunks)

    def test_chunk_section_path(self, kb, sample_md):
        parsed = kb._parse_file(sample_md)
        chunks = kb._chunk_structured(parsed, "test")
        # Section 1.1 should have path "Chapter 1 > Section 1.1"
        paths = [c.section_path for c in chunks]
        assert any("Chapter 1" in p and "Section 1.1" in p for p in paths)

    def test_chunk_page_numbers(self, kb, sample_md):
        parsed = kb._parse_file(sample_md)
        chunks = kb._chunk_structured(parsed, "test")
        for c in chunks:
            assert c.page_start >= 0

    def test_large_section_split(self, kb, tmp_path):
        """Large section should be split into multiple chunks."""
        p = tmp_path / "large.md"
        content = "# Big Section\n\n" + "\n\n".join(
            f"Paragraph {i} with enough content to fill space." for i in range(50))
        p.write_text(content, encoding="utf-8")
        parsed = kb._parse_file(str(p))
        chunks = kb._chunk_structured(parsed, "test")
        assert len(chunks) > 1

    def test_chunk_respects_tables(self, kb, sample_html):
        parsed = kb._parse_file(sample_html)
        chunks = kb._chunk_structured(parsed, "test")
        table_chunks = [c for c in chunks if c.chunk_type == "table"]
        assert len(table_chunks) >= 1

    def test_split_with_overlap(self, kb):
        text = "\n\n".join(f"Paragraph {i} content." for i in range(20))
        parts = kb._split_with_overlap(text, 100, 20)
        assert len(parts) > 1

    def test_split_short_text(self, kb):
        parts = kb._split_with_overlap("Short text.", 1000, 100)
        assert len(parts) == 1
        assert parts[0] == "Short text."

    def test_empty_content_produces_no_chunks(self, kb):
        parsed = {"name": "empty", "pages": [{"number": 1, "sections": [], "tables": []}], "total_pages": 1}
        chunks = kb._chunk_structured(parsed, "test")
        assert chunks == []


# ═══════════════════════════════════════
# INGESTION
# ═══════════════════════════════════════

class TestIngestion:
    async def test_ingest_markdown(self, kb, sample_md):
        result = await kb.ingest(sample_md)
        assert result["status"] == "ok"
        assert result["chunks"] > 0
        assert result["name"] == "sample"

    async def test_ingest_dedup(self, kb, sample_md):
        r1 = await kb.ingest(sample_md)
        assert r1["status"] == "ok"
        r2 = await kb.ingest(sample_md)
        assert r2["status"] == "already_exists"
        assert r2["doc_id"] == r1["doc_id"]

    async def test_ingest_creates_fts(self, kb, sample_md):
        await kb.ingest(sample_md)
        count = kb.db.execute("SELECT COUNT(*) FROM kb_fts").fetchone()[0]
        assert count > 0

    async def test_ingest_creates_vectors(self, kb, sample_md):
        await kb.ingest(sample_md)
        count = kb.db.execute(
            "SELECT COUNT(*) FROM kb_chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]
        assert count > 0

    async def test_ingest_with_metadata(self, kb, sample_md):
        result = await kb.ingest(sample_md, metadata={"author": "Test", "name": "My Book"})
        assert result["name"] == "My Book"

    async def test_ingest_plain_text(self, kb, sample_txt):
        result = await kb.ingest(sample_txt)
        assert result["status"] == "ok"
        assert result["chunks"] >= 3

    async def test_ingest_no_embedder(self, kb_config, sample_md):
        """KB without embedder should still work (BM25 only)."""
        kb_inst = KnowledgeBase(config=kb_config, embedder=None)
        try:
            result = await kb_inst.ingest(sample_md)
            assert result["status"] == "ok"
            # Chunks should have no embeddings
            count = kb_inst.db.execute(
                "SELECT COUNT(*) FROM kb_chunks WHERE embedding IS NOT NULL"
            ).fetchone()[0]
            assert count == 0
        finally:
            kb_inst.close()


# ═══════════════════════════════════════
# BM25 SEARCH
# ═══════════════════════════════════════

class TestBM25Search:
    async def test_bm25_search(self, kb, sample_md):
        await kb.ingest(sample_md)
        results = kb._bm25_search("chapter introduction", top_k=5)
        assert len(results) > 0
        assert results[0]["score"] > 0

    async def test_bm25_no_results(self, kb, sample_md):
        await kb.ingest(sample_md)
        results = kb._bm25_search("xyznonexistentterm", top_k=5)
        assert len(results) == 0

    async def test_bm25_empty_query(self, kb, sample_md):
        await kb.ingest(sample_md)
        results = kb._bm25_search("", top_k=5)
        assert len(results) == 0

    async def test_bm25_special_characters(self, kb, sample_md):
        await kb.ingest(sample_md)
        # Should not crash on special chars
        results = kb._bm25_search("chapter (1) + 'test'", top_k=5)
        # May or may not return results, but no crash
        assert isinstance(results, list)


# ═══════════════════════════════════════
# VECTOR SEARCH
# ═══════════════════════════════════════

class TestVectorSearch:
    async def test_vector_search(self, kb, sample_md):
        await kb.ingest(sample_md)
        results = kb._vector_search("introduction", top_k=5)
        assert len(results) > 0
        assert results[0]["score"] > 0

    async def test_vector_search_no_embedder(self, kb_config, sample_md):
        kb_inst = KnowledgeBase(config=kb_config, embedder=None)
        try:
            await kb_inst.ingest(sample_md)
            results = kb_inst._vector_search("test", top_k=5)
            assert results == []
        finally:
            kb_inst.close()


# ═══════════════════════════════════════
# HYBRID SEARCH + RRF
# ═══════════════════════════════════════

class TestHybridSearch:
    async def test_hybrid_search(self, kb, sample_md):
        await kb.ingest(sample_md)
        results = await kb.search("chapter introduction", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, KBSearchResult) for r in results)

    async def test_search_returns_citations(self, kb, sample_md):
        await kb.ingest(sample_md)
        results = await kb.search("section content")
        assert len(results) > 0
        assert results[0].source == "sample"

    async def test_search_empty_kb(self, kb):
        results = await kb.search("anything")
        assert results == []

    async def test_search_bm25_mode(self, kb, sample_md):
        await kb.ingest(sample_md)
        results = await kb.search("chapter", mode="bm25")
        assert len(results) > 0

    async def test_search_vector_mode(self, kb, sample_md):
        await kb.ingest(sample_md)
        results = await kb.search("chapter", mode="vector")
        assert len(results) > 0

    def test_rrf_merge(self):
        list_a = [
            {"chunk_id": "a", "content": "A", "score": 10},
            {"chunk_id": "b", "content": "B", "score": 5},
        ]
        list_b = [
            {"chunk_id": "b", "content": "B", "score": 8},
            {"chunk_id": "c", "content": "C", "score": 3},
        ]
        merged = KnowledgeBase._rrf_merge(list_a, list_b, k=60)
        # "b" appears in both lists, so should have highest RRF score
        ids = [m["chunk_id"] for m in merged]
        assert "b" in ids
        assert "a" in ids
        assert "c" in ids
        # b should be first (present in both lists)
        assert ids[0] == "b"

    def test_rrf_empty_lists(self):
        merged = KnowledgeBase._rrf_merge([], [])
        assert merged == []


# ═══════════════════════════════════════
# RERANKING
# ═══════════════════════════════════════

class TestReranking:
    async def test_rerank_llm_fallback(self, kb_with_provider, sample_md):
        """LLM-based reranking when no cross-encoder."""
        kb_with_provider._rerank_enabled = True
        kb_with_provider._provider.complete = AsyncMock(
            return_value=MagicMock(text="[8, 3, 5]"))
        results = [
            {"chunk_id": "a", "content": "Tax deductions", "score": 0.5},
            {"chunk_id": "b", "content": "Income report", "score": 0.4},
            {"chunk_id": "c", "content": "Balance sheet", "score": 0.3},
        ]
        reranked = await kb_with_provider._rerank("tax rules", results, top_k=2)
        assert len(reranked) == 2
        # "a" scored 8, should be first
        assert reranked[0]["chunk_id"] == "a"

    def test_parse_rerank_scores(self):
        scores = KnowledgeBase._parse_rerank_scores("[8, 3, 7, 1, 5]", 5)
        assert scores == [8.0, 3.0, 7.0, 1.0, 5.0]

    def test_parse_rerank_scores_fallback(self):
        scores = KnowledgeBase._parse_rerank_scores("invalid response", 3)
        assert scores == [5.0, 5.0, 5.0]

    def test_parse_rerank_scores_wrong_count(self):
        scores = KnowledgeBase._parse_rerank_scores("[8, 3]", 5)
        assert scores == [5.0, 5.0, 5.0, 5.0, 5.0]

    async def test_rerank_preserves_metadata(self, kb_with_provider):
        kb_with_provider._rerank_enabled = True
        kb_with_provider._provider.complete = AsyncMock(
            return_value=MagicMock(text="[9, 1]"))
        results = [
            {"chunk_id": "a", "content": "Tax", "score": 0.5,
             "doc_name": "Book1", "section_path": "Ch.1", "page_start": 5},
            {"chunk_id": "b", "content": "Law", "score": 0.4,
             "doc_name": "Book2", "section_path": "Ch.2", "page_start": 10},
        ]
        reranked = await kb_with_provider._rerank("tax", results, top_k=2)
        assert reranked[0]["doc_name"] == "Book1"
        assert reranked[0]["section_path"] == "Ch.1"


# ═══════════════════════════════════════
# QUERY REWRITING
# ═══════════════════════════════════════

class TestQueryRewrite:
    async def test_query_rewrite(self, kb_with_provider):
        sub_queries = await kb_with_provider._rewrite_query("How to calculate taxes?")
        assert len(sub_queries) == 2
        assert sub_queries[0] == "sub query 1"

    async def test_query_rewrite_no_provider(self, kb):
        sub_queries = await kb._rewrite_query("test query")
        assert sub_queries == []

    async def test_query_rewrite_handles_error(self, kb_with_provider):
        kb_with_provider._provider.complete = AsyncMock(
            side_effect=Exception("API error"))
        sub_queries = await kb_with_provider._rewrite_query("test")
        assert sub_queries == []

    async def test_query_rewrite_bad_json(self, kb_with_provider):
        kb_with_provider._provider.complete = AsyncMock(
            return_value=MagicMock(text="not json at all"))
        sub_queries = await kb_with_provider._rewrite_query("test")
        assert sub_queries == []


# ═══════════════════════════════════════
# CONTEXT BUILDER
# ═══════════════════════════════════════

class TestContextBuilder:
    def test_build_context_citations(self, kb):
        results = [
            KBSearchResult(content="Tax rate is 13%.", score=0.9,
                          source="Tax Code", page=42, section="Ch.3 > Rates",
                          chunk_id="a"),
        ]
        ctx = kb.build_context(results)
        assert 'Источник: "Tax Code"' in ctx
        assert "стр. 42" in ctx
        assert "Ch.3 > Rates" in ctx
        assert "Tax rate is 13%" in ctx

    def test_build_context_dedup(self, kb):
        results = [
            KBSearchResult(content="Same content.", score=0.9,
                          source="Book", page=1, section="", chunk_id="a"),
            KBSearchResult(content="Same content.", score=0.8,
                          source="Book", page=1, section="", chunk_id="a"),
        ]
        ctx = kb.build_context(results)
        # Should appear only once (dedup by chunk_id)
        assert ctx.count("Same content.") == 1

    def test_build_context_grouped(self, kb):
        results = [
            KBSearchResult(content="From book 1.", score=0.9,
                          source="Book1", page=5, section="", chunk_id="a"),
            KBSearchResult(content="From book 2.", score=0.8,
                          source="Book2", page=10, section="", chunk_id="b"),
        ]
        ctx = kb.build_context(results)
        assert '## Источник: "Book1"' in ctx
        assert '## Источник: "Book2"' in ctx

    def test_build_context_empty(self, kb):
        ctx = kb.build_context([])
        assert "не найдено" in ctx


# ═══════════════════════════════════════
# MANAGEMENT
# ═══════════════════════════════════════

class TestManagement:
    async def test_list_documents(self, kb, sample_md):
        await kb.ingest(sample_md)
        docs = await kb.list_documents()
        assert len(docs) == 1
        assert docs[0]["name"] == "sample"

    async def test_list_documents_empty(self, kb):
        docs = await kb.list_documents()
        assert docs == []

    async def test_delete_document(self, kb, sample_md):
        result = await kb.ingest(sample_md)
        doc_id = result["doc_id"]
        ok = await kb.delete_document(doc_id)
        assert ok is True
        docs = await kb.list_documents()
        assert len(docs) == 0
        # Chunks should be deleted too
        chunk_count = kb.db.execute("SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
        assert chunk_count == 0

    async def test_delete_by_name(self, kb, sample_md):
        await kb.ingest(sample_md)
        ok = await kb.delete_document("sample")
        assert ok is True

    async def test_delete_nonexistent(self, kb):
        ok = await kb.delete_document("nonexistent-id")
        assert ok is False

    async def test_get_stats(self, kb, sample_md):
        await kb.ingest(sample_md)
        stats = await kb.get_stats()
        assert stats["documents"] == 1
        assert stats["chunks"] > 0
        assert stats["embedded_chunks"] > 0
        assert stats["fts_available"] is True
        assert stats["embedder"] == "mock/test"

    async def test_get_stats_empty(self, kb):
        stats = await kb.get_stats()
        assert stats["documents"] == 0
        assert stats["chunks"] == 0


# ═══════════════════════════════════════
# QUALITY LOGGING
# ═══════════════════════════════════════

class TestQueryLogging:
    async def test_query_logged(self, kb, sample_md):
        await kb.ingest(sample_md)
        await kb.search("test query")
        logs = kb.db.execute("SELECT * FROM kb_query_log").fetchall()
        assert len(logs) == 1
        assert logs[0]["query"] == "test query"
        assert logs[0]["latency_ms"] >= 0


# ═══════════════════════════════════════
# INTEGRATION (E2E)
# ═══════════════════════════════════════

class TestIntegration:
    async def test_full_pipeline(self, kb, sample_md):
        """Ingest → search → context (end-to-end)."""
        result = await kb.ingest(sample_md)
        assert result["status"] == "ok"

        results = await kb.search("chapter introduction")
        assert len(results) > 0

        ctx = kb.build_context(results)
        assert "Источник" in ctx
        assert len(ctx) > 0

    async def test_graceful_no_embedder(self, kb_config, sample_md):
        """Works without embedder (BM25 only)."""
        kb_inst = KnowledgeBase(config=kb_config, embedder=None)
        try:
            await kb_inst.ingest(sample_md)
            results = await kb_inst.search("chapter", mode="bm25")
            assert len(results) > 0
        finally:
            kb_inst.close()

    async def test_multiple_documents(self, kb, sample_md, sample_txt):
        await kb.ingest(sample_md)
        await kb.ingest(sample_txt)
        docs = await kb.list_documents()
        assert len(docs) == 2
        stats = await kb.get_stats()
        assert stats["documents"] == 2

    async def test_search_across_documents(self, kb, sample_md, sample_txt):
        await kb.ingest(sample_md)
        await kb.ingest(sample_txt)
        results = await kb.search("paragraph", top_k=10)
        # Should find results from the txt file
        sources = {r.source for r in results}
        assert len(sources) >= 1

    async def test_query_rewrite_integration(self, kb_config, mock_embedder,
                                              mock_provider, sample_md):
        """Full pipeline with query rewriting enabled."""
        kb_config["query_rewrite"] = True
        kb_inst = KnowledgeBase(
            config=kb_config, embedder=mock_embedder, provider=mock_provider)
        try:
            await kb_inst.ingest(sample_md)
            results = await kb_inst.search("how chapters are organized")
            # Should work even with query rewriting
            assert isinstance(results, list)
        finally:
            kb_inst.close()


# ═══════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════

class TestDataClasses:
    def test_kb_document(self):
        doc = KBDocument(id="1", name="Test", source_path="/tmp/test.pdf",
                        doc_hash="abc123")
        assert doc.name == "Test"
        assert doc.page_count == 0

    def test_kb_chunk(self):
        chunk = KBChunk(id="1", doc_id="d1", content="Hello world")
        assert chunk.chunk_type == "text"
        assert chunk.section_path == ""

    def test_kb_search_result(self):
        r = KBSearchResult(content="Test", score=0.9, source="Book",
                          page=5, section="Ch.1", chunk_id="c1")
        assert r.score == 0.9
        assert r.page == 5


# ══════════════════════════════════════
# Phase 1: Contextual Retrieval Tests
# ══════════════════════════════════════

class TestContextualRetrieval:
    """Tests for contextual retrieval (Phase 1)."""

    def test_schema_has_context_prefix(self, tmp_path):
        """context_prefix column exists."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        cols = [row[1] for row in kb.db.execute("PRAGMA table_info(kb_chunks)").fetchall()]
        assert "context_prefix" in cols
        kb.close()

    def test_chunk_dataclass_has_context_prefix(self):
        """KBChunk has context_prefix field."""
        chunk = KBChunk(id="1", doc_id="d1", content="test")
        assert chunk.context_prefix == ""

    def test_search_result_has_context_prefix(self):
        """KBSearchResult has context_prefix field."""
        r = KBSearchResult(content="t", score=1.0, source="s", page=1,
                           section="", chunk_id="c1")
        assert r.context_prefix == ""

    @pytest.mark.asyncio
    async def test_generate_context_prefix(self, tmp_path):
        """_generate_context_prefix returns LLM-generated context."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=MagicMock(
            text="This chunk discusses tax rates for small businesses."))

        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db"),
                            "contextual_retrieval": True},
                           provider=mock_provider)
        chunk = KBChunk(id="1", doc_id="d1", content="Налоговая ставка 13%",
                        section_path="Гл.3 > Ставки")

        prefix = await kb._generate_context_prefix(chunk, "Tax Guide")
        assert "tax" in prefix.lower() or len(prefix) > 0
        kb.close()

    def test_config_flags_defaults(self, tmp_path):
        """New config flags default to False."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        assert kb._contextual_retrieval is False
        assert kb._parent_child is False
        assert kb._self_rag is False
        assert kb._self_rag_threshold == 0.35
        kb.close()

    def test_build_context_with_prefix(self, tmp_path):
        """build_context shows context_prefix."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        results = [KBSearchResult(
            content="Some tax info",
            score=0.9,
            source="Tax Book",
            page=5,
            section="Ch.1",
            chunk_id="c1",
            context_prefix="This discusses personal income tax rates."
        )]
        ctx = kb.build_context(results)
        assert "Контекст:" in ctx
        assert "personal income tax" in ctx
        kb.close()


# ══════════════════════════════════════
# Phase 2: Parent-Child Retrieval Tests
# ══════════════════════════════════════

class TestParentChildRetrieval:
    """Tests for parent-child retrieval (Phase 2)."""

    def test_schema_has_parent_id(self, tmp_path):
        """parent_id column exists."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        cols = [row[1] for row in kb.db.execute("PRAGMA table_info(kb_chunks)").fetchall()]
        assert "parent_id" in cols
        kb.close()

    def test_chunk_dataclass_has_parent_id(self):
        """KBChunk has parent_id field."""
        chunk = KBChunk(id="1", doc_id="d1", content="test")
        assert chunk.parent_id is None

    def test_create_parent_child_chunks(self, tmp_path):
        """_create_parent_child_chunks splits parents into children."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db"),
                            "parent_child_retrieval": True})
        parents = [
            KBChunk(id="p1", doc_id="d1", content="A" * 1000,
                    section_path="Sec1", chunk_index=0),
        ]
        result = kb._create_parent_child_chunks(parents)

        parent_chunks = [c for c in result if c.chunk_type == "parent"]
        child_chunks = [c for c in result if c.chunk_type == "child"]

        assert len(parent_chunks) == 1
        assert len(child_chunks) >= 1
        assert all(c.parent_id == "p1" for c in child_chunks)
        kb.close()

    def test_resolve_parents(self, tmp_path):
        """_resolve_parents replaces child content with parent."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db"),
                            "parent_child_retrieval": True})
        # Insert parent
        kb.db.execute(
            "INSERT INTO kb_documents (id, name, doc_hash) VALUES ('d1', 'test', 'h1')")
        kb.db.execute(
            "INSERT INTO kb_chunks (id, doc_id, content, chunk_type, section_path) "
            "VALUES ('p1', 'd1', 'Full parent content here', 'parent', 'Sec1')")
        kb.db.execute(
            "INSERT INTO kb_chunks (id, doc_id, content, chunk_type, parent_id, section_path) "
            "VALUES ('c1', 'd1', 'Small child', 'child', 'p1', 'Sec1')")
        kb.db.commit()

        results = [{"chunk_id": "c1", "content": "Small child",
                     "section_path": "Sec1", "page_start": 0, "page_end": 0,
                     "score": 0.9, "doc_name": "test"}]
        resolved = kb._resolve_parents(results)
        assert len(resolved) == 1
        assert resolved[0]["content"] == "Full parent content here"
        kb.close()


# ══════════════════════════════════════
# Phase 3: Self-RAG / CRAG Tests
# ══════════════════════════════════════

class TestSelfRAG:
    """Tests for Self-RAG / CRAG (Phase 3)."""

    @pytest.mark.asyncio
    async def test_crag_rewrite(self, tmp_path):
        """_crag_rewrite returns rewritten query."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=MagicMock(
            text="What are the income tax rates for individuals in Russia?"))

        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db"),
                            "self_rag": True},
                           provider=mock_provider)

        rewritten = await kb._crag_rewrite(
            "налоги", [{"content": "irrelevant stuff"}])
        assert len(rewritten) > 0
        assert rewritten != "налоги"
        kb.close()

    @pytest.mark.asyncio
    async def test_crag_correction_high_confidence(self, tmp_path):
        """_crag_correction skips when confidence is high."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db"),
                            "self_rag": True, "self_rag_threshold": 0.3})
        results = [
            KBSearchResult(content="tax info", score=0.8, source="s",
                           page=1, section="", chunk_id="c1"),
        ]
        corrected = await kb._crag_correction("taxes", results, 5, "hybrid")
        assert corrected == results  # No change needed
        kb.close()

    @pytest.mark.asyncio
    async def test_crag_rewrite_no_provider(self, tmp_path):
        """_crag_rewrite returns original query without provider."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db"),
                            "self_rag": True})
        result = await kb._crag_rewrite("test", [{"content": "x"}])
        assert result == "test"
        kb.close()


# ══════════════════════════════════════
# Phase 5: Entity Graph Tests
# ══════════════════════════════════════

class TestEntityGraph:
    """Tests for Light GraphRAG (Phase 5)."""

    def test_entity_tables_exist(self, tmp_path):
        """Entity tables are created."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        tables = [row[0] for row in kb.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        assert "kb_entities" in tables
        assert "kb_entity_mentions" in tables
        kb.close()

    def test_entity_search_empty(self, tmp_path):
        """_entity_search returns empty list when no entities."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        results = kb._entity_search("ст. 123", top_k=10)
        assert results == []
        kb.close()

    def test_entity_search_finds_article(self, tmp_path):
        """_entity_search finds chunks via entity match."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        # Setup: doc, chunk, entity, mention
        kb.db.execute(
            "INSERT INTO kb_documents (id, name, doc_hash) VALUES ('d1', 'Law', 'h1')")
        kb.db.execute(
            "INSERT INTO kb_chunks (id, doc_id, content, chunk_type) "
            "VALUES ('c1', 'd1', 'Статья 123 устанавливает...', 'text')")
        kb.db.execute(
            "INSERT INTO kb_entities (id, name, entity_type, doc_id, count) "
            "VALUES ('e1', 'ст. 123', 'article', 'd1', 1)")
        kb.db.execute(
            "INSERT INTO kb_entity_mentions (entity_id, chunk_id) VALUES ('e1', 'c1')")
        kb.db.commit()

        results = kb._entity_search("ст. 123")
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        kb.close()

    @pytest.mark.asyncio
    async def test_list_entities(self, tmp_path):
        """list_entities returns entity list."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        kb.db.execute(
            "INSERT INTO kb_documents (id, name, doc_hash) VALUES ('d1', 'Law', 'h1')")
        kb.db.execute(
            "INSERT INTO kb_entities (id, name, entity_type, doc_id, count) "
            "VALUES ('e1', 'ст. 123', 'article', 'd1', 3)")
        kb.db.commit()

        entities = await kb.list_entities()
        assert len(entities) == 1
        assert entities[0]["name"] == "ст. 123"
        assert entities[0]["count"] == 3
        kb.close()

    @pytest.mark.asyncio
    async def test_list_entities_by_doc(self, tmp_path):
        """list_entities filters by doc_id."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        kb.db.execute(
            "INSERT INTO kb_documents (id, name, doc_hash) VALUES ('d1', 'Law', 'h1')")
        kb.db.execute(
            "INSERT INTO kb_documents (id, name, doc_hash) VALUES ('d2', 'Tax', 'h2')")
        kb.db.execute(
            "INSERT INTO kb_entities (id, name, entity_type, doc_id, count) "
            "VALUES ('e1', 'ст. 123', 'article', 'd1', 1)")
        kb.db.execute(
            "INSERT INTO kb_entities (id, name, entity_type, doc_id, count) "
            "VALUES ('e2', '13%', 'percentage', 'd2', 2)")
        kb.db.commit()

        entities = await kb.list_entities(doc_id="d2")
        assert len(entities) == 1
        assert entities[0]["name"] == "13%"
        kb.close()


# ══════════════════════════════════════
# Phase 6: RAGAS Evaluation Tests
# ══════════════════════════════════════

class TestRAGASEvaluation:
    """Tests for RAGAS quality metrics (Phase 6)."""

    def test_compute_quality_metrics(self, tmp_path):
        """_compute_quality_metrics returns valid scores."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        results = [
            KBSearchResult(content="налоговая ставка 13 процентов",
                           score=0.9, source="s", page=1, section="",
                           chunk_id="c1"),
            KBSearchResult(content="другая информация о налогах",
                           score=0.5, source="s", page=2, section="",
                           chunk_id="c2"),
        ]
        metrics = kb._compute_quality_metrics("налоговая ставка", results)
        assert "faithfulness" in metrics
        assert "relevancy" in metrics
        assert "context_precision" in metrics
        assert 0.0 <= metrics["faithfulness"] <= 1.0
        assert 0.0 <= metrics["relevancy"] <= 1.0
        assert 0.0 <= metrics["context_precision"] <= 1.0
        kb.close()

    def test_compute_quality_metrics_empty(self, tmp_path):
        """_compute_quality_metrics handles empty results."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        metrics = kb._compute_quality_metrics("test", [])
        assert metrics["faithfulness"] == 0.0
        assert metrics["relevancy"] == 0.0
        assert metrics["context_precision"] == 0.0
        kb.close()

    def test_log_query_with_metrics(self, tmp_path):
        """_log_query stores quality metrics."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        metrics = {"faithfulness": 0.8, "relevancy": 0.7, "context_precision": 0.6}
        kb._log_query("test query", ["test"], ["c1"], 100, metrics)

        row = kb.db.execute(
            "SELECT faithfulness_score, relevancy_score, context_precision "
            "FROM kb_query_log ORDER BY id DESC LIMIT 1").fetchone()
        assert row is not None
        assert abs(row["faithfulness_score"] - 0.8) < 0.01
        assert abs(row["relevancy_score"] - 0.7) < 0.01
        assert abs(row["context_precision"] - 0.6) < 0.01
        kb.close()

    @pytest.mark.asyncio
    async def test_get_quality_stats(self, tmp_path):
        """get_quality_stats returns aggregated metrics."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        # Log some queries with metrics
        kb._log_query("q1", [], ["c1"], 50,
                       {"faithfulness": 0.8, "relevancy": 0.9, "context_precision": 0.7})
        kb._log_query("q2", [], ["c2"], 60,
                       {"faithfulness": 0.6, "relevancy": 0.7, "context_precision": 0.5})

        stats = await kb.get_quality_stats()
        assert stats["queries_analyzed"] == 2
        assert abs(stats["avg_faithfulness"] - 0.7) < 0.01
        assert abs(stats["avg_relevancy"] - 0.8) < 0.01
        assert abs(stats["avg_context_precision"] - 0.6) < 0.01
        kb.close()

    def test_query_log_schema_has_metric_columns(self, tmp_path):
        """kb_query_log has metric columns."""
        kb = KnowledgeBase({"db_path": str(tmp_path / "kb.db")})
        cols = [row[1] for row in kb.db.execute(
            "PRAGMA table_info(kb_query_log)").fetchall()]
        assert "faithfulness_score" in cols
        assert "relevancy_score" in cols
        assert "context_precision" in cols
        kb.close()
