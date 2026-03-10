"""Tests for Phase 1 memory improvements: FTS5, hybrid recall, RRF fusion, memory tools."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from liteagent.memory import MemorySystem


@pytest.fixture
def memory_config_fts(tmp_db):
    """Memory config with FTS5 enabled."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "keep_recent_messages": 6,
            "fts_enabled": True,
            "search_mode": "hybrid",
        }
    }


@pytest.fixture
def memory_fts(memory_config_fts):
    """MemorySystem with FTS5 enabled."""
    ms = MemorySystem(memory_config_fts)
    yield ms
    ms.close()


# ══════════════════════════════════════════
# FTS5 INITIALIZATION
# ══════════════════════════════════════════

class TestFTS5Init:
    """FTS5 virtual table creation and backfill."""

    def test_fts_available(self, memory_fts):
        assert memory_fts._fts_available is True

    def test_fts_table_exists(self, memory_fts):
        row = memory_fts.db.execute(
            "SELECT name FROM sqlite_master WHERE name='memory_fts'"
        ).fetchone()
        assert row is not None

    def test_fts_disabled_by_config(self, tmp_db):
        config = {
            "memory": {"db_path": tmp_db, "auto_learn": False, "fts_enabled": False}
        }
        ms = MemorySystem(config)
        assert ms._fts_available is False
        ms.close()

    @pytest.mark.asyncio
    async def test_backfill_existing_memories(self, tmp_db):
        """Memories created before FTS5 init should be backfilled."""
        # Phase 1: create memory system WITHOUT FTS
        config_no_fts = {
            "memory": {"db_path": tmp_db, "auto_learn": False, "fts_enabled": False}
        }
        ms1 = MemorySystem(config_no_fts)
        await ms1.remember("Python is great", "u1", "fact", 0.8)
        await ms1.remember("User likes coffee", "u1", "preference", 0.7)
        ms1.close()

        # Phase 2: re-open WITH FTS — should backfill
        config_fts = {
            "memory": {"db_path": tmp_db, "auto_learn": False, "fts_enabled": True}
        }
        ms2 = MemorySystem(config_fts)
        assert ms2._fts_available is True

        # FTS should find backfilled memories
        results = ms2._fts_search("Python", "u1", top_k=5)
        assert len(results) >= 1
        assert any("Python" in r["content"] for r in results)
        ms2.close()


# ══════════════════════════════════════════
# FTS5 SEARCH
# ══════════════════════════════════════════

class TestFTS5Search:
    """BM25 keyword search via FTS5."""

    @pytest.mark.asyncio
    async def test_basic_fts_search(self, memory_fts):
        await memory_fts.remember("Alice works at Google", "u1", "fact", 0.8)
        await memory_fts.remember("Bob prefers Python", "u1", "preference", 0.7)
        results = memory_fts._fts_search("Google", "u1", top_k=5)
        assert len(results) >= 1
        assert "Google" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_fts_user_isolation(self, memory_fts):
        await memory_fts.remember("Alice fact", "u1", "fact", 0.8)
        await memory_fts.remember("Bob fact", "u2", "fact", 0.8)
        results_u1 = memory_fts._fts_search("fact", "u1", top_k=5)
        results_u2 = memory_fts._fts_search("fact", "u2", top_k=5)
        assert all("Alice" in r["content"] for r in results_u1)
        assert all("Bob" in r["content"] for r in results_u2)

    @pytest.mark.asyncio
    async def test_fts_empty_query(self, memory_fts):
        await memory_fts.remember("Some fact", "u1", "fact", 0.5)
        results = memory_fts._fts_search("", "u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_fts_no_results(self, memory_fts):
        await memory_fts.remember("Hello world", "u1", "fact", 0.5)
        results = memory_fts._fts_search("xyznonexistent", "u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_fts_score_positive(self, memory_fts):
        await memory_fts.remember("Machine learning is cool", "u1", "fact", 0.8)
        results = memory_fts._fts_search("machine learning", "u1")
        assert len(results) >= 1
        assert results[0]["score"] > 0


# ══════════════════════════════════════════
# FTS5 SYNC ON MUTATIONS
# ══════════════════════════════════════════

class TestFTS5Sync:
    """FTS5 index stays in sync with memory mutations."""

    @pytest.mark.asyncio
    async def test_forget_removes_from_fts(self, memory_fts):
        await memory_fts.remember("Secret password is 1234", "u1", "fact", 0.5)
        results_before = memory_fts._fts_search("password", "u1")
        assert len(results_before) >= 1

        memory_fts.forget("u1", "Secret")
        results_after = memory_fts._fts_search("password", "u1")
        assert len(results_after) == 0

    @pytest.mark.asyncio
    async def test_delete_memory_removes_from_fts(self, memory_fts):
        await memory_fts.remember("Temporary note", "u1", "note", 0.3)
        all_mems = memory_fts.get_all_memories("u1")
        assert len(all_mems) == 1
        mem_id = all_mems[0]["id"]

        memory_fts.delete_memory(mem_id)
        results = memory_fts._fts_search("Temporary", "u1")
        assert len(results) == 0


# ══════════════════════════════════════════
# HYBRID RECALL (RRF FUSION)
# ══════════════════════════════════════════

class TestHybridRecall:
    """Hybrid search combining vector + BM25 + temporal decay."""

    @pytest.mark.asyncio
    async def test_recall_with_fts_only(self, memory_fts):
        """When no embedder — recall should use FTS5 keyword search."""
        await memory_fts.remember("User lives in Moscow", "u1", "fact", 0.8)
        await memory_fts.remember("User likes tea", "u1", "preference", 0.6)
        results = memory_fts.recall("Moscow", "u1", top_k=5)
        assert len(results) >= 1
        assert "Moscow" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_recall_keyword_mode(self, tmp_db):
        """search_mode=keyword should only use FTS5/fallback."""
        config = {
            "memory": {
                "db_path": tmp_db, "auto_learn": False,
                "search_mode": "keyword", "fts_enabled": True,
            }
        }
        ms = MemorySystem(config)
        await ms.remember("Python programming language", "u1", "fact", 0.8)
        results = ms.recall("Python", "u1")
        assert len(results) >= 1
        ms.close()

    @pytest.mark.asyncio
    async def test_recall_fallback_no_fts_no_embedder(self, tmp_db):
        """With both FTS5 and embedder unavailable, should fall back to word overlap."""
        config = {
            "memory": {
                "db_path": tmp_db, "auto_learn": False,
                "fts_enabled": False, "search_mode": "hybrid",
            }
        }
        ms = MemorySystem(config)
        await ms.remember("Alice works at company", "u1", "fact", 0.8)
        results = ms.recall("Alice works", "u1")
        assert len(results) >= 1
        assert "Alice" in results[0]["content"]
        ms.close()

    @pytest.mark.asyncio
    async def test_recall_empty_db(self, memory_fts):
        results = memory_fts.recall("anything", "u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_returns_scored_results(self, memory_fts):
        await memory_fts.remember("Important fact about databases", "u1", "fact", 0.9)
        await memory_fts.remember("Minor note about weather", "u1", "note", 0.2)
        results = memory_fts.recall("databases", "u1", top_k=5)
        assert len(results) >= 1
        assert all("score" in r for r in results)
        assert all(r["score"] >= 0 for r in results)

    @pytest.mark.asyncio
    async def test_recall_touch_on_access(self, memory_fts):
        """Recalled memories should have accessed_at updated."""
        await memory_fts.remember("Touchable memory", "u1", "fact", 0.8)
        memory_fts.recall("Touchable", "u1")
        row = memory_fts.db.execute(
            "SELECT accessed_at FROM memories WHERE user_id='u1'"
        ).fetchone()
        assert row is not None
        assert row[0] is not None


# ══════════════════════════════════════════
# RRF FUSION
# ══════════════════════════════════════════

class TestRRFFusion:
    """Reciprocal Rank Fusion algorithm."""

    def test_rrf_disjoint_lists(self):
        vector = [{"id": 1, "content": "a", "score": 0.9},
                  {"id": 2, "content": "b", "score": 0.8}]
        keyword = [{"id": 3, "content": "c", "score": 0.95},
                   {"id": 4, "content": "d", "score": 0.85}]
        result = MemorySystem._rrf_fusion(vector, keyword, k=60)
        assert len(result) == 4
        # All items should have RRF scores
        assert all("score" in r for r in result)

    def test_rrf_overlapping_lists(self):
        vector = [{"id": 1, "content": "a", "score": 0.9},
                  {"id": 2, "content": "b", "score": 0.5}]
        keyword = [{"id": 1, "content": "a", "score": 0.8},
                   {"id": 3, "content": "c", "score": 0.7}]
        result = MemorySystem._rrf_fusion(vector, keyword, k=60)
        assert len(result) == 3
        # Item 1 appears in both lists — should be ranked higher
        assert result[0]["id"] == 1

    def test_rrf_empty_lists(self):
        result = MemorySystem._rrf_fusion([], [])
        assert result == []

    def test_rrf_one_empty(self):
        vector = [{"id": 1, "content": "a", "score": 0.9}]
        result = MemorySystem._rrf_fusion(vector, [])
        assert len(result) == 1


# ══════════════════════════════════════════
# VECTOR SEARCH
# ══════════════════════════════════════════

class TestVectorSearch:
    """_vector_search and _keyword_fallback_search."""

    @pytest.mark.asyncio
    async def test_keyword_fallback_search(self, memory_fts):
        await memory_fts.remember("Python is the best language", "u1", "fact", 0.8)
        await memory_fts.remember("Java is also good", "u1", "fact", 0.5)
        results = memory_fts._keyword_fallback_search("Python language", "u1")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_keyword_fallback_empty(self, memory_fts):
        results = memory_fts._keyword_fallback_search("nothing", "u1")
        assert results == []


# ══════════════════════════════════════════
# SESSION SUMMARY COMPRESSION
# ══════════════════════════════════════════

class TestSessionCompression:
    """LLM-based session summary compression."""

    def test_short_summary_no_compression(self, memory_fts):
        """Short summaries should not trigger compression."""
        memory_fts._update_session_summary("u1", "Discussed Python project")
        summary = memory_fts._get_session_summary("u1")
        assert "Python" in summary

    def test_summary_accumulation(self, memory_fts):
        """Multiple updates should accumulate."""
        memory_fts._update_session_summary("u1", "First topic")
        memory_fts._update_session_summary("u1", "Second topic")
        summary = memory_fts._get_session_summary("u1")
        assert "First" in summary
        assert "Second" in summary

    def test_long_summary_truncated_without_llm(self, memory_fts):
        """Without LLM provider, long summaries should be truncated."""
        long_text = "Important context. " * 100  # ~1900 chars
        memory_fts._update_session_summary("u1", long_text)
        summary = memory_fts._get_session_summary("u1")
        assert len(summary) <= 800

    @pytest.mark.asyncio
    async def test_async_compress_summary(self, memory_fts):
        """_compress_summary should return truncated text without provider."""
        result = await memory_fts._compress_summary("x" * 700)
        assert len(result) <= 500

    @pytest.mark.asyncio
    async def test_async_compress_with_mock_provider(self, memory_fts):
        """_compress_summary should use LLM when provider available."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Compressed summary about topics")]
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        memory_fts.provider = mock_provider

        result = await memory_fts._compress_summary("x" * 700)
        assert result == "Compressed summary about topics"
        mock_provider.complete.assert_called_once()


# ══════════════════════════════════════════
# MEMORY METADATA HELPER
# ══════════════════════════════════════════

class TestMemoryMetadata:
    """_get_memory_metadata helper."""

    @pytest.mark.asyncio
    async def test_get_metadata(self, memory_fts):
        await memory_fts.remember("Test fact", "u1", "fact", 0.75)
        all_mems = memory_fts.get_all_memories("u1")
        assert len(all_mems) == 1
        mid = all_mems[0]["id"]

        meta = memory_fts._get_memory_metadata("u1", [mid])
        assert mid in meta
        assert meta[mid]["type"] == "fact"
        assert meta[mid]["importance"] == pytest.approx(0.75)

    def test_get_metadata_empty(self, memory_fts):
        meta = memory_fts._get_memory_metadata("u1", [])
        assert meta == {}

    @pytest.mark.asyncio
    async def test_get_metadata_wrong_user(self, memory_fts):
        await memory_fts.remember("Secret", "u1", "fact", 0.9)
        all_mems = memory_fts.get_all_memories("u1")
        mid = all_mems[0]["id"]
        meta = memory_fts._get_memory_metadata("u2", [mid])
        assert meta == {}


# ══════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ══════════════════════════════════════════

class TestBackwardCompat:
    """Ensure old behavior still works with new code."""

    @pytest.mark.asyncio
    async def test_old_remember_recall_still_works(self, memory_system):
        """memory_system fixture uses old config (no FTS keys) — should still work."""
        await memory_system.remember("User's name is Alice", "u1", "fact", 0.8)
        results = memory_system.recall("Alice name", "u1")
        assert len(results) > 0
        assert "Alice" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_old_dedup_still_works(self, memory_system):
        await memory_system.remember("User loves Python", "u1", "fact", 0.5)
        await memory_system.remember("User loves Python", "u1", "fact", 0.5)
        all_mems = memory_system.get_all_memories("u1")
        assert len(all_mems) == 1

    @pytest.mark.asyncio
    async def test_old_forget_still_works(self, memory_system):
        await memory_system.remember("Secret info", "u1", "fact", 0.5)
        memory_system.forget("u1", "Secret")
        assert memory_system.get_all_memories("u1") == []

    def test_old_recall_empty(self, memory_system):
        results = memory_system.recall("anything", "u1")
        assert results == []
