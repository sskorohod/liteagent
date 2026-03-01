"""Tests for memory conflict detection and resolution."""

import sqlite3
import pytest
import hashlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from liteagent.memory import MemorySystem


@pytest.fixture
def memory_system(tmp_path):
    """Create a MemorySystem with conflict detection enabled."""
    config = {
        "memory": {
            "db_path": str(tmp_path / "test_memory.db"),
            "auto_learn": False,
        },
        "features": {
            "memory_conflict_detection": {
                "enabled": True,
                "similarity_threshold": 0.75,
                "auto_resolve": True,
            }
        },
    }
    ms = MemorySystem(config)
    return ms


@pytest.fixture
def memory_no_conflict(tmp_path):
    """MemorySystem with conflict detection disabled."""
    config = {
        "memory": {
            "db_path": str(tmp_path / "test_memory_nc.db"),
            "auto_learn": False,
        },
        "features": {},
    }
    ms = MemorySystem(config)
    return ms


class TestConflictDetection:
    """Test detect_memory_conflicts method."""

    def test_no_conflicts_empty_db(self, memory_system):
        conflicts = memory_system.detect_memory_conflicts(
            "User works at Google", "user1")
        assert conflicts == []

    def test_no_conflict_different_content(self, memory_system):
        """Different topics should not conflict."""
        memory_system.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("user1", "User likes Python", "fact", 0.5,
             hashlib.md5(b"user likes python").hexdigest(),
             datetime.now().isoformat(), datetime.now().isoformat()))
        memory_system.db.commit()

        conflicts = memory_system.detect_memory_conflicts(
            "User works at Google", "user1")
        assert conflicts == []

    def test_detects_replacement_conflict_keyword(self, memory_system):
        """Keyword fallback: 'works at Google' vs 'works at Apple' should conflict."""
        # Without embedder, uses keyword matching
        memory_system._embedder = None
        content_old = "User works at Google as engineer"
        memory_system.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("user1", content_old, "fact", 0.5,
             hashlib.md5(content_old.lower().encode()).hexdigest(),
             datetime.now().isoformat(), datetime.now().isoformat()))
        memory_system.db.commit()

        # High keyword overlap but different value
        conflicts = memory_system.detect_memory_conflicts(
            "User works at Apple as engineer", "user1",
            threshold=0.3)  # Lower threshold for keyword matching
        assert len(conflicts) >= 1
        assert conflicts[0]["conflict_type"] == "replacement"

    def test_detects_negation_conflict(self, memory_system):
        """'User likes cats' vs 'User does not like cats' should conflict."""
        memory_system._embedder = None
        content_old = "User likes cats very much"
        memory_system.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("user1", content_old, "fact", 0.5,
             hashlib.md5(content_old.lower().encode()).hexdigest(),
             datetime.now().isoformat(), datetime.now().isoformat()))
        memory_system.db.commit()

        conflicts = memory_system.detect_memory_conflicts(
            "User does not like cats very much", "user1",
            threshold=0.3)
        assert len(conflicts) >= 1
        assert conflicts[0]["conflict_type"] == "negation"

    def test_no_conflict_for_archived(self, memory_system):
        """Archived memories should not be considered for conflicts."""
        memory_system._embedder = None
        content_old = "User works at Google as engineer"
        memory_system.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("user1", content_old, "fact", 0.5,
             hashlib.md5(content_old.lower().encode()).hexdigest(),
             datetime.now().isoformat(), datetime.now().isoformat(),
             datetime.now().isoformat()))
        memory_system.db.commit()

        conflicts = memory_system.detect_memory_conflicts(
            "User works at Apple as engineer", "user1",
            threshold=0.3)
        assert conflicts == []


class TestContradictionDetection:
    """Test _detect_contradiction_type static method."""

    def test_negation_detected(self):
        result = MemorySystem._detect_contradiction_type(
            "User does not like spicy food",
            "User likes spicy food")
        assert result == "negation"

    def test_replacement_detected(self):
        result = MemorySystem._detect_contradiction_type(
            "User works at Apple as engineer",
            "User works at Google as engineer")
        assert result == "replacement"

    def test_no_contradiction(self):
        result = MemorySystem._detect_contradiction_type(
            "User likes Python programming",
            "User enjoys hiking outdoors")
        assert result is None

    def test_russian_negation(self):
        result = MemorySystem._detect_contradiction_type(
            "Пользователь не любит кофе утром",
            "Пользователь любит кофе утром")
        assert result == "negation"


class TestConflictResolution:
    """Test resolve_memory_conflict method."""

    @pytest.mark.asyncio
    async def test_resolve_returns_replace(self, memory_system):
        """Mock provider returns 'replace'."""
        mock_provider = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="replace")]
        mock_provider.complete = AsyncMock(return_value=mock_result)
        memory_system.provider = mock_provider

        action = await memory_system.resolve_memory_conflict(
            "User works at Apple",
            {"id": 1, "content": "User works at Google", "type": "fact"},
            "user1")
        assert action == "replace"

    @pytest.mark.asyncio
    async def test_resolve_returns_merge(self, memory_system):
        mock_provider = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="merge")]
        mock_provider.complete = AsyncMock(return_value=mock_result)
        memory_system.provider = mock_provider

        action = await memory_system.resolve_memory_conflict(
            "User also likes tea",
            {"id": 1, "content": "User likes coffee", "type": "preference"},
            "user1")
        assert action == "merge"

    @pytest.mark.asyncio
    async def test_resolve_falls_back_on_error(self, memory_system):
        """On LLM failure, should return 'keep_both'."""
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(side_effect=Exception("API error"))
        memory_system.provider = mock_provider

        action = await memory_system.resolve_memory_conflict(
            "User works at Apple",
            {"id": 1, "content": "User works at Google", "type": "fact"},
            "user1")
        assert action == "keep_both"

    @pytest.mark.asyncio
    async def test_resolve_no_provider(self, memory_system):
        """Without provider, should return 'keep_both'."""
        memory_system.provider = None
        action = await memory_system.resolve_memory_conflict(
            "new", {"id": 1, "content": "old", "type": "fact"}, "user1")
        assert action == "keep_both"

    @pytest.mark.asyncio
    async def test_resolve_invalid_action_fallback(self, memory_system):
        """If LLM returns invalid action, fallback to keep_both."""
        mock_provider = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="destroy")]
        mock_provider.complete = AsyncMock(return_value=mock_result)
        memory_system.provider = mock_provider

        action = await memory_system.resolve_memory_conflict(
            "new", {"id": 1, "content": "old", "type": "fact"}, "user1")
        assert action == "keep_both"


class TestApplyResolution:
    """Test _apply_conflict_resolution method."""

    def _insert_memory(self, ms, content, user_id="user1"):
        h = hashlib.md5(content.lower().strip().encode()).hexdigest()
        ms.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, content, "fact", 0.5, h,
             datetime.now().isoformat(), datetime.now().isoformat()))
        ms.db.commit()
        return ms.db.execute("SELECT last_insert_rowid()").fetchone()[0]

    def test_replace(self, memory_system):
        mem_id = self._insert_memory(memory_system, "User works at Google")
        memory_system._apply_conflict_resolution(
            "replace", "User works at Apple",
            {"id": mem_id, "content": "User works at Google", "type": "fact"},
            "user1")

        row = memory_system.db.execute(
            "SELECT content FROM memories WHERE id=?", (mem_id,)).fetchone()
        assert row[0] == "User works at Apple"

    def test_archive_old(self, memory_system):
        mem_id = self._insert_memory(memory_system, "User works at Google")
        memory_system._apply_conflict_resolution(
            "archive_old", "User works at Apple",
            {"id": mem_id, "content": "User works at Google", "type": "fact"},
            "user1")

        # Old should be archived
        old = memory_system.db.execute(
            "SELECT archived_at FROM memories WHERE id=?", (mem_id,)).fetchone()
        assert old[0] is not None

        # New should be inserted
        new = memory_system.db.execute(
            "SELECT content FROM memories WHERE content=?",
            ("User works at Apple",)).fetchone()
        assert new is not None

    def test_merge(self, memory_system):
        mem_id = self._insert_memory(memory_system, "User likes coffee")
        memory_system._apply_conflict_resolution(
            "merge", "User also likes tea",
            {"id": mem_id, "content": "User likes coffee", "type": "preference"},
            "user1")

        row = memory_system.db.execute(
            "SELECT content FROM memories WHERE id=?", (mem_id,)).fetchone()
        assert "coffee" in row[0]
        assert "tea" in row[0]

    def test_keep_both(self, memory_system):
        mem_id = self._insert_memory(memory_system, "User knows Python")
        memory_system._apply_conflict_resolution(
            "keep_both", "User knows JavaScript",
            {"id": mem_id, "content": "User knows Python", "type": "fact"},
            "user1")

        # Both should exist
        count = memory_system.db.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id='user1'").fetchone()[0]
        assert count == 2


class TestRecallFiltersArchived:
    """Test that recall() excludes archived memories."""

    def test_recall_excludes_archived(self, memory_system):
        memory_system._embedder = None
        # Active memory
        memory_system.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("user1", "User likes cats", "fact", 0.8,
             "hash_active", datetime.now().isoformat(), datetime.now().isoformat()))
        # Archived memory
        memory_system.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("user1", "User likes dogs", "fact", 0.8,
             "hash_archived", datetime.now().isoformat(),
             datetime.now().isoformat(), datetime.now().isoformat()))
        memory_system.db.commit()

        results = memory_system.recall("likes", "user1")
        contents = [r["content"] for r in results]
        assert "User likes cats" in contents
        assert "User likes dogs" not in contents


class TestRememberAsync:
    """Test async remember() with and without conflict detection."""

    @pytest.mark.asyncio
    async def test_remember_no_conflict_feature(self, memory_no_conflict):
        """Without conflict detection, remember inserts normally."""
        await memory_no_conflict.remember("Test fact", "user1", "fact", 0.5)
        row = memory_no_conflict.db.execute(
            "SELECT content FROM memories WHERE user_id='user1'").fetchone()
        assert row[0] == "Test fact"

    @pytest.mark.asyncio
    async def test_remember_dedup_still_works(self, memory_system):
        """Hash-based dedup should still work even with conflict detection."""
        await memory_system.remember("Same content", "user1", "fact", 0.5)
        await memory_system.remember("Same content", "user1", "fact", 0.5)
        count = memory_system.db.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id='user1'").fetchone()[0]
        assert count == 1

    @pytest.mark.asyncio
    async def test_remember_with_conflict_calls_resolve(self, memory_system):
        """When conflict is detected, resolution should be called."""
        memory_system._embedder = None

        # Insert existing memory
        memory_system.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("user1", "User works at Google as engineer", "fact", 0.5,
             hashlib.md5(b"user works at google as engineer").hexdigest(),
             datetime.now().isoformat(), datetime.now().isoformat()))
        memory_system.db.commit()

        # Mock provider to return "replace"
        mock_provider = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = [MagicMock(text="replace")]
        mock_provider.complete = AsyncMock(return_value=mock_result)
        memory_system.provider = mock_provider

        # Lower threshold for keyword matching
        memory_system._features_config["memory_conflict_detection"]["similarity_threshold"] = 0.3

        await memory_system.remember(
            "User works at Apple as engineer", "user1", "fact", 0.6)

        # Should have resolved by replacing
        rows = memory_system.db.execute(
            "SELECT content FROM memories WHERE user_id='user1' AND archived_at IS NULL"
        ).fetchall()
        contents = [r[0] for r in rows]
        assert "User works at Apple as engineer" in contents


class TestArchivedMemories:
    """Test get_archived_memories."""

    def test_get_archived_empty(self, memory_system):
        result = memory_system.get_archived_memories("user1")
        assert result == []

    def test_get_archived_returns_data(self, memory_system):
        now = datetime.now().isoformat()
        memory_system.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash,
               created_at, accessed_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("user1", "Old memory", "fact", 0.5, "test_hash",
             now, now, now))
        memory_system.db.commit()

        result = memory_system.get_archived_memories("user1")
        assert len(result) == 1
        assert result[0]["content"] == "Old memory"
        assert result[0]["archived_at"] is not None
