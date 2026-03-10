"""Tests for Phase 2: Episodic memory — episodes, turns, topic shift, recall."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from liteagent.memory import MemorySystem


@pytest.fixture
def ep_config(tmp_db):
    """Memory config with episodic memory enabled."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "keep_recent_messages": 6,
            "fts_enabled": True,
            "episodic_memory": True,
            "episode_topic_shift_threshold": 0.3,
            "max_episode_context": 3,
        }
    }


@pytest.fixture
def ep_memory(ep_config):
    """MemorySystem with episodic memory enabled."""
    ms = MemorySystem(ep_config)
    yield ms
    ms.close()


@pytest.fixture
def ep_disabled_config(tmp_db):
    """Memory config with episodic memory disabled (default)."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "episodic_memory": False,
        }
    }


@pytest.fixture
def ep_disabled(ep_disabled_config):
    ms = MemorySystem(ep_disabled_config)
    yield ms
    ms.close()


# ══════════════════════════════════════════
# EPISODE LIFECYCLE
# ══════════════════════════════════════════

class TestEpisodeLifecycle:
    """Start, add turns, close episodes."""

    def test_episodic_enabled(self, ep_memory):
        assert ep_memory._episodic_enabled() is True

    def test_episodic_disabled_by_default(self, ep_disabled):
        assert ep_disabled._episodic_enabled() is False

    def test_start_episode(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        assert ep_id is not None
        assert len(ep_id) == 36  # UUID format
        assert ep_memory.get_active_episode("u1") == ep_id

    def test_start_multiple_episodes_different_users(self, ep_memory):
        ep1 = ep_memory.start_episode("u1")
        ep2 = ep_memory.start_episode("u2")
        assert ep1 != ep2
        assert ep_memory.get_active_episode("u1") == ep1
        assert ep_memory.get_active_episode("u2") == ep2

    def test_ensure_episode_starts_new(self, ep_memory):
        assert ep_memory.get_active_episode("u1") is None
        ep_id = ep_memory.ensure_episode("u1")
        assert ep_id is not None
        # Second call returns the same episode
        assert ep_memory.ensure_episode("u1") == ep_id

    def test_add_episode_turn(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "Hello", "Hi there!", None)
        ep_memory.add_episode_turn(ep_id, "How are you?", "I'm good!", None)

        turns = ep_memory.get_episode_turns(ep_id)
        assert len(turns) == 2
        assert turns[0]["user_input"] == "Hello"
        assert turns[1]["user_input"] == "How are you?"
        assert turns[0]["turn_index"] == 0
        assert turns[1]["turn_index"] == 1

    def test_add_turn_with_tool_calls(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        tools = [{"name": "web_search", "input": {"query": "test"}}]
        ep_memory.add_episode_turn(ep_id, "Search for X", "Found it!", tools)

        ep = ep_memory.get_episode(ep_id)
        assert "web_search" in ep["tools"]

    def test_add_turn_updates_count(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "msg1", "resp1", None)
        ep_memory.add_episode_turn(ep_id, "msg2", "resp2", None)

        ep = ep_memory.get_episode(ep_id)
        assert ep["turn_count"] == 2

    def test_add_turn_nonexistent_episode(self, ep_memory):
        """Adding a turn to a nonexistent episode should not crash."""
        ep_memory.add_episode_turn("nonexistent-id", "msg", "resp", None)
        # No exception raised

    @pytest.mark.asyncio
    async def test_close_episode_without_provider(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "Hello", "Hi", None)
        ep_memory.add_episode_turn(ep_id, "Bye", "See ya!", None)

        await ep_memory.close_episode(ep_id, outcome="completed")

        ep = ep_memory.get_episode(ep_id)
        assert ep["closed_at"] is not None
        assert ep["outcome"] == "completed"
        # Without provider, summary is the first user message
        assert ep["summary"] is not None
        assert ep_memory.get_active_episode("u1") is None

    @pytest.mark.asyncio
    async def test_close_episode_with_mock_provider(self, ep_memory):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="User asked about Python and got a detailed explanation")]
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        ep_memory.provider = mock_provider

        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "Tell me about Python", "Python is...", None)
        ep_memory.add_episode_turn(ep_id, "More details", "Sure, Python was...", None)

        await ep_memory.close_episode(ep_id)

        ep = ep_memory.get_episode(ep_id)
        assert ep["summary"] == "User asked about Python and got a detailed explanation"

    @pytest.mark.asyncio
    async def test_close_already_closed(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "Hello", "Hi", None)
        await ep_memory.close_episode(ep_id)
        # Closing again should be a no-op
        await ep_memory.close_episode(ep_id)
        ep = ep_memory.get_episode(ep_id)
        assert ep["closed_at"] is not None


# ══════════════════════════════════════════
# EPISODE RETRIEVAL
# ══════════════════════════════════════════

class TestEpisodeRetrieval:
    """Get episode data."""

    def test_get_episode(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "Test", "Response", None)
        ep = ep_memory.get_episode(ep_id)
        assert ep is not None
        assert ep["id"] == ep_id
        assert ep["user_id"] == "u1"
        assert ep["turn_count"] == 1

    def test_get_nonexistent_episode(self, ep_memory):
        assert ep_memory.get_episode("nonexistent") is None

    def test_get_recent_episodes(self, ep_memory):
        for i in range(3):
            ep_memory.start_episode("u1")
        episodes = ep_memory.get_recent_episodes("u1")
        assert len(episodes) == 3

    def test_get_recent_episodes_user_isolation(self, ep_memory):
        ep_memory.start_episode("u1")
        ep_memory.start_episode("u2")
        assert len(ep_memory.get_recent_episodes("u1")) == 1
        assert len(ep_memory.get_recent_episodes("u2")) == 1


# ══════════════════════════════════════════
# EPISODE RECALL (SEARCH)
# ══════════════════════════════════════════

class TestEpisodeRecall:
    """Search past episodes by query."""

    @pytest.mark.asyncio
    async def test_recall_closed_episodes(self, ep_memory):
        # Create and close an episode with summary
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "Deploy to production", "Deployed successfully", None)
        ep_memory.add_episode_turn(ep_id, "Check status", "All good", None)
        # Manually set summary (since no LLM provider)
        ep_memory.db.execute(
            "UPDATE episodes SET summary = ?, closed_at = ? WHERE id = ?",
            ("User deployed to production and checked status", datetime.now().isoformat(), ep_id))
        ep_memory.db.commit()

        results = ep_memory.recall_episodes("deploy production", "u1")
        assert len(results) >= 1
        assert "deploy" in results[0]["summary"].lower()

    @pytest.mark.asyncio
    async def test_recall_empty(self, ep_memory):
        results = ep_memory.recall_episodes("anything", "u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_user_isolation(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.db.execute(
            "UPDATE episodes SET summary = ?, closed_at = ? WHERE id = ?",
            ("Secret episode", datetime.now().isoformat(), ep_id))
        ep_memory.db.commit()

        results = ep_memory.recall_episodes("Secret", "u2")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_recall_excludes_open_episodes(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.db.execute(
            "UPDATE episodes SET summary = ? WHERE id = ?",
            ("Open episode about testing", ep_id))
        ep_memory.db.commit()
        # Episode not closed — should not be recalled
        results = ep_memory.recall_episodes("testing", "u1")
        assert len(results) == 0


# ══════════════════════════════════════════
# TOPIC SHIFT DETECTION
# ══════════════════════════════════════════

class TestTopicShift:
    """Topic shift detection between turns."""

    def test_no_shift_without_active_episode(self, ep_memory):
        assert ep_memory.detect_topic_shift("u1", "Hello") is False

    def test_no_shift_without_embedder(self, ep_memory):
        """Without embedder, topic shift can't be detected."""
        ep_memory.start_episode("u1")
        assert ep_memory.detect_topic_shift("u1", "Totally different topic") is False

    def test_no_shift_empty_episode(self, ep_memory):
        """New episode with no turns — no shift."""
        ep_memory.start_episode("u1")
        assert ep_memory.detect_topic_shift("u1", "First message") is False

    def test_shift_detected_with_mock_embedder(self, ep_memory):
        """With embedder, similar messages should not shift, dissimilar should."""
        import numpy as np

        class MockEmbedder:
            dim = 3
            def __init__(self):
                self._call_count = 0
            def encode(self, text):
                # Return different vectors for different topics
                if "Python" in text or "programming" in text:
                    return np.array([1.0, 0.0, 0.0])
                elif "cooking" in text or "recipe" in text:
                    return np.array([0.0, 1.0, 0.0])
                else:
                    return np.array([0.5, 0.5, 0.0])

        ep_memory._embedder = MockEmbedder()
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "Tell me about Python programming", "Python is...", None)

        # Similar topic — no shift
        assert ep_memory.detect_topic_shift("u1", "More about Python programming") is False

        # Very different topic — shift
        assert ep_memory.detect_topic_shift("u1", "Give me a cooking recipe") is True


# ══════════════════════════════════════════
# EPISODE PRUNING
# ══════════════════════════════════════════

class TestEpisodePruning:
    """Prune old episode turns."""

    @pytest.mark.asyncio
    async def test_prune_old_turns(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "msg1", "resp1", None)
        ep_memory.add_episode_turn(ep_id, "msg2", "resp2", None)

        # Close and backdate
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        ep_memory.db.execute(
            "UPDATE episodes SET closed_at = ?, summary = ? WHERE id = ?",
            (old_date, "Old episode", ep_id))
        ep_memory.db.commit()

        turns_before = ep_memory.get_episode_turns(ep_id)
        assert len(turns_before) == 2

        ep_memory.prune_episode_turns(days=7)

        turns_after = ep_memory.get_episode_turns(ep_id)
        assert len(turns_after) == 0

        # Episode itself (with summary) should still exist
        ep = ep_memory.get_episode(ep_id)
        assert ep is not None
        assert ep["summary"] == "Old episode"

    @pytest.mark.asyncio
    async def test_prune_keeps_recent_turns(self, ep_memory):
        ep_id = ep_memory.start_episode("u1")
        ep_memory.add_episode_turn(ep_id, "recent", "msg", None)

        # Close with current date
        ep_memory.db.execute(
            "UPDATE episodes SET closed_at = ? WHERE id = ?",
            (datetime.now().isoformat(), ep_id))
        ep_memory.db.commit()

        ep_memory.prune_episode_turns(days=7)

        turns = ep_memory.get_episode_turns(ep_id)
        assert len(turns) == 1  # Not pruned — too recent


# ══════════════════════════════════════════
# TABLES SCHEMA
# ══════════════════════════════════════════

class TestEpisodicSchema:
    """Verify episodic tables exist."""

    def test_episodes_table_exists(self, ep_memory):
        row = ep_memory.db.execute(
            "SELECT name FROM sqlite_master WHERE name='episodes'"
        ).fetchone()
        assert row is not None

    def test_episode_turns_table_exists(self, ep_memory):
        row = ep_memory.db.execute(
            "SELECT name FROM sqlite_master WHERE name='episode_turns'"
        ).fetchone()
        assert row is not None

    def test_tables_exist_even_when_disabled(self, ep_disabled):
        """Tables are always created, feature is gated by config check."""
        row = ep_disabled.db.execute(
            "SELECT name FROM sqlite_master WHERE name='episodes'"
        ).fetchone()
        assert row is not None


# ══════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ══════════════════════════════════════════

class TestEpisodicBackwardCompat:
    """Ensure episodic features don't break existing memory."""

    @pytest.mark.asyncio
    async def test_memory_still_works_with_episodes(self, ep_memory):
        await ep_memory.remember("User likes Python", "u1", "fact", 0.8)
        results = ep_memory.recall("Python", "u1")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_memory_works_without_episodic(self, ep_disabled):
        await ep_disabled.remember("User likes Java", "u1", "fact", 0.8)
        results = ep_disabled.recall("Java", "u1")
        assert len(results) >= 1

    def test_old_fixture_unaffected(self, memory_system):
        """Original memory_system fixture should still work."""
        memory_system.add_message("u1", "user", "hello")
        history = memory_system.get_history("u1")
        assert len(history) == 1
