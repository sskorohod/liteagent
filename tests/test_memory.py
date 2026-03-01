"""Tests for the 4-layer memory system."""

import pytest
from datetime import datetime, timedelta


class TestConversationMemory:
    """L1: In-memory conversation buffer."""

    def test_add_and_get_history(self, memory_system):
        memory_system.add_message("u1", "user", "hello")
        memory_system.add_message("u1", "assistant", "hi there")
        history = memory_system.get_history("u1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["content"] == "hi there"

    def test_user_isolation(self, memory_system):
        memory_system.add_message("u1", "user", "msg for u1")
        memory_system.add_message("u2", "user", "msg for u2")
        assert len(memory_system.get_history("u1")) == 1
        assert len(memory_system.get_history("u2")) == 1
        assert memory_system.get_history("u1")[0]["content"] == "msg for u1"

    def test_compressed_history_within_limit(self, memory_system):
        for i in range(4):
            memory_system.add_message("u1", "user", f"msg {i}")
        result = memory_system.get_compressed_history("u1")
        # 4 messages < 6 (keep_recent), so all should be returned
        assert any(m["content"] == "msg 0" for m in result)

    def test_compressed_history_drops_old(self, memory_system):
        for i in range(10):
            memory_system.add_message("u1", "user" if i % 2 == 0 else "assistant", f"msg {i}")
        result = memory_system.get_compressed_history("u1")
        # Should have at most 6 recent + possible summary prefix
        contents = [m["content"] for m in result]
        assert "msg 9" in contents  # Most recent should be there
        assert "msg 0" not in contents  # Oldest should be dropped

    def test_clear_conversation(self, memory_system):
        memory_system.add_message("u1", "user", "hello")
        memory_system.clear_conversation("u1")
        assert memory_system.get_history("u1") == []


class TestScopedState:
    """L2: Scoped key-value state."""

    def test_session_state(self, memory_system):
        memory_system.set_state("temp_key", "value1")
        assert memory_system.get_state("temp_key") == "value1"

    def test_user_state(self, memory_system):
        memory_system.set_state("user:pref", "dark_mode", user_id="u1")
        assert memory_system.get_state("user:pref", user_id="u1") == "dark_mode"
        assert memory_system.get_state("user:pref", user_id="u2") is None

    def test_app_state(self, memory_system):
        memory_system.set_state("app:version", "1.0")
        assert memory_system.get_state("app:version") == "1.0"


class TestSemanticMemory:
    """L3: Persistent semantic memory."""

    def test_remember_and_recall(self, memory_system):
        memory_system.remember("User's name is Alice", "u1", "fact", 0.8)
        results = memory_system.recall("Alice name", "u1")
        assert len(results) > 0
        assert "Alice" in results[0]["content"]

    def test_deduplication(self, memory_system):
        memory_system.remember("User loves Python", "u1", "fact", 0.5)
        memory_system.remember("User loves Python", "u1", "fact", 0.5)
        all_mems = memory_system.get_all_memories("u1")
        assert len(all_mems) == 1
        # Importance should have been bumped
        assert all_mems[0]["importance"] > 0.5

    def test_forget(self, memory_system):
        memory_system.remember("Secret info", "u1", "fact", 0.5)
        memory_system.forget("u1", "Secret")
        all_mems = memory_system.get_all_memories("u1")
        assert len(all_mems) == 0

    def test_recall_empty(self, memory_system):
        results = memory_system.recall("anything", "u1")
        assert results == []

    def test_recall_user_isolation(self, memory_system):
        memory_system.remember("Alice fact", "u1", "fact", 0.8)
        results = memory_system.recall("Alice", "u2")
        assert len(results) == 0


class TestUsageTracking:
    """Usage stats and cost tracking."""

    def test_track_and_get_cost(self, memory_system):
        class MockUsage:
            input_tokens = 1000
            output_tokens = 500
            cache_read_input_tokens = 0

        memory_system.track_usage("u1", "claude-sonnet-4-20250514", MockUsage(), cost_usd=0.05)
        cost = memory_system.get_today_cost()
        assert cost == pytest.approx(0.05)

    def test_usage_summary(self, memory_system):
        class MockUsage:
            input_tokens = 100
            output_tokens = 50
            cache_read_input_tokens = 0

        memory_system.track_usage("u1", "claude-haiku-4-5-20251001", MockUsage(), 0.01)
        memory_system.track_usage("u1", "claude-haiku-4-5-20251001", MockUsage(), 0.01)
        summary = memory_system.get_usage_summary(days=1)
        assert len(summary) == 1
        assert summary[0]["calls"] == 2


class TestRecencyScore:
    """Helper: recency scoring."""

    def test_recent_memory_high_score(self, memory_system):
        score = memory_system._recency_score(datetime.now().isoformat())
        assert score > 0.9

    def test_old_memory_low_score(self, memory_system):
        old_date = (datetime.now() - timedelta(days=300)).isoformat()
        score = memory_system._recency_score(old_date)
        assert score < 0.2

    def test_invalid_date_returns_default(self, memory_system):
        assert memory_system._recency_score("not-a-date") == 0.5
