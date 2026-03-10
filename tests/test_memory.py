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

    def test_compressed_history_keeps_all_within_budget(self, memory_system):
        for i in range(10):
            memory_system.add_message("u1", "user" if i % 2 == 0 else "assistant", f"msg {i}")
        result = memory_system.get_compressed_history("u1")
        # All messages are small — they fit the budget, nothing dropped
        contents = [m["content"] for m in result]
        assert "msg 9" in contents   # Most recent must be present
        assert "msg 0" in contents   # Oldest also present (full context like Claude.ai)

    def test_compressed_history_trims_over_token_budget(self, memory_system):
        for i in range(10):
            memory_system.add_message("u1", "user" if i % 2 == 0 else "assistant", f"msg {i} " + "x" * 200)
        memory_system.config["max_history_tokens"] = 1  # force aggressive trim
        result = memory_system.get_compressed_history("u1")
        contents = [m["content"] for m in result]
        assert any("msg 9" in c for c in contents)  # Most recent must survive
        memory_system.config.pop("max_history_tokens", None)

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

    @pytest.mark.asyncio
    async def test_remember_and_recall(self, memory_system):
        await memory_system.remember("User's name is Alice", "u1", "fact", 0.8)
        results = memory_system.recall("Alice name", "u1")
        assert len(results) > 0
        assert "Alice" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_deduplication(self, memory_system):
        await memory_system.remember("User loves Python", "u1", "fact", 0.5)
        await memory_system.remember("User loves Python", "u1", "fact", 0.5)
        all_mems = memory_system.get_all_memories("u1")
        assert len(all_mems) == 1
        # Importance should have been bumped
        assert all_mems[0]["importance"] > 0.5

    @pytest.mark.asyncio
    async def test_forget(self, memory_system):
        await memory_system.remember("Secret info", "u1", "fact", 0.5)
        memory_system.forget("u1", "Secret")
        all_mems = memory_system.get_all_memories("u1")
        assert len(all_mems) == 0

    def test_recall_empty(self, memory_system):
        results = memory_system.recall("anything", "u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_user_isolation(self, memory_system):
        await memory_system.remember("Alice fact", "u1", "fact", 0.8)
        results = memory_system.recall("Alice", "u2")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_reinforce_recall_boosts_affinity_for_repeated_query(self, memory_system):
        first_id = await memory_system.remember("Preferred editor is Neovim", "u1", "fact", 0.45)
        second_id = await memory_system.remember("Preferred editor is VS Code", "u1", "fact", 0.72)

        before = memory_system.recall("preferred editor", "u1", top_k=2)
        assert before[0]["id"] == second_id

        boosted = memory_system.reinforce_recall(
            "preferred editor", "u1", [first_id], strength=1.0, source="test")
        assert boosted == 1

        after = memory_system.recall("preferred editor", "u1", top_k=2)
        assert after[0]["id"] == first_id
        all_mems = {m["id"]: m for m in memory_system.get_all_memories("u1")}
        assert all_mems[first_id]["importance"] > 0.45

    @pytest.mark.asyncio
    async def test_register_recall_feedback_penalizes_unused_candidates(self, memory_system):
        stale_id = await memory_system.remember("Deployment checklist includes smoke tests", "u1", "fact", 0.78)
        useful_id = await memory_system.remember("Deployment checklist includes rollback notes", "u1", "fact", 0.56)

        before = memory_system.recall("deployment checklist", "u1", top_k=2)
        assert before[0]["id"] == stale_id

        feedback = memory_system.register_recall_feedback(
            "deployment checklist",
            "u1",
            shown_ids=[stale_id, useful_id],
            used_ids=[useful_id],
            strength=1.0,
            source="test",
        )
        assert feedback["reinforced"] == 1
        assert feedback["penalized"] == 1

        after = memory_system.recall("deployment checklist", "u1", top_k=2)
        assert after[0]["id"] == useful_id

        all_mems = {m["id"]: m for m in memory_system.get_all_memories("u1")}
        assert all_mems[stale_id]["importance"] < 0.78
        assert all_mems[useful_id]["importance"] > 0.56

    @pytest.mark.asyncio
    async def test_recall_filters_cross_script_noise(self, memory_system):
        await memory_system.remember("프로젝트 개발 디렉터리가 아직 생성되지 않았습니다", "u1", "correction", 0.9)
        await memory_system.remember("Меня зовут Вячеслав", "u1", "fact", 0.9)

        results = memory_system.recall("как меня зовут", "u1", top_k=5)

        contents = [item["content"] for item in results]
        assert any("Вячеслав" in content for content in contents)
        assert all("프로젝트" not in content for content in contents)

    def test_thinking_cloud_upsert_deduplicates_and_builds_theme_edges(self, memory_system):
        first = memory_system.upsert_thinking_note(
            "u1",
            "idea",
            "Build a local-first agent workspace with stronger memory",
            themes=["local models", "memory"],
            confidence=0.72,
            strategic_importance=0.88,
        )
        second = memory_system.upsert_thinking_note(
            "u1",
            "idea",
            "Build a local-first agent workspace with stronger memory",
            themes=["memory"],
            confidence=0.8,
            strategic_importance=0.9,
        )
        assert first == second

        summary = memory_system.get_thinking_cloud_summary("u1", limit=6)
        assert summary["overview"]["total_notes"] >= 3  # idea + themes
        assert any(theme["label"] == "memory" for theme in summary["themes"])
        assert summary["directions"][0]["recurrence"] >= 2

    def test_thinking_cloud_recall_prioritizes_matching_constraints(self, memory_system):
        memory_system.upsert_thinking_note(
            "u1",
            "constraint",
            "Prefer local models even if they are slower, because cost matters less than privacy",
            themes=["local models", "privacy"],
            confidence=0.9,
            strategic_importance=0.95,
        )
        memory_system.upsert_thinking_note(
            "u1",
            "idea",
            "Make the dashboard more editorial and SaaS-like",
            themes=["dashboard", "design"],
            confidence=0.74,
            strategic_importance=0.7,
        )

        results = memory_system.recall_thinking_cloud("local models privacy", "u1", top_k=2)
        assert results
        assert results[0]["type"] == "constraint"
        assert "local models" in [theme.lower() for theme in results[0]["themes"]]


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
