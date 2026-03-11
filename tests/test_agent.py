"""Tests for the agent core logic (no API calls)."""

import asyncio
import os
import pytest
from unittest.mock import AsyncMock

from liteagent.agent import LiteAgent, MODEL_PRICING
from liteagent.hooks import HookContext
from liteagent.providers import LLMResponse, TextBlock


class TestModelSelection:
    """Cascade routing model selection."""

    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 3,
                "default_model": "claude-sonnet-4-20250514",
                "models": {
                    "simple": "claude-haiku-4-5-20251001",
                    "medium": "claude-sonnet-4-20250514",
                    "complex": "claude-opus-4-20250115",
                }
            },
            "cost": {"cascade_routing": True, "budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_simple_question_scores_low(self, agent):
        score = agent._complexity_score("What time is it?")
        assert score < 1  # Trivial question

    def test_simple_question_routes_to_haiku_or_medium(self, agent):
        # Tool-capability guard may promote simple→medium when tools registered
        model = agent._select_model("What time is it?")
        assert "haiku" in model or "sonnet" in model

    def test_medium_task_routes_to_sonnet(self, agent):
        model = agent._select_model("Write a function to parse JSON")
        assert "sonnet" in model

    def test_complex_task_routes_to_opus(self, agent):
        model = agent._select_model("Analyze the architecture and refactor the payment module")
        assert "opus" in model

    def test_long_input_increases_complexity(self, agent):
        long_input = "Please help with: " + "a" * 600
        model = agent._select_model(long_input)
        # Long input should at least get medium
        assert "haiku" not in model

    def test_code_detected_as_medium(self, agent):
        score = agent._complexity_score("What does this do?\n```python\ndef foo(): pass\n```")
        assert score >= 1  # Code fences bump to medium+

    def test_multipart_request_complex(self, agent):
        score = agent._complexity_score("1. Fix the bug\n2. Add tests\n3. Update docs")
        assert score >= 2  # 3 numbered items = complex

    def test_word_boundary_no_false_positive(self, agent):
        # "fix" should not match in "prefix"
        score_prefix = agent._complexity_score("Check the prefix value")
        score_fix = agent._complexity_score("Fix the broken function")
        assert score_fix > score_prefix

    def test_cross_provider_tier_model_is_normalized(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-qwen-key")
        config = {
            "agent": {
                "provider": "qwen",
                "default_model": "qwen-plus",
                "models": {
                    "simple": "qwen3-coder:30b",
                    "medium": "qwen3-coder:30b",
                    "complex": "qwen3-coder:30b",
                },
            },
            "cost": {"cascade_routing": True, "budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        agent = LiteAgent(config)
        try:
            assert agent.models["simple"] == "ollama:qwen3-coder:30b"
            assert agent.models["medium"] == "ollama:qwen3-coder:30b"
            assert agent.models["complex"] == "ollama:qwen3-coder:30b"
        finally:
            agent.memory.close()

    def test_resolve_requested_model_infers_provider_for_bare_local_model(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-qwen-key")
        config = {
            "agent": {
                "provider": "qwen",
                "default_model": "qwen-plus",
            },
            "cost": {"cascade_routing": True, "budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        agent = LiteAgent(config)
        try:
            bare = agent._resolve_requested_model("qwen3-coder:30b")
            assert bare == "qwen3-coder:30b"
            assert type(agent.provider).__name__ == "OllamaProvider"
        finally:
            agent.memory.close()

    @pytest.mark.asyncio
    async def test_select_model_for_request_uses_advisor_when_available(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 3,
                "default_model": "claude-sonnet-4-20250514",
                "models": {
                    "simple": "claude-haiku-4-5-20251001",
                    "medium": "claude-sonnet-4-20250514",
                    "complex": "claude-opus-4-20250115",
                }
            },
            "cost": {
                "cascade_routing": True,
                "budget_daily_usd": 100.0,
                "intelligent_routing": {"enabled": True, "use_llm": True, "min_complexity": 1},
            },
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        agent = LiteAgent(config)
        try:
            agent._call_routing_advisor = AsyncMock(return_value={
                "model_spec": "claude-opus-4-20250115",
                "tier": "complex",
                "decision_source": "advisor",
                "objective": "quality",
                "reason": "Strongest reasoning candidate",
                "gap": "none",
                "recommendation": "",
                "advisor_model": "claude-haiku-4-5-20251001",
            })
            choice = await agent._select_model_for_request(
                "Analyze the architecture and refactor the payment module"
            )
            assert choice["model"] == "claude-opus-4-20250115"
            assert choice["decision_source"] == "advisor"
            assert choice["objective"] == "quality"
        finally:
            agent.memory.close()

    def test_cascade_recommendations_detect_same_model_gap(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 3,
                "provider": "ollama",
                "default_model": "qwen2.5:latest",
                "models": {
                    "simple": "qwen2.5:latest",
                    "medium": "qwen2.5:latest",
                    "complex": "qwen2.5:latest",
                }
            },
            "cost": {"cascade_routing": True, "budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        agent = LiteAgent(config)
        try:
            recs = agent.get_cascade_dashboard_state()["recommendations"]
            gaps = {item["gap"] for item in recs}
            assert "diversity" in gaps
            assert "strong_reasoning" in gaps
        finally:
            agent.memory.close()


class TestCostCalculation:
    """Token cost calculation."""

    def test_sonnet_cost(self):
        class MockUsage:
            input_tokens = 1_000_000
            output_tokens = 1_000_000
            cache_read_input_tokens = 0

        cost = LiteAgent._calculate_cost("claude-sonnet-4-20250514", MockUsage())
        expected = 3.00 + 15.00  # $3/M input + $15/M output
        assert cost == pytest.approx(expected)

    def test_haiku_with_cache(self):
        class MockUsage:
            input_tokens = 100_000
            output_tokens = 50_000
            cache_read_input_tokens = 500_000

        cost = LiteAgent._calculate_cost("claude-haiku-4-5-20251001", MockUsage())
        expected = 0.08 + 0.20 + 0.04  # input + output + cache
        assert cost == pytest.approx(expected)

    def test_unknown_model_falls_back(self):
        class MockUsage:
            input_tokens = 1000
            output_tokens = 1000
            cache_read_input_tokens = 0

        # Should not raise, falls back to Sonnet pricing
        cost = LiteAgent._calculate_cost("unknown-model-123", MockUsage())
        assert cost > 0


class TestCascadeTier:
    """Cascade tier mapping and history."""

    def test_tier_for_score_simple(self):
        assert LiteAgent._tier_for_score(0) == "simple"
        assert LiteAgent._tier_for_score(-1) == "simple"

    def test_tier_for_score_medium(self):
        assert LiteAgent._tier_for_score(1) == "medium"
        assert LiteAgent._tier_for_score(2) == "medium"

    def test_tier_for_score_complex(self):
        assert LiteAgent._tier_for_score(3) == "complex"
        assert LiteAgent._tier_for_score(5) == "complex"

    def test_record_cascade_decision(self):
        # Clear history
        LiteAgent._cascade_history = []
        LiteAgent._record_cascade_decision("claude-haiku-4-5-20251001", "simple", 0)
        LiteAgent._record_cascade_decision("claude-sonnet-4-20250514", "medium", 2)
        assert len(LiteAgent._cascade_history) == 2
        assert LiteAgent._cascade_history[0]["tier"] == "simple"
        assert LiteAgent._cascade_history[1]["model"] == "claude-sonnet-4-20250514"

    def test_cascade_history_max_cap(self):
        LiteAgent._cascade_history = []
        for i in range(60):
            LiteAgent._record_cascade_decision(f"model-{i}", "simple", 0)
        assert len(LiteAgent._cascade_history) == LiteAgent._CASCADE_HISTORY_MAX

    def test_get_cascade_summary(self):
        LiteAgent._cascade_history = []
        LiteAgent._record_cascade_decision("haiku", "simple", 0)
        LiteAgent._record_cascade_decision("haiku", "simple", -1)
        LiteAgent._record_cascade_decision("sonnet", "medium", 2)
        LiteAgent._record_cascade_decision("opus", "complex", 4)
        summary = LiteAgent.get_cascade_summary()
        assert summary["tier_counts"]["simple"] == 2
        assert summary["tier_counts"]["medium"] == 1
        assert summary["tier_counts"]["complex"] == 1
        assert summary["total_decisions"] == 4
        assert summary["last_decision"]["model"] == "opus"

    def test_get_cascade_history(self):
        LiteAgent._cascade_history = []
        LiteAgent._record_cascade_decision("haiku", "simple", 0)
        history = LiteAgent.get_cascade_history()
        assert len(history) == 1
        assert "timestamp" in history[0]


class TestTextExtraction:
    """Response text extraction."""

    def test_extract_single_text_block(self):
        class MockBlock:
            type = "text"
            text = "Hello world"

        class MockResponse:
            content = [MockBlock()]

        result = LiteAgent._extract_text(MockResponse())
        assert result == "Hello world"

    def test_extract_multiple_blocks(self):
        class TextBlock:
            type = "text"
            def __init__(self, t): self.text = t

        class ToolBlock:
            type = "tool_use"

        class MockResponse:
            content = [TextBlock("Part 1"), ToolBlock(), TextBlock("Part 2")]

        result = LiteAgent._extract_text(MockResponse())
        assert "Part 1" in result
        assert "Part 2" in result


class TestCommands:
    """Agent slash commands."""

    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {"max_iterations": 3},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_help_command(self, agent):
        result = agent.handle_command("/help")
        assert result is not None
        assert "/memories" in result
        assert "/usage" in result

    def test_clear_command(self, agent):
        agent.memory.add_message("u1", "user", "hello")
        result = agent.handle_command("/clear", "u1")
        assert "Conversation cleared" in result
        assert agent.memory.get_history("u1") == []

    def test_unknown_returns_none(self, agent):
        result = agent.handle_command("not a command")
        assert result is None

    def test_memories_empty(self, agent):
        result = agent.handle_command("/memories")
        assert "No memories" in result


class TestUserIdResolution:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {"max_iterations": 3},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_dashboard_alias_stays_when_no_primary_user(self, agent):
        assert agent.resolve_user_id("dashboard-user") == "dashboard-user"

    @pytest.mark.asyncio
    async def test_dashboard_alias_resolves_to_primary_tg_user(self, agent):
        await agent.memory.remember("User's name is Влад", "tg-456", "fact", 0.8)
        agent.memory.add_message("tg-456", "user", "привет")
        assert agent.resolve_user_id("dashboard-user") == "tg-456"

    @pytest.mark.asyncio
    async def test_handle_command_uses_resolved_user(self, agent):
        await agent.memory.remember("User likes coffee", "tg-456", "fact", 0.8)
        agent.memory.add_message("tg-456", "user", "hello")
        result = agent.handle_command("/memories", "dashboard-user")
        assert result is not None
        assert "coffee" in result.lower()

    def test_persistent_identity_mapping_preferred(self, agent):
        agent.memory.set_user_alias("dashboard-user", "tg-999", source="test")
        assert agent.resolve_user_id("dashboard-user") == "tg-999"

    @pytest.mark.asyncio
    async def test_safe_extract_uses_queue_when_daemon_enabled(self, agent, monkeypatch):
        monkeypatch.setattr(agent.memory, "extract_and_learn", AsyncMock())
        monkeypatch.setattr(agent.memory, "_memory_exchange_daemon_enabled", lambda: True)
        enqueue = AsyncMock(return_value={"status": "queued"})
        run_cycle = AsyncMock(return_value={"status": "ok"})
        monkeypatch.setattr(agent.memory, "enqueue_memory_exchange_intent", enqueue)
        monkeypatch.setattr(agent.memory, "run_memory_exchange_cycle", run_cycle)

        await agent._safe_extract("hello", "world", "u1")

        assert enqueue.await_count == 1
        assert run_cycle.await_count == 0


class TestKBAutoContext:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {"max_iterations": 2},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
            "knowledge_base": {
                "enabled": True,
                "search_mode": "hybrid",
                "auto_context_top_k": 3,
                "auto_context_max_chars": 1000,
                "auto_context_mode": "always",
            },
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    @pytest.mark.asyncio
    async def test_auto_retrieve_kb_context_truncates_and_uses_config(self, agent):
        class DummyKB:
            def __init__(self):
                self.last = None

            async def search(self, query, top_k=6, mode=None):
                self.last = (query, top_k, mode)
                return [object()]

            def build_context(self, results):
                return "X" * 5000

        dummy = DummyKB()
        agent._knowledge_base = dummy
        query = "what is classical logic?"
        ctx = await agent._auto_retrieve_kb_context(query)

        assert dummy.last == (query, 3, "hybrid")
        assert ctx.endswith("...[truncated]")
        assert len(ctx) <= 1015

    @pytest.mark.asyncio
    async def test_auto_retrieve_kb_context_skips_trivial_greeting(self, agent):
        class DummyKB:
            def __init__(self):
                self.called = False

            async def search(self, query, top_k=6, mode=None):
                self.called = True
                return [object()]

            def build_context(self, results):
                return "x"

        dummy = DummyKB()
        agent._knowledge_base = dummy
        ctx = await agent._auto_retrieve_kb_context("привет")
        assert ctx == ""
        assert dummy.called is False

    @pytest.mark.asyncio
    async def test_auto_retrieve_kb_context_skips_personal_memory_query(self, agent):
        class DummyKB:
            def __init__(self):
                self.called = False

            async def search(self, query, top_k=6, mode=None):
                self.called = True
                return [object()]

            def build_context(self, results):
                return "x"

        dummy = DummyKB()
        agent._knowledge_base = dummy
        ctx = await agent._auto_retrieve_kb_context("Как меня зовут?")
        assert ctx == ""
        assert dummy.called is False

    def test_inject_kb_context_text_and_multimodal(self):
        injected_text = LiteAgent._inject_kb_context("What is logic?", "KB DATA")
        assert "<kb_context>" in injected_text
        assert "Вопрос пользователя" in injected_text
        assert "персональных вопросов" in injected_text

        injected_blocks = LiteAgent._inject_kb_context(
            [{"type": "text", "text": "What is logic?"}],
            "KB DATA",
        )
        assert isinstance(injected_blocks, list)
        assert injected_blocks[0]["type"] == "text"
        assert "<kb_context>" in injected_blocks[0]["text"]

    @pytest.mark.asyncio
    async def test_auto_retrieve_kb_context_on_demand_needs_doc_marker(self, agent):
        class DummyKB:
            def __init__(self):
                self.called = False

            async def search(self, query, top_k=6, mode=None):
                self.called = True
                return [object()]

            def build_context(self, results):
                return "context"

        dummy = DummyKB()
        agent._knowledge_base = dummy
        # Simulate default mode (no explicit mode key) -> on_demand
        agent.config["knowledge_base"].pop("auto_context_mode", None)

        ctx_plain = await agent._auto_retrieve_kb_context("Что такое логика?")
        assert ctx_plain == ""
        assert dummy.called is False

        ctx_doc = await agent._auto_retrieve_kb_context(
            "Что такое логика в open-logic-complete.pdf?"
        )
        assert ctx_doc == "context"
        assert dummy.called is True

    def test_prune_kb_tools_for_personal_query(self, agent):
        tools = [
            {"name": "memory_search"},
            {"name": "kb_search"},
            {"name": "kb_list"},
            {"name": "web_search"},
        ]
        filtered = agent._prune_kb_tools_for_personal_query(
            tools, "как меня зовут?"
        )
        names = [t["name"] for t in filtered]
        assert "memory_search" in names
        assert "web_search" in names
        assert "kb_search" not in names
        assert "kb_list" not in names


class TestMediaUnderstanding:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 2,
                "provider": "openai",
                "default_model": "gpt-4o-mini",
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {
                "builtin": [],
                "media_understanding": {
                    "enabled": True,
                    "max_images": 2,
                    "max_documents": 1,
                    "max_tokens": 180,
                },
            },
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    @pytest.mark.asyncio
    async def test_run_multimodal_injects_image_summary_and_preserves_image_block(self, agent, monkeypatch):
        calls = []

        class Resp:
            def __init__(self, text):
                self.content = [type("Block", (), {"type": "text", "text": text})()]
                self.stop_reason = "end_turn"
                self.usage = type(
                    "Usage",
                    (),
                    {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                )()

        async def fake_complete(model, max_tokens, messages, system=None, tools=None, temperature=None):
            calls.append({
                "model": model,
                "messages": messages,
                "system": system,
            })
            if len(calls) == 1:
                return Resp("A login screen with email and password fields.")
            return Resp("Это экран логина.")

        monkeypatch.setattr(agent, "_vision_model_candidates", lambda requested_model="": ["vision-main"])
        agent.provider.complete = fake_complete

        response = await agent.run([
            {"type": "text", "text": "Что на этом скриншоте?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
        ], "u1")

        assert response == "Это экран логина."
        assert len(calls) == 2
        final_user_content = calls[1]["messages"][-1]["content"]
        assert isinstance(final_user_content, list)
        assert final_user_content[0]["type"] == "text"
        assert final_user_content[1]["type"] == "text"
        assert "Auto media understanding" in final_user_content[1]["text"]
        assert "login screen" in final_user_content[1]["text"]
        # After pre-analysis, raw image blocks are stripped so the main text LLM
        # works from the injected text summary (not re-processed as a vision task).
        assert not any(block.get("type") == "image" for block in final_user_content)
        assert calls[0]["model"] == "vision-main"
        assert calls[1]["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_run_multimodal_labels_screenshot_summary(self, agent):
        calls = []

        class Resp:
            def __init__(self, text):
                self.content = [type("Block", (), {"type": "text", "text": text})()]
                self.stop_reason = "end_turn"
                self.usage = type(
                    "Usage",
                    (),
                    {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                )()

        async def fake_complete(model, max_tokens, messages, system=None, tools=None, temperature=None):
            calls.append(messages)
            if len(calls) == 1:
                return Resp("Dashboard screenshot with a red warning banner.")
            return Resp("Похоже на dashboard.")

        agent.provider.complete = fake_complete

        response = await agent.run([
            {"type": "text", "text": "Что на этом скриншоте dashboard?"},
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "abc123", "filename": "screenshot.png"
            }},
        ], "u1")

        assert response == "Похоже на dashboard."
        final_user_content = calls[-1][-1]["content"]
        assert "[Screenshot 1]" in final_user_content[1]["text"]

    @pytest.mark.asyncio
    async def test_run_multimodal_injects_document_summary(self, agent, monkeypatch):
        calls = []

        class Resp:
            def __init__(self, text):
                self.content = [type("Block", (), {"type": "text", "text": text})()]
                self.stop_reason = "end_turn"
                self.usage = type(
                    "Usage",
                    (),
                    {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                )()

        async def fake_complete(model, max_tokens, messages, system=None, tools=None, temperature=None):
            calls.append({"model": model, "messages": messages})
            if len(calls) == 1:
                return Resp("Document title: Tax Guide. Sections: VAT, exemptions, deadlines.")
            return Resp("Это налоговый документ.")

        monkeypatch.setattr(agent, "_document_model_candidates", lambda requested_model="": ["gpt-4o-mini"])
        agent.provider.complete = fake_complete

        response = await agent.run([
            {"type": "text", "text": "О чем этот PDF?"},
            {"type": "document", "source": {
                "type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQ=", "filename": "tax-guide.pdf"
            }},
        ], "u1")

        assert response == "Это налоговый документ."
        final_user_content = calls[-1]["messages"][-1]["content"]
        assert final_user_content[1]["type"] == "text"
        assert "[Document 1]" in final_user_content[1]["text"]
        assert "Tax Guide" in final_user_content[1]["text"]
        # Raw document blocks are stripped after pre-analysis; text summary is sufficient.
        assert not any(block.get("type") == "document" for block in final_user_content)

    @pytest.mark.asyncio
    async def test_complete_multimodal_with_fallback_tries_next_candidate(self, agent, monkeypatch):
        attempts = []

        class Resp:
            def __init__(self, text):
                self.content = [type("Block", (), {"type": "text", "text": text})()]
                self.stop_reason = "end_turn"
                self.usage = type("Usage", (), {})()

        async def fake_complete(model, max_tokens, messages, system=None, tools=None, temperature=None):
            attempts.append(model)
            if model == "bad-model":
                raise RuntimeError("primary failed")
            return Resp("secondary ok")

        monkeypatch.setattr(agent, "_vision_model_candidates", lambda requested_model="": ["bad-model", "good-model"])
        agent.provider.complete = fake_complete

        text = await agent._complete_multimodal_with_fallback(
            [{"type": "text", "text": "Describe image"}],
            max_tokens=120,
        )

        assert text == "secondary ok"
        assert attempts == ["bad-model", "good-model"]

    def test_select_multimodal_response_model_promotes_text_model(self, agent, monkeypatch):
        monkeypatch.setattr(agent, "_vision_model_candidates", lambda requested_model="": ["qwen-vl-plus"])
        chosen = agent._select_multimodal_response_model(
            "qwen-plus",
            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc"}}],
        )
        assert chosen == "qwen-vl-plus"

    @pytest.mark.asyncio
    async def test_run_multimodal_continues_when_media_understanding_fails(self, agent):
        calls = []

        class Resp:
            def __init__(self, text):
                self.content = [type("Block", (), {"type": "text", "text": text})()]
                self.stop_reason = "end_turn"
                self.usage = type(
                    "Usage",
                    (),
                    {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                )()

        async def fake_complete(model, max_tokens, messages, system=None, tools=None, temperature=None):
            calls.append(messages)
            if len(calls) == 1:
                raise RuntimeError("vision prepass failed")
            return Resp("Основной ответ без pre-pass.")

        agent._vision_model_candidates = lambda requested_model="": ["bad-model"]
        agent.provider.complete = fake_complete

        response = await agent.run([
            {"type": "text", "text": "Что на картинке?"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}},
        ], "u1")

        assert response == "Основной ответ без pre-pass."
        final_user_content = calls[-1][-1]["content"]
        assert isinstance(final_user_content, list)
        assert not any(
            block.get("type") == "text" and "Auto media understanding" in block.get("text", "")
            for block in final_user_content
        )
        assert any(block.get("type") == "image" for block in final_user_content)


class TestSlowLocalMode:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "provider": "ollama",
                "default_model": "qwen3:30b",
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
            "knowledge_base": {
                "enabled": True,
                "search_mode": "hybrid",
                "auto_context_top_k": 6,
                "auto_context_max_chars": 9000,
                "auto_context_mode": "always",
            },
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    @pytest.mark.asyncio
    async def test_slow_local_clamps_kb_topk_and_context_size(self, agent):
        class DummyKB:
            def __init__(self):
                self.calls = []

            async def search(self, query, top_k=6, mode=None):
                self.calls.append((query, top_k, mode))
                return [object(), object(), object()]

            def build_context(self, results):
                return "X" * 7000

        agent._knowledge_base = DummyKB()
        ctx = await agent._auto_retrieve_kb_context(
            "What is logic in open-logic-complete.pdf?"
        )
        # Slow-local defaults: top_k capped to 2 and context capped to ~2500 chars.
        assert agent._knowledge_base.calls[0][1] == 2
        assert ctx.endswith("...[truncated]")
        assert len(ctx) <= 2515

    @pytest.mark.asyncio
    async def test_slow_local_uses_kb_cache_for_identical_query(self, agent):
        class DummyKB:
            def __init__(self):
                self.calls = 0

            async def search(self, query, top_k=6, mode=None):
                self.calls += 1
                return [object()]

            def build_context(self, results):
                return "cached-context"

        kb = DummyKB()
        agent._knowledge_base = kb
        q = "What is logic in open-logic-complete.pdf?"
        first = await agent._auto_retrieve_kb_context(q)
        second = await agent._auto_retrieve_kb_context(q)

        assert first == "cached-context"
        assert second == "cached-context"
        assert kb.calls == 1


class TestPinnedProfilePrompt:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 2,
                "provider": "ollama",
                "default_model": "qwen3:30b",
            },
            "cost": {"budget_daily_usd": 100.0, "prompt_caching": False},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_build_system_prompt_includes_pinned_profile(self, agent):
        agent.memory.set_state("app:onboarding_complete", True)
        agent.memory.set_state(
            "user:profile_facts",
            {"name": "Влад", "language": "ru"},
            user_id="u1",
        )
        prompt = agent._build_system_prompt("как дела?", "u1")
        assert isinstance(prompt, str)
        assert "User profile (pinned facts)" in prompt
        assert "Влад" in prompt

    def test_build_system_prompt_backfills_profile_from_history(self, agent):
        agent.memory.set_state("app:onboarding_complete", True)
        agent.memory.add_message("u1", "user", "Меня зовут Влад")
        prompt = agent._build_system_prompt("как тебя зовут?", "u1")
        assert isinstance(prompt, str)
        assert "User profile (pinned facts)" in prompt
        assert "Влад" in prompt

    def test_build_system_prompt_registers_recall_feedback(self, agent):
        agent.memory.set_state("app:onboarding_complete", True)
        agent._cached_recall = lambda *_args, **_kwargs: [
            {"id": 41, "content": "User prefers concise replies", "type": "preference", "score": 0.82},
            {"id": 42, "content": "Favorite editor is Neovim", "type": "fact", "score": 0.05},
        ]
        calls = []

        def capture_feedback(query, user_id, shown_ids, used_ids, *, strength=1.0, source="recall"):
            calls.append(
                {
                    "query": query,
                    "user_id": user_id,
                    "shown_ids": shown_ids,
                    "used_ids": used_ids,
                    "strength": strength,
                    "source": source,
                }
            )
            return {"shown": len(shown_ids), "used": len(used_ids), "reinforced": len(used_ids), "penalized": 0}

        agent.memory.register_recall_feedback = capture_feedback

        prompt = agent._build_system_prompt("remember my preferences", "u1")

        assert "User prefers concise replies" in prompt
        assert calls == [
            {
                "query": "remember my preferences",
                "user_id": "u1",
                "shown_ids": [41, 42],
                "used_ids": [41],
                "strength": 0.7,
                "source": "system_prompt",
            }
        ]

    def test_type_aware_recall_prefers_profile_slot(self, agent):
        agent.memory.set_state("app:onboarding_complete", True)
        agent.memory.upsert_canonical_slot("u1", "name", "Влад", confidence=0.95, source="test")
        results = agent._cached_recall("как меня зовут", "u1", top_k=3)
        assert results
        assert results[0]["type"] == "profile_slot"
        assert "Влад" in results[0]["content"]

    def test_direct_profile_memory_answer_from_canonical_slot(self, agent):
        agent.memory.upsert_canonical_slot("u1", "name", "Влад", confidence=0.91, source="test")
        answer = agent._direct_profile_memory_answer("посмотри в памяти как меня зовут", "u1")
        assert answer is not None
        assert "Влад" in answer

    @pytest.mark.asyncio
    async def test_run_short_circuits_name_query_without_llm_call(self, agent):
        agent.memory.upsert_canonical_slot("u1", "name", "Влад", confidence=0.88, source="test")
        agent.provider.complete = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        res = await agent.run("как меня зовут?", "u1")
        assert "Влад" in res

    def test_send_text_to_user_uses_saved_telegram_key(self, agent, monkeypatch):
        calls = {}

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"ok":true}'

        def fake_urlopen(req, timeout=0):
            calls["url"] = req.full_url
            calls["payload"] = req.data.decode("utf-8")
            calls["timeout"] = timeout
            return DummyResponse()

        monkeypatch.setattr("liteagent.config.get_api_key", lambda provider: "telegram-test-token" if provider == "telegram" else "")
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        token = agent._set_current_chat_id("123456")
        try:
            result = agent.tools._handlers["send_text_to_user"]("Smoke delivery")
        finally:
            agent._reset_current_chat_id(token)

        assert result == "Message sent to Telegram chat."
        assert calls["url"].endswith("/bottelegram-test-token/sendMessage")
        assert '"chat_id": "123456"' in calls["payload"]
        assert '"text": "Smoke delivery"' in calls["payload"]

    def test_send_text_to_user_derives_private_chat_from_canonical_tg_user(self, agent, monkeypatch):
        calls = {}

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"ok":true}'

        def fake_urlopen(req, timeout=0):
            calls["url"] = req.full_url
            calls["payload"] = req.data.decode("utf-8")
            calls["timeout"] = timeout
            return DummyResponse()

        monkeypatch.setattr("liteagent.config.get_api_key", lambda provider: "telegram-test-token" if provider == "telegram" else "")
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        agent._current_user_id = "tg-169108358"
        result = agent.tools._handlers["send_text_to_user"]("Derived delivery")

        assert result == "Message sent to Telegram chat."
        assert '"chat_id": "169108358"' in calls["payload"]
        assert '"text": "Derived delivery"' in calls["payload"]

    @pytest.mark.asyncio
    async def test_send_stored_file_to_telegram_uses_storage_key_and_current_chat(self, agent, monkeypatch):
        calls = {}

        class DummyResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"ok":true}'

        def fake_urlopen(req, timeout=0):
            calls["url"] = req.full_url
            calls["payload"] = req.data
            calls["timeout"] = timeout
            calls["content_type"] = req.headers.get("Content-type") or req.headers.get("Content-Type")
            return DummyResponse()

        class Storage:
            async def async_download(self, storage_key):
                calls["downloaded_key"] = storage_key
                return b"%PDF-telegram"

        agent._storage = Storage()
        agent._file_manager = type(
            "FM",
            (),
            {
                "list_files": lambda self, user_id=None, limit=200: [{
                    "storage_key": "files/api/passport.pdf",
                    "original_name": "passport.pdf",
                    "mime_type": "application/pdf",
                }]
            },
        )()
        agent._wire_storage_tools()

        monkeypatch.setattr("liteagent.config.get_api_key", lambda provider: "telegram-test-token" if provider == "telegram" else "")
        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        token = agent._set_current_chat_id("123456")
        try:
            result = await agent.tools._handlers["send_stored_file_to_telegram"](
                "files/api/passport.pdf",
                "Passport delivery",
            )
        finally:
            agent._reset_current_chat_id(token)

        assert result == "Stored file sent to Telegram chat: passport.pdf"
        assert calls["downloaded_key"] == "files/api/passport.pdf"
        assert calls["url"].endswith("/bottelegram-test-token/sendDocument")
        assert calls["timeout"] == 20
        assert "multipart/form-data" in (calls["content_type"] or "")
        assert b'name="chat_id"' in calls["payload"]
        assert b"123456" in calls["payload"]
        assert b'name="document"; filename="passport.pdf"' in calls["payload"]
        assert b"%PDF-telegram" in calls["payload"]

    @pytest.mark.asyncio
    async def test_run_followup_telegram_delivery_sends_last_substantive_answer(self, agent, monkeypatch):
        delivered = {}

        def fake_send_text(message):
            delivered["message"] = message
            return "Message sent to Telegram chat."

        monkeypatch.setitem(agent.tools._handlers, "send_text_to_user", fake_send_text)

        agent.memory.add_message("tg-42", "assistant", {"text": "Вот три важные AI-новости за сегодня:\n1. Новость один\n2. Новость два\n3. Новость три"})

        res = await agent.run("попробуй эти новости отправить в телеграм", "tg-42")

        assert res == "Последний подготовленный ответ отправлен в Telegram."
        assert "Вот три важные AI-новости" in delivered["message"]

    def test_build_system_prompt_includes_telegram_delivery_context(self, agent, monkeypatch):
        agent.memory.set_state("app:onboarding_complete", True)
        agent.memory.set_state("user:telegram_chat_id", "987654", user_id="tg-42")
        monkeypatch.setattr("liteagent.config.get_api_key", lambda provider: "telegram-test-token" if provider == "telegram" else "")

        prompt = agent._build_system_prompt("отправь это мне в телеграм", "tg-42")
        if isinstance(prompt, list):
            prompt_text = "\n".join(block.get("text", "") for block in prompt)
        else:
            prompt_text = prompt

        assert "Telegram delivery context" in prompt_text
        assert "Never ask the user for the Telegram token" in prompt_text
        assert "Never ask the user for chat_id" in prompt_text
        assert "A value like @SomeBot is the bot's username" in prompt_text
        assert "send_text_to_user" in prompt_text
        assert "Never use exec_command, curl, or raw Telegram HTTP calls" in prompt_text

    @pytest.mark.asyncio
    async def test_run_explains_bot_username_is_not_chat_id_in_telegram_context(self, agent):
        res = await agent.run("@Jess_skorxbot", "dashboard-user")

        assert "username самого Telegram-бота" in res
        assert "не destination chat_id" in res
        assert "numeric chat_id" in res

    def test_sanitize_unverified_completion_blocks_false_telegram_delivery_claim(self, agent):
        text = "Сообщение успешно отправлено в Telegram."
        user_input = "отправь мне это в телеграм"
        tool_calls = [{"name": "send_status", "error": False, "result_preview": "ok (dashboard only)"}]

        sanitized = agent._sanitize_unverified_completion_response(text, user_input, tool_calls)

        assert "отправка в Telegram пока не подтверждена" in sanitized

    def test_sanitize_unverified_completion_allows_verified_telegram_delivery_claim(self, agent):
        text = "Сообщение успешно отправлено в Telegram."
        user_input = "отправь мне это в телеграм"
        tool_calls = [{"name": "send_text_to_user", "error": False, "result_preview": "Message sent to Telegram chat."}]

        sanitized = agent._sanitize_unverified_completion_response(text, user_input, tool_calls)

        assert sanitized == text

    def test_sanitize_unverified_completion_blocks_false_delivery_to_bot_username(self, agent):
        text = "Новости успешно отправлены в `@Jess_skorxbot`."
        user_input = "попробуй эти новости отправить в телеграм"
        tool_calls = [{"name": "send_status", "error": False, "result_preview": "ok"}]

        sanitized = agent._sanitize_unverified_completion_response(text, user_input, tool_calls)

        assert "отправка в Telegram пока не подтверждена" in sanitized

    @pytest.mark.asyncio
    async def test_stream_short_circuits_name_query_without_llm_call(self, agent):
        agent.memory.upsert_canonical_slot("u1", "name", "Влад", confidence=0.88, source="test")
        agent.provider.stream = AsyncMock(side_effect=AssertionError("LLM stream must not be called"))
        chunks = []
        async for part in agent.stream("как меня зовут?", "u1"):
            chunks.append(part)
        assert chunks
        assert "Влад" in "".join(chunks)

    @pytest.mark.asyncio
    async def test_direct_profile_answer_ignores_contradicted_name(self, agent):
        agent.memory.upsert_canonical_slot("u1", "name", "Jess", confidence=0.92, source="test")
        await agent.memory.remember("name_is_not_Jess", "u1", "fact", 0.9)
        answer = agent._direct_profile_memory_answer("как меня зовут?", "u1")
        assert answer is not None
        assert "пока нет вашего имени" in answer.lower()

    @pytest.mark.asyncio
    async def test_direct_profile_answer_recovers_name_from_recent_memory_signals(self, agent):
        agent.memory.upsert_canonical_slot("u1", "name", "Jess", confidence=0.92, source="test")
        await agent.memory.remember("name_is_not_Jess", "u1", "fact", 0.9)
        agent.memory.add_message("u1", "user", "Слава запиши в свою память")
        await agent.memory.remember("имя пользователя - Слава", "u1", "fact", 0.65)

        answer = agent._direct_profile_memory_answer("как меня зовут?", "u1")
        assert answer is not None
        assert "Слава" in answer

    @pytest.mark.asyncio
    async def test_run_explicit_name_update_short_circuits_without_llm_call(self, agent):
        agent.provider.complete = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        res = await agent.run("Слава запиши в свою память", "u1")
        assert "Запомнил" in res
        assert "Слава" in res
        slot = agent.memory.get_canonical_slot("u1", "name")
        assert slot is not None
        assert slot.get("slot_value") == "Слава"

    @pytest.mark.asyncio
    async def test_stream_explicit_name_update_short_circuits_without_llm_call(self, agent):
        agent.provider.stream = AsyncMock(side_effect=AssertionError("LLM stream must not be called"))
        chunks = []
        async for part in agent.stream("Слава запиши в свою память", "u1"):
            chunks.append(part)
        text = "".join(chunks)
        assert "Запомнил" in text
        assert "Слава" in text
        slot = agent.memory.get_canonical_slot("u1", "name")
        assert slot is not None
        assert slot.get("slot_value") == "Слава"

    @pytest.mark.asyncio
    async def test_run_personal_memory_summary_short_circuits_without_llm_call(self, agent):
        agent.provider.complete = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        await agent.memory.remember("User prefers concise replies", "u1", "preference", 0.9)
        await agent.memory.remember("User works with audio files", "u1", "fact", 0.8)

        res = await agent.run("что ты помнишь обо мне?", "u1")
        assert "Помню о вас" in res
        assert ("audio" in res.lower()) or ("аудио" in res.lower())

    @pytest.mark.asyncio
    async def test_stream_personal_memory_summary_short_circuits_without_llm_call(self, agent):
        agent.provider.stream = AsyncMock(side_effect=AssertionError("LLM stream must not be called"))
        await agent.memory.remember("User works with audio files", "u1", "fact", 0.8)

        chunks = []
        async for part in agent.stream("что ты знаешь обо мне?", "u1"):
            chunks.append(part)
        text = "".join(chunks)
        assert "Помню о вас" in text
        assert ("audio" in text.lower()) or ("аудио" in text.lower())

    @pytest.mark.asyncio
    async def test_personal_memory_summary_includes_historical_fact(self, agent):
        agent.provider.complete = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        old_id = await agent.memory.remember(
            "User built Telegram bot deployment flow",
            "u1",
            "fact",
            0.95,
        )
        # Simulate long-term memory from several days ago.
        agent.memory.db.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?",
            ("2026-02-28T10:00:00", old_id),
        )
        agent.memory.db.commit()
        await agent.memory.remember("язык общения — русский", "u1", "preference", 0.8)
        await agent.memory.remember("предпочитает краткие ответы", "u1", "fact", 0.6)

        res = await agent.run("что ты помнишь обо мне?", "u1")
        assert "Помню о вас" in res
        assert ("telegram" in res.lower()) or ("телеграм" in res.lower())

    @pytest.mark.asyncio
    async def test_run_historical_request_query_short_circuits_without_llm_call(self, agent):
        agent.provider.complete = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        agent.memory.add_message("u1", "user", "сделай в настройках загрузку документов в базу знаний")
        agent.memory.add_message("u1", "user", "добавь memory exchange monitor в dashboard")

        # Force those messages to yesterday to match deterministic date filter.
        agent.memory.db.execute(
            "UPDATE chat_history SET created_at = '2026-03-04 15:00:00' WHERE user_id='u1' AND role='user'"
        )
        agent.memory.db.commit()

        res = await agent.run("что я просил разработать тебя вчера ?", "u1")
        assert ("вчера вы просили" in res.lower()) or ("помню ваши прошлые запросы" in res.lower())
        assert ("загрузку документов" in res.lower()) or ("memory exchange monitor" in res.lower())

    def test_sanitize_memory_limit_response_replaces_false_disclaimer(self, agent):
        agent.memory.add_message("u1", "user", "сделай блок памяти")
        agent.memory.db.execute(
            "UPDATE chat_history SET created_at = '2026-03-04 12:00:00' WHERE user_id='u1' AND role='user'"
        )
        agent.memory.db.commit()
        repaired = agent._sanitize_memory_limit_response(
            "У меня нет доступа к истории вчерашних сессий.",
            "что я просил разработать тебя вчера ?",
            "u1",
        )
        assert "нет доступа к истории" not in repaired.lower()

    def test_sanitize_unverified_completion_blocks_false_done_claim(self, agent):
        text = agent._sanitize_unverified_completion_response(
            "Готово, всё исправил и перезапустил.",
            "исправь память и перезапусти сервер",
            tool_calls_log=[],
        )
        assert "выполнение пока не подтверждено" in text.lower()

    def test_sanitize_unverified_completion_allows_verified_tool_success(self, agent):
        original = "Done, fixed and restarted."
        text = agent._sanitize_unverified_completion_response(
            original,
            "fix memory and restart",
            tool_calls_log=[{"name": "exec_command", "result": "ok", "error": ""}],
        )
        assert text == original

    def test_sanitize_unverified_completion_blocks_claims_on_short_confirmation_turn(self, agent):
        text = agent._sanitize_unverified_completion_response(
            "Фронтенд добавлен и запущен. Доступно по http://localhost:8000",
            "да",
            tool_calls_log=[],
        )
        assert "выполнение пока не подтверждено" in text.lower()

    def test_force_tool_continuation_detects_status_only_guard_on_side_effect_task(self, agent):
        assert agent._should_force_tool_continuation(
            "Промежуточный статус: выполнение пока не подтверждено инструментами. "
            "Запущу нужные действия и вернусь с проверенным результатом.",
            "сделай full-stack проект и запусти сервер",
            tool_calls_log=[],
            forced_attempts=0,
        ) is True

    @pytest.mark.asyncio
    async def test_run_auto_continues_after_status_only_guard_until_tool_executes(self, agent, monkeypatch):
        from liteagent.providers import LLMResponse, TextBlock, ToolUseBlock

        usage = type("Usage", (), {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0})()
        first = LLMResponse(
            content=[TextBlock(type="text", text=(
                "Промежуточный статус: выполнение пока не подтверждено инструментами. "
                "Запущу нужные действия и вернусь с проверенным результатом."
            ))],
            stop_reason="end_turn",
            usage=usage,
        )
        second = LLMResponse(
            content=[ToolUseBlock(id="tool-1", name="exec_command", input={"command": "echo ok", "timeout": 5})],
            stop_reason="tool_use",
            usage=usage,
        )
        third = LLMResponse(
            content=[TextBlock(type="text", text="Готово. Команда выполнена и работа продолжена.")],
            stop_reason="end_turn",
            usage=usage,
        )

        call_seq = AsyncMock(side_effect=[first, second, third])
        monkeypatch.setattr(agent, "_call_api", call_seq)
        monkeypatch.setattr(agent.tools, "execute_parallel", AsyncMock(return_value=[{
            "type": "tool_result",
            "tool_use_id": "tool-1",
            "content": "ok",
            "_meta": {"tool_name": "exec_command", "error": False, "result_preview": "ok", "duration_ms": 5},
        }]))
        agent.max_iterations = 4

        result = await agent.run("сделай full-stack проект и запусти сервер", "u1")

        assert "Готово" in result
        assert call_seq.await_count == 3
        agent.tools.execute_parallel.assert_awaited()

    def test_parse_bracket_style_tool_calls_from_local_model_text(self, agent):
        tool_defs = [{
            "name": "exec_command",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "approved": {"type": "boolean"},
                },
            },
        }]
        parsed = agent._try_parse_text_tool_calls(
            '[exec_command("cd /tmp && python3 main.py", 30)]',
            tool_defs,
        )
        assert parsed == [{
            "name": "exec_command",
            "arguments": {
                "command": "cd /tmp && python3 main.py",
                "timeout": 30,
            },
        }]

    def test_parse_plain_text_tool_calls_from_local_model_text(self, agent):
        tool_defs = [{
            "name": "exec_command",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer"},
                    "approved": {"type": "boolean"},
                },
            },
        }]
        parsed = agent._try_parse_text_tool_calls(
            'Сейчас вызываю exec_command("cd /tmp && python3 main.py", timeout=30) и жду результат.',
            tool_defs,
        )
        assert parsed == [{
            "name": "exec_command",
            "arguments": {
                "command": "cd /tmp && python3 main.py",
                "timeout": 30,
            },
        }]

    @pytest.mark.asyncio
    async def test_historical_request_uses_memories_fallback(self, agent):
        agent.provider.complete = AsyncMock(side_effect=AssertionError("LLM must not be called"))
        await agent.memory.remember(
            "User requested that a dashboard be created in the develop folder and named dashboard.",
            "u1",
            "fact",
            0.8,
        )
        await agent.memory.remember(
            "User wants memory exchange monitor in dashboard.",
            "u1",
            "fact",
            0.8,
        )
        res = await agent.run("что я просил разработать тебя вчера ?", "u1")
        assert ("прошлые запросы" in res.lower()) or ("вчера вы просили" in res.lower())
        assert ("dashboard" in res.lower()) or ("monitor" in res.lower())

    @pytest.mark.asyncio
    async def test_build_system_prompt_includes_memory_exchange_context(self, agent):
        agent.memory.set_state("app:onboarding_complete", True)
        await agent.memory.remember("We deploy with Docker compose.", "u1", "fact", 0.8)
        await agent.memory.run_memory_exchange_cycle("docker deploy", "u1", "")

        prompt = agent._build_system_prompt("docker deploy", "u1")
        assert isinstance(prompt, str)
        assert "Memory Exchange (precomputed)" in prompt


class TestToolAutonomy:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {"max_iterations": 2},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": ["read_file", "write_file", "exec_command"]},
            "features": {"auto_tool_synthesis": {"enabled": True}},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_autonomy_registers_vision_tool_when_query_requires_image_understanding(self, agent):
        assert "vision_analyze_image" not in agent.tools._tools
        tool_defs = agent._ensure_tool_autonomy(
            "Создай и используй инструмент для распознавания изображений",
            [],
        )
        names = {t["name"] for t in tool_defs}
        assert "vision_analyze_image" in names
        assert agent.tools.has_tool("vision_analyze_image")

    def test_autonomy_adds_synthesize_tool_for_explicit_tool_gap_requests(self, agent):
        tool_defs = agent._ensure_tool_autonomy(
            "Если нет инструмента, создай новый и используй его",
            [],
        )
        names = {t["name"] for t in tool_defs}
        assert "synthesize_tool" in names

    def test_autonomy_keeps_core_workspace_tools_for_build_queries(self, agent):
        tool_defs = agent._ensure_tool_autonomy(
            "Создай full-stack проект на FastAPI с фронтендом и запусти сервер на порту 8091",
            [],
        )
        names = {t["name"] for t in tool_defs}
        assert {"read_file", "write_file", "exec_command"} <= names


class TestIterationControl:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "provider": "ollama",
                "default_model": "qwen3-coder:30b",
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": ["read_file", "write_file", "exec_command"]},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_dynamic_iteration_budget_expands_for_large_fullstack_debug_task(self, agent):
        budget = agent._dynamic_iteration_budget(
            "Создай full-stack проект с фронтендом и бекендом, запусти сервер, "
            "прогони browser MCP e2e и исправь баги после проверки",
            complexity_score=6,
            tool_defs=agent.tools.get_definitions(),
        )
        assert budget >= 90

    def test_explicit_max_iterations_remains_hard_cap(self, tmp_path):
        config = {
            "agent": {
                "provider": "ollama",
                "default_model": "qwen3-coder:30b",
                "max_iterations": 7,
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": ["exec_command"]},
        }
        agent = LiteAgent(config)
        try:
            budget = agent._dynamic_iteration_budget(
                "Build and debug a large full-stack project with browser tests",
                complexity_score=6,
                tool_defs=agent.tools.get_definitions(),
            )
            assert budget == 7
        finally:
            agent.memory.close()

    def test_no_progress_tracker_stops_on_repeated_identical_tool_results(self, agent):
        tracker = None
        repeated = [{
            "name": "exec_command",
            "input": {"command": "curl http://127.0.0.1:8091/health"},
            "error": False,
            "result_preview": '{"status":"ok"}',
            "duration_ms": 30,
        }]
        effective_max = 96
        tracker = agent._advance_progress_tracker(tracker, repeated)
        for _ in range(agent._no_progress_limit(effective_max)):
            tracker = agent._advance_progress_tracker(tracker, repeated)
        assert agent._should_stop_for_no_progress(tracker, effective_max)

    def test_failure_only_tracker_forces_targeted_repair_prompt(self, agent):
        tracker = None
        failed = [{
            "name": "exec_command",
            "input": {"command": "cd frontend && npm start"},
            "error": True,
            "result_preview": "Could not find a required file. Name: index.html",
            "duration_ms": 120,
        }]
        effective_max = 96
        for _ in range(agent._failure_only_repair_limit(effective_max)):
            tracker = agent._advance_progress_tracker(tracker, failed)
        assert tracker["failure_only_count"] >= agent._failure_only_repair_limit(effective_max)
        assert agent._should_force_failed_tool_repair(
            tracker,
            "сделай full-stack проект и исправь баги после проверки",
            forced_attempts=0,
            iteration_calls=failed,
            effective_max=effective_max,
        ) is True

    @pytest.mark.asyncio
    async def test_self_healing_health_snapshot_includes_memory_and_env_health(self, agent, monkeypatch):
        class MockHealth:
            def __init__(self, name, status, latency_ms=0.0, error_message=""):
                self.name = name
                self.status = status
                self.latency_ms = latency_ms
                self.error_message = error_message

        class MockMonitor:
            async def run_all_checks(self):
                return {
                    "ollama": MockHealth("ollama", "healthy", 42.0),
                    "api": MockHealth("api", "degraded", 0.0, "connection refused"),
                }

        monkeypatch.setattr(
            agent.memory,
            "memory_health_check",
            lambda user_id: {"status": "warning", "issues": ["Low hit rate"]},
        )
        agent._health_monitor = MockMonitor()
        snapshot = await agent._collect_self_healing_health_snapshot(
            "u1",
            [{"name": "exec_command", "error": True, "result_preview": "npm start failed"}],
        )
        assert "Self-healing health snapshot" in snapshot
        assert "Memory health: warning" in snapshot
        assert "Environment health:" in snapshot
        assert "api=degraded" in snapshot
        assert "npm start failed" in snapshot

    def test_no_tool_recovery_triggers_for_side_effect_task_without_tool_results(self, agent):
        assert agent._should_force_no_tool_recovery(
            "сделай full-stack проект и исправь баги после проверки",
            tool_calls_log=[],
            forced_attempts=0,
            no_tool_passes=1,
        ) is True
        assert agent._should_force_no_tool_recovery(
            "объясни что такое FastAPI",
            tool_calls_log=[],
            forced_attempts=0,
            no_tool_passes=1,
        ) is False
        assert agent._should_force_no_tool_recovery(
            "сделай full-stack проект и исправь баги после проверки",
            tool_calls_log=[{"name": "read_file", "error": False}],
            forced_attempts=0,
            no_tool_passes=1,
        ) is False

    def test_permission_seeking_reply_triggers_autonomy_recovery(self, agent):
        assert agent._should_force_autonomy_recovery(
            "сделай full-stack проект и исправь баги после проверки",
            "Хочешь, чтобы я сначала запросил подтверждение перед правками?",
            tool_calls_log=[],
            forced_attempts=0,
        ) is True
        assert agent._should_force_autonomy_recovery(
            "сделай full-stack проект и исправь баги после проверки",
            "Нужен API key и токен, без них я заблокирован.",
            tool_calls_log=[],
            forced_attempts=0,
        ) is False

    @pytest.mark.asyncio
    async def test_run_forces_targeted_repair_after_repeated_failed_tool_iterations(self, agent, monkeypatch):
        from liteagent.providers import LLMResponse, TextBlock, ToolUseBlock

        usage = type("Usage", (), {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0})()
        first = LLMResponse(
            content=[ToolUseBlock(id="tool-1", name="exec_command", input={"command": "cd frontend && npm start"})],
            stop_reason="tool_use",
            usage=usage,
        )
        second = LLMResponse(
            content=[ToolUseBlock(id="tool-2", name="exec_command", input={"command": "cd frontend && npm start"})],
            stop_reason="tool_use",
            usage=usage,
        )
        third = LLMResponse(
            content=[ToolUseBlock(
                id="tool-3",
                name="write_file",
                input={"path": "frontend/public/index.html", "content": "<!doctype html>"},
            )],
            stop_reason="tool_use",
            usage=usage,
        )
        fourth = LLMResponse(
            content=[TextBlock(type="text", text="Исправил bootstrap фронтенда и подтвердил проблему.")],
            stop_reason="end_turn",
            usage=usage,
        )

        responses = [first, second, third, fourth]
        seen_messages = []

        async def fake_call_api(**kwargs):
            seen_messages.append(kwargs.get("messages", []))
            return responses.pop(0)

        monkeypatch.setattr(agent, "_call_api", AsyncMock(side_effect=fake_call_api))
        monkeypatch.setattr(agent.tools, "execute_parallel", AsyncMock(side_effect=[
            [{
                "type": "tool_result",
                "tool_use_id": "tool-1",
                "content": "Could not find a required file. Name: index.html",
                "_meta": {
                    "tool_name": "exec_command",
                    "error": True,
                    "result_preview": "Could not find a required file. Name: index.html",
                    "duration_ms": 30,
                },
            }],
            [{
                "type": "tool_result",
                "tool_use_id": "tool-2",
                "content": "Could not find a required file. Name: index.html",
                "_meta": {
                    "tool_name": "exec_command",
                    "error": True,
                    "result_preview": "Could not find a required file. Name: index.html",
                    "duration_ms": 31,
                },
            }],
            [{
                "type": "tool_result",
                "tool_use_id": "tool-3",
                "content": "Written 15 chars to frontend/public/index.html",
                "_meta": {
                    "tool_name": "write_file",
                    "error": False,
                    "result_preview": "Written 15 chars to frontend/public/index.html",
                    "duration_ms": 8,
                },
            }],
        ]))
        class MockHealth:
            def __init__(self, name, status, latency_ms=0.0, error_message=""):
                self.name = name
                self.status = status
                self.latency_ms = latency_ms
                self.error_message = error_message

        class MockMonitor:
            async def run_all_checks(self):
                return {"ollama": MockHealth("ollama", "healthy", 35.0)}

        agent._health_monitor = MockMonitor()
        agent.max_iterations = 6

        result = await agent.run("сделай full-stack проект и исправь баги после проверки", "u1")

        assert "Исправил bootstrap" in result
        assert any(
            LiteAgent._forced_failed_tool_repair_prompt() in str(messages)
            for messages in seen_messages
        )
        assert any(
            "Self-healing health snapshot" in str(messages)
            for messages in seen_messages
        )
        assert agent.tools.execute_parallel.await_count == 3

    @pytest.mark.asyncio
    async def test_run_forces_tool_first_recovery_after_no_tool_side_effect_reply(self, agent, monkeypatch):
        from liteagent.providers import LLMResponse, TextBlock, ToolUseBlock

        usage = type("Usage", (), {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0})()
        first = LLMResponse(
            content=[TextBlock(type="text", text="Сначала проанализирую проект и составлю план исправления.")],
            stop_reason="end_turn",
            usage=usage,
        )
        second = LLMResponse(
            content=[ToolUseBlock(id="tool-1", name="read_file", input={"path": "frontend/package.json"})],
            stop_reason="tool_use",
            usage=usage,
        )
        third = LLMResponse(
            content=[TextBlock(type="text", text="Исправил контракт и подтвердил проблему.")],
            stop_reason="end_turn",
            usage=usage,
        )

        responses = [first, second, third]
        seen_messages = []

        async def fake_call_api(**kwargs):
            seen_messages.append(kwargs.get("messages", []))
            return responses.pop(0)

        monkeypatch.setattr(agent, "_call_api", AsyncMock(side_effect=fake_call_api))
        monkeypatch.setattr(agent.tools, "execute_parallel", AsyncMock(return_value=[{
            "type": "tool_result",
            "tool_use_id": "tool-1",
            "content": "{\"name\":\"frontend\"}",
            "_meta": {
                "tool_name": "read_file",
                "error": False,
                "result_preview": "{\"name\":\"frontend\"}",
                "duration_ms": 7,
            },
        }]))
        agent.max_iterations = 4

        result = await agent.run("сделай full-stack проект и исправь баги после проверки", "u1")

        assert "Исправил контракт" in result
        assert len(seen_messages) == 3
        agent.tools.execute_parallel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_forces_autonomy_recovery_after_permission_seeking_reply(self, agent, monkeypatch):
        from liteagent.providers import LLMResponse, TextBlock, ToolUseBlock

        usage = type("Usage", (), {"input_tokens": 1, "output_tokens": 1, "cache_read_input_tokens": 0})()
        first = LLMResponse(
            content=[TextBlock(type="text", text="Хочешь, чтобы я сначала получил подтверждение перед изменениями?")],
            stop_reason="end_turn",
            usage=usage,
        )
        second = LLMResponse(
            content=[ToolUseBlock(id="tool-1", name="read_file", input={"path": "frontend/package.json"})],
            stop_reason="tool_use",
            usage=usage,
        )
        third = LLMResponse(
            content=[TextBlock(type="text", text="Проверил контекст, внес правки и перепроверил результат.")],
            stop_reason="end_turn",
            usage=usage,
        )

        responses = [first, second, third]
        seen_messages = []

        async def fake_call_api(**kwargs):
            seen_messages.append(kwargs.get("messages", []))
            return responses.pop(0)

        monkeypatch.setattr(agent, "_call_api", AsyncMock(side_effect=fake_call_api))
        monkeypatch.setattr(agent.tools, "execute_parallel", AsyncMock(return_value=[{
            "type": "tool_result",
            "tool_use_id": "tool-1",
            "content": "{\"name\":\"frontend\"}",
            "_meta": {
                "tool_name": "read_file",
                "error": False,
                "result_preview": "{\"name\":\"frontend\"}",
                "duration_ms": 6,
            },
        }]))
        agent.max_iterations = 4

        result = await agent.run("сделай full-stack проект и исправь баги после проверки", "u1")

        assert "внес правки" in result
        assert any(
            "Do not ask the user for routine permission or confirmation here." in str(messages)
            for messages in seen_messages
        )
        agent.tools.execute_parallel.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_returns_timeout_recovery_message_instead_of_raising(self, agent, monkeypatch):
        monkeypatch.setattr(agent, "_run_impl", AsyncMock(side_effect=TimeoutError("LLM provider timed out after 300.0s")))

        result = await agent.run(
            "сделай full-stack проект и исправь баги после проверки",
            "u1",
            requested_model="qwen3-coder:30b",
        )

        assert "не успела ответить вовремя" in result
        assert "выполнение задачи не подтверждено инструментами" in result.lower()
        history = agent.memory.get_history("u1")
        assert any(
            isinstance(msg, dict) and msg.get("role") == "assistant"
            and "не успела ответить вовремя" in str(msg.get("content", ""))
            for msg in history
        )


class TestEvolutionHooks:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "provider": "ollama",
                "default_model": "qwen3-coder:30b",
                "models": {
                    "simple": "qwen2.5:latest",
                    "medium": "qwen3-coder:30b",
                    "complex": "qwen3-coder:30b",
                },
                "max_iterations": 2,
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {
                "db_path": str(tmp_path / "test.db"),
                "auto_learn": False,
                "extraction_model": "qwen2.5:latest",
            },
            "tools": {"builtin": []},
            "features": {"self_evolving_prompt": {"enabled": True, "min_friction_signals": 1}},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    @pytest.mark.asyncio
    async def test_friction_hook_auto_applies_patch_and_passes_local_model_context(
        self, agent, monkeypatch
    ):
        seen = {}

        async def fake_synthesize(provider, db, cfg):
            seen["cfg"] = dict(cfg)
            now = "2026-03-09T00:00:00"
            db.execute(
                "INSERT INTO prompt_patches (patch_text, reason, applied, created_at) VALUES (?, ?, 0, ?)",
                ("Verify frontend in browser", "test", now),
            )
            db.commit()
            return ["Verify frontend in browser"]

        monkeypatch.setattr(
            "liteagent.evolution.synthesize_prompt_patches",
            fake_synthesize,
        )

        ctx = HookContext(
            agent=agent,
            user_id="u1",
            model="qwen3-coder:30b",
            response_text="Initial response",
            extra={"user_input_text": "Нет, это неправильно, переделай"},
        )
        await agent.hooks.emit("after_response", ctx)
        if agent._background_tasks:
            await asyncio.gather(*list(agent._background_tasks))

        applied = agent.memory.db.execute(
            "SELECT applied FROM prompt_patches WHERE patch_text=?",
            ("Verify frontend in browser",),
        ).fetchone()
        assert applied is not None
        assert applied[0] == 1
        assert seen["cfg"]["_agent_config"]["default_model"] == "qwen3-coder:30b"
        assert seen["cfg"]["extraction_model"] == "qwen2.5:latest"


class TestProactiveFeatureInjection:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 2,
                "default_model": "claude-sonnet-4-20250514",
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
            "features": {
                "proactive_agent": {
                    "enabled": True,
                    "pattern_window_days": 30,
                    "min_pattern_occurrences": 3,
                }
            },
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_build_feature_section_includes_proactive_suggestions(self, agent):
        for idx in range(3):
            agent.memory.db.execute(
                "INSERT INTO interaction_log (user_id, user_input, agent_response, tool_calls_json, success, confidence, model_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "u1",
                    "sync drive files",
                    "done",
                    '[{"name":"exec_command"}]',
                    1,
                    0.9,
                    "local",
                    f"2026-03-09T09:0{idx * 2}:00",
                ),
            )
            agent.memory.db.execute(
                "INSERT INTO interaction_log (user_id, user_input, agent_response, tool_calls_json, success, confidence, model_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "u1",
                    "search files by content",
                    "done",
                    '[{"name":"read_file"}]',
                    1,
                    0.9,
                    "local",
                    f"2026-03-09T09:0{idx * 2 + 1}:00",
                ),
            )
        agent.memory.db.commit()

        text = agent._build_feature_section("sync files from google drive", "u1")
        assert "Proactive suggestions" in text
        assert "search files by content" in text.lower()


class TestThinkingCloudPromptInjection:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 2,
                "default_model": "claude-sonnet-4-20250514",
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_build_system_prompt_includes_thinking_cloud_context(self, agent):
        agent._skip_onboarding_for_request = True
        agent.memory.upsert_thinking_note(
            "u1",
            "constraint",
            "Prefer local models even if they are slower, because privacy and control matter more than speed.",
            themes=["local models", "privacy"],
            confidence=0.92,
            strategic_importance=0.96,
        )
        agent.memory.upsert_thinking_note(
            "u1",
            "direction",
            "Make the agent more autonomous while still verifying its work before finalizing.",
            themes=["autonomy", "verification"],
            confidence=0.88,
            strategic_importance=0.94,
        )

        prompt = agent._build_system_prompt("как сделать агента автономнее на локальных моделях", "u1")
        text = "".join(block.get("text", "") for block in prompt) if isinstance(prompt, list) else str(prompt)
        assert "User thinking cloud" in text
        assert "Dominant themes" in text
        assert "local models" in text.lower()
        assert "Active directions" in text

    def test_build_feature_section_includes_human_support_opportunities(self, agent):
        agent.config.setdefault("features", {})["human_support_agent"] = {
            "enabled": True,
            "max_suggestions": 3,
            "min_pattern_occurrences": 3,
        }
        agent._features["human_support_agent"] = dict(agent.config["features"]["human_support_agent"])
        text = agent._build_feature_section(
            "Я перегружен, плохо сплю и не могу сосредоточиться на задачах",
            "u1",
        )
        assert "Human support opportunities" in text
        assert "never force" in text
        assert "focus reset" in text.lower()


class TestCriticalResponseReview:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 3,
                "default_model": "claude-sonnet-4-20250514",
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    def test_should_run_critical_review_only_for_important_response(self, agent):
        assert not agent._should_run_critical_response_review(
            "привет",
            "Привет! Чем помочь?",
            [],
        )
        assert agent._should_run_critical_response_review(
            "исправь проект и полностью его перепроверь",
            "Исправил проект, проверил логи, прогнал тесты и перепроверил все ключевые сценарии.",
            [],
        )

    @pytest.mark.asyncio
    async def test_critical_review_rewrites_overclaiming_answer(self, agent):
        class Resp:
            def __init__(self, text):
                self.content = [type("Block", (), {"type": "text", "text": text})()]
                self.stop_reason = "end_turn"
                self.usage = type(
                    "Usage",
                    (),
                    {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 0,
                    },
                )()

        async def fake_complete(model, max_tokens, messages, system=None, tools=None, temperature=None):
            return Resp(
                '{"needs_revision": true, "issues": ["overclaiming"], '
                '"revised_answer": "Исправления выполнены частично; подтверждены только те шаги, которые реально проверены инструментами."}'
            )

        agent.provider.complete = fake_complete

        reviewed = await agent._critical_review_response_if_needed(
            user_input="исправь проект и проверь его полностью",
            response_text="Все полностью исправлено и проверено.",
            user_id="u1",
            tool_calls_log=[],
            model=agent.default_model,
        )

        assert "подтверждены" in reviewed
        assert agent._last_response_meta["critical_review"]["applied"] is True
        assert agent._last_response_meta["critical_review"]["revised"] is True

    @pytest.mark.asyncio
    async def test_run_applies_critical_review_before_return(self, agent, monkeypatch):
        from liteagent.providers import LLMResponse, TextBlock

        usage = type(
            "Usage",
            (),
            {
                "input_tokens": 1,
                "output_tokens": 1,
                "cache_read_input_tokens": 0,
            },
        )()
        response = LLMResponse(
            content=[TextBlock(type="text", text="Все полностью исправлено и готово.")],
            stop_reason="end_turn",
            usage=usage,
        )

        monkeypatch.setattr(agent, "_call_api", AsyncMock(return_value=response))
        monkeypatch.setattr(
            agent,
            "_critical_review_response_if_needed",
            AsyncMock(return_value="Исправления внесены, но финальная готовность подтверждена только частично."),
        )

        result = await agent.run(
            "Оцени риски этого решения и дай аккуратный итоговый вывод",
            "u1",
        )

        assert "подтверждена только частично" in result
        agent._critical_review_response_if_needed.assert_awaited_once()


class TestRecentFileFollowups:
    @pytest.fixture
    def agent(self, tmp_path):
        config = {
            "agent": {
                "max_iterations": 3,
                "default_model": "claude-sonnet-4-20250514",
            },
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        a = LiteAgent(config)
        yield a
        a.memory.close()

    @pytest.mark.asyncio
    async def test_run_answers_recent_file_location_without_llm(self, agent, monkeypatch):
        agent._file_manager = type(
            "FM",
            (),
            {
                "list_files": lambda self, user_id=None, limit=1: [{
                    "original_name": "guide.pdf",
                    "storage_key": "files/telegram/abc_guide.pdf",
                    "source": "telegram",
                    "description": "indexed pdf",
                }]
            },
        )()
        monkeypatch.setattr(agent.provider, "complete", AsyncMock(side_effect=AssertionError("LLM must not be called")))

        result = await agent.run("Где он хранится ?", "u1")

        assert "guide.pdf" in result
        assert "files/telegram/abc_guide.pdf" in result

    @pytest.mark.asyncio
    async def test_run_sends_recent_original_without_llm(self, agent, monkeypatch, tmp_path):
        agent._file_manager = type(
            "FM",
            (),
            {
                "list_files": lambda self, user_id=None, limit=1: [{
                    "original_name": "guide.pdf",
                    "storage_key": "files/telegram/abc_guide.pdf",
                    "source": "telegram",
                    "description": "indexed pdf",
                }]
            },
        )()

        class Storage:
            async def async_download(self, storage_key):
                return b"%PDF-1.4"

        queued = {}

        def fake_enqueue(path, caption=""):
            queued["path"] = path
            queued["caption"] = caption

        agent._storage = Storage()
        monkeypatch.setattr(agent.provider, "complete", AsyncMock(side_effect=AssertionError("LLM must not be called")))
        monkeypatch.setattr("liteagent.file_queue.enqueue_file", fake_enqueue)

        result = await agent.run("Пришли оригинал", "u1")

        assert "поставлен в очередь" in result.lower()
        assert queued["caption"] == "guide.pdf"

    @pytest.mark.asyncio
    async def test_run_sends_owned_document_without_unlock_phrase_when_not_configured(self, agent, monkeypatch):
        agent._file_manager = type(
            "FM",
            (),
            {
                "search": lambda self, query, user_id=None, top_k=8: [{
                    "original_name": "passport_scan.pdf",
                    "storage_key": "files/telegram/passport_scan.pdf",
                    "description": "passport scan",
                    "created_at": "2026-03-10T10:00:00",
                    "score": 0.91,
                }],
                "list_files": lambda self, user_id=None, limit=200: [],
            },
        )()

        class Storage:
            async def async_download(self, storage_key):
                return b"%PDF-owned-doc"

        queued = {}

        def fake_enqueue(path, caption=""):
            queued["path"] = path
            queued["caption"] = caption

        agent._storage = Storage()
        monkeypatch.setattr(agent.provider, "complete", AsyncMock(side_effect=AssertionError("LLM must not be called")))
        monkeypatch.setattr("liteagent.config.get_api_key", lambda name: None)
        monkeypatch.setattr("liteagent.file_queue.enqueue_file", fake_enqueue)

        result = await agent.run("Пришли мой паспорт из базы", "u1")

        assert "passport_scan.pdf" in result
        assert queued["caption"] == "passport_scan.pdf"

    @pytest.mark.asyncio
    async def test_run_requires_unlock_phrase_for_owned_document_when_configured(self, agent, monkeypatch):
        monkeypatch.setattr(agent.provider, "complete", AsyncMock(side_effect=AssertionError("LLM must not be called")))
        monkeypatch.setattr("liteagent.config.get_api_key", lambda name: "alpha-omega" if name == "document_unlock_phrase" else None)

        result = await agent.run("Пришли мой паспорт из базы", "u1")

        assert "кодовое слово" in result.lower()

    @pytest.mark.asyncio
    async def test_run_unlocks_and_sends_owned_document_with_phrase_then_reuses_session(self, agent, monkeypatch):
        agent._file_manager = type(
            "FM",
            (),
            {
                "search": lambda self, query, user_id=None, top_k=8: [{
                    "original_name": "passport_scan.pdf",
                    "storage_key": "files/telegram/passport_scan.pdf",
                    "description": "passport scan",
                    "created_at": "2026-03-10T10:00:00",
                    "score": 0.91,
                }],
                "list_files": lambda self, user_id=None, limit=200: [],
            },
        )()

        class Storage:
            async def async_download(self, storage_key):
                return b"%PDF-owned-doc"

        queued = []

        def fake_enqueue(path, caption=""):
            queued.append((path, caption))

        agent._storage = Storage()
        monkeypatch.setattr(agent.provider, "complete", AsyncMock(side_effect=AssertionError("LLM must not be called")))
        monkeypatch.setattr("liteagent.config.get_api_key", lambda name: "alpha-omega" if name == "document_unlock_phrase" else None)
        monkeypatch.setattr("liteagent.file_queue.enqueue_file", fake_enqueue)

        first = await agent.run("alpha-omega пришли мой паспорт из базы", "u1")
        second = await agent.run("Пришли мой паспорт из базы", "u1")

        assert "passport_scan.pdf" in first
        assert "passport_scan.pdf" in second
        assert len(queued) == 2

    @pytest.mark.asyncio
    async def test_internal_autonomous_prompt_bypasses_document_unlock_handler(self, agent, monkeypatch):
        monkeypatch.setattr("liteagent.config.get_api_key", lambda name: "alpha-omega" if name == "document_unlock_phrase" else None)
        monkeypatch.setattr(agent.provider, "complete", AsyncMock(return_value=LLMResponse(
            content=[TextBlock(text="planner response")],
            stop_reason="end_turn",
        )))

        result = await agent.run(
            "You are running one autonomous self-improvement cycle. Use actual tools and continue improving after the original objective.",
            "tg-169108358",
        )

        assert "кодовое слово" not in result.lower()
        assert result == "planner response"

    @pytest.mark.asyncio
    async def test_run_sends_recent_markdown_file_without_llm(self, agent, monkeypatch, tmp_path):
        md_path = tmp_path / "5M_Strategy_Summary.md"
        md_path.write_text("# Summary\nbody", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        agent.memory.add_message("u1", "assistant", "Файл `5M_Strategy_Summary.md` успешно создан и готов.")

        queued = {}

        def fake_send_file_to_user(file_path: str = "", caption: str = "", content: str = ""):
            queued["file_path"] = file_path
            queued["caption"] = caption
            return f"File queued for delivery: {os.path.basename(file_path)}"

        monkeypatch.setitem(agent.tools._handlers, "send_file_to_user", fake_send_file_to_user)
        monkeypatch.setattr(agent.provider, "complete", AsyncMock(side_effect=AssertionError("LLM must not be called")))

        result = await agent.run("Пришли мне этот файл Markdown", "u1")

        assert "5M_Strategy_Summary.md" in result
        assert queued["caption"] == "5M_Strategy_Summary.md"
        assert queued["file_path"] == str(md_path)
