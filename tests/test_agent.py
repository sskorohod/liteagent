"""Tests for the agent core logic (no API calls)."""

import pytest

from liteagent.agent import LiteAgent, MODEL_PRICING


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

    def test_simple_question_routes_to_haiku(self, agent):
        model = agent._select_model("What time is it?")
        assert "haiku" in model

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
