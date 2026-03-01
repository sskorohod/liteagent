"""Tests for internal monologue / planning module."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from liteagent.planning import generate_plan, format_plan_for_prompt, reflect_on_progress


def _mock_provider(response_text: str):
    """Create a mock provider that returns given text."""
    provider = AsyncMock()
    result = MagicMock()
    result.content = [MagicMock(text=response_text)]
    provider.complete = AsyncMock(return_value=result)
    return provider


class TestGeneratePlan:
    """Test generate_plan function."""

    @pytest.mark.asyncio
    async def test_complex_request_returns_plan(self):
        plan_json = json.dumps({
            "steps": ["Search for relevant files", "Read and analyze code",
                      "Write implementation"],
            "complexity": "complex",
            "tools_needed": ["read_file", "write_file"],
            "estimated_iterations": 5,
        })
        provider = _mock_provider(plan_json)
        config = {"planning_model": "claude-haiku-4-5-20251001", "skip_simple": True}

        plan = await generate_plan(
            provider, "Refactor the authentication system",
            [], [{"name": "read_file"}, {"name": "write_file"}], config)

        assert plan is not None
        assert plan["complexity"] == "complex"
        assert len(plan["steps"]) == 3
        assert "read_file" in plan["tools_needed"]

    @pytest.mark.asyncio
    async def test_simple_request_returns_none(self):
        plan_json = json.dumps({
            "steps": ["Answer directly"],
            "complexity": "simple",
            "tools_needed": [],
            "estimated_iterations": 1,
        })
        provider = _mock_provider(plan_json)
        config = {"skip_simple": True}

        plan = await generate_plan(provider, "Hello!", [], [], config)
        assert plan is None

    @pytest.mark.asyncio
    async def test_simple_request_not_skipped_when_disabled(self):
        plan_json = json.dumps({
            "steps": ["Greet user"],
            "complexity": "simple",
            "tools_needed": [],
            "estimated_iterations": 1,
        })
        provider = _mock_provider(plan_json)
        config = {"skip_simple": False}

        plan = await generate_plan(provider, "Hello!", [], [], config)
        assert plan is not None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=Exception("API error"))
        config = {}

        plan = await generate_plan(provider, "Do something complex", [], [], config)
        assert plan is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        provider = _mock_provider("not valid json at all")
        config = {}

        plan = await generate_plan(provider, "test", [], [], config)
        assert plan is None

    @pytest.mark.asyncio
    async def test_markdown_wrapped_json(self):
        """Plan wrapped in markdown code blocks should be parsed."""
        plan_json = json.dumps({
            "steps": ["Step 1"], "complexity": "medium",
            "tools_needed": [], "estimated_iterations": 2,
        })
        provider = _mock_provider(f"```json\n{plan_json}\n```")
        config = {"skip_simple": True}

        plan = await generate_plan(provider, "Write code", [], [], config)
        assert plan is not None
        assert plan["complexity"] == "medium"

    @pytest.mark.asyncio
    async def test_missing_complexity_defaults_to_medium(self):
        plan_json = json.dumps({
            "steps": ["Step 1", "Step 2"],
            "tools_needed": ["read_file"],
        })
        provider = _mock_provider(plan_json)
        config = {"skip_simple": True}

        plan = await generate_plan(provider, "Analyze", [], [], config)
        assert plan is not None
        assert plan["complexity"] == "medium"

    @pytest.mark.asyncio
    async def test_memories_included_in_context(self):
        """Verify provider is called with memory context."""
        plan_json = json.dumps({
            "steps": ["Use context"],
            "complexity": "medium",
            "tools_needed": [],
            "estimated_iterations": 1,
        })
        provider = _mock_provider(plan_json)
        config = {"skip_simple": True}

        memories = [
            {"content": "User is a Python developer", "score": 0.8},
            {"content": "Low relevance", "score": 0.05},
        ]
        await generate_plan(provider, "Help me", memories, [], config)

        # Check the prompt sent to provider
        call_args = provider.complete.call_args
        prompt_text = call_args.kwargs["messages"][0]["content"]
        assert "Python developer" in prompt_text
        # Low-score memory should be filtered
        assert "Low relevance" not in prompt_text


class TestFormatPlan:
    """Test format_plan_for_prompt function."""

    def test_basic_formatting(self):
        plan = {
            "steps": ["Read the code", "Write tests"],
            "tools_needed": ["read_file", "write_file"],
            "estimated_iterations": 3,
        }
        text = format_plan_for_prompt(plan)
        assert "## Your execution plan:" in text
        assert "1. Read the code" in text
        assert "2. Write tests" in text
        assert "read_file, write_file" in text
        assert "3" in text

    def test_empty_tools(self):
        plan = {"steps": ["Think", "Answer"], "tools_needed": [], "estimated_iterations": 1}
        text = format_plan_for_prompt(plan)
        assert "Tools to use:" not in text

    def test_no_iterations(self):
        plan = {"steps": ["Do it"], "tools_needed": []}
        text = format_plan_for_prompt(plan)
        assert "Estimated iterations:" not in text


class TestReflection:
    """Test reflect_on_progress function."""

    @pytest.mark.asyncio
    async def test_no_change_needed(self):
        provider = _mock_provider("NO_CHANGE")
        config = {"planning_model": "claude-haiku-4-5-20251001"}
        plan = {"steps": ["Step 1", "Step 2"]}
        completed = [{"name": "read_file"}, {"name": "write_file"}, {"name": "exec_command"}]

        result = await reflect_on_progress(provider, plan, completed, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_adjustment_returned(self):
        provider = _mock_provider("Consider also checking the test files before proceeding.")
        config = {}
        plan = {"steps": ["Read code", "Write fix"]}
        completed = [{"name": "read_file"}, {"name": "read_file"}, {"name": "read_file"}]

        result = await reflect_on_progress(provider, plan, completed, config)
        assert result is not None
        assert "test files" in result

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=Exception("timeout"))
        config = {}

        result = await reflect_on_progress(
            provider, {"steps": []}, [{"name": "tool1"}], config)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_change_case_insensitive(self):
        provider = _mock_provider("No change needed, everything is on track.")
        config = {}

        result = await reflect_on_progress(
            provider, {"steps": []}, [], config)
        assert result is None
