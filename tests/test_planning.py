"""Tests for internal monologue / planning module."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from liteagent.planning import (
    generate_plan, format_plan_for_prompt, reflect_on_progress,
    resolve_planning_model, track_step_completion,
)


def _mock_provider(response_text: str, cls_name: str = "AnthropicProvider"):
    """Create a mock provider that returns given text."""
    # Create a real class so __class__.__name__ works correctly
    ProviderClass = type(cls_name, (), {"complete": None})
    provider = ProviderClass()
    result = MagicMock()
    result.content = [MagicMock(text=response_text)]
    provider.complete = AsyncMock(return_value=result)
    return provider


# ══════════════════════════════════════════
# resolve_planning_model
# ══════════════════════════════════════════

class TestResolvePlanningModel:
    """Test automatic planning model selection."""

    def test_explicit_model(self):
        provider = _mock_provider("")
        assert resolve_planning_model(provider, {"planning_model": "my-model"}) == "my-model"

    def test_auto_returns_default_string(self):
        provider = _mock_provider("")
        result = resolve_planning_model(provider, {"planning_model": "auto"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_anthropic_cheapest(self):
        provider = _mock_provider("", cls_name="AnthropicProvider")
        result = resolve_planning_model(provider, {})
        assert "haiku" in result

    def test_openai_cheapest(self):
        provider = _mock_provider("", cls_name="OpenAIProvider")
        result = resolve_planning_model(provider, {})
        assert "nano" in result or "mini" in result or "gpt" in result

    def test_gemini_cheapest(self):
        provider = _mock_provider("", cls_name="GeminiProvider")
        result = resolve_planning_model(provider, {})
        assert "gemini" in result

    def test_ollama_uses_default_model(self):
        provider = _mock_provider("", cls_name="OllamaProvider")
        result = resolve_planning_model(provider, {"_default_model": "llama3.1:latest"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_provider_fallback(self):
        provider = _mock_provider("", cls_name="UnknownProvider")
        result = resolve_planning_model(provider, {})
        assert isinstance(result, str)


# ══════════════════════════════════════════
# generate_plan
# ══════════════════════════════════════════

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
        provider = _mock_provider("")
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
    async def test_estimated_iterations_clamped(self):
        """Estimated iterations should be clamped to 1-10."""
        plan_json = json.dumps({
            "steps": ["Step 1"],
            "complexity": "medium",
            "tools_needed": [],
            "estimated_iterations": 99,
        })
        provider = _mock_provider(plan_json)
        config = {"skip_simple": True}

        plan = await generate_plan(provider, "test", [], [], config)
        assert plan is not None
        assert plan["estimated_iterations"] == 10

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


# ══════════════════════════════════════════
# format_plan_for_prompt
# ══════════════════════════════════════════

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
        assert "Follow this plan" in text

    def test_empty_tools(self):
        plan = {"steps": ["Think", "Answer"], "tools_needed": [], "estimated_iterations": 1}
        text = format_plan_for_prompt(plan)
        assert "Tools to use:" not in text

    def test_no_iterations(self):
        plan = {"steps": ["Do it"], "tools_needed": []}
        text = format_plan_for_prompt(plan)
        assert "Estimated iterations:" not in text


# ══════════════════════════════════════════
# reflect_on_progress (updated signature)
# ══════════════════════════════════════════

class TestReflection:
    """Test reflect_on_progress function."""

    @pytest.mark.asyncio
    async def test_no_change_needed(self):
        provider = _mock_provider("NO_CHANGE")
        config = {"planning_model": "claude-haiku-4-5-20251001"}
        plan = {"steps": ["Step 1", "Step 2"]}
        completed = [{"name": "read_file"}, {"name": "write_file"}, {"name": "exec_command"}]

        result = await reflect_on_progress(provider, plan, completed, None, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_adjustment_returned(self):
        provider = _mock_provider("Consider also checking the test files before proceeding.")
        config = {}
        plan = {"steps": ["Read code", "Write fix"]}
        completed = [{"name": "read_file"}, {"name": "read_file"}, {"name": "read_file"}]

        result = await reflect_on_progress(provider, plan, completed, None, config)
        assert result is not None
        assert "test files" in result

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        provider = _mock_provider("")
        provider.complete = AsyncMock(side_effect=Exception("timeout"))
        config = {}

        result = await reflect_on_progress(
            provider, {"steps": []}, [{"name": "tool1"}], None, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_no_change_case_insensitive(self):
        provider = _mock_provider("No change needed, everything is on track.")
        config = {}

        result = await reflect_on_progress(
            provider, {"steps": []}, [], None, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_results_included_in_prompt(self):
        """Tool results summary should appear in the reflection prompt."""
        provider = _mock_provider("NO_CHANGE")
        config = {"planning_model": "test-model"}
        plan = {"steps": ["Read and fix"]}
        completed = [{"name": "read_file"}, {"name": "exec_command"}]
        results = ["def main(): pass", "FAIL: test_auth AssertionError"]

        await reflect_on_progress(provider, plan, completed, results, config)

        call_args = provider.complete.call_args
        prompt = call_args.kwargs["messages"][0]["content"]
        assert "read_file" in prompt
        assert "FAIL" in prompt
        assert "exec_command" in prompt


# ══════════════════════════════════════════
# track_step_completion
# ══════════════════════════════════════════

class TestStepTracking:
    """Test zero-cost step completion tracking."""

    def test_basic_tracking(self):
        plan = {"steps": ["Read the file", "Write tests", "Run tests"]}
        tool_calls = [
            {"name": "read_file", "input": {"path": "/tmp/main.py"}},
            {"name": "write_file", "input": {"path": "/tmp/test.py", "content": "test code"}},
        ]
        results = ["def main(): pass", "ok"]

        tracking = track_step_completion(plan, tool_calls, results)
        assert tracking["total"] == 3
        assert tracking["completed_count"] >= 1
        # "Read the file" should match read_file tool
        assert tracking["steps"][0]["status"] == "done"

    def test_empty_plan(self):
        tracking = track_step_completion({"steps": []}, [], [])
        assert tracking["total"] == 0
        assert tracking["completed_count"] == 0
        assert tracking["steps"] == []

    def test_no_matches(self):
        plan = {"steps": ["Deploy to production"]}
        tool_calls = [{"name": "read_file", "input": {"path": "/tmp/x"}}]

        tracking = track_step_completion(plan, tool_calls, None)
        assert tracking["completed_count"] == 0
        assert tracking["steps"][0]["status"] == "pending"

    def test_with_results_context(self):
        plan = {"steps": ["Search for error logs"]}
        tool_calls = [{"name": "exec_command", "input": {"command": "grep error"}}]
        results = ["error: connection refused at line 42 in logs"]

        tracking = track_step_completion(plan, tool_calls, results)
        # "error" and "logs" appear in both step and results
        assert tracking["steps"][0]["status"] == "done"


# ══════════════════════════════════════════
# Message alternation fix
# ══════════════════════════════════════════

class TestMessageAlternation:
    """Test that reflection merges into tool_results (no consecutive user msgs)."""

    def test_reflection_merges_into_tool_results(self):
        """Reflection should be appended to existing tool_results user message."""
        messages = [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "1", "name": "read_file", "input": {}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": "file data"}
            ]},
        ]
        # Simulate what _apply_reflection does
        adjustment = "Consider also checking logs"
        last_msg = messages[-1]
        if (last_msg.get("role") == "user"
                and isinstance(last_msg.get("content"), list)):
            last_msg["content"].append({
                "type": "text",
                "text": f"\n[Internal reflection: {adjustment}]",
            })

        # Verify: still only 2 messages, no consecutive user messages
        assert len(messages) == 2
        assert messages[-1]["role"] == "user"
        assert len(messages[-1]["content"]) == 2
        assert "reflection" in messages[-1]["content"][-1]["text"]
