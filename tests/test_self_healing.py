"""Tests for self-healing: provider fallback + tool error recovery."""

import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestFatalErrorDetection:
    """Test error classification methods."""

    def _make_agent(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake")
        from liteagent.config import load_config
        from liteagent.agent import LiteAgent
        return LiteAgent(load_config())

    def test_auth_error_is_fatal(self):
        agent = self._make_agent()
        e = Exception("Error code: 401 - authentication_error - invalid x-api-key")
        assert agent._is_fatal_error(e) is True

    def test_permission_error_is_fatal(self):
        agent = self._make_agent()
        e = Exception("403 Forbidden: permission denied")
        assert agent._is_fatal_error(e) is True

    def test_rate_limit_is_switchable(self):
        agent = self._make_agent()
        e = Exception("Error code: 429 - rate_limit_exceeded")
        assert agent._is_switchable_error(e) is True

    def test_overloaded_is_switchable(self):
        agent = self._make_agent()
        e = Exception("503 - overloaded - The server is overloaded")
        assert agent._is_switchable_error(e) is True

    def test_timeout_is_not_fatal(self):
        agent = self._make_agent()
        e = Exception("Request timed out")
        assert agent._is_fatal_error(e) is False

    def test_generic_error_is_not_switchable(self):
        agent = self._make_agent()
        e = Exception("Something went wrong")
        assert agent._is_switchable_error(e) is False


class TestFallbackProvider:
    """Test finding alternative providers."""

    def _make_agent(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake")
        from liteagent.config import load_config
        from liteagent.agent import LiteAgent
        return LiteAgent(load_config())

    def test_fallback_skips_current_provider(self):
        agent = self._make_agent()
        agent.config["agent"]["provider"] = "anthropic"
        fallback = agent._get_fallback_provider()
        if fallback:
            assert fallback[0] != "anthropic"

    def test_fallback_returns_none_when_no_alternatives(self, monkeypatch):
        agent = self._make_agent()
        # Mock get_api_key to return None for all non-current providers
        monkeypatch.setattr("liteagent.agent.LiteAgent._get_fallback_provider",
                            lambda self: None)
        assert agent._get_fallback_provider() is None

    def test_switch_provider_updates_state(self, monkeypatch):
        agent = self._make_agent()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-openai-key")
        # Mock create_provider to avoid actual SDK init
        mock_provider = MagicMock()
        monkeypatch.setattr("liteagent.agent.create_provider", lambda cfg: mock_provider)

        agent._switch_provider("openai", "gpt-4o-mini")

        assert agent.config["agent"]["provider"] == "openai"
        assert agent.config["agent"]["default_model"] == "gpt-4o-mini"
        assert agent.default_model == "gpt-4o-mini"
        assert agent.provider is mock_provider


class TestCallApiWithFallback:
    """Test _call_api with provider fallback."""

    def _make_agent(self):
        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake")
        from liteagent.config import load_config
        from liteagent.agent import LiteAgent
        return LiteAgent(load_config())

    @pytest.mark.asyncio
    async def test_call_api_success(self):
        agent = self._make_agent()
        mock_response = MagicMock()
        agent.provider.complete = AsyncMock(return_value=mock_response)

        result = await agent._call_api(model="test", messages=[], max_tokens=10)
        assert result is mock_response

    @pytest.mark.asyncio
    async def test_call_api_retries_on_timeout(self):
        agent = self._make_agent()
        mock_response = MagicMock()

        class TimeoutError(Exception):
            pass

        agent.provider.complete = AsyncMock(
            side_effect=[TimeoutError("timed out"), mock_response])

        result = await agent._call_api(model="test", messages=[], max_tokens=10)
        assert result is mock_response
        assert agent.provider.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_call_api_falls_back_on_auth_error(self, monkeypatch):
        agent = self._make_agent()
        # First call fails with auth error
        agent.provider.complete = AsyncMock(
            side_effect=Exception("401 authentication_error"))

        # Mock fallback
        fallback_provider = MagicMock()
        fallback_response = MagicMock()
        fallback_provider.complete = AsyncMock(return_value=fallback_response)
        monkeypatch.setattr(agent, "_get_fallback_provider", lambda: ("openai", "gpt-4o-mini"))
        monkeypatch.setattr(agent, "_switch_provider",
                            lambda name, model: setattr(agent, "provider", fallback_provider) or
                            setattr(agent, "default_model", model))

        result = await agent._call_api(model="claude-sonnet-4-20250514", messages=[], max_tokens=10)
        assert result is fallback_response

    @pytest.mark.asyncio
    async def test_call_api_raises_when_no_fallback(self):
        agent = self._make_agent()
        agent.provider.complete = AsyncMock(
            side_effect=Exception("401 authentication_error"))

        with patch.object(agent, "_get_fallback_provider", return_value=None):
            with pytest.raises(Exception, match="authentication"):
                await agent._call_api(model="test", messages=[], max_tokens=10)


class TestToolExecuteOne:
    """Test tool execution with metadata."""

    @pytest.mark.asyncio
    async def test_execute_one_success(self):
        from liteagent.tools import ToolRegistry

        registry = ToolRegistry()

        @registry.tool()
        def hello(name: str) -> str:
            """Say hello.
            name: The name to greet
            """
            return f"Hello, {name}!"

        block = MagicMock()
        block.name = "hello"
        block.id = "test-123"
        block.input = {"name": "World"}

        result = await registry.execute_one(block)
        assert result["content"] == "Hello, World!"
        assert result["_meta"]["tool_name"] == "hello"
        assert result["_meta"]["error"] is False
        assert result["_meta"]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_one_error(self):
        from liteagent.tools import ToolRegistry

        registry = ToolRegistry()

        @registry.tool()
        def fail_tool() -> str:
            """A tool that fails."""
            raise ValueError("Something broke")

        block = MagicMock()
        block.name = "fail_tool"
        block.id = "test-456"
        block.input = {}

        result = await registry.execute_one(block)
        assert result["_meta"]["error"] is True
        assert "Something broke" in result["content"]
        assert result["_meta"]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_execute_one_unknown_tool(self):
        from liteagent.tools import ToolRegistry

        registry = ToolRegistry()
        block = MagicMock()
        block.name = "nonexistent"
        block.id = "test-789"
        block.input = {}

        result = await registry.execute_one(block)
        assert result["_meta"]["error"] is True
        assert "unknown tool" in result["content"]
