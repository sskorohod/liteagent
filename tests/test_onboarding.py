"""Tests for onboarding — agent-driven setup on first launch."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock


class TestNeedsOnboarding:
    """Test onboarding detection logic."""

    def test_needs_onboarding_when_no_flag(self):
        """Should need onboarding when app:onboarding_complete is not set."""
        from liteagent.onboarding import needs_onboarding
        agent = MagicMock()
        agent.memory.get_state.return_value = None
        assert needs_onboarding(agent) is True
        agent.memory.get_state.assert_called_with("app:onboarding_complete")

    def test_no_onboarding_when_complete(self):
        """Should not need onboarding when flag is set."""
        from liteagent.onboarding import needs_onboarding
        agent = MagicMock()
        agent.memory.get_state.return_value = True
        assert needs_onboarding(agent) is False

    def test_no_onboarding_when_false_string(self):
        """Any truthy value means onboarding is complete."""
        from liteagent.onboarding import needs_onboarding
        agent = MagicMock()
        agent.memory.get_state.return_value = "true"
        assert needs_onboarding(agent) is False


class TestOnboardingPrompt:
    """Test that onboarding prompt is well-formed."""

    def test_prompt_not_empty(self):
        from liteagent.onboarding import ONBOARDING_PROMPT
        assert len(ONBOARDING_PROMPT) > 100

    def test_prompt_mentions_features(self):
        from liteagent.onboarding import ONBOARDING_PROMPT
        assert "style_adaptation" in ONBOARDING_PROMPT or "Адаптация стиля" in ONBOARDING_PROMPT

    def test_prompt_mentions_setup_agent(self):
        from liteagent.onboarding import ONBOARDING_PROMPT
        assert "setup_agent" in ONBOARDING_PROMPT


class TestFeatureCatalog:
    """Test feature catalog and presets."""

    def test_catalog_has_all_features(self):
        from liteagent.onboarding import FEATURE_CATALOG
        expected = {
            "style_adaptation", "confidence_gate", "self_evolving_prompt",
            "proactive_agent", "skill_crystallization", "auto_tool_synthesis",
            "dream_cycle", "counterfactual_replay",
        }
        assert set(FEATURE_CATALOG.keys()) == expected

    def test_basic_preset_is_subset(self):
        from liteagent.onboarding import FEATURE_PRESETS, FEATURE_CATALOG
        for f in FEATURE_PRESETS["basic"]:
            assert f in FEATURE_CATALOG

    def test_all_preset_matches_catalog(self):
        from liteagent.onboarding import FEATURE_PRESETS, FEATURE_CATALOG
        assert set(FEATURE_PRESETS["all"]) == set(FEATURE_CATALOG.keys())


class TestRegisterOnboardingTool:
    """Test tool registration/unregistration."""

    def test_register_adds_tool(self):
        from liteagent.onboarding import register_onboarding_tool
        from liteagent.tools import ToolRegistry
        agent = MagicMock()
        agent.tools = ToolRegistry()
        register_onboarding_tool(agent)
        assert "setup_agent" in agent.tools._tools
        assert "setup_agent" in agent.tools._handlers

    def test_unregister_removes_tool(self):
        from liteagent.onboarding import register_onboarding_tool, unregister_onboarding_tool
        from liteagent.tools import ToolRegistry
        agent = MagicMock()
        agent.tools = ToolRegistry()
        register_onboarding_tool(agent)
        unregister_onboarding_tool(agent)
        assert "setup_agent" not in agent.tools._tools
        assert "setup_agent" not in agent.tools._handlers

    def test_unregister_idempotent(self):
        """Unregistering when not registered should not raise."""
        from liteagent.onboarding import unregister_onboarding_tool
        from liteagent.tools import ToolRegistry
        agent = MagicMock()
        agent.tools = ToolRegistry()
        unregister_onboarding_tool(agent)  # Should not raise


class TestSetupAgentTool:
    """Test the setup_agent tool execution."""

    def _make_agent(self, tmp_path):
        from liteagent.tools import ToolRegistry
        agent = MagicMock()
        agent.tools = ToolRegistry()
        agent.config = {
            "agent": {"name": "LiteAgent", "soul": str(tmp_path / "soul.md")},
            "features": {
                "style_adaptation": {"enabled": False},
                "confidence_gate": {"enabled": False},
                "self_evolving_prompt": {"enabled": False},
                "proactive_agent": {"enabled": False},
                "skill_crystallization": {"enabled": False},
                "auto_tool_synthesis": {"enabled": False},
                "dream_cycle": {"enabled": False},
                "counterfactual_replay": {"enabled": False},
            }
        }
        agent.memory = MagicMock()
        return agent

    def test_writes_soul_md(self, tmp_path, monkeypatch):
        from liteagent.onboarding import register_onboarding_tool

        agent = self._make_agent(tmp_path)
        soul_path = tmp_path / "soul.md"
        soul_path.write_text("old soul")
        agent.config["agent"]["soul"] = str(soul_path)

        # Redirect config save to tmp_path
        monkeypatch.setattr("liteagent.config.DEFAULT_CONFIG_WRITE_PATH", tmp_path / "config.json")

        register_onboarding_tool(agent)
        handler = agent.tools._handlers["setup_agent"]
        result = handler(
            soul_prompt="You are a helpful coder.",
            agent_name="CodeBot",
            features_preset="basic",
        )

        assert soul_path.read_text() == "You are a helpful coder."
        assert "CodeBot" in result

    def test_updates_features(self, tmp_path, monkeypatch):
        from liteagent.onboarding import register_onboarding_tool

        agent = self._make_agent(tmp_path)
        soul_path = tmp_path / "soul.md"
        soul_path.write_text("old")
        agent.config["agent"]["soul"] = str(soul_path)

        saved_configs = []
        monkeypatch.setattr("liteagent.config.DEFAULT_CONFIG_WRITE_PATH", tmp_path / "config.json")

        register_onboarding_tool(agent)
        handler = agent.tools._handlers["setup_agent"]
        handler(
            soul_prompt="test",
            agent_name="Test",
            features_preset="basic",
        )

        # basic = style_adaptation, confidence_gate, skill_crystallization
        assert agent.config["features"]["style_adaptation"]["enabled"] is True
        assert agent.config["features"]["confidence_gate"]["enabled"] is True
        assert agent.config["features"]["skill_crystallization"]["enabled"] is True
        assert agent.config["features"]["dream_cycle"]["enabled"] is False

    def test_custom_features(self, tmp_path, monkeypatch):
        from liteagent.onboarding import register_onboarding_tool

        agent = self._make_agent(tmp_path)
        soul_path = tmp_path / "soul.md"
        soul_path.write_text("old")
        agent.config["agent"]["soul"] = str(soul_path)
        monkeypatch.setattr("liteagent.config.DEFAULT_CONFIG_WRITE_PATH", tmp_path / "config.json")

        register_onboarding_tool(agent)
        handler = agent.tools._handlers["setup_agent"]
        handler(
            soul_prompt="test",
            agent_name="Test",
            enabled_features="dream_cycle, proactive_agent",
        )

        assert agent.config["features"]["dream_cycle"]["enabled"] is True
        assert agent.config["features"]["proactive_agent"]["enabled"] is True
        assert agent.config["features"]["style_adaptation"]["enabled"] is False

    def test_sets_onboarding_flag(self, tmp_path, monkeypatch):
        from liteagent.onboarding import register_onboarding_tool

        agent = self._make_agent(tmp_path)
        soul_path = tmp_path / "soul.md"
        soul_path.write_text("old")
        agent.config["agent"]["soul"] = str(soul_path)
        monkeypatch.setattr("liteagent.config.DEFAULT_CONFIG_WRITE_PATH", tmp_path / "config.json")

        register_onboarding_tool(agent)
        handler = agent.tools._handlers["setup_agent"]
        handler(soul_prompt="test", agent_name="Test")

        agent.memory.set_state.assert_called_with("app:onboarding_complete", True)

    def test_updates_runtime_soul(self, tmp_path, monkeypatch):
        from liteagent.onboarding import register_onboarding_tool

        agent = self._make_agent(tmp_path)
        agent._soul_prompt = "old soul"
        agent._features = {}
        soul_path = tmp_path / "soul.md"
        soul_path.write_text("old")
        agent.config["agent"]["soul"] = str(soul_path)
        monkeypatch.setattr("liteagent.config.DEFAULT_CONFIG_WRITE_PATH", tmp_path / "config.json")

        register_onboarding_tool(agent)
        handler = agent.tools._handlers["setup_agent"]
        handler(soul_prompt="New personalized prompt", agent_name="Bot")

        assert agent._soul_prompt == "New personalized prompt"

    def test_invalid_features_ignored(self, tmp_path, monkeypatch):
        from liteagent.onboarding import register_onboarding_tool

        agent = self._make_agent(tmp_path)
        soul_path = tmp_path / "soul.md"
        soul_path.write_text("old")
        agent.config["agent"]["soul"] = str(soul_path)
        monkeypatch.setattr("liteagent.config.DEFAULT_CONFIG_WRITE_PATH", tmp_path / "config.json")

        register_onboarding_tool(agent)
        handler = agent.tools._handlers["setup_agent"]
        result = handler(
            soul_prompt="test",
            agent_name="Test",
            enabled_features="fake_feature, style_adaptation",
        )

        assert agent.config["features"]["style_adaptation"]["enabled"] is True
        # fake_feature should be silently ignored
        assert "fake_feature" not in agent.config["features"]


class TestSaveConfig:
    """Test config saving."""

    def test_save_and_reload(self, tmp_path):
        from liteagent.config import save_config
        config = {"agent": {"name": "TestBot"}, "features": {"x": {"enabled": True}}}
        path = tmp_path / "config.json"
        save_config(config, str(path))

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["agent"]["name"] == "TestBot"
        assert loaded["features"]["x"]["enabled"] is True

    def test_save_excludes_internal_keys(self, tmp_path):
        from liteagent.config import save_config
        config = {"agent": {"name": "Bot"}, "_resolved": True}
        path = tmp_path / "config.json"
        save_config(config, str(path))

        with open(path) as f:
            loaded = json.load(f)
        assert "_resolved" not in loaded

    def test_save_unicode(self, tmp_path):
        from liteagent.config import save_config
        config = {"agent": {"name": "Бот-помощник"}}
        path = tmp_path / "config.json"
        save_config(config, str(path))

        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["agent"]["name"] == "Бот-помощник"


class TestAgentOnboardingIntegration:
    """Test that agent correctly switches between onboarding and normal mode."""

    def test_build_system_prompt_returns_onboarding(self):
        """When onboarding not complete, should return onboarding prompt."""
        from liteagent.config import load_config
        from liteagent.agent import LiteAgent
        from liteagent.onboarding import ONBOARDING_PROMPT
        import os

        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake")
        config = load_config()
        agent = LiteAgent(config)
        # Force onboarding mode
        agent.memory.set_state("app:onboarding_complete", None)
        # Clear the flag by manipulating directly
        agent.memory.db.execute("DELETE FROM app_state WHERE key='app:onboarding_complete'")
        agent.memory.db.commit()

        prompt = agent._build_system_prompt("hello", "test-user")
        assert prompt == ONBOARDING_PROMPT

    def test_build_system_prompt_returns_soul_when_complete(self):
        """When onboarding complete, should return normal soul prompt."""
        from liteagent.config import load_config
        from liteagent.agent import LiteAgent
        from liteagent.onboarding import ONBOARDING_PROMPT
        import os

        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake")
        config = load_config()
        agent = LiteAgent(config)
        agent.memory.set_state("app:onboarding_complete", True)

        prompt = agent._build_system_prompt("hello", "test-user")
        assert prompt != ONBOARDING_PROMPT

    def test_ensure_onboarding_tool_registers(self):
        """Tool should be registered when onboarding needed."""
        from liteagent.config import load_config
        from liteagent.agent import LiteAgent
        import os

        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake")
        config = load_config()
        agent = LiteAgent(config)
        agent.memory.db.execute("DELETE FROM app_state WHERE key='app:onboarding_complete'")
        agent.memory.db.commit()

        agent._ensure_onboarding_tool()
        assert "setup_agent" in agent.tools._tools

    def test_ensure_onboarding_tool_unregisters(self):
        """Tool should be removed when onboarding complete."""
        from liteagent.config import load_config
        from liteagent.agent import LiteAgent
        import os

        os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-fake")
        config = load_config()
        agent = LiteAgent(config)

        # First register it
        agent.memory.db.execute("DELETE FROM app_state WHERE key='app:onboarding_complete'")
        agent.memory.db.commit()
        agent._ensure_onboarding_tool()
        assert "setup_agent" in agent.tools._tools

        # Now complete onboarding
        agent.memory.set_state("app:onboarding_complete", True)
        agent._ensure_onboarding_tool()
        assert "setup_agent" not in agent.tools._tools
