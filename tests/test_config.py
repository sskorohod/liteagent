"""Tests for configuration loader."""

import json
import os
import pytest

from liteagent.config import load_config, validate_config, get_soul_prompt


class TestLoadConfig:

    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "agent": {"name": "TestAgent", "max_iterations": 5}
        }))
        config = load_config(str(config_file))
        assert config["agent"]["name"] == "TestAgent"
        assert config["agent"]["max_iterations"] == 5

    def test_load_missing_config_returns_empty(self, tmp_path):
        config = load_config(str(tmp_path / "nonexistent.json"))
        assert config == {}

    def test_env_var_resolution(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({}))
        load_config(str(config_file))
        assert os.environ.get("ANTHROPIC_API_KEY") == "test-key-123"


class TestValidateConfig:

    def test_valid_config_no_warnings(self):
        config = {
            "agent": {"name": "Test", "max_iterations": 10},
            "memory": {"db_path": "/tmp/test.db"},
            "cost": {"budget_daily_usd": 5.0},
        }
        warnings = validate_config(config)
        assert len(warnings) == 0

    def test_unknown_top_level_key(self):
        config = {"agnet": {}}  # Typo
        warnings = validate_config(config)
        assert any("agnet" in w for w in warnings)

    def test_unknown_cost_key(self):
        config = {"cost": {"budgett_daily_usd": 5.0}}  # Typo
        warnings = validate_config(config)
        assert any("budgett_daily_usd" in w for w in warnings)

    def test_unknown_agent_key(self):
        config = {"agent": {"max_iteratons": 10}}  # Typo
        warnings = validate_config(config)
        assert any("max_iteratons" in w for w in warnings)


class TestSoulPrompt:

    def test_load_existing_soul(self, tmp_path):
        soul_file = tmp_path / "soul.md"
        soul_file.write_text("You are a test agent.")
        config = {"agent": {"soul": str(soul_file)}}
        prompt = get_soul_prompt(config)
        assert prompt == "You are a test agent."

    def test_fallback_when_missing(self):
        config = {"agent": {"soul": "/nonexistent/soul.md"}}
        prompt = get_soul_prompt(config)
        assert "helpful AI assistant" in prompt
