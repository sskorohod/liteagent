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

    def test_memory_extraction_keys_are_known(self):
        config = {
            "memory": {
                "extraction_provider": "ollama",
                "extraction_model": "qwen2.5:latest",
                "extraction_max_concurrency": 1,
                "memory_exchange_enabled": True,
                "memory_exchange_pack_budget_tokens": 420,
                "memory_exchange_max_packs": 2,
                "memory_exchange_context_budget_tokens": 700,
                "memory_exchange_daemon_enabled": True,
                "memory_exchange_daemon_interval_sec": 1.0,
                "memory_exchange_daemon_batch_size": 3,
                "memory_exchange_daemon_auto_pause": True,
                "memory_exchange_daemon_pause_active_requests": 1,
                "memory_exchange_daemon_pause_queued_requests": 2,
                "memory_exchange_queue_max_pending": 5000,
                "memory_exchange_max_attempts": 3,
                "memory_local_worker_enabled": True,
                "memory_local_worker_interval_sec": 12.0,
                "memory_local_worker_batch_size": 24,
                "user_aliases": {"dashboard-user": "tg-123"},
                "shadow_twin_enabled": True,
                "shadow_twin_predictions": 3,
                "shadow_twin_use_llm": True,
                "_default_model": "qwen-plus",
            }
        }
        warnings = validate_config(config)
        assert not any("Unknown memory key" in w for w in warnings)


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
