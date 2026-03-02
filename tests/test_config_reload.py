"""Tests for config watcher module."""
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from liteagent.config_watcher import config_diff, ConfigWatcher


class TestConfigDiff:
    """Test config diff calculation."""

    def test_simple_change(self):
        old = {"agent": {"default_model": "old-model"}}
        new = {"agent": {"default_model": "new-model"}}
        changes = config_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["path"] == "agent.default_model"
        assert changes[0]["old"] == "old-model"
        assert changes[0]["new"] == "new-model"

    def test_no_changes(self):
        cfg = {"agent": {"provider": "ollama"}}
        changes = config_diff(cfg, cfg)
        assert len(changes) == 0

    def test_nested_change(self):
        old = {"cost": {"cascade_routing": True, "budget_daily_usd": 5.0}}
        new = {"cost": {"cascade_routing": False, "budget_daily_usd": 10.0}}
        changes = config_diff(old, new)
        assert len(changes) == 2

    def test_added_key(self):
        old = {"agent": {}}
        new = {"agent": {"max_iterations": 20}}
        changes = config_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["old"] is None

    def test_non_reloadable_flagged(self):
        old = {"agent": {"provider": "ollama"}}
        new = {"agent": {"provider": "openai"}}
        changes = config_diff(old, new)
        assert len(changes) == 1
        assert changes[0]["reloadable"] is False


class TestConfigWatcher:
    """Test ConfigWatcher lifecycle."""

    @pytest.fixture
    def config_file(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"agent": {"default_model": "test"}}))
        return path

    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.config = {"agent": {"default_model": "test"}}
        agent.apply_config_update = MagicMock()
        return agent

    def test_watcher_init(self, config_file, mock_agent):
        watcher = ConfigWatcher(str(config_file), mock_agent)
        assert watcher._path == config_file
        assert watcher._last_hash is not None

    async def test_force_reload(self, config_file, mock_agent):
        watcher = ConfigWatcher(str(config_file), mock_agent)
        new_cfg = {"agent": {"default_model": "new-model"}}
        # Modify config
        config_file.write_text(json.dumps(new_cfg))
        # Patch load_config where it is defined (liteagent.config),
        # since _reload imports it with: from .config import load_config
        with patch("liteagent.config.load_config", return_value=new_cfg):
            await watcher.force_reload()
        mock_agent.apply_config_update.assert_called_once_with(new_cfg)

    async def test_no_reload_if_unchanged(self, config_file, mock_agent):
        watcher = ConfigWatcher(str(config_file), mock_agent)
        # Return same config so no diff detected
        with patch("liteagent.config.load_config",
                   return_value={"agent": {"default_model": "test"}}):
            await watcher.force_reload()
        # No changes => apply_config_update not called
        mock_agent.apply_config_update.assert_not_called()
