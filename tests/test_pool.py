"""Tests for the multi-agent pool."""

import pytest

from liteagent.pool import AgentPool


class TestSingleAgentFallback:
    def test_no_agents_key_creates_main(self, tmp_path):
        config = {
            "agent": {"max_iterations": 3},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        pool = AgentPool.from_config(config)
        assert "main" in pool.list_agents()
        assert pool.default is not None
        pool.close_all()

    def test_empty_agents_key(self, tmp_path):
        config = {
            "agent": {"max_iterations": 3},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
            "agents": {},
        }
        pool = AgentPool.from_config(config)
        assert pool.list_agents() == ["main"]
        pool.close_all()


class TestMultiAgent:
    def test_two_agents(self, tmp_path):
        config = {
            "agent": {"max_iterations": 3},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "main.db"), "auto_learn": False},
            "tools": {"builtin": []},
            "agents": {
                "main": {"default": True},
                "helper": {
                    "memory": {"db_path": str(tmp_path / "helper.db")},
                },
            },
        }
        pool = AgentPool.from_config(config)
        assert sorted(pool.list_agents()) == ["helper", "main"]
        assert pool.default is pool.get("main")
        # Delegation tool should be wired
        assert pool.get("main").tools.has_tool("delegate")
        assert pool.get("helper").tools.has_tool("delegate")
        pool.close_all()

    def test_get_nonexistent(self, tmp_path):
        config = {
            "agent": {"max_iterations": 3},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        pool = AgentPool.from_config(config)
        assert pool.get("nonexistent") is None
        pool.close_all()

    def test_close_all_safe(self, tmp_path):
        config = {
            "agent": {"max_iterations": 3},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
        }
        pool = AgentPool.from_config(config)
        pool.close_all()
        # Should not raise on double close
        pool.close_all()
