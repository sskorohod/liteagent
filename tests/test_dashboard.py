"""Tests for dashboard API endpoints."""

import pytest
from pathlib import Path

from liteagent.agent import LiteAgent
from liteagent.channels.api import create_app


@pytest.fixture
def client(tmp_path):
    """Create FastAPI TestClient with real agent."""
    config = {
        "agent": {"max_iterations": 3},
        "cost": {"budget_daily_usd": 100.0},
        "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
        "tools": {"builtin": []},
        "channels": {"api": {"rate_limit": {"requests_per_minute": 100}}},
    }
    agent = LiteAgent(config)
    app = create_app(agent)
    from starlette.testclient import TestClient
    c = TestClient(app)
    yield c, agent
    agent.memory.close()


class TestHealthEndpoint:
    def test_health(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestOverviewEndpoint:
    def test_overview_returns_kpis(self, client):
        c, _ = client
        resp = c.get("/api/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_calls" in data
        assert "memory_count" in data
        assert "today_cost_usd" in data
        assert "tools_count" in data

    def test_overview_zero_state(self, client):
        c, _ = client
        data = c.get("/api/overview").json()
        assert data["total_calls"] == 0
        assert data["total_cost_usd"] == 0


class TestMemoriesEndpoints:
    def test_list_memories_empty(self, client):
        c, _ = client
        resp = c.get("/api/memories")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_delete_nonexistent(self, client):
        c, _ = client
        resp = c.delete("/api/memories/999")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_add_and_delete_memory(self, client):
        c, agent = client
        await agent.memory.remember("Test fact", "dashboard-user", "fact", 0.5)
        memories = c.get("/api/memories").json()
        assert len(memories) == 1
        resp = c.delete(f"/api/memories/{memories[0]['id']}")
        assert resp.status_code == 200
        assert c.get("/api/memories").json() == []


class TestUsageEndpoints:
    def test_usage_empty(self, client):
        c, _ = client
        resp = c.get("/api/usage?days=7")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_daily_usage_empty(self, client):
        c, _ = client
        resp = c.get("/api/usage/daily?days=14")
        assert resp.status_code == 200
        assert resp.json() == []


class TestToolsEndpoint:
    def test_tools_returns_list(self, client):
        c, _ = client
        resp = c.get("/api/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert isinstance(tools, list)
        # memory_search is always registered
        names = [t["name"] for t in tools]
        assert "memory_search" in names


class TestConfigEndpoint:
    def test_config_returns_dict(self, client):
        c, _ = client
        resp = c.get("/api/config")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)


class TestExportEndpoints:
    @pytest.mark.asyncio
    async def test_export_memories_json(self, client):
        c, agent = client
        await agent.memory.remember("Export test", "test-user", "fact", 0.5)
        resp = c.get("/api/export/memories?format=json")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    @pytest.mark.asyncio
    async def test_export_memories_csv(self, client):
        c, agent = client
        await agent.memory.remember("CSV test", "test-user", "fact", 0.5)
        resp = c.get("/api/export/memories?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "CSV test" in resp.text

    @pytest.mark.asyncio
    async def test_export_memories_markdown(self, client):
        c, agent = client
        await agent.memory.remember("MD test", "test-user", "fact", 0.5)
        resp = c.get("/api/export/memories?format=md")
        assert resp.status_code == 200
        assert "MD test" in resp.text

    def test_export_usage_json(self, client):
        c, _ = client
        resp = c.get("/api/export/usage?format=json")
        assert resp.status_code == 200

    def test_export_usage_csv(self, client):
        c, _ = client
        resp = c.get("/api/export/usage?format=csv")
        assert resp.status_code == 200


class TestMCPEndpoints:
    def test_mcp_servers_empty(self, client):
        c, _ = client
        resp = c.get("/api/mcp/servers")
        assert resp.status_code == 200
        assert resp.json() == []


class TestSchedulerEndpoint:
    def test_scheduler_jobs_empty(self, client):
        c, _ = client
        resp = c.get("/api/scheduler/jobs")
        assert resp.status_code == 200
        assert resp.json() == []


class TestProviderSettings:
    def test_get_providers_returns_all(self, client):
        c, _ = client
        resp = c.get("/api/settings/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "active_provider" in data
        assert "active_model" in data
        assert "providers" in data
        assert "anthropic" in data["providers"]
        assert "openai" in data["providers"]
        assert "gemini" in data["providers"]
        assert "ollama" in data["providers"]

    def test_provider_has_models(self, client):
        c, _ = client
        data = c.get("/api/settings/providers").json()
        for name, info in data["providers"].items():
            assert "models" in info
            assert len(info["models"]) > 0
            assert "has_key" in info

    def test_ollama_always_has_key(self, client):
        c, _ = client
        data = c.get("/api/settings/providers").json()
        assert data["providers"]["ollama"]["has_key"] is True
        assert data["providers"]["ollama"]["key_preview"] == "(local)"

    def test_save_key_missing_name(self, client):
        c, _ = client
        resp = c.post("/api/settings/provider/key",
                       json={"provider": "", "api_key": "sk-test"})
        assert resp.status_code == 400

    def test_save_key_missing_key(self, client):
        c, _ = client
        resp = c.post("/api/settings/provider/key",
                       json={"provider": "openai", "api_key": ""})
        assert resp.status_code == 400

    def test_save_key_unknown_provider(self, client):
        c, _ = client
        resp = c.post("/api/settings/provider/key",
                       json={"provider": "nonexistent", "api_key": "x"})
        assert resp.status_code == 400

    def test_save_key_bad_format_anthropic(self, client):
        """Anthropic key must start with sk-ant-."""
        c, _ = client
        resp = c.post("/api/settings/provider/key",
                       json={"provider": "anthropic", "api_key": "xai-wrong-prefix"})
        assert resp.status_code == 400
        assert "sk-ant-" in resp.json()["detail"]

    def test_save_key_bad_format_openai(self, client):
        """OpenAI key must start with sk-."""
        c, _ = client
        resp = c.post("/api/settings/provider/key",
                       json={"provider": "openai", "api_key": "bad-prefix-key"})
        assert resp.status_code == 400
        assert "sk-" in resp.json()["detail"]

    def test_save_key_valid_format(self, client, monkeypatch, tmp_path):
        """Valid key format should save successfully."""
        keys_path = tmp_path / "keys.json"
        monkeypatch.setattr("liteagent.config.KEYS_PATH", keys_path)
        monkeypatch.setattr("liteagent.config.KEYS_DIR", tmp_path)
        c, _ = client
        resp = c.post("/api/settings/provider/key",
                       json={"provider": "anthropic", "api_key": "sk-ant-valid-key-12345"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_apply_provider_missing_name(self, client):
        c, _ = client
        resp = c.post("/api/settings/provider",
                       json={"provider": "", "model": "gpt-4o"})
        assert resp.status_code == 400

    def test_apply_provider_unknown(self, client):
        c, _ = client
        resp = c.post("/api/settings/provider",
                       json={"provider": "nonexistent", "model": "x"})
        assert resp.status_code == 400

    def test_apply_anthropic_works(self, client, monkeypatch):
        """Anthropic SDK is always installed, so applying should work with key in env."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-fake-key-12345")
        c, _ = client
        resp = c.post("/api/settings/provider",
                       json={"provider": "anthropic", "model": "claude-haiku-4-5-20251001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True

    def test_delete_key_not_found(self, client):
        c, _ = client
        resp = c.delete("/api/settings/provider/openai/key")
        # May be 404 if no key saved
        assert resp.status_code in (200, 404)

    def test_test_provider_no_key(self, client):
        c, _ = client
        resp = c.post("/api/settings/provider/test",
                       json={"provider": "openai", "api_key": ""})
        data = resp.json()
        # Should return ok=false (no key or no SDK)
        assert data["ok"] is False


class TestKeyManagement:
    """Test config.py key management functions."""

    def test_save_and_load_key(self, tmp_path, monkeypatch):
        from liteagent.config import (
            load_provider_keys, save_provider_key,
            delete_provider_key, get_api_key, key_preview,
            KEYS_DIR,
        )
        import liteagent.config as config_mod

        # Override paths to use tmp_path
        monkeypatch.setattr(config_mod, "KEYS_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "KEYS_PATH", tmp_path / "keys.json")

        # Save a key
        save_provider_key("openai", "sk-test-key-12345")
        keys = load_provider_keys()
        assert keys["openai"] == "sk-test-key-12345"

        # Get key
        key = get_api_key("openai")
        assert key == "sk-test-key-12345"

        # Preview
        assert key_preview("sk-test-key-12345") == "sk-tes...2345"

        # Delete
        assert delete_provider_key("openai") is True
        assert delete_provider_key("openai") is False
        assert get_api_key("openai") is None  # unless env var set

    def test_key_preview_short(self):
        from liteagent.config import key_preview
        assert key_preview("") == ""
        assert key_preview("abcdefghij") == "abc...ij"
        assert key_preview("sk-ant-api03-longkey1234") == "sk-ant...1234"

    def test_get_api_key_from_env(self, monkeypatch):
        from liteagent.config import get_api_key
        import liteagent.config as config_mod
        # No keys.json → should fall back to env
        monkeypatch.setattr(config_mod, "KEYS_PATH", Path("/nonexistent/keys.json"))
        monkeypatch.setenv("OPENAI_API_KEY", "env-key-123")
        key = get_api_key("openai")
        assert key == "env-key-123"
