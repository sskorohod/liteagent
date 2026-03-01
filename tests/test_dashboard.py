"""Tests for dashboard API endpoints."""

import pytest

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

    def test_add_and_delete_memory(self, client):
        c, agent = client
        agent.memory.remember("Test fact", "dashboard-user", "fact", 0.5)
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
    def test_export_memories_json(self, client):
        c, agent = client
        agent.memory.remember("Export test", "test-user", "fact", 0.5)
        resp = c.get("/api/export/memories?format=json")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_export_memories_csv(self, client):
        c, agent = client
        agent.memory.remember("CSV test", "test-user", "fact", 0.5)
        resp = c.get("/api/export/memories?format=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "CSV test" in resp.text

    def test_export_memories_markdown(self, client):
        c, agent = client
        agent.memory.remember("MD test", "test-user", "fact", 0.5)
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
