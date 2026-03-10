import pytest

from liteagent.agent import LiteAgent
from liteagent.tasks import TaskManager


@pytest.fixture
def document_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = {
        "agent": {"max_iterations": 2},
        "cost": {"budget_daily_usd": 100.0},
        "memory": {"db_path": str(tmp_path / "memory.db"), "auto_learn": False},
        "tools": {"builtin": []},
        "channels": {"api": {"auth_enabled": False}},
    }
    agent = LiteAgent(config)
    agent.provider = None
    tm = TaskManager(agent.memory.db)
    agent.enable_tasks(tm)
    yield agent
    agent.memory.close()


@pytest.mark.asyncio
async def test_document_pipeline_creates_notes_tasks_and_calendar(document_agent):
    result = await document_agent.process_document_upload(
        (
            b"Passport renewal notice\n"
            b"This passport expires on 2026-12-30.\n"
            b"Renew passport before expiration and prepare all supporting documents.\n"
        ),
        "passport.txt",
        source="dashboard",
        user_id="dashboard-user",
        mime_type="text/plain",
    )

    assert result["status"] == "ok"
    assert result["analysis"]["summary"]
    assert result["notes_saved"] >= 1
    assert result["tasks_created"]
    assert result["calendar_events"]

    events = document_agent._document_pipeline.list_calendar_events("dashboard-user", limit=5)
    assert events
    assert events[0]["source_ref"] == result["review_id"]

    reviews = document_agent._document_pipeline.list_reviews("dashboard-user", limit=5)
    assert reviews
    assert reviews[0]["review_id"] == result["review_id"]


@pytest.mark.asyncio
async def test_document_pipeline_respects_missing_kb_and_storage(document_agent):
    result = await document_agent.process_document_upload(
        b"Invoice total 1500 USD due 2026-10-15",
        "invoice.txt",
        source="dashboard",
        user_id="dashboard-user",
        mime_type="text/plain",
    )

    assert result["storage"]["enabled"] is False
    assert result["knowledge_base"]["enabled"] is False
    assert result["analysis"]["document_type"]
