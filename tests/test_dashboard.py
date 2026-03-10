"""Tests for dashboard API endpoints."""

import pytest
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock

from liteagent.agent import LiteAgent
from liteagent.channels.api import create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create FastAPI TestClient with real agent."""
    import liteagent.config as config_mod
    config = {
        "agent": {"max_iterations": 3},
        "cost": {"budget_daily_usd": 100.0},
        "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
        "tools": {"builtin": []},
        "channels": {"api": {"rate_limit": {"requests_per_minute": 100}, "auth_enabled": False}},
    }
    agent = LiteAgent(config)
    app = create_app(agent)
    # Prevent tests from overwriting real config.json
    monkeypatch.setattr("liteagent.config.save_config", lambda *a, **kw: None)
    # Isolate keys.json to prevent real key deletion
    monkeypatch.setattr(config_mod, "KEYS_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "KEYS_PATH", tmp_path / "keys.json")
    monkeypatch.setattr(config_mod, "KEYS_BACKUP_PATH", tmp_path / "keys.json.bak")
    from starlette.testclient import TestClient
    c = TestClient(app)
    yield c, agent
    agent.memory.close()


class TestAPISecurityDefaults:
    def test_public_host_requires_password_by_default(self, tmp_path):
        config = {
            "agent": {"max_iterations": 1},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
            "channels": {"api": {"host": "0.0.0.0", "rate_limit": {"requests_per_minute": 100}}},
        }
        agent = LiteAgent(config)
        with pytest.raises(RuntimeError, match="Unsafe API config"):
            create_app(agent)
        agent.memory.close()

    def test_public_host_can_be_explicitly_unauthenticated(self, tmp_path):
        config = {
            "agent": {"max_iterations": 1},
            "cost": {"budget_daily_usd": 100.0},
            "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
            "tools": {"builtin": []},
            "channels": {"api": {
                "host": "0.0.0.0",
                "allow_unauthenticated_public": True,
                "rate_limit": {"requests_per_minute": 100},
            }},
        }
        agent = LiteAgent(config)
        app = create_app(agent)
        assert app is not None
        agent.memory.close()


class TestHealthEndpoint:
    def test_health(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestFileRevealEndpoint:
    def test_reveal_file_requires_path(self, client):
        c, _ = client
        resp = c.post("/api/files/reveal", json={})
        assert resp.status_code == 400

    def test_reveal_file_opens_parent_folder(self, client, tmp_path, monkeypatch):
        c, _ = client
        target = tmp_path / "notes.md"
        target.write_text("hello", encoding="utf-8")
        calls = []

        monkeypatch.setattr(
            "liteagent.channels.dashboard._reveal_in_file_manager",
            lambda path: calls.append(str(path)),
        )

        resp = c.post("/api/files/reveal", json={"path": str(target)})
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["path"] == str(target)
        assert data["folder"] == str(target.parent)
        assert calls == [str(target)]

    def test_reveal_file_strips_line_suffix(self, client, tmp_path, monkeypatch):
        c, _ = client
        target = tmp_path / "report.md"
        target.write_text("hello", encoding="utf-8")
        calls = []

        monkeypatch.setattr(
            "liteagent.channels.dashboard._reveal_in_file_manager",
            lambda path: calls.append(str(path)),
        )

        resp = c.post("/api/files/reveal", json={"path": f"{target}:12"})
        assert resp.status_code == 200
        assert calls == [str(target)]


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


class TestThinkingCloudEndpoint:
    def test_memory_thinking_returns_structured_cloud(self, client):
        c, agent = client
        agent.memory.upsert_thinking_note(
            "dashboard-user",
            "direction",
            "Move the agent toward a more autonomous local-first workflow",
            themes=["autonomy", "local models"],
            confidence=0.82,
            strategic_importance=0.9,
        )
        agent.memory.upsert_thinking_note(
            "dashboard-user",
            "open_question",
            "How to make self-healing reliable without excessive prompts?",
            themes=["self-healing", "autonomy"],
            confidence=0.78,
            strategic_importance=0.84,
        )

        resp = c.get("/api/memory/thinking?user_id=dashboard-user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "dashboard-user"
        assert data["cloud"]["enabled"] is True
        assert data["cloud"]["overview"]["total_notes"] >= 2
        assert any(item["type"] == "direction" for item in data["cloud"]["directions"])
        assert any(theme["label"] == "autonomy" for theme in data["cloud"]["themes"])

    def test_memory_human_support_returns_snapshot(self, client):
        c, agent = client
        now = "2026-03-09T23:40:00"
        for text in (
            "Я перегружен и не могу сосредоточиться",
            "Нужно продлить документы до дедлайна",
            "Снова работаю поздно ночью",
        ):
            agent.memory.db.execute(
                "INSERT INTO interaction_log (user_id, user_input, agent_response, tool_calls_json, success, confidence, model_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("dashboard-user", text, "done", "[]", 1, 0.9, "local", now),
            )
        agent.memory.db.commit()

        resp = c.get("/api/memory/human_support?user_id=dashboard-user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "dashboard-user"
        assert data["support"]["enabled"] is True
        assert data["support"]["metrics"]["overload_signals"] >= 1
        assert isinstance(data["support"]["opportunities"], list)


class TestOpsActiveEndpoint:
    def test_ops_active_contains_background_fields(self, client):
        c, _ = client
        resp = c.get("/api/ops/active")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests" in data
        assert "queued" in data
        assert "scheduler_jobs_running" in data
        assert "background_tasks_running" in data
        assert "background_pending" in data
        assert "background_daemon" in data
        assert "goals_running" in data
        assert "goals_pending" in data
        assert "goal_coordinator" in data
        assert "summary" in data
        assert "lanes" in data
        assert "daemons" in data
        assert isinstance(data["background_tasks_running"], list)
        assert isinstance(data["background_pending"], int)
        assert isinstance(data["goals_running"], list)
        assert isinstance(data["goals_pending"], int)
        assert isinstance(data["summary"], dict)
        assert isinstance(data["lanes"], list)
        assert isinstance(data["daemons"], list)


class TestGoalStatusEndpoint:
    def test_goal_status_includes_session_report(self, client):
        c, agent = client
        from liteagent.goals import GoalManager

        gm = GoalManager(agent.memory.db)
        agent._goal_manager = gm
        goal = gm.add_goal(
            title="Night coding status",
            objective="Collect a compact report",
            user_id="dashboard-user",
            goal_type="autonomous_coding",
            config={"workspace": "/tmp/workspace"},
            source="dashboard",
        )
        gid = int(goal["id"])
        gm.add_attempt(gid, outcome="done", summary="Implemented health guard", progress_delta=0.2)
        gm.add_attempt(gid, outcome="failed", summary="Regression in smoke test", error="smoke failed")

        resp = c.get(f"/api/goals/{gid}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["report"]["goal_type"] == "autonomous_coding"
        assert data["report"]["attempts_analyzed"] >= 2
        assert data["report"]["outcomes"]["done"] == 1

    def test_goal_report_markdown_download(self, client):
        c, agent = client
        from liteagent.goals import GoalManager

        gm = GoalManager(agent.memory.db)
        agent._goal_manager = gm
        goal = gm.add_goal(
            title="Night coding export",
            objective="Prepare a downloadable report",
            user_id="dashboard-user",
            goal_type="autonomous_coding",
            config={"workspace": "/tmp/workspace", "local_model": "qwen3-coder:30b"},
            source="dashboard",
        )
        gid = int(goal["id"])
        gm.add_attempt(gid, outcome="done", summary="Finished one verified patch", progress_delta=0.1)

        resp = c.get(f"/api/goals/{gid}/report?format=markdown")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        assert "attachment; filename=" in resp.headers["content-disposition"]
        body = resp.text
        assert "# Night coding export" in body
        assert "Finished one verified patch" in body

    def test_self_improvement_goal_status_includes_morning_report(self, client):
        c, agent = client
        from liteagent.goals import GoalManager

        gm = GoalManager(agent.memory.db)
        agent._goal_manager = gm
        goal = gm.add_goal(
            title="Self improvement status",
            objective="Summarize self-improvement findings",
            user_id="dashboard-user",
            goal_type="self_improvement",
            config={"workspace": "/Users/vskorokhod/liteagent"},
            source="dashboard",
        )
        gid = int(goal["id"])
        gm.add_attempt(
            gid,
            outcome="failed",
            summary="Memory recall brought back low-signal noise",
            error="noise in recall ranking",
            insight="Try a stronger anti-noise ranking penalty before the next cycle",
        )
        gm.add_attempt(
            gid,
            outcome="done",
            summary="Added a stronger memory relevance filter",
            progress_delta=0.15,
        )

        resp = c.get(f"/api/goals/{gid}/status")
        assert resp.status_code == 200
        data = resp.json()
        morning = data["report"]["morning_report"]
        assert any("noise" in item.lower() for item in morning["found_problems"])
        assert any("memory relevance filter" in item.lower() for item in morning["accepted_decisions"])
        assert any("ranking penalty" in item.lower() for item in morning["unvalidated_ideas"])

    def test_ops_active_normalizes_parallel_and_background_items(self, client):
        c, agent = client
        from liteagent.agent import LiteAgent

        prev_active = dict(LiteAgent._active_requests)
        prev_queued = dict(LiteAgent._queued_requests)
        class DummyBgDaemon:
            def get_active_tasks(self):
                return [{
                    "task_id": 41,
                    "name": "reindex memory",
                    "user_id": "dashboard-user",
                    "priority": 2,
                    "attempt": 2,
                    "max_attempts": 5,
                    "retry_delay_sec": 30,
                    "query_preview": "Reindex long-term memory graph",
                    "source": "agent",
                    "status": "running",
                    "phase_label": "Background daemon execution",
                    "started_at": "2026-03-05T10:00:03",
                }]

            def state(self):
                return {
                    "enabled": True,
                    "running": True,
                    "active_count": 1,
                    "processed_total": 3,
                    "failed_total": 1,
                }

        class DummyGoalDaemon:
            def get_active_goals(self):
                return [{
                    "goal_id": 9,
                    "title": "Ship memory monitor",
                    "user_id": "dashboard-user",
                    "priority": 3,
                    "progress": 0.42,
                    "current_phase": "executing",
                    "step_title": "Patch dashboard UI",
                    "plan_version": 2,
                    "stalled_cycles": 0,
                    "started_at": "2026-03-05T10:00:04",
                }]

            def state(self):
                return {
                    "enabled": True,
                    "running": True,
                    "active_count": 1,
                    "planned_total": 4,
                    "replanned_total": 1,
                }

        class DummyGoalManager:
            def count_pending_goals(self):
                return 2

        class DummyTaskManager:
            def count_background_pending(self):
                return 4

        class DummyScheduler:
            _jobs = [{
                "name": "night_worker",
                "_running": True,
                "_run_started": "2026-03-05T10:00:05",
                "max_runtime_sec": 600,
                "status": "running",
                "retry_on_fail": False,
            }]

        try:
            LiteAgent._active_requests = {
                1: {
                    "id": 1,
                    "user_id": "dashboard-user",
                    "started_at": "2026-03-05T10:00:00+00:00",
                    "updated_at": "2026-03-05T10:00:01+00:00",
                    "model": "qwen-plus",
                    "input_preview": "Inspect repo status",
                    "status": "running",
                    "phase": "parallel_tools",
                    "phase_label": "Parallel tools 1/2",
                    "progress_label": "Iteration 1/3 · tools 1/2",
                    "iteration": 1,
                    "max_iterations": 3,
                    "parallel_total": 2,
                    "parallel_completed": 1,
                    "parallel_children": [
                        {"tool_use_id": "t1", "name": "read_file", "status": "done", "duration_ms": 12, "error": False},
                        {"tool_use_id": "t2", "name": "rg", "status": "running", "duration_ms": 0, "error": False},
                    ],
                    "complexity_score": 2,
                    "cascade_tier": "medium",
                }
            }
            LiteAgent._queued_requests = {
                7: {"id": 7, "user_id": "dashboard-user", "queued_at": "2026-03-05T10:00:02+00:00"}
            }
            agent._background_task_daemon = DummyBgDaemon()
            agent._goal_coordinator = DummyGoalDaemon()
            agent._goal_manager = DummyGoalManager()
            agent._task_manager = DummyTaskManager()
            agent._scheduler = DummyScheduler()

            data = c.get("/api/ops/active").json()
            assert data["summary"]["active_total"] == 4
            assert data["summary"]["queued_total"] == 3
            assert data["summary"]["parallel_units"] >= 5
            foreground = next(l for l in data["lanes"] if l["id"] == "foreground")
            assert foreground["count"] == 1
            assert foreground["items"][0]["parallel_total"] == 2
            assert len(foreground["items"][0]["parallel_children"]) == 2
            autonomous = next(l for l in data["lanes"] if l["id"] == "autonomous")
            kinds = {item["kind"] for item in autonomous["items"]}
            assert {"background_task", "goal", "scheduler_job"} <= kinds
            queued = next(l for l in data["lanes"] if l["id"] == "queued")
            queued_kinds = {item["kind"] for item in queued["items"]}
            assert {"queued_request", "background_queue", "goal_queue"} <= queued_kinds
            assert any(d["id"] == "background" and d["running"] for d in data["daemons"])
        finally:
            LiteAgent._active_requests = prev_active
            LiteAgent._queued_requests = prev_queued


def _setup_goals_for_agent(agent):
    from liteagent.goals import GoalManager, setup_goal_coordinator_daemon

    gm = GoalManager(agent.memory.db)
    daemon = setup_goal_coordinator_daemon(
        agent,
        gm,
        {"scheduler": {"goals": {"enabled": True, "interval_sec": 60, "batch_size": 1}}},
    )
    agent._goal_manager = gm
    agent._goal_coordinator = daemon
    return gm, daemon


class TestCascadeEndpoint:
    def test_cascade_returns_structure(self, client):
        c, _ = client
        resp = c.get("/api/ops/cascade")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "models" in data
        assert "tier_costs" in data
        assert "summary" in data
        assert "history" in data
        assert "advisor" in data
        assert "recommendations" in data
        assert "is_local_only_now" in data

    def test_cascade_tier_costs_have_pricing(self, client):
        c, _ = client
        data = c.get("/api/ops/cascade").json()
        for tier in ("simple", "medium", "complex"):
            assert tier in data["tier_costs"]
            assert "model" in data["tier_costs"][tier]
            assert "input_per_mtok" in data["tier_costs"][tier]
            assert "output_per_mtok" in data["tier_costs"][tier]

    def test_cascade_summary_structure(self, client):
        c, _ = client
        data = c.get("/api/ops/cascade").json()
        summary = data["summary"]
        assert "tier_counts" in summary
        assert "source_counts" in summary
        assert "objective_counts" in summary
        assert "total_decisions" in summary
        assert set(summary["tier_counts"].keys()) == {"simple", "medium", "complex"}

    def test_cascade_records_history(self, client):
        from liteagent.agent import LiteAgent
        LiteAgent._cascade_history = []
        LiteAgent._record_cascade_decision("test-model", "simple", 0)
        c, _ = client
        data = c.get("/api/ops/cascade").json()
        assert len(data["history"]) >= 1
        assert data["history"][-1]["model"] == "test-model"

    def test_routing_settings_include_and_save_intelligent_routing(self, client):
        c, agent = client
        initial = c.get("/api/settings/routing")
        assert initial.status_code == 200
        assert "intelligent_routing" in initial.json()

        saved = c.post("/api/settings/routing", json={
            "intelligent_routing": {
                "enabled": True,
                "use_llm": False,
                "advisor_model": "gpt-4o-mini",
                "min_complexity": 2,
                "local_min_complexity": 3,
            }
        })
        assert saved.status_code == 200
        assert saved.json()["ok"] is True
        cfg = agent.config["cost"]["intelligent_routing"]
        assert cfg["enabled"] is True
        assert cfg["use_llm"] is False
        assert cfg["advisor_model"] == "gpt-4o-mini"
        assert cfg["min_complexity"] == 2
        assert cfg["local_min_complexity"] == 3


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
        data = resp.json()
        assert data["models"] == []
        assert data["today_calls"] == 0
        assert data["hour_calls"] == 0

    def test_daily_usage_empty(self, client):
        c, _ = client
        resp = c.get("/api/usage/daily?days=14")
        assert resp.status_code == 200
        assert resp.json() == []


class TestTasksEndpoints:
    def test_task_update_via_api(self, client):
        from liteagent.tasks import TaskManager
        c, agent = client
        tm = TaskManager(agent.memory.db)
        agent.enable_tasks(tm)
        created = c.post("/api/tasks", json={
            "name": "task one",
            "query": "do thing",
            "run_at": "2026-03-06T10:00:00",
        })
        assert created.status_code == 200
        task_id = created.json()["id"]

        updated = c.put(f"/api/tasks/{task_id}", json={
            "name": "task two",
            "query": "do another thing",
            "run_at": "2026-03-06T11:00:00",
            "priority": 2,
            "background": True,
            "retry_delay_sec": 30,
            "max_attempts": 5,
        })
        assert updated.status_code == 200
        body = updated.json()
        assert body["name"] == "task two"
        assert body["query"] == "do another thing"
        assert body["priority"] == 2
        assert body["background"] == 1
        assert body["retry_delay_sec"] == 30
        assert body["max_attempts"] == 5

    def test_task_update_running_conflict(self, client):
        from liteagent.tasks import TaskManager
        c, agent = client
        tm = TaskManager(agent.memory.db)
        agent.enable_tasks(tm)
        task = tm.add_task(
            "run", "q", "dashboard-user",
            task_type="one_shot", run_at="2026-03-06T10:00:00",
        )
        tm.mark_running(task["id"])

        resp = c.put(f"/api/tasks/{task['id']}", json={"name": "new", "query": "new q"})
        assert resp.status_code == 409


class TestGoalsEndpoints:
    def test_goals_create_status_pause_resume_cancel(self, client):
        c, agent = client
        _, daemon = _setup_goals_for_agent(agent)
        daemon.start = AsyncMock(return_value={"status": "started"})

        created = c.post("/api/goals", json={
            "title": "Ship memory fixes",
            "objective": "Close anti-pollution and queue cleanup issues",
            "priority": 2,
            "target_steps": 3,
        })
        assert created.status_code == 200
        goal = created.json()
        goal_id = int(goal["id"])
        assert goal["status"] == "active"

        status = c.get(f"/api/goals/{goal_id}/status")
        assert status.status_code == 200
        status_body = status.json()
        assert status_body["goal"]["id"] == goal_id
        assert isinstance(status_body["events"], list)
        assert status_body["events"]
        assert "plan" in status_body
        assert "recent_attempts" in status_body

        paused = c.post(f"/api/goals/{goal_id}/pause", json={"paused": True})
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"

        resumed = c.post(f"/api/goals/{goal_id}/pause", json={"paused": False})
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "active"

        cancelled = c.post(f"/api/goals/{goal_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"

    def test_goals_summary_and_list(self, client):
        c, agent = client
        gm, daemon = _setup_goals_for_agent(agent)
        daemon.start = AsyncMock(return_value={"status": "started"})
        gm.add_goal(
            title="Goal 1",
            objective="Do one thing",
            user_id="dashboard-user",
            priority=3,
            target_steps=2,
            source="dashboard",
        )

        listed = c.get("/api/goals")
        assert listed.status_code == 200
        goals = listed.json()
        assert isinstance(goals, list)
        assert goals

        summary = c.get("/api/goals/summary")
        assert summary.status_code == 200
        body = summary.json()
        assert "counts" in body
        assert "avg_progress" in body
        assert "goals" in body
        assert "coordinator" in body
        assert "running" in body
        assert "health" in body
        assert "lanes" in body
        assert set(body["lanes"].keys()) == {"running", "pipeline", "attention", "recent"}

    def test_create_autonomous_coding_goal(self, client):
        c, agent = client
        _, daemon = _setup_goals_for_agent(agent)
        daemon.start = AsyncMock(return_value={"status": "started"})

        created = c.post("/api/goals", json={
            "title": "Night Coding Session",
            "objective": "Finish the requested feature and keep improving reliability",
            "goal_type": "autonomous_coding",
            "config": {
                "workspace": "/tmp/night-workspace",
                "local_model": "qwen3-coder:30b",
                "continue_after_objective": True,
                "verification_commands": ["pytest -q"],
            },
        })
        assert created.status_code == 200
        body = created.json()
        assert body["goal_type"] == "autonomous_coding"
        assert body["config"]["workspace"] == "/tmp/night-workspace"
        assert body["config"]["local_model"] == "qwen3-coder:30b"
        assert body["config"]["continue_after_objective"] is True
        assert agent._goal_manager.get_goal(int(body["id"]))["goal_type"] == "autonomous_coding"

    def test_create_self_improvement_goal(self, client):
        c, agent = client
        _, daemon = _setup_goals_for_agent(agent)
        daemon.start = AsyncMock(return_value={"status": "started"})

        created = c.post("/api/goals", json={
            "title": "Night Self Improvement",
            "objective": "Improve LiteAgent using nightly local-model work",
            "goal_type": "self_improvement",
            "config": {
                "workspace": "/Users/vskorokhod/liteagent",
                "local_model": "qwen3-coder:30b",
                "continue_after_objective": True,
            },
        })
        assert created.status_code == 200
        body = created.json()
        assert body["goal_type"] == "self_improvement"
        assert body["config"]["workspace"] == "/Users/vskorokhod/liteagent"
        assert agent._goal_manager.get_goal(int(body["id"]))["goal_type"] == "self_improvement"

    def test_goals_summary_is_idle_when_worker_running_but_no_goals(self, client):
        c, agent = client
        _, daemon = _setup_goals_for_agent(agent)
        daemon.state = lambda: {
            "enabled": True,
            "running": True,
            "worker_id": "goal-test",
            "active_count": 0,
            "pending": 0,
            "last_pause_reason": "",
            "last_pause_at": 0.0,
            "last_cycle_at": "",
            "processed_total": 0,
            "failed_total": 0,
            "planned_total": 0,
            "replanned_total": 0,
        }

        summary = c.get("/api/goals/summary")
        assert summary.status_code == 200
        body = summary.json()
        assert body["coordinator"]["running"] is True
        assert body["health"]["state"] == "idle"
        assert body["counts"]["active"] == 0
        assert body["counts"]["running"] == 0

    def test_goal_plan_save_and_replan(self, client):
        c, agent = client
        gm, daemon = _setup_goals_for_agent(agent)
        daemon.start = AsyncMock(return_value={"status": "started"})
        goal = gm.add_goal(
            title="Plan editor goal",
            objective="Support manual plan editing from dashboard",
            user_id="dashboard-user",
            source="dashboard",
        )
        goal_id = int(goal["id"])

        saved = c.post(f"/api/goals/{goal_id}/plan", json={
            "strategy": "Ship in small verified slices.",
            "steps": [
                {"id": "s1", "title": "Inspect issue", "action": "Review failing flow"},
                {"id": "s2", "title": "Patch code", "action": "Implement fix"},
            ],
        })
        assert saved.status_code == 200
        saved_body = saved.json()
        assert saved_body["ok"] is True
        assert saved_body["plan"]["strategy"] == "Ship in small verified slices."
        assert len(saved_body["plan"]["steps"]) == 2

        async def _fake_plan(goal_obj, reason):
            return gm.upsert_plan(
                int(goal_obj["id"]),
                strategy="Alternative route after review.",
                steps=[{"id": "b1", "title": "Try fallback", "action": "Use fallback flow"}],
                trigger=reason,
            )

        daemon._plan_goal = AsyncMock(side_effect=_fake_plan)
        replanned = c.post(f"/api/goals/{goal_id}/replan")
        assert replanned.status_code == 200
        replan_body = replanned.json()
        assert replan_body["ok"] is True
        assert "Alternative route" in replan_body["plan"]["strategy"]

        status = c.get(f"/api/goals/{goal_id}/status")
        assert status.status_code == 200
        status_body = status.json()
        assert status_body["plan"]["version"] >= 2


class TestMemoryExchangeEndpoints:
    def test_memory_settings_get_defaults(self, client):
        c, _ = client
        resp = c.get("/api/settings/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert "memory_exchange_enabled" in data
        assert "shadow_twin_enabled" in data
        assert "extraction_max_concurrency" in data
        assert "memory_exchange_context_budget_tokens" in data
        assert "memory_local_worker_enabled" in data
        assert "memory_local_worker_interval_sec" in data
        assert "memory_local_worker_batch_size" in data

    def test_memory_settings_save(self, client):
        c, agent = client
        resp = c.post("/api/settings/memory", json={
            "memory_exchange_enabled": True,
            "memory_exchange_top_k": 12,
            "memory_exchange_pack_budget_tokens": 600,
            "memory_exchange_context_budget_tokens": 1200,
            "memory_exchange_max_packs": 3,
            "shadow_twin_enabled": True,
            "shadow_twin_predictions": 4,
            "shadow_twin_use_llm": False,
            "memory_local_worker_enabled": True,
            "memory_local_worker_interval_sec": 15,
            "memory_local_worker_batch_size": 30,
            "extraction_provider": "",
            "extraction_model": "test-memory-model",
            "extraction_max_concurrency": 4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["settings"]["memory_exchange_top_k"] == 12
        assert data["settings"]["shadow_twin_predictions"] == 4
        assert data["settings"]["memory_local_worker_interval_sec"] == 15
        assert agent.config["memory"]["extraction_model"] == "test-memory-model"
        assert agent.config["memory"]["memory_exchange_context_budget_tokens"] == 1200
        assert agent.memory._extraction_semaphore._value == 4

    def test_memory_exchange_metrics_empty(self, client):
        c, _ = client
        resp = c.get("/api/memory/exchange")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope_user"] == "all"
        assert data["counts"]["intents_total"] == 0
        assert data["counts"]["packs_total"] == 0
        assert data["counts"]["predictions_total"] == 0
        assert "prediction_hit_rate" in data["quality"]
        assert "daemon" in data
        assert "quality_metrics" in data
        assert "identity" in data
        assert "explainability" in data
        assert data["queue"]["pending_total"] == 0
        assert "shadow_cleanup_removed" in data["queue"]
        assert "shadow_cleanup_at" in data["queue"]
        assert data["tokens"]["pack_tokens_cached"] == 0
        assert data["tokens"]["pack_tokens_saved_est"] == 0

    @pytest.mark.asyncio
    async def test_memory_exchange_metrics_with_activity(self, client):
        c, agent = client
        user_id = "dashboard-user"
        await agent.memory.remember(
            "User coffee preference: black coffee without sugar",
            user_id,
            "fact",
            0.8,
        )
        agent.memory.add_message(user_id, "user", "coffee preference no sugar")
        result = await agent.memory.run_memory_exchange_cycle(
            "coffee preference no sugar",
            user_id,
            "saved your coffee preference",
        )
        assert result["status"] == "ok"

        resp = c.get(f"/api/memory/exchange?user_id={user_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scope_user"] == user_id
        assert data["counts"]["intents_total"] >= 1
        assert data["counts"]["packs_total"] >= 1
        assert data["counts"]["predictions_total"] >= 1
        assert isinstance(data["recent"]["top_intents"], list)
        assert data["queue"]["pending_total"] >= 1
        assert "pack_tokens_served" in data["tokens"]
        assert "recall_at_k" in data["quality_metrics"]

    def test_memory_identity_endpoint(self, client):
        c, _ = client
        resp = c.post("/api/memory/identity", json={
            "alias_user_id": "dashboard-user",
            "person_id": "tg-456",
            "source": "test",
        })
        assert resp.status_code == 200
        body = c.get("/api/memory/identity?user_id=dashboard-user").json()
        assert body["person_id"] == "tg-456"

    def test_memory_explain_endpoint(self, client):
        c, agent = client
        agent.memory._store_recall_trace("dashboard-user", "как меня зовут", "type_aware", [
            {"id": -1, "type": "profile_slot", "score": 1.1, "content": "User name is Влад."}
        ], {"slot": "name"}, "Влад")
        resp = c.get("/api/memory/explain?user_id=dashboard-user&limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert "traces" in data
        assert isinstance(data["traces"], list)


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

    def test_tools_have_source_and_params(self, client):
        c, _ = client
        tools = c.get("/api/tools").json()
        for t in tools:
            assert "source" in t
            assert t["source"] in ("builtin", "mcp", "custom", "onboarding")
            assert "parameters" in t

    def test_add_custom_tool_missing_name(self, client):
        c, _ = client
        resp = c.post("/api/tools/custom", json={"name": "", "code": "def x(): pass"})
        assert resp.status_code == 400

    def test_add_custom_tool_missing_code(self, client):
        c, _ = client
        resp = c.post("/api/tools/custom", json={"name": "test_tool", "code": ""})
        assert resp.status_code == 400

    def test_add_custom_tool_blocked_pattern(self, client):
        c, _ = client
        resp = c.post("/api/tools/custom", json={
            "name": "bad_tool",
            "code": "import os\ndef bad_tool(): return os.listdir('/')"
        })
        assert resp.status_code == 400
        # Either AST validator or restricted builtins will block dangerous code
        detail = resp.json()["detail"].lower()
        assert any(kw in detail for kw in ["validation failed", "not allowed", "invalid code", "blocked"])

    def test_add_custom_tool_invalid_name_rejected(self, client):
        c, _ = client
        resp = c.post("/api/tools/custom", json={
            "name": "../../escape_probe",
            "code": "def safe_tool():\n    return 'ok'",
        })
        assert resp.status_code == 400
        assert "Invalid tool name" in resp.json()["detail"]

    def test_add_custom_tool_vars_bypass_rejected(self, client):
        c, _ = client
        resp = c.post("/api/tools/custom", json={
            "name": "sneaky_tool",
            "code": (
                "def helper():\n"
                "    return 'ok'\n"
                "vars()['sneaky_tool'] = helper\n"
            ),
        })
        assert resp.status_code == 400
        assert "blocked" in resp.json()["detail"].lower()

    def test_add_and_delete_custom_tool(self, client, monkeypatch, tmp_path):
        from liteagent.channels import dashboard
        monkeypatch.setattr(dashboard, "CUSTOM_TOOLS_DIR", tmp_path / "custom_tools")
        c, _ = client
        resp = c.post("/api/tools/custom", json={
            "name": "greet",
            "description": "Greet someone",
            "code": "def greet(name: str) -> str:\n    return f'Hello, {name}!'"
        })
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        # Tool should appear in list
        tools = c.get("/api/tools").json()
        names = [t["name"] for t in tools]
        assert "greet" in names
        # Delete it
        resp = c.delete("/api/tools/custom/greet")
        assert resp.status_code == 200
        tools = c.get("/api/tools").json()
        assert "greet" not in [t["name"] for t in tools]

    def test_delete_builtin_tool_fails(self, client):
        c, _ = client
        resp = c.delete("/api/tools/custom/memory_search")
        assert resp.status_code == 400

    def test_path_traversal_not_written(self, client, monkeypatch, tmp_path):
        from liteagent.channels import dashboard
        custom_dir = tmp_path / "custom_tools"
        monkeypatch.setattr(dashboard, "CUSTOM_TOOLS_DIR", custom_dir)
        escape_path = (custom_dir / "../../escape_probe.py").resolve()
        if escape_path.exists():
            escape_path.unlink()
        c, _ = client
        resp = c.post("/api/tools/custom", json={
            "name": "../../escape_probe",
            "code": "def escape_probe():\n    return 'ok'",
        })
        assert resp.status_code == 400
        assert not escape_path.exists()


class TestChatHistoryPersistence:
    def test_history_empty_initially(self, client):
        c, _ = client
        resp = c.get("/api/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_history_persists_messages(self, client):
        c, agent = client
        agent.memory.add_message("dashboard-user", "user", "Hello")
        agent.memory.add_message("dashboard-user", "assistant", "Hi there!")
        resp = c.get("/api/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["role"] == "user"
        assert data[0]["content"] == "Hello"
        assert data[1]["role"] == "assistant"
        assert "created_at" in data[0]

    def test_history_uses_resolved_user_alias(self, client):
        c, agent = client
        agent.memory.add_message("tg-456", "user", "Hello from Telegram")
        resp = c.get("/api/history")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "Hello from Telegram"

    def test_history_survives_ram_clear(self, client):
        c, agent = client
        agent.memory.add_message("dashboard-user", "user", "Persistent msg")
        agent.memory.clear_conversation("dashboard-user")
        # RAM is empty, but SQLite still has it
        resp = c.get("/api/history")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["content"] == "Persistent msg"

    def test_history_preserves_assistant_meta_payload(self, client):
        c, agent = client
        agent.memory.add_message("dashboard-user", "assistant", {
            "text": "Ответ с explainability",
            "meta": {
                "media_explainability": [
                    {"label": "Image", "index": 1, "summary": "Login screen", "model": "gpt-4o", "provider": "openai"}
                ]
            },
        })
        resp = c.get("/api/history")
        data = resp.json()
        assert len(data) == 1
        assert isinstance(data[0]["content"], dict)
        assert data[0]["content"]["text"] == "Ответ с explainability"
        assert data[0]["content"]["meta"]["media_explainability"][0]["summary"] == "Login screen"

    def test_clear_history(self, client):
        c, agent = client
        agent.memory.add_message("dashboard-user", "user", "To be deleted")
        resp = c.delete("/api/history")
        assert resp.status_code == 200
        assert c.get("/api/history").json() == []

    def test_load_history_restores_ram(self, client):
        c, agent = client
        agent.memory.add_message("dashboard-user", "user", "Msg 1")
        agent.memory.add_message("dashboard-user", "assistant", "Msg 2")
        agent.memory.clear_conversation("dashboard-user")
        assert agent.memory.get_history("dashboard-user") == []
        loaded = agent.memory.load_history("dashboard-user")
        assert len(loaded) == 2


class TestMultimodalChatExplainability:
    def test_chat_forwards_requested_model(self, client, monkeypatch):
        c, agent = client
        captured = {}

        async def fake_run(user_input, user_id="default", requested_model=None):
            captured["user_input"] = user_input
            captured["user_id"] = user_id
            captured["requested_model"] = requested_model
            agent._last_response_meta = {
                "response_route": {
                    "provider": "openai",
                    "model": requested_model or "gpt-4o-mini",
                    "requested_model": requested_model,
                    "mode": "manual",
                }
            }
            return "ok"

        monkeypatch.setattr(agent, "run", fake_run)

        resp = c.post("/chat", json={
            "message": "hello",
            "user_id": "dashboard-user",
            "model": "gpt-4o",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "ok"
        assert data["meta"]["response_route"]["model"] == "gpt-4o"
        assert captured["requested_model"] == "gpt-4o"

    def test_chat_stream_done_event_uses_actual_route(self, client, monkeypatch):
        c, agent = client

        async def fake_stream(user_input, user_id="default", requested_model=None):
            agent._last_response_meta = {
                "response_route": {
                    "provider": "qwen",
                    "model": "qwen-plus",
                    "requested_model": requested_model,
                    "mode": "manual",
                }
            }
            yield "partial"

        monkeypatch.setattr(agent, "stream", fake_stream)

        resp = c.get("/chat/stream", params={
            "message": "hello",
            "user_id": "dashboard-user",
            "model": "qwen-plus",
        })
        assert resp.status_code == 200
        body = resp.text
        assert '"text": "partial"' in body
        assert '"provider": "qwen"' in body
        assert '"model": "qwen-plus"' in body
        assert '"requested_model": "qwen-plus"' in body

    def test_chat_multimodal_returns_meta(self, client, monkeypatch):
        c, agent = client

        async def fake_run(user_input, user_id="default", requested_model=None):
            agent._last_response_meta = {
                "media_explainability": [
                    {"label": "Screenshot", "index": 1, "summary": "Dashboard with warning", "model": "qwen-vl-plus", "provider": "qwen"}
                ],
                "response_route": {
                    "provider": "qwen",
                    "model": requested_model or "qwen-vl-plus",
                    "requested_model": requested_model,
                    "mode": "manual",
                },
            }
            return "Готово"

        monkeypatch.setattr(agent, "run", fake_run)

        resp = c.post(
            "/chat/multimodal",
            data={"message": "Что на скриншоте?", "user_id": "dashboard-user", "model": "qwen-vl-plus"},
            files={"files": ("shot.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Готово"
        assert data["meta"]["media_explainability"][0]["label"] == "Screenshot"
        assert data["meta"]["response_route"]["requested_model"] == "qwen-vl-plus"

    def test_chat_multimodal_enriches_ingested_file_description(self, client, monkeypatch):
        c, agent = client
        updates = []

        async def fake_run(user_input, user_id="default", requested_model=None):
            agent._last_response_meta = {
                "media_explainability": [
                    {
                        "label": "Passport",
                        "index": 1,
                        "summary": "Ukrainian passport for Inga Smelova, number FP139628",
                        "model": "qwen-vl-plus",
                        "provider": "qwen",
                    }
                ]
            }
            return "Готово"

        async def fake_ingest_file(data, filename, **kwargs):
            return {"storage_key": "files/api/passport_photo.jpg"}

        agent._file_manager = type(
            "FM",
            (),
            {
                "update_description": lambda self, storage_key, description, user_id=None: updates.append(
                    (storage_key, description, user_id)
                ) or True
            },
        )()

        monkeypatch.setattr(agent, "run", fake_run)
        monkeypatch.setattr(agent, "ingest_file", fake_ingest_file)

        resp = c.post(
            "/chat/multimodal",
            data={"message": "Что на фото?", "user_id": "dashboard-user"},
            files={"files": ("passport.jpg", b"\xff\xd8\xff" + b"0" * 32, "image/jpeg")},
        )

        assert resp.status_code == 200
        assert updates == [
            (
                "files/api/passport_photo.jpg",
                "Ukrainian passport for Inga Smelova, number FP139628",
                "dashboard-user",
            )
        ]


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

    def test_export_thinking_cloud_json(self, client):
        c, agent = client
        agent.memory.upsert_thinking_note(
            "dashboard-user",
            "constraint",
            "Prefer local-first tools and exports.",
            title="Local-first preference",
            themes=["Local-first"],
            strategic_importance=0.82,
        )
        resp = c.get("/api/export/thinking?format=json&user_id=dashboard-user")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cloud"]["overview"]["total_notes"] >= 1

    def test_export_thinking_cloud_obsidian_zip(self, client):
        c, agent = client
        agent.memory.upsert_thinking_note(
            "dashboard-user",
            "direction",
            "Build Obsidian-compatible exports for strategic notes.",
            title="Obsidian export",
            themes=["Knowledge graph", "Obsidian"],
            strategic_importance=0.9,
        )
        resp = c.get("/api/export/thinking?format=obsidian&user_id=dashboard-user")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        archive = zipfile.ZipFile(BytesIO(resp.content))
        names = archive.namelist()
        assert any(name.endswith("/Thinking Cloud.md") for name in names)
        assert any(name.endswith("/Thinking Cloud.canvas") for name in names)
        note_name = next(name for name in names if "/Directions/" in name and name.endswith(".md"))
        note_body = archive.read(note_name).decode("utf-8")
        assert "liteagent-thinking-cloud" in note_body
        assert "[[Knowledge graph]]" in note_body


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

    def test_ollama_status(self, client):
        c, _ = client
        data = c.get("/api/settings/providers").json()
        ollama = data["providers"]["ollama"]
        # Ollama status depends on whether it's running locally
        assert "has_key" in ollama
        assert ollama["key_preview"] in ("(running)", "(not running)")
        assert isinstance(ollama["models"], list)

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
        # Prevent save_config from overwriting real config.json
        monkeypatch.setattr("liteagent.config.save_config", lambda *a, **kw: None)
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

    def test_test_provider_no_key(self, client, monkeypatch):
        # Ensure no real OPENAI_API_KEY leaks in from the test environment
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        c, _ = client
        resp = c.post("/api/settings/provider/test",
                       json={"provider": "openai", "api_key": ""})
        data = resp.json()
        # Should return ok=false (no key or no SDK)
        assert data["ok"] is False


class TestVoiceSettings:
    def test_voice_settings_expose_groq_russian_direct_mode(self, client):
        c, _ = client
        resp = c.get("/api/settings/voice")
        assert resp.status_code == 200
        data = resp.json()
        groq = data["tts"]["providers_meta"]["groq"]
        assert "ru" in groq["languages"]
        assert "ru" in groq["experimental_languages"]
        assert "ru" in groq["model_info"]["playai-tts"]["supported_languages"]
        assert "ru" in groq["model_info"]["playai-tts"]["experimental_languages"]


class TestKeyManagement:
    """Test config.py key management functions."""

    def test_save_and_load_key(self, tmp_path, monkeypatch):
        from liteagent.config import (
            load_provider_keys, save_provider_key,
            delete_provider_key, get_api_key, key_preview,
            KEYS_DIR,
        )
        import liteagent.config as config_mod

        # Override paths to use tmp_path (including backup to prevent cross-test contamination)
        monkeypatch.setattr(config_mod, "KEYS_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "KEYS_PATH", tmp_path / "keys.json")
        monkeypatch.setattr(config_mod, "KEYS_BACKUP_PATH", tmp_path / "keys.json.bak")
        # Prevent real env key from being found after we delete from keys.json
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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


class TestKnowledgeBaseEndpoints:
    """Tests for Knowledge Base dashboard endpoints."""

    def test_kb_settings_get_disabled(self, client):
        """GET /api/settings/knowledge_base returns disabled state."""
        c, _ = client
        resp = c.get("/api/settings/knowledge_base")
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "search_mode" in data
        assert "chunk_size" in data

    def test_kb_settings_save(self, client):
        """POST /api/settings/knowledge_base saves config."""
        c, _ = client
        resp = c.post("/api/settings/knowledge_base",
            json={"enabled": True, "search_mode": "hybrid", "chunk_size": 1000})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("ok") is True

    def test_kb_settings_save_accepts_legacy_max_file_size_key(self, client):
        """POST /api/settings/knowledge_base accepts max_file_size alias."""
        c, agent = client
        resp = c.post("/api/settings/knowledge_base", json={"enabled": True, "max_file_size": 33})
        assert resp.status_code == 200
        assert resp.json().get("ok") is True
        assert agent.config["knowledge_base"]["max_file_size_mb"] == 33

    def test_kb_documents_empty(self, client):
        """GET /api/knowledge_base/documents returns empty list when KB not enabled."""
        c, _ = client
        resp = c.get("/api/knowledge_base/documents")
        assert resp.status_code in (200, 400)  # 400 if KB not enabled

    def test_kb_documents_has_chunks_aliases(self, client):
        """GET /api/knowledge_base/documents exposes pages/chunks aliases."""
        c, agent = client

        class DummyKB:
            async def list_documents(self):
                return [{
                    "id": "doc-1",
                    "name": "sample.pdf",
                    "page_count": 12,
                    "chunk_count": 48,
                }]

            async def get_stats(self):
                return {"documents": 1, "chunks": 48}

        agent._knowledge_base = DummyKB()
        resp = c.get("/api/knowledge_base/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"][0]["chunk_count"] == 48
        assert data["documents"][0]["chunks"] == 48
        assert data["documents"][0]["page_count"] == 12
        assert data["documents"][0]["pages"] == 12

    def test_kb_document_by_id(self, client):
        """GET /api/knowledge_base/documents/{id} returns a single document."""
        c, agent = client

        class DummyKB:
            async def get_document(self, doc_id):
                if doc_id != "doc-1":
                    return None
                return {
                    "id": "doc-1",
                    "name": "sample.pdf",
                    "page_count": 12,
                    "chunk_count": 48,
                }

        agent._knowledge_base = DummyKB()
        resp = c.get("/api/knowledge_base/documents/doc-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "doc-1"
        assert data["chunks"] == 48
        assert data["pages"] == 12

    def test_kb_document_missing(self, client):
        """GET /api/knowledge_base/documents/{id} returns 404 when absent."""
        c, agent = client

        class DummyKB:
            async def get_document(self, doc_id):
                return None

        agent._knowledge_base = DummyKB()
        resp = c.get("/api/knowledge_base/documents/missing")
        assert resp.status_code == 404

    def test_kb_search_not_enabled(self, client):
        """POST /api/knowledge_base/search returns error when KB not enabled."""
        c, _ = client
        resp = c.post("/api/knowledge_base/search", json={"query": "test"})
        assert resp.status_code == 400

    def test_kb_query_log_empty(self, client):
        """GET /api/knowledge_base/query_log returns empty when KB not enabled."""
        c, _ = client
        resp = c.get("/api/knowledge_base/query_log")
        assert resp.status_code in (200, 400)

    def test_kb_upload_not_enabled(self, client):
        """POST /api/knowledge_base/upload still analyzes documents when KB is disabled."""
        c, _ = client
        resp = c.post(
            "/api/knowledge_base/upload",
            files={"file": ("doc.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["knowledge_base"]["enabled"] is False

    def test_kb_upload_success(self, client, tmp_path, monkeypatch):
        """POST /api/knowledge_base/upload stores file and calls kb.ingest()."""
        c, agent = client
        monkeypatch.setenv("HOME", str(tmp_path))
        agent.config.setdefault("knowledge_base", {})["max_file_size_mb"] = 5
        agent.provider = None

        class DummyKB:
            def __init__(self):
                self.path = None
                self.metadata = None

            async def ingest(self, path, metadata=None):
                self.path = path
                self.metadata = metadata or {}
                return {"status": "ok", "name": self.metadata.get("name"), "chunks": 1}

        kb = DummyKB()
        agent._knowledge_base = kb

        resp = c.post(
            "/api/knowledge_base/upload",
            files={"file": ("report.md", b"# Title\nbody", "text/markdown")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["name"] == "report.md"
        assert data["analysis"]["summary"]
        assert data["knowledge_base"]["status"] == "ok"

        saved = Path(kb.path)
        assert saved.exists()
        assert saved.parent == tmp_path / ".liteagent" / "document_uploads"
        assert kb.metadata["uploaded_via"] == "dashboard"

    def test_kb_upload_rejects_oversized_file(self, client, tmp_path, monkeypatch):
        """POST /api/knowledge_base/upload enforces KB max_file_size_mb."""
        c, agent = client
        monkeypatch.setenv("HOME", str(tmp_path))
        agent.config.setdefault("knowledge_base", {})["max_file_size_mb"] = 1

        class DummyKB:
            def __init__(self):
                self.called = False

            async def ingest(self, path, metadata=None):
                self.called = True
                return {"status": "ok"}

        kb = DummyKB()
        agent._knowledge_base = kb

        resp = c.post(
            "/api/knowledge_base/upload",
            files={"file": ("big.txt", b"x" * (1024 * 1024 + 1), "text/plain")},
        )
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()
        assert kb.called is False

    def test_document_reviews_endpoint_returns_recent_reviews(self, client, tmp_path, monkeypatch):
        c, agent = client
        monkeypatch.setenv("HOME", str(tmp_path))
        agent.provider = None

        resp = c.post(
            "/api/documents/upload",
            files={"file": ("contract.txt", b"Contract renewal deadline 2026-12-31", "text/plain")},
        )
        assert resp.status_code == 200
        review_id = resp.json()["review_id"]

        recent = c.get("/api/documents/reviews")
        assert recent.status_code == 200
        reviews = recent.json()["reviews"]
        assert reviews
        assert any(item["review_id"] == review_id for item in reviews)

        detail = c.get(f"/api/documents/reviews/{review_id}")
        assert detail.status_code == 200
        assert detail.json()["review_id"] == review_id


class TestFileManagerEndpoints:
    """Tests for File Manager dashboard endpoints."""

    def test_files_stats_no_storage(self, client):
        """GET /api/files/stats returns zeros when storage not enabled."""
        c, _ = client
        resp = c.get("/api/files/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_files"] == 0
        assert data["total_size_bytes"] == 0
        assert data["sources"] == {}

    def test_files_list_no_storage(self, client):
        """GET /api/files returns 400 when file manager not enabled."""
        c, _ = client
        resp = c.get("/api/files")
        assert resp.status_code == 400

    def test_files_search_no_storage(self, client):
        """GET /api/files/search returns 400 when file manager not enabled."""
        c, _ = client
        resp = c.get("/api/files/search?q=test")
        assert resp.status_code == 400

    def test_files_count_no_storage(self, client):
        """GET /api/files/count returns 0 when file manager not enabled."""
        c, _ = client
        resp = c.get("/api/files/count")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_files_delete_no_storage(self, client):
        """DELETE /api/files/{key} returns 400 when file manager not enabled."""
        c, _ = client
        resp = c.delete("/api/files/test/key.txt")
        assert resp.status_code == 400
