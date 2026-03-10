"""Tests for long-running goals manager and coordinator daemon."""

from unittest.mock import AsyncMock

import pytest

from liteagent.agent import LiteAgent
from liteagent.goals import GoalCoordinatorDaemon, GoalManager


@pytest.fixture
def goals_agent(tmp_path):
    config = {
        "agent": {"max_iterations": 2},
        "cost": {"budget_daily_usd": 100.0},
        "memory": {"db_path": str(tmp_path / "test_goals.db"), "auto_learn": False},
        "tools": {"builtin": []},
    }
    agent = LiteAgent(config)
    yield agent
    agent.memory.close()


class TestGoalManager:
    def test_goal_lifecycle(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        goal = gm.add_goal(
            title="Improve memory quality",
            objective="Reduce memory pollution and improve retrieval relevance",
            user_id="u1",
            priority=2,
            target_steps=3,
            cooldown_sec=30,
            source="dashboard",
        )
        goal_id = int(goal["id"])
        assert goal["status"] == "active"

        assert gm.claim_running(goal_id) is True
        running = gm.get_goal(goal_id)
        assert running["status"] == "running"

        updated = gm.mark_cycle_result(
            goal_id,
            progress_delta=0.3,
            completed=False,
            phase="indexing",
            summary="Indexed one batch",
            next_action="Run next batch",
        )
        assert updated is not None
        assert updated["status"] == "active"
        assert float(updated["progress"]) > 0.0

        paused = gm.pause_goal(goal_id)
        assert paused is not None
        assert paused["status"] == "paused"

        resumed = gm.resume_goal(goal_id)
        assert resumed is not None
        assert resumed["status"] == "active"

        cancelled = gm.cancel_goal(goal_id)
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"

        events = gm.get_goal_events(goal_id, limit=20)
        event_types = {e["event_type"] for e in events}
        assert "created" in event_types
        assert "paused" in event_types
        assert "resumed" in event_types
        assert "cancelled" in event_types

    def test_goal_plan_and_attempt_journal(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        goal = gm.add_goal(
            title="Ship proactive memory",
            objective="Build layered memory with robust retrieval",
            user_id="u1",
            source="dashboard",
        )
        gid = int(goal["id"])

        plan = gm.upsert_plan(
            gid,
            strategy="Start with canonical profile slots then enrich with graph links.",
            steps=[
                {"id": "s1", "title": "Stabilize retrieval", "action": "Implement type-aware ranking"},
                {"id": "s2", "title": "Add telemetry", "action": "Expose recall metrics in dashboard"},
            ],
            trigger="initial",
        )
        assert plan is not None
        assert int(plan["version"]) == 1
        step = gm.get_next_plan_step(gid)
        assert step is not None
        assert step["id"] == "s1"

        updated = gm.update_plan_step(
            gid,
            plan_version=int(plan["version"]),
            step_id="s1",
            status="done",
            note="implemented",
        )
        assert updated is not None
        assert updated["status"] == "done"

        attempt = gm.add_attempt(
            gid,
            plan_version=int(plan["version"]),
            step_id="s1",
            step_title="Stabilize retrieval",
            action_query="Implement type-aware ranking",
            outcome="done",
            progress_delta=0.2,
            summary="Added profile-slot-first retrieval",
            insight="Name queries now deterministic",
        )
        assert attempt["outcome"] == "done"

        attempts = gm.get_recent_attempts(gid, limit=5)
        assert attempts
        assert attempts[0]["step_id"] == "s1"

        plan2 = gm.upsert_plan(
            gid,
            strategy="Pivot to fallback strategy due blockers.",
            steps=[{"id": "n1", "title": "Alternative", "action": "Try different approach"}],
            trigger="blocked",
        )
        assert plan2 is not None
        assert int(plan2["version"]) == 2
        history = gm.get_plan_history(gid, limit=5)
        assert len(history) >= 2
        assert history[0]["version"] == 2

    def test_autonomous_coding_goal_persists_type_and_config(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        goal = gm.add_goal(
            title="Night coding",
            objective="Improve the backend overnight",
            user_id="u1",
            goal_type="autonomous_coding",
            config={
                "workspace": "/tmp/workspace",
                "local_model": "qwen3-coder:30b",
                "continue_after_objective": True,
            },
            source="dashboard",
        )
        fetched = gm.get_goal(int(goal["id"]))
        assert fetched is not None
        assert fetched["goal_type"] == "autonomous_coding"
        assert fetched["config"]["workspace"] == "/tmp/workspace"
        assert fetched["config"]["local_model"] == "qwen3-coder:30b"

    def test_recover_orphaned_running_goals_requeues_stale_claims(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        goal = gm.add_goal(
            title="Recover me",
            objective="Resume after restart",
            user_id="u1",
            source="dashboard",
        )
        gid = int(goal["id"])
        assert gm.claim_running(gid) is True

        recovered = gm.recover_orphaned_running_goals(reason="test_restart")
        assert recovered == 1

        updated = gm.get_goal(gid)
        assert updated is not None
        assert updated["status"] == "active"
        events = gm.get_goal_events(gid, limit=5)
        assert any(e["event_type"] == "recovered" for e in events)

    def test_build_goal_report_summarizes_attempts(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        goal = gm.add_goal(
            title="Night coding report",
            objective="Summarize the session",
            user_id="u1",
            goal_type="autonomous_coding",
            config={"workspace": "/tmp/workspace"},
            source="dashboard",
        )
        gid = int(goal["id"])
        gm.add_attempt(gid, outcome="done", summary="Patched failing API route", progress_delta=0.2)
        gm.add_attempt(gid, outcome="blocked", summary="Frontend smoke failed", insight="Missing entrypoint")
        gm.mark_cycle_result(
            gid,
            progress_delta=0.2,
            completed=False,
            phase="working",
            summary="Patched failing API route",
            next_action="Repair frontend bootstrap",
        )
        report = gm.build_goal_report(gid, attempt_limit=10)
        assert report["goal_type"] == "autonomous_coding"
        assert report["outcomes"]["done"] == 1
        assert report["outcomes"]["blocked"] == 1
        assert any("Patched failing API route" in item for item in report["highlights"])
        assert any("Missing entrypoint" in item for item in report["blockers"])

    def test_render_goal_report_markdown_contains_summary_sections(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        goal = gm.add_goal(
            title="Night coding export",
            objective="Leave a clear morning report",
            user_id="u1",
            goal_type="autonomous_coding",
            config={"workspace": "/tmp/workspace", "local_model": "qwen3-coder:30b"},
            source="dashboard",
        )
        gid = int(goal["id"])
        gm.upsert_plan(
            gid,
            strategy="Fix regressions, then verify.",
            steps=[{"id": "s1", "title": "Fix tests", "action": "Run pytest -q", "success_criteria": "tests pass"}],
            trigger="initial",
        )
        gm.add_attempt(gid, outcome="done", step_title="Fix tests", summary="Resolved two failing tests", progress_delta=0.2)
        md = gm.render_goal_report_markdown(gid, attempt_limit=10)
        assert "# Night coding export" in md
        assert "## Session Summary" in md
        assert "## Active Plan" in md
        assert "## Recent Attempts" in md
        assert "Resolved two failing tests" in md

    def test_self_improvement_report_contains_morning_brief_sections(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        goal = gm.add_goal(
            title="Self-improvement export",
            objective="Leave a focused morning brief",
            user_id="u1",
            goal_type="self_improvement",
            config={"workspace": "/Users/vskorokhod/liteagent", "local_model": "qwen3-coder:30b"},
            source="dashboard",
        )
        gid = int(goal["id"])
        gm.add_attempt(
            gid,
            outcome="failed",
            step_title="Stabilize tool loop",
            summary="Tool loop stalled on repeated no-tool answers",
            error="no-tool loop remained unresolved",
            insight="Add a stricter recovery path after repeated no-tool responses",
        )
        gm.add_attempt(
            gid,
            outcome="done",
            step_title="Harden runtime guard",
            summary="Added a runtime guard for repeated no-tool responses",
            progress_delta=0.2,
        )
        report = gm.build_goal_report(gid, attempt_limit=10)
        morning = report.get("morning_report") or {}
        assert report["goal_type"] == "self_improvement"
        assert any("Tool loop stalled" in item or "no-tool loop" in item for item in morning.get("found_problems", []))
        assert any("runtime guard" in item.lower() for item in morning.get("accepted_decisions", []))
        assert any("recovery path" in item.lower() for item in morning.get("unvalidated_ideas", []))

        md = gm.render_goal_report_markdown(gid, attempt_limit=10)
        assert "## Self-Improvement Morning Report" in md
        assert "### Found Problems" in md
        assert "### Accepted Decisions" in md
        assert "### Unvalidated Ideas" in md


class TestGoalCoordinator:
    @pytest.mark.asyncio
    async def test_process_once_pauses_under_high_load(self, goals_agent, monkeypatch):
        gm = GoalManager(goals_agent.memory.db)
        daemon = GoalCoordinatorDaemon(goals_agent, gm, {
            "enabled": True,
            "auto_pause": True,
            "pause_active_requests": 1,
            "pause_queued_requests": 100,
        })

        monkeypatch.setattr(
            LiteAgent,
            "get_active_requests",
            classmethod(lambda cls: [{"user_id": "u-load"}]),
        )
        monkeypatch.setattr(
            LiteAgent,
            "get_queued_requests",
            classmethod(lambda cls: []),
        )

        gm.add_goal(
            title="Goal under load",
            objective="Should pause before execution",
            user_id="u1",
            source="dashboard",
        )
        res = await daemon.process_once()
        assert res["status"] == "paused"
        assert "active_requests=" in res["reason"]

    @pytest.mark.asyncio
    async def test_process_once_advances_goal(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        daemon = GoalCoordinatorDaemon(goals_agent, gm, {
            "enabled": True,
            "auto_pause": False,
            "batch_size": 1,
        })
        goals_agent.run = AsyncMock(
            return_value='{"progress_delta":0.22,"completed":false,"phase":"working","summary":"Cycle done","next_action":"Continue"}'
        )

        goal = gm.add_goal(
            title="Advance one step",
            objective="Complete at least one cycle",
            user_id="u1",
            cooldown_sec=60,
            source="dashboard",
        )
        res = await daemon.process_once()
        assert res["status"] == "ok"
        assert res["processed"] == 1

        updated = gm.get_goal(int(goal["id"]))
        assert updated is not None
        assert float(updated["progress"]) > 0.0
        assert updated["status"] in {"active", "completed"}
        assert gm.get_active_plan(int(goal["id"])) is not None

    @pytest.mark.asyncio
    async def test_process_once_replans_on_blocked_outcome(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        daemon = GoalCoordinatorDaemon(goals_agent, gm, {
            "enabled": True,
            "auto_pause": False,
            "batch_size": 1,
            "replan_stall_cycles": 1,
        })
        goals_agent.run = AsyncMock(side_effect=[
            (
                '{"strategy":"Use approach A",'
                '"steps":[{"id":"s1","title":"Try A","action":"Do A","success_criteria":"A works"}]}'
            ),
            (
                '{"outcome":"blocked","progress_delta":0.0,"completed":false,'
                '"phase":"blocked","summary":"A failed","next_action":"Try B",'
                '"insight":"dependency issue","alternative":"switch approach"}'
            ),
            (
                '{"strategy":"Use approach B",'
                '"steps":[{"id":"b1","title":"Try B","action":"Do B","success_criteria":"B works"}]}'
            ),
        ])

        goal = gm.add_goal(
            title="Recover from blocker",
            objective="Find alternative solution when blocked",
            user_id="u1",
            cooldown_sec=30,
            source="dashboard",
        )
        res = await daemon.process_once()
        assert res["status"] == "ok"
        assert res["processed"] == 1

        gid = int(goal["id"])
        active_plan = gm.get_active_plan(gid)
        assert active_plan is not None
        assert int(active_plan["version"]) >= 2
        assert "approach b" in str(active_plan["strategy"]).lower()

        latest = gm.get_goal(gid)
        assert latest is not None
        assert latest["status"] == "active"
        assert int(latest.get("stalled_cycles") or 0) >= 1

    @pytest.mark.asyncio
    async def test_autonomous_coding_uses_requested_local_model_and_keeps_running(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        daemon = GoalCoordinatorDaemon(goals_agent, gm, {
            "enabled": True,
            "auto_pause": False,
            "batch_size": 1,
        })
        goals_agent.run = AsyncMock(side_effect=[
            (
                '{"strategy":"Finish requested work then keep improving",'
                '"steps":[{"id":"s1","title":"Patch bug","action":"Patch and verify","success_criteria":"tests pass"}]}'
            ),
            (
                '{"outcome":"done","progress_delta":0.0,"completed":true,'
                '"phase":"verified","summary":"Patched and verified","next_action":"Pick next improvement"}'
            ),
        ])

        goal = gm.add_goal(
            title="Night coding",
            objective="Finish the bugfix and keep improving the workspace",
            user_id="u1",
            cooldown_sec=30,
            source="dashboard",
            goal_type="autonomous_coding",
            config={
                "workspace": "/tmp/workspace",
                "local_model": "ollama:qwen3-coder:30b",
                "continue_after_objective": True,
                "verification_commands": ["pytest -q"],
            },
        )
        res = await daemon.process_once()
        assert res["status"] == "ok"
        assert res["processed"] == 1
        assert goals_agent.run.await_args_list[0].kwargs["requested_model"] == "ollama:qwen3-coder:30b"
        assert goals_agent.run.await_args_list[1].kwargs["requested_model"] == "ollama:qwen3-coder:30b"

        updated = gm.get_goal(int(goal["id"]))
        assert updated is not None
        assert updated["status"] == "active"
        assert updated["current_phase"] == "verified"

    @pytest.mark.asyncio
    async def test_autonomous_coding_completes_when_window_expires(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        daemon = GoalCoordinatorDaemon(goals_agent, gm, {
            "enabled": True,
            "auto_pause": False,
            "batch_size": 1,
        })
        goals_agent.run = AsyncMock()

        goal = gm.add_goal(
            title="Expired night coding",
            objective="Should stop when the window ends",
            user_id="u1",
            source="dashboard",
            goal_type="autonomous_coding",
            config={
                "workspace": "/tmp/workspace",
                "local_model": "qwen3-coder:30b",
                "stop_at": "2000-01-01T00:00:00",
            },
        )
        res = await daemon.process_once()
        assert res["status"] == "ok"
        updated = gm.get_goal(int(goal["id"]))
        assert updated is not None
        assert updated["status"] == "completed"
        assert goals_agent.run.await_count == 0

    @pytest.mark.asyncio
    async def test_self_improvement_uses_local_model_and_default_next_action(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        daemon = GoalCoordinatorDaemon(goals_agent, gm, {
            "enabled": True,
            "auto_pause": False,
            "batch_size": 1,
        })
        goals_agent.run = AsyncMock(side_effect=[
            (
                '{"strategy":"Harden the runtime",'
                '"steps":[{"id":"s1","title":"Add guard","action":"Patch guard","success_criteria":"tests pass"}]}'
            ),
            (
                '{"outcome":"done","progress_delta":0.0,"completed":true,'
                '"phase":"verified","summary":"Added a runtime guard","next_action":""}'
            ),
        ])
        goal = gm.add_goal(
            title="Self improvement",
            objective="Improve LiteAgent using local models only",
            user_id="u1",
            source="dashboard",
            goal_type="self_improvement",
            config={
                "workspace": "/Users/vskorokhod/liteagent",
                "local_model": "qwen3-coder:30b",
                "continue_after_objective": True,
            },
        )
        res = await daemon.process_once()
        assert res["status"] == "ok"
        assert goals_agent.run.await_args_list[0].kwargs["requested_model"] == "qwen3-coder:30b"
        updated = gm.get_goal(int(goal["id"]))
        assert updated is not None
        assert updated["status"] == "active"
        assert "LiteAgent self-improvement" in str(updated.get("last_result") or "")

    @pytest.mark.asyncio
    async def test_process_once_recovers_orphaned_running_goal(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        daemon = GoalCoordinatorDaemon(goals_agent, gm, {
            "enabled": True,
            "auto_pause": False,
            "batch_size": 1,
        })
        goal = gm.add_goal(
            title="Recover stale running night coding",
            objective="Should be completed after expiry once recovered",
            user_id="u1",
            source="dashboard",
            goal_type="autonomous_coding",
            config={
                "workspace": "/tmp/workspace",
                "local_model": "qwen3-coder:30b",
                "stop_at": "2000-01-01T00:00:00",
            },
        )
        gid = int(goal["id"])
        assert gm.claim_running(gid) is True

        res = await daemon.process_once()
        assert res["status"] == "ok"
        updated = gm.get_goal(gid)
        assert updated is not None
        assert updated["status"] == "completed"

    @pytest.mark.asyncio
    async def test_autonomous_coding_guard_pauses_after_failed_streak(self, goals_agent):
        gm = GoalManager(goals_agent.memory.db)
        daemon = GoalCoordinatorDaemon(goals_agent, gm, {
            "enabled": True,
            "auto_pause": False,
            "batch_size": 1,
        })
        goals_agent.run = AsyncMock(side_effect=[
            '{"strategy":"Tight loop","steps":[{"id":"s1","title":"Verify","action":"Run tests","success_criteria":"tests pass"}]}',
            '{"outcome":"failed","progress_delta":0.0,"completed":false,"phase":"verifying","summary":"Tests failed","next_action":"Retry","insight":"Assertion broke"}',
        ])
        goal = gm.add_goal(
            title="Guarded night coding",
            objective="Stop after too many failed cycles",
            user_id="u1",
            source="dashboard",
            goal_type="autonomous_coding",
            config={
                "workspace": "/tmp/workspace",
                "local_model": "qwen3-coder:30b",
                "max_failed_cycles": 2,
            },
        )
        gid = int(goal["id"])
        gm.add_attempt(gid, outcome="failed", summary="Earlier failure", progress_delta=0.0)

        res = await daemon.process_once()
        assert res["status"] == "ok"
        updated = gm.get_goal(gid)
        assert updated is not None
        assert updated["status"] == "paused"
        events = gm.get_goal_events(gid, limit=10)
        assert any(e["event_type"] == "guard_stop" for e in events)
