"""Tests for the TaskManager (liteagent/tasks.py)."""

import sqlite3
from datetime import datetime, timedelta

import pytest

from liteagent.tasks import TaskManager


@pytest.fixture
def db():
    """In-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def tm(db):
    """Fresh TaskManager instance."""
    return TaskManager(db)


# ── CRUD ──────────────────────────────────────────

class TestAddTask:
    def test_add_one_shot(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        task = tm.add_task("call mom", "Remind to call mom", "user1",
                           task_type="one_shot", run_at=run_at)
        assert task["name"] == "call mom"
        assert task["query"] == "Remind to call mom"
        assert task["user_id"] == "user1"
        assert task["task_type"] == "one_shot"
        assert task["status"] == "pending"
        assert task["run_at"] == run_at
        assert task["next_run_at"] == run_at
        assert task["id"] is not None

    def test_add_recurring(self, tm):
        task = tm.add_task("weather", "Check weather", "user1",
                           task_type="recurring", cron_expr="0 9 * * *")
        assert task["task_type"] == "recurring"
        assert task["cron_expr"] == "0 9 * * *"
        assert task["next_run_at"] is not None
        assert task["status"] == "pending"

    def test_add_one_shot_without_run_at_raises(self, tm):
        with pytest.raises(ValueError, match="run_at"):
            tm.add_task("bad", "test", "u", task_type="one_shot")

    def test_add_recurring_without_cron_raises(self, tm):
        with pytest.raises(ValueError, match="cron"):
            tm.add_task("bad", "test", "u", task_type="recurring")

    def test_add_invalid_cron_raises(self, tm):
        with pytest.raises(ValueError):
            tm.add_task("bad", "test", "u", task_type="recurring",
                        cron_expr="not a cron")

    def test_add_with_chat_id(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        task = tm.add_task("tg task", "test", "tg-123",
                           task_type="one_shot", run_at=run_at,
                           chat_id="456789")
        assert task["chat_id"] == "456789"


class TestListTasks:
    def test_list_empty(self, tm):
        assert tm.list_tasks() == []

    def test_list_all(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        tm.add_task("t1", "q1", "u1", task_type="one_shot", run_at=run_at)
        tm.add_task("t2", "q2", "u2", task_type="one_shot", run_at=run_at)
        tasks = tm.list_tasks()
        assert len(tasks) == 2

    def test_list_filter_user(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        tm.add_task("t1", "q1", "u1", task_type="one_shot", run_at=run_at)
        tm.add_task("t2", "q2", "u2", task_type="one_shot", run_at=run_at)
        tasks = tm.list_tasks(user_id="u1")
        assert len(tasks) == 1
        assert tasks[0]["name"] == "t1"

    def test_list_filter_status(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        t = tm.add_task("t1", "q1", "u1", task_type="one_shot", run_at=run_at)
        tm.cancel_task(t["id"])
        assert len(tm.list_tasks(status="cancelled")) == 1
        assert len(tm.list_tasks(status="pending")) == 0


class TestGetTask:
    def test_get_existing(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        t = tm.add_task("t1", "q1", "u1", task_type="one_shot", run_at=run_at)
        got = tm.get_task(t["id"])
        assert got["name"] == "t1"

    def test_get_nonexistent(self, tm):
        assert tm.get_task(999) is None


class TestCancelTask:
    def test_cancel_pending(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        t = tm.add_task("t1", "q1", "u1", task_type="one_shot", run_at=run_at)
        assert tm.cancel_task(t["id"]) is True
        got = tm.get_task(t["id"])
        assert got["status"] == "cancelled"

    def test_cancel_already_cancelled(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        t = tm.add_task("t1", "q1", "u1", task_type="one_shot", run_at=run_at)
        tm.cancel_task(t["id"])
        assert tm.cancel_task(t["id"]) is False

    def test_cancel_nonexistent(self, tm):
        assert tm.cancel_task(999) is False


class TestDeleteTask:
    def test_delete_existing(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        t = tm.add_task("t1", "q1", "u1", task_type="one_shot", run_at=run_at)
        assert tm.delete_task(t["id"]) is True
        assert tm.get_task(t["id"]) is None

    def test_delete_nonexistent(self, tm):
        assert tm.delete_task(999) is False


# ── Execution ──────────────────────────────────────

class TestGetDueTasks:
    def test_no_due_tasks(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        tm.add_task("future", "q", "u", task_type="one_shot", run_at=run_at)
        assert tm.get_due_tasks() == []

    def test_due_task_found(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        tm.add_task("past", "q", "u", task_type="one_shot", run_at=run_at)
        due = tm.get_due_tasks()
        assert len(due) == 1
        assert due[0]["name"] == "past"

    def test_cancelled_not_due(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        t = tm.add_task("past", "q", "u", task_type="one_shot", run_at=run_at)
        tm.cancel_task(t["id"])
        assert tm.get_due_tasks() == []


class TestMarkRunning:
    def test_mark_running_success(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        t = tm.add_task("t", "q", "u", task_type="one_shot", run_at=run_at)
        assert tm.mark_running(t["id"]) is True
        got = tm.get_task(t["id"])
        assert got["status"] == "running"

    def test_mark_running_already_running(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        t = tm.add_task("t", "q", "u", task_type="one_shot", run_at=run_at)
        tm.mark_running(t["id"])
        # Second attempt should fail (atomicity)
        assert tm.mark_running(t["id"]) is False


class TestMarkCompleted:
    def test_one_shot_completed(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        t = tm.add_task("t", "q", "u", task_type="one_shot", run_at=run_at)
        tm.mark_running(t["id"])
        tm.mark_completed(t["id"], "Done!")
        got = tm.get_task(t["id"])
        assert got["status"] == "completed"
        assert got["last_result"] == "Done!"
        assert got["run_count"] == 1

    def test_recurring_reschedules(self, tm):
        # Use every-5-minutes cron so next_run changes visibly
        t = tm.add_task("frequent", "check", "u",
                        task_type="recurring", cron_expr="*/5 * * * *")
        tm.mark_running(t["id"])
        tm.mark_completed(t["id"], "Sunny!")
        got = tm.get_task(t["id"])
        assert got["status"] == "pending"  # back to pending
        assert got["last_result"] == "Sunny!"
        assert got["run_count"] == 1
        assert got["next_run_at"] is not None
        # Next run should be in the future
        next_dt = datetime.fromisoformat(got["next_run_at"])
        assert next_dt > datetime.now()


class TestMarkFailed:
    def test_one_shot_failed(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        t = tm.add_task("t", "q", "u", task_type="one_shot", run_at=run_at)
        tm.mark_running(t["id"])
        tm.mark_failed(t["id"], "timeout")
        got = tm.get_task(t["id"])
        assert got["status"] == "failed"
        assert got["last_error"] == "timeout"

    def test_recurring_failed_reschedules(self, tm):
        t = tm.add_task("daily", "q", "u",
                        task_type="recurring", cron_expr="0 9 * * *")
        tm.mark_running(t["id"])
        tm.mark_failed(t["id"], "API error")
        got = tm.get_task(t["id"])
        assert got["status"] == "pending"  # back to pending despite failure
        assert got["last_error"] == "API error"
        assert got["next_run_at"] is not None


# ── Next Run Calculation ──────────────────────────

class TestCalculateNextRun:
    def test_cron_daily_9am(self, tm):
        ref = datetime(2026, 3, 1, 10, 0, 0)  # 10:00, already past 9
        result = tm._calculate_next_run_from_params(
            "recurring", None, "0 9 * * *", ref)
        assert result is not None
        dt = datetime.fromisoformat(result)
        assert dt.hour == 9
        assert dt.minute == 0
        assert dt.day == 2  # next day

    def test_cron_every_5_minutes(self, tm):
        ref = datetime(2026, 3, 1, 10, 3, 0)
        result = tm._calculate_next_run_from_params(
            "recurring", None, "*/5 * * * *", ref)
        assert result is not None
        dt = datetime.fromisoformat(result)
        assert dt.minute == 5

    def test_one_shot_returns_run_at(self, tm):
        run_at = "2026-12-25T12:00:00"
        result = tm._calculate_next_run_from_params(
            "one_shot", run_at, None, datetime.now())
        assert result == run_at
