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


def test_migrates_legacy_schema_without_background_columns(db):
    """Legacy tasks table should be auto-migrated with new background columns."""
    db.executescript("""
        CREATE TABLE tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            query TEXT NOT NULL,
            user_id TEXT NOT NULL,
            task_type TEXT NOT NULL DEFAULT 'one_shot',
            run_at TEXT,
            cron_expr TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            last_run_at TEXT,
            next_run_at TEXT,
            last_result TEXT,
            last_error TEXT,
            run_count INTEGER DEFAULT 0,
            chat_id TEXT
        );
    """)
    db.commit()

    TaskManager(db)
    cols = {row[1] for row in db.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "background" in cols
    assert "priority" in cols
    assert "retry_delay_sec" in cols
    assert "max_attempts" in cols
    assert "attempt_count" in cols


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

    def test_add_background_task_fields(self, tm):
        run_at = (datetime.now() + timedelta(minutes=1)).isoformat()
        task = tm.add_task(
            "bg",
            "long task",
            "u1",
            task_type="one_shot",
            run_at=run_at,
            background=True,
            priority=2,
            retry_delay_sec=30,
            max_attempts=0,
            source="agent",
        )
        assert task["background"] == 1
        assert task["priority"] == 2
        assert task["retry_delay_sec"] == 30
        assert task["max_attempts"] == 0
        assert task["source"] == "agent"


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


class TestUpdateTask:
    def test_update_one_shot_task(self, tm):
        run_at = (datetime.now() + timedelta(hours=1)).isoformat()
        t = tm.add_task("t1", "q1", "u1", task_type="one_shot", run_at=run_at)
        new_run = (datetime.now() + timedelta(hours=2)).isoformat()
        updated = tm.update_task(
            t["id"],
            name="t2",
            query="q2",
            run_at=new_run,
            priority=2,
            retry_delay_sec=20,
            max_attempts=3,
        )
        assert updated["name"] == "t2"
        assert updated["query"] == "q2"
        assert updated["run_at"] == new_run
        assert updated["priority"] == 2
        assert updated["retry_delay_sec"] == 20
        assert updated["max_attempts"] == 3
        assert updated["status"] == "pending"

    def test_update_to_background_recurring_rejected(self, tm):
        t = tm.add_task("r1", "q1", "u1", task_type="recurring", cron_expr="0 9 * * *")
        with pytest.raises(ValueError, match="Background tasks support one-shot"):
            tm.update_task(t["id"], background=1)

    def test_update_missing_task_returns_none(self, tm):
        assert tm.update_task(999, name="x", query="y", run_at="2026-01-01T00:00:00") is None


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

    def test_background_excluded_from_regular_due(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        tm.add_task("bg", "q", "u", task_type="one_shot", run_at=run_at, background=True)
        assert tm.get_due_tasks() == []
        bg_due = tm.get_due_background_tasks()
        assert len(bg_due) == 1
        assert bg_due[0]["background"] == 1

    def test_due_background_sorted_by_priority(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        tm.add_task("p5", "q", "u", task_type="one_shot", run_at=run_at, background=True, priority=5)
        tm.add_task("p1", "q", "u", task_type="one_shot", run_at=run_at, background=True, priority=1)
        tm.add_task("p3", "q", "u", task_type="one_shot", run_at=run_at, background=True, priority=3)
        due = tm.get_due_background_tasks(limit=3)
        assert [t["name"] for t in due] == ["p1", "p3", "p5"]


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

    def test_mark_running_increments_attempt_count(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        t = tm.add_task("t", "q", "u", task_type="one_shot", run_at=run_at, background=True)
        assert tm.mark_running(t["id"]) is True
        got = tm.get_task(t["id"])
        assert got["attempt_count"] == 1


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

    def test_failed_background_requeue(self, tm):
        run_at = (datetime.now() - timedelta(minutes=1)).isoformat()
        t = tm.add_task(
            "bg", "q", "u",
            task_type="one_shot", run_at=run_at,
            background=True, retry_delay_sec=15,
        )
        tm.mark_running(t["id"])
        tm.mark_failed(t["id"], "temporary error", requeue=True)
        got = tm.get_task(t["id"])
        assert got["status"] == "pending"
        assert got["last_error"] == "temporary error"
        assert got["next_run_at"] is not None
        assert datetime.fromisoformat(got["next_run_at"]) > datetime.now()


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
