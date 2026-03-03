"""User-facing task scheduler — persistent tasks with one-shot and recurring support.

Builds on the existing Scheduler (cron engine) by adding a SQLite-backed task table,
CRUD operations, and a 'task_checker' job that executes due tasks through agent.run().
Results are delivered via WebSocket and optionally Telegram.
"""

import asyncio
import logging
import sqlite3
from datetime import datetime, timedelta

from .scheduler import parse_cron, cron_matches

logger = logging.getLogger(__name__)

TG_MAX_LENGTH = 4096


# ── TaskManager ─────────────────────────────────────────────

class TaskManager:
    """Manages persistent user tasks in SQLite."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._init_table()

    def _init_table(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
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
            CREATE INDEX IF NOT EXISTS idx_tasks_status_next
                ON tasks(status, next_run_at);
            CREATE INDEX IF NOT EXISTS idx_tasks_user
                ON tasks(user_id);
        """)
        self.db.commit()

    # ── CRUD ──

    def add_task(self, name: str, query: str, user_id: str,
                 task_type: str = "one_shot",
                 run_at: str | None = None,
                 cron_expr: str | None = None,
                 chat_id: str | None = None) -> dict:
        """Create a new task. Returns the task dict."""
        now = datetime.now()
        created_at = now.isoformat()

        # Validate
        if task_type == "recurring" and not cron_expr:
            raise ValueError("Recurring tasks require a cron expression")
        if task_type == "one_shot" and not run_at:
            raise ValueError("One-shot tasks require run_at datetime")
        if cron_expr:
            parse_cron(cron_expr)  # validate syntax

        # Calculate next run
        next_run = self._calculate_next_run_from_params(
            task_type, run_at, cron_expr, now)

        cur = self.db.execute("""
            INSERT INTO tasks (name, query, user_id, task_type, run_at,
                               cron_expr, status, created_at, next_run_at, chat_id)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
        """, (name, query, user_id, task_type, run_at,
              cron_expr, created_at, next_run, chat_id))
        self.db.commit()
        return self.get_task(cur.lastrowid)

    def list_tasks(self, user_id: str | None = None,
                   status: str | None = None) -> list[dict]:
        """List tasks with optional filters."""
        sql = "SELECT * FROM tasks WHERE 1=1"
        params = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        rows = self.db.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_task(self, task_id: int) -> dict | None:
        """Get a single task by ID."""
        row = self.db.execute(
            "SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def cancel_task(self, task_id: int) -> bool:
        """Cancel a pending task."""
        cur = self.db.execute(
            "UPDATE tasks SET status = 'cancelled' WHERE id = ? AND status IN ('pending', 'failed')",
            (task_id,))
        self.db.commit()
        return cur.rowcount > 0

    def delete_task(self, task_id: int) -> bool:
        """Hard delete a task."""
        cur = self.db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.db.commit()
        return cur.rowcount > 0

    # ── Execution helpers ──

    def get_due_tasks(self) -> list[dict]:
        """Find tasks ready to execute (next_run_at <= now, status=pending)."""
        now = datetime.now().isoformat()
        rows = self.db.execute("""
            SELECT * FROM tasks
            WHERE status = 'pending' AND next_run_at IS NOT NULL AND next_run_at <= ?
            ORDER BY next_run_at ASC
        """, (now,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def mark_running(self, task_id: int) -> bool:
        """Atomically mark a task as running (prevents double execution)."""
        cur = self.db.execute(
            "UPDATE tasks SET status = 'running' WHERE id = ? AND status = 'pending'",
            (task_id,))
        self.db.commit()
        return cur.rowcount > 0

    def mark_completed(self, task_id: int, result: str):
        """Mark task completed. For recurring, schedule next run."""
        now = datetime.now()
        task = self.get_task(task_id)
        if not task:
            return

        if task["task_type"] == "recurring" and task["cron_expr"]:
            # Schedule next run
            next_run = self._calculate_next_run_from_params(
                "recurring", None, task["cron_expr"], now + timedelta(minutes=1))
            self.db.execute("""
                UPDATE tasks
                SET status = 'pending', last_run_at = ?, last_result = ?,
                    last_error = NULL, run_count = run_count + 1, next_run_at = ?
                WHERE id = ?
            """, (now.isoformat(), result[:5000], next_run, task_id))
        else:
            # One-shot: mark completed
            self.db.execute("""
                UPDATE tasks
                SET status = 'completed', last_run_at = ?, last_result = ?,
                    last_error = NULL, run_count = run_count + 1
                WHERE id = ?
            """, (now.isoformat(), result[:5000], task_id))
        self.db.commit()

    def mark_failed(self, task_id: int, error: str):
        """Mark task as failed."""
        now = datetime.now()
        task = self.get_task(task_id)
        if not task:
            return

        if task["task_type"] == "recurring" and task["cron_expr"]:
            # Recurring: reschedule despite failure
            next_run = self._calculate_next_run_from_params(
                "recurring", None, task["cron_expr"], now + timedelta(minutes=1))
            self.db.execute("""
                UPDATE tasks
                SET status = 'pending', last_run_at = ?, last_error = ?,
                    run_count = run_count + 1, next_run_at = ?
                WHERE id = ?
            """, (now.isoformat(), error[:2000], next_run, task_id))
        else:
            self.db.execute("""
                UPDATE tasks
                SET status = 'failed', last_run_at = ?, last_error = ?,
                    run_count = run_count + 1
                WHERE id = ?
            """, (now.isoformat(), error[:2000], task_id))
        self.db.commit()

    # ── Helpers ──

    def _calculate_next_run_from_params(
        self, task_type: str, run_at: str | None,
        cron_expr: str | None, ref_time: datetime,
    ) -> str | None:
        """Calculate next run time for a task."""
        if task_type == "one_shot" and run_at:
            return run_at
        if task_type == "recurring" and cron_expr:
            parsed = parse_cron(cron_expr)
            # Scan forward minute-by-minute (max 48h = 2880 minutes)
            dt = ref_time.replace(second=0, microsecond=0)
            for _ in range(2880):
                if cron_matches(parsed, dt):
                    return dt.isoformat()
                dt += timedelta(minutes=1)
            logger.warning("Could not find next cron match within 48h for '%s'", cron_expr)
            return None
        return None

    def _row_to_dict(self, row: sqlite3.Row | tuple) -> dict:
        """Convert a DB row to dict."""
        if row is None:
            return {}
        if isinstance(row, sqlite3.Row):
            return dict(row)
        cols = [
            "id", "name", "query", "user_id", "task_type", "run_at",
            "cron_expr", "status", "created_at", "last_run_at", "next_run_at",
            "last_result", "last_error", "run_count", "chat_id",
        ]
        return dict(zip(cols, row))


# ── Task Checker Job ────────────────────────────────────────

def setup_task_checker(scheduler, agent, task_manager: TaskManager):
    """Register 'task_checker' scheduler job that runs every minute."""

    async def _check_and_run():
        due = task_manager.get_due_tasks()
        if not due:
            return
        logger.info("Task checker found %d due task(s)", len(due))

        for task in due:
            if not task_manager.mark_running(task["id"]):
                continue  # already picked up by another cycle

            try:
                logger.info("Executing task #%d '%s' for user %s",
                            task["id"], task["name"], task["user_id"])
                result = await agent.run(task["query"], task["user_id"])
                task_manager.mark_completed(task["id"], result)

                # Broadcast to dashboard
                agent._ws_broadcast("task_completed", {
                    "task_id": task["id"],
                    "name": task["name"],
                    "result": result[:500],
                    "user_id": task["user_id"],
                })

                # Telegram notification
                await _notify_telegram(agent, task, result)

                logger.info("Task #%d '%s' completed", task["id"], task["name"])

            except Exception as e:
                logger.error("Task #%d '%s' failed: %s",
                             task["id"], task["name"], e, exc_info=True)
                task_manager.mark_failed(task["id"], str(e))
                agent._ws_broadcast("task_failed", {
                    "task_id": task["id"],
                    "name": task["name"],
                    "error": str(e)[:200],
                })

    scheduler.add_job("task_checker", "* * * * *", _check_and_run,
                      max_runtime_sec=120, retry_on_fail=False)
    logger.info("Task checker registered (runs every minute)")


async def _notify_telegram(agent, task: dict, result: str):
    """Send task result to Telegram if the user has a chat_id."""
    chat_id = task.get("chat_id")
    if not chat_id:
        return

    tg_app = getattr(agent, "_telegram_app", None)
    if not tg_app:
        return

    try:
        text = f"Task: {task['name']}\n\n{result}"
        # Respect Telegram message length limit
        for i in range(0, len(text), TG_MAX_LENGTH):
            await tg_app.bot.send_message(
                chat_id=int(chat_id), text=text[i:i + TG_MAX_LENGTH])
    except Exception as e:
        logger.warning("Failed to send Telegram notification for task #%d: %s",
                       task["id"], e)
