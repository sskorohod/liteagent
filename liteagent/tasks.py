"""User-facing task scheduler — persistent tasks with one-shot and recurring support.

Builds on the existing Scheduler (cron engine) by adding a SQLite-backed task table,
CRUD operations, and a 'task_checker' job that executes due tasks through agent.run().
Results are delivered via WebSocket and optionally Telegram.
"""

import asyncio
import logging
import sqlite3
from contextlib import suppress
from datetime import datetime, timedelta
import uuid

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
                chat_id TEXT,
                priority INTEGER NOT NULL DEFAULT 5,
                background INTEGER NOT NULL DEFAULT 0,
                retry_delay_sec INTEGER NOT NULL DEFAULT 45,
                max_attempts INTEGER NOT NULL DEFAULT 0,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'user',
                parent_task_id INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_tasks_status_next
                ON tasks(status, next_run_at);
            CREATE INDEX IF NOT EXISTS idx_tasks_user
                ON tasks(user_id);
        """)
        self._ensure_schema_migrations()
        self.db.commit()

    def _ensure_schema_migrations(self):
        """Best-effort additive migrations for existing task tables."""
        existing = {str(r[1]) for r in self.db.execute("PRAGMA table_info(tasks)").fetchall()}
        migrations = [
            ("priority", "ALTER TABLE tasks ADD COLUMN priority INTEGER NOT NULL DEFAULT 5"),
            ("background", "ALTER TABLE tasks ADD COLUMN background INTEGER NOT NULL DEFAULT 0"),
            ("retry_delay_sec", "ALTER TABLE tasks ADD COLUMN retry_delay_sec INTEGER NOT NULL DEFAULT 45"),
            ("max_attempts", "ALTER TABLE tasks ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 0"),
            ("attempt_count", "ALTER TABLE tasks ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0"),
            ("source", "ALTER TABLE tasks ADD COLUMN source TEXT NOT NULL DEFAULT 'user'"),
            ("parent_task_id", "ALTER TABLE tasks ADD COLUMN parent_task_id INTEGER"),
        ]
        for col, sql in migrations:
            if col in existing:
                continue
            with suppress(Exception):
                self.db.execute(sql)
        with suppress(Exception):
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_tasks_bg_queue
                ON tasks(background, status, priority, next_run_at, created_at)
            """)

    # ── CRUD ──

    def add_task(self, name: str, query: str, user_id: str,
                 task_type: str = "one_shot",
                 run_at: str | None = None,
                 cron_expr: str | None = None,
                 chat_id: str | None = None,
                 priority: int = 5,
                 background: bool = False,
                 retry_delay_sec: int = 45,
                 max_attempts: int = 0,
                 source: str = "user",
                 parent_task_id: int | None = None) -> dict:
        """Create a new task. Returns the task dict."""
        now = datetime.now()
        created_at = now.isoformat()
        priority = max(1, min(int(priority), 9))
        retry_delay_sec = max(5, min(int(retry_delay_sec), 86400))
        max_attempts = max(0, min(int(max_attempts), 100))
        source = str(source or "user")[:32]
        background_int = 1 if background else 0

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
                               cron_expr, status, created_at, next_run_at, chat_id,
                               priority, background, retry_delay_sec, max_attempts,
                               attempt_count, source, parent_task_id)
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?,
                    ?, ?, ?, ?, 0, ?, ?)
        """, (name, query, user_id, task_type, run_at,
              cron_expr, created_at, next_run, chat_id,
              priority, background_int, retry_delay_sec, max_attempts,
              source, parent_task_id))
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

    def update_task(self, task_id: int, **changes) -> dict | None:
        """Update editable task fields. Returns updated task dict."""
        task = self.get_task(task_id)
        if not task:
            return None

        name = str(changes.get("name", task.get("name", "")) or "").strip()
        query = str(changes.get("query", task.get("query", "")) or "").strip()
        if not name or not query:
            raise ValueError("name and query are required")

        run_at = changes.get("run_at", task.get("run_at"))
        cron_expr = changes.get("cron_expr", task.get("cron_expr"))
        if run_at == "":
            run_at = None
        if cron_expr == "":
            cron_expr = None
        if cron_expr:
            cron_expr = str(cron_expr).strip()
        if run_at:
            run_at = str(run_at).strip()

        background = int(changes.get("background", task.get("background", 0)) or 0)
        priority = max(1, min(int(changes.get("priority", task.get("priority", 5)) or 5), 9))
        retry_delay_sec = max(
            5, min(int(changes.get("retry_delay_sec", task.get("retry_delay_sec", 45)) or 45), 86400))
        max_attempts = max(
            0, min(int(changes.get("max_attempts", task.get("max_attempts", 0)) or 0), 100))

        task_type = "recurring" if cron_expr else "one_shot"
        if task_type == "one_shot" and not run_at:
            raise ValueError("One-shot tasks require run_at datetime")
        if task_type == "recurring":
            if background:
                raise ValueError("Background tasks support one-shot only (no recurring cron)")
            parse_cron(cron_expr)

        next_run_at = self._calculate_next_run_from_params(
            task_type, run_at, cron_expr, datetime.now())
        status = str(task.get("status") or "pending")
        if status in {"pending", "failed", "cancelled"}:
            status = "pending"

        self.db.execute("""
            UPDATE tasks
            SET name = ?, query = ?, task_type = ?, run_at = ?, cron_expr = ?,
                next_run_at = ?, status = ?, background = ?, priority = ?,
                retry_delay_sec = ?, max_attempts = ?, last_error = NULL
            WHERE id = ?
        """, (
            name, query, task_type, run_at, cron_expr,
            next_run_at, status, background, priority,
            retry_delay_sec, max_attempts, task_id,
        ))
        self.db.commit()
        return self.get_task(task_id)

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

    def get_due_tasks(self, include_background: bool = False) -> list[dict]:
        """Find due non-background tasks (unless include_background=True)."""
        now = datetime.now().isoformat()
        sql = """
            SELECT * FROM tasks
            WHERE status = 'pending' AND next_run_at IS NOT NULL AND next_run_at <= ?
        """
        params: list = [now]
        if not include_background:
            sql += " AND background = 0"
        sql += """
            ORDER BY next_run_at ASC
        """
        rows = self.db.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_due_background_tasks(self, limit: int = 1) -> list[dict]:
        """Find due background tasks ordered by priority."""
        now = datetime.now().isoformat()
        lim = max(1, min(int(limit), 50))
        rows = self.db.execute("""
            SELECT * FROM tasks
            WHERE status = 'pending'
              AND background = 1
              AND next_run_at IS NOT NULL
              AND next_run_at <= ?
            ORDER BY priority ASC, next_run_at ASC, created_at ASC
            LIMIT ?
        """, (now, lim)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_background_pending(self) -> int:
        """Number of background tasks waiting in queue."""
        row = self.db.execute("""
            SELECT COUNT(*) FROM tasks
            WHERE status = 'pending' AND background = 1
        """).fetchone()
        return int(row[0] if row else 0)

    def get_background_running(self) -> list[dict]:
        """Currently running background tasks (DB state)."""
        rows = self.db.execute("""
            SELECT * FROM tasks
            WHERE status = 'running' AND background = 1
            ORDER BY last_run_at DESC, created_at DESC
        """).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def mark_running(self, task_id: int) -> bool:
        """Atomically mark a task as running (prevents double execution)."""
        cur = self.db.execute(
            """UPDATE tasks
               SET status = 'running',
                   attempt_count = COALESCE(attempt_count, 0) + 1
               WHERE id = ? AND status = 'pending'""",
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

    def mark_failed(self, task_id: int, error: str, *,
                    requeue: bool = False,
                    retry_delay_sec: int | None = None):
        """Mark task as failed (optionally requeue for retry)."""
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
        elif requeue:
            delay = retry_delay_sec if retry_delay_sec is not None else int(task.get("retry_delay_sec") or 45)
            delay = max(5, min(int(delay), 86400))
            next_run = (now + timedelta(seconds=delay)).isoformat()
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
            "priority", "background", "retry_delay_sec", "max_attempts",
            "attempt_count", "source", "parent_task_id",
        ]
        return dict(zip(cols, row))


class BackgroundTaskDaemon:
    """Always-on worker for autonomous background tasks."""

    def __init__(self, agent, task_manager: TaskManager, config: dict | None = None):
        self.agent = agent
        self.task_manager = task_manager
        self.config = config or {}
        self._task: asyncio.Task | None = None
        self._running = False
        self._worker_id = f"bg-{uuid.uuid4().hex[:8]}"
        self._active: dict[int, dict] = {}
        self._last_pause_reason = ""
        self._last_pause_at = 0.0
        self._last_cycle_at = ""
        self._processed_total = 0
        self._failed_total = 0

    def _enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _interval_sec(self) -> float:
        try:
            raw = float(self.config.get("interval_sec", 1.0))
        except (TypeError, ValueError):
            raw = 1.0
        return max(0.2, min(raw, 60.0))

    def _batch_size(self) -> int:
        try:
            raw = int(self.config.get("batch_size", 1))
        except (TypeError, ValueError):
            raw = 1
        return max(1, min(raw, 10))

    def _auto_pause(self) -> bool:
        return bool(self.config.get("auto_pause", True))

    def _pause_active_threshold(self) -> int:
        try:
            raw = int(self.config.get("pause_active_requests", 1))
        except (TypeError, ValueError):
            raw = 1
        return max(0, min(raw, 100))

    def _pause_queued_threshold(self) -> int:
        try:
            raw = int(self.config.get("pause_queued_requests", 2))
        except (TypeError, ValueError):
            raw = 2
        return max(0, min(raw, 1000))

    def _retry_delay_default(self) -> int:
        try:
            raw = int(self.config.get("retry_delay_sec", 45))
        except (TypeError, ValueError):
            raw = 45
        return max(5, min(raw, 86400))

    def _max_attempts_default(self) -> int:
        # 0 means unlimited retries ("work until solved"), bounded by backoff delay.
        try:
            raw = int(self.config.get("max_attempts", 0))
        except (TypeError, ValueError):
            raw = 0
        return max(0, min(raw, 100))

    def _is_high_load(self) -> tuple[bool, str]:
        if not self._auto_pause():
            return False, ""
        try:
            from .agent import LiteAgent
            active = len(LiteAgent.get_active_requests())
            queued = len(LiteAgent.get_queued_requests())
        except Exception:
            return False, ""

        active_threshold = self._pause_active_threshold()
        queued_threshold = self._pause_queued_threshold()
        if active_threshold > 0 and active >= active_threshold:
            return True, f"active_requests={active}"
        if queued_threshold > 0 and queued >= queued_threshold:
            return True, f"queued_requests={queued}"
        return False, ""

    async def process_once(self) -> dict:
        """Process one daemon cycle (priority queue + retry semantics)."""
        if not self._enabled():
            return {"status": "disabled", "processed": 0, "failed": 0, "retried": 0}

        paused, reason = self._is_high_load()
        if paused:
            self._last_pause_reason = reason
            self._last_pause_at = datetime.now().timestamp()
            return {"status": "paused", "reason": reason, "processed": 0, "failed": 0, "retried": 0}

        due = self.task_manager.get_due_background_tasks(limit=self._batch_size() * 3)
        if not due:
            return {"status": "idle", "processed": 0, "failed": 0, "retried": 0}

        claimed: list[dict] = []
        for task in due:
            if len(claimed) >= self._batch_size():
                break
            task_id = int(task["id"])
            if not self.task_manager.mark_running(task_id):
                continue
            fresh = self.task_manager.get_task(task_id) or task
            claimed.append(fresh)

        if not claimed:
            return {"status": "idle", "processed": 0, "failed": 0, "retried": 0}

        processed = 0
        failed = 0
        retried = 0
        for task in claimed:
            task_id = int(task["id"])
            task_name = str(task.get("name") or f"task-{task_id}")
            attempt = int(task.get("attempt_count") or 1)
            self._active[task_id] = {
                "task_id": task_id,
                "name": task_name,
                "user_id": task.get("user_id", ""),
                "priority": int(task.get("priority") or 5),
                "attempt": attempt,
                "max_attempts": int(task.get("max_attempts") or self._max_attempts_default()),
                "retry_delay_sec": int(task.get("retry_delay_sec") or self._retry_delay_default()),
                "query_preview": str(task.get("query") or "")[:220],
                "source": str(task.get("source") or "user"),
                "status": "running",
                "phase_label": "Background daemon execution",
                "started_at": datetime.now().isoformat(),
            }
            self.agent._ws_broadcast("background_task_started", self._active[task_id])
            try:
                result = await self.agent.run(str(task.get("query") or ""), str(task.get("user_id") or "default"))
                self.task_manager.mark_completed(task_id, result)
                processed += 1
                self._processed_total += 1
                self.agent._ws_broadcast("background_task_done", {
                    "task_id": task_id,
                    "name": task_name,
                    "user_id": task.get("user_id", ""),
                    "attempt": attempt,
                    "result": str(result)[:300],
                })
                _publish_task_message(
                    self.agent,
                    task,
                    f"✅ Background task \"{task_name}\" completed.\n\n{str(result)[:3500]}",
                )
                await _notify_telegram(self.agent, task, result)
            except Exception as e:
                err = str(e)
                failed += 1
                self._failed_total += 1
                max_attempts = int(task.get("max_attempts") or self._max_attempts_default())
                max_attempts = max(0, max_attempts)
                retry_delay = int(task.get("retry_delay_sec") or self._retry_delay_default())
                should_retry = max_attempts == 0 or attempt < max_attempts
                self.task_manager.mark_failed(
                    task_id,
                    err,
                    requeue=should_retry,
                    retry_delay_sec=retry_delay,
                )
                event_data = {
                    "task_id": task_id,
                    "name": task_name,
                    "user_id": task.get("user_id", ""),
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "retrying": should_retry,
                    "retry_delay_sec": retry_delay if should_retry else 0,
                    "error": err[:220],
                }
                if should_retry:
                    retried += 1
                    self.agent._ws_broadcast("background_task_retry", event_data)
                    _publish_task_message(
                        self.agent,
                        task,
                        f"⚠️ Background task \"{task_name}\" failed (attempt {attempt}), retrying in {retry_delay}s.\nError: {err[:900]}",
                    )
                else:
                    self.agent._ws_broadcast("background_task_failed", event_data)
                    _publish_task_message(
                        self.agent,
                        task,
                        f"❌ Background task \"{task_name}\" failed.\nError: {err[:1200]}",
                    )
            finally:
                self._active.pop(task_id, None)

        return {"status": "ok", "processed": processed, "failed": failed, "retried": retried}

    async def _loop(self):
        interval = self._interval_sec()
        while self._running:
            self._last_cycle_at = datetime.now().isoformat()
            try:
                result = await self.process_once()
                if result.get("status") == "ok" and int(result.get("processed", 0)) > 0:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Background task daemon error: %s", e)
                await asyncio.sleep(interval)

    async def start(self) -> dict:
        if not self._enabled():
            return {"status": "disabled"}
        if self._task and not self._task.done():
            return {"status": "already_running", "worker_id": self._worker_id}
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Background task daemon started (worker=%s)", self._worker_id)
        return {"status": "started", "worker_id": self._worker_id}

    async def stop(self) -> dict:
        self._running = False
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        return {"status": "stopped"}

    def get_active_tasks(self) -> list[dict]:
        return list(self._active.values())

    def state(self) -> dict:
        running = bool(self._task and not self._task.done())
        return {
            "enabled": self._enabled(),
            "running": running,
            "worker_id": self._worker_id,
            "active_count": len(self._active),
            "pending": self.task_manager.count_background_pending(),
            "last_pause_reason": self._last_pause_reason,
            "last_pause_at": self._last_pause_at,
            "last_cycle_at": self._last_cycle_at,
            "processed_total": self._processed_total,
            "failed_total": self._failed_total,
        }


def setup_background_task_daemon(agent, task_manager: TaskManager,
                                 config: dict | None = None) -> BackgroundTaskDaemon:
    """Create daemon instance configured from scheduler.background_tasks."""
    cfg = config or {}
    sched_cfg = cfg.get("scheduler", {}) if isinstance(cfg, dict) else {}
    bg_cfg = sched_cfg.get("background_tasks", {}) if isinstance(sched_cfg, dict) else {}
    daemon = BackgroundTaskDaemon(agent, task_manager, bg_cfg)
    logger.info(
        "Background task daemon configured (enabled=%s, interval=%.2fs, batch=%d)",
        daemon._enabled(), daemon._interval_sec(), daemon._batch_size(),
    )
    return daemon


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
                _publish_task_message(
                    agent, task,
                    f"✅ Task \"{task['name']}\" completed.\n\n{result[:3500]}",
                )

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
                _publish_task_message(
                    agent, task,
                    f"❌ Task \"{task['name']}\" failed: {str(e)[:1200]}",
                )

    scheduler.add_job("task_checker", "* * * * *", _check_and_run,
                      max_runtime_sec=120, retry_on_fail=False)
    logger.info("Task checker registered (runs every minute)")


async def _notify_telegram(agent, task: dict, result: str):
    """Send task result to Telegram if the user has a chat_id."""
    chat_id = task.get("chat_id")
    if not chat_id:
        tg_cfg = (agent.config or {}).get("channels", {}).get("telegram", {})
        chat_id = tg_cfg.get("chat_id")
        if not chat_id:
            chat_ids = tg_cfg.get("chat_ids")
            if isinstance(chat_ids, (list, tuple)) and chat_ids:
                chat_id = chat_ids[0]
    if not chat_id:
        return
    chat_id_str = str(chat_id).strip()
    if "," in chat_id_str:
        chat_id_str = chat_id_str.split(",", 1)[0].strip()
    if not chat_id_str:
        return

    tg_app = getattr(agent, "_telegram_app", None)
    if not tg_app:
        return

    try:
        text = f"Task: {task['name']}\n\n{result}"
        # Respect Telegram message length limit
        for i in range(0, len(text), TG_MAX_LENGTH):
            await tg_app.bot.send_message(
                chat_id=int(chat_id_str), text=text[i:i + TG_MAX_LENGTH])
    except Exception as e:
        logger.warning("Failed to send Telegram notification for task #%d: %s",
                       task["id"], e)


def _publish_task_message(agent, task: dict, text: str):
    """Deliver task completion message into chat history + live dashboard WS."""
    user_id = str(task.get("user_id") or "dashboard-user")
    msg = str(text or "").strip()
    if not msg:
        return
    try:
        if hasattr(agent, "memory"):
            agent.memory.add_message(user_id, "assistant", msg)
    except Exception:
        pass
    try:
        agent._ws_broadcast("task_message", {
            "task_id": task.get("id"),
            "name": task.get("name", ""),
            "user_id": user_id,
            "message": msg[:5000],
        })
    except Exception:
        pass
