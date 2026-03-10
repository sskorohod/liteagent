"""Persistent long-running goals + proactive coordinator daemon."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from contextlib import suppress
from datetime import datetime, timedelta
from typing import Any

from .night_coding import (
    build_execute_prompt as build_night_coding_execute_prompt,
    build_plan_prompt as build_night_coding_plan_prompt,
    normalize_session_config,
    session_expired as night_coding_session_expired,
)

logger = logging.getLogger(__name__)


_PLAN_STEP_STATUSES = {"pending", "in_progress", "done", "blocked", "failed", "cancelled"}
_PLAN_ACTIVE_STATUSES = {"pending", "in_progress", "blocked", "failed"}


class GoalManager:
    """CRUD + lifecycle + planning state for long-running user goals."""

    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self._goal_columns_cache: list[str] | None = None
        self._init_tables()

    def _init_tables(self):
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS goals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                objective TEXT NOT NULL,
                user_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                priority INTEGER NOT NULL DEFAULT 5,
                progress REAL NOT NULL DEFAULT 0.0,
                target_steps INTEGER NOT NULL DEFAULT 1,
                completed_steps INTEGER NOT NULL DEFAULT 0,
                current_phase TEXT NOT NULL DEFAULT 'planned',
                strategy TEXT NOT NULL DEFAULT '',
                run_count INTEGER NOT NULL DEFAULT 0,
                cycle_count INTEGER NOT NULL DEFAULT 0,
                max_cycles INTEGER NOT NULL DEFAULT 0,
                cooldown_sec INTEGER NOT NULL DEFAULT 90,
                source TEXT NOT NULL DEFAULT 'dashboard',
                goal_type TEXT NOT NULL DEFAULT 'generic',
                config_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_run_at TEXT,
                next_run_at TEXT,
                last_result TEXT,
                last_error TEXT,
                stalled_cycles INTEGER NOT NULL DEFAULT 0,
                last_progress_at TEXT,
                last_plan_at TEXT,
                plan_version INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_goals_status_next
                ON goals(status, priority, next_run_at, created_at);
            CREATE INDEX IF NOT EXISTS idx_goals_user_status
                ON goals(user_id, status, updated_at DESC);

            CREATE TABLE IF NOT EXISTS goal_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                message TEXT DEFAULT '',
                payload_json TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_goal_events_goal_created
                ON goal_events(goal_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS goal_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                trigger TEXT NOT NULL DEFAULT 'initial',
                strategy TEXT DEFAULT '',
                steps_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_goal_plans_goal_version
                ON goal_plans(goal_id, version);
            CREATE INDEX IF NOT EXISTS idx_goal_plans_goal_status
                ON goal_plans(goal_id, status, updated_at DESC);

            CREATE TABLE IF NOT EXISTS goal_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id INTEGER NOT NULL,
                plan_version INTEGER NOT NULL DEFAULT 0,
                step_id TEXT DEFAULT '',
                step_title TEXT DEFAULT '',
                action_query TEXT DEFAULT '',
                outcome TEXT NOT NULL DEFAULT 'unknown',
                progress_delta REAL NOT NULL DEFAULT 0.0,
                summary TEXT DEFAULT '',
                insight TEXT DEFAULT '',
                error TEXT DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_goal_attempts_goal_created
                ON goal_attempts(goal_id, created_at DESC);
            """
        )
        self._ensure_schema_migrations()
        self.db.commit()

    def _ensure_schema_migrations(self):
        """Best-effort additive migrations for existing goals table."""
        existing = {str(r[1]) for r in self.db.execute("PRAGMA table_info(goals)").fetchall()}
        migrations = [
            ("stalled_cycles", "ALTER TABLE goals ADD COLUMN stalled_cycles INTEGER NOT NULL DEFAULT 0"),
            ("last_progress_at", "ALTER TABLE goals ADD COLUMN last_progress_at TEXT"),
            ("last_plan_at", "ALTER TABLE goals ADD COLUMN last_plan_at TEXT"),
            ("plan_version", "ALTER TABLE goals ADD COLUMN plan_version INTEGER NOT NULL DEFAULT 0"),
            ("goal_type", "ALTER TABLE goals ADD COLUMN goal_type TEXT NOT NULL DEFAULT 'generic'"),
            ("config_json", "ALTER TABLE goals ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'"),
        ]
        for col, sql in migrations:
            if col in existing:
                continue
            with suppress(Exception):
                self.db.execute(sql)
        self._goal_columns_cache = None

    def _goal_columns(self) -> list[str]:
        if self._goal_columns_cache:
            return list(self._goal_columns_cache)
        rows = self.db.execute("PRAGMA table_info(goals)").fetchall()
        cols = [str(r[1]) for r in rows] if rows else []
        self._goal_columns_cache = cols
        return list(cols)

    @staticmethod
    def _now() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _clamp_int(value: int, lo: int, hi: int) -> int:
        return max(lo, min(int(value), hi))

    @staticmethod
    def _clamp_progress(value: float) -> float:
        return max(0.0, min(float(value), 1.0))

    @staticmethod
    def _safe_json_obj(raw: str | None) -> dict:
        try:
            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _safe_json_list(raw: str | None) -> list:
        try:
            parsed = json.loads(raw or "[]")
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    @staticmethod
    def _normalize_step(step: Any, idx: int) -> dict | None:
        if isinstance(step, str):
            title = str(step).strip()
            if not title:
                return None
            return {
                "id": f"s{idx + 1}",
                "title": title[:200],
                "action": title[:600],
                "success_criteria": "",
                "status": "pending",
                "note": "",
                "updated_at": "",
            }
        if not isinstance(step, dict):
            return None

        sid = str(step.get("id") or f"s{idx + 1}").strip()[:40] or f"s{idx + 1}"
        title = str(step.get("title") or step.get("step") or step.get("action") or "").strip()[:200]
        action = str(step.get("action") or title or "").strip()[:1200]
        if not title:
            title = action[:200]
        if not title:
            return None

        status = str(step.get("status") or "pending").strip().lower()
        if status not in _PLAN_STEP_STATUSES:
            status = "pending"

        return {
            "id": sid,
            "title": title,
            "action": action,
            "success_criteria": str(step.get("success_criteria") or "").strip()[:500],
            "status": status,
            "note": str(step.get("note") or "").strip()[:1000],
            "updated_at": str(step.get("updated_at") or "").strip()[:40],
        }

    @classmethod
    def _normalize_steps(cls, steps: list[Any], max_steps: int = 8) -> list[dict]:
        out: list[dict] = []
        for idx, step in enumerate(steps or []):
            if len(out) >= max(1, min(int(max_steps), 20)):
                break
            normalized = cls._normalize_step(step, idx)
            if normalized:
                out.append(normalized)
        return out

    def _row_to_dict(self, row: sqlite3.Row | tuple | None) -> dict:
        if row is None:
            return {}
        def _enrich(data: dict) -> dict:
            config = self._safe_json_obj(data.get("config_json"))
            data["goal_type"] = str(data.get("goal_type") or "generic")
            data["config"] = config
            return data
        if isinstance(row, sqlite3.Row):
            return _enrich(dict(row))
        cols = self._goal_columns()
        return _enrich(dict(zip(cols, row)))

    def _add_event(self, goal_id: int, event_type: str, message: str = "", payload: dict | None = None):
        self.db.execute(
            """INSERT INTO goal_events (goal_id, event_type, message, payload_json, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                int(goal_id),
                str(event_type or "event")[:40],
                str(message or "")[:1000],
                json.dumps(payload or {}, ensure_ascii=False),
                self._now(),
            ),
        )

    # --- CRUD ---

    def add_goal(self, *, title: str, objective: str, user_id: str,
                 priority: int = 5, target_steps: int = 4,
                 max_cycles: int = 0, cooldown_sec: int = 90,
                 source: str = "dashboard",
                 goal_type: str = "generic",
                 config: dict | None = None) -> dict:
        now = self._now()
        priority = self._clamp_int(priority, 1, 9)
        target_steps = self._clamp_int(target_steps, 1, 1000)
        max_cycles = self._clamp_int(max_cycles, 0, 10000)
        cooldown_sec = self._clamp_int(cooldown_sec, 5, 86400)
        goal_type = str(goal_type or "generic").strip().lower() or "generic"
        config_json = json.dumps(config or {}, ensure_ascii=False)
        cur = self.db.execute(
            """INSERT INTO goals
               (title, objective, user_id, status, priority, progress,
                target_steps, completed_steps, current_phase, strategy,
                run_count, cycle_count, max_cycles, cooldown_sec, source, goal_type, config_json,
                created_at, updated_at, next_run_at,
                stalled_cycles, last_progress_at, last_plan_at, plan_version)
               VALUES (?, ?, ?, 'active', ?, 0.0,
                       ?, 0, 'planned', '',
                       0, 0, ?, ?, ?, ?, ?,
                       ?, ?, ?,
                       0, NULL, NULL, 0)""",
            (
                str(title or "").strip()[:160],
                str(objective or "").strip()[:4000],
                str(user_id or "default"),
                priority,
                target_steps,
                max_cycles,
                cooldown_sec,
                str(source or "dashboard")[:40],
                goal_type,
                config_json,
                now,
                now,
                now,
            ),
        )
        goal_id = int(cur.lastrowid)
        self._add_event(goal_id, "created", f"Goal created: {title}", {"source": source})
        self.db.commit()
        return self.get_goal(goal_id)

    def get_goal(self, goal_id: int) -> dict | None:
        row = self.db.execute(
            "SELECT * FROM goals WHERE id = ?",
            (int(goal_id),),
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_goals(self, *, user_id: str | None = None,
                   statuses: list[str] | None = None,
                   limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM goals WHERE 1=1"
        params: list = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if statuses:
            clean = [str(s).strip().lower() for s in statuses if str(s).strip()]
            if clean:
                placeholders = ",".join("?" for _ in clean)
                sql += f" AND status IN ({placeholders})"
                params.extend(clean)
        lim = self._clamp_int(limit, 1, 300)
        sql += (
            " ORDER BY "
            "CASE status "
            "WHEN 'running' THEN 0 "
            "WHEN 'active' THEN 1 "
            "WHEN 'paused' THEN 2 "
            "WHEN 'failed' THEN 3 "
            "WHEN 'completed' THEN 4 "
            "ELSE 5 END, "
            "priority ASC, updated_at DESC "
            "LIMIT ?"
        )
        params.append(lim)
        rows = self.db.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_goal_events(self, goal_id: int, limit: int = 20) -> list[dict]:
        lim = self._clamp_int(limit, 1, 200)
        rows = self.db.execute(
            """SELECT id, goal_id, event_type, message, payload_json, created_at
               FROM goal_events
               WHERE goal_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (int(goal_id), lim),
        ).fetchall()
        out = []
        for row in rows:
            payload = self._safe_json_obj(row[4])
            out.append({
                "id": int(row[0]),
                "goal_id": int(row[1]),
                "event_type": str(row[2] or ""),
                "message": str(row[3] or ""),
                "payload": payload,
                "created_at": row[5],
            })
        return out

    # --- Planning state ---

    def get_active_plan(self, goal_id: int) -> dict | None:
        row = self.db.execute(
            """SELECT id, goal_id, version, status, trigger, strategy, steps_json, created_at, updated_at
               FROM goal_plans
               WHERE goal_id = ? AND status = 'active'
               ORDER BY version DESC
               LIMIT 1""",
            (int(goal_id),),
        ).fetchone()
        if not row:
            return None
        return {
            "id": int(row[0]),
            "goal_id": int(row[1]),
            "version": int(row[2]),
            "status": str(row[3] or "active"),
            "trigger": str(row[4] or ""),
            "strategy": str(row[5] or ""),
            "steps": self._normalize_steps(self._safe_json_list(row[6])),
            "created_at": row[7],
            "updated_at": row[8],
        }

    def get_plan_history(self, goal_id: int, limit: int = 5) -> list[dict]:
        lim = self._clamp_int(limit, 1, 50)
        rows = self.db.execute(
            """SELECT id, goal_id, version, status, trigger, strategy, steps_json, created_at, updated_at
               FROM goal_plans
               WHERE goal_id = ?
               ORDER BY version DESC
               LIMIT ?""",
            (int(goal_id), lim),
        ).fetchall()
        out = []
        for row in rows:
            out.append({
                "id": int(row[0]),
                "goal_id": int(row[1]),
                "version": int(row[2]),
                "status": str(row[3] or ""),
                "trigger": str(row[4] or ""),
                "strategy": str(row[5] or ""),
                "steps": self._normalize_steps(self._safe_json_list(row[6])),
                "created_at": row[7],
                "updated_at": row[8],
            })
        return out

    def upsert_plan(self, goal_id: int, *, strategy: str, steps: list[Any], trigger: str = "initial") -> dict | None:
        goal = self.get_goal(goal_id)
        if not goal:
            return None

        normalized = self._normalize_steps(steps)
        if not normalized:
            objective = str(goal.get("objective") or "").strip()[:180]
            normalized = [
                {
                    "id": "s1",
                    "title": "Clarify next concrete action",
                    "action": f"Break objective into one executable action: {objective}",
                    "success_criteria": "One clear action defined",
                    "status": "pending",
                    "note": "",
                    "updated_at": "",
                }
            ]

        now = self._now()
        row = self.db.execute(
            "SELECT COALESCE(MAX(version), 0) FROM goal_plans WHERE goal_id = ?",
            (int(goal_id),),
        ).fetchone()
        version = int((row[0] if row else 0) or 0) + 1

        self.db.execute(
            "UPDATE goal_plans SET status = 'superseded', updated_at = ? WHERE goal_id = ? AND status = 'active'",
            (now, int(goal_id)),
        )
        self.db.execute(
            """INSERT INTO goal_plans
               (goal_id, version, status, trigger, strategy, steps_json, created_at, updated_at)
               VALUES (?, ?, 'active', ?, ?, ?, ?, ?)""",
            (
                int(goal_id),
                version,
                str(trigger or "initial")[:60],
                str(strategy or "").strip()[:2000],
                json.dumps(normalized, ensure_ascii=False),
                now,
                now,
            ),
        )
        self.db.execute(
            """UPDATE goals
               SET strategy = ?,
                   plan_version = ?,
                   last_plan_at = ?,
                   current_phase = 'planning',
                   updated_at = ?
               WHERE id = ?""",
            (
                str(strategy or "").strip()[:2000],
                int(version),
                now,
                now,
                int(goal_id),
            ),
        )
        self._add_event(
            int(goal_id),
            "plan",
            f"Plan v{version} created ({str(trigger or 'initial')[:50]})",
            {
                "version": int(version),
                "trigger": str(trigger or "initial")[:60],
                "steps": len(normalized),
            },
        )
        self.db.commit()
        return self.get_active_plan(goal_id)

    def get_next_plan_step(self, goal_id: int) -> dict | None:
        plan = self.get_active_plan(goal_id)
        if not plan:
            return None
        steps = list(plan.get("steps") or [])
        for wanted in ("pending", "in_progress"):
            for step in steps:
                if str(step.get("status") or "pending").lower() == wanted:
                    out = dict(step)
                    out["plan_version"] = int(plan.get("version") or 0)
                    return out
        return None

    def update_plan_step(self, goal_id: int, *, plan_version: int, step_id: str,
                         status: str, note: str = "", last_output: str = "") -> dict | None:
        row = self.db.execute(
            """SELECT id, status, steps_json
               FROM goal_plans
               WHERE goal_id = ? AND version = ?
               LIMIT 1""",
            (int(goal_id), int(plan_version)),
        ).fetchone()
        if not row:
            return None

        plan_id = int(row[0])
        plan_status = str(row[1] or "active")
        steps = self._normalize_steps(self._safe_json_list(row[2]))
        now = self._now()
        target = None
        for step in steps:
            if str(step.get("id") or "") == str(step_id):
                target = step
                break
        if target is None and steps:
            target = steps[0]
        if target is None:
            return None

        new_status = str(status or "pending").strip().lower()
        if new_status not in _PLAN_STEP_STATUSES:
            new_status = "pending"
        target["status"] = new_status
        if note:
            target["note"] = str(note).strip()[:1000]
        if last_output:
            target["last_output"] = str(last_output).strip()[:2000]
        target["updated_at"] = now

        remaining = [s for s in steps if str(s.get("status") or "pending") in _PLAN_ACTIVE_STATUSES]
        new_plan_status = plan_status
        if not remaining:
            new_plan_status = "done"

        self.db.execute(
            """UPDATE goal_plans
               SET status = ?, steps_json = ?, updated_at = ?
               WHERE id = ?""",
            (new_plan_status, json.dumps(steps, ensure_ascii=False), now, plan_id),
        )
        self.db.commit()
        out = dict(target)
        out["plan_status"] = new_plan_status
        return out

    def add_attempt(self, goal_id: int, *, plan_version: int = 0,
                    step_id: str = "", step_title: str = "", action_query: str = "",
                    outcome: str = "unknown", progress_delta: float = 0.0,
                    summary: str = "", insight: str = "", error: str = "") -> dict:
        now = self._now()
        normalized_outcome = str(outcome or "unknown").strip().lower()[:20]
        cur = self.db.execute(
            """INSERT INTO goal_attempts
               (goal_id, plan_version, step_id, step_title, action_query,
                outcome, progress_delta, summary, insight, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(goal_id),
                int(plan_version or 0),
                str(step_id or "")[:40],
                str(step_title or "")[:200],
                str(action_query or "")[:2000],
                normalized_outcome,
                float(progress_delta or 0.0),
                str(summary or "")[:2000],
                str(insight or "")[:1200],
                str(error or "")[:2000],
                now,
            ),
        )
        self.db.commit()
        return {
            "id": int(cur.lastrowid),
            "goal_id": int(goal_id),
            "plan_version": int(plan_version or 0),
            "step_id": str(step_id or ""),
            "step_title": str(step_title or ""),
            "outcome": normalized_outcome,
            "progress_delta": float(progress_delta or 0.0),
            "summary": str(summary or ""),
            "insight": str(insight or ""),
            "error": str(error or ""),
            "created_at": now,
        }

    def get_recent_attempts(self, goal_id: int, limit: int = 8) -> list[dict]:
        lim = self._clamp_int(limit, 1, 200)
        rows = self.db.execute(
            """SELECT id, goal_id, plan_version, step_id, step_title, action_query,
                      outcome, progress_delta, summary, insight, error, created_at
               FROM goal_attempts
               WHERE goal_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (int(goal_id), lim),
        ).fetchall()
        out = []
        for row in rows:
            out.append({
                "id": int(row[0]),
                "goal_id": int(row[1]),
                "plan_version": int(row[2] or 0),
                "step_id": str(row[3] or ""),
                "step_title": str(row[4] or ""),
                "action_query": str(row[5] or ""),
                "outcome": str(row[6] or ""),
                "progress_delta": float(row[7] or 0.0),
                "summary": str(row[8] or ""),
                "insight": str(row[9] or ""),
                "error": str(row[10] or ""),
                "created_at": row[11],
            })
        return out

    def build_goal_report(self, goal_id: int, *, attempt_limit: int = 12) -> dict:
        goal = self.get_goal(goal_id)
        if not goal:
            return {}
        attempts = self.get_recent_attempts(goal_id, limit=max(1, min(int(attempt_limit or 12), 50)))
        events = self.get_goal_events(goal_id, limit=20)
        attempts_chrono = list(reversed(attempts))
        completed = [a for a in attempts_chrono if str(a.get("outcome") or "") == "done"]
        failed = [a for a in attempts_chrono if str(a.get("outcome") or "") in {"failed", "blocked", "error"}]
        progress = [a for a in attempts_chrono if float(a.get("progress_delta") or 0.0) > 0.0001]
        latest_attempt = attempts_chrono[-1] if attempts_chrono else {}
        recent_event = events[0] if events else {}
        next_actions: list[str] = []
        seen_actions: set[str] = set()
        for source in (latest_attempt, recent_event):
            if not isinstance(source, dict):
                continue
            for key in ("summary", "insight", "message"):
                value = str(source.get(key) or "").strip()
                if not value:
                    continue
                head = value.splitlines()[0].strip()[:220]
                if head and head not in seen_actions:
                    seen_actions.add(head)
                    next_actions.append(head)
                if len(next_actions) >= 3:
                    break
            if len(next_actions) >= 3:
                break

        outcome_counts = {
            "done": len([a for a in attempts_chrono if str(a.get("outcome") or "") == "done"]),
            "progress": len([a for a in attempts_chrono if str(a.get("outcome") or "") == "progress"]),
            "blocked": len([a for a in attempts_chrono if str(a.get("outcome") or "") == "blocked"]),
            "failed": len([a for a in attempts_chrono if str(a.get("outcome") or "") == "failed"]),
            "error": len([a for a in attempts_chrono if str(a.get("outcome") or "") == "error"]),
        }
        highlights: list[str] = []
        for item in completed[-3:]:
            summary = str(item.get("summary") or item.get("step_title") or "").strip()
            if summary:
                highlights.append(summary[:220])
        blockers: list[str] = []
        for item in failed[-3:]:
            note = str(item.get("error") or item.get("insight") or item.get("summary") or item.get("step_title") or "").strip()
            if note:
                blockers.append(note[:220])

        return {
            "goal_id": int(goal_id),
            "goal_type": str(goal.get("goal_type") or "generic"),
            "status": str(goal.get("status") or ""),
            "phase": str(goal.get("current_phase") or ""),
            "progress": float(goal.get("progress") or 0.0),
            "cycle_count": int(goal.get("cycle_count") or 0),
            "attempts_analyzed": len(attempts_chrono),
            "outcomes": outcome_counts,
            "highlights": highlights,
            "blockers": blockers,
            "recent_successes": len(completed),
            "recent_failures": len(failed),
            "recent_progress_cycles": len(progress),
            "last_summary": str(goal.get("last_result") or latest_attempt.get("summary") or "").strip()[:400],
            "next_recommended_actions": next_actions[:3],
            "updated_at": goal.get("updated_at"),
        }

    def render_goal_report_markdown(self, goal_id: int, *, attempt_limit: int = 15) -> str:
        goal = self.get_goal(goal_id)
        if not goal:
            return ""
        report = self.build_goal_report(goal_id, attempt_limit=attempt_limit)
        plan = self.get_active_plan(goal_id) or {}
        attempts = list(reversed(self.get_recent_attempts(goal_id, limit=attempt_limit)))
        cfg = goal.get("config") if isinstance(goal.get("config"), dict) else {}

        lines = [
            f"# {str(goal.get('title') or 'Goal').strip()}",
            "",
            f"- Goal type: `{str(goal.get('goal_type') or 'generic')}`",
            f"- Status: `{str(goal.get('status') or '')}`",
            f"- Phase: `{str(goal.get('current_phase') or '')}`",
            f"- Progress: `{round(float(goal.get('progress') or 0.0) * 100, 1)}%`",
            f"- Cycle count: `{int(goal.get('cycle_count') or 0)}`",
            f"- Updated: `{str(goal.get('updated_at') or '')}`",
        ]
        workspace = str(cfg.get("workspace") or "").strip()
        local_model = str(cfg.get("local_model") or "").strip()
        if workspace:
            lines.append(f"- Workspace: `{workspace}`")
        if local_model:
            lines.append(f"- Local model: `{local_model}`")
        stop_at = str(cfg.get("stop_at") or "").strip()
        if stop_at:
            lines.append(f"- Stop at: `{stop_at}`")
        lines.extend([
            "",
            "## Objective",
            "",
            str(goal.get("objective") or "").strip() or "_No objective recorded._",
            "",
            "## Session Summary",
            "",
            f"- Attempts analyzed: `{int(report.get('attempts_analyzed') or 0)}`",
            f"- Successes: `{int(report.get('recent_successes') or 0)}`",
            f"- Failures: `{int(report.get('recent_failures') or 0)}`",
            f"- Progress cycles: `{int(report.get('recent_progress_cycles') or 0)}`",
            "",
        ])
        last_summary = str(report.get("last_summary") or "").strip()
        if last_summary:
            lines.extend(["### Last Summary", "", last_summary, ""])
        highlights = list(report.get("highlights") or [])
        lines.extend(["### Highlights", ""])
        if highlights:
            lines.extend([f"- {str(item).strip()}" for item in highlights if str(item).strip()])
        else:
            lines.append("- None yet")
        blockers = list(report.get("blockers") or [])
        lines.extend(["", "### Blockers", ""])
        if blockers:
            lines.extend([f"- {str(item).strip()}" for item in blockers if str(item).strip()])
        else:
            lines.append("- None recorded")
        next_actions = list(report.get("next_recommended_actions") or [])
        lines.extend(["", "### Next Recommended Actions", ""])
        if next_actions:
            lines.extend([f"- {str(item).strip()}" for item in next_actions if str(item).strip()])
        else:
            lines.append("- None yet")
        if plan:
            lines.extend(["", "## Active Plan", ""])
            strategy = str(plan.get("strategy") or "").strip()
            if strategy:
                lines.extend([strategy, ""])
            steps = list(plan.get("steps") or [])
            if steps:
                for idx, step in enumerate(steps, start=1):
                    lines.append(
                        f"{idx}. `{str(step.get('status') or 'pending')}` {str(step.get('title') or '').strip()}"
                    )
                    action = str(step.get("action") or "").strip()
                    if action:
                        lines.append(f"   - Action: {action}")
                    criteria = str(step.get("success_criteria") or "").strip()
                    if criteria:
                        lines.append(f"   - Success: {criteria}")
            else:
                lines.append("_No plan steps recorded._")
        lines.extend(["", "## Recent Attempts", ""])
        if attempts:
            for item in attempts:
                outcome = str(item.get("outcome") or "unknown")
                title = str(item.get("step_title") or item.get("step_id") or "Goal attempt").strip()
                lines.append(f"- `{outcome}` {title}")
                summary = str(item.get("summary") or item.get("error") or item.get("action_query") or "").strip()
                if summary:
                    lines.append(f"  - {summary}")
                insight = str(item.get("insight") or "").strip()
                if insight:
                    lines.append(f"  - Insight: {insight}")
        else:
            lines.append("- No attempts recorded yet")
        lines.append("")
        return "\n".join(lines)

    # --- Runtime selection / state ---

    def get_due_goals(self, limit: int = 1) -> list[dict]:
        now = self._now()
        lim = self._clamp_int(limit, 1, 50)
        rows = self.db.execute(
            """SELECT * FROM goals
               WHERE status = 'active'
                 AND next_run_at IS NOT NULL
                 AND next_run_at <= ?
               ORDER BY priority ASC, next_run_at ASC, created_at ASC
               LIMIT ?""",
            (now, lim),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def claim_running(self, goal_id: int) -> bool:
        now = self._now()
        cur = self.db.execute(
            """UPDATE goals
               SET status = 'running',
                   cycle_count = cycle_count + 1,
                   last_run_at = ?,
                   updated_at = ?
               WHERE id = ? AND status = 'active'""",
            (now, now, int(goal_id)),
        )
        self.db.commit()
        return cur.rowcount > 0

    def get_running_goals(self) -> list[dict]:
        rows = self.db.execute(
            """SELECT * FROM goals
               WHERE status = 'running'
               ORDER BY priority ASC, updated_at DESC"""
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def recover_orphaned_running_goals(self, *, reason: str = "coordinator_restart") -> int:
        """Release stale DB-level running claims so the daemon can continue them."""
        now = self._now()
        rows = self.db.execute(
            """SELECT id, title
               FROM goals
               WHERE status = 'running'"""
        ).fetchall()
        if not rows:
            return 0
        self.db.execute(
            """UPDATE goals
               SET status = 'active',
                   updated_at = ?,
                   next_run_at = COALESCE(next_run_at, ?)
               WHERE status = 'running'""",
            (now, now),
        )
        for row in rows:
            goal_id = int(row[0] or 0)
            title = str(row[1] or f"goal-{goal_id}")
            self._add_event(
                goal_id,
                "recovered",
                f"Recovered running goal after {reason.replace('_', ' ')}",
                {"reason": str(reason or "coordinator_restart"), "title": title[:160]},
            )
        self.db.commit()
        return len(rows)

    def count_pending_goals(self) -> int:
        row = self.db.execute(
            "SELECT COUNT(*) FROM goals WHERE status = 'active'"
        ).fetchone()
        return int(row[0] if row else 0)

    def pause_goal(self, goal_id: int) -> dict | None:
        now = self._now()
        cur = self.db.execute(
            """UPDATE goals
               SET status = 'paused',
                   updated_at = ?,
                   next_run_at = NULL
               WHERE id = ? AND status IN ('active', 'running')""",
            (now, int(goal_id)),
        )
        if cur.rowcount <= 0:
            self.db.commit()
            return None
        self._add_event(int(goal_id), "paused", "Goal paused")
        self.db.commit()
        return self.get_goal(goal_id)

    def resume_goal(self, goal_id: int) -> dict | None:
        now = self._now()
        cur = self.db.execute(
            """UPDATE goals
               SET status = 'active',
                   updated_at = ?,
                   next_run_at = ?,
                   last_error = NULL
               WHERE id = ? AND status = 'paused'""",
            (now, now, int(goal_id)),
        )
        if cur.rowcount <= 0:
            self.db.commit()
            return None
        self._add_event(int(goal_id), "resumed", "Goal resumed")
        self.db.commit()
        return self.get_goal(goal_id)

    def cancel_goal(self, goal_id: int) -> dict | None:
        now = self._now()
        cur = self.db.execute(
            """UPDATE goals
               SET status = 'cancelled',
                   updated_at = ?,
                   next_run_at = NULL
               WHERE id = ? AND status IN ('active', 'running', 'paused', 'failed')""",
            (now, int(goal_id)),
        )
        if cur.rowcount <= 0:
            self.db.commit()
            return None
        self._add_event(int(goal_id), "cancelled", "Goal cancelled")
        self.db.commit()
        return self.get_goal(goal_id)

    def mark_cycle_result(self, goal_id: int, *,
                          progress_delta: float,
                          completed: bool,
                          phase: str,
                          summary: str,
                          next_action: str,
                          strategy: str | None = None,
                          allow_auto_complete: bool = True) -> dict | None:
        goal = self.get_goal(goal_id)
        if not goal:
            return None
        now_dt = datetime.now()
        now = now_dt.isoformat()
        cooldown = self._clamp_int(goal.get("cooldown_sec") or 90, 5, 86400)
        max_cycles = self._clamp_int(goal.get("max_cycles") or 0, 0, 10000)
        cycle_count = int(goal.get("cycle_count") or 0)

        raw_delta = max(0.0, float(progress_delta or 0.0))
        progress = self._clamp_progress(float(goal.get("progress") or 0.0) + raw_delta)
        completed_steps = int(goal.get("completed_steps") or 0) + 1
        target_steps = self._clamp_int(goal.get("target_steps") or 1, 1, 1000)

        made_progress = bool(completed) or raw_delta > 0.0001
        prev_stalled = int(goal.get("stalled_cycles") or 0)
        stalled_cycles = 0 if made_progress else (prev_stalled + 1)
        last_progress_at = now if made_progress else (goal.get("last_progress_at") or None)

        terminal = bool(completed) if allow_auto_complete else False
        if allow_auto_complete:
            terminal = terminal or progress >= 0.999 or completed_steps >= target_steps
        status = "completed" if terminal else "active"
        next_run_at = None if terminal else (now_dt + timedelta(seconds=cooldown)).isoformat()
        if (not terminal) and max_cycles > 0 and cycle_count >= max_cycles:
            status = "failed"
            next_run_at = None

        strategy_to_store = str(strategy if strategy is not None else (goal.get("strategy") or "")).strip()[:2000]

        self.db.execute(
            """UPDATE goals
               SET status = ?,
                   progress = ?,
                   completed_steps = ?,
                   current_phase = ?,
                   strategy = ?,
                   stalled_cycles = ?,
                   last_progress_at = ?,
                   last_result = ?,
                   last_error = NULL,
                   run_count = run_count + 1,
                   updated_at = ?,
                   next_run_at = ?
               WHERE id = ?""",
            (
                status,
                progress,
                completed_steps,
                str(phase or "working")[:120],
                strategy_to_store,
                int(stalled_cycles),
                last_progress_at,
                f"{str(summary or '').strip()[:2500]}\n{str(next_action or '').strip()[:1200]}".strip(),
                now,
                next_run_at,
                int(goal_id),
            ),
        )
        if status == "completed":
            self._add_event(int(goal_id), "completed", str(summary or "Goal completed"), {
                "progress": progress,
                "next_action": str(next_action or "")[:400],
            })
        elif status == "failed":
            self._add_event(int(goal_id), "failed", "Goal failed: max_cycles reached", {
                "cycle_count": cycle_count,
                "max_cycles": max_cycles,
            })
        else:
            self._add_event(int(goal_id), "cycle", str(summary or "Cycle done"), {
                "progress": progress,
                "phase": str(phase or "")[:120],
                "next_action": str(next_action or "")[:400],
                "stalled_cycles": int(stalled_cycles),
            })
        self.db.commit()
        return self.get_goal(goal_id)

    def complete_goal(self, goal_id: int, *, summary: str, phase: str = "completed") -> dict | None:
        goal = self.get_goal(goal_id)
        if not goal:
            return None
        now = self._now()
        self.db.execute(
            """UPDATE goals
               SET status = 'completed',
                   progress = 1.0,
                   current_phase = ?,
                   last_result = ?,
                   last_error = NULL,
                   updated_at = ?,
                   next_run_at = NULL
               WHERE id = ?""",
            (str(phase or "completed")[:120], str(summary or "")[:2500], now, int(goal_id)),
        )
        self._add_event(int(goal_id), "completed", str(summary or "Goal completed")[:1000], {})
        self.db.commit()
        return self.get_goal(goal_id)

    def mark_cycle_error(self, goal_id: int, error: str) -> dict | None:
        goal = self.get_goal(goal_id)
        if not goal:
            return None
        now_dt = datetime.now()
        now = now_dt.isoformat()
        cooldown = self._clamp_int(goal.get("cooldown_sec") or 90, 5, 86400)
        max_cycles = self._clamp_int(goal.get("max_cycles") or 0, 0, 10000)
        cycle_count = int(goal.get("cycle_count") or 0)
        status = "active"
        next_run_at = (now_dt + timedelta(seconds=cooldown)).isoformat()
        if max_cycles > 0 and cycle_count >= max_cycles:
            status = "failed"
            next_run_at = None
        stalled_cycles = int(goal.get("stalled_cycles") or 0) + 1
        self.db.execute(
            """UPDATE goals
               SET status = ?,
                   stalled_cycles = ?,
                   last_error = ?,
                   run_count = run_count + 1,
                   updated_at = ?,
                   next_run_at = ?
               WHERE id = ?""",
            (status, stalled_cycles, str(error or "")[:2000], now, next_run_at, int(goal_id)),
        )
        self._add_event(int(goal_id), "error", str(error or "")[:500], {
            "status_after": status,
            "next_run_at": next_run_at,
            "stalled_cycles": stalled_cycles,
        })
        self.db.commit()
        return self.get_goal(goal_id)

    def summary(self, *, user_id: str | None = None, limit: int = 8) -> dict:
        where = "WHERE user_id = ?" if user_id else ""
        params = (user_id,) if user_id else ()
        rows = self.db.execute(
            f"""SELECT status, COUNT(*) FROM goals {where}
                GROUP BY status""",
            params,
        ).fetchall()
        counts = {
            "active": 0,
            "running": 0,
            "paused": 0,
            "completed": 0,
            "cancelled": 0,
            "failed": 0,
        }
        for st, cnt in rows:
            key = str(st or "").lower()
            if key in counts:
                counts[key] = int(cnt or 0)

        goal_rows = self.db.execute(
            f"""SELECT * FROM goals {where}
                ORDER BY
                  CASE status
                    WHEN 'running' THEN 0
                    WHEN 'active' THEN 1
                    WHEN 'paused' THEN 2
                    WHEN 'failed' THEN 3
                    ELSE 4
                  END,
                  priority ASC,
                  updated_at DESC
                LIMIT ?""",
            (*params, self._clamp_int(limit, 1, 30)),
        ).fetchall()
        goals = [self._row_to_dict(r) for r in goal_rows]

        avg_row = self.db.execute(
            f"""SELECT AVG(progress)
                FROM goals {where} AND status IN ('active', 'running')"""
            if where else
            "SELECT AVG(progress) FROM goals WHERE status IN ('active', 'running')",
            params,
        ).fetchone()
        avg_progress = float(avg_row[0] or 0.0) if avg_row else 0.0

        return {
            "counts": counts,
            "avg_progress": round(avg_progress, 3),
            "goals": goals,
        }


class GoalCoordinatorDaemon:
    """Always-on worker that proactively plans, executes, reflects and replans."""

    def __init__(self, agent, goal_manager: GoalManager, config: dict | None = None):
        self.agent = agent
        self.goal_manager = goal_manager
        self.config = config or {}
        self._task: asyncio.Task | None = None
        self._running = False
        self._worker_id = f"goal-{uuid.uuid4().hex[:8]}"
        self._active: dict[int, dict] = {}
        self._last_pause_reason = ""
        self._last_pause_at = 0.0
        self._last_cycle_at = ""
        self._processed_total = 0
        self._failed_total = 0
        self._planned_total = 0
        self._replanned_total = 0

    def _enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def _interval_sec(self) -> float:
        try:
            raw = float(self.config.get("interval_sec", 2.0))
        except (TypeError, ValueError):
            raw = 2.0
        return max(0.5, min(raw, 120.0))

    def _batch_size(self) -> int:
        try:
            raw = int(self.config.get("batch_size", 1))
        except (TypeError, ValueError):
            raw = 1
        return max(1, min(raw, 8))

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

    def _plan_max_steps(self) -> int:
        try:
            raw = int(self.config.get("plan_max_steps", 6))
        except (TypeError, ValueError):
            raw = 6
        return max(2, min(raw, 12))

    def _replan_stall_cycles(self) -> int:
        try:
            raw = int(self.config.get("replan_stall_cycles", 2))
        except (TypeError, ValueError):
            raw = 2
        return max(1, min(raw, 20))

    def _attempt_history_limit(self) -> int:
        try:
            raw = int(self.config.get("attempt_history_limit", 6))
        except (TypeError, ValueError):
            raw = 6
        return max(2, min(raw, 20))

    def _is_high_load(self) -> tuple[bool, str]:
        if not self._auto_pause():
            return False, ""
        try:
            from .agent import LiteAgent
            active = len(LiteAgent.get_active_requests())
            queued = len(LiteAgent.get_queued_requests())
        except Exception:
            return False, ""
        if self._pause_active_threshold() > 0 and active >= self._pause_active_threshold():
            return True, f"active_requests={active}"
        if self._pause_queued_threshold() > 0 and queued >= self._pause_queued_threshold():
            return True, f"queued_requests={queued}"
        return False, ""

    @staticmethod
    def _extract_json(text: str) -> dict:
        raw = str(text or "").strip()
        if not raw:
            return {}
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _summarize_attempts(attempts: list[dict]) -> str:
        if not attempts:
            return "- no previous attempts"
        lines = []
        for item in attempts[:8]:
            outcome = str(item.get("outcome") or "unknown")
            step = str(item.get("step_title") or item.get("step_id") or "step")
            delta = float(item.get("progress_delta") or 0.0)
            summary = str(item.get("summary") or item.get("error") or "")[:220]
            lines.append(f"- {step}: {outcome}, progress_delta={delta:.3f}, note={summary}")
        return "\n".join(lines)

    @staticmethod
    def _goal_type(goal: dict) -> str:
        return str(goal.get("goal_type") or "generic").strip().lower() or "generic"

    def _goal_config(self, goal: dict) -> dict:
        raw = goal.get("config")
        if not isinstance(raw, dict):
            raw = GoalManager._safe_json_obj(goal.get("config_json"))
        if self._goal_type(goal) in {"autonomous_coding", "self_improvement"}:
            return normalize_session_config(raw, self.agent.config)
        return raw or {}

    def _goal_requested_model(self, goal: dict) -> str | None:
        if self._goal_type(goal) not in {"autonomous_coding", "self_improvement"}:
            return None
        cfg = self._goal_config(goal)
        model = str(cfg.get("local_model") or "").strip()
        return model or None

    def _goal_session_expired(self, goal: dict) -> bool:
        if self._goal_type(goal) not in {"autonomous_coding", "self_improvement"}:
            return False
        return night_coding_session_expired(self._goal_config(goal))

    def _negative_attempt_streak(self, goal_id: int, *, limit: int = 6) -> int:
        attempts = self.goal_manager.get_recent_attempts(goal_id, limit=max(1, min(int(limit or 6), 20)))
        streak = 0
        for item in attempts:
            outcome = str(item.get("outcome") or "").strip().lower()
            delta = float(item.get("progress_delta") or 0.0)
            if outcome in {"failed", "blocked", "error"} or (outcome == "progress" and delta <= 0.0):
                streak += 1
                continue
            break
        return streak

    def _fallback_plan(self, goal: dict, reason: str) -> dict:
        objective = str(goal.get("objective") or "").strip()[:240]
        return {
            "strategy": (
                f"Fallback strategy ({reason}): iterate with small verifiable steps, "
                "validate outputs, and pivot quickly on blockers."
            ),
            "steps": [
                {
                    "id": "s1",
                    "title": "Clarify immediate next action",
                    "action": f"Define one concrete executable action toward objective: {objective}",
                    "success_criteria": "Action is specific and testable",
                },
                {
                    "id": "s2",
                    "title": "Execute and validate",
                    "action": "Execute the action using available tools and check objective signal.",
                    "success_criteria": "Measured progress with evidence",
                },
                {
                    "id": "s3",
                    "title": "Adapt strategy",
                    "action": "If blocked, identify root cause and choose an alternative approach.",
                    "success_criteria": "Alternative route defined",
                },
            ],
        }

    def _parse_plan_output(self, text: str) -> dict:
        data = self._extract_json(text)
        if data:
            strategy = str(data.get("strategy") or data.get("approach") or data.get("plan_summary") or "").strip()
            steps_raw = data.get("steps")
            if not isinstance(steps_raw, list):
                steps_raw = data.get("plan")
            if not isinstance(steps_raw, list):
                steps_raw = data.get("milestones")
            if not isinstance(steps_raw, list):
                steps_raw = []
            steps = GoalManager._normalize_steps(steps_raw, max_steps=self._plan_max_steps())
            if strategy or steps:
                return {
                    "strategy": strategy[:2000],
                    "steps": steps,
                }
        return {"strategy": "", "steps": []}

    @staticmethod
    def _parse_execution_output(text: str) -> dict:
        data = GoalCoordinatorDaemon._extract_json(text)
        if data:
            outcome = str(data.get("outcome") or data.get("step_outcome") or "progress").strip().lower()
            aliases = {
                "ok": "done",
                "success": "done",
                "completed": "done",
                "progress": "progress",
                "blocked": "blocked",
                "fail": "failed",
                "failed": "failed",
                "error": "failed",
            }
            outcome = aliases.get(outcome, outcome)
            if outcome not in {"done", "progress", "blocked", "failed"}:
                outcome = "progress"

            completed = bool(data.get("completed", False))
            raw_delta = data.get("progress_delta", 0.08)
            try:
                parsed_delta = float(raw_delta)
            except (TypeError, ValueError):
                parsed_delta = 0.08
            progress_delta = max(0.0, min(parsed_delta, 0.45))
            if completed and progress_delta <= 0:
                progress_delta = 0.15
            return {
                "progress_delta": progress_delta,
                "completed": completed,
                "phase": str(data.get("phase", "working")).strip()[:120],
                "summary": str(data.get("summary", "")).strip()[:2500],
                "next_action": str(data.get("next_action", "")).strip()[:1200],
                "outcome": outcome,
                "insight": str(data.get("insight") or data.get("lesson") or "").strip()[:1200],
                "alternative": str(data.get("alternative") or data.get("next_try") or "").strip()[:1200],
            }

        s = str(text or "").strip()
        low = s.lower()
        completed = any(k in low for k in ("выполнено", "готово", "done", "completed", "finished"))
        blocked = any(k in low for k in ("заблок", "не удалось", "blocked", "cannot", "failed"))
        outcome = "done" if completed else ("blocked" if blocked else "progress")
        return {
            "progress_delta": 0.18 if completed else (0.0 if blocked else 0.06),
            "completed": completed,
            "phase": "completed" if completed else ("blocked" if blocked else "working"),
            "summary": s[:2500],
            "next_action": "" if completed else "Continue with next concrete sub-step.",
            "outcome": outcome,
            "insight": "",
            "alternative": "",
        }

    def _build_plan_prompt(self, goal: dict, attempts: list[dict], reason: str) -> str:
        if self._goal_type(goal) in {"autonomous_coding", "self_improvement"}:
            enriched = dict(goal)
            enriched["config"] = self._goal_config(goal)
            return build_night_coding_plan_prompt(enriched, self._summarize_attempts(attempts))
        title = str(goal.get("title") or "Goal").strip()
        objective = str(goal.get("objective") or "").strip()
        progress = float(goal.get("progress") or 0.0)
        phase = str(goal.get("current_phase") or "planned").strip()
        strategy = str(goal.get("strategy") or "").strip()
        last_error = str(goal.get("last_error") or "").strip()[:600]
        attempts_text = self._summarize_attempts(attempts)
        return (
            "You are a proactive Goal Planner. Build the best plan to achieve the objective.\n"
            "You must adapt strategy when previous attempts failed or stalled.\n\n"
            f"Goal title: {title}\n"
            f"Objective: {objective}\n"
            f"Current progress: {progress:.3f}\n"
            f"Current phase: {phase}\n"
            f"Current strategy: {strategy or '[none]'}\n"
            f"Planning trigger: {reason}\n"
            f"Last error: {last_error or '[none]'}\n\n"
            f"Recent attempts:\n{attempts_text}\n\n"
            "Return ONLY JSON:\n"
            '{"strategy":"...","steps":[{"id":"s1","title":"...","action":"...","success_criteria":"..."}]}\n'
            "Rules:\n"
            "- 3..8 steps max, concrete and executable\n"
            "- Prefer tool-usable actions and verifiable outputs\n"
            "- If blocked recently, provide a different approach\n"
            "- Do not include markdown, prose, or code fences"
        )

    def _build_execute_prompt(self, goal: dict, plan: dict, step: dict, attempts: list[dict]) -> str:
        if self._goal_type(goal) in {"autonomous_coding", "self_improvement"}:
            enriched = dict(goal)
            enriched["config"] = self._goal_config(goal)
            return build_night_coding_execute_prompt(enriched, plan, step, self._summarize_attempts(attempts))
        title = str(goal.get("title") or "Goal").strip()
        objective = str(goal.get("objective") or "").strip()
        strategy = str(plan.get("strategy") or goal.get("strategy") or "").strip()
        step_title = str(step.get("title") or "").strip()
        step_action = str(step.get("action") or step_title).strip()
        success_criteria = str(step.get("success_criteria") or "").strip()
        attempts_text = self._summarize_attempts(attempts)
        return (
            "You are a proactive Goal Executor.\n"
            "Execute the current step and judge outcome realistically.\n"
            "Use available tools when needed; do not claim success without evidence.\n\n"
            f"Goal title: {title}\n"
            f"Objective: {objective}\n"
            f"Strategy: {strategy or '[none]'}\n"
            f"Current step: {step_title}\n"
            f"Action: {step_action}\n"
            f"Success criteria: {success_criteria or '[none]'}\n\n"
            f"Recent attempts:\n{attempts_text}\n\n"
            "Return ONLY JSON:\n"
            '{"outcome":"done|progress|blocked|failed","progress_delta":0.08,'
            '"completed":false,"phase":"...","summary":"...",'
            '"next_action":"...","insight":"...","alternative":"..."}\n'
            "Rules:\n"
            "- outcome=done only when this step is actually finished\n"
            "- completed=true only when full objective is truly achieved\n"
            "- progress_delta must be 0..0.45\n"
            "- For blocked/failed include root-cause in insight and alternative approach"
        )

    def _broadcast(self, event_type: str, payload: dict):
        try:
            self.agent._ws_broadcast(event_type, payload)
        except Exception:
            pass

    async def _plan_goal(self, goal: dict, reason: str) -> dict:
        gid = int(goal.get("id") or 0)
        user_id = str(goal.get("user_id") or "default")
        attempts = self.goal_manager.get_recent_attempts(gid, limit=self._attempt_history_limit())
        prompt = self._build_plan_prompt(goal, attempts, reason)
        raw = await self.agent.run(prompt, user_id, requested_model=self._goal_requested_model(goal))
        parsed = self._parse_plan_output(raw)
        if not parsed.get("steps"):
            parsed = self._fallback_plan(goal, reason)
        plan = self.goal_manager.upsert_plan(
            gid,
            strategy=str(parsed.get("strategy") or "").strip(),
            steps=list(parsed.get("steps") or []),
            trigger=reason,
        )
        if not plan:
            raise RuntimeError("Failed to persist goal plan")
        self._planned_total += 1
        if reason not in {"initial", "missing_plan", "plan_exhausted"}:
            self._replanned_total += 1
        self._broadcast("goal_plan_updated", {
            "goal_id": gid,
            "user_id": user_id,
            "version": int(plan.get("version") or 0),
            "trigger": reason,
            "steps": len(plan.get("steps") or []),
            "strategy": str(plan.get("strategy") or "")[:300],
        })
        return plan

    async def _ensure_plan(self, goal: dict) -> dict:
        gid = int(goal.get("id") or 0)
        plan = self.goal_manager.get_active_plan(gid)
        if plan and self.goal_manager.get_next_plan_step(gid):
            return plan
        if not plan:
            return await self._plan_goal(goal, "missing_plan")
        return await self._plan_goal(goal, "plan_exhausted")

    async def process_once(self) -> dict:
        if not self._enabled():
            return {"status": "disabled", "processed": 0, "failed": 0}

        paused, reason = self._is_high_load()
        if paused:
            self._last_pause_reason = reason
            self._last_pause_at = datetime.now().timestamp()
            return {"status": "paused", "reason": reason, "processed": 0, "failed": 0}

        due = self.goal_manager.get_due_goals(limit=self._batch_size() * 3)
        if not due and not self._active:
            recovered = self.goal_manager.recover_orphaned_running_goals(reason="orphaned_cycle_recovery")
            if recovered:
                due = self.goal_manager.get_due_goals(limit=self._batch_size() * 3)
        if not due:
            return {"status": "idle", "processed": 0, "failed": 0}

        claimed: list[dict] = []
        for goal in due:
            if len(claimed) >= self._batch_size():
                break
            gid = int(goal["id"])
            if not self.goal_manager.claim_running(gid):
                continue
            fresh = self.goal_manager.get_goal(gid) or goal
            claimed.append(fresh)

        if not claimed:
            return {"status": "idle", "processed": 0, "failed": 0}

        processed = 0
        failed = 0
        for goal in claimed:
            gid = int(goal["id"])
            title = str(goal.get("title") or f"goal-{gid}")
            user_id = str(goal.get("user_id") or "default")
            if self._goal_session_expired(goal):
                stop_summary = (
                    "Autonomous self-improvement session stopped because its time window ended."
                    if self._goal_type(goal) == "self_improvement"
                    else "Autonomous coding session stopped because its time window ended."
                )
                self.goal_manager.complete_goal(
                    gid,
                    summary=stop_summary,
                    phase="window_complete",
                )
                self._broadcast("goal_completed", {
                    "goal_id": gid,
                    "title": title,
                    "user_id": user_id,
                    "status": "completed",
                    "progress": 1.0,
                    "phase": "window_complete",
                    "next_run_at": None,
                    "last_result": stop_summary,
                    "stalled_cycles": 0,
                    "plan_version": int(goal.get("plan_version") or 0),
                })
                processed += 1
                self._processed_total += 1
                continue
            self._active[gid] = {
                "goal_id": gid,
                "title": title,
                "user_id": user_id,
                "priority": int(goal.get("priority") or 5),
                "progress": float(goal.get("progress") or 0.0),
                "current_phase": str(goal.get("current_phase") or "planning"),
                "strategy": str(goal.get("strategy") or "")[:280],
                "stalled_cycles": int(goal.get("stalled_cycles") or 0),
                "started_at": datetime.now().isoformat(),
                "goal_type": self._goal_type(goal),
                "config": self._goal_config(goal),
            }
            self._broadcast("goal_started", self._active[gid])
            try:
                plan = await self._ensure_plan(goal)
                step = self.goal_manager.get_next_plan_step(gid)
                if not step:
                    plan = await self._plan_goal(goal, "plan_exhausted")
                    step = self.goal_manager.get_next_plan_step(gid)
                if not step:
                    raise RuntimeError("No executable plan step")

                step_id = str(step.get("id") or "")
                step_title = str(step.get("title") or step_id or "step")
                plan_version = int(step.get("plan_version") or plan.get("version") or 0)
                self.goal_manager.update_plan_step(
                    gid,
                    plan_version=plan_version,
                    step_id=step_id,
                    status="in_progress",
                    note="step started",
                )
                if gid in self._active:
                    self._active[gid]["current_phase"] = "executing"
                    self._active[gid]["step_id"] = step_id
                    self._active[gid]["step_title"] = step_title
                    self._active[gid]["plan_version"] = plan_version
                self._broadcast("goal_step_started", {
                    "goal_id": gid,
                    "user_id": user_id,
                    "plan_version": plan_version,
                    "step_id": step_id,
                    "step_title": step_title,
                })

                attempts = self.goal_manager.get_recent_attempts(gid, limit=self._attempt_history_limit())
                execute_prompt = self._build_execute_prompt(goal, plan, step, attempts)
                raw = await self.agent.run(
                    execute_prompt,
                    user_id,
                    requested_model=self._goal_requested_model(goal),
                )
                parsed = self._parse_execution_output(raw)

                outcome = str(parsed.get("outcome") or "progress")
                progress_delta = float(parsed.get("progress_delta") or 0.0)
                completed = bool(parsed.get("completed", False))
                phase = str(parsed.get("phase") or "working")
                summary = str(parsed.get("summary") or "")
                next_action = str(parsed.get("next_action") or "")
                insight = str(parsed.get("insight") or "")
                alternative = str(parsed.get("alternative") or "")

                allow_auto_complete = True
                if self._goal_type(goal) in {"autonomous_coding", "self_improvement"}:
                    cfg = self._goal_config(goal)
                    allow_auto_complete = not bool(cfg.get("continue_after_objective", True))
                    if not allow_auto_complete:
                        completed = False
                    if not next_action:
                        next_action = (
                            "Pick the next highest-value verified LiteAgent self-improvement backed by evidence."
                            if self._goal_type(goal) == "self_improvement"
                            else "Pick the next highest-value verified improvement in the same workspace."
                        )
                    if outcome == "done" and progress_delta <= 0:
                        progress_delta = 0.04

                step_status = "done"
                if outcome in {"blocked", "failed"}:
                    step_status = outcome
                elif outcome == "progress" and progress_delta <= 0 and not completed:
                    step_status = "in_progress"

                self.goal_manager.update_plan_step(
                    gid,
                    plan_version=plan_version,
                    step_id=step_id,
                    status=step_status,
                    note=summary or insight or alternative,
                    last_output=raw,
                )

                self.goal_manager.add_attempt(
                    gid,
                    plan_version=plan_version,
                    step_id=step_id,
                    step_title=step_title,
                    action_query=str(step.get("action") or step_title),
                    outcome=outcome,
                    progress_delta=progress_delta,
                    summary=summary,
                    insight=(insight or alternative),
                    error="",
                )

                updated = self.goal_manager.mark_cycle_result(
                    gid,
                    progress_delta=progress_delta,
                    completed=completed,
                    phase=phase,
                    summary=summary,
                    next_action=next_action or alternative,
                    strategy=str(plan.get("strategy") or ""),
                    allow_auto_complete=allow_auto_complete,
                )
                processed += 1
                self._processed_total += 1

                if updated and updated.get("status") == "active":
                    guard_triggered = False
                    if self._goal_type(goal) in {"autonomous_coding", "self_improvement"}:
                        cfg = self._goal_config(goal)
                        failed_limit = max(1, min(int(cfg.get("max_failed_cycles", 3) or 3), 8))
                        negative_streak = self._negative_attempt_streak(gid, limit=failed_limit + 1)
                        if negative_streak >= failed_limit:
                            paused = self.goal_manager.pause_goal(gid)
                            message = (
                                (
                                    "Self-improvement session paused by regression guard after "
                                    if self._goal_type(goal) == "self_improvement"
                                    else "Night coding session paused by regression guard after "
                                ) + f"{negative_streak} non-improving cycles."
                            )
                            self.goal_manager._add_event(gid, "guard_stop", message[:1000], {
                                "negative_streak": negative_streak,
                                "failed_limit": failed_limit,
                                "reason": "regression_guard",
                            })
                            self.goal_manager.db.commit()
                            guard_goal = paused or self.goal_manager.get_goal(gid) or updated
                            self._broadcast("goal_failed", {
                                "goal_id": gid,
                                "title": title,
                                "user_id": user_id,
                                "error": message,
                                "status": (guard_goal or {}).get("status", "paused"),
                            })
                            guard_triggered = True
                    stalled_cycles = int((self.goal_manager.get_goal(gid) or updated).get("stalled_cycles") or 0)
                    latest_active = self.goal_manager.get_goal(gid) or updated
                    if (not guard_triggered) and latest_active and latest_active.get("status") == "active":
                        if outcome in {"blocked", "failed"} or stalled_cycles >= self._replan_stall_cycles():
                            reason = "blocked" if outcome in {"blocked", "failed"} else "stalled"
                            replanned = await self._plan_goal(latest_active, reason)
                            nstep = self.goal_manager.get_next_plan_step(gid)
                            self._broadcast("goal_replanned", {
                                "goal_id": gid,
                                "user_id": user_id,
                                "reason": reason,
                                "version": int(replanned.get("version") or 0),
                                "next_step": (nstep or {}).get("title", ""),
                            })

                latest = self.goal_manager.get_goal(gid) or updated or goal
                if latest:
                    if gid in self._active:
                        self._active[gid]["progress"] = float(latest.get("progress") or 0.0)
                        self._active[gid]["current_phase"] = str(latest.get("current_phase") or "")
                        self._active[gid]["stalled_cycles"] = int(latest.get("stalled_cycles") or 0)
                        self._active[gid]["last_result"] = str(latest.get("last_result") or "")[:280]
                    evt = "goal_completed" if latest.get("status") == "completed" else "goal_updated"
                    self._broadcast(evt, {
                        "goal_id": gid,
                        "title": title,
                        "user_id": user_id,
                        "status": latest.get("status", ""),
                        "progress": float(latest.get("progress") or 0.0),
                        "phase": latest.get("current_phase", ""),
                        "next_run_at": latest.get("next_run_at"),
                        "last_result": str(latest.get("last_result") or "")[:600],
                        "stalled_cycles": int(latest.get("stalled_cycles") or 0),
                        "plan_version": int(latest.get("plan_version") or 0),
                    })
            except Exception as e:
                failed += 1
                self._failed_total += 1
                self.goal_manager.add_attempt(
                    gid,
                    plan_version=int((goal.get("plan_version") or 0)),
                    step_id="",
                    step_title="",
                    action_query="",
                    outcome="error",
                    progress_delta=0.0,
                    summary="",
                    insight="",
                    error=str(e),
                )
                updated = self.goal_manager.mark_cycle_error(gid, str(e))
                if updated and updated.get("status") == "active":
                    stalled_cycles = int(updated.get("stalled_cycles") or 0)
                    if stalled_cycles >= self._replan_stall_cycles():
                        with suppress(Exception):
                            replanned = await self._plan_goal(updated, "error_recovery")
                            self._broadcast("goal_replanned", {
                                "goal_id": gid,
                                "title": title,
                                "user_id": user_id,
                                "reason": "error_recovery",
                                "version": int(replanned.get("version") or 0),
                            })
                self._broadcast("goal_failed", {
                    "goal_id": gid,
                    "title": title,
                    "user_id": user_id,
                    "error": str(e)[:220],
                    "status": (updated or {}).get("status", "failed"),
                })
            finally:
                self._active.pop(gid, None)

        return {"status": "ok", "processed": processed, "failed": failed}

    async def _loop(self):
        interval = self._interval_sec()
        while self._running:
            self._last_cycle_at = datetime.now().isoformat()
            try:
                res = await self.process_once()
                if res.get("status") == "ok" and int(res.get("processed", 0)) > 0:
                    await asyncio.sleep(0)
                else:
                    await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Goal coordinator daemon error: %s", e)
                await asyncio.sleep(interval)

    async def start(self) -> dict:
        if not self._enabled():
            return {"status": "disabled"}
        if self._task and not self._task.done():
            return {"status": "already_running", "worker_id": self._worker_id}
        recovered = self.goal_manager.recover_orphaned_running_goals(reason="daemon_start")
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Goal coordinator daemon started (worker=%s)", self._worker_id)
        return {"status": "started", "worker_id": self._worker_id, "recovered": recovered}

    async def stop(self) -> dict:
        self._running = False
        task = self._task
        self._task = None
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        return {"status": "stopped"}

    def get_active_goals(self) -> list[dict]:
        return list(self._active.values())

    def state(self) -> dict:
        running = bool(self._task and not self._task.done())
        return {
            "enabled": self._enabled(),
            "running": running,
            "worker_id": self._worker_id,
            "active_count": len(self._active),
            "pending": self.goal_manager.count_pending_goals(),
            "last_pause_reason": self._last_pause_reason,
            "last_pause_at": self._last_pause_at,
            "last_cycle_at": self._last_cycle_at,
            "processed_total": self._processed_total,
            "failed_total": self._failed_total,
            "planned_total": self._planned_total,
            "replanned_total": self._replanned_total,
        }


def setup_goal_coordinator_daemon(agent, goal_manager: GoalManager,
                                  config: dict | None = None) -> GoalCoordinatorDaemon:
    """Create daemon from scheduler.goals config."""
    cfg = config or {}
    sched_cfg = cfg.get("scheduler", {}) if isinstance(cfg, dict) else {}
    goals_cfg = sched_cfg.get("goals", {}) if isinstance(sched_cfg, dict) else {}
    daemon = GoalCoordinatorDaemon(agent, goal_manager, goals_cfg)
    logger.info(
        "Goal coordinator configured (enabled=%s, interval=%.2fs, batch=%d)",
        daemon._enabled(), daemon._interval_sec(), daemon._batch_size(),
    )
    return daemon
