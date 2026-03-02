"""Asyncio-based scheduler with cron-like syntax. Zero external dependencies.

v2 additions: status tracking, retry-on-fail, max runtime, run_now().
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


def parse_cron(expr: str) -> dict:
    """Parse cron expression 'min hour day month weekday' into match spec.
    Supports: *, specific numbers, */N intervals, comma-separated values."""
    fields = expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Invalid cron expression (need 5 fields): {expr}")
    names = ["minute", "hour", "day", "month", "weekday"]
    parsed = {}
    for name, val in zip(names, fields):
        if val == "*":
            parsed[name] = None  # Match any
        elif val.startswith("*/"):
            parsed[name] = ("every", int(val[2:]))
        elif "," in val:
            parsed[name] = ("set", {int(v) for v in val.split(",")})
        elif "-" in val:
            lo, hi = val.split("-", 1)
            parsed[name] = ("range", (int(lo), int(hi)))
        else:
            parsed[name] = ("exact", int(val))
    return parsed


def cron_matches(parsed: dict, dt: datetime) -> bool:
    """Check if datetime matches parsed cron expression."""
    checks = {
        "minute": dt.minute,
        "hour": dt.hour,
        "day": dt.day,
        "month": dt.month,
        "weekday": dt.isoweekday() % 7,  # 0=Sun ... 6=Sat
    }
    for field, spec in parsed.items():
        if spec is None:
            continue
        kind, value = spec
        actual = checks[field]
        if kind == "exact":
            # Standard cron: weekday 7 == 0 (both mean Sunday)
            if field == "weekday" and value == 7:
                value = 0
            if actual != value:
                return False
        if kind == "every" and actual % value != 0:
            return False
        if kind == "set":
            # Normalize weekday 7 → 0 in set values
            match_set = value
            if field == "weekday" and 7 in value:
                match_set = (value - {7}) | {0}
            if actual not in match_set:
                return False
        if kind == "range" and not (value[0] <= actual <= value[1]):
            return False
    return True


class Scheduler:
    """Lightweight asyncio scheduler with retry, timeout, and status tracking."""

    def __init__(self):
        self._jobs: list[dict] = []
        self._running = False
        self._task: asyncio.Task | None = None
        # WebSocket hub reference (set by api.py at startup)
        self._ws_hub = None

    def add_job(self, name: str, cron_expr: str,
                handler: Callable[[], Awaitable[None]],
                max_runtime_sec: int | None = None,
                retry_on_fail: bool = False,
                retry_delay_sec: int = 60):
        """Register a scheduled job with optional timeout and retry."""
        parsed = parse_cron(cron_expr)
        self._jobs.append({
            "name": name,
            "cron": parsed,
            "cron_expr": cron_expr,
            "handler": handler,
            "last_run": None,
            # v2 fields
            "status": "idle",           # idle | running | failed | disabled
            "last_error": None,
            "run_count": 0,
            "fail_count": 0,
            "max_runtime_sec": max_runtime_sec,
            "retry_on_fail": retry_on_fail,
            "retry_delay_sec": retry_delay_sec,
            # compat
            "_running": False,
            "_run_started": None,
        })
        logger.info("Scheduled job '%s' with cron '%s'", name, cron_expr)

    async def _execute_job(self, job: dict, is_retry: bool = False):
        """Execute a single job with timeout and optional retry."""
        job["status"] = "running"
        job["_running"] = True
        job["_run_started"] = datetime.now().isoformat()
        self._ws_broadcast("scheduler_job_started", {"name": job["name"]})
        try:
            if job["max_runtime_sec"]:
                await asyncio.wait_for(
                    job["handler"](), timeout=job["max_runtime_sec"])
            else:
                await job["handler"]()
            job["last_run"] = datetime.now().isoformat()
            job["run_count"] += 1
            job["status"] = "idle"
            job["last_error"] = None
            self._ws_broadcast("scheduler_job_done", {
                "name": job["name"], "status": "success"})
        except asyncio.TimeoutError:
            job["status"] = "failed"
            job["fail_count"] += 1
            job["last_error"] = f"Timed out after {job['max_runtime_sec']}s"
            logger.error("Job '%s' timed out after %ds",
                         job["name"], job["max_runtime_sec"])
            self._ws_broadcast("scheduler_job_done", {
                "name": job["name"], "status": "timeout"})
        except Exception as e:
            job["status"] = "failed"
            job["fail_count"] += 1
            job["last_error"] = str(e)[:500]
            logger.error("Job '%s' failed: %s", job["name"], e)
            self._ws_broadcast("scheduler_job_done", {
                "name": job["name"], "status": "failed", "error": str(e)[:200]})
            # One retry attempt (not recursive beyond that)
            if job["retry_on_fail"] and not is_retry:
                logger.info("Retrying job '%s' in %ds…",
                            job["name"], job["retry_delay_sec"])
                await asyncio.sleep(job["retry_delay_sec"])
                await self._execute_job(job, is_retry=True)
        finally:
            job["_running"] = False
            job["_run_started"] = None

    async def _loop(self):
        """Main scheduler loop. Checks every 30 seconds."""
        last_run_minute = -1
        while self._running:
            now = datetime.now()
            current_minute = now.hour * 60 + now.minute
            if current_minute != last_run_minute:
                last_run_minute = current_minute
                for job in self._jobs:
                    if job.get("status") == "disabled":
                        continue
                    if job.get("_running"):
                        continue  # still running from last time
                    if cron_matches(job["cron"], now):
                        logger.info("Running scheduled job: %s", job["name"])
                        asyncio.create_task(self._execute_job(job))
            await asyncio.sleep(30)

    def start(self):
        """Start the scheduler as a background asyncio task."""
        if not self._jobs:
            logger.info("No jobs scheduled, scheduler not started")
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Scheduler started with %d jobs", len(self._jobs))

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            logger.info("Scheduler stopped")

    async def run_now(self, name: str) -> dict:
        """Trigger immediate execution of a named job."""
        for job in self._jobs:
            if job["name"] == name:
                if job["_running"]:
                    return {"error": f"Job '{name}' is already running"}
                asyncio.create_task(self._execute_job(job))
                return {"status": "triggered", "name": name}
        return {"error": f"Job '{name}' not found"}

    def get_jobs(self) -> list[dict]:
        """Return enriched job list for API/dashboard."""
        return [{
            "name": j["name"],
            "cron": j["cron_expr"],
            "last_run": j["last_run"],
            "status": j["status"],
            "last_error": j["last_error"],
            "run_count": j["run_count"],
            "fail_count": j["fail_count"],
            "running": j["_running"],
        } for j in self._jobs]

    def _ws_broadcast(self, event_type: str, data: dict):
        """Non-blocking broadcast to WebSocket hub (if connected)."""
        hub = self._ws_hub
        if hub:
            try:
                asyncio.get_event_loop().call_soon(
                    lambda: asyncio.ensure_future(hub.broadcast(event_type, data)))
            except RuntimeError:
                pass


def setup_scheduler(agent, config: dict) -> Scheduler | None:
    """Create and configure scheduler from config. Returns None if disabled."""
    sched_cfg = config.get("scheduler", {})
    if not sched_cfg.get("enabled", False):
        return None

    scheduler = Scheduler()
    jobs = sched_cfg.get("jobs", {})

    # Built-in: memory_prune
    if "memory_prune" in jobs:
        mem_cfg = config.get("memory", {})

        async def prune_job():
            agent.memory.prune_old_memories(
                days=mem_cfg.get("prune_days", 90),
                min_importance=mem_cfg.get("prune_min_importance", 0.3))
        scheduler.add_job("memory_prune", jobs["memory_prune"]["cron"], prune_job)

    # Built-in: daily_report
    if "daily_report" in jobs:
        async def report_job():
            summary = agent.memory.get_usage_summary(days=1)
            cost = agent.memory.get_today_cost()
            logger.info("📊 Daily report: cost=$%.4f, models=%s", cost, summary)
        scheduler.add_job("daily_report", jobs["daily_report"]["cron"], report_job)

    # Custom agent query jobs
    for name, job_cfg in jobs.items():
        if name in ("memory_prune", "daily_report"):
            continue
        if "query" in job_cfg:
            query = job_cfg["query"]
            user_id = job_cfg.get("user_id", f"scheduler-{name}")

            async def query_job(q=query, uid=user_id):
                logger.info("Scheduler running query: %s", q[:100])
                await agent.run(q, uid)
            scheduler.add_job(name, job_cfg["cron"], query_job)

    # ── Feature scheduler jobs ──────────────────
    features = config.get("features", {})

    # Dream Cycle
    dream_cfg = features.get("dream_cycle", {})
    if dream_cfg.get("enabled"):
        async def dream_job():
            from .metacognition import run_dream_cycle
            stats = await run_dream_cycle(
                agent.provider, agent.memory.db, agent.memory, dream_cfg)
            logger.info("Dream cycle complete: %s", stats)
        scheduler.add_job(
            "dream_cycle", dream_cfg.get("cron", "0 3 * * *"), dream_job,
            max_runtime_sec=300)

    # Counterfactual Replay
    replay_cfg = features.get("counterfactual_replay", {})
    if replay_cfg.get("enabled"):
        async def replay_job():
            from .metacognition import run_counterfactual_replay
            count = await run_counterfactual_replay(
                agent.provider, agent.memory.db, agent.memory, replay_cfg)
            logger.info("Counterfactual replay: %d lessons extracted", count)
        scheduler.add_job(
            "counterfactual_replay",
            replay_cfg.get("cron", "0 4 * * *"), replay_job,
            max_runtime_sec=300)

    # Self-Evolving Prompt review
    evolve_cfg = features.get("self_evolving_prompt", {})
    if evolve_cfg.get("enabled"):
        async def evolve_job():
            from .evolution import synthesize_prompt_patches
            patches = await synthesize_prompt_patches(
                agent.provider, agent.memory.db, evolve_cfg)
            if patches:
                logger.info("Prompt patches proposed: %s", patches)
                if evolve_cfg.get("auto_apply"):
                    for p in patches:
                        agent.memory.db.execute(
                            "UPDATE prompt_patches SET applied=1 "
                            "WHERE patch_text=?", (p,))
                    agent.memory.db.commit()
        scheduler.add_job(
            "self_evolving_prompt",
            evolve_cfg.get("review_cron", "0 4 * * 0"), evolve_job,
            max_runtime_sec=300)

    # Auto-backup
    backup_cfg = sched_cfg.get("auto_backup", {})
    if backup_cfg.get("enabled"):
        async def backup_job():
            from .backup import backup, prune_old_backups
            backup(config.get("_config_path"))
            prune_old_backups(keep=backup_cfg.get("keep", 7))
        scheduler.add_job(
            "auto_backup",
            backup_cfg.get("cron", "0 2 * * *"), backup_job,
            max_runtime_sec=120)

    # Session Reaper — clean up stale data
    reaper_cfg = sched_cfg.get("session_reaper", {})
    if reaper_cfg.get("enabled", True):
        async def reaper_job():
            import glob
            import tempfile
            import os as _os

            # 1. Prune old chat history (configurable retention)
            retention_days = reaper_cfg.get("chat_history_days", 30)
            try:
                agent.memory.db.execute(
                    "DELETE FROM chat_history WHERE created_at < datetime('now', ?)",
                    (f"-{retention_days} days",))
                agent.memory.db.commit()
            except Exception as e:
                logger.debug("Session reaper: chat_history cleanup error: %s", e)

            # 2. Clean up temp voice files older than 24h
            tmp = tempfile.gettempdir()
            for f in glob.glob(_os.path.join(tmp, "voice_*.ogg")):
                try:
                    import time
                    age_hours = (time.time() - _os.path.getmtime(f)) / 3600
                    if age_hours > 24:
                        _os.remove(f)
                except OSError:
                    pass

            # 3. Clean up old interaction_log entries
            log_days = reaper_cfg.get("interaction_log_days", 90)
            try:
                agent.memory.db.execute(
                    "DELETE FROM interaction_log WHERE created_at < datetime('now', ?)",
                    (f"-{log_days} days",))
                agent.memory.db.commit()
            except Exception as e:
                logger.debug("Session reaper: interaction_log cleanup error: %s", e)

            logger.info("Session reaper: cleanup complete "
                        "(chat_history>%dd, voice>24h, logs>%dd)",
                        retention_days, log_days)

        scheduler.add_job(
            "session_reaper",
            reaper_cfg.get("cron", "0 5 * * *"),
            reaper_job,
            max_runtime_sec=60)

    return scheduler
