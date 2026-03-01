"""Asyncio-based scheduler with cron-like syntax. Zero external dependencies."""

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
        if kind == "exact" and actual != value:
            return False
        if kind == "every" and actual % value != 0:
            return False
        if kind == "set" and actual not in value:
            return False
        if kind == "range" and not (value[0] <= actual <= value[1]):
            return False
    return True


class Scheduler:
    """Lightweight asyncio scheduler."""

    def __init__(self):
        self._jobs: list[dict] = []
        self._running = False
        self._task: asyncio.Task | None = None

    def add_job(self, name: str, cron_expr: str,
                handler: Callable[[], Awaitable[None]]):
        """Register a scheduled job."""
        parsed = parse_cron(cron_expr)
        self._jobs.append({
            "name": name,
            "cron": parsed,
            "cron_expr": cron_expr,
            "handler": handler,
            "last_run": None,
        })
        logger.info("Scheduled job '%s' with cron '%s'", name, cron_expr)

    async def _loop(self):
        """Main scheduler loop. Checks every 30 seconds."""
        last_run_minute = -1
        while self._running:
            now = datetime.now()
            current_minute = now.hour * 60 + now.minute
            if current_minute != last_run_minute:
                last_run_minute = current_minute
                for job in self._jobs:
                    if cron_matches(job["cron"], now):
                        logger.info("Running scheduled job: %s", job["name"])
                        try:
                            await job["handler"]()
                            job["last_run"] = now.isoformat()
                        except Exception as e:
                            logger.error("Job '%s' failed: %s", job["name"], e)
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

    def get_jobs(self) -> list[dict]:
        """Return job list for API/dashboard."""
        return [{"name": j["name"], "cron": j["cron_expr"],
                 "last_run": j["last_run"]} for j in self._jobs]


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
            user_id = job_cfg.get("user_id", "scheduler")

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
            "dream_cycle", dream_cfg.get("cron", "0 3 * * *"), dream_job)

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
            replay_cfg.get("cron", "0 4 * * *"), replay_job)

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
            evolve_cfg.get("review_cron", "0 4 * * 0"), evolve_job)

    return scheduler
