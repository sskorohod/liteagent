"""Structured logging — JSON file + human-readable console."""

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

# Extra fields that get_structured loggers may attach via `extra={}`
_EXTRA_KEYS = ("user_id", "model", "duration_ms", "cost", "tool_name", "job_name", "action")


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects (JSON Lines)."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "message": record.getMessage(),
        }
        for key in _EXTRA_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info and record.exc_info[1]:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_structured_logging(config: dict) -> None:
    """Configure structured logging: console (human) + file (JSON, rotating).

    Replaces the basic ``logging.basicConfig()`` from ``config.py``.
    """
    log_cfg = config.get("logging", {})
    level_name = log_cfg.get("level", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)

    log_dir = Path.home() / ".liteagent"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "liteagent.log"

    root = logging.getLogger("liteagent")
    root.setLevel(level)

    # Remove any previously-attached handlers (from basicConfig, re-imports, etc.)
    root.handlers.clear()

    # ── Console handler (human-readable) ──
    console_fmt = log_cfg.get(
        "format", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(console_fmt))
    console.setLevel(level)
    root.addHandler(console)

    # ── File handler (JSON Lines, 10 MB × 5 rotations) ──
    max_bytes = log_cfg.get("max_bytes", 10 * 1024 * 1024)
    backup_count = log_cfg.get("backup_count", 5)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(logging.DEBUG)  # capture everything to file
    root.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def read_log_lines(limit: int = 50, level: str = "all",
                   search: str = "") -> list[dict]:
    """Read last *limit* log entries from ~/.liteagent/liteagent.log.

    Returns parsed JSON dicts, newest first.
    Filters by *level* (e.g. ``"error"``) and/or substring *search*.
    """
    log_file = Path.home() / ".liteagent" / "liteagent.log"
    if not log_file.exists():
        return []
    # Read all lines, take tail, reverse for newest-first
    try:
        lines = log_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    level_upper = level.upper() if level != "all" else None
    for raw in reversed(lines):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if level_upper and entry.get("level") != level_upper:
            continue
        if search and search.lower() not in raw.lower():
            continue
        entries.append(entry)
        if len(entries) >= limit:
            break
    return entries
