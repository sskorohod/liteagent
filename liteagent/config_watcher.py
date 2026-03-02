"""Hot config reload — file watcher with mtime + SHA-256 verification."""

import asyncio
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Config paths that require a full restart (cannot be hot-reloaded).
NON_RELOADABLE: set[tuple[str, ...]] = {
    ("agent", "provider"),
    ("channels", "api", "port"),
    ("channels", "api", "host"),
    ("channels", "telegram", "token"),
}


# ── Config diff ───────────────────────────────────────────


def config_diff(old: dict, new: dict, prefix: tuple = ()) -> list[dict]:
    """Recursively diff two config dicts.

    Returns ``[{"path": "a.b.c", "old": ..., "new": ..., "reloadable": bool}]``.
    """
    changes: list[dict] = []
    all_keys = set(list(old.keys()) + list(new.keys()))
    for key in sorted(all_keys):
        if key.startswith("_"):
            continue  # skip internal keys
        path = prefix + (key,)
        old_val = old.get(key)
        new_val = new.get(key)
        if old_val == new_val:
            continue
        if isinstance(old_val, dict) and isinstance(new_val, dict):
            changes.extend(config_diff(old_val, new_val, path))
        else:
            changes.append({
                "path": ".".join(path),
                "old": old_val,
                "new": new_val,
                "reloadable": path not in NON_RELOADABLE,
            })
    return changes


# ── File helpers ──────────────────────────────────────────


def _file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


# ── Watcher ───────────────────────────────────────────────


class ConfigWatcher:
    """Watch config file for changes and apply hot-reload.

    Parameters
    ----------
    config_path : str
        Absolute path to ``config.json``.
    agent : LiteAgent
        The running agent instance (has ``apply_config_update``).
    scheduler : Scheduler | None
        Optional scheduler to reload jobs.
    on_reload : async callback(changes) | None
        Called after successful reload (e.g., to broadcast WS event).
    check_interval : float
        Seconds between mtime checks (default 5).
    """

    def __init__(self, config_path: str, agent, scheduler=None,
                 on_reload=None, check_interval: float = 5.0):
        self._path = Path(config_path)
        self._agent = agent
        self._scheduler = scheduler
        self._on_reload = on_reload
        self._interval = check_interval
        self._last_hash = _file_hash(self._path)
        self._last_mtime = _file_mtime(self._path)
        self._task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._watch_loop())
            logger.info("Config watcher started (checking every %.0fs)", self._interval)

    def stop(self):
        if self._task:
            self._task.cancel()
            self._task = None

    # ── Watch loop ────────────────────────────────────────

    async def _watch_loop(self):
        while True:
            await asyncio.sleep(self._interval)
            try:
                if not self._path.exists():
                    continue
                mtime = _file_mtime(self._path)
                if mtime == self._last_mtime:
                    continue
                self._last_mtime = mtime
                new_hash = _file_hash(self._path)
                if new_hash == self._last_hash:
                    continue
                self._last_hash = new_hash
                await self._reload()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Config watch error: %s", exc)

    # ── Reload ────────────────────────────────────────────

    async def _reload(self):
        logger.info("Config file changed, reloading…")
        from .config import load_config
        try:
            new_config = load_config(str(self._path))
        except Exception as exc:
            logger.error("Failed to parse updated config: %s", exc)
            return

        changes = config_diff(self._agent.config, new_config)
        if not changes:
            logger.debug("Config hash changed but no effective diff")
            return

        non_reloadable = [c for c in changes if not c["reloadable"]]
        for c in non_reloadable:
            logger.warning("Non-reloadable config changed: %s (restart required)", c["path"])

        reloadable = [c for c in changes if c["reloadable"]]
        if reloadable:
            self._agent.apply_config_update(new_config)
            logger.info("Hot-reloaded %d config fields: %s",
                        len(reloadable), ", ".join(c["path"] for c in reloadable))

        if self._on_reload:
            try:
                await self._on_reload(changes)
            except Exception as exc:
                logger.debug("on_reload callback error: %s", exc)

    async def force_reload(self):
        """Manually trigger reload (e.g., from API endpoint or SIGHUP)."""
        self._last_hash = ""
        self._last_mtime = 0.0
        await self._reload()
