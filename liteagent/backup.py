"""Backup and restore for LiteAgent data files."""

import logging
import os
import tarfile
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

LITEAGENT_DIR = Path.home() / ".liteagent"
BACKUP_DIR = LITEAGENT_DIR / "backups"

# Files to include (only those that exist are added).
_BACKUP_CANDIDATES = [
    LITEAGENT_DIR / "memory.db",
    LITEAGENT_DIR / "keys.json",
    LITEAGENT_DIR / "keys.enc",
    LITEAGENT_DIR / "auth_token",
]


def backup(config_path: str | None = None) -> Path:
    """Create a tar.gz backup of all LiteAgent data files.

    Performs a SQLite WAL checkpoint before archiving so the backup is
    self-contained (no ``-wal`` / ``-shm`` dependency).

    Returns the ``Path`` to the created archive.
    """
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"liteagent_backup_{timestamp}.tar.gz"

    # WAL checkpoint — flush pending writes into the main DB file
    db_path = LITEAGENT_DIR / "memory.db"
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.close()
        except Exception as exc:
            logger.warning("WAL checkpoint skipped: %s", exc)

    with tarfile.open(backup_path, "w:gz") as tar:
        for file_path in _BACKUP_CANDIDATES:
            if file_path.exists():
                tar.add(str(file_path), arcname=file_path.name)
        # Include config.json (may live outside ~/.liteagent)
        if config_path:
            cfg = Path(config_path)
            if cfg.exists():
                tar.add(str(cfg), arcname="config.json")

    size_kb = backup_path.stat().st_size / 1024
    logger.info("Backup created: %s (%.1f KB)", backup_path.name, size_kb)
    return backup_path


def restore(backup_path: str) -> None:
    """Restore data from a tar.gz backup archive.

    Validates member names to prevent path-traversal attacks.
    """
    path = Path(backup_path)
    if not path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    with tarfile.open(str(path), "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.startswith("/") or ".." in member.name:
                raise ValueError(f"Unsafe path in backup archive: {member.name}")
        tar.extractall(path=LITEAGENT_DIR)

    logger.info("Restored from backup: %s", path.name)


def prune_old_backups(keep: int = 7) -> int:
    """Delete backups beyond the *keep* most recent. Returns count deleted."""
    if not BACKUP_DIR.exists():
        return 0
    backups = sorted(BACKUP_DIR.glob("liteagent_backup_*.tar.gz"))
    to_delete = backups[:-keep] if len(backups) > keep else []
    for old in to_delete:
        old.unlink()
        logger.debug("Pruned old backup: %s", old.name)
    return len(to_delete)


def list_backups() -> list[dict]:
    """Return metadata for all available backups (newest first)."""
    if not BACKUP_DIR.exists():
        return []
    backups = sorted(BACKUP_DIR.glob("liteagent_backup_*.tar.gz"), reverse=True)
    result = []
    for b in backups:
        st = b.stat()
        result.append({
            "name": b.name,
            "path": str(b),
            "size_kb": round(st.st_size / 1024, 1),
            "created_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
        })
    return result
