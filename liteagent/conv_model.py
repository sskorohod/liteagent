"""Per-conversation model overrides (from OpenClaw model-overrides.ts).

Each user/chat can override the default model.  Stored in memory.db.
Persists across restarts.

Usage (in agent):
    from liteagent.conv_model import ConversationModelStore
    store = ConversationModelStore(db)
    store.set(user_id, "claude-opus-4-20250115")
    model = store.get(user_id)           # -> "claude-opus-4-20250115" or None

Telegram command: /model <name>  saves override for that user.
Dashboard: shows/edits per-user overrides via /api/conv-models.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conv_model_overrides (
    user_id  TEXT PRIMARY KEY,
    model    TEXT NOT NULL,
    provider TEXT DEFAULT '',
    set_at   TEXT NOT NULL,
    set_via  TEXT DEFAULT 'command'
);
"""


class ConversationModelStore:
    """SQLite-backed per-user model override store."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        db.execute(_SCHEMA)
        db.commit()

    # ── read ─────────────────────────────────────────────────────────────

    def get(self, user_id: str) -> str | None:
        """Return the model override for *user_id*, or None if not set."""
        row = self._db.execute(
            "SELECT model FROM conv_model_overrides WHERE user_id = ?",
            (user_id,)).fetchone()
        return row[0] if row else None

    def get_with_meta(self, user_id: str) -> dict | None:
        """Return override dict {model, provider, set_at, set_via} or None."""
        row = self._db.execute(
            "SELECT model, provider, set_at, set_via "
            "FROM conv_model_overrides WHERE user_id = ?",
            (user_id,)).fetchone()
        if not row:
            return None
        return {"model": row[0], "provider": row[1], "set_at": row[2], "set_via": row[3]}

    def list_all(self) -> list[dict]:
        """Return all active overrides."""
        rows = self._db.execute(
            "SELECT user_id, model, provider, set_at, set_via "
            "FROM conv_model_overrides ORDER BY set_at DESC").fetchall()
        return [{"user_id": r[0], "model": r[1], "provider": r[2],
                 "set_at": r[3], "set_via": r[4]} for r in rows]

    # ── write ─────────────────────────────────────────────────────────────

    def set(self, user_id: str, model: str,
            provider: str = "", set_via: str = "command") -> None:
        """Save or update model override for *user_id*."""
        now = datetime.now().isoformat()
        self._db.execute(
            "INSERT INTO conv_model_overrides (user_id, model, provider, set_at, set_via) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "model=excluded.model, provider=excluded.provider, "
            "set_at=excluded.set_at, set_via=excluded.set_via",
            (user_id, model, provider, now, set_via))
        self._db.commit()
        logger.info("Model override set: user=%s model=%s via=%s", user_id, model, set_via)

    def clear(self, user_id: str) -> bool:
        """Remove override for *user_id*. Returns True if it existed."""
        cur = self._db.execute(
            "DELETE FROM conv_model_overrides WHERE user_id = ?",
            (user_id,))
        self._db.commit()
        return cur.rowcount > 0

    def clear_all(self) -> int:
        """Remove all overrides. Returns count deleted."""
        cur = self._db.execute("DELETE FROM conv_model_overrides")
        self._db.commit()
        return cur.rowcount


# ── helpers for agent ──────────────────────────────────────────────────────

def resolve_model_for_user(store: ConversationModelStore | None,
                            user_id: str,
                            default_model: str) -> str:
    """Return per-user override model, or default_model if not set."""
    if store is None:
        return default_model
    override = store.get(user_id)
    return override if override else default_model


def parse_model_command(text: str) -> tuple[str | None, str]:
    """Parse /model command text.

    Returns (model_name, error_msg).
    - "/model" → (None, "show current")
    - "/model claude-opus-4" → ("claude-opus-4-20250115", "")
    - "/model reset" → ("__reset__", "")
    """
    parts = text.strip().split(None, 1)
    if len(parts) < 2:
        return None, "show"

    arg = parts[1].strip()
    if arg.lower() in ("reset", "clear", "default", "off"):
        return "__reset__", ""

    # Expand common short names
    _SHORTCUTS: dict[str, str] = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-20250514",
        "opus": "claude-opus-4-20250115",
        "gpt4": "gpt-4o",
        "gpt4mini": "gpt-4o-mini",
        "gpt4.1": "gpt-4.1",
        "flash": "gemini-2.5-flash",
        "pro": "gemini-2.5-pro",
    }
    expanded = _SHORTCUTS.get(arg.lower(), arg)
    return expanded, ""
