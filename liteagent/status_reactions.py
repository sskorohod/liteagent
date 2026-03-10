"""Channel-agnostic status reaction controller.

Ported from OpenClaw status-reactions.ts.

Provides a unified interface for displaying agent status via emoji reactions
(or similar channel-specific signals).  The controller tracks the current
lifecycle state and debounces rapid transitions.

Usage:
    from liteagent.status_reactions import StatusReactionController, StatusReactionAdapter

    class MyAdapter(StatusReactionAdapter):
        async def set_reaction(self, emoji: str) -> None:
            await bot.set_reaction(chat_id, message_id, emoji)

    ctrl = StatusReactionController(adapter=MyAdapter())
    await ctrl.set_thinking()
    ...
    await ctrl.set_tool("web_search")
    ...
    await ctrl.set_done()
    await asyncio.sleep(1.5)
    await ctrl.clear()
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Adapter protocol ──────────────────────────────────────────────────────────

class StatusReactionAdapter(ABC):
    """Abstract adapter — implement this for each channel."""

    @abstractmethod
    async def set_reaction(self, emoji: str) -> None:
        """Set / replace the current status emoji."""

    async def remove_reaction(self, emoji: str) -> None:  # noqa: B027
        """Remove a specific reaction (optional — needed for Discord-style)."""


# ── Emoji sets ────────────────────────────────────────────────────────────────

@dataclass
class StatusEmojis:
    queued: str = "👀"
    thinking: str = "🤔"
    tool: str = "🔥"
    coding: str = "👨‍💻"
    web: str = "⚡"
    done: str = "✅"
    error: str = "😱"
    stall_soft: str = "🥱"
    stall_hard: str = "😨"


@dataclass
class StatusTiming:
    """Timing configuration (all values in seconds)."""
    debounce_s: float = 0.7
    stall_soft_s: float = 10.0
    stall_hard_s: float = 30.0
    done_hold_s: float = 1.5
    error_hold_s: float = 2.5


_TOOL_EMOJI_MAP: dict[str, str] = {
    "web_search": "⚡",
    "search": "⚡",
    "browse": "⚡",
    "fetch": "⚡",
    "exec_command": "👨‍💻",
    "run_code": "👨‍💻",
    "write_file": "👨‍💻",
    "read_file": "👨‍💻",
    "edit_file": "👨‍💻",
    "bash": "👨‍💻",
    "python": "👨‍💻",
}


def _resolve_tool_emoji(tool_name: str | None, emojis: StatusEmojis) -> str:
    if tool_name:
        for key, emoji in _TOOL_EMOJI_MAP.items():
            if key in tool_name.lower():
                return emoji
    return emojis.tool


# ── Controller ────────────────────────────────────────────────────────────────

class StatusReactionController:
    """Manages emoji-based status signalling for a single message.

    Thread-safe via asyncio.Lock.
    """

    def __init__(
        self,
        adapter: StatusReactionAdapter,
        emojis: StatusEmojis | None = None,
        timing: StatusTiming | None = None,
    ) -> None:
        self._adapter = adapter
        self._emojis = emojis or StatusEmojis()
        self._timing = timing or StatusTiming()
        self._lock = asyncio.Lock()
        self._current_emoji: str | None = None
        self._stall_task: asyncio.Task | None = None
        self._last_update_at: float = 0.0
        self._closed = False

    # ── public API ─────────────────────────────────────────────────────────

    async def set_queued(self) -> None:
        await self._update(self._emojis.queued)

    async def set_thinking(self) -> None:
        await self._update(self._emojis.thinking)
        self._arm_stall_detector()

    async def set_tool(self, tool_name: str | None = None) -> None:
        emoji = _resolve_tool_emoji(tool_name, self._emojis)
        await self._update(emoji)
        self._arm_stall_detector()

    async def set_done(self) -> None:
        self._cancel_stall()
        await self._update(self._emojis.done)

    async def set_error(self) -> None:
        self._cancel_stall()
        await self._update(self._emojis.error)

    async def clear(self) -> None:
        """Remove the current reaction."""
        self._cancel_stall()
        async with self._lock:
            if self._current_emoji and not self._closed:
                try:
                    await self._adapter.remove_reaction(self._current_emoji)
                except Exception as exc:
                    logger.debug("remove_reaction failed: %s", exc)
            self._current_emoji = None

    async def restore_initial(self) -> None:
        """Reset to queued state (useful after error recovery)."""
        await self.set_queued()

    def close(self) -> None:
        """Mark controller as closed; prevent further reactions."""
        self._cancel_stall()
        self._closed = True

    # ── internal ───────────────────────────────────────────────────────────

    async def _update(self, emoji: str) -> None:
        if self._closed:
            return
        now = time.monotonic()
        elapsed = now - self._last_update_at
        # Debounce: skip rapid identical transitions
        if emoji == self._current_emoji and elapsed < self._timing.debounce_s:
            return
        async with self._lock:
            if self._closed:
                return
            try:
                await self._adapter.set_reaction(emoji)
                self._current_emoji = emoji
                self._last_update_at = time.monotonic()
            except Exception as exc:
                logger.debug("set_reaction(%s) failed: %s", emoji, exc)

    def _arm_stall_detector(self) -> None:
        """Start background task that escalates to stall emojis if silent too long."""
        self._cancel_stall()

        async def _stall_loop() -> None:
            try:
                await asyncio.sleep(self._timing.stall_soft_s)
                if not self._closed:
                    await self._update(self._emojis.stall_soft)
                await asyncio.sleep(
                    self._timing.stall_hard_s - self._timing.stall_soft_s)
                if not self._closed:
                    await self._update(self._emojis.stall_hard)
            except asyncio.CancelledError:
                pass

        self._stall_task = asyncio.create_task(_stall_loop())

    def _cancel_stall(self) -> None:
        if self._stall_task and not self._stall_task.done():
            self._stall_task.cancel()
        self._stall_task = None


# ── Typing lifecycle ──────────────────────────────────────────────────────────
# Ported from OpenClaw typing.ts / typing-lifecycle.ts

class TypingLifecycle:
    """Robust typing indicator with keepalive, TTL, and failure handling.

    Usage:
        async def start_typing():
            await bot.send_chat_action(chat_id, "typing")

        lifecycle = TypingLifecycle(start_fn=start_typing)
        await lifecycle.start()
        # ... do work ...
        lifecycle.stop()
    """

    def __init__(
        self,
        start_fn,                        # async () -> None
        stop_fn=None,                    # async () -> None | None
        keepalive_s: float = 3.0,
        max_duration_s: float = 60.0,
        max_consecutive_failures: int = 2,
    ) -> None:
        self._start_fn = start_fn
        self._stop_fn = stop_fn
        self._keepalive_s = keepalive_s
        self._max_duration_s = max_duration_s
        self._max_failures = max_consecutive_failures
        self._task: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        """Begin typing indicator with automatic keepalive."""
        if self._closed or self._task:
            return

        async def _loop() -> None:
            consecutive_failures = 0
            start_time = time.monotonic()
            while not self._closed:
                try:
                    await self._start_fn()
                    consecutive_failures = 0
                except Exception as exc:
                    consecutive_failures += 1
                    logger.debug("typing start failed (%d): %s",
                                 consecutive_failures, exc)
                    if consecutive_failures >= self._max_failures:
                        logger.debug("typing keepalive stopped after %d failures",
                                     consecutive_failures)
                        break

                elapsed = time.monotonic() - start_time
                if elapsed >= self._max_duration_s:
                    logger.debug("typing TTL exceeded (%.1fs)", elapsed)
                    break

                try:
                    await asyncio.sleep(self._keepalive_s)
                except asyncio.CancelledError:
                    break

        self._task = asyncio.create_task(_loop())

    def stop(self) -> None:
        """Stop the typing keepalive loop."""
        self._closed = True
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None


# ── Null adapter (for testing / channels without reaction support) ─────────────

class NullStatusReactionAdapter(StatusReactionAdapter):
    """No-op adapter for channels that don't support reactions."""

    async def set_reaction(self, emoji: str) -> None:
        pass

    async def remove_reaction(self, emoji: str) -> None:
        pass


def make_null_controller() -> StatusReactionController:
    return StatusReactionController(adapter=NullStatusReactionAdapter())
