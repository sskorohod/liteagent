"""Tests for status_reactions module."""

import asyncio
import pytest

from liteagent.status_reactions import (
    StatusReactionAdapter,
    StatusReactionController,
    StatusEmojis,
    StatusTiming,
    TypingLifecycle,
    NullStatusReactionAdapter,
    make_null_controller,
)


# ── Test adapter ──────────────────────────────────────────────────────────────

class RecordingAdapter(StatusReactionAdapter):
    """Adapter that records calls for test assertions."""

    def __init__(self):
        self.set_calls: list[str] = []
        self.remove_calls: list[str] = []

    async def set_reaction(self, emoji: str) -> None:
        self.set_calls.append(emoji)

    async def remove_reaction(self, emoji: str) -> None:
        self.remove_calls.append(emoji)


class FailingAdapter(StatusReactionAdapter):
    """Adapter that always raises to test error handling."""

    async def set_reaction(self, emoji: str) -> None:
        raise RuntimeError("reaction API failure")

    async def remove_reaction(self, emoji: str) -> None:
        raise RuntimeError("reaction API failure")


# ── NullStatusReactionAdapter ─────────────────────────────────────────────────

class TestNullAdapter:
    @pytest.mark.asyncio
    async def test_set_no_error(self):
        adapter = NullStatusReactionAdapter()
        await adapter.set_reaction("👀")  # should not raise

    @pytest.mark.asyncio
    async def test_remove_no_error(self):
        adapter = NullStatusReactionAdapter()
        await adapter.remove_reaction("👀")  # should not raise


# ── make_null_controller ──────────────────────────────────────────────────────

class TestMakeNullController:
    def test_returns_controller(self):
        ctrl = make_null_controller()
        assert isinstance(ctrl, StatusReactionController)

    @pytest.mark.asyncio
    async def test_full_lifecycle_no_error(self):
        ctrl = make_null_controller()
        await ctrl.set_queued()
        await ctrl.set_thinking()
        await ctrl.set_tool("web_search")
        await ctrl.set_done()
        await ctrl.clear()


# ── StatusReactionController ──────────────────────────────────────────────────

class TestStatusReactionController:
    def _make_ctrl(self, adapter=None, debounce_s=0.0):
        adapter = adapter or RecordingAdapter()
        timing = StatusTiming(debounce_s=debounce_s, stall_soft_s=999, stall_hard_s=9999)
        return StatusReactionController(adapter=adapter, timing=timing), adapter

    @pytest.mark.asyncio
    async def test_set_queued_sends_emoji(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_queued()
        assert "👀" in adapter.set_calls

    @pytest.mark.asyncio
    async def test_set_thinking_sends_emoji(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_thinking()
        ctrl._cancel_stall()  # prevent background task from running
        assert "🤔" in adapter.set_calls

    @pytest.mark.asyncio
    async def test_set_done_sends_emoji(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_done()
        assert "✅" in adapter.set_calls

    @pytest.mark.asyncio
    async def test_set_error_sends_emoji(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_error()
        assert "😱" in adapter.set_calls

    @pytest.mark.asyncio
    async def test_clear_calls_remove(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_queued()
        await ctrl.clear()
        assert "👀" in adapter.remove_calls

    @pytest.mark.asyncio
    async def test_clear_when_no_reaction_no_remove(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.clear()  # nothing set yet
        assert adapter.remove_calls == []

    @pytest.mark.asyncio
    async def test_set_tool_default_emoji(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_tool()
        ctrl._cancel_stall()
        assert "🔥" in adapter.set_calls

    @pytest.mark.asyncio
    async def test_set_tool_web_search(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_tool("web_search")
        ctrl._cancel_stall()
        assert "⚡" in adapter.set_calls

    @pytest.mark.asyncio
    async def test_set_tool_exec_command(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_tool("exec_command")
        ctrl._cancel_stall()
        assert "👨‍💻" in adapter.set_calls

    @pytest.mark.asyncio
    async def test_debounce_skips_rapid_same_emoji(self):
        """Same emoji within debounce window should not call adapter again."""
        ctrl, adapter = self._make_ctrl(debounce_s=10.0)
        await ctrl.set_queued()
        await ctrl.set_queued()  # should be debounced
        assert adapter.set_calls.count("👀") == 1

    @pytest.mark.asyncio
    async def test_closed_controller_ignores_updates(self):
        ctrl, adapter = self._make_ctrl()
        ctrl.close()
        await ctrl.set_queued()
        assert adapter.set_calls == []

    @pytest.mark.asyncio
    async def test_failing_adapter_no_crash(self):
        """Even if adapter raises, controller must not propagate exceptions."""
        adapter = FailingAdapter()
        timing = StatusTiming(debounce_s=0.0, stall_soft_s=999, stall_hard_s=9999)
        ctrl = StatusReactionController(adapter=adapter, timing=timing)
        await ctrl.set_queued()  # should not raise
        await ctrl.clear()       # should not raise

    @pytest.mark.asyncio
    async def test_restore_initial_resets_to_queued(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_done()
        await ctrl.restore_initial()
        assert adapter.set_calls[-1] == "👀"

    @pytest.mark.asyncio
    async def test_stall_cancelled_on_done(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_thinking()
        assert ctrl._stall_task is not None
        await ctrl.set_done()
        # task should be cancelled or None
        assert ctrl._stall_task is None or ctrl._stall_task.done()

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        ctrl, adapter = self._make_ctrl()
        await ctrl.set_queued()
        await ctrl.set_thinking()
        ctrl._cancel_stall()
        await ctrl.set_tool("web_search")
        ctrl._cancel_stall()
        await ctrl.set_done()
        await ctrl.clear()

        assert "👀" in adapter.set_calls
        assert "🤔" in adapter.set_calls
        assert "⚡" in adapter.set_calls
        assert "✅" in adapter.set_calls


# ── TypingLifecycle ────────────────────────────────────────────────────────────

class TestTypingLifecycle:
    @pytest.mark.asyncio
    async def test_start_calls_fn(self):
        calls = []

        async def start_fn():
            calls.append(1)

        lifecycle = TypingLifecycle(start_fn=start_fn, keepalive_s=0.01)
        await lifecycle.start()
        await asyncio.sleep(0.05)
        lifecycle.stop()

        assert len(calls) >= 1

    @pytest.mark.asyncio
    async def test_stop_terminates_loop(self):
        calls = []

        async def start_fn():
            calls.append(1)

        lifecycle = TypingLifecycle(start_fn=start_fn, keepalive_s=0.01)
        await lifecycle.start()
        await asyncio.sleep(0.05)
        count_before = len(calls)
        lifecycle.stop()
        await asyncio.sleep(0.05)
        count_after = len(calls)
        assert count_after - count_before <= 1  # may get one more call in-flight

    @pytest.mark.asyncio
    async def test_start_twice_no_duplicate(self):
        calls = []

        async def start_fn():
            calls.append(1)

        lifecycle = TypingLifecycle(start_fn=start_fn, keepalive_s=0.01)
        await lifecycle.start()
        await lifecycle.start()  # second call ignored
        await asyncio.sleep(0.05)
        lifecycle.stop()
        # Should have one background task, not two exponential growth
        assert len(calls) < 20

    @pytest.mark.asyncio
    async def test_consecutive_failures_stop_loop(self):
        fail_count = 0

        async def start_fn():
            nonlocal fail_count
            fail_count += 1
            raise RuntimeError("typing failed")

        lifecycle = TypingLifecycle(
            start_fn=start_fn,
            keepalive_s=0.01,
            max_consecutive_failures=2,
        )
        await lifecycle.start()
        await asyncio.sleep(0.1)
        lifecycle.stop()
        # Should stop after 2 consecutive failures
        assert fail_count <= 3

    @pytest.mark.asyncio
    async def test_ttl_stops_loop(self):
        calls = []

        async def start_fn():
            calls.append(1)

        lifecycle = TypingLifecycle(
            start_fn=start_fn,
            keepalive_s=0.01,
            max_duration_s=0.05,
        )
        await lifecycle.start()
        await asyncio.sleep(0.2)
        lifecycle.stop()
        # After TTL, loop should have exited
        count_at_stop = len(calls)
        await asyncio.sleep(0.1)
        assert len(calls) == count_at_stop


# ── StatusEmojis customisation ────────────────────────────────────────────────

class TestCustomEmojis:
    @pytest.mark.asyncio
    async def test_custom_emojis(self):
        adapter = RecordingAdapter()
        emojis = StatusEmojis(queued="⏳", thinking="💭", done="✔️", error="❌")
        timing = StatusTiming(debounce_s=0.0, stall_soft_s=999, stall_hard_s=9999)
        ctrl = StatusReactionController(adapter=adapter, emojis=emojis, timing=timing)

        await ctrl.set_queued()
        assert "⏳" in adapter.set_calls

        await ctrl.set_thinking()
        ctrl._cancel_stall()
        assert "💭" in adapter.set_calls

        await ctrl.set_done()
        assert "✔️" in adapter.set_calls
