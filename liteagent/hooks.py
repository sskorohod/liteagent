"""Lifecycle hook system — extensible event-driven architecture.

Inspired by OpenClaw's 30+ hook pattern. Handlers register at named
hook points with priority ordering (lower = runs first).

Hook points:
    agent_startup     — after __init__ + plugin load
    before_run        — start of _run_impl, after MCP load
    after_model_select — after complexity scoring / model selection
    before_api_call   — just before provider.complete()
    after_api_call    — after provider returns, before tool execution
    before_tool_call  — before tools.execute()
    after_tool_call   — after tool execution, with results
    before_response   — final text ready, before returning to user
    after_response    — after response delivered (background work)
    on_error          — when any stage catches an exception
    on_provider_switch — when circuit breaker triggers failover
    message_received  — when any channel receives a message
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class HookContext:
    """Mutable context passed through hook chains."""
    agent: Any = None
    user_id: str = ""
    user_input: str | list = ""
    model: str = ""
    system_prompt: Any = None
    tool_defs: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    response_text: str = ""
    tool_calls_log: list = field(default_factory=list)
    error: Exception | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class HookHandler:
    """A registered hook handler with priority and metadata."""
    name: str
    callback: Callable[[HookContext], Awaitable[None] | None]
    priority: int = 100
    plugin: str = ""


class HookRegistry:
    """Central registry for lifecycle hooks with priority ordering."""

    def __init__(self):
        self._hooks: dict[str, list[HookHandler]] = {}

    def register(self, hook_point: str, name: str,
                 callback: Callable[[HookContext], Awaitable[None] | None],
                 priority: int = 100, plugin: str = "") -> None:
        """Register a hook handler at a specific lifecycle point.

        Args:
            hook_point: Lifecycle event name (e.g. "after_response")
            name: Unique handler name (for unregister / dashboard)
            callback: Async or sync callable(HookContext)
            priority: Execution order — lower runs first (default 100)
            plugin: Source plugin name (empty for builtins)
        """
        handler = HookHandler(
            name=name, callback=callback,
            priority=priority, plugin=plugin)
        if hook_point not in self._hooks:
            self._hooks[hook_point] = []
        self._hooks[hook_point].append(handler)
        # Keep sorted by priority (stable sort preserves registration order for same priority)
        self._hooks[hook_point].sort(key=lambda h: h.priority)
        logger.debug("Hook registered: %s → %s (priority=%d, plugin=%s)",
                     hook_point, name, priority, plugin or "builtin")

    def unregister(self, hook_point: str, name: str) -> bool:
        """Remove a named handler from a hook point. Returns True if found."""
        handlers = self._hooks.get(hook_point, [])
        before = len(handlers)
        self._hooks[hook_point] = [h for h in handlers if h.name != name]
        removed = before - len(self._hooks[hook_point])
        if removed:
            logger.debug("Hook unregistered: %s → %s", hook_point, name)
        return removed > 0

    async def emit(self, hook_point: str, ctx: HookContext) -> HookContext:
        """Execute all handlers for a hook point in priority order.

        Each handler receives the (mutable) context and may modify it.
        Errors in individual handlers are caught and logged — they never
        crash the main flow.

        Returns:
            The (potentially modified) HookContext.
        """
        handlers = self._hooks.get(hook_point, [])
        for handler in handlers:
            try:
                result = handler.callback(ctx)
                if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                    await result
            except Exception as e:
                logger.warning("Hook %s/%s error: %s", hook_point, handler.name, e)
        return ctx

    def get_registered(self) -> dict[str, list[dict]]:
        """Return hook registry state for dashboard display."""
        out: dict[str, list[dict]] = {}
        for point, handlers in self._hooks.items():
            out[point] = [
                {"name": h.name, "priority": h.priority, "plugin": h.plugin}
                for h in handlers
            ]
        return out

    def clear(self, hook_point: str | None = None) -> None:
        """Clear all handlers, or handlers for a specific hook point."""
        if hook_point:
            self._hooks.pop(hook_point, None)
        else:
            self._hooks.clear()
