"""Tests for lifecycle hook system (hooks.py) and plugin loader (plugins.py)."""

import asyncio
import pytest
from liteagent.hooks import HookRegistry, HookContext, HookHandler


# ══════════════════════════════════════════
# HOOK REGISTRY TESTS
# ══════════════════════════════════════════

class TestHookRegistry:
    def setup_method(self):
        self.registry = HookRegistry()

    def test_register_basic(self):
        async def handler(ctx): pass
        self.registry.register("after_response", "test_handler", handler)
        registered = self.registry.get_registered()
        assert "after_response" in registered
        assert len(registered["after_response"]) == 1
        assert registered["after_response"][0]["name"] == "test_handler"

    def test_register_with_priority(self):
        async def h1(ctx): pass
        async def h2(ctx): pass
        async def h3(ctx): pass
        self.registry.register("after_response", "high", h1, priority=10)
        self.registry.register("after_response", "low", h2, priority=200)
        self.registry.register("after_response", "medium", h3, priority=100)
        handlers = self.registry.get_registered()["after_response"]
        assert [h["name"] for h in handlers] == ["high", "medium", "low"]

    def test_register_with_plugin_name(self):
        async def handler(ctx): pass
        self.registry.register("before_run", "my_handler", handler,
                               plugin="my_plugin")
        info = self.registry.get_registered()["before_run"][0]
        assert info["plugin"] == "my_plugin"

    def test_unregister(self):
        async def handler(ctx): pass
        self.registry.register("after_response", "to_remove", handler)
        assert self.registry.unregister("after_response", "to_remove") is True
        assert len(self.registry.get_registered().get("after_response", [])) == 0

    def test_unregister_nonexistent(self):
        assert self.registry.unregister("after_response", "nonexistent") is False

    def test_clear_specific_hook(self):
        async def handler(ctx): pass
        self.registry.register("after_response", "h1", handler)
        self.registry.register("before_run", "h2", handler)
        self.registry.clear("after_response")
        assert "after_response" not in self.registry.get_registered()
        assert "before_run" in self.registry.get_registered()

    def test_clear_all(self):
        async def handler(ctx): pass
        self.registry.register("after_response", "h1", handler)
        self.registry.register("before_run", "h2", handler)
        self.registry.clear()
        assert self.registry.get_registered() == {}

    def test_multiple_hooks_same_point(self):
        async def h1(ctx): pass
        async def h2(ctx): pass
        self.registry.register("after_response", "h1", h1)
        self.registry.register("after_response", "h2", h2)
        assert len(self.registry.get_registered()["after_response"]) == 2


# ══════════════════════════════════════════
# ASYNC EMIT TESTS
# ══════════════════════════════════════════

class TestHookEmit:
    def setup_method(self):
        self.registry = HookRegistry()

    @pytest.mark.asyncio
    async def test_emit_basic(self):
        calls = []
        async def handler(ctx):
            calls.append("called")
        self.registry.register("after_response", "basic", handler)
        ctx = HookContext(response_text="hello")
        await self.registry.emit("after_response", ctx)
        assert calls == ["called"]

    @pytest.mark.asyncio
    async def test_emit_modifies_context(self):
        async def handler(ctx):
            ctx.response_text = ctx.response_text.upper()
        self.registry.register("after_response", "upper", handler)
        ctx = HookContext(response_text="hello")
        result = await self.registry.emit("after_response", ctx)
        assert result.response_text == "HELLO"

    @pytest.mark.asyncio
    async def test_emit_priority_order(self):
        order = []
        async def h1(ctx): order.append("first")
        async def h2(ctx): order.append("second")
        async def h3(ctx): order.append("third")
        self.registry.register("test", "third", h3, priority=300)
        self.registry.register("test", "first", h1, priority=10)
        self.registry.register("test", "second", h2, priority=100)
        await self.registry.emit("test", HookContext())
        assert order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_emit_error_isolation(self):
        """Errors in one handler should not prevent others from running."""
        calls = []
        async def failing_handler(ctx):
            raise ValueError("boom")
        async def good_handler(ctx):
            calls.append("good")
        self.registry.register("test", "bad", failing_handler, priority=10)
        self.registry.register("test", "good", good_handler, priority=20)
        ctx = HookContext()
        await self.registry.emit("test", ctx)
        assert calls == ["good"]

    @pytest.mark.asyncio
    async def test_emit_nonexistent_hook_point(self):
        ctx = HookContext(response_text="unchanged")
        result = await self.registry.emit("nonexistent_hook", ctx)
        assert result.response_text == "unchanged"

    @pytest.mark.asyncio
    async def test_emit_sync_handler(self):
        """Sync handlers should also work."""
        calls = []
        def sync_handler(ctx):
            calls.append("sync")
        self.registry.register("test", "sync", sync_handler)
        await self.registry.emit("test", HookContext())
        assert calls == ["sync"]

    @pytest.mark.asyncio
    async def test_emit_chained_mutations(self):
        """Multiple handlers mutating the same context field."""
        async def add_prefix(ctx):
            ctx.response_text = "[PREFIX] " + ctx.response_text
        async def add_suffix(ctx):
            ctx.response_text = ctx.response_text + " [SUFFIX]"
        self.registry.register("test", "prefix", add_prefix, priority=10)
        self.registry.register("test", "suffix", add_suffix, priority=20)
        ctx = HookContext(response_text="body")
        result = await self.registry.emit("test", ctx)
        assert result.response_text == "[PREFIX] body [SUFFIX]"

    @pytest.mark.asyncio
    async def test_emit_extra_dict(self):
        """Handlers can pass data to each other via extra dict."""
        async def producer(ctx):
            ctx.extra["computed"] = 42
        async def consumer(ctx):
            ctx.response_text = str(ctx.extra.get("computed", 0))
        self.registry.register("test", "producer", producer, priority=10)
        self.registry.register("test", "consumer", consumer, priority=20)
        ctx = HookContext(response_text="")
        result = await self.registry.emit("test", ctx)
        assert result.response_text == "42"


# ══════════════════════════════════════════
# HOOK CONTEXT TESTS
# ══════════════════════════════════════════

class TestHookContext:
    def test_default_values(self):
        ctx = HookContext()
        assert ctx.agent is None
        assert ctx.user_id == ""
        assert ctx.model == ""
        assert ctx.response_text == ""
        assert ctx.tool_calls_log == []
        assert ctx.error is None
        assert ctx.extra == {}

    def test_init_with_values(self):
        ctx = HookContext(user_id="u1", model="claude-3", response_text="hi")
        assert ctx.user_id == "u1"
        assert ctx.model == "claude-3"
        assert ctx.response_text == "hi"

    def test_extra_independent(self):
        """Each HookContext instance should have its own extra dict."""
        ctx1 = HookContext()
        ctx2 = HookContext()
        ctx1.extra["key"] = "val"
        assert "key" not in ctx2.extra


# ══════════════════════════════════════════
# PLUGIN LOADER TESTS
# ══════════════════════════════════════════

class TestPluginLoader:
    def test_load_plugins_no_dir(self, tmp_path, monkeypatch):
        """Should return empty list if plugins dir doesn't exist."""
        import liteagent.plugins as plugins_mod
        monkeypatch.setattr(plugins_mod, "PLUGINS_DIR", tmp_path / "nonexistent")
        registry = HookRegistry()
        loaded = plugins_mod.load_plugins(registry, {})
        assert loaded == []

    def test_load_plugins_empty_dir(self, tmp_path, monkeypatch):
        """Should return empty list if plugins dir is empty."""
        import liteagent.plugins as plugins_mod
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        monkeypatch.setattr(plugins_mod, "PLUGINS_DIR", plugins_dir)
        registry = HookRegistry()
        loaded = plugins_mod.load_plugins(registry, {})
        assert loaded == []

    def test_load_valid_plugin(self, tmp_path, monkeypatch):
        """Should load a plugin with register() function."""
        import liteagent.plugins as plugins_mod
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        plugin_file = plugins_dir / "test_plugin.py"
        plugin_file.write_text(
            "def register(hooks, config):\n"
            "    hooks.register('after_response', 'test_from_plugin', lambda ctx: None, plugin='test_plugin')\n"
        )
        monkeypatch.setattr(plugins_mod, "PLUGINS_DIR", plugins_dir)
        registry = HookRegistry()
        loaded = plugins_mod.load_plugins(registry, {})
        assert loaded == ["test_plugin"]
        assert "after_response" in registry.get_registered()

    def test_skip_underscore_files(self, tmp_path, monkeypatch):
        """Should skip files starting with underscore."""
        import liteagent.plugins as plugins_mod
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "_internal.py").write_text("def register(hooks, config): pass\n")
        monkeypatch.setattr(plugins_mod, "PLUGINS_DIR", plugins_dir)
        registry = HookRegistry()
        loaded = plugins_mod.load_plugins(registry, {})
        assert loaded == []

    def test_skip_plugin_without_register(self, tmp_path, monkeypatch):
        """Should skip plugins missing register() function."""
        import liteagent.plugins as plugins_mod
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "no_register.py").write_text("x = 1\n")
        monkeypatch.setattr(plugins_mod, "PLUGINS_DIR", plugins_dir)
        registry = HookRegistry()
        loaded = plugins_mod.load_plugins(registry, {})
        assert loaded == []

    def test_broken_plugin_doesnt_crash(self, tmp_path, monkeypatch):
        """A plugin with syntax errors should not crash the loader."""
        import liteagent.plugins as plugins_mod
        plugins_dir = tmp_path / "plugins"
        plugins_dir.mkdir()
        (plugins_dir / "broken.py").write_text("def register(hooks, config):\n    raise RuntimeError('oops')\n")
        (plugins_dir / "good.py").write_text("def register(hooks, config):\n    pass\n")
        monkeypatch.setattr(plugins_mod, "PLUGINS_DIR", plugins_dir)
        registry = HookRegistry()
        loaded = plugins_mod.load_plugins(registry, {})
        # broken plugin fails, good plugin loads
        assert "good" in loaded
        assert "broken" not in loaded
