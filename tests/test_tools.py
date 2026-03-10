"""Tests for the tool registry and built-in tools."""

import os
import subprocess
import pytest

from liteagent.file_queue import get_file_queue, init_file_queue
from liteagent.tools import ToolRegistry, register_builtin_tools, DANGEROUS_COMMAND_PATTERNS


class TestToolRegistry:
    """Core registry functionality."""

    def test_decorator_registers_tool(self):
        registry = ToolRegistry()

        @registry.tool(name="greet", description="Say hello")
        def greet(name: str) -> str:
            return f"Hello, {name}"

        assert registry.has_tool("greet")
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "greet"
        assert defs[0]["description"] == "Say hello"

    def test_schema_generation_from_types(self):
        registry = ToolRegistry()

        @registry.tool(name="calc")
        def calc(x: int, y: float, verbose: bool = False) -> str:
            """x: First number
            y: Second number
            verbose: Show steps"""
            return str(x + y)

        schema = registry._tools["calc"]["input_schema"]
        assert schema["properties"]["x"]["type"] == "integer"
        assert schema["properties"]["y"]["type"] == "number"
        assert schema["properties"]["verbose"]["type"] == "boolean"
        assert "x" in schema["required"]
        assert "y" in schema["required"]
        assert "verbose" not in schema["required"]

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        registry = ToolRegistry()

        @registry.tool(name="echo")
        def echo(text: str) -> str:
            return text

        class MockBlock:
            type = "tool_use"
            name = "echo"
            input = {"text": "hello"}
            id = "test-id-1"

        results = await registry.execute([MockBlock()])
        assert len(results) == 1
        # Tool output is wrapped in XML anti-injection markers
        assert "hello" in results[0]["content"]
        assert "<tool_output" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        registry = ToolRegistry()

        @registry.tool(name="async_echo")
        async def async_echo(text: str) -> str:
            return f"async: {text}"

        class MockBlock:
            type = "tool_use"
            name = "async_echo"
            input = {"text": "world"}
            id = "test-id-2"

        results = await registry.execute([MockBlock()])
        assert "async: world" in results[0]["content"]
        assert "<tool_output" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        registry = ToolRegistry()

        class MockBlock:
            type = "tool_use"
            name = "nonexistent"
            input = {}
            id = "test-id-3"

        results = await registry.execute([MockBlock()])
        assert "Error: unknown tool" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_output_capped_at_10k(self):
        registry = ToolRegistry()

        @registry.tool(name="big_output")
        def big_output() -> str:
            return "x" * 20000

        class MockBlock:
            type = "tool_use"
            name = "big_output"
            input = {}
            id = "test-id-4"

        results = await registry.execute([MockBlock()])
        # Content is capped at 10k + XML wrapper (~50 chars overhead)
        assert len(results[0]["content"]) <= 10100

    @pytest.mark.asyncio
    async def test_execute_parallel_reports_progress_events(self):
        registry = ToolRegistry()

        @registry.tool(name="echo")
        async def echo(text: str) -> str:
            return text

        class MockBlock:
            type = "tool_use"

            def __init__(self, name: str, text: str, block_id: str):
                self.name = name
                self.input = {"text": text}
                self.id = block_id

        events = []

        async def on_progress(event):
            events.append((event["event"], event["tool_use_id"], event["tool_name"]))

        results = await registry.execute_parallel([
            MockBlock("echo", "one", "b1"),
            MockBlock("echo", "two", "b2"),
        ], on_progress=on_progress)

        assert len(results) == 2
        assert events.count(("start", "b1", "echo")) == 1
        assert events.count(("start", "b2", "echo")) == 1
        assert events.count(("done", "b1", "echo")) == 1
        assert events.count(("done", "b2", "echo")) == 1

    def test_semantic_tool_selection_falls_back_when_query_embedding_is_invalid(self):
        registry = ToolRegistry()

        @registry.tool(name="one", description="First tool")
        def one() -> str:
            return "one"

        @registry.tool(name="two", description="Second tool")
        def two() -> str:
            return "two"

        class BadEmbedder:
            def encode(self, text: str):
                raise ValueError("broken embedding")

        defs = registry.get_relevant_definitions("build project", top_k=1, embedder=BadEmbedder())
        assert {d["name"] for d in defs} == {"one", "two"}

    def test_semantic_tool_selection_skips_incompatible_tool_embeddings(self):
        registry = ToolRegistry()

        @registry.tool(name="one", description="First tool")
        def one() -> str:
            return "one"

        @registry.tool(name="two", description="Second tool")
        def two() -> str:
            return "two"

        class Vec:
            def __init__(self, values):
                self.values = list(values)

            def __matmul__(self, other):
                if len(self.values) != len(other.values):
                    raise ValueError("dimension mismatch")
                return sum(a * b for a, b in zip(self.values, other.values))

        class FlakyEmbedder:
            def encode(self, text: str):
                if text.startswith("two:"):
                    return Vec([1.0])
                return Vec([1.0, 0.0])

        defs = registry.get_relevant_definitions("build project", top_k=1, embedder=FlakyEmbedder())
        assert [d["name"] for d in defs] == ["one"]

    def test_keyword_tool_selection_prefers_compact_dev_bundle(self):
        registry = ToolRegistry()
        register_builtin_tools(
            registry,
            enabled=[
                "read_file", "write_file", "edit_file", "exec_command",
                "glob_files", "grep_search", "memory_search",
            ],
        )

        @registry.tool(name="transcribe_voice", description="Transcribe audio to text")
        def transcribe_voice() -> str:
            return "ok"

        defs = registry.get_keyword_relevant_definitions(
            "fix the frontend build and debug the backend project",
            top_k=5,
        )
        names = [d["name"] for d in defs]

        assert "exec_command" in names
        assert "read_file" in names
        assert "write_file" in names or "edit_file" in names
        assert "transcribe_voice" not in names
        assert len(defs) <= 6

    def test_keyword_tool_selection_prefers_browser_tools_for_ui_e2e(self):
        registry = ToolRegistry()

        @registry.tool(name="chrome_devtools__new_page", description="Open a page in Chrome DevTools")
        def new_page() -> str:
            return "ok"

        @registry.tool(name="chrome_devtools__click", description="Click an element in Chrome DevTools")
        def click() -> str:
            return "ok"

        @registry.tool(name="exec_command", description="Run a shell command")
        def exec_command(command: str) -> str:
            return command

        @registry.tool(name="memory_search", description="Search memory")
        def memory_search(query: str) -> str:
            return query

        defs = registry.get_keyword_relevant_definitions(
            "browser e2e ui test for frontend in chrome devtools",
            top_k=3,
        )
        names = [d["name"] for d in defs]

        assert "chrome_devtools__new_page" in names
        assert "chrome_devtools__click" in names
        assert "exec_command" in names


class TestBuiltinTools:
    """Built-in tool implementations."""

    def test_read_file_exists(self, tool_registry, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        handler = tool_registry._handlers["read_file"]
        result = handler(path=str(test_file))
        # Enhanced read_file returns line-numbered output
        assert "hello world" in result
        assert "1\t" in result  # line number format

    def test_read_file_not_found(self, tool_registry):
        handler = tool_registry._handlers["read_file"]
        result = handler(path="/nonexistent/file.txt")
        assert "File not found" in result

    def test_exec_command_basic(self, tool_registry):
        handler = tool_registry._handlers["exec_command"]
        result = handler(command="echo test123")
        assert "test123" in result

    def test_exec_command_timeout(self, tool_registry):
        handler = tool_registry._handlers["exec_command"]
        # Use python3 (in allowlist) instead of sleep (not in allowlist)
        result = handler(command='python3 -c "import time; time.sleep(10)"', timeout=1)
        assert "timed out" in result

    def test_exec_command_background_shell_returns_immediately(self, monkeypatch):
        registry = ToolRegistry()
        register_builtin_tools(registry, enabled=["exec_command"], allow_shell=True)
        handler = registry._handlers["exec_command"]

        called = {}

        class DummyProc:
            pid = 43210

        def fake_popen(command, shell, stdout, stderr, cwd, start_new_session):
            called["command"] = command
            called["shell"] = shell
            called["cwd"] = cwd
            called["start_new_session"] = start_new_session
            return DummyProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        result = handler(command="nohup python3 main.py > server.log 2>&1 &")

        assert "Started background command" in result
        assert "43210" in result
        assert called["command"] == "nohup python3 main.py > server.log 2>&1 &"
        assert called["shell"] is True
        assert called["start_new_session"] is True

    def test_exec_command_foreground_server_is_detached(self, monkeypatch, tmp_path):
        registry = ToolRegistry()
        register_builtin_tools(registry, enabled=["exec_command"], allow_shell=True)
        handler = registry._handlers["exec_command"]

        called = {}
        monkeypatch.chdir(tmp_path)

        class DummyProc:
            pid = 54321

        def fake_popen(command, shell, stdout, stderr, cwd, start_new_session):
            called["command"] = command
            called["shell"] = shell
            called["cwd"] = cwd
            called["start_new_session"] = start_new_session
            called["stdout_name"] = getattr(stdout, "name", "")
            return DummyProc()

        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        result = handler(command="python3 -m uvicorn main:app --host 0.0.0.0 --port 8091 --reload")

        assert "Started long-running server command in background" in result
        assert "54321" in result
        assert called["command"] == "python3 -m uvicorn main:app --host 0.0.0.0 --port 8091 --reload"
        assert called["shell"] is True
        assert called["start_new_session"] is True
        assert called["cwd"] == str(tmp_path)
        assert called["stdout_name"].endswith("liteagent-bg.log")

    def test_exec_command_blocks_dangerous(self, tool_registry):
        handler = tool_registry._handlers["exec_command"]
        result = handler(command="rm -rf /")
        assert "Blocked" in result

    def test_key_dangerous_patterns_blocked(self, tool_registry):
        handler = tool_registry._handlers["exec_command"]
        # Test patterns that are safe to check (won't actually execute on this OS)
        test_patterns = ["rm -rf /", "rm -fr /", "mkfs", "dd if=", "> /dev/sd",
                         ":(){ :", "format c:"]
        for pattern in test_patterns:
            result = handler(command=pattern)
            assert "Blocked" in result, f"Pattern not blocked: {pattern}"

    def test_download_file_blocks_ssrf(self):
        registry = ToolRegistry()
        register_builtin_tools(registry, enabled=["download_file"])
        handler = registry._handlers["download_file"]
        result = handler(url="http://127.0.0.1:9/probe")
        assert "Blocked" in result
        assert "SSRF" in result

    def test_download_file_blocks_non_http_scheme(self):
        registry = ToolRegistry()
        register_builtin_tools(registry, enabled=["download_file"])
        handler = registry._handlers["download_file"]
        result = handler(url="file:///etc/passwd")
        assert "Blocked" in result
        assert "http/https" in result

    def test_send_file_to_user_wraps_text_content_as_temp_attachment(self):
        registry = ToolRegistry()
        register_builtin_tools(registry, enabled=["send_file_to_user"])
        handler = registry._handlers["send_file_to_user"]

        init_file_queue()
        result = handler(content="hello from agent", caption="summary")

        assert "queued" in result.lower()
        queue = get_file_queue()
        assert len(queue) == 1
        assert queue[0].path.endswith(".txt")
        assert queue[0].caption == "summary"
        with open(queue[0].path, "r", encoding="utf-8") as f:
            assert f.read() == "hello from agent"
