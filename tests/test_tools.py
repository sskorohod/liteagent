"""Tests for the tool registry and built-in tools."""

import os
import pytest

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
        assert results[0]["content"] == "hello"

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
        assert results[0]["content"] == "async: world"

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
        assert len(results[0]["content"]) <= 10000


class TestBuiltinTools:
    """Built-in tool implementations."""

    def test_read_file_exists(self, tool_registry, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        handler = tool_registry._handlers["read_file"]
        result = handler(path=str(test_file))
        assert result == "hello world"

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
        result = handler(command="sleep 10", timeout=1)
        assert "timed out" in result

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
