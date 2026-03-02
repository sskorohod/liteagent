"""Tests for boot checks system (boot.py)."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from liteagent.boot import (
    parse_boot_file, find_boot_file, execute_boot_instruction,
    run_boot_checks, BootInstruction
)


class TestParseBootFile:
    def test_parse_check_providers(self):
        content = "# Boot\n\n## CHECK providers\nVerify all keys are valid."
        instructions = parse_boot_file(content)
        assert len(instructions) == 1
        assert instructions[0].type == "check"
        assert instructions[0].target == "providers"
        assert "Verify" in instructions[0].content

    def test_parse_check_channels(self):
        content = "## CHECK channels\nTest connectivity."
        instructions = parse_boot_file(content)
        assert len(instructions) == 1
        assert instructions[0].type == "check"
        assert instructions[0].target == "channels"

    def test_parse_task(self):
        content = "## TASK daily_summary\nGenerate a brief summary."
        instructions = parse_boot_file(content)
        assert len(instructions) == 1
        assert instructions[0].type == "task"
        assert instructions[0].target == "daily_summary"
        assert "Generate" in instructions[0].content

    def test_parse_message(self):
        content = "## MESSAGE tg-12345\nGood morning!"
        instructions = parse_boot_file(content)
        assert len(instructions) == 1
        assert instructions[0].type == "message"
        assert instructions[0].target == "tg-12345"
        assert "Good morning" in instructions[0].content

    def test_parse_multiple_directives(self):
        content = """# Boot Instructions

## CHECK providers
Verify keys.

## TASK cleanup
Clean old files.

## MESSAGE admin
System started.
"""
        instructions = parse_boot_file(content)
        assert len(instructions) == 3
        assert instructions[0].type == "check"
        assert instructions[1].type == "task"
        assert instructions[2].type == "message"

    def test_parse_empty_file(self):
        instructions = parse_boot_file("")
        assert instructions == []

    def test_parse_no_directives(self):
        instructions = parse_boot_file("# Just a title\n\nSome text.")
        assert instructions == []

    def test_parse_case_insensitive(self):
        content = "## check Providers\nTest."
        instructions = parse_boot_file(content)
        assert len(instructions) == 1
        assert instructions[0].type == "check"

    def test_parse_multiline_content(self):
        content = """## TASK report
Line 1 of the task.
Line 2 of the task.
Line 3 of the task.
"""
        instructions = parse_boot_file(content)
        assert len(instructions) == 1
        assert "Line 1" in instructions[0].content
        assert "Line 3" in instructions[0].content


class TestFindBootFile:
    def test_no_boot_file(self, tmp_path, monkeypatch):
        import liteagent.boot as boot_mod
        monkeypatch.setattr(boot_mod, "BOOT_PATHS", [tmp_path / "nonexistent.md"])
        result = find_boot_file({})
        assert result is None

    def test_finds_boot_file(self, tmp_path, monkeypatch):
        import liteagent.boot as boot_mod
        boot_file = tmp_path / "boot.md"
        boot_file.write_text("## CHECK providers\ntest")
        monkeypatch.setattr(boot_mod, "BOOT_PATHS", [boot_file])
        result = find_boot_file({})
        assert result == boot_file

    def test_custom_path_from_config(self, tmp_path):
        boot_file = tmp_path / "custom_boot.md"
        boot_file.write_text("## CHECK providers\ntest")
        result = find_boot_file({"boot": {"file": str(boot_file)}})
        assert result == boot_file

    def test_relative_to_config(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("{}")
        boot_file = tmp_path / "boot.md"
        boot_file.write_text("## CHECK providers\ntest")
        result = find_boot_file({"_config_path": str(config_file)})
        assert result == boot_file


class TestExecuteBootInstruction:
    @pytest.mark.asyncio
    async def test_check_providers(self):
        agent = MagicMock()
        agent.config = {"agent": {"provider": "anthropic"}, "providers": {}}
        with patch("liteagent.config.get_api_key", return_value="sk-test-key"):
            instr = BootInstruction(type="check", target="providers", content="")
            result = await execute_boot_instruction(agent, instr)
        assert result["status"] == "ok"
        assert "anthropic" in result["message"]

    @pytest.mark.asyncio
    async def test_check_channels(self):
        agent = MagicMock()
        agent.config = {"channels": {"api": {"enabled": True}, "telegram": {"enabled": False}}}
        instr = BootInstruction(type="check", target="channels", content="")
        result = await execute_boot_instruction(agent, instr)
        assert result["status"] == "ok"
        assert "api" in result["message"]

    @pytest.mark.asyncio
    async def test_task_runs_agent(self):
        agent = MagicMock()
        agent.run = AsyncMock(return_value="Task completed successfully")
        instr = BootInstruction(type="task", target="daily", content="Do something")
        result = await execute_boot_instruction(agent, instr)
        assert result["status"] == "ok"
        assert "Task completed" in result["message"]
        agent.run.assert_called_once_with("Do something", user_id="boot")

    @pytest.mark.asyncio
    async def test_message_queued(self):
        agent = MagicMock()
        instr = BootInstruction(type="message", target="tg-123", content="Hello!")
        result = await execute_boot_instruction(agent, instr)
        assert result["status"] == "queued"

    @pytest.mark.asyncio
    async def test_unknown_check_target(self):
        agent = MagicMock()
        agent.config = {}
        instr = BootInstruction(type="check", target="unknown_thing", content="")
        result = await execute_boot_instruction(agent, instr)
        assert result["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        agent = MagicMock()
        agent.run = AsyncMock(side_effect=RuntimeError("boom"))
        instr = BootInstruction(type="task", target="failing", content="fail")
        result = await execute_boot_instruction(agent, instr)
        assert result["status"] == "error"
        assert "boom" in result["message"]


class TestRunBootChecks:
    @pytest.mark.asyncio
    async def test_disabled_in_config(self):
        agent = MagicMock()
        results = await run_boot_checks(agent, {"boot": {"enabled": False}})
        assert results == []

    @pytest.mark.asyncio
    async def test_no_boot_file(self, tmp_path, monkeypatch):
        import liteagent.boot as boot_mod
        monkeypatch.setattr(boot_mod, "BOOT_PATHS", [tmp_path / "nonexistent.md"])
        agent = MagicMock()
        results = await run_boot_checks(agent, {})
        assert results == []

    @pytest.mark.asyncio
    async def test_runs_instructions(self, tmp_path, monkeypatch):
        import liteagent.boot as boot_mod
        boot_file = tmp_path / "boot.md"
        boot_file.write_text("## CHECK channels\nTest connectivity.")
        monkeypatch.setattr(boot_mod, "BOOT_PATHS", [boot_file])
        agent = MagicMock()
        agent.config = {"channels": {"api": {}}}
        results = await run_boot_checks(agent, {})
        assert len(results) == 1
        assert results[0]["status"] == "ok"
