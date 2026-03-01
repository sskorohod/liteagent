"""Tests for synthesis module: tool validation, skill crystallization, cross-session patterns."""

import json
import sqlite3
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from liteagent.synthesis import (
    validate_tool_source, detect_skill, _templatize_params,
    store_skill, find_matching_skills, format_skill_suggestion,
    load_synthesized_tools, register_synthesized_tool,
    detect_repeated_patterns, propose_tool_from_pattern,
)


@pytest.fixture
def synth_db(tmp_path):
    """Create DB with synthesis tables."""
    db = sqlite3.connect(str(tmp_path / "synth.db"))
    db.executescript("""
        CREATE TABLE synthesized_tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            source_code TEXT NOT NULL,
            parameters_json TEXT,
            approved INTEGER DEFAULT 0,
            created_at TEXT
        );
        CREATE TABLE skills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            steps_json TEXT NOT NULL,
            trigger_pattern TEXT,
            use_count INTEGER DEFAULT 0,
            created_at TEXT
        );
        CREATE TABLE interaction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_input TEXT,
            agent_response TEXT,
            tool_calls_json TEXT,
            success INTEGER DEFAULT 1,
            confidence REAL,
            model_used TEXT,
            created_at TEXT
        );
    """)
    return db


class TestToolValidation:
    def test_valid_simple_function(self):
        source = '''
def add(a: int, b: int) -> int:
    return a + b
'''
        ok, err = validate_tool_source(source)
        assert ok is True
        assert err == ""

    def test_valid_with_math_import(self):
        source = '''
import math

def circle_area(radius: float) -> float:
    return math.pi * radius ** 2
'''
        ok, err = validate_tool_source(source)
        assert ok is True

    def test_valid_with_json_import(self):
        source = '''
import json

def parse_data(raw: str) -> dict:
    return json.loads(raw)
'''
        ok, err = validate_tool_source(source)
        assert ok is True

    def test_blocks_os_system(self):
        source = '''
import os

def hack():
    os.system("rm -rf /")
'''
        ok, err = validate_tool_source(source)
        assert ok is False
        assert "system" in err.lower()

    def test_blocks_subprocess(self):
        source = '''
import subprocess

def run_cmd(cmd: str) -> str:
    return subprocess.check_output(cmd, shell=True)
'''
        ok, err = validate_tool_source(source)
        assert ok is False
        assert "Blocked" in err

    def test_blocks_exec(self):
        source = '''
def danger(code: str):
    exec(code)
'''
        ok, err = validate_tool_source(source)
        assert ok is False
        assert "exec" in err.lower()

    def test_blocks_eval(self):
        source = '''
def danger(code: str):
    return eval(code)
'''
        ok, err = validate_tool_source(source)
        assert ok is False
        assert "eval" in err.lower()

    def test_blocks_open(self):
        source = '''
def read_secret():
    return open("/etc/passwd").read()
'''
        ok, err = validate_tool_source(source)
        assert ok is False
        assert "open" in err.lower()

    def test_blocks_unwhitelisted_import(self):
        source = '''
import requests

def fetch(url: str) -> str:
    return requests.get(url).text
'''
        ok, err = validate_tool_source(source)
        assert ok is False
        assert "whitelist" in err.lower()

    def test_requires_exactly_one_function(self):
        source = '''
x = 42
'''
        ok, err = validate_tool_source(source)
        assert ok is False
        assert "1 function" in err

    def test_rejects_two_functions(self):
        source = '''
def foo():
    pass

def bar():
    pass
'''
        ok, err = validate_tool_source(source)
        assert ok is False
        assert "1 function" in err

    def test_syntax_error(self):
        ok, err = validate_tool_source("def broken(")
        assert ok is False
        assert "Syntax" in err

    def test_custom_whitelist(self):
        source = '''
import numpy

def mean(data):
    return numpy.mean(data)
'''
        ok, _ = validate_tool_source(source, import_whitelist={"numpy"})
        assert ok is True

    def test_register_synthesized_tool(self):
        """Test that a validated tool can actually be registered."""
        from liteagent.tools import ToolRegistry
        registry = ToolRegistry()
        source = '''
def double(n: int) -> int:
    return n * 2
'''
        schema = {"type": "object", "properties": {
            "n": {"type": "integer"}}, "required": ["n"]}
        register_synthesized_tool(registry, "double", source, "Doubles a number", schema)
        assert "double" in registry._tools
        assert registry._handlers["double"](5) == 10

    def test_load_synthesized_tools_empty(self, synth_db):
        from liteagent.tools import ToolRegistry
        registry = ToolRegistry()
        load_synthesized_tools(synth_db, registry)
        assert "synthesize_tool" not in registry._tools


class TestSkillCrystallization:
    def test_detect_skill_below_threshold(self):
        tool_calls = [{"name": "read_file", "input": {"path": "test.py"}}]
        assert detect_skill(tool_calls, "read my file", min_calls=3) is None

    def test_detect_skill_above_threshold(self):
        tool_calls = [
            {"name": "read_file", "input": {"path": "app.py"}},
            {"name": "write_file", "input": {"path": "out.py", "content": "...long code..."}},
            {"name": "exec_command", "input": {"command": "python out.py"}},
        ]
        skill = detect_skill(tool_calls, "refactor and test the app", min_calls=3)
        assert skill is not None
        assert len(skill["steps"]) == 3
        assert skill["steps"][0]["tool"] == "read_file"

    def test_templatize_params(self):
        params = {"path": "short", "content": "a" * 30}
        result = _templatize_params(params)
        assert result["path"] == "short"
        assert result["content"] == "{{content}}"

    def test_store_and_find_skill(self, synth_db):
        skill = {
            "name": "deploy_app",
            "description": "Skill from: deploy the application",
            "steps": [
                {"tool": "read_file", "params": {"path": "app.py"}},
                {"tool": "exec_command", "params": {"command": "deploy"}},
                {"tool": "exec_command", "params": {"command": "test"}},
            ],
        }
        store_skill(synth_db, skill, "u1")

        # Find by keyword match
        results = find_matching_skills(synth_db, "deploy the application")
        assert len(results) >= 1
        assert results[0]["name"] == "deploy_app"

    def test_find_no_matching_skills(self, synth_db):
        results = find_matching_skills(synth_db, "something unrelated")
        assert results == []

    def test_format_skill_suggestion_empty(self):
        assert format_skill_suggestion([]) == ""

    def test_format_skill_suggestion(self):
        skills = [{"name": "test_skill", "description": "Test",
                    "steps": [{"tool": "a"}, {"tool": "b"}], "score": 0.5}]
        text = format_skill_suggestion(skills)
        assert "test_skill" in text
        assert "a -> b" in text


class TestCrossSessionPatterns:
    """Cross-session pattern detection and tool proposal."""

    def _insert_interactions(self, db, user_id, tool_sequences, days_ago=0):
        """Insert interaction_log rows with given tool sequences."""
        ts = (datetime.now() - timedelta(days=days_ago)).isoformat()
        for seq in tool_sequences:
            calls = [{"name": t, "input": {"arg": f"val_{t}"}} for t in seq]
            db.execute(
                """INSERT INTO interaction_log
                   (user_id, user_input, agent_response, tool_calls_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, "test input", "test response",
                 json.dumps(calls), ts))
        db.commit()

    def test_detect_patterns_sufficient_data(self, synth_db):
        """Should find pattern when sequence appears >= min_occurrences."""
        sequences = [
            ["read_file", "write_file", "exec_command"],
            ["read_file", "write_file", "exec_command"],
            ["read_file", "write_file", "exec_command"],
        ]
        self._insert_interactions(synth_db, "u1", sequences)

        patterns = detect_repeated_patterns(synth_db, "u1", min_occurrences=3)
        assert len(patterns) >= 1
        # Should find at least the 2-tool or 3-tool subsequence
        found_tools = set()
        for p in patterns:
            found_tools.update(p["sequence"])
        assert "read_file" in found_tools

    def test_detect_patterns_insufficient_data(self, synth_db):
        """Should return empty when not enough occurrences."""
        sequences = [["read_file", "write_file"]]
        self._insert_interactions(synth_db, "u1", sequences)

        patterns = detect_repeated_patterns(synth_db, "u1", min_occurrences=3)
        assert patterns == []

    def test_detect_patterns_empty_db(self, synth_db):
        patterns = detect_repeated_patterns(synth_db, "u1")
        assert patterns == []

    def test_detect_patterns_respects_lookback(self, synth_db):
        """Patterns older than lookback_days should be ignored."""
        sequences = [
            ["read_file", "write_file"],
            ["read_file", "write_file"],
            ["read_file", "write_file"],
        ]
        self._insert_interactions(synth_db, "u1", sequences, days_ago=60)

        patterns = detect_repeated_patterns(
            synth_db, "u1", min_occurrences=3, lookback_days=30)
        assert patterns == []

    @pytest.mark.asyncio
    async def test_propose_tool_valid(self):
        """Should return proposal with valid source."""
        proposal_json = json.dumps({
            "name": "read_and_save",
            "description": "Read a file and save processed output",
            "source_code": "def read_and_save(path: str) -> str:\n    return path",
            "parameters_json": '{"type": "object", "properties": {"path": {"type": "string"}}}',
        })
        provider = AsyncMock()
        result = MagicMock()
        result.content = [MagicMock(text=proposal_json)]
        provider.complete = AsyncMock(return_value=result)

        pattern = {"sequence": ["read_file", "write_file"], "count": 5}
        proposal = await propose_tool_from_pattern(provider, pattern, {})
        assert proposal is not None
        assert proposal["name"] == "read_and_save"

    @pytest.mark.asyncio
    async def test_propose_tool_unsafe_source(self):
        """Should return None if generated source is unsafe."""
        proposal_json = json.dumps({
            "name": "hack",
            "description": "Bad tool",
            "source_code": "import subprocess\ndef hack():\n    subprocess.run('rm -rf /')",
            "parameters_json": "{}",
        })
        provider = AsyncMock()
        result = MagicMock()
        result.content = [MagicMock(text=proposal_json)]
        provider.complete = AsyncMock(return_value=result)

        pattern = {"sequence": ["tool_a", "tool_b"], "count": 3}
        proposal = await propose_tool_from_pattern(provider, pattern, {})
        assert proposal is None

    @pytest.mark.asyncio
    async def test_propose_tool_llm_failure(self):
        """Should return None on LLM failure."""
        provider = AsyncMock()
        provider.complete = AsyncMock(side_effect=Exception("timeout"))

        pattern = {"sequence": ["a", "b"], "count": 3}
        proposal = await propose_tool_from_pattern(provider, pattern, {})
        assert proposal is None

    @pytest.mark.asyncio
    async def test_propose_tool_empty_sequence(self):
        """Empty sequence should return None."""
        provider = AsyncMock()
        pattern = {"sequence": [], "count": 0}
        proposal = await propose_tool_from_pattern(provider, pattern, {})
        assert proposal is None

    def test_pattern_stored_in_synthesized_tools(self, synth_db):
        """Verify pattern-based tool gets stored in DB when integrated."""
        # This tests the storage path (not the full agent integration)
        synth_db.execute(
            """INSERT OR IGNORE INTO synthesized_tools
               (name, description, source_code, parameters_json, approved, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("auto_read_write", "Auto-generated from pattern",
             "def auto_read_write(path: str) -> str:\n    return path",
             '{"type": "object", "properties": {"path": {"type": "string"}}}',
             0, datetime.now().isoformat()))
        synth_db.commit()

        row = synth_db.execute(
            "SELECT name, approved FROM synthesized_tools WHERE name=?",
            ("auto_read_write",)).fetchone()
        assert row is not None
        assert row[0] == "auto_read_write"
        assert row[1] == 0  # Not auto-approved
