"""Shared fixtures for LiteAgent tests."""

import pytest

from liteagent.memory import MemorySystem
from liteagent.tools import ToolRegistry, register_builtin_tools


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary SQLite DB path."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def memory_config(tmp_db):
    """Provide a minimal memory config with temp DB."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "keep_recent_messages": 6,
        }
    }


@pytest.fixture
def memory_system(memory_config):
    """Provide a MemorySystem with temporary DB."""
    ms = MemorySystem(memory_config)
    yield ms
    ms.close()


@pytest.fixture
def tool_registry():
    """Provide a ToolRegistry with read_file only."""
    registry = ToolRegistry()
    register_builtin_tools(registry, enabled=["read_file", "exec_command", "memory_search"])
    return registry
