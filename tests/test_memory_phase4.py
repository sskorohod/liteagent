"""Tests for Phase 4: Procedural memory — learned workflows, crystallization, recall."""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from liteagent.memory import MemorySystem


@pytest.fixture
def proc_config(tmp_db):
    """Memory config with procedural memory enabled."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "fts_enabled": True,
            "episodic_memory": True,
        },
        "features": {
            "procedural_memory": {"enabled": True},
        },
    }


@pytest.fixture
def proc_memory(proc_config):
    """MemorySystem with procedural memory enabled."""
    ms = MemorySystem(proc_config)
    yield ms
    ms.close()


@pytest.fixture
def proc_disabled_config(tmp_db):
    """Memory config with procedural memory disabled."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
        },
        "features": {
            "procedural_memory": {"enabled": False},
        },
    }


@pytest.fixture
def proc_disabled(proc_disabled_config):
    ms = MemorySystem(proc_disabled_config)
    yield ms
    ms.close()


# ══════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════

class TestProcedureSchema:
    """Verify procedures table exists."""

    def test_procedures_table_exists(self, proc_memory):
        row = proc_memory.db.execute(
            "SELECT name FROM sqlite_master WHERE name='procedures'"
        ).fetchone()
        assert row is not None

    def test_table_exists_when_disabled(self, proc_disabled):
        row = proc_disabled.db.execute(
            "SELECT name FROM sqlite_master WHERE name='procedures'"
        ).fetchone()
        assert row is not None


# ══════════════════════════════════════════
# FEATURE FLAG
# ══════════════════════════════════════════

class TestProceduralFeatureFlag:
    def test_procedural_enabled(self, proc_memory):
        assert proc_memory._procedural_enabled() is True

    def test_procedural_disabled(self, proc_disabled):
        assert proc_disabled._procedural_enabled() is False

    def test_procedural_default_disabled(self, memory_system):
        assert memory_system._procedural_enabled() is False


# ══════════════════════════════════════════
# PROCEDURE CRUD
# ══════════════════════════════════════════

class TestProcedureCRUD:
    """Create, update, get, list, delete procedures."""

    def test_save_procedure_new(self, proc_memory):
        pid = proc_memory.save_procedure(
            name="deploy_web",
            description="Deploy a web application",
            steps=[{"tool": "exec_command", "reasoning": "Build project"},
                   {"tool": "exec_command", "reasoning": "Deploy to server"}],
            user_id="u1",
            trigger_patterns=["deploy", "push to production"],
        )
        assert pid is not None
        assert len(pid) == 36  # UUID

    def test_save_procedure_updates_existing(self, proc_memory):
        pid1 = proc_memory.save_procedure(
            name="deploy_web",
            description="Old description",
            steps=[{"tool": "exec_command"}],
            user_id="u1",
        )
        pid2 = proc_memory.save_procedure(
            name="deploy_web",
            description="New description",
            steps=[{"tool": "exec_command"}, {"tool": "web_search"}],
            user_id="u1",
        )
        assert pid1 == pid2  # Same procedure updated
        proc = proc_memory.get_procedure(pid1)
        assert proc["description"] == "New description"
        assert len(proc["steps"]) == 2

    def test_save_empty_name(self, proc_memory):
        pid = proc_memory.save_procedure("", "desc", [], "u1")
        assert pid == ""

    def test_get_procedure(self, proc_memory):
        pid = proc_memory.save_procedure(
            name="test_proc",
            description="A test procedure",
            steps=[{"tool": "read_file", "reasoning": "Read config"}],
            user_id="u1",
            trigger_patterns=["test"],
            preconditions="Requires config.json",
        )
        proc = proc_memory.get_procedure(pid)
        assert proc is not None
        assert proc["name"] == "test_proc"
        assert proc["description"] == "A test procedure"
        assert len(proc["steps"]) == 1
        assert proc["trigger_patterns"] == ["test"]
        assert proc["preconditions"] == "Requires config.json"
        assert proc["success_rate"] == 1.0
        assert proc["use_count"] == 0

    def test_get_procedure_nonexistent(self, proc_memory):
        assert proc_memory.get_procedure("nonexistent") is None

    def test_get_procedures_list(self, proc_memory):
        proc_memory.save_procedure("proc1", "desc1", [{"tool": "a"}], "u1")
        proc_memory.save_procedure("proc2", "desc2", [{"tool": "b"}], "u1")
        proc_memory.save_procedure("proc3", "desc3", [{"tool": "c"}], "u1")
        procs = proc_memory.get_procedures("u1")
        assert len(procs) == 3

    def test_get_procedures_user_isolation(self, proc_memory):
        proc_memory.save_procedure("proc1", "desc", [], "u1")
        proc_memory.save_procedure("proc2", "desc", [], "u2")
        assert len(proc_memory.get_procedures("u1")) == 1
        assert len(proc_memory.get_procedures("u2")) == 1

    def test_get_procedures_ordered_by_use_count(self, proc_memory):
        pid1 = proc_memory.save_procedure("rare", "desc", [], "u1")
        pid2 = proc_memory.save_procedure("popular", "desc", [], "u1")
        proc_memory.record_procedure_use(pid2)
        proc_memory.record_procedure_use(pid2)
        procs = proc_memory.get_procedures("u1")
        assert procs[0]["name"] == "popular"

    def test_delete_procedure(self, proc_memory):
        pid = proc_memory.save_procedure("to_delete", "desc", [], "u1")
        proc_memory.delete_procedure(pid)
        assert proc_memory.get_procedure(pid) is None


# ══════════════════════════════════════════
# PROCEDURE USE TRACKING
# ══════════════════════════════════════════

class TestProcedureUseTracking:
    """Track procedure usage and success rate."""

    def test_record_successful_use(self, proc_memory):
        pid = proc_memory.save_procedure("proc", "desc", [], "u1")
        proc_memory.record_procedure_use(pid, success=True)
        proc = proc_memory.get_procedure(pid)
        assert proc["use_count"] == 1
        assert proc["success_rate"] == 1.0  # 0.3*1.0 + 0.7*1.0 = 1.0

    def test_record_failed_use(self, proc_memory):
        pid = proc_memory.save_procedure("proc", "desc", [], "u1")
        proc_memory.record_procedure_use(pid, success=False)
        proc = proc_memory.get_procedure(pid)
        assert proc["use_count"] == 1
        assert proc["success_rate"] < 1.0  # 0.3*0.0 + 0.7*1.0 = 0.7

    def test_record_multiple_uses(self, proc_memory):
        pid = proc_memory.save_procedure("proc", "desc", [], "u1")
        proc_memory.record_procedure_use(pid, success=True)
        proc_memory.record_procedure_use(pid, success=True)
        proc_memory.record_procedure_use(pid, success=False)
        proc = proc_memory.get_procedure(pid)
        assert proc["use_count"] == 3
        assert 0 < proc["success_rate"] < 1.0

    def test_record_nonexistent(self, proc_memory):
        # Should not crash
        proc_memory.record_procedure_use("nonexistent")


# ══════════════════════════════════════════
# PROCEDURE RECALL
# ══════════════════════════════════════════

class TestProcedureRecall:
    """Search procedures by query."""

    def test_recall_by_name(self, proc_memory):
        proc_memory.save_procedure(
            "deploy_web", "Deploy a web application",
            [{"tool": "exec_command"}], "u1")
        results = proc_memory.recall_procedures("deploy", "u1")
        assert len(results) >= 1
        assert results[0]["name"] == "deploy_web"

    def test_recall_by_description(self, proc_memory):
        proc_memory.save_procedure(
            "backup_db", "Create database backup and upload",
            [{"tool": "exec_command"}], "u1")
        results = proc_memory.recall_procedures("database backup", "u1")
        assert len(results) >= 1

    def test_recall_by_trigger_pattern(self, proc_memory):
        proc_memory.save_procedure(
            "fix_bugs", "Debug and fix bugs",
            [{"tool": "read_file"}, {"tool": "exec_command"}], "u1",
            trigger_patterns=["debug", "fix error", "bug"])
        results = proc_memory.recall_procedures("fix error in code", "u1")
        assert len(results) >= 1

    def test_recall_disabled(self, proc_disabled):
        proc_disabled.save_procedure("proc", "desc", [], "u1")
        results = proc_disabled.recall_procedures("proc", "u1")
        assert results == []

    def test_recall_empty(self, proc_memory):
        results = proc_memory.recall_procedures("anything", "u1")
        assert results == []

    def test_recall_user_isolation(self, proc_memory):
        proc_memory.save_procedure("secret", "desc", [], "u1")
        results = proc_memory.recall_procedures("secret", "u2")
        assert results == []

    def test_recall_top_k(self, proc_memory):
        for i in range(5):
            proc_memory.save_procedure(f"proc_{i}", f"description {i}", [], "u1")
        results = proc_memory.recall_procedures("proc", "u1", top_k=2)
        assert len(results) <= 2


# ══════════════════════════════════════════
# CRYSTALLIZATION FROM EPISODES
# ══════════════════════════════════════════

class TestCrystallization:
    """Create procedures from recurring episode patterns."""

    @pytest.mark.asyncio
    async def test_crystallize_from_repeating_episodes(self, proc_memory):
        # Create 3 closed episodes with the same tool sequence
        for i in range(3):
            ep_id = proc_memory.start_episode("u1")
            proc_memory.add_episode_turn(ep_id, f"Task {i}", "Done", [{"name": "web_search"}])
            proc_memory.add_episode_turn(ep_id, "Now run it", "OK", [{"name": "exec_command"}])
            proc_memory.db.execute(
                """UPDATE episodes SET closed_at = ?, summary = ?,
                   tool_sequence = ? WHERE id = ?""",
                (datetime.now().isoformat(), f"User searched and ran command {i}",
                 '["web_search", "exec_command"]', ep_id))
        proc_memory.db.commit()

        created = await proc_memory.crystallize_procedures("u1", min_occurrences=2)
        assert len(created) >= 1
        # Procedure should exist
        procs = proc_memory.get_procedures("u1")
        assert len(procs) >= 1
        assert any("web_search" in p["name"] or "exec_command" in p["name"] for p in procs)

    @pytest.mark.asyncio
    async def test_crystallize_no_repeating_pattern(self, proc_memory):
        # Only 1 episode — not enough for crystallization
        ep_id = proc_memory.start_episode("u1")
        proc_memory.add_episode_turn(ep_id, "Do X", "Done", [{"name": "read_file"}])
        proc_memory.db.execute(
            """UPDATE episodes SET closed_at = ?, summary = ?,
               tool_sequence = ? WHERE id = ?""",
            (datetime.now().isoformat(), "Read a file",
             '["read_file"]', ep_id))
        proc_memory.db.commit()

        created = await proc_memory.crystallize_procedures("u1", min_occurrences=2)
        assert created == []

    @pytest.mark.asyncio
    async def test_crystallize_disabled(self, proc_disabled):
        created = await proc_disabled.crystallize_procedures("u1")
        assert created == []

    @pytest.mark.asyncio
    async def test_crystallize_no_duplicates(self, proc_memory):
        # Create repeating pattern
        for i in range(3):
            ep_id = proc_memory.start_episode("u1")
            proc_memory.db.execute(
                """UPDATE episodes SET closed_at = ?, summary = ?,
                   tool_sequence = ? WHERE id = ?""",
                (datetime.now().isoformat(), f"Run tests {i}",
                 '["exec_command"]', ep_id))
        proc_memory.db.commit()

        # Crystallize twice — should not create duplicates
        await proc_memory.crystallize_procedures("u1", min_occurrences=2)
        await proc_memory.crystallize_procedures("u1", min_occurrences=2)
        procs = proc_memory.get_procedures("u1")
        assert len(procs) == 1


# ══════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ══════════════════════════════════════════

class TestProceduralBackwardCompat:
    """Ensure procedural features don't break existing memory."""

    @pytest.mark.asyncio
    async def test_memory_works_with_procedures(self, proc_memory):
        await proc_memory.remember("User likes Python", "u1", "fact", 0.8)
        results = proc_memory.recall("Python", "u1")
        assert len(results) >= 1

    def test_old_fixture_unaffected(self, memory_system):
        memory_system.add_message("u1", "user", "hello")
        history = memory_system.get_history("u1")
        assert len(history) == 1
