"""Tests for Phase 5: Memory quality metrics — query logging, health check, staleness."""

import pytest
from datetime import datetime, timedelta

from liteagent.memory import MemorySystem


@pytest.fixture
def metrics_config(tmp_db):
    """Memory config with metrics enabled."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "fts_enabled": True,
            "metrics_enabled": True,
        }
    }


@pytest.fixture
def metrics_memory(metrics_config):
    """MemorySystem with metrics enabled."""
    ms = MemorySystem(metrics_config)
    yield ms
    ms.close()


@pytest.fixture
def no_metrics_config(tmp_db):
    """Memory config with metrics disabled."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "metrics_enabled": False,
        }
    }


@pytest.fixture
def no_metrics(no_metrics_config):
    ms = MemorySystem(no_metrics_config)
    yield ms
    ms.close()


# ══════════════════════════════════════════
# SCHEMA
# ══════════════════════════════════════════

class TestMetricsSchema:
    def test_query_log_table_exists(self, metrics_memory):
        row = metrics_memory.db.execute(
            "SELECT name FROM sqlite_master WHERE name='memory_query_log'"
        ).fetchone()
        assert row is not None

    def test_table_exists_when_disabled(self, no_metrics):
        row = no_metrics.db.execute(
            "SELECT name FROM sqlite_master WHERE name='memory_query_log'"
        ).fetchone()
        assert row is not None


# ══════════════════════════════════════════
# QUERY LOGGING
# ══════════════════════════════════════════

class TestQueryLogging:
    """Query log records written on recall()."""

    @pytest.mark.asyncio
    async def test_recall_logs_query(self, metrics_memory):
        await metrics_memory.remember("Test fact about Python", "u1", "fact", 0.8)
        metrics_memory.recall("Python", "u1")

        count = metrics_memory.db.execute(
            "SELECT COUNT(*) FROM memory_query_log WHERE user_id = 'u1'"
        ).fetchone()[0]
        assert count == 1

    @pytest.mark.asyncio
    async def test_recall_logs_latency(self, metrics_memory):
        await metrics_memory.remember("Test fact", "u1", "fact", 0.5)
        metrics_memory.recall("Test", "u1")

        row = metrics_memory.db.execute(
            "SELECT latency_ms FROM memory_query_log WHERE user_id = 'u1'"
        ).fetchone()
        assert row[0] > 0  # Should have non-zero latency

    @pytest.mark.asyncio
    async def test_recall_logs_result_count(self, metrics_memory):
        await metrics_memory.remember("Alpha beta", "u1", "fact", 0.8)
        await metrics_memory.remember("Alpha gamma", "u1", "fact", 0.7)
        metrics_memory.recall("Alpha", "u1")

        row = metrics_memory.db.execute(
            "SELECT result_count FROM memory_query_log WHERE user_id = 'u1'"
        ).fetchone()
        assert row[0] >= 1

    @pytest.mark.asyncio
    async def test_recall_logs_top_score(self, metrics_memory):
        await metrics_memory.remember("Important fact", "u1", "fact", 0.9)
        metrics_memory.recall("Important", "u1")

        row = metrics_memory.db.execute(
            "SELECT top_score FROM memory_query_log WHERE user_id = 'u1'"
        ).fetchone()
        assert row[0] > 0

    @pytest.mark.asyncio
    async def test_no_log_when_disabled(self, no_metrics):
        await no_metrics.remember("Test", "u1", "fact", 0.5)
        no_metrics.recall("Test", "u1")

        count = no_metrics.db.execute(
            "SELECT COUNT(*) FROM memory_query_log WHERE user_id = 'u1'"
        ).fetchone()[0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_empty_recall_logs_zero_results(self, metrics_memory):
        metrics_memory.recall("nothing", "u1")

        # Empty recall shouldn't reach the logging code (returns early)
        # but if it does, count should be 0
        count = metrics_memory.db.execute(
            "SELECT COUNT(*) FROM memory_query_log"
        ).fetchone()[0]
        # recall returns [] early, so no logging happens
        assert count == 0


# ══════════════════════════════════════════
# MARK QUERY USED
# ══════════════════════════════════════════

class TestMarkQueryUsed:
    def test_mark_used(self, metrics_memory):
        metrics_memory._log_query("u1", "test query", "hybrid", 3, 0.8, 10.0)
        log_id = metrics_memory.db.execute(
            "SELECT id FROM memory_query_log ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]
        metrics_memory.mark_query_used(log_id)
        row = metrics_memory.db.execute(
            "SELECT was_used FROM memory_query_log WHERE id = ?",
            (log_id,)).fetchone()
        assert row[0] == 1


# ══════════════════════════════════════════
# MEMORY METRICS
# ══════════════════════════════════════════

class TestMemoryMetrics:
    """get_memory_metrics() stats computation."""

    @pytest.mark.asyncio
    async def test_metrics_empty(self, metrics_memory):
        metrics = metrics_memory.get_memory_metrics("u1")
        assert metrics["total_queries"] == 0
        assert metrics["total_memories"] == 0
        assert metrics["hit_rate"] == 0

    @pytest.mark.asyncio
    async def test_metrics_after_queries(self, metrics_memory):
        await metrics_memory.remember("Fact A", "u1", "fact", 0.8)
        await metrics_memory.remember("Fact B", "u1", "fact", 0.7)
        metrics_memory.recall("Fact", "u1")
        metrics_memory.recall("Fact", "u1")

        metrics = metrics_memory.get_memory_metrics("u1")
        assert metrics["total_queries"] == 2
        assert metrics["total_memories"] == 2
        assert metrics["hit_rate"] > 0
        assert metrics["avg_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_metrics_stale_memories(self, metrics_memory):
        await metrics_memory.remember("Old memory", "u1", "fact", 0.5)
        # Backdate accessed_at to 60 days ago
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        metrics_memory.db.execute(
            "UPDATE memories SET accessed_at = ? WHERE user_id = 'u1'",
            (old_date,))
        metrics_memory.db.commit()

        metrics = metrics_memory.get_memory_metrics("u1")
        assert metrics["stale_memories"] >= 1

    @pytest.mark.asyncio
    async def test_metrics_entity_count(self, metrics_memory):
        metrics_memory.upsert_entity("Alice", "person", "u1")
        metrics_memory.upsert_entity("Bob", "person", "u1")
        metrics = metrics_memory.get_memory_metrics("u1")
        assert metrics["entity_count"] == 2

    @pytest.mark.asyncio
    async def test_metrics_episode_count(self, metrics_memory):
        metrics_memory.start_episode("u1")
        metrics_memory.start_episode("u1")
        metrics = metrics_memory.get_memory_metrics("u1")
        assert metrics["episode_count"] == 2

    @pytest.mark.asyncio
    async def test_metrics_procedure_count(self, metrics_memory):
        metrics_memory.save_procedure("proc1", "desc", [], "u1")
        metrics = metrics_memory.get_memory_metrics("u1")
        assert metrics["procedure_count"] == 1

    @pytest.mark.asyncio
    async def test_metrics_all_users(self, metrics_memory):
        await metrics_memory.remember("Fact for user one", "u1", "fact", 0.5)
        await metrics_memory.remember("Fact for user two", "u2", "fact", 0.5)
        metrics = metrics_memory.get_memory_metrics()  # No user filter
        assert metrics["total_memories"] == 2


# ══════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════

class TestHealthCheck:
    """memory_health_check() returns status and issues."""

    def test_health_check_healthy(self, metrics_memory):
        result = metrics_memory.memory_health_check("u1")
        assert result["status"] == "healthy"
        assert result["issues"] == []
        assert "metrics" in result

    @pytest.mark.asyncio
    async def test_health_check_stale_warning(self, metrics_memory):
        # Create many stale memories
        for i in range(10):
            await metrics_memory.remember(f"Fact {i}", "u1", "fact", 0.5)
        old_date = (datetime.now() - timedelta(days=60)).isoformat()
        metrics_memory.db.execute(
            "UPDATE memories SET accessed_at = ? WHERE user_id = 'u1'",
            (old_date,))
        metrics_memory.db.commit()

        result = metrics_memory.memory_health_check("u1")
        assert result["status"] == "warning"
        assert any("stale" in issue for issue in result["issues"])

    def test_health_check_low_hit_rate_warning(self, metrics_memory):
        # Simulate many queries with 0 results
        for i in range(15):
            metrics_memory._log_query("u1", f"query {i}", "hybrid", 0, 0.0, 5.0)

        result = metrics_memory.memory_health_check("u1")
        assert result["status"] == "warning"
        assert any("hit rate" in issue.lower() for issue in result["issues"])

    def test_health_check_no_user_filter(self, metrics_memory):
        result = metrics_memory.memory_health_check()
        assert result["status"] == "healthy"


# ══════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ══════════════════════════════════════════

class TestMetricsBackwardCompat:
    @pytest.mark.asyncio
    async def test_memory_works_with_metrics(self, metrics_memory):
        await metrics_memory.remember("User likes Python", "u1", "fact", 0.8)
        results = metrics_memory.recall("Python", "u1")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_memory_works_without_metrics(self, no_metrics):
        await no_metrics.remember("User likes Java", "u1", "fact", 0.8)
        results = no_metrics.recall("Java", "u1")
        assert len(results) >= 1

    def test_old_fixture_unaffected(self, memory_system):
        memory_system.add_message("u1", "user", "hello")
        history = memory_system.get_history("u1")
        assert len(history) == 1
