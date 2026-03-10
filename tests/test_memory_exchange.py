"""Tests for Memory Exchange + Shadow Twin pipeline."""

import hashlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from liteagent.memory import MemorySystem


@pytest.fixture
def exchange_config(tmp_db):
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "memory_exchange_enabled": True,
            "memory_exchange_top_k": 8,
            "memory_exchange_pack_budget_tokens": 420,
            "memory_exchange_max_packs": 2,
            "memory_exchange_context_budget_tokens": 700,
            "shadow_twin_enabled": True,
            "shadow_twin_predictions": 3,
            "shadow_twin_use_llm": False,
        }
    }


@pytest.fixture
def exchange_memory(exchange_config):
    ms = MemorySystem(exchange_config)
    yield ms
    ms.close()


class TestMemoryExchange:
    @pytest.mark.asyncio
    async def test_cycle_creates_packs_and_context(self, exchange_memory):
        await exchange_memory.remember(
            "We deploy services with Docker and run FastAPI behind nginx.",
            "u1",
            "fact",
            0.8,
        )
        await exchange_memory.remember(
            "Production database is PostgreSQL with nightly backups.",
            "u1",
            "fact",
            0.75,
        )

        result = await exchange_memory.run_memory_exchange_cycle(
            "docker deployment process",
            "u1",
            "Use docker compose with rollback",
        )
        assert result["status"] == "ok"
        assert result["packs_created"] >= 1
        intents = exchange_memory.db.execute(
            "SELECT COUNT(*) FROM memory_exchange_intents WHERE user_id = ?",
            ("u1",),
        ).fetchone()[0]
        assert intents >= 1

        ctx = exchange_memory.get_memory_exchange_context(
            "docker deployment process",
            "u1",
        )
        assert "Memory Exchange (precomputed)" in ctx
        assert "Docker" in ctx or "docker" in ctx

    @pytest.mark.asyncio
    async def test_context_respects_max_packs(self, exchange_memory):
        await exchange_memory.remember("Alpha project uses Redis and Celery.", "u1", "fact", 0.7)
        await exchange_memory.remember("Beta project uses Kafka and ClickHouse.", "u1", "fact", 0.7)

        await exchange_memory.run_memory_exchange_cycle("redis celery pipeline", "u1", "")
        await exchange_memory.run_memory_exchange_cycle("kafka clickhouse analytics", "u1", "")

        ctx = exchange_memory.get_memory_exchange_context(
            "analytics pipeline",
            "u1",
            max_packs=1,
            token_budget=500,
        )
        # Should include only one selected pack marker
        assert ctx.count("[Pack ") == 1

    @pytest.mark.asyncio
    async def test_cycle_creates_shadow_prediction_rows(self, exchange_memory):
        await exchange_memory.remember("User works with Terraform and Kubernetes.", "u1", "fact", 0.8)
        await exchange_memory.run_memory_exchange_cycle("kubernetes setup", "u1", "")

        count = exchange_memory.db.execute(
            "SELECT COUNT(*) FROM memory_shadow_predictions WHERE user_id = ?",
            ("u1",),
        ).fetchone()[0]
        assert count >= 1

    @pytest.mark.asyncio
    async def test_cycle_fallbacks_to_chat_history_when_no_memories(self, exchange_memory):
        exchange_memory.add_message("u2", "user", "We deploy FastAPI with Docker in production")
        exchange_memory.add_message("u2", "assistant", "Noted: FastAPI + Docker stack")

        result = await exchange_memory.run_memory_exchange_cycle(
            "fastapi docker deployment",
            "u2",
            "",
        )
        assert result["packs_created"] >= 1
        ctx = exchange_memory.get_memory_exchange_context("fastapi docker deployment", "u2")
        assert "FastAPI" in ctx or "Docker" in ctx

    @pytest.mark.asyncio
    async def test_shadow_llm_uses_dedicated_extraction_provider(self, tmp_db):
        config = {
            "memory": {
                "db_path": tmp_db,
                "auto_learn": False,
                "memory_exchange_enabled": True,
                "shadow_twin_enabled": True,
                "shadow_twin_use_llm": True,
                "shadow_twin_predictions": 3,
                "extraction_model": "qwen2.5:latest",
            }
        }
        main_provider = AsyncMock()
        main_provider.complete = AsyncMock()
        ms = MemorySystem(config, provider=main_provider)
        try:
            await ms.remember(
                "Kubernetes autoscaling and cost optimization playbook.",
                "u1",
                "fact",
                0.9,
            )

            llm_response = MagicMock()
            llm_response.content = [MagicMock(text='["kubernetes autoscaling", "cost optimization kubernetes"]')]
            extraction_provider = AsyncMock()
            extraction_provider.complete = AsyncMock(return_value=llm_response)
            ms._extraction_provider = extraction_provider

            result = await ms.run_memory_exchange_cycle("infra performance", "u1", "")

            assert result["packs_created"] >= 1
            assert extraction_provider.complete.await_count == 1
            assert main_provider.complete.await_count == 0
        finally:
            ms.close()


class TestMemoryExchangeDaemon:
    @pytest.mark.asyncio
    async def test_queue_respects_priority_order(self, exchange_memory, monkeypatch):
        order: list[str] = []

        async def _fake_core(anchor: str, user_id: str, assistant_response: str = "") -> dict:
            order.append(anchor)
            return {"status": "ok", "packs_created": 1, "predictions_created": 1}

        monkeypatch.setattr(exchange_memory, "_run_memory_exchange_cycle_core", _fake_core)

        await exchange_memory.enqueue_memory_exchange_intent(
            "A very long low-priority analytics request " + ("x" * 120),
            "u1",
            "",
            priority=7,
        )
        await exchange_memory.enqueue_memory_exchange_intent(
            "Как меня зовут?",
            "u1",
            "",
            priority=1,
        )

        result = await exchange_memory.process_memory_exchange_queue_once(max_items=2)
        assert result["processed"] == 2
        assert order == ["Как меня зовут?", "A very long low-priority analytics request " + ("x" * 120)]

    @pytest.mark.asyncio
    async def test_queue_auto_pauses_under_high_load(self, exchange_memory, monkeypatch):
        from liteagent.agent import LiteAgent

        monkeypatch.setattr(
            LiteAgent,
            "get_active_requests",
            classmethod(lambda cls: [{"id": 1, "user_id": "u1"}]),
        )
        monkeypatch.setattr(
            LiteAgent,
            "get_queued_requests",
            classmethod(lambda cls: []),
        )

        await exchange_memory.enqueue_memory_exchange_intent(
            "kubernetes deployment",
            "u1",
        )
        result = await exchange_memory.process_memory_exchange_queue_once(max_items=1)
        assert result["status"] == "paused"

        pending = exchange_memory.db.execute(
            "SELECT COUNT(*) FROM memory_exchange_intents WHERE status = 'queued'"
        ).fetchone()[0]
        assert pending >= 1

    @pytest.mark.asyncio
    async def test_daemon_start_stop(self, exchange_memory):
        started = await exchange_memory.start_memory_exchange_daemon()
        assert started["status"] in {"started", "already_running"}
        state = exchange_memory.memory_exchange_daemon_state()
        assert state["enabled"] is True
        assert state["running"] is True

        stopped = await exchange_memory.stop_memory_exchange_daemon()
        assert stopped["status"] == "stopped"

    def test_shadow_queue_cleanup_prunes_stale_duplicates_and_orphans(self, exchange_memory):
        exchange_memory.config["shadow_queue_cleanup_enabled"] = True
        exchange_memory.config["shadow_queue_cleanup_interval_sec"] = 1
        exchange_memory.config["shadow_ready_ttl_hours"] = 1
        exchange_memory.config["shadow_used_ttl_hours"] = 1
        exchange_memory.config["shadow_max_ready_per_user"] = 2

        now = datetime.now()
        fresh = now.isoformat()
        old = (now - timedelta(hours=5)).isoformat()

        exchange_memory.db.execute(
            """INSERT INTO memory_context_packs
               (id, user_id, query_hint, title, content, token_estimate, score, created_at, updated_at)
               VALUES ('pack-ok', 'u1', 'hint', 'title', 'content', 50, 0.7, ?, ?)""",
            (fresh, fresh),
        )
        rows = [
            ("dup-old", "u1", "anchor", "same q", 0.60, "pack-ok", "ready", (now - timedelta(minutes=10)).isoformat(), None),
            ("dup-new", "u1", "anchor", "same q", 0.92, "pack-ok", "ready", fresh, None),
            ("stale-ready", "u1", "anchor", "stale ready", 0.30, "pack-ok", "ready", old, None),
            ("stale-used", "u1", "anchor", "stale used", 0.30, "pack-ok", "used", old, old),
            ("orphan-pack", "u1", "anchor", "orphan", 0.40, "pack-missing", "ready", fresh, None),
        ]
        rows.extend(
            (f"cap-{i}", "u1", "anchor", f"cap {i}", 0.50 + (i * 0.001), "pack-ok", "ready", fresh, None)
            for i in range(1, 26)
        )
        exchange_memory.db.executemany(
            """INSERT INTO memory_shadow_predictions
               (id, user_id, anchor_query, predicted_query, confidence, pack_id, status, created_at, used_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        exchange_memory.db.commit()

        stats = exchange_memory.cleanup_shadow_prediction_queue(force=True)
        assert stats["status"] == "ok"
        assert stats["removed_orphan"] >= 1
        assert stats["removed_stale_ready"] >= 1
        assert stats["removed_stale_used"] >= 1

        ready_rows = exchange_memory.db.execute(
            "SELECT predicted_query FROM memory_shadow_predictions WHERE user_id = 'u1' AND status = 'ready'"
        ).fetchall()
        assert len(ready_rows) <= 20
        assert len({r[0].lower() for r in ready_rows}) == len(ready_rows)


class TestMemoryIdentityAndQuality:
    def test_canonical_slot_versioning(self, exchange_memory):
        exchange_memory.upsert_canonical_slot("u1", "name", "Alice", confidence=0.78, source="test")
        exchange_memory.upsert_canonical_slot("u1", "name", "Alicia", confidence=0.95, source="test")

        slot = exchange_memory.get_canonical_slot("u1", "name")
        assert slot is not None
        assert slot["slot_value"] == "Alicia"
        assert slot["version"] >= 2

        history = exchange_memory.get_canonical_profile_history("u1", "name", limit=10)
        assert len(history) >= 2

    @pytest.mark.asyncio
    async def test_type_aware_retrieval_prefers_profile_slot(self, exchange_memory):
        exchange_memory.upsert_canonical_slot("u1", "name", "Влад", confidence=0.96, source="test")
        await exchange_memory.remember("User likes espresso.", "u1", "fact", 0.7)

        results = exchange_memory.recall_type_aware("как меня зовут", "u1", top_k=3)
        assert results
        assert results[0]["type"] == "profile_slot"
        assert "Влад" in results[0]["content"]

        traces = exchange_memory.get_last_recall_trace("u1", limit=1)
        assert traces
        assert traces[0]["intent_slot"] == "name"

    @pytest.mark.asyncio
    async def test_pollution_memory_is_blocked(self, exchange_memory):
        mid = await exchange_memory.remember(
            "Я не помню прошлые разговоры и только текущий чат.",
            "u1",
            "fact",
            0.9,
        )
        assert mid is None
        count = exchange_memory.db.execute(
            "SELECT COUNT(*) FROM memories WHERE user_id = ?",
            ("u1",),
        ).fetchone()[0]
        assert count == 0

    @pytest.mark.asyncio
    async def test_extract_and_learn_filters_assistant_pollution(self, tmp_db):
        config = {
            "memory": {
                "db_path": tmp_db,
                "auto_learn": True,
                "memory_exchange_enabled": True,
            }
        }
        ms = MemorySystem(config)
        try:
            response = MagicMock()
            response.content = [MagicMock(text=(
                '{"facts":["Assistant: I have no long-term memory","User name is Slava"],'
                '"preferences":[],"corrections":["I cannot remember previous conversations"],'
                '"session_summary":""}'
            ))]
            extraction_provider = AsyncMock()
            extraction_provider.complete = AsyncMock(return_value=response)
            ms._extraction_provider = extraction_provider

            await ms.extract_and_learn(
                "Меня зовут Слава",
                "Я не помню прошлые разговоры",
                "u1",
            )

            rows = ms.db.execute(
                "SELECT content FROM memories WHERE user_id = ?",
                ("u1",),
            ).fetchall()
            assert all("no long-term memory" not in (r[0] or "").lower() for r in rows)
            assert all("cannot remember previous conversations" not in (r[0] or "").lower() for r in rows)
            slot = ms.get_canonical_slot("u1", "name")
            assert slot is not None
            assert "слава" in slot["slot_value"].lower()
        finally:
            ms.close()

    @pytest.mark.asyncio
    async def test_extract_and_learn_tolerates_malformed_shapes(self, tmp_db):
        config = {
            "memory": {
                "db_path": tmp_db,
                "auto_learn": True,
            }
        }
        ms = MemorySystem(config)
        try:
            response = MagicMock()
            response.content = [MagicMock(text=(
                '{"facts":"Меня зовут Вячеслав",'
                '"preferences":[{"text":"отвечай кратко"}],'
                '"corrections":["Assistant said the folder does not exist"],'
                '"session_summary":["обсуждали память"],'
                '"entities":"bad",'
                '"relations":{"source":"u","target":"p","type":"uses"}}'
            ))]
            extraction_provider = AsyncMock()
            extraction_provider.complete = AsyncMock(return_value=response)
            ms._extraction_provider = extraction_provider

            await ms.extract_and_learn(
                "Меня зовут Вячеслав, отвечай кратко",
                "Папка develop не существует",
                "u1",
            )

            rows = ms.db.execute(
                "SELECT content, type FROM memories WHERE user_id = ? ORDER BY id",
                ("u1",),
            ).fetchall()
            assert any("Вячеслав" in (row[0] or "") for row in rows)
            assert any(row[1] == "preference" for row in rows)
            assert all("folder does not exist" not in (row[0] or "").lower() for row in rows)
            summary = ms._get_session_summary("u1")
            assert "обсуждали память" in summary
        finally:
            ms.close()

    @pytest.mark.asyncio
    async def test_local_worker_backfills_canonical_slot(self, exchange_memory):
        raw = "Меня зовут Пётр"
        now = datetime.now().isoformat()
        exchange_memory.db.execute(
            """INSERT INTO memories (user_id, content, type, importance, hash, created_at, accessed_at)
               VALUES (?, ?, 'fact', 0.5, ?, ?, ?)""",
            ("u1", raw, hashlib.md5(raw.lower().encode()).hexdigest(), now, now),
        )
        exchange_memory.db.commit()

        res = await exchange_memory.run_local_memory_worker_once(max_items=10)
        assert res["status"] in {"ok", "idle"}
        slot = exchange_memory.get_canonical_slot("u1", "name")
        assert slot is not None
        assert "пётр" in slot["slot_value"].lower()

    def test_quality_metrics_schema_and_values(self, exchange_memory):
        exchange_memory.config["metrics_enabled"] = True
        exchange_memory.db.execute(
            """INSERT INTO memory_query_log
               (user_id, query, search_mode, result_count, top_score, latency_ms, was_used, created_at)
               VALUES (?, ?, 'hybrid', 1, 0.88, 12.0, 1, ?)""",
            ("u1", "как меня зовут", datetime.now().isoformat()),
        )
        exchange_memory.db.execute(
            """INSERT INTO memory_extraction_runs
               (user_id, total_candidates, saved_count, dropped_pollution, created_at)
               VALUES (?, 10, 7, 3, ?)""",
            ("u1", datetime.now().isoformat()),
        )
        exchange_memory.db.commit()

        metrics = exchange_memory.get_memory_quality_metrics("u1", days=30, k=5)
        assert "recall_at_k" in metrics
        assert "profile_accuracy" in metrics
        assert "contradiction_rate" in metrics
        assert "memory_poison_rate" in metrics
        assert metrics["memory_poison_rate"] == pytest.approx(0.3, rel=1e-3)
