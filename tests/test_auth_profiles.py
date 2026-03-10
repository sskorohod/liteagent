"""Tests for auth_profiles module (multi-key rotation + cooldown)."""

import asyncio
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from liteagent.auth_profiles import (
    AuthProfileManager,
    AuthProfile,
    ProfileStats,
    register_keys_from_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def manager(tmp_path):
    """Fresh AuthProfileManager backed by a tmp store path."""
    store_path = tmp_path / "auth_profiles.json"
    with patch("liteagent.auth_profiles._STORE_PATH", store_path):
        mgr = AuthProfileManager()
        yield mgr


# ── register ──────────────────────────────────────────────────────────────────

class TestRegister:
    def test_register_single_key(self, manager):
        manager.register("anthropic", ["key1"])
        profiles = manager.list_profiles("anthropic")
        assert len(profiles) == 1
        assert profiles[0].api_key == "key1"

    def test_register_multiple_keys(self, manager):
        manager.register("anthropic", ["key1", "key2", "key3"])
        profiles = manager.list_profiles("anthropic")
        assert len(profiles) == 3

    def test_register_string_shorthand(self, manager):
        manager.register("anthropic", "key1")
        profiles = manager.list_profiles("anthropic")
        assert len(profiles) == 1

    def test_register_no_duplicates(self, manager):
        manager.register("anthropic", ["key1", "key2"])
        manager.register("anthropic", ["key2", "key3"])
        profiles = manager.list_profiles("anthropic")
        keys = [p.api_key for p in profiles]
        assert keys.count("key2") == 1
        assert len(profiles) == 3

    def test_register_different_providers(self, manager):
        manager.register("anthropic", ["key1"])
        manager.register("openai", ["oai-key1"])
        assert len(manager.list_profiles("anthropic")) == 1
        assert len(manager.list_profiles("openai")) == 1

    def test_register_with_labels(self, manager):
        manager.register("anthropic", ["key1", "key2"], labels=["Primary", "Secondary"])
        profiles = manager.list_profiles("anthropic")
        labels = [p.label for p in profiles]
        assert "Primary" in labels
        assert "Secondary" in labels


# ── get_next_key ──────────────────────────────────────────────────────────────

class TestGetNextKey:
    @pytest.mark.asyncio
    async def test_single_key_returned(self, manager):
        manager.register("anthropic", ["key1"])
        key = await manager.get_next_key("anthropic")
        assert key == "key1"

    @pytest.mark.asyncio
    async def test_no_provider_returns_none(self, manager):
        key = await manager.get_next_key("anthropic")
        assert key is None

    @pytest.mark.asyncio
    async def test_round_robin_two_keys(self, manager):
        manager.register("anthropic", ["key1", "key2"])
        k1 = await manager.get_next_key("anthropic")
        k2 = await manager.get_next_key("anthropic")
        k3 = await manager.get_next_key("anthropic")
        assert k1 != k2
        assert k3 == k1  # wraps around

    @pytest.mark.asyncio
    async def test_cooldown_key_skipped(self, manager):
        manager.register("anthropic", ["key1", "key2"])
        # Put key1 on cooldown
        profile1 = manager._find("anthropic", "key1")
        profile1.stats.cooldown_until = time.time() + 9999

        # Should always return key2
        for _ in range(5):
            key = await manager.get_next_key("anthropic")
            assert key == "key2"

    @pytest.mark.asyncio
    async def test_all_keys_in_cooldown_returns_least_cooled(self, manager):
        manager.register("anthropic", ["key1", "key2"])
        now = time.time()
        p1 = manager._find("anthropic", "key1")
        p2 = manager._find("anthropic", "key2")
        p1.stats.cooldown_until = now + 100
        p2.stats.cooldown_until = now + 200

        # Both in cooldown — should still return something
        key = await manager.get_next_key("anthropic")
        assert key in ("key1", "key2")

    @pytest.mark.asyncio
    async def test_total_calls_incremented(self, manager):
        manager.register("anthropic", ["key1"])
        await manager.get_next_key("anthropic")
        await manager.get_next_key("anthropic")
        profile = manager._find("anthropic", "key1")
        assert profile.stats.total_calls == 2


# ── report_success / report_failure ──────────────────────────────────────────

class TestReportSuccess:
    @pytest.mark.asyncio
    async def test_clears_consecutive_failures(self, manager):
        manager.register("anthropic", ["key1"])
        p = manager._find("anthropic", "key1")
        p.stats.consecutive_failures = 3
        p.stats.cooldown_until = time.time() + 999

        await manager.report_success("anthropic", "key1")
        assert p.stats.consecutive_failures == 0
        assert p.stats.cooldown_until == 0.0

    @pytest.mark.asyncio
    async def test_unknown_key_no_crash(self, manager):
        await manager.report_success("anthropic", "nonexistent-key")


class TestReportFailure:
    @pytest.mark.asyncio
    async def test_first_failure_applies_shortest_cooldown(self, manager):
        manager.register("anthropic", ["key1"])
        before = time.time()
        await manager.report_failure("anthropic", "key1", "rate_limit")
        p = manager._find("anthropic", "key1")
        assert p.stats.consecutive_failures == 1
        assert p.stats.total_failures == 1
        # 1st cooldown = 30s
        assert p.stats.cooldown_until >= before + 25  # allow small timing slack

    @pytest.mark.asyncio
    async def test_consecutive_failures_escalate_cooldown(self, manager):
        manager.register("anthropic", ["key1"])
        await manager.report_failure("anthropic", "key1", "unknown")
        await manager.report_failure("anthropic", "key1", "unknown")
        p = manager._find("anthropic", "key1")
        assert p.stats.consecutive_failures == 2
        # 2nd cooldown = 60s
        assert p.stats.cooldown_until >= time.time() + 50

    @pytest.mark.asyncio
    async def test_max_cooldown_capped(self, manager):
        manager.register("anthropic", ["key1"])
        for _ in range(20):  # exceed schedule length
            await manager.report_failure("anthropic", "key1", "unknown")
        p = manager._find("anthropic", "key1")
        # Should not exceed max schedule value (1800s)
        assert p.stats.cooldown_until <= time.time() + 1800 + 5

    @pytest.mark.asyncio
    async def test_failure_reason_recorded(self, manager):
        manager.register("anthropic", ["key1"])
        await manager.report_failure("anthropic", "key1", "auth_error")
        p = manager._find("anthropic", "key1")
        assert p.stats.last_failure_reason == "auth_error"

    @pytest.mark.asyncio
    async def test_unknown_key_no_crash(self, manager):
        await manager.report_failure("anthropic", "nonexistent-key", "unknown")


# ── is_healthy ────────────────────────────────────────────────────────────────

class TestIsHealthy:
    def test_fresh_key_is_healthy(self, manager):
        manager.register("anthropic", ["key1"])
        assert manager.is_healthy("anthropic", "key1") is True

    def test_cooled_key_not_healthy(self, manager):
        manager.register("anthropic", ["key1"])
        p = manager._find("anthropic", "key1")
        p.stats.cooldown_until = time.time() + 999
        assert manager.is_healthy("anthropic", "key1") is False

    def test_expired_cooldown_is_healthy(self, manager):
        manager.register("anthropic", ["key1"])
        p = manager._find("anthropic", "key1")
        p.stats.cooldown_until = time.time() - 1  # past
        assert manager.is_healthy("anthropic", "key1") is True

    def test_unknown_key_returns_false(self, manager):
        assert manager.is_healthy("anthropic", "nonexistent") is False


# ── stats_summary ─────────────────────────────────────────────────────────────

class TestStatsSummary:
    def test_empty_provider(self, manager):
        summary = manager.stats_summary("anthropic")
        assert summary["total"] == 0
        assert summary["healthy"] == 0

    def test_all_healthy(self, manager):
        manager.register("anthropic", ["k1", "k2"])
        summary = manager.stats_summary("anthropic")
        assert summary["total"] == 2
        assert summary["healthy"] == 2
        assert summary["in_cooldown"] == 0

    def test_partial_cooldown(self, manager):
        manager.register("anthropic", ["k1", "k2"])
        p = manager._find("anthropic", "k1")
        p.stats.cooldown_until = time.time() + 999
        summary = manager.stats_summary("anthropic")
        assert summary["healthy"] == 1
        assert summary["in_cooldown"] == 1


# ── register_keys_from_config ─────────────────────────────────────────────────

class TestRegisterKeysFromConfig:
    def test_registers_from_config(self, tmp_path):
        store_path = tmp_path / "auth_profiles.json"
        with patch("liteagent.auth_profiles._STORE_PATH", store_path):
            with patch("liteagent.auth_profiles._manager", None):
                config = {
                    "providers": {
                        "anthropic": {
                            "api_key_env": "ANTHROPIC_API_KEY",
                            "api_keys": ["key1", "key2"],
                        }
                    }
                }
                register_keys_from_config(config)
                from liteagent.auth_profiles import get_manager
                mgr = get_manager()
                profiles = mgr.list_profiles("anthropic")
                assert len(profiles) == 2

    def test_empty_providers_no_crash(self, tmp_path):
        store_path = tmp_path / "auth_profiles.json"
        with patch("liteagent.auth_profiles._STORE_PATH", store_path):
            with patch("liteagent.auth_profiles._manager", None):
                register_keys_from_config({})
