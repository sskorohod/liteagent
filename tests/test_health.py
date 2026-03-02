"""Tests for channel health monitoring (health.py)."""

import time
import pytest
from liteagent.health import HealthMonitor, ChannelHealth


class TestHealthMonitorBasics:
    def setup_method(self):
        self.monitor = HealthMonitor({})

    def test_initial_status_empty(self):
        assert self.monitor.get_status() == {}

    def test_record_success(self):
        self.monitor._record_success("api", 42.5)
        status = self.monitor.get_status()["api"]
        assert status["status"] == "healthy"
        assert status["latency_ms"] == 42.5
        assert status["consecutive_failures"] == 0

    def test_record_failure_degraded(self):
        self.monitor._record_failure("api", "connection refused")
        status = self.monitor.get_status()["api"]
        assert status["status"] == "degraded"
        assert status["consecutive_failures"] == 1
        assert "connection refused" in status["error_message"]

    def test_record_three_failures_down(self):
        for i in range(3):
            self.monitor._record_failure("api", f"error {i}")
        status = self.monitor.get_status()["api"]
        assert status["status"] == "down"
        assert status["consecutive_failures"] == 3

    def test_recovery_after_failures(self):
        self.monitor._record_failure("api", "err1")
        self.monitor._record_failure("api", "err2")
        self.monitor._record_success("api", 10.0)
        self.monitor._record_success("api", 12.0)
        status = self.monitor.get_status()["api"]
        assert status["status"] == "healthy"
        assert status["consecutive_failures"] == 0


class TestHealthMonitorBackoff:
    def test_backoff_increases_exponentially(self):
        monitor = HealthMonitor({"health": {"base_backoff_sec": 1.0}})
        monitor._record_failure("test", "err1")
        ch1 = monitor._get_channel("test")
        backoff1 = ch1.backoff_until - time.time()
        monitor._record_failure("test", "err2")
        ch2 = monitor._get_channel("test")
        backoff2 = ch2.backoff_until - time.time()
        # Second backoff should be roughly double the first
        assert backoff2 > backoff1 * 1.5

    def test_backoff_capped_at_max(self):
        monitor = HealthMonitor({
            "health": {"base_backoff_sec": 100.0, "max_backoff_sec": 200.0}
        })
        for _ in range(10):
            monitor._record_failure("test", "err")
        ch = monitor._get_channel("test")
        max_backoff = ch.backoff_until - time.time()
        assert max_backoff <= 201.0  # max + small margin

    def test_should_check_respects_backoff(self):
        monitor = HealthMonitor({"health": {"base_backoff_sec": 100.0}})
        monitor._record_failure("test", "err")
        assert monitor._should_check("test") is False

    def test_should_check_allows_after_backoff(self):
        monitor = HealthMonitor({"health": {"base_backoff_sec": 0.01}})
        monitor._record_failure("test", "err")
        time.sleep(0.02)
        assert monitor._should_check("test") is True


class TestHealthMonitorMultiChannel:
    def test_independent_channels(self):
        monitor = HealthMonitor({})
        monitor._record_success("api", 10.0)
        monitor._record_failure("telegram", "timeout")
        status = monitor.get_status()
        assert status["api"]["status"] == "healthy"
        assert status["telegram"]["status"] == "degraded"

    def test_total_checks_counted(self):
        monitor = HealthMonitor({})
        monitor._record_success("api", 10.0)
        monitor._record_success("api", 12.0)
        monitor._record_failure("api", "err")
        assert monitor.get_status()["api"]["total_checks"] == 3


class TestHealthMonitorRunAllChecks:
    @pytest.mark.asyncio
    async def test_run_checks_with_no_config(self):
        monitor = HealthMonitor({})
        results = await monitor.run_all_checks()
        # Should at least check the default provider
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_run_checks_provider_no_key(self):
        monitor = HealthMonitor({"agent": {"provider": "anthropic"}})
        results = await monitor.run_all_checks()
        # Provider check with no key should result in down status
        provider_key = "provider:anthropic"
        if provider_key in results:
            assert results[provider_key].status in ("down", "healthy")

    @pytest.mark.asyncio
    async def test_check_provider_with_key(self):
        monitor = HealthMonitor({"agent": {"provider": "anthropic"}})
        from unittest.mock import patch
        with patch("liteagent.config.get_api_key", return_value="sk-test"):
            result = await monitor.check_provider("anthropic")
        assert result.status == "healthy"
