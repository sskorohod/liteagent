"""Tests for circuit breaker pattern (circuit_breaker.py)."""

import time
import pytest
from unittest.mock import patch
from liteagent.circuit_breaker import CircuitBreaker, ProviderState


class TestCircuitBreakerStates:
    def setup_method(self):
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

    def test_initial_state_is_closed(self):
        assert self.cb.can_call("anthropic") is True
        status = self.cb.get_status()
        assert status["anthropic"]["state"] == "closed"

    def test_success_keeps_closed(self):
        self.cb.record_success("anthropic")
        st = self.cb.get_status()["anthropic"]
        assert st["state"] == "closed"
        assert st["success_count"] == 1

    def test_failures_below_threshold_stay_closed(self):
        for _ in range(2):
            self.cb.record_failure("anthropic", RuntimeError("fail"))
        assert self.cb.can_call("anthropic") is True
        assert self.cb.get_status()["anthropic"]["state"] == "closed"

    def test_failures_at_threshold_opens_circuit(self):
        for i in range(3):
            self.cb.record_failure("anthropic", RuntimeError(f"fail {i}"))
        assert self.cb.get_status()["anthropic"]["state"] == "open"

    def test_open_circuit_blocks_calls(self):
        for _ in range(3):
            self.cb.record_failure("anthropic", RuntimeError("fail"))
        assert self.cb.can_call("anthropic") is False

    def test_open_transitions_to_half_open_after_cooldown(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure("test", RuntimeError("1"))
        cb.record_failure("test", RuntimeError("2"))
        assert cb.get_status()["test"]["state"] == "open"
        time.sleep(0.02)
        assert cb.can_call("test") is True
        assert cb.get_status()["test"]["state"] == "half_open"

    def test_half_open_success_closes_circuit(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure("test", RuntimeError("1"))
        cb.record_failure("test", RuntimeError("2"))
        time.sleep(0.02)
        cb.can_call("test")  # transition to half_open
        cb.record_success("test")
        assert cb.get_status()["test"]["state"] == "closed"

    def test_half_open_failure_reopens_circuit(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        cb.record_failure("test", RuntimeError("1"))
        cb.record_failure("test", RuntimeError("2"))
        time.sleep(0.02)
        cb.can_call("test")  # transition to half_open
        cb.record_failure("test", RuntimeError("3"))
        assert cb.get_status()["test"]["state"] == "open"

    def test_success_resets_consecutive_failures(self):
        self.cb.record_failure("p", RuntimeError("1"))
        self.cb.record_failure("p", RuntimeError("2"))
        self.cb.record_success("p")
        assert self.cb.get_status()["p"]["consecutive_failures"] == 0
        # One more failure should not open (needs 3 consecutive)
        self.cb.record_failure("p", RuntimeError("3"))
        assert self.cb.get_status()["p"]["state"] == "closed"


class TestCircuitBreakerMultiProvider:
    def test_independent_providers(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure("anthropic", RuntimeError("1"))
        cb.record_failure("anthropic", RuntimeError("2"))
        cb.record_success("openai")
        assert cb.can_call("anthropic") is False
        assert cb.can_call("openai") is True

    def test_status_shows_all_providers(self):
        cb = CircuitBreaker()
        cb.record_success("anthropic")
        cb.record_success("openai")
        cb.record_success("ollama")
        status = cb.get_status()
        assert set(status.keys()) == {"anthropic", "openai", "ollama"}


class TestCircuitBreakerErrorHistory:
    def test_error_history_limited(self):
        cb = CircuitBreaker(failure_threshold=100, max_error_history=3)
        for i in range(10):
            cb.record_failure("test", RuntimeError(f"error {i}"))
        errors = cb.get_status()["test"]["recent_errors"]
        assert len(errors) <= 3

    def test_error_contains_type(self):
        cb = CircuitBreaker()
        cb.record_failure("test", ValueError("bad input"))
        err = cb.get_status()["test"]["recent_errors"][0]
        assert err["type"] == "ValueError"
        assert "bad input" in err["error"]


class TestCircuitBreakerReset:
    def test_reset_specific_provider(self):
        cb = CircuitBreaker()
        cb.record_success("anthropic")
        cb.record_success("openai")
        cb.reset("anthropic")
        status = cb.get_status()
        assert "anthropic" not in status
        assert "openai" in status

    def test_reset_all(self):
        cb = CircuitBreaker()
        cb.record_success("anthropic")
        cb.record_success("openai")
        cb.reset()
        assert cb.get_status() == {}


class TestCircuitBreakerTotals:
    def test_total_calls_counted(self):
        cb = CircuitBreaker()
        cb.record_success("test")
        cb.record_success("test")
        cb.record_failure("test", RuntimeError("x"))
        st = cb.get_status()["test"]
        assert st["total_calls"] == 3
        assert st["success_count"] == 2
        assert st["failure_count"] == 1
