"""Circuit breaker pattern for LLM provider resilience.

Tracks provider health and automatically routes around failing providers.
Inspired by OpenClaw's auth profile rotation with failure tracking.

States:
    closed    — normal operation, calls go through
    open      — provider failing, all calls bypass to fallback
    half_open — testing recovery, limited calls allowed
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProviderState:
    """Tracks health state for a single LLM provider."""
    name: str
    failure_count: int = 0
    success_count: int = 0
    state: str = "closed"  # closed | open | half_open
    last_failure_at: float = 0.0
    last_success_at: float = 0.0
    cooldown_until: float = 0.0
    total_calls: int = 0
    consecutive_failures: int = 0
    error_history: list = field(default_factory=list)


class CircuitBreaker:
    """Circuit breaker for LLM provider calls.

    Args:
        failure_threshold: Consecutive failures before opening circuit.
        recovery_timeout: Seconds before retrying an open circuit (half_open).
        half_open_max_calls: Max calls allowed in half_open state before
            deciding to close (recovered) or re-open (still failing).
        max_error_history: Max recent errors to keep per provider.
    """

    def __init__(self, failure_threshold: int = 3,
                 recovery_timeout: float = 300.0,
                 half_open_max_calls: int = 1,
                 max_error_history: int = 10):
        self._states: dict[str, ProviderState] = {}
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.max_error_history = max_error_history

    def _get_state(self, provider: str) -> ProviderState:
        if provider not in self._states:
            self._states[provider] = ProviderState(name=provider)
        return self._states[provider]

    def record_success(self, provider: str) -> None:
        """Record a successful API call. Resets failure counters."""
        st = self._get_state(provider)
        st.success_count += 1
        st.total_calls += 1
        st.consecutive_failures = 0
        st.last_success_at = time.time()
        if st.state in ("open", "half_open"):
            logger.info("Circuit breaker: %s recovered (closed)", provider)
            st.state = "closed"
            st.cooldown_until = 0.0

    def record_failure(self, provider: str, error: Exception) -> None:
        """Record a failed API call. May open the circuit."""
        st = self._get_state(provider)
        st.failure_count += 1
        st.total_calls += 1
        st.consecutive_failures += 1
        st.last_failure_at = time.time()
        st.error_history.append({
            "error": str(error)[:200],
            "type": type(error).__name__,
            "at": time.time(),
        })
        if len(st.error_history) > self.max_error_history:
            st.error_history = st.error_history[-self.max_error_history:]

        if st.state == "half_open":
            # Failed during recovery test — re-open
            st.state = "open"
            st.cooldown_until = time.time() + self.recovery_timeout
            logger.warning("Circuit breaker: %s failed recovery, re-opened "
                           "(cooldown %.0fs)", provider, self.recovery_timeout)
        elif (st.state == "closed"
              and st.consecutive_failures >= self.failure_threshold):
            st.state = "open"
            st.cooldown_until = time.time() + self.recovery_timeout
            logger.warning("Circuit breaker: %s opened after %d consecutive "
                           "failures (cooldown %.0fs)",
                           provider, st.consecutive_failures,
                           self.recovery_timeout)

    def can_call(self, provider: str) -> bool:
        """Check if a provider is available for calls.

        Returns True for:
        - closed state (normal)
        - half_open state (testing recovery, limited calls)
        - open state past cooldown (transitions to half_open)
        """
        st = self._get_state(provider)
        if st.state == "closed":
            return True
        if st.state == "half_open":
            return True
        # open — check if cooldown elapsed
        if time.time() >= st.cooldown_until:
            st.state = "half_open"
            logger.info("Circuit breaker: %s cooldown elapsed, "
                        "trying half_open", provider)
            return True
        return False

    def get_status(self) -> dict[str, dict]:
        """Return health status for all tracked providers (for dashboard)."""
        result = {}
        for name, st in self._states.items():
            result[name] = {
                "state": st.state,
                "total_calls": st.total_calls,
                "success_count": st.success_count,
                "failure_count": st.failure_count,
                "consecutive_failures": st.consecutive_failures,
                "last_failure_at": st.last_failure_at,
                "last_success_at": st.last_success_at,
                "cooldown_until": st.cooldown_until,
                "recent_errors": st.error_history[-3:],
            }
        return result

    def reset(self, provider: str | None = None) -> None:
        """Reset circuit breaker state. If provider is None, resets all."""
        if provider:
            self._states.pop(provider, None)
        else:
            self._states.clear()
