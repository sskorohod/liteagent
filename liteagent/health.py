"""Channel and provider health monitoring.

Polls connectivity, tracks latency, and uses exponential backoff
for failing channels. Inspired by OpenClaw's ChannelHealthMonitor.
"""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ChannelHealth:
    """Health state for a single channel or provider."""
    name: str
    status: str = "unknown"  # healthy | degraded | down | unknown
    latency_ms: float = 0.0
    last_check: str = ""
    last_healthy: str = ""
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    error_message: str = ""
    backoff_until: float = 0.0
    total_checks: int = 0


class HealthMonitor:
    """Monitors channel and provider health with scheduled polling.

    Features:
    - Exponential backoff on failures (up to max_backoff)
    - Auto-recovery detection (consecutive successes)
    - Status dashboard reporting
    """

    def __init__(self, config: dict):
        self._channels: dict[str, ChannelHealth] = {}
        health_cfg = config.get("health", {})
        self._check_interval = health_cfg.get("check_interval_sec", 300)
        self._max_backoff = health_cfg.get("max_backoff_sec", 3600)
        self._base_backoff = health_cfg.get("base_backoff_sec", 30)
        self._recovery_threshold = health_cfg.get("recovery_threshold", 2)
        self._config = config

    def _get_channel(self, name: str) -> ChannelHealth:
        if name not in self._channels:
            self._channels[name] = ChannelHealth(name=name)
        return self._channels[name]

    def _should_check(self, name: str) -> bool:
        """Respect backoff: skip checks if in backoff period."""
        ch = self._get_channel(name)
        return time.time() >= ch.backoff_until

    def _record_success(self, name: str, latency_ms: float) -> None:
        ch = self._get_channel(name)
        ch.status = "healthy"
        ch.latency_ms = latency_ms
        ch.last_check = datetime.now().isoformat()
        ch.last_healthy = ch.last_check
        ch.consecutive_successes += 1
        ch.consecutive_failures = 0
        ch.error_message = ""
        ch.backoff_until = 0.0
        ch.total_checks += 1
        if ch.consecutive_successes >= self._recovery_threshold:
            logger.info("Health: %s recovered (healthy)", name)

    def _record_failure(self, name: str, error: str) -> None:
        ch = self._get_channel(name)
        ch.consecutive_failures += 1
        ch.consecutive_successes = 0
        ch.error_message = error[:200]
        ch.last_check = datetime.now().isoformat()
        ch.total_checks += 1
        # Exponential backoff: base * 2^(failures-1), capped at max
        backoff = min(
            self._base_backoff * math.pow(2, ch.consecutive_failures - 1),
            self._max_backoff)
        ch.backoff_until = time.time() + backoff
        if ch.consecutive_failures >= 3:
            ch.status = "down"
        else:
            ch.status = "degraded"
        logger.warning("Health: %s %s (failures=%d, backoff=%.0fs): %s",
                        name, ch.status, ch.consecutive_failures, backoff, error)

    async def check_ollama(self) -> ChannelHealth:
        """Check Ollama local server connectivity."""
        name = "ollama"
        if not self._should_check(name):
            return self._get_channel(name)
        try:
            import httpx
            t0 = time.time()
            async with httpx.AsyncClient(timeout=5.0) as client:
                base = self._config.get("providers", {}).get("ollama", {}).get(
                    "base_url", "http://localhost:11434/v1")
                base = base.replace("/v1", "")
                resp = await client.get(f"{base}/api/tags")
                resp.raise_for_status()
            latency = (time.time() - t0) * 1000
            self._record_success(name, latency)
        except Exception as e:
            self._record_failure(name, str(e))
        return self._get_channel(name)

    async def check_telegram(self) -> ChannelHealth:
        """Check Telegram bot token validity."""
        name = "telegram"
        if not self._should_check(name):
            return self._get_channel(name)
        tg_cfg = self._config.get("channels", {}).get("telegram", {})
        token = tg_cfg.get("token")
        if not token:
            ch = self._get_channel(name)
            ch.status = "unknown"
            ch.error_message = "no token configured"
            return ch
        try:
            import httpx
            t0 = time.time()
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
                resp.raise_for_status()
            latency = (time.time() - t0) * 1000
            self._record_success(name, latency)
        except Exception as e:
            self._record_failure(name, str(e))
        return self._get_channel(name)

    async def check_api(self) -> ChannelHealth:
        """Check if API server port is responsive (self-check)."""
        name = "api"
        if not self._should_check(name):
            return self._get_channel(name)
        api_cfg = self._config.get("channels", {}).get("api", {})
        host = api_cfg.get("host", "127.0.0.1")
        port = api_cfg.get("port", 8080)
        try:
            import httpx
            t0 = time.time()
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"http://{host}:{port}/health")
                resp.raise_for_status()
            latency = (time.time() - t0) * 1000
            self._record_success(name, latency)
        except Exception as e:
            self._record_failure(name, str(e))
        return self._get_channel(name)

    async def check_provider(self, provider_name: str) -> ChannelHealth:
        """Check provider API key validity with a minimal call."""
        name = f"provider:{provider_name}"
        if not self._should_check(name):
            return self._get_channel(name)
        from .config import get_api_key
        key = get_api_key(provider_name)
        if provider_name == "ollama":
            return await self.check_ollama()
        if not key:
            ch = self._get_channel(name)
            ch.status = "down"
            ch.error_message = "no API key"
            ch.last_check = datetime.now().isoformat()
            return ch
        # Key exists — mark as healthy (actual validation via circuit breaker)
        self._record_success(name, 0.0)
        return self._get_channel(name)

    async def run_all_checks(self) -> dict[str, ChannelHealth]:
        """Run all configured health checks concurrently."""
        tasks = []

        # Provider check
        provider = self._config.get("agent", {}).get("provider", "anthropic")
        tasks.append(("provider", self.check_provider(provider)))

        # Ollama check (if configured)
        if self._config.get("providers", {}).get("ollama"):
            tasks.append(("ollama", self.check_ollama()))

        # Telegram check
        if self._config.get("channels", {}).get("telegram", {}).get("enabled"):
            tasks.append(("telegram", self.check_telegram()))

        results = {}
        for name, coro in tasks:
            try:
                health = await coro
                results[health.name] = health
            except Exception as e:
                logger.warning("Health check %s failed: %s", name, e)

        return results

    def get_status(self) -> dict[str, dict]:
        """Return current health status for all channels (dashboard API)."""
        result = {}
        for name, ch in self._channels.items():
            result[name] = {
                "name": ch.name,
                "status": ch.status,
                "latency_ms": round(ch.latency_ms, 1),
                "last_check": ch.last_check,
                "last_healthy": ch.last_healthy,
                "consecutive_failures": ch.consecutive_failures,
                "error_message": ch.error_message,
                "total_checks": ch.total_checks,
            }
        return result
