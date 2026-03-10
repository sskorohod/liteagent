"""Auth profiles — multiple API keys per provider with round-robin + cooldown.

Ported from OpenClaw auth-profiles.ts.

Supports:
- Multiple API keys per provider (rotate automatically on failure)
- Per-key cooldown after failure (exponential back-off)
- Round-robin scheduling when all keys are healthy
- Persistent state stored in ~/.liteagent/auth_profiles.json

Usage:
    from liteagent.auth_profiles import AuthProfileManager, get_next_api_key

    # Get the next available key for a provider
    key = await get_next_api_key("anthropic")

    # Report failure (triggers cooldown)
    await report_key_failure("anthropic", key, "rate_limit")

    # Report success (resets consecutive failures)
    await report_key_success("anthropic", key)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_STORE_PATH = Path("~/.liteagent/auth_profiles.json").expanduser()

# Cooldown schedule (seconds) indexed by consecutive failure count
_COOLDOWN_SCHEDULE: list[float] = [30, 60, 120, 300, 600, 1800]

FailureReason = Literal["rate_limit", "auth_error", "quota", "network", "unknown"]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ProfileStats:
    total_calls: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    last_used_at: float = 0.0
    last_failure_at: float = 0.0
    cooldown_until: float = 0.0
    last_failure_reason: str = ""


@dataclass
class AuthProfile:
    provider: str
    profile_id: str     # stable ID (e.g. "anthropic-key-2")
    api_key: str
    label: str = ""
    enabled: bool = True
    stats: ProfileStats = field(default_factory=ProfileStats)


@dataclass
class AuthProfileStore:
    profiles: list[AuthProfile] = field(default_factory=list)
    round_robin_index: dict[str, int] = field(default_factory=dict)


# ── Persistence ───────────────────────────────────────────────────────────────

def _load_store() -> AuthProfileStore:
    if not _STORE_PATH.exists():
        return AuthProfileStore()
    try:
        raw = json.loads(_STORE_PATH.read_text())
        profiles = []
        for p in raw.get("profiles", []):
            stats_raw = p.pop("stats", {})
            stats = ProfileStats(**{k: v for k, v in stats_raw.items()
                                    if k in ProfileStats.__dataclass_fields__})
            profiles.append(AuthProfile(stats=stats, **{
                k: v for k, v in p.items() if k in AuthProfile.__dataclass_fields__
            }))
        store = AuthProfileStore(
            profiles=profiles,
            round_robin_index=raw.get("round_robin_index", {}),
        )
        return store
    except Exception as exc:
        logger.warning("Failed to load auth_profiles.json: %s", exc)
        return AuthProfileStore()


def _save_store(store: AuthProfileStore) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "profiles": [
            {**{k: v for k, v in asdict(p).items() if k != "stats"},
             "stats": asdict(p.stats)}
            for p in store.profiles
        ],
        "round_robin_index": store.round_robin_index,
    }
    tmp = _STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(_STORE_PATH)


# ── Manager ───────────────────────────────────────────────────────────────────

class AuthProfileManager:
    """Manages multiple API keys per provider with round-robin + cooldown."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._store: AuthProfileStore = _load_store()

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        provider: str,
        api_keys: list[str] | str,
        labels: list[str] | None = None,
    ) -> None:
        """Register one or more API keys for a provider.

        Existing profiles are preserved; new keys are appended.
        """
        if isinstance(api_keys, str):
            api_keys = [api_keys]

        existing_keys = {p.api_key for p in self._store.profiles
                         if p.provider == provider}

        for i, key in enumerate(api_keys):
            if key in existing_keys:
                continue
            label = (labels[i] if labels and i < len(labels)
                     else f"{provider}-key-{i + 1}")
            profile_id = f"{provider}-{i}"
            self._store.profiles.append(AuthProfile(
                provider=provider,
                profile_id=profile_id,
                api_key=key,
                label=label,
            ))

        _save_store(self._store)

    # ── Key retrieval ─────────────────────────────────────────────────────

    async def get_next_key(self, provider: str) -> str | None:
        """Return the next available (non-cooldown) key for *provider*.

        Uses round-robin among healthy keys.  Falls back to the least-cooled
        key if all are in cooldown.
        """
        async with self._lock:
            candidates = [
                p for p in self._store.profiles
                if p.provider == provider and p.enabled and p.api_key
            ]
            if not candidates:
                return None

            now = time.time()
            healthy = [c for c in candidates
                       if c.stats.cooldown_until <= now]

            pool = healthy if healthy else candidates

            # Round-robin within pool
            rr_key = f"{provider}"
            idx = self._store.round_robin_index.get(rr_key, 0)
            idx = idx % len(pool)
            chosen = pool[idx]
            self._store.round_robin_index[rr_key] = (idx + 1) % len(pool)

            # Update usage stats
            chosen.stats.last_used_at = now
            chosen.stats.total_calls += 1
            _save_store(self._store)

            return chosen.api_key

    async def report_success(self, provider: str, api_key: str) -> None:
        """Reset consecutive failures for *api_key*."""
        async with self._lock:
            profile = self._find(provider, api_key)
            if profile:
                profile.stats.consecutive_failures = 0
                profile.stats.cooldown_until = 0.0
                _save_store(self._store)

    async def report_failure(
        self,
        provider: str,
        api_key: str,
        reason: FailureReason = "unknown",
    ) -> None:
        """Increment failure counter and apply cooldown."""
        async with self._lock:
            profile = self._find(provider, api_key)
            if not profile:
                return
            stats = profile.stats
            stats.total_failures += 1
            stats.consecutive_failures += 1
            stats.last_failure_at = time.time()
            stats.last_failure_reason = reason

            # Apply cooldown
            idx = min(stats.consecutive_failures - 1, len(_COOLDOWN_SCHEDULE) - 1)
            cooldown_s = _COOLDOWN_SCHEDULE[idx]
            stats.cooldown_until = time.time() + cooldown_s

            logger.warning(
                "Auth profile %s/%s failure #%d (%s) — cooldown %.0fs",
                provider, profile.label or api_key[:8],
                stats.consecutive_failures, reason, cooldown_s)

            _save_store(self._store)

    def list_profiles(self, provider: str) -> list[AuthProfile]:
        """Return all registered profiles for *provider*."""
        return [p for p in self._store.profiles if p.provider == provider]

    def is_healthy(self, provider: str, api_key: str) -> bool:
        """Return True if *api_key* is not in cooldown."""
        profile = self._find(provider, api_key)
        if not profile:
            return False
        return profile.stats.cooldown_until <= time.time()

    def stats_summary(self, provider: str) -> dict:
        """Return a summary dict for dashboard display."""
        profiles = self.list_profiles(provider)
        now = time.time()
        return {
            "total": len(profiles),
            "healthy": sum(1 for p in profiles if p.stats.cooldown_until <= now),
            "in_cooldown": sum(1 for p in profiles if p.stats.cooldown_until > now),
            "profiles": [
                {
                    "label": p.label,
                    "healthy": p.stats.cooldown_until <= now,
                    "consecutive_failures": p.stats.consecutive_failures,
                    "total_calls": p.stats.total_calls,
                    "cooldown_remaining_s": max(0.0, p.stats.cooldown_until - now),
                }
                for p in profiles
            ],
        }

    # ── helpers ───────────────────────────────────────────────────────────

    def _find(self, provider: str, api_key: str) -> AuthProfile | None:
        for p in self._store.profiles:
            if p.provider == provider and p.api_key == api_key:
                return p
        return None


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: AuthProfileManager | None = None


def get_manager() -> AuthProfileManager:
    global _manager
    if _manager is None:
        _manager = AuthProfileManager()
    return _manager


def register_keys_from_config(config: dict) -> None:
    """Populate the manager from config.providers.<name>.api_keys list."""
    mgr = get_manager()
    providers_cfg = config.get("providers", {})
    for provider, pcfg in providers_cfg.items():
        if not isinstance(pcfg, dict):
            continue
        extra_keys = pcfg.get("api_keys", [])
        if extra_keys:
            labels = pcfg.get("api_key_labels", [])
            mgr.register(provider, extra_keys, labels)


async def get_next_api_key(provider: str) -> str | None:
    return await get_manager().get_next_key(provider)


async def report_key_success(provider: str, api_key: str) -> None:
    await get_manager().report_success(provider, api_key)


async def report_key_failure(
    provider: str,
    api_key: str,
    reason: FailureReason = "unknown",
) -> None:
    await get_manager().report_failure(provider, api_key, reason)
