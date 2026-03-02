"""Tests for per-user queue and locking."""
import asyncio
import itertools
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from liteagent.agent import LiteAgent


class TestConcurrencyLocks:
    """Test per-user request serialization."""

    @pytest.fixture(autouse=True)
    def reset_class_state(self):
        """Reset class-level lock state between tests."""
        LiteAgent._user_locks = {}
        LiteAgent._locks_guard = None
        LiteAgent._requests_lock = None
        LiteAgent._provider_lock = None
        LiteAgent._active_requests = {}
        LiteAgent._queued_requests = {}
        LiteAgent._request_counter = itertools.count(1)
        LiteAgent._queue_counter = itertools.count(1)
        LiteAgent._ws_hub = None
        yield
        LiteAgent._user_locks = {}
        LiteAgent._locks_guard = None
        LiteAgent._requests_lock = None
        LiteAgent._provider_lock = None
        LiteAgent._active_requests = {}
        LiteAgent._queued_requests = {}
        LiteAgent._request_counter = itertools.count(1)
        LiteAgent._queue_counter = itertools.count(1)

    async def test_ensure_locks_creates_locks(self):
        LiteAgent._ensure_locks()
        assert LiteAgent._locks_guard is not None
        assert LiteAgent._requests_lock is not None
        assert LiteAgent._provider_lock is not None

    async def test_ensure_locks_idempotent(self):
        LiteAgent._ensure_locks()
        guard = LiteAgent._locks_guard
        LiteAgent._ensure_locks()
        assert LiteAgent._locks_guard is guard

    async def test_get_user_lock_creates_lock(self):
        LiteAgent._ensure_locks()
        agent = MagicMock(spec=LiteAgent)
        agent._get_user_lock = LiteAgent._get_user_lock.__get__(agent)
        lock = await agent._get_user_lock("user1")
        assert isinstance(lock, asyncio.Lock)
        assert "user1" in LiteAgent._user_locks

    async def test_get_user_lock_returns_same(self):
        LiteAgent._ensure_locks()
        agent = MagicMock(spec=LiteAgent)
        agent._get_user_lock = LiteAgent._get_user_lock.__get__(agent)
        lock1 = await agent._get_user_lock("user1")
        lock2 = await agent._get_user_lock("user1")
        assert lock1 is lock2

    async def test_different_users_get_different_locks(self):
        LiteAgent._ensure_locks()
        agent = MagicMock(spec=LiteAgent)
        agent._get_user_lock = LiteAgent._get_user_lock.__get__(agent)
        lock1 = await agent._get_user_lock("user1")
        lock2 = await agent._get_user_lock("user2")
        assert lock1 is not lock2

    def test_track_queued(self):
        agent = MagicMock(spec=LiteAgent)
        agent._track_queued = LiteAgent._track_queued.__get__(agent)
        agent._ws_broadcast = MagicMock()
        q_id = agent._track_queued("user1")
        assert q_id in LiteAgent._queued_requests
        assert LiteAgent._queued_requests[q_id]["user_id"] == "user1"

    def test_untrack_queued(self):
        agent = MagicMock(spec=LiteAgent)
        agent._track_queued = LiteAgent._track_queued.__get__(agent)
        agent._untrack_queued = LiteAgent._untrack_queued.__get__(agent)
        agent._ws_broadcast = MagicMock()
        q_id = agent._track_queued("user1")
        agent._untrack_queued(q_id)
        assert q_id not in LiteAgent._queued_requests

    async def test_track_request_async(self):
        LiteAgent._ensure_locks()
        agent = MagicMock(spec=LiteAgent)
        agent._track_request_start = LiteAgent._track_request_start.__get__(agent)
        agent._track_request_end = LiteAgent._track_request_end.__get__(agent)
        agent._ws_broadcast = MagicMock()

        req_id = await agent._track_request_start("user1", "hello", "model-x")
        assert req_id in LiteAgent._active_requests
        assert LiteAgent._active_requests[req_id]["user_id"] == "user1"

        await agent._track_request_end(req_id)
        assert req_id not in LiteAgent._active_requests

    def test_get_active_requests(self):
        LiteAgent._active_requests = {1: {"id": 1, "user_id": "u1"}}
        result = LiteAgent.get_active_requests()
        assert len(result) == 1
        assert result[0]["user_id"] == "u1"

    def test_get_queued_requests(self):
        LiteAgent._queued_requests = {1: {"id": 1, "user_id": "u1"}}
        result = LiteAgent.get_queued_requests()
        assert len(result) == 1
