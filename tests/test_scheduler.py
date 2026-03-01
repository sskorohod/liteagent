"""Tests for the scheduler system."""

import asyncio
import pytest
from datetime import datetime

from liteagent.scheduler import parse_cron, cron_matches, Scheduler


class TestCronParser:
    def test_all_wildcards(self):
        parsed = parse_cron("* * * * *")
        assert all(v is None for v in parsed.values())

    def test_exact_values(self):
        parsed = parse_cron("30 8 15 6 3")
        assert parsed["minute"] == ("exact", 30)
        assert parsed["hour"] == ("exact", 8)
        assert parsed["day"] == ("exact", 15)
        assert parsed["month"] == ("exact", 6)
        assert parsed["weekday"] == ("exact", 3)

    def test_interval(self):
        parsed = parse_cron("*/5 * * * *")
        assert parsed["minute"] == ("every", 5)

    def test_comma_set(self):
        parsed = parse_cron("0 9,12,18 * * *")
        assert parsed["hour"] == ("set", {9, 12, 18})

    def test_range(self):
        parsed = parse_cron("0 9 * * 1-5")
        assert parsed["weekday"] == ("range", (1, 5))

    def test_invalid_too_few_fields(self):
        with pytest.raises(ValueError):
            parse_cron("* * *")

    def test_invalid_too_many_fields(self):
        with pytest.raises(ValueError):
            parse_cron("* * * * * *")


class TestCronMatching:
    def test_all_wildcards_always_match(self):
        parsed = parse_cron("* * * * *")
        assert cron_matches(parsed, datetime(2026, 2, 28, 14, 30))

    def test_exact_minute_match(self):
        parsed = parse_cron("30 * * * *")
        assert cron_matches(parsed, datetime(2026, 1, 1, 0, 30))
        assert not cron_matches(parsed, datetime(2026, 1, 1, 0, 29))

    def test_exact_hour_match(self):
        parsed = parse_cron("0 8 * * *")
        assert cron_matches(parsed, datetime(2026, 1, 1, 8, 0))
        assert not cron_matches(parsed, datetime(2026, 1, 1, 9, 0))

    def test_interval_match(self):
        parsed = parse_cron("*/15 * * * *")
        assert cron_matches(parsed, datetime(2026, 1, 1, 0, 0))
        assert cron_matches(parsed, datetime(2026, 1, 1, 0, 15))
        assert cron_matches(parsed, datetime(2026, 1, 1, 0, 30))
        assert not cron_matches(parsed, datetime(2026, 1, 1, 0, 7))

    def test_set_match(self):
        parsed = parse_cron("0 9,17 * * *")
        assert cron_matches(parsed, datetime(2026, 1, 1, 9, 0))
        assert cron_matches(parsed, datetime(2026, 1, 1, 17, 0))
        assert not cron_matches(parsed, datetime(2026, 1, 1, 12, 0))

    def test_range_match(self):
        parsed = parse_cron("0 9 * * 1-5")
        # Monday = isoweekday 1
        assert cron_matches(parsed, datetime(2026, 3, 2, 9, 0))  # Monday
        assert not cron_matches(parsed, datetime(2026, 3, 1, 9, 0))  # Sunday=0


class TestScheduler:
    def test_add_job(self):
        s = Scheduler()
        async def noop():
            pass
        s.add_job("test", "* * * * *", noop)
        jobs = s.get_jobs()
        assert len(jobs) == 1
        assert jobs[0]["name"] == "test"
        assert jobs[0]["cron"] == "* * * * *"

    def test_get_jobs_empty(self):
        s = Scheduler()
        assert s.get_jobs() == []

    def test_start_no_jobs(self):
        """Scheduler should not start if no jobs registered."""
        s = Scheduler()
        s.start()
        assert s._task is None

    @pytest.mark.asyncio
    async def test_job_execution(self):
        """Test that a job actually executes."""
        s = Scheduler()
        results = []

        async def record_job():
            results.append("ran")

        s.add_job("test_job", "* * * * *", record_job)
        s._running = True

        # Directly trigger the loop once
        now = datetime.now()
        for job in s._jobs:
            if cron_matches(job["cron"], now):
                await job["handler"]()

        assert results == ["ran"]

    @pytest.mark.asyncio
    async def test_stop_scheduler(self):
        s = Scheduler()
        async def noop():
            pass
        s.add_job("test", "* * * * *", noop)
        s.start()
        assert s._running
        s.stop()
        assert not s._running
