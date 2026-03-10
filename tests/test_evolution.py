"""Tests for evolution module: friction detection, style adaptation, proactive agent."""

import sqlite3
import pytest
from datetime import datetime, timedelta

from liteagent.evolution import (
    detect_friction, store_friction, get_active_patches,
    analyze_style, update_style_profile, get_style_instruction,
    detect_patterns, detect_human_support_opportunities,
    get_human_support_snapshot,
    _inputs_similar, synthesize_prompt_patches,
)


@pytest.fixture
def evo_db(tmp_path):
    """Create DB with feature tables."""
    db = sqlite3.connect(str(tmp_path / "evo.db"))
    db.executescript("""
        CREATE TABLE friction_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            user_input TEXT,
            agent_response TEXT,
            extracted_lesson TEXT,
            created_at TEXT
        );
        CREATE TABLE prompt_patches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patch_text TEXT NOT NULL,
            reason TEXT,
            applied INTEGER DEFAULT 0,
            created_at TEXT
        );
        CREATE TABLE style_profiles (
            user_id TEXT PRIMARY KEY,
            formality REAL DEFAULT 0.5,
            verbosity REAL DEFAULT 0.5,
            technical_level REAL DEFAULT 0.5,
            emoji_usage REAL DEFAULT 0.0,
            language TEXT DEFAULT 'en',
            updated_at TEXT
        );
        CREATE TABLE interaction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_input TEXT,
            agent_response TEXT,
            tool_calls_json TEXT,
            success INTEGER DEFAULT 1,
            confidence REAL,
            model_used TEXT,
            created_at TEXT
        );
    """)
    return db


class TestFrictionDetection:
    def test_detects_correction_english(self):
        assert detect_friction("No, I meant the other one") == "correction"

    def test_detects_wrong(self):
        assert detect_friction("That's wrong, try again") == "correction"

    def test_detects_correction_russian(self):
        assert detect_friction("Нет, я имел в виду другое") == "correction"

    def test_detects_rephrase_russian(self):
        assert detect_friction("Неправильно, переделай") == "correction"

    def test_normal_input_returns_none(self):
        assert detect_friction("What's the weather today?") is None

    def test_simple_greeting_returns_none(self):
        assert detect_friction("Hello, how are you?") is None

    def test_store_friction(self, evo_db):
        store_friction(evo_db, "u1", "correction", "wrong answer", "bad response")
        rows = evo_db.execute("SELECT * FROM friction_signals").fetchall()
        assert len(rows) == 1
        assert rows[0][2] == "correction"

    def test_get_active_patches_empty(self, evo_db):
        assert get_active_patches(evo_db) == []

    def test_get_active_patches(self, evo_db):
        evo_db.execute(
            "INSERT INTO prompt_patches VALUES (NULL, 'Be concise', 'test', 1, ?)",
            (datetime.now().isoformat(),))
        evo_db.execute(
            "INSERT INTO prompt_patches VALUES (NULL, 'Pending patch', 'test', 0, ?)",
            (datetime.now().isoformat(),))
        evo_db.commit()
        patches = get_active_patches(evo_db)
        assert patches == ["Be concise"]

    @pytest.mark.asyncio
    async def test_synthesize_prompt_patches_uses_local_agent_model(self, evo_db):
        evo_db.execute(
            "INSERT INTO friction_signals VALUES (NULL, 'u1', 'correction', 'wrong answer', 'bad response', NULL, ?)",
            (datetime.now().isoformat(),),
        )
        evo_db.commit()

        class OllamaProvider:
            def __init__(self):
                self.seen_model = None

            async def complete(self, model, max_tokens, messages):
                self.seen_model = model

                class Block:
                    text = '{"patches":["Always verify the root route in browser before claiming frontend is ready."]}'

                class Result:
                    content = [Block()]

                return Result()

        provider = OllamaProvider()
        patches = await synthesize_prompt_patches(
            provider,
            evo_db,
            {
                "min_friction_signals": 1,
                "_agent_config": {"default_model": "qwen3-coder:30b"},
            },
        )
        assert provider.seen_model == "qwen3-coder:30b"
        assert len(patches) == 1


class TestStyleAdaptation:
    def test_analyze_casual_text(self):
        style = analyze_style("hey lol can u help me pls thx")
        assert style["formality"] < 0.5
        assert style["language"] == "en"

    def test_analyze_formal_text(self):
        style = analyze_style(
            "I would like to request your assistance with the following matter. "
            "Please provide a detailed analysis of the architectural decisions.")
        assert style["formality"] > 0.5
        assert style["verbosity"] > 0.3

    def test_analyze_technical_text(self):
        style = analyze_style(
            "The `async` function returns a JSON response via the HTTP API. "
            "Use `import json` to parse the class definition.")
        assert style["technical_level"] > 0.3

    def test_analyze_russian_text(self):
        style = analyze_style("Привет, помоги мне с этим проектом пожалуйста")
        assert style["language"] == "ru"

    def test_update_creates_new_profile(self, evo_db):
        style = {"formality": 0.8, "verbosity": 0.6, "technical_level": 0.3,
                 "emoji_usage": 0.0, "language": "en"}
        update_style_profile(evo_db, "u1", style)
        row = evo_db.execute(
            "SELECT formality FROM style_profiles WHERE user_id='u1'"
        ).fetchone()
        assert row[0] == pytest.approx(0.8, abs=0.01)

    def test_update_ema(self, evo_db):
        """EMA should blend old and new values."""
        style1 = {"formality": 1.0, "verbosity": 1.0, "technical_level": 0.0,
                  "emoji_usage": 0.0, "language": "en"}
        update_style_profile(evo_db, "u1", style1, alpha=0.3)

        style2 = {"formality": 0.0, "verbosity": 0.0, "technical_level": 1.0,
                  "emoji_usage": 0.0, "language": "en"}
        update_style_profile(evo_db, "u1", style2, alpha=0.3)

        row = evo_db.execute(
            "SELECT formality, technical_level FROM style_profiles WHERE user_id='u1'"
        ).fetchone()
        # After EMA: 1.0 * 0.7 + 0.0 * 0.3 = 0.7
        assert row[0] == pytest.approx(0.7, abs=0.01)
        # After EMA: 0.0 * 0.7 + 1.0 * 0.3 = 0.3
        assert row[1] == pytest.approx(0.3, abs=0.01)

    def test_get_style_instruction_empty(self, evo_db):
        assert get_style_instruction(evo_db, "nonexistent") == ""

    def test_get_style_instruction_casual(self, evo_db):
        evo_db.execute(
            "INSERT INTO style_profiles VALUES ('u1', 0.1, 0.1, 0.1, 0.0, 'ru', ?)",
            (datetime.now().isoformat(),))
        evo_db.commit()
        instruction = get_style_instruction(evo_db, "u1")
        assert "casual" in instruction.lower()
        assert "concise" in instruction.lower()
        assert "Russian" in instruction


class TestProactiveAgent:
    def test_insufficient_data_returns_empty(self, evo_db):
        result = detect_patterns(evo_db, "u1", "test",
                                 {"pattern_window_days": 30,
                                  "min_pattern_occurrences": 3})
        assert result == []

    def test_inputs_similar(self):
        assert _inputs_similar("check my email", "check my email") is True
        assert _inputs_similar("check my email", "buy groceries") is False

    def test_inputs_similar_partial_overlap(self):
        assert _inputs_similar("check my email inbox",
                               "check my email") is True

    def test_inputs_similar_empty(self):
        assert _inputs_similar("", "") is False

    def test_detects_recurring_workflow_for_similar_request(self, evo_db):
        now = datetime.now().isoformat()
        rows = [
            ("build fastapi dashboard", '[{"name":"read_file"},{"name":"write_file"}]'),
            ("build fastapi admin dashboard", '[{"name":"read_file"},{"name":"write_file"}]'),
            ("build dashboard in fastapi", '[{"name":"read_file"},{"name":"write_file"}]'),
        ]
        for user_input, tools in rows:
            evo_db.execute(
                "INSERT INTO interaction_log VALUES (NULL, 'u1', ?, 'done', ?, 1, 0.9, 'local', ?)",
                (user_input, tools, now),
            )
        evo_db.commit()

        result = detect_patterns(
            evo_db,
            "u1",
            "build a fastapi dashboard for analytics",
            {"pattern_window_days": 30, "min_pattern_occurrences": 3},
        )
        assert any("recurring workflow" in item.lower() for item in result)
        assert any("write_file" in item for item in result)

    def test_detects_likely_followup_step(self, evo_db):
        base_time = datetime.now() - timedelta(days=1)
        for i in range(3):
            t1 = (base_time + timedelta(minutes=i * 10)).isoformat()
            t2 = (base_time + timedelta(minutes=i * 10 + 1)).isoformat()
            evo_db.execute(
                "INSERT INTO interaction_log VALUES (NULL, 'u1', ?, 'done', '[]', 1, 0.8, 'local', ?)",
                ("sync drive files", t1),
            )
            evo_db.execute(
                "INSERT INTO interaction_log VALUES (NULL, 'u1', ?, 'done', '[]', 1, 0.8, 'local', ?)",
                ("search files by content", t2),
            )
        evo_db.commit()

        result = detect_patterns(
            evo_db,
            "u1",
            "sync files from google drive",
            {"pattern_window_days": 30, "min_pattern_occurrences": 3},
        )
        assert any("asks next to" in item.lower() for item in result)
        assert any("search files by content" in item.lower() for item in result)

    def test_ignores_failed_runs_for_proactive_suggestions(self, evo_db):
        now = datetime.now().isoformat()
        for _ in range(4):
            evo_db.execute(
                "INSERT INTO interaction_log VALUES (NULL, 'u1', ?, 'failed', '[]', 0, 0.2, 'local', ?)",
                ("deploy production build", now),
            )
        evo_db.commit()

        result = detect_patterns(
            evo_db,
            "u1",
            "deploy the app",
            {"pattern_window_days": 30, "min_pattern_occurrences": 3, "min_confidence": 0.5},
        )
        assert result == []


class TestHumanSupportAgent:
    def test_detects_overload_and_focus_support(self, evo_db):
        result = detect_human_support_opportunities(
            evo_db,
            "u1",
            "Я перегружен и не могу сосредоточиться, слишком много задач",
            {"enabled": True, "max_suggestions": 3},
        )
        text = "\n".join(result).lower()
        assert "overloaded" in text
        assert "focus reset" in text

    def test_detects_late_night_energy_pattern(self, evo_db):
        for day in range(3):
            evo_db.execute(
                "INSERT INTO interaction_log VALUES (NULL, 'u1', ?, 'done', '[]', 1, 0.9, 'local', ?)",
                ("finish work late at night", f"2026-03-0{day + 1}T23:45:00"),
            )
        evo_db.commit()
        result = detect_human_support_opportunities(
            evo_db,
            "u1",
            "надо доделать еще сегодня",
            {"enabled": True, "min_pattern_occurrences": 3, "late_night_hour": 23, "early_hour": 6},
        )
        assert any("energy" in item.lower() for item in result)

    def test_detects_admin_burden_and_suggests_automation(self, evo_db):
        timestamps = ["2026-03-01T10:00:00", "2026-03-02T11:00:00", "2026-03-03T12:00:00"]
        for ts in timestamps:
            evo_db.execute(
                "INSERT INTO interaction_log VALUES (NULL, 'u1', ?, 'done', '[]', 1, 0.8, 'local', ?)",
                ("renew passport reminder deadline", ts),
            )
        evo_db.commit()
        result = detect_human_support_opportunities(
            evo_db,
            "u1",
            "надо продлить документы",
            {"enabled": True, "min_pattern_occurrences": 3},
        )
        assert any("reminders" in item.lower() or "calendar" in item.lower() for item in result)

    def test_builds_human_support_snapshot(self, evo_db):
        evo_db.execute(
            "INSERT INTO interaction_log VALUES (NULL, 'u1', ?, 'done', '[]', 1, 0.9, 'local', ?)",
            ("I am overwhelmed and need a reminder for renewal", "2026-03-09T23:50:00"),
        )
        evo_db.commit()
        snapshot = get_human_support_snapshot(evo_db, "u1", "", {"enabled": True})
        assert snapshot["enabled"] is True
        assert snapshot["metrics"]["overload_signals"] >= 1
        assert isinstance(snapshot["opportunities"], list)
