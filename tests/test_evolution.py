"""Tests for evolution module: friction detection, style adaptation, proactive agent."""

import sqlite3
import pytest
from datetime import datetime, timedelta

from liteagent.evolution import (
    detect_friction, store_friction, get_active_patches,
    analyze_style, update_style_profile, get_style_instruction,
    detect_patterns, _inputs_similar,
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
