"""Tests for liteagent/cognition.py — cognitive intelligence features."""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from liteagent.cognition import (
    COGNITIVE_STATES,
    compute_cognitive_state,
    detect_and_update_outcomes,
    extract_predictions,
    get_active_goals_prompt,
    get_calibration_stats,
    init_cognition_tables,
    infer_session_goal,
    socratic_challenge,
    store_predictions,
    update_cognitive_signal,
    _upsert_session_goal,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    # Minimal user_state table (mirrors memory.py)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS user_state (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        );
    """)
    init_cognition_tables(conn)
    return conn


def _inject_signal(db, user_id, timestamps, lengths, avg_word_lens=None):
    """Helper: inject a CSA signal directly into user_state."""
    if avg_word_lens is None:
        avg_word_lens = [4.0] * len(timestamps)
    signal = {"timestamps": timestamps, "lengths": lengths, "avg_word_lens": avg_word_lens}
    db.execute(
        "INSERT OR REPLACE INTO user_state VALUES (?, ?, ?, ?)",
        (user_id, "user:csa_signal", json.dumps(signal), datetime.now().isoformat()),
    )
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# 1. SOCRATIC SELF-ADVERSARY
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_socratic_revises_when_strong_counterarg():
    """When counterargument strength >= 4, the response should be revised."""
    mock_provider = MagicMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text=json.dumps({
        "counterargs": [{"arg": "This is wrong because X", "strength": 4}],
        "needs_revision": True,
        "revised": "Corrected and improved answer that addresses the flaw.",
    }))]
    mock_provider.complete = AsyncMock(return_value=mock_result)

    result = await socratic_challenge(
        mock_provider,
        user_input="Explain how X works",
        response="Original answer about X.",
        model="test-model",
    )
    assert result == "Corrected and improved answer that addresses the flaw."


@pytest.mark.asyncio
async def test_socratic_keeps_original_when_weak():
    """When counterarguments are weak (<4), original response is returned."""
    mock_provider = MagicMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text=json.dumps({
        "counterargs": [{"arg": "Minor quibble", "strength": 2}],
        "needs_revision": False,
        "revised": "",
    }))]
    mock_provider.complete = AsyncMock(return_value=mock_result)

    original = "Original answer that is fine."
    result = await socratic_challenge(
        mock_provider, "User question", original, "test-model"
    )
    assert result == original


@pytest.mark.asyncio
async def test_socratic_returns_original_on_error():
    """On LLM error, original response is returned gracefully."""
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock(side_effect=Exception("LLM unavailable"))

    original = "My original response."
    result = await socratic_challenge(mock_provider, "Question", original, "model")
    assert result == original


@pytest.mark.asyncio
async def test_socratic_requires_revised_min_length():
    """'revised' field must be >= 20 chars to be accepted."""
    mock_provider = MagicMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text=json.dumps({
        "counterargs": [{"arg": "Fatal flaw", "strength": 5}],
        "needs_revision": True,
        "revised": "Too short",  # < 20 chars
    }))]
    mock_provider.complete = AsyncMock(return_value=mock_result)

    original = "My original detailed response."
    result = await socratic_challenge(mock_provider, "Question", original, "model")
    assert result == original


# ══════════════════════════════════════════════════════════════════════════════
# 2. COGNITIVE STATE ADAPTATION
# ══════════════════════════════════════════════════════════════════════════════

def _ts(hour: int, minutes_ago: int = 0) -> str:
    """Build ISO timestamp at given UTC hour, optionally minutes in the past."""
    dt = datetime.now(timezone.utc).replace(hour=hour, minute=0, second=0, microsecond=0)
    dt -= timedelta(minutes=minutes_ago)
    return dt.isoformat()


def test_cognitive_state_focused(db):
    """Long messages in the morning → focused."""
    timestamps = [_ts(10, i * 5) for i in range(6, 0, -1)]
    lengths = [250, 300, 280, 320, 240, 260]
    avg_word_lens = [6.0, 7.0, 6.5, 7.2, 6.8, 7.0]
    _inject_signal(db, "user1", timestamps, lengths, avg_word_lens)

    state = compute_cognitive_state(db, "user1")
    assert state == "focused", f"Expected 'focused', got '{state}'"


def test_cognitive_state_fatigued(db):
    """Short messages late at night → fatigued."""
    timestamps = [_ts(23, i * 3) for i in range(6, 0, -1)]
    lengths = [10, 8, 12, 5, 9, 7]
    avg_word_lens = [3.5, 3.2, 3.8, 3.1, 3.6, 3.4]
    _inject_signal(db, "user2", timestamps, lengths, avg_word_lens)

    state = compute_cognitive_state(db, "user2")
    assert state == "fatigued", f"Expected 'fatigued', got '{state}'"


def test_cognitive_state_urgent(db):
    """Very short, rapid messages → urgent."""
    # Fast inter-message gaps (< 30s)
    base = datetime.now(timezone.utc).replace(hour=15)
    timestamps = [(base - timedelta(seconds=i * 10)).isoformat() for i in range(6, 0, -1)]
    lengths = [8, 6, 10, 5, 9, 7]
    _inject_signal(db, "user3", timestamps, lengths)

    state = compute_cognitive_state(db, "user3")
    assert state in ("urgent", "casual", ""), f"Unexpected state: '{state}'"


def test_cognitive_state_no_signal(db):
    """No signal stored → empty string."""
    state = compute_cognitive_state(db, "unknown_user")
    assert state == ""


def test_cognitive_state_not_enough_data(db):
    """Only 1 message → insufficient data → empty string."""
    _inject_signal(db, "user4", [_ts(10)], [100])
    state = compute_cognitive_state(db, "user4")
    assert state == ""


def test_update_cognitive_signal_ring(db):
    """Signal ring should keep only last _CSA_RING_SIZE entries."""
    from liteagent.cognition import _CSA_RING_SIZE, _CSA_STATE_KEY
    for i in range(_CSA_RING_SIZE + 5):
        update_cognitive_signal(db, "ring_user", "hello world test message")

    row = db.execute(
        "SELECT value FROM user_state WHERE user_id='ring_user' AND key=?",
        (_CSA_STATE_KEY,),
    ).fetchone()
    assert row is not None
    signal = json.loads(row[0])
    assert len(signal["timestamps"]) == _CSA_RING_SIZE


def test_cognitive_states_dict_complete():
    """All expected states should have directives."""
    for state in ("focused", "fatigued", "creative", "urgent", "casual"):
        assert state in COGNITIVE_STATES
        assert len(COGNITIVE_STATES[state]) > 10


# ══════════════════════════════════════════════════════════════════════════════
# 3. SESSION GOAL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_infer_session_goal_stores(db):
    """Goal inference should store a non-null goal in session_goals."""
    mock_provider = MagicMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="Get promoted to senior engineer")]
    mock_provider.complete = AsyncMock(return_value=mock_result)

    await infer_session_goal(
        mock_provider,
        user_input="How do I write a performance self-review that gets me promoted?",
        history_tail="user: I'm working on my promotion case",
        model="test-model",
        db=db,
        user_id="goal_user",
    )

    row = db.execute(
        "SELECT goal_text FROM session_goals WHERE user_id='goal_user'"
    ).fetchone()
    assert row is not None
    assert "promot" in row[0].lower() or "engineer" in row[0].lower()


@pytest.mark.asyncio
async def test_infer_session_goal_ignores_null(db):
    """Goal inference should NOT store when LLM responds with null."""
    mock_provider = MagicMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="null")]
    mock_provider.complete = AsyncMock(return_value=mock_result)

    await infer_session_goal(
        mock_provider,
        user_input="What time is it?",
        history_tail="",
        model="test-model",
        db=db,
        user_id="null_goal_user",
    )

    rows = db.execute(
        "SELECT id FROM session_goals WHERE user_id='null_goal_user'"
    ).fetchall()
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_infer_session_goal_skips_short_messages(db):
    """Messages shorter than 30 chars should not trigger goal inference."""
    mock_provider = MagicMock()
    mock_provider.complete = AsyncMock()

    await infer_session_goal(
        mock_provider,
        user_input="hi",
        history_tail="",
        model="test-model",
        db=db,
        user_id="short_user",
    )

    mock_provider.complete.assert_not_called()


def test_get_active_goals_prompt_empty(db):
    """No goals → empty string."""
    result = get_active_goals_prompt(db, "no_goal_user")
    assert result == ""


def test_get_active_goals_prompt_formats(db):
    """Goals within 7 days should be formatted for prompt."""
    _upsert_session_goal(db, "gp_user", "Become a better public speaker")
    _upsert_session_goal(db, "gp_user", "Launch side project by Q3")

    result = get_active_goals_prompt(db, "gp_user", max_goals=2)
    assert "Become a better public speaker" in result or "Launch side project" in result
    assert "[confidence:" in result


def test_upsert_session_goal_deduplicates(db):
    """Goals with >60% word overlap should update, not insert new row."""
    _upsert_session_goal(db, "dedup_user", "Get promoted to senior engineer")
    _upsert_session_goal(db, "dedup_user", "Get promoted to senior developer engineer")

    rows = db.execute(
        "SELECT COUNT(*) FROM session_goals WHERE user_id='dedup_user'"
    ).fetchone()
    assert rows[0] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. EPISTEMIC CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def test_extract_predictions_en():
    """English prediction markers should be extracted."""
    text = (
        "This is likely to work. I think you should try it. "
        "The weather is nice today. I believe this approach is correct."
    )
    preds = extract_predictions(text)
    assert len(preds) >= 2
    assert all("confidence_word" in p for p in preds)
    assert all("text" in p for p in preds)


def test_extract_predictions_ru():
    """Russian prediction markers should be extracted."""
    text = (
        "Скорее всего это сработает. Думаю, что нужно попробовать. "
        "Сейчас хорошая погода. Вероятно, это правильный подход."
    )
    preds = extract_predictions(text)
    assert len(preds) >= 2


def test_extract_predictions_no_markers():
    """Response without prediction markers → empty list."""
    text = "The sky is blue. Water is wet. This is a fact."
    preds = extract_predictions(text)
    assert preds == []


def test_extract_predictions_caps_at_five():
    """Should not return more than 5 predictions."""
    text = " ".join(["I think statement number " + str(i) + " is true." for i in range(10)])
    preds = extract_predictions(text)
    assert len(preds) <= 5


def test_store_and_detect_confirmed(db):
    """Storing a prediction then receiving confirmation updates outcome."""
    preds = [{"text": "This will probably work.", "confidence_word": "probably"}]
    store_predictions(db, "ep_user", preds)

    count = detect_and_update_outcomes(db, "ep_user", "Да, ты прав, так и есть!")
    assert count == 1

    row = db.execute(
        "SELECT outcome FROM epistemic_predictions WHERE user_id='ep_user'"
    ).fetchone()
    assert row[0] == "confirmed"


def test_store_and_detect_denied(db):
    """Denial markers should set outcome=denied."""
    preds = [{"text": "I believe this is the answer.", "confidence_word": "i believe"}]
    store_predictions(db, "ep_user2", preds)

    count = detect_and_update_outcomes(db, "ep_user2", "Нет, ты ошибся.")
    assert count == 1

    row = db.execute(
        "SELECT outcome FROM epistemic_predictions WHERE user_id='ep_user2'"
    ).fetchone()
    assert row[0] == "denied"


def test_detect_outcomes_no_match(db):
    """Neutral messages should not update any prediction."""
    preds = [{"text": "Likely correct.", "confidence_word": "likely"}]
    store_predictions(db, "ep_user3", preds)

    count = detect_and_update_outcomes(db, "ep_user3", "Tell me more about that.")
    assert count == 0

    row = db.execute(
        "SELECT outcome FROM epistemic_predictions WHERE user_id='ep_user3'"
    ).fetchone()
    assert row[0] == "pending"


def test_get_calibration_stats_empty(db):
    """No predictions → zero stats."""
    stats = get_calibration_stats(db, "empty_user")
    assert stats["total"] == 0
    assert stats["accuracy_pct"] is None


def test_get_calibration_stats_with_data(db):
    """Correct accuracy computation."""
    now = datetime.now().isoformat()
    db.execute(
        "INSERT INTO epistemic_predictions "
        "(user_id, prediction_text, confidence_word, created_at, outcome) "
        "VALUES (?, ?, ?, ?, ?)",
        ("stats_user", "Prediction A", "likely", now, "confirmed"),
    )
    db.execute(
        "INSERT INTO epistemic_predictions "
        "(user_id, prediction_text, confidence_word, created_at, outcome) "
        "VALUES (?, ?, ?, ?, ?)",
        ("stats_user", "Prediction B", "probably", now, "denied"),
    )
    db.execute(
        "INSERT INTO epistemic_predictions "
        "(user_id, prediction_text, confidence_word, created_at, outcome) "
        "VALUES (?, ?, ?, ?, ?)",
        ("stats_user", "Prediction C", "i think", now, "pending"),
    )
    db.commit()

    stats = get_calibration_stats(db, "stats_user")
    assert stats["total"] == 3
    assert stats["confirmed"] == 1
    assert stats["denied"] == 1
    assert stats["pending"] == 1
    assert stats["accuracy_pct"] == 50.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. INTEGRATION — HOOK REGISTRATION
# ══════════════════════════════════════════════════════════════════════════════

def test_cognition_hooks_registered_when_enabled():
    """All 4 cognition hooks should be registered when feature is enabled."""
    config = {
        "agent": {"provider": "anthropic", "default_model": "claude-haiku-4-5-20251001"},
        "features": {"cognition": {"enabled": True}},
        "memory": {},
        "tools": {},
        "cost": {},
        "providers": {"anthropic": {}},
    }
    from unittest.mock import patch, MagicMock
    with patch("liteagent.agent.create_provider", return_value=MagicMock()), \
         patch("liteagent.agent.MemorySystem") as MockMem, \
         patch("liteagent.agent.ToolRegistry"), \
         patch("liteagent.agent.register_builtin_tools"), \
         patch("liteagent.agent.load_plugins", return_value=[]), \
         patch("liteagent.agent.get_soul_prompt", return_value="soul"), \
         patch("liteagent.cognition.init_cognition_tables"):

        mock_mem = MagicMock()
        mock_mem.db = sqlite3.connect(":memory:")
        mock_mem.db.executescript("""
            CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT, key TEXT, value TEXT, updated_at TEXT,
                PRIMARY KEY (user_id, key)
            );
        """)
        MockMem.return_value = mock_mem

        from liteagent.agent import LiteAgent
        agent = LiteAgent(config)

    hook_names = {
        h.name
        for handlers in agent.hooks._hooks.values()
        for h in handlers
    }
    assert "socratic_adversary" in hook_names
    assert "goal_inference" in hook_names
    assert "cognitive_state_update" in hook_names
    assert "epistemic_calibration" in hook_names


def test_cognition_hooks_not_registered_when_disabled():
    """No cognition hooks when feature is disabled."""
    config = {
        "agent": {"provider": "anthropic", "default_model": "claude-haiku-4-5-20251001"},
        "features": {"cognition": {"enabled": False}},
        "memory": {},
        "tools": {},
        "cost": {},
        "providers": {"anthropic": {}},
    }
    from unittest.mock import patch, MagicMock
    with patch("liteagent.agent.create_provider", return_value=MagicMock()), \
         patch("liteagent.agent.MemorySystem") as MockMem, \
         patch("liteagent.agent.ToolRegistry"), \
         patch("liteagent.agent.register_builtin_tools"), \
         patch("liteagent.agent.load_plugins", return_value=[]), \
         patch("liteagent.agent.get_soul_prompt", return_value="soul"):

        mock_mem = MagicMock()
        mock_mem.db = sqlite3.connect(":memory:")
        mock_mem.db.executescript("""
            CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT, key TEXT, value TEXT, updated_at TEXT,
                PRIMARY KEY (user_id, key)
            );
        """)
        MockMem.return_value = mock_mem

        from liteagent.agent import LiteAgent
        agent = LiteAgent(config)

    hook_names = {
        h.name
        for handlers in agent.hooks._hooks.values()
        for h in handlers
    }
    assert "socratic_adversary" not in hook_names
    assert "goal_inference" not in hook_names
    assert "cognitive_state_update" not in hook_names
    assert "epistemic_calibration" not in hook_names
