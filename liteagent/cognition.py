"""Cognitive Intelligence Module — 4 autonomous features that make the agent smarter.

Features:
    1. Socratic Self-Adversary  — challenges its own responses before sending
    2. Session Goal Inference   — infers the meta-goal behind user requests
    3. Cognitive State Adapter  — adapts style to user's current mental state
    4. Epistemic Calibration    — tracks prediction accuracy over time
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── shared JSON parser (mirrors metacognition.py / planning.py) ──────────────

def _safe_parse_llm_json(text: str, fallback: Any) -> Any:
    import re as _re, json as _json
    text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Strip <think>…</think> blocks from reasoning models
    text = _re.sub(r'<think>[\s\S]*?</think>', '', text, flags=_re.IGNORECASE).strip()
    # Strip markdown fences
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        return _json.loads(text)
    except Exception:
        pass
    for opener, closer in (('{', '}'), ('[', ']')):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(text[start:i + 1])
                    except Exception:
                        break
    return fallback


def _cheapest_model(provider) -> str:
    from .planning import _CHEAPEST_MODEL
    return _CHEAPEST_MODEL.get(provider.__class__.__name__, "claude-haiku-4-5-20251001")


# ══════════════════════════════════════════════════════════════════════════════
# DB INIT
# ══════════════════════════════════════════════════════════════════════════════

def init_cognition_tables(db: sqlite3.Connection) -> None:
    db.executescript("""
        CREATE TABLE IF NOT EXISTS epistemic_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            prediction_text TEXT NOT NULL,
            confidence_word TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            outcome TEXT DEFAULT 'pending',
            outcome_at TEXT,
            outcome_evidence TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_ep_user_outcome
            ON epistemic_predictions(user_id, outcome, created_at DESC);

        CREATE TABLE IF NOT EXISTS session_goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            goal_text TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_sg_user_seen
            ON session_goals(user_id, last_seen_at DESC);
    """)
    db.commit()


# ══════════════════════════════════════════════════════════════════════════════
# 1. SOCRATIC SELF-ADVERSARY
# ══════════════════════════════════════════════════════════════════════════════

async def socratic_challenge(provider, user_input: str, response: str,
                              model: str) -> str:
    """Challenge the planned response with counterarguments.

    Makes one cheap LLM call. If counterarguments are strong (strength >= 4),
    returns a revised answer. Otherwise returns original.
    """
    prompt = (
        "You just wrote the response below. Your task: generate the 2 strongest "
        "counterarguments or flaws in it. Rate each counterargument 1-5 (5=fatal flaw).\n"
        "If any strength >= 4, also write a revised, improved answer.\n\n"
        f"User asked: {user_input[:400]}\n"
        f"Your response: {response[:700]}\n\n"
        "Return ONLY valid JSON:\n"
        '{"counterargs": [{"arg": "...", "strength": N}], '
        '"needs_revision": true/false, "revised": "..." }'
    )
    try:
        result = await provider.complete(
            model=model,
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = result.content[0].text if result.content else ""
        data = _safe_parse_llm_json(raw, {})
        if not isinstance(data, dict):
            return response

        needs = data.get("needs_revision", False)
        revised = (data.get("revised") or "").strip()
        if needs and len(revised) >= 20:
            logger.debug("Socratic: revision accepted (len=%d)", len(revised))
            return revised

        # Even if needs_revision is false, check strengths manually
        counterargs = data.get("counterargs") or []
        if isinstance(counterargs, list):
            for ca in counterargs:
                if isinstance(ca, dict) and int(ca.get("strength", 0) or 0) >= 4:
                    if revised and len(revised) >= 20:
                        logger.debug("Socratic: forced revision due to strong arg: %s", ca.get("arg", ""))
                        return revised
                    break

    except Exception as e:
        logger.debug("Socratic challenge error: %s", e)

    return response


# ══════════════════════════════════════════════════════════════════════════════
# 2. COGNITIVE STATE ADAPTATION
# ══════════════════════════════════════════════════════════════════════════════

# State → instruction injected into system prompt
COGNITIVE_STATES: dict[str, str] = {
    "focused":  "User is in deep-focus mode. Provide thorough, structured response with details.",
    "fatigued": "User appears tired/low-energy. Be concise and direct. Max 3 key points, no preamble.",
    "creative": "User is in exploratory mode. Offer alternatives, examples, and lateral ideas.",
    "urgent":   "User needs quick action. Lead immediately with the answer. Skip context and caveats.",
    "casual":   "Casual chat mode. Match relaxed tone, keep it conversational and brief.",
}

_CSA_STATE_KEY = "user:csa_signal"
_CSA_RING_SIZE = 12


def update_cognitive_signal(db: sqlite3.Connection, user_id: str, msg_text: str) -> None:
    """Store message timestamp + length in a ring buffer (last 12 messages)."""
    try:
        from .memory import MemorySystem
        # get_state / set_state can be called directly on the db if we replicate the logic
        row = db.execute(
            "SELECT value FROM user_state WHERE user_id=? AND key=?",
            (user_id, _CSA_STATE_KEY),
        ).fetchone()
        signal: dict = json.loads(row[0]) if row else {}

        timestamps: list[str] = signal.get("timestamps", [])
        lengths: list[int] = signal.get("lengths", [])
        words: list[float] = signal.get("avg_word_lens", [])

        now_iso = datetime.now(timezone.utc).isoformat()
        timestamps.append(now_iso)
        lengths.append(len(msg_text))

        words_in_msg = [w for w in msg_text.split() if w.isalpha()]
        avg_wl = (sum(len(w) for w in words_in_msg) / len(words_in_msg)) if words_in_msg else 4.0
        words.append(round(avg_wl, 2))

        # Keep ring
        timestamps = timestamps[-_CSA_RING_SIZE:]
        lengths = lengths[-_CSA_RING_SIZE:]
        words = words[-_CSA_RING_SIZE:]

        signal = {"timestamps": timestamps, "lengths": lengths, "avg_word_lens": words}
        now_db = datetime.now().isoformat()
        db.execute(
            "INSERT OR REPLACE INTO user_state VALUES (?, ?, ?, ?)",
            (user_id, _CSA_STATE_KEY, json.dumps(signal), now_db),
        )
        db.commit()
    except Exception as e:
        logger.debug("CSA signal update error: %s", e)


def compute_cognitive_state(db: sqlite3.Connection, user_id: str) -> str:
    """Infer cognitive state from stored signals. Returns state key or empty string."""
    try:
        row = db.execute(
            "SELECT value FROM user_state WHERE user_id=? AND key=?",
            (user_id, _CSA_STATE_KEY),
        ).fetchone()
        if not row:
            return ""
        signal: dict = json.loads(row[0])

        timestamps: list[str] = signal.get("timestamps", [])
        lengths: list[int] = signal.get("lengths", [])
        avg_word_lens: list[float] = signal.get("avg_word_lens", [])

        if len(timestamps) < 2:
            return ""

        # ── Signal 1: Time of day ──
        try:
            last_dt = datetime.fromisoformat(timestamps[-1])
            hour = last_dt.hour
        except Exception:
            hour = 12
        if 22 <= hour or hour < 6:
            time_signal = "fatigued"
        elif 9 <= hour < 14:
            time_signal = "focused"
        elif 18 <= hour < 22:
            time_signal = "casual"
        else:
            time_signal = "neutral"

        # ── Signal 2: Recent average message length ──
        recent_len = lengths[-5:] if len(lengths) >= 5 else lengths
        avg_len = sum(recent_len) / len(recent_len)
        if avg_len < 25:
            len_signal = "urgent"
        elif avg_len > 200:
            len_signal = "focused"
        elif avg_len < 60:
            len_signal = "casual"
        else:
            len_signal = "neutral"

        # ── Signal 3: Inter-message velocity ──
        vel_signal = "neutral"
        if len(timestamps) >= 3:
            try:
                t1 = datetime.fromisoformat(timestamps[-1])
                t2 = datetime.fromisoformat(timestamps[-2])
                gap_sec = abs((t1 - t2).total_seconds())
                if gap_sec < 30:
                    vel_signal = "urgent"
                elif gap_sec > 300:
                    vel_signal = "focused"
                elif gap_sec > 600:
                    vel_signal = "casual"
            except Exception:
                pass

        # ── Signal 4: Word complexity ──
        recent_words = avg_word_lens[-5:] if len(avg_word_lens) >= 5 else avg_word_lens
        avg_wl = sum(recent_words) / len(recent_words) if recent_words else 4.0
        if avg_wl > 6.5:
            word_signal = "focused"
        elif avg_wl < 3.5:
            word_signal = "casual"
        else:
            word_signal = "neutral"

        # ── Vote ──
        signals = [time_signal, len_signal, vel_signal, word_signal]
        counts: dict[str, int] = {}
        for s in signals:
            if s != "neutral":
                counts[s] = counts.get(s, 0) + 1

        if not counts:
            return ""

        winner = max(counts, key=lambda k: counts[k])
        # Need at least 2 signals agreeing, or time_signal alone if strong
        if counts[winner] >= 2 or (winner in ("fatigued", "focused") and time_signal == winner):
            return winner
        return ""

    except Exception as e:
        logger.debug("CSA compute_state error: %s", e)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 3. SESSION GOAL INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

_GOAL_NULL_RESPONSES = frozenset({
    "null", "none", "n/a", "small talk", "casual conversation",
    "нет", "пустой", "светская беседа", "просто разговор",
})


async def infer_session_goal(provider, user_input: str, history_tail: str,
                              model: str, db: sqlite3.Connection,
                              user_id: str) -> None:
    """Infer the underlying meta-goal. Runs as a background fire-and-forget task."""
    if len(user_input.strip()) < 30:
        return

    prompt = (
        "What is the ONE underlying meta-goal or intent behind this user request?\n"
        "Answer in 1 concise sentence (e.g. 'Get promoted at work', 'Build a side business', "
        "'Understand machine learning fundamentals').\n"
        "If this is just small talk or a simple lookup, respond with exactly: null\n\n"
        f"User request: {user_input[:350]}\n"
        f"Recent context: {history_tail[:250]}"
    )
    try:
        result = await provider.complete(
            model=model,
            max_tokens=80,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (result.content[0].text if result.content else "").strip()
        # Strip quotes and whitespace
        goal = raw.strip('"\'').strip()

        if not goal or goal.lower() in _GOAL_NULL_RESPONSES:
            return
        if len(goal) < 8 or len(goal) > 200:
            return

        _upsert_session_goal(db, user_id, goal)
        logger.debug("Goal inference stored: %r for user %s", goal[:80], user_id)

    except Exception as e:
        logger.debug("Goal inference error: %s", e)


def _upsert_session_goal(db: sqlite3.Connection, user_id: str, goal_text: str) -> None:
    """Insert or update session goal (deduplicates by similarity using simple word overlap)."""
    now = datetime.now().isoformat()
    try:
        rows = db.execute(
            "SELECT id, goal_text FROM session_goals WHERE user_id=? "
            "ORDER BY last_seen_at DESC LIMIT 10",
            (user_id,),
        ).fetchall()

        # Check for semantic duplicate (simple word-overlap heuristic)
        goal_words = set(goal_text.lower().split())
        for row_id, existing in rows:
            existing_words = set(existing.lower().split())
            if not goal_words or not existing_words:
                continue
            overlap = len(goal_words & existing_words) / max(len(goal_words), len(existing_words))
            if overlap > 0.6:
                # Update last_seen
                db.execute(
                    "UPDATE session_goals SET last_seen_at=?, confidence=MIN(confidence+0.1, 1.0) "
                    "WHERE id=?",
                    (now, row_id),
                )
                db.commit()
                return

        # Insert new goal
        db.execute(
            "INSERT INTO session_goals (user_id, goal_text, confidence, created_at, last_seen_at) "
            "VALUES (?, ?, 0.5, ?, ?)",
            (user_id, goal_text, now, now),
        )
        db.commit()
    except Exception as e:
        logger.debug("Goal upsert error: %s", e)


def get_active_goals_prompt(db: sqlite3.Connection, user_id: str,
                             max_goals: int = 2) -> str:
    """Return formatted string for system prompt injection, or empty string."""
    try:
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        rows = db.execute(
            "SELECT goal_text, confidence FROM session_goals "
            "WHERE user_id=? AND last_seen_at >= ? "
            "ORDER BY confidence DESC, last_seen_at DESC LIMIT ?",
            (user_id, cutoff, max_goals),
        ).fetchall()

        if not rows:
            return ""

        lines = []
        for goal_text, confidence in rows:
            conf_label = "high" if confidence >= 0.7 else ("medium" if confidence >= 0.4 else "low")
            lines.append(f"- {goal_text} [confidence: {conf_label}]")
        return "\n".join(lines)

    except Exception as e:
        logger.debug("get_active_goals_prompt error: %s", e)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 4. EPISTEMIC CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

_PREDICTION_PATTERNS = [
    r'\b(probably|likely|I think|I believe|I expect|I suspect|should be|ought to be)\b',
    r'\b(скорее всего|вероятно|думаю|полагаю|считаю|должно быть|наверное|похоже что)\b',
]
_PREDICTION_PATTERN = re.compile(
    "|".join(_PREDICTION_PATTERNS), re.IGNORECASE
)

_CONFIRMATION_PATTERNS = re.compile(
    r'\b(да|верно|точно|правильно|именно|ты прав|correct|exactly|right|yes|confirmed|'
    r'подтверждаю|оказалось так|так и есть|именно так)\b',
    re.IGNORECASE,
)
_DENIAL_PATTERNS = re.compile(
    r'\b(нет|неверно|неправильно|ты ошибся|не так|wrong|incorrect|no|'
    r'оказалось не так|ошиблись|неточно)\b',
    re.IGNORECASE,
)


def extract_predictions(response_text: str) -> list[dict]:
    """Extract sentences containing prediction markers from a response."""
    sentences = re.split(r'(?<=[.!?])\s+', response_text)
    results = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 20 or len(sentence) > 300:
            continue
        m = _PREDICTION_PATTERN.search(sentence)
        if m:
            results.append({
                "text": sentence,
                "confidence_word": m.group(0).lower(),
            })
    return results[:5]  # cap at 5 per response


def store_predictions(db: sqlite3.Connection, user_id: str,
                       predictions: list[dict]) -> None:
    if not predictions:
        return
    now = datetime.now().isoformat()
    try:
        db.executemany(
            "INSERT INTO epistemic_predictions "
            "(user_id, prediction_text, confidence_word, created_at) VALUES (?, ?, ?, ?)",
            [(user_id, p["text"], p.get("confidence_word", ""), now) for p in predictions],
        )
        db.commit()
    except Exception as e:
        logger.debug("store_predictions error: %s", e)


def detect_and_update_outcomes(db: sqlite3.Connection, user_id: str,
                                current_input: str) -> int:
    """Check if the current user message confirms/denies a pending prediction.

    Returns count of outcomes updated.
    """
    text = current_input.strip()
    if len(text) < 2 or len(text) > 400:
        return 0

    confirmed = bool(_CONFIRMATION_PATTERNS.search(text))
    denied = bool(_DENIAL_PATTERNS.search(text))

    if not confirmed and not denied:
        return 0

    outcome = "confirmed" if confirmed else "denied"

    try:
        # Find the most recent pending prediction for this user
        row = db.execute(
            "SELECT id FROM epistemic_predictions "
            "WHERE user_id=? AND outcome='pending' "
            "ORDER BY created_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if not row:
            return 0

        now = datetime.now().isoformat()
        db.execute(
            "UPDATE epistemic_predictions SET outcome=?, outcome_at=?, outcome_evidence=? "
            "WHERE id=?",
            (outcome, now, text[:200], row[0]),
        )
        db.commit()
        logger.debug("Epistemic: prediction %d marked as %s", row[0], outcome)
        return 1

    except Exception as e:
        logger.debug("detect_and_update_outcomes error: %s", e)
        return 0


def get_calibration_stats(db: sqlite3.Connection, user_id: str) -> dict:
    """Return calibration stats: total/confirmed/denied/pending/accuracy_pct."""
    try:
        rows = db.execute(
            "SELECT outcome, COUNT(*) FROM epistemic_predictions "
            "WHERE user_id=? GROUP BY outcome",
            (user_id,),
        ).fetchall()

        counts = {r[0]: r[1] for r in rows}
        total = sum(counts.values())
        confirmed = counts.get("confirmed", 0)
        denied = counts.get("denied", 0)
        pending = counts.get("pending", 0)
        resolved = confirmed + denied
        accuracy = round(confirmed / resolved * 100, 1) if resolved > 0 else None

        return {
            "total": total,
            "confirmed": confirmed,
            "denied": denied,
            "pending": pending,
            "accuracy_pct": accuracy,
        }
    except Exception as e:
        logger.debug("get_calibration_stats error: %s", e)
        return {"total": 0, "confirmed": 0, "denied": 0, "pending": 0, "accuracy_pct": None}
