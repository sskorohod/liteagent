"""Evolution — self-evolving prompt, style adaptation, proactive agent."""

import json
import logging
import re
from collections import Counter
from datetime import datetime


def _safe_parse_llm_json(text: str, fallback):
    """Parse JSON from LLM output, tolerating control chars, truncation and wrong root type."""
    import re as _re, json as _json
    text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
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


logger = logging.getLogger(__name__)

_OVERLOAD_MARKERS = (
    "overwhelmed", "burned out", "burnt out", "stressed", "stress", "anxious",
    "exhausted", "too much", "can't keep up", "behind", "swamped",
    "перегруж", "выгор", "стресс", "тревож", "устал", "измот", "не успева",
)
_FOCUS_MARKERS = (
    "can't focus", "cannot focus", "distracted", "procrastin", "scattered",
    "unfocused", "too many tabs", "hard to start",
    "не могу сосредоточ", "отвлека", "прокраст", "не могу начать", "расфокус",
)
_RECOVERY_MARKERS = (
    "sleep", "slept", "insomnia", "tired", "rest", "break", "recovery",
    "сон", "не выспал", "не выспалась", "устал", "отдохн", "перерыв",
)
_ADMIN_MARKERS = (
    "renew", "renewal", "deadline", "expires", "expiration", "submission",
    "follow up", "follow-up", "send by", "remind", "calendar",
    "продлить", "истекает", "срок", "дедлайн", "напомн", "календар", "отправить до",
)
_HEALTH_MARKERS = (
    "exercise", "workout", "walk", "hydration", "food", "diet", "meditation",
    "здоров", "трениров", "прогул", "вода", "питани", "медитац",
)


# ══════════════════════════════════════════
# SELF-EVOLVING PROMPT (Friction Detection)
# ══════════════════════════════════════════

FRICTION_PATTERNS = [
    r"no,?\s+i\s+meant",
    r"that'?s\s+not\s+what",
    r"(?:^|\s)wrong\b",
    r"actually,?\s+i",
    r"try\s+again",
    r"not\s+what\s+i\s+asked",
    r"i\s+said\b",
    r"нет,?\s+я\s+имел",
    r"не\s+то\b",
    r"неправильно",
    r"я\s+просил",
    r"переделай",
]


def detect_friction(user_input: str) -> str | None:
    """Detect friction signals in user input. Returns signal_type or None."""
    text = user_input.lower()
    for pattern in FRICTION_PATTERNS:
        if re.search(pattern, text):
            return "correction"
    return None


def store_friction(db, user_id: str, signal_type: str,
                   user_input: str, agent_response: str):
    """Store friction signal in DB."""
    db.execute(
        """INSERT INTO friction_signals (user_id, signal_type, user_input,
           agent_response, created_at) VALUES (?, ?, ?, ?, ?)""",
        (user_id, signal_type, user_input[:500], agent_response[:500],
         datetime.now().isoformat()))
    db.commit()


async def synthesize_prompt_patches(provider, db, config: dict) -> list[str]:
    """Analyze accumulated friction signals and propose prompt improvements."""
    min_signals = config.get("min_friction_signals", 5)
    count = db.execute(
        "SELECT COUNT(*) FROM friction_signals WHERE extracted_lesson IS NULL"
    ).fetchone()[0]

    if count < min_signals:
        return []

    signals = db.execute(
        """SELECT user_input, agent_response, signal_type
           FROM friction_signals WHERE extracted_lesson IS NULL
           ORDER BY created_at DESC LIMIT 20""").fetchall()

    examples = "\n".join(
        f"- [{s[2]}] User: {s[0][:100]} | Agent: {s[1][:100]}"
        for s in signals)

    prompt = (
        "Analyze these friction signals (user corrections/misunderstandings) "
        "and propose 1-3 specific system prompt improvements:\n\n"
        f"{examples}\n\n"
        'Return JSON: {"patches":["improvement 1","improvement 2"]}\n'
        "Each patch should be a specific behavioral instruction."
    )

    model = config.get("extraction_model")
    if not model:
        from .metacognition import _resolve_model
        model = await _resolve_model(provider, config)
    if not model:
        return []
    result = await provider.complete(
        model=model, max_tokens=300,
        messages=[{"role": "user", "content": prompt}])
    text = result.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    # Strip thinking tags from thinking models
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    data = _safe_parse_llm_json(text, {"patches": []})
    if not isinstance(data, dict):
        data = {"patches": []}
    patches = data.get("patches", [])

    now = datetime.now().isoformat()
    for p in patches:
        db.execute(
            "INSERT INTO prompt_patches (patch_text, reason, created_at) VALUES (?, ?, ?)",
            (p, f"From {count} friction signals", now))
    # Mark signals as processed
    db.execute(
        "UPDATE friction_signals SET extracted_lesson = 'processed' "
        "WHERE extracted_lesson IS NULL")
    db.commit()
    return patches


def get_active_patches(db) -> list[str]:
    """Get all applied prompt patches for injection into system prompt."""
    rows = db.execute(
        "SELECT patch_text FROM prompt_patches WHERE applied = 1"
    ).fetchall()
    return [r[0] for r in rows]


# ══════════════════════════════════════════
# STYLE ADAPTATION
# ══════════════════════════════════════════

def analyze_style(text: str) -> dict:
    """Analyze message style features. Returns style dict."""
    words = text.split()
    word_count = max(len(words), 1)

    # Formality: presence of informal markers
    informal_markers = sum(1 for w in words if w.lower() in {
        "hey", "lol", "btw", "pls", "thx", "gonna", "wanna", "ya", "yep",
        "nope", "nah", "cool", "haha", "ok", "ty", "np", "bruh"})
    formality = max(0.0, min(1.0, 1.0 - informal_markers / word_count * 10))

    # Verbosity: avg sentence length
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    avg_sentence = (sum(len(s.split()) for s in sentences)
                    / max(len(sentences), 1))
    verbosity = min(1.0, avg_sentence / 20)

    # Technical level: code blocks, technical terms
    tech_markers = len(re.findall(
        r'`[^`]+`|```|\b(?:API|JSON|SQL|HTTP|async|function|class|import|'
        r'def|return|const|var|let)\b', text, re.I))
    technical = min(1.0, tech_markers / word_count * 20)

    # Emoji usage
    emoji_count = len(re.findall(
        r'[\U0001F600-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]', text))
    emoji = min(1.0, emoji_count / word_count * 10)

    # Language detection (simple heuristic)
    cyrillic = len(re.findall(r'[а-яА-ЯёЁ]', text))
    language = "ru" if cyrillic > len(text) * 0.3 else "en"

    return {"formality": formality, "verbosity": verbosity,
            "technical_level": technical, "emoji_usage": emoji,
            "language": language}


def update_style_profile(db, user_id: str, new_style: dict,
                         alpha: float = 0.3):
    """Update user style profile with exponential moving average."""
    row = db.execute(
        "SELECT formality, verbosity, technical_level, emoji_usage, language "
        "FROM style_profiles WHERE user_id = ?", (user_id,)).fetchone()

    now = datetime.now().isoformat()
    if row:
        updated = {
            "formality": row[0] * (1 - alpha) + new_style["formality"] * alpha,
            "verbosity": row[1] * (1 - alpha) + new_style["verbosity"] * alpha,
            "technical_level": row[2] * (1 - alpha) + new_style["technical_level"] * alpha,
            "emoji_usage": row[3] * (1 - alpha) + new_style["emoji_usage"] * alpha,
            "language": new_style["language"],
        }
        db.execute(
            """UPDATE style_profiles SET formality=?, verbosity=?,
               technical_level=?, emoji_usage=?, language=?, updated_at=?
               WHERE user_id=?""",
            (updated["formality"], updated["verbosity"],
             updated["technical_level"], updated["emoji_usage"],
             updated["language"], now, user_id))
    else:
        db.execute(
            "INSERT INTO style_profiles VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, new_style["formality"], new_style["verbosity"],
             new_style["technical_level"], new_style["emoji_usage"],
             new_style["language"], now))
    db.commit()


def get_style_instruction(db, user_id: str) -> str:
    """Generate style guidance for system prompt injection."""
    row = db.execute(
        "SELECT formality, verbosity, technical_level, emoji_usage, language "
        "FROM style_profiles WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        return ""

    parts = []
    if row[0] < 0.3:
        parts.append("Use casual, informal tone")
    elif row[0] > 0.7:
        parts.append("Use formal, professional tone")
    if row[1] < 0.3:
        parts.append("Be very concise and brief")
    elif row[1] > 0.7:
        parts.append("Provide detailed explanations")
    if row[2] > 0.7:
        parts.append("Use technical terminology freely")
    elif row[2] < 0.3:
        parts.append("Explain technical concepts simply")
    if row[3] > 0.5:
        parts.append("Include relevant emoji")
    if row[4] == "ru":
        parts.append("Respond in Russian")

    if not parts:
        return ""
    return "\n\n## Style guidance:\n" + "\n".join(f"- {p}" for p in parts)


# ══════════════════════════════════════════
# PROACTIVE AGENT
# ══════════════════════════════════════════

def detect_patterns(db, user_id: str, current_input: str,
                    config: dict) -> list[str]:
    """Detect predictable next actions based on interaction patterns.
    Returns list of suggested actions (max 2)."""
    window_days = config.get("pattern_window_days", 30)
    min_occurrences = config.get("min_pattern_occurrences", 3)
    min_confidence = config.get("min_confidence", 0.0)
    max_suggestions = config.get("max_suggestions", 2)

    rows = db.execute(
        """SELECT user_input, tool_calls_json, created_at, success, confidence
           FROM interaction_log
           WHERE user_id = ? AND created_at > datetime('now', ?)
           ORDER BY created_at""",
        (user_id, f"-{window_days} days")).fetchall()

    if len(rows) < min_occurrences:
        return []

    history = []
    for inp, tools, ts, success, confidence in rows:
        inp = (inp or "").strip()
        if not inp:
            continue
        if not success:
            continue
        if confidence is not None and confidence < min_confidence:
            continue
        history.append({
            "input": inp,
            "normalized": _normalize_input(inp),
            "created_at": ts or "",
            "tool_names": _extract_tool_names(tools),
        })

    if len(history) < min_occurrences:
        return []

    suggestions: list[str] = []
    seen: set[str] = set()
    current_norm = _normalize_input(current_input)
    current_display = _shorten_phrase(current_input)

    # Pattern 1: recurring workflow similar to current task
    recurring_matches = [
        item for item in history
        if current_norm and _inputs_similar(current_norm, item["normalized"])
    ]
    if len(recurring_matches) >= min_occurrences:
        tool_names = [name for item in recurring_matches for name in item["tool_names"]]
        common_tools = [name for name, count in Counter(tool_names).most_common(2) if count >= min_occurrences]
        suffix = f" Common tools in that workflow: {', '.join(common_tools)}." if common_tools else ""
        _append_suggestion(
            suggestions,
            seen,
            f"This looks like a recurring workflow ({len(recurring_matches)} similar runs). Offer to save or automate '{_shorten_phrase(current_display)}'.{suffix}",
        )

    # Pattern 2: likely next step after similar prior request
    follow_ups: Counter[str] = Counter()
    for i in range(len(history) - 1):
        prev_item = history[i]
        next_item = history[i + 1]
        if current_norm and _inputs_similar(current_norm, prev_item["normalized"]):
            if not _inputs_similar(prev_item["normalized"], next_item["normalized"]):
                follow_ups[next_item["input"]] += 1

    for followup, count in follow_ups.most_common():
        if count >= min_occurrences:
            _append_suggestion(
                suggestions,
                seen,
                f"After similar requests, the user often asks next to: {_shorten_phrase(followup)}",
            )
            break

    # Pattern 3: time-of-day habit
    current_hour = datetime.now().hour
    hour_actions: Counter[str] = Counter()
    for item in history:
        try:
            ts = item["created_at"]
            h = int(ts[11:13])
            if abs(h - current_hour) <= 1:
                key = item["input"]
                if key:
                    hour_actions[key] += 1
        except (ValueError, IndexError):
            pass

    for action, count in hour_actions.most_common():
        if count >= min_occurrences:
            _append_suggestion(
                suggestions,
                seen,
                f"Around this time, the user often works on: {_shorten_phrase(action)}",
            )
            break

    return suggestions[:max_suggestions]


def detect_human_support_opportunities(db, user_id: str, current_input: str,
                                       config: dict) -> list[str]:
    """Detect safe, high-value wellbeing and productivity support opportunities.

    These suggestions are intentionally lightweight: they help the agent reduce
    overload, improve focus, convert admin burden into reminders, and encourage
    healthy pacing. They must never diagnose or pressure the user.
    """
    if not config.get("enabled", True):
        return []

    max_suggestions = max(1, min(int(config.get("max_suggestions", 3) or 3), 5))
    min_occurrences = max(2, min(int(config.get("min_pattern_occurrences", 3) or 3), 8))
    late_night_hour = max(20, min(int(config.get("late_night_hour", 23) or 23), 23))
    early_hour = max(0, min(int(config.get("early_hour", 6) or 6), 8))
    history_days = max(7, min(int(config.get("pattern_window_days", 30) or 30), 120))

    rows = db.execute(
        """SELECT user_input, created_at, success
           FROM interaction_log
           WHERE user_id = ? AND created_at > datetime('now', ?)
           ORDER BY created_at DESC LIMIT 120""",
        (user_id, f"-{history_days} days"),
    ).fetchall()

    text = (current_input or "").lower()
    suggestions: list[str] = []
    seen: set[str] = set()

    recent_inputs = [str(r[0] or "") for r in rows if str(r[0] or "").strip()]
    late_night_count = 0
    admin_count = 0
    health_count = 0
    for user_input, created_at, success in rows:
        if not success:
            continue
        low = str(user_input or "").lower()
        if any(marker in low for marker in _ADMIN_MARKERS):
            admin_count += 1
        if any(marker in low for marker in _HEALTH_MARKERS):
            health_count += 1
        try:
            hour = int(str(created_at or "")[11:13])
        except (TypeError, ValueError, IndexError):
            hour = -1
        if hour >= late_night_hour or (0 <= hour <= early_hour):
            late_night_count += 1

    if any(marker in text for marker in _OVERLOAD_MARKERS):
        _append_suggestion(
            suggestions,
            seen,
            "The user sounds overloaded. Offer to reduce cognitive load: summarize the situation, choose the top 1-3 priorities, and convert follow-ups into tasks or reminders.",
        )

    if any(marker in text for marker in _FOCUS_MARKERS):
        _append_suggestion(
            suggestions,
            seen,
            "The user sounds unfocused. Offer a focus reset: define one concrete outcome, the smallest next 10-15 minute step, and remove nonessential work for now.",
        )

    if any(marker in text for marker in _RECOVERY_MARKERS) or late_night_count >= min_occurrences:
        _append_suggestion(
            suggestions,
            seen,
            "Protect the user's energy. Suggest a lighter pace, a short break, or scheduling non-urgent work for tomorrow instead of pushing through fatigue. Never diagnose health issues.",
        )

    if any(marker in text for marker in _ADMIN_MARKERS) or admin_count >= min_occurrences:
        _append_suggestion(
            suggestions,
            seen,
            "Recurring admin burden is visible. Offer to turn deadlines, renewals, submissions, and follow-ups into reminders, tasks, or calendar events automatically.",
        )

    if any(marker in text for marker in _HEALTH_MARKERS) or health_count >= min_occurrences:
        _append_suggestion(
            suggestions,
            seen,
            "If the user is trying to improve health habits, offer a simple sustainable system: one small routine, a reminder, and lightweight weekly check-ins rather than a heavy plan.",
        )

    if len(recent_inputs) >= min_occurrences:
        normalized = [_normalize_input(item) for item in recent_inputs if _normalize_input(item)]
        unique_norm = {item for item in normalized if item}
        if len(unique_norm) >= min_occurrences * 2 and any(len(item.split()) >= 2 for item in unique_norm):
            _append_suggestion(
                suggestions,
                seen,
                "The user appears to be context-switching across many topics. Offer to consolidate work into a short plan, batch related tasks, and keep a single running list of open loops.",
            )

    return suggestions[:max_suggestions]


def get_human_support_snapshot(db, user_id: str, current_input: str = "",
                               config: dict | None = None) -> dict:
    """Build a compact structured view of wellbeing/productivity support signals."""
    cfg = dict(config or {})
    cfg.setdefault("enabled", True)
    history_days = max(7, min(int(cfg.get("pattern_window_days", 30) or 30), 120))
    rows = db.execute(
        """SELECT user_input, created_at, success
           FROM interaction_log
           WHERE user_id = ? AND created_at > datetime('now', ?)
           ORDER BY created_at DESC LIMIT 120""",
        (user_id, f"-{history_days} days"),
    ).fetchall()

    metrics = {
        "recent_messages": 0,
        "late_night_sessions": 0,
        "admin_burden_signals": 0,
        "focus_signals": 0,
        "overload_signals": 0,
        "recovery_signals": 0,
        "health_habit_signals": 0,
    }
    latest_signals: list[str] = []

    for user_input, created_at, success in rows:
        if not success:
            continue
        low = str(user_input or "").lower()
        if not low.strip():
            continue
        metrics["recent_messages"] += 1
        if any(marker in low for marker in _ADMIN_MARKERS):
            metrics["admin_burden_signals"] += 1
        if any(marker in low for marker in _FOCUS_MARKERS):
            metrics["focus_signals"] += 1
        if any(marker in low for marker in _OVERLOAD_MARKERS):
            metrics["overload_signals"] += 1
        if any(marker in low for marker in _RECOVERY_MARKERS):
            metrics["recovery_signals"] += 1
        if any(marker in low for marker in _HEALTH_MARKERS):
            metrics["health_habit_signals"] += 1
        try:
            hour = int(str(created_at or "")[11:13])
        except (TypeError, ValueError, IndexError):
            hour = -1
        if hour >= int(cfg.get("late_night_hour", 23) or 23) or (0 <= hour <= int(cfg.get("early_hour", 6) or 6)):
            metrics["late_night_sessions"] += 1
        if len(latest_signals) < 4:
            latest_signals.append(str(user_input or "").strip())

    suggestions = detect_human_support_opportunities(db, user_id, current_input, cfg)
    priorities = []
    if metrics["overload_signals"] or metrics["focus_signals"]:
        priorities.append("reduce cognitive load")
    if metrics["admin_burden_signals"]:
        priorities.append("convert obligations into reminders")
    if metrics["late_night_sessions"] or metrics["recovery_signals"]:
        priorities.append("protect recovery and energy")
    if metrics["health_habit_signals"]:
        priorities.append("support consistent health habits")

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "metrics": metrics,
        "priorities": priorities[:4],
        "signals": latest_signals,
        "opportunities": suggestions,
        "updated_at": datetime.now().isoformat(),
    }


def _append_suggestion(suggestions: list[str], seen: set[str], text: str):
    key = text.lower().strip()
    if key and key not in seen:
        seen.add(key)
        suggestions.append(text)


def _shorten_phrase(text: str, limit: int = 80) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


_STOP_WORDS = {
    "the", "a", "an", "to", "for", "and", "or", "of", "in", "on", "with", "my",
    "this", "that", "is", "are", "be", "please", "me", "you", "i", "we",
    "как", "что", "это", "для", "мне", "меня", "тебя", "тебе", "и", "или",
    "на", "в", "по", "с", "к", "из", "а", "но", "не", "надо", "нужно",
}


def _normalize_input(text: str) -> str:
    words = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9_#./:-]+", (text or "").lower())
    filtered = [w for w in words if len(w) > 1 and w not in _STOP_WORDS]
    return " ".join(filtered[:12])


def _extract_tool_names(tools_json: str | None) -> list[str]:
    if not tools_json:
        return []
    try:
        data = json.loads(tools_json)
    except Exception:
        return []
    names = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                if name:
                    names.append(name)
    return names


def _inputs_similar(a: str, b: str) -> bool:
    """Quick check if two inputs are similar (word overlap > 50%)."""
    words_a = set(_normalize_input(a).split())
    words_b = set(_normalize_input(b).split())
    if not words_a or not words_b:
        return False
    if words_a == words_b:
        return True
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    union_ratio = len(words_a & words_b) / max(len(words_a | words_b), 1)
    return overlap >= 0.6 or union_ratio >= 0.5
