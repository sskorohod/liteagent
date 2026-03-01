"""Evolution — self-evolving prompt, style adaptation, proactive agent."""

import json
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


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

    model = config.get("extraction_model", "claude-haiku-4-5-20251001")
    result = await provider.complete(
        model=model, max_tokens=300,
        messages=[{"role": "user", "content": prompt}])
    text = result.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
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

    rows = db.execute(
        """SELECT user_input, tool_calls_json, created_at
           FROM interaction_log
           WHERE user_id = ? AND created_at > datetime('now', ?)
           ORDER BY created_at""",
        (user_id, f"-{window_days} days")).fetchall()

    if len(rows) < min_occurrences:
        return []

    suggestions = []

    # Pattern 1: Time-of-day patterns
    current_hour = datetime.now().hour
    hour_actions = {}
    for inp, tools, ts in rows:
        try:
            h = int(ts[11:13])
            if abs(h - current_hour) <= 1:
                key = (inp or "")[:50].lower()
                if key:
                    hour_actions[key] = hour_actions.get(key, 0) + 1
        except (ValueError, IndexError):
            pass

    for action, count in sorted(hour_actions.items(), key=lambda x: -x[1]):
        if count >= min_occurrences:
            suggestions.append(f"You often do this around now: {action}")
            break

    # Pattern 2: Sequential patterns (A usually follows B)
    current_lower = current_input.lower()[:50]
    follow_ups = {}
    for i in range(len(rows) - 1):
        prev_inp = (rows[i][0] or "")[:50].lower()
        next_inp = (rows[i + 1][0] or "")[:50].lower()
        if prev_inp and next_inp and _inputs_similar(current_lower, prev_inp):
            follow_ups[next_inp] = follow_ups.get(next_inp, 0) + 1

    for followup, count in sorted(follow_ups.items(), key=lambda x: -x[1]):
        if count >= min_occurrences:
            suggestions.append(f"You might want next: {followup}")
            break

    return suggestions[:2]


def _inputs_similar(a: str, b: str) -> bool:
    """Quick check if two inputs are similar (word overlap > 50%)."""
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
    return overlap > 0.5
