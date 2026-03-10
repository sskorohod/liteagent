"""Query expansion for FTS-only search mode.

Ported from OpenClaw query-expansion.ts.

When no embedding provider is available, FTS (full-text search) works best
with specific keywords.  Users often ask conversational queries like
"that thing we discussed yesterday" — this module extracts meaningful
keywords from such queries to improve FTS recall.

Usage:
    from liteagent.query_expansion import expand_query

    keywords = expand_query("what was that recipe you recommended?")
    # → "recipe recommended"
"""

from __future__ import annotations

import re

# ── stop words ────────────────────────────────────────────────────────────────

_STOP_EN: frozenset[str] = frozenset([
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # pronouns
    "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "it", "they", "them", "their",
    # auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might",
    # prepositions
    "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "over",
    # conjunctions
    "and", "or", "but", "if", "because", "as", "while",
    # filler
    "just", "also", "already", "so", "then", "than",
    "more", "most", "very", "too", "much",
    # time references that are conversational (not useful as FTS keywords)
    "yesterday", "today", "tomorrow", "ago",
    "last", "next", "recent", "previously",
    # relative references
    "here", "there", "where", "when", "how", "what", "which", "who", "whom",
    "something", "anything", "everything", "nothing",
    "someone", "anyone", "everyone",
    "thing", "things",
])

# Russian stop words
_STOP_RU: frozenset[str] = frozenset([
    "и", "в", "во", "не", "что", "он", "на", "я", "с",
    "со", "как", "а", "то", "все", "она", "так", "его",
    "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "её", "мне", "было", "вот",
    "от", "меня", "ещё", "нет", "о", "из", "ему",
    "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него",
    "до", "вас", "нибудь", "опять", "уж", "вам", "ведь",
    "там", "потом", "себя", "ничего", "ей", "может",
    "они", "тут", "где", "есть", "надо", "ней", "для",
    "мы", "тебя", "их", "чем", "была", "сам", "чтоб",
    "без", "будто", "человек", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот",
    "этого", "того", "том", "нас", "про", "его", "своей",
    "этой", "перед", "иногда", "лучше", "чуть",
    # conversational references
    "вчера", "сегодня", "завтра", "недавно",
    "раньше", "прошлый", "последний",
    "тот", "та", "те", "эти",
    "это", "такой", "такая", "такие",
    "какой", "какая", "которой", "которая",
])

_ALL_STOPS = _STOP_EN | _STOP_RU

# Minimum token length to include
_MIN_TOKEN_LEN = 3

# Preferred maximum number of keywords to return
_MAX_KEYWORDS = 8


# ── helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens (letters + digits + CJK)."""
    # Keep CJK characters as single-char tokens — they're meaningful
    tokens = re.findall(r"[\w\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+", text.lower())
    return tokens


def _is_meaningful(token: str) -> bool:
    """Return True if token should be kept as a keyword."""
    if len(token) < _MIN_TOKEN_LEN:
        return False
    if token in _ALL_STOPS:
        return False
    # Pure numbers are rarely helpful FTS keywords
    if token.isdigit():
        return False
    return True


def _score_token(token: str) -> float:
    """Heuristic relevance score for ranking tokens."""
    # Longer tokens tend to be more specific
    return float(len(token))


# ── public API ────────────────────────────────────────────────────────────────

def expand_query(query: str, max_keywords: int = _MAX_KEYWORDS) -> str:
    """Extract meaningful keywords from *query* for use in FTS.

    Returns a space-separated keyword string.  If the query is already short
    and keyword-like, returns it unchanged.

    Examples:
        "what was that recipe you recommended last week?" → "recipe recommended"
        "помнишь ту задачу про базу данных?" → "задачу базу данных"
    """
    if not query or not query.strip():
        return query

    tokens = _tokenize(query)
    keywords = [t for t in tokens if _is_meaningful(t)]

    if not keywords:
        return query  # nothing to filter — return original

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in keywords:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    # Sort by score (descending), then cap
    ranked = sorted(unique, key=_score_token, reverse=True)
    result = " ".join(ranked[:max_keywords])

    # If result is very short relative to original, keep original
    if len(result) < 3:
        return query

    return result


def should_expand(query: str) -> bool:
    """Return True if *query* is conversational and benefits from expansion.

    Short keyword queries (< 3 words) are already FTS-friendly.
    """
    words = query.strip().split()
    # Long queries or queries with stop words likely need expansion
    if len(words) > 4:
        return True
    conversational_phrases = [
        "remember", "recall", "what was", "that thing",
        "you said", "we discussed", "помнишь", "что было",
        "то что", "мы говорили", "расскажи про",
    ]
    lower = query.lower()
    return any(ph in lower for ph in conversational_phrases)


def maybe_expand(query: str, max_keywords: int = _MAX_KEYWORDS) -> str:
    """Expand query only when it seems conversational; otherwise return as-is."""
    if should_expand(query):
        return expand_query(query, max_keywords)
    return query
