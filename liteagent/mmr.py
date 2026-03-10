"""Maximal Marginal Relevance (MMR) re-ranking.

Ported from OpenClaw mmr.ts (Carbonell & Goldstein, 1998).

MMR balances relevance with diversity by iteratively selecting results
that maximize: λ * relevance - (1-λ) * max_similarity_to_selected

Usage:
    from liteagent.mmr import mmr_rerank, MMRConfig

    items = [{"id": "1", "score": 0.9, "content": "..."},  ...]
    config = MMRConfig(enabled=True, lambda_=0.7)
    reranked = mmr_rerank(items, config, top_k=5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TypedDict


class MMRItem(TypedDict):
    """An item eligible for MMR re-ranking."""
    id: str
    score: float      # relevance score [0..1]
    content: str      # text content for diversity computation


@dataclass
class MMRConfig:
    """Configuration for MMR re-ranking."""
    enabled: bool = False
    lambda_: float = 0.7  # 0 = max diversity, 1 = max relevance


DEFAULT_MMR_CONFIG = MMRConfig(enabled=False, lambda_=0.7)


# ── tokenization ─────────────────────────────────────────────────────────────

def _tokenize(text: str) -> frozenset[str]:
    """Extract alphanumeric tokens, normalised to lowercase."""
    return frozenset(re.findall(r"[a-z0-9_]+", text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity ∈ [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def text_similarity(content_a: str, content_b: str) -> float:
    """Jaccard similarity on tokenised content."""
    return _jaccard(_tokenize(content_a), _tokenize(content_b))


# ── core algorithm ────────────────────────────────────────────────────────────

def mmr_rerank(
    items: list[MMRItem],
    config: MMRConfig | None = None,
    top_k: int | None = None,
) -> list[MMRItem]:
    """Re-rank *items* using MMR.

    If config.enabled is False (default), returns items unchanged.
    If top_k is None, returns all items in re-ranked order.
    """
    cfg = config or DEFAULT_MMR_CONFIG
    if not cfg.enabled or len(items) <= 1:
        return items[:top_k] if top_k else items

    lam = max(0.0, min(1.0, cfg.lambda_))
    k = top_k if top_k is not None else len(items)
    k = min(k, len(items))

    remaining = list(items)
    selected: list[MMRItem] = []
    token_cache: dict[str, frozenset[str]] = {}

    def get_tokens(item: MMRItem) -> frozenset[str]:
        iid = item["id"]
        if iid not in token_cache:
            token_cache[iid] = _tokenize(item["content"])
        return token_cache[iid]

    for _ in range(k):
        if not remaining:
            break

        best_idx = -1
        best_score = float("-inf")

        for i, candidate in enumerate(remaining):
            relevance = candidate["score"]
            max_sim = max(
                (text_similarity(candidate["content"], s["content"]) for s in selected),
                default=0.0,
            )
            # Compute token sets lazily
            if selected:
                toks_c = get_tokens(candidate)
                max_sim = max(
                    _jaccard(toks_c, get_tokens(s)) for s in selected
                )
            mmr_score = lam * relevance - (1 - lam) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx == -1:
            break

        selected.append(remaining.pop(best_idx))

    return selected
