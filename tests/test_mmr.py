"""Tests for MMR (Maximal Marginal Relevance) re-ranking module."""

import pytest
from liteagent.mmr import (
    MMRConfig,
    MMRItem,
    mmr_rerank,
    text_similarity,
    DEFAULT_MMR_CONFIG,
)


# ── MMRConfig ─────────────────────────────────────────────────────────────────

class TestMMRConfig:
    def test_defaults(self):
        cfg = MMRConfig()
        assert cfg.enabled is False
        assert cfg.lambda_ == 0.7

    def test_custom(self):
        cfg = MMRConfig(enabled=True, lambda_=0.5)
        assert cfg.enabled is True
        assert cfg.lambda_ == 0.5

    def test_default_config_constant(self):
        assert DEFAULT_MMR_CONFIG.enabled is False
        assert DEFAULT_MMR_CONFIG.lambda_ == 0.7


# ── text_similarity ───────────────────────────────────────────────────────────

class TestTextSimilarity:
    def test_identical_texts(self):
        s = text_similarity("hello world", "hello world")
        assert s == 1.0

    def test_disjoint_texts(self):
        s = text_similarity("apple banana", "cherry date")
        assert s == 0.0

    def test_partial_overlap(self):
        s = text_similarity("apple banana cherry", "banana cherry date")
        assert 0.0 < s < 1.0

    def test_case_insensitive(self):
        s = text_similarity("Hello World", "hello world")
        assert s == 1.0

    def test_empty_both(self):
        # Both empty → Jaccard = 1.0 (identical)
        s = text_similarity("", "")
        assert s == 1.0

    def test_one_empty(self):
        s = text_similarity("hello", "")
        assert s == 0.0


# ── mmr_rerank — disabled mode ────────────────────────────────────────────────

class TestMMRRerank:
    def _make_items(self, count: int) -> list[MMRItem]:
        return [
            MMRItem(id=str(i), score=1.0 - i * 0.1, content=f"item {i} unique content")
            for i in range(count)
        ]

    def test_disabled_returns_unchanged(self):
        items = self._make_items(5)
        cfg = MMRConfig(enabled=False)
        result = mmr_rerank(items, cfg)
        assert result == items

    def test_disabled_with_top_k(self):
        items = self._make_items(5)
        cfg = MMRConfig(enabled=False)
        result = mmr_rerank(items, cfg, top_k=3)
        assert result == items[:3]

    def test_none_config_uses_default(self):
        items = self._make_items(5)
        result = mmr_rerank(items, None)
        # default is disabled → unchanged
        assert result == items

    def test_single_item_returns_unchanged(self):
        items = [MMRItem(id="1", score=0.9, content="single item")]
        cfg = MMRConfig(enabled=True, lambda_=0.7)
        result = mmr_rerank(items, cfg)
        assert result == items

    def test_empty_returns_empty(self):
        result = mmr_rerank([], MMRConfig(enabled=True))
        assert result == []

    # ── enabled mode ──────────────────────────────────────────────────────

    def test_enabled_returns_top_k_items(self):
        items = self._make_items(10)
        cfg = MMRConfig(enabled=True, lambda_=0.7)
        result = mmr_rerank(items, cfg, top_k=5)
        assert len(result) == 5

    def test_enabled_no_top_k_returns_all(self):
        items = self._make_items(5)
        cfg = MMRConfig(enabled=True, lambda_=0.7)
        result = mmr_rerank(items, cfg)
        assert len(result) == 5

    def test_lambda_1_picks_by_relevance(self):
        """With λ=1.0 diversity has no weight → highest relevance wins first."""
        items = [
            MMRItem(id="a", score=0.9, content="apple banana cherry"),
            MMRItem(id="b", score=0.5, content="apple banana cherry"),  # identical content
            MMRItem(id="c", score=0.1, content="apple banana cherry"),
        ]
        cfg = MMRConfig(enabled=True, lambda_=1.0)
        result = mmr_rerank(items, cfg, top_k=2)
        assert result[0]["id"] == "a"
        assert result[1]["id"] == "b"

    def test_lambda_0_favours_diversity(self):
        """With λ=0.0 only diversity matters → diverse items preferred."""
        items = [
            MMRItem(id="a", score=0.9, content="python programming language"),
            MMRItem(id="b", score=0.8, content="python programming code"),  # similar to a
            MMRItem(id="c", score=0.7, content="music guitar jazz"),        # very different
        ]
        cfg = MMRConfig(enabled=True, lambda_=0.0)
        result = mmr_rerank(items, cfg, top_k=2)
        # First pick is best-relevance (no selected yet, so max_sim=0 for all)
        assert result[0]["id"] == "a"
        # Second should favour diversity → 'c' wins over 'b'
        assert result[1]["id"] == "c"

    def test_output_items_are_subset_of_input(self):
        items = self._make_items(8)
        cfg = MMRConfig(enabled=True, lambda_=0.5)
        result = mmr_rerank(items, cfg, top_k=4)
        result_ids = {r["id"] for r in result}
        input_ids = {i["id"] for i in items}
        assert result_ids <= input_ids

    def test_no_duplicates_in_output(self):
        items = self._make_items(6)
        cfg = MMRConfig(enabled=True, lambda_=0.7)
        result = mmr_rerank(items, cfg)
        ids = [r["id"] for r in result]
        assert len(ids) == len(set(ids))

    def test_top_k_larger_than_items(self):
        items = self._make_items(3)
        cfg = MMRConfig(enabled=True, lambda_=0.7)
        result = mmr_rerank(items, cfg, top_k=100)
        assert len(result) == 3

    def test_lambda_clamped(self):
        """Values outside [0,1] should not crash."""
        items = self._make_items(4)
        cfg = MMRConfig(enabled=True, lambda_=1.5)
        result = mmr_rerank(items, cfg, top_k=3)
        assert len(result) == 3

        cfg2 = MMRConfig(enabled=True, lambda_=-0.5)
        result2 = mmr_rerank(items, cfg2, top_k=3)
        assert len(result2) == 3
