"""Tests for query_expansion module."""

import pytest
from liteagent.query_expansion import (
    expand_query,
    maybe_expand,
    should_expand,
)


class TestExpandQuery:
    def test_empty_string_returns_as_is(self):
        assert expand_query("") == ""

    def test_whitespace_only_returns_as_is(self):
        result = expand_query("   ")
        assert result == "   "

    def test_removes_english_stop_words(self):
        result = expand_query("what was that recipe you recommended last week")
        keywords = result.split()
        assert "what" not in keywords
        assert "was" not in keywords
        assert "that" not in keywords
        assert "you" not in keywords
        assert "last" not in keywords
        # "week" is a meaningful term and is kept

    def test_keeps_meaningful_words(self):
        result = expand_query("what was that recipe you recommended last week")
        keywords = result.split()
        assert "recipe" in keywords
        assert "recommended" in keywords

    def test_russian_query(self):
        result = expand_query("помнишь ту задачу про базу данных")
        keywords = result.split()
        # "про" and "ту" are stop words
        assert "про" not in keywords
        assert "ту" not in keywords
        # Meaningful words should remain
        assert any(k in keywords for k in ["задачу", "базу", "данных"])

    def test_short_keyword_query_returns_keywords(self):
        # pure keyword query should still work
        result = expand_query("database schema")
        assert len(result.split()) >= 1

    def test_pure_stop_words_returns_original(self):
        # All stop words → return original
        result = expand_query("what is the")
        assert result == "what is the"

    def test_max_keywords_limit(self):
        long_query = "database schema table index column relationship foreign key primary"
        result = expand_query(long_query, max_keywords=3)
        assert len(result.split()) <= 3

    def test_deduplication(self):
        result = expand_query("database database database schema schema")
        keywords = result.split()
        assert keywords.count("database") == 1
        assert keywords.count("schema") == 1

    def test_longer_tokens_ranked_higher(self):
        # "programming" (11 chars) > "code" (4 chars) → "programming" comes first
        result = expand_query("programming code")
        keywords = result.split()
        assert keywords.index("programming") < keywords.index("code")

    def test_numbers_excluded(self):
        result = expand_query("give me the results from 2023")
        keywords = result.split()
        assert "2023" not in keywords

    def test_very_short_result_returns_original(self):
        # If result is < 3 chars, return original
        result = expand_query("is an a")
        # All are short stop words, so original is returned
        assert result == "is an a"


class TestShouldExpand:
    def test_long_query_returns_true(self):
        assert should_expand("what was that thing we discussed about the project") is True

    def test_short_keyword_query_no_expand(self):
        assert should_expand("python code") is False

    def test_three_word_no_conversational(self):
        assert should_expand("database index schema") is False

    def test_conversational_english_triggers(self):
        assert should_expand("remember that task") is True
        assert should_expand("what was the recipe") is True

    def test_conversational_russian_triggers(self):
        assert should_expand("помнишь то задание") is True
        assert should_expand("расскажи про базу") is True

    def test_short_query_below_threshold(self):
        # 2 words, no conversational phrase
        assert should_expand("docker compose") is False


class TestMaybeExpand:
    def test_short_keyword_query_unchanged(self):
        query = "database schema"
        result = maybe_expand(query)
        assert result == query

    def test_long_conversational_expanded(self):
        query = "what was that recipe you recommended last week"
        result = maybe_expand(query)
        # Should be shorter (stop words removed)
        assert len(result) < len(query)

    def test_conversational_russian_expanded(self):
        query = "помнишь ту задачу про базу данных"
        result = maybe_expand(query)
        # should remove "ту" and "про"
        assert "ту" not in result.split()
        assert "про" not in result.split()

    def test_remember_phrase_triggers_expansion(self):
        query = "remember that discussion"
        result = maybe_expand(query)
        # "that" is a stop word — should be removed; "remember" and "discussion" are kept
        assert "that" not in result.split()
        assert "discussion" in result.split()
        assert "remember" in result.split()

    def test_empty_string(self):
        assert maybe_expand("") == ""
