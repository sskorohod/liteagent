"""Tests for conv_model module (per-conversation model overrides)."""

import sqlite3
import pytest

from liteagent.conv_model import (
    ConversationModelStore,
    parse_model_command,
    resolve_model_for_user,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def store(db):
    return ConversationModelStore(db)


# ── ConversationModelStore ────────────────────────────────────────────────────

class TestConversationModelStore:
    def test_get_returns_none_for_unknown_user(self, store):
        assert store.get("user123") is None

    def test_set_and_get(self, store):
        store.set("user1", "claude-opus-4-20250115")
        assert store.get("user1") == "claude-opus-4-20250115"

    def test_set_overwrites(self, store):
        store.set("user1", "claude-haiku-4-5-20251001")
        store.set("user1", "gpt-4o")
        assert store.get("user1") == "gpt-4o"

    def test_set_with_provider(self, store):
        store.set("user1", "gpt-4o", provider="openai")
        meta = store.get_with_meta("user1")
        assert meta["provider"] == "openai"

    def test_set_with_set_via(self, store):
        store.set("user1", "gpt-4o", set_via="dashboard")
        meta = store.get_with_meta("user1")
        assert meta["set_via"] == "dashboard"

    def test_clear_existing(self, store):
        store.set("user1", "gpt-4o")
        removed = store.clear("user1")
        assert removed is True
        assert store.get("user1") is None

    def test_clear_nonexistent(self, store):
        removed = store.clear("nonexistent")
        assert removed is False

    def test_clear_all(self, store):
        store.set("user1", "gpt-4o")
        store.set("user2", "gemini-2.5-pro")
        store.set("user3", "claude-haiku-4-5-20251001")
        count = store.clear_all()
        assert count == 3
        assert store.get("user1") is None
        assert store.get("user2") is None

    def test_list_all_empty(self, store):
        assert store.list_all() == []

    def test_list_all_returns_all(self, store):
        store.set("user1", "gpt-4o")
        store.set("user2", "gemini-2.5-pro")
        items = store.list_all()
        assert len(items) == 2
        user_ids = {i["user_id"] for i in items}
        assert "user1" in user_ids
        assert "user2" in user_ids

    def test_list_all_has_meta_fields(self, store):
        store.set("user1", "gpt-4o", provider="openai", set_via="command")
        items = store.list_all()
        assert len(items) == 1
        item = items[0]
        assert "model" in item
        assert "provider" in item
        assert "set_at" in item
        assert "set_via" in item

    def test_get_with_meta_returns_none_for_unknown(self, store):
        assert store.get_with_meta("unknown") is None

    def test_get_with_meta_returns_dict(self, store):
        store.set("user1", "claude-opus-4-20250115", provider="anthropic", set_via="telegram")
        meta = store.get_with_meta("user1")
        assert meta is not None
        assert meta["model"] == "claude-opus-4-20250115"
        assert meta["provider"] == "anthropic"
        assert meta["set_via"] == "telegram"
        assert "set_at" in meta

    def test_multiple_users_isolated(self, store):
        store.set("user1", "gpt-4o")
        store.set("user2", "gemini-2.5-pro")
        assert store.get("user1") == "gpt-4o"
        assert store.get("user2") == "gemini-2.5-pro"

    def test_schema_created_on_init(self, db):
        # Re-create store on same db — should not fail
        store2 = ConversationModelStore(db)
        assert store2.get("x") is None


# ── parse_model_command ───────────────────────────────────────────────────────

class TestParseModelCommand:
    def test_no_arg_returns_show(self):
        model, err = parse_model_command("/model")
        assert model is None
        assert err == "show"

    def test_reset_variants(self):
        for arg in ("reset", "clear", "default", "off"):
            model, err = parse_model_command(f"/model {arg}")
            assert model == "__reset__", f"Expected __reset__ for /model {arg}"
            assert err == ""

    def test_haiku_shortcut(self):
        model, err = parse_model_command("/model haiku")
        assert model == "claude-haiku-4-5-20251001"
        assert err == ""

    def test_sonnet_shortcut(self):
        model, err = parse_model_command("/model sonnet")
        assert model == "claude-sonnet-4-20250514"
        assert err == ""

    def test_opus_shortcut(self):
        model, err = parse_model_command("/model opus")
        assert model == "claude-opus-4-20250115"
        assert err == ""

    def test_gpt4_shortcut(self):
        model, err = parse_model_command("/model gpt4")
        assert model == "gpt-4o"
        assert err == ""

    def test_gpt4mini_shortcut(self):
        model, err = parse_model_command("/model gpt4mini")
        assert model == "gpt-4o-mini"
        assert err == ""

    def test_flash_shortcut(self):
        model, err = parse_model_command("/model flash")
        assert model == "gemini-2.5-flash"
        assert err == ""

    def test_pro_shortcut(self):
        model, err = parse_model_command("/model pro")
        assert model == "gemini-2.5-pro"
        assert err == ""

    def test_full_model_name_passthrough(self):
        model, err = parse_model_command("/model claude-opus-4-20250115")
        assert model == "claude-opus-4-20250115"
        assert err == ""

    def test_unknown_name_passthrough(self):
        model, err = parse_model_command("/model my-custom-model-v2")
        assert model == "my-custom-model-v2"
        assert err == ""

    def test_case_insensitive_shortcut(self):
        model, _ = parse_model_command("/model HAIKU")
        assert model == "claude-haiku-4-5-20251001"

    def test_extra_whitespace(self):
        model, _ = parse_model_command("/model  opus  ")
        assert model == "claude-opus-4-20250115"


# ── resolve_model_for_user ────────────────────────────────────────────────────

class TestResolveModelForUser:
    def test_no_store_returns_default(self):
        result = resolve_model_for_user(None, "user1", "claude-sonnet-4-20250514")
        assert result == "claude-sonnet-4-20250514"

    def test_no_override_returns_default(self, store):
        result = resolve_model_for_user(store, "user1", "claude-sonnet-4-20250514")
        assert result == "claude-sonnet-4-20250514"

    def test_override_returned(self, store):
        store.set("user1", "gpt-4o")
        result = resolve_model_for_user(store, "user1", "claude-sonnet-4-20250514")
        assert result == "gpt-4o"

    def test_different_user_gets_default(self, store):
        store.set("user1", "gpt-4o")
        result = resolve_model_for_user(store, "user2", "claude-sonnet-4-20250514")
        assert result == "claude-sonnet-4-20250514"

    def test_after_clear_returns_default(self, store):
        store.set("user1", "gpt-4o")
        store.clear("user1")
        result = resolve_model_for_user(store, "user1", "claude-sonnet-4-20250514")
        assert result == "claude-sonnet-4-20250514"
