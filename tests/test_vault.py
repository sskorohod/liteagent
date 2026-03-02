"""Tests for vault module."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestVault:
    """Vault encryption/decryption tests."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.test_dir = tmp_path / ".liteagent"
        self.test_dir.mkdir()
        self.keys_path = self.test_dir / "keys.json"
        self.enc_path = self.test_dir / "keys.enc"
        monkeypatch.setattr("liteagent.vault.KEYS_DIR", self.test_dir)
        monkeypatch.setattr("liteagent.vault.KEYS_PATH", self.keys_path)
        monkeypatch.setattr("liteagent.vault.VAULT_PATH", self.enc_path)

    def test_vault_disabled_without_env(self, monkeypatch):
        monkeypatch.delenv("LITEAGENT_VAULT_KEY", raising=False)
        from liteagent.vault import vault_enabled
        assert vault_enabled() is False

    def test_vault_enabled_with_env(self, monkeypatch):
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "test-secret-password")
        from liteagent.vault import vault_enabled
        assert vault_enabled() is True

    def test_save_and_load_keys(self, monkeypatch):
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "test-secret-password")
        from liteagent.vault import save_keys, load_keys
        keys = {"anthropic": "sk-ant-test123", "openai": "sk-test456"}
        save_keys(keys)
        assert self.enc_path.exists()
        loaded = load_keys()
        assert loaded == keys

    def test_vault_set_and_list(self, monkeypatch):
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "test-secret-password")
        from liteagent.vault import vault_set, vault_list, load_keys
        vault_set("anthropic", "sk-ant-new-key")
        vault_set("openai", "sk-openai-key")
        providers = vault_list()
        assert "anthropic" in providers
        assert "openai" in providers
        keys = load_keys()
        assert keys["anthropic"] == "sk-ant-new-key"

    def test_vault_delete(self, monkeypatch):
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "test-secret-password")
        from liteagent.vault import vault_set, vault_delete, vault_list
        vault_set("test_provider", "test-key")
        assert "test_provider" in vault_list()
        deleted = vault_delete("test_provider")
        assert deleted is True
        assert "test_provider" not in vault_list()

    def test_vault_delete_nonexistent(self, monkeypatch):
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "test-secret-password")
        from liteagent.vault import vault_delete
        assert vault_delete("nonexistent") is False

    def test_migrate_to_vault(self, monkeypatch):
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "test-secret-password")
        # Create plain keys.json
        keys = {"anthropic": "sk-ant-test", "openai": "sk-openai-test"}
        self.keys_path.write_text(json.dumps(keys))
        from liteagent.vault import migrate_to_vault, load_keys
        migrate_to_vault()
        # Encrypted file should exist
        assert self.enc_path.exists()
        # Plain file should be gone
        assert not self.keys_path.exists()
        # Can still load keys
        loaded = load_keys()
        assert loaded == keys

    def test_wrong_password_fails(self, monkeypatch):
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "password1")
        from liteagent.vault import save_keys, load_keys
        save_keys({"test": "value"})
        # Change password
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "password2")
        # load_keys catches exceptions and returns {} instead of raising
        loaded = load_keys()
        assert loaded == {}

    def test_load_empty_vault(self, monkeypatch):
        monkeypatch.setenv("LITEAGENT_VAULT_KEY", "test-secret")
        from liteagent.vault import load_keys
        # No vault file yet
        assert load_keys() == {}
