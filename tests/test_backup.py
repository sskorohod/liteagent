"""Tests for backup module."""
import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from liteagent.backup import (
    backup, restore, prune_old_backups, list_backups,
    BACKUP_DIR, LITEAGENT_DIR,
)


class TestBackup:
    """Backup/restore cycle tests."""

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_path, monkeypatch):
        """Use temp directories for all tests."""
        self.test_dir = tmp_path / ".liteagent"
        self.test_dir.mkdir()
        self.backup_dir = self.test_dir / "backups"
        monkeypatch.setattr("liteagent.backup.LITEAGENT_DIR", self.test_dir)
        monkeypatch.setattr("liteagent.backup.BACKUP_DIR", self.backup_dir)
        # Patch _BACKUP_CANDIDATES
        monkeypatch.setattr("liteagent.backup._BACKUP_CANDIDATES", [
            self.test_dir / "memory.db",
            self.test_dir / "keys.json",
        ])

    def test_backup_creates_archive(self):
        """Backup creates a tar.gz file."""
        # Create test data
        (self.test_dir / "memory.db").write_text("test db")
        (self.test_dir / "keys.json").write_text('{"test": "key"}')

        path = backup()
        assert path.exists()
        assert path.suffix == ".gz"
        assert "liteagent_backup_" in path.name

    def test_backup_includes_config(self, tmp_path):
        """Backup includes config.json if path provided."""
        config_path = tmp_path / "config.json"
        config_path.write_text('{"agent": {}}')
        (self.test_dir / "memory.db").write_text("test db")

        path = backup(str(config_path))
        with tarfile.open(str(path), "r:gz") as tar:
            names = tar.getnames()
            assert "config.json" in names

    def test_restore_extracts_files(self, tmp_path):
        """Restore extracts tar.gz into liteagent dir."""
        (self.test_dir / "memory.db").write_text("test db")
        path = backup()

        # Clear and restore
        (self.test_dir / "memory.db").unlink()
        restore(str(path))
        assert (self.test_dir / "memory.db").exists()
        assert (self.test_dir / "memory.db").read_text() == "test db"

    def test_restore_rejects_path_traversal(self, tmp_path):
        """Restore rejects archives with path traversal."""
        evil_path = tmp_path / "evil.tar.gz"
        with tarfile.open(str(evil_path), "w:gz") as tar:
            import io
            data = b"evil content"
            info = tarfile.TarInfo(name="../../../etc/evil")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        with pytest.raises(ValueError, match="Unsafe path"):
            restore(str(evil_path))

    def test_prune_old_backups(self):
        """Prune keeps only N most recent backups."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            p = self.backup_dir / f"liteagent_backup_20240{i+1}01_000000.tar.gz"
            p.write_bytes(b"x")

        deleted = prune_old_backups(keep=3)
        assert deleted == 2
        remaining = list(self.backup_dir.glob("*.tar.gz"))
        assert len(remaining) == 3

    def test_list_backups(self):
        """list_backups returns metadata sorted newest first."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            p = self.backup_dir / f"liteagent_backup_20240{i+1}01_000000.tar.gz"
            p.write_bytes(b"x" * (i + 1) * 100)

        result = list_backups()
        assert len(result) == 3
        assert result[0]["name"] > result[1]["name"]  # newest first
        assert "size_kb" in result[0]
        assert "created_at" in result[0]

    def test_list_backups_empty(self):
        """list_backups returns empty list when no backups exist."""
        assert list_backups() == []
