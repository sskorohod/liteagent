"""Tests for Telegram file upload → store → recall flow."""

import sqlite3

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Helpers ──────────────────────────────────────────────

def _make_update(chat_id=123, user_id=456):
    """Create a mock Telegram Update."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()
    update.message.reply_document = AsyncMock()
    return update


def _make_api_client():
    client = MagicMock()
    client.chat_multimodal = AsyncMock(return_value={
        "response": "File received.",
        "files": [],
        "files_processed": [],
    })
    return client


def _make_file_index_db(tmp_path):
    """Create a SQLite DB with file_index table and sample data."""
    db_path = str(tmp_path / "test.db")
    db = sqlite3.connect(db_path)
    db.execute("""
        CREATE TABLE IF NOT EXISTS file_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            storage_key TEXT UNIQUE NOT NULL,
            original_name TEXT NOT NULL,
            mime_type TEXT DEFAULT '',
            size_bytes INTEGER DEFAULT 0,
            source TEXT DEFAULT 'unknown',
            user_id TEXT DEFAULT 'system',
            description TEXT DEFAULT '',
            embedding BLOB,
            created_at TEXT DEFAULT (datetime('now')),
            accessed_at TEXT DEFAULT (datetime('now'))
        )
    """)
    db.commit()
    return db


# ══════════════════════════════════════════
# DOCUMENT HANDLER — INGEST NOTE
# ══════════════════════════════════════════

class TestDocumentHandlerIngestNote:
    """Verify that document handler includes ingest confirmation in message."""

    @pytest.mark.asyncio
    async def test_document_handler_sends_ingest_note(self):
        """Document upload should include storage confirmation note."""
        from liteagent.channels.telegram import _make_document_handler

        client = _make_api_client()
        handler = _make_document_handler(client)

        update = _make_update()
        doc = MagicMock()
        doc.file_name = "report.pdf"
        doc.mime_type = "application/pdf"
        doc.file_size = 5000
        file_mock = MagicMock()
        file_mock.download_as_bytearray = AsyncMock(return_value=bytearray(b"PDF content"))
        doc.get_file = AsyncMock(return_value=file_mock)
        update.message.document = doc
        update.message.caption = None

        await handler(update, MagicMock())

        # Check that chat_multimodal was called with ingest note
        call_args = client.chat_multimodal.call_args
        message = call_args.kwargs.get("message") or call_args[1].get("message", "")
        assert "auto-saved to storage" in message
        assert "report.pdf" in message

    @pytest.mark.asyncio
    async def test_document_handler_with_caption(self):
        """Document with caption should include both caption and ingest note."""
        from liteagent.channels.telegram import _make_document_handler

        client = _make_api_client()
        handler = _make_document_handler(client)

        update = _make_update()
        doc = MagicMock()
        doc.file_name = "notes.txt"
        doc.mime_type = "text/plain"
        doc.file_size = 200
        file_mock = MagicMock()
        file_mock.download_as_bytearray = AsyncMock(return_value=bytearray(b"hello"))
        doc.get_file = AsyncMock(return_value=file_mock)
        update.message.document = doc
        update.message.caption = "Мой файл"

        await handler(update, MagicMock())

        call_args = client.chat_multimodal.call_args
        message = call_args.kwargs.get("message") or call_args[1].get("message", "")
        assert "Мой файл" in message
        assert "auto-saved to storage" in message


# ══════════════════════════════════════════
# PHOTO HANDLER — INGEST NOTE
# ══════════════════════════════════════════

class TestPhotoHandlerIngestNote:
    """Verify that photo handler includes ingest confirmation."""

    @pytest.mark.asyncio
    async def test_photo_handler_sends_ingest_note(self):
        """Photo upload should include storage confirmation note."""
        from liteagent.channels.telegram import _make_photo_handler

        client = _make_api_client()
        handler = _make_photo_handler(client)

        update = _make_update()
        photo_mock = MagicMock()
        file_mock = MagicMock()
        file_mock.download_as_bytearray = AsyncMock(return_value=bytearray(b"\x89PNG"))
        photo_mock.get_file = AsyncMock(return_value=file_mock)
        update.message.photo = [photo_mock]
        update.message.caption = "Скриншот"

        await handler(update, MagicMock())

        call_args = client.chat_multimodal.call_args
        message = call_args.kwargs.get("message") or call_args[1].get("message", "")
        assert "auto-saved to storage" in message
        assert "Скриншот" in message


# ══════════════════════════════════════════
# FILE CATALOG IN SYSTEM PROMPT
# ══════════════════════════════════════════

class TestFileCatalogInSystemPrompt:
    """Verify that file catalog is injected into agent system prompt."""

    def test_file_catalog_injected(self, tmp_path):
        """_build_feature_section should include file catalog when files exist."""
        db = _make_file_index_db(tmp_path)
        db.execute("""
            INSERT INTO file_index (storage_key, original_name, mime_type,
                                    size_bytes, source, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        """, ("files/tg/abc_report.pdf", "report.pdf", "application/pdf",
              50000, "telegram", "tg-456"))
        db.commit()

        # Create a minimal FileManager mock
        from liteagent.file_manager import FileManager
        fm = FileManager(storage=MagicMock(), db=db)

        # Create a minimal agent mock to test _build_feature_section
        agent = MagicMock()
        agent._file_manager = fm
        agent._features = {}
        agent.config = {}
        agent.skill_registry = MagicMock()
        agent.skill_registry.get_triggered_prompt = MagicMock(return_value="")

        from liteagent.agent import LiteAgent
        result = LiteAgent._build_feature_section(agent, "hello", "tg-456")

        assert "report.pdf" in result
        assert "files/tg/abc_report.pdf" in result
        assert "search_files" in result

    def test_no_catalog_when_no_files(self, tmp_path):
        """No catalog section when user has no files."""
        db = _make_file_index_db(tmp_path)

        from liteagent.file_manager import FileManager
        fm = FileManager(storage=MagicMock(), db=db)

        agent = MagicMock()
        agent._file_manager = fm
        agent._features = {}
        agent.config = {}
        agent.skill_registry = MagicMock()
        agent.skill_registry.get_triggered_prompt = MagicMock(return_value="")

        from liteagent.agent import LiteAgent
        result = LiteAgent._build_feature_section(agent, "hello", "tg-456")

        assert "Файлы пользователя" not in result


# ══════════════════════════════════════════
# KB AUTO-INGEST
# ══════════════════════════════════════════

class TestKBAutoIngest:
    """Verify Knowledge Base auto-ingest for documents."""

    def _make_agent_mock(self):
        """Create agent mock with class-level constants for ingest_file."""
        from liteagent.agent import LiteAgent
        agent = MagicMock()
        agent._KB_DOCUMENT_MIMES = LiteAgent._KB_DOCUMENT_MIMES
        agent._KB_DOCUMENT_EXTS = LiteAgent._KB_DOCUMENT_EXTS
        return agent

    @pytest.mark.asyncio
    async def test_kb_auto_ingest_pdf(self):
        """PDF files should be auto-ingested into Knowledge Base."""
        agent = self._make_agent_mock()
        fm = MagicMock()
        fm.ingest = AsyncMock(return_value={"storage_key": "files/tg/abc.pdf"})
        agent._file_manager = fm

        kb = MagicMock()
        kb.ingest = AsyncMock(return_value={"status": "ok"})
        agent._knowledge_base = kb

        from liteagent.agent import LiteAgent
        result = await LiteAgent.ingest_file(
            agent, b"%PDF-1.4 content", "invoice.pdf",
            source="telegram", user_id="tg-456",
            mime_type="application/pdf")

        assert result is not None
        fm.ingest.assert_awaited_once()
        kb.ingest.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_kb_auto_ingest_skips_images(self):
        """Image files should NOT be ingested into Knowledge Base."""
        agent = self._make_agent_mock()
        fm = MagicMock()
        fm.ingest = AsyncMock(return_value={"storage_key": "files/tg/abc.jpg"})
        agent._file_manager = fm

        kb = MagicMock()
        kb.ingest = AsyncMock()
        agent._knowledge_base = kb

        from liteagent.agent import LiteAgent
        result = await LiteAgent.ingest_file(
            agent, b"\x89PNG", "photo.jpg",
            source="telegram", user_id="tg-456",
            mime_type="image/jpeg")

        assert result is not None
        kb.ingest.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_kb_auto_ingest_markdown(self):
        """Markdown files should be auto-ingested into KB."""
        agent = self._make_agent_mock()
        fm = MagicMock()
        fm.ingest = AsyncMock(return_value={"storage_key": "files/tg/abc.md"})
        agent._file_manager = fm

        kb = MagicMock()
        kb.ingest = AsyncMock(return_value={"status": "ok"})
        agent._knowledge_base = kb

        from liteagent.agent import LiteAgent
        result = await LiteAgent.ingest_file(
            agent, b"# Hello\n\nWorld", "notes.md",
            source="telegram", user_id="tg-456",
            mime_type="text/markdown")

        assert result is not None
        kb.ingest.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_kb_auto_ingest_docx(self):
        """Extractable office docs should be auto-ingested into KB."""
        agent = self._make_agent_mock()
        fm = MagicMock()
        fm.ingest = AsyncMock(return_value={"storage_key": "files/tg/abc.docx"})
        agent._file_manager = fm

        kb = MagicMock()
        kb.ingest = AsyncMock(return_value={"status": "ok"})
        agent._knowledge_base = kb

        from liteagent.agent import LiteAgent
        result = await LiteAgent.ingest_file(
            agent,
            b"PK\x03\x04" + b"\x00" * 64,
            "contract.docx",
            source="telegram",
            user_id="tg-456",
            mime_type="application/octet-stream",
        )

        assert result is not None
        kb.ingest.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_kb_auto_ingest_graceful_on_failure(self):
        """KB ingest failure should not break file ingestion."""
        agent = self._make_agent_mock()
        fm = MagicMock()
        fm.ingest = AsyncMock(return_value={"storage_key": "files/tg/abc.pdf"})
        agent._file_manager = fm

        kb = MagicMock()
        kb.ingest = AsyncMock(side_effect=RuntimeError("KB broken"))
        agent._knowledge_base = kb

        from liteagent.agent import LiteAgent
        result = await LiteAgent.ingest_file(
            agent, b"%PDF", "doc.pdf",
            source="telegram", user_id="tg-456",
            mime_type="application/pdf")

        # Should succeed despite KB failure
        assert result is not None
        assert result["storage_key"] == "files/tg/abc.pdf"


# ══════════════════════════════════════════
# /files COMMAND
# ══════════════════════════════════════════

class TestFilesCommand:
    """Verify /files Telegram command."""

    def test_files_command_returns_list(self, tmp_path):
        """'/files' should list stored files."""
        db = _make_file_index_db(tmp_path)
        db.execute("""
            INSERT INTO file_index (storage_key, original_name, mime_type,
                                    size_bytes, source, user_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, ("files/tg/abc_report.pdf", "report.pdf", "application/pdf",
              50000, "telegram", "tg-456", "2026-03-01 10:00:00"))
        db.commit()

        from liteagent.file_manager import FileManager
        fm = FileManager(storage=MagicMock(), db=db)

        from liteagent.channels.telegram import TelegramCommandHandler
        cmd = TelegramCommandHandler(
            memory=MagicMock(), config={}, file_manager=fm)

        result = cmd.handle("/files", "tg-456")
        assert "report.pdf" in result
        assert "48KB" in result  # 50000 // 1024 = 48

    def test_files_command_empty(self, tmp_path):
        """'/files' with no files should say so."""
        db = _make_file_index_db(tmp_path)

        from liteagent.file_manager import FileManager
        fm = FileManager(storage=MagicMock(), db=db)

        from liteagent.channels.telegram import TelegramCommandHandler
        cmd = TelegramCommandHandler(
            memory=MagicMock(), config={}, file_manager=fm)

        result = cmd.handle("/files", "tg-456")
        assert "нет сохранённых" in result

    def test_files_command_no_file_manager(self):
        """'/files' without file manager should say storage not configured."""
        from liteagent.channels.telegram import TelegramCommandHandler
        cmd = TelegramCommandHandler(
            memory=MagicMock(), config={})

        result = cmd.handle("/files", "tg-456")
        assert "не настроено" in result

    def test_help_includes_files(self):
        """'/help' should mention /files command."""
        from liteagent.channels.telegram import TelegramCommandHandler
        cmd = TelegramCommandHandler(
            memory=MagicMock(), config={})

        result = cmd.handle("/help", "tg-456")
        assert "/files" in result
