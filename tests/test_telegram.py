"""Tests for Telegram photo and document handlers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_update(chat_id=123, user_id=456):
    """Create a mock Telegram Update with common attributes."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()
    return update


def _make_agent():
    """Create a mock agent with run() method."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value="I see a photo of a cat.")
    return agent


class TestPhotoHandler:
    """Test _make_photo_handler."""

    def _get_handler(self, agent, allowed=None):
        from liteagent.channels.telegram import _make_photo_handler
        return _make_photo_handler(agent, allowed)

    @pytest.mark.asyncio
    async def test_photo_sends_image_block(self):
        """Photo handler downloads photo and sends image content block to agent."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = _make_update()
        update.message.caption = None

        # Mock photo array (last = largest)
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"\xff\xd8\xff\xe0" + b"\x00" * 100))

        photo_size = MagicMock()
        photo_size.get_file = AsyncMock(return_value=mock_file)
        update.message.photo = [MagicMock(), photo_size]  # smallest, largest

        await handler(update, MagicMock())

        # Agent should receive content blocks list
        agent.run.assert_awaited_once()
        content_blocks = agent.run.call_args[0][0]
        assert isinstance(content_blocks, list)
        assert len(content_blocks) == 2

        # First block: text prompt
        assert content_blocks[0]["type"] == "text"
        assert "photo" in content_blocks[0]["text"].lower() or "describe" in content_blocks[0]["text"].lower()

        # Second block: image
        assert content_blocks[1]["type"] == "image"
        assert content_blocks[1]["source"]["media_type"] == "image/jpeg"

        # Response sent to user
        update.message.reply_text.assert_awaited()

    @pytest.mark.asyncio
    async def test_photo_with_caption(self):
        """Caption should be used as the text block instead of default."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = _make_update()
        update.message.caption = "What breed is this dog?"

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8" + b"\x00" * 50))
        photo_size = MagicMock()
        photo_size.get_file = AsyncMock(return_value=mock_file)
        update.message.photo = [photo_size]

        await handler(update, MagicMock())

        content_blocks = agent.run.call_args[0][0]
        assert content_blocks[0]["text"] == "What breed is this dog?"

    @pytest.mark.asyncio
    async def test_photo_filtered_by_chat_id(self):
        """Photo from non-allowed chat should be ignored."""
        agent = _make_agent()
        handler = self._get_handler(agent, allowed={999})

        update = _make_update(chat_id=123)
        update.message.photo = [MagicMock()]

        await handler(update, MagicMock())
        agent.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_photo_no_message(self):
        """No message → handler returns early."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = MagicMock()
        update.message = None

        await handler(update, MagicMock())
        agent.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_photo_typing_indicator(self):
        """Should show typing indicator while processing."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = _make_update()
        update.message.caption = None

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8" + b"\x00" * 50))
        photo_size = MagicMock()
        photo_size.get_file = AsyncMock(return_value=mock_file)
        update.message.photo = [photo_size]

        await handler(update, MagicMock())

        # Should have called typing at least once
        update.message.chat.send_action.assert_awaited()
        assert update.message.chat.send_action.call_args_list[0][0][0] == "typing"


class TestDocumentHandler:
    """Test _make_document_handler."""

    def _get_handler(self, agent, allowed=None):
        from liteagent.channels.telegram import _make_document_handler
        return _make_document_handler(agent, allowed)

    @pytest.mark.asyncio
    async def test_pdf_document(self):
        """PDF file → document content block."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = _make_update()
        update.message.caption = None
        update.message.document.file_name = "report.pdf"
        update.message.document.mime_type = "application/pdf"
        update.message.document.file_size = 5000

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"%PDF-1.4" + b"\x00" * 100))
        update.message.document.get_file = AsyncMock(return_value=mock_file)

        await handler(update, MagicMock())

        content_blocks = agent.run.call_args[0][0]
        assert len(content_blocks) == 2
        assert content_blocks[0]["type"] == "text"
        assert "report.pdf" in content_blocks[0]["text"]
        assert content_blocks[1]["type"] == "document"
        assert content_blocks[1]["source"]["media_type"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_python_file(self):
        """Python file → text content block."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = _make_update()
        update.message.caption = "Review this code"
        update.message.document.file_name = "main.py"
        update.message.document.mime_type = "text/x-python"
        update.message.document.file_size = 200

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b'def hello():\n    print("world")\n'))
        update.message.document.get_file = AsyncMock(return_value=mock_file)

        await handler(update, MagicMock())

        content_blocks = agent.run.call_args[0][0]
        assert content_blocks[0]["text"] == "Review this code"
        assert content_blocks[1]["type"] == "text"
        assert "def hello()" in content_blocks[1]["text"]
        assert "--- File: main.py ---" in content_blocks[1]["text"]

    @pytest.mark.asyncio
    async def test_too_large_file_rejected(self):
        """File over 10MB should be rejected before download."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = _make_update()
        update.message.document.file_name = "huge.zip"
        update.message.document.mime_type = "application/zip"
        update.message.document.file_size = 15 * 1024 * 1024  # 15 MB

        await handler(update, MagicMock())

        # Should NOT call agent.run
        agent.run.assert_not_awaited()
        # Should reply with size error
        update.message.reply_text.assert_awaited_once()
        error_text = update.message.reply_text.call_args[0][0]
        assert "too large" in error_text.lower()

    @pytest.mark.asyncio
    async def test_document_with_caption(self):
        """Caption should be used as text block."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = _make_update()
        update.message.caption = "Please analyze this CSV"
        update.message.document.file_name = "data.csv"
        update.message.document.mime_type = "text/csv"
        update.message.document.file_size = 500

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"name,age\nAlice,30\n"))
        update.message.document.get_file = AsyncMock(return_value=mock_file)

        await handler(update, MagicMock())

        content_blocks = agent.run.call_args[0][0]
        assert content_blocks[0]["text"] == "Please analyze this CSV"

    @pytest.mark.asyncio
    async def test_document_filtered_by_chat_id(self):
        """Document from non-allowed chat should be ignored."""
        agent = _make_agent()
        handler = self._get_handler(agent, allowed={999})

        update = _make_update(chat_id=123)
        update.message.document = MagicMock()

        await handler(update, MagicMock())
        agent.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_image_as_document(self):
        """Image sent as document → image content block."""
        agent = _make_agent()
        handler = self._get_handler(agent)

        update = _make_update()
        update.message.caption = None
        update.message.document.file_name = "photo.jpg"
        update.message.document.mime_type = "image/jpeg"
        update.message.document.file_size = 5000

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"\xff\xd8\xff\xe0" + b"\x00" * 100))
        update.message.document.get_file = AsyncMock(return_value=mock_file)

        await handler(update, MagicMock())

        content_blocks = agent.run.call_args[0][0]
        assert content_blocks[1]["type"] == "image"
        assert content_blocks[1]["source"]["media_type"] == "image/jpeg"


class TestProviderConversion:
    """Test that image blocks are properly converted by providers."""

    def test_openai_converts_image_block(self):
        """OpenAI _convert_messages should handle image blocks."""
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake")
        try:
            from liteagent.providers import OpenAIProvider
        except ImportError:
            pytest.skip("openai package not installed")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "abc123",
                }},
            ]
        }]
        result = OpenAIProvider._convert_messages(messages)

        # Should produce a single user message with parts
        assert len(result) == 1
        assert result[0]["role"] == "user"
        parts = result[0]["content"]
        assert isinstance(parts, list)
        assert len(parts) == 2
        assert parts[0]["type"] == "text"
        assert parts[1]["type"] == "image_url"
        assert "data:image/jpeg;base64,abc123" in parts[1]["image_url"]["url"]

    def test_openai_converts_document_to_text_note(self):
        """OpenAI should convert PDF document blocks to text notes."""
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake")
        try:
            from liteagent.providers import OpenAIProvider
        except ImportError:
            pytest.skip("openai package not installed")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Read this PDF"},
                {"type": "document", "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "pdfdata",
                }},
            ]
        }]
        result = OpenAIProvider._convert_messages(messages)
        parts = result[0]["content"]
        assert parts[1]["type"] == "text"
        assert "PDF" in parts[1]["text"]

    def test_openai_tool_result_still_works(self):
        """Verify tool_result blocks still work after refactor."""
        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake")
        try:
            from liteagent.providers import OpenAIProvider
        except ImportError:
            pytest.skip("openai package not installed")

        messages = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_123", "content": "Tool output"},
            ]
        }]
        result = OpenAIProvider._convert_messages(messages)
        assert any(m["role"] == "tool" for m in result)
        tool_msg = [m for m in result if m["role"] == "tool"][0]
        assert tool_msg["tool_call_id"] == "call_123"
        assert tool_msg["content"] == "Tool output"

    def test_gemini_converts_image_block(self):
        """Gemini _convert_messages should handle image blocks."""
        try:
            from liteagent.providers import GeminiProvider
        except ImportError:
            pytest.skip("google-generativeai package not installed")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "imgdata",
                }},
            ]
        }]
        result = GeminiProvider._convert_messages(messages)
        assert len(result) == 1
        parts = result[0]["parts"]
        assert len(parts) == 2
        assert parts[0] == "What is this?"
        assert parts[1]["inline_data"]["mime_type"] == "image/png"
        assert parts[1]["inline_data"]["data"] == "imgdata"
