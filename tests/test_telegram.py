"""Tests for Telegram channel — handlers, file sending, API client adapter."""

import io
import os
import tempfile

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_update(chat_id=123, user_id=456):
    """Create a mock Telegram Update with common attributes."""
    update = MagicMock()
    update.effective_chat.id = chat_id
    update.effective_user.id = user_id
    update.message.chat.send_action = AsyncMock()
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()
    update.message.reply_document = AsyncMock()
    update.message.reply_audio = AsyncMock()
    update.message.reply_video = AsyncMock()
    return update


def _make_api_client(chat_response=None, multimodal_response=None, command_response=None):
    """Create a mock API client (TelegramAPIClient interface)."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=chat_response or {
        "response": "Hello from API!",
        "files": [],
    })
    client.chat_multimodal = AsyncMock(return_value=multimodal_response or {
        "response": "I see a photo of a cat.",
        "files": [],
        "files_processed": [],
    })
    client.chat_voice = AsyncMock(return_value={
        "response": "You said hello!",
        "files": [],
    })
    client.command = AsyncMock(return_value=command_response or {
        "response": "Command done.",
        "files": [],
    })
    return client


class TestMessageHandler:
    """Test _make_message_handler with API client."""

    def _get_handler(self, api_client, allowed=None):
        from liteagent.channels.telegram import _make_message_handler
        return _make_message_handler(api_client, allowed)

    @pytest.mark.asyncio
    async def test_text_message_via_api(self):
        """Text message → api_client.chat() → reply_text()."""
        client = _make_api_client()
        handler = self._get_handler(client)

        update = _make_update()
        update.message.text = "Hello"

        await handler(update, MagicMock())

        client.chat.assert_awaited_once_with("Hello", "tg-456")
        update.message.reply_text.assert_awaited_once_with("Hello from API!")

    @pytest.mark.asyncio
    async def test_filtered_by_chat_id(self):
        """Message from non-allowed chat should be ignored."""
        client = _make_api_client()
        handler = self._get_handler(client, allowed={999})

        update = _make_update(chat_id=123)
        update.message.text = "Hello"

        await handler(update, MagicMock())
        client.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_message(self):
        """No message → handler returns early."""
        client = _make_api_client()
        handler = self._get_handler(client)

        update = MagicMock()
        update.message = None

        await handler(update, MagicMock())
        client.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_long_response_split(self):
        """Response > 4096 chars should be split."""
        long_text = "A" * 5000
        client = _make_api_client(chat_response={"response": long_text, "files": []})
        handler = self._get_handler(client)

        update = _make_update()
        update.message.text = "Test"

        await handler(update, MagicMock())

        assert update.message.reply_text.await_count == 2
        first_call = update.message.reply_text.call_args_list[0][0][0]
        assert len(first_call) == 4096


class TestPhotoHandler:
    """Test _make_photo_handler."""

    def _get_handler(self, api_client, allowed=None):
        from liteagent.channels.telegram import _make_photo_handler
        return _make_photo_handler(api_client, allowed)

    @pytest.mark.asyncio
    async def test_photo_sent_to_api(self):
        """Photo handler downloads photo and sends via api_client.chat_multimodal()."""
        client = _make_api_client()
        handler = self._get_handler(client)

        update = _make_update()
        update.message.caption = None

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"\xff\xd8\xff\xe0" + b"\x00" * 100))

        photo_size = MagicMock()
        photo_size.get_file = AsyncMock(return_value=mock_file)
        update.message.photo = [MagicMock(), photo_size]

        await handler(update, MagicMock())

        client.chat_multimodal.assert_awaited_once()
        call_args = client.chat_multimodal.call_args
        assert call_args.kwargs["user_id"] == "tg-456"
        assert len(call_args.kwargs["files"]) == 1
        assert call_args.kwargs["files"][0][0] == "photo.jpg"

        update.message.reply_text.assert_awaited()

    @pytest.mark.asyncio
    async def test_photo_with_caption(self):
        """Caption used as message text."""
        client = _make_api_client()
        handler = self._get_handler(client)

        update = _make_update()
        update.message.caption = "What breed is this dog?"

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8" + b"\x00" * 50))
        photo_size = MagicMock()
        photo_size.get_file = AsyncMock(return_value=mock_file)
        update.message.photo = [photo_size]

        await handler(update, MagicMock())

        call_args = client.chat_multimodal.call_args
        assert call_args.kwargs["message"] == "What breed is this dog?"

    @pytest.mark.asyncio
    async def test_photo_filtered_by_chat_id(self):
        """Photo from non-allowed chat should be ignored."""
        client = _make_api_client()
        handler = self._get_handler(client, allowed={999})

        update = _make_update(chat_id=123)
        update.message.photo = [MagicMock()]

        await handler(update, MagicMock())
        client.chat_multimodal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_photo_typing_indicator(self):
        """Should show typing indicator while processing."""
        client = _make_api_client()
        handler = self._get_handler(client)

        update = _make_update()
        update.message.caption = None

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"\xff\xd8" + b"\x00" * 50))
        photo_size = MagicMock()
        photo_size.get_file = AsyncMock(return_value=mock_file)
        update.message.photo = [photo_size]

        await handler(update, MagicMock())

        update.message.chat.send_action.assert_awaited()
        assert update.message.chat.send_action.call_args_list[0][0][0] == "typing"


class TestDocumentHandler:
    """Test _make_document_handler."""

    def _get_handler(self, api_client, allowed=None):
        from liteagent.channels.telegram import _make_document_handler
        return _make_document_handler(api_client, allowed)

    @pytest.mark.asyncio
    async def test_pdf_document(self):
        """PDF file sent via api_client.chat_multimodal()."""
        client = _make_api_client()
        handler = self._get_handler(client)

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

        client.chat_multimodal.assert_awaited_once()
        call_args = client.chat_multimodal.call_args
        assert call_args.kwargs["files"][0][0] == "report.pdf"
        assert call_args.kwargs["files"][0][2] == "application/pdf"

    @pytest.mark.asyncio
    async def test_too_large_file_rejected(self):
        """File over 10MB should be rejected before download."""
        client = _make_api_client()
        handler = self._get_handler(client)

        update = _make_update()
        update.message.document.file_name = "huge.zip"
        update.message.document.mime_type = "application/zip"
        update.message.document.file_size = 15 * 1024 * 1024

        await handler(update, MagicMock())

        client.chat_multimodal.assert_not_awaited()
        update.message.reply_text.assert_awaited_once()
        error_text = update.message.reply_text.call_args[0][0]
        assert "too large" in error_text.lower()

    @pytest.mark.asyncio
    async def test_document_with_caption(self):
        """Caption used as message text."""
        client = _make_api_client()
        handler = self._get_handler(client)

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

        call_args = client.chat_multimodal.call_args
        assert call_args.kwargs["message"] == "Please analyze this CSV"

    @pytest.mark.asyncio
    async def test_document_filtered_by_chat_id(self):
        """Document from non-allowed chat should be ignored."""
        client = _make_api_client()
        handler = self._get_handler(client, allowed={999})

        update = _make_update(chat_id=123)
        update.message.document = MagicMock()

        await handler(update, MagicMock())
        client.chat_multimodal.assert_not_awaited()


class TestVoiceHandler:
    """Test _make_voice_handler."""

    def _get_handler(self, api_client, allowed=None, config=None):
        from liteagent.channels.telegram import _make_voice_handler
        return _make_voice_handler(api_client, allowed, config or {})

    @pytest.mark.asyncio
    async def test_voice_sent_to_api(self):
        """Voice handler downloads audio and sends via api_client.chat_voice()."""
        client = _make_api_client()
        handler = self._get_handler(client)

        update = _make_update()
        voice = MagicMock()
        voice.duration = 5
        update.message.voice = voice
        update.message.audio = None

        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(
            return_value=bytearray(b"\x00" * 200))
        voice.get_file = AsyncMock(return_value=mock_file)

        await handler(update, MagicMock())

        client.chat_voice.assert_awaited_once()
        call_args = client.chat_voice.call_args
        assert call_args[0][1] == "tg-456"  # user_id
        assert call_args[0][2] == "5"  # duration


class TestSendFiles:
    """Test _send_files helper."""

    @pytest.mark.asyncio
    async def test_send_image(self):
        """Image file sent via reply_photo."""
        from liteagent.channels.telegram import _send_files

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            path = f.name

        message = MagicMock()
        message.reply_photo = AsyncMock()
        message.reply_document = AsyncMock()

        try:
            await _send_files(message, [{
                "path": path,
                "filename": "photo.jpg",
                "caption": "A photo",
                "mime_type": "image/jpeg",
                "delete_after_send": False,
            }])

            message.reply_photo.assert_awaited_once()
            call_args = message.reply_photo.call_args
            assert call_args.kwargs["caption"] == "A photo"
            message.reply_document.assert_not_awaited()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_send_document(self):
        """Non-image file sent via reply_document."""
        from liteagent.channels.telegram import _send_files

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"%PDF-1.4" + b"\x00" * 100)
            path = f.name

        message = MagicMock()
        message.reply_photo = AsyncMock()
        message.reply_document = AsyncMock()

        try:
            await _send_files(message, [{
                "path": path,
                "filename": "report.pdf",
                "caption": "Your report",
                "mime_type": "application/pdf",
                "delete_after_send": False,
            }])

            message.reply_document.assert_awaited_once()
            call_args = message.reply_document.call_args
            assert call_args.kwargs["caption"] == "Your report"
            assert call_args.kwargs["filename"] == "report.pdf"
            message.reply_photo.assert_not_awaited()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    @pytest.mark.asyncio
    async def test_send_with_cleanup(self):
        """File deleted after send when delete_after_send=True."""
        from liteagent.channels.telegram import _send_files

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"temp data")
            path = f.name

        message = MagicMock()
        message.reply_document = AsyncMock()

        await _send_files(message, [{
            "path": path,
            "filename": "temp.txt",
            "caption": "",
            "mime_type": "text/plain",
            "delete_after_send": True,
        }])

        assert not os.path.exists(path)

    @pytest.mark.asyncio
    async def test_send_missing_file(self):
        """Missing file should be skipped without error."""
        from liteagent.channels.telegram import _send_files

        message = MagicMock()
        message.reply_document = AsyncMock()

        await _send_files(message, [{
            "path": "/nonexistent/file.txt",
            "filename": "file.txt",
            "caption": "",
            "mime_type": "text/plain",
            "delete_after_send": True,
        }])

        message.reply_document.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_send_empty_files(self):
        """Empty files list → no sends."""
        from liteagent.channels.telegram import _send_files

        message = MagicMock()
        await _send_files(message, [])
        # No assertions needed — just checking no errors

    @pytest.mark.asyncio
    async def test_reply_with_result_text_and_files(self):
        """_reply_with_result sends both text and files."""
        from liteagent.channels.telegram import _reply_with_result

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"data")
            path = f.name

        message = MagicMock()
        message.reply_text = AsyncMock()
        message.reply_document = AsyncMock()

        try:
            await _reply_with_result(message, {
                "response": "Here is your file.",
                "files": [{
                    "path": path,
                    "filename": "data.txt",
                    "caption": "Report",
                    "mime_type": "text/plain",
                    "delete_after_send": False,
                }],
            })

            message.reply_text.assert_awaited_once_with("Here is your file.")
            message.reply_document.assert_awaited_once()
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestCommandHandler:
    """Test _make_command_handler."""

    def _get_handler(self, api_client, allowed=None):
        from liteagent.channels.telegram import _make_command_handler
        return _make_command_handler(api_client, allowed)

    @pytest.mark.asyncio
    async def test_command_via_api(self):
        """Command → api_client.command() → reply_text()."""
        client = _make_api_client()
        handler = self._get_handler(client)

        update = _make_update()
        update.message.text = "/help"

        await handler(update, MagicMock())

        client.command.assert_awaited_once_with("/help", "tg-456")
        update.message.reply_text.assert_awaited()

    @pytest.mark.asyncio
    async def test_unknown_command(self):
        """Unknown command → response None → 'Unknown command' message."""
        client = _make_api_client(command_response={"response": None, "files": []})
        handler = self._get_handler(client)

        update = _make_update()
        update.message.text = "/unknown"

        await handler(update, MagicMock())

        text = update.message.reply_text.call_args[0][0]
        assert "unknown" in text.lower() or "Unknown" in text


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
