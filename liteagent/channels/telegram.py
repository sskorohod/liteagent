"""Telegram channel adapter for LiteAgent — polling + webhook modes.

Supports two backends:
  - HTTP proxy: Telegram handlers send HTTP requests to local API (default when running alongside API)
  - Direct: Telegram handlers call agent.run() directly (standalone --channel telegram fallback)
"""

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# Max message length in Telegram
TG_MAX_LENGTH = 4096


# ══════════════════════════════════════════
# TELEGRAM API CLIENT (HTTP proxy to local API)
# ══════════════════════════════════════════

class TelegramAPIClient:
    """HTTP client for Telegram → local API communication."""

    def __init__(self, base_url: str, internal_token: str = ""):
        self.base_url = base_url.rstrip("/")
        self.internal_token = internal_token
        self._client = None

    async def _get_client(self):
        if self._client is None or self._client.is_closed:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={"x-internal-token": self.internal_token} if self.internal_token else {},
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def chat(self, message: str, user_id: str) -> dict:
        """POST /chat → {response, files}"""
        client = await self._get_client()
        resp = await client.post("/chat", json={"message": message, "user_id": user_id})
        resp.raise_for_status()
        return resp.json()

    async def chat_multimodal(self, message: str, user_id: str,
                               files: list[tuple[str, bytes, str]]) -> dict:
        """POST /chat/multimodal with file uploads.
        files: list of (filename, data_bytes, content_type)
        """
        client = await self._get_client()
        multipart_files = [("files", (fname, data, ct)) for fname, data, ct in files]
        resp = await client.post(
            "/chat/multimodal",
            data={"message": message, "user_id": user_id},
            files=multipart_files,
        )
        resp.raise_for_status()
        return resp.json()

    async def chat_voice(self, audio_bytes: bytes, user_id: str, duration: str = "0") -> dict:
        """POST /chat/voice with audio file."""
        client = await self._get_client()
        resp = await client.post(
            "/chat/voice",
            data={"user_id": user_id, "duration": duration},
            files=[("audio", ("voice.ogg", audio_bytes, "audio/ogg"))],
        )
        resp.raise_for_status()
        return resp.json()

    async def command(self, cmd: str, user_id: str) -> dict:
        """POST /command"""
        client = await self._get_client()
        resp = await client.post("/command", params={"cmd": cmd, "user_id": user_id})
        if resp.status_code == 404:
            return {"response": None, "files": []}
        resp.raise_for_status()
        return resp.json()

    async def health_check(self) -> bool:
        """Check if API is ready."""
        try:
            client = await self._get_client()
            resp = await client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


class _DirectAPIAdapter:
    """Adapter wrapping direct agent.run() to match TelegramAPIClient interface.
    Used for standalone --channel telegram mode (no API server).
    """

    def __init__(self, agent):
        self.agent = agent

    async def chat(self, message: str, user_id: str) -> dict:
        response = await self.agent.run(message, user_id)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def chat_multimodal(self, message: str, user_id: str,
                               files: list[tuple[str, bytes, str]]) -> dict:
        from ..multimodal import file_to_content_block
        content_blocks = [{"type": "text", "text": message}]
        for fname, data, ct in files:
            block = file_to_content_block(data, fname, ct)
            content_blocks.append(block)
            # Auto-ingest uploaded files to S3
            await self.agent.ingest_file(
                data, fname, source="telegram", user_id=user_id, mime_type=ct)
        response = await self.agent.run(content_blocks, user_id)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def chat_voice(self, audio_bytes: bytes, user_id: str, duration: str = "0") -> dict:
        import tempfile
        import uuid

        voice_id = f"voice_{uuid.uuid4().hex[:8]}"
        tmp_path = f"{tempfile.gettempdir()}/{voice_id}.ogg"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        # Auto-ingest voice to S3
        await self.agent.ingest_file(
            audio_bytes, f"{voice_id}.ogg",
            source="voice", user_id=user_id, mime_type="audio/ogg",
            description=f"Voice message ({duration}s)")

        tg_cfg = self.agent.config.get("channels", {}).get("telegram", {})
        if hasattr(self.agent, 'store_voice'):
            self.agent.store_voice(voice_id, audio_bytes, tg_cfg)

        # Build transcription prompt
        tool_hint = "Use any available transcription tool."
        if hasattr(self.agent, 'tools'):
            tool_names = list(self.agent.tools._tools.keys())
            mcp_voice = [n for n in tool_names if "transcribe_voice" in n]
            builtin_voice = "transcribe_voice" in tool_names
            if mcp_voice:
                tool_hint = (f"Use tool '{mcp_voice[0]}' with argument "
                             f"path=\"{tmp_path}\" to transcribe.")
            elif builtin_voice:
                tool_hint = (f"Use tool 'transcribe_voice' with argument "
                             f"voice_id=\"{voice_id}\" to transcribe.")

        voice_msg = (
            f"[User sent a voice message ({duration}s). "
            f"Audio file: {tmp_path} | voice_id: {voice_id}. "
            f"{tool_hint} "
            f"Transcribe the audio, then respond directly to the user's request. "
            f"Do NOT repeat or show the transcription text to the user — "
            f"just answer as if they typed the message themselves.]"
        )

        response = await self.agent.run(voice_msg, user_id)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def command(self, cmd: str, user_id: str) -> dict:
        result = self.agent.handle_command(cmd, user_id)
        return {"response": result, "files": []}

    async def close(self):
        pass


# ══════════════════════════════════════════
# FILE SENDING HELPER
# ══════════════════════════════════════════

async def _send_files(message, files: list[dict]):
    """Send queued files to user via Telegram Bot API."""
    for f in files:
        path = f.get("path", "")
        if not path or not os.path.exists(path):
            logger.warning("Queued file not found: %s", path)
            continue

        caption = f.get("caption", "") or None
        mime = f.get("mime_type", "")
        filename = f.get("filename", os.path.basename(path))

        try:
            with open(path, "rb") as fh:
                data = fh.read()
            import io
            buf = io.BytesIO(data)
            buf.name = filename

            if f.get("voice_compatible"):
                # Send as voice bubble (Opus format — round playback in Telegram)
                await message.reply_voice(voice=buf, caption=caption)
            elif mime.startswith("image/"):
                await message.reply_photo(photo=buf, caption=caption)
            elif mime.startswith("audio/"):
                await message.reply_audio(audio=buf, caption=caption, filename=filename)
            elif mime.startswith("video/"):
                await message.reply_video(video=buf, caption=caption, filename=filename)
            else:
                await message.reply_document(document=buf, caption=caption, filename=filename)
        except Exception as e:
            logger.error("Failed to send file %s: %s", filename, e)

        # Cleanup temp files
        if f.get("delete_after_send", True):
            try:
                os.unlink(path)
            except OSError:
                pass


async def _reply_with_result(message, result: dict):
    """Send API result (text + files) back to Telegram user."""
    response = result.get("response", "")
    if response:
        for i in range(0, len(response), TG_MAX_LENGTH):
            await message.reply_text(response[i:i + TG_MAX_LENGTH])
    files = result.get("files", [])
    if files:
        await _send_files(message, files)


# ══════════════════════════════════════════
# MESSAGE HANDLERS (unified — work with both TelegramAPIClient and _DirectAPIAdapter)
# ══════════════════════════════════════════

def _make_message_handler(api_client, allowed_chat_ids: set | None = None):
    """Create handler for text messages."""
    async def handle_message(update, context):
        if not update.message or not update.message.text:
            return
        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            logger.info("Ignoring message from chat_id=%s (not in allowed list)", chat_id)
            return

        user_id = f"tg-{update.effective_user.id}"
        text = update.message.text
        logger.info("Message from %s (chat=%s): %s", user_id, chat_id, text[:80])

        try:
            result = await api_client.chat(text, user_id)
            await _reply_with_result(update.message, result)
        except Exception as e:
            logger.error("Error handling Telegram message: %s", e, exc_info=True)
            await update.message.reply_text(f"Error: {e}")

    return handle_message


def _make_command_handler(api_client, allowed_chat_ids: set | None = None):
    """Create handler for /slash commands."""
    async def handle_command(update, context):
        if not update.message or not update.message.text:
            return
        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            return

        user_id = f"tg-{update.effective_user.id}"
        text = update.message.text
        logger.info("Command from %s (chat=%s): %s", user_id, chat_id, text)

        try:
            result = await api_client.command(text, user_id)
            response = result.get("response")
            if response is not None:
                for i in range(0, len(response), TG_MAX_LENGTH):
                    await update.message.reply_text(response[i:i + TG_MAX_LENGTH])
            else:
                await update.message.reply_text(
                    "Unknown command. Use /help to see available commands.")
        except Exception as e:
            logger.error("Error handling command: %s", e, exc_info=True)
            await update.message.reply_text(f"Error: {e}")

    return handle_command


def _make_voice_handler(api_client, allowed_chat_ids: set | None, config: dict):
    """Create handler for voice/audio messages."""
    async def handle_voice(update, context):
        if not update.message:
            return
        voice = update.message.voice or update.message.audio
        if not voice:
            return

        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            logger.info("Ignoring voice from chat_id=%s (not in allowed list)", chat_id)
            return

        user_id = f"tg-{update.effective_user.id}"
        duration = str(getattr(voice, 'duration', '?'))
        logger.info("Voice message from %s (chat=%s, duration=%ss)", user_id, chat_id, duration)

        await update.message.chat.send_action("typing")

        try:
            file = await voice.get_file()
            audio_bytes = bytes(await file.download_as_bytearray())
            logger.info("Downloaded voice file: %d bytes", len(audio_bytes))

            await update.message.chat.send_action("typing")
            result = await api_client.chat_voice(audio_bytes, user_id, duration)
            await _reply_with_result(update.message, result)

        except Exception as e:
            logger.error("Error handling voice message: %s", e, exc_info=True)
            await update.message.reply_text(f"Error: {e}")

    return handle_voice


def _make_photo_handler(api_client, allowed_chat_ids: set | None = None):
    """Create handler for photo messages."""
    async def handle_photo(update, context):
        if not update.message or not update.message.photo:
            return
        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            return

        user_id = f"tg-{update.effective_user.id}"
        caption = update.message.caption or "User sent a photo. Describe what you see and respond."

        await update.message.chat.send_action("typing")

        try:
            photo = update.message.photo[-1]
            file = await photo.get_file()
            photo_bytes = bytes(await file.download_as_bytearray())

            logger.info("Photo from %s (chat=%s): %d bytes, caption='%s'",
                        user_id, chat_id, len(photo_bytes), caption[:60])

            await update.message.chat.send_action("typing")
            result = await api_client.chat_multimodal(
                message=caption,
                user_id=user_id,
                files=[("photo.jpg", photo_bytes, "image/jpeg")],
            )
            await _reply_with_result(update.message, result)

        except Exception as e:
            logger.error("Error handling photo: %s", e, exc_info=True)
            await update.message.reply_text(f"Error processing photo: {e}")

    return handle_photo


def _make_document_handler(api_client, allowed_chat_ids: set | None = None):
    """Create handler for document/file messages."""
    async def handle_document(update, context):
        if not update.message or not update.message.document:
            return
        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            return

        user_id = f"tg-{update.effective_user.id}"
        doc = update.message.document
        caption = update.message.caption or ""
        filename = doc.file_name or "document"
        mime_type = doc.mime_type or ""

        # Pre-check file size (Telegram bot API limit: 20MB, our limit: 10MB)
        if doc.file_size and doc.file_size > 10 * 1024 * 1024:
            size_mb = doc.file_size / (1024 * 1024)
            await update.message.reply_text(
                f"File '{filename}' is too large ({size_mb:.1f} MB). Maximum: 10 MB.")
            return

        await update.message.chat.send_action("typing")

        try:
            file = await doc.get_file()
            doc_bytes = bytes(await file.download_as_bytearray())

            logger.info("Document from %s (chat=%s): '%s' (%s, %d bytes)",
                        user_id, chat_id, filename, mime_type, len(doc_bytes))

            message = caption or f"User sent a file: {filename}. Analyze its content and respond."

            await update.message.chat.send_action("typing")
            result = await api_client.chat_multimodal(
                message=message,
                user_id=user_id,
                files=[(filename, doc_bytes, mime_type)],
            )
            await _reply_with_result(update.message, result)

        except Exception as e:
            logger.error("Error handling document '%s': %s", filename, e, exc_info=True)
            await update.message.reply_text(f"Error processing file: {e}")

    return handle_document


# ══════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════

def _parse_chat_ids(config: dict) -> set | None:
    """Parse allowed chat_ids from config. Returns None if not set (accept all)."""
    raw = config.get("chat_id") or config.get("chat_ids")
    if not raw:
        return None
    if isinstance(raw, (int, str)):
        raw = [raw]
    ids = set()
    for cid in raw:
        try:
            ids.add(int(cid))
        except (ValueError, TypeError):
            pass
    return ids if ids else None


BOT_COMMANDS = [
    ("help", "Список команд"),
    ("model", "Показать/сменить модель"),
    ("clear", "Очистить историю диалога"),
    ("memories", "Показать воспоминания"),
    ("usage", "Расход токенов и бюджет"),
]


async def _set_bot_commands(bot, has_rag: bool = False):
    """Set bot menu commands visible in Telegram."""
    try:
        from telegram import BotCommand
    except ImportError:
        return

    commands = [BotCommand(cmd, desc) for cmd, desc in BOT_COMMANDS]
    if has_rag:
        commands.append(BotCommand("ingest", "Загрузить файл/папку в RAG"))
    try:
        await bot.delete_my_commands()
        await bot.set_my_commands(commands)
        logger.info("Bot menu commands set: %s", [c.command for c in commands])
    except Exception as e:
        logger.warning("Failed to set bot commands: %s", e)


def _register_all_handlers(tg_app, api_client, allowed_chat_ids, config):
    """Register all message handlers on a telegram Application."""
    try:
        from telegram.ext import CommandHandler, MessageHandler, filters
    except ImportError:
        raise ImportError("python-telegram-bot is required: pip install liteagent[telegram]")

    # Text messages
    msg_handler = _make_message_handler(api_client, allowed_chat_ids)
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_handler))

    # Slash commands
    cmd_handler = _make_command_handler(api_client, allowed_chat_ids)
    known_cmds = [c for c, _ in BOT_COMMANDS] + ["ingest", "forget", "conflicts", "documents"]
    for cmd_name in known_cmds:
        tg_app.add_handler(CommandHandler(cmd_name, cmd_handler))

    # Voice messages
    if config.get("voice_enabled", True):
        voice_handler = _make_voice_handler(api_client, allowed_chat_ids, config)
        tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_handler))
        logger.info("Voice handler registered")

    # Photo messages
    photo_handler = _make_photo_handler(api_client, allowed_chat_ids)
    tg_app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    logger.info("Photo handler registered")

    # Document/file messages
    doc_handler = _make_document_handler(api_client, allowed_chat_ids)
    tg_app.add_handler(MessageHandler(filters.Document.ALL, doc_handler))
    logger.info("Document handler registered")


# ══════════════════════════════════════════
# STANDALONE POLLING MODE (--channel telegram, no API server)
# ══════════════════════════════════════════

async def run_telegram(agent, config: dict):
    """Run Telegram bot — dispatches to polling or webhook based on config."""
    mode = config.get("mode", "polling")
    if mode == "webhook":
        raise ValueError(
            "Webhook mode requires FastAPI. Use --channel api with "
            "telegram webhook_url configured, or use polling mode."
        )
    await _run_polling(agent, config)


async def _run_polling(agent, config: dict):
    """Run Telegram bot with long-polling (standalone, no API server).
    Uses _DirectAPIAdapter for backward compatibility.
    """
    try:
        from telegram.ext import Application
    except ImportError:
        raise ImportError("python-telegram-bot is required: pip install liteagent[telegram]")

    token = config.get("token")
    if not token:
        raise ValueError("Telegram token not configured. Set token in channels.telegram config.")

    allowed_chat_ids = _parse_chat_ids(config)
    api_client = _DirectAPIAdapter(agent)

    app = Application.builder().token(token).build()
    _register_all_handlers(app, api_client, allowed_chat_ids, config)

    logging.getLogger("liteagent").setLevel(logging.INFO)
    if allowed_chat_ids:
        logger.info("Telegram bot started (polling), allowed chats: %s", allowed_chat_ids)
    else:
        logger.info("Telegram bot started (polling), accepting all chats")
    print("[Jess] Telegram bot running (standalone). Send a message to your bot!")

    await app.initialize()
    await app.updater.start_polling(drop_pending_updates=True)
    await app.start()

    agent._telegram_app = app

    has_rag = hasattr(agent, '_rag') and agent._rag is not None
    await _set_bot_commands(app.bot, has_rag=has_rag)

    try:
        stop_event = asyncio.Event()
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await api_client.close()
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


# ══════════════════════════════════════════
# WEBHOOK MODE
# ══════════════════════════════════════════

def setup_webhook_route(fastapi_app, agent, tg_config: dict):
    """Mount Telegram webhook route on a FastAPI app."""
    try:
        from telegram import Update
        from telegram.ext import Application
    except ImportError:
        raise ImportError("python-telegram-bot is required: pip install liteagent[telegram]")

    token = tg_config.get("token")
    webhook_url = tg_config.get("webhook_url")
    webhook_secret = tg_config.get("webhook_secret", "")

    if not token or not webhook_url:
        raise ValueError("Telegram webhook requires token and webhook_url in config")

    tg_app = Application.builder().token(token).build()
    allowed_chat_ids = _parse_chat_ids(tg_config)

    # Use HTTP proxy if internal token available, otherwise direct adapter
    internal_token = getattr(fastapi_app.state, 'internal_token', '')
    api_cfg = agent.config.get("channels", {}).get("api", {})
    host = api_cfg.get("host", "127.0.0.1")
    port = api_cfg.get("port", 8080)
    api_host = "127.0.0.1" if host == "0.0.0.0" else host
    api_client = TelegramAPIClient(f"http://{api_host}:{port}", internal_token)

    _register_all_handlers(tg_app, api_client, allowed_chat_ids, tg_config)

    @fastapi_app.on_event("startup")
    async def _setup_webhook():
        await tg_app.initialize()
        await tg_app.start()
        has_rag = hasattr(agent, '_rag') and agent._rag is not None
        await _set_bot_commands(tg_app.bot, has_rag=has_rag)
        await tg_app.bot.set_webhook(
            url=f"{webhook_url}/telegram/webhook",
            secret_token=webhook_secret)
        logger.info("Telegram webhook set to %s/telegram/webhook", webhook_url)

    @fastapi_app.on_event("shutdown")
    async def _cleanup_webhook():
        await api_client.close()
        await tg_app.stop()
        await tg_app.shutdown()

    @fastapi_app.post("/telegram/webhook")
    async def telegram_webhook(request):
        """Receive Telegram webhook updates."""
        body = await request.json()
        update = Update.de_json(body, tg_app.bot)
        await tg_app.process_update(update)
        return {"ok": True}

    logger.info("Telegram webhook route mounted")
