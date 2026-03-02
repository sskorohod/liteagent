"""Telegram channel adapter for LiteAgent — polling + webhook modes."""

import asyncio
import logging
import tempfile
import uuid

logger = logging.getLogger(__name__)

# Max message length in Telegram
TG_MAX_LENGTH = 4096


# ── Message handlers ────────────────────────────────────────

def _make_message_handler(agent, allowed_chat_ids: set | None = None):
    """Create reusable message handler for both polling and webhook modes."""
    async def handle_message(update, context):
        if not update.message or not update.message.text:
            return

        # Filter by chat_id if configured
        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            logger.info("Ignoring message from chat_id=%s (not in allowed list)", chat_id)
            return

        user_id = f"tg-{update.effective_user.id}"
        text = update.message.text
        logger.info("Message from %s (chat=%s): %s", user_id, chat_id, text[:80])

        # Run agent
        try:
            response = await agent.run(text, user_id)
            logger.info("Response to %s: %s", user_id, response[:80])
            for i in range(0, len(response), TG_MAX_LENGTH):
                await update.message.reply_text(response[i:i + TG_MAX_LENGTH])
        except Exception as e:
            logger.error("Error handling Telegram message: %s", e, exc_info=True)
            await update.message.reply_text(f"Error: {e}")

    return handle_message


def _make_command_handler(agent, allowed_chat_ids: set | None = None):
    """Create handler for /slash commands (shown in Telegram menu)."""
    async def handle_command(update, context):
        if not update.message or not update.message.text:
            return

        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            return

        user_id = f"tg-{update.effective_user.id}"
        text = update.message.text
        logger.info("Command from %s (chat=%s): %s", user_id, chat_id, text)

        cmd_response = agent.handle_command(text, user_id)
        if cmd_response is not None:
            for i in range(0, len(cmd_response), TG_MAX_LENGTH):
                await update.message.reply_text(cmd_response[i:i + TG_MAX_LENGTH])
        else:
            await update.message.reply_text("Unknown command. Use /help to see available commands.")

    return handle_command


def _make_voice_handler(agent, allowed_chat_ids: set | None, config: dict):
    """Create handler for voice/audio messages.

    Downloads audio from Telegram, stores it in agent._voice_store,
    and passes a message to agent.run() so the agent can use the
    transcribe_voice tool to transcribe and respond.
    """
    async def handle_voice(update, context):
        if not update.message:
            return

        # Accept both voice messages and audio files
        voice = update.message.voice or update.message.audio
        if not voice:
            return

        # Filter by chat_id
        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            logger.info("Ignoring voice from chat_id=%s (not in allowed list)", chat_id)
            return

        user_id = f"tg-{update.effective_user.id}"
        duration = getattr(voice, 'duration', '?')
        logger.info("Voice message from %s (chat=%s, duration=%ss)",
                     user_id, chat_id, duration)

        # Show typing indicator while processing
        await update.message.chat.send_action("typing")

        try:
            # Download voice file from Telegram
            file = await voice.get_file()
            audio_bytes = await file.download_as_bytearray()
            logger.info("Downloaded voice file: %d bytes", len(audio_bytes))

            # Save audio to temp file (accessible by MCP server or built-in tool)
            voice_id = f"voice_{uuid.uuid4().hex[:8]}"
            tmp_path = f"{tempfile.gettempdir()}/{voice_id}.ogg"
            with open(tmp_path, "wb") as f:
                f.write(bytes(audio_bytes))
            logger.info("Saved voice to temp file: %s", tmp_path)

            # Also store in agent for built-in transcribe_voice tool (fallback)
            agent.store_voice(voice_id, bytes(audio_bytes), config)

            # Tell the agent about the voice message — it will use a transcription tool
            # Determine correct tool name based on what's registered
            if hasattr(agent, 'tools'):
                tool_names = list(agent.tools._tools.keys())
                mcp_voice = [n for n in tool_names if "transcribe_voice" in n]
                builtin_voice = "transcribe_voice" in tool_names
                if mcp_voice:
                    tool_hint = (f"Use tool '{mcp_voice[0]}' with argument "
                                 f"path=\"{tmp_path}\" to transcribe.")
                elif builtin_voice:
                    tool_hint = (f"Use tool 'transcribe_voice' with argument "
                                 f"voice_id=\"{voice_id}\" to transcribe.")
                else:
                    tool_hint = "Use any available transcription tool."
            else:
                tool_hint = "Use any available transcription tool."

            voice_msg = (
                f"[User sent a voice message ({duration}s). "
                f"Audio file: {tmp_path} | voice_id: {voice_id}. "
                f"{tool_hint} "
                f"Transcribe what the user said, then respond to their request in Russian.]"
            )

            await update.message.chat.send_action("typing")
            response = await agent.run(voice_msg, user_id)
            logger.info("Response to %s (voice): %s", user_id, response[:80])
            for i in range(0, len(response), TG_MAX_LENGTH):
                await update.message.reply_text(response[i:i + TG_MAX_LENGTH])

        except Exception as e:
            logger.error("Error handling voice message: %s", e, exc_info=True)
            await update.message.reply_text(f"Ошибка обработки голосового сообщения: {e}")

    return handle_voice


def _make_photo_handler(agent, allowed_chat_ids: set | None = None):
    """Create handler for photo messages.

    Downloads the largest photo from Telegram, converts to base64 image
    content block, and sends to agent.run() as multimodal input for vision.
    """
    from ..multimodal import file_to_content_block

    async def handle_photo(update, context):
        if not update.message or not update.message.photo:
            return

        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            return

        user_id = f"tg-{update.effective_user.id}"
        caption = update.message.caption or ""

        await update.message.chat.send_action("typing")

        try:
            # Get the largest resolution (last in the array)
            photo = update.message.photo[-1]
            file = await photo.get_file()
            photo_bytes = await file.download_as_bytearray()

            logger.info("Photo from %s (chat=%s): %d bytes, caption='%s'",
                        user_id, chat_id, len(photo_bytes), caption[:60])

            block = file_to_content_block(bytes(photo_bytes), "photo.jpg", "image/jpeg")

            # Build content blocks: text prompt first, then image
            content_blocks = []
            if caption:
                content_blocks.append({"type": "text", "text": caption})
            else:
                content_blocks.append({
                    "type": "text",
                    "text": "User sent a photo. Describe what you see and respond."
                })
            content_blocks.append(block)

            await update.message.chat.send_action("typing")
            response = await agent.run(content_blocks, user_id)

            for i in range(0, len(response), TG_MAX_LENGTH):
                await update.message.reply_text(response[i:i + TG_MAX_LENGTH])

        except ValueError as e:
            await update.message.reply_text(str(e))
        except Exception as e:
            logger.error("Error handling photo: %s", e, exc_info=True)
            await update.message.reply_text(f"Error processing photo: {e}")

    return handle_photo


def _make_document_handler(agent, allowed_chat_ids: set | None = None):
    """Create handler for document/file messages.

    Downloads document from Telegram, converts to appropriate content block
    (image, PDF document, or text), and sends to agent.run().
    """
    from ..multimodal import file_to_content_block

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
            doc_bytes = await file.download_as_bytearray()

            logger.info("Document from %s (chat=%s): '%s' (%s, %d bytes)",
                        user_id, chat_id, filename, mime_type, len(doc_bytes))

            block = file_to_content_block(bytes(doc_bytes), filename, mime_type)

            # Build content blocks: text prompt first, then file
            content_blocks = []
            if caption:
                content_blocks.append({"type": "text", "text": caption})
            else:
                content_blocks.append({
                    "type": "text",
                    "text": f"User sent a file: {filename}. Analyze its content and respond."
                })
            content_blocks.append(block)

            await update.message.chat.send_action("typing")
            response = await agent.run(content_blocks, user_id)

            for i in range(0, len(response), TG_MAX_LENGTH):
                await update.message.reply_text(response[i:i + TG_MAX_LENGTH])

        except ValueError as e:
            await update.message.reply_text(str(e))
        except Exception as e:
            logger.error("Error handling document '%s': %s", filename, e, exc_info=True)
            await update.message.reply_text(f"Error processing file: {e}")

    return handle_document


# ── Helpers ─────────────────────────────────────────────────

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


# ── Polling mode ────────────────────────────────────────────

async def run_telegram(agent, config: dict):
    """Run Telegram bot — dispatches to polling or webhook based on config."""
    mode = config.get("mode", "polling")
    if mode == "webhook":
        raise ValueError(
            "Webhook mode requires FastAPI. Use --channel api with "
            "telegram webhook_url configured, or use polling mode."
        )
    await _run_polling(agent, config)


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
        # Clear any old/stale commands first, then set only ours
        await bot.delete_my_commands()
        await bot.set_my_commands(commands)
        logger.info("Bot menu commands set: %s", [c.command for c in commands])
    except Exception as e:
        logger.warning("Failed to set bot commands: %s", e)


async def _run_polling(agent, config: dict):
    """Run Telegram bot with long-polling."""
    try:
        from telegram import Update
        from telegram.ext import Application, CommandHandler, MessageHandler, filters
    except ImportError:
        raise ImportError(
            "python-telegram-bot is required: pip install liteagent[telegram]"
        )

    token = config.get("token")
    if not token:
        raise ValueError("Telegram token not configured. Set token in channels.telegram config.")

    allowed_chat_ids = _parse_chat_ids(config)

    # Text messages (non-command)
    msg_handler = _make_message_handler(agent, allowed_chat_ids)
    app = Application.builder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_handler))

    # Slash commands → agent.handle_command()
    cmd_handler = _make_command_handler(agent, allowed_chat_ids)
    known_cmds = [c for c, _ in BOT_COMMANDS] + ["ingest", "forget", "conflicts", "documents"]
    for cmd_name in known_cmds:
        app.add_handler(CommandHandler(cmd_name, cmd_handler))

    # Voice messages → Whisper transcription
    if config.get("voice_enabled", True):
        voice_handler = _make_voice_handler(agent, allowed_chat_ids, config)
        app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_handler))
        logger.info("Voice message handler registered (Whisper transcription)")

    # Photo messages → vision (multimodal)
    photo_handler = _make_photo_handler(agent, allowed_chat_ids)
    app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    logger.info("Photo handler registered (vision)")

    # Document/file messages → multimodal analysis
    doc_handler = _make_document_handler(agent, allowed_chat_ids)
    app.add_handler(MessageHandler(filters.Document.ALL, doc_handler))
    logger.info("Document handler registered (file analysis)")

    # Ensure we see Telegram logs
    logging.getLogger("liteagent").setLevel(logging.INFO)

    if allowed_chat_ids:
        logger.info("Telegram bot started (polling), allowed chats: %s", allowed_chat_ids)
    else:
        logger.info("Telegram bot started (polling), accepting all chats")
    print("[Jess] Telegram bot running. Send a message to your bot!")

    # Use low-level init/start instead of run_polling() which tries to own the event loop
    await app.initialize()
    await app.updater.start_polling(drop_pending_updates=True)
    await app.start()

    # Set bot menu commands (burger button)
    has_rag = hasattr(agent, '_rag') and agent._rag is not None
    await _set_bot_commands(app.bot, has_rag=has_rag)

    # Keep running until interrupted
    try:
        stop_event = asyncio.Event()
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


# ── Webhook mode ────────────────────────────────────────────

def setup_webhook_route(app, agent, tg_config: dict):
    """Mount Telegram webhook route on a FastAPI app."""
    try:
        from telegram import Update
        from telegram.ext import Application, MessageHandler, filters
    except ImportError:
        raise ImportError(
            "python-telegram-bot is required: pip install liteagent[telegram]"
        )

    token = tg_config.get("token")
    webhook_url = tg_config.get("webhook_url")
    webhook_secret = tg_config.get("webhook_secret", "")

    if not token or not webhook_url:
        raise ValueError("Telegram webhook requires token and webhook_url in config")

    # Build telegram Application for update processing
    tg_app = Application.builder().token(token).build()
    allowed_chat_ids = _parse_chat_ids(tg_config)
    msg_handler = _make_message_handler(agent, allowed_chat_ids)
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_handler))

    # Slash commands
    cmd_handler = _make_command_handler(agent, allowed_chat_ids)
    known_cmds = [c for c, _ in BOT_COMMANDS] + ["ingest", "forget", "conflicts", "documents"]
    for cmd_name in known_cmds:
        try:
            from telegram.ext import CommandHandler
            tg_app.add_handler(CommandHandler(cmd_name, cmd_handler))
        except ImportError:
            break

    # Voice messages → Whisper transcription
    if tg_config.get("voice_enabled", True):
        voice_handler = _make_voice_handler(agent, allowed_chat_ids, tg_config)
        tg_app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, voice_handler))

    # Photo messages → vision (multimodal)
    photo_handler = _make_photo_handler(agent, allowed_chat_ids)
    tg_app.add_handler(MessageHandler(filters.PHOTO, photo_handler))

    # Document/file messages → multimodal analysis
    doc_handler = _make_document_handler(agent, allowed_chat_ids)
    tg_app.add_handler(MessageHandler(filters.Document.ALL, doc_handler))

    @app.on_event("startup")
    async def _setup_webhook():
        await tg_app.initialize()
        await tg_app.start()
        has_rag = hasattr(agent, '_rag') and agent._rag is not None
        await _set_bot_commands(tg_app.bot, has_rag=has_rag)
        await tg_app.bot.set_webhook(
            url=f"{webhook_url}/telegram/webhook",
            secret_token=webhook_secret)
        logger.info("Telegram webhook set to %s/telegram/webhook", webhook_url)

    @app.on_event("shutdown")
    async def _cleanup_webhook():
        await tg_app.stop()
        await tg_app.shutdown()

    @app.post("/telegram/webhook")
    async def telegram_webhook(request):
        """Receive Telegram webhook updates."""
        body = await request.json()
        update = Update.de_json(body, tg_app.bot)
        await tg_app.process_update(update)
        return {"ok": True}

    logger.info("Telegram webhook route mounted")
