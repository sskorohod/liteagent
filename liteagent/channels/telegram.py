"""Telegram channel adapter for LiteAgent — polling + webhook modes.

Supports two backends:
  - HTTP proxy: Telegram handlers send HTTP requests to local API (default when running alongside API)
  - Direct: Telegram handlers call agent.run() directly (standalone --channel telegram fallback)

OpenClaw features integrated:
  - TelegramDraftStream: live message editing while tokens arrive (draft-stream.ts)
  - TypingLifecycle + StatusReactionController: emoji reactions + robust typing
  - Mention gating: only respond when @mentioned in group chats
"""

import asyncio
import contextlib
import logging
import os
import time

logger = logging.getLogger(__name__)

# Max message length in Telegram
TG_MAX_LENGTH = 4096

# Minimum chars before sending first draft message (avoids push notification spam)
_DRAFT_MIN_INITIAL_CHARS = 80
# Throttle interval for draft edits (seconds) — Telegram allows ~1 edit/sec
_DRAFT_THROTTLE_S = 1.1
# Cursor appended while streaming
_DRAFT_CURSOR = " ▍"


# ══════════════════════════════════════════
# TELEGRAM DRAFT STREAM (from OpenClaw draft-stream.ts)
# Live message editing as tokens arrive
# ══════════════════════════════════════════

class TelegramDraftStream:
    """Stream agent response into a live-edited Telegram message.

    Usage:
        draft = TelegramDraftStream(message)
        async for chunk in agent.stream(...):
            draft.push(chunk)
        await draft.finalize()
    """

    def __init__(self, reply_to_message, parse_mode: str | None = None):
        self._reply_msg = reply_to_message
        self._parse_mode = parse_mode
        self._sent_message = None      # the bot's reply message object
        self._text = ""                # accumulated text so far
        self._last_sent_text = ""      # last text actually sent/edited
        self._last_edit_at = 0.0
        self._closed = False
        self._flush_task: asyncio.Task | None = None

    def push(self, chunk: str) -> None:
        """Accumulate a new text chunk."""
        self._text += chunk

    async def tick(self) -> None:
        """Attempt to send/edit the current text (throttled).

        Call this periodically during streaming.
        """
        if self._closed:
            return
        now = time.monotonic()
        if now - self._last_edit_at < _DRAFT_THROTTLE_S:
            return
        await self._flush(cursor=True)

    async def finalize(self) -> str:
        """Send the final text (no cursor). Returns full text."""
        self._closed = True
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        await self._flush(cursor=False)
        return self._text

    async def _flush(self, cursor: bool = False) -> None:
        """Send or edit the message with current text."""
        text = self._text
        if not text.strip():
            return

        display = text + _DRAFT_CURSOR if cursor else text

        # First send
        if self._sent_message is None:
            if len(text) < _DRAFT_MIN_INITIAL_CHARS and cursor:
                return  # Wait until we have enough content
            try:
                kwargs = {}
                if self._parse_mode:
                    kwargs["parse_mode"] = self._parse_mode
                self._sent_message = await self._reply_msg.reply_text(
                    display[:TG_MAX_LENGTH], **kwargs)
                self._last_sent_text = display[:TG_MAX_LENGTH]
                self._last_edit_at = time.monotonic()
            except Exception as exc:
                logger.debug("Draft initial send failed: %s", exc)
            return

        # Don't edit if text hasn't changed meaningfully
        new_display = display[:TG_MAX_LENGTH]
        if new_display == self._last_sent_text:
            return

        # If text exceeded limit, send a new message
        if len(text) > TG_MAX_LENGTH:
            remainder = text[TG_MAX_LENGTH:]
            if not cursor:
                try:
                    kwargs = {}
                    if self._parse_mode:
                        kwargs["parse_mode"] = self._parse_mode
                    # First finalize the old message without cursor
                    try:
                        await self._sent_message.edit_text(
                            text[:TG_MAX_LENGTH], **kwargs)
                    except Exception:
                        pass
                    # Send continuation as new message
                    self._sent_message = await self._reply_msg.reply_text(
                        remainder[:TG_MAX_LENGTH], **kwargs)
                    self._last_sent_text = remainder[:TG_MAX_LENGTH]
                    self._last_edit_at = time.monotonic()
                except Exception as exc:
                    logger.debug("Draft continuation failed: %s", exc)
            return

        try:
            kwargs = {}
            if self._parse_mode:
                kwargs["parse_mode"] = self._parse_mode
            await self._sent_message.edit_text(new_display, **kwargs)
            self._last_sent_text = new_display
            self._last_edit_at = time.monotonic()
        except Exception as exc:
            # "Message is not modified" — ignore
            if "not modified" not in str(exc).lower():
                logger.debug("Draft edit failed: %s", exc)


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
                timeout=httpx.Timeout(300.0, connect=10.0),
            )
        return self._client

    async def chat(self, message: str, user_id: str, chat_id: int | None = None) -> dict:
        """POST /chat → {response, files}"""
        client = await self._get_client()
        payload = {"message": message, "user_id": user_id}
        if chat_id is not None:
            payload["chat_id"] = str(chat_id)
        resp = await client.post("/chat", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def chat_multimodal(self, message: str, user_id: str,
                               chat_id: int | None,
                               files: list[tuple[str, bytes, str]]) -> dict:
        """POST /chat/multimodal with file uploads.
        files: list of (filename, data_bytes, content_type)
        """
        client = await self._get_client()
        multipart_files = [("files", (fname, data, ct)) for fname, data, ct in files]
        resp = await client.post(
            "/chat/multimodal",
            data={"message": message, "user_id": user_id, "chat_id": str(chat_id or "")},
            files=multipart_files,
        )
        resp.raise_for_status()
        return resp.json()

    async def chat_voice(self, audio_bytes: bytes, user_id: str,
                         chat_id: int | None = None, duration: str = "0") -> dict:
        """POST /chat/voice with audio file."""
        client = await self._get_client()
        resp = await client.post(
            "/chat/voice",
            data={"user_id": user_id, "chat_id": str(chat_id or ""), "duration": duration},
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

    async def stream_chat(self, message: str, user_id: str,
                          chat_id: int | None = None) -> "AsyncGenerator[str, None]":
        """Stream chat via GET /chat/stream SSE endpoint."""
        import json as _json

        async def _gen():
            client = await self._get_client()
            params = {"message": message, "user_id": user_id}
            if chat_id is not None:
                params["chat_id"] = str(chat_id)
            async with client.stream("GET", "/chat/stream", params=params,
                                     timeout=300.0) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if not payload:
                        continue
                    try:
                        data = _json.loads(payload)
                    except Exception:
                        continue
                    if "text" in data:
                        yield data["text"]
                    elif data.get("done"):
                        break

        return _gen()

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

    def _remember_telegram_chat(self, user_id: str, chat_id: int | None = None) -> None:
        uid = str(user_id or "").strip()
        cid = str(chat_id or "").strip()
        if not uid.startswith("tg-") or not cid:
            return
        try:
            self.agent.memory.set_state("user:telegram_chat_id", cid, user_id=uid)
        except Exception:
            pass

    async def chat(self, message: str, user_id: str, chat_id: int | None = None) -> dict:
        self._remember_telegram_chat(user_id, chat_id)
        token = None
        try:
            token = self.agent._set_current_chat_id(chat_id)
            response = await self.agent.run(message, user_id)
        finally:
            if token is not None:
                self.agent._reset_current_chat_id(token)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def chat_multimodal(self, message: str, user_id: str,
                               chat_id: int | None,
                               files: list[tuple[str, bytes, str]]) -> dict:
        self._remember_telegram_chat(user_id, chat_id)
        from ..multimodal import file_to_content_block
        content_blocks = [{"type": "text", "text": message}]
        for fname, data, ct in files:
            block = file_to_content_block(data, fname, ct)
            content_blocks.append(block)
            # Auto-ingest uploaded files to S3
            await self.agent.ingest_file(
                data, fname, source="telegram", user_id=user_id, mime_type=ct)
        token = None
        try:
            token = self.agent._set_current_chat_id(chat_id)
            response = await self.agent.run(content_blocks, user_id)
        finally:
            if token is not None:
                self.agent._reset_current_chat_id(token)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def chat_voice(self, audio_bytes: bytes, user_id: str,
                         chat_id: int | None = None, duration: str = "0") -> dict:
        self._remember_telegram_chat(user_id, chat_id)
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

        # Pre-transcribe: try to transcribe audio BEFORE sending to model
        transcription = None
        try:
            from ..voice import transcribe as voice_transcribe
            stt_result = await voice_transcribe(audio_bytes, self.agent.config)
            if stt_result.success and stt_result.text:
                transcription = stt_result.text.strip()
                logger.info("Pre-transcribed voice (%s): %s",
                            voice_id, transcription[:100])
        except Exception as e:
            logger.warning("Pre-transcription failed: %s", e)

        if transcription:
            voice_msg = transcription
        else:
            # Fallback: ask model to call transcribe tool
            tool_hint = "Use any available transcription tool."
            if hasattr(self.agent, 'tools'):
                tool_names = list(self.agent.tools._tools.keys())
                mcp_voice = [n for n in tool_names if "transcribe" in n and "__" in n]
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

        token = None
        try:
            token = self.agent._set_current_chat_id(chat_id)
            response = await self.agent.run(voice_msg, user_id)
        finally:
            if token is not None:
                self.agent._reset_current_chat_id(token)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def stream_chat(self, message: str, user_id: str,
                          chat_id: int | None = None):
        """Stream via agent.stream() directly — returns AsyncGenerator[str]."""
        self._remember_telegram_chat(user_id, chat_id)
        token = None
        try:
            token = self.agent._set_current_chat_id(chat_id)
            async for chunk in self.agent.stream(message, user_id):
                yield chunk
        finally:
            if token is not None:
                self.agent._reset_current_chat_id(token)

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
# HARDCODED COMMAND HANDLER (instant, agent-independent)
# ══════════════════════════════════════════

class TelegramCommandHandler:
    """Handles slash commands directly, bypassing the agent pipeline.

    Accesses memory/config/rag by reference — no HTTP round-trip, no agent lock.
    """

    def __init__(self, memory, config: dict, rag=None, file_manager=None):
        self.memory = memory
        self.config = config
        self._rag = rag
        self._file_manager = file_manager

    def handle(self, command: str, user_id: str) -> str | None:
        """Handle command instantly.  Returns response text or None."""
        cmd = command.strip().lower()

        if cmd == "/help":
            return self._help()
        if cmd == "/clear":
            self.memory.clear_conversation(user_id)
            return "Диалог очищен."
        if cmd == "/memories":
            return self._memories(user_id)
        if cmd == "/usage":
            return self._usage()
        if cmd.startswith("/forget "):
            fragment = command[8:].strip()
            self.memory.forget(user_id, fragment)
            return f"Забыты воспоминания по: {fragment}"
        if cmd == "/conflicts":
            return self._conflicts(user_id)
        if cmd == "/documents":
            return self._documents()
        if cmd == "/files":
            return self._files(user_id)
        if cmd == "/model":
            return self._model_status()
        if cmd in ("/rate", "/like", "👍"):
            if hasattr(self.memory, "rate_last_response"):
                self.memory.rate_last_response(user_id, 5)
            return "👍 Спасибо за оценку!"
        if cmd in ("/dislike", "👎"):
            if hasattr(self.memory, "rate_last_response"):
                self.memory.rate_last_response(user_id, 1)
            return "👎 Понял, постараюсь лучше."
        if cmd.startswith("/rate "):
            try:
                val = int(command.split(None, 1)[1].strip())
                if 1 <= val <= 5:
                    if hasattr(self.memory, "rate_last_response"):
                        self.memory.rate_last_response(user_id, val)
                    return f"{'⭐' * val} Оценка {val}/5 записана, спасибо!"
            except (ValueError, IndexError):
                pass
            return "Используйте: /rate 1..5"
        # /model <args> and /ingest — handled elsewhere (need agent)
        return None

    # ── /help ──

    def _help(self) -> str:
        lines = [
            "Команды:",
            "  /model      — Показать/сменить модель",
            "  /memories   — Показать воспоминания",
            "  /usage      — Расход токенов и бюджет",
            "  /clear      — Очистить историю диалога",
            "  /forget X   — Забыть воспоминания по X",
            "  /conflicts  — Решённые конфликты памяти",
            "  /files      — Список сохранённых файлов",
            "  /rate 1-5   — Оценить последний ответ (👍 = /rate 5)",
        ]
        if self._rag:
            lines += [
                "  /ingest X   — Загрузить файл/папку в RAG",
                "  /documents  — Список документов RAG",
            ]
        lines.append("  /help       — Эта справка")
        return "\n".join(lines)

    # ── /memories ──

    def _memories(self, user_id: str) -> str:
        memories = self.memory.get_all_memories(user_id)
        if not memories:
            return "Воспоминаний пока нет."
        lines = [f"{len(memories)} воспоминаний:\n"]
        for m in memories[:20]:
            lines.append(
                f"  [{m['type']}] {m['content'][:80]}  (imp: {m['importance']:.1f})")
        return "\n".join(lines)

    # ── /usage ──

    def _usage(self) -> str:
        summary = self.memory.get_usage_summary(days=7)
        if not summary:
            return "Данных об использовании пока нет."
        today_cost = self.memory.get_today_cost()
        budget = self.config.get("cost", {}).get("budget_daily_usd", 5.0)
        lines = [f"Сегодня: ${today_cost:.4f} / ${budget:.2f}\n",
                 "За 7 дней:"]
        for s in summary:
            lines.append(
                f"  {s['model']}: {s['calls']} вызовов, "
                f"{s['input_tokens']+s['output_tokens']:,} токенов, "
                f"${s['cost_usd']:.4f}")
        return "\n".join(lines)

    # ── /conflicts ──

    def _conflicts(self, user_id: str) -> str:
        archived = self.memory.get_archived_memories(user_id, limit=20)
        if not archived:
            return "Конфликтов памяти пока не было."
        lines = [f"{len(archived)} архивных воспоминаний (решённые конфликты):\n"]
        for m in archived:
            lines.append(
                f"  [{m['type']}] {m['content'][:80]}  (архив: {m['archived_at'][:10]})")
        return "\n".join(lines)

    # ── /documents ──

    def _documents(self) -> str:
        if not self._rag:
            return "RAG не включён. Установите rag.enabled: true в конфиге."
        docs = self._rag.list_documents()
        if not docs:
            return "Документов пока нет. Используйте /ingest <путь> для загрузки."
        lines = [f"{len(docs)} документов:\n"]
        for d in docs:
            lines.append(f"  [{d['id']}] {d['name']} — {d['chunks']} частей")
        return "\n".join(lines)

    # ── /files ──

    def _files(self, user_id: str) -> str:
        fm = self._file_manager
        if not fm:
            return "Хранилище файлов не настроено."
        files = fm.list_files(user_id=user_id, limit=20)
        if not files:
            return "У вас пока нет сохранённых файлов."
        lines = [f"{len(files)} файлов:\n"]
        for f in files:
            size_kb = f['size_bytes'] // 1024
            date = f['created_at'][:10] if f.get('created_at') else ''
            lines.append(
                f"  {f['original_name']} ({size_kb}KB, {f['source']}, {date})")
        return "\n".join(lines)

    # ── /model (read-only, no args) ──

    def _model_status(self) -> str:
        from ..providers import PROVIDER_MODELS

        agent_cfg = self.config.get("agent", {})
        provider_name = agent_cfg.get("provider", "anthropic")
        default_model = agent_cfg.get("default_model", "?")
        models = agent_cfg.get("models", {})
        cascade = self.config.get("cost", {}).get("cascade_routing", False)

        lines = [
            f"Провайдер: {provider_name}",
            f"Модель: {default_model}",
        ]
        if models:
            lines.append(
                "\nКаскадная маршрутизация"
                + (" (вкл)" if cascade else " (выкл)") + ":")
            for tier in ("simple", "medium", "complex"):
                m = models.get(tier)
                if m:
                    lines.append(f"  {tier}: {m}")

        available = PROVIDER_MODELS.get(provider_name, [])
        if available:
            lines.append(f"\nДоступные ({provider_name}):")
            for m in available[:15]:
                marker = " <" if m == default_model else ""
                lines.append(f"  {m}{marker}")

        lines.append(
            "\nИспользование:\n"
            "  /model <имя> — сменить модель\n"
            "  /model simple|medium|complex <имя> — сменить тир")
        return "\n".join(lines)


# ══════════════════════════════════════════
# TYPING INDICATOR + STATUS REACTIONS + ERROR HELPERS
# (OpenClaw-inspired: TypingLifecycle + StatusReactions)
# ══════════════════════════════════════════

def _build_reaction_adapter(message):
    """Build a StatusReactionAdapter that sets Telegram emoji reactions."""
    from ..status_reactions import StatusReactionAdapter

    class _TelegramReactionAdapter(StatusReactionAdapter):
        async def set_reaction(self, emoji: str) -> None:
            try:
                from telegram import ReactionTypeEmoji
                await message.set_reaction([ReactionTypeEmoji(emoji=emoji)])
            except Exception as exc:
                logger.debug("set_reaction(%s) failed: %s", emoji, exc)

        async def remove_reaction(self, emoji: str) -> None:
            try:
                await message.set_reaction([])
            except Exception as exc:
                logger.debug("remove_reaction failed: %s", exc)

    return _TelegramReactionAdapter()


@contextlib.asynccontextmanager
async def _typing_loop(chat, reply_func=None, interval=4.0, status_after=15.0,
                       message=None, use_reactions: bool = True):
    """Send 'typing' action with TypingLifecycle keepalive (from OpenClaw).

    Optionally shows emoji status reactions on the triggering message.
    Falls back to text status indicator if needed.
    """
    from ..status_reactions import TypingLifecycle, StatusReactionController, NullStatusReactionAdapter

    # Build status reaction controller if we have a message object
    reaction_ctrl = None
    if use_reactions and message is not None:
        try:
            adapter = _build_reaction_adapter(message)
            from ..status_reactions import StatusReactionController
            reaction_ctrl = StatusReactionController(adapter=adapter)
        except Exception:
            reaction_ctrl = None

    async def _start_typing():
        try:
            await chat.send_action("typing")
        except Exception:
            pass

    lifecycle = TypingLifecycle(
        start_fn=_start_typing,
        keepalive_s=interval,
        max_duration_s=120.0,
    )

    if reaction_ctrl:
        await reaction_ctrl.set_queued()

    await lifecycle.start()

    if reaction_ctrl:
        await reaction_ctrl.set_thinking()

    # Long-operation status hint (fallback text)
    status_task = None
    if reply_func and status_after > 0:
        async def _status_hint():
            await asyncio.sleep(status_after)
            try:
                await reply_func("Работаю над запросом...")
            except Exception:
                pass
        status_task = asyncio.create_task(_status_hint())

    try:
        yield reaction_ctrl  # expose controller to caller (optional)
    finally:
        lifecycle.stop()
        if status_task and not status_task.done():
            status_task.cancel()
        if reaction_ctrl:
            reaction_ctrl.close()


# ── Mention gating (from OpenClaw mention-gating.ts) ─────────────────────────

def _is_group_chat(update) -> bool:
    """Return True if the update is from a group or supergroup chat."""
    chat_type = getattr(update.effective_chat, 'type', 'private')
    return chat_type in ('group', 'supergroup', 'channel')


def _was_bot_mentioned(update, bot_username: str | None) -> bool:
    """Return True if the bot was @mentioned in the message."""
    if not update.message:
        return False
    # Check entities for mentions
    entities = update.message.entities or []
    text = update.message.text or ""
    for ent in entities:
        if ent.type == "mention":
            mention_text = text[ent.offset:ent.offset + ent.length]
            if bot_username and mention_text.lower() == f"@{bot_username.lower()}":
                return True
        elif ent.type == "bot_command":
            # Commands always bypass mention gate
            return True
    return False


def _check_mention_gate(update, bot_username: str | None,
                         require_mention_in_groups: bool = True) -> bool:
    """Return True if the message should be processed.

    In group chats with require_mention=True, skip messages that don't @mention the bot.
    """
    if not _is_group_chat(update):
        return True  # Always respond in DMs
    if not require_mention_in_groups:
        return True  # Configured to respond to all group messages
    return _was_bot_mentioned(update, bot_username)


def _user_friendly_error(exc: Exception) -> str:
    """Convert exception to a user-friendly message in Russian."""
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        return "Запрос занял слишком много времени. Попробуйте ещё раз."
    if "rate" in msg and "limit" in msg:
        return "Слишком много запросов. Подождите немного."
    if "budget" in msg:
        return "Дневной бюджет исчерпан. Сбросится завтра."
    if "connect" in msg or "connection" in msg:
        return "Ошибка соединения. Попробуйте позже."
    return "Произошла ошибка. Попробуйте ещё раз."


# ══════════════════════════════════════════
# MESSAGE HANDLERS (unified — work with both TelegramAPIClient and _DirectAPIAdapter)
# ══════════════════════════════════════════

def _make_message_handler(api_client, allowed_chat_ids: set | None = None,
                          tg_config: dict | None = None):
    """Create handler for text messages.

    Includes mention gating (OpenClaw): in group chats, only responds when @mentioned,
    unless require_mention_in_groups=false in config.
    Includes status reactions (OpenClaw): emoji status on the triggering message.
    """
    cfg = tg_config or {}
    require_mention = cfg.get("require_mention_in_groups", True)

    async def handle_message(update, context):
        if not update.message or not update.message.text:
            return
        chat_id = update.effective_chat.id
        if allowed_chat_ids and chat_id not in allowed_chat_ids:
            logger.info("Ignoring message from chat_id=%s (not in allowed list)", chat_id)
            return

        # Mention gating (from OpenClaw) — skip non-mentions in group chats
        bot_username = context.bot.username if context.bot else None
        if not _check_mention_gate(update, bot_username, require_mention):
            logger.debug("Mention gate: skipping group message (no @mention)")
            return

        user_id = f"tg-{update.effective_user.id}"
        text = update.message.text

        # Strip bot mention from text in group chats
        if bot_username and _is_group_chat(update):
            text = text.replace(f"@{bot_username}", "").strip()
            if not text:
                return

        logger.info("Message from %s (chat=%s): %s", user_id, chat_id, text[:80])

        # Status reactions setup (from OpenClaw)
        reaction_ctrl = None
        try:
            adapter = _build_reaction_adapter(update.message)
            from ..status_reactions import StatusReactionController
            reaction_ctrl = StatusReactionController(adapter=adapter)
            await reaction_ctrl.set_queued()
        except Exception:
            reaction_ctrl = None

        # Start typing lifecycle
        from ..status_reactions import TypingLifecycle

        async def _start_typing():
            try:
                await update.message.chat.send_action("typing")
            except Exception:
                pass

        typing = TypingLifecycle(start_fn=_start_typing, keepalive_s=4.0, max_duration_s=180.0)
        await typing.start()

        try:
            if reaction_ctrl:
                await reaction_ctrl.set_thinking()

            # Use draft streaming if available (from OpenClaw draft-stream.ts)
            # Check with isasyncgenfunction to avoid false-positives on MagicMock
            import inspect as _inspect
            _stream_fn = getattr(api_client, 'stream_chat', None)
            if _stream_fn is not None and _inspect.isasyncgenfunction(_stream_fn):
                draft = TelegramDraftStream(update.message)
                full_text = ""
                try:
                    async for chunk in api_client.stream_chat(text, user_id, chat_id=chat_id):
                        full_text += chunk
                        draft.push(chunk)
                        await draft.tick()
                    await draft.finalize()
                except Exception as stream_err:
                    logger.warning("Stream failed, falling back to non-stream: %s", stream_err)
                    if not full_text:
                        # Fallback to non-streaming
                        result = await api_client.chat(text, user_id, chat_id=chat_id)
                        await _reply_with_result(update.message, result)
                    # If we got partial text, it's already displayed
            else:
                result = await api_client.chat(text, user_id, chat_id=chat_id)
                await _reply_with_result(update.message, result)

            # Done reaction
            if reaction_ctrl:
                await reaction_ctrl.set_done()
                await asyncio.sleep(1.5)
                await reaction_ctrl.clear()

        except Exception as e:
            logger.error("Error handling Telegram message: %s", e, exc_info=True)
            if reaction_ctrl:
                try:
                    await reaction_ctrl.set_error()
                    await asyncio.sleep(2.5)
                    await reaction_ctrl.clear()
                except Exception:
                    pass
            await update.message.reply_text(_user_friendly_error(e))
        finally:
            typing.stop()

    return handle_message


def _make_command_handler(cmd_handler: TelegramCommandHandler, api_client,
                          allowed_chat_ids: set | None = None):
    """Create handler for /slash commands.

    Instant commands are handled by `cmd_handler` directly (no agent dependency).
    Slow commands (/model <args>, /ingest) fall through to `api_client`.
    """
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
            # Try instant handling first (no agent dependency)
            response = cmd_handler.handle(text, user_id)
            if response is not None:
                for i in range(0, len(response), TG_MAX_LENGTH):
                    await update.message.reply_text(response[i:i + TG_MAX_LENGTH])
                return

            # /ingest — slow, with progress indicator
            cmd_lower = text.strip().lower()
            if cmd_lower.startswith("/ingest"):
                path = text[7:].strip()
                if not path:
                    await update.message.reply_text(
                        "Использование: /ingest <путь к файлу или папке>")
                    return
                await _handle_ingest(update, cmd_handler, path)
                return

            # /model <args> — delegates to agent (mutates agent state)
            if cmd_lower.startswith("/model "):
                async with _typing_loop(update.message.chat):
                    result = await api_client.command(text, user_id)
                agent_response = result.get("response")
                if agent_response:
                    for i in range(0, len(agent_response), TG_MAX_LENGTH):
                        await update.message.reply_text(
                            agent_response[i:i + TG_MAX_LENGTH])
                return

            await update.message.reply_text(
                "Неизвестная команда. Используйте /help для списка команд.")
        except Exception as e:
            logger.error("Error handling command: %s", e, exc_info=True)
            await update.message.reply_text(_user_friendly_error(e))

    return handle_command


async def _handle_ingest(update, cmd_handler: TelegramCommandHandler, path: str):
    """Handle /ingest with progress indicator (inherently slow)."""
    if not cmd_handler._rag:
        await update.message.reply_text(
            "RAG не включён. Установите rag.enabled: true в конфиге.")
        return

    await update.message.reply_text(f"Загружаю '{path}'...")

    try:
        async with _typing_loop(update.message.chat):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, cmd_handler._rag.ingest, path)

        if "files" in result:
            errors = f"\nОшибки: {result['errors']}" if result.get("errors") else ""
            response = (f"Загружено {result['files']} файлов, "
                        f"{result['chunks']} частей.{errors}")
        else:
            response = f"{result['path']}: {result['status']} ({result['chunks']} частей)"
        await update.message.reply_text(response)
    except Exception as e:
        logger.error("Ingest error: %s", e, exc_info=True)
        await update.message.reply_text(f"Ошибка загрузки: {_user_friendly_error(e)}")


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

        try:
            async with _typing_loop(update.message.chat, update.message.reply_text):
                file = await voice.get_file()
                audio_bytes = bytes(await file.download_as_bytearray())
                logger.info("Downloaded voice file: %d bytes", len(audio_bytes))

                result = await api_client.chat_voice(
                    audio_bytes,
                    user_id,
                    chat_id=chat_id,
                    duration=duration,
                )

            await _reply_with_result(update.message, result)

        except Exception as e:
            logger.error("Error handling voice message: %s", e, exc_info=True)
            await update.message.reply_text(_user_friendly_error(e))

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

        try:
            async with _typing_loop(update.message.chat, update.message.reply_text):
                photo = update.message.photo[-1]
                file = await photo.get_file()
                photo_bytes = bytes(await file.download_as_bytearray())

                logger.info("Photo from %s (chat=%s): %d bytes, caption='%s'",
                            user_id, chat_id, len(photo_bytes), caption[:60])

                ingest_note = (
                    f"[System: photo ({len(photo_bytes)} bytes) has been "
                    f"auto-saved to storage. Confirm storage to the user.]")
                caption_with_note = caption + "\n\n" + ingest_note

                result = await api_client.chat_multimodal(
                    message=caption_with_note,
                    user_id=user_id,
                    chat_id=chat_id,
                    files=[("photo.jpg", photo_bytes, "image/jpeg")],
                )
            await _reply_with_result(update.message, result)

        except Exception as e:
            logger.error("Error handling photo: %s", e, exc_info=True)
            await update.message.reply_text(_user_friendly_error(e))

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
                f"Файл '{filename}' слишком большой ({size_mb:.1f} МБ). Максимум: 10 МБ.")
            return

        try:
            async with _typing_loop(update.message.chat, update.message.reply_text):
                file = await doc.get_file()
                doc_bytes = bytes(await file.download_as_bytearray())

                logger.info("Document from %s (chat=%s): '%s' (%s, %d bytes)",
                            user_id, chat_id, filename, mime_type, len(doc_bytes))

                ingest_note = (
                    f"[System: file '{filename}' ({mime_type}, "
                    f"{len(doc_bytes)} bytes) has been auto-saved to storage "
                    f"and indexed for future search. Confirm storage to the user.]")
                message = (caption + "\n\n" + ingest_note) if caption else ingest_note

                result = await api_client.chat_multimodal(
                    message=message,
                    user_id=user_id,
                    chat_id=chat_id,
                    files=[(filename, doc_bytes, mime_type)],
                )
            await _reply_with_result(update.message, result)

        except Exception as e:
            logger.error("Error handling document '%s': %s", filename, e, exc_info=True)
            await update.message.reply_text(_user_friendly_error(e))

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
    ("forget", "Забыть воспоминания по тексту"),
    ("conflicts", "Решённые конфликты памяти"),
    ("files", "Список сохранённых файлов"),
    ("documents", "Список документов RAG"),
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


def _register_all_handlers(tg_app, api_client, cmd_handler: TelegramCommandHandler,
                           allowed_chat_ids, config):
    """Register all message handlers on a telegram Application."""
    try:
        from telegram.ext import CommandHandler, MessageHandler, filters
    except ImportError:
        raise ImportError("python-telegram-bot is required: pip install liteagent[telegram]")

    # Text messages (with mention gating + status reactions from OpenClaw)
    msg_handler = _make_message_handler(api_client, allowed_chat_ids, tg_config=config)
    tg_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, msg_handler))

    # Slash commands (instant via cmd_handler, slow via api_client fallback)
    slash_handler = _make_command_handler(cmd_handler, api_client, allowed_chat_ids)
    known_cmds = [c for c, _ in BOT_COMMANDS] + ["ingest"]
    for cmd_name in known_cmds:
        tg_app.add_handler(CommandHandler(cmd_name, slash_handler))

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
    cmd_handler = TelegramCommandHandler(
        memory=agent.memory, config=agent.config,
        rag=getattr(agent, '_rag', None),
        file_manager=getattr(agent, '_file_manager', None))

    app = Application.builder().token(token).build()
    _register_all_handlers(app, api_client, cmd_handler, allowed_chat_ids, config)

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
    cmd_handler = TelegramCommandHandler(
        memory=agent.memory, config=agent.config,
        rag=getattr(agent, '_rag', None),
        file_manager=getattr(agent, '_file_manager', None))

    _register_all_handlers(tg_app, api_client, cmd_handler, allowed_chat_ids, tg_config)

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
