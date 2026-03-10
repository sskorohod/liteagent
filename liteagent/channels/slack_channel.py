"""Slack channel adapter for LiteAgent.

Inspired by OpenClaw's Slack channel architecture.

Features (ported from OpenClaw):
- Status reactions: emoji reactions on messages showing agent state
- Mention gating: only respond when @mentioned in channels
- Typing indicator: Slack typing via chat.postMessage updates
- Thread support: responds in the same thread as the triggering message
- Block kit fallback: plain text fallback for message limits

Setup:
  1. Create a Slack app at https://api.slack.com/apps
  2. Enable "Socket Mode" for easy setup (or use Slack Events API with a webhook URL)
  3. Add OAuth scopes:
     - Bot token: channels:history, channels:read, chat:write, reactions:write,
                  groups:history, im:history, mpim:history, users:read
  4. Add your tokens to config.json:

     "channels": {
       "slack": {
         "bot_token_env": "SLACK_BOT_TOKEN",
         "app_token_env": "SLACK_APP_TOKEN",   // for socket mode
         "require_mention_in_channels": true,
         "allowed_channel_ids": [],             // empty = all channels
         "allowed_user_ids": []                 // empty = all users
       }
     }

  5. Run: python -m liteagent --channel slack

Required packages:
  pip install slack_sdk "slack-bolt>=1.18"  (or: pip install liteagent[slack])
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

SLACK_MAX_LENGTH = 40000  # Slack's effective text limit


# ══════════════════════════════════════════
# STATUS REACTIONS (emoji) FOR SLACK
# ══════════════════════════════════════════

def _build_slack_reaction_adapter(client, channel: str, timestamp: str):
    """Build StatusReactionAdapter for a Slack message."""
    from ..status_reactions import StatusReactionAdapter

    class _SlackReactionAdapter(StatusReactionAdapter):
        def __init__(self):
            self._current: str | None = None

        def _emoji_name(self, emoji: str) -> str:
            """Map unicode emoji to Slack :name: format."""
            mapping = {
                "👀": "eyes", "🤔": "thinking_face", "🔥": "fire",
                "👨‍💻": "man-technologist", "⚡": "zap",
                "✅": "white_check_mark", "😱": "scream",
                "🥱": "yawning_face", "😨": "fearful",
            }
            return mapping.get(emoji, "white_circle")

        async def set_reaction(self, emoji: str) -> None:
            try:
                if self._current:
                    await self.remove_reaction(self._current)
                await asyncio.to_thread(
                    client.reactions_add,
                    channel=channel,
                    timestamp=timestamp,
                    name=self._emoji_name(emoji),
                )
                self._current = emoji
            except Exception as exc:
                logger.debug("Slack set_reaction(%s) failed: %s", emoji, exc)

        async def remove_reaction(self, emoji: str) -> None:
            try:
                await asyncio.to_thread(
                    client.reactions_remove,
                    channel=channel,
                    timestamp=timestamp,
                    name=self._emoji_name(emoji),
                )
                self._current = None
            except Exception as exc:
                logger.debug("Slack remove_reaction(%s) failed: %s", emoji, exc)

    return _SlackReactionAdapter()


# ══════════════════════════════════════════
# MENTION GATING (from OpenClaw)
# ══════════════════════════════════════════

def _is_dm_channel(channel_type: str | None) -> bool:
    return channel_type in ("im", "mpim")


def _was_bot_mentioned(text: str, bot_user_id: str) -> bool:
    return f"<@{bot_user_id}>" in (text or "")


def _check_mention_gate(event: dict, bot_user_id: str,
                         require_mention: bool) -> bool:
    channel_type = event.get("channel_type", "")
    if _is_dm_channel(channel_type):
        return True  # Always respond in DMs
    if not require_mention:
        return True
    text = event.get("text", "")
    return _was_bot_mentioned(text, bot_user_id)


# ══════════════════════════════════════════
# ACCESS CONTROL
# ══════════════════════════════════════════

def _is_allowed(event: dict, cfg: dict) -> bool:
    allowed_channels = set(str(c) for c in cfg.get("allowed_channel_ids", []))
    allowed_users = set(str(u) for u in cfg.get("allowed_user_ids", []))

    if allowed_channels and event.get("channel"):
        if event["channel"] not in allowed_channels:
            return False
    if allowed_users and event.get("user"):
        if event["user"] not in allowed_users:
            return False
    return True


# ══════════════════════════════════════════
# SLACK BOT
# ══════════════════════════════════════════

class LiteAgentSlackBot:
    """Slack bot using Slack Bolt (Socket Mode for easy setup)."""

    def __init__(self, api_client, config: dict):
        self._api_client = api_client
        self._cfg = config
        self._require_mention = config.get("require_mention_in_channels", True)

    def _post_reply(self, say, client, channel: str, thread_ts: str | None,
                     text: str) -> None:
        """Post a reply, threading to the original message if applicable."""
        kwargs = {"text": text[:SLACK_MAX_LENGTH], "channel": channel}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        say(**kwargs)

    def build_app(self, bot_token: str, app_token: str):
        """Build a Slack Bolt app with Socket Mode."""
        from slack_bolt import App
        from slack_bolt.adapter.socket_mode import SocketModeHandler

        app = App(token=bot_token)

        # Get bot user ID for mention detection
        bot_user_id = self._get_bot_user_id(bot_token)
        logger.info("Slack bot user ID: %s", bot_user_id)

        @app.event("message")
        def handle_message(event, say, client):
            asyncio.run(self._handle_message(event, say, client, bot_user_id))

        @app.event("app_mention")
        def handle_mention(event, say, client):
            asyncio.run(self._handle_message(event, say, client, bot_user_id,
                                              was_mentioned=True))

        handler = SocketModeHandler(app, app_token)
        return app, handler

    def _get_bot_user_id(self, token: str) -> str:
        try:
            from slack_sdk import WebClient
            wc = WebClient(token=token)
            resp = wc.auth_test()
            return resp["user_id"]
        except Exception as exc:
            logger.warning("Failed to get bot user ID: %s", exc)
            return ""

    async def _handle_message(self, event: dict, say, client,
                               bot_user_id: str, was_mentioned: bool = False):
        """Process an incoming Slack message event."""
        # Ignore bot messages
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        # Access control
        if not _is_allowed(event, self._cfg):
            return

        # Mention gating
        if not was_mentioned and not _check_mention_gate(
                event, bot_user_id, self._require_mention):
            return

        text = event.get("text", "")
        channel = event.get("channel", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        user = event.get("user", "")

        # Strip bot mention from text
        if bot_user_id:
            text = text.replace(f"<@{bot_user_id}>", "").strip()
        if not text and not event.get("files"):
            return

        user_id = f"slack-{user}"
        logger.info("Slack message from %s: %s", user_id, text[:80])

        # Status reactions (from OpenClaw)
        reaction_ctrl = None
        try:
            from slack_sdk import WebClient
            wc = WebClient(token=client.token)
            adapter = _build_slack_reaction_adapter(wc, channel, event.get("ts", ""))
            from ..status_reactions import StatusReactionController
            reaction_ctrl = StatusReactionController(adapter=adapter)
            await reaction_ctrl.set_queued()
        except Exception:
            reaction_ctrl = None

        try:
            if reaction_ctrl:
                await reaction_ctrl.set_thinking()

            # Handle file attachments
            if event.get("files"):
                result = await self._handle_with_files(event, text, user_id)
            else:
                result = await self._api_client.chat(
                    text, user_id, chat_id=channel)

            if reaction_ctrl:
                await reaction_ctrl.set_done()
                await asyncio.sleep(1.5)
                await reaction_ctrl.clear()

            # Send response in thread
            response = result.get("response", "")
            if response:
                # Split into chunks respecting Slack limits
                chunks = [response[i:i+SLACK_MAX_LENGTH]
                          for i in range(0, len(response), SLACK_MAX_LENGTH)]
                for chunk in chunks:
                    self._post_reply(say, client, channel, thread_ts, chunk)

        except Exception as exc:
            logger.error("Slack message error: %s", exc, exc_info=True)
            if reaction_ctrl:
                try:
                    await reaction_ctrl.set_error()
                    await asyncio.sleep(2.5)
                    await reaction_ctrl.clear()
                except Exception:
                    pass
            self._post_reply(say, client, channel, thread_ts,
                              _user_friendly_error(exc))

    async def _handle_with_files(self, event: dict, text: str, user_id: str) -> dict:
        """Handle message with Slack file attachments."""
        files_data = []
        for f in event.get("files", []):
            try:
                url = f.get("url_private_download") or f.get("url_private", "")
                if not url:
                    continue
                import httpx
                async with httpx.AsyncClient() as hc:
                    resp = await hc.get(url, timeout=30.0)
                    data = resp.content
                files_data.append((
                    f.get("name", "file"),
                    data,
                    f.get("mimetype", "application/octet-stream"),
                ))
            except Exception as exc:
                logger.warning("Failed to download Slack file: %s", exc)

        if files_data:
            return await self._api_client.chat_multimodal(
                message=text or "User sent files.",
                user_id=user_id,
                chat_id=None,
                files=files_data,
            )
        return await self._api_client.chat(text, user_id)

    async def close(self):
        pass


def _user_friendly_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "timeout" in msg:
        return "Request timed out. Please try again."
    if "rate" in msg and "limit" in msg:
        return "Rate limit hit. Please wait a moment."
    return "An error occurred. Please try again."


# ══════════════════════════════════════════
# DIRECT ADAPTER
# ══════════════════════════════════════════

class _SlackDirectAdapter:
    def __init__(self, agent):
        self.agent = agent

    async def chat(self, message: str, user_id: str, chat_id=None) -> dict:
        response = await self.agent.run(message, user_id)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def chat_multimodal(self, message: str, user_id: str,
                               chat_id=None,
                               files: list[tuple[str, bytes, str]] = None) -> dict:
        from ..multimodal import file_to_content_block
        content_blocks = [{"type": "text", "text": message}]
        for fname, data, ct in (files or []):
            block = file_to_content_block(data, fname, ct)
            content_blocks.append(block)
            await self.agent.ingest_file(
                data, fname, source="slack", user_id=user_id, mime_type=ct)
        response = await self.agent.run(content_blocks, user_id)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def close(self):
        pass


# ══════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════

async def run_slack(agent, config: dict):
    """Run Slack bot (standalone mode using Socket Mode)."""
    try:
        from slack_bolt import App  # noqa: F401
        from slack_bolt.adapter.socket_mode import SocketModeHandler  # noqa: F401
    except ImportError:
        raise ImportError(
            "slack-bolt is required: pip install liteagent[slack]\n"
            "  or: pip install slack_sdk 'slack-bolt>=1.18'")

    slack_cfg = config.get("channels", {}).get("slack", {})
    bot_token_env = slack_cfg.get("bot_token_env", "SLACK_BOT_TOKEN")
    app_token_env = slack_cfg.get("app_token_env", "SLACK_APP_TOKEN")

    bot_token = os.environ.get(bot_token_env) or slack_cfg.get("bot_token", "")
    app_token = os.environ.get(app_token_env) or slack_cfg.get("app_token", "")

    if not bot_token:
        raise ValueError(
            f"Slack bot token not found. Set {bot_token_env} env var "
            "or configure channels.slack.bot_token in config.json")
    if not app_token:
        raise ValueError(
            f"Slack app token not found (required for Socket Mode). "
            f"Set {app_token_env} env var or configure channels.slack.app_token")

    api_client = _SlackDirectAdapter(agent)
    bot = LiteAgentSlackBot(api_client=api_client, config=slack_cfg)

    logger.info("Starting Slack bot (Socket Mode)...")
    print("[Slack] Bot starting (Socket Mode)...")

    app, handler = bot.build_app(bot_token, app_token)

    try:
        # SocketModeHandler.start() is blocking; run in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, handler.start)
    finally:
        await bot.close()
        await api_client.close()
