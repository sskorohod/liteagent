"""Discord channel adapter for LiteAgent.

Inspired by OpenClaw's Discord channel architecture.

Features (ported from OpenClaw):
- Status reactions: emoji reactions on messages showing agent state
  (queued → thinking → tool → done / error)
- Mention gating: in servers, only respond when @mentioned
- Typing lifecycle: robust typing indicator with keepalive
- Allow/block lists: per-channel and per-user access control
- Thread/channel routing: DMs always respond; servers require @mention by default

Setup:
  1. Create a Discord application & bot at https://discord.com/developers/applications
  2. Enable "Message Content Intent" in the bot settings
  3. Add your bot token to config.json:

     "channels": {
       "discord": {
         "token_env": "DISCORD_BOT_TOKEN",
         "require_mention_in_servers": true,
         "allowed_guild_ids": [],       // empty = all guilds
         "allowed_channel_ids": [],     // empty = all channels
         "allowed_user_ids": [],        // empty = all users
         "command_prefix": "/"
       }
     }

  4. Run: python -m liteagent --channel discord

Required package:
  pip install "discord.py>=2.3"  (or: pip install liteagent[discord])
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════
# STATUS REACTION ADAPTER FOR DISCORD
# ══════════════════════════════════════════

def _build_discord_reaction_adapter(message):
    """Build StatusReactionAdapter for a Discord message."""
    from ..status_reactions import StatusReactionAdapter

    class _DiscordReactionAdapter(StatusReactionAdapter):
        def __init__(self):
            self._current: str | None = None

        async def set_reaction(self, emoji: str) -> None:
            try:
                if self._current:
                    await self.remove_reaction(self._current)
                await message.add_reaction(emoji)
                self._current = emoji
            except Exception as exc:
                logger.debug("Discord set_reaction(%s) failed: %s", emoji, exc)

        async def remove_reaction(self, emoji: str) -> None:
            try:
                await message.remove_reaction(emoji, message.guild.me
                                               if message.guild else message.author)
            except Exception as exc:
                logger.debug("Discord remove_reaction(%s) failed: %s", emoji, exc)

    return _DiscordReactionAdapter()


# ══════════════════════════════════════════
# MENTION GATING (from OpenClaw mention-gating.ts)
# ══════════════════════════════════════════

def _is_dm(message) -> bool:
    import discord
    return isinstance(message.channel, discord.DMChannel)


def _was_bot_mentioned(message, bot_user) -> bool:
    """Return True if the bot user was mentioned in the message."""
    if bot_user in message.mentions:
        return True
    # Check for @everyone / @here — don't treat as mention
    return False


def _check_mention_gate(message, bot_user, require_mention: bool) -> bool:
    """Return True if message should be processed."""
    if _is_dm(message):
        return True  # Always respond in DMs
    if not require_mention:
        return True
    return _was_bot_mentioned(message, bot_user)


# ══════════════════════════════════════════
# ACCESS CONTROL (from OpenClaw allow-from.ts)
# ══════════════════════════════════════════

def _is_allowed(message, cfg: dict) -> bool:
    """Check if message sender/channel/guild is in the allow list."""
    allowed_guilds = set(cfg.get("allowed_guild_ids", []))
    allowed_channels = set(cfg.get("allowed_channel_ids", []))
    allowed_users = set(cfg.get("allowed_user_ids", []))

    if allowed_guilds and message.guild:
        if message.guild.id not in allowed_guilds:
            return False

    if allowed_channels:
        if message.channel.id not in allowed_channels:
            return False

    if allowed_users:
        if message.author.id not in allowed_users:
            return False

    return True


# ══════════════════════════════════════════
# RESPONSE CHUNKING (Discord 2000 char limit)
# ══════════════════════════════════════════

DISCORD_MAX_LENGTH = 2000


def _split_message(text: str, max_len: int = DISCORD_MAX_LENGTH) -> list[str]:
    """Split a message into Discord-safe chunks."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        # Find a good split point (prefer newline, then space)
        split_at = text.rfind("\n", 0, max_len)
        if split_at <= 0:
            split_at = text.rfind(" ", 0, max_len)
        if split_at <= 0:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


# ══════════════════════════════════════════
# DISCORD BOT CLIENT
# ══════════════════════════════════════════

class LiteAgentDiscordBot:
    """Discord bot that wraps LiteAgent.

    Routes messages through TelegramAPIClient-style HTTP proxy when running
    alongside API server, or directly via _DirectAPIAdapter otherwise.
    """

    def __init__(self, api_client, config: dict):
        self._api_client = api_client
        self._cfg = config
        self._require_mention = config.get("require_mention_in_servers", True)
        self._prefix = config.get("command_prefix", "/")
        self._client = None

    def _build_client(self):
        import discord

        intents = discord.Intents.default()
        intents.message_content = True  # Required for reading message content
        intents.members = True

        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            logger.info("Discord bot ready: %s (id=%s)", client.user, client.user.id)
            print(f"[Discord] Bot logged in as {client.user} ({client.user.id})")

        @client.event
        async def on_message(message):
            await self._handle_message(client, message)

        self._client = client
        return client

    async def _handle_message(self, client, message):
        """Process an incoming Discord message."""
        # Ignore own messages
        if message.author == client.user:
            return

        # Access control
        if not _is_allowed(message, self._cfg):
            logger.debug("Discord message blocked: %s/%s", message.guild, message.author)
            return

        # Mention gating (from OpenClaw)
        if not _check_mention_gate(message, client.user, self._require_mention):
            logger.debug("Discord mention gate: skipping message (no mention)")
            return

        # Strip bot mention from text
        text = message.content
        if client.user.mention in text:
            text = text.replace(client.user.mention, "").strip()
        if not text and not message.attachments:
            return

        user_id = f"discord-{message.author.id}"
        logger.info("Discord message from %s: %s", user_id, text[:80])

        # Build status reaction controller (from OpenClaw)
        from ..status_reactions import StatusReactionController
        reaction_ctrl = None
        try:
            adapter = _build_discord_reaction_adapter(message)
            reaction_ctrl = StatusReactionController(adapter=adapter)
            await reaction_ctrl.set_queued()
        except Exception:
            reaction_ctrl = None

        try:
            # Show typing indicator
            async with message.channel.typing():
                if reaction_ctrl:
                    await reaction_ctrl.set_thinking()

                # Handle attachments (images, documents)
                if message.attachments:
                    result = await self._handle_with_attachments(
                        message, text, user_id, reaction_ctrl)
                else:
                    result = await self._api_client.chat(
                        text, user_id, chat_id=message.channel.id)

            # Done reaction
            if reaction_ctrl:
                await reaction_ctrl.set_done()
                await asyncio.sleep(1.5)
                await reaction_ctrl.clear()

            # Send response
            response = result.get("response", "")
            if response:
                for chunk in _split_message(response):
                    await message.reply(chunk)

        except Exception as exc:
            logger.error("Discord message error: %s", exc, exc_info=True)
            if reaction_ctrl:
                try:
                    await reaction_ctrl.set_error()
                    await asyncio.sleep(2.5)
                    await reaction_ctrl.clear()
                except Exception:
                    pass
            await message.reply(_user_friendly_error(exc))

    async def _handle_with_attachments(self, message, text: str,
                                        user_id: str, reaction_ctrl) -> dict:
        """Handle message with file attachments."""
        files = []
        for attachment in message.attachments:
            try:
                data = await attachment.read()
                files.append((attachment.filename, data, attachment.content_type or ""))
                logger.info("Discord attachment: %s (%d bytes)",
                            attachment.filename, len(data))
            except Exception as exc:
                logger.warning("Failed to download attachment %s: %s",
                                attachment.filename, exc)

        if files:
            return await self._api_client.chat_multimodal(
                message=text or "User sent files.",
                user_id=user_id,
                chat_id=message.channel.id,
                files=files,
            )
        return await self._api_client.chat(text, user_id, chat_id=message.channel.id)

    async def start(self, token: str):
        """Start the Discord bot."""
        client = self._build_client()
        await client.start(token)

    async def close(self):
        if self._client:
            await self._client.close()


def _user_friendly_error(exc: Exception) -> str:
    msg = str(exc).lower()
    if "timeout" in msg:
        return "Request timed out. Please try again."
    if "rate" in msg and "limit" in msg:
        return "Rate limit hit. Please wait a moment."
    return "An error occurred. Please try again."


# ══════════════════════════════════════════
# DIRECT ADAPTER (no HTTP proxy)
# ══════════════════════════════════════════

class _DiscordDirectAdapter:
    """Direct adapter for standalone Discord mode (no API server)."""

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
                data, fname, source="discord", user_id=user_id, mime_type=ct)
        response = await self.agent.run(content_blocks, user_id)
        from ..file_queue import get_file_queue, serialize_file_queue
        return {"response": response, "files": serialize_file_queue(get_file_queue())}

    async def close(self):
        pass


# ══════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════

async def run_discord(agent, config: dict):
    """Run Discord bot (standalone mode)."""
    try:
        import discord  # noqa: F401
    except ImportError:
        raise ImportError(
            "discord.py is required: pip install liteagent[discord]\n"
            "  or: pip install 'discord.py>=2.3'")

    discord_cfg = config.get("channels", {}).get("discord", {})
    token_env = discord_cfg.get("token_env", "DISCORD_BOT_TOKEN")
    token = os.environ.get(token_env) or discord_cfg.get("token", "")
    if not token:
        raise ValueError(
            f"Discord token not found. Set {token_env} environment variable "
            "or configure channels.discord.token in config.json")

    api_client = _DiscordDirectAdapter(agent)
    bot = LiteAgentDiscordBot(api_client=api_client, config=discord_cfg)

    logger.info("Starting Discord bot...")
    print("[Discord] Bot starting...")
    try:
        await bot.start(token)
    finally:
        await bot.close()
        await api_client.close()
