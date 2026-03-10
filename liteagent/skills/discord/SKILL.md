---
name: discord
description: "Discord channel knowledge — bot behaviour, mention gating, status reactions, access control, and channel routing."
metadata:
  keywords:
    - discord
    - дискорд
    - discord bot
    - discord сервер
    - discord канал
    - mention gating
    - реакции дискорд
    - emoji reaction
    - discord server
    - discord channel
    - discord dm
    - дискорд бот
    - @mention
    - require mention
    - discord guild
    - allowed_guild_ids
    - allowed_channel_ids
    - discord attachment
    - discord file
  requires: []
  tools: []
---

## Discord Channel

LiteAgent runs as a Discord bot. Key behaviours to know:

### Mention Gating
In **servers (guilds)**, the bot only responds when **@mentioned** by default.
- DMs: bot always responds.
- Servers: requires `@BotName` mention unless `require_mention_in_servers: false` in config.
- The mention text is stripped before processing (agent sees a clean message).

### Status Reactions
The bot adds emoji reactions to messages to show its live state:
- 👀 — message queued
- 🤔 — thinking / generating
- ⚡ — running web search tool
- 👨‍💻 — running code / file tool
- 🔥 — running a generic tool
- ✅ — done (removed after 1.5s)
- 😱 — error occurred

Reactions update in real-time and are automatically cleaned up.

### Access Control
Configure allow-lists in `config.json`:
```json
{
  "channels": {
    "discord": {
      "allowed_guild_ids": [],   // [] = all guilds allowed
      "allowed_channel_ids": [], // [] = all channels allowed
      "allowed_user_ids": []     // [] = all users allowed
    }
  }
}
```
All lists are optional. Non-empty list restricts to only those IDs.

### File Attachments
Discord attachments (images, documents, etc.) are automatically downloaded and sent to the agent as multimodal content.

### Message Length
Discord has a 2000 character limit per message. Long responses are automatically split at natural break points (newlines, then spaces).

### Configuration Reference
```json
{
  "channels": {
    "discord": {
      "token_env": "DISCORD_BOT_TOKEN",
      "require_mention_in_servers": true,
      "command_prefix": "/",
      "allowed_guild_ids": [],
      "allowed_channel_ids": [],
      "allowed_user_ids": []
    }
  }
}
```

### Starting the Bot
```bash
python -m liteagent --channel discord
```
Or with API server: configure `channels.discord.enabled: true` in config.
