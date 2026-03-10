---
name: slack
description: "Slack channel knowledge — Socket Mode bot, mention gating, emoji reactions, thread replies, and access control."
metadata:
  keywords:
    - slack
    - слак
    - slack bot
    - slack канал
    - slack channel
    - slack message
    - slack workspace
    - slack dm
    - слак бот
    - slack thread
    - slack reply
    - slack app
    - slack emoji
    - socket mode
    - bolt
    - app token
    - bot token
    - allowed_channel_ids slack
    - slack mention
  requires: []
  tools: []
---

## Slack Channel

LiteAgent runs as a Slack app via **Socket Mode** (no public URL needed).

### Mention Gating
In **channels**, the bot only responds when **@mentioned** by default.
- DMs / App Home: bot always responds.
- Channels: requires `@BotName` mention unless `require_mention_in_channels: false` in config.
- The mention text is stripped before the message is processed.

### Replies and Threads
- **DMs**: replies are sent directly.
- **Channels**: replies are posted in the same **thread** as the original message (Slack convention).

### Status Reactions
The bot adds emoji reactions to messages to show its live state (Slack uses `:name:` format):
- 👀 `:eyes:` — message queued
- 🤔 `:thinking_face:` — thinking / generating
- ⚡ `:zap:` — running web search
- 👨‍💻 `:technologist:` — running code / file tool
- 🔥 `:fire:` — running a generic tool
- ✅ `:white_check_mark:` — done
- 😱 `:scream:` — error

Reactions are replaced (not accumulated) as state changes, and removed after completion.

### Access Control
```json
{
  "channels": {
    "slack": {
      "allowed_channel_ids": [],  // [] = all channels
      "allowed_user_ids": []      // [] = all users
    }
  }
}
```

### Setup Requirements
Slack requires two tokens:
1. **Bot token** (`xoxb-...`) — from OAuth & Permissions page. Required scopes: `chat:write`, `reactions:write`, `im:history`, `channels:history`, `groups:history`.
2. **App-level token** (`xapp-...`) — from App-Level Tokens (with `connections:write` scope). Needed for Socket Mode.

```json
{
  "channels": {
    "slack": {
      "bot_token_env": "SLACK_BOT_TOKEN",
      "app_token_env": "SLACK_APP_TOKEN",
      "require_mention_in_channels": true,
      "allowed_channel_ids": [],
      "allowed_user_ids": []
    }
  }
}
```

### Starting the Bot
```bash
python -m liteagent --channel slack
```

Enable Socket Mode in your Slack app settings (App Settings → Socket Mode → Enable).
