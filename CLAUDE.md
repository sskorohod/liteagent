# CLAUDE.md — LiteAgent Developer Guide

## Quick Start

```bash
cd /Users/vskorokhod/liteagent
source .venv/bin/activate
python -m liteagent                    # CLI mode
python -m liteagent --api              # API + Dashboard (default :8080)
python -m liteagent --telegram         # Telegram bot
python -m liteagent --api --telegram   # Both channels
```

**Tests:**
```bash
.venv/bin/python -m pytest tests/ -v          # All tests (~556)
.venv/bin/python -m pytest tests/test_voice.py -v  # Single module
```

**Install (editable):**
```bash
pip install -e ".[all]"    # All extras
pip install -e ".[dev]"    # Dev only
```

---

## Project Structure

```
liteagent/
├── __init__.py              # Package marker, __version__ = "1.0.0"
├── __main__.py              # python -m liteagent entry point
├── main.py                  # CLI arg parser, vault/backup CLI, channel router
├── agent.py                 # Core agent loop (LiteAgent class)
├── providers.py             # LLM providers (Anthropic, OpenAI, Ollama, Gemini)
├── config.py                # Config loader, key management, validation
├── memory.py                # 4-layer memory system (L1-L4)
├── tools.py                 # ToolRegistry, MCP support, builtin tools
├── rag.py                   # RAG pipeline (ingest, chunk, search)
├── voice.py                 # Voice engine (TTS + STT, 3+3 providers)
├── metacognition.py         # Confidence gate, counterfactual replay, dream cycle
├── evolution.py             # Self-evolving prompt, style adaptation, proactive agent
├── synthesis.py             # Auto tool synthesis, skill crystallization
├── planning.py              # Internal monologue, chain-of-thought
├── onboarding.py            # Interactive setup wizard
├── scheduler.py             # Async cron scheduler with retry/timeout
├── pool.py                  # Multi-agent pool, cross-agent delegation
├── circuit_breaker.py       # Provider resilience (closed/open/half_open)
├── boot.py                  # Proactive startup checks from boot.md
├── health.py                # Channel/provider health polling
├── hooks.py                 # Lifecycle hook system (30+ hook points)
├── plugins.py               # Plugin loader (~/.liteagent/plugins/)
├── file_queue.py            # ContextVar per-request file queue
├── multimodal.py            # Content blocks (images, PDFs, code)
├── storage.py               # S3/MinIO file storage
├── vault.py                 # Encrypted key vault (Fernet + PBKDF2)
├── backup.py                # tar.gz backup/restore
├── config_watcher.py        # Hot config reload (mtime + SHA-256)
├── logging_config.py        # Structured JSON logging + console
├── tasks.py                 # User-facing task manager (SQLite)
├── channels/
│   ├── cli.py               # Interactive REPL
│   ├── api.py               # FastAPI REST + SSE + WebSocket + TTS
│   ├── dashboard.py         # Web SPA routes (6 tabs)
│   └── telegram.py          # Telegram bot (python-telegram-bot)
└── static/
    └── dashboard.html       # SPA (Tailwind, Chart.js, WebSocket)

tests/                       # 31 test files, ~6,500 LOC
├── conftest.py              # Shared fixtures
├── test_agent.py            # Core loop, cascade routing
├── test_voice.py            # TTS/STT providers (44 tests)
├── test_dashboard.py        # Dashboard API routes
├── test_telegram.py         # Telegram handlers
├── ... (28 more test files)

config.json                  # Main config (gitignored)
soul.md                      # System prompt / personality
pyproject.toml               # Build config, 12 optional extras
Makefile                     # 20+ targets
install.sh                   # Interactive installer
Dockerfile / docker-compose.yml
```

---

## Architecture Overview

### Core Agent (agent.py — LiteAgent class)

The central class. Key methods:

- `run(message, user_id)` — single response (non-streaming)
- `stream(message, user_id)` — async generator yielding chunks
- `_call_api(messages, model, tools)` — LLM call with provider abstraction
- `_select_model(message)` — cascade routing by complexity score
- `handle_command(text)` — /help, /clear, /memories, etc.
- `_wire_voice_tool()` — registers `transcribe_voice` tool (delegates to voice.py)
- `_apply_voice_transcription_mode()` — auto-transcription for voice messages

**Per-user locking:** `asyncio.Lock` per `user_id` prevents race conditions.

**Message flow:**
1. Channel receives message → `agent.run()` or `agent.stream()`
2. Cascade model selection (Haiku/Sonnet/Opus by complexity)
3. System prompt (soul.md) + memory context + tool definitions
4. LLM call → tool execution loop (max_iterations)
5. Post-processing: auto-TTS, file queue flush
6. Response returned to channel

### Providers (providers.py)

Unified interface for 4 LLM backends:

| Provider | Class | Models |
|----------|-------|--------|
| Anthropic | `AnthropicProvider` | Claude Haiku/Sonnet/Opus |
| OpenAI | `OpenAIProvider` | GPT-4o, GPT-4o-mini |
| Gemini | `GeminiProvider` | Gemini 2.0 Flash/Pro |
| Ollama | `OllamaProvider` | Any local model |

**Key data classes:** `TextBlock`, `ToolUseBlock`, `LLMResponse`, `TokenUsage`

Select via `config.agent.provider` (default: "anthropic").

### Memory (memory.py — 4 layers)

- **L1 Conversation** — RAM, current session messages
- **L2 Scoped State** — SQLite, per-user key-value store
- **L3 Semantic Recall** — Embeddings + keyword search, cosine similarity
- **L4 Knowledge** — Auto-extracted facts via Haiku, importance scoring

**Temporal decay:** `score * exp(-decay_rate * days_since_access)` for graceful forgetting.

### Voice Engine (voice.py)

**TTS providers (text → audio):**
- **OpenAI TTS** — 14 voices (alloy, ash, coral, echo, nova, sage...), models: tts-1, tts-1-hd, gpt-4o-mini-tts
- **ElevenLabs** — REST API, voice_settings (stability, similarity_boost, style, speed)
- **Edge TTS** — Free Microsoft neural TTS, no API key needed

**STT providers (audio → text):**
- **OpenAI Whisper** — whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe
- **Deepgram** — Nova-3 model, REST API
- **Groq** — whisper-large-v3, OpenAI-compatible endpoint

**Auto-TTS modes:** `off`, `always`, `inbound` (echo voice), `tagged` (only `[[tts:...]]` directives)

**Provider fallback:** primary → remaining providers → fail gracefully.

**Output format:** Opus for Telegram (voice bubble), MP3 for API/dashboard.

**Built-in presets:** `professional`, `casual`, `storyteller`, `fast_free`, `russian` — agent can save/load custom presets.

**Cost awareness:** `TTS_COST_INFO` / `STT_COST_INFO` — agent knows pricing per provider.

**Voice config tools (6 tools for agent self-configuration):**
- `get_voice_settings` — current config + provider status + pricing + presets
- `set_voice_settings` — update any TTS/STT parameter with validation
- `list_voice_providers` — all providers with models, voices, configured status
- `test_tts` — convert text to audio (with optional voice/provider override)
- `save_voice_preset` / `load_voice_preset` — named voice profiles

**Voice skill prompt injection:** Technical prompt about voice tools is injected into system prompt **only** when user mentions voice-related keywords (голос, tts, stt, voice, озвуч, etc.). Not loaded during regular chat.

### Tools (tools.py — ToolRegistry)

- Decorator-based: `@tool_registry.register(name, description, parameters)`
- MCP server support (external tool sources)
- Builtin: `read_file`, `write_file`, `exec_command`, `web_search`, `transcribe_voice`, `download_file`, `send_file_to_user`
- Voice config: `get_voice_settings`, `set_voice_settings`, `list_voice_providers`, `test_tts`, `save_voice_preset`, `load_voice_preset`
- Secret scanning on outputs
- Command allowlist for exec_command

### Channels

**CLI** (`channels/cli.py`): Interactive REPL with streaming, /commands.

**API** (`channels/api.py`): FastAPI app with:
- `POST /chat` — text chat (+ auto-TTS)
- `POST /chat/voice` — voice message (STT + response + auto-TTS)
- `POST /chat/multimodal` — file upload (images, PDFs, code)
- `GET /chat/stream` — SSE streaming
- `WebSocket /ws` — real-time events hub
- `POST /tts/convert` — text → audio conversion
- `GET /tts/status`, `GET /tts/providers` — TTS info
- Session auth via `config.channels.api.password`

**Dashboard** (`channels/dashboard.py`): 6 tabs — Overview, Usage, Memories, Tools, Chat, Settings.
- Settings includes: Agent, Provider, Telegram, Voice (TTS+STT), Features
- REST API under `/api/` prefix

**Telegram** (`channels/telegram.py`): python-telegram-bot, routes through local API via `TelegramAPIClient`.
- Voice messages: download → STT → response → optional TTS voice bubble
- File support: download_file tool, send_file_to_user tool, voice_compatible flag

### File Queue (file_queue.py)

`ContextVar`-based per-request queue. Agent tools call `enqueue_file(path, caption, mime_type, voice_compatible)` → API/Telegram flushes queue after response.

---

## Config Reference (config.json)

```json
{
  "agent": {
    "name": "Agent Name",
    "soul": "soul.md",
    "provider": "anthropic",
    "default_model": "claude-sonnet-4-20250514",
    "timezone": "Europe/Moscow"
  },
  "memory": {
    "db_path": "~/.liteagent/memory.db",
    "max_history_tokens": 8000,
    "keep_recent_messages": 20,
    "auto_learn": true,
    "temporal_decay_enabled": true,
    "temporal_decay_rate": 0.01
  },
  "cost": {
    "cascade_routing": true,
    "prompt_caching": true,
    "context_compression": true,
    "budget_daily_usd": 5.0
  },
  "channels": {
    "telegram": { "token_env": "TELEGRAM_BOT_TOKEN" },
    "api": { "host": "0.0.0.0", "port": 8080, "password": null }
  },
  "voice": {
    "tts": {
      "auto": "off",
      "provider": "openai",
      "max_length": 1500,
      "openai": { "model": "tts-1", "voice": "alloy" },
      "elevenlabs": { "voice_id": "pMsXgVXv3BLzUgSXRplE", "model_id": "eleven_multilingual_v2" },
      "edge": { "voice": "ru-RU-SvetlanaNeural" }
    },
    "stt": {
      "provider": "openai",
      "openai": { "model": "whisper-1" }
    }
  },
  "features": {
    "dream_cycle": false,
    "self_evolving_prompt": false,
    "proactive_agent": false,
    "confidence_gate": false,
    "planning": true
  },
  "providers": {
    "anthropic": { "api_key_env": "ANTHROPIC_API_KEY" },
    "openai": {},
    "gemini": {},
    "ollama": { "base_url": "http://localhost:11434" }
  },
  "hooks": {},
  "plugins": {},
  "health": { "enabled": false },
  "boot": { "enabled": false }
}
```

**API keys** stored separately in `~/.liteagent/keys.json` (chmod 600) or encrypted vault.

---

## Key Patterns

### Async-First
Everything is `async/await`. Use `asyncio.Lock` per user_id. Never block the event loop — use `run_in_executor()` for sync I/O (e.g., ElevenLabs/Deepgram HTTP calls).

### Graceful Degradation
- Missing optional dependencies → feature silently disabled
- Provider fails → circuit breaker → fallback to next provider
- TTS/STT fails → text-only response (no crash)
- Config file missing → defaults work

### Import Guards
Optional packages imported inside functions, not at module level:
```python
def some_function():
    import openai  # Only fails if actually called without package
```

### Config Validation
`validate_config()` in config.py checks known keys, warns on unknown. Top-level keys: `agent, memory, tools, channels, cost, providers, logging, features, rag, storage, hooks, plugins, boot, health, voice, scheduler, agents`.

### Testing
- Mock patterns: use `patch.dict("sys.modules", {"openai": mock})` for packages imported inside functions
- Use `patch("urllib.request.urlopen")` for HTTP calls via urllib
- Shared fixtures in `conftest.py`: `mock_agent`, `mock_config`, `tmp_path` configs
- `asyncio_mode = "auto"` in pyproject.toml

---

## Dependencies (pyproject.toml extras)

| Extra | Packages |
|-------|----------|
| `api` | fastapi, uvicorn |
| `telegram` | python-telegram-bot |
| `openai` | openai |
| `gemini` | google-generativeai |
| `ollama` | openai (compatible API) |
| `embeddings` | sentence-transformers |
| `pdf` | pymupdf |
| `storage` | boto3 |
| `qdrant` | qdrant-client |
| `vault` | cryptography |
| `voice` | edge-tts |
| `dev` | pytest, pytest-asyncio, pytest-cov |
| `all` | Everything above |

**Core dependency:** `anthropic>=0.42.0` (always installed).

---

## Data Locations

| Path | Purpose |
|------|---------|
| `~/.liteagent/` | User data directory |
| `~/.liteagent/keys.json` | API keys (chmod 600) |
| `~/.liteagent/auth_token` | API bearer token |
| `~/.liteagent/memory.db` | SQLite (memory, tasks, scheduler) |
| `~/.liteagent/liteagent.log` | Structured JSON log |
| `~/.liteagent/plugins/` | User plugins |
| `~/.liteagent/backups/` | tar.gz backups |

---

## Metrics

- **Package code:** ~13,800 LOC (30 modules)
- **Test code:** ~6,700 LOC (31 files, 580 tests)
- **Dashboard SPA:** ~152 KB single HTML file
- **Version:** 1.0.0
- **Python:** >=3.10
- **License:** MIT
