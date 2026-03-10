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

### Memory (memory.py — current implementation, Mar 2026)

The memory system is now a hybrid stack with identity normalization, canonical profile slots,
type-aware retrieval, explainability traces, and background indexing workers.

**Core layers (still valid):**
- **L1 Conversation** — RAM message buffers (`_conversations`) + persisted `chat_history`
- **L2 Scoped State** — `user_state` / `app_state`
- **L3 Semantic Recall** — hybrid vector + FTS/BM25 + graph recall + temporal decay
- **L4 Knowledge Extractor** — auto extraction (`facts/preferences/corrections`) from turns
- **L5 Memory Exchange** — precomputed context packs + shadow predictions
- **L6 Quality/Explainability** — recall traces + quality KPIs

#### 1) Canonical identity across channels

**Problem solved:** `dashboard-user`, `api-user`, `tg-*` could diverge.

**Now:**
- `user_identity_map(alias_user_id -> person_id, source, confidence)` stores canonical mapping.
- `MemorySystem.get_canonical_person_id()` resolves every memory operation to canonical `person_id`.
- `MemorySystem.set_user_alias()` persists mapping and triggers `_merge_identity_data(alias, person)`:
  - rewrites `user_id` in memory/chat/metrics tables
  - merges `user_state`, `session_summaries`, `style_profiles`, canonical slot history
  - merges in-RAM conversation buffers
- `LiteAgent.resolve_user_id()` now checks persistent identity map first, then config aliases,
  then heuristic auto-aliasing.

#### 2) Canonical profile slots with versioning

Added tables:
- `canonical_profile_slots(person_id, slot_key, slot_value, confidence, version, source, updated_at)`
- `canonical_profile_slot_history(...)` immutable history

Supported canonical slots: `name`, `language`, `role`.

`upsert_canonical_slot()` behavior:
- confidence-aware updates (keeps stronger existing value unless new evidence is better)
- increments `version` on real value change
- writes immutable history row on every accepted update

`get_user_profile()` now overlays canonical slot values on top of `user:profile_facts`.

#### 3) Anti-pollution memory filter

`_is_memory_pollution_text()` denies low-signal assistant disclaimers from entering long-term memory.

Examples filtered:
- "I can't remember previous conversations"
- "As an AI..."
- RU equivalents ("я не помню прошлые разговоры", etc.)

Applied in:
- `remember()` for `fact/preference/correction`
- `extract_and_learn()` cleanup pipeline

Extraction quality accounting is stored in `memory_extraction_runs`:
- `total_candidates`, `saved_count`, `dropped_pollution`

#### 4) Type-aware retrieval (personal queries)

Added `recall_type_aware(query, user_id)`:
- classifies personal intent (`_classify_query_intent`) and slot (`name/language/role`)
- injects canonical profile slot memory first when available
- ranks by slot-match + memory type bonuses:
  - profile slot > fact > preference > correction > generic
- then falls back to standard hybrid recall ordering

This is now used by agent prompt building via `_cached_recall()` and by `memory_search` tool.

#### 5) Memory Exchange daemon + local worker

Daemon queue (`memory_exchange_intents`) already existed; now extended with:
- priority scheduling, retry/fail status lifecycle, auto-pause on high active/queued load
- always-on local worker loop (`run_local_memory_worker_once`) running inside daemon loop:
  - backfills canonical slots from newly stored memories
  - can back-link memories to graph entities
  - controlled by:
    - `memory_local_worker_enabled`
    - `memory_local_worker_interval_sec`
    - `memory_local_worker_batch_size`

Daemon state exposes local-worker stats:
- `local_worker_last_run`, `local_worker_last_stats`

#### 6) Explainability and quality metrics

Explainability:
- `memory_recall_traces` stores latest retrieval traces:
  - query, strategy, intent slot, profile expected/hit, top 3-5 memories with scores
- APIs use `get_last_recall_trace()`

Quality KPIs (`get_memory_quality_metrics()`):
- `recall_at_k`
- `recall_confident_at_k`
- `profile_accuracy`
- `contradiction_rate` (canonical slot value conflicts)
- `memory_poison_rate`

These are attached to `get_memory_metrics()` and dashboard telemetry.

#### 7) Dashboard/API surfaces for memory ops

Key endpoints:
- `GET /api/memory/exchange`:
  - queue + daemon + token economics
  - `identity`, `quality_metrics`, `explainability`
- `GET /api/memory/explain?user_id=&limit=`
- `GET /api/memory/identity?user_id=`
- `POST /api/memory/identity` (manual alias mapping)

Settings endpoint (`POST /api/settings/memory`) now also saves local-worker controls.

#### 8) Practical debugging checklist (for future Claude edits)

When user says "agent forgot my name":
1. `GET /api/memory/identity?user_id=dashboard-user` — verify canonical `person_id`.
2. `GET /api/memory/exchange` — inspect `identity.aliases`, `quality_metrics.profile_accuracy`.
3. `GET /api/memory/explain?limit=5` — verify top memories and whether profile slot was used.
4. Check canonical slot state in DB (`canonical_profile_slots`) for `slot_key='name'`.
5. Ensure polluted corrections are not dominating (`memory_poison_rate` high => tune filters).

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
- Dashboard-mounted memory endpoints:
  - `GET /api/memory/exchange`
  - `GET /api/memory/explain`
  - `GET/POST /api/memory/identity`
- Session auth via `config.channels.api.password`

**Dashboard** (`channels/dashboard.py`): 6 tabs — Overview, Usage, Memories, Tools, Chat, Settings.
- Memory panel includes Memory Exchange Monitor + Explainability + quality KPIs
- Settings includes: Agent, Provider, Telegram, Voice (TTS+STT), Features, Memory worker controls
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
    "temporal_decay_rate": 0.01,
    "memory_exchange_enabled": true,
    "memory_exchange_top_k": 8,
    "memory_exchange_pack_budget_tokens": 420,
    "memory_exchange_max_packs": 2,
    "memory_exchange_context_budget_tokens": 700,
    "memory_exchange_daemon_enabled": true,
    "memory_exchange_daemon_interval_sec": 1.0,
    "memory_exchange_daemon_batch_size": 3,
    "memory_exchange_daemon_auto_pause": true,
    "memory_exchange_daemon_pause_active_requests": 1,
    "memory_exchange_daemon_pause_queued_requests": 2,
    "memory_exchange_queue_max_pending": 5000,
    "memory_exchange_max_attempts": 3,
    "memory_local_worker_enabled": true,
    "memory_local_worker_interval_sec": 12.0,
    "memory_local_worker_batch_size": 24,
    "shadow_twin_enabled": true,
    "shadow_twin_predictions": 3,
    "shadow_twin_use_llm": false
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
