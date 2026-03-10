# LiteAgent Project Overview

This file is a practical handoff for any future agent or engineer working on LiteAgent.
It is intentionally focused on how the system works now, where the important code lives,
how to run and verify it, and what has already been implemented recently.

## 1. What This Project Is

LiteAgent is a local-first, extensible AI agent platform with:

- chat via API, dashboard, and Telegram
- long-term memory with identity normalization and explainability
- tools, file handling, storage, knowledge base, and document intelligence
- voice I/O with multiple TTS/STT providers
- proactive/metacognitive features
- long-running autonomous goals
- an autonomous night-coding mode for overnight local-model development work

Primary repo path:

- `/Users/vskorokhod/liteagent`

Main package:

- `/Users/vskorokhod/liteagent/liteagent`

Primary web UI:

- `/Users/vskorokhod/liteagent/liteagent/static/dashboard.html`

## 2. Current Runtime Model

Current default runtime is local-model-first:

- provider: `ollama`
- default model: `qwen3-coder:30b`

Current config file:

- `/Users/vskorokhod/liteagent/config.json`

Important current runtime facts:

- storage is enabled and connected to MinIO
- knowledge base is enabled
- Chrome DevTools MCP is configured and available
- many autonomy/metacognition features are enabled

## 3. How To Run

From repo root:

```bash
cd /Users/vskorokhod/liteagent
source .venv/bin/activate
```

CLI:

```bash
python -m liteagent
```

API + dashboard:

```bash
python -m liteagent --channel api
```

Telegram:

```bash
python -m liteagent --channel telegram
```

Dashboard URL:

- [http://127.0.0.1:8080](http://127.0.0.1:8080)

Health:

- [http://127.0.0.1:8080/health](http://127.0.0.1:8080/health)

## 4. Core Architecture

### 4.1 Agent Loop

Main class:

- `/Users/vskorokhod/liteagent/liteagent/agent.py`

The agent is responsible for:

- request handling
- prompt building
- memory recall injection
- tool loop execution
- model routing
- self-healing and response guards
- file queue flushing
- delivery to API/Telegram/dashboard

Important runtime additions already implemented:

- side-effect overclaim guard
- forced recovery after no-tool responses
- health-aware self-healing
- critical response review for important answers
- Telegram delivery correctness guards
- direct file/storage follow-up handlers

### 4.2 Providers

- `/Users/vskorokhod/liteagent/liteagent/providers.py`

Supports:

- Anthropic
- OpenAI
- Gemini
- Ollama

Current practical focus is Ollama/local models.

### 4.3 Memory

- `/Users/vskorokhod/liteagent/liteagent/memory.py`

Memory is not just chat history. It includes:

- canonical identity mapping across channels
- canonical profile slots
- semantic recall
- recall explainability traces
- memory exchange / shadow predictions
- thinking cloud
- human support snapshot
- reinforcement and penalty for recall quality

Recent custom layers added:

- `Thinking Cloud`
- Obsidian export for thinking cloud
- human support opportunities snapshot
- positive and negative recall reinforcement

### 4.4 Goals / Long-Running Work

- `/Users/vskorokhod/liteagent/liteagent/goals.py`

This is the system for background autonomous work.

It provides:

- persistent goals
- plans and plan history
- attempts journal
- cycle execution
- replanning
- daemon coordinator

Goal data is stored in the main SQLite DB.

### 4.5 Night Coding

- `/Users/vskorokhod/liteagent/liteagent/night_coding.py`

This is the autonomous overnight coding mode.

It adds:

- `autonomous_coding` goal type
- local-only coding session config normalization
- stop window support
- browser verification flag
- internet research flag
- continue-after-objective behavior
- max patch files per cycle
- max failed cycles guard

Recent improvements already implemented:

- orphaned `running` goal recovery after coordinator restart
- regression guard pauses night coding after repeated non-improving cycles
- session report generation
- downloadable markdown report per goal

### 4.6 Dashboard

- `/Users/vskorokhod/liteagent/liteagent/channels/dashboard.py`
- `/Users/vskorokhod/liteagent/liteagent/static/dashboard.html`

The dashboard includes:

- overview / live ops
- memories
- tools
- chat
- tasks
- settings
- background goals

Important recent dashboard work:

- professionalized chat rendering
- better markdown, code block controls, copy behavior
- file path reveal from chat
- background goals redesign
- night coding card
- goal inspector
- goal session report in inspector
- download report action

## 5. Storage / S3 / MinIO

Storage backend:

- `/Users/vskorokhod/liteagent/liteagent/storage.py`

Current connected storage:

- endpoint: `http://192.168.1.244:9000`
- bucket: `liteagent`

Credentials are stored in:

- `/Users/vskorokhod/.liteagent/keys.json`

Relevant key names:

- `minio_access`
- `minio_secret`

Current config state:

- `storage.enabled = true`

Dashboard storage endpoints:

- `GET /api/settings/storage`
- `POST /api/settings/storage`
- `POST /api/settings/storage/test`
- `GET /api/storage/status`
- `GET /api/storage/files`

Agent storage tools are wired when storage is connected.

That includes:

- save file
- list files
- get file URL
- send stored file
- send stored file to Telegram

## 6. Knowledge Base and Documents

Knowledge base and document workflows are spread across:

- `/Users/vskorokhod/liteagent/liteagent/knowledge_base.py`
- `/Users/vskorokhod/liteagent/liteagent/documents.py`
- `/Users/vskorokhod/liteagent/liteagent/multimodal.py`
- `/Users/vskorokhod/liteagent/liteagent/file_types.py`
- `/Users/vskorokhod/liteagent/liteagent/file_manager.py`

Current document pipeline behavior:

- store originals in storage
- extract/analyze content
- KB ingest for searchable retrieval
- summary and key point extraction
- save important findings into notes/memory
- create tasks/calendar events when dates/reminders are found

Supported file recognition is broad, including:

- text/code
- PDF
- modern Office formats
- legacy Office formats
- ODF
- EPUB
- SVG
- images/audio/video classification
- archives
- datasets/binary types

## 7. Voice

Voice engine:

- `/Users/vskorokhod/liteagent/liteagent/voice.py`

Current notable state:

- Groq TTS Russian mode is allowed
- Edge TTS has dropdown-based voice/language metadata in dashboard
- voice settings UI has been significantly improved

Dashboard voice settings live in:

- `/Users/vskorokhod/liteagent/liteagent/channels/dashboard.py`
- `/Users/vskorokhod/liteagent/liteagent/static/dashboard.html`

## 8. Telegram

Telegram channel:

- `/Users/vskorokhod/liteagent/liteagent/channels/telegram.py`

Important fixes already done:

- sending text vs file to Telegram
- token fallback from keys.json
- remember Telegram chat target
- allow dashboard/API-to-Telegram delivery
- direct stored-file-to-Telegram tool
- guard against false claims of delivery

Document unlock phrase is implemented for owner document retrieval.

Current unlock phrase key:

- `document_unlock_phrase` in `/Users/vskorokhod/.liteagent/keys.json`

## 9. Intelligent Routing / Autonomy / Metacognition

Important modules:

- `/Users/vskorokhod/liteagent/liteagent/planning.py`
- `/Users/vskorokhod/liteagent/liteagent/metacognition.py`
- `/Users/vskorokhod/liteagent/liteagent/evolution.py`

Important features already implemented or improved:

- planning with assumptions and verification steps
- confidence gate
- self-evolving prompt
- dream cycle
- counterfactual replay
- proactive agent improvements
- human support agent
- critical response review
- more autonomous side-effect behavior

Cascade routing / advisor:

- dashboard can show cascade monitor and recommendations
- routing can recommend better model mix for speed/cost/quality

## 10. Thinking Cloud

Thinking cloud is a separate strategic memory layer.

Implemented in:

- `/Users/vskorokhod/liteagent/liteagent/memory.py`

Exposed in dashboard via:

- `GET /api/memory/thinking`
- `GET /api/export/thinking`

Current behavior:

- stores ideas, directions, constraints, open questions, decision signals
- supports Obsidian-compatible export
- supports markdown notes + canvas export

This is one of the key “human model” features of the system.

## 11. Important Dashboard/API Endpoints

General:

- `GET /health`
- `GET /api/overview`
- `GET /api/overview/enhanced`
- `GET /api/ops/active`
- `GET /api/ops/system`
- `GET /api/ops/recent`

Memory:

- `GET /api/memory/exchange`
- `GET /api/memory/explain`
- `GET /api/memory/identity`
- `POST /api/memory/identity`
- `GET /api/memory/thinking`
- `GET /api/memory/human_support`
- `GET /api/export/thinking`

Goals:

- `GET /api/goals`
- `GET /api/goals/summary`
- `GET /api/goals/{id}/status`
- `GET /api/goals/{id}/report`
- `POST /api/goals`
- `POST /api/goals/{id}/plan`
- `POST /api/goals/{id}/replan`
- `POST /api/goals/{id}/pause`
- `POST /api/goals/{id}/cancel`

Storage:

- `GET /api/settings/storage`
- `POST /api/settings/storage`
- `POST /api/settings/storage/test`
- `GET /api/storage/status`
- `GET /api/storage/files`

Documents:

- `POST /api/documents/upload`
- `GET /api/documents/reviews`
- `GET /api/documents/reviews/{id}`

## 12. Where To Look First For Future Work

If the issue is about chat behavior:

- `/Users/vskorokhod/liteagent/liteagent/agent.py`

If it is about background work or autonomy:

- `/Users/vskorokhod/liteagent/liteagent/goals.py`
- `/Users/vskorokhod/liteagent/liteagent/night_coding.py`

If it is about dashboard rendering:

- `/Users/vskorokhod/liteagent/liteagent/static/dashboard.html`
- `/Users/vskorokhod/liteagent/liteagent/channels/dashboard.py`

If it is about memory quality:

- `/Users/vskorokhod/liteagent/liteagent/memory.py`

If it is about document/file understanding:

- `/Users/vskorokhod/liteagent/liteagent/file_types.py`
- `/Users/vskorokhod/liteagent/liteagent/multimodal.py`
- `/Users/vskorokhod/liteagent/liteagent/documents.py`

If it is about storage/file delivery:

- `/Users/vskorokhod/liteagent/liteagent/storage.py`
- `/Users/vskorokhod/liteagent/liteagent/agent.py`

## 13. Recommended Verification Workflow

When changing backend/runtime:

```bash
.venv/bin/python -m pytest tests/test_agent.py tests/test_dashboard.py tests/test_goals.py -q
```

When changing memory:

```bash
.venv/bin/python -m pytest tests/test_memory.py tests/test_agent.py -q
```

When changing Telegram:

```bash
.venv/bin/python -m pytest tests/test_telegram.py tests/test_telegram_files.py -q
```

When changing voice:

```bash
.venv/bin/python -m pytest tests/test_voice.py tests/test_dashboard.py -q
```

For live verification:

1. start API
2. open dashboard
3. check `/health`
4. check relevant `/api/...` endpoint directly
5. if UI-related, use Chrome DevTools MCP

## 14. Current Known Practical Follow-Ups

These are sensible next improvements, not necessarily bugs:

- auto-checkpoint git snapshots after verified night-coding cycles
- morning report delivery to Telegram
- optional report archival/export bundle per autonomous session
- stronger rollback strategy for night coding after bad cycles
- broader live browser acceptance coverage for long autonomous sessions
- optional offsite backup workflow using storage, not just attachment storage

## 15. Minimal Mental Model

If a future agent needs the shortest useful summary:

- `agent.py` is the brain
- `memory.py` is long-term context and thinking cloud
- `goals.py` is background autonomy
- `night_coding.py` is the overnight local coding mode
- `dashboard.py` + `dashboard.html` are the control plane
- `storage.py` + file/document modules handle persistent file workflows
- the system is now local-first, storage-connected, dashboard-heavy, and built around autonomous but guarded operation

