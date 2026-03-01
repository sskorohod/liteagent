# LiteAgent

Ultra-lightweight AI agent with persistent memory, multi-provider support, RAG pipeline, 8 metacognition features, and aggressive cost optimization.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-182%20passed-brightgreen.svg)]()

**~4,000 LOC core** | **182 tests** | **4 LLM providers** | **Zero bloat**

---

## Features

| Category | Details |
|----------|---------|
| **Multi-Provider** | Anthropic Claude, OpenAI GPT, Google Gemini, Ollama (local) |
| **4-Layer Memory** | Conversation (RAM), Scoped State (SQLite), Semantic Recall (embeddings), Auto-Learning (Haiku) |
| **RAG Pipeline** | Document ingestion, recursive chunking, cosine similarity search with keyword fallback |
| **8 Metacognition Features** | Dream Cycle, Self-Evolving Prompt, Proactive Agent, Auto Tool Synthesis, Confidence Gate, Style Adaptation, Skill Crystallization, Counterfactual Replay |
| **5 Cost Optimizations** | Cascade routing (Haiku/Sonnet/Opus), prompt caching, context compression, semantic tool loading, daily budget |
| **Multi-Channel** | CLI, REST API + Web Dashboard (SSE streaming), Telegram bot |
| **MCP Support** | Connect any MCP server via JSON config |
| **Multi-Agent Pool** | Delegate tasks between specialized agents |
| **Async Scheduler** | Cron-based background tasks (zero dependencies) |
| **Docker Ready** | Multi-stage build, compose profiles |

---

## Quick Start

### Install from source

```bash
git clone https://github.com/vskorokhod/liteagent.git
cd liteagent

# Option 1: Interactive installer
./install.sh

# Option 2: Manual install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[api,embeddings]"
```

### Install from PyPI

```bash
pip install liteagent          # minimal (Anthropic only)
pip install liteagent[all]     # everything
```

### Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or for other providers:
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

### Run

```bash
# Interactive CLI
liteagent

# One-shot mode
liteagent -1 "What is the capital of France?"

# Web Dashboard + API
liteagent --channel api
# Open http://localhost:8080

# Telegram bot
liteagent --channel telegram
```

---

## Multi-Provider Support

LiteAgent supports 4 LLM providers with a unified interface. Switch providers via `config.json`:

```json
{
  "agent": {
    "provider": "anthropic",
    "default_model": "claude-sonnet-4-20250514",
    "models": {
      "simple": "claude-haiku-4-5-20251001",
      "medium": "claude-sonnet-4-20250514",
      "complex": "claude-opus-4-20250115"
    }
  }
}
```

### OpenAI

```json
{
  "agent": { "provider": "openai", "default_model": "gpt-4o" },
  "providers": { "openai": { "api_key_env": "OPENAI_API_KEY" } }
}
```

### Ollama (local)

```json
{
  "agent": { "provider": "ollama", "default_model": "llama3.2" },
  "providers": { "ollama": { "base_url": "http://localhost:11434/v1" } }
}
```

### Gemini

```json
{
  "agent": { "provider": "gemini", "default_model": "gemini-2.5-flash" },
  "providers": { "gemini": { "api_key_env": "GOOGLE_API_KEY" } }
}
```

---

## RAG Pipeline

Ingest documents and query them with semantic or keyword search:

```bash
# In CLI
/ingest ~/Documents/project-docs
/documents

# Agent automatically uses rag_search tool when you ask about your documents
```

Enable in config:

```json
{
  "rag": {
    "enabled": true,
    "chunk_size": 500,
    "overlap": 50,
    "top_k": 5
  }
}
```

Supported formats: `.txt`, `.md`, `.html`, `.pdf`, `.py`, `.js`, `.json`, `.yaml`, `.rst`

---

## Configuration

Full `config.json` reference:

```json
{
  "agent": {
    "name": "LiteAgent",
    "soul": "soul.md",
    "max_iterations": 15,
    "provider": "anthropic",
    "default_model": "claude-sonnet-4-20250514",
    "models": {
      "simple": "claude-haiku-4-5-20251001",
      "medium": "claude-sonnet-4-20250514",
      "complex": "claude-opus-4-20250115"
    }
  },
  "memory": {
    "db_path": "~/.liteagent/memory.db",
    "max_history_tokens": 2000,
    "keep_recent_messages": 6,
    "auto_learn": true,
    "extraction_model": "claude-haiku-4-5-20251001"
  },
  "tools": {
    "builtin": ["read_file", "write_file", "exec_command", "web_search"],
    "mcp_servers": {}
  },
  "rag": {
    "enabled": false,
    "chunk_size": 500,
    "overlap": 50,
    "top_k": 5
  },
  "channels": {
    "cli": { "enabled": true },
    "telegram": { "enabled": false, "token_env": "TELEGRAM_BOT_TOKEN" },
    "api": { "enabled": false, "host": "0.0.0.0", "port": 8080 }
  },
  "cost": {
    "cascade_routing": true,
    "prompt_caching": true,
    "context_compression": true,
    "budget_daily_usd": 5.0,
    "track_usage": true
  },
  "features": {
    "dream_cycle":            { "enabled": false },
    "self_evolving_prompt":   { "enabled": false },
    "proactive_agent":        { "enabled": false },
    "auto_tool_synthesis":    { "enabled": false },
    "confidence_gate":        { "enabled": false, "threshold": 6 },
    "style_adaptation":       { "enabled": false },
    "skill_crystallization":  { "enabled": false },
    "counterfactual_replay":  { "enabled": false }
  }
}
```

---

## 8 Metacognition Features

All features are opt-in and have zero overhead when disabled.

| Feature | Description |
|---------|-------------|
| **Dream Cycle** | Off-hours memory consolidation: decay old memories, merge similar ones, extract insights |
| **Self-Evolving Prompt** | Detects user friction, synthesizes prompt patches, applies them with approval |
| **Proactive Agent** | Detects usage patterns and proactively offers relevant suggestions |
| **Auto Tool Synthesis** | Generates new tools from conversation context, sandboxed and approvable |
| **Confidence Gate** | Self-assesses response confidence, escalates to better models when uncertain |
| **Style Adaptation** | Adapts response style (formality, technicality, verbosity) to match user |
| **Skill Crystallization** | Detects repeated tool-call patterns and crystallizes them into reusable skills |
| **Counterfactual Replay** | Replays failed interactions to extract lessons for future improvement |

---

## Architecture

```
liteagent/
  agent.py          Core agent loop, cascade routing, context building
  providers.py      Multi-provider abstraction (Anthropic, OpenAI, Ollama, Gemini)
  memory.py         4-layer memory system (conversation, state, semantic, learning)
  rag.py            RAG pipeline (ingestion, chunking, retrieval)
  tools.py          Tool registry, schema generation, MCP support
  config.py         Config loader, validation, env vars
  metacognition.py  Confidence gate, counterfactual replay, dream cycle
  evolution.py      Self-evolving prompt, style adaptation, proactive agent
  synthesis.py      Auto tool synthesis, skill crystallization
  scheduler.py      Async cron scheduler (zero dependencies)
  pool.py           Multi-agent pool
  main.py           CLI entry point
  channels/
    cli.py          Interactive REPL
    api.py          FastAPI REST + SSE streaming
    dashboard.py    Web dashboard API routes
    telegram.py     Telegram bot channel
  static/
    dashboard.html  Web dashboard SPA
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/memories` | Show stored memories |
| `/usage` | Show token usage and costs |
| `/clear` | Clear conversation history |
| `/forget X` | Forget memories matching X |
| `/ingest X` | Ingest file/directory into RAG |
| `/documents` | List ingested documents |
| `/help` | Show all commands |

---

## Web Dashboard

Access at `http://localhost:8080` when running with `--channel api`.

Features:
- **Overview** — KPI cards, daily usage chart
- **Usage & Cost** — Model breakdown, doughnut charts, detailed table
- **Memories** — Browse, search, delete memories
- **Tools** — View all registered tools and parameters
- **Chat** — Real-time streaming chat with markdown rendering
- **Settings** — Read-only config viewer

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send message (JSON body: `{message, user_id, stream}`) |
| `GET` | `/chat/stream` | SSE streaming chat (`?message=...&user_id=...`) |
| `POST` | `/chat/multimodal` | Send message with image attachment |
| `POST` | `/command` | Execute slash command |
| `GET` | `/api/overview` | Dashboard KPI data |
| `GET` | `/api/usage` | Usage breakdown by model |
| `GET` | `/api/memories` | List all memories |
| `GET` | `/api/rag/documents` | List RAG documents |
| `POST` | `/api/rag/ingest` | Ingest file/directory |
| `GET` | `/health` | Health check |

---

## MCP Servers

Connect external tool servers via config:

```json
{
  "tools": {
    "mcp_servers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-filesystem", "/home/user"]
      },
      "github": {
        "command": "npx",
        "args": ["-y", "@anthropic/mcp-github"],
        "env": { "GITHUB_TOKEN": "ghp_..." }
      }
    }
  }
}
```

---

## Docker

```bash
# Build
docker build -t liteagent .

# Run CLI
docker run -it -e ANTHROPIC_API_KEY liteagent

# Run API
docker run -p 8080:8080 -e ANTHROPIC_API_KEY liteagent --channel api

# Docker Compose
docker compose --profile api up
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=liteagent --cov-report=term-missing

# Useful make targets
make test          # Run all tests
make run           # Start CLI
make api           # Start API server
make docker        # Build Docker image
```

---

## Optional Dependencies

| Extra | Package | Purpose |
|-------|---------|---------|
| `api` | FastAPI + Uvicorn | REST API + Dashboard |
| `telegram` | python-telegram-bot | Telegram bot channel |
| `embeddings` | sentence-transformers | Semantic memory + RAG |
| `openai` | openai SDK | OpenAI GPT provider |
| `gemini` | google-generativeai | Google Gemini provider |
| `ollama` | openai SDK | Ollama local models |
| `pdf` | pymupdf | PDF document support for RAG |
| `dev` | pytest + coverage | Development tools |
| `all` | Everything above | Full installation |

---

## License

MIT License. See [LICENSE](LICENSE).
