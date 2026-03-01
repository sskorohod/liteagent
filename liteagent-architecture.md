# LiteAgent — Архитектура и План Разработки

## Философия проекта

**LiteAgent** — ультра-лёгкий AI-агент (~1,500 строк Python), который берёт лучшее из 14+ фреймворков:

| Что берём | Откуда | Почему |
|---|---|---|
| Минимальный agent loop | Nanobot / Strands | ~800 строк ядра, zero bloat |
| Обучаемая память | Agno | Агент умнеет с каждой сессией |
| Scoped state (prefix convention) | Google ADK | Элегантное разделение session/user/app |
| Prompt caching + cascade routing | Anthropic API | 60-95% экономия токенов |
| Context compression | CoALA / AgentScope | 30-40% экономия на длинных диалогах |
| MCP для инструментов | Anthropic (universal) | 0 кастомных интеграций, 97M downloads |
| Decorator-based tools | Universal pattern | 3 строки на инструмент |
| Provider registry | Nanobot | 2 шага для нового провайдера |
| Channel adapters | Nanobot / OpenClaw | ~150 строк на канал |
| Config-driven | Nanobot | JSON конфиг, не код |

**Антипаттерны, которых избегаем:**
- ❌ Graph definitions (LangGraph) — overkill для single-agent
- ❌ Role/backstory система (CrewAI) — overhead для одного агента
- ❌ Framework dependencies — только anthropic SDK + каналы
- ❌ In-memory-only state — всё персистентно с SQLite

---

## Архитектура

### Высокоуровневая схема

```
                    ┌─────────────────────────────┐
                    │        Channels Layer         │
                    │  CLI │ Telegram │ API │ WS    │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │       Agent Core (~800 LOC)   │
                    │                               │
                    │  ┌─────────────────────────┐  │
                    │  │     Token Optimizer       │  │
                    │  │  • Cascade routing        │  │
                    │  │  • Prompt caching         │  │
                    │  │  • Context compression    │  │
                    │  │  • Semantic tool loading   │  │
                    │  └─────────────────────────┘  │
                    │                               │
                    │  ┌───────────┐ ┌───────────┐  │
                    │  │ Agent Loop│ │   Tools    │  │
                    │  │ (minimal) │ │ (MCP+local)│  │
                    │  └───────────┘ └───────────┘  │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │      Memory System            │
                    │                               │
                    │  L1: Conversation (in-memory)  │
                    │  L2: Scoped State (SQLite)     │
                    │  L3: Semantic Memory (SQLite    │
                    │      + embeddings)              │
                    │  L4: Knowledge Extractor        │
                    │      (auto-learns from chats)   │
                    └────────────────────────────────┘
```

### Файловая структура

```
liteagent/
├── main.py              # Точка входа, CLI (~50 LOC)
├── config.py            # Загрузка конфига (~60 LOC)
├── agent.py             # Agent loop + token optimizer (~400 LOC)
├── memory.py            # 4-layer memory system (~350 LOC)
├── tools.py             # Tool registry + MCP loader (~150 LOC)
├── channels/            # Channel adapters
│   ├── base.py          # Базовый класс (~40 LOC)
│   ├── cli.py           # CLI интерфейс (~60 LOC)
│   ├── telegram.py      # Telegram bot (~150 LOC)
│   └── api.py           # FastAPI endpoint (~100 LOC)
├── providers.py         # Provider registry (~80 LOC)
├── soul.md              # Идентичность агента
└── config.json          # Конфигурация
                         # ИТОГО: ~1,440 LOC
```

---

## Детальная архитектура каждого модуля

### 1. Agent Loop (`agent.py`) — Сердце системы

Минимальный цикл в стиле Strands — модель сама решает, что делать:

```python
class LiteAgent:
    """Ultra-lightweight agent with smart token management."""

    async def run(self, user_input: str, user_id: str) -> str:
        # 1. Собрать контекст (token-efficient)
        context = await self._build_context(user_input, user_id)

        # 2. Выбрать модель (cascade routing)
        model = self._select_model(user_input, context)

        # 3. Agentic loop
        messages = context["messages"]
        messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_iterations):
            response = await self.client.messages.create(
                model=model,
                max_tokens=4096,
                system=context["system_prompt"],
                tools=context["tools"],
                messages=messages,
                # PROMPT CACHING — ключевая оптимизация
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
            )

            if response.stop_reason == "tool_use":
                tool_results = await self.tools.execute(response.content)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                # Track token usage
                self.stats.add(response.usage)
            else:
                text = self._extract_text(response)
                # 4. Post-processing: extract knowledge, compress, save
                await self._post_process(user_input, text, user_id, messages)
                return text

    async def _build_context(self, user_input, user_id):
        """Собрать минимальный, но достаточный контекст."""
        # Базовый system prompt (кешируется — 95% экономии)
        system = await self._cached_system_prompt(user_id)

        # Релевантные воспоминания (не все — только нужные)
        memories = await self.memory.recall(user_input, user_id, top_k=5)

        # Сжатая история (не полная — compressed)
        history = await self.memory.get_compressed_history(user_id, max_tokens=2000)

        # Релевантные инструменты (не все — семантический отбор)
        tools = await self.tools.get_relevant(user_input, top_k=8)

        return {
            "system_prompt": system + "\n\n" + self._format_memories(memories),
            "messages": history,
            "tools": tools,
        }

    def _select_model(self, user_input, context):
        """Cascade routing — 14% лучше чем один model."""
        # Простые запросы → Haiku (быстро, дёшево)
        # Средние → Sonnet (баланс)
        # Сложные → Opus (максимум качества)
        complexity = self._estimate_complexity(user_input, context)
        return {
            "simple": "claude-haiku-4-5-20251001",
            "medium": "claude-sonnet-4-20250514",
            "complex": "claude-opus-4-20250115",
        }[complexity]

    def _estimate_complexity(self, user_input, context):
        """Быстрая эвристика без LLM-вызова."""
        score = 0
        # Длина запроса
        if len(user_input) > 500: score += 2
        elif len(user_input) > 100: score += 1
        # Количество инструментов в контексте
        if len(context["tools"]) > 5: score += 1
        # Ключевые слова сложности
        complex_markers = ["проанализируй", "сравни", "спланируй",
                          "analyze", "compare", "plan", "architect",
                          "напиши код", "write code", "debug"]
        if any(m in user_input.lower() for m in complex_markers): score += 2
        # Пороги
        if score >= 4: return "complex"
        if score >= 2: return "medium"
        return "simple"

    async def _post_process(self, user_input, response, user_id, messages):
        """Background: извлечь знания, сжать контекст, обновить память."""
        # Извлечь факты из диалога (async, не блокирует ответ)
        asyncio.create_task(
            self.memory.extract_and_learn(user_input, response, user_id)
        )
        # Обновить compressed history
        await self.memory.update_history(user_id, messages)
```

### 2. Memory System (`memory.py`) — 4 слоя

Это ключевое отличие от большинства фреймворков. Не просто "сохранить историю", а **умная память, которая учится**:

```python
class MemorySystem:
    """4-layer memory: conversation → scoped state → semantic → knowledge."""

    def __init__(self, db_path="~/.liteagent/memory.db"):
        self.db = sqlite3.connect(os.path.expanduser(db_path))
        self._init_tables()

    # ══════════════════════════════════════════
    # L1: CONVERSATION MEMORY (in-memory buffer)
    # ══════════════════════════════════════════
    # Текущий диалог. Живёт только в рамках сессии.
    # При переполнении → сжатие через L2.

    async def get_compressed_history(self, user_id, max_tokens=2000):
        """Вернуть сжатую историю, влезающую в max_tokens."""
        # 1. Взять последние N сообщений (свежие = точные)
        recent = self._get_recent_messages(user_id, count=6)
        recent_tokens = self._count_tokens(recent)

        if recent_tokens < max_tokens:
            # Есть место для summary старых сообщений
            remaining = max_tokens - recent_tokens
            summary = self._get_session_summary(user_id)
            if summary:
                return [{"role": "system",
                         "content": f"Краткое содержание ранее: {summary}"}
                       ] + recent
        return recent

    # ══════════════════════════════════════════
    # L2: SCOPED STATE (SQLite, persistent)
    # ══════════════════════════════════════════
    # Google ADK prefix convention:
    #   no prefix = session state
    #   user:     = cross-session per user
    #   app:      = global

    async def get_state(self, key: str, user_id: str = None):
        """Get scoped state value."""
        if key.startswith("user:") and user_id:
            return self.db.execute(
                "SELECT value FROM user_state WHERE user_id=? AND key=?",
                (user_id, key)).fetchone()
        elif key.startswith("app:"):
            return self.db.execute(
                "SELECT value FROM app_state WHERE key=?",
                (key,)).fetchone()
        else:  # session state
            return self._session_state.get(key)

    async def set_state(self, key: str, value, user_id: str = None):
        """Set scoped state value."""
        if key.startswith("user:"):
            self.db.execute(
                "INSERT OR REPLACE INTO user_state VALUES (?, ?, ?)",
                (user_id, key, json.dumps(value)))
        elif key.startswith("app:"):
            self.db.execute(
                "INSERT OR REPLACE INTO app_state VALUES (?, ?)",
                (key, json.dumps(value)))
        else:
            self._session_state[key] = value
        self.db.commit()

    # ══════════════════════════════════════════
    # L3: SEMANTIC MEMORY (SQLite + embeddings)
    # ══════════════════════════════════════════
    # Долгосрочные факты о пользователе, его предпочтениях,
    # проектах, контактах — всё, что агент "выучил".

    async def recall(self, query: str, user_id: str, top_k=5):
        """Найти релевантные воспоминания по семантическому сходству."""
        query_emb = await self._embed(query)
        rows = self.db.execute(
            """SELECT content, type, importance, embedding
               FROM memories WHERE user_id = ?
               ORDER BY importance DESC""",
            (user_id,)).fetchall()

        # Ранжировать по cosine similarity + importance
        scored = []
        for content, mtype, importance, emb_blob in rows:
            emb = self._deserialize_embedding(emb_blob)
            similarity = self._cosine_sim(query_emb, emb)
            score = similarity * 0.7 + importance * 0.3
            scored.append((content, mtype, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]

    async def remember(self, content: str, user_id: str,
                       memory_type: str = "fact", importance: float = 0.5):
        """Сохранить новое воспоминание."""
        # Дедупликация: проверить, нет ли похожего
        existing = await self.recall(content, user_id, top_k=1)
        if existing and existing[0][2] > 0.9:
            # Обновить существующее (merge)
            await self._merge_memory(existing[0], content, user_id)
            return

        embedding = await self._embed(content)
        self.db.execute(
            """INSERT INTO memories
               (user_id, content, type, importance, embedding, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (user_id, content, memory_type, importance,
             self._serialize_embedding(embedding),
             datetime.now().isoformat()))
        self.db.commit()

    # ══════════════════════════════════════════
    # L4: KNOWLEDGE EXTRACTOR (auto-learning)
    # ══════════════════════════════════════════
    # После каждого диалога — фоновая задача:
    # извлечь факты, обновить профиль, сжать сессию.

    async def extract_and_learn(self, user_input, agent_response, user_id):
        """Background: извлечь знания из диалога (cheap model)."""
        extraction_prompt = f"""Проанализируй этот обмен и извлеки ТОЛЬКО новые факты.

Пользователь: {user_input}
Ассистент: {agent_response}

Верни JSON:
{{
  "facts": ["факт1", "факт2"],
  "preferences": ["предпочтение1"],
  "project_updates": ["обновление1"],
  "session_summary_update": "краткое обновление"
}}

Если новых фактов нет — верни пустые массивы. Не выдумывай."""

        # Используем HAIKU — дёшево, быстро, фоново
        result = await self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": extraction_prompt}]
        )

        try:
            data = json.loads(result.content[0].text)
            for fact in data.get("facts", []):
                await self.remember(fact, user_id, "fact", 0.6)
            for pref in data.get("preferences", []):
                await self.remember(pref, user_id, "preference", 0.8)
            for update in data.get("project_updates", []):
                await self.remember(update, user_id, "project", 0.7)
            if data.get("session_summary_update"):
                await self._update_session_summary(
                    user_id, data["session_summary_update"])
        except json.JSONDecodeError:
            pass  # Haiku не всегда отвечает JSON — OK, skip

    # ══════════════════════════════════════════
    # EMBEDDINGS — лёгкие, без внешних зависимостей
    # ══════════════════════════════════════════

    async def _embed(self, text: str) -> list[float]:
        """Получить embedding. Варианты по приоритету:
        1. Anthropic Voyager (если доступен) — лучше качество
        2. Local sentence-transformers — бесплатно, быстро
        3. TF-IDF fallback — zero dependencies
        """
        if self.embedding_provider == "local":
            # all-MiniLM-L6-v2 — 22MB, быстрый, достаточный
            return self.local_model.encode(text).tolist()
        else:
            # TF-IDF fallback — работает без ML моделей
            return self._tfidf_embed(text)
```

### 3. Token Optimizer — Стратегии экономии

Встроен в agent.py, но логически отдельный слой:

```python
class TokenOptimizer:
    """Все стратегии экономии токенов в одном месте."""

    # ── 1. PROMPT CACHING (60-95% на system prompt) ──
    def build_cached_system(self, base_prompt, memories):
        """System prompt с cache_control — Anthropic кеширует до 5 мин."""
        return [
            {
                "type": "text",
                "text": base_prompt,  # Статичная часть — ВСЕГДА кешируется
                "cache_control": {"type": "ephemeral"}
            },
            {
                "type": "text",
                "text": f"\nРелевантные воспоминания:\n{memories}"
                # Динамическая часть — НЕ кешируется
            }
        ]

    # ── 2. CONTEXT COMPRESSION (30-40% на истории) ──
    async def compress_history(self, messages, keep_recent=6, max_tokens=2000):
        """Сжать старые сообщения, оставить свежие точными."""
        if self._count_tokens(messages) <= max_tokens:
            return messages

        recent = messages[-keep_recent:]
        old = messages[:-keep_recent]

        # Сжимаем через Haiku (дёшево)
        summary = await self._summarize_with_haiku(old)
        return [{"role": "system",
                 "content": f"[Сжатый контекст]: {summary}"}] + recent

    # ── 3. SEMANTIC TOOL LOADING (98%+ на tool definitions) ──
    async def select_tools(self, query, all_tools, top_k=8):
        """Загружать только релевантные инструменты."""
        if len(all_tools) <= top_k:
            return all_tools

        query_emb = await self.memory._embed(query)
        scored = []
        for tool in all_tools:
            tool_emb = self._tool_embeddings.get(tool["name"])
            if tool_emb:
                sim = cosine_sim(query_emb, tool_emb)
                scored.append((tool, sim))
            else:
                scored.append((tool, 0.5))  # Default

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:top_k]]

    # ── 4. CASCADE MODEL ROUTING (14% efficiency gain) ──
    # Реализовано в agent._select_model()

    # ── 5. RESPONSE STREAMING (perceived latency) ──
    async def stream_response(self, **kwargs):
        """Стриминг — пользователь видит ответ сразу."""
        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
```

### 4. Tools System (`tools.py`)

MCP-first + лёгкие локальные инструменты:

```python
class ToolRegistry:
    """MCP servers + local tools, unified interface."""

    def __init__(self, config):
        self.local_tools = {}
        self.mcp_tools = {}

    # ── Local tools via decorator ──
    def tool(self, name=None, description=None):
        """@registry.tool decorator — 3 строки на инструмент."""
        def decorator(func):
            tool_name = name or func.__name__
            schema = self._generate_schema(func)
            self.local_tools[tool_name] = {
                "definition": {
                    "name": tool_name,
                    "description": description or func.__doc__,
                    "input_schema": schema,
                },
                "handler": func,
            }
            return func
        return decorator

    # ── MCP tools from config ──
    async def load_mcp_servers(self, mcp_config):
        """Загрузить MCP серверы из конфига."""
        for name, server_config in mcp_config.items():
            process = await asyncio.create_subprocess_exec(
                server_config["command"], *server_config["args"],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            tools = await self._discover_tools(process)
            for tool in tools:
                self.mcp_tools[tool["name"]] = {
                    "definition": tool,
                    "process": process,
                }

    def get_all_definitions(self):
        """Все определения для LLM."""
        defs = [t["definition"] for t in self.local_tools.values()]
        defs += [t["definition"] for t in self.mcp_tools.values()]
        return defs

    async def execute(self, content_blocks):
        """Выполнить tool calls из ответа LLM."""
        results = []
        for block in content_blocks:
            if block.type != "tool_use":
                continue
            name = block.name
            if name in self.local_tools:
                result = await self.local_tools[name]["handler"](block.input)
            elif name in self.mcp_tools:
                result = await self._call_mcp(name, block.input)
            else:
                result = f"Unknown tool: {name}"
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result),
            })
        return results
```

### 5. Config (`config.json`)

Всё поведение через конфиг, не код:

```json
{
  "agent": {
    "name": "LiteAgent",
    "soul": "soul.md",
    "max_iterations": 15,
    "default_model": "claude-sonnet-4-20250514",
    "models": {
      "simple": "claude-haiku-4-5-20251001",
      "medium": "claude-sonnet-4-20250514",
      "complex": "claude-opus-4-20250115"
    }
  },
  "providers": {
    "anthropic": {
      "api_key_env": "ANTHROPIC_API_KEY"
    },
    "ollama": {
      "api_base": "http://localhost:11434",
      "model": "llama3.2:3b"
    }
  },
  "memory": {
    "db_path": "~/.liteagent/memory.db",
    "embedding_provider": "local",
    "max_history_tokens": 2000,
    "keep_recent_messages": 6,
    "extraction_model": "claude-haiku-4-5-20251001",
    "auto_learn": true
  },
  "tools": {
    "local": ["read_file", "write_file", "exec_command", "web_search"],
    "mcp_servers": {
      "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"]
      },
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace"]
      }
    }
  },
  "channels": {
    "cli": { "enabled": true },
    "telegram": { "enabled": false, "token_env": "TELEGRAM_BOT_TOKEN" },
    "api": { "enabled": true, "port": 8080 }
  },
  "cost": {
    "cascade_routing": true,
    "prompt_caching": true,
    "context_compression": true,
    "semantic_tool_loading": true,
    "budget_daily_usd": 5.0,
    "budget_alert_threshold": 0.8
  }
}
```

---

## План разработки

### Phase 1: Core (2-3 дня)
**Цель:** работающий агент с CLI, базовая память

```
[ ] config.py — загрузка config.json + env vars
[ ] providers.py — registry для Anthropic + Ollama
[ ] agent.py — минимальный agent loop (без оптимизаций)
[ ] tools.py — decorator-based local tools (read, write, exec, search)
[ ] memory.py — L1 (conversation) + L2 (scoped state с SQLite)
[ ] channels/cli.py — REPL интерфейс
[ ] main.py — точка входа
[ ] soul.md — базовый system prompt

Тест: `python main.py` → диалог через CLI с инструментами
Размер: ~700 LOC
```

### Phase 2: Smart Memory (2-3 дня)
**Цель:** агент, который учится и помнит

```
[ ] memory.py — L3: semantic memory (embeddings + SQLite)
[ ] memory.py — L4: knowledge extractor (Haiku background)
[ ] memory.py — context compression (summarize old messages)
[ ] Embedding: local all-MiniLM-L6-v2 (22MB) или TF-IDF fallback
[ ] recall() — семантический поиск по воспоминаниям
[ ] remember() — дедупликация + merge
[ ] extract_and_learn() — автоматическое извлечение фактов

Тест: 10 диалогов → агент помнит факты из 1-го в 10-м
Размер: +350 LOC = ~1,050 LOC
```

### Phase 3: Token Optimization (1-2 дня)
**Цель:** 60-90% экономия токенов

```
[ ] Prompt caching — cache_control на system prompt
[ ] Cascade routing — Haiku/Sonnet/Opus по сложности
[ ] Semantic tool loading — только релевантные tools
[ ] Context compression — сжатие старых сообщений
[ ] Cost tracker — подсчёт и логирование расхода
[ ] Budget alerts — предупреждение при 80% дневного бюджета

Тест: сравнить cost с/без оптимизаций на 50 запросов
Размер: +200 LOC = ~1,250 LOC
```

### Phase 4: MCP + Channels (2-3 дня)
**Цель:** внешние инструменты и мульти-канальность

```
[ ] tools.py — MCP server discovery + execution
[ ] channels/telegram.py — Telegram bot adapter
[ ] channels/api.py — FastAPI + WebSocket endpoint
[ ] AG-UI streaming events (optional)

Тест: агент работает через Telegram + API одновременно
Размер: +400 LOC = ~1,500 LOC (финал)
```

### Phase 5: Production Hardening (2-3 дня)
**Цель:** надёжность в проде

```
[ ] Error handling — retry с exponential backoff
[ ] Input sanitization — prompt injection defense
[ ] Rate limiting — per user
[ ] Logging — structured (JSON)
[ ] Health check endpoint
[ ] Docker + docker-compose.yml
[ ] README.md + примеры

Тест: 1000 запросов без падений
```

---

## Метрики успеха

| Метрика | Цель |
|---|---|
| Размер ядра | ≤ 1,500 LOC Python |
| Запуск | < 2 секунды |
| Память (RAM) | < 100 MB |
| Стоимость за запрос (среднее) | < $0.01 |
| Экономия vs наивный подход | 60-90% |
| Recall accuracy (память) | > 80% на 10 сессий |
| Зависимости | anthropic + sqlite3 (stdlib) + channel libs |
| Установка | `pip install -e .` + config.json |

---

## Сравнение с альтернативами

| | LiteAgent | LangGraph | CrewAI | Agno | Nanobot |
|---|---|---|---|---|---|
| LOC | ~1,500 | 50K+ | 30K+ | 20K+ | ~4K |
| Memory layers | 4 | 1 (checkpoint) | 2 (RAG+history) | 3 (learning) | 1 (file) |
| Token optimization | 5 стратегий | manual | none | none | none |
| Cascade routing | built-in | manual | none | none | none |
| MCP | yes | via adapter | yes | yes | yes |
| Dependencies | 2 | 15+ | 10+ | 8+ | 3 |
| Learning | auto | no | no | yes | no |
| Startup time | <2s | 5-10s | 3-5s | 2-3s | <2s |

---

## Технологический стек

| Компонент | Выбор | Почему |
|---|---|---|
| LLM API | Anthropic SDK | Prompt caching, streaming, tools |
| Database | SQLite (stdlib) | Zero dependencies, <1ms latency |
| Embeddings | all-MiniLM-L6-v2 / TF-IDF | 22MB local, no API calls |
| Web framework | FastAPI (optional) | Async, lightweight, standard |
| Telegram | python-telegram-bot | Mature, async |
| MCP | stdio transport | Universal, simple |
| Package | pyproject.toml | Modern Python packaging |
