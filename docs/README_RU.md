# LiteAgent

[English](../README.md) | **Русский**

Ультра-легкий AI-агент с персистентной памятью, мульти-провайдером, RAG-пайплайном, 8 метакогнитивными фичами и агрессивной оптимизацией стоимости.

**~4,000 LOC ядро** | **182 теста** | **4 LLM провайдера** | **Zero bloat**

---

## Возможности

- **4 LLM провайдера** — Anthropic Claude, OpenAI GPT, Google Gemini, Ollama (локальные модели)
- **4-уровневая память** — разговор (RAM), scoped state (SQLite), семантический recall (embeddings), авто-обучение (Haiku)
- **RAG-пайплайн** — загрузка документов, рекурсивное разбиение, поиск по косинусному сходству
- **8 метакогнитивных фич** — Dream Cycle, Self-Evolving Prompt, Proactive Agent, Auto Tool Synthesis, Confidence Gate, Style Adaptation, Skill Crystallization, Counterfactual Replay
- **5 оптимизаций стоимости** — каскадная маршрутизация (Haiku/Sonnet/Opus), кэширование промптов, сжатие контекста, семантическая загрузка инструментов, дневной бюджет
- **Мульти-канальность** — CLI, REST API + Web Dashboard (SSE streaming), Telegram бот
- **MCP поддержка** — подключение любых MCP серверов через конфиг
- **Пул агентов** — делегирование задач между специализированными агентами
- **Асинхронный планировщик** — cron-задачи без внешних зависимостей

---

## Быстрый старт

```bash
# Клонировать и установить
git clone https://github.com/vskorokhod/liteagent.git
cd liteagent
./install.sh

# Установить API ключ
export ANTHROPIC_API_KEY="sk-ant-..."

# Запустить
liteagent                          # интерактивный CLI
liteagent -1 "Привет!"            # одноразовый режим
liteagent --channel api            # веб-дашборд на :8080
```

---

## Переключение провайдера

```json
{
  "agent": { "provider": "openai", "default_model": "gpt-4o" },
  "providers": { "openai": { "api_key_env": "OPENAI_API_KEY" } }
}
```

```json
{
  "agent": { "provider": "ollama", "default_model": "llama3.2" },
  "providers": { "ollama": { "base_url": "http://localhost:11434/v1" } }
}
```

---

## RAG-пайплайн

```bash
# В CLI
/ingest ~/Documents/мои-документы
/documents

# Агент автоматически использует rag_search при вопросах о документах
```

Конфиг:
```json
{ "rag": { "enabled": true, "chunk_size": 500, "top_k": 5 } }
```

Поддерживаемые форматы: `.txt`, `.md`, `.html`, `.pdf`, `.py`, `.js`, `.json`, `.yaml`, `.rst`

---

## Команды CLI

| Команда | Описание |
|---------|----------|
| `/memories` | Показать сохраненные воспоминания |
| `/usage` | Показать расход токенов и стоимость |
| `/clear` | Очистить историю разговора |
| `/forget X` | Забыть воспоминания, содержащие X |
| `/ingest X` | Загрузить файл/директорию в RAG |
| `/documents` | Список загруженных документов |
| `/help` | Показать все команды |

---

## Тестирование

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v          # 182 теста
python -m pytest tests/ --cov=liteagent  # с покрытием
```

---

## Лицензия

MIT License. См. [LICENSE](../LICENSE).
