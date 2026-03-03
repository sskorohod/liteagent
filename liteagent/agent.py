"""Core agent loop with cascade routing, prompt caching, and context compression."""

import asyncio
import itertools
import json
import logging
import os
import random
import time
from datetime import datetime, timezone, timedelta
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

from .circuit_breaker import CircuitBreaker
from .config import get_soul_prompt
from .hooks import HookRegistry, HookContext
from .memory import MemorySystem
from .plugins import load_plugins
from .providers import create_provider, get_pricing, MODEL_PRICING, TextBlock, ToolUseBlock
from .skills import SkillRegistry
from .tools import ToolRegistry, register_builtin_tools



def _serialize_content(content: list) -> list[dict]:
    """Convert LLMResponse content blocks (TextBlock/ToolUseBlock) to dicts for API re-submission."""
    result = []
    for block in content:
        if isinstance(block, TextBlock):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            result.append({"type": "tool_use", "id": block.id,
                           "name": block.name, "input": block.input})
        elif isinstance(block, dict):
            result.append(block)
        else:
            # Fallback: try dataclass-like access
            d = {"type": getattr(block, "type", "text")}
            if hasattr(block, "text"):
                d["text"] = block.text
            if hasattr(block, "id"):
                d["id"] = block.id
            if hasattr(block, "name"):
                d["name"] = block.name
            if hasattr(block, "input"):
                d["input"] = block.input
            result.append(d)
    return result

COMPLEXITY_MARKERS_COMPLEX = {
    "проанализируй", "сравни", "спланируй", "архитектур", "рефактор",
    "analyze", "compare", "plan", "architect", "refactor", "debug complex",
    "напиши большой", "write a full", "design system", "evaluate",
}
COMPLEXITY_MARKERS_MEDIUM = {
    "напиши", "объясни", "помоги с", "создай", "сделай",
    "write", "explain", "help with", "create", "build", "implement",
    "fix", "исправь", "обнови", "update",
}


class LiteAgent:
    """Ultra-lightweight agent with smart token management."""

    # ── Concurrency control (class-level, shared) ──
    _user_locks: dict[str, asyncio.Lock] = {}
    _locks_guard: asyncio.Lock | None = None     # meta-lock for _user_locks dict
    _requests_lock: asyncio.Lock | None = None   # protects _active_requests
    _provider_lock: asyncio.Lock | None = None   # protects provider switching

    # ── In-flight request tracking (class-level, shared) ──
    _active_requests: dict = {}
    _queued_requests: dict = {}
    _request_counter = itertools.count(1)
    _queue_counter = itertools.count(1)

    # ── WebSocket hub reference (set by api.py at startup) ──
    _ws_hub = None

    # ── Cascade decision history (class-level, for dashboard) ──
    _cascade_history: list = []
    _CASCADE_HISTORY_MAX = 50

    @classmethod
    def _ensure_locks(cls):
        """Lazily create asyncio.Lock instances (needs running event loop)."""
        if cls._locks_guard is None:
            cls._locks_guard = asyncio.Lock()
            cls._requests_lock = asyncio.Lock()
            cls._provider_lock = asyncio.Lock()

    async def _get_user_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create per-user asyncio.Lock for request serialization."""
        LiteAgent._ensure_locks()
        async with LiteAgent._locks_guard:
            if user_id not in LiteAgent._user_locks:
                LiteAgent._user_locks[user_id] = asyncio.Lock()
            return LiteAgent._user_locks[user_id]

    def _track_queued(self, user_id: str) -> int:
        """Register a queued request (waiting for user lock)."""
        q_id = next(LiteAgent._queue_counter)
        LiteAgent._queued_requests[q_id] = {
            "id": q_id, "user_id": user_id,
            "queued_at": datetime.now(timezone.utc).isoformat(),
        }
        self._ws_broadcast("request_queued", {"user_id": user_id, "id": q_id})
        return q_id

    def _untrack_queued(self, q_id: int):
        """Remove a queued request (lock acquired or timed out)."""
        LiteAgent._queued_requests.pop(q_id, None)

    async def _track_request_start(self, user_id: str, input_preview: str, model: str,
                                    complexity_score: int = -1, cascade_tier: str = "") -> int:
        """Register an in-flight request. Returns request ID."""
        async with LiteAgent._requests_lock:
            req_id = next(LiteAgent._request_counter)
            LiteAgent._active_requests[req_id] = {
                "id": req_id,
                "user_id": user_id,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "input_preview": input_preview[:120],
                "status": "running",
                "complexity_score": complexity_score,
                "cascade_tier": cascade_tier,
            }
        self._ws_broadcast("request_started", LiteAgent._active_requests.get(req_id, {}))
        return req_id

    async def _track_request_end(self, req_id: int):
        """Remove a completed in-flight request."""
        async with LiteAgent._requests_lock:
            info = LiteAgent._active_requests.pop(req_id, None)
        if info:
            elapsed = 0.0
            try:
                started = datetime.fromisoformat(info["started_at"])
                elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            except Exception:
                pass
            self._ws_broadcast("request_done", {
                "id": req_id,
                "user_id": info.get("user_id"),
                "model": info.get("model"),
                "complexity_score": info.get("complexity_score"),
                "cascade_tier": info.get("cascade_tier"),
                "elapsed_sec": round(elapsed, 2),
            })

    @classmethod
    def get_active_requests(cls) -> list:
        """Return list of currently in-flight requests (for dashboard)."""
        return list(cls._active_requests.values())

    @classmethod
    def get_queued_requests(cls) -> list:
        """Return list of queued requests waiting for user lock."""
        return list(cls._queued_requests.values())

    def _ws_broadcast(self, event_type: str, data: dict):
        """Non-blocking broadcast to WebSocket hub (if connected)."""
        hub = LiteAgent._ws_hub
        if hub:
            try:
                asyncio.get_event_loop().call_soon(
                    lambda: asyncio.ensure_future(hub.broadcast(event_type, data)))
            except RuntimeError:
                pass  # no event loop (e.g., during tests)

    def __init__(self, config: dict):
        self.config = config
        agent_cfg = config.get("agent", {})
        cost_cfg = config.get("cost", {})

        self.provider = create_provider(config)
        self.max_iterations = agent_cfg.get("max_iterations", 15)
        self.default_model = agent_cfg.get("default_model", "claude-sonnet-4-20250514")
        self.models = agent_cfg.get("models", {
            "simple": "claude-haiku-4-5-20251001",
            "medium": "claude-sonnet-4-20250514",
            "complex": "claude-opus-4-20250115",
        })

        # Cost controls
        self.cascade_routing = cost_cfg.get("cascade_routing", True)
        self.prompt_caching = cost_cfg.get("prompt_caching", True)
        self.budget_daily = cost_cfg.get("budget_daily_usd", 5.0)

        # Memory
        self.memory = MemorySystem(config, provider=self.provider)

        # Tools (with security sandbox)
        self.tools = ToolRegistry()
        tools_cfg = config.get("tools", {})
        builtin = tools_cfg.get("builtin", ["read_file", "write_file", "exec_command",
                                            "download_file", "send_file_to_user"])
        sandbox_root = tools_cfg.get("sandbox_root")  # None = sensitive-path blocking only
        cmd_allowlist = set(tools_cfg["command_allowlist"]) if "command_allowlist" in tools_cfg else None
        allow_shell = tools_cfg.get("allow_shell", False)
        register_builtin_tools(
            self.tools, enabled=builtin + ["memory_search"],
            sandbox_root=sandbox_root,
            command_allowlist=cmd_allowlist,
            allow_shell=allow_shell,
        )

        # Wire memory_search to actual memory
        self._wire_memory_tool()

        # MCP servers (loaded lazily on first run)
        self._mcp_config = config.get("tools", {}).get("mcp_servers", {})
        self._mcp_loaded = False

        # Current user context (for tool closures)
        self._current_user_id: str = "default"

        # Voice message store (channel → agent tool pipeline)
        self._voice_store: dict[str, dict] = {}
        self._wire_voice_tool()
        self._wire_voice_config_tools()

        # Background task tracking (prevents "Task destroyed" warnings)
        self._background_tasks: set[asyncio.Task] = set()

        # Soul prompt (cached across calls)
        self._soul_prompt = get_soul_prompt(config)

        # Feature flags (metacognition, evolution, synthesis)
        self._features = config.get("features", {})

        # Load synthesized tools if enabled (with execution budgets)
        if self._features.get("auto_tool_synthesis", {}).get("enabled"):
            from .synthesis import (load_synthesized_tools, create_synthesize_meta_tool,
                                    DEFAULT_SYNTH_TIMEOUT_SEC, DEFAULT_SYNTH_MAX_OUTPUT_CHARS)
            synth_cfg = self._features["auto_tool_synthesis"]
            _synth_timeout = synth_cfg.get("timeout_sec", DEFAULT_SYNTH_TIMEOUT_SEC)
            _synth_max_out = synth_cfg.get("max_output_chars", DEFAULT_SYNTH_MAX_OUTPUT_CHARS)
            load_synthesized_tools(
                self.memory.db, self.tools,
                set(synth_cfg.get("import_whitelist", [])) or None,
                timeout_sec=_synth_timeout,
                max_output=_synth_max_out)
            create_synthesize_meta_tool(
                self.tools, self.memory.db, synth_cfg)

        # Storage backend (MinIO/S3) + File Manager
        self._storage = None
        self._file_manager = None
        if config.get("storage", {}).get("enabled", False):
            from .storage import create_storage
            self._storage = create_storage(config)
            if self._storage:
                from .file_manager import create_file_manager
                self._file_manager = create_file_manager(self)
                self._wire_storage_tools()

        # Knowledge base (separate from RAG, for books/reference materials)
        self._knowledge_base = None
        kb_cfg = config.get("knowledge_base", {})
        if kb_cfg.get("enabled", False):
            self._init_knowledge_base(kb_cfg)

        # RAG pipeline (optional, with Qdrant support + sandbox)
        self._rag = None
        rag_cfg = config.get("rag", {})
        if rag_cfg.get("enabled", False):
            from .rag import RAGPipeline
            self._rag = RAGPipeline(
                self.memory.db,
                embedder=self.memory._embedder,
                config=rag_cfg,
                sandbox_root=sandbox_root)
            self._rag.init_backend(config)
            self._wire_rag_tool()
            # Connect RAG to FileManager for full-content indexing
            if self._file_manager:
                self._file_manager._rag = self._rag

        # Task manager (set by main.py after scheduler setup)
        self._task_manager = None

        # Auto-ingestion config
        self._auto_ingestion = config.get("features", {}).get("auto_ingestion", {})
        if self._auto_ingestion.get("enabled", False):
            self._init_file_access_tracking()

        # Auto-prune old memories on startup
        mem_cfg = config.get("memory", {})
        if mem_cfg.get("auto_prune", False):
            self.memory.prune_old_memories(
                days=mem_cfg.get("prune_days", 90),
                min_importance=mem_cfg.get("prune_min_importance", 0.3))

        # Circuit breaker for provider resilience
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("providers", {}).get("circuit_breaker_threshold", 3),
            recovery_timeout=config.get("providers", {}).get("circuit_breaker_cooldown", 300.0),
        )

        # Hook system (lifecycle events with priority ordering)
        self.hooks = HookRegistry()
        self._register_builtin_hooks()
        # Load user plugins from ~/.liteagent/plugins/
        self._loaded_plugins = load_plugins(self.hooks, config)

        # Web tools (fetch, search, crawl, extract) — enabled by default
        self._web_cache = None
        web_cfg = config.get("web", {})
        if web_cfg.get("enabled", True):
            self._wire_web_tools()

        # Skill system (modular prompt injection with progressive disclosure)
        self.skill_registry = SkillRegistry()
        self.skill_registry.load_all(config)

    def _register_builtin_hooks(self):
        """Register built-in metacognition features as hook handlers."""
        agent = self

        # Confidence Gate (priority 50 — runs early to potentially escalate)
        cg_cfg = self._features.get("confidence_gate", {})
        if cg_cfg.get("enabled"):
            async def confidence_gate_handler(ctx: HookContext):
                try:
                    from .metacognition import assess_confidence
                    assessment = await assess_confidence(
                        agent.provider, ctx.extra.get("user_input_text", ""),
                        ctx.response_text,
                        agent.config.get("memory", {}))
                    confidence = assessment.get("confidence", 10)
                    ctx.extra["confidence"] = confidence
                    threshold = cg_cfg.get("threshold", 6)
                    if confidence < threshold:
                        action = assessment.get("action", "admit")
                        if action == "escalate" and cg_cfg.get("escalate_to_model", True):
                            better = agent.models.get("complex", agent.default_model)
                            if better != ctx.model:
                                ctx.response_text = await agent._escalated_run(
                                    better, ctx.system_prompt, ctx.tool_defs, ctx.messages)
                        elif action == "admit":
                            ctx.response_text += (
                                "\n\n\u26a0\ufe0f I'm not fully confident in this "
                                "answer. Please verify independently.")
                except Exception as e:
                    logger.debug("Confidence gate error: %s", e)

            self.hooks.register("after_response", "confidence_gate",
                                confidence_gate_handler, priority=50)

        # Style Adaptation (priority 100)
        if self._features.get("style_adaptation", {}).get("enabled"):
            async def style_adaptation_handler(ctx: HookContext):
                try:
                    from .evolution import analyze_style, update_style_profile
                    style = analyze_style(ctx.extra.get("user_input_text", ""))
                    update_style_profile(
                        agent.memory.db, ctx.user_id, style,
                        agent._features["style_adaptation"].get("ema_alpha", 0.3))
                except Exception as e:
                    logger.debug("Style adaptation error: %s", e)

            self.hooks.register("after_response", "style_adaptation",
                                style_adaptation_handler, priority=100)

        # Skill Crystallization (priority 150)
        sk_cfg = self._features.get("skill_crystallization", {})
        if sk_cfg.get("enabled"):
            async def skill_crystallization_handler(ctx: HookContext):
                min_calls = sk_cfg.get("min_tool_calls", 3)
                if len(ctx.tool_calls_log) >= min_calls:
                    try:
                        from .synthesis import detect_skill, store_skill
                        skill = detect_skill(
                            ctx.tool_calls_log,
                            ctx.extra.get("user_input_text", ""), min_calls)
                        if skill:
                            store_skill(agent.memory.db, skill, ctx.user_id)
                    except Exception as e:
                        logger.debug("Skill crystallization error: %s", e)

            self.hooks.register("after_response", "skill_crystallization",
                                skill_crystallization_handler, priority=150)

        # Interaction Logging (priority 200 — always runs)
        async def interaction_log_handler(ctx: HookContext):
            try:
                from .metacognition import log_interaction
                _has_tool_errors = any(
                    tc.get("error") for tc in ctx.tool_calls_log
                ) if ctx.tool_calls_log else False
                _is_error = (ctx.response_text.startswith("\u26a0\ufe0f")
                             or ctx.response_text.startswith("\u274c")
                             or "error" in ctx.response_text[:50].lower())
                _success = 0 if (_has_tool_errors or _is_error) else 1
                log_interaction(
                    agent.memory.db, ctx.user_id,
                    ctx.extra.get("user_input_text", ""),
                    ctx.response_text,
                    ctx.tool_calls_log, _success,
                    ctx.extra.get("confidence"), ctx.model)
            except Exception as e:
                logger.debug("Interaction logging error: %s", e)

        self.hooks.register("after_response", "interaction_log",
                            interaction_log_handler, priority=200)

        # Auto Tool Synthesis: cross-session pattern detection (priority 250)
        ats_cfg = self._features.get("auto_tool_synthesis", {})
        if ats_cfg.get("enabled") and ats_cfg.get("cross_session_detection", False):
            async def tool_synthesis_handler(ctx: HookContext):
                try:
                    from .synthesis import (detect_repeated_patterns,
                                             propose_tool_from_pattern,
                                             register_synthesized_tool)
                    patterns = detect_repeated_patterns(
                        agent.memory.db, ctx.user_id,
                        min_occurrences=ats_cfg.get("min_pattern_occurrences", 3),
                        lookback_days=ats_cfg.get("pattern_lookback_days", 30))
                    for pattern in patterns[:1]:
                        proposal = await propose_tool_from_pattern(
                            agent.provider, pattern, ats_cfg)
                        if proposal:
                            approved = 1 if ats_cfg.get("auto_approve", False) else 0
                            agent.memory.db.execute(
                                """INSERT OR IGNORE INTO synthesized_tools
                                   (name, description, source_code, parameters_json,
                                    approved, created_at) VALUES (?,?,?,?,?,?)""",
                                (proposal["name"], proposal.get("description", ""),
                                 proposal["source_code"],
                                 proposal.get("parameters_json", "{}"),
                                 approved, datetime.now().isoformat()))
                            agent.memory.db.commit()
                            if approved:
                                register_synthesized_tool(
                                    agent.tools, proposal["name"],
                                    proposal["source_code"],
                                    proposal.get("description", ""),
                                    json.loads(proposal.get("parameters_json", "{}")))
                except Exception as e:
                    logger.debug("Cross-session synthesis error: %s", e)

            self.hooks.register("after_response", "auto_tool_synthesis",
                                tool_synthesis_handler, priority=250)

    def _wire_memory_tool(self):
        """Connect memory_search tool to actual memory system."""
        memory = self.memory
        agent = self

        async def memory_search_handler(query: str) -> str:
            results = memory.recall(query, user_id=agent._current_user_id, top_k=5)
            if not results:
                return "No relevant memories found."
            lines = []
            for m in results:
                lines.append(f"- [{m['type']}] {m['content']} (relevance: {m['score']:.2f})")
            return "\n".join(lines)

        self.tools._handlers["memory_search"] = memory_search_handler

    def _wire_rag_tool(self):
        """Register rag_search tool if RAG pipeline is enabled."""
        rag = self._rag

        async def rag_search_handler(query: str, top_k: int = 5) -> str:
            """Search ingested documents for relevant content."""
            results = rag.search(query, top_k=top_k)
            if not results:
                return "No relevant documents found."
            lines = []
            for r in results:
                lines.append(f"[{r['source']}] (score: {r['score']}) {r['content'][:500]}")
            body = "\n---\n".join(lines)
            return f"<rag_context>\n{body}\n</rag_context>"

        # Register tool with schema
        self.tools._tools["rag_search"] = {
            "name": "rag_search",
            "description": "Search ingested documents (RAG) for relevant content. "
                           "Use this when the user asks about their documents or files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {"type": "integer", "description": "Number of results (default 5)"},
                },
                "required": ["query"],
            },
        }
        self.tools._handlers["rag_search"] = rag_search_handler

    # ═══════════════════════════════════════════════════════════
    # KNOWLEDGE BASE
    # ═══════════════════════════════════════════════════════════

    def _init_knowledge_base(self, kb_cfg: dict):
        """Initialize knowledge base (separate from RAG)."""
        try:
            from .knowledge_base import KnowledgeBase
            self._knowledge_base = KnowledgeBase(
                config=kb_cfg,
                embedder=self.memory._embedder,
                provider=self.provider,
            )
            self._wire_knowledge_base_tools()
            logger.info("Knowledge base initialized")
        except Exception as e:
            logger.warning("Knowledge base init failed: %s", e)
            self._knowledge_base = None

    def _wire_knowledge_base_tools(self):
        """Register 6 KB tools: kb_search, kb_ingest, kb_list, kb_delete, kb_stats, kb_entities."""
        kb = self._knowledge_base

        async def kb_search_handler(query: str, top_k: int = 6,
                                     mode: str = "hybrid") -> str:
            results = await kb.search(query, top_k=top_k, mode=mode)
            if not results:
                return "В базе знаний релевантной информации не найдено."
            context = kb.build_context(results)
            return f"<kb_context>\n{context}\n</kb_context>"

        async def kb_ingest_handler(path: str) -> str:
            result = await kb.ingest(path)
            return json.dumps(result, ensure_ascii=False)

        async def kb_list_handler() -> str:
            docs = await kb.list_documents()
            if not docs:
                return "База знаний пуста. Загрузите документы с помощью kb_ingest."
            lines = []
            for d in docs:
                lines.append(
                    f"- {d['name']} (id: {d['id'][:8]}..., "
                    f"{d['chunk_count']} чанков, {d['page_count']} стр.)")
            return "\n".join(lines)

        async def kb_delete_handler(doc_id: str) -> str:
            ok = await kb.delete_document(doc_id)
            if ok:
                return f"Документ удалён: {doc_id}"
            return f"Документ не найден: {doc_id}"

        async def kb_stats_handler() -> str:
            stats = await kb.get_stats()
            return json.dumps(stats, ensure_ascii=False, indent=2)

        async def kb_entities_handler(doc_id: str = "") -> str:
            entities = await kb.list_entities(doc_id=doc_id if doc_id else None)
            if not entities:
                return "Сущности не найдены. Запустите ночной обработчик для извлечения сущностей."
            lines = []
            for e in entities:
                lines.append(f"- {e['name']} ({e['entity_type']}) — {e.get('doc_name', '?')}, x{e['count']}")
            return "\n".join(lines)

        # Register tools
        tools_defs = [
            {
                "name": "kb_search",
                "description": (
                    "Search the knowledge base (books, reference materials). "
                    "Returns relevant excerpts with citations (source, page, section). "
                    "Use for accounting, law, regulations, and domain questions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Search query"},
                        "top_k": {"type": "integer",
                                  "description": "Number of results (default 6)"},
                        "mode": {"type": "string",
                                 "description": "Search mode: hybrid, bm25, vector (default hybrid)"},
                    },
                    "required": ["query"],
                },
                "_handler": kb_search_handler,
            },
            {
                "name": "kb_ingest",
                "description": (
                    "Load a document (PDF, TXT, MD, HTML) into the knowledge base. "
                    "Parses structure, creates semantic chunks, indexes for search."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string",
                                 "description": "Path to the file to ingest"},
                    },
                    "required": ["path"],
                },
                "_handler": kb_ingest_handler,
            },
            {
                "name": "kb_list",
                "description": "List all documents in the knowledge base.",
                "input_schema": {"type": "object", "properties": {}},
                "_handler": kb_list_handler,
            },
            {
                "name": "kb_delete",
                "description": "Delete a document from the knowledge base by ID or name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string",
                                   "description": "Document ID or name to delete"},
                    },
                    "required": ["doc_id"],
                },
                "_handler": kb_delete_handler,
            },
            {
                "name": "kb_stats",
                "description": "Get knowledge base statistics (documents, chunks, search mode, storage size).",
                "input_schema": {"type": "object", "properties": {}},
                "_handler": kb_stats_handler,
            },
            {
                "name": "kb_entities",
                "description": "List extracted entities (people, laws, dates, terms) from the knowledge base.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "doc_id": {
                            "type": "string",
                            "description": "Optional document ID to filter entities"
                        }
                    }
                },
                "_handler": kb_entities_handler,
            },
        ]

        for td in tools_defs:
            handler = td.pop("_handler")
            self.tools._tools[td["name"]] = td
            self.tools._handlers[td["name"]] = handler

    def _wire_web_tools(self):
        """Register web tools: web_fetch, web_search, web_crawl, web_extract."""
        agent = self
        web_cfg = self.config.get("web", {})
        cache_cfg = web_cfg.get("cache", {})

        if cache_cfg.get("enabled", True):
            from .web import WebCache
            self._web_cache = WebCache(
                default_ttl=cache_cfg.get("ttl", 300),
                max_entries=cache_cfg.get("max_entries", 200))

        async def web_fetch_handler(url: str, max_length: int = 10000) -> str:
            from .web import web_fetch
            result = await web_fetch(url, config=agent.config.get("web", {}),
                                     cache=agent._web_cache)
            if result.error:
                return f"Error fetching {url}: {result.error}"
            content = result.content[:max_length]
            truncated = " (truncated)" if len(result.content) > max_length else ""
            footer = (f"\n---\nSource: {result.url} | "
                      f"Extractor: {result.extractor} | "
                      f"{result.extracted_length} chars{truncated}"
                      + (" | cached" if result.cached else ""))
            title = f"# {result.title}\n\n" if result.title else ""
            return f"{title}{content}{footer}"

        async def web_search_handler(query: str, count: int = 5,
                                     language: str = "", freshness: str = "") -> str:
            from .web import web_search
            resp = await web_search(query, config=agent.config.get("web", {}),
                                    cache=agent._web_cache,
                                    count=min(count, 20),
                                    language=language, freshness=freshness)
            if resp.error:
                return f"Web search error: {resp.error}"
            if not resp.results:
                return "No results found."
            lines = [f'Search results for "{query}" (via {resp.provider}, '
                     f'{len(resp.results)} results):\n']
            for i, r in enumerate(resp.results, 1):
                lines.append(f"{i}. **{r.title}**\n   {r.snippet}\n   URL: {r.url}")
            footer = (f"\n---\nProvider: {resp.provider}"
                      + (" | Cached" if resp.cached else ""))
            return "\n\n".join(lines) + footer

        async def web_crawl_handler(url: str, max_depth: int = 1,
                                    max_pages: int = 5) -> str:
            from .web import web_crawl
            from urllib.parse import urlparse as _urlparse
            results = await web_crawl(url, config=agent.config.get("web", {}),
                                      cache=agent._web_cache,
                                      max_depth=min(max_depth, 3),
                                      max_pages=min(max_pages, 20))
            if not results:
                return f"Crawl returned no pages for {url}"
            lines = [f"Crawled {len(results)} pages from "
                     f"{_urlparse(url).netloc}:\n"]
            total_chars = 0
            for r in results:
                if r.error:
                    lines.append(f"## Error: {r.url}\n{r.error}")
                    continue
                excerpt = r.content[:2000]
                trunc = "..." if len(r.content) > 2000 else ""
                lines.append(f"## {r.title or r.url} (depth: {r.depth})\n"
                             f"{excerpt}{trunc}")
                total_chars += len(r.content)
            lines.append(f"\n---\nPages: {len(results)} | "
                         f"Total content: {total_chars} chars")
            return "\n\n".join(lines)

        async def web_extract_handler(url: str, selector: str = "",
                                      extract: str = "") -> str:
            from .web import web_extract
            result = await web_extract(url, config=agent.config.get("web", {}),
                                       selectors={"css": selector} if selector else None)
            if result.error:
                return f"Error extracting from {url}: {result.error}"
            return json.dumps({
                "url": result.url,
                "title": result.title,
                "description": result.description,
                "og_tags": result.og_tags,
                "headings": result.headings[:50],
                "links": result.links[:50],
                "images": result.images[:30],
                "tables": result.tables[:10],
            }, ensure_ascii=False, indent=2)

        tools_defs = [
            {
                "name": "web_fetch",
                "description": (
                    "Fetch a web page and extract its readable content as clean text. "
                    "Use this to read articles, documentation, blog posts, or any web page. "
                    "Returns cleaned, readable content with the page title."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string",
                                "description": "URL to fetch (http/https)"},
                        "max_length": {"type": "integer",
                                       "description": "Max characters to return (default 10000)"},
                    },
                    "required": ["url"],
                },
                "_handler": web_fetch_handler,
            },
            {
                "name": "web_search",
                "description": (
                    "Search the web for current information, facts, and research. "
                    "Returns top results with titles, descriptions, and URLs. "
                    "Supports multiple search providers with automatic fallback."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string",
                                  "description": "Search query"},
                        "count": {"type": "integer",
                                  "description": "Number of results (1-20, default 5)"},
                        "language": {"type": "string",
                                     "description": "Language code (e.g. 'en', 'ru', 'de')"},
                        "freshness": {"type": "string",
                                      "description": "Time filter: 'day', 'week', 'month', 'year'"},
                    },
                    "required": ["query"],
                },
                "_handler": web_search_handler,
            },
            {
                "name": "web_crawl",
                "description": (
                    "Crawl multiple pages from a website. Follows internal links "
                    "up to a specified depth. Respects robots.txt and rate limits. "
                    "Use for gathering content from documentation sites or multi-page articles."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string",
                                "description": "Starting URL to crawl"},
                        "max_depth": {"type": "integer",
                                      "description": "Max link depth (default 1, max 3)"},
                        "max_pages": {"type": "integer",
                                      "description": "Max pages to crawl (default 5, max 20)"},
                    },
                    "required": ["url"],
                },
                "_handler": web_crawl_handler,
            },
            {
                "name": "web_extract",
                "description": (
                    "Extract structured data from a web page: title, description, "
                    "metadata (OG tags), headings, links, images, and tables. "
                    "Optionally use CSS selectors to target specific elements."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string",
                                "description": "URL to extract from"},
                        "selector": {"type": "string",
                                     "description": "CSS selector to narrow extraction (optional)"},
                        "extract": {"type": "string",
                                    "description": "What to extract: links,images,headings,tables,metadata (default: all)"},
                    },
                    "required": ["url"],
                },
                "_handler": web_extract_handler,
            },
        ]

        for td in tools_defs:
            handler = td.pop("_handler")
            self.tools._tools[td["name"]] = td
            self.tools._handlers[td["name"]] = handler
        logger.info("Web tools registered: web_fetch, web_search, web_crawl, web_extract")

    def store_voice(self, voice_id: str, audio_bytes: bytes, config: dict | None = None):
        """Store voice audio bytes for transcription via agent tool.

        Called by channel adapters (e.g. Telegram) before passing
        the voice message to agent.run().
        """
        self._voice_store[voice_id] = {
            "audio_bytes": audio_bytes,
            "config": config or {},
        }

    def _wire_voice_tool(self):
        """Register transcribe_voice tool — delegates to voice.py multi-provider STT."""
        agent = self

        async def transcribe_voice_handler(voice_id: str) -> str:
            """Transcribe a voice message by its ID using configured STT provider."""
            voice_data = agent._voice_store.pop(voice_id, None)
            if not voice_data:
                return f"Voice message '{voice_id}' not found or already transcribed."

            audio_bytes = voice_data["audio_bytes"]
            logger.info("STT: transcribing %s (%d bytes)", voice_id, len(audio_bytes))

            from .voice import transcribe
            result = await transcribe(audio_bytes, agent.config)

            if result.success:
                logger.info("STT: transcribed via %s (%s): %d chars",
                            result.provider, result.model, len(result.text))
                return result.text

            logger.warning("STT: builtin failed for %s: %s", voice_id, result.error)

            # Fallback to MCP transcription tools (e.g. mywhisper)
            mcp_transcribe = [
                n for n in agent.tools._handlers
                if "transcribe" in n and "__" in n
            ]
            if mcp_transcribe:
                import tempfile, os
                tmp_path = os.path.join(tempfile.gettempdir(), f"{voice_id}.ogg")
                if not os.path.exists(tmp_path):
                    with open(tmp_path, "wb") as f:
                        f.write(audio_bytes)

                for mcp_name in mcp_transcribe:
                    try:
                        logger.info("STT: trying MCP fallback %s for %s", mcp_name, voice_id)
                        mcp_handler = agent.tools._handlers[mcp_name]
                        if asyncio.iscoroutinefunction(mcp_handler):
                            mcp_result = await mcp_handler(path=tmp_path)
                        else:
                            mcp_result = mcp_handler(path=tmp_path)
                        if mcp_result and not mcp_result.startswith("Error"):
                            logger.info("STT: transcribed via MCP %s: %d chars",
                                        mcp_name, len(mcp_result))
                            return mcp_result
                    except Exception as e:
                        logger.warning("STT: MCP %s failed: %s", mcp_name, e)

            return f"Voice transcription error: {result.error}"

        self.tools._tools["transcribe_voice"] = {
            "name": "transcribe_voice",
            "description": (
                "Transcribe a voice message from the user. When a user sends a voice "
                "message (e.g. via Telegram), the audio is stored with a voice_id. "
                "Call this tool with that voice_id to get the text transcription. "
                "You MUST call this tool to understand what the user said in their "
                "voice message before you can respond. "
                "IMPORTANT: Do NOT show or repeat the transcription to the user — "
                "just respond to their message as if they typed it."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "voice_id": {
                        "type": "string",
                        "description": "The voice message identifier from the user's message",
                    },
                },
                "required": ["voice_id"],
            },
        }
        self.tools._handlers["transcribe_voice"] = transcribe_voice_handler

    def _wire_voice_config_tools(self):
        """Register voice configuration tools — let the agent self-configure TTS/STT."""
        agent = self

        def get_voice_settings_handler() -> str:
            """Get current voice (TTS/STT) settings, provider status, and available presets."""
            from .voice import (resolve_voice_config, TTS_PROVIDERS, STT_PROVIDERS,
                                TTS_COST_INFO, STT_COST_INFO, BUILTIN_PRESETS)
            from .config import get_api_key

            cfg = resolve_voice_config(agent.config)

            # Provider availability
            providers = {}
            for p in TTS_PROVIDERS:
                configured = True
                if p == "openai":
                    configured = bool(get_api_key("openai"))
                elif p == "elevenlabs":
                    configured = bool(get_api_key("elevenlabs") or os.environ.get("ELEVENLABS_API_KEY"))
                providers[p] = {
                    "configured": configured,
                    "cost": TTS_COST_INFO.get(p, "unknown"),
                }

            # Presets
            custom_presets = list(agent.config.get("voice", {}).get("presets", {}).keys())
            all_presets = list(BUILTIN_PRESETS.keys()) + custom_presets

            result = {
                "tts": {
                    "auto": cfg["tts"]["auto"],
                    "provider": cfg["tts"]["provider"],
                    "voice": cfg["tts"].get(cfg["tts"]["provider"], {}).get("voice",
                             cfg["tts"].get("openai", {}).get("voice", "alloy")),
                    "model": cfg["tts"].get("openai", {}).get("model", "tts-1"),
                    "max_length": cfg["tts"]["max_length"],
                },
                "stt": {
                    "provider": cfg["stt"]["provider"],
                    "model": cfg["stt"].get(cfg["stt"]["provider"], {}).get("model", "whisper-1"),
                    "language": cfg["stt"].get(cfg["stt"]["provider"], {}).get("language"),
                },
                "providers": providers,
                "presets": all_presets,
            }
            return json.dumps(result, indent=2, ensure_ascii=False)

        def set_voice_settings_handler(
            tts_auto: str = "",
            tts_provider: str = "",
            tts_voice: str = "",
            tts_model: str = "",
            tts_speed: float = 0,
            tts_max_length: int = 0,
            elevenlabs_voice_id: str = "",
            elevenlabs_stability: float = -1,
            elevenlabs_similarity_boost: float = -1,
            stt_provider: str = "",
            stt_model: str = "",
            stt_language: str = "",
        ) -> str:
            """Update voice settings. All parameters are optional — only provided ones are changed.

            tts_auto: Auto-TTS mode (off, always, inbound, tagged)
            tts_provider: TTS provider (openai, elevenlabs, edge)
            tts_voice: Voice name (openai: alloy/nova/etc, edge: ru-RU-SvetlanaNeural/etc)
            tts_model: TTS model (tts-1, tts-1-hd, gpt-4o-mini-tts)
            tts_speed: Speech speed (0.25-4.0)
            tts_max_length: Max text length for TTS
            elevenlabs_voice_id: ElevenLabs voice ID
            elevenlabs_stability: ElevenLabs stability (0-1)
            elevenlabs_similarity_boost: ElevenLabs similarity boost (0-1)
            stt_provider: STT provider (openai, deepgram, groq)
            stt_model: STT model name
            stt_language: STT language code
            """
            from .voice import TTS_PROVIDERS, STT_PROVIDERS, OPENAI_TTS_MODELS, OPENAI_TTS_VOICES
            from .config import save_config, get_api_key

            voice = agent.config.setdefault("voice", {})
            tts = voice.setdefault("tts", {})
            stt = voice.setdefault("stt", {})
            changes = []
            warnings = []

            # TTS auto mode
            if tts_auto:
                valid_modes = ("off", "always", "inbound", "tagged")
                if tts_auto not in valid_modes:
                    return json.dumps({"error": f"Invalid tts_auto: '{tts_auto}'. Valid: {valid_modes}"})
                tts["auto"] = tts_auto
                changes.append(f"auto={tts_auto}")

            # TTS provider
            if tts_provider:
                if tts_provider not in TTS_PROVIDERS:
                    return json.dumps({"error": f"Invalid tts_provider: '{tts_provider}'. Valid: {list(TTS_PROVIDERS)}"})
                if tts_provider == "openai" and not get_api_key("openai"):
                    warnings.append("OpenAI API key not configured — TTS may fail")
                if tts_provider == "elevenlabs" and not (get_api_key("elevenlabs") or os.environ.get("ELEVENLABS_API_KEY")):
                    warnings.append("ElevenLabs API key not configured — TTS may fail")
                tts["provider"] = tts_provider
                changes.append(f"provider={tts_provider}")

            # TTS voice
            if tts_voice:
                provider = tts_provider or tts.get("provider", "openai")
                if provider == "openai":
                    tts.setdefault("openai", {})["voice"] = tts_voice
                elif provider == "edge":
                    tts.setdefault("edge", {})["voice"] = tts_voice
                changes.append(f"voice={tts_voice}")

            # TTS model
            if tts_model:
                if tts_model not in OPENAI_TTS_MODELS:
                    warnings.append(f"Unknown model '{tts_model}', setting anyway")
                tts.setdefault("openai", {})["model"] = tts_model
                changes.append(f"model={tts_model}")

            # TTS speed
            if tts_speed > 0:
                tts.setdefault("openai", {})["speed"] = max(0.25, min(4.0, tts_speed))
                changes.append(f"speed={tts_speed}")

            # TTS max length
            if tts_max_length > 0:
                tts["max_length"] = tts_max_length
                changes.append(f"max_length={tts_max_length}")

            # ElevenLabs settings
            if elevenlabs_voice_id:
                tts.setdefault("elevenlabs", {})["voice_id"] = elevenlabs_voice_id
                changes.append(f"elevenlabs_voice_id={elevenlabs_voice_id}")
            if elevenlabs_stability >= 0:
                tts.setdefault("elevenlabs", {})["stability"] = max(0, min(1, elevenlabs_stability))
                changes.append(f"elevenlabs_stability={elevenlabs_stability}")
            if elevenlabs_similarity_boost >= 0:
                tts.setdefault("elevenlabs", {})["similarity_boost"] = max(0, min(1, elevenlabs_similarity_boost))
                changes.append(f"elevenlabs_similarity_boost={elevenlabs_similarity_boost}")

            # STT provider
            if stt_provider:
                if stt_provider not in STT_PROVIDERS:
                    return json.dumps({"error": f"Invalid stt_provider: '{stt_provider}'. Valid: {list(STT_PROVIDERS)}"})
                stt["provider"] = stt_provider
                changes.append(f"stt_provider={stt_provider}")

            # STT model
            if stt_model:
                provider = stt_provider or stt.get("provider", "openai")
                stt.setdefault(provider, {})["model"] = stt_model
                changes.append(f"stt_model={stt_model}")

            # STT language
            if stt_language:
                provider = stt_provider or stt.get("provider", "openai")
                stt.setdefault(provider, {})["language"] = stt_language
                changes.append(f"stt_language={stt_language}")

            if not changes:
                return json.dumps({"status": "no_changes", "message": "No parameters provided"})

            save_config(agent.config)
            result = {"status": "updated", "changes": changes}
            if warnings:
                result["warnings"] = warnings
            return json.dumps(result, ensure_ascii=False)

        def list_voice_providers_handler() -> str:
            """List available TTS and STT providers with their capabilities and pricing."""
            from .voice import (TTS_PROVIDERS, STT_PROVIDERS, OPENAI_TTS_VOICES,
                                OPENAI_TTS_MODELS, TTS_COST_INFO, STT_COST_INFO,
                                resolve_voice_config)
            from .config import get_api_key

            cfg = resolve_voice_config(agent.config)

            tts_list = []
            for p in TTS_PROVIDERS:
                entry = {"id": p, "cost": TTS_COST_INFO.get(p, "unknown")}
                if p == "openai":
                    entry["configured"] = bool(get_api_key("openai"))
                    entry["models"] = list(OPENAI_TTS_MODELS)
                    entry["voices"] = list(OPENAI_TTS_VOICES)
                elif p == "elevenlabs":
                    entry["configured"] = bool(get_api_key("elevenlabs") or os.environ.get("ELEVENLABS_API_KEY"))
                    entry["models"] = ["eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_monolingual_v1"]
                    entry["voices"] = ["Use voice_id from ElevenLabs dashboard"]
                elif p == "edge":
                    entry["configured"] = True
                    entry["models"] = []
                    entry["voices"] = [
                        "ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural",
                        "en-US-MichelleNeural", "en-US-GuyNeural",
                        "en-GB-SoniaNeural", "de-DE-KatjaNeural",
                        "fr-FR-DeniseNeural", "es-ES-ElviraNeural",
                        "ja-JP-NanamiNeural", "zh-CN-XiaoxiaoNeural",
                    ]
                tts_list.append(entry)

            stt_list = []
            for p in STT_PROVIDERS:
                entry = {"id": p, "cost": STT_COST_INFO.get(p, "unknown")}
                if p == "openai":
                    entry["configured"] = bool(get_api_key("openai"))
                    entry["models"] = ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe"]
                elif p == "deepgram":
                    entry["configured"] = bool(get_api_key("deepgram") or os.environ.get("DEEPGRAM_API_KEY"))
                    entry["models"] = ["nova-3", "nova-2"]
                elif p == "groq":
                    entry["configured"] = bool(get_api_key("groq") or os.environ.get("GROQ_API_KEY"))
                    entry["models"] = ["whisper-large-v3"]
                stt_list.append(entry)

            return json.dumps({
                "tts_providers": tts_list,
                "stt_providers": stt_list,
                "active_tts": cfg["tts"]["provider"],
                "active_stt": cfg["stt"]["provider"],
            }, indent=2, ensure_ascii=False)

        async def test_tts_handler(text: str, voice: str = "", provider: str = "") -> str:
            """Generate audio from text using current or overridden TTS settings.

            text: Text to convert to speech
            voice: Optional voice override (without changing settings)
            provider: Optional provider override (without changing settings)
            """
            from .voice import text_to_speech, resolve_voice_config
            from .file_queue import enqueue_file

            cfg = resolve_voice_config(agent.config)
            tts_cfg = cfg["tts"]

            # Apply overrides without modifying persistent config
            if provider:
                tts_cfg = {**tts_cfg, "provider": provider}
            if voice:
                p = provider or tts_cfg["provider"]
                if p == "openai":
                    tts_cfg = {**tts_cfg, "openai": {**tts_cfg.get("openai", {}), "voice": voice}}
                elif p == "edge":
                    tts_cfg = {**tts_cfg, "edge": {**tts_cfg.get("edge", {}), "voice": voice}}

            result = await text_to_speech(text, tts_cfg, agent.config)
            if result.success and result.audio_path:
                enqueue_file(
                    result.audio_path,
                    caption="",
                    mime_type="audio/opus" if result.voice_compatible else "audio/mpeg",
                    voice_compatible=result.voice_compatible,
                )
                resp = {
                    "status": "ok",
                    "provider": result.provider,
                    "format": result.output_format,
                    "latency_ms": result.latency_ms,
                }
                return json.dumps(resp)
            else:
                return json.dumps({"status": "error", "error": result.error or "TTS failed"})

        def save_voice_preset_handler(name: str, description: str = "") -> str:
            """Save current TTS settings as a named preset for quick loading later.

            name: Preset name (e.g. 'my_voice', 'work', 'podcast')
            description: Optional description of the preset
            """
            from .voice import resolve_voice_config
            from .config import save_config

            cfg = resolve_voice_config(agent.config)
            preset_data = {
                "provider": cfg["tts"]["provider"],
                "openai": cfg["tts"]["openai"],
                "elevenlabs": cfg["tts"]["elevenlabs"],
                "edge": cfg["tts"]["edge"],
            }
            if description:
                preset_data["description"] = description

            voice = agent.config.setdefault("voice", {})
            presets = voice.setdefault("presets", {})
            presets[name] = preset_data
            save_config(agent.config)
            return json.dumps({"status": "saved", "name": name, "settings": preset_data}, ensure_ascii=False)

        def load_voice_preset_handler(name: str) -> str:
            """Load a saved or built-in voice preset and apply its settings.

            name: Preset name (built-in: professional, casual, storyteller, fast_free, russian)
            """
            from .voice import BUILTIN_PRESETS, resolve_voice_config
            from .config import save_config
            import copy

            # Check custom presets first, then builtins
            custom = agent.config.get("voice", {}).get("presets", {})
            if name in custom:
                preset = custom[name]
            elif name in BUILTIN_PRESETS:
                preset = BUILTIN_PRESETS[name]
            else:
                available = list(BUILTIN_PRESETS.keys()) + list(custom.keys())
                return json.dumps({
                    "error": f"Preset '{name}' not found",
                    "available": available,
                }, ensure_ascii=False)

            # Apply preset to config
            voice = agent.config.setdefault("voice", {})
            tts = voice.setdefault("tts", {})
            if "provider" in preset:
                tts["provider"] = preset["provider"]
            if "openai" in preset:
                tts["openai"] = {**tts.get("openai", {}), **preset["openai"]}
            if "elevenlabs" in preset:
                tts["elevenlabs"] = {**tts.get("elevenlabs", {}), **preset["elevenlabs"]}
            if "edge" in preset:
                tts["edge"] = {**tts.get("edge", {}), **preset["edge"]}

            save_config(agent.config)
            cfg = resolve_voice_config(agent.config)
            return json.dumps({
                "status": "loaded",
                "preset": name,
                "applied": {
                    "provider": cfg["tts"]["provider"],
                    "voice": cfg["tts"].get(cfg["tts"]["provider"], {}).get("voice", ""),
                },
            }, ensure_ascii=False)

        # ── Register all 6 voice config tools ──

        self.tools._tools["get_voice_settings"] = {
            "name": "get_voice_settings",
            "description": (
                "Get current voice configuration: TTS auto-mode, provider, voice, model, "
                "STT settings, provider availability and pricing, saved presets."
            ),
            "input_schema": {"type": "object", "properties": {}},
        }
        self.tools._handlers["get_voice_settings"] = get_voice_settings_handler

        self.tools._tools["set_voice_settings"] = {
            "name": "set_voice_settings",
            "description": (
                "Update voice settings. Change TTS provider/voice/model/auto-mode, "
                "STT provider/model/language, ElevenLabs parameters. "
                "Only provided parameters are changed; others stay as-is. "
                "Settings are persisted to config.json."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "tts_auto": {
                        "type": "string",
                        "description": "Auto-TTS mode: off, always, inbound (echo voice), tagged (only [[tts]] directives)",
                    },
                    "tts_provider": {
                        "type": "string",
                        "description": "TTS provider: openai, elevenlabs, edge",
                    },
                    "tts_voice": {
                        "type": "string",
                        "description": "Voice name (openai: alloy/ash/coral/echo/fable/nova/onyx/sage/shimmer/verse; edge: ru-RU-SvetlanaNeural etc)",
                    },
                    "tts_model": {
                        "type": "string",
                        "description": "TTS model: tts-1, tts-1-hd, gpt-4o-mini-tts",
                    },
                    "tts_speed": {
                        "type": "number",
                        "description": "Speech speed (0.25-4.0, default 1.0)",
                    },
                    "tts_max_length": {
                        "type": "integer",
                        "description": "Max text length for TTS (default 1500)",
                    },
                    "elevenlabs_voice_id": {
                        "type": "string",
                        "description": "ElevenLabs voice ID",
                    },
                    "elevenlabs_stability": {
                        "type": "number",
                        "description": "ElevenLabs voice stability (0-1)",
                    },
                    "elevenlabs_similarity_boost": {
                        "type": "number",
                        "description": "ElevenLabs similarity boost (0-1)",
                    },
                    "stt_provider": {
                        "type": "string",
                        "description": "STT provider: openai, deepgram, groq",
                    },
                    "stt_model": {
                        "type": "string",
                        "description": "STT model name",
                    },
                    "stt_language": {
                        "type": "string",
                        "description": "STT language code (e.g. ru, en)",
                    },
                },
            },
        }
        self.tools._handlers["set_voice_settings"] = set_voice_settings_handler

        self.tools._tools["list_voice_providers"] = {
            "name": "list_voice_providers",
            "description": (
                "List all available TTS and STT providers with their models, "
                "voices, pricing, and whether they are configured (have API keys)."
            ),
            "input_schema": {"type": "object", "properties": {}},
        }
        self.tools._handlers["list_voice_providers"] = list_voice_providers_handler

        self.tools._tools["test_tts"] = {
            "name": "test_tts",
            "description": (
                "Convert text to speech audio. Optionally override voice/provider "
                "for testing without changing persistent settings. "
                "The audio file is sent to the user."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to convert to speech",
                    },
                    "voice": {
                        "type": "string",
                        "description": "Optional voice override for this request only",
                    },
                    "provider": {
                        "type": "string",
                        "description": "Optional provider override for this request only",
                    },
                },
                "required": ["text"],
            },
        }
        self.tools._handlers["test_tts"] = test_tts_handler

        self.tools._tools["save_voice_preset"] = {
            "name": "save_voice_preset",
            "description": (
                "Save current TTS settings as a named preset. "
                "Presets can be loaded later to quickly switch voice profiles."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Preset name (e.g. 'my_voice', 'work', 'podcast')",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional description of the preset",
                    },
                },
                "required": ["name"],
            },
        }
        self.tools._handlers["save_voice_preset"] = save_voice_preset_handler

        self.tools._tools["load_voice_preset"] = {
            "name": "load_voice_preset",
            "description": (
                "Load a voice preset and apply its TTS settings. "
                "Built-in presets: professional, casual, storyteller, fast_free, russian. "
                "Custom presets saved via save_voice_preset are also available."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Preset name to load",
                    },
                },
                "required": ["name"],
            },
        }
        self.tools._handlers["load_voice_preset"] = load_voice_preset_handler

    def _wire_storage_tools(self):
        """Register file storage + file manager tools when storage is available."""
        storage = self._storage
        fm = self._file_manager
        agent = self

        # ── save_file: save text content to storage ──
        async def save_file_handler(filename: str, content: str) -> str:
            """Save text content to cloud storage."""
            data = content.encode("utf-8")
            if fm:
                info = await fm.ingest(
                    data, filename,
                    source="agent", user_id=agent._current_user_id)
                return (f"File saved: {info['original_name']} → {info['storage_key']} "
                        f"({info['size_bytes']} bytes)")
            key = await storage.async_upload(filename, data)
            return f"File saved to storage: {key} ({len(data)} bytes)"

        # ── get_file: retrieve text content ──
        async def get_file_handler(storage_key: str) -> str:
            """Retrieve file content from storage by its key."""
            if ".." in storage_key or storage_key.startswith("/"):
                return "Access denied: invalid key"
            try:
                data = await storage.async_download(storage_key)
                return data.decode("utf-8", errors="replace")
            except Exception as e:
                return f"Error retrieving file: {e}"

        # ── search_files: semantic + keyword search across all user files ──
        async def search_files_handler(query: str, limit: int = 10) -> str:
            """Search files by description, name, or content. Uses semantic search."""
            if not fm:
                return "File manager not available."
            results = fm.search(query, top_k=min(limit, 50))
            if not results:
                return "No files found matching your query."
            lines = []
            for f in results:
                size_kb = f['size_bytes'] / 1024
                lines.append(
                    f"• {f['original_name']} ({size_kb:.1f} KB, {f['source']}) "
                    f"— {f['description'][:100]}\n"
                    f"  key: {f['storage_key']}")
            return f"Found {len(results)} files:\n" + "\n".join(lines)

        # ── list_all_files: list all files in storage ──
        async def list_all_files_handler(source: str = "", limit: int = 50) -> str:
            """List all indexed files. Optionally filter by source (telegram, api, voice, download, agent)."""
            if not fm:
                files = await storage.async_list_files(limit=limit)
                if not files:
                    return "No files in storage."
                lines = [f"{f['key']} ({f['size']} bytes)" for f in files]
                return "\n".join(lines)
            files = fm.list_files(
                source=source or None, limit=limit)
            if not files:
                return "No files found."
            total = fm.count_files()
            lines = []
            for f in files:
                size_kb = f['size_bytes'] / 1024
                lines.append(
                    f"• {f['original_name']} ({size_kb:.1f} KB, {f['source']}, "
                    f"{f['created_at'][:10]})\n"
                    f"  key: {f['storage_key']}")
            header = f"Files in storage ({len(files)} shown, {total} total):\n"
            return header + "\n".join(lines)

        # ── get_file_url: generate download link ──
        async def get_file_url_handler(storage_key: str, expires_hours: int = 1) -> str:
            """Generate a temporary download URL for a file."""
            if ".." in storage_key or storage_key.startswith("/"):
                return "Access denied: invalid key"
            try:
                if fm:
                    url = await fm.get_download_url(
                        storage_key, expires=expires_hours * 3600)
                else:
                    url = await storage.async_get_url(
                        storage_key, expires=expires_hours * 3600)
                return f"Download URL (valid {expires_hours}h): {url}"
            except Exception as e:
                return f"Error generating URL: {e}"

        # ── send_stored_file: download from S3 and send to user via file_queue ──
        async def send_stored_file_handler(storage_key: str, caption: str = "") -> str:
            """Send a file from storage directly to the user (Telegram/API)."""
            if ".." in storage_key or storage_key.startswith("/"):
                return "Access denied: invalid key"
            try:
                import tempfile
                data = await storage.async_download(storage_key)
                name = storage_key.rsplit("/", 1)[-1]
                tmp = os.path.join(tempfile.gettempdir(), f"s3_{name}")
                with open(tmp, "wb") as f:
                    f.write(data)
                from .file_queue import enqueue_file
                enqueue_file(tmp, caption=caption or name)
                return f"File queued for delivery: {name} ({len(data)} bytes)"
            except Exception as e:
                return f"Error sending file: {e}"

        # ── propose_cleanup: suggest files for deletion ──
        async def propose_cleanup_handler(days_unused: int = 30) -> str:
            """Propose old unused files for deletion. User MUST confirm before deleting."""
            if not fm:
                return "File manager not available."
            candidates = fm.propose_cleanup(days_unused=max(days_unused, 7))
            if not candidates:
                return "No cleanup candidates found. All files are recent."
            lines = []
            total_bytes = 0
            for f in candidates:
                size_kb = f['size_bytes'] / 1024
                total_bytes += f['size_bytes']
                lines.append(
                    f"• {f['original_name']} ({size_kb:.1f} KB, "
                    f"last access: {f['accessed_at'][:10]})\n"
                    f"  key: {f['storage_key']}")
            total_mb = total_bytes / (1024 * 1024)
            header = (
                f"Cleanup candidates ({len(candidates)} files, {total_mb:.1f} MB total):\n"
                f"⚠️ Show this list to the user and ask which files to delete.\n"
                f"DO NOT delete without explicit user confirmation!\n\n")
            return header + "\n".join(lines)

        # ── confirm_cleanup: actually delete after user says yes ──
        async def confirm_cleanup_handler(storage_keys: str) -> str:
            """Delete specific files from storage. Only call AFTER user confirmed.
            storage_keys: comma-separated list of storage keys."""
            if not fm:
                return "File manager not available."
            keys = [k.strip() for k in storage_keys.split(",") if k.strip()]
            if not keys:
                return "No keys provided."
            result = await fm.confirm_cleanup(keys)
            deleted = len(result.get("deleted", []))
            errors = len(result.get("errors", []))
            return f"Deleted {deleted} files. Errors: {errors}."

        # Register all tools
        tools_defs = [
            ("save_file", save_file_handler,
             "Save text content to cloud storage. Automatically indexed and searchable.",
             {"type": "object", "properties": {
                 "filename": {"type": "string", "description": "Filename (e.g. 'notes.txt', 'report.md')"},
                 "content": {"type": "string", "description": "File content (text)"},
             }, "required": ["filename", "content"]}),
            ("get_file", get_file_handler,
             "Retrieve text file content from storage by storage key.",
             {"type": "object", "properties": {
                 "storage_key": {"type": "string", "description": "Storage key (from search or list)"},
             }, "required": ["storage_key"]}),
            ("search_files", search_files_handler,
             "Search through all stored files by name, description, or content. "
             "Use this to find specific documents, images, or data the user uploaded.",
             {"type": "object", "properties": {
                 "query": {"type": "string", "description": "Search query (name, topic, content keywords)"},
                 "limit": {"type": "integer", "description": "Max results (default 10)"},
             }, "required": ["query"]}),
            ("list_all_files", list_all_files_handler,
             "List all files in cloud storage. Filter by source: telegram, api, voice, download, agent.",
             {"type": "object", "properties": {
                 "source": {"type": "string", "description": "Filter by source (optional)"},
                 "limit": {"type": "integer", "description": "Max files to show (default 50)"},
             }, "required": []}),
            ("get_file_url", get_file_url_handler,
             "Generate a temporary download URL for a file in storage. "
             "Give this link to the user so they can download the file.",
             {"type": "object", "properties": {
                 "storage_key": {"type": "string", "description": "Storage key of the file"},
                 "expires_hours": {"type": "integer", "description": "URL validity in hours (default 1)"},
             }, "required": ["storage_key"]}),
            ("send_stored_file", send_stored_file_handler,
             "Download a file from storage and send it to the user as an attachment "
             "(works in Telegram and API chat). Use when user wants to receive a specific file.",
             {"type": "object", "properties": {
                 "storage_key": {"type": "string", "description": "Storage key of the file"},
                 "caption": {"type": "string", "description": "Caption for the file (optional)"},
             }, "required": ["storage_key"]}),
            ("propose_cleanup", propose_cleanup_handler,
             "Analyze storage for unused files and propose candidates for deletion. "
             "IMPORTANT: Never delete files without showing the list to the user first and getting confirmation.",
             {"type": "object", "properties": {
                 "days_unused": {"type": "integer", "description": "Days since last access (default 30, min 7)"},
             }, "required": []}),
            ("confirm_cleanup", confirm_cleanup_handler,
             "Delete files from storage. ONLY call this after the user explicitly confirmed "
             "which files to delete from the propose_cleanup list.",
             {"type": "object", "properties": {
                 "storage_keys": {"type": "string", "description": "Comma-separated storage keys to delete"},
             }, "required": ["storage_keys"]}),
        ]
        for name, handler, desc, schema in tools_defs:
            self.tools._tools[name] = {
                "name": name, "description": desc, "input_schema": schema,
            }
            self.tools._handlers[name] = handler

    async def _auto_ingest_tool_file(self, block, user_id: str):
        """Auto-upload files produced by download_file or write_file to S3."""
        fm = self._file_manager
        if not fm:
            return
        if not hasattr(block, 'name') or not isinstance(getattr(block, 'input', None), dict):
            return
        try:
            if block.name == "download_file":
                # download_file returns "Downloaded to: /path (N bytes)"
                # The file is at the path in block.input
                url = block.input.get("url", "")
                filename = block.input.get("filename", "")
                # Find the result — check tool_results which are already in messages
                # Simpler: just find the file in downloads dir
                import glob as _glob
                downloads_dir = os.path.join(os.path.expanduser("~"), ".liteagent", "downloads")
                if not filename:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(url)
                    filename = os.path.basename(parsed.path) or "download"
                # Find most recent matching file
                pattern = os.path.join(downloads_dir, f"*_{filename}")
                matches = sorted(_glob.glob(pattern), key=os.path.getmtime, reverse=True)
                if matches:
                    await fm.ingest_local(
                        matches[0], source="download", user_id=user_id,
                        description=f"Downloaded from {url}")
            elif block.name == "write_file":
                path = block.input.get("path", "")
                if path and os.path.exists(path):
                    await fm.ingest_local(
                        path, source="agent", user_id=user_id)
        except Exception as e:
            logger.debug("Auto-ingest failed for %s: %s", block.name, e)

    async def ingest_file(self, data: bytes, filename: str, *,
                          source: str = "unknown", user_id: str = "system",
                          mime_type: str = "", description: str = "") -> dict | None:
        """Public method for channels to auto-ingest files into S3."""
        fm = self._file_manager
        if not fm:
            return None
        try:
            return await fm.ingest(
                data, filename, source=source, user_id=user_id,
                mime_type=mime_type, description=description)
        except Exception as e:
            logger.warning("File ingestion failed: %s", e)
            return None

    def enable_tasks(self, task_manager):
        """Enable task scheduling tools. Called by main.py after TaskManager is created."""
        self._task_manager = task_manager
        self._wire_task_tools()

    def _wire_task_tools(self):
        """Register schedule_task, list_tasks, cancel_task tools for LLM use."""
        import json as _json
        import re as _re
        from datetime import datetime as _dt, timedelta as _td
        agent = self
        tm = self._task_manager

        def _parse_schedule(schedule: str) -> tuple[str | None, str | None]:
            """Parse human-readable schedule into (run_at, cron_expr).

            Returns exactly one of them set, the other None.
            Supports Russian and English, relative and absolute times.
            """
            raw = schedule.strip()
            s = raw.lower()

            # ── Already valid 5-field cron ──
            if _re.match(r'^[\d*/,-]+\s+[\d*/,-]+\s+[\d*/,-]+\s+[\d*/,-]+\s+[\d*/,-]+$', s):
                return None, s

            # ── Already ISO datetime ──
            if _re.match(r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', s):
                return raw, None

            now = _dt.now()

            # ── Relative: "через N минут/часов" / "in N minutes/hours" ──
            m = _re.search(r'(?:через|in|after)\s+(\d+)\s*(?:мин|min)', s)
            if m:
                dt = now + _td(minutes=int(m.group(1)))
                return dt.isoformat(timespec='seconds'), None
            m = _re.search(r'(?:через|in|after)\s+(\d+)\s*(?:час|hour|hr)', s)
            if m:
                dt = now + _td(hours=int(m.group(1)))
                return dt.isoformat(timespec='seconds'), None
            m = _re.search(r'(?:через|in|after)\s+(\d+)\s*(?:сек|sec)', s)
            if m:
                dt = now + _td(seconds=max(60, int(m.group(1))))
                return dt.isoformat(timespec='seconds'), None

            # ── Recurring: "каждый день/ежедневно/daily" ──
            daily_rx = r'(?:каждый\s*день|ежедневно|daily|dayly|every\s*day)'
            m = _re.search(daily_rx + r'(?:\s+(?:в|at))?\s+(\d{1,2})[:\.](\d{2})', s)
            if m:
                return None, f"{int(m.group(2))} {int(m.group(1))} * * *"
            m = _re.search(daily_rx + r'(?:\s+(?:в|at))?\s+(\d{1,2})(?:\s|$)', s)
            if m:
                return None, f"0 {int(m.group(1))} * * *"
            if _re.search(daily_rx, s):
                # "каждый день" без времени → 9:00
                return None, "0 9 * * *"

            # ── Recurring: "каждые N минут/часов" / "every N min/hours" ──
            m = _re.search(r'(?:каждые?|every)\s+(\d+)\s*(?:мин|min)', s)
            if m:
                return None, f"*/{m.group(1)} * * * *"
            m = _re.search(r'(?:каждые?|every)\s+(\d+)\s*(?:час|hour|hr)', s)
            if m:
                return None, f"0 */{m.group(1)} * * *"
            if _re.search(r'(?:каждую\s*минуту|every\s*min)', s):
                return None, "* * * * *"
            if _re.search(r'(?:каждый\s*час|every\s*hour)', s):
                return None, "0 * * * *"

            # ── Recurring: "по будням / weekdays" ──
            m = _re.search(r'(?:по\s*будням|будни|weekdays?)(?:\s+(?:в|at))?\s+(\d{1,2})[:\.](\d{2})', s)
            if m:
                return None, f"{int(m.group(2))} {int(m.group(1))} * * 1-5"

            # ── Recurring: "по понедельникам/вторникам..." ──
            day_map = {
                r'понедельник|monday|mon': '1', r'вторник|tuesday|tue': '2',
                r'сред[аы]|wednesday|wed': '3', r'четверг|thursday|thu': '4',
                r'пятниц[аы]|friday|fri': '5', r'суббот[аы]|saturday|sat': '6',
                r'воскресень[еям]|sunday|sun': '0',
            }
            for pattern, dow in day_map.items():
                m_day = _re.search(r'(?:по\s*|every\s*)?' + f'(?:{pattern})', s)
                if m_day:
                    m_time = _re.search(r'(?:в|at)?\s*(\d{1,2})[:\.](\d{2})', s)
                    if m_time:
                        return None, f"{int(m_time.group(2))} {int(m_time.group(1))} * * {dow}"
                    return None, f"0 9 * * {dow}"

            # ── One-shot: "завтра в HH:MM" / "tomorrow at HH:MM" ──
            m = _re.search(r'(?:завтра|tomorrow)(?:\s+(?:в|at))?\s+(\d{1,2})[:\.](\d{2})', s)
            if m:
                dt = (now + _td(days=1)).replace(
                    hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
                return dt.isoformat(timespec='seconds'), None
            if _re.search(r'завтра|tomorrow', s):
                dt = (now + _td(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
                return dt.isoformat(timespec='seconds'), None

            # ── One-shot: "сегодня в HH:MM" / "today at HH:MM" ──
            m = _re.search(r'(?:сегодня|today)(?:\s+(?:в|at))?\s+(\d{1,2})[:\.](\d{2})', s)
            if m:
                dt = now.replace(
                    hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
                if dt <= now:
                    dt += _td(days=1)
                return dt.isoformat(timespec='seconds'), None

            # ── Bare "HH:MM" → one-shot today (or tomorrow if past) ──
            m = _re.match(r'^(\d{1,2})[:\.](\d{2})$', s)
            if m:
                dt = now.replace(
                    hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
                if dt <= now:
                    dt += _td(days=1)
                return dt.isoformat(timespec='seconds'), None

            # ── Can't parse → return as-is in cron (will fail with clear error) ──
            return None, s

        async def schedule_task_handler(name: str, query: str,
                                        schedule: str = "") -> str:
            """Schedule a task. The schedule is parsed automatically.
            name: Short task name (e.g. "Проверка погоды", "Напоминание")
            query: What the agent should do when the task fires
            schedule: When to run. Examples:
              - "через 30 минут" / "in 30 minutes"
              - "завтра в 9:00" / "tomorrow at 9:00"
              - "каждый день в 8:00" / "daily at 8:00"
              - "каждые 2 часа" / "every 2 hours"
              - "по будням в 10:00" / "weekdays 10:00"
              - "0 9 * * *" (raw cron)
            """
            if not schedule:
                return "Error: schedule is required (e.g. 'через 30 минут', 'каждый день в 9:00')"

            run_at, cron = _parse_schedule(schedule)
            if not run_at and not cron:
                return f"Error: could not parse schedule '{schedule}'"

            task_type = "recurring" if cron else "one_shot"
            uid = agent._current_user_id
            chat_id = None
            if uid.startswith("tg-"):
                chat_id = getattr(agent, "_current_chat_id", None)
            try:
                task = tm.add_task(
                    name=name, query=query, user_id=uid,
                    task_type=task_type,
                    run_at=run_at,
                    cron_expr=cron,
                    chat_id=str(chat_id) if chat_id else None,
                )
                return _json.dumps(task, ensure_ascii=False, default=str)
            except Exception as e:
                return f"Error creating task: {e}"

        async def list_tasks_handler() -> str:
            """List all your scheduled tasks."""
            tasks = tm.list_tasks(user_id=agent._current_user_id)
            if not tasks:
                return "No tasks scheduled."
            lines = []
            for t in tasks:
                schedule = t.get("cron_expr") or t.get("run_at") or "?"
                lines.append(
                    f"#{t['id']} [{t['status']}] {t['name']} "
                    f"({t['task_type']}, {schedule})"
                )
            return "\n".join(lines)

        async def cancel_task_handler(task_id: int) -> str:
            """Cancel a scheduled task by its ID.
            task_id: The numeric ID of the task to cancel
            """
            ok = tm.cancel_task(int(task_id))
            return "Task cancelled." if ok else "Task not found or already completed/cancelled."

        for name, handler, desc, schema in [
            ("schedule_task", schedule_task_handler,
             "Schedule a task for the user. Call this when the user wants to be reminded about something, "
             "or wants something done at a specific time or on a recurring schedule. "
             "The 'schedule' parameter accepts natural language in Russian or English: "
             "'через 30 минут', 'завтра в 9:00', 'каждый день в 8:00', 'каждые 2 часа', "
             "'по будням в 10:00', 'по понедельникам в 14:00', 'every 30 minutes', 'daily 9:00'. "
             "Also accepts cron: '0 9 * * *'. "
             "The 'query' is the instruction the agent will execute when the task fires.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "Short task name (e.g. 'Проверка погоды')"},
                 "query": {"type": "string", "description": "Instruction for the agent to execute when task fires"},
                 "schedule": {"type": "string",
                              "description": "When to run: 'через 30 минут', 'каждый день в 8:00', 'завтра в 9:00', 'daily 9:00', '0 9 * * *'"},
             }, "required": ["name", "query", "schedule"]}),

            ("list_tasks", list_tasks_handler,
             "List all your scheduled and completed tasks with their IDs, statuses, and schedules.",
             {"type": "object", "properties": {}}),

            ("cancel_task", cancel_task_handler,
             "Cancel a scheduled task by its numeric ID. Use list_tasks first to see IDs.",
             {"type": "object", "properties": {
                 "task_id": {"type": "integer", "description": "Task ID to cancel"},
             }, "required": ["task_id"]}),
        ]:
            self.tools._tools[name] = {
                "name": name, "description": desc, "input_schema": schema,
            }
            self.tools._handlers[name] = handler

        logger.info("Task scheduling tools registered (schedule_task, list_tasks, cancel_task)")

    def _init_file_access_tracking(self):
        """Initialize SQLite table for file access tracking (auto-ingestion)."""
        self.memory.db.executescript("""
            CREATE TABLE IF NOT EXISTS file_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                user_id TEXT NOT NULL,
                access_count INTEGER DEFAULT 1,
                last_accessed TEXT,
                indexed INTEGER DEFAULT 0,
                UNIQUE(path, user_id)
            );
        """)
        self.memory.db.commit()

    def track_file_access(self, path: str, user_id: str):
        """Track file access for auto-ingestion suggestions."""
        now = datetime.now().isoformat()
        self.memory.db.execute("""
            INSERT INTO file_access_log (path, user_id, access_count, last_accessed)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(path, user_id) DO UPDATE SET
                access_count = access_count + 1,
                last_accessed = ?
        """, (path, user_id, now, now))
        self.memory.db.commit()

    def get_ingestion_suggestions(self, user_id: str) -> list[str]:
        """Get files that have been accessed frequently but not yet indexed."""
        threshold = self._auto_ingestion.get("access_threshold", 3)
        rows = self.memory.db.execute(
            "SELECT path FROM file_access_log WHERE user_id=? AND access_count >= ? AND indexed = 0",
            (user_id, threshold)).fetchall()
        return [r[0] for r in rows]

    def mark_file_indexed(self, path: str, user_id: str):
        """Mark a file as indexed in the access log."""
        self.memory.db.execute(
            "UPDATE file_access_log SET indexed = 1 WHERE path = ? AND user_id = ?",
            (path, user_id))
        self.memory.db.commit()

    # ══════════════════════════════════════════
    # TRANSLATION LAYER (for Ollama/Qwen)
    # ══════════════════════════════════════════

    def _needs_translation(self) -> bool:
        """Check if translation layer is needed (Ollama provider)."""
        provider = self.config.get("agent", {}).get("provider", "anthropic")
        return provider == "ollama"

    async def _translate(self, text: str, to_lang: str = "en") -> str:
        """Translate text using the current model.

        to_lang: 'en' for Russian→English, 'ru' for English→Russian.
        """
        if not text or not text.strip():
            return text

        if to_lang == "en":
            prompt = (
                "Translate the following Russian text to English. "
                "Output ONLY the translation, nothing else. "
                "Keep technical terms, code, and proper nouns unchanged.\n\n"
                f"{text}"
            )
        else:
            prompt = (
                "Translate the following English text to Russian. "
                "Output ONLY the translation, nothing else. "
                "Keep technical terms, code, and proper nouns unchanged.\n\n"
                f"{text}"
            )

        try:
            response = await self._call_api(
                model=self.default_model,
                max_tokens=2048,
                system="You are a professional translator. Output ONLY the translation.",
                tools=[],
                messages=[{"role": "user", "content": prompt}],
            )
            return self._extract_text(response) or text
        except Exception as e:
            logger.warning("Translation failed: %s", e)
            return text

    # ══════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════

    async def _ensure_mcp_loaded(self):
        """Lazy-load MCP servers on first use."""
        if not self._mcp_loaded and self._mcp_config:
            await self.tools.load_mcp_servers(self._mcp_config)
            self._mcp_loaded = True
            self._apply_voice_transcription_mode()

    def apply_config_update(self, new_config: dict):
        """Apply reloadable config changes at runtime (called by ConfigWatcher)."""
        agent_cfg = new_config.get("agent", {})
        cost_cfg = new_config.get("cost", {})

        # Update cascade models
        if "models" in agent_cfg:
            self.models = agent_cfg["models"]
        if "default_model" in agent_cfg:
            self.default_model = agent_cfg["default_model"]

        # Cost controls
        if "cascade_routing" in cost_cfg:
            self.cascade_routing = cost_cfg["cascade_routing"]
        if "budget_daily_usd" in cost_cfg:
            self.budget_daily = cost_cfg["budget_daily_usd"]

        # Max iterations
        if "max_iterations" in agent_cfg:
            self.max_iterations = agent_cfg["max_iterations"]

        # MCP: if changed, mark for reload
        new_mcp = new_config.get("tools", {}).get("mcp_servers", {})
        if new_mcp != self._mcp_config:
            self._mcp_config = new_mcp
            self._mcp_loaded = False
            logger.info("MCP config changed — will reload on next request")

        self.config = new_config
        logger.info("Config update applied at runtime")

    def _apply_voice_transcription_mode(self):
        """Apply voice_transcription setting: 'auto', 'builtin', or 'mcp'.

        - auto (default): MCP overrides builtin if MCP provides transcription
        - builtin: always use built-in OpenAI Whisper, disable MCP transcription
        - mcp: always use MCP transcription, disable builtin
        """
        mode = (self.config.get("channels", {})
                .get("telegram", {})
                .get("voice_transcription", "auto"))

        mcp_transcribe = [n for n in self.tools._tools
                          if "transcribe" in n and "__" in n]

        if mode == "builtin":
            # Remove MCP transcription tools, keep builtin
            for name in mcp_transcribe:
                del self.tools._tools[name]
                self.tools._handlers.pop(name, None)
            # Re-register builtin if missing
            if "transcribe_voice" not in self.tools._tools:
                self._wire_voice_tool()
            if "get_voice_settings" not in self.tools._tools:
                self._wire_voice_config_tools()
            logger.info("Voice transcription: builtin (OpenAI Whisper)")

        elif mode == "mcp":
            # Remove builtin, keep MCP
            if mcp_transcribe:
                self.tools._tools.pop("transcribe_voice", None)
                self.tools._handlers.pop("transcribe_voice", None)
                logger.info("Voice transcription: MCP (%s)",
                            ", ".join(mcp_transcribe))
            else:
                logger.warning("Voice transcription: MCP requested but no MCP "
                               "transcription tool found, keeping builtin")

        else:  # auto
            if mcp_transcribe and "transcribe_voice" in self.tools._tools:
                del self.tools._tools["transcribe_voice"]
                self.tools._handlers.pop("transcribe_voice", None)
                logger.info("Voice transcription: auto → MCP (%s)",
                            ", ".join(mcp_transcribe))
            else:
                logger.info("Voice transcription: auto → builtin")

    async def reload_mcp(self):
        """Reload MCP servers from config."""
        await self.tools.close_mcp_servers()
        # Remove MCP tools from registry
        mcp_tools = [n for n in list(self.tools._tools) if "__" in n]
        for t in mcp_tools:
            del self.tools._tools[t]
            if t in self.tools._handlers:
                del self.tools._handlers[t]
        # Re-register builtin voice tools (may have been removed by previous mode)
        if "transcribe_voice" not in self.tools._tools:
            self._wire_voice_tool()
        if "get_voice_settings" not in self.tools._tools:
            self._wire_voice_config_tools()
        self._mcp_loaded = False
        await self._ensure_mcp_loaded()
        logger.info("MCP servers reloaded: %d servers",
                    len(self.tools.get_mcp_server_info()))

    async def run(self, user_input: str | list, user_id: str = "default") -> str:
        """Run agent on user input with per-user serialization."""
        LiteAgent._ensure_locks()
        lock = await self._get_user_lock(user_id)
        q_id = self._track_queued(user_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("Request timeout: user %s waited >60s for lock", user_id)
            return "⏳ Request queued too long. Please try again in a moment."
        finally:
            self._untrack_queued(q_id)
        try:
            return await self._run_impl(user_input, user_id)
        finally:
            lock.release()

    async def _run_impl(self, user_input: str | list, user_id: str = "default") -> str:
        """Run agent on user input. Accepts str or list of content blocks (multimodal)."""
        self._current_user_id = user_id
        from .file_queue import init_file_queue
        init_file_queue()
        await self._ensure_mcp_loaded()
        self._ensure_onboarding_tool()
        # Load persisted history on first interaction
        if not self.memory.get_history(user_id):
            self.memory.load_history(user_id)

        # Normalize multimodal input
        _file_metas = []  # collected file/image metadata for memory
        if isinstance(user_input, list):
            text_for_memory = " ".join(
                b.get("text", "") for b in user_input if b.get("type") == "text")
            content_for_api = user_input
            # Collect file metadata from multimodal blocks
            for block in user_input:
                btype = block.get("type", "")
                if btype == "image":
                    src = block.get("source", {})
                    _file_metas.append({"type": "image", "mime": src.get("media_type", "image/unknown")})
                elif btype == "document":
                    src = block.get("source", {})
                    _file_metas.append({"type": "document", "mime": src.get("media_type", "application/octet-stream")})
                elif btype == "text":
                    txt = block.get("text", "")
                    # Detect file markers from text wrappers
                    import re as _re
                    fm = _re.search(r'--- File:\s*(.+?)\s*(?:\(([^)]+)\))?\s*---', txt)
                    if fm:
                        _file_metas.append({"type": "file", "filename": fm.group(1),
                                            "info": fm.group(2) or ""})
        else:
            text_for_memory = user_input
            content_for_api = user_input

        # Budget check
        if self.memory.get_today_cost() >= self.budget_daily:
            return f"⚠️ Daily budget (${self.budget_daily:.2f}) reached. Reset tomorrow."

        # Persist user message immediately (original language)
        self.memory.add_message(user_id, "user", text_for_memory)

        # Translation layer: translate user input to English for Ollama/Qwen
        _translate_back = False
        if self._needs_translation() and isinstance(content_for_api, str):
            translated = await self._translate(content_for_api, to_lang="en")
            if translated != content_for_api:
                logger.info("Translated user input to English (%d→%d chars)",
                            len(content_for_api), len(translated))
                content_for_api = translated
                _translate_back = True

        # Build context (token-efficient)
        system_prompt = self._build_system_prompt(text_for_memory, user_id)
        messages = self.memory.get_compressed_history(user_id)
        messages.append({"role": "user", "content": content_for_api})

        # Select model (cascade routing — may switch provider for cross-provider cascade)
        complexity_score = self._complexity_score(text_for_memory)
        model = self._model_for_score(complexity_score) if self.cascade_routing else self.default_model
        _cascade_tier = self._tier_for_score(complexity_score) if self.cascade_routing else "fixed"
        _provider_switched = hasattr(self, '_original_provider')
        logger.info("Model: %s | User: %s | Input: %d chars | Complexity: %d | Tier: %s%s",
                     model, user_id, len(text_for_memory), complexity_score, _cascade_tier,
                     " [cross-provider]" if _provider_switched else "")
        LiteAgent._record_cascade_decision(model, _cascade_tier, complexity_score)

        # Tool selection: skip tools for trivial messages (greetings, short chat)
        # But always include tools from triggered skills (voice mode switching, etc.)
        _triggered_skills = self.skill_registry.get_triggered_skills(text_for_memory)
        _skill_tool_names = set()
        for _sk in _triggered_skills:
            _skill_tool_names.update(_sk.metadata.tools or [])

        if complexity_score <= 0 and len(text_for_memory) < 60 and not _skill_tool_names:
            tool_defs = []
            logger.debug("Skipping tools for simple message")
        elif self.memory._embedder and len(self.tools._tools) > 8:
            tool_defs = self.tools.get_relevant_definitions(
                text_for_memory, top_k=8, embedder=self.memory._embedder)
        else:
            tool_defs = self.tools.get_definitions()

        # Ensure triggered skill tools are always included in tool_defs
        if _skill_tool_names:
            existing_names = {td["name"] for td in tool_defs}
            for _stn in _skill_tool_names:
                if _stn not in existing_names and _stn in self.tools._tools:
                    tool_defs.append(self.tools._tools[_stn])
                    logger.debug("Added skill tool: %s", _stn)

        # Track tool calls for skill crystallization
        _tool_calls_log = []
        _tool_results_summary = []

        # Internal monologue: pre-planning
        _plan, tool_defs, model, _effective_max = await self._apply_planning(
            text_for_memory, user_id, system_prompt, tool_defs, model)

        # Track in-flight request for dashboard
        _req_id = await self._track_request_start(
            user_id,
            text_for_memory[:120] if isinstance(text_for_memory, str) else "multimodal",
            model,
            complexity_score=complexity_score,
            cascade_tier=_cascade_tier)

        # Agent loop
        try:
            for iteration in range(_effective_max):
                t0 = time.time()

                response = await self._call_api(
                    model=model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=tool_defs,
                    messages=messages,
                )

                # Track usage
                cost = self._calculate_cost(model, response.usage)
                self.memory.track_usage(user_id, model, response.usage, cost)

                elapsed = time.time() - t0
                logger.debug("Iteration %d: %.2fs, $%.6f, stop=%s",
                             iteration, elapsed, cost, response.stop_reason)

                if response.stop_reason == "tool_use":
                    # Execute tools
                    tool_results = await self.tools.execute(response.content)
                    messages.append({"role": "assistant", "content": _serialize_content(response.content)})
                    messages.append({"role": "user", "content": tool_results})
                    # Log tool calls for skill crystallization + file access tracking
                    for block in response.content:
                        if hasattr(block, 'type') and block.type == "tool_use":
                            _tool_calls_log.append({
                                "name": block.name,
                                "input": block.input,
                            })
                            # Auto-ingestion: track file reads
                            if (self._auto_ingestion.get("enabled")
                                    and block.name == "read_file"
                                    and isinstance(block.input, dict)):
                                file_path = block.input.get("path", "")
                                if file_path:
                                    self.track_file_access(file_path, user_id)
                            # Auto-upload downloaded files to S3
                            await self._auto_ingest_tool_file(block, user_id)
                    # Collect tool result summaries for reflection
                    for tr in tool_results:
                        content = tr.get("content", "") if isinstance(tr, dict) else str(tr)
                        _tool_results_summary.append(str(content)[:200])
                    # Mid-loop reflection (internal monologue) — merges into last message
                    await self._apply_reflection(
                        messages, _plan, _tool_calls_log, _tool_results_summary)
                else:
                    # Done — extract text
                    text = self._extract_text(response)

                    # ── Fallback: parse tool call from plain text ──
                    # Some models (e.g. Ollama/qwen) output tool calls as text JSON
                    # instead of structured tool_use. Detect and execute them.
                    parsed_tool = self._try_parse_text_tool_call(text, tool_defs)
                    if parsed_tool:
                        block = ToolUseBlock(
                            id=f"fallback_{iteration}",
                            name=parsed_tool["name"],
                            input=parsed_tool["arguments"],
                        )
                        logger.info("Fallback tool call parsed from text: %s", block.name)
                        tool_results = await self.tools.execute([block])
                        messages.append({"role": "assistant", "content": _serialize_content([block])})
                        messages.append({"role": "user", "content": tool_results})
                        _tool_calls_log.append({
                            "name": block.name, "input": block.input,
                        })
                        continue

                    # Save assistant response (user message already persisted above)
                    self.memory.add_message(user_id, "assistant", text)

                    # ── Feature hooks (post-response via hook system) ──
                    hook_ctx = HookContext(
                        agent=self, user_id=user_id,
                        user_input=content_for_api, model=model,
                        system_prompt=system_prompt, tool_defs=tool_defs,
                        messages=messages, response_text=text,
                        tool_calls_log=_tool_calls_log,
                        extra={"user_input_text": text_for_memory})
                    hook_ctx = await self.hooks.emit("after_response", hook_ctx)
                    text = hook_ctx.response_text

                    # Auto-ingestion suggestions
                    if (self._auto_ingestion.get("enabled")
                            and self._auto_ingestion.get("suggest_in_chat", True)
                            and self._rag):
                        suggestions = self.get_ingestion_suggestions(user_id)
                        if suggestions:
                            paths = ", ".join(suggestions[:3])
                            text += (f"\n\n💡 Я часто обращаюсь к: {paths}. "
                                     "Хочешь, проиндексирую в RAG для мгновенного поиска?")

                    # Translation layer: translate response back to Russian
                    if _translate_back:
                        text = await self._translate(text, to_lang="ru")
                        logger.info("Translated response back to Russian")

                    # Restore original provider after cross-provider cascade
                    if _provider_switched:
                        self._cascade_restore_provider()

                    # Background: extract knowledge (non-blocking)
                    task = asyncio.create_task(
                        self._safe_extract(text_for_memory, text, user_id, file_meta=_file_metas)
                    )
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)

                    return text

            # Restore provider even on max iterations
            if _provider_switched:
                self._cascade_restore_provider()
            return "⚠️ Max iterations reached. Try a simpler request."
        finally:
            await self._track_request_end(_req_id)
            # Safety: always restore provider if still switched (e.g., on exception)
            if hasattr(self, '_original_provider'):
                self._cascade_restore_provider()

    async def stream(self, user_input: str, user_id: str = "default") -> AsyncGenerator[str, None]:
        """Stream agent response with per-user serialization."""
        LiteAgent._ensure_locks()
        lock = await self._get_user_lock(user_id)
        q_id = self._track_queued(user_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning("Stream timeout: user %s waited >60s for lock", user_id)
            yield "⏳ Request queued too long. Please try again in a moment."
            self._untrack_queued(q_id)
            return
        self._untrack_queued(q_id)
        try:
            async for chunk in self._stream_impl(user_input, user_id):
                yield chunk
        finally:
            lock.release()

    async def _stream_impl(self, user_input: str, user_id: str = "default") -> AsyncGenerator[str, None]:
        """Stream agent response token by token."""
        self._current_user_id = user_id
        await self._ensure_mcp_loaded()
        self._ensure_onboarding_tool()
        # Load persisted history on first interaction
        if not self.memory.get_history(user_id):
            self.memory.load_history(user_id)

        if self.memory.get_today_cost() >= self.budget_daily:
            yield f"⚠️ Daily budget (${self.budget_daily:.2f}) reached."
            return

        _file_metas = []  # no multimodal in stream path

        # Persist user message immediately (before streaming starts)
        self.memory.add_message(user_id, "user", user_input)

        system_prompt = self._build_system_prompt(user_input, user_id)
        messages = self.memory.get_compressed_history(user_id)
        messages.append({"role": "user", "content": user_input})
        complexity_score = self._complexity_score(user_input)
        model = self._model_for_score(complexity_score) if self.cascade_routing else self.default_model
        _cascade_tier = self._tier_for_score(complexity_score) if self.cascade_routing else "fixed"
        _provider_switched = hasattr(self, '_original_provider')
        logger.info("Stream | Model: %s | User: %s | Complexity: %d | Tier: %s%s",
                     model, user_id, complexity_score, _cascade_tier,
                     " [cross-provider]" if _provider_switched else "")
        LiteAgent._record_cascade_decision(model, _cascade_tier, complexity_score)

        # Skip tools for trivial messages, but always include triggered skill tools
        _triggered_skills = self.skill_registry.get_triggered_skills(user_input)
        _skill_tool_names = set()
        for _sk in _triggered_skills:
            _skill_tool_names.update(_sk.metadata.tools or [])

        if complexity_score <= 0 and len(user_input) < 60 and not _skill_tool_names:
            tool_defs = []
        elif self.memory._embedder and len(self.tools._tools) > 8:
            tool_defs = self.tools.get_relevant_definitions(
                user_input, top_k=8, embedder=self.memory._embedder)
        else:
            tool_defs = self.tools.get_definitions()

        # Ensure triggered skill tools are always included
        if _skill_tool_names:
            existing_names = {td["name"] for td in tool_defs}
            for _stn in _skill_tool_names:
                if _stn not in existing_names and _stn in self.tools._tools:
                    tool_defs.append(self.tools._tools[_stn])

        # Internal monologue: pre-planning (stream)
        _tool_calls_log = []
        _tool_results_summary = []
        _plan, tool_defs, model, _effective_max = await self._apply_planning(
            user_input, user_id, system_prompt, tool_defs, model)

        # Track in-flight request for dashboard
        _req_id = await self._track_request_start(
            user_id, user_input[:120], model,
            complexity_score=complexity_score,
            cascade_tier=_cascade_tier)

        full_text = ""

        try:
            for iteration in range(_effective_max):
                # Use streaming API with self-healing fallback
                try:
                    async for delta in self.provider.stream(
                        model=model,
                        max_tokens=4096,
                        system=system_prompt,
                        tools=tool_defs,
                        messages=messages,
                    ):
                        full_text += delta
                        yield delta
                except Exception as e:
                    if self._is_model_error(e):
                        # Model not found — fall back to default_model
                        logger.warning("Model '%s' not found, falling back to '%s'",
                                       model, self.default_model)
                        model = self.default_model
                        continue
                    elif self._is_fatal_error(e) or self._is_switchable_error(e):
                        fallback = self._get_fallback_provider()
                        if fallback:
                            fb_name, fb_model = fallback
                            logger.warning("Self-healing stream: %s → %s (%s)",
                                           self.config.get("agent", {}).get("provider"), fb_name, e)
                            await self._switch_provider(fb_name, fb_model)
                            model = fb_model
                            yield f"\n⚡ Switched to {fb_name} ({fb_model}) — retrying...\n"
                            # Retry with new provider
                            async for delta in self.provider.stream(
                                model=model, max_tokens=4096,
                                system=system_prompt, tools=tool_defs, messages=messages,
                            ):
                                full_text += delta
                                yield delta
                        else:
                            yield f"\n❌ Error: {e}\n"
                            return
                    else:
                        yield f"\n❌ Error: {e}\n"
                        return

                response = self.provider._last_stream_response

                cost = self._calculate_cost(model, response.usage)
                self.memory.track_usage(user_id, model, response.usage, cost)

                if response.stop_reason == "tool_use":
                    # Execute tools one by one with structured events
                    tool_blocks = [b for b in response.content if b.type == "tool_use"]
                    tool_results = []
                    for block in tool_blocks:
                        # Signal tool start
                        yield f"\n__TOOL_START__{json.dumps({'name': block.name, 'input': block.input, 'id': block.id}, default=str)}__TOOL_END__\n"
                        # Execute and collect result
                        result = await self.tools.execute_one(block)
                        tool_results.append(result)
                        meta = result.get("_meta", {})
                        # Signal tool result
                        yield f"\n__TOOL_RESULT__{json.dumps({'name': meta.get('tool_name', block.name), 'id': block.id, 'duration_ms': meta.get('duration_ms', 0), 'error': meta.get('error', False), 'preview': meta.get('result_preview', '')[:300]}, default=str)}__TOOL_END__\n"

                    # Strip _meta before sending to LLM
                    clean_results = [{k: v for k, v in r.items() if k != "_meta"} for r in tool_results]
                    messages.append({"role": "assistant", "content": _serialize_content(response.content)})
                    messages.append({"role": "user", "content": clean_results})
                    # Track for planning reflection (stream)
                    for block in tool_blocks:
                        _tool_calls_log.append({"name": block.name, "input": block.input})
                    for r in clean_results:
                        content = r.get("content", "") if isinstance(r, dict) else str(r)
                        _tool_results_summary.append(str(content)[:200])
                    # Mid-loop reflection (stream)
                    await self._apply_reflection(
                        messages, _plan, _tool_calls_log, _tool_results_summary)
                    full_text = ""  # Reset for next iteration
                else:
                    # ── Fallback: parse tool call from plain text (stream) ──
                    parsed_tool = self._try_parse_text_tool_call(full_text, tool_defs)
                    if parsed_tool:
                        block = ToolUseBlock(
                            id=f"fallback_s{iteration}",
                            name=parsed_tool["name"],
                            input=parsed_tool["arguments"],
                        )
                        logger.info("Fallback tool call (stream) parsed: %s", block.name)
                        yield f"\n__TOOL_START__{json.dumps({'name': block.name, 'input': block.input, 'id': block.id}, default=str)}__TOOL_END__\n"
                        result = await self.tools.execute_one(block)
                        meta = result.get("_meta", {})
                        yield f"\n__TOOL_RESULT__{json.dumps({'name': meta.get('tool_name', block.name), 'id': block.id, 'duration_ms': meta.get('duration_ms', 0), 'error': meta.get('error', False), 'preview': meta.get('result_preview', '')[:300]}, default=str)}__TOOL_END__\n"
                        clean_results = [{k: v for k, v in r.items() if k != "_meta"} for r in [result]]
                        messages.append({"role": "assistant", "content": _serialize_content([block])})
                        messages.append({"role": "user", "content": clean_results})
                        full_text = ""
                        continue

                    # User message already persisted at stream start
                    self.memory.add_message(user_id, "assistant", full_text)

                    # Post-response hooks (log_interaction, confidence, style, skills)
                    try:
                        await self._post_response_hooks(
                            full_text, user_input, user_id, model,
                            system_prompt, tool_defs, messages, _tool_calls_log)
                    except Exception as e:
                        logger.debug("Stream post-response hooks error: %s", e)

                    # Restore provider after cross-provider cascade
                    if _provider_switched:
                        self._cascade_restore_provider()

                    task = asyncio.create_task(self._safe_extract(user_input, full_text, user_id, file_meta=_file_metas))
                    self._background_tasks.add(task)
                    task.add_done_callback(self._background_tasks.discard)
                    return

            # Restore provider even on max iterations
            if _provider_switched:
                self._cascade_restore_provider()
            yield "\n⚠️ Max iterations reached."
        finally:
            await self._track_request_end(_req_id)
            # Safety: always restore provider if still switched (e.g., on exception)
            if hasattr(self, '_original_provider'):
                self._cascade_restore_provider()

    # ══════════════════════════════════════════
    # SELF-HEALING: PROVIDER FALLBACK
    # ══════════════════════════════════════════

    _FATAL_ERRORS = ("authentication", "auth", "401", "permission", "forbidden", "403")
    _SWITCHABLE_ERRORS = ("rate", "limit", "429", "quota", "overloaded", "503", "capacity")
    _MODEL_ERRORS = ("not found", "404", "does not exist", "no such model", "unknown model")

    def _get_fallback_provider(self) -> tuple[str, str] | None:
        """Find an alternative provider with a saved key. Returns (name, model) or None."""
        from .config import get_api_key
        from .providers import PROVIDER_MODELS
        current = self.config.get("agent", {}).get("provider", "anthropic")
        # Prefer providers in this order
        _FALLBACK_ORDER = ["anthropic", "openai", "gemini", "ollama"]
        for name in _FALLBACK_ORDER:
            if name == current:
                continue
            key = get_api_key(name)
            if key or name == "ollama":
                # Check SDK availability
                _SDK = {"anthropic": "anthropic", "openai": "openai",
                        "gemini": "google.generativeai", "ollama": "openai"}
                try:
                    __import__(_SDK.get(name, name))
                except ImportError:
                    continue
                models = PROVIDER_MODELS.get(name, [])
                default_model = models[0] if models else "gpt-4o-mini"
                return (name, default_model)
        return None

    async def _switch_provider(self, provider_name: str, model: str):
        """Switch to a fallback provider at runtime (serialized via lock)."""
        LiteAgent._ensure_locks()
        async with LiteAgent._provider_lock:
            import os
            from .config import get_api_key, PROVIDER_ENV_VARS
            key = get_api_key(provider_name)
            env_var = PROVIDER_ENV_VARS.get(provider_name)
            if key and env_var:
                os.environ[env_var] = key
            self.config.setdefault("agent", {})["provider"] = provider_name
            self.config["agent"]["default_model"] = model
            self.provider = create_provider(self.config)
            self.default_model = model
            logger.info("Self-healing: switched to provider %s / %s", provider_name, model)

    def _is_fatal_error(self, e: Exception) -> bool:
        """Check if error is non-retryable (bad key, permission denied)."""
        err_str = f"{type(e).__name__} {e}".lower()
        return any(kw in err_str for kw in self._FATAL_ERRORS)

    def _is_switchable_error(self, e: Exception) -> bool:
        """Check if error suggests switching provider (rate limit, quota, overloaded)."""
        err_str = f"{type(e).__name__} {e}".lower()
        return any(kw in err_str for kw in self._SWITCHABLE_ERRORS)

    def _is_model_error(self, e: Exception) -> bool:
        """Check if error is a model-not-found error (wrong model name)."""
        err_str = f"{type(e).__name__} {e}".lower()
        return any(kw in err_str for kw in self._MODEL_ERRORS)

    # ══════════════════════════════════════════
    # API CALL WITH RETRY + FALLBACK
    # ══════════════════════════════════════════

    async def _call_api(self, **kwargs) -> "LLMResponse":
        """Call LLM provider with retry, circuit breaker, and provider fallback."""
        provider_name = self.config.get("agent", {}).get("provider", "anthropic")

        # Circuit breaker: check if current provider is available
        if not self._circuit_breaker.can_call(provider_name):
            fallback = self._get_fallback_provider()
            if fallback:
                fb_name, fb_model = fallback
                if self._circuit_breaker.can_call(fb_name):
                    logger.warning("Circuit breaker: %s unavailable, routing to %s",
                                   provider_name, fb_name)
                    await self._switch_provider(fb_name, fb_model)
                    kwargs["model"] = fb_model
                    provider_name = fb_name
                    # Emit hook for alerting
                    await self.hooks.emit("on_provider_switch", HookContext(
                        agent=self, model=fb_model,
                        extra={"from": provider_name, "to": fb_name, "reason": "circuit_breaker"}))

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await self.provider.complete(**kwargs)
                self._circuit_breaker.record_success(provider_name)
                return result
            except Exception as e:
                self._circuit_breaker.record_failure(provider_name, e)

                # Model not found → fall back to default_model and retry
                if self._is_model_error(e):
                    logger.warning("Model '%s' not found, falling back to '%s'",
                                   kwargs.get("model"), self.default_model)
                    kwargs["model"] = self.default_model
                    continue

                # Fatal error (auth) → try fallback provider immediately
                if self._is_fatal_error(e) or self._is_switchable_error(e):
                    fallback = self._get_fallback_provider()
                    if fallback:
                        fb_name, fb_model = fallback
                        logger.warning("Self-healing: %s failed (%s), switching to %s",
                                       self.config.get("agent", {}).get("provider"), e, fb_name)
                        await self._switch_provider(fb_name, fb_model)
                        kwargs["model"] = fb_model
                        provider_name = fb_name
                        continue
                    raise

                err_name = type(e).__name__
                retryable = any(kw in err_name.lower() for kw in
                                ("rate", "timeout", "connection", "server", "503", "429"))
                if not retryable or attempt == max_retries - 1:
                    raise
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning("API call failed (attempt %d/%d): %s. Retrying in %.1fs",
                               attempt + 1, max_retries, e, wait)
                await asyncio.sleep(wait)

    # ══════════════════════════════════════════
    # CONTEXT BUILDING
    # ══════════════════════════════════════════

    def _ensure_onboarding_tool(self):
        """Register/unregister onboarding tool based on state."""
        from .onboarding import needs_onboarding, register_onboarding_tool, unregister_onboarding_tool
        if needs_onboarding(self):
            if "setup_agent" not in self.tools._tools:
                register_onboarding_tool(self)
                logger.info("Onboarding tool registered")
        else:
            if "setup_agent" in self.tools._tools:
                unregister_onboarding_tool(self)
                logger.info("Onboarding tool unregistered")

    def _build_system_prompt(self, user_input: str, user_id: str) -> str | list[dict]:
        """Build system prompt with memories + feature injections."""
        # Onboarding mode — return special prompt
        from .onboarding import needs_onboarding, ONBOARDING_PROMPT
        if needs_onboarding(self):
            return ONBOARDING_PROMPT

        # Recall relevant memories
        memories = self.memory.recall(user_input, user_id, top_k=5)
        memory_section = ""
        if memories:
            memory_lines = [f"- {m['content']}" for m in memories if m['score'] > 0.1]
            if memory_lines:
                memory_section = "\n\n## What you know about this user:\n" + "\n".join(memory_lines)

        # Current date/time injection
        tz_name = self.config.get("agent", {}).get("timezone", "UTC")
        try:
            import zoneinfo
            tz = zoneinfo.ZoneInfo(tz_name)
        except Exception:
            # Fallback: parse simple offset like "UTC+3"
            offset_h = 0
            if "+" in tz_name:
                try: offset_h = int(tz_name.split("+")[1])
                except ValueError: pass
            elif "-" in tz_name and tz_name != "UTC":
                try: offset_h = -int(tz_name.split("-")[1])
                except ValueError: pass
            tz = timezone(timedelta(hours=offset_h))
        now = datetime.now(tz)
        weekdays_ru = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        time_section = f"\n\n## Текущее время\n{now.strftime('%Y-%m-%d %H:%M')} ({weekdays_ru[now.weekday()]}), часовой пояс: {tz_name}"

        # Feature injections (dynamic, not cached)
        feature_section = self._build_feature_section(user_input, user_id)

        # Skills catalog (static, cacheable — only name + description per skill)
        skills_cfg = self.config.get("skills", {})
        catalog = self.skill_registry.get_catalog_prompt(
            max_chars=skills_cfg.get("max_catalog_chars", 5000))

        dynamic_text = memory_section + time_section + feature_section

        if self.prompt_caching:
            # Soul + skills catalog → cached together (both static between requests)
            cached_text = self._soul_prompt
            if catalog:
                cached_text += catalog
            blocks = [
                {
                    "type": "text",
                    "text": cached_text,
                    "cache_control": {"type": "ephemeral"},  # Static part cached
                },
            ]
            if dynamic_text.strip():
                blocks.append({
                    "type": "text",
                    "text": dynamic_text,  # Dynamic — not cached
                })
            return blocks
        else:
            return self._soul_prompt + (catalog or "") + dynamic_text

    def _build_feature_section(self, user_input: str, user_id: str) -> str:
        """Build feature injections for system prompt."""
        parts = []

        # Style adaptation
        if self._features.get("style_adaptation", {}).get("enabled"):
            from .evolution import get_style_instruction
            style = get_style_instruction(self.memory.db, user_id)
            if style:
                parts.append(style)

        # Applied prompt patches (self-evolving prompt)
        if self._features.get("self_evolving_prompt", {}).get("enabled"):
            from .evolution import get_active_patches
            patches = get_active_patches(self.memory.db)
            if patches:
                parts.append("\n\n## Learned behaviors:\n"
                             + "\n".join(f"- {p}" for p in patches))

        # Proactive suggestions
        if self._features.get("proactive_agent", {}).get("enabled"):
            from .evolution import detect_patterns
            suggestions = detect_patterns(
                self.memory.db, user_id, user_input,
                self._features["proactive_agent"])
            if suggestions:
                parts.append("\n\n## Proactive suggestions (offer if relevant):\n"
                             + "\n".join(f"- {s}" for s in suggestions))

        # Skill suggestions
        if self._features.get("skill_crystallization", {}).get("enabled"):
            from .synthesis import find_matching_skills, format_skill_suggestion
            skills = find_matching_skills(self.memory.db, user_input)
            skill_text = format_skill_suggestion(skills)
            if skill_text:
                parts.append(skill_text)

        # Skill system — inject triggered skill bodies (progressive disclosure)
        skills_cfg = self.config.get("skills", {})
        triggered = self.skill_registry.get_triggered_prompt(
            user_input,
            max_chars=skills_cfg.get("max_triggered_chars", 10000))
        if triggered:
            parts.append("\n\n" + triggered)

        return "".join(parts)

    # ══════════════════════════════════════════
    # INTERNAL MONOLOGUE (PLANNING)
    # ══════════════════════════════════════════

    async def _apply_planning(self, user_input: str, user_id: str,
                              system_prompt, tool_defs: list, model: str,
                              ) -> tuple:
        """Apply internal-monologue planning before the agent loop.

        Returns ``(plan, tool_defs, model, effective_max_iterations)``.
        All arguments may be modified; ``system_prompt`` is mutated in place.
        On any error the original values are returned unchanged.
        """
        im_cfg = self._features.get("internal_monologue", {})
        if not im_cfg.get("enabled"):
            return None, tool_defs, model, self.max_iterations

        try:
            from .planning import generate_plan, format_plan_for_prompt

            # Pass default_model so resolve_planning_model can use it for Ollama
            plan_cfg = dict(im_cfg)
            plan_cfg["_default_model"] = self.default_model

            plan = await generate_plan(
                self.provider, user_input,
                self.memory.recall(user_input, user_id, top_k=3),
                tool_defs, plan_cfg)

            if not plan:
                return None, tool_defs, model, self.max_iterations

            # 1. Inject plan text into system prompt
            plan_text = format_plan_for_prompt(plan)
            if isinstance(system_prompt, list):
                system_prompt[-1]["text"] += plan_text
            else:
                system_prompt += plan_text

            # 2. Cap iterations (×1.5 buffer, minimum 2, capped by max_iterations)
            effective_max = self.max_iterations
            est = plan.get("estimated_iterations")
            if est and isinstance(est, int) and est > 0:
                effective_max = min(int(est * 1.5) + 1, self.max_iterations)
                effective_max = max(effective_max, 2)  # at least 2 iterations

            # 3. Filter tools to planned set + memory_search (fallback: keep all)
            planned_tools = plan.get("tools_needed")
            if planned_tools and isinstance(planned_tools, list):
                allowed = set(planned_tools) | {"memory_search"}
                filtered = [t for t in tool_defs if t.get("name") in allowed]
                if filtered:
                    tool_defs = filtered
                    logger.debug("Planning: filtered tools to %s",
                                 [t["name"] for t in filtered])

            # 4. Upgrade model for complex tasks
            if plan.get("complexity") == "complex" and self.cascade_routing:
                model = self.models.get("complex", model)

            return plan, tool_defs, model, effective_max

        except Exception as e:
            logger.debug("Internal monologue error: %s", e)
            return None, tool_defs, model, self.max_iterations

    async def _apply_reflection(self, messages: list, plan: dict,
                                tool_calls_log: list,
                                tool_results_summary: list) -> None:
        """Apply mid-loop planning reflection.

        Merges adjustment note into the *last* user message (tool_results)
        to avoid breaking Anthropic's alternating-message requirement.
        """
        im_cfg = self._features.get("internal_monologue", {})
        if not (plan and im_cfg.get("enabled")):
            return

        reflect_every = im_cfg.get("reflect_every_n_tools", 3)
        if not (len(tool_calls_log) % reflect_every == 0
                and len(tool_calls_log) > 0):
            return

        try:
            from .planning import reflect_on_progress
            # Pass default_model for resolve_planning_model
            ref_cfg = dict(im_cfg)
            ref_cfg["_default_model"] = self.default_model

            adjustment = await reflect_on_progress(
                self.provider, plan, tool_calls_log,
                tool_results_summary, ref_cfg)
            if adjustment:
                # Merge into last user message to avoid breaking alternating-role requirement
                last_msg = messages[-1]
                reflection_text = f"\n[Internal reflection: {adjustment}]"
                if last_msg.get("role") == "user":
                    if isinstance(last_msg.get("content"), list):
                        last_msg["content"].append({
                            "type": "text",
                            "text": reflection_text,
                        })
                    elif isinstance(last_msg.get("content"), str):
                        last_msg["content"] += reflection_text
                    else:
                        last_msg["content"] = [
                            {"type": "text", "text": str(last_msg.get("content", ""))},
                            {"type": "text", "text": reflection_text},
                        ]
                else:
                    # Last message is assistant — safe to append a new user message
                    messages.append({"role": "user", "content": [
                        {"type": "text", "text": reflection_text}
                    ]})
        except Exception as e:
            logger.debug("Reflection error: %s", e)

    # ══════════════════════════════════════════
    # CASCADE MODEL ROUTING
    # ══════════════════════════════════════════

    def _complexity_score(self, user_input: str) -> int:
        """Score query complexity (0=trivial, 1-2=medium, 3+=complex)."""
        text = user_input.lower()
        score = 0

        # Length heuristic
        if len(user_input) > 500:
            score += 2
        elif len(user_input) > 100:
            score += 1

        # Keyword markers — accumulate from complex set
        complex_hits = sum(1 for m in COMPLEXITY_MARKERS_COMPLEX if m in text)
        score += complex_hits * 2

        # Medium markers (only if no complex hits)
        if complex_hits == 0:
            medium_hits = sum(1 for m in COMPLEXITY_MARKERS_MEDIUM if m in text)
            score += medium_hits

        # Short simple questions
        if text.endswith("?") and len(user_input) < 80 and complex_hits == 0:
            score -= 1

        return score

    def _is_local_only_hours(self) -> bool:
        """Check if current time falls within local-only hours schedule.

        Config: cost.local_only_hours = {enabled, start: "HH:MM", end: "HH:MM"}
        During these hours, only local (Ollama) models are used.
        """
        schedule = self.config.get("cost", {}).get("local_only_hours", {})
        if not schedule.get("enabled"):
            return False

        try:
            tz_name = self.config.get("agent", {}).get("timezone")
            if tz_name:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo(tz_name))
            else:
                now = datetime.now()

            start_h, start_m = map(int, schedule["start"].split(":"))
            end_h, end_m = map(int, schedule["end"].split(":"))
            current = now.hour * 60 + now.minute
            start = start_h * 60 + start_m
            end = end_h * 60 + end_m

            if start <= end:
                return start <= current < end
            else:
                # Overnight range (e.g. 22:00 → 08:00)
                return current >= start or current < end
        except Exception as e:
            logger.debug("local_only_hours check error: %s", e)
            return False

    def _model_for_score(self, score: int) -> str:
        """Pick model based on complexity score.

        Models can use 'provider:model' format for cross-provider cascade.
        E.g. 'ollama:qwen2.5:latest' or 'anthropic:claude-haiku-4-5-20251001'.
        If no provider prefix, uses the current provider.

        During local_only_hours, forces local models only (strips cloud providers).
        """
        local_only = self._is_local_only_hours()

        if score >= 3:
            candidate = self.models.get("complex", self.default_model)
        elif score >= 1:
            candidate = self.models.get("medium", self.default_model)
        else:
            candidate = self.models.get("simple", self.default_model)

        # Local-only mode: force local model, skip cloud providers
        if local_only:
            if ":" in candidate and candidate.split(":")[0] in ("anthropic", "openai", "gemini"):
                # Use medium local model for complex tasks, or default
                fallback = self.models.get("medium", self.default_model)
                if ":" in fallback and fallback.split(":")[0] in ("anthropic", "openai", "gemini"):
                    fallback = self.default_model
                logger.info("Local-only hours: %s → %s", candidate, fallback)
                return fallback

        # Cross-provider cascade: 'provider:model' format
        if ":" in candidate and candidate.split(":")[0] in ("anthropic", "openai", "gemini", "ollama", "qwen", "grok"):
            parts = candidate.split(":", 1)
            target_provider = parts[0]
            target_model = parts[1]
            current_provider = self.config.get("agent", {}).get("provider", "anthropic")

            if target_provider != current_provider:
                # Switch provider temporarily for this call
                self._cascade_switch_provider(target_provider)
                logger.info("Cascade: switching to %s/%s for score=%d",
                            target_provider, target_model, score)
            return target_model

        # Guard: if cascade model doesn't match provider, fall back to default_model
        if not self.provider.supports_model(candidate):
            logger.warning("Cascade model '%s' not supported by provider, using default '%s'",
                           candidate, self.default_model)
            return self.default_model
        return candidate

    def _cascade_switch_provider(self, provider_name: str):
        """Temporarily switch provider for cascade routing (preserves config)."""
        import os
        from .config import get_api_key, PROVIDER_ENV_VARS

        # Cache the original provider for restoration
        if not hasattr(self, '_original_provider'):
            self._original_provider = self.provider
            self._original_provider_name = self.config.get("agent", {}).get("provider")

        key = get_api_key(provider_name)
        env_var = PROVIDER_ENV_VARS.get(provider_name)
        if key and env_var:
            os.environ[env_var] = key

        # Temporarily change provider in config for create_provider
        saved_provider = self.config.get("agent", {}).get("provider")
        self.config.setdefault("agent", {})["provider"] = provider_name
        self.provider = create_provider(self.config)
        # Restore config (we only changed the runtime provider, not the config)
        self.config["agent"]["provider"] = saved_provider

    def _cascade_restore_provider(self):
        """Restore original provider after cascade switch."""
        if hasattr(self, '_original_provider'):
            self.provider = self._original_provider
            del self._original_provider
            del self._original_provider_name

    def _select_model(self, user_input: str) -> str:
        """Route to Haiku/Sonnet/Opus based on query complexity."""
        return self._model_for_score(self._complexity_score(user_input))

    @staticmethod
    def _tier_for_score(score: int) -> str:
        """Map complexity score to cascade tier name."""
        if score >= 3:
            return "complex"
        elif score >= 1:
            return "medium"
        return "simple"

    @classmethod
    def _record_cascade_decision(cls, model: str, tier: str, score: int):
        """Record a cascade routing decision for dashboard visualization."""
        cls._cascade_history.append({
            "model": model,
            "tier": tier,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(cls._cascade_history) > cls._CASCADE_HISTORY_MAX:
            cls._cascade_history = cls._cascade_history[-cls._CASCADE_HISTORY_MAX:]

    @classmethod
    def get_cascade_history(cls) -> list:
        """Return recent cascade decisions for dashboard."""
        return list(cls._cascade_history)

    @classmethod
    def get_cascade_summary(cls) -> dict:
        """Aggregate cascade stats for today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_decisions = [d for d in cls._cascade_history if d["timestamp"].startswith(today)]
        tier_counts = {"simple": 0, "medium": 0, "complex": 0}
        for d in today_decisions:
            t = d.get("tier", "medium")
            tier_counts[t] = tier_counts.get(t, 0) + 1
        last = cls._cascade_history[-1] if cls._cascade_history else None
        return {
            "tier_counts": tier_counts,
            "total_decisions": len(today_decisions),
            "last_decision": last,
        }

    # ══════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════

    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from API response."""
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def _try_parse_text_tool_call(self, text: str, tool_defs: list) -> dict | None:
        """Try to parse a tool call from plain text (fallback for models without structured tool_use).

        Some models (e.g. Ollama/qwen) output tool calls as text with single quotes
        instead of structured tool_use blocks. This method tries multiple parsing strategies.

        Returns {"name": ..., "arguments": ...} or None.
        """
        import ast, re
        text = text.strip()
        obj = None

        # Strategy 1: standard JSON (double quotes)
        try:
            obj = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Strategy 2: Python literal (single quotes — common with Ollama models)
        if obj is None:
            try:
                obj = ast.literal_eval(text)
            except Exception:
                pass

        # Strategy 3: extract from markdown code block
        if obj is None:
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                try:
                    obj = json.loads(m.group(1))
                except (json.JSONDecodeError, ValueError):
                    try:
                        obj = ast.literal_eval(m.group(1))
                    except Exception:
                        pass

        # Strategy 4: replace single quotes → double quotes
        if obj is None and text.startswith("{"):
            try:
                obj = json.loads(text.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                pass

        if obj is None:
            return None

        # Validate: must have "name" and "arguments"
        if not isinstance(obj, dict):
            return None
        name = obj.get("name")
        arguments = obj.get("arguments")
        if not name or not isinstance(arguments, dict):
            return None

        # Must match a known tool
        known_names = {t["name"] for t in tool_defs}
        if name not in known_names:
            # Fuzzy match for MCP tools: model may output 'transcribe_voice_file'
            # but real tool is 'mywhisper__transcribe_voice_file'
            matched = [kn for kn in known_names if kn.endswith(f"__{name}")]
            if matched:
                name = matched[0]
                logger.info("Fuzzy-matched tool name: %s", name)
            else:
                return None

        return {"name": name, "arguments": arguments}

    @staticmethod
    def _calculate_cost(model: str, usage) -> float:
        """Calculate cost in USD (includes cache read + cache creation)."""
        pricing = get_pricing(model)
        input_cost = getattr(usage, 'input_tokens', 0) * pricing["input"] / 1_000_000
        output_cost = getattr(usage, 'output_tokens', 0) * pricing["output"] / 1_000_000
        cache_read_cost = getattr(usage, 'cache_read_input_tokens', 0) * pricing["cache_read"] / 1_000_000
        # Cache creation costs same as input tokens
        cache_create_cost = getattr(usage, 'cache_creation_input_tokens', 0) * pricing["input"] / 1_000_000
        return input_cost + output_cost + cache_read_cost + cache_create_cost

    async def _escalated_run(self, model: str, system_prompt,
                              tool_defs, messages) -> str:
        """Re-run with a better model after confidence gate triggers."""
        logger.info("Confidence gate escalating to %s", model)
        response = await self._call_api(
            model=model, max_tokens=4096, system=system_prompt,
            tools=tool_defs, messages=messages)
        cost = self._calculate_cost(model, response.usage)
        self.memory.track_usage(self._current_user_id, model,
                                response.usage, cost)
        return self._extract_text(response)

    async def _safe_extract(self, user_input: str, response: str, user_id: str,
                            file_meta: list | None = None):
        """Safe wrapper for knowledge extraction — never crashes."""
        try:
            await self.memory.extract_and_learn(user_input, response, user_id,
                                                file_meta=file_meta)
        except Exception as e:
            logger.warning("Knowledge extraction failed: %s", e)

    # ══════════════════════════════════════════
    # AGENT COMMANDS (meta)
    # ══════════════════════════════════════════

    def handle_command(self, command: str, user_id: str = "default") -> str | None:
        """Handle special /commands. Returns response or None if not a command."""
        cmd = command.strip().lower()

        if cmd == "/memories":
            memories = self.memory.get_all_memories(user_id)
            if not memories:
                return "No memories stored yet."
            lines = [f"📝 {len(memories)} memories:\n"]
            for m in memories[:20]:
                lines.append(f"  [{m['type']}] {m['content'][:80]}  (imp: {m['importance']:.1f})")
            return "\n".join(lines)

        elif cmd == "/usage":
            summary = self.memory.get_usage_summary(days=7)
            if not summary:
                return "No usage data yet."
            today_cost = self.memory.get_today_cost()
            lines = [f"💰 Today: ${today_cost:.4f} / ${self.budget_daily:.2f}\n",
                     "Last 7 days:"]
            for s in summary:
                lines.append(f"  {s['model']}: {s['calls']} calls, "
                             f"{s['input_tokens']+s['output_tokens']:,} tokens, "
                             f"${s['cost_usd']:.4f}")
            return "\n".join(lines)

        elif cmd == "/clear":
            self.memory.clear_conversation(user_id)
            return "🗑️ Conversation cleared."

        elif cmd.startswith("/forget "):
            fragment = command[8:].strip()
            self.memory.forget(user_id, fragment)
            return f"🗑️ Forgotten memories matching: {fragment}"

        elif cmd.startswith("/ingest "):
            path = command[8:].strip()
            if not self._rag:
                return "⚠️ RAG is not enabled. Set `rag.enabled: true` in config."
            try:
                result = self._rag.ingest(path)
                if "files" in result:
                    errors = f"\nErrors: {result['errors']}" if result.get("errors") else ""
                    return (f"📄 Ingested {result['files']} files, "
                            f"{result['chunks']} chunks.{errors}")
                return f"📄 {result['path']}: {result['status']} ({result['chunks']} chunks)"
            except Exception as e:
                return f"❌ Ingest error: {e}"

        elif cmd == "/documents":
            if not self._rag:
                return "⚠️ RAG is not enabled. Set `rag.enabled: true` in config."
            docs = self._rag.list_documents()
            if not docs:
                return "No documents ingested yet. Use /ingest <path> to add files."
            lines = [f"📚 {len(docs)} documents:\n"]
            for d in docs:
                lines.append(f"  [{d['id']}] {d['name']} — {d['chunks']} chunks")
            return "\n".join(lines)

        elif cmd == "/conflicts":
            archived = self.memory.get_archived_memories(user_id, limit=20)
            if not archived:
                return "No memory conflicts resolved yet."
            lines = [f"🔀 {len(archived)} archived memories (conflict resolutions):\n"]
            for m in archived:
                lines.append(f"  [{m['type']}] {m['content'][:80]}  (archived: {m['archived_at'][:10]})")
            return "\n".join(lines)

        elif cmd == "/model" or cmd.startswith("/model "):
            return self._handle_model_command(command)

        elif cmd == "/help":
            help_text = (
                "Commands:\n"
                "  /model      — Show/switch models\n"
                "  /memories   — Show stored memories\n"
                "  /usage      — Show token usage and costs\n"
                "  /clear      — Clear conversation history\n"
                "  /forget X   — Forget memories matching X\n"
                "  /conflicts  — Show resolved memory conflicts\n"
            )
            if self._rag:
                help_text += (
                    "  /ingest X   — Ingest file or directory into RAG\n"
                    "  /documents  — List ingested documents\n"
                )
            help_text += "  /help       — This message"
            return help_text

        return None  # Not a command

    # ── /model command ────────────────────────────────────────

    def _handle_model_command(self, command: str) -> str:
        """Handle /model command — show or switch models.

        /model              — show current model + cascade tiers + available models
        /model <name>       — set default model
        /model simple <name> — set cascade tier model
        /model medium <name>
        /model complex <name>
        """
        from .providers import PROVIDER_MODELS

        parts = command.strip().split(maxsplit=2)
        # parts[0] = "/model"

        # ── /model (no args) → show info ──
        if len(parts) == 1:
            return self._model_status(PROVIDER_MODELS)

        arg1 = parts[1].strip().lower()

        # ── /model simple|medium|complex <name> → set cascade tier ──
        if arg1 in ("simple", "medium", "complex"):
            if len(parts) < 3:
                current = self.models.get(arg1, "—")
                return f"⚙️ Current {arg1} model: {current}\n\nUsage: /model {arg1} <model_name>"
            model_name = parts[2].strip()
            self.models[arg1] = model_name
            self.config.setdefault("agent", {}).setdefault("models", {})[arg1] = model_name
            self._save_model_config()
            return f"✅ {arg1.capitalize()} model → {model_name}"

        # ── /model <name> → set default model ──
        model_name = parts[1].strip()
        # Allow full args in case model name has spaces or extra
        if len(parts) > 2:
            model_name = f"{parts[1].strip()} {parts[2].strip()}"
        model_name = model_name.strip()

        old_model = self.default_model
        self.default_model = model_name
        self.config.setdefault("agent", {})["default_model"] = model_name
        self._save_model_config()
        return f"✅ Default model: {old_model} → {model_name}"

    def _model_status(self, provider_models: dict) -> str:
        """Format current model configuration for display."""
        provider_name = self.config.get("agent", {}).get("provider", "?")
        lines = [
            f"🤖 Provider: {provider_name}",
            f"📌 Default: {self.default_model}",
        ]

        # Cascade tiers
        if self.models:
            lines.append("\n🔀 Cascade routing" + (" ✅" if self.cascade_routing else " ❌") + ":")
            for tier in ("simple", "medium", "complex"):
                m = self.models.get(tier)
                if m:
                    lines.append(f"  {tier}: {m}")

        # Available models for current provider
        available = provider_models.get(provider_name, [])
        if available:
            lines.append(f"\n📋 Available ({provider_name}):")
            for m in available[:15]:
                marker = " ◀" if m == self.default_model else ""
                lines.append(f"  • {m}{marker}")

        lines.append("\nUsage:\n  /model <name> — set default\n  /model simple|medium|complex <name> — set tier")
        return "\n".join(lines)

    def _save_model_config(self):
        """Persist model changes to config file."""
        try:
            from .config import save_config
            save_config(self.config)
            logger.info("Model config saved: default=%s, models=%s",
                        self.default_model, self.models)
        except Exception as e:
            logger.warning("Failed to save model config: %s", e)
