"""Core agent loop with cascade routing, prompt caching, and context compression."""

import asyncio
import itertools
import json
import logging
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

    async def _track_request_start(self, user_id: str, input_preview: str, model: str) -> int:
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
            }
        self._ws_broadcast("request_started", LiteAgent._active_requests.get(req_id, {}))
        return req_id

    async def _track_request_end(self, req_id: int):
        """Remove a completed in-flight request."""
        async with LiteAgent._requests_lock:
            info = LiteAgent._active_requests.pop(req_id, None)
        if info:
            self._ws_broadcast("request_done", {"id": req_id, "user_id": info.get("user_id")})

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
        builtin = tools_cfg.get("builtin", ["read_file", "write_file", "exec_command"])
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

        # Storage backend (MinIO/S3)
        self._storage = None
        if config.get("storage", {}).get("enabled", False):
            from .storage import create_storage
            self._storage = create_storage(config)
            if self._storage:
                self._wire_storage_tools()

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
            self._wire_rag_tool()

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
        """Register transcribe_voice tool — lets the agent transcribe voice messages."""
        agent = self

        async def transcribe_voice_handler(voice_id: str) -> str:
            """Transcribe a voice message by its ID using OpenAI Whisper API."""
            voice_data = agent._voice_store.pop(voice_id, None)
            if not voice_data:
                return f"Voice message '{voice_id}' not found or already transcribed."

            audio_bytes = voice_data["audio_bytes"]
            config = voice_data["config"]

            try:
                import openai
            except ImportError:
                return ("OpenAI SDK is required for voice transcription: "
                        "pip install liteagent[openai]")

            from .config import get_api_key
            api_key = get_api_key("openai")
            if not api_key:
                return ("OpenAI API key required for voice transcription. "
                        "Set it via dashboard Settings → Providers → OpenAI, "
                        "or set OPENAI_API_KEY environment variable.")

            import io
            client = openai.AsyncOpenAI(api_key=api_key)
            model = config.get("voice_model", "whisper-1")
            language = config.get("voice_language")  # None = auto-detect

            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "voice.ogg"

            kwargs = {"model": model, "file": audio_file}
            if language:
                kwargs["language"] = language

            try:
                response = await client.audio.transcriptions.create(**kwargs)
                return response.text
            except Exception as e:
                return f"Voice transcription error: {e}"

        self.tools._tools["transcribe_voice"] = {
            "name": "transcribe_voice",
            "description": (
                "Transcribe a voice message from the user. When a user sends a voice "
                "message (e.g. via Telegram), the audio is stored with a voice_id. "
                "Call this tool with that voice_id to get the text transcription. "
                "You MUST call this tool to understand what the user said in their "
                "voice message before you can respond."
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

    def _wire_storage_tools(self):
        """Register file storage tools when storage backend is available."""
        storage = self._storage
        storage_sandbox = self.config.get("storage", {}).get("sandbox_prefix", "")

        def _validate_storage_key(key: str) -> str | None:
            """Validate storage key: block path traversal and enforce sandbox prefix."""
            # Block path traversal
            if ".." in key or key.startswith("/"):
                return "Access denied: path traversal detected in storage key"
            # Enforce sandbox prefix if configured
            if storage_sandbox and not key.startswith(storage_sandbox):
                return f"Access denied: storage key must start with '{storage_sandbox}'"
            return None

        async def save_file_handler(path: str, content: str) -> str:
            """Save a file to cloud storage (MinIO/S3)."""
            err = _validate_storage_key(path)
            if err:
                return err
            data = content.encode("utf-8")
            key = storage.upload(path, data)
            return f"File saved to storage: {key} ({len(data)} bytes)"

        async def get_file_handler(path: str) -> str:
            """Retrieve a file from cloud storage."""
            err = _validate_storage_key(path)
            if err:
                return err
            try:
                data = storage.download(path)
                return data.decode("utf-8", errors="replace")
            except Exception as e:
                return f"Error retrieving file: {e}"

        async def list_storage_handler(prefix: str = "") -> str:
            """List files in cloud storage."""
            if storage_sandbox and prefix and not prefix.startswith(storage_sandbox):
                return f"Access denied: listing must use prefix '{storage_sandbox}'"
            files = storage.list_files(prefix=prefix or storage_sandbox)
            if not files:
                return "No files found in storage."
            lines = [f"{f['key']} ({f['size']} bytes)" for f in files]
            return "\n".join(lines)

        for name, handler, desc, schema in [
            ("save_file", save_file_handler,
             "Save a file to cloud storage (MinIO/S3). Use for persisting data, documents, exports.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "File path/key in storage"},
                 "content": {"type": "string", "description": "File content (text)"},
             }, "required": ["path", "content"]}),
            ("get_file", get_file_handler,
             "Retrieve a file from cloud storage by its path/key.",
             {"type": "object", "properties": {
                 "path": {"type": "string", "description": "File path/key in storage"},
             }, "required": ["path"]}),
            ("list_storage", list_storage_handler,
             "List files in cloud storage. Optionally filter by prefix.",
             {"type": "object", "properties": {
                 "prefix": {"type": "string", "description": "Filter by prefix (optional)"},
             }, "required": []}),
        ]:
            self.tools._tools[name] = {
                "name": name, "description": desc, "input_schema": schema,
            }
            self.tools._handlers[name] = handler

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
        # Re-register builtin voice tool (may have been removed by previous mode)
        if "transcribe_voice" not in self.tools._tools:
            self._wire_voice_tool()
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
        await self._ensure_mcp_loaded()
        self._ensure_onboarding_tool()
        # Load persisted history on first interaction
        if not self.memory.get_history(user_id):
            self.memory.load_history(user_id)

        # Normalize multimodal input
        if isinstance(user_input, list):
            text_for_memory = " ".join(
                b.get("text", "") for b in user_input if b.get("type") == "text")
            content_for_api = user_input
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
        _provider_switched = hasattr(self, '_original_provider')
        logger.info("Model: %s | User: %s | Input: %d chars | Complexity: %d%s",
                     model, user_id, len(text_for_memory), complexity_score,
                     " [cross-provider]" if _provider_switched else "")

        # Tool selection: skip tools for trivial messages (greetings, short chat)
        if complexity_score <= 0 and len(text_for_memory) < 60:
            tool_defs = []
            logger.debug("Skipping tools for simple message")
        elif self.memory._embedder and len(self.tools._tools) > 8:
            tool_defs = self.tools.get_relevant_definitions(
                text_for_memory, top_k=8, embedder=self.memory._embedder)
        else:
            tool_defs = self.tools.get_definitions()

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
            model)

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
                        self._safe_extract(text_for_memory, text, user_id)
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

        # Persist user message immediately (before streaming starts)
        self.memory.add_message(user_id, "user", user_input)

        system_prompt = self._build_system_prompt(user_input, user_id)
        messages = self.memory.get_compressed_history(user_id)
        messages.append({"role": "user", "content": user_input})
        complexity_score = self._complexity_score(user_input)
        model = self._model_for_score(complexity_score) if self.cascade_routing else self.default_model
        _provider_switched = hasattr(self, '_original_provider')
        logger.info("Stream | Model: %s | User: %s | Complexity: %d%s",
                     model, user_id, complexity_score,
                     " [cross-provider]" if _provider_switched else "")

        # Skip tools for trivial messages
        if complexity_score <= 0 and len(user_input) < 60:
            tool_defs = []
        elif self.memory._embedder and len(self.tools._tools) > 8:
            tool_defs = self.tools.get_relevant_definitions(
                text_for_memory, top_k=8, embedder=self.memory._embedder)
        else:
            tool_defs = self.tools.get_definitions()

        # Internal monologue: pre-planning (stream)
        _tool_calls_log = []
        _tool_results_summary = []
        _plan, tool_defs, model, _effective_max = await self._apply_planning(
            user_input, user_id, system_prompt, tool_defs, model)

        # Track in-flight request for dashboard
        _req_id = await self._track_request_start(user_id, user_input[:120], model)

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

                    task = asyncio.create_task(self._safe_extract(user_input, full_text, user_id))
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

        dynamic_text = memory_section + time_section + feature_section

        if self.prompt_caching:
            blocks = [
                {
                    "type": "text",
                    "text": self._soul_prompt,
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
            return self._soul_prompt + dynamic_text

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
        if ":" in candidate and candidate.split(":")[0] in ("anthropic", "openai", "gemini", "ollama"):
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

    async def _safe_extract(self, user_input: str, response: str, user_id: str):
        """Safe wrapper for knowledge extraction — never crashes."""
        try:
            await self.memory.extract_and_learn(user_input, response, user_id)
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
