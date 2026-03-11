"""Core agent loop with cascade routing, prompt caching, and context compression."""

import asyncio
import copy
import contextvars
import itertools
import json
import logging
import os
import random
import time
from .file_types import detect_file_type
from contextlib import suppress
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
    # Action/execution continuation words — short but tool-requiring
    "реализуй", "реализац", "приступай", "приступи", "выполни", "выполни",
    "запусти", "начни", "сгенерируй", "напиши код", "запусти",
    "execute", "run", "start", "generate", "proceed",
    # Web search / tool-dependent queries → need capable model
    "новости", "найди", "поиск", "погугли", "загугли", "интернет",
    "news", "search", "find", "look up", "google", "browse", "fetch",
    "погода", "weather", "курс", "цена", "price",
}

TOOL_GAP_MARKERS = (
    "создай инструмент", "сделай инструмент", "добавь инструмент",
    "нет инструмента", "если нет инструмента", "доработай инструмент",
    "улучши инструмент", "create tool", "build tool", "missing tool",
    "no tool", "improve tool", "enhance tool",
)

VISION_QUERY_MARKERS = (
    "распоз", "изображ", "картин", "фото", "скриншот", "ocr", "vision",
    "image", "photo", "picture", "screenshot", "describe what you see",
)

VISION_TOOL_NAMES = (
    "vision_analyze_image", "analyze_image", "image_ocr",
    "describe_image", "vision_describe", "image_describe",
)

_REQUEST_TRACKING_CLEAR = object()


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
    _AUTO_ALIAS_IDS = frozenset({"dashboard-user", "api-user", "tg-user"})
    _RESERVED_USER_IDS = frozenset({"", "default", "dashboard-user", "api-user", "tg-user", "system"})

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
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "model": model,
                "input_preview": input_preview[:120],
                "status": "running",
                "phase": "queued",
                "phase_label": "Queued for execution",
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

    async def _update_request_progress(self, req_id: int, **changes) -> dict | None:
        """Update in-flight request telemetry and broadcast a live snapshot."""
        async with LiteAgent._requests_lock:
            info = LiteAgent._active_requests.get(req_id)
            if not info:
                return None
            changed = False
            for key, value in changes.items():
                if value is _REQUEST_TRACKING_CLEAR:
                    if key in info:
                        info.pop(key, None)
                        changed = True
                    continue
                if info.get(key) != value:
                    info[key] = value
                    changed = True
            if not changed:
                return copy.deepcopy(info)
            info["updated_at"] = datetime.now(timezone.utc).isoformat()
            snapshot = copy.deepcopy(info)
        self._ws_broadcast("request_progress", snapshot)
        return snapshot

    async def _update_request_tool_progress(self, req_id: int, event: dict) -> dict | None:
        """Update request telemetry for parallel tool execution."""
        tool_use_id = str(event.get("tool_use_id") or "")
        async with LiteAgent._requests_lock:
            info = LiteAgent._active_requests.get(req_id)
            if not info:
                return None
            children = list(info.get("parallel_children") or [])
            for child in children:
                if str(child.get("tool_use_id") or "") != tool_use_id:
                    continue
                child["status"] = "running" if event.get("event") == "start" else (
                    "error" if event.get("error") else "done"
                )
                if event.get("event") == "done":
                    child["duration_ms"] = int(event.get("duration_ms") or 0)
                    child["error"] = bool(event.get("error"))
                    child["result_preview"] = str(event.get("result_preview") or "")[:220]
                break
            info["parallel_children"] = children
            info["parallel_completed"] = sum(
                1 for child in children if child.get("status") in {"done", "error"}
            )
            info["phase"] = "parallel_tools"
            info["phase_label"] = (
                f"Parallel tools {info['parallel_completed']}/{max(1, int(info.get('parallel_total') or len(children) or 1))}"
            )
            info["progress_label"] = (
                f"Iteration {int(info.get('iteration') or 0)}/{max(1, int(info.get('max_iterations') or 1))} · "
                f"tools {info['parallel_completed']}/{max(1, int(info.get('parallel_total') or len(children) or 1))}"
            )
            info["updated_at"] = datetime.now(timezone.utc).isoformat()
            snapshot = copy.deepcopy(info)
        self._ws_broadcast("request_progress", snapshot)
        return snapshot

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
        self._max_iterations_explicit = "max_iterations" in agent_cfg
        self.max_iterations = agent_cfg.get("max_iterations", 15)
        self.default_model = agent_cfg.get("default_model", "claude-sonnet-4-20250514")
        configured_models = agent_cfg.get("models", {}) or {}
        self.models = {
            **self._provider_default_models(
                str(agent_cfg.get("provider", "anthropic") or "anthropic"),
                self.default_model,
            ),
            **configured_models,
        }
        self._normalize_runtime_model_config()

        # Cost controls
        self.cascade_routing = cost_cfg.get("cascade_routing", True)
        self.prompt_caching = cost_cfg.get("prompt_caching", True)
        self.budget_daily = cost_cfg.get("budget_daily_usd", 5.0)
        self._intelligent_routing_cfg = self._build_intelligent_routing_config(cost_cfg)
        self._last_cascade_route: dict = {}

        # Memory
        self.memory = MemorySystem(config, provider=self.provider)

        # Tools (with security sandbox)
        self.tools = ToolRegistry(config)
        tools_cfg = config.get("tools", {})
        builtin = tools_cfg.get("builtin", ["read_file", "write_file", "exec_command",
                                            "download_file", "send_file_to_user"])
        sandbox_root = tools_cfg.get("sandbox_root")  # None = sensitive-path blocking only
        cmd_allowlist = set(tools_cfg["command_allowlist"]) if "command_allowlist" in tools_cfg else None
        allow_shell = tools_cfg.get("allow_shell", False)
        command_timeout = tools_cfg.get("command_timeout", 120)
        register_builtin_tools(
            self.tools, enabled=builtin + ["memory_search"],
            sandbox_root=sandbox_root,
            command_allowlist=cmd_allowlist,
            allow_shell=allow_shell,
            command_timeout=command_timeout,
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
        # Merge user config over safe defaults (opt-out is possible via "enabled": false)
        _feature_defaults = {
            "style_adaptation": {"enabled": True, "ema_alpha": 0.3},
            "critical_response_review": {
                "enabled": True,
                "min_complexity": 3,
                "min_response_chars": 220,
                "timeout_sec": 20.0,
                "max_issues": 3,
                "max_tool_evidence": 4,
            },
            "human_support_agent": {
                "enabled": True,
                "max_suggestions": 3,
                "min_pattern_occurrences": 3,
                "pattern_window_days": 30,
                "late_night_hour": 23,
                "early_hour": 6,
            },
            "self_evolving_prompt": {
                "enabled": True,
                "min_friction_signals": 5,
                "auto_apply": True,
            },
        }
        _cfg_features = config.get("features", {})
        self._features = {}
        for key, default_val in _feature_defaults.items():
            user_val = _cfg_features.get(key)
            if user_val is None:
                self._features[key] = default_val
            elif isinstance(user_val, dict):
                merged = dict(default_val)
                merged.update(user_val)
                self._features[key] = merged
            else:
                self._features[key] = user_val
        # Add remaining user-configured features (not in defaults)
        for key, val in _cfg_features.items():
            if key not in self._features:
                self._features[key] = val

        # Load synthesized tools if enabled (with execution budgets)
        if self._features.get("auto_tool_synthesis", {}).get("enabled"):
            from .synthesis import (load_synthesized_tools, create_synthesize_meta_tool,
                                    DEFAULT_SYNTH_TIMEOUT_SEC, DEFAULT_SYNTH_MAX_OUTPUT_CHARS)
            synth_cfg = dict(self._features["auto_tool_synthesis"])
            # Practical default for autonomous tooling: create and use in the same run.
            synth_cfg.setdefault("auto_approve", True)
            self._features["auto_tool_synthesis"] = synth_cfg
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

        # Document intelligence workflow: originals -> storage, content -> KB, summary -> notes/tasks.
        self._document_pipeline = None
        try:
            from .documents import DocumentPipeline
            self._document_pipeline = DocumentPipeline(self)
        except Exception as e:
            logger.warning("Document pipeline initialization failed: %s", e)

        # Per-conversation model overrides (from OpenClaw model-overrides.ts)
        from .conv_model import ConversationModelStore
        self._conv_model = ConversationModelStore(self.memory.db)

        # Task manager (set by main.py after scheduler setup)
        self._task_manager = None
        self._background_task_daemon = None
        self._goal_manager = None
        self._goal_coordinator = None
        self._current_chat_id = None
        self._current_chat_id_ctx = contextvars.ContextVar(
            "liteagent_current_chat_id", default=None)
        self._last_response_meta: dict = {}

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

        # Browser automation (Playwright) — disabled by default (heavyweight)
        self._browser_engine = None
        browser_cfg = config.get("browser", {})
        if browser_cfg.get("enabled", False):
            self._wire_browser_tools()

        # Skill system (modular prompt injection with progressive disclosure)
        self.skill_registry = SkillRegistry()
        self.skill_registry.load_all(config)
        self._wire_skill_tools()
        self._wire_status_tool()

        # Slow local models profile (latency-oriented defaults for provider=ollama).
        self._slow_local_mode = self._resolve_slow_local_mode()
        self._slow_local_cfg = self._build_slow_local_profile()
        self._kb_auto_context_cache: dict[str, tuple[float, str]] = {}
        self._user_resolution_cache: dict[str, tuple[float, str]] = {}

        # Cognitive Intelligence features (Socratic, Goals, Cognitive State, Epistemic)
        cog_cfg = self._features.get("cognition", {})
        if cog_cfg.get("enabled", True):
            self._init_cognition()

    @staticmethod
    def _provider_default_models(provider_name: str, default_model: str) -> dict[str, str]:
        """Choose provider-compatible cascade defaults when config.models is absent."""
        provider = str(provider_name or "anthropic").strip().lower()
        default_model = str(default_model or "").strip() or "claude-sonnet-4-20250514"
        if provider == "openai":
            return {
                "simple": "gpt-4o-mini",
                "medium": default_model,
                "complex": "gpt-4o",
            }
        if provider == "gemini":
            return {
                "simple": "gemini-2.5-flash",
                "medium": default_model,
                "complex": "gemini-2.5-pro",
            }
        if provider == "anthropic":
            return {
                "simple": "claude-haiku-4-5-20251001",
                "medium": default_model,
                "complex": "claude-opus-4-20250115",
            }
        return {
            "simple": default_model,
            "medium": default_model,
            "complex": default_model,
        }

    @staticmethod
    def _build_intelligent_routing_config(cost_cfg: dict | None) -> dict:
        """Normalize intelligent cascade routing settings."""
        cfg = dict((cost_cfg or {}).get("intelligent_routing", {}) or {})

        def _int(name: str, default: int, lo: int, hi: int) -> int:
            try:
                value = int(cfg.get(name, default))
            except (TypeError, ValueError):
                value = default
            return max(lo, min(value, hi))

        def _float(name: str, default: float, lo: float, hi: float) -> float:
            try:
                value = float(cfg.get(name, default))
            except (TypeError, ValueError):
                value = default
            return max(lo, min(value, hi))

        return {
            "enabled": bool(cfg.get("enabled", True)),
            "use_llm": bool(cfg.get("use_llm", True)),
            "advisor_model": str(cfg.get("advisor_model", "") or "").strip(),
            "min_complexity": _int("min_complexity", 1, 0, 10),
            "local_min_complexity": _int("local_min_complexity", 2, 0, 10),
            "timeout_sec": _float("timeout_sec", 8.0, 1.0, 30.0),
        }

    def _normalize_model_name_for_provider(self, model_name: str, provider_name: str) -> str:
        """Normalize configured models so bare cross-provider names route correctly."""
        model_name = str(model_name or "").strip()
        provider_name = str(provider_name or "").strip().lower()
        if not model_name:
            return str(self.default_model or "").strip()
        if ":" in model_name:
            prefix, rest = model_name.split(":", 1)
            if prefix.lower() in self._KNOWN_PROVIDERS and rest.strip():
                return model_name

        if provider_name == "ollama":
            return model_name
        if self._model_matches_provider(provider_name, model_name):
            return model_name

        inferred = self._infer_provider_for_model(model_name)
        if inferred and inferred != provider_name:
            logger.warning(
                "Normalizing model '%s' for provider '%s' -> '%s:%s'",
                model_name, provider_name, inferred, model_name,
            )
            return f"{inferred}:{model_name}"

        available = PROVIDER_MODELS.get(provider_name, [])
        if available:
            fallback = available[0]
            logger.warning(
                "Model '%s' is incompatible with provider '%s'; falling back to '%s'",
                model_name, provider_name, fallback,
            )
            return fallback
        return model_name

    def _normalize_runtime_model_config(self) -> None:
        """Normalize default/tier models after init or config reload."""
        provider_name = str(self.config.get("agent", {}).get("provider", "anthropic")).strip().lower()

        if not self._model_matches_provider(provider_name, self.default_model):
            inferred = self._infer_provider_for_model(self.default_model)
            if inferred and inferred != provider_name:
                fallback_models = PROVIDER_MODELS.get(provider_name, [])
                if fallback_models:
                    logger.warning(
                        "Default model '%s' does not match provider '%s'; using '%s' instead",
                        self.default_model, provider_name, fallback_models[0],
                    )
                    self.default_model = fallback_models[0]
            elif provider_name != "ollama":
                fallback_models = PROVIDER_MODELS.get(provider_name, [])
                if fallback_models:
                    self.default_model = fallback_models[0]

        self.models = {
            tier: self._normalize_model_name_for_provider(model, provider_name)
            for tier, model in (self.models or {}).items()
        }

    @staticmethod
    def _model_matches_provider(provider_name: str, model_name: str) -> bool:
        """True when a bare model name belongs to the given provider."""
        provider_name = str(provider_name or "").strip().lower()
        model_name = str(model_name or "").strip().lower()
        if not provider_name or not model_name:
            return False
        if provider_name == "ollama":
            return True
        if ":" in model_name and model_name.split(":", 1)[0] not in LiteAgent._KNOWN_PROVIDERS:
            return False
        if provider_name == "anthropic":
            return model_name.startswith("claude-") or model_name.startswith("anthropic/")
        if provider_name == "openai":
            return model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3")
        if provider_name == "gemini":
            return model_name.startswith("gemini-") or model_name.startswith("models/")
        if provider_name == "qwen":
            return model_name.startswith("qwen")
        if provider_name == "grok":
            return model_name.startswith("grok")
        return False

    def _init_cognition(self) -> None:
        """Initialize cognitive intelligence features and register their hooks."""
        from .cognition import init_cognition_tables
        init_cognition_tables(self.memory.db)
        self._register_cognition_hooks()

    def _register_cognition_hooks(self) -> None:
        """Register all 4 cognitive intelligence hooks."""
        agent = self
        cog_cfg = agent._features.get("cognition", {})

        # ── 1. Socratic Self-Adversary (priority 20 — runs before confidence gate) ──
        soc_cfg = cog_cfg.get("socratic", {})
        if soc_cfg.get("enabled", True):
            min_complexity = soc_cfg.get("min_complexity", 2)
            min_len = soc_cfg.get("min_response_len", 100)

            async def socratic_handler(ctx):
                try:
                    from .cognition import socratic_challenge, _cheapest_model
                    user_text = ctx.extra.get("user_input_text", "")
                    if not user_text or len(ctx.response_text) < min_len:
                        return
                    score = ctx.agent._complexity_score(user_text)
                    if score < min_complexity:
                        return
                    model = _cheapest_model(ctx.agent.provider)
                    revised = await socratic_challenge(
                        ctx.agent.provider, user_text, ctx.response_text, model)
                    if revised != ctx.response_text:
                        ctx.response_text = revised
                except Exception as e:
                    logger.debug("Socratic hook error: %s", e)

            agent.hooks.register("after_response", "socratic_adversary",
                                 socratic_handler, priority=20)

        # ── 2. Goal Inference (priority 35 — background fire-and-forget) ──
        goal_cfg = cog_cfg.get("goal_inference", {})
        if goal_cfg.get("enabled", True):
            max_goals = goal_cfg.get("max_active_goals", 2)

            async def goal_inference_handler(ctx):
                try:
                    from .cognition import infer_session_goal, _cheapest_model
                    user_text = ctx.extra.get("user_input_text", "")
                    if not user_text or len(user_text) < 30:
                        return
                    score = ctx.agent._complexity_score(user_text)
                    if score < 1:
                        return
                    # Build history tail from last 2 messages
                    history_tail = ""
                    msgs = ctx.messages or []
                    for m in msgs[-4:]:
                        role = m.get("role", "")
                        content = m.get("content", "")
                        if isinstance(content, str):
                            history_tail += f"{role}: {content[:100]}\n"
                    model = _cheapest_model(ctx.agent.provider)
                    db = ctx.agent.memory.db
                    # Fire-and-forget background task
                    import asyncio as _asyncio
                    task = _asyncio.ensure_future(
                        infer_session_goal(ctx.agent.provider, user_text,
                                           history_tail, model, db, ctx.user_id))
                    ctx.agent._background_tasks.add(task)
                    task.add_done_callback(ctx.agent._background_tasks.discard)
                except Exception as e:
                    logger.debug("Goal inference hook error: %s", e)

            agent.hooks.register("after_response", "goal_inference",
                                 goal_inference_handler, priority=35)

        # ── 3. Cognitive State update (priority 80 — pure heuristic, very fast) ──
        csa_cfg = cog_cfg.get("cognitive_state", {})
        if csa_cfg.get("enabled", True):
            async def cognitive_state_handler(ctx):
                try:
                    from .cognition import update_cognitive_signal
                    user_text = ctx.extra.get("user_input_text", "")
                    if user_text:
                        update_cognitive_signal(ctx.agent.memory.db,
                                                ctx.user_id, user_text)
                except Exception as e:
                    logger.debug("Cognitive state hook error: %s", e)

            agent.hooks.register("after_response", "cognitive_state_update",
                                 cognitive_state_handler, priority=80)

        # ── 4. Epistemic Calibration (priority 85) ──
        ep_cfg = cog_cfg.get("epistemic", {})
        if ep_cfg.get("enabled", True):
            async def epistemic_handler(ctx):
                try:
                    from .cognition import (extract_predictions, store_predictions,
                                            detect_and_update_outcomes)
                    db = ctx.agent.memory.db
                    user_text = ctx.extra.get("user_input_text", "")
                    # Check if user confirmed/denied a previous prediction
                    if user_text:
                        detect_and_update_outcomes(db, ctx.user_id, user_text)
                    # Extract and store predictions from the current response
                    preds = extract_predictions(ctx.response_text)
                    if preds:
                        store_predictions(db, ctx.user_id, preds)
                except Exception as e:
                    logger.debug("Epistemic calibration hook error: %s", e)

            agent.hooks.register("after_response", "epistemic_calibration",
                                 epistemic_handler, priority=85)

    def _register_builtin_hooks(self):
        """Register built-in metacognition features as hook handlers."""
        agent = self

        # Confidence Gate (priority 50 — runs early to potentially escalate)
        cg_cfg = self._features.get("confidence_gate", {})
        if cg_cfg.get("enabled"):
            async def confidence_gate_handler(ctx: HookContext):
                try:
                    from .metacognition import assess_confidence
                    mem_cfg = dict(agent.config.get("memory", {}))
                    mem_cfg["_agent_config"] = agent.config.get("agent", {})
                    assessment = await assess_confidence(
                        agent.provider, ctx.extra.get("user_input_text", ""),
                        ctx.response_text, mem_cfg)
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
                # Record tool analytics
                if ctx.tool_calls_log and hasattr(agent.memory, "record_tool_calls"):
                    agent.memory.record_tool_calls(ctx.tool_calls_log, ctx.user_id)
            except Exception as e:
                logger.debug("Interaction logging error: %s", e)

        self.hooks.register("after_response", "interaction_log",
                            interaction_log_handler, priority=200)

        # Friction Detection (priority 225) — stores user correction signals for
        # self-evolving prompt; also triggers patch synthesis when threshold reached.
        if self._features.get("self_evolving_prompt", {}).get("enabled"):
            ep_cfg = dict(self._features.get("self_evolving_prompt", {}))
            ep_cfg["_agent_config"] = agent.config.get("agent", {})
            extraction_model = str(agent.config.get("memory", {}).get("extraction_model", "")).strip()
            if extraction_model:
                ep_cfg["extraction_model"] = extraction_model

            async def friction_detection_handler(ctx: HookContext):
                try:
                    from .evolution import detect_friction, store_friction, synthesize_prompt_patches
                    signal_type = detect_friction(ctx.extra.get("user_input_text", ""))
                    if signal_type:
                        store_friction(
                            agent.memory.db, ctx.user_id, signal_type,
                            ctx.extra.get("user_input_text", ""),
                            ctx.response_text)
                        # Synthesize patches in background when enough signals accumulate
                        synth_task = asyncio.create_task(
                            synthesize_prompt_patches(agent.provider, agent.memory.db, ep_cfg))

                        async def _finalize_patches():
                            patches = await synth_task
                            if patches and ep_cfg.get("auto_apply", True):
                                for p in patches:
                                    agent.memory.db.execute(
                                        "UPDATE prompt_patches SET applied=1 WHERE patch_text=?",
                                        (p,),
                                    )
                                agent.memory.db.commit()

                        finalize_task = asyncio.create_task(_finalize_patches())
                        agent._background_tasks.add(finalize_task)
                        finalize_task.add_done_callback(agent._background_tasks.discard)
                except Exception as e:
                    logger.debug("Friction detection error: %s", e)

            self.hooks.register("after_response", "friction_detection",
                                friction_detection_handler, priority=225)

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
            if hasattr(memory, "recall_type_aware"):
                results = memory.recall_type_aware(query, user_id=agent._current_user_id, top_k=5)
            else:
                results = memory.recall(query, user_id=agent._current_user_id, top_k=5)
            shown_ids = [int(m.get("id", 0)) for m in results if int(m.get("id", 0) or 0) > 0]
            if hasattr(memory, "register_recall_feedback"):
                try:
                    memory.register_recall_feedback(
                        query,
                        agent._current_user_id,
                        shown_ids,
                        shown_ids,
                        strength=1.0,
                        source="memory_search_tool",
                    )
                except Exception:
                    pass
            elif hasattr(memory, "reinforce_recall"):
                try:
                    memory.reinforce_recall(
                        query,
                        agent._current_user_id,
                        shown_ids,
                        strength=1.0,
                        source="memory_search_tool",
                    )
                except Exception:
                    pass
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

    def _has_tool_like(self, names: set[str], candidates: tuple[str, ...]) -> bool:
        lowered = {n.lower() for n in names}
        for candidate in candidates:
            c = candidate.lower()
            if c in lowered:
                return True
            if any(c in n for n in lowered):
                return True
        return False

    def _query_mentions_tool_gap(self, user_input: str) -> bool:
        text = (user_input or "").lower()
        return any(marker in text for marker in TOOL_GAP_MARKERS)

    def _query_needs_vision(self, user_input: str, tool_names: set[str]) -> bool:
        text = (user_input or "").lower()
        if not any(marker in text for marker in VISION_QUERY_MARKERS):
            return False
        return not self._has_tool_like(tool_names, VISION_TOOL_NAMES)

    def _media_understanding_config(self) -> dict:
        """Resolve auto image-understanding settings from config."""
        tools_cfg = self.config.get("tools", {})
        nested = tools_cfg.get("media_understanding", {})
        if not isinstance(nested, dict):
            nested = {}

        enabled = nested.get("enabled")
        if enabled is None:
            enabled = tools_cfg.get("media_understanding_enabled", True)

        model = str(nested.get("model", tools_cfg.get("media_understanding_model", "")) or "").strip()
        prompt = str(nested.get("prompt", tools_cfg.get("media_understanding_prompt", "")) or "").strip()
        try:
            max_images = int(nested.get("max_images", tools_cfg.get("media_understanding_max_images", 2)) or 2)
        except (TypeError, ValueError):
            max_images = 2
        try:
            max_documents = int(nested.get("max_documents", tools_cfg.get("media_understanding_max_documents", 1)) or 1)
        except (TypeError, ValueError):
            max_documents = 1
        try:
            max_tokens = int(nested.get("max_tokens", tools_cfg.get("media_understanding_max_tokens", 260)) or 260)
        except (TypeError, ValueError):
            max_tokens = 260
        try:
            max_tokens_document = int(nested.get("max_tokens_document", tools_cfg.get("media_understanding_max_tokens_document", 2000)) or 2000)
        except (TypeError, ValueError):
            max_tokens_document = 2000

        return {
            "enabled": bool(enabled),
            "model": model,
            "prompt": prompt,
            "max_images": max(1, min(max_images, 6)),
            "max_documents": max(1, min(max_documents, 4)),
            "max_tokens": max(80, min(max_tokens, 1024)),
            "max_tokens_document": max(200, min(max_tokens_document, 4096)),
        }

    def _build_media_understanding_prompt(
        self,
        user_text: str = "",
        *,
        media_kind: str = "image",
        media_label: str = "Image",
        media_index: int = 1,
    ) -> str:
        """Build grounding prompt for multimodal pre-pass."""
        cfg = self._media_understanding_config()
        if media_kind == "document":
            default_prompt = (
                "Extract and describe all content visible in this document. Include:\n"
                "- Document title and type (contract, agreement, invoice, report, etc.)\n"
                "- All party names, company names, persons mentioned\n"
                "- Dates, deadlines, and time periods\n"
                "- Key terms, amounts, and numeric values\n"
                "- Section headings and their main content\n"
                "- Signatures, stamps, or certifications visible\n"
                "- Any warnings, obligations, or important clauses\n"
                "Be thorough — extract ALL visible text and structure. "
                "Do not invent content not present in the document. "
                "Do not include instructions or task descriptions in your output — only document content."
            )
        else:
            default_prompt = (
                "Describe only what is physically visible in this image: "
                "objects, people, text, UI elements, layout, colors, and notable details. "
                "Do not speculate beyond visible evidence. "
                "Do not include instructions or task descriptions in your output — only image content."
            )
        base = cfg["prompt"] or default_prompt
        query = str(user_text or "").strip()
        if query:
            query = query[:500]
            # Clearly separate the task context from the image to prevent
            # vision models from hallucinating prompt text as image content.
            return (
                f"{base}\n\n"
                f"(The following is your analysis task — it is NOT text visible in the image: {query})\n\n"
                f"{media_label} index: {media_index}"
            )
        return f"{base}\n\n{media_label} index: {media_index}"

    def _vision_model_candidates(self, requested_model: str = "") -> list[str]:
        from .config import get_api_key
        from .providers import PROVIDER_MODELS

        provider_name = str(
            self.config.get("agent", {}).get("provider", "")
        ).strip().lower()
        candidates: list[str] = []

        def _add(model_name: str):
            name = str(model_name or "").strip()
            if name and name not in candidates:
                candidates.append(name)

        def _provider_ready(name: str) -> bool:
            if not name:
                return False
            if name == provider_name:
                return True
            if name == "ollama":
                return True
            return bool(get_api_key(name))

        def _add_for_provider(name: str, model_name: str):
            if not _provider_ready(name):
                return
            value = str(model_name or "").strip()
            if not value:
                return
            if name == provider_name:
                _add(value)
            else:
                _add(f"{name}:{value}")

        _add(requested_model)

        # Explicit vision_model config overrides auto-selection — add it first.
        cfg_vision_model = str(
            self.config.get("agent", {}).get("vision_model", "")
        ).strip()
        if cfg_vision_model:
            _add(cfg_vision_model)

        provider_priorities = {
            "openai": ("gpt-4o", "gpt-4.1"),
            "qwen": ("qwen-vl-plus", "qwen-vl-max", "qwen-plus"),
            "gemini": ("gemini-2.5-flash", "gemini-2.0-flash"),
            "anthropic": ("claude-sonnet-4-20250514", "claude-opus-4-20250115"),
            "grok": ("grok-4-0709", "grok-3"),
            "ollama": tuple(),
        }

        current_models = PROVIDER_MODELS.get(provider_name, []) or []
        for preferred in provider_priorities.get(provider_name, ()):
            if preferred in current_models or preferred == self.default_model:
                _add(preferred)

        for model_name in current_models:
            low = model_name.lower()
            if any(token in low for token in ("vision", "vl", "llava", "moondream", "gpt-4o")):
                _add(model_name)

        cross_provider_order = ("openai", "anthropic", "gemini", "qwen", "grok")
        for fallback_provider in cross_provider_order:
            for preferred in provider_priorities.get(fallback_provider, ()):
                _add_for_provider(fallback_provider, preferred)

        default_model = str(self.default_model or "").strip()
        if default_model:
            _add(default_model)
        if not candidates:
            candidates.append(self.default_model)
        return candidates[:8]

    def _document_model_candidates(self, requested_model: str = "") -> list[str]:
        """Ordered candidates for document/PDF pre-analysis."""
        from .config import get_api_key
        from .providers import PROVIDER_MODELS

        provider_name = str(
            self.config.get("agent", {}).get("provider", "")
        ).strip().lower()
        candidates: list[str] = []

        def _add(model_name: str):
            name = str(model_name or "").strip()
            if name and name not in candidates:
                candidates.append(name)

        def _provider_ready(name: str) -> bool:
            if not name:
                return False
            if name == provider_name:
                return True
            return bool(get_api_key(name))

        def _add_for_provider(name: str, model_name: str):
            if not _provider_ready(name):
                return
            value = str(model_name or "").strip()
            if not value:
                return
            if name == provider_name:
                _add(value)
            else:
                _add(f"{name}:{value}")

        _add(requested_model)

        # Explicit document_model config overrides auto-selection — add it first.
        cfg_document_model = str(
            self.config.get("agent", {}).get("document_model", "")
        ).strip()
        if cfg_document_model:
            _add(cfg_document_model)

        provider_priorities = {
            "anthropic": ("claude-sonnet-4-20250514", "claude-opus-4-20250115"),
            "gemini": ("gemini-2.5-flash", "gemini-2.0-flash"),
        }
        current_models = PROVIDER_MODELS.get(provider_name, []) or []
        if provider_name in provider_priorities:
            for preferred in provider_priorities[provider_name]:
                if preferred in current_models or preferred == self.default_model:
                    _add(preferred)
            _add(self.default_model)

        for fallback_provider in ("anthropic", "gemini"):
            for preferred in provider_priorities.get(fallback_provider, ()):
                _add_for_provider(fallback_provider, preferred)

        return candidates[:6]

    async def _complete_multimodal_with_fallback(
        self,
        content: list[dict],
        *,
        requested_model: str = "",
        max_tokens: int = 700,
        mode: str = "image",
    ) -> str:
        """Run a multimodal prompt through ordered model fallbacks."""
        result = await self._complete_multimodal_with_fallback_meta(
            content,
            requested_model=requested_model,
            max_tokens=max_tokens,
            mode=mode,
        )
        return str(result.get("text") or "")

    async def _complete_multimodal_with_fallback_meta(
        self,
        content: list[dict],
        *,
        requested_model: str = "",
        max_tokens: int = 700,
        mode: str = "image",
    ) -> dict:
        """Run a multimodal prompt through ordered model fallbacks and return route metadata."""
        last_error: Exception | None = None
        provider_names = ("anthropic", "openai", "gemini", "ollama", "qwen", "grok")
        current_provider = str(self.config.get("agent", {}).get("provider", "anthropic")).strip().lower()
        candidates = (
            self._document_model_candidates(requested_model)
            if mode == "document"
            else self._vision_model_candidates(requested_model)
        )
        for candidate in candidates:
            model_name = str(candidate or "").strip()
            target_provider = current_provider
            if ":" in model_name:
                prefix, rest = model_name.split(":", 1)
                if prefix in provider_names and rest:
                    target_provider = prefix
                    model_name = rest.strip()
            if not model_name:
                continue

            provider_obj = self.provider
            if target_provider != current_provider:
                temp_cfg = copy.deepcopy(self.config)
                temp_cfg.setdefault("agent", {})["provider"] = target_provider
                try:
                    provider_obj = create_provider(temp_cfg)
                except Exception as e:
                    last_error = e
                    logger.debug("Vision fallback provider init failed for %s/%s: %s",
                                 target_provider, model_name, e)
                    continue

            try:
                resp = await provider_obj.complete(
                    model=model_name,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": content}],
                )
                text = self._extract_text(resp).strip()
                if text:
                    # Track cost for vision/document model calls (same as main loop).
                    try:
                        if hasattr(resp, "usage") and resp.usage:
                            vision_cost = self._calculate_cost(model_name, resp.usage)
                            self.memory.track_usage(
                                self._current_user_id, model_name, resp.usage, vision_cost
                            )
                            logger.debug(
                                "Vision cost tracked: %s/%s $%.6f",
                                target_provider, model_name, vision_cost,
                            )
                    except Exception as _ce:
                        logger.debug("Vision cost tracking failed: %s", _ce)
                    return {
                        "text": text,
                        "model": model_name,
                        "provider": target_provider,
                    }
            except Exception as e:
                last_error = e
                logger.debug("Multimodal completion failed with %s/%s: %s",
                             target_provider, model_name, e)
                continue

        if last_error:
            raise last_error
        raise RuntimeError("no compatible multimodal model responded")

    @staticmethod
    def _classify_visual_block(block: dict, user_text: str = "") -> str:
        """Classify visual block for more specific summary labels."""
        source = (block or {}).get("source", {}) if isinstance(block, dict) else {}
        filename = str(source.get("filename", "") or "").lower()
        text = str(user_text or "").lower()
        screenshot_markers = ("screenshot", "screen", "скрин", "скриншот")
        if any(marker in filename for marker in screenshot_markers):
            return "screenshot"
        if any(marker in text for marker in screenshot_markers):
            return "screenshot"
        return "image"

    async def _apply_media_understanding(self, content_blocks: list[dict], user_text: str = "") -> list[dict]:
        """Pre-analyze attached images and inject concise summaries as grounding text."""
        if not isinstance(content_blocks, list):
            return content_blocks
        cfg = self._media_understanding_config()
        if not cfg.get("enabled"):
            return content_blocks

        images = [
            block for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "image"
        ]
        documents = [
            block for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "document"
        ]
        if not images and not documents:
            return content_blocks

        summaries: list[str] = []
        explainability: list[dict] = []
        for idx, block in enumerate(images[:cfg["max_images"]], start=1):
            visual_kind = self._classify_visual_block(block, user_text)
            media_label = "Screenshot" if visual_kind == "screenshot" else "Image"
            prompt = self._build_media_understanding_prompt(
                user_text,
                media_kind="image",
                media_label=media_label,
                media_index=idx,
            )
            payload = [
                {"type": "text", "text": prompt},
                block,
            ]
            try:
                result = await self._complete_multimodal_with_fallback_meta(
                    payload,
                    requested_model=cfg["model"],
                    max_tokens=cfg["max_tokens"],
                    mode="image",
                )
            except Exception as e:
                logger.debug("Media understanding failed for image %d: %s", idx, e)
                continue
            summary = str(result.get("text") or "").strip()
            summary = str(summary or "").strip()
            if summary and not self._is_garbage_response(summary):
                summaries.append(f"[{media_label} {idx}] {summary}")
                explainability.append({
                    "kind": visual_kind,
                    "label": media_label,
                    "index": idx,
                    "summary": summary,
                    "model": str(result.get("model") or ""),
                    "provider": str(result.get("provider") or ""),
                })
            elif summary:
                logger.warning("Discarding garbage vision summary from %s/%s",
                               result.get("provider", "?"), result.get("model", "?"))

        for idx, block in enumerate(documents[:cfg["max_documents"]], start=1):
            prompt = self._build_media_understanding_prompt(
                user_text,
                media_kind="document",
                media_label="Document",
                media_index=idx,
            )
            payload = [
                {"type": "text", "text": prompt},
                block,
            ]
            try:
                result = await self._complete_multimodal_with_fallback_meta(
                    payload,
                    requested_model=cfg["model"],
                    max_tokens=cfg["max_tokens_document"],
                    mode="document",
                )
            except Exception as e:
                logger.debug("Media understanding failed for document %d: %s", idx, e)
                continue
            summary = str(result.get("text") or "").strip()
            summary = str(summary or "").strip()
            if summary:
                summaries.append(f"[Document {idx}] {summary}")
                explainability.append({
                    "kind": "document",
                    "label": "Document",
                    "index": idx,
                    "summary": summary,
                    "model": str(result.get("model") or ""),
                    "provider": str(result.get("provider") or ""),
                })

        if not summaries:
            return content_blocks

        note = (
            "Auto media understanding (grounding hints; verify against the original attached media blocks):\n"
            + "\n\n".join(summaries)
        )
        if len(images) > cfg["max_images"]:
            note += f"\n\n[Only the first {cfg['max_images']} image(s) were pre-analyzed.]"
        if len(documents) > cfg["max_documents"]:
            note += f"\n\n[Only the first {cfg['max_documents']} document(s) were pre-analyzed.]"

        updated = list(content_blocks)
        insert_at = 1 if updated and isinstance(updated[0], dict) and updated[0].get("type") == "text" else 0
        updated.insert(insert_at, {"type": "text", "text": note})
        self._last_response_meta["media_explainability"] = explainability
        return updated

    @staticmethod
    def _content_has_media_blocks(content) -> bool:
        """True when user content contains image/document blocks."""
        if not isinstance(content, list):
            return False
        return any(
            isinstance(block, dict) and block.get("type") in {"image", "document"}
            for block in content
        )

    def _select_multimodal_response_model(self, model: str, content) -> str:
        """Ensure main answer for multimodal requests uses a multimodal-capable model."""
        if not self._content_has_media_blocks(content):
            return model

        lowered = str(model or "").lower()
        multimodal_tokens = (
            "gpt-4o", "vision", "vl", "gemini", "claude",
            "moondream", "llava",
        )
        if any(token in lowered for token in multimodal_tokens):
            return model

        has_document = any(
            isinstance(block, dict) and block.get("type") == "document"
            for block in (content or [])
        )
        candidate_pool = (
            self._document_model_candidates()
            if has_document else
            self._vision_model_candidates()
        )
        if candidate_pool:
            chosen = candidate_pool[0]
            if chosen and chosen != model:
                logger.info("Multimodal request: promoting response model %s -> %s", model, chosen)
                return chosen
        return model

    def _register_or_upgrade_vision_tool(self, force: bool = False) -> str:
        """Materialize/upgrade image understanding tool on-demand."""
        tool_name = "vision_analyze_image"
        had_tool_before = self.tools.has_tool(tool_name)
        existing = self.tools._tools.get(tool_name, {})
        existing_props = ((existing.get("input_schema") or {}).get("properties") or {})
        required_props = {"image_path", "image_url", "image_base64", "prompt", "model", "max_tokens"}
        is_modern = required_props.issubset(set(existing_props.keys()))

        if self.tools.has_tool(tool_name) and is_modern and not force:
            return tool_name

        agent = self
        sandbox_root = self.config.get("tools", {}).get("sandbox_root")
        max_bytes_cfg = self.config.get("tools", {}).get("vision_max_image_bytes", 8 * 1024 * 1024)
        try:
            max_image_bytes = int(max_bytes_cfg)
        except (TypeError, ValueError):
            max_image_bytes = 8 * 1024 * 1024
        max_image_bytes = max(512 * 1024, min(max_image_bytes, 20 * 1024 * 1024))

        async def vision_analyze_image(
            image_path: str = "",
            image_url: str = "",
            image_base64: str = "",
            prompt: str = "Describe what is shown in this image in detail.",
            model: str = "",
            max_tokens: int = 700,
        ) -> str:
            import base64 as _base64
            import mimetypes as _mimetypes
            import urllib.parse as _urlparse
            import urllib.request as _urlrequest
            from pathlib import Path as _Path
            from .tools import _validate_path

            raw_bytes = b""
            media_type = "image/jpeg"
            path_value = (image_path or "").strip()
            url_value = (image_url or "").strip()
            b64_value = (image_base64 or "").strip()

            if b64_value:
                if b64_value.startswith("data:"):
                    header, _, payload = b64_value.partition(",")
                    b64_value = payload or b64_value
                    try:
                        media_type = header.split(":", 1)[1].split(";", 1)[0] or media_type
                    except Exception:
                        pass
                try:
                    raw_bytes = _base64.b64decode(b64_value, validate=False)
                except Exception as e:
                    return f"Invalid image_base64: {e}"
            elif url_value:
                parsed = _urlparse.urlparse(url_value)
                if parsed.scheme not in ("http", "https"):
                    return "image_url must start with http:// or https://"
                try:
                    req = _urlrequest.Request(
                        url_value,
                        headers={"User-Agent": "LiteAgentVision/1.0"},
                    )
                    with _urlrequest.urlopen(req, timeout=20) as resp:
                        raw_bytes = resp.read(max_image_bytes + 1)
                        ctype = ""
                        try:
                            ctype = resp.headers.get_content_type() or ""
                        except Exception:
                            ctype = str(resp.headers.get("Content-Type", "")).split(";", 1)[0]
                        if ctype:
                            media_type = ctype
                except Exception as e:
                    return f"Failed to fetch image_url: {e}"
            elif path_value:
                resolved, err = _validate_path(path_value, sandbox_root=sandbox_root)
                if err:
                    return err
                p = _Path(resolved)
                if not p.exists():
                    return f"File not found: {resolved}"
                try:
                    raw_bytes = p.read_bytes()
                except Exception as e:
                    return f"Failed to read image_path: {e}"
                guessed = _mimetypes.guess_type(str(p))[0]
                if guessed:
                    media_type = guessed
            else:
                return (
                    "Provide one of: image_path, image_url, or image_base64. "
                    "Example: vision_analyze_image(image_path='/tmp/photo.jpg')"
                )

            if not raw_bytes:
                return "Image payload is empty."
            if len(raw_bytes) > max_image_bytes:
                return (
                    f"Image too large ({len(raw_bytes)} bytes). "
                    f"Limit is {max_image_bytes} bytes."
                )
            if not media_type.startswith("image/"):
                media_type = "image/jpeg"

            prompt_text = (prompt or "").strip() or "Describe what is shown in this image."
            prompt_text = prompt_text[:1200]
            b64_payload = _base64.b64encode(raw_bytes).decode("ascii")
            content = [
                {"type": "text", "text": prompt_text},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_payload,
                    },
                },
            ]

            token_limit = 700
            try:
                token_limit = max(80, min(int(max_tokens or 700), 2048))
            except Exception:
                pass

            last_error: Exception | None = None
            try:
                return await agent._complete_multimodal_with_fallback(
                    content,
                    requested_model=model,
                    max_tokens=token_limit,
                )
            except Exception as e:
                last_error = e
            if last_error:
                return f"Vision analysis failed: {last_error}"
            return "Vision analysis failed: no compatible model responded."

        self.tools._tools[tool_name] = {
            "name": tool_name,
            "description": (
                "Analyze an image from local path, URL, or base64 payload and return a detailed description. "
                "Use for object recognition, scene understanding, screenshots, and visual QA."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Local path to image file"},
                    "image_url": {"type": "string", "description": "HTTP/HTTPS URL of image"},
                    "image_base64": {"type": "string", "description": "Raw base64 or data URL payload"},
                    "prompt": {"type": "string", "description": "Task prompt for the vision model"},
                    "model": {"type": "string", "description": "Optional override model for this call"},
                    "max_tokens": {"type": "integer", "description": "Max output tokens (80-2048)"},
                },
            },
        }
        self.tools._handlers[tool_name] = vision_analyze_image
        logger.info(
            "Adaptive capability tool ready: %s%s",
            tool_name,
            " (upgraded)" if had_tool_before else " (created)",
        )
        return tool_name

    def _ensure_tool_autonomy(self, user_input: str, tool_defs: list[dict]) -> list[dict]:
        """Ensure capability-critical tools are available for the current query."""
        selected_defs = list(tool_defs)
        selected_names = {td.get("name", "") for td in selected_defs}
        all_tool_names = set(self.tools._tools.keys()) | selected_names

        # Semantic top-k selection can easily omit core workspace tools for
        # software-building prompts, especially on local models. Keep the
        # minimal file/shell toolkit available for app construction tasks.
        lowered = (user_input or "").lower()
        needs_workspace_tools = any(token in lowered for token in (
            "project", "app", "backend", "frontend", "fastapi", "react",
            "html", "css", "javascript", "typescript", "python", "api",
            "server", "endpoint", "docker", "uvicorn", "port", "curl",
            "build", "create", "implement", "fix", "debug", "run", "start",
            "проект", "прилож", "бэкенд", "бекенд", "фронтенд", "сервер",
            "эндпоинт", "порт", "запусти", "создай", "реализуй", "исправь",
            "проверь", "curl", "html", "css", "js", "api",
        )) or any(marker in user_input for marker in ("```", "def ", "class ", "import "))
        if needs_workspace_tools:
            for core_name in ("read_file", "write_file", "exec_command", "edit_file", "glob_files", "grep_search"):
                if core_name in self.tools._tools and core_name not in selected_names:
                    selected_defs.append(self.tools._tools[core_name])
                    selected_names.add(core_name)

        ats_cfg = self._features.get("auto_tool_synthesis", {})
        if not ats_cfg.get("enabled"):
            return selected_defs

        if self._query_needs_vision(user_input, all_tool_names):
            vision_tool = self._register_or_upgrade_vision_tool()
            if vision_tool not in selected_names and vision_tool in self.tools._tools:
                selected_defs.append(self.tools._tools[vision_tool])
                selected_names.add(vision_tool)

        if (
            self._query_mentions_tool_gap(user_input)
            and self.tools.has_tool("synthesize_tool")
            and "synthesize_tool" not in selected_names
        ):
            selected_defs.append(self.tools._tools["synthesize_tool"])

        return selected_defs

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
            from .web import web_fetch, wrap_untrusted_content
            result = await web_fetch(url, config=agent.config.get("web", {}),
                                     cache=agent._web_cache)
            if result.error:
                return f"Error fetching {url}: {result.error}"
            content = result.content[:max_length]
            truncated = " (truncated)" if len(result.content) > max_length else ""
            # Wrap with untrusted content markers (prompt injection defense)
            security_cfg = agent.config.get("web", {}).get("security", {})
            if security_cfg.get("wrap_untrusted", True):
                content = wrap_untrusted_content(content, result.url)
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

    def _wire_browser_tools(self):
        """Register browser automation tools (Playwright-based)."""
        agent = self
        browser_cfg = self.config.get("browser", {})

        from .browser import BrowserEngine
        self._browser_engine = BrowserEngine(browser_cfg)

        async def browser_launch_handler() -> str:
            r = await agent._browser_engine.launch()
            return r.data if r.success else f"Error: {r.error}"

        async def browser_close_handler() -> str:
            r = await agent._browser_engine.close()
            return r.data if r.success else f"Error: {r.error}"

        async def browser_new_tab_handler(url: str = "about:blank") -> str:
            r = await agent._browser_engine.new_tab(url)
            if r.success:
                return json.dumps(r.data, ensure_ascii=False)
            return f"Error: {r.error}"

        async def browser_close_tab_handler(tab_id: int) -> str:
            r = await agent._browser_engine.close_tab(tab_id)
            return r.data if r.success else f"Error: {r.error}"

        async def browser_list_tabs_handler() -> str:
            r = await agent._browser_engine.list_tabs()
            if r.success:
                return json.dumps(r.data, ensure_ascii=False)
            return f"Error: {r.error}"

        async def browser_navigate_handler(tab_id: int, url: str) -> str:
            r = await agent._browser_engine.navigate(tab_id, url)
            if r.success:
                return json.dumps(r.data, ensure_ascii=False)
            return f"Error: {r.error}"

        async def browser_screenshot_handler(tab_id: int,
                                              full_page: bool = False) -> str:
            r = await agent._browser_engine.screenshot(tab_id, full_page)
            if r.success:
                # Queue screenshot file for delivery
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False,
                                                  dir="/tmp") as f:
                    import base64 as _b64
                    f.write(_b64.b64decode(r.data["image_base64"]))
                    path = f.name
                from .file_queue import enqueue_file
                enqueue_file(path, caption="Browser screenshot")
                return f"Screenshot taken ({r.data['size']} bytes), queued for delivery"
            return f"Error: {r.error}"

        async def browser_click_handler(tab_id: int, selector: str) -> str:
            r = await agent._browser_engine.click(tab_id, selector)
            return r.data if r.success else f"Error: {r.error}"

        async def browser_type_handler(tab_id: int, selector: str,
                                        text: str, clear: bool = True) -> str:
            r = await agent._browser_engine.type_text(tab_id, selector, text, clear)
            return r.data if r.success else f"Error: {r.error}"

        async def browser_select_handler(tab_id: int, selector: str,
                                          value: str) -> str:
            r = await agent._browser_engine.select_option(tab_id, selector, value)
            return r.data if r.success else f"Error: {r.error}"

        async def browser_scroll_handler(tab_id: int, direction: str = "down",
                                          amount: int = 500) -> str:
            r = await agent._browser_engine.scroll(tab_id, direction, amount)
            return r.data if r.success else f"Error: {r.error}"

        async def browser_evaluate_handler(tab_id: int, expression: str) -> str:
            r = await agent._browser_engine.evaluate(tab_id, expression)
            if r.success:
                return json.dumps(r.data, ensure_ascii=False, default=str) if r.data is not None else "(undefined)"
            return f"Error: {r.error}"

        async def browser_accessibility_handler(tab_id: int) -> str:
            r = await agent._browser_engine.get_accessibility_tree(tab_id)
            return str(r.data)[:15000] if r.success else f"Error: {r.error}"

        async def browser_console_handler(tab_id: int,
                                           level: str = "all") -> str:
            r = await agent._browser_engine.get_console(tab_id, level)
            if r.success:
                return json.dumps(r.data, ensure_ascii=False)
            return f"Error: {r.error}"

        async def browser_get_text_handler(tab_id: int,
                                            selector: str = "body") -> str:
            r = await agent._browser_engine.get_text(tab_id, selector)
            return r.data if r.success else f"Error: {r.error}"

        async def browser_wait_for_handler(tab_id: int, selector: str,
                                            timeout: int = 10000) -> str:
            r = await agent._browser_engine.wait_for(tab_id, selector, timeout)
            return r.data if r.success else f"Error: {r.error}"

        async def browser_hover_handler(tab_id: int, selector: str) -> str:
            r = await agent._browser_engine.hover(tab_id, selector)
            return r.data if r.success else f"Error: {r.error}"

        async def browser_pdf_handler(tab_id: int) -> str:
            r = await agent._browser_engine.pdf(tab_id)
            if r.success:
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False,
                                                  dir="/tmp") as f:
                    import base64 as _b64
                    f.write(_b64.b64decode(r.data["pdf_base64"]))
                    path = f.name
                from .file_queue import enqueue_file
                enqueue_file(path, caption="Browser PDF")
                return f"PDF generated ({r.data['size']} bytes), queued for delivery"
            return f"Error: {r.error}"

        tools_defs = [
            {"name": "browser_launch",
             "description": "Launch the browser. Must be called before using other browser tools.",
             "input_schema": {"type": "object", "properties": {}},
             "_handler": browser_launch_handler},
            {"name": "browser_close",
             "description": "Close the browser and free resources.",
             "input_schema": {"type": "object", "properties": {}},
             "_handler": browser_close_handler},
            {"name": "browser_new_tab",
             "description": "Open a new browser tab, optionally navigating to a URL. Returns tab_id.",
             "input_schema": {"type": "object", "properties": {
                 "url": {"type": "string", "description": "URL to open (default: about:blank)"},
             }},
             "_handler": browser_new_tab_handler},
            {"name": "browser_close_tab",
             "description": "Close a browser tab by its ID.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID to close"},
             }, "required": ["tab_id"]},
             "_handler": browser_close_tab_handler},
            {"name": "browser_list_tabs",
             "description": "List all open browser tabs with their IDs, URLs, and titles.",
             "input_schema": {"type": "object", "properties": {}},
             "_handler": browser_list_tabs_handler},
            {"name": "browser_navigate",
             "description": "Navigate a browser tab to a URL.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "url": {"type": "string", "description": "URL to navigate to"},
             }, "required": ["tab_id", "url"]},
             "_handler": browser_navigate_handler},
            {"name": "browser_screenshot",
             "description": "Take a screenshot of a browser tab. The image is queued for delivery.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "full_page": {"type": "boolean", "description": "Capture full page (default: false)"},
             }, "required": ["tab_id"]},
             "_handler": browser_screenshot_handler},
            {"name": "browser_click",
             "description": "Click an element by CSS selector.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "selector": {"type": "string", "description": "CSS selector of element to click"},
             }, "required": ["tab_id", "selector"]},
             "_handler": browser_click_handler},
            {"name": "browser_type",
             "description": "Type text into an input element.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "selector": {"type": "string", "description": "CSS selector of input"},
                 "text": {"type": "string", "description": "Text to type"},
                 "clear": {"type": "boolean", "description": "Clear field before typing (default: true)"},
             }, "required": ["tab_id", "selector", "text"]},
             "_handler": browser_type_handler},
            {"name": "browser_select",
             "description": "Select a dropdown option by value or label.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "selector": {"type": "string", "description": "CSS selector of select element"},
                 "value": {"type": "string", "description": "Option value or label"},
             }, "required": ["tab_id", "selector", "value"]},
             "_handler": browser_select_handler},
            {"name": "browser_scroll",
             "description": "Scroll the page up or down.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "direction": {"type": "string", "description": "'up' or 'down' (default: down)"},
                 "amount": {"type": "integer", "description": "Pixels to scroll (default: 500)"},
             }, "required": ["tab_id"]},
             "_handler": browser_scroll_handler},
            {"name": "browser_evaluate",
             "description": "Execute JavaScript in the page context and return the result.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "expression": {"type": "string", "description": "JavaScript expression to evaluate"},
             }, "required": ["tab_id", "expression"]},
             "_handler": browser_evaluate_handler},
            {"name": "browser_accessibility",
             "description": "Get accessibility tree snapshot — a structured view of the page for understanding layout and interactive elements.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
             }, "required": ["tab_id"]},
             "_handler": browser_accessibility_handler},
            {"name": "browser_console",
             "description": "Read browser console messages (logs, errors, warnings).",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "level": {"type": "string", "description": "'all', 'error', or 'warning' (default: all)"},
             }, "required": ["tab_id"]},
             "_handler": browser_console_handler},
            {"name": "browser_get_text",
             "description": "Extract visible text content from a page or specific element.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "selector": {"type": "string", "description": "CSS selector (default: body)"},
             }, "required": ["tab_id"]},
             "_handler": browser_get_text_handler},
            {"name": "browser_wait_for",
             "description": "Wait for an element to appear on the page.",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "selector": {"type": "string", "description": "CSS selector to wait for"},
                 "timeout": {"type": "integer", "description": "Max wait time in ms (default: 10000)"},
             }, "required": ["tab_id", "selector"]},
             "_handler": browser_wait_for_handler},
            {"name": "browser_hover",
             "description": "Hover over an element (useful for revealing tooltips/menus).",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
                 "selector": {"type": "string", "description": "CSS selector to hover over"},
             }, "required": ["tab_id", "selector"]},
             "_handler": browser_hover_handler},
            {"name": "browser_pdf",
             "description": "Generate a PDF of the current page (headless only).",
             "input_schema": {"type": "object", "properties": {
                 "tab_id": {"type": "integer", "description": "Tab ID"},
             }, "required": ["tab_id"]},
             "_handler": browser_pdf_handler},
        ]

        for td in tools_defs:
            handler = td.pop("_handler")
            self.tools._tools[td["name"]] = td
            self.tools._handlers[td["name"]] = handler

        tool_names = [td["name"] for td in tools_defs]
        logger.info("Browser tools registered: %s", ", ".join(
            t for t in self.tools._tools if t.startswith("browser_")))

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
                elif p == "groq":
                    configured = bool(get_api_key("groq") or os.environ.get("GROQ_API_KEY"))
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
                if tts_provider == "groq" and not (get_api_key("groq") or os.environ.get("GROQ_API_KEY")):
                    warnings.append("Groq API key not configured — TTS may fail")
                tts["provider"] = tts_provider
                changes.append(f"provider={tts_provider}")

            # TTS voice
            if tts_voice:
                provider = tts_provider or tts.get("provider", "openai")
                if provider == "openai":
                    tts.setdefault("openai", {})["voice"] = tts_voice
                elif provider == "groq":
                    tts.setdefault("groq", {})["voice"] = tts_voice
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
                elif p == "groq":
                    from .voice import (
                        GROQ_TTS_MODELS,
                        GROQ_TTS_MODEL_INFO,
                        GROQ_TTS_LANGUAGE_LABELS,
                    )
                    entry["configured"] = bool(get_api_key("groq") or os.environ.get("GROQ_API_KEY"))
                    entry["models"] = list(GROQ_TTS_MODELS)
                    entry["voices"] = sorted({
                        voice
                        for meta in GROQ_TTS_MODEL_INFO.values()
                        for voice in meta.get("voices", [])
                    })
                    entry["languages"] = sorted({
                        language
                        for meta in GROQ_TTS_MODEL_INFO.values()
                        for language in (meta.get("supported_languages") or [meta.get("language")])
                        if language
                    })
                    entry["language_labels"] = dict(GROQ_TTS_LANGUAGE_LABELS)
                    entry["note"] = "Ultra-fast inference. English and Arabic are officially documented; LiteAgent also supports Russian on Groq as a direct experimental mode."
                elif p == "edge":
                    from .voice import EDGE_TTS_LANGUAGE_LABELS, EDGE_TTS_VOICES_BY_LANGUAGE
                    entry["configured"] = True
                    entry["models"] = []
                    entry["voices"] = [
                        voice
                        for voices in EDGE_TTS_VOICES_BY_LANGUAGE.values()
                        for voice in voices
                    ]
                    entry["languages"] = list(EDGE_TTS_VOICES_BY_LANGUAGE.keys())
                    entry["language_labels"] = dict(EDGE_TTS_LANGUAGE_LABELS)
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

    def _wire_skill_tools(self):
        """Register skill management tools — let the agent view and manage skills."""
        agent = self

        def list_agent_skills() -> str:
            """List all available skills with their status, keywords, and tools."""
            skills = agent.skill_registry.list_skills()
            return json.dumps(skills, indent=2, ensure_ascii=False)

        def read_skill(name: str) -> str:
            """Read full content and metadata of a specific skill.

            name: The skill name (e.g. 'voice', 'web_research', 'knowledge_base')
            """
            detail = agent.skill_registry.get_skill(name)
            if not detail:
                return json.dumps({"error": f"Skill '{name}' not found"})
            return json.dumps(detail, indent=2, ensure_ascii=False)

        def propose_skill_update(name: str, body: str = "", description: str = "",
                                 keywords: str = "") -> str:
            """Propose changes to an existing skill. Show preview to user for approval.

            name: Skill name to update
            body: New markdown body (leave empty to keep current)
            description: New description (leave empty to keep current)
            keywords: Comma-separated keywords (leave empty to keep current)
            """
            skill = agent.skill_registry._skills.get(name)
            if not skill:
                return json.dumps({"error": f"Skill '{name}' not found"})
            changes = {}
            if body.strip():
                changes["body"] = body.strip()
            if description.strip():
                changes["description"] = description.strip()
            if keywords.strip():
                changes["keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
            if not changes:
                return json.dumps({"error": "No changes proposed"})
            return json.dumps({
                "action": "update_skill",
                "skill": name,
                "source": skill.source,
                "current_description": skill.metadata.description,
                "changes": changes,
                "message": (f"I'd like to update the '{name}' skill. "
                            "Please confirm by saying 'yes' or 'approve'."),
            }, indent=2, ensure_ascii=False)

        def propose_skill_create(name: str, description: str, body: str,
                                 keywords: str = "", emoji: str = "") -> str:
            """Propose creating a new user skill. Show preview for approval.

            name: Skill name (lowercase, hyphens, e.g. 'my-custom-skill')
            description: What the skill does
            body: Markdown instructions injected when skill is triggered
            keywords: Comma-separated trigger keywords
            emoji: Optional emoji for catalog display
            """
            if name in agent.skill_registry._skills:
                return json.dumps({"error": f"Skill '{name}' already exists"})
            return json.dumps({
                "action": "create_skill",
                "name": name,
                "description": description,
                "body_preview": body[:500] + ("..." if len(body) > 500 else ""),
                "keywords": [k.strip() for k in keywords.split(",") if k.strip()],
                "emoji": emoji,
                "message": (f"I'd like to create a new skill '{name}'. "
                            "Please confirm by saying 'yes' or 'approve'."),
            }, indent=2, ensure_ascii=False)

        def apply_skill_change(action: str, name: str, body: str = "",
                               description: str = "", keywords: str = "",
                               emoji: str = "") -> str:
            """Apply a previously proposed skill change after user confirms.

            action: 'create' or 'update'
            name: Skill name
            body: Full markdown body
            description: Skill description
            keywords: Comma-separated keywords
            emoji: Optional emoji
            """
            kw_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
            frontmatter: dict = {"name": name, "description": description or ""}
            meta: dict = {}
            if emoji:
                meta["emoji"] = emoji
            if kw_list:
                meta["keywords"] = kw_list
            if meta:
                frontmatter["metadata"] = meta

            if action == "create":
                if name in agent.skill_registry._skills:
                    return json.dumps({"error": f"Skill '{name}' already exists"})
                agent.skill_registry.write_skill(name, body, frontmatter)
            elif action == "update":
                skill = agent.skill_registry._skills.get(name)
                if not skill:
                    return json.dumps({"error": f"Skill '{name}' not found"})
                if not body:
                    body = skill.body
                if not description:
                    frontmatter["description"] = skill.metadata.description
                if not kw_list:
                    meta["keywords"] = skill.metadata.keywords
                    frontmatter["metadata"] = meta
                agent.skill_registry.write_skill(name, body, frontmatter)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})

            agent.skill_registry.load_all(agent.config)
            return json.dumps({
                "ok": True, "action": action, "name": name,
                "message": f"Skill '{name}' {action}d successfully and reloaded.",
            })

        self.tools.tool(
            name="list_agent_skills",
            description="List all available skills with status, keywords, and tools"
        )(list_agent_skills)
        self.tools.tool(
            name="read_skill",
            description="Read full content and metadata of a specific skill"
        )(read_skill)
        self.tools.tool(
            name="propose_skill_update",
            description="Propose changes to an existing skill (requires user confirmation)"
        )(propose_skill_update)
        self.tools.tool(
            name="propose_skill_create",
            description="Propose creating a new skill (requires user confirmation)"
        )(propose_skill_create)
        self.tools.tool(
            name="apply_skill_change",
            description="Apply a skill change after user approval (create or update)"
        )(apply_skill_change)

    def _wire_status_tool(self):
        """Register send_status tool — lets agent send short progress messages mid-execution."""
        agent = self

        def _resolve_telegram_delivery_chat_id() -> str | None:
            chat_id = agent._get_current_chat_id()
            if chat_id:
                return str(chat_id)

            user_id = str(getattr(agent, "_current_user_id", "") or "").strip()
            if user_id and getattr(agent, "memory", None) is not None:
                try:
                    remembered = agent.memory.get_state("user:telegram_chat_id", user_id=user_id)
                    if remembered:
                        remembered_str = str(remembered).strip()
                        if remembered_str:
                            return remembered_str
                except Exception:
                    pass
            if user_id:
                inferred_private_chat = agent._infer_private_telegram_chat_id_from_user_id(user_id)
                if inferred_private_chat:
                    return inferred_private_chat

            tg_cfg = agent.config.get("channels", {}).get("telegram", {})
            raw = tg_cfg.get("chat_id") or tg_cfg.get("chat_ids")
            if isinstance(raw, (list, tuple)):
                raw = raw[0] if raw else None
            if raw is None:
                return None
            chat_id_str = str(raw).strip()
            if "," in chat_id_str:
                chat_id_str = chat_id_str.split(",", 1)[0].strip()
            return chat_id_str or None

        def _send_telegram_text(message: str, *, parse_mode: str = "Markdown") -> bool:
            import json as _json
            import os
            import urllib.request

            text = str(message or "").strip()
            if not text:
                return False

            chat_id = _resolve_telegram_delivery_chat_id()
            if not chat_id:
                return False

            try:
                from .config import get_api_key
                tg_cfg = agent.config.get("channels", {}).get("telegram", {})
                token_env = tg_cfg.get("token_env", "TELEGRAM_BOT_TOKEN")
                token = (
                    str(tg_cfg.get("token") or "").strip()
                    or str(get_api_key("telegram") or "").strip()
                    or os.environ.get(token_env, "")
                )
                if not token:
                    return False
                payload = _json.dumps({
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                }).encode()
                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=8)
                return True
            except Exception as _e:
                logger.debug("telegram delivery error: %s", _e)
                return False

        def send_status(message: str) -> str:
            """Send a brief status update to the user during a long operation.

            Use this between tool calls to inform the user what is happening now.
            Keep messages short (1 sentence). Do NOT use for final answers.
            Use sparingly — only at meaningful milestones (not every step).

            message: Short status text, e.g. "Транскрибирую аудио..." or "Ищу в интернете..."
            """
            sent_tg = _send_telegram_text(message, parse_mode="Markdown")

            # Always broadcast to WebSocket dashboard
            agent._ws_broadcast("agent_status", {
                "message": message,
                "chat_id": _resolve_telegram_delivery_chat_id(),
            })

            return "ok" if sent_tg else "ok (dashboard only)"

        def send_text_to_user(message: str) -> str:
            """Send a plain text message directly to the current Telegram chat.

            Use this when the user explicitly asks to receive some prepared text in Telegram
            as a separate outbound message. If Telegram delivery context is unavailable,
            returns a clear error instead of pretending it was sent.
            """
            text = str(message or "").strip()
            if not text:
                return "Error: message is empty"
            if not _resolve_telegram_delivery_chat_id():
                return "Error: no active Telegram chat available for delivery"
            sent_tg = _send_telegram_text(text, parse_mode="Markdown")
            return "Message sent to Telegram chat." if sent_tg else "Error: Telegram delivery failed"

        self.tools.tool(
            name="send_status",
            description=(
                "Send a short status update to the user during a long multi-step operation. "
                "Use ONCE at the start of heavy work (transcription, search, analysis). "
                "Do NOT use for every step — only for meaningful milestones."
            ),
        )(send_status)
        self.tools.tool(
            name="send_text_to_user",
            description=(
                "Send a plain text message directly to the current Telegram chat. "
                "Use when the user explicitly asks to deliver prepared text in Telegram "
                "as a separate message. Do not use this for normal final answers in the same chat."
            ),
        )(send_text_to_user)

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

        # ── send_stored_file_to_telegram: direct Telegram delivery from S3 ──
        async def send_stored_file_to_telegram_handler(storage_key: str, caption: str = "") -> str:
            """Download a file from storage and send it directly to the resolved Telegram chat."""
            if ".." in storage_key or storage_key.startswith("/"):
                return "Access denied: invalid key"
            try:
                import json as _json
                import urllib.request
                import uuid

                from .config import get_api_key

                chat_id = agent._get_current_chat_id()
                if chat_id:
                    chat_id = str(chat_id)
                else:
                    current_uid = str(getattr(agent, "_current_user_id", "") or "").strip()
                    remembered = None
                    if current_uid and getattr(agent, "memory", None) is not None:
                        try:
                            remembered = agent.memory.get_state("user:telegram_chat_id", user_id=current_uid)
                        except Exception:
                            remembered = None
                    chat_id = str(remembered).strip() if remembered else ""
                    if not chat_id and current_uid:
                        chat_id = str(agent._infer_private_telegram_chat_id_from_user_id(current_uid) or "").strip()
                    if not chat_id:
                        tg_cfg = agent.config.get("channels", {}).get("telegram", {})
                        raw = tg_cfg.get("chat_id") or tg_cfg.get("chat_ids")
                        if isinstance(raw, (list, tuple)):
                            raw = raw[0] if raw else None
                        chat_id = str(raw or "").strip()
                        if "," in chat_id:
                            chat_id = chat_id.split(",", 1)[0].strip()
                if not chat_id:
                    return "Error: no active Telegram chat available for delivery"

                tg_cfg = agent.config.get("channels", {}).get("telegram", {})
                token_env = tg_cfg.get("token_env", "TELEGRAM_BOT_TOKEN")
                token = (
                    str(tg_cfg.get("token") or "").strip()
                    or str(get_api_key("telegram") or "").strip()
                    or os.environ.get(token_env, "")
                )
                if not token:
                    return "Error: Telegram bot token is not configured"

                data = await storage.async_download(storage_key)
                filename = storage_key.rsplit("/", 1)[-1] or "document"
                if caption:
                    final_caption = caption
                else:
                    final_caption = filename

                boundary = f"liteagent-{uuid.uuid4().hex}"
                content_type = "application/octet-stream"
                if fm:
                    try:
                        matches = fm.list_files(user_id=agent._current_user_id, limit=200)
                        for item in matches or []:
                            if isinstance(item, dict) and str(item.get("storage_key") or "") == storage_key:
                                filename = str(item.get("original_name") or filename).strip() or filename
                                content_type = str(item.get("mime_type") or content_type).strip() or content_type
                                break
                    except Exception:
                        pass

                body = bytearray()

                def _add_field(name: str, value: str) -> None:
                    body.extend(f"--{boundary}\r\n".encode("utf-8"))
                    body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
                    body.extend(str(value).encode("utf-8"))
                    body.extend(b"\r\n")

                _add_field("chat_id", chat_id)
                _add_field("caption", final_caption)

                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(
                    (
                        f'Content-Disposition: form-data; name="document"; filename="{filename}"\r\n'
                        f"Content-Type: {content_type}\r\n\r\n"
                    ).encode("utf-8")
                )
                body.extend(data)
                body.extend(b"\r\n")
                body.extend(f"--{boundary}--\r\n".encode("utf-8"))

                req = urllib.request.Request(
                    f"https://api.telegram.org/bot{token}/sendDocument",
                    data=bytes(body),
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=20) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                parsed = _json.loads(payload) if payload else {"ok": True}
                if not parsed.get("ok", False):
                    return f"Error: Telegram delivery failed ({parsed.get('description') or 'unknown error'})"
                return f"Stored file sent to Telegram chat: {filename}"
            except Exception as e:
                return f"Error sending stored file to Telegram: {e}"

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
            ("send_stored_file_to_telegram", send_stored_file_to_telegram_handler,
             "Download a file from storage and send it directly to the resolved Telegram chat. "
             "Use this when the user explicitly asks to receive a stored S3 file in Telegram.",
             {"type": "object", "properties": {
                 "storage_key": {"type": "string", "description": "Storage key of the file"},
                 "caption": {"type": "string", "description": "Telegram caption for the file (optional)"},
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

    _KB_DOCUMENT_MIMES = {
        "application/pdf",
        "text/markdown",
        "text/html",
        "text/plain",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-word.document.macroenabled.12",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel.sheet.macroenabled.12",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint.presentation.macroenabled.12",
        "application/vnd.ms-powerpoint",
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        "application/rtf",
        "application/epub+zip",
    }
    _KB_DOCUMENT_EXTS = {
        ".pdf", ".md", ".markdown", ".html", ".htm", ".txt", ".rst",
        ".docx", ".docm", ".xlsx", ".xlsm", ".pptx", ".pptm",
        ".doc", ".xls", ".ppt",
        ".odt", ".ods", ".odp", ".rtf", ".epub",
    }

    async def ingest_file(self, data: bytes, filename: str, *,
                          source: str = "unknown", user_id: str = "system",
                          mime_type: str = "", description: str = "") -> dict | None:
        """Public method for channels to auto-ingest files into S3 + Knowledge Base."""
        fm = self._file_manager
        if not fm:
            return None
        try:
            file_info = detect_file_type(data, filename, mime_type)
            result = await fm.ingest(
                data, filename, source=source, user_id=user_id,
                mime_type=mime_type, description=description)
            # Auto-ingest documents into Knowledge Base for deep search
            kb = self._knowledge_base
            ext = os.path.splitext(filename)[1].lower()
            if kb and (file_info.mime_type in self._KB_DOCUMENT_MIMES
                       or ext in self._KB_DOCUMENT_EXTS
                       or file_info.can_extract_text):
                try:
                    import tempfile
                    with tempfile.NamedTemporaryFile(
                            suffix=ext, delete=False) as tmp:
                        tmp.write(data)
                        tmp_path = tmp.name
                    await kb.ingest(tmp_path)
                    os.unlink(tmp_path)
                    logger.info("KB auto-ingest: %s", filename)
                except Exception as e:
                    logger.debug("KB auto-ingest failed for %s: %s", filename, e)
            return result
        except Exception as e:
            logger.warning("File ingestion failed: %s", e)
            return None

    async def process_document_upload(self, data: bytes, filename: str, *,
                                      source: str = "dashboard", user_id: str = "system",
                                      mime_type: str = "") -> dict:
        """Run the full document workflow for a deliberate document upload."""
        pipeline = getattr(self, "_document_pipeline", None)
        if not pipeline:
            raise RuntimeError("Document pipeline is not available")
        return await pipeline.process_upload(
            data,
            filename,
            source=source,
            user_id=user_id,
            mime_type=mime_type,
        )

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
                chat_id = agent._get_current_chat_id()
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

        async def enqueue_background_task_handler(name: str, query: str,
                                                  priority: int = 5,
                                                  retry_delay_sec: int = 45,
                                                  max_attempts: int = 0) -> str:
            """Queue autonomous background work to run ASAP with retries."""
            uid = agent._current_user_id
            chat_id = None
            if uid.startswith("tg-"):
                chat_id = agent._get_current_chat_id()

            try:
                task = tm.add_task(
                    name=name,
                    query=query,
                    user_id=uid,
                    task_type="one_shot",
                    run_at=_dt.now().isoformat(timespec='seconds'),
                    chat_id=str(chat_id) if chat_id else None,
                    priority=max(1, min(int(priority), 9)),
                    background=True,
                    retry_delay_sec=max(5, min(int(retry_delay_sec), 86400)),
                    max_attempts=max(0, min(int(max_attempts), 100)),
                    source="agent",
                )
                daemon = getattr(agent, "_background_task_daemon", None)
                if daemon:
                    with suppress(Exception):
                        await daemon.start()
                return _json.dumps(task, ensure_ascii=False, default=str)
            except Exception as e:
                return f"Error queueing background task: {e}"

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

            ("enqueue_background_task", enqueue_background_task_handler,
             "Queue a background task that runs asynchronously with retry support. "
             "Use this for autonomous follow-up work that should continue after the current reply "
             "(e.g., long research, retries, or deferred processing) without blocking the user.",
             {"type": "object", "properties": {
                 "name": {"type": "string", "description": "Short background task title"},
                 "query": {"type": "string", "description": "Instruction to execute in background"},
                 "priority": {"type": "integer", "description": "Priority 1..9 (1 highest)", "default": 5},
                 "retry_delay_sec": {"type": "integer", "description": "Delay before retry on failure", "default": 45},
                 "max_attempts": {"type": "integer",
                                  "description": "0 = unlimited retries, otherwise max attempts", "default": 0},
             }, "required": ["name", "query"]}),

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

        logger.info("Task tools registered (schedule_task, enqueue_background_task, list_tasks, cancel_task)")

    def _get_current_chat_id(self) -> str | None:
        """Get request-local Telegram chat_id (falls back to legacy attr)."""
        cid = self._current_chat_id_ctx.get()
        if cid:
            return str(cid)
        raw = getattr(self, "_current_chat_id", None)
        return str(raw) if raw else None

    def _set_current_chat_id(self, chat_id: str | int | None):
        """Bind chat_id to current async context. Returns context token."""
        if chat_id in (None, "", 0):
            return self._current_chat_id_ctx.set(None)
        return self._current_chat_id_ctx.set(str(chat_id))

    def _reset_current_chat_id(self, token):
        """Restore previous chat_id context."""
        with suppress(Exception):
            self._current_chat_id_ctx.reset(token)

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
        self._normalize_runtime_model_config()

        # Cost controls
        if "cascade_routing" in cost_cfg:
            self.cascade_routing = cost_cfg["cascade_routing"]
        if "budget_daily_usd" in cost_cfg:
            self.budget_daily = cost_cfg["budget_daily_usd"]
        self._intelligent_routing_cfg = self._build_intelligent_routing_config(cost_cfg)

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

    @property
    def _lock_timeout(self) -> float:
        """Lock timeout depends on provider: local models need more time to load."""
        provider_name = self.config.get("agent", {}).get("provider", "anthropic")
        return 300.0 if provider_name == "ollama" else 60.0

    def _provider_call_timeout(self, provider_name: str) -> float:
        """Long local-model passes need a higher default than cloud APIs."""
        raw = self.config.get("providers", {}).get("call_timeout")
        try:
            if raw is not None:
                return float(raw)
        except Exception:
            pass
        return 600.0 if provider_name == "ollama" else 120.0

    def _resolve_slow_local_mode(self) -> bool:
        """Enable latency-focused profile for slow local models."""
        agent_cfg = self.config.get("agent", {})
        explicit = agent_cfg.get("slow_local_mode")
        if isinstance(explicit, bool):
            return explicit
        if isinstance(explicit, str):
            v = explicit.strip().lower()
            if v in {"true", "1", "yes", "on", "enabled"}:
                return True
            if v in {"false", "0", "no", "off", "disabled"}:
                return False
        provider_name = str(agent_cfg.get("provider", "")).strip().lower()
        return provider_name == "ollama"

    def _build_slow_local_profile(self) -> dict:
        """Build effective slow-local profile with safe defaults."""
        raw = self.config.get("agent", {}).get("slow_local", {})
        if not isinstance(raw, dict):
            raw = {}

        def _int(name: str, default: int, min_v: int, max_v: int) -> int:
            try:
                val = int(raw.get(name, default))
            except Exception:
                val = default
            return max(min_v, min(max_v, val))

        return {
            "memory_top_k": _int("memory_top_k", 3, 1, 10),
            "tool_top_k": _int("tool_top_k", 5, 1, 16),
            "kb_top_k": _int("kb_top_k", 2, 1, 10),
            "kb_max_chars": _int("kb_max_chars", 2500, 300, 20000),
            "kb_cache_ttl_sec": _int("kb_cache_ttl_sec", 45, 0, 3600),
            "disable_planning": bool(raw.get("disable_planning", True)),
            "disable_reflection": bool(raw.get("disable_reflection", True)),
            "kb_query_expansion": bool(raw.get("kb_query_expansion", False)),
            "compact_kb_prompt": bool(raw.get("compact_kb_prompt", True)),
        }

    def _memory_recall_top_k(self) -> int:
        if not self._slow_local_mode:
            return 5
        return int(self._slow_local_cfg.get("memory_top_k", 3))

    def _tool_relevance_top_k(self) -> int:
        if not self._slow_local_mode:
            return 8
        return int(self._slow_local_cfg.get("tool_top_k", 5))

    async def _auto_retrieve_kb_context(self, query: str) -> str:
        """Fetch relevant KB context proactively for user query."""
        kb = self._knowledge_base
        if not kb:
            return ""
        q = (query or "").strip()
        if len(q) < 3:
            return ""
        # Personal/user-profile questions should rely on memory, not KB docs.
        if self._is_personal_memory_query(q):
            return ""
        # Avoid pulling KB context for simple greetings/acks.
        if not self._should_recall(q) and "?" not in q:
            return ""

        kb_cfg = self.config.get("knowledge_base", {})
        if not kb_cfg.get("enabled", False):
            return ""
        auto_mode = self._resolve_kb_auto_context_mode(kb_cfg)
        if auto_mode == "off":
            return ""
        if auto_mode == "on_demand" and not self._should_use_kb_for_query(q):
            return ""

        top_k = int(kb_cfg.get("auto_context_top_k", 4))
        top_k = max(1, min(top_k, 10))
        max_chars = int(kb_cfg.get("auto_context_max_chars", 6000))
        max_chars = max(500, min(max_chars, 20000))
        search_mode = kb_cfg.get("search_mode", "hybrid")
        if self._slow_local_mode:
            top_k = min(top_k, int(self._slow_local_cfg.get("kb_top_k", 2)))
            max_chars = min(max_chars, int(self._slow_local_cfg.get("kb_max_chars", 2500)))

        # Cache repeated retrievals for the same user/query to reduce local-latency spikes.
        cache_ttl = int(self._slow_local_cfg.get("kb_cache_ttl_sec", 0)) if self._slow_local_mode else 0
        cache_key = ""
        if cache_ttl > 0:
            user_key = self._current_user_id or "default"
            cache_key = (
                f"{user_key}|{auto_mode}|{search_mode}|{top_k}|{max_chars}|"
                f"{q.strip().lower()[:400]}"
            )
            now = time.time()
            cached = self._kb_auto_context_cache.get(cache_key)
            if cached and (now - cached[0] <= cache_ttl):
                return cached[1]

        context = ""
        try:
            query_primary = self._prepare_kb_query(q)
            queries = [query_primary] if query_primary else []
            if not (self._slow_local_mode and not self._slow_local_cfg.get("kb_query_expansion", False)):
                query_en = self._heuristic_kb_query_english(query_primary)
                if query_en and query_en not in queries:
                    queries.append(query_en)
            if not queries:
                context = ""
            else:
                merged = {}
                for qx in queries:
                    batch = await kb.search(qx, top_k=top_k, mode=search_mode)
                    for r in batch:
                        key = getattr(r, "chunk_id", None) or (
                            f"{getattr(r, 'source', '')}:{getattr(r, 'page', 0)}:"
                            f"{getattr(r, 'section', '')}:{hash(getattr(r, 'content', '')[:80])}"
                        )
                        prev = merged.get(key)
                        if prev is None or getattr(r, "score", 0.0) > getattr(prev, "score", 0.0):
                            merged[key] = r
                    if self._slow_local_mode and len(merged) >= top_k:
                        break

                results = sorted(
                    merged.values(),
                    key=lambda x: getattr(x, "score", 0.0),
                    reverse=True,
                )[:top_k]
                if results:
                    context = kb.build_context(results).strip()
                    if context and len(context) > max_chars:
                        context = context[:max_chars].rstrip() + "\n...[truncated]"
        except Exception as e:
            logger.debug("KB auto-retrieval failed: %s", e)
            context = ""

        if cache_key:
            self._kb_auto_context_cache[cache_key] = (time.time(), context)
            # Bounded cache size
            if len(self._kb_auto_context_cache) > 256:
                oldest_key = min(self._kb_auto_context_cache, key=lambda k: self._kb_auto_context_cache[k][0])
                del self._kb_auto_context_cache[oldest_key]
        return context

    @staticmethod
    def _resolve_kb_auto_context_mode(kb_cfg: dict) -> str:
        """Resolve KB auto-context mode with backwards compatibility."""
        mode = str(kb_cfg.get("auto_context_mode", "")).strip().lower()
        if mode in {"off", "on_demand", "always"}:
            return mode
        # Backward compatibility for old boolean flag.
        if "auto_context" in kb_cfg:
            return "always" if kb_cfg.get("auto_context", True) else "off"
        # SaaS-safe default: use KB only when query implies document grounding.
        return "on_demand"

    @staticmethod
    def _should_use_kb_for_query(query: str) -> bool:
        """Heuristic: use KB only when user asks about docs/KB/books/files."""
        q = (query or "").lower()
        markers = (
            "база знаний",
            "в базе",
            "в учебнике",
            "в документ",
            "из документа",
            "из учебника",
            "из файла",
            "по файлу",
            "по документ",
            "согласно документу",
            "в pdf",
            "страница",
            "глава",
            "цитату",
            "цитируй",
            "knowledge base",
            "in the knowledge base",
            "in the document",
            "from the document",
            "from the file",
            "from the pdf",
            "according to the book",
            "according to the document",
            "cite",
            "chapter",
            "page",
        )
        if any(m in q for m in markers):
            return True
        # Explicit file mention, e.g. open-logic-complete.pdf
        if any(ext in q for ext in (".pdf", ".md", ".txt", ".html", ".csv", ".json")):
            return True
        return False

    @staticmethod
    def _is_personal_memory_query(query: str) -> bool:
        """Detect queries that should prioritize user memory over KB documents."""
        q = (query or "").strip().lower()
        markers = (
            "как меня зовут",
            "мое имя",
            "моё имя",
            "кто я",
            "что ты знаешь обо мне",
            "помнишь меня",
            "мой день рождения",
            "моя дата рождения",
            "where i live",
            "where do i live",
            "my name",
            "what is my name",
            "who am i",
            "what do you know about me",
            "remember my name",
        )
        return any(m in q for m in markers)

    @staticmethod
    def _classify_profile_slot_query(query: str) -> str:
        """Return canonical profile slot key for direct personal-profile queries."""
        q = " ".join((query or "").strip().lower().split())
        if not q:
            return ""
        slot_markers = {
            "name": (
                "как меня зовут", "мое имя", "моё имя", "my name",
                "what is my name", "remember my name",
            ),
            "language": (
                "на каком языке", "мой язык", "какой язык",
                "which language", "language do i prefer",
            ),
            "role": (
                "кто я по роли", "моя роль", "кем я работаю",
                "my role", "what is my role", "what do i do",
            ),
        }
        for slot, markers in slot_markers.items():
            if any(m in q for m in markers):
                return slot
        return ""

    @staticmethod
    def _is_personal_memory_summary_query(query: str) -> bool:
        """Detect direct requests asking what agent remembers about the user."""
        q = " ".join((query or "").strip().lower().split())
        if not q:
            return False
        markers = (
            "что ты помнишь обо мне",
            "что ты знаешь обо мне",
            "что ты помнишь про меня",
            "помнишь обо мне",
            "что помнишь обо мне",
            "who am i for you",
            "what do you remember about me",
            "what do you know about me",
            "remember me",
            "about me",
        )
        return any(m in q for m in markers)

    def _direct_personal_memory_summary(self, user_input: str, user_id: str) -> str | None:
        """Deterministic personal memory summary from profile + retrieved memories."""
        if not self._is_personal_memory_summary_query(user_input):
            return None
        if self._classify_profile_slot_query(user_input):
            return None

        lines: list[str] = []
        try:
            profile = self.memory.get_user_profile(user_id) or {}
        except Exception:
            profile = {}

        name = str(profile.get("name", "")).strip()
        role = str(profile.get("role", "")).strip()
        location = str(profile.get("location", "")).strip()
        language = str(profile.get("language", "")).strip()
        if name:
            lines.append(f"Тебя зовут {name}.")
        if role:
            role_line = role
            if role_line.lower().startswith("с "):
                role_line = f"работаешь {role_line}"
            elif role_line.lower().startswith("работ"):
                role_line = f"{role_line}"
            lines.append(f"Ты {role_line}.")
        if location:
            lines.append(f"Ты живешь в {location}.")
        if language:
            low_lang = language.lower()
            if low_lang in {"ru", "русский", "russian"}:
                lines.append("Предпочитаемый язык: русский.")
            elif low_lang in {"en", "english", "английский"}:
                lines.append("Предпочитаемый язык: английский.")
            else:
                lines.append(f"Предпочитаемый язык: {language}.")

        try:
            all_mem = self.memory.get_all_memories(user_id)
        except Exception:
            all_mem = []

        resolved_name = ""
        if name:
            resolved_name = name.lower()
        elif hasattr(self.memory, "resolve_profile_slot"):
            try:
                resolved_name = str(
                    (self.memory.resolve_profile_slot(user_id, "name", lookback=350, auto_heal=False) or {}).get("value", "")
                ).strip().lower()
            except Exception:
                resolved_name = ""

        def _score(item: dict, idx: int, total: int) -> float:
            imp = float(item.get("importance", 0.5) or 0.5)
            typ = str(item.get("type", "fact") or "fact")
            type_bonus = {"fact": 0.35, "preference": 0.22, "correction": 0.1}.get(typ, 0.0)
            rec = max(0.05, 1.0 - (idx / max(total, 1)))
            return imp * 1.8 + type_bonus + rec * 0.65

        def _parse_dt(v: str) -> datetime | None:
            s = str(v or "").strip()
            if not s:
                return None
            try:
                # sqlite rows usually store naive ISO; treat as UTC-like for relative age.
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return None

        filtered: list[tuple[float, str, bool]] = []
        now = datetime.now()
        total = max(len(all_mem), 1)
        for idx, item in enumerate(all_mem[:900]):
            content = " ".join(str(item.get("content", "")).strip().split())
            if not content:
                continue
            low = content.lower()
            if len(low) < 10:
                continue
            if any(ch in low for ch in ("{", "}", "__tool_", "```")):
                continue
            if any(p in low for p in (
                "assistant response", "assistant says", "as an ai", "как ии",
                "i only remember this chat", "только в текущей сессии",
                "между сессиями информация не сохраняется",
                "system_does_not_store_data_between_sessions",
                "does_not_have_long_term_memory",
            )):
                continue
            if hasattr(self.memory, "_is_memory_pollution_text"):
                try:
                    if self.memory._is_memory_pollution_text(content):
                        continue
                except Exception:
                    pass
            if hasattr(self.memory, "_is_self_referential_memory_limit"):
                try:
                    if self.memory._is_self_referential_memory_limit(content):
                        continue
                except Exception:
                    pass
            # Avoid surfacing stale contradictory "name is X" items for other names.
            if resolved_name and any(k in low for k in ("имя", "зовут", "name")) and resolved_name not in low:
                continue
            # Profile-level language/name are already emitted above.
            if language and any(k in low for k in ("language:", "язык общения", "preferred language", "предпочитаемый язык")):
                continue
            if resolved_name and any(k in low for k in ("имя", "зовут", "name")):
                continue
            created = _parse_dt(item.get("created_at", ""))
            age_h = 0.0
            if created:
                try:
                    age_h = max(0.0, (now - created.replace(tzinfo=None)).total_seconds() / 3600.0)
                except Exception:
                    age_h = 0.0
            is_historic = age_h >= 48.0
            score = _score(item, idx, total) + (0.18 if is_historic else 0.0)
            filtered.append((score, content, is_historic))

        filtered.sort(key=lambda x: x[0], reverse=True)

        import re as _re

        def _wordset(s: str) -> set[str]:
            return {
                w for w in _re.findall(r"[A-Za-zА-Яа-яЁё0-9_]+", s.lower())
                if len(w) >= 4
            }

        chosen: list[str] = []
        chosen_sets: list[set[str]] = []

        def _add_if_diverse(content: str) -> bool:
            ws = _wordset(content)
            if not ws:
                return False
            for ex in chosen_sets:
                inter = len(ws & ex)
                union = len(ws | ex) or 1
                if (inter / union) >= 0.6:
                    return False
            chosen.append(content)
            chosen_sets.append(ws)
            return True

        # OpenClaw-like balance: keep both recent and older stable memories.
        historic_quota = 2
        historic_added = 0
        for _, content, is_historic in filtered:
            if not is_historic:
                continue
            if _add_if_diverse(content):
                historic_added += 1
                if historic_added >= historic_quota:
                    break

        for _, content, _ in filtered:
            if len(chosen) >= 6:
                break
            _add_if_diverse(content)

        for content in chosen:
            lines.append(content.rstrip(".") + ".")
            if len(lines) >= 7:
                break

        if not lines:
            return "Пока помню о вас мало. Скажите, что важно запомнить, и я буду на это опираться."
        return "Помню о вас:\n- " + "\n- ".join(lines[:6])

    @staticmethod
    def _is_historical_request_query(query: str) -> bool:
        """Detect requests about prior-day/previous-session user asks."""
        q = " ".join((query or "").strip().lower().split())
        if not q:
            return False
        markers = (
            "что я просил", "что я просил разработать", "что мы делали",
            "о чем мы говорили", "о чём мы говорили", "что я говорил",
            "что просил вчера", "что я просил вчера", "вчера",
            "позавчера", "last time", "yesterday", "what did i ask",
            "what i asked yesterday", "what did we discuss",
        )
        return any(m in q for m in markers)

    def _direct_historical_request_answer(self, user_input: str, user_id: str) -> str | None:
        """Deterministic answer for 'what did I ask yesterday' queries from chat history."""
        if not self._is_historical_request_query(user_input):
            return None
        if not getattr(self, "memory", None):
            return None

        import re as _re
        from datetime import timedelta as _td

        q = " ".join((user_input or "").strip().lower().split())
        now = datetime.now()
        target_date = None
        if "позавчера" in q:
            target_date = (now - _td(days=2)).date()
        elif "вчера" in q or "yesterday" in q:
            target_date = (now - _td(days=1)).date()

        try:
            rows = self.memory.db.execute(
                """SELECT content, created_at
                   FROM chat_history
                   WHERE user_id = ? AND role = 'user'
                   ORDER BY id DESC LIMIT 600""",
                (user_id,),
            ).fetchall()
        except Exception:
            rows = []

        if not rows:
            return "Не нашла предыдущие запросы в истории."

        noise_patterns = (
            "как меня зовут", "что ты помнишь", "что ты знаешь обо мне",
            "remember", "about me", "кто я", "помнишь меня",
        )
        task_markers = (
            "сделай", "добав", "реализ", "исправ", "почини", "перезапусти", "обнови",
            "проверь", "пофикс", "внедри", "настрой", "создай", "аудит", "памят",
            "implement", "add", "fix", "build", "create", "restart", "check",
        )
        prefer_dev = any(k in q for k in ("разработ", "сделал", "implement", "build", "create", "fix"))
        dev_markers = (
            "dashboard", "дашборд", "виджет", "memory", "памят", "monitor", "worker",
            "daemon", "upload", "документ", "база знаний", "knowledge base", "api",
            "настройк", "websocket", "kb", "tool", "agent",
        )

        def _parse_dt(v: str) -> datetime | None:
            s = str(v or "").strip()
            if not s:
                return None
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return None

        candidates: list[str] = []
        for row in rows:
            content = " ".join(str((row or [""])[0] or "").strip().split())
            if not content:
                continue
            low = content.lower()
            if low == q:
                continue
            if any(p in low for p in noise_patterns):
                continue
            if len(low) < 8:
                continue
            dt = _parse_dt((row or ["", ""])[1] or "")
            if target_date and dt and dt.date() != target_date:
                continue
            if prefer_dev and not any(m in low for m in task_markers):
                continue
            if prefer_dev and not any(m in low for m in dev_markers):
                continue
            # Keep actionable task-ish user requests.
            if not any(m in low for m in task_markers):
                continue
            # Avoid raw dump-like wrappers
            if _re.search(r"^\[user sent a voice message", low):
                continue
            candidates.append(content)
            if len(candidates) >= 12:
                break

        # Fallback: if strict filter was empty for target_date, loosen to last 7 days.
        if not candidates:
            for row in rows:
                content = " ".join(str((row or [""])[0] or "").strip().split())
                low = content.lower()
                if not content or low == q:
                    continue
                if any(p in low for p in noise_patterns):
                    continue
                if len(low) < 8:
                    continue
                dt = _parse_dt((row or ["", ""])[1] or "")
                if dt and (now - dt.replace(tzinfo=None)).days > 7:
                    continue
                if any(m in low for m in task_markers):
                    candidates.append(content)
                if len(candidates) >= 10:
                    break

        # Final fallback: mine extracted long-term memories for task requests.
        if not candidates:
            mem_markers = (
                "user requested", "requested that", "user wants", "wants to",
                "пользователь запросил", "пользователь хочет", "пользователь попросил",
                "я просил", "я попросил", "попросил", "просил", "попросила", "хочу",
            )
            try:
                mrows = self.memory.db.execute(
                    """SELECT content, created_at
                       FROM memories
                       WHERE user_id = ? AND archived_at IS NULL
                         AND type IN ('fact', 'correction', 'preference')
                       ORDER BY id DESC LIMIT 800""",
                    (user_id,),
                ).fetchall()
            except Exception:
                mrows = []

            dated: list[str] = []
            recent: list[str] = []
            for row in mrows:
                content = " ".join(str((row or [""])[0] or "").strip().split())
                low = content.lower()
                if not content or len(low) < 10:
                    continue
                if any(p in low for p in noise_patterns):
                    continue
                dt = _parse_dt((row or ["", ""])[1] or "")
                if dt and (now - dt.replace(tzinfo=None)).days > 10:
                    continue
                if not any(m in low for m in mem_markers):
                    continue
                if prefer_dev and not any(m in low for m in dev_markers):
                    continue
                if hasattr(self.memory, "_is_memory_pollution_text"):
                    try:
                        if self.memory._is_memory_pollution_text(content):
                            continue
                    except Exception:
                        pass
                if target_date and dt and dt.date() == target_date:
                    dated.append(content)
                else:
                    recent.append(content)
                if (len(dated) + len(recent)) >= 18:
                    break
            candidates.extend(dated if dated else recent)

        if not candidates:
            if target_date:
                label = "вчера" if ("вчера" in q or "yesterday" in q) else "позавчера"
                return f"В истории не нашла явных запросов на {label}. Могу показать последние задачи за 7 дней."
            return "В истории не нашла явных прошлых задач."

        uniq: list[str] = []
        seen: set[str] = set()
        for c in candidates:
            key = c.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
            if len(uniq) >= 5:
                break

        def _humanize(line: str) -> str:
            s = " ".join(str(line or "").strip().split())
            low = s.lower()
            if low.startswith("user wants "):
                return "Вы хотели " + s[11:]
            if low.startswith("the user wants "):
                return "Вы хотели " + s[15:]
            if low.startswith("user requested "):
                return "Вы просили " + s[15:]
            if low.startswith("user requested that "):
                return "Вы просили " + s[20:]
            if low.startswith("the user requested "):
                return "Вы просили " + s[19:]
            if low.startswith("пользователь запросил "):
                return "Вы просили " + s[len("пользователь запросил "):]
            if low.startswith("пользователь хочет "):
                return "Вы хотели " + s[len("пользователь хочет "):]
            return s

        title = "Вчера вы просили:" if ("вчера" in q or "yesterday" in q) else "Помню ваши прошлые запросы:"
        return title + "\n- " + "\n- ".join(_humanize(x) for x in uniq)

    def _sanitize_memory_limit_response(self, text: str, user_input: str, user_id: str) -> str:
        """Replace false 'no memory' disclaimers with deterministic memory-based answer."""
        s = " ".join(str(text or "").strip().lower().split())
        if not s:
            return text
        disclaimers = (
            "нет доступа к истории", "не могу помнить", "не помню прошлые разговоры",
            "нет долгосрочной памяти", "только текущ", "only remember this chat",
            "don't remember previous conversations", "no long-term memory",
        )
        if not any(p in s for p in disclaimers):
            return text

        repaired = (
            self._direct_historical_request_answer(user_input, user_id)
            or self._direct_personal_memory_summary(user_input, user_id)
            or self._direct_profile_memory_answer(user_input, user_id)
        )
        return repaired or text

    @staticmethod
    def _looks_like_side_effect_request(user_input: str) -> bool:
        s = " ".join(str(user_input or "").strip().lower().split())
        if not s:
            return False
        markers = (
            "сделай", "исправ", "добав", "обнов", "перезапуст", "запусти",
            "удали", "реализ", "внедри", "измени", "отредактируй",
            "fix ", "implement", "add ", "update ", "restart ", "run ", "delete ", "modify ",
        )
        if any(m in s for m in markers):
            return True
        # Requests mentioning concrete code paths/endpoints are often side-effect prone,
        # but generic nouns like "dashboard" or "project" are too noisy.
        return any(x in s for x in (
            ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css",
            "/api/", "localhost", "http://", "https://",
            "docker", "nginx", "frontend", "backend", "index.html", "main.py", "static/",
        ))

    @staticmethod
    def _contains_completion_claim(text: str) -> bool:
        s = " ".join(str(text or "").strip().lower().split())
        if not s:
            return False
        done_markers = (
            "готово", "выполнено", "сделано", "исправил", "добавил", "обновил",
            "перезапустил", "запустил", "реализовал", "пофиксил", "внедрил",
            "done", "completed", "fixed", "implemented", "updated", "restarted", "finished",
        )
        return any(m in s for m in done_markers)

    @staticmethod
    def _looks_like_side_effect_response(text: str) -> bool:
        """Detect unverified claims about created/updated/running project artifacts."""
        s = " ".join(str(text or "").strip().lower().split())
        if not s:
            return False
        effect_markers = (
            "доступен", "работает", "запущен", "перезапущен", "создан", "добавлен",
            "обновл", "подключ", "отда", "исправлен", "исправлено",
            "available", "running", "restarted", "created", "added", "updated",
            "served", "mounted", "fixed",
        )
        artifact_markers = (
            "frontend", "backend", "server", "container", "docker", "nginx",
            "index.html", "main.py", "static/", "api", "route", "localhost",
            "http://", "https://", "фронтенд", "бэкенд", "сервер", "контейнер",
            "статик", "маршрут", "страница", "проект", "файл",
        )
        return any(m in s for m in effect_markers) and any(m in s for m in artifact_markers)

    def _sanitize_unverified_completion_response(self, text: str, user_input: str,
                                                 tool_calls_log: list[dict] | None) -> str:
        """Block unverified 'done/completed' claims when no successful tool execution exists."""
        s = " ".join(str(text or "").strip().lower().split())
        if not s:
            return text
        if self._looks_like_telegram_delivery_request(user_input) and self._looks_like_telegram_delivery_claim(text):
            if self._has_verified_telegram_delivery(tool_calls_log):
                return text
            return (
                "Промежуточный статус: отправка в Telegram пока не подтверждена. "
                "Сначала выполню реальную доставку в Telegram и только потом подтвержу результат."
            )
        if not self._looks_like_side_effect_request(user_input) and not self._looks_like_side_effect_response(text):
            return text
        if not self._contains_completion_claim(text) and not self._looks_like_side_effect_response(text):
            return text

        negative_markers = (
            "не выполн", "не сделал", "не удалось", "ошибка", "failed",
            "not completed", "not done", "couldn't", "cannot",
        )
        if any(m in s for m in negative_markers):
            return text

        has_successful_tool = any(
            isinstance(tc, dict) and not bool(tc.get("error"))
            for tc in (tool_calls_log or [])
        )
        if has_successful_tool:
            return text

        return (
            "Промежуточный статус: выполнение пока не подтверждено инструментами. "
            "Запущу нужные действия и вернусь с проверенным результатом."
        )

    @staticmethod
    def _looks_like_telegram_delivery_request(text: str) -> bool:
        s = " ".join(str(text or "").strip().lower().split())
        if not s:
            return False
        return any(marker in s for marker in (
            "в телеграм", "в telegram", "to telegram", "send to telegram",
            "отправь мне в телеграм", "пришли в телеграм", "telegram",
        ))

    @staticmethod
    def _looks_like_telegram_delivery_claim(text: str) -> bool:
        s = " ".join(str(text or "").strip().lower().split())
        if not s:
            return False
        delivery_markers = (
            "отправлено в telegram", "отправлено в телеграм", "отправил в telegram",
            "отправил в телеграм", "сообщение отправлено", "message sent to telegram",
            "sent to telegram", "успешно отправлено в telegram", "успешно отправлено в телеграм",
            "успешно отправлены", "отправлены в @", "отправлены в `@",
        )
        return any(marker in s for marker in delivery_markers)

    @staticmethod
    def _has_verified_telegram_delivery(tool_calls_log: list[dict] | None) -> bool:
        allowed_tools = {"send_text_to_user", "send_file_to_user", "send_stored_file", "send_stored_file_to_telegram"}
        for tc in tool_calls_log or []:
            if not isinstance(tc, dict) or bool(tc.get("error")):
                continue
            name = str(tc.get("name") or tc.get("tool_name") or "").strip()
            if name in allowed_tools:
                return True
        return False

    @staticmethod
    def _is_status_only_tool_guard_response(text: str) -> bool:
        s = " ".join(str(text or "").strip().lower().split())
        if not s:
            return False
        return any(marker in s for marker in (
            "выполнение пока не подтверждено инструментами",
            "not yet confirmed by tools",
            "execution is not yet confirmed by tools",
        ))

    def _should_force_tool_continuation(
        self,
        text: str,
        user_input: str,
        tool_calls_log: list[dict] | None,
        forced_attempts: int,
    ) -> bool:
        if forced_attempts >= 1:
            return False
        if not self._looks_like_side_effect_request(user_input):
            return False
        if not self._is_status_only_tool_guard_response(text):
            return False
        return not any(
            isinstance(tc, dict) and not bool(tc.get("error"))
            for tc in (tool_calls_log or [])
        )

    @staticmethod
    def _forced_tool_continuation_prompt() -> str:
        return (
            "[System] Your previous reply was only a status update without any real tool execution. "
            "Do not summarize or promise future work. Start doing the task now with actual tools. "
            "Inspect the workspace, create or edit files, run commands, and only finalize after at least "
            "one real tool_result."
        )

    def _should_force_no_tool_recovery(
        self,
        user_input: str,
        tool_calls_log: list[dict] | None,
        forced_attempts: int,
        no_tool_passes: int,
    ) -> bool:
        s = " ".join(str(user_input or "").strip().lower().split())
        imperative_markers = (
            "сделай", "исправ", "добав", "обнов", "перезапуст", "запусти",
            "удали", "реализ", "внедри", "измени", "отредактируй",
            "fix ", "implement", "add ", "update ", "restart ", "run ", "delete ", "modify ",
        )
        if forced_attempts >= 2:
            return False
        if no_tool_passes < 1:
            return False
        if not self._looks_like_side_effect_request(user_input):
            return False
        if not any(marker in s for marker in imperative_markers):
            return False
        return not any(
            isinstance(tc, dict) and not bool(tc.get("error"))
            for tc in (tool_calls_log or [])
        )

    @staticmethod
    def _forced_no_tool_recovery_prompt() -> str:
        return (
            "[System] You just answered a side-effect task without using any real tools. "
            "Stop planning and stop summarizing. In the very next turn, call concrete tools immediately: "
            "inspect files, run verification commands, and edit the project where needed. "
            "Do not return another analysis-only answer."
        )

    @staticmethod
    def _looks_like_permission_seeking_response(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        patterns = (
            "хочешь, чтобы я",
            "хотите, чтобы я",
            "подтверди",
            "подтвердите",
            "мне продолжать",
            "могу продолжить, если",
            "нужно ваше разрешение",
            "нужно подтверждение",
            "should i ",
            "do you want me to",
            "would you like me to",
            "please confirm",
            "confirm that i should",
            "can you confirm",
            "i can continue if",
            "need your permission",
            "need confirmation",
        )
        return any(p in normalized for p in patterns)

    @classmethod
    def _permission_question_has_real_blocker(cls, text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        blocker_markers = (
            "api key", "token", "credentials", "credential", "password",
            "login", "auth", "authentication", "access denied", "permission denied",
            "403", "401", "chat id", "bot token",
            "ключ api", "токен", "учетные данные", "пароль",
            "логин", "доступ", "нет доступа", "permission denied",
            "необратим", "irreversible", "destructive", "удалени",
        )
        return any(marker in normalized for marker in blocker_markers)

    def _should_force_autonomy_recovery(
        self,
        user_input: str,
        response_text: str,
        tool_calls_log: list[dict] | None,
        forced_attempts: int,
    ) -> bool:
        if forced_attempts >= 2:
            return False
        if not self._looks_like_side_effect_request(user_input):
            return False
        if not self._looks_like_permission_seeking_response(response_text):
            return False
        if self._permission_question_has_real_blocker(response_text):
            return False
        return not any(
            isinstance(tc, dict) and not bool(tc.get("error"))
            for tc in (tool_calls_log or [])
        )

    @staticmethod
    def _forced_autonomy_recovery_prompt() -> str:
        return (
            "[System] Do not ask the user for routine permission or confirmation here. "
            "Before your next action, do a silent critical review: "
            "1) restate the objective, 2) inspect available context, memory, config, and workspace, "
            "3) make only the minimum safe assumptions needed to proceed, "
            "4) choose the smallest reversible action, "
            "5) verify the result after acting. "
            "Only ask the user if you are truly blocked by missing credentials/access, an irreversible destructive choice, "
            "or conflicting requirements that cannot be resolved from context. "
            "Now continue autonomously and call real tools in the very next reply."
        )

    @staticmethod
    def _looks_like_high_stakes_topic(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        markers = (
            "medical", "medicine", "health", "diagnosis", "symptom", "treatment",
            "legal", "law", "contract", "visa", "immigration", "tax", "deadline",
            "finance", "financial", "investment", "bank", "loan", "insurance",
            "security", "credential", "password", "token", "delete", "destructive",
            "медицин", "здоров", "симптом", "лечение",
            "юрид", "закон", "договор", "виза", "иммиграц", "налог", "срок",
            "финанс", "инвест", "банк", "страхов",
            "безопас", "парол", "токен", "удален", "удалени", "необратим",
        )
        return any(marker in normalized for marker in markers)

    def _should_run_critical_response_review(
        self,
        user_input: str,
        response_text: str,
        tool_calls_log: list[dict] | None,
        original_input=None,
    ) -> bool:
        cfg = self._features.get("critical_response_review", {})
        if not cfg.get("enabled"):
            return False
        draft = str(response_text or "").strip()
        request = str(user_input or "").strip()
        if len(draft) < 20:
            return False
        if self._is_status_only_tool_guard_response(draft):
            return False

        try:
            min_complexity = int(cfg.get("min_complexity", 3) or 3)
        except Exception:
            min_complexity = 3
        try:
            min_response_chars = int(cfg.get("min_response_chars", 220) or 220)
        except Exception:
            min_response_chars = 220

        complexity = self._complexity_score(request) if request else 0
        has_tool_activity = bool(tool_calls_log)
        request_high_stakes = self._looks_like_high_stakes_topic(request)
        draft_high_stakes = self._looks_like_high_stakes_topic(draft)
        high_stakes = request_high_stakes or draft_high_stakes
        side_effect = self._looks_like_side_effect_request(request)
        if isinstance(original_input, list) and not any((side_effect, has_tool_activity, request_high_stakes)):
            return False
        return any((
            side_effect,
            has_tool_activity,
            complexity >= min_complexity,
            len(draft) >= min_response_chars,
            high_stakes,
        ))

    def _critical_review_tool_evidence(
        self,
        tool_calls_log: list[dict] | None,
        max_items: int,
    ) -> list[str]:
        evidence = []
        for call in tool_calls_log or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or call.get("tool_name") or "tool").strip() or "tool"
            status = "error" if bool(call.get("error")) else "ok"
            preview = " ".join(str(call.get("result_preview", "") or "").split())
            if len(preview) > 160:
                preview = preview[:157].rstrip() + "..."
            evidence.append(f"{name}[{status}] -> {preview or 'no preview'}")
            if len(evidence) >= max_items:
                break
        return evidence

    async def _critical_review_response_if_needed(
        self,
        *,
        user_input: str,
        response_text: str,
        user_id: str,
        tool_calls_log: list[dict] | None,
        model: str,
        original_input=None,
    ) -> str:
        cfg = dict(self._features.get("critical_response_review", {}) or {})
        meta = {"enabled": bool(cfg.get("enabled")), "applied": False, "revised": False, "issues": []}
        self._last_response_meta["critical_review"] = meta

        if not self._should_run_critical_response_review(
            user_input, response_text, tool_calls_log, original_input=original_input
        ):
            return response_text

        from .planning import resolve_planning_model

        review_model_spec = str(cfg.get("model", "") or "").strip()
        if not review_model_spec or review_model_spec == "auto":
            review_model_spec = resolve_planning_model(
                self.provider,
                {"_default_model": self.default_model},
            )

        provider_obj = self.provider
        current_provider = str(self.config.get("agent", {}).get("provider", "anthropic")).strip().lower()
        review_provider, review_model = self._split_model_spec(review_model_spec)
        if review_provider != current_provider:
            temp_cfg = copy.deepcopy(self.config)
            temp_cfg.setdefault("agent", {})["provider"] = review_provider
            try:
                provider_obj = create_provider(temp_cfg)
            except Exception as e:
                logger.debug("Critical review provider init failed: %s", e)
                return response_text

        try:
            timeout_sec = float(cfg.get("timeout_sec", 20.0) or 20.0)
        except Exception:
            timeout_sec = 20.0
        try:
            max_issues = max(1, min(int(cfg.get("max_issues", 3) or 3), 5))
        except Exception:
            max_issues = 3
        try:
            max_tool_evidence = max(0, min(int(cfg.get("max_tool_evidence", 4) or 4), 8))
        except Exception:
            max_tool_evidence = 4

        evidence = self._critical_review_tool_evidence(tool_calls_log, max_tool_evidence)
        prompt = (
            "You are an internal critical reviewer for an AI assistant.\n"
            "Review the draft answer before it is shown to the user.\n"
            "Find only material issues: overclaiming, contradictions, unsupported certainty, "
            "missing caveats for high-stakes topics, or statements not backed by tool evidence.\n"
            "Preserve the same language and overall tone. Keep the answer concise.\n"
            "If the draft is already good enough, do not rewrite it.\n\n"
            f"User request:\n{str(user_input or '')[:1800]}\n\n"
            f"Current model:\n{model}\n\n"
            f"Draft answer:\n{str(response_text or '')[:5000]}\n\n"
            f"Tool evidence:\n" + ("\n".join(f"- {item}" for item in evidence) if evidence else "- no tool evidence") + "\n\n"
            "Return JSON only with this schema:\n"
            '{"needs_revision": true|false, "issues": ["issue1"], "revised_answer": "text"}\n'
            f"Rules:\n- issues: at most {max_issues}\n"
            "- If there is no material issue, set needs_revision=false and revised_answer=\"\".\n"
            "- Never invent new facts. If support is weak, reduce certainty instead.\n"
            "- Do not mention that this was an internal review."
        )

        try:
            review_response = await asyncio.wait_for(
                provider_obj.complete(
                    model=review_model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                    system="Return JSON only. Do not use tools.",
                    tools=None,
                    temperature=0,
                ),
                timeout=timeout_sec,
            )
            if getattr(review_response, "usage", None):
                try:
                    review_cost = self._calculate_cost(review_model, review_response.usage)
                    self.memory.track_usage(user_id, review_model, review_response.usage, review_cost)
                except Exception:
                    logger.debug("Failed to track critical review usage", exc_info=True)
            parsed = self._extract_json_object(self._extract_text(review_response))
        except Exception as e:
            logger.debug("Critical response review failed: %s", e)
            return response_text

        issues = [
            str(item).strip() for item in (parsed.get("issues") or [])
            if str(item).strip()
        ][:max_issues]
        revised = str(parsed.get("revised_answer") or "").strip()
        needs_revision = bool(parsed.get("needs_revision"))
        meta["applied"] = True
        meta["issues"] = issues

        if needs_revision and revised:
            meta["revised"] = True
            return revised
        return response_text

    def _direct_profile_memory_answer(self, user_input: str, user_id: str) -> str | None:
        """Deterministic answer for profile-slot queries to avoid LLM memory misses."""
        slot = self._classify_profile_slot_query(user_input)
        if not slot:
            return None

        value = ""
        if hasattr(self.memory, "resolve_profile_slot"):
            try:
                resolved = self.memory.resolve_profile_slot(user_id, slot, lookback=320, auto_heal=True) or {}
                value = str(resolved.get("value", "") or "").strip()
            except Exception:
                pass
        if hasattr(self.memory, "get_canonical_slot"):
            try:
                cslot = self.memory.get_canonical_slot(user_id, slot)
                if (not value) and cslot and cslot.get("slot_value"):
                    value = str(cslot.get("slot_value", "")).strip()
            except Exception:
                pass
        if not value:
            try:
                profile = self.memory.get_user_profile(user_id) or {}
                value = str(profile.get(slot, "")).strip()
            except Exception:
                value = ""
        if value and hasattr(self.memory, "is_slot_value_contradicted"):
            try:
                if self.memory.is_slot_value_contradicted(user_id, slot, value):
                    value = ""
            except Exception:
                pass

        if slot == "name":
            if not value:
                return "В памяти пока нет вашего имени. Напишите: «Меня зовут ...» или «<Имя> запиши в память»."
            return f"Помню: вас зовут {value}."
        if slot == "language":
            if not value:
                return "В памяти пока нет предпочитаемого языка."
            return f"Помню: ваш предпочитаемый язык — {value}."
        if slot == "role":
            if not value:
                return "В памяти пока нет вашей роли."
            return f"Помню: ваша роль — {value}."
        return None

    def _direct_telegram_target_guidance(self, user_input: str, user_id: str) -> str | None:
        """Explain that a bot username is not a destination chat identifier."""
        text = str(user_input or "").strip()
        if not text or not self._looks_like_telegram_bot_username(text):
            return None

        return (
            "Это username самого Telegram-бота, а не destination chat_id. "
            "Для отправки из обычного чата нужен либо сохраненный numeric chat_id личного диалога с ботом, "
            "либо channel username вида @channelname для канала. "
            "Если ты уже писал этому боту раньше, агент должен использовать связанный Telegram chat автоматически, "
            "а не запрашивать token/chat_id заново и не пытаться слать через curl."
        )

    def _direct_followup_telegram_delivery(self, user_input: str, user_id: str) -> str | None:
        """Send the latest substantive assistant answer to Telegram on short follow-up requests."""
        text = " ".join(str(user_input or "").strip().lower().split())
        if not text or not self._looks_like_telegram_delivery_request(text):
            return None
        followup_markers = (
            "отправь", "пришли", "перешли", "скинь", "попробуй отправить",
            "проробуй", "эти новости", "это в телеграм", "их в телеграм",
        )
        if not any(marker in text for marker in followup_markers):
            return None

        recent_text = self._get_recent_substantive_assistant_text(user_id)
        if not recent_text:
            return None

        sender = self.tools._handlers.get("send_text_to_user")
        if not callable(sender):
            return None

        result = str(sender(recent_text) or "").strip()
        if result == "Message sent to Telegram chat.":
            return "Последний подготовленный ответ отправлен в Telegram."
        if result.startswith("Error:"):
            return f"Не удалось отправить подготовленный ответ в Telegram. {result}"
        return None

    @staticmethod
    def _looks_like_recent_file_location_request(user_input: str) -> bool:
        text = " ".join(str(user_input or "").strip().lower().split())
        if not text:
            return False
        markers = (
            "где он хранится", "где хранится", "где лежит", "where is it stored",
            "where is it saved", "where is it", "storage key", "ключ хранения",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _looks_like_recent_file_send_original_request(user_input: str) -> bool:
        text = " ".join(str(user_input or "").strip().lower().split())
        if not text:
            return False
        markers = (
            "пришли оригинал", "отправь оригинал", "пришли файл", "пришли документ",
            "send original", "send the original", "send file", "send document",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _looks_like_storage_confirmation_notice(user_input: str) -> bool:
        text = str(user_input or "").strip().lower()
        if not text:
            return False
        return (
            "has been auto-saved to storage" in text
            and "indexed for future search" in text
            and "confirm storage to the user" in text
        )

    def _get_recent_user_file(self, user_id: str) -> dict | None:
        fm = self._file_manager
        if not fm:
            return None
        try:
            files = fm.list_files(user_id=user_id, limit=1)
        except Exception:
            return None
        if not files:
            return None
        recent = files[0]
        return recent if isinstance(recent, dict) else None

    def _get_document_unlock_phrase(self) -> str:
        """Return optional passphrase required for sensitive document delivery."""
        from .config import get_api_key

        configured = str((self.config.get("storage", {}) or {}).get("document_unlock_phrase") or "").strip()
        if configured:
            return configured
        return str(get_api_key("document_unlock_phrase") or "").strip()

    @staticmethod
    def _document_unlock_phrase_present(user_input: str, phrase: str) -> bool:
        if not phrase:
            return False
        return phrase.lower() in str(user_input or "").lower()

    @staticmethod
    def _strip_document_unlock_phrase(user_input: str, phrase: str) -> str:
        if not phrase:
            return str(user_input or "")
        import re

        cleaned = re.sub(re.escape(phrase), " ", str(user_input or ""), flags=re.IGNORECASE)
        return " ".join(cleaned.split())

    def _document_unlock_session_active(self, user_id: str) -> bool:
        try:
            state = self.memory.get_state("user:document_unlock_session", user_id=user_id) or {}
        except Exception:
            return False
        if not isinstance(state, dict):
            return False
        expires_at = str(state.get("expires_at") or "").strip()
        if not expires_at:
            return False
        try:
            expires = datetime.fromisoformat(expires_at)
        except Exception:
            return False
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= datetime.now(timezone.utc):
            try:
                self.memory.set_state("user:document_unlock_session", {}, user_id=user_id)
            except Exception:
                pass
            return False
        return True

    def _activate_document_unlock_session(self, user_id: str, ttl_sec: int = 900) -> None:
        now = datetime.now(timezone.utc)
        self.memory.set_state(
            "user:document_unlock_session",
            {
                "active": True,
                "activated_at": now.isoformat(),
                "expires_at": (now + timedelta(seconds=max(60, ttl_sec))).isoformat(),
            },
            user_id=user_id,
        )

    @staticmethod
    def _looks_like_owned_document_request(user_input: str) -> bool:
        text = " ".join(str(user_input or "").strip().lower().split())
        if not text:
            return False
        action_markers = (
            "пришли", "отправь", "вышли", "скинь", "дай", "скачай", "покажи",
            "send", "download", "get", "show", "deliver",
        )
        specific_markers = (
            "паспорт", "документ", "документы", "скан", "pdf",
            "passport", "document", "documents", "scan",
            "id ", " id", "license", "visa", "policy", "полис", "страхов",
            "права", "удостовер", "карточ", "свидетельств",
        )
        generic_file_markers = ("файл", "файлы", "file", "files")
        storage_markers = (
            "из базы", "из хранилища", "из storage", "из s3", "from storage",
            "from s3", "from the database", "из документов", "моих документов",
        )
        has_action = any(marker in text for marker in action_markers)
        has_specific = any(marker in text for marker in specific_markers)
        has_generic_storage = any(marker in text for marker in generic_file_markers) and any(
            marker in text for marker in storage_markers
        )
        return has_action and (has_specific or has_generic_storage)

    @staticmethod
    def _document_query_tokens(query: str) -> list[str]:
        import re

        stop_words = {
            "пришли", "отправь", "вышли", "скинь", "дай", "скачай", "покажи",
            "send", "download", "get", "show", "deliver", "мой", "мои", "моих",
            "my", "me", "мне", "из", "базы", "хранилища", "storage", "from",
            "the", "a", "an", "please", "пожалуйста", "нужен", "нужно", "files",
            "file", "document", "documents", "документ", "документы",
        }
        raw = re.findall(r"[a-zA-Zа-яА-Я0-9_-]+", str(query or "").lower())
        return [token for token in raw if len(token) > 1 and token not in stop_words]

    def _find_owned_document_candidates(self, user_id: str, query: str, limit: int = 5) -> list[dict]:
        fm = self._file_manager
        if not fm:
            return []

        merged: dict[str, dict] = {}
        try:
            for item in fm.search(query, user_id=user_id, top_k=max(limit, 8)) or []:
                if not isinstance(item, dict):
                    continue
                candidate = dict(item)
                candidate["score"] = float(candidate.get("score") or 0.0)
                merged[candidate.get("storage_key") or ""] = candidate
        except Exception:
            pass

        query_tokens = self._document_query_tokens(query)
        if query_tokens:
            try:
                files = fm.list_files(user_id=user_id, limit=200) or []
            except Exception:
                files = []
            for item in files:
                if not isinstance(item, dict):
                    continue
                haystack = " ".join(
                    [
                        str(item.get("original_name") or ""),
                        str(item.get("description") or ""),
                    ]
                ).lower()
                matched = sum(1 for token in query_tokens if token in haystack)
                if not matched:
                    continue
                heuristic_score = 0.25 + (matched / max(len(query_tokens), 1)) * 0.75
                key = str(item.get("storage_key") or "")
                existing = merged.get(key)
                if existing:
                    existing["score"] = max(float(existing.get("score") or 0.0), heuristic_score)
                else:
                    merged[key] = {**item, "score": heuristic_score}

        candidates = [item for key, item in merged.items() if key]
        candidates.sort(
            key=lambda item: (
                float(item.get("score") or 0.0),
                str(item.get("created_at") or ""),
            ),
            reverse=True,
        )
        return candidates[:limit]

    async def _direct_owned_document_delivery(self, user_input: str, user_id: str) -> str | None:
        """Deterministically deliver the user's own stored documents without LLM refusal loops."""
        phrase = self._get_document_unlock_phrase()
        phrase_present = self._document_unlock_phrase_present(user_input, phrase)
        if phrase_present:
            self._activate_document_unlock_session(user_id)

        stripped_input = self._strip_document_unlock_phrase(user_input, phrase)
        if phrase_present and not self._looks_like_owned_document_request(stripped_input):
            return (
                "Кодовое слово принято. В течение ближайших 15 минут могу без лишних вопросов "
                "выдавать твои документы из хранилища."
            )

        if not self._looks_like_owned_document_request(stripped_input):
            return None

        if phrase and not (phrase_present or self._document_unlock_session_active(user_id)):
            return (
                "Для мгновенной выдачи документов из твоего хранилища назови кодовое слово в этом сообщении. "
                "После этого открою короткую unlock-сессию и сразу пришлю нужный файл."
            )

        candidates = self._find_owned_document_candidates(user_id, stripped_input, limit=5)
        if not candidates:
            return "Не нашёл подходящий документ в твоём хранилище. Назови точнее тип документа или часть имени файла."

        top = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None
        top_score = float(top.get("score") or 0.0)
        runner_score = float(runner_up.get("score") or 0.0) if runner_up else 0.0
        if runner_up and top_score < 0.72 and runner_score >= top_score - 0.05:
            options = "\n".join(
                f"• `{item.get('original_name') or item.get('storage_key')}`"
                for item in candidates[:3]
            )
            return (
                "Нашёл несколько похожих документов. Уточни, какой именно нужен:\n"
                f"{options}"
            )

        storage_key = str(top.get("storage_key") or "").strip()
        original_name = str(top.get("original_name") or "").strip() or "document"
        if not storage_key or not self._storage:
            return "Не удалось отправить документ: storage key или хранилище недоступны."

        try:
            import tempfile
            from .file_queue import enqueue_file

            data = await self._storage.async_download(storage_key)
            suffix = os.path.splitext(original_name)[1]
            with tempfile.NamedTemporaryFile(prefix="owned_doc_", suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            enqueue_file(tmp_path, caption=original_name)
            return (
                f"Отправляю твой документ `{original_name}` из хранилища."
                f"\n\nStorage key: `{storage_key}`"
            )
        except Exception as e:
            return f"Не удалось отправить документ `{original_name}`. Ошибка: {e}"

    async def _direct_recent_file_followup(self, user_input: str, user_id: str) -> str | None:
        """Handle simple follow-up questions about the most recent stored file without LLM/tool loop."""
        recent = self._get_recent_user_file(user_id)
        if not recent:
            return None

        original_name = str(recent.get("original_name") or "").strip() or "file"
        storage_key = str(recent.get("storage_key") or "").strip()
        source = str(recent.get("source") or "").strip()
        description = str(recent.get("description") or "").strip()

        if self._looks_like_storage_confirmation_notice(user_input):
            text = f"Документ `{original_name}` сохранён в хранилище и проиндексирован для поиска."
            if storage_key:
                text += f"\n\nStorage key: `{storage_key}`"
            if description:
                text += f"\nОписание: {description}"
            return text

        if self._looks_like_recent_file_location_request(user_input):
            text = f"Последний файл хранится в облачном хранилище как `{original_name}`."
            if storage_key:
                text += f"\n\nStorage key: `{storage_key}`"
            if source:
                text += f"\nИсточник: `{source}`"
            return text

        if self._looks_like_recent_file_send_original_request(user_input):
            if not self._storage or not storage_key:
                return "Не удалось отправить оригинал: хранилище или ключ файла недоступны."
            try:
                import tempfile
                from .file_queue import enqueue_file

                data = await self._storage.async_download(storage_key)
                suffix = os.path.splitext(original_name)[1]
                with tempfile.NamedTemporaryFile(prefix="stored_", suffix=suffix, delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                enqueue_file(tmp_path, caption=original_name)
                return f"Оригинал `{original_name}` поставлен в очередь на отправку."
            except Exception as e:
                return f"Не удалось отправить оригинал `{original_name}`. Ошибка: {e}"

        return None

    @staticmethod
    def _looks_like_markdown_file_delivery_request(user_input: str) -> bool:
        text = " ".join(str(user_input or "").strip().lower().split())
        if not text:
            return False
        action_markers = ("пришли", "отправь", "скинь", "вышли", "дай", "send", "deliver", "attach")
        file_markers = ("markdown", "md файл", "md-файл", ".md", "этот файл", "this file", "markdown file")
        return any(marker in text for marker in action_markers) and any(marker in text for marker in file_markers)

    def _recent_referenced_local_files(self, user_id: str, *, limit: int = 5) -> list[str]:
        try:
            history = list(self.memory.get_history(user_id) or [])
        except Exception:
            return []

        import re

        file_names: list[str] = []
        seen: set[str] = set()
        pattern = re.compile(r"(?:(?:`|\b)([A-Za-z0-9_./-]+\.(?:md|markdown|txt|pdf|json|csv|html))(?:`|\b))", re.IGNORECASE)
        for message in reversed(history):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = self._flatten_message_content_to_text(message.get("content"))
            if not content:
                continue
            for match in pattern.findall(content):
                candidate = str(match or "").strip()
                if not candidate:
                    continue
                key = candidate.casefold()
                if key in seen:
                    continue
                seen.add(key)
                file_names.append(candidate)
                if len(file_names) >= limit:
                    return file_names
        return file_names

    def _resolve_recent_local_file_reference(self, user_id: str) -> str | None:
        def _search_by_basename(basename: str) -> str | None:
            root = os.getcwd()
            skip_dirs = {".git", ".venv", ".audit-venv", "__pycache__", "node_modules", "develop"}
            matches: list[tuple[float, str]] = []
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in skip_dirs]
                if basename in filenames:
                    path = os.path.join(dirpath, basename)
                    try:
                        matches.append((os.path.getmtime(path), path))
                    except OSError:
                        matches.append((0.0, path))
                    if len(matches) >= 10:
                        break
            if not matches:
                return None
            matches.sort(key=lambda item: item[0], reverse=True)
            return matches[0][1]

        for candidate in self._recent_referenced_local_files(user_id, limit=8):
            expanded = os.path.expanduser(candidate)
            if os.path.isabs(expanded) and os.path.isfile(expanded):
                return expanded
            basename = os.path.basename(candidate)
            if not basename:
                continue
            found = _search_by_basename(basename)
            if found:
                return found
        return None

    async def _direct_recent_markdown_file_delivery(self, user_input: str, user_id: str) -> str | None:
        if not self._looks_like_markdown_file_delivery_request(user_input):
            return None
        file_path = self._resolve_recent_local_file_reference(user_id)
        if not file_path:
            return None
        sender = self.tools._handlers.get("send_file_to_user")
        if not callable(sender):
            return None
        result = str(sender(file_path=file_path, caption=os.path.basename(file_path)) or "").strip()
        if result.startswith("File queued for delivery:"):
            return f"Файл `{os.path.basename(file_path)}` поставлен в очередь на отправку."
        if result.startswith("Error:"):
            return f"Не удалось отправить markdown-файл. {result}"
        return None

    def _get_recent_substantive_assistant_text(self, user_id: str) -> str:
        """Return the latest non-status assistant message that can be forwarded to Telegram."""
        try:
            history = list(self.memory.get_history(user_id) or [])
        except Exception:
            return ""

        skip_markers = (
            "отправляю", "промежуточный статус", "выполнение пока не подтверждено",
            "отправка в telegram пока не подтверждена", "message sent to telegram",
            "успешно отправлено в telegram", "успешно отправлены", "проверьте, пожалуйста, бота",
            "для отправки в telegram", "токен telegram не найден",
        )
        for message in reversed(history):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = self._flatten_message_content_to_text(message.get("content"))
            normalized = " ".join(content.strip().lower().split())
            if len(content.strip()) < 80:
                continue
            if any(marker in normalized for marker in skip_markers):
                continue
            return content.strip()
        return ""

    @staticmethod
    def _flatten_message_content_to_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            if "text" in content:
                return str(content.get("text") or "")
            return json.dumps(content, ensure_ascii=False)
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    text = str(block.get("text") or "").strip()
                    if text:
                        parts.append(text)
                else:
                    raw = str(block).strip()
                    if raw:
                        parts.append(raw)
            return "\n".join(parts)
        return str(content or "")

    def _direct_profile_update_ack(self, user_input: str, user_id: str) -> str | None:
        """Immediately persist explicit user profile updates (name/role/language)."""
        text = str(user_input or "").strip()
        if not text or len(text) > 800:
            return None
        if not hasattr(self.memory, "apply_explicit_profile_update"):
            return None
        try:
            updates = self.memory.apply_explicit_profile_update(user_id, text) or {}
        except Exception:
            updates = {}
        if not updates:
            return None

        labels = {
            "name": "ваше имя",
            "language": "предпочитаемый язык",
            "role": "ваша роль",
            "location": "локация",
        }
        parts = []
        for key in ("name", "language", "role", "location"):
            if key in updates:
                val = str((updates.get(key) or {}).get("value") or "").strip()
                if val:
                    parts.append(f"{labels.get(key, key)} — {val}")
        if not parts:
            return None
        return "Запомнил: " + "; ".join(parts) + "."

    @staticmethod
    def _looks_like_telegram_bot_username(text: str) -> bool:
        s = str(text or "").strip()
        if not s.startswith("@") or " " in s:
            return False
        body = s[1:]
        if len(body) < 5:
            return False
        return body.lower().endswith("bot")

    @staticmethod
    def _infer_private_telegram_chat_id_from_user_id(user_id: str | None) -> str | None:
        uid = str(user_id or "").strip()
        if not uid.startswith("tg-"):
            return None
        candidate = uid[3:].strip()
        return candidate if candidate.isdigit() else None

    @staticmethod
    def _is_internal_autonomous_prompt(text: str) -> bool:
        normalized = " ".join(str(text or "").strip().lower().split())
        if not normalized:
            return False
        return (
            normalized.startswith("you are the planner for an autonomous ")
            or normalized.startswith("you are running one autonomous ")
        )

    @staticmethod
    def _prepare_kb_query(query: str) -> str:
        """Extract the most semantic part of user input for KB retrieval."""
        q = (query or "").strip()
        if not q:
            return ""
        lowered = q.lower()
        markers = (
            "что такое", "что ", "как ", "почему ", "зачем ", "где ", "когда ",
            "кто ", "какой ", "какая ", "какие ", "о чем ", "о чём ",
            "what is", "what ", "how ", "why ", "where ", "when ", "which ",
        )
        start = -1
        for m in markers:
            i = lowered.rfind(m)
            if i > start:
                start = i
        if start > 0:
            q = q[start:]
        return q.strip()

    @staticmethod
    def _heuristic_kb_query_english(query: str) -> str:
        """Best-effort RU→EN query hints for English textbooks in KB."""
        import re as _re

        q = (query or "").lower()
        if not _re.search(r"[а-яё]", q):
            return ""

        terms = []
        glossary = (
            ("закон исключенного третьего", "law of excluded middle"),
            ("закон исключённого третьего", "law of excluded middle"),
            ("исключенного третьего", "excluded middle"),
            ("исключённого третьего", "excluded middle"),
            ("закон непротиворечия", "law of non contradiction"),
            ("закон тождества", "law of identity"),
            ("формальная логика", "formal logic"),
            ("логика", "logic"),
            ("квантор", "quantifier"),
            ("семантика", "semantics"),
            ("доказательство", "proof"),
            ("теорема", "theorem"),
            ("глава", "chapter"),
            ("определение", "definition"),
        )
        for ru, en in glossary:
            if ru in q:
                terms.append(en)

        if not terms:
            return ""
        return " ".join(dict.fromkeys(terms))

    @staticmethod
    def _inject_kb_context(content: str | list, kb_context: str, compact: bool = False) -> str | list:
        """Inject KB context into model input while preserving original user message."""
        if not kb_context:
            return content
        if compact:
            guidance = (
                "Ниже контекст из базы знаний. Используй его только для вопросов по документам. "
                "Для персональных вопросов о пользователе приоритет у памяти пользователя.\n\n"
                f"<kb_context>\n{kb_context}\n</kb_context>"
            )
        else:
            guidance = (
                "Контекст из базы знаний ниже используй только когда вопрос относится "
                "к содержимому документов/предметным знаниям. Для персональных вопросов "
                "о пользователе (имя, предпочтения, история диалога) приоритет у памяти пользователя. "
                "Если данных в kb_context недостаточно, явно укажи ограничение. "
                "Не утверждай, что документ недоступен, если информация уже есть в kb_context.\n\n"
                f"<kb_context>\n{kb_context}\n</kb_context>"
            )
        if isinstance(content, list):
            return [{"type": "text", "text": guidance}] + content
        return f"{guidance}\n\nВопрос пользователя:\n{content}"

    def _prune_kb_tools_for_personal_query(self, tool_defs: list, user_input: str) -> list:
        """Disable KB tools for personal-memory questions (name/preferences/etc)."""
        if not self._is_personal_memory_query(user_input):
            return tool_defs
        filtered = [t for t in tool_defs if not str(t.get("name", "")).startswith("kb_")]
        if len(filtered) != len(tool_defs):
            logger.debug("Pruned KB tools for personal query")
        return filtered

    _FILE_SEARCH_TOOLS = frozenset({
        "search_files", "list_all_files", "glob_files", "grep_search",
        "get_file", "get_file_url", "send_stored_file", "send_stored_file_to_telegram",
    })

    def _prune_file_search_for_inline_media(
        self, tool_defs: list, content_for_api
    ) -> list:
        """When the current message already contains inline media (image/document
        content blocks), suppress file-storage search tools.
        This prevents the agent from searching the file store for a file that is
        already present in the conversation, which causes contradictory responses.
        """
        has_media = False
        if isinstance(content_for_api, list):
            for block in content_for_api:
                if isinstance(block, dict) and block.get("type") in (
                    "image", "document", "image_url"
                ):
                    has_media = True
                    break
        if not has_media:
            return tool_defs
        filtered = [
            t for t in tool_defs
            if str(t.get("name", "")) not in self._FILE_SEARCH_TOOLS
        ]
        if len(filtered) != len(tool_defs):
            logger.debug(
                "Pruned file-search tools for message with inline media (%d removed)",
                len(tool_defs) - len(filtered),
            )
        return filtered

    def _infer_primary_user_id(self) -> str | None:
        """Infer the dominant real user_id from persisted activity."""
        db = getattr(self.memory, "db", None)
        if db is None:
            return None

        stats: dict[str, dict[str, int | str]] = {}

        def _entry(uid: str) -> dict[str, int | str]:
            if uid not in stats:
                stats[uid] = {"chat": 0, "mem": 0, "last": ""}
            return stats[uid]

        try:
            chat_rows = db.execute(
                """SELECT user_id, COUNT(*), MAX(created_at)
                   FROM chat_history
                   GROUP BY user_id"""
            ).fetchall()
            for uid, count, last in chat_rows:
                uid = str(uid or "").strip()
                if not uid:
                    continue
                st = _entry(uid)
                st["chat"] = int(count or 0)
                st["last"] = str(last or st["last"] or "")
        except Exception:
            return None

        try:
            mem_rows = db.execute(
                """SELECT user_id, COUNT(*), MAX(created_at)
                   FROM memories
                   WHERE archived_at IS NULL
                   GROUP BY user_id"""
            ).fetchall()
            for uid, count, last in mem_rows:
                uid = str(uid or "").strip()
                if not uid:
                    continue
                st = _entry(uid)
                st["mem"] = int(count or 0)
                if str(last or "") > str(st["last"] or ""):
                    st["last"] = str(last or "")
        except Exception:
            return None

        candidates: list[tuple[str, int, str, bool]] = []
        for uid, st in stats.items():
            if uid in self._RESERVED_USER_IDS:
                continue
            score = int(st["chat"]) * 2 + int(st["mem"])
            if score <= 0:
                continue
            candidates.append((uid, score, str(st["last"] or ""), uid.startswith("tg-")))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
        if len(candidates) > 1:
            best = candidates[0]
            second = candidates[1]
            same_group = best[3] == second[3]
            too_close = best[1] < max(3, int(second[1] * 1.1))
            if same_group and too_close:
                return None

        return candidates[0][0]

    def resolve_user_id(self, user_id: str | None) -> str:
        """Resolve channel-level aliases (dashboard/api placeholders) to canonical user_id."""
        raw = str(user_id or "").strip() or "default"

        aliases: dict[str, str] = {}
        for src in (
            self.config.get("memory", {}).get("user_aliases", {}),
            self.config.get("agent", {}).get("user_aliases", {}),
        ):
            if isinstance(src, dict):
                for k, v in src.items():
                    kk = str(k or "").strip()
                    vv = str(v or "").strip()
                    if kk and vv:
                        aliases[kk] = vv

        # Persistent identity map (cross-channel canonical person_id).
        try:
            mapped = self.memory.get_canonical_person_id(raw)
            if mapped and mapped != raw:
                return mapped
        except Exception:
            pass

        explicit = aliases.get(raw)
        if explicit:
            try:
                self.memory.set_user_alias(raw, explicit, source="config", confidence=1.0)
            except Exception:
                pass
            return explicit

        if raw not in self._AUTO_ALIAS_IDS:
            return raw

        now = time.time()
        cached = self._user_resolution_cache.get(raw)
        if cached and now - cached[0] < 10:
            return cached[1]

        inferred = self._infer_primary_user_id()
        resolved = inferred or raw
        self._user_resolution_cache[raw] = (now, resolved)
        if resolved != raw:
            try:
                self.memory.set_user_alias(raw, resolved, source="auto", confidence=0.86)
            except Exception:
                pass
            logger.debug("Resolved user_id alias: %s -> %s", raw, resolved)
        return resolved

    def _set_response_route_meta(self, *, provider: str = "", model: str = "",
                                 requested_model: str | None = None,
                                 mode: str = "", details: dict | None = None) -> None:
        """Persist actual response route for explainability/UI badges."""
        route = dict(self._last_response_meta.get("response_route") or {})
        if provider:
            route["provider"] = provider
        if model:
            route["model"] = model
        if requested_model:
            route["requested_model"] = requested_model
        if mode:
            route["mode"] = mode
        if details:
            route.update({k: v for k, v in details.items() if v not in (None, "", [])})
        if route.get("requested_model") and route.get("model"):
            route["resolved_from_request"] = route["requested_model"] != route["model"]
        if route:
            self._last_response_meta["response_route"] = route

    async def run(self, user_input: str | list, user_id: str = "default",
                  requested_model: str | None = None) -> str:
        """Run agent on user input with per-user serialization."""
        self._last_response_meta = {}
        user_id = self.resolve_user_id(user_id)
        LiteAgent._ensure_locks()
        lock = await self._get_user_lock(user_id)
        q_id = self._track_queued(user_id)
        _timeout = self._lock_timeout
        try:
            await asyncio.wait_for(lock.acquire(), timeout=_timeout)
        except asyncio.TimeoutError:
            logger.warning("Request timeout: user %s waited >%.0fs for lock", user_id, _timeout)
            return "⏳ Request queued too long. Please try again in a moment."
        finally:
            self._untrack_queued(q_id)
        try:
            return await self._run_impl(user_input, user_id, requested_model=requested_model)
        except TimeoutError as exc:
            response = self._build_timeout_recovery_response(
                user_input,
                requested_model=requested_model,
                error=exc,
            )
            payload = response
            if self._last_response_meta:
                payload = {"text": response, "meta": dict(self._last_response_meta)}
            try:
                self.memory.add_message(user_id, "assistant", payload)
            except Exception:
                logger.debug("Failed to persist timeout recovery response", exc_info=True)
            return response
        finally:
            lock.release()

    def _build_timeout_recovery_response(
        self,
        user_input: str | list,
        requested_model: str | None = None,
        error: Exception | None = None,
    ) -> str:
        provider_name = str(self.config.get("agent", {}).get("provider", "")).strip() or "provider"
        model_name = (requested_model or self._last_response_meta.get("response_route", {}).get("model")
                      or self.default_model)
        detail = str(error or "").strip()
        if self._looks_like_side_effect_request(user_input if isinstance(user_input, str) else ""):
            return (
                f"⚠️ Локальная модель `{model_name}` через `{provider_name}` не успела ответить вовремя. "
                "Выполнение задачи не подтверждено инструментами. "
                "Нужно продолжить более узким шагом или повторить запуск после упрощения контекста."
                + (f" Причина: {detail}." if detail else "")
            )
        return (
            f"⚠️ Модель `{model_name}` через `{provider_name}` не успела ответить вовремя."
            + (f" Причина: {detail}." if detail else "")
        )

    async def _run_impl(self, user_input: str | list, user_id: str = "default",
                        requested_model: str | None = None) -> str:
        """Run agent on user input. Accepts str or list of content blocks (multimodal)."""
        self._current_user_id = user_id
        self._last_response_meta = {}
        requested_model = (requested_model or "").strip() or None
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

        if isinstance(content_for_api, list):
            content_for_api = await self._apply_media_understanding(content_for_api, text_for_memory)
            # If pre-analysis produced summaries, strip raw media blocks so the main
            # text LLM is not forced into a vision-model path (e.g. llava on Ollama).
            # The injected text summaries are sufficient grounding for the main response.
            if self._last_response_meta.get("media_explainability"):
                content_for_api = [
                    b for b in content_for_api
                    if not (isinstance(b, dict) and b.get("type") in {"image", "document"})
                ]

        # Budget check
        if self.memory.get_today_cost() >= self.budget_daily:
            return f"⚠️ Daily budget (${self.budget_daily:.2f}) reached. Reset tomorrow."

        # Persist user message immediately (original language)
        self.memory.add_message(user_id, "user", text_for_memory)

        # Immediate profile updates from explicit user statements.
        direct_profile_update = self._direct_profile_update_ack(text_for_memory, user_id)
        if direct_profile_update:
            self.memory.add_message(user_id, "assistant", direct_profile_update)
            return direct_profile_update

        direct_history = self._direct_historical_request_answer(text_for_memory, user_id)
        if direct_history:
            self.memory.add_message(user_id, "assistant", direct_history)
            return direct_history

        direct_memory_summary = self._direct_personal_memory_summary(text_for_memory, user_id)
        if direct_memory_summary:
            self.memory.add_message(user_id, "assistant", direct_memory_summary)
            return direct_memory_summary

        if not self._is_internal_autonomous_prompt(text_for_memory):
            direct_telegram_guidance = self._direct_telegram_target_guidance(text_for_memory, user_id)
            if direct_telegram_guidance:
                self.memory.add_message(user_id, "assistant", direct_telegram_guidance)
                return direct_telegram_guidance

            direct_followup_delivery = self._direct_followup_telegram_delivery(text_for_memory, user_id)
            if direct_followup_delivery:
                self.memory.add_message(user_id, "assistant", direct_followup_delivery)
                return direct_followup_delivery

            direct_markdown_file = await self._direct_recent_markdown_file_delivery(text_for_memory, user_id)
            if direct_markdown_file:
                self.memory.add_message(user_id, "assistant", direct_markdown_file)
                return direct_markdown_file

            direct_owned_document = await self._direct_owned_document_delivery(text_for_memory, user_id)
            if direct_owned_document:
                self.memory.add_message(user_id, "assistant", direct_owned_document)
                return direct_owned_document

            direct_recent_file = await self._direct_recent_file_followup(text_for_memory, user_id)
            if direct_recent_file:
                self.memory.add_message(user_id, "assistant", direct_recent_file)
                return direct_recent_file

        # Deterministic profile-slot answers prevent occasional LLM misses on
        # questions like "как меня зовут".
        direct_profile = self._direct_profile_memory_answer(text_for_memory, user_id)
        if direct_profile:
            self.memory.add_message(user_id, "assistant", direct_profile)
            return direct_profile

        # Build context (token-efficient). Inline media requests should not be
        # hijacked by first-run onboarding copy; the user is already giving a task.
        _prev_skip_onboarding = getattr(self, "_skip_onboarding_for_request", False)
        self._skip_onboarding_for_request = self._content_has_media_blocks(content_for_api)
        try:
            system_prompt = self._build_system_prompt(text_for_memory, user_id)
        finally:
            self._skip_onboarding_for_request = _prev_skip_onboarding
        kb_context = await self._auto_retrieve_kb_context(text_for_memory)
        content_for_api = self._inject_kb_context(
            content_for_api,
            kb_context,
            compact=(self._slow_local_mode and self._slow_local_cfg.get("compact_kb_prompt", True)),
        )
        if kb_context:
            logger.debug("Injected KB context (%d chars) for user %s",
                         len(kb_context), user_id)
        messages = self.memory.get_compressed_history(user_id)
        messages.append({"role": "user", "content": content_for_api})

        # Select model (cascade routing — may switch provider for cross-provider cascade)
        complexity_score = self._complexity_score(text_for_memory)
        # Boost: short continuation messages in an ongoing conversation inherit medium complexity
        # so the agent doesn't downgrade to a lightweight model mid-task.
        if complexity_score == 0 and len(messages) > 4 and len(text_for_memory) < 80:
            complexity_score = 1
        route_choice = await self._select_model_for_request(
            text_for_memory,
            user_id=user_id,
            requested_model=requested_model,
            complexity_score=complexity_score,
        )
        model = route_choice.get("model", self.default_model)
        _cascade_tier = str(route_choice.get("tier") or "fixed")
        _resolved_requested = model  # bare model name after provider prefix stripping
        model = self._select_multimodal_response_model(model, content_for_api)
        if requested_model and model != _resolved_requested:
            _cascade_tier = "manual-multimodal"
        self._set_response_route_meta(
            provider=self.config.get("agent", {}).get("provider", "anthropic"),
            model=model,
            requested_model=requested_model,
            mode=_cascade_tier,
            details={
                "decision_source": route_choice.get("decision_source"),
                "objective": route_choice.get("objective"),
                "reason": route_choice.get("reason"),
                "gap": route_choice.get("gap"),
                "recommendation": route_choice.get("recommendation"),
                "advisor_model": route_choice.get("advisor_model"),
            },
        )
        _provider_switched = hasattr(self, '_original_provider')
        logger.info("Model: %s | User: %s | Input: %d chars | Complexity: %d | Tier: %s%s",
                     model, user_id, len(text_for_memory), complexity_score, _cascade_tier,
                     " [cross-provider]" if _provider_switched else "")
        LiteAgent._record_cascade_decision(
            model, _cascade_tier, complexity_score,
            decision_source=str(route_choice.get("decision_source") or ""),
            objective=str(route_choice.get("objective") or ""),
            gap=str(route_choice.get("gap") or ""),
        )

        # Tool selection: skip tools for trivial messages (greetings, short chat)
        # But always include tools from triggered skills (voice mode switching, etc.)
        _triggered_skills = self.skill_registry.get_triggered_skills(text_for_memory)
        _skill_tool_names = set()
        for _sk in _triggered_skills:
            _skill_tool_names.update(_sk.metadata.tools or [])

        # Skip tools only for truly trivial first messages (greetings, etc. with no history).
        # Never skip if tool-capability guard promoted the model (tools are needed).
        # Never skip in multi-turn conversations — short confirmations like "да", "хочу",
        # "подтверждаю" continue an ongoing task and must have tools available.
        _tool_guard_active = (self.cascade_routing and complexity_score < 1
                              and hasattr(self, "tools") and self.tools._tools
                              and self.models.get("simple") != self.models.get("medium"))
        _has_conv_history = len(messages) > 1  # history beyond the just-appended user turn
        if (complexity_score <= 0 and len(text_for_memory) < 60
                and not _skill_tool_names and not _tool_guard_active
                and not _has_conv_history):
            tool_defs = []
            logger.debug("Skipping tools for simple first message (no history)")
        elif self.memory._embedder and len(self.tools._tools) > 8:
            tool_defs = self.tools.get_relevant_definitions(
                text_for_memory,
                top_k=self._tool_relevance_top_k(),
                embedder=self.memory._embedder,
            )
        elif self._slow_local_mode and len(self.tools._tools) > self._tool_relevance_top_k():
            tool_defs = self.tools.get_keyword_relevant_definitions(
                text_for_memory,
                top_k=self._tool_relevance_top_k(),
            )
        else:
            tool_defs = self.tools.get_definitions()

        # Ensure triggered skill tools are always included in tool_defs
        if _skill_tool_names:
            existing_names = {td["name"] for td in tool_defs}
            for _stn in _skill_tool_names:
                if _stn not in existing_names and _stn in self.tools._tools:
                    tool_defs.append(self.tools._tools[_stn])
                    logger.debug("Added skill tool: %s", _stn)

        # Personal memory queries should not route to KB tools.
        tool_defs = self._prune_kb_tools_for_personal_query(tool_defs, text_for_memory)
        # Inline media: don't search file store when image/doc already in the message.
        tool_defs = self._prune_file_search_for_inline_media(tool_defs, content_for_api)
        # Autonomous tooling: use existing tools first, materialize missing capability tools when needed.
        tool_defs = self._ensure_tool_autonomy(text_for_memory, tool_defs)

        # Track tool calls for skill crystallization
        _tool_calls_log = []
        _tool_results_summary = []
        _progress_tracker = {"stall_count": 0, "last_signature": None}

        # Internal monologue: pre-planning
        _plan, tool_defs, model, _effective_max = await self._apply_planning(
            text_for_memory, user_id, system_prompt, tool_defs, model,
            complexity_score=complexity_score)
        if requested_model:
            model = self._select_multimodal_response_model(requested_model, content_for_api)
            if model != requested_model:
                _cascade_tier = "manual-multimodal"
            self._set_response_route_meta(
                provider=self.config.get("agent", {}).get("provider", "anthropic"),
                model=model,
                requested_model=requested_model,
                mode=_cascade_tier,
            )

        # Track in-flight request for dashboard
        _req_id = await self._track_request_start(
            user_id,
            text_for_memory[:120] if isinstance(text_for_memory, str) else "multimodal",
            model,
            complexity_score=complexity_score,
            cascade_tier=_cascade_tier)
        await self._update_request_progress(
            _req_id,
            phase="reasoning",
            phase_label="Context ready, starting model pass",
            iteration=0,
            max_iterations=_effective_max,
            progress_label=f"0/{_effective_max} cycles",
        )

        # Agent loop
        try:
            _forced_tool_continuations = 0
            _forced_failed_tool_repairs = 0
            _forced_no_tool_recoveries = 0
            _forced_autonomy_recoveries = 0
            _no_tool_response_passes = 0
            for iteration in range(_effective_max):
                _iteration_calls = []
                await self._update_request_progress(
                    _req_id,
                    phase="reasoning",
                    phase_label=f"Model pass {iteration + 1}/{_effective_max}",
                    iteration=iteration + 1,
                    max_iterations=_effective_max,
                    progress_label=f"Iteration {iteration + 1} of {_effective_max}",
                    parallel_total=_REQUEST_TRACKING_CLEAR,
                    parallel_completed=_REQUEST_TRACKING_CLEAR,
                    parallel_children=_REQUEST_TRACKING_CLEAR,
                )
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
                    _no_tool_response_passes = 0
                    # Stuck loop detection: same tool+args 5 times in a row
                    _current_calls = [
                        {"name": b.name, "input": b.input}
                        for b in response.content
                        if hasattr(b, 'type') and b.type == "tool_use"
                    ]
                    if (len(_tool_calls_log) >= 4 and _current_calls
                            and all(
                                c["name"] == _tool_calls_log[-1].get("name")
                                and c["input"] == _tool_calls_log[-1].get("input")
                                for c in _current_calls
                            )
                            and _tool_calls_log[-1] == _tool_calls_log[-2]
                            and _tool_calls_log[-2] == _tool_calls_log[-3]
                            and _tool_calls_log[-3] == _tool_calls_log[-4]):
                        logger.warning("Stuck loop detected: %s called 3x with same args",
                                       _current_calls[0]["name"])
                        messages.append({"role": "assistant", "content": _serialize_content(response.content)})
                        messages.append({"role": "user", "content": [{
                            "type": "tool_result", "tool_use_id": response.content[0].id if hasattr(response.content[0], 'id') else "stuck",
                            "content": ("[System] You've called the same tool with identical "
                                        "arguments 3 times. The result won't change. "
                                        "Try a different approach or provide your answer."),
                        }]})
                        continue

                    # Execute tools in parallel
                    _tool_blocks = [
                        b for b in response.content
                        if hasattr(b, 'type') and b.type == "tool_use"
                    ]
                    _tool_children = [{
                        "tool_use_id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "status": "pending",
                        "duration_ms": 0,
                        "error": False,
                    } for block in _tool_blocks]
                    await self._update_request_progress(
                        _req_id,
                        phase="parallel_tools",
                        phase_label=f"Executing {len(_tool_children)} tool(s) in parallel",
                        progress_label=(
                            f"Iteration {iteration + 1}/{_effective_max} · "
                            f"tools 0/{len(_tool_children)}"
                        ),
                        parallel_total=len(_tool_children),
                        parallel_completed=0,
                        parallel_children=_tool_children,
                    )

                    async def _on_tool_progress(event: dict):
                        await self._update_request_tool_progress(_req_id, event)

                    tool_results = await self.tools.execute_parallel(
                        response.content,
                        on_progress=_on_tool_progress,
                    )
                    # Strip _meta before sending to LLM
                    clean_results = [{k: v for k, v in r.items() if k != "_meta"}
                                     for r in tool_results]
                    result_meta_by_id = {
                        r.get("tool_use_id"): r.get("_meta", {})
                        for r in tool_results
                        if isinstance(r, dict)
                    }
                    messages.append({"role": "assistant", "content": _serialize_content(response.content)})
                    messages.append({"role": "user", "content": clean_results})
                    # Log tool calls for skill crystallization + file access tracking
                    for block in response.content:
                        if hasattr(block, 'type') and block.type == "tool_use":
                            _meta = result_meta_by_id.get(getattr(block, "id", ""), {})
                            _call_info = {
                                "name": block.name,
                                "input": block.input,
                                "error": bool(_meta.get("error")),
                                "result_preview": str(_meta.get("result_preview", ""))[:220],
                                "duration_ms": int(_meta.get("duration_ms") or 0),
                            }
                            _tool_calls_log.append(_call_info)
                            _iteration_calls.append(_call_info)
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
                    for tr in clean_results:
                        content = tr.get("content", "") if isinstance(tr, dict) else str(tr)
                        _tool_results_summary.append(str(content)[:200])
                    # Mid-loop reflection (internal monologue) — merges into last message
                    await self._update_request_progress(
                        _req_id,
                        phase="reflection",
                        phase_label="Synthesizing tool outputs",
                        progress_label=f"Iteration {iteration + 1}/{_effective_max} · reflection",
                    )
                    await self._apply_reflection(
                        messages, _plan, _tool_calls_log, _tool_results_summary)
                    _progress_tracker = self._advance_progress_tracker(
                        _progress_tracker, _iteration_calls)
                    if self._should_stop_for_no_progress(_progress_tracker, _effective_max):
                        logger.warning(
                            "Stopping run loop for no progress after %d stalled tool iterations",
                            _progress_tracker.get("stall_count", 0),
                        )
                        return ("⚠️ Останавливаю цикл: несколько итераций подряд не дали "
                                "нового результата. Уточните задачу или разбейте её на шаги.")
                    if self._should_force_failed_tool_repair(
                        _progress_tracker,
                        text_for_memory,
                        _forced_failed_tool_repairs,
                        _iteration_calls,
                        _effective_max,
                    ):
                        logger.info(
                            "Forcing failed-tool repair continuation after %d failure-only tool rounds",
                            _progress_tracker.get("failure_only_count", 0),
                        )
                        repair_prompt = await self._build_failed_tool_repair_prompt_with_health(
                            user_id,
                            _iteration_calls,
                        )
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": repair_prompt,
                            }],
                        })
                        _forced_failed_tool_repairs += 1
                        continue
                else:
                    # Done — extract text
                    text = self._extract_text(response)
                    text = self._sanitize_memory_limit_response(text, text_for_memory, user_id)
                    text = self._sanitize_unverified_completion_response(text, text_for_memory, _tool_calls_log)

                    # ── Fallback: parse tool call(s) from plain text ──
                    # Some models (e.g. Ollama/qwen) output tool calls as text JSON
                    # instead of structured tool_use. Detect and execute them.
                    parsed_tools = self._try_parse_text_tool_calls(text, tool_defs)
                    if parsed_tools:
                        _no_tool_response_passes = 0
                        blocks = [
                            ToolUseBlock(
                                id=f"fallback_{iteration}_{i}",
                                name=pt["name"],
                                input=pt["arguments"],
                            )
                            for i, pt in enumerate(parsed_tools)
                        ]
                        for b in blocks:
                            logger.info("Fallback tool call parsed from text: %s", b.name)
                        _tool_children = [{
                            "tool_use_id": getattr(b, "id", ""),
                            "name": getattr(b, "name", ""),
                            "status": "pending",
                            "duration_ms": 0,
                            "error": False,
                        } for b in blocks]
                        await self._update_request_progress(
                            _req_id,
                            phase="parallel_tools",
                            phase_label=f"Executing fallback tools ({len(_tool_children)})",
                            progress_label=(
                                f"Iteration {iteration + 1}/{_effective_max} · "
                                f"fallback 0/{len(_tool_children)}"
                            ),
                            parallel_total=len(_tool_children),
                            parallel_completed=0,
                            parallel_children=_tool_children,
                        )

                        async def _on_fallback_tool_progress(event: dict):
                            await self._update_request_tool_progress(_req_id, event)

                        tool_results = await self.tools.execute_parallel(
                            blocks,
                            on_progress=_on_fallback_tool_progress,
                        )
                        clean_results = [{k: v for k, v in r.items() if k != "_meta"}
                                         for r in tool_results]
                        result_meta_by_id = {
                            r.get("tool_use_id"): r.get("_meta", {})
                            for r in tool_results
                            if isinstance(r, dict)
                        }
                        messages.append({"role": "assistant", "content": _serialize_content(blocks)})
                        messages.append({"role": "user", "content": clean_results})
                        for b in blocks:
                            _meta = result_meta_by_id.get(getattr(b, "id", ""), {})
                            _call_info = {
                                "name": b.name, "input": b.input,
                                "error": bool(_meta.get("error")),
                                "result_preview": str(_meta.get("result_preview", ""))[:220],
                                "duration_ms": int(_meta.get("duration_ms") or 0),
                            }
                            _tool_calls_log.append(_call_info)
                            _iteration_calls.append(_call_info)
                        _progress_tracker = self._advance_progress_tracker(
                            _progress_tracker, _iteration_calls)
                        if self._should_stop_for_no_progress(_progress_tracker, _effective_max):
                            logger.warning(
                                "Stopping run loop for no progress after %d stalled fallback iterations",
                                _progress_tracker.get("stall_count", 0),
                            )
                            return ("⚠️ Останавливаю цикл: несколько итераций подряд не дали "
                                    "нового результата. Уточните задачу или разбейте её на шаги.")
                        if self._should_force_failed_tool_repair(
                            _progress_tracker,
                            text_for_memory,
                            _forced_failed_tool_repairs,
                            _iteration_calls,
                            _effective_max,
                        ):
                            logger.info(
                                "Forcing failed-tool repair continuation after %d failure-only fallback rounds",
                                _progress_tracker.get("failure_only_count", 0),
                            )
                            repair_prompt = await self._build_failed_tool_repair_prompt_with_health(
                                user_id,
                                _iteration_calls,
                            )
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "text",
                                    "text": repair_prompt,
                                }],
                            })
                            _forced_failed_tool_repairs += 1
                            continue
                        continue

                    if self._should_force_tool_continuation(
                        text, text_for_memory, _tool_calls_log, _forced_tool_continuations
                    ):
                        logger.info(
                            "Forcing tool continuation after status-only guard response on side-effect task"
                        )
                        messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": self._forced_tool_continuation_prompt()}],
                        })
                        _forced_tool_continuations += 1
                        continue

                    if self._should_force_autonomy_recovery(
                        text_for_memory,
                        text,
                        _tool_calls_log,
                        _forced_autonomy_recoveries,
                    ):
                        logger.info(
                            "Forcing autonomy recovery after permission-seeking reply on side-effect task"
                        )
                        recovery_prompt = await self._build_autonomy_recovery_prompt_with_health(
                            user_id,
                            _tool_calls_log,
                        )
                        messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": recovery_prompt}],
                        })
                        _forced_autonomy_recoveries += 1
                        continue

                    _no_tool_response_passes += 1
                    if self._should_force_no_tool_recovery(
                        text_for_memory,
                        _tool_calls_log,
                        _forced_no_tool_recoveries,
                        _no_tool_response_passes,
                    ):
                        logger.info(
                            "Forcing tool-first recovery after %d no-tool assistant pass(es) on side-effect task",
                            _no_tool_response_passes,
                        )
                        recovery_prompt = await self._build_no_tool_recovery_prompt_with_health(
                            user_id,
                            _tool_calls_log,
                        )
                        messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": recovery_prompt}],
                        })
                        _forced_no_tool_recoveries += 1
                        continue

                    # ── Feature hooks (post-response via hook system) ──
                    await self._update_request_progress(
                        _req_id,
                        phase="finalizing",
                        phase_label="Composing final response",
                        progress_label="Final response",
                        parallel_total=_REQUEST_TRACKING_CLEAR,
                        parallel_completed=_REQUEST_TRACKING_CLEAR,
                        parallel_children=_REQUEST_TRACKING_CLEAR,
                    )
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

                    text = self._sanitize_unverified_completion_response(text, text_for_memory, _tool_calls_log)

                    # Garbage detection: if the model produced incoherent output,
                    # try to get a clean response from a fallback provider.
                    if self._is_garbage_response(text):
                        logger.warning("Garbage response detected from model '%s', attempting fallback", model)
                        fallback = self._get_fallback_provider()
                        if fallback:
                            fb_name, fb_model = fallback
                            try:
                                import copy as _copy
                                _temp_cfg = _copy.deepcopy(self.config)
                                _temp_cfg.setdefault("agent", {})["provider"] = fb_name
                                from .providers import create_provider as _create_provider
                                _fb_provider = _create_provider(_temp_cfg)
                                _fb_resp = await _fb_provider.complete(
                                    model=fb_model,
                                    max_tokens=4096,
                                    system=system_prompt,
                                    messages=messages,
                                )
                                _fb_text = self._extract_text(_fb_resp).strip()
                                if _fb_text and not self._is_garbage_response(_fb_text):
                                    text = _fb_text
                                    logger.info("Garbage fallback succeeded via %s/%s", fb_name, fb_model)
                                    try:
                                        if hasattr(_fb_resp, "usage") and _fb_resp.usage:
                                            _fb_cost = self._calculate_cost(fb_model, _fb_resp.usage)
                                            self.memory.track_usage(user_id, fb_model, _fb_resp.usage, _fb_cost)
                                    except Exception:
                                        pass
                            except Exception as _fb_err:
                                logger.debug("Garbage fallback failed: %s", _fb_err)
                        if self._is_garbage_response(text):
                            text = (
                                "Извините, модель вернула некорректный ответ. "
                                "Попробуйте повторить запрос или выберите другую модель в настройках."
                            )

                    # Strip tool-narration artifacts before storing/returning
                    text = self._clean_response_artifacts(text)
                    await self._update_request_progress(
                        _req_id,
                        phase="critical_review",
                        phase_label="Running internal critical review",
                        progress_label="Critical review",
                        parallel_total=_REQUEST_TRACKING_CLEAR,
                        parallel_completed=_REQUEST_TRACKING_CLEAR,
                        parallel_children=_REQUEST_TRACKING_CLEAR,
                    )
                    text = await self._critical_review_response_if_needed(
                        user_input=text_for_memory,
                        response_text=text,
                        user_id=user_id,
                        tool_calls_log=_tool_calls_log,
                        model=model,
                        original_input=user_input,
                    )
                    text = self._clean_response_artifacts(text)
                    text = self._sanitize_unverified_completion_response(text, text_for_memory, _tool_calls_log)

                    # Save assistant response (user message already persisted above)
                    payload = text
                    if self._last_response_meta:
                        payload = {"text": text, "meta": dict(self._last_response_meta)}
                    self.memory.add_message(user_id, "assistant", payload)

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

    async def stream(self, user_input: str, user_id: str = "default",
                     requested_model: str | None = None) -> AsyncGenerator[str, None]:
        """Stream agent response with per-user serialization."""
        self._last_response_meta = {}
        user_id = self.resolve_user_id(user_id)
        LiteAgent._ensure_locks()
        lock = await self._get_user_lock(user_id)
        q_id = self._track_queued(user_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self._lock_timeout)
        except asyncio.TimeoutError:
            logger.warning("Stream timeout: user %s waited >%.0fs for lock",
                           user_id, self._lock_timeout)
            yield "⏳ Request queued too long. Please try again in a moment."
            self._untrack_queued(q_id)
            return
        self._untrack_queued(q_id)
        try:
            async for chunk in self._stream_impl(user_input, user_id, requested_model=requested_model):
                yield chunk
        finally:
            lock.release()

    async def _stream_impl(self, user_input: str, user_id: str = "default",
                           requested_model: str | None = None) -> AsyncGenerator[str, None]:
        """Stream agent response token by token."""
        self._current_user_id = user_id
        self._last_response_meta = {}
        requested_model = (requested_model or "").strip() or None
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

        direct_profile_update = self._direct_profile_update_ack(user_input, user_id)
        if direct_profile_update:
            self.memory.add_message(user_id, "assistant", direct_profile_update)
            yield direct_profile_update
            return

        direct_history = self._direct_historical_request_answer(user_input, user_id)
        if direct_history:
            self.memory.add_message(user_id, "assistant", direct_history)
            yield direct_history
            return

        direct_memory_summary = self._direct_personal_memory_summary(user_input, user_id)
        if direct_memory_summary:
            self.memory.add_message(user_id, "assistant", direct_memory_summary)
            yield direct_memory_summary
            return

        if not self._is_internal_autonomous_prompt(user_input):
            direct_telegram_guidance = self._direct_telegram_target_guidance(user_input, user_id)
            if direct_telegram_guidance:
                self.memory.add_message(user_id, "assistant", direct_telegram_guidance)
                yield direct_telegram_guidance
                return

            direct_followup_delivery = self._direct_followup_telegram_delivery(user_input, user_id)
            if direct_followup_delivery:
                self.memory.add_message(user_id, "assistant", direct_followup_delivery)
                yield direct_followup_delivery
                return

            direct_markdown_file = await self._direct_recent_markdown_file_delivery(user_input, user_id)
            if direct_markdown_file:
                self.memory.add_message(user_id, "assistant", direct_markdown_file)
                yield direct_markdown_file
                return

            direct_owned_document = await self._direct_owned_document_delivery(user_input, user_id)
            if direct_owned_document:
                self.memory.add_message(user_id, "assistant", direct_owned_document)
                yield direct_owned_document
                return

            direct_recent_file = await self._direct_recent_file_followup(user_input, user_id)
            if direct_recent_file:
                self.memory.add_message(user_id, "assistant", direct_recent_file)
                yield direct_recent_file
                return

        direct_profile = self._direct_profile_memory_answer(user_input, user_id)
        if direct_profile:
            self.memory.add_message(user_id, "assistant", direct_profile)
            yield direct_profile
            return

        system_prompt = self._build_system_prompt(user_input, user_id)
        kb_context = await self._auto_retrieve_kb_context(user_input)
        content_for_api = self._inject_kb_context(
            user_input,
            kb_context,
            compact=(self._slow_local_mode and self._slow_local_cfg.get("compact_kb_prompt", True)),
        )
        if kb_context:
            logger.debug("Injected KB context (%d chars) for stream user %s",
                         len(kb_context), user_id)
        messages = self.memory.get_compressed_history(user_id)
        messages.append({"role": "user", "content": content_for_api})
        complexity_score = self._complexity_score(user_input)
        # Boost: short continuation in ongoing conversation → at least medium model
        if complexity_score == 0 and len(messages) > 4 and len(user_input) < 80:
            complexity_score = 1
        route_choice = await self._select_model_for_request(
            user_input,
            user_id=user_id,
            requested_model=requested_model,
            complexity_score=complexity_score,
        )
        model = route_choice.get("model", self.default_model)
        _cascade_tier = str(route_choice.get("tier") or "fixed")
        self._set_response_route_meta(
            provider=self.config.get("agent", {}).get("provider", "anthropic"),
            model=model,
            requested_model=requested_model,
            mode=_cascade_tier,
            details={
                "decision_source": route_choice.get("decision_source"),
                "objective": route_choice.get("objective"),
                "reason": route_choice.get("reason"),
                "gap": route_choice.get("gap"),
                "recommendation": route_choice.get("recommendation"),
                "advisor_model": route_choice.get("advisor_model"),
            },
        )
        _provider_switched = hasattr(self, '_original_provider')
        logger.info("Stream | Model: %s | User: %s | Complexity: %d | Tier: %s%s",
                     model, user_id, complexity_score, _cascade_tier,
                     " [cross-provider]" if _provider_switched else "")
        LiteAgent._record_cascade_decision(
            model, _cascade_tier, complexity_score,
            decision_source=str(route_choice.get("decision_source") or ""),
            objective=str(route_choice.get("objective") or ""),
            gap=str(route_choice.get("gap") or ""),
        )

        # Skip tools for trivial messages, but always include triggered skill tools
        _triggered_skills = self.skill_registry.get_triggered_skills(user_input)
        _skill_tool_names = set()
        for _sk in _triggered_skills:
            _skill_tool_names.update(_sk.metadata.tools or [])

        _has_conv_history = len(messages) > 1  # history beyond just-appended user turn
        if (complexity_score <= 0 and len(user_input) < 60
                and not _skill_tool_names and not _has_conv_history):
            tool_defs = []
        elif self.memory._embedder and len(self.tools._tools) > 8:
            tool_defs = self.tools.get_relevant_definitions(
                user_input,
                top_k=self._tool_relevance_top_k(),
                embedder=self.memory._embedder,
            )
        elif self._slow_local_mode and len(self.tools._tools) > self._tool_relevance_top_k():
            tool_defs = self.tools.get_keyword_relevant_definitions(
                user_input,
                top_k=self._tool_relevance_top_k(),
            )
        else:
            tool_defs = self.tools.get_definitions()

        # Ensure triggered skill tools are always included
        if _skill_tool_names:
            existing_names = {td["name"] for td in tool_defs}
            for _stn in _skill_tool_names:
                if _stn not in existing_names and _stn in self.tools._tools:
                    tool_defs.append(self.tools._tools[_stn])

        # Personal memory queries should not route to KB tools.
        tool_defs = self._prune_kb_tools_for_personal_query(tool_defs, user_input)
        # Inline media: don't search file store when image/doc already in the message.
        tool_defs = self._prune_file_search_for_inline_media(tool_defs, content_for_api)
        # Autonomous tooling: use existing tools first, materialize missing capability tools when needed.
        tool_defs = self._ensure_tool_autonomy(user_input, tool_defs)

        # Internal monologue: pre-planning (stream)
        _tool_calls_log = []
        _tool_results_summary = []
        _progress_tracker = {"stall_count": 0, "last_signature": None}
        _plan, tool_defs, model, _effective_max = await self._apply_planning(
            user_input, user_id, system_prompt, tool_defs, model,
            complexity_score=complexity_score)
        if requested_model:
            # Restore manually-requested model after planning (use bare name, provider already switched)
            model = self._resolve_requested_model(requested_model)
            self._set_response_route_meta(
                provider=self.config.get("agent", {}).get("provider", "anthropic"),
                model=model,
                requested_model=requested_model,
                mode=_cascade_tier,
            )

        # Track in-flight request for dashboard
        _req_id = await self._track_request_start(
            user_id, user_input[:120], model,
            complexity_score=complexity_score,
            cascade_tier=_cascade_tier)
        await self._update_request_progress(
            _req_id,
            phase="reasoning",
            phase_label="Context ready, starting stream",
            iteration=0,
            max_iterations=_effective_max,
            progress_label=f"0/{_effective_max} cycles",
        )

        full_text = ""
        _forced_tool_continuations = 0
        _forced_failed_tool_repairs = 0
        _forced_no_tool_recoveries = 0
        _forced_autonomy_recoveries = 0
        _no_tool_response_passes = 0

        # Pre-strip tools for vision-only models in stream path too
        if self._is_vision_only_model(model) and tool_defs:
            logger.debug("Vision-only model '%s': stripping tools upfront (stream)", model)
            tool_defs = []

        try:
            for iteration in range(_effective_max):
                _iteration_calls = []
                await self._update_request_progress(
                    _req_id,
                    phase="reasoning",
                    phase_label=f"Streaming pass {iteration + 1}/{_effective_max}",
                    iteration=iteration + 1,
                    max_iterations=_effective_max,
                    progress_label=f"Iteration {iteration + 1} of {_effective_max}",
                    parallel_total=_REQUEST_TRACKING_CLEAR,
                    parallel_completed=_REQUEST_TRACKING_CLEAR,
                    parallel_children=_REQUEST_TRACKING_CLEAR,
                )
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
                    elif self._is_no_tools_error(e) and tool_defs:
                        # Model doesn't support tools — retry without tools
                        logger.warning("Model '%s' does not support tools, retrying without tools", model)
                        tool_defs = None
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
                self._set_response_route_meta(
                    provider=self.config.get("agent", {}).get("provider", "anthropic"),
                    model=model,
                )

                cost = self._calculate_cost(model, response.usage)
                self.memory.track_usage(user_id, model, response.usage, cost)

                if response.stop_reason == "tool_use":
                    _no_tool_response_passes = 0
                    tool_blocks = [b for b in response.content if b.type == "tool_use"]

                    # Stuck loop detection (stream)
                    _current_calls = [{"name": b.name, "input": b.input} for b in tool_blocks]
                    if (len(_tool_calls_log) >= 4 and _current_calls
                            and all(
                                c["name"] == _tool_calls_log[-1].get("name")
                                and c["input"] == _tool_calls_log[-1].get("input")
                                for c in _current_calls
                            )
                            and _tool_calls_log[-1] == _tool_calls_log[-2]
                            and _tool_calls_log[-2] == _tool_calls_log[-3]
                            and _tool_calls_log[-3] == _tool_calls_log[-4]):
                        logger.warning("Stuck loop detected (stream): %s", _current_calls[0]["name"])
                        yield "\n⚠️ Stuck loop detected, trying different approach...\n"
                        messages.append({"role": "assistant", "content": _serialize_content(response.content)})
                        messages.append({"role": "user", "content": [{
                            "type": "tool_result", "tool_use_id": tool_blocks[0].id,
                            "content": ("[System] You've called the same tool with identical "
                                        "arguments 3 times. Try a different approach or answer directly."),
                        }]})
                        full_text = ""
                        continue

                    # Signal tool starts
                    for block in tool_blocks:
                        yield f"\n__TOOL_START__{json.dumps({'name': block.name, 'input': block.input, 'id': block.id}, default=str)}__TOOL_END__\n"

                    # Execute all tools in parallel
                    _tool_children = [{
                        "tool_use_id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "status": "pending",
                        "duration_ms": 0,
                        "error": False,
                    } for block in tool_blocks]
                    await self._update_request_progress(
                        _req_id,
                        phase="parallel_tools",
                        phase_label=f"Executing {len(_tool_children)} tool(s) in parallel",
                        progress_label=(
                            f"Iteration {iteration + 1}/{_effective_max} · "
                            f"tools 0/{len(_tool_children)}"
                        ),
                        parallel_total=len(_tool_children),
                        parallel_completed=0,
                        parallel_children=_tool_children,
                    )

                    async def _on_stream_tool_progress(event: dict):
                        await self._update_request_tool_progress(_req_id, event)

                    tool_results = await self.tools.execute_parallel(
                        tool_blocks,
                        on_progress=_on_stream_tool_progress,
                    )

                    # Signal tool results
                    for result in tool_results:
                        meta = result.get("_meta", {})
                        yield f"\n__TOOL_RESULT__{json.dumps({'name': meta.get('tool_name', ''), 'id': meta.get('tool_input', {}).get('id', ''), 'duration_ms': meta.get('duration_ms', 0), 'error': meta.get('error', False), 'preview': meta.get('result_preview', '')[:300]}, default=str)}__TOOL_END__\n"

                    # Strip _meta before sending to LLM
                    clean_results = [{k: v for k, v in r.items() if k != "_meta"} for r in tool_results]
                    result_meta_by_id = {
                        r.get("tool_use_id"): r.get("_meta", {})
                        for r in tool_results
                        if isinstance(r, dict)
                    }
                    messages.append({"role": "assistant", "content": _serialize_content(response.content)})
                    messages.append({"role": "user", "content": clean_results})
                    # Track for planning reflection (stream)
                    for block in tool_blocks:
                        _meta = result_meta_by_id.get(getattr(block, "id", ""), {})
                        _call_info = {
                            "name": block.name,
                            "input": block.input,
                            "error": bool(_meta.get("error")),
                            "result_preview": str(_meta.get("result_preview", ""))[:220],
                            "duration_ms": int(_meta.get("duration_ms") or 0),
                        }
                        _tool_calls_log.append(_call_info)
                        _iteration_calls.append(_call_info)
                    for r in clean_results:
                        content = r.get("content", "") if isinstance(r, dict) else str(r)
                        _tool_results_summary.append(str(content)[:200])
                    # Mid-loop reflection (stream)
                    await self._update_request_progress(
                        _req_id,
                        phase="reflection",
                        phase_label="Synthesizing tool outputs",
                        progress_label=f"Iteration {iteration + 1}/{_effective_max} · reflection",
                    )
                    await self._apply_reflection(
                        messages, _plan, _tool_calls_log, _tool_results_summary)
                    _progress_tracker = self._advance_progress_tracker(
                        _progress_tracker, _iteration_calls)
                    if self._should_stop_for_no_progress(_progress_tracker, _effective_max):
                        logger.warning(
                            "Stopping stream loop for no progress after %d stalled tool iterations",
                            _progress_tracker.get("stall_count", 0),
                        )
                        yield ("\n⚠️ Останавливаю цикл: несколько итераций подряд не дали "
                               "нового результата. Уточните задачу или разбейте её на шаги.\n")
                        return
                    if self._should_force_failed_tool_repair(
                        _progress_tracker,
                        user_input,
                        _forced_failed_tool_repairs,
                        _iteration_calls,
                        _effective_max,
                    ):
                        logger.info(
                            "Forcing failed-tool repair continuation in stream after %d failure-only tool rounds",
                            _progress_tracker.get("failure_only_count", 0),
                        )
                        repair_prompt = await self._build_failed_tool_repair_prompt_with_health(
                            user_id,
                            _iteration_calls,
                        )
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "text",
                                "text": repair_prompt,
                            }],
                        })
                        _forced_failed_tool_repairs += 1
                        full_text = ""
                        yield ("\n[system] Switching to targeted bugfix mode after repeated tool failures.\n")
                        continue
                    full_text = ""  # Reset for next iteration
                else:
                    # ── Fallback: parse tool call(s) from plain text (stream) ──
                    parsed_tools = self._try_parse_text_tool_calls(full_text, tool_defs)
                    if parsed_tools:
                        _no_tool_response_passes = 0
                        blocks = [
                            ToolUseBlock(
                                id=f"fallback_s{iteration}_{i}",
                                name=pt["name"],
                                input=pt["arguments"],
                            )
                            for i, pt in enumerate(parsed_tools)
                        ]
                        for b in blocks:
                            logger.info("Fallback tool call (stream) parsed: %s", b.name)
                            yield f"\n__TOOL_START__{json.dumps({'name': b.name, 'input': b.input, 'id': b.id}, default=str)}__TOOL_END__\n"
                        _tool_children = [{
                            "tool_use_id": getattr(b, "id", ""),
                            "name": getattr(b, "name", ""),
                            "status": "pending",
                            "duration_ms": 0,
                            "error": False,
                        } for b in blocks]
                        await self._update_request_progress(
                            _req_id,
                            phase="parallel_tools",
                            phase_label=f"Executing fallback tools ({len(_tool_children)})",
                            progress_label=(
                                f"Iteration {iteration + 1}/{_effective_max} · "
                                f"fallback 0/{len(_tool_children)}"
                            ),
                            parallel_total=len(_tool_children),
                            parallel_completed=0,
                            parallel_children=_tool_children,
                        )

                        async def _on_stream_fallback_tool_progress(event: dict):
                            await self._update_request_tool_progress(_req_id, event)

                        tool_results = await self.tools.execute_parallel(
                            blocks,
                            on_progress=_on_stream_fallback_tool_progress,
                        )
                        for result in tool_results:
                            meta = result.get("_meta", {})
                            yield f"\n__TOOL_RESULT__{json.dumps({'name': meta.get('tool_name', ''), 'id': meta.get('tool_input', {}).get('id', ''), 'duration_ms': meta.get('duration_ms', 0), 'error': meta.get('error', False), 'preview': meta.get('result_preview', '')[:300]}, default=str)}__TOOL_END__\n"
                        clean_results = [{k: v for k, v in r.items() if k != "_meta"} for r in tool_results]
                        result_meta_by_id = {
                            r.get("tool_use_id"): r.get("_meta", {})
                            for r in tool_results
                            if isinstance(r, dict)
                        }
                        messages.append({"role": "assistant", "content": _serialize_content(blocks)})
                        messages.append({"role": "user", "content": clean_results})
                        for b in blocks:
                            _meta = result_meta_by_id.get(getattr(b, "id", ""), {})
                            _call_info = {
                                "name": b.name,
                                "input": b.input,
                                "error": bool(_meta.get("error")),
                                "result_preview": str(_meta.get("result_preview", ""))[:220],
                                "duration_ms": int(_meta.get("duration_ms") or 0),
                            }
                            _tool_calls_log.append(_call_info)
                            _iteration_calls.append(_call_info)
                        _progress_tracker = self._advance_progress_tracker(
                            _progress_tracker, _iteration_calls)
                        if self._should_stop_for_no_progress(_progress_tracker, _effective_max):
                            logger.warning(
                                "Stopping stream loop for no progress after %d stalled fallback iterations",
                                _progress_tracker.get("stall_count", 0),
                            )
                            yield ("\n⚠️ Останавливаю цикл: несколько итераций подряд не дали "
                                   "нового результата. Уточните задачу или разбейте её на шаги.\n")
                            return
                        if self._should_force_failed_tool_repair(
                            _progress_tracker,
                            user_input,
                            _forced_failed_tool_repairs,
                            _iteration_calls,
                            _effective_max,
                        ):
                            logger.info(
                                "Forcing failed-tool repair continuation in stream after %d failure-only fallback rounds",
                                _progress_tracker.get("failure_only_count", 0),
                            )
                            repair_prompt = await self._build_failed_tool_repair_prompt_with_health(
                                user_id,
                                _iteration_calls,
                            )
                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "text",
                                    "text": repair_prompt,
                                }],
                            })
                            _forced_failed_tool_repairs += 1
                            full_text = ""
                            yield ("\n[system] Switching to targeted bugfix mode after repeated tool failures.\n")
                            continue
                        full_text = ""
                        continue

                    # User message already persisted at stream start
                    await self._update_request_progress(
                        _req_id,
                        phase="finalizing",
                        phase_label="Composing final response",
                        progress_label="Final response",
                        parallel_total=_REQUEST_TRACKING_CLEAR,
                        parallel_completed=_REQUEST_TRACKING_CLEAR,
                        parallel_children=_REQUEST_TRACKING_CLEAR,
                    )
                    full_text = self._sanitize_memory_limit_response(full_text, user_input, user_id)
                    full_text = self._sanitize_unverified_completion_response(
                        full_text, user_input, _tool_calls_log
                    )
                    if self._should_force_tool_continuation(
                        full_text, user_input, _tool_calls_log, _forced_tool_continuations
                    ):
                        logger.info(
                            "Forcing stream tool continuation after status-only guard response on side-effect task"
                        )
                        messages.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": full_text}],
                        })
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": self._forced_tool_continuation_prompt()}],
                        })
                        _forced_tool_continuations += 1
                        full_text = ""
                        yield "\n[system] Continuing automatically because no tool execution was confirmed yet.\n"
                        continue
                    if self._should_force_autonomy_recovery(
                        user_input,
                        full_text,
                        _tool_calls_log,
                        _forced_autonomy_recoveries,
                    ):
                        logger.info(
                            "Forcing stream autonomy recovery after permission-seeking reply"
                        )
                        recovery_prompt = await self._build_autonomy_recovery_prompt_with_health(
                            user_id,
                            _tool_calls_log,
                        )
                        messages.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": full_text}],
                        })
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": recovery_prompt}],
                        })
                        _forced_autonomy_recoveries += 1
                        full_text = ""
                        yield "\n[system] Continuing automatically after internal autonomy review.\n"
                        continue
                    _no_tool_response_passes += 1
                    if self._should_force_no_tool_recovery(
                        user_input,
                        _tool_calls_log,
                        _forced_no_tool_recoveries,
                        _no_tool_response_passes,
                    ):
                        logger.info(
                            "Forcing stream tool-first recovery after %d no-tool assistant pass(es)",
                            _no_tool_response_passes,
                        )
                        recovery_prompt = await self._build_no_tool_recovery_prompt_with_health(
                            user_id,
                            _tool_calls_log,
                        )
                        messages.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": full_text}],
                        })
                        messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": recovery_prompt}],
                        })
                        _forced_no_tool_recoveries += 1
                        full_text = ""
                        yield "\n[system] Continuing automatically because the task still has no real tool execution.\n"
                        continue
                    full_text = self._clean_response_artifacts(full_text)
                    await self._update_request_progress(
                        _req_id,
                        phase="critical_review",
                        phase_label="Running internal critical review",
                        progress_label="Critical review",
                        parallel_total=_REQUEST_TRACKING_CLEAR,
                        parallel_completed=_REQUEST_TRACKING_CLEAR,
                        parallel_children=_REQUEST_TRACKING_CLEAR,
                    )
                    full_text = await self._critical_review_response_if_needed(
                        user_input=user_input,
                        response_text=full_text,
                        user_id=user_id,
                        tool_calls_log=_tool_calls_log,
                        model=model,
                        original_input=user_input,
                    )
                    full_text = self._clean_response_artifacts(full_text)
                    full_text = self._sanitize_unverified_completion_response(
                        full_text, user_input, _tool_calls_log
                    )
                    payload = full_text
                    if self._last_response_meta:
                        payload = {"text": full_text, "meta": dict(self._last_response_meta)}
                    self.memory.add_message(user_id, "assistant", payload)

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
    # POST-RESPONSE HOOKS (shared by run + stream)
    # ══════════════════════════════════════════

    async def _post_response_hooks(
        self,
        response_text: str,
        user_input: str,
        user_id: str,
        model: str,
        system_prompt,
        tool_defs: list,
        messages: list,
        tool_calls_log: list,
    ) -> None:
        """Fire after_response hooks. Called by both _run_impl and _stream_impl."""
        hook_ctx = HookContext(
            agent=self,
            user_id=user_id,
            user_input=user_input,
            model=model,
            system_prompt=system_prompt,
            tool_defs=tool_defs,
            messages=messages,
            response_text=response_text,
            tool_calls_log=tool_calls_log,
            extra={"user_input_text": user_input if isinstance(user_input, str) else ""},
        )
        await self.hooks.emit("after_response", hook_ctx)

    # ══════════════════════════════════════════
    # SELF-HEALING: PROVIDER FALLBACK
    # ══════════════════════════════════════════

    _FATAL_ERRORS = ("authentication", "auth", "401", "permission", "forbidden", "403")
    _SWITCHABLE_ERRORS = ("rate", "limit", "429", "quota", "overloaded", "503", "capacity")
    _MODEL_ERRORS = ("not found", "404", "does not exist", "no such model", "unknown model")
    _NO_TOOLS_ERRORS = ("does not support tools", "tool_use is not supported", "tools are not supported")
    # Vision-only models that never support tool calling — strip tools upfront to avoid 400 retry cycle
    _VISION_ONLY_MODEL_TOKENS = ("llava", "moondream", "bakllava", "minicpm-v", "llava-llama3", "llava-phi3")

    def _is_vision_only_model(self, model: str) -> bool:
        """True for models that accept images but never support tool calling (e.g. llava)."""
        m = (model or "").lower()
        return any(tok in m for tok in self._VISION_ONLY_MODEL_TOKENS)

    def _get_fallback_provider(self) -> tuple[str, str] | None:
        """Find an alternative provider with a saved key. Returns (name, model) or None."""
        from .config import get_api_key
        from .providers import PROVIDER_MODELS
        current = self.config.get("agent", {}).get("provider", "anthropic")
        if str(current or "").strip().lower() == "ollama":
            # Local-only runs must not silently escalate to cloud providers.
            return None
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
        if any(kw in err_str for kw in self._MODEL_ERRORS):
            return True
        # httpx.HTTPStatusError from OpenAI-compatible providers (Ollama/DashScope)
        # returns 400 or 404 when model doesn't exist; check the response body.
        try:
            resp = getattr(e, "response", None)
            if resp is not None:
                body = (getattr(resp, "text", None) or "").lower()
                if any(kw in body for kw in self._MODEL_ERRORS):
                    return True
                status = getattr(resp, "status_code", None)
                if status == 404:
                    return True
        except Exception:
            pass
        return False

    def _is_no_tools_error(self, e: Exception) -> bool:
        """Check if error is because the model doesn't support tools/function calling."""
        err_str = f"{type(e).__name__} {e}".lower()
        return any(kw in err_str for kw in self._NO_TOOLS_ERRORS)

    # Phrases that indicate a model leaked agent internals into user-facing response
    _INTERNAL_LEAK_MARKERS = (
        "пользователь имеет доступ к инструменту",
        "user has access to tool",
        "инструмент `read_file`",
        "инструмент `write_file`",
        "инструмент `exec_command`",
        "tool_use_id",
        "content_block_start",
        "system prompt",
        "системный промпт",
        "<tool_use>",
        "<function_calls>",
    )

    def _is_garbage_response(self, text: str) -> bool:
        """Detect incoherent/garbage model output (e.g. from llava hallucinations).

        Checks for:
        - Mixed Cyrillic+Latin characters within single words (encoding corruption)
        - High ratio of garbled tokens
        - Agent internal details leaked into user-facing text
        """
        if not text or len(text) < 10:
            return False
        s_lower = text.lower()
        # Internal state leakage
        if any(marker in s_lower for marker in self._INTERNAL_LEAK_MARKERS):
            return True
        # Mixed-script words: Cyrillic and Latin mixed inside same token
        words = text.split()
        if not words:
            return False
        garbage_words = 0
        for word in words:
            has_cyrillic = any('\u0400' <= c <= '\u04FF' for c in word)
            has_latin = any(c.isascii() and c.isalpha() for c in word)
            if has_cyrillic and has_latin and len(word) > 3:
                garbage_words += 1
        return garbage_words / max(len(words), 1) > 0.08

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

        provider_timeout = self._provider_call_timeout(provider_name)

        # Pre-strip tools for vision-only models — they never support tool calling,
        # so sending tools causes a guaranteed 400 error that wastes a round-trip.
        if self._is_vision_only_model(str(kwargs.get("model", ""))) and kwargs.get("tools"):
            logger.debug("Vision-only model '%s': stripping tools upfront", kwargs.get("model"))
            kwargs = {**kwargs, "tools": None}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await asyncio.wait_for(
                    self.provider.complete(**kwargs), timeout=provider_timeout)
                self._circuit_breaker.record_success(provider_name)
                self._set_response_route_meta(
                    provider=provider_name,
                    model=str(kwargs.get("model") or ""),
                )
                return result
            except asyncio.TimeoutError:
                self._circuit_breaker.record_failure(
                    provider_name, TimeoutError(f"LLM call timed out after {provider_timeout}s"))
                if attempt < max_retries - 1:
                    fallback = self._get_fallback_provider()
                    if fallback:
                        fb_name, fb_model = fallback
                        logger.warning("LLM timeout: %s → switching to %s",
                                       provider_name, fb_name)
                        await self._switch_provider(fb_name, fb_model)
                        kwargs["model"] = fb_model
                        provider_name = fb_name
                        continue
                raise TimeoutError(f"LLM provider timed out after {provider_timeout}s")
            except Exception as e:
                self._circuit_breaker.record_failure(provider_name, e)

                # Model not found → fall back to default_model and retry
                if self._is_model_error(e):
                    logger.warning("Model '%s' not found, falling back to '%s'",
                                   kwargs.get("model"), self.default_model)
                    kwargs["model"] = self.default_model
                    continue

                # Model doesn't support tools → retry without tools
                if self._is_no_tools_error(e) and kwargs.get("tools"):
                    logger.warning("Model '%s' does not support tools, retrying without tools",
                                   kwargs.get("model"))
                    kwargs = {**kwargs, "tools": None}
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

    # ── Recall cache + trivial message detection ──

    _TRIVIAL_MESSAGES = frozenset({
        "hello", "hi", "hey", "ok", "okay", "thanks", "thank you",
        "bye", "goodbye", "yes", "no", "hm", "hmm", "yep", "nope",
        "привет", "ок", "спасибо", "пока", "да", "нет", "хорошо",
        "ладно", "понял", "ясно", "угу", "ага", "здравствуйте",
    })

    def _should_recall(self, user_input: str) -> bool:
        """Skip memory recall for trivial messages (greetings, acks)."""
        text = user_input.strip().lower()
        if len(text) < 12:
            return False
        if text in self._TRIVIAL_MESSAGES:
            return False
        return True

    def _cached_recall(self, query: str, user_id: str, top_k: int = 5) -> list:
        """Memory recall with short-lived cache to deduplicate within one request."""
        if not hasattr(self, '_recall_cache'):
            self._recall_cache: dict[str, tuple[float, list]] = {}
        key = f"{user_id}:{query[:100]}"
        now = time.time()
        cached = self._recall_cache.get(key)
        if cached and now - cached[0] < 5.0:
            return cached[1][:top_k]
        if hasattr(self.memory, "recall_type_aware"):
            results = self.memory.recall_type_aware(query, user_id, top_k=max(top_k, 5))
        else:
            results = self.memory.recall(query, user_id, top_k=max(top_k, 5))
        self._recall_cache[key] = (now, results)
        # Evict old entries
        if len(self._recall_cache) > 50:
            oldest = min(self._recall_cache, key=lambda k: self._recall_cache[k][0])
            del self._recall_cache[oldest]
        return results[:top_k]

    def _build_system_prompt(self, user_input: str, user_id: str) -> str | list[dict]:
        """Build system prompt with memories + feature injections."""
        # Onboarding mode — return special prompt
        from .onboarding import needs_onboarding, ONBOARDING_PROMPT
        if needs_onboarding(self) and not getattr(self, "_skip_onboarding_for_request", False):
            return ONBOARDING_PROMPT

        # Recall relevant memories (skip for trivial messages, use cache)
        if self._should_recall(user_input):
            memories = self._cached_recall(
                user_input, user_id, top_k=self._memory_recall_top_k())
        else:
            memories = []
        memory_section = ""
        if memories:
            memory_lines = [f"- {m['content']}" for m in memories if m['score'] > 0.1]
            if memory_lines:
                shown_ids = [int(m.get("id", 0)) for m in memories if int(m.get("id", 0) or 0) > 0]
                used_ids = [
                    int(m.get("id", 0))
                    for m in memories
                    if int(m.get("id", 0) or 0) > 0 and float(m.get("score", 0.0) or 0.0) > 0.1
                ]
                if hasattr(self.memory, "register_recall_feedback"):
                    try:
                        self.memory.register_recall_feedback(
                            user_input,
                            user_id,
                            shown_ids,
                            used_ids,
                            strength=0.7,
                            source="system_prompt",
                        )
                    except Exception:
                        pass
                elif hasattr(self.memory, "reinforce_recall"):
                    try:
                        self.memory.reinforce_recall(
                            user_input,
                            user_id,
                            used_ids,
                            strength=0.7,
                            source="system_prompt",
                        )
                    except Exception:
                        pass
                memory_section = "\n\n## What you know about this user:\n" + "\n".join(memory_lines)

        # Precomputed compact packs (Memory Exchange + Shadow Twin).
        exchange_section = ""
        if self._should_recall(user_input):
            ex_max = 1 if self._slow_local_mode else 2
            ex_budget = 420 if self._slow_local_mode else 700
            ex_ctx = self.memory.get_memory_exchange_context(
                user_input, user_id, max_packs=ex_max, token_budget=ex_budget)
            if ex_ctx:
                exchange_section = "\n\n" + ex_ctx

        # Pinned user profile facts (name/location/language/etc.) for stable recall.
        profile_section = ""
        profile = self.memory.get_user_profile(user_id)
        canonical_profile = {}
        if hasattr(self.memory, "get_canonical_profile"):
            canonical_profile = self.memory.get_canonical_profile(user_id)
        if not profile:
            profile = self.memory.ensure_user_profile(user_id)
        if profile:
            ordered_keys = ("name", "role", "location", "language")
            labels = {
                "name": "Name",
                "role": "Role",
                "location": "Location",
                "language": "Language",
            }
            lines = []
            for k in ordered_keys:
                if profile.get(k):
                    conf_meta = canonical_profile.get(k, {}) if isinstance(canonical_profile, dict) else {}
                    conf = float(conf_meta.get("confidence", 0.0) or 0.0) if conf_meta else 0.0
                    if conf > 0:
                        lines.append(f"- {labels[k]}: {profile[k]} (confidence: {conf:.2f})")
                    else:
                        lines.append(f"- {labels[k]}: {profile[k]}")
            for k, v in profile.items():
                if k not in labels and v:
                    lines.append(f"- {k}: {v}")
            if lines:
                if self._slow_local_mode:
                    lines = lines[:4]
                profile_section = "\n\n## User profile (pinned facts):\n" + "\n".join(lines)

        thinking_section = ""
        if hasattr(self.memory, "get_thinking_cloud_context") and self._should_recall(user_input):
            try:
                thinking_section = self.memory.get_thinking_cloud_context(
                    user_id,
                    query=user_input,
                    top_k=3 if self._slow_local_mode else 5,
                )
            except Exception:
                thinking_section = ""

        # Persistent memory guardrail: avoid false "I have no long-term memory" responses.
        memory_policy_section = ""
        if getattr(self, "memory", None) is not None:
            memory_policy_section = (
                "\n\n## Memory policy:\n"
                "- You DO have persistent long-term memory for this user.\n"
                "- Never claim you only remember the current chat/session.\n"
                "- If asked what you remember, rely on User profile + recalled memories."
            )

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

        dynamic_text = (
            profile_section
            + thinking_section
            + exchange_section
            + memory_section
            + memory_policy_section
            + time_section
            + feature_section
        )

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

        if self._features.get("human_support_agent", {}).get("enabled"):
            from .evolution import detect_human_support_opportunities
            support = detect_human_support_opportunities(
                self.memory.db,
                user_id,
                user_input,
                self._features["human_support_agent"],
            )
            if support:
                parts.append(
                    "\n\n## Human support opportunities (offer gently, never force):\n"
                    "- Focus on helping the user become healthier, calmer, more organized, and more productive over time.\n"
                    "- Do not diagnose medical conditions, moralize, or pressure the user.\n"
                    "- Prefer small sustainable actions, reminders, checklists, and workload reduction.\n"
                    + "\n".join(f"- {s}" for s in support)
                )

        if self._looks_like_side_effect_request(user_input):
            parts.append(
                "\n\n## Autonomous execution policy:\n"
                "- Сначала выполни внутреннюю критическую проверку: цель, контекст, память, конфиг, риски.\n"
                "- Затем выбери минимальное обратимое действие, которое даст проверяемое доказательство.\n"
                "- После каждого существенного шага перепроверь результат по файлам, логам, тестам или ответам инструментов.\n"
                "- Не спрашивай у пользователя рутинного разрешения или подтверждения, если решение можно вывести из контекста.\n"
                "- Спрашивай только при реальном блокере: нет доступа/ключей, нужен необратимый destructive шаг, или требования конфликтуют."
            )

        if self._features.get("critical_response_review", {}).get("enabled"):
            parts.append(
                "\n\n## Critical review policy:\n"
                "- Перед важным финальным ответом выполни тихую внутреннюю перепроверку.\n"
                "- Ищи overclaiming, пропущенные caveats, противоречия контексту, и утверждения без подтверждения инструментами.\n"
                "- Если доказательств недостаточно, смягчи формулировку и явно отдели подтвержденное от предположений.\n"
                "- Не показывай сам внутренний review пользователю; просто дай более точный финальный ответ."
            )

        # Autonomous tool policy
        if self._features.get("auto_tool_synthesis", {}).get("enabled"):
            parts.append(
                "\n\n## Tool autonomy policy:\n"
                "- Сначала используй существующие инструменты.\n"
                "- Если для задачи не хватает инструмента: используй synthesize_tool, "
                "создай минимальный безопасный инструмент и сразу примени его.\n"
                "- Для задач по изображениям используй vision_analyze_image.\n"
                "- Не утверждай, что действие выполнено, пока нет реального результата tool_result."
            )

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

        # File catalog — agent awareness of user's stored files
        fm = self._file_manager
        if fm:
            try:
                recent = fm.list_files(user_id=user_id, limit=10)
                if recent:
                    lines = ["\n\n## Файлы пользователя в хранилище:"]
                    for f in recent:
                        size_kb = f['size_bytes'] // 1024
                        lines.append(
                            f"- {f['original_name']} ({f['mime_type']}, "
                            f"{size_kb}KB, key={f['storage_key']})")
                    lines.append(
                        "Используй search_files/get_file для работы с этими файлами.")
                    parts.append("\n".join(lines))
            except Exception:
                pass

        # Telegram delivery context
        try:
            from .config import get_api_key
            tg_token_present = bool(
                str(self.config.get("channels", {}).get("telegram", {}).get("token") or "").strip()
                or str(get_api_key("telegram") or "").strip()
            )
        except Exception:
            tg_token_present = False
        tg_chat_id = self._get_current_chat_id()
        if not tg_chat_id:
            try:
                remembered = self.memory.get_state("user:telegram_chat_id", user_id=user_id)
                tg_chat_id = str(remembered).strip() if remembered else ""
            except Exception:
                tg_chat_id = ""
        if not tg_chat_id:
            raw_chat = self.config.get("channels", {}).get("telegram", {}).get("chat_id") or ""
            tg_chat_id = str(raw_chat).strip()
        if tg_token_present or tg_chat_id:
            lines = ["\n\n## Telegram delivery context:"]
            if tg_token_present:
                lines.append("- Telegram bot token is already configured. Never ask the user for the Telegram token.")
            if tg_chat_id:
                lines.append("- Telegram delivery target is already available. Never ask the user for chat_id or ID чата for this request.")
                lines.append("- If the user is already messaging from Telegram, the current chat is the default delivery target.")
                lines.append("- A value like @SomeBot is the bot's username, not a destination chat_id.")
                lines.append("- If the user asks to send prepared text into Telegram, use send_text_to_user.")
                lines.append("- If the user asks to send a file or attachment, use send_file_to_user, send_stored_file, or send_stored_file_to_telegram for explicit Telegram delivery.")
                lines.append("- Never use exec_command, curl, or raw Telegram HTTP calls when Telegram delivery tools are available.")
            else:
                lines.append("- If there is no active Telegram delivery target, ask only for the destination chat, not for the token.")
            parts.append("\n".join(lines))

        # Cognitive Intelligence — Cognitive State directive
        cog_cfg = self._features.get("cognition", {})
        if cog_cfg.get("enabled", True):
            if cog_cfg.get("cognitive_state", {}).get("enabled", True):
                try:
                    from .cognition import compute_cognitive_state, COGNITIVE_STATES
                    state = compute_cognitive_state(self.memory.db, user_id)
                    directive = COGNITIVE_STATES.get(state, "")
                    if directive:
                        parts.append(f"\n\n## Response style hint:\n{directive}")
                except Exception:
                    pass

            # Cognitive Intelligence — Active session goals
            if cog_cfg.get("goal_inference", {}).get("enabled", True):
                try:
                    from .cognition import get_active_goals_prompt
                    goals_ctx = get_active_goals_prompt(self.memory.db, user_id)
                    if goals_ctx:
                        parts.append(
                            "\n\n## User's underlying goals (inferred from conversation):\n"
                            + goals_ctx
                            + "\nKeep these meta-goals in mind when answering."
                        )
                except Exception:
                    pass

        return "".join(parts)

    # ══════════════════════════════════════════
    # INTERNAL MONOLOGUE (PLANNING)
    # ══════════════════════════════════════════

    def _iteration_hard_cap(self) -> int:
        """Hard upper bound for agent loops.

        Explicit user config wins. Otherwise keep a larger safety ceiling for
        autonomous build/debug tasks, especially on slow local models.
        """
        try:
            configured = int(self.max_iterations)
        except Exception:
            configured = 15
        configured = max(1, configured)
        if self._max_iterations_explicit:
            return configured
        default_floor = 120 if self._slow_local_mode else 80
        return max(configured, default_floor)

    def _dynamic_iteration_budget(self, user_input: str, complexity_score: int,
                                  tool_defs: list | None, plan: dict | None = None) -> int:
        """Task-size-aware iteration budget under a hard safety cap."""
        text = str(user_input or "").lower()
        tool_count = len(tool_defs or [])
        hard_cap = self._iteration_hard_cap()

        if complexity_score >= 5:
            budget = 72
        elif complexity_score >= 3:
            budget = 48
        elif complexity_score >= 1:
            budget = 24
        else:
            budget = 8

        if tool_count >= 8:
            budget += 8
        elif tool_count >= 4:
            budget += 4

        big_project_markers = (
            "full-stack", "full stack", "frontend", "backend", "fastapi", "react",
            "vue", "next.js", "nextjs", "docker", "e2e", "browser", "mcp",
            "debug", "fix", "refactor", "project", "приложен", "фронтенд",
            "бэкенд", "бекенд", "браузер", "debug", "исправ", "проект",
            "запусти", "запустить", "провер", "тест",
        )
        hits = sum(1 for marker in big_project_markers if marker in text)
        if hits >= 6:
            budget = max(budget, 96)
        elif hits >= 3:
            budget = max(budget, 64)

        if any(token in text for token in ("browser", "mcp", "e2e", "chrome", "браузер")):
            budget += 8
        if any(token in text for token in ("fix", "debug", "исправ", "отлад")):
            budget += 8

        est = (plan or {}).get("estimated_iterations")
        if isinstance(est, int) and est > 0:
            budget = max(budget, max(2, int(est * 1.5) + 1))

        return max(2, min(budget, hard_cap))

    @staticmethod
    def _progress_signature(tool_calls: list[dict]) -> tuple:
        """Compact signature of the latest tool results for stall detection."""
        normalized = []
        for call in tool_calls:
            try:
                raw_input = json.dumps(call.get("input", {}), sort_keys=True, default=str)
            except Exception:
                raw_input = str(call.get("input", ""))
            normalized.append((
                str(call.get("name", "")),
                raw_input[:300],
                bool(call.get("error")),
                str(call.get("result_preview", ""))[:220],
            ))
        return tuple(normalized)

    def _advance_progress_tracker(self, tracker: dict | None,
                                  iteration_calls: list[dict]) -> dict:
        """Track whether recent iterations produced materially new results."""
        tracker = dict(tracker or {})
        signature = self._progress_signature(iteration_calls)
        progressed = bool(iteration_calls) and signature != tracker.get("last_signature")
        any_success = any(
            isinstance(call, dict) and not bool(call.get("error"))
            for call in (iteration_calls or [])
        )
        any_failure = any(
            isinstance(call, dict) and bool(call.get("error"))
            for call in (iteration_calls or [])
        )
        tracker["last_signature"] = signature
        tracker["stall_count"] = 0 if progressed else int(tracker.get("stall_count", 0)) + 1
        tracker["last_progressed"] = progressed
        tracker["failure_only_count"] = (
            int(tracker.get("failure_only_count", 0)) + 1
            if iteration_calls and any_failure and not any_success
            else 0
        )
        return tracker

    def _no_progress_limit(self, effective_max: int) -> int:
        """Allow some retries, but stop obvious low-value loops early."""
        if effective_max >= 80:
            return 6
        if effective_max >= 40:
            return 5
        return 4

    def _should_stop_for_no_progress(self, tracker: dict | None, effective_max: int) -> bool:
        return int((tracker or {}).get("stall_count", 0)) >= self._no_progress_limit(effective_max)

    def _failure_only_repair_limit(self, effective_max: int) -> int:
        """After repeated error-only tool rounds, force concrete bugfix behavior."""
        if effective_max >= 80:
            return 3
        if effective_max >= 40:
            return 2
        return 2

    def _should_force_failed_tool_repair(
        self,
        tracker: dict | None,
        user_input: str,
        forced_attempts: int,
        iteration_calls: list[dict] | None,
        effective_max: int,
    ) -> bool:
        if forced_attempts >= 1:
            return False
        if not self._looks_like_side_effect_request(user_input):
            return False
        if not iteration_calls:
            return False
        return int((tracker or {}).get("failure_only_count", 0)) >= self._failure_only_repair_limit(
            effective_max
        )

    @staticmethod
    def _forced_failed_tool_repair_prompt() -> str:
        return (
            "[System] Several consecutive tool attempts failed. Stop generic reasoning and do targeted "
            "debugging now. Use the concrete failing command output or logs from the last tool results, "
            "identify the root cause, change the relevant files, and rerun the failing verification. "
            "Do not repeat the same failing command until you have changed something relevant."
        )

    async def _build_failed_tool_repair_prompt_with_health(
        self,
        user_id: str,
        iteration_calls: list[dict] | None,
    ) -> str:
        base = self._forced_failed_tool_repair_prompt()
        snapshot = await self._collect_self_healing_health_snapshot(user_id, iteration_calls)
        guidance = (
            "\n[System] If the health snapshot shows degraded/down infrastructure, "
            "diagnose and fix the environment first. If the environment is healthy, "
            "focus on the project code/configuration."
        )
        return base + "\n\n" + snapshot + guidance

    def _no_tool_recovery_limit(self, effective_max: int) -> int:
        """Force material action if a side-effect task burns multiple passes without any tools."""
        if effective_max >= 80:
            return 2
        return 1

    def _should_force_tool_first_recovery(
        self,
        user_input: str,
        tool_calls_log: list[dict] | None,
        forced_attempts: int,
        no_tool_passes: int,
        effective_max: int,
    ) -> bool:
        if forced_attempts >= 1:
            return False
        if not self._looks_like_side_effect_request(user_input):
            return False
        if any(isinstance(tc, dict) for tc in (tool_calls_log or [])):
            return False
        return no_tool_passes >= self._no_tool_recovery_limit(effective_max)

    @staticmethod
    def _forced_tool_first_recovery_prompt() -> str:
        return (
            "[System] You have already spent multiple model passes on a side-effect task without any real "
            "tool_result. Stop planning and stop summarizing. In the very next reply, call concrete tools "
            "immediately: inspect files, run commands, or edit code. Do not answer with analysis-only text."
        )

    async def _build_no_tool_recovery_prompt_with_health(
        self,
        user_id: str,
        tool_calls_log: list[dict] | None = None,
    ) -> str:
        base = self._forced_no_tool_recovery_prompt()
        snapshot = await self._collect_self_healing_health_snapshot(user_id, tool_calls_log)
        guidance = (
            "\n[System] Start with the smallest health-check or inspection step that can confirm "
            "whether the blocker is environment, configuration, or project code."
        )
        return base + "\n\n" + snapshot + guidance

    async def _build_autonomy_recovery_prompt_with_health(
        self,
        user_id: str,
        tool_calls_log: list[dict] | None = None,
    ) -> str:
        base = self._forced_autonomy_recovery_prompt()
        snapshot = await self._collect_self_healing_health_snapshot(user_id, tool_calls_log)
        guidance = (
            "\n[System] Use the snapshot to decide whether to inspect environment, configuration, "
            "or project files first. Prefer evidence over asking the user."
        )
        return base + "\n\n" + snapshot + guidance

    async def _collect_self_healing_health_snapshot(
        self,
        user_id: str,
        recent_tool_calls: list[dict] | None = None,
    ) -> str:
        """Build a compact environment and failure snapshot for health-aware recovery."""
        route = dict(self._last_response_meta.get("response_route", {}) or {})
        provider_name = str(
            route.get("provider")
            or self.config.get("agent", {}).get("provider", "")
            or "unknown"
        ).strip()
        model_name = str(route.get("model") or self.default_model or "unknown").strip()

        lines = [
            "[System] Self-healing health snapshot:",
            f"- Runtime: provider={provider_name}, model={model_name}",
        ]

        recent_failures = []
        for call in (recent_tool_calls or [])[-3:]:
            if not isinstance(call, dict):
                continue
            status = "error" if call.get("error") else "ok"
            preview = str(call.get("result_preview", "") or "").strip()
            preview = " ".join(preview.split())
            if len(preview) > 140:
                preview = preview[:137].rstrip() + "..."
            recent_failures.append(
                f"{call.get('name', 'tool')}[{status}] -> {preview or 'no preview'}"
            )
        if recent_failures:
            lines.append("- Recent tool outcomes:")
            lines.extend(f"  {item}" for item in recent_failures)

        memory_health = None
        if hasattr(self.memory, "memory_health_check"):
            try:
                memory_health = self.memory.memory_health_check(user_id)
            except Exception as e:
                memory_health = {"status": "unknown", "issues": [f"memory health check failed: {e}"]}
        if isinstance(memory_health, dict):
            status = str(memory_health.get("status", "unknown"))
            issues = [str(x) for x in (memory_health.get("issues") or []) if str(x).strip()]
            if issues:
                lines.append(f"- Memory health: {status} ({'; '.join(issues[:2])})")
            else:
                lines.append(f"- Memory health: {status}")

        hm = getattr(self, "_health_monitor", None)
        if hm and hasattr(hm, "run_all_checks"):
            try:
                results = await hm.run_all_checks()
                if isinstance(results, dict) and results:
                    summarized = []
                    for name, health in results.items():
                        health_name = str(getattr(health, "name", name) or name)
                        status = str(getattr(health, "status", "unknown") or "unknown")
                        latency_ms = getattr(health, "latency_ms", 0.0) or 0.0
                        err = str(getattr(health, "error_message", "") or "").strip()
                        piece = f"{health_name}={status}"
                        if latency_ms:
                            piece += f" ({latency_ms:.0f}ms)"
                        if err and status != "healthy":
                            piece += f" [{err[:80]}]"
                        summarized.append(piece)
                    lines.append("- Environment health: " + "; ".join(summarized[:4]))
            except Exception as e:
                lines.append(f"- Environment health: unavailable ({e})")

        return "\n".join(lines)

    async def _apply_planning(self, user_input: str, user_id: str,
                              system_prompt, tool_defs: list, model: str,
                              complexity_score: int = 0,
                              ) -> tuple:
        """Apply internal-monologue planning before the agent loop.

        Returns ``(plan, tool_defs, model, effective_max_iterations)``.
        All arguments may be modified; ``system_prompt`` is mutated in place.
        On any error the original values are returned unchanged.
        """
        im_cfg = self._features.get("internal_monologue", {})
        if not im_cfg.get("enabled"):
            return None, tool_defs, model, self._dynamic_iteration_budget(
                user_input, complexity_score, tool_defs)
        if (self._slow_local_mode
                and self._slow_local_cfg.get("disable_planning", True)
                and self._complexity_score(user_input) < 3):
            return None, tool_defs, model, self._dynamic_iteration_budget(
                user_input, complexity_score, tool_defs)

        try:
            from .planning import generate_plan, format_plan_for_prompt

            # Pass default_model so resolve_planning_model can use it for Ollama
            plan_cfg = dict(im_cfg)
            plan_cfg["_default_model"] = self.default_model

            plan = await generate_plan(
                self.provider, user_input,
                self._cached_recall(user_input, user_id, top_k=3),
                tool_defs, plan_cfg)

            if not plan:
                return None, tool_defs, model, self._dynamic_iteration_budget(
                    user_input, complexity_score, tool_defs)

            # 1. Inject plan text into system prompt
            plan_text = format_plan_for_prompt(plan)
            if isinstance(system_prompt, list):
                system_prompt[-1]["text"] += plan_text
            else:
                system_prompt += plan_text

            # 2. Cap iterations (×1.5 buffer, minimum 2, capped by max_iterations)
            effective_max = self._dynamic_iteration_budget(
                user_input, complexity_score, tool_defs, plan=plan)
            est = plan.get("estimated_iterations")
            if est and isinstance(est, int) and est > 0:
                effective_max = min(int(est * 1.5) + 1, effective_max)
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
            return None, tool_defs, model, self._dynamic_iteration_budget(
                user_input, complexity_score, tool_defs)

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
        if self._slow_local_mode and self._slow_local_cfg.get("disable_reflection", True):
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
        import re as _re
        text = user_input.lower()
        score = 0

        # Length heuristic
        if len(user_input) > 500:
            score += 2
        elif len(user_input) > 100:
            score += 1

        # Code detection: fences or language-specific patterns
        if "```" in user_input:
            score += 2
        elif any(pat in text for pat in (
            "def ", "class ", "function ", "import ", "from ",
            "const ", "var ", "let ", "#include", "async def ",
        )):
            score += 1

        # Multi-part requests (numbered lists, bullet points)
        numbered = _re.findall(r'^\s*\d+[.)]\s', user_input, _re.MULTILINE)
        bullets = _re.findall(r'^\s*[-*]\s', user_input, _re.MULTILINE)
        if len(numbered) >= 3 or len(bullets) >= 3:
            score += 2
        elif len(numbered) >= 2 or len(bullets) >= 2:
            score += 1

        # Keyword markers with word boundaries — accumulate from complex set
        complex_hits = sum(
            1 for m in COMPLEXITY_MARKERS_COMPLEX
            if _re.search(r'\b' + _re.escape(m) + r'\b', text)
        )
        score += complex_hits * 2

        # Medium markers (only if no complex hits), with word boundaries
        if complex_hits == 0:
            medium_hits = sum(
                1 for m in COMPLEXITY_MARKERS_MEDIUM
                if _re.search(r'\b' + _re.escape(m) + r'\b', text)
            )
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

    def _split_model_spec(self, model_spec: str) -> tuple[str, str]:
        """Return (provider, bare_model) for a model spec."""
        spec = str(model_spec or "").strip()
        current_provider = str(
            self.config.get("agent", {}).get("provider", "anthropic")
        ).strip().lower()
        if not spec:
            return current_provider, str(self.default_model or "").strip()
        if ":" in spec:
            prefix, rest = spec.split(":", 1)
            if prefix.lower() in self._KNOWN_PROVIDERS and rest.strip():
                return prefix.lower(), rest.strip()
        inferred = self._infer_provider_for_model(spec)
        if inferred:
            return inferred, spec
        return current_provider, spec

    def _routing_task_profile(self, user_input: str, complexity_score: int) -> dict:
        """Summarize request needs for intelligent routing."""
        import re as _re

        text = str(user_input or "")
        low = text.lower()
        code_markers = (
            "```", "def ", "class ", "function ", "import ", "from ",
            "const ", "let ", "var ", "async def ", "pytest", "fastapi",
            "react", "typescript", "javascript", "python", "refactor",
        )
        coding = any(marker in low for marker in code_markers)
        planning = any(token in low for token in (
            "plan", "спланируй", "architecture", "архитектур", "design",
            "strategy", "strateg", "debug", "analyze", "проанализируй",
        ))
        multipart = bool(
            len(_re.findall(r'^\s*\d+[.)]\s', text, _re.MULTILINE)) >= 2
            or len(_re.findall(r'^\s*[-*]\s', text, _re.MULTILINE)) >= 2
        )
        multimodal = any(token in low for token in (
            "image", "vision", "screenshot", "pdf", "document", "docx",
            "изображ", "скриншот", "картин", "pdf", "документ",
        ))
        urgency = len(text) < 120 and complexity_score <= 1 and not multipart and not planning
        quality_priority = complexity_score >= 3 or planning or multipart or len(text) > 500
        cost_priority = not quality_priority and complexity_score <= 1 and not coding
        speed_priority = urgency and not coding
        objective = "balanced"
        if quality_priority:
            objective = "quality"
        elif speed_priority:
            objective = "speed"
        elif cost_priority:
            objective = "cost"
        return {
            "complexity_score": int(complexity_score),
            "tier": self._tier_for_score(complexity_score),
            "coding": bool(coding),
            "planning": bool(planning),
            "multipart": bool(multipart),
            "multimodal": bool(multimodal),
            "quality_priority": bool(quality_priority),
            "speed_priority": bool(speed_priority),
            "cost_priority": bool(cost_priority),
            "objective": objective,
            "local_only": bool(self._is_local_only_hours()),
        }

    def _model_routing_profile(self, model_spec: str) -> dict:
        """Infer cost/latency/capability profile for a model."""
        provider_name, bare_model = self._split_model_spec(model_spec)
        low = bare_model.lower()
        pricing = get_pricing(bare_model)
        local = provider_name == "ollama"
        combined_price = float(pricing.get("input", 0.0)) + float(pricing.get("output", 0.0))

        speed_rank = 2
        quality_rank = 2
        coding_rank = 1

        if local:
            speed_rank = 3
            if any(token in low for token in ("30b", "32b", "34b", "70b", "72b")):
                speed_rank = 1
            elif any(token in low for token in ("14b", "13b")):
                speed_rank = 2
            elif any(token in low for token in ("7b", "8b", "latest", "mini")):
                speed_rank = 3

        if any(token in low for token in ("mini", "nano", "flash", "haiku", "turbo", "fast")):
            speed_rank = max(speed_rank, 3)
        if any(token in low for token in ("opus", "pro", "max", "o1", "o3", "30b", "32b", "70b", "72b")):
            speed_rank = min(speed_rank, 1)

        if any(token in low for token in ("opus", "pro", "max", "o1", "o3", "grok-4", "30b", "32b", "70b", "72b")):
            quality_rank = 3
        elif any(token in low for token in ("sonnet", "4o", "4.1", "plus", "14b", "13b", "coder")):
            quality_rank = 2
        elif any(token in low for token in ("mini", "nano", "flash", "haiku", "turbo", "7b", "8b")):
            quality_rank = 1

        if any(token in low for token in ("coder", "sonnet", "4.1", "4o", "qwen-max", "opus", "pro", "max", "30b", "32b")):
            coding_rank = 3
        elif any(token in low for token in ("plus", "flash", "haiku", "mini", "14b", "13b")):
            coding_rank = 2

        if local or combined_price <= 1.0:
            cost_rank = 3
        elif combined_price <= 8.0:
            cost_rank = 2
        else:
            cost_rank = 1

        return {
            "model": model_spec,
            "provider": provider_name,
            "bare_model": bare_model,
            "local": local,
            "pricing": pricing,
            "speed_rank": speed_rank,
            "quality_rank": quality_rank,
            "coding_rank": coding_rank,
            "cost_rank": cost_rank,
            "vision": any(token in low for token in ("vl", "vision", "gpt-4o", "gemini", "llava", "moondream")),
            "input_per_mtok": float(pricing.get("input", 0.0)),
            "output_per_mtok": float(pricing.get("output", 0.0)),
        }

    def _cascade_candidates(self) -> list[dict]:
        """Return distinct cascade candidates with tier metadata."""
        entries: dict[str, dict] = {}
        for tier_name in ("simple", "medium", "complex"):
            model_spec = str(self.models.get(tier_name, self.default_model) or "").strip()
            if not model_spec:
                continue
            entry = entries.setdefault(model_spec, {"model": model_spec, "tiers": []})
            entry["tiers"].append(tier_name)
        default_model = str(self.default_model or "").strip()
        if default_model:
            entry = entries.setdefault(default_model, {"model": default_model, "tiers": []})
            if "default" not in entry["tiers"]:
                entry["tiers"].append("default")

        results = []
        for model_spec, entry in entries.items():
            prof = self._model_routing_profile(model_spec)
            prof["tiers"] = list(entry["tiers"])
            results.append(prof)
        return results

    def _score_route_candidate(self, candidate: dict, task: dict) -> tuple[float, list[str]]:
        """Score a candidate model for a specific task profile."""
        score = 0.0
        reasons: list[str] = []

        if task.get("local_only") and not candidate.get("local"):
            return -999.0, ["blocked by local-only schedule"]

        if task.get("quality_priority"):
            score += candidate.get("quality_rank", 1) * 3.0
            if candidate.get("quality_rank", 1) >= 3:
                reasons.append("strong quality tier")
        else:
            score += candidate.get("quality_rank", 1) * 1.2

        if task.get("coding"):
            score += candidate.get("coding_rank", 1) * 2.5
            if candidate.get("coding_rank", 1) >= 3:
                reasons.append("coding-optimized")
        else:
            score += candidate.get("coding_rank", 1) * 0.4

        if task.get("speed_priority"):
            score += candidate.get("speed_rank", 1) * 2.5
            if candidate.get("speed_rank", 1) >= 3:
                reasons.append("fast path")
        else:
            score += candidate.get("speed_rank", 1) * 0.8

        if task.get("cost_priority"):
            score += candidate.get("cost_rank", 1) * 2.5
            if candidate.get("cost_rank", 1) >= 3:
                reasons.append("low cost")
        else:
            score += candidate.get("cost_rank", 1) * 0.8

        if task.get("multimodal"):
            if candidate.get("vision"):
                score += 1.5
                reasons.append("vision capable")
            else:
                score -= 1.0

        if "complex" in candidate.get("tiers", []):
            score += 0.6
        elif "medium" in candidate.get("tiers", []):
            score += 0.3

        return score, reasons[:3]

    def _route_gap_suggestions(self, gap: str) -> list[str]:
        """Return concrete model suggestions for a detected cascade gap."""
        suggestions = {
            "diversity": ["qwen2.5:latest", "gpt-4o-mini", "claude-opus-4-20250115"],
            "fast_cheap": ["qwen2.5:latest", "gpt-4o-mini", "gemini-2.5-flash"],
            "strong_reasoning": ["qwen3-coder:30b", "claude-opus-4-20250115", "gemini-2.5-pro"],
            "coding": ["qwen3-coder:30b", "gpt-4.1", "claude-sonnet-4-20250514"],
            "local_coverage": ["qwen2.5:latest", "qwen3-coder:30b", "llama3.1:8b"],
        }
        values = []
        for model_name in suggestions.get(gap, []):
            if model_name not in values:
                values.append(model_name)
        return values[:3]

    def _cascade_recommendations(self) -> list[dict]:
        """Detect gaps in current cascade config and recommend additions."""
        candidates = self._cascade_candidates()
        if not candidates:
            return []
        recommendations: list[dict] = []
        unique_models = {c["model"] for c in candidates}
        tier_map = {}
        for tier_name in ("simple", "medium", "complex"):
            tier_map[tier_name] = self._model_routing_profile(
                self.models.get(tier_name, self.default_model)
            )

        def _add(gap: str, severity: str, message: str):
            recommendations.append({
                "gap": gap,
                "severity": severity,
                "message": message,
                "suggested_models": self._route_gap_suggestions(gap),
            })

        if len(unique_models) == 1:
            _add("diversity", "high", "All cascade tiers currently use the same model, so routing cannot optimize for speed, cost, or quality.")
        if tier_map["simple"].get("cost_rank", 1) < 3 or tier_map["simple"].get("speed_rank", 1) < 3:
            _add("fast_cheap", "medium", "Simple tier is not especially cheap or fast. Add a lighter model for low-risk prompts.")
        if tier_map["complex"].get("quality_rank", 1) < 3:
            _add("strong_reasoning", "high", "Complex tier lacks a clearly stronger reasoning model for hard planning, debugging, and large code tasks.")
        if max(c.get("coding_rank", 1) for c in candidates) < 3:
            _add("coding", "medium", "Cascade has no clearly coding-optimized model. Large implementation tasks will be slower or less reliable.")
        if self._is_local_only_hours() and not any(c.get("local") for c in candidates):
            _add("local_coverage", "high", "Local-only hours are enabled, but the cascade has no local model configured.")

        last_gap = dict(self._last_cascade_route or {}).get("gap")
        if last_gap and last_gap not in {item["gap"] for item in recommendations} and last_gap != "none":
            _add(last_gap, "high", str(self._last_cascade_route.get("recommendation") or "Latest routed task exposed a capability gap in the current cascade."))

        return recommendations[:4]

    def get_cascade_dashboard_state(self) -> dict:
        """Return routing advisor state + recommendations for dashboard."""
        candidates = self._cascade_candidates()
        return {
            "advisor": {
                "enabled": bool(self._intelligent_routing_cfg.get("enabled", True)),
                "use_llm": bool(self._intelligent_routing_cfg.get("use_llm", True)),
                "advisor_model": str(self._intelligent_routing_cfg.get("advisor_model", "") or "").strip(),
                "last_route": dict(self._last_cascade_route or {}),
            },
            "candidates": [
                {
                    "model": item.get("model"),
                    "tiers": item.get("tiers", []),
                    "provider": item.get("provider"),
                    "local": bool(item.get("local")),
                    "speed_rank": int(item.get("speed_rank", 0)),
                    "quality_rank": int(item.get("quality_rank", 0)),
                    "coding_rank": int(item.get("coding_rank", 0)),
                    "cost_rank": int(item.get("cost_rank", 0)),
                    "input_per_mtok": float(item.get("input_per_mtok", 0.0)),
                    "output_per_mtok": float(item.get("output_per_mtok", 0.0)),
                }
                for item in candidates
            ],
            "recommendations": self._cascade_recommendations(),
        }

    def _heuristic_route_decision(self, user_input: str, complexity_score: int) -> dict:
        """Choose best available cascade candidate without calling another model."""
        task = self._routing_task_profile(user_input, complexity_score)
        candidates = self._cascade_candidates()
        scored = []
        for candidate in candidates:
            score, reasons = self._score_route_candidate(candidate, task)
            scored.append((score, candidate, reasons))
        scored.sort(key=lambda item: item[0], reverse=True)

        chosen = scored[0][1] if scored else self._model_routing_profile(self.default_model)
        chosen_score = scored[0][0] if scored else 0.0
        reasons = scored[0][2] if scored else []
        gap = "none"
        recommendation = ""
        if task.get("quality_priority") and int(chosen.get("quality_rank", 1)) < 3:
            gap = "strong_reasoning"
            recommendation = "Current cascade has no clearly stronger reasoning model for this task."
        elif task.get("coding") and int(chosen.get("coding_rank", 1)) < 3:
            gap = "coding"
            recommendation = "Current cascade lacks a coding-optimized model for this task."
        elif task.get("speed_priority") and int(chosen.get("speed_rank", 1)) < 3:
            gap = "fast_cheap"
            recommendation = "Current cascade lacks a fast low-cost model for lightweight prompts."

        return {
            "model_spec": chosen.get("model", self.default_model),
            "tier": (
                chosen.get("tiers", [self._tier_for_score(complexity_score)])[0]
                if chosen.get("tiers") else self._tier_for_score(complexity_score)
            ),
            "decision_source": "heuristic",
            "objective": task.get("objective", "balanced"),
            "reason": ", ".join(reasons) if reasons else "best available match from current cascade",
            "gap": gap,
            "recommendation": recommendation,
            "task_profile": task,
            "candidates": [
                {
                    "model": item[1].get("model"),
                    "score": round(float(item[0]), 2),
                    "reasons": item[2],
                }
                for item in scored[:3]
            ],
        }

    @staticmethod
    def _extract_json_object(text: str) -> dict:
        """Extract a JSON object from a short LLM response."""
        raw = str(text or "").strip()
        if not raw:
            return {}
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    async def _call_routing_advisor(self, user_input: str, complexity_score: int, heuristic: dict) -> dict:
        """Ask a routing advisor model to choose the best candidate from current cascade."""
        cfg = dict(self._intelligent_routing_cfg or {})
        if not cfg.get("enabled", True) or not cfg.get("use_llm", True):
            return {}
        if complexity_score < int(cfg.get("min_complexity", 1) or 1):
            return {}
        if self._slow_local_mode and complexity_score < int(cfg.get("local_min_complexity", 2) or 2):
            return {}

        candidates = self._cascade_candidates()
        if len(candidates) < 2:
            return {}

        advisor_model = str(cfg.get("advisor_model", "") or "").strip() or str(self.models.get("simple", "") or "").strip()
        if not advisor_model:
            return {}
        advisor_provider, advisor_bare = self._split_model_spec(advisor_model)
        if not advisor_bare:
            return {}

        provider_obj = self.provider
        current_provider = str(self.config.get("agent", {}).get("provider", "anthropic")).strip().lower()
        if advisor_provider != current_provider:
            temp_cfg = copy.deepcopy(self.config)
            temp_cfg.setdefault("agent", {})["provider"] = advisor_provider
            try:
                provider_obj = create_provider(temp_cfg)
            except Exception as e:
                logger.debug("Routing advisor provider init failed for %s/%s: %s",
                             advisor_provider, advisor_bare, e)
                return {}

        task = heuristic.get("task_profile") or self._routing_task_profile(user_input, complexity_score)
        compact_candidates = []
        for item in candidates:
            compact_candidates.append({
                "model": item.get("model"),
                "tiers": item.get("tiers", []),
                "provider": item.get("provider"),
                "local": bool(item.get("local")),
                "input_per_mtok": float(item.get("input_per_mtok", 0.0)),
                "output_per_mtok": float(item.get("output_per_mtok", 0.0)),
                "speed_rank": int(item.get("speed_rank", 0)),
                "quality_rank": int(item.get("quality_rank", 0)),
                "coding_rank": int(item.get("coding_rank", 0)),
                "cost_rank": int(item.get("cost_rank", 0)),
            })

        prompt = (
            "You are a routing advisor for an LLM cascade.\n"
            "Pick the cheapest and fastest candidate that can still solve the task reliably.\n"
            "Escalate quality only when the task genuinely needs it.\n"
            "Return JSON only.\n\n"
            f"Task: {json.dumps(task, ensure_ascii=True)}\n"
            f"Heuristic baseline: {json.dumps({k: heuristic.get(k) for k in ('model_spec', 'tier', 'objective', 'gap')}, ensure_ascii=True)}\n"
            f"Candidates: {json.dumps(compact_candidates, ensure_ascii=True)}\n\n"
            'Schema: {"target_model":"candidate model string","tradeoff":"cost|speed|quality|balanced",'
            '"reason":"short reason","gap":"none|fast_cheap|strong_reasoning|coding|local_coverage",'
            '"recommendation":"optional short recommendation"}'
        )

        try:
            timeout_sec = float(cfg.get("timeout_sec", 8.0) or 8.0)
            response = await asyncio.wait_for(
                provider_obj.complete(
                    advisor_bare,
                    max_tokens=220,
                    messages=[{"role": "user", "content": prompt}],
                    system="Return JSON only. Do not use tools.",
                    tools=None,
                    temperature=0,
                ),
                timeout=timeout_sec,
            )
        except Exception as e:
            logger.debug("Routing advisor failed: %s", e)
            return {}

        parsed = self._extract_json_object(self._extract_text(response))
        target_model = str(parsed.get("target_model", "") or "").strip()
        if target_model not in {item.get("model") for item in compact_candidates}:
            return {}
        return {
            "model_spec": target_model,
            "tier": next(
                (
                    (item.get("tiers") or [heuristic.get("tier") or self._tier_for_score(complexity_score)])[0]
                    for item in compact_candidates if item.get("model") == target_model
                ),
                heuristic.get("tier") or self._tier_for_score(complexity_score),
            ),
            "decision_source": "advisor",
            "objective": str(parsed.get("tradeoff") or task.get("objective") or "balanced"),
            "reason": str(parsed.get("reason") or "")[:240],
            "gap": str(parsed.get("gap") or "none"),
            "recommendation": str(parsed.get("recommendation") or "")[:240],
            "advisor_model": advisor_model,
            "task_profile": task,
            "candidates": heuristic.get("candidates", []),
        }

    async def _select_model_for_request(
        self,
        user_input: str,
        *,
        user_id: str = "",
        requested_model: str | None = None,
        complexity_score: int | None = None,
    ) -> dict:
        """Return selected model metadata for a request."""
        score = self._complexity_score(user_input) if complexity_score is None else int(complexity_score)
        selected_tier = self._tier_for_score(score) if self.cascade_routing else "fixed"

        if requested_model:
            model = self._resolve_requested_model(requested_model)
            result = {
                "model": model,
                "tier": "manual",
                "complexity_score": score,
                "decision_source": "manual",
                "requested_model": requested_model,
            }
            self._last_cascade_route = dict(result)
            return result

        if user_id and hasattr(self, "_conv_model") and self._conv_model:
            override = self._conv_model.get(user_id)
            if override:
                result = {
                    "model": override,
                    "tier": "override",
                    "complexity_score": score,
                    "decision_source": "conversation_override",
                }
                self._last_cascade_route = dict(result)
                return result

        if not self.cascade_routing:
            result = {
                "model": self.default_model,
                "tier": "fixed",
                "complexity_score": score,
                "decision_source": "fixed",
            }
            self._last_cascade_route = dict(result)
            return result

        heuristic = self._heuristic_route_decision(user_input, score)
        route = dict(heuristic)
        advisor = await self._call_routing_advisor(user_input, score, heuristic)
        if advisor:
            route.update(advisor)
        model = self._resolve_requested_model(route.get("model_spec", self.default_model))
        result = {
            "model": model,
            "tier": route.get("tier") or selected_tier,
            "complexity_score": score,
            "decision_source": route.get("decision_source", "heuristic"),
            "objective": route.get("objective", "balanced"),
            "reason": route.get("reason", ""),
            "gap": route.get("gap", "none"),
            "recommendation": route.get("recommendation", ""),
            "advisor_model": route.get("advisor_model", ""),
            "candidates": route.get("candidates", []),
        }
        self._last_cascade_route = dict(result)
        return result

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

        # Tool-capability guard: small models often can't call tools reliably.
        # If we're routing to "simple" and there are registered tools,
        # promote to "medium" model which is more likely to support tool use.
        if score < 1 and hasattr(self, "tools") and self.tools:
            simple_model = self.models.get("simple", "")
            medium_model = self.models.get("medium", self.default_model)
            if simple_model and simple_model != medium_model and candidate == simple_model:
                logger.info("Tool-capability guard: %s → %s (tools registered, "
                            "simple model may not support tool calling)", candidate, medium_model)
                candidate = medium_model

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

        # Guard: if cascade model doesn't match current provider, try auto-switching
        if not self.provider.supports_model(candidate):
            inferred = self._infer_provider_for_model(candidate)
            if inferred:
                current_provider = self.config.get("agent", {}).get("provider", "anthropic")
                if inferred != current_provider:
                    self._cascade_switch_provider(inferred)
                    logger.info("Cascade: auto-switching provider %s → %s for model %s",
                                current_provider, inferred, candidate)
                    return candidate
            logger.warning("Cascade model '%s' not supported by any known provider, "
                           "falling back to '%s'", candidate, self.default_model)
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

    _KNOWN_PROVIDERS = frozenset(("anthropic", "openai", "gemini", "ollama", "qwen", "grok"))

    def _resolve_requested_model(self, requested_model: str) -> str:
        """Parse provider:model prefix, switch provider if needed, return bare model name.

        Allows dropdown/API callers to select cross-provider models without cascade routing.
        E.g. 'qwen:qwen-vl-plus' → switches to qwen provider, returns 'qwen-vl-plus'.
        Plain model names are sent to the current provider unchanged.
        """
        if not requested_model:
            return requested_model
        if ":" in requested_model:
            parts = requested_model.split(":", 1)
            prefix, bare = parts[0].lower(), parts[1]
            if prefix in self._KNOWN_PROVIDERS:
                current = self.config.get("agent", {}).get("provider", "anthropic")
                if prefix != current:
                    self._cascade_switch_provider(prefix)
                    logger.info("Manual model selection: switching provider %s → %s (%s)",
                                current, prefix, bare)
                return bare
        current = self.config.get("agent", {}).get("provider", "anthropic")
        if not self._model_matches_provider(current, requested_model):
            inferred = self._infer_provider_for_model(requested_model)
            if inferred and inferred != current:
                self._cascade_switch_provider(inferred)
                logger.info("Manual model selection: inferred provider %s → %s (%s)",
                            current, inferred, requested_model)
        return requested_model

    def _infer_provider_for_model(self, model: str) -> str:
        """Infer the correct provider from a bare model name.

        Used when cascade routing config has cloud model names (claude-*, gpt-*, gemini-*)
        without an explicit provider: prefix. Returns provider name or '' if unknown.
        """
        m = model.lower()
        if ":" in m and m.split(":", 1)[0] not in self._KNOWN_PROVIDERS:
            return "ollama"
        if m.startswith("claude-") or m.startswith("anthropic/"):
            return "anthropic"
        if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
            return "openai"
        if m.startswith("gemini-") or m.startswith("models/"):
            return "gemini"
        if m.startswith("qwen"):
            return "qwen"
        if m.startswith("grok"):
            return "grok"
        return ""

    def _select_model(self, user_input: str, user_id: str = "") -> str:
        """Route to Haiku/Sonnet/Opus based on query complexity.

        Per-conversation override (from OpenClaw model-overrides.ts) takes
        precedence over cascade routing when set via /model command.
        """
        # Check per-user model override first
        if user_id and hasattr(self, '_conv_model') and self._conv_model:
            from .conv_model import resolve_model_for_user
            override = self._conv_model.get(user_id)
            if override:
                return override
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
    def _record_cascade_decision(cls, model: str, tier: str, score: int,
                                 decision_source: str = "", objective: str = "",
                                 gap: str = ""):
        """Record a cascade routing decision for dashboard visualization."""
        cls._cascade_history.append({
            "model": model,
            "tier": tier,
            "score": score,
            "decision_source": decision_source or "heuristic",
            "objective": objective or "balanced",
            "gap": gap or "none",
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
        source_counts = {"heuristic": 0, "advisor": 0, "manual": 0, "fixed": 0, "other": 0}
        objectives = {"speed": 0, "cost": 0, "quality": 0, "balanced": 0}
        gaps = {}
        for d in today_decisions:
            t = d.get("tier", "medium")
            if t in tier_counts:
                tier_counts[t] = tier_counts.get(t, 0) + 1
            source = str(d.get("decision_source") or "other")
            if source.startswith("manual") or source == "conversation_override":
                source_counts["manual"] += 1
            elif source in source_counts:
                source_counts[source] += 1
            else:
                source_counts["other"] += 1
            objective = str(d.get("objective") or "balanced")
            if objective in objectives:
                objectives[objective] += 1
            gap = str(d.get("gap") or "none")
            if gap and gap != "none":
                gaps[gap] = gaps.get(gap, 0) + 1
        last = cls._cascade_history[-1] if cls._cascade_history else None
        return {
            "tier_counts": tier_counts,
            "source_counts": source_counts,
            "objective_counts": objectives,
            "gaps": gaps,
            "total_decisions": len(today_decisions),
            "last_decision": last,
        }

    # ══════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════

    @staticmethod
    @staticmethod
    def _extract_text(response) -> str:
        """Extract text from API response, stripping thinking/reasoning blocks."""
        import re
        parts = []
        for block in response.content:
            if block.type == "text":
                parts.append(block.text)
        text = "\n".join(parts)
        # Strip <think>...</think> reasoning blocks (Ollama thinking models)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        return text

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Strip <think>...</think> reasoning blocks from model output."""
        import re
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    @staticmethod
    def _clean_response_artifacts(text: str) -> str:
        """Strip model tool-narration artifacts from the final response text.

        Some models output <execute_tool>print(...)</execute_tool> blocks and
        _thought internal monologue alongside native tool_use calls.  These
        should never be stored in memory or shown to the user.
        """
        import re
        # Remove <execute_tool>...</execute_tool> blocks
        text = re.sub(r'<execute_tool>.*?</execute_tool>\s*', '', text, flags=re.DOTALL)
        # Remove _thought\n... blocks (internal reasoning)
        text = re.sub(r'_thought\s*\n.*', '', text, flags=re.DOTALL)
        # Remove inline send_status("...") / send_status('...') calls left as text
        text = re.sub(r'\bsend_status\(["\'].*?["\']\)\s*\n?', '', text)
        # Clean up extra blank lines left after removal
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _try_parse_text_tool_call(self, text: str, tool_defs: list) -> dict | None:
        """Try to parse a tool call from plain text (fallback for models without structured tool_use).

        Some models (e.g. Ollama/qwen) output tool calls as text with single quotes
        instead of structured tool_use blocks. This method tries multiple parsing strategies.

        Returns {"name": ..., "arguments": ...} or None.
        """
        results = self._try_parse_text_tool_calls(text, tool_defs)
        return results[0] if results else None

    def _try_parse_text_tool_calls(self, text: str, tool_defs: list) -> list[dict]:
        """Try to parse one or more tool calls from plain text.

        Returns list of {"name": ..., "arguments": ...} dicts.
        """
        import ast, re

        def _extract_parenthesized_call_args(src: str, open_paren_idx: int) -> tuple[str, int] | None:
            """Return inner args text and closing paren index for a function call."""
            if open_paren_idx < 0 or open_paren_idx >= len(src) or src[open_paren_idx] != "(":
                return None
            depth = 0
            quote: str | None = None
            escaped = False
            for idx in range(open_paren_idx, len(src)):
                ch = src[idx]
                if quote:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == quote:
                        quote = None
                    continue
                if ch in {"'", '"'}:
                    quote = ch
                    continue
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        return src[open_paren_idx + 1:idx], idx
            return None

        # Strip thinking/reasoning blocks before parsing
        text = self._strip_thinking(text).strip()
        if not text:
            return []

        known_names = {t["name"] for t in tool_defs}
        results = []

        # Strategy 5: XML/function tag format (qwen3-coder, etc.)
        # Pattern: <function=tool_name>\n<parameter=key>value</parameter>\n</function>
        # Check FIRST because these can co-exist with other text
        if "<function=" in text:
            for fn_match in re.finditer(
                    r'<function=(\w+)>(.*?)</function>', text, re.DOTALL):
                fn_name = fn_match.group(1)
                fn_body = fn_match.group(2)
                params = {}
                for pm in re.finditer(
                        r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', fn_body, re.DOTALL):
                    val = pm.group(2).strip()
                    # Try to parse value as JSON for proper typing
                    try:
                        val = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        pass
                    params[pm.group(1)] = val
                if fn_name and params:
                    validated = self._validate_tool_call(fn_name, params, known_names)
                    if validated:
                        results.append(validated)
                        logger.info("Parsed XML-style tool call: %s(%s)", fn_name, list(params.keys()))
            if results:
                return results

        # Strategy 6b: <execute_tool>print(fn_name(key=val, ...))</execute_tool>
        # Some models output tool calls as Python print() calls inside <execute_tool> tags.
        # Multiple calls per block and multiple blocks per response supported.
        if "<execute_tool>" in text:
            for et_match in re.finditer(r'<execute_tool>(.*?)</execute_tool>', text, re.DOTALL):
                block = et_match.group(1).strip()
                for line in block.splitlines():
                    line = line.strip()
                    # Match: print(fn_name(arg=val, ...)) or fn_name(arg=val)
                    m = re.match(r'(?:print\()?(\w+)\((.+?)\)\)?$', line, re.DOTALL)
                    if not m:
                        continue
                    fn_name, args_str = m.group(1), m.group(2)
                    if fn_name not in known_names:
                        continue
                    try:
                        import ast as _ast
                        # Parse kwargs from the argument string
                        expr = _ast.parse(f"_f({args_str})", mode='eval')
                        call = expr.body
                        params = {}
                        for kw in call.keywords:
                            if kw.arg:
                                try:
                                    params[kw.arg] = _ast.literal_eval(kw.value)
                                except Exception:
                                    params[kw.arg] = _ast.unparse(kw.value)
                        if params:
                            validated = self._validate_tool_call(fn_name, params, known_names)
                            if validated:
                                results.append(validated)
                                logger.info("Parsed <execute_tool> call: %s(%s)", fn_name, list(params.keys()))
                    except Exception:
                        pass
            if results:
                return results

        # Strategy 6: tool_call XML format (may contain multiple)
        # Pattern: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
        if "<tool_call>" in text:
            for tc_match in re.finditer(
                    r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
                obj = self._try_parse_json_obj(tc_match.group(1))
                if obj:
                    validated = self._validate_tool_call(
                        obj.get("name"), obj.get("arguments") or obj.get("parameters"), known_names)
                    if validated:
                        results.append(validated)
            if results:
                return results

        # Strategy 5b: Python function call inside ```python``` code blocks
        # Gemini outputs: ```python\nfn_name(kwarg=val, ...)\n```
        if '```' in text:
            import ast as _ast

            def _extract_calls_from_code(code: str) -> list[dict]:
                """Try ast.parse first; fallback to regex extraction for complex cases."""
                found: list[dict] = []
                # Attempt 1: full AST parse
                try:
                    tree = _ast.parse(code, mode='exec')
                    for node in _ast.walk(tree):
                        if not isinstance(node, _ast.Expr):
                            continue
                        call = node.value
                        if not isinstance(call, _ast.Call):
                            continue
                        fn_node = call.func
                        fn_name = fn_node.id if isinstance(fn_node, _ast.Name) else None
                        if not fn_name or fn_name not in known_names:
                            continue
                        params: dict = {}
                        for kw in call.keywords:
                            if kw.arg:
                                try:
                                    params[kw.arg] = _ast.literal_eval(kw.value)
                                except Exception:
                                    params[kw.arg] = _ast.unparse(kw.value)
                        if call.args:
                            tool_def = next((t for t in tool_defs if t['name'] == fn_name), None)
                            prop_names = list((tool_def or {}).get('input_schema', {}).get('properties', {}).keys())
                            for i, arg in enumerate(call.args):
                                key = prop_names[i] if i < len(prop_names) else f"arg{i}"
                                try:
                                    params[key] = _ast.literal_eval(arg)
                                except Exception:
                                    params[key] = _ast.unparse(arg)
                        if params:
                            v = self._validate_tool_call(fn_name, params, known_names)
                            if v:
                                found.append(v)
                                logger.info("Parsed Python code-block tool call: %s(%s)", fn_name, list(params.keys()))
                    return found
                except SyntaxError:
                    pass

                # Attempt 2: regex fallback for complex multiline strings
                # Find: fn_name(\n    key="val", ...\n)
                for fn_name in known_names:
                    if fn_name not in code:
                        continue
                    call_pattern = re.compile(
                        rf'{re.escape(fn_name)}\s*\((.*?)\n\)', re.DOTALL)
                    for m in call_pattern.finditer(code):
                        args_text = m.group(1)
                        params = {}
                        # Extract keyword args: key="value" or key='''value'''
                        for kw_m in re.finditer(
                                r'(\w+)\s*=\s*(\'\'\'.*?\'\'\'|""".*?"""|\'[^\']*\'|"[^"]*")',
                                args_text, re.DOTALL):
                            key = kw_m.group(1)
                            val_raw = kw_m.group(2)
                            try:
                                params[key] = _ast.literal_eval(val_raw)
                            except Exception:
                                # Strip quotes manually
                                if val_raw.startswith("'''") and val_raw.endswith("'''"):
                                    params[key] = val_raw[3:-3]
                                elif val_raw.startswith('"""') and val_raw.endswith('"""'):
                                    params[key] = val_raw[3:-3]
                                else:
                                    params[key] = val_raw.strip("'\"")
                        if params:
                            v = self._validate_tool_call(fn_name, params, known_names)
                            if v:
                                found.append(v)
                                logger.info("Parsed code-block (regex fallback): %s(%s)", fn_name, list(params.keys()))
                return found

            for code_match in re.finditer(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL):
                code = code_match.group(1).strip()
                if not code or not any(name in code for name in known_names):
                    continue
                results.extend(_extract_calls_from_code(code))
            if results:
                return results

        # Strategy 5c: bracket-style calls in plain text
        # Pattern: [exec_command("...")] or [write_file("/path", "...")]
        if "[" in text and "]" in text:
            import ast as _ast
            for m in re.finditer(r'\[(\w+)\((.*?)\)\]', text, re.DOTALL):
                fn_name = m.group(1)
                args_text = m.group(2).strip()
                if fn_name not in known_names or not args_text:
                    continue
                try:
                    expr = _ast.parse(f"_f({args_text})", mode='eval')
                    call = expr.body
                    params: dict = {}
                    tool_def = next((t for t in tool_defs if t['name'] == fn_name), None)
                    prop_names = list((tool_def or {}).get('input_schema', {}).get('properties', {}).keys())

                    for i, arg in enumerate(getattr(call, "args", []) or []):
                        key = prop_names[i] if i < len(prop_names) else f"arg{i}"
                        try:
                            params[key] = _ast.literal_eval(arg)
                        except Exception:
                            params[key] = _ast.unparse(arg)

                    for kw in getattr(call, "keywords", []) or []:
                        if kw.arg:
                            try:
                                params[kw.arg] = _ast.literal_eval(kw.value)
                            except Exception:
                                params[kw.arg] = _ast.unparse(kw.value)

                    if params:
                        validated = self._validate_tool_call(fn_name, params, known_names)
                        if validated:
                            results.append(validated)
                            logger.info("Parsed bracket-style tool call: %s(%s)", fn_name, list(params.keys()))
                except Exception:
                    continue
            if results:
                return results

        # Strategy 5d: plain function calls embedded in natural language
        # Examples:
        #   "вызываю exec_command(\"pwd\", timeout=30)"
        #   "Next run: write_file(\"/tmp/x\", \"...\")"
        for fn_name in known_names:
            marker = f"{fn_name}("
            search_from = 0
            while True:
                start = text.find(marker, search_from)
                if start < 0:
                    break
                # Avoid matching attribute access like obj.exec_command(...)
                if start > 0:
                    prev = text[start - 1]
                    if prev.isalnum() or prev in "._":
                        search_from = start + len(marker)
                        continue
                parsed_call = _extract_parenthesized_call_args(text, start + len(fn_name))
                if not parsed_call:
                    search_from = start + len(marker)
                    continue
                args_text, close_idx = parsed_call
                search_from = close_idx + 1
                args_text = args_text.strip()
                if not args_text:
                    continue
                try:
                    expr = ast.parse(f"_f({args_text})", mode='eval')
                    call = expr.body
                    params: dict = {}
                    tool_def = next((t for t in tool_defs if t["name"] == fn_name), None)
                    prop_names = list((tool_def or {}).get("input_schema", {}).get("properties", {}).keys())

                    for i, arg in enumerate(getattr(call, "args", []) or []):
                        key = prop_names[i] if i < len(prop_names) else f"arg{i}"
                        try:
                            params[key] = ast.literal_eval(arg)
                        except Exception:
                            params[key] = ast.unparse(arg)

                    for kw in getattr(call, "keywords", []) or []:
                        if kw.arg:
                            try:
                                params[kw.arg] = ast.literal_eval(kw.value)
                            except Exception:
                                params[kw.arg] = ast.unparse(kw.value)

                    if params:
                        validated = self._validate_tool_call(fn_name, params, known_names)
                        if validated:
                            results.append(validated)
                            logger.info("Parsed plain-text tool call: %s(%s)", fn_name, list(params.keys()))
                except Exception:
                    continue
        if results:
            return results

        # Strategy 1: standard JSON (double quotes) — single object
        obj = self._try_parse_json_obj(text)

        # Strategy 2: Python literal (single quotes — common with Ollama models)
        if obj is None:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    obj = parsed
            except Exception:
                pass

        # Strategy 3: extract from markdown code block
        if obj is None:
            m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
            if m:
                obj = self._try_parse_json_obj(m.group(1))
                if obj is None:
                    try:
                        parsed = ast.literal_eval(m.group(1))
                        if isinstance(parsed, dict):
                            obj = parsed
                    except Exception:
                        pass

        # Strategy 4: replace single quotes → double quotes
        if obj is None and text.startswith("{"):
            try:
                obj = json.loads(text.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 7: find JSON object anywhere in text (model may add preamble/commentary)
        if obj is None:
            m = re.search(r'(\{"name"\s*:\s*"[^"]+"\s*,\s*"(?:arguments|parameters)"\s*:\s*\{.*?\})\s*\}',
                          text, re.DOTALL)
            if m:
                obj = self._try_parse_json_obj(m.group(0))

        if obj is None:
            return []

        # Validate parsed object
        validated = self._validate_tool_call(
            obj.get("name"), obj.get("arguments") or obj.get("parameters"), known_names)
        if validated:
            return [validated]
        return []

    @staticmethod
    def _try_parse_json_obj(text: str) -> dict | None:
        """Try to parse text as JSON dict."""
        try:
            obj = json.loads(text.strip())
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    def _validate_tool_call(self, name, arguments, known_names: set) -> dict | None:
        """Validate and normalize a parsed tool call."""
        if not name or not isinstance(arguments, dict):
            return None

        # Must match a known tool
        if name not in known_names:
            # Fuzzy match for MCP tools: model may output 'transcribe_voice_file'
            # but real tool is 'mywhisper__transcribe_voice_file'
            matched = [kn for kn in known_names if kn.endswith(f"__{name}")]
            if not matched:
                # Also try partial match: model outputs 'web_search' → matches 'brave__web_search'
                matched = [kn for kn in known_names if name in kn]
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
        try:
            if self.memory._memory_exchange_daemon_enabled():
                await self.memory.enqueue_memory_exchange_intent(
                    user_input, user_id, response, source="turn")
            else:
                await self.memory.run_memory_exchange_cycle(user_input, user_id, response)
        except Exception as e:
            logger.warning("Memory exchange cycle failed: %s", e)

    # ══════════════════════════════════════════
    # AGENT COMMANDS (meta)
    # ══════════════════════════════════════════

    def handle_command(self, command: str, user_id: str = "default") -> str | None:
        """Handle special /commands. Returns response or None if not a command."""
        user_id = self.resolve_user_id(user_id)
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
            return self._handle_model_command(command, user_id=user_id)

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

    def _handle_model_command(self, command: str, user_id: str = "") -> str:
        """Handle /model command — show or switch models.

        /model                   — show current model + per-user override + cascade tiers
        /model <name>            — set per-user override for this conversation (persisted)
        /model reset             — clear per-user override, return to global default
        /model global <name>     — set global default model (affects all users)
        /model simple|medium|complex <name> — set cascade tier
        """
        from .providers import PROVIDER_MODELS
        from .conv_model import parse_model_command

        parts = command.strip().split(maxsplit=2)

        # ── /model (no args) → show info ──
        if len(parts) == 1:
            return self._model_status(PROVIDER_MODELS, user_id=user_id)

        arg1 = parts[1].strip().lower()

        # ── /model reset → clear per-user override ──
        if arg1 in ("reset", "clear", "default", "off"):
            if user_id and hasattr(self, '_conv_model') and self._conv_model:
                was_set = self._conv_model.clear(user_id)
                if was_set:
                    return f"✅ Model override cleared. Using global default: {self.default_model}"
            return f"No override active. Using: {self.default_model}"

        # ── /model global <name> → set global default ──
        if arg1 == "global":
            if len(parts) < 3:
                return f"Usage: /model global <model_name>"
            model_name = parts[2].strip()
            old_model = self.default_model
            self.default_model = model_name
            self.config.setdefault("agent", {})["default_model"] = model_name
            self._save_model_config()
            return f"✅ Global default: {old_model} → {model_name}"

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

        # ── /model <name> → set per-user override (persisted per conversation) ──
        model_name, err = parse_model_command(command)
        if err == "show":
            return self._model_status(PROVIDER_MODELS, user_id=user_id)
        if model_name == "__reset__":
            if user_id and hasattr(self, '_conv_model') and self._conv_model:
                self._conv_model.clear(user_id)
            return f"✅ Model override cleared. Using: {self.default_model}"
        if model_name:
            if user_id and hasattr(self, '_conv_model') and self._conv_model:
                self._conv_model.set(user_id, model_name, set_via="command")
                return (f"✅ Model override for this chat: {model_name}\n"
                        f"Use /model reset to return to global default ({self.default_model})")
            else:
                # No user context — fall back to global
                old = self.default_model
                self.default_model = model_name
                self.config.setdefault("agent", {})["default_model"] = model_name
                self._save_model_config()
                return f"✅ Default model: {old} → {model_name}"

        return self._model_status(PROVIDER_MODELS, user_id=user_id)

    def _model_status(self, provider_models: dict, user_id: str = "") -> str:
        """Format current model configuration for display."""
        provider_name = self.config.get("agent", {}).get("provider", "?")
        lines = [
            f"🤖 Provider: {provider_name}",
            f"📌 Default: {self.default_model}",
        ]
        # Show per-user override if active
        if user_id and hasattr(self, '_conv_model') and self._conv_model:
            override = self._conv_model.get(user_id)
            if override:
                lines.append(f"🎯 Your override: {override}  (use /model reset to clear)")

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
