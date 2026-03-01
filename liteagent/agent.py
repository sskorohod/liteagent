"""Core agent loop with cascade routing, prompt caching, and context compression."""

import asyncio
import json
import logging
import random
import time
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

from .config import get_soul_prompt
from .memory import MemorySystem
from .providers import create_provider, get_pricing, MODEL_PRICING
from .tools import ToolRegistry, register_builtin_tools
from .cognitive import register_cognitive_tools

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

        # Tools
        self.tools = ToolRegistry()
        builtin = config.get("tools", {}).get("builtin", ["read_file", "write_file", "exec_command"])
        register_builtin_tools(self.tools, enabled=builtin + ["memory_search"])
        register_cognitive_tools(self)

        # Wire memory_search to actual memory
        self._wire_memory_tool()

        # MCP servers (loaded lazily on first run)
        self._mcp_config = config.get("tools", {}).get("mcp_servers", {})
        self._mcp_loaded = False

        # Current user context (for tool closures)
        self._current_user_id: str = "default"

        # Background task tracking (prevents "Task destroyed" warnings)
        self._background_tasks: set[asyncio.Task] = set()

        # Soul prompt (cached across calls)
        self._soul_prompt = get_soul_prompt(config)

        # Feature flags (metacognition, evolution, synthesis)
        self._features = config.get("features", {})

        # Load synthesized tools if enabled
        if self._features.get("auto_tool_synthesis", {}).get("enabled"):
            from .synthesis import load_synthesized_tools, create_synthesize_meta_tool
            load_synthesized_tools(
                self.memory.db, self.tools,
                set(self._features["auto_tool_synthesis"].get(
                    "import_whitelist", [])) or None)
            create_synthesize_meta_tool(
                self.tools, self.memory.db,
                self._features["auto_tool_synthesis"])

        # RAG pipeline (optional)
        self._rag = None
        rag_cfg = config.get("rag", {})
        if rag_cfg.get("enabled", False):
            from .rag import RAGPipeline
            self._rag = RAGPipeline(
                self.memory.db,
                embedder=self.memory._embedder,
                config=rag_cfg)
            self._wire_rag_tool()

        # Auto-prune old memories on startup
        mem_cfg = config.get("memory", {})
        if mem_cfg.get("auto_prune", False):
            self.memory.prune_old_memories(
                days=mem_cfg.get("prune_days", 90),
                min_importance=mem_cfg.get("prune_min_importance", 0.3))

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
            return "\n---\n".join(lines)

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

    # ══════════════════════════════════════════
    # MAIN ENTRY POINT
    # ══════════════════════════════════════════

    async def _ensure_mcp_loaded(self):
        """Lazy-load MCP servers on first use."""
        if not self._mcp_loaded and self._mcp_config:
            await self.tools.load_mcp_servers(self._mcp_config)
            self._mcp_loaded = True

    async def reload_mcp(self):
        """Reload MCP servers from config."""
        await self.tools.close_mcp_servers()
        # Remove MCP tools from registry
        mcp_tools = [n for n in list(self.tools._tools) if "__" in n]
        for t in mcp_tools:
            del self.tools._tools[t]
            if t in self.tools._handlers:
                del self.tools._handlers[t]
        self._mcp_loaded = False
        await self._ensure_mcp_loaded()
        logger.info("MCP servers reloaded: %d servers",
                    len(self.tools.get_mcp_server_info()))

    async def run(self, user_input: str | list, user_id: str = "default") -> str:
        """Run agent on user input. Accepts str or list of content blocks (multimodal)."""
        self._current_user_id = user_id
        await self._ensure_mcp_loaded()

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

        # Build context (token-efficient)
        system_prompt = self._build_system_prompt(text_for_memory, user_id)
        messages = self.memory.get_compressed_history(user_id)
        messages.append({"role": "user", "content": content_for_api})

        # Select model (cascade routing)
        model = self._select_model(text_for_memory) if self.cascade_routing else self.default_model
        logger.info("Model: %s | User: %s | Input: %d chars", model, user_id, len(text_for_memory))
        # Semantic tool selection (reduces token usage with many MCP tools)
        if self.memory._embedder and len(self.tools._tools) > 8:
            tool_defs = self.tools.get_relevant_definitions(
                user_input, top_k=8, embedder=self.memory._embedder)
        else:
            tool_defs = self.tools.get_definitions()

        # Track tool calls for skill crystallization
        _tool_calls_log = []

        # Agent loop
        for iteration in range(self.max_iterations):
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
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                # Log tool calls for skill crystallization
                if self._features.get("skill_crystallization", {}).get("enabled"):
                    for block in response.content:
                        if hasattr(block, 'type') and block.type == "tool_use":
                            _tool_calls_log.append({
                                "name": block.name,
                                "input": block.input,
                            })
            else:
                # Done — extract text
                text = self._extract_text(response)

                # Save conversation to memory
                self.memory.add_message(user_id, "user", text_for_memory)
                self.memory.add_message(user_id, "assistant", text)

                # ── Feature hooks (post-response) ──
                text = await self._post_response_hooks(
                    text, text_for_memory, user_id, model,
                    system_prompt, tool_defs, messages, _tool_calls_log)

                # Background: extract knowledge (non-blocking)
                task = asyncio.create_task(
                    self._safe_extract(text_for_memory, text, user_id)
                )
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

                return text

        return "⚠️ Max iterations reached. Try a simpler request."

    async def stream(self, user_input: str, user_id: str = "default") -> AsyncGenerator[str, None]:
        """Stream agent response token by token."""
        self._current_user_id = user_id
        await self._ensure_mcp_loaded()

        if self.memory.get_today_cost() >= self.budget_daily:
            yield f"⚠️ Daily budget (${self.budget_daily:.2f}) reached."
            return

        system_prompt = self._build_system_prompt(user_input, user_id)
        messages = self.memory.get_compressed_history(user_id)
        messages.append({"role": "user", "content": user_input})
        model = self._select_model(user_input) if self.cascade_routing else self.default_model
        logger.info("Stream | Model: %s | User: %s", model, user_id)
        tool_defs = self.tools.get_definitions()

        full_text = ""

        for iteration in range(self.max_iterations):
            # Use streaming API for real token-by-token output
            async for delta in self.provider.stream(
                model=model,
                max_tokens=4096,
                system=system_prompt,
                tools=tool_defs,
                messages=messages,
            ):
                full_text += delta
                yield delta

            response = self.provider._last_stream_response

            cost = self._calculate_cost(model, response.usage)
            self.memory.track_usage(user_id, model, response.usage, cost)

            if response.stop_reason == "tool_use":
                # Show tool usage indicator
                for block in response.content:
                    if block.type == "tool_use":
                        yield f"\n🔧 Using {block.name}...\n"
                tool_results = await self.tools.execute(response.content)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                full_text = ""  # Reset for next iteration
            else:
                self.memory.add_message(user_id, "user", user_input)
                self.memory.add_message(user_id, "assistant", full_text)
                task = asyncio.create_task(self._safe_extract(user_input, full_text, user_id))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                return

        yield "\n⚠️ Max iterations reached."

    # ══════════════════════════════════════════
    # API CALL WITH RETRY
    # ══════════════════════════════════════════

    async def _call_api(self, **kwargs) -> "LLMResponse":
        """Call LLM provider with retry + exponential backoff."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self.provider.complete(**kwargs)
            except Exception as e:
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

    def _build_system_prompt(self, user_input: str, user_id: str) -> str | list[dict]:
        """Build system prompt with memories + feature injections."""
        # Recall relevant memories
        memories = self.memory.recall(user_input, user_id, top_k=5)
        memory_section = ""
        if memories:
            memory_lines = [f"- {m['content']}" for m in memories if m['score'] > 0.1]
            if memory_lines:
                memory_section = "\n\n## What you know about this user:\n" + "\n".join(memory_lines)

        # Feature injections (dynamic, not cached)
        feature_section = self._build_feature_section(user_input, user_id)

        dynamic_text = memory_section + feature_section

        if self.prompt_caching:
            return [
                {
                    "type": "text",
                    "text": self._soul_prompt,
                    "cache_control": {"type": "ephemeral"},  # Static part cached
                },
                {
                    "type": "text",
                    "text": dynamic_text,  # Dynamic — not cached
                },
            ]
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
    # CASCADE MODEL ROUTING
    # ══════════════════════════════════════════

    def _select_model(self, user_input: str) -> str:
        """Route to Haiku/Sonnet/Opus based on query complexity."""
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

        # Select model
        if score >= 3:
            return self.models.get("complex", self.default_model)
        elif score >= 1:
            return self.models.get("medium", self.default_model)
        else:
            return self.models.get("simple", self.default_model)

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

    async def _post_response_hooks(self, text: str, user_input: str,
                                    user_id: str, model: str,
                                    system_prompt, tool_defs, messages,
                                    tool_calls_log: list) -> str:
        """Run post-response feature hooks. Returns potentially modified text."""
        _confidence = None
        _success = 1

        # Confidence Gate
        cg_cfg = self._features.get("confidence_gate", {})
        if cg_cfg.get("enabled"):
            try:
                from .metacognition import assess_confidence
                assessment = await assess_confidence(
                    self.provider, user_input, text,
                    self.config.get("memory", {}))
                _confidence = assessment.get("confidence", 10)
                threshold = cg_cfg.get("threshold", 6)
                if _confidence < threshold:
                    action = assessment.get("action", "admit")
                    if action == "escalate" and cg_cfg.get("escalate_to_model", True):
                        better = self.models.get("complex", self.default_model)
                        if better != model:
                            text = await self._escalated_run(
                                better, system_prompt, tool_defs, messages)
                    elif action == "admit":
                        text += ("\n\n⚠️ I'm not fully confident in this "
                                 "answer. Please verify independently.")
            except Exception as e:
                logger.debug("Confidence gate error: %s", e)

        # Style adaptation (update profile from user input)
        if self._features.get("style_adaptation", {}).get("enabled"):
            try:
                from .evolution import analyze_style, update_style_profile
                style = analyze_style(user_input)
                update_style_profile(
                    self.memory.db, user_id, style,
                    self._features["style_adaptation"].get("ema_alpha", 0.3))
            except Exception as e:
                logger.debug("Style adaptation error: %s", e)

        # Skill crystallization detection
        sk_cfg = self._features.get("skill_crystallization", {})
        if sk_cfg.get("enabled"):
            min_calls = sk_cfg.get("min_tool_calls", 3)
            if len(tool_calls_log) >= min_calls:
                try:
                    from .synthesis import detect_skill, store_skill
                    skill = detect_skill(tool_calls_log, user_input, min_calls)
                    if skill:
                        store_skill(self.memory.db, skill, user_id)
                except Exception as e:
                    logger.debug("Skill crystallization error: %s", e)

        # Log interaction (for counterfactual replay + proactive)
        if any(self._features.get(f, {}).get("enabled")
               for f in ("counterfactual_replay", "proactive_agent",
                          "confidence_gate")):
            try:
                from .metacognition import log_interaction
                log_interaction(
                    self.memory.db, user_id, user_input, text,
                    tool_calls_log, _success, _confidence, model)
            except Exception as e:
                logger.debug("Interaction logging error: %s", e)

        return text

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

        elif cmd == "/help":
            help_text = (
                "Commands:\n"
                "  /memories   — Show stored memories\n"
                "  /usage      — Show token usage and costs\n"
                "  /clear      — Clear conversation history\n"
                "  /forget X   — Forget memories matching X\n"
            )
            if self._rag:
                help_text += (
                    "  /ingest X   — Ingest file or directory into RAG\n"
                    "  /documents  — List ingested documents\n"
                )
            help_text += "  /help       — This message"
            return help_text

        return None  # Not a command
