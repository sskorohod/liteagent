"""Internal monologue: chain-of-thought planning before agent execution."""
from __future__ import annotations

import json
import logging
import re


def _safe_parse_llm_json(text: str, fallback):
    """Parse JSON from LLM output, tolerating control chars, truncation and wrong root type."""
    import re as _re, json as _json
    text = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    try:
        return _json.loads(text)
    except Exception:
        pass
    for opener, closer in (('{', '}'), ('[', ']')):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return _json.loads(text[start:i + 1])
                    except Exception:
                        break
    return fallback


logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# PLANNING MODEL AUTO-DETECTION
# ══════════════════════════════════════════

# Cheapest/fastest model per provider (used for planning to minimise cost/latency)
_CHEAPEST_MODEL = {
    "AnthropicProvider": "claude-haiku-4-5-20251001",
    "OpenAIProvider":    "gpt-4.1-nano",
    "GeminiProvider":    "gemini-2.0-flash",
}


def resolve_planning_model(provider, config: dict) -> str:
    """Pick the cheapest model for planning based on current provider.

    Priority:
      1. Explicit ``planning_model`` in config (unless "auto")
      2. Provider-specific cheapest model
      3. First discovered Ollama model / default_model for Ollama
      4. Hardcoded fallback
    """
    explicit = config.get("planning_model", "auto")
    if explicit and explicit != "auto":
        return explicit

    provider_cls = provider.__class__.__name__

    # For Ollama use whatever is locally available
    if provider_cls == "OllamaProvider":
        try:
            from .providers import PROVIDER_MODELS
            ollama_models = PROVIDER_MODELS.get("ollama", [])
            if ollama_models:
                return ollama_models[0]
        except Exception:
            pass
        # Fallback to default_model from agent config (passed through)
        default = config.get("_default_model")
        if default:
            return default
        return "qwen2.5:latest"  # reasonable Ollama fallback

    return _CHEAPEST_MODEL.get(provider_cls, "claude-haiku-4-5-20251001")



# ══════════════════════════════════════════
# INTENT CLASSIFICATION (zero LLM cost)
# ══════════════════════════════════════════

# Pattern sets for each intent type (EN + RU).
# Evaluated in priority order; first match wins.
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("search", [
        r"найди в интернете", r"загугли", r"поищи в сети", r"web search",
        r"search (the web|online|internet)", r"look up online",
    ]),
    ("creative", [
        r"(напиши|сочини|придумай|сгенерируй).{0,30}(текст|стихи?|историю|рассказ|сценарий|рекламу|слоган|эссе|пост)",
        r"(write|create|generate|compose).{0,30}(poem|story|essay|slogan|ad|script|creative|fiction)",
        r"придумай (идею|название|концепцию)",
        r"(creative writing|roleplay|generate story)",
    ]),
    ("command", [
        r"^(сделай|создай|напиши|переведи|вычисли|посчитай|конвертируй|запусти|выполни|открой|закрой|удали|скопируй|переименуй)",
        r"^(do|make|create|write|translate|calculate|compute|run|execute|open|close|delete|copy|rename)",
        r"(код|code|скрипт|script|программ).{0,30}(написать|создать|write|create|generate)",
        r"(fix|исправь|починить|отладь|debug)",
        r"помог.{0,10}(написать|создать|сделать)",
    ]),
    ("question", [
        r"^(что|как|почему|зачем|когда|где|кто|какой|какая|какие|сколько)",
        r"^(what|how|why|when|where|who|which|whose|whom)",
        r"\?$",
        r"(объясни|расскажи|опиши|explain|describe|tell me|what is|what are)",
    ]),
    ("chat", [
        r"^(привет|здравствуй|добрый|hi|hello|hey|good morning|good evening)",
        r"^(спасибо|благодарю|thanks|thank you)",
        r"^(пока|до свидания|goodbye|bye)",
        r"^(как дела|как ты|how are you)",
    ]),
]

_COMPILED: list[tuple[str, list]] | None = None


def classify_intent(text: str) -> str:
    """Classify user intent using fast regex patterns (no LLM call).

    Returns one of: "search", "creative", "command", "question", "chat", "unknown".
    """
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = [
            (intent, [re.compile(p, re.IGNORECASE) for p in patterns])
            for intent, patterns in _INTENT_PATTERNS
        ]

    t = text.strip()
    for intent, compiled_patterns in _COMPILED:
        for pat in compiled_patterns:
            if pat.search(t):
                return intent
    return "unknown"


# ══════════════════════════════════════════
# PLAN GENERATION
# ══════════════════════════════════════════

async def generate_plan(provider, user_input: str, memories: list,
                        tools: list, config: dict) -> dict | None:
    """Generate an execution plan using a cheap model before the main agent loop.

    Returns plan dict with steps, complexity, tools_needed, estimated_iterations.
    Returns None if the request is simple (to skip planning overhead).
    """
    skip_simple = config.get("skip_simple", True)
    model = resolve_planning_model(provider, config)

    # Fast zero-cost intent classification
    intent = classify_intent(user_input)

    # Build context
    tool_names = [t.get("name", "") for t in tools] if tools else []
    memory_context = ""
    if memories:
        memory_lines = [m.get("content", "") for m in memories[:3] if m.get("score", 0) > 0.1]
        if memory_lines:
            memory_context = "\nKnown context: " + "; ".join(memory_lines)

    intent_hint = f"\nDetected intent type: {intent}" if intent != "unknown" else ""

    prompt = (
        "You are a planning module. Analyze the user's request and produce a brief execution plan.\n\n"
        f"User request: {user_input[:500]}\n"
        f"Available tools: {', '.join(tool_names[:15])}\n"
        f"{memory_context}{intent_hint}\n\n"
        'Return ONLY valid JSON:\n'
        '{"steps": ["step1", "step2", ...], '
        '"complexity": "simple" or "medium" or "complex", '
        '"tools_needed": ["tool1", ...], '
        '"estimated_iterations": N, '
        '"assumptions": ["assumption 1", ...], '
        '"verification_steps": ["check 1", ...], '
        '"ask_user": false, '
        '"ask_user_reason": ""}\n\n'
        "Rules:\n"
        "- steps: 1-5 brief action steps\n"
        "- complexity: simple (greeting, factual Q&A), medium (code/analysis), complex (multi-step research/creation)\n"
        "- tools_needed: which tools from the available list are needed (empty array if none)\n"
        "- estimated_iterations: how many LLM turns needed (1-10)\n"
        "- assumptions: 0-3 short working assumptions you can safely make to keep moving\n"
        "- verification_steps: 0-3 checks to confirm the result before finalizing\n"
        "- ask_user: false by default; set true only if blocked by missing credentials/access, an irreversible destructive choice, or conflicting requirements you cannot resolve from context\n"
        "- ask_user_reason: short reason, only when ask_user=true\n"
        "- Keep it concise. Max 5 steps."
    )

    try:
        result = await provider.complete(
            model=model,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        plan = _safe_parse_llm_json(text, {})
        if not isinstance(plan, dict):
            return None

        # Validate structure
        if not isinstance(plan.get("steps"), list):
            return None
        if "complexity" not in plan:
            plan["complexity"] = "medium"
        if not isinstance(plan.get("tools_needed"), list):
            plan["tools_needed"] = []
        if not isinstance(plan.get("assumptions"), list):
            plan["assumptions"] = []
        else:
            plan["assumptions"] = [str(item).strip() for item in plan["assumptions"][:3] if str(item).strip()]
        if not isinstance(plan.get("verification_steps"), list):
            plan["verification_steps"] = []
        else:
            plan["verification_steps"] = [str(item).strip() for item in plan["verification_steps"][:3] if str(item).strip()]
        plan["ask_user"] = bool(plan.get("ask_user", False))
        plan["ask_user_reason"] = str(plan.get("ask_user_reason", "") or "").strip()[:240]

        # Attach pre-computed intent (zero-cost, overrides any LLM-inferred value)
        plan["intent"] = intent

        # Clamp estimated_iterations to sane range
        est = plan.get("estimated_iterations")
        if est is not None:
            try:
                plan["estimated_iterations"] = max(1, min(int(est), 10))
            except (ValueError, TypeError):
                plan["estimated_iterations"] = 3

        # Skip simple requests if configured
        if skip_simple and plan.get("complexity") == "simple":
            logger.debug("Planning: skipping simple request")
            return None

        logger.info("Planning: complexity=%s, steps=%d, tools=%s, est_iter=%s",
                     plan.get("complexity"), len(plan.get("steps", [])),
                     plan.get("tools_needed", []),
                     plan.get("estimated_iterations"))
        return plan

    except Exception as e:
        logger.debug("Planning failed (non-critical): %s", e)
        return None


# ══════════════════════════════════════════
# FORMAT FOR SYSTEM PROMPT
# ══════════════════════════════════════════

_INTENT_STYLE_HINTS: dict[str, str] = {
    "question":  "Give a clear, direct answer. Cite sources or reasoning when helpful.",
    "command":   "Execute the task step by step. Confirm completion concisely.",
    "creative":  "Be imaginative and original. Prioritise quality over brevity.",
    "search":    "Retrieve up-to-date information using available search tools.",
    "chat":      "Keep the tone conversational and friendly.",
}


def format_plan_for_prompt(plan: dict) -> str:
    """Convert a plan dict into a system prompt section."""
    steps = plan.get("steps", [])
    tools = plan.get("tools_needed", [])
    assumptions = plan.get("assumptions", [])
    verification_steps = plan.get("verification_steps", [])
    ask_user = bool(plan.get("ask_user"))
    ask_user_reason = str(plan.get("ask_user_reason", "") or "").strip()

    lines = ["\n\n## Your execution plan:"]
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step}")

    if tools:
        lines.append(f"\nTools to use: {', '.join(tools)}")

    est = plan.get("estimated_iterations")
    if est:
        lines.append(f"Estimated iterations: {est}")
    if assumptions:
        lines.append("\nWorking assumptions:")
        lines.extend(f"- {item}" for item in assumptions)
    if verification_steps:
        lines.append("\nCritical verification before final answer:")
        lines.extend(f"- {item}" for item in verification_steps)

    intent = plan.get("intent", "unknown")
    style_hint = _INTENT_STYLE_HINTS.get(intent)
    if style_hint:
        lines.append(f"Response style ({intent}): {style_hint}")

    lines.append(
        "\nFollow this plan. "
        "⚡ AUTONOMOUS EXECUTION REQUIRED: First do a silent critical review of the plan, "
        "assumptions, available context, memory, and tool options. Then execute ALL steps above "
        "sequentially without stopping. Prefer the smallest reversible action that creates evidence. "
        "Re-check your work with at least one verification step before you finalize. "
        "Do NOT ask the user for routine confirmation between steps. "
        "Do NOT send intermediate progress messages. Call tools one after another until the ENTIRE "
        "task is complete. Only ask the user if you are blocked by missing credentials/access, an "
        "irreversible destructive decision, or conflicting requirements you cannot resolve from context."
    )
    if ask_user and ask_user_reason:
        lines.append(f"Potential blocker if still unresolved after inspection: {ask_user_reason}")

    return "\n".join(lines)


# ══════════════════════════════════════════
# MID-LOOP REFLECTION
# ══════════════════════════════════════════

async def reflect_on_progress(provider, plan: dict, completed_tools: list,
                               tool_results_summary: list | None,
                               config: dict) -> str | None:
    """Mid-loop reflection: check if the plan needs adjustment.

    Called after every N tool executions.
    ``tool_results_summary`` is a list of truncated result strings (≤200 chars each).
    Returns adjustment note or None if no change needed.
    """
    model = resolve_planning_model(provider, config)

    plan_steps = plan.get("steps", [])

    # Build tool execution summary with results
    exec_lines = []
    for i, tc in enumerate(completed_tools):
        name = tc.get("name", "?")
        result_preview = ""
        if tool_results_summary and i < len(tool_results_summary):
            result_preview = f" → {tool_results_summary[i][:200]}"
        exec_lines.append(f"  - {name}{result_preview}")
    exec_summary = "\n".join(exec_lines) if exec_lines else "(none)"

    prompt = (
        "You are monitoring an AI agent's execution.\n\n"
        f"Original plan steps: {json.dumps(plan_steps)}\n"
        f"Tools executed so far (with results):\n{exec_summary}\n\n"
        "Should the agent adjust its approach? "
        "If yes, provide a brief adjustment note (1-2 sentences). "
        "If no adjustment needed, respond with exactly: NO_CHANGE"
    )

    try:
        result = await provider.complete(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text.strip()

        if text == "NO_CHANGE" or "no change" in text.lower():
            return None

        logger.debug("Planning reflection: %s", text[:100])
        return text

    except Exception as e:
        logger.debug("Planning reflection failed (non-critical): %s", e)
        return None


# ══════════════════════════════════════════
# STEP COMPLETION TRACKING
# ══════════════════════════════════════════

_STEP_KEYWORDS = re.compile(r'[a-zA-Zа-яА-ЯёЁ]{3,}')


def track_step_completion(plan: dict, tool_calls: list,
                          results_summary: list | None = None) -> dict:
    """Track which plan steps have likely been addressed (heuristic, zero-cost).

    Uses keyword overlap between step descriptions and tool call names/inputs/results.
    Returns ``{total, completed_count, steps: [{text, status}]}``.
    """
    steps = plan.get("steps", [])
    if not steps:
        return {"total": 0, "completed_count": 0, "steps": []}

    # Build a set of words from all tool activity
    activity_words: set[str] = set()
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        activity_words.add(tool_name.lower())
        # Also split tool name on underscores (read_file → {read, file})
        for part in tool_name.split("_"):
            if len(part) >= 3:
                activity_words.add(part.lower())
        inp = tc.get("input", {})
        if isinstance(inp, dict):
            for v in inp.values():
                for w in _STEP_KEYWORDS.findall(str(v)):
                    activity_words.add(w.lower())
    if results_summary:
        for rs in results_summary:
            for w in _STEP_KEYWORDS.findall(str(rs)):
                activity_words.add(w.lower())

    tracked = []
    completed = 0
    for step_text in steps:
        step_words = {w.lower() for w in _STEP_KEYWORDS.findall(step_text)}
        if not step_words:
            tracked.append({"text": step_text, "status": "pending"})
            continue
        overlap = len(step_words & activity_words) / len(step_words)
        status = "done" if overlap >= 0.4 else "pending"
        if status == "done":
            completed += 1
        tracked.append({"text": step_text, "status": status})

    return {"total": len(steps), "completed_count": completed, "steps": tracked}
