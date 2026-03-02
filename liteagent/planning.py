"""Internal monologue: chain-of-thought planning before agent execution."""
from __future__ import annotations

import json
import logging
import re

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

    # Build context
    tool_names = [t.get("name", "") for t in tools] if tools else []
    memory_context = ""
    if memories:
        memory_lines = [m.get("content", "") for m in memories[:3] if m.get("score", 0) > 0.1]
        if memory_lines:
            memory_context = "\nKnown context: " + "; ".join(memory_lines)

    prompt = (
        "You are a planning module. Analyze the user's request and produce a brief execution plan.\n\n"
        f"User request: {user_input[:500]}\n"
        f"Available tools: {', '.join(tool_names[:15])}\n"
        f"{memory_context}\n\n"
        'Return ONLY valid JSON:\n'
        '{"steps": ["step1", "step2", ...], '
        '"complexity": "simple" or "medium" or "complex", '
        '"tools_needed": ["tool1", ...], '
        '"estimated_iterations": N}\n\n'
        "Rules:\n"
        "- steps: 1-5 brief action steps\n"
        "- complexity: simple (greeting, factual Q&A), medium (code/analysis), complex (multi-step research/creation)\n"
        "- tools_needed: which tools from the available list are needed (empty array if none)\n"
        "- estimated_iterations: how many LLM turns needed (1-10)\n"
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

        plan = json.loads(text)

        # Validate structure
        if not isinstance(plan.get("steps"), list):
            return None
        if "complexity" not in plan:
            plan["complexity"] = "medium"

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

def format_plan_for_prompt(plan: dict) -> str:
    """Convert a plan dict into a system prompt section."""
    steps = plan.get("steps", [])
    tools = plan.get("tools_needed", [])

    lines = ["\n\n## Your execution plan:"]
    for i, step in enumerate(steps, 1):
        lines.append(f"{i}. {step}")

    if tools:
        lines.append(f"\nTools to use: {', '.join(tools)}")

    est = plan.get("estimated_iterations")
    if est:
        lines.append(f"Estimated iterations: {est}")

    lines.append("\nFollow this plan step by step. Use only the listed tools unless "
                 "absolutely necessary.")

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
