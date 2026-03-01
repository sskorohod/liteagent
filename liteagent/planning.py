"""Internal monologue: chain-of-thought planning before agent execution."""

import json
import logging

logger = logging.getLogger(__name__)


async def generate_plan(provider, user_input: str, memories: list,
                        tools: list, config: dict) -> dict | None:
    """Generate an execution plan using a cheap model before the main agent loop.

    Returns plan dict with steps, complexity, tools_needed, estimated_iterations.
    Returns None if the request is simple (to skip planning overhead).
    """
    skip_simple = config.get("skip_simple", True)
    model = config.get("planning_model", "claude-haiku-4-5-20251001")

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

        # Skip simple requests if configured
        if skip_simple and plan.get("complexity") == "simple":
            logger.debug("Planning: skipping simple request")
            return None

        logger.info("Planning: complexity=%s, steps=%d, tools=%s",
                     plan.get("complexity"), len(plan.get("steps", [])),
                     plan.get("tools_needed", []))
        return plan

    except Exception as e:
        logger.debug("Planning failed (non-critical): %s", e)
        return None


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

    return "\n".join(lines)


async def reflect_on_progress(provider, plan: dict, completed_tools: list,
                               config: dict) -> str | None:
    """Mid-loop reflection: check if the plan needs adjustment.

    Called after every N tool executions.
    Returns adjustment note or None if no change needed.
    """
    model = config.get("planning_model", "claude-haiku-4-5-20251001")

    completed_names = [tc.get("name", "") for tc in completed_tools]
    plan_steps = plan.get("steps", [])

    prompt = (
        "You are monitoring an AI agent's execution.\n\n"
        f"Original plan steps: {json.dumps(plan_steps)}\n"
        f"Tools executed so far: {', '.join(completed_names)}\n\n"
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
