"""Tool synthesis and skill crystallization."""

import ast
import json
import logging
from collections import Counter
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Safe import whitelist
DEFAULT_IMPORT_WHITELIST = {
    "math", "json", "re", "datetime", "collections", "itertools",
    "functools", "hashlib", "urllib", "os", "pathlib",
    "string", "textwrap", "statistics", "decimal", "fractions",
}

# Blocked builtins that must never be called
BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "breakpoint",
}

# Blocked attribute calls (subprocess/os execution)
BLOCKED_ATTRS = {"system", "popen", "call", "check_output", "run", "Popen"}
BLOCKED_MODULES = {"subprocess", "shutil"}


# ══════════════════════════════════════════
# AUTO TOOL SYNTHESIS
# ══════════════════════════════════════════

def validate_tool_source(source: str,
                         import_whitelist: set | None = None) -> tuple[bool, str]:
    """Validate synthesized tool source for safety.

    Returns (ok, error_msg).
    """
    whitelist = import_whitelist or DEFAULT_IMPORT_WHITELIST

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Check imports against whitelist
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_root = alias.name.split(".")[0]
                if mod_root in BLOCKED_MODULES:
                    return False, f"Blocked module: {alias.name}"
                if mod_root not in whitelist:
                    return False, f"Import not whitelisted: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_root = node.module.split(".")[0]
                if mod_root in BLOCKED_MODULES:
                    return False, f"Blocked module: {node.module}"
                if mod_root not in whitelist:
                    return False, f"Import not whitelisted: {node.module}"
        # Check for dangerous builtin calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_BUILTINS:
                    return False, f"Blocked builtin: {node.func.id}"
                if node.func.id == "open":
                    return False, "File I/O blocked: open()"
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_BUILTINS:
                    return False, f"Blocked call: {node.func.attr}"
                if node.func.attr in BLOCKED_ATTRS:
                    return False, f"Blocked method: {node.func.attr}"

    # Must define exactly one function
    funcs = [n for n in ast.iter_child_nodes(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(funcs) != 1:
        return False, f"Must define exactly 1 function, found {len(funcs)}"

    return True, ""


def register_synthesized_tool(registry, name: str, source: str,
                              description: str, schema: dict):
    """Compile and register a synthesized tool into the registry."""
    namespace = {}
    exec(compile(ast.parse(source), f"<synth:{name}>", "exec"), namespace)  # noqa: S102

    # Find the defined function
    func_name = None
    for key, val in namespace.items():
        if callable(val) and not key.startswith("_"):
            func_name = key
            break

    if not func_name:
        raise ValueError("No callable function found in source")

    handler = namespace[func_name]
    registry._tools[name] = {
        "name": name,
        "description": description,
        "input_schema": schema,
    }
    registry._handlers[name] = handler
    logger.info("Registered synthesized tool: %s", name)


def load_synthesized_tools(db, registry, import_whitelist: set | None = None):
    """Load approved synthesized tools from DB on startup."""
    rows = db.execute(
        "SELECT name, description, source_code, parameters_json "
        "FROM synthesized_tools WHERE approved = 1").fetchall()

    for name, desc, source, params_json in rows:
        ok, err = validate_tool_source(source, import_whitelist)
        if ok:
            schema = json.loads(params_json) if params_json else {
                "type": "object", "properties": {}}
            try:
                register_synthesized_tool(registry, name, source, desc, schema)
            except Exception as e:
                logger.error("Failed to load synth tool '%s': %s", name, e)
        else:
            logger.warning("Skipping invalid synth tool '%s': %s", name, err)


def create_synthesize_meta_tool(registry, db, config: dict):
    """Register the 'synthesize_tool' meta-tool that creates new tools."""
    whitelist = set(config.get("import_whitelist", DEFAULT_IMPORT_WHITELIST))
    auto_approve = config.get("auto_approve", False)

    async def synthesize_tool(name: str, description: str,
                              source_code: str,
                              parameters_json: str = "{}") -> str:
        """Create a new Python tool at runtime.

        name: Tool name (snake_case)
        description: What the tool does
        source_code: Python source defining exactly one function
        parameters_json: JSON schema for parameters
        """
        ok, err = validate_tool_source(source_code, whitelist)
        if not ok:
            return f"Validation failed: {err}"

        try:
            schema = json.loads(parameters_json)
        except json.JSONDecodeError:
            return "Invalid parameters_json"

        approved = 1 if auto_approve else 0
        db.execute(
            """INSERT OR REPLACE INTO synthesized_tools
               (name, description, source_code, parameters_json, approved,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (name, description, source_code, parameters_json,
             approved, datetime.now().isoformat()))
        db.commit()

        if approved:
            try:
                register_synthesized_tool(
                    registry, name, source_code, description, schema)
                return f"Tool '{name}' created and registered."
            except Exception as e:
                return f"Tool created but registration failed: {e}"
        return f"Tool '{name}' created (pending approval)."

    registry._tools["synthesize_tool"] = {
        "name": "synthesize_tool",
        "description": (
            "Create a new Python tool at runtime. The source must define "
            "exactly one function. Imports are restricted to safe modules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "Tool name in snake_case"},
                "description": {"type": "string",
                                "description": "What the tool does"},
                "source_code": {"type": "string",
                                "description": "Python source defining one function"},
                "parameters_json": {
                    "type": "string",
                    "description": "JSON schema for parameters",
                    "default": "{}"},
            },
            "required": ["name", "description", "source_code"],
        },
    }
    registry._handlers["synthesize_tool"] = synthesize_tool
    logger.info("Registered meta-tool: synthesize_tool")


# ══════════════════════════════════════════
# SKILL CRYSTALLIZATION
# ══════════════════════════════════════════

def detect_skill(tool_calls: list, user_input: str,
                 min_calls: int = 3) -> dict | None:
    """Detect if a conversation contains a reusable skill pattern.

    Returns skill dict or None.
    """
    if len(tool_calls) < min_calls:
        return None

    steps = []
    for tc in tool_calls:
        step = {
            "tool": tc.get("name", ""),
            "params": _templatize_params(tc.get("input", {})),
        }
        steps.append(step)

    # Generate skill name from input
    words = user_input.split()[:5]
    name = "_".join(w.lower() for w in words if w.isalnum())[:50]

    return {
        "name": name or "unnamed_skill",
        "description": f"Skill from: {user_input[:100]}",
        "steps": steps,
    }


def _templatize_params(params: dict) -> dict:
    """Convert concrete params to templates with {{placeholders}}."""
    template = {}
    for k, v in params.items():
        if isinstance(v, str) and len(v) > 20:
            template[k] = f"{{{{{k}}}}}"  # {{param_name}}
        else:
            template[k] = v
    return template


def store_skill(db, skill: dict, user_id: str):
    """Store a crystallized skill."""
    db.execute(
        """INSERT INTO skills (name, description, steps_json,
           trigger_pattern, created_at) VALUES (?, ?, ?, ?, ?)""",
        (skill["name"], skill["description"], json.dumps(skill["steps"]),
         skill["name"], datetime.now().isoformat()))
    db.commit()
    logger.info("Crystallized skill: %s (%d steps)",
                skill["name"], len(skill["steps"]))


def find_matching_skills(db, query: str, top_k: int = 3) -> list[dict]:
    """Find skills matching a query by keyword overlap."""
    rows = db.execute(
        "SELECT id, name, description, steps_json, use_count FROM skills"
    ).fetchall()

    if not rows:
        return []

    query_lower = query.lower()
    query_words = set(query_lower.split())
    scored = []

    for row_id, name, desc, steps_json, use_count in rows:
        combined = f"{name} {desc}".lower()
        match_words = set(combined.split())
        overlap = len(query_words & match_words) / max(len(query_words), 1)
        score = overlap + use_count * 0.01
        if score > 0.1:
            scored.append({
                "id": row_id, "name": name, "description": desc,
                "steps": json.loads(steps_json), "score": score,
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def format_skill_suggestion(skills: list) -> str:
    """Format skill suggestions for system prompt injection."""
    if not skills:
        return ""
    lines = ["\n\n## Available skills (reusable workflows):"]
    for s in skills:
        tools = " -> ".join(step["tool"] for step in s["steps"])
        lines.append(f"- **{s['name']}**: {s['description']} [{tools}]")
    return "\n".join(lines)


# ══════════════════════════════════════════
# CROSS-SESSION PATTERN DETECTION
# ══════════════════════════════════════════

def detect_repeated_patterns(db, user_id: str,
                             min_occurrences: int = 3,
                             lookback_days: int = 30) -> list[dict]:
    """Detect recurring tool-call sequences across sessions.

    Queries interaction_log for recent entries, extracts tool sequences,
    and finds patterns that appear >= min_occurrences times.

    Returns list of {"sequence": [str], "count": int, "example_params": dict}.
    """
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    rows = db.execute(
        """SELECT tool_calls_json FROM interaction_log
           WHERE user_id = ? AND tool_calls_json IS NOT NULL
                 AND tool_calls_json != '[]'
                 AND created_at >= ?
           ORDER BY created_at DESC LIMIT 200""",
        (user_id, cutoff)).fetchall()

    if not rows:
        return []

    # Extract all tool sequences
    all_sequences = []
    all_params: dict[str, dict] = {}

    for (tc_json,) in rows:
        try:
            calls = json.loads(tc_json)
            if not calls or not isinstance(calls, list):
                continue
            tool_names = [c.get("name", "") for c in calls if c.get("name")]
            if len(tool_names) >= 2:
                all_sequences.append(tuple(tool_names))
                # Store example params for first occurrence
                key = "->".join(tool_names)
                if key not in all_params:
                    all_params[key] = calls[0].get("input", {}) if calls else {}
        except (json.JSONDecodeError, TypeError):
            continue

    if not all_sequences:
        return []

    # Sliding window: find recurring subsequences (size 2-4)
    subseq_counter: Counter = Counter()
    subseq_params: dict[tuple, dict] = {}

    for seq in all_sequences:
        for window_size in range(2, min(5, len(seq) + 1)):
            for i in range(len(seq) - window_size + 1):
                subseq = seq[i:i + window_size]
                subseq_counter[subseq] += 1
                if subseq not in subseq_params:
                    subseq_params[subseq] = all_params.get(
                        "->".join(subseq), {})

    # Filter by min_occurrences and deduplicate (prefer longer patterns)
    patterns = []
    seen_tools = set()

    for subseq, count in subseq_counter.most_common():
        if count < min_occurrences:
            continue
        # Skip if this is a subset of an already-found longer pattern
        seq_key = "->".join(subseq)
        if any(seq_key in s for s in seen_tools):
            continue
        seen_tools.add(seq_key)
        patterns.append({
            "sequence": list(subseq),
            "count": count,
            "example_params": subseq_params.get(subseq, {}),
        })

    return patterns[:5]  # Max 5 patterns


async def propose_tool_from_pattern(provider, pattern: dict,
                                     config: dict) -> dict | None:
    """Use LLM to generate a combined tool from a detected pattern.

    Returns {"name": str, "description": str, "source_code": str, "parameters_json": str}
    or None if generation fails.
    """
    model = config.get("synthesis_model",
                        config.get("planning_model", "claude-haiku-4-5-20251001"))
    sequence = pattern.get("sequence", [])
    count = pattern.get("count", 0)

    if not sequence:
        return None

    tool_chain = " → ".join(sequence)
    prompt = (
        "Users repeatedly call these tools in sequence (detected pattern):\n"
        f"Pattern: {tool_chain} (seen {count} times)\n\n"
        "Generate a single Python function that combines this workflow into one tool.\n\n"
        "Requirements:\n"
        "- Function name should describe the combined workflow (snake_case)\n"
        "- Accept necessary parameters with type hints\n"
        "- Only use safe imports (json, re, math, os.path, pathlib, etc.)\n"
        "- Do NOT use subprocess, exec, eval, open()\n"
        "- Keep it simple — the function orchestrates the steps\n\n"
        "Return ONLY valid JSON:\n"
        '{"name": "tool_name", "description": "What it does", '
        '"source_code": "def tool_name(...): ...", '
        '"parameters_json": "{\\"type\\": \\"object\\", \\"properties\\": {...}}"}'
    )

    try:
        result = await provider.complete(
            model=model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        proposal = json.loads(text)

        # Validate required fields
        if not all(k in proposal for k in ("name", "source_code")):
            return None

        # Validate source safety
        ok, err = validate_tool_source(proposal["source_code"])
        if not ok:
            logger.warning("Proposed tool source validation failed: %s", err)
            return None

        # Ensure parameters_json is a string
        if isinstance(proposal.get("parameters_json"), dict):
            proposal["parameters_json"] = json.dumps(proposal["parameters_json"])
        elif not proposal.get("parameters_json"):
            proposal["parameters_json"] = json.dumps({
                "type": "object", "properties": {}})

        logger.info("Proposed tool from pattern: %s (%s)",
                     proposal["name"], tool_chain)
        return proposal

    except Exception as e:
        logger.debug("Tool proposal from pattern failed: %s", e)
        return None
