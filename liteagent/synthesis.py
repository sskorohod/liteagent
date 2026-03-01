"""Tool synthesis and skill crystallization."""

import ast
import json
import logging
from datetime import datetime

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
