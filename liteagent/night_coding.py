"""Autonomous coding/self-improvement session helpers."""

from __future__ import annotations

from datetime import datetime


DEFAULT_VERIFY_COMMANDS = [
    "pytest -q",
]


def infer_default_local_model(config: dict) -> str:
    """Pick the best local coding model from config."""
    agent_cfg = config.get("agent", {}) if isinstance(config, dict) else {}
    models = agent_cfg.get("models", {}) if isinstance(agent_cfg.get("models", {}), dict) else {}
    for key in ("complex", "medium", "simple"):
        value = str(models.get(key, "") or "").strip()
        if value:
            return value
    default_model = str(agent_cfg.get("default_model", "") or "").strip()
    if default_model:
        return default_model
    return "qwen3-coder:30b"


def normalize_session_config(raw: dict | None, app_config: dict | None = None) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    workspace = str(raw.get("workspace") or "").strip()
    verify = raw.get("verification_commands")
    if isinstance(verify, str):
        verify_list = [line.strip() for line in verify.splitlines() if line.strip()]
    elif isinstance(verify, list):
        verify_list = [str(item or "").strip() for item in verify if str(item or "").strip()]
    else:
        verify_list = []
    if not verify_list:
        verify_list = list(DEFAULT_VERIFY_COMMANDS)

    stop_at = str(raw.get("stop_at") or raw.get("run_until") or "").strip()
    if stop_at:
        try:
            stop_at = datetime.fromisoformat(stop_at).isoformat()
        except ValueError:
            stop_at = ""

    objective_mode = str(raw.get("objective_mode") or "improve").strip().lower()
    if objective_mode not in {"finish_then_improve", "improve", "repair_only"}:
        objective_mode = "improve"

    browser_verification = bool(raw.get("browser_verification", True))
    internet_research = bool(raw.get("internet_research", False))
    continue_after_objective = bool(raw.get("continue_after_objective", True))
    local_only = bool(raw.get("local_only", True))

    config = {
        "workspace": workspace,
        "branch_prefix": str(raw.get("branch_prefix") or "codex/night").strip() or "codex/night",
        "local_model": str(raw.get("local_model") or infer_default_local_model(app_config or {})).strip(),
        "internet_research": internet_research,
        "browser_verification": browser_verification,
        "continue_after_objective": continue_after_objective,
        "objective_mode": objective_mode,
        "verification_commands": verify_list[:8],
        "stop_at": stop_at,
        "local_only": local_only,
        "max_patch_files": max(1, min(int(raw.get("max_patch_files", 12) or 12), 50)),
        "max_failed_cycles": max(1, min(int(raw.get("max_failed_cycles", 3) or 3), 8)),
        "notes": str(raw.get("notes") or "").strip()[:1200],
    }
    return config


def session_expired(config: dict, now: datetime | None = None) -> bool:
    stop_at = str((config or {}).get("stop_at") or "").strip()
    if not stop_at:
        return False
    current = now or datetime.now()
    try:
        return current >= datetime.fromisoformat(stop_at)
    except ValueError:
        return False


def stop_label(config: dict) -> str:
    stop_at = str((config or {}).get("stop_at") or "").strip()
    if not stop_at:
        return "manual stop"
    try:
        return datetime.fromisoformat(stop_at).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return stop_at


def build_plan_prompt(goal: dict, attempts_text: str) -> str:
    cfg = normalize_session_config(goal.get("config"), {})
    goal_type = str(goal.get("goal_type") or "autonomous_coding").strip().lower()
    is_self_improvement = goal_type == "self_improvement"
    title = str(goal.get("title") or "").strip()
    objective = str(goal.get("objective") or "").strip()
    session_label = "autonomous self-improvement session" if is_self_improvement else "autonomous night coding session"
    scope_line = (
        "The target system is LiteAgent itself. Prioritize agent reliability, self-healing, tools, planning, memory quality, and operator UX."
        if is_self_improvement else
        "The target is the requested workspace or product code."
    )
    return (
        f"You are the planner for an {session_label}.\n"
        "Build a compact, high-leverage plan for iterative software improvement.\n"
        "The agent will work unattended on local models only.\n\n"
        f"Goal title: {title}\n"
        f"Objective: {objective}\n"
        f"Workspace: {cfg.get('workspace') or '[not set]'}\n"
        f"Goal type: {goal_type}\n"
        f"Objective mode: {cfg.get('objective_mode')}\n"
        f"Continue after objective: {cfg.get('continue_after_objective')}\n"
        f"Internet research allowed: {cfg.get('internet_research')}\n"
        f"Browser verification allowed: {cfg.get('browser_verification')}\n"
        f"Verification commands: {', '.join(cfg.get('verification_commands') or [])}\n"
        f"Session stop: {stop_label(cfg)}\n"
        f"Extra notes: {cfg.get('notes') or '[none]'}\n\n"
        f"Scope: {scope_line}\n\n"
        f"Recent attempts:\n{attempts_text}\n\n"
        "Return ONLY JSON:\n"
        '{"strategy":"...",'
        '"steps":[{"id":"s1","title":"...","action":"...","success_criteria":"..."}]}\n'
        "Rules:\n"
        "- 4..8 steps max\n"
        "- Start with the highest-value unfinished work first\n"
        "- Prioritize bug fixes, verification, runtime reliability, maintainable improvements, and quality gates\n"
        "- For self-improvement goals, prefer improvements backed by real failures, repeated operator pain, missing guards, or weak tests\n"
        "- Every step must be verifiable with tools/tests/build/browser checks\n"
        "- Prefer incremental, reversible changes over large rewrites\n"
        "- Do not ask the user for permission unless blocked by missing credentials/access"
    )


def build_execute_prompt(goal: dict, plan: dict, step: dict, attempts_text: str) -> str:
    cfg = normalize_session_config(goal.get("config"), {})
    goal_type = str(goal.get("goal_type") or "autonomous_coding").strip().lower()
    is_self_improvement = goal_type == "self_improvement"
    title = str(goal.get("title") or "").strip()
    objective = str(goal.get("objective") or "").strip()
    session_label = "autonomous self-improvement cycle" if is_self_improvement else "autonomous night-coding cycle"
    scope_line = (
        "Target LiteAgent itself. Favor changes that measurably improve reliability, planning, tools, memory, delivery, or autonomous execution."
        if is_self_improvement else
        "Target the requested workspace and improve it incrementally."
    )
    verification = "\n".join(f"- {cmd}" for cmd in cfg.get("verification_commands") or [])
    return (
        f"You are running one {session_label}.\n"
        "Think critically before acting, then make one bounded improvement and verify it.\n"
        "Use actual tools. Do not claim success without evidence.\n"
        "You are allowed to continue improving after the original objective if it is already complete.\n\n"
        f"Goal title: {title}\n"
        f"Objective: {objective}\n"
        f"Workspace: {cfg.get('workspace') or '[not set]'}\n"
        f"Goal type: {goal_type}\n"
        f"Local-only session: {cfg.get('local_only')}\n"
        f"Requested local model: {cfg.get('local_model')}\n"
        f"Internet research allowed: {cfg.get('internet_research')}\n"
        f"Browser verification allowed: {cfg.get('browser_verification')}\n"
        f"Objective mode: {cfg.get('objective_mode')}\n"
        f"Continue after objective: {cfg.get('continue_after_objective')}\n"
        f"Max patch files this cycle: {cfg.get('max_patch_files')}\n"
        f"Guard stop after failed cycles: {cfg.get('max_failed_cycles')}\n"
        f"Session stop: {stop_label(cfg)}\n"
        f"Extra notes: {cfg.get('notes') or '[none]'}\n\n"
        f"Scope: {scope_line}\n\n"
        f"Strategy: {str(plan.get('strategy') or goal.get('strategy') or '').strip()}\n"
        f"Current step: {str(step.get('title') or '').strip()}\n"
        f"Action: {str(step.get('action') or '').strip()}\n"
        f"Success criteria: {str(step.get('success_criteria') or '').strip()}\n\n"
        f"Recent attempts:\n{attempts_text}\n\n"
        "Required behavior:\n"
        "- Start by reading the relevant code and deciding the smallest high-value change.\n"
        "- If you modify code, run verification commands afterward.\n"
        "- If frontend/UI changes are involved and browser verification is allowed, use browser tooling to smoke-test the result.\n"
        "- If internet research is allowed, keep research focused and only use it when it materially improves the implementation.\n"
        "- If the originally requested feature is already done, pick the next highest-value improvement in the same workspace.\n"
        "- For self-improvement goals, prefer improvements with strong evidence from logs, tests, recurring failures, or operator pain instead of speculative refactors.\n"
        "- Never use cloud-only models or require paid services for reasoning.\n\n"
        f"Verification commands:\n{verification}\n\n"
        "Return ONLY JSON:\n"
        '{"outcome":"done|progress|blocked|failed","progress_delta":0.08,'
        '"completed":false,"phase":"...","summary":"...",'
        '"next_action":"...","insight":"...","alternative":"..."}'
    )
