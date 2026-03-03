"""Dashboard API routes for LiteAgent web UI."""

import csv
import io
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DASHBOARD_USER = "dashboard-user"
CUSTOM_TOOLS_DIR = Path.home() / ".liteagent" / "custom_tools"

_FILE_BROWSER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Files — LiteAgent</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>body{font-family:Inter,system-ui,sans-serif}
.file-row:hover{background:#f8fafc}</style>
</head>
<body class="bg-gray-50 min-h-screen">
<div class="max-w-5xl mx-auto px-4 py-8">
  <div class="flex items-center justify-between mb-6">
    <h1 class="text-2xl font-bold text-gray-800">File Storage</h1>
    <div class="flex gap-3 items-center">
      <input id="searchInput" type="text" placeholder="Search files..."
        class="px-3 py-2 border rounded-lg text-sm w-64 focus:ring-2 focus:ring-blue-400 outline-none">
      <select id="sourceFilter" class="px-3 py-2 border rounded-lg text-sm bg-white">
        <option value="">All sources</option>
        <option value="telegram">Telegram</option>
        <option value="api">API / Chat</option>
        <option value="voice">Voice</option>
        <option value="download">Downloads</option>
        <option value="agent">Agent</option>
      </select>
      <span id="fileCount" class="text-sm text-gray-500"></span>
    </div>
  </div>
  <div id="fileList" class="bg-white rounded-xl shadow-sm border divide-y"></div>
  <div id="emptyState" class="hidden text-center py-16 text-gray-400">
    <p class="text-lg">No files yet</p>
    <p class="text-sm mt-1">Files uploaded via Telegram, chat, or API will appear here</p>
  </div>
</div>
<script>
const API = window.location.origin;
let allFiles = [];

function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024*1024) return (bytes/1024).toFixed(1) + ' KB';
  return (bytes/(1024*1024)).toFixed(1) + ' MB';
}

function sourceIcon(s) {
  const m = {telegram:'💬',api:'🌐',voice:'🎙',download:'⬇️',agent:'🤖'};
  return m[s] || '📁';
}

function renderFiles(files) {
  const el = document.getElementById('fileList');
  const empty = document.getElementById('emptyState');
  if (!files.length) { el.innerHTML=''; empty.classList.remove('hidden'); return; }
  empty.classList.add('hidden');
  el.innerHTML = files.map(f => `
    <div class="file-row flex items-center px-4 py-3 gap-3 cursor-pointer"
         onclick="downloadFile('${f.storage_key}')">
      <span class="text-xl">${sourceIcon(f.source)}</span>
      <div class="flex-1 min-w-0">
        <div class="font-medium text-gray-800 truncate">${f.original_name}</div>
        <div class="text-xs text-gray-400 truncate">${f.description||''}</div>
      </div>
      <div class="text-right shrink-0">
        <div class="text-sm text-gray-600">${formatSize(f.size_bytes)}</div>
        <div class="text-xs text-gray-400">${(f.created_at||'').slice(0,10)}</div>
      </div>
      <span class="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-500">${f.source}</span>
    </div>`).join('');
}

async function downloadFile(key) {
  const resp = await fetch(API+'/api/files/url/'+key);
  const data = await resp.json();
  if (data.url) window.open(data.url, '_blank');
}

async function loadFiles() {
  const source = document.getElementById('sourceFilter').value;
  const params = new URLSearchParams({limit:'500'});
  if (source) params.set('source', source);
  const resp = await fetch(API+'/api/files?'+params);
  allFiles = await resp.json();
  document.getElementById('fileCount').textContent = allFiles.length+' files';
  filterAndRender();
}

function filterAndRender() {
  const q = document.getElementById('searchInput').value.toLowerCase();
  const filtered = q
    ? allFiles.filter(f => (f.original_name+' '+f.description).toLowerCase().includes(q))
    : allFiles;
  renderFiles(filtered);
}

document.getElementById('searchInput').addEventListener('input', filterAndRender);
document.getElementById('sourceFilter').addEventListener('change', loadFiles);
loadFiles();
</script>
</body>
</html>
"""


def mount_dashboard(app, agent):
    """Mount dashboard API routes onto FastAPI app."""
    try:
        from fastapi import HTTPException
        from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
    except ImportError:
        raise ImportError("FastAPI is required: pip install liteagent[api]")

    import os
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page():
        """Serve the dashboard SPA."""
        html_path = os.path.join(STATIC_DIR, "dashboard.html")
        if not os.path.exists(html_path):
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return FileResponse(html_path, media_type="text/html")

    @app.get("/favicon.ico")
    async def favicon():
        """Return empty 204 for favicon requests."""
        return Response(status_code=204)

    @app.get("/api/overview")
    async def api_overview():
        """KPI overview data."""
        mem = agent.memory
        usage = mem.get_total_usage_stats()
        return {
            "total_calls": usage["total_calls"],
            "total_cost_usd": usage["total_cost_usd"],
            "total_tokens": usage["total_input_tokens"] + usage["total_output_tokens"],
            "memory_count": mem.get_memory_count(),
            "today_cost_usd": round(mem.get_today_cost(), 4),
            "tools_count": len(agent.tools.get_definitions()),
        }

    @app.get("/api/overview/enhanced")
    async def api_overview_enhanced():
        """Consolidated overview data for redesigned dashboard."""
        mem = agent.memory
        usage = mem.get_total_usage_stats()
        today = mem.get_today_stats()
        yesterday = mem.get_yesterday_stats()

        kpi = {
            "total_calls": usage["total_calls"],
            "total_cost_usd": usage["total_cost_usd"],
            "total_tokens": usage["total_input_tokens"] + usage["total_output_tokens"],
            "memory_count": mem.get_memory_count(),
            "today_cost_usd": round(mem.get_today_cost(), 4),
            "today_calls": today["calls"],
            "tools_count": len(agent.tools.get_definitions()),
            "success_rate": mem.get_success_rate(24),
            "avg_confidence": mem.get_avg_confidence(24),
            "cache_efficiency": mem.get_cache_efficiency(),
            "yesterday_cost_usd": round(yesterday["cost"], 4),
            "yesterday_calls": yesterday["calls"],
        }

        # Composite health status
        budget_pct = round(today["cost"] / agent.budget_daily * 100, 1) if agent.budget_daily > 0 else 0
        error_rate = 100 - kpi["success_rate"]
        health_status = "healthy"
        if error_rate > 20 or budget_pct > 90:
            health_status = "down"
        elif error_rate > 10 or budget_pct > 70:
            health_status = "degraded"

        return {
            "kpi": kpi,
            "health": {
                "status": health_status,
                "error_rate_24h": round(error_rate, 1),
                "budget_pct": budget_pct,
            },
            "model_distribution": mem.get_model_distribution_today(),
        }

    @app.get("/api/usage")
    async def api_usage(days: int = 7):
        """Usage breakdown by model with KPI stats."""
        mem = agent.memory
        total = mem.get_total_usage_stats()
        today = mem.get_today_stats()
        hour = mem.get_hour_cost()
        return {
            "models": mem.get_usage_summary(days),
            "today_cost": round(today["cost"], 4),
            "today_calls": today["calls"],
            "hour_cost": round(hour["cost"], 4),
            "hour_calls": hour["calls"],
            "total_cost": round(total["total_cost_usd"], 4),
            "total_calls": total["total_calls"],
            "total_tokens": total["total_input_tokens"] + total["total_output_tokens"],
        }

    @app.get("/api/usage/daily")
    async def api_usage_daily(days: int = 14):
        """Daily usage for chart."""
        return agent.memory.get_daily_usage(days)

    @app.get("/api/memories")
    async def api_memories():
        """All memories."""
        return agent.memory.get_all_memories()

    @app.delete("/api/memories/{memory_id}")
    async def api_delete_memory(memory_id: int):
        """Delete a memory."""
        ok = agent.memory.delete_memory(memory_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "deleted"}

    @app.get("/api/tools")
    async def api_tools():
        """List registered tools with source info."""
        defs = agent.tools.get_definitions()
        # Determine tool source (builtin, mcp, custom, onboarding)
        builtin_names = set(agent.config.get("tools", {}).get(
            "builtin", ["read_file", "write_file", "exec_command", "memory_search"]))
        builtin_names.add("rag_search")
        mcp_names = {n for n in agent.tools._tools if "__" in n}
        onboarding_names = {"setup_agent"}
        custom_dir = CUSTOM_TOOLS_DIR

        result = []
        for d in defs:
            name = d["name"]
            if name in onboarding_names:
                source = "onboarding"
            elif name in mcp_names:
                source = "mcp"
            elif name in builtin_names:
                source = "builtin"
            else:
                source = "custom"
            schema = d.get("input_schema", {})
            params = []
            for pname, pinfo in schema.get("properties", {}).items():
                params.append({
                    "name": pname,
                    "type": pinfo.get("type", "string"),
                    "required": pname in schema.get("required", []),
                })
            result.append({
                "name": name,
                "description": d.get("description", ""),
                "source": source,
                "parameters": params,
            })
        return result

    @app.post("/api/tools/custom")
    async def api_tools_add_custom(body: dict):
        """Add a custom Python tool from code string."""
        name = body.get("name", "").strip()
        description = body.get("description", "").strip()
        code = body.get("code", "").strip()

        if not name:
            raise HTTPException(status_code=400, detail="Tool name required")
        if not code:
            raise HTTPException(status_code=400, detail="Tool code required")
        if name in agent.tools._tools:
            raise HTTPException(status_code=400, detail=f"Tool '{name}' already exists")

        # Validate with AST-based analysis (same as synthesis.py)
        from ..synthesis import validate_tool_source
        ok, err = validate_tool_source(code)
        if not ok:
            raise HTTPException(status_code=400, detail=f"Code validation failed: {err}")

        # Compile and register (with restricted builtins)
        try:
            import builtins as _builtins
            _BLOCKED_BUILTINS = {"exec", "eval", "compile", "__import__", "open",
                                 "globals", "locals", "getattr", "setattr", "delattr",
                                 "breakpoint", "exit", "quit"}
            safe_builtins = {k: v for k, v in vars(_builtins).items()
                             if k not in _BLOCKED_BUILTINS}
            namespace = {"__builtins__": safe_builtins}
            exec(compile(code, f"<custom_tool_{name}>", "exec"), namespace)  # noqa: S102
            func = namespace.get(name)
            if not func or not callable(func):
                raise HTTPException(status_code=400,
                    detail=f"Code must define a function named '{name}'")

            agent.tools.tool(name=name, description=description or func.__doc__)(func)

            # Save to disk for persistence
            CUSTOM_TOOLS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            tool_path = CUSTOM_TOOLS_DIR / f"{name}.py"
            tool_path.write_text(code, encoding="utf-8")
            logger.info("Custom tool '%s' added and saved to %s", name, tool_path)

            return {"ok": True, "name": name}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid code: {e}")

    @app.delete("/api/tools/custom/{name}")
    async def api_tools_delete_custom(name: str):
        """Remove a custom tool."""
        builtin_names = set(agent.config.get("tools", {}).get(
            "builtin", ["read_file", "write_file", "exec_command", "memory_search"]))
        builtin_names.update(["rag_search", "memory_search", "setup_agent"])
        if name in builtin_names or "__" in name:
            raise HTTPException(status_code=400, detail="Cannot delete builtin/MCP tools")
        if name not in agent.tools._tools:
            raise HTTPException(status_code=404, detail=f"Tool '{name}' not found")

        del agent.tools._tools[name]
        if name in agent.tools._handlers:
            del agent.tools._handlers[name]

        # Remove from disk
        tool_path = CUSTOM_TOOLS_DIR / f"{name}.py"
        if tool_path.exists():
            tool_path.unlink()
            logger.info("Custom tool '%s' removed from disk", name)

        return {"ok": True, "name": name}

    @app.get("/api/config")
    async def api_config():
        """Read-only agent config (strip sensitive keys)."""
        import copy
        cfg = copy.deepcopy(agent.config)
        # Strip ALL secret fields from API response
        _SECRET_DISPLAY_PATHS = [
            ("providers", "anthropic", "api_key"),
            ("providers", "openai", "api_key"),
            ("providers", "grok", "api_key"),
            ("providers", "gemini", "api_key"),
            ("channels", "telegram", "token"),
            ("tools", "brave_api_key"),
            ("storage", "access_key"),
            ("storage", "secret_key"),
            ("rag", "qdrant", "api_key"),
        ]
        for key_path in _SECRET_DISPLAY_PATHS:
            obj = cfg
            for k in key_path[:-1]:
                obj = obj.get(k, {}) if isinstance(obj, dict) else {}
            if isinstance(obj, dict) and key_path[-1] in obj:
                obj[key_path[-1]] = "***"
        return cfg

    @app.post("/api/config")
    async def api_config_save(body: dict):
        """Save full config JSON from the dashboard editor.

        Secrets shown as '***' in the editor are preserved from the
        current running config so they are never accidentally wiped.
        """
        import copy
        from ..config import save_config as _save_config

        _SECRET_MERGE_PATHS = [
            ("providers", "anthropic", "api_key"),
            ("providers", "openai", "api_key"),
            ("providers", "grok", "api_key"),
            ("providers", "gemini", "api_key"),
            ("channels", "telegram", "token"),
            ("tools", "brave_api_key"),
            ("storage", "access_key"),
            ("storage", "secret_key"),
            ("rag", "qdrant", "api_key"),
        ]

        # Deep-copy incoming to avoid mutation
        new_cfg = copy.deepcopy(body)

        # Preserve secrets: if value is "***", restore from current config
        for key_path in _SECRET_MERGE_PATHS:
            # Navigate new config
            new_obj = new_cfg
            for k in key_path[:-1]:
                new_obj = new_obj.get(k, {}) if isinstance(new_obj, dict) else {}
            # Navigate current config
            cur_obj = agent.config
            for k in key_path[:-1]:
                cur_obj = cur_obj.get(k, {}) if isinstance(cur_obj, dict) else {}
            final_key = key_path[-1]
            if isinstance(new_obj, dict) and new_obj.get(final_key) == "***":
                if isinstance(cur_obj, dict) and final_key in cur_obj:
                    new_obj[final_key] = cur_obj[final_key]

        # Preserve internal keys from current config
        for k, v in agent.config.items():
            if k.startswith("_") and k not in new_cfg:
                new_cfg[k] = v

        # Apply to running agent config
        agent.config.update(new_cfg)

        # Save to disk
        try:
            _save_config(agent.config)
        except Exception as exc:
            raise HTTPException(500, f"Failed to save config: {exc}")

        # Trigger runtime config update if watcher exists
        watcher = getattr(app.state, "config_watcher", None)
        if watcher:
            try:
                await watcher.force_reload()
            except Exception:
                pass  # Non-critical

        return {"ok": True}

    @app.get("/api/history")
    async def api_history():
        """Conversation history for dashboard user (persisted)."""
        return agent.memory.get_chat_history_for_display(DASHBOARD_USER, limit=100)

    @app.delete("/api/history")
    async def api_clear_history():
        """Clear chat history for dashboard user."""
        agent.memory.clear_chat_history(DASHBOARD_USER)
        agent.memory.clear_conversation(DASHBOARD_USER)
        return {"ok": True}

    # ── Export endpoints ──────────────────────

    @app.get("/api/export/memories")
    async def export_memories(format: str = "json"):
        """Export memories as JSON, CSV, or Markdown."""
        memories = agent.memory.get_all_memories()
        if format == "csv":
            output = io.StringIO()
            writer = csv.DictWriter(output,
                fieldnames=["id", "user_id", "content", "type", "importance", "created_at"],
                extrasaction="ignore")
            writer.writeheader()
            writer.writerows(memories)
            return Response(content=output.getvalue(), media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=memories.csv"})
        elif format == "md":
            lines = ["# Memories\n"]
            for m in memories:
                lines.append(f"- **[{m['type']}]** {m['content']} _(importance: {m['importance']})_")
            return Response(content="\n".join(lines), media_type="text/markdown",
                headers={"Content-Disposition": "attachment; filename=memories.md"})
        return memories

    @app.get("/api/export/history")
    async def export_history(format: str = "json"):
        """Export conversation history."""
        msgs = agent.memory.get_history(DASHBOARD_USER)
        if format == "md":
            lines = ["# Conversation History\n"]
            for m in msgs:
                role = "**User**" if m["role"] == "user" else "**Assistant**"
                content = m["content"] if isinstance(m["content"], str) else str(m["content"])
                lines.append(f"{role}: {content}\n")
            return Response(content="\n".join(lines), media_type="text/markdown",
                headers={"Content-Disposition": "attachment; filename=history.md"})
        return msgs

    @app.get("/api/export/usage")
    async def export_usage(format: str = "json", days: int = 30):
        """Export usage data."""
        data = agent.memory.get_usage_summary(days)
        if format == "csv":
            output = io.StringIO()
            if data:
                writer = csv.DictWriter(output, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            return Response(content=output.getvalue(), media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=usage.csv"})
        return data

    # ── MCP Management ────────────────────────

    @app.get("/api/mcp/servers")
    async def api_mcp_servers():
        """List connected MCP servers."""
        return agent.tools.get_mcp_server_info()

    @app.post("/api/mcp/reload")
    async def api_mcp_reload():
        """Reload MCP servers from config."""
        await agent.reload_mcp()
        return {"status": "reloaded", "servers": agent.tools.get_mcp_server_info()}

    # ── Scheduler info ────────────────────────

    @app.get("/api/scheduler/jobs")
    async def api_scheduler_jobs():
        """List scheduled jobs."""
        sched = getattr(agent, '_scheduler', None)
        if sched:
            return sched.get_jobs()
        return []

    # ── Operations Dashboard ────────────────────

    @app.get("/api/ops/active")
    async def api_ops_active():
        """Currently executing requests and scheduler job status."""
        from ..agent import LiteAgent
        active = LiteAgent.get_active_requests()

        scheduler_running = []
        sched = getattr(agent, '_scheduler', None)
        if sched:
            for job in sched._jobs:
                if job.get("_running"):
                    scheduler_running.append({
                        "name": job["name"],
                        "started_at": job.get("_run_started"),
                    })

        queued = LiteAgent.get_queued_requests()

        return {
            "requests": active,
            "queued": queued,
            "scheduler_jobs_running": scheduler_running,
        }

    @app.get("/api/ops/recent")
    async def api_ops_recent(limit: int = 15):
        """Recent agent interactions for activity feed."""
        try:
            rows = agent.memory.db.execute(
                """SELECT id, user_id, user_input, agent_response,
                          tool_calls_json, success, confidence, model_used, created_at
                   FROM interaction_log
                   ORDER BY id DESC LIMIT ?""",
                (min(limit, 50),)).fetchall()
        except Exception:
            return []

        result = []
        for r in rows:
            tool_calls = []
            try:
                tool_calls = json.loads(r[4]) if r[4] else []
            except Exception:
                pass
            result.append({
                "id": r[0],
                "user_id": r[1],
                "input_preview": (r[2] or "")[:150],
                "response_preview": (r[3] or "")[:150],
                "tool_calls": [tc.get("name", "?") for tc in tool_calls][:5],
                "tool_count": len(tool_calls),
                "success": r[5],
                "confidence": r[6],
                "model": r[7],
                "created_at": r[8],
            })
        return result

    @app.get("/api/ops/system")
    async def api_ops_system():
        """System status: provider, model, features, scheduler, budget."""
        from ..agent import LiteAgent
        from ..scheduler import cron_matches
        from datetime import datetime, timedelta

        agent_cfg = agent.config.get("agent", {})

        # Provider info
        provider_info = {
            "provider": agent_cfg.get("provider", "anthropic"),
            "model": agent.default_model,
            "cascade_routing": agent.cascade_routing,
            "models": agent.models,
        }

        # Budget info
        today_cost = agent.memory.get_today_cost()
        budget_info = {
            "daily_budget": agent.budget_daily,
            "today_cost": round(today_cost, 4),
            "budget_pct": round(today_cost / agent.budget_daily * 100, 1) if agent.budget_daily > 0 else 0,
        }

        # Scheduler jobs with next run calculation
        sched = getattr(agent, '_scheduler', None)
        jobs = []
        if sched:
            now = datetime.now()
            for job in sched._jobs:
                # Calculate next run by scanning forward (max 7 days)
                next_run = None
                check = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                for _ in range(7 * 24 * 60):
                    if cron_matches(job["cron"], check):
                        next_run = check.isoformat()
                        break
                    check += timedelta(minutes=1)

                jobs.append({
                    "name": job["name"],
                    "cron_expr": job["cron_expr"],
                    "last_run": job["last_run"],
                    "next_run": next_run,
                    "running": job.get("_running", False),
                })

        # Feature flags (compact)
        features_cfg = agent.config.get("features", {})
        features = {}
        for name in ["dream_cycle", "self_evolving_prompt", "proactive_agent",
                      "auto_tool_synthesis", "confidence_gate", "style_adaptation",
                      "skill_crystallization", "counterfactual_replay",
                      "internal_monologue"]:
            cfg = features_cfg.get(name, {})
            features[name] = cfg.get("enabled", False)

        return {
            "provider": provider_info,
            "budget": budget_info,
            "scheduler_jobs": jobs,
            "features": features,
            "active_request_count": len(LiteAgent._active_requests),
        }

    # ── Cascade routing monitor ────────────────

    @app.get("/api/ops/cascade")
    async def api_ops_cascade():
        """Cascade routing status, tier costs, and recent history."""
        from ..agent import LiteAgent
        from ..providers import get_pricing

        summary = LiteAgent.get_cascade_summary()
        history = LiteAgent.get_cascade_history()[-20:]

        # Cost per tier
        tier_costs = {}
        for tier_name in ("simple", "medium", "complex"):
            model_name = agent.models.get(tier_name, agent.default_model)
            lookup = model_name
            if ":" in model_name and model_name.split(":")[0] in (
                "anthropic", "openai", "gemini", "ollama"
            ):
                lookup = model_name.split(":", 1)[1]
            pricing = get_pricing(lookup)
            tier_costs[tier_name] = {
                "model": model_name,
                "input_per_mtok": pricing.get("input", 0),
                "output_per_mtok": pricing.get("output", 0),
            }

        return {
            "enabled": agent.cascade_routing,
            "models": agent.models,
            "default_model": agent.default_model,
            "tier_costs": tier_costs,
            "summary": summary,
            "history": history,
            "is_local_only_now": agent._is_local_only_hours(),
        }

    # ── Feature monitoring ────────────────────

    @app.get("/api/features/status")
    async def api_features_status():
        """Status of all 9 metacognition/evolution/synthesis features."""
        features = agent.config.get("features", {})
        feature_names = [
            "dream_cycle", "self_evolving_prompt", "proactive_agent",
            "auto_tool_synthesis", "confidence_gate", "style_adaptation",
            "skill_crystallization", "counterfactual_replay",
            "internal_monologue",
        ]
        status = {}
        for name in feature_names:
            cfg = features.get(name, {})
            status[name] = {"enabled": cfg.get("enabled", False)}

        db = agent.memory.db
        try:
            status["confidence_gate"]["logged_interactions"] = db.execute(
                "SELECT COUNT(*) FROM interaction_log").fetchone()[0]
            status["self_evolving_prompt"]["friction_signals"] = db.execute(
                "SELECT COUNT(*) FROM friction_signals").fetchone()[0]
            status["self_evolving_prompt"]["patches_applied"] = db.execute(
                "SELECT COUNT(*) FROM prompt_patches WHERE applied=1"
            ).fetchone()[0]
            status["auto_tool_synthesis"]["tools_created"] = db.execute(
                "SELECT COUNT(*) FROM synthesized_tools WHERE approved=1"
            ).fetchone()[0]
            status["skill_crystallization"]["skills_count"] = db.execute(
                "SELECT COUNT(*) FROM skills").fetchone()[0]
            status["style_adaptation"]["profiles_count"] = db.execute(
                "SELECT COUNT(*) FROM style_profiles").fetchone()[0]
        except Exception:
            pass  # Tables may not exist in older DBs
        return status

    @app.post("/api/features/toggle")
    async def api_features_toggle(body: dict):
        """Toggle a single feature on/off. Body: {name: str, enabled: bool}."""
        from ..config import save_config

        ALL_FEATURES = [
            "dream_cycle", "self_evolving_prompt", "proactive_agent",
            "auto_tool_synthesis", "confidence_gate", "style_adaptation",
            "skill_crystallization", "counterfactual_replay",
            "internal_monologue",
        ]
        name = body.get("name", "")
        if name not in ALL_FEATURES:
            return {"ok": False, "error": f"Unknown feature: {name}"}

        enabled = bool(body.get("enabled", False))
        features = agent.config.setdefault("features", {})
        feat_cfg = features.setdefault(name, {})
        feat_cfg["enabled"] = enabled

        save_config(agent.config)
        logger.info("Feature toggled: %s = %s", name, enabled)
        return {"ok": True, "name": name, "enabled": enabled}

    @app.post("/api/features/preset")
    async def api_features_preset(body: dict):
        """Apply a feature preset. Body: {preset: "basic"|"all"|"none"}."""
        from ..config import save_config

        PRESETS = {
            "basic": ["style_adaptation", "confidence_gate", "skill_crystallization"],
            "all": [
                "dream_cycle", "self_evolving_prompt", "proactive_agent",
                "auto_tool_synthesis", "confidence_gate", "style_adaptation",
                "skill_crystallization", "counterfactual_replay",
                "internal_monologue",
            ],
            "none": [],
        }
        ALL_FEATURES = PRESETS["all"]

        preset = body.get("preset", "")
        if preset not in PRESETS:
            return {"ok": False, "error": f"Unknown preset: {preset}"}

        enabled_list = PRESETS[preset]
        features = agent.config.setdefault("features", {})
        for name in ALL_FEATURES:
            feat_cfg = features.setdefault(name, {})
            feat_cfg["enabled"] = name in enabled_list

        save_config(agent.config)
        logger.info("Features preset applied: %s", preset)
        return {"ok": True, "preset": preset, "enabled": enabled_list}

    @app.get("/api/features/patches")
    async def api_prompt_patches():
        """List prompt patches."""
        rows = agent.memory.db.execute(
            "SELECT id, patch_text, reason, applied, created_at "
            "FROM prompt_patches ORDER BY created_at DESC").fetchall()
        return [{"id": r[0], "patch": r[1], "reason": r[2],
                 "applied": r[3], "created_at": r[4]} for r in rows]

    @app.post("/api/features/patches/{patch_id}/apply")
    async def api_apply_patch(patch_id: int):
        """Apply a prompt patch."""
        agent.memory.db.execute(
            "UPDATE prompt_patches SET applied=1 WHERE id=?", (patch_id,))
        agent.memory.db.commit()
        return {"status": "applied"}

    @app.get("/api/features/synth-tools")
    async def api_synth_tools():
        """List synthesized tools."""
        rows = agent.memory.db.execute(
            "SELECT id, name, description, approved, created_at "
            "FROM synthesized_tools ORDER BY created_at DESC").fetchall()
        return [{"id": r[0], "name": r[1], "description": r[2],
                 "approved": r[3], "created_at": r[4]} for r in rows]

    @app.post("/api/features/synth-tools/{tool_id}/approve")
    async def api_approve_tool(tool_id: int):
        """Approve a synthesized tool."""
        row = agent.memory.db.execute(
            "SELECT name, description, source_code, parameters_json "
            "FROM synthesized_tools WHERE id=?", (tool_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Tool not found")
        agent.memory.db.execute(
            "UPDATE synthesized_tools SET approved=1 WHERE id=?", (tool_id,))
        agent.memory.db.commit()
        from ..synthesis import validate_tool_source, register_synthesized_tool
        ok, err = validate_tool_source(row[2])
        if ok:
            schema = json.loads(row[3]) if row[3] else {
                "type": "object", "properties": {}}
            register_synthesized_tool(
                agent.tools, row[0], row[2], row[1], schema)
        return {"status": "approved", "validation": "ok" if ok else err}

    # ── Provider Settings ─────────────────────

    @app.get("/api/settings/providers")
    async def api_settings_providers():
        """Get provider settings with key status and available models."""
        from ..config import get_api_key, key_preview, PROVIDER_ENV_VARS
        from ..providers import PROVIDER_MODELS, refresh_ollama_models, is_ollama_available

        agent_cfg = agent.config.get("agent", {})
        active_provider = agent_cfg.get("provider", "anthropic")
        active_model = agent_cfg.get("default_model", "claude-sonnet-4-20250514")

        # Auto-discover Ollama models from local instance
        ollama_running = is_ollama_available()
        if ollama_running:
            refresh_ollama_models()

        providers = {}
        for name, models in PROVIDER_MODELS.items():
            key = get_api_key(name)
            if name == "ollama":
                providers[name] = {
                    "has_key": ollama_running,
                    "key_preview": "(running)" if ollama_running else "(not running)",
                    "models": models,
                    "local": True,
                }
            else:
                providers[name] = {
                    "has_key": bool(key),
                    "key_preview": key_preview(key) if key else "",
                    "models": models,
                }

        return {
            "active_provider": active_provider,
            "active_model": active_model,
            "providers": providers,
        }

    @app.post("/api/settings/provider/key")
    async def api_settings_save_key(body: dict):
        """Save an API key for a provider (does NOT switch active provider)."""
        from ..config import save_provider_key
        from ..providers import PROVIDER_MODELS

        provider_name = body.get("provider", "").strip()
        api_key = body.get("api_key", "").strip()

        if not provider_name:
            raise HTTPException(status_code=400, detail="Provider name required")
        if provider_name not in PROVIDER_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")
        if not api_key:
            raise HTTPException(status_code=400, detail="API key required")
        if provider_name == "ollama":
            return {"ok": True, "message": "Ollama doesn't need an API key"}

        # Validate key format
        _KEY_PREFIXES = {
            "anthropic": ("sk-ant-", "Anthropic keys start with 'sk-ant-'. Get yours at console.anthropic.com/settings/keys"),
            "openai": ("sk-", "OpenAI keys start with 'sk-'. Get yours at platform.openai.com/api-keys"),
            "grok": ("xai-", "xAI keys start with 'xai-'. Get yours at console.x.ai"),
            "qwen": ("sk-", "DashScope keys start with 'sk-'. Get yours at dashscope.console.aliyun.com"),
        }
        prefix_info = _KEY_PREFIXES.get(provider_name)
        if prefix_info:
            prefix, hint = prefix_info
            if not api_key.startswith(prefix):
                logger.warning("Invalid key format for %s: starts with '%s...'", provider_name, api_key[:6])
                raise HTTPException(status_code=400, detail=f"Invalid key format. {hint}")

        logger.info("Saving API key for provider: %s (key: %s...%s)", provider_name, api_key[:6], api_key[-4:])
        save_provider_key(provider_name, api_key)

        # If this is the active provider, also update env + recreate provider
        from ..config import PROVIDER_ENV_VARS
        import os as _os
        active_provider = agent.config.get("agent", {}).get("provider", "anthropic")
        env_var = PROVIDER_ENV_VARS.get(provider_name)
        if env_var:
            _os.environ[env_var] = api_key
            logger.info("Updated env var %s", env_var)

        if provider_name == active_provider:
            try:
                from ..providers import create_provider
                agent.provider = create_provider(agent.config)
                logger.info("Recreated active provider: %s", provider_name)
            except Exception as e:
                logger.warning("Failed to recreate provider: %s", e)

        return {"ok": True, "provider": provider_name}

    @app.post("/api/settings/provider")
    async def api_settings_apply_provider(body: dict):
        """Switch active provider and model. Optionally save API key."""
        from ..config import save_provider_key, get_api_key, PROVIDER_ENV_VARS
        from ..providers import PROVIDER_MODELS
        import os as _os

        provider_name = body.get("provider", "").strip()
        api_key = body.get("api_key", "").strip()
        model = body.get("model", "").strip()

        if not provider_name:
            raise HTTPException(status_code=400, detail="Provider name required")
        if provider_name not in PROVIDER_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {provider_name}")

        # Save API key if provided (with format validation)
        if api_key and provider_name != "ollama":
            _KEY_PREFIXES_APPLY = {
                "anthropic": "sk-ant-",
                "openai": "sk-",
                "grok": "xai-",
                "qwen": "sk-",
            }
            prefix = _KEY_PREFIXES_APPLY.get(provider_name)
            if prefix and not api_key.startswith(prefix):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid key format for {provider_name}. Key should start with '{prefix}'.")
            save_provider_key(provider_name, api_key)

        # Ensure API key is available in env
        if provider_name != "ollama":
            key = api_key or get_api_key(provider_name)
            env_var = PROVIDER_ENV_VARS.get(provider_name)
            if key and env_var:
                _os.environ[env_var] = key
            elif not key and provider_name != "ollama":
                raise HTTPException(
                    status_code=400,
                    detail=f"No API key for {provider_name}. Save a key first.")

        # Check if SDK is installed before switching
        _SDK_PACKAGES = {
            "anthropic": "anthropic",
            "openai": "openai",
            "grok": "openai",
            "gemini": "google.generativeai",
            "ollama": "openai",
        }
        pkg = _SDK_PACKAGES.get(provider_name)
        if pkg:
            try:
                __import__(pkg)
            except ImportError:
                pip_extra = {"openai": "openai", "grok": "openai", "gemini": "gemini", "ollama": "ollama"}.get(provider_name, provider_name)
                raise HTTPException(
                    status_code=400,
                    detail=f"SDK not installed. Run: pip install liteagent[{pip_extra}]")

        # Update config and recreate provider
        agent.config.setdefault("agent", {})["provider"] = provider_name

        # Auto-select model if not specified
        if not model and provider_name == "ollama":
            from ..providers import refresh_ollama_models
            ollama_models = refresh_ollama_models()
            model = ollama_models[0] if ollama_models else ""
        if not model:
            models = PROVIDER_MODELS.get(provider_name, [])
            model = models[0] if models else ""

        if model:
            agent.config["agent"]["default_model"] = model
            agent.default_model = model
            # Update cascade models to use the same Ollama model
            if provider_name == "ollama":
                agent.models = {"simple": model, "medium": model, "complex": model}
                agent.config["agent"]["models"] = {"simple": model, "medium": model, "complex": model}

        try:
            from ..providers import create_provider
            agent.provider = create_provider(agent.config)

            # Persist to config.json so settings survive server restart
            from ..config import save_config
            save_config(agent.config)

            return {"ok": True, "provider": provider_name, "model": model}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Provider init failed: {e}")

    @app.post("/api/settings/provider/test")
    async def api_settings_test_provider(body: dict):
        """Test provider connectivity with given API key."""
        import time

        provider_name = body.get("provider", "").strip()
        api_key = body.get("api_key", "").strip()

        if not provider_name:
            raise HTTPException(status_code=400, detail="Provider name required")

        # Check SDK first
        _SDK = {"anthropic": "anthropic", "openai": "openai",
                "gemini": "google.generativeai", "ollama": "openai"}
        pkg = _SDK.get(provider_name)
        if pkg:
            try:
                __import__(pkg)
            except ImportError:
                pip_extra = {"openai": "openai", "grok": "openai", "gemini": "gemini", "ollama": "ollama"}.get(provider_name, provider_name)
                return {"ok": False, "error": f"SDK not installed. Run: pip install liteagent[{pip_extra}]"}

        # For Ollama, no key needed
        if provider_name == "ollama":
            api_key = "ollama"

        if not api_key:
            # Try existing key
            from ..config import get_api_key
            api_key = get_api_key(provider_name)
            if not api_key:
                return {"ok": False, "error": "No API key provided or saved"}

        try:
            from ..providers import create_test_provider
            logger.info("Testing %s connectivity (key: %s...)", provider_name, api_key[:8] if api_key else "none")
            provider = create_test_provider(provider_name, api_key)
            start = time.time()
            # Minimal test call — for Ollama use first available model
            from ..providers import PROVIDER_MODELS
            _default_test_models = {
                "anthropic": "claude-haiku-4-5-20251001",
                "openai": "gpt-4o-mini",
                "grok": "grok-3-mini",
                "gemini": "gemini-2.0-flash",
                "qwen": "qwen-turbo",
            }
            if provider_name == "ollama":
                ollama_models = PROVIDER_MODELS.get("ollama", [])
                test_model = ollama_models[0] if ollama_models else "llama3.1"
            else:
                test_model = _default_test_models.get(provider_name, "gpt-4o-mini")

            await provider.complete(
                model=test_model, max_tokens=5,
                messages=[{"role": "user", "content": [{"type": "text", "text": "Hi"}]}])
            latency_ms = int((time.time() - start) * 1000)
            logger.info("Test %s OK (%dms)", provider_name, latency_ms)
            return {"ok": True, "latency_ms": latency_ms}
        except Exception as e:
            logger.warning("Test %s FAILED: %s", provider_name, e)
            return {"ok": False, "error": str(e)}

    @app.delete("/api/settings/provider/{name}/key")
    async def api_settings_delete_key(name: str):
        """Delete a saved provider API key."""
        from ..config import delete_provider_key
        deleted = delete_provider_key(name)
        if not deleted:
            raise HTTPException(status_code=404, detail="No saved key found")
        return {"ok": True}

    # ── Routing Settings ──────────────────

    @app.get("/api/settings/routing")
    async def api_settings_routing():
        """Get cascade routing mode and model config."""
        cost_cfg = agent.config.get("cost", {})
        agent_cfg = agent.config.get("agent", {})
        return {
            "cascade_routing": cost_cfg.get("cascade_routing", True),
            "models": agent_cfg.get("models", {}),
            "default_model": agent_cfg.get("default_model", ""),
            "local_only_hours": cost_cfg.get("local_only_hours", {
                "enabled": False, "start": "00:00", "end": "08:00"
            }),
        }

    @app.post("/api/settings/routing")
    async def api_settings_routing_save(body: dict):
        """Save routing mode and cascade model config."""
        from ..config import save_config

        # Update cascade_routing flag
        if "cascade_routing" in body:
            agent.config.setdefault("cost", {})["cascade_routing"] = bool(body["cascade_routing"])
            agent.cascade_routing = bool(body["cascade_routing"])

        # Update cascade models
        if "models" in body and isinstance(body["models"], dict):
            models = body["models"]
            agent_models = agent.config.setdefault("agent", {}).setdefault("models", {})
            for level in ("simple", "medium", "complex"):
                if level in models and models[level]:
                    agent_models[level] = models[level]
            agent.models = agent_models

        # Update local-only hours schedule
        if "local_only_hours" in body and isinstance(body["local_only_hours"], dict):
            loh = body["local_only_hours"]
            schedule = agent.config.setdefault("cost", {}).setdefault("local_only_hours", {})
            if "enabled" in loh:
                schedule["enabled"] = bool(loh["enabled"])
            if "start" in loh:
                schedule["start"] = loh["start"]
            if "end" in loh:
                schedule["end"] = loh["end"]

        save_config(agent.config)
        logger.info("Routing config saved: cascade=%s", agent.cascade_routing)
        return {"ok": True}

    # ── Planning (Internal Monologue) Settings ─

    @app.get("/api/settings/planning")
    async def api_settings_planning():
        """Get internal monologue / planning configuration."""
        im = agent.config.get("features", {}).get("internal_monologue", {})
        return {
            "enabled": im.get("enabled", False),
            "planning_model": im.get("planning_model", "auto"),
            "skip_simple": im.get("skip_simple", True),
            "reflect_every_n_tools": im.get("reflect_every_n_tools", 3),
        }

    @app.post("/api/settings/planning")
    async def api_settings_planning_save(body: dict):
        """Save planning / internal monologue settings."""
        from ..config import save_config

        features = agent.config.setdefault("features", {})
        im = features.setdefault("internal_monologue", {})

        if "enabled" in body:
            im["enabled"] = bool(body["enabled"])
        if "planning_model" in body:
            im["planning_model"] = str(body["planning_model"]).strip() or "auto"
        if "skip_simple" in body:
            im["skip_simple"] = bool(body["skip_simple"])
        if "reflect_every_n_tools" in body:
            val = body["reflect_every_n_tools"]
            try:
                im["reflect_every_n_tools"] = max(1, min(int(val), 10))
            except (ValueError, TypeError):
                pass

        save_config(agent.config)
        logger.info("Planning config saved: enabled=%s, model=%s",
                     im.get("enabled"), im.get("planning_model"))
        return {"ok": True}

    # ── Telegram Settings ───────────────────

    @app.get("/api/settings/telegram")
    async def api_settings_telegram():
        """Get Telegram bot configuration status."""
        from ..config import get_api_key, key_preview
        tg_cfg = agent.config.get("channels", {}).get("telegram", {})
        token = tg_cfg.get("token") or get_api_key("telegram")
        chat_id = tg_cfg.get("chat_id", "")
        return {
            "configured": bool(token),
            "token_preview": key_preview(token) if token else "",
            "chat_id": str(chat_id) if chat_id else "",
            "mode": tg_cfg.get("mode", "polling"),
            "webhook_url": tg_cfg.get("webhook_url", ""),
            "voice_transcription": tg_cfg.get("voice_transcription", "auto"),
        }

    @app.post("/api/settings/telegram")
    async def api_settings_telegram_save(body: dict):
        """Save Telegram bot token and chat_id."""
        from ..config import save_provider_key, save_config, key_preview
        token = body.get("token", "").strip()
        chat_id = body.get("chat_id", "").strip()

        if not token:
            raise HTTPException(status_code=400, detail="Token required")
        if not token.count(":") == 1 or not token.split(":")[0].isdigit():
            raise HTTPException(status_code=400,
                detail="Invalid token format. Get it from @BotFather in Telegram")

        # Save token to keys.json under "telegram" key
        save_provider_key("telegram", token)

        # Update runtime config
        tg = agent.config.setdefault("channels", {}).setdefault("telegram", {})
        tg["token"] = token
        tg["enabled"] = True
        if chat_id:
            tg["chat_id"] = chat_id
        elif "chat_id" in tg:
            del tg["chat_id"]

        # Persist to config.json (without token — it's in keys.json)
        save_config(agent.config)
        logger.info("Telegram config saved (chat_id: %s)", chat_id or "all")

        return {"ok": True, "token_preview": key_preview(token)}

    @app.post("/api/settings/telegram/test")
    async def api_settings_telegram_test(body: dict):
        """Test Telegram bot token validity."""
        from ..config import get_api_key
        import urllib.request

        token = body.get("token", "").strip()
        if not token:
            token = agent.config.get("channels", {}).get("telegram", {}).get("token")
        if not token:
            token = get_api_key("telegram")
        if not token:
            return {"ok": False, "error": "No token provided or saved"}

        try:
            url = f"https://api.telegram.org/bot{token}/getMe"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            if data.get("ok"):
                bot = data["result"]
                return {
                    "ok": True,
                    "bot_name": bot.get("first_name", ""),
                    "bot_username": bot.get("username", ""),
                }
            return {"ok": False, "error": data.get("description", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.delete("/api/settings/telegram")
    async def api_settings_telegram_delete():
        """Remove saved Telegram token."""
        from ..config import delete_provider_key
        deleted = delete_provider_key("telegram")
        agent.config.get("channels", {}).get("telegram", {}).pop("token", None)
        if not deleted:
            raise HTTPException(status_code=404, detail="No token saved")
        return {"ok": True}

    # ── Voice Settings (TTS + STT) ─────────

    @app.get("/api/settings/voice")
    async def api_settings_voice():
        """Get full voice settings: TTS + STT configuration."""
        from ..voice import resolve_voice_config, _get_tts_api_key, get_last_tts_attempt
        from ..voice import OPENAI_TTS_VOICES, OPENAI_TTS_MODELS, STT_PROVIDERS

        voice_cfg = resolve_voice_config(agent.config)
        tts = voice_cfg["tts"]
        stt = voice_cfg["stt"]

        # STT backend detection (legacy compat)
        tg_cfg = agent.config.get("channels", {}).get("telegram", {})
        stt_mode = tg_cfg.get("voice_transcription", "auto")
        has_builtin = "transcribe_voice" in agent.tools._tools
        mcp_tools = [n for n in agent.tools._tools
                     if "transcribe" in n and "__" in n]
        has_mcp = bool(mcp_tools)
        if stt_mode == "builtin" or (stt_mode == "auto" and not has_mcp):
            stt_active = "builtin"
        elif stt_mode == "mcp" or (stt_mode == "auto" and has_mcp):
            stt_active = "mcp"
        else:
            stt_active = "builtin"

        return {
            # TTS settings
            "tts": {
                "auto": tts["auto"],
                "provider": tts["provider"],
                "max_length": tts["max_length"],
                "has_openai": bool(_get_tts_api_key("openai", agent.config)),
                "has_elevenlabs": bool(_get_tts_api_key("elevenlabs", agent.config)),
                "has_edge": tts["edge"]["enabled"],
                "openai": tts["openai"],
                "elevenlabs": tts["elevenlabs"],
                "edge": tts["edge"],
                "voices": list(OPENAI_TTS_VOICES),
                "models": list(OPENAI_TTS_MODELS),
                "last_attempt": get_last_tts_attempt(),
            },
            # STT settings
            "stt": {
                "mode": stt_mode,
                "active": stt_active,
                "provider": stt["provider"],
                "providers": list(STT_PROVIDERS),
                "has_builtin": has_builtin or stt_mode == "builtin",
                "has_mcp": has_mcp or (agent._mcp_config and
                                       any("whisper" in k.lower() or "transcri" in k.lower()
                                           for k in agent._mcp_config)),
                "mcp_tools": mcp_tools,
                "openai": stt["openai"],
                "deepgram": stt["deepgram"],
                "groq": stt["groq"],
            },
        }

    @app.post("/api/settings/voice")
    async def api_settings_voice_save(body: dict):
        """Save voice settings: TTS + STT configuration."""
        from ..config import save_config

        voice = agent.config.setdefault("voice", {})

        # TTS settings
        if "tts" in body:
            tts_data = body["tts"]
            tts = voice.setdefault("tts", {})
            if "auto" in tts_data:
                if tts_data["auto"] not in ("off", "always", "inbound", "tagged"):
                    raise HTTPException(400, "auto must be: off, always, inbound, or tagged")
                tts["auto"] = tts_data["auto"]
            if "provider" in tts_data:
                if tts_data["provider"] not in ("openai", "elevenlabs", "edge"):
                    raise HTTPException(400, "provider must be: openai, elevenlabs, or edge")
                tts["provider"] = tts_data["provider"]
            if "max_length" in tts_data:
                tts["max_length"] = int(tts_data["max_length"])
            if "openai" in tts_data:
                tts["openai"] = {**tts.get("openai", {}), **tts_data["openai"]}
            if "elevenlabs" in tts_data:
                tts["elevenlabs"] = {**tts.get("elevenlabs", {}), **tts_data["elevenlabs"]}
            if "edge" in tts_data:
                tts["edge"] = {**tts.get("edge", {}), **tts_data["edge"]}

        # STT settings
        if "stt" in body:
            stt_data = body["stt"]
            stt = voice.setdefault("stt", {})
            if "provider" in stt_data:
                if stt_data["provider"] not in ("openai", "deepgram", "groq"):
                    raise HTTPException(400, "stt provider must be: openai, deepgram, or groq")
                stt["provider"] = stt_data["provider"]
            if "openai" in stt_data:
                stt["openai"] = {**stt.get("openai", {}), **stt_data["openai"]}
            if "deepgram" in stt_data:
                stt["deepgram"] = {**stt.get("deepgram", {}), **stt_data["deepgram"]}
            if "groq" in stt_data:
                stt["groq"] = {**stt.get("groq", {}), **stt_data["groq"]}

        # Legacy STT mode (voice_transcription in telegram config)
        if "mode" in body:
            mode = body["mode"]
            if mode not in ("auto", "builtin", "mcp"):
                raise HTTPException(400, "Mode must be: auto, builtin, or mcp")
            tg = agent.config.setdefault("channels", {}).setdefault("telegram", {})
            tg["voice_transcription"] = mode
            if "transcribe_voice" not in agent.tools._tools:
                agent._wire_voice_tool()
            agent._apply_voice_transcription_mode()

        save_config(agent.config)
        return {"ok": True}

    @app.post("/api/tts/test")
    async def api_tts_test(body: dict):
        """Test TTS: convert text to audio and return base64."""
        import base64
        text = body.get("text", "").strip()
        if not text:
            raise HTTPException(400, "text is required")

        from ..voice import text_to_speech, resolve_voice_config
        voice_cfg = resolve_voice_config(agent.config)
        result = await text_to_speech(text, voice_cfg, agent.config, channel="api")

        if not result.success:
            return {"ok": False, "error": result.error}

        with open(result.audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()

        # Cleanup
        try:
            os.unlink(result.audio_path)
        except OSError:
            pass

        return {
            "ok": True,
            "audio": audio_b64,
            "format": result.output_format or "mp3",
            "provider": result.provider,
            "latency_ms": result.latency_ms,
        }

    # ── RAG Document Management ─────────────

    @app.get("/api/rag/documents")
    async def api_rag_documents():
        """List ingested RAG documents."""
        rag = getattr(agent, '_rag', None)
        if not rag:
            return []
        docs = rag.list_documents()
        stats = rag.get_stats()
        return {"documents": docs, "stats": stats}

    @app.post("/api/rag/ingest")
    async def api_rag_ingest(path: str):
        """Ingest a file or directory into RAG."""
        rag = getattr(agent, '_rag', None)
        if not rag:
            raise HTTPException(status_code=400, detail="RAG is not enabled")
        try:
            result = rag.ingest(path)
            return result
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/rag/documents/{doc_id}")
    async def api_rag_delete(doc_id: int):
        """Delete an ingested RAG document."""
        rag = getattr(agent, '_rag', None)
        if not rag:
            raise HTTPException(status_code=400, detail="RAG is not enabled")
        ok = rag.delete_document(doc_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "deleted"}

    # ── Storage Management ──────────────────

    @app.get("/api/storage/status")
    async def api_storage_status():
        """Get storage backend status."""
        storage = getattr(agent, '_storage', None)
        if not storage:
            return {"enabled": False}
        try:
            stats = storage.get_stats()
            return {"enabled": True, "connected": True, **stats}
        except Exception as e:
            return {"enabled": True, "connected": False, "error": str(e)}

    @app.get("/api/storage/files")
    async def api_storage_files(prefix: str = "", limit: int = 100):
        """List files in storage."""
        storage = getattr(agent, '_storage', None)
        if not storage:
            return []
        return storage.list_files(prefix=prefix, limit=limit)

    @app.post("/api/storage/upload")
    async def api_storage_upload(body: dict):
        """Upload file content to storage."""
        storage = getattr(agent, '_storage', None)
        if not storage:
            raise HTTPException(status_code=400, detail="Storage not enabled")
        key = body.get("key", "").strip()
        content = body.get("content", "")
        if not key:
            raise HTTPException(status_code=400, detail="File key required")
        try:
            data = content.encode("utf-8") if isinstance(content, str) else content
            storage.upload(key, data)
            return {"ok": True, "key": key, "size": len(data)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/storage/files/{key:path}")
    async def api_storage_delete(key: str):
        """Delete a file from storage."""
        storage = getattr(agent, '_storage', None)
        if not storage:
            raise HTTPException(status_code=400, detail="Storage not enabled")
        ok = storage.delete(key)
        if not ok:
            raise HTTPException(status_code=500, detail="Delete failed")
        return {"ok": True}

    # ── File Manager (index + search + browse) ────

    @app.get("/api/files")
    async def api_files_list(source: str = "", user_id: str = "",
                             limit: int = 100, offset: int = 0):
        """List indexed files with optional filters."""
        fm = getattr(agent, '_file_manager', None)
        if not fm:
            raise HTTPException(400, "File manager not enabled (enable storage)")
        return fm.list_files(
            user_id=user_id or None, source=source or None,
            limit=limit, offset=offset)

    @app.get("/api/files/search")
    async def api_files_search(q: str, limit: int = 20):
        """Semantic search through indexed files."""
        fm = getattr(agent, '_file_manager', None)
        if not fm:
            raise HTTPException(400, "File manager not enabled")
        return fm.search(q, top_k=limit)

    @app.get("/api/files/count")
    async def api_files_count():
        """Get total file count."""
        fm = getattr(agent, '_file_manager', None)
        if not fm:
            return {"count": 0}
        return {"count": fm.count_files()}

    @app.get("/api/files/download/{key:path}")
    async def api_files_download(key: str, expires: int = 3600):
        """Get presigned download URL for a file."""
        fm = getattr(agent, '_file_manager', None)
        if not fm:
            raise HTTPException(400, "File manager not enabled")
        try:
            url = await fm.get_download_url(key, expires=expires)
            from starlette.responses import RedirectResponse
            return RedirectResponse(url=url)
        except Exception as e:
            raise HTTPException(404, f"File not found: {e}")

    @app.get("/api/files/url/{key:path}")
    async def api_files_url(key: str, expires: int = 3600):
        """Get presigned download URL (JSON, no redirect)."""
        fm = getattr(agent, '_file_manager', None)
        if not fm:
            raise HTTPException(400, "File manager not enabled")
        try:
            url = await fm.get_download_url(key, expires=expires)
            return {"url": url, "expires_sec": expires}
        except Exception as e:
            raise HTTPException(404, str(e))

    @app.get("/files")
    async def files_browse_page():
        """File browser HTML page with download links."""
        fm = getattr(agent, '_file_manager', None)
        storage = getattr(agent, '_storage', None)
        if not fm or not storage:
            from starlette.responses import HTMLResponse
            return HTMLResponse("<h2>Storage not enabled</h2><p>Enable S3/MinIO in Settings → Storage</p>")
        from starlette.responses import HTMLResponse
        return HTMLResponse(_FILE_BROWSER_HTML)

    # ── Storage Settings ──────────────────

    @app.get("/api/settings/storage")
    async def api_settings_storage():
        """Get storage configuration status."""
        from ..config import get_api_key, key_preview
        storage_cfg = agent.config.get("storage", {})
        storage = getattr(agent, '_storage', None)
        access_key = storage_cfg.get("access_key") or get_api_key("minio_access") or ""
        return {
            "enabled": storage_cfg.get("enabled", False),
            "connected": storage is not None,
            "endpoint": storage_cfg.get("endpoint", ""),
            "bucket": storage_cfg.get("bucket", "liteagent"),
            "access_key_preview": key_preview(access_key) if access_key else "",
        }

    @app.post("/api/settings/storage")
    async def api_settings_storage_save(body: dict):
        """Save storage configuration."""
        from ..config import save_provider_key
        from ..storage import create_storage

        endpoint = body.get("endpoint", "").strip()
        access_key = body.get("access_key", "").strip()
        secret_key = body.get("secret_key", "").strip()
        bucket = body.get("bucket", "liteagent").strip()

        if not endpoint:
            raise HTTPException(status_code=400, detail="Endpoint required")
        if not access_key or not secret_key:
            raise HTTPException(status_code=400, detail="Access key and secret key required")

        # Save credentials
        save_provider_key("minio_access", access_key)
        save_provider_key("minio_secret", secret_key)

        # Update config
        agent.config.setdefault("storage", {}).update({
            "enabled": True, "endpoint": endpoint,
            "access_key": access_key, "secret_key": secret_key,
            "bucket": bucket,
        })

        # Persist config to disk
        from ..config import save_config
        save_config(agent.config)

        # Reconnect storage
        try:
            agent._storage = create_storage(agent.config)
            if agent._storage:
                agent._wire_storage_tools()
            return {"ok": True, "connected": agent._storage is not None}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/settings/storage/test")
    async def api_settings_storage_test(body: dict):
        """Test storage connection."""
        endpoint = body.get("endpoint", "").strip()
        access_key = body.get("access_key", "").strip()
        secret_key = body.get("secret_key", "").strip()
        bucket = body.get("bucket", "liteagent").strip()

        if not endpoint or not access_key or not secret_key:
            return {"ok": False, "error": "Missing credentials"}

        try:
            from ..storage import StorageBackend
            import time
            start = time.time()
            s = StorageBackend({
                "endpoint": endpoint, "access_key": access_key,
                "secret_key": secret_key, "bucket": bucket,
            })
            stats = s.get_stats()
            latency_ms = int((time.time() - start) * 1000)
            return {"ok": True, "latency_ms": latency_ms, **stats}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Vector Search Settings ────────────────────

    @app.get("/api/settings/vector")
    async def api_settings_vector():
        """Get vector search configuration + status."""
        from ..config import get_api_key, key_preview
        rag_cfg = agent.config.get("rag", {})
        rag = getattr(agent, '_rag', None)
        stats = rag.get_stats() if rag else {}

        # Embedding config
        emb_cfg = rag_cfg.get("embedding", {})

        # Qdrant config
        qdrant_cfg = rag_cfg.get("qdrant", {})
        qdrant_key = qdrant_cfg.get("api_key") or get_api_key("qdrant") or ""

        # Search config
        search_cfg = rag_cfg.get("search", {})

        # File indexing config
        fi_cfg = rag_cfg.get("file_indexing", {})

        return {
            "enabled": rag_cfg.get("enabled", False),
            "vector_backend": rag_cfg.get("vector_backend", "auto"),
            "chunk_size": rag_cfg.get("chunk_size", 1000),
            "chunk_overlap": rag_cfg.get("overlap", 200),
            "embedding": {
                "provider": emb_cfg.get("provider", "auto"),
                "model": emb_cfg.get("model", "nomic-embed-text"),
                "openai_model": emb_cfg.get("openai_model", "text-embedding-3-small"),
                "dimension": emb_cfg.get("dimension"),
            },
            "search": {
                "mode": search_cfg.get("mode", "hybrid"),
                "rrf_k": search_cfg.get("rrf_k", 60),
                "vector_top_k": search_cfg.get("vector_top_k", 50),
                "keyword_top_k": search_cfg.get("keyword_top_k", 50),
            },
            "qdrant": {
                "url": qdrant_cfg.get("url", ""),
                "collection": qdrant_cfg.get("collection", "liteagent_rag"),
                "api_key_preview": key_preview(qdrant_key) if qdrant_key else "",
            },
            "file_indexing": {
                "enabled": fi_cfg.get("enabled", True),
                "max_file_size_mb": fi_cfg.get("max_file_size_mb", 10),
            },
            "stats": stats,
        }

    @app.post("/api/settings/vector")
    async def api_settings_vector_save(body: dict):
        """Save vector search configuration and reinitialize RAG."""
        from ..config import save_config, save_provider_key
        from ..rag import RAGPipeline

        rag_cfg = agent.config.setdefault("rag", {})
        rag_cfg["enabled"] = True
        rag_cfg["vector_backend"] = body.get("vector_backend", "auto")
        rag_cfg["chunk_size"] = max(200, min(4000, int(body.get("chunk_size", 1000))))
        rag_cfg["overlap"] = max(0, min(1000, int(body.get("chunk_overlap", 200))))

        # Embedding config
        emb = body.get("embedding", {})
        rag_cfg["embedding"] = {
            "provider": emb.get("provider", "auto"),
            "model": emb.get("model", "nomic-embed-text"),
            "openai_model": emb.get("openai_model", "text-embedding-3-small"),
        }
        if emb.get("dimension"):
            rag_cfg["embedding"]["dimension"] = int(emb["dimension"])

        # Search config
        search = body.get("search", {})
        rag_cfg["search"] = {
            "mode": search.get("mode", "hybrid"),
            "rrf_k": int(search.get("rrf_k", 60)),
            "vector_top_k": int(search.get("vector_top_k", 50)),
            "keyword_top_k": int(search.get("keyword_top_k", 50)),
        }

        # Qdrant config
        qdrant = body.get("qdrant", {})
        qdrant_url = qdrant.get("url", "").strip()
        qdrant_key = qdrant.get("api_key", "").strip()
        qdrant_collection = qdrant.get("collection", "liteagent_rag").strip()
        if qdrant_url:
            rag_cfg["qdrant"] = {
                "url": qdrant_url,
                "collection": qdrant_collection,
            }
            if qdrant_key:
                save_provider_key("qdrant", qdrant_key)

        # File indexing config
        fi = body.get("file_indexing", {})
        rag_cfg["file_indexing"] = {
            "enabled": fi.get("enabled", True),
            "max_file_size_mb": int(fi.get("max_file_size_mb", 10)),
        }

        save_config(agent.config)

        # Reinitialize RAG pipeline
        try:
            agent._rag = RAGPipeline(
                agent.memory.db,
                embedder=agent.memory._embedder,
                config=rag_cfg)
            agent._rag.init_backend(agent.config)
            agent._wire_rag_tool()
            # Reconnect FileManager to new RAG
            if agent._file_manager:
                agent._file_manager._rag = agent._rag
            stats = agent._rag.get_stats()
            return {"ok": True, "stats": stats}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/settings/vector/test")
    async def api_settings_vector_test(body: dict):
        """Test Qdrant connection (when backend=qdrant)."""
        url = body.get("url", "").strip()
        api_key = body.get("api_key", "").strip()

        if not url:
            return {"ok": False, "error": "URL required"}

        try:
            from qdrant_client import QdrantClient
            import time
            start = time.time()
            client = QdrantClient(url=url, api_key=api_key or None, timeout=5)
            collections = client.get_collections()
            latency_ms = int((time.time() - start) * 1000)
            names = [c.name for c in collections.collections]
            return {"ok": True, "latency_ms": latency_ms, "collections": names}
        except ImportError:
            return {"ok": False, "error": "qdrant-client not installed. Run: pip install liteagent[qdrant]"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/settings/vector/reindex")
    async def api_settings_vector_reindex():
        """Reindex all files from FileManager through RAG pipeline."""
        rag = getattr(agent, '_rag', None)
        fm = getattr(agent, '_file_manager', None)
        if not rag:
            return {"ok": False, "error": "RAG not enabled"}

        files = fm.list_files(limit=500) if fm else []
        indexed = 0
        errors = []
        for f in files:
            try:
                key = f["storage_key"]
                data = await agent._storage.async_download(key)
                if data:
                    text = fm._extract_text(data, f["mime_type"])
                    if text and len(text.strip()) > 20:
                        ext = Path(f["original_name"]).suffix
                        rag.index_content(text, source_key=key,
                                         source_name=f["original_name"],
                                         file_type=ext)
                        indexed += 1
            except Exception as e:
                errors.append(f"{f['original_name']}: {e}")

        return {"ok": True, "indexed": indexed, "errors": errors[:10],
                "stats": rag.get_stats()}

    # ── Knowledge Base Management ─────────

    @app.get("/api/settings/knowledge_base")
    async def api_settings_knowledge_base():
        """Get knowledge base configuration + status."""
        kb_cfg = agent.config.get("knowledge_base", {})
        kb = getattr(agent, '_knowledge_base', None)
        stats = {}
        if kb:
            try:
                stats = await kb.get_stats()
            except Exception:
                pass
        return {
            "enabled": kb_cfg.get("enabled", False),
            "chunk_size": kb_cfg.get("chunk_size", 800),
            "chunk_overlap": kb_cfg.get("chunk_overlap", 150),
            "search_mode": kb_cfg.get("search_mode", "hybrid"),
            "rerank": kb_cfg.get("rerank", True),
            "rerank_model": kb_cfg.get("rerank_model",
                                        "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            "query_rewrite": kb_cfg.get("query_rewrite", True),
            "max_file_size_mb": kb_cfg.get("max_file_size_mb", 50),
            "db_path": kb_cfg.get("db_path", "~/.liteagent/knowledge_base.db"),
            "stats": stats,
        }

    @app.post("/api/settings/knowledge_base")
    async def api_settings_knowledge_base_save(body: dict):
        """Save knowledge base settings and reinitialize."""
        from ..config import save_config

        kb_cfg = agent.config.setdefault("knowledge_base", {})
        kb_cfg["enabled"] = body.get("enabled", False)
        kb_cfg["chunk_size"] = max(200, min(2000, int(body.get("chunk_size", 800))))
        kb_cfg["chunk_overlap"] = max(0, min(500, int(body.get("chunk_overlap", 150))))
        kb_cfg["search_mode"] = body.get("search_mode", "hybrid")
        kb_cfg["rerank"] = body.get("rerank", True)
        if body.get("rerank_model"):
            kb_cfg["rerank_model"] = body["rerank_model"]
        kb_cfg["query_rewrite"] = body.get("query_rewrite", True)
        kb_cfg["max_file_size_mb"] = max(1, min(200, int(body.get("max_file_size_mb", 50))))

        save_config(agent.config)

        try:
            if kb_cfg["enabled"]:
                agent._init_knowledge_base(kb_cfg)
                stats = {}
                if agent._knowledge_base:
                    stats = await agent._knowledge_base.get_stats()
                return {"ok": True, "stats": stats}
            else:
                agent._knowledge_base = None
                return {"ok": True, "stats": {}}
        except Exception as e:
            logger.warning("KB reinit failed: %s", e)
            return {"ok": False, "error": str(e)}

    @app.get("/api/knowledge_base/documents")
    async def api_kb_documents():
        """List documents in the knowledge base."""
        kb = getattr(agent, '_knowledge_base', None)
        if not kb:
            return {"documents": [], "stats": {}}
        docs = await kb.list_documents()
        stats = await kb.get_stats()
        return {"documents": docs, "stats": stats}

    @app.post("/api/knowledge_base/ingest")
    async def api_kb_ingest(body: dict):
        """Ingest a document into the knowledge base."""
        kb = getattr(agent, '_knowledge_base', None)
        if not kb:
            return JSONResponse(status_code=400,
                                content={"detail": "Knowledge Base is not enabled"})
        path = body.get("path", "").strip()
        if not path:
            return JSONResponse(status_code=400,
                                content={"detail": "Path is required"})
        try:
            result = await kb.ingest(path)
            return result
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    @app.delete("/api/knowledge_base/documents/{doc_id}")
    async def api_kb_delete(doc_id: str):
        """Delete a document from the knowledge base."""
        kb = getattr(agent, '_knowledge_base', None)
        if not kb:
            return JSONResponse(status_code=400,
                                content={"detail": "Knowledge Base is not enabled"})
        ok = await kb.delete_document(doc_id)
        if not ok:
            return JSONResponse(status_code=404,
                                content={"detail": "Document not found"})
        return {"status": "deleted"}

    @app.post("/api/knowledge_base/search")
    async def api_kb_search(body: dict):
        """Test search against the knowledge base."""
        kb = getattr(agent, '_knowledge_base', None)
        if not kb:
            return JSONResponse(status_code=400,
                                content={"detail": "Knowledge Base is not enabled"})
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse(status_code=400,
                                content={"detail": "Query required"})
        top_k = int(body.get("top_k", 6))
        mode = body.get("mode")
        results = await kb.search(query, top_k=top_k, mode=mode)
        return {
            "results": [
                {
                    "content": r.content[:500],
                    "score": round(r.score, 4),
                    "source": r.source,
                    "page": r.page,
                    "section": r.section,
                }
                for r in results
            ],
            "count": len(results),
        }

    @app.get("/api/knowledge_base/query_log")
    async def api_kb_query_log(limit: int = 20):
        """Get recent query log entries."""
        kb = getattr(agent, '_knowledge_base', None)
        if not kb:
            return {"queries": []}
        import json as _json
        try:
            rows = kb.db.execute(
                "SELECT query, rewritten_queries, result_count, latency_ms, "
                "created_at FROM kb_query_log ORDER BY id DESC LIMIT ?",
                (min(limit, 100),)).fetchall()
        except Exception:
            return {"queries": []}
        queries = []
        for row in rows:
            queries.append({
                "query": row[0],
                "sub_queries": _json.loads(row[1]) if row[1] else [],
                "result_count": row[2],
                "latency_ms": row[3],
                "created_at": row[4],
            })
        return {"queries": queries}

    # ── Night Worker ───────────────────────

    @app.get("/api/settings/night_worker")
    async def api_settings_night_worker():
        """Get night worker config + queue stats."""
        try:
            nw_cfg = agent.config.get("night_worker", {})

            # Get queue stats if KB is available
            stats = {}
            kb = getattr(agent, '_knowledge_base', None)
            if kb:
                try:
                    from ..night_worker import NightWorker
                    worker = NightWorker(nw_cfg, kb.db)
                    stats = worker.get_queue_stats()
                except Exception:
                    pass

            return JSONResponse({
                "enabled": nw_cfg.get("enabled", False),
                "model": nw_cfg.get("model", "qwen2.5:latest"),
                "batch_size": nw_cfg.get("batch_size", 20),
                "max_tasks_per_run": nw_cfg.get("max_tasks_per_run", 200),
                "max_runtime_sec": nw_cfg.get("max_runtime_sec", 3600),
                "cron": nw_cfg.get("cron", "0 22 * * *"),
                "queue_stats": stats,
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/settings/night_worker")
    async def api_settings_night_worker_save(body: dict):
        """Save night worker settings."""
        try:
            from ..config import save_config

            nw_cfg = agent.config.get("night_worker", {})
            for key in ("enabled", "model", "batch_size", "max_tasks_per_run",
                         "max_runtime_sec", "cron"):
                if key in body:
                    nw_cfg[key] = body[key]
            agent.config["night_worker"] = nw_cfg
            save_config(agent.config)
            return JSONResponse({"status": "ok"})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/night_worker/run")
    async def api_night_worker_run():
        """Manually trigger night worker."""
        try:
            nw_cfg = agent.config.get("night_worker", {})
            kb = getattr(agent, '_knowledge_base', None)
            if not kb:
                return JSONResponse({"error": "Knowledge base not configured"},
                                    status_code=400)

            from ..night_worker import NightWorker

            # Try to create an Ollama provider
            provider = None
            try:
                from ..providers import OllamaProvider
                model = nw_cfg.get("model", "qwen2.5:latest")
                ollama_cfg = agent.config.get("providers", {}).get("ollama", {})
                base_url = ollama_cfg.get("base_url", "http://localhost:11434")
                provider = OllamaProvider({"base_url": base_url},
                                          default_model=model)
            except Exception:
                pass

            embedder = getattr(kb, '_embedder', None)
            worker = NightWorker(nw_cfg, kb.db, provider=provider,
                                 embedder=embedder)
            result = await worker.run()
            return JSONResponse(result)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.post("/api/night_worker/enqueue")
    async def api_night_worker_enqueue():
        """Enqueue unenriched chunks for night processing."""
        try:
            nw_cfg = agent.config.get("night_worker", {})
            kb = getattr(agent, '_knowledge_base', None)
            if not kb:
                return JSONResponse({"error": "Knowledge base not configured"},
                                    status_code=400)

            from ..night_worker import NightWorker
            worker = NightWorker(nw_cfg, kb.db)
            counts = worker.enqueue_unenriched()
            return JSONResponse({"status": "ok", "enqueued": counts})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ── KB Quality Metrics ─────────────────

    @app.get("/api/knowledge_base/quality")
    async def api_kb_quality():
        """Get KB quality metrics."""
        try:
            kb = getattr(agent, '_knowledge_base', None)
            if not kb:
                return JSONResponse({"error": "Knowledge base not configured"},
                                    status_code=400)

            stats = await kb.get_quality_stats()
            return JSONResponse(stats)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    # ── MCP Server Management ──────────────

    @app.get("/api/mcp/config")
    async def api_mcp_config():
        """Get MCP server configuration (from config.json)."""
        mcp_cfg = agent.config.get("tools", {}).get("mcp_servers", {})
        servers = []
        connected = agent.tools.get_mcp_server_info()
        connected_names = {s.get("name", "") for s in connected}
        for name, cfg in mcp_cfg.items():
            servers.append({
                "name": name,
                "command": cfg.get("command", ""),
                "args": cfg.get("args", []),
                "env": {k: "***" for k in cfg.get("env", {})},
                "enabled": cfg.get("enabled", True),
                "connected": name in connected_names,
            })
        return servers

    @app.post("/api/mcp/servers")
    async def api_mcp_add_server(body: dict):
        """Add an MCP server to config and reload."""
        name = body.get("name", "").strip()
        command = body.get("command", "").strip()
        args = body.get("args", [])
        env = body.get("env", {})

        if not name:
            raise HTTPException(status_code=400, detail="Server name required")
        if not command:
            raise HTTPException(status_code=400, detail="Command required")

        # Support JSON mode: if body has "json_config", parse it
        json_config = body.get("json_config", "").strip()
        if json_config:
            try:
                parsed = json.loads(json_config)
                if isinstance(parsed, dict):
                    # Could be a single server config or {name: config}
                    if "command" in parsed:
                        command = parsed["command"]
                        args = parsed.get("args", [])
                        env = parsed.get("env", {})
                    else:
                        # Format: {"server_name": {"command": ..., "args": ...}}
                        for srv_name, srv_cfg in parsed.items():
                            name = srv_name
                            command = srv_cfg.get("command", "")
                            args = srv_cfg.get("args", [])
                            env = srv_cfg.get("env", {})
                            break
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        # Add to config
        agent.config.setdefault("tools", {}).setdefault("mcp_servers", {})
        agent.config["tools"]["mcp_servers"][name] = {
            "command": command, "args": args, "env": env,
        }
        agent._mcp_config = agent.config["tools"]["mcp_servers"]

        # Reload MCP
        try:
            await agent.reload_mcp()
            servers = agent.tools.get_mcp_server_info()
            return {"ok": True, "servers": servers}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.delete("/api/mcp/servers/{name}")
    async def api_mcp_delete_server(name: str):
        """Remove an MCP server from config and reload."""
        mcp_cfg = agent.config.get("tools", {}).get("mcp_servers", {})
        if name not in mcp_cfg:
            raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
        del mcp_cfg[name]
        agent._mcp_config = mcp_cfg
        try:
            await agent.reload_mcp()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/mcp/servers/{name}/toggle")
    async def api_mcp_toggle_server(name: str, body: dict):
        """Enable or disable an MCP server without removing it."""
        mcp_cfg = agent.config.get("tools", {}).get("mcp_servers", {})
        if name not in mcp_cfg:
            raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
        enabled = body.get("enabled", True)
        mcp_cfg[name]["enabled"] = bool(enabled)
        agent._mcp_config = mcp_cfg
        try:
            await agent.reload_mcp()
            return {"ok": True, "enabled": bool(enabled)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Logs Viewer ─────────────────────────

    @app.get("/api/logs")
    async def api_logs(level: str = "all", limit: int = 50, search: str = ""):
        """Read recent structured log entries."""
        try:
            from ..logging_config import read_log_lines
            entries = read_log_lines(limit=min(limit, 200))
            # Filter by level
            if level and level != "all":
                entries = [e for e in entries if e.get("level", "").upper() == level.upper()]
            # Filter by search
            if search:
                search_l = search.lower()
                entries = [e for e in entries
                           if search_l in e.get("message", "").lower()
                           or search_l in e.get("module", "").lower()]
            return entries
        except Exception as e:
            logger.warning("Log read failed: %s", e)
            return []

    # ── Backup Management ─────────────────────

    @app.get("/api/backups")
    async def api_backups():
        """List available backups."""
        from ..backup import list_backups
        return list_backups()

    @app.post("/api/backup")
    async def api_backup_create():
        """Create a new backup."""
        from ..backup import backup
        config_path = agent.config.get("_config_path")
        path = backup(config_path)
        return {"ok": True, "path": str(path), "name": path.name}

    @app.get("/api/backup/download")
    async def api_backup_download(name: str = ""):
        """Download a backup file."""
        from ..backup import BACKUP_DIR
        if not name:
            # Download latest
            from ..backup import list_backups
            backups = list_backups()
            if not backups:
                raise HTTPException(404, "No backups available")
            name = backups[0]["name"]
        backup_path = BACKUP_DIR / name
        if not backup_path.exists() or ".." in name:
            raise HTTPException(404, "Backup not found")
        return FileResponse(str(backup_path), filename=name,
                            media_type="application/gzip")

    # ── Scheduler Run Now ─────────────────────

    @app.post("/api/ops/scheduler/{name}/run")
    async def api_scheduler_run_now(name: str):
        """Trigger immediate execution of a scheduled job."""
        sched = getattr(agent, '_scheduler', None)
        if not sched:
            raise HTTPException(400, "Scheduler not running")
        result = await sched.run_now(name)
        if result.get("error"):
            raise HTTPException(400, result["error"])
        return result

    # ── Config Reload ─────────────────────────

    @app.post("/api/config/reload")
    async def api_config_reload():
        """Force hot-reload of config.json."""
        watcher = getattr(app.state, 'config_watcher', None)
        if not watcher:
            raise HTTPException(400, "Config watcher not running")
        changes = await watcher.force_reload()
        return {"ok": True, "changes": changes}

    # ── Tasks API ──────────────────────────────

    @app.get("/api/tasks")
    async def api_tasks_list(status: str = None):
        """List all user tasks."""
        tm = getattr(agent, '_task_manager', None)
        if not tm:
            return []
        return tm.list_tasks(status=status)

    @app.get("/api/tasks/{task_id}")
    async def api_task_detail(task_id: int):
        """Get a single task."""
        tm = getattr(agent, '_task_manager', None)
        if not tm:
            raise HTTPException(404, "Tasks not available")
        task = tm.get_task(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        return task

    @app.post("/api/tasks")
    async def api_task_create(body: dict):
        """Create a task from the dashboard."""
        tm = getattr(agent, '_task_manager', None)
        if not tm:
            raise HTTPException(400, "Tasks not available")
        name = body.get("name", "").strip()
        query = body.get("query", "").strip()
        run_at = body.get("run_at") or None
        cron = body.get("cron") or None
        if not name or not query:
            raise HTTPException(400, "name and query are required")
        if not run_at and not cron:
            raise HTTPException(400, "run_at or cron is required")
        task_type = "recurring" if cron else "one_shot"
        try:
            task = tm.add_task(
                name=name, query=query, user_id=DASHBOARD_USER,
                task_type=task_type, run_at=run_at, cron_expr=cron)
            return task
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/tasks/{task_id}/cancel")
    async def api_task_cancel(task_id: int):
        """Cancel a task."""
        tm = getattr(agent, '_task_manager', None)
        if not tm:
            raise HTTPException(400, "Tasks not available")
        ok = tm.cancel_task(task_id)
        if not ok:
            raise HTTPException(404, "Task not found or already completed/cancelled")
        return {"status": "cancelled"}

    @app.post("/api/tasks/{task_id}/run")
    async def api_task_run_now(task_id: int):
        """Execute a task immediately."""
        import asyncio as _aio
        tm = getattr(agent, '_task_manager', None)
        if not tm:
            raise HTTPException(400, "Tasks not available")
        task = tm.get_task(task_id)
        if not task:
            raise HTTPException(404, "Task not found")

        async def _run():
            tm.mark_running(task_id)
            try:
                result = await agent.run(task["query"], task["user_id"])
                tm.mark_completed(task_id, result)
                agent._ws_broadcast("task_completed", {
                    "task_id": task_id, "name": task["name"],
                    "result": result[:500], "user_id": task["user_id"],
                })
            except Exception as e:
                tm.mark_failed(task_id, str(e))
                agent._ws_broadcast("task_failed", {
                    "task_id": task_id, "name": task["name"],
                    "error": str(e)[:200],
                })

        _aio.create_task(_run())
        return {"status": "triggered"}

    @app.delete("/api/tasks/{task_id}")
    async def api_task_delete(task_id: int):
        """Delete a task permanently."""
        tm = getattr(agent, '_task_manager', None)
        if not tm:
            raise HTTPException(400, "Tasks not available")
        ok = tm.delete_task(task_id)
        if not ok:
            raise HTTPException(404, "Task not found")
        return {"status": "deleted"}

    logger.info("Dashboard routes mounted")
