"""Dashboard API routes for LiteAgent web UI."""

import csv
import io
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DASHBOARD_USER = "dashboard-user"
CUSTOM_TOOLS_DIR = Path.home() / ".liteagent" / "custom_tools"


def mount_dashboard(app, agent):
    """Mount dashboard API routes onto FastAPI app."""
    try:
        from fastapi import HTTPException
        from fastapi.responses import HTMLResponse, FileResponse, Response
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

    @app.get("/api/usage")
    async def api_usage(days: int = 7):
        """Usage breakdown by model."""
        return agent.memory.get_usage_summary(days)

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
                fieldnames=["id", "user_id", "content", "type", "importance", "created_at"])
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

    # ── Feature monitoring ────────────────────

    @app.get("/api/features/status")
    async def api_features_status():
        """Status of all 8 metacognition/evolution/synthesis features."""
        features = agent.config.get("features", {})
        feature_names = [
            "dream_cycle", "self_evolving_prompt", "proactive_agent",
            "auto_tool_synthesis", "confidence_gate", "style_adaptation",
            "skill_crystallization", "counterfactual_replay",
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

        # Save API key if provided
        if api_key and provider_name != "ollama":
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
            "gemini": "google.generativeai",
            "ollama": "openai",
        }
        pkg = _SDK_PACKAGES.get(provider_name)
        if pkg:
            try:
                __import__(pkg)
            except ImportError:
                pip_extra = {"openai": "openai", "gemini": "gemini", "ollama": "ollama"}.get(provider_name, provider_name)
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
                pip_extra = {"openai": "openai", "gemini": "gemini", "ollama": "ollama"}.get(provider_name, provider_name)
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
                "gemini": "gemini-2.0-flash",
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

    # ── Voice Transcription Settings ─────────

    @app.get("/api/settings/voice")
    async def api_settings_voice():
        """Get voice transcription mode and available backends."""
        tg_cfg = agent.config.get("channels", {}).get("telegram", {})
        mode = tg_cfg.get("voice_transcription", "auto")

        # Check what's actually available
        has_builtin = "transcribe_voice" in agent.tools._tools
        mcp_tools = [n for n in agent.tools._tools
                     if "transcribe" in n and "__" in n]
        has_mcp = bool(mcp_tools)

        # Determine active backend
        if mode == "builtin" or (mode == "auto" and not has_mcp):
            active = "builtin"
        elif mode == "mcp" or (mode == "auto" and has_mcp):
            active = "mcp"
        else:
            active = "builtin"

        return {
            "mode": mode,
            "active": active,
            "has_builtin": has_builtin or mode == "builtin",
            "has_mcp": has_mcp or (agent._mcp_config and
                                   any("whisper" in k.lower() or "transcri" in k.lower()
                                       for k in agent._mcp_config)),
            "mcp_tools": mcp_tools,
        }

    @app.post("/api/settings/voice")
    async def api_settings_voice_save(body: dict):
        """Switch voice transcription mode: auto, builtin, mcp."""
        from ..config import save_config
        mode = body.get("mode", "auto")
        if mode not in ("auto", "builtin", "mcp"):
            raise HTTPException(400, "Mode must be: auto, builtin, or mcp")

        tg = agent.config.setdefault("channels", {}).setdefault("telegram", {})
        tg["voice_transcription"] = mode
        save_config(agent.config)

        # Apply immediately — re-register tools based on new mode
        if "transcribe_voice" not in agent.tools._tools:
            agent._wire_voice_tool()
        agent._apply_voice_transcription_mode()

        return {"ok": True, "mode": mode}

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

        # Reconnect storage
        try:
            agent._storage = create_storage(agent.config)
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

    # ── Qdrant Settings ────────────────────

    @app.get("/api/settings/qdrant")
    async def api_settings_qdrant():
        """Get Qdrant configuration status."""
        from ..config import get_api_key, key_preview
        rag_cfg = agent.config.get("rag", {})
        qdrant_cfg = rag_cfg.get("qdrant", {})
        api_key = qdrant_cfg.get("api_key") or get_api_key("qdrant") or ""
        rag = getattr(agent, '_rag', None)
        connected = rag.is_qdrant_connected() if rag else False
        stats = rag.get_stats() if rag else {}
        return {
            "enabled": rag_cfg.get("enabled", False),
            "connected": connected,
            "url": qdrant_cfg.get("url", ""),
            "collection": qdrant_cfg.get("collection", "liteagent_rag"),
            "api_key_preview": key_preview(api_key) if api_key else "",
            "stats": stats,
        }

    @app.post("/api/settings/qdrant")
    async def api_settings_qdrant_save(body: dict):
        """Save Qdrant configuration and reconnect RAG."""
        from ..config import save_provider_key
        from ..rag import RAGPipeline

        url = body.get("url", "").strip()
        api_key = body.get("api_key", "").strip()
        collection = body.get("collection", "liteagent_rag").strip()

        if not url:
            raise HTTPException(status_code=400, detail="Qdrant URL required")

        if api_key:
            save_provider_key("qdrant", api_key)

        # Update config
        agent.config.setdefault("rag", {}).update({"enabled": True})
        agent.config["rag"]["qdrant"] = {
            "url": url, "api_key": api_key, "collection": collection,
        }

        # Reconnect RAG
        try:
            agent._rag = RAGPipeline(
                agent.memory.db,
                embedder=agent.memory._embedder,
                config=agent.config["rag"])
            agent._wire_rag_tool()
            connected = agent._rag.is_qdrant_connected()
            return {"ok": True, "connected": connected}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.post("/api/settings/qdrant/test")
    async def api_settings_qdrant_test(body: dict):
        """Test Qdrant connection."""
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

    logger.info("Dashboard routes mounted")
