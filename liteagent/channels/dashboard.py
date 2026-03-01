"""Dashboard API routes for LiteAgent web UI."""

import csv
import io
import json
import logging

logger = logging.getLogger(__name__)

DASHBOARD_USER = "dashboard-user"


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
    async def dashboard_page():
        """Serve the dashboard SPA."""
        html_path = os.path.join(STATIC_DIR, "dashboard.html")
        if not os.path.exists(html_path):
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return FileResponse(html_path, media_type="text/html")

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
        """List registered tools."""
        defs = agent.tools.get_definitions()
        return [{"name": d["name"], "description": d.get("description", ""),
                 "parameters": list(d.get("input_schema", {}).get("properties", {}).keys())}
                for d in defs]

    @app.get("/api/config")
    async def api_config():
        """Read-only agent config (strip sensitive keys)."""
        import copy
        cfg = copy.deepcopy(agent.config)
        # Strip secrets
        for key_path in [
            ("providers", "anthropic", "api_key"),
            ("channels", "telegram", "token"),
            ("tools", "brave_api_key"),
        ]:
            obj = cfg
            for k in key_path[:-1]:
                obj = obj.get(k, {}) if isinstance(obj, dict) else {}
            if isinstance(obj, dict) and key_path[-1] in obj:
                obj[key_path[-1]] = "***"
        return cfg

    @app.get("/api/history")
    async def api_history():
        """Conversation history for dashboard user."""
        msgs = agent.memory.get_history(DASHBOARD_USER)
        return msgs[-50:]  # last 50 messages

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

    logger.info("Dashboard routes mounted")
