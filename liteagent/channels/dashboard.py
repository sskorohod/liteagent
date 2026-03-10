"""Dashboard API routes for LiteAgent web UI."""

import asyncio
import csv
import io
import json
import logging
import os
import re
import subprocess
import sys
import zipfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DASHBOARD_USER = "dashboard-user"
CUSTOM_TOOLS_DIR = Path.home() / ".liteagent" / "custom_tools"
_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _validate_tool_name(name: str) -> bool:
    """Allow only safe Python-style identifiers for custom tool names."""
    return bool(_TOOL_NAME_RE.fullmatch(name))


def _safe_custom_tool_path(name: str) -> Path:
    """Resolve custom tool file path and ensure it stays under CUSTOM_TOOLS_DIR."""
    base = CUSTOM_TOOLS_DIR.resolve()
    tool_path = (CUSTOM_TOOLS_DIR / f"{name}.py").resolve()
    try:
        tool_path.relative_to(base)
    except ValueError:
        raise ValueError("Unsafe tool path")
    return tool_path


def _coerce_local_path(raw_path: str) -> Path:
    """Normalize a local file reference from the dashboard."""
    raw = (raw_path or "").strip()
    if not raw:
        raise ValueError("Path is required")

    raw = raw.split("#", 1)[0].strip()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if not candidate.exists():
        trimmed = re.sub(r":\d+(?::\d+)?$", "", str(candidate))
        if trimmed != str(candidate):
            candidate = Path(trimmed).expanduser().resolve()

    if not candidate.exists():
        raise FileNotFoundError("Path not found")
    return candidate


def _reveal_in_file_manager(target: Path) -> None:
    """Reveal file or folder in the native file manager."""
    if sys.platform == "darwin":
        if target.is_dir():
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["open", "-R", str(target)])
        return
    if sys.platform.startswith("win"):
        if target.is_dir():
            subprocess.Popen(["explorer", str(target)])
        else:
            subprocess.Popen(["explorer", "/select,", str(target)])
        return

    folder = target if target.is_dir() else target.parent
    subprocess.Popen(["xdg-open", str(folder)])

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
        from fastapi import File, HTTPException, UploadFile
        from fastapi.responses import HTMLResponse, FileResponse, Response, JSONResponse
    except ImportError:
        raise ImportError("FastAPI is required: pip install liteagent[api]")

    import os
    from ..skills import _USER_SKILLS_DIR
    STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")

    def _dashboard_user_id() -> str:
        resolver = getattr(agent, "resolve_user_id", None)
        if callable(resolver):
            try:
                return resolver(DASHBOARD_USER)
            except Exception:
                return DASHBOARD_USER
        return DASHBOARD_USER

    def _feature_enabled(name: str) -> bool:
        runtime = getattr(agent, "_features", {}) or {}
        if name in runtime:
            cfg = runtime.get(name)
            if isinstance(cfg, dict):
                return bool(cfg.get("enabled", False))
            return bool(cfg)
        cfg = agent.config.get("features", {}).get(name, {})
        if isinstance(cfg, dict):
            return bool(cfg.get("enabled", False))
        return bool(cfg)

    def _memory_settings_payload() -> dict:
        mem_cfg = agent.config.get("memory", {})
        agent_provider = str(agent.config.get("agent", {}).get("provider", "")).strip().lower()
        requested_provider = str(mem_cfg.get("extraction_provider", "")).strip().lower()
        dedicated = bool(requested_provider and requested_provider != agent_provider)
        dedicated_ready = (not dedicated) or bool(getattr(agent.memory, "_extraction_provider", None))
        provider_mode = "dedicated" if dedicated else "shared"

        def _safe_int(value, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _safe_float(value, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        return {
            "extraction_provider": requested_provider,
            "extraction_provider_mode": provider_mode,
            "extraction_provider_ready": dedicated_ready,
            "extraction_model": str(mem_cfg.get("extraction_model", "")).strip(),
            "extraction_max_concurrency": max(
                1, min(_safe_int(mem_cfg.get("extraction_max_concurrency", 1) or 1, 1), 8)
            ),
            "memory_exchange_enabled": bool(mem_cfg.get("memory_exchange_enabled", True)),
            "memory_exchange_top_k": max(
                3, min(_safe_int(mem_cfg.get("memory_exchange_top_k", 8) or 8, 8), 20)
            ),
            "memory_exchange_pack_budget_tokens": max(
                100,
                min(_safe_int(mem_cfg.get("memory_exchange_pack_budget_tokens", 450) or 450, 450), 3000),
            ),
            "memory_exchange_max_packs": max(
                1, min(_safe_int(mem_cfg.get("memory_exchange_max_packs", 2) or 2, 2), 5)
            ),
            "memory_exchange_context_budget_tokens": max(
                120,
                min(_safe_int(mem_cfg.get("memory_exchange_context_budget_tokens", 700) or 700, 700), 4000),
            ),
            "memory_local_worker_enabled": bool(mem_cfg.get("memory_local_worker_enabled", True)),
            "memory_local_worker_interval_sec": max(
                2.0,
                min(_safe_float(mem_cfg.get("memory_local_worker_interval_sec", 12.0) or 12.0, 12.0), 300.0),
            ),
            "memory_local_worker_batch_size": max(
                4, min(_safe_int(mem_cfg.get("memory_local_worker_batch_size", 24) or 24, 24), 500)
            ),
            "shadow_twin_enabled": bool(mem_cfg.get("shadow_twin_enabled", True)),
            "shadow_twin_predictions": max(
                1, min(_safe_int(mem_cfg.get("shadow_twin_predictions", 3) or 3, 3), 8)
            ),
            "shadow_twin_use_llm": bool(mem_cfg.get("shadow_twin_use_llm", False)),
        }

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_page():
        """Serve the dashboard SPA."""
        html_path = os.path.join(STATIC_DIR, "dashboard.html")
        if not os.path.exists(html_path):
            raise HTTPException(status_code=404, detail="Dashboard not found")
        return FileResponse(
            html_path,
            media_type="text/html",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

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

    @app.get("/api/memory/metrics")
    async def api_memory_metrics():
        """Memory health metrics and stats."""
        return agent.memory.get_memory_metrics()

    @app.get("/api/memory/health")
    async def api_memory_health():
        """Memory health check — status + issues."""
        return agent.memory.memory_health_check()

    @app.get("/api/memory/explain")
    async def api_memory_explain(user_id: str = "", limit: int = 5):
        """Explainability: which memories were used and with what score."""
        uid = user_id.strip() or _dashboard_user_id()
        lim = max(1, min(int(limit), 20))
        return {
            "user_id": uid,
            "identity": agent.memory.get_identity_snapshot(uid),
            "traces": agent.memory.get_last_recall_trace(uid, limit=lim),
        }

    @app.get("/api/memory/thinking")
    async def api_memory_thinking(user_id: str = "", limit: int = 8):
        """Structured view of the user's strategic ideas, themes, and open questions."""
        uid = user_id.strip() or _dashboard_user_id()
        lim = max(4, min(int(limit), 20))
        return {
            "user_id": uid,
            "identity": agent.memory.get_identity_snapshot(uid),
            "cloud": agent.memory.get_thinking_cloud_summary(uid, limit=lim),
        }

    @app.get("/api/memory/human_support")
    async def api_memory_human_support(user_id: str = "", current_input: str = ""):
        """Structured view of focus/energy/admin-support opportunities for the user."""
        from ..evolution import get_human_support_snapshot

        uid = user_id.strip() or _dashboard_user_id()
        raw_cfg = (getattr(agent, "_features", {}) or {}).get(
            "human_support_agent",
            agent.config.get("features", {}).get("human_support_agent", {}),
        )
        cfg = dict(raw_cfg) if isinstance(raw_cfg, dict) else {"enabled": bool(raw_cfg)}
        return {
            "user_id": uid,
            "identity": agent.memory.get_identity_snapshot(uid),
            "support": get_human_support_snapshot(
                agent.memory.db,
                uid,
                current_input=current_input,
                config=cfg,
            ),
        }

    @app.get("/api/memory/identity")
    async def api_memory_identity(user_id: str = ""):
        """Get canonical identity snapshot for user/channel id."""
        uid = user_id.strip() or _dashboard_user_id()
        return agent.memory.get_identity_snapshot(uid)

    @app.post("/api/memory/identity")
    async def api_memory_identity_map(body: dict):
        """Create or update channel alias -> canonical person mapping."""
        alias_user_id = str(body.get("alias_user_id", "")).strip()
        person_id = str(body.get("person_id", "")).strip()
        source = str(body.get("source", "dashboard")).strip() or "dashboard"
        if not alias_user_id or not person_id:
            raise HTTPException(status_code=400, detail="alias_user_id and person_id are required")
        mapped = agent.memory.set_user_alias(
            alias_user_id,
            person_id,
            source=source,
            confidence=0.95,
        )
        return {"ok": True, "alias_user_id": alias_user_id, "person_id": mapped}

    @app.get("/api/memory/exchange")
    async def api_memory_exchange(user_id: str = ""):
        """Detailed memory exchange and shadow twin telemetry."""
        db = agent.memory.db
        uid = user_id.strip()
        if uid:
            uid = agent.memory.get_canonical_person_id(uid)
        where = "WHERE user_id = ?" if uid else ""
        params = [uid] if uid else []

        def _and(extra: str) -> str:
            return f"{where} AND {extra}" if where else f"WHERE {extra}"

        def _scalar(query: str, query_params: list):
            row = db.execute(query, query_params).fetchone()
            return row[0] if row and row[0] is not None else 0

        recent_cond = (
            "COALESCE(datetime(created_at), datetime(replace(created_at, 'T', ' '))) "
            ">= datetime('now', '-1 day')"
        )
        ready_cond = "status = 'ready'"
        intent_pending_cond = "status IN ('queued', 'running')"
        nonempty_anchor_cond = "anchor_query != ''"

        intents_total = int(_scalar(
            f"SELECT COUNT(*) FROM memory_exchange_intents {where}",
            params))
        intents_pending = int(_scalar(
            f"SELECT COUNT(*) FROM memory_exchange_intents {_and(intent_pending_cond)}",
            params))
        intents_24h = int(_scalar(
            f"SELECT COUNT(*) FROM memory_exchange_intents {_and(recent_cond)}",
            params))

        packs_total = int(_scalar(
            f"SELECT COUNT(*) FROM memory_context_packs {where}",
            params))
        packs_24h = int(_scalar(
            f"SELECT COUNT(*) FROM memory_context_packs {_and(recent_cond)}",
            params))
        packs_used = int(_scalar(
            f"SELECT COUNT(*) FROM memory_context_packs {_and('hit_count > 0')}",
            params))

        preds_total = int(_scalar(
            f"SELECT COUNT(*) FROM memory_shadow_predictions {where}",
            params))
        preds_ready = int(_scalar(
            f"SELECT COUNT(*) FROM memory_shadow_predictions {_and(ready_cond)}",
            params))
        preds_24h = int(_scalar(
            f"SELECT COUNT(*) FROM memory_shadow_predictions {_and(recent_cond)}",
            params))
        preds_used = int(_scalar(
            f"""SELECT COUNT(*) FROM memory_shadow_predictions
                {_and('(hit_count > 0 OR used_at IS NOT NULL)')}""",
            params))
        preds_unused = int(_scalar(
            f"""SELECT COUNT(*) FROM memory_shadow_predictions
                {_and('(hit_count = 0 AND used_at IS NULL)')}""",
            params))

        quality_row = db.execute(
            f"""SELECT AVG(score), AVG(token_estimate), AVG(hit_count)
                FROM memory_context_packs {where}""",
            params,
        ).fetchone() or (0.0, 0.0, 0.0)
        avg_pack_score = float(quality_row[0] or 0.0)
        avg_pack_tokens = float(quality_row[1] or 0.0)
        avg_pack_hits = float(quality_row[2] or 0.0)
        token_row = db.execute(
            f"""SELECT
                    COALESCE(SUM(token_estimate), 0),
                    COALESCE(SUM(token_estimate * hit_count), 0),
                    COALESCE(SUM(
                        CASE
                            WHEN hit_count > 1 THEN token_estimate * (hit_count - 1)
                            ELSE 0
                        END
                    ), 0)
                FROM memory_context_packs {where}""",
            params,
        ).fetchone() or (0, 0, 0)
        pack_tokens_cached = int(token_row[0] or 0)
        pack_tokens_served = int(token_row[1] or 0)
        pack_tokens_saved_est = int(token_row[2] or 0)
        queue_pending_total = int(intents_pending + preds_unused)
        token_savings_ratio = round(
            pack_tokens_saved_est / max(pack_tokens_served, 1), 3
        )

        top_pack_rows = db.execute(
            f"""SELECT id, title, query_hint, score, token_estimate, hit_count, updated_at
                FROM memory_context_packs {where}
                ORDER BY hit_count DESC, score DESC, updated_at DESC
                LIMIT 5""",
            params,
        ).fetchall()
        top_packs = [{
            "id": r[0],
            "title": r[1] or "",
            "query_hint": r[2] or "",
            "score": round(float(r[3] or 0.0), 3),
            "token_estimate": int(r[4] or 0),
            "hit_count": int(r[5] or 0),
            "updated_at": r[6],
        } for r in top_pack_rows]

        top_intent_rows = db.execute(
            f"""SELECT anchor_query, COUNT(*) AS cnt, MAX(created_at) AS last_seen
                FROM memory_exchange_intents
                {_and(nonempty_anchor_cond)}
                GROUP BY anchor_query
                ORDER BY cnt DESC, last_seen DESC
                LIMIT 5""",
            params,
        ).fetchall()
        top_intents = [{
            "anchor_query": r[0] or "",
            "count": int(r[1] or 0),
            "last_seen": r[2],
        } for r in top_intent_rows]

        pred_rows = db.execute(
            f"""SELECT predicted_query, anchor_query, confidence, hit_count,
                       status, created_at, used_at
                FROM memory_shadow_predictions {where}
                ORDER BY created_at DESC
                LIMIT 6""",
            params,
        ).fetchall()
        recent_predictions = [{
            "predicted_query": r[0] or "",
            "anchor_query": r[1] or "",
            "confidence": round(float(r[2] or 0.0), 3),
            "hit_count": int(r[3] or 0),
            "status": r[4] or "ready",
            "created_at": r[5],
            "used_at": r[6],
        } for r in pred_rows]

        settings = _memory_settings_payload()
        daemon = agent.memory.memory_exchange_daemon_state()
        shadow_cleanup_stats = daemon.get("shadow_cleanup_last_stats", {}) if isinstance(daemon, dict) else {}
        quality_metrics = agent.memory.get_memory_quality_metrics(user_id=uid or None, days=30, k=5)
        explainability = agent.memory.get_last_recall_trace(
            user_id=(uid or _dashboard_user_id()),
            limit=5,
        )
        identity = agent.memory.get_identity_snapshot(uid or _dashboard_user_id())
        return {
            "scope_user": uid or "all",
            "settings": settings,
            "daemon": daemon,
            "identity": identity,
            "quality_metrics": quality_metrics,
            "explainability": explainability,
            "counts": {
                "intents_total": intents_total,
                "intents_pending": intents_pending,
                "intents_24h": intents_24h,
                "packs_total": packs_total,
                "packs_24h": packs_24h,
                "predictions_total": preds_total,
                "predictions_ready": preds_ready,
                "predictions_24h": preds_24h,
            },
            "quality": {
                "prediction_hit_rate": round(preds_used / max(preds_total, 1), 3),
                "pack_usage_rate": round(packs_used / max(packs_total, 1), 3),
                "avg_pack_score": round(avg_pack_score, 3),
                "avg_pack_tokens": round(avg_pack_tokens, 1),
                "avg_pack_hits": round(avg_pack_hits, 2),
            },
            "queue": {
                "pending_total": queue_pending_total,
                "intents_pending": intents_pending,
                "predictions_ready": preds_ready,
                "predictions_unused": preds_unused,
                "shadow_cleanup_removed": int(shadow_cleanup_stats.get("removed_total", 0) or 0),
                "shadow_cleanup_at": shadow_cleanup_stats.get("updated_at"),
            },
            "tokens": {
                "pack_tokens_cached": pack_tokens_cached,
                "pack_tokens_served": pack_tokens_served,
                "pack_tokens_saved_est": pack_tokens_saved_est,
                "pack_savings_ratio": token_savings_ratio,
            },
            "recent": {
                "top_intents": top_intents,
                "top_packs": top_packs,
                "predictions": recent_predictions,
            },
        }

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
        if not _validate_tool_name(name):
            raise HTTPException(
                status_code=400,
                detail="Invalid tool name. Use 1-64 chars: letters, numbers, underscore; must start with letter/underscore.",
            )
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
                                 "breakpoint", "exit", "quit", "vars"}
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
            try:
                tool_path = _safe_custom_tool_path(name)
            except ValueError:
                raise HTTPException(status_code=400, detail="Unsafe tool path")
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
        if not _validate_tool_name(name):
            raise HTTPException(status_code=400, detail="Invalid tool name")
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
        try:
            tool_path = _safe_custom_tool_path(name)
        except ValueError:
            raise HTTPException(status_code=400, detail="Unsafe tool path")
        if tool_path.exists():
            tool_path.unlink()
            logger.info("Custom tool '%s' removed from disk", name)

        return {"ok": True, "name": name}

    @app.post("/api/ratings")
    async def api_rate_response(body: dict):
        """Rate the last agent response. body: {user_id, rating} where rating is 1-5 (1=👎, 5=👍)."""
        user_id = body.get("user_id", "dashboard-user")
        rating = body.get("rating")
        if rating is None or not isinstance(rating, int) or not (1 <= rating <= 5):
            raise HTTPException(status_code=400, detail="rating must be integer 1-5")
        if not hasattr(agent.memory, "rate_last_response"):
            raise HTTPException(status_code=501, detail="Ratings not supported")
        updated = agent.memory.rate_last_response(user_id, rating)
        return {"ok": updated, "rating": rating}

    @app.get("/api/ratings/stats")
    async def api_rating_stats(days: int = 30):
        """Return rating statistics for the last N days."""
        if not hasattr(agent.memory, "get_rating_stats"):
            return {}
        return agent.memory.get_rating_stats(days=days)

    @app.get("/api/analytics/tools")
    async def api_tool_analytics(days: int = 30):
        """Return aggregated tool usage stats: calls, success rate, avg/max duration."""
        if not hasattr(agent.memory, "get_tool_analytics"):
            return []
        return agent.memory.get_tool_analytics(days=days)

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
        return agent.memory.get_chat_history_for_display(_dashboard_user_id(), limit=100)

    @app.delete("/api/history")
    async def api_clear_history():
        """Clear chat history for dashboard user."""
        uid = _dashboard_user_id()
        agent.memory.clear_chat_history(uid)
        agent.memory.clear_conversation(uid)
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
        msgs = agent.memory.get_history(_dashboard_user_id())
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

    @app.get("/api/export/thinking")
    async def export_thinking_cloud(format: str = "json", user_id: str = "", limit: int = 200):
        """Export Thinking Cloud as JSON, Markdown, or Obsidian-compatible vault archive."""
        uid = user_id.strip() or _dashboard_user_id()
        lim = max(20, min(int(limit or 200), 500))
        bundle = agent.memory.export_thinking_cloud_obsidian(uid, limit=lim)

        if format == "md":
            content = bundle["files"].get("Thinking Cloud.md", "# Thinking Cloud\n")
            return Response(
                content=content,
                media_type="text/markdown",
                headers={"Content-Disposition": "attachment; filename=thinking-cloud.md"},
            )

        if format == "obsidian":
            safe_uid = re.sub(r"[^A-Za-z0-9._-]+", "-", str(bundle.get("user_id") or uid)).strip("-") or "user"
            archive_name = f"thinking-cloud-{safe_uid}.zip"
            vault_root = str(bundle.get("vault_name") or f"LiteAgent Thinking Cloud - {safe_uid}").strip() or "Thinking Cloud"
            payload = io.BytesIO()
            with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for rel_path, content in bundle.get("files", {}).items():
                    archive.writestr(f"{vault_root}/{rel_path}", content)
            return Response(
                content=payload.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{archive_name}"'},
            )

        return {
            "user_id": uid,
            "identity": agent.memory.get_identity_snapshot(uid),
            "cloud": agent.memory.get_thinking_cloud_summary(uid, limit=min(lim, 12)),
        }

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
        """Currently executing requests, queues, scheduler and background daemon."""
        from ..agent import LiteAgent
        active = LiteAgent.get_active_requests()

        now = datetime.now(timezone.utc)

        def _clip(text: str, limit: int = 180) -> str:
            raw = str(text or "").strip()
            if len(raw) <= limit:
                return raw
            return raw[:limit - 1].rstrip() + "…"

        def _parse_dt(value: str | None):
            if not value:
                return None
            try:
                dt = datetime.fromisoformat(str(value))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                return None

        def _elapsed_sec(value: str | None) -> float | None:
            dt = _parse_dt(value)
            if not dt:
                return None
            return max(0.0, (now - dt).total_seconds())

        def _fmt_progress(value: float | None) -> float | None:
            if value is None:
                return None
            return round(max(0.0, min(float(value), 100.0)), 1)

        def _normalize_parallel_children(children) -> list[dict]:
            out = []
            for child in list(children or [])[:8]:
                out.append({
                    "tool_use_id": str(child.get("tool_use_id") or ""),
                    "name": str(child.get("name") or ""),
                    "status": str(child.get("status") or "pending"),
                    "duration_ms": int(child.get("duration_ms") or 0),
                    "error": bool(child.get("error")),
                    "result_preview": _clip(child.get("result_preview") or "", 160),
                })
            return out

        def _request_progress(req: dict) -> float | None:
            max_iterations = max(1, int(req.get("max_iterations") or 1))
            iteration = max(0, int(req.get("iteration") or 0))
            phase = str(req.get("phase") or "reasoning")
            parallel_total = max(0, int(req.get("parallel_total") or 0))
            parallel_completed = max(0, int(req.get("parallel_completed") or 0))
            if phase == "finalizing":
                return 96.0
            if phase == "reflection":
                return _fmt_progress((max(iteration, 1) / max_iterations) * 86.0)
            if phase == "parallel_tools" and parallel_total > 0:
                base_units = max(iteration - 1, 0) + (parallel_completed / max(parallel_total, 1))
                return _fmt_progress((base_units / max_iterations) * 88.0)
            if iteration > 0:
                return _fmt_progress((iteration / max_iterations) * 72.0)
            return 8.0

        def _lane_item(*, item_id: str, kind: str, title: str, status: str,
                       phase: str, description: str = "", progress: float | None = None,
                       progress_label: str = "", elapsed_sec: float | None = None,
                       meta: list[str] | None = None, parallel_children: list[dict] | None = None,
                       parallel_total: int = 0, parallel_completed: int = 0) -> dict:
            return {
                "id": item_id,
                "kind": kind,
                "title": _clip(title, 120) or item_id,
                "status": str(status or "running"),
                "phase": _clip(phase, 120),
                "description": _clip(description, 220),
                "progress": _fmt_progress(progress),
                "progress_label": _clip(progress_label, 120),
                "elapsed_sec": round(float(elapsed_sec), 1) if elapsed_sec is not None else None,
                "meta": [m for m in (meta or []) if str(m or "").strip()][:4],
                "parallel_children": parallel_children or [],
                "parallel_total": max(0, int(parallel_total or 0)),
                "parallel_completed": max(0, int(parallel_completed or 0)),
            }

        scheduler_running = []
        sched = getattr(agent, '_scheduler', None)
        if sched:
            for job in sched._jobs:
                if job.get("_running"):
                    scheduler_running.append({
                        "name": job["name"],
                        "started_at": job.get("_run_started"),
                        "max_runtime_sec": int(job.get("max_runtime_sec") or 0),
                        "status": str(job.get("status") or "running"),
                        "retry_on_fail": bool(job.get("retry_on_fail")),
                    })

        queued = LiteAgent.get_queued_requests()
        bg_daemon = getattr(agent, "_background_task_daemon", None)
        bg_running = bg_daemon.get_active_tasks() if bg_daemon else []
        bg_state = bg_daemon.state() if bg_daemon else {"enabled": False, "running": False}
        goal_daemon = getattr(agent, "_goal_coordinator", None)
        goals_running = goal_daemon.get_active_goals() if goal_daemon else []
        goal_state = goal_daemon.state() if goal_daemon else {"enabled": False, "running": False}

        bg_pending = 0
        tm = getattr(agent, "_task_manager", None)
        if tm:
            try:
                bg_pending = int(tm.count_background_pending())
            except Exception:
                bg_pending = 0
        goals_pending = 0
        gm = getattr(agent, "_goal_manager", None)
        if gm:
            try:
                goals_pending = int(gm.count_pending_goals())
            except Exception:
                goals_pending = 0

        request_items = []
        for req in active:
            request_items.append(_lane_item(
                item_id=f"request-{req.get('id')}",
                kind="request",
                title=req.get("input_preview") or "Request",
                status=req.get("status") or "running",
                phase=req.get("phase_label") or req.get("phase") or "Running",
                description="",
                progress=_request_progress(req),
                progress_label=req.get("progress_label") or "",
                elapsed_sec=_elapsed_sec(req.get("started_at")),
                meta=[
                    str(req.get("user_id") or ""),
                    str(req.get("model") or ""),
                    f"tier {req.get('cascade_tier')}" if req.get("cascade_tier") else "",
                    f"score {req.get('complexity_score')}" if req.get("complexity_score") not in (None, "", -1) else "",
                ],
                parallel_children=_normalize_parallel_children(req.get("parallel_children")),
                parallel_total=int(req.get("parallel_total") or 0),
                parallel_completed=int(req.get("parallel_completed") or 0),
            ))

        background_items = []
        for task in bg_running:
            max_attempts = int(task.get("max_attempts") or 0)
            attempt = max(1, int(task.get("attempt") or 1))
            progress = (attempt / max_attempts * 100.0) if max_attempts > 0 else None
            progress_label = (
                f"attempt {attempt}/{max_attempts}" if max_attempts > 0
                else f"attempt {attempt} · retry until solved"
            )
            background_items.append(_lane_item(
                item_id=f"background-{task.get('task_id')}",
                kind="background_task",
                title=task.get("name") or f"task-{task.get('task_id')}",
                status=task.get("status") or "running",
                phase=task.get("phase_label") or "Background daemon execution",
                description=task.get("query_preview") or "",
                progress=progress,
                progress_label=progress_label,
                elapsed_sec=_elapsed_sec(task.get("started_at")),
                meta=[
                    str(task.get("user_id") or ""),
                    f"p{int(task.get('priority') or 5)}",
                    str(task.get("source") or ""),
                    f"retry {int(task.get('retry_delay_sec') or 0)}s",
                ],
            ))

        goal_items = []
        for goal in goals_running:
            goal_items.append(_lane_item(
                item_id=f"goal-{goal.get('goal_id')}",
                kind="goal",
                title=goal.get("title") or f"goal-{goal.get('goal_id')}",
                status="running",
                phase=goal.get("step_title") or goal.get("current_phase") or "Goal execution",
                description=goal.get("last_result") or goal.get("strategy") or "",
                progress=float(goal.get("progress") or 0.0) * 100.0,
                progress_label=(
                    f"phase {goal.get('current_phase') or 'planned'}"
                    + (f" · stalled {int(goal.get('stalled_cycles') or 0)}" if int(goal.get("stalled_cycles") or 0) > 0 else "")
                ),
                elapsed_sec=_elapsed_sec(goal.get("started_at")),
                meta=[
                    str(goal.get("user_id") or ""),
                    f"p{int(goal.get('priority') or 5)}",
                    f"plan v{int(goal.get('plan_version') or 0)}" if int(goal.get("plan_version") or 0) > 0 else "",
                ],
            ))

        scheduler_items = []
        for job in scheduler_running:
            elapsed = _elapsed_sec(job.get("started_at"))
            max_runtime = int(job.get("max_runtime_sec") or 0)
            scheduler_items.append(_lane_item(
                item_id=f"scheduler-{job.get('name')}",
                kind="scheduler_job",
                title=job.get("name") or "scheduler job",
                status=job.get("status") or "running",
                phase="Scheduled job runtime",
                description="",
                progress=((elapsed or 0.0) / max_runtime * 100.0) if elapsed is not None and max_runtime > 0 else None,
                progress_label=(
                    f"{round(elapsed or 0.0, 1)}s / {max_runtime}s cap" if max_runtime > 0
                    else f"{round(elapsed or 0.0, 1)}s runtime"
                ),
                elapsed_sec=elapsed,
                meta=["scheduler", "retry enabled" if job.get("retry_on_fail") else ""],
            ))

        queued_items = []
        for q in queued:
            queued_items.append(_lane_item(
                item_id=f"queue-{q.get('id')}",
                kind="queued_request",
                title=f"Queued request #{q.get('id')}",
                status="queued",
                phase="Waiting for user lock",
                description="",
                progress=None,
                progress_label="Queued",
                elapsed_sec=_elapsed_sec(q.get("queued_at")),
                meta=[str(q.get("user_id") or "")],
            ))

        if bg_pending > 0:
            daemon_label = (
                "daemon active" if bg_state.get("running")
                else ("daemon paused" if bg_state.get("enabled") else "daemon disabled")
            )
            queued_items.append(_lane_item(
                item_id="background-pending",
                kind="background_queue",
                title="Background queue",
                status="queued",
                phase=daemon_label,
                description=f"{bg_pending} autonomous task(s) waiting in queue",
                progress=None,
                progress_label=f"{bg_pending} pending",
                meta=[
                    f"processed {int(bg_state.get('processed_total') or 0)}",
                    f"failed {int(bg_state.get('failed_total') or 0)}",
                ],
            ))

        if goals_pending > 0:
            daemon_label = (
                "coordinator active" if goal_state.get("running")
                else ("coordinator paused" if goal_state.get("enabled") else "coordinator disabled")
            )
            queued_items.append(_lane_item(
                item_id="goals-pending",
                kind="goal_queue",
                title="Goal queue",
                status="queued",
                phase=daemon_label,
                description=f"{goals_pending} goal cycle(s) waiting for execution",
                progress=None,
                progress_label=f"{goals_pending} pending",
                meta=[
                    f"planned {int(goal_state.get('planned_total') or 0)}",
                    f"replanned {int(goal_state.get('replanned_total') or 0)}",
                ],
            ))

        lanes = [
            {
                "id": "foreground",
                "label": "Live Requests",
                "tone": "amber",
                "items": request_items,
                "count": len(request_items),
                "empty": "No live requests",
            },
            {
                "id": "autonomous",
                "label": "Autonomous Work",
                "tone": "cyan",
                "items": background_items + goal_items + scheduler_items,
                "count": len(background_items) + len(goal_items) + len(scheduler_items),
                "empty": "No background or scheduled execution",
            },
            {
                "id": "queued",
                "label": "Queue",
                "tone": "indigo",
                "items": queued_items,
                "count": len(queued_items),
                "empty": "Queues are clear",
            },
        ]

        active_total = len(request_items) + len(background_items) + len(goal_items) + len(scheduler_items)
        queued_total = len(queued_items)
        parallel_units = (
            sum(max(0, int(item.get("parallel_total") or 0)) for item in request_items)
            + len(background_items) + len(goal_items) + len(scheduler_items)
        )
        summary = {
            "active_total": active_total,
            "live_total": len(request_items),
            "autonomous_total": len(background_items) + len(goal_items) + len(scheduler_items),
            "queued_total": queued_total,
            "parallel_units": parallel_units,
            "background_pending": bg_pending,
            "goals_pending": goals_pending,
        }
        daemons = [
            {
                "id": "background",
                "label": "Background daemon",
                "running": bool(bg_state.get("running")),
                "enabled": bool(bg_state.get("enabled")),
                "pending": bg_pending,
                "active_count": int(bg_state.get("active_count") or 0),
                "last_pause_reason": str(bg_state.get("last_pause_reason") or ""),
            },
            {
                "id": "goals",
                "label": "Goal coordinator",
                "running": bool(goal_state.get("running")),
                "enabled": bool(goal_state.get("enabled")),
                "pending": goals_pending,
                "active_count": int(goal_state.get("active_count") or 0),
                "last_pause_reason": str(goal_state.get("last_pause_reason") or ""),
            },
        ]

        return {
            "requests": active,
            "queued": queued,
            "scheduler_jobs_running": scheduler_running,
            "background_tasks_running": bg_running,
            "background_pending": bg_pending,
            "background_daemon": bg_state,
            "goals_running": goals_running,
            "goals_pending": goals_pending,
            "goal_coordinator": goal_state,
            "summary": summary,
            "lanes": lanes,
            "daemons": daemons,
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
        features = {}
        for name in ["dream_cycle", "self_evolving_prompt", "proactive_agent",
                      "critical_response_review",
                      "human_support_agent",
                      "auto_tool_synthesis", "confidence_gate", "style_adaptation",
                      "skill_crystallization", "counterfactual_replay",
                      "internal_monologue"]:
            features[name] = _feature_enabled(name)

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
        routing_state = agent.get_cascade_dashboard_state() if hasattr(agent, "get_cascade_dashboard_state") else {}

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
            "advisor": routing_state.get("advisor", {}),
            "candidates": routing_state.get("candidates", []),
            "recommendations": routing_state.get("recommendations", []),
            "is_local_only_now": agent._is_local_only_hours(),
        }

    # ── Feature monitoring ────────────────────

    @app.get("/api/features/status")
    async def api_features_status():
        """Status of all 9 metacognition/evolution/synthesis features."""
        feature_names = [
            "dream_cycle", "self_evolving_prompt", "proactive_agent",
            "critical_response_review",
            "human_support_agent",
            "auto_tool_synthesis", "confidence_gate", "style_adaptation",
            "skill_crystallization", "counterfactual_replay",
            "internal_monologue",
        ]
        status = {}
        for name in feature_names:
            status[name] = {"enabled": _feature_enabled(name)}

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
            "critical_response_review",
            "human_support_agent",
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
            "basic": ["style_adaptation", "confidence_gate", "critical_response_review", "skill_crystallization", "human_support_agent"],
            "all": [
                "dream_cycle", "self_evolving_prompt", "proactive_agent",
                "critical_response_review",
                "human_support_agent",
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

    # ── Skills Management ─────────────────────

    @app.get("/api/settings/skills")
    async def api_settings_skills():
        """List all skills with status for dashboard."""
        skills = agent.skill_registry.list_skills()
        disabled = agent.config.get("skills", {}).get("disabled", [])
        return {
            "skills": skills,
            "disabled": disabled,
            "skills_enabled": agent.config.get("skills", {}).get("enabled", True),
            "user_skills_dir": str(_USER_SKILLS_DIR),
            "count": len(skills),
        }

    @app.get("/api/skills/{name}")
    async def api_skill_detail(name: str):
        """Get full skill details including body."""
        detail = agent.skill_registry.get_skill(name)
        if not detail:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
        return detail

    @app.post("/api/files/reveal")
    async def api_reveal_file(body: dict):
        """Reveal a local file or its parent folder in the system file manager."""
        raw_path = str(body.get("path", "")).strip()
        if not raw_path:
            raise HTTPException(status_code=400, detail="path is required")
        try:
            target = _coerce_local_path(raw_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid path: {e}") from e

        try:
            _reveal_in_file_manager(target)
        except Exception as e:
            logger.warning("Failed to reveal local path %s: %s", target, e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to open file manager: {e}",
            ) from e

        folder = target if target.is_dir() else target.parent
        return {"ok": True, "path": str(target), "folder": str(folder)}

    @app.put("/api/skills/{name}")
    async def api_skill_update(name: str, body: dict):
        """Update a skill's body or metadata. For bundled skills, creates user override."""
        skill = agent.skill_registry._skills.get(name)
        if not skill:
            raise HTTPException(404, f"Skill '{name}' not found")

        new_body = body.get("body", skill.body)
        new_meta = body.get("metadata", {})
        frontmatter = {
            "name": name,
            "description": new_meta.get("description", skill.metadata.description),
        }
        meta_section = {}
        if skill.metadata.emoji or new_meta.get("emoji"):
            meta_section["emoji"] = new_meta.get("emoji", skill.metadata.emoji)
        kw = new_meta.get("keywords", skill.metadata.keywords)
        if kw:
            meta_section["keywords"] = kw
        tools = new_meta.get("tools", skill.metadata.tools)
        if tools:
            meta_section["tools"] = tools
        if skill.metadata.always or new_meta.get("always"):
            meta_section["always"] = new_meta.get("always", skill.metadata.always)
        if meta_section:
            frontmatter["metadata"] = meta_section

        agent.skill_registry.write_skill(name, new_body, frontmatter)
        agent.skill_registry.load_all(agent.config)
        return {"ok": True, "name": name}

    @app.post("/api/skills")
    async def api_skill_create(body: dict):
        """Create a new user skill."""
        name = body.get("name", "").strip().lower().replace(" ", "-")
        if not name:
            raise HTTPException(400, "Skill name required")
        if name in agent.skill_registry._skills:
            raise HTTPException(400, f"Skill '{name}' already exists")

        description = body.get("description", "")
        skill_body = body.get("body", "")
        keywords = body.get("keywords", [])
        emoji = body.get("emoji", "")
        tools = body.get("tools", [])

        frontmatter: dict = {"name": name, "description": description}
        meta: dict = {}
        if emoji:
            meta["emoji"] = emoji
        if keywords:
            meta["keywords"] = keywords
        if tools:
            meta["tools"] = tools
        if meta:
            frontmatter["metadata"] = meta

        agent.skill_registry.write_skill(name, skill_body, frontmatter)
        agent.skill_registry.load_all(agent.config)
        return {"ok": True, "name": name}

    @app.delete("/api/skills/{name}")
    async def api_skill_delete(name: str):
        """Delete a user/project skill."""
        skill = agent.skill_registry._skills.get(name)
        if not skill:
            raise HTTPException(404, f"Skill '{name}' not found")
        if skill.source == "bundled":
            raise HTTPException(400, "Cannot delete bundled skills")
        ok = agent.skill_registry.delete_skill(name)
        if not ok:
            raise HTTPException(500, "Failed to delete skill")
        return {"ok": True, "name": name}

    @app.post("/api/skills/{name}/toggle")
    async def api_skill_toggle(name: str, body: dict):
        """Enable or disable a skill via config.skills.disabled list."""
        from ..config import save_config

        enabled = body.get("enabled", True)
        skills_cfg = agent.config.setdefault("skills", {})
        disabled = skills_cfg.setdefault("disabled", [])

        if enabled and name in disabled:
            disabled.remove(name)
        elif not enabled and name not in disabled:
            disabled.append(name)

        save_config(agent.config)
        agent.skill_registry.load_all(agent.config)
        return {"ok": True, "name": name, "enabled": enabled}

    @app.post("/api/skills/reload")
    async def api_skills_reload():
        """Reload all skills from disk."""
        agent.skill_registry.load_all(agent.config)
        return {
            "ok": True,
            "count": len(agent.skill_registry._skills),
            "skills": list(agent.skill_registry._skills.keys()),
        }

    # ── Memory Settings ───────────────────

    @app.get("/api/settings/memory")
    async def api_settings_memory():
        """Get memory exchange + shadow twin configuration."""
        return _memory_settings_payload()

    @app.post("/api/settings/memory")
    async def api_settings_memory_save(body: dict):
        """Save memory exchange + shadow twin configuration."""
        from ..config import save_config

        def _as_bool(value, default: bool) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                low = value.strip().lower()
                if low in {"1", "true", "yes", "on"}:
                    return True
                if low in {"0", "false", "no", "off"}:
                    return False
            return default

        mem_cfg = agent.config.setdefault("memory", {})
        if "extraction_provider" in body:
            mem_cfg["extraction_provider"] = str(
                body.get("extraction_provider", "")
            ).strip().lower()
        if "extraction_model" in body:
            mem_cfg["extraction_model"] = str(body.get("extraction_model", "")).strip()
        if "extraction_max_concurrency" in body:
            try:
                val = int(body.get("extraction_max_concurrency", 1))
            except (TypeError, ValueError):
                val = 1
            mem_cfg["extraction_max_concurrency"] = max(1, min(val, 8))

        if "memory_exchange_enabled" in body:
            mem_cfg["memory_exchange_enabled"] = _as_bool(
                body.get("memory_exchange_enabled"), True
            )
        if "memory_exchange_top_k" in body:
            try:
                val = int(body.get("memory_exchange_top_k", 8))
            except (TypeError, ValueError):
                val = 8
            mem_cfg["memory_exchange_top_k"] = max(3, min(val, 20))
        if "memory_exchange_pack_budget_tokens" in body:
            try:
                val = int(body.get("memory_exchange_pack_budget_tokens", 450))
            except (TypeError, ValueError):
                val = 450
            mem_cfg["memory_exchange_pack_budget_tokens"] = max(100, min(val, 3000))
        if "memory_exchange_max_packs" in body:
            try:
                val = int(body.get("memory_exchange_max_packs", 2))
            except (TypeError, ValueError):
                val = 2
            mem_cfg["memory_exchange_max_packs"] = max(1, min(val, 5))
        if "memory_exchange_context_budget_tokens" in body:
            try:
                val = int(body.get("memory_exchange_context_budget_tokens", 700))
            except (TypeError, ValueError):
                val = 700
            mem_cfg["memory_exchange_context_budget_tokens"] = max(120, min(val, 4000))
        if "memory_local_worker_enabled" in body:
            mem_cfg["memory_local_worker_enabled"] = _as_bool(
                body.get("memory_local_worker_enabled"), True
            )
        if "memory_local_worker_interval_sec" in body:
            try:
                val = float(body.get("memory_local_worker_interval_sec", 12.0))
            except (TypeError, ValueError):
                val = 12.0
            mem_cfg["memory_local_worker_interval_sec"] = max(2.0, min(val, 300.0))
        if "memory_local_worker_batch_size" in body:
            try:
                val = int(body.get("memory_local_worker_batch_size", 24))
            except (TypeError, ValueError):
                val = 24
            mem_cfg["memory_local_worker_batch_size"] = max(4, min(val, 500))

        if "shadow_twin_enabled" in body:
            mem_cfg["shadow_twin_enabled"] = _as_bool(
                body.get("shadow_twin_enabled"), True
            )
        if "shadow_twin_predictions" in body:
            try:
                val = int(body.get("shadow_twin_predictions", 3))
            except (TypeError, ValueError):
                val = 3
            mem_cfg["shadow_twin_predictions"] = max(1, min(val, 8))
        if "shadow_twin_use_llm" in body:
            mem_cfg["shadow_twin_use_llm"] = _as_bool(
                body.get("shadow_twin_use_llm"), False
            )

        save_config(agent.config)

        try:
            max_extract_concurrency = int(
                mem_cfg.get("extraction_max_concurrency", 1) or 1
            )
            agent.memory._extraction_semaphore = asyncio.Semaphore(
                max(1, max_extract_concurrency)
            )
            agent.memory._extraction_provider = agent.memory._init_extraction_provider()
            agent.memory._extraction_provider_name = str(
                mem_cfg.get("extraction_provider", "")
            ).strip().lower()
        except Exception as e:
            logger.warning("Failed to refresh memory extraction runtime config: %s", e)

        return {"ok": True, "settings": _memory_settings_payload()}

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

        # Backward-compatible flat list used by some settings widgets
        # (planning dropdown, older dashboard builds, etc.)
        available_models: list[str] = []
        for m in providers.get(active_provider, {}).get("models", []) or []:
            if m and m not in available_models:
                available_models.append(m)
        if active_model and active_model not in available_models:
            available_models.insert(0, active_model)

        return {
            "active_provider": active_provider,
            "active_model": active_model,
            "providers": providers,
            "available_models": available_models,
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

    # ── Auth Profiles ──────────────────────

    @app.get("/api/auth-profiles")
    async def api_auth_profiles():
        """Return auth profile summary for all configured providers."""
        try:
            from ..auth_profiles import get_manager
            mgr = get_manager()
            providers = ["anthropic", "openai", "gemini", "ollama"]
            result = {}
            for p in providers:
                summary = mgr.stats_summary(p)
                if summary["total"] > 0:
                    result[p] = summary
            return {"providers": result}
        except Exception as exc:
            return {"providers": {}, "error": str(exc)}

    @app.post("/api/auth-profiles/clear-cooldown")
    async def api_auth_profiles_clear_cooldown(body: dict):
        """Clear cooldown for a specific provider key."""
        import time
        provider = body.get("provider", "").strip()
        label = body.get("label", "").strip()
        if not provider:
            raise HTTPException(status_code=400, detail="provider required")
        try:
            from ..auth_profiles import get_manager
            mgr = get_manager()
            cleared = 0
            for profile in mgr.list_profiles(provider):
                if not label or profile.label == label:
                    profile.stats.cooldown_until = 0.0
                    profile.stats.consecutive_failures = 0
                    cleared += 1
            from ..auth_profiles import _save_store
            _save_store(mgr._store)
            return {"ok": True, "cleared": cleared}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Routing Settings ──────────────────────

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
            "intelligent_routing": dict(cost_cfg.get("intelligent_routing", {}) or {}),
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

        if "intelligent_routing" in body and isinstance(body["intelligent_routing"], dict):
            ir = body["intelligent_routing"]
            current = agent.config.setdefault("cost", {}).setdefault("intelligent_routing", {})
            if "enabled" in ir:
                current["enabled"] = bool(ir["enabled"])
            if "use_llm" in ir:
                current["use_llm"] = bool(ir["use_llm"])
            if "advisor_model" in ir:
                current["advisor_model"] = str(ir["advisor_model"] or "").strip()
            if "min_complexity" in ir:
                try:
                    current["min_complexity"] = max(0, min(int(ir["min_complexity"]), 10))
                except (TypeError, ValueError):
                    pass
            if "local_min_complexity" in ir:
                try:
                    current["local_min_complexity"] = max(0, min(int(ir["local_min_complexity"]), 10))
                except (TypeError, ValueError):
                    pass

        if hasattr(agent, "_normalize_runtime_model_config"):
            agent._normalize_runtime_model_config()
        if hasattr(agent, "_build_intelligent_routing_config"):
            agent._intelligent_routing_cfg = agent._build_intelligent_routing_config(
                agent.config.get("cost", {})
            )

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
        from ..voice import (
            OPENAI_TTS_VOICES, OPENAI_TTS_MODELS, STT_PROVIDERS, TTS_PROVIDERS,
            DEFAULT_EDGE_LANGUAGE, DEFAULT_GROQ_TTS_LANGUAGE, DEFAULT_TTS_LANGUAGE,
            EDGE_TTS_LANGUAGE_LABELS, EDGE_TTS_VOICES_BY_LANGUAGE,
            GROQ_TTS_MODELS, GROQ_TTS_MODEL_INFO, GROQ_TTS_LANGUAGE_LABELS,
            TTS_LANGUAGE_OPTIONS,
            TTS_COST_INFO,
        )

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

        # Build per-provider metadata for the UI
        providers_meta = {
            "openai": {
                "available": bool(_get_tts_api_key("openai", agent.config)),
                "voices": list(OPENAI_TTS_VOICES),
                "models": list(OPENAI_TTS_MODELS),
                "cost": TTS_COST_INFO.get("openai", ""),
            },
            "elevenlabs": {
                "available": bool(_get_tts_api_key("elevenlabs", agent.config)),
                "voices": [],
                "models": ["eleven_multilingual_v2", "eleven_turbo_v2_5", "eleven_monolingual_v1"],
                "cost": TTS_COST_INFO.get("elevenlabs", ""),
            },
            "groq": {
                "available": bool(_get_tts_api_key("groq", agent.config)),
                "voices": sorted({
                    voice
                    for meta in GROQ_TTS_MODEL_INFO.values()
                    for voice in meta.get("voices", [])
                }),
                "models": list(GROQ_TTS_MODELS),
                "languages": list(TTS_LANGUAGE_OPTIONS),
                "language_labels": dict(GROQ_TTS_LANGUAGE_LABELS),
                "model_info": dict(GROQ_TTS_MODEL_INFO),
                "default_language": DEFAULT_GROQ_TTS_LANGUAGE,
                "experimental_languages": ["ru"],
                "cost": TTS_COST_INFO.get("groq", ""),
            },
            "edge": {
                "available": True,
                "voices": sorted({
                    voice
                    for voices in EDGE_TTS_VOICES_BY_LANGUAGE.values()
                    for voice in voices
                }),
                "models": [],
                "languages": list(EDGE_TTS_VOICES_BY_LANGUAGE.keys()),
                "language_labels": dict(EDGE_TTS_LANGUAGE_LABELS),
                "voices_by_language": dict(EDGE_TTS_VOICES_BY_LANGUAGE),
                "default_language": DEFAULT_EDGE_LANGUAGE,
                "cost": TTS_COST_INFO.get("edge", "free"),
            },
        }

        return {
            # TTS settings
            "tts": {
                "auto": tts["auto"],
                "provider": tts["provider"],
                "max_length": tts["max_length"],
                "language": tts.get("language", DEFAULT_TTS_LANGUAGE),
                "has_openai": providers_meta["openai"]["available"],
                "has_elevenlabs": providers_meta["elevenlabs"]["available"],
                "has_groq": providers_meta["groq"]["available"],
                "has_edge": True,
                "openai": tts["openai"],
                "elevenlabs": tts["elevenlabs"],
                "groq": tts.get(
                    "groq",
                    {
                        "model": "playai-tts",
                        "voice": "Fritz-PlayAI",
                        "language": DEFAULT_GROQ_TTS_LANGUAGE,
                        "speed": 1.0,
                    },
                ),
                "edge": tts["edge"],
                "providers_meta": providers_meta,
                "last_attempt": get_last_tts_attempt(),
                # Legacy compat
                "voices": list(OPENAI_TTS_VOICES),
                "models": list(OPENAI_TTS_MODELS),
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
                if tts_data["provider"] not in ("openai", "elevenlabs", "groq", "edge"):
                    raise HTTPException(400, "provider must be: openai, elevenlabs, groq, or edge")
                tts["provider"] = tts_data["provider"]
            if "language" in tts_data:
                if tts_data["language"] not in ("ru", "en", "ar"):
                    raise HTTPException(400, "language must be: ru, en, or ar")
                tts["language"] = tts_data["language"]
            if "max_length" in tts_data:
                tts["max_length"] = int(tts_data["max_length"])
            if "openai" in tts_data:
                tts["openai"] = {**tts.get("openai", {}), **tts_data["openai"]}
            if "elevenlabs" in tts_data:
                tts["elevenlabs"] = {**tts.get("elevenlabs", {}), **tts_data["elevenlabs"]}
            if "groq" in tts_data:
                groq_data = dict(tts_data["groq"])
                if "language" in groq_data and groq_data["language"] not in ("ru", "en", "ar"):
                    raise HTTPException(400, "groq language must be: ru, en, or ar")
                tts["groq"] = {**tts.get("groq", {}), **groq_data}
            if "edge" in tts_data:
                edge_data = dict(tts_data["edge"])
                if "language" in edge_data and edge_data["language"] not in EDGE_TTS_VOICES_BY_LANGUAGE:
                    raise HTTPException(400, "edge language is not supported")
                tts["edge"] = {**tts.get("edge", {}), **edge_data}

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

    @app.get("/api/files/stats")
    async def api_files_stats():
        """File statistics: total count, total size, per-source breakdown."""
        fm = getattr(agent, '_file_manager', None)
        if not fm:
            return {"total_files": 0, "total_size_bytes": 0, "sources": {}}
        try:
            row = fm._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM file_index"
            ).fetchone()
            source_rows = fm._db.execute(
                "SELECT source, COUNT(*), COALESCE(SUM(size_bytes), 0) "
                "FROM file_index GROUP BY source ORDER BY COUNT(*) DESC"
            ).fetchall()
            return {
                "total_files": row[0],
                "total_size_bytes": row[1],
                "sources": {
                    r[0]: {"count": r[1], "size_bytes": r[2]}
                    for r in source_rows
                },
            }
        except Exception:
            return {"total_files": 0, "total_size_bytes": 0, "sources": {}}

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

    @app.delete("/api/files/{key:path}")
    async def api_files_delete(key: str):
        """Delete a file from S3, file_index, and RAG chunks."""
        fm = getattr(agent, '_file_manager', None)
        storage = getattr(agent, '_storage', None)
        if not fm or not storage:
            raise HTTPException(400, "File manager not enabled")
        # 1. Delete from S3
        try:
            await storage.async_delete(key)
        except Exception:
            pass
        # 2. Delete RAG chunks if any
        rag = getattr(agent, '_rag', None)
        if rag:
            try:
                row = fm._db.execute(
                    "SELECT id FROM rag_documents WHERE path = ?", (key,)
                ).fetchone()
                if row:
                    rag.delete_document(row[0])
            except Exception:
                pass
        # 3. Delete from file_index
        fm._db.execute(
            "DELETE FROM file_index WHERE storage_key = ?", (key,))
        fm._db.commit()
        return {"ok": True, "key": key}

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
        mode = str(kb_cfg.get("auto_context_mode", "")).strip().lower()
        if mode not in {"off", "on_demand", "always"}:
            if "auto_context" in kb_cfg:
                mode = "always" if kb_cfg.get("auto_context", True) else "off"
            else:
                mode = "on_demand"
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
            "max_file_size": kb_cfg.get("max_file_size_mb", 50),  # backwards-compatible alias
            "auto_context_mode": mode,
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
        raw_max_size = body.get("max_file_size_mb", body.get("max_file_size", 50))
        kb_cfg["max_file_size_mb"] = max(1, min(200, int(raw_max_size)))
        if "auto_context_mode" in body:
            mode = str(body.get("auto_context_mode", "on_demand")).strip().lower()
            if mode not in {"off", "on_demand", "always"}:
                mode = "on_demand"
            kb_cfg["auto_context_mode"] = mode
        elif "auto_context" in body:
            # Backward-compatible bool API.
            kb_cfg["auto_context_mode"] = "always" if body.get("auto_context", True) else "off"

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
        # Backward-compatible aliases for dashboard/UI consumers.
        for d in docs:
            if "chunks" not in d and "chunk_count" in d:
                d["chunks"] = d.get("chunk_count", 0)
            if "pages" not in d and "page_count" in d:
                d["pages"] = d.get("page_count", 0)
        stats = await kb.get_stats()
        return {"documents": docs, "stats": stats}

    @app.get("/api/knowledge_base/documents/{doc_id}")
    async def api_kb_document(doc_id: str):
        """Get a single knowledge base document by id."""
        kb = getattr(agent, '_knowledge_base', None)
        if not kb:
            return JSONResponse(status_code=400,
                                content={"detail": "Knowledge Base is not enabled"})

        if hasattr(kb, "get_document"):
            doc = await kb.get_document(doc_id)
        else:
            docs = await kb.list_documents()
            doc = next(
                (d for d in docs if d.get("id") == doc_id or d.get("name") == doc_id),
                None,
            )
        if not doc:
            return JSONResponse(status_code=404,
                                content={"detail": "Document not found"})
        if "chunks" not in doc and "chunk_count" in doc:
            doc["chunks"] = doc.get("chunk_count", 0)
        if "pages" not in doc and "page_count" in doc:
            doc["pages"] = doc.get("page_count", 0)
        return doc

    async def _handle_document_upload(file: UploadFile):
        """Store, analyze, and optionally index a user document."""
        pipeline = getattr(agent, "_document_pipeline", None)
        if not pipeline:
            return JSONResponse(status_code=500, content={"detail": "Document pipeline is not available"})

        filename = (file.filename or "").strip()
        if not filename:
            return JSONResponse(status_code=400, content={"detail": "Filename is required"})

        safe_name = Path(filename).name
        safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", safe_name).strip("._ ")
        if not safe_name:
            safe_name = "upload.bin"

        kb_cfg = agent.config.get("knowledge_base", {})
        doc_cfg = agent.config.get("documents", {})
        try:
            max_size_mb = max(
                1,
                int(doc_cfg.get("max_file_size_mb", kb_cfg.get("max_file_size_mb", 50))),
            )
        except (TypeError, ValueError):
            max_size_mb = 50
        max_size_bytes = max_size_mb * 1024 * 1024

        buf = bytearray()
        try:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > max_size_bytes:
                    raise HTTPException(status_code=413, detail=f"File too large (max {max_size_mb}MB)")

            if not buf:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            result = await agent.process_document_upload(
                bytes(buf),
                safe_name,
                source="dashboard",
                user_id=_dashboard_user_id(),
                mime_type=str(getattr(file, "content_type", "") or ""),
            )
            payload = dict(result)
            kb_payload = dict(payload.get("knowledge_base") or {})
            payload.setdefault("status", str(kb_payload.get("status") or "ok"))
            payload.setdefault("doc_id", str(kb_payload.get("doc_id") or payload.get("review_id") or ""))
            payload.setdefault("name", safe_name)
            payload.setdefault("chunks", int(kb_payload.get("chunks", 0) or 0))
            payload["ok"] = True
            return payload
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
        except (ImportError, ModuleNotFoundError) as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})
        finally:
            await file.close()

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
        except (ImportError, ModuleNotFoundError) as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        except FileNotFoundError as e:
            return JSONResponse(status_code=404, content={"detail": str(e)})
        except ValueError as e:
            return JSONResponse(status_code=400, content={"detail": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"detail": str(e)})

    @app.post("/api/knowledge_base/upload")
    async def api_kb_upload(file: UploadFile = File(...)):
        """Upload, analyze, and index a document."""
        return await _handle_document_upload(file)

    @app.post("/api/documents/upload")
    async def api_documents_upload(file: UploadFile = File(...)):
        """Primary document upload endpoint for storage + analysis + KB indexing."""
        return await _handle_document_upload(file)

    @app.get("/api/documents/reviews")
    async def api_document_reviews(limit: int = 20, user_id: str = DASHBOARD_USER):
        pipeline = getattr(agent, "_document_pipeline", None)
        if not pipeline:
            return {"reviews": []}
        uid = user_id or _dashboard_user_id()
        return {"reviews": pipeline.list_reviews(uid, limit=limit)}

    @app.get("/api/documents/reviews/{review_id}")
    async def api_document_review(review_id: str):
        pipeline = getattr(agent, "_document_pipeline", None)
        if not pipeline:
            return JSONResponse(status_code=404, content={"detail": "Document review not found"})
        review = pipeline.get_review(review_id)
        if not review:
            return JSONResponse(status_code=404, content={"detail": "Document review not found"})
        return review

    @app.get("/api/calendar/events")
    async def api_calendar_events(limit: int = 20, user_id: str = DASHBOARD_USER):
        pipeline = getattr(agent, "_document_pipeline", None)
        if not pipeline:
            return {"events": []}
        uid = user_id or _dashboard_user_id()
        return {"events": pipeline.list_calendar_events(uid, limit=limit)}

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
        def _to_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            s = str(v or "").strip().lower()
            return s in {"1", "true", "yes", "on"}
        name = body.get("name", "").strip()
        query = body.get("query", "").strip()
        run_at = body.get("run_at") or None
        cron = body.get("cron") or None
        background = _to_bool(body.get("background", False))
        try:
            priority = int(body.get("priority", 5) or 5)
            retry_delay_sec = int(body.get("retry_delay_sec", 45) or 45)
            max_attempts = int(body.get("max_attempts", 0) or 0)
        except (TypeError, ValueError):
            raise HTTPException(400, "priority/retry_delay_sec/max_attempts must be integers")
        if not name or not query:
            raise HTTPException(400, "name and query are required")
        if background and cron:
            raise HTTPException(400, "background tasks only support one-shot execution")
        if background and not run_at:
            from datetime import datetime
            run_at = datetime.now().isoformat(timespec='seconds')
        if not run_at and not cron:
            raise HTTPException(400, "run_at or cron is required")
        task_type = "recurring" if cron else "one_shot"
        try:
            task = tm.add_task(
                name=name, query=query, user_id=_dashboard_user_id(),
                task_type=task_type, run_at=run_at, cron_expr=cron,
                background=background, priority=priority,
                retry_delay_sec=retry_delay_sec, max_attempts=max_attempts,
                source="dashboard")
            if background:
                daemon = getattr(agent, "_background_task_daemon", None)
                if daemon:
                    await daemon.start()
            return task
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.put("/api/tasks/{task_id}")
    async def api_task_update(task_id: int, body: dict):
        """Update an existing task from dashboard settings."""
        tm = getattr(agent, '_task_manager', None)
        if not tm:
            raise HTTPException(400, "Tasks not available")
        task = tm.get_task(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        if str(task.get("status")) == "running":
            raise HTTPException(409, "Cannot edit a task while it is running")

        name = str(body.get("name", task.get("name", "")) or "").strip()
        query = str(body.get("query", task.get("query", "")) or "").strip()
        run_at = body.get("run_at", task.get("run_at"))
        cron = body.get("cron", task.get("cron_expr"))

        def _to_bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            s = str(v or "").strip().lower()
            return s in {"1", "true", "yes", "on"}

        background = _to_bool(body.get("background", task.get("background", 0)))
        try:
            priority = int(body.get("priority", task.get("priority", 5)) or 5)
            retry_delay_sec = int(body.get("retry_delay_sec", task.get("retry_delay_sec", 45)) or 45)
            max_attempts = int(body.get("max_attempts", task.get("max_attempts", 0)) or 0)
        except (TypeError, ValueError):
            raise HTTPException(400, "priority/retry_delay_sec/max_attempts must be integers")

        try:
            updated = tm.update_task(
                task_id,
                name=name,
                query=query,
                run_at=run_at,
                cron_expr=cron,
                background=1 if background else 0,
                priority=priority,
                retry_delay_sec=retry_delay_sec,
                max_attempts=max_attempts,
            )
            if not updated:
                raise HTTPException(404, "Task not found")
            if int(updated.get("background") or 0):
                daemon = getattr(agent, "_background_task_daemon", None)
                if daemon:
                    await daemon.start()
            return updated
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
        from ..tasks import _notify_telegram, _publish_task_message
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
                _publish_task_message(
                    agent, task,
                    f"✅ Task \"{task['name']}\" completed.\n\n{result[:3500]}",
                )
                await _notify_telegram(agent, task, result)
            except Exception as e:
                tm.mark_failed(task_id, str(e))
                agent._ws_broadcast("task_failed", {
                    "task_id": task_id, "name": task["name"],
                    "error": str(e)[:200],
                })
                _publish_task_message(
                    agent, task,
                    f"❌ Task \"{task['name']}\" failed: {str(e)[:1200]}",
                )

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

    # ── Goals API ─────────────────────────────

    @app.get("/api/goals")
    async def api_goals_list(status: str = "", limit: int = 50):
        """List goals for current dashboard user."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            return []
        statuses = [s.strip().lower() for s in str(status or "").split(",") if s.strip()]
        return gm.list_goals(
            user_id=_dashboard_user_id(),
            statuses=statuses or None,
            limit=max(1, min(int(limit or 50), 300)),
        )

    @app.get("/api/goals/summary")
    async def api_goals_summary(limit: int = 8):
        """Goal progress summary + coordinator daemon state."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            return {
                "counts": {"active": 0, "running": 0, "paused": 0, "completed": 0, "cancelled": 0, "failed": 0},
                "avg_progress": 0.0,
                "goals": [],
                "coordinator": {"enabled": False, "running": False},
                "running": [],
                "health": {
                    "state": "disabled",
                    "stalled_count": 0,
                    "attention_count": 0,
                    "throughput_total": 0,
                    "failure_total": 0,
                },
                "lanes": {"running": [], "pipeline": [], "attention": [], "recent": []},
            }
        user_id = _dashboard_user_id()
        lim = max(1, min(int(limit or 8), 30))
        summary = gm.summary(user_id=user_id, limit=lim)
        daemon = getattr(agent, "_goal_coordinator", None)
        summary["coordinator"] = daemon.state() if daemon else {"enabled": False, "running": False}
        summary["running"] = daemon.get_active_goals() if daemon else []

        focus_goals = gm.list_goals(
            user_id=user_id,
            statuses=["running", "active", "paused", "failed"],
            limit=max(lim, 12),
        )
        recent_goals = gm.list_goals(
            user_id=user_id,
            statuses=["completed", "cancelled"],
            limit=min(max(4, lim // 2), 8),
        )

        def _clip(text: str, limit: int = 220) -> str:
            raw = str(text or "").strip()
            if len(raw) <= limit:
                return raw
            return raw[:limit - 1].rstrip() + "…"

        def _goal_item(goal: dict) -> dict:
            cfg = goal.get("config") if isinstance(goal.get("config"), dict) else {}
            return {
                "id": int(goal.get("id") or 0),
                "title": str(goal.get("title") or ""),
                "status": str(goal.get("status") or ""),
                "goal_type": str(goal.get("goal_type") or "generic"),
                "priority": int(goal.get("priority") or 5),
                "progress": float(goal.get("progress") or 0.0),
                "current_phase": str(goal.get("current_phase") or "planned"),
                "next_run_at": goal.get("next_run_at"),
                "updated_at": goal.get("updated_at"),
                "cycle_count": int(goal.get("cycle_count") or 0),
                "max_cycles": int(goal.get("max_cycles") or 0),
                "cooldown_sec": int(goal.get("cooldown_sec") or 0),
                "stalled_cycles": int(goal.get("stalled_cycles") or 0),
                "plan_version": int(goal.get("plan_version") or 0),
                "summary": _clip(goal.get("last_result") or goal.get("strategy") or goal.get("objective") or "", 220),
                "objective": _clip(goal.get("objective") or "", 200),
                "workspace": str(cfg.get("workspace") or ""),
                "local_model": str(cfg.get("local_model") or ""),
                "stop_at": cfg.get("stop_at"),
            }

        running_ids = {
            int(item.get("goal_id") or 0) for item in list(summary.get("running") or [])
            if int(item.get("goal_id") or 0) > 0
        }
        running_lane = []
        for goal in summary.get("running") or []:
            cfg = goal.get("config") if isinstance(goal.get("config"), dict) else {}
            running_lane.append({
                "id": int(goal.get("goal_id") or 0),
                "title": str(goal.get("title") or ""),
                "status": "running",
                "goal_type": str(goal.get("goal_type") or "generic"),
                "priority": int(goal.get("priority") or 5),
                "progress": float(goal.get("progress") or 0.0),
                "current_phase": str(goal.get("current_phase") or "executing"),
                "next_run_at": None,
                "updated_at": goal.get("started_at"),
                "cycle_count": 0,
                "max_cycles": 0,
                "cooldown_sec": 0,
                "stalled_cycles": int(goal.get("stalled_cycles") or 0),
                "plan_version": int(goal.get("plan_version") or 0),
                "summary": _clip(goal.get("last_result") or goal.get("strategy") or "", 220),
                "objective": "",
                "step_title": str(goal.get("step_title") or ""),
                "workspace": str(cfg.get("workspace") or ""),
                "local_model": str(cfg.get("local_model") or ""),
                "stop_at": cfg.get("stop_at"),
            })

        pipeline_lane = []
        attention_lane = []
        for goal in focus_goals:
            gid = int(goal.get("id") or 0)
            status = str(goal.get("status") or "")
            item = _goal_item(goal)
            if gid in running_ids or status == "running":
                continue
            if status in {"paused", "failed"} or int(goal.get("stalled_cycles") or 0) > 0:
                attention_lane.append(item)
            else:
                pipeline_lane.append(item)

        stalled_count = sum(1 for goal in focus_goals if int(goal.get("stalled_cycles") or 0) > 0)
        attention_count = len(attention_lane)
        coord = summary["coordinator"]
        open_goal_count = (
            int(summary.get("counts", {}).get("active") or 0)
            + int(summary.get("counts", {}).get("running") or 0)
            + int(summary.get("counts", {}).get("paused") or 0)
            + int(summary.get("counts", {}).get("failed") or 0)
        )
        has_goal_work = bool(open_goal_count or running_lane or pipeline_lane or attention_lane or int(coord.get("pending") or 0) > 0)
        health_state = "disabled"
        if coord.get("enabled"):
            if not has_goal_work:
                health_state = "idle"
            elif coord.get("running"):
                health_state = "running"
            elif coord.get("last_pause_reason"):
                health_state = "paused"
            else:
                health_state = "idle"

        summary["health"] = {
            "state": health_state,
            "stalled_count": stalled_count,
            "attention_count": attention_count,
            "throughput_total": int(coord.get("processed_total") or 0),
            "failure_total": int(coord.get("failed_total") or 0),
            "planned_total": int(coord.get("planned_total") or 0),
            "replanned_total": int(coord.get("replanned_total") or 0),
        }
        summary["lanes"] = {
            "running": running_lane[:6],
            "pipeline": pipeline_lane[:6],
            "attention": attention_lane[:6],
            "recent": [_goal_item(goal) for goal in recent_goals[:6]],
        }
        return summary

    @app.post("/api/goals")
    async def api_goal_create(body: dict):
        """Create a long-running goal."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            raise HTTPException(400, "Goals not available")
        title = str(body.get("title", "") or "").strip()
        objective = str(body.get("objective", "") or "").strip()
        if not title or not objective:
            raise HTTPException(400, "title and objective are required")
        try:
            priority = int(body.get("priority", 5) or 5)
            target_steps = int(body.get("target_steps", 4) or 4)
            max_cycles = int(body.get("max_cycles", 0) or 0)
            cooldown_sec = int(body.get("cooldown_sec", 90) or 90)
        except (TypeError, ValueError):
            raise HTTPException(400, "priority/target_steps/max_cycles/cooldown_sec must be integers")
        goal_type = str(body.get("goal_type", "generic") or "generic").strip().lower()
        if goal_type not in {"generic", "autonomous_coding", "self_improvement"}:
            raise HTTPException(400, "goal_type must be generic, autonomous_coding, or self_improvement")
        config = body.get("config")
        if config is not None and not isinstance(config, dict):
            raise HTTPException(400, "config must be an object")
        if goal_type in {"autonomous_coding", "self_improvement"}:
            from ..night_coding import normalize_session_config

            config = normalize_session_config(config, agent.config)
            if not str(config.get("workspace") or "").strip():
                raise HTTPException(400, "Autonomous coding goals require workspace")
        goal = gm.add_goal(
            title=title,
            objective=objective,
            user_id=_dashboard_user_id(),
            priority=priority,
            target_steps=target_steps,
            max_cycles=max_cycles,
            cooldown_sec=cooldown_sec,
            source="dashboard",
            goal_type=goal_type,
            config=config or {},
        )
        daemon = getattr(agent, "_goal_coordinator", None)
        if daemon:
            with suppress(Exception):
                await daemon.start()
        agent._ws_broadcast("goal_created", {
            "goal_id": goal.get("id"),
            "title": goal.get("title", ""),
            "user_id": goal.get("user_id", ""),
            "status": goal.get("status", ""),
            "goal_type": goal.get("goal_type", "generic"),
            "progress": float(goal.get("progress") or 0.0),
        })
        return goal

    @app.get("/api/goals/{goal_id}/status")
    async def api_goal_status(goal_id: int):
        """Get goal status + recent events."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            raise HTTPException(400, "Goals not available")
        goal = gm.get_goal(goal_id)
        if not goal or str(goal.get("user_id", "")) != _dashboard_user_id():
            raise HTTPException(404, "Goal not found")
        payload = {
            "goal": goal,
            "events": gm.get_goal_events(goal_id, limit=25),
        }
        if hasattr(gm, "get_active_plan"):
            with suppress(Exception):
                payload["plan"] = gm.get_active_plan(goal_id)
        if hasattr(gm, "get_plan_history"):
            with suppress(Exception):
                payload["plan_history"] = gm.get_plan_history(goal_id, limit=5)
        if hasattr(gm, "get_recent_attempts"):
            with suppress(Exception):
                payload["recent_attempts"] = gm.get_recent_attempts(goal_id, limit=15)
        if hasattr(gm, "build_goal_report"):
            with suppress(Exception):
                payload["report"] = gm.build_goal_report(goal_id, attempt_limit=15)
        return payload

    @app.get("/api/goals/{goal_id}/report")
    async def api_goal_report(goal_id: int, format: str = "markdown"):
        """Export a goal session report."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            raise HTTPException(400, "Goals not available")
        goal = gm.get_goal(goal_id)
        if not goal or str(goal.get("user_id", "")) != _dashboard_user_id():
            raise HTTPException(404, "Goal not found")
        fmt = str(format or "markdown").strip().lower()
        if fmt in {"json"}:
            return {
                "goal": goal,
                "report": gm.build_goal_report(goal_id, attempt_limit=20),
                "plan": gm.get_active_plan(goal_id) if hasattr(gm, "get_active_plan") else None,
                "recent_attempts": gm.get_recent_attempts(goal_id, limit=20) if hasattr(gm, "get_recent_attempts") else [],
            }
        if fmt not in {"markdown", "md"}:
            raise HTTPException(400, "Unsupported format")
        body = gm.render_goal_report_markdown(goal_id, attempt_limit=20)
        filename = f"goal-{goal_id}-report.md"
        return Response(
            content=body,
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post("/api/goals/{goal_id}/plan")
    async def api_goal_plan_save(goal_id: int, body: dict):
        """Manually replace active goal plan from dashboard editor."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            raise HTTPException(400, "Goals not available")
        goal = gm.get_goal(goal_id)
        if not goal or str(goal.get("user_id", "")) != _dashboard_user_id():
            raise HTTPException(404, "Goal not found")

        strategy = str(body.get("strategy", "") or "").strip()
        steps = body.get("steps", [])
        if not isinstance(steps, list):
            raise HTTPException(400, "steps must be a list")
        if not strategy and not steps:
            raise HTTPException(400, "strategy or steps required")

        plan = gm.upsert_plan(
            goal_id,
            strategy=strategy,
            steps=steps,
            trigger="manual_edit",
        )
        if not plan:
            raise HTTPException(500, "Failed to save plan")
        updated_goal = gm.get_goal(goal_id) or goal
        agent._ws_broadcast("goal_plan_updated", {
            "goal_id": goal_id,
            "title": updated_goal.get("title", ""),
            "user_id": updated_goal.get("user_id", ""),
            "version": int(plan.get("version") or 0),
            "trigger": "manual_edit",
            "steps": len(plan.get("steps") or []),
            "strategy": str(plan.get("strategy") or "")[:300],
        })
        agent._ws_broadcast("goal_updated", {
            "goal_id": updated_goal.get("id"),
            "title": updated_goal.get("title", ""),
            "user_id": updated_goal.get("user_id", ""),
            "status": updated_goal.get("status", ""),
            "progress": float(updated_goal.get("progress") or 0.0),
        })
        return {"ok": True, "goal": updated_goal, "plan": plan}

    @app.post("/api/goals/{goal_id}/replan")
    async def api_goal_replan(goal_id: int):
        """Force goal replanning from dashboard inspector."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            raise HTTPException(400, "Goals not available")
        goal = gm.get_goal(goal_id)
        if not goal or str(goal.get("user_id", "")) != _dashboard_user_id():
            raise HTTPException(404, "Goal not found")

        daemon = getattr(agent, "_goal_coordinator", None)
        plan = None
        if daemon and hasattr(daemon, "_plan_goal"):
            with suppress(Exception):
                plan = await daemon._plan_goal(goal, "manual_replan")
        if not plan:
            current = gm.get_active_plan(goal_id) if hasattr(gm, "get_active_plan") else None
            plan = gm.upsert_plan(
                goal_id,
                strategy=str(current.get("strategy", "") if current else goal.get("strategy", "")).strip(),
                steps=list((current or {}).get("steps") or []),
                trigger="manual_replan",
            )
        if not plan:
            raise HTTPException(500, "Failed to replan goal")

        updated_goal = gm.get_goal(goal_id) or goal
        agent._ws_broadcast("goal_replanned", {
            "goal_id": goal_id,
            "title": updated_goal.get("title", ""),
            "user_id": updated_goal.get("user_id", ""),
            "reason": "manual_replan",
            "version": int(plan.get("version") or 0),
            "next_step": ((plan.get("steps") or [{}])[0] or {}).get("title", ""),
        })
        return {"ok": True, "goal": updated_goal, "plan": plan}

    @app.post("/api/goals/{goal_id}/pause")
    async def api_goal_pause(goal_id: int, body: dict | None = None):
        """Pause goal (or resume when paused=false)."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            raise HTTPException(400, "Goals not available")
        current = gm.get_goal(goal_id)
        if not current or str(current.get("user_id", "")) != _dashboard_user_id():
            raise HTTPException(404, "Goal not found")
        payload = body or {}
        paused = bool(payload.get("paused", True))
        updated = gm.pause_goal(goal_id) if paused else gm.resume_goal(goal_id)
        if not updated:
            raise HTTPException(409, "Goal cannot be changed in current state")
        agent._ws_broadcast("goal_updated", {
            "goal_id": updated.get("id"),
            "title": updated.get("title", ""),
            "user_id": updated.get("user_id", ""),
            "status": updated.get("status", ""),
            "progress": float(updated.get("progress") or 0.0),
        })
        return updated

    @app.post("/api/goals/{goal_id}/cancel")
    async def api_goal_cancel(goal_id: int):
        """Cancel goal execution."""
        gm = getattr(agent, "_goal_manager", None)
        if not gm:
            raise HTTPException(400, "Goals not available")
        current = gm.get_goal(goal_id)
        if not current or str(current.get("user_id", "")) != _dashboard_user_id():
            raise HTTPException(404, "Goal not found")
        updated = gm.cancel_goal(goal_id)
        if not updated:
            raise HTTPException(409, "Goal cannot be cancelled in current state")
        agent._ws_broadcast("goal_cancelled", {
            "goal_id": updated.get("id"),
            "title": updated.get("title", ""),
            "user_id": updated.get("user_id", ""),
            "status": updated.get("status", ""),
            "progress": float(updated.get("progress") or 0.0),
        })
        return updated

    logger.info("Dashboard routes mounted")
