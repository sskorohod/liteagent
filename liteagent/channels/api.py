"""FastAPI channel for LiteAgent — REST + streaming API + WebSocket hub."""

import asyncio
import hashlib
import json
import logging
import secrets
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# WEBSOCKET HUB
# ══════════════════════════════════════════

class WSHub:
    """Broadcast WebSocket hub for real-time dashboard events."""

    def __init__(self):
        self._clients: set = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def disconnect(self, ws):
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, event_type: str, data: dict):
        """Broadcast event to all connected WebSocket clients."""
        if not self._clients:
            return
        message = json.dumps({"type": event_type, "data": data}, default=str)
        async with self._lock:
            dead = set()
            for ws in self._clients:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

    @property
    def client_count(self) -> int:
        return len(self._clients)


class RateLimiter:
    """In-memory sliding window rate limiter."""

    def __init__(self, rpm: int = 30):
        self.rpm = rpm
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> bool:
        now = time.time()
        window = [t for t in self._requests[client_id] if now - t < 60]
        self._requests[client_id] = window
        if len(window) >= self.rpm:
            return False
        window.append(now)
        return True


# ══════════════════════════════════════════
# SESSION AUTH
# ══════════════════════════════════════════

class SessionStore:
    """In-memory session store with TTL."""

    def __init__(self, ttl: int = 86400):
        self.ttl = ttl
        self._sessions: dict[str, float] = {}  # token → last_used timestamp

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        self._sessions[token] = time.time()
        return token

    def validate(self, token: str) -> bool:
        if token not in self._sessions:
            return False
        last = self._sessions[token]
        if time.time() - last > self.ttl:
            del self._sessions[token]
            return False
        self._sessions[token] = time.time()
        return True

    def revoke(self, token: str):
        self._sessions.pop(token, None)


def create_app(agent):
    """Create FastAPI application wired to agent."""
    try:
        from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, WebSocket, WebSocketDisconnect
        from fastapi.responses import StreamingResponse, JSONResponse
        from pydantic import BaseModel, Field
        from typing import List
    except ImportError:
        raise ImportError(
            "FastAPI is required: pip install liteagent[api]"
        )

    app = FastAPI(title="LiteAgent API", version="0.1.0",
                  description="Ultra-lightweight AI agent API")

    api_cfg = agent.config.get("channels", {}).get("api", {})

    # ── WebSocket Hub (shared across agent + scheduler) ──
    hub = WSHub()
    app.state.ws_hub = hub

    # Wire hub to agent and scheduler
    from ..agent import LiteAgent
    LiteAgent._ws_hub = hub

    sched = getattr(agent, '_scheduler', None)
    if sched:
        sched._ws_hub = hub

    # ── CORS middleware ──
    try:
        from fastapi.middleware.cors import CORSMiddleware
        cors_origins = api_cfg.get("cors_origins", ["*"])
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
            allow_credentials=True,
        )
    except ImportError:
        pass  # CORSMiddleware not available

    # ── Global exception handler ──
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    # ── Session Auth (optional, if password is set) ──
    auth_password = api_cfg.get("password")
    sessions = SessionStore() if auth_password else None

    PUBLIC_PATHS = {"/health", "/api/login", "/api/auth/check", "/favicon.ico"}

    if auth_password:
        @app.post("/api/login")
        async def login(body: dict):
            pwd = body.get("password", "")
            if pwd != auth_password:
                raise HTTPException(status_code=401, detail="Invalid password")
            token = sessions.create()
            response = JSONResponse(content={"ok": True})
            response.set_cookie("session", token, httponly=True, samesite="lax", max_age=86400)
            return response

        @app.post("/api/logout")
        async def logout(request: Request):
            token = request.cookies.get("session", "")
            sessions.revoke(token)
            response = JSONResponse(content={"ok": True})
            response.delete_cookie("session")
            return response

        @app.middleware("http")
        async def auth_middleware(request: Request, call_next):
            path = request.url.path
            if path in PUBLIC_PATHS or path == "/" or path == "/dashboard":
                return await call_next(request)
            # Check session cookie or Authorization header
            token = request.cookies.get("session", "")
            if not token:
                auth_header = request.headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    token = auth_header[7:]
            if not sessions.validate(token):
                if path.startswith("/api/") or path.startswith("/chat"):
                    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
                # For HTML pages, serve them anyway (login overlay handles it)
            return await call_next(request)

    @app.get("/api/auth/check")
    async def auth_check(request: Request):
        if not auth_password:
            return {"authenticated": True, "auth_required": False}
        token = request.cookies.get("session", "")
        if not token:
            auth_header = request.headers.get("authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
        return {
            "authenticated": sessions.validate(token) if sessions else True,
            "auth_required": bool(auth_password),
        }

    # ── Rate limiting ──
    rate_cfg = api_cfg.get("rate_limit", {})
    rpm = rate_cfg.get("requests_per_minute", 60)
    limiter = RateLimiter(rpm=rpm)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/api/") or request.url.path == "/chat":
            client_ip = request.client.host if request.client else "unknown"
            if not limiter.check(client_ip):
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        return await call_next(request)

    # ── Health endpoint (public, no auth) ──
    @app.get("/health")
    async def health():
        return {"status": "ok", "ws_clients": hub.client_count}

    @app.get("/api/providers/health")
    async def provider_health():
        """Circuit breaker status for all LLM providers."""
        if hasattr(agent, '_circuit_breaker'):
            return agent._circuit_breaker.get_status()
        return {}

    @app.get("/api/health/channels")
    async def channels_health():
        """Health status for channels and providers."""
        if hasattr(agent, '_health_monitor'):
            return agent._health_monitor.get_status()
        return {}

    @app.get("/api/hooks")
    async def hooks_info():
        """Registered hook handlers (for dashboard)."""
        if hasattr(agent, 'hooks'):
            return agent.hooks.get_registered()
        return {}

    # ── WebSocket endpoint ──
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        # Check auth for WebSocket if password is set
        if auth_password and sessions:
            # Get session from query params or cookies
            token = ws.query_params.get("token", "")
            if not token:
                cookies = ws.cookies
                token = cookies.get("session", "")
            if not sessions.validate(token):
                await ws.close(code=4001, reason="Unauthorized")
                return

        await hub.connect(ws)
        logger.info("WebSocket client connected (%d total)", hub.client_count)
        try:
            while True:
                # Keep connection alive, handle client pings
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "ping":
                        await ws.send_text(json.dumps({"type": "pong"}))
                except (json.JSONDecodeError, KeyError):
                    pass
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            await hub.disconnect(ws)
            logger.info("WebSocket client disconnected (%d remaining)", hub.client_count)

    class ChatRequest(BaseModel):
        message: str = Field(..., max_length=50000)
        user_id: str = "api-user"
        stream: bool = False
        agent_name: str | None = None

    class ChatResponse(BaseModel):
        response: str

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """Send a message and get a response."""
        if req.stream:
            async def generate():
                async for chunk in agent.stream(req.message, req.user_id):
                    yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream")

        response = await agent.run(req.message, req.user_id)
        return ChatResponse(response=response)

    @app.get("/chat/stream")
    async def chat_stream(message: str, user_id: str = "dashboard-user"):
        """SSE streaming endpoint for real-time chat."""
        import re
        _TOOL_START_RE = re.compile(r'__TOOL_START__(.+?)__TOOL_END__')
        _TOOL_RESULT_RE = re.compile(r'__TOOL_RESULT__(.+?)__TOOL_END__')

        async def event_generator():
            try:
                buffer = ""
                async for chunk in agent.stream(message, user_id):
                    buffer += chunk
                    # Check for tool markers in buffer
                    while True:
                        # Tool start event
                        m = _TOOL_START_RE.search(buffer)
                        if m:
                            # Send any text before the marker
                            pre = buffer[:m.start()].strip()
                            if pre:
                                yield f"data: {json.dumps({'text': pre})}\n\n"
                            # Send structured tool_start event
                            tool_data = json.loads(m.group(1))
                            yield f"data: {json.dumps({'tool_start': tool_data})}\n\n"
                            buffer = buffer[m.end():]
                            continue

                        # Tool result event
                        m = _TOOL_RESULT_RE.search(buffer)
                        if m:
                            pre = buffer[:m.start()].strip()
                            if pre:
                                yield f"data: {json.dumps({'text': pre})}\n\n"
                            tool_data = json.loads(m.group(1))
                            yield f"data: {json.dumps({'tool_result': tool_data})}\n\n"
                            buffer = buffer[m.end():]
                            continue

                        break

                    # Flush text that definitely doesn't contain markers
                    # Keep potential partial markers in buffer
                    if "__TOOL" not in buffer and buffer:
                        yield f"data: {json.dumps({'text': buffer})}\n\n"
                        buffer = ""

                # Flush remaining buffer
                if buffer.strip():
                    yield f"data: {json.dumps({'text': buffer})}\n\n"
                # Include provider/model info in done event
                provider_name = agent.config.get("agent", {}).get("provider", "anthropic")
                model_name = agent.default_model
                yield f"data: {json.dumps({'done': True, 'provider': provider_name, 'model': model_name})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/command")
    async def command(cmd: str, user_id: str = "api-user"):
        """Execute an agent slash command."""
        result = agent.handle_command(cmd, user_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Unknown command")
        return {"response": result}

    @app.post("/chat/multimodal")
    async def chat_multimodal(
        message: str = Form(...),
        user_id: str = Form("dashboard-user"),
        files: List[UploadFile] = File(default=[]),
    ):
        """Send a message with optional file attachments (images, PDFs, text, code)."""
        from ..multimodal import file_to_content_block, file_to_emoji

        content_blocks = [{"type": "text", "text": message}]
        file_descriptions = []

        for file in files:
            data = await file.read()
            fname = file.filename or "unknown"
            ct = file.content_type or ""
            try:
                block = file_to_content_block(data, fname, ct)
                content_blocks.append(block)
                file_descriptions.append(file_to_emoji(block, fname))
            except ValueError as e:
                raise HTTPException(400, str(e))

        response = await agent.run(content_blocks, user_id)
        return {"response": response, "files_processed": file_descriptions}

    # Mount dashboard routes (overview, memories, tools, usage, config, history)
    from .dashboard import mount_dashboard
    mount_dashboard(app, agent)

    return app


def run_api(agent, config: dict):
    """Start the API server."""
    run_api_with_scheduler(agent, config, scheduler=None)


def run_api_with_scheduler(agent, config: dict, scheduler=None, full_config: dict | None = None):
    """Start the API server with optional scheduler integration."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required: pip install liteagent[api]")

    app = create_app(agent)

    # Wire scheduler hub if created after app
    if scheduler and hasattr(app.state, 'ws_hub'):
        scheduler._ws_hub = app.state.ws_hub

    @app.on_event("startup")
    async def _startup_tasks():
        # Eagerly load MCP servers so tools appear in dashboard immediately
        try:
            await agent._ensure_mcp_loaded()
        except Exception as e:
            logger.warning("MCP startup load failed: %s", e)
        # Start scheduler if configured
        if scheduler:
            scheduler.start()
        # Start config watcher if full config available
        if full_config:
            config_path = full_config.get("_config_path")
            if config_path:
                from ..config_watcher import ConfigWatcher
                watcher = ConfigWatcher(config_path, agent, scheduler)
                watcher.start()
                app.state.config_watcher = watcher

    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8080)

    # Auto-open dashboard in browser
    import webbrowser
    import threading
    open_host = "localhost" if host == "0.0.0.0" else host
    dashboard_url = f"http://{open_host}:{port}"
    print(f"\n   Dashboard: {dashboard_url}\n")
    threading.Timer(1.5, webbrowser.open, args=[dashboard_url]).start()

    logger.info("Starting API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
