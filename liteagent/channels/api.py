"""FastAPI channel for LiteAgent — REST + streaming API."""

import json
import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


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


def create_app(agent):
    """Create FastAPI application wired to agent."""
    try:
        from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
        from fastapi.responses import StreamingResponse, JSONResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required: pip install liteagent[api]"
        )

    app = FastAPI(title="LiteAgent API", version="0.1.0",
                  description="Ultra-lightweight AI agent API")

    # Rate limiting
    rate_cfg = agent.config.get("channels", {}).get("api", {}).get("rate_limit", {})
    rpm = rate_cfg.get("requests_per_minute", 60)
    limiter = RateLimiter(rpm=rpm)

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path.startswith("/api/") or request.url.path == "/chat":
            client_ip = request.client.host if request.client else "unknown"
            if not limiter.check(client_ip):
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
        return await call_next(request)

    class ChatRequest(BaseModel):
        message: str
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
                yield f"data: {json.dumps({'done': True})}\n\n"
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
        user_id: str = Form("api-user"),
        file: UploadFile | None = File(None),
    ):
        """Send a message with optional image attachment."""
        ALLOWED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
        content_blocks = [{"type": "text", "text": message}]
        if file:
            if file.content_type not in ALLOWED_TYPES:
                raise HTTPException(400, f"Unsupported file type: {file.content_type}")
            import base64
            data = await file.read()
            if len(data) > 5 * 1024 * 1024:
                raise HTTPException(400, "File too large (max 5MB)")
            b64 = base64.b64encode(data).decode()
            content_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": file.content_type, "data": b64}
            })
        response = await agent.run(content_blocks, user_id)
        return ChatResponse(response=response)

    @app.get("/health")
    async def health():
        return {"status": "ok", "agent": "liteagent", "version": "0.1.0"}

    # Mount dashboard routes (overview, memories, tools, usage, config, history)
    from .dashboard import mount_dashboard
    mount_dashboard(app, agent)

    return app


def run_api(agent, config: dict):
    """Start the API server."""
    run_api_with_scheduler(agent, config, scheduler=None)


def run_api_with_scheduler(agent, config: dict, scheduler=None):
    """Start the API server with optional scheduler integration."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError("uvicorn is required: pip install liteagent[api]")

    app = create_app(agent)

    if scheduler:
        @app.on_event("startup")
        async def _start_scheduler():
            scheduler.start()

    host = config.get("host", "0.0.0.0")
    port = config.get("port", 8080)
    logger.info("Starting API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
