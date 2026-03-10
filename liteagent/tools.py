"""Tool registry — decorator-based local tools + MCP support."""

import asyncio
import inspect
import json
import logging
import os
import re
import shlex
import subprocess
from typing import Any, Callable, get_type_hints

logger = logging.getLogger(__name__)

TOOL_SELECTION_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this",
    "что", "как", "для", "это", "или", "also", "then",
}

# Dangerous shell patterns to block (last-resort safety net)
DANGEROUS_COMMAND_PATTERNS = [
    "rm -rf /", "rm -fr /", "mkfs", "dd if=", "> /dev/sd",
    ":(){ :", "chmod -R 777 /", "mv /* ", "wget -O- | sh",
    "curl -s | sh", "echo '' > /etc/", "format c:",
]

# Allowed commands for exec_command (whitelist approach)
COMMAND_ALLOWLIST = {
    "ls", "cat", "head", "tail", "wc", "find", "grep", "rg", "ag",
    "git", "python", "python3", "python3.11", "node", "npm", "npx",
    "pip", "pip3", "poetry", "make", "cargo", "go", "rustc",
    "date", "echo", "pwd", "which", "whoami", "file", "stat",
    "diff", "sort", "uniq", "tee", "tr", "cut", "xargs",
    "curl", "wget", "ssh", "scp", "rsync",
    "mkdir", "cp", "mv", "touch", "ln", "chmod", "chown",
    "tar", "zip", "unzip", "gzip", "gunzip",
    "jq", "sed", "awk", "bc", "env", "printenv",
    "docker", "docker-compose", "kubectl",
    "ollama", "brew",
}

# ── Secret scanning patterns for write_file ─────────────────
SECRET_PATTERNS = [
    (re.compile(r'(?:sk-|sk-proj-)[A-Za-z0-9_-]{20,}'), "OpenAI API key"),
    (re.compile(r'AIza[A-Za-z0-9_-]{35}'), "Google API key"),
    (re.compile(r'AKIA[A-Z0-9]{16}'), "AWS Access Key ID"),
    (re.compile(r'ghp_[A-Za-z0-9]{36,}'), "GitHub personal access token"),
    (re.compile(r'gho_[A-Za-z0-9]{36,}'), "GitHub OAuth token"),
    (re.compile(r'glpat-[A-Za-z0-9_-]{20,}'), "GitLab personal access token"),
    (re.compile(r'xox[boaprs]-[A-Za-z0-9-]{10,}'), "Slack token"),
    (re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'), "SSH/TLS private key"),
    (re.compile(r'-----BEGIN PGP PRIVATE KEY BLOCK-----'), "PGP private key"),
    (re.compile(r'(?:^|["\'])(?:eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+)'), "JWT token"),
    (re.compile(r'(?:heroku|npm|pypi|nuget)[_-]?(?:api[_-]?key|token)\s*[:=]\s*\S{10,}', re.I), "Service token"),
]


def _scan_secrets(content: str) -> str | None:
    """Scan content for secret patterns. Returns first match description or None."""
    for pattern, label in SECRET_PATTERNS:
        if pattern.search(content):
            return label
    return None


def _looks_like_background_shell_command(command: str) -> bool:
    """Heuristic: shell command intentionally launches a long-lived process in background."""
    cmd = (command or "").strip()
    if not cmd:
        return False
    if cmd.endswith("&"):
        return True
    lowered = cmd.lower()
    return " nohup " in f" {lowered} " or lowered.startswith("nohup ")


def _looks_like_foreground_server_command(command: str) -> bool:
    """Heuristic: command likely starts a long-lived dev/app server in foreground."""
    lowered = (command or "").strip().lower()
    if not lowered or _looks_like_background_shell_command(lowered):
        return False
    server_markers = (
        "uvicorn ",
        " flask run",
        "flask run",
        "django-admin runserver",
        "manage.py runserver",
        "npm run dev",
        "npm start",
        "pnpm dev",
        "yarn dev",
        "vite",
        "next dev",
        "bun dev",
        "python3 main.py",
        "python main.py",
        "node server.js",
        "node app.js",
    )
    return any(marker in lowered for marker in server_markers)


# Commands that require explicit user approval before execution
APPROVAL_REQUIRED_COMMANDS = {
    "curl", "wget", "ssh", "scp", "rsync",
    "docker", "docker-compose", "kubectl",
    "chmod", "chown",
}

# Sensitive path components that should never be accessible by LLM tools
SENSITIVE_PATH_COMPONENTS = {
    ".ssh", ".gnupg", ".gpg", ".aws", ".azure", ".gcloud",
    "keys.json", "auth_token", ".env",
    ".liteagent/keys.json", ".liteagent/auth_token",
    "id_rsa", "id_ed25519", "id_ecdsa",
    "credentials", ".netrc", ".npmrc",
}


def _validate_path(path: str, sandbox_root: str | None = None) -> tuple[str, str | None]:
    """Validate and resolve a file path for security.

    Returns (resolved_path, error_or_none). If error is not None, access should be denied.
    """
    resolved = os.path.realpath(os.path.expanduser(path))

    # Block sensitive paths
    for sensitive in SENSITIVE_PATH_COMPONENTS:
        if sensitive in resolved:
            return resolved, f"Access denied: path contains sensitive component '{sensitive}'"

    # Sandbox check (if configured)
    if sandbox_root:
        root = os.path.realpath(os.path.expanduser(sandbox_root))
        if not resolved.startswith(root + os.sep) and resolved != root:
            return resolved, f"Access denied: path outside sandbox '{root}'"

    return resolved, None


# Type mapping for JSON Schema
_TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


class ToolRegistry:
    """Manages local tools and MCP server tools."""

    # Default per-tool timeouts (seconds)
    DEFAULT_TOOL_TIMEOUTS: dict[str, float] = {
        "web_fetch": 30, "web_search": 20, "web_crawl": 60, "web_extract": 30,
        "exec_command": 130, "download_file": 60, "read_file": 30, "write_file": 10,
        "edit_file": 10, "glob_files": 30, "grep_search": 60,
        "send_file_to_user": 10, "memory_search": 10,
    }
    DEFAULT_TIMEOUT: float = 120.0

    # Per-tool result size limits (chars)
    DEFAULT_RESULT_LIMITS: dict[str, int] = {
        "web_fetch": 15000, "web_search": 5000, "web_crawl": 20000,
        "web_extract": 15000, "read_file": 50000, "exec_command": 30000,
        "grep_search": 30000, "glob_files": 15000, "edit_file": 10000,
        "memory_search": 3000,
    }
    DEFAULT_RESULT_LIMIT: int = 10000

    # Tools that benefit from retry on transient errors
    RETRYABLE_TOOLS: set[str] = {
        "web_fetch", "web_search", "web_crawl", "web_extract",
        "download_file", "send_file_to_user",
    }
    TRANSIENT_PATTERNS: list[str] = [
        "timeout", "timed out", "connection", "503", "429", "rate limit",
        "temporary", "unavailable", "reset by peer", "econnrefused",
    ]

    def __init__(self, config: dict | None = None):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}
        # Load overrides from config
        tools_cfg = (config or {}).get("tools", {})
        self._timeouts: dict[str, float] = {
            **self.DEFAULT_TOOL_TIMEOUTS,
            **tools_cfg.get("timeouts", {}),
        }
        self._result_limits: dict[str, int] = {
            **self.DEFAULT_RESULT_LIMITS,
            **tools_cfg.get("result_limits", {}),
        }

    def tool(self, name: str | None = None, description: str | None = None):
        """Decorator to register a tool. Auto-generates JSON schema from type hints."""
        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

            # Generate input_schema from type hints
            hints = get_type_hints(func)
            sig = inspect.signature(func)
            properties = {}
            required = []

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                param_type = hints.get(param_name, str)
                json_type = _TYPE_MAP.get(param_type, "string")
                prop: dict[str, Any] = {"type": json_type}

                # Extract description from docstring
                if func.__doc__:
                    for line in func.__doc__.split("\n"):
                        stripped = line.strip()
                        if stripped.startswith(f"{param_name}:"):
                            prop["description"] = stripped.split(":", 1)[1].strip()

                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
                elif param.default is not None:
                    prop["default"] = param.default

                properties[param_name] = prop

            schema = {
                "type": "object",
                "properties": properties,
            }
            if required:
                schema["required"] = required

            self._tools[tool_name] = {
                "name": tool_name,
                "description": tool_desc.strip(),
                "input_schema": schema,
            }
            self._handlers[tool_name] = func
            return func

        return decorator

    def get_definitions(self) -> list[dict]:
        """Get all tool definitions for LLM."""
        return list(self._tools.values())

    def get_relevant_definitions(self, query: str, top_k: int = 8,
                                  embedder=None) -> list[dict]:
        """Get tool definitions most relevant to the query (semantic selection).
        Falls back to all tools if embedder is unavailable or tool count <= top_k."""
        import math

        all_tools = list(self._tools.values())
        if len(all_tools) <= top_k or embedder is None:
            return all_tools

        # Cache tool embeddings (recompute only when tools change)
        if not hasattr(self, '_tool_emb_cache'):
            self._tool_emb_cache = {}
            self._tool_emb_gen = 0
        current_gen = len(self._tools)
        if current_gen != self._tool_emb_gen:
            self._tool_emb_cache.clear()
            self._tool_emb_gen = current_gen

        try:
            query_emb = embedder.encode(query)
            # Validate query embedding early; some local embedders can return
            # malformed vectors or incompatible shapes during warm-up/failure.
            _ = float(query_emb @ query_emb)
        except Exception as e:
            logger.warning("Semantic tool selection disabled for query embedding error: %s", e)
            return all_tools

        scored = []
        for tool in all_tools:
            name = tool['name']
            try:
                if name not in self._tool_emb_cache:
                    desc = f"{name}: {tool.get('description', '')}"
                    self._tool_emb_cache[name] = embedder.encode(desc)
                tool_emb = self._tool_emb_cache[name]
                # Cosine similarity
                dot = float(query_emb @ tool_emb)
                norm_q = float(math.sqrt(query_emb @ query_emb))
                norm_t = float(math.sqrt(tool_emb @ tool_emb))
                sim = dot / (norm_q * norm_t) if norm_q and norm_t else 0.0
            except Exception as e:
                logger.warning("Skipping semantic rank for tool '%s' due to embedding error: %s", name, e)
                continue
            scored.append((tool, sim))

        if not scored:
            logger.warning("Semantic tool selection produced no valid scores; falling back to all tools")
            return all_tools

        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in scored[:top_k]]

        # Always include memory_search if it exists and wasn't selected
        if self.has_tool("memory_search"):
            names = {t["name"] for t in selected}
            if "memory_search" not in names:
                selected.append(self._tools["memory_search"])

        logger.debug("Semantic tool selection: %d/%d tools for query",
                      len(selected), len(all_tools))
        return selected

    def get_keyword_relevant_definitions(self, query: str, top_k: int = 8) -> list[dict]:
        """Heuristic tool selection when embeddings are unavailable.

        This is intentionally conservative: for coding/debug tasks we prefer a
        compact workspace-editing toolset over exposing every available tool to
        slower local models.
        """
        all_tools = list(self._tools.values())
        if len(all_tools) <= top_k:
            return all_tools

        q = (query or "").lower()
        tokens = {
            tok for tok in re.split(r"[^a-z0-9_а-яё.-]+", q)
            if len(tok) >= 3 and tok not in TOOL_SELECTION_STOPWORDS
        }

        dev_task = any(marker in q for marker in (
            "fix", "bug", "debug", "build", "frontend", "backend", "api",
            "project", "workspace", "full-stack", "full stack", "test",
            "исправ", "почин", "баг", "отлад", "проект", "фронтенд",
            "бэкенд", "бекенд", "сборк", "проверь", "тест",
        ))
        browser_task = any(marker in q for marker in (
            "browser", "chrome", "devtools", "e2e", "ui", "frontend",
            "браузер", "фронтенд", "интерфейс", "ui",
        ))
        kb_task = any(marker in q for marker in (
            "knowledge", "kb", "rag", "document", "docs", "search docs",
            "база знаний", "документ", "rag", "kb_",
        ))
        voice_task = any(marker in q for marker in (
            "voice", "tts", "stt", "audio", "speech",
            "голос", "озвуч", "аудио", "речь",
        ))
        web_task = any(marker in q for marker in (
            "web", "url", "website", "research", "http://", "https://",
            "веб", "сайт", "страниц", "интернет",
        ))
        personal_task = any(marker in q for marker in (
            "remember", "memory", "about me", "who am i",
            "помни", "памят", "обо мне", "как меня зовут",
        ))

        preferred: list[str] = []
        if dev_task:
            preferred.extend([
                "read_file", "glob_files", "grep_search",
                "exec_command", "edit_file", "write_file",
            ])
        if browser_task:
            preferred.extend([
                "chrome_devtools__new_page",
                "chrome_devtools__take_snapshot",
                "chrome_devtools__click",
                "chrome_devtools__fill",
                "chrome_devtools__wait_for",
                "exec_command",
            ])
        if kb_task:
            preferred.extend(["kb_search", "rag_search", "kb_list", "read_file"])
        if voice_task:
            preferred.extend([
                "transcribe_voice", "get_voice_settings",
                "set_voice_settings", "test_tts",
            ])
        if web_task:
            preferred.extend(["web_search", "web_fetch", "web_extract"])
        if personal_task:
            preferred.extend(["memory_search"])

        preferred = [name for name in preferred if name in self._tools]
        preferred = list(dict.fromkeys(preferred))
        preferred_set = set(preferred)

        scored: list[tuple[int, int, dict]] = []
        for index, tool in enumerate(all_tools):
            name = tool["name"]
            text = f"{name} {tool.get('description', '')}".lower()
            score = 0
            if name in preferred_set:
                # Preserve explicit priority order from the preferred list.
                score += 1000 - preferred.index(name)
            for tok in tokens:
                if tok in text:
                    score += 8
                elif tok in name:
                    score += 10
            if dev_task and name in {"read_file", "write_file", "edit_file", "exec_command", "glob_files", "grep_search"}:
                score += 100
            if browser_task and name.startswith("chrome_devtools__"):
                score += 90
            if kb_task and (name.startswith("kb_") or name == "rag_search"):
                score += 70
            if web_task and name.startswith("web_"):
                score += 60
            if voice_task and name in {"transcribe_voice", "get_voice_settings", "set_voice_settings", "test_tts"}:
                score += 60
            if personal_task and name == "memory_search":
                score += 60
            scored.append((score, -index, tool))

        scored.sort(reverse=True)
        limit = max(top_k, len(preferred))
        selected: list[dict] = []
        seen: set[str] = set()
        for score, _, tool in scored:
            name = tool["name"]
            if name in seen:
                continue
            if score <= 0 and len(selected) >= max(1, min(top_k, len(all_tools))):
                break
            selected.append(tool)
            seen.add(name)
            if len(selected) >= limit:
                break

        if not selected:
            return all_tools[:top_k]

        logger.debug("Keyword tool selection: %d/%d tools for query", len(selected), len(all_tools))
        return selected

    async def execute(self, content_blocks) -> list[dict]:
        """Execute tool calls from LLM response (sequentially, legacy)."""
        results = []
        for block in content_blocks:
            if not hasattr(block, 'type') or block.type != "tool_use":
                continue
            result = await self.execute_one(block)
            # Strip _meta for backward compatibility
            results.append({k: v for k, v in result.items() if k != "_meta"})
        return results

    async def execute_parallel(self, content_blocks, on_progress=None) -> list[dict]:
        """Execute tool calls in parallel using asyncio.gather.

        `on_progress` optionally receives per-tool lifecycle events:
        `start` before execution and `done` after completion.
        """
        tool_blocks = [b for b in content_blocks
                       if hasattr(b, 'type') and b.type == "tool_use"]
        if not tool_blocks:
            return []

        async def _emit(event: dict):
            if not on_progress:
                return
            try:
                maybe = on_progress(event)
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception:
                logger.debug("Tool progress callback failed", exc_info=True)

        async def _run_with_progress(block, index: int):
            await _emit({
                "event": "start",
                "index": index,
                "tool_use_id": getattr(block, "id", ""),
                "tool_name": getattr(block, "name", ""),
                "tool_input": getattr(block, "input", {}),
            })
            try:
                result = await self.execute_one(block)
                meta = result.get("_meta", {}) if isinstance(result, dict) else {}
                await _emit({
                    "event": "done",
                    "index": index,
                    "tool_use_id": getattr(block, "id", ""),
                    "tool_name": getattr(block, "name", ""),
                    "tool_input": getattr(block, "input", {}),
                    "duration_ms": int(meta.get("duration_ms") or 0),
                    "error": bool(meta.get("error")),
                    "result_preview": str(meta.get("result_preview", ""))[:300],
                })
                return result
            except Exception as e:
                await _emit({
                    "event": "done",
                    "index": index,
                    "tool_use_id": getattr(block, "id", ""),
                    "tool_name": getattr(block, "name", ""),
                    "tool_input": getattr(block, "input", {}),
                    "duration_ms": 0,
                    "error": True,
                    "result_preview": str(e)[:300],
                })
                raise

        tasks = [_run_with_progress(block, i) for i, block in enumerate(tool_blocks)]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, result in enumerate(raw_results):
            if isinstance(result, Exception):
                block = tool_blocks[i]
                content = f'<tool_output name="{block.name}">\nError: {result}\n</tool_output>'
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "_meta": {
                        "tool_name": block.name,
                        "tool_input": block.input,
                        "duration_ms": 0,
                        "error": True,
                        "result_preview": content[:300],
                    },
                })
            else:
                results.append(result)
        return results

    def _truncate_result(self, tool_name: str, result: str) -> str:
        """Smart truncation: per-tool limits, break on sentence/paragraph boundary."""
        limit = self._result_limits.get(
            tool_name, self.DEFAULT_RESULT_LIMIT)
        if len(result) <= limit:
            return result
        truncated = result[:limit]
        # Find last clean break point (paragraph > sentence > word)
        for sep in ["\n\n", "\n", ". ", " "]:
            pos = truncated.rfind(sep)
            if pos > limit * 0.8:
                truncated = truncated[:pos + len(sep)]
                break
        chars_lost = len(result) - len(truncated)
        return f"{truncated}\n[... {chars_lost} characters truncated]"

    def _is_transient_error(self, error_msg: str) -> bool:
        """Check if error message indicates a transient/retriable failure."""
        msg = error_msg.lower()
        return any(p in msg for p in self.TRANSIENT_PATTERNS)

    async def _execute_handler(self, block) -> str:
        """Run a tool handler, converting sync handlers to async."""
        handler = self._handlers.get(block.name)
        if not handler:
            raise ValueError(f"Unknown tool '{block.name}'")
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**block.input)
        else:
            result = handler(**block.input)
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, default=str)
        return result

    async def execute_one(self, block) -> dict:
        """Execute a single tool call with timeout, retry, and smart truncation."""
        import time
        start = time.time()
        error = False
        result = ""

        handler = self._handlers.get(block.name)
        if not handler:
            result = f"Error: unknown tool '{block.name}'"
            error = True
        else:
            tool_timeout = self._timeouts.get(
                block.name, self.DEFAULT_TIMEOUT)
            # MCP tools also get retry
            is_retryable = (block.name in self.RETRYABLE_TOOLS
                            or "__" in block.name)
            max_retries = 2 if is_retryable else 0

            for attempt in range(max_retries + 1):
                try:
                    result = await asyncio.wait_for(
                        self._execute_handler(block), timeout=tool_timeout)
                    break  # success
                except asyncio.TimeoutError:
                    result = (f"Error: {block.name} timed out "
                              f"after {tool_timeout}s")
                    error = True
                    break  # timeouts are not retriable
                except Exception as e:
                    err_msg = str(e)
                    if (attempt < max_retries
                            and self._is_transient_error(err_msg)):
                        wait = (2 ** attempt) * 0.5
                        logger.debug("Retrying %s (attempt %d/%d): %s",
                                     block.name, attempt + 1, max_retries, e)
                        await asyncio.sleep(wait)
                        continue
                    result = f"Error executing {block.name}: {e}"
                    error = True
                    break

        duration_ms = int((time.time() - start) * 1000)
        raw = self._truncate_result(block.name, str(result))
        content = f"<tool_output name=\"{block.name}\">\n{raw}\n</tool_output>"
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": content,
            "_meta": {
                "tool_name": block.name,
                "tool_input": block.input,
                "duration_ms": duration_ms,
                "error": error,
                "result_preview": content[:300],
            },
        }

    def has_tool(self, name: str) -> bool:
        return name in self._handlers

    def get_mcp_server_info(self) -> list[dict]:
        """Return list of connected MCP servers with tool counts."""
        servers: dict[str, list[str]] = {}
        for tool_name in self._tools:
            if "__" in tool_name:
                server_name = tool_name.split("__")[0]
                servers.setdefault(server_name, [])
                servers[server_name].append(tool_name)
        return [{"name": name, "tool_count": len(tools), "tools": tools}
                for name, tools in servers.items()]

    # ══════════════════════════════════════════
    # MCP SERVER SUPPORT
    # ══════════════════════════════════════════

    async def load_mcp_servers(self, mcp_config: dict):
        """Load MCP servers and discover their tools.

        Supports two transport modes:
        - stdio: {"command": "...", "args": [...]} — spawns subprocess
        - HTTP:  {"url": "http://..."} — connects to HTTP MCP endpoint
        """
        self._mcp_processes: dict[str, asyncio.subprocess.Process] = {}
        self._mcp_http_urls: dict[str, str] = {}  # tool_name → base URL
        self._mcp_id_counter = 0

        for name, server_cfg in mcp_config.items():
            if not server_cfg.get("enabled", True):
                logger.info("MCP server '%s' is disabled, skipping", name)
                continue

            url = server_cfg.get("url")
            command = server_cfg.get("command")

            if url:
                await self._load_mcp_http(name, url, server_cfg)
            elif command:
                await self._load_mcp_stdio(name, command, server_cfg)
            else:
                logger.warning("MCP server '%s' has no command or url, skipping", name)

    async def _load_mcp_stdio(self, name: str, command: str, server_cfg: dict):
        """Load MCP server via stdio (subprocess) transport."""
        args = server_cfg.get("args", [])
        try:
            env = None
            if server_cfg.get("env"):
                import os
                env = {**os.environ, **server_cfg["env"]}

            process = await asyncio.create_subprocess_exec(
                command, *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Initialize MCP session
            init_resp = await self._mcp_rpc_stdio(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "liteagent", "version": "0.1.0"},
            })

            # Send initialized notification
            notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
            process.stdin.write(notif.encode())
            await process.stdin.drain()

            # Discover tools
            tools_resp = await self._mcp_rpc_stdio(process, "tools/list", {})
            tools = tools_resp.get("result", {}).get("tools", [])

            for tool in tools:
                tool_name = f"{name}__{tool['name']}"
                self._tools[tool_name] = {
                    "name": tool_name,
                    "description": tool.get("description", tool["name"]),
                    "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
                }
                self._mcp_processes[tool_name] = process

                async def _make_mcp_handler(proc, orig_name):
                    async def handler(**kwargs):
                        resp = await self._mcp_rpc_stdio(proc, "tools/call", {
                            "name": orig_name, "arguments": kwargs,
                        })
                        result = resp.get("result", {})
                        content = result.get("content", [])
                        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                        return "\n".join(texts) or json.dumps(result)
                    return handler

                self._handlers[tool_name] = await _make_mcp_handler(process, tool["name"])

            logger.info("Loaded %d tools from MCP server '%s' (stdio)", len(tools), name)

        except Exception as e:
            logger.error("Failed to load MCP server '%s' (stdio): %s", name, e)

    async def _load_mcp_http(self, name: str, url: str, server_cfg: dict):
        """Load MCP server via HTTP (Streamable HTTP) transport."""
        try:
            # Initialize MCP session over HTTP
            init_resp = await self._mcp_rpc_http(url, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "liteagent", "version": "0.1.0"},
            })

            # Send initialized notification (fire-and-forget)
            await self._mcp_rpc_http(url, "notifications/initialized", {},
                                     is_notification=True)

            # Discover tools
            tools_resp = await self._mcp_rpc_http(url, "tools/list", {})
            tools = tools_resp.get("result", {}).get("tools", [])

            for tool in tools:
                tool_name = f"{name}__{tool['name']}"
                self._tools[tool_name] = {
                    "name": tool_name,
                    "description": tool.get("description", tool["name"]),
                    "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
                }
                self._mcp_http_urls[tool_name] = url

                async def _make_http_handler(base_url, orig_name):
                    async def handler(**kwargs):
                        resp = await self._mcp_rpc_http(base_url, "tools/call", {
                            "name": orig_name, "arguments": kwargs,
                        })
                        result = resp.get("result", {})
                        content = result.get("content", [])
                        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                        return "\n".join(texts) or json.dumps(result)
                    return handler

                self._handlers[tool_name] = await _make_http_handler(url, tool["name"])

            logger.info("Loaded %d tools from MCP server '%s' (HTTP: %s)", len(tools), name, url)

        except Exception as e:
            logger.error("Failed to load MCP server '%s' (HTTP %s): %s", name, url, e)

    async def _mcp_rpc_stdio(self, process: asyncio.subprocess.Process,
                              method: str, params: dict) -> dict:
        """Send JSON-RPC request to MCP server via stdio."""
        self._mcp_id_counter += 1
        request = {"jsonrpc": "2.0", "id": self._mcp_id_counter,
                    "method": method, "params": params}
        data = json.dumps(request) + "\n"
        process.stdin.write(data.encode())
        await process.stdin.drain()

        response_line = await asyncio.wait_for(
            process.stdout.readline(), timeout=30
        )
        return json.loads(response_line)

    # Keep old name as alias for backward compatibility
    _mcp_rpc = _mcp_rpc_stdio

    async def _mcp_rpc_http(self, url: str, method: str, params: dict,
                             is_notification: bool = False) -> dict:
        """Send JSON-RPC request to MCP server via HTTP POST."""
        import urllib.request
        import urllib.error

        self._mcp_id_counter += 1
        request = {"jsonrpc": "2.0", "method": method, "params": params}
        if not is_notification:
            request["id"] = self._mcp_id_counter

        data = json.dumps(request).encode("utf-8")

        # Use asyncio to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        def _do_request():
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read().decode("utf-8")
                    if not body.strip():
                        return {}
                    # Handle SSE-style response (event stream)
                    if resp.headers.get("Content-Type", "").startswith("text/event-stream"):
                        return self._parse_sse_response(body)
                    return json.loads(body)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                logger.error("MCP HTTP error %d: %s", e.code, body[:200])
                raise
            except urllib.error.URLError as e:
                logger.error("MCP HTTP connection error: %s", e.reason)
                raise

        if is_notification:
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, _do_request), timeout=10)
            except Exception:
                pass  # Notifications are fire-and-forget
            return {}

        return await asyncio.wait_for(
            loop.run_in_executor(None, _do_request), timeout=60)

    @staticmethod
    def _parse_sse_response(body: str) -> dict:
        """Parse SSE (text/event-stream) response to extract JSON-RPC result."""
        last_data = None
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                last_data = line[5:].strip()
        if last_data:
            try:
                return json.loads(last_data)
            except json.JSONDecodeError:
                pass
        return {}

    async def close_mcp_servers(self):
        """Shutdown all MCP server processes (stdio only; HTTP needs no cleanup)."""
        seen = set()
        for proc in getattr(self, '_mcp_processes', {}).values():
            if id(proc) not in seen:
                seen.add(id(proc))
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()


def register_builtin_tools(registry: ToolRegistry, enabled: list[str] | None = None,
                           sandbox_root: str | None = None,
                           command_allowlist: set[str] | None = None,
                           allow_shell: bool = False,
                           command_timeout: int = 120):
    """Register built-in tools based on config.

    Args:
        sandbox_root: If set, restrict read_file/write_file to this directory.
        command_allowlist: Override default allowed commands for exec_command.
        allow_shell: If True, use shell=True (full access, like Claude Code).
        command_timeout: Default timeout for exec_command in seconds (default 120).
    """
    enabled = enabled or ["read_file", "write_file", "exec_command"]
    _allowlist = command_allowlist or COMMAND_ALLOWLIST

    if "read_file" in enabled:
        @registry.tool(name="read_file",
                       description="Read contents of a file. Returns numbered lines. "
                                   "Use offset/limit for large files.")
        def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
            """path: Absolute or relative file path to read
            offset: Start from this line number (1-based, 0 = beginning)
            limit: Max lines to return (0 = all)"""
            resolved, err = _validate_path(path, sandbox_root)
            if err:
                logger.warning("read_file blocked: %s → %s", path, err)
                return err
            if not os.path.exists(resolved):
                return f"File not found: {path}"
            if os.path.isdir(resolved):
                return f"Error: '{path}' is a directory, not a file"
            try:
                with open(resolved, "r", errors="replace") as f:
                    lines = f.readlines()
                total = len(lines)

                # Apply offset/limit (1-based offset)
                start = max(0, offset - 1) if offset > 0 else 0
                if limit > 0:
                    selected = lines[start:start + limit]
                else:
                    selected = lines[start:]

                # Format with line numbers (cat -n style)
                numbered = []
                for i, line in enumerate(selected, start=start + 1):
                    numbered.append(f"{i:>6}\t{line.rstrip()}")
                result = "\n".join(numbered)

                # Truncation for very large output
                if len(result) > 200000:
                    head = "\n".join(numbered[:len(numbered)//2])[:100000]
                    tail = "\n".join(numbered[len(numbered)//2:])[-100000:]
                    chars_lost = len(result) - 200000
                    result = (f"{head}\n\n... [{chars_lost} chars truncated, "
                              f"use offset/limit] ...\n\n{tail}")

                header = f"File: {path} ({total} lines total)"
                if offset > 0 or limit > 0:
                    end_line = start + len(selected)
                    header += f" — showing lines {start+1}-{end_line}"
                return f"{header}\n{result}"
            except Exception as e:
                return f"Error reading {path}: {e}"

    if "write_file" in enabled:
        @registry.tool(name="write_file", description="Write content to a file. Creates directories if needed. Blocks writes containing secrets (API keys, private keys).")
        def write_file(path: str, content: str) -> str:
            """path: File path to write to
            content: Content to write"""
            resolved, err = _validate_path(path, sandbox_root)
            if err:
                logger.warning("write_file blocked: %s → %s", path, err)
                return err
            # Secret scanning — block writes containing API keys, private keys, etc.
            secret_match = _scan_secrets(content)
            if secret_match:
                logger.warning("write_file blocked: secret detected (%s) in %s", secret_match, path)
                return (f"Blocked: content appears to contain a secret ({secret_match}). "
                        "Never write credentials to files. Use environment variables instead.")
            try:
                os.makedirs(os.path.dirname(resolved) or ".", exist_ok=True)
                with open(resolved, "w") as f:
                    f.write(content)
                return f"Written {len(content)} chars to {path}"
            except Exception as e:
                return f"Error writing {path}: {e}"

    _default_cmd_timeout = command_timeout

    if "exec_command" in enabled:
        @registry.tool(name="exec_command",
                       description="Execute a shell command on the LOCAL machine and return output. "
                                   "This runs directly on the host system — NOT in any isolated or sandboxed environment. "
                                   "Use for: docker, docker-compose, npm, pip, git, make, cargo, python, bash scripts, builds, tests, starting/stopping services. "
                                   "When user asks to run a build, start the app, or execute any command — call this tool immediately. "
                                   "IMPORTANT: if you need the process to keep running while you do more checks, start it in the background "
                                   "(for example with 'nohup ... > server.log 2>&1 &' or an equivalent detached form), then verify it with curl/lsof/logs. "
                                   "Do not start long-running dev servers in the foreground unless the task explicitly ends there.")
        def exec_command(command: str, timeout: int = 0, approved: bool = False) -> str:
            """command: Shell command to execute (supports pipes and shell syntax)
            timeout: Max seconds to wait (0 = use default from config, typically 120s)
            approved: Set to true after user explicitly confirms execution of sensitive commands"""
            try:
                timeout = int(timeout)
            except (TypeError, ValueError):
                timeout = 0
            if isinstance(approved, str):
                approved = approved.strip().lower() in {"1", "true", "yes", "y"}
            _timeout = timeout if timeout > 0 else _default_cmd_timeout

            # Safety layer 1: block dangerous patterns
            cmd_lower = command.lower().strip()
            for pattern in DANGEROUS_COMMAND_PATTERNS:
                if pattern in cmd_lower:
                    logger.warning("Blocked dangerous command: %s (matched: %s)", command, pattern)
                    return f"Blocked: command matches dangerous pattern '{pattern}'"

            # Safety layer 1.5: capability gating (only when NOT in allow_shell mode)
            if not allow_shell:
                try:
                    _parts = shlex.split(command)
                except ValueError:
                    _parts = command.split()
                if _parts:
                    _cmd_base = os.path.basename(_parts[0])
                    if _cmd_base in APPROVAL_REQUIRED_COMMANDS and not approved:
                        logger.info("Command '%s' requires user approval (gated)", _cmd_base)
                        return (
                            f"⚠️ Command '{_cmd_base}' requires explicit user approval.\n"
                            f"Full command: `{command}`\n"
                            "Please ask the user to confirm, then call exec_command "
                            "again with approved=true."
                        )

            if allow_shell:
                # Full shell access (like Claude Code)
                logger.info("Executing (shell): %s", command[:200])
                if _looks_like_background_shell_command(command):
                    try:
                        proc = subprocess.Popen(
                            command,
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            cwd=os.getcwd(),
                            start_new_session=True,
                        )
                        return f"Started background command (pid {proc.pid})"
                    except Exception as e:
                        return f"Error: {e}"
                if _looks_like_foreground_server_command(command):
                    log_path = os.path.join(os.getcwd(), "liteagent-bg.log")
                    try:
                        log_file = open(log_path, "ab")
                        proc = subprocess.Popen(
                            command,
                            shell=True,
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            cwd=os.getcwd(),
                            start_new_session=True,
                        )
                        log_file.close()
                        return (
                            f"Started long-running server command in background "
                            f"(pid {proc.pid}, log: {log_path})"
                        )
                    except Exception as e:
                        return f"Error: {e}"
                try:
                    result = subprocess.run(
                        command, shell=True, capture_output=True, text=True,
                        timeout=_timeout, cwd=os.getcwd()
                    )
                except subprocess.TimeoutExpired:
                    return f"Command timed out after {_timeout}s"
                except Exception as e:
                    return f"Error: {e}"
            else:
                # Secure mode: parse command, check allowlist, no shell
                try:
                    parts = shlex.split(command)
                except ValueError as e:
                    return f"Invalid command syntax: {e}"
                if not parts:
                    return "Empty command"

                # Safety layer 2: allowlist check on base command
                cmd_name = os.path.basename(parts[0])
                if cmd_name not in _allowlist:
                    logger.warning("Command '%s' not in allowlist", cmd_name)
                    return (f"Command '{cmd_name}' not in allowlist. "
                            f"Allowed: {', '.join(sorted(_allowlist)[:20])}...")

                logger.info("Executing: %s", command[:200])
                try:
                    result = subprocess.run(
                        parts, shell=False, capture_output=True, text=True,
                        timeout=_timeout, cwd=os.getcwd()
                    )
                except subprocess.TimeoutExpired:
                    return f"Command timed out after {_timeout}s"
                except FileNotFoundError:
                    return f"Command not found: {parts[0]}"
                except Exception as e:
                    return f"Error: {e}"

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"
            if result.returncode != 0:
                output += f"\n[exit code: {result.returncode}]"
            return output.strip()[:30000] or "(no output)"

    # NOTE: web_search builtin removed — superseded by web.py multi-provider
    # search registered via _wire_web_tools() in agent.py

    if "memory_search" in enabled:
        @registry.tool(name="memory_search", description="Search your long-term memory for facts about the user, past conversations, and learned knowledge.")
        def memory_search(query: str) -> str:
            """query: What to search for in memory"""
            # Will be connected to MemorySystem in agent.py
            return f"[Memory search stub for: {query}]"

    if "download_file" in enabled:
        @registry.tool(name="download_file",
                       description="Download a file from a URL and save it locally. Returns the local file path. Use this when you need to fetch files from the internet.")
        def download_file(url: str, filename: str = "") -> str:
            """url: URL to download from
            filename: Optional filename (auto-detected from URL if empty)"""
            import urllib.request
            import urllib.parse
            from .web import is_ssrf_target

            downloads_dir = os.path.join(os.path.expanduser("~"), ".liteagent", "downloads")
            os.makedirs(downloads_dir, exist_ok=True)

            parsed = urllib.parse.urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                return "Blocked: only http/https URLs are allowed"
            if is_ssrf_target(url):
                return "Blocked: URL blocked (SSRF protection)"

            if not filename:
                filename = os.path.basename(parsed.path) or "download"
            filename = os.path.basename(filename).strip() or "download"

            import uuid as _uuid
            dest = os.path.realpath(os.path.join(downloads_dir, f"{_uuid.uuid4().hex[:8]}_{filename}"))
            downloads_real = os.path.realpath(downloads_dir)
            if not dest.startswith(downloads_real + os.sep):
                return "Blocked: invalid destination path"

            try:
                req = urllib.request.Request(url, headers={"User-Agent": "LiteAgent/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read(50 * 1024 * 1024 + 1)  # 50MB limit
                    if len(data) > 50 * 1024 * 1024:
                        return "Error: file too large (>50MB)"
                    with open(dest, "wb") as f:
                        f.write(data)
                return f"Downloaded to: {dest} ({len(data)} bytes)"
            except Exception as e:
                return f"Download error: {e}"

    if "send_file_to_user" in enabled:
        @registry.tool(name="send_file_to_user",
                       description="Queue a file to be sent to the user alongside your text response. "
                                   "The file will be delivered as an attachment (e.g. in Telegram as a document/photo). "
                                   "The file must exist on the local filesystem. "
                                   "Compatibility fallback: if you only have plain text content, pass it via content "
                                   "and the tool will create a temporary .txt attachment.")
        def send_file_to_user(file_path: str = "", caption: str = "", content: str = "") -> str:
            """file_path: Path to the local file to send
            caption: Optional caption for the file
            content: Optional plain text payload to wrap into a temporary text file"""
            from .file_queue import enqueue_file
            import tempfile

            if content and not str(file_path or "").strip():
                fd, tmp_path = tempfile.mkstemp(prefix="liteagent_message_", suffix=".txt")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        f.write(str(content))
                except Exception:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
                enqueue_file(tmp_path, caption=caption or "message.txt", delete_after_send=True)
                return "Text content wrapped and queued for delivery as message.txt"

            resolved = os.path.realpath(os.path.expanduser(file_path))
            if not os.path.exists(resolved):
                return f"Error: file not found: {file_path}"
            if not os.path.isfile(resolved):
                return f"Error: not a file: {file_path}"

            enqueue_file(resolved, caption=caption)
            return f"File queued for delivery: {os.path.basename(resolved)}"

    # ── edit_file: precise string replacement (like Claude Code Edit) ────
    if "edit_file" in enabled:
        @registry.tool(name="edit_file",
                       description="Patch a file by replacing an exact substring. "
                                   "IMPORTANT: This tool does NOT write full file content — "
                                   "use write_file to create or overwrite a file. "
                                   "edit_file only replaces old_string with new_string inside an existing file. "
                                   "old_string must match exactly (whitespace, indentation). "
                                   "Parameters: file_path, old_string, new_string, replace_all.")
        def edit_file(file_path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> str:
            """file_path: Path to the existing file to patch
            old_string: Exact substring to find (must exist in file, whitespace-sensitive)
            new_string: Replacement text for old_string
            replace_all: Replace ALL occurrences if true (default: first only)"""
            resolved, err = _validate_path(file_path, sandbox_root)
            if err:
                logger.warning("edit_file blocked: %s → %s", file_path, err)
                return err
            if not os.path.exists(resolved):
                return f"File not found: {file_path}"
            if not os.path.isfile(resolved):
                return f"Error: not a file: {file_path}"

            # Secret scanning on new content
            secret_match = _scan_secrets(new_string)
            if secret_match:
                logger.warning("edit_file blocked: secret detected (%s) in %s",
                               secret_match, file_path)
                return (f"Blocked: new_string contains a secret ({secret_match}). "
                        "Never write credentials to files.")

            try:
                with open(resolved, "r", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                return f"Error reading {file_path}: {e}"

            # Count occurrences
            count = content.count(old_string)
            if count == 0:
                # Show a snippet of actual file content to help agent self-correct
                preview = content[:400].replace("\n", "↵")
                return (f"Error: old_string not found in {file_path}. "
                        "The file content may have changed since you last read it. "
                        f"Use read_file('{file_path}') to get current content, then retry edit_file with exact matching text. "
                        f"File starts with: {preview!r}")
            if count > 1 and not replace_all:
                return (f"Error: old_string found {count} times in {file_path}. "
                        "Provide more context to make it unique, or set replace_all=true.")

            # Perform replacement
            if replace_all:
                new_content = content.replace(old_string, new_string)
                replacements = count
            else:
                new_content = content.replace(old_string, new_string, 1)
                replacements = 1

            try:
                with open(resolved, "w") as f:
                    f.write(new_content)
            except Exception as e:
                return f"Error writing {file_path}: {e}"

            # Show diff preview (±3 lines around first replacement)
            new_lines = new_content.splitlines()
            # Find first occurrence of new_string in the result
            preview_lines = []
            for i, line in enumerate(new_lines):
                if new_string.splitlines()[0] if new_string else "" in line:
                    start = max(0, i - 3)
                    end = min(len(new_lines), i + len(new_string.splitlines()) + 3)
                    for j in range(start, end):
                        preview_lines.append(f"{j+1:>6}\t{new_lines[j]}")
                    break

            result = f"Replaced {replacements} occurrence(s) in {file_path}"
            if preview_lines:
                result += "\n\nContext:\n" + "\n".join(preview_lines)
            return result

    # ── glob_files: file pattern matching (like Claude Code Glob) ────
    if "glob_files" in enabled:
        @registry.tool(name="glob_files",
                       description="Find files matching a glob pattern. "
                                   "Supports ** for recursive matching. "
                                   "Returns file paths sorted by modification time.")
        def glob_files(pattern: str, path: str = ".") -> str:
            """pattern: Glob pattern (e.g. '**/*.py', 'src/**/*.ts', '*.json')
            path: Root directory to search in (default: current dir)"""
            import pathlib
            root = pathlib.Path(os.path.expanduser(path)).resolve()
            if not root.exists():
                return f"Directory not found: {path}"
            if not root.is_dir():
                return f"Not a directory: {path}"

            try:
                matches = list(root.glob(pattern))
            except Exception as e:
                return f"Glob error: {e}"

            # Filter out sensitive paths
            filtered = []
            for m in matches:
                skip = False
                for sensitive in SENSITIVE_PATH_COMPONENTS:
                    if sensitive in str(m):
                        skip = True
                        break
                if not skip and m.is_file():
                    filtered.append(m)

            # Sort by mtime (newest first)
            filtered.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            # Limit results
            total = len(filtered)
            max_results = 500
            if total > max_results:
                filtered = filtered[:max_results]

            # Format output
            lines = []
            for p in filtered:
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    rel = p
                lines.append(str(rel))

            header = f"Found {total} files matching '{pattern}' in {path}"
            if total > max_results:
                header += f" (showing first {max_results})"
            return header + "\n" + "\n".join(lines)

    # ── grep_search: content search with regex (like Claude Code Grep) ────
    if "grep_search" in enabled:
        @registry.tool(name="grep_search",
                       description="Search file contents using regex patterns. "
                                   "Tries ripgrep (rg) first, falls back to grep, then Python. "
                                   "Returns matching lines with file paths and line numbers.")
        def grep_search(pattern: str, path: str = ".", file_glob: str = "",
                        context: int = 0, include_line_numbers: bool = True) -> str:
            """pattern: Regex pattern to search for
            path: Directory or file to search in (default: current dir)
            file_glob: File glob filter (e.g. '*.py', '*.js')
            context: Lines of context around matches (default: 0)
            include_line_numbers: Show line numbers (default: true)"""
            import shutil

            search_path = os.path.realpath(os.path.expanduser(path))
            if not os.path.exists(search_path):
                return f"Path not found: {path}"

            # Try ripgrep first (fastest)
            rg_path = shutil.which("rg")
            if rg_path:
                cmd = [rg_path, "--no-heading", "--with-filename"]
                if include_line_numbers:
                    cmd.append("--line-number")
                if context > 0:
                    cmd.extend(["-C", str(context)])
                if file_glob:
                    cmd.extend(["--glob", file_glob])
                cmd.extend(["--max-count", "200"])  # safety limit
                cmd.append(pattern)
                cmd.append(search_path)

                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30)
                    output = result.stdout.strip()
                    if not output and result.returncode != 0:
                        if result.returncode == 1:
                            return f"No matches found for pattern '{pattern}' in {path}"
                        return f"ripgrep error: {result.stderr.strip()}"
                    return output or f"No matches found for pattern '{pattern}' in {path}"
                except subprocess.TimeoutExpired:
                    return "Search timed out after 30s"
                except Exception as e:
                    logger.debug("ripgrep failed, trying grep: %s", e)

            # Fallback to grep
            grep_path = shutil.which("grep")
            if grep_path:
                cmd = [grep_path, "-r", "-n" if include_line_numbers else ""]
                cmd = [c for c in cmd if c]  # remove empty
                if context > 0:
                    cmd.extend(["-C", str(context)])
                if file_glob:
                    cmd.extend(["--include", file_glob])
                cmd.append(pattern)
                cmd.append(search_path)

                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=30)
                    output = result.stdout.strip()
                    if not output:
                        return f"No matches found for pattern '{pattern}' in {path}"
                    # Limit output
                    lines = output.split("\n")
                    if len(lines) > 500:
                        output = "\n".join(lines[:500]) + f"\n\n... [{len(lines)-500} more matches]"
                    return output
                except subprocess.TimeoutExpired:
                    return "Search timed out after 30s"
                except Exception as e:
                    logger.debug("grep failed, trying Python: %s", e)

            # Pure Python fallback
            try:
                import pathlib
                regex = re.compile(pattern)
                results = []
                root = pathlib.Path(search_path)
                glob_pat = file_glob if file_glob else "**/*"
                files = root.glob(glob_pat) if root.is_dir() else [root]

                for fp in files:
                    if not fp.is_file():
                        continue
                    # Skip sensitive and binary files
                    skip = False
                    for sensitive in SENSITIVE_PATH_COMPONENTS:
                        if sensitive in str(fp):
                            skip = True
                            break
                    if skip:
                        continue
                    try:
                        text = fp.read_text(errors="replace")
                        for i, line in enumerate(text.splitlines(), 1):
                            if regex.search(line):
                                prefix = f"{fp}:{i}:" if include_line_numbers else f"{fp}:"
                                results.append(f"{prefix}{line}")
                                if len(results) >= 500:
                                    results.append("... [limit reached]")
                                    return "\n".join(results)
                    except Exception:
                        continue
                if not results:
                    return f"No matches found for pattern '{pattern}' in {path}"
                return "\n".join(results)
            except re.error as e:
                return f"Invalid regex pattern: {e}"
            except Exception as e:
                return f"Search error: {e}"
