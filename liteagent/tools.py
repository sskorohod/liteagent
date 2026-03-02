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

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}

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

        query_emb = embedder.encode(query)
        scored = []
        for tool in all_tools:
            name = tool['name']
            if name not in self._tool_emb_cache:
                desc = f"{name}: {tool.get('description', '')}"
                self._tool_emb_cache[name] = embedder.encode(desc)
            tool_emb = self._tool_emb_cache[name]
            # Cosine similarity
            dot = float(query_emb @ tool_emb)
            norm_q = float(math.sqrt(query_emb @ query_emb))
            norm_t = float(math.sqrt(tool_emb @ tool_emb))
            sim = dot / (norm_q * norm_t) if norm_q and norm_t else 0.0
            scored.append((tool, sim))

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

    async def execute(self, content_blocks) -> list[dict]:
        """Execute tool calls from LLM response."""
        results = []
        for block in content_blocks:
            if not hasattr(block, 'type') or block.type != "tool_use":
                continue

            handler = self._handlers.get(block.name)
            if not handler:
                result = f"Error: unknown tool '{block.name}'"
            else:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(**block.input)
                    else:
                        result = handler(**block.input)
                    if not isinstance(result, str):
                        result = json.dumps(result, ensure_ascii=False, default=str)
                except Exception as e:
                    result = f"Error executing {block.name}: {e}"

            # Wrap in XML markers for anti-injection (LLM knows this is tool output, not instructions)
            raw = str(result)[:10000]
            content = f"<tool_output name=\"{block.name}\">\n{raw}\n</tool_output>"
            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": content,
            })
        return results

    async def execute_one(self, block) -> dict:
        """Execute a single tool call and return result with metadata."""
        import time
        handler = self._handlers.get(block.name)
        start = time.time()
        error = False
        if not handler:
            result = f"Error: unknown tool '{block.name}'"
            error = True
        else:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**block.input)
                else:
                    result = handler(**block.input)
                if not isinstance(result, str):
                    result = json.dumps(result, ensure_ascii=False, default=str)
            except Exception as e:
                result = f"Error executing {block.name}: {e}"
                error = True

        duration_ms = int((time.time() - start) * 1000)
        raw = str(result)[:10000]
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
                           allow_shell: bool = False):
    """Register built-in tools based on config.

    Args:
        sandbox_root: If set, restrict read_file/write_file to this directory.
        command_allowlist: Override default allowed commands for exec_command.
        allow_shell: If True, use shell=True (less secure, backward compat).
    """
    enabled = enabled or ["read_file", "write_file", "exec_command"]
    _allowlist = command_allowlist or COMMAND_ALLOWLIST

    if "read_file" in enabled:
        @registry.tool(name="read_file", description="Read contents of a file from the filesystem.")
        def read_file(path: str) -> str:
            """path: Absolute or relative file path to read"""
            resolved, err = _validate_path(path, sandbox_root)
            if err:
                logger.warning("read_file blocked: %s → %s", path, err)
                return err
            if not os.path.exists(resolved):
                return f"File not found: {path}"
            try:
                with open(resolved, "r") as f:
                    content = f.read()
                if len(content) > 50000:
                    return content[:25000] + f"\n\n... [{len(content)-50000} chars truncated] ...\n\n" + content[-25000:]
                return content
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

    if "exec_command" in enabled:
        @registry.tool(name="exec_command", description="Execute a shell command and return output. Use for system tasks, running scripts, git, etc. Some commands (curl, docker, ssh, etc.) require explicit user approval — pass approved=true after confirmation.")
        def exec_command(command: str, timeout: int = 30, approved: bool = False) -> str:
            """command: Shell command to execute
            timeout: Max seconds to wait (default 30)
            approved: Set to true after user explicitly confirms execution of sensitive commands"""
            # Safety layer 1: block dangerous patterns
            cmd_lower = command.lower().strip()
            for pattern in DANGEROUS_COMMAND_PATTERNS:
                if pattern in cmd_lower:
                    logger.warning("Blocked dangerous command: %s (matched: %s)", command, pattern)
                    return f"Blocked: command matches dangerous pattern '{pattern}'"

            # Safety layer 1.5: capability gating — require user approval for sensitive commands
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
                # Legacy mode: shell=True (less secure)
                logger.info("Executing (shell): %s", command[:200])
                try:
                    result = subprocess.run(
                        command, shell=True, capture_output=True, text=True,
                        timeout=timeout, cwd=os.getcwd()
                    )
                except subprocess.TimeoutExpired:
                    return f"Command timed out after {timeout}s"
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
                        timeout=timeout, cwd=os.getcwd()
                    )
                except subprocess.TimeoutExpired:
                    return f"Command timed out after {timeout}s"
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
            return output.strip()[:10000] or "(no output)"

    if "web_search" in enabled:
        @registry.tool(name="web_search", description="Search the web for current information, facts, and research. Returns top results with titles and descriptions.")
        def web_search(query: str) -> str:
            """query: Search query string"""
            import urllib.request
            import urllib.parse

            api_key = os.environ.get("BRAVE_SEARCH_API_KEY")
            if not api_key:
                return ("Web search not configured. "
                        "Set BRAVE_SEARCH_API_KEY environment variable to enable.")

            try:
                url = (f"https://api.search.brave.com/res/v1/web/search"
                       f"?q={urllib.parse.quote(query)}&count=5")
                req = urllib.request.Request(url, headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())

                results = []
                for r in data.get("web", {}).get("results", [])[:5]:
                    title = r.get("title", "")
                    desc = r.get("description", "")
                    link = r.get("url", "")
                    results.append(f"**{title}**\n{desc}\nURL: {link}")

                return "\n\n".join(results) or "No results found."
            except Exception as e:
                logger.warning("Web search failed: %s", e)
                return f"Web search error: {e}"

    if "memory_search" in enabled:
        @registry.tool(name="memory_search", description="Search your long-term memory for facts about the user, past conversations, and learned knowledge.")
        def memory_search(query: str) -> str:
            """query: What to search for in memory"""
            # Will be connected to MemorySystem in agent.py
            return f"[Memory search stub for: {query}]"
