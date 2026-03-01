"""Tool registry — decorator-based local tools + MCP support."""

import asyncio
import inspect
import json
import logging
import os
import subprocess
from typing import Any, Callable, get_type_hints

logger = logging.getLogger(__name__)

# Dangerous shell patterns to block
DANGEROUS_COMMAND_PATTERNS = [
    "rm -rf /", "rm -fr /", "mkfs", "dd if=", "> /dev/sd",
    ":(){ :", "chmod -R 777 /", "mv /* ", "wget -O- | sh",
    "curl -s | sh", "echo '' > /etc/", "format c:",
]


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

        query_emb = embedder.encode(query)
        scored = []
        for tool in all_tools:
            desc = f"{tool['name']}: {tool.get('description', '')}"
            tool_emb = embedder.encode(desc)
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

            results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(result)[:10000],  # Cap output
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
        content = str(result)[:10000]
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
        """Load MCP servers and discover their tools via JSON-RPC stdio."""
        self._mcp_processes: dict[str, asyncio.subprocess.Process] = {}
        self._mcp_id_counter = 0

        for name, server_cfg in mcp_config.items():
            command = server_cfg.get("command")
            args = server_cfg.get("args", [])
            if not command:
                logger.warning("MCP server '%s' has no command, skipping", name)
                continue

            try:
                process = await asyncio.create_subprocess_exec(
                    command, *args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                # Initialize MCP session
                init_resp = await self._mcp_rpc(process, "initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "liteagent", "version": "0.1.0"},
                })

                # Send initialized notification
                notif = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
                process.stdin.write(notif.encode())
                await process.stdin.drain()

                # Discover tools
                tools_resp = await self._mcp_rpc(process, "tools/list", {})
                tools = tools_resp.get("result", {}).get("tools", [])

                for tool in tools:
                    tool_name = f"{name}__{tool['name']}"
                    self._tools[tool_name] = {
                        "name": tool_name,
                        "description": tool.get("description", tool["name"]),
                        "input_schema": tool.get("inputSchema", {"type": "object", "properties": {}}),
                    }
                    self._mcp_processes[tool_name] = process

                    # Create handler closure for this tool
                    async def _make_mcp_handler(proc, orig_name):
                        async def handler(**kwargs):
                            resp = await self._mcp_rpc(proc, "tools/call", {
                                "name": orig_name, "arguments": kwargs,
                            })
                            result = resp.get("result", {})
                            content = result.get("content", [])
                            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
                            return "\n".join(texts) or json.dumps(result)
                        return handler

                    self._handlers[tool_name] = await _make_mcp_handler(process, tool["name"])

                logger.info("Loaded %d tools from MCP server '%s'", len(tools), name)

            except Exception as e:
                logger.error("Failed to load MCP server '%s': %s", name, e)

    async def _mcp_rpc(self, process: asyncio.subprocess.Process,
                       method: str, params: dict) -> dict:
        """Send JSON-RPC request to MCP server and read response."""
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

    async def close_mcp_servers(self):
        """Shutdown all MCP server processes."""
        seen = set()
        for proc in getattr(self, '_mcp_processes', {}).values():
            if id(proc) not in seen:
                seen.add(id(proc))
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()


def register_builtin_tools(registry: ToolRegistry, enabled: list[str] | None = None):
    """Register built-in tools based on config."""
    enabled = enabled or ["read_file", "write_file", "exec_command"]

    if "read_file" in enabled:
        @registry.tool(name="read_file", description="Read contents of a file from the filesystem.")
        def read_file(path: str) -> str:
            """path: Absolute or relative file path to read"""
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                return f"File not found: {path}"
            try:
                with open(path, "r") as f:
                    content = f.read()
                if len(content) > 50000:
                    return content[:25000] + f"\n\n... [{len(content)-50000} chars truncated] ...\n\n" + content[-25000:]
                return content
            except Exception as e:
                return f"Error reading {path}: {e}"

    if "write_file" in enabled:
        @registry.tool(name="write_file", description="Write content to a file. Creates directories if needed.")
        def write_file(path: str, content: str) -> str:
            """path: File path to write to
            content: Content to write"""
            path = os.path.expanduser(path)
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                return f"Written {len(content)} chars to {path}"
            except Exception as e:
                return f"Error writing {path}: {e}"

    if "exec_command" in enabled:
        @registry.tool(name="exec_command", description="Execute a shell command and return output. Use for system tasks, running scripts, git, etc.")
        def exec_command(command: str, timeout: int = 30) -> str:
            """command: Shell command to execute
            timeout: Max seconds to wait (default 30)"""
            # Safety: block dangerous patterns
            cmd_lower = command.lower().strip()
            for pattern in DANGEROUS_COMMAND_PATTERNS:
                if pattern in cmd_lower:
                    logger.warning("Blocked dangerous command: %s (matched: %s)", command, pattern)
                    return f"Blocked: command matches dangerous pattern '{pattern}'"

            logger.info("Executing command: %s", command[:200])
            try:
                result = subprocess.run(
                    command, shell=True, capture_output=True, text=True,
                    timeout=timeout, cwd=os.getcwd()
                )
                output = ""
                if result.stdout:
                    output += result.stdout
                if result.stderr:
                    output += f"\n[stderr]: {result.stderr}"
                if result.returncode != 0:
                    output += f"\n[exit code: {result.returncode}]"
                return output.strip()[:10000] or "(no output)"
            except subprocess.TimeoutExpired:
                return f"Command timed out after {timeout}s"
            except Exception as e:
                return f"Error: {e}"

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
