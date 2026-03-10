"""LLM provider abstraction layer — Anthropic, OpenAI, Ollama, Gemini."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# UNIFIED DATA STRUCTURES
# ══════════════════════════════════════════

@dataclass
class TokenUsage:
    """Unified token usage across all providers."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class TextBlock:
    """A text content block."""
    type: str = "text"
    text: str = ""


@dataclass
class ToolUseBlock:
    """A tool_use content block."""
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""
    content: list  # list of TextBlock / ToolUseBlock
    stop_reason: str  # "end_turn", "tool_use", etc.
    usage: TokenUsage = field(default_factory=TokenUsage)


# ══════════════════════════════════════════
# MODEL PRICING (per 1M tokens)
# ══════════════════════════════════════════

MODEL_PRICING: dict[str, dict[str, float]] = {
    # Anthropic
    "claude-haiku-4-5-20251001":  {"input": 0.80, "output": 4.00, "cache_read": 0.08},
    "claude-sonnet-4-20250514":   {"input": 3.00, "output": 15.00, "cache_read": 0.30},
    "claude-opus-4-20250115":     {"input": 15.00, "output": 75.00, "cache_read": 1.50},
    # OpenAI
    "gpt-4o":                     {"input": 2.50, "output": 10.00, "cache_read": 0.0},
    "gpt-4o-mini":                {"input": 0.15, "output": 0.60, "cache_read": 0.0},
    "gpt-4.1":                    {"input": 2.00, "output": 8.00, "cache_read": 0.0},
    "gpt-4.1-mini":               {"input": 0.40, "output": 1.60, "cache_read": 0.0},
    "gpt-4.1-nano":               {"input": 0.10, "output": 0.40, "cache_read": 0.0},
    "o1":                         {"input": 15.00, "output": 60.00, "cache_read": 0.0},
    "o3-mini":                    {"input": 1.10, "output": 4.40, "cache_read": 0.0},
    # Gemini
    "gemini-2.0-flash":           {"input": 0.10, "output": 0.40, "cache_read": 0.0},
    "gemini-2.5-pro":             {"input": 1.25, "output": 10.00, "cache_read": 0.0},
    "gemini-2.5-flash":           {"input": 0.15, "output": 0.60, "cache_read": 0.0},
    # Qwen / DashScope (Alibaba Cloud)
    "qwen-turbo":                 {"input": 0.30, "output": 0.60, "cache_read": 0.0},
    "qwen-plus":                  {"input": 0.80, "output": 2.00, "cache_read": 0.0},
    "qwen-max":                   {"input": 2.00, "output": 6.00, "cache_read": 0.0},
    "qwen-vl-plus":               {"input": 0.80, "output": 2.00, "cache_read": 0.0},
    "qwen-vl-max":                {"input": 2.00, "output": 6.00, "cache_read": 0.0},
    "qwen-long":                  {"input": 0.50, "output": 2.00, "cache_read": 0.0},
    # Grok / xAI
    "grok-3":                     {"input": 3.00, "output": 15.00, "cache_read": 0.0},
    "grok-3-mini":                {"input": 0.30, "output": 0.50, "cache_read": 0.0},
    "grok-4-0709":                {"input": 3.00, "output": 15.00, "cache_read": 0.0},
    "grok-3-fast":                {"input": 5.00, "output": 25.00, "cache_read": 0.0},
    "grok-3-mini-fast":           {"input": 0.60, "output": 4.00, "cache_read": 0.0},
    # Ollama (local, free)
    "ollama/*":                   {"input": 0.0, "output": 0.0, "cache_read": 0.0},
}


def get_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model, with fallback."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    # Ollama models: "ollama/" prefix, or present in discovered list,
    # or local model format like "qwen3:32b", "llama3.1:latest"
    if (model.startswith("ollama/")
            or model in PROVIDER_MODELS.get("ollama", [])
            or ":" in model):
        return MODEL_PRICING["ollama/*"]
    return {"input": 3.0, "output": 15.0, "cache_read": 0.3}  # safe fallback


# ══════════════════════════════════════════
# ANTHROPIC PROVIDER
# ══════════════════════════════════════════

class AnthropicProvider:
    """Anthropic Claude provider (default).

    Supports auth profile key rotation (from OpenClaw): if multiple API keys
    are registered via config.providers.anthropic.api_keys, rotates through
    them automatically on failures.
    """

    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self._last_stream_response: LLMResponse | None = None
        self._current_key: str | None = None

    def supports_model(self, model: str) -> bool:
        return model.startswith("claude-") or model.startswith("anthropic/")

    async def _get_client(self):
        """Return client, optionally rotating key via auth profiles."""
        import anthropic
        try:
            from .auth_profiles import get_next_api_key
            key = await get_next_api_key("anthropic")
            if key and key != self._current_key:
                self._current_key = key
                self.client = anthropic.AsyncAnthropic(api_key=key)
        except Exception:
            pass
        return self.client

    async def complete(self, model: str, max_tokens: int, messages: list,
                       system=None, tools=None, temperature=None) -> LLMResponse:
        client = await self._get_client()
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            response = await client.messages.create(**kwargs)
            if self._current_key:
                from .auth_profiles import report_key_success
                await report_key_success("anthropic", self._current_key)
            return self._to_response(response)
        except Exception as exc:
            if self._current_key:
                from .auth_profiles import report_key_failure
                reason = "rate_limit" if "rate" in str(exc).lower() else \
                         "auth_error" if "auth" in str(exc).lower() else "unknown"
                await report_key_failure("anthropic", self._current_key, reason)
            raise

    async def stream(self, model: str, max_tokens: int, messages: list,
                     system=None, tools=None, temperature=None) -> AsyncGenerator:
        client = await self._get_client()
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if hasattr(event, 'type') and event.type == "content_block_delta":
                    if hasattr(event.delta, 'text'):
                        yield event.delta.text
            final = await stream.get_final_message()
            self._last_stream_response = self._to_response(final)

    @staticmethod
    def _to_response(raw) -> LLMResponse:
        blocks = []
        for block in raw.content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(ToolUseBlock(
                    id=block.id, name=block.name, input=block.input))
        usage = TokenUsage(
            input_tokens=getattr(raw.usage, 'input_tokens', 0),
            output_tokens=getattr(raw.usage, 'output_tokens', 0),
            cache_read_input_tokens=getattr(raw.usage, 'cache_read_input_tokens', 0),
            cache_creation_input_tokens=getattr(raw.usage, 'cache_creation_input_tokens', 0),
        )
        return LLMResponse(content=blocks, stop_reason=raw.stop_reason, usage=usage)


# ══════════════════════════════════════════
# OPENAI PROVIDER
# ══════════════════════════════════════════

class OpenAIProvider:
    """OpenAI GPT provider."""

    def __init__(self, base_url: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY"):
        import openai
        self._base_url = base_url
        self._api_key_env = api_key_env
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        key = os.environ.get(api_key_env)
        if key:
            kwargs["api_key"] = key
        self.client = openai.AsyncOpenAI(**kwargs)
        self._current_key: str | None = key
        self._last_stream_response: LLMResponse | None = None

    def supports_model(self, model: str) -> bool:
        return (model.startswith("gpt-") or model.startswith("o1")
                or model.startswith("o3") or model.startswith("qwen")
                or model.startswith("grok"))

    async def _get_client(self):
        """Return client, optionally rotating key via auth profiles."""
        import openai
        try:
            from .auth_profiles import get_next_api_key
            key = await get_next_api_key("openai")
            if key and key != self._current_key:
                self._current_key = key
                kwargs = {"api_key": key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self.client = openai.AsyncOpenAI(**kwargs)
        except Exception:
            pass
        return self.client

    async def complete(self, model: str, max_tokens: int, messages: list,
                       system=None, tools=None, temperature=None) -> LLMResponse:
        client = await self._get_client()
        oai_messages = self._convert_messages(messages, system)
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": oai_messages}
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature
        try:
            response = await client.chat.completions.create(**kwargs)
            if self._current_key:
                from .auth_profiles import report_key_success
                await report_key_success("openai", self._current_key)
            return self._to_response(response)
        except Exception as exc:
            if self._current_key:
                from .auth_profiles import report_key_failure
                reason = "rate_limit" if "rate" in str(exc).lower() else \
                         "auth_error" if "auth" in str(exc).lower() else "unknown"
                await report_key_failure("openai", self._current_key, reason)
            raise

    async def stream(self, model: str, max_tokens: int, messages: list,
                     system=None, tools=None, temperature=None) -> AsyncGenerator:
        client = await self._get_client()
        oai_messages = self._convert_messages(messages, system)
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": oai_messages,
                  "stream": True}
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = "auto"
        if temperature is not None:
            kwargs["temperature"] = temperature

        full_text = ""
        tool_calls_data = {}
        finish_reason = "stop"
        import json as _json

        response = await client.chat.completions.create(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            if delta.content:
                full_text += delta.content
                yield delta.content
            # Accumulate tool calls from stream
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_data[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc.function.arguments

        # Build final response
        blocks = []
        if full_text:
            blocks.append(TextBlock(text=full_text))
        for tc_data in tool_calls_data.values():
            try:
                args = _json.loads(tc_data["arguments"])
            except Exception:
                args = {}
            blocks.append(ToolUseBlock(
                id=tc_data["id"], name=tc_data["name"], input=args))

        stop = "tool_use" if finish_reason == "tool_calls" else "end_turn"
        self._last_stream_response = LLMResponse(
            content=blocks, stop_reason=stop, usage=TokenUsage())

    @staticmethod
    def _convert_messages(messages: list, system=None) -> list:
        """Convert Anthropic-format messages to OpenAI format."""
        import json as _json
        oai = []

        # System message
        if system is not None:
            if isinstance(system, list):
                # Flatten cache_control blocks into text
                text = " ".join(
                    b.get("text", "") for b in system if isinstance(b, dict))
            else:
                text = system
            if text.strip():
                oai.append({"role": "system", "content": text})

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "assistant" and isinstance(content, list):
                # May contain text + tool_use blocks
                text_parts = []
                tool_calls = []
                for block in content:
                    if hasattr(block, 'type'):
                        if block.type == "text":
                            text_parts.append(block.text)
                        elif block.type == "tool_use":
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": _json.dumps(block.input),
                                },
                            })
                    elif isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            tool_calls.append({
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": _json.dumps(block.get("input", {})),
                                },
                            })
                m = {"role": "assistant"}
                if text_parts:
                    m["content"] = "\n".join(text_parts)
                if tool_calls:
                    m["tool_calls"] = tool_calls
                oai.append(m)

            elif role == "user" and isinstance(content, list):
                # May contain text, image, document, or tool_result blocks.
                # Build a single multimodal user message with parts array.
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        # Flush accumulated parts as user message first
                        if parts:
                            oai.append({"role": "user", "content": parts})
                            parts = []
                        oai.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append({"type": "text", "text": block.get("text", "")})
                    elif isinstance(block, dict) and block.get("type") == "image":
                        # Convert Anthropic image block → OpenAI vision format
                        source = block.get("source", {})
                        media = source.get("media_type", "image/jpeg")
                        data = source.get("data", "")
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{media};base64,{data}"},
                        })
                    elif isinstance(block, dict) and block.get("type") == "document":
                        # OpenAI doesn't support PDF blocks natively; note it
                        parts.append({
                            "type": "text",
                            "text": "[PDF document attached — content passed as document block]",
                        })
                    else:
                        parts.append({"type": "text", "text": str(block)})
                if parts:
                    oai.append({"role": "user", "content": parts})
            else:
                oai.append({"role": role, "content": content if isinstance(content, str)
                            else str(content)})

        return oai

    @staticmethod
    def _convert_tools(tools: list) -> list:
        """Convert Anthropic tool format to OpenAI function format."""
        oai_tools = []
        for tool in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {
                        "type": "object", "properties": {}}),
                },
            })
        return oai_tools

    @staticmethod
    def _to_response(raw) -> LLMResponse:
        """Convert OpenAI response to LLMResponse."""
        import json as _json
        choice = raw.choices[0] if raw.choices else None
        blocks = []
        stop = "end_turn"

        if choice:
            if choice.message.content:
                blocks.append(TextBlock(text=choice.message.content))
            if choice.message.tool_calls:
                stop = "tool_use"
                for tc in choice.message.tool_calls:
                    try:
                        args = _json.loads(tc.function.arguments)
                    except Exception:
                        args = {}
                    blocks.append(ToolUseBlock(
                        id=tc.id, name=tc.function.name, input=args))
            if choice.finish_reason == "tool_calls":
                stop = "tool_use"

        usage = TokenUsage()
        if raw.usage:
            usage.input_tokens = raw.usage.prompt_tokens or 0
            usage.output_tokens = raw.usage.completion_tokens or 0

        return LLMResponse(content=blocks, stop_reason=stop, usage=usage)


# ══════════════════════════════════════════
# OLLAMA PROVIDER (OpenAI-compatible)
# ══════════════════════════════════════════

class OllamaProvider(OpenAIProvider):
    """Ollama local models via OpenAI-compatible API.

    Overrides stream() to handle reasoning/thinking models (qwen3, etc.)
    where tokens arrive in a 'reasoning' field that the OpenAI SDK ignores.
    """

    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        import openai
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key="ollama")
        self._base_url = base_url
        self._current_key: str | None = None
        self._last_stream_response: LLMResponse | None = None

    def supports_model(self, model: str) -> bool:
        # Ollama accepts any model name — local models don't have a known prefix
        return True

    async def stream(self, model: str, max_tokens: int, messages: list,
                     system=None, tools=None, temperature=None) -> AsyncGenerator:
        """Stream with reasoning/thinking model support (qwen3, etc.).

        Uses httpx directly to access 'reasoning' field that openai SDK drops.
        """
        import httpx

        oai_messages = self._convert_messages(messages, system)
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": oai_messages,
            "stream": True,
        }
        if tools:
            body["tools"] = self._convert_tools(tools)
        if temperature is not None:
            body["temperature"] = temperature

        url = self._base_url.rstrip("/") + "/chat/completions"

        full_text = ""
        reasoning_text = ""
        tool_calls_data: dict = {}
        finish_reason = "stop"
        reasoning_started = False

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0, connect=10.0)
        ) as client:
            async with client.stream(
                "POST", url, json=body,
                headers={"Authorization": "Bearer ollama"},
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except (json.JSONDecodeError, ValueError):
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    # Reasoning tokens (thinking models like qwen3)
                    reasoning = (
                        delta.get("reasoning", "")
                        or delta.get("reasoning_content", "")
                    )
                    if reasoning:
                        if not reasoning_started:
                            reasoning_started = True
                            yield "<think>"
                        reasoning_text += reasoning
                        yield reasoning

                    # Content tokens
                    content = delta.get("content", "")
                    if content:
                        if reasoning_started and not full_text:
                            yield "</think>\n"
                        full_text += content
                        yield content

                    # Tool calls
                    for tc in (delta.get("tool_calls") or []):
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                "id": "", "name": "", "arguments": "",
                            }
                        if tc.get("id"):
                            tool_calls_data[idx]["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_calls_data[idx]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_calls_data[idx]["arguments"] += fn["arguments"]

        # Build final response
        blocks: list = []
        if full_text:
            blocks.append(TextBlock(text=full_text))
        elif reasoning_text:
            # Model used all tokens for thinking — expose reasoning as content
            blocks.append(TextBlock(text=reasoning_text))

        for tc_data in tool_calls_data.values():
            try:
                args = json.loads(tc_data["arguments"])
            except Exception:
                args = {}
            blocks.append(ToolUseBlock(
                id=tc_data["id"], name=tc_data["name"], input=args))

        stop = "tool_use" if (
            finish_reason in ("tool_calls", "stop") and tool_calls_data
        ) else "end_turn"
        self._last_stream_response = LLMResponse(
            content=blocks, stop_reason=stop, usage=TokenUsage())


# ══════════════════════════════════════════
# GEMINI PROVIDER
# ══════════════════════════════════════════

class GeminiProvider:
    """Google Gemini provider."""

    def __init__(self, api_key_env: str = "GOOGLE_API_KEY"):
        import google.generativeai as genai
        self._api_key_env = api_key_env
        api_key = os.environ.get(api_key_env, "")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._current_key: str | None = api_key or None
        self._last_stream_response: LLMResponse | None = None

    def supports_model(self, model: str) -> bool:
        return model.startswith("gemini-") or model.startswith("models/")

    async def _rotate_key(self) -> None:
        """Rotate to next key via auth profiles if available."""
        try:
            from .auth_profiles import get_next_api_key
            key = await get_next_api_key("gemini")
            if key and key != self._current_key:
                self._current_key = key
                self._genai.configure(api_key=key)
        except Exception:
            pass

    async def complete(self, model: str, max_tokens: int, messages: list,
                       system=None, tools=None, temperature=None) -> LLMResponse:
        await self._rotate_key()
        try:
            result = await self._do_complete(model, max_tokens, messages, system, temperature, tools)
            if self._current_key:
                from .auth_profiles import report_key_success
                await report_key_success("gemini", self._current_key)
            return result
        except Exception as exc:
            if self._current_key:
                from .auth_profiles import report_key_failure
                reason = "rate_limit" if "rate" in str(exc).lower() or "quota" in str(exc).lower() else \
                         "auth_error" if "auth" in str(exc).lower() or "api_key" in str(exc).lower() else "unknown"
                await report_key_failure("gemini", self._current_key, reason)
            raise

    async def _do_complete(self, model: str, max_tokens: int, messages: list,
                           system=None, temperature=None, tools=None) -> LLMResponse:
        gen_model = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=self._flatten_system(system) or None)
        contents = self._convert_messages(messages)

        gen_config = {"max_output_tokens": max_tokens}
        if temperature is not None:
            gen_config["temperature"] = temperature

        kwargs: dict = {"generation_config": gen_config}
        tools_param = self._convert_tools_to_gemini(tools)
        if tools_param:
            kwargs["tools"] = tools_param
            kwargs["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

        response = await asyncio.to_thread(
            gen_model.generate_content, contents, **kwargs)

        usage = TokenUsage()
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage.input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            usage.output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        # Parse response: may contain text parts and/or function_call parts
        blocks: list = []
        has_tool_use = False
        try:
            candidates = response.candidates or []
        except Exception:
            candidates = []

        for candidate in candidates:
            try:
                parts = candidate.content.parts or []
            except Exception:
                parts = []
            for part in parts:
                if hasattr(part, 'function_call') and getattr(part.function_call, 'name', None):
                    fc = part.function_call
                    blocks.append(ToolUseBlock(
                        id=f"gemini_{fc.name}_{id(fc)}",
                        name=fc.name,
                        input=dict(fc.args) if fc.args else {},
                    ))
                    has_tool_use = True
                elif hasattr(part, 'text') and part.text:
                    blocks.append(TextBlock(text=part.text))

        if not blocks:
            # Fallback: try response.text shortcut
            try:
                text = response.text or ""
            except Exception:
                text = ""
            blocks = [TextBlock(text=text)]

        return LLMResponse(
            content=blocks,
            stop_reason="tool_use" if has_tool_use else "end_turn",
            usage=usage)

    async def stream(self, model: str, max_tokens: int, messages: list,
                     system=None, tools=None, temperature=None) -> AsyncGenerator:
        await self._rotate_key()
        gen_model = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=self._flatten_system(system) or None)
        contents = self._convert_messages(messages)

        gen_config = {"max_output_tokens": max_tokens}
        if temperature is not None:
            gen_config["temperature"] = temperature

        kwargs: dict = {"generation_config": gen_config, "stream": True}
        tools_param = self._convert_tools_to_gemini(tools)
        if tools_param:
            kwargs["tools"] = tools_param
            kwargs["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}

        response = await asyncio.to_thread(
            gen_model.generate_content, contents, **kwargs)

        full_text = ""
        tool_blocks: list[ToolUseBlock] = []

        for chunk in response:
            # Try to get parts from candidates first (needed when tools are active)
            got_parts = False
            try:
                for candidate in (chunk.candidates or []):
                    for part in (candidate.content.parts or []):
                        got_parts = True
                        if hasattr(part, 'function_call') and getattr(part.function_call, 'name', None):
                            fc = part.function_call
                            tool_blocks.append(ToolUseBlock(
                                id=f"gemini_{fc.name}_{id(fc)}",
                                name=fc.name,
                                input=dict(fc.args) if fc.args else {},
                            ))
                        elif hasattr(part, 'text') and part.text:
                            full_text += part.text
                            yield part.text
            except Exception:
                pass

            if not got_parts:
                # Fallback: try chunk.text
                try:
                    t = chunk.text
                    if t:
                        full_text += t
                        yield t
                except Exception:
                    pass

        content_blocks: list = []
        if full_text:
            content_blocks.append(TextBlock(text=full_text))
        content_blocks.extend(tool_blocks)

        self._last_stream_response = LLMResponse(
            content=content_blocks,
            stop_reason="tool_use" if tool_blocks else "end_turn",
            usage=TokenUsage())

    @staticmethod
    def _flatten_system(system) -> str:
        if system is None:
            return ""
        if isinstance(system, list):
            return " ".join(b.get("text", "") for b in system if isinstance(b, dict))
        return str(system)

    @staticmethod
    def _convert_schema(schema: dict) -> dict:
        """Convert JSON Schema to Gemini Schema format (uppercase types)."""
        if not schema:
            return {"type": "OBJECT"}
        _type_map = {
            "string": "STRING", "number": "NUMBER", "integer": "INTEGER",
            "boolean": "BOOLEAN", "array": "ARRAY", "object": "OBJECT",
        }
        result: dict = {}
        jtype = schema.get("type", "object")
        result["type"] = _type_map.get(str(jtype).lower(), "STRING")
        if "description" in schema:
            result["description"] = schema["description"]
        if "enum" in schema:
            result["enum"] = schema["enum"]
        if "properties" in schema:
            result["properties"] = {
                k: GeminiProvider._convert_schema(v)
                for k, v in schema["properties"].items()
            }
        if "required" in schema:
            result["required"] = schema["required"]
        if "items" in schema:
            result["items"] = GeminiProvider._convert_schema(schema["items"])
        return result

    def _convert_tools_to_gemini(self, tools) -> list | None:
        """Convert Anthropic-format tool defs to Gemini function_declarations."""
        if not tools:
            return None
        function_declarations = []
        for tool in tools:
            fd: dict = {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
            }
            schema = tool.get("input_schema", {})
            if schema:
                fd["parameters"] = self._convert_schema(schema)
            function_declarations.append(fd)
        return [{"function_declarations": function_declarations}]

    @staticmethod
    def _convert_messages(messages: list) -> list:
        """Convert to Gemini format (text + images + tool_use + tool_result)."""
        # Build tool_use_id → tool_name map needed for function_response
        tool_id_to_name: dict[str, str] = {}
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_id_to_name[block.get("id", "")] = block.get("name", "tool")
                    elif hasattr(block, 'type') and getattr(block, 'type', None) == "tool_use":
                        tool_id_to_name[block.id] = block.name

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]

            if isinstance(content, str):
                contents.append({"role": role, "parts": [content]})
                continue

            if isinstance(content, list):
                parts: list = []
                fn_response_parts: list = []

                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            parts.append(block.get("text", ""))
                        elif btype == "tool_use":
                            parts.append({
                                "function_call": {
                                    "name": block.get("name", ""),
                                    "args": block.get("input", {}),
                                }
                            })
                        elif btype == "tool_result":
                            tool_id = block.get("tool_use_id", "")
                            fn_name = tool_id_to_name.get(tool_id, tool_id or "tool")
                            raw = block.get("content", "")
                            response_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                            fn_response_parts.append({
                                "function_response": {
                                    "name": fn_name,
                                    "response": {"result": response_text},
                                }
                            })
                        elif btype == "image":
                            source = block.get("source", {})
                            parts.append({
                                "inline_data": {
                                    "mime_type": source.get("media_type", "image/jpeg"),
                                    "data": source.get("data", ""),
                                }
                            })
                        elif btype == "document":
                            source = block.get("source", {})
                            parts.append({
                                "inline_data": {
                                    "mime_type": source.get("media_type", "application/pdf"),
                                    "data": source.get("data", ""),
                                }
                            })
                    elif hasattr(block, 'type'):
                        btype = getattr(block, 'type', None)
                        if btype == "text":
                            parts.append(block.text)
                        elif btype == "tool_use":
                            parts.append({
                                "function_call": {
                                    "name": block.name,
                                    "args": block.input,
                                }
                            })

                if fn_response_parts:
                    contents.append({"role": "user", "parts": fn_response_parts})
                elif parts:
                    contents.append({"role": role, "parts": parts})

        return contents


# ══════════════════════════════════════════
# PROVIDER MODELS CATALOG
# ══════════════════════════════════════════

PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-20250514", "claude-haiku-4-5-20251001",
        "claude-opus-4-20250115",
    ],
    "openai": [
        "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
        "gpt-4.1-nano", "o1", "o3-mini",
    ],
    "gemini": [
        "gemini-2.0-flash", "gemini-2.5-pro", "gemini-2.5-flash",
    ],
    "qwen": [
        "qwen-turbo", "qwen-plus", "qwen-max",
        "qwen-vl-plus", "qwen-vl-max", "qwen-long",
    ],
    "grok": [
        "grok-3", "grok-3-mini", "grok-4-0709",
        "grok-3-fast", "grok-3-mini-fast",
    ],
    "ollama": [],  # Populated dynamically by discover_ollama_models()
}


def discover_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Query local Ollama instance for available models. Returns model names."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        logger.info("Discovered %d Ollama models: %s", len(models), models)
        return models
    except Exception as e:
        logger.debug("Ollama not available: %s", e)
        return []


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running locally."""
    import urllib.request
    try:
        urllib.request.urlopen(f"{base_url}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def refresh_ollama_models(base_url: str = "http://localhost:11434"):
    """Refresh the PROVIDER_MODELS['ollama'] list from local instance."""
    models = discover_ollama_models(base_url)
    PROVIDER_MODELS["ollama"] = models
    return models


# ══════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════

def create_test_provider(provider_name: str, api_key: str, base_url: str | None = None):
    """Create a temporary provider instance for testing connectivity.

    Sets the API key in env temporarily, creates the provider, then restores env.
    """
    from .config import PROVIDER_ENV_VARS
    env_var = PROVIDER_ENV_VARS.get(provider_name)

    old_val = None
    if env_var:
        old_val = os.environ.get(env_var)
        os.environ[env_var] = api_key

    try:
        if provider_name == "anthropic":
            return AnthropicProvider()
        elif provider_name == "openai":
            return OpenAIProvider(base_url=base_url)
        elif provider_name == "qwen":
            return OpenAIProvider(
                base_url=base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                api_key_env="DASHSCOPE_API_KEY")
        elif provider_name == "grok":
            return OpenAIProvider(
                base_url=base_url or "https://api.x.ai/v1",
                api_key_env="XAI_API_KEY")
        elif provider_name == "ollama":
            return OllamaProvider(base_url=base_url or "http://localhost:11434/v1")
        elif provider_name == "gemini":
            return GeminiProvider()
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    finally:
        # Restore original env
        if env_var:
            if old_val is not None:
                os.environ[env_var] = old_val
            elif env_var in os.environ:
                del os.environ[env_var]


def create_provider(config: dict):
    """Create LLM provider from config.

    If the configured provider's API key is missing, falls back to
    the first provider that has a key available.

    Auth profiles (from OpenClaw): if config.providers.<name>.api_keys is a list,
    registers all keys in the AuthProfileManager for round-robin rotation.
    """
    from .config import get_api_key, PROVIDER_ENV_VARS

    # Register any multi-key profiles from config (OpenClaw auth profiles)
    try:
        from .auth_profiles import register_keys_from_config
        register_keys_from_config(config)
    except Exception:
        pass

    agent_cfg = config.get("agent", {})
    provider_name = agent_cfg.get("provider", "anthropic")
    providers_cfg = config.get("providers", {})

    # Load key from keys.json into env (always update to pick up latest saved key)
    env_var = PROVIDER_ENV_VARS.get(provider_name)
    if env_var:
        key = get_api_key(provider_name)
        if key:
            os.environ[env_var] = key

    # Check if the configured provider has a key (ollama doesn't need one)
    if provider_name != "ollama" and env_var and not get_api_key(provider_name):
        # Try to fallback to a provider that has a key
        fallback = _find_fallback_provider(provider_name, config)
        if fallback:
            fb_name, fb_model = fallback
            logger.warning(
                "No API key for '%s'. Falling back to '%s' (model: %s). "
                "Set the key via: liteagent --config or OPENAI_API_KEY env var.",
                provider_name, fb_name, fb_model)
            provider_name = fb_name
            agent_cfg["provider"] = fb_name
            agent_cfg["default_model"] = fb_model
            # Load fallback key into env
            fb_env = PROVIDER_ENV_VARS.get(fb_name)
            if fb_env:
                fb_key = get_api_key(fb_name)
                if fb_key:
                    os.environ[fb_env] = fb_key
        else:
            raise ValueError(
                f"No API key found for provider '{provider_name}'. "
                f"Set it via: liteagent --config, or set the "
                f"{env_var} environment variable.")

    if provider_name == "anthropic":
        return AnthropicProvider()

    elif provider_name == "openai":
        pcfg = providers_cfg.get("openai", {})
        return OpenAIProvider(
            base_url=pcfg.get("base_url"),
            api_key_env=pcfg.get("api_key_env", "OPENAI_API_KEY"))

    elif provider_name == "ollama":
        pcfg = providers_cfg.get("ollama", {})
        base = pcfg.get("base_url", "http://localhost:11434/v1")
        # Auto-discover available models
        api_base = base.replace("/v1", "") if base.endswith("/v1") else base
        refresh_ollama_models(api_base)
        return OllamaProvider(base_url=base)

    elif provider_name == "qwen":
        pcfg = providers_cfg.get("qwen", {})
        return OpenAIProvider(
            base_url=pcfg.get("base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
            api_key_env=pcfg.get("api_key_env", "DASHSCOPE_API_KEY"))

    elif provider_name == "grok":
        pcfg = providers_cfg.get("grok", {})
        return OpenAIProvider(
            base_url=pcfg.get("base_url", "https://api.x.ai/v1"),
            api_key_env=pcfg.get("api_key_env", "XAI_API_KEY"))

    elif provider_name == "gemini":
        pcfg = providers_cfg.get("gemini", {})
        return GeminiProvider(
            api_key_env=pcfg.get("api_key_env", "GOOGLE_API_KEY"))

    else:
        raise ValueError(f"Unknown provider: {provider_name}. "
                         f"Supported: anthropic, openai, grok, qwen, ollama, gemini")


def _find_fallback_provider(current: str, config: dict):
    """Find a provider with an available API key (for auto-fallback)."""
    from .config import get_api_key

    # Preferred fallback order
    fallback_order = [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o-mini"),
        ("grok", "grok-3-mini"),
        ("qwen", "qwen-plus"),
        ("gemini", "gemini-2.0-flash"),
    ]
    for name, default_model in fallback_order:
        if name == current:
            continue
        key = get_api_key(name)
        if key:
            return (name, default_model)
    return None
