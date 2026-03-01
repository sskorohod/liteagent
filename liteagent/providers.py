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
    # Ollama (local, free)
    "ollama/*":                   {"input": 0.0, "output": 0.0, "cache_read": 0.0},
}


def get_pricing(model: str) -> dict[str, float]:
    """Get pricing for a model, with fallback."""
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    if model.startswith("ollama/"):
        return MODEL_PRICING["ollama/*"]
    return {"input": 3.0, "output": 15.0, "cache_read": 0.3}  # safe fallback


# ══════════════════════════════════════════
# ANTHROPIC PROVIDER
# ══════════════════════════════════════════

class AnthropicProvider:
    """Anthropic Claude provider (default)."""

    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self._last_stream_response: LLMResponse | None = None

    async def complete(self, model: str, max_tokens: int, messages: list,
                       system=None, tools=None, temperature=None) -> LLMResponse:
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await self.client.messages.create(**kwargs)
        return self._to_response(response)

    async def stream(self, model: str, max_tokens: int, messages: list,
                     system=None, tools=None, temperature=None) -> AsyncGenerator:
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        async with self.client.messages.stream(**kwargs) as stream:
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
        kwargs = {}
        if base_url:
            kwargs["base_url"] = base_url
        key = os.environ.get(api_key_env)
        if key:
            kwargs["api_key"] = key
        self.client = openai.AsyncOpenAI(**kwargs)
        self._last_stream_response: LLMResponse | None = None

    async def complete(self, model: str, max_tokens: int, messages: list,
                       system=None, tools=None, temperature=None) -> LLMResponse:
        oai_messages = self._convert_messages(messages, system)
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": oai_messages}
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await self.client.chat.completions.create(**kwargs)
        return self._to_response(response)

    async def stream(self, model: str, max_tokens: int, messages: list,
                     system=None, tools=None, temperature=None) -> AsyncGenerator:
        oai_messages = self._convert_messages(messages, system)
        kwargs = {"model": model, "max_tokens": max_tokens, "messages": oai_messages,
                  "stream": True}
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        if temperature is not None:
            kwargs["temperature"] = temperature

        full_text = ""
        tool_calls_data = {}
        finish_reason = "stop"
        import json as _json

        response = await self.client.chat.completions.create(**kwargs)
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
                # May contain tool_result blocks
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        oai.append({
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": block.get("content", ""),
                        })
                    elif isinstance(block, dict) and block.get("type") == "text":
                        oai.append({"role": "user", "content": block.get("text", "")})
                    else:
                        oai.append({"role": "user", "content": str(block)})
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
    """Ollama local models via OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:11434/v1"):
        import openai
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key="ollama")
        self._last_stream_response: LLMResponse | None = None


# ══════════════════════════════════════════
# GEMINI PROVIDER
# ══════════════════════════════════════════

class GeminiProvider:
    """Google Gemini provider."""

    def __init__(self, api_key_env: str = "GOOGLE_API_KEY"):
        import google.generativeai as genai
        api_key = os.environ.get(api_key_env, "")
        genai.configure(api_key=api_key)
        self._genai = genai
        self._last_stream_response: LLMResponse | None = None

    async def complete(self, model: str, max_tokens: int, messages: list,
                       system=None, tools=None, temperature=None) -> LLMResponse:
        gen_model = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=self._flatten_system(system) or None)
        contents = self._convert_messages(messages)

        gen_config = {"max_output_tokens": max_tokens}
        if temperature is not None:
            gen_config["temperature"] = temperature

        response = await asyncio.to_thread(
            gen_model.generate_content, contents,
            generation_config=gen_config)

        text = response.text or ""
        usage = TokenUsage()
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage.input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            usage.output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

        return LLMResponse(
            content=[TextBlock(text=text)],
            stop_reason="end_turn",
            usage=usage)

    async def stream(self, model: str, max_tokens: int, messages: list,
                     system=None, tools=None, temperature=None) -> AsyncGenerator:
        gen_model = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=self._flatten_system(system) or None)
        contents = self._convert_messages(messages)

        gen_config = {"max_output_tokens": max_tokens}
        if temperature is not None:
            gen_config["temperature"] = temperature

        response = await asyncio.to_thread(
            gen_model.generate_content, contents,
            generation_config=gen_config, stream=True)

        full_text = ""
        for chunk in response:
            if chunk.text:
                full_text += chunk.text
                yield chunk.text

        self._last_stream_response = LLMResponse(
            content=[TextBlock(text=full_text)],
            stop_reason="end_turn",
            usage=TokenUsage())

    @staticmethod
    def _flatten_system(system) -> str:
        if system is None:
            return ""
        if isinstance(system, list):
            return " ".join(b.get("text", "") for b in system if isinstance(b, dict))
        return str(system)

    @staticmethod
    def _convert_messages(messages: list) -> list:
        """Convert to Gemini format."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if isinstance(content, str):
                contents.append({"role": role, "parts": [content]})
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif hasattr(block, 'type') and block.type == "text":
                        text_parts.append(block.text)
                if text_parts:
                    contents.append({"role": role, "parts": text_parts})
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
    """Create LLM provider from config."""
    from .config import get_api_key, PROVIDER_ENV_VARS

    agent_cfg = config.get("agent", {})
    provider_name = agent_cfg.get("provider", "anthropic")
    providers_cfg = config.get("providers", {})

    # Load key from keys.json into env (always update to pick up latest saved key)
    env_var = PROVIDER_ENV_VARS.get(provider_name)
    if env_var:
        key = get_api_key(provider_name)
        if key:
            os.environ[env_var] = key

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

    elif provider_name == "gemini":
        pcfg = providers_cfg.get("gemini", {})
        return GeminiProvider(
            api_key_env=pcfg.get("api_key_env", "GOOGLE_API_KEY"))

    else:
        raise ValueError(f"Unknown provider: {provider_name}. "
                         f"Supported: anthropic, openai, ollama, gemini")
