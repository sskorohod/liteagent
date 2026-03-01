"""Tests for LLM provider abstraction layer."""
import pytest
from liteagent.providers import (
    TokenUsage, TextBlock, ToolUseBlock, LLMResponse,
    MODEL_PRICING, get_pricing, create_provider,
    AnthropicProvider, OpenAIProvider, OllamaProvider,
)


class TestDataStructures:
    def test_token_usage_defaults(self):
        u = TokenUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_read_input_tokens == 0

    def test_text_block(self):
        b = TextBlock(text="hello")
        assert b.type == "text"
        assert b.text == "hello"

    def test_tool_use_block(self):
        b = ToolUseBlock(id="t1", name="read_file", input={"path": "/tmp"})
        assert b.type == "tool_use"
        assert b.name == "read_file"
        assert b.input == {"path": "/tmp"}

    def test_llm_response(self):
        r = LLMResponse(
            content=[TextBlock(text="hi")],
            stop_reason="end_turn",
            usage=TokenUsage(input_tokens=10, output_tokens=5))
        assert r.stop_reason == "end_turn"
        assert r.content[0].text == "hi"
        assert r.usage.input_tokens == 10


class TestPricing:
    def test_known_model(self):
        p = get_pricing("claude-sonnet-4-20250514")
        assert p["input"] == 3.0
        assert p["output"] == 15.0

    def test_haiku_pricing(self):
        p = get_pricing("claude-haiku-4-5-20251001")
        assert p["input"] == 0.8

    def test_openai_pricing(self):
        p = get_pricing("gpt-4o")
        assert p["input"] == 2.5

    def test_ollama_wildcard(self):
        p = get_pricing("ollama/llama3")
        assert p["input"] == 0.0
        assert p["output"] == 0.0

    def test_unknown_model_fallback(self):
        p = get_pricing("unknown-model-xyz")
        assert "input" in p
        assert "output" in p


class TestFactory:
    def test_default_creates_anthropic(self):
        # create_provider should create AnthropicProvider by default
        # This will fail without ANTHROPIC_API_KEY, but tests the factory path
        provider = create_provider({})
        assert isinstance(provider, AnthropicProvider)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider({"agent": {"provider": "nonexistent"}})


class TestOpenAIConversions:
    def test_convert_messages_system(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = OpenAIProvider._convert_messages(msgs, system="You are helpful")
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "hello"}

    def test_convert_messages_system_list(self):
        """System as list of cache_control blocks (Anthropic format)."""
        system = [
            {"type": "text", "text": "Part 1", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Part 2"},
        ]
        result = OpenAIProvider._convert_messages(
            [{"role": "user", "content": "hi"}], system=system)
        assert result[0]["role"] == "system"
        assert "Part 1" in result[0]["content"]
        assert "Part 2" in result[0]["content"]

    def test_convert_tools(self):
        anthropic_tools = [{
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }]
        result = OpenAIProvider._convert_tools(anthropic_tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_file"
        assert "path" in result[0]["function"]["parameters"]["properties"]

    def test_convert_messages_tool_result(self):
        """tool_result messages should become role=tool."""
        msgs = [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": "file contents here"},
            ]},
        ]
        result = OpenAIProvider._convert_messages(msgs)
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "t1"
        assert result[0]["content"] == "file contents here"

    def test_convert_messages_plain_user(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = OpenAIProvider._convert_messages(msgs)
        assert result[0] == {"role": "user", "content": "hi"}
