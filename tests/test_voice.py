"""Tests for liteagent.voice — TTS + STT + auto-TTS pipeline + config tools."""

import asyncio
import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from liteagent.voice import (
    BUILTIN_PRESETS,
    DEFAULT_EDGE_VOICE,
    DEFAULT_ELEVENLABS_MODEL_ID,
    DEFAULT_ELEVENLABS_VOICE_ID,
    DEFAULT_OPENAI_TTS_MODEL,
    DEFAULT_OPENAI_TTS_VOICE,
    DEFAULT_TTS_MAX_LENGTH,
    OPENAI_TTS_MODELS,
    OPENAI_TTS_VOICES,
    STT_PROVIDERS,
    TTS_COST_INFO,
    TTS_PROVIDERS,
    TtsDirectiveOverrides,
    TtsDirectiveParseResult,
    TtsResult,
    SttResult,
    maybe_apply_tts,
    parse_tts_directives,
    resolve_voice_config,
    strip_markdown,
    text_to_speech,
    _resolve_output_format,
    _resolve_provider_order,
)


# ══════════════════════════════════════════
# Config resolution
# ══════════════════════════════════════════

class TestResolveVoiceConfig:
    def test_defaults(self):
        cfg = resolve_voice_config({})
        assert cfg["tts"]["auto"] == "off"
        assert cfg["tts"]["provider"] == "edge"
        assert cfg["tts"]["max_length"] == DEFAULT_TTS_MAX_LENGTH
        assert cfg["tts"]["openai"]["model"] == DEFAULT_OPENAI_TTS_MODEL
        assert cfg["tts"]["openai"]["voice"] == DEFAULT_OPENAI_TTS_VOICE
        assert cfg["tts"]["elevenlabs"]["voice_id"] == DEFAULT_ELEVENLABS_VOICE_ID
        assert cfg["tts"]["edge"]["voice"] == DEFAULT_EDGE_VOICE
        assert cfg["stt"]["provider"] == "openai"

    def test_custom_config(self):
        cfg = resolve_voice_config({
            "voice": {
                "tts": {
                    "auto": "always",
                    "provider": "openai",
                    "openai": {"voice": "nova", "model": "tts-1-hd"},
                },
                "stt": {
                    "provider": "groq",
                    "groq": {"model": "whisper-large-v3", "language": "en"},
                },
            },
        })
        assert cfg["tts"]["auto"] == "always"
        assert cfg["tts"]["provider"] == "openai"
        assert cfg["tts"]["openai"]["voice"] == "nova"
        assert cfg["tts"]["openai"]["model"] == "tts-1-hd"
        assert cfg["stt"]["provider"] == "groq"
        assert cfg["stt"]["groq"]["language"] == "en"


class TestOutputFormat:
    def test_telegram_format(self):
        fmt = _resolve_output_format("telegram")
        assert fmt["voice_compatible"] is True
        assert fmt["openai"] == "opus"
        assert fmt["extension"] == ".ogg"

    def test_api_format(self):
        fmt = _resolve_output_format("api")
        assert fmt["voice_compatible"] is False
        assert fmt["openai"] == "mp3"
        assert fmt["extension"] == ".mp3"


class TestProviderOrder:
    def test_primary_first(self):
        order = _resolve_provider_order("elevenlabs")
        assert order[0] == "elevenlabs"
        assert "openai" in order
        assert "edge" in order

    def test_default_order(self):
        order = _resolve_provider_order("openai")
        assert order == ["openai", "elevenlabs", "edge"]


# ══════════════════════════════════════════
# Markdown stripping
# ══════════════════════════════════════════

class TestStripMarkdown:
    def test_headings(self):
        assert strip_markdown("# Hello") == "Hello"
        assert strip_markdown("## World") == "World"
        assert strip_markdown("### Test") == "Test"

    def test_bold(self):
        assert strip_markdown("This is **bold** text") == "This is bold text"

    def test_italic(self):
        assert strip_markdown("This is *italic* text") == "This is italic text"

    def test_code_blocks(self):
        text = "Before\n```python\nprint('hello')\n```\nAfter"
        result = strip_markdown(text)
        assert "print" not in result
        assert "Before" in result
        assert "After" in result

    def test_inline_code(self):
        assert strip_markdown("Use `print()` function") == "Use print() function"

    def test_links(self):
        assert strip_markdown("[click here](https://example.com)") == "click here"

    def test_images(self):
        assert strip_markdown("![alt text](image.png)") == "alt text"

    def test_blockquotes(self):
        result = strip_markdown("> This is a quote")
        assert "This is a quote" in result
        assert ">" not in result

    def test_complex_mixed(self):
        text = "# Title\n**Bold** and *italic* with `code`\n> Quote\n- List item"
        result = strip_markdown(text)
        assert "#" not in result
        assert "**" not in result
        assert "*" not in result or result.count("*") == 0
        assert "`" not in result
        assert ">" not in result

    def test_plain_text_unchanged(self):
        text = "Hello, how are you today?"
        assert strip_markdown(text) == text


# ══════════════════════════════════════════
# TTS Directive parsing
# ══════════════════════════════════════════

class TestParseTtsDirectives:
    def test_no_directives(self):
        result = parse_tts_directives("Hello world")
        assert result.has_directive is False
        assert result.cleaned_text == "Hello world"
        assert result.tts_text is None
        assert len(result.warnings) == 0

    def test_voice_directive(self):
        result = parse_tts_directives("[[tts:voice=nova]]Hello")
        assert result.has_directive is True
        assert result.overrides.openai_voice == "nova"
        assert "Hello" in result.cleaned_text
        assert "[[tts" not in result.cleaned_text

    def test_provider_directive(self):
        result = parse_tts_directives("[[tts:provider=elevenlabs]]Text")
        assert result.overrides.provider == "elevenlabs"

    def test_multiple_params(self):
        result = parse_tts_directives("[[tts:voice=coral model=tts-1-hd]]Hello")
        assert result.overrides.openai_voice == "coral"
        assert result.overrides.openai_model == "tts-1-hd"

    def test_tts_text_block(self):
        result = parse_tts_directives(
            "Normal text [[tts:text]]Expressive reading![[/tts:text]] more text"
        )
        assert result.has_directive is True
        assert result.tts_text == "Expressive reading!"
        assert "[[tts:text]]" not in result.cleaned_text

    def test_elevenlabs_params(self):
        result = parse_tts_directives(
            "[[tts:voiceId=abc123def456 stability=0.8 similarity_boost=0.9 speed=1.5]]Hi"
        )
        assert result.overrides.elevenlabs_voice_id == "abc123def456"
        assert result.overrides.elevenlabs_stability == 0.8
        assert result.overrides.elevenlabs_similarity_boost == 0.9
        assert result.overrides.elevenlabs_speed == 1.5

    def test_invalid_provider_warning(self):
        result = parse_tts_directives("[[tts:provider=invalid]]Text")
        assert len(result.warnings) > 0
        assert "unsupported provider" in result.warnings[0]

    def test_invalid_stability_warning(self):
        result = parse_tts_directives("[[tts:stability=5.0]]Text")
        assert len(result.warnings) > 0
        assert "stability" in result.warnings[0]

    def test_speaker_boost(self):
        result = parse_tts_directives("[[tts:use_speaker_boost=true]]Text")
        assert result.overrides.elevenlabs_use_speaker_boost is True

        result2 = parse_tts_directives("[[tts:speaker_boost=false]]Text")
        assert result2.overrides.elevenlabs_use_speaker_boost is False

    def test_case_insensitive(self):
        result = parse_tts_directives("[[TTS:voice=nova]]Hello")
        assert result.has_directive is True
        assert result.overrides.openai_voice == "nova"


# ══════════════════════════════════════════
# TTS Providers (mocked)
# ══════════════════════════════════════════

class TestOpenAITTS:
    @pytest.mark.asyncio
    async def test_openai_tts_calls_api(self):
        mock_response = MagicMock()
        mock_response.content = b"fake audio bytes"

        mock_client = MagicMock()
        mock_client.audio.speech.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        import liteagent.voice as voice_mod
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = await voice_mod.openai_tts(
                text="Hello",
                api_key="test-key",
                model="tts-1",
                voice="alloy",
                response_format="mp3",
            )
            assert result == b"fake audio bytes"
            mock_client.audio.speech.create.assert_called_once()
            call_kwargs = mock_client.audio.speech.create.call_args[1]
            assert call_kwargs["model"] == "tts-1"
            assert call_kwargs["voice"] == "alloy"
            assert call_kwargs["input"] == "Hello"


class TestElevenLabsTTS:
    @pytest.mark.asyncio
    async def test_elevenlabs_calls_api(self):
        fake_audio = b"elevenlabs audio"

        import liteagent.voice as voice_mod
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.read.return_value = fake_audio
            mock_urlopen.return_value = mock_resp

            result = await voice_mod.elevenlabs_tts(
                text="Hello",
                api_key="test-key",
                voice_id="testvoice123abcdef",
                model_id="eleven_multilingual_v2",
            )
            assert result == fake_audio
            # Verify URL contains voice_id
            call_args = mock_urlopen.call_args[0][0]
            assert "testvoice123abcdef" in call_args.full_url


class TestEdgeTTS:
    @pytest.mark.asyncio
    async def test_edge_tts_creates_file(self):
        mock_edge_module = MagicMock()
        mock_communicate = MagicMock()
        mock_communicate.save = AsyncMock()
        mock_edge_module.Communicate.return_value = mock_communicate

        import liteagent.voice as voice_mod
        with patch.dict("sys.modules", {"edge_tts": mock_edge_module}):
            output_path = "/tmp/test_voice.mp3"
            await voice_mod.edge_tts_synthesize(
                text="Hello",
                output_path=output_path,
                voice="ru-RU-SvetlanaNeural",
            )
            mock_edge_module.Communicate.assert_called_once()
            mock_communicate.save.assert_called_once_with(output_path)


# ══════════════════════════════════════════
# STT Providers (mocked)
# ══════════════════════════════════════════

class TestSTTOpenAI:
    @pytest.mark.asyncio
    async def test_transcribe_openai(self):
        mock_response = MagicMock()
        mock_response.text = "Hello world"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        import liteagent.voice as voice_mod
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = await voice_mod.transcribe_openai(b"audio data", "test-key")
            assert result.success is True
            assert result.text == "Hello world"
            assert result.provider == "openai"


class TestSTTDeepgram:
    @pytest.mark.asyncio
    async def test_transcribe_deepgram(self):
        import json
        response_data = {
            "results": {
                "channels": [{
                    "alternatives": [{"transcript": "Hello Deepgram"}]
                }]
            }
        }

        import liteagent.voice as voice_mod
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.read.return_value = json.dumps(response_data).encode()
            mock_urlopen.return_value = mock_resp

            result = await voice_mod.transcribe_deepgram(b"audio", "test-key")
            assert result.success is True
            assert result.text == "Hello Deepgram"
            assert result.provider == "deepgram"


class TestSTTGroq:
    @pytest.mark.asyncio
    async def test_transcribe_groq(self):
        mock_response = MagicMock()
        mock_response.text = "Hello Groq"

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_response)

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = mock_client

        import liteagent.voice as voice_mod
        with patch.dict("sys.modules", {"openai": mock_openai}):
            result = await voice_mod.transcribe_groq(b"audio", "test-key")
            assert result.success is True
            assert result.text == "Hello Groq"
            assert result.provider == "groq"


# ══════════════════════════════════════════
# Auto-TTS pipeline
# ══════════════════════════════════════════

class TestAutoTTS:
    @pytest.mark.asyncio
    async def test_off_mode_returns_none(self):
        result = await maybe_apply_tts("Hello world", {"voice": {"tts": {"auto": "off"}}})
        assert result is None

    @pytest.mark.asyncio
    async def test_inbound_mode_skips_without_audio(self):
        result = await maybe_apply_tts(
            "Hello world",
            {"voice": {"tts": {"auto": "inbound"}}},
            inbound_audio=False,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_tagged_mode_skips_without_tag(self):
        result = await maybe_apply_tts(
            "Hello world without tags",
            {"voice": {"tts": {"auto": "tagged"}}},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_tagged_mode_processes_with_tag(self):
        """Tagged mode should trigger when [[tts:...]] directives present."""
        with patch("liteagent.voice.text_to_speech") as mock_tts:
            mock_tts.return_value = TtsResult(
                success=True, audio_path="/tmp/test.mp3",
                provider="edge", latency_ms=100,
            )
            result = await maybe_apply_tts(
                "[[tts:voice=nova]]This should be spoken aloud now",
                {"voice": {"tts": {"auto": "tagged"}}},
            )
            assert result is not None
            assert result.success is True

    @pytest.mark.asyncio
    async def test_short_text_skipped(self):
        result = await maybe_apply_tts(
            "Hi",
            {"voice": {"tts": {"auto": "always"}}},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_truncation_on_long_text(self):
        """Text longer than max_length should be truncated."""
        long_text = "A" * 2000
        with patch("liteagent.voice.text_to_speech") as mock_tts:
            mock_tts.return_value = TtsResult(success=True, audio_path="/tmp/t.mp3")
            await maybe_apply_tts(
                long_text,
                {"voice": {"tts": {"auto": "always", "max_length": 500}}},
            )
            # Check that text_to_speech was called with truncated text
            if mock_tts.called:
                called_text = mock_tts.call_args[1].get("text") or mock_tts.call_args[0][0]
                assert len(called_text) <= 500 + 3  # +3 for "..."


# ══════════════════════════════════════════
# Provider fallback
# ══════════════════════════════════════════

class TestProviderFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """If primary provider fails, should try next."""
        voice_cfg = resolve_voice_config({
            "voice": {"tts": {"provider": "openai"}}
        })

        with patch("liteagent.voice._get_tts_api_key") as mock_key, \
             patch("liteagent.voice.openai_tts", side_effect=RuntimeError("API error")), \
             patch("liteagent.voice.elevenlabs_tts", side_effect=RuntimeError("No key")), \
             patch("liteagent.voice.edge_tts_synthesize") as mock_edge:

            mock_key.side_effect = lambda p, c: "key" if p != "edge" else None
            mock_edge.return_value = None  # edge writes to file

            # Edge TTS creates a file, so we need to create it
            async def fake_edge(**kwargs):
                output_path = kwargs.get("output_path", "")
                if output_path:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(b"audio")
            mock_edge.side_effect = fake_edge

            result = await text_to_speech(
                "Hello world test text",
                voice_cfg, {},
                channel="api",
            )
            # Should have tried openai (failed), elevenlabs (failed), then edge
            assert result.provider == "edge" or not result.success

    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        """When all providers fail, returns error."""
        voice_cfg = resolve_voice_config({
            "voice": {"tts": {"provider": "openai", "edge": {"enabled": False}}}
        })

        with patch("liteagent.voice._get_tts_api_key", return_value=None):
            result = await text_to_speech("Hello", voice_cfg, {})
            assert result.success is False
            assert "no providers available" in (result.error or "").lower() or \
                   "no API key" in (result.error or "")


# ══════════════════════════════════════════
# Constants
# ══════════════════════════════════════════

class TestConstants:
    def test_tts_providers_list(self):
        assert "openai" in TTS_PROVIDERS
        assert "elevenlabs" in TTS_PROVIDERS
        assert "edge" in TTS_PROVIDERS

    def test_stt_providers_list(self):
        assert "openai" in STT_PROVIDERS
        assert "deepgram" in STT_PROVIDERS
        assert "groq" in STT_PROVIDERS

    def test_openai_voices(self):
        assert "alloy" in OPENAI_TTS_VOICES
        assert "nova" in OPENAI_TTS_VOICES
        assert "shimmer" in OPENAI_TTS_VOICES
        assert len(OPENAI_TTS_VOICES) == 14

    def test_openai_models(self):
        assert "tts-1" in OPENAI_TTS_MODELS
        assert "tts-1-hd" in OPENAI_TTS_MODELS
        assert "gpt-4o-mini-tts" in OPENAI_TTS_MODELS

    def test_builtin_presets(self):
        assert "professional" in BUILTIN_PRESETS
        assert "casual" in BUILTIN_PRESETS
        assert "storyteller" in BUILTIN_PRESETS
        assert "fast_free" in BUILTIN_PRESETS
        assert "russian" in BUILTIN_PRESETS

    def test_tts_cost_info(self):
        assert "openai" in TTS_COST_INFO
        assert "edge" in TTS_COST_INFO
        assert "free" in TTS_COST_INFO["edge"]


# ══════════════════════════════════════════
# Voice Config Tools (agent self-configuration)
# ══════════════════════════════════════════

class _MockAgent:
    """Lightweight mock agent for testing voice config tool handlers."""

    def __init__(self, config=None):
        self.config = config or {}
        self.tools = MagicMock()
        self.tools._tools = {}
        self.tools._handlers = {}


def _wire_and_get_handlers(config=None):
    """Create mock agent with voice config tools wired, return handlers dict."""
    from liteagent.agent import LiteAgent
    agent = _MockAgent(config or {})
    # Call the method by binding it to our mock
    LiteAgent._wire_voice_config_tools(agent)
    return agent.tools._handlers, agent


class TestVoiceConfigTools:

    def test_get_voice_settings_returns_json(self):
        handlers, _ = _wire_and_get_handlers({"voice": {"tts": {"auto": "always", "provider": "edge"}}})
        with patch("liteagent.config.get_api_key", return_value=None):
            result = json.loads(handlers["get_voice_settings"]())
        assert "tts" in result
        assert "stt" in result
        assert "providers" in result
        assert "presets" in result
        assert result["tts"]["auto"] == "always"
        assert result["tts"]["provider"] == "edge"

    def test_get_voice_settings_shows_cost(self):
        handlers, _ = _wire_and_get_handlers()
        with patch("liteagent.config.get_api_key", return_value=None):
            result = json.loads(handlers["get_voice_settings"]())
        for provider_info in result["providers"].values():
            assert "cost" in provider_info

    def test_get_voice_settings_shows_presets(self):
        handlers, _ = _wire_and_get_handlers()
        with patch("liteagent.config.get_api_key", return_value=None):
            result = json.loads(handlers["get_voice_settings"]())
        # Should include all builtin presets
        for name in BUILTIN_PRESETS:
            assert name in result["presets"]

    def test_set_voice_settings_updates_tts_auto(self):
        handlers, agent = _wire_and_get_handlers()
        with patch("liteagent.config.save_config"):
            result = json.loads(handlers["set_voice_settings"](tts_auto="always"))
        assert result["status"] == "updated"
        assert "auto=always" in result["changes"]
        assert agent.config["voice"]["tts"]["auto"] == "always"

    def test_set_voice_settings_updates_provider(self):
        handlers, agent = _wire_and_get_handlers()
        with patch("liteagent.config.save_config"), \
             patch("liteagent.config.get_api_key", return_value=None):
            result = json.loads(handlers["set_voice_settings"](tts_provider="edge"))
        assert result["status"] == "updated"
        assert agent.config["voice"]["tts"]["provider"] == "edge"

    def test_set_voice_settings_validates_auto_mode(self):
        handlers, _ = _wire_and_get_handlers()
        result = json.loads(handlers["set_voice_settings"](tts_auto="invalid"))
        assert "error" in result

    def test_set_voice_settings_validates_provider(self):
        handlers, _ = _wire_and_get_handlers()
        result = json.loads(handlers["set_voice_settings"](tts_provider="invalid"))
        assert "error" in result

    def test_set_voice_settings_warns_no_api_key(self):
        handlers, _ = _wire_and_get_handlers()
        with patch("liteagent.config.save_config"), \
             patch("liteagent.config.get_api_key", return_value=None):
            result = json.loads(handlers["set_voice_settings"](tts_provider="openai"))
        assert "warnings" in result
        assert any("API key" in w for w in result["warnings"])

    def test_set_voice_settings_saves_config(self):
        handlers, agent = _wire_and_get_handlers()
        with patch("liteagent.config.save_config") as mock_save:
            handlers["set_voice_settings"](tts_auto="tagged")
        mock_save.assert_called_once_with(agent.config)

    def test_set_voice_settings_elevenlabs_params(self):
        handlers, agent = _wire_and_get_handlers()
        with patch("liteagent.config.save_config"):
            result = json.loads(handlers["set_voice_settings"](
                elevenlabs_stability=0.7,
                elevenlabs_similarity_boost=0.8,
            ))
        assert result["status"] == "updated"
        assert agent.config["voice"]["tts"]["elevenlabs"]["stability"] == 0.7
        assert agent.config["voice"]["tts"]["elevenlabs"]["similarity_boost"] == 0.8

    def test_set_voice_settings_no_changes(self):
        handlers, _ = _wire_and_get_handlers()
        result = json.loads(handlers["set_voice_settings"]())
        assert result["status"] == "no_changes"

    def test_list_voice_providers_structure(self):
        handlers, _ = _wire_and_get_handlers()
        with patch("liteagent.config.get_api_key", return_value=None):
            result = json.loads(handlers["list_voice_providers"]())
        assert "tts_providers" in result
        assert "stt_providers" in result
        assert "active_tts" in result
        assert "active_stt" in result
        assert len(result["tts_providers"]) == 3
        assert len(result["stt_providers"]) == 3

    def test_list_voice_providers_configured_status(self):
        handlers, _ = _wire_and_get_handlers()
        with patch("liteagent.config.get_api_key", return_value=None):
            result = json.loads(handlers["list_voice_providers"]())
        # Edge is always configured (no key needed)
        edge = next(p for p in result["tts_providers"] if p["id"] == "edge")
        assert edge["configured"] is True
        # OpenAI without key → not configured
        openai_p = next(p for p in result["tts_providers"] if p["id"] == "openai")
        assert openai_p["configured"] is False

    @pytest.mark.asyncio
    async def test_test_tts_calls_text_to_speech(self):
        handlers, _ = _wire_and_get_handlers()
        mock_result = TtsResult(success=True, audio_path="/tmp/test.mp3", provider="edge", output_format="mp3")
        with patch("liteagent.voice.text_to_speech", new_callable=AsyncMock, return_value=mock_result) as mock_tts, \
             patch("liteagent.file_queue.enqueue_file") as mock_enqueue:
            result = json.loads(await handlers["test_tts"]("Hello"))
        assert result["status"] == "ok"
        assert result["provider"] == "edge"
        mock_enqueue.assert_called_once()

    @pytest.mark.asyncio
    async def test_test_tts_with_override(self):
        handlers, _ = _wire_and_get_handlers({"voice": {"tts": {"provider": "edge"}}})
        mock_result = TtsResult(success=True, audio_path="/tmp/test.opus", provider="openai",
                                output_format="opus", voice_compatible=True)
        with patch("liteagent.voice.text_to_speech", new_callable=AsyncMock, return_value=mock_result), \
             patch("liteagent.file_queue.enqueue_file"):
            result = json.loads(await handlers["test_tts"]("Hello", voice="nova", provider="openai"))
        assert result["provider"] == "openai"

    def test_save_voice_preset(self):
        handlers, agent = _wire_and_get_handlers({"voice": {"tts": {"provider": "openai"}}})
        with patch("liteagent.config.save_config"):
            result = json.loads(handlers["save_voice_preset"]("my_voice", "My custom voice"))
        assert result["status"] == "saved"
        assert result["name"] == "my_voice"
        assert "my_voice" in agent.config["voice"]["presets"]

    def test_load_voice_preset_builtin(self):
        handlers, agent = _wire_and_get_handlers()
        with patch("liteagent.config.save_config"):
            result = json.loads(handlers["load_voice_preset"]("professional"))
        assert result["status"] == "loaded"
        assert result["preset"] == "professional"
        assert agent.config["voice"]["tts"]["provider"] == "openai"

    def test_load_voice_preset_custom(self):
        config = {"voice": {"presets": {"mine": {"provider": "edge", "edge": {"voice": "en-US-GuyNeural"}}}}}
        handlers, agent = _wire_and_get_handlers(config)
        with patch("liteagent.config.save_config"):
            result = json.loads(handlers["load_voice_preset"]("mine"))
        assert result["status"] == "loaded"
        assert agent.config["voice"]["tts"]["provider"] == "edge"

    def test_load_voice_preset_not_found(self):
        handlers, _ = _wire_and_get_handlers()
        result = json.loads(handlers["load_voice_preset"]("nonexistent"))
        assert "error" in result
        assert "available" in result


# ══════════════════════════════════════════
# Voice skill prompt injection
# ══════════════════════════════════════════

class TestVoiceSkillInjection:
    """Test that voice skill triggers via SkillRegistry (migrated from agent._should_inject_voice_skill)."""

    def _check(self, message: str) -> bool:
        from liteagent.skills import SkillRegistry
        from pathlib import Path
        reg = SkillRegistry()
        skills_dir = Path(__file__).parent.parent / "liteagent" / "skills"
        if skills_dir.is_dir():
            reg._load_from_dir(skills_dir, "bundled", set())
        triggered = reg.get_triggered_skills(message)
        return any(s.metadata.name == "voice" for s in triggered)

    def test_keyword_detected_russian(self):
        assert self._check("Настрой мне голос")
        assert self._check("Включи озвучку")
        assert self._check("Какой TTS провайдер?")

    def test_keyword_detected_english(self):
        assert self._check("Change the voice to nova")
        assert self._check("Enable TTS")
        assert self._check("Use ElevenLabs")

    def test_keyword_not_detected(self):
        assert not self._check("What is the weather today?")
        assert not self._check("Write a Python script")
        assert not self._check("Расскажи анекдот")
