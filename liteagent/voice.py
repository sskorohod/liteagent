"""Voice engine — TTS and STT provider abstraction layer.

TTS providers: OpenAI, ElevenLabs, Edge TTS (free, no API key)
STT providers: OpenAI Whisper, Deepgram, Groq

Architecture inspired by OpenClaw's voice system, adapted for Python async.
"""

import asyncio
import io
import logging
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════

TTS_PROVIDERS = ("openai", "elevenlabs", "edge")
STT_PROVIDERS = ("openai", "deepgram", "groq")

OPENAI_TTS_MODELS = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")
OPENAI_TTS_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "fable",
    "juniper",
    "marin",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
)

DEFAULT_TTS_MAX_LENGTH = 1500
DEFAULT_TTS_MAX_TEXT_LENGTH = 4096
DEFAULT_TTS_TIMEOUT_MS = 30_000

DEFAULT_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
DEFAULT_ELEVENLABS_VOICE_ID = "pMsXgVXv3BLzUgSXRplE"
DEFAULT_ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_ELEVENLABS_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True,
    "speed": 1.0,
}

DEFAULT_OPENAI_TTS_MODEL = "tts-1"
DEFAULT_OPENAI_TTS_VOICE = "nova"

DEFAULT_EDGE_VOICE = "ru-RU-SvetlanaNeural"
DEFAULT_EDGE_RATE = "+0%"
DEFAULT_EDGE_VOLUME = "+0%"
DEFAULT_EDGE_PITCH = "+0Hz"

# Output format routing (like OpenClaw's TELEGRAM_OUTPUT / DEFAULT_OUTPUT)
TELEGRAM_FORMAT = {
    "openai": "opus",
    "elevenlabs": "opus_48000_64",
    "edge": "webm-24khz-16bit-mono-opus",
    "extension": ".ogg",
    "voice_compatible": True,
}
# MS Edge TTS output format for requesting Opus directly (no ffmpeg needed)
EDGE_OPUS_FORMAT = "webm-24khz-16bit-mono-opus"
EDGE_MP3_FORMAT = "audio-24khz-48kbitrate-mono-mp3"
DEFAULT_FORMAT = {
    "openai": "mp3",
    "elevenlabs": "mp3_44100_128",
    "edge": "mp3",
    "extension": ".mp3",
    "voice_compatible": False,
}

VOICE_BUBBLE_CHANNELS = {"telegram"}

# ══════════════════════════════════════════
# BUILTIN PRESETS (improvement over OpenClaw)
# ══════════════════════════════════════════

BUILTIN_PRESETS: dict[str, dict] = {
    "professional": {
        "provider": "openai",
        "openai": {"voice": "onyx", "model": "tts-1-hd", "speed": 1.0},
    },
    "casual": {
        "provider": "openai",
        "openai": {"voice": "nova", "model": "tts-1", "speed": 1.1},
    },
    "storyteller": {
        "provider": "openai",
        "openai": {"voice": "fable", "model": "tts-1-hd", "speed": 0.9},
    },
    "fast_free": {
        "provider": "edge",
        "edge": {"voice": "en-US-MichelleNeural", "rate": "+10%"},
    },
    "russian": {
        "provider": "edge",
        "edge": {"voice": "ru-RU-SvetlanaNeural", "rate": "+0%"},
    },
}

TTS_COST_INFO = {
    "openai": "~$15/1M chars (tts-1), ~$30/1M (tts-1-hd)",
    "elevenlabs": "~$30/1M chars (depends on plan)",
    "edge": "free (unlimited)",
}

STT_COST_INFO = {
    "openai": "~$0.006/min (whisper-1)",
    "deepgram": "~$0.0043/min (nova-3)",
    "groq": "free (rate-limited)",
}

# ══════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════


@dataclass
class TtsResult:
    success: bool
    audio_path: str | None = None
    error: str | None = None
    latency_ms: int | None = None
    provider: str | None = None
    output_format: str | None = None
    voice_compatible: bool = False


@dataclass
class SttResult:
    success: bool
    text: str = ""
    error: str | None = None
    provider: str | None = None
    model: str | None = None


@dataclass
class TtsDirectiveOverrides:
    tts_text: str | None = None
    provider: str | None = None
    openai_voice: str | None = None
    openai_model: str | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model_id: str | None = None
    elevenlabs_stability: float | None = None
    elevenlabs_similarity_boost: float | None = None
    elevenlabs_style: float | None = None
    elevenlabs_speed: float | None = None
    elevenlabs_use_speaker_boost: bool | None = None


@dataclass
class TtsDirectiveParseResult:
    cleaned_text: str
    tts_text: str | None = None
    has_directive: bool = False
    overrides: TtsDirectiveOverrides = field(default_factory=TtsDirectiveOverrides)
    warnings: list[str] = field(default_factory=list)


# Last TTS attempt status (module-level, like OpenClaw)
_last_tts_attempt: dict | None = None


# ══════════════════════════════════════════
# CONFIG RESOLUTION
# ══════════════════════════════════════════


def resolve_voice_config(config: dict) -> dict:
    """Resolve voice config with defaults (like OpenClaw's resolveTtsConfig)."""
    voice = config.get("voice", {})
    tts = voice.get("tts", {})
    stt = voice.get("stt", {})

    return {
        "tts": {
            "auto": tts.get("auto", "off"),
            "provider": tts.get("provider", "edge"),
            "max_length": tts.get("max_length", DEFAULT_TTS_MAX_LENGTH),
            "max_text_length": tts.get("max_text_length", DEFAULT_TTS_MAX_TEXT_LENGTH),
            "timeout_ms": tts.get("timeout_ms", DEFAULT_TTS_TIMEOUT_MS),
            "openai": {
                "model": tts.get("openai", {}).get("model", DEFAULT_OPENAI_TTS_MODEL),
                "voice": tts.get("openai", {}).get("voice", DEFAULT_OPENAI_TTS_VOICE),
                "speed": tts.get("openai", {}).get("speed", 1.0),
            },
            "elevenlabs": {
                "voice_id": tts.get("elevenlabs", {}).get(
                    "voice_id", DEFAULT_ELEVENLABS_VOICE_ID
                ),
                "model_id": tts.get("elevenlabs", {}).get(
                    "model_id", DEFAULT_ELEVENLABS_MODEL_ID
                ),
                "stability": tts.get("elevenlabs", {}).get(
                    "stability", DEFAULT_ELEVENLABS_VOICE_SETTINGS["stability"]
                ),
                "similarity_boost": tts.get("elevenlabs", {}).get(
                    "similarity_boost",
                    DEFAULT_ELEVENLABS_VOICE_SETTINGS["similarity_boost"],
                ),
                "style": tts.get("elevenlabs", {}).get(
                    "style", DEFAULT_ELEVENLABS_VOICE_SETTINGS["style"]
                ),
                "use_speaker_boost": tts.get("elevenlabs", {}).get(
                    "use_speaker_boost",
                    DEFAULT_ELEVENLABS_VOICE_SETTINGS["use_speaker_boost"],
                ),
                "speed": tts.get("elevenlabs", {}).get(
                    "speed", DEFAULT_ELEVENLABS_VOICE_SETTINGS["speed"]
                ),
            },
            "edge": {
                "enabled": tts.get("edge", {}).get("enabled", True),
                "voice": tts.get("edge", {}).get("voice", DEFAULT_EDGE_VOICE),
                "rate": tts.get("edge", {}).get("rate", DEFAULT_EDGE_RATE),
                "volume": tts.get("edge", {}).get("volume", DEFAULT_EDGE_VOLUME),
                "pitch": tts.get("edge", {}).get("pitch") or DEFAULT_EDGE_PITCH,
            },
        },
        "stt": {
            "provider": stt.get("provider", "openai"),
            "openai": {
                "model": stt.get("openai", {}).get("model", "whisper-1"),
                "language": stt.get("openai", {}).get("language"),
            },
            "deepgram": {
                "model": stt.get("deepgram", {}).get("model", "nova-3"),
                "language": stt.get("deepgram", {}).get("language", "ru"),
            },
            "groq": {
                "model": stt.get("groq", {}).get("model", "whisper-large-v3"),
                "language": stt.get("groq", {}).get("language"),
            },
        },
    }


def _resolve_output_format(channel: str = "api") -> dict:
    """Select output format based on channel (like OpenClaw's resolveOutputFormat)."""
    if channel in VOICE_BUBBLE_CHANNELS:
        return TELEGRAM_FORMAT
    return DEFAULT_FORMAT


def _get_tts_api_key(provider: str, config: dict) -> str | None:
    """Get API key for TTS provider from config or env."""
    from .config import get_api_key

    if provider == "openai":
        return get_api_key("openai")
    if provider == "elevenlabs":
        # Check config first, then env vars (matching OpenClaw: ELEVENLABS_API_KEY, XI_API_KEY)
        key = get_api_key("elevenlabs")
        if key:
            return key
        return os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("XI_API_KEY")
    return None  # edge doesn't need a key


def _get_stt_api_key(provider: str) -> str | None:
    """Get API key for STT provider from config or env."""
    from .config import get_api_key

    if provider == "openai":
        return get_api_key("openai")
    if provider == "deepgram":
        key = get_api_key("deepgram")
        return key or os.environ.get("DEEPGRAM_API_KEY")
    if provider == "groq":
        key = get_api_key("groq")
        return key or os.environ.get("GROQ_API_KEY")
    return None


def _resolve_provider_order(primary: str) -> list[str]:
    """Build fallback chain: primary first, then remaining providers in order."""
    return [primary] + [p for p in TTS_PROVIDERS if p != primary]


def _get_tts_provider(voice_cfg: dict, config: dict) -> str:
    """Determine TTS provider: config priority → fallback to edge if unavailable."""
    provider = voice_cfg["tts"]["provider"]
    if provider == "edge":
        return "edge"
    if _get_tts_api_key(provider, config):
        return provider
    return "edge"


# ══════════════════════════════════════════
# MARKDOWN STRIP (for cleaner TTS output)
# ══════════════════════════════════════════

_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_ITALIC_UNDER_RE = re.compile(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
_STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~")
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^\)]+\)")
_BLOCKQUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
_UNORDERED_LIST_RE = re.compile(r"^[\s]*[-*+]\s", re.MULTILINE)
_ORDERED_LIST_RE = re.compile(r"^[\s]*\d+\.\s", re.MULTILINE)
_HR_RE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def strip_markdown(text: str) -> str:
    """Remove markdown formatting for cleaner TTS output."""
    result = text
    # Remove code blocks first (they may contain other patterns)
    result = _CODE_BLOCK_RE.sub("", result)
    # Remove images (before links, since image syntax includes link syntax)
    result = _IMAGE_RE.sub(r"\1", result)
    # Remove links, keep text
    result = _LINK_RE.sub(r"\1", result)
    # Remove inline code
    result = _INLINE_CODE_RE.sub(r"\1", result)
    # Remove formatting markers
    result = _BOLD_RE.sub(r"\1", result)
    result = _ITALIC_RE.sub(r"\1", result)
    result = _ITALIC_UNDER_RE.sub(r"\1", result)
    result = _STRIKETHROUGH_RE.sub(r"\1", result)
    # Remove structural elements
    result = _HEADING_RE.sub("", result)
    result = _BLOCKQUOTE_RE.sub("", result)
    result = _HR_RE.sub("", result)
    # Clean up list markers (keep text)
    result = _UNORDERED_LIST_RE.sub("", result)
    result = _ORDERED_LIST_RE.sub("", result)
    # Remove HTML tags
    result = _HTML_TAG_RE.sub("", result)
    # Clean up extra whitespace
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ══════════════════════════════════════════
# TTS DIRECTIVE PARSING (like OpenClaw's parseTtsDirectives)
# ══════════════════════════════════════════

_TTS_BLOCK_RE = re.compile(
    r"\[\[tts:text\]\]([\s\S]*?)\[\[/tts:text\]\]", re.IGNORECASE
)
_TTS_DIRECTIVE_RE = re.compile(r"\[\[tts:([^\]]+)\]\]", re.IGNORECASE)


def _parse_bool(value: str) -> bool | None:
    v = value.strip().lower()
    if v in ("true", "1", "yes", "on"):
        return True
    if v in ("false", "0", "no", "off"):
        return False
    return None


def _parse_float(value: str) -> float | None:
    try:
        v = float(value)
        if not (v != v):  # not NaN
            return v
    except (ValueError, TypeError):
        pass
    return None


def parse_tts_directives(text: str) -> TtsDirectiveParseResult:
    """Parse [[tts:...]] directives from model output text."""
    overrides = TtsDirectiveOverrides()
    warnings: list[str] = []
    cleaned = text
    has_directive = False

    # Extract [[tts:text]]...[[/tts:text]] blocks
    def _replace_block(m):
        nonlocal has_directive
        has_directive = True
        if overrides.tts_text is None:
            overrides.tts_text = m.group(1).strip()
        return ""

    cleaned = _TTS_BLOCK_RE.sub(_replace_block, cleaned)

    # Extract [[tts:key=value ...]] directives
    def _replace_directive(m):
        nonlocal has_directive
        has_directive = True
        body = m.group(1)
        tokens = body.split()
        for token in tokens:
            eq_idx = token.find("=")
            if eq_idx == -1:
                continue
            key = token[:eq_idx].strip().lower()
            value = token[eq_idx + 1 :].strip()
            if not key or not value:
                continue
            try:
                if key == "provider":
                    if value in TTS_PROVIDERS:
                        overrides.provider = value
                    else:
                        warnings.append(f'unsupported provider "{value}"')
                elif key in ("voice", "openai_voice"):
                    if value in OPENAI_TTS_VOICES or True:  # Allow any for flexibility
                        overrides.openai_voice = value
                elif key in ("voiceid", "voice_id", "elevenlabs_voice"):
                    overrides.elevenlabs_voice_id = value
                elif key in ("model", "openai_model"):
                    if value in OPENAI_TTS_MODELS:
                        overrides.openai_model = value
                    else:
                        overrides.elevenlabs_model_id = value
                elif key == "stability":
                    v = _parse_float(value)
                    if v is not None and 0 <= v <= 1:
                        overrides.elevenlabs_stability = v
                    else:
                        warnings.append("invalid stability value")
                elif key in ("similarity", "similarityboost", "similarity_boost"):
                    v = _parse_float(value)
                    if v is not None and 0 <= v <= 1:
                        overrides.elevenlabs_similarity_boost = v
                    else:
                        warnings.append("invalid similarity_boost value")
                elif key == "style":
                    v = _parse_float(value)
                    if v is not None and 0 <= v <= 1:
                        overrides.elevenlabs_style = v
                    else:
                        warnings.append("invalid style value")
                elif key == "speed":
                    v = _parse_float(value)
                    if v is not None and 0.5 <= v <= 4.0:
                        overrides.elevenlabs_speed = v
                    else:
                        warnings.append("invalid speed value")
                elif key in ("speakerboost", "speaker_boost", "use_speaker_boost"):
                    v = _parse_bool(value)
                    if v is not None:
                        overrides.elevenlabs_use_speaker_boost = v
                    else:
                        warnings.append("invalid use_speaker_boost value")
            except Exception as e:
                warnings.append(str(e))
        return ""

    cleaned = _TTS_DIRECTIVE_RE.sub(_replace_directive, cleaned)

    return TtsDirectiveParseResult(
        cleaned_text=cleaned,
        tts_text=overrides.tts_text,
        has_directive=has_directive,
        overrides=overrides,
        warnings=warnings,
    )


# ══════════════════════════════════════════
# TTS PROVIDER IMPLEMENTATIONS
# ══════════════════════════════════════════


async def openai_tts(
    text: str,
    api_key: str,
    model: str = DEFAULT_OPENAI_TTS_MODEL,
    voice: str = DEFAULT_OPENAI_TTS_VOICE,
    response_format: str = "mp3",
    speed: float = 1.0,
    timeout_ms: int = DEFAULT_TTS_TIMEOUT_MS,
) -> bytes:
    """Synthesize speech using OpenAI TTS API."""
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai package required: pip install liteagent[openai]")

    client = openai.AsyncOpenAI(api_key=api_key, timeout=timeout_ms / 1000)

    kwargs = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": response_format,
    }
    if speed != 1.0:
        kwargs["speed"] = speed

    response = await client.audio.speech.create(**kwargs)
    return response.content


async def elevenlabs_tts(
    text: str,
    api_key: str,
    voice_id: str = DEFAULT_ELEVENLABS_VOICE_ID,
    model_id: str = DEFAULT_ELEVENLABS_MODEL_ID,
    output_format: str = "mp3_44100_128",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
    use_speaker_boost: bool = True,
    speed: float = 1.0,
    timeout_ms: int = DEFAULT_TTS_TIMEOUT_MS,
) -> bytes:
    """Synthesize speech using ElevenLabs REST API (no SDK required)."""
    import json as _json

    try:
        from urllib.request import Request, urlopen
    except ImportError:
        raise RuntimeError("urllib required for ElevenLabs TTS")

    url = f"{DEFAULT_ELEVENLABS_BASE_URL}/v1/text-to-speech/{voice_id}"
    if output_format:
        url += f"?output_format={output_format}"

    body = _json.dumps(
        {
            "text": text,
            "model_id": model_id,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
                "style": style,
                "use_speaker_boost": use_speaker_boost,
                "speed": speed,
            },
        }
    ).encode("utf-8")

    req = Request(url, data=body, method="POST")
    req.add_header("xi-api-key", api_key)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "audio/mpeg")

    # Run blocking urllib in thread pool to stay async
    loop = asyncio.get_event_loop()
    timeout_sec = timeout_ms / 1000

    def _fetch():
        import urllib.error

        try:
            resp = urlopen(req, timeout=timeout_sec)
            if resp.status != 200:
                raise RuntimeError(f"ElevenLabs API error ({resp.status})")
            return resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ElevenLabs API error ({e.code})") from e

    return await loop.run_in_executor(None, _fetch)


async def _convert_to_ogg_opus(input_path: str, output_path: str) -> bool:
    """Convert audio file to OGG Opus optimized for Telegram voice messages.

    Telegram requires OGG Opus for full voice message support (waveform,
    speed control). Parameters: 48kHz mono, 48kbps VBR, VOIP application.
    Returns True if conversion succeeded, False if ffmpeg unavailable.
    """
    import shutil
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found — cannot convert to OGG Opus")
        return False

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-c:a", "libopus",
        "-b:a", "48k",
        "-vbr", "on",
        "-compression_level", "10",
        "-frame_duration", "60",
        "-application", "voip",
        "-ar", "48000",
        "-ac", "1",
        output_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            logger.warning("ffmpeg OGG Opus conversion failed: %s", stderr.decode()[-200:])
            return False
        return True
    except (asyncio.TimeoutError, FileNotFoundError) as e:
        logger.warning("ffmpeg conversion error: %s", e)
        return False


async def edge_tts_synthesize(
    text: str,
    output_path: str,
    voice: str = DEFAULT_EDGE_VOICE,
    rate: str = DEFAULT_EDGE_RATE,
    volume: str = DEFAULT_EDGE_VOLUME,
    pitch: str = "+0Hz",
    timeout_ms: int = DEFAULT_TTS_TIMEOUT_MS,
    output_format: str | None = None,
    retries: int = 2,
) -> None:
    """Synthesize speech using Microsoft Edge TTS (free, no API key).

    Args:
        output_format: MS TTS output format string. None uses library default (MP3).
                       Use EDGE_OPUS_FORMAT for WebM/Opus (Telegram voice bubbles).
        retries: Number of retry attempts on transient failure (default 2 = up to 3 total).
    """
    try:
        import edge_tts
    except ImportError:
        raise RuntimeError("edge-tts package required: pip install liteagent[voice]")

    timeout_sec = timeout_ms / 1000
    last_error: Exception | None = None

    for attempt in range(1 + retries):
        try:
            communicate = edge_tts.Communicate(
                text, voice, rate=rate, volume=volume, pitch=pitch
            )

            if output_format and output_format != EDGE_MP3_FORMAT:
                await asyncio.wait_for(
                    _edge_tts_save_custom_format(communicate, output_path, output_format),
                    timeout=timeout_sec,
                )
            else:
                await asyncio.wait_for(
                    communicate.save(output_path),
                    timeout=timeout_sec,
                )
            return  # success
        except (asyncio.TimeoutError, RuntimeError, Exception) as e:
            last_error = e
            if attempt < retries:
                wait = 0.3 * (attempt + 1)
                logger.warning("Edge TTS attempt %d/%d failed: %s — retrying in %.1fs",
                               attempt + 1, 1 + retries, e, wait)
                await asyncio.sleep(wait)
            else:
                logger.warning("Edge TTS failed after %d attempts: %s", 1 + retries, e)

    raise last_error or RuntimeError("Edge TTS failed")


async def _edge_tts_save_custom_format(
    communicate, output_path: str, output_format: str
) -> None:
    """Save edge-tts audio with a custom MS outputFormat (e.g. WebM/Opus).

    The edge-tts library hardcodes MP3 format and validates content-type
    as audio/mpeg. This function patches both to support alternative formats
    like webm-24khz-16bit-mono-opus directly from the MS TTS service.
    """
    import aiohttp
    import edge_tts.communicate as _comm

    # 1. Patch websocket send to replace output format in config message
    _orig_send = aiohttp.ClientWebSocketResponse.send_str

    async def _patched_send(ws_self, data, compress=None):
        if EDGE_MP3_FORMAT in data:
            data = data.replace(EDGE_MP3_FORMAT, output_format)
        return await _orig_send(ws_self, data, compress=compress)

    # 2. Patch header parser to rewrite audio/* content-type to audio/mpeg
    #    so edge-tts doesn't reject WebM/Opus responses
    _orig_get_headers = _comm.get_headers_and_data

    def _patched_get_headers(data: bytes, header_length: int):
        headers, body = _orig_get_headers(data, header_length)
        ct = headers.get(b"Content-Type", None)
        if ct and ct != b"audio/mpeg" and ct.startswith(b"audio/"):
            headers[b"Content-Type"] = b"audio/mpeg"
        return headers, body

    aiohttp.ClientWebSocketResponse.send_str = _patched_send
    _comm.get_headers_and_data = _patched_get_headers
    try:
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        if not audio_data:
            raise RuntimeError("Edge TTS returned no audio data")
        with open(output_path, "wb") as f:
            f.write(audio_data)
    finally:
        aiohttp.ClientWebSocketResponse.send_str = _orig_send
        _comm.get_headers_and_data = _orig_get_headers


# ══════════════════════════════════════════
# TTS ORCHESTRATION (like OpenClaw's textToSpeech + maybeApplyTtsToPayload)
# ══════════════════════════════════════════


async def text_to_speech(
    text: str,
    voice_cfg: dict,
    config: dict,
    channel: str = "api",
    overrides: TtsDirectiveOverrides | None = None,
) -> TtsResult:
    """Convert text to speech with provider fallback chain.

    Matches OpenClaw's textToSpeech() architecture:
    primary provider → remaining providers → fail.
    """
    tts_cfg = voice_cfg["tts"]
    output = _resolve_output_format(channel)

    if len(text) > tts_cfg["max_text_length"]:
        return TtsResult(
            success=False,
            error=f"Text too long ({len(text)} chars, max {tts_cfg['max_text_length']})",
        )

    # Determine provider order
    primary = (
        overrides.provider
        if (overrides and overrides.provider)
        else _get_tts_provider(voice_cfg, config)
    )
    providers = _resolve_provider_order(primary)

    errors: list[str] = []

    for provider in providers:
        start = time.monotonic()
        try:
            if provider == "edge":
                if not tts_cfg["edge"]["enabled"]:
                    errors.append("edge: disabled")
                    continue

                edge_cfg = tts_cfg["edge"]
                needs_opus = output.get("voice_compatible", False)
                tmp_dir = tempfile.mkdtemp(prefix="tts-")

                # Always generate MP3 first (most reliable)
                mp3_path = os.path.join(
                    tmp_dir, f"voice-{int(time.time())}.mp3"
                )

                await edge_tts_synthesize(
                    text=text,
                    output_path=mp3_path,
                    voice=edge_cfg["voice"],
                    rate=edge_cfg["rate"],
                    volume=edge_cfg["volume"],
                    pitch=edge_cfg.get("pitch") or DEFAULT_EDGE_PITCH,
                    output_format=None,  # default MP3
                    timeout_ms=tts_cfg["timeout_ms"],
                )

                # For Telegram: convert MP3 → OGG Opus (proper format)
                if needs_opus:
                    audio_path = os.path.join(
                        tmp_dir, f"voice-{int(time.time())}.ogg"
                    )
                    converted = await _convert_to_ogg_opus(mp3_path, audio_path)
                    if converted:
                        os.unlink(mp3_path)
                        edge_format = EDGE_OPUS_FORMAT
                    else:
                        # ffmpeg not available — send MP3 as fallback
                        audio_path = mp3_path
                        edge_format = EDGE_MP3_FORMAT
                else:
                    audio_path = mp3_path
                    edge_format = EDGE_MP3_FORMAT

                latency = int((time.monotonic() - start) * 1000)
                return TtsResult(
                    success=True,
                    audio_path=audio_path,
                    latency_ms=latency,
                    provider=provider,
                    output_format=edge_format,
                    voice_compatible=output["voice_compatible"],
                )

            api_key = _get_tts_api_key(provider, config)
            if not api_key:
                errors.append(f"{provider}: no API key")
                continue

            if provider == "elevenlabs":
                el_cfg = tts_cfg["elevenlabs"]
                audio_bytes = await elevenlabs_tts(
                    text=text,
                    api_key=api_key,
                    voice_id=(overrides.elevenlabs_voice_id if overrides else None)
                    or el_cfg["voice_id"],
                    model_id=(overrides.elevenlabs_model_id if overrides else None)
                    or el_cfg["model_id"],
                    output_format=output["elevenlabs"],
                    stability=el_cfg["stability"]
                    if not (overrides and overrides.elevenlabs_stability is not None)
                    else overrides.elevenlabs_stability,
                    similarity_boost=el_cfg["similarity_boost"]
                    if not (
                        overrides and overrides.elevenlabs_similarity_boost is not None
                    )
                    else overrides.elevenlabs_similarity_boost,
                    style=el_cfg["style"]
                    if not (overrides and overrides.elevenlabs_style is not None)
                    else overrides.elevenlabs_style,
                    use_speaker_boost=el_cfg["use_speaker_boost"]
                    if not (
                        overrides and overrides.elevenlabs_use_speaker_boost is not None
                    )
                    else overrides.elevenlabs_use_speaker_boost,
                    speed=el_cfg["speed"]
                    if not (overrides and overrides.elevenlabs_speed is not None)
                    else overrides.elevenlabs_speed,
                    timeout_ms=tts_cfg["timeout_ms"],
                )
            else:  # openai
                oai_cfg = tts_cfg["openai"]
                audio_bytes = await openai_tts(
                    text=text,
                    api_key=api_key,
                    model=(overrides.openai_model if overrides else None)
                    or oai_cfg["model"],
                    voice=(overrides.openai_voice if overrides else None)
                    or oai_cfg["voice"],
                    response_format=output["openai"],
                    speed=oai_cfg.get("speed", 1.0),
                    timeout_ms=tts_cfg["timeout_ms"],
                )

            latency = int((time.monotonic() - start) * 1000)

            # Save to temp file
            tmp_dir = tempfile.mkdtemp(prefix="tts-")
            audio_path = os.path.join(
                tmp_dir, f"voice-{int(time.time())}{output['extension']}"
            )
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            return TtsResult(
                success=True,
                audio_path=audio_path,
                latency_ms=latency,
                provider=provider,
                output_format=output["openai"]
                if provider == "openai"
                else output["elevenlabs"],
                voice_compatible=output["voice_compatible"],
            )

        except Exception as e:
            err_msg = f"{provider}: {e}"
            if "AbortError" in str(type(e).__name__) or "timeout" in str(e).lower():
                err_msg = f"{provider}: request timed out"
            errors.append(err_msg)
            logger.warning("TTS provider %s failed: %s", provider, e)

    return TtsResult(
        success=False,
        error=f"TTS conversion failed: {'; '.join(errors) or 'no providers available'}",
    )


async def maybe_apply_tts(
    text: str,
    config: dict,
    channel: str = "api",
    inbound_audio: bool = False,
) -> TtsResult | None:
    """Apply auto-TTS to agent response if enabled.

    Matches OpenClaw's maybeApplyTtsToPayload() logic:
    1. Check auto mode (off/always/inbound/tagged)
    2. Parse [[tts:...]] directives
    3. Skip if text too short (<10 chars) or has media
    4. Truncate if text > max_length
    5. Strip markdown
    6. Call TTS with fallback
    """
    global _last_tts_attempt

    voice_cfg = resolve_voice_config(config)
    auto_mode = voice_cfg["tts"]["auto"]

    if auto_mode == "off":
        return None

    # Parse directives
    directives = parse_tts_directives(text)
    if directives.warnings:
        logger.debug("TTS: directive warnings: %s", "; ".join(directives.warnings))

    cleaned = directives.cleaned_text.strip()
    tts_text = directives.tts_text or cleaned

    # Mode gating
    if auto_mode == "tagged" and not directives.has_directive:
        return None
    if auto_mode == "inbound" and not inbound_audio:
        return None

    # Skip short/empty text
    if not tts_text.strip() or len(tts_text.strip()) < 10:
        return None

    # Truncate if needed
    max_length = voice_cfg["tts"]["max_length"]
    if len(tts_text) > max_length:
        logger.debug("TTS: truncating text (%d > %d)", len(tts_text), max_length)
        tts_text = tts_text[: max_length - 3] + "..."

    # Strip markdown for cleaner TTS
    tts_text = strip_markdown(tts_text).strip()
    if len(tts_text) < 10:
        return None

    # Hard cap
    max_text = voice_cfg["tts"]["max_text_length"]
    if len(tts_text) > max_text:
        tts_text = tts_text[: max_text - 3] + "..."

    # Synthesize
    result = await text_to_speech(
        text=tts_text,
        voice_cfg=voice_cfg,
        config=config,
        channel=channel,
        overrides=directives.overrides,
    )

    # Track last attempt
    _last_tts_attempt = {
        "timestamp": time.time(),
        "success": result.success,
        "text_length": len(text),
        "provider": result.provider,
        "latency_ms": result.latency_ms,
        "error": result.error,
    }

    if result.success:
        logger.info(
            "TTS: %s generated in %dms (%s)",
            result.provider,
            result.latency_ms or 0,
            result.output_format,
        )
    else:
        logger.warning("TTS: failed — %s", result.error)

    return result


def get_last_tts_attempt() -> dict | None:
    """Get info about the last TTS attempt."""
    return _last_tts_attempt


# ══════════════════════════════════════════
# STT PROVIDER IMPLEMENTATIONS
# ══════════════════════════════════════════


async def transcribe_openai(
    audio_bytes: bytes,
    api_key: str,
    model: str = "whisper-1",
    language: str | None = None,
) -> SttResult:
    """Transcribe audio using OpenAI Whisper API."""
    try:
        import openai
    except ImportError:
        return SttResult(
            success=False,
            error="openai package required: pip install liteagent[openai]",
        )

    client = openai.AsyncOpenAI(api_key=api_key)
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice.ogg"

    kwargs = {"model": model, "file": audio_file}
    if language:
        kwargs["language"] = language

    try:
        response = await client.audio.transcriptions.create(**kwargs)
        return SttResult(
            success=True, text=response.text, provider="openai", model=model
        )
    except Exception as e:
        return SttResult(
            success=False, error=f"OpenAI transcription error: {e}", provider="openai"
        )


async def transcribe_deepgram(
    audio_bytes: bytes,
    api_key: str,
    model: str = "nova-3",
    language: str | None = None,
) -> SttResult:
    """Transcribe audio using Deepgram REST API (no SDK required)."""
    import json as _json

    try:
        from urllib.request import Request, urlopen
    except ImportError:
        return SttResult(success=False, error="urllib required for Deepgram STT")

    url = f"https://api.deepgram.com/v1/listen?model={model}"
    if language:
        url += f"&language={language}"

    req = Request(url, data=audio_bytes, method="POST")
    req.add_header("Authorization", f"Token {api_key}")
    req.add_header("Content-Type", "audio/ogg")

    loop = asyncio.get_event_loop()

    def _fetch():
        import urllib.error

        try:
            resp = urlopen(req, timeout=30)
            data = _json.loads(resp.read())
            transcript = (
                data.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )
            if not transcript:
                return SttResult(
                    success=False,
                    error="Empty transcript from Deepgram",
                    provider="deepgram",
                )
            return SttResult(
                success=True, text=transcript, provider="deepgram", model=model
            )
        except urllib.error.HTTPError as e:
            return SttResult(
                success=False,
                error=f"Deepgram API error ({e.code})",
                provider="deepgram",
            )
        except Exception as e:
            return SttResult(
                success=False, error=f"Deepgram error: {e}", provider="deepgram"
            )

    return await loop.run_in_executor(None, _fetch)


async def transcribe_groq(
    audio_bytes: bytes,
    api_key: str,
    model: str = "whisper-large-v3",
    language: str | None = None,
) -> SttResult:
    """Transcribe audio using Groq API (OpenAI-compatible)."""
    try:
        import openai
    except ImportError:
        return SttResult(
            success=False,
            error="openai package required: pip install liteagent[openai]",
        )

    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = "voice.ogg"

    kwargs = {"model": model, "file": audio_file}
    if language:
        kwargs["language"] = language

    try:
        response = await client.audio.transcriptions.create(**kwargs)
        return SttResult(success=True, text=response.text, provider="groq", model=model)
    except Exception as e:
        return SttResult(
            success=False, error=f"Groq transcription error: {e}", provider="groq"
        )


async def transcribe(
    audio_bytes: bytes,
    config: dict,
) -> SttResult:
    """Transcribe audio using the configured STT provider with fallback."""
    # Validate audio before sending to providers
    if not audio_bytes:
        return SttResult(success=False, error="Empty audio data")
    if len(audio_bytes) < 100:
        return SttResult(
            success=False,
            error=f"Audio too small ({len(audio_bytes)} bytes) — likely empty or corrupted",
        )

    voice_cfg = resolve_voice_config(config)
    stt_cfg = voice_cfg["stt"]
    provider = stt_cfg["provider"]
    errors: list[str] = []

    # Try primary provider
    result = await _transcribe_with_provider(audio_bytes, provider, stt_cfg)
    if result.success:
        return result
    errors.append(f"{provider}: {result.error}")
    logger.warning("STT primary provider %s failed: %s", provider, result.error)

    # Fallback to other providers
    for fallback in STT_PROVIDERS:
        if fallback == provider:
            continue
        api_key = _get_stt_api_key(fallback)
        if not api_key:
            errors.append(f"{fallback}: no API key")
            continue
        logger.info("STT: %s failed, trying %s", provider, fallback)
        result = await _transcribe_with_provider(audio_bytes, fallback, stt_cfg)
        if result.success:
            return result
        errors.append(f"{fallback}: {result.error}")
        logger.warning("STT fallback %s failed: %s", fallback, result.error)

    return SttResult(success=False, error=f"All STT providers failed: {'; '.join(errors)}")


async def _transcribe_with_provider(
    audio_bytes: bytes,
    provider: str,
    stt_cfg: dict,
) -> SttResult:
    """Transcribe with a specific STT provider."""
    api_key = _get_stt_api_key(provider)
    if not api_key:
        return SttResult(
            success=False, error=f"{provider}: no API key", provider=provider
        )

    pcfg = stt_cfg.get(provider, {})

    if provider == "openai":
        return await transcribe_openai(
            audio_bytes,
            api_key,
            model=pcfg.get("model", "whisper-1"),
            language=pcfg.get("language"),
        )
    elif provider == "deepgram":
        return await transcribe_deepgram(
            audio_bytes,
            api_key,
            model=pcfg.get("model", "nova-3"),
            language=pcfg.get("language"),
        )
    elif provider == "groq":
        return await transcribe_groq(
            audio_bytes,
            api_key,
            model=pcfg.get("model", "whisper-large-v3"),
            language=pcfg.get("language"),
        )
    else:
        return SttResult(success=False, error=f"Unknown STT provider: {provider}")
