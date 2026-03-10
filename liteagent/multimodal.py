"""Shared content-block builders for multimodal input (photos, documents, code).

Used by both the API channel (dashboard file upload) and the Telegram channel
(photo/document download). Converts raw file bytes into content block dicts
that agent.run() accepts.
"""

import base64
import logging

from .file_types import (
    CODE_EXTENSIONS,
    IMAGE_MIME_TYPES,
    TEXT_EXTENSIONS,
    detect_file_type,
    extract_text_from_file,
)

logger = logging.getLogger(__name__)

# ── Supported file types ──────────────────────────────────────

IMAGE_TYPES = IMAGE_MIME_TYPES

MAX_FILE_SIZE = 10 * 1024 * 1024   # 10 MB
MAX_TEXT_CHARS = 100_000           # ~100K chars


def file_to_content_block(
    data: bytes,
    filename: str,
    mime_type: str = "",
) -> dict:
    """Convert raw file bytes into a content block dict for agent.run().

    Returns one of:
      - {"type": "image", "source": {"type": "base64", ...}}   for images
      - {"type": "document", "source": {"type": "base64", ...}} for PDFs
      - {"type": "text", "text": "..."}                         for text/code/binary

    Raises:
        ValueError: if file exceeds MAX_FILE_SIZE
    """
    if len(data) > MAX_FILE_SIZE:
        size_mb = len(data) / (1024 * 1024)
        raise ValueError(
            f"File '{filename}' too large ({size_mb:.1f} MB, max {MAX_FILE_SIZE // (1024*1024)} MB)"
        )

    info = detect_file_type(data, filename, mime_type)

    # ── Image ──
    if info.is_image:
        b64 = base64.b64encode(data).decode()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": info.mime_type,
                "data": b64,
                "filename": filename,
            },
        }

    # ── PDF ──
    if info.is_pdf:
        b64 = base64.b64encode(data).decode()
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": b64,
                "filename": filename,
            },
        }

    # ── Text / Code / Extractable office docs ──
    if info.can_extract_text:
        text = extract_text_from_file(data, info)
        if not text and (info.mime_type.startswith("text/") or info.extension in TEXT_EXTENSIONS or info.extension in CODE_EXTENSIONS):
            text = _decode_text(data)
        if not text:
            size_kb = len(data) / 1024
            return {
                "type": "text",
                "text": (f"[Binary file attached: {filename} "
                         f"({info.label}, {info.mime_type or 'unknown type'}, {size_kb:.1f} KB) "
                         f"— content extraction unavailable]"),
            }
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n\n... [truncated]"
        return {
            "type": "text",
            "text": f"--- File: {filename} ---\n{text}\n--- End of {filename} ---",
        }

    # ── Unknown-ish type: try text, fallback to typed binary note ──
    size_kb = len(data) / 1024
    return {
        "type": "text",
        "text": (f"[Binary file attached: {filename} "
                 f"({info.label}, {info.mime_type or 'unknown type'}, {size_kb:.1f} KB) "
                 f"— cannot display content]"),
    }


def file_to_emoji(block: dict, filename: str = "") -> str:
    """Return a short emoji description for a file block (for UI)."""
    btype = block.get("type", "")
    if btype == "image":
        return f"\U0001f5bc {filename}"
    elif btype == "document":
        return f"\U0001f4c4 {filename}"
    else:
        text = block.get("text", "")
        if "[Binary file" in text:
            return f"\U0001f4e6 {filename}"
        return f"\U0001f4ce {filename}"


def _decode_text(data: bytes) -> str:
    """Try UTF-8, fallback to latin-1."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")
