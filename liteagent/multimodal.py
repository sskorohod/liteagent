"""Shared content-block builders for multimodal input (photos, documents, code).

Used by both the API channel (dashboard file upload) and the Telegram channel
(photo/document download). Converts raw file bytes into content block dicts
that agent.run() accepts.
"""

import base64
import logging

logger = logging.getLogger(__name__)

# ── Supported file types ──────────────────────────────────────

IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

DOC_TYPES = {
    "application/pdf", "text/plain", "text/csv", "text/html",
    "text/markdown", "application/json", "text/xml", "application/xml",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
    ".html", ".log", ".ini", ".toml", ".env", ".cfg",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h",
    ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".cs", ".sh", ".bash",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sql", ".r", ".lua",
    ".css", ".scss", ".less", ".vue", ".svelte",
}

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

    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()

    # ── Image ──
    if mime_type in IMAGE_TYPES:
        b64 = base64.b64encode(data).decode()
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": b64},
        }

    # ── PDF ──
    if mime_type == "application/pdf" or ext == ".pdf":
        b64 = base64.b64encode(data).decode()
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
        }

    # ── Text / Code / Known doc types ──
    if (mime_type in DOC_TYPES
            or ext in TEXT_EXTENSIONS
            or ext in CODE_EXTENSIONS
            or mime_type.startswith("text/")):
        text = _decode_text(data)
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n\n... [truncated]"
        return {
            "type": "text",
            "text": f"--- File: {filename} ---\n{text}\n--- End of {filename} ---",
        }

    # ── Unknown type: try text, fallback to binary note ──
    try:
        text = data.decode("utf-8")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n\n... [truncated]"
        return {
            "type": "text",
            "text": f"--- File: {filename} ({mime_type or 'unknown type'}) ---\n{text}\n--- End of {filename} ---",
        }
    except UnicodeDecodeError:
        size_kb = len(data) / 1024
        return {
            "type": "text",
            "text": (f"[Binary file attached: {filename} "
                     f"({mime_type or 'unknown type'}, {size_kb:.1f} KB) "
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
