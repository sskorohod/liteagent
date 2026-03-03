"""Per-request file queue using ContextVar for async task isolation.

Tools like send_file_to_user() enqueue files during agent.run().
API endpoints call get_file_queue() after run() to collect pending files.
"""

import contextvars
import mimetypes
import os
from dataclasses import dataclass, field

_file_queue: contextvars.ContextVar[list] = contextvars.ContextVar(
    "file_queue", default=None
)


@dataclass
class QueuedFile:
    path: str
    caption: str = ""
    mime_type: str = ""
    delete_after_send: bool = True
    voice_compatible: bool = False  # Send as voice bubble in Telegram (Opus format)


def init_file_queue() -> list[QueuedFile]:
    """Initialize (reset) the file queue for the current request context."""
    q: list[QueuedFile] = []
    _file_queue.set(q)
    return q


def enqueue_file(path: str, caption: str = "", mime_type: str = "",
                 delete_after_send: bool = True, voice_compatible: bool = False) -> None:
    """Add a file to the current request's outbound queue."""
    q = _file_queue.get(None)
    if q is None:
        q = []
        _file_queue.set(q)
    q.append(QueuedFile(
        path=path, caption=caption, mime_type=mime_type,
        delete_after_send=delete_after_send,
        voice_compatible=voice_compatible,
    ))


def get_file_queue() -> list[QueuedFile]:
    """Get the current request's file queue (or empty list)."""
    return _file_queue.get(None) or []


def serialize_file_queue(queue: list[QueuedFile]) -> list[dict]:
    """Serialize file queue for JSON API response."""
    result = []
    for f in queue:
        mime = f.mime_type or mimetypes.guess_type(f.path)[0] or "application/octet-stream"
        result.append({
            "path": f.path,
            "filename": os.path.basename(f.path),
            "caption": f.caption,
            "mime_type": mime,
            "delete_after_send": f.delete_after_send,
            "voice_compatible": f.voice_compatible,
        })
    return result
