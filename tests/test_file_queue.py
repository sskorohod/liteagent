"""Tests for file_queue module — ContextVar-based per-request file queue."""

import asyncio
import os
import tempfile

import pytest

from liteagent.file_queue import (
    QueuedFile, init_file_queue, enqueue_file,
    get_file_queue, serialize_file_queue,
)


class TestFileQueue:

    def test_init_and_enqueue(self):
        """Basic init → enqueue → get cycle."""
        init_file_queue()
        enqueue_file("/tmp/test.pdf", caption="Report")
        q = get_file_queue()
        assert len(q) == 1
        assert q[0].path == "/tmp/test.pdf"
        assert q[0].caption == "Report"
        assert q[0].delete_after_send is True

    def test_empty_queue(self):
        """get_file_queue returns empty list when nothing enqueued."""
        init_file_queue()
        q = get_file_queue()
        assert q == []

    def test_multiple_files(self):
        """Multiple files can be enqueued."""
        init_file_queue()
        enqueue_file("/tmp/a.txt", caption="A")
        enqueue_file("/tmp/b.png", caption="B", mime_type="image/png")
        enqueue_file("/tmp/c.pdf", delete_after_send=False)
        q = get_file_queue()
        assert len(q) == 3
        assert q[1].mime_type == "image/png"
        assert q[2].delete_after_send is False

    def test_init_resets_queue(self):
        """Calling init_file_queue() resets previous entries."""
        init_file_queue()
        enqueue_file("/tmp/old.txt")
        assert len(get_file_queue()) == 1

        init_file_queue()
        assert len(get_file_queue()) == 0

    def test_enqueue_without_init(self):
        """Enqueue creates queue lazily if init wasn't called."""
        # Reset contextvar by setting to None-returning default
        from liteagent.file_queue import _file_queue
        _file_queue.set(None)

        enqueue_file("/tmp/lazy.txt")
        q = get_file_queue()
        assert len(q) == 1
        assert q[0].path == "/tmp/lazy.txt"

    @pytest.mark.asyncio
    async def test_contextvar_isolation(self):
        """ContextVar should isolate queues between concurrent asyncio tasks."""
        results = {}

        async def task_a():
            init_file_queue()
            enqueue_file("/tmp/a1.txt")
            enqueue_file("/tmp/a2.txt")
            await asyncio.sleep(0.01)  # Yield control
            results["a"] = len(get_file_queue())

        async def task_b():
            init_file_queue()
            enqueue_file("/tmp/b1.txt")
            await asyncio.sleep(0.01)
            results["b"] = len(get_file_queue())

        await asyncio.gather(task_a(), task_b())

        assert results["a"] == 2
        assert results["b"] == 1


class TestSerializeFileQueue:

    def test_serialize_basic(self):
        """Serialize produces correct dict structure."""
        files = [QueuedFile(path="/tmp/doc.pdf", caption="test")]
        result = serialize_file_queue(files)
        assert len(result) == 1
        assert result[0]["filename"] == "doc.pdf"
        assert result[0]["caption"] == "test"
        assert result[0]["path"] == "/tmp/doc.pdf"
        assert result[0]["delete_after_send"] is True

    def test_serialize_mime_detection(self):
        """MIME type auto-detected from filename if not provided."""
        files = [
            QueuedFile(path="/tmp/photo.jpg"),
            QueuedFile(path="/tmp/data.json"),
            QueuedFile(path="/tmp/unknown"),
        ]
        result = serialize_file_queue(files)
        assert result[0]["mime_type"] == "image/jpeg"
        assert result[1]["mime_type"] == "application/json"
        assert result[2]["mime_type"] == "application/octet-stream"

    def test_serialize_explicit_mime(self):
        """Explicit mime_type takes precedence."""
        files = [QueuedFile(path="/tmp/file.bin", mime_type="application/pdf")]
        result = serialize_file_queue(files)
        assert result[0]["mime_type"] == "application/pdf"

    def test_serialize_empty(self):
        """Empty queue serializes to empty list."""
        assert serialize_file_queue([]) == []

    def test_serialize_with_real_file(self):
        """Serialize works with a real existing file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"hello")
            path = f.name

        try:
            files = [QueuedFile(path=path, caption="greeting")]
            result = serialize_file_queue(files)
            assert result[0]["filename"].endswith(".txt")
            assert result[0]["mime_type"] == "text/plain"
            assert os.path.basename(path) == result[0]["filename"]
        finally:
            os.unlink(path)
