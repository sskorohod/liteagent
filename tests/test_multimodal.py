"""Tests for liteagent/multimodal.py — content-block builder."""

import base64
import pytest
from liteagent.multimodal import (
    file_to_content_block,
    file_to_emoji,
    IMAGE_TYPES,
    CODE_EXTENSIONS,
    MAX_FILE_SIZE,
    MAX_TEXT_CHARS,
)


class TestImageBlocks:
    """Image files → image content blocks."""

    def test_jpeg_image(self):
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100  # fake JPEG header
        block = file_to_content_block(data, "photo.jpg", "image/jpeg")
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "image/jpeg"
        assert block["source"]["data"] == base64.b64encode(data).decode()

    def test_png_image(self):
        data = b"\x89PNG" + b"\x00" * 50
        block = file_to_content_block(data, "screenshot.png", "image/png")
        assert block["type"] == "image"
        assert block["source"]["media_type"] == "image/png"

    def test_webp_image(self):
        data = b"RIFF" + b"\x00" * 50
        block = file_to_content_block(data, "photo.webp", "image/webp")
        assert block["type"] == "image"
        assert block["source"]["media_type"] == "image/webp"

    def test_gif_image(self):
        data = b"GIF89a" + b"\x00" * 50
        block = file_to_content_block(data, "animation.gif", "image/gif")
        assert block["type"] == "image"
        assert block["source"]["media_type"] == "image/gif"


class TestPDFBlocks:
    """PDF files → document content blocks."""

    def test_pdf_by_mime(self):
        data = b"%PDF-1.4" + b"\x00" * 100
        block = file_to_content_block(data, "report.pdf", "application/pdf")
        assert block["type"] == "document"
        assert block["source"]["type"] == "base64"
        assert block["source"]["media_type"] == "application/pdf"

    def test_pdf_by_extension(self):
        """Even without correct MIME, .pdf extension should work."""
        data = b"%PDF-1.4" + b"\x00" * 100
        block = file_to_content_block(data, "report.pdf", "")
        assert block["type"] == "document"
        assert block["source"]["media_type"] == "application/pdf"


class TestTextBlocks:
    """Text and code files → text content blocks."""

    def test_python_file(self):
        code = b'def hello():\n    print("world")\n'
        block = file_to_content_block(code, "main.py", "text/x-python")
        assert block["type"] == "text"
        assert "--- File: main.py ---" in block["text"]
        assert 'def hello()' in block["text"]
        assert "--- End of main.py ---" in block["text"]

    def test_javascript_file(self):
        code = b'console.log("hello");'
        block = file_to_content_block(code, "app.js", "")
        assert block["type"] == "text"
        assert "--- File: app.js ---" in block["text"]

    def test_txt_file(self):
        text = b"Hello, world! This is a plain text file."
        block = file_to_content_block(text, "notes.txt", "text/plain")
        assert block["type"] == "text"
        assert "Hello, world!" in block["text"]

    def test_csv_file(self):
        csv_data = b"name,age\nAlice,30\nBob,25\n"
        block = file_to_content_block(csv_data, "data.csv", "text/csv")
        assert block["type"] == "text"
        assert "Alice,30" in block["text"]

    def test_json_file(self):
        json_data = b'{"key": "value"}'
        block = file_to_content_block(json_data, "config.json", "application/json")
        assert block["type"] == "text"

    def test_markdown_file(self):
        md = b"# Title\n\nSome text"
        block = file_to_content_block(md, "readme.md", "text/markdown")
        assert block["type"] == "text"
        assert "# Title" in block["text"]

    def test_sql_file_by_extension(self):
        sql = b"SELECT * FROM users WHERE id = 1;"
        block = file_to_content_block(sql, "query.sql", "")
        assert block["type"] == "text"
        assert "SELECT" in block["text"]

    def test_code_extensions_recognized(self):
        """All code extensions should produce text blocks."""
        for ext in [".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".sh"]:
            data = b"// some code"
            block = file_to_content_block(data, f"file{ext}", "")
            assert block["type"] == "text", f"Extension {ext} not recognized as text"


class TestTextTruncation:
    """Large text files are truncated."""

    def test_text_truncated_at_limit(self):
        big_text = b"x" * (MAX_TEXT_CHARS + 10000)
        block = file_to_content_block(big_text, "big.txt", "text/plain")
        assert block["type"] == "text"
        assert "... [truncated]" in block["text"]
        # Should not contain more than MAX_TEXT_CHARS + wrapper
        assert len(block["text"]) < MAX_TEXT_CHARS + 500


class TestSizeLimit:
    """Files exceeding MAX_FILE_SIZE raise ValueError."""

    def test_too_large_raises(self):
        huge = b"\x00" * (MAX_FILE_SIZE + 1)
        with pytest.raises(ValueError, match="too large"):
            file_to_content_block(huge, "big.bin", "application/octet-stream")

    def test_exactly_at_limit_ok(self):
        data = b"\x00" * MAX_FILE_SIZE
        # Should not raise
        block = file_to_content_block(data, "exact.bin", "application/octet-stream")
        assert block is not None


class TestUnknownFiles:
    """Unknown file types: try text, fallback to binary note."""

    def test_unknown_text_readable(self):
        data = b"some readable text content"
        block = file_to_content_block(data, "unknown.xyz", "application/x-unknown")
        assert block["type"] == "text"
        assert "some readable text content" in block["text"]

    def test_binary_fallback(self):
        data = bytes(range(256)) * 10  # binary data
        block = file_to_content_block(data, "binary.dat", "application/octet-stream")
        assert block["type"] == "text"
        assert "[Binary file attached" in block["text"]
        assert "binary.dat" in block["text"]


class TestFileEmoji:
    """file_to_emoji returns correct emoji descriptions."""

    def test_image_emoji(self):
        block = {"type": "image", "source": {}}
        assert "\U0001f5bc" in file_to_emoji(block, "photo.jpg")

    def test_document_emoji(self):
        block = {"type": "document", "source": {}}
        assert "\U0001f4c4" in file_to_emoji(block, "report.pdf")

    def test_text_emoji(self):
        block = {"type": "text", "text": "some code"}
        assert "\U0001f4ce" in file_to_emoji(block, "code.py")

    def test_binary_emoji(self):
        block = {"type": "text", "text": "[Binary file attached: x.bin]"}
        assert "\U0001f4e6" in file_to_emoji(block, "x.bin")
