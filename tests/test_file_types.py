"""Tests for shared file type detection and text extraction helpers."""

from __future__ import annotations

import zipfile
from io import BytesIO

from liteagent.file_types import detect_file_type, extract_text_from_file


def _zip_bytes(files: dict[str, bytes | str]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name, content in files.items():
            if isinstance(content, str):
                content = content.encode("utf-8")
            archive.writestr(name, content)
    return buf.getvalue()


def make_docx_bytes(text: str = "Hello from DOCX") -> bytes:
    return _zip_bytes({
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8"?>
            <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
              <Override PartName="/word/document.xml"
               ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
            </Types>""",
        "word/document.xml": f"""<?xml version="1.0" encoding="UTF-8"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>
            </w:document>""",
    })


def make_xlsx_bytes(text: str = "Quarterly revenue") -> bytes:
    return _zip_bytes({
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8"?>
            <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
              <Override PartName="/xl/workbook.xml"
               ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
            </Types>""",
        "xl/workbook.xml": """<?xml version="1.0" encoding="UTF-8"?>
            <workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheets><sheet name="Summary" sheetId="1" r:id="rId1"
               xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/></sheets>
            </workbook>""",
        "xl/sharedStrings.xml": f"""<?xml version="1.0" encoding="UTF-8"?>
            <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <si><t>{text}</t></si>
            </sst>""",
        "xl/worksheets/sheet1.xml": """<?xml version="1.0" encoding="UTF-8"?>
            <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
              <sheetData><row r="1"><c r="A1" t="s"><v>0</v></c></row></sheetData>
            </worksheet>""",
    })


def make_pptx_bytes(text: str = "Launch plan") -> bytes:
    return _zip_bytes({
        "[Content_Types].xml": """<?xml version="1.0" encoding="UTF-8"?>
            <Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
              <Override PartName="/ppt/presentation.xml"
               ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
            </Types>""",
        "ppt/presentation.xml": """<?xml version="1.0" encoding="UTF-8"?>
            <p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"/>""",
        "ppt/slides/slide1.xml": f"""<?xml version="1.0" encoding="UTF-8"?>
            <p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                   xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
              <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{text}</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
            </p:sld>""",
    })


class TestFileTypeDetection:
    def test_detect_docx_by_signature_overrides_wrong_mime(self):
        data = make_docx_bytes("Contract draft")
        info = detect_file_type(data, "contract.bin", "application/octet-stream")
        assert info.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert info.category == "document"
        assert info.can_extract_text is True

    def test_detect_xlsx_and_extract_text(self):
        data = make_xlsx_bytes("Quarterly revenue")
        info = detect_file_type(data, "metrics.xlsx", "")
        assert info.category == "spreadsheet"
        assert info.can_extract_text is True
        text = extract_text_from_file(data, info)
        assert "Quarterly revenue" in text

    def test_detect_pptx_and_extract_text(self):
        data = make_pptx_bytes("Launch plan")
        info = detect_file_type(data, "deck.pptx", "")
        assert info.category == "presentation"
        assert info.can_extract_text is True
        text = extract_text_from_file(data, info)
        assert "Launch plan" in text

    def test_legacy_office_is_recognized_and_extracts_strings(self):
        data = (
            b"\xd0\xcf\x11\xe0" + b"\x00" * 32 +
            b"Quarterly planning memo\x00Budget review notes\x00"
        )
        info = detect_file_type(data, "legacy.doc", "")
        assert info.mime_type == "application/msword"
        assert info.category == "document"
        assert info.can_extract_text is True
        text = extract_text_from_file(data, info)
        assert "Quarterly planning memo" in text

    def test_archive_audio_video_and_database_detection(self):
        cases = [
            (b"PK\x03\x04" + b"\x00" * 64, "bundle.zip", "archive", False),
            (b"ID3" + b"\x00" * 64, "song.mp3", "audio", False),
            (b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64, "movie.mp4", "video", False),
            (b"SQLite format 3\x00" + b"\x00" * 64, "data.db", "dataset", False),
        ]
        for data, name, category, extractable in cases:
            info = detect_file_type(data, name, "")
            assert info.category == category
            assert info.can_extract_text is extractable

    def test_svg_is_treated_as_extractable_textual_image(self):
        data = b"<svg><text>System Diagram</text></svg>"
        info = detect_file_type(data, "diagram.svg", "")
        assert info.mime_type == "image/svg+xml"
        assert info.is_image is True
        assert info.can_extract_text is True
        assert "System Diagram" in extract_text_from_file(data, info)
