"""Shared file type detection and lightweight text extraction helpers."""

from __future__ import annotations

import html
import mimetypes
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree

TEXT_EXTENSIONS = {
    ".txt", ".text", ".md", ".markdown", ".rst", ".rtf",
    ".csv", ".tsv", ".json", ".jsonl", ".xml", ".yaml", ".yml",
    ".html", ".htm", ".svg", ".log", ".ini", ".toml", ".env", ".cfg",
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".cc",
    ".h", ".hpp", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".cs",
    ".sh", ".bash", ".zsh", ".fish", ".sql", ".r", ".lua", ".css",
    ".scss", ".less", ".vue", ".svelte",
}

IMAGE_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp",
    "image/tiff", "image/svg+xml", "image/x-icon", "image/heic",
    "image/heif", "image/avif",
}

TEXT_MIME_TYPES = {
    "text/plain", "text/markdown", "text/html", "text/csv", "text/xml",
    "application/json", "application/xml", "application/javascript",
    "text/javascript", "application/x-python", "text/x-python",
    "application/x-shellscript", "application/x-sh",
    "application/x-yaml", "text/yaml",
}

EXTENSION_MIME_MAP = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/plain",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".xml": "application/xml",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".html": "text/html",
    ".htm": "text/html",
    ".svg": "image/svg+xml",
    ".log": "text/plain",
    ".ini": "text/plain",
    ".toml": "text/plain",
    ".env": "text/plain",
    ".cfg": "text/plain",
    ".py": "text/x-python",
    ".js": "application/javascript",
    ".ts": "application/typescript",
    ".jsx": "text/jsx",
    ".tsx": "text/tsx",
    ".sql": "application/sql",
    ".sh": "application/x-shellscript",
    ".bash": "application/x-shellscript",
    ".zsh": "application/x-shellscript",
    ".css": "text/css",
    ".scss": "text/x-scss",
    ".less": "text/x-less",
    ".vue": "text/plain",
    ".svelte": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".docm": "application/vnd.ms-word.document.macroenabled.12",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xlsm": "application/vnd.ms-excel.sheet.macroenabled.12",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".pptm": "application/vnd.ms-powerpoint.presentation.macroenabled.12",
    ".doc": "application/msword",
    ".xls": "application/vnd.ms-excel",
    ".ppt": "application/vnd.ms-powerpoint",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odp": "application/vnd.oasis.opendocument.presentation",
    ".rtf": "application/rtf",
    ".epub": "application/epub+zip",
    ".zip": "application/zip",
    ".tar": "application/x-tar",
    ".gz": "application/gzip",
    ".bz2": "application/x-bzip2",
    ".xz": "application/x-xz",
    ".7z": "application/x-7z-compressed",
    ".rar": "application/vnd.rar",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".sqlite": "application/vnd.sqlite3",
    ".sqlite3": "application/vnd.sqlite3",
    ".db": "application/vnd.sqlite3",
    ".parquet": "application/x-parquet",
}

MIME_ALIASES = {
    "application/x-pdf": "application/pdf",
    "application/x-zip-compressed": "application/zip",
    "application/zip-compressed": "application/zip",
    "application/x-rar-compressed": "application/vnd.rar",
    "audio/mp3": "audio/mpeg",
    "application/x-gzip": "application/gzip",
    "application/x-wav": "audio/wav",
}

_GENERIC_BINARY_MIME_TYPES = {
    "application/octet-stream",
    "binary/octet-stream",
    "application/x-binary",
    "application/x-unknown",
    "application/unknown",
}


@dataclass(frozen=True)
class FileTypeInfo:
    mime_type: str
    extension: str
    category: str
    label: str

    @property
    def is_image(self) -> bool:
        return self.category == "image"

    @property
    def is_pdf(self) -> bool:
        return self.category == "pdf"

    @property
    def is_textual(self) -> bool:
        return self.category in {"text", "code", "document", "spreadsheet", "presentation", "ebook"}

    @property
    def can_extract_text(self) -> bool:
        if self.is_pdf:
            return True
        if self.mime_type.startswith("text/") or self.mime_type in TEXT_MIME_TYPES:
            return True
        if self.extension in TEXT_EXTENSIONS or self.extension in CODE_EXTENSIONS:
            return True
        if self.mime_type in _EXTRACTABLE_MIME_TYPES:
            return True
        if self.extension in _EXTRACTABLE_EXTENSIONS:
            return True
        return False


def normalize_mime_type(mime_type: str = "") -> str:
    raw = str(mime_type or "").strip().lower()
    if not raw:
        return ""
    raw = raw.split(";", 1)[0].strip()
    return MIME_ALIASES.get(raw, raw)


def detect_file_type(data: bytes, filename: str = "", mime_type: str = "") -> FileTypeInfo:
    ext = Path(filename or "").suffix.lower()
    normalized_mime = normalize_mime_type(mime_type)
    signature = _detect_signature(data, ext)
    inferred_mime = signature or normalized_mime or EXTENSION_MIME_MAP.get(ext) or _guess_mime_from_name(filename)

    if _looks_like_text(data) and (not inferred_mime or inferred_mime in _GENERIC_BINARY_MIME_TYPES):
        inferred_mime = "text/plain"

    if not inferred_mime:
        inferred_mime = "application/octet-stream"

    if inferred_mime == "application/zip" and ext in EXTENSION_MIME_MAP:
        inferred_mime = EXTENSION_MIME_MAP[ext]

    category, label = _categorize(inferred_mime, ext)
    return FileTypeInfo(
        mime_type=inferred_mime,
        extension=ext,
        category=category,
        label=label,
    )


def decode_text_bytes(data: bytes) -> str:
    if not data:
        return ""
    encodings = ["utf-8-sig", "utf-16", "utf-32", "cp1251", "latin-1"]
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def extract_text_from_file(data: bytes, file_info: FileTypeInfo) -> str:
    mime = file_info.mime_type
    ext = file_info.extension

    if mime.startswith("text/") or mime in TEXT_MIME_TYPES or ext in TEXT_EXTENSIONS or ext in CODE_EXTENSIONS:
        return decode_text_bytes(data)

    if file_info.is_pdf:
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception:
            return ""

    if ext == ".docx" or mime.endswith("wordprocessingml.document"):
        return _extract_docx_text(data)
    if ext == ".doc" or mime == "application/msword":
        return _extract_legacy_office_text(data, ext)
    if ext in {".xlsx", ".xlsm"} or mime.endswith("spreadsheetml.sheet") or mime.endswith("sheet.macroenabled.12"):
        return _extract_xlsx_text(data)
    if ext == ".xls" or mime == "application/vnd.ms-excel":
        return _extract_legacy_office_text(data, ext)
    if ext in {".pptx", ".pptm"} or mime.endswith("presentationml.presentation") or mime.endswith("presentation.macroenabled.12"):
        return _extract_pptx_text(data)
    if ext == ".ppt" or mime == "application/vnd.ms-powerpoint":
        return _extract_legacy_office_text(data, ext)
    if ext in {".odt", ".ods", ".odp"} or mime.startswith("application/vnd.oasis.opendocument"):
        return _extract_odf_text(data)
    if ext == ".epub" or mime == "application/epub+zip":
        return _extract_epub_text(data)
    if ext == ".rtf" or mime == "application/rtf":
        return _extract_rtf_text(data)

    if _looks_like_text(data):
        return decode_text_bytes(data)
    return ""


def _guess_mime_from_name(filename: str) -> str:
    guessed = mimetypes.guess_type(filename or "")[0]
    return normalize_mime_type(guessed or "")


def _categorize(mime: str, ext: str) -> tuple[str, str]:
    if mime in IMAGE_MIME_TYPES:
        return "image", "image"
    if mime == "application/pdf":
        return "pdf", "PDF document"
    if mime in {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
        "application/vnd.oasis.opendocument.text",
        "application/rtf",
    }:
        return "document", "document"
    if mime in {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel.sheet.macroenabled.12",
        "application/vnd.ms-excel",
        "application/vnd.oasis.opendocument.spreadsheet",
    }:
        return "spreadsheet", "spreadsheet"
    if mime in {
        "application/x-parquet",
        "application/vnd.sqlite3",
    }:
        return "dataset", "dataset"
    if mime in {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-powerpoint.presentation.macroenabled.12",
        "application/vnd.ms-powerpoint",
        "application/vnd.oasis.opendocument.presentation",
    }:
        return "presentation", "presentation"
    if mime == "application/epub+zip":
        return "ebook", "ebook"
    if mime.startswith("audio/"):
        return "audio", "audio"
    if mime.startswith("video/"):
        return "video", "video"
    if mime in {
        "application/zip", "application/x-tar", "application/gzip",
        "application/x-bzip2", "application/x-xz", "application/x-7z-compressed",
        "application/vnd.rar",
    }:
        return "archive", "archive"
    if mime.startswith("text/") or mime in TEXT_MIME_TYPES or ext in TEXT_EXTENSIONS:
        return "text", "text"
    if ext in CODE_EXTENSIONS:
        return "code", "code"
    return "binary", "binary"


def _detect_signature(data: bytes, ext: str) -> str:
    header = data[:512]
    if header.startswith(b"%PDF-"):
        return "application/pdf"
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if header.startswith(b"BM"):
        return "image/bmp"
    if header[:4] in {b"II*\x00", b"MM\x00*"}:
        return "image/tiff"
    if header.startswith(b"\x00\x00\x01\x00"):
        return "image/x-icon"
    if len(header) > 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    if len(header) > 12 and header[4:8] == b"ftyp":
        brand = header[8:16].decode("latin-1", errors="ignore").lower()
        if brand.startswith(("heic", "heix", "hevc", "hevx", "mif1", "msf1")):
            return "image/heic"
        if brand.startswith(("avif", "avis")):
            return "image/avif"
        if brand.startswith(("m4a", "m4b", "m4p")):
            return "audio/mp4"
        if brand.startswith(("qt  ",)):
            return "video/quicktime"
        return "video/mp4"
    if header.startswith(b"ID3") or (len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if header.startswith(b"OggS"):
        return "audio/ogg"
    if header.startswith(b"fLaC"):
        return "audio/flac"
    if len(header) > 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "audio/wav"
    if len(header) > 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI ":
        return "video/x-msvideo"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        if ext == ".webm":
            return "video/webm"
        return "video/x-matroska"
    if header.startswith(b"PK\x03\x04") or header.startswith(b"PK\x05\x06") or header.startswith(b"PK\x07\x08"):
        return _detect_zip_family(data, ext)
    if header.startswith(b"Rar!\x1a\x07"):
        return "application/vnd.rar"
    if header.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "application/x-7z-compressed"
    if header.startswith(b"\x1f\x8b"):
        return "application/gzip"
    if header.startswith(b"BZh"):
        return "application/x-bzip2"
    if header.startswith(b"\xfd7zXZ\x00"):
        return "application/x-xz"
    if len(data) > 262 and data[257:262] == b"ustar":
        return "application/x-tar"
    if header.startswith(b"SQLite format 3\x00"):
        return "application/vnd.sqlite3"
    if header.startswith(b"PAR1"):
        return "application/x-parquet"
    if header.startswith(b"\xd0\xcf\x11\xe0"):
        if ext == ".doc":
            return "application/msword"
        if ext == ".xls":
            return "application/vnd.ms-excel"
        if ext == ".ppt":
            return "application/vnd.ms-powerpoint"
        return "application/x-ole-storage"
    if _looks_like_svg(data):
        return "image/svg+xml"
    if _looks_like_json(data):
        return "application/json"
    if _looks_like_xml(data):
        return "application/xml"
    if _looks_like_html(data):
        return "text/html"
    return ""


def _detect_zip_family(data: bytes, ext: str) -> str:
    ext_mime = EXTENSION_MIME_MAP.get(ext)
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = set(archive.namelist())
            if "word/document.xml" in names:
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if "xl/workbook.xml" in names:
                return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if "ppt/presentation.xml" in names:
                return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            if "mimetype" in names:
                mimetype_text = archive.read("mimetype").decode("utf-8", errors="ignore").strip()
                if mimetype_text:
                    return normalize_mime_type(mimetype_text)
            if "META-INF/container.xml" in names:
                return "application/epub+zip"
    except Exception:
        pass
    return ext_mime or "application/zip"


def _looks_like_text(data: bytes) -> bool:
    if not data:
        return True
    sample = data[:4096]
    if b"\x00" in sample:
        # UTF-16 is allowed if BOM exists.
        if sample.startswith((b"\xff\xfe", b"\xfe\xff", b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
            return True
        return False
    text_chars = 0
    for byte in sample:
        if byte in (9, 10, 13) or 32 <= byte <= 126 or byte >= 128:
            text_chars += 1
    return (text_chars / max(len(sample), 1)) >= 0.92


def _looks_like_svg(data: bytes) -> bool:
    text = decode_text_bytes(data[:512]).lstrip().lower()
    return text.startswith("<svg") or "<svg" in text


def _looks_like_json(data: bytes) -> bool:
    text = decode_text_bytes(data[:512]).lstrip()
    return text.startswith("{") or text.startswith("[")


def _looks_like_xml(data: bytes) -> bool:
    text = decode_text_bytes(data[:512]).lstrip().lower()
    return text.startswith("<?xml")


def _looks_like_html(data: bytes) -> bool:
    text = decode_text_bytes(data[:512]).lstrip().lower()
    return text.startswith("<!doctype html") or text.startswith("<html") or "<body" in text


def _extract_docx_text(data: bytes) -> str:
    return _extract_zip_xml_text(
        data,
        prefixes=("word/document", "word/header", "word/footer", "word/comments", "word/footnotes", "word/endnotes"),
    )


def _extract_pptx_text(data: bytes) -> str:
    return _extract_zip_xml_text(data, prefixes=("ppt/slides/slide", "ppt/notesSlides/notesSlide"))


def _extract_xlsx_text(data: bytes) -> str:
    texts = []
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if "xl/workbook.xml" in archive.namelist():
                workbook = archive.read("xl/workbook.xml")
                texts.extend(_extract_xml_text(workbook))
            if "xl/sharedStrings.xml" in archive.namelist():
                shared = archive.read("xl/sharedStrings.xml")
                texts.extend(_extract_xml_text(shared))
            for name in archive.namelist():
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
                    texts.extend(_extract_xml_text(archive.read(name)))
    except Exception:
        return ""
    return "\n".join(t for t in texts if t).strip()


def _extract_odf_text(data: bytes) -> str:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            if "content.xml" not in archive.namelist():
                return ""
            return "\n".join(_extract_xml_text(archive.read("content.xml"))).strip()
    except Exception:
        return ""


def _extract_epub_text(data: bytes) -> str:
    texts = []
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            for name in archive.namelist():
                lowered = name.lower()
                if lowered.endswith((".xhtml", ".html", ".htm", ".opf", ".ncx")):
                    texts.append(_strip_html_tags(decode_text_bytes(archive.read(name))))
    except Exception:
        return ""
    return "\n\n".join(t.strip() for t in texts if t.strip()).strip()


def _extract_rtf_text(data: bytes) -> str:
    text = decode_text_bytes(data)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_zip_xml_text(data: bytes, prefixes: tuple[str, ...]) -> str:
    texts = []
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            for name in archive.namelist():
                if any(name.startswith(prefix) and name.endswith(".xml") for prefix in prefixes):
                    texts.extend(_extract_xml_text(archive.read(name)))
    except Exception:
        return ""
    return "\n".join(t for t in texts if t).strip()


def _extract_legacy_office_text(data: bytes, ext: str) -> str:
    commands_by_ext = {
        ".doc": [["antiword"], ["catdoc"]],
        ".xls": [["xls2csv"], ["catxls"]],
        ".ppt": [["catppt"]],
    }
    for base_cmd in commands_by_ext.get(ext, []):
        text = _extract_with_external_command(data, ext, base_cmd)
        if text:
            return text
    return _extract_printable_strings(data)


def _extract_with_external_command(data: bytes, ext: str, base_cmd: list[str]) -> str:
    if not base_cmd or not shutil.which(base_cmd[0]):
        return ""
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        proc = subprocess.run(
            [*base_cmd, tmp_path],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode != 0:
            return ""
        return _clean_extracted_text(proc.stdout or "")
    except Exception:
        return ""
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def _extract_printable_strings(data: bytes) -> str:
    chunks: list[str] = []

    ascii_strings = re.findall(rb"[\x20-\x7e\t\r\n]{4,}", data)
    for chunk in ascii_strings:
        text = chunk.decode("latin-1", errors="ignore").strip()
        if _looks_like_meaningful_text(text):
            chunks.append(text)

    for encoding in ("utf-16-le", "utf-16-be"):
        try:
            decoded = data.decode(encoding, errors="ignore")
        except Exception:
            continue
        for chunk in re.findall(r"[ -~\t\r\n]{4,}", decoded):
            text = chunk.strip()
            if _looks_like_meaningful_text(text):
                chunks.append(text)

    return _clean_extracted_text("\n".join(_dedupe_preserve_order(chunks)))


def _looks_like_meaningful_text(text: str) -> bool:
    if len(text) < 4:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= max(3, len(text) // 4)


def _clean_extracted_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if len(line) > 500:
            line = line[:500]
        if _looks_like_meaningful_text(line):
            lines.append(line)
    return "\n".join(_dedupe_preserve_order(lines)).strip()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _extract_xml_text(data: bytes) -> list[str]:
    try:
        root = ElementTree.fromstring(data)
    except Exception:
        text = decode_text_bytes(data)
        return [html.unescape(t).strip() for t in re.findall(r">([^<>]+)<", text) if t.strip()]

    values: list[str] = []
    for elem in root.iter():
        text = (elem.text or "").strip()
        if text:
            values.append(html.unescape(text))
    return values


def _strip_html_tags(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_EXTRACTABLE_MIME_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-word.document.macroenabled.12",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.macroenabled.12",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint.presentation.macroenabled.12",
    "application/vnd.ms-powerpoint",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.presentation",
    "application/rtf",
    "application/epub+zip",
    "image/svg+xml",
}

_EXTRACTABLE_EXTENSIONS = {
    ".docx", ".docm", ".xlsx", ".xlsm", ".pptx", ".pptm",
    ".doc", ".xls", ".ppt",
    ".odt", ".ods", ".odp", ".rtf", ".epub", ".svg",
}
