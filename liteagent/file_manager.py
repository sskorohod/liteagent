"""Centralized file management: auto-upload to S3, indexing, vectorization, search.

Every file that enters the agent (via Telegram, API chat, multimodal upload, download_file tool,
or voice messages) gets stored in S3 and indexed in SQLite with an embedding for search.
"""

import hashlib
import logging
import mimetypes
import os
import pickle
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class FileManager:
    """Central file storage + index.  S3 is the primary store, SQLite is the index."""

    # MIME types that can be full-text indexed via RAG
    _INDEXABLE_TEXT_TYPES = {
        "text/plain", "text/markdown", "text/html", "text/csv", "text/xml",
        "application/json", "application/xml", "application/javascript",
        "application/x-python", "application/pdf",
    }
    _INDEXABLE_EXTENSIONS = {
        ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".json", ".yaml",
        ".yml", ".toml", ".cfg", ".ini", ".csv", ".log", ".rst", ".html",
        ".htm", ".pdf", ".jsx", ".tsx", ".go", ".rs", ".java", ".c", ".cpp",
        ".h", ".hpp", ".sh", ".bash", ".sql", ".xml",
    }

    def __init__(self, storage, db: sqlite3.Connection, embedder=None, rag=None):
        """
        Args:
            storage: StorageBackend instance (S3/MinIO).
            db: Shared SQLite connection (same as memory.db).
            embedder: Optional embedder for file_index description embedding.
            rag: Optional RAGPipeline for full-content indexing.
        """
        self._storage = storage
        self._db = db
        self._embedder = embedder
        self._rag = rag

    # ── Core: ingest a file ──────────────────────────────────

    async def ingest(self, data: bytes, original_name: str, *,
                     source: str = "unknown", user_id: str = "system",
                     mime_type: str = "", description: str = "") -> dict:
        """Store file in S3 and index it.  Returns file metadata dict."""
        if not mime_type:
            mime_type = mimetypes.guess_type(original_name)[0] or "application/octet-stream"

        # Deterministic key: prefix/hash_originalname
        file_hash = hashlib.sha256(data).hexdigest()[:16]
        ext = Path(original_name).suffix
        safe_name = Path(original_name).stem[:60]
        key = f"files/{source}/{file_hash}_{safe_name}{ext}"

        # Upload to S3 (non-blocking)
        await self._storage.async_upload(key, data, content_type=mime_type)

        # Auto-describe: for text-based files, extract first lines
        if not description:
            description = self._auto_describe(data, mime_type, original_name)

        # Embed description for search
        embedding_blob = None
        if self._embedder and description:
            try:
                vec = self._embedder.encode(description)
                embedding_blob = pickle.dumps(vec)
            except Exception as e:
                logger.debug("File embedding failed: %s", e)

        # Upsert into file_index
        self._db.execute("""
            INSERT INTO file_index
                (storage_key, original_name, mime_type, size_bytes,
                 source, user_id, description, embedding, created_at, accessed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ON CONFLICT(storage_key) DO UPDATE SET
                description = excluded.description,
                embedding = excluded.embedding,
                accessed_at = datetime('now')
        """, (key, original_name, mime_type, len(data),
              source, user_id, description, embedding_blob))
        self._db.commit()

        # Full-content indexing via RAG pipeline
        if self._rag and self._should_index(mime_type, original_name, len(data)):
            try:
                text = self._extract_text(data, mime_type)
                if text and len(text.strip()) > 20:
                    ext = Path(original_name).suffix
                    self._rag.index_content(
                        text, source_key=key, source_name=original_name,
                        file_type=ext)
                    logger.debug("Full-text indexed: %s (%d chars)", original_name, len(text))
            except Exception as e:
                logger.debug("Full-text indexing failed for %s: %s", original_name, e)

        logger.info("File ingested: %s → %s (%d bytes, source=%s)",
                     original_name, key, len(data), source)
        return {
            "storage_key": key,
            "original_name": original_name,
            "mime_type": mime_type,
            "size_bytes": len(data),
            "source": source,
            "description": description,
        }

    async def ingest_local(self, local_path: str, *,
                           source: str = "unknown", user_id: str = "system",
                           description: str = "") -> dict:
        """Ingest a local file into S3 + index."""
        p = Path(local_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {local_path}")
        data = p.read_bytes()
        return await self.ingest(
            data, p.name, source=source, user_id=user_id, description=description)

    # ── Search ───────────────────────────────────────────────

    def search(self, query: str, user_id: str = None,
               top_k: int = 10) -> list[dict]:
        """Search files: combines metadata search with RAG content search.

        Two-pass approach:
        1. File metadata search (name, description) — from file_index
        2. Content search via RAG (full text of indexed files) — from rag_chunks
        Results are merged by storage_key, best score wins.
        """
        file_scores: dict[str, dict] = {}

        # ── Pass 1: Metadata search (file_index table) ──
        self._metadata_search(query, user_id, file_scores)

        # ── Pass 2: RAG content search (rag_chunks via hybrid search) ──
        if self._rag:
            self._rag_content_search(query, file_scores)

        if not file_scores:
            return []

        results = sorted(file_scores.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _metadata_search(self, query: str, user_id: str | None,
                         file_scores: dict[str, dict]):
        """Search file_index by name/description (embedding + keyword)."""
        where = "1=1"
        params: list = []
        if user_id:
            where = "user_id = ?"
            params.append(user_id)

        rows = self._db.execute(f"""
            SELECT id, storage_key, original_name, mime_type, size_bytes,
                   source, user_id, description, embedding, created_at
            FROM file_index WHERE {where}
            ORDER BY created_at DESC LIMIT 500
        """, params).fetchall()

        if not rows:
            return

        query_vec = None
        if self._embedder:
            try:
                query_vec = self._embedder.encode(query)
            except Exception:
                pass

        query_lower = query.lower()
        query_words = set(query_lower.split())

        for row in rows:
            (rid, key, name, mime, size, src, uid,
             desc, emb_blob, created) = row
            score = 0.0

            if query_vec is not None and emb_blob:
                try:
                    import numpy as np
                    emb = pickle.loads(emb_blob)
                    cos = float(np.dot(query_vec, emb) /
                                (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-9))
                    score = max(cos, 0.0)
                except Exception:
                    pass

            text = f"{name} {desc}".lower()
            matched = sum(1 for w in query_words if w in text)
            if matched:
                score += 0.3 * (matched / max(len(query_words), 1))

            if score > 0.05:
                file_scores[key] = {
                    "storage_key": key,
                    "original_name": name,
                    "mime_type": mime,
                    "size_bytes": size,
                    "source": src,
                    "user_id": uid,
                    "description": desc,
                    "created_at": created,
                    "score": round(score, 3),
                }

    def _rag_content_search(self, query: str, file_scores: dict[str, dict]):
        """Search full file content via RAG pipeline, merge results by storage_key."""
        try:
            rag_results = self._rag.search(query, top_k=20)
        except Exception as e:
            logger.debug("RAG content search failed: %s", e)
            return

        for hit in rag_results:
            # RAG returns doc_path as source — for S3 files this is the storage_key
            source_key = hit.get("source", "")
            if not source_key.startswith("files/"):
                continue  # Skip non-file RAG docs (local files, etc.)

            rag_score = hit.get("score", 0.0)
            if source_key in file_scores:
                # Boost existing entry
                existing = file_scores[source_key]
                existing["score"] = round(max(existing["score"], rag_score), 3)
                if hit.get("content"):
                    existing["_rag_snippet"] = hit["content"][:200]
            else:
                # Look up file metadata from file_index
                row = self._db.execute(
                    "SELECT original_name, mime_type, size_bytes, source, user_id, "
                    "description, created_at FROM file_index WHERE storage_key = ?",
                    (source_key,)).fetchone()
                if row:
                    file_scores[source_key] = {
                        "storage_key": source_key,
                        "original_name": row[0],
                        "mime_type": row[1],
                        "size_bytes": row[2],
                        "source": row[3],
                        "user_id": row[4],
                        "description": row[5],
                        "created_at": row[6],
                        "score": round(rag_score, 3),
                        "_rag_snippet": hit.get("content", "")[:200],
                    }

    # ── List all files ───────────────────────────────────────

    def list_files(self, user_id: str = None, source: str = None,
                   limit: int = 100, offset: int = 0) -> list[dict]:
        """List indexed files with optional filters."""
        conditions = []
        params: list = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if source:
            conditions.append("source = ?")
            params.append(source)
        where = " AND ".join(conditions) if conditions else "1=1"

        rows = self._db.execute(f"""
            SELECT storage_key, original_name, mime_type, size_bytes,
                   source, user_id, description, created_at
            FROM file_index WHERE {where}
            ORDER BY created_at DESC LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()

        return [
            {
                "storage_key": r[0], "original_name": r[1],
                "mime_type": r[2], "size_bytes": r[3],
                "source": r[4], "user_id": r[5],
                "description": r[6], "created_at": r[7],
            }
            for r in rows
        ]

    def count_files(self, user_id: str = None) -> int:
        where = "1=1"
        params: list = []
        if user_id:
            where = "user_id = ?"
            params.append(user_id)
        row = self._db.execute(
            f"SELECT COUNT(*) FROM file_index WHERE {where}", params).fetchone()
        return row[0] if row else 0

    # ── Get presigned URL ────────────────────────────────────

    async def get_download_url(self, storage_key: str,
                               expires: int = 3600) -> str:
        """Get a presigned download URL for a file."""
        # Touch accessed_at
        self._db.execute(
            "UPDATE file_index SET accessed_at = datetime('now') WHERE storage_key = ?",
            (storage_key,))
        self._db.commit()
        return await self._storage.async_get_url(storage_key, expires)

    # ── Cleanup (propose + confirm) ──────────────────────────

    def propose_cleanup(self, days_unused: int = 30,
                        max_items: int = 50) -> list[dict]:
        """Find files not accessed for N days. Returns candidates (NOT deleted)."""
        rows = self._db.execute("""
            SELECT storage_key, original_name, mime_type, size_bytes,
                   source, user_id, description, created_at, accessed_at
            FROM file_index
            WHERE accessed_at < datetime('now', ? || ' days')
            ORDER BY accessed_at ASC LIMIT ?
        """, (f"-{days_unused}", max_items)).fetchall()

        return [
            {
                "storage_key": r[0], "original_name": r[1],
                "mime_type": r[2], "size_bytes": r[3],
                "source": r[4], "user_id": r[5],
                "description": r[6], "created_at": r[7],
                "accessed_at": r[8],
            }
            for r in rows
        ]

    async def confirm_cleanup(self, keys_to_delete: list[str]) -> dict:
        """Actually delete files after user confirmation."""
        result = await self._storage.async_delete_many(keys_to_delete)
        # Remove from index
        for key in result.get("deleted", []):
            self._db.execute(
                "DELETE FROM file_index WHERE storage_key = ?", (key,))
        self._db.commit()
        logger.info("Cleanup: deleted %d files, %d errors",
                     len(result["deleted"]), len(result["errors"]))
        return result

    # ── Internal helpers ─────────────────────────────────────

    def _should_index(self, mime_type: str, name: str, size: int) -> bool:
        """Check if file should be full-text indexed via RAG."""
        max_size = 10 * 1024 * 1024  # 10 MB default
        if size > max_size:
            return False
        ext = Path(name).suffix.lower()
        if ext in self._INDEXABLE_EXTENSIONS:
            return True
        if mime_type in self._INDEXABLE_TEXT_TYPES:
            return True
        if mime_type.startswith("text/"):
            return True
        return False

    @staticmethod
    def _extract_text(data: bytes, mime_type: str) -> str:
        """Extract text content from file data for RAG indexing."""
        if mime_type.startswith("text/") or mime_type in (
            "application/json", "application/xml",
            "application/javascript", "application/x-python",
        ):
            return data.decode("utf-8", errors="replace")

        if mime_type == "application/pdf":
            try:
                import fitz
                doc = fitz.open(stream=data, filetype="pdf")
                text = "\n\n".join(page.get_text() for page in doc)
                doc.close()
                return text
            except ImportError:
                return ""
            except Exception:
                return ""

        # Fallback: try decoding as text
        try:
            return data.decode("utf-8", errors="replace")
        except Exception:
            return ""

    @staticmethod
    def _auto_describe(data: bytes, mime_type: str, name: str) -> str:
        """Generate a short description from file content."""
        parts = [name]

        # Text-based files: extract first lines
        if mime_type.startswith("text/") or mime_type in (
            "application/json", "application/xml",
            "application/javascript", "application/x-python",
        ):
            try:
                text = data[:2000].decode("utf-8", errors="replace")
                first_lines = text.strip().split("\n")[:5]
                parts.append(" ".join(first_lines)[:300])
            except Exception:
                pass

        # PDF: try extracting text from first page
        elif mime_type == "application/pdf":
            try:
                import fitz
                doc = fitz.open(stream=data, filetype="pdf")
                if doc.page_count > 0:
                    page_text = doc[0].get_text()[:500]
                    parts.append(page_text.strip()[:300])
                doc.close()
            except Exception:
                pass

        # Images: include dimensions if possible
        elif mime_type.startswith("image/"):
            try:
                from io import BytesIO
                # Try PIL for dimensions
                from PIL import Image
                img = Image.open(BytesIO(data))
                parts.append(f"image {img.width}x{img.height}")
            except Exception:
                parts.append("image file")

        return " | ".join(parts)


def create_file_manager(agent) -> "FileManager | None":
    """Create FileManager if storage is enabled. Called from agent.__init__."""
    storage = getattr(agent, "_storage", None)
    if not storage:
        return None
    db = agent.memory.db
    embedder = agent.memory._embedder
    rag = getattr(agent, "_rag", None)
    return FileManager(storage, db, embedder, rag=rag)
