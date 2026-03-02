"""RAG pipeline — document ingestion, chunking, and retrieval via Qdrant + SQLite metadata."""

import hashlib
import logging
import math
import pickle
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Document ingestion and retrieval using Qdrant vector DB with SQLite metadata."""

    def __init__(self, db: sqlite3.Connection, embedder=None, config: dict = None,
                 sandbox_root: str | None = None):
        self.db = db
        self._embedder = embedder
        self._sandbox_root = sandbox_root
        cfg = config or {}
        self.chunk_size = cfg.get("chunk_size", 500)
        self.chunk_overlap = cfg.get("overlap", 50)
        self.top_k = cfg.get("top_k", 5)

        # Qdrant setup
        self._qdrant = None
        self._collection = cfg.get("collection", "liteagent_rag")
        qdrant_cfg = cfg.get("qdrant", {})
        if qdrant_cfg.get("url"):
            self._init_qdrant(qdrant_cfg)

        self._init_tables()

    def _init_qdrant(self, qdrant_cfg: dict):
        """Initialize Qdrant client."""
        try:
            from qdrant_client import QdrantClient
            url = qdrant_cfg.get("url", "http://localhost:6333")
            api_key = qdrant_cfg.get("api_key")
            if not api_key:
                from .config import get_api_key
                api_key = get_api_key("qdrant")
            self._qdrant = QdrantClient(url=url, api_key=api_key, timeout=10)
            self._collection = qdrant_cfg.get("collection", self._collection)
            self._ensure_collection()
            logger.info("Qdrant connected: %s, collection: %s", url, self._collection)
        except ImportError:
            logger.debug("qdrant-client not installed, using keyword fallback")
        except Exception as e:
            logger.warning("Qdrant connection failed: %s — using fallback", e)

    def _ensure_collection(self):
        """Create Qdrant collection if not exists."""
        try:
            from qdrant_client.models import Distance, VectorParams
            collections = [c.name for c in self._qdrant.get_collections().collections]
            if self._collection not in collections:
                # all-MiniLM-L6-v2 produces 384-dim vectors
                vec_size = 384
                if self._embedder:
                    try:
                        test_vec = self._embedder.encode("test")
                        vec_size = len(test_vec)
                    except Exception:
                        pass
                self._qdrant.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
                )
                logger.info("Created Qdrant collection: %s (dim=%d)", self._collection, vec_size)
        except Exception as e:
            logger.warning("Could not ensure Qdrant collection: %s", e)
            self._qdrant = None

    def _init_tables(self):
        """SQLite metadata tables (vectors stored in Qdrant)."""
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                name TEXT,
                doc_hash TEXT,
                chunk_count INTEGER DEFAULT 0,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                qdrant_id TEXT,
                FOREIGN KEY (doc_id) REFERENCES rag_documents(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_rag_chunks_doc ON rag_chunks(doc_id);
        """)
        # Migration: add qdrant_id column if missing
        try:
            self.db.execute("ALTER TABLE rag_chunks ADD COLUMN qdrant_id TEXT")
        except sqlite3.OperationalError:
            pass
        self.db.commit()

    # ═══════════════════════════════════════
    # DOCUMENT LOADERS
    # ═══════════════════════════════════════

    def _validate_sandbox(self, path: Path) -> None:
        """Check that path is within sandbox_root (if configured)."""
        if not self._sandbox_root:
            return
        import os
        root = os.path.realpath(os.path.expanduser(self._sandbox_root))
        resolved = str(path.resolve())
        if not resolved.startswith(root + os.sep) and resolved != root:
            raise PermissionError(
                f"RAG access denied: path '{path}' is outside sandbox '{root}'"
            )

    def load_file(self, path: str) -> str:
        """Load document content from file. Supports txt, md, html, pdf."""
        p = Path(path).expanduser().resolve()
        self._validate_sandbox(p)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = p.suffix.lower()
        if suffix in (".txt", ".md", ".markdown", ".py", ".js", ".json", ".yaml", ".yml",
                       ".toml", ".cfg", ".ini", ".csv", ".log", ".rst"):
            return p.read_text(encoding="utf-8", errors="replace")
        elif suffix in (".html", ".htm"):
            return self._strip_html(p.read_text(encoding="utf-8", errors="replace"))
        elif suffix == ".pdf":
            return self._load_pdf(p)
        else:
            return p.read_text(encoding="utf-8", errors="replace")

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags using stdlib."""
        import re
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.S | re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                              ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
            text = text.replace(entity, char)
        return text

    @staticmethod
    def _load_pdf(path: Path) -> str:
        """Load PDF (requires optional pymupdf)."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            return "\n\n".join(page.get_text() for page in doc)
        except ImportError:
            raise ImportError("PDF support requires pymupdf: pip install liteagent[pdf]")

    # ═══════════════════════════════════════
    # CHUNKING
    # ═══════════════════════════════════════

    def chunk_text(self, text: str) -> list[str]:
        """Recursive text splitter with overlap."""
        if not text or not text.strip():
            return []
        separators = ["\n\n", "\n", ". ", " "]
        return self._recursive_split(text, separators, self.chunk_size, self.chunk_overlap)

    def _recursive_split(self, text: str, separators: list[str],
                          chunk_size: int, overlap: int) -> list[str]:
        """Split text recursively by trying separators in order."""
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []

        sep = separators[0] if separators else " "
        remaining_seps = separators[1:] if len(separators) > 1 else []
        parts = text.split(sep)

        chunks = []
        current = ""
        for part in parts:
            candidate = (current + sep + part).strip() if current else part.strip()
            if len(candidate) > chunk_size and current:
                chunks.append(current.strip())
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + sep + part
                else:
                    current = part
            else:
                current = candidate

        if current.strip():
            if len(current) > chunk_size and remaining_seps:
                chunks.extend(self._recursive_split(
                    current, remaining_seps, chunk_size, overlap))
            else:
                chunks.append(current.strip())

        return [c for c in chunks if c]

    # ═══════════════════════════════════════
    # INGESTION
    # ═══════════════════════════════════════

    def ingest(self, path: str) -> dict:
        """Ingest a file or directory. Returns stats dict."""
        p = Path(path).expanduser().resolve()
        self._validate_sandbox(p)
        if p.is_dir():
            return self._ingest_directory(p)
        return self._ingest_file(p)

    def _ingest_file(self, path: Path) -> dict:
        """Ingest a single file: chunk → embed → store in Qdrant + SQLite metadata."""
        content = self.load_file(str(path))
        doc_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if already ingested with same hash
        existing = self.db.execute(
            "SELECT id, doc_hash FROM rag_documents WHERE path = ?",
            (str(path),)).fetchone()
        if existing and existing[1] == doc_hash:
            return {"path": str(path), "status": "unchanged", "chunks": 0}

        # Delete old data if re-ingesting
        if existing:
            self._delete_doc_vectors(existing[0])
            self.db.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (existing[0],))
            self.db.execute("DELETE FROM rag_documents WHERE id = ?", (existing[0],))

        chunks = self.chunk_text(content)

        self.db.execute(
            "INSERT INTO rag_documents (path, name, doc_hash, chunk_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(path), path.name, doc_hash, len(chunks), datetime.now().isoformat()))
        doc_id = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]

        # Prepare vectors for Qdrant batch upsert
        qdrant_points = []
        for i, chunk in enumerate(chunks):
            embedding_blob = None
            vector = None
            qdrant_id = str(uuid.uuid4())

            if self._embedder:
                vec = self._embedder.encode(chunk)
                vector = vec.tolist()
                embedding_blob = pickle.dumps(vec)

            self.db.execute(
                "INSERT INTO rag_chunks (doc_id, chunk_index, content, embedding, qdrant_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, i, chunk, embedding_blob, qdrant_id))

            if vector and self._qdrant:
                qdrant_points.append({
                    "id": qdrant_id,
                    "vector": vector,
                    "payload": {
                        "doc_name": path.name,
                        "doc_path": str(path),
                        "chunk_index": i,
                        "content": chunk[:1000],
                    },
                })

        # Batch upsert to Qdrant
        if qdrant_points and self._qdrant:
            try:
                from qdrant_client.models import PointStruct
                points = [PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                          for p in qdrant_points]
                self._qdrant.upsert(collection_name=self._collection, points=points)
                logger.debug("Upserted %d vectors to Qdrant for %s", len(points), path.name)
            except Exception as e:
                logger.warning("Qdrant upsert failed: %s", e)

        self.db.commit()
        return {"path": str(path), "status": "ingested", "chunks": len(chunks)}

    def _delete_doc_vectors(self, doc_id: int):
        """Delete Qdrant vectors for a document."""
        if not self._qdrant:
            return
        rows = self.db.execute(
            "SELECT qdrant_id FROM rag_chunks WHERE doc_id = ? AND qdrant_id IS NOT NULL",
            (doc_id,)).fetchall()
        if rows:
            ids = [r[0] for r in rows]
            try:
                from qdrant_client.models import PointIdsList
                self._qdrant.delete(
                    collection_name=self._collection,
                    points_selector=PointIdsList(points=ids))
            except Exception as e:
                logger.warning("Qdrant delete failed: %s", e)

    def _ingest_directory(self, path: Path) -> dict:
        """Ingest all supported files in a directory."""
        supported = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf",
                     ".py", ".js", ".json", ".yaml", ".yml", ".rst"}
        results = {"files": 0, "chunks": 0, "errors": []}
        for f in sorted(path.rglob("*")):
            if f.suffix.lower() in supported and f.is_file():
                try:
                    r = self._ingest_file(f)
                    if r["status"] != "unchanged":
                        results["files"] += 1
                        results["chunks"] += r["chunks"]
                except Exception as e:
                    results["errors"].append(f"{f.name}: {e}")
        return results

    # ═══════════════════════════════════════
    # RETRIEVAL
    # ═══════════════════════════════════════

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Search: Qdrant vector search → fallback to SQLite/keyword."""
        k = top_k or self.top_k

        # Try Qdrant first
        if self._qdrant and self._embedder:
            try:
                return self._qdrant_search(query, k)
            except Exception as e:
                logger.warning("Qdrant search failed, falling back: %s", e)

        # Fallback: SQLite embedding search or keyword search
        if self._embedder:
            return self._sqlite_embedding_search(query, k)
        return self._keyword_search(query, k)

    def _qdrant_search(self, query: str, top_k: int) -> list[dict]:
        """Vector search via Qdrant."""
        query_vec = self._embedder.encode(query).tolist()
        results = self._qdrant.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=top_k,
        )
        return [
            {
                "content": hit.payload.get("content", ""),
                "source": hit.payload.get("doc_name", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "score": round(hit.score, 4),
            }
            for hit in results.points
        ]

    def _sqlite_embedding_search(self, query: str, top_k: int) -> list[dict]:
        """Fallback: cosine similarity search on SQLite-stored embeddings."""
        query_vec = self._embedder.encode(query)

        rows = self.db.execute(
            "SELECT c.id, c.content, c.embedding, d.name, c.chunk_index "
            "FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.id "
            "WHERE c.embedding IS NOT NULL").fetchall()

        if not rows:
            return self._keyword_search(query, top_k)

        scored = []
        for chunk_id, content, emb_blob, doc_name, chunk_idx in rows:
            try:
                chunk_vec = pickle.loads(emb_blob)
                sim = self._cosine_similarity(query_vec, chunk_vec)
                scored.append({
                    "content": content,
                    "source": doc_name,
                    "chunk_index": chunk_idx,
                    "score": round(sim, 4),
                })
            except Exception:
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """Fallback keyword-based search."""
        words = set(query.lower().split())
        if not words:
            return []

        rows = self.db.execute(
            "SELECT c.content, d.name, c.chunk_index "
            "FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.id"
        ).fetchall()

        scored = []
        for content, doc_name, chunk_idx in rows:
            chunk_words = set(content.lower().split())
            overlap = len(words & chunk_words) / max(len(words), 1)
            if overlap > 0:
                scored.append({
                    "content": content,
                    "source": doc_name,
                    "chunk_index": chunk_idx,
                    "score": round(overlap, 4),
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        dot = float(a @ b)
        norm_a = float(math.sqrt(a @ a))
        norm_b = float(math.sqrt(b @ b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    # ═══════════════════════════════════════
    # MANAGEMENT
    # ═══════════════════════════════════════

    def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        rows = self.db.execute(
            "SELECT id, path, name, chunk_count, created_at FROM rag_documents "
            "ORDER BY created_at DESC").fetchall()
        return [{"id": r[0], "path": r[1], "name": r[2],
                 "chunks": r[3], "created_at": r[4]} for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        """Remove a document, its chunks, and Qdrant vectors."""
        self._delete_doc_vectors(doc_id)
        self.db.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,))
        cur = self.db.execute("DELETE FROM rag_documents WHERE id = ?", (doc_id,))
        self.db.commit()
        return cur.rowcount > 0

    def get_stats(self) -> dict:
        """Get RAG pipeline statistics."""
        doc_count = self.db.execute(
            "SELECT COUNT(*) FROM rag_documents").fetchone()[0]
        chunk_count = self.db.execute(
            "SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        stats = {"documents": doc_count, "chunks": chunk_count, "backend": "keyword"}
        if self._qdrant:
            stats["backend"] = "qdrant"
            try:
                info = self._qdrant.get_collection(self._collection)
                stats["qdrant_vectors"] = info.points_count
            except Exception:
                pass
        elif self._embedder:
            stats["backend"] = "sqlite_embeddings"
        return stats

    def is_qdrant_connected(self) -> bool:
        """Check if Qdrant is available."""
        if not self._qdrant:
            return False
        try:
            self._qdrant.get_collections()
            return True
        except Exception:
            return False
