"""RAG pipeline — document ingestion, chunking, hybrid search (vector + keyword + RRF fusion).

Supports 3 vector backends (tiered):
  1. sqlite-vec  — native SQLite extension, exact KNN, <50K vectors
  2. Qdrant      — ANN server, scales to millions
  3. SQLite brute-force — pickle embeddings, O(n) cosine (legacy fallback)

Config (config.json → "rag"):
  {
    "enabled": true,
    "chunk_size": 1000,
    "overlap": 200,
    "top_k": 5,
    "vector_backend": "auto",       // auto | sqlite_vec | qdrant | sqlite | keyword
    "search": {
      "mode": "hybrid",             // hybrid | vector | keyword
      "rrf_k": 60,
      "vector_top_k": 50,
      "keyword_top_k": 50
    },
    "embedding": { ... },           // see embedders.py
    "qdrant": { "url": "", "api_key": "", "collection": "liteagent_rag" },
    "file_indexing": { "enabled": true, "max_file_size_mb": 10 }
  }
"""

import hashlib
import logging
import math
import pickle
import re
import sqlite3
import struct
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# VECTOR BACKEND INTERFACE
# ═══════════════════════════════════════════════════════════════

class VectorBackend:
    """Abstract interface for vector storage and search."""
    name: str = "base"

    def upsert(self, ids: list[str], vectors: list, payloads: list[dict]):
        raise NotImplementedError

    def search(self, query_vector, top_k: int = 10) -> list[dict]:
        """Returns [{"id": str, "score": float, "payload": dict}, ...]"""
        raise NotImplementedError

    def delete(self, ids: list[str]):
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class SqliteVecBackend(VectorBackend):
    """sqlite-vec extension — native C KNN, exact search. Best for <50K vectors."""
    name = "sqlite_vec"

    def __init__(self, db: sqlite3.Connection, dim: int, table: str = "vec_rag"):
        self._db = db
        self._dim = dim
        self._table = table
        self._meta_table = f"{table}_meta"
        self._init(db, dim)

    def _init(self, db, dim):
        import sqlite_vec
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        # vec0 virtual table
        db.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {self._table} "
            f"USING vec0(embedding float[{dim}])")
        # Metadata mapping: rowid → id + payload
        db.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._meta_table} (
                rowid INTEGER PRIMARY KEY,
                ext_id TEXT UNIQUE NOT NULL,
                payload_json TEXT DEFAULT '{{}}'
            )""")
        db.commit()

    def upsert(self, ids: list[str], vectors: list, payloads: list[dict]):
        for ext_id, vec, payload in zip(ids, vectors, payloads):
            vec_bytes = self._to_bytes(vec)
            # Check if exists
            row = self._db.execute(
                f"SELECT rowid FROM {self._meta_table} WHERE ext_id = ?",
                (ext_id,)).fetchone()
            if row:
                rid = row[0]
                self._db.execute(
                    f"UPDATE {self._table} SET embedding = ? WHERE rowid = ?",
                    (vec_bytes, rid))
                import json
                self._db.execute(
                    f"UPDATE {self._meta_table} SET payload_json = ? WHERE rowid = ?",
                    (json.dumps(payload, ensure_ascii=False), rid))
            else:
                self._db.execute(
                    f"INSERT INTO {self._table}(embedding) VALUES (?)",
                    (vec_bytes,))
                rid = self._db.execute("SELECT last_insert_rowid()").fetchone()[0]
                import json
                self._db.execute(
                    f"INSERT INTO {self._meta_table}(rowid, ext_id, payload_json) VALUES (?, ?, ?)",
                    (rid, ext_id, json.dumps(payload, ensure_ascii=False)))
        self._db.commit()

    def search(self, query_vector, top_k: int = 10) -> list[dict]:
        vec_bytes = self._to_bytes(query_vector)
        rows = self._db.execute(f"""
            SELECT v.rowid, v.distance, m.ext_id, m.payload_json
            FROM {self._table} v
            JOIN {self._meta_table} m ON m.rowid = v.rowid
            WHERE v.embedding MATCH ?
            ORDER BY v.distance LIMIT ?
        """, (vec_bytes, top_k)).fetchall()
        import json
        results = []
        for rid, distance, ext_id, pjson in rows:
            # sqlite-vec returns L2 distance; convert to similarity score 0-1
            score = 1.0 / (1.0 + distance)
            results.append({
                "id": ext_id,
                "score": round(score, 4),
                "payload": json.loads(pjson) if pjson else {},
            })
        return results

    def delete(self, ids: list[str]):
        for ext_id in ids:
            row = self._db.execute(
                f"SELECT rowid FROM {self._meta_table} WHERE ext_id = ?",
                (ext_id,)).fetchone()
            if row:
                self._db.execute(f"DELETE FROM {self._table} WHERE rowid = ?", (row[0],))
                self._db.execute(f"DELETE FROM {self._meta_table} WHERE rowid = ?", (row[0],))
        self._db.commit()

    def count(self) -> int:
        row = self._db.execute(f"SELECT COUNT(*) FROM {self._meta_table}").fetchone()
        return row[0] if row else 0

    def _to_bytes(self, vec) -> bytes:
        """Convert vector (list/array) to float32 bytes for sqlite-vec."""
        import numpy as np
        if isinstance(vec, np.ndarray):
            return vec.astype("float32").tobytes()
        return struct.pack(f"{len(vec)}f", *vec)


class QdrantBackend(VectorBackend):
    """Qdrant vector database — ANN, scales to millions of vectors."""
    name = "qdrant"

    def __init__(self, url: str, api_key: str = "",
                 collection: str = "liteagent_rag", dim: int = 384):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        self._client = QdrantClient(url=url, api_key=api_key or None, timeout=10)
        self._collection = collection
        # Ensure collection exists
        try:
            collections = [c.name for c in self._client.get_collections().collections]
            if collection not in collections:
                self._client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
                logger.info("Created Qdrant collection: %s (dim=%d)", collection, dim)
        except Exception as e:
            logger.warning("Qdrant collection setup failed: %s", e)

    def upsert(self, ids: list[str], vectors: list, payloads: list[dict]):
        from qdrant_client.models import PointStruct
        points = []
        for ext_id, vec, payload in zip(ids, vectors, payloads):
            v = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
            points.append(PointStruct(id=ext_id, vector=v, payload=payload))
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vector, top_k: int = 10) -> list[dict]:
        v = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
        results = self._client.query_points(
            collection_name=self._collection, query=v, limit=top_k)
        return [
            {"id": str(hit.id), "score": round(hit.score, 4), "payload": hit.payload or {}}
            for hit in results.points
        ]

    def delete(self, ids: list[str]):
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=ids))

    def count(self) -> int:
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            return 0


class SqliteBruteForceBackend(VectorBackend):
    """Legacy fallback: pickle-serialized embeddings in SQLite, O(n) cosine search."""
    name = "sqlite_brute_force"

    def __init__(self, db: sqlite3.Connection):
        self._db = db
        # Uses existing rag_chunks.embedding column (BLOB, pickle format)

    def upsert(self, ids: list[str], vectors: list, payloads: list[dict]):
        # Store in rag_chunks via embedding BLOB — handled by RAGPipeline directly
        pass  # No-op: the pipeline writes directly to rag_chunks

    def search(self, query_vector, top_k: int = 10) -> list[dict]:
        rows = self._db.execute(
            "SELECT c.id, c.content, c.embedding, d.name, c.chunk_index "
            "FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.id "
            "WHERE c.embedding IS NOT NULL").fetchall()
        if not rows:
            return []

        scored = []
        for chunk_id, content, emb_blob, doc_name, chunk_idx in rows:
            try:
                chunk_vec = pickle.loads(emb_blob)
                sim = _cosine_similarity(query_vector, chunk_vec)
                scored.append({
                    "id": str(chunk_id),
                    "score": round(sim, 4),
                    "payload": {
                        "content": content,
                        "doc_name": doc_name,
                        "chunk_index": chunk_idx,
                    },
                })
            except Exception:
                continue
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete(self, ids: list[str]):
        pass  # Handled by RAGPipeline via SQL

    def count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE embedding IS NOT NULL").fetchone()
        return row[0] if row else 0


def _cosine_similarity(a, b) -> float:
    """Cosine similarity between two vectors."""
    dot = float(a @ b)
    norm_a = float(math.sqrt(a @ a))
    norm_b = float(math.sqrt(b @ b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def create_vector_backend(config: dict, db: sqlite3.Connection,
                          embedder=None) -> VectorBackend | None:
    """Factory: create the best vector backend based on config.

    Priority for "auto": sqlite-vec → Qdrant (if configured) → brute-force → None
    """
    rag_cfg = config.get("rag", {})
    backend_name = rag_cfg.get("vector_backend", "auto")
    dim = getattr(embedder, 'dim', 384) if embedder else 384

    if backend_name == "keyword":
        return None

    if backend_name == "sqlite_vec" or backend_name == "auto":
        try:
            backend = SqliteVecBackend(db, dim)
            logger.info("Vector backend: sqlite-vec (dim=%d)", dim)
            return backend
        except Exception as e:
            if backend_name == "sqlite_vec":
                logger.warning("sqlite-vec requested but failed: %s", e)
                return None
            logger.debug("sqlite-vec not available: %s", e)

    if backend_name == "qdrant" or backend_name == "auto":
        qdrant_cfg = rag_cfg.get("qdrant", {})
        url = qdrant_cfg.get("url", "")
        if url:
            try:
                api_key = qdrant_cfg.get("api_key", "")
                if not api_key:
                    try:
                        from .config import get_api_key
                        api_key = get_api_key("qdrant") or ""
                    except Exception:
                        pass
                collection = qdrant_cfg.get("collection", "liteagent_rag")
                backend = QdrantBackend(url=url, api_key=api_key,
                                        collection=collection, dim=dim)
                logger.info("Vector backend: Qdrant (%s, collection=%s)", url, collection)
                return backend
            except ImportError:
                logger.debug("qdrant-client not installed")
            except Exception as e:
                if backend_name == "qdrant":
                    logger.warning("Qdrant requested but failed: %s", e)
                    return None
                logger.debug("Qdrant not available: %s", e)

    if backend_name == "sqlite" or backend_name == "auto":
        if embedder:
            logger.info("Vector backend: SQLite brute-force (legacy)")
            return SqliteBruteForceBackend(db)

    return None


# ═══════════════════════════════════════════════════════════════
# RAG PIPELINE
# ═══════════════════════════════════════════════════════════════

class RAGPipeline:
    """Document ingestion and retrieval with tiered vector backends and hybrid search."""

    def __init__(self, db: sqlite3.Connection, embedder=None, config: dict = None,
                 sandbox_root: str | None = None):
        self.db = db
        self._embedder = embedder
        self._sandbox_root = sandbox_root
        cfg = config or {}
        self.chunk_size = cfg.get("chunk_size", 1000)
        self.chunk_overlap = cfg.get("overlap", 200)
        self.top_k = cfg.get("top_k", 5)

        # Search config
        search_cfg = cfg.get("search", {})
        self._search_mode = search_cfg.get("mode", "hybrid")
        self._rrf_k = search_cfg.get("rrf_k", 60)
        self._vector_top_k = search_cfg.get("vector_top_k", 50)
        self._keyword_top_k = search_cfg.get("keyword_top_k", 50)

        # Vector backend (tiered: sqlite-vec → Qdrant → brute-force → None)
        self._backend: VectorBackend | None = None
        self._collection = cfg.get("collection", "liteagent_rag")

        # Legacy Qdrant compat: if qdrant.url is set in old-style config
        qdrant_cfg = cfg.get("qdrant", {})
        if qdrant_cfg.get("url") and not cfg.get("vector_backend"):
            cfg["vector_backend"] = "qdrant"

        self._init_tables()

    def init_backend(self, full_config: dict):
        """Initialize vector backend. Called after construction when full config is available."""
        self._backend = create_vector_backend(full_config, self.db, self._embedder)

    def _init_tables(self):
        """SQLite metadata tables."""
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
        # FTS5 for BM25 scoring — may not be available in all SQLite builds
        try:
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS rag_fts USING fts5(
                    content,
                    chunk_id UNINDEXED,
                    doc_name UNINDEXED,
                    tokenize='unicode61'
                )
            """)
            self._fts_available = True
        except Exception:
            logger.debug("FTS5 not available — using word-overlap fallback for keyword search")
            self._fts_available = False
        self.db.commit()

    # ═══════════════════════════════════════
    # DOCUMENT LOADERS
    # ═══════════════════════════════════════

    def _validate_sandbox(self, path: Path) -> None:
        if not self._sandbox_root:
            return
        import os
        root = os.path.realpath(os.path.expanduser(self._sandbox_root))
        resolved = str(path.resolve())
        if not resolved.startswith(root + os.sep) and resolved != root:
            raise PermissionError(
                f"RAG access denied: path '{path}' is outside sandbox '{root}'")

    def load_file(self, path: str) -> str:
        """Load document content from file."""
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
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html, flags=re.S | re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                              ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
            text = text.replace(entity, char)
        return text

    @staticmethod
    def _load_pdf(path: Path) -> str:
        try:
            import fitz
            doc = fitz.open(str(path))
            return "\n\n".join(page.get_text() for page in doc)
        except ImportError:
            raise ImportError("PDF support requires pymupdf: pip install liteagent[pdf]")

    # ═══════════════════════════════════════
    # CHUNKING (improved: markdown/code-aware)
    # ═══════════════════════════════════════

    def chunk_text(self, text: str, file_type: str = "") -> list[str]:
        """Recursive text splitter with overlap, file-type-aware separators."""
        if not text or not text.strip():
            return []

        # Protect fenced code blocks from splitting
        protected = text
        code_blocks = {}
        if "```" in text:
            protected, code_blocks = self._protect_code_blocks(text)

        # Choose separators by file type
        ft = file_type.lower().lstrip(".")
        if ft in ("md", "markdown"):
            separators = ["\n# ", "\n## ", "\n### ", "\n\n", "\n", ". ", " "]
        elif ft in ("py",):
            separators = ["\nclass ", "\ndef ", "\nasync def ", "\n\n", "\n", " "]
        elif ft in ("js", "ts", "jsx", "tsx"):
            separators = ["\nfunction ", "\nclass ", "\nexport ", "\n\n", "\n", " "]
        else:
            separators = ["\n\n", "\n", ". ", " "]

        chunks = self._recursive_split(
            protected, separators, self.chunk_size, self.chunk_overlap)

        # Restore code blocks
        if code_blocks:
            chunks = [self._restore_code_blocks(c, code_blocks) for c in chunks]

        return [c for c in chunks if c.strip()]

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

    @staticmethod
    def _protect_code_blocks(text: str) -> tuple[str, dict]:
        """Replace fenced code blocks with placeholders to prevent splitting inside them."""
        blocks = {}
        pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        def replacer(m):
            key = f"__CODE_BLOCK_{len(blocks)}__"
            blocks[key] = m.group(0)
            return key
        protected = pattern.sub(replacer, text)
        return protected, blocks

    @staticmethod
    def _restore_code_blocks(text: str, blocks: dict) -> str:
        for key, code in blocks.items():
            text = text.replace(key, code)
        return text

    # ═══════════════════════════════════════
    # INGESTION
    # ═══════════════════════════════════════

    def ingest(self, path: str) -> dict:
        """Ingest a file or directory."""
        p = Path(path).expanduser().resolve()
        self._validate_sandbox(p)
        if p.is_dir():
            return self._ingest_directory(p)
        return self._ingest_file(p)

    def _ingest_file(self, path: Path) -> dict:
        """Ingest a single file: chunk → embed → store in vector backend + SQLite."""
        content = self.load_file(str(path))
        doc_hash = hashlib.md5(content.encode()).hexdigest()

        existing = self.db.execute(
            "SELECT id, doc_hash FROM rag_documents WHERE path = ?",
            (str(path),)).fetchone()
        if existing and existing[1] == doc_hash:
            return {"path": str(path), "status": "unchanged", "chunks": 0}

        if existing:
            self._delete_doc_data(existing[0])

        file_type = path.suffix.lstrip(".")
        chunks = self.chunk_text(content, file_type=file_type)

        self.db.execute(
            "INSERT INTO rag_documents (path, name, doc_hash, chunk_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(path), path.name, doc_hash, len(chunks), datetime.now().isoformat()))
        doc_id = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]

        self._store_chunks(chunks, doc_id, path.name, str(path))
        self.db.commit()
        return {"path": str(path), "status": "ingested", "chunks": len(chunks)}

    def index_content(self, text: str, source_key: str, source_name: str,
                      file_type: str = "") -> dict:
        """Index arbitrary text content (used by FileManager for S3 files).

        Unlike ingest(), this doesn't read from local filesystem.
        """
        doc_hash = hashlib.md5(text.encode()).hexdigest()

        existing = self.db.execute(
            "SELECT id, doc_hash FROM rag_documents WHERE path = ?",
            (source_key,)).fetchone()
        if existing and existing[1] == doc_hash:
            return {"path": source_key, "status": "unchanged", "chunks": 0}
        if existing:
            self._delete_doc_data(existing[0])

        chunks = self.chunk_text(text, file_type=file_type)

        self.db.execute(
            "INSERT INTO rag_documents (path, name, doc_hash, chunk_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_key, source_name, doc_hash, len(chunks), datetime.now().isoformat()))
        doc_id = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]

        self._store_chunks(chunks, doc_id, source_name, source_key)
        self.db.commit()
        return {"path": source_key, "status": "indexed", "chunks": len(chunks)}

    def _store_chunks(self, chunks: list[str], doc_id: int,
                      doc_name: str, doc_path: str):
        """Embed and store chunks in vector backend + SQLite metadata."""
        vec_ids = []
        vec_vectors = []
        vec_payloads = []

        for i, chunk in enumerate(chunks):
            embedding_blob = None
            vector = None
            chunk_id = str(uuid.uuid4())

            if self._embedder:
                try:
                    vec = self._embedder.encode(chunk)
                    vector = vec
                    embedding_blob = pickle.dumps(vec)
                except Exception as e:
                    logger.debug("Embedding failed for chunk %d: %s", i, e)

            self.db.execute(
                "INSERT INTO rag_chunks (doc_id, chunk_index, content, embedding, qdrant_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, i, chunk, embedding_blob, chunk_id))

            # FTS5 index for BM25 keyword search
            if self._fts_available:
                try:
                    self.db.execute(
                        "INSERT INTO rag_fts (content, chunk_id, doc_name) VALUES (?, ?, ?)",
                        (chunk, chunk_id, doc_name))
                except Exception:
                    pass

            if vector is not None and self._backend:
                vec_ids.append(chunk_id)
                vec_vectors.append(vector)
                vec_payloads.append({
                    "content": chunk[:1000],
                    "doc_name": doc_name,
                    "doc_path": doc_path,
                    "chunk_index": i,
                })

        # Batch upsert to vector backend
        if vec_ids and self._backend:
            try:
                self._backend.upsert(vec_ids, vec_vectors, vec_payloads)
            except Exception as e:
                logger.warning("Vector backend upsert failed: %s", e)

    def _delete_doc_data(self, doc_id: int):
        """Delete vectors + chunks for a document."""
        if self._backend:
            rows = self.db.execute(
                "SELECT qdrant_id FROM rag_chunks WHERE doc_id = ? AND qdrant_id IS NOT NULL",
                (doc_id,)).fetchall()
            if rows:
                ids = [r[0] for r in rows]
                try:
                    self._backend.delete(ids)
                except Exception as e:
                    logger.warning("Vector backend delete failed: %s", e)
        # Clean FTS entries
        if self._fts_available:
            fts_ids = self.db.execute(
                "SELECT qdrant_id FROM rag_chunks WHERE doc_id = ?", (doc_id,)).fetchall()
            if fts_ids:
                placeholders = ",".join("?" for _ in fts_ids)
                try:
                    self.db.execute(
                        f"DELETE FROM rag_fts WHERE chunk_id IN ({placeholders})",
                        [r[0] for r in fts_ids])
                except Exception:
                    pass
        self.db.execute("DELETE FROM rag_chunks WHERE doc_id = ?", (doc_id,))
        self.db.execute("DELETE FROM rag_documents WHERE id = ?", (doc_id,))

    def _ingest_directory(self, path: Path) -> dict:
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
    # RETRIEVAL (hybrid: vector + keyword + RRF)
    # ═══════════════════════════════════════

    def search(self, query: str, top_k: int | None = None) -> list[dict]:
        """Hybrid search: vector + keyword with RRF fusion."""
        k = top_k or self.top_k
        mode = self._search_mode

        vector_results = []
        keyword_results = []

        # Vector search
        if mode in ("hybrid", "vector") and self._embedder:
            vector_results = self._vector_search(query, self._vector_top_k)

        # Keyword search
        if mode in ("hybrid", "keyword"):
            keyword_results = self._keyword_search(query, self._keyword_top_k)

        # Fusion
        if mode == "hybrid" and vector_results and keyword_results:
            return self._rrf_fusion(vector_results, keyword_results, k=self._rrf_k)[:k]

        # Single-mode results
        results = vector_results or keyword_results
        return results[:k]

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """Vector search via backend."""
        if not self._embedder:
            return []

        query_vec = self._embedder.encode(query)

        # Use the vector backend
        if self._backend:
            try:
                raw = self._backend.search(query_vec, top_k=top_k)
                return [
                    {
                        "content": hit["payload"].get("content", ""),
                        "source": hit["payload"].get("doc_name", ""),
                        "chunk_index": hit["payload"].get("chunk_index", 0),
                        "score": hit["score"],
                    }
                    for hit in raw
                ]
            except Exception as e:
                logger.warning("Vector search failed: %s", e)

        # Ultimate fallback: brute-force in Python
        return self._sqlite_embedding_search_raw(query_vec, top_k)

    def _sqlite_embedding_search_raw(self, query_vec, top_k: int) -> list[dict]:
        """Brute-force cosine similarity on SQLite embeddings (no backend needed)."""
        rows = self.db.execute(
            "SELECT c.content, c.embedding, d.name, c.chunk_index "
            "FROM rag_chunks c JOIN rag_documents d ON c.doc_id = d.id "
            "WHERE c.embedding IS NOT NULL").fetchall()
        if not rows:
            return []
        scored = []
        for content, emb_blob, doc_name, chunk_idx in rows:
            try:
                chunk_vec = pickle.loads(emb_blob)
                sim = _cosine_similarity(query_vec, chunk_vec)
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
        """Keyword search — BM25 via FTS5 with word-overlap fallback."""
        if self._fts_available:
            results = self._bm25_search(query, top_k)
            if results:
                return results
        return self._word_overlap_search(query, top_k)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """BM25 search via SQLite FTS5."""
        import re as _re
        safe_query = _re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe_query:
            return []
        terms = safe_query.split()
        fts_query = " OR ".join(terms)
        try:
            rows = self.db.execute("""
                SELECT f.chunk_id, f.doc_name, c.content, c.chunk_index,
                       bm25(rag_fts) as score
                FROM rag_fts f
                JOIN rag_chunks c ON c.qdrant_id = f.chunk_id
                WHERE rag_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (fts_query, top_k)).fetchall()
        except Exception as e:
            logger.debug("FTS5 search error: %s", e)
            return []
        results = []
        for row in rows:
            results.append({
                "content": row[2],
                "source": row[1],
                "chunk_index": row[3],
                "score": round(-row[4], 4),  # BM25 scores are negative in FTS5
            })
        return results

    def _word_overlap_search(self, query: str, top_k: int) -> list[dict]:
        """Legacy word-overlap fallback when FTS5 is not available."""
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
    def _rrf_fusion(vector_results: list[dict], keyword_results: list[dict],
                    k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion — merge two ranked lists into one.

        RRF score = Σ 1/(k + rank_i) for each list the document appears in.
        k=60 is the standard default (from the original RRF paper).
        """
        scores: dict[str, float] = {}
        data: dict[str, dict] = {}

        for rank, item in enumerate(vector_results):
            key = f"{item['source']}:{item['chunk_index']}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            data[key] = item

        for rank, item in enumerate(keyword_results):
            key = f"{item['source']}:{item['chunk_index']}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in data:
                data[key] = item

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for key, score in ranked:
            item = data[key].copy()
            item["score"] = round(score, 6)
            results.append(item)
        return results

    # ═══════════════════════════════════════
    # MANAGEMENT
    # ═══════════════════════════════════════

    def list_documents(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT id, path, name, chunk_count, created_at FROM rag_documents "
            "ORDER BY created_at DESC").fetchall()
        return [{"id": r[0], "path": r[1], "name": r[2],
                 "chunks": r[3], "created_at": r[4]} for r in rows]

    def delete_document(self, doc_id: int) -> bool:
        exists = self.db.execute(
            "SELECT 1 FROM rag_documents WHERE id = ?", (doc_id,)).fetchone()
        if not exists:
            return False
        self._delete_doc_data(doc_id)
        self.db.commit()
        return True

    def get_stats(self) -> dict:
        doc_count = self.db.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]
        chunk_count = self.db.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
        backend_name = self._backend.name if self._backend else "keyword"
        stats = {
            "documents": doc_count,
            "chunks": chunk_count,
            "backend": backend_name,
            "search_mode": self._search_mode,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedder": getattr(self._embedder, 'name', None),
            "embedder_dim": getattr(self._embedder, 'dim', None),
        }
        if self._backend:
            try:
                stats["vector_count"] = self._backend.count()
            except Exception:
                pass
        return stats

    def is_qdrant_connected(self) -> bool:
        """Check if Qdrant backend is active."""
        return isinstance(self._backend, QdrantBackend)

    # Legacy compat alias
    @staticmethod
    def _cosine_similarity(a, b) -> float:
        return _cosine_similarity(a, b)
