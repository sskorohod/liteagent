"""Knowledge Base — Advanced RAG pipeline for books and reference materials.

Separate from the general RAG pipeline (rag.py). Designed for domain knowledge
(accounting, law, regulations, etc.) with NotebookLM-quality retrieval.

Pipeline: Query Rewrite → Hybrid Search (BM25 + Vector) → Rerank → Context Builder

Features:
  - Structure-preserving PDF/Markdown/HTML parsing (headings, tables, pages)
  - Semantic chunking (by sections, not fixed-size)
  - SQLite FTS5 for BM25 scoring (zero extra deps)
  - Vector search via shared embedder (Ollama/ST/OpenAI)
  - Cross-encoder reranking (sentence-transformers) with LLM fallback
  - Query rewriting (LLM decomposes into sub-queries)
  - Citations with source, page, section
  - Quality logging (query log table)

Storage: separate SQLite DB at ~/.liteagent/knowledge_base.db

Config (config.json → "knowledge_base"):
  {
    "enabled": true,
    "db_path": "~/.liteagent/knowledge_base.db",
    "chunk_size": 800,
    "chunk_overlap": 150,
    "search_mode": "hybrid",
    "rerank": true,
    "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "query_rewrite": true,
    "max_file_size_mb": 50
  }
"""

import hashlib
import json
import logging
import math
import os
import pickle
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class KBDocument:
    id: str
    name: str
    source_path: str
    doc_hash: str
    page_count: int = 0
    chunk_count: int = 0
    metadata: dict = field(default_factory=dict)
    created_at: str = ""


@dataclass
class KBChunk:
    id: str
    doc_id: str
    content: str
    section_path: str = ""
    page_start: int = 0
    page_end: int = 0
    chunk_type: str = "text"
    chunk_index: int = 0
    context_prefix: str = ""
    parent_id: str | None = None


@dataclass
class KBSearchResult:
    content: str
    score: float
    source: str
    page: int
    section: str
    chunk_id: str
    context_prefix: str = ""


# ═══════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════

class KnowledgeBase:
    """Advanced RAG knowledge base for books and reference materials."""

    def __init__(self, config: dict, embedder=None, provider=None):
        self._config = config
        self._embedder = embedder
        self._provider = provider
        self.chunk_size = config.get("chunk_size", 800)
        self.chunk_overlap = config.get("chunk_overlap", 150)
        self._search_mode = config.get("search_mode", "hybrid")
        self._rerank_enabled = config.get("rerank", True)
        self._rerank_model = config.get("rerank_model",
                                        "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._query_rewrite_enabled = config.get("query_rewrite", True)
        self._max_file_size_mb = config.get("max_file_size_mb", 50)
        self._contextual_retrieval = config.get("contextual_retrieval", False)
        self._parent_child = config.get("parent_child_retrieval", False)
        self._self_rag = config.get("self_rag", False)
        self._self_rag_threshold = config.get("self_rag_threshold", 0.35)

        # Open separate database
        db_path = os.path.expanduser(config.get("db_path",
                                                 "~/.liteagent/knowledge_base.db"))
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

        # Cross-encoder reranker (lazy init)
        self._cross_encoder = None
        self._cross_encoder_checked = False

    # ═══════════════════════════════════════
    # SCHEMA
    # ═══════════════════════════════════════

    def _init_schema(self):
        """Create tables if not exist."""
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS kb_documents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_path TEXT,
                doc_hash TEXT UNIQUE,
                page_count INTEGER DEFAULT 0,
                chunk_count INTEGER DEFAULT 0,
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS kb_chunks (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES kb_documents(id),
                content TEXT NOT NULL,
                section_path TEXT DEFAULT '',
                page_start INTEGER DEFAULT 0,
                page_end INTEGER DEFAULT 0,
                chunk_type TEXT DEFAULT 'text',
                chunk_index INTEGER DEFAULT 0,
                embedding BLOB,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_kb_chunks_doc ON kb_chunks(doc_id);

            CREATE TABLE IF NOT EXISTS kb_query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                rewritten_queries TEXT,
                chunks_retrieved TEXT,
                result_count INTEGER,
                latency_ms INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        # FTS5 — may not be available in all SQLite builds
        try:
            self.db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts USING fts5(
                    content,
                    chunk_id UNINDEXED,
                    doc_name UNINDEXED,
                    section_path UNINDEXED,
                    tokenize='unicode61'
                )
            """)
            self._fts_available = True
        except Exception:
            logger.warning("FTS5 not available — BM25 search disabled, using vector-only")
            self._fts_available = False
        self.db.commit()

        # Schema migrations
        for migration in [
            "ALTER TABLE kb_chunks ADD COLUMN context_prefix TEXT DEFAULT ''",
            "ALTER TABLE kb_chunks ADD COLUMN parent_id TEXT DEFAULT NULL",
            "ALTER TABLE kb_query_log ADD COLUMN faithfulness_score REAL",
            "ALTER TABLE kb_query_log ADD COLUMN relevancy_score REAL",
            "ALTER TABLE kb_query_log ADD COLUMN context_precision REAL",
        ]:
            try:
                self.db.execute(migration)
                self.db.commit()
            except sqlite3.OperationalError:
                pass  # Column already exists

        try:
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_kb_chunks_parent ON kb_chunks(parent_id)")
            self.db.commit()
        except Exception:
            pass

        # Entity graph tables (Phase 5: GraphRAG)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS kb_entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                doc_id TEXT REFERENCES kb_documents(id),
                count INTEGER DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_kb_entities_name ON kb_entities(name);
            CREATE INDEX IF NOT EXISTS idx_kb_entities_doc ON kb_entities(doc_id);

            CREATE TABLE IF NOT EXISTS kb_entity_mentions (
                entity_id TEXT REFERENCES kb_entities(id),
                chunk_id TEXT REFERENCES kb_chunks(id),
                PRIMARY KEY (entity_id, chunk_id)
            );
            CREATE INDEX IF NOT EXISTS idx_kb_mentions_chunk ON kb_entity_mentions(chunk_id);
        """)

    async def _generate_context_prefix(self, chunk: KBChunk, doc_name: str) -> str:
        """Generate 2-3 sentence context prefix for a chunk using LLM (Anthropic Contextual Retrieval)."""
        if not self._provider:
            return ""
        prompt = (
            f"Document: \"{doc_name}\"\n"
            f"Section: \"{chunk.section_path}\"\n\n"
            f"Chunk content:\n{chunk.content[:1500]}\n\n"
            "Please give a short succinct context (2-3 sentences) to situate this chunk "
            "within the overall document for the purposes of improving search retrieval. "
            "Answer ONLY with the context, nothing else."
        )
        try:
            resp = await self._provider.complete(
                messages=[{"role": "user", "content": prompt}], max_tokens=200)
            return resp.text.strip()
        except Exception as e:
            logger.debug("Context prefix generation failed: %s", e)
            return ""

    # ═══════════════════════════════════════
    # PARSING (structure-preserving)
    # ═══════════════════════════════════════

    def _parse_file(self, path: str) -> dict:
        """Parse a file into structured format.

        Returns: {
            "name": str,
            "pages": [{"number": int, "sections": [{"level": int, "title": str, "content": str}], "tables": [str]}],
            "total_pages": int
        }
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > self._max_file_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.1f}MB (max {self._max_file_size_mb}MB)")

        suffix = p.suffix.lower()
        if suffix == ".pdf":
            return self._parse_pdf(p)
        elif suffix in (".md", ".markdown"):
            return self._parse_markdown(p)
        elif suffix in (".html", ".htm"):
            return self._parse_html(p)
        else:
            return self._parse_plain_text(p)

    def _parse_pdf(self, path: Path) -> dict:
        """Parse PDF preserving page numbers, headings (by font size), and tables."""
        try:
            import fitz
        except ImportError:
            raise ImportError(
                "PDF support requires pymupdf: pip install liteagent[pdf]")

        doc = fitz.open(str(path))
        pages = []

        for page_num, page in enumerate(doc, 1):
            blocks = page.get_text("dict", sort=True).get("blocks", [])
            sections = []
            tables = []
            current_text = []
            current_heading = ""
            current_level = 0

            # Collect font sizes to detect headings
            font_sizes = []
            for block in blocks:
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("text", "").strip():
                                font_sizes.append(span.get("size", 12))

            if not font_sizes:
                pages.append({"number": page_num, "sections": [], "tables": []})
                continue

            # Determine heading thresholds
            body_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 12
            heading_threshold = body_size * 1.15

            for block in blocks:
                if block.get("type") == 0:  # text block
                    for line in block.get("lines", []):
                        line_text = ""
                        line_size = 0
                        is_bold = False
                        for span in line.get("spans", []):
                            text = span.get("text", "")
                            line_text += text
                            line_size = max(line_size, span.get("size", 12))
                            if "bold" in span.get("font", "").lower():
                                is_bold = True

                        line_text = line_text.strip()
                        if not line_text:
                            continue

                        # Detect heading by font size or bold
                        is_heading = line_size >= heading_threshold
                        if is_heading:
                            # Save previous section
                            if current_text or current_heading:
                                sections.append({
                                    "level": current_level,
                                    "title": current_heading,
                                    "content": "\n".join(current_text).strip(),
                                })
                            # Determine heading level
                            if line_size >= body_size * 1.5:
                                current_level = 1
                            elif line_size >= body_size * 1.3:
                                current_level = 2
                            else:
                                current_level = 3
                            current_heading = line_text
                            current_text = []
                        else:
                            current_text.append(line_text)

                elif block.get("type") == 1:  # image block
                    pass  # skip images for now

            # Save last section
            if current_text or current_heading:
                sections.append({
                    "level": current_level,
                    "title": current_heading,
                    "content": "\n".join(current_text).strip(),
                })

            # Extract tables via pymupdf
            try:
                page_tables = page.find_tables()
                for table in page_tables:
                    rows = table.extract()
                    if rows:
                        table_text = self._format_table(rows)
                        if table_text.strip():
                            tables.append(table_text)
            except Exception:
                pass

            pages.append({
                "number": page_num,
                "sections": sections,
                "tables": tables,
            })

        doc.close()
        return {
            "name": path.stem,
            "pages": pages,
            "total_pages": len(pages),
        }

    @staticmethod
    def _format_table(rows: list[list]) -> str:
        """Format table rows as readable text."""
        if not rows:
            return ""
        lines = []
        for i, row in enumerate(rows):
            cells = [str(c).strip() if c else "" for c in row]
            lines.append(" | ".join(cells))
            if i == 0:
                lines.append("-" * len(lines[0]))
        return "\n".join(lines)

    def _parse_markdown(self, path: Path) -> dict:
        """Parse markdown with heading hierarchy."""
        text = path.read_text(encoding="utf-8", errors="replace")
        pages = [{"number": 1, "sections": [], "tables": []}]
        current_sections = pages[0]["sections"]

        lines = text.split("\n")
        current_heading = ""
        current_level = 0
        current_content = []

        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)', line)
            if heading_match:
                # Save previous section
                if current_content or current_heading:
                    current_sections.append({
                        "level": current_level,
                        "title": current_heading,
                        "content": "\n".join(current_content).strip(),
                    })
                current_level = len(heading_match.group(1))
                current_heading = heading_match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)

        # Save last section
        if current_content or current_heading:
            current_sections.append({
                "level": current_level,
                "title": current_heading,
                "content": "\n".join(current_content).strip(),
            })

        # Extract markdown tables
        table_pattern = re.compile(
            r'(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n)*)', re.MULTILINE)
        for match in table_pattern.finditer(text):
            pages[0]["tables"].append(match.group(0).strip())

        return {
            "name": path.stem,
            "pages": pages,
            "total_pages": 1,
        }

    def _parse_html(self, path: Path) -> dict:
        """Parse HTML preserving h1-h6 headings and tables."""
        text = path.read_text(encoding="utf-8", errors="replace")

        # Remove navigation, footer, sidebar and other non-content elements
        for tag in ("nav", "footer", "header", "aside", "menu"):
            text = re.sub(
                rf'<{tag}[\s>].*?</{tag}>', '', text, flags=re.S | re.I)
        # Remove divs with class/id containing common non-content names
        text = re.sub(
            r'<div[^>]+(?:class|id)=["\'][^"\']*'
            r'(?:footer|sidebar|menu|cookie|banner|navigation|breadcrumb)'
            r'[^"\']*["\'][^>]*>.*?</div>',
            '', text, flags=re.S | re.I)

        pages = [{"number": 1, "sections": [], "tables": []}]
        current_sections = pages[0]["sections"]

        # Extract headings and content between them
        heading_pattern = re.compile(
            r'<h([1-6])[^>]*>(.*?)</h\1>', re.S | re.I)
        parts = heading_pattern.split(text)

        # parts: [before_h1, level, title, after_h1_before_h2, level, title, ...]
        i = 0
        if parts and not re.match(r'^[1-6]$', parts[0]):
            # Content before first heading
            clean = self._strip_html(parts[0])
            if clean.strip():
                current_sections.append({
                    "level": 0, "title": "", "content": clean.strip(),
                })
            i = 1

        while i + 2 < len(parts):
            level = int(parts[i])
            title = self._strip_html(parts[i + 1]).strip()
            content = self._strip_html(parts[i + 2]).strip()
            current_sections.append({
                "level": level, "title": title, "content": content,
            })
            i += 3

        # If no headings found, treat as plain text
        if not current_sections:
            clean = self._strip_html(text)
            if clean.strip():
                current_sections.append({
                    "level": 0, "title": "", "content": clean.strip(),
                })

        # Extract tables
        table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.S | re.I)
        for match in table_pattern.finditer(text):
            table_text = self._strip_html(match.group(0))
            if table_text.strip():
                pages[0]["tables"].append(table_text.strip())

        return {
            "name": path.stem,
            "pages": pages,
            "total_pages": 1,
        }

    def _parse_plain_text(self, path: Path) -> dict:
        """Parse plain text with paragraph detection."""
        text = path.read_text(encoding="utf-8", errors="replace")
        pages = [{"number": 1, "sections": [], "tables": []}]

        # Split on double newlines as "sections"
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            para = para.strip()
            if para:
                pages[0]["sections"].append({
                    "level": 0,
                    "title": "",
                    "content": para,
                })

        return {
            "name": path.stem,
            "pages": pages,
            "total_pages": 1,
        }

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags and decode entities."""
        text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', html,
                       flags=re.S | re.I)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                              ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
            text = text.replace(entity, char)
        return text

    # ═══════════════════════════════════════
    # SEMANTIC CHUNKING
    # ═══════════════════════════════════════

    def _chunk_structured(self, parsed: dict, doc_name: str) -> list[KBChunk]:
        """Hierarchical chunking: section → subsection → paragraphs."""
        chunks = []
        chunk_idx = 0

        # Build section path stack
        heading_stack = []  # [(level, title)]

        for page in parsed.get("pages", []):
            page_num = page.get("number", 0)

            # Process sections
            for section in page.get("sections", []):
                level = section.get("level", 0)
                title = section.get("title", "")
                content = section.get("content", "")

                # Update heading stack
                if title and level > 0:
                    # Remove deeper or same-level headings
                    while heading_stack and heading_stack[-1][0] >= level:
                        heading_stack.pop()
                    heading_stack.append((level, title))

                section_path = " > ".join(t for _, t in heading_stack) if heading_stack else ""

                # Build full section text (include title)
                full_text = f"{title}\n{content}" if title else content
                full_text = full_text.strip()
                if not full_text:
                    continue

                # If section fits in one chunk
                if len(full_text) <= self.chunk_size:
                    chunks.append(KBChunk(
                        id=str(uuid.uuid4()),
                        doc_id="",  # set during ingestion
                        content=full_text,
                        section_path=section_path,
                        page_start=page_num,
                        page_end=page_num,
                        chunk_type="text",
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1
                else:
                    # Split large section by paragraphs
                    sub_chunks = self._split_with_overlap(
                        full_text, self.chunk_size, self.chunk_overlap)
                    for sc in sub_chunks:
                        chunks.append(KBChunk(
                            id=str(uuid.uuid4()),
                            doc_id="",
                            content=sc,
                            section_path=section_path,
                            page_start=page_num,
                            page_end=page_num,
                            chunk_type="text",
                            chunk_index=chunk_idx,
                        ))
                        chunk_idx += 1

            # Process tables as separate chunks
            for table_text in page.get("tables", []):
                if table_text.strip():
                    chunks.append(KBChunk(
                        id=str(uuid.uuid4()),
                        doc_id="",
                        content=table_text.strip(),
                        section_path=" > ".join(
                            t for _, t in heading_stack) if heading_stack else "",
                        page_start=page_num,
                        page_end=page_num,
                        chunk_type="table",
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1

        return chunks

    def _split_with_overlap(self, text: str, chunk_size: int,
                             overlap: int) -> list[str]:
        """Split text by paragraphs with overlap."""
        paragraphs = re.split(r'\n\s*\n|\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        for para in paragraphs:
            candidate = f"{current}\n{para}".strip() if current else para
            if len(candidate) > chunk_size and current:
                chunks.append(current.strip())
                # Overlap: keep tail of previous chunk
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + "\n" + para
                else:
                    current = para
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text.strip()]

    def _create_parent_child_chunks(self, parents: list[KBChunk]) -> list[KBChunk]:
        """Split parent chunks into smaller children for precise matching."""
        all_chunks = []
        child_size = 600  # ~200 tokens
        child_overlap = 100

        for parent in parents:
            parent.chunk_type = "parent"
            all_chunks.append(parent)

            # Subdivide parent into children
            text = parent.content
            if len(text) <= child_size:
                # Small enough — create single child pointing to parent
                child = KBChunk(
                    id=str(uuid.uuid4()),
                    doc_id=parent.doc_id,
                    content=text,
                    section_path=parent.section_path,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    chunk_type="child",
                    chunk_index=parent.chunk_index * 100,
                    parent_id=parent.id,
                )
                all_chunks.append(child)
            else:
                # Split into overlapping children
                start = 0
                child_idx = 0
                while start < len(text):
                    end = start + child_size
                    child_text = text[start:end].strip()
                    if child_text:
                        child = KBChunk(
                            id=str(uuid.uuid4()),
                            doc_id=parent.doc_id,
                            content=child_text,
                            section_path=parent.section_path,
                            page_start=parent.page_start,
                            page_end=parent.page_end,
                            chunk_type="child",
                            chunk_index=parent.chunk_index * 100 + child_idx,
                            parent_id=parent.id,
                        )
                        all_chunks.append(child)
                        child_idx += 1
                    start = end - child_overlap

        return all_chunks

    # ═══════════════════════════════════════
    # INGESTION
    # ═══════════════════════════════════════

    async def ingest(self, path: str, metadata: dict | None = None) -> dict:
        """Ingest a book/document into the knowledge base."""
        import asyncio

        p = Path(path).expanduser().resolve()
        # Compute hash for deduplication
        content_bytes = p.read_bytes()
        doc_hash = hashlib.md5(content_bytes).hexdigest()

        # Check dedup
        existing = self.db.execute(
            "SELECT id, name, chunk_count FROM kb_documents WHERE doc_hash = ?",
            (doc_hash,)).fetchone()
        if existing:
            return {
                "status": "already_exists",
                "doc_id": existing["id"],
                "name": existing["name"],
                "chunks": existing["chunk_count"],
            }

        # Parse
        parsed = await asyncio.to_thread(self._parse_file, str(p))
        doc_name = metadata.get("name", parsed["name"]) if metadata else parsed["name"]

        # Chunk
        chunks = self._chunk_structured(parsed, doc_name)
        if not chunks:
            return {"status": "error", "message": "No content extracted"}

        # Parent-child chunking
        if self._parent_child:
            chunks = self._create_parent_child_chunks(chunks)

        actual_chunk_count = len([c for c in chunks if c.chunk_type != "parent"])

        # Create document record
        doc_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO kb_documents (id, name, source_path, doc_hash, "
            "page_count, chunk_count, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc_id, doc_name, str(p), doc_hash, parsed["total_pages"],
             actual_chunk_count, json.dumps(metadata or {}, ensure_ascii=False)))

        # Insert chunks + FTS + embeddings
        for chunk in chunks:
            chunk.doc_id = doc_id
            embedding_blob = None

            # Contextual enrichment
            if self._contextual_retrieval and self._provider and chunk.chunk_type != "parent":
                chunk.context_prefix = await self._generate_context_prefix(chunk, doc_name)

            # Compute embedding — include context_prefix if available
            if self._embedder and chunk.chunk_type != "parent":
                try:
                    embed_text = chunk.content[:2000]
                    if chunk.context_prefix:
                        embed_text = f"{chunk.context_prefix}\n\n{embed_text}"
                    vec = await asyncio.to_thread(
                        self._embedder.encode, embed_text[:2000])
                    embedding_blob = pickle.dumps(vec)
                except Exception as e:
                    logger.debug("Embedding failed for chunk %s: %s",
                                 chunk.chunk_index, e)

            self.db.execute(
                "INSERT INTO kb_chunks "
                "(id, doc_id, content, section_path, page_start, page_end, "
                "chunk_type, chunk_index, embedding, context_prefix, parent_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (chunk.id, doc_id, chunk.content, chunk.section_path,
                 chunk.page_start, chunk.page_end, chunk.chunk_type,
                 chunk.chunk_index, embedding_blob, chunk.context_prefix,
                 getattr(chunk, 'parent_id', None)))

            # FTS5 index — include context_prefix for BM25
            if self._fts_available and chunk.chunk_type != "parent":
                try:
                    fts_content = chunk.content
                    if chunk.context_prefix:
                        fts_content = f"{chunk.context_prefix} {fts_content}"
                    self.db.execute(
                        "INSERT INTO kb_fts (content, chunk_id, doc_name, "
                        "section_path) VALUES (?, ?, ?, ?)",
                        (fts_content, chunk.id, doc_name, chunk.section_path))
                except Exception:
                    pass

        self.db.commit()
        logger.info("Ingested '%s': %d pages, %d chunks",
                     doc_name, parsed["total_pages"], actual_chunk_count)

        return {
            "status": "ok",
            "doc_id": doc_id,
            "name": doc_name,
            "pages": parsed["total_pages"],
            "chunks": actual_chunk_count,
        }

    # ═══════════════════════════════════════
    # SEARCH
    # ═══════════════════════════════════════

    async def search(self, query: str, top_k: int = 6,
                     mode: str | None = None) -> list[KBSearchResult]:
        """Full retrieval pipeline: rewrite → hybrid search → rerank."""
        import asyncio
        t0 = time.monotonic()
        mode = mode or self._search_mode

        # Step 1: Query rewriting
        queries = [query]
        if self._query_rewrite_enabled and self._provider:
            try:
                rewritten = await self._rewrite_query(query)
                if rewritten:
                    queries = [query] + rewritten
            except Exception as e:
                logger.debug("Query rewrite failed: %s", e)

        # Step 2: Search with each sub-query
        all_results = {}  # chunk_id → best result dict
        for q in queries:
            results = self._search_single(q, top_k=top_k * 5, mode=mode)
            for r in results:
                cid = r["chunk_id"]
                if cid not in all_results or r["score"] > all_results[cid]["score"]:
                    all_results[cid] = r

        # Merge and sort
        merged = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)

        # Step 3: Rerank
        if self._rerank_enabled and len(merged) > top_k:
            try:
                merged = await self._rerank(query, merged, top_k=top_k)
            except Exception as e:
                logger.debug("Rerank failed: %s", e)
                merged = merged[:top_k]
        else:
            merged = merged[:top_k]

        # Convert to KBSearchResult
        results = []
        for r in merged:
            # Load context_prefix from DB
            cp_row = self.db.execute(
                "SELECT context_prefix FROM kb_chunks WHERE id = ?",
                (r["chunk_id"],)).fetchone()
            results.append(KBSearchResult(
                content=r["content"],
                score=r["score"],
                source=r.get("doc_name", ""),
                page=r.get("page_start", 0),
                section=r.get("section_path", ""),
                chunk_id=r["chunk_id"],
                context_prefix=cp_row["context_prefix"] if cp_row and cp_row["context_prefix"] else "",
            ))

        # Self-RAG / CRAG correction
        if self._self_rag and self._provider and results:
            try:
                results = await self._crag_correction(
                    query, results, top_k, mode)
            except Exception as e:
                logger.debug("Self-RAG correction failed: %s", e)

        # Quality metrics (Phase 6: RAGAS)
        metrics = self._compute_quality_metrics(query, results)

        # Log query
        latency_ms = int((time.monotonic() - t0) * 1000)
        self._log_query(query, queries, [r.chunk_id for r in results], latency_ms, metrics)

        return results

    def _search_single(self, query: str, top_k: int = 30,
                        mode: str = "hybrid") -> list[dict]:
        """Single-query search with BM25 + vector + entity graph + RRF merge."""
        bm25_results = []
        vector_results = []

        if mode in ("hybrid", "bm25") and self._fts_available:
            bm25_results = self._bm25_search(query, top_k=top_k)

        if mode in ("hybrid", "vector") and self._embedder:
            vector_results = self._vector_search(query, top_k=top_k)

        # Entity graph search (Phase 5)
        entity_results = []
        try:
            entity_results = self._entity_search(query, top_k=top_k)
        except Exception:
            pass  # Entity tables may not exist yet

        # Parent-child resolution
        if self._parent_child:
            if mode == "bm25":
                return self._resolve_parents(bm25_results[:top_k])
            if mode == "vector":
                return self._resolve_parents(vector_results[:top_k])
            # Hybrid
            if bm25_results and vector_results:
                lists = [bm25_results, vector_results]
                if entity_results:
                    lists.append(entity_results)
                return self._resolve_parents(self._rrf_merge(*lists, k=60)[:top_k])
            base = (bm25_results or vector_results)[:top_k]
            if entity_results:
                return self._resolve_parents(self._rrf_merge(base, entity_results, k=60)[:top_k])
            return self._resolve_parents(base)

        if mode == "bm25":
            return bm25_results[:top_k]
        if mode == "vector":
            return vector_results[:top_k]

        # Hybrid: RRF merge
        if bm25_results and vector_results:
            lists = [bm25_results, vector_results]
            if entity_results:
                lists.append(entity_results)
            return self._rrf_merge(*lists, k=60)[:top_k]
        base = (bm25_results or vector_results)[:top_k]
        if entity_results:
            return self._rrf_merge(base, entity_results, k=60)[:top_k]
        return base

    def _bm25_search(self, query: str, top_k: int = 50) -> list[dict]:
        """BM25 search via SQLite FTS5."""
        if not self._fts_available:
            return []
        # Escape FTS5 special characters
        safe_query = re.sub(r'[^\w\s]', ' ', query).strip()
        if not safe_query:
            return []
        # Use OR between words for broader matching
        terms = safe_query.split()
        fts_query = " OR ".join(terms)

        try:
            rows = self.db.execute("""
                SELECT f.chunk_id, f.doc_name, f.section_path,
                       c.content, c.page_start, c.page_end, c.chunk_type,
                       bm25(kb_fts) as score
                FROM kb_fts f
                JOIN kb_chunks c ON c.id = f.chunk_id
                WHERE kb_fts MATCH ?
                ORDER BY score
                LIMIT ?
            """, (fts_query, top_k)).fetchall()
        except Exception as e:
            logger.debug("FTS5 search error: %s", e)
            return []

        results = []
        for row in rows:
            results.append({
                "chunk_id": row["chunk_id"],
                "doc_name": row["doc_name"],
                "section_path": row["section_path"],
                "content": row["content"],
                "page_start": row["page_start"],
                "page_end": row["page_end"],
                "chunk_type": row["chunk_type"],
                # BM25 scores are negative in FTS5, lower = better
                "score": -row["score"],
            })
        return results

    def _vector_search(self, query: str, top_k: int = 50) -> list[dict]:
        """Vector similarity search via embedder."""
        if not self._embedder:
            return []

        try:
            query_vec = self._embedder.encode(query)
        except Exception as e:
            logger.debug("Embedding query failed: %s", e)
            return []

        # Brute-force cosine similarity over stored embeddings
        rows = self.db.execute("""
            SELECT c.id, c.content, c.section_path, c.page_start, c.page_end,
                   c.chunk_type, c.embedding, d.name as doc_name
            FROM kb_chunks c
            JOIN kb_documents d ON d.id = c.doc_id
            WHERE c.embedding IS NOT NULL
        """).fetchall()

        scored = []
        for row in rows:
            try:
                stored_vec = pickle.loads(row["embedding"])
                score = self._cosine_similarity(query_vec, stored_vec)
                scored.append({
                    "chunk_id": row["id"],
                    "doc_name": row["doc_name"],
                    "section_path": row["section_path"],
                    "content": row["content"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "chunk_type": row["chunk_type"],
                    "score": float(score),
                })
            except Exception:
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Cosine similarity between two vectors."""
        import numpy as np
        a = np.asarray(a, dtype="float32")
        b = np.asarray(b, dtype="float32")
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    @staticmethod
    def _rrf_merge(*result_lists, k: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion to merge multiple ranked lists."""
        scores = {}  # chunk_id → {score, data}
        for rlist in result_lists:
            for rank, item in enumerate(rlist):
                cid = item["chunk_id"]
                rrf_score = 1.0 / (k + rank + 1)
                if cid in scores:
                    scores[cid]["score"] += rrf_score
                else:
                    scores[cid] = {**item, "score": rrf_score}
        merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return merged

    def _entity_search(self, query: str, top_k: int = 20) -> list[dict]:
        """Search via entity graph — find chunks mentioning entities from query."""
        # Extract potential entity patterns from query
        entities_in_query = []

        # Articles: ст. 123
        for m in re.finditer(r'(?:ст(?:атья|\.)\s*(\d+(?:\.\d+)*))', query, re.I):
            entities_in_query.append(f"ст. {m.group(1)}")

        # Laws: N 44-ФЗ
        for m in re.finditer(r'(?:[NН№]\s*(\d+[-–]?(?:ФЗ|фз)))', query, re.I):
            entities_in_query.append(f"№{m.group(1)}")

        # Also try to find entities by name match
        words = [w for w in query.split() if len(w) > 3]

        results = []
        seen_chunks = set()

        # Search by extracted entities
        for entity_name in entities_in_query:
            rows = self.db.execute("""
                SELECT DISTINCT c.id, c.content, c.section_path, c.page_start,
                       c.page_end, c.chunk_type, d.name as doc_name
                FROM kb_entities e
                JOIN kb_entity_mentions m ON m.entity_id = e.id
                JOIN kb_chunks c ON c.id = m.chunk_id
                JOIN kb_documents d ON d.id = c.doc_id
                WHERE e.name = ? AND c.chunk_type != 'parent'
                LIMIT ?
            """, (entity_name, top_k)).fetchall()

            for row in rows:
                cid = row["id"]
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    results.append({
                        "chunk_id": cid,
                        "doc_name": row["doc_name"],
                        "section_path": row["section_path"],
                        "content": row["content"],
                        "page_start": row["page_start"],
                        "page_end": row["page_end"],
                        "chunk_type": row["chunk_type"],
                        "score": 1.0,  # High confidence for exact entity match
                    })

        # Search by keyword match on entity names
        if not results and words:
            for word in words[:5]:
                rows = self.db.execute("""
                    SELECT DISTINCT c.id, c.content, c.section_path, c.page_start,
                           c.page_end, c.chunk_type, d.name as doc_name
                    FROM kb_entities e
                    JOIN kb_entity_mentions m ON m.entity_id = e.id
                    JOIN kb_chunks c ON c.id = m.chunk_id
                    JOIN kb_documents d ON d.id = c.doc_id
                    WHERE e.name LIKE ? AND c.chunk_type != 'parent'
                    LIMIT ?
                """, (f"%{word}%", top_k)).fetchall()

                for row in rows:
                    cid = row["id"]
                    if cid not in seen_chunks:
                        seen_chunks.add(cid)
                        results.append({
                            "chunk_id": cid,
                            "doc_name": row["doc_name"],
                            "section_path": row["section_path"],
                            "content": row["content"],
                            "page_start": row["page_start"],
                            "page_end": row["page_end"],
                            "chunk_type": row["chunk_type"],
                            "score": 0.5,  # Lower for fuzzy match
                        })

        return results[:top_k]

    def _resolve_parents(self, results: list[dict]) -> list[dict]:
        """Replace child content with parent content for richer context."""
        resolved = []
        seen_parents = set()

        for r in results:
            chunk_row = self.db.execute(
                "SELECT parent_id, chunk_type FROM kb_chunks WHERE id = ?",
                (r["chunk_id"],)).fetchone()

            if not chunk_row or chunk_row["chunk_type"] != "child" or not chunk_row["parent_id"]:
                resolved.append(r)
                continue

            parent_id = chunk_row["parent_id"]
            if parent_id in seen_parents:
                continue  # Deduplicate by parent
            seen_parents.add(parent_id)

            parent = self.db.execute(
                "SELECT content, section_path, page_start, page_end FROM kb_chunks WHERE id = ?",
                (parent_id,)).fetchone()
            if parent:
                r = dict(r)
                r["content"] = parent["content"]
                r["section_path"] = parent["section_path"]
                r["page_start"] = parent["page_start"]
                r["page_end"] = parent["page_end"]
            resolved.append(r)

        return resolved

    # ═══════════════════════════════════════
    # RERANKING
    # ═══════════════════════════════════════

    def _get_cross_encoder(self):
        """Lazy-load cross-encoder reranker."""
        if self._cross_encoder_checked:
            return self._cross_encoder
        self._cross_encoder_checked = True
        try:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._rerank_model)
            logger.info("Cross-encoder loaded: %s", self._rerank_model)
        except Exception as e:
            logger.debug("Cross-encoder not available: %s", e)
            self._cross_encoder = None
        return self._cross_encoder

    async def _rerank(self, query: str, results: list[dict],
                      top_k: int = 6) -> list[dict]:
        """Rerank results using cross-encoder or LLM fallback."""
        import asyncio

        # Try cross-encoder first
        encoder = await asyncio.to_thread(self._get_cross_encoder)
        if encoder:
            return await self._rerank_cross_encoder(query, results, top_k, encoder)

        # LLM fallback
        if self._provider:
            return await self._rerank_llm(query, results, top_k)

        # No reranker available
        return results[:top_k]

    async def _rerank_cross_encoder(self, query: str, results: list[dict],
                                     top_k: int, encoder) -> list[dict]:
        """Rerank using sentence-transformers CrossEncoder."""
        import asyncio

        pairs = [(query, r["content"][:1000]) for r in results]
        scores = await asyncio.to_thread(encoder.predict, pairs)

        for i, score in enumerate(scores):
            results[i]["score"] = float(score)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def _rerank_llm(self, query: str, results: list[dict],
                           top_k: int) -> list[dict]:
        """Rerank using LLM scoring (fallback when no cross-encoder)."""
        # Score in batches of 5 to reduce LLM calls
        batch_size = 5
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            prompt = self._build_rerank_prompt(query, batch)
            try:
                messages = [{"role": "user", "content": prompt}]
                resp = await self._provider.complete(
                    messages=messages, max_tokens=200)
                scores = self._parse_rerank_scores(resp.text, len(batch))
                for j, score in enumerate(scores):
                    results[i + j]["score"] = score
            except Exception as e:
                logger.debug("LLM rerank batch failed: %s", e)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    @staticmethod
    def _build_rerank_prompt(query: str, batch: list[dict]) -> str:
        parts = [f'Rate relevance 0-10 for query "{query}":\n']
        for i, item in enumerate(batch):
            text = item["content"][:300]
            parts.append(f"[{i}] {text}")
        parts.append("\nReturn ONLY a JSON array of scores, e.g. [8, 3, 7, 1, 5]")
        return "\n".join(parts)

    @staticmethod
    def _parse_rerank_scores(text: str, count: int) -> list[float]:
        """Parse LLM rerank response into scores."""
        match = re.search(r'\[[\d,\s.]+\]', text)
        if match:
            try:
                scores = json.loads(match.group(0))
                if len(scores) == count:
                    return [float(s) for s in scores]
            except (json.JSONDecodeError, ValueError):
                pass
        # Fallback: return uniform scores
        return [5.0] * count

    # ═══════════════════════════════════════
    # QUERY REWRITING
    # ═══════════════════════════════════════

    async def _rewrite_query(self, query: str) -> list[str]:
        """Decompose user query into 2-4 search sub-queries via LLM."""
        if not self._provider:
            return []

        prompt = (
            "You are a search query optimizer. Decompose the following user question "
            "into 2-4 specific search sub-queries that together cover the full intent. "
            "Return ONLY a JSON array of strings, no explanation.\n\n"
            f'User question: "{query}"\n\n'
            "Sub-queries:"
        )
        try:
            messages = [{"role": "user", "content": prompt}]
            resp = await self._provider.complete(
                messages=messages, max_tokens=300)
            match = re.search(r'\[.*\]', resp.text, re.S)
            if match:
                sub_queries = json.loads(match.group(0))
                if isinstance(sub_queries, list) and all(
                        isinstance(q, str) for q in sub_queries):
                    return sub_queries[:4]
        except Exception as e:
            logger.debug("Query rewrite failed: %s", e)
        return []

    # ═══════════════════════════════════════
    # SELF-RAG / CRAG
    # ═══════════════════════════════════════

    async def _crag_rewrite(self, query: str, low_results: list[dict]) -> str:
        """LLM rewrites query based on low-relevance results."""
        if not self._provider:
            return query

        snippets = "\n".join(
            f"- {r['content'][:200]}" for r in low_results[:3])

        prompt = (
            "The following search query returned low-relevance results:\n"
            f"Query: \"{query}\"\n\n"
            f"Results (low relevance):\n{snippets}\n\n"
            "Rewrite the query to better capture the user's intent. "
            "Return ONLY the rewritten query, nothing else."
        )
        try:
            resp = await self._provider.complete(
                messages=[{"role": "user", "content": prompt}], max_tokens=150)
            rewritten = resp.text.strip().strip('"\'')
            return rewritten if rewritten else query
        except Exception:
            return query

    async def _crag_correction(self, query: str, results: list[KBSearchResult],
                                top_k: int, mode: str,
                                max_iterations: int = 2) -> list[KBSearchResult]:
        """Self-RAG: rewrite query if confidence is low, re-search."""
        import asyncio

        for iteration in range(max_iterations):
            # Compute average score
            if not results:
                break
            avg_score = sum(r.score for r in results[:top_k]) / min(len(results), top_k)

            if avg_score >= self._self_rag_threshold:
                break  # Confidence is sufficient

            logger.debug("Self-RAG iteration %d: avg_score=%.3f < threshold=%.3f",
                         iteration + 1, avg_score, self._self_rag_threshold)

            # Convert results to dicts for _crag_rewrite
            result_dicts = [{"content": r.content} for r in results[:3]]
            rewritten_query = await self._crag_rewrite(query, result_dicts)

            if rewritten_query == query:
                break  # No improvement possible

            # Re-search with rewritten query
            new_results_raw = self._search_single(rewritten_query, top_k=top_k * 5, mode=mode)

            # Rerank if available
            if self._rerank_enabled and len(new_results_raw) > top_k:
                try:
                    new_results_raw = await self._rerank(rewritten_query, new_results_raw, top_k=top_k)
                except Exception:
                    new_results_raw = new_results_raw[:top_k]
            else:
                new_results_raw = new_results_raw[:top_k]

            # Convert to KBSearchResult
            new_results = []
            for r in new_results_raw:
                cp_row = self.db.execute(
                    "SELECT context_prefix FROM kb_chunks WHERE id = ?",
                    (r["chunk_id"],)).fetchone()
                new_results.append(KBSearchResult(
                    content=r["content"],
                    score=r["score"],
                    source=r.get("doc_name", ""),
                    page=r.get("page_start", 0),
                    section=r.get("section_path", ""),
                    chunk_id=r["chunk_id"],
                    context_prefix=cp_row["context_prefix"] if cp_row and cp_row["context_prefix"] else "",
                ))

            # Merge: keep the best from both sets
            all_by_id = {}
            for r in results + new_results:
                if r.chunk_id not in all_by_id or r.score > all_by_id[r.chunk_id].score:
                    all_by_id[r.chunk_id] = r
            results = sorted(all_by_id.values(), key=lambda x: x.score, reverse=True)[:top_k]
            query = rewritten_query  # Use rewritten for next iteration

        return results

    # ═══════════════════════════════════════
    # CONTEXT BUILDER
    # ═══════════════════════════════════════

    def build_context(self, results: list[KBSearchResult]) -> str:
        """Build structured context with citations, grouped by document/section."""
        if not results:
            return "В базе знаний релевантной информации не найдено."

        # Group by document
        by_doc = {}
        seen_chunks = set()
        for r in results:
            if r.chunk_id in seen_chunks:
                continue
            seen_chunks.add(r.chunk_id)
            doc_key = r.source or "Unknown"
            if doc_key not in by_doc:
                by_doc[doc_key] = []
            by_doc[doc_key].append(r)

        parts = []
        for doc_name, doc_results in by_doc.items():
            parts.append(f"## Источник: \"{doc_name}\"")
            for r in doc_results:
                citation = f"[стр. {r.page}" if r.page else "["
                if r.section:
                    citation += f", раздел \"{r.section}\""
                citation += "]"
                parts.append(f"\n{citation}")
                if r.context_prefix:
                    parts.append(f"*Контекст: {r.context_prefix}*")
                parts.append(r.content)
            parts.append("")

        return "\n".join(parts)

    # ═══════════════════════════════════════
    # MANAGEMENT
    # ═══════════════════════════════════════

    async def list_documents(self) -> list[dict]:
        """List all documents in the knowledge base."""
        rows = self.db.execute(
            "SELECT id, name, source_path, page_count, chunk_count, "
            "metadata_json, created_at FROM kb_documents "
            "ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks."""
        doc = self.db.execute(
            "SELECT id, name FROM kb_documents WHERE id = ? OR name = ?",
            (doc_id, doc_id)).fetchone()
        if not doc:
            return False

        real_id = doc["id"]
        # Get chunk IDs for FTS cleanup
        chunk_ids = [row["id"] for row in self.db.execute(
            "SELECT id FROM kb_chunks WHERE doc_id = ?",
            (real_id,)).fetchall()]

        # Delete FTS entries
        if self._fts_available and chunk_ids:
            placeholders = ",".join("?" for _ in chunk_ids)
            try:
                self.db.execute(
                    f"DELETE FROM kb_fts WHERE chunk_id IN ({placeholders})",
                    chunk_ids)
            except Exception:
                pass

        # Delete chunks and document
        self.db.execute("DELETE FROM kb_chunks WHERE doc_id = ?", (real_id,))
        self.db.execute("DELETE FROM kb_documents WHERE id = ?", (real_id,))
        self.db.commit()
        logger.info("Deleted document '%s' (%s)", doc["name"], real_id)
        return True

    async def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        doc_count = self.db.execute(
            "SELECT COUNT(*) FROM kb_documents").fetchone()[0]
        chunk_count = self.db.execute(
            "SELECT COUNT(*) FROM kb_chunks").fetchone()[0]
        embedded_count = self.db.execute(
            "SELECT COUNT(*) FROM kb_chunks WHERE embedding IS NOT NULL"
        ).fetchone()[0]

        # DB file size
        db_path = self._config.get("db_path", "~/.liteagent/knowledge_base.db")
        db_path = os.path.expanduser(db_path)
        size_mb = 0.0
        if os.path.exists(db_path):
            size_mb = os.path.getsize(db_path) / (1024 * 1024)

        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "embedded_chunks": embedded_count,
            "fts_available": self._fts_available,
            "embedder": self._embedder.name if self._embedder else None,
            "rerank_available": bool(self._get_cross_encoder()) if self._rerank_enabled else False,
            "search_mode": self._search_mode,
            "storage_size_mb": round(size_mb, 2),
        }

    async def list_entities(self, doc_id: str | None = None,
                            limit: int = 50) -> list[dict]:
        """List entities, optionally filtered by document."""
        try:
            if doc_id:
                rows = self.db.execute(
                    "SELECT e.id, e.name, e.entity_type, e.doc_id, e.count, "
                    "d.name as doc_name "
                    "FROM kb_entities e LEFT JOIN kb_documents d ON d.id = e.doc_id "
                    "WHERE e.doc_id = ? ORDER BY e.count DESC LIMIT ?",
                    (doc_id, limit)).fetchall()
            else:
                rows = self.db.execute(
                    "SELECT e.id, e.name, e.entity_type, e.doc_id, e.count, "
                    "d.name as doc_name "
                    "FROM kb_entities e LEFT JOIN kb_documents d ON d.id = e.doc_id "
                    "ORDER BY e.count DESC LIMIT ?",
                    (limit,)).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            return []  # Tables may not exist

    # ═══════════════════════════════════════
    # QUALITY LOGGING
    # ═══════════════════════════════════════

    def _log_query(self, query: str, rewritten: list[str],
                    chunks_used: list[str], latency_ms: int,
                    metrics: dict | None = None):
        """Log query for quality analysis."""
        try:
            faith = metrics.get("faithfulness") if metrics else None
            relev = metrics.get("relevancy") if metrics else None
            prec = metrics.get("context_precision") if metrics else None
            self.db.execute(
                "INSERT INTO kb_query_log "
                "(query, rewritten_queries, chunks_retrieved, result_count, "
                "latency_ms, faithfulness_score, relevancy_score, context_precision) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (query, json.dumps(rewritten, ensure_ascii=False),
                 json.dumps(chunks_used), len(chunks_used), latency_ms,
                 faith, relev, prec))
            self.db.commit()
        except Exception:
            pass

    def _compute_quality_metrics(self, query: str,
                                  results: list[KBSearchResult]) -> dict:
        """Compute heuristic quality metrics (RAGAS-inspired)."""
        if not results:
            return {"faithfulness": 0.0, "relevancy": 0.0, "context_precision": 0.0}

        query_words = set(query.lower().split())

        # Faithfulness: how well do results match query terms?
        faith_scores = []
        for r in results:
            content_words = set(r.content.lower().split())
            overlap = len(query_words & content_words)
            faith_scores.append(min(overlap / max(len(query_words), 1), 1.0))
        faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else 0.0

        # Relevancy: based on retrieval scores (normalized)
        scores = [r.score for r in results]
        max_score = max(scores) if scores else 1.0
        if max_score > 0:
            relevancy = sum(s / max_score for s in scores) / len(scores)
        else:
            relevancy = 0.0

        # Context precision: ratio of top-k results with above-average score
        if len(scores) > 1:
            avg_score = sum(scores) / len(scores)
            above_avg = sum(1 for s in scores if s >= avg_score)
            context_precision = above_avg / len(scores)
        else:
            context_precision = 1.0 if scores else 0.0

        return {
            "faithfulness": round(faithfulness, 3),
            "relevancy": round(relevancy, 3),
            "context_precision": round(context_precision, 3),
        }

    async def get_quality_stats(self, limit: int = 100) -> dict:
        """Get aggregated quality metrics from recent queries."""
        try:
            rows = self.db.execute(
                "SELECT faithfulness_score, relevancy_score, context_precision "
                "FROM kb_query_log "
                "WHERE faithfulness_score IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()

            if not rows:
                return {"queries_analyzed": 0,
                        "avg_faithfulness": None,
                        "avg_relevancy": None,
                        "avg_context_precision": None}

            faith = [r["faithfulness_score"] for r in rows if r["faithfulness_score"] is not None]
            relev = [r["relevancy_score"] for r in rows if r["relevancy_score"] is not None]
            prec = [r["context_precision"] for r in rows if r["context_precision"] is not None]

            return {
                "queries_analyzed": len(rows),
                "avg_faithfulness": round(sum(faith) / len(faith), 3) if faith else None,
                "avg_relevancy": round(sum(relev) / len(relev), 3) if relev else None,
                "avg_context_precision": round(sum(prec) / len(prec), 3) if prec else None,
            }
        except Exception:
            return {"queries_analyzed": 0,
                    "avg_faithfulness": None,
                    "avg_relevancy": None,
                    "avg_context_precision": None}

    # ═══════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════

    def close(self):
        """Close database connection."""
        try:
            self.db.close()
        except Exception:
            pass
