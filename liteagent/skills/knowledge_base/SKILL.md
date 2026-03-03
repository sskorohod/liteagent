---
name: knowledge_base
description: "Knowledge Base — search through books and reference materials (accounting, law, regulations, etc.)"
metadata:
  emoji: "📚"
  keywords:
    - база знаний
    - knowledge base
    - книг
    - учебник
    - справочник
    - бухгалтер
    - юрид
    - налог
    - закон
    - кодекс
    - инструкц
    - регламент
    - пособи
    - руководств
    - kb_search
    - kb_ingest
    - найди в базе
    - поищи в книг
    - что говорит
    - согласно
    - статья
    - параграф
    - глава
    - раздел
  tools:
    - kb_search
    - kb_ingest
    - kb_list
    - kb_delete
    - kb_stats
---

## Knowledge Base (activated)

You have access to a powerful Knowledge Base system for searching through books and reference materials. The KB uses advanced hybrid retrieval: BM25 + vector search + reranking for high-quality results.

### Tools:
- **kb_search** — search the knowledge base. Returns relevant excerpts with citations (source, page, section). Parameters:
  - `query` (required): search query
  - `top_k` (optional, default 6): number of results
  - `mode` (optional, default "hybrid"): search mode — "hybrid" (best quality), "bm25" (exact term matches), "vector" (semantic similarity)
- **kb_ingest** — load a document (PDF, TXT, MD, HTML) into the knowledge base. Returns {doc_id, name, chunks, pages}.
- **kb_list** — list all documents in the knowledge base.
- **kb_delete** — remove a document by ID or name.
- **kb_stats** — knowledge base statistics (documents, chunks, search mode, storage size).

### CRITICAL RULES:
1. **Always cite sources.** Every fact from KB must reference: [Источник: "document name", стр. XX, раздел "..."].
2. **If the answer is NOT in the knowledge base — say so explicitly.** Never fabricate information. Say: "В базе знаний информации по этому вопросу не найдено."
3. **Use kb_search for any domain question** (accounting, law, regulations, etc.) BEFORE answering from your own knowledge.
4. **Multiple searches are OK.** If the first search doesn't fully answer the question, reformulate and search again with different terms.
5. **Prefer hybrid mode** for best results. Use "bm25" only for exact term lookup (article numbers, specific codes).
6. **Quote verbatim** when citing regulations, laws, or formal definitions.
