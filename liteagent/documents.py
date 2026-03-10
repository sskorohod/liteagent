"""Document intelligence pipeline for uploads, storage, review, notes, and reminders."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import uuid
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

from .file_types import detect_file_type, extract_text_from_file
from .multimodal import file_to_content_block

logger = logging.getLogger(__name__)

_DATE_RX = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"
    r"|\d{2}\.\d{2}\.\d{4}"
    r"|\d{2}/\d{2}/\d{4})\b"
)
_REMINDER_HINTS = (
    "expire", "expires", "expiration", "renew", "renewal", "deadline",
    "due", "submit", "submission", "extend", "продлить", "истекает",
    "срок", "дедлайн", "подать", "отправить", "renew before",
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class DocumentPipeline:
    """Coordinates storage, KB indexing, review, notes, tasks, and calendar events."""

    def __init__(self, agent):
        self.agent = agent
        self.db: sqlite3.Connection = agent.memory.db
        self.config = agent.config.get("documents", {}) if isinstance(agent.config, dict) else {}
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_reviews (
                review_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                original_name TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'upload',
                mime_type TEXT DEFAULT '',
                size_bytes INTEGER DEFAULT 0,
                file_hash TEXT DEFAULT '',
                local_path TEXT DEFAULT '',
                storage_key TEXT DEFAULT '',
                kb_doc_id TEXT DEFAULT '',
                title TEXT DEFAULT '',
                doc_type TEXT DEFAULT '',
                summary TEXT DEFAULT '',
                content_excerpt TEXT DEFAULT '',
                key_points_json TEXT DEFAULT '[]',
                important_notes_json TEXT DEFAULT '[]',
                entities_json TEXT DEFAULT '[]',
                dates_json TEXT DEFAULT '[]',
                actions_json TEXT DEFAULT '[]',
                reminders_json TEXT DEFAULT '[]',
                tasks_json TEXT DEFAULT '[]',
                calendar_json TEXT DEFAULT '[]',
                metadata_json TEXT DEFAULT '{}',
                analysis_json TEXT DEFAULT '{}',
                notes_saved INTEGER DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'ok',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_document_reviews_user
                ON document_reviews(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_document_reviews_storage
                ON document_reviews(storage_key, created_at DESC);

            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'document',
                source_ref TEXT DEFAULT '',
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                event_at TEXT NOT NULL,
                remind_at TEXT DEFAULT '',
                status TEXT NOT NULL DEFAULT 'scheduled',
                metadata_json TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_calendar_events_user
                ON calendar_events(user_id, event_at DESC);
            CREATE INDEX IF NOT EXISTS idx_calendar_events_ref
                ON calendar_events(source_ref, event_at DESC);
            """
        )
        self.db.commit()

    async def process_upload(
        self,
        data: bytes,
        filename: str,
        *,
        user_id: str,
        source: str = "dashboard",
        mime_type: str = "",
    ) -> dict[str, Any]:
        """Store and analyze an uploaded document, then create notes and reminders."""
        canonical_user = self.agent.memory.get_canonical_person_id(user_id)
        safe_name = self._safe_filename(filename)
        info = detect_file_type(data, safe_name, mime_type)
        review_id = str(uuid.uuid4())
        file_hash = hashlib.md5(data).hexdigest()
        local_path = self._persist_local_copy(data, safe_name, review_id)

        storage_result = await self._store_original(data, safe_name, source, canonical_user, mime_type)
        kb_result = await self._ingest_to_kb(local_path, safe_name, canonical_user, source, storage_result)

        extracted_text = ""
        if getattr(info, "can_extract_text", False):
            with suppress_exception():
                extracted_text = extract_text_from_file(data, info)
        multimodal_summary = await self._extract_from_scan_if_needed(data, safe_name, mime_type, extracted_text)
        analysis = await self._analyze_document(
            filename=safe_name,
            file_info=info,
            extracted_text=extracted_text,
            multimodal_summary=multimodal_summary,
        )

        file_meta = {
            "review_id": review_id,
            "filename": safe_name,
            "storage_key": storage_result.get("storage_key", ""),
            "kb_doc_id": kb_result.get("doc_id", ""),
            "source": source,
        }
        notes_saved = await self._store_notes(canonical_user, analysis, file_meta)
        tasks_created = self._create_tasks(canonical_user, review_id, analysis)
        calendar_created = self._create_calendar_events(canonical_user, review_id, analysis, safe_name)

        summary = {
            "review_id": review_id,
            "status": "ok",
            "name": safe_name,
            "mime_type": info.mime_type or mime_type or "application/octet-stream",
            "size_bytes": len(data),
            "storage": storage_result,
            "knowledge_base": kb_result,
            "analysis": analysis,
            "notes_saved": notes_saved,
            "tasks_created": tasks_created,
            "calendar_events": calendar_created,
            "local_path": local_path,
        }
        self._persist_review(
            review_id=review_id,
            user_id=canonical_user,
            original_name=safe_name,
            source=source,
            mime_type=summary["mime_type"],
            size_bytes=len(data),
            file_hash=file_hash,
            local_path=local_path,
            storage_key=storage_result.get("storage_key", ""),
            kb_doc_id=kb_result.get("doc_id", ""),
            analysis=analysis,
            summary=summary,
            notes_saved=notes_saved,
        )
        return summary

    def list_reviews(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        uid = self.agent.memory.get_canonical_person_id(user_id)
        rows = self.db.execute(
            """
            SELECT review_id, original_name, title, doc_type, summary, storage_key,
                   kb_doc_id, created_at, notes_saved, tasks_json, calendar_json
            FROM document_reviews
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (uid, max(1, min(int(limit), 200))),
        ).fetchall()
        return [
            {
                "review_id": r[0],
                "original_name": r[1],
                "title": r[2],
                "doc_type": r[3],
                "summary": r[4],
                "storage_key": r[5],
                "kb_doc_id": r[6],
                "created_at": r[7],
                "notes_saved": int(r[8] or 0),
                "tasks_created": self._load_json_list(r[9]),
                "calendar_events": self._load_json_list(r[10]),
            }
            for r in rows
        ]

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        row = self.db.execute(
            """
            SELECT review_id, user_id, original_name, source, mime_type, size_bytes,
                   file_hash, local_path, storage_key, kb_doc_id, title, doc_type,
                   summary, content_excerpt, key_points_json, important_notes_json,
                   entities_json, dates_json, actions_json, reminders_json,
                   tasks_json, calendar_json, metadata_json, analysis_json, notes_saved,
                   status, created_at, updated_at
            FROM document_reviews WHERE review_id = ?
            """,
            (review_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "review_id": row[0],
            "user_id": row[1],
            "original_name": row[2],
            "source": row[3],
            "mime_type": row[4],
            "size_bytes": int(row[5] or 0),
            "file_hash": row[6],
            "local_path": row[7],
            "storage_key": row[8],
            "kb_doc_id": row[9],
            "title": row[10],
            "doc_type": row[11],
            "summary": row[12],
            "content_excerpt": row[13],
            "key_points": self._load_json_list(row[14]),
            "important_notes": self._load_json_list(row[15]),
            "entities": self._load_json_list(row[16]),
            "dates": self._load_json_list(row[17]),
            "actions": self._load_json_list(row[18]),
            "reminders": self._load_json_list(row[19]),
            "tasks_created": self._load_json_list(row[20]),
            "calendar_events": self._load_json_list(row[21]),
            "metadata": self._load_json_dict(row[22]),
            "analysis": self._load_json_dict(row[23]),
            "notes_saved": int(row[24] or 0),
            "status": row[25],
            "created_at": row[26],
            "updated_at": row[27],
        }

    def list_calendar_events(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        uid = self.agent.memory.get_canonical_person_id(user_id)
        rows = self.db.execute(
            """
            SELECT id, title, description, event_at, remind_at, status, source_ref, metadata_json, created_at
            FROM calendar_events
            WHERE user_id = ?
            ORDER BY event_at ASC, created_at DESC
            LIMIT ?
            """,
            (uid, max(1, min(int(limit), 200))),
        ).fetchall()
        return [
            {
                "id": int(r[0]),
                "title": r[1],
                "description": r[2],
                "event_at": r[3],
                "remind_at": r[4],
                "status": r[5],
                "source_ref": r[6],
                "metadata": self._load_json_dict(r[7]),
                "created_at": r[8],
            }
            for r in rows
        ]

    async def _store_original(
        self,
        data: bytes,
        filename: str,
        source: str,
        user_id: str,
        mime_type: str,
    ) -> dict[str, Any]:
        fm = getattr(self.agent, "_file_manager", None)
        if not fm:
            return {"enabled": False, "stored": False, "storage_key": "", "message": "Storage disabled"}
        try:
            stored = await fm.ingest(
                data,
                filename,
                source=source,
                user_id=user_id,
                mime_type=mime_type,
                description=f"Document upload: {filename}",
            )
            return {
                "enabled": True,
                "stored": True,
                "storage_key": stored.get("storage_key", ""),
                "mime_type": stored.get("mime_type", ""),
                "description": stored.get("description", ""),
            }
        except Exception as exc:
            logger.warning("Document storage failed for %s: %s", filename, exc)
            return {"enabled": True, "stored": False, "storage_key": "", "message": str(exc)}

    async def _ingest_to_kb(
        self,
        local_path: str,
        filename: str,
        user_id: str,
        source: str,
        storage_result: dict[str, Any],
    ) -> dict[str, Any]:
        kb = getattr(self.agent, "_knowledge_base", None)
        if not kb:
            return {"enabled": False, "status": "disabled", "doc_id": "", "chunks": 0}
        try:
            result = await kb.ingest(
                local_path,
                metadata={
                    "name": filename,
                    "uploaded_via": source,
                    "storage_key": storage_result.get("storage_key", ""),
                    "user_id": user_id,
                },
            )
            return {
                "enabled": True,
                "status": result.get("status", "ok"),
                "doc_id": result.get("doc_id", ""),
                "name": result.get("name", filename),
                "pages": int(result.get("pages", 0) or 0),
                "chunks": int(result.get("chunks", 0) or 0),
            }
        except Exception as exc:
            logger.warning("KB ingest failed for %s: %s", filename, exc)
            return {"enabled": True, "status": "error", "doc_id": "", "chunks": 0, "message": str(exc)}

    async def _extract_from_scan_if_needed(
        self,
        data: bytes,
        filename: str,
        mime_type: str,
        extracted_text: str,
    ) -> str:
        needs_scan_help = len((extracted_text or "").strip()) < 160
        if not needs_scan_help:
            return ""
        try:
            block = file_to_content_block(data, filename, mime_type)
        except Exception:
            return ""
        if block.get("type") not in {"document", "image"}:
            return ""
        prompt = self.agent._build_media_understanding_prompt(
            "Extract the main content of this document scan and list any dates, obligations, people, numbers, and expiry information.",
            media_kind="document" if block.get("type") == "document" else "image",
            media_label="Document",
            media_index=1,
        )
        try:
            result = await self.agent._complete_multimodal_with_fallback_meta(
                [{"type": "text", "text": prompt}, block],
                max_tokens=self.agent._media_understanding_config().get("max_tokens_document", 2000),
                mode="document" if block.get("type") == "document" else "image",
            )
            return str(result.get("text") or "").strip()
        except Exception as exc:
            logger.debug("Scan analysis fallback failed for %s: %s", filename, exc)
            return ""

    async def _analyze_document(
        self,
        *,
        filename: str,
        file_info,
        extracted_text: str,
        multimodal_summary: str,
    ) -> dict[str, Any]:
        analysis = {}
        if getattr(self.agent, "provider", None):
            with suppress_exception():
                analysis = await self._analyze_document_with_llm(
                    filename=filename,
                    file_info=file_info,
                    extracted_text=extracted_text,
                    multimodal_summary=multimodal_summary,
                )
        if not analysis:
            analysis = self._heuristic_analysis(filename, file_info, extracted_text, multimodal_summary)
        return self._normalize_analysis(analysis, filename, file_info, extracted_text, multimodal_summary)

    async def _analyze_document_with_llm(
        self,
        *,
        filename: str,
        file_info,
        extracted_text: str,
        multimodal_summary: str,
    ) -> dict[str, Any]:
        model = (
            str(self.agent.config.get("agent", {}).get("document_model", "")).strip()
            or str(self.agent.models.get("medium", "") or self.agent.default_model)
        )
        text_window = (extracted_text or "").strip()[:18000]
        visual_window = (multimodal_summary or "").strip()[:6000]
        prompt = (
            "Analyze this user document for long-term organization and reminders.\n"
            "Return JSON only with keys:\n"
            "title, document_type, language, summary, key_points, important_notes, "
            "entities, dates, actions, reminders, note_candidates, tags.\n"
            "Rules:\n"
            "- summary: 2-4 sentences\n"
            "- key_points and important_notes: concise bullet-style strings\n"
            "- entities: objects with name, type, role\n"
            "- dates: objects with label, raw_date, normalized_date, kind, confidence\n"
            "- actions: objects with title, reason, due_at, query\n"
            "- reminders: only when the document clearly implies a deadline, expiration, renewal, submission, follow-up, or preparation step; "
            "objects with title, due_at, remind_at, reason, query\n"
            "- note_candidates: important durable facts worth saving to notes\n"
            "- If data is missing, use empty strings/lists, not prose.\n\n"
            f"Filename: {filename}\n"
            f"Detected type: {getattr(file_info, 'label', '')}\n"
            f"Detected mime: {getattr(file_info, 'mime_type', '')}\n\n"
        )
        if text_window:
            prompt += f"Extracted text:\n{text_window}\n\n"
        if visual_window:
            prompt += f"Scan/vision extraction:\n{visual_window}\n\n"
        response = await self.agent.provider.complete(
            model=model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        text = self.agent._extract_text(response)
        return self.agent._extract_json_object(text)

    def _heuristic_analysis(
        self,
        filename: str,
        file_info,
        extracted_text: str,
        multimodal_summary: str,
    ) -> dict[str, Any]:
        combined = "\n".join(part for part in [multimodal_summary, extracted_text] if part).strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", combined) if p.strip()]
        first_line = ""
        for line in combined.splitlines():
            line = line.strip()
            if len(line) >= 4:
                first_line = line
                break
        summary = (paragraphs[0] if paragraphs else first_line or filename)[:500]
        key_points = []
        for item in paragraphs[:4]:
            short = re.sub(r"\s+", " ", item)[:180].strip()
            if short:
                key_points.append(short)
        dates = []
        for raw in _DATE_RX.findall(combined[:8000]):
            normalized = self._normalize_date_string(raw)
            if not normalized:
                continue
            line = self._line_for_value(combined, raw)
            lower = line.lower()
            kind = "deadline" if any(marker in lower for marker in _REMINDER_HINTS) else "mentioned"
            dates.append({
                "label": line[:120] or raw,
                "raw_date": raw,
                "normalized_date": normalized,
                "kind": kind,
                "confidence": 0.56 if kind == "deadline" else 0.42,
            })
        reminders = []
        for item in dates:
            if item["kind"] != "deadline":
                continue
            remind_at = self._offset_date(item["normalized_date"], days=-7) or item["normalized_date"]
            reminders.append({
                "title": f"Check {filename}",
                "due_at": item["normalized_date"],
                "remind_at": remind_at,
                "reason": item["label"],
                "query": f"Напомни мне про документ {filename}: {item['label']}",
            })
        return {
            "title": first_line[:120] or Path(filename).stem,
            "document_type": getattr(file_info, "label", "") or Path(filename).suffix.lstrip("."),
            "language": "",
            "summary": summary,
            "key_points": key_points,
            "important_notes": key_points[:3],
            "entities": [],
            "dates": dates,
            "actions": [],
            "reminders": reminders,
            "note_candidates": key_points[:3],
            "tags": [Path(filename).suffix.lstrip(".") or "document"],
        }

    def _normalize_analysis(
        self,
        analysis: dict[str, Any],
        filename: str,
        file_info,
        extracted_text: str,
        multimodal_summary: str,
    ) -> dict[str, Any]:
        if not isinstance(analysis, dict):
            analysis = {}
        summary = str(analysis.get("summary") or "").strip()
        if not summary:
            combined = "\n".join(part for part in [multimodal_summary, extracted_text] if part).strip()
            summary = re.sub(r"\s+", " ", combined)[:500] or f"Document uploaded: {filename}"
        title = str(analysis.get("title") or "").strip() or Path(filename).stem
        doc_type = str(analysis.get("document_type") or "").strip() or getattr(file_info, "label", "") or "document"
        key_points = self._clean_string_list(analysis.get("key_points"))
        important_notes = self._clean_string_list(analysis.get("important_notes"))
        note_candidates = self._clean_string_list(analysis.get("note_candidates"))
        if not note_candidates:
            note_candidates = important_notes[:]
        if not important_notes:
            important_notes = note_candidates[:]
        dates = self._clean_object_list(analysis.get("dates"), ("label", "raw_date", "normalized_date", "kind", "confidence"))
        reminders = self._clean_object_list(analysis.get("reminders"), ("title", "due_at", "remind_at", "reason", "query"))
        actions = self._clean_object_list(analysis.get("actions"), ("title", "reason", "due_at", "query"))
        entities = self._clean_object_list(analysis.get("entities"), ("name", "type", "role"))
        tags = self._clean_string_list(analysis.get("tags"))
        for reminder in reminders:
            reminder["due_at"] = self._normalize_date_string(str(reminder.get("due_at") or ""))
            reminder["remind_at"] = self._normalize_date_string(str(reminder.get("remind_at") or "")) or reminder["due_at"]
        for action in actions:
            action["due_at"] = self._normalize_date_string(str(action.get("due_at") or ""))
        normalized = {
            "title": title[:160],
            "document_type": doc_type[:80],
            "language": str(analysis.get("language") or "").strip()[:32],
            "summary": summary[:1200],
            "key_points": key_points[:8],
            "important_notes": important_notes[:8],
            "entities": entities[:12],
            "dates": dates[:12],
            "actions": actions[:8],
            "reminders": reminders[:8],
            "note_candidates": note_candidates[:8],
            "tags": tags[:10],
            "content_excerpt": re.sub(r"\s+", " ", (extracted_text or multimodal_summary or ""))[:1200],
        }
        return normalized

    async def _store_notes(self, user_id: str, analysis: dict[str, Any], file_meta: dict[str, Any]) -> int:
        notes = []
        title = analysis.get("title") or file_meta.get("filename") or "document"
        summary = str(analysis.get("summary") or "").strip()
        if summary:
            notes.append(f"Document '{title}': {summary}")
        for item in analysis.get("note_candidates", []):
            text = str(item or "").strip()
            if text:
                notes.append(f"{title}: {text}")
        count = 0
        encoded_meta = _json_dumps(file_meta)
        for note in notes[:10]:
            saved = await self.agent.memory.remember(note, user_id, "fact", 0.72, file_meta=encoded_meta)
            if saved:
                count += 1
        return count

    def _create_tasks(self, user_id: str, review_id: str, analysis: dict[str, Any]) -> list[dict[str, Any]]:
        tm = getattr(self.agent, "_task_manager", None)
        if not tm:
            return []
        created: list[dict[str, Any]] = []
        reminders = list(analysis.get("reminders") or [])
        actions = list(analysis.get("actions") or [])
        for item in reminders + actions:
            title = str(item.get("title") or "").strip()
            run_at = str(item.get("remind_at") or item.get("due_at") or "").strip()
            query = str(item.get("query") or "").strip()
            if not title or not run_at or not query:
                continue
            if not self._task_time_is_future(run_at):
                continue
            if self._task_exists(user_id, title, query, run_at):
                continue
            try:
                task = tm.add_task(
                    name=title[:120],
                    query=query[:1500],
                    user_id=user_id,
                    task_type="one_shot",
                    run_at=run_at,
                    source="document-review",
                )
            except Exception as exc:
                logger.debug("Document task creation failed for %s: %s", title, exc)
                continue
            created.append(
                {
                    "id": task.get("id"),
                    "name": task.get("name"),
                    "run_at": task.get("run_at"),
                    "source": "document-review",
                }
            )
        return created

    def _create_calendar_events(
        self,
        user_id: str,
        review_id: str,
        analysis: dict[str, Any],
        filename: str,
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        seen = set()
        for item in list(analysis.get("reminders") or []) + list(analysis.get("actions") or []):
            title = str(item.get("title") or "").strip()[:180]
            event_at = str(item.get("due_at") or "").strip()
            if not title or not event_at or not self._task_time_is_future(event_at):
                continue
            key = (title.lower(), event_at)
            if key in seen:
                continue
            seen.add(key)
            remind_at = str(item.get("remind_at") or "").strip()
            description = str(item.get("reason") or filename).strip()[:1000]
            if self._calendar_event_exists(user_id, review_id, title, event_at):
                continue
            meta = {"review_id": review_id, "filename": filename}
            cur = self.db.execute(
                """
                INSERT INTO calendar_events (user_id, source, source_ref, title, description, event_at, remind_at, metadata_json)
                VALUES (?, 'document', ?, ?, ?, ?, ?, ?)
                """,
                (user_id, review_id, title, description, event_at, remind_at, _json_dumps(meta)),
            )
            created.append(
                {
                    "id": int(cur.lastrowid),
                    "title": title,
                    "event_at": event_at,
                    "remind_at": remind_at,
                }
            )
        self.db.commit()
        return created

    def _persist_review(
        self,
        *,
        review_id: str,
        user_id: str,
        original_name: str,
        source: str,
        mime_type: str,
        size_bytes: int,
        file_hash: str,
        local_path: str,
        storage_key: str,
        kb_doc_id: str,
        analysis: dict[str, Any],
        summary: dict[str, Any],
        notes_saved: int,
    ) -> None:
        self.db.execute(
            """
            INSERT OR REPLACE INTO document_reviews (
                review_id, user_id, original_name, source, mime_type, size_bytes, file_hash,
                local_path, storage_key, kb_doc_id, title, doc_type, summary, content_excerpt,
                key_points_json, important_notes_json, entities_json, dates_json, actions_json,
                reminders_json, tasks_json, calendar_json, metadata_json, analysis_json,
                notes_saved, status, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                review_id,
                user_id,
                original_name,
                source,
                mime_type,
                size_bytes,
                file_hash,
                local_path,
                storage_key,
                kb_doc_id,
                analysis.get("title", ""),
                analysis.get("document_type", ""),
                analysis.get("summary", ""),
                analysis.get("content_excerpt", ""),
                _json_dumps(analysis.get("key_points", [])),
                _json_dumps(analysis.get("important_notes", [])),
                _json_dumps(analysis.get("entities", [])),
                _json_dumps(analysis.get("dates", [])),
                _json_dumps(analysis.get("actions", [])),
                _json_dumps(analysis.get("reminders", [])),
                _json_dumps(summary.get("tasks_created", [])),
                _json_dumps(summary.get("calendar_events", [])),
                _json_dumps({
                    "storage": summary.get("storage", {}),
                    "knowledge_base": summary.get("knowledge_base", {}),
                }),
                _json_dumps(analysis),
                int(notes_saved),
                "ok",
            ),
        )
        self.db.commit()

    def _persist_local_copy(self, data: bytes, filename: str, review_id: str) -> str:
        base = Path.home() / ".liteagent" / "document_uploads"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{review_id[:12]}_{filename}"
        path.write_bytes(data)
        return str(path)

    def _safe_filename(self, filename: str) -> str:
        name = Path(str(filename or "document.bin")).name
        name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip("._ ")
        return name or "document.bin"

    def _task_exists(self, user_id: str, title: str, query: str, run_at: str) -> bool:
        row = self.db.execute(
            """
            SELECT id FROM tasks
            WHERE user_id = ? AND name = ? AND query = ? AND run_at = ?
              AND source = 'document-review'
              AND status IN ('pending', 'running')
            LIMIT 1
            """,
            (user_id, title[:120], query[:1500], run_at),
        ).fetchone()
        return bool(row)

    def _calendar_event_exists(self, user_id: str, review_id: str, title: str, event_at: str) -> bool:
        row = self.db.execute(
            """
            SELECT id FROM calendar_events
            WHERE user_id = ? AND source_ref = ? AND title = ? AND event_at = ?
            LIMIT 1
            """,
            (user_id, review_id, title, event_at),
        ).fetchone()
        return bool(row)

    def _task_time_is_future(self, raw: str) -> bool:
        dt = self._parse_datetime(raw)
        if not dt:
            return False
        return dt > datetime.now() + timedelta(minutes=1)

    @staticmethod
    def _load_json_list(raw: Any) -> list[Any]:
        if isinstance(raw, list):
            return raw
        try:
            data = json.loads(str(raw or "[]"))
        except Exception:
            return []
        return data if isinstance(data, list) else []

    @staticmethod
    def _load_json_dict(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        try:
            data = json.loads(str(raw or "{}"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _clean_string_list(value: Any) -> list[str]:
        out: list[str] = []
        raw_items = value if isinstance(value, list) else [value]
        for item in raw_items:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text[:280])
        return out

    @staticmethod
    def _clean_object_list(value: Any, preferred_keys: tuple[str, ...]) -> list[dict[str, Any]]:
        items = value if isinstance(value, list) else []
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            row = {}
            for key in preferred_keys:
                val = item.get(key)
                if val is None:
                    val = ""
                row[key] = val
            if any(str(v or "").strip() for v in row.values()):
                out.append(row)
        return out

    @staticmethod
    def _line_for_value(text: str, value: str) -> str:
        for line in text.splitlines():
            if value in line:
                return re.sub(r"\s+", " ", line).strip()
        return value

    @staticmethod
    def _normalize_date_string(raw: str) -> str:
        dt = DocumentPipeline._parse_datetime(raw)
        if not dt:
            return ""
        return dt.isoformat(timespec="minutes")

    @staticmethod
    def _offset_date(raw: str, *, days: int = 0) -> str:
        dt = DocumentPipeline._parse_datetime(raw)
        if not dt:
            return ""
        return (dt + timedelta(days=days)).isoformat(timespec="minutes")

    @staticmethod
    def _parse_datetime(raw: str) -> datetime | None:
        value = str(raw or "").strip()
        if not value:
            return None
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}$", value):
            try:
                return datetime.combine(datetime.strptime(value, "%Y-%m-%d").date(), time(9, 0))
            except ValueError:
                return None
        formats = (
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M",
            "%d.%m.%Y",
            "%d/%m/%Y",
            "%m/%d/%Y",
        )
        for fmt in formats:
            try:
                dt = datetime.strptime(value, fmt)
            except ValueError:
                continue
            if fmt in {"%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"}:
                return datetime.combine(dt.date(), time(9, 0))
            return dt
        return None


class suppress_exception:
    """Small context manager to keep the pipeline resilient."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc:
            logger.debug("Suppressed document pipeline exception: %s", exc)
        return True
