"""Tests for metacognition module: confidence gate, counterfactual replay, dream cycle."""

import json
import math
import pickle
import sqlite3
import pytest
from datetime import datetime, timedelta
from array import array

from liteagent.metacognition import (
    log_interaction, run_counterfactual_replay, _find_clusters,
)


class SimpleVec:
    """Picklable vector-like object with @ (matmul) support for testing."""
    def __init__(self, data):
        self.data = list(data)
    def __matmul__(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))


def _make_embedding(values):
    """Create a pickle-serializable embedding for tests."""
    return SimpleVec(values)


@pytest.fixture
def meta_db(tmp_path):
    """Create DB with feature tables."""
    db = sqlite3.connect(str(tmp_path / "meta.db"))
    db.executescript("""
        CREATE TABLE interaction_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            user_input TEXT,
            agent_response TEXT,
            tool_calls_json TEXT,
            success INTEGER DEFAULT 1,
            confidence REAL,
            model_used TEXT,
            created_at TEXT
        );
        CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            type TEXT DEFAULT 'fact',
            importance REAL DEFAULT 0.5,
            hash TEXT UNIQUE,
            created_at TEXT,
            accessed_at TEXT,
            embedding BLOB
        );
        CREATE TABLE user_state (
            user_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            updated_at TEXT,
            PRIMARY KEY (user_id, key)
        );
        CREATE TABLE app_state (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        );
    """)
    return db


class TestConfidenceGate:
    def test_log_interaction(self, meta_db):
        log_interaction(meta_db, "u1", "hello", "hi there",
                        [{"name": "test"}], 1, 8.5, "haiku")
        rows = meta_db.execute("SELECT * FROM interaction_log").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "u1"
        assert rows[0][5] == 1  # success
        assert rows[0][6] == 8.5  # confidence

    def test_log_interaction_truncates(self, meta_db):
        long_input = "x" * 5000
        log_interaction(meta_db, "u1", long_input, "response",
                        [], 1, None, "sonnet")
        row = meta_db.execute(
            "SELECT user_input FROM interaction_log").fetchone()
        assert len(row[0]) == 2000

    def test_log_failure(self, meta_db):
        log_interaction(meta_db, "u1", "bad query", "wrong answer",
                        [], 0, 3.0, "haiku")
        row = meta_db.execute(
            "SELECT success, confidence FROM interaction_log").fetchone()
        assert row[0] == 0
        assert row[1] == 3.0


class TestCounterfactualReplay:
    @pytest.mark.asyncio
    async def test_no_failures_returns_zero(self, meta_db):
        """When no failures exist, replay returns 0."""
        class MockMemory:
            def __init__(self, db):
                self.db = db
            def get_state(self, key):
                row = self.db.execute(
                    "SELECT value FROM app_state WHERE key=?", (key,)
                ).fetchone()
                return json.loads(row[0]) if row else None
            def set_state(self, key, value):
                self.db.execute(
                    "INSERT OR REPLACE INTO app_state VALUES (?, ?, ?)",
                    (key, json.dumps(value), datetime.now().isoformat()))
                self.db.commit()
            def remember(self, *args, **kwargs):
                pass

        memory = MockMemory(meta_db)
        count = await run_counterfactual_replay(
            None, meta_db, memory, {"max_replays_per_run": 5})
        assert count == 0


class TestDreamCycle:
    def test_find_clusters_empty(self):
        """No rows = no clusters."""
        clusters = _find_clusters([], 0.85, None)
        assert clusters == []

    def test_find_clusters_similar(self):
        """Two identical embeddings should cluster."""
        vec = _make_embedding([1.0, 0.0, 0.0])
        emb = pickle.dumps(vec)

        class MockMemory:
            @staticmethod
            def _cosine_similarity(a, b):
                dot = float(a @ b)
                na = float(math.sqrt(a @ a))
                nb = float(math.sqrt(b @ b))
                return dot / (na * nb) if na and nb else 0.0

        rows = [
            (1, "fact A", emb, 0.5),
            (2, "fact A similar", emb, 0.5),
            (3, "fact C different",
             pickle.dumps(_make_embedding([0.0, 1.0, 0.0])), 0.5),
        ]
        clusters = _find_clusters(rows, 0.85, MockMemory())
        assert len(clusters) == 1
        assert set(clusters[0][0]) == {1, 2}

    def test_find_clusters_below_threshold(self):
        """Dissimilar embeddings should not cluster."""
        class MockMemory:
            @staticmethod
            def _cosine_similarity(a, b):
                dot = float(a @ b)
                na = float(math.sqrt(a @ a))
                nb = float(math.sqrt(b @ b))
                return dot / (na * nb) if na and nb else 0.0

        rows = [
            (1, "cat", pickle.dumps(_make_embedding([1.0, 0.0])), 0.5),
            (2, "dog", pickle.dumps(_make_embedding([0.0, 1.0])), 0.5),
        ]
        clusters = _find_clusters(rows, 0.85, MockMemory())
        assert clusters == []

    def test_importance_decay(self, meta_db):
        """Verify importance decay SQL works."""
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        meta_db.execute(
            "INSERT INTO memories VALUES (NULL, 'u1', 'old fact', 'fact', "
            "0.5, 'h1', ?, ?, NULL)",
            (old_date, old_date))
        meta_db.commit()

        meta_db.execute(
            """UPDATE memories SET importance = MAX(importance - 0.05, 0.05)
               WHERE accessed_at < datetime('now', '-7 days')
               AND importance > 0.1""")
        meta_db.commit()

        row = meta_db.execute(
            "SELECT importance FROM memories WHERE id=1").fetchone()
        assert row[0] == pytest.approx(0.45, abs=0.01)
