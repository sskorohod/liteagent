"""Tests for Phase 3: Graph memory — entities, relations, graph traversal, graph-enhanced recall."""

import json
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from liteagent.memory import MemorySystem


@pytest.fixture
def graph_config(tmp_db):
    """Memory config with graph memory enabled."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "keep_recent_messages": 6,
            "fts_enabled": True,
            "graph_memory": True,
            "max_graph_entities_in_context": 10,
        }
    }


@pytest.fixture
def graph_memory(graph_config):
    """MemorySystem with graph memory enabled."""
    ms = MemorySystem(graph_config)
    yield ms
    ms.close()


@pytest.fixture
def graph_disabled_config(tmp_db):
    """Memory config with graph memory disabled (default)."""
    return {
        "memory": {
            "db_path": tmp_db,
            "auto_learn": False,
            "graph_memory": False,
        }
    }


@pytest.fixture
def graph_disabled(graph_disabled_config):
    ms = MemorySystem(graph_disabled_config)
    yield ms
    ms.close()


# ══════════════════════════════════════════
# GRAPH SCHEMA
# ══════════════════════════════════════════

class TestGraphSchema:
    """Verify graph tables exist."""

    def test_entities_table_exists(self, graph_memory):
        row = graph_memory.db.execute(
            "SELECT name FROM sqlite_master WHERE name='memory_entities'"
        ).fetchone()
        assert row is not None

    def test_relations_table_exists(self, graph_memory):
        row = graph_memory.db.execute(
            "SELECT name FROM sqlite_master WHERE name='memory_relations'"
        ).fetchone()
        assert row is not None

    def test_mentions_table_exists(self, graph_memory):
        row = graph_memory.db.execute(
            "SELECT name FROM sqlite_master WHERE name='memory_entity_mentions'"
        ).fetchone()
        assert row is not None

    def test_tables_exist_when_disabled(self, graph_disabled):
        """Tables always created, feature gated by config check."""
        row = graph_disabled.db.execute(
            "SELECT name FROM sqlite_master WHERE name='memory_entities'"
        ).fetchone()
        assert row is not None


# ══════════════════════════════════════════
# FEATURE FLAG
# ══════════════════════════════════════════

class TestGraphFeatureFlag:
    def test_graph_enabled(self, graph_memory):
        assert graph_memory._graph_enabled() is True

    def test_graph_disabled_by_default(self, graph_disabled):
        assert graph_disabled._graph_enabled() is False


# ══════════════════════════════════════════
# ENTITY CRUD
# ══════════════════════════════════════════

class TestEntityCRUD:
    """Create, update, find, list entities."""

    def test_upsert_entity_new(self, graph_memory):
        eid = graph_memory.upsert_entity("Alice", "person", "u1")
        assert eid is not None
        assert len(eid) == 36  # UUID

    def test_upsert_entity_increments_count(self, graph_memory):
        eid = graph_memory.upsert_entity("Alice", "person", "u1")
        eid2 = graph_memory.upsert_entity("Alice", "person", "u1")
        assert eid == eid2  # Same entity
        ent = graph_memory.get_entity(eid)
        assert ent["mention_count"] == 2

    def test_upsert_entity_case_insensitive(self, graph_memory):
        eid1 = graph_memory.upsert_entity("Python", "tool", "u1")
        eid2 = graph_memory.upsert_entity("python", "tool", "u1")
        assert eid1 == eid2

    def test_upsert_entity_merges_properties(self, graph_memory):
        eid = graph_memory.upsert_entity("Alice", "person", "u1",
                                          properties={"role": "developer"})
        graph_memory.upsert_entity("Alice", "person", "u1",
                                    properties={"company": "Acme"})
        ent = graph_memory.get_entity(eid)
        assert ent["properties"]["role"] == "developer"
        assert ent["properties"]["company"] == "Acme"

    def test_upsert_empty_name_returns_empty(self, graph_memory):
        eid = graph_memory.upsert_entity("", "concept", "u1")
        assert eid == ""

    def test_find_entity(self, graph_memory):
        graph_memory.upsert_entity("ProjectX", "project", "u1")
        ent = graph_memory.find_entity("ProjectX", "u1")
        assert ent is not None
        assert ent["name"] == "ProjectX"
        assert ent["entity_type"] == "project"

    def test_find_entity_not_found(self, graph_memory):
        assert graph_memory.find_entity("Nonexistent", "u1") is None

    def test_find_entity_user_isolation(self, graph_memory):
        graph_memory.upsert_entity("Secret", "concept", "u1")
        assert graph_memory.find_entity("Secret", "u2") is None

    def test_get_entity(self, graph_memory):
        eid = graph_memory.upsert_entity("Bob", "person", "u1")
        ent = graph_memory.get_entity(eid)
        assert ent["name"] == "Bob"
        assert ent["entity_type"] == "person"
        assert ent["user_id"] == "u1"
        assert ent["mention_count"] == 1

    def test_get_entity_nonexistent(self, graph_memory):
        assert graph_memory.get_entity("nonexistent-id") is None

    def test_get_entities_list(self, graph_memory):
        graph_memory.upsert_entity("Alice", "person", "u1")
        graph_memory.upsert_entity("Bob", "person", "u1")
        graph_memory.upsert_entity("Python", "tool", "u1")
        entities = graph_memory.get_entities("u1")
        assert len(entities) == 3

    def test_get_entities_filter_by_type(self, graph_memory):
        graph_memory.upsert_entity("Alice", "person", "u1")
        graph_memory.upsert_entity("Python", "tool", "u1")
        people = graph_memory.get_entities("u1", entity_type="person")
        assert len(people) == 1
        assert people[0]["name"] == "Alice"

    def test_get_entities_ordered_by_mentions(self, graph_memory):
        graph_memory.upsert_entity("Rare", "concept", "u1")
        graph_memory.upsert_entity("Popular", "concept", "u1")
        graph_memory.upsert_entity("Popular", "concept", "u1")  # 2nd mention
        graph_memory.upsert_entity("Popular", "concept", "u1")  # 3rd mention
        entities = graph_memory.get_entities("u1")
        assert entities[0]["name"] == "Popular"

    def test_get_entities_user_isolation(self, graph_memory):
        graph_memory.upsert_entity("Private", "concept", "u1")
        assert len(graph_memory.get_entities("u2")) == 0


# ══════════════════════════════════════════
# RELATION CRUD
# ══════════════════════════════════════════

class TestRelationCRUD:
    """Create and query relations between entities."""

    def test_upsert_relation(self, graph_memory):
        graph_memory.upsert_entity("Alice", "person", "u1")
        graph_memory.upsert_entity("ProjectX", "project", "u1")
        rid = graph_memory.upsert_relation("Alice", "ProjectX", "works_on", "u1")
        assert rid is not None

    def test_upsert_relation_strengthens_weight(self, graph_memory):
        graph_memory.upsert_entity("Alice", "person", "u1")
        graph_memory.upsert_entity("Python", "tool", "u1")
        rid1 = graph_memory.upsert_relation("Alice", "Python", "uses", "u1")
        rid2 = graph_memory.upsert_relation("Alice", "Python", "uses", "u1")
        assert rid1 == rid2  # Same relation
        # Weight should have increased
        row = graph_memory.db.execute(
            "SELECT weight FROM memory_relations WHERE id = ?", (rid1,)).fetchone()
        assert row[0] == 1.5  # Started at 1.0, incremented by 0.5

    def test_upsert_relation_accumulates_evidence(self, graph_memory):
        graph_memory.upsert_entity("Bob", "person", "u1")
        graph_memory.upsert_entity("Go", "tool", "u1")
        graph_memory.upsert_relation("Bob", "Go", "uses", "u1", evidence="writes backend")
        graph_memory.upsert_relation("Bob", "Go", "uses", "u1", evidence="builds APIs")
        row = graph_memory.db.execute(
            "SELECT evidence FROM memory_relations WHERE user_id = 'u1'").fetchone()
        assert "writes backend" in row[0]
        assert "builds APIs" in row[0]

    def test_upsert_relation_missing_entity(self, graph_memory):
        graph_memory.upsert_entity("Alice", "person", "u1")
        rid = graph_memory.upsert_relation("Alice", "Unknown", "knows", "u1")
        assert rid is None

    def test_get_entity_relations(self, graph_memory):
        graph_memory.upsert_entity("Alice", "person", "u1")
        graph_memory.upsert_entity("ProjectX", "project", "u1")
        graph_memory.upsert_entity("Python", "tool", "u1")
        graph_memory.upsert_relation("Alice", "ProjectX", "works_on", "u1")
        graph_memory.upsert_relation("Alice", "Python", "uses", "u1")
        alice = graph_memory.find_entity("Alice", "u1")
        rels = graph_memory.get_entity_relations(alice["id"])
        assert len(rels) == 2
        rel_types = {r["relation_type"] for r in rels}
        assert "works_on" in rel_types
        assert "uses" in rel_types

    def test_get_entity_relations_empty(self, graph_memory):
        eid = graph_memory.upsert_entity("Lonely", "concept", "u1")
        rels = graph_memory.get_entity_relations(eid)
        assert rels == []


# ══════════════════════════════════════════
# ENTITY-MEMORY LINKING
# ══════════════════════════════════════════

class TestEntityMemoryLink:
    """Link entities to memories."""

    @pytest.mark.asyncio
    async def test_link_entity_to_memory(self, graph_memory):
        eid = graph_memory.upsert_entity("Python", "tool", "u1")
        await graph_memory.remember("User uses Python daily", "u1", "fact", 0.8)
        mems = graph_memory.get_all_memories("u1")
        mid = mems[0]["id"]
        graph_memory.link_entity_to_memory(eid, mid)
        # Verify link exists
        row = graph_memory.db.execute(
            "SELECT * FROM memory_entity_mentions WHERE entity_id = ? AND memory_id = ?",
            (eid, mid)).fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_link_duplicate_ignored(self, graph_memory):
        eid = graph_memory.upsert_entity("Python", "tool", "u1")
        await graph_memory.remember("Python fact", "u1", "fact", 0.5)
        mid = graph_memory.get_all_memories("u1")[0]["id"]
        graph_memory.link_entity_to_memory(eid, mid)
        graph_memory.link_entity_to_memory(eid, mid)  # Duplicate — no error
        count = graph_memory.db.execute(
            "SELECT COUNT(*) FROM memory_entity_mentions WHERE entity_id = ?",
            (eid,)).fetchone()[0]
        assert count == 1

    @pytest.mark.asyncio
    async def test_backfill_entity_mentions_links_existing_memories(self, graph_memory):
        eid = graph_memory.upsert_entity("Docker", "tool", "u1")
        await graph_memory.remember("User deploys services with Docker", "u1", "fact", 0.8)
        stats = graph_memory.backfill_entity_mentions("u1")
        assert stats["links_added"] >= 1
        count = graph_memory.db.execute(
            "SELECT COUNT(*) FROM memory_entity_mentions WHERE entity_id = ?",
            (eid,),
        ).fetchone()[0]
        assert count >= 1


# ══════════════════════════════════════════
# GRAPH TRAVERSAL
# ══════════════════════════════════════════

class TestGraphTraversal:
    """BFS neighborhood traversal."""

    def test_neighborhood_simple(self, graph_memory):
        graph_memory.upsert_entity("Alice", "person", "u1")
        graph_memory.upsert_entity("ProjectX", "project", "u1")
        graph_memory.upsert_relation("Alice", "ProjectX", "works_on", "u1")
        result = graph_memory.get_entity_neighborhood("Alice", "u1", hops=1)
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1
        names = {e["name"] for e in result["entities"]}
        assert "Alice" in names
        assert "ProjectX" in names

    def test_neighborhood_2_hops(self, graph_memory):
        graph_memory.upsert_entity("Alice", "person", "u1")
        graph_memory.upsert_entity("ProjectX", "project", "u1")
        graph_memory.upsert_entity("Python", "tool", "u1")
        graph_memory.upsert_relation("Alice", "ProjectX", "works_on", "u1")
        graph_memory.upsert_relation("ProjectX", "Python", "uses", "u1")
        result = graph_memory.get_entity_neighborhood("Alice", "u1", hops=2)
        names = {e["name"] for e in result["entities"]}
        assert "Python" in names  # Reached via 2 hops

    def test_neighborhood_nonexistent(self, graph_memory):
        result = graph_memory.get_entity_neighborhood("Nobody", "u1")
        assert result["entities"] == []
        assert result["relations"] == []

    def test_neighborhood_user_isolation(self, graph_memory):
        graph_memory.upsert_entity("Secret", "concept", "u1")
        result = graph_memory.get_entity_neighborhood("Secret", "u2")
        assert result["entities"] == []

    def test_neighborhood_deduplicates_relations(self, graph_memory):
        graph_memory.upsert_entity("A", "concept", "u1")
        graph_memory.upsert_entity("B", "concept", "u1")
        graph_memory.upsert_relation("A", "B", "related_to", "u1")
        result = graph_memory.get_entity_neighborhood("A", "u1", hops=2)
        # Each relation should appear only once
        rel_ids = [r["id"] for r in result["relations"]]
        assert len(rel_ids) == len(set(rel_ids))


# ══════════════════════════════════════════
# ENTITY SEARCH
# ══════════════════════════════════════════

class TestEntitySearch:
    """Search entities by name and embedding."""

    def test_search_by_substring(self, graph_memory):
        graph_memory.upsert_entity("Python", "tool", "u1")
        graph_memory.upsert_entity("JavaScript", "tool", "u1")
        results = graph_memory.entity_search("Python", "u1")
        assert len(results) >= 1
        assert results[0]["name"] == "Python"

    def test_search_case_insensitive(self, graph_memory):
        graph_memory.upsert_entity("FastAPI", "tool", "u1")
        results = graph_memory.entity_search("fastapi", "u1")
        assert len(results) >= 1

    def test_search_no_results(self, graph_memory):
        results = graph_memory.entity_search("nonexistent", "u1")
        assert results == []

    def test_search_user_isolation(self, graph_memory):
        graph_memory.upsert_entity("Private", "concept", "u1")
        results = graph_memory.entity_search("Private", "u2")
        assert results == []


# ══════════════════════════════════════════
# GRAPH RECALL (MEMORY SEARCH VIA GRAPH)
# ══════════════════════════════════════════

class TestGraphRecall:
    """Search memories through entity graph."""

    @pytest.mark.asyncio
    async def test_graph_recall_finds_linked_memories(self, graph_memory):
        eid = graph_memory.upsert_entity("Python", "tool", "u1")
        await graph_memory.remember("User prefers Python for web dev", "u1", "preference", 0.8)
        mid = graph_memory.get_all_memories("u1")[0]["id"]
        graph_memory.link_entity_to_memory(eid, mid)

        results = graph_memory._graph_recall("Python", "u1")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_graph_recall_disabled(self, graph_disabled):
        results = graph_disabled._graph_recall("anything", "u1")
        assert results == []

    @pytest.mark.asyncio
    async def test_graph_recall_no_entities(self, graph_memory):
        results = graph_memory._graph_recall("nonexistent", "u1")
        assert results == []


# ══════════════════════════════════════════
# ENTITY MERGING
# ══════════════════════════════════════════

class TestEntityMerging:
    """Merge duplicate entities."""

    def test_merge_entities(self, graph_memory):
        eid1 = graph_memory.upsert_entity("JS", "tool", "u1")
        eid2 = graph_memory.upsert_entity("JavaScript", "tool", "u1")
        graph_memory.upsert_entity("React", "tool", "u1")
        graph_memory.upsert_relation("JS", "React", "related_to", "u1")

        graph_memory.merge_entities([eid1, eid2], primary_name="JavaScript")

        # Secondary entity deleted
        assert graph_memory.get_entity(eid2) is None
        # Primary entity kept with updated name
        primary = graph_memory.get_entity(eid1)
        assert primary["name"] == "JavaScript"
        assert primary["mention_count"] == 2  # Summed from both

    def test_merge_single_entity_noop(self, graph_memory):
        eid = graph_memory.upsert_entity("Solo", "concept", "u1")
        graph_memory.merge_entities([eid])
        assert graph_memory.get_entity(eid) is not None

    def test_merge_redirects_relations(self, graph_memory):
        eid1 = graph_memory.upsert_entity("Py", "tool", "u1")
        eid2 = graph_memory.upsert_entity("Python", "tool", "u1")
        graph_memory.upsert_entity("Alice", "person", "u1")
        graph_memory.upsert_relation("Alice", "Python", "uses", "u1")

        graph_memory.merge_entities([eid1, eid2])

        # Relation should now point to primary entity
        rels = graph_memory.get_entity_relations(eid1)
        assert len(rels) == 1
        assert rels[0]["target_name"] == "Py" or rels[0]["source_name"] == "Py"


# ══════════════════════════════════════════
# GRAPH-ENHANCED RECALL (INTEGRATION)
# ══════════════════════════════════════════

class TestGraphEnhancedRecall:
    """Test that recall() integrates graph results."""

    @pytest.mark.asyncio
    async def test_recall_includes_graph_results(self, graph_memory):
        # Create entity + memory + link
        eid = graph_memory.upsert_entity("Docker", "tool", "u1")
        await graph_memory.remember("User runs services in Docker containers", "u1", "fact", 0.8)
        mid = graph_memory.get_all_memories("u1")[0]["id"]
        graph_memory.link_entity_to_memory(eid, mid)

        # Recall should find via graph even if keyword/FTS hits too
        results = graph_memory.recall("Docker", "u1")
        assert len(results) >= 1
        assert any("Docker" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_recall_works_without_graph(self, graph_disabled):
        await graph_disabled.remember("User likes coffee", "u1", "fact", 0.7)
        results = graph_disabled.recall("coffee", "u1")
        assert len(results) >= 1


# ══════════════════════════════════════════
# EXTRACT AND LEARN (ENTITY EXTRACTION)
# ══════════════════════════════════════════

class TestEntityExtraction:
    """Test entity/relation extraction in extract_and_learn()."""

    @pytest.mark.asyncio
    async def test_extract_entities_from_conversation(self, graph_memory):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"facts":["Alice works at Google"],'
                                                '"preferences":[],"corrections":[],'
                                                '"session_summary":"","entities":'
                                                '[{"name":"Alice","type":"person"},'
                                                '{"name":"Google","type":"concept"}],'
                                                '"relations":[{"source":"Alice",'
                                                '"target":"Google","type":"works_on"}]}')]
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        graph_memory.provider = mock_provider
        graph_memory.config["auto_learn"] = True  # Enable extraction

        await graph_memory.extract_and_learn(
            "I'm Alice and I work at Google",
            "Nice to meet you, Alice!",
            "u1")

        # Entities created
        alice = graph_memory.find_entity("Alice", "u1")
        assert alice is not None
        assert alice["entity_type"] == "person"

        google = graph_memory.find_entity("Google", "u1")
        assert google is not None

        # Relation created
        rels = graph_memory.get_entity_relations(alice["id"])
        assert len(rels) >= 1
        assert any(r["relation_type"] == "works_on" for r in rels)

        # Memory links created (entity -> memory mentions)
        mems = graph_memory.get_all_memories("u1")
        mem_id = next(m["id"] for m in mems if "Alice works at Google" in m["content"])
        links = graph_memory.db.execute(
            "SELECT entity_id FROM memory_entity_mentions WHERE memory_id = ?",
            (mem_id,),
        ).fetchall()
        linked_ids = {r[0] for r in links}
        assert alice["id"] in linked_ids
        assert google["id"] in linked_ids

    @pytest.mark.asyncio
    async def test_extract_no_entities_when_graph_disabled(self, graph_disabled):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"facts":["User likes Python"],'
                                                '"preferences":[],"corrections":[],'
                                                '"session_summary":""}')]
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        graph_disabled.provider = mock_provider
        graph_disabled.config["auto_learn"] = True

        await graph_disabled.extract_and_learn("I like Python", "Python is great!", "u1")

        # No entities should be created
        entities = graph_disabled.get_entities("u1")
        assert len(entities) == 0

    @pytest.mark.asyncio
    async def test_extract_uses_dedicated_extraction_provider(self, graph_disabled):
        # Main chat provider should NOT be used for memory extraction
        main_provider = AsyncMock()
        main_provider.complete = AsyncMock()
        graph_disabled.provider = main_provider

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"facts":["User likes Python"],'
                                                '"preferences":[],"corrections":[],'
                                                '"session_summary":""}')]
        extraction_provider = AsyncMock()
        extraction_provider.complete = AsyncMock(return_value=mock_response)
        graph_disabled._extraction_provider = extraction_provider
        graph_disabled.config["auto_learn"] = True

        await graph_disabled.extract_and_learn(
            "I like Python",
            "Great choice!",
            "u1",
        )

        assert extraction_provider.complete.await_count == 1
        assert main_provider.complete.await_count == 0

    @pytest.mark.asyncio
    async def test_extract_filters_self_limit_corrections(self, graph_disabled):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"facts":[],"preferences":[],'
                                                '"corrections":["I can\'t remember previous conversations.",'
                                                '"My name is Vlad, not Ivan"],'
                                                '"session_summary":""}')]
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=mock_response)
        graph_disabled.provider = mock_provider
        graph_disabled.config["auto_learn"] = True

        await graph_disabled.extract_and_learn(
            "Remember this",
            "Got it",
            "u1",
        )

        corrections = [
            m["content"] for m in graph_disabled.get_all_memories("u1")
            if m["type"] == "correction"
        ]
        assert "My name is Vlad, not Ivan" in corrections
        assert not any("can't remember previous conversations" in c.lower()
                       for c in corrections)

    @pytest.mark.asyncio
    async def test_recall_skips_legacy_self_limit_corrections(self, graph_disabled):
        await graph_disabled.remember(
            "I can't remember previous conversations, only this chat.",
            "u1",
            "correction",
            0.95,
        )
        await graph_disabled.remember(
            "Previous conversations included Kubernetes deploy notes.",
            "u1",
            "fact",
            0.7,
        )

        results = graph_disabled.recall("previous conversations", "u1", top_k=5)
        contents = [r["content"] for r in results]
        assert any("Kubernetes" in c for c in contents)
        assert all("can't remember previous conversations" not in c.lower()
                   for c in contents)

    @pytest.mark.asyncio
    async def test_recall_skips_third_person_session_limit_noise(self, graph_disabled):
        await graph_disabled.remember(
            "Система работает только в текущем контексте без сохранения данных между сессиями.",
            "u1",
            "correction",
            0.95,
        )
        await graph_disabled.remember(
            "Пользователь добавил долгосрочную память для проекта LiteAgent.",
            "u1",
            "fact",
            0.75,
        )

        results = graph_disabled.recall("долгосрочная память", "u1", top_k=5)
        contents = [r["content"] for r in results]
        assert any("долгосрочную память" in c.lower() for c in contents)
        assert all("без сохранения данных между сессиями" not in c.lower()
                   for c in contents)


class TestPinnedProfile:
    def test_profile_backfilled_from_existing_chat(self, graph_disabled):
        graph_disabled.add_message("u1", "user", "Привет, меня зовут Влад")
        profile = graph_disabled.ensure_user_profile("u1")
        assert profile.get("name") == "Влад"

    @pytest.mark.asyncio
    async def test_profile_updated_even_when_auto_learn_disabled(self, graph_disabled):
        graph_disabled.config["auto_learn"] = False
        graph_disabled.provider = None

        await graph_disabled.extract_and_learn(
            "Привет, меня зовут Влад",
            "Рад знакомству!",
            "u1",
        )

        profile = graph_disabled.get_user_profile("u1")
        assert profile.get("name") == "Влад"

    @pytest.mark.asyncio
    async def test_profile_not_extracted_from_assistant_self_intro(self, graph_disabled):
        graph_disabled.config["auto_learn"] = False
        graph_disabled.provider = None

        await graph_disabled.extract_and_learn(
            "Как тебя зовут?",
            "Меня зовут Jess — я твой AI-ассистент.",
            "u1",
        )

        profile = graph_disabled.get_user_profile("u1")
        assert "name" not in profile

    @pytest.mark.asyncio
    async def test_profile_hides_contradicted_canonical_name(self, graph_disabled):
        graph_disabled.upsert_canonical_slot("u1", "name", "Jess", confidence=0.92, source="test")
        await graph_disabled.remember("name_is_not_Jess", "u1", "fact", 0.9)

        profile = graph_disabled.get_user_profile("u1")
        assert profile.get("name") is None

        recalled = graph_disabled.recall_type_aware("как меня зовут", "u1", top_k=5)
        assert not any(
            (r.get("type") == "profile_slot" and "jess" in str(r.get("content", "")).lower())
            for r in recalled
        )

    def test_profile_extracts_name_from_user_name_statement(self, graph_disabled):
        profile = graph_disabled.update_user_profile_from_texts(
            "u1",
            ["User's name is Alice"],
        )
        assert profile.get("name") == "Alice"

    def test_profile_extracts_name_from_ru_user_label(self, graph_disabled):
        profile = graph_disabled.update_user_profile_from_texts(
            "u1",
            ["имя пользователя - Слава"],
        )
        assert profile.get("name") == "Слава"

    @pytest.mark.asyncio
    async def test_resolve_profile_slot_prefers_recent_user_name_signal(self, graph_disabled):
        graph_disabled.upsert_canonical_slot("u1", "name", "Jess", confidence=0.92, source="test")
        await graph_disabled.remember("name_is_not_Jess", "u1", "fact", 0.9)
        graph_disabled.add_message("u1", "user", "Слава запиши в свою память")
        await graph_disabled.remember("имя пользователя - Слава", "u1", "fact", 0.6)

        resolved = graph_disabled.resolve_profile_slot("u1", "name", auto_heal=True)
        assert resolved.get("value") == "Слава"
        assert float(resolved.get("confidence", 0.0)) >= 0.7

        slot = graph_disabled.get_canonical_slot("u1", "name")
        assert slot is not None
        assert slot.get("slot_value") == "Слава"

    def test_apply_explicit_profile_update_overrides_previous_name(self, graph_disabled):
        graph_disabled.upsert_canonical_slot("u1", "name", "Jess", confidence=0.99, source="test")
        updates = graph_disabled.apply_explicit_profile_update("u1", "Слава запиши в свою память")

        assert updates.get("name", {}).get("value") == "Слава"
        slot = graph_disabled.get_canonical_slot("u1", "name")
        assert slot is not None
        assert slot.get("slot_value") == "Слава"

    def test_apply_explicit_profile_update_ignores_non_explicit_text(self, graph_disabled):
        updates = graph_disabled.apply_explicit_profile_update("u1", "как меня зовут?")
        assert updates == {}


# ══════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ══════════════════════════════════════════

class TestGraphBackwardCompat:
    """Ensure graph features don't break existing memory."""

    @pytest.mark.asyncio
    async def test_memory_works_with_graph(self, graph_memory):
        await graph_memory.remember("User likes Python", "u1", "fact", 0.8)
        results = graph_memory.recall("Python", "u1")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_memory_works_without_graph(self, graph_disabled):
        await graph_disabled.remember("User likes Java", "u1", "fact", 0.8)
        results = graph_disabled.recall("Java", "u1")
        assert len(results) >= 1

    def test_old_fixture_unaffected(self, memory_system):
        """Original memory_system fixture should still work."""
        memory_system.add_message("u1", "user", "hello")
        history = memory_system.get_history("u1")
        assert len(history) == 1
