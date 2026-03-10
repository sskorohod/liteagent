"""Tests for skills management: API endpoints, registry methods, agent tools."""

import json
import textwrap
from pathlib import Path

import pytest

from liteagent.skills import SkillRegistry, _dict_to_yaml
from liteagent.agent import LiteAgent
from liteagent.channels.api import create_app


# ── Fixtures ────────────────────────────────────────


@pytest.fixture
def skills_dir(tmp_path):
    """Create temp skills directory with a test skill."""
    user_skill = tmp_path / "user-skills" / "test-skill"
    user_skill.mkdir(parents=True)
    (user_skill / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: test-skill
        description: "A test skill"
        metadata:
          emoji: "T"
          keywords:
            - testword
            - testing
        ---

        ## Test Skill

        Test instructions here.
    """))
    return tmp_path


@pytest.fixture
def client(tmp_path, monkeypatch, skills_dir):
    """Create FastAPI TestClient with skills support."""
    import liteagent.config as config_mod
    config = {
        "agent": {"max_iterations": 3},
        "cost": {"budget_daily_usd": 100.0},
        "memory": {"db_path": str(tmp_path / "test.db"), "auto_learn": False},
        "tools": {"builtin": []},
        "channels": {"api": {"rate_limit": {"requests_per_minute": 100},
                              "auth_enabled": False}},
        "skills": {"extra_dirs": [str(skills_dir / "user-skills")]},
    }
    agent = LiteAgent(config)
    app = create_app(agent)
    monkeypatch.setattr("liteagent.config.save_config", lambda *a, **kw: None)
    monkeypatch.setattr(config_mod, "KEYS_DIR", tmp_path)
    monkeypatch.setattr(config_mod, "KEYS_PATH", tmp_path / "keys.json")
    monkeypatch.setattr(config_mod, "KEYS_BACKUP_PATH", tmp_path / "keys.json.bak")
    from starlette.testclient import TestClient
    c = TestClient(app)
    yield c, agent
    agent.memory.close()


# ── SkillRegistry core methods ─────────────────────


class TestLoadAllClear:
    """load_all() must clear existing skills before reloading."""

    def test_reload_clears_stale(self, tmp_path):
        reg = SkillRegistry()
        d = tmp_path / "skills" / "alpha"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: alpha\ndescription: A\n---\nBody A")
        reg.load_all({"skills": {"extra_dirs": [str(tmp_path / "skills")]}})
        assert "alpha" in reg._skills

        import shutil
        shutil.rmtree(d)
        reg.load_all({"skills": {"extra_dirs": [str(tmp_path / "skills")]}})
        assert "alpha" not in reg._skills

    def test_reload_picks_up_new_skills(self, tmp_path):
        reg = SkillRegistry()
        skills_root = tmp_path / "skills"
        skills_root.mkdir()
        reg.load_all({"skills": {"extra_dirs": [str(skills_root)]}})
        assert "beta" not in reg._skills

        d = skills_root / "beta"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: beta\ndescription: B\n---\nBody B")
        reg.load_all({"skills": {"extra_dirs": [str(skills_root)]}})
        assert "beta" in reg._skills


class TestGetSkill:
    def test_get_existing(self, tmp_path):
        reg = SkillRegistry()
        d = tmp_path / "skills" / "gamma"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: gamma\ndescription: G desc\n---\n## Gamma body")
        reg.load_all({"skills": {"extra_dirs": [str(tmp_path / "skills")]}})
        detail = reg.get_skill("gamma")
        assert detail is not None
        assert detail["name"] == "gamma"
        assert detail["description"] == "G desc"
        assert "body" in detail
        assert "Gamma body" in detail["body"]
        assert detail["source"] == "extra"

    def test_get_missing(self):
        reg = SkillRegistry()
        assert reg.get_skill("nonexistent") is None


class TestWriteSkill:
    def test_write_creates_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("liteagent.skills._USER_SKILLS_DIR", tmp_path / "skills")
        reg = SkillRegistry()
        path = reg.write_skill("new-test", "## Body here", {
            "name": "new-test",
            "description": "A new skill",
            "metadata": {"emoji": "N", "keywords": ["kw1", "kw2"]},
        })
        assert path.exists()
        content = path.read_text()
        assert "new-test" in content
        assert "## Body here" in content

    def test_write_loads_after_reload(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "skills"
        monkeypatch.setattr("liteagent.skills._USER_SKILLS_DIR", user_dir)
        reg = SkillRegistry()
        reg.write_skill("reload-test", "Body", {
            "name": "reload-test", "description": "R"})
        reg.load_all({"skills": {}})
        assert "reload-test" in reg._skills


class TestDeleteSkill:
    def test_delete_user_skill(self, tmp_path):
        reg = SkillRegistry()
        d = tmp_path / "skills" / "del-me"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: del-me\ndescription: D\n---\nBody")
        reg.load_all({"skills": {"extra_dirs": [str(tmp_path / "skills")]}})
        assert "del-me" in reg._skills

        ok = reg.delete_skill("del-me")
        assert ok
        assert "del-me" not in reg._skills
        assert not d.exists()

    def test_delete_bundled_fails(self):
        reg = SkillRegistry()
        reg.load_all({})
        if "voice" in reg._skills:
            ok = reg.delete_skill("voice")
            assert not ok
            assert "voice" in reg._skills

    def test_delete_nonexistent(self):
        reg = SkillRegistry()
        assert not reg.delete_skill("no-such-skill")


class TestListSkillsEnhanced:
    def test_includes_missing_fields(self, tmp_path):
        reg = SkillRegistry()
        d = tmp_path / "skills" / "env-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: env-skill
            description: Needs env
            metadata:
              requires:
                env:
                  - NONEXISTENT_VAR_XYZ
            ---
            Body
        """))
        reg.load_all({"skills": {"extra_dirs": [str(tmp_path / "skills")]}})
        skills = reg.list_skills()
        env_skill = next((s for s in skills if s["name"] == "env-skill"), None)
        assert env_skill is not None
        assert "missing_env" in env_skill
        assert "NONEXISTENT_VAR_XYZ" in env_skill["missing_env"]
        assert env_skill["eligible"] is False

    def test_always_field_present(self, tmp_path):
        reg = SkillRegistry()
        d = tmp_path / "skills" / "always-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: always-skill\ndescription: A\nmetadata:\n  always: true\n---\nBody")
        reg.load_all({"skills": {"extra_dirs": [str(tmp_path / "skills")]}})
        skills = reg.list_skills()
        a = next((s for s in skills if s["name"] == "always-skill"), None)
        assert a is not None
        assert a["always"] is True


class TestDictToYaml:
    def test_simple(self):
        result = _dict_to_yaml({"name": "x", "description": "y"})
        assert "name: x" in result
        assert "description: y" in result

    def test_nested(self):
        result = _dict_to_yaml({"metadata": {"emoji": "E", "keywords": ["a", "b"]}})
        assert "metadata:" in result
        assert "  emoji: E" in result
        assert "  - a" in result
        assert "  - b" in result


# ── REST API endpoints ─────────────────────────────


class TestSkillsListAPI:
    def test_list_skills(self, client):
        c, agent = client
        resp = c.get("/api/settings/skills")
        assert resp.status_code == 200
        data = resp.json()
        assert "skills" in data
        assert "count" in data
        assert "disabled" in data
        assert data["count"] >= 1  # at least bundled skills

    def test_list_includes_extra_skill(self, client):
        c, agent = client
        data = c.get("/api/settings/skills").json()
        names = [s["name"] for s in data["skills"]]
        assert "test-skill" in names


class TestSkillDetailAPI:
    def test_get_skill(self, client):
        c, _ = client
        resp = c.get("/api/skills/test-skill")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-skill"
        assert "body" in data
        assert len(data["body"]) > 0
        assert data["source"] == "extra"

    def test_get_nonexistent(self, client):
        c, _ = client
        resp = c.get("/api/skills/nonexistent-xyz-abc")
        assert resp.status_code == 404


class TestSkillToggleAPI:
    def test_disable_skill(self, client):
        c, agent = client
        resp = c.post("/api/skills/test-skill/toggle",
                       json={"enabled": False})
        assert resp.status_code == 200
        assert "test-skill" in agent.config.get("skills", {}).get("disabled", [])

    def test_enable_skill(self, client):
        c, agent = client
        agent.config.setdefault("skills", {}).setdefault("disabled", []).append("test-skill")
        resp = c.post("/api/skills/test-skill/toggle",
                       json={"enabled": True})
        assert resp.status_code == 200
        assert "test-skill" not in agent.config.get("skills", {}).get("disabled", [])


class TestSkillCreateAPI:
    def test_create_skill(self, client, monkeypatch, tmp_path):
        c, agent = client
        user_dir = tmp_path / "user-skills-create"
        monkeypatch.setattr("liteagent.skills._USER_SKILLS_DIR", user_dir)
        resp = c.post("/api/skills", json={
            "name": "brand-new",
            "description": "A brand new skill",
            "body": "## New Skill\nDo something.",
            "keywords": ["newkw", "fresh"],
        })
        assert resp.status_code == 200
        assert resp.json()["ok"]
        assert (user_dir / "brand-new" / "SKILL.md").exists()

    def test_create_duplicate(self, client):
        c, _ = client
        resp = c.post("/api/skills", json={
            "name": "test-skill",
            "description": "dup"
        })
        assert resp.status_code == 400

    def test_create_empty_name(self, client):
        c, _ = client
        resp = c.post("/api/skills", json={
            "name": "",
            "description": "no name"
        })
        assert resp.status_code == 400


class TestSkillDeleteAPI:
    def test_delete_extra_skill(self, client):
        c, agent = client
        # test-skill is source=extra, should be deletable
        resp = c.delete("/api/skills/test-skill")
        assert resp.status_code == 200
        assert resp.json()["ok"]

    def test_cannot_delete_bundled(self, client):
        c, _ = client
        # voice is bundled
        resp = c.delete("/api/skills/voice")
        assert resp.status_code == 400

    def test_delete_nonexistent(self, client):
        c, _ = client
        resp = c.delete("/api/skills/no-such-skill-xyz")
        assert resp.status_code == 404


class TestSkillUpdateAPI:
    def test_update_skill(self, client, monkeypatch, tmp_path):
        c, agent = client
        user_dir = tmp_path / "user-skills-update"
        monkeypatch.setattr("liteagent.skills._USER_SKILLS_DIR", user_dir)
        resp = c.put("/api/skills/test-skill", json={
            "body": "## Updated body\nNew instructions.",
            "metadata": {
                "description": "Updated desc",
                "keywords": ["updated"],
            }
        })
        assert resp.status_code == 200
        assert resp.json()["ok"]
        # Should have written to user dir
        assert (user_dir / "test-skill" / "SKILL.md").exists()

    def test_update_nonexistent(self, client):
        c, _ = client
        resp = c.put("/api/skills/no-such-skill-abc", json={
            "body": "x"
        })
        assert resp.status_code == 404


class TestSkillReloadAPI:
    def test_reload(self, client):
        c, _ = client
        resp = c.post("/api/skills/reload")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"]
        assert data["count"] >= 0


# ── Agent tools ────────────────────────────────────


class TestAgentSkillTools:
    def test_list_tool_registered(self, client):
        _, agent = client
        assert "list_agent_skills" in agent.tools._handlers

    def test_read_tool_registered(self, client):
        _, agent = client
        assert "read_skill" in agent.tools._handlers

    def test_propose_tools_registered(self, client):
        _, agent = client
        assert "propose_skill_update" in agent.tools._handlers
        assert "propose_skill_create" in agent.tools._handlers
        assert "apply_skill_change" in agent.tools._handlers

    def test_list_tool_returns_skills(self, client):
        _, agent = client
        result = json.loads(agent.tools._handlers["list_agent_skills"]())
        assert isinstance(result, list)
        names = [s["name"] for s in result]
        assert "test-skill" in names

    def test_read_tool_returns_body(self, client):
        _, agent = client
        result = json.loads(agent.tools._handlers["read_skill"]("test-skill"))
        assert result["name"] == "test-skill"
        assert "body" in result

    def test_read_tool_missing(self, client):
        _, agent = client
        result = json.loads(agent.tools._handlers["read_skill"]("nope"))
        assert "error" in result

    def test_propose_update(self, client):
        _, agent = client
        result = json.loads(agent.tools._handlers["propose_skill_update"](
            "test-skill", body="new body"))
        assert result["action"] == "update_skill"
        assert "message" in result

    def test_propose_create(self, client):
        _, agent = client
        result = json.loads(agent.tools._handlers["propose_skill_create"](
            "my-new-one", "desc", "body text", "kw1, kw2"))
        assert result["action"] == "create_skill"
        assert result["name"] == "my-new-one"

    def test_propose_create_duplicate(self, client):
        _, agent = client
        result = json.loads(agent.tools._handlers["propose_skill_create"](
            "test-skill", "desc", "body"))
        assert "error" in result

    def test_apply_create(self, client, monkeypatch, tmp_path):
        _, agent = client
        user_dir = tmp_path / "user-skills-apply"
        monkeypatch.setattr("liteagent.skills._USER_SKILLS_DIR", user_dir)
        result = json.loads(agent.tools._handlers["apply_skill_change"](
            "create", "applied-skill", "## Body", "Applied skill", "kw1, kw2", "A"))
        assert result["ok"]
        assert (user_dir / "applied-skill" / "SKILL.md").exists()

    def test_apply_unknown_action(self, client):
        _, agent = client
        result = json.loads(agent.tools._handlers["apply_skill_change"](
            "delete", "x"))
        assert "error" in result


# ── Config validation ──────────────────────────────


class TestConfigValidation:
    def test_valid_skills_keys(self):
        from liteagent.config import validate_config
        warnings = validate_config({
            "skills": {"enabled": True, "disabled": [], "extra_dirs": []}
        })
        skill_warnings = [w for w in warnings if "skills" in w]
        assert len(skill_warnings) == 0

    def test_unknown_skills_key(self):
        from liteagent.config import validate_config
        warnings = validate_config({
            "skills": {"unknown_xyz": True}
        })
        skill_warnings = [w for w in warnings if "skills" in w]
        assert len(skill_warnings) == 1
        assert "unknown_xyz" in skill_warnings[0]
