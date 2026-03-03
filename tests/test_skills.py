"""Tests for the skill system (liteagent.skills)."""

import os
import platform
import textwrap

import pytest

from liteagent.skills import SkillRegistry, SkillMetadata, Skill, _minimal_yaml_parse


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def skill_dir(tmp_path):
    """Create a temporary skills directory with sample skills."""
    # Simple skill with keywords
    weather = tmp_path / "weather"
    weather.mkdir()
    (weather / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: weather
        description: "Get weather forecasts"
        metadata:
          emoji: "🌤️"
          keywords:
            - погода
            - weather
            - температур
          requires:
            bins:
              - curl
        ---

        ## Weather Skill
        Use `curl wttr.in/City` to get weather.
    """))

    # Skill with always=true
    system_info = tmp_path / "system-info"
    system_info.mkdir()
    (system_info / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: system-info
        description: "System information"
        metadata:
          always: true
        ---

        You are running on the current system.
    """))

    # Skill with OS restriction (windows only — should be ineligible on macOS/Linux)
    win_skill = tmp_path / "win-only"
    win_skill.mkdir()
    (win_skill / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: win-only
        description: "Windows-only skill"
        metadata:
          os:
            - win32
          keywords:
            - windows
        ---

        This only works on Windows.
    """))

    # Skill requiring missing binary
    rare = tmp_path / "rare-tool"
    rare.mkdir()
    (rare / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: rare-tool
        description: "Needs a rare binary"
        metadata:
          keywords:
            - rare
          requires:
            bins:
              - this_binary_does_not_exist_xyz123
        ---

        Rare tool instructions.
    """))

    # Skill with references directory
    documented = tmp_path / "documented"
    documented.mkdir()
    refs = documented / "references"
    refs.mkdir()
    (refs / "api.md").write_text("# API Reference\nEndpoint: /api/v1/data")
    (documented / "SKILL.md").write_text(textwrap.dedent("""\
        ---
        name: documented
        description: "A well-documented skill"
        metadata:
          keywords:
            - docs
            - document
        ---

        See references/api.md for API details.
    """))

    return tmp_path


@pytest.fixture
def registry(skill_dir):
    """Load skills from the tmp directory."""
    reg = SkillRegistry()
    reg._load_from_dir(skill_dir, "test", set())
    return reg


# ── Frontmatter Parsing ──────────────────────────────────────


class TestFrontmatterParsing:
    def test_valid_frontmatter(self):
        text = textwrap.dedent("""\
            ---
            name: test-skill
            description: "A test skill"
            ---

            Body content here.
        """)
        fm, body = SkillRegistry._split_frontmatter(text)
        assert fm["name"] == "test-skill"
        assert fm["description"] == "A test skill"
        assert "Body content" in body

    def test_no_frontmatter(self):
        text = "Just a plain markdown file."
        fm, body = SkillRegistry._split_frontmatter(text)
        assert fm == {}
        assert body == text

    def test_empty_frontmatter(self):
        text = "---\n---\n\nBody."
        fm, body = SkillRegistry._split_frontmatter(text)
        assert fm == {} or fm is None or isinstance(fm, dict)
        assert "Body" in body

    def test_frontmatter_with_metadata(self):
        text = textwrap.dedent("""\
            ---
            name: complex
            description: "Complex skill"
            metadata:
              emoji: "🔧"
              keywords:
                - tool
                - repair
              os:
                - darwin
              requires:
                bins:
                  - python3
            ---

            Instructions here.
        """)
        fm, body = SkillRegistry._split_frontmatter(text)
        assert fm["name"] == "complex"
        meta = fm["metadata"]
        assert meta["emoji"] == "🔧"
        assert "tool" in meta["keywords"]
        assert "darwin" in meta["os"]
        assert "python3" in meta["requires"]["bins"]


class TestMinimalYamlParse:
    def test_simple_key_value(self):
        text = 'name: my-skill\ndescription: "A skill"'
        result = _minimal_yaml_parse(text)
        assert result["name"] == "my-skill"
        assert result["description"] == "A skill"

    def test_boolean_values(self):
        text = "always: true\nenabled: false"
        result = _minimal_yaml_parse(text)
        assert result["always"] is True
        assert result["enabled"] is False

    def test_empty_lines_and_comments(self):
        text = "# comment\nname: test\n\n# another comment"
        result = _minimal_yaml_parse(text)
        assert result["name"] == "test"
        assert len(result) == 1


class TestMetadataParsing:
    def test_basic_metadata(self):
        data = {
            "name": "test",
            "description": "Test skill",
            "metadata": {
                "emoji": "🎯",
                "keywords": ["hello", "world"],
            }
        }
        meta = SkillRegistry._parse_metadata(data, "fallback")
        assert meta.name == "test"
        assert meta.description == "Test skill"
        assert meta.emoji == "🎯"
        assert meta.keywords == ["hello", "world"]

    def test_fallback_name(self):
        data = {"description": "No name"}
        meta = SkillRegistry._parse_metadata(data, "dir-name")
        assert meta.name == "dir-name"

    def test_keywords_from_top_level(self):
        data = {"name": "x", "keywords": ["a", "b"]}
        meta = SkillRegistry._parse_metadata(data, "x")
        assert meta.keywords == ["a", "b"]

    def test_keywords_as_string(self):
        data = {"name": "x", "keywords": "a, b, c"}
        meta = SkillRegistry._parse_metadata(data, "x")
        assert meta.keywords == ["a", "b", "c"]

    def test_requires_bins(self):
        data = {
            "name": "x",
            "metadata": {"requires": {"bins": ["python3", "curl"]}}
        }
        meta = SkillRegistry._parse_metadata(data, "x")
        assert meta.requires_bins == ["python3", "curl"]

    def test_tools_list(self):
        data = {
            "name": "x",
            "metadata": {"tools": ["tool_a", "tool_b"]}
        }
        meta = SkillRegistry._parse_metadata(data, "x")
        assert meta.tools == ["tool_a", "tool_b"]


# ── Skill Discovery & Loading ────────────────────────────────


class TestSkillDiscovery:
    def test_loads_skills_from_dir(self, registry):
        names = {s.metadata.name for s in registry._skills.values()}
        assert "weather" in names
        assert "system-info" in names
        assert "documented" in names

    def test_skips_dirs_without_skill_md(self, tmp_path):
        (tmp_path / "no-skill").mkdir()
        (tmp_path / "no-skill" / "README.md").write_text("Not a skill")
        reg = SkillRegistry()
        reg._load_from_dir(tmp_path, "test", set())
        assert "no-skill" not in reg._skills

    def test_disabled_skills_excluded(self, skill_dir):
        reg = SkillRegistry()
        reg._load_from_dir(skill_dir, "test", {"weather"})
        assert "weather" not in reg._skills

    def test_source_override_priority(self, tmp_path):
        """Later source overrides earlier by name."""
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        skill_a = dir_a / "my-skill"
        skill_a.mkdir()
        (skill_a / "SKILL.md").write_text("---\nname: my-skill\ndescription: Version A\n---\nBody A")

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        skill_b = dir_b / "my-skill"
        skill_b.mkdir()
        (skill_b / "SKILL.md").write_text("---\nname: my-skill\ndescription: Version B\n---\nBody B")

        reg = SkillRegistry()
        reg._load_from_dir(dir_a, "bundled", set())
        reg._load_from_dir(dir_b, "user", set())
        assert reg._skills["my-skill"].source == "user"
        assert "Version B" in reg._skills["my-skill"].metadata.description

    def test_load_all_with_config(self, tmp_path, monkeypatch):
        """load_all respects enabled=false."""
        reg = SkillRegistry()
        reg.load_all({"skills": {"enabled": False}})
        assert len(reg._skills) == 0


# ── Eligibility Gates ────────────────────────────────────────


class TestEligibility:
    def test_eligible_skill(self, registry):
        """Weather skill should be eligible (curl exists on most systems)."""
        weather = registry._skills.get("weather")
        if weather:
            import shutil
            if shutil.which("curl"):
                assert registry._check_eligibility(weather) is True

    def test_os_mismatch(self, registry):
        """win-only skill should be ineligible on macOS/Linux."""
        win = registry._skills.get("win-only")
        if win and platform.system().lower() != "windows":
            assert registry._check_eligibility(win) is False

    def test_missing_binary(self, registry):
        """Skill requiring nonexistent binary should be ineligible."""
        rare = registry._skills.get("rare-tool")
        assert rare is not None
        assert registry._check_eligibility(rare) is False

    def test_env_var_check(self, tmp_path, monkeypatch):
        skill_dir = tmp_path / "env-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: env-skill
            description: "Needs env var"
            metadata:
              requires:
                env:
                  - LITEAGENT_TEST_SECRET_XYZ
            ---
            Body.
        """))
        reg = SkillRegistry()
        reg._load_from_dir(tmp_path, "test", set())
        skill = reg._skills["env-skill"]

        # Without env var
        monkeypatch.delenv("LITEAGENT_TEST_SECRET_XYZ", raising=False)
        assert reg._check_eligibility(skill) is False

        # With env var
        monkeypatch.setenv("LITEAGENT_TEST_SECRET_XYZ", "secret123")
        assert reg._check_eligibility(skill) is True


# ── Keyword Triggering ────────────────────────────────────────


class TestTriggering:
    def test_keyword_match(self, registry):
        """'погода' should trigger weather skill."""
        triggered = registry.get_triggered_skills("Какая сегодня погода?")
        names = [s.metadata.name for s in triggered]
        assert "weather" in names

    def test_no_match(self, registry):
        """Random message should not trigger weather skill."""
        triggered = registry.get_triggered_skills("Привет, как дела?")
        names = [s.metadata.name for s in triggered]
        assert "weather" not in names

    def test_case_insensitive(self, registry):
        """Keywords should match case-insensitively."""
        triggered = registry.get_triggered_skills("WEATHER forecast please")
        names = [s.metadata.name for s in triggered]
        assert "weather" in names

    def test_always_skill_triggered(self, registry):
        """Skills with always=true should always trigger."""
        triggered = registry.get_triggered_skills("random message")
        names = [s.metadata.name for s in triggered]
        assert "system-info" in names

    def test_empty_input(self, registry):
        triggered = registry.get_triggered_skills("")
        # Only always=true skills
        assert all(s.metadata.always for s in triggered)

    def test_ineligible_skill_not_triggered(self, registry):
        """Even if keywords match, ineligible skills should not trigger."""
        triggered = registry.get_triggered_skills("I need this rare tool")
        names = [s.metadata.name for s in triggered]
        assert "rare-tool" not in names


# ── Prompt Generation ────────────────────────────────────────


class TestPromptGeneration:
    def test_catalog_prompt_format(self, registry):
        catalog = registry.get_catalog_prompt()
        assert "Available skills" in catalog
        assert "weather" in catalog

    def test_catalog_budget_enforcement(self, registry):
        catalog = registry.get_catalog_prompt(max_chars=100)
        assert len(catalog) <= 150  # some tolerance for the header

    def test_triggered_prompt_contains_body(self, registry):
        prompt = registry.get_triggered_prompt("Покажи погода сегодня")
        assert "curl" in prompt or "wttr" in prompt

    def test_triggered_prompt_empty_when_no_match(self, registry):
        prompt = registry.get_triggered_prompt("Привет!")
        # May contain always-on skills but not weather
        assert "wttr" not in prompt

    def test_triggered_prompt_budget(self, registry):
        prompt = registry.get_triggered_prompt("погода weather", max_chars=50)
        assert len(prompt) <= 100  # rough bound


# ── Reference Loading ────────────────────────────────────────


class TestReferences:
    def test_read_reference(self, registry):
        ref = registry.get_reference("documented", "api.md")
        assert ref is not None
        assert "API Reference" in ref

    def test_missing_reference(self, registry):
        ref = registry.get_reference("documented", "nonexistent.md")
        assert ref is None

    def test_unknown_skill_reference(self, registry):
        ref = registry.get_reference("no-such-skill", "api.md")
        assert ref is None

    def test_path_traversal_blocked(self, registry):
        ref = registry.get_reference("documented", "../../etc/passwd")
        assert ref is None


# ── Skill Listing ────────────────────────────────────────────


class TestListSkills:
    def test_list_returns_all_skills(self, registry):
        skills = registry.list_skills()
        names = {s["name"] for s in skills}
        assert "weather" in names
        assert "system-info" in names

    def test_list_includes_metadata(self, registry):
        skills = registry.list_skills()
        weather = next(s for s in skills if s["name"] == "weather")
        assert weather["emoji"] == "🌤️"
        assert weather["source"] == "test"
        assert "погода" in weather["keywords"]

    def test_list_shows_eligibility(self, registry):
        skills = registry.list_skills()
        rare = next(s for s in skills if s["name"] == "rare-tool")
        assert rare["eligible"] is False


# ── Voice Skill Migration ────────────────────────────────────


class TestVoiceSkillMigration:
    """Verify that the bundled voice skill works correctly."""

    def test_voice_skill_loads(self):
        """Voice skill should load from bundled skills directory."""
        from pathlib import Path
        skills_dir = Path(__file__).parent.parent / "liteagent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("Bundled skills not found")
        reg = SkillRegistry()
        reg._load_from_dir(skills_dir, "bundled", set())
        assert "voice" in reg._skills

    def test_voice_skill_triggers_on_russian(self):
        """Voice skill should trigger on Russian voice keywords."""
        reg = SkillRegistry()
        from pathlib import Path
        skills_dir = Path(__file__).parent.parent / "liteagent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("Bundled skills not found")
        reg._load_from_dir(skills_dir, "bundled", set())

        for keyword in ["голос", "tts", "озвуч", "voice", "пресет", "транскриб"]:
            triggered = reg.get_triggered_skills(f"Настрой мне {keyword}")
            names = [s.metadata.name for s in triggered]
            assert "voice" in names, f"Voice skill not triggered by '{keyword}'"

    def test_voice_skill_not_triggered_by_random(self):
        """Voice skill should NOT trigger on unrelated messages."""
        reg = SkillRegistry()
        from pathlib import Path
        skills_dir = Path(__file__).parent.parent / "liteagent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("Bundled skills not found")
        reg._load_from_dir(skills_dir, "bundled", set())

        triggered = reg.get_triggered_skills("Расскажи мне анекдот")
        names = [s.metadata.name for s in triggered]
        assert "voice" not in names

    def test_voice_skill_body_content(self):
        """Voice skill body should contain tool descriptions."""
        reg = SkillRegistry()
        from pathlib import Path
        skills_dir = Path(__file__).parent.parent / "liteagent" / "skills"
        if not skills_dir.is_dir():
            pytest.skip("Bundled skills not found")
        reg._load_from_dir(skills_dir, "bundled", set())

        voice = reg._skills.get("voice")
        assert voice is not None
        assert "get_voice_settings" in voice.body
        assert "set_voice_settings" in voice.body
        assert "test_tts" in voice.body
