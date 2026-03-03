"""Modular skill system — progressive disclosure for token-efficient prompt injection.

Skills are self-contained directories with a SKILL.md file (YAML frontmatter + markdown body).
Only skill metadata (name + description) is loaded into the system prompt catalog.
Full skill body is injected only when triggered by keyword match against user input.

Skill directory structure:
    skill-name/
    ├── SKILL.md          (required — YAML frontmatter + markdown instructions)
    ├── scripts/          (optional — executable code)
    ├── references/       (optional — documentation loaded on demand)
    └── assets/           (optional — templates, files used in output)

Loading sources (later overrides earlier by name):
    1. {package_root}/skills/     — bundled skills (lowest priority)
    2. ~/.liteagent/skills/       — user skills
    3. ./skills/                  — project skills (highest priority)
"""

import logging
import os
import platform
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_USER_SKILLS_DIR = Path.home() / ".liteagent" / "skills"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


@dataclass
class SkillMetadata:
    name: str
    description: str
    emoji: str = ""
    keywords: list[str] = field(default_factory=list)
    os: list[str] | None = None
    requires_bins: list[str] | None = None
    requires_env: list[str] | None = None
    tools: list[str] | None = None
    always: bool = False


@dataclass
class Skill:
    metadata: SkillMetadata
    body: str
    base_dir: Path
    source: str  # "bundled" | "user" | "project"


class SkillRegistry:
    """Discovers, loads, filters, and formats skills for the agent prompt."""

    def __init__(self):
        self._skills: dict[str, Skill] = {}  # name -> Skill (deduplicated)

    # ── Public API ────────────────────────────────────────────────

    def load_all(self, config: dict) -> None:
        """Load skills from all sources (bundled, user, project)."""
        skills_cfg = config.get("skills", {})
        if skills_cfg.get("enabled") is False:
            logger.info("Skills system disabled via config")
            return

        dirs: list[tuple[Path, str]] = []

        # 1. Bundled skills (inside package)
        bundled = Path(__file__).parent / "skills"
        if bundled.is_dir():
            dirs.append((bundled, "bundled"))

        # 2. User skills
        if _USER_SKILLS_DIR.is_dir():
            dirs.append((_USER_SKILLS_DIR, "user"))

        # 3. Extra dirs from config
        for extra in skills_cfg.get("extra_dirs", []):
            p = Path(extra).expanduser()
            if p.is_dir():
                dirs.append((p, "extra"))

        # 4. Project skills (CWD)
        project = Path.cwd() / "skills"
        if project.is_dir():
            dirs.append((project, "project"))

        disabled = set(skills_cfg.get("disabled", []))

        for base, source in dirs:
            self._load_from_dir(base, source, disabled)

        logger.info("Skills loaded: %d (%s)",
                     len(self._skills),
                     ", ".join(self._skills.keys()) if self._skills else "none")

    def get_catalog_prompt(self, max_chars: int = 5000) -> str:
        """Build compact catalog of all eligible skills for the system prompt.

        Only includes name + description (progressive disclosure level 1).
        """
        eligible = [s for s in self._skills.values() if self._check_eligibility(s)]
        if not eligible:
            return ""

        lines = ["\n\n## Available skills (say a keyword to activate):"]
        total = len(lines[0])
        for skill in eligible:
            emoji = f"{skill.metadata.emoji} " if skill.metadata.emoji else ""
            line = f"- **{emoji}{skill.metadata.name}** — {skill.metadata.description}"
            if total + len(line) + 1 > max_chars:
                lines.append(f"- ... and {len(eligible) - len(lines) + 1} more skills")
                logger.warning("Skills catalog truncated at %d chars", max_chars)
                break
            lines.append(line)
            total += len(line) + 1

        return "\n".join(lines)

    def get_triggered_skills(self, user_input: str) -> list[Skill]:
        """Return skills whose keywords match the user input."""
        if not user_input:
            return []
        msg_lower = user_input.lower()
        triggered = []
        for skill in self._skills.values():
            if not self._check_eligibility(skill):
                continue
            if skill.metadata.always:
                triggered.append(skill)
                continue
            for kw in skill.metadata.keywords:
                if kw in msg_lower:
                    triggered.append(skill)
                    break
        return triggered

    def get_triggered_prompt(self, user_input: str, max_chars: int = 10000) -> str:
        """Build prompt text from all triggered skills (progressive disclosure level 2)."""
        triggered = self.get_triggered_skills(user_input)
        if not triggered:
            return ""

        parts = []
        total = 0
        for skill in triggered:
            body = skill.body.strip()
            if not body:
                continue
            if total + len(body) > max_chars:
                logger.warning("Triggered skills truncated at %d chars (skill: %s)",
                               max_chars, skill.metadata.name)
                break
            parts.append(body)
            total += len(body)
            logger.debug("Skill triggered: %s (source=%s)", skill.metadata.name, skill.source)

        return "\n\n".join(parts) if parts else ""

    def get_reference(self, skill_name: str, ref_path: str) -> str | None:
        """Read a reference file from a skill's references/ directory."""
        skill = self._skills.get(skill_name)
        if not skill:
            return None
        ref_file = skill.base_dir / "references" / ref_path
        # Prevent path traversal
        try:
            ref_file.resolve().relative_to(skill.base_dir.resolve())
        except ValueError:
            logger.warning("Path traversal attempt: %s", ref_path)
            return None
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8")
        return None

    def list_skills(self) -> list[dict]:
        """Return list of all loaded skills with metadata (for API/dashboard)."""
        result = []
        for skill in self._skills.values():
            eligible = self._check_eligibility(skill)
            result.append({
                "name": skill.metadata.name,
                "description": skill.metadata.description,
                "emoji": skill.metadata.emoji,
                "source": skill.source,
                "eligible": eligible,
                "keywords": skill.metadata.keywords,
                "tools": skill.metadata.tools or [],
                "has_scripts": (skill.base_dir / "scripts").is_dir(),
                "has_references": (skill.base_dir / "references").is_dir(),
            })
        return result

    # ── Internal ──────────────────────────────────────────────────

    def _load_from_dir(self, base: Path, source: str, disabled: set[str]) -> None:
        """Scan a directory for skill subdirectories containing SKILL.md."""
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                skill = self._parse_skill(skill_md, source)
                if skill.metadata.name in disabled:
                    logger.debug("Skill disabled: %s", skill.metadata.name)
                    continue
                # Later sources override earlier (by name)
                self._skills[skill.metadata.name] = skill
            except Exception as e:
                logger.warning("Failed to parse skill %s: %s", child.name, e)

    def _parse_skill(self, skill_md: Path, source: str) -> Skill:
        """Parse SKILL.md into a Skill object."""
        text = skill_md.read_text(encoding="utf-8")
        frontmatter, body = self._split_frontmatter(text)
        metadata = self._parse_metadata(frontmatter, skill_md.parent.name)
        return Skill(
            metadata=metadata,
            body=body,
            base_dir=skill_md.parent,
            source=source,
        )

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[dict, str]:
        """Split SKILL.md into YAML frontmatter dict and markdown body."""
        match = _FRONTMATTER_RE.match(text)
        if not match:
            return {}, text

        yaml_text = match.group(1)
        body = text[match.end():]

        try:
            import yaml
            data = yaml.safe_load(yaml_text) or {}
        except ImportError:
            data = _minimal_yaml_parse(yaml_text)
        except Exception:
            data = _minimal_yaml_parse(yaml_text)

        return data, body

    @staticmethod
    def _parse_metadata(data: dict, fallback_name: str) -> SkillMetadata:
        """Extract SkillMetadata from parsed frontmatter."""
        meta = data.get("metadata", {})

        # Keywords can be in metadata or top-level
        keywords = meta.get("keywords", data.get("keywords", []))
        if isinstance(keywords, str):
            keywords = [k.strip() for k in keywords.split(",")]

        return SkillMetadata(
            name=data.get("name", fallback_name),
            description=data.get("description", ""),
            emoji=meta.get("emoji", data.get("emoji", "")),
            keywords=keywords,
            os=meta.get("os"),
            requires_bins=meta.get("requires", {}).get("bins") if isinstance(meta.get("requires"), dict) else None,
            requires_env=meta.get("requires", {}).get("env") if isinstance(meta.get("requires"), dict) else None,
            tools=meta.get("tools", data.get("tools")),
            always=meta.get("always", data.get("always", False)),
        )

    @staticmethod
    def _check_eligibility(skill: Skill) -> bool:
        """Check if skill is eligible on current platform."""
        meta = skill.metadata

        # OS check
        if meta.os:
            current_os = platform.system().lower()
            os_map = {"darwin": "darwin", "linux": "linux", "windows": "win32"}
            if os_map.get(current_os, current_os) not in meta.os:
                return False

        # Binary check
        if meta.requires_bins:
            for b in meta.requires_bins:
                if shutil.which(b) is None:
                    return False

        # Env var check
        if meta.requires_env:
            for env in meta.requires_env:
                if not os.environ.get(env):
                    return False

        return True


def _minimal_yaml_parse(text: str) -> dict:
    """Fallback YAML parser for simple key: value frontmatter (no PyYAML needed)."""
    result = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value.lower() in ("true",):
                result[key] = True
            elif value.lower() in ("false",):
                result[key] = False
            elif value:
                result[key] = value
    return result
