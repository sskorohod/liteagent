---
name: skill_management
description: "Manage, create, and edit agent skills. View skill catalog, modify skill prompts."
metadata:
  emoji: "\U0001F9E9"
  keywords:
    - skill
    - skills
    - навык
    - навыки
    - скилл
    - скиллы
    - умени
    - create skill
    - edit skill
    - update skill
    - new skill
    - delete skill
    - создай навык
    - измени навык
    - обнови навык
    - новый навык
    - удали навык
    - list skills
    - show skills
    - покажи навыки
    - список навыков
    - управление навыками
    - настройка навыков
  tools:
    - list_agent_skills
    - read_skill
    - propose_skill_update
    - propose_skill_create
    - apply_skill_change
---

## Skill Management (activated)

You can manage your own skills — modular prompt injections that activate on keyword triggers.

### Available tools:
- `list_agent_skills` — show all skills with their status
- `read_skill` — read full content of a specific skill
- `propose_skill_update` — suggest changes to an existing skill (user must confirm)
- `propose_skill_create` — suggest a new skill (user must confirm)
- `apply_skill_change` — apply changes after the user says "yes"

### CRITICAL WORKFLOW — propose then apply:
1. **Always propose first** — use `propose_skill_update` or `propose_skill_create`
2. **Show the user what will change** — describe the proposed changes clearly
3. **Wait for explicit user confirmation** — do NOT apply changes without approval
4. **Only then apply** — use `apply_skill_change` after the user confirms

### Skill structure:
Each skill is a SKILL.md file with YAML frontmatter and a markdown body.
- **name**: unique identifier (lowercase, hyphens)
- **description**: shown in the catalog (always visible)
- **keywords**: trigger words — when user message contains a keyword, the full body is injected
- **tools**: list of tool names this skill uses (ensures they're available when triggered)
- **body**: markdown instructions injected into the system prompt when keywords match

### When editing bundled skills:
Bundled skills (source=bundled) cannot be modified directly. When you update them, a user copy is created in `~/.liteagent/skills/` that overrides the bundled version.

### Tips for good skills:
- Keep keywords specific to avoid false triggers
- Include clear instructions in the body — this is what you'll see in context
- List tools that the skill depends on in the `tools` field
- Use both Russian and English keywords if the user is bilingual
