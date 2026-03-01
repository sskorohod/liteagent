"""Interactive onboarding — agent-driven setup on first launch."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# All available metacognitive features with descriptions
FEATURE_CATALOG = {
    "style_adaptation": "Адаптация стиля общения",
    "confidence_gate": "Автоматическое переключение на мощную модель при неуверенности",
    "self_evolving_prompt": "Эволюция промпта на основе опыта",
    "proactive_agent": "Проактивные предложения по автоматизации",
    "skill_crystallization": "Запоминание цепочек инструментов",
    "auto_tool_synthesis": "Автоматическое создание новых инструментов",
    "dream_cycle": "Консолидация и очистка памяти",
    "counterfactual_replay": "Анализ прошлых ответов для улучшения",
}

# Preset bundles
FEATURE_PRESETS = {
    "basic": ["style_adaptation", "confidence_gate", "skill_crystallization"],
    "all": list(FEATURE_CATALOG.keys()),
}

ONBOARDING_PROMPT = """You are LiteAgent — an AI assistant being set up for the first time.
The user has just launched you. Your job is to introduce yourself, explain your features,
and collect information to personalize the agent. Follow these phases:

## Phase 1: Introduction

Greet the user warmly. Briefly explain what you are:
- An AI assistant with persistent memory across sessions
- Tool access (file operations, commands, API calls)
- RAG pipeline for document analysis
- Multi-provider support (Anthropic, OpenAI, Gemini, Ollama)

Then explain your advanced metacognitive features in simple terms:

1. **Адаптация стиля** — подстраиваюсь под ваш стиль общения
2. **Уверенность** — если не уверен, переключаюсь на более мощную модель
3. **Эволюция промпта** — мой характер улучшается на основе опыта
4. **Проактивность** — замечаю повторяющиеся задачи и предлагаю автоматизацию
5. **Кристаллизация навыков** — запоминаю полезные цепочки инструментов
6. **Синтез инструментов** — создаю новые инструменты из повторяющихся паттернов
7. **Консолидация памяти** — периодически организую свои воспоминания
8. **Контрфактический анализ** — анализирую прошлые ответы и ищу улучшения

After introducing yourself and the features, proceed to Phase 2.

## Phase 2: Personalization questions

Ask these questions ONE AT A TIME, waiting for each response:

1. "Для чего вы планируете меня использовать? (кодинг, работа, учёба, личный помощник...)"
2. "Какой стиль общения вам ближе? (лаконичный / подробный / дружелюбный-неформальный)"
3. "На каком языке будем общаться?"
4. "Как хотите меня назвать? (или оставим LiteAgent)"
5. "Какие фичи включить? Скажите 'все', 'основные' (адаптация + уверенность + навыки), или перечислите"

## Phase 3: Setup

After collecting enough answers (at least 3), call the `setup_agent` tool with:
- `soul_prompt`: a complete personalized system prompt based on user's answers.
  It should include sections: Core Behavior, Memory Usage, Tool Usage, Communication Style.
  Write it in the language the user chose.
- `agent_name`: the name they chose (default: "LiteAgent")
- `features_preset`: "all", "basic", or "none"
- `enabled_features`: comma-separated list of specific features if not using preset
  Valid names: style_adaptation, confidence_gate, self_evolving_prompt, proactive_agent,
  skill_crystallization, auto_tool_synthesis, dream_cycle, counterfactual_replay

## Rules
- Match the user's language from the first message
- Be warm and welcoming — this is the first impression
- Ask questions ONE at a time
- Keep explanations concise, not walls of text
- After gathering answers, call setup_agent — don't ask for confirmation
"""


def register_onboarding_tool(agent):
    """Register the setup_agent tool on the agent's tool registry."""

    @agent.tools.tool(
        name="setup_agent",
        description="Finalize agent onboarding: save personalized soul prompt, "
                    "agent name, and enable selected features. "
                    "Call this after collecting user preferences."
    )
    def setup_agent(
        soul_prompt: str,
        agent_name: str = "LiteAgent",
        features_preset: str = "basic",
        enabled_features: str = "",
    ) -> str:
        """Set up the agent with personalized configuration.

        soul_prompt: The full personalized system prompt text
        agent_name: Name for the agent
        features_preset: Feature preset — 'all', 'basic', or 'none'
        enabled_features: Comma-separated feature names (overrides preset if provided)
        """
        # Determine which features to enable
        if enabled_features.strip():
            feature_names = [f.strip() for f in enabled_features.split(",") if f.strip()]
        elif features_preset in FEATURE_PRESETS:
            feature_names = FEATURE_PRESETS[features_preset]
        else:
            feature_names = []

        # Validate feature names
        valid = set(FEATURE_CATALOG.keys())
        feature_names = [f for f in feature_names if f in valid]

        # 1. Write soul.md
        soul_path = Path(agent.config.get("agent", {}).get("soul", "soul.md"))
        if not soul_path.is_absolute():
            # Try relative to project root
            for candidate in [Path("soul.md"), Path(__file__).parent.parent / "soul.md"]:
                if candidate.parent.exists():
                    soul_path = candidate
                    break
        soul_path.write_text(soul_prompt, encoding="utf-8")
        logger.info("Wrote personalized soul.md to %s (%d chars)", soul_path, len(soul_prompt))

        # 2. Update config.json
        from .config import save_config
        agent.config.setdefault("agent", {})["name"] = agent_name
        for feature_key in FEATURE_CATALOG:
            agent.config.setdefault("features", {}).setdefault(feature_key, {})
            agent.config["features"][feature_key]["enabled"] = feature_key in feature_names
        save_config(agent.config)
        logger.info("Updated config.json: name=%s, features=%s", agent_name, feature_names)

        # 3. Update runtime state
        agent._soul_prompt = soul_prompt
        agent._features = agent.config.get("features", {})

        # 4. Mark onboarding complete
        agent.memory.set_state("app:onboarding_complete", True)
        logger.info("Onboarding complete")

        enabled_list = ", ".join(feature_names) if feature_names else "none"
        return (
            f"Setup complete! Agent '{agent_name}' configured with features: {enabled_list}. "
            f"Soul prompt saved ({len(soul_prompt)} chars). "
            f"From now on I'll use the personalized configuration."
        )

    return setup_agent


def unregister_onboarding_tool(agent):
    """Remove the setup_agent tool after onboarding is complete."""
    if "setup_agent" in agent.tools._tools:
        del agent.tools._tools["setup_agent"]
    if "setup_agent" in agent.tools._handlers:
        del agent.tools._handlers["setup_agent"]


def needs_onboarding(agent) -> bool:
    """Check if onboarding has been completed."""
    return not agent.memory.get_state("app:onboarding_complete")
