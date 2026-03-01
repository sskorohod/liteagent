"""Configuration loader — JSON config + env vars + logging setup."""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATHS = [
    Path("config.json"),
    Path("~/.liteagent/config.json").expanduser(),
]


def setup_logging(config: dict) -> None:
    """Configure logging from config."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "WARNING").upper(), logging.WARNING)
    fmt = log_cfg.get("format", "%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    logging.basicConfig(level=level, format=fmt)


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load config from file, merge with defaults."""
    # Find config file
    if config_path:
        path = Path(config_path).expanduser()
    else:
        path = None
        for p in DEFAULT_CONFIG_PATHS:
            if p.exists():
                path = p
                break

    config = {}
    if path and path.exists():
        with open(path) as f:
            config = json.load(f)
        logger.info("Config loaded from %s", path)
    else:
        logger.info("No config file found, using defaults")

    # Setup logging early
    setup_logging(config)

    # Resolve env vars for API keys
    _resolve_env_vars(config)

    # Validate config
    warnings = validate_config(config)
    for w in warnings:
        logger.warning("Config: %s", w)

    # Ensure data dir exists
    data_dir = Path(config.get("memory", {}).get("db_path", "~/.liteagent/memory.db")).expanduser().parent
    data_dir.mkdir(parents=True, exist_ok=True)

    return config


_KNOWN_TOP_KEYS = {"agent", "memory", "tools", "channels", "cost", "providers", "logging",
                   "scheduler", "agents", "features", "rag"}
_KNOWN_AGENT_KEYS = {"name", "soul", "max_iterations", "default_model", "models", "provider"}
_KNOWN_COST_KEYS = {"cascade_routing", "prompt_caching", "context_compression", "budget_daily_usd", "track_usage"}
_KNOWN_MEMORY_KEYS = {"db_path", "max_history_tokens", "keep_recent_messages", "auto_learn",
                      "extraction_model", "auto_prune", "prune_days", "prune_min_importance"}


def validate_config(config: dict) -> list[str]:
    """Validate config structure and return list of warnings."""
    warnings = []
    for key in config:
        if key not in _KNOWN_TOP_KEYS:
            warnings.append(f"Unknown top-level key: '{key}'")

    for key in config.get("agent", {}):
        if key not in _KNOWN_AGENT_KEYS:
            warnings.append(f"Unknown agent key: '{key}'")

    for key in config.get("cost", {}):
        if key not in _KNOWN_COST_KEYS:
            warnings.append(f"Unknown cost key: '{key}'")

    for key in config.get("memory", {}):
        if key not in _KNOWN_MEMORY_KEYS:
            warnings.append(f"Unknown memory key: '{key}'")

    return warnings


def _resolve_env_vars(config: dict) -> None:
    """Replace env var references with actual values."""
    # API key from env
    if not os.environ.get("ANTHROPIC_API_KEY"):
        api_key_env = (config.get("providers", {})
                       .get("anthropic", {})
                       .get("api_key_env", "ANTHROPIC_API_KEY"))
        key = os.environ.get(api_key_env)
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key

    # Channel tokens from env
    channels = config.get("channels", {})
    for name, ch_config in channels.items():
        if isinstance(ch_config, dict) and "token_env" in ch_config:
            token = os.environ.get(ch_config["token_env"])
            if token:
                ch_config["token"] = token


def get_soul_prompt(config: dict) -> str:
    """Load soul.md system prompt."""
    soul_path = config.get("agent", {}).get("soul", "soul.md")

    # Try relative to config, then absolute
    for candidate in [Path(soul_path), Path(__file__).parent.parent / soul_path]:
        if candidate.exists():
            return candidate.read_text()

    # Fallback
    return "You are a helpful AI assistant."
