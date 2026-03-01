"""Configuration loader — JSON config + env vars + logging setup."""

import json
import logging
import os
import stat
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════
# PROVIDER KEY MANAGEMENT
# ══════════════════════════════════════════

KEYS_DIR = Path.home() / ".liteagent"
KEYS_PATH = KEYS_DIR / "keys.json"

# Env var name per provider
PROVIDER_ENV_VARS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "ollama": None,  # No key needed
}


def load_provider_keys() -> dict[str, str]:
    """Load API keys from ~/.liteagent/keys.json."""
    if KEYS_PATH.exists():
        try:
            return json.loads(KEYS_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load keys.json: %s", e)
    return {}


def save_provider_key(provider: str, key: str) -> None:
    """Save API key to ~/.liteagent/keys.json with chmod 600."""
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    keys = load_provider_keys()
    keys[provider] = key
    KEYS_PATH.write_text(json.dumps(keys, indent=2))
    try:
        os.chmod(KEYS_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 600
    except OSError:
        pass  # Windows doesn't support chmod


def delete_provider_key(provider: str) -> bool:
    """Remove API key from keys.json. Returns True if key existed."""
    keys = load_provider_keys()
    if provider in keys:
        del keys[provider]
        KEYS_PATH.write_text(json.dumps(keys, indent=2))
        return True
    return False


def get_api_key(provider: str) -> str | None:
    """Get API key: first from keys.json, then from env var."""
    # 1. Check keys.json
    keys = load_provider_keys()
    if provider in keys and keys[provider]:
        return keys[provider]
    # 2. Check env var
    env_var = PROVIDER_ENV_VARS.get(provider)
    if env_var:
        return os.environ.get(env_var)
    return None


def key_preview(key: str) -> str:
    """Return masked preview of API key: first 6 + last 4 chars."""
    if not key:
        return ""
    if len(key) <= 12:
        return key[:3] + "..." + key[-2:]
    return key[:6] + "..." + key[-4:]


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


# ══════════════════════════════════════════
# CONFIG WRITING
# ══════════════════════════════════════════

DEFAULT_CONFIG_WRITE_PATH = Path("config.json")


def save_config(config: dict, config_path: str | None = None):
    """Write config dict back to config.json."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_WRITE_PATH
    # Don't write internal/transient keys
    _SKIP_KEYS = {"_resolved"}
    clean = {k: v for k, v in config.items() if k not in _SKIP_KEYS}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info("Config saved to %s", path)


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
