"""Configuration loader — JSON config + env vars + logging setup."""

import copy
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
    """Load API keys — from encrypted vault if enabled, else keys.json."""
    from .vault import vault_enabled, load_keys
    if vault_enabled():
        return load_keys()
    if KEYS_PATH.exists():
        try:
            return json.loads(KEYS_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load keys.json: %s", e)
    return {}


def save_provider_key(provider: str, key: str) -> None:
    """Save API key — to vault if enabled, else keys.json."""
    from .vault import vault_enabled, vault_set
    if vault_enabled():
        vault_set(provider, key)
        return
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    keys = load_provider_keys()
    keys[provider] = key
    KEYS_PATH.write_text(json.dumps(keys, indent=2))
    try:
        os.chmod(KEYS_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 600
    except OSError:
        pass  # Windows doesn't support chmod


def delete_provider_key(provider: str) -> bool:
    """Remove API key. Returns True if key existed."""
    from .vault import vault_enabled, vault_delete
    if vault_enabled():
        return vault_delete(provider)
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


# ══════════════════════════════════════════
# AUTH TOKEN (bearer token for API/dashboard)
# ══════════════════════════════════════════

AUTH_TOKEN_PATH = KEYS_DIR / "auth_token"


def get_or_create_auth_token() -> str:
    """Load or generate the API bearer token.

    Stored in ~/.liteagent/auth_token with chmod 600.
    Generated once on first use, persists across restarts.
    """
    import secrets
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    if AUTH_TOKEN_PATH.exists():
        token = AUTH_TOKEN_PATH.read_text().strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    AUTH_TOKEN_PATH.write_text(token)
    try:
        os.chmod(AUTH_TOKEN_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 600
    except OSError:
        pass
    return token


DEFAULT_CONFIG_PATHS = [
    Path(__file__).resolve().parent.parent / "config.json",     # project root (next to liteagent/)
    Path("~/.liteagent/config.json").expanduser(),              # user home fallback
]


def setup_logging(config: dict) -> None:
    """Configure logging — structured JSON file + human-readable console."""
    from .logging_config import setup_structured_logging
    setup_structured_logging(config)


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
        config["_config_path"] = str(path.resolve())
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

# Legacy: tests monkeypatch this to redirect writes to tmp_path
DEFAULT_CONFIG_WRITE_PATH = Path(__file__).resolve().parent.parent / "config.json"


# Paths to secret fields that should NEVER be written to config.json.
# Each tuple represents a nested key path, e.g. ("channels", "telegram", "token").
_SECRET_PATHS = [
    ("channels", "telegram", "token"),
    ("storage", "access_key"),
    ("storage", "secret_key"),
    ("rag", "qdrant", "api_key"),
    ("providers", "anthropic", "api_key"),
    ("providers", "openai", "api_key"),
    ("providers", "gemini", "api_key"),
]


def _strip_secrets(config: dict) -> dict:
    """Deep-copy config and remove all secret fields before writing to disk."""
    clean = copy.deepcopy(config)
    for path in _SECRET_PATHS:
        obj = clean
        for key in path[:-1]:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                break
        else:
            if isinstance(obj, dict) and path[-1] in obj:
                del obj[path[-1]]
    return clean


def save_config(config: dict, config_path: str | None = None):
    """Write config dict back to the file it was loaded from.

    Secrets (tokens, API keys) are automatically stripped —
    they are stored separately in ~/.liteagent/keys.json.
    """
    if config_path:
        path = Path(config_path)
    elif config.get("_config_path"):
        path = Path(config["_config_path"])
    else:
        path = DEFAULT_CONFIG_WRITE_PATH
    # Strip secrets, then remove internal/transient keys (underscore-prefixed)
    stripped = _strip_secrets(config)
    clean = {k: v for k, v in stripped.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info("Config saved to %s", path)


_KNOWN_TOP_KEYS = {"agent", "memory", "tools", "channels", "cost", "providers", "logging",
                   "scheduler", "agents", "features", "rag", "storage",
                   "hooks", "plugins", "boot", "health"}
_KNOWN_AGENT_KEYS = {"name", "soul", "max_iterations", "default_model", "models", "provider", "timezone"}
_KNOWN_COST_KEYS = {"cascade_routing", "prompt_caching", "context_compression", "budget_daily_usd", "track_usage"}
_KNOWN_MEMORY_KEYS = {"db_path", "max_history_tokens", "keep_recent_messages", "auto_learn",
                      "extraction_model", "auto_prune", "prune_days", "prune_min_importance",
                      "temporal_decay_rate", "temporal_decay_enabled"}


def validate_config(config: dict) -> list[str]:
    """Validate config structure and return list of warnings."""
    warnings = []
    for key in config:
        if key.startswith("_"):
            continue  # internal keys
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

    candidates = [
        Path(soul_path),                                    # CWD
        Path(__file__).resolve().parent.parent / soul_path, # project root
    ]
    # Also try relative to config file location
    cfg_path = config.get("_config_path")
    if cfg_path:
        candidates.insert(1, Path(cfg_path).parent / soul_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.read_text()

    # Fallback
    return "You are a helpful AI assistant."
