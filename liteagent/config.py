"""Configuration loader — JSON config + env vars + logging setup."""

import copy
import fcntl
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
    "grok": "XAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "qwen": "DASHSCOPE_API_KEY",
    "ollama": None,  # No key needed
}


def load_provider_keys() -> dict[str, str]:
    """Load API keys — from encrypted vault if enabled, else keys.json.

    Safety: if vault is enabled but returns empty (decrypt failure, missing file),
    falls back to keys.json so keys are never silently lost.
    """
    from .vault import vault_enabled, load_keys
    if vault_enabled():
        vault_keys = load_keys()
        if vault_keys:
            return vault_keys
        # Vault enabled but empty/failed — fallback to keys.json
        logger.warning("Vault enabled but returned no keys — falling back to keys.json")
    if KEYS_PATH.exists():
        try:
            return json.loads(KEYS_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load keys.json: %s", e)
    return {}


def _write_keys_locked(keys: dict) -> None:
    """Write keys.json atomically with file lock to prevent race conditions."""
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = KEYS_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            # Re-read inside lock to prevent lost updates
            if KEYS_PATH.exists():
                try:
                    current = json.loads(KEYS_PATH.read_text())
                except (json.JSONDecodeError, OSError):
                    current = {}
            else:
                current = {}
            # Trace: detect key loss
            for k in current:
                if k not in keys and k not in current:
                    pass  # not losing anything
            merged = {**current, **keys}
            lost = [k for k in current if k not in merged]
            if lost:
                import traceback
                logger.error("KEYS LOSS DETECTED! Lost keys: %s. Traceback:\n%s",
                             lost, ''.join(traceback.format_stack()))
            current.update(keys)
            KEYS_PATH.write_text(json.dumps(current, indent=2))
            try:
                os.chmod(KEYS_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 600
            except OSError:
                pass
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def save_provider_key(provider: str, key: str) -> None:
    """Save API key — to vault if enabled, AND always to keys.json as backup.

    Double-write ensures keys survive vault failures (wrong master key,
    missing cryptography package, corrupted keys.enc).
    """
    from .vault import vault_enabled, vault_set
    if vault_enabled():
        try:
            vault_set(provider, key)
        except Exception as e:
            logger.warning("Failed to save key to vault: %s — saving to keys.json only", e)
    logger.info("Saving key for provider '%s' (preview: %s...)", provider, key[:6])
    _write_keys_locked({provider: key})


def delete_provider_key(provider: str) -> bool:
    """Remove API key from both vault and keys.json. Returns True if key existed."""
    import traceback
    logger.warning("delete_provider_key('%s') called from:\n%s",
                   provider, ''.join(traceback.format_stack()))
    from .vault import vault_enabled, vault_delete
    deleted = False
    # Delete from vault if enabled
    if vault_enabled():
        try:
            deleted = vault_delete(provider)
        except Exception as e:
            logger.warning("Failed to delete key from vault: %s", e)
    # Always also delete from keys.json
    lock_path = KEYS_PATH.with_suffix(".lock")
    with open(lock_path, "w") as lock_fd:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        try:
            if KEYS_PATH.exists():
                try:
                    keys = json.loads(KEYS_PATH.read_text())
                except (json.JSONDecodeError, OSError):
                    keys = {}
            else:
                keys = {}
            if provider in keys:
                logger.warning("Deleting key for provider '%s' — remaining keys: %s",
                               provider, [k for k in keys if k != provider])
                del keys[provider]
                KEYS_PATH.write_text(json.dumps(keys, indent=2))
                deleted = True
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    return deleted


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

    # Auto-migrate inline API keys to keys.json on load
    migrated = _migrate_inline_keys(config)
    if migrated:
        logger.info("Startup: migrated %d inline key(s) from config to keys.json", migrated)

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
    ("providers", "grok", "api_key"),
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


# Mapping: config secret path → keys.json key name for auto-migration
_SECRET_TO_KEYNAME = {
    ("providers", "anthropic", "api_key"): "anthropic",
    ("providers", "openai", "api_key"): "openai",
    ("providers", "grok", "api_key"): "grok",
    ("providers", "gemini", "api_key"): "gemini",
    ("providers", "qwen", "api_key"): "qwen",
    ("channels", "telegram", "token"): "telegram",
    ("storage", "access_key"): "minio_access",
    ("storage", "secret_key"): "minio_secret",
    ("rag", "qdrant", "api_key"): "qdrant",
}


def _migrate_inline_keys(config: dict) -> int:
    """Migrate API keys found inline in config dict to keys.json.

    If config has e.g. providers.openai.api_key = 'sk-xxx',
    save it to keys.json as 'openai' before stripping.
    Returns the number of keys migrated.
    """
    migrated = 0
    existing_keys = load_provider_keys()

    for secret_path, key_name in _SECRET_TO_KEYNAME.items():
        # Navigate config to find the value
        obj = config
        for k in secret_path[:-1]:
            obj = obj.get(k, {}) if isinstance(obj, dict) else {}
        if not isinstance(obj, dict):
            continue
        value = obj.get(secret_path[-1])
        if not value or not isinstance(value, str):
            continue
        # Skip placeholder values
        if value in ("***", "", "your-key-here"):
            continue
        # Only migrate if keys.json doesn't already have this key
        if key_name in existing_keys and existing_keys[key_name]:
            continue
        logger.info("Migrating inline key '%s' from config to keys.json", key_name)
        _write_keys_locked({key_name: value})
        migrated += 1

    return migrated


def save_config(config: dict, config_path: str | None = None):
    """Write config dict back to the file it was loaded from.

    Secrets (tokens, API keys) are automatically stripped —
    they are stored separately in ~/.liteagent/keys.json.
    Any inline keys found in config are auto-migrated to keys.json
    before stripping, so they are never lost.
    """
    if config_path:
        path = Path(config_path)
    elif config.get("_config_path"):
        path = Path(config["_config_path"])
    else:
        path = DEFAULT_CONFIG_WRITE_PATH

    # Auto-migrate any inline keys to keys.json before stripping
    migrated = _migrate_inline_keys(config)
    if migrated:
        logger.info("Migrated %d inline key(s) to keys.json", migrated)

    # Strip secrets, then remove internal/transient keys (underscore-prefixed)
    stripped = _strip_secrets(config)
    clean = {k: v for k, v in stripped.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    logger.info("Config saved to %s", path)


_KNOWN_TOP_KEYS = {"agent", "memory", "tools", "channels", "cost", "providers", "logging",
                   "scheduler", "agents", "features", "rag", "storage",
                   "hooks", "plugins", "boot", "health", "voice", "skills",
                   "knowledge_base", "web", "night_worker"}
_KNOWN_AGENT_KEYS = {"name", "soul", "max_iterations", "default_model", "models", "provider", "timezone"}
_KNOWN_COST_KEYS = {"cascade_routing", "prompt_caching", "context_compression", "budget_daily_usd", "track_usage"}
_KNOWN_MEMORY_KEYS = {"db_path", "max_history_tokens", "keep_recent_messages", "auto_learn",
                      "extraction_model", "auto_prune", "prune_days", "prune_min_importance",
                      "temporal_decay_rate", "temporal_decay_enabled"}
_KNOWN_RAG_KEYS = {"enabled", "chunk_size", "overlap", "top_k", "collection",
                   "vector_backend", "search", "embedding", "qdrant", "file_indexing"}
_KNOWN_RAG_SEARCH_KEYS = {"mode", "rrf_k", "vector_top_k", "keyword_top_k"}
_KNOWN_RAG_EMBEDDING_KEYS = {"provider", "model", "openai_model", "dimension",
                              "ollama_url", "st_model"}
_KNOWN_RAG_QDRANT_KEYS = {"url", "api_key", "collection"}
_KNOWN_RAG_FILE_INDEXING_KEYS = {"enabled", "max_file_size_mb"}
_KNOWN_KB_KEYS = {"enabled", "db_path", "chunk_size", "chunk_overlap",
                  "search_mode", "rerank", "rerank_model", "query_rewrite",
                  "max_file_size_mb", "contextual_retrieval",
                  "parent_child_retrieval", "self_rag", "self_rag_threshold"}
_KNOWN_NW_KEYS = {"enabled", "model", "batch_size", "max_tasks_per_run",
                  "max_runtime_sec", "cron"}
_KNOWN_WEB_KEYS = {"enabled", "user_agent", "timeout", "max_content_size",
                   "max_extract_length", "cache", "fetch", "search", "crawl", "security"}
_KNOWN_WEB_CACHE_KEYS = {"enabled", "ttl", "max_entries"}
_KNOWN_WEB_FETCH_KEYS = {"strategies", "firecrawl"}
_KNOWN_WEB_SEARCH_KEYS = {"providers", "default_count", "max_count", "fallback",
                           "brave", "duckduckgo", "tavily", "searxng", "perplexity"}
_KNOWN_WEB_CRAWL_KEYS = {"max_depth", "max_pages", "rate_limit_ms", "respect_robots_txt"}
_KNOWN_WEB_SECURITY_KEYS = {"ssrf_protection", "strip_invisible_unicode",
                             "blocked_domains", "allowed_domains"}


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

    rag_cfg = config.get("rag", {})
    for key in rag_cfg:
        if key not in _KNOWN_RAG_KEYS:
            warnings.append(f"Unknown rag key: '{key}'")
    for key in rag_cfg.get("search", {}):
        if key not in _KNOWN_RAG_SEARCH_KEYS:
            warnings.append(f"Unknown rag.search key: '{key}'")
    for key in rag_cfg.get("embedding", {}):
        if key not in _KNOWN_RAG_EMBEDDING_KEYS:
            warnings.append(f"Unknown rag.embedding key: '{key}'")
    for key in rag_cfg.get("qdrant", {}):
        if key not in _KNOWN_RAG_QDRANT_KEYS:
            warnings.append(f"Unknown rag.qdrant key: '{key}'")
    for key in rag_cfg.get("file_indexing", {}):
        if key not in _KNOWN_RAG_FILE_INDEXING_KEYS:
            warnings.append(f"Unknown rag.file_indexing key: '{key}'")

    for key in config.get("knowledge_base", {}):
        if key not in _KNOWN_KB_KEYS:
            warnings.append(f"Unknown knowledge_base key: '{key}'")

    nw_cfg = config.get("night_worker", {})
    for key in nw_cfg:
        if key not in _KNOWN_NW_KEYS:
            warnings.append(f"Unknown night_worker key: '{key}'")

    web_cfg = config.get("web", {})
    for key in web_cfg:
        if key not in _KNOWN_WEB_KEYS:
            warnings.append(f"Unknown web key: '{key}'")
    for key in web_cfg.get("cache", {}):
        if key not in _KNOWN_WEB_CACHE_KEYS:
            warnings.append(f"Unknown web.cache key: '{key}'")
    for key in web_cfg.get("fetch", {}):
        if key not in _KNOWN_WEB_FETCH_KEYS:
            warnings.append(f"Unknown web.fetch key: '{key}'")
    for key in web_cfg.get("search", {}):
        if key not in _KNOWN_WEB_SEARCH_KEYS:
            warnings.append(f"Unknown web.search key: '{key}'")
    for key in web_cfg.get("crawl", {}):
        if key not in _KNOWN_WEB_CRAWL_KEYS:
            warnings.append(f"Unknown web.crawl key: '{key}'")
    for key in web_cfg.get("security", {}):
        if key not in _KNOWN_WEB_SECURITY_KEYS:
            warnings.append(f"Unknown web.security key: '{key}'")

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
