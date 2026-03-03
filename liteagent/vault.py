"""Encrypted vault for API keys — Fernet symmetric encryption + PBKDF2.

Activation: set ``LITEAGENT_VAULT_KEY`` environment variable.
Without it, all functions fall through to the plain ``keys.json`` path
(full backward compatibility).
"""

import json
import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger(__name__)

KEYS_DIR = Path.home() / ".liteagent"
KEYS_PATH = KEYS_DIR / "keys.json"
VAULT_PATH = KEYS_DIR / "keys.enc"


# ── Core helpers ──────────────────────────────────────────


def vault_enabled() -> bool:
    """Return True if a vault master key is configured."""
    return bool(os.environ.get("LITEAGENT_VAULT_KEY"))


def _derive_fernet_key() -> bytes:
    """Derive a Fernet-compatible key from the master password via PBKDF2."""
    raw = os.environ["LITEAGENT_VAULT_KEY"]
    try:
        import base64
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
        from cryptography.hazmat.primitives import hashes
    except ImportError as exc:
        raise ImportError(
            "Vault requires the 'cryptography' package.  "
            "Install it with:  pip install liteagent[vault]"
        ) from exc
    salt = b"liteagent-vault-salt-v1"  # fixed salt (single-user local tool)
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                      salt=salt, iterations=480_000)
    return base64.urlsafe_b64encode(kdf.derive(raw.encode("utf-8")))


def _fernet():
    """Return a ``Fernet`` instance using the derived key."""
    from cryptography.fernet import Fernet
    return Fernet(_derive_fernet_key())


# ── Public API ────────────────────────────────────────────


def load_keys() -> dict[str, str]:
    """Decrypt and return keys from the vault file.

    Returns empty dict on any failure — caller should fallback to keys.json.
    """
    if not VAULT_PATH.exists():
        return {}
    try:
        encrypted = VAULT_PATH.read_bytes()
        if not encrypted:
            logger.warning("Vault file exists but is empty — ignoring")
            return {}
        decrypted = _fernet().decrypt(encrypted)
        return json.loads(decrypted)
    except Exception as exc:
        exc_type = type(exc).__name__
        logger.error(
            "Failed to decrypt vault (%s: %s). "
            "keys.json will be used as fallback. "
            "To fix: unset LITEAGENT_VAULT_KEY or delete %s",
            exc_type, exc or "(no details)", VAULT_PATH,
        )
        return {}


def save_keys(keys: dict[str, str]) -> None:
    """Encrypt and persist *keys* to the vault file."""
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    data = json.dumps(keys).encode("utf-8")
    VAULT_PATH.write_bytes(_fernet().encrypt(data))
    try:
        os.chmod(VAULT_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 600
    except OSError:
        pass


def vault_set(provider: str, api_key: str) -> None:
    """Add or update a single key in the vault.

    Safety: if vault is empty but keys.json has keys, merge them
    to prevent data loss on first vault write.
    """
    keys = load_keys()
    # If vault returned nothing, check if keys.json has data we should preserve
    if not keys and KEYS_PATH.exists():
        try:
            import json as _json
            json_keys = _json.loads(KEYS_PATH.read_text())
            if json_keys:
                logger.info("Vault empty — merging %d key(s) from keys.json into vault", len(json_keys))
                keys = json_keys
        except Exception:
            pass
    keys[provider] = api_key
    save_keys(keys)


def vault_delete(provider: str) -> bool:
    """Remove a key from the vault. Returns True if it existed."""
    keys = load_keys()
    if provider in keys:
        del keys[provider]
        save_keys(keys)
        return True
    return False


def vault_list() -> list[str]:
    """Return provider names stored in the vault (values are NOT exposed)."""
    return list(load_keys().keys())


def migrate_to_vault() -> None:
    """Encrypt existing ``keys.json`` into ``keys.enc`` and remove the plaintext file."""
    if not vault_enabled():
        raise RuntimeError("LITEAGENT_VAULT_KEY environment variable is not set")
    if not KEYS_PATH.exists():
        logger.info("No keys.json to migrate")
        return
    if VAULT_PATH.exists():
        logger.warning("keys.enc already exists — skipping migration")
        return
    keys = json.loads(KEYS_PATH.read_text())
    save_keys(keys)
    KEYS_PATH.unlink()
    logger.info("Migrated %d keys from keys.json → keys.enc (plaintext removed)", len(keys))
