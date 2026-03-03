"""Unified embedding module — single source of truth for all vector embeddings.

Supports multiple providers with automatic fallback:
  Ollama (local, free) → sentence-transformers → OpenAI API → None (keyword-only)

Config (in config.json under "rag.embedding"):
  {
    "provider": "auto",              // auto | ollama | sentence_transformers | openai | none
    "model": "nomic-embed-text",     // Ollama model name
    "openai_model": "text-embedding-3-small",
    "dimension": null,               // null = auto-detect
    "ollama_url": "http://localhost:11434"
  }
"""

import logging

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """Abstract embedder interface."""
    dim: int = 0
    name: str = "unknown"

    def encode(self, text: str):
        """Encode text to numpy float32 vector."""
        raise NotImplementedError

    def encode_batch(self, texts: list[str]) -> list:
        """Encode multiple texts. Default: sequential calls."""
        return [self.encode(t) for t in texts]


class OllamaEmbedder(BaseEmbedder):
    """Embedder using Ollama /api/embeddings endpoint."""

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434"):
        import numpy as np
        self._model = model
        self._url = f"{base_url}/api/embeddings"
        self._np = np
        self.name = f"ollama/{model}"
        # Probe dimension
        test = self.encode("test")
        self.dim = len(test)

    def encode(self, text: str):
        import urllib.request
        import json as _json
        body = _json.dumps({"model": self._model, "prompt": text}).encode()
        req = urllib.request.Request(
            self._url, data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read())
        return self._np.array(data["embedding"], dtype="float32")

    def encode_batch(self, texts: list[str]) -> list:
        # Ollama doesn't support batch — sequential
        return [self.encode(t) for t in texts]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using sentence-transformers library."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)
        self.name = f"st/{model_name}"
        test = self._model.encode("test")
        self.dim = len(test)

    def encode(self, text: str):
        return self._model.encode(text)

    def encode_batch(self, texts: list[str]) -> list:
        # sentence-transformers supports native batch
        return list(self._model.encode(texts))


class OpenAIEmbedder(BaseEmbedder):
    """Embedder using OpenAI Embeddings API (text-embedding-3-small/large)."""

    def __init__(self, model: str = "text-embedding-3-small",
                 api_key: str = "", dimension: int | None = None):
        import numpy as np
        self._np = np
        self._model = model
        self._api_key = api_key
        self._dimension = dimension  # Matryoshka: truncate to this dim
        self.name = f"openai/{model}"
        # Probe dimension
        test = self.encode("test")
        self.dim = len(test)

    def encode(self, text: str):
        import openai
        client = openai.OpenAI(api_key=self._api_key)
        kwargs = {"model": self._model, "input": text}
        if self._dimension:
            kwargs["dimensions"] = self._dimension
        resp = client.embeddings.create(**kwargs)
        return self._np.array(resp.data[0].embedding, dtype="float32")

    def encode_batch(self, texts: list[str]) -> list:
        import openai
        client = openai.OpenAI(api_key=self._api_key)
        kwargs = {"model": self._model, "input": texts}
        if self._dimension:
            kwargs["dimensions"] = self._dimension
        resp = client.embeddings.create(**kwargs)
        # Sort by index to preserve order
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [self._np.array(d.embedding, dtype="float32") for d in sorted_data]


def _detect_ollama_model(base_url: str = "http://localhost:11434") -> str | None:
    """Auto-detect best embedding model from running Ollama instance."""
    try:
        import urllib.request
        import json as _json
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                data = _json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                # Prefer dedicated embedding models
                embed_models = [m for m in models if "embed" in m or "minilm" in m]
                if embed_models:
                    return embed_models[0]
    except Exception:
        pass
    return None


def create_embedder(config: dict) -> BaseEmbedder | None:
    """Factory: create the best available embedder based on config.

    Config lookup:
      1. config["rag"]["embedding"] — primary settings
      2. Falls back to auto-detection

    Returns None if no embedder is available (keyword-only mode).
    """
    rag_cfg = config.get("rag", {})
    emb_cfg = rag_cfg.get("embedding", {})
    provider = emb_cfg.get("provider", "auto")

    if provider == "none":
        logger.info("Embedder disabled by config (provider=none)")
        return None

    # ── Explicit provider selection ──

    if provider == "ollama":
        return _create_ollama(emb_cfg)

    if provider == "sentence_transformers":
        return _create_sentence_transformer(emb_cfg)

    if provider == "openai":
        return _create_openai(emb_cfg, config)

    # ── Auto: try each in order ──

    if provider == "auto":
        # 1. Ollama (free, local)
        ollama_url = emb_cfg.get("ollama_url", "http://localhost:11434")
        model = emb_cfg.get("model") or _detect_ollama_model(ollama_url)
        if model:
            try:
                emb = OllamaEmbedder(model=model, base_url=ollama_url)
                logger.info("Using Ollama embeddings: %s (dim=%d)", model, emb.dim)
                return emb
            except Exception as e:
                logger.debug("Ollama embedder failed: %s", e)

        # 2. sentence-transformers
        try:
            st_model = emb_cfg.get("st_model", "all-MiniLM-L6-v2")
            emb = SentenceTransformerEmbedder(st_model)
            logger.info("Using sentence-transformers: %s (dim=%d)", st_model, emb.dim)
            return emb
        except ImportError:
            logger.debug("sentence-transformers not installed")
        except Exception as e:
            logger.debug("SentenceTransformer failed: %s", e)

        # 3. OpenAI (if key available)
        try:
            from .config import get_api_key
            api_key = get_api_key("openai") or ""
            if api_key:
                openai_model = emb_cfg.get("openai_model", "text-embedding-3-small")
                dim = emb_cfg.get("dimension")
                emb = OpenAIEmbedder(model=openai_model, api_key=api_key, dimension=dim)
                logger.info("Using OpenAI embeddings: %s (dim=%d)", openai_model, emb.dim)
                return emb
        except Exception as e:
            logger.debug("OpenAI embedder failed: %s", e)

        logger.info("No embedder available — keyword-only search")
        return None

    logger.warning("Unknown embedding provider: %s", provider)
    return None


def _create_ollama(emb_cfg: dict) -> BaseEmbedder | None:
    ollama_url = emb_cfg.get("ollama_url", "http://localhost:11434")
    model = emb_cfg.get("model") or _detect_ollama_model(ollama_url) or "nomic-embed-text"
    try:
        emb = OllamaEmbedder(model=model, base_url=ollama_url)
        logger.info("Using Ollama embeddings: %s (dim=%d)", model, emb.dim)
        return emb
    except Exception as e:
        logger.warning("Ollama embedder failed: %s", e)
        return None


def _create_sentence_transformer(emb_cfg: dict) -> BaseEmbedder | None:
    model = emb_cfg.get("st_model", "all-MiniLM-L6-v2")
    try:
        emb = SentenceTransformerEmbedder(model)
        logger.info("Using sentence-transformers: %s (dim=%d)", model, emb.dim)
        return emb
    except ImportError:
        logger.warning("sentence-transformers not installed")
        return None
    except Exception as e:
        logger.warning("SentenceTransformer failed: %s", e)
        return None


def _create_openai(emb_cfg: dict, config: dict) -> BaseEmbedder | None:
    try:
        from .config import get_api_key
        api_key = get_api_key("openai") or ""
        if not api_key:
            logger.warning("OpenAI API key not found for embeddings")
            return None
        model = emb_cfg.get("openai_model", "text-embedding-3-small")
        dim = emb_cfg.get("dimension")
        emb = OpenAIEmbedder(model=model, api_key=api_key, dimension=dim)
        logger.info("Using OpenAI embeddings: %s (dim=%d)", model, emb.dim)
        return emb
    except Exception as e:
        logger.warning("OpenAI embedder failed: %s", e)
        return None
