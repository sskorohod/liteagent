"""Tests for liteagent.embedders — unified embedding module."""

import importlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from liteagent.embedders import (
    BaseEmbedder,
    OllamaEmbedder,
    SentenceTransformerEmbedder,
    OpenAIEmbedder,
    create_embedder,
    _detect_ollama_model,
)


# ── BaseEmbedder ─────────────────────────────────

def test_base_embedder_interface():
    """BaseEmbedder raises NotImplementedError."""
    emb = BaseEmbedder()
    with pytest.raises(NotImplementedError):
        emb.encode("test")


def test_base_embedder_encode_batch_default():
    """Default encode_batch calls encode sequentially."""
    class DummyEmbedder(BaseEmbedder):
        dim = 3
        name = "dummy"
        def encode(self, text):
            return [len(text), 0.0, 1.0]

    emb = DummyEmbedder()
    results = emb.encode_batch(["hi", "hello", "world"])
    assert len(results) == 3
    assert results[0] == [2, 0.0, 1.0]


# ── OllamaEmbedder ──────────────────────────────

def test_ollama_embedder_encode():
    """OllamaEmbedder calls /api/embeddings and returns numpy array."""
    import numpy as np
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"embedding": [0.1, 0.2, 0.3]}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        emb = OllamaEmbedder(model="test-model", base_url="http://fake:11434")
        assert emb.dim == 3
        assert emb.name == "ollama/test-model"
        result = emb.encode("hello")
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32


def test_ollama_embedder_encode_batch():
    """encode_batch calls encode sequentially (Ollama has no batch API)."""
    import numpy as np
    call_count = 0
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"embedding": [0.1, 0.2]}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        emb = OllamaEmbedder(model="test", base_url="http://fake:11434")
        results = emb.encode_batch(["a", "b"])
        assert len(results) == 2


# ── SentenceTransformerEmbedder ──────────────────

def test_sentence_transformer_embedder():
    """SentenceTransformerEmbedder wraps sentence-transformers model."""
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode = MagicMock(
        side_effect=lambda x: np.array([0.1, 0.2, 0.3]) if isinstance(x, str)
        else [np.array([0.1, 0.2, 0.3]) for _ in x])

    mock_st_module = MagicMock()
    mock_st_module.SentenceTransformer.return_value = mock_model

    with patch.dict(sys.modules, {"sentence_transformers": mock_st_module}):
        # Re-import to use the mock
        emb = SentenceTransformerEmbedder.__new__(SentenceTransformerEmbedder)
        emb._model = mock_model
        emb.name = "st/test"
        emb.dim = 3

        result = emb.encode("hello")
        assert len(result) == 3

        results = emb.encode_batch(["a", "b"])
        assert len(results) == 2


# ── OpenAIEmbedder ───────────────────────────────

def test_openai_embedder_encode():
    """OpenAIEmbedder calls OpenAI API for single text."""
    import numpy as np

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2, 0.3, 0.4]
    mock_embedding.index = 0

    mock_resp = MagicMock()
    mock_resp.data = [mock_embedding]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_resp

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(sys.modules, {"openai": mock_openai}):
        emb = OpenAIEmbedder.__new__(OpenAIEmbedder)
        emb._model = "text-embedding-3-small"
        emb._api_key = "test-key"
        emb._dimension = None
        emb._np = np
        emb.name = "openai/text-embedding-3-small"
        emb.dim = 4

        result = emb.encode("hello")
        assert isinstance(result, np.ndarray)
        assert len(result) == 4


def test_openai_embedder_encode_batch():
    """OpenAIEmbedder sorts batch results by index."""
    import numpy as np

    embs = []
    for i in range(3):
        e = MagicMock()
        e.embedding = [float(i), 0.0]
        e.index = i
        embs.append(e)

    mock_resp = MagicMock()
    # Return in reverse order to test sorting
    mock_resp.data = list(reversed(embs))

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_resp

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(sys.modules, {"openai": mock_openai}):
        emb = OpenAIEmbedder.__new__(OpenAIEmbedder)
        emb._model = "text-embedding-3-small"
        emb._api_key = "test-key"
        emb._dimension = None
        emb._np = np
        emb.name = "openai/test"
        emb.dim = 2

        results = emb.encode_batch(["a", "b", "c"])
        assert len(results) == 3
        # Should be sorted by index
        assert results[0][0] == 0.0
        assert results[1][0] == 1.0
        assert results[2][0] == 2.0


def test_openai_embedder_with_dimension():
    """OpenAIEmbedder passes dimensions parameter for Matryoshka truncation."""
    import numpy as np

    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1, 0.2]
    mock_embedding.index = 0

    mock_resp = MagicMock()
    mock_resp.data = [mock_embedding]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_resp

    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(sys.modules, {"openai": mock_openai}):
        emb = OpenAIEmbedder.__new__(OpenAIEmbedder)
        emb._model = "text-embedding-3-small"
        emb._api_key = "key"
        emb._dimension = 256
        emb._np = np
        emb.name = "openai/test"
        emb.dim = 2

        emb.encode("test")
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["dimensions"] == 256


# ── _detect_ollama_model ─────────────────────────

def test_detect_ollama_model_found():
    """Detects embedding model from Ollama API."""
    response_data = b'{"models": [{"name": "llama3"}, {"name": "nomic-embed-text"}]}'
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = response_data
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        model = _detect_ollama_model("http://fake:11434")
        assert model == "nomic-embed-text"


def test_detect_ollama_model_none():
    """Returns None when no embedding model found."""
    response_data = b'{"models": [{"name": "llama3"}, {"name": "mistral"}]}'
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = response_data
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        model = _detect_ollama_model("http://fake:11434")
        assert model is None


def test_detect_ollama_model_error():
    """Returns None on connection error."""
    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        model = _detect_ollama_model("http://fake:11434")
        assert model is None


# ── create_embedder factory ──────────────────────

def test_create_embedder_none():
    """provider=none returns None."""
    config = {"rag": {"embedding": {"provider": "none"}}}
    assert create_embedder(config) is None


def test_create_embedder_unknown_provider():
    """Unknown provider returns None."""
    config = {"rag": {"embedding": {"provider": "foobar"}}}
    assert create_embedder(config) is None


def test_create_embedder_auto_no_providers():
    """Auto mode with no available providers returns None."""
    with patch("urllib.request.urlopen", side_effect=Exception("no connection")):
        config = {"rag": {"embedding": {"provider": "auto"}}}
        result = create_embedder(config)
        # May return None or sentence-transformers depending on environment
        # At minimum, should not crash
        assert result is None or hasattr(result, 'encode')


def test_create_embedder_empty_config():
    """Empty config uses auto mode."""
    with patch("urllib.request.urlopen", side_effect=Exception("no connection")):
        result = create_embedder({})
        assert result is None or hasattr(result, 'encode')


def test_create_embedder_ollama_explicit():
    """Explicit ollama provider creates OllamaEmbedder."""
    import numpy as np
    mock_response = MagicMock()
    mock_response.read.return_value = b'{"embedding": [0.1, 0.2, 0.3]}'
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        config = {"rag": {"embedding": {"provider": "ollama", "model": "test-model"}}}
        emb = create_embedder(config)
        assert emb is not None
        assert emb.name == "ollama/test-model"


def test_create_embedder_ollama_fail_returns_none():
    """When ollama provider is explicit but fails, returns None."""
    with patch("urllib.request.urlopen", side_effect=Exception("fail")):
        config = {"rag": {"embedding": {"provider": "ollama"}}}
        result = create_embedder(config)
        assert result is None
