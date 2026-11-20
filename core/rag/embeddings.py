"""Configurable embedding providers: offline hash n-gram, local SentenceTransformers, and API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np
from pydantic import SecretStr
from sklearn.feature_extraction.text import HashingVectorizer

from core.config import Settings, get_settings


class EmbeddingError(Exception):
    """Raised when generating text embeddings fails."""


class BaseEmbedder(ABC):
    """Provider-agnostic text embedding interface."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return output vector dimensionality."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts into dense L2-normalized float vectors."""

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        return self.embed_texts([query])[0]


class LocalHashEmbedder(BaseEmbedder):
    """Deterministic zero-network embedder combining hashed word + character n-grams.

    Provides fast, reproducible retrieval for local development, offline environments,
    and unit tests without requiring external model downloads or API keys.
    """

    def __init__(self, word_dims: int = 1024, char_dims: int = 512) -> None:
        """Configure dual word/subword hashing vectorizers."""
        self._word_dims = word_dims
        self._char_dims = char_dims
        self._word_vec = HashingVectorizer(
            n_features=word_dims,
            ngram_range=(1, 2),
            analyzer="word",
            lowercase=True,
            alternate_sign=False,
            norm="l2",
        )
        self._char_vec = HashingVectorizer(
            n_features=char_dims,
            ngram_range=(3, 5),
            analyzer="char_wb",
            lowercase=True,
            alternate_sign=False,
            norm="l2",
        )

    @property
    def dimension(self) -> int:
        """Return combined word + character embedding dimensionality."""
        return self._word_dims + self._char_dims

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts into L2-normalized dense vectors."""
        if not texts:
            return []
        text_list = [str(t) for t in texts]
        w_mat = self._word_vec.transform(text_list).toarray().astype(np.float32) * 1.25
        c_mat = self._char_vec.transform(text_list).toarray().astype(np.float32) * 0.75
        combined = np.concatenate([w_mat, c_mat], axis=1)
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normalized = combined / norms
        return [row.tolist() for row in normalized]


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local dense neural embedder backed by ``sentence-transformers``."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        """Lazily load the requested local SentenceTransformer model."""
        self._model_name = model_name
        self._model: Any = None
        self._dim: int | None = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "sentence-transformers is required for SentenceTransformerEmbedder."
                ) from exc
            self._model = SentenceTransformer(self._model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension from the loaded model."""
        self._get_model()
        assert self._dim is not None
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts into normalized float vectors using SentenceTransformers."""
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [vec.tolist() for vec in vectors]


class OpenAICompatibleEmbedder(BaseEmbedder):
    """API-backed embedder calling an OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        *,
        api_key: SecretStr,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        """Initialize the API embedding client without logging or exposing ``api_key``."""
        if not api_key.get_secret_value().strip():
            raise EmbeddingError("LLM_API_KEY is required for API-backed embeddings.")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._http_client = http_client
        self._dim = 1536

    @property
    def dimension(self) -> int:
        """Return last observed or default embedding dimension."""
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Request embeddings from the OpenAI-compatible HTTP endpoint."""
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key.get_secret_value().strip()}",
        }
        payload = {"model": self._model, "input": list(texts)}
        owns_client = self._http_client is None
        client = self._http_client or httpx.Client(timeout=self._timeout_seconds)
        try:
            response = client.post(
                url,
                headers=headers,
                json=payload,
                timeout=self._timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Embedding HTTP request failed: {exc}") from exc
        finally:
            if owns_client:
                client.close()

        if response.status_code >= 400:
            raise EmbeddingError(f"Embedding API returned HTTP {response.status_code}.")

        data = response.json()
        items = sorted(data["data"], key=lambda item: int(item["index"]))
        vectors = [[float(x) for x in item["embedding"]] for item in items]
        if vectors:
            self._dim = len(vectors[0])
        return vectors


def create_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Create the configured embedder from ``Settings.embedding_model`` and ``llm_provider``."""
    cfg = settings or get_settings()
    model_name = cfg.embedding_model.strip()

    if model_name.lower() in {"local-hash", "hash", "mock"} or cfg.llm_provider == "mock":
        return LocalHashEmbedder()

    if model_name.startswith("sentence-transformers/") or model_name.startswith("local:"):
        cleaned = model_name.removeprefix("local:")
        return SentenceTransformerEmbedder(model_name=cleaned)

    if cfg.llm_api_key.get_secret_value().strip():
        return OpenAICompatibleEmbedder(api_key=cfg.llm_api_key, model=model_name)

    return LocalHashEmbedder()

