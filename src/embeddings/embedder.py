"""
src/embeddings/embedder.py — Embedding Service for CV Matching AI Agent
=======================================================================
Provides a Protocol-based, swappable embedding interface for the Day 3
vector pipeline. Follows the same architecture pattern as src/parser/llm_extractor.py.

Clients:
    EmbeddingClient   — @runtime_checkable Protocol (interface contract)
    BGEEmbedder       — Production: BAAI/bge-m3 via sentence-transformers (1024-dim)
    MockEmbedder      — Testing: deterministic, hash-seeded vectors (zero dependencies)

Convenience functions:
    embed_resume()           — ResumeSchema → embedding vector
    embed_job()              — raw JD text → embedding vector
    batch_embed_resumes()    — List[ResumeSchema] → 2D ndarray (GPU-batched)

Run tests:
    .venv\\Scripts\\python.exe -m pytest tests/unit/test_embedder.py -v
"""

from __future__ import annotations

import hashlib
from typing import List, Optional, Protocol, runtime_checkable

import numpy as np
from loguru import logger

# ---------------------------------------------------------------------------
# Protocol (Interface Contract)
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingClient(Protocol):
    """
    Protocol defining the interface for all embedding model implementations.
    Any class implementing embed() and embed_batch() with correct signatures
    satisfies this protocol — no inheritance required.
    """

    @property
    def dimension(self) -> int:
        """Output vector dimensionality (e.g. 1024 for BGE-M3)."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string into a dense vector.

        Args:
            text: Input text (resume text, job description, etc.)

        Returns:
            np.ndarray of shape (dimension,), dtype float32, L2-normalized.
        """
        ...

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of text strings in a single batched forward pass.

        Args:
            texts: List of input strings.

        Returns:
            np.ndarray of shape (len(texts), dimension), dtype float32.
        """
        ...


# ---------------------------------------------------------------------------
# Production Implementation — BGE-M3 via sentence-transformers
# ---------------------------------------------------------------------------


class BGEEmbedder:
    """
    Production embedding client using BAAI/bge-m3 (1024-dim, multilingual).

    Automatically selects CUDA GPU if available, falls back to CPU.
    All embeddings are L2-normalized by default for cosine similarity compatibility.

    Requires: sentence-transformers, torch (with CUDA for GPU acceleration)

    Hardware tested: NVIDIA RTX 4070 (12GB VRAM) — ~6.3 GB used by Qwen2.5-7B,
    BGE-M3 adds ~2.2 GB, leaving adequate headroom.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_seq_length: int = 8192,
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Initialize and load the BGE-M3 model.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cuda', 'cpu', or None (auto-detect).
            batch_size: Number of texts per forward pass.
            max_seq_length: Maximum token length (BGE-M3 supports up to 8192).
            normalize_embeddings: L2-normalize output vectors (required for cosine similarity).
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "[BGEEmbedder] sentence-transformers is required. "
                "Install via: .venv/Scripts/pip.exe install sentence-transformers"
            ) from e

        self._batch_size = batch_size
        self._normalize = normalize_embeddings
        self._model_name = model_name

        # Auto-detect device
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self._device = device
        logger.info(f"[BGEEmbedder] Loading model '{model_name}' on device='{device}'...")

        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = max_seq_length
        # get_embedding_dimension() is the updated API in newer sentence-transformers versions
        get_dim = getattr(
            self._model, "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        self._dim = get_dim()

        logger.info(f"[BGEEmbedder] Model loaded. Dimension={self._dim}, device={device}.")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input text (may be empty string — returns zero-like normalized vector).

        Returns:
            np.ndarray shape (1024,), dtype float32, L2-normalized.
        """
        logger.debug(f"[BGEEmbedder] Embedding single text ({len(text)} chars).")
        result = self._model.encode(
            [text],
            batch_size=1,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return result[0].astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts in a single batched forward pass (GPU-parallel).

        Args:
            texts: List of input strings (may be empty list).

        Returns:
            np.ndarray shape (len(texts), 1024), dtype float32, L2-normalized.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)

        logger.debug(f"[BGEEmbedder] Batch embedding {len(texts)} texts.")
        result = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Test Double — MockEmbedder (deterministic, zero ML dependencies)
# ---------------------------------------------------------------------------


class MockEmbedder:
    """
    Zero-dependency embedding test double for unit and integration tests.

    Produces DETERMINISTIC vectors: the same input text ALWAYS produces the
    same output vector across calls and test runs. Different input texts
    produce reliably DIFFERENT vectors (via MD5 hash seeding of numpy RNG).

    All output vectors are L2-normalized to match BGE-M3 production output.

    This means mock-based integration tests can verify ranked ordering of
    cosine similarity results without loading any ML model.

    Usage:
        embedder = MockEmbedder(dimension=1024)
        vec = embedder.embed("Software Engineer with Python experience")
        assert vec.shape == (1024,)
    """

    def __init__(self, dimension: int = 1024) -> None:
        """
        Args:
            dimension: Output vector size. Default 1024 to match BGE-M3.
        """
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Converts text to a deterministic normalized float32 vector.
        Uses MD5 hash bytes as seed for numpy RandomState to guarantee
        reproducibility while still differentiating distinct texts.
        """
        seed_bytes = hashlib.md5(text.encode("utf-8")).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "little")
        rng = np.random.RandomState(seed_int)
        vec = rng.randn(self._dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed(self, text: str) -> np.ndarray:
        """
        Returns a deterministic L2-normalized vector for the given text.

        Args:
            text: Any string (including empty string).

        Returns:
            np.ndarray shape (dimension,), dtype float32.
        """
        return self._text_to_vector(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Returns a 2D array of deterministic vectors.

        Args:
            texts: List of strings (may be empty).

        Returns:
            np.ndarray shape (len(texts), dimension), dtype float32.
        """
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        return np.stack([self._text_to_vector(t) for t in texts])


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def embed_resume(resume: "ResumeSchema", client: EmbeddingClient) -> np.ndarray:
    """
    Generates a dense embedding vector from a parsed ResumeSchema.

    Internally calls resume.to_embedding_text() to synthesize a semantically
    rich text representation before embedding.

    Args:
        resume: A validated ResumeSchema (from src.parser.schemas).
        client: Any EmbeddingClient implementation.

    Returns:
        np.ndarray shape (client.dimension,), dtype float32.
    """
    embedding_text = resume.to_embedding_text()
    logger.debug(
        f"[embed_resume] Embedding resume for '{resume.basics.name}' "
        f"({len(embedding_text)} chars)."
    )
    return client.embed(embedding_text)


def embed_job(jd_text: str, client: EmbeddingClient) -> np.ndarray:
    """
    Generates a dense embedding vector from a job description text.

    Args:
        jd_text: Raw or structured job description string.
        client: Any EmbeddingClient implementation.

    Returns:
        np.ndarray shape (client.dimension,), dtype float32.
    """
    logger.debug(f"[embed_job] Embedding job description ({len(jd_text)} chars).")
    return client.embed(jd_text)


def batch_embed_resumes(
    resumes: "List[ResumeSchema]", client: EmbeddingClient
) -> np.ndarray:
    """
    Batch embeds a list of ResumeSchema objects in a single GPU forward pass.

    Converts each resume to its embedding text via to_embedding_text() and
    calls embed_batch() for maximum GPU parallelism.

    Args:
        resumes: List of validated ResumeSchema objects.
        client: Any EmbeddingClient implementation.

    Returns:
        np.ndarray shape (len(resumes), client.dimension), dtype float32.
        Returns empty array of shape (0, dimension) for empty input.
    """
    if not resumes:
        return np.empty((0, client.dimension), dtype=np.float32)

    texts = [r.to_embedding_text() for r in resumes]
    logger.info(f"[batch_embed_resumes] Batch embedding {len(texts)} resumes.")
    return client.embed_batch(texts)
