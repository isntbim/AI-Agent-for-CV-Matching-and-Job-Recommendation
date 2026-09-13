"""
src/embeddings/__init__.py — Public API for the embeddings package.
"""

from src.embeddings.embedder import (
    EmbeddingClient,
    BGEEmbedder,
    MockEmbedder,
    embed_resume,
    embed_job,
    batch_embed_resumes,
)

__all__ = [
    "EmbeddingClient",
    "BGEEmbedder",
    "MockEmbedder",
    "embed_resume",
    "embed_job",
    "batch_embed_resumes",
]
