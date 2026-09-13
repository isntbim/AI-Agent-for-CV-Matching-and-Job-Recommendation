"""
src/core/config.py — Centralized Settings & Environment Management
Reads environment variables (from .env or system env) using pydantic-settings.
All database credentials and model paths flow through this single config object.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Global application settings loaded from environment variables / .env file.
    Priority: Environment Variables > .env file > default values.
    """

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------------------------------------------------------------------------
    # PostgreSQL / pgvector Database Settings
    # ---------------------------------------------------------------------------
    pg_host: str = Field(
        default="capstonedb1.postgres.database.azure.com",
        description="PostgreSQL hostname (Azure Flexible Server FQDN)",
    )
    pg_port: int = Field(default=5432, description="PostgreSQL port")
    pg_db: str = Field(default="postgres", description="PostgreSQL database name")
    pg_user: str = Field(default="capstonedbadmin", description="PostgreSQL admin username")
    pg_password: str = Field(default="", description="PostgreSQL password")
    pg_sslmode: str = Field(default="require", description="SSL mode: require for Azure")

    # ---------------------------------------------------------------------------
    # Vector Store Settings
    # ---------------------------------------------------------------------------
    pg_table_name: str = Field(default="cv_embeddings", description="pgvector table name")
    vector_dimension: int = Field(default=1024, description="Embedding vector dimensions (BGE-M3)")
    hnsw_m: int = Field(default=16, description="HNSW index: max connections per layer")
    hnsw_ef_construction: int = Field(default=64, description="HNSW index: dynamic candidate list size")

    # ---------------------------------------------------------------------------
    # Embedding Model Settings
    # ---------------------------------------------------------------------------
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model id for the dense embedding model",
    )
    embedding_batch_size: int = Field(default=32, description="Batch size for embedding inference")
    embedding_max_seq_length: int = Field(default=8192, description="Max token length for BGE-M3")
    embedding_device: Optional[str] = Field(
        default=None,
        description="Torch device: 'cuda', 'cpu', or None for auto-detect",
    )

    @property
    def pg_conninfo(self) -> str:
        """
        Returns a psycopg-compatible connection string.
        Example: 'host=... port=5432 dbname=... user=... password=... sslmode=require'
        """
        return (
            f"host={self.pg_host} "
            f"port={self.pg_port} "
            f"dbname={self.pg_db} "
            f"user={self.pg_user} "
            f"password={self.pg_password} "
            f"sslmode={self.pg_sslmode} "
            f"connect_timeout=15"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns a cached singleton Settings instance.
    Use this everywhere instead of instantiating Settings() directly.
    """
    return Settings()
