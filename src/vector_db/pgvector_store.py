"""
src/vector_db/pgvector_store.py — Vector Store for CV Matching AI Agent
=======================================================================
Provides a Protocol-based, swappable vector store interface backed by
PostgreSQL + pgvector (production) and a pure-NumPy InMemoryVectorStore
(zero-dependency testing double).

Classes:
    SearchResult        — NamedTuple: id, score, payload
    VectorStore         — @runtime_checkable Protocol (interface contract)
    PGVectorStore       — Production: Azure PostgreSQL + pgvector (HNSW + Cosine <=>)
    InMemoryVectorStore — Testing: pure NumPy cosine similarity (zero Docker required)

Run unit tests:
    .venv\\Scripts\\python.exe -m pytest tests/unit/test_pgvector_store.py -v
"""

from __future__ import annotations

import json
import uuid
from typing import Dict, List, NamedTuple, Optional, Protocol, runtime_checkable

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------


class SearchResult(NamedTuple):
    """
    A single vector similarity search result.

    Attributes:
        id:      Unique identifier of the stored point (string UUID).
        score:   Cosine similarity score in range [-1.0, 1.0]. Higher = more similar.
        payload: Metadata dict stored alongside the vector (name, label, skills, etc.)
    """

    id: str
    score: float
    payload: Dict


# ---------------------------------------------------------------------------
# Protocol (Interface Contract)
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorStore(Protocol):
    """
    Protocol defining the interface for all vector store implementations.
    Any class implementing these methods satisfies the protocol.
    """

    def create_table(self) -> None:
        """Create the vector table and HNSW index if they do not exist."""
        ...

    def table_exists(self) -> bool:
        """Return True if the vector table already exists."""
        ...

    def drop_table(self) -> None:
        """Drop the vector table if it exists."""
        ...

    def upsert(self, point_id: str, vector: np.ndarray, payload: Dict) -> None:
        """
        Insert or update a single vector point.
        If a record with the same point_id already exists, update it.
        """
        ...

    def upsert_batch(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: List[Dict],
    ) -> None:
        """Batch insert or update multiple vector points."""
        ...

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Perform cosine similarity search against all stored vectors.

        Args:
            query_vector:    Query embedding np.ndarray shape (dimension,).
            top_k:           Maximum number of results to return.
            score_threshold: Minimum score filter (inclusive). None = no filter.

        Returns:
            List of SearchResult sorted by score descending.
        """
        ...

    def count(self) -> int:
        """Return the total number of stored vector points."""
        ...


# ---------------------------------------------------------------------------
# Production Implementation — Azure PostgreSQL + pgvector
# ---------------------------------------------------------------------------

# DDL templates separated for testability
_DDL_CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector;"

_DDL_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS {table} (
    id          VARCHAR(255) PRIMARY KEY,
    vector      vector({dim}) NOT NULL,
    payload     JSONB DEFAULT '{{}}'::jsonb,
    created_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
"""

_DDL_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS {table}_hnsw_idx
ON {table} USING hnsw (vector vector_cosine_ops)
WITH (m = {m}, ef_construction = {ef});
"""

_SQL_UPSERT = """
INSERT INTO {table} (id, vector, payload, updated_at)
VALUES (%(id)s, %(vector)s, %(payload)s, CURRENT_TIMESTAMP)
ON CONFLICT (id) DO UPDATE SET
    vector     = EXCLUDED.vector,
    payload    = EXCLUDED.payload,
    updated_at = EXCLUDED.updated_at;
"""

_SQL_SEARCH = """
SELECT id,
       (1.0 - (vector <=> %(query_vec)s::vector))::float AS score,
       payload
FROM {table}
ORDER BY vector <=> %(query_vec)s::vector ASC
LIMIT %(top_k)s;
"""

_SQL_SEARCH_WITH_THRESHOLD = """
SELECT id,
       (1.0 - (vector <=> %(query_vec)s::vector))::float AS score,
       payload
FROM {table}
WHERE (1.0 - (vector <=> %(query_vec)s::vector)) >= %(threshold)s
ORDER BY vector <=> %(query_vec)s::vector ASC
LIMIT %(top_k)s;
"""

_SQL_COUNT = "SELECT COUNT(*) FROM {table};"
_SQL_TABLE_EXISTS = (
    "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
    "WHERE table_name = %(table)s AND table_schema = 'public');"
)
_SQL_DROP_TABLE = "DROP TABLE IF EXISTS {table};"


class PGVectorStore:
    """
    Production vector store backed by Azure PostgreSQL 16 + pgvector 0.8.2.

    Uses:
        - HNSW index with vector_cosine_ops for fast Approximate Nearest Neighbor search.
        - Cosine distance operator <=> for semantic similarity.
        - Idempotent ON CONFLICT upserts for safe re-ingestion.
        - SSL required (sslmode='require') for secure Azure connection.

    Example:
        from src.core.config import get_settings
        settings = get_settings()
        store = PGVectorStore.from_settings(settings)
        store.create_table()
        store.upsert("resume-001", embedding_vector, {"name": "Lê Trí Dũng"})
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        conninfo: str,
        table_name: str = "cv_embeddings",
        vector_size: int = 1024,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
    ) -> None:
        """
        Args:
            conninfo:             psycopg connection string (host, port, dbname, user, password, sslmode).
            table_name:           Name of the table storing vector points.
            vector_size:          Dimensionality of stored vectors (must match embedding model output).
            hnsw_m:               HNSW max connections per layer (higher = better recall, more memory).
            hnsw_ef_construction: HNSW build-time candidate list size (higher = better index quality).
        """
        self._conninfo = conninfo
        self._table = table_name
        self._vector_size = vector_size
        self._hnsw_m = hnsw_m
        self._hnsw_ef = hnsw_ef_construction

    @classmethod
    def from_settings(cls, settings: "Settings") -> "PGVectorStore":
        """
        Convenience factory that reads configuration from a Settings instance.

        Args:
            settings: Settings object from src.core.config.

        Returns:
            Configured PGVectorStore instance.
        """
        return cls(
            conninfo=settings.pg_conninfo,
            table_name=settings.pg_table_name,
            vector_size=settings.vector_dimension,
            hnsw_m=settings.hnsw_m,
            hnsw_ef_construction=settings.hnsw_ef_construction,
        )

    def _connect(self):
        """Open a new psycopg connection with pgvector type registration."""
        import psycopg
        from pgvector.psycopg import register_vector

        conn = psycopg.connect(self._conninfo)
        register_vector(conn)
        return conn

    # -----------------------------------------------------------------------
    # DDL helpers (public for testability)
    # -----------------------------------------------------------------------

    def _build_create_table_ddl(self) -> str:
        return _DDL_CREATE_TABLE.format(table=self._table, dim=self._vector_size)

    def _build_create_index_ddl(self) -> str:
        return _DDL_CREATE_INDEX.format(
            table=self._table, m=self._hnsw_m, ef=self._hnsw_ef
        )

    def _build_upsert_sql(self) -> str:
        return _SQL_UPSERT.format(table=self._table)

    def _build_search_sql(self, with_threshold: bool = False) -> str:
        if with_threshold:
            return _SQL_SEARCH_WITH_THRESHOLD.format(table=self._table)
        return _SQL_SEARCH.format(table=self._table)

    # -----------------------------------------------------------------------
    # VectorStore Protocol Implementation
    # -----------------------------------------------------------------------

    def create_table(self) -> None:
        """
        Creates the vector table and HNSW cosine index.
        Idempotent — safe to call multiple times.
        """
        logger.info(f"[PGVectorStore] Creating table '{self._table}' (if not exists)...")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_DDL_CREATE_EXTENSION)
                cur.execute(self._build_create_table_ddl())
                cur.execute(self._build_create_index_ddl())
            conn.commit()
        logger.info(f"[PGVectorStore] Table '{self._table}' ready.")

    def table_exists(self) -> bool:
        """Returns True if the table exists in the public schema."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_SQL_TABLE_EXISTS, {"table": self._table})
                return cur.fetchone()[0]

    def drop_table(self) -> None:
        """Drops the vector table if it exists. Idempotent."""
        logger.warning(f"[PGVectorStore] Dropping table '{self._table}'...")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_SQL_DROP_TABLE.format(table=self._table))
            conn.commit()
        logger.info(f"[PGVectorStore] Table '{self._table}' dropped.")

    def upsert(self, point_id: str, vector: np.ndarray, payload: Dict) -> None:
        """
        Insert or update a single vector point (idempotent via ON CONFLICT).

        Args:
            point_id: Unique string identifier (e.g. UUID, file path hash).
            vector:   np.ndarray shape (vector_size,), dtype float32.
            payload:  Metadata dict (e.g. {"name": "...", "label": "...", "skills": [...]}).
        """
        logger.debug(f"[PGVectorStore] Upserting point id='{point_id}'.")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    self._build_upsert_sql(),
                    {
                        "id": point_id,
                        "vector": vector.tolist(),
                        "payload": json.dumps(payload),
                    },
                )
            conn.commit()

    def upsert_batch(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: List[Dict],
    ) -> None:
        """
        Batch insert or update multiple vector points in a single transaction.

        Args:
            ids:      List of unique string identifiers.
            vectors:  List of np.ndarray, each shape (vector_size,).
            payloads: List of metadata dicts.
        """
        if not ids:
            logger.debug("[PGVectorStore] upsert_batch called with empty list — skipping.")
            return

        logger.info(f"[PGVectorStore] Batch upserting {len(ids)} points...")
        sql = self._build_upsert_sql()
        records = [
            {"id": i, "vector": v.tolist(), "payload": json.dumps(p)}
            for i, v, p in zip(ids, vectors, payloads)
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, records)
            conn.commit()
        logger.info(f"[PGVectorStore] Batch upsert of {len(ids)} points complete.")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Cosine similarity search using pgvector's <=> operator.

        Args:
            query_vector:    Query np.ndarray shape (vector_size,).
            top_k:           Max results to return.
            score_threshold: Minimum cosine similarity score filter.

        Returns:
            List[SearchResult] sorted by score descending (highest similarity first).
        """
        logger.debug(
            f"[PGVectorStore] Searching top_k={top_k}, threshold={score_threshold}."
        )
        use_threshold = score_threshold is not None
        sql = self._build_search_sql(with_threshold=use_threshold)
        params: Dict = {"query_vec": query_vector.tolist(), "top_k": top_k}
        if use_threshold:
            params["threshold"] = score_threshold

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results = [
            SearchResult(id=str(row[0]), score=float(row[1]), payload=row[2])
            for row in rows
        ]
        logger.debug(f"[PGVectorStore] Search returned {len(results)} results.")
        return results

    def count(self) -> int:
        """Returns the total number of stored vector points."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_SQL_COUNT.format(table=self._table))
                return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Test Double — InMemoryVectorStore (pure NumPy, zero external dependencies)
# ---------------------------------------------------------------------------


class InMemoryVectorStore:
    """
    Zero-dependency in-memory vector store for unit and integration testing.

    Implements the full VectorStore protocol using plain Python dictionaries
    and NumPy cosine similarity. No database, no Docker, no network required.

    Semantics precisely mirror PGVectorStore:
        - Idempotent upsert (same id → update)
        - Cosine similarity search with optional score threshold
        - Ordered results (highest score first)

    Usage:
        store = InMemoryVectorStore(dimension=1024)
        store.create_table()
        store.upsert("cv-001", vector, {"name": "Test"})
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self, dimension: int = 1024) -> None:
        self._dim = dimension
        self._table_name: Optional[str] = None
        self._store: Dict[str, Dict] = {}  # {id: {"vector": ndarray, "payload": dict}}

    # -----------------------------------------------------------------------
    # VectorStore Protocol Implementation
    # -----------------------------------------------------------------------

    def create_table(self, table_name: str = "cv_embeddings") -> None:
        """Initialize the in-memory store. Idempotent."""
        if self._table_name is None:
            self._table_name = table_name
            self._store = {}
            logger.debug(f"[InMemoryVectorStore] Table '{table_name}' initialized.")

    def table_exists(self) -> bool:
        return self._table_name is not None

    def drop_table(self) -> None:
        """Reset the in-memory store."""
        self._table_name = None
        self._store = {}
        logger.debug("[InMemoryVectorStore] Table dropped.")

    def upsert(self, point_id: str, vector: np.ndarray, payload: Dict) -> None:
        """Insert or update a single point."""
        if vector.shape != (self._dim,):
            raise ValueError(
                f"[InMemoryVectorStore] Expected vector shape ({self._dim},), "
                f"got {vector.shape}."
            )
        self._store[point_id] = {
            "vector": vector.astype(np.float32),
            "payload": payload,
        }

    def upsert_batch(
        self,
        ids: List[str],
        vectors: List[np.ndarray],
        payloads: List[Dict],
    ) -> None:
        """Batch insert or update multiple points."""
        for point_id, vector, payload in zip(ids, vectors, payloads):
            self.upsert(point_id, vector, payload)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Cosine similarity search using NumPy dot product on L2-normalized vectors.

        Produces results semantically equivalent to pgvector's <=> cosine operator.
        """
        if not self._store:
            return []

        ids = list(self._store.keys())
        matrix = np.stack([self._store[i]["vector"] for i in ids])  # (N, dim)

        # Cosine similarity: dot product of normalized vectors
        q_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_normalized = matrix / m_norms
        scores = matrix_normalized @ q_norm  # shape (N,)

        # Apply threshold filter
        results = []
        for idx, (point_id, score) in enumerate(zip(ids, scores)):
            score_f = float(score)
            if score_threshold is not None and score_f < score_threshold:
                continue
            results.append(
                SearchResult(
                    id=point_id,
                    score=score_f,
                    payload=self._store[point_id]["payload"],
                )
            )

        # Sort by score descending (highest similarity first)
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self._store)
