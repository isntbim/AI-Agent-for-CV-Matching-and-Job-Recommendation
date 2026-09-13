"""
tests/unit/test_pgvector_store.py — Unit Tests for src/vector_db/pgvector_store.py
====================================================================================
Tests SearchResult, VectorStore Protocol, InMemoryVectorStore (full coverage),
and PGVectorStore SQL builder methods (no live DB required for SQL unit tests).

All primary tests use InMemoryVectorStore — zero Docker, zero network.

Run:
    .venv\\Scripts\\python.exe -m pytest tests/unit/test_pgvector_store.py -v
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.vector_db.pgvector_store import (
    InMemoryVectorStore,
    PGVectorStore,
    SearchResult,
    VectorStore,
    _DDL_CREATE_INDEX,
    _DDL_CREATE_TABLE,
    _SQL_SEARCH,
    _SQL_SEARCH_WITH_THRESHOLD,
    _SQL_UPSERT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 1024


def _random_vector(seed: int = 42) -> np.ndarray:
    """Returns a deterministic L2-normalized float32 vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _random_vectors(n: int, base_seed: int = 0) -> list[np.ndarray]:
    """Returns n deterministic distinct vectors."""
    return [_random_vector(seed=base_seed + i) for i in range(n)]


def _new_id() -> str:
    """Returns a fresh random UUID string."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# TestSearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Verify SearchResult NamedTuple structure and behaviour."""

    def test_namedtuple_fields(self):
        """SearchResult must expose id, score, and payload fields."""
        result = SearchResult(id="abc-123", score=0.95, payload={"name": "Test"})
        assert result.id == "abc-123"
        assert result.score == 0.95
        assert result.payload == {"name": "Test"}

    def test_creation(self):
        """SearchResult can be created with diverse payload types."""
        result = SearchResult(
            id=_new_id(),
            score=0.78,
            payload={"name": "Lê Trí Dũng", "skills": ["Python", "FastAPI"], "years": 3},
        )
        assert isinstance(result, SearchResult)
        assert result.score == 0.78

    def test_unpacking(self):
        """SearchResult supports tuple unpacking."""
        result = SearchResult(id="cv-001", score=0.88, payload={"label": "Engineer"})
        point_id, score, payload = result
        assert point_id == "cv-001"
        assert score == 0.88
        assert payload == {"label": "Engineer"}


# ---------------------------------------------------------------------------
# TestVectorStoreProtocol
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    """Verify VectorStore is a valid @runtime_checkable Protocol."""

    def test_in_memory_store_implements_protocol(self):
        """InMemoryVectorStore must satisfy the VectorStore protocol."""
        store = InMemoryVectorStore(dimension=DIM)
        store.create_table()
        assert isinstance(store, VectorStore), (
            "InMemoryVectorStore does not satisfy VectorStore protocol."
        )

    def test_pgvector_store_satisfies_protocol_structurally(self):
        """PGVectorStore must have all required protocol methods."""
        required_methods = [
            "create_table", "table_exists", "drop_table",
            "upsert", "upsert_batch", "search", "count",
        ]
        for method in required_methods:
            assert hasattr(PGVectorStore, method), (
                f"PGVectorStore is missing required method: '{method}'"
            )

    def test_protocol_is_runtime_checkable(self):
        """VectorStore protocol must support isinstance() checks."""
        class MinimalStore:
            def create_table(self): pass
            def table_exists(self) -> bool: return True
            def drop_table(self): pass
            def upsert(self, id, vector, payload): pass
            def upsert_batch(self, ids, vectors, payloads): pass
            def search(self, query_vector, top_k=10, score_threshold=None): return []
            def count(self) -> int: return 0

        assert isinstance(MinimalStore(), VectorStore)


# ---------------------------------------------------------------------------
# TestInMemoryVectorStoreTableManagement
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreTableManagement:
    """Lifecycle tests: create, exists, drop, recreate."""

    def test_create_table(self):
        """create_table() must make table_exists() return True."""
        store = InMemoryVectorStore(dimension=DIM)
        assert not store.table_exists()
        store.create_table()
        assert store.table_exists()

    def test_create_table_idempotent(self):
        """Calling create_table() multiple times must not raise."""
        store = InMemoryVectorStore(dimension=DIM)
        store.create_table()
        store.create_table()  # second call — must be safe
        assert store.table_exists()

    def test_table_not_exists_initially(self):
        """Fresh store must report table_exists() == False."""
        store = InMemoryVectorStore(dimension=DIM)
        assert not store.table_exists()

    def test_drop_table(self):
        """drop_table() must make table_exists() return False."""
        store = InMemoryVectorStore(dimension=DIM)
        store.create_table()
        store.drop_table()
        assert not store.table_exists()

    def test_drop_nonexistent_table(self):
        """Dropping a nonexistent table must not raise any exception."""
        store = InMemoryVectorStore(dimension=DIM)
        store.drop_table()  # table was never created

    def test_recreate_after_drop(self):
        """After drop + create, store must be empty and functional."""
        store = InMemoryVectorStore(dimension=DIM)
        store.create_table()
        store.upsert(_new_id(), _random_vector(), {"label": "test"})
        store.drop_table()
        store.create_table()
        assert store.table_exists()
        assert store.count() == 0


# ---------------------------------------------------------------------------
# TestInMemoryVectorStoreUpsert
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreUpsert:
    """Tests for single and batch upsert operations."""

    def test_upsert_single_point(self, in_memory_store: InMemoryVectorStore):
        """Single upsert must increase count to 1."""
        in_memory_store.upsert(_new_id(), _random_vector(1), {"name": "Alice"})
        assert in_memory_store.count() == 1

    def test_upsert_with_string_id(self, in_memory_store: InMemoryVectorStore):
        """String UUID and arbitrary string IDs must be accepted."""
        in_memory_store.upsert("resume-cv-001", _random_vector(2), {})
        in_memory_store.upsert(str(uuid.uuid4()), _random_vector(3), {})
        assert in_memory_store.count() == 2

    def test_upsert_with_payload(self, in_memory_store: InMemoryVectorStore):
        """Payload metadata must be stored and retrievable in search results."""
        point_id = "cv-payload-test"
        payload = {"name": "Lê Trí Dũng", "label": "AI Engineer", "skills": ["Python", "LLM"]}
        query_vec = _random_vector(10)
        in_memory_store.upsert(point_id, query_vec, payload)
        results = in_memory_store.search(query_vec, top_k=1)
        assert len(results) == 1
        assert results[0].payload["name"] == "Lê Trí Dũng"
        assert results[0].payload["label"] == "AI Engineer"

    def test_upsert_overwrites_existing_id(self, in_memory_store: InMemoryVectorStore):
        """Re-upserting the same ID must update the record, not duplicate it."""
        point_id = "duplicate-id"
        in_memory_store.upsert(point_id, _random_vector(1), {"version": 1})
        in_memory_store.upsert(point_id, _random_vector(2), {"version": 2})
        assert in_memory_store.count() == 1
        results = in_memory_store.search(_random_vector(2), top_k=1)
        assert results[0].payload["version"] == 2

    def test_upsert_multiple_distinct(self, in_memory_store: InMemoryVectorStore):
        """N distinct upserts must yield count() == N."""
        n = 7
        for i in range(n):
            in_memory_store.upsert(f"point-{i}", _random_vector(i), {"index": i})
        assert in_memory_store.count() == n

    def test_upsert_batch(self, in_memory_store: InMemoryVectorStore):
        """upsert_batch() must insert all provided records."""
        n = 5
        ids = [_new_id() for _ in range(n)]
        vectors = _random_vectors(n)
        payloads = [{"idx": i} for i in range(n)]
        in_memory_store.upsert_batch(ids, vectors, payloads)
        assert in_memory_store.count() == n

    def test_upsert_batch_empty(self, in_memory_store: InMemoryVectorStore):
        """upsert_batch() with empty lists must not raise and not change count."""
        in_memory_store.upsert_batch([], [], [])
        assert in_memory_store.count() == 0

    def test_upsert_batch_large(self, in_memory_store: InMemoryVectorStore):
        """upsert_batch() of 100 points must insert all correctly."""
        n = 100
        ids = [_new_id() for _ in range(n)]
        vectors = _random_vectors(n)
        payloads = [{} for _ in range(n)]
        in_memory_store.upsert_batch(ids, vectors, payloads)
        assert in_memory_store.count() == n

    def test_upsert_validates_dimension(self, in_memory_store: InMemoryVectorStore):
        """Upserting a vector with wrong dimension must raise ValueError."""
        wrong_dim_vector = np.ones(512, dtype=np.float32)
        with pytest.raises(ValueError, match="Expected vector shape"):
            in_memory_store.upsert(_new_id(), wrong_dim_vector, {})


# ---------------------------------------------------------------------------
# TestInMemoryVectorStoreSearch
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreSearch:
    """Tests for cosine similarity search semantics."""

    def _populate(self, store: InMemoryVectorStore, n: int) -> list[str]:
        """Helper: insert n distinct random points and return their IDs."""
        ids = [_new_id() for _ in range(n)]
        vectors = _random_vectors(n, base_seed=100)
        payloads = [{"idx": i} for i in range(n)]
        store.upsert_batch(ids, vectors, payloads)
        return ids

    def test_search_returns_list(self, in_memory_store: InMemoryVectorStore):
        """search() must return a list."""
        self._populate(in_memory_store, 3)
        results = in_memory_store.search(_random_vector(0), top_k=5)
        assert isinstance(results, list)

    def test_search_empty_table(self, in_memory_store: InMemoryVectorStore):
        """search() on an empty store must return an empty list."""
        results = in_memory_store.search(_random_vector(0), top_k=5)
        assert results == []

    def test_search_returns_correct_top_k(self, in_memory_store: InMemoryVectorStore):
        """search() with top_k=3 must return exactly 3 results from 10 points."""
        self._populate(in_memory_store, 10)
        results = in_memory_store.search(_random_vector(99), top_k=3)
        assert len(results) == 3

    def test_search_top_k_exceeds_count(self, in_memory_store: InMemoryVectorStore):
        """search() with top_k > number of points must return all points."""
        self._populate(in_memory_store, 5)
        results = in_memory_store.search(_random_vector(99), top_k=100)
        assert len(results) == 5

    def test_search_relevance_ordering(self, in_memory_store: InMemoryVectorStore):
        """Results must be sorted by score descending (highest similarity first)."""
        self._populate(in_memory_store, 10)
        results = in_memory_store.search(_random_vector(77), top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), (
            "Search results are not ordered by descending cosine similarity."
        )

    def test_search_exact_match_highest_score(self, in_memory_store: InMemoryVectorStore):
        """Querying with the exact same vector as a stored point must score ≈ 1.0."""
        vec = _random_vector(42)
        point_id = _new_id()
        # Add some noise vectors to have competition
        self._populate(in_memory_store, 5)
        in_memory_store.upsert(point_id, vec, {"exact": True})
        results = in_memory_store.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0].id == point_id
        assert abs(results[0].score - 1.0) < 1e-4, (
            f"Exact match score is {results[0].score}, expected ≈ 1.0"
        )

    def test_search_result_contains_payload(self, in_memory_store: InMemoryVectorStore):
        """Each search result must contain the full payload dict."""
        payload = {"name": "Nguyễn Danh Bằng", "label": "Software Engineer"}
        vec = _random_vector(7)
        in_memory_store.upsert(_new_id(), vec, payload)
        results = in_memory_store.search(vec, top_k=1)
        assert results[0].payload == payload

    def test_search_result_contains_id(self, in_memory_store: InMemoryVectorStore):
        """Each search result must contain the correct string ID."""
        point_id = "known-id-check"
        vec = _random_vector(8)
        in_memory_store.upsert(point_id, vec, {})
        results = in_memory_store.search(vec, top_k=1)
        assert results[0].id == point_id

    def test_search_with_score_threshold(self, in_memory_store: InMemoryVectorStore):
        """score_threshold must filter out results with score below the threshold."""
        query_vec = _random_vector(55)
        # Insert exact query vec (score=1.0) plus random dissimilar vectors
        in_memory_store.upsert("exact", query_vec, {})
        self._populate(in_memory_store, 10)
        # High threshold: only exact match should survive
        results = in_memory_store.search(query_vec, top_k=50, score_threshold=0.99)
        assert all(r.score >= 0.99 for r in results), (
            "Results below score_threshold were returned."
        )
        assert any(r.id == "exact" for r in results), "Exact match was filtered by threshold."

    def test_search_threshold_filters_all(self, in_memory_store: InMemoryVectorStore):
        """score_threshold=1.1 (impossible) must return empty list."""
        self._populate(in_memory_store, 5)
        results = in_memory_store.search(_random_vector(0), top_k=10, score_threshold=1.1)
        assert results == []

    def test_search_after_upsert_update(self, in_memory_store: InMemoryVectorStore):
        """Updating a point's vector must be reflected in subsequent searches."""
        point_id = "mutable-point"
        original_vec = _random_vector(1)
        new_vec = _random_vector(99)

        in_memory_store.upsert(point_id, original_vec, {"version": 1})
        in_memory_store.upsert(point_id, new_vec, {"version": 2})

        results = in_memory_store.search(new_vec, top_k=1)
        assert results[0].id == point_id
        assert results[0].payload["version"] == 2


# ---------------------------------------------------------------------------
# TestInMemoryVectorStoreCount
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreCount:
    """Tests for count() correctness."""

    def test_count_empty(self, in_memory_store: InMemoryVectorStore):
        """Empty store must report count() == 0."""
        assert in_memory_store.count() == 0

    def test_count_after_upserts(self, in_memory_store: InMemoryVectorStore):
        """count() must match the number of distinct IDs upserted."""
        for i in range(6):
            in_memory_store.upsert(f"id-{i}", _random_vector(i), {})
        assert in_memory_store.count() == 6

    def test_count_after_drop_and_recreate(self, in_memory_store: InMemoryVectorStore):
        """After drop + recreate, count() must be 0."""
        for i in range(3):
            in_memory_store.upsert(f"id-{i}", _random_vector(i), {})
        in_memory_store.drop_table()
        in_memory_store.create_table()
        assert in_memory_store.count() == 0


# ---------------------------------------------------------------------------
# TestPGVectorStoreSQL — SQL query builder validation (no live DB needed)
# ---------------------------------------------------------------------------


class TestPGVectorStoreSQL:
    """
    Tests that PGVectorStore generates correct SQL DDL and DML strings.
    No live PostgreSQL connection required — tests only inspect SQL strings.
    """

    @pytest.fixture
    def store(self) -> PGVectorStore:
        """A PGVectorStore instance with test configuration (no live connection)."""
        return PGVectorStore(
            conninfo="host=localhost port=5432 dbname=test user=test password=test",
            table_name="cv_embeddings",
            vector_size=1024,
            hnsw_m=16,
            hnsw_ef_construction=64,
        )

    def test_build_create_table_ddl_contains_vector_type(self, store: PGVectorStore):
        """CREATE TABLE DDL must include vector(1024) column type."""
        ddl = store._build_create_table_ddl()
        assert "vector(1024)" in ddl, f"Expected 'vector(1024)' in DDL:\n{ddl}"

    def test_build_create_table_ddl_contains_primary_key(self, store: PGVectorStore):
        """CREATE TABLE DDL must define id as PRIMARY KEY."""
        ddl = store._build_create_table_ddl()
        assert "PRIMARY KEY" in ddl

    def test_build_hnsw_index_ddl(self, store: PGVectorStore):
        """HNSW index DDL must specify cosine ops and correct parameters."""
        ddl = store._build_create_index_ddl()
        assert "USING hnsw" in ddl, "HNSW index type not found."
        assert "vector_cosine_ops" in ddl, "Cosine ops not found."
        assert "m = 16" in ddl, "HNSW m parameter not found."
        assert "ef_construction = 64" in ddl, "HNSW ef_construction not found."

    def test_build_upsert_sql_has_on_conflict(self, store: PGVectorStore):
        """Upsert SQL must use ON CONFLICT DO UPDATE for idempotency."""
        sql = store._build_upsert_sql()
        assert "ON CONFLICT (id) DO UPDATE" in sql, (
            "Upsert SQL does not use ON CONFLICT for idempotent upsert."
        )

    def test_build_search_sql_uses_cosine_operator(self, store: PGVectorStore):
        """Search SQL must use pgvector's cosine distance operator <=>."""
        sql = store._build_search_sql(with_threshold=False)
        assert "<=>" in sql, "Cosine distance operator <=> not found in search SQL."

    def test_build_search_sql_handles_threshold(self, store: PGVectorStore):
        """Search SQL with threshold must include threshold filter condition."""
        sql_no_threshold = store._build_search_sql(with_threshold=False)
        sql_with_threshold = store._build_search_sql(with_threshold=True)
        assert "threshold" in sql_with_threshold.lower() or "%(threshold)s" in sql_with_threshold, (
            "Threshold filter not present in threshold-filtered search SQL."
        )
        assert "%(threshold)s" not in sql_no_threshold, (
            "Threshold parameter found in non-threshold search SQL."
        )
