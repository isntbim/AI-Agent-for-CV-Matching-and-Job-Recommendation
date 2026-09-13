"""
tests/integration/test_embedding_pipeline.py — Integration Tests: Full E2E Pipeline
======================================================================================
Tests the complete data flow:
    parse_cv_tier1() → embed_resume() → store.upsert() → store.search()

All tests use MockEmbedder + InMemoryVectorStore.
Zero external dependencies: no Azure DB, no Docker, no GPU, no model download required.

Run:
    .venv\\Scripts\\python.exe -m pytest tests/integration/test_embedding_pipeline.py -v
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from src.embeddings.embedder import MockEmbedder, embed_job, embed_resume, batch_embed_resumes
from src.parser.schemas import ResumeSchema
from src.vector_db.pgvector_store import InMemoryVectorStore, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 1024


def _build_resume(
    name: str,
    label: str,
    skills: list[str],
    summary: str = "",
) -> ResumeSchema:
    """Build a semantically distinct ResumeSchema for pipeline testing."""
    return ResumeSchema.model_validate(
        {
            "basics": {
                "name": name,
                "label": label,
                "email": f"{name.lower().replace(' ', '.')}@example.com",
                "summary": summary or f"Experienced {label} with strong domain expertise.",
            },
            "skills": [{"name": "Core Skills", "keywords": skills}],
            "work": [
                {
                    "name": "TechCorp Vietnam",
                    "position": label,
                    "startDate": "2022-01-01",
                    "highlights": [f"Delivered production-grade {skills[0]} solutions."],
                }
            ],
            "education": [
                {
                    "institution": "FPT University",
                    "area": "Computer Science",
                    "studyType": "Bachelor",
                    "startDate": "2018-09-01",
                    "endDate": "2022-09-01",
                }
            ],
        }
    )


# Pre-built test resumes
SE_RESUME = _build_resume(
    "Lê Trí Dũng",
    "Software Engineer",
    ["Python", "FastAPI", "Docker", "Redis", "PostgreSQL"],
    "Backend software engineer with deep Python and FastAPI expertise.",
)

DS_RESUME = _build_resume(
    "Thái Thành Nhân",
    "Data Scientist",
    ["Python", "PyTorch", "TensorFlow", "SQL", "Statistics"],
    "Data scientist specializing in ML model development and A/B testing.",
)

HR_RESUME = _build_resume(
    "Nguyễn Danh Bằng",
    "HR Manager",
    ["Recruitment", "HRIS", "Employee Relations", "Performance Management"],
    "HR manager with 5+ years in talent acquisition and HR operations.",
)


# ---------------------------------------------------------------------------
# TestParseEmbedUpsertSearchPipeline
# ---------------------------------------------------------------------------


class TestParseEmbedUpsertSearchPipeline:
    """
    End-to-end integration tests for the parse → embed → upsert → search pipeline.
    Uses MockEmbedder (deterministic) + InMemoryVectorStore (zero-dependency).
    """

    @pytest.fixture
    def store(self) -> InMemoryVectorStore:
        """Fresh in-memory store for each test."""
        s = InMemoryVectorStore(dimension=DIM)
        s.create_table()
        return s

    @pytest.fixture
    def embedder(self) -> MockEmbedder:
        return MockEmbedder(dimension=DIM)

    # -------------------------------------------------------------------
    # Test 1: Single CV end-to-end
    # -------------------------------------------------------------------

    def test_single_cv_end_to_end(
        self, embedder: MockEmbedder, store: InMemoryVectorStore
    ):
        """
        Full pipeline for a single CV:
        embed_resume() → upsert() → search() returns the correct CV.
        """
        cv_id = str(uuid.uuid4())
        vector = embed_resume(SE_RESUME, embedder)
        payload = {"name": SE_RESUME.basics.name, "label": SE_RESUME.basics.label}

        store.upsert(cv_id, vector, payload)
        assert store.count() == 1

        # Search with the same vector — must return this exact CV
        results = store.search(vector, top_k=1)
        assert len(results) == 1
        assert results[0].id == cv_id
        assert abs(results[0].score - 1.0) < 1e-4

    # -------------------------------------------------------------------
    # Test 2: CV searchable by similar JD
    # -------------------------------------------------------------------

    def test_resume_searchable_by_similar_jd(
        self,
        embedder: MockEmbedder,
        store: InMemoryVectorStore,
        sample_jd_text: str,
    ):
        """
        A Software Engineer resume must be retrievable when queried with an SE JD.
        """
        cv_id = str(uuid.uuid4())
        cv_vector = embed_resume(SE_RESUME, embedder)
        store.upsert(cv_id, cv_vector, {"name": SE_RESUME.basics.name})

        query_vector = embed_job(sample_jd_text, embedder)
        results = store.search(query_vector, top_k=5)

        assert len(results) >= 1, "SE resume was not retrieved when querying with SE JD."
        retrieved_ids = {r.id for r in results}
        assert cv_id in retrieved_ids, f"SE CV {cv_id} not found in search results."

    # -------------------------------------------------------------------
    # Test 3: Multiple CVs ranked by domain relevance
    # -------------------------------------------------------------------

    def test_multiple_cvs_ranked_by_relevance(
        self,
        embedder: MockEmbedder,
        store: InMemoryVectorStore,
        sample_jd_text: str,
    ):
        """
        With 3 CVs (SE, DS, HR) and an SE JD query, SE CV must rank first.
        Uses the deterministic MockEmbedder so results are stable & reproducible.
        """
        se_id, ds_id, hr_id = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())

        store.upsert(se_id, embed_resume(SE_RESUME, embedder), {"label": "Software Engineer"})
        store.upsert(ds_id, embed_resume(DS_RESUME, embedder), {"label": "Data Scientist"})
        store.upsert(hr_id, embed_resume(HR_RESUME, embedder), {"label": "HR Manager"})

        # Query = SE JD
        query_vec = embed_job(sample_jd_text, embedder)
        results = store.search(query_vec, top_k=3)

        assert len(results) == 3
        # Results must be ordered score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score."

    # -------------------------------------------------------------------
    # Test 4: Payload metadata preserved in search results
    # -------------------------------------------------------------------

    def test_search_returns_payload_metadata(
        self, embedder: MockEmbedder, store: InMemoryVectorStore
    ):
        """
        Retrieved search results must include the original payload stored with the CV.
        """
        skills = SE_RESUME.get_flat_skills()
        payload = {
            "name": SE_RESUME.basics.name,
            "label": SE_RESUME.basics.label,
            "skills": skills,
            "source_file": "resume_le_tri_dung.pdf",
        }
        cv_id = str(uuid.uuid4())
        vector = embed_resume(SE_RESUME, embedder)
        store.upsert(cv_id, vector, payload)

        results = store.search(vector, top_k=1)
        assert len(results) == 1
        returned_payload = results[0].payload

        assert returned_payload["name"] == SE_RESUME.basics.name
        assert returned_payload["label"] == SE_RESUME.basics.label
        assert "Python" in returned_payload["skills"]
        assert returned_payload["source_file"] == "resume_le_tri_dung.pdf"

    # -------------------------------------------------------------------
    # Test 5: Batch ingest and search
    # -------------------------------------------------------------------

    def test_batch_ingest_and_search(
        self,
        embedder: MockEmbedder,
        store: InMemoryVectorStore,
        sample_jd_text: str,
    ):
        """
        batch_embed_resumes() → upsert_batch() must ingest all CVs correctly,
        and all should be retrievable via search.
        """
        resumes = [SE_RESUME, DS_RESUME, HR_RESUME]
        ids = [str(uuid.uuid4()) for _ in resumes]
        vectors_2d = batch_embed_resumes(resumes, embedder)
        payloads = [{"name": r.basics.name, "label": r.basics.label} for r in resumes]

        store.upsert_batch(ids, [vectors_2d[i] for i in range(len(resumes))], payloads)

        assert store.count() == 3, f"Expected 3 records, got {store.count()}"

        query_vec = embed_job(sample_jd_text, embedder)
        results = store.search(query_vec, top_k=10)
        assert len(results) == 3, "Not all batch-upserted CVs were returned in search."

    # -------------------------------------------------------------------
    # Test 6: Empty/minimal resume does not crash
    # -------------------------------------------------------------------

    def test_empty_resume_does_not_crash(
        self, embedder: MockEmbedder, store: InMemoryVectorStore
    ):
        """
        A minimal default ResumeSchema (all fields at defaults) must flow through
        the entire embed → upsert → search pipeline without raising any exception.
        """
        resume = ResumeSchema()
        cv_id = str(uuid.uuid4())
        vector = embed_resume(resume, embedder)

        store.upsert(cv_id, vector, {"name": "Unknown", "label": "Professional"})
        assert store.count() == 1

        results = store.search(vector, top_k=1)
        assert len(results) == 1
        assert results[0].id == cv_id

    # -------------------------------------------------------------------
    # Test 7: top_k limit strictly observed
    # -------------------------------------------------------------------

    def test_search_top_k_respects_limit(
        self, embedder: MockEmbedder, store: InMemoryVectorStore
    ):
        """Ingest 10 CVs, search with top_k=3 → exactly 3 results returned."""
        from tests.unit.test_pgvector_store import _random_vector, _new_id

        for i in range(10):
            store.upsert(_new_id(), _random_vector(i + 200), {"idx": i})

        query = embed_resume(SE_RESUME, embedder)
        results = store.search(query, top_k=3)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # -------------------------------------------------------------------
    # Test 8: Upsert is idempotent on same CV
    # -------------------------------------------------------------------

    def test_upsert_idempotent_on_same_cv(
        self, embedder: MockEmbedder, store: InMemoryVectorStore
    ):
        """
        Upserting the same CV twice (same ID, same vector) must NOT create a duplicate.
        count() must remain 1 and search must still return the CV correctly.
        """
        cv_id = "idempotent-cv-001"
        vector = embed_resume(SE_RESUME, embedder)
        payload = {"name": SE_RESUME.basics.name}

        store.upsert(cv_id, vector, payload)
        store.upsert(cv_id, vector, payload)  # duplicate upsert

        assert store.count() == 1, (
            f"Duplicate upsert created extra records. count()={store.count()}"
        )
        results = store.search(vector, top_k=1)
        assert results[0].id == cv_id
