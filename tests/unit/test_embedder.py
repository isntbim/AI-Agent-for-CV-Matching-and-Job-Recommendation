"""
tests/unit/test_embedder.py — Unit Tests for src/embeddings/embedder.py
========================================================================
Tests the EmbeddingClient protocol, MockEmbedder determinism, BGEEmbedder
(conditionally skipped without GPU/model), and all three convenience functions.

Run:
    .venv\\Scripts\\python.exe -m pytest tests/unit/test_embedder.py -v
    .venv\\Scripts\\python.exe -m pytest tests/unit/test_embedder.py -v -k "not BGE"  # skip GPU tests
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embeddings.embedder import (
    EmbeddingClient,
    BGEEmbedder,
    MockEmbedder,
    embed_resume,
    embed_job,
    batch_embed_resumes,
)
from src.parser.schemas import ResumeSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIMENSION = 1024


def _make_resume(label: str = "Software Engineer", skills: list[str] | None = None) -> ResumeSchema:
    """Build a minimal but valid ResumeSchema for testing."""
    skill_keywords = skills or ["Python", "FastAPI", "Docker"]
    return ResumeSchema.model_validate(
        {
            "basics": {
                "name": "Test Candidate",
                "label": label,
                "email": "test@example.com",
                "summary": f"Experienced {label} with strong problem-solving skills.",
            },
            "skills": [{"name": "Tech", "keywords": skill_keywords}],
            "work": [
                {
                    "name": "TechCorp",
                    "position": label,
                    "startDate": "2022-01-01",
                    "highlights": ["Built scalable systems"],
                }
            ],
            "education": [
                {
                    "institution": "FPT University",
                    "area": "Computer Science",
                    "studyType": "Bachelor",
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# TestEmbeddingClientProtocol
# ---------------------------------------------------------------------------


class TestEmbeddingClientProtocol:
    """Verify that EmbeddingClient is a valid @runtime_checkable Protocol."""

    def test_mock_embedder_implements_protocol(self):
        """MockEmbedder must satisfy the EmbeddingClient protocol at runtime."""
        mock = MockEmbedder(dimension=DIMENSION)
        assert isinstance(mock, EmbeddingClient), (
            "MockEmbedder does not satisfy EmbeddingClient protocol."
        )

    def test_bge_embedder_class_satisfies_protocol_structurally(self):
        """BGEEmbedder must have the required interface methods/properties."""
        assert hasattr(BGEEmbedder, "embed"), "BGEEmbedder missing 'embed' method."
        assert hasattr(BGEEmbedder, "embed_batch"), "BGEEmbedder missing 'embed_batch' method."

    def test_protocol_is_runtime_checkable(self):
        """EmbeddingClient must support isinstance() checks at runtime."""
        # A class with correct interface should pass
        class CustomEmbedder:
            @property
            def dimension(self) -> int:
                return 512

            def embed(self, text: str) -> np.ndarray:
                return np.zeros(512, dtype=np.float32)

            def embed_batch(self, texts: list) -> np.ndarray:
                return np.zeros((len(texts), 512), dtype=np.float32)

        assert isinstance(CustomEmbedder(), EmbeddingClient)


# ---------------------------------------------------------------------------
# TestMockEmbedder
# ---------------------------------------------------------------------------


class TestMockEmbedder:
    """Comprehensive tests for MockEmbedder determinism, shapes, and edge cases."""

    @pytest.fixture
    def embedder(self) -> MockEmbedder:
        return MockEmbedder(dimension=DIMENSION)

    def test_returns_ndarray(self, embedder: MockEmbedder):
        """embed() must return a numpy ndarray."""
        result = embedder.embed("Software Engineer with Python experience")
        assert isinstance(result, np.ndarray)

    def test_correct_dimension(self, embedder: MockEmbedder):
        """embed() must return a vector of the configured dimension."""
        result = embedder.embed("AI Engineer, NLP, LLM")
        assert result.shape == (DIMENSION,), f"Expected ({DIMENSION},), got {result.shape}"

    def test_custom_dimension(self):
        """MockEmbedder must respect custom dimension parameter."""
        embedder = MockEmbedder(dimension=384)
        result = embedder.embed("Custom dimension test")
        assert result.shape == (384,)

    def test_deterministic_same_input(self, embedder: MockEmbedder):
        """Identical inputs must always produce identical vectors."""
        text = "Deterministic embedding test — same text"
        vec1 = embedder.embed(text)
        vec2 = embedder.embed(text)
        assert np.allclose(vec1, vec2), "MockEmbedder is not deterministic for same input."

    def test_different_input_different_output(self, embedder: MockEmbedder):
        """Different input texts must produce different vectors."""
        vec1 = embedder.embed("Software Engineer with Python")
        vec2 = embedder.embed("Marketing Manager with Excel")
        assert not np.allclose(vec1, vec2), "Different inputs produced identical vectors."

    def test_embed_batch_returns_2d_array(self, embedder: MockEmbedder):
        """embed_batch() must return shape (N, dimension)."""
        texts = ["Text A", "Text B", "Text C"]
        result = embedder.embed_batch(texts)
        assert result.shape == (3, DIMENSION)

    def test_embed_batch_single_item(self, embedder: MockEmbedder):
        """embed_batch() with one item must return shape (1, dimension)."""
        result = embedder.embed_batch(["Single text"])
        assert result.shape == (1, DIMENSION)

    def test_embed_batch_matches_individual_embeds(self, embedder: MockEmbedder):
        """Each row in embed_batch() must exactly match individual embed() calls."""
        texts = ["Alpha text", "Beta text", "Gamma text"]
        batch_result = embedder.embed_batch(texts)
        for i, text in enumerate(texts):
            individual = embedder.embed(text)
            assert np.allclose(batch_result[i], individual), (
                f"Batch row {i} does not match individual embed for '{text}'."
            )

    def test_embed_empty_string(self, embedder: MockEmbedder):
        """embed() must handle empty string without raising any exception."""
        result = embedder.embed("")
        assert isinstance(result, np.ndarray)
        assert result.shape == (DIMENSION,)

    def test_dimension_property(self, embedder: MockEmbedder):
        """dimension property must return the configured value."""
        assert embedder.dimension == DIMENSION

    def test_output_is_normalized(self, embedder: MockEmbedder):
        """Output vectors must be L2-normalized (norm ≈ 1.0)."""
        vec = embedder.embed("Normalized vector test")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Vector is not normalized: norm={norm}"

    def test_embed_batch_empty_list(self, embedder: MockEmbedder):
        """embed_batch() with empty list must return shape (0, dimension)."""
        result = embedder.embed_batch([])
        assert result.shape == (0, DIMENSION)


# ---------------------------------------------------------------------------
# TestBGEEmbedder — Real model (conditionally skipped without GPU/model)
# ---------------------------------------------------------------------------

def _bge_available() -> bool:
    """Check if sentence-transformers is importable (model may still need download)."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _bge_available(),
    reason="sentence-transformers not installed — skipping BGE model tests.",
)
class TestBGEEmbedder:
    """Tests for BGEEmbedder using the real BAAI/bge-m3 model (requires GPU/internet)."""

    @pytest.fixture(scope="class")
    @classmethod
    def embedder(cls) -> BGEEmbedder:
        """Load the BGE-M3 model once for the entire class (slow fixture)."""
        return BGEEmbedder(model_name="BAAI/bge-m3", normalize_embeddings=True)

    def test_initializes_correctly(self, embedder: BGEEmbedder):
        """BGEEmbedder must load the model and report correct dimension."""
        assert embedder.dimension == 1024

    def test_embed_text_returns_ndarray(self, embedder: BGEEmbedder):
        """embed() must return a numpy ndarray."""
        result = embedder.embed("Test embedding for BGE-M3 model.")
        assert isinstance(result, np.ndarray)

    def test_embed_text_correct_dimension(self, embedder: BGEEmbedder):
        """embed() must output shape (1024,) for BGE-M3."""
        result = embedder.embed("AI Engineer with PyTorch and LLM experience")
        assert result.shape == (1024,)

    def test_embed_batch_shape(self, embedder: BGEEmbedder):
        """embed_batch() for 3 texts must return shape (3, 1024)."""
        texts = ["Python developer", "Java engineer", "Data scientist"]
        result = embedder.embed_batch(texts)
        assert result.shape == (3, 1024)

    def test_vectors_are_normalized(self, embedder: BGEEmbedder):
        """Output vectors must be L2-normalized (norm ≈ 1.0)."""
        vec = embedder.embed("Normalized embedding output verification.")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4, f"Vector norm is {norm}, expected ≈ 1.0"

    def test_similar_texts_high_cosine(self, embedder: BGEEmbedder):
        """Semantically similar texts must yield cosine similarity > 0.7."""
        vec1 = embedder.embed("Senior Python backend engineer with FastAPI and Docker.")
        vec2 = embedder.embed("Experienced backend developer, Python, FastAPI, microservices.")
        cosine = float(np.dot(vec1, vec2))
        assert cosine > 0.7, f"Similar texts cosine={cosine:.4f}, expected > 0.7"

    def test_dissimilar_texts_low_cosine(self, embedder: BGEEmbedder):
        """Semantically unrelated texts must yield cosine similarity < 0.5."""
        vec1 = embedder.embed("Python backend engineer specializing in FastAPI and PostgreSQL.")
        vec2 = embedder.embed("Professional chef with expertise in French cuisine and pastry.")
        cosine = float(np.dot(vec1, vec2))
        assert cosine < 0.5, f"Unrelated texts cosine={cosine:.4f}, expected < 0.5"

    def test_embed_long_text(self, embedder: BGEEmbedder):
        """BGE-M3 must handle texts up to ~8000 characters without error."""
        long_text = "Software Engineer with Python experience. " * 200  # ~8200 chars
        result = embedder.embed(long_text)
        assert result.shape == (1024,)

    def test_embed_vietnamese_text(self, embedder: BGEEmbedder):
        """BGE-M3 must correctly embed Vietnamese text (multilingual model)."""
        vn_text = (
            "Kỹ sư phần mềm với 3 năm kinh nghiệm Python, FastAPI, và hệ thống RAG. "
            "Tốt nghiệp Đại học FPT ngành Kỹ thuật Phần mềm."
        )
        result = embedder.embed(vn_text)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1024,)


# ---------------------------------------------------------------------------
# TestEmbedResume
# ---------------------------------------------------------------------------


class TestEmbedResume:
    """Tests for the embed_resume() convenience function."""

    def test_returns_ndarray(self, mock_embedder: MockEmbedder, sample_resume: ResumeSchema):
        """embed_resume() must return a numpy ndarray."""
        result = embed_resume(sample_resume, mock_embedder)
        assert isinstance(result, np.ndarray)

    def test_correct_dimension(self, mock_embedder: MockEmbedder, sample_resume: ResumeSchema):
        """embed_resume() must return shape (1024,)."""
        result = embed_resume(sample_resume, mock_embedder)
        assert result.shape == (DIMENSION,)

    def test_uses_to_embedding_text(self, mock_embedder: MockEmbedder, sample_resume: ResumeSchema):
        """embed_resume() must call to_embedding_text() internally."""
        # The deterministic mock ensures the result matches direct embed of the text
        embedding_text = sample_resume.to_embedding_text()
        expected = mock_embedder.embed(embedding_text)
        result = embed_resume(sample_resume, mock_embedder)
        assert np.allclose(result, expected), (
            "embed_resume() did not use resume.to_embedding_text() as embedding source."
        )

    def test_minimal_resume(self, mock_embedder: MockEmbedder):
        """A minimal default ResumeSchema must embed without raising any exception."""
        resume = ResumeSchema()
        result = embed_resume(resume, mock_embedder)
        assert result.shape == (DIMENSION,)

    def test_resume_with_no_skills(self, mock_embedder: MockEmbedder):
        """Resume without any skills must still produce a valid embedding."""
        resume = ResumeSchema.model_validate(
            {"basics": {"name": "No Skills Candidate", "label": "Junior Developer"}}
        )
        result = embed_resume(resume, mock_embedder)
        assert isinstance(result, np.ndarray)
        assert result.shape == (DIMENSION,)

    def test_different_resumes_produce_different_vectors(self, mock_embedder: MockEmbedder):
        """Two distinctly different resumes must produce different vectors."""
        resume_se = _make_resume("Software Engineer", ["Python", "FastAPI", "Docker"])
        resume_hr = _make_resume("HR Manager", ["Recruitment", "HRIS", "Performance Management"])
        vec_se = embed_resume(resume_se, mock_embedder)
        vec_hr = embed_resume(resume_hr, mock_embedder)
        assert not np.allclose(vec_se, vec_hr), (
            "Two different resumes produced identical embedding vectors."
        )


# ---------------------------------------------------------------------------
# TestEmbedJob
# ---------------------------------------------------------------------------


class TestEmbedJob:
    """Tests for the embed_job() convenience function."""

    def test_returns_ndarray(self, mock_embedder: MockEmbedder, sample_jd_text: str):
        """embed_job() must return a numpy ndarray."""
        result = embed_job(sample_jd_text, mock_embedder)
        assert isinstance(result, np.ndarray)

    def test_correct_dimension(self, mock_embedder: MockEmbedder, sample_jd_text: str):
        """embed_job() must return shape (1024,)."""
        result = embed_job(sample_jd_text, mock_embedder)
        assert result.shape == (DIMENSION,)

    def test_empty_jd_text(self, mock_embedder: MockEmbedder):
        """embed_job() must handle empty JD text without raising an exception."""
        result = embed_job("", mock_embedder)
        assert isinstance(result, np.ndarray)
        assert result.shape == (DIMENSION,)

    def test_different_jds_produce_different_vectors(
        self, mock_embedder: MockEmbedder, sample_jd_text: str, sample_jd_text_data_science: str
    ):
        """Two different JD texts must produce different embedding vectors."""
        vec_se = embed_job(sample_jd_text, mock_embedder)
        vec_ds = embed_job(sample_jd_text_data_science, mock_embedder)
        assert not np.allclose(vec_se, vec_ds), (
            "Two different job descriptions produced identical embedding vectors."
        )


# ---------------------------------------------------------------------------
# TestBatchEmbedResumes
# ---------------------------------------------------------------------------


class TestBatchEmbedResumes:
    """Tests for the batch_embed_resumes() convenience function."""

    def test_returns_2d_array(self, mock_embedder: MockEmbedder):
        """batch_embed_resumes() must return a 2D ndarray of shape (N, dimension)."""
        resumes = [
            _make_resume("Software Engineer"),
            _make_resume("Data Scientist"),
            _make_resume("HR Manager"),
        ]
        result = batch_embed_resumes(resumes, mock_embedder)
        assert result.shape == (3, DIMENSION)

    def test_single_resume_batch(self, mock_embedder: MockEmbedder):
        """batch_embed_resumes() with one resume must return shape (1, dimension)."""
        resumes = [_make_resume("Frontend Developer")]
        result = batch_embed_resumes(resumes, mock_embedder)
        assert result.shape == (1, DIMENSION)

    def test_empty_list(self, mock_embedder: MockEmbedder):
        """batch_embed_resumes() with empty list must return shape (0, dimension)."""
        result = batch_embed_resumes([], mock_embedder)
        assert result.shape == (0, DIMENSION)

    def test_matches_individual_embeds(self, mock_embedder: MockEmbedder):
        """Each row in batch result must exactly match individual embed_resume() output."""
        resumes = [
            _make_resume("Backend Engineer", ["Python", "Go"]),
            _make_resume("Frontend Engineer", ["JavaScript", "React"]),
        ]
        batch = batch_embed_resumes(resumes, mock_embedder)
        for i, resume in enumerate(resumes):
            individual = embed_resume(resume, mock_embedder)
            assert np.allclose(batch[i], individual), (
                f"Batch row {i} does not match individual embed_resume() result."
            )

    def test_preserves_order(self, mock_embedder: MockEmbedder):
        """Output rows must correspond to input resumes in the same order."""
        labels = ["Backend Engineer", "Data Scientist", "DevOps Engineer", "ML Engineer"]
        resumes = [_make_resume(label) for label in labels]
        batch = batch_embed_resumes(resumes, mock_embedder)
        for i, resume in enumerate(resumes):
            individual = embed_resume(resume, mock_embedder)
            assert np.allclose(batch[i], individual), (
                f"Order mismatch at index {i}: label='{labels[i]}'"
            )
