import math
from copy import deepcopy

import numpy as np
import pytest

from src.embeddings.embedder import MockEmbedder
from src.retrieval.reranker import Candidate, MockReranker, rerank
from src.retrieval.lexical import BM25, rrf, tokenize
from src.retrieval.metrics import ndcg, reciprocal_rank, known_positive_recall, judged_coverage
from src.retrieval.text import job_text, resume_text
from src.retrieval.corpus import clean_records, digest
from src.retrieval.cache import EmbeddingCache
from src.retrieval.pipeline import RetrievalPipeline
from src.vector_db.pgvector_store import InMemoryVectorStore


def candidates():
    return [Candidate("b", "Python SQL", .9), Candidate("a", "Python", .8), Candidate("c", "Java", .7)]


def test_reranker_order_metadata_and_stable_ties():
    docs = candidates()
    ranked = rerank("Python", docs, MockReranker())
    assert [c.id for c in ranked] == ["b", "a", "c"]
    assert [c.original_rank for c in ranked] == [1, 2, 3]
    assert [c.final_rank for c in ranked] == [1, 2, 3]
    assert ranked[0].retrieval_score == .9
    assert ranked[0].reranker_score == 1
    assert docs[0].reranker_score is None
    ranked = rerank("Java", docs, MockReranker())
    assert [c.id for c in ranked] == ["c", "b", "a"]
    assert ranked[0].original_rank == 3


@pytest.mark.parametrize("scores", [[1], [1, 2, 3, 4], [1, float("nan"), 0], [1, float("inf"), 0]])
def test_bad_reranker_scores(scores):
    class Client:
        def score_pairs(self, pairs):
            return scores
    with pytest.raises(ValueError):
        rerank("q", candidates(), Client())


def test_duplicates_and_empty():
    with pytest.raises(ValueError):
        rerank("q", [candidates()[0], candidates()[0]], MockReranker())
    class Never:
        def score_pairs(self, pairs):
            raise AssertionError("Empty list must not invoke model")
    assert rerank("q", [], Never()) == []


def test_technical_tokenization():
    assert tokenize("C++ C# .NET ASP.NET Node.js CI/CD c C") == ["c++", "c#", ".net", "asp.net", "node.js", "ci", "cd", "c", "c"]


def test_bm25_hand_calculation_and_ties():
    docs = [Candidate("a", "python python", 0), Candidate("b", "java", 0)]
    scores = BM25(docs).scores("python")
    idf = math.log(1 + 1.5 / 1.5)
    expected = idf * 2 * 2.5 / (2 + 1.5 * (.25 + .75 * 2 / 1.5))
    assert scores["a"] == pytest.approx(expected)
    assert scores["b"] == 0
    assert [c.id for c in BM25(docs).search("absent")] == ["a", "b"]
    assert BM25([]).search("q") == []
    with pytest.raises(ValueError):
        BM25(docs, k1=0)


def test_rrf_hand_calculation_and_order():
    a, b, c = candidates()
    result = rrf([[a, b], [b, c]])
    assert [r.id for r in result] == ["a", "b", "c"]  # candidate b has id a and occurs twice
    assert result[0].retrieval_score == pytest.approx(1 / 62 + 1 / 61)
    assert result[1].retrieval_score == pytest.approx(1 / 61)
    assert result[0].metadata["retrieval_stages"]["1"]["rank"] == 1
    with pytest.raises(ValueError):
        rrf([[a, a]])


def test_ranking_metrics_hand_calculated():
    grades = {"a": 2, "b": 1, "c": 0}
    ids = ["c", "b", "a"]
    assert reciprocal_rank(ids, grades) == .5
    assert ndcg(ids, grades, 3) == pytest.approx((1 / math.log2(3) + 3 / 2) / (3 + 1 / math.log2(3)))
    assert ndcg(["a", "b", "c"], grades) == 1
    assert reciprocal_rank(["c"], grades) == 0
    assert known_positive_recall(["unjudged", "a"], grades) == .5
    assert judged_coverage(["unjudged", "a"], grades, 10) == .5
    assert known_positive_recall(["c"], {"c": 0}) is None
    with pytest.raises(ValueError):
        ndcg(["unjudged"], grades)
    with pytest.raises(ValueError):
        ndcg(["a", "a"], grades)


def test_serializers_exclude_metadata_and_defaults():
    record = {"title": "Developer", "category": "secret_domain", "overall_score": 97, "grade": 2,
              "company": {}, "location": {}, "seniority": {"min_years": 0}, "skills": {}, "description": {}}
    assert job_text(record) == "Job title: Developer\nMinimum experience (years): 0"
    assert resume_text({"domain": "secret_domain", "quality_tier": "gold", "cv_id": "private"}) == ""
    assert "secret" not in job_text(record)


def test_cleaning_keeps_supported_skills_and_is_idempotent():
    records = [{"id": "jd", "title_normalized": "developer", "skills": {"required": [
        {"name": "Developer"}, {"name": "Google Tag Manager"}, {"name": "Python"}]}}]
    original = deepcopy(records)
    review = [{"key": digest(["jd", "required", name]), "jd_id": "jd", "bucket": "required", "skill": name,
               "decision": decision, "reason": "Source-reviewed role/tool distinction",
               "review_status": "assistant_reviewed_awaiting_human_approval"}
              for name, decision in [("Developer", "remove_role_tag"), ("Google Tag Manager", "retain_skill")]]
    cleaned, ledger = clean_records(records, review)
    assert [s["name"] for s in cleaned[0]["skills"]["required"]] == ["Google Tag Manager", "Python"]
    assert records == original
    assert clean_records(cleaned, review)[0] == cleaned
    assert len(ledger) == 2
    with pytest.raises(ValueError):
        clean_records(records, [])


def test_cache_invalidates_on_revision_and_text(tmp_path):
    client = MockEmbedder(dimension=8)
    cache = EmbeddingCache(tmp_path, "model", "revision1")
    first = cache.embed(["python", "java", "python"], client)
    assert first.shape == (3, 8) and cache.misses == 2
    assert np.array_equal(first[0], first[2])
    assert np.allclose(cache.embed(["python"], client)[0], first[0])
    assert cache.hits == 1
    assert cache.key("python") != EmbeddingCache(tmp_path, "model", "revision2").key("python")


def test_retrieval_fusion_reranking_integration():
    docs = candidates()
    client = MockEmbedder(dimension=8)
    store = InMemoryVectorStore(dimension=8)
    store.create_table("test")
    store.upsert_batch([c.id for c in docs], list(client.embed_batch([c.text for c in docs])), [{} for c in docs])
    flow = RetrievalPipeline(docs, store, client, MockReranker())
    result = flow.query("Java")
    assert result["results"][0].id == "c"
    assert len({c.id for c in result["hybrid"]}) == 3
    assert result["results"][0].reranker_score == 1
    assert result["truncations"] == []
