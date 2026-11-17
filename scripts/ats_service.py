"""One CV in, ranked and explained JDs out: V4 extraction (validate-repair-retry) -> hybrid retrieval of a pool ->
the final ranking model reranks the pool -> explanation of each returned match by the O5 adapter (Ollama model
qwen-o5), written only from the V3.1 scorer's facts and checked by the faithfulness guard; a text that fails the guard
twice is replaced by a fixed template filled from the same facts.

Ranking = "Hybrid + TF-IDF (Qwen3)" (configs/hybrid_qwen_v1.json, written by scripts/match_ir_eval.py): logistic
weights on the V3.1 dimensions, Qwen3-Embedding-4B whole-text and title cosines and TF-IDF cosine, fitted on the
med2425 dev half (test NDCG@10 0.630 vs Qwen3 cosine 0.591, p<0.001; output/match_eval/lab/benchmark.md). The weights
come from English synthetic labels; on the Vietnamese CareerViet corpus they are unvalidated until human labels exist.

  python -m scripts.ats_service CV.txt --as-of 2026-11-02 [--top 5] [--pool 20] [--jd jd_001 ...] [--rerank]
  python -m scripts.ats_service --prepare-jds [--workers 3]   # extract every corpus JD once (offline; resumable)

Retrieval runs over the 361-JD English corpus (output/retrieval/corpus_v2 + index_v2) unless --jd names the JDs.
Extractions are cached under output/service/extractions, so a JD or CV is extracted once per model/prompt version.
Retrieval models run on the CPU; the ranking embeddings come from Ollama (qwen3-embedding:4b) next to the extractor.
On a 12 GB GPU both fit only with one extraction slot: serve with OLLAMA_NUM_PARALLEL=1 for the demo (qwen-v4 6.6 GB +
embeddings 3.1 GB); with 3 slots qwen-v4 takes 10 GB and Ollama swaps the two models (~5 s per request).
"""

import argparse
import functools
import hashlib
import json
import math
import os
import re
from datetime import date
from pathlib import Path

from src.scoring.extraction import ExtractionCache, write_json
from src.scoring.models import ScorerConfig
from src.scoring.normalization import contained_role
from src.scoring.rules import _latest_title
from src.scoring.runtime_v3 import score_pair_v3
from src.scoring.v4_client import V4Client

ROOT = Path(__file__).resolve().parent.parent
JD_MANIFEST = ROOT / "output/scorer_v3/teacher_labels/jd/manifest.json"
RELATIONS = ROOT / "configs/skill_relations_near_v1.json"
HYBRID = ROOT / "configs/hybrid_qwen_v1.json"
JD_EMB = ROOT / "output/service/jd_emb_qwen3_embedding.npz"
GRADES = ("No Fit", "Potential Fit", "Good Fit")
O5_MODEL = os.environ.get("O5_MODEL", "qwen-o5")  # scripts/qwen_build_ollama.py --name qwen-o5 --adapter <O5 v3>


def v31_config() -> ScorerConfig:
    """V3.1 as evaluated (output/match_eval/lab run v32_text_soft: test-half AUC 0.917, all 59,337 pairs). Skills
    policy text_and_soft: soft skills/languages/Vietnam law listed apart, CV-text mentions count (skills AUC 0.679 -> 0.765,
    Good pairs with 0% skills 39% -> 21%; ranking unchanged within noise). The near relations are
    auto-built and treated as approved after a 60-item hand audit (~90% precise); grade cut-offs 43 / 58 (dev-fitted
    43.4 / 57.8 for macro-F1)."""
    table = json.loads(RELATIONS.read_text(encoding="utf-8"))
    return ScorerConfig(version="ats-en-v3", review_policy="automatic", title_policy="contained",
                        experience_policy="lower_bound", lower_threshold=43, upper_threshold=58,
                        skills_policy="text_and_soft",
                        relations=[r | {"review_status": "approved"} for r in table["relations"]],
                        relation_version=table["version"])


class Ranker:
    """The final ranking model over one JD corpus. Embeds with Qwen3-Embedding through Ollama, caches corpus JD
    embeddings on disk and fits TF-IDF on the corpus JD texts."""

    def __init__(self, jd_texts: dict[str, str]):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        from src.embeddings.qwen import QwenEmbedder

        self.model = json.loads(HYBRID.read_text(encoding="utf-8"))
        self.emb = QwenEmbedder()
        cached = dict(np.load(JD_EMB)) if JD_EMB.exists() else {}
        missing = [j for j in jd_texts if j not in cached]
        if missing:
            cached |= dict(zip(missing, self.emb.encode([jd_texts[j] for j in missing])))
            JD_EMB.parent.mkdir(parents=True, exist_ok=True)
            np.savez(JD_EMB, **cached)
        self.jd_emb, self.jd_texts = cached, jd_texts
        self.tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(list(jd_texts.values()))

    def scores(self, cv_text: str, cv_title: str | None, rows: list[tuple[str, str | None, dict]],
               texts: dict[str, str] | None = None) -> list[float]:
        """rows: (jd id, jd title, V3.1 unrounded dimension scores) -> hybrid logits (higher = better fit)."""
        cv = self.emb.encode([cv_text], query=True)[0]
        titles = self.emb.encode([cv_title or "unknown"] + [t or "unknown" for _, t, _ in rows])
        extra = list(dict.fromkeys(jd for jd, _, _ in rows if jd not in self.jd_emb))
        extra_emb = dict(zip(extra, self.emb.encode([texts[j] for j in extra]))) if extra else {}
        tf_cv = self.tfidf.transform([cv_text])
        w, out, self.last_features = dict(zip(self.model["features"], self.model["coefficients"])), [], []
        for k, (jd, _, dims) in enumerate(rows):
            jd_vec = self.jd_emb[jd] if jd in self.jd_emb else extra_emb[jd]
            jd_txt = self.jd_texts[jd] if jd in self.jd_texts else texts[jd]
            feats = {d: dims[d] / 100 for d in ("skills", "title", "experience", "education", "industry")}
            feats |= {"text_cos": float(cv @ jd_vec), "title_cos": float(titles[0] @ titles[k + 1]),
                      "tfidf_cos": float((tf_cv @ self.tfidf.transform([jd_txt]).T).toarray()[0, 0])}
            out.append(self.model["intercept"] + sum(w[f] * feats[f] for f in self.model["features"]))
            self.last_features.append({f: w[f] * feats[f] for f in self.model["features"]} | {"_values": feats})
        return out

    @staticmethod
    def why(contrib: dict) -> str:
        """One sentence on what drives the fit: similarity (text, title, shared words) vs rule evidence."""
        sim = contrib["text_cos"] + contrib["title_cos"] + contrib["tfidf_cos"]
        rules = sum(v for k, v in contrib.items() if k not in ("text_cos", "title_cos", "tfidf_cos", "_values"))
        v = contrib["_values"]
        share = sim / (abs(sim) + abs(rules)) if sim or rules else 0
        lead = "mostly from how similar the CV and the job are" if share >= .66 else (
            "mostly from the rule evidence below" if share <= .33 else "from both similarity and the rule evidence")
        return (f"Fit comes {lead}: whole-text similarity {v['text_cos']:.2f}, job-title similarity "
                f"{v['title_cos']:.2f}, shared wording {v['tfidf_cos']:.2f}.")

    def grade(self, logit: float) -> int:
        lo, hi = self.model["grade_cutoffs"]
        return 0 if logit < lo else 1 if logit < hi else 2


VAGUE_WORDS = {"it", "leader", "team", "staff", "member", "employee", "intern", "fresher", "junior", "senior",
               "lead", "head", "manager", "officer", "executive", "associate", "assistant", "specialist", "worker"}


def _r0(x: float) -> int:
    return math.floor(x + 0.5)  # half-up, matching ScoreResultV3 display_score and browser Math.round


def vague_titles(resume) -> list[str]:
    """The CV's job titles when none of them names a known role ("IT", "Leader"): title and experience can't be compared."""
    titles = list(dict.fromkeys(w.position for w in resume.document.work if w.position))
    if not titles or any(contained_role(t)[1] for t in titles):
        return []
    return titles if all((w := set(re.findall(r"[a-z0-9+#]+", t.casefold()))) and (len(w) == 1 or w <= VAGUE_WORDS) for t in titles) else []


def explain(result, grade: int | None = None, fit: float | None = None, why: str | None = None,
            vague: list[str] | None = None) -> list[str]:
    """Plain-English reasons, one line per dimension, read straight off the score result. With `grade`/`fit` the
    headline gives the ranking model's verdict and keeps the rule score beside it; `why` says what drove that fit.
    A Good Fit whose listed requirements matched no CV skill gets a caution: the verdict then rests on similarity.
    `vague` (from vague_titles) explains why title/experience could not be scored."""
    d = result.dimensions
    head = (f"{GRADES[grade]}: fit {_r0(fit)}/100 (rule score {result.overall_score:.1f}/100)." if grade is not None
            else f"{GRADES[result.grade]}: {result.overall_score:.1f}/100.")
    lines = [head] + ([why] if why else [])
    if grade == 2 and d["skills"].supported and d["skills"].score < 10:
        lines.append("Caution: almost none of the job's listed requirements matched your CV's skills exactly. "
                     "Check the Missing list before applying.")
    if d["skills"].supported:
        lines.append(f"Skills {_r0(d['skills'].score)}/100. Matched: {', '.join(result.matched_skills[:8]) or 'none'}."
                     + (f" Missing: {', '.join(result.missing_skills[:8])}." if result.missing_skills else ""))
    breakdown = result.skills_breakdown or {}
    from_text = list(dict.fromkeys(r["candidate_skill"] for r in breakdown.get("requirements", []) if r.get("match_type") == "cv_text"))
    if from_text:
        lines.append(f"Found in your CV text (not in the extracted skills list): {', '.join(from_text[:8])}.")
    if breakdown.get("soft_skills"):
        lines.append(f"Also asked for, not scored (soft skills, languages, Vietnam-specific law): {', '.join(breakdown['soft_skills'][:8])}.")
    if d["title"].supported:
        lines.append(f"Job title {_r0(d['title'].score)}/100: " + ("same role as your CV title or work history." if d["title"].score >= 100
                     else "a related role in your CV title or work history." if d["title"].score > 0
                     else "no matching role in your CV title or work history."))
    if d["experience"].supported:
        s = d["experience"].score
        lines.append("Experience: no minimum required." if d["experience"].basis == "explicit_no_experience_requirement"
                     else "Experience: meets the required years." if s >= 100
                     else f"Experience: your relevant work covers about {_r0(s)}% of the required years.")
    if d["education"].supported:
        s = d["education"].score
        lines.append("Education: no degree required." if d["education"].basis == "explicit_no_education_requirement"
                     else "Education: meets the required degree." if s >= 100
                     else "Education: one level below the required degree." if s >= 50 else "Education: below the required degree.")
    if d["industry"].supported:
        lines.append(f"Industry: your background covers {_r0(d['industry'].score)}% of the job's domains.")
    for key in ("location", "salary"):
        if d[key].supported:
            lines.append(f"{key.capitalize()} {_r0(d[key].score)}/100.")
    unknown = [k for k, v in d.items() if not v.supported]
    missed = [k for k in ("title", "experience") if k in unknown]
    if vague and missed:
        lines.append(f"{' and '.join(missed).capitalize()} not scored: your CV's job titles are too general to compare "
                     f"({', '.join(vague[:4])}). Name the role, e.g. \"Backend Developer\" or \"Project Manager\".")
    if unknown:
        lines.append(f"Not scored (evidence missing or inconclusive, neutral value used): {', '.join(unknown)}.")
    return lines


@functools.lru_cache(maxsize=1)
def _retrieval_index():
    """The corpus index, its vector store and the bge-m3 query embedder: loaded once per process (~6 s on CPU)."""
    from types import SimpleNamespace

    from scripts.retrieval import load_index
    from src.embeddings.embedder import BGEEmbedder
    from src.retrieval.cache import EmbeddingCache
    from src.vector_db.pgvector_store import InMemoryVectorStore

    args = SimpleNamespace(corpus=ROOT / "output/retrieval/corpus_v2", index=ROOT / "output/retrieval/index_v2")
    docs, _, vectors, _, _, im = load_index(args)
    cache = ROOT / ".cache/retrieval"
    embedder = BGEEmbedder(revision=im["revision"], device="cpu", batch_size=1, cache_folder=str(cache / "huggingface"))
    ids = [d["id"] for d in docs]
    if dsn := os.environ.get("VECTOR_DSN"):  # the Docker stack's pgvector database
        from src.vector_db.pgvector_store import PGVectorStore

        # one table per corpus + embedding model, so a rebuilt corpus never reads stale vectors
        key = hashlib.sha256(json.dumps([ids, im["model"], im["revision"]]).encode()).hexdigest()[:12]
        store = PGVectorStore(dsn, table_name=f"jd_vectors_{key}", vector_size=im["dimension"])
        store.create_table()
        if store.count() != len(ids):  # first start on this corpus: load its vectors once
            store.upsert_batch(ids, list(vectors), docs)
    else:
        store = InMemoryVectorStore(dimension=im["dimension"])
        store.create_table("service")
        store.upsert_batch(ids, list(vectors), docs)
    return docs, store, embedder, EmbeddingCache(cache / "embeddings", im["model"], im["revision"])


def retrieve(cv_document: dict, top: int, rerank: bool) -> list[str]:
    from src.retrieval.pipeline import RetrievalPipeline
    from src.retrieval.reranker import BGEReranker, Candidate
    from src.retrieval.text import resume_text

    docs, store, embedder, emb_cache = _retrieval_index()
    text = resume_text(cv_document)
    vector = emb_cache.embed([text], embedder)[0]
    cache = ROOT / ".cache/retrieval"
    reranker = BGEReranker(device="cpu", cache_dir=str(cache / "huggingface")) if rerank else None
    result = RetrievalPipeline([Candidate(d["id"], d["text"], 0) for d in docs], store, None, reranker).query(text, vector)
    # "results" is capped at 10 for the retrieval benchmark; without the reranker the full fused list is the same order
    return [c.id for c in (result["results"] if rerank else result["hybrid"])[:top]]


@functools.lru_cache(maxsize=1)
def _llm():
    """The V4 extractor client (its HF tokenizer takes 2-9 s to load): one per process, shared by requests."""
    from src.scoring.v4_client import QwenV4
    return QwenV4()


def _cache(as_of: str | None = None, client: V4Client | None = None) -> ExtractionCache:
    return ExtractionCache(ROOT / "output/service/extractions", client or V4Client(), "ollama", "qwen2.5-7b-v4-qlora",
                           max_tokens=16384, ruleset="v3", as_of=as_of)


def _jd_paths() -> dict[str, str]:
    return {d["id"]: d["text_path"] for d in json.loads(JD_MANIFEST.read_text(encoding="utf-8"))["documents"]}


def corpus_ids() -> list[str]:
    return [d["id"] for d in json.loads((ROOT / "output/retrieval/corpus_v2/corpus.json").read_text(encoding="utf-8"))]


def corpus_ranker() -> "Ranker":
    """The ranking model over the whole 361-JD corpus (the service's search space)."""
    paths = _jd_paths()
    return Ranker({j: (ROOT / paths[j]).read_text(encoding="utf-8") for j in corpus_ids()})


def match(cv_text: str, as_of: str, top: int = 5, pool: int = 20, jd_ids: list[str] | None = None,
          rerank: bool = False, ranker: "Ranker | None" = None, on_step=None, on_token=None,
          jd_text: str | None = None) -> dict:
    """The service call: extract the CV, retrieve `pool` JDs, score each with V3.1, return the best `top` explained.
    When `jd_text` is given, retrieval is skipped and the CV is scored against that single pasted JD (`top` forced to 1).
    A JD whose extraction fails is reported in `skipped`, not silently dropped; a JD whose text repeats an earlier
    one in the pool is left out. Runs as the fixed LangGraph flow of service_graph().
    on_step(node) is called after each graph node; on_token(text) for each token the extractor generates."""
    if jd_text is not None:
        jd_ids, jd_texts, top = ["pasted"], {"pasted": jd_text}, 1
    else:
        jd_texts = None
    llm, answer = _llm() if on_token else None, None
    if llm:  # shortcut: one hook on the shared client, safe only while matches run one at a time (API LOCK)
        llm.on_token = on_token
    try:
        for update in service_graph().stream({"cv_text": cv_text, "as_of": as_of, "top": top, "pool": pool,
                                              "jd_ids": jd_ids, "jd_texts": jd_texts, "rerank": rerank, "ranker": ranker},
                                             stream_mode="updates"):
            for node, out in update.items():
                answer = (out or {}).get("answer", answer)
                if on_step:
                    on_step(node)
    finally:
        if llm:
            llm.on_token = None
    return answer


@functools.lru_cache(maxsize=1)
def service_graph():
    """extract (V4 + schema check + retry, itself a LangGraph in v4_client) -> retrieve (skipped when the caller
    names the JDs) -> score (extract each JD, V3.1) -> rank (final ranking model + explanations)."""
    from typing import Any, TypedDict

    from langgraph.graph import END, StateGraph

    class S(TypedDict, total=False):
        cv_text: str
        as_of: str
        top: int
        pool: int
        jd_ids: Any
        jd_texts: dict[str, str] | None
        rerank: bool
        ranker: Any
        client: Any
        resume: Any
        retrieved: bool
        scored: list
        skipped: dict
        answer: dict

    def extract(s: S) -> S:
        client = V4Client(_llm())
        try:
            return {"client": client, "resume": _cache(s["as_of"], client).get(s["cv_text"], "resume")}
        except Exception as exc:
            exc.source = "cv"
            raise

    def find(s: S) -> S:
        return {"retrieved": True,
                "jd_ids": retrieve(s["resume"].document.model_dump(mode="json"), s["pool"], s["rerank"])}

    def score(s: S) -> S:
        config, jd_texts, paths, skipped, scored, seen = v31_config(), s.get("jd_texts") or {}, None, {}, [], set()
        for rank, jd in enumerate(s["jd_ids"], 1):
            try:
                if jd in jd_texts:
                    text = jd_texts[jd]
                else:
                    if paths is None:
                        paths = _jd_paths()
                    text = (ROOT / paths[jd]).read_text(encoding="utf-8")
                if text in seen:  # the corpus holds a few postings twice (jd_150 = jd_342): identical text, identical score
                    continue
                seen.add(text)
                job = _cache(client=s["client"]).get(text, "job")
            except Exception as exc:  # one bad JD must not sink a corpus run; a pasted JD has no fallback and must fail
                if jd == "pasted":
                    exc.source = "jd"
                    raise
                skipped[jd] = f"{type(exc).__name__}: {str(exc)[:200]}"
                continue
            scored.append((rank, jd, job, score_pair_v3(s["resume"], job, as_of_date=date.fromisoformat(s["as_of"]),
                                                        config=config, cv_text=s["cv_text"])))
        return {"scored": scored, "skipped": skipped}

    def rank(s: S) -> S:
        rows, scored, resume, jd_texts = [], s["scored"], s["resume"], s.get("jd_texts") or {}
        if scored:
            ranker = s.get("ranker") or corpus_ranker()  # TF-IDF weights come from the whole corpus, never just the pool
            cv_title = _latest_title(resume, date.fromisoformat(s["as_of"])) or next(
                (w.position for w in resume.document.work if w.position), None)
            logits = ranker.scores(s["cv_text"], cv_title, [(jd, job.document.title, r.unrounded_scores.model_dump())
                                                            for _, jd, job, r in scored], texts=jd_texts or None)
            for (rnk, jd, job, result), logit, contrib in zip(scored, logits, ranker.last_features):
                grade, fit = ranker.grade(logit), 100 / (1 + math.exp(-logit))
                row = {"jd": jd, "title": job.document.title, "retrieval_rank": rnk, "fit": round(fit, 1),
                       "_logit": logit, "grade": GRADES[grade], "rule_score": result.overall_score,
                       "scores": result.scores.model_dump(), "matched_skills": result.matched_skills,
                       "missing_skills": result.missing_skills,
                       "explanation": explain(result, grade, round(fit, 1), Ranker.why(contrib), vague_titles(resume)),
                       "evidence": {k: v.evidence[:3] for k, v in result.dimensions.items()
                                    if v.supported and v.evidence}, "_o5": (job, result, grade, fit)}
                if jd in jd_texts:
                    row["job_text"] = jd_texts[jd]
                rows.append(row)
        rows.sort(key=lambda r: (-r.pop("_logit"), r["retrieval_rank"]))
        if rows:  # O5 writes the explanations of the returned matches only, in the CV's language
            from scripts.o5_build_pairs import facts
            from scripts.o5_faithfulness import is_vietnamese

            llm, vi, as_of = _llm(), is_vietnamese(s["cv_text"]), date.fromisoformat(s["as_of"])
            for row in rows[:s["top"]]:
                job, result, grade, fit = row["_o5"]
                row["summary"], row["summary_source"] = o5_explain(facts(resume, job, result, grade, fit, as_of), vi, llm)
        for row in rows:
            row.pop("_o5")
        return {"answer": {"as_of": s["as_of"], "retrieved": bool(s.get("retrieved")), "pool": len(s["jd_ids"]),
                           "matches": rows[:s["top"]], "skipped": s["skipped"], "ranking_model": "hybrid_qwen_v1"}}

    g = StateGraph(S)
    for name, fn in (("extract", extract), ("retrieve", find), ("score", score), ("rank", rank)):
        g.add_node(name, fn)
    g.set_entry_point("extract")
    g.add_conditional_edges("extract", lambda s: "score" if s.get("jd_ids") else "retrieve",
                            {"retrieve": "retrieve", "score": "score"})
    g.add_edge("retrieve", "score")
    g.add_edge("score", "rank")
    g.add_edge("rank", END)
    return g.compile()


def _o5_generate(facts: dict, vi: bool, llm, penalty: float = 1.0) -> str:
    """One O5 generation in the format the adapter was trained on: its system prompt + the facts rendered as plain
    statements, Qwen chat template, greedy."""
    import httpx

    from scripts.o5_export_sft import SYSTEM, SYSTEM_VI, render
    from src.scoring.v4_client import ollama_headers

    prompt = llm.tok.apply_chat_template([{"role": "system", "content": SYSTEM_VI if vi else SYSTEM},
                                          {"role": "user", "content": render(facts)}], tokenize=False, add_generation_prompt=True)
    r = httpx.post(f"{llm.url}/api/generate", timeout=180, headers=ollama_headers(), json={
        "model": O5_MODEL, "prompt": prompt, "raw": True, "stream": False,
        "options": {"temperature": 0, "seed": 0, "num_ctx": 4096, "num_predict": 400, "repeat_penalty": penalty}})
    r.raise_for_status()
    return r.json()["response"].strip()


def o5_explain(facts: dict, vi: bool, llm) -> tuple[str, str]:
    """(text, source) for one match. scripts.o5_faithfulness.violations is the guard: a wrong verdict, number or skill,
    a claim about an unscored dimension or a contradicted fact rejects the text. A rejected text is regenerated once with
    repetition penalty 1.1 (greedy would repeat it), then the template is used, as it is when the explainer cannot be
    reached: an explanation never blocks the match."""
    from scripts.o5_export_sft import fallback
    from scripts.o5_faithfulness import violations

    for penalty in (1.0, 1.1):
        try:
            text = _o5_generate(facts, vi, llm, penalty)
        except Exception:  # explainer down or slow: the template still states the facts
            break
        if text and not violations(text, facts):
            return text, "o5"
    return fallback(facts, vi), "template"


def prepare_jds(workers: int = 1) -> None:
    """Extract every corpus JD into the cache so a live request only waits for its CV. workers > 1 needs
    OLLAMA_NUM_PARALLEL >= workers on the server (as in scripts/match_eval.py)."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from types import SimpleNamespace

    from scripts.retrieval import load_index
    from src.scoring.v4_client import QwenV4

    docs = load_index(SimpleNamespace(corpus=ROOT / "output/retrieval/corpus_v2", index=ROOT / "output/retrieval/index_v2"))[0]
    llm, local, paths, failed = QwenV4(), threading.local(), _jd_paths(), []  # tokenizer loaded once, in this thread

    def one(jd: str) -> None:
        if not hasattr(local, "cache"):  # V4Client keeps per-call metadata on itself: one per thread
            local.cache = _cache(client=V4Client(llm))
        local.cache.get((ROOT / paths[jd]).read_text(encoding="utf-8"), "job")

    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, d["id"]): d["id"] for d in docs}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
                status = "ok"
            except Exception as exc:
                failed.append(futures[future])
                status = f"FAILED {type(exc).__name__}"
            print(f"{i}/{len(docs)} {futures[future]} {status}", flush=True)
    print(f"done: {len(docs) - len(failed)} extracted, {len(failed)} failed: {sorted(failed)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cv", type=Path, nargs="?", help="CV as plain text")
    p.add_argument("--as-of", help="YYYY-MM-DD: the date employment durations are measured to")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--pool", type=int, default=20, help="how many retrieved JDs the scorer reranks")
    p.add_argument("--jd", nargs="*", help="score these corpus JD ids instead of retrieving")
    p.add_argument("--rerank", action="store_true", help="also apply the BGE cross-encoder inside retrieval")
    p.add_argument("--prepare-jds", action="store_true")
    p.add_argument("--workers", type=int, default=1, help="--prepare-jds: parallel requests (OLLAMA_NUM_PARALLEL)")
    p.add_argument("--out", type=Path, default=ROOT / "output/service/last_match.json")
    args = p.parse_args()
    if args.prepare_jds:
        return prepare_jds(args.workers)
    if not (args.cv and args.as_of):
        p.error("cv and --as-of are required")
    out = match(args.cv.read_text(encoding="utf-8"), args.as_of, args.top, args.pool, args.jd, args.rerank)
    write_json(args.out, {"cv": str(args.cv), **out})
    for r in out["matches"]:
        print(f"\n{r['fit']:5.1f}  {r['jd']}  {r['title']}  (retrieval rank {r['retrieval_rank']})")
        print("\n".join(f"   - {line}" for line in r["explanation"]))
    if out["skipped"]:
        print(f"\nskipped (extraction failed): {', '.join(out['skipped'])}")
    print(f"details: {args.out}")


if __name__ == "__main__":
    main()
