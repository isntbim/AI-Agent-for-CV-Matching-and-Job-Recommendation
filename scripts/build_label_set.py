"""Human-label set for the CareerViet corpus: 20 held-out CVs (extractor test split, never trained on) x a pooled,
method-blind candidate list of corpus JDs, for two teammates to label independently (Good / Potential / No Fit).

Pooling (standard IR practice, so no method is favoured): per CV the union of the top-4 JDs of TF-IDF, of bge-m3
cosine and of the service ranking (Hybrid + TF-IDF over the retrieval pool), plus 2 random JDs; shuffled with a seed.
label_set_v1 was pooled with the bge-m3 version of the service ranking (before the switch to Qwen3 embeddings);
a rerun now pools with the Qwen3 service, so it would not reproduce v1 exactly.
Contact details and the candidate's name are redacted before the text leaves this machine.

  python -m scripts.build_label_set         # writes output/labeling/label_set_v1.json (+ sources, kept local)
"""
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from scripts import ats_service as svc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output/labeling"
MANIFEST = ROOT / "data/datasets/qwen_training_v4/manifest.json"
CVS = ["cv_002", "cv_005", "cv_022", "cv_028", "cv_030", "cv_033", "cv_034", "cv_040", "cv_041", "cv_021",
       "cv_074", "cv_087", "cv_048", "cv_050", "cv_055", "cv_058", "cv_059", "cv_062", "cv_064", "cv_073"]
PER_METHOD, RANDOM_EXTRA, SEED = 4, 2, 2026
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
PHONE = re.compile(r"(?<!\w)(\+?\d[\d ().–—-]{7,}\d)(?!\w)")
STREET = re.compile(r"\b\d{1,6}\s+(?:[A-Z][\w.]*\s+)+(?:Street|St|Road|Rd|Avenue|Ave|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b\.?")
ZIP_LINE = re.compile(r"^.*\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b.*$", re.M)  # "Saint Jacob IL 62281"
SPACED = re.compile(r"^(?:[A-Z] ){4,}[A-Z]$", re.M)  # "N O L A N D R U C K S": a spaced-out name
URL = re.compile(r"(https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+)", re.I)


def redact(text: str, name: str | None) -> str:
    text = URL.sub("[link]", EMAIL.sub("[email]", text))
    text = SPACED.sub("[Candidate]", ZIP_LINE.sub("[address]", STREET.sub("[address]", text)))
    text = PHONE.sub(lambda m: "[phone]" if sum(c.isdigit() for c in m.group(0)) >= 9 else m.group(0), text)
    for part in filter(None, [name] + (name or "").split()):
        if len(part) > 2:
            text = re.sub(rf"(?<!\w){re.escape(part)}(?!\w)", "[Candidate]", text, flags=re.I)
    return text


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer

    rows = {r["id"]: r for r in json.loads(MANIFEST.read_text(encoding="utf-8"))["tasks"]["cv_extraction"]}
    cv_text = {c: (ROOT / rows[c]["text"]).read_text(encoding="utf-8") for c in CVS}
    corpus = [d["id"] for d in json.loads((ROOT / "output/retrieval/corpus_v2/corpus.json").read_text(encoding="utf-8"))]
    paths = svc._jd_paths()
    jd_text = {j: (ROOT / paths[j]).read_text(encoding="utf-8") for j in corpus}

    # 1. extract the CVs once (cached; 3 parallel requests need OLLAMA_NUM_PARALLEL >= 3)
    from src.scoring.v4_client import QwenV4, V4Client
    llm, local = QwenV4(), threading.local()

    def extract(c):
        svc._cache(rows[c]["as_of"], V4Client(llm)).get(cv_text[c], "resume")
        print("extracted", c, flush=True)
    with ThreadPoolExecutor(3) as pool:
        list(pool.map(extract, CVS))

    # 2. pooled candidates
    ranker = svc.Ranker(jd_text)
    tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(list(jd_text.values()) + list(cv_text.values()))
    J = tfidf.transform([jd_text[j] for j in corpus])
    from sentence_transformers import SentenceTransformer
    bge_model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    bge_model.max_seq_length = 2048
    E = bge_model.encode([jd_text[j] for j in corpus], normalize_embeddings=True)
    rng, pairs, sources = random.Random(SEED), [], {}
    for c in CVS:
        nominated = {}
        tf = (tfidf.transform([cv_text[c]]) @ J.T).toarray()[0]
        bge = E @ bge_model.encode([cv_text[c]], normalize_embeddings=True)[0]
        hyb = [m["jd"] for m in svc.match(cv_text[c], rows[c]["as_of"], top=PER_METHOD, pool=20, ranker=ranker)["matches"]]
        for method, top in (("tfidf", [corpus[i] for i in np.argsort(-tf)[:PER_METHOD]]),
                            ("bge_m3", [corpus[i] for i in np.argsort(-bge)[:PER_METHOD]]), ("hybrid", hyb)):
            for j in top:
                nominated.setdefault(j, []).append(method)
        rest = [j for j in corpus if j not in nominated]
        for j in rng.sample(rest, RANDOM_EXTRA):
            nominated[j] = ["random"]
        order = list(nominated)
        rng.shuffle(order)
        pairs += [{"cv": c, "jd": j} for j in order]
        sources |= {f"{c}|{j}": m for j, m in nominated.items()}
        print(c, len(order), "candidates", flush=True)

    used = sorted({p["jd"] for p in pairs})
    names = {c: (json.loads((ROOT / rows[c]["label"]).read_text(encoding="utf-8"))["document"].get("basics") or {}).get("name")
             for c in CVS}
    data = {"version": "label_set_v1", "labels": ["No Fit", "Potential Fit", "Good Fit"],
            "cvs": [{"id": c, "text": redact(cv_text[c], names[c])} for c in CVS],
            "jds": [{"id": j, "text": jd_text[j]} for j in used], "pairs": pairs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "label_set_v1.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (OUT / "label_set_v1_sources.json").write_text(json.dumps(sources, indent=1), encoding="utf-8")  # local only: blind
    print(f"{len(pairs)} pairs, {len(used)} JDs -> {OUT / 'label_set_v1.json'}")


if __name__ == "__main__":
    main()
