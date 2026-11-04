"""Baseline for the paper: bge-m3 cosine between raw CV text and raw JD text, on the same scored pairs and CV split.
Also writes the title embeddings the hybrid uses: the CV's latest job title (else its first position) and the JD title,
from the extracted documents (lab/titles.json, lab/title_emb_bge_m3.npz). Run after scorer_lab --build."""
import json
import pickle
from pathlib import Path


import numpy as np
from sentence_transformers import SentenceTransformer

from scripts import match_eval as m
from scripts.scorer_lab import AS_OF, PKL, split_of
from src.scoring.rules import _latest_title

OUT = Path("output/match_eval/lab")
pairs = m.load_pairs()
texts = {}
for p in pairs:
    texts.setdefault(m.doc_id("cv", p["cv"]), p["cv"])
    texts.setdefault(m.doc_id("jd", p["jd"]), p["jd"])
ids = sorted(texts)
emb_path = OUT / "doc_emb_bge_m3.npz"
if emb_path.exists():
    emb = dict(np.load(emb_path))
else:
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    model.max_seq_length = 2048
    vecs = model.encode([texts[i] for i in ids], batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    emb = dict(zip(ids, vecs))
    np.savez(emb_path, **emb)
docs = pickle.loads(PKL.read_bytes())["docs"]
with (OUT / "text_cosine.jsonl").open("w", encoding="utf-8") as f:
    for p in pairs:
        cv, jd = m.doc_id("cv", p["cv"]), m.doc_id("jd", p["jd"])
        if docs[cv] is None or docs[jd] is None:  # same 48,010 pairs as the scorer
            continue
        f.write(json.dumps({"cv": cv, "jd": jd, "label": p["label"], "split": split_of(cv),
                            "cos": float(emb[cv] @ emb[jd])}) + "\n")
titles = {k: (_latest_title(d, AS_OF) or next((w.position for w in d.document.work if w.position), None)
              if k.startswith("cv") else d.document.title) or "" for k, d in docs.items() if d is not None}
model = SentenceTransformer("BAAI/bge-m3", device="cpu")  # short strings; the GPU may be busy with the extractor
ids = sorted(titles)
np.savez(OUT / "title_emb_bge_m3.npz", **dict(zip(ids, model.encode([titles[i] or "unknown" for i in ids],
                                                                      normalize_embeddings=True, batch_size=64))))
(OUT / "titles.json").write_text(json.dumps(titles, indent=0), encoding="utf-8")
