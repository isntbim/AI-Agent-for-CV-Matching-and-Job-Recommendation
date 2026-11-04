"""Baseline for the paper: bge-m3 cosine between raw CV text and raw JD text, on the same scored pairs and CV split."""
import json
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts import match_eval as m
from scripts.scorer_lab import PKL, split_of

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
