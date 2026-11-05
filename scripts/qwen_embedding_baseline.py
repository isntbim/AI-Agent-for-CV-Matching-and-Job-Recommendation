"""Qwen Embedding baseline (proposal "Baseline Models") and the final ranking model's embeddings: qwen3-embedding:4b
(Ollama, Q4_K_M, 2,048-token context like bge-m3) cosine between raw CV text and raw JD text, on the same scored pairs
as text_baseline.py. The CV is the query and gets the model's retrieval instruction; JDs and titles go in plain.
Writes lab/doc_emb_qwen3_embedding.npz, lab/qwen_embedding_cosine.jsonl and lab/title_emb_qwen3_embedding.npz
(titles from lab/titles.json, written by text_baseline.py), which match_ir_eval.py picks up.
Needs `ollama serve`; stop qwen-v4 first if the GPU is short.

  python -m scripts.qwen_embedding_baseline
"""
import json
import pickle

import numpy as np

from scripts import match_eval as m
from scripts.scorer_lab import PKL
from src.embeddings.qwen import QwenEmbedder

OUT = m.OUT / "lab"


def main() -> None:
    texts = {}
    for p in m.load_pairs():
        texts.setdefault(m.doc_id("cv", p["cv"]), p["cv"])
        texts.setdefault(m.doc_id("jd", p["jd"]), p["jd"])
    model, emb_path = QwenEmbedder(), OUT / "doc_emb_qwen3_embedding.npz"
    if emb_path.exists():
        emb = dict(np.load(emb_path))
    else:
        cvs, jds = sorted(k for k in texts if k.startswith("cv")), sorted(k for k in texts if k.startswith("jd"))
        emb = dict(zip(cvs, model.encode([texts[k] for k in cvs], query=True)))
        emb |= dict(zip(jds, model.encode([texts[k] for k in jds])))
        np.savez(emb_path, **emb)
    docs = pickle.loads(PKL.read_bytes())["docs"]
    with (OUT / "qwen_embedding_cosine.jsonl").open("w", encoding="utf-8") as f:
        for p in m.load_pairs():
            cv, jd = m.doc_id("cv", p["cv"]), m.doc_id("jd", p["jd"])
            if docs[cv] is not None and docs[jd] is not None:  # same scored pairs as the other baselines
                f.write(json.dumps({"cv": cv, "jd": jd, "cos": float(emb[cv] @ emb[jd])}) + "\n")
    titles = json.loads((OUT / "titles.json").read_text(encoding="utf-8"))
    ids = sorted(titles)
    np.savez(OUT / "title_emb_qwen3_embedding.npz", **dict(zip(ids, model.encode([titles[i] or "unknown" for i in ids]))))


if __name__ == "__main__":
    main()
