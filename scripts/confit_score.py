"""Score all 59,337 CV-JD pairs with a trained ConFit v2 checkpoint.

Writes JSONL rows `{"cv": cv_id, "jd": jd_id, "cos": cosine}` matching the
schema of `output/match_eval/lab/qwen_embedding_cosine.jsonl`.
Defaults to CPU so background GPU workloads stay untouched unless `--device cuda`
is passed explicitly.

Usage (CPU smoke check):
    D:\\confit_v2\\.venv\\Scripts\\python.exe scripts\\confit_score.py ^
        --checkpoint D:\\confit_v2\\checkpoints\\smoke\\last.ckpt ^
        --limit 100 --device cpu ^
        --out output\\match_eval\\lab\\confit_v2_cosine_smoke100.jsonl

Usage (full run after GPU training):
    D:\\confit_v2\\.venv\\Scripts\\python.exe scripts\\confit_score.py ^
        --checkpoint D:\\confit_v2\\checkpoints\\e5base_full\\best.ckpt ^
        --device cuda
"""

from __future__ import annotations

import argparse
import glob
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

if "--device" in sys.argv:
    _dev_idx = sys.argv.index("--device")
    _req_dev = sys.argv[_dev_idx + 1] if _dev_idx + 1 < len(sys.argv) else "cpu"
else:
    _req_dev = "cpu"
if _req_dev != "cuda":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def load_pairs_raw() -> list[dict]:
    """Load the clean 59,337 CV-JD pairs matching scripts/match_eval.py."""
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    try:
        from scripts import match_eval as m

        return m.load_pairs()
    except ImportError:
        import pyarrow.parquet as pq

        labels = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
        paths = sorted(glob.glob(str(ROOT / "data/raw/external/med2425_resume_job_fit_merged_v1/data/train-*.parquet")))
        rows = [r for p in paths for r in pq.read_table(p, columns=["resume", "jd", "label"]).to_pylist()]
        manifest = json.loads((ROOT / "output/scorer_v3/teacher_labels/manifest_external.json").read_text(encoding="utf-8"))
        trained = {(ROOT / d["text_path"]).read_text(encoding="utf-8") for d in manifest["documents"] if d["source"] == "med2425"}
        pairs = [
            {"cv": r["resume"].strip(), "jd": r["jd"].strip(), "label": labels[r["label"]]}
            for r in rows
            if r["resume"].strip() not in trained
        ]
        assert len(trained) == 157 and len(pairs) == 59337, (len(trained), len(pairs))
        return pairs


def doc_id(kind: str, text: str) -> str:
    return f"{kind}_{sha256(text.encode()).hexdigest()[:12]}"


def resolve_checkpoint(path: Path) -> tuple[Path, Path]:
    """Return (checkpoint_file, model_dir) from either a .ckpt path or run directory."""
    if path.is_file():
        return path, path.parent
    ckpts = [p for p in path.glob("*.ckpt") if p.name != "last.ckpt"]
    if len(ckpts) != 1:  # training keeps exactly one best-val/loss checkpoint (--save_top_k 1); never guess
        raise FileNotFoundError(f"expected exactly one best checkpoint under {path}, found {[p.name for p in ckpts]}; "
                                "pass the .ckpt path explicitly")
    return ckpts[0], path


def load_confit_model(ckpt_path: Path, model_dir: Path, confit_repo: Path, device: str):
    confit_root = confit_repo.resolve()
    saved_path = sys.path[:]
    # Exclude repo ROOT / CWD so ConFit-v2's namespace `src` beats ROOT's regular `src/__init__.py`.
    sys.path = [str(confit_root)] + [
        p for p in saved_path if p not in ("", ".") and Path(p).resolve() not in (ROOT, confit_root)
    ]
    try:
        from src.config.dataset import DATASET_CONFIG
        from src.model.confit import ConFitModel, ConFitModelArguments
    finally:
        sys.path = saved_path
        if str(confit_root) not in sys.path:
            sys.path.insert(0, str(confit_root))

    model_args_dict = json.loads((model_dir / "model_args.json").read_text(encoding="utf-8"))
    data_args_path = model_dir / "data_args.json"
    data_args_dict = json.loads(data_args_path.read_text(encoding="utf-8")) if data_args_path.exists() else {}

    m_args = ConFitModelArguments(**model_args_dict)
    assert m_args.encode_all, "confit_score.py requires models trained with encode_all=True"
    model = ConFitModel(m_args)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model = model.float().to(device)
    model.eval()

    dataset_type = data_args_dict.get("dataset_type", "recruiting_data_v2")
    cfg = DATASET_CONFIG[dataset_type]
    query_prefix = data_args_dict.get("query_prefix", "")
    tokenizer = AutoTokenizer.from_pretrained(m_args.pretrained_encoder)
    return model, m_args, cfg, query_prefix, tokenizer


def encode_texts(
    model,
    m_args,
    tokenizer,
    texts_by_id: dict[str, str],
    data_type: str,
    taxon_token: str,
    key_name: str,
    max_len: int,
    query_prefix: str,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Encode unique CV or JD texts matching RJPairSimplifiedDataset/RJPairNewContrastiveDataset."""
    ids = sorted(texts_by_id)
    out: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start : start + batch_size]
            formatted = []
            for did in batch_ids:
                content = f"{taxon_token}: section {key_name}:\n{texts_by_id[did]}".strip()
                if query_prefix:
                    content = f"{query_prefix}: {content}"
                formatted.append(content)
            enc = tokenizer(
                formatted,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            batched = {"all": {"encoded_key_values": enc}}
            emb_out = model.forward(batched, data_type=data_type)
            vec = (
                emb_out.coarse_embedding
                if m_args.finegrained_loss == "noop"
                else emb_out.finegrained_embedding
            )
            vec = F.normalize(vec.float().to(device), p=2, dim=1).cpu().numpy()
            for did, v in zip(batch_ids, vec):
                out[did] = v
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True, help="Path to .ckpt or checkpoint directory")
    p.add_argument("--confit-repo", type=Path, default=Path(r"D:\confit_v2\ConFit-v2"))
    p.add_argument("--out", type=Path, default=ROOT / "output/match_eval/lab/confit_v2_cosine.jsonl")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--limit", type=int, default=0, help="Score only the first N pairs (0 = all 59,337)")
    p.add_argument("--texts", type=Path, default=None,
                   help='JSON {"cv": {id: text}, "jd": {id: text}}: embed these instead; --out gets the vectors')
    args = p.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        torch.backends.mkldnn.enabled = False

    ckpt_path, model_dir = resolve_checkpoint(args.checkpoint)
    model, m_args, cfg, query_prefix, tokenizer = load_confit_model(
        ckpt_path, model_dir, args.confit_repo, args.device
    )
    r_key, j_key = cfg["resume_key_names"][0], cfg["job_key_names"][0]
    r_len = sum(cfg["max_seq_len_per_feature"][k] for k in cfg["resume_key_names"])
    j_len = sum(cfg["max_seq_len_per_feature"][k] for k in cfg["job_key_names"])

    if args.texts:  # other evaluation sets: {"cv": {id: text}, "jd": {id: text}} -> {"cv": {id: vec}, "jd": {id: vec}}
        texts = json.loads(args.texts.read_text(encoding="utf-8"))
        out = {"checkpoint": str(ckpt_path),
               "cv": {k: v.tolist() for k, v in encode_texts(model, m_args, tokenizer, texts["cv"], "resume",
                      cfg["resume_taxon_token"], r_key, r_len, query_prefix, args.batch_size, args.device).items()},
               "jd": {k: v.tolist() for k, v in encode_texts(model, m_args, tokenizer, texts["jd"], "job",
                      cfg["job_taxon_token"], j_key, j_len, query_prefix, args.batch_size, args.device).items()}}
        args.out.write_text(json.dumps(out), encoding="utf-8")
        print(f"checkpoint={ckpt_path} cvs={len(out['cv'])} jds={len(out['jd'])} -> {args.out}")
        return

    pairs = load_pairs_raw()
    if args.limit > 0:
        pairs = pairs[: args.limit]

    cv_texts: dict[str, str] = {}
    jd_texts: dict[str, str] = {}
    pair_ids: list[tuple[str, str]] = []
    for row in pairs:
        cid = doc_id("cv", row["cv"])
        jid = doc_id("jd", row["jd"])
        cv_texts.setdefault(cid, row["cv"])
        jd_texts.setdefault(jid, row["jd"])
        pair_ids.append((cid, jid))

    cv_emb = encode_texts(
        model, m_args, tokenizer, cv_texts, "resume",
        cfg["resume_taxon_token"], r_key, r_len, query_prefix, args.batch_size, args.device,
    )
    jd_emb = encode_texts(
        model, m_args, tokenizer, jd_texts, "job",
        cfg["job_taxon_token"], j_key, j_len, query_prefix, args.batch_size, args.device,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for cid, jid in pair_ids:
            cos = float(np.dot(cv_emb[cid], jd_emb[jid]))
            f.write(json.dumps({"cv": cid, "jd": jid, "cos": cos}) + "\n")

    print(
        f"checkpoint={ckpt_path} device={args.device} pairs={len(pair_ids)} "
        f"unique_cvs={len(cv_emb)} unique_jds={len(jd_emb)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
