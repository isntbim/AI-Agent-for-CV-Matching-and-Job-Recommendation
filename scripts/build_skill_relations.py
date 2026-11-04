"""Build an automatic BGE-M3 + token-containment skill-relation table for the scorer lab."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import pickle
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from src.parser.skill_v2 import SkillRelation, SkillRelationTable
from src.scoring.skills_v3 import _prepare_candidates, _prepare_requirements

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PKL = ROOT / "output/match_eval/lab_docs.pkl"
DEFAULT_EMB = ROOT / "output/match_eval/lab/skill_emb.npz"
DEFAULT_OUT = ROOT / "output/match_eval/lab/relations.json"
MODEL_NAME = "BAAI/bge-m3"

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "s", "such", "that",
    "the", "their", "then", "there", "these", "they", "this", "to",
    "was", "will", "with",
})


def content_tokens(name: str) -> set[str]:
    """Extract normalized non-stopword content tokens from a skill name."""
    tokens = re.findall(r"[a-z0-9+#]+", name.casefold())
    return {t for t in tokens if t not in STOPWORDS}


def token_containment(cv_name: str, jd_name: str) -> bool:
    """Return True when the two names differ and all content tokens of the shorter occur in the longer."""
    if cv_name.casefold() == jd_name.casefold():
        return False
    cv_tok = content_tokens(cv_name)
    jd_tok = content_tokens(jd_name)
    if not cv_tok or not jd_tok:
        return False
    shorter, longer = (cv_tok, jd_tok) if len(cv_tok) <= len(jd_tok) else (jd_tok, cv_tok)
    return shorter <= longer


def collect_skill_names(pkl_path: Path = DEFAULT_PKL) -> tuple[list[str], list[str]]:
    """Collect sorted distinct canonical CV and JD skill names exactly as skills_v3 sees them."""
    data = pickle.loads(pkl_path.read_bytes())
    docs = data["docs"]
    cv_names: set[str] = set()
    jd_names: set[str] = set()
    for doc_id, doc in docs.items():
        if doc is None:
            continue
        if doc_id.startswith("cv_"):
            candidates, _, _ = _prepare_candidates(doc)
            cv_names.update(c.name for c in candidates if c.name)
        elif doc_id.startswith("jd_"):
            requirements, _, _ = _prepare_requirements(doc)
            for req in requirements:
                jd_names.update(opt for opt in req.options if opt)
    return sorted(cv_names), sorted(jd_names)


def load_or_compute_embeddings(
    cv_names: list[str],
    jd_names: list[str],
    cache_path: Path = DEFAULT_EMB,
    model_name: str = MODEL_NAME,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached L2-normalized embeddings from skill_emb.npz or compute with BAAI/bge-m3."""
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if (
                "cv_names" in cached
                and "jd_names" in cached
                and cached["cv_names"].tolist() == cv_names
                and cached["jd_names"].tolist() == jd_names
            ):
                return cached["cv_emb"].astype(np.float32), cached["jd_emb"].astype(np.float32)

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    all_names = sorted(set(cv_names) | set(jd_names))
    all_emb = model.encode(
        all_names,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}
    cv_emb = all_emb[[name_to_idx[name] for name in cv_names]]
    jd_emb = all_emb[[name_to_idx[name] for name in jd_names]]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cv_names=np.array(cv_names),
        jd_names=np.array(jd_names),
        cv_emb=cv_emb,
        jd_emb=jd_emb,
    )
    return cv_emb, jd_emb


def build_relations(
    cv_names: Sequence[str],
    jd_names: Sequence[str],
    sim_matrix: np.ndarray,
    *,
    near: float = 0.80,
    transfer: float = 0.70,
    top_k: int = 3,
) -> tuple[SkillRelationTable, list[tuple[SkillRelation, float]]]:
    """Select top-k near and transfer relations per JD skill and validate SkillRelationTable."""
    if sim_matrix.shape != (len(cv_names), len(jd_names)):
        raise ValueError(f"Expected shape {(len(cv_names), len(jd_names))}, got {sim_matrix.shape}")

    cv_toks = [content_tokens(name) for name in cv_names]
    jd_toks = [content_tokens(name) for name in jd_names]
    cv_fold = [name.casefold() for name in cv_names]
    jd_fold = [name.casefold() for name in jd_names]

    inv: dict[str, list[int]] = defaultdict(list)
    for i, toks in enumerate(cv_toks):
        for tok in toks:
            inv[tok].append(i)

    raw_selected: list[tuple[str, str, str, str, float]] = []
    for j, jd_name in enumerate(jd_names):
        j_fold = jd_fold[j]
        j_tok = jd_toks[j]

        contained_indices: set[int] = set()
        if j_tok:
            candidate_pool: set[int] = set()
            for tok in j_tok:
                candidate_pool.update(inv.get(tok, ()))
            for i in candidate_pool:
                c_tok = cv_toks[i]
                if c_tok and cv_fold[i] != j_fold:
                    shorter, longer = (c_tok, j_tok) if len(c_tok) <= len(j_tok) else (j_tok, c_tok)
                    if shorter <= longer:
                        contained_indices.add(i)

        col = sim_matrix[:, j]
        thresh_indices = set(np.flatnonzero(col >= transfer).tolist())
        all_candidates = contained_indices | thresh_indices

        near_candidates: list[tuple[float, str, str]] = []
        transfer_candidates: list[tuple[float, str, str]] = []
        for i in all_candidates:
            if cv_fold[i] == j_fold:
                continue
            cos = float(col[i])
            is_contained = i in contained_indices
            if is_contained or cos >= near:
                reason = "token containment" if is_contained else f"cosine {cos:.2f}"
                near_candidates.append((cos, cv_names[i], reason))
            elif cos >= transfer:
                transfer_candidates.append((cos, cv_names[i], f"cosine {cos:.2f}"))

        near_candidates.sort(key=lambda item: (-item[0], item[1]))
        transfer_candidates.sort(key=lambda item: (-item[0], item[1]))

        for cos, cv_name, reason in near_candidates[:top_k]:
            raw_selected.append((cv_name, jd_name, "near", reason, cos))
        for cos, cv_name, reason in transfer_candidates[:top_k]:
            raw_selected.append((cv_name, jd_name, "transfer", reason, cos))

    relations_with_cos: list[tuple[SkillRelation, float]] = []
    relations_Payload: list[dict[str, str]] = []
    for idx, (cv_name, jd_name, kind, reason, cos) in enumerate(raw_selected, start=1):
        rel_dict = {
            "id": f"rel_{idx:05d}",
            "candidate": cv_name,
            "requirement": jd_name,
            "relation": kind,
            "reason": reason,
            "source": "auto:bge-m3",
            "review_status": "pending",
        }
        relations_Payload.append(rel_dict)
        relations_with_cos.append((SkillRelation.model_validate(rel_dict), cos))

    version = f"auto-bge-m3-v1-n{near}-t{transfer}"
    table = SkillRelationTable.model_validate({"version": version, "relations": relations_Payload})
    return table, relations_with_cos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near", type=float, default=0.80, help="Cosine threshold for near relation")
    parser.add_argument("--transfer", type=float, default=0.70, help="Cosine threshold for transfer relation")
    parser.add_argument("--top-k", type=int, default=3, help="Max relations per JD skill per kind")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output SkillRelationTable JSON path")
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL, help="Input lab_docs.pkl path")
    parser.add_argument("--emb-cache", type=Path, default=DEFAULT_EMB, help="Embedding cache .npz path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for printing sample relations")
    args = parser.parse_args()

    cv_names, jd_names = collect_skill_names(args.pkl)
    cv_emb, jd_emb = load_or_compute_embeddings(cv_names, jd_names, args.emb_cache)
    sim_matrix = cv_emb @ jd_emb.T

    table, relations_with_cos = build_relations(
        cv_names,
        jd_names,
        sim_matrix,
        near=args.near,
        transfer=args.transfer,
        top_k=args.top_k,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    near_count = sum(1 for r in table.relations if r.relation == "near")
    transfer_count = sum(1 for r in table.relations if r.relation == "transfer")
    containment_count = sum(1 for r in table.relations if r.reason == "token containment")
    cosine_near_count = near_count - containment_count

    print(
        f"CV skills: {len(cv_names)} | JD skills: {len(jd_names)} | "
        f"Total relations: {len(table.relations)} (near={near_count} [token_containment={containment_count}, "
        f"cosine={cosine_near_count}], transfer={transfer_count}) -> {args.out}"
    )

    rng = random.Random(args.seed)
    sample = rng.sample(relations_with_cos, k=min(30, len(relations_with_cos)))
    print("\n30 random sample relations:")
    for rel, cos in sample:
        print(f"  [{rel.id}] {rel.candidate!r} -> {rel.requirement!r} ({rel.relation}, {rel.reason}, cos={cos:.4f})")


if __name__ == "__main__":
    main()
