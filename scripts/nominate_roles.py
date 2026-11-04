"""Nominate role-map additions (aliases and new canonical roles) for the V3 title dimension."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from src.scoring.normalization import ROLE_INDEX, contained_role, normalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TITLES = ROOT / "output/match_eval/lab/titles.json"
DEFAULT_OUT = ROOT / "output/match_eval/lab/role_nominations.json"
MODEL_NAME = "BAAI/bge-m3"

SENIORITY_RE = re.compile(
    r"\b(?:senior\s+level|entry\s+level|mid\s+level|senior|junior|sr|jr|lead|principal|staff|experienced)\b\.?",
    re.I,
)
CONTRACT_REMOTE_RE = re.compile(
    r"\b(?:contract-to-hire|contract\s+to\s+hire|contractor|contract|remote|hybrid|full-time|full\s+time|part-time|part\s+time|w2\s+positions?|w2)\b",
    re.I,
)
LEVEL_RE = re.compile(
    r"\b(?:level\s*(?:[1-9]|i|ii|iii|iv|v)|i|ii|iii|iv|v|ll|lll|[1-5])\b",
    re.I,
)


def clean_title(title: str) -> str:
    """Drop seniority words, level markers (I/II/3), and contract/remote/location noise."""
    if not title:
        return ""
    s = normalize(title).replace("_", " ")
    # Drop inline "Location: ..." or "Type: ..." tails (e.g. "PMLocation: Chicago, ILType: full-time (Hybrid)")
    s = re.sub(r"(?:location|type)\s*:.*$", "", s, flags=re.I)
    # Drop "w2 positions:" prefix before parentheticals so "W2 positions: (Marketing)" keeps "marketing"
    s = re.sub(r"^\s*w2\s+positions?\s*:\s*", "", s, flags=re.I)

    # If the entire remaining string is inside parentheses (e.g. "(marketing)"), unwrap it;
    # otherwise remove parenthetical expressions.
    stripped = s.strip()
    if stripped.startswith("(") and stripped.endswith(")") and "(" not in stripped[1:-1]:
        s = stripped[1:-1]
    else:
        without_parens = re.sub(r"\([^)]*\)", " ", s)
        if normalize(without_parens):
            s = without_parens

    s = CONTRACT_REMOTE_RE.sub(" ", s)
    s = SENIORITY_RE.sub(" ", s)
    s = LEVEL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip(" -/,.:;")
    return s or normalize(title)


def rank_roles_for_titles(
    cleaned_titles: list[str],
    title_emb: np.ndarray,
    alias_names: list[str],
    alias_emb: np.ndarray,
) -> list[dict[str, Any]]:
    """Compute top-2 canonical roles and best different-family margin for each cleaned title."""
    sim = title_emb @ alias_emb.T  # shape: (len(cleaned_titles), len(alias_names))

    # Group alias indices by (canonical_role, family)
    role_to_indices: dict[tuple[str, str], list[int]] = {}
    for idx, alias in enumerate(alias_names):
        role_fam = ROLE_INDEX[alias]
        role_to_indices.setdefault(role_fam, []).append(idx)

    role_keys = sorted(role_to_indices.keys())
    results: list[dict[str, Any]] = []

    for i in range(len(cleaned_titles)):
        row_sim = sim[i]
        scored_roles: list[tuple[float, str, str]] = []
        for canonical_role, family in role_keys:
            indices = role_to_indices[(canonical_role, family)]
            best_alias_cos = float(np.max(row_sim[indices]))
            scored_roles.append((best_alias_cos, canonical_role, family))

        scored_roles.sort(key=lambda item: (-item[0], item[1], item[2]))
        top1_cos, top1_role, top1_family = scored_roles[0]
        top2_cos, top2_role, top2_family = scored_roles[1]

        diff_family_scores = [cos for cos, _, fam in scored_roles if fam != top1_family]
        best_diff_cos = max(diff_family_scores) if diff_family_scores else 0.0
        margin = top1_cos - best_diff_cos

        if top1_cos >= 0.80 and margin >= 0.05:
            action = "alias"
        elif top1_cos >= 0.70:
            action = "new_role"
        else:
            action = "unresolved"

        results.append(
            {
                "action": action,
                "role": top1_role,
                "family": top1_family,
                "cosine": round(top1_cos, 4),
                "margin": round(margin, 4),
                "top2_role": top2_role,
                "top2_family": top2_family,
                "top2_cosine": round(top2_cos, 4),
            }
        )
    return results


def build_nominations(
    titles_path: Path = DEFAULT_TITLES,
    model_name: str = MODEL_NAME,
) -> list[dict[str, Any]]:
    """Collect unmapped titles from titles.json, embed cleaned titles & ROLE_INDEX, and nominate roles."""
    titles_map: dict[str, str] = json.loads(titles_path.read_text(encoding="utf-8"))
    counts = Counter(
        title for title in titles_map.values() if title and contained_role(title)[1] is None
    )

    distinct_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    raw_titles = [item[0] for item in distinct_items]
    n_docs_list = [item[1] for item in distinct_items]
    cleaned_titles = [clean_title(t) for t in raw_titles]
    alias_names = sorted(ROLE_INDEX.keys())

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    unique_strings = sorted(set(cleaned_titles) | set(alias_names))
    all_emb = model.encode(
        unique_strings,
        batch_size=128,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    str_to_idx = {s: idx for idx, s in enumerate(unique_strings)}

    title_emb = all_emb[[str_to_idx[s] for s in cleaned_titles]]
    alias_emb = all_emb[[str_to_idx[s] for s in alias_names]]

    ranked = rank_roles_for_titles(cleaned_titles, title_emb, alias_names, alias_emb)
    nominations: list[dict[str, Any]] = []
    for raw_t, clean_t, n_docs, info in zip(raw_titles, cleaned_titles, n_docs_list, ranked):
        nominations.append(
            {
                "title": raw_t,
                "cleaned": clean_t,
                "action": info["action"],
                "role": info["role"],
                "family": info["family"],
                "cosine": info["cosine"],
                "margin": info["margin"],
                "n_docs": n_docs,
            }
        )
    return nominations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--titles", type=Path, default=DEFAULT_TITLES, help="Path to titles.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output JSON path")
    args = parser.parse_args()

    nominations = build_nominations(args.titles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(nominations, indent=2, ensure_ascii=False), encoding="utf-8")

    by_action = Counter(item["action"] for item in nominations)
    docs_by_action = Counter()
    for item in nominations:
        docs_by_action[item["action"]] += item["n_docs"]
    print(
        f"Distinct unmapped titles: {len(nominations)} (total docs={sum(docs_by_action.values())}) | "
        f"alias={by_action['alias']} ({docs_by_action['alias']} docs), "
        f"new_role={by_action['new_role']} ({docs_by_action['new_role']} docs), "
        f"unresolved={by_action['unresolved']} ({docs_by_action['unresolved']} docs)"
    )


if __name__ == "__main__":
    main()
