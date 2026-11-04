"""Unit tests for scripts/build_skill_relations.py (no model download)."""

from __future__ import annotations

import numpy as np

from scripts.build_skill_relations import build_relations, content_tokens, token_containment
from src.parser.skill_v2 import SkillRelationTable


def test_token_containment_rule() -> None:
    assert token_containment("microsoft office", "microsoft office suite programs")
    assert token_containment("microsoft office suite programs", "microsoft office")
    assert token_containment("excel", "ms excel")
    assert token_containment("point of sale", "point sale systems")
    assert not token_containment("excel", "excel")
    assert not token_containment("Excel", "excel")
    assert not token_containment("c#", "c++")
    assert not token_containment("microsoft word", "microsoft excel")
    assert content_tokens("management of the project") == {"management", "project"}


def test_threshold_and_top_k_selection_on_handmade_matrix() -> None:
    cv_names = [
        "microsoft office",      # 0: token containment with jd[0], low cosine (0.60) -> still near
        "office 365",            # 1: cosine 0.92 with jd[0] -> near
        "libreoffice",           # 2: cosine 0.85 with jd[0] -> near
        "google workspace",      # 3: cosine 0.81 with jd[0] -> near (dropped when top_k=2)
        "openoffice",            # 4: cosine 0.76 with jd[0] -> transfer
        "iwork",                 # 5: cosine 0.74 with jd[0] -> transfer
        "wps office",            # 6: cosine 0.71 with jd[0] -> transfer (dropped when top_k=2)
        "notepad",               # 7: cosine 0.40 with jd[0] -> below transfer
        "python",                # 8: exact match with jd[1] -> must be excluded even with cosine 1.0
    ]
    jd_names = [
        "microsoft office suite programs",
        "python",
    ]

    sim = np.array(
        [
            [0.60, 0.10],  # microsoft office
            [0.92, 0.15],  # office 365
            [0.85, 0.12],  # libreoffice
            [0.81, 0.11],  # google workspace
            [0.76, 0.20],  # openoffice
            [0.74, 0.18],  # iwork
            [0.71, 0.05],  # wps office
            [0.40, 0.02],  # notepad
            [0.10, 1.00],  # python (same as jd[1])
        ],
        dtype=np.float32,
    )

    # With top_k=2, near=0.80, transfer=0.70:
    # For jd[0], near candidates are office 365 (0.92), libreoffice (0.85), google workspace (0.81), microsoft office (0.60 via containment).
    # Highest 2 cosines among near: office 365 (0.92), libreoffice (0.85).
    # Transfer candidates for jd[0]: openoffice (0.76), iwork (0.74), wps office (0.71).
    # Highest 2 cosines among transfer: openoffice (0.76), iwork (0.74).
    # For jd[1] ("python"), exact match "python" is skipped and all others are < 0.70.
    table, with_cos = build_relations(cv_names, jd_names, sim, near=0.80, transfer=0.70, top_k=2)
    validated = SkillRelationTable.model_validate(table.model_dump(mode="json"))

    assert validated.version == "auto-bge-m3-v1-n0.8-t0.7"
    assert [r.id for r in validated.relations] == ["rel_00001", "rel_00002", "rel_00003", "rel_00004"]
    assert [(r.candidate, r.relation, r.reason) for r in validated.relations] == [
        ("office 365", "near", "cosine 0.92"),
        ("libreoffice", "near", "cosine 0.85"),
        ("openoffice", "transfer", "cosine 0.76"),
        ("iwork", "transfer", "cosine 0.74"),
    ]
    assert all(r.source == "auto:bge-m3" and r.review_status == "pending" for r in validated.relations)
    assert len(with_cos) == 4

    # With top_k=4, token containment candidate ("microsoft office", cos=0.60 < 0.80) is included as "near"
    table4, _ = build_relations(cv_names, jd_names, sim, near=0.80, transfer=0.70, top_k=4)
    near_rels = [r for r in table4.relations if r.relation == "near"]
    assert [r.candidate for r in near_rels] == [
        "office 365",
        "libreoffice",
        "google workspace",
        "microsoft office",
    ]
    assert near_rels[-1].reason == "token containment"
