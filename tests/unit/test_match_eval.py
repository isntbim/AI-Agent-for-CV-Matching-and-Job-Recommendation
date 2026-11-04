from scripts.match_eval import auc, metrics


def test_auc_counts_ties_as_half():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([2], [2]) == 0.5
    assert auc([2, 3], [1, 2]) == 0.875  # 3 wins + 1 tie of 4 comparisons


def test_metrics_perfect_ranking():
    rows = [{"cv_id": cv, "label": label, "score": 10 * label + i}
            for cv in ("a", "b") for i, label in enumerate((0, 0, 1, 2))]
    m = metrics(rows)
    assert m["auc_good_vs_no"] == m["auc_fit_vs_no"] == 1.0
    assert m["mean_per_cv_spearman"] > 0.9


def test_replay_returns_pod_rows_and_flags_missing_penalties(tmp_path):
    import json
    import pytest
    from scripts.match_eval import MissingGeneration, Replay
    (tmp_path / "prompts.jsonl").write_text(json.dumps({"id": "cv_x", "kind": "resume", "system": "s", "user": "u"}) + "\n")
    (tmp_path / "preds.jsonl").write_text(json.dumps({"id": "cv_x", "prediction": "{}", "finish_reason": "length",
                                                      "repetition_penalty": 1.05}) + "\n")
    replay = Replay(tmp_path / "prompts.jsonl", [tmp_path / "preds.jsonl"])
    assert replay.complete("s", "u", 1.05) == ("{}", "length")
    with pytest.raises(MissingGeneration):
        replay.complete("s", "u", 1.1)


def test_paired_difference_has_the_right_sign():
    import numpy as np
    from scripts.match_ir_eval import paired
    rng = np.random.default_rng(0)
    cv = np.repeat(np.arange(30), 20).astype(str)
    label = rng.integers(0, 3, len(cv))
    good, noise = label + rng.random(len(cv)) * 0.5, rng.random(len(cv))
    out = paired(cv, label, good, noise, reps=200)
    assert out["NDCG@10"]["diff"] > 0 and out["NDCG@10"]["ci95"][0] > 0 and out["NDCG@10"]["p"] < 0.05
