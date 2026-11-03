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
