"""Agreement and ranking maths of the human-label evaluation (no models)."""
import numpy as np

from scripts.human_label_eval import agreement, evaluate, ranking_metrics


def test_agreement_and_consensus():
    export = {"a": {"cv1": {"j1": 2, "j2": 0, "j3": 1}}, "b": {"cv1": {"j1": 2, "j2": 1, "j3": 1, "j9": 0}}}
    stats, consensus = agreement(export)
    assert stats["pairs_labelled_by_all"] == 3 and stats["percent_agreement"] == round(2 / 3, 4)
    assert consensus == {("cv1", "j1"): 2.0, ("cv1", "j2"): 0.5, ("cv1", "j3"): 1.0}  # j9 only labelled by b


def test_ranking_metrics_perfect_order():
    m = ranking_metrics(np.array([3.0, 2.0, 1.0]), np.array([2.0, 1.0, 0.0]))
    assert m["NDCG@5"] == 1.0 and m["MRR"] == 1.0 and m["P@3"] == 2 / 3 and m["Kendall"] == 1.0


def test_evaluate_prefers_the_method_that_matches_humans():
    consensus = {(f"c{i}", f"j{k}"): float(k % 3) for i in range(12) for k in range(6)}
    good = {key: v for key, v in consensus.items()}
    bad = {key: -v for key, v in consensus.items()}
    table, test = evaluate(consensus, {"Final: Hybrid + TF-IDF (Qwen3, service)": good, "TF-IDF + cosine": bad}, reps=200)
    assert table["Final: Hybrid + TF-IDF (Qwen3, service)"]["NDCG@5"] == 1.0 and test["NDCG@5_diff"] > 0 and test["p"] < 0.05


def test_redact_removes_contact_details_but_keeps_dates_and_numbers():
    from scripts.build_label_set import redact
    text = ("Taylor M. Smith\nN O L A N D R U C K S\nsmithtm@gmail.com - (312) 826 \u2013 6800 - linkedin.com/in/x\n"
            "14 Conwell Street\nSaint Jacob IL 62281\nWorked 2015-2018 at Barney, sales up 17%")
    out = redact(text, "Taylor M. Smith")
    for leak in ("Taylor", "Smith", "N O L A N", "gmail", "826", "Conwell", "62281", "linkedin"):
        assert leak not in out
    assert "2015-2018" in out and "17%" in out
