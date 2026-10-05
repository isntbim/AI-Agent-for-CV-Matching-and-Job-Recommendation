"""Check that the EDA distinguishes score errors, rank agreement and class balance."""

import pandas as pd
import pytest

from scripts.ats_silver_eda import correlation, metrics


def test_score_shift_keeps_ranks_but_reports_error_and_bias():
    frame = pd.DataFrame({"cv_id": ["cv1"] * 3, "silver_score": [10., 20., 30.],
                          "runtime_score": [15., 25., 35.], "silver_grade": [0, 1, 2],
                          "runtime_grade": [0, 1, 1], "coverage": [.3, .4, .5]})
    result = metrics(frame)
    assert result["spearman"] == pytest.approx(1)
    assert result["pearson"] == pytest.approx(1)
    assert result["mae"] == pytest.approx(5)
    assert result["mean_bias"] == pytest.approx(5)
    assert result["agreement"]["accuracy"] == pytest.approx(2 / 3)
    assert result["majority_accuracy"] == pytest.approx(1 / 3)
    assert result["per_cv_spearman"]["median"] == pytest.approx(1)


def test_constant_scores_have_no_defined_rank_correlation():
    assert correlation(pd.Series([1., 2., 3.]), pd.Series([50., 50., 50.]), rank=True) is None


def test_tied_values_use_average_ranks():
    assert correlation(pd.Series([10., 10., 20.]), pd.Series([1., 1., 2.]), rank=True) == pytest.approx(1)
