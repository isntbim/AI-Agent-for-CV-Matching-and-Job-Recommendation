"""Prespecified parameters are harness-only and restored even on failure."""
import pytest
from src.scoring import skills_v3
from src.scoring.v3_depth_sensitivity import depth_credit,parameter_scope,controlled_rows


@pytest.mark.parametrize("step,gap,credit",[
    (.25,1,.75),(.25,2,.5),(.15,1,.85),(.15,2,.7),
    (.35,1,.65),(.35,2,.3),(.35,4,0),(.25,10,0)])
def test_credit_and_zero_floor(step,gap,credit):
    assert depth_credit(gap,step)==pytest.approx(credit)


def test_restoration_and_scope_validation():
    original=skills_v3._depth
    with pytest.raises(RuntimeError):
        with parameter_scope(.15):
            with pytest.raises(ValueError):
                with parameter_scope(.35):
                    pass
            raise RuntimeError("controlled failure")
    assert skills_v3._depth is original
    with pytest.raises(ValueError):
        with parameter_scope(.2):
            pass


def test_constructed_gaps_are_sensitive_but_assumed_baseline_is_not():
    with parameter_scope(.25):
        base={r["pair_id"]:r["result"] for r in controlled_rows()}
    with parameter_scope(.35):
        steep={r["pair_id"]:r["result"] for r in controlled_rows()}
    assert base["intermediate::expert"]["grade"]==1
    assert steep["intermediate::expert"]["grade"]==0
    assert steep["normal::expert"]["unrounded_scores"]["skills"]==100
    assert steep["normal::expert"]["grade"]==base["normal::expert"]["grade"]

