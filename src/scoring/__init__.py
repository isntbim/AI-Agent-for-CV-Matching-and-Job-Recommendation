"""Evidence-backed, deterministic CV/JD scoring."""

from .models import JobInput, ResumeInput, SalaryPreference, ScorerConfig, ScoreResult
from .rules import score_pair as _score_pair_legacy


def score_pair(resume, job, preferences=None, config=None, as_of_date=None):
    if config is not None and config.version == "ats-en-v3":
        from .runtime_v3 import score_pair_v3
        return score_pair_v3(resume, job, preferences, config, as_of_date)
    return _score_pair_legacy(resume, job, preferences, config, as_of_date)

__all__ = ["JobInput", "ResumeInput", "SalaryPreference", "ScoreResult", "ScorerConfig", "score_pair"]
