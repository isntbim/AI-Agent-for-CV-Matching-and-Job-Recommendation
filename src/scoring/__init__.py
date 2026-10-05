"""Evidence-backed, deterministic CV/JD scoring."""

from .models import JobInput, ResumeInput, SalaryPreference, ScorerConfig, ScoreResult
from .rules import score_pair

__all__ = ["JobInput", "ResumeInput", "SalaryPreference", "ScoreResult", "ScorerConfig", "score_pair"]
