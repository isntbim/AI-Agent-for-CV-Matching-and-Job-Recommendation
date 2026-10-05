"""Pure deterministic rules. Each dimension carries an attribution rule ID."""

import re
from datetime import date

from src.parser.match_schemas import ATSDimensionalScores

from .models import (
    DimensionResult,
    JobInput,
    ResumeInput,
    SalaryPreference,
    ScorerConfig,
    ScoreResult,
    runtime_grade,
    supported_coverage,
)
from .normalization import country_name, domain_set, normalize, role, skill_set


def _result(score: float, supported: bool, basis: str, rule_id: str,
            resume: ResumeInput, job: JobInput, dimension: str) -> DimensionResult:
    return DimensionResult(score=max(0, min(100, round(score))), supported=supported,
                           basis=basis, rule_id=rule_id,
                           evidence=[e.quote for obj in (resume, job)
                                     for e in obj.facts.evidence.get(dimension, [])])


def _month(value: str | None, as_of: date, ongoing: bool = False) -> int | None:
    if ongoing and value and normalize(value) in {"present", "current", "ongoing", "now"}:
        return as_of.year * 12 + as_of.month
    if not value or not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", value):
        return None
    try:
        parsed = date.fromisoformat(value if len(value) == 10 else value + "-01")
    except ValueError:
        return None
    if parsed > as_of:
        return None
    return parsed.year * 12 + parsed.month


def relevant_months(resume: ResumeInput, job_title: str, as_of: date) -> int | None:
    """Union of relevant dated intervals, never a sum of overlapping jobs."""
    target, family = role(job_title)
    intervals = []
    relevant = False
    ambiguous_role = False
    for work in resume.document.work:
        name, work_family = role(work.position)
        if not name or not (name == target or (family and work_family == family)):
            if not name or not work_family:
                ambiguous_role = True
            continue
        relevant = True
        start = _month(work.start_date, as_of)
        end = _month(work.end_date, as_of, ongoing=True)
        if start is None or end is None or end < start:
            return None
        intervals.append((start, end))
    if ambiguous_role:
        return None
    if not relevant:
        # Empty work list may be incomplete parsing, not proof of no experience.
        return 0 if resume.document.work else None
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _latest_title(resume: ResumeInput, as_of: date) -> str | None:
    if resume.document.basics.label:
        return resume.document.basics.label
    dated = [(end, work.position) for work in resume.document.work
             if work.position and (end := _month(work.end_date, as_of, ongoing=True)) is not None]
    return max(dated, key=lambda item: item[0])[1] if dated else None


def score_pair(resume: ResumeInput, job: JobInput, preferences: SalaryPreference | None = None,
               config: ScorerConfig | None = None, as_of_date: date | None = None) -> ScoreResult:
    config = config or ScorerConfig()
    preferences = preferences or SalaryPreference()
    if as_of_date is None:
        raise ValueError("Pass as_of_date explicitly for reproducible employment durations")
    dimensions = {}

    def put(key, score, supported, basis, rule_id):
        dimensions[key] = _result(score, supported, basis, rule_id, resume, job, key)

    cv_skills = skill_set(resume.document.get_flat_skills())
    required = skill_set([s.name for s in job.document.skills.required])
    preferred = skill_set([s.name for s in job.document.skills.preferred])
    targets = required or preferred
    matched = sorted(cv_skills & targets)
    missing = sorted(targets - cv_skills)
    known_skills = resume.facts.has("skills") and job.facts.has("skills") and bool(targets)
    put("skills", 100 * len(matched) / len(targets) if known_skills else config.neutral,
        known_skills, "required_coverage" if required and known_skills else
        "preferred_only_coverage" if known_skills else "missing_skill_evidence", "R1")
    preferred_coverage = (round(100 * len(cv_skills & preferred) / len(preferred), 2)
                          if preferred and resume.facts.has("skills") and job.facts.has("skills") else None)

    candidate_title = _latest_title(resume, as_of_date)
    cv_role, cv_family = role(candidate_title)
    jd_role, jd_family = role(job.document.title)
    known_title = bool(cv_role and jd_role and resume.facts.has("title") and job.facts.has("title")
                       and (cv_role == jd_role or (cv_family and jd_family)))
    title_score = 100 if cv_role == jd_role else 70 if cv_family and cv_family == jd_family else 0
    put("title", title_score if known_title else config.neutral, known_title,
        "canonical_role" if known_title else "missing_title_evidence", "P1")

    months = relevant_months(resume, job.document.title, as_of_date)
    years = job.document.seniority.min_years
    exp_known = job.facts.has("experience") and resume.facts.has("experience") and months is not None and years is not None
    if job.facts.has("experience") and (job.facts.no_experience_required or years == 0):
        put("experience", 100, True, "explicit_no_experience_requirement", "P2")
    elif exp_known and years > 0:
        put("experience", 100 * min(months / (12 * years), 1), True, "relevant_months_union", "P2")
    else:
        put("experience", config.neutral, False, "missing_duration_or_requirement", "P2")

    ranks = {"high_school": 1, "associate": 2, "bachelor": 3, "master": 4, "doctorate": 5}
    minimum = ranks.get(job.facts.minimum_degree)
    attained = max((ranks[d] for d in resume.facts.completed_degrees), default=None)
    if job.facts.has("education") and job.facts.no_education_required:
        put("education", 100, True, "explicit_no_education_requirement", "R2")
    elif minimum is not None and attained is not None and resume.facts.has("education") and job.facts.has("education"):
        gap = minimum - attained
        put("education", 100 if gap <= 0 else 50 if gap == 1 else 0, True, "completed_degree_requirement", "R2")
    else:
        put("education", config.neutral, False, "missing_degree_evidence", "R2")

    cv_domains = domain_set(resume.facts.domains, [candidate_title or "", *[w.position or "" for w in resume.document.work]])
    jd_domains = domain_set(job.facts.domains, [job.document.title])
    known_domain = bool(cv_domains and jd_domains and resume.facts.has("industry") and job.facts.has("industry"))
    put("industry", 100 * len(cv_domains & jd_domains) / len(jd_domains) if known_domain else config.neutral,
        known_domain, "professional_domain_coverage" if known_domain else "missing_domain_evidence", "R3")

    cv_country = country_name(resume.facts.country)
    jd_country = country_name(job.facts.country)
    eligible = {country_name(v) for v in job.facts.eligible_countries}
    cv_city = normalize(resume.document.basics.location.city or "") if resume.document.basics.location else ""
    jd_city = normalize(job.document.location.city or "")
    location_score, location_known, location_basis = config.neutral, False, "missing_location_constraints"
    if job.facts.has("location") and job.facts.remote and job.facts.worldwide_remote:
        location_score, location_known, location_basis = 100, True, "explicit_worldwide_remote"
    elif resume.facts.has("location") and job.facts.has("location"):
        if job.facts.remote and eligible and cv_country:
            location_score, location_known, location_basis = (100 if cv_country in eligible else 0), True, "remote_country_eligibility"
        elif not job.facts.remote and cv_country and jd_country and cv_country != jd_country:
            location_score, location_known, location_basis = 0, True, "explicit_country_mismatch"
        elif not job.facts.remote and cv_city and jd_city and cv_country and jd_country:
            location_score, location_known, location_basis = (100 if cv_city == jd_city else 0), True, "onsite_city_match"
    put("location", location_score, location_known, location_basis, "P3")

    salary_score, salary_known, salary_basis = config.salary_neutral, False, "negotiable_default"
    if preferences.mode == "explicit":
        salary_basis = "undisclosed_or_incomparable_salary"
        comp = job.document.compensation
        cap = comp.max_monthly
        if cap is None and comp.max_annual is not None:
            cap = comp.max_annual / 12
        expected = preferences.minimum / 12 if preferences.period == "annual" else preferences.minimum
        if job.facts.has("salary") and job.facts.currency and normalize(job.facts.currency) == normalize(preferences.currency) and cap and cap > 0:
            salary_score, salary_known, salary_basis = 100 * min(cap / expected, 1), True, "explicit_salary_comparison"
    put("salary", salary_score, salary_known, salary_basis, "P4")
    scores = ATSDimensionalScores(**{key: value.score for key, value in dimensions.items()})
    total = scores.compute_overall()
    return ScoreResult(scores=scores, overall_score=total,
                       grade=runtime_grade(total, scores.skills, scores.title, config),
                       coverage=supported_coverage(dimensions), dimensions=dimensions,
                       matched_skills=matched, missing_skills=missing,
                       preferred_skill_coverage=preferred_coverage, config=config,
                       as_of_date=as_of_date.isoformat())
