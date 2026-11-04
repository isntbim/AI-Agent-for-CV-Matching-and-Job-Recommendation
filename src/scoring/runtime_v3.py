"""V3 full-precision aggregation; shared legacy runtime remains byte-identical."""
from datetime import date
from src.parser.match_schemas import ATS_WEIGHTS
from .models import (JobInput, ResumeInput, SalaryPreference, ScorerConfig,
                     runtime_grade, supported_coverage)
from .models_v3 import (ATSDimensionalScoresV3, DimensionResultV3, ScoreResultV3,
                        display_score)
from .normalization import contained_role, country_name, domain_set, normalize, role
from .rules import _latest_title, is_partial_history, relevant_months
from .skills_v3 import score_skills_v3


def score_pair_v3(resume: ResumeInput, job: JobInput, preferences: SalaryPreference | None = None,
                  config: ScorerConfig | None = None, as_of_date: date | None = None) -> ScoreResultV3:
    config = config or ScorerConfig(version="ats-en-v3")
    if config.version != "ats-en-v3":
        raise ValueError("The precision runtime requires ats-en-v3")
    preferences = preferences or SalaryPreference()
    if as_of_date is None:
        raise ValueError("Pass as_of_date explicitly for reproducible employment durations")
    dimensions = {}

    def put(key, score, supported, basis, rule_id):
        raw = max(0, min(100, score))
        dimensions[key] = DimensionResultV3(score=raw, unrounded_score=raw,
            supported=supported, basis=basis, rule_id=rule_id,
            evidence=[e.quote for obj in (resume, job) for e in obj.facts.evidence.get(key, [])])

    skills_breakdown = score_skills_v3(resume, job, config)
    matched = skills_breakdown["matched_skills"]
    missing = skills_breakdown["missing_skills"]
    preferred_coverage = skills_breakdown["preferred_skill_coverage"]
    put("skills", skills_breakdown["score"], skills_breakdown["supported"],
        skills_breakdown["basis"], "R1-v3")
    candidate_title = _latest_title(resume, as_of_date)
    title_role = contained_role if config.title_policy == "contained" else role
    cv_role, cv_family = title_role(candidate_title)
    jd_role, jd_family = title_role(job.document.title)
    known_title = bool(cv_role and jd_role and resume.facts.has("title") and job.facts.has("title")
                       and (cv_role == jd_role or (cv_family and jd_family)))
    title_score = 100 if cv_role == jd_role else 70 if cv_family and cv_family == jd_family else 0
    put("title", title_score if known_title else config.neutral, known_title,
        "canonical_role" if known_title else "missing_title_evidence", "P1")

    lower_bound = config.experience_policy == "lower_bound"
    months = relevant_months(resume, job.document.title, as_of_date, lower_bound=lower_bound)
    years = job.document.seniority.min_years
    exp_known = job.facts.has("experience") and resume.facts.has("experience") and months is not None and years is not None
    if exp_known and lower_bound and months < 12 * years and is_partial_history(resume, job.document.title):
        exp_known = False  # a short lower bound cannot show the requirement is missed
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
    unrounded = {key: value.unrounded_score for key, value in dimensions.items()}
    total = sum(ATS_WEIGHTS[key] * unrounded[key] for key in ATS_WEIGHTS)
    grade = runtime_grade(total, unrounded["skills"], unrounded["title"], config, skills_breakdown)
    # Format only after the entire aggregation and grade decision.
    for value in dimensions.values():
        value.score = display_score(value.unrounded_score)
    return ScoreResultV3(
        scores=ATSDimensionalScoresV3(**{key: value.score for key, value in dimensions.items()}),
        unrounded_scores=ATSDimensionalScoresV3(**unrounded), unrounded_overall_score=total,
        overall_score=display_score(total), grade=grade,
        coverage=supported_coverage(dimensions), dimensions=dimensions,
        matched_skills=matched, missing_skills=missing, preferred_skill_coverage=preferred_coverage,
        config=config, as_of_date=as_of_date.isoformat(), skills_breakdown=skills_breakdown)

