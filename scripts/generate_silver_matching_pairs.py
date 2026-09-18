"""
LLM-as-a-Judge Silver Annotation Pipeline (ATS Scoring Matrix)

Generates ground-truth benchmark CV-JD matching pairs
(data/datasets/matching_pairs/) by evaluating standardized gold CVs against
structured JDs with a 7-dimensional ATS rubric.

Architecture (per implementation plan):
- The LLM judge (google/gemini-3.7-flash-high via Nova Gateway) scores ONLY the
  7 ATS dimensions (0-100 each) + rationale + matched/missing skills.
- overall_score is computed deterministically by this script (ATS_WEIGHTS)
  and grade by compute_grade() — zero LLM drift on the headline numbers.
- Pair selection uses a static domain-mapping table (NO vector DB), avoiding
  the circular dependency of needing retrieval to build the retrieval benchmark.

Pair tiers:
  Tier A  in_domain (10 JDs/CV)           ~880 pairs  -> LLM judge
  Tier B  cross_domain hard negatives (3)  ~264 pairs  -> LLM judge
  Tier C  heuristic disjoint negatives (6)  ~528 pairs  -> script-generated (0 API)

Resilience:
- Incremental cache .cache_silver_pairs.jsonl (1 line / completed LLM pair);
  on restart, cached (cv_id, jd_id) pairs are skipped.
- Retry x3 with exponential backoff (2s/4s/8s); failures logged to
  .cache_errors.jsonl and excluded from the benchmark.
- Rate limiting: asyncio.Semaphore(10) + 0.1s pacing between request starts.

Usage:
  python scripts/generate_silver_matching_pairs.py              # DRY-RUN: 5 curated pairs, prints only
  python scripts/generate_silver_matching_pairs.py --full       # full generation + outputs + stats
  python scripts/generate_silver_matching_pairs.py --full --limit 100   # capped LLM batch (staged Phase 2)
  python scripts/generate_silver_matching_pairs.py --heuristic-only     # Tier C only (0 API calls)
"""

import asyncio
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import BadRequestError, OpenAI

from src.parser.match_schemas import (  # noqa: E402
    ATSDimensionalScores,
    LLMJudgeResponse,
    SilverMatchPair,
    compute_grade,
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CV_INDEX = PROJECT_ROOT / "data" / "processed" / "structured_cvs" / "all_cvs.json"
JD_INDEX = PROJECT_ROOT / "data" / "processed" / "structured_jobs" / "all_jobs_en.json"
OUT_DIR = PROJECT_ROOT / "data" / "datasets" / "matching_pairs"
NOVA_KEY_FILE = PROJECT_ROOT / "knowledge-base" / "nova-api.txt"
NOVA_BASE_URL = "https://novagateway.net/v1"
JUDGE_MODEL = "google/gemini-3.7-flash-high"

CACHE_FILE = OUT_DIR / ".cache_silver_pairs.jsonl"
ERROR_FILE = OUT_DIR / ".cache_errors.jsonl"
BENCHMARK_FILE = OUT_DIR / "benchmark_matching_pairs.json"
STATS_FILE = OUT_DIR / "annotation_stats.json"

EXCLUDED_CATEGORIES = {"security_engineer", "devops_cloud"}  # not in the 6-domain mapping

# CV domain -> (primary JD category, cross-over JD categories, disjoint JD categories)
DOMAIN_MAP = {
    "software_engineer": {
        "primary": "software_engineer",
        "cross": ["frontend_web_developer", "data_scientist_analyst"],
        "disjoint": ["accountant", "hr", "marketing_executive"],
    },
    "frontend_web_developer": {
        "primary": "frontend_web_developer",
        "cross": ["software_engineer"],
        "disjoint": ["accountant", "hr", "marketing_executive", "data_scientist_analyst"],
    },
    "data_scientist_analyst": {
        "primary": "data_scientist_analyst",
        "cross": ["software_engineer"],
        "disjoint": ["accountant", "hr", "marketing_executive", "frontend_web_developer"],
    },
    "accountant": {
        "primary": "accountant",
        "cross": ["hr"],
        "disjoint": ["software_engineer", "frontend_web_developer", "data_scientist_analyst", "marketing_executive"],
    },
    "hr": {
        "primary": "hr",
        "cross": ["accountant", "marketing_executive"],
        "disjoint": ["software_engineer", "frontend_web_developer", "data_scientist_analyst"],
    },
    "marketing_executive": {
        "primary": "marketing_executive",
        "cross": ["hr"],
        "disjoint": ["software_engineer", "frontend_web_developer", "data_scientist_analyst", "accountant"],
    },
}

TIER_A_PER_CV = 10
TIER_B_PER_CV = 3
TIER_C_PER_CV = 6

DRY_RUN_PAIRS = [  # curated: (cv_domain, jd_category) covering all expected grades
    ("software_engineer", "software_engineer"),      # in-domain, expect Grade 2/1
    ("marketing_executive", "marketing_executive"),  # in-domain, expect Grade 1/2
    ("frontend_web_developer", "software_engineer"), # cross-domain, expect Grade 0/1
    ("accountant", "hr"),                            # cross-domain, expect Grade 0/1
    ("software_engineer", "accountant"),             # disjoint, expect Grade 0
]

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a senior ATS recruiter with 15 years of experience evaluating "
    "candidate resumes against job descriptions. Score each dimension strictly on "
    "a 0-100 scale and be calibrated: 100 = perfect top-decile alignment, 70-89 = "
    "strong fit, 50-69 = acceptable with gaps, below 50 = poor. If the job posting "
    "does not disclose salary or location, apply the rubric defaults (salary=75, "
    "location=75). Return ONLY a valid JSON object, no markdown fences, no commentary."
)

RUBRIC_TEXT = """Rubric (score each dimension 0-100):
- skills (weight 0.30): % of required JD skills matched by CV tech stack. Bonus for preferred skills.
- title (weight 0.20): role alignment: exact match=100, related role=60-80, unrelated=0-30.
- experience (weight 0.15): years fit: exceeds/meets=90-100, slightly under=60-80, far under=0-40.
- education (weight 0.10): degree level + field relevance: exact=100, adjacent field=70, unrelated=30.
- industry (weight 0.10): employer industry overlap: same=100, adjacent=60-80, unrelated but tech=40-60.
- location (weight 0.08): same city or Remote=100, Hybrid nearby=80, relocation=30-50. Default 80 if undisclosed.
- salary (weight 0.07): within range=100, slightly above=60-80, far above/below=20-40. Default 75 if undisclosed.

Output JSON schema:
{
  "scores": {"skills": 0-100, "title": 0-100, "experience": 0-100, "education": 0-100,
             "industry": 0-100, "location": 0-100, "salary": 0-100},
  "rationale": "1-3 sentence justification",
  "missing_skills": ["concrete skill nouns from the job requirements the resume does not demonstrate (NO job titles, role names, or seniority labels)"],
  "matched_skills": ["concrete skill nouns from the job requirements the resume demonstrates (NO job titles, role names, or seniority labels)"]
}"""

# ---------------------------------------------------------------------------
# Title-level skill filtering (CareerViet tag pollution guard)
# ---------------------------------------------------------------------------
# CareerViet tag clouds leak job-title variants and seniority labels into the
# skills block (e.g. jd_047 required skills are 7 rewordings of "Accountant").
# These are not skills: they must not reach the judge, the benchmark, or the
# reported matched/missing lists.  _is_title_like() is the deterministic guard.
ROLE_WORDS = {
    "developer", "engineer", "analyst", "accountant", "manager", "executive",
    "director", "specialist", "officer", "supervisor", "consultant", "intern",
    "trainee", "fresher", "junior", "senior", "middle", "lead", "expert",
}
ROLE_SUBSTRINGS = ("nhân viên", "chuyên viên", "kiểm toán viên", "kế toán viên",
                   "thực tập sinh", "trưởng phòng", "quản lý")


def _is_title_like(name: str, title_normalized: str) -> bool:
    """True if a skill-list entry is actually a job title, role name, or seniority label."""
    n = (name or "").strip()
    if not n:
        return True
    nf = n.casefold()
    t = (title_normalized or "").replace("_", " ").strip().casefold()
    # exact title or containment either way (handles 'Accountant', 'Frontend Developer', ...)
    if t and (nf == t or t in nf or nf in t):
        return True
    # whole-word role/seniority labels
    for word in nf.split():
        if word in ROLE_WORDS:
            return True
    # multiword Vietnamese role phrases
    for marker in ROLE_SUBSTRINGS:
        if marker in nf:
            return True
    return False


def _jd_prompt_text(jd_record: dict) -> str:
    """JobDescriptionSchema.to_embedding_text() with title-like tokens stripped from
    skill lists, so the judge only sees genuine skills (jd_047-style pollution excluded)."""
    from src.parser.job_schemas import JobDescriptionSchema

    jd = JobDescriptionSchema.model_validate(jd_record)
    t = jd_record.get("title_normalized") or ""
    jd.skills.required = [s for s in jd.skills.required if not _is_title_like(s.name, t)]
    jd.skills.preferred = [s for s in jd.skills.preferred if not _is_title_like(s.name, t)]
    text = jd.to_embedding_text()
    # Annotate fully-filtered headlines so the judge knows the JD lists no
    # concrete skills instead of silently seeing "Required Skills: "
    text = text.replace(
        "Required Skills: \n",
        "Required Skills: (posting lists only role labels — no concrete skills)\n",
    )
    if text.endswith("Required Skills: "):
        text = text[:-len("Required Skills: ")] + (
            "Required Skills: (posting lists only role labels — no concrete skills)")
    return text


def _filter_judge_skills(llm: LLMJudgeResponse, jd_record: dict) -> LLMJudgeResponse:
    """Deterministic backstop: strip title-like tokens from the judge's
    matched/missing lists and dedupe (case-insensitive), preserving order."""
    t = jd_record.get("title_normalized") or ""

    def keep(items):
        out, seen = [], set()
        for it in items:
            key = (it or "").strip().casefold()
            if key and key not in seen and not _is_title_like(it, t):
                seen.add(key)
                out.append(it.strip())
        return out

    llm.missing_skills = keep(llm.missing_skills)
    llm.matched_skills = keep(llm.matched_skills)
    return llm


def build_judge_prompt(cv_record, jd_record):
    cv_text = cv_record.get("summary_text") or _resume_text_from_record(cv_record)
    jd_text = jd_record.get("_jd_prompt_text") or _jd_prompt_text(jd_record)

    user_prompt = f"""Evaluate this CV against this job description using the ATS rubric.

===== CANDIDATE RESUME (domain: {cv_record['domain']}) =====
{cv_text}

===== JOB DESCRIPTION (category: {jd_record['category']}, id: {jd_record['id']}) =====
{jd_text}

{RUBRIC_TEXT}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _resume_text_from_record(cv_record):
    """Fallback embedding-text synth if summary_text absent (mirrors ResumeSchema.to_embedding_text)."""
    from src.parser.schemas import ResumeSchema

    return ResumeSchema.model_validate(cv_record).to_embedding_text()


def call_judge(client, cv_record, jd_record):
    """Call the LLM judge once, parse + validate response. Returns LLMJudgeResponse or raises."""
    messages = build_judge_prompt(cv_record, jd_record)
    try:
        resp = _create_completion(client, messages, use_json_mode=True)
    except Exception as exc:  # gateway may reject json_object mode — retry without it
        if isinstance(exc, BadRequestError) or "response_format" in str(exc).lower() or "json" in str(exc).lower():
            resp = _create_completion(client, messages, use_json_mode=False)
        else:
            raise

    content = resp.choices[0].message.content
    data = _parse_llm_json(content)
    return LLMJudgeResponse.model_validate(data)


def _create_completion(client, messages, use_json_mode):
    kwargs = {"model": JUDGE_MODEL, "messages": messages, "temperature": 0.1, "max_tokens": 800}
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs)

    content = resp.choices[0].message.content
    data = _parse_llm_json(content)
    return LLMJudgeResponse.model_validate(data)


def _parse_llm_json(content):
    """Parse LLM JSON output, tolerating markdown fences or trailing prose."""
    if content is None:
        raise ValueError("LLM returned empty content")
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: extract the first balanced {...} block
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


async def judge_pair(client, cv_record, jd_record, max_retries=3, backoff=(2, 4, 8)):
    """Evaluate one pair with retries. Returns SilverMatchPair or None (after exhausting retries)."""
    for attempt in range(max_retries + 1):
        try:
            llm = await asyncio.to_thread(call_judge, client, cv_record, jd_record)
            llm = _filter_judge_skills(llm, jd_record)
            overall = llm.scores.compute_overall()
            grade = compute_grade(overall, llm.scores.skills, llm.scores.title)
            return SilverMatchPair(
                cv_id=cv_record["cv_id"],
                jd_id=jd_record["id"],
                cv_domain=cv_record["domain"],
                jd_domain=jd_record["category"],
                pair_type="heuristic_negative",  # patched below by caller
                grade=grade,
                overall_score=overall,
                scores=llm.scores,
                rationale=llm.rationale,
                missing_skills=llm.missing_skills,
                matched_skills=llm.matched_skills,
            )
        except Exception as exc:  # noqa: BLE001 — retry any judge failure
            last_exc = exc
            if attempt < max_retries:
                await asyncio.sleep(backoff[attempt])
    print(f"    !! judge failed after {max_retries + 1} attempts: {cv_record['cv_id']} x {jd_record['id']}: {last_exc}")
    append_error(cv_record["cv_id"], jd_record["id"], str(last_exc))
    return None


# ---------------------------------------------------------------------------
# Heuristic Tier C
# ---------------------------------------------------------------------------
HEURISTIC_RANGES = {
    "skills": (5, 15),
    "title": (5, 15),
    "experience": (10, 25),
    "education": (20, 40),
    "industry": (5, 15),
    "location": (75, 75),
    "salary": (75, 75),
}


def make_heuristic_pair(rng, cv_record, jd_record):
    """Deterministically generate a disjoint-domain negative pair (0 API calls)."""
    values = {k: rng.randint(lo, hi) for k, (lo, hi) in HEURISTIC_RANGES.items()}
    scores = ATSDimensionalScores(**values)
    overall = scores.compute_overall()
    return SilverMatchPair(
        cv_id=cv_record["cv_id"],
        jd_id=jd_record["id"],
        cv_domain=cv_record["domain"],
        jd_domain=jd_record["category"],
        pair_type="heuristic_negative",
        grade=compute_grade(overall, scores.skills, scores.title),  # always 0 for disjoint
        overall_score=overall,
        scores=scores,
        rationale=f"Complete domain mismatch: {cv_record['domain']} candidate has no relevant "
                  f"skills for {jd_record['category']} position.",
        missing_skills=[],
        matched_skills=[],
    )


# ---------------------------------------------------------------------------
# Pair generation (no vector DB — static domain mapping)
# ---------------------------------------------------------------------------
def build_pools(jobs):
    pools = defaultdict(list)
    for jd in jobs:
        if jd["category"] in EXCLUDED_CATEGORIES:
            continue
        pools[jd["category"]].append(jd)
    for cat in pools:
        pools[cat].sort(key=lambda j: j["id"])
    return pools


def generate_llm_pairs(cvs, pools, rng):
    """Tier A + Tier B. Returns list of (cv_record, jd_record, pair_type)."""
    pairs = []
    for cv in cvs:
        dom = cv["domain"]
        cfg = DOMAIN_MAP[dom]
        # Tier A: in-domain
        primary_pool = pools[cfg["primary"]]
        if len(primary_pool) < TIER_A_PER_CV:
            print(f"  WARNING: primary pool for {dom} has {len(primary_pool)} JDs (< {TIER_A_PER_CV})")
        for jd in rng.sample(primary_pool, min(TIER_A_PER_CV, len(primary_pool))):
            pairs.append((cv, jd, "in_domain"))
        # Tier B: cross-domain hard negatives (union of cross-over pools)
        cross_pool = []
        for cat in cfg["cross"]:
            cross_pool.extend(pools[cat])
        for jd in rng.sample(cross_pool, min(TIER_B_PER_CV, len(cross_pool))):
            pairs.append((cv, jd, "cross_domain"))
    return pairs


def generate_heuristic_pairs(cvs, pools, rng):
    """Tier C: disjoint-domain negatives. Returns list of SilverMatchPair."""
    pairs = []
    for cv in cvs:
        cfg = DOMAIN_MAP[cv["domain"]]
        disjoint_pool = []
        for cat in cfg["disjoint"]:
            disjoint_pool.extend(pools[cat])
        for jd in rng.sample(disjoint_pool, min(TIER_C_PER_CV, len(disjoint_pool))):
            pairs.append(make_heuristic_pair(rng, cv, jd))
    return pairs


# ---------------------------------------------------------------------------
# Cache / persistence
# ---------------------------------------------------------------------------
def load_cached_pairs():
    """Load all completed LLM pairs from the incremental cache (resume support).

    Used both to skip already-evaluated pairs AND to reload them into the
    output, so a resumed run never loses previously completed annotations.
    """
    pairs = {}
    if CACHE_FILE.exists():
        for line in CACHE_FILE.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
                p = SilverMatchPair.model_validate(rec)
                pairs[f"{p.cv_id}|{p.jd_id}"] = p
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    return pairs


def append_cache(pair: SilverMatchPair):
    with open(CACHE_FILE, "a", encoding="utf-8") as fp:
        fp.write(pair.model_dump_json() + "\n")


def append_error(cv_id, jd_id, message):
    with open(ERROR_FILE, "a", encoding="utf-8") as fp:
        fp.write(json.dumps({"cv_id": cv_id, "jd_id": jd_id, "error": message,
                             "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
async def run_llm_batch(client, llm_pairs, concurrency=10, use_cache=True):
    """Evaluate LLM pairs, skipping cached ones. Returns (results, failures, cached_skipped)."""
    cached_pairs = load_cached_pairs() if use_cache else {}
    results = {}
    failures = []
    cached_skipped = 0
    sem = asyncio.Semaphore(concurrency)

    async def process(cv, jd, pair_type):
        nonlocal cached_skipped
        key = f"{cv['cv_id']}|{jd['id']}"
        if key in cached_pairs:
            cached_skipped += 1
            results[key] = cached_pairs[key]  # reload, never lose completed pairs
            return
        async with sem:
            await asyncio.sleep(0.1)  # pacing: ~10 rps ceiling
            pair = await judge_pair(client, cv, jd)
        if pair is None:
            failures.append((cv["cv_id"], jd["id"]))
            return
        pair.pair_type = pair_type
        if use_cache:
            append_cache(pair)
        results[key] = pair

    await asyncio.gather(*[process(cv, jd, pt) for cv, jd, pt in llm_pairs])
    return results, failures, cached_skipped


def write_outputs(llm_results, heuristic_pairs, failures, cached_skipped):
    records = list(llm_results.values()) + heuristic_pairs
    records.sort(key=lambda p: (p.cv_id, p.jd_id))
    with open(BENCHMARK_FILE, "w", encoding="utf-8") as fp:
        json.dump([r.model_dump() for r in records], fp, ensure_ascii=False, indent=2)

    stats = build_stats(records, failures, cached_skipped, len(llm_results))
    with open(STATS_FILE, "w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    return stats


def build_stats(records, failures, cached_skipped, llm_count):
    grades = Counter(r.grade for r in records)
    by_domain = defaultdict(Counter)
    for r in records:
        by_domain[r.cv_domain][r.grade] += 1
    pair_types = Counter(r.pair_type for r in records)
    buckets = Counter()
    for r in records:
        buckets[_bucket(r.overall_score)] += 1
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_pairs": len(records),
        "llm_evaluated": llm_count,
        "heuristic_pairs": pair_types.get("heuristic_negative", 0),
        "cached_skipped_on_resume": cached_skipped,
        "llm_failures": len(failures),
        "pair_types": dict(pair_types),
        "grade_distribution": dict(grades),
        "grade_by_domain": {d: dict(c) for d, c in sorted(by_domain.items())},
        "score_buckets": dict(sorted(buckets.items())),
        "matched_skill_total": sum(len(r.matched_skills) for r in records if r.pair_type != "heuristic_negative"),
        "missing_skill_total": sum(len(r.missing_skills) for r in records if r.pair_type != "heuristic_negative"),
    }


def _bucket(score):
    if score >= 75:
        return "75-100"
    if score >= 50:
        return "50-74"
    if score >= 25:
        return "25-49"
    return "0-24"


def load_inputs():
    with open(CV_INDEX, encoding="utf-8") as fp:
        cvs = json.load(fp)
    with open(JD_INDEX, encoding="utf-8") as fp:
        jobs = json.load(fp)
    cvs.sort(key=lambda c: c["cv_id"])
    for jd in jobs:  # precompute cleaned judge text once (in-memory only)
        jd["_jd_prompt_text"] = _jd_prompt_text(jd)
    return cvs, jobs


def get_client():
    if not NOVA_KEY_FILE.exists():
        print(f"FATAL: Nova Gateway key file not found: {NOVA_KEY_FILE}")
        sys.exit(1)
    key_text = NOVA_KEY_FILE.read_text(encoding="utf-8").strip()
    if "=" in key_text:  # tolerate KEY=value format
        key_text = key_text.split("=", 1)[1].strip()
    from openai import OpenAI

    return OpenAI(api_key=key_text, base_url=NOVA_BASE_URL, timeout=120)


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------
def dry_run():
    cvs, jobs = load_inputs()
    pools = build_pools(jobs)
    cv_by_domain = defaultdict(list)
    for cv in cvs:
        cv_by_domain[cv["domain"]].append(cv)
    client = get_client()

    print("=" * 80)
    print(f"DRY-RUN: {len(DRY_RUN_PAIRS)} curated pairs — LLM judge via {JUDGE_MODEL}")
    print("=" * 80)

    selected = []
    for cv_dom, jd_cat in DRY_RUN_PAIRS:
        cv = cv_by_domain[cv_dom][0]
        jd = pools[jd_cat][0]
        pair_type = ("in_domain" if jd_cat == DOMAIN_MAP[cv_dom]["primary"]
                     else "cross_domain" if jd_cat in DOMAIN_MAP[cv_dom]["cross"]
                     else "heuristic_negative")
        selected.append((cv, jd, pair_type))

    results = asyncio.run(run_llm_batch(client, selected, concurrency=4, use_cache=False))
    llm_results, failures, _ = results

    print("\n  SUMMARY — 7-dim ATS + deterministic overall/grade:")
    header = f"  {'cv_id':8s} {'jd_id':8s} {'pair_type':18s} {'overall':>8s} {'grade':>5s}  dimensions"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cv, jd, pair_type in selected:
        key = f"{cv['cv_id']}|{jd['id']}"
        pair = llm_results.get(key)
        if pair is None:
            print(f"  {cv['cv_id']:8s} {jd['id']:8s} {pair_type:18s} {'FAILED':>8s}")
            continue
        s = pair.scores
        print(f"  {pair.cv_id:8s} {pair.jd_id:8s} {pair.pair_type:18s} {pair.overall_score:8.1f} {pair.grade:5d}  "
              f"sk={s.skills} ti={s.title} ex={s.experience} ed={s.education} in={s.industry} lo={s.location} sa={s.salary}")
        print(f"      rationale: {pair.rationale[:160]}")
        if pair.matched_skills:
            print(f"      matched : {', '.join(pair.matched_skills[:10])}")
        if pair.missing_skills:
            print(f"      missing : {', '.join(pair.missing_skills[:10])}")

    if failures:
        print(f"\n  !! {len(failures)} judge failures: {failures}")
    print("\n  Dry-run wrote nothing to cache or outputs. Inspect the 7 subscores,")
    print("  then run with --full to generate the complete benchmark.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    full = "--full" in args
    heuristic_only = "--heuristic-only" in args
    limit = None
    concurrency = 10
    seed = 42

    for i, a in enumerate(args):
        if a == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1])
        elif a == "--concurrency" and i + 1 < len(args):
            concurrency = int(args[i + 1])
        elif a == "--seed" and i + 1 < len(args):
            seed = int(args[i + 1])

    if not full:
        dry_run()
        return

    cvs, jobs = load_inputs()
    rng = random.Random(seed)

    print("=" * 80)
    print(f"FULL SILVER ANNOTATION RUN  (seed={seed}, concurrency={concurrency}, limit={limit})")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pools = build_pools(jobs)

    heuristic_pairs = generate_heuristic_pairs(cvs, pools, random.Random(seed + 1))
    print(f"\n[Tier C] Heuristic disjoint negatives: {len(heuristic_pairs)} pairs (0 API calls)")

    if heuristic_only:
        print("  --heuristic-only: skipping LLM tiers.")
        llm_results, failures, cached_skipped = {}, [], 0
    else:
        llm_pairs = generate_llm_pairs(cvs, pools, rng)
        if limit is not None:
            llm_pairs = llm_pairs[:limit]
        print(f"[Tier A+B] LLM-evaluated pairs: {len(llm_pairs)}")
        client = get_client()
        llm_results, failures, cached_skipped = asyncio.run(
            run_llm_batch(client, llm_pairs, concurrency=concurrency)
        )
        print(f"  Evaluated: {len(llm_results)}  Failed: {len(failures)}  Cached-skipped: {cached_skipped}")

    stats = write_outputs(llm_results, heuristic_pairs, failures, cached_skipped)
    print("\n" + "=" * 80)
    print(f"OUTPUTS")
    print(f"  {BENCHMARK_FILE}")
    print(f"  {STATS_FILE}")
    print(f"  total pairs: {stats['total_pairs']}")
    print(f"  grades: {stats['grade_distribution']}")
    print(f"  score buckets: {stats['score_buckets']}")


if __name__ == "__main__":
    main()