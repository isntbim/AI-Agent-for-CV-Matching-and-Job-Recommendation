# Phase 3 verification: schema validation + distribution + resume math (no API)
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, ".")
import scripts.generate_silver_matching_pairs as g
from src.parser.match_schemas import SilverMatchPair

print("=" * 70)
print("PHASE 3 VERIFICATION — benchmark_matching_pairs.json")
print("=" * 70)

# 1) 100% schema validation
with open(g.BENCHMARK_FILE, encoding="utf-8") as f:
    records = json.load(f)
print(f"\n[1] Records in benchmark: {len(records)}")

errors = 0
for r in records:
    try:
        SilverMatchPair.model_validate(r)
    except Exception as e:
        errors += 1
        print(f"  INVALID: {r.get('cv_id')} x {r.get('jd_id')}: {e}")
print(f"    schema-valid: {len(records) - errors}/{len(records)} "
      f"({'100% PASS' if errors == 0 else 'FAIL'})")

# 2) Distribution checks (plan Phase 3: balanced grades + all domains per grade)
grades = Counter(r["grade"] for r in records)
print(f"\n[2] Grade distribution: {dict(grades)}")
grade_by_domain = defaultdict(lambda: Counter())
for r in records:
    grade_by_domain[r["cv_domain"]][r["grade"]] += 1
print("    grade x domain (rows = cv domain):")
for d in sorted(grade_by_domain):
    print(f"      {d:24s} " + ", ".join(f"g{g}={grade_by_domain[d][g]}" for g in (2, 1, 0)))

pt = Counter(r["pair_type"] for r in records)
print(f"    pair types: {dict(pt)}")

buckets = Counter()
for r in records:
    s = r["overall_score"]
    b = "75-100" if s >= 75 else "50-74" if s >= 50 else "25-49" if s >= 25 else "0-24"
    buckets[b] += 1
print(f"    score buckets: {dict(sorted(buckets.items()))}")

# 3) Resume math: how many of the 1,144 LLM pairs are still missing?
cvs, jobs = g.load_inputs()
pools = g.build_pools(jobs)
import random
llm_pairs = g.generate_llm_pairs(cvs, pools, random.Random(42))
cached = g.load_cached_pairs()
missing = [(c["cv_id"], j["id"]) for c, j, _ in llm_pairs if f"{c['cv_id']}|{j['id']}" not in cached]
print(f"\n[3] Resume plan: {len(llm_pairs)} total LLM pairs, {len(cached)} cached, "
      f"{len(missing)} still to evaluate (re-run --full will only do these)")

# 4) Error log check
err_lines = 0
if g.ERROR_FILE.exists():
    err_lines = sum(1 for _ in open(g.ERROR_FILE, encoding="utf-8"))
print(f"    .cache_errors.jsonl lines: {err_lines} (judge failures tracked here)")

# 5) Spot check a few records from each grade
print(f"\n[4] Spot-check (one per grade):")
for gd in (2, 1, 0):
    sample = next((r for r in records if r["grade"] == gd), None)
    if sample:
        print(f"  g{gd}: {sample['cv_id']} x {sample['jd_id']} "
              f"(overall {sample['overall_score']}, missing={sample['missing_skills'][:4]})")