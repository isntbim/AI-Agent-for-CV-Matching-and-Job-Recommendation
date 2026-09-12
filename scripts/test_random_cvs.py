"""
test_random_cvs.py — Test parsing 6 diverse CVs across all categories.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import random
import time
import json
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.pipeline import parse_cv
from src.parser.llm_extractor import OllamaClient

def main():
    cvs_root = Path('data/raw/cvs')
    categories = sorted([d for d in cvs_root.iterdir() if d.is_dir()])

    # Pick 1 random CV from each category (6 categories total)
    selected_cvs = []
    random.seed(42)  # For reproducible sampling
    for cat in categories:
        pdf_files = sorted(list(cat.glob('*.pdf')))
        if pdf_files:
            chosen = random.choice(pdf_files)
            selected_cvs.append((cat.name, chosen))

    print(f"=== Selected {len(selected_cvs)} CVs across categories ===")
    for cat_name, cv_file in selected_cvs:
        print(f"  * [{cat_name}]: {cv_file.name}")

    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5:7b-instruct")

    results = []
    for idx, (cat_name, cv_file) in enumerate(selected_cvs, 1):
        print("\n" + "=" * 65)
        print(f"[{idx}/{len(selected_cvs)}] Processing [{cat_name}] -> {cv_file.name}")
        print("=" * 65)
        t0 = time.time()
        try:
            resume = parse_cv(cv_file, llm_client=client)
            dur = time.time() - t0
            print(f"-> STATUS: SUCCESS ({dur:.1f}s)")
            print(f"   Name:        {resume.basics.name}")
            print(f"   Role/Label:  {resume.basics.label}")
            print(f"   Email:       {resume.basics.email}")
            print(f"   Phone:       {resume.basics.phone}")
            print(f"   Website/URL: {resume.basics.url}")
            loc_str = resume.basics.location.city if resume.basics.location else None
            print(f"   Location:    {loc_str}")
            print(f"   Work:        {len(resume.work)} jobs")
            if resume.work:
                w0 = resume.work[0]
                print(f"     Latest:    {w0.position} at {w0.name} ({w0.start_date} -> {w0.end_date})")
            print(f"   Education:   {len(resume.education)} degrees")
            if resume.education:
                e0 = resume.education[0]
                print(f"     Degree:    {e0.study_type} in {e0.area} from {e0.institution} (Grad: {e0.end_date})")
            print(f"   Skills:      {len(resume.skills)} domains, {len(resume.get_flat_skills())} flat skills")
            domain_names = [s.name for s in resume.skills]
            print(f"     Domains:   {domain_names}")
            print(f"     Samples:   {resume.get_flat_skills()[:8]}")
            print(f"   Projects:    {len(resume.projects)} entries")
            print(f"   Awards/Cert: {len(resume.certificates)} entries")
            lang_names = [l.language for l in resume.languages]
            print(f"   Languages:   {lang_names}")

            results.append({
                "category": cat_name,
                "file": cv_file.name,
                "status": "SUCCESS",
                "duration_s": round(dur, 1),
                "name": resume.basics.name,
                "label": resume.basics.label,
                "work_count": len(resume.work),
                "edu_count": len(resume.education),
                "skill_count": len(resume.get_flat_skills()),
                "proj_count": len(resume.projects),
                "cert_count": len(resume.certificates),
            })
        except Exception as e:
            dur = time.time() - t0
            print(f"-> STATUS: FAILED ({dur:.1f}s): {e}")
            results.append({
                "category": cat_name,
                "file": cv_file.name,
                "status": "FAILED",
                "duration_s": round(dur, 1),
                "error": str(e),
            })

    print("\n" + "=" * 80)
    print(f"{'CATEGORY':<24} | {'CANDIDATE NAME':<20} | {'EXP':<4} | {'EDU':<4} | {'SKILLS':<6} | {'TIME'}")
    print("=" * 80)
    for r in results:
        if r["status"] == "SUCCESS":
            cat = r["category"][:22]
            name = (r["name"] or "Unknown")[:18]
            print(f"{cat:<24} | {name:<20} | {r['work_count']:<4} | {r['edu_count']:<4} | {r['skill_count']:<6} | {r['duration_s']}s")
        else:
            cat = r["category"][:22]
            err = r.get("error", "Error")[:35]
            print(f"{cat:<24} | FAILED: {err:<28} | {r['duration_s']}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
