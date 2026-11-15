"""O5 SFT data for the second QLoRA adapter (explanations), in the same JSONL format as the extraction sets.

train / validation: facts -> the teacher's explanation, only explanations that pass scripts.o5_faithfulness.
test: prompts only (id, kind, system, user) for training/scripts/generate.py --prompts; scored locally afterwards with
`python -m scripts.o5_faithfulness` on the generated text, for the base model and for the adapter.

Vietnamese (user decision 2026-11-10): a subset of train/val pairs also has a Vietnamese teacher explanation in
output/o5/teacher_vi/; those rows get the id "<pair_id>_vi" and SYSTEM_VI. Every test pair gets both prompts.

    python -m scripts.o5_export_sft     # -> data/datasets/o5_explanation/
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.o5_faithfulness import O5, VERDICTS_VI, check, unscored  # export filter unchanged: v3 data stays as trained

OUT = Path(__file__).resolve().parents[1] / "data/datasets/o5_explanation"
# v2 (2026-11-11): the facts go in as plain statements (render()), not raw JSON. The v1 adapter misread the JSON:
# unscored dimensions carried a placeholder score of 50, so it claimed "relevant degree", "same field" etc. for things
# that were never checked (Opus review: 36% EN / 25% VI wrong). The teacher explanations are unchanged.
SYSTEM = ("You explain one CV-job match result to the job seeker, using only the facts given. Write 3 to 5 sentences "
          "(90-120 words) in plain English, addressed to the candidate (\"your CV\"). Start with the verdict and fit, e.g. "
          "\"Potential Fit (fit 47/100): ...\", using the verdict and fit exactly as given. Then the strongest evidence, the "
          "main gaps, and one practical next step built only from the gaps; at most 3 items in any list. Describe each "
          "CHECKED result only as the facts state it (met, partly met, not met, same or different field). For NOT CHECKED "
          "items, you may say that one of them could not be checked, but never say it matches, is relevant, close, above, "
          "below, met or missing. Never mention location or salary. Only use numbers that appear in the facts. Only name "
          "skills, titles, degrees and dates that appear in the facts. Never invent anything.")
SYSTEM_VI = ("Bạn giải thích một kết quả so khớp CV với công việc cho người tìm việc, chỉ dùng các dữ kiện được cho. Viết "
             "bằng tiếng Việt, 3 đến 5 câu (100-160 tiếng), xưng \"bạn\" với ứng viên. Mở đầu bằng kết luận và độ phù hợp, "
             "ví dụ \"Có tiềm năng (độ phù hợp 47/100): ...\", dùng đúng nhãn: "
             + ", ".join(f"{en} = \"{vi}\"" for en, vi in VERDICTS_VI.items()) + ", và đúng con số độ phù hợp. Sau đó nêu "
             "bằng chứng mạnh nhất, các thiếu sót chính và một bước tiếp theo cụ thể dựa trên thiếu sót; mỗi danh sách tối "
             "đa 3 mục. Mục ĐÃ ĐỐI CHIẾU (CHECKED) chỉ được mô tả đúng như dữ kiện nêu. Mục KHÔNG ĐỐI CHIẾU (NOT CHECKED) "
             "chỉ được nói là không thể đối chiếu, không bao giờ nói là phù hợp, liên quan, gần với, cao hơn, thấp hơn, đáp "
             "ứng hay thiếu. Không nhắc địa điểm hay lương. Chỉ dùng các con số có trong dữ kiện. Giữ nguyên tên kỹ năng, "
             "chức danh, bằng cấp và ngày tháng như trong dữ kiện (không dịch). Không bịa bất cứ điều gì.")


def _join(items) -> str:
    return "; ".join(items) if items else "none"


def render(f: dict) -> str:
    """The facts as plain statements: every CHECKED dimension states its result in words, every NOT CHECKED one only its
    reason (no placeholder score). Same text for the English and Vietnamese prompts; the service must use this too."""
    dims, why = f["dimensions"], f.get("not_scored_because", {})
    years, degree = f.get("job_min_years"), f.get("job_min_degree")
    need_years = f"the {years}-year experience requirement" if years else "the experience requirement"
    need_degree = f"the {degree} degree requirement" if degree else "the degree requirement"

    def checked(name: str, s: int) -> str:
        if name == "title":
            return ("the CV's job titles match this job title" if s == 100 else
                    "the CV's job titles do not match this job title" if s == 0 else "the CV's job titles partly match this job title")
        if name == "experience":
            if s == 100:
                return ("met: the job asks for no prior experience" if f.get("experience_basis") == "explicit_no_experience_requirement"
                        else f"met: the CV meets {need_years}")
            return f"not met: the CV does not meet {need_years}" if s == 0 else f"partly met: the CV does not fully meet {need_years}"
        if name == "education":
            return (f"met: the CV meets {need_degree}" if s == 100 else
                    f"not met: the CV does not meet {need_degree}" if s == 0 else f"partly met: the CV only partly meets {need_degree}")
        if name == "industry":
            return ("the CV's work is in the same field as this job" if s == 100 else
                    "the CV's work is in a different field from this job" if s == 0 else "the CV's work is in a related field")
        return ""

    lines = [f"Verdict: {f['verdict']}. Fit: {f['fit']}/100. Rule score: {f['rule_score']}/100.",
             f"Job: {f.get('job_title')}. Job level: {f.get('job_level') or 'not stated'}. "
             f"Minimum years asked: {years if years else 'none stated'}. Minimum degree asked: {degree or 'none stated'}.",
             f"CV latest title: {f.get('cv_title')}. Recent roles: "
             + _join(f"{r['title']} ({r['dates']})" if r.get("dates") else r["title"] for r in f.get("cv_recent_roles", []))
             + ". Education listed: " + _join(f.get("cv_education", [])) + "."]
    if f.get("vague_cv_titles"):
        lines.append("Vague one-word CV titles (not job titles): " + _join(f["vague_cv_titles"]) + ".")

    lines += ["", "CHECKED:"]
    if dims["skills"]["scored"]:
        lines.append(f"- Skills: {dims['skills']['score']}/100.")
    for name, label in (("title", "Job title"), ("experience", "Experience"), ("education", "Education"), ("industry", "Industry")):
        if dims[name]["scored"]:
            lines.append(f"- {label}: {dims[name]['score']}/100, {checked(name, dims[name]['score'])}.")
    lines += ["", "NOT CHECKED (no score; say at most that one of them could not be checked):"]
    for name, label in (("skills", "Skills"), ("title", "Job title"), ("experience", "Experience"), ("education", "Education"),
                        ("industry", "Industry")):
        if not dims[name]["scored"]:
            lines.append(f"- {label}: {why.get(name, 'not enough evidence')}.")
    lines.append("- Location and salary: never assessed, do not mention them.")

    reqs = f.get("requirements", [])
    by = lambda st: [r["requirement"] + (f" (your skill: {r['your_skill']})" if r.get("your_skill") else "")
                     for r in reqs if r["status"] == st]
    lines += ["", f"SKILLS ({len(f['matched_skills'])} matched, {len(f['missing_skills'])} missing):",
              "- Matched on the CV: " + _join(f["matched_skills"]) + ".",
              "- Missing from the CV: " + _join(f["missing_skills"]) + ".",
              f"- Listed job requirements ({len(reqs)} shown, the job may have more): met: {_join(by('met'))}. "
              f"Partly met: {_join(by('partly met'))}. Not met: {_join(by('not met'))}.",
              "- Nice-to-have matched: " + _join(f.get("nice_to_have_matched", [])) + ". Nice-to-have missing: "
              + _join(f.get("nice_to_have_missing", [])) + "."]
    if f.get("found_in_cv_text"):
        lines.append("- Mentioned in the CV text but not as a listed skill: " + _join(f["found_in_cv_text"]) + ".")
    if f.get("soft_skills_not_scored"):
        lines.append("- Soft skills the job asks for (not scored): " + _join(f["soft_skills_not_scored"]) + ".")
    return "\n".join(lines)


_DIM_EN = {"title": ("Your job titles match this job title.", "Your job titles partly match this job title.",
                     "Your job titles do not match this job title."),
           "experience": ("Your CV meets the experience requirement.", "Your CV only partly meets the experience requirement.",
                          "Your CV does not meet the experience requirement."),
           "education": ("Your CV meets the degree requirement.", "Your CV only partly meets the degree requirement.",
                         "Your CV does not meet the degree requirement."),
           "industry": ("Your work is in the same field as this job.", "Your work is in a related field.",
                        "Your work is in a different field from this job.")}
_DIM_VI = {"title": ("Chức danh trong CV khớp với chức danh của vị trí này.", "Chức danh trong CV khớp một phần với chức danh của vị trí này.",
                     "Chức danh trong CV không khớp với chức danh của vị trí này."),
           "experience": ("CV đáp ứng yêu cầu kinh nghiệm.", "CV chỉ đáp ứng một phần yêu cầu kinh nghiệm.", "CV chưa đáp ứng yêu cầu kinh nghiệm."),
           "education": ("CV đáp ứng yêu cầu bằng cấp.", "CV chỉ đáp ứng một phần yêu cầu bằng cấp.", "CV chưa đáp ứng yêu cầu bằng cấp."),
           "industry": ("Công việc của bạn cùng lĩnh vực với vị trí này.", "Công việc của bạn thuộc lĩnh vực gần với vị trí này.",
                        "Công việc của bạn khác lĩnh vực với vị trí này.")}
_UNCHECKED = {"skills": ("The job's skill requirements", "Các yêu cầu kỹ năng của vị trí này"),
              "title": ("Your job titles", "Chức danh của bạn"), "experience": ("Your years of experience", "Số năm kinh nghiệm của bạn"),
              "education": ("Your education", "Học vấn của bạn"), "industry": ("Your field of work", "Lĩnh vực làm việc của bạn")}


def fallback(f: dict, vi: bool = False) -> str:
    """The explanation the service shows when the adapter's text fails scripts.o5_faithfulness.violations twice: fixed
    sentences filled only from the facts, so it cannot misstate them. Plain on purpose (no advice beyond the first gap)."""
    dims, has, gaps = f["dimensions"], f["matched_skills"][:3], f["missing_skills"][:3]
    unchecked = [n for n in ("skills", "title", "experience", "education", "industry") if not dims[n]["scored"]]
    could_not = " không thể đối chiếu." if vi else " could not be checked."
    opening = f"{VERDICTS_VI[f['verdict']]} (độ phù hợp {f['fit']}/100): " if vi else f"{f['verdict']} (fit {f['fit']}/100): "
    if has:
        lines = [opening + (f"CV của bạn có {', '.join(has)}." if vi else f"your CV shows {', '.join(has)}.")]
    elif gaps:
        lines = [opening + ("CV của bạn chưa thể hiện kỹ năng nào trong các yêu cầu được liệt kê." if vi else
                            "your CV does not show any of the listed required skills.")]
    else:  # no skill on either side (skills not scored): saying "shows none" would be a claim the facts do not make
        first = unchecked.pop(0) if unchecked else None
        lines = [opening + ((_UNCHECKED[first][vi][0].lower() + _UNCHECKED[first][vi][1:] + could_not) if first else
                            ("đây là kết quả so khớp tổng thể giữa CV và vị trí này." if vi else
                             "this is the overall result of comparing your CV with this job."))]
        unchecked = []  # at most one unchecked item, and it is already said
    if gaps:
        lines.append(f"CV chưa thể hiện {', '.join(gaps)}." if vi else f"It does not show {', '.join(gaps)}.")
    for name in ("title", "experience", "education", "industry"):
        if dims[name]["scored"]:
            s = dims[name]["score"]
            lines.append((_DIM_VI if vi else _DIM_EN)[name][0 if s == 100 else 2 if s == 0 else 1])
    if unchecked:  # at most one, as the prompt asks
        lines.append(_UNCHECKED[unchecked[0]][vi] + could_not)
    if gaps:
        lines.append(f"Nếu bạn đã dùng {gaps[0]}, hãy bổ sung một ví dụ cụ thể vào CV trước khi ứng tuyển." if vi else
                     f"If you have used {gaps[0]}, add a clear example to your CV before applying.")
    return " ".join(lines)


def facts_of(pid: str) -> dict:
    return json.loads((O5 / "inputs" / f"{pid}.json").read_text(encoding="utf-8"))["facts"]


def main() -> None:
    pairs = json.loads((O5 / "pairs.json").read_text(encoding="utf-8"))
    (OUT / "explanation").mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, name in (("train", "train"), ("val", "validation")):
        rows = []
        for p in (p for p in pairs if p["split"] == split):
            for folder, suffix, system in (("teacher", "", SYSTEM), ("teacher_vi", "_vi", SYSTEM_VI)):
                f = O5 / folder / f"{p['pair_id']}.json"
                if not f.exists():
                    continue
                expl, facts = json.loads(f.read_text(encoding="utf-8"))["explanation"], facts_of(p["pair_id"])
                if check(expl, facts) or unscored(expl, facts):
                    continue  # never train on an unfaithful target, nor one that uses an unscored dimension (v3)
                rows.append({"id": p["pair_id"] + suffix, "split": name, "source": folder, "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": render(facts)},
                    {"role": "assistant", "content": expl}]})
        (OUT / "explanation" / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        counts[name] = len(rows)
    # dev_prompts = the validation pairs in both languages: the checker's rules are written on these outputs, so the
    # test prompts stay unseen until the final measurement
    for split, name in (("test", "test_prompts"), ("val", "dev_prompts")):
        rows = [{"id": p["pair_id"] + suffix, "kind": "explanation", "system": system, "user": render(facts_of(p["pair_id"]))}
                for suffix, system in (("", SYSTEM), ("_vi", SYSTEM_VI)) for p in pairs if p["split"] == split]
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        counts[name] = len(rows)
    (OUT / "stats.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print(counts)


if __name__ == "__main__":
    main()
