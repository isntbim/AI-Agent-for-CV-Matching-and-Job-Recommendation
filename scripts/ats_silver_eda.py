"""Build reproducible EDA artifacts from the full offline silver-scoring run."""

import argparse
import base64
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.parser.match_schemas import ATS_WEIGHTS
from src.scoring.evaluation import classification_metrics

LABELS = ["Mismatch", "Partial", "Strong"]
DIMS = ["skills", "title", "experience", "education", "industry", "location", "salary"]
COLORS = ["#64748b", "#d97706", "#059669"]


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def correlation(x, y, rank=False):
    if rank:
        x, y = x.rank(method="average"), y.rank(method="average")
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return None
    return finite(x.corr(y))


def distribution(values):
    return {"mean": finite(values.mean()), "std": finite(values.std(ddof=0)),
            "min": finite(values.min()), "q25": finite(values.quantile(.25)),
            "median": finite(values.median()), "q75": finite(values.quantile(.75)), "max": finite(values.max())}


def metrics(frame):
    if frame.empty:
        return {"n": 0}
    silver, runtime = frame.silver_score, frame.runtime_score
    majority = int(frame.silver_grade.mode().iloc[0])
    grades = classification_metrics(frame.silver_grade.tolist(), frame.runtime_grade.tolist())
    per_cv = [correlation(group.silver_score, group.runtime_score, rank=True)
              for _, group in frame.groupby("cv_id") if len(group) >= 3]
    valid = [value for value in per_cv if value is not None]
    return {"n": len(frame), "agreement": grades,
            "majority_label": majority, "majority_accuracy": float((frame.silver_grade == majority).mean()),
            "mean_bias": float((runtime - silver).mean()),
            "mae": float((runtime - silver).abs().mean()),
            "rmse": float(np.sqrt(((runtime - silver) ** 2).mean())),
            "pearson": correlation(silver, runtime), "spearman": correlation(silver, runtime, rank=True),
            "silver_distribution": distribution(silver), "runtime_distribution": distribution(runtime),
            "mean_coverage": float(frame.coverage.mean()),
            "per_cv_spearman": {"eligible_cv_count": len(per_cv), "nonconstant_cv_count": len(valid),
                                "median": finite(np.median(valid)) if valid else None,
                                "positive_fraction": finite(np.mean(np.array(valid) > 0)) if valid else None},
            "by_grade": {str(grade): {"n": len(group), "silver_score": distribution(group.silver_score),
                                      "runtime_score": distribution(group.runtime_score), "coverage": float(group.coverage.mean())}
                         for grade, group in frame.groupby("silver_grade")}}


def table(frame):
    return frame.to_markdown(index=False, floatfmt=".3f")


def fmt(value, pattern=".3f"):
    return "undefined (constant scores)" if value is None else format(value, pattern)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "output/ats/silver_full")
    args = parser.parse_args()
    source = json.loads((args.input / "report.json").read_text(encoding="utf-8"))
    rows = json.loads((args.input / "results.json").read_text(encoding="utf-8"))
    if not source.get("full_dataset") or source["failures"] or len(rows) != source["population"]:
        raise ValueError("EDA requires a complete successful full-dataset run")
    flattened = []
    for row in rows:
        result = row["result"]
        values = {key: row[key] for key in ("cv_id", "jd_id", "pair_type", "cv_domain", "jd_domain", "cv_title", "jd_title")}
        values.update(silver_grade=row["silver_grade"], runtime_grade=result["grade"],
                      silver_score=row["silver_overall_score"], runtime_score=result["overall_score"],
                      coverage=result["coverage"], removed_skill_tags="; ".join(row["removed_title_skill_tags"]),
                      matched_skills="; ".join(result["matched_skills"]), missing_skills="; ".join(result["missing_skills"]))
        for dim in DIMS:
            values[f"silver_{dim}"] = row["silver_scores"][dim]
            values[f"runtime_{dim}"] = result["scores"][dim]
            values[f"supported_{dim}"] = result["dimensions"][dim]["supported"]
            values[f"basis_{dim}"] = result["dimensions"][dim]["basis"]
        flattened.append(values)
    frame = pd.DataFrame(flattened)
    frame["source"] = np.where(frame.pair_type == "heuristic_negative", "Heuristic", "Gemini")
    frame["delta"] = frame.runtime_score - frame.silver_score
    frame["absolute_delta"] = frame.delta.abs()
    teacher = frame[frame.source == "Gemini"]
    heuristic = frame[frame.source == "Heuristic"]
    groups = {"All pairs": frame, "Gemini only": teacher, "Heuristic only": heuristic}
    summaries = {name: metrics(group) for name, group in groups.items()}
    comparison = pd.DataFrame([{"Source": name, "Pairs": len(group),
                                "Agreement %": 100 * summaries[name]["agreement"]["accuracy"],
                                "Majority %": 100 * summaries[name]["majority_accuracy"],
                                "Macro F1": summaries[name]["agreement"]["macro_f1"],
                                "MAE": summaries[name]["mae"], "Bias": summaries[name]["mean_bias"],
                                "Pearson": summaries[name]["pearson"], "Spearman": summaries[name]["spearman"]}
                               for name, group in groups.items()])
    dimensions = []
    for dim in DIMS:
        difference = teacher[f"runtime_{dim}"] - teacher[f"silver_{dim}"]
        supported = teacher[f"supported_{dim}"]
        supported_diff = difference[supported]
        dimensions.append({"Dimension": dim, "Weight": ATS_WEIGHTS[dim],
                           "Gemini mean": teacher[f"silver_{dim}"].mean(),
                           "Runtime mean": teacher[f"runtime_{dim}"].mean(),
                           "Bias": difference.mean(), "MAE": difference.abs().mean(),
                           "Weighted bias": difference.mean() * ATS_WEIGHTS[dim],
                           "Supported %": 100 * supported.mean(), "Supported N": int(supported.sum()),
                           "Supported-only MAE": finite(supported_diff.abs().mean()),
                           "Neutral fallback %": 100 * (~supported).mean(),
                           "Runtime zero %": 100 * (teacher[f"runtime_{dim}"] == 0).mean()})
    dim_frame = pd.DataFrame(dimensions)
    by_grade = pd.DataFrame([{"Silver grade": LABELS[grade], "Pairs": len(group),
                             "Gemini mean": group.silver_score.mean(), "Runtime mean": group.runtime_score.mean(),
                             "Runtime median": group.runtime_score.median(), "Runtime std": group.runtime_score.std(ddof=0),
                             "MAE": group.absolute_delta.mean(), "Agreement %": 100 * (group.silver_grade == group.runtime_grade).mean()}
                            for grade, group in teacher.groupby("silver_grade")])
    domain_metrics = {domain: metrics(group) for domain, group in teacher.groupby("cv_domain")}
    domain_frame = pd.DataFrame([{"CV domain": domain, "Pairs": stat["n"], "Agreement %": 100 * stat["agreement"]["accuracy"],
                                 "Majority %": 100 * stat["majority_accuracy"], "MAE": stat["mae"],
                                 "Spearman": stat["spearman"], "Coverage %": 100 * stat["mean_coverage"]}
                                for domain, stat in domain_metrics.items()])
    type_metrics = {kind: metrics(group) for kind, group in frame.groupby("pair_type")}
    type_frame = pd.DataFrame([{"Pair type": kind, "Pairs": stat["n"], "Agreement %": 100 * stat["agreement"]["accuracy"],
                               "Silver mean": stat["silver_distribution"]["mean"], "Runtime mean": stat["runtime_distribution"]["mean"],
                               "MAE": stat["mae"], "Spearman": stat["spearman"]} for kind, stat in type_metrics.items()])
    grade_counts = pd.DataFrame({"Grade": LABELS,
                                 "Silver all": [int((frame.silver_grade == k).sum()) for k in range(3)],
                                 "Runtime all": [int((frame.runtime_grade == k).sum()) for k in range(3)],
                                 "Silver Gemini": [int((teacher.silver_grade == k).sum()) for k in range(3)],
                                 "Runtime Gemini": [int((teacher.runtime_grade == k).sum()) for k in range(3)]})
    basis_counts = {dim: teacher[f"basis_{dim}"].value_counts().to_dict() for dim in DIMS}
    identity = {"scope": source["scope"], "pairs": len(frame), "cvs": int(frame.cv_id.nunique()), "jds": int(frame.jd_id.nunique()),
                "as_of_date": source["as_of_date"], "config": source["config"], "api_requests": 0,
                "score_results_sha256": hashlib.sha256((args.input / "results.json").read_bytes()).hexdigest(),
                "code_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in [ROOT / "scripts/ats_silver_eval.py", ROOT / "scripts/ats_silver_eda.py",
                                             *[ROOT / "src/scoring" / name for name in ("models.py", "normalization.py", "rules.py", "silver.py")]]}}
    artifact = {"run": identity, "groups": summaries, "dimensions_gemini": dim_frame.replace({np.nan: None}).to_dict("records"),
                "by_cv_domain_gemini": domain_metrics, "by_pair_type": type_metrics,
                "basis_counts_gemini": basis_counts, "duplicate_pair_keys": int(frame.duplicated(["cv_id", "jd_id"]).sum()),
                "runtime_distinct_totals": int(frame.runtime_score.nunique()), "silver_distinct_totals": int(frame.silver_score.nunique()),
                "runtime_mode_totals": {str(k): int(v) for k, v in frame.runtime_score.value_counts().head(10).items()},
                "removed_title_tag_pairs": int((frame.removed_skill_tags != "").sum()),
                "zero_skill_pairs_gemini": int((teacher.runtime_skills == 0).sum())}
    out = args.input / "eda"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "pair_comparison.csv", index=False, encoding="utf-8-sig")
    frame[(frame.silver_grade != frame.runtime_grade)].sort_values("absolute_delta", ascending=False).to_csv(out / "disagreements.csv", index=False, encoding="utf-8-sig")
    teacher[teacher.silver_grade == 2].to_csv(out / "silver_strong_pairs.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    fig.suptitle("ATS runtime vs silver scoring — 1,672 pairs\nGemini comparisons use 1,144 pairs; heuristic negatives are separated", fontsize=18, fontweight="bold")
    ax = axes[0, 0]
    x = np.arange(3)
    ax.bar(x - .18, grade_counts["Silver all"], .36, label="Silver", color="#2563eb")
    ax.bar(x + .18, grade_counts["Runtime all"], .36, label="Runtime", color="#e87936")
    ax.set_xticks(x, LABELS)
    ax.set_title("Grade distribution • all pairs")
    ax.set_ylabel("Pairs")
    ax.legend()
    for bars in ax.containers:
        ax.bar_label(bars, padding=3)
    ax = axes[0, 1]
    bins = np.arange(0, 105, 5)
    ax.hist(teacher.silver_score, bins=bins, alpha=.55, label="Gemini", color="#2563eb")
    ax.hist(teacher.runtime_score, bins=bins, alpha=.55, label="Runtime", color="#e87936")
    ax.set_title("Total-score distributions • Gemini pairs")
    ax.set_xlabel("Score / 100")
    ax.set_ylabel("Pairs")
    ax.legend()
    ax = axes[0, 2]
    for name, group in groups.items():
        if name == "All pairs":
            continue
        ax.scatter(group.silver_score, group.runtime_score, s=9, alpha=.35, label=name)
    ax.plot([0, 100], [0, 100], color="#334155", linestyle="--", linewidth=1)
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Silver total", ylabel="Runtime total", title="Paired totals • sources separated")
    ax.legend()
    ax = axes[1, 0]
    matrix = np.array(summaries["Gemini only"]["agreement"]["confusion_matrix"])
    percent = matrix / matrix.sum(axis=1, keepdims=True)
    ax.imshow(percent, vmin=0, vmax=1, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]}\n{percent[i, j]:.1%}", ha="center", va="center", color="white" if percent[i, j] > .5 else "#0f172a")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set(xlabel="Runtime grade", ylabel="Gemini grade", title="Confusion matrix • Gemini pairs")
    ax = axes[1, 1]
    ax.barh(DIMS, dim_frame["Supported %"], color="#0d9488")
    ax.set(xlim=(0, 105), xlabel="Pairs with supported dimension (%)", title="Evidence support • Gemini pairs")
    ax.invert_yaxis()
    for i, val in enumerate(dim_frame["Supported %"]):
        ax.text(val + 1, i, f"{val:.1f}%", va="center")
    ax = axes[1, 2]
    ax.barh(DIMS, dim_frame["Bias"], color=["#e87936" if value < 0 else "#2563eb" for value in dim_frame["Bias"]])
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set(xlabel="Mean runtime minus Gemini score", title="Dimension bias • Gemini pairs")
    ax.invert_yaxis()
    fig.savefig(out / "overview.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    silver_groups = [teacher.loc[teacher.silver_grade == k, "silver_score"] for k in range(3)]
    runtime_groups = [teacher.loc[teacher.silver_grade == k, "runtime_score"] for k in range(3)]
    for ax, values, title in zip(axes, [silver_groups, runtime_groups], ["Gemini totals", "Runtime totals"]):
        parts = ax.boxplot(values, tick_labels=LABELS, patch_artist=True, showfliers=False)
        for box, color in zip(parts["boxes"], COLORS):
            box.set_facecolor(color)
            box.set_alpha(.6)
        ax.set(ylim=(0, 100), ylabel="Score / 100", title=title + " by Gemini grade")
    fig.savefig(out / "score_by_grade.png", dpi=160)
    plt.close(fig)

    main_stat = summaries["Gemini only"]
    worst = teacher.sort_values("absolute_delta", ascending=False).head(12)
    outlier_columns = ["cv_id", "jd_id", "cv_domain", "silver_grade", "runtime_grade", "silver_score", "runtime_score", "delta", "coverage"]
    confusion_table = pd.DataFrame(matrix, columns=LABELS)
    confusion_table.insert(0, "Gemini grade", LABELS)
    lines = ["# EDA: ATS runtime scorer vs silver scoring", "",
             f"Scored **{len(frame):,}/{source['population']:,} pairs**, {identity['cvs']} CVs and {identity['jds']} JDs. **0 errors, 0 API calls**. As-of date: {source['as_of_date']}.", "",
             "## Main findings", "",
             f"- Against **Gemini labels alone**, grade agreement is **{main_stat['agreement']['accuracy']:.1%}**, versus **{main_stat['majority_accuracy']:.1%}** for always predicting Mismatch. Macro F1 is **{main_stat['agreement']['macro_f1']:.3f}**.",
             f"- Total-score MAE is **{main_stat['mae']:.2f}/100**, mean bias **{main_stat['mean_bias']:+.2f}**, Pearson **{fmt(main_stat['pearson'])}**, and Spearman **{fmt(main_stat['spearman'])}** on Gemini pairs.",
             f"- Runtime predicts **{int((frame.runtime_grade == 2).sum())} Strong** pairs. Gemini-labelled Strong recall is **{main_stat['agreement']['per_label']['2']['recall']:.1%}** across {int((teacher.silver_grade == 2).sum())} silver Strong pairs.",
             f"- Title and industry support are **{dim_frame.loc[dim_frame.Dimension == 'title', 'Supported %'].iloc[0]:.1f}%** and **{dim_frame.loc[dim_frame.Dimension == 'industry', 'Supported %'].iloc[0]:.1f}%** on Gemini pairs. Unsupported dimensions use neutral defaults; coverage is supported weight, not extraction accuracy.",
             f"- **{artifact['zero_skill_pairs_gemini']}/{len(teacher)} Gemini pairs** receive a runtime skills score of zero. Skills have the largest weight (30%); this deserves source/alias review before threshold tuning.", "",
             "![Overview](overview.png)", "", "## 1. Label balance and source separation", "", table(grade_counts), "", table(comparison), "",
             "All-pair agreement is inflated by the 528 heuristic negatives. Their subscores were generated from random ranges rather than full factual scoring. Primary teacher agreement and correlations therefore use Gemini pairs only. Macro F1 averages all three fixed classes; the heuristic-only group contains only Mismatch, so its macro F1 is not comparable to the three-class groups.", "",
             "## 2. Continuous scores and label separation", "", table(by_grade), "", "![Score distributions by grade](score_by_grade.png)", "",
             "Means by label are descriptive, not numerical ground truth or a probability of being hired. Spearman measures rank agreement with the silver totals; Pearson measures linear association. Neither establishes hiring validity.", "",
             f"For rankings within the same CV, median Spearman is **{fmt(main_stat['per_cv_spearman']['median'])}** across **{main_stat['per_cv_spearman']['nonconstant_cv_count']}** CVs with at least three Gemini pairs and nonconstant scores. Ties receive average ranks. This is diagnostic agreement within the preselected candidate jobs, not a retrieval benchmark.", "",
             "## 3. Grade confusion: Gemini only", "", table(confusion_table), "",
             "Rows are Gemini grades; columns are runtime grades. Thresholds remain 50/75, with skills and title both at least 70 required for Strong.", "",
             "## 4. Dimension differences and missing evidence", "", table(dim_frame), "",
             "Bias means runtime minus Gemini. Weighted bias is each dimension's contribution to the total difference before total rounding. Supported-only MAE excludes neutral fallbacks and can have a very small sample; null means no supported cases. Salary defaults to negotiable 75 and has no candidate salary evidence.", "",
             "## 5. Pair types and CV domains", "", table(type_frame), "", "### CV domain: Gemini pairs only", "", table(domain_frame), "",
             "CV-domain and JD-category metadata are used for EDA grouping only. They were not supplied as scoring facts. Groups have different class balances and candidate-pair selection, so compare them with their majority baselines and support counts.", "",
             "## 6. Largest Gemini/runtime score differences", "", table(worst[outlier_columns]), "",
             "All 34 silver Strong pairs are exported in `silver_strong_pairs.csv`; every grade disagreement is in `disagreements.csv`. Inspect original structured fields and dimension evidence in the parent `results.json` before changing any rule.", "",
             "## 7. Interpretation and next changes", "",
             "1. Review a fixed set of Strong and Partial disagreements against the CV/JD source facts. Diagnose missing parsed skills separately from genuine skill mismatches.",
             "2. Expand title aliases and handle title qualifiers using explicit rules. The current exact dictionary leaves most title/domain values unknown and also limits relevant-experience scoring.",
             "3. Audit skill synonyms, multilingual/mojibake tags and the title-tag filter. Avoid treating employer sectors or pair-domain labels as proof of skills.",
             "4. Retest on a development partition with CV/JD isolation before calibration. Keep an independent test partition for reporting; do not tune and validate on these same 1,672 pairs.",
             "5. Use independently reviewed labels or observed outcomes when claiming real-world suitability. This comparison identifies disagreement with Gemini and heuristics, not which score is correct.", "",
             "## Method and reproducibility", "",
             "- Weights: skills 30%, title 20%, experience 15%, education 10%, industry 10%, location 8%, salary 7%. No weights, thresholds or mappings were tuned for this run.",
             "- Source evidence is stored structured JSON, not new LLM extraction and not freshly verified original PDF/raw-text quotes. Optional degrees, incomplete degrees and missing facts stay unknown. Remote does not automatically mean worldwide eligibility.",
             "- Present employment uses the fixed as-of date; historical CVs may overstate current tenure. The existing title/seniority skill-tag filter is applied to JDs.",
             "- All pairs share a small set of CVs and JDs. This is an in-dataset agreement audit, not held-out evaluation. Source and code SHA-256 hashes are saved in `summary.json` and the parent `report.json`.",
             "- GitHub method reference: [Sara12-2/ResumeMatch_AI, pinned commit](https://github.com/Sara12-2/ResumeMatch_AI/tree/a30c9e32c0d801b0b2a48f3b8883245ad67636b7). `src/skill_extractor.py`: canonical skill overlap (`_get_matcher`, `_build_matcher`, `extract_skills`, `analyze_skill_gap`), degree ranking (`highest_degree_level`), and professional-domain mapping (`detect_domains`, `get_domain_matcher`). `src/ats_score.py`: degree attainment comparison (`EnhancedATSScore._compute_education_match`). These methods inform auditable set/degree comparisons; our scoring scales, taxonomy, and weights are project choices. Title, experience, location and salary rules are project implementations. No upstream code is imported.", "",
             "```powershell", "python scripts/ats_silver_eval.py --full --as-of 2026-09-30", "python scripts/ats_silver_eda.py", "```", "",
             "Artifacts: `pair_comparison.csv`, `disagreements.csv`, `silver_strong_pairs.csv`, `summary.json`, `overview.png`, `score_by_grade.png`, and interactive `report.html`."]
    markdown = "\n".join(lines) + "\n"
    (out / "report.md").write_text(markdown, encoding="utf-8")
    # Embed both figures and JavaScript so the HTML can be shared as a standalone file.
    import markdown as md

    body = md.markdown(markdown, extensions=["tables", "fenced_code"])
    for name in ("overview.png", "score_by_grade.png"):
        data_uri = "data:image/png;base64," + base64.b64encode((out / name).read_bytes()).decode("ascii")
        body = body.replace(f'src="{name}"', f'src="{data_uri}"')
    scatter_frame = frame.copy()
    scatter_frame["silver_label"] = scatter_frame.silver_grade.map(dict(enumerate(LABELS)))
    scatter = px.scatter(scatter_frame, x="silver_score", y="runtime_score", color="source",
                         hover_data=["cv_id", "jd_id", "cv_domain", "pair_type", "silver_label", "runtime_grade", "coverage"],
                         labels={"silver_score": "Silver total score", "runtime_score": "Runtime total score"},
                         title="Interactive paired scores — click a legend source to hide/show it", opacity=.5)
    scatter.add_shape(type="line", x0=0, x1=100, y0=0, y1=100, line={"color": "#334155", "dash": "dash"})
    scatter.update_layout(template="plotly_white", height=560, xaxis_range=[0, 100], yaxis_range=[0, 100])
    interactive = scatter.to_html(full_html=False, include_plotlyjs=False, div_id="paired-scores")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ATS vs silver — full EDA</title>
    <style>body{{margin:0;background:#f1f5f9;color:#0f172a;font:16px/1.6 system-ui,sans-serif}}main{{max-width:1280px;margin:24px auto;background:white;padding:36px;border-radius:16px}}h1,h2,h3{{line-height:1.25}}h2{{margin-top:38px;border-top:1px solid #e2e8f0;padding-top:24px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;display:block;overflow:auto;margin:20px 0}}th,td{{padding:9px 12px;border-bottom:1px solid #e2e8f0;white-space:nowrap;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eff6ff}}a{{color:#1d4ed8}}pre{{overflow:auto;padding:16px;background:#f8fafc}}.note{{padding:16px;background:#eff6ff;border-radius:8px}}</style>
    <script>{get_plotlyjs()}</script></head><body><main><p class="note">Complete offline audit: 1,672 pairs. Primary comparison: 1,144 Gemini-labelled pairs. Heuristics shown separately.</p>
    <h2>Explore individual pairs</h2>{interactive}{body}<footer>Generated from frozen score results; zero new LLM/API requests.</footer></main></body></html>"""
    (out / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps({"report": str(out / "report.html"), "comparison": comparison.replace({np.nan: None}).to_dict("records"),
                      "gemini_cv_rank": main_stat["per_cv_spearman"], "gemini_grade_means": by_grade.to_dict("records")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
