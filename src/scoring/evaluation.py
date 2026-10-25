"""Calibration and metrics implemented with the standard library."""

import random
import statistics
from collections import Counter, defaultdict

from .models import ScorerConfig, runtime_grade


def _rows_config(rows: list[dict]) -> ScorerConfig:
    if len({r["result"].get("precision_policy_version", "legacy") for r in rows}) != 1:
        raise ValueError("Cannot combine scores from different precision policies")
    configs = [ScorerConfig.model_validate(r["result"].get("config", {})) for r in rows]
    base = configs[0]
    identity = base.model_dump(exclude={"lower_threshold", "upper_threshold"})
    if any(config.model_dump(exclude={"lower_threshold", "upper_threshold"}) != identity for config in configs):
        raise ValueError("Cannot combine scores from different scorer configurations")
    artifacts = [r.get("artifact_identity") for r in rows]
    if any(artifacts):
        from .v2_benchmark import require_same_identity
        if not all(artifacts):
            raise ValueError("Cannot combine scores with missing artifact identity")
        require_same_identity(*artifacts)
    return base


def classification_metrics(labels: list[int], predictions: list[int]) -> dict:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Metrics require equal-length, nonempty inputs")
    matrix = [[0] * 3 for _ in range(3)]
    for actual, prediction in zip(labels, predictions):
        if actual not in range(3) or prediction not in range(3):
            raise ValueError("Labels and predictions must be 0, 1, or 2")
        matrix[actual][prediction] += 1
    by_label = {}
    for k in range(3):
        tp, support, predicted = matrix[k][k], sum(matrix[k]), sum(row[k] for row in matrix)
        precision = tp / predicted if predicted else 0
        recall = tp / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        by_label[str(k)] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return {"macro_f1": statistics.mean(d["f1"] for d in by_label.values()),
                "balanced_accuracy": statistics.mean(d["recall"] for d in by_label.values()),
                "accuracy": sum(matrix[k][k] for k in range(3)) / len(labels),
                "per_label": by_label, "confusion_matrix": matrix, "n": len(labels)}


def _grade_values(result, config):
    if config.version == "ats-en-v3" and result.get("precision_policy_version"):
        if "unrounded_scores" not in result or "unrounded_overall_score" not in result:
            raise ValueError("Full-precision v3 evaluation requires unrounded scores")
        scores = result["unrounded_scores"]
        return result["unrounded_overall_score"], scores["skills"], scores["title"]
    return result["overall_score"], result["scores"]["skills"], result["scores"]["title"]


def calibrate(rows: list[dict], baseline: bool = False) -> ScorerConfig:
    if any(row["split"] != "development" for row in rows):
        raise ValueError("Calibration accepts development rows only")
    if {r["label_id"] for r in rows} != {0, 1, 2}:
        raise ValueError("Calibration requires all three development labels")
    source_config = _rows_config(rows)
    base_config = ScorerConfig() if baseline else source_config
    best_key, best = None, None
    # A histogram makes the exhaustive 5,050-threshold search inexpensive.
    histogram = Counter()
    for r in rows:
        result = r["result"]
        raw_total, raw_skills, raw_title = _grade_values(result, source_config)
        value = r["baseline_skill_score"] if baseline else raw_total
        gate = result.get("skills_breakdown") or {}
        blocked = base_config.version in {"ats-en-v2", "ats-en-v3"} and gate.get("strong_blocked", False)
        forced = not baseline and base_config.version in {"ats-en-v2", "ats-en-v3"} and gate.get("force_mismatch", False)
        eligible = baseline or (not blocked and raw_skills >= base_config.strong_skill_min
                                and raw_title >= base_config.strong_title_min)
        histogram[(r["label_id"], value, eligible, forced)] += 1
    for lower in range(100):
        for upper in range(lower + 1, 101):
            config = base_config.model_copy(update={"lower_threshold": lower, "upper_threshold": upper})
            matrix = [[0] * 3 for _ in range(3)]
            for (label, value, eligible, forced), count in histogram.items():
                prediction = runtime_grade(value, base_config.strong_skill_min if eligible else -1,
                    base_config.strong_title_min if eligible else -1, config, {"force_mismatch": forced})
                matrix[label][prediction] += count
            f1s = []
            for label in range(3):
                denominator = sum(matrix[label]) + sum(row[label] for row in matrix)
                f1s.append(2 * matrix[label][label] / denominator if denominator else 0)
            key = (statistics.mean(f1s), -(abs(lower - 50) + abs(upper - 75)), -lower, -upper)
            if best_key is None or key > best_key:
                best_key, best = key, config
    return best


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    values = sorted(values)
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
                "std": statistics.pstdev(values), "min": values[0], "max": values[-1],
                "p25": values[int((len(values) - 1) * .25)], "p75": values[int((len(values) - 1) * .75)]}


def _overlap(first: list[float], second: list[float]) -> float | None:
    """Shared probability mass in fixed five-point bins (0..100)."""
    if not first or not second:
        return None
    a, b = Counter(min(int(v // 5), 19) for v in first), Counter(min(int(v // 5), 19) for v in second)
    return sum(min(a[k] / len(first), b[k] / len(second)) for k in range(20))


def evaluate(rows: list[dict], config: ScorerConfig, baseline_config: ScorerConfig,
             majority_label: int, bootstrap: int = 500, seed: int = 42) -> dict:
    if not rows or any(r["split"] != "test" for r in rows):
        raise ValueError("Evaluation requires held-out test rows")
    if {r["label_id"] for r in rows} != {0, 1, 2}:
        raise ValueError("Successful test extraction must retain all three labels")
    source_config = _rows_config(rows)
    if source_config.model_dump(exclude={"lower_threshold", "upper_threshold"}) != config.model_dump(exclude={"lower_threshold", "upper_threshold"}):
        raise ValueError("Evaluation configuration does not match score artifact")
    labels = [r["label_id"] for r in rows]
    predictions = [runtime_grade(*_grade_values(r["result"], config), config,
                                 r["result"].get("skills_breakdown")) for r in rows]
    simple = [2 if r["baseline_skill_score"] >= baseline_config.upper_threshold else
              1 if r["baseline_skill_score"] >= baseline_config.lower_threshold else 0 for r in rows]
    groups = {str(label): [r["result"]["overall_score"] for r in rows if r["label_id"] == label] for label in range(3)}
    report = {"scorer": classification_metrics(labels, predictions),
                  "skill_baseline": classification_metrics(labels, simple),
                  "majority_baseline": classification_metrics(labels, [majority_label] * len(rows)),
                  "scores_by_label": {k: _summary(v) for k, v in groups.items()},
                  "score_overlap": {f"{a}-{b}": _overlap(groups[str(a)], groups[str(b)]) for a, b in [(0, 1), (1, 2), (0, 2)]},
                  "coverage": _summary([r["result"]["coverage"] for r in rows]),
                  "config": config.model_dump(), "baseline_config": baseline_config.model_dump()}
    report["dimension_coverage"] = {key: statistics.mean(r["result"]["dimensions"][key]["supported"] for r in rows)
                                     for key in rows[0]["result"]["dimensions"]}
    report["errors"] = [{"pair_id": r["pair_id"], "actual": label, "predicted": prediction,
                             "score": r["result"]["overall_score"], "coverage": r["result"]["coverage"]}
                         for r, label, prediction in zip(rows, labels, predictions) if label != prediction][:30]
    clusters = defaultdict(list)
    for i, row in enumerate(rows):
        clusters[row["cv_id"]].append(i)
    keys, rng, samples = sorted(clusters), random.Random(seed), []
    for _ in range(bootstrap):
        indices = [i for key in rng.choices(keys, k=len(keys)) for i in clusters[key]]
        metrics = classification_metrics([labels[i] for i in indices], [predictions[i] for i in indices])
        samples.append(metrics)
    report["bootstrap"] = {"unit": "cv_id", "iterations": bootstrap, "seed": seed, "confidence": .95,
                               "intervals": {key: [sorted(s[key] for s in samples)[int((bootstrap - 1) * q)] for q in [.025, .975]]
                                          for key in ["macro_f1", "balanced_accuracy"]} if samples else {}}
    return report
