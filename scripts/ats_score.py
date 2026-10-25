"""Score one source-backed CV/JD envelope pair with an explicit evaluation date."""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.scoring import JobInput, ResumeInput, ScorerConfig, score_pair
from src.scoring.extraction import write_json
from src.scoring.v2_benchmark import load_relations, variant_config
from src.scoring.v2_validation import validate_skill_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True, help="ResumeInput JSON envelope")
    parser.add_argument("--job", type=Path, required=True, help="JobInput JSON envelope")
    parser.add_argument("--ruleset", choices=["v1", "v2", "v3"], default="v1")
    parser.add_argument("--depth-policy", choices=["off", "score_only", "full"])
    parser.add_argument("--preferred-policy", choices=["separate", "legacy_share"], help="v3 defaults to separate Preferred coverage")
    parser.add_argument("--variant", choices=["equal_groups", "group_weights", "semantic", "proficiency", "full"], default="full")
    parser.add_argument("--relations", type=Path)
    parser.add_argument("--allow-provisional-relations", action="store_true")
    parser.add_argument("--resume-source", type=Path, help="Optional original extracted text for source validation")
    parser.add_argument("--job-source", type=Path, help="Optional original extracted text for source validation")
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    cv = ResumeInput.model_validate_json(args.resume.read_text(encoding="utf-8-sig"))
    jd = JobInput.model_validate_json(args.job.read_text(encoding="utf-8-sig"))
    if args.ruleset == "v1" and (args.relations or args.allow_provisional_relations):
        parser.error("Semantic relations require --ruleset v2")
    relations = load_relations(args.relations)
    if args.depth_policy and args.ruleset != "v3":
        parser.error("--depth-policy requires --ruleset v3")
    if args.preferred_policy and args.ruleset != "v3":
        parser.error("--preferred-policy requires --ruleset v3")
    if args.ruleset in {"v2", "v3"}:
        validator = validate_skill_evidence
        if args.ruleset == "v3":
            from src.scoring.v3_validation import validate_skill_evidence as v3_validator
            validator = v3_validator
        for envelope, path in ((cv, args.resume_source), (jd, args.job_source)):
            if path:
                source_text = path.read_bytes().decode("utf-8-sig")
                expected_sha = envelope.annotation.text_sha256 if envelope.annotation else None
                # Existing corpus extraction uses universal newlines. Preserve raw
                # text when its checksum matches; accept LF only when it matches
                # the recorded identity, so unrelated source edits still fail.
                if expected_sha and hashlib.sha256(source_text.encode()).hexdigest() != expected_sha:
                    lf_text = source_text.replace("\r\n", "\n").replace("\r", "\n")
                    if hashlib.sha256(lf_text.encode()).hexdigest() == expected_sha:
                        source_text = lf_text
                issues = validator(envelope, source_text)
                if issues and envelope.annotation:
                    envelope.annotation.skills_status = "partial"
    config = ScorerConfig() if args.ruleset == "v1" else variant_config(args.variant, relations,
        args.allow_provisional_relations)
    if args.ruleset == "v3":
        config = ScorerConfig(**{**config.model_dump(), "version": "ats-en-v3", "depth_policy": args.depth_policy or "full",
                                 "preferred_policy": args.preferred_policy or "separate"})
    result = score_pair(cv, jd, config=config, as_of_date=args.as_of)
    payload = result.model_dump(mode="json")
    if args.output:
        write_json(args.output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


if __name__ == "__main__":
    main()
