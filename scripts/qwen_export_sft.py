"""Export Qwen QLoRA SFT chat JSONL files for the CV or JD extraction task from manifest.json (--task)."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from scripts.cv_teacher_validate import check
from scripts.jd_teacher_check import check_jd
from src.scoring import extraction
from src.scoring.extraction import write_json

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "data/datasets/qwen_training/manifest.json"
TASK_KINDS = {"cv_extraction": "resume", "jd_extraction": "job"}
SPLITS = ("train", "validation", "test")
ALLOWED_STATUS = {"pass", "partial"}
MAX_CHARS = 40000


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _char_stats(lengths: list[int]) -> dict[str, int | float]:
    if not lengths:
        return {"max": 0, "median": 0}
    return {"max": max(lengths), "median": median(lengths)}


def export_sft(
    manifest_path: Path = DEFAULT_MANIFEST,
    out_dir: Path | None = None,
    root: Path = ROOT,
    task: str = "cv_extraction",
) -> dict:
    if task not in TASK_KINDS:
        raise ValueError(f"Unsupported task: {task!r}")
    kind = TASK_KINDS[task]
    is_cv = kind == "resume"
    if out_dir is None:
        out_dir = ROOT / "data/datasets/qwen_training" / task

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("tasks", {}).get(task, [])

    splits_by_id: dict[str, str] = {}
    for item in items:
        item_id, split = item["id"], item["split"]
        if split not in SPLITS:
            raise ValueError(f"Invalid split {split!r} for {item_id}")
        assert splits_by_id.setdefault(item_id, split) == split, f"Duplicate id across splits: {item_id}"

    by_split: dict[str, list[dict]] = {s: [] for s in SPLITS}
    skipped: list[dict] = []
    user_lengths: list[int] = []
    assistant_lengths: list[int] = []
    system_sha256: str | None = None
    split_by_text: dict[str, tuple[str, str]] = {}
    as_of_dates: set[str] = set()

    for item in sorted(items, key=lambda r: r["id"]):
        item_id, split, status = item["id"], item["split"], item.get("status", "")
        as_of = item.get("as_of")
        if is_cv and not as_of:  # each batch's degree calls hold on its own labelling date
            raise ValueError(f"{item_id}: manifest item has no as_of")
        if status not in ALLOWED_STATUS:
            skipped.append({"id": item_id, "split": split, "status": status, "reason": f"status:{status}"})
            continue

        text_path = root / item["text"]
        text = text_path.read_text(encoding="utf-8")
        if len(text) > MAX_CHARS:
            skipped.append({
                "id": item_id,
                "split": split,
                "status": status,
                "reason": f"max_chars_exceeded:{len(text)}>{MAX_CHARS}",
            })
            continue

        # the same text under two ids must not sit in two splits (train/test leak)
        seen = split_by_text.setdefault(_sha256(" ".join(text.split())), (item_id, split))
        if seen[1] != split:
            raise ValueError(f"Same text in {seen[1]} ({seen[0]}) and {split} ({item_id})")

        label_path = root / item["label"]
        label = json.loads(label_path.read_text(encoding="utf-8"))
        if not isinstance(label, dict) or set(label) != {"document", "facts"}:
            raise ValueError(f"Label {item_id} must contain exactly 'document' and 'facts', got {sorted(label)}")
        live = check(label_path, text, {}) if is_cv else check_jd(label_path, text)  # never trust a stored status: the label or text may have changed
        if live["status"] not in ALLOWED_STATUS:
            skipped.append({"id": item_id, "split": split, "status": live["status"], "reason": (live.get("error") or "")[:300]})
            continue

        if is_cv:
            system, user = extraction._prompts("resume", text, "v3", as_of)
            as_of_dates.add(as_of)
        else:
            system, user = extraction._prompts("job", text, "v3")
        assistant = json.dumps(label, ensure_ascii=False, separators=(",", ":"))
        system_sha256 = _sha256(system)
        user_lengths.append(len(user))
        assistant_lengths.append(len(assistant))

        by_split[split].append({
            "id": item_id,
            "split": split,
            "source": item["source"],
            "text_sha256": _sha256(text),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })

    if system_sha256 is None:
        system, _ = extraction._prompts(kind, "", "v3")
        system_sha256 = _sha256(system)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in by_split[split])
        (out_dir / f"{split}.jsonl").write_text(lines, encoding="utf-8")

    stats = {
        "task": task,
        "ruleset": "v3",
        "max_chars": MAX_CHARS,
        "as_of_dates": sorted(as_of_dates),
        "counts": {split: len(by_split[split]) for split in SPLITS},
        "total_exported": sum(len(by_split[s]) for s in SPLITS),
        "skipped": skipped,
        "user_chars": _char_stats(user_lengths),
        "assistant_chars": _char_stats(assistant_lengths),
        "system_prompt_sha256": system_sha256,
    }
    write_json(out_dir / "stats.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task", choices=tuple(TASK_KINDS), default="cv_extraction")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    export_sft(args.manifest, args.out, task=args.task)


if __name__ == "__main__":
    main()
