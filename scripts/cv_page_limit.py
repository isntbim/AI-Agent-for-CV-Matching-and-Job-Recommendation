"""How long a CV can the extractor read? Measures the fixed prompt overhead, tokens per page on the CV pool, and the
token density of Vietnamese text, then derives the page limit from the 32,768-token context (mentor M6.5).

Constraint (src/scoring/v4_client.py): prompt tokens + output cap <= NUM_CTX, output cap = min(16384, 3 x source + 4000).
Above 40,000 characters a CV is read in pieces of at most 20,000 characters and merged (src/scoring/extraction.py), so the
limit applies per piece, and a CV of any length is accepted as long as each piece fits.

    python -m scripts.cv_page_limit            # -> output/review3/page_limit.json
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pypdf
from transformers import AutoTokenizer

from src.parser.extractors import extract_text_from_file
from src.scoring.extraction import extract_document
from src.scoring.v4_client import MAX_NEW_TOKENS, NUM_CTX, token_cap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/review3/page_limit.json"
MAX_CHARS = 40000  # extract_document default used by the service; longer CVs are split into pieces of MAX_CHARS // 2


class _Capture:
    """Records the exact system + user prompt the extractor would send, then stops."""

    def __init__(self):
        self.prompts = None

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=4096):
        self.prompts = (system_prompt, user_prompt)
        raise StopIteration


def overhead(tok) -> int:
    """Prompt tokens that are not the CV text: chat template + system prompt + JSON wrapper of the user turn."""
    cap = _Capture()
    sample = "Jane Doe\nAccountant\nSkills: Excel, SAP\n" * 10
    try:
        extract_document(sample, "resume", cap, max_tokens=MAX_NEW_TOKENS, ruleset="v3")
    except StopIteration:
        pass
    system, user = cap.prompts
    prompt = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True)
    source = json.loads(user)["source_text"]
    return len(tok(prompt)["input_ids"]) - len(tok(source)["input_ids"])


def fits(tok_source: int, p0: int) -> bool:
    return p0 + tok_source + token_cap(tok_source) <= NUM_CTX


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    p0 = overhead(tok)
    max_source = max(s for s in range(0, NUM_CTX) if fits(s, p0))  # monotone, so the largest fitting length

    rows, skipped = [], 0
    for pdf in sorted((ROOT / "data/raw/cvs").rglob("*.pdf")):
        try:
            pages = len(pypdf.PdfReader(str(pdf)).pages)
            text = extract_text_from_file(pdf)
        except Exception:
            skipped += 1
            continue
        if pages == 0 or len(text.strip()) < 200:  # scanned or empty: no text layer, the API rejects these
            skipped += 1
            continue
        n = len(tok(text)["input_ids"])
        rows.append({"pages": pages, "chars": len(text), "tokens": n, "tok_per_page": n / pages,
                     "chars_per_tok": len(text) / n})

    vi = [p.read_text(encoding="utf-8") for p in sorted((ROOT / "data/raw/jobs_vi").glob("*.txt"))]
    vi_chars_per_tok = sum(map(len, vi)) / sum(len(tok(t)["input_ids"]) for t in vi)
    en_chars_per_tok = sum(r["chars"] for r in rows) / sum(r["tokens"] for r in rows)
    tpp = [r["tok_per_page"] for r in rows]
    cpp = [r["chars"] / r["pages"] for r in rows]

    # A CV of up to MAX_CHARS characters is read in one call; the largest such text in tokens decides whether it fits.
    one_call_tokens = {"english": MAX_CHARS / en_chars_per_tok, "vietnamese": MAX_CHARS / vi_chars_per_tok}
    piece_tokens = {k: v / 2 for k, v in one_call_tokens.items()}
    result = {
        "num_ctx": NUM_CTX, "output_cap_max": MAX_NEW_TOKENS, "prompt_overhead_tokens": p0,
        "max_source_tokens_one_call": max_source,
        "cv_pool": {"cvs": len(rows), "skipped_no_text": skipped,
                    "pages": {"median": st.median(r["pages"] for r in rows), "max": max(r["pages"] for r in rows)},
                    "tokens_per_page": {"median": round(st.median(tpp)), "p90": round(pct(tpp, 0.9)), "max": round(max(tpp))},
                    "chars_per_page": {"median": round(st.median(cpp)), "p90": round(pct(cpp, 0.9))},
                    "tokens_per_cv": {"median": round(st.median(r["tokens"] for r in rows)),
                                      "max": max(r["tokens"] for r in rows)}},
        "chars_per_token": {"english_cvs": round(en_chars_per_tok, 2), "vietnamese_jds": round(vi_chars_per_tok, 2)},
        "max_chars_one_call": MAX_CHARS,
        "tokens_at_max_chars": {k: round(v) for k, v in one_call_tokens.items()},
        "fits_at_max_chars": {k: fits(round(v), p0) for k, v in one_call_tokens.items()},
        "pieces_fit": {k: fits(round(v), p0) for k, v in piece_tokens.items()},
        "pages_in_one_call": {
            "english_median_page": round(max_source / st.median(tpp), 1),
            "english_dense_page_p90": round(max_source / pct(tpp, 0.9), 1),
            "vietnamese_median_page_est": round(max_source / (st.median(cpp) / vi_chars_per_tok), 1)},
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
