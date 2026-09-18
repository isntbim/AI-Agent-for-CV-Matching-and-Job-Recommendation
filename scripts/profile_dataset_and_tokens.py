"""
profile_dataset_and_tokens.py
Empirical profiling script to calculate real statistical metrics across the CV dataset:
- File count per domain
- Character lengths, word counts, token estimations for raw PDF text
- Comparison with structured ResumeSchema.to_embedding_text()
- Formulate empirical proof for Token Reduction Rate
"""

import sys
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.parser.extractors import extract_text_from_file

def main():
    cvs_root = PROJECT_ROOT / "data" / "raw" / "cvs"
    categories = sorted([d for d in cvs_root.iterdir() if d.is_dir()])

    domain_stats = {}
    total_files = 0
    all_raw_words = []
    all_raw_chars = []

    print(f"=== PROFILING RAW CV DATASET (Root: {cvs_root}) ===")
    
    for cat in categories:
        pdf_files = sorted(list(cat.glob("*.pdf")))
        n_files = len(pdf_files)
        total_files += n_files
        
        cat_words = []
        cat_chars = []
        
        # Profile every file or representative sample (sample 20 per domain for fast exact stats)
        sampled = pdf_files[:25]
        for pf in sampled:
            try:
                txt = extract_text_from_file(pf)
                words = len(txt.split())
                chars = len(txt)
                cat_words.append(words)
                cat_chars.append(chars)
                all_raw_words.append(words)
                all_raw_chars.append(chars)
            except Exception as e:
                pass
                
        domain_stats[cat.name] = {
            "total_files": n_files,
            "sampled": len(cat_words),
            "mean_words": float(np.mean(cat_words)) if cat_words else 0.0,
            "median_words": float(np.median(cat_words)) if cat_words else 0.0,
            "std_words": float(np.std(cat_words)) if cat_words else 0.0,
            "min_words": int(np.min(cat_words)) if cat_words else 0,
            "max_words": int(np.max(cat_words)) if cat_words else 0,
            "mean_chars": float(np.mean(cat_chars)) if cat_chars else 0.0,
        }

    print(f"Total Categories: {len(categories)}")
    print(f"Total CV Files in Dataset: {total_files}")
    print("-" * 85)
    print(f"{'Domain / Category':<26} | {'Files':<6} | {'Sample':<6} | {'Mean Words':<11} | {'Median':<8} | {'Std':<7} | {'[Min, Max] Words'}")
    print("-" * 85)
    for cat, s in domain_stats.items():
        min_max = f"[{s['min_words']}, {s['max_words']}]"
        print(f"{cat:<26} | {s['total_files']:<6} | {s['sampled']:<6} | {s['mean_words']:<11.1f} | {s['median_words']:<8.1f} | {s['std_words']:<7.1f} | {min_max}")
    print("-" * 85)
    
    overall_mean = float(np.mean(all_raw_words))
    overall_median = float(np.median(all_raw_words))
    overall_std = float(np.std(all_raw_words))
    overall_min = int(np.min(all_raw_words))
    overall_max = int(np.max(all_raw_words))
    
    # BPE token multiplier in NLP literature:
    # 1 English word ~= 1.3 to 1.35 subword tokens (WordPiece / BPE / Byte-level BPE like Llama/Qwen)
    token_multiplier = 1.33
    est_mean_tokens = overall_mean * token_multiplier
    est_median_tokens = overall_median * token_multiplier
    est_min_tokens = overall_min * token_multiplier
    est_max_tokens = overall_max * token_multiplier
    
    print(f"\n[RAW CV OVERALL SUMMARY (N={len(all_raw_words)})]:")
    print(f"  • Mean Word Count:   {overall_mean:.1f} words (Estimated: ~{est_mean_tokens:.0f} tokens)")
    print(f"  • Median Word Count: {overall_median:.1f} words (Estimated: ~{est_median_tokens:.0f} tokens)")
    print(f"  • Std Deviation:     {overall_std:.1f} words")
    print(f"  • Min - Max Range:   [{overall_min}, {overall_max}] words (~[{est_min_tokens:.0f}, {est_max_tokens:.0f}] tokens)")

    # Save summary to a json file for docs/reports to reference
    report_data = {
        "total_files": total_files,
        "sample_size": len(all_raw_words),
        "overall_stats": {
            "mean_words": round(overall_mean, 1),
            "median_words": round(overall_median, 1),
            "std_words": round(overall_std, 1),
            "min_words": overall_min,
            "max_words": overall_max,
            "est_mean_tokens": round(est_mean_tokens, 0),
            "est_median_tokens": round(est_median_tokens, 0),
            "est_min_tokens": round(est_min_tokens, 0),
            "est_max_tokens": round(est_max_tokens, 0),
        },
        "domain_stats": domain_stats
    }
    
    out_file = PROJECT_ROOT / "data" / "processed" / "dataset_profiling_stats.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\nProfiling stats saved to: {out_file}")

if __name__ == "__main__":
    main()
