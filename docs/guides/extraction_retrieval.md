# Extraction pilot and CV-to-jobs retrieval

The local implementation uses the existing 88 gold CVs and 361 primary JDs. It produces a source-reviewed extraction pilot and a separate cleaned retrieval corpus. The pilot and cleaning decisions are assistant-reviewed and await human approval. ATS scoring rules and extraction schemas are preserved; weighted skills and Qwen training still need mentor decisions.

The real-model diagnostic benchmark meets the local criterion for recommending reranking: macro NDCG@10 improves without lowering macro MRR@10. Reranking remains an explicit CLI option. The silver benchmark was previously inspected and is not an independent test set. Azure verification is complete: 361 rows pass consistency and repeat-upsert checks, and all 88 fixed queries pass with 100% HNSW/exact top-20 overlap. The live verification report is stored locally at `output/retrieval/azure_v2/report.md`; it is not included in Git.

## Local deliverables

Local evidence is stored at `knowledge-base/sessions/extraction-retrieval/extraction_retrieval_handoff.md`, `output/extraction_pilot/report.md`, and `output/retrieval/benchmark_v2/report.md`. Generated source text, decisions, datasets, embeddings, rankings and reports stay under ignored `output/` and `.cache/` directories. Those local review decisions must be retained to reproduce this pilot; they are deliberately absent from a fresh Git checkout.

The source manifest resolves PDFs by domain and original filename. CareerViet evidence comes from original workbook cells, with worksheet row numbers. ITviec evidence uses saved description snapshots and excludes regenerated location, category, seniority and compensation headers. The primary dataset's upstream language tags differ from heuristic detection on original source text; both are retained. The separate Vietnamese corpus is reserved for later evaluation.

## Environment and pinned models

Run from the repository root in PowerShell with Python 3.12. PyTorch must match the hardware. The verified machine used CUDA PyTorch 2.11.0+cu128; use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) for another machine. CPU reranking uses FP32; CUDA uses FP16.

```powershell
py -3.12 -m venv .venv-retrieval
.venv-retrieval/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv-retrieval/Scripts/python.exe -m pip install -r requirements-retrieval.txt
```

The tested package versions are recorded in `output/retrieval/environment.json`. Install CPU PyTorch instead of the CUDA wheel when necessary.

| Component | Model | Resolved revision |
| --- | --- | --- |
| Embeddings | BAAI/bge-m3 | `5617a9f61b028005a4858fdac845db406aefb181` |
| Reranker | BAAI/bge-reranker-v2-m3 | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| Token lengths | Qwen/Qwen2.5-7B-Instruct tokenizer | `a09a35458c702b33eeacc393d103063234e8bc28` |

Download these pinned revisions when preparing another machine. This command downloads BGE weights and only Qwen tokenizer files. It performs no LLM generation.

```powershell
.venv-retrieval/Scripts/python.exe -m scripts.retrieval models --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --tokenizer-revision a09a35458c702b33eeacc393d103063234e8bc28
```

## Reproduce the frozen pilot

These commands require the existing source files and `output/ats/prepared` public benchmark manifests. `near_duplicate_review.json` must cover every nominated pair before splitting. Splits are deterministic grouped 70/15/15 assignments with seed 42, stratified by document kind and six domain families. Near duplicates use five-word-shingle Jaccard >=0.90. Public benchmark assignments are preserved; overlapping duplicate groups are excluded from the new pilot.

```powershell
.venv-retrieval/Scripts/python.exe -m scripts.extraction_pilot sources
.venv-retrieval/Scripts/python.exe -m scripts.extraction_pilot split
.venv-retrieval/Scripts/python.exe -m scripts.extraction_pilot select
.venv-retrieval/Scripts/python.exe -m scripts.extraction_pilot reproduce
.venv-retrieval/Scripts/python.exe -m scripts.extraction_pilot validate
.venv-retrieval/Scripts/python.exe -m scripts.extraction_pilot tokens --revision a09a35458c702b33eeacc393d103063234e8bc28
.venv-retrieval/Scripts/python.exe -m scripts.extraction_pilot validate-disagreements
```

`reproduce` regenerates the manifest, splits and selections twice and checks their bytes against the existing artifacts, while checking that the public benchmark files remain unchanged. Source resolution errors are explicit in `source_errors.json`.

The completed targets are in `targets/`, with per-field ledgers in `reviews/`, consolidated `review_ledger.json`, decisions in `review_decisions.json`, and conversational examples in `conversations.jsonl`. Every populated final field is audited. Schema-required placeholders and booleans are identified in the notes and are not evidence-backed facts. Quotations are checked against original text. Human approval remains pending.

For a new pilot version, run `review-proposals` after selection, review original source text and every populated field, then populate `review_decisions.json` and run `apply-review` followed by `validate`. `review-proposals` overwrites provisional targets and reviews; keep completed versions in a separate output directory. `apply-review` applies explicit source-supported corrections, but does not confer human approval.

`disagreements` nominates four cases per CV domain with both documents in training. Review the complete sources and store classifications and findings in `disagreement_review.json`, then run `validate-disagreements`. It checks original scores, labels, source identity and manifest bindings. The current training subset contains no actual strong-grade gate cases; the six gate slots use documented fallback cases with hypothetical gate diagnostics. Original scores remain unchanged.

## Prepare and index the retrieval corpus

```powershell
.venv-retrieval/Scripts/python.exe -m scripts.retrieval nominate
.venv-retrieval/Scripts/python.exe -m scripts.retrieval prepare
.venv-retrieval/Scripts/python.exe -m scripts.retrieval index --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 --device cuda
```

`nominate` creates candidates for review; `prepare` requires `output/retrieval/cleaning_review.json` with explicit source-reviewed remove/retain decisions. The final corpus is `output/retrieval/corpus_v2`. IDs and original records are preserved. The versioned correction ledger records every removal and retained candidate, its source checksum, and review rationale. Preparation rejects stale or incomplete review decisions. A second cleaning pass leaves the corpus unchanged.

`index` produces 1,024-dimensional BGE-M3 vectors for the corpus and frozen CV queries before a reranker is loaded. The cache key includes text checksum, model name, resolved revision and preprocessing version. Vectors are validated and normalized. Index manifests record source and vector checksums, runtime, peak allocated GPU memory, and inputs exceeding the 8,192-token embedding limit.

## Query and evaluate

```powershell
# Existing indexed CV; omit --rerank to return hybrid retrieval only.
.venv-retrieval/Scripts/python.exe -m scripts.retrieval query --cv-id cv_005 --rerank --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --output output/retrieval/queries_v2

# Another structured CV, using the same serializer and embedding revision.
.venv-retrieval/Scripts/python.exe -m scripts.retrieval query --cv-json path/to/cv.json --rerank --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --output output/retrieval/queries_v2

.venv-retrieval/Scripts/python.exe -m scripts.retrieval evaluate --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --device cuda
```

The query JSON file can contain a structured resume or a `document` envelope. Empty retrieval text is rejected. A new CV is embedded and its embedder released before the reranker loads. Query reports retain candidate text and scores in the chosen ignored output directory; stdout shows IDs, ranks and scores.

Dense and BM25 each retrieve 50 candidates. BM25 uses `k1=1.5`, `b=0.75` and `technical-tokens-v1`, preserving `C++`, `C#`, `.NET` and dotted technology names. Equal-weight RRF uses constant 60. The optional reranker scores the top 20 fused candidates and returns ten. Serializers include populated titles, skills, experience, education, requirements and responsibilities while omitting domain labels, silver labels, ATS totals, contact information and invented missing values.

The reranker interface is `RerankerClient.score_pairs(pairs) -> scores`. `MockReranker` is deterministic; `BGEReranker` uses Transformers sequence classification and raw relevance logits. These scores remain separate from ATS totals. Batch size starts at one, and pair limit is 8,192 tokens. Pair truncation is measured before `longest_first` truncation and recorded with removed token counts. Ranking rejects duplicate IDs, score-count mismatches and non-finite scores. Ties retain original candidate order, followed by stable ID.

The benchmark freezes candidate IDs and labels in `frozen_pools.json`. Primary diagnostics use Gemini-labelled pairs; heuristic negatives are reported in the separate expanded pool. Relevance is grade >=1 and NDCG gain is `2^grade - 1`. Reports include MRR@10, NDCG@5/10, domain macro averages, positive-only summaries, four queries with no labelled positives, latency, memory, failures and truncations. Full-corpus retrieval reports known-positive Recall@50 and judged-result coverage; unjudged results are not scored as negatives. Stage timings use precomputed query embeddings, exclude startup from per-stage latency and reuse cached scores across pools. Use a new benchmark output directory for a cold score-cache run.

Use versioned `--corpus`, `--index` and `--output` paths for changed data, model revisions or evaluation settings. Existing frozen pools reject changes. `--device cpu` provides FP32 operation. Application and web integration are outside this package.

## Verification

```powershell
.venv-retrieval/Scripts/python.exe -m pytest tests/unit/test_retrieval.py tests/unit/test_extraction_pilot.py -q
.venv-retrieval/Scripts/python.exe -m pytest tests/unit/test_ats_scorer.py tests/unit/test_ats_evaluation.py tests/unit/test_ats_silver.py tests/unit/test_ats_silver_eda.py tests/unit/test_gemini_extraction.py tests/unit/test_embedder.py tests/unit/test_pgvector_store.py tests/unit/test_llm_extractor.py -q

# Mock CLI integration, with independent mock index and score cache.
.venv-retrieval/Scripts/python.exe -m scripts.retrieval index --mock --index output/retrieval/mock_index_v2 --cache .cache/retrieval_mock
.venv-retrieval/Scripts/python.exe -m scripts.retrieval evaluate --mock --index output/retrieval/mock_index_v2 --cache .cache/retrieval_mock --output output/retrieval/mock_benchmark_v2
.venv-retrieval/Scripts/python.exe -m scripts.retrieval query --mock --index output/retrieval/mock_index_v2 --rerank --cv-id cv_005 --output output/retrieval/mock_queries_v2
```

The real benchmark is separate from the mock tests. Reranking is eligible for a default recommendation only when primary macro NDCG@10 improves, macro MRR@10 does not fall, and queries have no failures. The current result passes this criterion, but data-analysis NDCG and frontend MRR regressions are retained in the report.

## Azure checkpoint

Finish local work, notify the user that the database is needed, and wait for confirmation that Azure PostgreSQL is on. Do not run the next command before that confirmation. Existing `.env`/Settings credential handling is reused and credentials are not written into artifacts.

```powershell
.venv-retrieval/Scripts/python.exe -m scripts.retrieval azure-verify --reranker-revision 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e --device cuda --output output/retrieval/azure_v2
```

The final manifest selects isolated table `retrieval_ae041ac4ce81381e7965`. Verification preserves other tables, upserts all 361 cleaned JDs, checks row IDs, payload/text checksums and vectors, and repeats upserts to verify idempotence. It compares exact cosine search to local vectors across all 88 fixed queries, then tests HNSW separately with `ef_search=100`, requiring >=95% top-20 overlap for every query. Stable ID ordering resolves equal-distance boundaries among returned candidates. The same hybrid/reranking pipeline uses Azure for dense candidates. `report.json` contains plans, comparisons and failures; `flows.json` records returned rankings and truncations. Notify the user when database testing has finished.

The completed Azure flow reproduces the local ordered top ten for 87/88 CVs. For `cv_016`, two separate JD IDs have identical retrieval text and vectors and tie at exact cosine ranks 50 and 51. HNSW returns the other tied ID at its top-50 cutoff, producing one final ID replacement. Azure exact search matches the local order, and all HNSW top-20 checks meet the acceptance threshold. The local artifact `output/retrieval/azure_v2/rank_difference_review.json` preserves this result.

Commit and push require a separate user instruction.
