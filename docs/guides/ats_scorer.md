# Running and evaluating the ATS scorer

For the Phase 2 design, current rules and proposed weighted-skill extension, see
[Explanation: the Phase 2 ATS scorer](../explanations/scorer_explanation.md).

The first version scores English structured documents with deterministic code.
LLMs extract source-backed facts only. The public dataset's labels never enter
the extraction prompts or the scorer. See [scorer_references.md](../reference/scorer_references.md)
for the exact GitHub functions referenced and all project scoring decisions.

## Runtime API

```python
from datetime import date
from src.scoring import JobInput, ResumeInput, SalaryPreference, score_pair

# Wrap existing ResumeSchema / JobDescriptionSchema with Facts and literal
# Evidence quotes. The extraction adapters produce these wrappers automatically.
result = score_pair(resume_input, job_input,
                    preferences=SalaryPreference(),  # Negotiable by default
                    as_of_date=date(2026, 9, 30))
print(result.overall_score, result.grade, result.coverage)
```

`ResumeInput` and `JobInput` each contain `document` (the existing parser model)
and `facts` (evidence, degrees, professional domains, explicit location/currency,
and remote eligibility). Existing schema defaults do not supply evidence.
`ScoreResult` returns all seven subscores, total, grade, coverage, matched/missing
skills, preferred coverage, per-dimension evidence/basis/rule IDs, configuration
and the fixed employment evaluation date. No generated explanation is required.

Extraction uses the existing `LLMClient`, `VLLMClient`, `OllamaClient`, and CV
validation API. A CV envelope adapter replaces the old length-limited prompt so
all input text is sent within the configured 40,000-character limit. Longer inputs
fail visibly, rather than being silently truncated. Source quotes must literally
occur in the text. This validates quotation provenance, not semantic correctness
of every extracted fact. Extraction errors are reported and can be retried.

The Ollama adapter explicitly requests `num_ctx` (default 32,768) and checks a
conservative UTF-8 byte budget including output and prompt-template headroom.
This avoids relying on the server's potentially smaller default context. Configure
`--context-window` consistently for extraction and scoring and ensure the model
supports that size; increasing context requires more memory. vLLM must be started
with sufficient model context and reject oversized requests. API/context details:
[Ollama context length](https://docs.ollama.com/context-length) and
[generation API](https://docs.ollama.com/api/generate).

## Reproducible commands

Run from the repository root using a Python environment with the project
dependencies installed (`requirements.txt`). These commands write to ignored
`output/ats/`; raw texts and extraction caches are not committed.

```powershell
python -m scripts.ats_eval download
python -m scripts.ats_eval prepare --csv output/ats/raw/train.csv output/ats/raw/test.csv --revision 08978e21714984bb417547d2c0f9b477f5298163

# Start with three short development pairs (one per label).
python -m scripts.ats_eval smoke --split development --backend ollama --endpoint http://127.0.0.1:11434 --model qwen2.5:7b-instruct --as-of 2026-09-30

# Use an actual serving model; replace the example endpoint/model as needed.
python -m scripts.ats_eval extract --split development --backend ollama --endpoint http://localhost:11434 --model qwen2.5:7b-instruct
python -m scripts.ats_eval score --split development --backend ollama --endpoint http://localhost:11434 --model qwen2.5:7b-instruct --as-of 2026-09-30
python -m scripts.ats_eval calibrate

python -m scripts.ats_eval extract --split test --backend ollama --endpoint http://localhost:11434 --model qwen2.5:7b-instruct
python -m scripts.ats_eval score --split test --backend ollama --endpoint http://localhost:11434 --model qwen2.5:7b-instruct --as-of 2026-09-30 --calibration output/ats/calibration.json
python -m scripts.ats_eval report
```

For vLLM use `--backend vllm`, its actual base URL, and its registered model name.
For the authorized Gemini comparison, supply `GEMINI_API_KEY` only in the process
environment and use the following command. Never put the credential in a command
argument, configuration file, committed file, or report.

```powershell
python -m scripts.ats_eval smoke --split development --backend gemini --endpoint https://generativelanguage.googleapis.com/v1beta --model gemini-3.7-flash --as-of 2026-09-30 --output output/ats/gemini_results
```

The Gemini client sends credentials in the `x-goog-api-key` header to the official
HTTPS API only. The supplied project's limit is five requests per minute:
requests are spaced at least 13 seconds apart, including metadata requests and
retries. Run one Gemini extraction process at a time. HTTP 429/503 responses get
at most two retries; high-demand 503 responses can persist within the quota.
Daily quota exhaustion is not retried. This key's diagnostic additionally reported
a 20-request daily cap for gemini-3.7-flash. See
[gemini_smoke_observations.md](../reports/model-extraction/gemini_smoke_observations.md) for the actual attempts.
Provider failures are distinct from invalid extraction output. Provider model
version and token usage are recorded in successful cache entries. API reference:
[Gemini generateContent](https://ai.google.dev/api/generate-content).

Backend, endpoint, model, extraction settings and the date must remain consistent
across scoring stages. Cache identity also contains document hash, prompt version
and schema version. Version prompts/schemas when extraction behavior changes.
Pin the actual serving model revision externally; a model name alone does not
guarantee unchanged weights. `extract --limit N` is a smoke check, not full evaluation.

`smoke` extracts both documents of each selected development pair and calculates
default-rule scores without calibration. It saves `results/smoke.json`, including
the local Ollama model digest, successes and failures. Selection uses shortest
combined text length per label, so it is explicitly unsuitable for accuracy claims.
Ollama generation is constrained to the wrapper's JSON schema, in addition to
the subsequent Pydantic and source-quotation validation.
Whitespace-only quote differences are aligned to the exact original source span;
changed wording is rejected. Failed model responses are retained locally for
review. See [base_qwen_observations.md](../reports/model-extraction/base_qwen_observations.md) for the first
real base-model smoke-test results and candidate training objectives.

Scoring reads successful cached extractions only. Test scoring requires calibration
and refuses a changed dataset manifest, rule-code digest, extraction configuration,
or employment evaluation date. A missing extraction fails that pair and is included
in failure reporting; it is never replaced by fabricated scores. Review extraction
failures and coverage before interpreting accuracy.

## Data audit and split

Source: [cnamuangtoun/resume-job-description-fit](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit),
pinned revision `08978e21714984bb417547d2c0f9b477f5298163`.

Actual audit of train.csv and test.csv:

| Result | Count |
|---|---:|
| Original pairs | 8,000 |
| Additional duplicate pair rows | 7 |
| Quarantined pair identities with conflicting labels | 6 |
| Cross-partition pair identities excluded | 3,063 |
| Development pairs | 4,342 |
| Held-out test pairs | 582 |
| Retained unique CVs / JDs | 611 / 342 |

| Label | Development | Test |
|---|---:|---:|
| No Fit | 2,115 | 292 |
| Potential Fit | 1,091 | 152 |
| Good Fit | 1,136 | 138 |

Original published splits are pooled for this project split. CV and JD identities
are whitespace-normalized SHA-256 hashes. Each side is assigned independently
with development probability .75, test .25, seed 42. Retain only pairs whose
documents land in the same partition. Pair probabilities are therefore about
56.25% development and 6.25% test, with the rest excluded. Actual counts depend
on the document graph. Both partitions contain all three labels. This deliberately
trades pair retention for no CV or JD identity overlap. Near duplicates beyond
whitespace normalization are not detected in v1.

Artifacts preserve labels and file/line provenance, input checksums, exclusions,
split policy and seed. Conflicting duplicate labels are quarantined, never resolved
by guessing. The existing 1,672 silver pairs are retained separately and are not
part of threshold tuning or the new held-out score report.

## Evaluation interpretation

`report.json` contains macro-F1, balanced accuracy, accuracy, precision/recall/F1
by label, confusion matrices, label score mean/median/spread, five-point histogram
overlap, evidence coverage, failure rate, CV-cluster bootstrap 95% intervals and
representative errors. Baselines are normalized skill coverage with its own
development-tuned cutoffs and the development majority label.

Scores predict three-class agreement, not probabilities of recruitment success.
Dataset label provenance is undocumented on its public card. Label means are
descriptive diagnostics, not continuous numerical ground truth. No held-out fit
metrics should be reported until actual extraction, calibration and evaluation
finish. The default dictionary is intentionally finite: unsupported occupation
aliases produce missing evidence, and its coverage must be measured.

Unit tests include a fixture-only cached pipeline. Its metrics verify the command
contracts, not model quality. Run meaningful checks with:

```powershell
python -m pytest tests/unit/test_ats_scorer.py tests/unit/test_ats_evaluation.py tests/unit/test_match_schemas.py -q
```

## Small offline silver comparison

Run against stored structured CVs and JDs without an extraction service:

```powershell
python scripts/ats_silver_eval.py --per-grade 10 --seed 42 --as-of 2026-09-30
```

This scores only 30 pairs: ten per silver grade, including five Gemini-labelled
and five heuristic Mismatch pairs. The original 1,672-pair dataset is unchanged.
Outputs under ignored `output/ats/silver_sample/` are `report.md`, `report.json`,
`results.json` and `results.csv`. Labels and teacher subscores are compared only
after runtime scoring; CV domain metadata and JD categories do not supply facts.
The adapter uses structured JSON values as evidence, not newly verified raw-text
extraction. JD title/seniority skill tags use the existing annotation script's
filter. Optional/ambiguous degree requirements remain unknown; “MS Office” does
not establish a master's-degree requirement.

The 2026-09-30 sample completed 30/30 pairs with zero API calls. Final grade
agreement was 8/25 (32%) on Gemini-labelled pairs and 5/5 on heuristic negatives,
or 13/30 (43.3%) combined. Gemini-only total-score MAE was 24.0/100. Runtime
predicted 26 Mismatch, four Partial, and zero Strong. Mean runtime totals by
silver grade were 37.4 Mismatch, 45.8 Partial and 42.9 Strong. Title/domain
support was only 1/30; skills supported 28/30 but averaged 12/100. These findings
suggest reviewing title vocabulary and structured skill completeness before
calibrating thresholds. They do not establish which scoring system is correct.
This balanced sample is a diagnostic comparison with teacher/heuristic labels,
not an estimate of full-dataset accuracy or validation against hiring decisions.

## Full silver audit and EDA

```powershell
python scripts/ats_silver_eval.py --full --as-of 2026-09-30
python scripts/ats_silver_eda.py
```

The EDA script additionally requires `numpy`, `pandas`, `matplotlib`, `plotly`,
`markdown` and `tabulate`. Generated artifacts remain ignored under
`output/ats/silver_full/eda/`: standalone interactive `report.html`, `report.md`,
two chart PNGs, `summary.json`, `pair_comparison.csv`, `disagreements.csv`, and
`silver_strong_pairs.csv`. The complete paired dimensional evidence remains in
`output/ats/silver_full/results.json`. Sample artifacts are retained separately.

The full 2026-09-30 audit scored 1,672/1,672 pairs successfully: 88 CVs, 353 JDs,
zero errors and zero API requests. Gemini-only agreement is 59.3% across 1,144
pairs, compared with a 57.5% majority baseline. Total-score MAE is 14.67/100 and
Spearman rank correlation is 0.208. Agreement on 528 heuristic negatives is
96.8%; combined agreement is 71.1%, compared with a 70.9% majority baseline.
Runtime produces 1,543 Mismatch, 129 Partial and zero Strong grades. Title
support on Gemini pairs is 1.0%, domain support is 2.0%, and 833/1,144 pairs
receive a runtime skill score of zero. Audit extraction/aliases before tuning.

EDA correlations and within-CV rankings compare continuous silver totals, while
confusion matrices compare grades. Gemini and heuristic sources are separated;
CV-domain metadata is used only for grouping. Code/result hashes, missing-fact
bases, supported-only dimensional errors, distributions and majority baselines
are saved with the report. These metrics describe agreement in the existing
dataset and do not establish correctness against hiring decisions or an
independent human review. No calibration or training used these comparison pairs.
