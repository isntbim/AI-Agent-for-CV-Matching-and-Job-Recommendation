# Running and evaluating the ATS scorer

For the final formulas, evidence policy and compatibility behavior, see
[Explanation: deterministic ATS scorer v1 and v2](../explanations/scorer_explanation.md).

`ats-en-v1` remains the default. V2 is optional and prepared annotations remain
provisional until you or a teammate completes review. Runtime scoring is offline;
an extraction service is only needed to create new envelopes.

## Select v1 or v2

```python
from datetime import date
from src.scoring import ScorerConfig, score_pair

# Existing consumers keep v1.
v1 = score_pair(resume_input, job_input, as_of_date=date(2026, 9, 30))

# V2 requires source-backed per-skill details and JD requirement units.
config = ScorerConfig(version="ats-en-v2", variant="full")
v2 = score_pair(resume_input, job_input, config=config,
                as_of_date=date(2026, 9, 30))
print(v2.skills_breakdown)
```

V2 variants are `equal_groups`, `group_weights`, `semantic`, `proficiency` and
`full`. They progressively add the final policies for diagnostic ablations.
Use `full` for the complete v2 rubric. A v1 envelope without complete v2 skill
annotations remains readable but receives neutral skills and an unknown gate
when scored with v2; extraction omissions never establish a Core failure.

Score one saved `ResumeInput` / `JobInput` JSON pair:

```powershell
python -m scripts.ats_score --resume path/to/resume_envelope.json --job path/to/job_envelope.json --ruleset v2 --as-of 2026-09-30 --output output/scorer_v2/pair.json
```

Add `--resume-source path/to/resume.txt --job-source path/to/job.txt` to validate
the envelopes against supplied source text. To activate a relation table, add
`--relations path/to/relations.json`. Only approved directed relations are active
by default. `--allow-provisional-relations` explicitly permits pending relations
for diagnostics, and the output records provisional status. Use `--ruleset v1`
for the existing scorer.

The corpus text checksum uses universal newlines (LF). The pair CLI accepts raw
or LF-normalized source text only when its SHA matches the recorded checksum;
other source changes are refused. Portable source copies retain their original
bytes, with a separate `text_file_sha256` in `source_links.json`.

The prepared corpus includes saved scoring envelopes. This concrete example uses
the same fixed date and approved-relation policy as the full benchmark:

```powershell
python -m scripts.ats_score --resume output/scorer_v2/envelopes/cv_068.json --job output/scorer_v2/envelopes/jd_273.json --ruleset v2 --as-of 2026-09-30 --relations output/scorer_v2/relations.json --resume-source output/extraction_pilot/sources/cv_068.txt --job-source output/extraction_pilot/sources/jd_273.txt --output output/scorer_v2/one_pair_score.json
```

`skills_breakdown` separates group fulfillment, exact-only Core gate fulfillment,
base skills, proficiency bonus and the final score. Its source checksums, review
status and ruleset identity make the calculation traceable.

Extraction callers select a separate v2 prompt/schema/cache identity with
`ExtractionCache(..., ruleset="v2")`. A v1 cache entry is not reused as a v2
annotation. Quote validation checks source spans; reviewers still assess whether
the wording supports Core importance or individual proficiency.

## Prepare and review the 449-document corpus

Run these commands from the repository root. Preparation preserves the original
88 CVs and 361 JDs, sources, split assignments, pilot decisions and original v1
envelopes. It proposes source-backed skill changes and keeps every document's
review status explicit.

The prepared handoff contains 449 documents: 255 have complete machine skill
annotations, 178 partial annotations and 16 missing annotations. All 449 remain
pending human review. "Complete" describes annotation coverage; it does not mean
that the facts are approved or that the document is training gold.

```powershell
python -m scripts.ats_v2_review prepare --as-of 2026-09-30
python -m scripts.ats_v2_review validate
python -m scripts.ats_v2_review export-review
```

Generated annotations live in `output/scorer_v2/corpus.json`. Review artifacts
live in `human-verify/scorer-v2/`: linked source/quote data, before/after values,
CV proficiency, JD requirements, OR choices, semantic relations, ambiguities and
document status. CSV tables, JSON decisions and an XLSX checklist support handoff.
Stable IDs and checksums bind a decision to its exact annotation. Sources outside
skills are recorded separately in the corpus and report.

The portable review folder contains `README.md`, `checklist.xlsx`, `items/`,
`sources/`, `originals/` and `source_links.json`. Keep these siblings together
when transferring or extracting `output/scorer_v2/scorer_v2_review_package.zip`.
The workbook has six sheets and 12,591 unique decision items. `Choices` mirrors
35 JD units for inspection; edit their decisions on `JD requirements`.

Headers are on row 5; `B2` holds the corpus identity. Reviewers fill columns C–F
(`decision`, `reviewer`, `reason`, `replacement`). Preserve IDs, checksums and
identity cells. Column G links the copied source text, R the full proposal JSON,
and S the original document. Display excerpts may be shortened or replace PDF
layout control characters; the linked JSON and source copies preserve the full
evidence.

Fill `decisions.json` with `accept`, `edit`, `reject` or `pending`. Accept/edit/
reject require a reviewer; an edit supplies the complete replacement JSON object.
Empty cells and pending values remain pending. Import and validate:

```powershell
python -m scripts.ats_v2_review import-review --decisions human-verify/scorer-v2/decisions.json
python -m scripts.ats_v2_review validate
```

To import the workbook directly:

```powershell
python -m scripts.ats_v2_review import-review --decisions human-verify/scorer-v2/checklist.xlsx
python -m scripts.ats_v2_review validate
```

CSV decisions are also accepted. Reimporting the same transaction is idempotent;
stale checksums are refused. Regenerate the review package after revisions to
obtain current checksums, after saving any human decisions. Global `--corpus` and `--root` arguments precede the
subcommand when using a different corpus or workspace.

The checklist builder uses a Node environment with `@oai/artifact-tool` available:

```powershell
node scripts/ats_v2_checklist.mjs .
```

It copies source files and writes sheet previews and inspection artifacts under
`output/scorer_v2/checklist/`. It refuses to overwrite an existing workbook.
`--rebuild-fresh` permits rebuilding a generated workbook before human editing;
use it only while preparing a fresh package. The delivered workbook is already
built and ready for review.

Skills-only approval does not approve the full envelope. Training export requires
`split == "train"`, complete skills, approved skills and an approved full envelope:

```powershell
python -m scripts.ats_v2_review export-training
```

Pending or holdout documents are excluded. This command exports eligible data;
it does not train Qwen.

To propose relations with the existing BGE-M3 setup, use its pinned revision:

```powershell
python -m scripts.ats_v2_review propose-relations --model BAAI/bge-m3 --revision 5617a9f61b028005a4858fdac845db406aefb181 --device cpu --top-k 5 --attach
```

Proposals use the training vocabulary and need human review. `--attach` explicitly
adds them to the corpus as pending. A cosine score does not grant matching credit.
The delivered proposals were generated locally on CPU from 993 training-vocabulary
skills: 4,965 candidates, at most five per skill. All are pending and none are
approved, so the official semantic variant currently awards no relation credit.
The standalone versioned relation JSON has `version` and `relations`; each record
contains `id`, `candidate`, `requirement`, `relation` (`near` or `transfer`),
`reason`, `source` and `review_status` (`pending`, `approved`, `rejected`).

## Run the seven fixed silver ablations

Use the same fixed date as the corpus. The complete runner requires all 449
documents and all 1,672 unchanged pair labels:

```powershell
python -m scripts.ats_v2_benchmark --as-of 2026-09-30 --annotation-corpus output/scorer_v2/corpus.json --relations output/scorer_v2/relations.json --output output/scorer_v2/benchmark
```

The order is v1 original, v1 source-scanned proposals, equal v2 group shares, 70/20/10
shares, semantic relations, proficiency/bonus, then full Core gates. Add
`--annotation-corpus`, `--relations`, `--allow-provisional-relations`, repeated
`--variant` or `--output` to select explicit inputs or a subset.

`output/scorer_v2/benchmark/` contains `benchmark_identity.json`, one JSON artifact
per variant, and `report.json` / `report.md`. Reports show the 1,144 Gemini pairs
as the primary diagnostic and the 528 heuristic negatives separately, then
domains and the 24 historical disagreement cases. They retain every failed pair
and exit nonzero on scoring failure. Empty/all-failed subsets have no invented
metrics. Metrics include per-grade precision/recall, confusion matrices,
macro-F1, balanced accuracy, distributions, coverage, latency, grade shifts and
Core/proficiency/semantic usage. The label order is fixed at 0/1/2; heuristic
negatives contain only grade 0, so inspect their per-grade support and matrix.

Identity includes working-tree code hashes, configuration, source/annotation
checksums, relation decisions, selected pairs and the evaluation date. The runner
refuses to mix a stale output identity with a new run. Choose a fresh `--output`
directory after changing inputs or code; matching identities can be reproduced.
These diagnostics use already inspected silver data and do not measure independent
test accuracy or hiring outcomes. The report states review fraction, source
incompleteness and provisional status.

`output/scorer_v2/verification.json` records the final regression run, exact v1
reproduction, artifact identity checks, source/checklist verification and the
pending-workbook import round trip. See
[the handoff ledger](../../knowledge-base/sessions/ats-scorer/scorer_v2_handoff.md)
for the release boundary and remaining human review.

The existing silver runner also supports a v2 sample or full run:

```powershell
python -m scripts.ats_silver_eval --ruleset v2 --annotation-corpus output/scorer_v2/corpus.json --variant full --full --as-of 2026-09-30 --output output/scorer_v2/silver_full
python -m scripts.ats_silver_eval --ruleset v1 --annotation-corpus output/scorer_v2/corpus.json --full --as-of 2026-09-30 --output output/scorer_v2/v1_reviewed
```

Without `--ruleset v2` or `--annotation-corpus`, its original v1 behavior and
output format remain unchanged. The following sections retain that v1 extraction
and evaluation workflow.

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

## Optional JD skill depth: v3

`ats-en-v3` compares explicit skill-specific JD minima with evidenced CV levels.
V1 remains the default. V2 originals and the verified working-tree snapshot are
preserved; v3 outputs use `output/scorer_v3/` and `human-verify/scorer-v3/`.
See the [research record](../research/ats_scorer/README.md) for decisions,
methodology, experiments, item-level changes and limitations.

Use `ScorerConfig(version="ats-en-v3")` through the existing `score_pair` API.
V3 defaults to `depth_policy="full"`. `off` retains v2 arithmetic on equivalent
supported inputs. `score_only` applies depth to allocation/base skills while
retaining legacy bonus and Core gate factors. `full` also requires Advanced/Expert
bonus eligibility to meet any explicit minimum and uses depth in the Core gate.
Depth policies are accepted only for v3.

The v3 document and annotation versions are `resume-v3`, `job-v3` and
`ats-skills-v3`; extraction uses `ats-facts-en-v11-skills-v3-cv-label-rules-asof-inprogress` /
`ats-envelope-v3-no-beginner-v1`. Scoring receipts bind the v3 level policy
`explicit-skill-depth-v3-no-beginner-v1`.
CV extraction on v3 needs the reference date for past-dated degrees:
`ExtractionCache(..., ruleset="v3", as_of="YYYY-MM-DD")`. Without it a CV cache miss raises, because
every Qwen training prompt carries `as_of_date`; the date is part of the cache key for resumes only.
JD requirement string `options` remain compatible with v2. Add `option_details`
to identify each alternative's minimum and evidence, for example:

```json
{
  "id": "python-or-java",
  "options": ["Python", "Java"],
  "importance": "standard",
  "quote": "Advanced Python or Intermediate Java",
  "option_details": [
    {"name": "Python", "minimum_proficiency": "advanced", "raw_proficiency": "Advanced", "proficiency_quote": "Advanced Python"},
    {"name": "Java", "minimum_proficiency": "intermediate", "raw_proficiency": "Intermediate", "proficiency_quote": "Intermediate Java"}
  ]
}
```

Explicit scoring levels are Intermediate, Advanced and Expert. `normal` is
unspecified. Literal Beginner wording maps to normal in v3; `raw_proficiency` and
the proficiency quote remain visible. It receives full existing credit, no bonus
and no depth gap. Gap credits are 1/.75/.50 for zero/one/two levels below the
minimum. Unknown CV depth preserves credit and
gate behavior and is reported separately. Related-skill depth cannot establish
the required skill's depth or enter the Core gate. Titles require body review;
years and specialization do not imply Advanced. Skill-specific years are stored
as `option_details[].experience` with `min_years`, `max_years` and a literal
`quote`, and remain unscored. Importance needs its own independent evidence.

The Core boundaries remain unrounded: below .7 caps skills at 49; below .5 forces
Mismatch. Exactly .5 does not force Mismatch, and exactly .7 passes. Explanations
include required/candidate levels, quotes, gap, depth credit, effective
fulfillment, bonus eligibility and unknown/unsupported depth status.

```powershell
python -m scripts.ats_score --resume cv.json --job jd.json --ruleset v3 --depth-policy full --as-of 2026-09-30 --output output/scorer_v3/pair.json
python -m scripts.ats_v3_review validate
python -m scripts.ats_v3_review import-review --decisions human-verify/scorer-v3/imported_approvals.json
python -m scripts.ats_v3_benchmark --as-of 2026-09-30 --output output/scorer_v3/benchmark
python -m scripts.ats_silver_eval --ruleset v3 --depth-policy full --full --as-of 2026-09-30 --output output/scorer_v3/silver_full
```

The supplied corpus and baseline already exist; preparation refuses to overwrite
them. Imported sample approvals are bound to actual IDs, source/item hashes and
literal quotes. A repeated transaction is a no-op; stale decisions and unsupported
edits fail before mutation. Samples 1/2/3/4/6/7/8 have direct item approvals; derived
changes are recorded separately. Sample 2 is approved normal, and sample 6 is
approved Golang OR JavaScript. Other exceptions remain pending
in batches of 5–10 under `human-verify/scorer-v3/batches/`. Full envelopes remain
pending. Inspect the linked source and use current `review_items.json` checksums
for new decisions; the original approval exchange preserves its imported identity.

The four-way report preserves all 1,672 labels, with 1,144 Gemini pairs and 528
heuristic negatives reported separately. Complete metrics, confusion matrices,
score/grade changes, domains, latency, coverage, gate causes and all 24 historical
cases are retained. The experiment identity includes the full relation-review
table; only approved relations enter runtime scoring. A changed code/input/policy
identity requires a fresh output directory. These are diagnostics on inspected
silver data. Sparse explicit depth and unknown candidate levels limit conclusions;
the gap credits are project design choices. See the saved
[results](../research/ats_scorer/results.md) and
[verification receipt](../research/ats_scorer/verification.json).

### V3 validation correction

The v3 runtime and extraction cache identify the current evidence policy as `explicit-skill-depth-v3`. Depth-enabled OR choices follow the allocator's lexicographic priorities. Negated or training-other clauses cannot establish positive depth, even when a short quote omits that context. Rejected CV details cannot earn credit; if no supported evidence remains, the gate is unknown. Depth-off retains legacy option selection on equivalent supported inputs. Explicit item approval can resolve a title choice, as for sample 6; title-only depth still requires body evidence.

Use a fresh benchmark output directory after code or corpus changes. The verified current run is `output/scorer_v3/jd012_decision_2026-10-11/benchmark`; exact commands, the direct-human OR approval and fresh frozen/current reproduction paths are in [the JD 012 decision record](../research/ats_scorer/jd012_decision_2026-10-11.md). Prior benchmark directories remain immutable historical artifacts.


## Current optional v3 policy (2026-10-11)

V1 remains the API and CLI default. For `ScorerConfig(version="ats-en-v3")`, depth defaults to `full` and Preferred defaults to `separate`. Preferred coverage and its matched/missing skills are reported in the breakdown; they do not add score, bonus or Core gate credit. When a JD contains no Core/Standard requirements, Preferred supply base-score targets without bonus/gate. Required skills are allocated first and optional reporting uses remaining CV skills. Incomplete annotations retain conservative neutral handling.

Both `scripts/ats_score.py` and `scripts/ats_silver_eval.py` accept `--preferred-policy separate|legacy_share` for v3. `legacy_share` retains the previous 10% Preferred share and joint allocation as an explicit comparison; it is not the default. `scripts/ats_v3_benchmark.py --preferred-policies separate legacy_share` runs both families, and its source-bound reference options isolate annotation from policy changes. Never combine artifacts with different code, corpus, configuration, relation or evaluation-date identities.

The latest source-bound JD 012 four-way OR requirement is Preferred after checking its `Nice to Have` heading; the earlier Standard relay was an error and remains preserved in the research history. Final follow-up: 122 targeted / 273 full ATS tests, zero historical v1/v2 drift, and 30 score / 6 grade changes versus today's prior JD 012 run, all due to Preferred policy rather than the importance correction alone. [Current handoff](../research/ats_scorer/handoff.md) and [process, limitations and receipts](../research/ats_scorer/preferred_policy_2026-10-11.md) are authoritative. Core/Standard weights remain unchanged pending the reported Core5/Standard2 question.
