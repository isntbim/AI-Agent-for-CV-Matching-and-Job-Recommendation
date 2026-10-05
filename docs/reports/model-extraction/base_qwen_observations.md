# Base Qwen2.5 extraction observations

Date: 2026-09-30. These observations are development smoke-test findings for
later extraction training, not ATS accuracy results.

## Model and run

- Model: unmodified `qwen2.5:7b-instruct`, 7.6B parameters, Q4_K_M.
- Ollama digest: `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
- Context: 32,768; runtime reported the model fully loaded on the GPU.
- Sampling: temperature 0, maximum output 8,192 tokens.
- Dataset: pinned resume-job-description-fit revision
  `08978e21714984bb417547d2c0f9b477f5298163`.
- Sample: shortest development CV/JD pair for each of No Fit, Potential Fit and
  Good Fit; three pairs total. Labels were used for sampling only, never extraction.
- Final extraction prompt: `ats-facts-en-v4`, with JSON-schema-constrained output.
- Result: 0/3 pairs accepted. Each stopped on CV evidence validation, before JD
  extraction or scoring. No scores or fitting metrics were fabricated.

The local service was started at `http://127.0.0.1:11434`. It remains available
for subsequent runs. Ignored artifacts live in `output/ats/`.

## Observed model-output issues

| Issue | Observation | Action taken | Candidate training objective |
|---|---|---|---|
| Incorrect evidence container | Initial responses used a flat list instead of a dictionary keyed by dimension. | Added schema-constrained generation. | Produce the exact structured envelope when constraints are unavailable; retain schema constraints in production. |
| Incorrect evidence keys | Responses placed factual fields such as completed_degrees and country inside evidence. | Restricted the schema to seven evidence keys and clarified sibling field placement. | Distinguish extracted values from the source evidence supporting a dimension. |
| Unsupported domain vocabulary | A response used a domain outside the configured project vocabulary. | Constrained domain output to the versioned enum. | Map occupational evidence to the supported professional families; abstain when ambiguous. |
| Nonliteral evidence quotations | After structure/domain constraints, all three CVs still emitted an evidence quote absent from the source. | Rejected the documents. Whitespace-only differences may be aligned back to the original source span; changed words or punctuation remain failures. | Copy exact short source spans; preserve dates and wording; omit evidence when unsupported. |

Latest failure details (the quoted strings are rejected model output, not verified
source facts):

| Label used for sampling | CV document hash | Failed field | Rejected quotation |
|---|---|---|---|
| No Fit | `85c293a56702fe894365c3b163bcef6c60f5e94cdd7e230703f260ce5107f9b2` | education | `Expected in12007` |
| Potential Fit | `c9f3b24e69d9bbf160f03d7a827d759ef6bc7ba19575db4abc9ad7ef6415544a` | experience | `2015 to Present International Tax Accountant` |
| Good Fit | `3f37a16fe43367d0bdd6fab4b3f550d18490c67f246e99ea71ede84afe186449` | experience | `12/2015 to Present` |

Exact responses and extraction identities are retained in the ignored cache's
`.error.json` records. The dated smoke report and `training_observations.json`
link these failures to their source documents and model digest.

## Integration issue corrected separately

While tightening the schema, an adapter-side JSON-schema reference temporarily
caused a Python KeyError before inference. Inline evidence schemas corrected it.
This was a code defect and must not be added as a model training failure.

## Using these observations for later training

Review each source document and create a corrected structured extraction plus
verified source spans. Failed responses are examples for error analysis; they
are not supervised targets. Check missing/ambiguous dates, completed versus
ongoing education, and source formatting while making the correction.

Use only development documents for training and prompt work. Preserve the frozen
test CVs/JDs for comparison of base and fine-tuned models. Evaluate extraction
validity, source-span fidelity and field correctness before comparing downstream
fit grades. Three deliberately short examples cannot establish general failure
rates or demonstrate that fine-tuning alone will solve the observed problems.
