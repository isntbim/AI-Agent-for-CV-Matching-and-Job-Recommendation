# Documentation index

This folder contains the published guides, explanations, references and model-extraction observations listed below. Additional local reports stay ignored. Generated run outputs remain in `output/`; private plans and handoffs remain in `knowledge-base/`.

## Start here

- [Extraction and retrieval run instructions](guides/extraction_retrieval.md)
- [ATS scorer guide](guides/ats_scorer.md), [scoring explanation](explanations/scorer_explanation.md), and [references](reference/scorer_references.md)
- [Base Qwen observations](reports/model-extraction/base_qwen_observations.md) and [Gemini testing observations](reports/model-extraction/gemini_smoke_observations.md)

## Folder map

| Folder | Contents |
|---|---|
| `guides/` | Setup, usage, and project guides |
| `explanations/` | How scoring works and its design rationale |
| `reference/` | Scoring sources, schema snapshots, and report templates |
| `reports/extraction-retrieval/` | Detailed session implementation report |
| `reports/model-extraction/` | Qwen and Gemini extraction observations |
| `reports/data-pipeline/` | Data and pipeline report in Markdown and Word |
| `reports/capstone-review-1/` | Review drafts and progress reports; earlier numbered versions in `archive/` |
| `reports/mentor-feedback/` | Mentor feedback reports; earlier Word version in `archive/` |
| `reports/presentations/` | Slide source and presentation files |
| `assets/images/`, `assets/diagrams/` | Supporting figures and diagram source files |
| `archive/` | Original documentation archive, preserved unchanged |

Report drafts and schema/diagram snapshots retain their original status; folder placement does not establish approval or make them runtime specifications. Runtime extraction schemas are defined in the source modules.

## Local evidence and private notes

- Extraction pilot results: `output/extraction_pilot/report.md`
- Local ranking benchmark: `output/retrieval/benchmark_v2/report.md`
- Azure verification: `output/retrieval/azure_v2/report.md`
- Session reports: `docs/reports/extraction-retrieval/`
- Private planning and session notes: `knowledge-base/`

These paths refer to ignored files in a local workspace and are not included in a fresh Git checkout. The folder map also describes local documentation areas; their presence does not mean their contents are published. Create reusable documentation here and keep run-specific data and machine receipts in `output/`. Keep credentials, private planning, and session handoffs in `knowledge-base/`. Only this index and the selected guides, explanation, references and model-observation documents are eligible for Git.
