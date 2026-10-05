# Explanation: the Phase 2 ATS scorer

**Audience:** project developers and reviewers who need to understand how a CV–JD score is produced and what an improvement would change.

**Updated:** 2026-10-01.

**Current baseline:** `ats-en-v1`, with the full silver comparison evaluated as of 2026-09-30.

This document explains the current implementation and its limitations, then proposes weighted skill matching. The proposal has not been implemented or validated. Implementation commands are maintained separately in [Running and evaluating the ATS scorer](../guides/ats_scorer.md).

## Contents

1. [Purpose and implementation status](#1-purpose-and-implementation-status)
2. [From CV/JD information to scorer inputs](#2-from-cvjd-information-to-scorer-inputs)
3. [How the current scorer calculates results](#3-how-the-current-scorer-calculates-results)
4. [Module responsibilities](#4-module-responsibilities)
5. [GitHub references and project decisions](#5-github-references-and-project-decisions)
6. [Current limitations and evaluation findings](#6-current-limitations-and-evaluation-findings)
7. [Proposal: weighted skill matching](#7-proposal-weighted-skill-matching)

## 1. Purpose and implementation status

The scorer evaluates a structured candidate CV against a structured job description. It returns seven component scores, a weighted total, a grade, evidence coverage, and matched/missing skills. Its score calculations are deterministic Python rules: the same inputs, configuration, code version and evaluation date produce the same result.

The implementation currently exists as local scoring modules and evaluation runners. The recorded tests demonstrate local behavior and offline comparison with silver labels. Web deployment and satisfactory production extraction quality remain separate Phase 2 integration and validation work.

### Two different kinds of weight

The current system does **not** give every score the same weight:

| Level | Current behavior |
|---|---|
| Seven ATS dimensions | Different weights: skills 30%, title 20%, experience 15%, education 10%, industry 10%, location 8%, salary 7% |
| Individual required skills inside the skills dimension | Each unique canonical skill contributes equally |
| Candidate skill proficiency | Stored in the CV schema, but does not affect matching credit |
| Core versus Standard required skills | No such importance category is currently implemented |

Weighted skill matching would change the second row. It would not automatically change the skills dimension's 30% contribution to the overall score.

### Qwen's place in the intended deployment flow

The intended production flow is raw CV/JD text, Qwen extraction, validated structured inputs, normalization, and deterministic scoring. A later explanation step may use Qwen to turn the calculated scores and supporting evidence into readable feedback. That explanation should preserve the calculated results and stay within the supplied evidence.

Qwen extraction can influence scores because missing or incorrectly interpreted input facts affect the rules. Deterministic arithmetic therefore does not guarantee correct inputs or a valid suitability judgment.

**Gemini is a testing backend only. It is outside the deployment workflow.** The existing Gemini-labelled silver dataset is also an offline comparison source, rather than an input to the scoring calculation.

## 2. From CV/JD information to scorer inputs

### `document` preserves parsed information

[`ResumeInput` and `JobInput`](../../src/scoring/models.py) each contain a `document` and `facts`:

| Input | `document` model | Examples of retained information |
|---|---|---|
| `ResumeInput` | [`ResumeSchema`](../../src/parser/schemas.py) | Basics, work history, education, grouped skills, projects, certificates, languages and summary |
| `JobInput` | [`JobDescriptionSchema`](../../src/parser/job_schemas.py) | Identifiers, titles, company, location, seniority, skills, description, compensation, benefits and metadata |

The wrapper contains the declared parser schema. It preserves fields such as a skill group's name and level even when the current rules do not use them. Arbitrary extra metadata is not guaranteed to remain in the validated document. The silver evaluation runner handles CV identifiers and pair-domain metadata separately.

The following is an illustrative fragment of the existing CV structure:

```json
{
  "skills": [
    {
      "name": "Basic Knowledge",
      "level": "Basic knowledge",
      "keywords": ["HTTP", "OOP", "SOLID", "Design patterns"]
    },
    {
      "name": "Frontend",
      "level": null,
      "keywords": ["HTML", "CSS", "JavaScript", "React"]
    },
    {
      "name": "Development and Operations",
      "level": null,
      "keywords": ["Linux", "Git", "Docker", "Kubernetes", "AWS"]
    }
  ]
}
```

`name` is a group heading. `level` is a stated proficiency. `keywords` contains the individual skills. The scorer collects the keywords across groups and normalizes them; it does not need to classify a heading as technical or soft to perform that comparison. A matching SOLID keyword currently receives the same matching credit whether its group says Basic or Advanced.

### `Facts` adds interpretation and supporting evidence

`Facts` is a companion to the parsed document. It stores information useful to the rules, including completed degrees, the job's minimum degree, professional domains, explicit country/currency, remote eligibility and explicit absence of experience/education requirements. It also groups supporting quotes by dimension.

For education, a document entry might say “Bachelor of Science, expected graduation 2027.” The entry remains in `document.education`, while `facts.completed_degrees` should not count it as completed. A completed degree can be represented as `bachelor`, supported by the relevant source information.

For skills, the actual skill list remains in `document.skills`. The following illustrates supporting evidence; it is not a complete scorer input:

```json
{
  "evidence": {
    "skills": [
      {"quote": "Skills: Python, SQL, Excel"}
    ]
  }
}
```

`facts.has("skills")` checks whether the dimension has evidence. It does not prove every listed skill is correctly extracted. The rules also need usable fields, and quotation validation checks whether quoted text occurs in the source. Neither check alone establishes semantic correctness or complete extraction.

### Extraction and the offline silver adapter have different provenance

The Qwen extraction path requests schema-conforming JSON and validates evidence quotations against the supplied text. Equivalent whitespace can be aligned with the original source span; changed wording is rejected. Failed extraction is reported as a failure, rather than assigned an invented score. Accepted extraction results can be cached with extraction identity and prompt/schema versions.

The full silver comparison used [`silver.py`](../../src/scoring/silver.py) to construct inputs from stored structured CV/JD records. Its evidence consists of structured JSON values. Those values were not newly verified against original PDFs or raw JD text. This adapter also applies the existing JD title/seniority skill-tag filter. CV-domain and JD-category metadata are retained for comparison grouping, but are not supplied as scoring facts.

## 3. How the current scorer calculates results

### 3.1 Normalization

[`normalization.py`](../../src/scoring/normalization.py) standardizes text and applies explicit mappings. Known skill aliases such as Python3/Python 3 and JS/JavaScript map to canonical names. Duplicate canonical skills count once. Unknown names can still match when their normalized strings are identical; unlisted synonyms are not automatically inferred.

Role normalization removes recognized seniority words and looks up canonical roles and families. For example, Software Developer can map to Software Engineer in the software family. The vocabulary is finite, and title qualifiers can prevent an exact dictionary match. Country aliases similarly allow equivalent names such as USA and United States to be compared.

The current baseline performs dictionary/set comparisons rather than semantic similarity scoring or reasoning about experience narratives.

### 3.2 Component rules

[`score_pair()` in `rules.py`](../../src/scoring/rules.py) calculates the following components. Scores are rounded to integers and bounded to 0–100.

| Dimension | Current calculation | Important boundary |
|---|---|---|
| Skills | `100 × matched targets / total targets`. Targets are unique canonical required skills; preferred skills are targets only when no required skills exist. | Matching requires skill evidence on both sides. Preferred coverage is reported separately and does not add a bonus to a required-skill score. Proficiency is ignored. |
| Title | Same canonical role 100; different roles in the same recognized family 70; recognized different families 0; insufficient mapping/evidence 50. | The CV headline takes precedence. Otherwise, use the latest dated work title. |
| Experience | Relevant months divided by required months, capped at 100. Explicitly no experience required gives 100. | Relevant work uses the JD role/family. Overlapping intervals count once. Ambiguous roles or missing/invalid dates can make duration unknown. |
| Education | Highest completed degree meets/exceeds minimum: 100; one rank below: 50; two or more below: 0. Explicitly no education required gives 100. | Degree order is high school, associate, bachelor, master, doctorate. Field of study and equivalent-experience substitutions are not scored. |
| Industry | `100 × shared professional domains / identified JD professional domains`. Unknown gives 50. | The `industry` key means professional alignment, such as software or finance, rather than employer sector. |
| Location | Explicit compatible country/city or remote eligibility: 100; explicit incompatibility: 0; incomplete constraints: 50. | Remote does not automatically imply worldwide eligibility. Distance, relocation willingness and personal preferences are not currently modeled. |
| Salary | Negotiable/unknown/incomparable currency: 75. Explicit expectation within JD maximum: 100. Above maximum: `100 × JD maximum / expectation`. | Candidate salary preference is separate input. Monthly/annual values are normalized; currency conversion is not performed. |

### 3.3 Why all required skills contribute equally today

Suppose the JD requires React, Docker, MySQL and Python, while the CV demonstrates React, Docker and MySQL. With sufficient evidence:

```text
Unique required skills: 4
Matched required skills: 3
Skills score: 100 × 3 / 4 = 75
```

Each match contributes one unit. The current formula does not give Docker more credit because it seems harder, or reduce credit for a skill marked Basic. Group names such as Frontend and Backend also do not affect this calculation.

The missing-skill list describes missing matches in the supplied skill data. It does not establish that the candidate lacks a skill in real life; the extractor or source CV may have omitted it.

### 3.4 Employment dates and reproducibility

Experience uses the union of relevant dated work intervals. January 2020–January 2023 and January 2022–January 2024 produce 48 months, rather than 60 months. Months are calculated as end-month minus start-month.

`Present` uses the caller's explicit `as_of_date`. Year-only employment dates, missing end dates, invalid/future dates and uncertain role relevance can produce an unknown result. A missing end date is not implicitly Present. A fully recognized unrelated work history contributes zero relevant months; an empty history can be incomplete extraction and stays unknown.

The fixed date matters for reproducibility. A historical CV that still says Present may overstate current tenure when evaluated years later.

### 3.5 Missing information, evidence and coverage

Unknown dimensions receive 50, except salary, which receives 75. Their original weights remain in the total. A demonstrated mismatch can receive 0; missing information should not be treated as a demonstrated mismatch.

Each dimension result records its score, support flag, calculation basis, rule ID and evidence. The support flag depends on whether the rule has sufficient usable facts, not merely whether an evidence object exists.

Coverage is the sum of the weights of supported dimensions. If skills and title alone are supported, coverage is 30% + 20% = 50%. Coverage is not a confidence probability and does not independently verify extraction accuracy.

An all-unknown input receives seven defaults whose weighted total rounds to **51.8**, with **zero coverage** and a **Partial** grade. The current grade calculation has no minimum-coverage gate. Scores and grades therefore need to be presented with coverage and missing-evidence information.

### 3.6 Overall score and grade

Weights are defined in [`match_schemas.py`](../../src/parser/match_schemas.py):

```text
Total = skills × 0.30 + title × 0.20 + experience × 0.15
      + education × 0.10 + industry × 0.10
      + location × 0.08 + salary × 0.07
```

The total is rounded to one decimal. Default grade rules are:

| Grade | Condition |
|---|---|
| Strong (`2`) | Total at least 75, skills at least 70 and title at least 70 |
| Partial (`1`) | Total at least 50, but Strong conditions are not satisfied |
| Mismatch (`0`) | Total below 50 |

A hypothetical component result illustrates both weighted aggregation and unknown defaults:

| Component | Score | Weighted contribution | Supported? |
|---|---:|---:|---|
| Skills | 75 | 22.50 | Yes |
| Title | 70 | 14.00 | Yes |
| Experience | 83 | 12.45 | Yes |
| Education | 100 | 10.00 | Yes |
| Industry | 100 | 10.00 | Yes |
| Location | 50 | 4.00 | No: unknown |
| Salary | 75 | 5.25 | No: negotiable default |
| **Total** | **78.2** | **78.20** | **85% coverage** |

This receives Strong under the current gates. Conversely, an unknown title scores 50 and prevents Strong, even if the total exceeds 75. The grade is a rubric classification, not a probability of recruitment success.

## 4. Module responsibilities

| Module | Role in Phase 2 |
|---|---|
| [`models.py`](../../src/scoring/models.py) | Input/output contracts, evidence, preferences, configuration, grade and coverage helpers |
| [`normalization.py`](../../src/scoring/normalization.py) | Canonical skills, roles/families and countries |
| [`rules.py`](../../src/scoring/rules.py) | Seven component calculations and assembly of the final result |
| [`extraction.py`](../../src/scoring/extraction.py) | Preprocessing: extraction requests, validation and caches, including local model adapters |

The first three form the core scoring system. Extraction prepares its inputs. Natural-language explanation is a further consumer of the scorer output, rather than a calculation inside `score_pair()`.

Other modules serve offline/testing purposes: `gemini.py` supplies the testing backend; `silver.py` adapts stored silver records; `dataset.py` prepares evaluation data; `evaluation.py` measures agreement and supports development calibration. Their presence in the repository does not make Gemini part of production deployment.

The scorer result includes component scores, overall score, grade, coverage, matched/missing skills, preferred coverage, per-dimension reasons/evidence/rule IDs, configuration and evaluation date. These fields allow a web interface or explanation generator to show the basis and uncertainty of a result.

## 5. GitHub references and project decisions

The implementation references selected methods from [Sara12-2/ResumeMatch_AI](https://github.com/Sara12-2/ResumeMatch_AI/tree/a30c9e32c0d801b0b2a48f3b8883245ad67636b7), pinned at commit `a30c9e32c0d801b0b2a48f3b8883245ad67636b7`. No upstream source code or taxonomy assets have been copied or imported. Attribution is maintained in [scorer_references.md](../reference/scorer_references.md).

| Dimension | Exact GitHub code used as a method reference | Adopted idea and reason | Project-specific changes |
|---|---|---|---|
| Skills (`R1`) | [`_get_matcher()`, `_build_matcher()`, `extract_skills()`, `analyze_skill_gap()` in `src/skill_extractor.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/skill_extractor.py) | Canonical aliases and set intersection/difference make matched/missing skills inspectable. | Our dictionary, structured required/preferred handling, evidence defaults and scoring targets; no spaCy PhraseMatcher or upstream taxonomy import. |
| Education (`R2`) | [`highest_degree_level()` in `src/skill_extractor.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/skill_extractor.py) and [`EnhancedATSScore._compute_education_match()` in `src/ats_score.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/ats_score.py) | Degree ranking and attainment comparison are easier to audit than whole-document similarity. | Completed-degree/minimum-requirement handling and 100/50/0 scores; no upstream semantic score or degree-match floor. |
| Industry (`R3`) | [`detect_domains()`, `_get_domain_matcher()` in `src/skill_extractor.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/skill_extractor.py) and [`EnhancedATSScore._compute_project_relevance()` in `src/ats_score.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/ats_score.py) | Canonical domain labels and overlap give an inspectable alignment method. | Our professional families and JD-domain coverage formula; no project-relevance floor. |

Title (`P1`), experience (`P2`), location (`P3`) and salary (`P4`) are project rules. The seven dimension weights, neutral defaults, evidence coverage, grade thresholds and Strong gates are also project decisions. Numeric rules are hypotheses requiring evaluation; a GitHub reference does not validate their suitability for this project's jobs and candidates.

## 6. Current limitations and evaluation findings

### Full silver comparison

The 2026-09-30 audit scored all **1,672 pairs**, involving **88 CVs and 353 JDs**, with zero scoring failures and zero model/API calls. It used stored structured records, rather than a fresh Qwen extraction run. No weights, thresholds or mappings were tuned using this comparison.

The labels comprise 1,144 Gemini-labelled pairs and 528 heuristic negatives. Heuristic negative subscores were generated from ranges; they are a different comparison source from Gemini's judgments.

| Comparison source | Pairs | Grade agreement | Always-majority baseline | Total-score MAE | Spearman rank correlation |
|---|---:|---:|---:|---:|---:|
| Gemini only | 1,144 | 59.3% | 57.5% | 14.67/100 | 0.208 |
| Heuristic negatives | 528 | 96.8% | 100.0% | 15.36/100 | -0.015 |
| Combined | 1,672 | 71.1% | 70.9% | 14.89/100 | 0.211 |

Combined agreement is only slightly above always predicting Mismatch. Macro F1 is 0.316 on Gemini pairs and 0.346 overall. The system produced 1,543 Mismatch, 129 Partial and zero Strong grades, while silver labels contain 1,186 Mismatch, 452 Partial and 34 Strong.

On Gemini-labelled pairs, mean scores by silver grade are:

| Silver grade | Pairs | Gemini mean | Runtime mean |
|---|---:|---:|---:|
| Mismatch | 658 | 33.2 | 39.4 |
| Partial | 452 | 61.5 | 41.5 |
| Strong | 34 | 79.3 | 45.1 |

The runtime means increase with silver grade, but the separation is small and the runtime totals are compressed. Zero Strong predictions are consistent with those low totals and limited title/skill support.

### What needs investigation

- **Title vocabulary:** only 12/1,144 Gemini pairs have supported title scores, approximately 1.0%. Most titles use neutral defaults. Industry support is 23/1,144, approximately 2.0%; both dimensions depend partly on recognized role families.
- **Skill representation:** 833/1,144 Gemini pairs receive zero skills scores. Literal coverage, unrecognized aliases, missing extracted skills, multilingual or corrupted tags, and genuine mismatch can all contribute. A zero score alone does not identify which cause applies.
- **Current evidence coverage:** Gemini-only experience support is about 10.8%. Many results therefore combine explicit matches with neutral assumptions.
- **Equal skill contribution:** missing a central requirement has the same effect as missing any other required skill. Whether weighting helps requires suitable importance annotations and comparison data.
- **Proficiency:** a Basic claim and an Advanced claim currently earn identical presence credit. This can overstate suitability when a JD requires a specific depth.
- **Correlated dimensions:** title, professional domain and relevant experience reuse role-family recognition, so these are not independent signals.
- **Source quality:** evidence extracted from text or adapted from JSON can be incomplete or incorrectly interpreted. Qwen's extraction behavior must be evaluated separately from score arithmetic.

The base Qwen smoke tests produced extraction validation failures, including nonliteral evidence quotes. Training and prompt work should address schema compliance, quotation fidelity, uncertainty handling and extraction completeness. These observations do not establish successful end-to-end Qwen scoring. See [base_qwen_observations.md](../reports/model-extraction/base_qwen_observations.md).

The silver audit measures agreement with Gemini and heuristics. It is not held-out validation against observed hiring decisions or an independent review, and does not identify which scorer is correct. The 88 CVs and repeated JDs also make pair results dependent. The small balanced sample previously discussed should not be used as the full population estimate.

Local audit artifacts, when generated, are stored under `output/ats/silver_full/eda/`: `report.html` (interactive EDA), `report.md` (Markdown EDA), `pair_comparison.csv` (complete paired scores), and `summary.json` (recorded summary). These generated files are ignored by Git and will not be available in a fresh clone until regenerated.

## 7. Proposal: weighted skill matching

**Status: proposal only.** The current scorer, schemas and extractor do not implement Core/Standard importance or proficiency-adjusted matching. The design below is a candidate improvement, not an approved numerical rubric or a demonstrated performance gain.

### 7.1 Weight job importance, not universal difficulty

A skill matters because of the job's requirements. Kubernetes can be central to an infrastructure role and irrelevant to another role. Communication can be central to a customer-facing role. Technical versus soft classification does not determine importance.

Three separate concepts should remain distinguishable:

| Concept | Question | Proposal stage |
|---|---|---|
| Job importance | How important is this skill to this JD? | First weighted-matching proposal |
| Candidate proficiency | Does the candidate demonstrate sufficient depth? | Separate later proposal |
| Global difficulty | Is this skill generally difficult to learn? | No universal difficulty scale selected |

### 7.2 Proposed categories and initial weights

| Skill requirement | Candidate policy | Illustrative numerical weight |
|---|---|---:|
| Core required | Explicitly identified as essential/central under a documented policy | 5 |
| Standard required | Other required skills, including uncertain importance | 2 |
| Preferred | Retain in the existing preferred list and report separate coverage | Outside the required-score denominator |

The 5:2 ratio is an example, not a validated setting. It should be versioned and evaluated. When a JD has no required skills, the current preferred-only fallback needs an explicit backward-compatible policy; a proposed first implementation would keep equal weighting of those preferred targets.

“Required” establishes that a skill is expected; it does not automatically distinguish it as Core relative to every other required skill. If a JD labels every skill essential, the extractor should reflect the source rather than force a ranking. Equal importance categories then give equal matching credit.

### 7.3 Proposed JD schema and evidence

Importance belongs to individual JD requirements. The CV groups remain unchanged. The following is a proposed fragment, not valid current-schema output:

```json
{
  "required": [
    {
      "name": "Kubernetes",
      "importance": "core",
      "importance_evidence": "Operating Kubernetes clusters is essential to this role."
    },
    {
      "name": "Git",
      "importance": "standard"
    },
    {
      "name": "Communication",
      "importance": "standard"
    }
  ]
}
```

Implementation would extend the JD skill model with a constrained importance category and define where per-skill importance evidence lives, either alongside the requirement or in a structured Facts extension. Importance evidence must identify the source requirement it supports. A single generic skill-evidence quote is insufficient to justify a separate Core claim for every skill.

For older records with no importance, every required skill would default to Standard. Because all weights would then be equal, the weighted formula would reproduce the existing count-based score. Existing missing-evidence defaults should remain explicit and unchanged in the first comparison.

### 7.4 Qwen's proposed classification role

Qwen would extract the skill, its required/preferred status, proposed importance and supporting source text. Code would validate the category/evidence structure and translate categories into configured numerical weights. Qwen would not invent numerical weights or alter the calculated ATS score.

The first policy should emphasize explicit source statements. Ambiguous importance defaults to Standard and is recorded as defaulted. Classifying importance from broad role context is a possible later extension that needs its own annotation policy and evaluation.

Importance should be assigned using the JD independently of the candidate's skill coverage. Otherwise, the same job could receive different weights for different applicants. Extracting and caching the JD once per version helps enforce consistency. Neither silver grades nor teacher subscores should enter that classification request.

Source-quote validation establishes that a quote exists. It does not prove that the quote makes a skill Core. Category correctness therefore needs separate review and evaluation. Since the base Qwen model already exhibited evidence errors, importance extraction is a new task to evaluate and potentially train; it should not be assumed reliable from the model name alone.

### 7.5 Proposed weighted calculation

For canonical target skills with positive weights:

```text
Skills score = 100 × sum(weights of matched target skills)
                   / sum(weights of all target skills)
```

Using the proposed example, a CV matches Git and Communication but misses Kubernetes:

```text
Equal-weight score:     100 × 2 / 3 = 67 after component rounding
Weighted score:        100 × (2 + 2) / (5 + 2 + 2) = 44 after rounding
```

Missing Kubernetes has more effect because this JD identifies it as Core. The skills dimension would still contribute 30% to the overall total unless a separate change is proposed and evaluated.

Normalization must preserve importance metadata. A plain set of names is insufficient for weighted targets. A canonical weight map can retain it:

```python
# Illustrative intermediate representation for the proposal.
required_weights = {
    "kubernetes": 5,
    "git": 2,
    "communication": 2,
}
```

Aliases should resolve before aggregation. Duplicate requirements must count once. A candidate first policy is to retain the highest supported importance for a canonical skill and report conflicting annotations for inspection, rather than add duplicate weights. Unsupported Core claims should follow an explicit validation/default policy.

An empty target set must follow the existing unknown/preferred fallback policy rather than divide by zero. Configured weights must be positive. Equal-weight, missing-evidence and alias behaviors should be tested before comparing model quality.

### 7.6 Implementation responsibilities

| Area | Proposed change |
|---|---|
| `src/parser/job_schemas.py` | Add per-skill importance fields and validation/defaults |
| `src/scoring/models.py` | Define versioned importance-weight configuration and any Facts/output extensions |
| Normalization and `rules.py` | Preserve weights through canonicalization, deduplicate, and replace count coverage with weighted coverage |
| `extraction.py` and its prompt/schema contract | Request importance and per-skill evidence; validate and record uncertain/defaulted classifications |
| Cache identity/versioning | Increment relevant prompt, schema and rule versions so old extraction/configuration is not mistaken for the new design |
| Output/web consumers | Show importance, weight, match status, evidence, and contributions for individual skills |
| Offline evaluation | Compare equal and weighted rules on the same annotated inputs, keeping teacher scores out of the calculations |

Gemini can remain a testing option for comparisons; the proposed deployed extractor remains Qwen. Existing silver records need importance annotations before they can exercise the new weighting meaningfully. Simply defaulting every requirement to Standard preserves the old score and provides no test of importance classification.

### 7.7 Evaluation before adoption

The first comparison should isolate importance weighting from other changes:

1. Establish an annotation policy and review JD skill-importance/evidence examples. Label importance independently of candidate matches and silver scores.
2. Measure Qwen's skill extraction, required/preferred classification, importance classification, evidence fidelity and failure rate separately. Record ambiguous cases for later training.
3. Verify calculation contracts: equal weights reproduce the baseline; aliases/duplicates count once; missing Core hurts more than missing Standard; unchanged facts preserve other dimensions; empty/missing data remains explicit.
4. Compare equal and weighted rules on the same structured inputs with fixed non-skill rules and fixed initial grade thresholds. This isolates the effect of importance weighting.
5. Tune any category weights or thresholds on development data only. Keep CV/JD identities isolated across development and a separately reserved test set, and avoid treating the already inspected silver audit as independent validation.
6. Report grade macro F1, per-grade recall, majority baselines, continuous-score/rank agreement, coverage, extraction failures and representative errors. Agreement with Gemini remains teacher agreement; independently reviewed suitability or observed outcomes are needed for stronger real-world claims.

Weighted matching will not fix missing skills or unrecognized titles. Those problems need separate diagnosis and versioned changes so their effects can be distinguished. With the current Strong title gate, improving skill weighting alone may still leave suitable pairs unable to reach Strong.

### 7.8 Separate later proposals

**Proficiency-aware matching:** candidate levels such as Basic and Advanced could be compared with explicit JD depth requirements. This requires a consistent proficiency scale, source evidence and an agreed partial-credit policy. No scale or factor has been selected. A self-reported level alone does not prove production capability.

**Insufficient-evidence presentation:** a later web integration could introduce a review/insufficient-information status alongside the existing grade when coverage is low. This would need an explicit policy and schema/output design. It is not a current grade rule.

**Alias and domain improvements:** expand vocabulary from reviewed failures and improve qualifier handling. Maintain provenance and inspect effects across occupations, rather than silently substituting pair-domain labels for source facts.

The proposal's intended outcome is an auditable score whose skill contributions reflect explicit job importance. Adoption depends on trustworthy annotations, measured extraction quality and evaluation beyond the original teacher-labelled comparison.
