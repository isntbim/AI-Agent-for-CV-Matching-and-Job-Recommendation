# Explanation: deterministic ATS scorer v1 and v2

**Audience:** developers and source-data reviewers. **Updated:** 2026-10-06.
**Default:** `ats-en-v1`. **Optional ruleset:** `ats-en-v2`.

V2 replaces the earlier weighted-skills proposal with the final policy in the implementation plan. The plan adopts the PDF's requirement weights and group shares, then adds explicit rules for evidence, proficiency, allocation, bonus and Core gates. This document specifies those rules; it makes no additional claim about the PDF's contents. Running commands and review steps are in [the ATS scorer guide](../guides/ats_scorer.md).

The scorer consumes validated CV/JD envelopes and calculates seven scores with deterministic Python. Retrieval and reranker scores remain independent. No LLM, GPU or embedding model decides the ATS score. Extraction quality still matters: reproducible arithmetic cannot establish that an extracted fact is correct.

## Input contracts and evidence

`ResumeInput` and `JobInput` retain `document` and `facts`. `Facts` contains supporting quotes, completed degrees, job constraints, professional domains and other evidence used by the unchanged dimension rules. V1 documents remain readable.

V2 CV documents use `schema_version: resume-v2`. Each skill group retains its existing `keywords` and adds `details`: skill name, normalized proficiency, original proficiency wording, skill quote, proficiency quote and provenance. Keywords must agree with the detail list. The old group's `level` never supplies a proficiency value or a bonus for individual skills.

V2 JD documents use `schema_version: job-v2`. `document.skills.requirements` is the scoring source: stable requirement ID, one or more skill options, importance (`core`, `standard`, `preferred`), source quote, importance quote and review state. Existing required/preferred lists remain available to old consumers. The JD field `proficiency` still means required/preferred; it is not a candidate proficiency scale. V2 introduces no minimum skill-level requirement for JDs.

The envelope's `annotation` records schema identity, source/text checksums, extraction completeness and review states. Skills-only approval and full-envelope approval are separate. A reviewer approving skills does not approve every CV/JD fact or turn the envelope into training gold.

Quotes must exist in the source and support the specific skill or requirement. Core needs evidence of mandatory importance; a proficiency other than `normal` needs explicit wording tied to that skill. Standard is the default when a Required item has no supported Core classification. Defaults receive no invented quote. Validation detects quote problems, negation, inconsistent metadata, proficiency conflicts, duplicate requirements and ambiguous option groups. Literal quote validation does not replace human review of the interpretation.

## Normalization, alternatives and allocation

Known aliases resolve before matching. Exact, normalized exact and equivalent aliases receive the same full credit: JS and JavaScript are equivalent. Unknown names can match by their normalized exact strings. Normalization removes duplicate CV skills and merges duplicate requirement units before calculating denominators. A duplicate with different importance retains the highest supported importance and records the merge.

An `A OR B` requirement is one unit with two options and one weight; matching either option fulfills that unit. AND creates independent units. Overlapping option groups, unclear scopes and complex compound expressions require review before confirmation.

One CV skill can fulfill at most one independent requirement. Allocation optimizes these objectives in order:

1. Core fulfillment by exact/alias matching.
2. All Required (Core and Standard) fulfillment by exact/alias matching.
3. Remaining weighted Required fulfillment.
4. Preferred fulfillment using remaining skills.

Sorted canonical names and stable requirement IDs resolve ties. Input ordering does not change the result.

Semantic relations are directed, versioned review records. They include candidate skill, target requirement skill, relation type, reason, source and approval status. Reversing a relation or chaining relations does not create another allowed relation. Approved relations are active by default; an explicit provisional option permits pending relations for diagnostics. Rejected relations remain inactive.

| Match | Coefficient `m` | Counts for Core gate? |
|---|---:|---|
| Exact / normalized exact / alias | 1.00 | Yes |
| Approved near relation | 0.70 | No |
| Approved weak transfer relation | 0.40 | No |
| No valid match | 0.00 | No |

BGE-M3 can propose up to five neighbors per training-vocabulary skill for reviewer inspection. A high cosine score alone creates no runtime matching permission. The scorer only reads the relation table.

## Proficiency

| CV proficiency | Fulfillment factor `f` | Bonus factor `b` |
|---|---:|---:|
| Beginner | 0.50 | 0 |
| normal | 1.00 | 0 |
| Intermediate | 1.00 | 0 |
| Advanced | 1.00 | 0.10 |
| Expert | 1.00 | 0.20 |

`normal` is neutral default behavior, not a claim of Intermediate capability. Senior titles, years, projects and a level attached to an entire skill group do not establish individual proficiency. Unmapped wording stays in the record and enters review; explicit conflicting levels must be resolved before confirmation. The factors describe this rubric's treatment of self-reported evidence, not independently measured ability.

## V2 skills arithmetic

There are two levels of weighting. Each requirement has weight `w`; each present group has a share in the skills score.

| Requirement group | Weight per requirement `w` | Group share |
|---|---:|---:|
| Core | 1.00 | 70% |
| Standard | 0.60 | 20% |
| Preferred | 0.40 | 10% |

For each present group `g`:

```text
F_g = sum(w_i * m_i * f_i) / sum(w_i)
```

All requirements inside one group have equal weights, so `w` cancels within that group's fulfillment. After computing fulfillment, group shares determine how much each group contributes. Multiplying by 1/0.6/0.4 again would incorrectly apply requirement importance twice.

```text
normalized_share_g = original_share_g / sum(original_share of present groups)
base_skills = 100 * sum(normalized_share_g * F_g)
```

| Groups present | Normalized shares |
|---|---|
| Core, Standard, Preferred | 70% / 20% / 10% |
| Core, Standard | 77.78% / 22.22% |
| Core, Preferred | 87.5% / 12.5% |
| Standard, Preferred | 66.67% / 33.33% |
| Any one group | 100% for that group |

An absent group has fulfillment `null`, not zero. Preferred contributes to the base score. With all three groups, its maximum base contribution is ten points; a Preferred-only JD is scored entirely from Preferred fulfillment.

### Proficiency bonus

Only an Advanced/Expert CV skill exactly matching a Core or Standard requirement creates a bonus. Preferred matches, semantic matches and skills outside the JD do not create a bonus.

Let `e_i` be one for exact/alias and zero otherwise; `W_required` is the sum of Core and Standard requirement weights.

```text
bonus = min(5, 25 * sum(w_i * e_i * f_i * b_i) / W_required)
skills_before_gate = min(100, base_skills + bonus)
```

If there is no Required group, bonus is zero. Fully matching Required skills at Advanced adds at most 2.5 points; Expert adds at most five. Skills always remain at most 100.

Example: Core Python; Standard SQL and Docker; Preferred Kubernetes. The CV has Python Expert and SQL normal, with no valid Docker/Kubernetes match.

```text
F_core = 1; F_standard = 0.5; F_preferred = 0
base_skills = 100 * (0.70 * 1 + 0.20 * 0.5 + 0.10 * 0) = 80
W_required = 1 + 0.6 + 0.6 = 2.2
bonus = 25 * (1 * 0.20) / 2.2 = 2.2727...
skills_before_gate = 82.2727...
```

### Core gate

Core fulfillment for the base score may contain semantic credit. Gate fulfillment is a separate calculation using exact/alias credit and the Beginner factor only:

```text
core_gate_fulfillment = sum(w_i * e_i * f_i for Core) / sum(w_i for Core)
```

| Gate fulfillment | Result |
|---|---|
| At least 70% | `passed`; no Core limitation |
| At least 50%, below 70% | `failed`; skills capped at 49; Strong blocked |
| Below 50% | `failed`; skills capped at 49; grade forced to Mismatch (`0`) |
| No Core requirements | `not_applicable` |
| Required evidence incomplete | `unknown`; no invented Core failure |

The exact boundaries matter: 50% avoids the forced Mismatch but remains capped; 70% passes. Bonus and semantic matches cannot rescue a failed gate. The scorer evaluates gate ratios before rounding.

No skill requirements or insufficient CV/JD skill evidence produces neutral skills 50 with an unsupported basis. Missing extraction is distinct from a complete, supported annotation with no valid match. The former never establishes that the candidate lacks Core skills.

## Overall score and grade

V2 changes the skills pipeline only. The other six dimension formulas and overall weights remain the v1 rules:

| Dimension | Overall share | Formula |
|---|---:|---|
| Skills | 30% | Selected v1 or v2 skills rules |
| Title | 20% | Same recognized role 100; same recognized family 70; different families 0; unknown 50 |
| Experience | 15% | Relevant non-overlapping months / required months, capped at 100; explicitly none required 100 |
| Education | 10% | Completed degree meets minimum 100; one rank below 50; at least two below 0 |
| Industry | 10% | Shared professional domains / identified JD domains; unknown 50 |
| Location | 8% | Supported compatible constraints 100; incompatible 0; unknown 50 |
| Salary | 7% | Within comparable maximum 100; above maximum proportional; negotiable/unknown 75 |

Experience uses the fixed `as_of_date` for Present; overlapping intervals count once. Missing dates are not automatically Present. Remote does not imply worldwide eligibility. Schema-default country/currency values do not supply evidence. Industry means professional alignment rather than the employer's business sector.

Components are integers; the overall score uses those components and rounds to one decimal. V2 breakdowns retain arithmetic before component rounding. Coverage is the sum of supported dimension weights, not a confidence probability.

| Grade | Final rule |
|---|---|
| Strong (`2`) | Overall at least 75; skills at least 70; title at least 70; Core gate does not block Strong |
| Partial (`1`) | Overall at least 50 and Core gate does not force Mismatch |
| Mismatch (`0`) | Overall below 50 or Core gate forces Mismatch |

There is no new experience >=60 gate or undefined severe-mismatch rule. Runtime, calibration and evaluation use the same grade/gate helper.

## V1 compatibility and output

`ScorerConfig()` still chooses v1. Its skills score is unique canonical required-skill coverage; Preferred becomes the target set only when no required skills exist. V1 ignores skill proficiency and has no Core gate. Older schemas and the original silver adapter remain supported.

V2 is selected explicitly with `ScorerConfig(version="ats-en-v2")`. It returns all existing result fields plus `skills_breakdown`: requirement matches and chosen options; group fulfillment; Core gate fulfillment; base score; bonus; pre/post-gate scores; full/partial/unmet status; evidence, provenance, checksums, rule/relation versions and review status. Template explanations describe the calculated values. No LLM decides points.

The main implementation modules are [models.py](../../src/scoring/models.py), [rules.py](../../src/scoring/rules.py), [skills_v2.py](../../src/scoring/skills_v2.py), [skill_v2.py](../../src/parser/skill_v2.py), [v2_corpus.py](../../src/scoring/v2_corpus.py) and [v2_benchmark.py](../../src/scoring/v2_benchmark.py). Extraction prompts, schema/adapter identities and cache identities distinguish v1 from v2.

Existing attribution and baseline method references are retained in [scorer_references.md](../reference/scorer_references.md).

## Review corpus and benchmark interpretation

The v2 corpus preserves all 88 CV and 361 JD identities, source/text checksums and split assignments. `before` preserves the original v1 envelope; `envelope` contains source-backed v2 annotations and inherited resolved pilot decisions. Source problems outside skills are recorded separately so their impact is visible in the v1 original-to-reviewed comparison.

Prepared annotations are provisional until you or a teammate accepts, corrects or rejects them. Empty review cells remain pending. Imports validate document/checksum identity and are idempotent; stale decisions are refused. Only fully approved envelopes in the training partition may be exported for training. This implementation includes no Qwen training, VietJobs ingestion, web integration, Azure changes or new experience gate.

The fixed offline benchmark keeps the original 1,672 pair labels. Primary reporting uses the 1,144 Gemini pairs; 528 heuristic negatives are reported separately. Seven ablations separate source corrections, group weighting, semantics, proficiency and Core gates:

1. V1 on original data.
2. V1 on source-reviewed data.
3. V2 with equal shares among present groups; no semantics, proficiency or gates.
4. V2 with 70/20/10 shares.
5. Add approved semantic relations.
6. Add proficiency and bonus.
7. Add Core gates: full v2.

Reports include grade macro-F1, balanced accuracy, per-grade precision/recall, confusion matrices, score distributions, coverage, every failure, latency, source/grade changes, domains and the 24 historical disagreements. Code, annotation/source checksums, relation table, configuration, selected pairs and date form a report identity. Artifacts from different identities cannot be combined.

These are diagnostics on previously inspected silver data. Agreement is not independent test accuracy or correctness against hiring outcomes. Pending review, extraction incompleteness and the use of Core/proficiency/semantic rules remain explicit in the report. V1 stays the default while the v2 code and provisional review package are handed off.

## Optional v3 depth extension

V3 adds explicit, skill-specific JD minimum proficiency independently of
Core/Standard/Preferred importance. OR options may have different minima. The
three explicit ranks are Intermediate/Advanced/Expert; `normal` is unknown.
Literal Beginner wording now maps to normal in v3, preserving `raw_proficiency`
and its quote. It receives full existing fulfillment, no depth gap and no bonus.
For exact/alias matches, gaps of 0/1/2 levels receive 1/.75/.50 depth credit.
Unknown CV levels keep existing credit;
semantic matches keep existing credit but cannot prove required-skill depth.

Full v3 uses effective fulfillment in allocation, base skills and Core gating.
Advanced/Expert bonuses require the match to meet any explicit minimum; without
a minimum, legacy eligibility remains. The unrounded .5/.7 Core boundaries are
unchanged. `score_only` changes allocation/base skills while retaining legacy
bonus/gate factors; `off` preserves equivalent supported v2 arithmetic. Literal
Beginner normalization is an intentional v3 difference, including in `off`. V1 remains
the runtime default, and v2's verified source, receipts and artifacts are frozen.

Skill-specific experience years are recorded but unscored. Titles nominate
review; explicit human item approval may resolve a title choice. Titles alone
cannot establish proficiency depth. Direct approvals, rule-derived changes and whole-envelope approval are
kept separate. All material changes and positive/negative experiment outcomes
are saved in the [research package](../research/ats_scorer/README.md), with
[methodology](../research/ats_scorer/methodology.md),
[results](../research/ats_scorer/results.md) and
[unresolved reviews](../research/ats_scorer/unresolved_review.json).

### 2026-10-08 v3 review correction

Depth-enabled alternatives now use the same ordered allocation objectives as global matching, so a semantic alternative cannot displace the exact Core or Standard objective solely because its weighted credit is higher. Preferred alternatives retain weighted selection, and depth-off preserves v2 behavior. Negated depth claims and rejected CV details cannot provide positive proficiency credit. Missing evidence remains unknown. [The remediation record](../research/ats_scorer/remediation_2026-10-08.md) documents the policy, tests and unchanged silver-data results.
