# ATS scorer references and rubric

The [Phase 2 scorer explanation](../explanations/scorer_explanation.md) connects these references
to the current workflow and the proposed weighted-skill extension.

The runtime scorer is an independent implementation. GitHub references explain
selected methods; they do not establish that the resulting score predicts hiring.
No upstream source or taxonomy assets have been copied or imported. Check the
license and preserve attribution before introducing copied material in the future.

## Pinned reference

Repository: [Sara12-2/ResumeMatch_AI](https://github.com/Sara12-2/ResumeMatch_AI)

Inspected commit: `a30c9e32c0d801b0b2a48f3b8883245ad67636b7`.
The inspected source contains both deterministic legacy helpers and enhanced
LLM/semantic paths. This project references the specific deterministic methods
below. An upstream README alone is insufficient to describe the inspected code.

| Rule | Exact upstream code | Original behavior | Adopted method and changes | Why use it |
|---|---|---|---|---|
| R1 skills | [`_get_matcher()`, `_build_matcher()`, `extract_skills()`, `analyze_skill_gap()` in `src/skill_extractor.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/skill_extractor.py) | Lowercase taxonomy aliases become canonical names; set intersection/difference produces matched, missing and extra skills; coverage is matched / all extracted JD skills. | Reference canonical equivalence and set coverage. Our code scores structured required skills, reports preferred coverage separately, retains unlisted skills, and treats absent evidence as unknown. Preferred skills are the target only when no required skills exist. Our mappings are project-authored and we do not use spaCy PhraseMatcher. | Coverage is understandable, deduplicated, and yields traceable matched/missing skills. |
| R2 education | [`highest_degree_level()` in `src/skill_extractor.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/skill_extractor.py), [`EnhancedATSScore._compute_education_match()` in `src/ats_score.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/ats_score.py) | Rank recognized degree phrases; compare candidate rank with JD rank, then apply a degree-match floor to a text/semantic score. | Reference degree ranking and the attainment comparison only. Our extractor identifies completed degrees and the minimum accepted JD degree (including alternatives). Our 100/50/0 degree-gap scores are project decisions; no upstream floor or text/semantic score is adopted. | Explicit attainment is easier to audit than similarity between an education paragraph and a whole JD. |
| R3 professional domain | [`detect_domains()`, `_get_domain_matcher()` in `src/skill_extractor.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/skill_extractor.py), [`EnhancedATSScore._compute_project_relevance()` in `src/ats_score.py`](https://github.com/Sara12-2/ResumeMatch_AI/blob/a30c9e32c0d801b0b2a48f3b8883245ad67636b7/src/ats_score.py) | Map text keywords to technical domain labels; intersect domain sets and apply a project score floor. | Reference canonical domain labels and their overlap. Our vocabulary covers professional families (software, data, finance, sales, etc.), uses source-backed extracted facts and title families, and computes matched JD domains / JD domains. Our formula and vocabulary are project decisions; no project relevance floor is adopted. | Domain evidence remains inspectable and can cover the dataset's nontechnical occupations. |

## Project decisions

`src/scoring/normalization.py` defines English aliases and role/domain vocabulary.
`src/scoring/rules.py` implements rules; results expose R1/R2/R3/P1/P2/P3/P4.
All numeric component rules are initial hypotheses to be evaluated. The label
dataset validates agreement of total grades only, not the individual subscores.

| Rule | Runtime behavior | Rationale |
|---|---|---|
| P1 title | Exact canonical role 100; distinct roles in the same known family 70; known different families 0; insufficient mapping/evidence 50. Target CV title takes precedence, otherwise use latest dated work title. | Distinguish occupation alignment from skill coverage. Unrecognized different titles cannot establish a mismatch. Seniority tokens are removed for title comparison. |
| P2 experience | Union of dated work intervals in the JD role family, divided by the minimum required months, capped at 100. Explicitly no experience required gives 100. Unknown requirements, ambiguous role relevance, missing/invalid/future dates, or year-only dates give neutral 50. | Compare required duration while avoiding double counting concurrent jobs. Work with a different known family contributes zero months. Months are end-month minus start-month; Present uses the supplied fixed date. Missing end dates are not implicitly Present. |
| R2 education numeric rule | Completed degree at or above minimum 100; one level below 50; two or more below 0. Ranks: high school, associate, bachelor, master, doctorate. Explicitly no education required gives 100. Unknown attainment/requirement gives 50. | A simple ordered requirement comparison. Field of study and equivalent-experience substitutions are not scored in v1 and must be reflected in interpretation. |
| R3 domain numeric rule | 100 times fraction of identified JD professional domains represented by CV domains; unknown gives 50. | Alignment of professional work, independent of employer sector. Title/domain correlations are a known limitation of this rubric. |
| P3 location | Explicit compatible country/city or remote eligibility 100; explicit incompatible country/city 0; incomplete constraints 50. Worldwide remote requires explicit evidence; remote without eligibility scope is unknown. | Avoid equating remote with worldwide eligibility or using default Vietnam as evidence. Relocation and user location preferences can be added in a later web integration. |
| P4 salary | Default Negotiable, undisclosed JD maximum or incomparable currency gives 75 without supported coverage. Explicit minimum within JD maximum gives 100; above maximum gives 100 times JD maximum / expected minimum. | Salary is optional candidate input. Normalize monthly/annual values but perform no exchange-rate conversion. A stated comparable maximum remains usable even when a JD says negotiable. |

Weights originate in the project's existing `src/parser/match_schemas.py`:
skills .30, title .20, experience .15, education .10, domain (`industry` key) .10,
location .08, salary .07. They have not been sourced from the GitHub reference.
The existing silver annotation grade calculation remains unchanged.

Runtime defaults are 50/75 grade cutoffs, with Strong requiring skills >= 70 and
title >= 70. Development calibration searches integer cutoffs maximizing macro-F1,
ties resolved by distance from 50/75, then lower cutoffs. These thresholds are
data-calibrated project decisions, not copied from an external scorer.

Totals use the existing deterministic weighted sum rounded to one decimal.
Components round to integers. Coverage is the sum of weights with supported
evidence; neutral dimensions retain their weights. An all-unknown input therefore
scores 51.8 with coverage 0 and Partial grade. Consumers must present coverage and
missing evidence alongside the total; a grade alone is insufficient to assess it.
