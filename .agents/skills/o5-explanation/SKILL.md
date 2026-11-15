---
name: o5-explanation
description: Use whenever writing O5 match explanations as a teacher (writing output/o5/teacher/<pair_id>.json from output/o5/inputs/<pair_id>.json). Gives the facts format, what a good explanation says, the style and length rules, the hard faithfulness rules, and the self-check command to run after every file.
---

# O5 explanation teacher (v4, strict)

You write the short explanation a job seeker reads next to one CV–job match result. The scorer already computed the
scores; you explain **why** the result came out this way and **what the candidate can do**, the way a good recruiter
would after reading both documents. **Never invent anything.** Do NOT spawn subagents; do one pair at a time.

Repo root: `D:/SS9 - FinalSeason`.

## Input: `output/o5/inputs/<pair_id>.json` → `facts`
- `verdict` ("Good Fit" / "Potential Fit" / "No Fit"), `fit` (0–100, final ranking model), `rule_score` (0–100).
- `job_title`, `job_level` (e.g. "Senior", may be null), `job_min_years`, `job_min_degree`.
- `cv_title`, `cv_recent_roles` (title + dates, most recent first), `cv_education`.
- `dimensions`: skills, title, experience, education, industry, location, salary: `score` + `scored`.
  `not_scored_because` gives the true reason for an unscored dimension.
- `requirements`: the job's requirements, most important first: `requirement`, `importance` ("must-have" /
  "required"), `status` ("met" / "partly met" / "not met"), and `your_skill` for a near or text match.
- `matched_skills`, `missing_skills`, `found_in_cv_text`, `nice_to_have_matched`, `nice_to_have_missing`,
  `soft_skills_not_scored`, `vague_cv_titles` (one-word titles such as "IT", "Leader").

## Output: `output/o5/teacher/<pair_id>.json`
```json
{"pair_id": "<pair_id>", "explanation": "<your text>"}
```

## What a good explanation does (3 to 5 sentences, 90 to 120 words, hard limit)
1. **Verdict, fit and the one main reason, in one sentence.** Start with the verdict words exactly:
   "Potential Fit (fit 47/100): you work in the same field, but the job's core tools are missing from your CV."
2. **The strongest evidence, specific and short.** At most 3 items: met requirements, a near match ("your Tableau
   experience partly covers Power BI"), a matching recent role with its dates, or the degree that meets the requirement.
3. **The gaps that matter, at most 3**, most important first. Choose specific, meaningful requirements (a tool, method
   or qualification).
4. **One concrete next step**, built from the gaps: what to add or make visible in the CV if they have it, or whether
   this job is worth applying for.

## Judgement rules (these decide whether an explanation is good)
- **Skip generic requirements.** Never present a bare generic word as a gap or a match: "math", "science",
  "technical", "listening", "communication", "programming languages", "team-oriented and collaborative working
  environments". Pick the specific ones. Leaving an item out is always allowed.
- **Don't argue with the obvious.** If a senior software engineer's CV is "missing" something that their role clearly
  implies (e.g. "data structures"), don't list it as a gap; pick a specific tool instead, or say the CV does not
  name it explicitly.
- **Seniority.** If `job_level` or `job_min_years` clearly differs from the CV's recent roles (a Director applying for
  a 2-year role, a junior for a Senior role), say so in one short clause. Only from the facts: the titles, dates,
  `job_level` and `job_min_years`.
- **Unscored dimensions:** mention at most one, only title, experience or education, and only when it changes how the
  reader should read the result (e.g. "your experience could not be checked against the 5 years asked for"). Never
  mention location or salary. Never list several unscored items.
- **Never use an unscored dimension as evidence or as a gap** (`scored: false`), not even in passing. Education
  unscored: do not name the CV's degree at all. Title unscored: do not compare the CV's titles or roles with the job
  ("close to", "overlaps", "relevant", "same kind of role", "different line of work"). Industry unscored: do not name a
  field. Experience unscored: do not judge the years, level or fit of the experience. The only allowed sentence about it
  is the optional one that says it could not be checked. Build the explanation from the scored dimensions, the skills
  and the requirement statuses instead. `python -m scripts.o5_faithfulness check` flags these as "unscored:".
- **Different line of work (No Fit):** say the roles are a different line of work first (write "line of work" or "career", not "field" or "industry": the app shows a separate Industry score, often 100 even when the work differs), name the CV's field from its recent
  roles, then at most 2 gaps, then advice.
- **Good/Potential Fit with no requirement met:** say the verdict comes mainly from how similar the CV and the job
  are overall, and the candidate should check the gaps before applying.
- **Tone.** Plain, direct, kind. State the evidence and let the reader judge. No "easily", "perfect", "ready to apply",
  "strong candidate"; never call the candidate weak or unqualified.
- **No system words.** Never write "scorer", "scored separately", "rule score", "dimension", "facts", "the model",
  "algorithm" or "basis". Say it the way a recruiter would: "your title could not be compared with this one", not
  "the scorer could not compare your title".
- **Vary the opening.** Don't open every No Fit with "your background and this X are in different fields". Lead with
  what is specific to this pair: "This Senior Accountant role sits outside your network engineering career", "Your
  experience is in data entry and office support, while this role is a Senior Accountant position", and so on.
- **Style.** Vary sentence openings between explanations. Write skill names exactly as given (including the source's
  odd spellings such as "UIUX principles"). No lists of more than 3 items. No repeated words like
  "required requirement".

## Hard rules (checked automatically)
1. The verdict words exactly as given, and no other verdict words.
2. Numbers only from the facts: `fit`, `rule_score`, a scored dimension's score, 100 (as in "/100"),
   `job_min_years`, years and dates in `cv_recent_roles`, or a count of listed items. No percentages or years you
   worked out yourself.
3. Name a skill only if it is in the facts (requirements, `your_skill`, or the skill lists).
4. No company names, no people's names, nothing about the candidate beyond the facts.

## Strict self-review before you save (score each line 0, 1 or 2; rewrite until the total is at least 8 of 10)
1. **Reason first:** the first sentence states the one real reason for the verdict, specific to this pair.
2. **Specific evidence:** the matches named are concrete (a tool, a dated role, a near match, a degree), not generic.
3. **Gaps that matter:** at most 3, specific and in order of importance; no generic words; nothing the CV's role
   obviously implies.
4. **Useful next step:** one concrete action this candidate can take, or a clear apply / don't-apply judgement.
5. **Reads well:** 90 to 120 words, no system words, no lists over 3 items, no repeated phrases, kind and plain.
Do not write the score into the file; just rewrite until it would pass.

## Self-check after every file
```
.venv/Scripts/python.exe -m scripts.o5_faithfulness check <pair_id>
```
It checks faithfulness and the strict style rules (length 90 to 120, system words, overclaims, location/salary,
generic gaps, more than one unscored item). Fix your file until it prints `faithful 1/1`.
