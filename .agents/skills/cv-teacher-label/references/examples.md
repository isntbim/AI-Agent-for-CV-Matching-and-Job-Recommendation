# Worked examples (real arbiter decisions, chunks 1–3)

Each line: what the CV says → what a teacher wrote → the correct label.

## Skills
- "Processed personnel actions" → skill "personnel actions" → **drop** (an item handled, not a capability).
- "contracts", "benefits", "inventory of food and supplies" listed in duties → skills → **drop** (objects handled). Keep "benefits administration" only if those words are written in the CV; never compose it.
- "Attention to detail" → skill → **drop** (generic trait).
- "Reconciled vendor rebates and lease accounting" → nothing → **add** "lease accounting" (a discipline named in a duty line); not "vendor rebates" (transactions handled).
- "Performed account reconciliations using SAP" → nothing → **add** "account reconciliations" and "SAP" (duty lines name skills too).
- "Report on breakeven points by products" → "Breakeven Analysis" → **drop** (renamed duty; the name is not in the source).
- "Meta Ads (Facebook)" → "Facebook" → **edit** to "Meta Ads" with the full quote (fullest named span).
- "Excel: filters, VLOOKUP, pivot tables" → 4 skills → **one**: "Excel" (features belong to the tool).
- "Project\nManagement" (wrapped by the PDF) → skipped → **add** "Project Management", skill_quote "Project\nManagement".
- "HTML,CSS,jQuery,andJavaScript" (spaces lost in the PDF) → quote re-spaced as "HTML, CSS, jQuery, and JavaScript" → **wrong**: quote the exact glued text; the checker WARNs, that is expected.

## Levels
- "Word (Advanced)" → advanced, raw "Advanced", proficiency_quote "Word (Advanced)". ✔
- "Advanced: R, Python" (a heading over a list) → advanced for both → **normal** for both, raw "Advanced", proficiency_quote the exact span.
- Flattened skill table where the columns no longer line up and you cannot tell which level goes with which skill → **normal**, keep raw + quote, mention in the final note. If a row clearly pairs one skill with one level, keep the level.
- "Proficient and literate in Spanish" → intermediate (direct claim about Spanish) → keep; checker WARNs → human review.
- "FluentSpanish", "BasicGerman" → languages with fluency "Fluent"/"Basic", Languages skill details proficiency normal, raw "Fluent"/"Basic".
- "Proficient in Java, C, HTML and CSS" → intermediate for all → **normal** for all, raw "Proficient", proficiency_quote the exact span ("Proficient in" over a list is a group label; decided in chunk 2).

## Degrees
- "Master of Science, May 2026" (no "expected") and `as_of_date` is 2026-10-15 → **completed** (past date). A teacher who left it out was wrong.
- "Sep 2025 to Dec 2026, Master of Science" and `as_of_date` is 2026-10-25 → **not completed** (end date after as_of_date), endDate null.
- "B.S. Computer Science, expected June 2025" read in a CV labelled in 2026 → still **not completed** ("expected" wording wins; the CV gives no award).
- "Ph.D. candidate" / "Class of 2027" / "Pursuing" → **not completed**, endDate null.
- "Diploma (NIPA)" → education only; not in completed_degrees (no level in the enum).

## Location
- "Sydney, Australia" in the contact line → country "AU". ✔
- "San Diego, California", "Omaha, NE", "NewYork, NY" → city/region only, **country null** (the country is not written).
- University address in the header ("Rutgers ... New Brunswick, NJ") → not the candidate's residence → null.

## Records
- Volunteer firefighter, no organisation named → employer "New Hamburg, New York" or "New Hamburg Fire Department" → **null** (a town is not an employer; do not compose a name).
- Employer written as "Company Name" (template placeholder) → **null**. Never borrow a company name from another line.
- "Honors: 2022-2024 Dean's List" → certificate type award, date `2024` (end year), never `"2022-2024"`.
- "Summer 2012" → `2012`; "Current"/"to date" for a job → `"Present"`; an empty date → `null`, never `""`.
- One role "2010 - 2013, 2016 - 2019" → **two** work records.
