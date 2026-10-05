# Gemini pipeline smoke test

Date: 2026-09-30. Model: `gemini-3.7-flash`, accessed through Google's official
Gemini API. Credentials were supplied through a hidden process prompt and were
not saved in source, configuration, command arguments, reports, or request URLs.

The same three short development pairs used in the Qwen run were selected.
No dataset labels entered extraction or scoring. This sample is not an accuracy
benchmark. The initial prompt was ats-facts-en-v4; the later prompt is v5.

## Results and limits

- v4: 0/3 complete pairs. Each CV produced structured JSON and literal source
  quotes, but supplied domain tags without supporting industry-dimension evidence.
- v5 clarified that industry means professional domain and that facts.domains
  requires role/task quotes under evidence.industry. This is a prompt correction.
- Two CVs (No Fit and Potential Fit sample rows) subsequently passed validation
  and remain cached. Their JD requests were affected by HTTP 503 high demand and
  later HTTP 429 quota exhaustion, so they did not reach scoring.
- The Good Fit sample CV generated a Python evidence quote although Python is
  absent from its source, including a case-insensitive check. That response was
  rejected and archived for later review; it is not a corrected training target.
- Current final three-pair attempt: 0/3 complete pairs, blocked by HTTP 429.
  No ATS score or model accuracy claim was produced.

The user specified a five-requests-per-minute limit. Client requests are paced
at least 13 seconds apart, including retries and metadata requests. A diagnostic
request additionally confirmed this quota violation:

`GenerateRequestsPerDayPerProjectPerModel-FreeTier`, value `20`, for
`gemini-3.7-flash`.

The daily limit was exhausted. A generic retry-delay value was also present in
the response; the daily quota identifier takes precedence. The client stops
retrying daily exhaustion and blocks subsequent calls within that client session.
Resume from cache after the provider quota becomes available again.

## Retained artifacts

- `output/ats/gemini_results/smoke_v4_20260930.json`: initial matching-prompt attempt.
- `output/ats/gemini_results/smoke_v5_before_rate_limit_20260930.json`: clarified
  prompt attempt, showing high-demand errors and the unsupported Python quote.
- `output/ats/gemini_results/smoke.json`: latest paced attempt.
- `output/ats/gemini_results/single_pair_probe.json`: exact daily quota diagnostic.
- `output/ats/cache/`: successful structured CVs, model version and token usage.
- `output/ats/cache/failure_archive/`: historical error records and failed model
  responses. Later successful retries do not erase these historical observations.

Provider unavailability/quota failures and the prompt mapping gap are separate
from unsupported generated facts. Only source-reviewed corrected extractions
should become supervised training targets. The original Qwen observations used
v4, so a fair later model comparison must rerun both models with the same frozen
prompt and schema. The current attempts do not establish which model is better.
