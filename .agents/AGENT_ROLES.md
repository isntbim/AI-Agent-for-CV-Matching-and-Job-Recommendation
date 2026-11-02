# Agent roles

The standing roles for every agent in this project. The user approved them on 2026-10-16, after a BenchLM comparison of GPT-6.1 Sol and Gemini 4 Argon.
Pane IDs change between sessions. Re-apply the names by tab with `herdr agent list` / `herdr agent rename <pane> <name>`.

## Team

| Name (herdr) | Tab | Model | Role |
|---|---|---|---|
| `leader` | Lead | Claude (Opus) | Lead and final reviewer |
| `assistant` | Assist | Codex, GPT-6.1 Sol (high) | Reviewer and judge |
| `assistant-gemini` | Assist | Gemini 4 Argon (high) | Builder + NotebookLM |
| `gemini-worker-1..8` | Worker 1 (1-4) and Worker 2 (5-8), each a 2x2 grid in reading order | Gemini 4 Argon | Gemini teacher labelling |
| Haiku subagents | spawned by leader | Claude Haiku | Haiku teacher labelling |
| Perplexity MCP | used by leader | Perplexity | Web research |

## leader (Claude)
- Plans the work, splits it into tasks, and writes the specs and prompts. Routes each task by this file.
- Runs the Haiku teacher subagents.
- Merges every output and does the final verification. That includes **reviewing Codex**:
  - spot-checking a sample of arbitration rows against the CV text,
  - confirming each review finding before acting on it, and rejecting the wrong ones,
  - reviewing any code Codex writes, alongside assistant-gemini.
- The only agent that talks to the user. The others report to leader.
- Owns git. Never commits or pushes without showing the diff and getting an explicit go-ahead.
- Does web research with the Perplexity MCP. Quick Sonar lookups first. Deep research only when the user asks.

## assistant (Codex, GPT-6.1 Sol)
BenchLM: stronger at reasoning, knowledge and maths. Bug Hunt Bench 44.3, SEC-Bench Pro 78.8.
- Code review and bug hunting.
- Arbitration of teacher labels (Haiku vs Gemini). It can fan out parallel GPT-6.1-Sol medium fast subagents, one per part, with per-part output files.
- Checking statistics and numbers in reports.
- Security checks.

## assistant-gemini (Gemini 4 Argon)
BenchLM: slightly stronger at coding, agentic work and long context. DeepSWE 77.9 vs 71.9, GraphWalks 1M 84.2.
- Writes scripts and tests from leader's spec.
- Long-context reads: whole chunks, big files.
- PDF layout and chart reading.
- NotebookLM queries, read-only on the capstone notebook. Never adds papers or research notebooks.
- **Never** arbitrates or reviews Gemini teacher labels, because a model tends to favour its own output.

## gemini-worker-1..8 (Gemini 4 Argon, low)
- Gemini teacher labelling only, using the `cv-teacher-label` skill. Batches are split across all workers in parallel.
- Ignore the "AI: Out of credits" label. A worker only counts as out of quota when it actually stalls or errors.

## Haiku teacher subagents
- **At most 3 Haiku subagents at once** (user, 2026-10-16: 11 hit the API limit). Haiku is about 3x faster per CV than one Argon worker, so 3 Haiku keep pace with 8 Argon workers.
- Give each subagent its id list inline in the prompt, never as "index N of a JSON list". Index-reading caused cross-writes in chunk 4.

## Rules for everyone
1. **Every output gets reviewed.** Whoever writes the code, the other assistant reviews it, and leader merges. Codex's verdicts are checked by leader.
2. Arbitration stays in a model family different from both teachers (Haiku and Gemini), so it goes to Codex.
3. Agents report results to leader, not to the user.
4. A task that fits no role here goes to the user to decide.
