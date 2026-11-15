# AI Agent for CV Matching and Job Recommendation Using Large Language Models

[![ci](https://github.com/isntbim/AI-Agent-for-CV-Matching-and-Job-Recommendation/actions/workflows/ci.yml/badge.svg)](https://github.com/isntbim/AI-Agent-for-CV-Matching-and-Job-Recommendation/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)

## 1. What it does

Upload one CV (PDF, DOCX or text). The service returns the best-fitting jobs from a corpus of 361 Vietnamese and English
job descriptions. Each job has a verdict (Good / Potential / No Fit), a fit score, seven component scores, the matched and
missing skills with quotes from the CV, and an explanation. Everything runs on one 12 GB GPU; no commercial LLM API is
used at run time.

| Part | Result (test sets, details in the report) |
| :--- | :--- |
| Extraction: Qwen2.5-7B-Instruct + QLoRA adapter "V4" | Skill F1 0.71 (CVs) and 0.72 (JDs), mean of 3 training seeds (best 0.73), from 0.49 and 0.33 for the unmodified model |
| Validate–repair–retry agent (LangGraph) | Schema-valid output 29.0% → 98.2% on 765 documents; 100% with loop removal during generation |
| Retrieval: BGE-M3 dense + BM25, Reciprocal Rank Fusion | MRR@10 0.841 with the BGE reranker (silver labels): above BM25 (p = 0.01), level with dense alone (p = 0.35) |
| Ranking: rule scores + Qwen3-Embedding cosines + TF-IDF, logistic model | NDCG@10 0.630, AUC 0.953 on 31,150 test pairs |
| Explanations: second QLoRA adapter + faithfulness guard (offline evaluation) | Wrong: 4.9% English, 8.2% Vietnamese (LLM review) |

## 2. Pipeline

```
CV file ─► text extraction + cleaning
        ─► extract   qwen-v4 (Ollama) + validate → repair → retry      cached by text hash
        ─► retrieve  BGE-M3 dense + BM25 → RRF → pool of 20 jobs        skipped when job ids are given
        ─► score     seven rule scores per job (skills, title, experience, education, industry, location, salary)
        ─► rank      logistic model → verdict, fit, order
        ─► explain   matched / missing skills with quotes, "could not be checked" notes
        ─► JSON / web page
```

The whole flow is one LangGraph graph (`scripts/ats_service.py`, `service_graph`). Job descriptions are extracted once,
offline (`--prepare-jds`).

## 3. Run it

**Requirements:** an NVIDIA GPU with 12 GB, [Ollama](https://ollama.com) 0.40, Python 3.12, and the local data under
`output/` (corpus, retrieval index, extraction cache). The data is not in this repository (see section 7).

```bash
python -m venv .venv && .venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r docker/requirements-api.txt
python -m scripts.qwen_build_ollama                        # once: merge the V4 adapter, quantise to Q4_K_M, `ollama create qwen-v4`
ollama pull qwen3-embedding:4b
OLLAMA_NUM_PARALLEL=1 ollama serve                         # one slot: both models fit in 12 GB
python -m scripts.ats_service --prepare-jds                # once: extract every corpus job description
uvicorn src.api.app:app --port 8000                        # web page: http://localhost:8000, API docs: http://localhost:8000/docs
```

Command line, same graph: `python -m scripts.ats_service CV.txt --as-of 2026-11-13 --top 5`.

**Docker Compose** (needs the NVIDIA container toolkit; the models must already exist in the host's Ollama folder):

```bash
docker compose -f docker/docker-compose.yml up --build
```

| Service | Port | Role |
| :--- | :--- | :--- |
| `ollama` | – | GPU inference: `qwen-v4` extractor and `qwen3-embedding:4b` |
| `api` | 8000 | FastAPI + the LangGraph flow + web page |
| `prometheus` | 9090 | Scrapes `api:8000/metrics/` every 15 s |
| `grafana` | 3001 | "CV Match service" dashboard (admin / admin on first login) |

Corpus, index and caches are mounted from the host's `output/` folder; they hold personal data and never enter the image.

## 4. API

Interactive documentation (Swagger) is served at `/docs`.

| Method and path | Purpose |
| :--- | :--- |
| `POST /api/match` | Form fields `file` (PDF/DOCX/TXT, 5 MB) or `text`, `top` (1–20), `as_of` (date). Returns the ranked jobs and a `run_id`. |
| `POST /api/match/stream` | Same input; server-sent events: `token` (extractor output so far), `step` (each graph node), then `result` or `error`. |
| `GET /api/history`, `GET /api/history/{run_id}` | Recent runs and one saved run (SQLite, `output/service/history.sqlite`). |
| `POST /api/feedback` | JSON `{run_id, jd, vote: 1 or -1, grade?, comment?}`: thumbs up/down and the user's own verdict. |
| `GET /api/health` | Readiness; reports whether the extractor is reachable. |
| `GET /metrics/` | Prometheus metrics: response time per route, match time, queue length, tokens, CPU, RAM, VRAM per model. |

One match runs at a time (one GPU); further requests wait in a queue.

## 5. Tests and CI

```bash
pytest tests/unit -q --ignore tests/unit/test_ats_silver_eda.py
ruff check src scripts tests --select E9,F63,F7,F82
```

Tests need no GPU, network or cloud access. Tests that need the git-ignored gold data skip themselves when it is absent.
GitHub Actions (`.github/workflows/ci.yml`) runs the lint and the tests on every push and pull request, and on `main`
builds the API image and pushes it to GitHub Container Registry.

## 6. Reproducing the results

| Step | Entry point |
| :--- | :--- |
| QLoRA training (extraction and explanation adapters) | `training/scripts/train_qlora.py` (seeds 3407, 1, 2; logs to Weights & Biases) |
| Generation for evaluation | `training/scripts/generate.py` (vLLM, A100) or `training/scripts/generate_local.py` (Unsloth 4-bit, local GPU) |
| Extraction scoring against gold | `python -m scripts.qwen_eval preds/base_test.jsonl preds/ft_test.jsonl` |
| Matching evaluation on the labelled pairs | `python -m scripts.match_eval` |
| Explanation faithfulness checker and guard | `scripts/o5_faithfulness.py`, `generate_local.py --guard-from` |

Every training and evaluation run, with its configuration, data artifact, adapter and predictions, is logged to the
Weights & Biases project `capstone_project`. Greedy generation with the 4-bit model is deterministic for a fixed batch
shape but changes with batch size and padding, so regenerated explanations can differ from the reviewed files.

## 7. Data and privacy

The CVs are public CVs found online and two public Hugging Face datasets; the job descriptions were collected from
CareerViet and ITviec. Raw files, extracted texts, gold labels and all outputs contain personal data and stay out of git
(`data/raw/`, `output/`, `knowledge-base/` are ignored). The system ranks and explains; it must not be used to reject
candidates automatically, and no fairness audit has been done.

## 8. Repository structure

```
configs/      ranking model weights and scorer configuration (JSON)
docker/       Dockerfile, docker-compose.yml, monitoring/ (Prometheus + Grafana)
docs/         guides and research notes (start at docs/README.md)
frontend/     Next.js front end (adapted from the open-source ResuMatch project)
scripts/      service, evaluation, labelling and data scripts
src/          api/ (FastAPI + web page), parser/, retrieval/, scoring/, embeddings/, vector_db/ (pgvector store)
tests/        unit and integration tests
training/     QLoRA training and generation scripts
```

## 9. Team

* **Supervisors:** Mr. Nguyễn Xuân Huy, Mr. Nguyễn Quốc Trung
* **Engineering Team:**
  * **Lê Trí Dũng** (Leader) - Student Code: SE196261
  * **Thái Thành Nhân** (Member) - Student Code: SE196293
  * **Nguyễn Danh Bằng** (Member) - Student Code: SE194871
