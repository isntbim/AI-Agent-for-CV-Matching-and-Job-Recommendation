# ResuMatch Frontend (ATS CV–JD Matching System)

Next.js 14 + TypeScript + Tailwind CSS **Bento Grid SaaS** frontend for the **ATS CV–JD Matching Capstone System**, adapted from [Gaurav Singh's ResuMatch](https://github.com/gaurav-singh7092/ResuMatch) (MIT License — see [`NOTICE.md`](./NOTICE.md)) and wired directly to the local FastAPI matching service (`src/api/app.py`).

## What It Connects To

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/api/health` | `GET` | Checks FastAPI status, Ollama extractor availability, corpus size (`361` jobs), and active ranking model (`hybrid_qwen_v1`). |
| `/api/match/stream` | `POST` (`multipart/form-data`) | Streams live SSE `token`, `step` (`extract`, `retrieve`, `score`, `rank`), and final `result` events while matching a CV. |
| `/api/match` | `POST` (`multipart/form-data`) | Synchronous fallback endpoint returning ranked CareerViet jobs, 7-dimension rule scores, matched/missing skills, explanations, and literal quotes. |
| `/api/history` | `GET` | Lists recent saved matching runs from local SQLite (`output/service/runs.sqlite`). |
| `/api/history/{run_id}` | `GET` | Reloads a saved run and its recorded human feedback. |
| `/api/feedback` | `POST` | Saves `+1` / `-1` thumbs votes and optional `Good Fit` / `Potential Fit` / `No Fit` overrides. |

## Running Locally

1. **Start the FastAPI backend** from the repository root (`D:\SS9 - FinalSeason`) with Ollama running (`qwen-v4` + `qwen3-embedding:4b`):
   ```bash
   .venv/Scripts/python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
   ```

2. **Install frontend dependencies and start Next.js** from `frontend/`:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. Open **http://localhost:3000** (or **http://localhost:3000/analyze** to upload or paste a CV directly).

To point the frontend at a custom backend URL, set `NEXT_PUBLIC_API_URL` (defaults to `http://127.0.0.1:8000`).

## Production Build

```bash
cd frontend
npm run build
npm run start
```

## License & Attribution

Frontend UI adapted from **ResuMatch** (`https://github.com/gaurav-singh7092/ResuMatch`) by **Gaurav Singh**, licensed under the MIT License. Full copyright and permission notice are preserved in [`NOTICE.md`](./NOTICE.md).
