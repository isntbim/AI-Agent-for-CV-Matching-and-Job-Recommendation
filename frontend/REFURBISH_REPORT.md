# Frontend Refurbish Report

## 1. Summary of Changes

Refurbished the ResuMatch Next.js 14 + TypeScript + Tailwind CSS frontend (`frontend/`) for our FastAPI CV-to-CareerViet-Jobs matching service (`http://localhost:8000`).

### API & State Layer
- **`frontend/services/api.ts`**:
  - Replaced legacy single-JD `/analyze`, `/batch-analyze`, and `/job-status` endpoints with `getHealth()` (`GET /api/health`) and `matchCv({ file, text, top, asOf })` (`POST /api/match` with `multipart/form-data`).
  - Configured Axios timeout to `300000` ms (5 minutes) to accommodate first-time GPU CV extraction with Qwen2.5-7B + LangGraph validate-repair-retry (~60–120 s).
  - Defined strict TypeScript interfaces (`HealthResponse`, `FitGrade`, `ScoreDimension`, `DimensionScores`, `JobMatch`, `MatchResponse`, `MatchCvParams`) and `parseUnscoredDimensions(explanation)` helper.
  - Surfaced backend HTTP error `detail` strings (`{"detail": "..."}`) verbatim.
- **`frontend/hooks/useAnalysis.ts`**:
  - Replaced dual-file + JD hook with `useAnalysis()` managing `isLoading`, `elapsedSeconds` (live timer while waiting), `results: MatchResponse | null`, `error: string | null`, `runMatch`, and `reset`.
- **`frontend/next.config.js` & `frontend/.env.example`**:
  - Updated `/api/:path*` rewrite destination to `${process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}/api/:path*`.
  - Simplified `.env.example` to `NEXT_PUBLIC_API_URL=http://localhost:8000`.

### Match Flow & Results UI
- **`frontend/pages/analyze.tsx`**:
  - Removed the job-description textarea and recruiter batch mode completely.
  - Added tabbed CV input: drag-and-drop file upload (`.pdf`, `.docx`, `.txt`, max 5 MB) or pasted CV text (`>= 200` characters, with live character count and explicit `<label>`).
  - Added top-N selector (`Top 3`, `Top 5` default, `Top 10`).
  - Added live elapsed-seconds progress panel with factual copy: *"Reading your CV with our fine-tuned model (about a minute for a new CV)"*.
  - Added prominent alert banner displaying verbatim backend `detail` errors.
- **`frontend/components/FileUpload.tsx`**:
  - Removed the artificial 1.5 s fake upload progress delay; selecting a file immediately updates parent state.
  - Added controlled `selectedFile` synchronization, keyboard focus rings, and accessible labels.
- **`frontend/components/AnalysisResults.tsx`**:
  - Added summary header: `"{matches.length} best of {result.pool} candidate jobs · {result.seconds.toFixed(1)} s"` and ranking model badge (`hybrid_tfidf_v1`).
  - Rendered ranked job cards (`#1..#N`) with rank badge, job title, `jd` ID, `retrieval_rank`, `rule_score`, SVG fit-score ring (`0–100`), and icon + text grade pill (`Good Fit`, `Potential Fit`, `No Fit`).
  - Built an inline plain-SVG 7-axis radar chart (`skills`, `title`, `experience`, `education`, `industry`, `location`, `salary`) paired with a 7-dimension breakdown legend. Axes listed in `"Not scored (evidence missing or inconclusive, neutral value used): ..."` are rendered with dashed spokes, hollow grey dots, and `"not scored"` badges — never displaying neutral fallback values (`50` or `75`) as real scores.
  - Added `Has (Matched Skills)` and `Missing Skills` chip groups (with `"No skill overlap recorded"` fallback), plain-English explanation bullets (`explanation.slice(1)`), expandable `"Read the job posting"` (`job_text`), and expandable `"Evidence"` quotes per dimension.

### Navigation, Copy & Attribution
- **`frontend/components/Header.tsx`**:
  - Added live service health badge calling `getHealth()` on mount: green `"361 jobs ready"` (`status === "ok"`), amber `"Extractor offline — cached CVs only"` (`status === "degraded"`), or red `"API offline"`.
- **`frontend/components/Footer.tsx` & `frontend/NOTICE.md`**:
  - Kept `frontend/NOTICE.md` intact and added a `"Based on ResuMatch"` credit link (`https://github.com/gaurav-singh7092/ResuMatch`) in the footer.
- **`frontend/pages/index.tsx`, `frontend/components/Hero.tsx`, `frontend/components/Features.tsx`, `frontend/pages/about.tsx`, `frontend/pages/help.tsx`, `frontend/pages/404.tsx`, `frontend/README.md`**:
  - Rewrote all copy to accurately describe our pipeline: fine-tuned Qwen2.5-7B (QLoRA) extractor, LangGraph validate-repair-retry loop, hybrid BGE-M3 + BM25 retrieval (top 20 of 361 CareerViet jobs), and V3.1 7-dimension rule + `hybrid_tfidf_v1` ranking.
  - Removed all invented statistics ("99% accuracy", "10,000+ resumes"), upskilling roadmaps, BERT/SpaCy/NLTK claims, and batch processing references.

---

## 2. Build & Audit Summary

- **Dependencies (`frontend/package.json`)**:
  - Upgraded `next` and `eslint-config-next` from `14.0.4` to `14.2.35` (keeping React 18) and ran `npm install` + `npm audit fix`.
- **Production Build (`npm run build`)**:
  - Succeeded with exit code `0`, zero TypeScript or ESLint errors.
  - Static routes generated cleanly: `/` (`5.49 kB`), `/analyze` (`26.9 kB`), `/about` (`4.02 kB`), `/help` (`5.65 kB`), `/404` (`1.4 kB`).
- **Security Audit (`npm audit --omit=dev`)**:
  - `npm audit fix` resolved all non-breaking runtime vulnerabilities (including `axios`, `picomatch`, `postcss-selector-parser`, `source-map-js`, and `yaml`), reducing `--omit=dev` advisories from 14 to 5.
  - The remaining 5 advisories (`next` < 16.x, its bundled `postcss`, and `sucrase`/`glob`/`minimatch` under `tailwindcss`) require a breaking upgrade to Next.js 16 (`npm audit fix --force`), which was intentionally skipped per the constraint to stay on Next.js `14.2.x` and React 18.

---

## 3. End-to-End Live Verification

Verified against the running FastAPI service (`http://localhost:8000`) and Next.js server:
- `GET /api/health` returned `{"status":"ok","extractor":"up","jobs":361,"ranking_model":"hybrid_tfidf_v1"}`.
- `POST /api/match` with `cv.txt` (`top=5`) returned `5` ranked, explained jobs from a candidate pool of `20` (`ranking_model: "hybrid_tfidf_v1"`, `seconds: 8.8`, top match `jd_042` — *Data Analyst & Data Scientist*, `fit: 52.3`, `grade: "Potential Fit"`, `rule_score: 44.1`).
- All Next.js routes (`/`, `/analyze`, `/about`, `/help`) returned HTTP `200`.

---

## 4. Remaining Items

- None within `frontend/`.
