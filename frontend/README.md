# ResuMatch Frontend (CareerViet CV-to-Jobs Matcher)

Next.js 14 + TypeScript + Tailwind CSS frontend for the ATS CV-to-CareerViet-Jobs matching service (`src/api/app.py`).
Adapted from [ResuMatch](https://github.com/gaurav-singh7092/ResuMatch) by Gaurav Singh (MIT License — see `NOTICE.md`).

## Quick Start

1. **Install dependencies:**
   ```bash
   cd frontend
   npm install
   ```

2. **Configure environment (optional):**
   ```bash
   cp .env.example .env.local
   ```
   Defaults to `NEXT_PUBLIC_API_URL=http://localhost:8000`.

3. **Run development server:**
   ```bash
   npm run dev
   ```
   Open `http://localhost:3000`.

4. **Production build:**
   ```bash
   npm run build
   npm start
   ```

## Backend API Endpoints Used

- `GET /api/health` — Returns service health (`ok` / `degraded`), extractor status, job count (`361`), and active ranking model (`hybrid_tfidf_v1`).
- `POST /api/match` — Multipart form upload (`file` or `text`, `top`, optional `as_of`) returning top-N ranked CareerViet job matches with 7-dimension scores, matched/missing skills, plain-English explanations, and supporting quotes.
