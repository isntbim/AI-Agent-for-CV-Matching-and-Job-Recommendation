"""Web service: upload a CV (PDF, DOCX or TXT), get the best-fitting jobs from the CareerViet corpus, each with a fit
score and a plain-English explanation. The pipeline is scripts/ats_service.match(): V4 extraction (local Ollama) ->
hybrid retrieval -> Hybrid + TF-IDF ranking -> V3.1 explanation.

  uvicorn src.api.app:app --port 8000        # needs `ollama serve` with the qwen-v4 model; then open http://localhost:8000
"""
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from scripts import ats_service as svc

STATIC = Path(__file__).resolve().parent / "static"
MAX_BYTES = 5 * 1024 * 1024
TYPES = {".pdf", ".docx", ".txt"}
STATE: dict = {}
LOCK = threading.Lock()  # shortcut: one match at a time (one GPU); a queue with job ids if several users must overlap


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["ranker"] = svc.corpus_ranker()  # bge-m3 + TF-IDF over the corpus, loaded once (~20 s on CPU)
    STATE["paths"] = svc._jd_paths()
    yield


app = FastAPI(title="CV Matching AI Agent", lifespan=lifespan)
# the Next.js frontend (frontend/, ResuMatch-based) calls this API from its own dev/prod origin
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def cv_text(upload: UploadFile | None, text: str | None) -> str:
    """The CV as plain text, from an uploaded file or a pasted text. Rejects anything else."""
    if upload is None or not upload.filename:
        if not text or len(text.strip()) < 200:
            raise HTTPException(400, "Upload a CV file (PDF, DOCX or TXT) or paste at least 200 characters of CV text.")
        return text.replace("\r\n", "\n").replace("\r", "\n")  # browsers send form text with \r\n
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in TYPES:
        raise HTTPException(415, f"{suffix or 'This file type'} is not supported. Upload a PDF, DOCX or TXT file.")
    data = upload.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "The file is larger than 5 MB.")
    if suffix == ".txt":  # same newlines as a text file read on the server, so a known CV hits the extraction cache
        return data.decode("utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    from src.parser.extractors import extract_text_from_file
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"cv{suffix}"
        path.write_bytes(data)
        try:
            out = extract_text_from_file(path)
        except Exception as exc:  # a corrupt or scanned file: tell the user, not a 500
            raise HTTPException(422, f"Could not read text from this file ({type(exc).__name__}).") from exc
    if len(out.strip()) < 200:
        raise HTTPException(422, "Almost no text could be read from this file. Is it a scanned image? Try a DOCX or TXT.")
    return out


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    import httpx
    try:
        ollama = httpx.get("http://localhost:11434/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        ollama = False
    return {"status": "ok" if ollama else "degraded", "extractor": "up" if ollama else "down: start `ollama serve`",
            "jobs": len(svc.corpus_ids()), "ranking_model": "hybrid_qwen_v1"}


@app.post("/api/match")
def match(file: UploadFile | None = File(None), text: str | None = Form(None), top: int = Form(5),
          as_of: str | None = Form(None)) -> dict:
    """Synchronous: the CV extraction takes about a minute on the local GPU (instant for a CV seen before)."""
    if not 1 <= top <= 20:
        raise HTTPException(400, "top must be between 1 and 20.")
    try:
        as_of = (date.fromisoformat(as_of) if as_of else date.today()).isoformat()
    except ValueError as exc:
        raise HTTPException(400, "as_of must be a date like 2026-11-04.") from exc
    body = cv_text(file, text)
    start = time.perf_counter()
    with LOCK:
        try:
            out = svc.match(body, as_of, top=top, pool=20, ranker=STATE["ranker"])
        except Exception as exc:
            from src.parser.llm_extractor import ExtractionError, LLMConnectionError
            if isinstance(exc, ExtractionError) and not isinstance(exc, LLMConnectionError):
                raise HTTPException(422, "The CV could not be read reliably by the extractor. Try a cleaner file.") from exc
            raise HTTPException(503, f"The matching service failed ({type(exc).__name__}). Is Ollama running?") from exc
    for m in out["matches"]:
        m["job_text"] = (svc.ROOT / STATE["paths"][m["jd"]]).read_text(encoding="utf-8")
    return out | {"seconds": round(time.perf_counter() - start, 1)}
