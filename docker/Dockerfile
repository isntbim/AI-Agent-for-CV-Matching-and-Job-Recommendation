# The matching API (FastAPI + the LangGraph service flow). The LLMs run in the separate Ollama container.
# Build from the repo root: docker compose -f docker/docker-compose.yml up --build
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.cache/hf
WORKDIR /app

COPY docker/requirements-api.txt .
# bge-m3 retrieval runs on the CPU, so the CPU torch wheel is enough (the GPU is Ollama's).
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
