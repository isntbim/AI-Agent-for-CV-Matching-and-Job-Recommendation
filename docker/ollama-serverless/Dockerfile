# Ollama as a Runpod load-balancing serverless worker: the GPU half of the stack (qwen-v4 + qwen3-embedding).
# Runpod routes https://<endpoint>.api.runpod.ai/<path> to Ollama on $PORT and probes GET /ping on $PORT_HEALTH.
# The models live on the endpoint's network volume (/runpod-volume), so the image stays small and a cold start
# only loads them: put qwen-v4-q4_K_M.gguf in /runpod-volume/models/ once (see docker/ollama-serverless/README.md).
FROM ollama/ollama:0.40.14
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY start.sh health.py /worker/
COPY Modelfile.qwen-v4 /worker/
ENV PORT=11434 PORT_HEALTH=8080 OLLAMA_MODELS=/runpod-volume/ollama OLLAMA_NUM_PARALLEL=1 OLLAMA_KEEP_ALIVE=-1
ENTRYPOINT ["/bin/sh", "/worker/start.sh"]
