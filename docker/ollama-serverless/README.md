# Ollama on a Runpod serverless endpoint

The GPU half of the stack. Everything else (`docker/docker-compose.yml`) runs on any machine with Docker, no GPU.

```
laptop (Docker Compose): frontend -> api -> vectordb, historydb, prometheus, grafana
                                      |
                                      +--HTTPS + Bearer key--> Runpod load-balancing endpoint: Ollama (qwen-v4 + qwen3-embedding)
```

## One-time setup
1. **Network volume** (20 GB) in a data center that has serverless GPUs in the endpoint's pools.
2. **Model files:** copy `qwen-v4-q4_K_M.gguf` and `qwen-o5-q4_k_m.gguf` (both built by `scripts/qwen_build_ollama.py`)
   to the volume's `models/` folder (via Runpod's S3 API or a temporary pod with the volume attached).
3. **Endpoint:** type *load balancer*, image `ghcr.io/isntbim/ai-agent-for-cv-matching-and-job-recommendation/ollama-serverless:latest`
   (built by CI), the volume attached, ports `11434/http` and `8080/http`, env `PORT=11434`, `PORT_HEALTH=8080`,
   GPU pools with at least 20 GB (the three models take about 14 GB), workers min 0 (set min 1 during a demo to skip the cold start).
4. **API key** restricted to that endpoint, in `docker/.env` (git-ignored) as `OLLAMA_API_KEY`, with
   `OLLAMA_URL=https://<endpoint-id>.api.runpod.ai`.

## What a worker does on start (`start.sh`)
`/ping` answers 204 while it starts. Ollama keeps its models on the volume (`OLLAMA_MODELS=/runpod-volume/ollama`):
the first worker ever creates `qwen-v4` and `qwen-o5` from the GGUFs and pulls `qwen3-embedding:4b`, later workers find them there.
All three are loaded into VRAM, then `/ping` answers 200 and Runpod routes requests to Ollama on port 11434.

## Cost notes
- Billed per second while a worker runs; with min 0 workers an idle endpoint costs nothing (the volume is billed monthly).
- The API never polls a remote Ollama (health and `/metrics` skip it when `OLLAMA_API_KEY` is set): a 15-second
  Prometheus scrape would otherwise keep a GPU worker awake around the clock.
