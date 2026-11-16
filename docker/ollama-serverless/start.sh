#!/bin/sh
# /ping answers 204 (starting) until the three models are in Ollama and loaded on the GPU, then 200 (ready).
set -e
python3 /worker/health.py &
OLLAMA_HOST="0.0.0.0:${PORT}" ollama serve &
export OLLAMA_HOST="127.0.0.1:${PORT}"
until ollama list > /dev/null 2>&1; do sleep 1; done
ollama show qwen-v4 > /dev/null 2>&1 || ollama create qwen-v4 -f /worker/Modelfile.qwen-v4
ollama show qwen-o5 > /dev/null 2>&1 || ollama create qwen-o5 -f /worker/Modelfile.qwen-o5
ollama show qwen3-embedding:4b > /dev/null 2>&1 || ollama pull qwen3-embedding:4b
# load all three into VRAM now (6.2 + 4.8 + 3.2 GB: a 20 GB card), so the first real request does not pay for it
python3 - <<'PY'
import json, os, urllib.request
url = f"http://127.0.0.1:{os.environ['PORT']}/api/"
for path, body in (("generate", {"model": "qwen-v4", "prompt": "", "keep_alive": -1}),
                   ("generate", {"model": "qwen-o5", "prompt": "", "keep_alive": -1}),
                   ("embed", {"model": "qwen3-embedding:4b", "input": "warm up", "keep_alive": -1})):
    urllib.request.urlopen(urllib.request.Request(url + path, json.dumps(body).encode(),
                                                  {"Content-Type": "application/json"}), timeout=600).read()
PY
touch /tmp/ready
wait
