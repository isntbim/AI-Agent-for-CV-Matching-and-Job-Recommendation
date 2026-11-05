"""Qwen3-Embedding-4B through Ollama: the ranking model's text and title embeddings (benchmark and service share it).
Context 2,048 tokens, longer text cut: the same budget as bge-m3, and 8,192 scored the same (NDCG@10 0.590 vs 0.591)."""
import numpy as np
import requests

MODEL = "qwen3-embedding:4b"  # Ollama tag, Q4_K_M
NUM_CTX = 2048
INSTRUCT = "Instruct: Given a candidate's resume, retrieve job descriptions that fit the candidate\nQuery: "


class QwenEmbedder:
    def __init__(self, url: str = "http://localhost:11434", timeout: float = 600):
        self.url, self.timeout = url, timeout

    def encode(self, texts: list[str], query: bool = False, batch: int = 8) -> np.ndarray:
        """Unit vectors. query=True adds the retrieval instruction (the CV side); documents and titles go in plain."""
        out = []
        for i in range(0, len(texts), batch):
            r = requests.post(f"{self.url}/api/embed", timeout=self.timeout, json={
                "model": MODEL, "input": [(INSTRUCT if query else "") + t for t in texts[i:i + batch]],
                "truncate": True, "options": {"num_ctx": NUM_CTX}})
            r.raise_for_status()
            out.append(np.array(r.json()["embeddings"], dtype=np.float32))
        v = np.vstack(out)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
