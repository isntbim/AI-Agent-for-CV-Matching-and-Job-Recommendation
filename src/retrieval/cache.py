"""Content/revision/preprocessing keyed embedding cache with validated entries."""

from pathlib import Path
import os
import uuid
import numpy as np

from .corpus import digest
from .text import TEXT_VERSION


def valid_vectors(vectors, count, dimension):
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape != (count, dimension) or not np.isfinite(values).all():
        raise ValueError("Invalid embedding shape or non-finite values")
    norms = np.linalg.norm(values, axis=1)
    if count and (norms <= 0).any():
        raise ValueError("Zero embedding vector")
    return values / norms[:, None] if count else values


def save_array(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class EmbeddingCache:
    def __init__(self, directory: Path, model: str, revision: str, preprocessing: str = TEXT_VERSION):
        if not revision:
            raise ValueError("Embedding cache requires a resolved model revision")
        self.directory, self.model, self.revision, self.preprocessing = directory, model, revision, preprocessing
        self.hits = self.misses = 0

    def key(self, text):
        return digest([digest(text.encode("utf-8")), self.model, self.revision, self.preprocessing])

    def embed(self, texts, client, batch_size=1):
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive")
        values, missing = {}, {}
        for text in texts:
            key = self.key(text)
            if key in values or key in missing:
                continue
            path = self.directory / (key + ".npy")
            if path.exists():
                values[key] = valid_vectors(np.load(path, allow_pickle=False).reshape(1, -1), 1, client.dimension)[0]
                self.hits += 1
            else:
                missing[key] = text
        pending = list(missing)
        for start in range(0, len(pending), batch_size):
            keys = pending[start:start + batch_size]
            vectors = valid_vectors(client.embed_batch([missing[k] for k in keys]), len(keys), client.dimension)
            for key, vector in zip(keys, vectors):
                save_array(self.directory / (key + ".npy"), vector)
                values[key] = vector
                self.misses += 1
        return np.stack([values[self.key(t)] for t in texts]) if texts else np.empty((0, client.dimension), np.float32)
