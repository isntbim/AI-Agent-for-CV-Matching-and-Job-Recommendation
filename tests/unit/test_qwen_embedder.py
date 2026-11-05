import numpy as np

from src.embeddings import qwen


def test_query_gets_instruction_and_vectors_are_unit_length(monkeypatch):
    sent = []

    class Resp:
        def __init__(self, n):
            self.n = n

        def raise_for_status(self):
            pass

        def json(self):
            return {"embeddings": [[3.0, 4.0]] * self.n}

    def post(url, timeout, json):
        sent.append(json)
        return Resp(len(json["input"]))

    monkeypatch.setattr(qwen.requests, "post", post)
    v = qwen.QwenEmbedder().encode(["a", "b", "c"], query=True, batch=2)
    assert v.shape == (3, 2) and np.allclose(np.linalg.norm(v, axis=1), 1)
    assert [len(s["input"]) for s in sent] == [2, 1]
    assert sent[0]["input"][0] == qwen.INSTRUCT + "a" and sent[0]["options"]["num_ctx"] == 2048
    qwen.QwenEmbedder().encode(["doc"])
    assert sent[-1]["input"] == ["doc"]
