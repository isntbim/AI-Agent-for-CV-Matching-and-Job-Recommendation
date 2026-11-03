"""Local V4 extractor (Qwen2.5-7B + V4 QLoRA, merged, int4 on Ollama) with validate-and-retry as a LangGraph graph.

generate -> repair (scripts.qwen_repair: literal quotes, unsupported claims, derived fields, stray brackets) ->
valid? done : retry at the next repetition penalty. It is an LLMClient, so ExtractionCache / extract_document
use it unchanged and still run the production validation on the repaired JSON.
"""

import json
from typing import TypedDict

from src.parser.llm_extractor import ExtractionError

PENALTIES = (1.05, 1.1, 1.15, 1.2)  # first try = the evaluated setting; retries as on the V4 test set
MAX_NEW_TOKENS = 16384  # same cap as generate.py; longest training answer is ~12k tokens


class State(TypedDict, total=False):
    system: str
    user: str
    text: str
    task: str
    attempt: int
    raw: str
    finish: str
    label: dict | None
    status: str
    log: list[str]


class QwenV4:
    """V4 served by local Ollama (scripts/qwen_build_ollama.py). The prompt is built with the base Qwen chat template
    and sent raw, as in the vLLM evaluation; greedy, no schema-constrained decoding. Ollama's repeat_penalty over the
    whole context (repeat_last_n = num_ctx) matches vLLM/HF repetition_penalty, which also covers the prompt tokens."""

    def __init__(self, model: str = "qwen-v4", url: str = "http://localhost:11434", timeout: float = 1800):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.model, self.url, self.timeout = model, url, timeout

    def complete(self, system: str, user: str, repetition_penalty: float) -> tuple[str, str]:
        import httpx

        prompt = self.tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                              tokenize=False, add_generation_prompt=True)
        r = httpx.post(f"{self.url}/api/generate", timeout=self.timeout, json={
            "model": self.model, "prompt": prompt, "raw": True, "stream": False,
            "options": {"temperature": 0, "seed": 0, "num_ctx": 32768, "num_predict": MAX_NEW_TOKENS,
                        "repeat_penalty": repetition_penalty, "repeat_last_n": 32768}})
        r.raise_for_status()
        out = r.json()
        return out["response"], "length" if out.get("done_reason") == "length" else "stop"


def build_graph(llm):
    from langgraph.graph import END, StateGraph

    from scripts.jd_teacher_check import JD_SCHEMA
    from scripts.qwen_repair import parse_balanced, repair

    def generate(s: State) -> State:
        raw, finish = llm.complete(s["system"], s["user"], PENALTIES[s["attempt"]])
        return {"raw": raw, "finish": finish}

    def fix(s: State) -> State:
        label = parse_balanced(s["raw"], JD_SCHEMA if s["task"] == "jd_extraction" else None)
        tag = f"attempt {s['attempt'] + 1} (repetition_penalty {PENALTIES[s['attempt']]}, {s['finish']})"
        if label is None:
            return {"label": None, "status": "unparsed", "log": s["log"] + [f"{tag}: unparsed"]}
        label, status, log = repair(label, s["text"], s["task"])
        return {"label": label, "status": status, "log": s["log"] + [f"{tag}: {status}"] + log}

    def route(s: State) -> str:
        return "done" if s["status"] in ("pass", "partial") or s["attempt"] + 1 >= len(PENALTIES) else "retry"

    graph = StateGraph(State)
    graph.add_node("generate", generate)
    graph.add_node("repair", fix)
    graph.add_node("next_attempt", lambda s: {"attempt": s["attempt"] + 1})
    graph.set_entry_point("generate")
    graph.add_edge("generate", "repair")
    graph.add_conditional_edges("repair", route, {"done": END, "retry": "next_attempt"})
    graph.add_edge("next_attempt", "generate")
    return graph.compile()


class V4Client:
    """LLMClient for extract_document / ExtractionCache. The model loads on first use."""

    def __init__(self, llm=None):
        self.llm, self.graph, self.last_metadata = llm, None, None

    def generate(self, system_prompt, user_prompt, temperature=0.0, max_tokens=4096):
        if self.graph is None:
            self.llm = self.llm or QwenV4()
            self.graph = build_graph(self.llm)
        user = json.loads(user_prompt)
        task = "cv_extraction" if user["document_kind"] == "resume" else "jd_extraction"
        s = self.graph.invoke({"system": system_prompt, "user": user_prompt, "text": user["source_text"], "task": task,
                               "attempt": 0, "log": []})
        self.last_metadata = {"model": "qwen2.5-7b-v4-qlora", "attempts": s["attempt"] + 1,
                              "repetition_penalty": PENALTIES[s["attempt"]], "status": s["status"], "log": s["log"]}
        if s["status"] not in ("pass", "partial"):
            raise ExtractionError(f"V4 output still invalid after {s['attempt'] + 1} attempts: {s['status']}")
        return json.dumps(s["label"], ensure_ascii=False)
