"""Prompt-injection guard for CV and job text (OWASP LLM01), English and Vietnamese.

A CV is untrusted input that goes into an LLM prompt. Lines that address the AI (override its instructions, claim to be
a system message, tell it how to rate the candidate) or that try to break out of the JSON the extractor writes are
removed before extraction, and reported so the user can see what was dropped. Ordinary CV wording ("followed company
instructions", "system administrator", "ranked top 5% of the sales team") is not touched: every pattern needs an
instruction aimed at the model, a rating order about this candidate, or JSON syntax.

shortcut: pattern lists, not a classifier; a reworded attack can pass. The numbers stay safe regardless, because scores
come from rules over extracted facts and every extracted skill must be quoted from the text (see the injection test).
"""
from __future__ import annotations

import re

_I = re.IGNORECASE
PATTERNS: dict[str, list[re.Pattern]] = {
    "override": [
        re.compile(r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|preceding|all|any|"
                   r"your|system|the rest|remainder)\b.{0,30}\b(instructions?|prompts?|rules|directions|guidelines|"
                   r"requirements|criteria|document|text|context)\b", _I),
        re.compile(r"\b(ignore|disregard)\s+(the|all|any|these|those)\s+(job\s+)?(requirements|criteria|instructions|rules)\b", _I),
        re.compile(r"(?<!không )\bbỏ qua\b.{0,30}\b(hướng dẫn|chỉ dẫn|chỉ thị|lệnh|quy tắc|tiêu chí|nội dung trên)\b", _I),
        re.compile(r"\b(quên|phớt lờ)\b.{0,30}\b(hướng dẫn|chỉ dẫn|chỉ thị|quy tắc|tiêu chí)\b", _I),
        re.compile(r"\bkhông cần (đánh giá|xem xét|chấm)\b", _I),
    ],
    "fake_system_message": [
        re.compile(r"\b(system|developer|admin(istrator)?)\s+(note|prompt|message|instructions?|override)\s*:", _I),
        re.compile(r"<\|?\s*/?\s*(im_start|im_end|system|assistant|endoftext|instructions?)\s*\|?>", _I),
        re.compile(r"^\s*[\[\(]\s*(system|admin|developer|assistant|instructions?)\s*[\]\)]\s*:?", _I),
        re.compile(r"^\s*#{2,}\s*(system|instructions?)\b", _I),
        re.compile(r"\b(lưu ý|thông báo|chỉ thị|hướng dẫn)\s+(của\s+)?hệ thống\s*:", _I),
    ],
    "addresses_the_model": [
        re.compile(r"\b(instructions?|notes?|messages?)\s+(to|for)\s+(the\s+)?(\w+\s+){0,2}(parser|model|ai|assistant|llm|"
                   r"chatbot|screener|ats|language model)\b", _I),
        re.compile(r"\b(you are now|as an ai\b|dear (ai|chatgpt|assistant|model|ats))", _I),
        re.compile(r"\bwhen you (read|process|parse|review|see|evaluate) this (cv|resume|document|profile|application)\b", _I),
        re.compile(r"^\s*(assistant|chatgpt|gpt|claude|gemini|llm|ai model)\s*[,:]\s*(please|when|you|ignore|output|return|"
                   r"respond|give|mark)\b", _I),
        re.compile(r"\b(gửi|dành cho)\s+(ai|mô hình|trợ lý|hệ thống chấm|chatgpt)\b", _I),
        re.compile(r"\b(ghi chú|lưu ý|nhắn|yêu cầu)\s+(cho|tới|đến|gửi)\s+(hệ thống|ai|mô hình|trợ lý|chatgpt)\b", _I),
    ],
    "rating_order": [
        re.compile(r"\b(rate|score|rank|grade|mark|classify|label)\s+(this|the|them|him|her|me)\s*(candidate|cv|resume|"
                   r"applicant|profile)?\b.{0,40}\b(100|perfect|good fit|top|highest|maximum|excellent)\b", _I),
        re.compile(r"\b(evaluate|assess|judge|treat|consider)\s+(this|the)\s+(resume|cv|candidate|applicant|profile)\s+as\b", _I),
        re.compile(r"\b(mark|label|return|output|respond|answer|classify|set|give|assign)\b.{0,30}\b(good fit|potential fit|"
                   r"fit\s*[:=]?\s*100|100\s*/\s*100|highest score|maximum score|full marks)\b", _I),
        re.compile(r"\b(perfect|ideal|strong)\s+match\s+for\s+(every|all|any)\s+(job|role|position)s?\b", _I),
        re.compile(r"\b(hãy\s+)?(chấm|cho)\s+(\w+\s+){0,3}(100|tối đa)\s*điểm\b", _I),
        re.compile(r"\b(cho|chấm)\s+điểm\s+(cao nhất|tối đa|tuyệt đối)\b", _I),
        re.compile(r"\b(cho|chấm|đánh giá)\b.{0,20}\b100\s*/\s*100\b", _I),
        re.compile(r"\bxếp\s+(loại|hạng)\b.{0,30}\b(phù hợp tốt|cao nhất|tốt nhất)\b", _I),
    ],
    "json_injection": [
        re.compile(r"\"\s*\}\s*\]\s*\}"),                                          # closes the extractor's JSON: "}]}
        re.compile(r"\{\s*\"[A-Za-z_]+\"\s*:\s*[^,{}]+,\s*\"[A-Za-z_]+\"\s*:"),     # an inline JSON object with 2+ keys
    ],
}


def find_injections(text: str) -> list[dict]:
    """[{line, kind, text}] for every line that matches; line numbers start at 1."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for kind, patterns in PATTERNS.items():
            if any(p.search(line) for p in patterns):
                out.append({"line": n, "kind": kind, "text": line.strip()[:200]})
                break
    return out


def strip_injections(text: str) -> tuple[str, list[dict]]:
    """(text without the flagged lines, findings). The rest of the text is unchanged, so quotes still match it."""
    found = find_injections(text)
    if not found:
        return text, []
    drop = {f["line"] for f in found}
    kept = [line for n, line in enumerate(text.splitlines(), 1) if n not in drop]
    return "\n".join(kept) + ("\n" if text.endswith("\n") else ""), found
