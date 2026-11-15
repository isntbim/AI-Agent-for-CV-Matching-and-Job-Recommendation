"""O5 metric: is an explanation faithful to the scored facts? (promised at Review 2: "every skill and score the
explanation mentions must exist in the extraction and the score table; the score never changes").

An explanation is faithful when it has no violation of these kinds:
  verdict  - it does not state the pair's verdict, or states another one;
  number   - a number that is not the fit, the rule score, a scored dimension's score or a count of the listed skills
             (numbers inside skill names, e.g. "GA4", are ignored);
  skill    - a known skill name (from any extracted CV or JD) that is not in this pair's facts.
The scores themselves are never produced by the explainer: the service shows the scorer's numbers, so "the score
never changes" holds by construction.

  python -m scripts.o5_faithfulness check o5_011241e0d9 [--dir teacher]   # one file: prints "faithful 1/1" or the problems
  python -m scripts.o5_faithfulness report teacher                         # rate over a folder of outputs
  python -m scripts.o5_faithfulness preds o5_base_test.jsonl o5_adapter_test.jsonl   # generate.py outputs, side by side
"""
from __future__ import annotations

import argparse
import functools
import json
import pickle
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
O5 = ROOT / "output/o5"
VERDICTS = ("Good Fit", "Potential Fit", "No Fit")
# Vietnamese explanations open with the verdict in Vietnamese (user decision 2026-11-10)
VERDICTS_VI = {"Good Fit": "Phù hợp tốt", "Potential Fit": "Có tiềm năng", "No Fit": "Chưa phù hợp"}
VI_LETTERS = re.compile(r"[ăâđêôơưàáạảãầấậẩẫằắặẳẵèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]")


def is_vietnamese(text: str) -> bool:
    return len(VI_LETTERS.findall(unicodedata.normalize("NFC", text).lower())) >= 5
# Everyday words that also occur as extracted "skills": too generic to count as a skill claim.
GENERIC = set("""data design management communication leadership sales marketing research analysis analytics development
teamwork english experience education industry skills skill title job role roles software business customer service
support training reporting planning writing accounting finance financial operations quality testing security cloud web
mobile team work project projects problem solving organization organizational time detail attention creative creativity
strategy strategic technical technology engineering administration office documentation presentation presentations
negotiation coordination collaboration interpersonal verbal written analytical critical thinking adaptability
flexibility mentoring coaching consulting compliance budget budgeting forecasting excel word it hr law""".split())


@functools.lru_cache(maxsize=1)
def vocab() -> list[str]:
    """Skill names from every extracted document of the validation set, longest first (cached to output/o5)."""
    path = O5 / "vocab.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    docs = pickle.loads((ROOT / "output/match_eval/lab_docs.pkl").read_bytes())["docs"]
    names = set()
    for d in docs.values():
        if d is None:
            continue
        dump = d.document.model_dump()
        sk = dump.get("skills")
        items = sk if isinstance(sk, list) else (sk.get("required", []) + sk.get("preferred", [])) if isinstance(sk, dict) else []
        names |= {(s.get("name") or "").strip().lower() for s in items if isinstance(s, dict)}
    # a plain word that the source documents use in lowercase prose (>= 2% of them) is English, not a skill claim
    from scripts import match_eval as m
    texts = list({t for p in m.load_pairs() for t in (p["cv"], p["jd"])})
    def everyday(n: str) -> bool:
        if not n.isalpha() or " " in n:
            return False
        if len(n) <= 2:
            return True
        hits = sum(re.search(r"(?<![\w])" + n + r"(?![\w])", t) is not None for t in texts)
        return hits >= .02 * len(texts)
    names = sorted((n for n in names if 2 <= len(n) <= 40 and n not in GENERIC and not n.isdigit() and not everyday(n)),
                   key=lambda n: (-len(n), n))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    return names


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def _mentions(text: str, term: str) -> bool:
    return re.search(r"(?<![\w+#.])" + re.escape(term) + r"(?![\w+#])", text) is not None


def check(explanation: str, facts: dict) -> list[str]:
    text, problems = _norm(explanation), []
    # the stated fit must be this pair's fit: "47" is also allowed as some other number, so the number check misses it
    problems += [f"fit: says {n}/100 but the fit is {facts['fit']}" for n in re.findall(r"(?:fit|độ phù hợp)\s+(\d+)\s*/\s*100", text)
                 if int(n) != facts["fit"]]
    if is_vietnamese(text):
        # "phù hợp" is everyday Vietnamese, so only the opening label is checked; English labels anywhere are wrong
        want = VERDICTS_VI[facts["verdict"]].lower()
        if not text.startswith(want):
            problems.append(f"verdict: does not open with '{VERDICTS_VI[facts['verdict']]}'")
        problems += [f"verdict: says '{v}' but the verdict is '{facts['verdict']}'" for v in VERDICTS if v.lower() in text]
    else:
        said = [v for v in VERDICTS if v.lower() in text]
        if facts["verdict"].lower() not in text:
            problems.append(f"verdict: does not say '{facts['verdict']}'")
        problems += [f"verdict: says '{v}' but the verdict is '{facts['verdict']}'" for v in said if v != facts["verdict"]]

    listed = [*facts["matched_skills"], *facts["missing_skills"], *facts["found_in_cv_text"], *facts["soft_skills_not_scored"],
              *(r[k] for r in facts.get("requirements", []) for k in ("requirement", "your_skill") if r.get(k)),
              *facts.get("nice_to_have_matched", []), *facts.get("nice_to_have_missing", [])]
    allowed = {_norm(s) for s in listed}
    context = _norm(" ".join(filter(None, [facts.get("job_title"), facts.get("cv_title"), *facts.get("vague_cv_titles", []),
                                            *(r.get("title") for r in facts.get("cv_recent_roles", [])), *facts.get("cv_education", [])])))
    masked = text
    for term in sorted(allowed, key=len, reverse=True):  # remove allowed names first so "sql" inside "sql server" is fine
        masked = re.sub(r"(?<![\w+#.])" + re.escape(term) + r"(?![\w+#])", " ", masked)
    for term in vocab():
        if _mentions(masked, term) and term not in context and not any(term in a for a in allowed):
            problems.append(f"skill: '{term}' is not in this pair's facts")
            masked = masked.replace(term, " ")

    numbers = {facts["fit"], facts["rule_score"], 100}
    numbers |= {v["score"] for v in facts["dimensions"].values() if v["scored"]}
    numbers |= {len(facts["matched_skills"]), len(facts["missing_skills"]), len(facts["found_in_cv_text"]),
                len(facts["matched_skills"]) + len(facts["missing_skills"])}
    numbers |= {float(x) for x in re.findall(r"\d+(?:\.\d+)?", json.dumps({k: facts.get(k) for k in (
        "cv_recent_roles", "cv_education", "job_min_years")}))}  # years and dates the facts state
    for n in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", masked):
        if float(n) not in numbers:
            problems.append(f"number: {n} is not a score or count in the facts")
    return problems


CANT_CHECK = re.compile(r"could not be|couldn't be|not scored|not checked|was not compared|không thể đối chiếu|chưa thể đối chiếu|"
                        r"không đối chiếu được|chưa đối chiếu được|không được đối chiếu|does not clearly state|doesn't clearly state|"
                        r"states no|asks for no|không nêu rõ|chưa nêu rõ|không yêu cầu")
RELATION = (r"\b(?:close to|closer to|closest|similar|overlap\w*|align\w*|match\w*|relevant|relates?|related|same|"
            r"supports?|fits?|suits?|transferable|different|other)\b|gần với|sát với|"
            r"tương tự|đúng hướng|phù hợp|liên quan|giống|khác|cùng|đúng với|đúng nghề|trùng|khớp")
# no bare "ms"/"bs"/"master"/"associate"/"graduate": they are also MS Office, Master Electrician, Associate Engineer titles
DEGREE = (r"\b(?:degrees?|bachelor\w*|master'?s|masters? (?:degree|of|in)|mba|diploma|associate'?s? (?:degree|of|in)|ph\.?d|"
          r"b\.?sc|beng|meng|education\w*)\b|bằng (?:cấp|cử nhân|thạc sĩ|đại học|cao đẳng|kỹ sư|master|bachelor|associate|mba)|"
          r"học vấn|cử nhân|thạc sĩ|tốt nghiệp")  # bare "bằng" is also the preposition "by"
FIELD = r"\b(?:field|industry|sector|discipline|domain)\b|lĩnh vực|ngành"
# "your Java experience partly covers ..." is a skill sentence, so only the candidate's experience as a whole counts
EXPERIENCE_CLAIM = (r"\b(?:your|relevant|related|enough|more|lacks?|lacking|no|little|prior|previous|professional|work|"
                    r"\d+\+? years? of|years of) (?:work )?experience\b|\bexperience (?:requirement|level)\b|"
                    r"kinh nghiệm (?:phù hợp|liên quan|chưa đủ|đủ|vượt|lâu năm|cao hơn|thấp hơn)|"
                    r"(?:thiếu|đủ|nhiều năm|lâu năm|vượt)[^.]{0,30}kinh nghiệm|làm lâu năm|vượt yêu cầu|"
                    r"đáp ứng (?:đủ )?yêu cầu (?:về )?(?:số năm |\d+ năm )?kinh nghiệm|đáp ứng yêu cầu \d+ năm|"
                    r"\bmeets? the [^.,;]{0,25}(?:years?|experience) requirement|nhiều năm|lâu năm|"
                    r"\b(?:many|several) years\b|\blong[- ](?:standing|time|term)\b|\bextensive\b|\bseasoned\b|\bdecades?\b")
# with neither the title nor the field checked, nothing supports "this job is outside your line of work"
APART = (r"nằm ngoài|khác với|khác hướng|khác mảng|\boutside (?:of )?your\b|\bdifferent from your\b|\bfar from your\b|"
         r"\b(?:while|whereas) your (?:career|background|roles?|work|experience)\b")
SENIORITY = (r"\bmore (?:senior|junior)\b|\b(?:over|under)-?qualified\b|\b(?:lower|higher)[- ]level\b|\btoo (?:senior|junior)\b|"
             r"cấp (?:bậc )?thấp hơn|cấp (?:bậc )?cao hơn|(?:senior|junior) hơn|cấp \w+ hoặc (?:cao|thấp) hơn|cùng cấp|"
             r"\bsame (?:level|seniority)\b|\bat (?:the|a) (?:same|similar|lower|higher) level\b")
TOPIC = {"title": r"\bjob titles?\b|\byour titles?\b|chức danh", "experience": r"\bexperience\b|kinh nghiệm",
         "education": r"\bdegrees?\b|\beducation\b|bằng cấp|học vấn", "industry": r"\bfield\b|\bindustry\b|lĩnh vực|ngành"}
UNCHECKED = (r"could not be (?:checked|compared|matched|verified|assessed)|couldn't be (?:checked|compared)|was not (?:checked|compared)|"
             r"(?:không|chưa) thể đối chiếu|(?:không|chưa) đối chiếu được|không được đối chiếu")


def unscored(explanation: str, facts: dict) -> list[str]:
    """A dimension that was NOT scored may only be named as "could not be checked". Any other sentence that talks about it
    (the degree as evidence, the CV title as close to the job, a same/different field, the years as enough) is a claim the
    facts do not support. Heuristic, sentence level; validated against the Opus review of the v2 outputs (2026-11-11)."""
    dims, problems = facts["dimensions"], []
    roles = [_norm(t) for t in [facts.get("cv_title"), *(r.get("title") for r in facts.get("cv_recent_roles", []))] if t]
    schools = [_norm(e) for e in facts.get("cv_education", []) if e]
    for sentence in re.split(r"(?<=[.!?;:])\s+", _norm(explanation)):
        if CANT_CHECK.search(sentence):
            continue
        if not dims["title"]["scored"] and re.search(SENIORITY, sentence):
            problems.append(f"unscored: job title was not scored, but this sentence judges seniority: '{sentence[:90]}'")
        if not dims["title"]["scored"] and not dims["industry"]["scored"] and re.search(APART, sentence):
            problems.append(f"unscored: neither job title nor field was scored, but this sentence sets them apart: '{sentence[:90]}'")
        if not dims["education"]["scored"] and (re.search(DEGREE, sentence) or any(s[:25] in sentence for s in schools)):
            problems.append(f"unscored: education was not scored, but this sentence uses it: '{sentence[:90]}'")
        about_field = re.search(FIELD + r"|line of work|loại công việc|công việc khác", sentence)
        if not dims["title"]["scored"] and re.search(RELATION, sentence) and not (dims["industry"]["scored"] and about_field) and (
                any(r[:25] in sentence for r in roles)
                or re.search(r"\byour (?:job |current |recent |past )?(?:titles?|roles?|background|career)\b|chức danh của bạn", sentence)):
            problems.append(f"unscored: job title was not scored, but this sentence compares it: '{sentence[:90]}'")
        if not dims["industry"]["scored"] and re.search(FIELD, sentence):
            problems.append(f"unscored: industry was not scored, but this sentence names a field: '{sentence[:90]}'")
        if not dims["experience"]["scored"] and re.search(EXPERIENCE_CLAIM, sentence):
            problems.append(f"unscored: experience was not scored, but this sentence judges it: '{sentence[:90]}'")
    return problems


MISSING_CUE = (r"(?:does not|doesn't|do not|did not) (?:yet )?(?:show|list|name|mention|include|cover|state)|not (?:shown|listed|named)|"
               r"\bmissing\b|\black(?:s|ing)?\b|no evidence of|chưa thể hiện|chưa nêu|chưa có|chưa ghi|không có|không thấy|thiếu")
HAVE_CUE = (r"your cv (?:already |also |clearly )?(?:shows|lists|names|has|includes|covers|meets|matches)|you (?:already |also )?have\b|"
            r"bạn (?:đã )?có\b|cv (?:của bạn )?(?:đã |cũng )?(?:có|thể hiện|nêu|đáp ứng)\b")
CONTRAST = r"\bbut\b|\bwhile\b|\bhowever\b|\balthough\b|\bthough\b|\byet\b|;|nhưng|tuy nhiên|trong khi|song\b|dù\b"
ADVICE = (r"\bif\b|\bunless\b|\bwhether\b|\badd\b|\blearn|\bconsider\b|\bshould\b|\bcould\b|\bwould\b|\bany\b|\bdescribe\b|"
          r"\bbefore applying\b|nếu|hãy|nên\b|bổ sung|trước khi")
# "X is required", "listed as a plus", "là điểm cộng": the word says something about the names BEFORE it
COPULA = r"(?:\b(?:is|are|be|was|were)\b[^,]{0,12}|\b(?:listed|lists|marked|named|counts?) (?:them |it )?as (?:an? )?|là |được |đều )$"
NEGATED = r"(?:\bnot\b|n't\b|\bno\b|\bnothing\b|\bnone\b|\bnever\b|chưa|không|chẳng)[^,;]{0,22}$"  # just before a cue: flips it
SPAN_END = r"\bbecause\b|\bsince\b|\bso\b|\bwhich\b|\bthat\b|\bto\b|\bfor\b|\bas\b|\bin\b|\bwith\b|vì\b|do đó|nên\b|khác\b|để\b|cho\b|trong\b|với\b"
SEGMENT = CONTRAST + r"|, and\b|, và\b|, còn\b|ngoài ra|đồng thời|bên cạnh đó"
MUST_WORD = r"\brequired\b|\brequirements?\b|\bmust\b|\bessential\b|bắt buộc|yêu cầu"
NICE_WORD = (r"nice[- ]to[- ]have|\bpreferred\b|\bpreference\b|\bbonus\b|\ba plus\b|\boptional\b|\bdesir(?:able|ed)\b|\bextra\b|ưu tiên|điểm cộng|lợi thế|"
             r"không bắt buộc")
CLOSE_WORD = r"\bclose to\b|\bcloser to\b|\bsimilar to\b|\brelated to\b|\bnear\b|gần với|sát với|tương tự|gần giống"
THIS_JOB = r"this (?:job|role|position|opening)|the job title|vị trí này|công việc này|chức danh này"


def _named(span: str, wanted: set[str], others: set[str]) -> list[str]:
    """Names from `wanted` in the span. Every name is matched longest first and then masked, whichever set it is in: so
    "sql" inside "sql server" is not a hit, and "accounting" does not eat "insurance accounting"."""
    found = []
    for term in sorted(wanted | others, key=len, reverse=True):
        pattern = r"(?<![\w+#.])" + re.escape(term) + r"(?![\w+#])"
        if re.search(pattern, span):
            span = re.sub(pattern, " ", span)
            if term in wanted:
                found.append(term)
    return found


def contradicts(explanation: str, facts: dict) -> list[str]:
    """Claims that contradict a CHECKED fact. check() only tests that names and numbers exist in the facts; this tests
    what is said about them: a matched skill called missing (or the reverse), a nice-to-have called required (or the
    reverse), a met title called "close to", a same field called different, a requirement in years the job never states.
    Heuristic, sentence level. Patterns were written on the validation prompts' outputs and the teacher texts, never
    on the test outputs (2026-11-12)."""
    dims, problems = facts["dimensions"], []
    reqs = facts.get("requirements", [])
    have = {_norm(s) for s in [*facts["matched_skills"], *facts.get("nice_to_have_matched", []),
                               *(r["requirement"] for r in reqs if r["status"] == "met")]}
    lack = {_norm(s) for s in [*facts["missing_skills"], *facts.get("nice_to_have_missing", []),
                               *(r["requirement"] for r in reqs if r["status"] == "not met")]}
    in_text = {_norm(s) for s in facts.get("found_in_cv_text", [])}
    partly = {_norm(r["requirement"]) for r in reqs if r["status"] == "partly met"}
    have, lack = have - lack - partly, lack - have - partly - in_text  # a name on both sides proves nothing
    nice = {_norm(s) for s in [*facts.get("nice_to_have_matched", []), *facts.get("nice_to_have_missing", [])]}
    must = {_norm(r["requirement"]) for r in reqs} | {_norm(s) for s in [*facts["matched_skills"], *facts["missing_skills"]]}
    nice, must = nice - must, must - nice
    job = _norm(facts.get("job_title") or "")
    years = facts.get("job_min_years")

    def scored(name: str) -> int | None:
        return dims[name]["score"] if dims.get(name, {}).get("scored") else None

    for sentence in re.split(r"(?<=[.!?])\s+", _norm(explanation)):
        short = sentence[:90]
        # 1. status: the names listed right after "does not show" / "your CV shows" must be on that side of the facts
        cues = sorted([(m.start(), m.end(), "lack") for m in re.finditer(MISSING_CUE, sentence)]
                      + [(m.start(), m.end(), "have") for m in re.finditer(HAVE_CUE, sentence)]
                      + [(m.start(), m.end(), None) for m in re.finditer(CONTRAST, sentence)])
        for (start, cue_end, side), nxt in zip(cues, [*cues[1:], (len(sentence),)]):
            if side is None or re.search(NEGATED, sentence[:start]) or (side == "have" and (
                    re.search(ADVICE, sentence[:start]) or re.match(r" (?:none|no|nothing|few|little)\b", sentence[cue_end:]))):
                continue  # "nothing is missing", "does not show that you have X", "if you have X, add it" claim nothing
            span = re.split(SPAN_END, sentence[cue_end:nxt[0]])[0][:120]  # only the list itself, not the clause after it
            if side == "lack":
                problems += [f"status: '{t}' is matched in the facts, but this says it is missing: '{short}'" for t in _named(span, have, lack)]
            else:
                problems += [f"status: '{t}' is missing in the facts, but this says the CV has it: '{short}'"
                             for t in _named(span, lack, have | in_text)]
        # 2. importance: a nice-to-have called required, a requirement called nice-to-have. A word governs the names before
        # it in its segment and the ones straight after it, up to the next comma.
        for segment in re.split(SEGMENT, sentence):
            plain = re.sub(r"yêu cầu (?:được )?ưu tiên", "là ưu tiên", segment).replace("preferred requirement", "preferred item")
            plain = re.sub(r"(?:nên|hãy|có thể|cần|sẽ) ưu tiên|ưu tiên (?:học|bổ sung|ứng tuyển|các vị trí|những vị trí|tìm)", " ", plain)  # the verb "prioritise"
            for words, other, wanted, rest, say in ((MUST_WORD, NICE_WORD, nice, must, "is a nice-to-have, but this calls it required"),
                                                    (NICE_WORD, MUST_WORD, must, nice, "is a requirement, but this calls it nice-to-have")):
                if re.search(other, plain):
                    continue
                for m in re.finditer(words, plain):
                    # "the preferred CPA" is about what follows; "CPA is preferred" / "the Excel requirement" about what precedes
                    before = re.search(COPULA, plain[:m.start()]) or m.group().startswith("requirement")
                    governed = (plain[max(0, m.start() - 80):m.start()] if before else
                                re.split(SPAN_END + r"|,|\balongside\b|\bof\b", plain[m.end():])[0][:40])
                    problems += [f"importance: '{t}' {say}: '{short}'" for t in _named(governed, wanted, rest)]
        # 3. a checked dimension said the other way round
        if scored("title") == 100 and (re.search(r"(?:" + CLOSE_WORD + r")[^.]{0,60}(?:" + THIS_JOB + (r"|" + re.escape(job) if job else "") + r")", sentence)
                                       or re.search(r"title (?:difference|gap|mismatch)|different (?:job )?title|titles? (?:do|does) not match|"
                                                    r"khác chức danh|chức danh khác", sentence)):
            problems.append(f"dimension: the job title matches (100), but this says otherwise: '{short}'")
        if scored("industry") == 100 and re.search(r"different (?:field|industry|sector|domain)|another (?:field|industry|sector)|"
                                                   r"khác (?:lĩnh vực|ngành)|(?:lĩnh vực|ngành)(?: nghề)? khác", sentence):
            problems.append(f"dimension: same field in the facts (100), but this says a different field: '{short}'")
        if scored("industry") == 0 and re.search(r"\bsame (?:field|industry|sector)|cùng (?:lĩnh vực|ngành)", sentence) \
                and not re.search(r"\bnot\b|không|chưa", sentence):
            problems.append(f"dimension: different field in the facts (0), but this says the same field: '{short}'")
        for name, topic in (("education", r"degree|bằng cấp|học vấn"), ("experience", r"experience|kinh nghiệm")):
            s = scored(name)
            if s is None:
                continue
            met = re.search(r"(?<!not )(?<!partly )(?<!fully )(?<!n't )\bmeets? the [^.,;]{0,40}(?:" + topic + r")[^.,;]{0,15}requirement|"
                            r"(?<!chưa )(?<!không )(?<!phần )đáp ứng (?:đủ )?yêu cầu (?:về )?(?:số năm )?(?:\d+ năm )?(?:" + topic + r")", sentence)
            unmet = re.search(r"(?:does not|doesn't) (?:fully )?meet the [^.,;]{0,40}(?:" + topic + r")|"
                              r"(?:chưa|không) đáp ứng (?:đủ )?yêu cầu (?:về )?(?:số năm )?(?:\d+ năm )?(?:" + topic + r")", sentence)
            if met and s < 100:
                problems.append(f"dimension: {name} is {s}/100, but this says the requirement is met: '{short}'")
            if unmet and s == 100:
                problems.append(f"dimension: {name} is met (100), but this says it is not: '{short}'")
        if scored("experience") == 100 and re.search(
                r"(?:chưa đủ|không đủ|thiếu)[^.,;]{0,25}(?:kinh nghiệm|năm)|kinh nghiệm[^.,;]{0,30}(?:chưa đủ|không đủ)|"
                r"\bnot enough (?:years|experience)|\bshort of the [^.,;]{0,25}years?|\bfewer years|\bless experience than", sentence):
            problems.append(f"dimension: experience is met (100), but this says it is not enough: '{short}'")
        for name, topic in TOPIC.items():  # "your experience could not be checked" about a dimension that WAS checked
            said = re.search(r"(?:" + topic + r")([^,;]{0,35}?)(?:" + UNCHECKED + r")", sentence)
            if said and scored(name) is not None and not re.search(CONTRAST + r"|\broles?\b|vai trò", said.group(1))                     and not any(scored(o) is None and re.search(t, said.group()) for o, t in TOPIC.items()):
                problems.append(f"dimension: {name} was checked ({scored(name)}/100), but this says it could not be: '{short}'")
        # 4. a requirement in years must be the job's own
        for n in re.findall(r"(?:requires?|required|requirement of|asks? for|minimum of|at least|yêu cầu|tối thiểu|ít nhất)[^.,;]{0,25}?"
                            r"(\d+)\+?[- ](?:years?|năm)|(\d+)\+?[- ](?:years?|năm)[^.,;]{0,25}?(?:requirement|required|requested|asked|yêu cầu)", sentence):
            n = int(n[0] or n[1])
            if n != years and n < 60:
                problems.append(f"years: says the job asks for {n} years, the facts say {years or 'no minimum'}: '{short}'")
    return problems


def violations(explanation: str, facts: dict) -> list[str]:
    """Everything the guard rejects: the service regenerates an explanation with any of these, then falls back to the
    template (scripts.o5_export_sft.fallback). The reported "faithful" rate stays check() alone, comparable with v1-v3."""
    return check(explanation, facts) + unscored(explanation, facts) + contradicts(explanation, facts)


GENERIC_GAPS = {"math", "science", "technical", "listening", "communication", "programming languages", "teamwork",
                "problem solving", "team-oriented and collaborative working environments", "attention to detail"}
JARGON = ("scorer", "scored separately", "rule score", "dimension", "facts", "the model", "algorithm", "basis")
OVERCLAIM = ("easily", "perfect", "ready to apply", "strong candidate", "ideal candidate", "definitely", "guaranteed",
             "weak", "unqualified")


def style(explanation: str, facts: dict) -> list[str]:
    """The strict teacher's style rules (o5-explanation skill). Not part of the faithfulness metric."""
    if is_vietnamese(explanation):
        return style_vi(explanation, facts)
    text, problems = _norm(explanation), []
    words = len(explanation.split())
    if not 85 <= words <= 125:
        problems.append(f"style: {words} words, write 90 to 120")
    problems += [f"style: system word '{w}', say it in the candidate's terms" for w in JARGON if re.search(rf"\b{re.escape(w)}\b", text)]
    problems += [f"style: overclaim or harsh word '{w}'" for w in OVERCLAIM if re.search(rf"\b{re.escape(w)}\b", text)]
    problems += [f"style: mentions '{w}', which is never scored here" for w in ("location", "salary") if w in text]
    missing = {_norm(r["requirement"]) for r in facts.get("requirements", []) if r.get("status") == "not met"}
    problems += [f"style: generic gap '{g}', pick a specific requirement" for g in GENERIC_GAPS & missing if _mentions(text, g)]
    if text.count("not scored") + text.count("could not be") > 1:
        problems.append("style: more than one unscored item mentioned, keep at most one")
    return problems


JARGON_VI = ("điểm quy tắc", "chiều đánh giá", "mô hình", "thuật toán", "hệ thống chấm", "dữ liệu đầu vào", "facts")
OVERCLAIM_VI = ("dễ dàng", "hoàn hảo", "chắc chắn", "đảm bảo", "ứng viên lý tưởng", "ứng viên mạnh", "yếu kém",
                "không đủ năng lực", "kém cỏi")
UNSCORED_VI = "không thể đối chiếu"


def style_vi(explanation: str, facts: dict) -> list[str]:
    """Vietnamese version of style(): a Vietnamese syllable is one space-separated token, about 1 per English word once names are counted."""
    text, problems = _norm(explanation), []
    words = len(explanation.split())
    if not 95 <= words <= 165:
        problems.append(f"style: {words} tiếng (syllables), write 100 to 160")
    problems += [f"style: system word '{w}', say it in the candidate's terms" for w in JARGON_VI if w in text]
    problems += [f"style: overclaim or harsh word '{w}'" for w in OVERCLAIM_VI if w in text]
    problems += [f"style: mentions '{w}', which is never scored here" for w in ("địa điểm", "lương") if w in text]
    missing = {_norm(r["requirement"]) for r in facts.get("requirements", []) if r.get("status") == "not met"}
    problems += [f"style: generic gap '{g}', pick a specific requirement" for g in GENERIC_GAPS & missing if _mentions(text, g)]
    if text.count(UNSCORED_VI) > 1:
        problems.append("style: more than one unscored item mentioned, keep at most one")
    return problems


def load(pair_id: str, folder: str) -> tuple[str, dict]:
    out = json.loads((O5 / folder / f"{pair_id}.json").read_text(encoding="utf-8"))
    facts = json.loads((O5 / "inputs" / f"{pair_id}.json").read_text(encoding="utf-8"))["facts"]
    return out["explanation"], facts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check"); c.add_argument("pair_ids", nargs="+"); c.add_argument("--dir", default="teacher")
    r = sub.add_parser("report"); r.add_argument("dir")
    pr = sub.add_parser("preds"); pr.add_argument("files", nargs="+", type=Path)
    a = p.parse_args()
    if a.cmd == "preds":
        out = {}
        for f in a.files:
            every = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            for lang in ("en", "vi"):  # "_vi" ids were prompted in Vietnamese; reported apart
                rows = [r for r in every if r["id"].endswith("_vi") == (lang == "vi")]
                if not rows:
                    continue
                kinds, ok, words, cut, right_lang, uns, con, clean = {}, 0, [], 0, 0, 0, 0, 0
                for row in rows:
                    facts = json.loads((O5 / "inputs" / f"{row['id'].removesuffix('_vi')}.json").read_text(encoding="utf-8"))["facts"]
                    problems = check(row["prediction"], facts)
                    uns += bool(unscored(row["prediction"], facts))  # reported apart: "faithful" stays comparable with v1/v2
                    con += bool(contradicts(row["prediction"], facts))
                    clean += not violations(row["prediction"], facts)
                    ok += not problems; words.append(len(row["prediction"].split())); cut += row.get("finish_reason") == "length"
                    right_lang += is_vietnamese(row["prediction"]) == (lang == "vi")
                    for pr_ in problems:
                        kinds[pr_.split(":")[0]] = kinds.get(pr_.split(":")[0], 0) + 1
                words.sort()
                out[f"{f.stem}:{lang}"] = {"n": len(rows), "faithful": ok, "rate": round(ok / len(rows), 4), "violations": kinds,
                                           "unscored_flagged": uns, "contradiction_flagged": con, "passes_guard": clean,
                                           "retried": sum(r.get("attempts", 1) > 1 for r in rows), "fallback": sum(bool(r.get("fallback")) for r in rows),
                                           "right_language": right_lang,
                                           "median_words": words[len(words) // 2], "cut_off": cut}
        print(json.dumps(out, indent=1))
        return
    if a.cmd == "check":
        ok = 0
        for pid in a.pair_ids:
            try:
                expl, facts = load(pid, a.dir)
                problems = check(expl, facts) + unscored(expl, facts) + style(expl, facts)
            except Exception as exc:
                problems = [f"file: {type(exc).__name__}: {exc}"]
            ok += not problems
            for pr in problems:
                print(f"{pid}: {pr}")
        print(f"faithful {ok}/{len(a.pair_ids)}")  # faithful AND within the style rules
        sys.exit(0 if ok == len(a.pair_ids) else 1)
    files = sorted((O5 / a.dir).glob("o5_*.json"))
    kinds, bad = {}, 0
    for f in files:
        expl, facts = load(f.stem, a.dir)
        problems = check(expl, facts)
        bad += bool(problems)
        for pr in problems:
            kinds[pr.split(":")[0]] = kinds.get(pr.split(":")[0], 0) + 1
    n = len(files)
    print(json.dumps({"dir": a.dir, "n": n, "faithful": n - bad, "rate": round((n - bad) / n, 4) if n else None,
                      "violations": kinds}, indent=1))


if __name__ == "__main__":
    main()
