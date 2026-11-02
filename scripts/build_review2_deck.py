"""Build the Review 2 deck on the team template (Canva export in knowledge-base/slide-temp, local-only).

Usage (system python: python-pptx lives there, not in .venv): python scripts/build_review2_deck.py [--png]
Writes output/review2/review2_deck.pptx; --png also renders slides through PowerPoint COM to output/review2/deck_png/.
Numbers are copied from output/review2/slide_outline.md, qwen/model_comparison.md and cost_latency.md.
"""

import copy
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "knowledge-base/slide-temp/Blue Modern Business Pitch Deck Presentation.pptx"
OUT = ROOT / "output/review2/review2_deck.pptx"
IMG = ROOT / "output/review2"
BLUE, DARK, LIGHT, CARD_DARK, WHITE = (RGBColor.from_string(c) for c in ("0077FC", "231F20", "F5F5F5", "3A3536", "FFFFFF"))
FONT, MIN_PT = "Inter", 18
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
BANNED = ("Salford", "Wardiere", "reallygreatsite", "Adeline", "Curabitur", "Lorem", "2035", "Rimberio", "Borcelle", "Larana", "Pitch Deck")

prs = Presentation(TEMPLATE)
T = list(prs.slides)  # template slides: 0 title, 2 white content, 3 dark content, 11 closing
BLANK = prs.slide_layouts[6]


def style(run, size, bold=False, color=None):
    run.font.name, run.font.size, run.font.bold = FONT, Pt(size), bold
    if color is not None:
        run.font.color.rgb = color


def chrome_text(slide, page: int | None) -> None:
    """Rewrite the template's header/footer text in place and lift it to the 18 pt minimum."""
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        text = sh.text_frame.text.strip()
        new = {"Salford & Co.": "ATS CV-JD Matching", "www.reallygreatsite.com": "Review 2 - Task 2"}.get(text)
        if text.isdigit() and len(text) == 2 and page is not None:
            new = f"{page:02d}"
        for p in sh.text_frame.paragraphs:
            for i, r in enumerate(p.runs):
                if new is not None:
                    r.text = new if i == 0 else ""
                r.font.name = FONT
                if r.font.size is not None and r.font.size < Pt(MIN_PT):
                    r.font.size = Pt(MIN_PT)


def is_logo(sh) -> bool:
    """The template's top-left brand mark (icon + name), removed at the user's request."""
    return Inches(0.5) < sh.top < Inches(1.1) and sh.left < Inches(2)


def drop_logo(slide) -> None:
    for sh in [sh for sh in slide.shapes if is_logo(sh)]:
        sh._element.getparent().remove(sh._element)


def content_slide(dark: bool, title: str, page: int):
    """New slide = template background + header/footer chrome only (shapes above 1.3 in or below 10 in)."""
    base = T[3] if dark else T[2]
    s = prs.slides.add_slide(BLANK)
    for sh in list(s.shapes):
        sh._element.getparent().remove(sh._element)
    copied = []
    if base._element.cSld.bg is not None:
        copied.append(copy.deepcopy(base._element.cSld.bg))
        s._element.cSld.insert(0, copied[-1])
    for sh in base.shapes:
        if (sh.top < Inches(1.3) or sh.top > Inches(10)) and not is_logo(sh):
            copied.append(copy.deepcopy(sh._element))
            s.shapes._spTree.append(copied[-1])
    for el in (e for c in copied for e in c.iter()):  # icons are image-filled shapes: re-point their image links
        for attr in ("embed", "link"):
            rid = el.get(f"{{{RNS}}}{attr}")
            if rid:
                rel = base.part.rels[rid]
                el.set(f"{{{RNS}}}{attr}", s.part.relate_to(rel.target_ref if rel.is_external else rel.target_part, rel.reltype, rel.is_external))
    chrome_text(s, page)
    s.dark = dark
    text(s, title, 1.02, 1.3, 17.9, 1.2, 46, bold=True)
    return s


def fg(s):
    return WHITE if s.dark else DARK


def text(s, value, x, y, w, h, size=22, bold=False, color=None, align=None, anchor=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    if anchor:
        tf.vertical_anchor = anchor
    for i, line in enumerate(value if isinstance(value, list) else [value]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = align
        style(p.add_run(), size, bold, color or fg(s))
        p.runs[0].text = line
    return tb


def bullets(s, items, x, y, w, h, size=22, gap=10):
    tb = text(s, [f"•  {t}" for t in items], x, y, w, h, size)
    for p in tb.text_frame.paragraphs:
        p.space_after = Pt(gap)
    return tb


def box(s, label, x, y, w, h, fill, size=20, bold=False, color=WHITE, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    b.fill.solid()
    b.fill.fore_color.rgb = fill
    b.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        b.adjustments[0] = 0.08
    tf = b.text_frame
    tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
    for i, line in enumerate(label if isinstance(label, list) else [label]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        style(p.add_run(), size, bold or i == 0 and isinstance(label, list), color)
        p.runs[0].text = line
    return b


def arrow(s, x, y, w=0.55, h=0.45, down=False):
    a = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW if down else MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid()
    a.fill.fore_color.rgb = BLUE
    a.line.fill.background()


def picture(s, rel, x, y, w, h):
    """Fit the image inside the box, aspect kept, centred."""
    iw, ih = Image.open(IMG / rel).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    s.shapes.add_picture(str(IMG / rel), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


def num(cell: str) -> float | None:
    m = re.match(r"\s*\$?(-?\d+(?:\.\d+)?)", cell)
    return float(m.group(1)) if m else None


def table(s, rows, x, y, w, h, widths=None, best=None, col_best=None, size=MIN_PT):
    """rows[0] is the header. best: {row: 'max'|'min'} bolds the best cell of a metric row; col_best does it per column."""
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, cw in enumerate(widths or [w / len(rows[0])] * len(rows[0])):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        vals = [num(c) for c in row[1:]]
        target = None
        if best and i in best and any(v is not None for v in vals):
            target = (max if best[i] == "max" else min)(v for v in vals if v is not None)
        for j, value in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb, color = BLUE, WHITE
            elif s.dark:
                c.fill.fore_color.rgb, color = (CARD_DARK if i % 2 else DARK), WHITE
            else:
                c.fill.fore_color.rgb, color = (LIGHT if i % 2 else WHITE), DARK
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.text_frame.word_wrap = True
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            col = [num(r[j]) for r in rows[1:]] if col_best and j in col_best else []
            col_target = (max if col_best[j] == "max" else min)(v for v in col if v is not None) if col else None
            bold = i == 0 or (target is not None and j > 0 and vals[j - 1] == target) or (i > 0 and col_target is not None and num(value) == col_target)
            style(p.add_run(), size, bold, color)
            p.runs[0].text = value
    return t


def card(s, x, y, w, h, fill, top, head, body, color, top_size=54, body_size=20):
    """Template-style card: big number/value, bold heading, short left-aligned lines."""
    box(s, "", x, y, w, h, fill)
    text(s, top, x + 0.35, y + 0.3, w - 0.7, 1.1, top_size, bold=True, color=color)
    text(s, head, x + 0.35, y + 1.45, w - 0.7, 0.9, 26, bold=True, color=color)
    tb = text(s, body, x + 0.35, y + 2.35, w - 0.7, h - 2.6, body_size, color=color)
    for para in tb.text_frame.paragraphs:
        para.space_after = Pt(8)


def stat(s, value, label, x, y, w, color, size=40, gap=0.85):
    text(s, value, x, y, w, gap, size, bold=True, color=BLUE)
    text(s, label, x, y + gap, w, 0.9, 20, color=color)


def lane(s, label, x, y, w, h, right=False):
    """Outlined container with a blue label, for the architecture lanes."""
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    c.adjustments[0] = 0.06
    c.fill.background()
    c.line.color.rgb, c.line.width = CARD_DARK, Pt(1.5)
    text(s, label, x + w - 6.3 if right else x + 0.3, y + 0.1, 6, 0.5, 22, bold=True, color=BLUE,
         align=PP_ALIGN.RIGHT if right else None)


# ---------------------------------------------------------------- slides (outline order)
slides = []

# 1. Title: edit template slide 1 in place.
title = T[0]
for sh in list(title.shapes):
    if sh.shape_id == 9:  # "2035" pill
        sh._element.getparent().remove(sh._element)
repl = {"Pitch Deck": "ATS CV-JD Matching", "Welcome to Wardiere Inc.": "Review 2 - Task 2",
        "Adeline Palmerston": "Lê Trí Dũng (Leader)  |  Thái Thành Nhân  |  Nguyễn Danh Bằng", "22 December 2035": "30 October 2026",
        "Website": "Group", "www.reallygreatsite.com": "GFA26AI26", "E-mail": "Supervisors", "hello@reallygreatsite.com": "ThS. Nguyễn Xuân Huy, TS. Nguyễn Quốc Trung"}
for sh in title.shapes:
    if sh.has_text_frame and sh.text_frame.text.strip() in repl:
        runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
        runs[0].text = repl[sh.text_frame.text.strip()]
        for r in runs[1:]:
            r.text = ""
        if runs[0].text == "ATS CV-JD Matching":
            runs[0].font.size = Pt(120)
chrome_text(title, None)
drop_logo(title)
title.dark = False
text(title, "Hybrid retrieval + fine-tuned Qwen2.5-7B extractor", 1.04, 6.55, 17, 0.7, 32, color=BLUE, bold=True)
for sh in title.shapes:
    if sh.has_text_frame and sh.text_frame.text.startswith("Lê Trí Dũng"):
        sh.width = Inches(15)
    elif sh.has_text_frame and sh.text_frame.text.strip() == "Presented by":
        sh.top = Inches(7.45)
slides.append(title)

# 2. Agenda
s = content_slide(True, "Agenda", 2)
for k, (head, body) in enumerate((("I. Data", "Sources, cleaning, labelling, imbalance (slides 3-6)"),
                                  ("II. Baselines", "Configuration, baselines, paper vs ours, gaps (7-9)"),
                                  ("III. Method", "Architecture, V1 → V3b results, failures, ablation (10-14)"),
                                  ("IV. Feasibility", "Retrieval, latency and cost, risks, Task 3 plan (15-19)"))):
    box(s, [head, body], 1.02 + k * 4.55, 3.4, 4.3, 5.2, BLUE if k == 0 else CARD_DARK, size=24)
slides.append(s)

# 3. Data sources
s = content_slide(False, "Data sources", 3)
table(s, [["Source", "Type", "Access", "Size", "Used for"],
          ["Online CV PDFs", "CV", "public (online)", "601 files (577 distinct)", "88 gold (human-checked) + teacher pool"],
          ["Saba06 (LiveCareer)", "CV", "public", "2,484 rows → 437 kept", "external teacher labels"],
          ["med2425", "CV", "public", "638 CVs → 157 kept", "external teacher labels"],
          ["CareerViet", "JD", "online (scraped)", "802 JDs: vi 693, mixed 74, en 35", "JD extraction + retrieval corpus"],
          ["med2425 fit pairs", "CV-JD", "public", "59,337 clean pairs (485 CVs x 280 JDs)", "matching validation (3 labels)"]],
      1.02, 2.8, 17.9, 4.6, widths=[3.4, 1.6, 2.6, 5.2, 5.1])
bullets(s, ["CV extraction set 1,106 rows = train 1,058 / val 12 / test 36 (gold, 6 per domain)",
            "JD extraction set 802 rows = train 694 / val 54 / test 54"], 1.02, 7.8, 17.9, 2.0, 24)
slides.append(s)

# 4. Cleaning and labelling
s = content_slide(True, "Cleaning and labelling", 4)
flows = [("CV PDFs", ["601 PDFs", "577 distinct", "88 gold + 489 pool"], ["found online", "exact-hash dedup", "human-checked + teacher"]),
         ("CV datasets", ["2,484 + 638", "filter", "594 CVs"], ["Saba06 + med2425", "other domains, copies, test CVs", "teacher-labelled"]),
         ("Labelling", ["2 teachers", "blind arbiter", "validator"], ["Haiku, Gemini", "+ rule sweeps", "schema, exact quotes, dates"])]
for r, (name, heads, subs) in enumerate(flows):
    y = 2.75 + r * 1.75
    text(s, name, 1.02, y + 0.35, 2.4, 0.7, 22, bold=True, color=BLUE)
    for c, (head, sub) in enumerate(zip(heads, subs)):
        x = 3.5 + c * 3.75
        box(s, [head, sub], x, y, 3.15, 1.4, BLUE if c == 2 else CARD_DARK, size=20)
        if c < 2:
            arrow(s, x + 3.2, y + 0.48, 0.5, 0.45)
text(s, ["Teacher agreement", "n = 1,018 CVs"], 14.95, 2.6, 4.2, 1.0, 20, bold=True, color=WHITE)
for k, (value, label) in enumerate((("0.906", "degree, Cohen's \u03ba"), ("0.662", "domains, mean \u03ba"), ("0.599", "skills, mean Jaccard"))):
    stat(s, value, label, 14.95, 3.75 + k * 1.5, 4.0, LIGHT, size=40, gap=0.75)
for k, (value, label) in enumerate((("CV labels", "711 pass  |  395 partial"), ("JD labels", "605 pass  |  197 partial"))):
    box(s, f"{value}:  {label}", 1.02 + k * 6.75, 8.35, 6.5 if k == 0 else 7.0, 0.95, CARD_DARK, size=20)
slides.append(s)

# 5. Statistics and imbalance
s = content_slide(False, "Statistics and imbalance", 5)
picture(s, "stage1/cv_domain_by_split.png", 1.02, 2.6, 8.8, 4.4)
picture(s, "stage1/cv_degree_by_split.png", 10.1, 2.6, 8.8, 4.4)
table(s, [["Train split", "Largest / smallest", "Ratio"],
          ["CV domain", "marketing 234 / frontend 92", "2.5x"], ["JD domain", "accountant 178 / data analyst 53", "3.4x"],
          ["CV degree", "bachelor 485 / associate 41", "11.8x"], ["Fit pairs (val)", "No Fit 60,313 / Good Fit 5,797", "10.4x"]],
      1.02, 7.15, 12.4, 2.75, widths=[3.0, 7.0, 2.4])
text(s, ["CV test is balanced by construction: 6 per domain (36)."], 13.8, 7.3, 5.1, 2.4, 22)
slides.append(s)

# 6. Imbalance handling and leakage
s = content_slide(True, "Imbalance handling and leakage control", 6)
bullets(s, ["Domains: mild (2.5-3.4x), smallest CV domain keeps 92 training CVs → no domain resampling",
            "Sources: 594 external labels are denser than gold → upsample the 86 gold CVs x4, train only",
            "Val/test untouched; test stratified 6 CVs per domain; bootstrap 95% CI on every metric",
            "Leakage: 264 copies of our CV PDFs + 84 scorer-test CVs removed from external data; val/test near-duplicates max 8% 5-gram overlap"],
        1.02, 2.8, 8.6, 7.0, 22, gap=14)
table(s, [["Model", "Train mix", "CV skill F1", "Skills / CV"],
          ["V2", "all, no upsampling", "0.55", "25.1"], ["V3a", "no external, gold x4", "0.64", "19.1"],
          ["V3b", "all, gold x4", "0.65", "25.1"], ["Gold labels", "-", "-", "18.0"]],
      10.0, 3.0, 8.95, 3.6, widths=[1.9, 3.35, 1.9, 1.8], col_best={2: "max"})
text(s, "Before/after: gold x4 moves CV skill F1 from 0.55 to 0.64-0.65.", 10.0, 7.0, 8.95, 1.2, 22, color=LIGHT)
slides.append(s)

# 7. Environment and configuration
s = content_slide(False, "Environment and configuration", 7)
table(s, [["Item", "Setting"],
          ["Base model", "Qwen2.5-7B-Instruct, 7.6B parameters, frozen"],
          ["Fine-tuning", "QLoRA 4-bit, LoRA r=16, alpha=32, all linear layers"],
          ["Optimiser", "AdamW 8-bit, lr 2e-4 cosine, 3% warmup, 1 epoch"],
          ["Batch / length", "1 x grad-accum 16; max 16,384 tokens (longer rows dropped)"],
          ["Loss / selection", "answer tokens only; best checkpoint by validation loss; seed 3407 (single)"],
          ["Hardware", "A100 80GB (Runpod), peak 10.4 GB during training"],
          ["Generation", "vLLM greedy, repetition penalty 1.05, same 90 test docs for every model"],
          ["Retrieval", "BGE-M3 (1,024-d) + BGE reranker FP16; BM25 k1 1.5, b 0.75; RRF k=60, rerank top 20"]],
      1.02, 2.8, 17.9, 6.9, widths=[3.6, 14.3])
slides.append(s)

# 8. Baselines and paper vs ours
s = content_slide(True, "Baselines and paper vs ours", 8)
table(s, [["System", "Task", "Reported", "Ours, same metric family"],
          ["BM25 (baseline 1)", "CV → JD retrieval", "-", "MRR@10 0.746"],
          ["Dense BGE-M3 (baseline 2)", "CV → JD retrieval", "-", "MRR@10 0.810"],
          ["Raw Qwen2.5-7B (extraction baseline)", "CV / JD extraction", "-", "skill F1 CV 0.49, JD 0.33"],
          ["ChatGLM-9B + LoRA, resume NER (2024)", "entity extraction", "F1 95.14", "V3b skill F1 0.65 (exact quotes + schema)"],
          ["LoRA/QLoRA resume screening (2026)", "24-class classification", "macro-F1 0.913", "V3b CV domain F1 0.76"],
          ["ConFit (2024)", "resume-job ranking", "+19 nDCG@10 vs BM25", "+9.5 MRR@10 vs BM25"]],
      1.02, 2.8, 17.9, 5.9, widths=[6.2, 3.7, 3.4, 4.6])
text(s, "[verify each literature figure in the paper before presenting] Tasks and metrics differ; not like-for-like.",
     1.02, 9.0, 17.9, 0.8, 18, color=LIGHT)
slides.append(s)

# 9. Gap analysis
s = content_slide(False, "Gap analysis", 9)
table(s, [["Gap vs literature", "Effect", "Our response"],
          ["Small gold test (36 CV / 54 JD)", "CI about +/-12 pts", "CI on every number; +48 reserved CVs later"],
          ["Silver labels (LLM teachers)", "ceiling = teacher quality", "agreement reported: kappa 0.91 / 0.66, Jaccard 0.60"],
          ["Stricter task (quotes, schema, 86% Vietnamese JDs)", "F1 not comparable to NER", "raw vs fine-tuned on the same test + scorer"],
          ["Single seed", "no run-to-run variance", "bootstrap CI over documents; multi-seed planned"],
          ["No public Vietnamese JD benchmark", "JD numbers cannot be placed", "test split + scorer kept reusable"]],
      1.02, 2.8, 17.9, 6.6, widths=[6.6, 4.8, 6.5])
slides.append(s)

# 10. Architecture: the proposal's two phases; dashed boxes are planned (Task 3).
s = content_slide(True, "Architecture", 10)


def flow(s, y, steps, blue):
    for c, (head, sub, planned) in enumerate(steps):
        b = box(s, [head, sub], 1.4 + c * 3.55, y, 3.0, 1.3, DARK if planned else BLUE if c == blue else CARD_DARK,
                size=20, color=LIGHT if planned else WHITE)
        if planned:
            b.line.color.rgb, b.line.width, b.line.dash_style = LIGHT, Pt(1.5), MSO_LINE.DASH
        if c < len(steps) - 1:
            arrow(s, 1.4 + c * 3.55 + 3.05, y + 0.43, 0.45, 0.45)


lane(s, "Phase 1 · Offline: prepare once", 1.02, 2.4, 17.9, 2.25)
flow(s, 3.05, (("Documents", "1,106 CVs, 802 JDs", False), ("Teacher labels", "2 LLMs + arbiter + validator", False),
               ("QLoRA SFT", "Qwen2.5-7B, 1,858 rows", False), ("Extraction", "Qwen2.5-7B + V3b adapter", False),
               ("Indexes", "bge-m3 on pgvector + BM25", False)), blue=2)
lane(s, "Phase 2 · Online: per request", 1.02, 4.85, 17.9, 2.25, right=True)
flow(s, 5.5, (("Request", "CV → jobs, ready records", False), ("Hybrid retrieval", "dense 50 + BM25 50, RRF", False),
              ("Rerank", "BGE, top 20", False), ("ATS scorer", "7 rule-based scores", False),
              ("LLM rationale", "explains, never changes scores", True)), blue=3)
text(s, ["Dashed = planned (Task 3): LangGraph orchestration with retry, match history, JD → candidates, web UI",
         "Order changed: QLoRA on the extractor first (measured bottleneck); rationale adapter in Task 3"],
     1.02, 7.2, 17.9, 0.9, 18, color=LIGHT)
for k, (value, label) in enumerate((("361", "cleaned JDs in corpus"), ("88", "gold CV queries"), ("2.18 GiB", "index, Azure vector store"),
                                    ("472 ms", "median per CV query"))):
    stat(s, value, label, 1.02 + k * 4.5, 8.35, 4.3, LIGHT, size=30, gap=0.6)
slides.append(s)

# 11. Results
s = content_slide(False, "Results: raw Qwen → V3b (test set)", 11)
table(s, [["Metric", "Raw Qwen", "V2", "V3a", "V3b"],
          ["CV skill F1 (n=36)", "0.49 [0.41, 0.57]", "0.55", "0.64", "0.65 [0.58, 0.71]"],
          ["CV skill precision / recall", "0.53 / 0.45", "0.52 / 0.58", "0.66 / 0.62", "0.56 / 0.76"],
          ["CV domain F1", "0.51", "0.64", "0.68", "0.76"],
          ["CV schema-valid", "0.00", "0.31", "0.28", "0.42"],
          ["CV outputs looping (of 36)", "1", "7", "4", "1"],
          ["JD skill F1 (n=54)", "0.33 [0.24, 0.42]", "0.70", "0.69", "0.73 [0.69, 0.77]"],
          ["JD domain F1", "0.68", "0.85", "0.82", "0.87"],
          ["JD schema-valid", "0.00", "0.24", "0.39", "0.32"]],
      1.02, 2.6, 17.9, 6.0, widths=[5.5, 3.3, 2.6, 2.6, 3.9], best={1: "max", 3: "max", 4: "max", 5: "min", 6: "max", 7: "max", 8: "max"})
text(s, ["Same QLoRA setup, data differs. V2: all labels · V3a: no external CVs, gold x4 · V3b: all labels, gold x4",
         "Value [95% bootstrap CI]. Raw vs V3b CIs do not overlap for CV and JD skill F1. Single seed."], 1.02, 8.75, 17.9, 1.0, 18)
slides.append(s)

# 12. Failure cases
s = content_slide(True, "Failure cases, grouped (counts of test documents)", 12)
table(s, [["CV (n=36)", "Raw", "V2", "V3a", "V3b"], ["truncated (loop)", "1", "7", "4", "1"],
          ["schema invalid", "35", "18", "22", "20"], ["invented quotes", "33", "7", "11", "15"],
          ["missed skills", "34", "23", "28", "29"], ["extra skills", "24", "28", "28", "31"],
          ["domain wrong", "23", "19", "18", "19"]],
      1.02, 2.7, 8.7, 4.9, widths=[3.5, 1.3, 1.3, 1.3, 1.3], best={i: "min" for i in range(1, 7)})
table(s, [["JD (n=54)", "Raw", "V2", "V3a", "V3b"], ["task mix-up", "44", "0", "0", "0"],
          ["wrong types", "6", "0", "0", "0"], ["schema invalid", "52", "41", "33", "37"],
          ["invented quotes", "36", "5", "6", "5"], ["domain wrong", "22", "14", "15", "14"]],
      10.25, 2.7, 8.7, 4.2, widths=[3.5, 1.3, 1.3, 1.3, 1.3], best={i: "min" for i in range(1, 6)})
bullets(s, ["Fine-tuning removes type and task errors; what remains is skill-list disagreement",
            "About half of missed skills are near-name variants: relaxed F1 V3b CV 0.70 vs exact 0.65"], 1.02, 8.0, 17.9, 1.9, 22)
slides.append(s)

# 13. Error → change → result
s = content_slide(False, "From errors to V2, V3a, V3b", 13)
table(s, [["Error seen", "Change", "Result (CV test)"],
          ["7/36 CV outputs loop to the token limit", "repetition penalty 1.05 at generation", "V3b: 1 loop"],
          ["V2 skill F1 flat (0.55); 25.1 skills/CV vs gold 18.0", "V3a: drop external labels, gold x4", "F1 0.64, precision 0.52 → 0.66"],
          ["V3a discards 594 labelled CVs", "V3b: keep them, gold x4", "F1 0.65, recall 0.76, schema-valid 0.42"]],
      1.02, 2.7, 17.9, 3.6, widths=[6.6, 5.6, 5.7])
picture(s, "qwen/loss_curves.png", 1.02, 6.55, 10.4, 3.5)
text(s, ["Validation loss: V2 0.069, V3a 0.065, V3b 0.063.", "Next (V4): rewrite external labels in gold style for V3a precision with V3b recall."],
     11.8, 6.8, 7.1, 3.0, 22)
slides.append(s)

# 14. Ablation
s = content_slide(True, "Ablation: one change at a time", 14)
table(s, [["Retrieval stage", "MRR@10", "NDCG@10"], ["BM25", "0.746", "0.664"], ["Dense", "0.810", "0.758"],
          ["Hybrid (RRF)", "0.789", "0.734"], ["Hybrid + rerank", "0.841", "0.773"]],
      1.02, 2.7, 8.6, 3.9, widths=[4.0, 2.3, 2.3], col_best={1: "max", 2: "max"})
table(s, [["Extraction step", "CV F1", "JD F1"], ["Raw Qwen", "0.49", "0.33"], ["+ QLoRA, all data (V2)", "0.55", "0.70"],
          ["drop external, gold x4 (V3a)", "0.64", "0.69"], ["keep external, gold x4 (V3b)", "0.65", "0.73"]],
      10.3, 2.7, 8.65, 3.9, widths=[4.85, 1.9, 1.9], col_best={1: "max", 2: "max"})
bullets(s, ["Skill F1 columns. Retrieval: primary pools, 88 gold CV queries",
            "Regressions after rerank: data-analysis NDCG@10 0.656 → 0.623; frontend MRR 0.917 → 0.881"], 1.02, 7.1, 17.9, 2.5, 22)
slides.append(s)

# 15. Retrieval results
s = content_slide(False, "Retrieval results", 15)
table(s, [["Setting", "Dense", "BM25", "Hybrid", "Reranked"],
          ["Primary pools, MRR@10", "0.810", "0.746", "0.789", "0.841"],
          ["Expanded pools (+528 negatives), MRR@10", "0.794", "0.734", "-", "0.819"],
          ["Full corpus, known-positive Recall@50", "0.590", "0.455", "0.592", "-"]],
      1.02, 2.8, 17.9, 3.4, widths=[7.5, 2.6, 2.6, 2.6, 2.6], best={1: "max", 2: "max", 3: "max"})
bullets(s, ["Full corpus: judged coverage@50 only 0.105 → no reliable full-corpus NDCG",
            "Labels here: 1,672 LLM-judged pairs; next validation: 59,337 clean med2425 fit pairs"], 1.02, 6.8, 17.9, 2.6, 24)
slides.append(s)

# 16. Feasibility
s = content_slide(True, "Feasibility: latency, size, cost", 16)
table(s, [["Component", "Measured"],
          ["Retrieval per CV query (end-to-end)", "median 472 ms, p95 882 ms"],
          ["Rerank stage / BM25 stage", "median 338 ms / 9 ms"],
          ["Reranker GPU memory", "1.24 GiB"],
          ["Extractor size", "7.6B base (frozen) + 154 MB LoRA adapter"],
          ["Extraction, 90 test docs (A100, vLLM batched)", "raw 266 s, V3b 294 s (about 3.3 s per doc)"],
          ["Training (A100 80GB, 1 epoch)", "V2 63 min $1.89; V3a 51 min $1.52; V3b 74 min $2.21"],
          ["Training peak GPU memory", "10.4 GB"]],
      1.02, 2.7, 17.9, 6.6, widths=[7.6, 10.3])
slides.append(s)

# 17. Limitations and risks
s = content_slide(False, "Limitations and risks", 17)
for k, (head, body) in enumerate((
        ("Data", ["395 CV + 197 JD labels partial", "test 36 CV / 54 JD: CI about \u00b112 pts"]),
        ("Model", ["single seed", "V3a vs V3b gap inside the CI", "V3b over-predicts: 25 vs 18 skills/CV", "1 of 36 CVs still loops"]),
        ("Inputs", ["cv_027: 29,618 tokens, 8,192 kept", "52 reranker truncations", "JD prompt asks English; 86% of JDs Vietnamese"]),
        ("Benchmark", ["no public Vietnamese JD benchmark", "possible Saba06 / LiveCareer overlap with literature"]))):
    fill, color = ((BLUE, WHITE), (DARK, WHITE), (LIGHT, DARK), (LIGHT, DARK))[k]
    card(s, 1.02 + k * 4.55, 2.75, 4.3, 7.0, fill, f"0{k + 1}.", head, [f"\u2022 {b}" for b in body], color)
slides.append(s)

# 18. Task 3 plan
s = content_slide(True, "Task 3 plan", 18)
rail = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(2.5), Inches(4.3), Inches(15), Inches(0.08))
rail.fill.solid()
rail.fill.fore_color.rgb = LIGHT
rail.line.fill.background()
for k, (head, body) in enumerate((("Deploy", ["V3b extractor + retrieval + scorer as one service"]),
                                  ("V4 labels", ["rewrite external labels in gold style", "retrain: about 75 min on A100"]),
                                  ("Evaluation", ["+48 reserved CVs", "59,337 fit-pair validation", "multi-seed runs"]),
                                  ("Paper", ["write-up of data, method, results"]))):
    cx = 2.5 + k * 5.0
    dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 0.8), Inches(3.54), Inches(1.6), Inches(1.6))
    dot.fill.solid()
    dot.fill.fore_color.rgb = BLUE if k % 2 == 0 else WHITE
    dot.line.fill.background()
    tf = dot.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    style(tf.paragraphs[0].add_run(), 22, True, WHITE if k % 2 == 0 else DARK)
    tf.paragraphs[0].runs[0].text = f"Phase {k + 1}"
    text(s, head, cx - 2.2, 5.5, 4.4, 0.7, 28, bold=True, align=PP_ALIGN.CENTER)
    tb = text(s, body, cx - 2.2, 6.25, 4.4, 2.2, 20, align=PP_ALIGN.CENTER, color=LIGHT)
    for para in tb.text_frame.paragraphs:
        para.space_after = Pt(6)
    text(s, "[owner, week]", cx - 2.2, 8.0, 4.4, 0.6, 20, bold=True, align=PP_ALIGN.CENTER, color=BLUE)
slides.append(s)

# 19. Summary
s = content_slide(False, "Summary", 19)
for k, (big, head, body) in enumerate((("0.49\u21920.65", "CV skill F1", ["raw Qwen \u2192 V3b, n = 36", "CIs do not overlap"]),
                                       ("0.33\u21920.73", "JD skill F1", ["raw Qwen \u2192 V3b, n = 54", "task mix-ups 81% \u2192 0%"]),
                                       ("0.841", "MRR@10", ["hybrid + rerank", "BM25 baseline 0.746"]),
                                       ("0.906", "Degree \u03ba", ["agreement of the 2 LLM teachers", "n = 1,018 CVs"]))):
    fill, color = ((BLUE, WHITE), (BLUE, WHITE), (DARK, WHITE), (LIGHT, DARK))[k]
    card(s, 1.02 + k * 4.55, 2.75, 4.3, 5.0, fill, big, head, body, color, top_size=38)
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.02), Inches(8.2), Inches(0.12), Inches(1.4))
bar.fill.solid()
bar.fill.fore_color.rgb = BLUE
bar.line.fill.background()
text(s, ["Fine-tuning alone (V2) barely moved CV skills.", "The gain came from diagnosing label style and rebalancing sources."],
     1.4, 8.15, 17.5, 1.5, 26, bold=True)
slides.append(s)

# 20. Closing: edit template slide 12 in place.
close = T[11]
for sh in list(close.shapes):
    if sh.shape_id == 5:  # "contact us" pill
        sh._element.getparent().remove(sh._element)
repl = {"Let’s Build the Future Together": "Thank you. Questions?", "22 December 2035": "30 October 2026", "Website": "Group",
        "www.reallygreatsite.com": "GFA26AI26", "E-mail": "Supervisors", "hello@reallygreatsite.com": "ThS. Nguyễn Xuân Huy, TS. Nguyễn Quốc Trung"}
for sh in close.shapes:
    if sh.has_text_frame and sh.text_frame.text.strip() in repl:
        runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
        runs[0].text = repl[sh.text_frame.text.strip()]
        for r in runs[1:]:
            r.text = ""
chrome_text(close, 20)
drop_logo(close)
slides.append(close)

# 21. Backup
s = content_slide(True, "Backup: expected questions", 21)
bullets(s, ["Who labelled? Two LLM teachers + arbiter; 88 gold CVs human-checked (val/test are gold only)",
            "Why not test on teacher labels? They disagree (domain kappa 0.66) and share the training style bias",
            "Split before processing? Yes; duplicates removed across sets, near-duplicate overlap max 8%",
            "Significance? Bootstrap 95% CI over documents; raw vs V3b skill F1 CIs do not overlap; single seed",
            "Offline sample count? 601 CV PDFs found online (577 distinct), no partner data"], 1.02, 2.8, 17.9, 7.0, 24, gap=16)
slides.append(s)

# Reorder to the slides above and drop the unused template slides.
ids = prs.slides._sldIdLst
by_part = {prs.part.related_part(e.rId): e for e in list(ids)}
keep = [by_part[sl.part] for sl in slides]
for e in list(ids):
    if e not in keep:
        prs.part.drop_rel(e.rId)
    ids.remove(e)
for e in keep:
    ids.append(e)

# Self-check.
assert len(prs.slides) == len(slides) == 21, len(prs.slides)
for n, sl in enumerate(prs.slides, 1):
    for sh in sl.shapes:
        frames = [sh.text_frame] if sh.has_text_frame else []
        if sh.has_table:
            frames = [c.text_frame for row in sh.table.rows for c in row.cells]
        for tf in frames:
            assert not any(b in tf.text for b in BANNED), (n, tf.text[:60])
            for p in tf.paragraphs:
                for r in p.runs:
                    assert r.font.size is None or r.font.size >= Pt(MIN_PT) or not r.text.strip(), (n, r.text, r.font.size.pt)
    for rel in sl.part.rels.values():
        assert rel.is_external or rel.target_part is not None, (n, rel.rId)
prs.save(OUT)
print("OK", OUT)

if "--png" in sys.argv:
    png = OUT.parent / "deck_png"
    png.mkdir(exist_ok=True)
    ps = (f'$pp = New-Object -ComObject PowerPoint.Application; $d = $pp.Presentations.Open("{OUT}", $true, $false, $false); '
          f'foreach ($i in 1..$d.Slides.Count) {{ $d.Slides($i).Export("{png}\\slide_$($i.ToString("00")).png", "PNG", 1600, 900) }}; '
          f'$d.Close(); $pp.Quit()')
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print("PNG", png)
