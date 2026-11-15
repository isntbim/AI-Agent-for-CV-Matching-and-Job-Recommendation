"""Build the Review 3 presentation deck from output/review3/slide_outline.md.

Usage (system python: python-pptx and matplotlib live there):
    python scripts/build_review3_deck.py [--no-png]

Outputs:
    - output/review3/charts/*.png
    - output/review3/review3_deck.pptx
    - output/review3/deck_png/slide_01.png .. slide_22.png (via PowerPoint COM)
"""

import copy
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "knowledge-base/slide-temp/Blue Modern Business Pitch Deck Presentation.pptx"
OUTLINE = ROOT / "output/review3/slide_outline.md"
OUT_DIR = ROOT / "output/review3"
CHARTS_DIR = OUT_DIR / "charts"
PNG_DIR = OUT_DIR / "deck_png"
OUT = OUT_DIR / "review3_deck.pptx"
LOSS_CURVES = ROOT / "output/review2/qwen/loss_curves.png"

BLUE = RGBColor.from_string("0077FC")
CYAN = RGBColor.from_string("38BDF8")
DARK = RGBColor.from_string("231F20")
LIGHT = RGBColor.from_string("F5F5F5")
CARD_DARK = RGBColor.from_string("3A3536")
PANEL_DARK = RGBColor.from_string("2E2A2B")
WHITE = RGBColor.from_string("FFFFFF")
MUTED_DARK = RGBColor.from_string("B0B5BD")
MUTED_LIGHT = RGBColor.from_string("555B66")
BORDER_LIGHT = RGBColor.from_string("D1D5DB")
BORDER_DARK = RGBColor.from_string("52525B")
GREEN_ACCENT = RGBColor.from_string("16A34A")
AMBER_ACCENT = RGBColor.from_string("D97706")

FONT = "Inter"
MIN_PT = 18
RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
BANNED = (
    "Salford",
    "Wardiere",
    "reallygreatsite",
    "Adeline",
    "Palmerston",
    "Curabitur",
    "Lorem",
    "2035",
    "Rimberio",
    "Borcelle",
    "Larana",
    "Pitch Deck",
    "hello@",
)

NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Charts (matplotlib, sized 1:1 in inches so 18 pt in matplotlib = 18 pt on slide)
# ---------------------------------------------------------------------------

def _setup_chart_style(dark: bool) -> dict:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.titleweight": "bold",
        "axes.labelsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })
    if dark:
        return {
            "bg": "#2E2A2B",
            "fg": "#FFFFFF",
            "grid": "#4B4648",
            "muted": "#9CA3AF",
            "accent": "#0077FC",
            "accent2": "#38BDF8",
            "amber": "#F59E0B",
        }
    return {
        "bg": "#FFFFFF",
        "fg": "#231F20",
        "grid": "#E5E7EB",
        "muted": "#6B7280",
        "accent": "#0077FC",
        "accent2": "#0284C7",
        "amber": "#D97706",
    }


def _style_ax(ax, st: dict):
    ax.set_facecolor(st["bg"])
    ax.tick_params(colors=st["fg"], length=4)
    ax.xaxis.label.set_color(st["fg"])
    ax.yaxis.label.set_color(st["fg"])
    ax.title.set_color(st["fg"])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(st["muted"])
        ax.spines[spine].set_linewidth(1.0)


def generate_charts() -> dict[str, Path]:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # 1. Slide 6: Seven rule score weights (%)
    st = _setup_chart_style(dark=False)
    fig, ax = plt.subplots(figsize=(8.6, 6.7), dpi=220)
    fig.patch.set_facecolor(st["bg"])
    _style_ax(ax, st)
    labels = ["Salary", "Location", "Industry", "Education", "Experience", "Title", "Skills"]
    vals = [7, 8, 10, 10, 15, 20, 30]
    colors = ["#94A3B8", "#94A3B8", "#64748B", "#64748B", "#38BDF8", "#0284C7", "#0077FC"]
    bars = ax.barh(labels, vals, color=colors, height=0.62)
    ax.set_xlim(0, 36)
    ax.set_xlabel("Weight (%)")
    ax.set_title("Seven Rule Score Weights (%)", pad=14)
    ax.grid(axis="x", color=st["grid"], linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.7, bar.get_y() + bar.get_height() / 2, f"{v}%",
                va="center", ha="left", fontweight="bold", fontsize=18, color=st["fg"])
    fig.tight_layout(pad=1.0)
    p = CHARTS_DIR / "slide06_weights.png"
    fig.savefig(p, dpi=220, facecolor=st["bg"])
    plt.close(fig)
    paths["slide06"] = p

    # 2. Slide 7: Funnel 80,017 -> 59,337 + class-share bar (77.6 / 15.9 / 6.6 %)
    st = _setup_chart_style(dark=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 7.0), dpi=220, gridspec_kw={"height_ratios": [1.0, 1.1]})
    fig.patch.set_facecolor(st["bg"])

    # Funnel top subplot (centered horizontal bars)
    _style_ax(ax1, st)
    stages = ["Clean pairs", "Public pairs"]
    counts = [59337, 80017]
    bar_colors = [st["accent"], "#64748B"]
    for idx, (lbl, cnt, col) in enumerate(zip(stages, counts, bar_colors)):
        left = (80017 - cnt) / 2
        ax1.barh(idx, cnt, left=left, height=0.52, color=col)
        ax1.text(40008.5, idx, f"{cnt:,}", va="center", ha="center",
                 fontweight="bold", fontsize=18, color="#FFFFFF")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(stages)
    ax1.set_xlim(0, 86000)
    ax1.set_xticks([0, 40000, 80000])
    ax1.set_xticklabels(["0", "40,000", "80,000"])
    ax1.set_xlabel("Pair count")
    ax1.set_title("Leakage Funnel (25.8% dropped)", pad=10)
    ax1.annotate("", xy=(40008.5, 0.28), xytext=(40008.5, 0.72),
                 arrowprops=dict(arrowstyle="->", color=st["accent2"], lw=2.5))

    # Class share bottom subplot
    _style_ax(ax2, st)
    cls_names = ["Good", "Potential", "No Fit"]
    cls_vals = [6.6, 15.9, 77.6]
    cls_cols = ["#22C55E", st["amber"], st["accent"]]
    bars = ax2.barh(cls_names, cls_vals, color=cls_cols, height=0.55)
    ax2.set_xlim(0, 96)
    ax2.set_xlabel("Share (%)")
    ax2.set_title("Fit-Label Class Share (59,337 pairs)", pad=10)
    ax2.grid(axis="x", color=st["grid"], linestyle="--", linewidth=0.8)
    ax2.set_axisbelow(True)
    for bar, v in zip(bars, cls_vals):
        ax2.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%",
                 va="center", ha="left", fontweight="bold", fontsize=18, color=st["fg"])

    fig.tight_layout(pad=1.2)
    p = CHARTS_DIR / "slide07_data.png"
    fig.savefig(p, dpi=220, facecolor=st["bg"])
    plt.close(fig)
    paths["slide07"] = p

    # 3. Slide 8: CV skill F1 by version with CI bars
    st = _setup_chart_style(dark=False)
    fig, ax = plt.subplots(figsize=(7.3, 6.7), dpi=220)
    fig.patch.set_facecolor(st["bg"])
    _style_ax(ax, st)
    versions = ["Raw", "V2", "V3b", "V4 +\nrepair/retry"]
    f1s = [0.49, 0.55, 0.65, 0.73]
    # CI bars where given: Raw [0.41, 0.57], V2 None, V3b [0.58, 0.71], V4 [0.68, 0.77]
    err_lo = [0.49 - 0.41, 0.0, 0.65 - 0.58, 0.73 - 0.68]
    err_hi = [0.57 - 0.49, 0.0, 0.71 - 0.65, 0.77 - 0.73]
    tops = [0.57, 0.55, 0.71, 0.77]
    b_cols = ["#94A3B8", "#64748B", "#38BDF8", "#0077FC"]
    x_pos = range(len(versions))
    bars = ax.bar(x_pos, f1s, color=b_cols, width=0.56)
    for idx in (0, 2, 3):
        ax.errorbar(idx, f1s[idx], yerr=[[err_lo[idx]], [err_hi[idx]]],
                    fmt="none", ecolor="#231F20", elinewidth=2.2, capsize=7, capthick=2.2)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(versions)
    ax.set_ylim(0, 0.88)
    ax.set_xlabel("Extractor version")
    ax.set_ylabel("CV skill F1")
    ax.set_title("CV Skill F1 by Version (n = 36, 95% CI)", pad=14)
    ax.grid(axis="y", color=st["grid"], linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    for idx, (v, top) in enumerate(zip(f1s, tops)):
        ax.text(idx, top + 0.025, f"{v:.2f}", ha="center", va="bottom",
                fontweight="bold", fontsize=18, color=st["fg"])
    fig.tight_layout(pad=1.0)
    p = CHARTS_DIR / "slide08_f1.png"
    fig.savefig(p, dpi=220, facecolor=st["bg"])
    plt.close(fig)
    paths["slide08"] = p

    # 4. Slide 9: NDCG@10 bars (BM25, TF-IDF, Rule scorer, Proposed V1, Proposed V2)
    st = _setup_chart_style(dark=True)
    fig, ax = plt.subplots(figsize=(7.8, 7.0), dpi=220)
    fig.patch.set_facecolor(st["bg"])
    _style_ax(ax, st)
    methods = ["BM25", "TF-IDF", "Rule\nscorer", "Proposed\nV1", "Proposed\nV2"]
    ndcg = [0.436, 0.468, 0.415, 0.564, 0.630]
    m_cols = ["#64748B", "#64748B", "#64748B", st["accent2"], st["accent"]]
    x_pos = range(len(methods))
    ax.bar(x_pos, ndcg, color=m_cols, width=0.58)
    ax.errorbar(4, 0.630, yerr=[[0.630 - 0.600], [0.661 - 0.630]],
                fmt="none", ecolor="#FFFFFF", elinewidth=2.2, capsize=7, capthick=2.2)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 0.78)
    ax.set_xlabel("Method")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Matching NDCG@10 (31,150 pairs)", pad=14)
    ax.grid(axis="y", color=st["grid"], linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    for idx, v in enumerate(ndcg):
        top = 0.661 if idx == 4 else v
        ax.text(idx, top + 0.02, f"{v:.3f}", ha="center", va="bottom",
                fontweight="bold", fontsize=18, color=st["fg"])
    fig.tight_layout(pad=1.0)
    p = CHARTS_DIR / "slide09_ndcg.png"
    fig.savefig(p, dpi=220, facecolor=st["bg"])
    plt.close(fig)
    paths["slide09"] = p

    # 5. Slide 10: Step chart of schema-valid share 29.0 -> 89.9 -> 98.2 -> 100 %
    st = _setup_chart_style(dark=False)
    fig, ax = plt.subplots(figsize=(7.8, 6.8), dpi=220)
    fig.patch.set_facecolor(st["bg"])
    _style_ax(ax, st)
    steps = ["Raw", "+ Repair", "+ Retry", "+ Loop\nsurgery"]
    shares = [29.0, 89.9, 98.2, 100.0]
    x_pos = list(range(len(steps)))
    ax.step(x_pos, shares, where="mid", color=st["accent"], linewidth=3.2)
    ax.fill_between(x_pos, shares, step="mid", color=st["accent"], alpha=0.14)
    ax.plot(x_pos, shares, "o", color=st["accent"], markersize=11)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(steps)
    ax.set_xlim(-0.45, 3.45)
    ax.set_ylim(0, 118)
    ax.set_xlabel("Validation stage")
    ax.set_ylabel("Schema-valid (%)")
    ax.set_title("Schema-Valid Share (765 Documents)", pad=14)
    ax.grid(axis="y", color=st["grid"], linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    offsets = [(0, 10), (0, -24), (-14, -24), (0, 10)]
    labels_txt = ["29.0%", "89.9%", "98.2%", "100%"]
    for idx, (val, txt, (dx, dy)) in enumerate(zip(shares, labels_txt, offsets)):
        ax.annotate(txt, (idx, val), xytext=(dx, dy), textcoords="offset points",
                    ha="center", va="center", fontweight="bold", fontsize=18, color=st["fg"])
    fig.tight_layout(pad=1.0)
    p = CHARTS_DIR / "slide10_step.png"
    fig.savefig(p, dpi=220, facecolor=st["bg"])
    plt.close(fig)
    paths["slide10"] = p

    # 6. Slide 11: Line chart of wrong-explanation rate by version (EN & VI)
    st = _setup_chart_style(dark=True)
    fig, ax = plt.subplots(figsize=(9.6, 6.8), dpi=220)
    fig.patch.set_facecolor(st["bg"])
    _style_ax(ax, st)
    vers = ["base", "v1", "v2", "v3", "v3 +\nguard"]
    en = [33.6, 36.1, 29.5, 13.1, 4.9]
    vi = [48.4, 24.6, 27.0, 26.2, 8.2]
    x_pos = list(range(len(vers)))
    ax.plot(x_pos, en, "-o", color=st["accent2"], linewidth=3.0, markersize=10, label="English")
    ax.plot(x_pos, vi, "-s", color=st["amber"], linewidth=3.0, markersize=10, label="Vietnamese")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(vers)
    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(0, 58)
    ax.set_xlabel("Model version")
    ax.set_ylabel("Wrong rate (%)")
    ax.set_title("Wrong Explanation Rate by Version (%)", pad=14)
    ax.grid(axis="y", color=st["grid"], linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="upper right", frameon=True, facecolor="#231F20", edgecolor=st["muted"])
    for txt in leg.get_texts():
        txt.set_color("#FFFFFF")

    en_dy = [-22, 14, 14, -22, -20]
    vi_dy = [14, -22, -22, 14, 22]
    for idx, (e, v, edy, vdy) in enumerate(zip(en, vi, en_dy, vi_dy)):
        ax.annotate(f"{e:.1f}%", (idx, e), xytext=(0, edy), textcoords="offset points",
                    ha="center", va="center", fontweight="bold", fontsize=18, color=st["accent2"])
        ax.annotate(f"{v:.1f}%", (idx, v), xytext=(0, vdy), textcoords="offset points",
                    ha="center", va="center", fontweight="bold", fontsize=18, color=st["amber"])

    fig.tight_layout(pad=1.0)
    p = CHARTS_DIR / "slide11_wrong_rate.png"
    fig.savefig(p, dpi=220, facecolor=st["bg"])
    plt.close(fig)
    paths["slide11"] = p

    return paths


# ---------------------------------------------------------------------------
# PPTX helpers (reusing Review 2 template mechanics)
# ---------------------------------------------------------------------------

prs = Presentation(TEMPLATE)
T = list(prs.slides)
BLANK = prs.slide_layouts[6]


def style(run, size, bold=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def _is_logo_icon(sh) -> bool:
    """Drop only the template's top-left Salford square mark icon (not text boxes)."""
    return sh.shape_type == 5 and Inches(0.5) < sh.top < Inches(1.1) and sh.left < Inches(1.6)


def _iter_text_frames(shape):
    if shape.has_text_frame:
        yield shape.text_frame
    if shape.has_table:
        for row in shape.table.rows:
            for cell in row.cells:
                yield cell.text_frame
    if shape.shape_type == 6:  # GROUP
        for sub in shape.shapes:
            yield from _iter_text_frames(sub)


def chrome_text(slide, page: int | None, dark: bool = False) -> None:
    """Rewrite template header/footer text and lift every run to >= 18 pt."""
    hdr_color = MUTED_DARK if dark else MUTED_LIGHT
    for sh in slide.shapes:
        if not sh.has_text_frame:
            continue
        raw = sh.text_frame.text.strip()
        new = None
        if raw == "Salford & Co.":
            new = "ATS CV-JD Matching"
            sh.left = Inches(1.03)
            sh.width = Inches(8.0)
        elif raw == "www.reallygreatsite.com":
            new = "Review 3"
            sh.width = Inches(5.0)
        elif raw == "Page":
            sh.left = Inches(17.25)
            sh.width = Inches(1.05)
        elif raw.isdigit() and len(raw) == 2 and page is not None:
            new = f"{page:02d}"
            sh.left = Inches(18.28)
            sh.width = Inches(0.68)

        for p in sh.text_frame.paragraphs:
            for i, r in enumerate(p.runs):
                if new is not None:
                    r.text = new if i == 0 else ""
                r.font.name = FONT
                if r.font.size is not None and r.font.size < Pt(MIN_PT):
                    r.font.size = Pt(MIN_PT)
                if raw == "Salford & Co.":
                    r.font.color.rgb = hdr_color


def _copy_rels_and_shapes(base, target, keep_predicate):
    copied = []
    if base._element.cSld.bg is not None:
        copied.append(copy.deepcopy(base._element.cSld.bg))
        target._element.cSld.insert(0, copied[-1])
    for sh in base.shapes:
        if keep_predicate(sh):
            copied.append(copy.deepcopy(sh._element))
            target.shapes._spTree.append(copied[-1])
    for el in (e for c in copied for e in c.iter()):
        for attr in ("embed", "link"):
            rid = el.get(f"{{{RNS}}}{attr}")
            if rid:
                rel = base.part.rels[rid]
                new_rid = target.part.relate_to(
                    rel.target_ref if rel.is_external else rel.target_part,
                    rel.reltype,
                    rel.is_external,
                )
                el.set(f"{{{RNS}}}{attr}", new_rid)


def content_slide(dark: bool, title: str, page: int, title_size: int = 40):
    """Create a content slide reusing template T[3] (dark) or T[2] (white) chrome."""
    base = T[3] if dark else T[2]
    s = prs.slides.add_slide(BLANK)
    for sh in list(s.shapes):
        sh._element.getparent().remove(sh._element)
    _copy_rels_and_shapes(
        base,
        s,
        lambda sh: (sh.top < Inches(1.3) or sh.top > Inches(10)) and not _is_logo_icon(sh),
    )
    chrome_text(s, page, dark=dark)
    s.dark = dark
    text(s, title, 1.02, 1.22, 17.9, 0.95, title_size, bold=True)
    return s


def fg(s):
    return WHITE if s.dark else DARK


def text(s, value, x, y, w, h, size=20, bold=False, color=None, align=None, anchor=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    if anchor:
        tf.vertical_anchor = anchor
    lines = value if isinstance(value, list) else [value]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if align:
            p.alignment = align
        r = p.add_run()
        style(r, size, bold, color or fg(s))
        r.text = line
    return tb


def bullets(s, items, x, y, w, h, size=19, gap=10, color=None):
    """Items can be strings or list of (text, bold) spans per bullet."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    c = color or fg(s)
    lead_color = CYAN if s.dark else BLUE
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        r0 = p.add_run()
        has_colon = isinstance(item, str) and ":" in item[:60]
        style(r0, size, True, lead_color if (isinstance(item, list) or has_colon) else c)
        if isinstance(item, list):
            r0.text = "•  "
            for span_txt, span_bold in item:
                rs = p.add_run()
                style(rs, size, span_bold, c)
                rs.text = span_txt
        elif has_colon:
            head, rest = item.split(":", 1)
            r0.text = f"•  {head}:"
            r1 = p.add_run()
            style(r1, size, False, c)
            r1.text = rest
        else:
            style(r0, size, False, c)
            r0.text = f"•  {item}"
    return tb


def pending_bar(s, msg: str, x=1.02, y=8.95, w=17.9, h=0.80, size=18):
    """Render a PENDING bullet as a single muted line starting with 'Pending:'."""
    clean = re.sub(r"^PENDING:\s*", "", msg.strip(), flags=re.IGNORECASE)
    col = MUTED_DARK if s.dark else MUTED_LIGHT
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    r0 = p.add_run()
    style(r0, size, True, col)
    r0.text = "Pending: "
    r1 = p.add_run()
    style(r1, size, False, col)
    r1.text = clean
    return tb


def box(
    s,
    label,
    x,
    y,
    w,
    h,
    fill,
    size=19,
    sub_size=None,
    bold=False,
    color=WHITE,
    shape=MSO_SHAPE.ROUNDED_RECTANGLE,
    border=None,
    dashed=False,
    align=PP_ALIGN.CENTER,
    anchor=MSO_ANCHOR.MIDDLE,
):
    b = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        b.fill.background()
    else:
        b.fill.solid()
        b.fill.fore_color.rgb = fill
    if border is not None:
        b.line.color.rgb = border
        b.line.width = Pt(1.75)
        if dashed:
            b.line.dash_style = MSO_LINE.DASH
    else:
        b.line.fill.background()
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        b.adjustments[0] = 0.08
    tf = b.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.14)
    tf.margin_top = tf.margin_bottom = Inches(0.12 if anchor == MSO_ANCHOR.TOP else 0.06)
    lines = label if isinstance(label, list) else [label]
    if sub_size is None:
        sub_size = max(MIN_PT, 19 if size >= 26 else size - 1)
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i == 0 and len(lines) > 1:
            p.space_after = Pt(6)
        r = p.add_run()
        is_b = bold if i == 0 or not isinstance(label, list) else False
        if isinstance(label, list) and i == 0:
            is_b = True
        style(r, size if i == 0 or not isinstance(label, list) else sub_size, is_b, color)
        r.text = line
    return b


def arrow(s, x, y, w=0.45, h=0.38, direction="right", color=BLUE):
    shape_map = {
        "right": MSO_SHAPE.RIGHT_ARROW,
        "left": MSO_SHAPE.LEFT_ARROW,
        "down": MSO_SHAPE.DOWN_ARROW,
        "up": MSO_SHAPE.UP_ARROW,
    }
    a = s.shapes.add_shape(shape_map[direction], Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    return a


def picture(s, img_path: Path, x, y, w, h):
    """Fit image inside (x, y, w, h), keeping aspect ratio, centered."""
    iw, ih = Image.open(img_path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    return s.shapes.add_picture(
        str(img_path),
        Inches(x + (w - pw) / 2),
        Inches(y + (h - ph) / 2),
        Inches(pw),
        Inches(ph),
    )


def table(s, rows, x, y, w, h, widths=None, bold_cells=None, size=18, align_cols=None):
    """Real python-pptx table. bold_cells: set of (row_idx, col_idx) to bold."""
    bold_cells = bold_cells or set()
    t = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    for j, cw in enumerate(widths or [w / len(rows[0])] * len(rows[0])):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            c = t.cell(i, j)
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb, col = BLUE, WHITE
            elif s.dark:
                c.fill.fore_color.rgb, col = (CARD_DARK if i % 2 else PANEL_DARK), WHITE
            else:
                c.fill.fore_color.rgb, col = (LIGHT if i % 2 else WHITE), DARK
            c.margin_left = c.margin_right = Inches(0.10)
            c.margin_top = c.margin_bottom = Inches(0.05)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.text_frame.word_wrap = True
            p = c.text_frame.paragraphs[0]
            if align_cols and j in align_cols:
                p.alignment = align_cols[j]
            else:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            is_bold = (i == 0) or ((i, j) in bold_cells)
            r = p.add_run()
            style(r, size, is_bold, col)
            r.text = value
    return t


def card(s, x, y, w, h, fill, top, head, body, color, top_size=46, head_size=24, body_size=19, border=None):
    box(s, "", x, y, w, h, fill, border=border)
    text(s, top, x + 0.30, y + 0.25, w - 0.60, 0.85, top_size, bold=True, color=color)
    text(s, head, x + 0.30, y + 1.15, w - 0.60, 0.65, head_size, bold=True, color=color)
    tb = text(s, body, x + 0.30, y + 1.90, w - 0.60, h - 2.10, body_size, color=color)
    for para in tb.text_frame.paragraphs:
        para.space_after = Pt(8)


def lane(s, label, x, y, w, h, right=False, dashed=False):
    c = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    c.adjustments[0] = 0.05
    c.fill.background()
    c.line.color.rgb = BORDER_DARK if s.dark else BORDER_LIGHT
    c.line.width = Pt(1.75)
    if dashed:
        c.line.dash_style = MSO_LINE.DASH
    lw = max(3.0, w - 0.56)
    text(
        s,
        label,
        x + w - lw - 0.28 if right else x + 0.28,
        y + 0.10,
        lw,
        0.45,
        20,
        bold=True,
        color=CYAN if s.dark else BLUE,
        align=PP_ALIGN.RIGHT if right else PP_ALIGN.LEFT,
    )
    return c


# ---------------------------------------------------------------------------
# Build all 22 slides
# ---------------------------------------------------------------------------

def build_deck(charts: dict[str, Path]) -> list:
    slides = []

    # 1. Cover (edit template slide 1 T[0] in place)
    cover = T[0]
    for sh in list(cover.shapes):
        if sh.shape_id == 9 or _is_logo_icon(sh):
            sh._element.getparent().remove(sh._element)
    cover_repl = {
        "Pitch Deck": "AI Agent for CV Matching",
        "Welcome to Wardiere Inc.": "Review 3",
        "Adeline Palmerston": "[Students]  |  Supervisor: [Supervisor]",
        "22 December 2035": "[Date]",
        "Website": "Project",
        "www.reallygreatsite.com": "ATS CV-JD Matching",
        "E-mail": "Institution",
        "hello@reallygreatsite.com": "[Council / Faculty / University]",
        "Salford & Co.": "ATS CV-JD Matching",
    }
    for sh in cover.shapes:
        if not sh.has_text_frame:
            continue
        raw = sh.text_frame.text.strip()
        if raw in cover_repl:
            runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
            runs[0].text = cover_repl[raw]
            for r in runs[1:]:
                r.text = ""
            if raw == "Pitch Deck":
                runs[0].font.name = FONT
                runs[0].font.size = Pt(92)
                sh.text_frame.vertical_anchor = MSO_ANCHOR.TOP
                for p in sh.text_frame.paragraphs:
                    p.line_spacing = 1.05
                    p.space_before = Pt(0)
                    p.space_after = Pt(0)
                sh.top = Inches(3.90)
                sh.height = Inches(1.50)
                sh.width = Inches(18.0)
            elif raw == "Welcome to Wardiere Inc.":
                sh.top = Inches(3.05)
                sh.height = Inches(0.65)
            elif raw == "Salford & Co.":
                sh.left = Inches(1.04)
                sh.width = Inches(8.0)
                runs[0].font.color.rgb = MUTED_LIGHT
            elif raw == "Adeline Palmerston":
                sh.width = Inches(15.5)
            elif raw == "hello@reallygreatsite.com":
                sh.width = Inches(5.8)
    chrome_text(cover, None, dark=False)
    cover.dark = False
    text(cover, "A local, explainable CV–job matching agent", 1.04, 5.65, 17.0, 0.70, 32, bold=True, color=BLUE)
    for sh in cover.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip() == "Presented by":
            sh.top = Inches(7.35)
    slides.append(cover)

    # 2. Problem and pain points (White bg, template slide 3 three-numbered-panel design)
    s = content_slide(False, "Problem and pain points", 2, title_size=42)
    panels = [
        ("1.", "Keyword ATS",
         ["Brittle on synonyms", "Broken by multi-column layouts (~100 of 441 CV texts had PDF artifacts)", "One opaque score"],
         BLUE, WHITE, None),
        ("2.", "Commercial LLM APIs",
         ["Candidate personal data leaves the organisation", "Cost grows linearly with document volume"],
         DARK, WHITE, None),
        ("3.", "Open 7B out of the box",
         ["Not enough without tuning and guardrails", "CV skill F1 0.49, JD skill F1 0.33", "0% schema-valid JSON output"],
         LIGHT, DARK, BORDER_LIGHT),
    ]
    for k, (num_lbl, head, body_lines, fill, col, brd) in enumerate(panels):
        card(s, 1.02 + k * 6.06, 2.55, 5.83, 5.15, fill, num_lbl, head,
             [f"•  {b}" for b in body_lines], col, top_size=54, head_size=26, body_size=19, border=brd)
    # Starting point strip
    box(s, "", 1.02, 8.05, 17.9, 1.55, LIGHT, border=BORDER_LIGHT)
    text(s, "Starting point (raw Qwen2.5-7B):", 1.35, 8.20, 6.2, 1.25, 22, bold=True, color=DARK, anchor=MSO_ANCHOR.MIDDLE)
    for idx, (val, lbl) in enumerate([("0.49", "CV skill F1"), ("0.33", "JD skill F1"), ("0%", "schema-valid output")]):
        bx = 7.7 + idx * 3.75
        text(s, val, bx, 8.15, 3.4, 0.72, 36, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
        text(s, lbl, bx, 8.90, 3.4, 0.55, 18, color=DARK, align=PP_ALIGN.CENTER)
    slides.append(s)

    # 3. Objectives and key contributions (Dark bg, real python-pptx table + C1-C4 cards)
    s = content_slide(True, "Objectives and key contributions", 3, title_size=42)
    text(s, "Five measurable objectives (narrowed from eight after Review 1):",
         1.02, 2.28, 17.9, 0.45, 20, bold=True, color=CYAN)
    t_rows = [
        ["Objective", "Scope", "Measured result", "Status"],
        ["O1", "Extraction (C1)", "CV skill F1 0.49 → 0.727; JD 0.33 → 0.730", "Met"],
        ["O2", "Retrieval", "Hybrid + reranker MRR@10 0.841 (88 CV queries)", "Met"],
        ["O3", "Matching (C3)", "Final ranker NDCG@10 0.630, AUC 0.953 (fit); 0.511 (hire)", "Fit only"],
        ["O4", "Check-and-retry (C2)", "Schema-valid output 29.0% → 100% on 765 docs", "Met"],
        ["O5", "Faithful explanations (C4)", "Wrong rate 4.9% (English), 8.2% (Vietnamese) + guard", "Met (guard)"],
    ]
    bolds = {(1, 3), (2, 3), (4, 3), (5, 3)}
    table(s, t_rows, 1.02, 2.82, 17.9, 3.55, widths=[1.55, 3.65, 10.15, 2.55], bold_cells=bolds, size=18,
          align_cols={0: PP_ALIGN.CENTER, 1: PP_ALIGN.LEFT, 2: PP_ALIGN.LEFT, 3: PP_ALIGN.CENTER})
    contribs = [
        ("C1 · QLoRA Extractor", "Distilled from LLM teachers: CV skill F1 0.49 → 0.727, JD 0.33 → 0.730.", BLUE),
        ("C2 · Repair-Retry Agent", "Schema-valid 29.0% → 100% on 765 docs, no retraining.", CARD_DARK),
        ("C3 · Ranker Benchmark", "Separates accuracy (embeddings) from explainability (rules): NDCG@10 0.630, AUC 0.953.", CARD_DARK),
        ("C4 · Faithfulness Guard", "Replaces contradicting outputs: wrong 4.9% (English), 8.2% (Vietnamese).", CARD_DARK),
    ]
    for k, (ch, cb, fill) in enumerate(contribs):
        box(s, [ch, cb], 1.02 + k * 4.55, 6.68, 4.25, 2.95, fill, size=20, sub_size=18,
            bold=True, color=WHITE, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    slides.append(s)

    # 4. End-to-end pipeline (White bg, python-pptx shapes + arrows)
    s = content_slide(False, "End-to-end pipeline", 4, title_size=42)
    lane(s, "Online LangGraph flow (service_graph) — Input → Model → Output", 1.02, 2.38, 17.9, 5.25)

    r1 = [
        ("1. CV file", "PDF / DOCX / text", DARK, WHITE),
        ("2. Clean", "Text cleaning", DARK, WHITE),
        ("3. Extract", "qwen-v4 + validate-repair-retry (hash cache)", BLUE, WHITE),
        ("4. Retrieve", "BGE-M3 + BM25 → RRF → pool of 20 JDs", DARK, WHITE),
    ]
    r2 = [
        ("5. Score", "Seven rule scores", DARK, WHITE),
        ("6. Rank", "Final ranker model", DARK, WHITE),
        ("7. Explain", "Adapter + faithfulness checker", BLUE, WHITE),
        ("8. Web page", "Ranked jobs output", LIGHT, DARK),
    ]
    for c, (hd, sub, fill, col) in enumerate(r1):
        x = 1.45 + c * 4.45
        box(s, [hd, sub], x, 3.05, 3.70, 1.55, fill, size=19, sub_size=18, color=col,
            border=BORDER_LIGHT if fill == LIGHT else None)
        if c < 3:
            arrow(s, x + 3.82, 3.63, 0.50, 0.38, "right")

    # Stepped connector from box 4 (x=14.80..18.50) down and across to box 5 (x=1.45..5.15)
    for bx, by, bw, bh in [
        (16.62, 4.60, 0.08, 0.34),
        (3.26, 4.90, 13.44, 0.08),
    ]:
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(bx), Inches(by), Inches(bw), Inches(bh))
        bar.fill.solid()
        bar.fill.fore_color.rgb = BLUE
        bar.line.fill.background()
    arrow(s, 3.08, 4.98, 0.44, 0.36, "down")

    for c, (hd, sub, fill, col) in enumerate(r2):
        x = 1.45 + c * 4.45
        box(s, [hd, sub], x, 5.38, 3.70, 1.55, fill, size=19, sub_size=18, color=col,
            border=BLUE if fill == LIGHT else None)
        if c < 3:
            arrow(s, x + 3.82, 5.96, 0.50, 0.38, "right")

    # Offline JD note box + bottom bullets
    box(s, ["Offline corpus", "361 JDs extracted once"], 1.02, 7.90, 4.50, 1.65, LIGHT, size=19, sub_size=18, color=DARK, border=BORDER_LIGHT)
    bullets(s, [
        "One LangGraph flow (service_graph); the 361 JDs are extracted once, offline.",
        "No commercial LLM API in the production path; CV extraction is cached by text hash.",
    ], 5.85, 7.98, 13.05, 1.60, size=20, gap=12)
    slides.append(s)

    # 5. Core improvement over the baseline: labels and validation (Dark bg, two diagrams)
    s = content_slide(True, "Core improvement over the baseline: labels and validation", 5, title_size=36)

    # Top lane: distillation & V4 filter
    lane(s, "Distillation & Gold-style Filter (V4)", 1.02, 2.30, 17.9, 2.15)
    d1 = [
        ("2 LLM teachers", "independent", CARD_DARK),
        ("Rule checker", "schema, quotes", CARD_DARK),
        ("Arbiter", "merges splits", CARD_DARK),
        ("5-rule filter (V4)", "gold-style", BLUE),
        ("Training set", "1,058 CV + 694 JD", CARD_DARK),
    ]
    for c, (hd, sub, fill) in enumerate(d1):
        x = 1.32 + c * 3.55
        box(s, [hd, sub], x, 2.95, 3.00, 1.28, fill, size=18, sub_size=18, color=WHITE)
        if c < 4:
            arrow(s, x + 3.07, 3.40, 0.40, 0.34, "right")

    # Middle lane: validate-repair-retry loop
    lane(s, "Validate–Repair–Retry Loop (Inference)", 1.02, 4.62, 17.9, 2.68)
    d2 = [
        ("Generate", "Qwen2.5-7B + V4", CARD_DARK),
        ("Repair + Loop surgery", "cuts repeated items", BLUE),
        ("Validate", "schema & quotes", CARD_DARK),
        ("Output", "valid JSON", BLUE),
    ]
    for c, (hd, sub, fill) in enumerate(d2):
        x = 1.45 + c * 4.45
        box(s, [hd, sub], x, 5.24, 3.70, 1.12, fill, size=18, sub_size=18, color=WHITE)
        if c < 3:
            arrow(s, x + 3.82, 5.62, 0.50, 0.36, "right")
    # Feedback loop bar under Generate <-> Validate
    for bx, by, bw, bh in [(3.26, 6.36, 0.08, 0.28), (3.26, 6.58, 8.95, 0.06), (12.15, 6.36, 0.08, 0.28)]:
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(bx), Inches(by), Inches(bw), Inches(bh))
        bar.fill.solid()
        bar.fill.fore_color.rgb = CYAN
        bar.line.fill.background()
    text(s, "Retry on fail: repetition penalty 1.05 → 1.1 → 1.15 → 1.2",
         3.60, 6.72, 8.30, 0.45, 18, bold=True, color=CYAN, align=PP_ALIGN.CENTER)

    bullets(s, [
        "Gold-style filter (V4): 40% of teacher skills were absent from gold; five rules raised blind-half F1 0.571 → 0.662 (+0.090, CI [+0.023, +0.165]).",
        "Training: QLoRA rank 16 on Qwen2.5-7B-Instruct (4-bit); 71 minutes on a rented A100; adapter 154 MB.",
    ], 1.02, 7.52, 17.9, 2.25, size=19, gap=12)
    slides.append(s)

    # 6. Scoring, ranking and explanation design (White bg, diagram + 7-weight chart)
    s = content_slide(False, "Scoring, ranking and explanation design", 6, title_size=40)
    lane(s, "Feature → Ranker → Verdict", 1.02, 2.42, 8.85, 2.55)
    f_steps = [
        ("8 Features", "5 rules + Qwen3 + title + TF-IDF", DARK),
        ("Ranker", "Logistic reg. (9 params)", BLUE),
        ("Verdict", "Score + guarded text", DARK),
    ]
    for c, (hd, sub, fill) in enumerate(f_steps):
        x = 1.25 + c * 2.90
        box(s, [hd, sub], x, 3.05, 2.52, 1.62, fill, size=18, color=WHITE)
        if c < 2:
            arrow(s, x + 2.56, 3.68, 0.30, 0.30, "right")

    bullets(s, [
        "Seven rule scores: if evidence is absent → \"not scored\", never a guess.",
        "Final ranker: logistic regression, eight features, nine parameters (five rule scores + Qwen3-Embedding cosine + title cosine + TF-IDF cosine).",
        "Explanation: second QLoRA adapter writes English or Vietnamese from scored facts; checker replaces contradictions with a template.",
        "Retrieval store: BGE-M3 vectors + BM25, in-memory index for 361 JDs (pgvector available for larger corpus).",
    ], 1.02, 5.20, 8.85, 4.55, size=18, gap=10)
    picture(s, charts["slide06"], 10.15, 2.42, 8.75, 7.20)
    slides.append(s)

    # 7. Datasets, splits and leakage control (Dark bg, table + funnel & class-share chart)
    s = content_slide(True, "Datasets, splits and leakage control", 7, title_size=40)
    d_rows = [
        ["Dataset", "Count / Composition", "Split / Notes"],
        ["Public CV PDFs", "601 found online (577 distinct)", "Gold test 36 (6 / family); train gold x4"],
        ["JDs (CareerViet)", "802 distinct (693 vi, 74 mixed, 35 en)", "Test 54 JDs; stratified"],
        ["Fit-labelled pairs", "80,017 → 59,337 (485 CVs x 280 JDs)", "25.8% leakage dropped; split by CV hash"],
    ]
    table(s, d_rows, 1.02, 2.50, 9.85, 3.60, widths=[2.55, 3.95, 3.35], size=18)
    bullets(s, [
        "Class imbalance: No Fit 77.6%, Potential 15.9%, Good 6.6% → accuracy not used (always \"No Fit\" = 0.775).",
        "Ranking metrics used instead: AUC, macro-F1 and NDCG.",
        "Leakage control: removed 25.8% of 80,017 pairs sharing CVs with extractor training; dev/test split by CV hash.",
        "Extraction balance: gold CVs upsampled x4 in train; test stratified by job family.",
    ], 1.02, 6.35, 9.85, 3.40, size=18, gap=10)
    picture(s, charts["slide07"], 11.12, 2.45, 7.80, 7.20)
    slides.append(s)

    # 8. Extraction benchmark (O1): baselines vs proposed (White bg, table + bar chart + loss_curves.png)
    s = content_slide(False, "Extraction benchmark (O1): baselines vs proposed", 8, title_size=38)
    e_rows = [
        ["Model", "CV skill F1", "JD skill F1", "CV schema-valid"],
        ["Baseline 1: Raw Qwen", "0.49 [0.41, 0.57]", "0.33 [0.24, 0.42]", "0%"],
        ["Baseline 2: V2 (all teacher)", "0.55", "0.70", "31%"],
        ["Proposed V1: V3b", "0.65 [0.58, 0.71]", "0.73", "42%"],
        ["Proposed V2: V4 + repair/retry", "0.73 [0.68, 0.77]", "0.73 [0.69, 0.77]", "100%"],
    ]
    table(s, e_rows, 1.02, 2.42, 10.35, 3.25, widths=[3.70, 2.45, 2.45, 1.75],
          bold_cells={(4, 1), (4, 2), (4, 3)}, size=18)
    text(s, "Three seeds (CV skill F1): 0.727 / 0.703 / 0.708 = 0.713 ± 0.013 (headline is best seed).",
         1.02, 5.90, 10.35, 0.65, 18, bold=True, color=DARK)
    picture(s, LOSS_CURVES, 1.02, 6.56, 10.35, 3.16)
    picture(s, charts["slide08"], 11.62, 2.42, 7.30, 7.20)
    slides.append(s)

    # 9. Matching benchmark (O2, O3): baselines vs proposed (Dark bg, table + NDCG@10 chart)
    s = content_slide(True, "Matching benchmark (O2, O3): baselines vs proposed", 9, title_size=38)
    m_rows = [
        ["Method (31,150 pairs, 253 CVs)", "NDCG@10", "Notes"],
        ["Rule scorer V3.1", "0.415", "Deterministic rules only"],
        ["BM25", "0.436", "Lexical baseline"],
        ["TF-IDF", "0.468", "Strongest lexical baseline"],
        ["Proposed V1", "0.564", "Rules + BGE-M3 + TF-IDF"],
        ["Proposed V2 (Final)", "0.630 [0.600, 0.661]", "AUC 0.953 (rules + Qwen3 + TF-IDF)"],
    ]
    table(s, m_rows, 1.02, 2.45, 9.85, 3.85, widths=[3.45, 2.95, 3.45],
          bold_cells={(5, 1), (5, 2)}, size=18)
    bullets(s, [
        "Final vs TF-IDF: +0.162 NDCG@10 (CI [+0.135, +0.190]); vs Qwen3 cosine alone: +0.039 (p < 0.001).",
        "Retrieval (88 CV queries, 361 JDs, diagnostic): MRR@10 BM25 0.746, dense 0.810, hybrid + reranker 0.841.",
        "Human labels (19 CVs, 206 pairs): final model NDCG@5 0.892 vs TF-IDF 0.817 (p = 0.043).",
    ], 1.02, 6.62, 9.85, 3.10, size=18, gap=11)
    picture(s, charts["slide09"], 11.12, 2.45, 7.80, 7.20)
    slides.append(s)

    # 10. Ablation study (White bg, table + step chart)
    s = content_slide(False, "Ablation study", 10, title_size=42)
    a_rows = [
        ["Component / Change", "Measured effect", "Outcome"],
        ["Remove 5 rule scores in ranker", "0.626 vs 0.630 (+0.004, p = 0.57)", "Keep (rules explain)"],
        ["Cross-encoder reranker in hybrid", "+0.001 (p = 0.78)", "Rejected"],
        ["Validation loop (765 docs)", "29.0% → 89.9% → 98.2% → 100%", "Keep (+100% valid)"],
        ["Gold-style filter (V3b → V4)", "Precision +0.134 (CI [+0.082, +0.189])", "Keep (19.0 vs 18.0)"],
    ]
    table(s, a_rows, 1.02, 2.48, 9.85, 4.10, widths=[3.45, 4.05, 2.35],
          bold_cells={(3, 1), (3, 2), (4, 1)}, size=18)
    bullets(s, [
        "Embeddings rank; rules explain: removing the five rule scores barely moves NDCG@10 (0.626 vs 0.630, p = 0.57).",
        "Validation loop stages on 765 docs: raw 29.0% → + repair 89.9% → + retry 98.2% → + loop surgery 100%.",
        "Gold-style filter (V3b → V4): precision +0.134; predicted skills per CV drop 25.1 → 19.0 (gold 18.0).",
    ], 1.02, 6.82, 9.85, 2.95, size=18, gap=10)
    picture(s, charts["slide10"], 11.12, 2.45, 7.80, 7.20)
    slides.append(s)

    # 11. Explanations (O5): three versions and the guard (Dark bg, line chart + bullets)
    s = content_slide(True, "Explanations (O5): three versions and the guard", 11, title_size=40)
    picture(s, charts["slide11"], 1.02, 2.45, 9.65, 7.15)
    # Right side KPI cards + bullets
    box(s, ["4.9%", "English wrong rate (v3 + guard)"], 10.95, 2.48, 3.85, 2.05, BLUE, size=40, sub_size=19, bold=True, color=WHITE)
    box(s, ["8.2%", "Vietnamese wrong rate (v3 + guard)"], 15.05, 2.48, 3.85, 2.05, CARD_DARK, size=40, sub_size=19, bold=True, color=WHITE, border=CYAN)
    bullets(s, [
        "Strict LLM review (base → v3 + guard): English 33.6% → 36.1% → 29.5% → 13.1% → 4.9%; Vietnamese 48.4% → 24.6% → 27.0% → 26.2% → 8.2%.",
        "Guard: 49 of 244 outputs replaced by a template (21 English, 28 Vietnamese).",
        "Template audit: 48 templates reviewed, 0 wrong.",
        "Caveats: reviewer is an LLM; rule types chosen from this test set's error categories; semantic review on one of three seeds.",
    ], 10.95, 4.82, 7.95, 4.85, size=18, gap=12)
    slides.append(s)

    # 12. Failure case analysis (White bg, structured failure rows — no invented text)
    s = content_slide(False, "Failure case analysis", 12, title_size=42)
    f_rows = [
        ["Area / Input", "Observed failure", "Root cause & fix"],
        ["Select / reject labels\n(964 pairs)", "Every method at chance: final AUC 0.511 [0.474, 0.546]", "Scores measure CV–JD fit, not hiring decisions"],
        ["Repetition loops\n(485 CVs, 361 JDs)", "14 of 485 CVs looped; 2 CVs > 40,000 chars & 31 very short JDs still fail", "Only true model failure; 14 loops fixed by loop surgery"],
        ["Explanations\n(v2 input format)", "v2 new input format did not help (29.5% EN / 27.0% VI wrong)", "Cause in targets: 41% with unscored education named degree → fixed in v3 + guard"],
        ["Vietnamese\nexplanations", "Wrong rate stays higher (26.2% v3, 8.2% + guard)", "Data imbalance: only 330 of 1,639 training targets are Vietnamese"],
    ]
    table(s, f_rows, 1.02, 2.55, 17.9, 6.85, widths=[3.65, 6.95, 7.30], size=19)
    slides.append(s)

    # 13. Model optimisation benchmark (Dark bg, real before/after table + Pending)
    s = content_slide(True, "Model optimisation benchmark", 13, title_size=42)
    opt_rows = [
        ["Configuration", "CV skill F1", "JD F1 delta", "Size / VRAM (32,768 ctx)", "Throughput"],
        ["16-bit (A100, vLLM)", "0.73", "baseline", "-", "760–1,050 tokens/s"],
        ["Q4_K_M GGUF (RTX 4070 SUPER, Ollama)", "0.71", "-0.008", "4.7 GB file; about 6.6 GB VRAM", "68 tokens/s (1 stream)"],
    ]
    table(s, opt_rows, 1.02, 2.55, 17.9, 3.25, widths=[4.65, 2.05, 2.05, 5.20, 3.95],
          bold_cells={(1, 1), (1, 4), (2, 3)}, size=18)
    bullets(s, [
        "Technique: adapter merged, 16-bit GGUF → Q4_K_M quantisation (4.7 GB file), served by Ollama; vLLM for batch evaluation.",
        "Accuracy delta: CV skill F1 0.73 → 0.71 (-0.018, CI [-0.059, +0.020], not significant); JD -0.008.",
        "Memory fit: Q4 (~6.6 GB VRAM at 32,768-token context) + embedder (3.1 GB) fit together on one 12 GB RTX 4070 SUPER.",
    ], 1.02, 6.08, 17.9, 2.65, size=19, gap=12)
    pending_bar(s, "PENDING: a before / after table on the SAME hardware (16-bit vs Q4: ms/token, VRAM, F1); the 16-bit model was only measured on the A100.")
    slides.append(s)

    # 14. System architecture and Docker (White bg, python-pptx two-container diagram + Pending)
    s = content_slide(False, "System architecture and Docker", 14, title_size=42)
    lane(s, "Docker Compose (2 services today)", 1.02, 2.45, 17.9, 4.70)

    box(s, ["Browser / Client", "HTTP + SSE"],
        1.45, 3.20, 2.85, 2.05, LIGHT, size=19, sub_size=18, color=DARK, border=BORDER_LIGHT)
    arrow(s, 4.42, 4.04, 0.55, 0.36, "right")
    box(s, ["api container", "FastAPI + LangGraph flow", "Health check · SSE streaming"],
        5.10, 3.20, 5.55, 2.05, DARK, size=20, sub_size=18, color=WHITE)
    arrow(s, 10.80, 4.04, 1.30, 0.36, "right")
    text(s, "REST", 10.80, 3.55, 1.30, 0.42, 18, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
    box(s, ["ollama container (GPU)", "Pinned image 0.40.14 · 1 slot", "Local Q4_K_M + embedder"],
        12.25, 3.20, 6.25, 2.05, BLUE, size=20, sub_size=18, color=WHITE)

    # Host volumes row under api & ollama
    text(s, "Mounted from host:\npersonal data stays\nout of the image",
         1.45, 5.75, 3.25, 1.15, 18, bold=True, color=DARK, align=PP_ALIGN.LEFT)
    for k, vol in enumerate(["Host volume: Corpus", "Host volume: Index", "Host volume: Cache"]):
        vx = 5.10 + k * 4.55
        box(s, vol, vx, 5.85, 4.25, 0.90, LIGHT, size=18, bold=True, color=DARK, border=BLUE)
        arrow(s, vx + 1.92, 5.33, 0.40, 0.45, "up")

    bullets(s, [
        "Two Docker Compose services today: ollama on the GPU (one slot, pinned image 0.40.14) and api (FastAPI + LangGraph flow, health check); REST between them.",
        "Corpus, index and cache are mounted from the host so personal data stays out of the image.",
    ], 1.02, 7.35, 17.9, 1.45, size=19, gap=10)
    pending_bar(s, "PENDING: separate frontend container, and a database container for history and feedback (checklist 2.2).")
    slides.append(s)

    # 15. User interface and guardrails (Dark bg, structured bullets + UI wireframe + Pending)
    s = content_slide(True, "User interface and guardrails", 15, title_size=42)
    bullets(s, [
        "Input & Output: upload PDF / DOCX / TXT (5 MB limit) or paste text → ranked job cards, verdict, seven-score radar chart, matched & missing skills.",
        "Citation: every skill shows the literal quote from the CV; a dimension without evidence shows \"could not be checked\".",
        "Guardrail: scores come from deterministic rules over quoted facts; explanation checker blocks text that contradicts them.",
        "Streaming: extractor output appears as generated with pipeline steps ticked off (SSE); a match that has started is saved even if the connection drops.",
        "History & feedback: every run saved (SQLite); thumbs up / down and user's own verdict per job.",
    ], 1.02, 2.48, 10.15, 6.25, size=18, gap=12)

    # Right side: Annotated UI & Guardrail layout wireframe
    lane(s, "Web UI Layout (Annotated)", 11.45, 2.45, 7.45, 6.32)
    box(s, ["1. Input", "PDF / DOCX / TXT (5 MB) or paste text"],
        11.75, 3.02, 6.85, 0.92, CARD_DARK, size=18, sub_size=18, color=WHITE)
    box(s, ["2. Live SSE Stream", "Step ticks + live stream; started match saved if dropped"],
        11.75, 4.08, 6.85, 1.10, CARD_DARK, size=18, sub_size=18, color=WHITE)
    box(s, ["3. Job Card + Guardrails", "Verdict · 7-score radar · literal quotes · \"could not be checked\""],
        11.75, 5.32, 6.85, 1.42, BLUE, size=18, sub_size=18, color=WHITE)
    box(s, ["4. History & Feedback (SQLite)", "Saved runs · thumbs up / down · user verdict"],
        11.75, 6.88, 6.85, 1.10, CARD_DARK, size=18, sub_size=18, color=WHITE, border=CYAN)
    text(s, "[Annotated page screenshot placeholder]", 11.75, 8.16, 6.85, 0.40, 18, color=MUTED_DARK, align=PP_ALIGN.CENTER)

    pending_bar(s, "PENDING: prompt-injection test set (checklist 1.6).")
    slides.append(s)

    # 16. MLOps and DevOps (White bg, built CI/CD + tracking + monitoring + pending run history / screenshot)
    s = content_slide(False, "MLOps and DevOps", 16, title_size=42)

    # Top full-width lane: Implemented CI/CD (GitHub Actions)
    lane(s, "Implemented: CI/CD Pipeline (GitHub Actions)", 1.02, 2.45, 17.90, 2.65)
    ci_flow = [
        (1.28, 2.55, "Push", "Every push / main", DARK),
        (4.31, 2.55, "Lint", "Code quality", DARK),
        (7.34, 2.95, "813 unit tests", "Offline test suite", BLUE),
        (10.77, 2.85, "Image build", "API image on main", DARK),
        (14.10, 4.54, "Push to GitHub Container Registry", "Published on main", BLUE),
    ]
    for idx, (cx, cw, hd, sub, fill) in enumerate(ci_flow):
        box(s, [hd, sub], cx, 3.04, cw, 1.22, fill, size=18, sub_size=18, color=WHITE)
        if idx < len(ci_flow) - 1:
            arrow(s, cx + cw + 0.08, 3.48, 0.32, 0.34, "right", color=BLUE)
    text(s, "Every push runs lint and 813 unit tests; push to main builds and pushes the API image.",
         1.28, 4.44, 11.40, 0.45, 18, color=DARK)
    box(s, "CI run history [Pending after first push]",
        12.88, 4.38, 5.76, 0.56, LIGHT, size=18, bold=True, color=MUTED_LIGHT, border=BORDER_LIGHT, dashed=True)

    # Bottom-left lane: Implemented Tracking & Reproducibility
    lane(s, "Implemented: Tracking & Reproducibility", 1.02, 5.26, 7.65, 3.50)
    box(s, ["Weights & Biases Tracking", "Config, data artifact, adapters & predictions per run"],
        1.28, 5.92, 7.13, 1.20, DARK, size=19, sub_size=18, color=WHITE)
    box(s, ["Reproducible Experiments", "Fixed test splits, 3 seeds, stored model artifacts"],
        1.28, 7.28, 7.13, 1.20, BLUE, size=19, sub_size=18, color=WHITE)

    # Bottom-right lane: Implemented Monitoring (Prometheus + Grafana) + pending dashboard screenshot
    lane(s, "Implemented: Monitoring (Prometheus + Grafana)", 8.88, 5.26, 10.04, 3.50)
    mon_flow = [
        (9.14, 2.76, "API /metrics", "FastAPI endpoint", DARK),
        (12.38, 2.76, "Prometheus", "Scrapes metrics", DARK),
        (15.62, 3.04, "Grafana dashboard", "Built & provisioned", BLUE),
    ]
    for idx, (mx, mw, hd, sub, fill) in enumerate(mon_flow):
        box(s, [hd, sub], mx, 5.92, mw, 1.10, fill, size=18, sub_size=18, color=WHITE)
        if idx < len(mon_flow) - 1:
            arrow(s, mx + mw + 0.08, 6.30, 0.32, 0.34, "right", color=BLUE)
    box(s, ["Running Dashboard Screenshot [Pending]",
            "Response time · RPS · error rate · match time · queue · CPU · RAM · VRAM per model · GPU use"],
        9.14, 7.18, 9.52, 1.34, LIGHT, size=18, sub_size=18, bold=True, color=MUTED_LIGHT, border=BORDER_LIGHT, dashed=True)

    pending_bar(s, "PENDING: CI run history (after the first push), a screenshot of the running dashboard (checklist 2.1, 2.3).")
    slides.append(s)

    # 17. System testing (Dark bg, test summary KPIs + Pending load test / UAT)
    s = content_slide(True, "System testing", 17, title_size=42)
    kpis = [
        ("813", "Unit tests pass (offline)", BLUE),
        ("about 1 s", "Cached CV latency", CARD_DARK),
        ("about 30 s", "New CV median (17–53 s, 7 CVs)", CARD_DARK),
        ("472 ms", "Retrieval median (p95: 882 ms)", CARD_DARK),
    ]
    for k, (val, lbl, fill) in enumerate(kpis):
        box(s, [val, lbl], 1.02 + k * 4.55, 2.48, 4.25, 2.10, fill, size=36, sub_size=19, bold=True, color=WHITE)

    # Left panel: Unit test coverage & scoring speed
    box(s, "", 1.02, 4.90, 9.85, 3.70, PANEL_DARK, border=BORDER_DARK)
    text(s, "Offline Unit Test & Latency Summary", 1.32, 5.08, 9.25, 0.50, 22, bold=True, color=CYAN)
    bullets(s, [
        "Unit tests: 813 pass offline (parsing, schema validation, repair rules, scorer, retry loop, faithfulness checker, API).",
        "Rule scoring: about 9 ms per CV–JD pair.",
        "New CV latency: 17–53 s (median about 30 s over 7 CVs); cached CV: about 1 s.",
    ], 1.32, 5.65, 9.25, 2.80, size=18, gap=12)

    # Right panel: Pending load test & UAT placeholders
    box(s, "", 11.20, 4.90, 7.70, 3.70, DARK, border=BORDER_DARK, dashed=True)
    text(s, "Not Yet Measured (Pending)", 11.50, 5.08, 7.10, 0.50, 22, bold=True, color=MUTED_DARK)
    box(s, ["Load-test chart [Pending]", "Concurrent users, RPS, avg latency, failure rate"],
        11.50, 5.75, 7.10, 1.25, PANEL_DARK, size=18, color=MUTED_DARK, border=BORDER_DARK, dashed=True)
    box(s, ["UAT score [Pending]", "SUS or CSAT score, number of users ([TEAM])"],
        11.50, 7.18, 7.10, 1.20, PANEL_DARK, size=18, color=MUTED_DARK, border=BORDER_DARK, dashed=True)

    pending_bar(s, "PENDING: load test (concurrent users, RPS, average response time, failure rate) and UAT score (SUS or CSAT, number of users) (checklist 3.2, 3.3).")
    slides.append(s)

    # 18. Demo (White bg, flow + [TEAM] live/video placeholders)
    s = content_slide(False, "Demo", 18, title_size=42)
    demo_steps = [
        ("1. Upload CV", "PDF / DOCX / TXT or paste", DARK),
        ("2. Stream Extract", "Live SSE + repair-retry", BLUE),
        ("3. Score & Rank", "7 rule scores + ranker", DARK),
        ("4. Inspect Results", "Jobs, quotes & explanation", BLUE),
    ]
    for c, (hd, sub, fill) in enumerate(demo_steps):
        x = 1.02 + c * 4.60
        box(s, [hd, sub], x, 2.65, 4.00, 1.85, fill, size=22, sub_size=19, color=WHITE)
        if c < 3:
            arrow(s, x + 4.08, 3.38, 0.44, 0.38, "right")

    box(s, ["[Live demo]",
            "Upload a CV → ranked jobs with scores, literal quotes and explanation.",
            "No real personal data on screen."],
        1.02, 5.05, 8.70, 4.30, BLUE, size=34, sub_size=21, bold=True, color=WHITE)
    box(s, ["[Demo video]",
            "Backup: recorded 3–5 minute video ready.",
            "[TEAM] Show live page or play backup video."],
        10.20, 5.05, 8.70, 4.30, LIGHT, size=34, sub_size=21, bold=True, color=DARK, border=BLUE, dashed=True)
    slides.append(s)

    # 19. Cost and deployment footprint (Dark bg, real python-pptx table + [TEAM])
    s = content_slide(True, "Cost and deployment footprint", 19, title_size=42)
    c_rows = [
        ["Workload", "Hardware", "Duration / Throughput", "VRAM", "Cost"],
        ["Single training run", "Rented A100", "51–74 minutes", "-", "US$1.5–2.2"],
        ["All recorded GPU runs", "Rented A100", "Recorded runs", "-", "about US$16"],
        ["Online serving (1 stream)", "1 x RTX 4070 SUPER", "68 tokens/s (Q4)", "12 GB card", "Local GPU"],
        ["Batch extraction (3 slots)", "1 x RTX 4070 SUPER (Ollama)", "about 134 tokens/s", "10 GB VRAM", "Local GPU"],
    ]
    table(s, c_rows, 1.02, 2.55, 17.9, 4.00, widths=[3.90, 4.25, 4.00, 2.65, 3.10],
          bold_cells={(1, 2), (1, 4), (4, 2), (4, 3)}, size=18)
    bullets(s, [
        "One RTX 4070 SUPER (12 GB) serves the whole pipeline locally with no commercial LLM API cost.",
        "Batch extraction runs three Ollama slots at about 134 tokens/s within 10 GB VRAM.",
        "[TEAM] Confirm total rented GPU spend (about US$16) on the billing page.",
    ], 1.02, 6.95, 17.9, 2.65, size=20, gap=14)
    slides.append(s)

    # 20. Summary and self-assessment (White bg, 5 objective tiles + summary bar)
    s = content_slide(False, "Summary and self-assessment", 20, title_size=42)
    tiles = [
        ("0.73", "O1 · Extraction", "Status: Met", "CV & JD skill F1 (3 seeds: 0.713 ± 0.013)", BLUE, WHITE, None),
        ("0.841", "O2 · Retrieval", "Status: Met", "MRR@10 (hybrid + reranker)", BLUE, WHITE, None),
        ("0.953", "O3 · Matching", "Status: Partial", "AUC 0.953 final (0.916 rules) on fit; not select/reject", LIGHT, DARK, BLUE),
        ("100%", "O4 · Retry agent", "Status: Met", "Schema-valid rate (up from 29.0%)", BLUE, WHITE, None),
        ("4.9% / 8.2%", "O5 · Explanations", "Status: Met (guard)", "Wrong rate EN / VI with guard", DARK, WHITE, None),
    ]
    for k, (big, head, status, desc, fill, col, brd) in enumerate(tiles):
        tx = 1.02 + k * 3.63
        box(s, "", tx, 2.55, 3.38, 5.25, fill, border=brd)
        text(s, big, tx + 0.22, 2.82, 2.94, 0.95, 36 if k < 4 else 30, bold=True, color=col)
        text(s, head, tx + 0.22, 3.95, 2.94, 0.55, 22, bold=True, color=col)
        text(s, status, tx + 0.22, 4.62, 2.94, 0.55, 19, bold=True, color=CYAN if fill == DARK else (BLUE if fill == LIGHT else WHITE))
        text(s, desc, tx + 0.22, 5.35, 2.94, 2.15, 18, color=col)

    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.02), Inches(8.18), Inches(0.14), Inches(1.38))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    text(s, "Four of five objectives met as stated.", 1.40, 8.15, 17.5, 0.65, 26, bold=True, color=DARK)
    text(s, "O3 is met for CV–JD fit labels (AUC 0.916 rules, 0.953 final), not for select/reject hiring labels.",
         1.40, 8.85, 17.5, 0.65, 20, color=DARK)
    slides.append(s)

    # 21. Limitations and future work (Dark bg, limitation->mitigation table + next-steps timeline)
    s = content_slide(True, "Limitations and future work", 21, title_size=42)
    lim_rows = [
        ["Limitation", "Impact", "Mitigation / Next step"],
        ["Small gold test sets (36 CVs, 54 JDs)", "CIs of about ±5 to ±12 points", "Evaluate on the 48 reserved test CVs"],
        ["Synthetic English fit labels", "Human labels few (218 pairs), non-overlapping", "Real hiring outcomes from an HR partner; more Vietnamese data"],
        ["LLM explanation review (1 seed)", "Adapter not yet in the live service", "Multi-seed review & integrate explanation adapter"],
        ["No bias audit yet", "Fairness risk if automated", "Decision support only (never auto-reject); run fairness audit"],
    ]
    table(s, lim_rows, 1.02, 2.48, 17.9, 4.25, widths=[4.90, 5.70, 7.30], size=18)

    # Next-steps timeline rail at bottom (4 milestones)
    text(s, "Next steps:", 1.02, 7.05, 3.0, 0.45, 20, bold=True, color=CYAN)
    rail = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3.25), Inches(7.85), Inches(13.5), Inches(0.06))
    rail.fill.solid()
    rail.fill.fore_color.rgb = LIGHT
    rail.line.fill.background()
    next_steps = [
        ("1. Reserved test", "48 reserved test CVs", BLUE, WHITE),
        ("2. Vietnamese data", "More Vietnamese targets", WHITE, DARK),
        ("3. Fairness audit", "Bias & subgroup check", BLUE, WHITE),
        ("4. HR outcomes", "Real hiring labels", WHITE, DARK),
    ]
    for k, (nh, nb, nfill, ncol) in enumerate(next_steps):
        cx = 3.25 + k * 4.50
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 0.55), Inches(7.33), Inches(1.10), Inches(1.10))
        dot.fill.solid()
        dot.fill.fore_color.rgb = nfill
        dot.line.fill.background()
        dtf = dot.text_frame
        dtf.vertical_anchor = MSO_ANCHOR.MIDDLE
        dtf.paragraphs[0].alignment = PP_ALIGN.CENTER
        r = dtf.paragraphs[0].add_run()
        style(r, 20, True, ncol)
        r.text = str(k + 1)
        text(s, nh, cx - 2.05, 8.52, 4.10, 0.45, 19, bold=True, align=PP_ALIGN.CENTER, color=WHITE)
        text(s, nb, cx - 2.05, 8.98, 4.10, 0.45, 18, align=PP_ALIGN.CENTER, color=MUTED_DARK)
    slides.append(s)

    # 22. Closing (edit template slide 12 T[11] in place)
    close = T[11]
    for sh in list(close.shapes):
        if sh.shape_id == 5 or _is_logo_icon(sh):
            sh._element.getparent().remove(sh._element)
    close_repl = {
        "Let’s Build the Future Together": "Thank you. Questions?",
        "22 December 2035": "[Date]",
        "Website": "Review",
        "www.reallygreatsite.com": "Review 3",
        "E-mail": "Project",
        "hello@reallygreatsite.com": "ATS CV-JD Matching",
        "Salford & Co.": "ATS CV-JD Matching",
    }
    for sh in close.shapes:
        if not sh.has_text_frame:
            continue
        raw = sh.text_frame.text.strip()
        if raw in close_repl:
            runs = [r for p in sh.text_frame.paragraphs for r in p.runs]
            runs[0].text = close_repl[raw]
            for r in runs[1:]:
                r.text = ""
            if raw == "Salford & Co.":
                sh.left = Inches(1.04)
                sh.width = Inches(8.0)
                runs[0].font.color.rgb = DARK
            elif raw == "Let’s Build the Future Together":
                sh.top = Inches(5.35)
                sh.height = Inches(2.55)
                sh.width = Inches(14.5)
    chrome_text(close, 22, dark=False)
    close.dark = False
    text(close, "[TEAM] Thank you to [Supervisor] and the evaluation council.",
         1.02, 8.20, 16.0, 0.65, 24, bold=True, color=BLUE)
    slides.append(close)

    return slides


def reorder_and_prune(slides: list) -> None:
    ids = prs.slides._sldIdLst
    by_part = {prs.part.related_part(e.rId): e for e in list(ids)}
    keep = [by_part[sl.part] for sl in slides]
    for e in list(ids):
        if e not in keep:
            prs.part.drop_rel(e.rId)
        ids.remove(e)
    for e in keep:
        ids.append(e)


def self_check() -> None:
    assert len(prs.slides) == 22, f"Expected 22 slides, got {len(prs.slides)}"

    outline_text = OUTLINE.read_text(encoding="utf-8")
    outline_nums = set(NUM_RE.findall(outline_text))
    # Page numbers 02..22 appear in footer chrome; accept normalized integer match only for the footer Page NN shape.
    bad_nums = []

    for n, sl in enumerate(prs.slides, 1):
        for sh in sl.shapes:
            for tf in _iter_text_frames(sh):
                txt = tf.text
                for b in BANNED:
                    assert b.lower() not in txt.lower(), f"Slide {n} still has template text {b!r}: {txt[:80]}"
                for p in tf.paragraphs:
                    for r in p.runs:
                        if r.font.size is not None and r.font.size < Pt(MIN_PT) and r.text.strip():
                            raise AssertionError(f"Slide {n} font < 18pt ({r.font.size.pt}pt): {r.text!r}")

                stripped = txt.strip()
                is_footer_page = stripped.isdigit() and len(stripped) == 2 and sh.top > Inches(10.0) and sh.left > Inches(17.5)
                for m in NUM_RE.findall(txt):
                    if is_footer_page and int(m) == n:
                        continue
                    if m not in outline_nums:
                        bad_nums.append((n, m, stripped[:80]))

        for rel in sl.part.rels.values():
            assert rel.is_external or rel.target_part is not None, f"Slide {n} broken rel {rel.rId}"

    assert not bad_nums, f"Numbers not found in slide_outline.md: {bad_nums}"


def render_pngs() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    for old in PNG_DIR.glob("*.png"):
        old.unlink()
    ps = (
        "$pp = $null; $d = $null; "
        "try { "
        "$pp = New-Object -ComObject PowerPoint.Application; "
        f'$d = $pp.Presentations.Open("{OUT}", $true, $false, $false); '
        f'foreach ($i in 1..$d.Slides.Count) {{ $d.Slides($i).Export("{PNG_DIR}\\slide_$($i.ToString("00")).png", "PNG", 1600, 900) }} '
        "} finally { "
        "if ($d -ne $null) { $d.Close() }; "
        "if ($pp -ne $null) { $pp.Quit() } "
        "}"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print("PNG", PNG_DIR)


def main() -> None:
    charts = generate_charts()
    slides = build_deck(charts)
    reorder_and_prune(slides)
    self_check()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print("OK", OUT)
    if "--no-png" not in sys.argv:
        render_pngs()


if __name__ == "__main__":
    main()
