"""Thesis figures drawn with matplotlib (no Graphviz/Mermaid on the build machine): PNG (300 dpi) + SVG.

  python -m scripts.thesis_figures      # -> output/review3/figures/fig_*.png|svg

fig_service_flow      end-to-end service flow (replaces the text diagram "Figure 1")
fig_extraction_loop   validate-repair-retry extraction loop (replaces the text diagram "Figure 2")
fig_confit_vs_ours    ConFit v2 (paper baseline) next to our pipeline
fig_deployment        deployment used for Review 3: Compose on the presenting PC, models on Runpod, databases on Azure
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "output/review3/figures"
INK, MUTED, LINE = "#1f2933", "#52606d", "#9aa5b1"
FILL = {"io": "#eef2f7", "llm": "#dbe8f6", "rule": "#e3f1e8", "store": "#f6eedb", "guard": "#f7e1dc", "ext": "#ece6f6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def box(ax, x, y, w, h, text, kind="io", size=10, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=FILL[kind], ec=LINE, lw=1.1))
    ax.text(x, y, text, ha="center", va="center", color=INK, fontsize=size, fontweight="bold" if bold else "normal",
            linespacing=1.35, wrap=True)


def arrow(ax, a, b, text=None, rad=0.0, color=MUTED):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text:
        ax.text((a[0] + b[0]) / 2 + 0.08, (a[1] + b[1]) / 2, text, fontsize=8.5, color=MUTED, va="center")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def service_flow():
    fig, ax = canvas(9, 11)
    steps = [
        ("CV file (PDF, DOCX, TXT) or pasted text\noptional pasted job description", "io"),
        ("Text extraction + injection guard\n(lines aimed at the AI are removed and reported)", "guard"),
        ("[extract]  qwen-v4: Qwen2.5-7B + V4 QLoRA adapter\nvalidate – repair – retry  ·  cached by text hash", "llm"),
        ("[retrieve]  BGE-M3 dense + BM25 → reciprocal rank fusion\npool of 20 JDs  (skipped for named or pasted JDs)", "store"),
        ("[score]  scorer V3.1: seven rule scores per CV–JD pair\nskills · title · experience · education · field · location · salary", "rule"),
        ("[rank]  final model: rule scores + Qwen3-Embedding cosines + TF-IDF\n→ verdict and fit (0–100), ordered jobs", "rule"),
        ("[explain]  qwen-o5 writes the explanation from the facts sheet\nfact guard: rejected twice → template from the facts", "llm"),
        ("JSON response or server-sent events → web page\nranked cards, seven-score radar, explanation, removed lines", "io"),
    ]
    ys = [10.2 - i * 1.35 for i in range(len(steps))]
    for (t, k), y in zip(steps, ys):
        box(ax, 4.5, y, 7.6, 0.95, t, k, size=9.5)
    for y0, y1 in zip(ys, ys[1:]):
        arrow(ax, (4.5, y0 - 0.48), (4.5, y1 + 0.48))
    save(fig, "fig_service_flow")


def extraction_loop():
    fig, ax = canvas(10, 4.2)
    box(ax, 1.3, 2.6, 1.9, 0.9, "generate\n(qwen-v4, greedy)", "llm")
    box(ax, 3.9, 2.6, 1.9, 0.9, "repair\n(deterministic)", "rule")
    box(ax, 6.5, 2.6, 1.9, 0.9, "validate\n(schema + quotes)", "rule")
    box(ax, 9.0, 2.6, 1.6, 0.9, "valid?", "guard", bold=True)
    box(ax, 9.0, 0.7, 1.6, 0.7, "done", "io", bold=True)
    for a, b in (((2.25, 2.6), (2.95, 2.6)), ((4.85, 2.6), (5.55, 2.6)), ((7.45, 2.6), (8.2, 2.6))):
        arrow(ax, a, b)
    arrow(ax, (9.0, 2.15), (9.0, 1.05), "yes")
    arrow(ax, (9.0, 3.05), (1.3, 3.05), rad=0.35)
    ax.text(5.15, 3.95, "no → next attempt with repetition penalty 1.05 → 1.1 → 1.15 → 1.2", ha="center", color=MUTED, fontsize=9)
    ax.text(3.8, 0.55, "token cap = min(16,384, 3 × source tokens + 4,000)\nan output that stops on length is never accepted",
            ha="center", color=MUTED, fontsize=8.5)
    save(fig, "fig_extraction_loop")


def confit_vs_ours():
    fig, ax = canvas(11, 7.6)
    ax.text(2.7, 7.25, "ConFit v2 (paper baseline)", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(8.2, 7.25, "Our pipeline", ha="center", fontsize=12, fontweight="bold", color=INK)
    left = [("Resume text  +  job post text", "io"),
            ("HyRE: an LLM writes a hypothetical resume\nadded to the job post (left out in our run:\nno commercial API allowed)", "ext"),
            ("Dual encoder (e5-base)\ncontrastive training, runner-up hard negatives", "llm"),
            ("Cosine similarity", "rule"),
            ("Ranked list: one number per pair", "io")]
    right = [("CV file  +  job descriptions (Vietnamese / English)", "io"),
             ("LLM extraction to a schema, every skill quoted\nvalidate – repair – retry (O1, O4)", "llm"),
             ("Hybrid retrieval: BM25 + BGE-M3, RRF (O2)", "store"),
             ("Seven rule scores + Qwen3 cosines + TF-IDF\n→ verdict and fit (O3)", "rule"),
             ("Verdict, fit, matched / missing skills with quotes,\nfact-checked explanation (O5)", "io")]
    ys = [6.3 - i * 1.4 for i in range(5)]
    for col, x in ((left, 2.7), (right, 8.2)):
        for (t, k), y in zip(col, ys):
            box(ax, x, y, 4.6, 1.05, t, k, size=9.2)
        for y0, y1 in zip(ys, ys[1:]):
            arrow(ax, (x, y0 - 0.53), (x, y1 + 0.53))
    ax.plot([5.45, 5.45], [0.3, 7.0], color=LINE, lw=0.8, ls=(0, (4, 4)))
    save(fig, "fig_confit_vs_ours")


def deployment():
    fig, ax = canvas(11, 6.4)
    ax.add_patch(FancyBboxPatch((0.3, 0.4), 5.6, 5.4, boxstyle="round,pad=0.02,rounding_size=0.12", fc="#fafbfc", ec=LINE, lw=1))
    ax.text(3.1, 5.55, "Presenting PC · Docker Compose", ha="center", fontweight="bold", color=INK)
    box(ax, 1.4, 4.6, 1.7, 0.75, "Browser", "io")
    box(ax, 4.2, 4.6, 2.6, 0.75, "frontend\nNext.js  :3000", "io")
    box(ax, 4.2, 3.0, 2.6, 1.0, "api\nFastAPI + LangGraph  :8000\nBGE-M3 on CPU", "rule")
    box(ax, 1.6, 1.4, 2.2, 0.75, "Prometheus  :9090", "store", size=9)
    box(ax, 4.4, 1.4, 2.2, 0.75, "Grafana  :3001", "store", size=9)
    box(ax, 8.6, 4.4, 4.2, 1.5, "Runpod serverless endpoint\n(load balancer, scales to zero)\nOllama: qwen-v4, qwen-o5,\nqwen3-embedding", "llm", size=9)
    box(ax, 8.6, 2.0, 4.2, 1.25, "Azure Database for PostgreSQL\npgvector (HNSW, cosine) · match history", "store", size=9)
    arrow(ax, (2.25, 4.6), (2.9, 4.6))
    arrow(ax, (4.2, 4.22), (4.2, 3.5), "/api")
    arrow(ax, (5.5, 3.3), (6.5, 3.95))
    ax.text(5.95, 3.3, "HTTPS\nAPI key", fontsize=8.5, color=MUTED, va="top")
    arrow(ax, (5.5, 2.75), (6.5, 2.2), "TLS")
    arrow(ax, (2.0, 1.78), (3.1, 2.6), "/metrics", rad=-0.2)
    arrow(ax, (2.7, 1.4), (3.3, 1.4))
    ax.text(5.5, 0.05, "Offline mode: Compose profiles local-gpu (Ollama on the PC) and local-db (PostgreSQL containers) replace the cloud parts.",
            ha="center", color=MUTED, fontsize=8.5)
    save(fig, "fig_deployment")


if __name__ == "__main__":
    service_flow()
    extraction_loop()
    confit_vs_ours()
    deployment()
