"""Generate editable vector diagrams for the introduction and model overview."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (Arc, Circle, FancyArrowPatch, FancyBboxPatch,
                                Rectangle)

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figs")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

SURFACE = "#F7F7F8"
MUTED = "#EFEFF2"
TEXT = "#171717"
TEXT_MUTED = "#52525B"
BORDER = "#D4D4D8"
BRAND = "#0FDC78"
BRAND_SOFT = "#E5FFF2"
BRAND_TEXT = "#004D26"
PRIMARY = "#02743B"


def setup(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    return fig, ax


def box(ax, height, x, y, w, h, title, body="", note="", focus=False):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=12",
        linewidth=2.0 if focus else 1.25,
        edgecolor=PRIMARY if focus else BORDER,
        facecolor=BRAND_SOFT if focus else SURFACE,
        zorder=2,
    )
    ax.add_patch(patch)
    color = BRAND_TEXT if focus else TEXT
    ax.text(x + w / 2, y + 34, title, ha="center", va="center",
            fontsize=13, fontweight="bold", color=color, zorder=3)
    if body:
        ax.text(x + w / 2, y + 82, body, ha="center", va="center",
                fontsize=10.5, color=TEXT, zorder=3)
    if note:
        ax.text(x + w / 2, y + h - 30, note, ha="center", va="center",
                fontsize=9.5, color=TEXT_MUTED, zorder=3)
    return patch


def arrow(ax, x1, y1, x2, y2, primary=False, dashed=False, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
        linewidth=2.0 if primary else 1.5,
        color=PRIMARY if primary else TEXT_MUTED,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3, zorder=1,
    )
    ax.add_patch(patch)
    return patch


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        kwargs = {"dpi": 300} if ext == "png" else {}
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight",
                    pad_inches=0.06, facecolor="white", **kwargs)
    plt.close(fig)


def _panel(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
        linewidth=0.9, edgecolor=BORDER, facecolor="#FCFCFD", zorder=0))


def _dot(ax, xy, color, r=4.2):
    ax.add_patch(Circle(xy, r, facecolor=color, edgecolor="white",
                        linewidth=1.1, zorder=6))


def _card(ax, x, y, w, h, num, title, body, focus=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=8",
        linewidth=1.6 if focus else 0.9,
        edgecolor=PRIMARY if focus else BORDER,
        facecolor=BRAND_SOFT if focus else "white",
        zorder=2))
    cx, cy = x + 26, y + h - 24
    ax.add_patch(Circle((cx, cy), 10.5,
                        facecolor=PRIMARY if focus else MUTED,
                        edgecolor="none", zorder=3))
    ax.text(cx, cy - 0.4, str(num), ha="center", va="center", fontsize=7.6,
            fontweight="bold", color="white" if focus else TEXT_MUTED, zorder=4)
    ax.text(x + 46, cy, title, ha="left", va="center", fontsize=8.6,
            fontweight="bold", color=BRAND_TEXT if focus else TEXT, zorder=4)
    ax.text(x + 14, y + 14, body, ha="left", va="bottom", fontsize=7.0,
            color=TEXT_MUTED, linespacing=1.55, zorder=4)


def overview():
    width, height = 820, 400
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    for x, label in ((12, "A"), (410, "B")):
        ax.text(x, 382, f"{label}", fontsize=10.5, fontweight="bold", color=PRIMARY)
    ax.text(34, 382, "DIRECTION IS NOT AMPLITUDE", fontsize=10.5,
            fontweight="bold", color=TEXT)
    ax.text(432, 382, "THREE-TIER AUDIT OF ONE LATENT SYSTEM", fontsize=10.5,
            fontweight="bold", color=TEXT)

    _panel(ax, 12, 12, 386, 344)
    _panel(ax, 410, 12, 398, 344)

    # --- Panel A: latent-space sketch -------------------------------------
    ax.text(24, 342, "schematic: one source state, one target, one policy displacement",
            fontsize=6.9, color=TEXT_MUTED)

    S, T, P = (72, 242), (123, 318), (173, 288)
    ax.add_patch(Arc(S, 84, 84, theta1=26, theta2=58,
                     linewidth=0.9, color=TEXT_MUTED, zorder=2))
    arrow(ax, S[0], S[1], T[0], T[1], primary=False, dashed=True)
    arrow(ax, S[0], S[1], P[0], P[1], primary=True)
    _dot(ax, S, TEXT)
    _dot(ax, T, "#71717A")
    _dot(ax, P, PRIMARY)

    ax.text(72, 222, r"source  p$K_a$ 5.0", ha="center", va="center",
            fontsize=7.0, color=TEXT)
    ax.text(133, 319, r"target  p$K_a$ 7.5", ha="left", va="center",
            fontsize=7.0, color=TEXT_MUTED)
    ax.text(183, 285, r"policy $g$", ha="left", va="center",
            fontsize=7.0, color=PRIMARY)
    ax.text(93, 293, r"true $\Delta h$", ha="right", va="center",
            fontsize=7.0, color=TEXT_MUTED)
    ax.text(150, 258, "cosine 0.58", ha="left", va="center",
            fontsize=6.8, color=TEXT_MUTED)

    # amplitude scale below
    ax.text(24, 186, r"amplitude after projecting $g$ onto each readout",
            fontsize=6.9, color=TEXT_MUTED)
    zero = 200.0
    y_ridge, y_head = 142, 118
    ax.plot([24, 392], [98, 98], color=BORDER, linewidth=0.9, zorder=1)
    ax.plot([zero, zero], [104, 158], color=BORDER, linewidth=0.9,
            linestyle=(0, (3, 3)), zorder=1)
    for tick, lab in ((zero, "0"), (zero + 85, "0.3"), (zero + 170, "0.6")):
        ax.plot([tick, tick], [98, 94], color=BORDER, linewidth=0.9)
        ax.text(tick, 86, lab, ha="center", va="top", fontsize=6.3, color=TEXT_MUTED)

    ax.text(190, y_ridge, r"ridge $\sigma{=}g^{\top}w$", ha="right", va="center",
            fontsize=6.7, color=TEXT_MUTED)
    ax.text(190, y_head, "PkaHead gradient", ha="right", va="center",
            fontsize=6.7, color=TEXT_MUTED)
    ax.add_patch(Circle((zero, y_ridge), 4.0, facecolor="white",
                        edgecolor="#A1A1AA", linewidth=1.0, zorder=3))
    ax.add_patch(Rectangle((zero, y_head - 6), 154, 12,
                           facecolor=PRIMARY, edgecolor="none", zorder=3))
    ax.text(zero + 12, y_ridge, "≈ 0  (null)", ha="left", va="center",
            fontsize=6.7, color=TEXT_MUTED)
    ax.text(zero + 158, y_head, "0.585", ha="left", va="center",
            fontsize=6.7, color=PRIMARY)

    ax.text(24, 46, "Same displacement; the amplitude verdict flips with the readout.",
            fontsize=7.2, style="italic", color=TEXT)

    # --- Panel B: three tiers ---------------------------------------------
    cards = [
        (262, "Scalar prediction",
         "five architecture controls + repaired prompt\nisolate the gain of the latent pathway", False),
        (160, "Direction vs. amplitude",
         "target-conditioned policy (3 seeds) vs. two readouts:\nridge orthogonal, PkaHead relative order only", True),
        (58, "Decoded structural control",
         "input-slot editing (P3) and direct $h_T$ injection,\n80 sources, two decoding protocols", False),
    ]
    for i, (y, title, body, focus) in enumerate(cards, start=1):
        _card(ax, 424, y, 370, 84, i, title, body, focus)
    for y1, y2 in ((262, 244), (160, 142)):
        arrow(ax, 609, y1, 609, y2, primary=False)
    ax.text(609, 34, "Direction is learnable; amplitude is readout-dependent;\ndecoded control remains bounded.",
            ha="center", va="center", fontsize=7.6, fontweight="bold",
            color=PRIMARY, linespacing=1.6)

    save(fig, "fig0_latent_audit_overview")


def algorithm():
    width, height = 1200, 720
    fig, ax = setup(width, height)
    ax.text(35, 42, "A   PROPERTY READOUT", fontsize=11, fontweight="bold", color=TEXT_MUTED)
    ax.plot([35, 1165], [64, 64], color=BORDER, lw=1.0)

    box(ax, height, 30, 100, 220, 160, "Prompt embeddings", "text + ChemTokens", "+ six <latent> slots")
    box(ax, height, 300, 100, 240, 160, "Coconut feedback", "hidden state → slot", "six feedback writes", focus=True)
    box(ax, height, 590, 100, 220, 160, "Final hidden state", "h_T", "designated readout position")
    box(ax, height, 860, 100, 240, 160, "PkaHead", "nonlinear scalar readout", "predicted p$K_a$")
    for x1, x2 in [(250, 300), (540, 590), (810, 860)]:
        arrow(ax, x1, 180, x2, 180, primary=True)
    for j in range(6):
        ax.add_patch(Circle((361 + j * 24, 207), 6.5, facecolor="white",
                            edgecolor=PRIMARY, linewidth=1.3, zorder=4))

    ax.text(35, 355, "B   POLICY + GENERATION AUDIT", fontsize=11, fontweight="bold", color=TEXT_MUTED)
    ax.plot([35, 1165], [377, 377], color=BORDER, lw=1.0)
    box(ax, height, 30, 425, 220, 165, "Pair conditioning", "source state and p$K_a$\ntarget Δp$K_a$", "training pairs only")
    box(ax, height, 300, 425, 240, 165, "Policy ensemble", "three seeds → unit mean g", "direction is separate from scale")
    box(ax, height, 590, 425, 240, 165, "Two control surfaces", "P3: optimize six input slots\nDirect: $h_T \\leftarrow h_T+s\\hat g_i$", "direct path: no optimization", focus=True)
    box(ax, height, 880, 425, 240, 165, "Frozen decoder", "greedy + sampled", "SMILES + audit metrics")
    for x1, x2 in [(250, 300), (540, 590), (830, 880)]:
        arrow(ax, x1, 507, x2, 507, primary=False, dashed=True)
    arrow(ax, 700, 260, 700, 413, primary=False, dashed=True)
    ax.text(718, 335, "target state", fontsize=9.5, color=TEXT_MUTED, va="center")

    patch = FancyBboxPatch((215, 635), 770, 48,
                           boxstyle="round,pad=0.012,rounding_size=12",
                           linewidth=1.0, edgecolor=BORDER, facecolor=MUTED)
    ax.add_patch(patch)
    ax.text(600, 659, "Controls: no-op  •  fixed seeds  •  validity  •  Tanimoto  •  learned-oracle direction",
            ha="center", va="center", fontsize=9.5, color=TEXT_MUTED)
    save(fig, "fig0_latent_algorithm")


if __name__ == "__main__":
    overview()
    algorithm()
    print("Wrote fig0_latent_audit_overview and fig0_latent_algorithm (SVG/PDF/PNG)")
