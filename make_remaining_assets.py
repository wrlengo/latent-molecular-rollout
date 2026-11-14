"""Build the remaining main-text and SI figures from verified result files."""
import collections
import csv
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")
TABLES = os.path.join(ROOT, "tables")
os.makedirs(FIGS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
})

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
SKY = "#56B4E9"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#999999"
RNG = np.random.default_rng(42)


def load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def save_figure(fig, stem):
    for ext in ("png", "pdf"):
        path = os.path.join(FIGS, f"{stem}.{ext}")
        fig.savefig(path, dpi=300)
        print(f"saved {path}")


def panel_label(ax, label):
    ax.text(-0.15, 1.04, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def box_jitter(ax, values, position, color, width=0.52, point_size=22):
    values = np.asarray(values, dtype=float)
    ax.boxplot([values], positions=[position], widths=[width], showfliers=False,
               patch_artist=True, whis=1.5,
               boxprops=dict(facecolor=color, alpha=0.20, edgecolor=color, lw=1.2),
               medianprops=dict(color="black", lw=1.5),
               whiskerprops=dict(color=color, lw=1.1),
               capprops=dict(color=color, lw=1.1))
    jitter = RNG.uniform(-width * 0.27, width * 0.27, len(values))
    ax.scatter(np.full(len(values), position) + jitter, values, s=point_size,
               color=color, alpha=0.82, edgecolor="white", linewidth=0.35, zorder=3)


def violin_jitter(ax, values, position, color, width=0.72, point_size=22):
    values = np.asarray(values, dtype=float)
    parts = ax.violinplot([values], positions=[position], widths=[width],
                          showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
        body.set_linewidth(1.0)
    ax.boxplot([values], positions=[position], widths=[0.12], showfliers=False,
               patch_artist=True, whis=1.5,
               boxprops=dict(facecolor="white", alpha=0.85, edgecolor=color, lw=1.0),
               medianprops=dict(color="black", lw=1.4),
               whiskerprops=dict(color=color, lw=1.0),
               capprops=dict(color=color, lw=1.0))
    jitter = RNG.uniform(-width * 0.18, width * 0.18, len(values))
    ax.scatter(np.full(len(values), position) + jitter, values, s=point_size,
               color=color, alpha=0.78, edgecolor="white", linewidth=0.35, zorder=3)


def add_bar_labels(ax, bars, fmt="{:.3f}", dy=0.02, fontsize=8):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + dy, fmt.format(value),
                ha="center", va="bottom", fontsize=fontsize)


# ---------------------------------------------------------------------------
# Main Fig. 2: G1 interpolation and G2 on-manifold optimization trajectories.
# ---------------------------------------------------------------------------
interp = load_json("fig1_interp.json")
alphas = np.asarray(interp["alphas"])
curves = np.asarray(interp["curves"])
assert curves.shape == (38, 21)
assert interp["monotone_frac"] == 1.0
net_change = curves[:, -1] - curves[:, 0]
n_up = int(np.sum(net_change > 0))
n_down = int(np.sum(net_change < 0))

rollout = load_json("fig2_final.json")
pca_xy = np.asarray(rollout["pca_xy"])
pred = np.asarray(rollout["pred"])
trajectories = rollout["trajectories"]
assert pca_xy.shape == (2302, 2) and len(trajectories) == 30
off_mean = float(np.mean([t["off_manifold"] for t in trajectories]))
n_loss_down = sum(t["loss1"] < t["loss0"] for t in trajectories)
assert abs(off_mean - 0.0716579153) < 1e-8 and n_loss_down == 30

fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), gridspec_kw={"width_ratios": [0.92, 1.25]})
ax = axes[0]
for curve, change in zip(curves, net_change):
    color = GREEN if change > 0 else VERMILION if change < 0 else GRAY
    ax.plot(alphas, curve, color=color, alpha=0.58, lw=1.0)
ax.plot([], [], color=GREEN, label=f"Increasing (n = {n_up})")
ax.plot([], [], color=VERMILION, label=f"Decreasing (n = {n_down})")
ax.set_xlabel(r"Interpolation coefficient $\alpha$")
ax.set_ylabel(r"Predicted p$K_a$")
ax.set_title("G1 | Latent interpolation is continuous and monotonic")
ax.text(0.03, 0.97,
        f"Monotonic: 38/38 (100%)\nMedian jump/span: {interp['jump_ratio_median']:.3f}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.16, lw=0.5)
panel_label(ax, "a")

ax = axes[1]
background = ax.scatter(pca_xy[:, 0], pca_xy[:, 1], c=pred, s=5, alpha=0.28,
                        cmap="viridis", edgecolors="none", rasterized=True)
for i, trajectory in enumerate(trajectories):
    path = np.asarray(trajectory["path"])
    ax.plot(path[:, 0], path[:, 1], color=VERMILION, alpha=0.52, lw=0.85,
            zorder=3, label="Optimization path" if i == 0 else None)
    ax.scatter(path[0, 0], path[0, 1], s=13, color="white", edgecolor=VERMILION,
               linewidth=0.55, zorder=4)
    ax.scatter(path[-1, 0], path[-1, 1], s=12, color=VERMILION, edgecolor="none", zorder=4)
ax.set_xlabel(f"PC1 ({rollout['explained'][0] * 100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({rollout['explained'][1] * 100:.1f}% variance)")
ax.set_title("G2 | Gradient trajectories remain near the data manifold")
ax.text(0.02, 0.98,
        f"30 trajectories; 20 steps each\nMean off-manifold: {off_mean:.3f}\nTarget loss reduced: {n_loss_down}/30",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax.legend(frameon=False, loc="lower right")
cbar = fig.colorbar(background, ax=ax, pad=0.02, fraction=0.045)
cbar.set_label(r"Background predicted p$K_a$")
panel_label(ax, "b")
fig.tight_layout(w_pad=2.0)
save_figure(fig, "fig2_latent_physical_structure_main")
plt.close(fig)


# ---------------------------------------------------------------------------
# Main Fig. 3: G3 latent intervention and G5 direction-sign alignment.
# ---------------------------------------------------------------------------
diagnosis = load_json("fig3_nonresp_diagnosis.json")
responsive = diagnosis["details"]["responsive"]
nonresponsive = diagnosis["details"]["nonresponsive"]
assert len(responsive) == 18 and len(nonresponsive) == 22
intervention_rows = responsive + nonresponsive
delta_pred = np.asarray([r["delta_pred"] for r in intervention_rows], dtype=float)
response = np.abs(delta_pred) > 0
moved = np.asarray([r["moved"] for r in intervention_rows], dtype=bool)
assert response.sum() == 18 and moved.sum() == 12
assert abs(delta_pred.mean() - (-0.04)) < 1e-8
assert abs(delta_pred.std() - 0.3942080669) < 1e-6

gate5 = load_json("gate45_alignment.json")["gate5"]
n_correct = gate5["n_correct"]
n_total = gate5["n_pairs"]
n_incorrect = n_total - n_correct
p_binom = sum(math.comb(n_total, k) for k in range(n_correct, n_total + 1)) / 2 ** n_total
assert (n_correct, n_total) == (71, 105)

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3), gridspec_kw={"width_ratios": [1.0, 0.9, 0.75]})
ax = axes[0]
positions = np.where(response, 1, 0) + RNG.uniform(-0.12, 0.12, len(delta_pred))
for flag, color, label in ((False, GRAY, "No response"), (True, BLUE, "Response")):
    mask = response == flag
    ax.scatter(positions[mask], delta_pred[mask], s=34, color=color, alpha=0.85,
               edgecolor="white", linewidth=0.4, label=f"{label} (n = {mask.sum()})")
ax.axhline(0, color="0.25", lw=0.9, ls="--")
ax.set_xlim(-0.55, 1.55)
ax.set_xticks([0, 1], ["No response\n(n = 22)", "Response\n(n = 18)"])
ax.set_ylabel(r"$\Delta$ text-parsed p$K_a$ after latent intervention")
ax.set_title("G3 | Latent intervention conditionally changes decoded text")
ax.text(0.03, 0.97,
        f"Response: 18/40 (45%)\nMoved toward target: 12/40 (30%)\nMean = {delta_pred.mean():.3f}; SD = {delta_pred.std():.3f}",
        transform=ax.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax.text(0.03, 0.04, "Response: generated text changed; moved: parsed value closer to target",
        transform=ax.transAxes, fontsize=7.5, color="0.35")
ax.grid(axis="y", alpha=0.18, lw=0.5)
panel_label(ax, "a")

ax_hist = axes[1]
abs_delta = np.abs(delta_pred)
median_abs = float(np.median(abs_delta))
frac_below_02 = float(np.mean(abs_delta < 0.2))
assert median_abs == 0.0 and abs(frac_below_02 - 0.75) < 1e-12
ax_hist.hist(abs_delta, bins=np.linspace(0, max(1.0, abs_delta.max()), 16),
             color=SKY, edgecolor="white", linewidth=0.6)
ax_hist.axvline(0.2, color=ORANGE, ls="--", lw=1.1, label="0.2 threshold")
ax_hist.axvline(median_abs, color="black", ls=":", lw=1.2, label="Median = 0")
ax_hist.set_xlabel(r"$|\Delta$ text-parsed p$K_a|$")
ax_hist.set_ylabel("Number of interventions")
ax_hist.set_title("G3 | Text-parsed responses are quantized and often absent")
ax_hist.text(0.97, 0.96, "All predictions on a 0.1 grid\n22/40 leave decoded text unchanged",
             transform=ax_hist.transAxes, va="top", ha="right",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax_hist.grid(axis="y", alpha=0.18, lw=0.5)
ax_hist.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.68), fontsize=7.5)
panel_label(ax_hist, "b")

ax = axes[2]
bars = ax.bar([0, 1], [n_correct, n_incorrect], color=[GREEN, GRAY], width=0.60)
ax.axhline(n_total / 2, color=ORANGE, ls="--", lw=1.1, label="Chance expectation")
ax.set_xticks([0, 1], ["Correct sign", "Incorrect sign"])
ax.set_ylim(0, 88)
ax.set_ylabel("Number of molecular pairs")
ax.set_title(r"G5 | Gradient direction predicts p$K_a$ sign")
for bar, value in zip(bars, [n_correct, n_incorrect]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 2, str(value), ha="center", fontsize=10)
ax.text(0.97, 0.97,
        f"Accuracy = {n_correct}/{n_total} = {n_correct / n_total:.3f}\nOne-sided exact binomial\np = {p_binom:.2e}",
        transform=ax.transAxes, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", alpha=0.18, lw=0.5)
panel_label(ax, "c")
fig.tight_layout(w_pad=2.5)
save_figure(fig, "fig3_causal_intervention_direction_main")
plt.close(fig)


# ---------------------------------------------------------------------------
# Main Fig. 5: stratified G4. Small gaps remain misaligned.
# ---------------------------------------------------------------------------
g4 = load_json("gate4_stratified.json")["pair_sets"]
original = g4["original_top20_tani>=0.6"]["original_signed_cosine"]
bins = ["1.0-2.0", "2.0-4.0", ">=4.0"]
bin_labels = [r"$1 \leq |\Delta\mathrm{pKa}| < 2$", r"$2 \leq |\Delta\mathrm{pKa}| < 4$", r"$|\Delta\mathrm{pKa}| \geq 4$"]
assert [original[b]["n"] for b in bins] == [6, 7, 7]
assert all(original[b]["n_cosine_gt_0.5"] == 0 for b in bins)

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True,
                         gridspec_kw={"width_ratios": [1.05, 1.0]})
ax = axes[0]
for i, (b, label) in enumerate(zip(bins, bin_labels)):
    box_jitter(ax, original[b]["values"], i, [BLUE, ORANGE, VERMILION][i], point_size=32)
    ax.text(i, -0.41, f"n = {original[b]['n']}; median = {original[b]['median']:.3f}",
            ha="center", va="bottom", fontsize=8)
ax.axhline(0, color="0.25", lw=0.8)
ax.axhline(0.2, color=ORANGE, lw=1.0, ls=":", label="Weak/strong boundary = 0.2")
ax.axhline(0.5, color=GREEN, lw=1.0, ls="--", label="Upper-range check = 0.5")
ax.set_xticks(range(3), bin_labels)
ax.set_ylim(-0.45, 0.61)
ax.set_ylabel(r"Cosine: $\nabla_h\,\mathrm{PkaHead}$ vs. latent secant")
ax.set_title(r"Original top-20 pairs, stratified by p$K_a$ gap")
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
ax.grid(axis="y", alpha=0.16, lw=0.5)
panel_label(ax, "a")

ax = axes[1]
selection_specs = [
    ("original_top20_tani>=0.6", "Original top 20\nTani ≥ 0.6", BLUE),
    ("all_tani>=0.6", "All pairs\nTani ≥ 0.6", PURPLE),
    ("all_tani>=0.7", "All pairs\nTani ≥ 0.7", GREEN),
]
for i, (key, label, color) in enumerate(selection_specs):
    stats = g4[key]["original_signed_cosine"]["1.0-2.0"]
    box_jitter(ax, stats["values"], i, color, point_size=32)
    ax.text(i, -0.41, f"n = {stats['n']}; median = {stats['median']:.3f}",
            ha="center", va="bottom", fontsize=8)
ax.axhline(0, color="0.25", lw=0.8)
ax.axhline(0.2, color=ORANGE, lw=1.0, ls=":")
ax.axhline(0.5, color=GREEN, lw=1.0, ls="--")
ax.set_xticks(range(3), [s[1] for s in selection_specs])
ax.set_title("Small-gap result is robust to pair selection")
ax.text(0.97, 0.96, "All small-gap medians < 0.2\n0/20 original pairs have cosine > 0.5",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax.grid(axis="y", alpha=0.16, lw=0.5)
panel_label(ax, "b")
fig.suptitle("G4 | Weak gradient–secant alignment is not explained by large pKa jumps", y=1.03, fontsize=11)
fig.tight_layout(w_pad=2.2)
save_figure(fig, "fig5_g4_stratified_main")
plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. S1: SMI-TED zero ablation for B′.
# ---------------------------------------------------------------------------
ablation = load_json("ablation_smited.json")
assert ablation["n"] == 1151
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
labels = ["Normal", "SMI-TED zeroed"]
colors = [GREEN, GRAY]
ax = axes[0]
bars = ax.bar([0, 1], [ablation["mae_normal"], ablation["mae_zero"]], color=colors, width=0.58)
ax.set_xticks([0, 1], labels)
ax.set_ylim(0, 3.25)
ax.set_ylabel(r"Held-out MAE (p$K_a$ units)")
ax.set_title("Prediction error")
add_bar_labels(ax, bars, dy=0.06, fontsize=9)
ax.text(0.5, 0.90, f"ΔMAE = +{ablation['delta']:.3f}", transform=ax.transAxes,
        ha="center", color=VERMILION, fontweight="bold")
ax.grid(axis="y", alpha=0.18, lw=0.5)
panel_label(ax, "a")

ax = axes[1]
bars = ax.bar([0, 1], [ablation["pred_std_normal"], ablation["pred_std_zero"]], color=colors, width=0.58)
ax.axhline(ablation["gt_std"], color=BLUE, ls="--", lw=1.0,
           label=f"Ground-truth SD = {ablation['gt_std']:.3f}")
ax.set_xticks([0, 1], labels)
ax.set_ylim(0, 3.95)
ax.set_ylabel(r"Standard deviation of p$K_a$")
ax.set_title("Prediction variation collapses")
add_bar_labels(ax, bars, dy=0.07, fontsize=9)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.18, lw=0.5)
panel_label(ax, "b")
fig.suptitle(f"Fig. S1 | B′ causally depends on molecular features (held-out n = {ablation['n']})", y=1.03, fontsize=11)
fig.tight_layout(w_pad=2.6)
save_figure(fig, "figS1_smited_zero_ablation")
plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. S2: head-input controls (B′/B″) versus latent intervention (C).
# ---------------------------------------------------------------------------
bp = load_json("fig3_Bp_hT_intervene.json")
bpp = load_json("fig3_Bpp_hT_intervene.json")
intervention_data = [bp["intervene"], bpp["intervene"], intervention_rows]
response_rates = [1.0, 1.0, response.mean()]
moved_rates = [1.0, 0.975, moved.mean()]
assert bpp["intervene_summary"]["n_responded"] == 40
assert bpp["intervene_summary"]["n_moved_toward_target"] == 39

fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1.05, 1.0]})
ax = axes[0]
for i, (rows, color) in enumerate(zip(intervention_data, [GREEN, SKY, VERMILION])):
    violin_jitter(ax, [r["delta_pred"] for r in rows], i, color, point_size=18)
ax.axhline(0, color="0.25", ls="--", lw=0.9)
ax.set_xticks([0, 1, 2], ["B′: $h_T$", "B″: $h_T$", "C: $z$ via LLM"])
ax.set_ylabel(r"$\Delta$ predicted p$K_a$")
ax.set_title("Intervention effect distributions (n = 40 per group)")
ax.grid(axis="y", alpha=0.17, lw=0.5)
panel_label(ax, "a")

ax = axes[1]
x = np.arange(3)
width = 0.34
bars1 = ax.bar(x - width / 2, response_rates, width, color=BLUE, label="Response")
bars2 = ax.bar(x + width / 2, moved_rates, width, color=ORANGE, label="Moved toward target")
ax.set_xticks(x, ["B′: $h_T$", "B″: $h_T$", "C: $z$ via LLM"])
ax.set_ylim(0, 1.18)
ax.set_ylabel("Fraction of 40 samples")
ax.set_title("Head-input controls exceed latent-path responses")
for bars in (bars1, bars2):
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value * 100:.1f}%",
                ha="center", fontsize=8)
ax.legend(frameon=False, loc="upper right")
ax.grid(axis="y", alpha=0.17, lw=0.5)
panel_label(ax, "b")
fig.text(0.5, -0.01,
         "B′/B″: constructive intervention at the PkaHead input (no LLM forward); C: latent intervention propagated through the LLM.",
         ha="center", fontsize=8.5)
fig.suptitle("Fig. S2 | Readout-head controls and latent-path intervention", y=1.03, fontsize=11)
fig.tight_layout(w_pad=2.4)
save_figure(fig, "figS2_head_input_intervention_controls")
plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. S3: responsive versus non-responsive intervention diagnostics.
# ---------------------------------------------------------------------------
gaps_r = np.asarray(diagnosis["responsive"]["gaps"])
gaps_n = np.asarray(diagnosis["nonresponsive"]["gaps"])
ion_r = diagnosis["responsive"]["ionizable_frac"]
ion_n = diagnosis["nonresponsive"]["ionizable_frac"]
assert (len(gaps_r), len(gaps_n)) == (18, 22)

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
ax = axes[0]
violin_jitter(ax, gaps_n, 0, GRAY, point_size=28)
violin_jitter(ax, gaps_r, 1, BLUE, point_size=28)
ax.axhline(2.0, color=ORANGE, ls="--", lw=1.0)
ax.text(1.42, 2.15, "Gap = 2.0", color=ORANGE, fontsize=8, ha="right", va="bottom")
ax.set_xticks([0, 1], [f"Non-responsive\n(n = 22)", f"Responsive\n(n = 18)"])
ax.set_ylabel(r"$|\mathrm{target} - \mathrm{base\ prediction}|$")
ax.set_title("Target-gap distributions overlap")
ax.text(0, max(gaps_n) + 0.22, f"Median = {np.median(gaps_n):.2f}\nGap > 2: {np.mean(gaps_n > 2):.2f}", ha="center", fontsize=8)
ax.text(1, max(gaps_r) + 0.22, f"Median = {np.median(gaps_r):.2f}\nGap > 2: {np.mean(gaps_r > 2):.2f}", ha="center", fontsize=8)
ax.set_ylim(0, max(gaps_r.max(), gaps_n.max()) + 1.35)
ax.grid(axis="y", alpha=0.17, lw=0.5)
panel_label(ax, "a")

ax = axes[1]
bars = ax.bar([0, 1], [ion_n, ion_r], color=[GRAY, BLUE], width=0.58)
ax.set_xticks([0, 1], ["Non-responsive\n(n = 22)", "Responsive\n(n = 18)"])
ax.set_ylim(0, 1.0)
ax.set_ylabel("Fraction with an ionizable group")
ax.set_title("Ionizability does not separate outcomes")
for bar, value, count, total in zip(bars, [ion_n, ion_r], [17, 11], [22, 18]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035,
            f"{value:.3f}\n({count}/{total})", ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.17, lw=0.5)
panel_label(ax, "b")
fig.text(0.5, -0.01,
         "These overlapping summaries do not support target inaccessibility as a simple explanation of non-response.",
         ha="center", fontsize=8.5)
fig.suptitle("Fig. S3 | Diagnostics of the 40 latent-intervention samples", y=1.03, fontsize=11)
fig.tight_layout(w_pad=2.6)
save_figure(fig, "figS3_intervention_nonresponse_diagnostics")
plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. S5: mild and cross-band OOD MAE.
# ---------------------------------------------------------------------------
band = load_json("figdata.json")["band"]
groups = ["A", "B", "Bp", "C"]
group_labels = ["A: Text + CoT", "B: Text only", "B′: Molecule + head", "C: Molecule + latent"]
group_colors = [BLUE, ORANGE, GREEN, VERMILION]
expected_mild = [7.8620370370, 6.0855185185, 9.9411171332, 9.1582495886]
expected_cross = [17.9372602740, 17.6519863014, 16.3747245224, 16.3236832459]
mild = [band[g]["mild"]["mae"] for g in groups]
cross = [band[g]["cross"]["mae"] for g in groups]
assert np.allclose(mild, expected_mild) and np.allclose(cross, expected_cross)

fig, ax = plt.subplots(figsize=(9.8, 4.5))
x = np.arange(2)
width = 0.18
for i, (label, color, values) in enumerate(zip(group_labels, group_colors, zip(mild, cross))):
    bars = ax.bar(x + (i - 1.5) * width, values, width, color=color, label=label)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.28, f"{value:.2f}", ha="center", fontsize=7.5)
ax.set_xticks(x, ["Mild band\n(n = 810)", "Cross band\n(n = 292)"])
ax.set_ylim(0, 20.8)
ax.set_ylabel(r"MAE (p$K_a$ units)")
ax.set_title("Fig. S5 | Band-specific OOD errors after tiny-molecule cleaning")
ax.legend(frameon=False, ncol=2, loc="upper left")
ax.grid(axis="y", alpha=0.18, lw=0.5)
ax.text(0.98, 0.96, "Text-only B is best in the mild band;\nall groups have large cross-band errors.",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
fig.tight_layout()
save_figure(fig, "figS5_band_mae")
plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. S6: pair construction, scaffold composition, and OOD cleaning.
# ---------------------------------------------------------------------------
with open(os.path.join(DATA, "pairs_train.jsonl")) as f:
    pair_rows = [json.loads(line) for line in f if line.strip()]
assert len(pair_rows) == 5456
after_counts = collections.Counter(r["scaffold"] for r in pair_rows)
before_counts = [6310, 339, 222, 216, 211, 191, 177, 171]
scaffolds = [
    "c1ccccc1",
    "c1ccncc1",
    "c1cncnc1",
    "c1ccc(Nc2ccccc2)cc1",
    "O=C(Nc1ccccc1)NS(=O)(=O)c1ccccc1",
    "c1cnc2ncncc2n1",
    "O=S(=O)(Nc1ccccc1)c1ccccc1",
    "c1ccc2[nH]cnc2c1",
]
before_share = np.asarray(before_counts) / 10822 * 100
after_share = np.asarray([after_counts[s] for s in scaffolds]) / 5456 * 100
assert before_counts[0] == 6310 and after_counts[scaffolds[0]] == 2234

removed = load_json("removed_tiny_molecules.json")
assert len(removed) == 33
removed_csv = os.path.join(TABLES, "tableS6_removed_tiny_molecules.csv")
with open(removed_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["original_index", "smiles", "gt_pKa"])
    for row in removed:
        writer.writerow([row["idx"], row["smiles"], row["gt"]])
print(f"saved {removed_csv}")

scaffold_csv = os.path.join(TABLES, "tableS6_scaffold_distribution.csv")
with open(scaffold_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["id", "scaffold_smiles", "before_count", "before_share",
                     "after_count", "after_share"])
    for i, (scaffold, before, after) in enumerate(zip(scaffolds, before_counts, after_share), 1):
        writer.writerow([f"S{i}", scaffold, before, before / 10822,
                         after_counts[scaffold], after_counts[scaffold] / 5456])
print(f"saved {scaffold_csv}")

fig = plt.figure(figsize=(16.2, 7.0))
gs = fig.add_gridspec(2, 3, height_ratios=[5.0, 1.15],
                      width_ratios=[1.0, 1.9, 0.82], wspace=0.30, hspace=0.16,
                      left=0.04, right=0.99, top=0.90, bottom=0.04)
ax = fig.add_subplot(gs[0, 0])
ax.axis("off")
flow_boxes = [
    (0.80, "Exact similar pairs\n10,822 total\ntrain 9,693 | test 1,129"),
    (0.48, "Filtered directed pair library\n5,456 records\n1.5 ≤ |ΔpKa| ≤ 4; Tani ≥ 0.5"),
    (0.16, "Goal-conditioned split\ntrain 5,184 | validation 272"),
]
for y, text in flow_boxes:
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.55", fc="#F4F7FA", ec=BLUE, lw=1.0))
for y1, y2 in [(0.70, 0.59), (0.38, 0.27)]:
    ax.annotate("", xy=(0.5, y2), xytext=(0.5, y1), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=1.2, color="0.30"))
ax.set_title("Pair construction and split")
panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
y = np.arange(len(scaffolds))
height = 0.36
ax.barh(y - height / 2, before_share, height, color=SKY, label="Before filtering (n = 10,822)")
ax.barh(y + height / 2, after_share, height, color=VERMILION, label="After filtering (n = 5,456)")
short_labels = ["S1  Benzene", "S2  Pyridine", "S3  Pyrimidine"] + [f"S{i}" for i in range(4, 9)]
ax.set_yticks(y, short_labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Share of pair records (%)")
ax.set_title("Top scaffold contributions before and after filtering")
ax.legend(frameon=False, loc="lower right", fontsize=7.5)
ax.grid(axis="x", alpha=0.17, lw=0.5)
ax.text(0.97, 0.80, "Benzene scaffold\n58.3% → 40.9%", transform=ax.transAxes,
        ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
panel_label(ax, "b")

ax = fig.add_subplot(gs[0, 2])
bars = ax.bar([0, 1], [1135, 1102], color=[GRAY, GREEN], width=0.58)
ax.set_xticks([0, 1], ["Before cleaning", "After cleaning"])
ax.set_ylim(0, 1330)
ax.set_ylabel("OOD molecules")
ax.set_title("Tiny-molecule cleaning")
for bar, value in zip(bars, [1135, 1102]):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 25, str(value), ha="center", fontsize=10)
ax.annotate("33 removed", xy=(1, 1110), xytext=(0.48, 1235),
            arrowprops=dict(arrowstyle="->", color=VERMILION, lw=1.1),
            color=VERMILION, fontsize=10, fontweight="bold", ha="center")
ax.text(0.5, 0.43, "Cleaned split\nmild = 810\ncross = 292", transform=ax.transAxes,
        ha="center", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax.grid(axis="y", alpha=0.17, lw=0.5)
panel_label(ax, "c")

ax_key = fig.add_subplot(gs[1, :])
ax_key.axis("off")
ax_key.text(0.5, 0.94, "Scaffold SMILES key", ha="center", va="top",
            fontsize=8.5, fontweight="bold")
for i, scaffold in enumerate(scaffolds):
    column = 0 if i < 4 else 1
    row = i if i < 4 else i - 4
    ax_key.text(0.04 + column * 0.50, 0.72 - row * 0.23, f"S{i + 1}  {scaffold}",
                ha="left", va="center", fontsize=7.8, fontfamily="DejaVu Sans Mono")
fig.suptitle("Fig. S6 | Pair-bank composition and data-cleaning audit", y=0.985, fontsize=11)
save_figure(fig, "figS6_pair_and_cleaning_statistics")
plt.close(fig)


# ---------------------------------------------------------------------------
# Fig. S7: goal-conditioned cheat guardrail and training pKa loss.
# ---------------------------------------------------------------------------
with open(os.path.join(DATA, "goal_cond_1647561.log")) as f:
    log_text = f.read()
step_rows = re.findall(r"\[step (\d+)\].*?pka=([0-9.]+)", log_text)
steps = np.asarray([int(s) for s, _ in step_rows])
pka_loss = np.asarray([float(v) for _, v in step_rows])
assert steps.tolist() == [50, 100, 150, 200, 250, 300]
assert np.allclose(pka_loss, [18.441, 10.569, 5.773, 4.548, 4.355, 2.129])
cheat_steps = np.asarray([0, 300])
cheat_mae = np.asarray([5.321, 1.084])
val_sign = [0.0, 1.0]

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
ax = axes[0]
ax.axhspan(0, 1.5, color=VERMILION, alpha=0.08)
ax.axhline(1.5, color=VERMILION, ls="--", lw=1.1, label="Cheat threshold = 1.5")
ax.plot(cheat_steps, cheat_mae, color=VERMILION, lw=1.2, ls=":", marker="o", ms=8,
        markerfacecolor="white", markeredgewidth=1.6, label="Guardrail probe (2 evaluations)")
for xval, yval, sign in zip(cheat_steps, cheat_mae, val_sign):
    ax.annotate(f"{yval:.3f}\nval sign = {sign:.3f}", (xval, yval),
                xytext=(10 if xval == 0 else -12, 12 if xval == 0 else 22),
                textcoords="offset points", ha="left" if xval == 0 else "right", fontsize=8.5)
ax.annotate("STOP: cheat detected", xy=(300, 1.084), xytext=(175, 2.35),
            arrowprops=dict(arrowstyle="->", color=VERMILION, lw=1.0),
            color=VERMILION, fontsize=9, fontweight="bold")
ax.set_xlim(-15, 325)
ax.set_ylim(0, 6.15)
ax.set_xlabel("Training step")
ax.set_ylabel("Cheater probe MAE")
ax.set_title("Target-copying probe crosses the guardrail")
ax.legend(frameon=False, loc="upper right", fontsize=7.5)
ax.grid(alpha=0.16, lw=0.5)
panel_label(ax, "a")

ax = axes[1]
ax.plot(steps, pka_loss, color=BLUE, marker="o", ms=5, lw=1.5)
for xval, yval in zip(steps, pka_loss):
    ax.text(xval, yval + 0.55, f"{yval:.3f}", ha="center", fontsize=7.5)
ax.axvline(300, color=VERMILION, ls="--", lw=1.0)
ax.set_xlim(35, 315)
ax.set_ylim(0, 20.7)
ax.set_xlabel("Training step")
ax.set_ylabel(r"Training p$K_a$ loss")
ax.set_title("Optimization improves the exploitable objective")
ax.text(0.97, 0.95, "Anchor MAE at step 300 = 4.363\nGuardrail stopped training at step 300",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax.grid(alpha=0.16, lw=0.5)
panel_label(ax, "b")
fig.text(0.5, -0.01,
         "Only the pre-epoch and step-300 guardrail evaluations were logged; the dotted line does not represent intermediate measurements.",
         ha="center", fontsize=8.3)
fig.suptitle("Fig. S7 | Goal-conditioned regression exhibits target copying within 300 steps", y=1.03, fontsize=11)
fig.tight_layout(w_pad=2.5)
save_figure(fig, "figS7_goal_conditioned_cheat_detection")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig. S8: cross-fitted shared-direction projection versus existing G4.
# ---------------------------------------------------------------------------
projection = load_json("projection_diagnosis.json")
projection_rows = projection["per_pair"]
proj = np.asarray([r["projection_leave_scaffold_out"] for r in projection_rows])
head = np.asarray([r["g4_low_to_high_cosine"] for r in projection_rows])
abs_delta = np.asarray([r["abs_delta_pka"] for r in projection_rows])
proj_masks = [
    (abs_delta >= 1) & (abs_delta < 2),
    (abs_delta >= 2) & (abs_delta < 4),
    abs_delta >= 4,
]
proj_labels = [r"$1 \leq |\Delta\mathrm{pKa}| < 2$", r"$2 \leq |\Delta\mathrm{pKa}| < 4$", r"$|\Delta\mathrm{pKa}| \geq 4$"]
proj_colors = [BLUE, ORANGE, VERMILION]
null_info = projection["signed_orientation_null"]
null_distribution = np.asarray(null_info["null_mean_distribution"])
assert len(proj) == 43 and len(null_distribution) == 2000

fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.7),
                         gridspec_kw={"width_ratios": [1.03, 1.0, 0.92]})
ax = axes[0]
for i, (mask, color) in enumerate(zip(proj_masks, proj_colors)):
    violin_jitter(ax, proj[mask], i, color, point_size=25)
    ax.text(i, 0.91, f"n = {mask.sum()}; median = {np.median(proj[mask]):.3f}",
            ha="center", va="top", fontsize=8)
ax.axhline(0, color="0.25", lw=0.8)
ax.axhline(0.2, color=ORANGE, lw=1.0, ls=":")
ax.axhline(0.5, color=GREEN, lw=1.0, ls="--")
ax.text(2.42, 0.5, "0.5", color=GREEN, fontsize=7.5, va="center")
ax.text(2.42, 0.2, "0.2", color=ORANGE, fontsize=7.5, va="center")
ax.set_xticks(range(3), proj_labels)
ax.set_xlim(-0.55, 2.55)
ax.set_ylim(-1.0, 1.02)
ax.set_ylabel("Cross-fitted projection of unit displacement")
ax.set_title(r"Shared-direction projection by p$K_a$ gap")
trend = projection["delta_gap_trend"]
ax.text(0.03, 0.03,
        f"Exploratory Spearman $\\rho$ = {trend['spearman_rho']:.3f}\n"
        f"Two-sided permutation p = {trend['two_sided_permutation_p']:.3f}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.5, alpha=0.9))
ax.grid(axis="y", alpha=0.16, lw=0.5)
panel_label(ax, "a")

ax = axes[1]
for mask, label, color in zip(proj_masks, ["Small Δ", "Mid Δ", "Large Δ"], proj_colors):
    ax.scatter(head[mask], proj[mask], s=40, color=color, alpha=0.82,
               edgecolor="white", linewidth=0.45, label=f"{label} (n = {mask.sum()})")
lim = 0.92
ax.plot([-lim, lim], [-lim, lim], color="0.45", lw=0.9, ls="--", label="Equal alignment")
ax.axhline(0, color="0.25", lw=0.7)
ax.axvline(0, color="0.25", lw=0.7)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_xlabel(r"Existing head gradient cosine (low → high p$K_a$)")
ax.set_ylabel("Shared-direction projection")
ax.set_title("Projection and head alignment are not interchangeable")
ax.legend(frameon=False, loc="lower left", fontsize=7.2)
ax.grid(alpha=0.13, lw=0.5)
panel_label(ax, "b")

ax = axes[2]
null_ci = null_info["null_mean_95pct_interval"]
ax.hist(null_distribution, bins=42, color=GRAY, alpha=0.62, edgecolor="none")
ax.axvspan(null_ci[0], null_ci[1], color=GRAY, alpha=0.12, label="Null 95% interval")
ax.axvline(null_info["observed_mean_projection"], color=VERMILION, lw=1.8,
           label=f"Observed = {null_info['observed_mean_projection']:.3f}")
ax.set_xlabel("Mean projection across 43 pairs")
ax.set_ylabel("Frequency")
ax.set_title("Shared orientation exceeds the sign-flip null")
ax.text(0.97, 0.95, f"One-sided empirical\np = {null_info['one_sided_empirical_p']:.2e}",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
ax.grid(axis="y", alpha=0.16, lw=0.5)
panel_label(ax, "c")
fig.text(0.5, -0.015,
         "Exploratory cross-fitting on 43 test pairs with Murcko-scaffold exclusion; not independent train-to-test validation. "
         "The trend permutation is pair-level and does not model dependence from shared molecules or cross-fitting.",
         ha="center", fontsize=8.3)
fig.suptitle("Fig. S8 | Shared pKa displacement directions versus existing head gradients", y=1.03, fontsize=11)
fig.tight_layout(w_pad=2.5)
save_figure(fig, "figS8_projection_diagnostic")
plt.close(fig)

print("All remaining main-text and SI assets were generated successfully.")
