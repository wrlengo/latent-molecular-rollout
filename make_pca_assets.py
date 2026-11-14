"""Build Fig. S10 and Table S10 from the hash-verified PCA direction analysis."""
import csv
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
FIGS = os.path.join(ROOT, "figs")
TABLES = os.path.join(ROOT, "tables")


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


result = json.load(open(os.path.join(DATA, "pca_direction_results.json")))
fit = np.load(os.path.join(DATA, "pca_direction_fit.npz"))
assert str(fit["train_latent_sha256"]) == result["artifact_sha256"]["h_train.npz"]
assert str(fit["test_latent_sha256"]) == result["artifact_sha256"]["h_test.npz"]
for name in ("h_test.npz", "ridge_selected_weights.npz", "frozen_ridge_test_pairs.csv",
             "projection_diagnosis.json", "gate4_stratified.json"):
    assert sha256(os.path.join(DATA, name)) == result["artifact_sha256"][name]
with open(os.path.join(DATA, "pca_direction_test_pairs.csv")) as f:
    pair_rows = list(csv.DictReader(f))
assert len(pair_rows) == 43
h = np.load(os.path.join(DATA, "h_test.npz"), allow_pickle=True)["h"].astype(np.float64)
components = fit["pca90_components"].astype(np.float64)
assert components.shape == (result["pca"]["k90"], 4096)
assert np.max(np.abs(components @ components.T - np.eye(len(components)))) < 1e-9
norm_fraction, full_cosine, top90_cosine = [], [], []
u = fit["mean_fit_direction"].astype(np.float64)
u90 = fit["mean_fit_direction_top90"].astype(np.float64)
for row in pair_rows:
    d = h[int(row["high"])] - h[int(row["low"])]
    d90 = (d @ components.T) @ components
    norm_fraction.append(np.linalg.norm(d90) / np.linalg.norm(d))
    full_cosine.append(np.dot(d, u) / (np.linalg.norm(d) * np.linalg.norm(u)))
    top90_cosine.append(np.dot(d, u90) / (np.linalg.norm(d) * np.linalg.norm(u90)))
norm_fraction = np.asarray(norm_fraction)
full_cosine = np.asarray(full_cosine)
top90_cosine = np.asarray(top90_cosine)
assert np.max(np.abs(norm_fraction - np.asarray([float(r["top90_norm_fraction"]) for r in pair_rows]))) < 1e-12
assert np.max(np.abs(full_cosine - np.asarray([float(r["mean_fit_direction_full_cosine"]) for r in pair_rows]))) < 1e-12
assert np.max(np.abs(top90_cosine - np.asarray([float(r["mean_fit_direction_top90_cosine"]) for r in pair_rows]))) < 1e-12

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight"})
blue, orange, green, gray = "#0072B2", "#E69F00", "#009E73", "#999999"
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))

ax = axes[0]
cumulative = fit["cumulative_variance_ratio"]
x = np.arange(1, len(cumulative) + 1)
ax.semilogx(x, cumulative, color=blue, lw=1.5)
k90 = result["pca"]["k90"]
ax.axhline(0.9, color=gray, ls="--", lw=1.0)
ax.axvline(k90, color=green, ls="--", lw=1.2, label=f"k = {k90}: {cumulative[k90 - 1]:.3f}")
ax.scatter([1, k90], [cumulative[0], cumulative[k90 - 1]], s=28, color=[orange, green], zorder=3)
ax.set_xlim(1, len(cumulative))
ax.set_ylim(0, 1.02)
ax.set_xlabel("Number of raw-h principal components (log scale)")
ax.set_ylabel("Cumulative explained variance")
ax.set_title("Latent variance is strongly concentrated")
ax.grid(alpha=0.16, lw=0.5)
ax.legend(frameon=False, loc="lower right")
ax.text(-0.17, 1.04, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")

ax = axes[1]
fit_norm = result["top90_displacement_projection"]["fit_norm_fraction"]
ci = fit_norm["median_bootstrap_95ci"]
ax.errorbar([0], [fit_norm["median"]], yerr=[[fit_norm["median"] - ci[0]], [ci[1] - fit_norm["median"]]],
            fmt="s", ms=6, color=blue, ecolor=blue, elinewidth=1.3, capsize=4)
parts = ax.violinplot([norm_fraction], positions=[1], widths=0.65,
                      showmeans=False, showmedians=False, showextrema=False)
for body in parts["bodies"]:
    body.set_facecolor(green)
    body.set_alpha(0.22)
    body.set_edgecolor(green)
rng = np.random.default_rng(42)
ax.scatter(1 + rng.uniform(-0.11, 0.11, len(norm_fraction)), norm_fraction, s=13,
           color=green, alpha=0.72, edgecolors="white", linewidths=0.25)
ax.hlines(np.median(norm_fraction), 0.75, 1.25, color="black", lw=1.4)
ax.axhline(0.2, color="#D55E00", ls=":", lw=1.1, label="Low-coverage threshold = 0.2")
ax.set_xticks([0, 1], [f"Fit pairs\n(n = {fit_norm['n']})", f"Test pairs\n(n = {len(norm_fraction)})"])
ax.set_ylim(0, 1.03)
ax.set_ylabel(r"$||P_{90}d||_2 / ||d||_2$")
ax.set_title("Displacements occupy the main subspace")
ax.grid(axis="y", alpha=0.16, lw=0.5)
ax.legend(frameon=False, loc="lower left", fontsize=7.2)
ax.text(-0.17, 1.04, "b", transform=ax.transAxes, fontsize=12, fontweight="bold")

ax = axes[2]
direction = result["mean_fit_direction"]
fit_full = direction["fit_full_cosine"]
fit_top = direction["fit_top90_cosine"]
for position, summary, color in ((0, fit_full, blue), (2, fit_top, orange)):
    ci = summary["median_bootstrap_95ci"]
    ax.errorbar([position], [summary["median"]],
                yerr=[[summary["median"] - ci[0]], [ci[1] - summary["median"]]],
                fmt="s", ms=5, color=color, ecolor=color, elinewidth=1.2, capsize=3)
for values, position, color in ((full_cosine, 1, blue), (top90_cosine, 3, orange)):
    ax.scatter(position + rng.uniform(-0.11, 0.11, len(values)), values, s=12,
               color=color, alpha=0.68, edgecolors="white", linewidths=0.25)
    ax.hlines(np.median(values), position - 0.24, position + 0.24, color="black", lw=1.3)
ax.axhline(0, color="0.25", lw=0.8)
ax.axhline(0.2, color=green, ls="--", lw=1.0, label="Cosine = 0.2")
ax.set_xticks([0, 1, 2, 3], ["Fit\nfull", "Test\nfull", "Fit\ntop-90%", "Test\ntop-90%"])
ax.set_ylabel("Cosine to mean fit direction")
ax.set_title("Weak direction persists after projection")
ax.grid(axis="y", alpha=0.16, lw=0.5)
ax.legend(frameon=False, loc="upper right", fontsize=7.2)
ax.text(-0.17, 1.04, "c", transform=ax.transAxes, fontsize=12, fontweight="bold")

fig.suptitle("Fig. S10 | Readout-free PCA diagnosis of pKa displacement geometry", fontsize=12, y=1.02)
fig.text(0.5, -0.015,
         f"Raw-h PCA uses only the 2,855 fit rows; the first {k90} PCs explain {cumulative[k90 - 1]:.2%} of variance. "
         "Neither test latents nor test labels inform PCA or the mean fit direction.\n"
         "Panel b measures subspace coverage, not directional correctness. Panel c uses 381 fit pairs to construct one mean direction; "
         "test median CIs cross zero, so transfer is weak and exploratory.\n"
         "Error bars and CIs are pair-level bootstrap 95% intervals and do not model pair dependence.",
         ha="center", va="top", fontsize=7.6)
fig.tight_layout(w_pad=2.2)
for ext in ("pdf", "png"):
    path = os.path.join(FIGS, f"figS10_pca_direction.{ext}")
    fig.savefig(path, dpi=300)
    print(f"saved {path}")
plt.close(fig)

projection = result["top90_displacement_projection"]
rows = [
    ["k90", k90], ["variance_captured_at_k90", result["pca"]["variance_captured_at_k90"]],
    ["top1_explained_variance_ratio", result["pca"]["top1_explained_variance_ratio"]],
    ["fit_top90_norm_fraction_median", projection["fit_norm_fraction"]["median"]],
    ["test_top90_norm_fraction_median", projection["test_norm_fraction"]["median"]],
    ["test_top90_norm_fraction_ci_low", projection["test_norm_fraction"]["median_bootstrap_95ci"][0]],
    ["test_top90_norm_fraction_ci_high", projection["test_norm_fraction"]["median_bootstrap_95ci"][1]],
    ["test_top90_energy_fraction_median", projection["test_energy_fraction"]["median"]],
    ["mean_direction_top90_norm_fraction", direction["top90_norm_fraction_of_unit_direction"]],
    ["test_full_cosine_median", direction["test_full_cosine"]["median"]],
    ["test_full_cosine_ci_low", direction["test_full_cosine"]["median_bootstrap_95ci"][0]],
    ["test_full_cosine_ci_high", direction["test_full_cosine"]["median_bootstrap_95ci"][1]],
    ["test_top90_cosine_median", direction["test_top90_cosine"]["median"]],
    ["test_top90_cosine_ci_low", direction["test_top90_cosine"]["median_bootstrap_95ci"][0]],
    ["test_top90_cosine_ci_high", direction["test_top90_cosine"]["median_bootstrap_95ci"][1]],
    ["subspace_decision", projection["decision"]],
]
path = os.path.join(TABLES, "tableS10_pca_direction.csv")
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerows(rows)
print(f"saved {path}")
