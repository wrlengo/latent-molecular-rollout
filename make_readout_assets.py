"""Build the frozen-h readout figure and tables from verified remote outputs."""
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
os.makedirs(FIGS, exist_ok=True)
os.makedirs(TABLES, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.bbox": "tight",
})

def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


result = json.load(open(os.path.join(DATA, "frozen_ridge_results.json")))
with open(os.path.join(DATA, "frozen_ridge_test_pairs.csv")) as f:
    pairs = list(csv.DictReader(f))
assert len(pairs) == result["counts"]["test_pairs"] == 43
# Fail loudly if the latent and readout artifacts come from different runs.
latents = np.load(os.path.join(DATA, "h_test.npz"), allow_pickle=True)
weights = np.load(os.path.join(DATA, "ridge_selected_weights.npz"))
h = latents["h"].astype(np.float64)
y = latents["gt"].astype(np.float64)
w = weights["w_original"].astype(np.float64)
b = float(weights["intercept_original"])
pred = h @ w + b
assert abs(np.mean(np.abs(pred - y)) - result["mae"]["test"]) < 1e-6
for row in pairs:
    displacement = h[int(row["high"])] - h[int(row["low"])]
    cosine = np.dot(w, displacement) / (np.linalg.norm(w) * np.linalg.norm(displacement))
    assert abs(cosine - float(row["ridge_cosine"])) < 1e-10
weight_energy = np.sort(w ** 2)[::-1]
top2_weight_energy = float(weight_energy[:2].sum() / weight_energy.sum())
whitened = json.load(open(os.path.join(DATA, "frozen_ridge_whitened_cosine.json")))
for name in ("h_test.npz", "ridge_selected_weights.npz", "frozen_ridge_results.json",
             "frozen_ridge_test_pairs.csv", "projection_diagnosis.json"):
    assert sha256(os.path.join(DATA, name)) == whitened["artifact_sha256"][name]
whitened_weights = np.load(os.path.join(DATA, "ridge_whitened_weights.npz"))
w_z = whitened_weights["w_z"].astype(np.float64)
scale = whitened_weights["fit_scale"].astype(np.float64)
ridge_z_cos = []
for row in pairs:
    displacement = (h[int(row["high"])] - h[int(row["low"])]) / scale
    cosine = np.dot(w_z, displacement) / (np.linalg.norm(w_z) * np.linalg.norm(displacement))
    ridge_z_cos.append(cosine)
ridge_z_cos = np.asarray(ridge_z_cos)
assert abs(np.median(ridge_z_cos) - whitened["whitened_cosine"]["test"]["median"]) < 1e-12
head_cos = np.asarray([float(row["pka_head_gradient_cosine"]) for row in pairs])
ridge_cos = np.asarray([float(row["ridge_cosine"]) for row in pairs])
alpha_rows = result["alpha_validation"]
alphas = np.asarray([row["alpha"] for row in alpha_rows])
fit_curve = np.asarray([row["fit_mae"] for row in alpha_rows])
val_curve = np.asarray([row["validation_mae"] for row in alpha_rows])
selected_alpha = result["protocol"]["selected_alpha"]
assert alphas[np.argmin(val_curve)] == selected_alpha

fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0))
blue, orange, green, gray = "#0072B2", "#E69F00", "#009E73", "#999999"

ax = axes[0]
ax.semilogx(alphas, fit_curve, "o-", ms=4, lw=1.4, color=blue, label="Fit MAE")
ax.semilogx(alphas, val_curve, "o-", ms=4, lw=1.4, color=orange, label="Validation MAE")
ax.axvline(selected_alpha, color=green, ls="--", lw=1.2,
           label=f"Selected $\\alpha$ = {selected_alpha:g}")
ax.set_xlabel("Ridge regularization $\\alpha$")
ax.set_ylabel(r"MAE (p$K_a$ units)")
ax.set_title("Model selection uses validation only")
ax.grid(alpha=0.18, lw=0.5)
ax.legend(frameon=False, loc="best", fontsize=7.5)
ax.text(-0.17, 1.04, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")

ax = axes[1]
mae = result["mae"]
labels = ["Ridge\nfit", "Ridge\nvalidation", "Ridge\ntest", "Existing head\ntest"]
values = [mae["fit"], mae["validation"], mae["test"], mae["existing_pka_head_test"]]
colors = [blue, orange, green, gray]
bars = ax.bar(np.arange(4), values, color=colors, width=0.66)
ax.axhline(2.5, color="#D55E00", ls=":", lw=1.2, label="Prespecified MAE = 2.5")
ax.set_xticks(np.arange(4), labels)
ax.set_ylabel(r"MAE (p$K_a$ units; lower is better)")
ax.set_title("Regression error at the selected $\\alpha$")
ax.grid(axis="y", alpha=0.18, lw=0.5)
ax.set_ylim(0, max(max(values) * 1.22, 3.0))
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.05, f"{value:.3f}",
            ha="center", va="bottom", fontsize=8)
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
ax.text(-0.17, 1.04, "b", transform=ax.transAxes, fontsize=12, fontweight="bold")

ax = axes[2]
positions = [0, 1, 2, 3, 4]
fit_summary = result["ridge_low_to_high_cosine"]["fit"]
test_summary = result["ridge_low_to_high_cosine"]["test"]
white_fit_summary = whitened["whitened_cosine"]["fit"]
white_test_summary = whitened["whitened_cosine"]["test"]
rng = np.random.default_rng(42)
for values, position, color in ((ridge_cos, 1, blue), (ridge_z_cos, 3, green), (head_cos, 4, gray)):
    jitter = rng.uniform(-0.12, 0.12, len(values))
    ax.scatter(np.full(len(values), position) + jitter, values, s=12, color=color,
               alpha=0.68, edgecolors="white", linewidths=0.25, zorder=3)
for position, summary, color in ((0, fit_summary, blue), (1, test_summary, blue),
                                 (2, white_fit_summary, orange), (3, white_test_summary, green)):
    ci = summary["median_bootstrap_95ci"]
    ax.errorbar([position], [summary["median"]],
                yerr=[[summary["median"] - ci[0]], [ci[1] - summary["median"]]],
                fmt="s", ms=5, color=color, ecolor=color, elinewidth=1.2, capsize=3, zorder=4)
ax.axhline(0, color="0.25", lw=0.8)
ax.axhline(0.2, color=green, ls="--", lw=1.0, label="Sensitivity threshold = 0.2")
ax.axhline(0.05, color="#D55E00", ls=":", lw=1.0, label="Low-cosine threshold = 0.05")
ax.set_xticks(positions, [f"Original h\nfit\n(n = {fit_summary['n']})", "Original h\ntest\n(n = 43)",
                          f"Z-scored\nfit\n(n = {white_fit_summary['n']})", "Z-scored\ntest\n(n = 43)",
                          "Head\noriginal h\ntest"])
ax.tick_params(axis="x", labelsize=6.8)
ax.set_ylabel(r"Low-to-high p$K_a$ direction cosine")
ax.set_title("Scaling sensitivity remains near zero")
ax.grid(axis="y", alpha=0.15, lw=0.5)
ax.legend(frameon=False, loc="upper left", fontsize=6.8)
ax.text(-0.17, 1.04, "c", transform=ax.transAxes, fontsize=12, fontweight="bold")

fig.suptitle("Fig. S9 | Frozen-h ridge readout and cosine scaling sensitivity",
             fontsize=12, y=1.02)
counts = result["counts"]
fig.text(0.5, -0.015,
         f"Original train n = {counts['original_train']}; test-scaffold/exact-SMILES cleaning removed "
         f"n = {counts['removed_train_test_overlap']}; fit n = {counts['fit']}, validation n = {counts['validation']}.\n"
         "The frozen backbone was originally trained on the full training split; the 43 test pairs informed earlier exploratory diagnostics.\n"
         f"Ridge alpha = {selected_alpha:g} was selected by validation MAE; direction alignment was not a selection criterion.\n"
         f"Two coordinates carry {top2_weight_energy:.2%} of the original-h weight energy; the original-h cosine is scale-sensitive.\n"
         f"Z-scored test median = {white_test_summary['median']:.5f} "
         f"(95% CI {white_test_summary['median_bootstrap_95ci'][0]:.5f} to {white_test_summary['median_bootstrap_95ci'][1]:.5f}); "
         "whitening changes the geometry and does not establish an original-h conclusion.\n"
         "Error bars are pair-level bootstrap 95% CIs and do not model pair dependence.",
         ha="center", va="top", fontsize=7.6)
fig.tight_layout(w_pad=2.2)
for ext in ("pdf", "png"):
    path = os.path.join(FIGS, f"figS9_frozen_readout.{ext}")
    fig.savefig(path, dpi=300)
    print(f"saved {path}")
plt.close(fig)

summary_rows = [
    ["selected_alpha", selected_alpha],
    ["fit_mae", mae["fit"]],
    ["validation_mae", mae["validation"]],
    ["test_mae", mae["test"]],
    ["existing_pka_head_test_mae", mae["existing_pka_head_test"]],
    ["original_h_fit_cosine_median", fit_summary["median"]],
    ["original_h_test_cosine_median", test_summary["median"]],
    ["original_h_test_cosine_mean", test_summary["mean"]],
    ["whitened_fit_cosine_median", white_fit_summary["median"]],
    ["whitened_test_cosine_median", white_test_summary["median"]],
    ["whitened_test_cosine_mean", white_test_summary["mean"]],
    ["whitened_test_cosine_ci_low", white_test_summary["median_bootstrap_95ci"][0]],
    ["whitened_test_cosine_ci_high", white_test_summary["median_bootstrap_95ci"][1]],
    ["existing_head_original_h_test_cosine_median", result["existing_pka_head_low_to_high_cosine_test"]["median"]],
    ["top2_original_h_weight_energy_fraction", top2_weight_energy],
    ["original_h_decision", whitened["decision"]["original_h_conclusion"]],
    ["whitened_sensitivity_decision", whitened["decision"]["category"]],
]
path = os.path.join(TABLES, "tableS9_frozen_readout.csv")
with open(path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "value"])
    writer.writerows(summary_rows)
print(f"saved {path}")

path = os.path.join(TABLES, "tableS9_frozen_ridge_alpha_validation.csv")
with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(alpha_rows[0]))
    writer.writeheader()
    writer.writerows(alpha_rows)
print(f"saved {path}")

audit = json.load(open(os.path.join(DATA, "readout_split_audit.json")))
path = os.path.join(TABLES, "tableS9_actual_split_audit.csv")
with open(path, "w", newline="") as f:
    fields = ["split_pair", "n_left", "n_right", "n_shared_nonempty_scaffolds",
              "n_left_rows_with_shared_scaffold", "n_right_rows_with_shared_scaffold",
              "n_shared_exact_canonical_smiles"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for split_pair, row in audit["pairwise"].items():
        writer.writerow({"split_pair": split_pair, **{key: row[key] for key in fields[1:]}})
print(f"saved {path}")
