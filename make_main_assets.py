"""Build main-text assets and the pre-repair shortcut-diagnosis SI assets.

Scatter inputs are the held-out test predictions (n=1151 per group). Generation
metrics are from the rescored validation outputs and bon_gen2f_val_n8.json.
The shortcut probe has a separate 24-row cohort and an aggregate-only conflict
record; it is never merged into the five-group regression comparison.
"""
import csv
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


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_figure(fig, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(FIGS, f"{stem}.{ext}")
        fig.savefig(path, dpi=300)
        print(f"saved {path}")


def panel_label(ax, label):
    ax.text(-0.17, 1.04, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def write_csv(name, header, rows):
    path = os.path.join(TABLES, name + ".csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"saved {path}")


def write_tex(name, content):
    path = os.path.join(TABLES, name + ".tex")
    with open(path, "w") as f:
        f.write(content)
    print(f"saved {path}")


# ---------------------------------------------------------------------------
# Fig. 1: five held-out scatter panels + a cross-group MAE summary.
# ---------------------------------------------------------------------------
groups = ["A", "B", "Bp", "Bpp", "C"]
files = {
    "A": "eval_A_fix_test.json",
    "B": "eval_B_fix_test.json",
    "Bp": "eval_Bp_fix_test.json",
    "Bpp": "eval_Bpp_fix_test.json",
    "C": "eval_C_fix_test.json",
}
labels = {
    "A": "A: Text + CoT",
    "B": "B: Text only",
    "Bp": "B′: Molecule + head",
    "Bpp": "B″: Molecule + random slots + head",
    "C": "C: Molecule + latent + head",
}
colors = {"A": "#0072B2", "B": "#E69F00", "Bp": "#009E73",
          "Bpp": "#56B4E9", "C": "#D55E00"}
preds = {}
for group in groups:
    path = os.path.join(DATA, files[group])
    if group in ("A", "B"):
        rows = json.load(open(path))["results"]
        gt = np.array([r["gt"] for r in rows], dtype=float)
        pred = np.array([r["pred"] for r in rows], dtype=float)
    else:
        rows = load_jsonl(path)
        gt = np.array([r["gt"] for r in rows], dtype=float)
        pred = np.array([r["p0"] for r in rows], dtype=float)
    assert len(gt) == 1151, (group, len(gt))
    preds[group] = (gt, pred)
mae = {g: float(np.mean(np.abs(preds[g][1] - preds[g][0]))) for g in groups}
expected = {"A": 2.895, "B": 4.028, "Bp": 2.363, "Bpp": 2.343, "C": 2.087}
for g in groups:
    assert abs(mae[g] - expected[g]) < 0.002, (g, mae[g])

# ---------------------------------------------------------------------------
# Fig. S0 / Table S0: pre-repair shortcut diagnosis on its own cohorts.
# ---------------------------------------------------------------------------
pre_repair = json.load(open(os.path.join(DATA, "fig3_causal.json")))
pre_gt = np.asarray(pre_repair["gt"], dtype=float)
pre_ref = np.asarray(pre_repair["ref_head"], dtype=float)
pre_h = np.asarray(pre_repair["h_head"], dtype=float)
assert len(pre_gt) == len(pre_ref) == len(pre_h) == 24
assert pre_repair["n_latent"] == 1
pre_ref_mae = float(np.mean(np.abs(pre_ref - pre_gt)))
pre_h_mae = float(np.mean(np.abs(pre_h - pre_gt)))
assert abs(pre_ref_mae - 1.103411411246) < 1e-9
assert abs(pre_h_mae - 1.101539433002) < 1e-9

# The conflict experiment survives only as this aggregate table. Parse the
# archived rows rather than inventing row-level predictions or intervals.
diagnosis_path = os.path.join(ROOT, "DIAGNOSIS.md")
with open(diagnosis_path) as f:
    diagnosis_text = f.read()
conflict_labels = {
    "base": "base",
    "text→donor, mol→source": "text_donor",
    "mol→donor, text→source": "molecule_donor",
}
conflict = {}
for line in diagnosis_text.splitlines():
    cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
    if len(cells) == 3 and cells[0] in conflict_labels:
        conflict[conflict_labels[cells[0]]] = {
            "mae_vs_source": float(cells[1]),
            "mae_vs_donor": float(cells[2]),
        }
assert conflict == {
    "base": {"mae_vs_source": 1.17, "mae_vs_donor": 4.06},
    "text_donor": {"mae_vs_source": 4.08, "mae_vs_donor": 1.20},
    "molecule_donor": {"mae_vs_source": 1.17, "mae_vs_donor": 4.07},
}, conflict
token_probe = json.load(open(os.path.join(DATA, "probe_tokenfix.json")))
assert token_probe["K"] == 6 and all(n == 6 for n in token_probe["slot_counts"])
assert token_probe["dl_ids"] == token_probe["model_ids"]

shortcut_summary = {
    "pre_repair_c": {
        "source": "data/fig3_causal.json",
        "checkpoint": "v3_C_latent_head/stage3_sub3",
        "cohort": "24 held-out diagnostic probe rows; not the 1,151-row test set",
        "n": 24,
        "n_latent_recognized": pre_repair["n_latent"],
        "mae_ref_head": pre_ref_mae,
        "mae_h_head": pre_h_mae,
    },
    "repaired_c": {
        "source": "data/eval_C_fix_test.json",
        "cohort": "1,151 held-out test rows",
        "n": 1151,
        "mae": mae["C"],
    },
    "modality_conflict": {
        "source": "DIAGNOSIS.md",
        "evidence_status": "archived aggregate summary; per-row conflict predictions were not preserved",
        "n": 24,
        "conditions": conflict,
        "uncertainty_intervals_available": False,
        "output_invariance_count_available": False,
    },
    "repair_boundary": {
        "combined_repairs": [
            "remove SMILES from the pKa query",
            "align latent-token IDs and restore all six latent slots",
        ],
        "token_repair_source": "data/probe_tokenfix.json",
        "n_latent_recognized_after_repair": token_probe["K"],
        "repairs_separately_ablated": False,
        "pre_and_post_cohorts_match": False,
        "interpretation": "The numerical difference between 1.103 and 2.087 is not an isolated estimate of shortcut removal.",
    },
    "eval_text_truncation": {
        "evidence_status": "historical engineering note; no independent raw diagnostic artifact preserved",
        "scope": "text-path scoring only; not a PkaHead-based comparison",
    },
}
summary_path = os.path.join(DATA, "shortcut_diagnosis_summary.json")
with open(summary_path, "w") as f:
    json.dump(shortcut_summary, f, indent=2, ensure_ascii=False)
    f.write("\n")
print(f"saved {summary_path}")

fig, ax = plt.subplots(figsize=(7.2, 4.2))
condition_order = ["base", "text_donor", "molecule_donor"]
condition_names = ["Base", "Text → donor\nmolecule → source", "Features → donor\ntext → source"]
x = np.arange(len(condition_order))
width = 0.34
source_values = [conflict[k]["mae_vs_source"] for k in condition_order]
donor_values = [conflict[k]["mae_vs_donor"] for k in condition_order]
bars_source = ax.bar(x - width / 2, source_values, width, label="MAE vs source GT", color="#0072B2")
bars_donor = ax.bar(x + width / 2, donor_values, width, label="MAE vs donor GT", color="#E69F00")
ax.set_xticks(x, condition_names)
ax.set_ylim(0, 4.8)
ax.set_ylabel(r"MAE (p$K_a$ units)")
ax.set_title("Pre-repair modality-conflict probe (n = 24)")
ax.grid(axis="y", alpha=0.20, lw=0.5)
ax.legend(frameon=False, loc="upper left")
for bars in (bars_source, bars_donor):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.09,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
fig.text(0.99, 0.015, "Archived aggregates; row-level conflict predictions unavailable",
         ha="right", va="bottom", fontsize=7.4, color="0.35")
fig.tight_layout(rect=(0, 0.055, 1, 1))
save_figure(fig, "figS0_text_shortcut_conflict")
plt.close(fig)

shortcut_table_lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\caption{Pre-repair text-shortcut diagnosis. The C PkaHead value $1.103$ is recomputed from 24 held-out diagnostic rows; repaired C is evaluated on a different cohort of 1{,}151 test rows after two combined repairs. Modality-conflict values are archived aggregate MAEs ($n=24$), with source and donor ground truth (GT) scored separately. Per-row conflict predictions were not preserved, so no intervals or output-invariance counts are reported. The pre/post values are not a controlled ablation.}",
    r"\label{tab:shortcut}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{4pt}",
    r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.28\linewidth}>{\raggedright\arraybackslash}p{0.12\linewidth}>{\raggedright\arraybackslash}p{0.24\linewidth}rr@{}}",
    r"\toprule",
    r"Condition & Query SMILES & Cohort / evidence & Source MAE & Donor MAE \\",
    r"\midrule",
    fr"C pre-repair, \texttt{{PkaHead}} & present & 24-row probe; recomputed & {pre_ref_mae:.3f} & --- \\",
    fr"C repaired, combined repairs & removed & 1{{,}}151-row test; recomputed & {mae['C']:.3f} & --- \\",
    r"\midrule",
    fr"Conflict: base & source & 24-row archived aggregate & {conflict['base']['mae_vs_source']:.2f} & {conflict['base']['mae_vs_donor']:.2f} \\",
    fr"Conflict: text $\rightarrow$ donor; molecule $\rightarrow$ source & donor & 24-row archived aggregate & {conflict['text_donor']['mae_vs_source']:.2f} & {conflict['text_donor']['mae_vs_donor']:.2f} \\",
    fr"Conflict: features $\rightarrow$ donor; text $\rightarrow$ source & source & 24-row archived aggregate & {conflict['molecule_donor']['mae_vs_source']:.2f} & {conflict['molecule_donor']['mae_vs_donor']:.2f} \\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
    "",
]
write_tex("tableS0_text_shortcut", "\n".join(shortcut_table_lines))

fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5), sharex=False, sharey=False)
lo, hi = -3.0, 17.0
for idx, group in enumerate(groups):
    ax = axes[idx // 3, idx % 3]
    gt, pred = preds[group]
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color="0.25", zorder=1)
    ax.scatter(gt, pred, s=6, alpha=0.30, color=colors[group],
               edgecolors="none", rasterized=True, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(labels[group], pad=5)
    ax.text(0.04, 0.95, f"MAE = {mae[group]:.3f}\nn = {len(gt)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.75", lw=0.6))
    ax.grid(alpha=0.15, lw=0.5)
    ax.set_xlabel(r"Ground-truth p$K_a$")
    ax.set_ylabel(r"Predicted p$K_a$")
    panel_label(ax, chr(ord("a") + idx))

ax = axes[1, 2]
y = np.arange(len(groups))
ax.barh(y, [mae[g] for g in groups], color=[colors[g] for g in groups], height=0.64)
ax.set_yticks(y, [labels[g].split(":")[0] for g in groups])
ax.invert_yaxis()
ax.set_xlim(0, 4.8)
ax.set_xlabel(r"MAE (p$K_a$ units; lower is better)")
ax.set_title("Held-out error summary", pad=5)
ax.grid(axis="x", alpha=0.20, lw=0.5)
for yi, group in enumerate(groups):
    ax.text(mae[group] + 0.08, yi, f"{mae[group]:.3f}", va="center", fontsize=9)
ax.text(0.98, 0.06, "C vs B′: −0.276", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=9, color=colors["C"], fontweight="bold")
panel_label(ax, "f")
fig.tight_layout(w_pad=2.2, h_pad=2.0)
save_figure(fig, "fig1_pka_scatter_main")
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig. 4: paired single-sample gains, constant baseline, and oracle upper bound.
# ---------------------------------------------------------------------------
sft = json.load(open(os.path.join(DATA, "rejection_sft_results.json")))
base_single = sft["single_sample"]["gen2_final"]
new_single = sft["single_sample"]["oracle_best_sft"]
paired = sft["single_sample"]["paired_comparison"]
constant = sft["single_sample"]["train_selected_constant_all_255"]
base_all = constant["gen2_final"]
new_all = constant["oracle_best_sft"]
assert base_single["unique_gen"] == 227 and new_single["unique_gen"] == 241
assert paired["n_both_scored"] == 235 and constant["n"] == 255
assert constant["constant"] == 3.74 and new_all["n_missing_noop"] == 3

base_rows = {r["i"]: r for r in load_jsonl(os.path.join(DATA, "gen2f_val_rescored.jsonl"))}
new_rows = {r["i"]: r for r in load_jsonl(os.path.join(DATA, "rej1_val_rescored.jsonl"))}
paired_errors = []
for i in sorted(base_rows):
    b, n = base_rows[i], new_rows[i]
    if b.get("pred_pka") is not None and n.get("pred_pka") is not None:
        paired_errors.append((abs(b["pred_pka"] - b["tgt_pka"]), abs(n["pred_pka"] - n["tgt_pka"])))
paired_errors = np.asarray(paired_errors)
assert paired_errors.shape == (235, 2)
assert abs(np.mean(paired_errors[:, 1] - paired_errors[:, 0]) - paired["new_minus_base_mae"]) < 1e-12

fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.15))
ax = axes[0]
unique_values = [base_single["unique_gen"], new_single["unique_gen"]]
x = np.arange(2)
bars = ax.bar(x, unique_values, color=["#0072B2", "#009E73"], width=0.58)
ax.set_xticks(x, ["gen2-final", "SFT\nsingle"])
ax.set_ylim(0, 280)
ax.set_ylabel("Unique valid molecules (n = 255)")
ax.set_title("Input-conditioned structure generation")
ax.grid(axis="y", alpha=0.20, lw=0.5)
for bar, value in zip(bars, unique_values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 6, str(value), ha="center", fontsize=9)
panel_label(ax, "a")

ax = axes[1]
direction_values = [base_all["direction_accuracy"], new_all["direction_accuracy"],
                    sft["best_of_8"]["oracle_best_sft"]["oracle_at_8"]["direction_accuracy"],
                    constant["constant_direction_accuracy"]]
labels = ["gen2-final\nsingle", "SFT\nsingle", "SFT\nBest-of-8", "Best\nconstant"]
colors = ["#0072B2", "#009E73", "#56B4E9", "#E69F00"]
x = np.arange(4)
bars = ax.bar(x, direction_values, color=colors, width=0.66)
ax.set_xticks(x, labels)
ax.set_ylim(0, 0.94)
ax.set_ylabel("Direction accuracy")
ax.set_title("No detected advantage over constant")
ax.grid(axis="y", alpha=0.20, lw=0.5)
for bar, value in zip(bars, direction_values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.014, f"{value:.3f}", ha="center", fontsize=8)
paired_delta = paired["new_direction_accuracy"] - paired["base_direction_accuracy"]
constant_delta = new_all["direction_accuracy"] - constant["constant_direction_accuracy"]
ax.text(0.03, 0.98,
        f"Paired SFT vs gen2 (n=235): {paired_delta:+.4f}; p = 0.00418\n"
        f"All-row SFT vs constant (n=255): {constant_delta:+.4f}; p = 0.7239",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.2)
panel_label(ax, "b")

ax = axes[2]
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", lw=0.9, color="0.25", zorder=1)
ax.scatter(paired_errors[:, 0], paired_errors[:, 1], s=10, alpha=0.42,
           color="#0072B2", edgecolors="none", rasterized=True, zorder=2)
limit = float(max(paired_errors.max() * 1.03, 1.0))
ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("gen2-final absolute error")
ax.set_ylabel("SFT absolute error")
ax.set_title("Paired absolute-error reduction")
ax.grid(alpha=0.15, lw=0.5)
ci = paired["new_minus_base_mae_pair_bootstrap_95ci"]
ax.text(0.04, 0.96, f"ΔMAE = {paired['new_minus_base_mae']:.3f}\n95% CI [{ci[0]:.3f}, {ci[1]:.3f}]\nn = 235",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", lw=0.5))
panel_label(ax, "c")
fig.text(0.5, -0.01,
         "All four direction bars use 255 rows; missing single-sample predictions are no-op. Best-of-8 remains oracle-only.\n"
         "The constant 3.74 is selected on 2,295 training rows. The separate SFT–gen2 paired test uses 235 both-scored rows.",
         ha="center", va="top", fontsize=7.4)
fig.tight_layout(w_pad=2.3)
save_figure(fig, "fig4_generation_path")
plt.close(fig)

# ---------------------------------------------------------------------------
# Table 2: five gates, with G4 updated after stratified recomputation.
# ---------------------------------------------------------------------------
table2_rows = [
    ["G1 Interpolation monotonicity", "1.000 (38/38)", "Pass"],
    ["G2 On-manifold trajectory", "off_manifold = 0.072", "Pass"],
    ["G3 Latent intervention", "response = 45%; moved = 30%", "Conditional causal"],
    ["G4 Gradient–chemistry alignment",
     "median cos = 0.107 (n = 43); weak-direction band", "Descriptive"],
    ["G5 Direction sign", "0.676 (71/105)", "Pass"],
]
write_csv("table2_gates", ["Gate", "Result", "Reading"], table2_rows)
write_tex("table2_gates", r"""\begin{table}[t]
\centering
\caption{Five tests of the latent pKa rollout. Entries labeled Pass or Conditional refer to prespecified thresholds for G1--G3 and G5. G4 is reported descriptively rather than as a binary gate: the bands are $|\cos|<0.05$ (random), $0.05$--$0.2$ (weak direction), and $\geq0.2$ (strong direction). The G4 row uses the expanded 43-pair median; the original top-20 bank and its small-gap subset are a separate selection.}
\label{tab:gates}
\small
\begin{tabular}{p{0.27\linewidth}p{0.45\linewidth}p{0.18\linewidth}}
\toprule
Gate & Result & Reading \\
\midrule
G1 Interpolation monotonicity & 1.000 (38/38) & Pass \\
G2 On-manifold trajectory & off-manifold = 0.072 & Pass \\
G3 Latent intervention & response = 45\%; moved = 30\% & Conditional causal \\
G4 Gradient--chemistry alignment & median cos $=0.107$ (n = 43); weak-direction band & Descriptive \\
G5 Direction sign & 0.676 (71/105) & Pass \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table 3: responsive versus non-responsive intervention samples.
# ---------------------------------------------------------------------------
diag = json.load(open(os.path.join(DATA, "fig3_nonresp_diagnosis.json")))
table3_rows = []
for subset, key in (("Responsive", "responsive"), ("Non-responsive", "nonresponsive")):
    gaps = np.asarray(diag[key]["gaps"], dtype=float)
    table3_rows.append([
        subset, len(gaps), f"{np.median(gaps):.2f}", f"{np.mean(gaps > 2.0):.2f}",
        f"{diag[key]['ionizable_frac']:.2f}",
    ])
assert table3_rows == [
    ["Responsive", 18, "2.79", "0.72", "0.61"],
    ["Non-responsive", 22, "2.42", "0.59", "0.77"],
], table3_rows
write_csv("table3_nonresponse",
          ["Subset", "n", "Median |target-base|", "Fraction >2.0", "Ionizable fraction"],
          table3_rows)
write_tex("table3_nonresponse", r"""\begin{table}[t]
\centering
\caption{Diagnosis of the 22 non-responsive samples in the C-group intervention. The smaller target gap and higher ionizable-group fraction in the non-responsive subset are inconsistent with chemical inaccessibility as the sole explanation. Ionizable fraction is the archived pipeline annotation; its underlying functional-group rule is not preserved in the plotting artifact.}
\label{tab:nonresponse}
\begin{tabular}{lcccc}
\toprule
Subset & n & Median $|$target$-$base$|$ & Fraction $>2.0$ & Ionizable \\
\midrule
Responsive & 18 & 2.79 & 0.72 & 0.61 \\
Non-responsive & 22 & 2.42 & 0.59 & 0.77 \\
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------------------------------------------------------------------
# Table 4: paired generation gains, oracle ceiling, and separate references.
# ---------------------------------------------------------------------------
best8 = sft["best_of_8"]["oracle_best_sft"]["oracle_at_8"]
noop_mae = sft["no_op_reference"]["mae_all_255"]
assert abs(noop_mae - 2.2932156862745097) < 1e-12
assert abs(sft["no_op_reference"]["mae_sft_scored_252"] - 2.2969841269841265) < 1e-12
assert abs(paired["base_direction_accuracy"] - 0.6425531914893617) < 1e-12
assert abs(paired["new_direction_accuracy"] - 0.6936170212765957) < 1e-12
assert abs(paired["mcnemar"]["two_sided_exact_p"] - 0.004180908203125) < 1e-15
assert abs(constant["constant_direction_accuracy"] - 0.6705882352941176) < 1e-12
assert abs(new_all["two_sided_exact_p"] - 0.723948145746914) < 1e-15
assert abs(best8["direction_accuracy"] - 0.8) < 1e-12

table4_rows = [
    ["Direction accuracy (all n=255; missing=no-op)", f"{base_all['direction_accuracy']:.4f}",
     f"{new_all['direction_accuracy']:.4f}", f"{best8['direction_accuracy']:.4f} (oracle)",
     f"{constant['constant_direction_accuracy']:.4f}"],
    ["Direction accuracy (paired n=235)", f"{paired['base_direction_accuracy']:.4f}",
     f"{paired['new_direction_accuracy']:.4f}", "—", "—"],
    ["Target MAE (paired n=235)", f"{paired['base_mae']:.3f}", f"{paired['new_mae']:.3f}", "—", "—"],
    ["Best-of-8 target MAE (oracle n=255)", "—", "—", f"{best8['mae_vs_target']:.3f}", "—"],
    ["Unique valid molecules", base_single["unique_gen"], new_single["unique_gen"], "—", "—"],
    ["Copy rate", f"{base_single['copy_rate']:.3f}", f"{new_single['copy_rate']:.3f}", "—", "—"],
    ["Separate no-op reference (n=255)", f"Direction N/A; target MAE = {noop_mae:.3f}", "", "", ""],
]
write_csv("table4_generation",
          ["Metric", "gen2-final", "SFT single", "SFT Best-of-8", "Train-selected constant"], table4_rows)
table4_body = "\n".join([
    f"Direction accuracy (all $n=255$; missing=no-op) & {base_all['direction_accuracy']:.4f} & {new_all['direction_accuracy']:.4f} & {best8['direction_accuracy']:.4f} (oracle) & {constant['constant_direction_accuracy']:.4f} \\\\",
    f"Direction accuracy (paired $n=235$) & {paired['base_direction_accuracy']:.4f} & \\textbf{{{paired['new_direction_accuracy']:.4f}}} & --- & --- \\\\",
    f"Target MAE (paired $n=235$) & {paired['base_mae']:.3f} & \\textbf{{{paired['new_mae']:.3f}}} & --- & --- \\\\",
    f"Best-of-8 target MAE (oracle $n=255$) & --- & --- & {best8['mae_vs_target']:.3f} & --- \\\\",
    f"Unique valid molecules & {base_single['unique_gen']} & {new_single['unique_gen']} & --- & --- \\\\",
    f"Copy rate & {base_single['copy_rate']:.3f} & {new_single['copy_rate']:.3f} & --- & --- \\\\",
])
write_tex("table4_generation", r"""\begin{table*}[t]
\centering
\caption{Generation results with separated evaluation protocols. The all-row direction comparison uses the same 255 rows for gen2-final, SFT, and the train-selected constant $3.74$, with missing single-sample predictions treated as no-op; SFT does not have a detected advantage over the constant (34 constant-only versus 38 SFT-only correct, exact McNemar $p=0.7239$). The separate paired gen2-final/SFT comparison uses the 235 both-scored rows (direction $p=0.00418$); its MAE cells therefore share one denominator. SFT Best-of-8 is an oracle diagnostic on all 255 rows, not a deployable selector. The strict no-op reference has no direction prediction and is N/A rather than a direction baseline.}
\label{tab:generation}
\small
\begin{tabular}{@{}p{0.29\linewidth}cccc@{}}
\toprule
Metric & gen2-final & SFT single & SFT Best-of-8 & Train-selected constant \\
\midrule
""" + table4_body + r"""
\midrule
\multicolumn{5}{l}{Separate no-op reference (n=255): strict direction N/A; target MAE = 2.293.} \\
\bottomrule
\end{tabular}
\end{table*}
""")

print("ALL MAIN ASSETS DONE")
