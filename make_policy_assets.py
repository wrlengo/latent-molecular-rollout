"""Build direction/amplitude figures and tables from completed policy experiments."""
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
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
SKY = "#56B4E9"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#999999"


def load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def save_figure(fig, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(FIGS, f"{stem}.{ext}")
        fig.savefig(path, dpi=300)
        print(f"saved {path}")


def panel_label(ax, label):
    ax.text(-0.17, 1.04, label, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def write_tex(name, content):
    path = os.path.join(TABLES, name)
    with open(path, "w") as f:
        f.write(content)
    print(f"saved {path}")


def pass_mark(value):
    return r"\cmark" if value else r"\xmark"


base = load_json("policy_head_baseline_results.json")
amp = load_json("policy_head_amp_results.json")
steps = [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]
step_keys = [f"s={s}" for s in steps]
ens = base["ensemble"]
assert base["summary"]["criterion_met_by_ensemble"]
assert not amp["summary"]["amplitude_criterion_met_by_ensemble"]
assert abs(ens["test_median_cos"] - 0.5767258405685425) < 1e-10
for key in step_keys:
    assert abs(ens["amplitude_test"][key]["pearson_signed"] + 0.0199290155750938) < 1e-10

# Main Fig. 4: direction/amplitude dissociation and step-size sensitivity.
fig = plt.figure(figsize=(12.6, 4.8))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.30)
ax = fig.add_subplot(gs[0, 0])
ax.axvline(0.3, color=GREEN, ls="--", lw=1.1, label="Direction criterion = 0.3")
ax.axhline(0.5, color=VERMILION, ls="--", lw=1.1, label="Amplitude criterion = 0.5")
ax.axhline(0.0, color="0.65", lw=0.7)
for i, seed in enumerate(base["seed_results"]):
    x = seed["test_median_cos"]
    y = seed["amplitude_test"]["s=1.0"]["pearson_signed"]
    ax.scatter(x, y, s=48, color=SKY, edgecolor="white", linewidth=0.5,
               zorder=4, label="Policy seeds" if i == 0 else None)
    ax.text(x + 0.012, y + 0.018, f"seed {i}", fontsize=7.5, color="0.3")
ax.scatter(ens["test_median_cos"], ens["amplitude_test"]["s=1.0"]["pearson_signed"],
           s=115, marker="*", color=VERMILION, edgecolor="white", linewidth=0.6,
           zorder=5, label="Policy ensemble")
# Direction-only references are shown on a separate rug because no comparable
# amplitude correlation was measured for them.
rug_y = -0.25
references = [
    (base["g4_baseline_median_cos"], "G4", GRAY),
    (0.002171, "z-ridge", PURPLE),
    (0.676, "G5*", GREEN),
]
for x, label, color in references:
    ax.scatter(x, rug_y, s=46, marker="^", facecolor="white", edgecolor=color,
               linewidth=1.2, zorder=3)
    ax.text(x, rug_y + 0.032, label, ha="center", va="bottom", fontsize=7.2)
ax.text(0.02, 0.025, "Triangles: direction-only references; no amplitude correlation.\n*G5 is sign accuracy, not a cosine; shown only as a scale reference.",
        transform=ax.transAxes, fontsize=7.2, color="0.35", va="bottom")
ax.set_xlim(-0.06, 0.78)
ax.set_ylim(-0.39, 0.61)
ax.set_xlabel(r"Direction: median cosine with $h_b-h_a$")
ax.set_ylabel(r"Amplitude: corr($\Delta$pred, $\Delta$target)")
ax.set_title("Direction is learnable; amplitude is not yet controllable")
ax.legend(frameon=False, loc="upper left", fontsize=7.2)
ax.grid(alpha=0.14, lw=0.5)
panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
for i, seed in enumerate(base["seed_results"]):
    values = [seed["amplitude_test"][k]["median_abs_dpred"] for k in step_keys]
    ax.plot(steps, values, color=SKY, alpha=0.42, lw=1.0, marker="o", ms=3,
            label="Individual seeds" if i == 0 else None)
median_abs = [ens["amplitude_test"][k]["median_abs_dpred"] for k in step_keys]
ax.axhspan(0.2, 1.0, color=GREEN, alpha=0.09, label="Target small-step band [0.2, 1.0]")
ax.plot(steps, median_abs, color=BLUE, lw=2.0, marker="o", ms=4.5,
        label=r"Ensemble median $|\Delta \mathrm{p}K_a|$")
ax.set_xlabel("Fixed latent step size $s$")
ax.set_ylabel(r"Median $|\Delta \mathrm{p}K_a|$ (ridge readout)")
ax.set_ylim(0.0, 1.08)
ax.set_xlim(0.0, 3.08)
ax.grid(alpha=0.16, lw=0.5)
ax_corr = ax.twinx()
ax_corr.spines.right.set_visible(True)
corrs = [ens["amplitude_test"][k]["pearson_signed"] for k in step_keys]
ax_corr.plot(steps, corrs, color=VERMILION, lw=1.8, marker="s", ms=4,
             label=r"corr($\Delta$pred, $\Delta$target)")
ax_corr.axhline(0.0, color="0.45", lw=0.7)
ax_corr.axhline(0.5, color=VERMILION, lw=1.0, ls="--", label="Amplitude criterion = 0.5")
ax_corr.set_ylabel("Signed amplitude correlation", color=VERMILION)
ax_corr.tick_params(axis="y", colors=VERMILION)
ax_corr.set_ylim(-0.12, 0.57)
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax_corr.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left", fontsize=7.4)
ax.set_title("Larger steps change magnitude, not target correlation")
ax.text(1.55, 0.045, "Ensemble median enters the band at s = 1, 2, 3;\ncorr remains −0.0199", ha="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
panel_label(ax, "b")
save_figure(fig, "fig4_direction_amplitude_main")
plt.close(fig)

# Main Table 5: paired direction and amplitude criteria on the same policy head.
rows = [
    ("G4 naive gradient", "0.107", "---", "Fail", "---"),
    ("Frozen ridge ($z$-scored)", "0.00217", "---", "Fail", "---"),
]
for seed in base["seed_results"]:
    rows.append((
        f"Policy seed {seed['seed']}",
        f"{seed['test_median_cos']:.3f}",
        f"{seed['amplitude_test']['s=1.0']['pearson_signed']:.3f}",
        "Pass" if seed["test_median_cos"] > 0.3 else "Fail",
        "Pass" if seed["amplitude_test"]["s=1.0"]["pearson_signed"] > 0.5 else "Fail",
    ))
rows.append(("Policy ensemble", f"{ens['test_median_cos']:.3f}",
             f"{ens['amplitude_test']['s=1.0']['pearson_signed']:.3f}", "Pass", "Fail"))
body = "\n".join(
    f"{name} & {cos} & {corr} & {pass_mark(d == 'Pass')} & {pass_mark(a == 'Pass') if a != '---' else '---'} \\\\"
    for name, cos, corr, d, a in rows
)
write_tex("table5_direction_amplitude.tex", r"""\begin{table}[t]
\centering
\caption{Directional learnability and amplitude controllability evaluated on the same 43 held-out pairs. The direction criterion applies to the median point estimate ($>0.3$), not to every pair or to the CI lower bound; pair-level fractions and median CIs are given in Table~S3. The amplitude criterion is signed correlation $>0.5$. Amplitude correlations are invariant to the fixed step size. G4 and ridge have no comparable target-conditioned amplitude evaluation.}
\label{tab:direction_amplitude}
\small
\begin{tabular}{lrrcc}
\toprule
Method & Direction cos & Amplitude corr & Direction & Amplitude \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# SI Table S3: complete baseline seed detail.
si_rows = []
for seed in base["seed_results"] + [ens]:
    name = "Ensemble" if "seed" not in seed else f"Seed {seed['seed']}"
    a = seed["amplitude_test"]["s=1.0"]
    learned = seed["learned_amplitude_test"]
    ci = seed["test_median_cos_ci95"]
    n_above = int(np.sum(np.asarray(seed["test_cos"], dtype=float) > 0.3))
    si_rows.append(
        f"{name} & {seed['test_median_cos']:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] & "
        f"{n_above}/43 ({n_above / 43:.3f}) & {seed['test_small_gap_median_cos']:.3f} & "
        f"{a['pearson_signed']:.3f} & {a['sign_acc']:.3f} & {learned['step_norm_cv']:.3f} \\\\"
    )
write_tex("tableS3_policy_seeds.tex", r"""\begin{table}[h]
\centering
\caption{Target-conditioned policy-head seed detail on 43 held-out pairs. The prespecified direction criterion applies to the median point estimate, not to every pair or to the CI lower bound; all four median CIs have lower bounds below $0.3$. Small-gap pairs are the 13 pairs with $1\leq|\Delta\pka|<2$. Amplitude statistics use $s=1.0$; correlation and sign accuracy do not depend on the scale of a fixed nonzero step. Norm CV is the coefficient of variation of the raw policy displacement norms before fixed-step normalization.}
\label{tab:policy_seeds}
\scriptsize
\begin{tabular}{lccccccc}
\toprule
Seed & Cos & 95\% CI & Pairs $\cos>0.3$ & Small-gap cos & Amp. corr & Sign acc. & Norm CV \\
\midrule
""" + "\n".join(si_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# SI Table S4: explicit amplitude-supervision outcome.
amp_rows = []
for seed in amp["seed_results"] + [amp["ensemble"]]:
    name = "Ensemble" if "seed" not in seed else f"Seed {seed['seed']}"
    learned = seed["learned_amplitude_test"]
    passed = seed["test_median_cos"] >= 0.3 and learned["pearson_signed"] > 0.5
    amp_rows.append(
        f"{name} & {seed['test_median_cos']:.3f} & {learned['pearson_signed']:.3f} & "
        f"{learned['median_abs_dpred']:.3f} & {learned['median_step_norm']:.3f} & "
        f"{pass_mark(passed)} \\\\"
    )
write_tex("tableS4_amplitude_supervision.tex", r"""\begin{table}[h]
\centering
\caption{Explicit amplitude supervision with $\lambda_{\mathrm{amp}}=1$. The joint criterion requires median cosine $\geq0.3$ and learned-amplitude correlation $>0.5$. Neither any individual seed nor the ensemble satisfies both criteria. These values use the learned per-pair displacement norm, unlike the fixed-step sensitivity analysis.}
\label{tab:amplitude_supervision}
\scriptsize
\begin{tabular}{lccccc}
\toprule
Seed & Cos & Learned amp. corr & Median $|\Delta$pKa$|$ & Median step norm & Joint pass \\
\midrule
""" + "\n".join(amp_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# SI Table S5: complete fixed-step sensitivity for the direction ensemble.
step_rows = []
for s, key in zip(steps, step_keys):
    a = ens["amplitude_test"][key]
    step_rows.append(
        f"{s:.1f} & {a['median_abs_dpred']:.4f} & {a['frac_in_0.2_1.0']:.3f} & "
        f"{a['sign_acc']:.3f} & {a['pearson_signed']:.4f} \\\\"
    )
write_tex("tableS5_step_sensitivity.tex", r"""\begin{table}[h]
\centering
\caption{Fixed-step amplitude sensitivity for the unit-normalized policy ensemble on 43 held-out pairs. The target band is $0.2\leq|\Delta\pka|\leq1.0$. Signed Pearson correlation and sign accuracy are invariant to positive fixed rescaling. The calibration condition requires both a median in the target band and correlation $>0.3$; it is not met at any step.}
\label{tab:step_sensitivity}
\small
\begin{tabular}{rcccc}
\toprule
Step $s$ & Median $|\Delta\pka|$ & Fraction in $[0.2,1.0]$ & Sign acc. & Signed corr \\
\midrule
""" + "\n".join(step_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# P3: paired policy-direction decoding, separately for greedy and sampled runs.
def summarize_p3(p3):
    rows = p3["rows"]
    paired = []
    for row in rows:
        base_score = row.get("baseline_learned_oracle_pka")
        int_score = row.get("intervened_learned_oracle_pka")
        changed = row["baseline_text"] != row["intervened_text"]
        oracle_scored = base_score is not None and int_score is not None
        if not oracle_scored:
            assert not changed and row["baseline_smiles"] == row["intervened_smiles"]
            delta = 0.0
        else:
            delta = float(int_score - base_score)
        paired.append({
            "delta": delta,
            "target": float(row["target_delta_pka"]),
            "changed": changed,
            "oracle_scored": oracle_scored,
        })
    deltas = np.asarray([r["delta"] for r in paired], dtype=float)
    targets = np.asarray([r["target"] for r in paired], dtype=float)
    changed = np.asarray([r["changed"] for r in paired], dtype=bool)
    signs = deltas * targets > 0
    readout_deltas = np.asarray([r["readout_realized_delta_pka"] for r in rows], dtype=float)
    readout_targets = np.asarray([r["target_delta_pka"] for r in rows], dtype=float)
    intervened = p3["summary"]["intervened"]
    summary = {
        "n": len(rows),
        "n_paired_oracle_scored": sum(r["oracle_scored"] for r in paired),
        "n_identity_zero_without_oracle": sum(not r["oracle_scored"] for r in paired),
        "n_sequence_changed": int(changed.sum()),
        "n_smiles_changed": p3["summary"]["n_smiles_changed"],
        "h_target_cosine_median": p3["summary"]["h_target_cosine_median"],
        "h_target_relative_error_median": p3["summary"]["h_target_relative_error_median"],
        "readout_direction_accuracy": float(np.mean(readout_deltas * readout_targets > 0)),
        "readout_abs_delta_median": float(np.median(np.abs(readout_deltas))),
        "intervened_validity": intervened["validity"],
        "intervened_tanimoto_median_valid": intervened["tanimoto_median_valid"],
        "paired_direction_accuracy": float(signs.mean()),
        "paired_abs_delta_median": float(np.median(np.abs(deltas))),
        "paired_abs_delta_in_0.2_1.0": float(np.mean((np.abs(deltas) >= 0.2) & (np.abs(deltas) <= 1.0))),
        "paired_zero_delta": int(np.sum(deltas == 0)),
        "changed_subset_n": int(changed.sum()),
        "changed_subset_direction_accuracy": float(signs[changed].mean()) if changed.any() else None,
        "changed_subset_abs_delta_median": float(np.median(np.abs(deltas[changed]))) if changed.any() else None,
        "learned_oracle": p3["protocol"]["learned_oracle"],
        "checkpoint_compatibility": p3["protocol"]["checkpoint_compatibility"],
        "decoding_protocol": {key: p3["protocol"].get(key) for key in
                              ("decoding", "max_new_tokens", "temperature", "top_p", "paired_generation_seed")},
    }
    summary["criteria"] = {
        "validity_at_least_0.9": summary["intervened_validity"] >= 0.9,
        "tanimoto_median_at_least_0.6": summary["intervened_tanimoto_median_valid"] >= 0.6,
        "direction_at_least_0.65": summary["paired_direction_accuracy"] >= 0.65,
        "amplitude_median_in_0.2_1.0": 0.2 <= summary["paired_abs_delta_median"] <= 1.0,
    }
    return summary


p3_protocols = [
    ("greedy", "policy_decode_full.json", "policy_decode_paired_summary.json",
     "tableS6_policy_decode.tex", "tab:p3_decode"),
    ("sampled", "policy_decode_sample.json", "policy_decode_sample_paired_summary.json",
     "tableS7_policy_decode_sample.tex", "tab:p3_decode_sample"),
]
for decoding, artifact_name, summary_name, table_name, label in p3_protocols:
    if not os.path.exists(os.path.join(DATA, artifact_name)):
        continue
    p3 = load_json(artifact_name)
    assert p3["protocol"]["decoding"] == ("sample" if decoding == "sampled" else "greedy")
    paired_summary = summarize_p3(p3)
    with open(os.path.join(DATA, summary_name), "w") as f:
        json.dump(paired_summary, f, indent=2)
    print(f"saved data/{summary_name}")

    n = paired_summary["n"]
    compat = paired_summary["checkpoint_compatibility"]
    table_rows = [
        ("$h_T$ target cosine, median", f"{paired_summary['h_target_cosine_median']:.4f}", "---"),
        ("$h_T$ relative error, median", f"{paired_summary['h_target_relative_error_median']:.4f}", "---"),
        ("Generator-readout direction", f"{paired_summary['readout_direction_accuracy']:.3f}", "---"),
        ("Generator-readout median $|\\Delta\\pka|$", f"{paired_summary['readout_abs_delta_median']:.4f}", "---"),
        ("Intervened SMILES validity", f"{paired_summary['intervened_validity']:.3f}",
         pass_mark(paired_summary["criteria"]["validity_at_least_0.9"])),
        ("Source-anchored Tanimoto, median", f"{paired_summary['intervened_tanimoto_median_valid']:.3f}",
         pass_mark(paired_summary["criteria"]["tanimoto_median_at_least_0.6"])),
        ("Changed decoded sequences", f"{paired_summary['n_sequence_changed']}/{n}", "---"),
        ("Both learned-oracle scored", f"{paired_summary['n_paired_oracle_scored']}/{n}", "---"),
        ("Paired learned-oracle direction", f"{paired_summary['paired_direction_accuracy']:.3f}",
         pass_mark(paired_summary["criteria"]["direction_at_least_0.65"])),
        ("Paired learned-oracle median $|\\Delta\\pka|$", f"{paired_summary['paired_abs_delta_median']:.4f}",
         pass_mark(paired_summary["criteria"]["amplitude_median_in_0.2_1.0"])),
        ("Paired learned-oracle fraction in $[0.2,1.0]$", f"{paired_summary['paired_abs_delta_in_0.2_1.0']:.3f}", "---"),
    ]
    protocol_title = ("Greedy" if decoding == "greedy" else
                      "Sampled ($T=0.9$, top-$p=0.95$, seed 0)")
    write_tex(table_name, r"""\begin{table}[h]
\centering
\caption{""" + protocol_title + r""" P3 policy-direction decoding on all 80 eligible held-out source molecules. Paired effects are intervened minus baseline learned-oracle predictions, so unchanged sequences have zero effect and count against directional success. Identical outputs retain that zero effect by identity when their shared invalid SMILES cannot be oracle-scored; no changed output lacks a paired oracle score. Checks are validity $\geq0.9$, Tanimoto median $\geq0.6$, direction $\geq0.65$, and median $|\Delta\pka|\in[0.2,1.0]$. The MAE-1.1 PkaHead is a learned oracle, not external or experimental validation. Property/generator PkaHead and projector maximum absolute tensor differences are """ + f"{compat['pka_head_max_abs_diff']:.4f} and {compat['projector_max_abs_diff']:.4f}" + r""", respectively.}
\label{""" + label + r"""}
\small
\begin{tabular}{lrc}
\toprule
Metric & Value & Check \\
\midrule
""" + "\n".join(f"{name} & {value} & {check} \\\\" for name, value, check in table_rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# Quantify reuse of the frozen 43-pair direction-test cohort by the 80-source P3 audit.
if os.path.exists(os.path.join(DATA, "policy_decode_full.json")):
    from rdkit import Chem

    def canonical(smiles):
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

    p3 = load_json("policy_decode_full.json")
    with open(os.path.join(DATA, "frozen_ridge_test_pairs.csv")) as f:
        policy_rows = list(csv.DictReader(f))
    p3_sources = {canonical(r["src_smiles"]) for r in p3["rows"]}
    policy_sources = {canonical(r["src_smiles"]) for r in policy_rows}
    policy_endpoints = policy_sources | {canonical(r["tgt_smiles"]) for r in policy_rows}
    p3_pairs = {(canonical(r["src_smiles"]), canonical(r["tgt_smiles"])) for r in p3["rows"]}
    policy_pairs = {(canonical(r["src_smiles"]), canonical(r["tgt_smiles"])) for r in policy_rows}
    overlap = {
        "p3_sources": len(p3_sources),
        "policy_pairs": len(policy_rows),
        "policy_sources": len(policy_sources),
        "policy_endpoints": len(policy_endpoints),
        "source_source_overlap": len(p3_sources & policy_sources),
        "source_any_endpoint_overlap": len(p3_sources & policy_endpoints),
        "policy_pairs_with_p3_source": sum(canonical(r["src_smiles"]) in p3_sources for r in policy_rows),
        "exact_directed_pair_overlap": len(p3_pairs & policy_pairs),
        "interpretation": "P3 is a decoding audit on a cohort overlapping the policy test, not an independent replication.",
    }
    with open(os.path.join(DATA, "policy_decode_overlap.json"), "w") as f:
        json.dump(overlap, f, indent=2)
    print("saved data/policy_decode_overlap.json")

# Fig. S11: archived goal-conditioned evaluations. Only k=1,3,5 were saved.
goal = load_json("goal_rollout_results.json")
fig, ax = plt.subplots(figsize=(8.8, 4.8))
ks = [1, 3, 5]
for i, seed in enumerate(goal["seed_results"]):
    values = [seed[f"test_k{k}_median_readout_error"] for k in ks]
    ax.plot(ks, values, marker="o", lw=1.5, label=f"Seed {i}")
noop = goal["seed_results"][0]["test_noop_median_readout_error"]
oracle = goal["seed_results"][0]["test_oracle_endpoint_median_readout_error"]
ax.axhline(noop, color=GRAY, ls="--", lw=1.2, label=f"No-op = {noop:.2f}")
ax.axhline(oracle, color=BLUE, ls=":", lw=1.4, label=f"Oracle endpoint = {oracle:.2f}")
ax.axhline(0.5, color=VERMILION, ls="-.", lw=1.2, label="Criterion = 0.5")
ax.set_xticks(ks)
ax.set_yscale("log")
ax.set_xlabel("Rollout steps $k$ (archived evaluation points)")
ax.set_ylabel("Median $|$ridge$(h_k)-$target$|$ (log scale)")
ax.set_title("Goal-conditioned rollout does not reach the prespecified criterion")
ax.grid(alpha=0.16, lw=0.5, which="both")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
save_figure(fig, "figS11_goal_rollout_seeds")
plt.close(fig)

# Fig. S12: G2 squared-loss to absolute-error conversion.
rollout = load_json("fig2_final.json")
trajectories = rollout["trajectories"]
loss0 = np.asarray([t["loss0"] for t in trajectories], dtype=float)
loss1 = np.asarray([t["loss1"] for t in trajectories], dtype=float)
err0 = np.sqrt(loss0)
err1 = np.sqrt(loss1)
assert np.allclose(loss0, 4.0)
assert abs(np.median(loss1) - 0.06312691047787666) < 1e-12
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
for ax, before, after, ylabel, title in (
    (axes[0], loss0, loss1, "Squared target loss", "Archived loss"),
    (axes[1], err0, err1, r"$|$predicted pKa $-$ target$|$", "Converted absolute error"),
):
    ax.plot([0, 1], [np.median(before), np.median(after)], color=VERMILION,
            marker="o", ms=7, lw=1.8)
    ax.scatter(np.zeros(len(before)), before, s=10, color=GRAY, alpha=0.35)
    ax.scatter(np.ones(len(after)), after, s=10, color=BLUE, alpha=0.45)
    ax.set_xticks([0, 1], ["Step 0", "Step 20"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.17, lw=0.5)
    ax.text(0.5, 0.93, f"Median: {np.median(before):.3f} $\\rightarrow$ {np.median(after):.3f}",
            transform=ax.transAxes, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.75", lw=0.6))
axes[1].text(0.5, 0.82, f"Final mean = {np.mean(err1):.3f}\n" + r"Conversion: $|$error$|=\sqrt{loss}$",
             transform=axes[1].transAxes, ha="center", va="top", fontsize=8)
panel_label(axes[0], "a")
panel_label(axes[1], "b")
fig.tight_layout(w_pad=2.4)
save_figure(fig, "figS12_g2_loss_error_conversion")
plt.close(fig)

analysis = {
    "baseline": {
        "ensemble_median_cos": ens["test_median_cos"],
        "ensemble_amplitude_corr": ens["amplitude_test"]["s=1.0"]["pearson_signed"],
        "step_sizes": steps,
        "step_median_abs_dpred": median_abs,
        "all_steps_corr_below_0.1": all(c < 0.1 for c in corrs),
        "pairs_with_cos_above_0.3": {
            ("ensemble" if "seed" not in seed else f"seed_{seed['seed']}"): {
                "n": int(np.sum(np.asarray(seed["test_cos"], dtype=float) > 0.3)),
                "fraction": float(np.mean(np.asarray(seed["test_cos"], dtype=float) > 0.3)),
                "median_cos_ci95": seed["test_median_cos_ci95"],
            } for seed in base["seed_results"] + [ens]
        },
        "calibration_criterion_met": any(
            0.2 <= ens["amplitude_test"][k]["median_abs_dpred"] <= 1.0
            and ens["amplitude_test"][k]["pearson_signed"] > 0.3 for k in step_keys
        ),
    },
    "amplitude_supervision": {
        "ensemble_median_cos": amp["ensemble"]["test_median_cos"],
        "ensemble_learned_amplitude_corr": amp["ensemble"]["learned_amplitude_test"]["pearson_signed"],
        "joint_criterion_met": amp["summary"]["amplitude_criterion_met_by_ensemble"],
    },
    "g2": {
        "median_loss_initial": float(np.median(loss0)),
        "median_loss_final": float(np.median(loss1)),
        "median_abs_error_initial": float(np.median(err0)),
        "median_abs_error_final": float(np.median(err1)),
        "mean_abs_error_final": float(np.mean(err1)),
    },
}
with open(os.path.join(DATA, "policy_head_analysis.json"), "w") as f:
    json.dump(analysis, f, indent=2)
print(json.dumps(analysis, indent=2))
