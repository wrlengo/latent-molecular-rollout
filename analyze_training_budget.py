"""Verify preregistered training-budget reruns and build their tables."""
import ast
import hashlib
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TABLES = os.path.join(ROOT, "tables")
FIGS = os.path.join(ROOT, "figs")


def load(name):
    with open(os.path.join(DATA, name)) as handle:
        return json.load(handle)


def load_jsonl(name):
    with open(os.path.join(DATA, name)) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(name):
    with open(os.path.join(DATA, name), "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def direction(row, prediction):
    return (prediction - row["src_pka"]) * (row["tgt_pka"] - row["src_pka"]) > 0


def exact_mcnemar(base_ok, model_ok):
    base_only = sum(b and not m for b, m in zip(base_ok, model_ok))
    model_only = sum(m and not b for b, m in zip(base_ok, model_ok))
    discordant = base_only + model_only
    if not discordant:
        return {"constant_only_correct": 0, "model_only_correct": 0, "p_exact_two_sided": 1.0}
    tail = sum(math.comb(discordant, k) for k in range(min(base_only, model_only) + 1))
    return {
        "constant_only_correct": base_only,
        "model_only_correct": model_only,
        "discordant": discordant,
        "p_exact_two_sided": min(1.0, 2.0 * tail / 2 ** discordant),
    }


def constant_metrics(rows, predictions, constant):
    model_ok = [direction(row, prediction) for row, prediction in zip(rows, predictions)]
    constant_ok = [direction(row, constant) for row in rows]
    return {
        "n": len(rows),
        "n_scored": len(rows),
        "n_missing_noop": 0,
        "n_correct": sum(model_ok),
        "constant_n_correct": sum(constant_ok),
        "constant_direction_accuracy": float(np.mean(constant_ok)),
        "direction_accuracy": float(np.mean(model_ok)),
        "mae_all_rows_missing_noop": float(np.mean([
            abs(prediction - row["tgt_pka"]) for row, prediction in zip(rows, predictions)
        ])),
        "mcnemar_vs_constant": exact_mcnemar(constant_ok, model_ok),
    }


def sft_descriptive_metrics(rows):
    scored = [row for row in rows if row.get("pred_pka") is not None]
    tanimoto = [row["tanimoto"] for row in rows if row.get("tanimoto") is not None]
    return {
        "unique_valid": len({row["gen"] for row in rows if row.get("valid") and row.get("gen") is not None}),
        "validity": float(np.mean([bool(row["valid"]) for row in rows])),
        "tanimoto": float(np.mean(tanimoto)),
        "scored_direction_accuracy": float(np.mean([direction(row, row["pred_pka"]) for row in scored])),
        "scored_mae": float(np.mean([abs(row["pred_pka"] - row["tgt_pka"]) for row in scored])),
    }


def oracle_bank_metrics(rows):
    predictions = []
    for row in rows:
        if row["scored"]:
            predictions.append(min(row["scored"], key=lambda item: abs(item[1] - row["tgt_pka"]))[1])
        else:
            predictions.append(row["src_pka"])
    scored = [bool(row["scored"]) for row in rows]
    return {
        "n_scored": sum(scored),
        "direction_accuracy_scored": float(np.mean([
            direction(row, prediction) for row, prediction, keep in zip(rows, predictions, scored) if keep
        ])),
        "direction_accuracy_all_missing_noop": float(np.mean([
            direction(row, prediction) for row, prediction in zip(rows, predictions)
        ])),
        "mae_all_missing_noop": float(np.mean([
            abs(prediction - row["tgt_pka"]) for row, prediction in zip(rows, predictions)
        ])),
    }


def bank_predictions(rows):
    return [
        float(row["scored"][0][1]) if row["scored"] else row["src_pka"]
        for row in rows
    ]


def paired_bank_comparison(base_rows, new_rows):
    assert len(base_rows) == len(new_rows) == 255
    base_ok, new_ok = [], []
    for base, new in zip(base_rows, new_rows):
        assert base["src"] == new["src"]
        assert base["src_pka"] == new["src_pka"] and base["tgt_pka"] == new["tgt_pka"]
        base_prediction = bank_predictions([base])[0]
        new_prediction = bank_predictions([new])[0]
        base_ok.append(direction(base, base_prediction))
        new_ok.append(direction(new, new_prediction))
    return {
        "baseline_direction_accuracy": float(np.mean(base_ok)),
        "model_direction_accuracy": float(np.mean(new_ok)),
        "mcnemar_vs_rej1": exact_mcnemar(base_ok, new_ok),
    }


def sft_metrics(rows, constant):
    predictions = [row["pred_pka"] if row.get("pred_pka") is not None else row["src_pka"] for row in rows]
    result = constant_metrics(rows, predictions, constant)
    result.update({
        "n_scored": sum(row.get("pred_pka") is not None for row in rows),
        "n_missing_noop": sum(row.get("pred_pka") is None for row in rows),
        "copy_rate": float(np.mean([bool(row["copy"]) for row in rows])),
    })
    return result


def canon(smiles):
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def dpo_metrics(rows, constant):
    predictions, first_smiles = [], []
    for row in rows:
        if row["scored"]:
            first_smiles.append(row["scored"][0][0])
            predictions.append(float(row["scored"][0][1]))
        else:
            first_smiles.append(None)
            predictions.append(row["src_pka"])
    result = constant_metrics(rows, predictions, constant)
    result.update({
        "n_scored": sum(bool(row["scored"]) for row in rows),
        "n_missing_noop": sum(not row["scored"] for row in rows),
        "copy_rate": float(np.mean([
            smiles is not None and canon(smiles) == canon(row["src"])
            for row, smiles in zip(rows, first_smiles)
        ])),
    })
    return result


def sft_train_record(name, expected_epochs):
    with open(os.path.join(DATA, name)) as handle:
        records = [ast.literal_eval(line.strip()) for line in handle if "'train_runtime'" in line]
    assert len(records) == 1
    record = records[0]
    assert int(record["epoch"]) == expected_epochs
    return record


def paired_sft(base_rows, new_rows):
    base = {row["i"]: row for row in base_rows}
    new = {row["i"]: row for row in new_rows}
    assert base.keys() == new.keys()
    pairs = []
    for index in sorted(base):
        b, n = base[index], new[index]
        assert b["src"] == n["src"] and b["src_pka"] == n["src_pka"] and b["tgt_pka"] == n["tgt_pka"]
        if b.get("pred_pka") is not None and n.get("pred_pka") is not None:
            pairs.append((b, n))
    base_ok = [direction(b, b["pred_pka"]) for b, _ in pairs]
    new_ok = [direction(n, n["pred_pka"]) for _, n in pairs]
    deltas = np.asarray([abs(n["pred_pka"] - n["tgt_pka"]) - abs(b["pred_pka"] - b["tgt_pka"])
                         for b, n in pairs])
    rng = np.random.default_rng(42)
    boot = rng.choice(deltas, size=(10000, len(deltas)), replace=True).mean(axis=1)
    return {
        "n_both_scored": len(pairs),
        "gen2_direction_accuracy": float(np.mean(base_ok)),
        "sft_direction_accuracy": float(np.mean(new_ok)),
        "mcnemar_vs_gen2": exact_mcnemar(base_ok, new_ok),
        "new_minus_gen2_mae": float(deltas.mean()),
        "pair_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def verify_policy():
    original = load("policy_head_baseline_results.json")
    rerun = load("policy_head_training_budget_results.json")
    assert rerun["config"]["epochs"] == 150 and rerun["config"]["patience"] == 25
    seeds = []
    for old, new in zip(original["seed_results"], rerun["seed_results"]):
        assert old["seed"] == new["seed"]
        history = new["val_history"]
        best = max(history, key=lambda item: item["val_median_cos"])
        assert best["epoch"] == new["best_epoch_one_indexed"]
        assert new["stopped_epoch_one_indexed"] == len(history)
        seeds.append({
            "seed": new["seed"],
            "best_epoch": new["best_epoch_one_indexed"],
            "stopped_epoch": new["stopped_epoch_one_indexed"],
            "best_val_median_cos": new["best_val_median_cos"],
            "test_median_cos": new["test_median_cos"],
            "original_best_epoch": old["best_epoch"] + 1,
            "reproduces_original_best_epoch": new["best_epoch"] == old["best_epoch"],
            "test_median_cos_difference_from_original": new["test_median_cos"] - old["test_median_cos"],
            "val_history": history,
        })
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for seed in seeds:
        history = seed["val_history"]
        epochs = [item["epoch"] for item in history]
        values = [item["val_median_cos"] for item in history]
        ax.plot(epochs, values, lw=1.2, label=f"seed {seed['seed']}")
        ax.scatter([seed["best_epoch"]], [seed["best_val_median_cos"]], s=34, zorder=4)
        ax.axvline(seed["best_epoch"], color=f"C{seed['seed']}", lw=0.6, alpha=0.25)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation median cosine")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.18, lw=0.5)
    fig.tight_layout()
    for extension in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"figS_policy_validation_curves.{extension}"), dpi=300)
    plt.close(fig)
    return {
        "max_epochs": 150,
        "patience": 25,
        "seeds": seeds,
        "interpretation": "Best validation epochs are early and followed by the prespecified 25-epoch no-improvement stopping window; this tests the stopped-at-cap undertraining hypothesis but does not prove global convergence.",
    }


def main():
    prereg = load("training_budget_preregistration.json")
    assert prereg["status"] == "fixed_before_sft3_dpo4_submission"
    constant = prereg["common"]["constant"]
    assert constant == 3.74
    sft_training = {
        "1": sft_train_record("rej_sft_1649265.log", 1),
        "3": sft_train_record("rej_sft_1653233.log", 3),
    }
    dpo4_training = load("offline_dpo_e4_training_summary.json")
    assert dpo4_training["completed_epochs"] == 4
    assert dpo4_training["total_optimizer_steps"] == 2224
    assert dpo4_training["n_pairs"] == 2221 and dpo4_training["seed"] == 42
    base = load_jsonl("gen2f_val_rescored.jsonl")
    sft_rows = {
        "1": load_jsonl("rej1_val_rescored.jsonl"),
        "3": load_jsonl("rej3_val_rescored.jsonl"),
    }
    sft = {budget: sft_metrics(rows, constant) for budget, rows in sft_rows.items()}
    for budget, rows in sft_rows.items():
        sft[budget]["descriptive"] = sft_descriptive_metrics(rows)
    paired = {budget: paired_sft(base, rows) for budget, rows in sft_rows.items()}
    anchors = {budget: load(f"rej{budget}_anchor_val.json") for budget in ("1", "3")}
    sft_banks = {
        "1": load("bon_rej1_val_n8.json"),
        "3": load("bon_rej3_val_n8.json"),
    }
    sft_oracle = {budget: oracle_bank_metrics(rows) for budget, rows in sft_banks.items()}

    dpo_banks = {
        "2": load("offline_dpo_val_n8.json"),
        "4": load("offline_dpo_e4_val_n8.json"),
    }
    dpo_archived = {
        "2": load("offline_dpo_results.json"),
        "4": load("offline_dpo_e4_results.json"),
    }
    dpo = {budget: dpo_metrics(rows, constant) for budget, rows in dpo_banks.items()}
    dpo_paired = {budget: paired_bank_comparison(sft_banks["1"], rows) for budget, rows in dpo_banks.items()}
    for budget, metrics in dpo.items():
        archived = dpo_archived[budget]["offline_dpo"]
        metrics["best_of_8"] = oracle_bank_metrics(dpo_banks[budget])
        assert metrics["n_scored"] == archived["n_scored"]
        assert metrics["n_missing_noop"] == archived["n_missing"]
        assert abs(metrics["direction_accuracy"] - archived["dir_acc"]) < 1e-12
        assert abs(metrics["copy_rate"] - archived["copy_rate"]) < 1e-12
        assert abs(metrics["mcnemar_vs_constant"]["p_exact_two_sided"] -
                   archived["mcnemar_vs_constant"]["p_exact_two_sided"]) < 1e-12
        assert metrics["best_of_8"]["n_scored"] == archived["best_of_8"]["n_scored"]
        assert abs(metrics["best_of_8"]["direction_accuracy_scored"] -
                   archived["best_of_8"]["dir_acc_scored_only"]) < 1e-12
        assert abs(metrics["best_of_8"]["direction_accuracy_all_missing_noop"] -
                   archived["best_of_8"]["dir_acc_all_missing_noop"]) < 1e-12
        assert dpo_archived[budget]["training"]["epochs"] == int(budget)
        assert dpo_archived[budget]["training"]["beta"] == 0.1
        assert dpo_archived[budget]["training"]["learning_rate"] == 1e-5

    policy = verify_policy()
    inputs = [
        "training_budget_preregistration.json", "policy_head_baseline_results.json",
        "policy_head_training_budget_results.json", "policy_head_1653232.log", "gen2f_val_rescored.jsonl",
        "rej1_val_rescored.jsonl", "rej3_val_rescored.jsonl",
        "rej1_anchor_val.json", "rej3_anchor_val.json",
        "bon_rej1_val_n8.json", "bon_rej3_val_n8.json",
        "offline_dpo_val_n8.json", "offline_dpo_e4_val_n8.json",
        "offline_dpo_results.json", "offline_dpo_e4_results.json",
        "rej_sft_1649265.log", "rej_sft_1653233.log", "rej_eval_1653234.log",
        "offline_dpo_train_1653235.log", "offline_dpo_eval_1653789.log",
        "offline_dpo_e4_training_summary.json",
    ]
    result = {
        "schema": "pka.training_budget_results.v1",
        "preregistration_sha256": sha256("training_budget_preregistration.json"),
        "artifact_sha256": {name: sha256(name) for name in inputs},
        "constant": constant,
        "policy": policy,
        "rejection_sft": {
            "training_log_records": sft_training,
            "budgets": sft,
            "paired_vs_gen2": paired,
            "best_of_8": sft_oracle,
            "qualitative_checks": {
                "constant_advantage_detected": {
                    budget: metrics["mcnemar_vs_constant"]["p_exact_two_sided"] < 0.05
                    for budget, metrics in sft.items()
                },
                "paired_gen2_direction_improvement_detected": {
                    budget: (metrics["sft_direction_accuracy"] > metrics["gen2_direction_accuracy"] and
                             metrics["mcnemar_vs_gen2"]["p_exact_two_sided"] < 0.05)
                    for budget, metrics in paired.items()
                },
                "paired_gen2_mae_interval_excludes_zero": {
                    budget: metrics["pair_bootstrap_95ci"][1] < 0
                    for budget, metrics in paired.items()
                },
            },
            "source_anchor": {
                budget: {
                    "source_gt_mae_change": anchors[budget]["source_gt_mae_change"],
                    "pass_change_at_most_0.3": anchors[budget]["anchor_pass"],
                } for budget in anchors
            },
            "evaluation_protocol": prereg["rejection_sft_3"]["evaluation_protocol"],
        },
        "offline_dpo": {
            "four_epoch_training": {key: value for key, value in dpo4_training.items() if key != "history"},
            "budgets": dpo,
            "paired_vs_rej1": dpo_paired,
            "pass_criteria": {budget: dpo_archived[budget]["pass_criteria"] for budget in dpo},
            "qualitative_checks": {
                "all_pass": {budget: dpo_archived[budget]["all_pass"] for budget in dpo},
                "changed_criteria_from_two_epochs": [
                    key for key in dpo_archived["2"]["pass_criteria"]
                    if dpo_archived["2"]["pass_criteria"][key] != dpo_archived["4"]["pass_criteria"][key]
                ],
            },
            "evaluation_protocol": prereg["offline_dpo_4"]["evaluation_protocol"],
        },
        "interpretation_rule": prereg["common"]["decision_rule"],
    }
    with open(os.path.join(DATA, "training_budget_results.json"), "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Fixed training-budget ablations. SFT uses its original \texttt{eval\_gen} protocol; DPO uses the first sampled candidate. Both retain all 255 rows with missing predictions as source/no-op and use the fixed train-selected constant 3.74. $p$ is the exact two-sided McNemar test against that constant; cross-protocol rankings are not implied.}",
        r"\label{tab:training_budget}", r"\small",
        r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
        r"Model & Epochs & Scored & Direction & $p$ vs constant & Copy rate \\",
        r"\midrule",
    ]
    for budget in ("1", "3"):
        metrics = sft[budget]
        lines.append(f"Rejection SFT & {budget} & {metrics['n_scored']}/255 & {metrics['direction_accuracy']:.4f} & "
                     f"{metrics['mcnemar_vs_constant']['p_exact_two_sided']:.4g} & {metrics['copy_rate']:.4f} \\\\")
    lines.append(r"\midrule")
    for budget in ("2", "4"):
        metrics = dpo[budget]
        lines.append(f"Offline DPO & {budget} & {metrics['n_scored']}/255 & {metrics['direction_accuracy']:.4f} & "
                     f"{metrics['mcnemar_vs_constant']['p_exact_two_sided']:.4g} & {metrics['copy_rate']:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(TABLES, "table_training_budget.tex"), "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print(json.dumps({
        "policy": [{key: value for key, value in seed.items() if key != "val_history"} for seed in policy["seeds"]],
        "sft": sft,
        "sft_paired": paired,
        "dpo": dpo,
    }, indent=2))


if __name__ == "__main__":
    main()
