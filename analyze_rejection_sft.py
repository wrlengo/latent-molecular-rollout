"""Paired validation analysis for oracle-best rejection SFT."""
import csv
import hashlib
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TABLES = os.path.join(ROOT, "tables")
RNG = np.random.default_rng(42)


def file_sha256(name):
    with open(os.path.join(DATA, name), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_jsonl(name):
    with open(os.path.join(DATA, name)) as f:
        return [json.loads(line) for line in f if line.strip()]


def direction_ok(src, tgt, pred):
    return (pred - src) * (tgt - src) > 0


def best_constant(rows):
    src = np.asarray([r["src_pka"] for r in rows])
    tgt = np.asarray([r["tgt_pka"] for r in rows])
    candidates = np.arange(math.floor(tgt.min() - 1) * 100,
                           math.ceil(tgt.max() + 1) * 100 + 1) / 100
    direction = np.asarray([[direction_ok(s, t, c) for s, t in zip(src, tgt)] for c in candidates])
    mae = np.mean(np.abs(tgt[None, :] - candidates[:, None]), axis=1)
    best_d = int(np.argmax(direction.mean(axis=1)))
    best_m = int(np.argmin(mae))
    return {"direction_accuracy": float(direction[best_d].mean()), "direction_constant": float(candidates[best_d]),
            "mae": float(mae[best_m]), "mae_constant": float(candidates[best_m])}


def single_metrics(rows):
    scored = [r for r in rows if r.get("pred_pka") is not None]
    tanimoto = [r["tanimoto"] for r in rows if r.get("tanimoto") is not None]
    return {
        "n": len(rows), "n_scored": len(scored),
        "unique_gen": len({r["gen"] for r in rows if r.get("valid") and r.get("gen") is not None}),
        "validity": float(np.mean([bool(r["valid"]) for r in rows])),
        "tanimoto": float(np.mean(tanimoto)),
        "direction_accuracy": float(np.mean([direction_ok(r["src_pka"], r["tgt_pka"], r["pred_pka"]) for r in scored])),
        "mae_vs_target": float(np.mean([abs(r["pred_pka"] - r["tgt_pka"]) for r in scored])),
        "copy_rate": float(np.mean([bool(r["copy"]) for r in rows])),
        "best_constant_scored_subset": best_constant(scored),
    }


def exact_mcnemar(base_ok, new_ok):
    base_only = int(np.sum(base_ok & ~new_ok))
    new_only = int(np.sum(~base_ok & new_ok))
    n = base_only + new_only
    if n == 0:
        return {"base_only_correct": 0, "new_only_correct": 0, "two_sided_exact_p": 1.0}
    tail = sum(math.comb(n, k) for k in range(min(base_only, new_only) + 1)) / 2**n
    return {"base_only_correct": base_only, "new_only_correct": new_only,
            "two_sided_exact_p": float(min(1.0, 2 * tail))}


def paired_comparison(base_rows, new_rows):
    base = {r["i"]: r for r in base_rows}
    new = {r["i"]: r for r in new_rows}
    assert base.keys() == new.keys()
    paired = []
    for i in sorted(base):
        b, n = base[i], new[i]
        assert b["src"] == n["src"] and b["src_pka"] == n["src_pka"] and b["tgt_pka"] == n["tgt_pka"]
        if b.get("pred_pka") is not None and n.get("pred_pka") is not None:
            paired.append((b, n))
    base_ok = np.asarray([direction_ok(b["src_pka"], b["tgt_pka"], b["pred_pka"]) for b, _ in paired])
    new_ok = np.asarray([direction_ok(n["src_pka"], n["tgt_pka"], n["pred_pka"]) for _, n in paired])
    delta_absolute_error = np.asarray([abs(n["pred_pka"] - n["tgt_pka"]) - abs(b["pred_pka"] - b["tgt_pka"])
                                       for b, n in paired])
    samples = RNG.choice(delta_absolute_error, size=(10000, len(paired)), replace=True).mean(axis=1)
    return {"n_both_scored": len(paired),
            "base_direction_accuracy": float(base_ok.mean()), "new_direction_accuracy": float(new_ok.mean()),
            "mcnemar": exact_mcnemar(base_ok, new_ok),
            "base_mae": float(np.mean([abs(b["pred_pka"] - b["tgt_pka"]) for b, _ in paired])),
            "new_mae": float(np.mean([abs(n["pred_pka"] - n["tgt_pka"]) for _, n in paired])),
            "new_minus_base_mae": float(delta_absolute_error.mean()),
            "new_minus_base_mae_pair_bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
            "ci_limit": "Pair-level bootstrap on the validation pairs; not an external replication."}


def select_train_constant(rows):
    best_accuracy = -1.0
    plateau = []
    for index in range(2001):
        value = round(-5.0 + 0.01 * index, 2)
        accuracy = float(np.mean([direction_ok(r["src_pka"], r["tgt_pka"], value) for r in rows]))
        if accuracy > best_accuracy + 1e-12:
            best_accuracy = accuracy
            plateau = [value]
        elif abs(accuracy - best_accuracy) <= 1e-12:
            plateau.append(value)
    return plateau[0], best_accuracy, [plateau[0], plateau[-1]]


def all_row_constant_comparison(rows, value):
    predictions = [r["pred_pka"] if r.get("pred_pka") is not None else r["src_pka"] for r in rows]
    model_ok = np.asarray([direction_ok(r["src_pka"], r["tgt_pka"], p) for r, p in zip(rows, predictions)])
    constant_ok = np.asarray([direction_ok(r["src_pka"], r["tgt_pka"], value) for r in rows])
    test = exact_mcnemar(constant_ok, model_ok)
    return {
        "n": len(rows),
        "n_scored": sum(r.get("pred_pka") is not None for r in rows),
        "n_missing_noop": sum(r.get("pred_pka") is None for r in rows),
        "n_correct": int(model_ok.sum()),
        "direction_accuracy": float(model_ok.mean()),
        "mae_all_rows_missing_noop": float(np.mean([abs(p - r["tgt_pka"]) for r, p in zip(rows, predictions)])),
        "constant_only_correct": test["base_only_correct"],
        "model_only_correct": test["new_only_correct"],
        "two_sided_exact_p": test["two_sided_exact_p"],
    }


def bank_as_single_rows(rows):
    return [{
        "src": r["src"],
        "src_pka": r["src_pka"],
        "tgt_pka": r["tgt_pka"],
        "pred_pka": float(r["scored"][0][1]) if r["scored"] else None,
    } for r in rows]


def wilson_interval(n_correct, n):
    z = 1.959963984540054
    proportion = n_correct / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def bon_metrics(rows):
    selected, first, random_selected = [], [], []
    for i, row in enumerate(rows):
        local_rng = np.random.default_rng(i)
        if row["scored"]:
            chosen = min(row["scored"], key=lambda item: abs(item[1] - row["tgt_pka"]))
            selected.append(chosen)
            first.append(row["scored"][0])
            random_selected.append(row["scored"][local_rng.integers(len(row["scored"]))])
        else:
            # Match best_of_n.py: no valid candidate is a failed direction and a no-op MAE.
            noop = (row["src"], row["src_pka"])
            selected.append(noop)
            first.append(noop)
            random_selected.append(noop)

    def summarize(items):
        return {"direction_accuracy": float(np.mean([direction_ok(r["src_pka"], r["tgt_pka"], p)
                                                       for r, (_, p) in zip(rows, items)])),
                "mae_vs_target": float(np.mean([abs(p - r["tgt_pka"]) for r, (_, p) in zip(rows, items)]))}
    return {"n": len(rows), "n_with_valid_candidate": sum(bool(row["scored"]) for row in rows),
            "oracle_at_8": summarize(selected), "first_candidate": summarize(first),
            "random_candidate_seeded_by_row": summarize(random_selected)}


train_candidate_names = [f"candidates/bon_gen2f_train_n8_shard{shard}.json" for shard in range(4)]
input_names = ["gen2f_val_rescored.jsonl", "rej1_val_rescored.jsonl", "bon_gen2f_val_n8.json",
               "bon_rej1_val_n8.json", "rej1_anchor_val.json", "rejection_sft_train_summary.json",
               "offline_dpo_results.json", "offline_dpo_val_n8.json"] + train_candidate_names
base_rows = load_jsonl("gen2f_val_rescored.jsonl")
new_rows = load_jsonl("rej1_val_rescored.jsonl")
base_bon = json.load(open(os.path.join(DATA, "bon_gen2f_val_n8.json")))
new_bon = json.load(open(os.path.join(DATA, "bon_rej1_val_n8.json")))
dpo_bank = json.load(open(os.path.join(DATA, "offline_dpo_val_n8.json")))
dpo_summary = json.load(open(os.path.join(DATA, "offline_dpo_results.json")))
anchor = json.load(open(os.path.join(DATA, "rej1_anchor_val.json")))
train_summary = json.load(open(os.path.join(DATA, "rejection_sft_train_summary.json")))
train_candidates = []
for name in train_candidate_names:
    train_candidates.extend(json.load(open(os.path.join(DATA, name))))
assert len(train_candidates) == 2295
constant, train_constant_accuracy, constant_plateau = select_train_constant(train_candidates)
assert constant == 3.74 and constant_plateau == [3.74, 3.74]
assert abs(train_constant_accuracy - dpo_summary["primary_constant_protocol"]["train_dir_acc"]) < 1e-12
constant_ok = np.asarray([direction_ok(r["src_pka"], r["tgt_pka"], constant) for r in new_rows])
constant_protocol = {
    "constant": constant,
    "selected_on": "2,295 training candidate rows only; 0.01 grid from -5 to 15; lower plateau edge",
    "train_direction_accuracy": train_constant_accuracy,
    "best_plateau": constant_plateau,
    "n": len(new_rows),
    "missing_prediction_rule": "source/no-op; retained in all 255 denominators",
    "constant_n_correct": int(constant_ok.sum()),
    "constant_direction_accuracy": float(constant_ok.mean()),
    "gen2_final": all_row_constant_comparison(base_rows, constant),
    "oracle_best_sft": all_row_constant_comparison(new_rows, constant),
}
rej1_bank_rows = bank_as_single_rows(new_bon)
dpo_bank_rows = bank_as_single_rows(dpo_bank)
for reference, bank_rows in ((new_rows, rej1_bank_rows), (new_rows, dpo_bank_rows)):
    assert len(reference) == len(bank_rows) == 255
    for source, generated in zip(reference, bank_rows):
        assert source["src"] == generated["src"]
        assert source["src_pka"] == generated["src_pka"] and source["tgt_pka"] == generated["tgt_pka"]
rej1_bank_comparison = all_row_constant_comparison(rej1_bank_rows, constant)
dpo_comparison = all_row_constant_comparison(dpo_bank_rows, constant)
assert rej1_bank_comparison["n_scored"] == dpo_summary["baseline_rej1_recomputed"]["n_scored"]
assert dpo_comparison["n_scored"] == dpo_summary["offline_dpo"]["n_scored"]
assert dpo_comparison["n_missing_noop"] == dpo_summary["offline_dpo"]["n_missing"]
for computed, archived in ((rej1_bank_comparison, dpo_summary["baseline_rej1_recomputed"]),
                           (dpo_comparison, dpo_summary["offline_dpo"])):
    assert computed["constant_only_correct"] == archived["mcnemar_vs_constant"]["constant_only_correct"]
    assert computed["model_only_correct"] == archived["mcnemar_vs_constant"]["model_only_correct"]
assert abs(rej1_bank_comparison["direction_accuracy"] - dpo_summary["baseline_rej1_recomputed"]["dir_acc"]) < 1e-12
assert abs(dpo_comparison["direction_accuracy"] - dpo_summary["offline_dpo"]["dir_acc"]) < 1e-12
assert abs(dpo_comparison["two_sided_exact_p"] - dpo_summary["offline_dpo"]["mcnemar_vs_constant"]["p_exact_two_sided"]) < 1e-12
rej1_ok = np.asarray([direction_ok(r["src_pka"], r["tgt_pka"], r["pred_pka"] if r["pred_pka"] is not None else r["src_pka"])
                      for r in rej1_bank_rows])
dpo_ok = np.asarray([direction_ok(r["src_pka"], r["tgt_pka"], r["pred_pka"] if r["pred_pka"] is not None else r["src_pka"])
                     for r in dpo_bank_rows])
dpo_protocol_crosscheck = {
    "rej1_from_raw_bank": rej1_bank_comparison,
    "offline_dpo_from_raw_bank": dpo_comparison,
    "dpo_vs_rej1_mcnemar": exact_mcnemar(rej1_ok, dpo_ok),
    "dpo_direction_wilson_ci95": wilson_interval(dpo_comparison["n_correct"], 255),
    "reporting_rule": "0.72 is a design reference, not a statistical pass/fail threshold. The primary inferential test is exact McNemar against the train-selected constant; copy rate remains a design-bound diagnostic.",
}
result = {
    "interpretation": "Oracle-best distillation, not direction-correct-only rejection SFT; all candidates and final scoring use the repaired base pKa oracle.",
    "direction_definition": "strict (pred_pka-src_pka)*(tgt_pka-src_pka)>0; an exact source/no-op prediction is always direction-incorrect",
    "baseline_clarification": "0.642553 is gen2-final single-sample accuracy, not no-op accuracy; the paired p=0.00418 and MAE CI compare SFT with gen2-final, not with no-op.",
    "no_op_reference": {
        "n": len(new_rows), "direction_accuracy": None,
        "direction_note": "N/A by reporting convention; the strict direction predicate is false for an exact source prediction",
        "mae_all_255": float(np.mean([abs(r["src_pka"] - r["tgt_pka"]) for r in new_rows])),
        "mae_gen2_scored_235": float(np.mean([abs(r["src_pka"] - r["tgt_pka"]) for r in base_rows if r.get("pred_pka") is not None])),
        "mae_sft_scored_252": float(np.mean([abs(r["src_pka"] - r["tgt_pka"]) for r in new_rows if r.get("pred_pka") is not None])),
    },
    "artifact_sha256": {name: file_sha256(name) for name in input_names},
    "checkpoints": {"gen2_final": anchor["models"]["baseline"]["ckpt"],
                    "oracle_best_sft": anchor["models"]["rejection_sft"]["ckpt"]},
    "training_selection": train_summary,
    "single_sample": {"gen2_final": single_metrics(base_rows), "oracle_best_sft": single_metrics(new_rows),
                      "paired_comparison": paired_comparison(base_rows, new_rows),
                      "train_selected_constant_all_255": constant_protocol},
    "dpo_protocol_crosscheck": dpo_protocol_crosscheck,
    "best_of_8": {"gen2_final": bon_metrics(base_bon), "oracle_best_sft": bon_metrics(new_bon)},
    "source_anchor": {"baseline_mae": anchor["models"]["baseline"]["source_gt_mae"],
                      "new_mae": anchor["models"]["rejection_sft"]["source_gt_mae"],
                      "new_minus_baseline": anchor["source_gt_mae_change"],
                      "pass_change_at_most_0.3": anchor["anchor_pass"]},
}
# Anchor this analysis to the previously verified gen2-final outputs.
assert abs(result["single_sample"]["gen2_final"]["direction_accuracy"] - 0.643) < 0.002
assert abs(result["single_sample"]["gen2_final"]["mae_vs_target"] - 2.424) < 0.002
assert abs(result["best_of_8"]["gen2_final"]["oracle_at_8"]["direction_accuracy"] - 0.753) < 0.002
assert abs(result["best_of_8"]["gen2_final"]["oracle_at_8"]["mae_vs_target"] - 1.500) < 0.002
with open(os.path.join(DATA, "rejection_sft_results.json"), "w") as f:
    json.dump(result, f, indent=2)
rows = []
for model in ("gen2_final", "oracle_best_sft"):
    metrics = result["single_sample"][model]
    rows.append([model, metrics["n"], metrics["n_scored"], metrics["unique_gen"], metrics["validity"], metrics["tanimoto"],
                 metrics["direction_accuracy"], metrics["mae_vs_target"],
                 metrics["best_constant_scored_subset"]["direction_accuracy"],
                 metrics["best_constant_scored_subset"]["mae"]])
with open(os.path.join(TABLES, "table_rejection_sft_comparison.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "n", "n_scored", "unique_valid_gen", "validity", "tanimoto", "direction_accuracy",
                     "mae_vs_target", "best_constant_direction_accuracy", "best_constant_mae"])
    writer.writerows(rows)
constant_result = result["single_sample"]["train_selected_constant_all_255"]
with open(os.path.join(TABLES, "table_rejection_sft_constant_mcnemar.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "constant", "n", "n_scored", "n_missing_noop", "n_correct", "direction_accuracy",
                     "constant_n_correct", "constant_direction_accuracy", "constant_only_correct",
                     "model_only_correct", "two_sided_exact_p", "mae_all_rows_missing_noop"])
    for model in ("gen2_final", "oracle_best_sft"):
        metrics = constant_result[model]
        writer.writerow([model, constant_result["constant"], metrics["n"], metrics["n_scored"],
                         metrics["n_missing_noop"], metrics["n_correct"], metrics["direction_accuracy"],
                         constant_result["constant_n_correct"], constant_result["constant_direction_accuracy"],
                         metrics["constant_only_correct"], metrics["model_only_correct"],
                         metrics["two_sided_exact_p"], metrics["mae_all_rows_missing_noop"]])
print(json.dumps(result, indent=2))
