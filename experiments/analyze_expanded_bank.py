"""Compare policy direction metrics across the original 43-pair and expanded 151-pair banks.

The expanded bank is a strict superset of the original 43 and preserves their
relative order, so the per-pair cosine array of the expanded run can be sliced
into the original 43, the new 108, and the pooled 151 under one trained policy.
"""
import argparse
import csv
import json

import numpy as np

RNG = np.random.default_rng(0)


def ci(values, n_boot=10000):
    values = np.asarray(values, dtype=float)
    boot = np.median(RNG.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def summarise(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "median_ci95": ci(values),
        "frac_above_0.3": float(np.mean(values > 0.3)),
        "n_above_0.3": int((values > 0.3).sum()),
        "frac_above_0.5": float(np.mean(values > 0.5)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded", default="data/policy_head_expanded_results.json")
    ap.add_argument("--archived", default="data/policy_head_baseline_results.json")
    ap.add_argument("--repro", default="data/policy_head_baseline_repro_results.json")
    ap.add_argument("--expanded-pairs", default="data/policy_amplitude_readout_pairs_expanded.csv")
    ap.add_argument("--original-pairs", default="data/policy_amplitude_readout_pairs.csv")
    ap.add_argument("--out", default="data/policy_expanded_bank_analysis.json")
    args = ap.parse_args()

    exp = json.load(open(args.expanded))
    arc = json.load(open(args.archived))
    expanded_keys = [(r["src_smiles"], r["tgt_smiles"])
                     for r in csv.DictReader(open(args.expanded_pairs))]
    original_keys = [(r["src_smiles"], r["tgt_smiles"])
                     for r in csv.DictReader(open(args.original_pairs))]

    pos = [expanded_keys.index(k) for k in original_keys]
    assert pos == sorted(pos), "original pairs are not a contiguous-order subsequence"
    cos_all = np.asarray(exp["ensemble"]["test_cos"], dtype=float)
    assert len(cos_all) == len(expanded_keys)
    cos_orig = cos_all[pos]
    cos_new = np.delete(cos_all, pos)

    out = {
        "policy_recipe_unchanged": "same --pairs, seeds 0-2, 150 epochs, patience 25; only --test_pairs differs",
        "expanded_run_n_test_pairs": exp["n_test_pairs"],
        "original_bank_from_archived_run": summarise(np.asarray(arc["ensemble"]["test_cos"], dtype=float)),
        "original_bank_from_expanded_run": summarise(cos_orig),
        "new_pairs_only": summarise(cos_new),
        "pooled_expanded_bank": summarise(cos_all),
        "archived_amplitude_43": arc["ensemble"]["learned_amplitude_test"],
        "expanded_amplitude_151": exp["ensemble"]["learned_amplitude_test"],
        "archived_small_gap": {"n": arc["ensemble"]["test_small_gap_n"],
                               "median": arc["ensemble"]["test_small_gap_median_cos"]},
        "expanded_small_gap": {"n": exp["ensemble"]["test_small_gap_n"],
                               "median": exp["ensemble"]["test_small_gap_median_cos"]},
        "g4_reference_median_cos_archived_pairset": exp["g4_baseline_median_cos"],
    }
    try:
        repro = json.load(open(args.repro))
        out["baseline_reproducibility"] = {
            "archived_ensemble_median_43": arc["ensemble"]["test_median_cos"],
            "repro_ensemble_median_43": repro["ensemble"]["test_median_cos"],
            "abs_diff": abs(repro["ensemble"]["test_median_cos"]
                            - arc["ensemble"]["test_median_cos"]),
        }
    except FileNotFoundError:
        out["baseline_reproducibility"] = None
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
