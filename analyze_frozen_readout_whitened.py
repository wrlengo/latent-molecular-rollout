"""Sensitivity analysis of frozen-h ridge alignment in z-scored latent space."""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RNG = np.random.default_rng(42)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def make_pairs(smiles, targets):
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
                    for s in smiles]
    pairs = []
    for i in range(len(smiles)):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1:])
        for offset, similarity in enumerate(similarities, start=i + 1):
            if similarity < 0.6 or abs(float(targets[i] - targets[offset])) < 1.0:
                continue
            low, high = (i, offset) if targets[i] < targets[offset] else (offset, i)
            pairs.append({"low": low, "high": high})
    return pairs


def cosines(w, h, pairs, scale=None):
    values = []
    local_w = w if scale is None else w * scale
    for pair in pairs:
        d = h[pair["high"]] - h[pair["low"]]
        if scale is not None:
            d = d / scale
        values.append(float(np.dot(local_w, d) / (np.linalg.norm(local_w) * np.linalg.norm(d))))
    return np.asarray(values)


def summarize(values):
    boot = np.median(RNG.choice(values, size=(10000, len(values)), replace=True), axis=1)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "median_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-latents", default=os.path.join(DATA, "h_train.npz"))
    args = parser.parse_args()
    train_path = args.train_latents
    test_path = os.path.join(DATA, "h_test.npz")
    weights_path = os.path.join(DATA, "ridge_selected_weights.npz")
    result_path = os.path.join(DATA, "frozen_ridge_results.json")
    pairs_path = os.path.join(DATA, "frozen_ridge_test_pairs.csv")
    projection_path = os.path.join(DATA, "projection_diagnosis.json")
    input_paths = {"h_train.npz": train_path, "h_test.npz": test_path,
                   "ridge_selected_weights.npz": weights_path, "frozen_ridge_results.json": result_path,
                   "frozen_ridge_test_pairs.csv": pairs_path, "projection_diagnosis.json": projection_path}
    hashes = {name: sha256(path) for name, path in input_paths.items()}

    projection = json.load(open(projection_path))
    assert projection["method"]["latent_sha256"] == hashes["h_test.npz"]
    original_result = json.load(open(result_path))
    with open(pairs_path) as f:
        test_rows = list(csv.DictReader(f))
    test_pairs = [{"low": int(r["low"]), "high": int(r["high"])} for r in test_rows]
    weights = np.load(weights_path)
    fit_idx = weights["fit_indices"]
    w = weights["w_original"].astype(np.float64)
    b = float(weights["intercept_original"])

    train = np.load(train_path, allow_pickle=True)
    test = np.load(test_path, allow_pickle=True)
    fit32 = train["h"][fit_idx].astype(np.float32)
    mean = fit32.mean(axis=0, dtype=np.float64)
    scale = fit32.std(axis=0, dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    fit_h = fit32.astype(np.float64)
    test_h = test["h"].astype(np.float64)
    fit_pairs = make_pairs([str(train["smiles"][i]) for i in fit_idx], train["gt"][fit_idx])
    assert len(fit_pairs) == original_result["counts"]["fit_pairs"] == 381
    assert len(test_pairs) == original_result["counts"]["test_pairs"] == 43

    pred_test = test_h @ w + b
    test_mae = float(np.mean(np.abs(pred_test - test["gt"].astype(np.float64))))
    assert abs(test_mae - original_result["mae"]["test"]) < 1e-6
    original = {
        "fit": cosines(w, fit_h, fit_pairs),
        "test": cosines(w, test_h, test_pairs),
    }
    for split in ("fit", "test"):
        saved = original_result["ridge_low_to_high_cosine"][split]
        assert abs(np.median(original[split]) - saved["median"]) < 1e-12
        assert abs(np.mean(original[split]) - saved["mean"]) < 1e-12
    assert np.max(np.abs(original["test"] - np.asarray([float(r["ridge_cosine"]) for r in test_rows]))) < 1e-12

    whitened = {
        "fit": cosines(w, fit_h, fit_pairs, scale),
        "test": cosines(w, test_h, test_pairs, scale),
    }
    # Numerically, w_z·d_z equals w·d; only the cosine normalization changes.
    for split, pairs in (("fit", fit_pairs), ("test", test_pairs)):
        h = fit_h if split == "fit" else test_h
        for pair in pairs:
            d = h[pair["high"]] - h[pair["low"]]
            assert abs(np.dot(w * scale, d / scale) - np.dot(w, d)) < 1e-7

    original_summary = {split: summarize(original[split]) for split in ("fit", "test")}
    whitened_summary = {split: summarize(whitened[split]) for split in ("fit", "test")}
    median = whitened_summary["test"]["median"]
    if median >= 0.2:
        category = "whitened_test_median_at_least_0.2"
    elif median < 0.05:
        category = "whitened_test_median_below_0.05"
    else:
        category = "whitened_test_median_intermediate"
    energy = np.sort(w ** 2)[::-1]
    result = {
        "protocol": {
            "space": "z = (h-fit_mean)/fit_std; ridge gradient w_z=w_original*fit_std; displacement d_z=(h_high-h_low)/fit_std",
            "selection": "Post-hoc sensitivity analysis of the alpha=100 model selected by validation MAE; no model refitting",
            "decision_rule_for_requested_sensitivity": "test median >=0.2 versus <0.05; intermediate values remain undetermined",
            "metric_limit": "Whitened cosine changes the geometry and is not the original-h G4 cosine. It diagnoses scaling sensitivity and must not be compared as if it were the same metric.",
            "latent_identity": "The test latent SHA-256 must equal the value recorded in projection_diagnosis.json.",
        },
        "artifact_sha256": hashes,
        "counts": {"fit": len(fit_idx), "test": len(test_h), "fit_pairs": len(fit_pairs), "test_pairs": len(test_pairs)},
        "mae_test_unchanged": test_mae,
        "top2_original_h_weight_energy_fraction": float(energy[:2].sum() / energy.sum()),
        "original_h_cosine": original_summary,
        "whitened_cosine": whitened_summary,
        "decision": {"category": category,
                     "original_h_conclusion": "undetermined_due_to_extreme_weight_concentration",
                     "note": "This sensitivity result does not by itself prove or disprove direction recoverability in original h."},
    }
    out_json = os.path.join(DATA, "frozen_ridge_whitened_cosine.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    out_pairs = os.path.join(DATA, "frozen_ridge_whitened_pairs.csv")
    with open(out_pairs, "w", newline="") as f:
        fields = ["split", "low", "high", "original_h_cosine", "whitened_cosine",
                  "latent_sha256", "weights_sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split, pairs in (("fit", fit_pairs), ("test", test_pairs)):
            for i, pair in enumerate(pairs):
                writer.writerow({"split": split, **pair,
                                 "original_h_cosine": original[split][i],
                                 "whitened_cosine": whitened[split][i],
                                 "latent_sha256": hashes["h_train.npz" if split == "fit" else "h_test.npz"],
                                 "weights_sha256": hashes["ridge_selected_weights.npz"]})
    out_weights = os.path.join(DATA, "ridge_whitened_weights.npz")
    np.savez_compressed(out_weights, w_z=w * scale, fit_mean=mean, fit_scale=scale,
                        fit_indices=fit_idx, train_latent_sha256=hashes["h_train.npz"],
                        test_latent_sha256=hashes["h_test.npz"])
    print(json.dumps(result, indent=2))
    print(f"saved {out_json}\nsaved {out_pairs}\nsaved {out_weights}")


if __name__ == "__main__":
    main()
