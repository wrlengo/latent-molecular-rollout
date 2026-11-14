"""Readout-free PCA diagnosis of low-to-high pKa latent displacements."""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from scipy.linalg import svd

from analyze_frozen_readout_whitened import make_pairs

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
RNG = np.random.default_rng(42)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def summarize(values):
    values = np.asarray(values)
    boot = np.median(RNG.choice(values, size=(10000, len(values)), replace=True), axis=1)
    return {"n": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "median_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist()}


def pair_displacements(h, pairs):
    return np.asarray([h[p["high"]] - h[p["low"]] for p in pairs], dtype=np.float64)


def unit_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cosine_to_direction(d, u):
    return (d @ u) / (np.linalg.norm(d, axis=1) * np.linalg.norm(u))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-latents", default=os.path.join(DATA, "h_train.npz"))
    args = parser.parse_args()
    paths = {
        "h_train.npz": args.train_latents,
        "h_test.npz": os.path.join(DATA, "h_test.npz"),
        "ridge_selected_weights.npz": os.path.join(DATA, "ridge_selected_weights.npz"),
        "frozen_ridge_test_pairs.csv": os.path.join(DATA, "frozen_ridge_test_pairs.csv"),
        "projection_diagnosis.json": os.path.join(DATA, "projection_diagnosis.json"),
        "gate4_stratified.json": os.path.join(DATA, "gate4_stratified.json"),
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    projection = json.load(open(paths["projection_diagnosis.json"]))
    assert projection["method"]["latent_sha256"] == hashes["h_test.npz"]
    assert projection["method"]["pair_file_sha256"] == hashes["gate4_stratified.json"]

    weights = np.load(paths["ridge_selected_weights.npz"])
    fit_idx = weights["fit_indices"]
    train = np.load(paths["h_train.npz"], allow_pickle=True)
    test = np.load(paths["h_test.npz"], allow_pickle=True)
    fit_h = train["h"][fit_idx].astype(np.float64)
    test_h = test["h"].astype(np.float64)
    fit_pairs = make_pairs([str(train["smiles"][i]) for i in fit_idx], train["gt"][fit_idx])
    with open(paths["frozen_ridge_test_pairs.csv"]) as f:
        test_rows = list(csv.DictReader(f))
    test_pairs = [{"low": int(r["low"]), "high": int(r["high"])} for r in test_rows]
    assert len(fit_pairs) == 381 and len(test_pairs) == 43
    fit_d = pair_displacements(fit_h, fit_pairs)
    test_d = pair_displacements(test_h, test_pairs)

    print(f"Fitting raw-h PCA on {fit_h.shape} ...", flush=True)
    mean = fit_h.mean(axis=0)
    centered = fit_h - mean
    _, singular, components = svd(centered, full_matrices=False, overwrite_a=True,
                                  check_finite=False, lapack_driver="gesdd")
    explained = singular ** 2 / np.sum(singular ** 2)
    cumulative = np.cumsum(explained)
    k90 = int(np.searchsorted(cumulative, 0.9) + 1)
    pca90 = components[:k90]
    print(f"k90={k90}; cumulative variance={cumulative[k90 - 1]:.6f}", flush=True)

    def project(x):
        return (x @ pca90.T) @ pca90

    fit_d90 = project(fit_d)
    test_d90 = project(test_d)
    fit_norm_fraction = np.linalg.norm(fit_d90, axis=1) / np.linalg.norm(fit_d, axis=1)
    test_norm_fraction = np.linalg.norm(test_d90, axis=1) / np.linalg.norm(test_d, axis=1)

    fit_unit = unit_rows(fit_d)
    test_unit = unit_rows(test_d)
    mean_direction = fit_unit.mean(axis=0)
    mean_direction /= np.linalg.norm(mean_direction)
    mean_direction90 = project(mean_direction)
    mean_direction90 /= np.linalg.norm(mean_direction90)
    direction = {
        "fit_full": cosine_to_direction(fit_d, mean_direction),
        "test_full": cosine_to_direction(test_d, mean_direction),
        "fit_top90": cosine_to_direction(fit_d, mean_direction90),
        "test_top90": cosine_to_direction(test_d, mean_direction90),
    }

    fit_loadings = fit_unit @ components.T
    test_loadings = test_unit @ components.T
    train_mean_loadings = fit_loadings.mean(axis=0)
    ranked = np.argsort(np.abs(train_mean_loadings))[::-1][:30]
    component_rows = []
    for j in ranked:
        signed_test = np.sign(train_mean_loadings[j]) * test_loadings[:, j]
        component_rows.append({
            "variance_rank": int(j + 1), "within_top90": bool(j < k90),
            "explained_variance_ratio": float(explained[j]),
            "cumulative_variance_ratio": float(cumulative[j]),
            "train_mean_signed_unit_loading": float(train_mean_loadings[j]),
            "test_median_signed_unit_loading": float(np.median(test_loadings[:, j])),
            "test_sign_agreement_with_train": float(np.mean(signed_test > 0)),
        })

    result = {
        "protocol": {
            "pca": "Raw-h PCA fit on the same 2,855 fit rows as ridge; no test rows or test labels used; no per-dimension scaling",
            "primary_metric": "||P_90 d||/||d|| per low-to-high pKa displacement; norm fraction <0.2 means less than 20% of displacement norm lies in the top-90% variance subspace",
            "secondary_metric": "Cosine to the mean unit fit-displacement direction, before and after projection onto P_90; test pairs are not used to construct this direction",
            "independence_limit": "The 43 test pairs informed earlier exploratory diagnostics; pair bootstrap does not model pair dependence or external replication",
        },
        "artifact_sha256": hashes,
        "counts": {"fit": len(fit_h), "test": len(test_h), "fit_pairs": len(fit_pairs),
                   "test_pairs": len(test_pairs), "latent_dim": fit_h.shape[1]},
        "pca": {"rank": len(singular), "k90": k90, "variance_captured_at_k90": float(cumulative[k90 - 1]),
                "top1_explained_variance_ratio": float(explained[0]),
                "top10_cumulative_variance_ratio": float(cumulative[9]),
                "top100_cumulative_variance_ratio": float(cumulative[99])},
        "top90_displacement_projection": {
            "fit_norm_fraction": summarize(fit_norm_fraction),
            "test_norm_fraction": summarize(test_norm_fraction),
            "fit_energy_fraction": summarize(fit_norm_fraction ** 2),
            "test_energy_fraction": summarize(test_norm_fraction ** 2),
            "decision": "outside_top90" if np.median(test_norm_fraction) < 0.2 else "represented_in_top90",
        },
        "mean_fit_direction": {
            "top90_norm_fraction_of_unit_direction": float(np.linalg.norm(project(mean_direction))),
            "fit_full_cosine": summarize(direction["fit_full"]),
            "test_full_cosine": summarize(direction["test_full"]),
            "fit_top90_cosine": summarize(direction["fit_top90"]),
            "test_top90_cosine": summarize(direction["test_top90"]),
        },
        "interpretation_limit": "Subspace coverage and directional agreement are distinct. This readout-free analysis does not prove the absence of nonlinear or scaffold-specific directions.",
    }
    fit_artifact = os.path.join(DATA, "pca_direction_fit.npz")
    np.savez_compressed(fit_artifact, pca_mean=mean, pca90_components=pca90,
                        explained_variance_ratio=explained, cumulative_variance_ratio=cumulative,
                        mean_fit_direction=mean_direction, mean_fit_direction_top90=mean_direction90,
                        fit_indices=fit_idx, k90=k90,
                        train_latent_sha256=hashes["h_train.npz"], test_latent_sha256=hashes["h_test.npz"])
    result_path = os.path.join(DATA, "pca_direction_results.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    pair_path = os.path.join(DATA, "pca_direction_test_pairs.csv")
    with open(pair_path, "w", newline="") as f:
        fields = ["low", "high", "top90_norm_fraction", "top90_energy_fraction",
                  "mean_fit_direction_full_cosine", "mean_fit_direction_top90_cosine",
                  "test_latent_sha256", "pca_fit_latent_sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, pair in enumerate(test_pairs):
            writer.writerow({**pair, "top90_norm_fraction": test_norm_fraction[i],
                             "top90_energy_fraction": test_norm_fraction[i] ** 2,
                             "mean_fit_direction_full_cosine": direction["test_full"][i],
                             "mean_fit_direction_top90_cosine": direction["test_top90"][i],
                             "test_latent_sha256": hashes["h_test.npz"],
                             "pca_fit_latent_sha256": hashes["h_train.npz"]})
    component_path = os.path.join(DATA, "pca_direction_component_loadings.csv")
    with open(component_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)
    print(json.dumps(result, indent=2))
    print(f"saved {result_path}\nsaved {pair_path}\nsaved {component_path}")


if __name__ == "__main__":
    main()
