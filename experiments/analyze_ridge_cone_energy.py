"""Where does the selected ridge weight live, relative to the displacement cone?

Zero-cost linear algebra on archived artifacts: the fit is done, this script only
projects. It tests the mechanism hypothesis that least squares drives the readout
weight into high-variance directions while the displacement cone occupies the
complement, i.e. that the ridge is not merely failing to hit the cone at random
but is structurally pushed out of it.
"""
import argparse
import csv
import json

import numpy as np


def load_displacements(h, smiles, pair_csv, keys=None):
    idx = {s: i for i, s in enumerate(smiles)}
    unit, raw = [], []
    for row in csv.DictReader(open(pair_csv)):
        key = (row["src_smiles"], row["tgt_smiles"])
        if keys is not None and key not in keys:
            continue
        d = h[idx[row["tgt_smiles"]]].astype(np.float64) - h[idx[row["src_smiles"]]].astype(np.float64)
        raw.append(d)
        unit.append(d / np.linalg.norm(d))
    return np.asarray(unit), np.asarray(raw)


def energy_fraction(w, basis):
    """Fraction of ||w||^2 inside span(basis); basis rows are orthonormal."""
    proj = basis @ w
    return float((proj ** 2).sum() / (w ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca", default="data/pca_direction_fit.npz")
    ap.add_argument("--ridge", default="data/ridge_selected_weights.npz")
    ap.add_argument("--h-test", default="data/h_test.npz")
    ap.add_argument("--pairs-43", default="data/policy_amplitude_readout_pairs.csv")
    ap.add_argument("--pairs-151", default="data/policy_amplitude_readout_pairs_expanded.csv")
    ap.add_argument("--out", default="data/ridge_cone_energy.json")
    args = ap.parse_args()

    pca = np.load(args.pca, allow_pickle=True)
    P = pca["pca90_components"].astype(np.float64)      # (6, 4096), orthonormal rows
    k = int(pca["k90"])
    cum = pca["cumulative_variance_ratio"]
    w = np.load(args.ridge, allow_pickle=True)["w_original"].astype(np.float64)
    D = w.size

    lat = np.load(args.h_test, allow_pickle=True)
    h, smiles = lat["h"], list(lat["smiles"])

    rows43 = list(csv.DictReader(open(args.pairs_43)))
    keys43 = {(r["src_smiles"], r["tgt_smiles"]) for r in rows43}
    u43, d43 = load_displacements(h, smiles, args.pairs_43)
    u151, d151 = load_displacements(h, smiles, args.pairs_151)

    out = {"latent_dim": D, "pca_components_kept": k,
           "pca_cumulative_variance_top_k": float(cum[k - 1]),
           "random_reference_energy_fraction_top_k": k / D}

    out["w_energy_fraction_in_top_k_pca"] = energy_fraction(w, P)
    out["w_energy_top_coordinate_share"] = float(
        np.sort(w ** 2)[-1] / (w ** 2).sum())
    out["w_energy_top_two_coordinate_share"] = float(
        np.sort(w ** 2)[-2:].sum() / (w ** 2).sum())

    for tag, u, d in (("pairs43", u43, d43), ("pairs151", u151, d151)):
        # Orthonormal basis for the displacement span (rank is small).
        q, _ = np.linalg.qr(u.T)
        r = np.linalg.matrix_rank(u)
        basis = q[:, :r].T
        # Variance of the unit displacements captured by the PCA top-k subspace.
        proj = P @ u.T                                   # (k, n)
        var_total = float((u ** 2).sum())
        out[tag] = {
            "n_pairs": int(u.shape[0]),
            "displacement_span_rank": int(r),
            "w_energy_fraction_in_cone": energy_fraction(w, basis),
            "random_reference_energy_fraction_in_cone": r / D,
            "unit_displacement_energy_fraction_in_top_k_pca": float((proj ** 2).sum() / var_total),
            "mean_cos_w_with_displacements": float(np.mean((w @ u.T) / np.linalg.norm(w))),
        }

    # Robustness: drop the two dominant coordinates of w and repeat.
    drop = np.argsort(-np.abs(w))[:2]
    keep = np.setdiff1d(np.arange(D), drop)
    w2 = w[keep]
    P2 = P[:, keep]
    out["excluding_top2_w_coordinates"] = {
        "latent_dim": int(w2.size),
        "w_energy_fraction_in_top_k_pca": energy_fraction(w2, P2),
    }
    for tag, u in (("pairs43", u43), ("pairs151", u151)):
        q, _ = np.linalg.qr(u[:, keep].T)
        r = np.linalg.matrix_rank(u[:, keep])
        out["excluding_top2_w_coordinates"][tag] = {
            "displacement_span_rank": int(r),
            "w_energy_fraction_in_cone": energy_fraction(w2, q[:, :r].T),
            "random_reference_energy_fraction_in_cone": r / w2.size,
        }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
