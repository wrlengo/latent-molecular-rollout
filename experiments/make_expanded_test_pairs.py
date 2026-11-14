"""Build the pre-registered expanded held-out pair bank.

Rule (fixed before any policy was scored on it):
  all within-test pairs with |delta pKa| >= 1.0 and Morgan radius-2 2048-bit
  Tanimoto >= 0.5, oriented from lower to higher pKa.

This is the original 43-pair rule with the similarity floor lowered from 0.6 to
0.5, so the expanded bank is a strict superset. Both ridge_cosine and
pka_head_gradient_cosine are recomputed with the archived frozen ridge and
PkaHead so that the expanded CSV reproduces the original columns exactly on the
43 shared pairs.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

ROOT = os.path.expanduser("~/pKa")
CKPT = os.path.join(ROOT, "runs", "v3_C_latent_head_fix", "stage3_sub3")
FIELDS = ["low", "high", "src_smiles", "tgt_smiles", "delta_pka", "tanimoto",
          "ridge_cosine", "pka_head_gradient_cosine"]


def load_test_split():
    rows = json.load(open(os.path.join(ROOT, "pka_sft", "test", "pka_test.json")))
    smiles, targets = [], []
    for row in rows:
        meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"]
        smiles.append(meta["molecule"])
        targets.append(float(meta["gt"]))
    return smiles, np.asarray(targets, dtype=np.float64)


def make_pairs(smiles, targets, tmin, dmin):
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
                    for s in smiles]
    pairs = []
    for low_or_high in range(len(smiles)):
        sims = DataStructs.BulkTanimotoSimilarity(
            fingerprints[low_or_high], fingerprints[low_or_high + 1:])
        for offset, similarity in enumerate(sims, start=low_or_high + 1):
            if similarity < tmin or abs(float(targets[low_or_high] - targets[offset])) < dmin:
                continue
            low, high = ((low_or_high, offset) if targets[low_or_high] < targets[offset]
                         else (offset, low_or_high))
            pairs.append({"low": int(low), "high": int(high), "tanimoto": float(similarity),
                          "delta_pka": float(targets[high] - targets[low])})
    return pairs


def build_head():
    state = torch.load(os.path.join(CKPT, "mm_projector.pt"), map_location="cpu")["pka_head"]
    keys = [key for key in state if key.endswith("weight")]
    d_in, d_mid = state[keys[0]].shape[1], state[keys[0]].shape[0]
    head = torch.nn.Sequential(torch.nn.Linear(d_in, d_mid), torch.nn.GELU(),
                               torch.nn.Linear(d_mid, 1))
    mapped = {}
    for i, key in enumerate(keys):
        name = key.rsplit(".", 1)[0]
        mapped[f"{2 * i}.weight"] = state[name + ".weight"]
        mapped[f"{2 * i}.bias"] = state[name + ".bias"]
    head.load_state_dict(mapped)
    return head.eval()


def head_gradient_cosines(head, h, pairs):
    out = []
    for pair in pairs:
        point = torch.tensor(h[pair["low"]], dtype=torch.float32, requires_grad=True)
        gradient = torch.autograd.grad(head(point.unsqueeze(0)).squeeze(), point)[0].numpy()
        displacement = h[pair["high"]].astype(np.float32) - h[pair["low"]].astype(np.float32)
        out.append(float(np.dot(gradient, displacement)
                         / (np.linalg.norm(gradient) * np.linalg.norm(displacement))))
    return out


def ridge_cosines(w, h, pairs):
    w = w.astype(np.float64)
    wn = np.linalg.norm(w)
    out = []
    for pair in pairs:
        d = h[pair["high"]].astype(np.float64) - h[pair["low"]].astype(np.float64)
        out.append(float(np.dot(w, d) / (wn * np.linalg.norm(d))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmin", type=float, default=0.5)
    ap.add_argument("--dmin", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "experiments", "policy_amplitude_readout_pairs_expanded.csv"))
    ap.add_argument("--summary", default=os.path.join(
        ROOT, "experiments", "expanded_test_pairs_summary.json"))
    ap.add_argument("--check-original", default=os.path.join(
        ROOT, "experiments", "frozen_ridge_test_pairs.csv"))
    args = ap.parse_args()

    smiles, targets = load_test_split()
    latents = np.load(os.path.join(ROOT, "runs", "fig", "h_test.npz"), allow_pickle=True)
    h, gt = latents["h"].astype(np.float32), latents["gt"]
    assert len(smiles) == len(h), "test split / latent count mismatch"
    assert np.allclose(gt.astype(np.float64), targets, atol=1e-5), "pKa order mismatch"

    pairs = make_pairs(smiles, targets, args.tmin, args.dmin)
    w = np.load(os.path.join(ROOT, "experiments", "ridge_selected_weights.npz"),
                allow_pickle=True)["w_original"]
    head = build_head()
    ridge = ridge_cosines(w, h, pairs)
    headcos = head_gradient_cosines(head, h, pairs)

    rows = []
    for pair, rc, hc in zip(pairs, ridge, headcos):
        rows.append({"low": pair["low"], "high": pair["high"],
                     "src_smiles": smiles[pair["low"]], "tgt_smiles": smiles[pair["high"]],
                     "delta_pka": pair["delta_pka"], "tanimoto": pair["tanimoto"],
                     "ridge_cosine": rc, "pka_head_gradient_cosine": hc})

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Reproduce the archived columns on the 43 shared pairs.
    archived = list(csv.DictReader(open(args.check_original)))
    by_key = {(r["src_smiles"], r["tgt_smiles"]): r for r in rows}
    diffs = {"ridge": [], "head": []}
    for row in archived:
        mine = by_key.get((row["src_smiles"], row["tgt_smiles"]))
        assert mine is not None, f"original pair missing from superset: {row['src_smiles']}"
        diffs["ridge"].append(abs(mine["ridge_cosine"] - float(row["ridge_cosine"])))
        diffs["head"].append(abs(mine["pka_head_gradient_cosine"]
                                 - float(row["pka_head_gradient_cosine"])))

    new = [r for r in rows if r["tanimoto"] < 0.6]
    summary = {
        "rule": "within-test pairs, |delta pKa| >= %.3g, Morgan2-2048 Tanimoto >= %.3g, "
                "oriented low to high pKa" % (args.dmin, args.tmin),
        "n_test_states": len(smiles),
        "n_pairs_total": len(rows),
        "n_original_43_retained": len(archived),
        "n_new_pairs": len(new),
        "new_pair_tanimoto_range": [min(r["tanimoto"] for r in new),
                                    max(r["tanimoto"] for r in new)],
        "n_unique_sources_total": len({r["low"] for r in rows} | {r["high"] for r in rows}),
        "max_abs_ridge_cosine_diff_vs_archived": max(diffs["ridge"]),
        "max_abs_head_cosine_diff_vs_archived": max(diffs["head"]),
        "columns": FIELDS,
        "note": "pka_head_gradient_cosine is recomputed for every pair and is used only as a "
                "reference baseline; policy metrics do not read it.",
    }
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
