"""Fit a single global amplitude scale on training pairs and export it for test use.

The amplitude reread shows relative ordering under the PkaHead-gradient readout but
a compressed scale. This script asks whether one global multiplicative constant is
enough: fit sigma on training pairs only, then let the caller apply it to the held-out
43 pairs and report per-pair scale error.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from policy_head import PolicyHead, load_npz, load_pairs
from policy_amplitude_readout import build_head, head_gradients, load_policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_train", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--policy_models", required=True)
    ap.add_argument("--pka_head_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    h_train, smi_train, gt_train = load_npz(args.h_train)
    smi2tr = {str(s): i for i, s in enumerate(smi_train)}
    pairs = [p for p in load_pairs(args.pairs)
             if p["src_smiles"] in smi2tr and p["tgt_smiles"] in smi2tr]
    src = np.asarray([smi2tr[p["src_smiles"]] for p in pairs], dtype=np.int64)
    tgt = np.asarray([smi2tr[p["tgt_smiles"]] for p in pairs], dtype=np.int64)
    delta = np.asarray([p["tgt_pKa"] - p["src_pKa"] for p in pairs], dtype=np.float64)
    src_pka = np.asarray([p["src_pKa"] for p in pairs], dtype=np.float64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = load_policies(args.policy_models, device)
    h_t = torch.from_numpy(h_train[src]).to(device)
    d_t = torch.from_numpy(delta.astype(np.float32)).to(device)
    p_t = torch.from_numpy(src_pka.astype(np.float32)).to(device)
    with torch.no_grad():
        preds = torch.stack([m(h_t, d_t, p_t).float() for m in policies], dim=0).cpu().numpy()
    ens_raw = preds.mean(0).astype(np.float64)

    head, cfg = build_head(args.pka_head_ckpt)
    head.eval()
    g_head = np.empty_like(ens_raw)
    batch = 512
    for start in range(0, len(src), batch):
        stop = min(start + batch, len(src))
        point = torch.tensor(h_train[src[start:stop]], dtype=torch.float32, requires_grad=True)
        value = head(point).sum()
        grad = torch.autograd.grad(value, point)[0].numpy().astype(np.float64)
        g_head[start:stop] = grad

    pred = np.einsum("ij,ij->i", ens_raw, g_head)

    sigma_ls = float((pred @ delta) / (pred @ pred))
    ratio = pred / delta
    sigma_median = float(1.0 / np.median(ratio)) if np.median(ratio) != 0 else None
    res = {
        "protocol": "Global amplitude scale fitted on training pairs only; the amplitude inhibitor (PkaHead) and all policy checkpoints are frozen.",
        "n_train_pairs": int(len(pred)),
        "pka_head_arch": cfg,
        "sigma_least_squares": sigma_ls,
        "sigma_inverse_median_ratio": sigma_median,
        "train": {
            "fraction_positive_delta": float(np.mean(delta > 0)),
            "pearson": float(np.corrcoef(pred, delta)[0, 1]),
            "spearman": float(np.corrcoef(np.argsort(np.argsort(pred)),
                                          np.argsort(np.argsort(delta)))[0, 1]),
            "median_pred": float(np.median(pred)),
            "median_true": float(np.median(delta)),
            "median_ratio": float(np.median(ratio)),
            "frac_ratio_in_2_3_1_5": float(np.mean((ratio > 2 / 3) & (ratio < 1.5))),
            "mae": float(np.mean(np.abs(pred - delta))),
        },
        "note": "Apply sigma to the held-out pairs recorded in data/policy_amplitude_readout_pairs.csv.",
    }
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
