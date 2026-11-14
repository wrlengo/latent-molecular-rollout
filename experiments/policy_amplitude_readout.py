"""Re-measure policy amplitude control with a nonlinear (PkaHead gradient) readout.

Motivation: the manuscript reports amplitude correlation -0.0199 using a frozen
ridge readout whose own direction cosine against true displacements is ~0.002.
If w is nearly orthogonal to the displacement subspace, corr(unit(g) @ w, delta)
is mechanically near zero. This script recomputes the same quantity with the
PkaHead gradient g_head = d PkaHead / d h evaluated at the source latent, and
reports both readouts side by side.

Pre-registered decision rule (fixed before execution):
  * ridge ~ 0 and PkaHead ~ 0  -> amplitude non-control stands, no longer a
    measurement artifact.
  * PkaHead > 0.5              -> amplitude conclusion collapses; title changes.
  * otherwise                  -> downgrade to "not observed under either
    readout".
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from policy_head import PolicyHead, cosine_np


def load_test_pairs(path, smiles, gt):
    rows = list(csv.DictReader(open(path)))
    smi_to_idx = {str(s): i for i, s in enumerate(smiles)}
    kept = []
    for row in rows:
        src, tgt = row["src_smiles"], row["tgt_smiles"]
        if src not in smi_to_idx or tgt not in smi_to_idx:
            raise RuntimeError(f"SMILES missing from h_test: {src} -> {tgt}")
        src_i, tgt_i = smi_to_idx[src], smi_to_idx[tgt]
        delta = float(gt[tgt_i] - gt[src_i])
        if abs(delta - float(row["delta_pka"])) > 1e-3:
            raise RuntimeError(f"delta mismatch for {src} -> {tgt}")
        kept.append({"src": src_i, "tgt": tgt_i, "src_smiles": src, "tgt_smiles": tgt,
                     "delta_pka": delta, "ridge_cosine": float(row["ridge_cosine"]),
                     "pka_head_gradient_cosine": float(row["pka_head_gradient_cosine"])})
    return kept


def build_head(path):
    mm = torch.load(path, map_location="cpu")
    state = mm["pka_head"]
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
    return head.eval(), {"d_in": int(d_in), "d_mid": int(d_mid), "n_layers": len(keys)}


def head_gradients(head, h_src):
    """Return head prediction and d head / d h at each source latent."""
    preds, grads = [], []
    for row in h_src:
        point = torch.tensor(row, dtype=torch.float32, requires_grad=True)
        value = head(point.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(value, point)[0].numpy().astype(np.float64)
        preds.append(float(value.item()))
        grads.append(grad)
    return np.asarray(preds), np.asarray(grads)


def load_policies(path, device):
    policies = []
    for seed in range(3):
        ckpt = torch.load(os.path.join(path, f"policy_seed{seed}.pt"), map_location="cpu")
        model = PolicyHead(ckpt["latent_dim"], ckpt["hidden"], ckpt["dropout"])
        model.load_state_dict(ckpt["state_dict"])
        policies.append(model.to(device).eval())
    return policies


def predict(policies, h, delta, src_pka):
    preds = []
    with torch.no_grad():
        for model in policies:
            preds.append(model(h, delta, src_pka).float().cpu().numpy())
    unit = [p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None) for p in preds]
    return preds, np.mean(unit, axis=0), np.mean(preds, axis=0)


def corr(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def report(achieved, target):
    return {
        "pearson_signed": corr(achieved, target),
        "pearson_abs": corr(np.abs(achieved), np.abs(target)),
        "sign_acc": float(np.mean(np.sign(achieved) == np.sign(target))),
        "median_abs_dpred": float(np.median(np.abs(achieved))),
        "frac_in_0.2_1.0": float(np.mean((np.abs(achieved) >= 0.2) & (np.abs(achieved) <= 1.0))),
    }


def bootstrap_corr_ci(achieved, target, groups=None, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    achieved = np.asarray(achieved, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    values = []
    if groups is None:
        for _ in range(n_boot):
            idx = rng.integers(0, len(achieved), len(achieved))
            value = corr(achieved[idx], target[idx])
            if value is not None:
                values.append(value)
    else:
        groups = np.asarray(groups)
        unique = np.unique(groups)
        members = {g: np.flatnonzero(groups == g) for g in unique}
        for _ in range(n_boot):
            pick = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([members[g] for g in pick])
            value = corr(achieved[idx], target[idx])
            if value is not None:
                values.append(value)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_test", required=True)
    ap.add_argument("--test_pairs", required=True)
    ap.add_argument("--policy_models", required=True)
    ap.add_argument("--pka_head_ckpt", required=True)
    ap.add_argument("--ridge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    latent = np.load(args.h_test, allow_pickle=True)
    h_test = latent["h"].astype(np.float32)
    gt = latent["gt"].astype(np.float32)
    smiles = [str(s) for s in latent["smiles"]]
    pairs = load_test_pairs(args.test_pairs, smiles, gt)
    assert len(pairs) == 43, f"expected 43 pairs, got {len(pairs)}"

    src = np.asarray([p["src"] for p in pairs])
    tgt = np.asarray([p["tgt"] for p in pairs])
    target = np.asarray([p["delta_pka"] for p in pairs], dtype=np.float64)
    h_src = h_test[src]
    true_disp = h_test[tgt] - h_test[src]

    ridge_npz = np.load(args.ridge, allow_pickle=True)
    w = np.asarray(ridge_npz["w_original"], dtype=np.float64).ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policies = load_policies(args.policy_models, device)
    h_t = torch.from_numpy(h_src).to(device)
    delta_t = torch.from_numpy(target.astype(np.float32)).to(device)
    src_pka_t = torch.from_numpy(gt[src]).to(device)
    preds, ens_unit, ens_raw = predict(policies, h_t, delta_t, src_pka_t)
    assert ens_unit.shape == h_src.shape

    head, head_cfg = build_head(args.pka_head_ckpt)
    head.eval()
    head_pred, g_head = head_gradients(head, h_src)
    g_norm = np.linalg.norm(g_head, axis=1)

    true_unit = true_disp / np.clip(np.linalg.norm(true_disp, axis=1, keepdims=True), 1e-12, None)

    readouts = {
        "ridge_fixed_step_s1": ens_unit @ w,
        "ridge_learned_norm": ens_raw @ w,
        "pkahead_fixed_step_s1": np.einsum("ij,ij->i", ens_unit, g_head),
        "pkahead_learned_norm": np.einsum("ij,ij->i", ens_raw, g_head),
        "pkahead_true_direction_s1": np.einsum("ij,ij->i", true_unit, g_head),
    }

    results = {
        "description": "Amplitude reread with the nonlinear PkaHead gradient; same 43 pairs and policy checkpoints as the reported -0.0199.",
        "preregistered_rule": {
            "both_null": "ridge and PkaHead fixed-step correlations both <= 0.2 in magnitude -> amplitude non-control stands",
            "collapse": "PkaHead fixed-step correlation > 0.5 -> amplitude conclusion collapses and title must change",
            "otherwise": "inconsistent readouts -> downgrade to 'not observed under either readout'",
        },
        "inputs": {
            "h_test": args.h_test, "test_pairs": args.test_pairs,
            "policy_models": args.policy_models, "pka_head_ckpt": args.pka_head_ckpt,
            "ridge": args.ridge, "pka_head_arch": head_cfg,
            "n_pairs": len(pairs), "n_source_molecules": int(len(np.unique(src))),
            "device": str(device),
        },
        "sanity": {
            "median_policy_cosine": float(np.median(cosine_np(ens_unit, true_disp))),
            "median_ridge_cosine": float(np.median([p["ridge_cosine"] for p in pairs])),
            "median_pkahead_gradient_cosine": float(np.median([p["pka_head_gradient_cosine"] for p in pairs])),
            "recomputed_pkahead_gradient_cosine": float(np.median(cosine_np(g_head, true_disp))),
            "pkahead_source_mae": float(np.mean(np.abs(head_pred - gt[src]))),
            "pkahead_gradient_norm_median": float(np.median(g_norm)),
            "pkahead_gradient_norm_cv": float(np.std(g_norm) / max(np.mean(g_norm), 1e-12)),
        },
        "amplitude": {
            name: {**report(value, target),
                   "pair_bootstrap_95ci": bootstrap_corr_ci(value, target, seed=7),
                   "cluster_bootstrap_95ci": bootstrap_corr_ci(value, target, groups=src, seed=11)}
            for name, value in readouts.items()
        },
        "per_seed": [],
    }
    for seed, pred in enumerate(preds):
        unit = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12, None)
        results["per_seed"].append({
            "seed": seed,
            "test_median_cos": float(np.median(cosine_np(pred, true_disp))),
            "ridge_fixed_step_s1": report(unit @ w, target),
            "pkahead_fixed_step_s1": report(np.einsum("ij,ij->i", unit, g_head), target),
            "pkahead_learned_norm": report(np.einsum("ij,ij->i", pred.astype(np.float64), g_head), target),
        })

    head_corr = results["amplitude"]["pkahead_fixed_step_s1"]["pearson_signed"]
    ridge_corr = results["amplitude"]["ridge_fixed_step_s1"]["pearson_signed"]
    if head_corr is not None and head_corr > 0.5:
        decision = "collapse_amplitude_conclusion"
    elif head_corr is not None and abs(head_corr) <= 0.2 and ridge_corr is not None and abs(ridge_corr) <= 0.2:
        decision = "amplitude_non_control_stands"
    else:
        decision = "inconsistent_readouts_downgrade"
    results["decision"] = {"outcome": decision, "pkahead_fixed_step_corr": head_corr,
                           "ridge_fixed_step_corr": ridge_corr}

    rows = []
    for i, pair in enumerate(pairs):
        rows.append({
            "src_smiles": pair["src_smiles"], "tgt_smiles": pair["tgt_smiles"],
            "delta_pka": target[i],
            "policy_cos": float(cosine_np(ens_unit[i:i + 1], true_disp[i:i + 1])[0]),
            "ridge_cos": pair["ridge_cosine"], "pkahead_grad_cos": pair["pka_head_gradient_cosine"],
            "ridge_fixed_step_s1": float(readouts["ridge_fixed_step_s1"][i]),
            "pkahead_fixed_step_s1": float(readouts["pkahead_fixed_step_s1"][i]),
            "pkahead_learned_norm": float(readouts["pkahead_learned_norm"][i]),
            "pkahead_true_direction_s1": float(readouts["pkahead_true_direction_s1"][i]),
            "pkahead_grad_norm": float(g_norm[i]), "pkahead_source_pred": float(head_pred[i]),
            "source_gt": float(gt[src[i]]),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"decision": results["decision"],
                      "amplitude": results["amplitude"],
                      "sanity": results["sanity"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
