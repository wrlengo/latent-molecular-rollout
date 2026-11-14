"""Direction 1: learn a target-conditioned direction policy head.

g(h_a, target_delta, source_pka) -> displacement direction in latent space.

Training uses only train-split molecular pairs. The 43 test pairs are held out
and evaluated with the same signed cosine used by G4. The pre-registered
criterion is a test median cosine > 0.3 (G4 baseline: 0.107).
"""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    def __init__(self, d, hidden, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(d + 2, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.mid = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, delta, src_pka):
        x = torch.cat([h, delta[:, None], src_pka[:, None]], dim=1)
        x = self.drop(F.gelu(self.ln(self.inp(x))))
        x = self.drop(F.gelu(self.mid(x)))
        return self.out(x)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["h"].astype(np.float32), np.asarray(z["smiles"]).astype(str), z["gt"].astype(np.float32)


def load_pairs(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows


def load_test_pairs(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cosine_np(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.clip(na * nb, 1e-12, None)


def load_ridge(path):
    z = np.load(path, allow_pickle=True)
    keys = list(z.keys())
    wkey = "w_original" if "w_original" in keys else ("w" if "w" in keys else keys[0])
    w = np.asarray(z[wkey], dtype=np.float64).ravel()
    return w


def amplitude_eval(direction, ha, target_delta, w, steps):
    """Controlled small-step test: take a fixed-norm step along `direction`
    and read out the achieved pKa change with the frozen ridge head."""
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    g = direction / np.clip(norm, 1e-12, None)
    out = {}
    for s in steps:
        achieved = s * (g @ w)
        out[f"s={s}"] = {
            "median_abs_dpred": float(np.median(np.abs(achieved))),
            "frac_in_0.2_1.0": float(np.mean((np.abs(achieved) >= 0.2) & (np.abs(achieved) <= 1.0))),
            "sign_acc": float(np.mean(np.sign(achieved) == np.sign(target_delta))),
            "pearson_signed": float(np.corrcoef(achieved, target_delta)[0, 1])
            if np.std(achieved) > 1e-12 else None,
            "pearson_abs": float(np.corrcoef(np.abs(achieved), np.abs(target_delta))[0, 1])
            if np.std(achieved) > 1e-12 else None,
        }
    return out


def learned_amplitude_eval(direction, target_delta, w):
    """Evaluate the learned per-pair step norm through the frozen ridge readout."""
    achieved = direction @ w
    step_norms = np.linalg.norm(direction, axis=1)
    return {
        "median_step_norm": float(np.median(step_norms)),
        "step_norm_cv": float(np.std(step_norms) / max(np.mean(step_norms), 1e-12)),
        "median_abs_dpred": float(np.median(np.abs(achieved))),
        "frac_in_0.2_1.0": float(np.mean((np.abs(achieved) >= 0.2) & (np.abs(achieved) <= 1.0))),
        "sign_acc": float(np.mean(np.sign(achieved) == np.sign(target_delta))),
        "pearson_signed": float(np.corrcoef(achieved, target_delta)[0, 1])
        if np.std(achieved) > 1e-12 else None,
        "pearson_abs": float(np.corrcoef(np.abs(achieved), np.abs(target_delta))[0, 1])
        if np.std(achieved) > 1e-12 else None,
    }


def bootstrap_ci(values, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    med = np.median(values[idx], axis=1)
    return [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]


def scaffold_split(pairs, val_frac, seed):
    groups = sorted({p["scaffold"] for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * val_frac)))
    val_groups = set(groups[:n_val])
    train_idx = [i for i, p in enumerate(pairs) if p["scaffold"] not in val_groups]
    val_idx = [i for i, p in enumerate(pairs) if p["scaffold"] in val_groups]
    return train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_train", required=True)
    ap.add_argument("--h_test", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--test_pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_models", default=None,
                    help="Optional directory for best per-seed state_dict checkpoints.")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--ridge", default=None)
    ap.add_argument("--amp_weight", type=float, default=0.0)
    args = ap.parse_args()

    w = load_ridge(args.ridge) if args.ridge else None
    if args.amp_weight > 0.0 and w is None:
        raise ValueError("--amp_weight requires --ridge")
    h_train, smi_train, gt_train = load_npz(args.h_train)
    h_test, smi_test, gt_test = load_npz(args.h_test)
    smi2tr = {s: i for i, s in enumerate(smi_train)}
    smi2te = {s: i for i, s in enumerate(smi_test)}
    d = h_train.shape[1]

    raw_pairs = load_pairs(args.pairs)
    pairs = []
    for p in raw_pairs:
        if p["src_smiles"] in smi2tr and p["tgt_smiles"] in smi2tr:
            pairs.append(p)
    test_rows = load_test_pairs(args.test_pairs)
    test_pairs = [p for p in test_rows if p["src_smiles"] in smi2te and p["tgt_smiles"] in smi2te]
    if not pairs or not test_pairs:
        raise RuntimeError("empty pair set after SMILES mapping")

    tr_src = np.array([smi2tr[p["src_smiles"]] for p in pairs], dtype=np.int64)
    tr_tgt = np.array([smi2tr[p["tgt_smiles"]] for p in pairs], dtype=np.int64)
    tr_delta = np.array([p["tgt_pKa"] - p["src_pKa"] for p in pairs], dtype=np.float32)
    tr_src_pka = np.array([p["src_pKa"] for p in pairs], dtype=np.float32)

    te_src = np.array([smi2te[p["src_smiles"]] for p in test_pairs], dtype=np.int64)
    te_tgt = np.array([smi2te[p["tgt_smiles"]] for p in test_pairs], dtype=np.int64)
    te_src_pka = gt_test[te_src]
    te_delta = gt_test[te_tgt] - gt_test[te_src]
    te_cos = np.array([float(p["pka_head_gradient_cosine"]) for p in test_pairs], dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w_t = torch.from_numpy(w.astype(np.float32)).to(device) if w is not None else None
    h_tr_t = torch.from_numpy(h_train).to(device)
    h_te_t = torch.from_numpy(h_test).to(device)
    tr_src_t = torch.from_numpy(tr_src).to(device)
    tr_tgt_t = torch.from_numpy(tr_tgt).to(device)
    tr_delta_t = torch.from_numpy(tr_delta).to(device)
    tr_srcp_t = torch.from_numpy(tr_src_pka).to(device)
    te_src_t = torch.from_numpy(te_src).to(device)
    te_tgt_t = torch.from_numpy(te_tgt).to(device)
    te_delta_t = torch.from_numpy(te_delta).to(device)
    te_srcp_t = torch.from_numpy(te_src_pka).to(device)

    seed_results = []
    all_test_preds = []
    for seed in range(args.seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_idx, val_idx = scaffold_split(pairs, args.val_frac, seed)
        tr_i = torch.tensor(train_idx, dtype=torch.long, device=device)
        va_i = torch.tensor(val_idx, dtype=torch.long, device=device)

        model = PolicyHead(d, args.hidden, args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best_val, best_state, best_epoch = -1e9, None, -1
        val_history = []
        stopped_epoch = args.epochs

        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(tr_i), device=device)
            for start in range(0, len(tr_i), args.batch):
                sel = tr_i[perm[start:start + args.batch]]
                ha = h_tr_t[tr_src_t[sel]]
                hb = h_tr_t[tr_tgt_t[sel]]
                target = hb - ha
                pred = model(ha, tr_delta_t[sel], tr_srcp_t[sel])
                loss = 1.0 - F.cosine_similarity(pred, target, dim=1).mean()
                if args.amp_weight > 0.0:
                    # ||pred|| * (normalized(pred) dot w) == pred dot w.
                    amp_loss = F.l1_loss(pred @ w_t, tr_delta_t[sel])
                    loss = loss + args.amp_weight * amp_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                ha = h_tr_t[tr_src_t[va_i]]
                hb = h_tr_t[tr_tgt_t[va_i]]
                pred = model(ha, tr_delta_t[va_i], tr_srcp_t[va_i])
                val_cos = F.cosine_similarity(pred, hb - ha, dim=1)
                val_med = float(val_cos.median().item())
            val_history.append({"epoch": epoch + 1, "val_median_cos": val_med})
            if val_med > best_val:
                best_val, best_epoch = val_med, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch >= args.patience:
                stopped_epoch = epoch + 1
                break

        model.load_state_dict(best_state)
        model.eval()
        if args.save_models:
            os.makedirs(args.save_models, exist_ok=True)
            torch.save({
                "state_dict": best_state,
                "seed": seed,
                "best_epoch": int(best_epoch),
                "best_val_median_cos": float(best_val),
                "latent_dim": int(d),
                "hidden": int(args.hidden),
                "dropout": float(args.dropout),
            }, os.path.join(args.save_models, f"policy_seed{seed}.pt"))
        with torch.no_grad():
            ha = h_te_t[te_src_t]
            hb = h_te_t[te_tgt_t]
            pred = model(ha, te_delta_t, te_srcp_t)
            cos = F.cosine_similarity(pred, hb - ha, dim=1).cpu().numpy()
        pred_np = pred.cpu().numpy()
        all_test_preds.append(pred_np)
        small = (np.abs(te_delta) >= 1.0) & (np.abs(te_delta) < 2.0)
        seed_results.append({
            "seed": seed,
            "best_epoch": int(best_epoch),
            "best_epoch_one_indexed": int(best_epoch + 1),
            "stopped_epoch_one_indexed": int(stopped_epoch),
            "best_val_median_cos": float(best_val),
            "val_history": val_history,
            "test_median_cos": float(np.median(cos)),
            "test_mean_cos": float(np.mean(cos)),
            "test_median_cos_ci95": bootstrap_ci(cos, seed=seed),
            "test_small_gap_median_cos": float(np.median(cos[small])) if small.any() else None,
            "test_small_gap_n": int(small.sum()),
            "amplitude_test": amplitude_eval(pred_np, h_test[te_src], te_delta, w,
                                             [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]) if w is not None else None,
            "learned_amplitude_test": learned_amplitude_eval(pred_np, te_delta, w)
            if w is not None else None,
            "test_cos": [float(x) for x in cos],
        })
        print(json.dumps({k: v for k, v in seed_results[-1].items() if k != "test_cos"}), flush=True)

    # Ensemble: average unit-normalized predictions across seeds.
    ens = np.mean([p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None) for p in all_test_preds], axis=0)
    ens_raw = np.mean(all_test_preds, axis=0)
    ha_np = h_test[te_src]
    hb_np = h_test[te_tgt]
    true_d = hb_np - ha_np
    ens_cos = cosine_np(ens, true_d)
    small = (np.abs(te_delta) >= 1.0) & (np.abs(te_delta) < 2.0)
    medians = [r["test_median_cos"] for r in seed_results]

    result = {
        "description": "Target-conditioned direction policy head trained on train pairs; evaluated on the 43 held-out test pairs.",
        "criterion": "test median cosine > 0.3",
        "amplitude_supervision": {
            "formula": "L_amp = mean(abs(pred @ w - target_delta)); equivalent to norm(pred) * (unit(pred) @ w)",
            "weight": float(args.amp_weight),
            "criterion": "ensemble learned-amplitude Pearson corr(achieved, target) > 0.5 and ensemble median cosine >= 0.3",
        },
        "g4_baseline_median_cos": float(np.median(te_cos)),
        "n_train_pairs": len(pairs),
        "n_test_pairs": len(test_pairs),
        "latent_dim": int(d),
        "config": vars(args),
        "seed_results": seed_results,
        "ensemble": {
            "test_median_cos": float(np.median(ens_cos)),
            "test_mean_cos": float(np.mean(ens_cos)),
            "test_median_cos_ci95": bootstrap_ci(ens_cos, seed=999),
            "test_small_gap_median_cos": float(np.median(ens_cos[small])) if small.any() else None,
            "test_small_gap_n": int(small.sum()),
            "amplitude_test": amplitude_eval(ens, ha_np, te_delta, w,
                                             [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]) if w is not None else None,
            "learned_amplitude_test": learned_amplitude_eval(ens_raw, te_delta, w)
            if w is not None else None,
            "test_cos": [float(x) for x in ens_cos],
        },
        "summary": {
            "median_of_seed_medians": float(np.median(medians)),
            "best_seed_median": float(np.max(medians)),
            "worst_seed_median": float(np.min(medians)),
            "criterion_met_by_median_of_seeds": bool(np.median(medians) > 0.3),
            "criterion_met_by_best_seed": bool(np.max(medians) > 0.3),
            "ensemble_median": float(np.median(ens_cos)),
            "criterion_met_by_ensemble": bool(np.median(ens_cos) > 0.3),
        },
    }
    if w is not None:
        step_results = result["ensemble"]["amplitude_test"]
        learned = result["ensemble"]["learned_amplitude_test"]
        result["summary"]["step_sensitivity"] = {
            "calibration_criterion_met": any(
                0.2 <= r["median_abs_dpred"] <= 1.0 and r["pearson_signed"] > 0.3
                for r in step_results.values()
            ),
            "all_steps_corr_below_0.1": all(r["pearson_signed"] < 0.1 for r in step_results.values()),
        }
        result["summary"]["amplitude_criterion_met_by_ensemble"] = bool(
            learned["pearson_signed"] > 0.5 and result["ensemble"]["test_median_cos"] >= 0.3
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print("SUMMARY", json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
