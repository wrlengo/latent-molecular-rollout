"""Direction 3: target-conditioned latent rollout.

Learns a residual transition T(h, target_pka, step) -> h_next in the frozen
4096-d latent space. Training uses linear interpolation between train-split
pair endpoints as intermediate supervision. Evaluation rolls out K=5 steps on
the 43 held-out test pairs and reports both the frozen-ridge target error and
the latent endpoint distance, so a readout-exploiting "direct jump" is visible.

Pre-registered criterion: median |ridge(h_5) - target_pka| < 0.5.
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


class GoalTransition(nn.Module):
    def __init__(self, d, hidden, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, h, target, step):
        x = torch.cat([h, target[:, None], step[:, None]], dim=1)
        return h + self.net(x)


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["h"].astype(np.float32), np.asarray(z["smiles"]).astype(str), z["gt"].astype(np.float32)


def load_pairs(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def load_test_pairs(path):
    with open(path) as f:
        return list(csv.DictReader(f))


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


def ridge(h, w, b):
    return h @ w + b


def rollout(model, h0, target, k, step_scale=1.0):
    h = h0
    for i in range(k):
        h = model(h, target, torch.full_like(target, i / k * step_scale))
    return h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h_train", required=True)
    ap.add_argument("--h_test", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--test_pairs", required=True)
    ap.add_argument("--ridge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--lambda_dir", type=float, default=0.1)
    ap.add_argument("--lambda_readout", type=float, default=0.01)
    args = ap.parse_args()

    h_train, smi_train, gt_train = load_npz(args.h_train)
    h_test, smi_test, gt_test = load_npz(args.h_test)
    smi2tr = {s: i for i, s in enumerate(smi_train)}
    smi2te = {s: i for i, s in enumerate(smi_test)}
    d = h_train.shape[1]

    raw_pairs = [p for p in load_pairs(args.pairs) if p["src_smiles"] in smi2tr and p["tgt_smiles"] in smi2tr]
    test_rows = [p for p in load_test_pairs(args.test_pairs) if p["src_smiles"] in smi2te and p["tgt_smiles"] in smi2te]
    if not raw_pairs or not test_rows:
        raise RuntimeError("empty pair set after SMILES mapping")

    tr_src = np.array([smi2tr[p["src_smiles"]] for p in raw_pairs], dtype=np.int64)
    tr_tgt = np.array([smi2tr[p["tgt_smiles"]] for p in raw_pairs], dtype=np.int64)
    tr_target = np.array([p["tgt_pKa"] for p in raw_pairs], dtype=np.float32)
    te_src = np.array([smi2te[p["src_smiles"]] for p in test_rows], dtype=np.int64)
    te_tgt = np.array([smi2te[p["tgt_smiles"]] for p in test_rows], dtype=np.int64)
    te_src_pka = gt_test[te_src]
    te_tgt_pka = gt_test[te_tgt]
    te_delta = te_tgt_pka - te_src_pka

    rw = np.load(args.ridge, allow_pickle=True)
    w = torch.tensor(rw["w_original"], dtype=torch.float32)
    b = torch.tensor(float(rw["intercept_original"]), dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w, b = w.to(device), b.to(device)
    h_tr_t = torch.from_numpy(h_train).to(device)
    h_te_t = torch.from_numpy(h_test).to(device)
    tr_src_t = torch.from_numpy(tr_src).to(device)
    tr_tgt_t = torch.from_numpy(tr_tgt).to(device)
    tr_target_t = torch.from_numpy(tr_target).to(device)
    te_src_t = torch.from_numpy(te_src).to(device)
    te_tgt_t = torch.from_numpy(te_tgt).to(device)
    te_srcp_t = torch.from_numpy(te_src_pka).to(device)
    te_tgtp_t = torch.from_numpy(te_tgt_pka).to(device)

    seed_results = []
    for seed in range(args.seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        train_idx, val_idx = scaffold_split(raw_pairs, args.val_frac, seed)
        tr_i = torch.tensor(train_idx, dtype=torch.long, device=device)
        va_i = torch.tensor(val_idx, dtype=torch.long, device=device)

        model = GoalTransition(d, args.hidden, args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best_val, best_state, best_epoch = 1e18, None, -1

        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(len(tr_i), device=device)
            for start in range(0, len(tr_i), args.batch):
                sel = tr_i[perm[start:start + args.batch]]
                ha = h_tr_t[tr_src_t[sel]]
                hb = h_tr_t[tr_tgt_t[sel]]
                target = tr_target_t[sel]
                disp = hb - ha
                k = torch.randint(0, args.K, (len(sel),), device=device)
                frac = k.float() / args.K
                h_k = ha + disp * frac[:, None]
                h_next = ha + disp * ((k.float() + 1.0) / args.K)[:, None]
                pred = model(h_k, target, frac)
                loss = F.mse_loss(pred, h_next)
                loss = loss + args.lambda_dir * (1.0 - F.cosine_similarity(pred - h_k, disp, dim=1).mean())
                loss = loss + args.lambda_readout * F.mse_loss(ridge(pred, w, b), target)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

            model.eval()
            with torch.no_grad():
                ha = h_tr_t[tr_src_t[va_i]]
                hb = h_tr_t[tr_tgt_t[va_i]]
                target = tr_target_t[va_i]
                h5 = rollout(model, ha, target, args.K)
                err = (ridge(h5, w, b) - target).abs()
                val_med = float(err.median().item())
            if val_med < best_val:
                best_val, best_epoch = val_med, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if epoch - best_epoch >= args.patience:
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            ha = h_te_t[te_src_t]
            hb = h_te_t[te_tgt_t]
            target = te_tgtp_t
            h5 = rollout(model, ha, target, args.K)
            h1 = rollout(model, ha, target, 1)
            h3 = rollout(model, ha, target, 3)
            p5 = ridge(h5, w, b)
            p1 = ridge(h1, w, b)
            p3 = ridge(h3, w, b)
            p0 = ridge(ha, w, b)
            pb = ridge(hb, w, b)
            e5 = (p5 - target).abs()
            e1 = (p1 - target).abs()
            e3 = (p3 - target).abs()
            e0 = (p0 - target).abs()
            eb = (pb - target).abs()
            dir_ok = ((p5 - te_srcp_t) * (te_tgtp_t - te_srcp_t) > 0)
            cos = F.cosine_similarity(h5 - ha, hb - ha, dim=1)
            latent_dist = (h5 - hb).norm(dim=1)
            base_dist = (hb - ha).norm(dim=1)
            step_norms = []
            h = ha
            for i in range(args.K):
                hn = model(h, target, torch.full_like(target, i / args.K))
                step_norms.append((hn - h).norm(dim=1))
                h = hn
            step_norms = torch.stack(step_norms, dim=1)
        cpu = lambda x: x.detach().cpu().numpy()
        e5n, e1n, e3n, e0n, ebn = map(cpu, (e5, e1, e3, e0, eb))
        cosn, ldn, bdn, sn = map(cpu, (cos, latent_dist, base_dist, step_norms))
        seed_results.append({
            "seed": seed,
            "best_epoch": int(best_epoch),
            "best_val_median_readout_error": float(best_val),
            "test_k5_median_readout_error": float(np.median(e5n)),
            "test_k5_mean_readout_error": float(np.mean(e5n)),
            "test_k5_median_error_ci95": bootstrap_ci(e5n, seed=seed),
            "test_k1_median_readout_error": float(np.median(e1n)),
            "test_k3_median_readout_error": float(np.median(e3n)),
            "test_noop_median_readout_error": float(np.median(e0n)),
            "test_oracle_endpoint_median_readout_error": float(np.median(ebn)),
            "test_k5_direction_accuracy": float(dir_ok.float().mean().item()),
            "test_k5_median_cosine": float(np.median(cosn)),
            "test_k5_median_latent_endpoint_distance": float(np.median(ldn)),
            "test_median_true_displacement_norm": float(np.median(bdn)),
            "test_k5_mean_step_norm": float(sn.mean()),
            "test_k5_step_norm_cv": float(sn.std() / max(sn.mean(), 1e-9)),
            "per_pair_k5_error": [float(x) for x in e5n],
            "per_pair_k5_cosine": [float(x) for x in cosn],
        })
        print(json.dumps({k: v for k, v in seed_results[-1].items()
                          if k not in ("per_pair_k5_error", "per_pair_k5_cosine")}), flush=True)

    meds = [r["test_k5_median_readout_error"] for r in seed_results]
    result = {
        "description": "Target-conditioned residual latent rollout trained with K-step interpolation supervision; evaluated on 43 held-out test pairs.",
        "criterion": "median |ridge(h_5) - target_pka| < 0.5",
        "K": args.K,
        "n_train_pairs": len(raw_pairs),
        "n_test_pairs": len(te_src),
        "latent_dim": int(d),
        "config": vars(args),
        "seed_results": seed_results,
        "summary": {
            "median_of_seed_k5_medians": float(np.median(meds)),
            "best_seed_k5_median": float(np.min(meds)),
            "worst_seed_k5_median": float(np.max(meds)),
            "criterion_met_by_median_of_seeds": bool(np.median(meds) < 0.5),
            "criterion_met_by_best_seed": bool(np.min(meds) < 0.5),
            "readout_noise_floor_median": float(np.median([r["test_oracle_endpoint_median_readout_error"] for r in seed_results])),
            "k1_median": float(np.median([r["test_k1_median_readout_error"] for r in seed_results])),
            "k5_median": float(np.median(meds)),
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print("SUMMARY", json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
