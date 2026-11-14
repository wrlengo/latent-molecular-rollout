"""Paired bootstrap for the held-out C vs B-prime pKa predictions."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
N_BOOT = 10000
SEED = 0


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


rows_c = load_jsonl(os.path.join(DATA, "eval_C_fix_test.json"))
rows_bp = load_jsonl(os.path.join(DATA, "eval_Bp_fix_test.json"))
assert len(rows_c) == len(rows_bp) == 1151

gt_c = np.asarray([r["gt"] for r in rows_c], dtype=np.float64)
gt_bp = np.asarray([r["gt"] for r in rows_bp], dtype=np.float64)
assert np.allclose(gt_c, gt_bp), "C and B-prime rows are not paired by ground truth"
pred_c = np.asarray([r["p0"] for r in rows_c], dtype=np.float64)
pred_bp = np.asarray([r["p0"] for r in rows_bp], dtype=np.float64)
err_c = np.abs(pred_c - gt_c)
err_bp = np.abs(pred_bp - gt_bp)
delta = err_c - err_bp

rng = np.random.default_rng(SEED)
boot = np.empty(N_BOOT, dtype=np.float64)
for start in range(0, N_BOOT, 1000):
    n = min(1000, N_BOOT - start)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    boot[start:start + n] = delta[idx].mean(axis=1)

n_ge = int(np.count_nonzero(boot >= 0.0))
n_le = int(np.count_nonzero(boot <= 0.0))
p_two_sided = min(1.0, 2.0 * min(n_ge + 1, n_le + 1) / (N_BOOT + 1))
result = {
    "comparison": "C minus B-prime; per-row absolute pKa error",
    "n_pairs": len(delta),
    "n_bootstrap": N_BOOT,
    "seed": SEED,
    "mae_C": float(err_c.mean()),
    "mae_Bp": float(err_bp.mean()),
    "delta_mae": float(delta.mean()),
    "ci95_percentile": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "p_two_sided_bootstrap": float(p_two_sided),
    "n_bootstrap_delta_ge_zero": n_ge,
    "n_bootstrap_delta_le_zero": n_le,
    "paired_rows_identical_ground_truth": True,
}
out = os.path.join(DATA, "paired_bootstrap_C_vs_Bp.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
