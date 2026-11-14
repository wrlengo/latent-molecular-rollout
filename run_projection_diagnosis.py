"""Cross-fitted projection of pair displacements onto a shared pKa direction.

The direction is estimated from unit low-to-high pKa secants, independently of
PkaHead gradients. The primary analysis excludes every training pair sharing a
Murcko scaffold with either endpoint of the evaluation pair.
"""
import csv
import hashlib
import json
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TABLES = os.path.join(ROOT, "tables")
os.makedirs(TABLES, exist_ok=True)
RNG = np.random.default_rng(42)
N_NULL = 2000
N_BOOTSTRAP = 10000


def rankdata(values):
    """Average ranks with ties, avoiding an additional SciPy dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


latent_path = os.path.join(DATA, "h_test.npz")
with open(latent_path, "rb") as f:
    latent_sha256 = hashlib.sha256(f.read()).hexdigest()
latents = np.load(latent_path, allow_pickle=True)
h = latents["h"].astype(np.float32)
pair_path = os.path.join(DATA, "gate4_stratified.json")
with open(pair_path, "rb") as f:
    pair_file_sha256 = hashlib.sha256(f.read()).hexdigest()
pair_sets = json.load(open(pair_path))["pair_sets"]
pairs = pair_sets["all_tani>=0.6"]["pairs"]
n_pairs = len(pairs)
assert n_pairs == 43 and h.shape == (1151, 4096)
for row in pairs:
    assert str(latents["smiles"][row["low"]]) == row["src_smiles"]
    assert str(latents["smiles"][row["high"]]) == row["tgt_smiles"]

displacements = np.stack([h[r["high"]] - h[r["low"]] for r in pairs])
norms = np.linalg.norm(displacements, axis=1)
assert np.all(norms > 0)
unit_displacements = displacements / norms[:, None]
head_cosines = np.asarray([r["cos_low_to_high"] for r in pairs])
delta_pka = np.asarray([abs(r["delta_pka"]) for r in pairs])
tanimoto = np.asarray([r["tanimoto"] for r in pairs])

scaffolds = []
for row in pairs:
    endpoint_scaffolds = set()
    for key in ("src_smiles", "tgt_smiles"):
        mol = Chem.MolFromSmiles(row[key])
        endpoint_scaffolds.add(
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        )
    scaffolds.append(endpoint_scaffolds)


def training_mask(i, exclude_scaffolds):
    mask = np.arange(n_pairs) != i
    if exclude_scaffolds:
        mask &= np.asarray([
            not (scaffolds[i] & scaffolds[j]) for j in range(n_pairs)
        ])
    return mask


def cross_fit(exclude_scaffolds):
    cosines = np.empty(n_pairs)
    direction_norms = np.empty(n_pairs)
    n_train = np.empty(n_pairs, dtype=int)
    masks = []
    for i in range(n_pairs):
        mask = training_mask(i, exclude_scaffolds)
        direction = unit_displacements[mask].mean(axis=0)
        direction_norm = np.linalg.norm(direction)
        assert direction_norm > 0
        direction /= direction_norm
        cosines[i] = unit_displacements[i] @ direction
        direction_norms[i] = direction_norm
        n_train[i] = mask.sum()
        masks.append(mask)
    return cosines, direction_norms, n_train, masks


primary, direction_norms, n_train, primary_masks = cross_fit(True)
leave_one_out, _, _, _ = cross_fit(False)


def signed_flip_null():
    """Randomize low/high orientation in each training fold, preserving vectors."""
    null_cosines = np.empty((n_pairs, N_NULL), dtype=np.float32)
    for i, mask in enumerate(primary_masks):
        train = unit_displacements[mask]
        for start in range(0, N_NULL, 250):
            stop = min(start + 250, N_NULL)
            signs = RNG.choice(np.asarray([-1.0, 1.0], dtype=np.float32),
                               size=(stop - start, len(train)))
            random_directions = signs @ train / len(train)
            random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
            null_cosines[i, start:stop] = random_directions @ unit_displacements[i]
    return null_cosines


null = signed_flip_null()
observed_mean = float(primary.mean())
null_means = null.mean(axis=0)
mean_p_value = float((1 + np.sum(null_means >= observed_mean)) / (N_NULL + 1))


def bootstrap_median(values):
    values = np.asarray(values)
    samples = RNG.choice(values, size=(N_BOOTSTRAP, len(values)), replace=True)
    return np.quantile(np.median(samples, axis=1), [0.025, 0.975]).tolist()


original_keys = {(r["low"], r["high"]) for r in pair_sets["original_top20_tani>=0.6"]["pairs"]}
original_mask = np.asarray([(r["low"], r["high"]) in original_keys for r in pairs])
subsets = {
    "all": np.ones(n_pairs, dtype=bool),
    "small_1_to_2": (delta_pka >= 1) & (delta_pka < 2),
    "mid_2_to_4": (delta_pka >= 2) & (delta_pka < 4),
    "large_at_least_4": delta_pka >= 4,
    "tanimoto_at_least_0.7": tanimoto >= 0.7,
    "original_top20": original_mask,
}


def summarize(values, head_values):
    return {
        "n": int(len(values)),
        "signed_projection_mean": float(np.mean(values)),
        "signed_projection_median": float(np.median(values)),
        "signed_projection_median_bootstrap_95ci": bootstrap_median(values),
        "absolute_projection_median": float(np.median(np.abs(values))),
        "frac_signed_projection_gt_0.2": float(np.mean(values > 0.2)),
        "frac_signed_projection_gt_0.5": float(np.mean(values > 0.5)),
        "frac_absolute_projection_gt_0.5": float(np.mean(np.abs(values) > 0.5)),
        "g4_low_to_high_cosine_median_same_pairs": float(np.median(head_values)),
        "g4_low_to_high_cosine_mean_same_pairs": float(np.mean(head_values)),
    }


summary = {}
for name, mask in subsets.items():
    summary[name] = {
        "leave_scaffold_out": summarize(primary[mask], head_cosines[mask]),
        "leave_one_pair_out_sensitivity": summarize(leave_one_out[mask], head_cosines[mask]),
    }

per_pair = []
for i, row in enumerate(pairs):
    per_pair.append({
        "low_index": row["low"],
        "high_index": row["high"],
        "src_smiles": row["src_smiles"],
        "tgt_smiles": row["tgt_smiles"],
        "abs_delta_pka": float(delta_pka[i]),
        "tanimoto": float(tanimoto[i]),
        "n_direction_training_pairs": int(n_train[i]),
        "shared_direction_norm": float(direction_norms[i]),
        "projection_leave_scaffold_out": float(primary[i]),
        "projection_leave_one_pair_out": float(leave_one_out[i]),
        "g4_low_to_high_cosine": float(head_cosines[i]),
    })

# Exploratory continuous trend test; pair dependence makes this supportive,
# rather than definitive, evidence for the three-bin monotonic pattern.
delta_ranks = rankdata(delta_pka)
projection_ranks = rankdata(primary)
delta_centered = delta_ranks - delta_ranks.mean()
projection_centered = projection_ranks - projection_ranks.mean()
rank_denominator = np.linalg.norm(delta_centered) * np.linalg.norm(projection_centered)
observed_rho = float(delta_centered @ projection_centered / rank_denominator)
permuted_delta = np.stack([RNG.permutation(delta_centered) for _ in range(10000)])
null_rhos = permuted_delta @ projection_centered / rank_denominator
trend_p = float((1 + np.sum(np.abs(null_rhos) >= abs(observed_rho))) / 10001)
delta_gap_trend = {
    "n": n_pairs,
    "spearman_rho": observed_rho,
    "two_sided_permutation_p": trend_p,
    "n_permutations": 10000,
    "quantity": "Absolute pKa gap versus leave-scaffold-out signed projection",
    "limit": "Exploratory pair-level permutation; shared molecules and cross-fitted directions induce dependence not modelled by this test.",
}

result = {
    "delta_gap_trend": delta_gap_trend,
    "method": {
        "quantity": "cos(theta) = d^T u / ||d||, with d oriented from lower to higher pKa",
        "direction": "mean of training unit pair displacements; independent of PkaHead gradients",
        "primary_validation": "leave-one-pair-out plus exclusion of all pairs sharing a Murcko scaffold with either test endpoint",
        "sensitivity_validation": "ordinary leave-one-pair-out",
        "data_limit": "Only h_test and 43 G4 test pairs were available; this is cross-fitted test-pair evidence, not an independent train-pair-to-test-pair experiment.",
        "interpretation_limit": "A low global projection does not rule out learnable local or scaffold-specific geometry; a high projection does not by itself identify why the existing head gradient is misaligned.",
        "n_null_sign_flip_draws_per_fold": N_NULL,
        "n_bootstrap_draws": N_BOOTSTRAP,
        "random_seed": 42,
        "latent_file": "h_test.npz",
        "latent_sha256": latent_sha256,
        "pair_file": "gate4_stratified.json",
        "pair_file_sha256": pair_file_sha256,
    },
    "data": {
        "n_latent_molecules": int(h.shape[0]),
        "latent_dimension": int(h.shape[1]),
        "n_pairs": n_pairs,
        "n_unique_pair_molecules": len({r[k] for r in pairs for k in ("low", "high")}),
        "direction_training_pairs_min": int(n_train.min()),
        "direction_training_pairs_max": int(n_train.max()),
        "shared_direction_norm_median": float(np.median(direction_norms)),
    },
    "signed_orientation_null": {
        "observed_mean_projection": observed_mean,
        "null_mean_95pct_interval": np.quantile(null_means, [0.025, 0.975]).tolist(),
        "null_mean_distribution": null_means.tolist(),
        "one_sided_empirical_p": mean_p_value,
    },
    "summary": summary,
    "per_pair": per_pair,
}

json_path = os.path.join(DATA, "projection_diagnosis.json")
with open(json_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"saved {json_path}")

csv_path = os.path.join(TABLES, "table_projection_diagnosis.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "subset", "validation", "n", "projection_median", "projection_median_95ci_low",
        "projection_median_95ci_high", "projection_mean", "absolute_projection_median",
        "frac_projection_gt_0.2", "frac_projection_gt_0.5", "g4_low_to_high_median_same_pairs",
    ])
    for subset, methods in summary.items():
        for method, stats in methods.items():
            ci = stats["signed_projection_median_bootstrap_95ci"]
            writer.writerow([
                subset, method, stats["n"], stats["signed_projection_median"], ci[0], ci[1],
                stats["signed_projection_mean"], stats["absolute_projection_median"],
                stats["frac_signed_projection_gt_0.2"], stats["frac_signed_projection_gt_0.5"],
                stats["g4_low_to_high_cosine_median_same_pairs"],
            ])
print(f"saved {csv_path}")

pair_csv_path = os.path.join(TABLES, "table_projection_diagnosis_pairs.csv")
with open(pair_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(per_pair[0]))
    writer.writeheader()
    writer.writerows(per_pair)
print(f"saved {pair_csv_path}")

print(json.dumps({
    "primary_all": summary["all"]["leave_scaffold_out"],
    "primary_small": summary["small_1_to_2"]["leave_scaffold_out"],
    "delta_gap_trend": delta_gap_trend,
    "null": {k: v for k, v in result["signed_orientation_null"].items()
             if k != "null_mean_distribution"},
}, indent=2))
