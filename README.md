# Latent Molecular Rollout Towards Causal World Models

Paper source, figure-generation code, and evaluation artifacts accompanying
*Latent Molecular Rollout Towards Causal World Models*.

The paper asks how far a latent molecular rollout system can move toward a causal
world model. Using aqueous experimental pKa (iBonD) as a physically measurable
anchor, it audits three separable capabilities in order:

| Tier | Question | Evidence in this repo |
|---|---|---|
| 1 | Does predictive accuracy come from the molecular pathway rather than text shortcuts? | `data/fig2_final.json`, `data/eval_*_test.json` |
| 2 | Is target-conditioned latent direction learnable, and is amplitude calibrated independently of the readout? | `data/policy_*`, `data/frozen_ridge_results.json`, `data/amplitude_*` |
| 3 | Does latent control transfer to decoded molecular structure? | `data/policy_hidden_injection.json`, `data/policy_slot_*` |

## Repository layout

```
main.tex                 manuscript source
references.bib           bibliography
tables/                  LaTeX table fragments used by main.tex
figs/                    figure PDFs referenced by main.tex
make_*.py                figure/asset generation scripts (read from data/, write to figs/)
analyze_*.py             analyses behind the reported statistics
run_projection_diagnosis.py
make_evaluation_manifest.py   builds data/evaluation_artifact_manifest.json (SHA-256 provenance)
experiments/             experiment drivers for the audit tiers
data/                    evaluation outputs (JSON/CSV/JSONL) and the provenance manifest
```

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Figure regeneration (requires numpy/matplotlib):

```bash
python3 make_intro_diagrams.py
python3 make_main_assets.py
python3 make_policy_assets.py
python3 make_readout_assets.py
python3 make_remaining_assets.py
```

Provenance manifest:

```bash
python3 make_evaluation_manifest.py --allow-missing-train-latents
```

## Scope and limitations

- Only aqueous-phase iBonD measurements are used; DMSO and other non-aqueous
  records are excluded, so cross-solvent generalization is untested.
- No external validation loop (DFT, synthesis, robotic execution) is run; the
  audit is internal.
- The amplitude result is relative ordering under one nonlinear readout, not
  physical amplitude calibration.
- The Tier 3 nulls bound the two tested control surfaces (input-slot editing and
  final-hidden-state injection) at the evaluated perturbation sizes; they are not
  evidence of a decoder-wide bottleneck.

Scripts that require a large language-model backbone or a GPU are intentionally
not included in this repository (see `.gitignore`); the results they produced are
shipped under `data/` together with their hashes in
`data/evaluation_artifact_manifest.json`.
