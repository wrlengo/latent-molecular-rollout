"""Hash and cross-check the linked manuscript evaluation artifacts."""
import argparse
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
TRAIN_LATENT_SHA256 = "8f4157860cf3f9e57297e10d64c9690492ad187536252f32facfe73805613143"
TRAIN_LATENT_BYTES = 58522482


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def record(path):
    return {"sha256": sha256(path), "bytes": os.path.getsize(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-latents", default=os.path.join(DATA, "h_train.npz"))
    parser.add_argument(
        "--allow-missing-train-latents",
        action="store_true",
        help="Carry forward the prior h_train.npz hash with an explicit unverified status.",
    )
    args = parser.parse_args()
    manifest_path = os.path.join(DATA, "evaluation_artifact_manifest.json")
    groups = {
        "shortcut": ["fig3_causal.json", "probe_tokenfix.json", "shortcut_diagnosis_summary.json", "../DIAGNOSIS.md"],
        "g4_s8": ["gate4_stratified.json", "projection_diagnosis.json"],
        "ridge": ["frozen_ridge_results.json", "frozen_ridge_test_pairs.csv", "ridge_selected_weights.npz",
                  "frozen_ridge_whitened_cosine.json", "frozen_ridge_whitened_pairs.csv", "ridge_whitened_weights.npz"],
        "pca": ["pca_direction_results.json", "pca_direction_test_pairs.csv",
                "pca_direction_component_loadings.csv", "pca_direction_fit.npz"],
        "sft": ["gen2f_val_rescored.jsonl", "rej1_val_rescored.jsonl", "bon_gen2f_val_n8.json",
                "bon_rej1_val_n8.json", "rej1_anchor_val.json", "rejection_sft_train_summary.json",
                "rejection_sft_results.json"],
        "dpo": ["offline_dpo_results.json", "offline_dpo_val_n8.json", "offline_dpo_train_pairs.jsonl",
                "offline_dpo_pairs_summary.json", "offline_dpo_training_summary.json",
                "offline_dpo_train_1652930.log", "offline_dpo_eval_1652931.log",
                "candidates/bon_gen2f_train_n8_shard0.json", "candidates/bon_gen2f_train_n8_shard1.json",
                "candidates/bon_gen2f_train_n8_shard2.json", "candidates/bon_gen2f_train_n8_shard3.json"],
        "paired_regression": ["eval_C_fix_test.json", "eval_Bp_fix_test.json", "paired_bootstrap_C_vs_Bp.json"],
        "policy": ["policy_head_baseline_results.json", "policy_head_steps_results.json",
                   "policy_head_amp_results.json", "policy_head_analysis.json",
                   "policy_amplitude_readout_results.json", "policy_amplitude_readout_pairs.csv",
                   "amplitude_scale_train.json"],
        "training_budget": ["training_budget_preregistration.json", "training_budget_results.json",
                            "policy_head_training_budget_results.json", "policy_head_1653232.log",
                            "v3_C_latent_head_fix.log", "v3_Bp_answer_head_fix.log",
                            "v3_Bpp_random_pos_fix.log", "v3_C_gen22_1648587.log",
                            "rej_sft_1649265.log", "rej_sft_1653233.log", "rej_eval_1653234.log",
                            "rej3_val_rescored.jsonl", "bon_rej3_val_n8.json", "rej3_anchor_val.json",
                            "offline_dpo_train_1653235.log", "offline_dpo_eval_1653789.log",
                            "offline_dpo_e4_val_n8.json", "offline_dpo_e4_results.json",
                            "offline_dpo_e4_training_summary.json"],
        "early_prediction_diagnostics": ["figdata.json", "fig1_interp.json", "scaffold_pairs.json"],
        "rollout_diagnostics": ["fig2_final.json", "fig3_nonresp_diagnosis.json", "goal_rollout_results.json"],
        "tables": ["../tables/table_projection_diagnosis.csv", "../tables/table_projection_diagnosis_pairs.csv",
                   "../tables/tableS9_frozen_readout.csv", "../tables/tableS9_frozen_ridge_alpha_validation.csv",
                   "../tables/tableS9_actual_split_audit.csv", "../tables/table_rejection_sft_comparison.csv",
                   "../tables/table_rejection_sft_constant_mcnemar.csv", "../tables/tableS10_pca_direction.csv",
                   "../tables/table4_generation.csv", "../tables/table4_generation.tex",
                   "../tables/table5_direction_amplitude.tex", "../tables/tableS3_policy_seeds.tex",
                   "../tables/tableS4_amplitude_supervision.tex", "../tables/tableS5_step_sensitivity.tex",
                   "../tables/tableS6_policy_decode.tex", "../tables/tableS7_policy_decode_sample.tex",
                   "../tables/table_training_config.tex", "../tables/table_training_budget.tex",
                   "../tables/tableS0_text_shortcut.tex"],
        "figures": ["../figs/figS0_text_shortcut_conflict.pdf", "../figs/figS0_text_shortcut_conflict.png",
                    "../figs/figS8_projection_diagnostic.pdf", "../figs/figS8_projection_diagnostic.png",
                    "../figs/figS9_frozen_readout.pdf", "../figs/figS9_frozen_readout.png",
                    "../figs/figS10_pca_direction.pdf", "../figs/figS10_pca_direction.png",
                    "../figs/figS_policy_validation_curves.pdf", "../figs/figS_policy_validation_curves.png",
                    "../figs/fig4_generation_path.pdf", "../figs/fig4_generation_path.png",
                    "../figs/fig3_causal_intervention_direction_main.pdf",
                    "../figs/fig3_causal_intervention_direction_main.png",
                    "../figs/fig4_direction_amplitude_main.pdf", "../figs/fig4_direction_amplitude_main.png",
                    "../figs/figS11_goal_rollout_seeds.pdf", "../figs/figS11_goal_rollout_seeds.png",
                    "../figs/figS12_g2_loss_error_conversion.pdf", "../figs/figS12_g2_loss_error_conversion.png",
                    "../figs/fig1_indomain_scatter.pdf", "../figs/fig1_indomain_scatter.png",
                    "../figs/figS_Bp_fix_scatter.pdf", "../figs/figS_Bp_fix_scatter.png",
                    "../figs/fig2_band_mae.pdf", "../figs/fig2_band_mae.png",
                    "../figs/fig3_ood_histogram.pdf", "../figs/fig3_ood_histogram.png",
                    "../figs/figD_pair_stats.pdf", "../figs/figD_pair_stats.png",
                    "../figs/fig1_latent_interp.pdf", "../figs/fig1_latent_interp.png",
                    "../figs/fig2_latent_rollout.pdf", "../figs/fig2_latent_rollout.png",
                    "../figs/fig0_latent_audit_overview.svg", "../figs/fig0_latent_audit_overview.pdf",
                    "../figs/fig0_latent_audit_overview.png", "../figs/fig0_latent_algorithm.svg",
                    "../figs/fig0_latent_algorithm.pdf", "../figs/fig0_latent_algorithm.png"],
        "code": ["../run_projection_diagnosis.py", "../analyze_frozen_readout_whitened.py",
                 "../analyze_rejection_sft.py", "../make_readout_assets.py", "../make_remaining_assets.py",
                 "../analyze_pca_direction.py", "../make_pca_assets.py", "../make_main_assets.py",
                 "../experiments/paired_bootstrap.py", "../experiments/policy_head.py",
                 "../experiments/run_policy.sh", "../experiments/goal_rollout.py", "../make_policy_assets.py",
                 "../experiments/offline_dpo/build_pairs.py", "../experiments/offline_dpo/train_dpo.py",
                 "../experiments/offline_dpo/evaluate.py", "../experiments/offline_dpo/run_train.sh",
                 "../experiments/offline_dpo/run_eval.sh", "../experiments/policy_decode.py",
                 "../experiments/policy_slot_identity.py",
                 "../experiments/run_policy_decode.sh", "../experiments/run_rejection_sft.sh",
                 "../experiments/run_eval_rejection.sh", "../analyze_training_budget.py",
                 "../experiments/policy_amplitude_readout.py", "../experiments/run_policy_amplitude.sh",
                 "../experiments/fit_amplitude_scale.py", "../experiments/run_amplitude_scale.sh",
                 "../experiments/policy_hidden_injection.py", "../experiments/run_policy_hidden_injection.sh",
                 "../make_intro_diagrams.py", "../make_evaluation_manifest.py"],
    }
    p3_names = ["policy_decode_full.json", "policy_decode_paired_summary.json",
                "policy_decode_sample.json", "policy_decode_sample_paired_summary.json",
                "policy_decode_overlap.json", "policy_decode_1653226.log",
               "policy_slot_identity.json"]
    groups["p3_decode"] = p3_names
    groups["hidden_injection"] = ["policy_hidden_injection.json"]
    files = {}
    for group, names in groups.items():
        files[group] = {}
        for name in names:
            path = os.path.normpath(os.path.join(DATA, name))
            files[group][os.path.basename(name)] = record(path)
    if os.path.isfile(args.train_latents):
        train_record = record(args.train_latents)
        train_status = "recomputed"
    elif args.allow_missing_train_latents:
        with open(manifest_path) as f:
            previous = json.load(f)
        train_record = dict(previous["files"]["latents"]["h_train.npz"])
        train_record["verification_status"] = "carried_forward_source_unavailable"
        train_status = train_record["verification_status"]
    else:
        raise FileNotFoundError(args.train_latents)
    assert train_record["sha256"] == TRAIN_LATENT_SHA256
    assert train_record["bytes"] == TRAIN_LATENT_BYTES
    files["latents"] = {"h_train.npz": train_record,
                        "h_test.npz": record(os.path.join(DATA, "h_test.npz"))}

    projection = json.load(open(os.path.join(DATA, "projection_diagnosis.json")))
    whitened = json.load(open(os.path.join(DATA, "frozen_ridge_whitened_cosine.json")))
    pca = json.load(open(os.path.join(DATA, "pca_direction_results.json")))
    sft = json.load(open(os.path.join(DATA, "rejection_sft_results.json")))
    dpo = json.load(open(os.path.join(DATA, "offline_dpo_results.json")))
    budget = json.load(open(os.path.join(DATA, "training_budget_results.json")))
    p3_greedy = json.load(open(os.path.join(DATA, "policy_decode_paired_summary.json")))
    p3_sampled = json.load(open(os.path.join(DATA, "policy_decode_sample_paired_summary.json")))
    hidden = json.load(open(os.path.join(DATA, "policy_hidden_injection.json")))
    shortcut = json.load(open(os.path.join(DATA, "shortcut_diagnosis_summary.json")))
    pre_repair = json.load(open(os.path.join(DATA, "fig3_causal.json")))
    token_probe = json.load(open(os.path.join(DATA, "probe_tokenfix.json")))
    with open(os.path.join(DATA, "eval_C_fix_test.json")) as f:
        repaired_rows = [json.loads(line) for line in f if line.strip()]
    pre_mae = sum(abs(p - y) for p, y in zip(pre_repair["ref_head"], pre_repair["gt"])) / 24
    repaired_mae = sum(abs(r["p0"] - r["gt"]) for r in repaired_rows) / 1151
    assert len(pre_repair["gt"]) == len(pre_repair["ref_head"]) == 24
    assert pre_repair["n_latent"] == 1 and token_probe["K"] == 6
    assert token_probe["dl_ids"] == token_probe["model_ids"]
    assert abs(shortcut["pre_repair_c"]["mae_ref_head"] - pre_mae) < 1e-12
    assert len(repaired_rows) == shortcut["repaired_c"]["n"] == 1151
    assert abs(shortcut["repaired_c"]["mae"] - repaired_mae) < 1e-12
    assert shortcut["modality_conflict"]["conditions"] == {
        "base": {"mae_vs_source": 1.17, "mae_vs_donor": 4.06},
        "text_donor": {"mae_vs_source": 4.08, "mae_vs_donor": 1.20},
        "molecule_donor": {"mae_vs_source": 1.17, "mae_vs_donor": 4.07},
    }
    assert shortcut["modality_conflict"]["output_invariance_count_available"] is False
    assert shortcut["repair_boundary"]["repairs_separately_ablated"] is False
    assert shortcut["repair_boundary"]["pre_and_post_cohorts_match"] is False
    actual = {name: values["sha256"] for group in files.values() for name, values in group.items()}
    assert projection["method"]["latent_sha256"] == actual["h_test.npz"]
    assert projection["method"]["pair_file_sha256"] == actual["gate4_stratified.json"]
    for name, expected in whitened["artifact_sha256"].items():
        assert actual[name] == expected, (name, actual[name], expected)
    for name, expected in pca["artifact_sha256"].items():
        assert actual[name] == expected, (name, actual[name], expected)
    for name, expected in sft["artifact_sha256"].items():
        assert actual[os.path.basename(name)] == expected, (name, actual[os.path.basename(name)], expected)
    assert dpo["input_hashes"]["baseline_bank"] == actual["bon_rej1_val_n8.json"]
    assert dpo["input_hashes"]["dpo_bank"] == actual["offline_dpo_val_n8.json"]
    for shard in range(4):
        name = f"bon_gen2f_train_n8_shard{shard}.json"
        expected = dpo["input_hashes"]["train_candidates"][f"runs/eval/{name}"]
        assert actual[name] == expected, (name, actual[name], expected)
    assert dpo["training"]["summary_sha256"] == actual["offline_dpo_training_summary.json"]
    assert dpo["pair_construction"]["summary_sha256"] == actual["offline_dpo_pairs_summary.json"]
    assert dpo["pair_construction"]["output_sha256"] == actual["offline_dpo_train_pairs.jsonl"]
    assert dpo["offline_dpo"]["n"] == 255 and dpo["offline_dpo"]["n_missing"] == 9
    assert dpo["all_pass"] is False
    assert budget["preregistration_sha256"] == actual["training_budget_preregistration.json"]
    for name, expected in budget["artifact_sha256"].items():
        assert actual[name] == expected, (name, actual[name], expected)
    assert budget["constant"] == 3.74
    assert [seed["best_epoch"] for seed in budget["policy"]["seeds"]] == [28, 43, 21]
    assert [seed["stopped_epoch"] for seed in budget["policy"]["seeds"]] == [53, 68, 46]
    assert p3_greedy["n"] == p3_greedy["n_paired_oracle_scored"] == 80
    assert p3_greedy["n_sequence_changed"] == 1 and p3_greedy["paired_zero_delta"] == 79
    assert p3_sampled["n"] == 80 and p3_sampled["n_paired_oracle_scored"] == 75
    assert p3_sampled["n_identity_zero_without_oracle"] == 5
    assert p3_sampled["n_sequence_changed"] == 0 and p3_sampled["paired_zero_delta"] == 80
    for summary in (p3_greedy, p3_sampled):
        assert summary["paired_direction_accuracy"] == summary["paired_abs_delta_median"] == 0.0
        assert summary["h_target_cosine_median"] == 0.9999677538871765
        assert summary["h_target_relative_error_median"] == 0.008068433497101068
    assert len(hidden["rows"]) == hidden["protocol"]["n_selected"] == 80
    assert hidden["protocol"]["steps_absolute_h_units"] == [0.1, 0.5, 1.0, 2.0]
    assert hidden["preflight"]["pooled_hook_cosine"] == 1.0
    assert hidden["preflight"]["pooled_hook_relative_error"] == 0.0
    assert hidden["preflight"]["logit_delta_l2"] > 0
    expected_relative_medians = {
        "s0.1": 0.0006750466709490865,
        "s0.5": 0.004160084994509816,
        "s1": 0.008178026415407658,
        "s2": 0.016169684007763863,
    }
    for step, expected_median in expected_relative_medians.items():
        for protocol in ("greedy", "sample"):
            arm = hidden["summary"][step][protocol]
            assert arm["n"] == 80
            assert arm["n_sequence_changed"] == arm["n_smiles_changed"] == 0
            assert arm["direction_accuracy_all_noop"] == 0.0
            assert arm["tanimoto_to_baseline_median"] == arm["tanimoto_to_baseline_min"] == 1.0
            assert arm["n_both_valid"] == (80 if protocol == "greedy" else 75)
            assert abs(arm["relative_delta_h_norm_median"] - expected_median) < 1e-15
    for row in hidden["rows"]:
        for step in expected_relative_medians:
            for protocol in ("greedy", "sample"):
                arm = row["arms"][step][protocol]
                assert arm["n_injections"] == 1
                assert arm["text"] == row["baselines"][protocol]["text"]

    manifest = {
        "contract": {
            "artifact_verification": (
                "All listed artifact sizes and SHA-256 hashes are recomputed from the source files."
                if train_status == "recomputed" else
                "All listed artifact sizes and SHA-256 hashes except h_train.npz are recomputed from the source files. "
                "The h_train.npz record is carried forward from the prior manifest because the source file is unavailable; "
                "it is marked carried_forward_source_unavailable and is not represented as independently reverified."
            ),
            "shortcut_diagnosis": "Pre-repair MAE 1.103 is recomputed from 24 held-out diagnostic rows, not the full test set. Modality-conflict values are archived aggregate n=24 summaries; per-row conflict predictions and output-invariance counts are unavailable. Query-SMILES removal and token-id repair were both applied before the repaired comparison and were not separately ablated. Repaired MAE 2.087 uses a distinct 1,151-row cohort, so the pre/post difference is not an isolated shortcut-removal effect. The eval_text truncation fix is a historical engineering note without a preserved raw diagnostic and is not used in PkaHead-based comparisons.",
            "g4_s8_ridge_pca": "All linked latent calculations use the exact h_test.npz SHA-256 below; ridge fitting and PCA use the exact h_train.npz SHA-256 below. Cross-file numbers are generated only after hash assertions pass.",
            "sft": "SFT does not read h_test.npz. Its provenance uses checkpoint identities and SHA-256 hashes of every evaluation input; no unrelated latent hash is attached.",
            "generation_evaluation": "Generated from the verified SFT outputs: paired single-sample comparisons use 235 both-scored rows, Best-of-8 is an all-255-row oracle upper bound, and no-op direction is N/A with target MAE reported separately. SFT and DPO constant comparisons use the same train-selected 3.74, all 255 rows, missing=no-op, and exact McNemar inference.",
            "offline_dpo": "The DPO result is a generator-LoRA evaluation, not a reranker. Single-sample direction uses the first candidate on all 255 rows and treats nine missing predictions as no-op. The primary constant 3.74 is selected on 2,295 training candidate rows. Exact McNemar against that constant is the primary inference; direction 0.72 and copy 0.15 are design references. Best-of-8 remains oracle-only.",
            "paired_regression": "The C-versus-B-prime bootstrap uses the same 1,151 prediction rows, 10,000 resamples, and seed 0. Its p-value is the two-sided bootstrap sign-tail value; inference is limited to this fixed test split.",
            "policy": "The baseline, extended-step, and amplitude-supervision runs are hashed separately. Direction uses unit-normalized prediction ensembling; learned amplitude uses raw-prediction ensembling. Fixed-step correlation is scale-invariant. The amplitude reread holds the 43 pairs, policy checkpoints, and pairs fixed and replaces only the readout: the selected frozen ridge (median direction cosine 1.18e-5) is reported alongside the nonlinear PkaHead gradient (median 0.107), with the decision rule archived in the results JSON before execution. Fixed-step PkaHead correlation is 0.460 and learned-norm correlation is 0.585; the learned-norm value meets the joint criterion, so the amplitude conclusion is readout-dependent rather than a capability null. A single global scale fitted on the 5,456 training pairs (0.217 least squares, 0.821 median ratio) is archived separately and does not calibrate the held-out pairs, so the residual scale error is per-pair rather than a missing constant. P3 separately audits structural decoding and is reported as an unsuccessful cross-checkpoint pre-feedback input-slot editing attempt. The slot-identity check describes only the initial prompt embeddings before forward computation and Adam optimization; it does not characterize optimized slots, feedback-written states, or decoder hidden states.",
            "training_budget": "C, B-prime, B-double-prime, and gen2 training logs verify three epochs. The policy rerun preserves all seeds and hyperparameters and stores full validation curves. SFT three-epoch and DPO four-epoch runs are one-shot preregistered budget ablations with unchanged data, initialization, optimization settings, and decoding protocols; both budgets are reported regardless of outcome.",
            "p3_decode": "P3 evaluates 80 eligible held-out sources with paired intervened-minus-baseline effects under greedy and sampled decoding (temperature 0.9, top-p 0.95, seed 0, 96 new tokens). The MAE-1.1 text-query PkaHead is a learned oracle, not independent or experimental validation. All 37 policy-test sources and 37/43 directed pairs overlap P3, so this is not an independent replication. P3 paired direction is not directly comparable with DPO source-relative accuracy. Sampled P3 has 75 oracle-scored pairs and five identical invalid pairs retained as zero effects by output identity; no changed output lacks an oracle score. The slot-identity check reads pe[0, latent_pos] immediately after get_prompt_embeddings for 20 held-out sources, before any backbone forward or P3 Adam optimization, and compares flattened 6 x 4096 blocks across 190 molecule pairs. It reports median cross-molecule cosine 0.99978 and median relative L2 distance 0.021. Its seven-pass count is measured under model.forward only; P3 instead calls model.model.generate(inputs_embeds=...) directly and does not execute the coconut feedback loop. The check uses no decoded output and does not measure optimized slots, feedback-written states, or final hidden states.",
            "hidden_injection": "The direct-injection archive covers the same 80 P3 sources, four absolute steps, greedy decoding, and one fixed sampling seed. A forward hook adds the unit policy direction once to the backbone final hidden state at the final <end_latent> position during prefill, without optimization or direct key/value-cache edits. The preflight verifies identity with pooled h_T and a nonzero first-token logit change. All eight arms have zero changed sequences; baseline-to-intervention Tanimoto is 1.0 for every both-valid pair. The null is restricted to this site and the archived realized relative displacements, not decoder-wide capacity.",
            "rollout_diagnostics": "The goal-conditioned figure uses only archived k=1,3,5 results. G2 squared loss is converted to absolute error by square root; G3 response means a changed generated token sequence. All G3 predictions are parsed from text on a 0.1 grid, so zero denotes decoding-level non-response rather than an arbitrarily small continuous output change.",
        },
        "checkpoints": sft["checkpoints"],
        "files": files,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"saved {manifest_path}")
    print(f"artifact records: {sum(len(group) for group in files.values())}")
    print(f"h_train SHA-256: {actual['h_train.npz']} ({train_status})")
    print(f"h_test SHA-256:  {actual['h_test.npz']} (recomputed)")


if __name__ == "__main__":
    main()
