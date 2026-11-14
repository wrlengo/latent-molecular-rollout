"""Direct last-hidden-state injection at the final <end_latent> position.

Unlike P3, this script performs no input-embedding optimization. It adds a fixed
multiple of the unit policy direction to the backbone's final hidden state at
h_T during the generation prefill, then decodes. The prefill hook changes the
logits for the first generated token; cached K/V tensors are not directly edited.

A null result is limited to the tested injection site and magnitudes. The steps
are absolute h-space units, not relative fractions of ||h_T||; realized relative
norms are therefore recorded for every row and arm.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from policy_decode import (  # noqa: E402
    build_generator, build_heldout_records, canon, decode, generated_smiles,
    load_policies, policy_predictions, pooled_hT, prompt_for_row,
    score_learned_oracle, tanimoto,
)
from policy_head import cosine_np  # noqa: E402


def injection_hook(backbone, pool_pos, offset):
    state = {"n_backbone_calls": 0, "n_injections": 0, "hidden_shapes": []}

    def hook(module, args, output):
        state["n_backbone_calls"] += 1
        hidden = output.last_hidden_state
        if hidden.dim() == 4:
            hidden = hidden[-1]
        if len(state["hidden_shapes"]) < 4:
            state["hidden_shapes"].append(list(hidden.shape))
        # Each decode call has a fresh hook: its first backbone call is prefill,
        # and subsequent cached calls only process the newest token.
        if state["n_backbone_calls"] == 1:
            if hidden.shape[1] <= pool_pos:
                raise RuntimeError(f"Prefill does not contain pool_pos {pool_pos}: {hidden.shape}")
            before = hidden[0, pool_pos].float().clone()
            hidden[0, pool_pos] = hidden[0, pool_pos] + offset.to(hidden.dtype)
            after = hidden[0, pool_pos].float().clone()
            delta = after - before
            state.update({
                "n_injections": 1,
                "base_h_norm": float(before.norm()),
                "injected_h_norm": float(after.norm()),
                "delta_h_norm": float(delta.norm()),
                "relative_delta_h_norm": float(delta.norm() / before.norm().clamp_min(1e-12)),
                "base_h": before,
                "delta_h": delta,
            })
        return output

    handle = backbone.register_forward_hook(hook)
    return handle, state


def decode_with_injection(model, backbone, pe, pm, pool_pos, offset, args, protocol):
    handle, state = injection_hook(backbone, pool_pos, offset)
    try:
        text = decode(model, pe, pm, args.max_new_tokens, protocol,
                      args.temperature, args.top_p, args.seed)
    finally:
        handle.remove()
    if state["n_injections"] != 1:
        raise RuntimeError(f"Expected one prefill injection, got {state}")
    state.pop("base_h", None)
    state.pop("delta_h", None)
    state["text"] = text
    state["smiles"] = generated_smiles(text)
    return state


def hidden_metrics(row, step, protocol):
    arm = row["arms"][step][protocol]
    baseline = row["baselines"][protocol]
    return {
        "n": 1,
        "sequence_changed": baseline["text"] != arm["text"],
        "smiles_changed": baseline["smiles"] != arm["smiles"],
        "valid": arm["smiles"] is not None,
        "both_valid": baseline["smiles"] is not None and arm["smiles"] is not None,
        "tanimoto_to_baseline": tanimoto(baseline["smiles"], arm["smiles"]),
        "oracle_delta": arm.get("oracle_delta_vs_baseline"),
        "target_delta": row["target_delta_pka"],
        "relative_delta_h_norm": arm["relative_delta_h_norm"],
        "delta_h_norm": arm["delta_h_norm"],
        "base_h_norm": arm["base_h_norm"],
        "injected_h_norm": arm["injected_h_norm"],
    }


def summarize(records):
    n = len(records)
    changed = [r for r in records if r["sequence_changed"]]
    both_valid = [r for r in records if r["both_valid"]]
    tanis = [r["tanimoto_to_baseline"] for r in both_valid
             if r["tanimoto_to_baseline"] is not None]
    scored = [r for r in records if r["oracle_delta"] is not None]
    changed_scored = [r for r in changed if r["oracle_delta"] is not None]
    direction = [r for r in scored if r["oracle_delta"] * r["target_delta"] > 0]
    changed_direction = [r for r in changed_scored if r["oracle_delta"] * r["target_delta"] > 0]
    relative = np.asarray([r["relative_delta_h_norm"] for r in records], dtype=float)
    return {
        "n": n,
        "n_sequence_changed": len(changed),
        "sequence_change_rate": len(changed) / max(n, 1),
        "n_smiles_changed": sum(r["smiles_changed"] for r in records),
        "n_valid": sum(r["valid"] for r in records),
        "validity": sum(r["valid"] for r in records) / max(n, 1),
        "n_both_valid": len(both_valid),
        "tanimoto_to_baseline_median": float(np.median(tanis)) if tanis else None,
        "tanimoto_to_baseline_min": float(np.min(tanis)) if tanis else None,
        "n_both_oracle_scored": len(scored),
        "direction_accuracy_scored": len(direction) / len(scored) if scored else None,
        "direction_accuracy_all_noop": len(direction) / max(n, 1),
        "n_changed_both_oracle_scored": len(changed_scored),
        "direction_accuracy_changed_scored": (len(changed_direction) / len(changed_scored)
                                                if changed_scored else None),
        "relative_delta_h_norm_median": float(np.median(relative)),
        "relative_delta_h_norm_min": float(relative.min()),
        "relative_delta_h_norm_max": float(relative.max()),
        "delta_h_norm_median": float(np.median([r["delta_h_norm"] for r in records])),
        "base_h_norm_median": float(np.median([r["base_h_norm"] for r in records])),
        "injected_h_norm_median": float(np.median([r["injected_h_norm"] for r in records])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_models", required=True)
    ap.add_argument("--reference_policy_results", required=True)
    ap.add_argument("--h_test", required=True)
    ap.add_argument("--test_pairs", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--property_ckpt", required=True)
    ap.add_argument("--oracle_ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--steps", type=float, nargs="+", default=[0.1, 0.5, 1.0, 2.0])
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_oracle", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    latent = np.load(args.h_test, allow_pickle=True)
    h_test = latent["h"].astype(np.float32)
    gt_test = latent["gt"].astype(np.float32)
    smi_to_idx = {}
    for i, smi in enumerate(latent["smiles"]):
        canonical = canon(str(smi))
        if canonical is not None and canonical not in smi_to_idx:
            smi_to_idx[canonical] = i

    policies = load_policies(args.policy_models, "cuda")
    with open(args.test_pairs) as f:
        test_rows = list(csv.DictReader(f))
    test_rows = [r for r in test_rows
                 if canon(r["src_smiles"]) in smi_to_idx and canon(r["tgt_smiles"]) in smi_to_idx]
    te_src = np.asarray([smi_to_idx[canon(r["src_smiles"])] for r in test_rows], dtype=int)
    te_tgt = np.asarray([smi_to_idx[canon(r["tgt_smiles"])] for r in test_rows], dtype=int)
    ensemble = policy_predictions(
        policies, torch.from_numpy(h_test[te_src]).cuda(),
        torch.from_numpy(gt_test[te_tgt] - gt_test[te_src]).cuda(),
        torch.from_numpy(gt_test[te_src]).cuda())
    reference = json.load(open(args.reference_policy_results))
    reproduced_median = float(np.median(cosine_np(ensemble, h_test[te_tgt] - h_test[te_src])))
    reference_median = float(reference["ensemble"]["test_median_cos"])
    if abs(reproduced_median - reference_median) > 1e-3:
        raise RuntimeError(f"Policy reproduction failed: {reproduced_median} vs {reference_median}")

    raw, ds = build_heldout_records(latent["smiles"], gt_test)
    eligible, seen_sources = [], set()
    for i, (record, row) in enumerate(zip(raw, ds)):
        meta = json.loads(record["meta"])
        src = canon(meta["molecule"])
        target_delta = float(meta["tgt_pka"]) - float(meta["pka_target"])
        if src not in smi_to_idx or src in seen_sources or abs(target_delta) < 1e-8:
            continue
        seen_sources.add(src)
        eligible.append({"raw_index": i, "row": row, "meta": meta,
                         "src": src, "h_idx": smi_to_idx[src], "target_delta": target_delta})
    if args.limit and len(eligible) > args.limit:
        indices = np.linspace(0, len(eligible) - 1, args.limit).astype(int)
        eligible = [eligible[i] for i in indices]
    directions = policy_predictions(
        policies,
        torch.from_numpy(np.asarray([h_test[r["h_idx"]] for r in eligible])).cuda(),
        torch.from_numpy(np.asarray([r["target_delta"] for r in eligible], dtype=np.float32)).cuda(),
        torch.from_numpy(np.asarray([gt_test[r["h_idx"]] for r in eligible], dtype=np.float32)).cuda())
    directions /= np.clip(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12, None)

    model, checkpoint_compatibility = build_generator(args.ckpt, args.property_ckpt)
    backbone = model.model.base_model.model.model
    output_rows, preflight = [], None
    completed = set()
    if args.resume and os.path.exists(args.out):
        previous = json.load(open(args.out))
        output_rows = previous.get("rows", [])
        preflight = previous.get("preflight")
        completed = {r["raw_index"] for r in output_rows}
        print(f"Resuming with {len(completed)} completed rows", flush=True)

    step_names = {float(s): f"s{float(s):g}" for s in args.steps}
    for i, item in enumerate(eligible):
        if item["raw_index"] in completed:
            continue
        ids, pe, pm, latent_pos, pool_pos = prompt_for_row(model, item["row"])
        with torch.no_grad():
            pooled_base = pooled_hT(backbone, pe, pm, pool_pos).detach()
        direction = torch.from_numpy(directions[i]).to(pe.device)
        row_out = {
            "raw_index": item["raw_index"], "src_smiles": item["src"],
            "tgt_smiles": canon(item["meta"].get("answer")),
            "src_pka": float(item["meta"]["pka_target"]),
            "tgt_pka": float(item["meta"]["tgt_pka"]),
            "target_delta_pka": item["target_delta"],
            "pooled_baseline_h_norm": float(pooled_base.norm()),
            "baselines": {}, "arms": {},
        }
        for protocol in ("greedy", "sample"):
            text = decode(model, pe, pm, args.max_new_tokens, protocol,
                          args.temperature, args.top_p, args.seed)
            row_out["baselines"][protocol] = {"text": text, "smiles": generated_smiles(text)}
        for step in args.steps:
            step_name = step_names[float(step)]
            row_out["arms"][step_name] = {}
            for protocol in ("greedy", "sample"):
                arm = decode_with_injection(model, backbone, pe, pm, pool_pos,
                                            step * direction, args, protocol)
                if preflight is None:
                    # The hook and pooled_hT must expose the same pre-injection tensor.
                    with torch.no_grad():
                        base_out = model.model(inputs_embeds=pe, attention_mask=pm,
                                               return_dict=True, use_cache=False)
                        base_logits = base_out.logits[0, pool_pos].float().clone()
                    handle, state = injection_hook(backbone, pool_pos, step * direction)
                    try:
                        with torch.no_grad():
                            int_out = model.model(inputs_embeds=pe, attention_mask=pm,
                                                  return_dict=True, use_cache=False)
                            int_logits = int_out.logits[0, pool_pos].float().clone()
                    finally:
                        handle.remove()
                    hook_base = state["base_h"]
                    path_cos = float(F.cosine_similarity(pooled_base[None], hook_base[None]).item())
                    path_rel = float((hook_base - pooled_base).norm() / pooled_base.norm().clamp_min(1e-12))
                    logit_delta = float((int_logits - base_logits).norm())
                    preflight = {
                        "step": float(step), "pooled_hook_cosine": path_cos,
                        "pooled_hook_relative_error": path_rel,
                        "logit_delta_l2": logit_delta,
                        "logit_relative_l2": logit_delta / float(base_logits.norm().clamp_min(1e-12)),
                        "top1_changed": bool(base_logits.argmax() != int_logits.argmax()),
                    }
                    if path_cos < 0.9999 or path_rel > 1e-3 or logit_delta <= 0:
                        raise RuntimeError(f"Injection path check failed: {preflight}")
                    print("PREFLIGHT", json.dumps(preflight), flush=True)
                arm.pop("base_h", None)
                arm.pop("delta_h", None)
                row_out["arms"][step_name][protocol] = arm
                print(json.dumps({"row": i + 1, "n": len(eligible), "step": step,
                                  "protocol": protocol,
                                  "rel_dh": arm["relative_delta_h_norm"],
                                  "changed": arm["text"] != row_out["baselines"][protocol]["text"],
                                  "valid": arm["smiles"] is not None}), flush=True)
        output_rows.append(row_out)
        save_result(args, output_rows, preflight, reference_median, reproduced_median,
                    checkpoint_compatibility, step_names)

    if not args.skip_oracle:
        del model, backbone, policies
        torch.cuda.empty_cache()
        unique = {r["baselines"][p]["smiles"] for r in output_rows for p in ("greedy", "sample")}
        unique.update(r["arms"][s][p]["smiles"] for r in output_rows
                      for s in r["arms"] for p in ("greedy", "sample"))
        scores = score_learned_oracle(args.oracle_ckpt, sorted(s for s in unique if s))
        for row in output_rows:
            for protocol in ("greedy", "sample"):
                baseline = row["baselines"][protocol]
                baseline["learned_oracle_pka"] = scores.get(baseline["smiles"])
                for step in row["arms"]:
                    arm = row["arms"][step][protocol]
                    arm["learned_oracle_pka"] = scores.get(arm["smiles"])
                    if baseline["learned_oracle_pka"] is not None and arm["learned_oracle_pka"] is not None:
                        arm["oracle_delta_vs_baseline"] = arm["learned_oracle_pka"] - baseline["learned_oracle_pka"]
                    else:
                        arm["oracle_delta_vs_baseline"] = None
    result = save_result(args, output_rows, preflight, reference_median, reproduced_median,
                         checkpoint_compatibility, step_names)
    print("SUMMARY", json.dumps(result.get("summary", {}), indent=2), flush=True)
    print("ALL DONE", args.out, flush=True)



def save_result(args, rows, preflight, reference_median, reproduced_median,
                checkpoint_compatibility, step_names):
    summary = {}
    if all("learned_oracle_pka" in r["baselines"]["greedy"] for r in rows):
        for step in step_names.values():
            summary[step] = {
                protocol: summarize([hidden_metrics(r, step, protocol) for r in rows])
                for protocol in ("greedy", "sample")
            }
    result = {
        "protocol": {
            "intervention": "direct backbone final last_hidden_state injection at the final <end_latent> h_T during generation prefill",
            "optimization": "none",
            "steps_absolute_h_units": list(step_names.keys()),
            "direction": "unit-normalized three-seed policy ensemble",
            "cache_scope": "the hook changes first-token logits through h_T; cached K/V tensors are not directly edited",
            "decoding": ["greedy", "fixed-seed sampling"],
            "temperature": args.temperature, "top_p": args.top_p,
            "paired_generation_seed": args.seed, "max_new_tokens": args.max_new_tokens,
            "selection": "the same 80-source held-out cohort and uniform subsampling as P3",
            "n_selected": len(rows),
            "policy_reference_median": reference_median,
            "policy_reproduced_median": reproduced_median,
            "checkpoint_compatibility": checkpoint_compatibility,
            "interpretation_limit": "null results are limited to this injection site and the realized relative norms; they do not establish a decoder-wide bottleneck",
            "learned_oracle": "MAE-1.1 text-query PkaHead; not an external predictor or experimental measurement",
        },
        "artifacts": vars(args), "preflight": preflight, "rows": rows, "summary": summary,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    return result


if __name__ == "__main__":
    main()
