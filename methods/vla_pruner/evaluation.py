"""Reusable input-level paired evaluation. Closed-loop admission is deliberately separate."""
import copy
import json
import math
import os
from pathlib import Path
import time


def actions_array(value):
    import numpy as np
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[1] != 7 or not 1 <= result.shape[0] <= 8 or not np.isfinite(result).all():
        raise ValueError("Expected finite action chunk with shape [1..8, 7]")
    return result.copy()


def difference(a, b):
    import numpy as np
    a, b = actions_array(a), actions_array(b)
    if a.shape != b.shape:
        raise ValueError("Action shapes differ")
    error = np.abs(a-b)
    return {"mae": float(error.mean()), "max_abs": float(error.max()),
            "gripper_sign_agreement": float((np.sign(a[:, -1]) == np.sign(b[:, -1])).mean()),
            "matches": bool(np.allclose(a, b, atol=1e-6, rtol=1e-5))}


def screen(differences, speedup, thresholds):
    """Explicit thresholds only; even a passing input screen is not closed-loop admission."""
    missing = [key for key, value in thresholds.items() if value is None]
    failures = []
    for d in differences:
        for key, field in (("max_action_mae", "mae"), ("max_action_error", "max_abs")):
            if thresholds[key] is not None and d[field] > thresholds[key]:
                failures.append(key)
        if thresholds["min_gripper_agreement"] is not None and d["gripper_sign_agreement"] < thresholds["min_gripper_agreement"]:
            failures.append("min_gripper_agreement")
    if thresholds["min_speedup"] is not None and speedup < thresholds["min_speedup"]:
        failures.append("min_speedup")
    return {"decision": "INPUT_SCREEN_FAIL" if failures else "INCONCLUSIVE" if missing else "INPUT_SCREEN_PASS",
            "failed_gates": sorted(set(failures)), "unspecified_thresholds": missing,
            "closed_loop_quality_accepted": False}


def query(session, observation):
    o = copy.deepcopy(observation)
    return actions_array(session.step(image=o["full_image"], wrist_image=o.get("wrist_image"),
                        proprio=o.get("state"), instruction=o["instruction"])["actions"])


def trace(model, run):
    lengths, handles = {}, []
    def prehook(index):
        def observe(module, args, kwargs):
            value = kwargs.get("hidden_states", args[0] if args else None)
            if value is not None and value.ndim == 3:
                lengths.setdefault(str(index), []).append(int(value.shape[1]))
        return observe
    try:
        for i, layer in enumerate(model.language_model.model.layers):
            handles.append(layer.register_forward_pre_hook(prehook(i), with_kwargs=True))
        value = run()
        return value, lengths
    finally:
        for handle in handles:
            handle.remove()


def evaluate(bundle, cfg, output, pairs, warmup, thresholds):
    import numpy as np
    import torch
    from .accelerator import VLAPrunerAccelerator
    model = bundle["model"]
    native = bundle.get("native_query")
    reset = bundle.get("native_reset")
    if not callable(native) or not callable(reset):
        raise ValueError("Evaluation factory must provide native_query(observation) and native_reset()")
    observations = bundle["observations"]
    parameters = dict(model.named_parameters())
    samples = {n: p.detach().reshape(-1)[:64].cpu().clone() for n,p in parameters.items()}
    modules = {n: (m, m.forward) for n,m in model.named_modules()}
    refs, records, results = [], [], []
    report = {"protocol": "independent-observation-reset-with-repeated-warmup", "pairs_per_input": pairs,
              "warmup_per_mode": warmup, "records": records, "inputs": results,
              "closed_loop_tested": False, "formal_benchmark_valid": False, "thresholds": thresholds}
    def persist():
        Path(output, "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    def synchronize():
        devices = {p.device for p in parameters.values() if p.device.type == "cuda"}
        for device in devices:
            torch.cuda.synchronize(device)
    def native_reference(o):
        reset()
        with torch.inference_mode():
            for _ in range(warmup):
                native(copy.deepcopy(o))
            return actions_array(native(copy.deepcopy(o)))
    for observation in observations:
        refs.append(native_reference(observation))
    session = None
    try:
        session = VLAPrunerAccelerator.install(model, config=cfg,
            **{k:bundle[k] for k in ("processor", "action_head", "proprio_projector", "action_runner") if k in bundle})
        def prepare(enabled, observation):
            session.enable() if enabled else session.disable()
            for _ in range(warmup):
                query(session, observation)
        for index, o in enumerate(observations):
            item = {"id": str(o.get("id", index)), "index": index, "native_actions": refs[index].tolist()}
            results.append(item)
            prepare(False, o)
            control, before = trace(model, lambda: query(session, o))
            item["native_vs_control"] = difference(refs[index], control)
            if not item["native_vs_control"]["matches"]:
                raise RuntimeError("Disabled control does not reproduce native actions")
            prepare(True, o)
            accelerated, after = trace(model, lambda: query(session, o))
            item.update(layer_lengths={"disabled":before,"enabled":after}, stats=session.get_stats(),
                        diagnostic_difference=difference(control, accelerated),
                        mechanism_observed=any(y < x for k,v in before.items() for x,y in zip(v,after.get(k,[]))))
            if not item["mechanism_observed"]:
                raise RuntimeError("No actual decoder input shortening observed")
            for pair_id in range(pairs):
                for enabled in ((False, True) if pair_id%2 == 0 else (True, False)):
                    prepare(enabled, o)
                    synchronize()
                    start = time.perf_counter()
                    actions = query(session, o)
                    synchronize()
                    row = {"input_index":index,"pair_id":pair_id,"mode":"accelerated" if enabled else "baseline",
                           "query_ms":(time.perf_counter()-start)*1000,"actions":actions.tolist(),"stats":session.get_stats()}
                    records.append(row)
                    with Path(output, "records.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row,allow_nan=False)+"\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                persist()
            print(f"Evaluated input {index+1}/{len(observations)}", flush=True)
    finally:
        if session is not None:
            session.detach()
        current = dict(model.named_parameters())
        report["parameter_identity_and_samples_preserved"] = current.keys()==parameters.keys() and all(
            current[n] is p and torch.equal(p.detach().reshape(-1)[:64].cpu(),samples[n]) for n,p in parameters.items())
        current_modules = dict(model.named_modules())
        report["module_and_forward_identity_restored"] = current_modules.keys()==modules.keys() and all(
            current_modules[n] is m and m.forward==forward for n,(m,forward) in modules.items())
        report["restored_outputs"] = [difference(ref, native_reference(o)) for ref,o in zip(refs,observations)]
        persist()
    if not (report["parameter_identity_and_samples_preserved"] and report["module_and_forward_identity_restored"]
            and all(item["matches"] for item in report["restored_outputs"])):
        raise RuntimeError("Restoration check failed")
    comparisons=[]
    baseline_times, accelerated_times=[],[]
    for index in range(len(observations)):
        input_differences, input_baseline, input_accelerated = [], [], []
        for pair_id in range(pairs):
            a,b=[next(r for r in records if r["input_index"]==index and r["pair_id"]==pair_id and r["mode"]==mode)
                 for mode in ("baseline","accelerated")]
            delta = difference(a["actions"],b["actions"])
            comparisons.append(delta)
            input_differences.append(delta)
            input_baseline.append(a["query_ms"])
            input_accelerated.append(b["query_ms"])
            baseline_times.append(a["query_ms"])
            accelerated_times.append(b["query_ms"])
        input_ratio = float(np.median(input_baseline)/np.median(input_accelerated))
        results[index]["descriptive_median_ratio"] = input_ratio
        results[index]["input_screen"] = screen(input_differences, input_ratio, thresholds)
    speedup=float(np.median(baseline_times)/np.median(accelerated_times))
    screens = [item["input_screen"] for item in results]
    combined = {"decision": "INPUT_SCREEN_FAIL" if any(s["decision"]=="INPUT_SCREEN_FAIL" for s in screens)
                else "INCONCLUSIVE" if any(s["decision"]=="INCONCLUSIVE" for s in screens) else "INPUT_SCREEN_PASS",
                "failed_gates": sorted({gate for s in screens for gate in s["failed_gates"]}),
                "unspecified_thresholds": sorted({gate for s in screens for gate in s["unspecified_thresholds"]}),
                "closed_loop_quality_accepted": False}
    report.update(integration_passed=True, descriptive_median_ratio=speedup,
                  paired_action_differences=comparisons, input_screen=combined)
    persist()
    return report
