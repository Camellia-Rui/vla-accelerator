"""Fixed-observation integration check. Not a closed-loop or publication benchmark."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server_native_smoke import (config, digest, gpu_inventory, select_gpu,
                                 checkpoint_snapshot, write_json, DEFAULT_BACKEND, DEFAULT_CHECKPOINT)


def compare(a, b):
    import numpy as np
    delta = np.abs(a - b)
    return {"mae": float(delta.mean()), "max_abs": float(delta.max()),
            "arm_mae": float(delta[:, :6].mean()),
            "gripper_raw_mae": float(delta[:, 6].mean()),
            "raw_gripper_sign_agreement": float((np.sign(a[:, 6]) == np.sign(b[:, 6])).mean()),
            "close": bool(np.allclose(a, b, rtol=1e-5, atol=1e-6))}


def trace_query(model, run):
    """Observe actual decoder layer inputs; hooks exist only for diagnostic calls."""
    shapes, handles = {}, []
    def hook_for(index):
        def hook(module, args, kwargs):
            value = kwargs.get("hidden_states", args[0] if args else None)
            if value is not None:
                shapes.setdefault(str(index), []).append(list(value.shape))
        return hook
    try:
        for index, layer in enumerate(model.language_model.model.layers):
            handles.append(layer.register_forward_pre_hook(hook_for(index), with_kwargs=True))
        actions = run()
        return actions, shapes
    finally:
        for handle in handles:
            handle.remove()


def shortened(before, after):
    return any(len(shape) == 3 and len(other) == 3 and other[1] < shape[1]
               for key, values in before.items() for shape, other in zip(values, after.get(key, [])))


def parameter_samples(model):
    return {name: (p, p.detach().reshape(-1)[:64].cpu().clone()) for name, p in model.named_parameters()}


def samples_match(model, original):
    import torch
    current = dict(model.named_parameters())
    return current.keys() == original.keys() and all(
        current[name] is p and torch.equal(p.detach().reshape(-1)[:64].cpu(), values)
        for name, (p, values) in original.items())


def run_check(args, report):
    import numpy as np
    import torch
    from experiments.robot import openvla_utils as vu
    from methods.vla_pruner import AcceleratorConfig, VLAPrunerAccelerator
    capture = json.loads((args.observation_dir / "report.json").read_text(encoding="utf-8"))
    obs_path = args.observation_dir / "observation.npz"
    if not capture.get("passed") or capture.get("task_suite") != "libero_object" or digest(obs_path) != capture["observation_sha256"]:
        raise ValueError("Invalid capture provenance or changed observation")
    with np.load(obs_path, allow_pickle=False) as raw:
        observation = {k: raw[k].copy() for k in ("full_image", "wrist_image", "state")}
    report["observation_sha256"] = digest(obs_path)
    report["task_description"] = capture["task_description"]
    cfg = config(args.checkpoint)
    torch.manual_seed(7)
    np.random.seed(7)
    print("Loading native model once", flush=True)
    model = vu.get_vla(cfg)
    processor = vu.get_processor(cfg)
    head = vu.get_action_head(cfg, model.llm_dim)
    projector = vu.get_proprio_projector(cfg, model.llm_dim, proprio_dim=8)
    loaded_file = Path(inspect.getfile(type(model)))
    report["model_source"] = {"loaded_file": str(loaded_file), "loaded_sha256": digest(loaded_file),
                              "checkpoint_sha256": digest(args.checkpoint / "modeling_prismatic.py"),
                              "backend_sha256": digest(args.backend / "prismatic/extern/hf/modeling_prismatic.py")}
    if len(set(v for k, v in report["model_source"].items() if k.endswith("sha256"))) != 1:
        raise RuntimeError("Loaded/cache/checkpoint/backend model code differs; review before integration")
    report["attention"] = {"config": str(getattr(model.language_model.config, "_attn_implementation", None)),
                           "classes": sorted({type(layer.self_attn).__name__ for layer in model.language_model.model.layers}),
                           "note": "Configured classes only; per-call SDPA/eager fallback is not certified by this smoke test"}
    report["runtime"] = {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
                         "gpu": torch.cuda.get_device_name(0), "loader": vu.__file__}
    def valid(value):
        a = np.asarray(value)
        if a.shape != (8, 7) or not np.isfinite(a).all():
            raise ValueError(f"Expected finite 8x7 action chunk, got {a.shape}")
        return a.copy()
    def native():
        with torch.inference_mode():
            actions, _, _, _ = vu.get_vla_action(cfg, model, processor, copy.deepcopy(observation),
                capture["task_description"], action_head=head, proprio_projector=projector, last_caches=None)
        return valid(actions)
    model.reset_av_history()
    for _ in range(args.warmup):
        native()
    original, native_trace = trace_query(model, native)
    original_modules = {name: (module, module.forward) for name, module in model.named_modules()}
    samples = parameter_samples(model)
    session = None
    records = []
    report["records"] = records
    try:
        session = VLAPrunerAccelerator.install(model, config=AcceleratorConfig(
            checkpoint=str(args.checkpoint), unnorm_key="libero_object_no_noops",
            use_vla_pruner=False, fastv_r=args.prune_ratio),
            processor=processor, action_head=head, proprio_projector=projector)
        def query():
            return valid(session.step(image=observation["full_image"].copy(),
                wrist_image=observation["wrist_image"].copy(), instruction=capture["task_description"],
                proprio=observation["state"].copy())["actions"])
        def prepare(enabled):
            session.enable() if enabled else session.disable()
            for _ in range(args.warmup):
                query()
        prepare(False)
        control, control_trace = trace_query(model, query)
        report["native_vs_disabled"] = compare(original, control)
        if not report["native_vs_disabled"]["close"]:
            raise RuntimeError("Disabled toolbox path does not reproduce native actions")
        prepare(True)
        accelerated, prune_trace = trace_query(model, query)
        report.update(layer_inputs={"native": native_trace, "disabled": control_trace, "enabled": prune_trace},
                      pruning_stats=session.get_stats(), diagnostic_action_difference=compare(control, accelerated),
                      actual_shorter_layer_input=shortened(control_trace, prune_trace))
        # Identical warmup histories; alternate order. Timing excludes diagnostic hooks and reset/warmup.
        for pair_id in range(args.pairs):
            order = [False, True] if pair_id % 2 == 0 else [True, False]
            for enabled in order:
                prepare(enabled)
                torch.cuda.synchronize()
                started = time.perf_counter()
                action = query()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                row = dict(pair_id=pair_id, mode="enabled" if enabled else "disabled",
                           query_ms=elapsed, actions=action.tolist(), stats=session.get_stats())
                records.append(row)
                with (args.output / "records.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            print(f"Pair {pair_id + 1}/{args.pairs} saved", flush=True)
    finally:
        if session is not None:
            session.detach()
        current_modules = dict(model.named_modules())
        report["module_and_forward_identity_restored"] = current_modules.keys() == original_modules.keys() and all(
            current_modules[name] is module and module.forward == forward
            for name, (module, forward) in original_modules.items())
        report["parameter_identity_and_samples_preserved"] = samples_match(model, samples)
        model.reset_av_history()
        for _ in range(args.warmup):
            native()
        restored = native()
        report["native_vs_detached"] = compare(original, restored)
        np.savez_compressed(args.output / "lifecycle_actions.npz", native=original, detached=restored)
    pairs = []
    for pair_id in range(args.pairs):
        a, b = [next(r for r in records if r["pair_id"] == pair_id and r["mode"] == mode)
                for mode in ("disabled", "enabled")]
        pairs.append(dict(pair_id=pair_id, saved_ms=a["query_ms"] - b["query_ms"],
                          action_difference=compare(np.asarray(a["actions"]), np.asarray(b["actions"]))))
    report["pairs"] = pairs
    a = [r["query_ms"] for r in records if r["mode"] == "disabled"]
    b = [r["query_ms"] for r in records if r["mode"] == "enabled"]
    report["descriptive_latency"] = dict(disabled_median_ms=float(np.median(a)),
        enabled_median_ms=float(np.median(b)), median_ratio=float(np.median(a) / np.median(b)))
    checks = (report["actual_shorter_layer_input"], report["native_vs_detached"]["close"],
              report["module_and_forward_identity_restored"], report["parameter_identity_and_samples_preserved"])
    if not all(checks):
        raise RuntimeError("Mechanism or restoration check failed; inspect report")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", type=Path, default=Path(DEFAULT_BACKEND))
    p.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    p.add_argument("--observation-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--prune-ratio", type=float, default=0.5)
    args = p.parse_args()
    if args.pairs < 2 or args.warmup < 1 or not 0 < args.prune_ratio < 1:
        p.error("pairs >= 2, warmup >= 1, 0 < prune-ratio < 1 required")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(stage="fixed-observation-toolbox-integration", passed=False,
                  benchmark_valid=False, quality_accepted=False, closed_loop_tested=False,
                  protocol=vars(args).copy(), script_sha256=digest(__file__),
                  limitation="Repeated single observation and repeated warmup history; no task accuracy or general speedup claim")
    report["protocol"] = {k: str(v) if isinstance(v, Path) else v for k, v in report["protocol"].items()}
    before = None
    try:
        before = checkpoint_snapshot(args.checkpoint)
        rows = gpu_inventory()
        chosen = select_gpu(rows, args.gpu, 32000)
        report.update(gpu_inventory_before=rows, selected_gpu=chosen)
        print("Selected GPU:", chosen, flush=True)
        os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(chosen["index"]),
                          TF_FORCE_GPU_ALLOW_GROWTH="true", OPENVLA_SKIP_CHECKPOINT_SYNC="1",
                          HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(args.backend.resolve()))
        repo = args.backend.resolve().parents[1]
        subprocess.run(["git", "-C", str(repo), "diff", "--quiet",
            "84d4b7192c77abf1585610e2f12393319b7ebff9", "--", "src/openvla-oft/prismatic",
            "src/openvla-oft/transformers", "src/openvla-oft/experiments/robot/openvla_utils.py"], check=True)
        report["backend_head"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        run_check(args, report)
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        print(report["traceback"], file=sys.stderr)
    finally:
        if before is not None:
            report["checkpoint_metadata_and_sizes_unchanged"] = before == checkpoint_snapshot(args.checkpoint)
            if not report["checkpoint_metadata_and_sizes_unchanged"]:
                report.update(passed=False, checkpoint_error="Metadata or sizes changed")
        report["weight_content_hash_verified"] = False
        write_json(args.output / "report.json", report)
    print("REPORT:", args.output / "report.json", flush=True)
    print("INTEGRATION_PASS" if report["passed"] else "INTEGRATION_FAIL", flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
