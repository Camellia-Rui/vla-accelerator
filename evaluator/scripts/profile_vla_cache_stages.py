from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image

import experiments.robot.openvla_utils as openvla_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile baseline and VLA-Cache on saved LIBERO frames."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--unnorm-key", default="libero_spatial")
    parser.add_argument("--patch-top-k", type=int, default=130)
    parser.add_argument("--similarity-threshold", type=float, default=0.996)
    parser.add_argument("--attention-top-k", type=int, default=120)
    parser.add_argument("--growth-factor", type=float, default=0.55)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }


class StageCollector:
    def __init__(self) -> None:
        self.active = False
        self.cuda_pending: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self.cuda_stacks: dict[str, list[torch.cuda.Event]] = defaultdict(list)
        self.cpu_values: dict[str, list[float]] = defaultdict(list)

    def cuda_begin(self, name: str) -> None:
        if not self.active:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.cuda_stacks[name].append(event)

    def cuda_end(self, name: str) -> None:
        if not self.active:
            return
        start = self.cuda_stacks[name].pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.cuda_pending[name].append((start, end))

    def add_cpu(self, name: str, milliseconds: float) -> None:
        if self.active:
            self.cpu_values[name].append(milliseconds)

    def finish(self) -> dict[str, float]:
        torch.cuda.synchronize()
        result = {
            name: float(sum(start.elapsed_time(end) for start, end in pairs))
            for name, pairs in self.cuda_pending.items()
        }
        result.update(
            {
                name: float(sum(values))
                for name, values in self.cpu_values.items()
            }
        )
        self.cuda_pending.clear()
        self.cuda_stacks.clear()
        self.cpu_values.clear()
        return result


def load_instruction(frames_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    summary_path = frames_dir / "capture_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        if data.get("task_description"):
            return str(data["task_description"])
    raise ValueError("No instruction supplied and capture_summary.json has no task_description")


def build_cfg(args: argparse.Namespace, use_cache: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.checkpoint,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        task_suite_name=args.unnorm_key,
        unnorm_key=args.unnorm_key,
        use_vla_cache=use_cache,
        cache_patch_top_k=args.patch_top_k,
        cache_patch_similarity_threshold=args.similarity_threshold,
        cache_attention_top_k=args.attention_top_k,
        cache_growth_factor=args.growth_factor,
    )


def add_cuda_hook(
    handles: list[Any], collector: StageCollector, module: torch.nn.Module, name: str
) -> None:
    handles.append(
        module.register_forward_pre_hook(
            lambda _module, _args: collector.cuda_begin(name)
        )
    )
    handles.append(
        module.register_forward_hook(
            lambda _module, _args, _output: collector.cuda_end(name)
        )
    )


def install_utility_wrappers(
    collector: StageCollector,
) -> tuple[dict[str, Callable[..., Any]], dict[str, list[int]]]:
    originals: dict[str, Callable[..., Any]] = {}
    observations: dict[str, list[int]] = {"reusable_patch_counts": []}

    def wrap_cpu(name: str, stage: str, observe_reuse: bool = False) -> None:
        original = getattr(openvla_utils, name)
        originals[name] = original

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = original(*args, **kwargs)
            collector.add_cpu(stage, (time.perf_counter() - start) * 1000.0)
            if observe_reuse and collector.active:
                selected = result[1]
                observations["reusable_patch_counts"].append(
                    0 if selected is None else len(selected)
                )
            return result

        setattr(openvla_utils, name, wrapped)

    def wrap_cuda(name: str, stage: str) -> None:
        original = getattr(openvla_utils, name)
        originals[name] = original

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            collector.cuda_begin(stage)
            result = original(*args, **kwargs)
            collector.cuda_end(stage)
            return result

        setattr(openvla_utils, name, wrapped)

    wrap_cpu("find_static_patches", "cache_static_patch_cpu_ms")
    wrap_cpu("task_relevant_selection", "cache_task_selection_wall_ms", observe_reuse=True)
    wrap_cuda("get_layer_mask_schedule", "cache_layer_schedule_cuda_ms")
    return originals, observations


def restore_utility_wrappers(originals: dict[str, Callable[..., Any]]) -> None:
    for name, function in originals.items():
        setattr(openvla_utils, name, function)


def main() -> None:
    args = parse_args()
    if args.warmup_frames < 1:
        raise ValueError("--warmup-frames must be >= 1 so VLA-Cache can establish state")

    frames_dir = Path(args.frames_dir)
    frame_paths = sorted(frames_dir.glob("*.png"))[: args.max_frames]
    if len(frame_paths) <= args.warmup_frames:
        raise ValueError("Not enough PNG frames for warmup and measurement")
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    instruction = load_instruction(frames_dir, args.instruction)

    print("Loading VLA-Cache model...")
    load_cfg = build_cfg(args, use_cache=False)
    model = openvla_utils.get_vla(load_cfg)
    processor = openvla_utils.get_processor(load_cfg)
    model.eval()

    collector = StageCollector()
    handles: list[Any] = []
    add_cuda_hook(handles, collector, model.vision_backbone, "vision_backbone_cuda_ms")
    add_cuda_hook(handles, collector, model.projector, "projector_cuda_ms")

    decoder = model.language_model.model
    decoder_stack: list[str] = []

    def decoder_pre_hook(_module: Any, args_: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        inputs_embeds = kwargs.get("inputs_embeds")
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args_:
            input_ids = args_[0]
        stage = "llm_prefill_cuda_ms" if inputs_embeds is not None else "llm_decode_cuda_ms"
        decoder_stack.append(stage)
        collector.cuda_begin(stage)

    def decoder_post_hook(
        _module: Any, _args: tuple[Any, ...], _kwargs: dict[str, Any], _output: Any
    ) -> None:
        collector.cuda_end(decoder_stack.pop())

    handles.append(decoder.register_forward_pre_hook(decoder_pre_hook, with_kwargs=True))
    handles.append(decoder.register_forward_hook(decoder_post_hook, with_kwargs=True))
    originals, observations = install_utility_wrappers(collector)

    records: dict[str, dict[str, Any]] = {
        mode: {
            "wall_ms": [],
            "cuda_total_ms": [],
            "stages": defaultdict(list),
            "actions": [],
        }
        for mode in ("baseline", "cache")
    }
    baseline_cfg = build_cfg(args, use_cache=False)
    cache_cfg = build_cfg(args, use_cache=True)
    cache_state: Any = None

    def infer(mode: str, frame_index: int, measured: bool) -> np.ndarray:
        nonlocal cache_state
        cfg = baseline_cfg if mode == "baseline" else cache_cfg
        previous = frames[max(0, frame_index - 1)]
        observation = {
            "full_image": np.asarray(frames[frame_index]),
            "prev_image": np.asarray(previous),
        }
        prior_state = None if mode == "baseline" else cache_state
        collector.active = measured
        torch.cuda.synchronize()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        cuda_start.record()
        action, new_state, _result_image = openvla_utils.get_vla_action(
            cfg,
            model,
            processor,
            args.checkpoint,
            observation,
            instruction,
            args.unnorm_key,
            center_crop=True,
            last_caches=prior_state,
        )
        cuda_end.record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        cuda_ms = float(cuda_start.elapsed_time(cuda_end))
        stages = collector.finish() if measured else {}
        collector.active = False
        if mode == "cache":
            cache_state = new_state
        if measured:
            records[mode]["wall_ms"].append(wall_ms)
            records[mode]["cuda_total_ms"].append(cuda_ms)
            records[mode]["actions"].append(np.asarray(action, dtype=np.float64).tolist())
            for name, value in stages.items():
                records[mode]["stages"][name].append(value)
        return np.asarray(action, dtype=np.float64)

    try:
        print(f"Warmup/state establishment: {args.warmup_frames} frames")
        for frame_index in range(args.warmup_frames):
            infer("baseline", frame_index, measured=False)
            infer("cache", frame_index, measured=False)

        print(f"Profiling {len(frames) - args.warmup_frames} paired real frames...")
        paired_differences: list[np.ndarray] = []
        for frame_index in range(args.warmup_frames, len(frames)):
            order = ("baseline", "cache") if frame_index % 2 == 0 else ("cache", "baseline")
            actions: dict[str, np.ndarray] = {}
            for mode in order:
                actions[mode] = infer(mode, frame_index, measured=True)
            paired_differences.append(np.abs(actions["baseline"] - actions["cache"]))
    finally:
        restore_utility_wrappers(originals)
        for handle in handles:
            handle.remove()

    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "frames": [str(path) for path in frame_paths],
        "measured_frames": len(paired_differences),
        "parameters": {
            "patch_top_k": args.patch_top_k,
            "similarity_threshold": args.similarity_threshold,
            "attention_top_k": args.attention_top_k,
            "growth_factor": args.growth_factor,
        },
        "modes": {},
    }
    for mode, record in records.items():
        summary["modes"][mode] = {
            "wall_ms": stats(record["wall_ms"]),
            "cuda_total_ms": stats(record["cuda_total_ms"]),
            "stages": {
                name: stats(values) for name, values in record["stages"].items()
            },
            "actions": record["actions"],
        }

    all_differences = np.stack(paired_differences)
    cache_wall = summary["modes"]["cache"]["wall_ms"]["median"]
    baseline_wall = summary["modes"]["baseline"]["wall_ms"]["median"]
    summary["comparison"] = {
        "wall_speedup": float(baseline_wall / cache_wall),
        "latency_change_percent": float((cache_wall / baseline_wall - 1.0) * 100.0),
        "action_mae": float(all_differences.mean()),
        "action_maximum_difference": float(all_differences.max()),
        "exact_action_frames": int(np.all(all_differences == 0.0, axis=1).sum()),
        "large_difference_frames": int((all_differences.max(axis=1) > 0.1).sum()),
        "reusable_patch_counts": observations["reusable_patch_counts"],
        "average_reusable_patches": (
            float(np.mean(observations["reusable_patch_counts"]))
            if observations["reusable_patch_counts"]
            else None
        ),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== VLA-CACHE STAGE PROFILE ===")
    print("measured frames:", summary["measured_frames"])
    print("baseline wall median ms:", round(baseline_wall, 3))
    print("cache wall median ms:", round(cache_wall, 3))
    print("wall speedup:", round(summary["comparison"]["wall_speedup"], 3))
    print("latency change %:", round(summary["comparison"]["latency_change_percent"], 3))
    print("action MAE:", round(summary["comparison"]["action_mae"], 6))
    print("action max difference:", round(summary["comparison"]["action_maximum_difference"], 6))
    print("large-difference frames:", summary["comparison"]["large_difference_frames"])
    print("average reusable patches:", summary["comparison"]["average_reusable_patches"])
    for mode in ("baseline", "cache"):
        print(f"\n{mode} stage medians (ms):")
        for name, values in sorted(summary["modes"][mode]["stages"].items()):
            print(" ", name, "=", None if values["median"] is None else round(values["median"], 3))
    print("\nSaved:", output_path)
    print("VLA-CACHE STAGE PROFILE: OK")


if __name__ == "__main__":
    main()
