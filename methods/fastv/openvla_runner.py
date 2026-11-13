"""Run the existing LIBERO evaluator with baseline or FastV inference."""
import os
import argparse
import runpy
import sys
from pathlib import Path
from types import MethodType

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["baseline", "fastv"], required=True)
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--prune-ratio", type=float, default=0.5)
args = parser.parse_args()

if not 1 <= args.trials <= 50:
    parser.error("--trials must be between 1 and 50")
if not 0 <= args.prune_ratio < 1:
    parser.error("--prune-ratio must be in [0, 1)")

root = Path(os.environ["FASTV_WORKSPACE"])
sys.path.insert(0, str(root))

import tensorflow as tf
tf.config.set_visible_devices([], "GPU")

from experiments.robot import openvla_utils

if args.mode == "fastv":
    from experiments.robot.fastv_openvla import fastv_forward

    original_get_vla = openvla_utils.get_vla

    def get_vla_with_fastv(cfg):
        model = original_get_vla(cfg).eval()
        decoder = model.language_model.model

        if not 1 <= args.k < len(decoder.layers):
            raise ValueError("--k must be between 1 and number_of_layers - 1")

        settings = {
            "fastv_k": args.k,
            "fastv_r": args.prune_ratio,
            "image_token_start_index": 1,
            "image_token_length": None,
        }
        printed = False

        def capture_visual_length(module, inputs, output):
            settings["image_token_length"] = int(output.shape[1])

        def adapted_forward(self, *forward_args, **forward_kwargs):
            nonlocal printed
            visual = settings["image_token_length"]
            if visual is None:
                raise RuntimeError("Visual token length has not been captured")
            if round(visual * (1 - settings["fastv_r"])) < 1:
                raise ValueError("Pruning must retain at least one visual token")

            result = fastv_forward(
                self,
                *forward_args,
                **forward_kwargs,
                fastv_config=settings,
            )

            if not printed:
                lengths = [
                    int(key.shape[-2])
                    for key, value in result.past_key_values
                ]
                removed = visual - round(visual * (1 - settings["fastv_r"]))
                k = settings["fastv_k"]

                assert len(set(lengths[:k])) == 1
                assert all(
                    length == lengths[0] - removed
                    for length in lengths[k:]
                ), "Unexpected pruned cache lengths"

                print("FastV config:", settings, flush=True)
                print("Prefill cache lengths:", lengths, flush=True)
                print("Visual tokens removed:", removed, flush=True)
                printed = True

            return result

        model.projector.register_forward_hook(capture_visual_length)
        decoder.forward = MethodType(adapted_forward, decoder)
        print("FastV adapter enabled", flush=True)
        return model

    openvla_utils.get_vla = get_vla_with_fastv

checkpoint = Path(os.environ["FASTV_CHECKPOINT"])

tag = (
    "baseline-eager-cublas124"
    if args.mode == "baseline"
    else f"fastv-k{args.k}-r{args.prune_ratio:g}-eager-cublas124"
)

sys.argv = [
    str(root / "experiments/robot/libero/run_libero_eval.py"),
    "--model_family", "openvla",
    "--pretrained_checkpoint", str(checkpoint),
    "--task_suite_name", "libero_spatial",
    "--center_crop", "True",
    "--num_trials_per_task", str(args.trials),
    "--seed", str(args.seed),
    "--use_wandb", "False",
    "--run_id_note", f"{tag}-n{args.trials}-seed{args.seed}",
    "--local_log_dir", os.environ["FASTV_OUTPUT"],
]

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from methods.fastv.live import OpenVLARecorder
recorder=OpenVLARecorder(os.environ['FASTV_OUTPUT'],args.mode,args.trials,args.seed).install()
completed=False
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
    completed=True
finally:
    recorder.close(completed)
