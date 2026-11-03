"""Reusable OpenVLA-OFT loader and explicit user-factory contract."""
import copy
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict
from types import SimpleNamespace

from .backend import load_backend, BACKEND_COMMIT


def load_observation(directory):
    import numpy as np
    directory = Path(directory).resolve()
    metadata = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    path = directory / "observation.npz"
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get("passed") is not True or sha != metadata.get("observation_sha256"):
        raise ValueError(f"Observation checksum/provenance invalid: {directory}")
    with np.load(path, allow_pickle=False) as values:
        observation = {key: values[key].copy() for key in ("full_image", "wrist_image", "state")}
    observation.update(instruction=metadata["task_description"], id=sha,
                       task_id=metadata.get("task_id"), trial=metadata.get("trial"))
    return observation


def load_bundle(args, cfg):
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise ValueError("workspace must be an existing model import root")
    sys.path.insert(0, str(workspace))
    if args.factory:
        module, separator, name = args.factory.partition(":")
        if not separator or not module or not name:
            raise ValueError("factory must be module:function")
        options = json.loads(args.factory_kwargs)
        if not isinstance(options, dict):
            raise ValueError("factory-kwargs must be a JSON object")
        bundle = getattr(importlib.import_module(module), name)(**options)
        if not isinstance(bundle, dict):
            raise ValueError("factory must return a dict")
        return bundle
    if cfg.checkpoint is None or not Path(cfg.checkpoint).is_dir():
        raise ValueError("Built-in loader requires an existing local --checkpoint directory")
    if not args.observation_dir:
        raise ValueError("Built-in loader requires at least one --observation-dir")
    repo = Path(subprocess.check_output(["git", "-C", str(workspace), "rev-parse", "--show-toplevel"], text=True).strip())
    expected = repo / "src/openvla-oft"
    if workspace != expected.resolve():
        raise ValueError("Built-in loader workspace must be the official src/openvla-oft import root")
    subprocess.run(["git", "-C", str(repo), "diff", "--quiet", BACKEND_COMMIT, "--",
                    "src/openvla-oft/prismatic", "src/openvla-oft/transformers",
                    "src/openvla-oft/experiments/robot/openvla_utils.py"], check=True)
    observations = [load_observation(p) for p in args.observation_dir]
    backend = load_backend()
    if Path(backend.utils.__file__).resolve() != workspace / "experiments/robot/openvla_utils.py":
        raise ValueError("Imported model utilities from a different workspace")
    checkpoint = Path(cfg.checkpoint)
    source = workspace / "prismatic/extern/hf/modeling_prismatic.py"
    if source.read_bytes() != (checkpoint / source.name).read_bytes():
        raise ValueError("Checkpoint model code differs from pinned backend; review before loading")
    resources = backend.load_model(cfg)
    model = resources["model"]
    if Path(inspect.getfile(type(model))).read_bytes() != source.read_bytes():
        raise ValueError("Loaded remote model code differs from inspected source")
    values = asdict(cfg)
    values.update(use_vla_cache=False, use_fastv=False, use_vla_pruner=False)
    native_cfg = SimpleNamespace(**values)
    def native_query(observation):
        import torch
        with torch.inference_mode():
            actions, _, _, _ = backend.get_vla_action(native_cfg, model, resources["processor"],
                copy.deepcopy(observation), observation["instruction"],
                action_head=resources["action_head"], proprio_projector=resources["proprio_projector"],
                use_film=cfg.use_film, last_caches=None)
        return actions
    resources.update(observations=observations, native_query=native_query,
                     native_reset=model.reset_av_history,
                     provenance={"backend_head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
                                 "core_matches_pinned": True, "workspace": str(workspace),
                                 "checkpoint": str(checkpoint), "loaded_model_file": inspect.getfile(type(model))})
    return resources
