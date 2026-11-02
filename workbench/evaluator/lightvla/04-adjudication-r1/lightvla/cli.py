"""Run a synthetic smoke check or one query supplied by the user's model project."""

import argparse
import importlib
import json
from pathlib import Path
import sys

from .accelerator import LightVLAAccelerator
from .config import LightVLAConfig


def _smoke_bundle():
    import torch
    from transformers import LlamaConfig, LlamaModel

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.model = LlamaModel(LlamaConfig(
                vocab_size=32, hidden_size=16, intermediate_size=32,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=64,
            ))

        def get_num_patches(self):
            return 4

        def predict_action(self, **inputs):
            # Synthetic decoder output, not robot actions.
            return self.language_model.model(**inputs).last_hidden_state

    torch.manual_seed(0)
    model = TinyPolicy().eval()
    tokens = torch.randn(1, 8, 16)
    tokens[:, 1:5] = tokens[:, 1:2].clone().expand(-1, 4, -1)
    return {"model": model, "inputs": {"inputs_embeds": tokens, "use_cache": False}}


def _config(path):
    if path is None:
        return LightVLAConfig()
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Reading YAML requires PyYAML; JSON configs need no YAML dependency") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config must contain an object/mapping")
    return LightVLAConfig.from_mapping(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smoke", action="store_true", help="tiny random CPU LLaMA; no weights or dataset")
    source.add_argument("--factory", help="module:function returning model and prepared inputs or a run callable")
    parser.add_argument("--workspace", type=Path, help="model project import root; relative paths still use the current directory")
    parser.add_argument("--factory-kwargs", default="{}", help="JSON keyword arguments for the factory")
    parser.add_argument("--config", type=Path, help="JSON or YAML LightVLA config")
    parser.add_argument("--mode", choices=("packaged", "baseline", "builtin"))
    parser.add_argument("--output", type=Path, help="write summary JSON to a new file")
    args = parser.parse_args(argv)
    acc = None
    try:
        config = _config(args.config)
        if args.mode is not None:
            from dataclasses import replace
            config = replace(config, mode=args.mode)
        if args.smoke:
            bundle = _smoke_bundle()
        else:
            if args.workspace:
                if not args.workspace.is_dir():
                    raise ValueError("workspace must be an existing directory")
                sys.path.insert(0, str(args.workspace.resolve()))
            module_name, separator, function_name = args.factory.partition(":")
            if not separator or not module_name or not function_name:
                raise ValueError("factory must be module:function")
            options = json.loads(args.factory_kwargs)
            if not isinstance(options, dict):
                raise ValueError("factory-kwargs must be a JSON object")
            factory = getattr(importlib.import_module(module_name), function_name)
            bundle = factory(**options)
        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError("factory must return a dict containing model")
        model = bundle["model"]
        if model.training:
            raise ValueError("factory must return model.eval()")
        run = bundle.get("run")
        if run is not None and not callable(run):
            raise ValueError("bundle.run must be callable")
        inputs = bundle.get("inputs")
        if run is None and not isinstance(inputs, dict):
            raise ValueError("factory must provide inputs (dict) or run (zero-argument callable)")
        acc = LightVLAAccelerator.install(model, config=config)
        if run is not None:
            with acc._components.torch.inference_mode():
                output = run()
        else:
            output = acc.predict_action(**inputs)
        report = {"check": "synthetic-smoke-only" if args.smoke else "user-factory-query",
                  "output_type": type(output).__name__,
                  "output_shape": list(output.shape) if hasattr(output, "shape") else None,
                  "config": acc.describe()["config"],
                  "stats": acc.get_stats()}
        # Restoration is part of success; do it before reporting success.
        acc.detach()
        report["detached"] = True
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as file:
                file.write(text + "\n")
        print(text)
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2
    finally:
        if acc is not None and acc.model is not None:
            acc.detach()


if __name__ == "__main__":
    raise SystemExit(main())
