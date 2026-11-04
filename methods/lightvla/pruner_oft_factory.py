"""Experimental LightVLA-style pruning on the same pinned OFT weights as VLA-Pruner."""
from pathlib import Path
from types import SimpleNamespace


def build(workspace, checkpoint, observation_dir, unnorm_key="libero_object_no_noops"):
    import numpy as np
    from methods.vla_pruner.accelerator import AcceleratorConfig
    from methods.vla_pruner.runtime import load_bundle
    cfg=AcceleratorConfig(checkpoint=str(Path(checkpoint).resolve()),unnorm_key=unnorm_key)
    args=SimpleNamespace(workspace=Path(workspace),factory=None,observation_dir=[Path(observation_dir)])
    bundle=load_bundle(args,cfg)
    def run():
        bundle["native_reset"]()
        return np.asarray(bundle["native_query"](bundle["observations"][0]))
    return dict(model=bundle["model"],run=run,provenance=dict(bundle["provenance"],
        deployment_recommended=False,
        method_scope="LightVLA-style inference intervention on existing OFT weights; not original fine-tuned LightVLA reproduction"))
