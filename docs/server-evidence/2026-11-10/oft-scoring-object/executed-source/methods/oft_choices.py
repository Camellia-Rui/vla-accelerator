"""Exclusive, reversible inference choices for an audited pinned OFT bundle."""
from dataclasses import replace


class OFTChoice:
    modes=("baseline","lightvla","vla_pruner")

    def __init__(self,bundle,config,scoring=False):
        if scoring:
            self.modes = ("baseline", "fastv_scoring", "vla_pruner")
        self.bundle,self.config=bundle,config
        self.adapter=None
        self.mode="baseline"

    def select(self,mode):
        if mode not in self.modes:
            raise ValueError("Unknown OFT method: "+str(mode))
        self.close()
        model=self.bundle["model"]
        if mode=="lightvla":
            from methods.lightvla import LightVLAAccelerator
            self.adapter=LightVLAAccelerator.install(model,mode="packaged")
        elif mode in ("vla_pruner", "fastv_scoring"):
            from methods.vla_pruner.accelerator import VLAPrunerAccelerator
            self.adapter=VLAPrunerAccelerator.install(model,config=replace(self.config, vla_pruner_mode="fastv" if mode == "fastv_scoring" else self.config.vla_pruner_mode),
                **{k:self.bundle[k] for k in ("processor","action_head","proprio_projector")})
        self.mode=mode
        self.reset()

    def reset(self):
        self.bundle["native_reset"]()
        if self.adapter is not None:self.adapter.reset()

    def query(self,observation):
        from methods.vla_pruner.evaluation import actions_array,query
        if self.mode in ("vla_pruner", "fastv_scoring"):return query(self.adapter,observation)
        return actions_array(self.bundle["native_query"](observation))

    def stats(self):
        return self.adapter.get_stats() if self.adapter is not None else {}

    def close(self):
        if self.adapter is not None:
            self.adapter.detach()
            self.adapter=None
        self.mode="baseline"
        self.bundle["native_reset"]()
