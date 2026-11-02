"""Observe decoder block inputs BEFORE computation. Counts are not FLOP estimates."""


class LayerInputTrace:
    def __init__(self, layers):
        self.layers = list(layers)
        self.handles = []
        self.events = []
        self.forward_id = -1

    def clear(self):
        self.events.clear()
        self.forward_id = -1

    def attach(self):
        if self.handles:
            raise RuntimeError("Trace already attached")
        try:
            for index, layer in enumerate(self.layers):
                def capture(module, args, kwargs, index=index):
                    hidden = args[0] if args else kwargs.get("hidden_states")
                    if hidden is None or len(hidden.shape) != 3:
                        raise ValueError("Expected actual [B,S,D] block input")
                    if index == 0:
                        self.forward_id += 1
                    self.events.append({"forward_id": self.forward_id, "layer": index,
                                        "batch": int(hidden.shape[0]), "tokens": int(hidden.shape[1]),
                                        "hidden_size": int(hidden.shape[2]), "site": "forward_pre_hook"})
                self.handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
        except Exception:
            self.detach()
            raise
        return self

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self):
        shorter = []
        starts = {}
        for event in self.events:
            frame = event["forward_id"]
            if event["layer"] == 0:
                starts[frame] = event["tokens"]
            initial = starts.get(frame)
            if initial and initial > 1 and event["layer"] > 0 and 0 < event["tokens"] < initial:
                shorter.append(event)
        return {"events": list(self.events), "shorter_downstream_inputs": shorter,
                "shortening_observed": bool(shorter),
                "scope": "block input shapes only; not a complete method-specific mechanism or admission proof"}

    def __enter__(self):
        return self.attach()

    def __exit__(self, *args):
        self.detach()
