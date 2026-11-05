"""Experimental transactional attachment; numerical parity still requires a real probe."""

_MISSING = object()


class ModelSnapshot:
    def __init__(self, model):
        self.saved = []
        decoder = model.language_model.model
        for obj, names in (
            (model, ("predict_action", "_orig_predict_action")),
            (decoder, ("forward", "_specprune_orig_forward", "dynamic_importance_confidence",
                       "dynamic_importance_episode_key", "dynamic_importance_last_infer_step",
                       "token_importance_scores")),
        ):
            for name in names:
                self.saved.append((obj, name, vars(obj).get(name, _MISSING)))
        self.attentions = [(layer, layer.self_attn) for layer in decoder.layers]
        self.decoder = decoder

    def isolate(self):
        for name in ("dynamic_importance_confidence", "dynamic_importance_episode_key",
                     "dynamic_importance_last_infer_step", "token_importance_scores"):
            setattr(self.decoder, name, {} if name == "dynamic_importance_confidence" else None)

    def restore(self):
        for layer, attention in self.attentions:
            layer.self_attn = attention
        for obj, name, value in reversed(self.saved):
            if value is _MISSING:
                if name in vars(obj):
                    delattr(obj, name)
            else:
                setattr(obj, name, value)
