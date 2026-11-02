"""Read-only parameter and detach audits. This module does not patch models."""


def callable_identity(value):
    return (getattr(value, "__self__", None), getattr(value, "__func__", value))


class ParameterAudit:
    def __init__(self, model, sample_tensors=16, values_per_tensor=4096):
        if sample_tensors < 1 or values_per_tensor < 1:
            raise ValueError("Positive sample budget required")
        import torch
        self.torch = torch
        self.original = dict(model.named_parameters())
        if not self.original:
            raise ValueError("No model parameters to audit")
        values = list(self.original.items())
        count = min(sample_tensors, len(values))
        selected = sorted({round(i * (len(values) - 1) / max(count - 1, 1)) for i in range(count)})
        self.samples = [(values[i][0], values[i][1],
                         values[i][1].detach().reshape(-1)[:values_per_tensor].cpu().clone()) for i in selected]
        self.limit = values_per_tensor

    def active(self, model):
        registered = {id(value) for value in model.parameters()}
        checks = [{"name_before_attach": name,
                   "original_object_still_registered": id(parameter) in registered,
                   "sampled_values_equal": self.torch.equal(before, parameter.detach().reshape(-1)[:self.limit].cpu())}
                  for name, parameter, before in self.samples]
        return {"sampled_original_values_unchanged": all(x["sampled_values_equal"] for x in checks),
                "all_original_parameter_objects_still_registered": all(id(p) in registered for p in self.original.values()),
                "sampled_tensors": len(checks), "values_per_tensor_max": self.limit, "checks": checks,
                "scope": "sampled original tensors only; replaced parameter objects require a separate value mapping audit"}

    def detached(self, model):
        current = dict(model.named_parameters())
        return {"parameter_names_restored": set(current) == set(self.original),
                "parameter_objects_restored": set(current) == set(self.original) and
                    all(current[k] is self.original[k] for k in current),
                "active_sample_check": self.active(model)}


class DetachAudit:
    def __init__(self, model):
        self.original = dict(model.named_modules())
        self.forward = {name: callable_identity(module.forward) for name, module in self.original.items()}
        self.hooks = {name: self.hook_counts(module) for name, module in self.original.items()}
        self.predict = callable_identity(getattr(model, "predict_action", None))

    @staticmethod
    def hook_counts(module):
        return {name: len(getattr(module, name, {})) for name in
                ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")}

    def check(self, model):
        current = dict(model.named_modules())
        same = set(current) == set(self.original) and all(current[k] is self.original[k] for k in current)
        return {"module_objects_restored": same,
                "forward_callables_restored": same and all(callable_identity(current[k].forward) == self.forward[k] for k in current),
                "hook_counts_restored": same and all(self.hook_counts(current[k]) == self.hooks[k] for k in current),
                "predict_action_restored": callable_identity(getattr(model, "predict_action", None)) == self.predict,
                "native_output_restored": None,
                "scope": "structure and callables; buffers/config/cache and native output require method-specific checks"}
