from __future__ import annotations

from lightvla_pruner import (
    DifferentiableTokenPruner,
    PrefixLlamaPruningAdapter,
    StaticPrefixLayoutResolver,
)


def attach_pruner_to_prefix_vla(vla_model, num_patches_per_image: int, num_images_in_input: int = 1):
    """
    Example integration for OpenVLA/Prismatic/LLaVA-style models.

    Assumptions:
    - the multimodal sequence layout is [BOS][visual patch tokens][text/action tokens...]
    - `vla_model.language_model.model` is the decoder backbone that receives `inputs_embeds`
    - the decoder hidden size is available as `vla_model.config.text_config.hidden_size`
    """

    patch_count = num_patches_per_image * num_images_in_input
    hidden_size = vla_model.config.text_config.hidden_size

    pruner = DifferentiableTokenPruner(hidden_size=hidden_size)
    layout_resolver = StaticPrefixLayoutResolver(
        patch_count=patch_count,
        patch_start=1,
        score_context="suffix",
    )

    vla_model.language_model.model = PrefixLlamaPruningAdapter(
        decoder_model=vla_model.language_model.model,
        pruner=pruner,
        layout_resolver=layout_resolver,
    )

    return vla_model
