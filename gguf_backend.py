"""Direct GGUF loading using city96's unmodified ops and model patcher."""
import torch
import comfy.sd
from vendor.gguf_core.loader import gguf_sd_loader
from vendor.gguf_core.ops import GGMLOps
from vendor.gguf_core.patcher import GGUFModelPatcher


def load(path):
    state, extra = gguf_sd_loader(str(path))
    ops = GGMLOps()
    ops.Linear.dequant_dtype = "target"
    model = comfy.sd.load_diffusion_model_state_dict(
        state, model_options={"custom_operations": ops, "dtype": torch.bfloat16},
        metadata=extra.get("metadata", {}), disable_dynamic=True,
    )
    if model is None:
        raise ValueError("The selected GGUF does not contain a recognized FLUX.3 model.")
    return GGUFModelPatcher.clone(model)
