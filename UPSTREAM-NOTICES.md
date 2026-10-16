# Third-party source and models

This app does not require or launch a ComfyUI server.

- `vendor/comfy_core/comfy` and `node_helpers.py`: ComfyUI inference source,
  https://github.com/Comfy-Org/ComfyUI at
  `5a665890ded5020c282f63c4ea2dd9e083e025ff`, GPL-3.0-or-later. Unmodified files;
  complete inference package retained for its dynamic loader imports. Full license
  is `vendor/comfy_core/LICENSE`.
- `vendor/bfl/flux3`: selected original BFL reference helpers and prompt templates,
  https://github.com/black-forest-labs/flux-3-image-klein at
  `bbfe317dd81c825c30d76090a67d3d0db51da3cc`, Apache-2.0. No vLLM runtime used.
  Full license is `vendor/bfl/LICENSE`.
- `vendor/gguf_core`: city96 loader, quantization ops and dequantization plus
  GGUFModelPatcher extracted unchanged from its nodes module, excluding node/UI
  imports. https://github.com/city96/ComfyUI-GGUF at
  `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`, Apache-2.0.
  Full license is `vendor/gguf_core/LICENSE`; extraction script is in tools.
- FLUX.3 image model weights: BFL FLUX Kommunity License v1.0, not Apache-2.0.
  Model downloads are direct, user-authorized, pinned and verified. See
  MODEL-LICENSE.md and model-manifest.json. No weights are distributed in this
  repository or installer. Qwen3-VL-4B and the BFL prompt assistant are Apache-2.0.
- PyTorch, Gradio, Transformers, UV and other packages retain their own licenses
  in installed package metadata. The Python dependency versions are pinned.

BFL and ComfyUI names describe underlying technology, not endorsement.
