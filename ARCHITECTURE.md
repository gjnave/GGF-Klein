# GGF image app / FLUX.3 Klein

Status: working development preview; not a production website release. Updated 2026-10-15.

## Selected implementation

Use an app-owned Python 3.12 process with pinned Comfy inference modules, native
FLUX.3 loaders, INT8 convrot operations, Qwen3-VL-4B text encoding and Flux2 VAE.
No ComfyUI installation, server, browser frontend, workflow engine or node manager.
The complete `comfy` inference package is kept because model detection imports
multiple architectures dynamically. Removing apparently unrelated subdirectories
can break clean installations (as happened with `comfy.ldm.models` in Animate).
Only its small `node_helpers.py` root helper is included alongside the package.

The main interface will use the family's Gradio 6.29 navy/gold styling. This is a
local browser app, not a compiled native desktop executable. It retains desktop
and phone usability, independent local and optional remote servers, one GPU
worker, explicit cancellation and no compulsory remote password.

Run the prompt assistant, text encoder, reference encoder, image model and final
decoder sequentially, unloading between components. End the worker after each
job, so CUDA allocations cannot accumulate between jobs. This trades some reload
latency for predictable memory and reliable coexistence with other GGF apps.
No global Python or external ComfyUI is discovered or used.

The distilled model uses exactly 16 Euler flow-map steps, alpha 6.93 and no
classifier-free guidance. The native sampler supplies the target timestep to
the flow-map model. A generic old Flux pipeline is not interchangeable.

## Pinned upstreams

- BFL reference: https://github.com/black-forest-labs/flux-3-image-klein
  `bbfe317dd81c825c30d76090a67d3d0db51da3cc` (Apache-2.0 code).
- Inference core: https://github.com/Comfy-Org/ComfyUI
  `5a665890ded5020c282f63c4ea2dd9e083e025ff` (GPL-3.0-or-later).
- Official checkpoints: https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
  `5c9b714851dc072b04a23ef6d2d36421f5ccb3a0`.
- Community adapter investigated: https://github.com/city96/ComfyUI-GGUF.
- Alternative: https://huggingface.co/blog/flux-3 (new modular Diffusers path).

## Findings versus unverified options

| Format | Evidence | App status |
| --- | --- | --- |
| Official distilled INT8 convrot | Native loaders; files downloaded and SHA-256 verified | Generation, reference editing, layout and sequential browser batch passed on RTX 4090 |
| BF16 | Native core architecture and official checkpoint | Real 768×768 generation passed with INT8 text encoder on RTX 4090 |
| FP8 | Official reference implementation supports FP8; Windows reference dependency set includes vLLM/Triton | Not enabled; separate compatible conversion/loader and generation proof needed |
| NVFP4 | Official core checkpoint exists; accelerated hardware support requires Blackwell | Disabled on this RTX 4090; no Blackwell generation claimed |
| GGUF | Actual Abiray Q4_K_M metadata uses supported `flux` architecture; native core detects FLUX.3 tensors | Real generation and ten-reference tensor-path smoke passed via direct city96 operations; no custom-node/server import |

The official reference uses torch 2.11, vLLM 0.21 and Triton 3.6 and was tested
on B200. Installing that whole dependency set is not an acceptable Windows
baseline. Diffusers is another candidate for supported formats, not evidence
that it can load Comfy INT8 convrot checkpoints.

Upstream advertises INT8 on RTX 3070 / 8GB. That is an upstream statement, not
our measurement. This PC is an RTX 4090 / 24GB with 128GB system memory. No
8GB, 12GB, 16GB or Blackwell success can be inferred from this PC. Do not impose
an arbitrary 16GB minimum or advertise untested performance profiles.

CUDA 12.8 Torch 2.11 / torchvision 0.26 installed and passed a real CUDA matrix
operation. The current core disables accelerated Comfy Kitchen CUDA kernels
below CUDA 13. An app-specific CUDA 13 baseline has been requested from the user;
do not silently change the family's baseline or other apps' environments.

## Prompting, references and layout

Use the actual BFL fine-tuned Qwen3.5-9B prompt assistant with the shipped
`pup_prompts.json` system instructions, not canned prompt concatenation.
The assistant must see the edit references, in order. Native Qwen35 tokenizer
accepts a list of differently sized image tensors. Use BFL's reference resize,
alpha compositing and JPEG preprocessing. BFL bbox helpers preserve locked
identities and coordinates after expansion. The assistant is sequentially
unloaded before text/image inference. An advanced full-caption input can bypass
it explicitly; a raw one-line prompt should not be the default generation path.

Model limit is ten references. Reference zero is the primary edit image; order
matters. Layout annotations use `[top,left,bottom,right]` on a 0..1000 canvas.
Boxes are guidance, not hard inpaint masks. Do not present them as protection
against changes elsewhere in the image.

## Licenses and distribution

Reference code Apache-2.0; embedded inference core GPL-3.0-or-later. Keep full
notices and corresponding source in normal source repositories. Do not compile
or hide GPL source. Qwen text encoder and prompt assistant are Apache-2.0.

Image weights: FLUX Kommunity License v1.0, **not Apache-2.0**:
https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md
It permits noncommercial use and has a conditional commercial-output exception
for qualifying users below US$5M gross annualized revenue including affiliates.
That is not a blanket right to sell or redistribute the model itself. Applicable
output review/filtering and disclosure requirements remain the user's obligation.
Review generated images before distributing them; mark generated exports as AI
in metadata. This is an engineering summary, not legal advice.

Download gated weights directly from BFL after users obtain access themselves.
Never publish weights in GGF installer ZIPs or upload them as a workaround for
the gate. User accepted this model gate on 2026-10-15; saved HF account can now
download. Never commit tokens, account credentials, network settings or caches.

## Packaging contract

Outer CMD INSTALL/RUN/UPDATE BATs; private managed Python via pinned verified UV;
app source from host-generated Codeberg/GitHub revision archives. Keep source
repositories ordinary individual files, no maintained source ZIP dependency.
Models downloaded at install time or explicitly in Settings, with resumes and
SHA-256. A custom model folder is allowed; no automatic model deletion.

Customer ZIP allowlist: BATs, original about.nfo, original disclaimer.md, README
and required licenses only. No runtime, Python environment, models, source,
logs, tests, tokens or personal images. Updates preserve models/settings/outputs.
Publication waits until real generation/editing and a clean installer are proven.

## Handoff

Root: `D:\apps2review\flux3-klein`; app: `GGF-Klein`; research clones: `research`.
Private environment: `GGF-Klein\.venv` (3.12.13).
Model manifest records pinned file hashes; model_manager.py verifies downloads.
worker.py consumes a request JSON and emits `__GGF__` JSON events.
`smoke-request.json` is the first reproducible request, not a customer input.
Completed: actual generation metrics, visual output review, editing, repeated
jobs, cancellation, allocator-injected OOM handling, BF16 and GGUF trials,
browser workspace persistence and a clean dependency-only CMD installer check.
See GGF-Klein/VALIDATION.md for measurements and untested cases. On October 16
the user approved CUDA 13 wheels for this app only and named it
GGF Flux 3 Klein - Dream. Exact pins include +cu130 so rerunning setup replaces
an installed cu128 variant instead of incorrectly considering it satisfied.
Historical measurements above used cu128; new measurements are recorded separately.
Source mirrors: https://github.com/gjnave/GGF-Klein and
https://codeberg.org/Cognibuild/GGF-Klein. No existing GGF website/app changed.
