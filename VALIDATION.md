# Validation record — 2026-10-15

Hardware actually used: Windows x64, NVIDIA RTX 4090 (24GB), 128GB system RAM.
Development Python 3.12.13; clean managed install additionally tested 3.12.14.
Torch 2.11.0+cu128, torchvision 0.26.0+cu128, Gradio 6.29.0.
Peak measurements below are PyTorch allocated VRAM and process RSS sampled at
0.25 seconds, **not whole-PC memory usage or guarantees for another GPU**.

## Completed generations

| Test | Size | Image model | Seconds | Peak allocated VRAM GiB |
| --- | --- | --- | ---: | ---: |
| Text-to-image, assistant enabled | 768×768 | Official INT8 | 60.50 | 16.872 |
| Reference edit, assistant enabled | 768×768 | Official INT8 | 58.34 | 16.901 |
| Generation from browser | 768×768 | Official INT8 | 64.72 | 16.872 |
| Already-expanded caption | 768×768 | Official BF16 DiT + INT8 text | 10.83 | 10.630 |
| Already-expanded caption | 768×768 | Community GGUF Q4 + INT8 text | 10.80 | 4.819 |
| Assistant + aggressive offload, 16GiB reserved | 512×512 | GGUF Q4 | 71.90 | 9.686 |
| Assistant + locked layout coordinates | 512×512 | Official INT8 | 41.11 | 16.882 |
| Browser batch, image one (full caption) | 512×512 | Official INT8 | 9.28 | 5.923 |
| Browser batch, image two (full caption) | 512×512 | Official INT8 | 9.43 | 5.923 |
| Ten-reference tensor path, full caption supplied | 512×512 | GGUF Q4 | 25.20 | 4.783 |
| Browser edit, assistant enabled | 768×768 | Automatic → BF16 image | 56.37 | See per-image sidecar |
| Repeat browser edit, same caption reused | 768×768 | Automatic → BF16 image | 14.46 | See per-image sidecar |

Every image has its own JSON sidecar containing exact model, seed, caption,
timings by stage, measured VRAM/reserved VRAM and peak process RSS. The first
image was visually reviewed: red bicycle, pale wall, lavender pot. The edit
changed the frame to blue and preserved the scene. BF16 and Q4 outputs were
visually reviewed; both produced the requested scene. All runs used 16 Euler
flow-map steps with alpha 6.93 and no classifier-free guidance.

The assistant dominates end-to-end latency. Its normal peak was about 16.9GiB;
reserving 16GiB and offloading reduced the measured peak to 9.7GiB but slowed
the assistant. **This does not prove an 8GB configuration.** Reserving memory
is a loader preference, not a hard cap. The Q4 image model's 4.8GiB measurement
does not include the assistant in that run; never advertise that as total app
VRAM. The current core requires CUDA 13 wheels for optimized INT8 CUDA kernels;
CUDA 12.8 used a working but slower fallback. The user approved CUDA 13 on
October 16; this historical table remains the cu128 baseline. New results below.

## Reliability checks

### October 16 — GGF Flux 3 Klein - Dream / CUDA 13

User-approved CUDA 13 switch affects this app's private environment only.
Official torch 2.11.0+cu130 and torchvision 0.26.0+cu130; driver 616.56,
RTX 4090 (Ada, compute 8.9). CUDA matrix operation, pip check and full inference
imports passed. Compiled targets: sm_75, sm_80, sm_86, sm_90, sm_100, sm_120.
These include pre-Blackwell architectures; no physical 20/30/50 card was tested.
Current R580-or-newer driver required. No CUDA Toolkit installed.

Full 768×768 official INT8 text-to-image, seed 7, local assistant enabled:
31.64 seconds, peak allocated 11.188 GiB. Earlier cu128 run of the same short
request/seed/size/model path was 60.50 seconds. Prompt assistant is generative,
so expanded captions differ; this is an observed end-to-end comparison, not a
controlled identical-caption kernel benchmark. Output:
outputs/ggf-klein_20261016_093224_66933e55.png and JSON sidecar.
Smaller physical GPUs remain unvalidated; do not turn this peak into a VRAM
minimum guarantee.

- Eight automated tests passed: stable workspace token restoration; coordinate
  conversion/validation; protected/unsafe archive paths including case variants;
  refusal to install private data; blank password clears authentication; UI builds;
  portrait reference dimensions; imports do not start/recover the queue.
- The actual browser completed a generation through the queue.
- A page reload restored the edited prompt after correcting a Gradio State/JS
  transport issue. There is now exactly one load handler and a hidden textbox
  transports the localStorage token to server State.
- Reference upload in the actual browser opened its thumbnail preview and chose
  matching dimensions. Uploads live outside the disposable Gradio cache.
- Repeating the same reference edit reused the real expanded caption: 56.37s
  first run versus 14.46s repeat. Changing instructions, references or assistant
  file invalidates the cache; Expand prompt only explicitly forces a fresh one.
- Queue recovery/consumer initialization occurs only after the owning server's
  instance check, not on module import. Tests and a duplicate launch cannot reset
  another server's running job state.
- A real browser Stop cancelled its owned job and preserved outputs. A later
  two-image batch completed, one worker at a time. GPU returned to 480MiB idle.
- A deliberately restricted test-process CUDA allocator produced the friendly
  out-of-memory error, without exhausting the PC. This is fault injection, not
  a successful low-VRAM hardware profile. The disposable process exited.
- Clean CMD installer in a separate empty app folder bootstrapped verified UV,
  downloaded managed Python, installed all dependencies, passed `uv pip check`,
  a real CUDA matrix operation and complete core-loader imports. Stopped at
  `/dependencies-only`: no model downloads in the install check.
- Installer fallback was exercised with a deliberately unreachable first source
  and a local source-archive fixture second. Live host archives must separately
  pass after publication. No manually maintained source ZIP is an update input.
- The clean-install check caught an exclusion rule that incorrectly removed
  `vendor/comfy_core/comfy/ldm/models`. It was corrected before publishing; root
  model exclusions are now anchored, and the complete package passed imports.
- Mobile-size browser rendering inspected at 390×844. Text/dropdowns use explicit
  dark foreground/background pairing; accordion overflow was corrected.
- No existing GGF application/environment changed; no unrelated GPU processes killed.

## Still not verified

Physical 8GB/12GB/16GB/Blackwell systems, CUDA 13, FP8 and NVFP4, alternative
text-encoder precision, arbitrary extreme dimensions, prompt/model quality over
a large benchmark, long-duration soak testing, and live release mirrors.
Ten-reference testing used repeated copies of the same generated image: it
proves the tensor path, not ten distinct identities/composition or a ten-image
assistant expansion. One-reference assistant editing was tested.
Public phone transport and a real Android upload need an end-user check.

## Reproduce

Run `preflight.py`, then `python -m unittest discover -s tests`.
`worker.py <request.json>` emits progress/caption/result/error JSON prefixed by
`__GGF__`. Request keys include prompt, width, height, seed, enhance, precision
(`int8`, `bf16`, `gguf_q4`, `auto`), references, normalized boxes, reserve_vram
and low_vram. `tools/smoke_oom.py` intentionally fails under a tiny process-only
CUDA allowance. `tools/smoke_references.py <image>` tests ten ordered references.
Use app queue tests when the server is running; do not launch parallel manual
CUDA workers beside another job.
