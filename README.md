# GGF Flux 3 Klein - Dream

Create and edit images locally with FLUX.3 Image Klein. This is a **development
preview**, tested on an RTX 4090, not a claim that every NVIDIA card is supported.
The PC does the work; optional phone access is the same interface in your browser.
No separate ComfyUI installation or server is required.

## Install and run

Extract the slim installer ZIP into a writable folder on an internal SSD.
Double-click **INSTALL.bat**, accept the original GGF disclaimer, and follow the
Hugging Face sign-in instructions. Obtain access to the official model repository:
https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy

The installer uses UV to download its own Windows x64 Python 3.12. It does not use
or replace system Python. You need a compatible NVIDIA GPU and current driver;
it installs the PyTorch CUDA 13 runtime wheels, **not the CUDA Toolkit**.
CUDA 13 supports RTX 20/30/40/50-series GPU architectures, not only RTX 50.
Use a current NVIDIA driver from the R580 family or newer. GTX 10-series and
older architectures are not supported by these wheels. Architecture support
does not guarantee enough VRAM; this app's full generation tests used RTX 4090.
Allow about 21 GB for the default four model files plus Python/dependencies and
your outputs. Optional BF16 and GGUF image models are separate downloads.
INSTALL.bat downloads all four required models (image, text encoder, VAE and
prompt assistant) and verifies their SHA-256 hashes before declaring success.
If access or downloading fails, setup stays incomplete. Rerun INSTALL.bat to
resume; existing models, partial downloads and saved images are preserved.

Double-click **RUN.bat** after setup. Generate from a description, or upload
references and describe the changes you want. The local prompt assistant expands
your request; you can inspect and edit that caption under image options. To use
an edited caption unchanged, disable expansion. The distilled checkpoint uses
16 steps without classifier-free guidance. No unsupported LoRA/scheduler controls
are exposed. Source, environment, models and generated images are separate.

## Use it from your phone

Open Settings → Use it from your phone. Select Wi-Fi/LAN or a temporary public
link, optionally set a username/password, save and restart RUN.bat on the PC.
A blank password explicitly disables remote login. Local access always remains
login-free. Open the remote link on your phone while the PC stays running.
Public links may change on restart. Anyone with an unprotected link can use it.

## Reference editing and layout

Up to ten references; the first is the main edit image. Open View / reorder
references to change their order or remove one from the workspace (not disk).
Use History to select a result and make it the next input. Before/after comparison
is below the output. All PNG exports have unique filenames and metadata sidecars.

Composition boxes use percentages. Preview the guide, describe an element and
tap two opposite corners, or edit the coordinate table. Boxes are layout guidance,
not hard inpaint masks: the model may change pixels outside a box.

## Jobs, memory and updates

Press Ctrl+Enter or Generate to start. Generate turns grey while your job runs,
with loading/sampling progress beside it, and returns to normal when finished.
One GPU job runs at a time. Requests from other browsers queue; batches run sequentially and
reuse one expanded caption. Stop cancels your browser's active/queued jobs.
Unchanged requests also reuse their expanded caption on later runs; use Expand
prompt only to force a fresh expansion.
Closing/reloading the page does not stop work. The same browser and URL restore
your workspace; a new public hostname is a different browser origin.

Workers exit after each image, releasing GPU resources. Release models keeps
the UI/phone link open. Stop server requires confirmation and a local PC restart.
Out-of-memory failures preserve saved images; try smaller dimensions, fewer
references or aggressive offloading. The assistant can use more memory than the
image model. There is no arbitrary 16 GB minimum; smaller GPUs remain unverified.

Settings checks source-repository revisions. Update & restart downloads a
host-generated archive from Codeberg, falling back to GitHub. It backs up source,
updates dependencies and preserves models, settings, workspaces and outputs.
UPDATE.bat provides the same flow with the app closed. No maintained source ZIP
is required for updates. Source mirror verification is recorded in the handoff.

## Source installation / development

Inside a source checkout, use the pinned UV Windows executable (0.12.13), then:

```bat
uv venv --managed-python --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\python.exe preflight.py
.venv\Scripts\hf.exe auth login
.venv\Scripts\python.exe model_manager.py
.venv\Scripts\python.exe app.py
```

For automatic environment provisioning from the checkout, run the BATs in the
`installer` folder; they create their app folder beside those BATs. No system
Python or Git is needed by customers. Required model installation cannot be
skipped by the customer installer. Set `GGF_UNATTENDED=1` only for controlled
install checks; it skips pauses, never models.

Developer tests: `python -m unittest discover -s tests`. Real generation test
recipes and measurements are in VALIDATION.md. Logs remain local in `logs`;
Settings exports a small diagnostics ZIP that excludes credentials, images and
prompts. Logs themselves can contain prompts/local paths; review before sharing.

## Remove or relocate

Close this app first. Back up or move outputs, models and workspaces before
manually removing its folder. There is no automatic cleanup/deletion. Changing
the model directory in Settings does not move or delete existing models.

## Terms

App and embedded Comfy inference source: GPL-3.0-or-later, see LICENSE.
BFL reference helpers and city96 GGUF helpers retain their Apache-2.0 licenses.
Image weights use the FLUX Kommunity License, see MODEL-LICENSE.md. Users must
obtain authorized access themselves and review applicable commercial-use terms.
Review AI outputs before sharing. See UPSTREAM-NOTICES.md for exact source pins.
