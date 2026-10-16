"""Actual CUDA operation and full inference import closure, not just imports."""
from pathlib import Path
import sys
import torch

root = Path(__file__).resolve().parent
if sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
    raise SystemExit("This build requires managed Windows x64 Python 3.12.")
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected. Install a current NVIDIA driver and rerun setup.")
print("GPU:", torch.cuda.get_device_name(0))
print("Compute capability:", torch.cuda.get_device_capability(0))
print("Wheel architectures:", torch.cuda.get_arch_list())
x = torch.ones((32, 32), device="cuda")
assert (x @ x).mean().item() == 32.0
sys.path.insert(0, str(root / "vendor/comfy_core"))
import comfy.sd
import comfy.sample
import comfy.samplers
print("CUDA operation and app-owned inference imports passed. Models not loaded.")
