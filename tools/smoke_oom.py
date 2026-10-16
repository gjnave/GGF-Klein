"""Fault-injection check: restrict this test process, never exhaust the PC GPU."""
import json
from pathlib import Path
import runpy
import sys
import torch

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
request = root / "jobs/smoke-oom_request.json"
request.write_text(json.dumps({
    "prompt": "A studio photograph of a yellow ceramic mug on a seamless blue background, soft even lighting.",
    "width": 512, "height": 512, "seed": 1, "enhance": False,
}), encoding="utf-8")
torch.cuda.set_per_process_memory_fraction(0.03)
sys.argv = [str(root / "worker.py"), str(request)]
runpy.run_path(str(root / "worker.py"), run_name="__main__")
