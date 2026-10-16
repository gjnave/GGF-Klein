"""Exercise the ten-reference tensor path using an existing generated image."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path)
args = parser.parse_args()
request = root / "jobs/smoke-ten-refs_request.json"
request.write_text(json.dumps({
    "prompt": "A photograph of a red bicycle against a pale plaster wall beside a terracotta pot of purple lavender. Reproduce the scene shown by the reference images with warm late-afternoon sunlight and realistic fine detail.",
    "references": [str(args.image.resolve())] * 10,
    "width": 512, "height": 512, "seed": 31, "enhance": False, "precision": "gguf_q4",
}), encoding="utf-8")
raise SystemExit(subprocess.call([sys.executable, "-u", str(root / "worker.py"), str(request)], cwd=root))
