"""Reproducible optional-precision image tests. Does not claim other GPU results."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
original = json.loads((root / "outputs/ggf-klein_20261015_193500_88c33926.json").read_text())
results = []
for precision in ("bf16", "gguf_q4"):
    request = root / "jobs" / f"smoke-{precision}_request.json"
    request.write_text(json.dumps({
        "prompt": original["caption"], "width": 768, "height": 768, "seed": 7,
        "enhance": False, "precision": precision,
    }), encoding="utf-8")
    with (root / "logs" / f"smoke-{precision}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-u", str(root / "worker.py"), str(request)],
                                cwd=root, stdout=log, stderr=subprocess.STDOUT)
    lines = (root / "logs" / f"smoke-{precision}.log").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line[7:]) for line in lines if line.startswith("__GGF__")]
    outcome = next((e for e in reversed(events) if e["type"] in ("result", "error")), {})
    results.append({"precision": precision, "exit": result.returncode, "outcome": outcome})
    print(json.dumps(results[-1]), flush=True)
(root / "logs" / "precision-tests.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
