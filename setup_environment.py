"""Dependency pins live in source, so future upgrades do not require new BATs."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
uv = root / "runtime/uv.exe"
if not uv.is_file():
    raise SystemExit("App-managed UV is missing. Rerun INSTALL.bat; saved data stays in place.")
pins = json.loads((root / "cuda-requirements.json").read_text())
command = [str(uv), "pip", "install", "--python", sys.executable]
subprocess.run(command + [f"torch=={pins['torch']}", f"torchvision=={pins['torchvision']}",
                         "--index-url", pins["index"]], check=True)
subprocess.run(command + ["-r", str(root / "requirements.txt")], check=True)
subprocess.run([str(uv), "pip", "check", "--python", sys.executable], check=True)
