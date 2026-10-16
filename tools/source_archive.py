"""Local installer-test fixture only. Production downloads use host archives."""
from pathlib import Path
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
private = {".venv", ".git", "runtime", "models", "inputs", "outputs", "jobs", "workspaces", "logs", "diagnostics", "__pycache__", ".gradio"}
private_files = {"settings.json", "network_settings.json", "server-links.json", "app.lock", ".installed-revision.json"}
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file() or rel.parts[0] in private or "__pycache__" in rel.parts or path.name in private_files:
            continue
        if path.suffix in (".zip", ".pyc", ".safetensors", ".gguf", ".bin", ".pt", ".pth") or path.name.startswith("smoke-"):
            continue
        archive.write(path, "GGF-Klein-main/" + rel.as_posix())
print("Local source fixture created; never used as a release/update dependency.")
