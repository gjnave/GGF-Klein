"""Pinned, resumable, user-initiated FLUX.3 downloads; no weights in releases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
REPO = "black-forest-labs/flux-3-image-klein-comfy"
REVISION = "5c9b714851dc072b04a23ef6d2d36421f5ccb3a0"
FILES = {
    "image": "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors",
    "text": "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
    "vae": "vae/flux2-vae.safetensors",
    "upsampler": "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors",
}
OPTIONAL = {
    "bf16": (REPO, REVISION, "diffusion_models/flux3_image_klein_sd_bf16.safetensors"),
    "gguf_q4": ("Abiray/flux-3-image-klein-sd-GGUF", "1960618efcfaeec728f22fd7f656a7034ee5f73b", "flux3_image_klein_sd_Q4_K_M.gguf"),
}


class DownloadProgress(tqdm):
    def __init__(self, *args, **kwargs):
        self.last_notice = 0
        kwargs["disable"] = False
        super().__init__(*args, **kwargs)

    def update(self, amount=1):
        result = super().update(amount)
        now = time.monotonic()
        if self.total and now - self.last_notice > 0.5:
            print("__GGFDOWN__" + json.dumps({"file": self.desc, "bytes": self.n,
                                               "total": self.total, "percent": round(self.n / self.total * 100, 1)}), flush=True)
            self.last_notice = now
        return result


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(directory, components=None):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "model-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("components", {}) if manifest_path.is_file() else {}
    sources = {}
    for name in components or FILES:
        repo, revision, filename = OPTIONAL[name] if name in OPTIONAL else (REPO, REVISION, FILES[name])
        if (repo, revision) not in sources:
            info = HfApi().model_info(repo, revision=revision, files_metadata=True)
            sources[repo, revision] = {f.rfilename: f for f in info.siblings}
        record = sources[repo, revision][filename]
        sha = record.lfs.sha256
        manifest[name] = {"file": filename, "bytes": record.size, "sha256": sha, "repository": repo, "revision": revision}
        print(f"Downloading/verifying {name}: {filename}", flush=True)
        path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=directory,
                                  tqdm_class=DownloadProgress))
        if path.stat().st_size != record.size or digest(path) != sha:
            raise RuntimeError(f"Integrity check failed: {path}. File preserved for inspection.")
        print(f"Verified {name}", flush=True)
        (directory / "installed-manifest.json").write_text(json.dumps({
            "repository": REPO, "revision": REVISION, "components": manifest,
        }, indent=2), encoding="utf-8")


def verify_required(directory):
    """Installation is complete only when every default file matches its pin."""
    directory = Path(directory).resolve()
    manifest = json.loads((ROOT / "model-manifest.json").read_text(encoding="utf-8"))["components"]
    for name, filename in FILES.items():
        record = manifest[name]
        path = directory / filename
        if not path.is_file() or path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
            raise RuntimeError(f"Required model missing or incomplete: {path}. Rerun INSTALL.bat to resume.")
        print(f"Required model verified: {name}", flush=True)
    print("All four required models are ready.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=str(ROOT / "models"))
    parser.add_argument("--components", nargs="*", choices=list(FILES) + list(OPTIONAL))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_required(args.directory)
        else:
            download(args.directory, args.components)
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in (401, 403):
            raise SystemExit(f"Model access is required. Accept/request access at https://huggingface.co/{REPO}, "
                             "then run .venv\\Scripts\\hf.exe auth login with that account and rerun INSTALL.bat. "
                             "Setup is incomplete; existing files and partial downloads are preserved.") from None
        raise
