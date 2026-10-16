"""Pinned, resumable, user-initiated FLUX.3 downloads; no weights in releases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError
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
        try:
            path = Path(hf_hub_download(repo, filename, revision=revision, local_dir=directory,
                                      tqdm_class=DownloadProgress))
        except GatedRepoError:
            raise SystemExit(
                f"Accept access at https://huggingface.co/{repo}, then sign in to that "
                "account with hf auth login. Partial downloads are preserved."
            ) from None
        if path.stat().st_size != record.size or digest(path) != sha:
            raise RuntimeError(f"Integrity check failed: {path}. File preserved for inspection.")
        print(f"Verified {name}", flush=True)
        (directory / "installed-manifest.json").write_text(json.dumps({
            "repository": REPO, "revision": REVISION, "components": manifest,
        }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=str(ROOT / "models"))
    parser.add_argument("--components", nargs="*", choices=list(FILES) + list(OPTIONAL))
    args = parser.parse_args()
    download(args.directory, args.components)
