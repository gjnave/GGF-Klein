"""App-owned inference process. No ComfyUI server or workflow execution engine."""
import gc
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parent
request_path = Path(sys.argv[1])
request = json.loads(request_path.read_text(encoding="utf-8"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path[:0] = [str(ROOT / "vendor" / "comfy_core"), str(ROOT / "vendor" / "bfl")]
sys.argv = ["ggf-worker", "--use-pytorch-cross-attention", "--disable-comfy-compiler",
            "--reserve-vram", str(request.get("reserve_vram", 1.5))]
if request.get("low_vram"):
    sys.argv.append("--lowvram")
import comfy.options
comfy.options.enable_args_parsing()
import numpy as np
import psutil
import torch
from PIL import Image, ImageOps, PngImagePlugin
import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
from flux3 import bbox
from flux3.pup import policy_reference, I2I_SCAFFOLD
from model_manager import FILES, OPTIONAL

started = time.perf_counter()
stages = {}
peak_ram = 0
finished = threading.Event()


def event(kind, **data):
    print("__GGF__" + json.dumps({"type": kind, **data}), flush=True)


def stage(message):
    stages[message] = round(time.perf_counter() - started, 3)
    event("progress", message=message, elapsed=stages[message])


def measure_ram():
    global peak_ram
    process = psutil.Process()
    while not finished.wait(0.25):
        peak_ram = max(peak_ram, process.memory_info().rss)


def unload():
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()


def tensor(image):
    return torch.from_numpy(np.asarray(image).copy()).float().div_(255).unsqueeze(0)


def run():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. Install a current NVIDIA driver, then restart the app.")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    threading.Thread(target=measure_ram, daemon=True).start()
    models = Path(request.get("models", str(ROOT / "models"))).resolve()
    paths = {key: models / value for key, value in FILES.items()}
    precision = request.get("precision", "int8")
    if precision == "auto":
        bf16_file = models / OPTIONAL["bf16"][2]
        precision = "bf16" if torch.cuda.get_device_properties(0).total_memory >= 20 * 2**30 and bf16_file.is_file() else "int8"
    if precision in ("bf16", "gguf_q4"):
        paths["image"] = models / OPTIONAL[precision][2]
    elif precision != "int8":
        raise ValueError("This worker currently supports official INT8 or BF16, not that model format.")
    required = (("upsampler",) if request.get("action") == "enhance" else
                ("image", "text", "vae") + (("upsampler",) if request.get("enhance", True) else ()))
    for key in required:
        if not paths[key].is_file():
            raise FileNotFoundError(f"Missing {key}: {paths[key]}. Open Settings / Models to download.")
    images = []
    for path in request.get("references", []):
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGBA")
            base = Image.new("RGBA", image.size, "white")
            images.append(Image.alpha_composite(base, image).convert("RGB"))
    if len(images) > 10:
        raise ValueError("The model supports at most 10 reference images.")
    width, height = int(request.get("width", 768)), int(request.get("height", 768))
    if min(width, height) < 64 or width % 16 or height % 16:
        raise ValueError("Width and height must be positive multiples of 16, at least 64 pixels.")
    seed = int(request.get("seed", 1))
    prompt = request["prompt"].strip()
    if not prompt:
        raise ValueError("Describe the image or the change you want.")
    with torch.no_grad():
        text_options = {"load_device": mm.get_torch_device(), "offload_device": torch.device("cpu")}
        if request.get("enhance", True):
            stage("Loading prompt assistant")
            clip = comfy.sd.load_clip([str(paths["upsampler"])], model_options=text_options)
            system_key, user_text, locked = bbox.plan(prompt, request.get("boxes"), len(images))
            if images:
                user_text = I2I_SCAFFOLD[locked is not None].format(instruction=user_text)
            prompts = json.loads((ROOT / "vendor/bfl/flux3/pup_prompts.json").read_text(encoding="utf-8"))
            references = [tensor(policy_reference(image, len(images))) for image in images]
            tokens = clip.tokenize(user_text, images=references, thinking=False,
                                   system_prompt=prompts[system_key], min_length=1)
            stage("Expanding your scene description")
            generated = clip.generate(tokens, do_sample=False, max_length=3584, seed=seed,
                                      presence_penalty=0.0 if images else 1.5, mtp=False)
            caption = clip.decode(generated).strip()
            if "</think>" in caption:
                caption = caption.split("</think>", 1)[1].strip()
            if not caption:
                raise RuntimeError("Prompt assistant returned an empty caption. Try again or supply a full scene caption.")
            if locked:
                caption = bbox.reconcile(caption, locked)
            event("caption", caption=caption)
            del clip, tokens, generated, references
            unload()
        else:
            caption = prompt
            if request.get("boxes"):
                raise ValueError("Layout boxes require the prompt assistant in this build.")
        if request.get("action") == "enhance":
            event("result", caption=caption, elapsed=round(time.perf_counter() - started, 2))
            return
        stage("Encoding scene description")
        clip = comfy.sd.load_clip([str(paths["text"])], clip_type=comfy.sd.CLIPType.FLUX3, model_options=text_options)
        encoded = clip.encode_from_tokens(clip.tokenize(caption), return_dict=True)
        cond = encoded.pop("cond").cpu()
        positive = [[cond, encoded]]
        del clip
        unload()
        stage("Loading image decoder")
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(paths["vae"]), safe_load=True))
        if images:
            stage("Encoding reference images")
            refs = [vae.encode(tensor(policy_reference(image, len(images)))).cpu() for image in images]
            positive[0][1]["reference_latents"] = refs
            unload()
        stage(f"Loading {precision.upper()} image model")
        if precision == "gguf_q4":
            from gguf_backend import load
            model = load(paths["image"])
        else:
            model = comfy.sd.load_diffusion_model(str(paths["image"]))
        latent = torch.zeros((1, 128, height // 16, width // 16), dtype=torch.float32)
        noise = comfy.sample.prepare_noise(latent, seed)
        # Exact official flow-map schedule: 16 Euler steps, alpha 6.93, no CFG.
        times = torch.linspace(1, 0, 17)
        sigmas = 6.93 * times / (1 + (6.93 - 1) * times)

        def progress(step, x0, x, total):
            event("progress", message=f"Creating image · step {step + 1}/{total}",
                  step=step + 1, total=total, elapsed=round(time.perf_counter() - started, 2))

        stage("Creating image")
        samples = comfy.sample.sample_custom(
            model, noise, 1.0, comfy.samplers.sampler_object("euler"), sigmas,
            positive, positive, latent, callback=progress, disable_pbar=True, seed=seed,
        )
        del model, positive, cond, encoded
        unload()
        stage("Decoding image")
        pixels = vae.decode(samples).cpu().numpy()[0]
        output = Image.fromarray((pixels.clip(0, 1) * 255).round().astype(np.uint8))
        directory = ROOT / "outputs"
        directory.mkdir(exist_ok=True)
        filename = directory / f"ggf-klein_{time.strftime('%Y%m%d_%H%M%S')}_{os.urandom(4).hex()}.png"
        record = {
            "prompt": request.get("user_prompt", prompt), "caption": caption, "seed": seed, "width": width, "height": height,
            "steps": 16, "guidance": 1.0, "sampler": "euler", "schedule_alpha": 6.93,
            "model": paths["image"].name, "reference_count": len(images),
            "seconds": round(time.perf_counter() - started, 2),
            "peak_gpu_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_gpu_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "peak_process_ram_gib": round(peak_ram / 2**30, 3),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "stages": stages,
            "ai_generated": True,
            "prompt_assistant_reused": bool(request.get("prompt_assistant_reused")),
        }
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("GGF generation", json.dumps(record))
        output.save(filename, pnginfo=metadata)
        filename.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        event("result", path=str(filename), caption=caption, metrics=record)


if __name__ == "__main__":
    try:
        run()
    except torch.OutOfMemoryError:
        event("error", message="GPU memory ran out. Try a smaller image or fewer references and close other GPU apps. Saved images are unchanged.")
        sys.exit(2)
    except Exception as error:
        traceback.print_exc()
        event("error", message=f"{type(error).__name__}: {error}")
        sys.exit(1)
    finally:
        finished.set()
