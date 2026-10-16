"""GGF image creation/editing UI. CUDA inference is isolated in worker.py."""
import argparse
import base64
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import time
import zipfile

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import gradio as gr
from PIL import Image, ImageDraw
import network_settings as network
import runtime as jobs
import update_app
from model_manager import FILES, OPTIONAL

ROOT = Path(__file__).resolve().parent
SERVERS = []
DOWNLOAD = {"running": False, "message": ""}
MODES = [("This PC only", "local"), ("Local network / Wi-Fi", "lan"),
         ("Temporary public phone link", "public")]
PRESETS = {
    "Photo": "Natural photographic lighting, realistic materials and coherent perspective.",
    "Illustration": "A polished illustration with a unified palette and clearly readable composition.",
    "Product": "Clean product photography with controlled studio lighting and a simple background.",
}


def safe(function):
    def wrapped(*args):
        try:
            return function(*args)
        except (ValueError, OSError, RuntimeError) as error:
            raise gr.Error(str(error)) from None
    return wrapped


def model_status():
    directory = Path(jobs.config()["models"])
    manifest = jobs.read_json(ROOT / "model-manifest.json").get("components", {})
    lines = []
    for key, filename in FILES.items():
        path = directory / filename
        expected = manifest.get(key, {}).get("bytes")
        state = "Installed" if path.is_file() and (expected is None or path.stat().st_size == expected) else "Missing / incomplete"
        lines.append(f"- {key.title()}: {state}")
    for key, (_, _, filename) in OPTIONAL.items():
        lines.append(f"- Optional {key}: {'Installed' if (directory / filename).is_file() else 'Not downloaded'}")
    return "\n".join(lines) + ("\n\n" + DOWNLOAD["message"] if DOWNLOAD["message"] else "")


def download_models(token, precision="int8"):
    if precision == "auto":
        precision = "int8"
    with jobs.GUARD:
        if DOWNLOAD["running"]:
            return "Model download is already running."
        DOWNLOAD.update(running=True, message="Downloading and verifying the official INT8 model set…")
    directory = jobs.config()["models"]
    components = list(FILES) if precision == "int8" else ["text", "vae", "upsampler", precision]

    def run():
        try:
            env = {**os.environ, "PYTHONUTF8": "1"}
            if token:
                env["HF_TOKEN"] = token
            with (ROOT / "logs/model-download.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, "-u", str(ROOT / "model_manager.py"), "--directory", directory,
                     "--components", *components],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                    text=True, encoding="utf-8", errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("__GGFDOWN__"):
                        progress = json.loads(line[11:])
                        DOWNLOAD["message"] = f"Downloading {progress['file']} · {progress['percent']}%"
                    elif line.startswith(("Downloading/verifying", "Verified")):
                        DOWNLOAD["message"] = line.strip()
                code = process.wait()
            DOWNLOAD["message"] = ("All four components downloaded and SHA-256 verified." if code == 0 else
                "Download stopped. Check access/token and available disk space, then retry. Partial downloads are kept; see logs/model-download.log.")
        finally:
            DOWNLOAD["running"] = False

    threading.Thread(target=run, daemon=True).start()
    return DOWNLOAD["message"]


def restore(value):
    key = jobs.token(value)
    data = jobs.workspace(key)
    refs = [p for p in data.get("references", []) if jobs.owned(p, "inputs")]
    return (key, data.get("prompt", ""), data.get("width", 768), data.get("height", 768),
            data.get("seed", -1), data.get("count", 1), refs, [(p, f"Reference {i + 1}") for i, p in enumerate(refs)],
            data.get("caption", ""), data.get("enhance", True), data.get("boxes", []),
            gr.update(open=bool(refs)), "Custom" if refs or (data.get("width", 768), data.get("height", 768)) != (768, 768) else "Square")


def save_controls(key, prompt, width, height, seed, count, caption, enhance, boxes):
    if not key:
        return
    jobs.save_workspace(key, prompt=prompt, width=width, height=height, seed=seed,
                        count=count, caption=caption, enhance=enhance, boxes=boxes)


def import_refs(key, files, current):
    files = files or []
    if len(current or []) + len(files) > 10:
        raise gr.Error("Use at most ten references. Remove one from the list before adding more.")
    references = list(current or []) + [jobs.retain_image(path) for path in files]
    jobs.save_workspace(key, references=references)
    w, h = reference_dimensions(references)
    return references, [(p, f"Reference {i + 1}") for i, p in enumerate(references)], None, gr.update(open=True), "Custom", w, h


def reference_dimensions(references):
    if not references:
        return 768, 768
    with Image.open(references[0]) as image:
        w, h = image.size
    scale = min(1, (1024 * 1024 / (w * h)) ** 0.5)
    return max(64, round(w * scale / 16) * 16), max(64, round(h * scale / 16) * 16)


def reorder(key, refs, index, direction):
    refs = list(refs or [])
    index = int(index)
    if not 0 <= index < len(refs):
        return refs, [(p, f"Reference {i + 1}") for i, p in enumerate(refs)], index
    if direction == "remove":
        refs.pop(index)  # Remove from this workspace list; the stored image is not deleted.
        index = max(0, index - 1)
    else:
        target = max(0, min(len(refs) - 1, index + direction))
        refs[index], refs[target] = refs[target], refs[index]
        index = target
    jobs.save_workspace(key, references=refs)
    return refs, [(p, f"Reference {i + 1}") for i, p in enumerate(refs)], index


def select_reference(evt: gr.SelectData):
    return int(evt.index)


def boxes_to_model(rows):
    boxes = []
    for index, row in enumerate(rows or []):
        desc, left, top, right, bottom = row
        if not str(desc or "").strip():
            continue
        left, top, right, bottom = map(float, (left, top, right, bottom))
        if not (0 <= left < right <= 100 and 0 <= top < bottom <= 100):
            raise ValueError("Each layout box needs left < right and top < bottom, between 0 and 100 percent.")
        boxes.append({"id": f"obj_{index + 1}", "desc": str(desc),
                      "bbox": [top / 100, left / 100, bottom / 100, right / 100]})
    return boxes


def preview_layout(rows, width, height):
    boxes = boxes_to_model(rows)
    ratio = float(width) / float(height)
    w, h = (640, max(1, round(640 / ratio))) if ratio >= 1 else (max(1, round(640 * ratio)), 640)
    image = Image.new("RGB", (w, h), "#091425")
    draw = ImageDraw.Draw(image)
    for box in boxes:
        top, left, bottom, right = box["bbox"]
        rect = (round(left * w), round(top * h), round(right * w), round(bottom * h))
        draw.rectangle(rect, outline="#f5b942", width=3)
        draw.text((rect[0] + 5, rect[1] + 5), box["id"], fill="#f5b942")
    return image


def select_box(rows, width, height, description, corner, evt: gr.SelectData):
    image = preview_layout(rows, width, height)
    x, y = evt.index
    point = (round(x / image.width * 100, 2), round(y / image.height * 100, 2))
    if corner is None:
        ImageDraw.Draw(image).ellipse((x - 5, y - 5, x + 5, y + 5), fill="#f5b942")
        return rows, point, image, "Now tap the opposite corner."
    left, right = sorted((corner[0], point[0]))
    top, bottom = sorted((corner[1], point[1]))
    if left == right or top == bottom:
        return rows, None, image, "The box needs some width and height. Tap two different corners."
    if not str(description or "").strip():
        return rows, None, image, "Describe this element first, then tap its two corners."
    rows = [row for row in (rows or []) if str(row[0] or "").strip()]
    rows.append([description, left, top, right, bottom])
    return rows, None, preview_layout(rows, width, height), "Box added. Describe another element or generate."


def start(key, prompt, width, height, seed, count, caption, enhance, refs, rows, action):
    if not key:
        raise ValueError("Reload the page once to initialize this browser workspace.")
    if DOWNLOAD["running"]:
        raise ValueError("Let model installation finish before generating.")
    if not str(prompt).strip() and not str(caption).strip():
        raise ValueError("Describe an image or the change you want.")
    width, height, count = int(width), int(height), int(count)
    if min(width, height) < 64 or width % 16 or height % 16:
        raise ValueError("Image dimensions must be multiples of 16, at least 64 pixels.")
    if count < 1:
        raise ValueError("Images per job must be at least one.")
    selected_seed = secrets.randbits(63) if int(seed) < 0 else int(seed)
    if not 0 <= selected_seed < 2**64 - count:
        raise ValueError("Choose a seed within the unsigned 64-bit range.")
    boxes = boxes_to_model(rows)
    if action == "enhance":
        enhance, count = True, 1
    request = {"prompt": prompt if enhance else caption or prompt, "width": width, "height": height,
               "seed": selected_seed, "enhance": enhance, "count": count,
               "references": refs or [], "boxes": boxes, "action": action}
    save_controls(key, prompt, width, height, seed, count, caption, enhance, rows)
    jobs.submit(key, request)
    return "Queued. This browser's previous results remain in History.", [], [], None


def poll(key, last):
    record = jobs.status(key) if key else None
    if record is None:
        return (gr.skip(),) * 8
    elapsed = time.time() - record.get("started", record["created"])
    label = f"**{record['state'].upper()}** · {record['message']}"
    if record["state"] == "running":
        label += f" · {elapsed:.0f}s elapsed"
    fingerprint = f"{record['id']}:{record['updated']}"
    if fingerprint == last:
        return label, gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    paths = [p for p in record.get("outputs", []) if jobs.owned(p)]
    refs = record["request"].get("references", [])
    before = refs[0] if refs else None
    compare = (before, paths[-1]) if before and paths else None
    all_records = jobs.history(key)
    history = [(p, r["request"]["prompt"][:100]) for r in all_records for p in r.get("outputs", []) if jobs.owned(p)]
    queue_rows = [[r["id"][:8], r["state"], r["request"]["prompt"][:100]] for r in all_records[:30]]
    return label, paths, paths, compare, record.get("caption", gr.skip()), history, queue_rows, fingerprint


def cancel_all(key):
    count = 0
    for record in jobs.history(key):
        if record["state"] in ("queued", "running"):
            jobs.cancel(key, record["id"])
            count += 1
    return f"Stopped {count} active/queued job(s) from this browser. Saved images are kept."


def reuse(key, selected, evt: gr.SelectData):
    # Gallery event shapes vary; use the index against our own saved history.
    entries = [p for r in jobs.history(key) for p in r.get("outputs", []) if jobs.owned(p)]
    index = int(evt.index)
    return entries[index] if 0 <= index < len(entries) else selected


def use_result(key, path, refs):
    if not jobs.owned(path):
        raise ValueError("Select an image in History first.")
    new_ref = jobs.retain_image(path)
    jobs.save_workspace(key, references=[new_ref])
    w, h = reference_dimensions([new_ref])
    return [new_ref], [(new_ref, "Reference 1")], gr.update(selected="create"), gr.update(open=True), "Custom", w, h


def diagnostic():
    (ROOT / "diagnostics").mkdir(exist_ok=True)
    name = ROOT / "diagnostics" / f"diagnostics_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}.zip"
    # Explicit allowlist: no tokens, network credentials, images or prompts.
    with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in ("requirements.txt", "model-manifest.json", "VALIDATION.md"):
            path = ROOT / filename
            if path.is_file():
                archive.write(path, filename)
        result = subprocess.run([sys.executable, "-c",
            "import torch,platform;print(platform.platform());print(torch.__version__,torch.version.cuda);print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU');print(torch.cuda.get_arch_list())"],
            capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        archive.writestr("hardware.txt", result.stdout + result.stderr)
    return str(name)


def request_model_removal(name):
    filename = OPTIONAL[name][2] if name in OPTIONAL else FILES[name]
    path = (Path(jobs.config()["models"]) / filename).resolve()
    if not path.is_file():
        raise ValueError("That model is not installed at the selected model location.")
    return str(path), f"Delete only **{path.name}** ({path.stat().st_size / 1e9:.2f} GB) from `{path.parent}`? You would need to download it again.", gr.update(visible=True)


def confirm_model_removal(approved_path):
    with jobs.GUARD:
        if jobs.ACTIVE or jobs.PENDING.qsize() or DOWNLOAD["running"]:
            raise ValueError("Finish or stop jobs/downloads before removing a model.")
        path = Path(approved_path).resolve()
        base = Path(jobs.config()["models"]).resolve()
        allowed = {str((base / filename).resolve()) for filename in FILES.values()}
        allowed.update(str((base / item[2]).resolve()) for item in OPTIONAL.values())
        if str(path) not in allowed:
            raise ValueError("The model location changed. Select the model and confirm again.")
        path.unlink()  # Invoked only by the user's explicit filename-specific confirmation.
    return model_status(), gr.update(visible=False)


def shutdown():
    with jobs.GUARD:
        if jobs.ACTIVE or jobs.PENDING.qsize() or DOWNLOAD["running"]:
            return "Finish or stop jobs/downloads first."
        jobs.STOPPING = True

    def close():
        for server in SERVERS:
            server.close(verbose=False)
        import atexit
        atexit._run_exitfuncs()
        os._exit(0)

    timer = threading.Timer(2, close)
    timer.daemon = True
    timer.start()
    return "Shutting down this app. Restart RUN.bat on your PC to use it again."


def update_restart():
    with jobs.GUARD:
        if jobs.ACTIVE or jobs.PENDING.qsize() or DOWNLOAD["running"]:
            raise ValueError("Finish or stop jobs/downloads before updating.")
        update_app.start_restart()
        return shutdown() + " The updater will install the latest source and reopen it."


def theme():
    return gr.themes.Base(primary_hue="amber", secondary_hue="slate", neutral_hue="slate").set(
        background_fill_primary="#07101f", background_fill_primary_dark="#07101f",
        background_fill_secondary="#111c2f", background_fill_secondary_dark="#111c2f",
        body_background_fill="#07101f", body_background_fill_dark="#07101f",
        body_text_color="#e8eef9", body_text_color_dark="#e8eef9",
        block_background_fill="#111c2f", block_background_fill_dark="#111c2f",
        block_label_text_color="#e8eef9", block_label_text_color_dark="#e8eef9",
        block_label_background_fill="#111c2f", block_label_background_fill_dark="#111c2f",
        block_title_text_color="#e8eef9", block_title_text_color_dark="#e8eef9",
        input_background_fill="#091425", input_background_fill_dark="#091425",
        input_border_color="#435570", input_border_color_dark="#435570",
        input_placeholder_color="#9baec8", input_placeholder_color_dark="#9baec8",
        button_primary_background_fill="#f5b942", button_primary_background_fill_dark="#f5b942",
        button_primary_text_color="#091425", button_primary_text_color_dark="#091425")


def build_demo():
    settings = network.read_settings()
    logo = base64.b64encode((ROOT / "assets/ggf-brain-logo.png").read_bytes()).decode()
    with gr.Blocks(title="GGF Image Studio · Development") as demo:
        gr.HTML(f'<header id="hero"><div><span class="eyebrow">GET GOING FAST · LOCAL AI</span><h1>GGF Image Studio</h1><p>Create an image. Change what you want.</p><nav><a href="https://getgoingfast.pro" target="_blank" rel="noopener">GetGoingFast.pro ↗</a><a href="https://youtube.com/@TheAIHobbyGuy" target="_blank" rel="noopener">The AI Hobby Guy ↗</a></nav></div><img src="data:image/png;base64,{logo}" alt="GGF"></header>')
        activity = gr.Markdown("Ready. Development preview · INT8 inference verified on RTX 4090.", elem_id="activity")
        key, refs, selected_ref, last, selected_result = gr.State(""), gr.State([]), gr.State(0), gr.State(""), gr.State(None)
        browser_token = gr.Textbox(value="", visible=False)
        with gr.Tabs(selected="create") as tabs:
            with gr.Tab("Create & edit", id="create"):
                gr.Markdown("Describe an image to create it. Add references to edit or combine them; the first is your main image.")
                with gr.Row(elem_id="create-row", equal_height=False):
                    with gr.Column(min_width=280):
                        upload = gr.File(label="Reference images · optional · up to 10", file_types=["image"], file_count="multiple", type="filepath", height=120)
                        with gr.Accordion("View / reorder references", open=False) as reference_group:
                            gallery = gr.Gallery(label="References · select to reorder", columns=3, rows=1, height=210, object_fit="contain", interactive=False, buttons=["fullscreen"])
                            with gr.Row():
                                earlier, later, remove = gr.Button("← Earlier"), gr.Button("Later →"), gr.Button("Remove selected")
                        prompt = gr.Textbox(label="What would you like to create or change?", lines=4)
                        with gr.Row():
                            shape = gr.Dropdown(["Square", "Landscape", "Portrait", "Custom"], value="Square", label="Shape")
                            width = gr.Number(value=768, precision=0, minimum=64, step=16, label="Width")
                            height = gr.Number(value=768, precision=0, minimum=64, step=16, label="Height")
                        with gr.Accordion("Prompt assistant & image options", open=False):
                            with gr.Row():
                                preset = gr.Dropdown(list(PRESETS), value="Photo", label="Prompt preset")
                                apply_preset = gr.Button("Add preset")
                            enhance = gr.Checkbox(value=True, label="Expand my request with the local prompt assistant")
                            caption = gr.Textbox(label="Expanded full-scene caption · editable", lines=6)
                            expand = gr.Button("Expand prompt only")
                            gr.Markdown("Unchanged requests reuse their expanded caption for speed. Expand prompt only forces a fresh expansion. To use your edited caption unchanged, turn expansion off.")
                            with gr.Row():
                                seed = gr.Number(value=-1, precision=0, label="Seed · −1 random")
                                count = gr.Number(value=1, minimum=1, precision=0, label="Images per job · sequential")
                            gr.Markdown("16 steps · distilled INT8. Large images, many references and batches take more time and memory.")
                        with gr.Accordion("Composition / layout · optional", open=False):
                            gr.Markdown("Describe each element and its box as percentages. Boxes guide placement; they are not hard masks. Use the prompt assistant.")
                            boxes = gr.Dataframe(headers=["Element", "Left %", "Top %", "Right %", "Bottom %"],
                                datatype=["str", "number", "number", "number", "number"], value=[], type="array",
                                row_count=1, column_count=5, label="Layout boxes")
                            draw_layout = gr.Button("Preview box positions")
                            element_description = gr.Textbox(label="New element · describe it, then tap two corners in the guide")
                            corner = gr.State(None)
                            layout_status = gr.Markdown("Preview the guide, then tap two opposite corners to add a box. You can also edit the table.")
                            layout = gr.Image(label="Layout guide", interactive=False, height=320)
                        generate = gr.Button("Generate image / add to queue", variant="primary")
                        stop = gr.Button("Stop my running & queued jobs", variant="stop")
                    with gr.Column(min_width=280):
                        output = gr.Gallery(label="Your images · expand for fullscreen", height=540, columns=1,
                                            object_fit="contain", interactive=False, buttons=["download", "fullscreen"])
                        downloads = gr.File(label="Download original PNGs · unique filenames", file_count="multiple", interactive=False)
                        with gr.Accordion("Before & after", open=False):
                            compare = gr.ImageSlider(label="Reference / generated image", interactive=False)
            with gr.Tab("History & queue", id="history"):
                history = gr.Gallery(label="Saved images · select one to edit", columns=3, height=550, object_fit="contain",
                                     interactive=False, buttons=["download", "fullscreen"])
                use = gr.Button("Use selected image as my new reference", variant="primary")
                queue_view = gr.Dataframe(headers=["Job", "State", "Request"], interactive=False, label="This browser's jobs")
                gr.Markdown("Saved images stay in outputs. Closing the browser does not stop generation. Reopen the same link/browser to restore your workspace.")
            with gr.Tab("Settings", id="settings"):
                gr.Markdown("### Models & performance")
                model_text = gr.Markdown(model_status())
                gr.Timer(1).tick(model_status, outputs=model_text, queue=False, show_progress="hidden")
                gr.Button("Refresh model status").click(model_status, outputs=model_text, queue=False)
                gr.Markdown("[Accept official model access on Hugging Face](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy), then provide a read token if needed. Downloads go directly to your PC; no models are in the installer.")
                hf_token = gr.Textbox(label="Hugging Face read token · optional if already signed in · not saved", type="password")
                install = gr.Button("Download / verify official INT8 model set")
                install.click(lambda token: (download_models(token), ""), inputs=hf_token, outputs=[model_text, hf_token], queue=False)
                model_dir = gr.Textbox(value=jobs.config()["models"], label="Model folder")
                precision = gr.Dropdown([("Automatic · best installed tested format", "auto"), ("Balanced · official INT8", "int8"), ("Memory-saving image model · GGUF Q4", "gguf_q4"),
                                         ("Higher-precision image model · BF16", "bf16")],
                                         value=jobs.config()["precision"], label="Image model · tested on RTX 4090")
                low_vram = gr.Checkbox(value=jobs.config()["low_vram"], label="Aggressive component offloading · slower")
                reserve = gr.Number(value=jobs.config()["reserve_vram"], minimum=0, label="Reserve GPU memory for Windows / other apps · GB")
                save_models = gr.Button("Save model location / memory settings")
                save_models.click(safe(jobs.save_config), [model_dir, reserve, precision, low_vram], model_text, queue=False)
                gr.Button("Download selected optional image model").click(lambda t, p: (download_models(t, p), ""), [hf_token, precision], [model_text, hf_token], queue=False)
                gr.Markdown("**Validated here:** INT8, BF16 image model and GGUF Q4 image model, all with INT8 text encoder. The assistant loads separately and may use more memory than the image model. Smaller physical GPUs, FP8 and NVFP4 are not yet validated. Models unload after each image.")
                with gr.Accordion("Remove an installed model", open=False):
                    remove_choice = gr.Dropdown(list(FILES) + list(OPTIONAL), value="image", label="Model file")
                    remove_button = gr.Button("Review removal")
                    approved_path = gr.State("")
                    with gr.Column(visible=False) as remove_confirmation:
                        remove_text = gr.Markdown("")
                        confirm_remove = gr.Button("Yes, delete this named model file", variant="stop")
                        keep_model = gr.Button("Keep model")
                    remove_button.click(safe(request_model_removal), remove_choice, [approved_path, remove_text, remove_confirmation], queue=False)
                    confirm_remove.click(safe(confirm_model_removal), approved_path, [model_text, remove_confirmation], queue=False)
                    keep_model.click(lambda: gr.update(visible=False), outputs=remove_confirmation, queue=False)
                gr.Markdown("### Use it from your phone")
                mode = gr.Dropdown(MODES, value=settings["mode"], label="Access mode")
                username = gr.Textbox(value=settings["username"], label="Username · optional")
                password = gr.Textbox(type="password", label="Password · blank disables login")
                net_status = gr.Markdown("Your PC runs the models. Local access is always login-free. Network changes apply after restarting this app.")
                gr.Button("Save phone settings").click(safe(lambda m, u, p: (network.save_settings(m, u, p), "Saved. Restart RUN.bat on the PC to apply.")[1]), [mode, username, password], net_status, queue=False)
                gr.Markdown("### App & GPU memory")
                controls = gr.Markdown("")
                gr.Button("Release models / free GPU memory").click(safe(jobs.release), outputs=controls, queue=False)
                stop_server = gr.Button("Stop server")
                with gr.Column(visible=False) as confirmation:
                    gr.Markdown("This shuts down the app and phone link. You must restart it on the local PC. Choose **Release models** to keep the app open instead.")
                    yes, no = gr.Button("Yes, shut down app", variant="stop"), gr.Button("Keep app open")
                stop_server.click(lambda: gr.update(visible=True), outputs=confirmation, queue=False)
                no.click(lambda: gr.update(visible=False), outputs=confirmation, queue=False)
                yes.click(shutdown, outputs=controls, queue=False)
                gr.Markdown("### Diagnostics & updates")
                diagnostic_file = gr.File(label="Diagnostic export · excludes prompts, images & credentials")
                gr.Button("Export environment diagnostics").click(diagnostic, outputs=diagnostic_file)
                with gr.Row():
                    check_updates = gr.Button("Check for updates")
                    apply_update = gr.Button("Update & restart")
                update_status = gr.Markdown("Development build 2026.10.15 · release publishing pending.")
                check_updates.click(update_app.check_update, outputs=update_status, queue=False)
                apply_update.click(safe(update_restart), outputs=update_status, queue=False)
        gr.Markdown("Powered by FLUX.3 Image Klein · Review AI-generated outputs before sharing. [Model terms](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md)", elem_id="powered")
        controls_list = [prompt, width, height, seed, count, caption, enhance, boxes]
        # State inputs are server-owned: Gradio ignores JS replacements for them.
        # A hidden textbox transports the localStorage token to this one restore handler.
        demo.load(restore, inputs=browser_token, outputs=[key, prompt, width, height, seed, count, refs, gallery, caption, enhance, boxes, reference_group, shape],
                  js="(key) => {let k=localStorage.getItem('ggf-klein-workspace');if(!k){let b=new Uint8Array(24);crypto.getRandomValues(b);k=Array.from(b,x=>x.toString(16).padStart(2,'0')).join('');localStorage.setItem('ggf-klein-workspace',k);}return [k];}")
        for control in controls_list:
            control.change(save_controls, [key, *controls_list], queue=False, api_name=False)
        upload.upload(import_refs, [key, upload, refs], [refs, gallery, upload, reference_group, shape, width, height], queue=False)
        gallery.select(select_reference, outputs=selected_ref, queue=False)
        earlier.click(lambda k, r, i: reorder(k, r, i, -1), [key, refs, selected_ref], [refs, gallery, selected_ref], queue=False)
        later.click(lambda k, r, i: reorder(k, r, i, 1), [key, refs, selected_ref], [refs, gallery, selected_ref], queue=False)
        remove.click(lambda k, r, i: reorder(k, r, i, "remove"), [key, refs, selected_ref], [refs, gallery, selected_ref], queue=False)
        shape.input(lambda v, w, h: {"Square": (768, 768), "Landscape": (1024, 768), "Portrait": (768, 1024)}.get(v, (w, h)), [shape, width, height], [width, height], queue=False)
        apply_preset.click(lambda p, name: (p.rstrip() + "\n" + PRESETS[name]).strip(), [prompt, preset], prompt, queue=False)
        draw_layout.click(safe(preview_layout), [boxes, width, height], layout, queue=False)
        layout.select(select_box, [boxes, width, height, element_description, corner], [boxes, corner, layout, layout_status], queue=False)
        inputs = [key, *controls_list[:5], caption, enhance, refs, boxes]
        generate.click(safe(lambda *v: start(*v, "generate")), inputs, [activity, output, downloads, compare], queue=False)
        expand.click(safe(lambda *v: start(*v, "enhance")), inputs, [activity, output, downloads, compare], queue=False)
        stop.click(safe(cancel_all), key, activity, queue=False)
        history.select(reuse, [key, selected_result], selected_result, queue=False)
        use.click(safe(use_result), [key, selected_result, refs], [refs, gallery, tabs, reference_group, shape, width, height], queue=False)
        gr.Timer(1).tick(poll, [key, last], [activity, output, downloads, compare, caption, history, queue_view, last], queue=False, show_progress="hidden")
    return demo.queue(default_concurrency_limit=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    import psutil
    lock = ROOT / "app.lock"
    if lock.is_file():
        try:
            existing = psutil.Process(int(lock.read_text()))
            if str(ROOT).lower() in " ".join(existing.cmdline()).lower():
                print("This app is already running. Open:", (ROOT / "server-links.json").read_text())
                raise SystemExit(0)
        except (ValueError, OSError, psutil.NoSuchProcess):
            pass
    lock.write_text(str(os.getpid()))
    jobs.start_queue()
    network.install_upload_disconnect_handling()
    settings = network.read_settings()
    mode = "public" if args.public else settings["mode"]
    auth = (lambda u, p: network.verify_login(u, p, settings)) if settings["digest"] else None
    local, remote, local_url, remote_url = network.launch_access_servers(
        build_demo, mode=mode, preferred_port=7872, auth=auth, inbrowser=not args.no_browser,
        theme=theme(), css=(ROOT / "ui.css").read_text(encoding="utf-8"),
        allowed_paths=[str(ROOT / name) for name in ("outputs", "inputs", "diagnostics")],
    )
    SERVERS.extend(s for s in (local, remote) if s is not None)
    (ROOT / "server-links.json").write_text(json.dumps({"local": local_url, "remote": remote_url}), encoding="utf-8")
    try:
        while not jobs.STOPPING:
            time.sleep(1)
        time.sleep(5)
    except KeyboardInterrupt:
        jobs.stop_owned_process()
        for server in SERVERS:
            server.close()
