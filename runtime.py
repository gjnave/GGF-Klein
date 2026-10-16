"""One persistent queue, one disposable CUDA worker; browser-owned workspaces."""
import atexit
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
for name in ("jobs", "workspaces", "inputs", "outputs", "logs"):
    (ROOT / name).mkdir(exist_ok=True)
GUARD = threading.RLock()
PENDING = queue.Queue()
PROCESS = None
ACTIVE = None
STOPPING = False
QUEUE_STARTED = False


def token(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-f0-9]{48}", value) else secrets.token_hex(24)


def read_json(path, default=None):
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def owned(path, parent="outputs"):
    if not path:
        return None
    candidate = Path(path).resolve()
    return str(candidate) if candidate.is_file() and candidate.is_relative_to((ROOT / parent).resolve()) else None


def workspace(key):
    with GUARD:
        return read_json(ROOT / "workspaces" / f"{token(key)}.json")


def save_workspace(key, **updates):
    with GUARD:
        path = ROOT / "workspaces" / f"{token(key)}.json"
        data = read_json(path)
        data.update(updates)
        write_json(path, data)
        return data


def retain_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        destination = ROOT / "inputs" / f"{secrets.token_hex(12)}.png"
        image.save(destination)
    return str(destination)


def config():
    return {"models": str(ROOT / "models"), "reserve_vram": 1.5, "precision": "auto", "low_vram": False,
            **read_json(ROOT / "settings.json")}


def save_config(models, reserve, precision="auto", low_vram=False):
    directory = Path(models).expanduser().resolve()
    if not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=True)
    with GUARD:
        if ACTIVE:
            raise ValueError("Wait until the current job finishes before changing model storage.")
        if precision not in ("auto", "int8", "bf16", "gguf_q4") or not 0 <= float(reserve):
            raise ValueError("Choose a supported image model and a nonnegative memory reserve.")
        write_json(ROOT / "settings.json", {"models": str(directory), "reserve_vram": float(reserve),
                                           "precision": precision, "low_vram": bool(low_vram)})
    return "Saved. Existing models were not moved or deleted. New jobs use this location."


def job_path(job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{24}", job_id):
        raise ValueError("Invalid job ID.")
    return ROOT / "jobs" / f"{job_id}.json"


def submit(key, request):
    with GUARD:
        if STOPPING:
            raise ValueError("The app is shutting down. Restart it on the PC.")
        job_id = secrets.token_hex(12)
        now = time.time()
        record = {"id": job_id, "owner": key, "state": "queued", "message": "Queued",
                  "created": now, "updated": now, "request": {**config(), **request}, "outputs": []}
        write_json(job_path(job_id), record)
        save_workspace(key, current_job=job_id)
        PENDING.put(job_id)
        return job_id


def update(job_id, **updates):
    with GUARD:
        path = job_path(job_id)
        record = read_json(path)
        record.update(updates, updated=time.time())
        write_json(path, record)
        return record


def cancel(key, job_id):
    global PROCESS
    with GUARD:
        record = read_json(job_path(job_id))
        if record.get("owner") != key:
            raise ValueError("This job belongs to another browser workspace.")
        if record.get("state") in ("done", "error", "cancelled"):
            return "That job has already finished."
        update(job_id, state="cancelled", message="Stopped. Saved images are kept.")
        if ACTIVE == job_id and PROCESS is not None and PROCESS.poll() is None:
            PROCESS.terminate()
    return "Stopped. The app and saved images are kept."


def execute(job_id):
    global PROCESS
    record = read_json(job_path(job_id))
    request = record["request"]
    signature = None
    if request.get("enhance", True):
        from model_manager import FILES
        paths = [Path(p) for p in request.get("references", [])]
        paths.append(Path(request["models"]) / FILES["upsampler"])
        identity = {"prompt": request["prompt"], "boxes": request.get("boxes"),
                    "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) if p.is_file() else (str(p), None, None) for p in paths]}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        cached = workspace(record["owner"]).get("prompt_cache", {})
        if request.get("action") != "enhance" and cached.get("signature") == signature and cached.get("caption"):
            request.update(user_prompt=request["prompt"], prompt=cached["caption"],
                           enhance=False, boxes=None, prompt_assistant_reused=True)
            update(job_id, message="Reusing your unchanged expanded caption", caption=cached["caption"])
    count = int(request.get("count", 1))
    outputs = []
    for index in range(count):
        with GUARD:
            if read_json(job_path(job_id))["state"] == "cancelled":
                return
            item = {**request, "seed": int(request["seed"]) + index}
            request_file = ROOT / "jobs" / f"{job_id}_request.json"
            write_json(request_file, item)
            PROCESS = subprocess.Popen(
                [sys.executable, "-u", str(ROOT / "worker.py"), str(request_file)],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                env={**os.environ, "PYTHONUTF8": "1"},
            )
        result = None
        error = None
        with (ROOT / "logs" / f"{job_id}_{index}.log").open("w", encoding="utf-8") as log:
            for line in PROCESS.stdout:
                log.write(line)
                log.flush()
                if not line.startswith("__GGF__"):
                    continue
                event = json.loads(line[7:])
                if event["type"] == "progress":
                    update(job_id, message=f"Image {index + 1}/{count} · {event['message']}")
                elif event["type"] == "caption":
                    update(job_id, caption=event["caption"])
                    if signature:
                        save_workspace(record["owner"], prompt_cache={"signature": signature, "caption": event["caption"]})
                elif event["type"] == "error":
                    error = event["message"]
                elif event["type"] == "result":
                    result = event
        code = PROCESS.wait()
        with GUARD:
            PROCESS = None
            if read_json(job_path(job_id))["state"] == "cancelled":
                return
        if code or result is None:
            raise RuntimeError(error or f"Inference stopped (exit {code}). See the job log in Settings.")
        if result.get("path"):
            outputs.append(result["path"])
        if result.get("caption"):
            # One expansion per batch, not another expensive assistant run for every seed.
            request.update(user_prompt=request.get("user_prompt", request["prompt"]),
                           prompt=result["caption"], enhance=False, boxes=None)
        update(job_id, outputs=outputs, caption=result.get("caption", ""), metrics=result.get("metrics", {}))
    update(job_id, state="done", message="Complete. GPU models released.")


def consume():
    global ACTIVE
    while True:
        job_id = PENDING.get()
        try:
            with GUARD:
                if read_json(job_path(job_id))["state"] == "cancelled":
                    continue
                ACTIVE = job_id
                update(job_id, state="running", message="Starting image job", started=time.time())
            execute(job_id)
        except Exception as error:
            update(job_id, state="error", message=str(error))
        finally:
            with GUARD:
                ACTIVE = None
            PENDING.task_done()


def status(key):
    data = workspace(key)
    current = data.get("current_job")
    if not current:
        return None
    record = read_json(job_path(current))
    return record if record.get("owner") == key else None


def history(key):
    records = []
    for path in (ROOT / "jobs").glob("*.json"):
        if path.stem.endswith("_request"):
            continue
        record = read_json(path)
        if record.get("owner") == key:
            records.append(record)
    return sorted(records, key=lambda r: r["created"], reverse=True)


def release():
    with GUARD:
        if ACTIVE or PENDING.qsize():
            raise ValueError("Finish or cancel queued jobs first.")
    return "No models are held while idle. GPU memory is released after each image; the phone link stays open."


def stop_owned_process():
    if PROCESS is not None and PROCESS.poll() is None:
        PROCESS.terminate()
        PROCESS.wait(timeout=15)


atexit.register(stop_owned_process)


def start_queue():
    """Only the owning server calls this, after its single-instance check."""
    global QUEUE_STARTED
    with GUARD:
        if QUEUE_STARTED:
            return
        # A real restart must not pretend an interrupted job is still running.
        for path in (ROOT / "jobs").glob("*.json"):
            if path.stem.endswith("_request"):
                continue
            previous = read_json(path)
            if previous.get("state") in ("queued", "running"):
                previous.update(state="error", message="Interrupted by an app restart. Submit again to retry.")
                write_json(path, previous)
        QUEUE_STARTED = True
        threading.Thread(target=consume, daemon=True).start()
