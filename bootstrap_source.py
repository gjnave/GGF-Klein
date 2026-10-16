"""Install validated source only, with backups; never replace private user data."""
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

from update_app import stage_archive, safe_name


def install(source, target):
    if (target / "app.lock").is_file():
        import psutil
        try:
            process = psutil.Process(int((target / "app.lock").read_text()))
            if str(target).lower() in " ".join(process.cmdline()).lower():
                raise RuntimeError("Close this app before reinstalling. Existing files were not changed.")
        except (ValueError, psutil.NoSuchProcess):
            pass
    cfg = json.loads((source / "release_sources.json").read_text())
    archive = source.parent / "source.zip"
    stage = source.parent / "validated"
    files = stage_archive(archive, stage, cfg["required"])
    backup = target.parent / "app-backups" / (time.strftime("%Y%m%d-%H%M%S") + "-installer")
    for name in files:
        destination = target / name
        if not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError("A destination path leaves the app folder.")
        if destination.is_file():
            saved = backup / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, saved)
    for name in files:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stage / name, destination)
    print("Validated source installed; private files and models preserved.")


if __name__ == "__main__":
    install(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
