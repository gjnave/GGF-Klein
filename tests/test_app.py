import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import app
import network_settings
import runtime
import update_app


class AppTests(unittest.TestCase):
    def test_restore_stable_token(self):
        key = "b" * 48
        runtime.save_workspace(key, prompt="Saved prompt", width=512, height=768)
        restored = app.restore(key)
        self.assertEqual(restored[:4], (key, "Saved prompt", 512, 768))

    def test_boxes(self):
        box = app.boxes_to_model([["Mug", 20, 10, 80, 90]])[0]
        self.assertEqual(box["bbox"], [0.1, 0.2, 0.9, 0.8])
        with self.assertRaises(ValueError):
            app.boxes_to_model([["Bad", 80, 10, 20, 90]])

    def test_archive_paths(self):
        for name in ("../app.py", "/app.py", "C:/app.py", "models/a", "MODELS/a", "settings.json", "workspaces/a", "inputs/a"):
            self.assertFalse(update_app.safe_name(name), name)
        self.assertTrue(update_app.safe_name("vendor/comfy_core/comfy/ldm/models/autoencoder.py"))

    def test_archive_rejects_private_data(self):
        # These test artifacts are retained; this suite does not delete files.
        folder = Path(tempfile.mkdtemp(prefix="ggf-klein-test-"))
        archive = folder / "bad.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("repo/app.py", "")
            stream.writestr("repo/settings.json", "{}")
        with self.assertRaises(ValueError):
            update_app.stage_archive(archive, folder / "stage", ["app.py"])

    def test_blank_password_disables_auth(self):
        original = network_settings.SETTINGS_PATH
        folder = Path(tempfile.mkdtemp(prefix="ggf-klein-network-test-"))
        try:
            network_settings.SETTINGS_PATH = folder / "network.json"
            stored = network_settings.save_settings("public", "", "x")
            self.assertTrue(network_settings.verify_login("ggf", "x", stored))
            cleared = network_settings.save_settings("public", "", "")
            self.assertFalse(cleared["digest"])
            self.assertFalse(cleared["salt"])
        finally:
            network_settings.SETTINGS_PATH = original

    def test_ui_builds(self):
        self.assertGreater(len(app.build_demo().blocks), 80)


if __name__ == "__main__":
    unittest.main()
