import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from PIL import Image

import app
import network_settings
import runtime
import update_app


class AppTests(unittest.TestCase):
    def test_import_does_not_start_or_recover_queue(self):
        self.assertFalse(runtime.QUEUE_STARTED)

    def test_portrait_reference_dimensions(self):
        folder = Path(tempfile.mkdtemp(prefix="ggf-klein-portrait-test-"))
        path = folder / "portrait.png"
        Image.new("RGBA", (432, 768), (255, 0, 0, 128)).save(path)
        self.assertEqual(app.reference_dimensions([str(path)]), (432, 768))

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

    def test_submission_feedback_visible_beside_button(self):
        with patch.object(app, "start", return_value=("Queued", [], [], None)):
            self.assertEqual(app.submit_from_ui(action="generate"), ("Queued", "Queued", [], [], None))
        with patch.object(app, "start", side_effect=ValueError("Describe an image")):
            result = app.submit_from_ui(action="generate")
        self.assertEqual(result[0], result[1])
        self.assertIn("Could not start", result[1])
        self.assertIn("Describe an image", result[1])

    def test_progress_updates_both_status_locations(self):
        with patch.object(app, "poll", return_value=("RUNNING", 1, 2, 3, 4, 5, 6, 7)):
            self.assertEqual(app.poll_from_ui("browser", "")[:9], ("RUNNING", "RUNNING", 1, 2, 3, 4, 5, 6, 7))

    def test_busy_progress_and_terminal_button_recovery(self):
        for state in ("queued", "running", "done", "error", "cancelled"):
            record = {"state": state, "created": 0, "message": "Image 1/1 · Creating image · step 4/16"}
            with patch.object(app.jobs, "status", return_value=record):
                generate, expand, progress = app.generation_feedback("browser")
            busy = state in ("queued", "running")
            self.assertEqual(generate["interactive"], not busy)
            self.assertEqual(expand["interactive"], not busy)
            self.assertEqual(bool(progress), busy)
            if busy:
                self.assertIn('value="4" max="16"', progress)

    def test_loading_progress_indeterminate_and_escaped(self):
        record = {"state": "running", "created": 0, "message": "Loading <model>"}
        with patch.object(app.jobs, "status", return_value=record):
            progress = app.generation_feedback("browser")[2]
        self.assertIn("&lt;model&gt;", progress)
        self.assertNotIn('value="', progress)

    def test_duplicate_submit_rejected_before_queueing(self):
        with patch.object(runtime, "status", return_value={"state": "running"}):
            with self.assertRaisesRegex(ValueError, "already queued or running"):
                runtime.submit("browser", {})


if __name__ == "__main__":
    unittest.main()
