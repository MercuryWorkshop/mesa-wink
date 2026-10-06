# SPDX-License-Identifier: MIT
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import suite


class GateTests(unittest.TestCase):
    def test_process_failure_with_clean_log_never_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = suite.execute(["/bin/sh", "-c", "exit 7"], Path(root) / "log", 5)
            self.assertFalse(result["pass"])
            self.assertEqual(result["rc"], 7)

    def test_validation_error_with_zero_exit_never_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = suite.execute(["/bin/echo", "WebGPU error: invalid binding"], Path(root) / "log", 5)
            self.assertFalse(result["pass"])

    def test_xkeyboard_notice_does_not_hide_real_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            result = suite.execute(["/bin/echo", "Errors from xkbcomp are not fatal to the X server"], Path(root) / "log", 5)
            self.assertTrue(result["pass"])
            result = suite.execute(["/bin/echo", "FATAL: device lost"], Path(root) / "log", 5)
            self.assertFalse(result["pass"])

    def test_ignored_api_call_never_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = suite.execute(["/bin/echo", "warning: unsupported eglCreatePlatformWindowSurface call"], Path(root) / "log", 5)
            self.assertFalse(result["pass"])

    def test_minecraft_thread_error_never_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = suite.execute(["/bin/echo", "[Render thread/ERROR]: Shader compilation failed"], Path(root) / "log", 5)
            self.assertFalse(result["pass"])

    def test_ppsspp_subsystem_error_never_passes(self):
        with tempfile.TemporaryDirectory() as root:
            result = suite.execute(["/bin/echo", "Core.cpp E[G3D]: pipeline creation failed"], Path(root) / "log", 5)
            self.assertFalse(result["pass"])

    def test_timeout_never_passes(self):
        with tempfile.TemporaryDirectory() as root:
            import subprocess
            with patch("suite.subprocess.run", side_effect=subprocess.TimeoutExpired("replay", 1)):
                result = suite.execute(["replay"], Path(root) / "log", 1)
            self.assertEqual(result["status"], "timeout")

    def test_timeout_preserves_errors_without_unbounded_report(self):
        with tempfile.TemporaryDirectory() as root:
            import subprocess
            def failed(*args, **kwargs):
                kwargs["stdout"].write("GL_INVALID_FRAMEBUFFER_OPERATION\n" * 120)
                raise subprocess.TimeoutExpired("replay", 1)
            with patch("suite.subprocess.run", side_effect=failed):
                result = suite.execute(["replay"], Path(root) / "log", 1)
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["error_count"], 120)
            self.assertEqual(len(result["errors"]), 100)

    def test_missing_reference_fails(self):
        with tempfile.TemporaryDirectory() as root:
            result = suite.replay(None, {"id": "missing"}, Path(root), "st")
            self.assertFalse(result["pass"])
            self.assertEqual(result["status"], "missing-reference")

    def test_report_missing_cases_is_failure_in_json_and_junit(self):
        with tempfile.TemporaryDirectory() as root:
            a = SimpleNamespace(out=Path(root), action="run", tier="full")
            result = suite.report(a, [{"id": "iris-pack", "status": "missing-assets", "pass": False, "target": None}])
            self.assertEqual(result, 1)
            self.assertFalse(json.loads((a.out / "results.json").read_text())["pass"])
            self.assertIn('failures="1"', (a.out / "junit.xml").read_text())

    def test_altered_trace_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            case = {"id": "proof", "api": "gles"}
            (root / "capture.trace").write_bytes(b"changed")
            (root / "capture.lock.json").write_text(json.dumps({"case_sha256": suite.case_hash(case), "trace_sha256": "old"}))
            self.assertEqual(suite.replay(None, case, root, "st")["status"], "reference-incompatible")

    def test_reuse_checks_reference_pixels_and_preserves_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            root = out / "proof"
            (root / "reference-0").mkdir(parents=True)
            (out / "reference.lock.json").write_text("{}")
            case = {"id": "proof", "api": "gles", "min_frames": 1}
            trace = root / "capture.trace"
            trace.write_bytes(b"recorded trace")
            image = root / "reference-0/frame-1.png"
            Image.new("RGB", (2, 2), "red").save(image)
            lock = root / "capture.lock.json"
            lock.write_text(json.dumps({"case_sha256": suite.case_hash(case),
                "trace_sha256": suite.sha256(trace), "checkpoints": {image.name: suite.sha256(image)}}))
            before = lock.read_bytes()
            a = SimpleNamespace(out=out, reuse_references=True)
            with patch("suite.shutil.disk_usage", return_value=SimpleNamespace(free=8 * 1024**3)):
                self.assertEqual(suite.record(a, case, root)["status"], "reference-reused")
                Image.new("RGB", (2, 2), "green").save(image)
                self.assertFalse(suite.record(a, case, root)["pass"])
            self.assertEqual(lock.read_bytes(), before)

    def test_changed_application_tool_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            tools = out / "tools"
            tools.mkdir()
            binary = tools / "gltf_viewer"
            binary.write_bytes(b"original")
            manifest = tools / "tools.lock.json"
            manifest.write_text(json.dumps({"files": {binary.name: suite.sha256(binary)}}))
            case = {"id": "filament", "api": "gles", "command": ["/tools/gltf_viewer"],
                    "tools_manifest_sha256": suite.sha256(manifest)}
            binary.write_bytes(b"changed")
            (out / "reference.lock.json").write_text("{}")
            a = SimpleNamespace(out=out, tools=tools)
            with patch("suite.shutil.disk_usage", return_value=SimpleNamespace(free=8 * 1024**3)), patch("suite.execute") as execute:
                self.assertEqual(suite.record(a, case, out / "case")["status"], "missing-tools")
                execute.assert_not_called()

    def test_changed_fixture_rejected_before_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "reference.lock.json").write_text("{}")
            scene = out / "scene.gd"
            scene.write_text("original scene")
            case = {"id": "godot", "api": "gles", "command": ["godot"],
                    "inputs_sha256": {"/work/scene.gd": suite.sha256(scene)}}
            scene.write_text("changed scene")
            with patch("suite.shutil.disk_usage", return_value=SimpleNamespace(free=8 * 1024**3)), patch("suite.execute") as execute:
                self.assertEqual(suite.record(SimpleNamespace(out=out), case, out / "case")["status"], "missing-assets")
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
