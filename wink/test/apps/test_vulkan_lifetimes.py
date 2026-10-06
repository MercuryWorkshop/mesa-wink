# SPDX-License-Identifier: MIT
import json
from pathlib import Path
import tempfile
import unittest
import suite
from reference.vulkan_lifetimes import audit, digest


def call(name, **args):
    return {"function": {"name": name, "return": "VK_SUCCESS", "args": args}}


def capture():
    return [call("vkCreateImage", pImage=1),
            call("vkCreateImageView", pView=2, pCreateInfo={"image": 1}),
            call("vkCreateFramebuffer", pFramebuffer=3,
                 pCreateInfo={"flags": "0x0", "pAttachments": [2]}),
            call("vkCmdBeginRenderPass", pRenderPassBegin={"framebuffer": 3})]


def indexed(rows):
    return [dict(row, index=index) for index, row in enumerate(rows)]


class AttachmentLifetimeTests(unittest.TestCase):
    def test_valid_capture_and_cleanup(self):
        rows = capture() + [call("vkDestroyFramebuffer", framebuffer=3),
                            call("vkDestroyImageView", imageView=2), call("vkDestroyImage", image=1)]
        self.assertTrue(audit(indexed(rows))["pass"])

    def test_cached_framebuffer_after_view_destruction(self):
        rows = capture() + [call("vkDestroyImageView", imageView=2),
                            call("vkDestroyImage", image=1), capture()[-1]]
        result = audit(indexed(rows))
        self.assertFalse(result["pass"])
        self.assertEqual(result["errors"][0]["index"], 6)

    def test_handle_reuse_does_not_revive_old_framebuffer(self):
        rows = capture() + [call("vkDestroyImageView", imageView=2),
                            call("vkCreateImageView", pView=2, pCreateInfo={"image": 1}), capture()[-1]]
        self.assertFalse(audit(indexed(rows))["pass"])

    def test_destroyed_parent_with_live_view(self):
        self.assertFalse(audit(indexed(capture() + [call("vkDestroyImage", image=1), capture()[-1]]))["pass"])

    def test_swapchain_images_and_repeated_enumeration(self):
        rows = capture()
        rows[0] = call("vkGetSwapchainImagesKHR", pSwapchainImages=[1])
        rows.insert(3, call("vkGetSwapchainImagesKHR", pSwapchainImages=[1]))
        self.assertTrue(audit(indexed(rows))["pass"])

    def test_dynamic_and_imageless_attachments(self):
        rows = capture()[:2] + [call("vkDestroyImage", image=1),
            call("vkCmdBeginRendering", pRenderingInfo={"pColorAttachments": [{"imageView": 2}]}),
            call("vkCreateFramebuffer", pFramebuffer=3, pCreateInfo={"flags": "0x1", "pAttachments": None}),
            call("vkCmdBeginRenderPass2", pRenderPassBegin={"framebuffer": 3, "pNext": {
                "sType": "VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO", "pAttachments": [2]}})]
        self.assertEqual(audit(indexed(rows))["error_count"], 2)

    def test_invalid_audit_rejects_stable_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"api": "vulkan"}
            trace = root / "capture.gfxr"
            trace.write_bytes(b"capture")
            checksum = digest(trace)
            (root / "capture.lock.json").write_text(json.dumps({"case_sha256": suite.case_hash(case),
                                                               "trace_sha256": checksum}))
            result = audit(indexed(capture() + [call("vkDestroyImage", image=1), capture()[-1]]))
            result.update(trace_sha256=checksum, checker_sha256=digest(suite.HERE / "reference/vulkan_lifetimes.py"))
            (root / "vulkan-lifetimes.json").write_text(json.dumps(result))
            self.assertEqual(suite.replay(None, case, root, "st")["status"], "reference-incompatible")

    def test_missing_audit_rejects_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"api": "vulkan"}
            trace = root / "capture.gfxr"
            trace.write_bytes(b"capture")
            (root / "capture.lock.json").write_text(json.dumps({"case_sha256": suite.case_hash(case),
                                                               "trace_sha256": digest(trace)}))
            self.assertEqual(suite.replay(None, case, root, "st")["status"], "reference-incompatible")
