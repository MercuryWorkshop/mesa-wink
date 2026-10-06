# SPDX-License-Identifier: MIT
"""Check attachment lifetimes in an unmodified GFXReconstruct capture.

This checks framebuffer and dynamic-rendering attachments at recording time.
It is a focused admission check, not a replacement for Vulkan validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(rows):
    images, views, framebuffers = {}, {}, {}
    errors, error_count, checked = [], 0, 0

    def error(index, message):
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append({"index": index, "error": message})

    def check_view(view, index, saved=None):
        nonlocal checked
        if not view:
            return
        checked += 1
        current = views.get(view)
        if current is None or (saved is not None and current != saved):
            error(index, f"Attachment image view {view} was destroyed or replaced")
        elif current[0] not in images or images[current[0]] != current[1]:
            error(index, f"Attachment image {current[0]} was destroyed or replaced")

    for row in rows:
        f = row.get("function", {})
        name, args, index = f.get("name"), f.get("args", {}), row.get("index")
        if name in ("vkCreateImage", "vkCreateImageView", "vkCreateFramebuffer") and f.get("return") != "VK_SUCCESS":
            continue
        if name == "vkCreateImage":
            images[args["pImage"]] = index
        elif name == "vkGetSwapchainImagesKHR":
            for image in args.get("pSwapchainImages") or []:
                images.setdefault(image, index)
        elif name == "vkDestroyImage":
            images.pop(args["image"], None)
        elif name == "vkCreateImageView":
            image = args["pCreateInfo"]["image"]
            views[args["pView"]] = (image, images.get(image), index)
        elif name == "vkDestroyImageView":
            views.pop(args["imageView"], None)
        elif name == "vkCreateFramebuffer":
            info = args["pCreateInfo"]
            imageless = int(info["flags"], 0) & 1
            attachments = [] if imageless else info.get("pAttachments") or []
            framebuffers[args["pFramebuffer"]] = [(v, views.get(v)) for v in attachments]
            for view in attachments:
                check_view(view, index)
        elif name == "vkDestroyFramebuffer":
            framebuffers.pop(args["framebuffer"], None)
        elif name in ("vkCmdBeginRenderPass", "vkCmdBeginRenderPass2", "vkCmdBeginRenderPass2KHR"):
            info = args["pRenderPassBegin"]
            framebuffer = info["framebuffer"]
            if framebuffer not in framebuffers:
                error(index, f"Framebuffer {framebuffer} was destroyed or not created")
            for view, saved in framebuffers.get(framebuffer, []):
                check_view(view, index, saved)
            extension = info.get("pNext")
            while extension:
                if extension.get("sType") == "VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO":
                    for view in extension.get("pAttachments") or []:
                        check_view(view, index)
                extension = extension.get("pNext")
        elif name in ("vkCmdBeginRendering", "vkCmdBeginRenderingKHR"):
            info = args["pRenderingInfo"]
            attachments = list(info.get("pColorAttachments") or [])
            attachments += [info.get("pDepthAttachment"), info.get("pStencilAttachment")]
            for att in filter(None, attachments):
                check_view(att.get("imageView"), index)
                check_view(att.get("resolveImageView"), index)
    return {"schema": 1, "pass": error_count == 0, "checked_attachments": checked,
            "error_count": error_count, "errors": errors}


def audit_trace(trace, converter="gfxrecon-convert", jsonl=None):
    if jsonl:
        with Path(jsonl).open() as source:
            result = audit(json.loads(line) for line in source)
    else:
        with subprocess.Popen([converter, "--format", "jsonl", "--output", "stdout", str(trace)],
                              stdout=subprocess.PIPE, text=True) as process:
            try:
                result = audit(json.loads(line) for line in process.stdout)
            except BaseException:
                process.kill()
                raise
            if process.wait():
                raise RuntimeError("Vulkan capture conversion failed")
    result["trace_sha256"] = digest(trace)
    result["checker_sha256"] = digest(__file__)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path)
    parser.add_argument("--converter", default="gfxrecon-convert")
    args = parser.parse_args()
    result = audit_trace(args.trace, args.converter, args.jsonl)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Vulkan attachment lifetime audit: {result['error_count']} errors")
    raise SystemExit(0 if result["pass"] else 1)
