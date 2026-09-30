#!/usr/bin/env python3
"""Replay a gfxreconstruct capture on the host's Vulkan driver and on wink,
and compare what each presented, frame by frame.

    trace-diff.py CAPTURE.gfxr [--frames 10,30,50] [--out DIR] [--erode 2]

Capture on wink, so that the application only chose what wink offers:

    VK_DRIVER_FILES=<wink icd> VK_LAYER_PATH=<gfxrecon>/layer \\
      VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_gfxreconstruct \\
      GFXRECON_CAPTURE_FILE=app.gfxr APP

The host driver (RADV, say) is the reference: it replays with the portability
subset removed and resources rebound to its own requirements.

Two GPUs legitimately disagree along primitive edges -- a pixel either side
of where an edge lands, the last bit of an interpolation -- and never across
a region's interior. So the per-pixel difference mask is eroded: a differing
pixel counts only if everything within --erode of it differs too, which no
edge fringe does. What is left is a defect (a dropped draw, a wrong texture,
a bad blend), and its count is what should be zero. This is gex's
tools/surface-diff.py, for gfxreconstruct's screenshots.
"""
import argparse
import os
import subprocess
import sys

from PIL import Image, ImageChops, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
MESA = os.path.abspath(os.path.join(HERE, "../../../.."))


def replay(capture, frames, out, env_extra, flags):
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, **env_extra)
    cmd = [os.environ.get("GFXRECON_REPLAY", "gfxrecon-replay"), *flags, "--screenshots", frames,
           "--screenshot-format", "png", "--screenshot-dir", out, capture]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    fatal = [l for l in (result.stdout + result.stderr).splitlines() if "FATAL" in l or "ERROR" in l]
    return result.returncode, fatal[:3]


def compare(a_path, b_path, tolerance, erode, mask_path):
    a = Image.open(a_path).convert("RGB")
    b = Image.open(b_path).convert("RGB")
    if a.size != b.size:
        return None, f"sizes differ: {a.size} vs {b.size}"
    diff = ImageChops.difference(a, b).convert("L")
    raw = diff.point(lambda v: 255 if v > tolerance else 0)
    differing = raw.histogram()[255]
    eroded = raw.filter(ImageFilter.MinFilter(2 * erode + 1)) if erode else raw
    surviving = eroded.histogram()[255]
    if surviving:
        eroded.save(mask_path)
    return surviving, f"{differing} pixels differ, {surviving} survive erosion"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture")
    p.add_argument("--frames", default="10,30,50")
    p.add_argument("--out", default=None)
    p.add_argument("--tolerance", type=int, default=8, help="per-channel difference that counts")
    p.add_argument("--erode", type=int, default=2)
    p.add_argument("--wsi", default="xcb")
    args = p.parse_args()

    out = args.out or os.path.splitext(args.capture)[0] + ".diff"
    icd = os.environ.get("WINK_ICD") or next(
        (os.path.join(root, f) for root, _, files in os.walk(os.path.join(MESA, "build"))
         for f in files if f.startswith("wink_devenv_icd.") and f.endswith(".json")), None)
    if not icd:
        sys.exit("no wink ICD: set WINK_ICD")

    common = ["--wsi", args.wsi]
    ref_rc, ref_err = replay(args.capture, args.frames, os.path.join(out, "host"), {},
                             common + ["--remove-unsupported", "-m", "rebind"])
    wink_rc, wink_err = replay(args.capture, args.frames, os.path.join(out, "wink"), {"VK_DRIVER_FILES": icd},
                               common + ["-m", "remap"])
    for name, rc, err in (("host", ref_rc, ref_err), ("wink", wink_rc, wink_err)):
        if rc or err:
            print(f"{name} replay: exit {rc}", *err, sep="\n  ")

    bad = 0
    shots = sorted(f for f in os.listdir(os.path.join(out, "host")) if f.endswith(".png"))
    for shot in shots:
        wink = os.path.join(out, "wink", shot)
        if not os.path.exists(wink):
            print(f"{shot}: wink presented nothing")
            bad += 1
            continue
        surviving, note = compare(os.path.join(out, "host", shot), wink, args.tolerance, args.erode,
                                  os.path.join(out, "mask-" + shot))
        print(f"{shot}: {note}")
        bad += surviving is None or surviving > 0
    if not shots:
        print("the host replay presented nothing")
        bad = 1
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
