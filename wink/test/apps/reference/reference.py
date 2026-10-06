#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import argparse
import atexit
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from vulkan_lifetimes import audit_trace


def run(cmd, cwd=None, env=None):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, env=env, check=True, text=True)


def replay(api, trace, out, checkpoints):
    out.mkdir(parents=True, exist_ok=True)
    if api == "vulkan":
        cmd = ["gfxrecon-replay", "--wsi", "headless", "-m", "rebind", "--screenshots", checkpoints,
               "--screenshot-format", "png", "--screenshot-dir", str(out), str(trace)]
    else:
        cmd = ["eglretrace" if api == "gles" else "glretrace", "--singlethread", "--headless", "--debug",
               "-s", str(out / "frame-"), "-S", checkpoints, str(trace)]
    run(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["probe", "record", "replay", "shell"])
    p.add_argument("--api", choices=["gl", "gles", "vulkan"], default="gl")
    p.add_argument("--out", type=Path, default=Path("/work/probe"))
    p.add_argument("--trace", type=Path)
    p.add_argument("--checkpoints")
    p.add_argument("--env", action="append", default=[])
    p.add_argument("--tail-frames", type=int, default=64)
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    a = p.parse_args(argv[:split])
    a.command = argv[split + 1:]
    for value in a.env:
        key, value = value.split("=", 1)
        os.environ[key] = value
    if Path("/reference-tools/bin").is_dir():
        os.environ["PATH"] = "/reference-tools/bin:" + os.environ["PATH"]
        os.environ["LD_LIBRARY_PATH"] = "/opt/reference/lib:/opt/tools/lib:/reference-tools/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ.setdefault("VK_LAYER_PATH", "/opt/tools/share/vulkan/explicit_layer.d")
    if a.action == "shell":
        return subprocess.call(a.command or ["/bin/bash"])
    # Xvfb keeps real GLX/EGL application context creation, with CPU rendering.
    if not os.environ.get("DISPLAY"):
        # xvfb-run's SIGUSR1 handshake blocks with its shell as PID 1.
        # Keep Python as PID 1 and check Xvfb's socket instead.
        display = 99
        os.environ["DISPLAY"] = f":{display}"
        xvfb = subprocess.Popen(["Xvfb", f":{display}", "-screen", "0", "1280x720x24", "-ac", "-nolisten", "tcp"])
        atexit.register(xvfb.terminate)
        for _ in range(200):
            if Path(f"/tmp/.X11-unix/X{display}").exists():
                break
            if xvfb.poll() is not None:
                raise RuntimeError("Xvfb startup failed")
            time.sleep(0.1)
        else:
            raise RuntimeError("Xvfb startup timeout")
    a.out.mkdir(parents=True, exist_ok=True)
    info = subprocess.check_output(["glxinfo", "-B"], text=True)
    vk = subprocess.check_output(["vulkaninfo", "--summary"], text=True, stderr=subprocess.DEVNULL)
    if "llvmpipe" not in info or "llvmpipe" not in vk:
        raise RuntimeError("reference renderer is not llvmpipe/lavapipe")
    (a.out / "identity.json").write_text(json.dumps({"gl": info, "vulkan": vk,
        "packages": subprocess.check_output(["dpkg-query", "-W"], text=True)}, indent=2))
    checkpoints = a.checkpoints or "1,8,32,64"
    trace = a.trace or a.out / ("capture.gfxr" if a.api == "vulkan" else "capture.trace")
    if a.action in ("record", "probe"):
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        if a.action == "probe":
            command = ["wink-reference-vk-probe"] if a.api == "vulkan" else ["wink-reference-probe", a.api]
        if not command:
            raise RuntimeError("record requires an application command after --")
        if a.api == "vulkan":
            env = dict(os.environ, VK_INSTANCE_LAYERS="VK_LAYER_LUNARG_gfxreconstruct",
                       LD_LIBRARY_PATH="/opt/reference/lib:/opt/tools/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
                       GFXRECON_CAPTURE_FILE=str(trace), GFXRECON_CAPTURE_FILE_TIMESTAMP="0")
            run(command, a.out, env)
        else:
            run(["apitrace", "trace", "--api", "egl" if a.api == "gles" or a.action == "probe" else "gl",
                 "-o", str(trace), *command], a.out)
        if not trace.is_file() or not trace.stat().st_size:
            raise RuntimeError("application did not produce a trace")
    if a.api != "vulkan" and not a.checkpoints:
        dump = subprocess.check_output(["apitrace", "dump", "--calls=*/frame", str(trace)], text=True)
        calls = [int(v) for v in re.findall(r"^(\d+) ", dump, re.M)]
        if a.action == "probe":
            selected = calls[-2:]
        else:
            tail = calls[-a.tail_frames:]
            if len(tail) < 4:
                raise RuntimeError("too few application frames")
            selected = [tail[i] for i in sorted(set([len(tail)//8, len(tail)//3, 2*len(tail)//3, len(tail)-2]))]
        checkpoints = ",".join(map(str, selected))
        (a.out / "checkpoints.json").write_text(json.dumps({"calls": selected, "total_frames": len(calls)}))
    if a.api == "vulkan":
        audit = audit_trace(trace)
        (a.out / "vulkan-lifetimes.json").write_text(json.dumps(audit, indent=2) + "\n")
        if not audit["pass"]:
            raise RuntimeError("Vulkan attachment lifetime validation error: " + json.dumps(audit["errors"][:3]))
    # Reference repeatability is required, independent of the candidate.
    for repeat in range(3):
        replay(a.api, trace, a.out / f"reference-{repeat}", checkpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
