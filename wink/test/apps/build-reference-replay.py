#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build locked software GL retracers in a fresh local VM without BuildKit."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from build_replay_support import HERE, MESA, run
from suite import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--source-lock", type=Path, required=True)
    p.add_argument("--image", default="wink-reference-apps")
    p.add_argument("--resume")
    a = p.parse_args()
    a.out = a.out.resolve()
    a.sources = a.sources.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "reference-tools/tools.lock.json").exists():
        raise RuntimeError("Use a fresh output root; reference tools are locked")
    if a.resume and not a.resume.startswith("wink-reference-tools-"):
        p.error("--resume must name this builder's VM")
    if shutil.disk_usage(a.out).free < (1 if a.resume else 4) * 1024**3:
        raise RuntimeError("Insufficient space for the reference retracer build")
    lock = json.loads(a.source_lock.read_text())
    pins = json.loads((HERE / "pins.json").read_text())
    expected = next(v for k, v in lock["inputs"].items() if k.endswith("apitrace.tar.gz"))
    if lock["pins"]["apitrace"] != pins["apitrace"] or sha256(a.sources / "apitrace.tar.gz") != expected:
        raise RuntimeError("Reference apitrace archive or revision changed")
    identity = json.loads(subprocess.check_output(["container", "image", "inspect", a.image], text=True))
    query = "from pathlib import Path; import json; print(json.dumps(sorted({p.name for d in ('/lib/aarch64-linux-gnu','/usr/lib/aarch64-linux-gnu','/opt/reference/lib','/opt/tools/lib') for p in Path(d).glob('*') if p.is_file()})))"
    libraries = json.loads(subprocess.check_output(["container", "run", "--rm", "--disable-progress-updates", "--cpus", "1",
        "--memory", "256m", "--entrypoint", "python3", a.image, "-c", query], text=True))
    scripts = ["reference/build-retrace.py", "glreplay.py", "replay/thread_context.hpp", "replay/egl_sync.hpp", "build_replay_support.py", "build-replay.py"]
    name = a.resume or "wink-reference-tools-" + str(time.time_ns())
    spec = {"schema": 1, "apitrace": pins["apitrace"], "image_identity": identity, "build_vm": name,
        "archive_sha256": expected, "base_libraries": libraries,
        "builder_sources": {s: (HERE / s).read_text() for s in scripts},
        "builder_hashes": {s: sha256(HERE / s) for s in scripts}}
    path = a.out / "reference-replay-build.json"
    if a.resume:
        old = json.loads(path.read_text())
        if any(old[k] != spec[k] for k in ("build_vm", "archive_sha256", "image_identity")):
            raise RuntimeError("Resume VM, image or source changed")
    path.write_text(json.dumps(spec, indent=2) + "\n")
    print("Build VM:", name, "(retained on failure; use --resume)", flush=True)
    command = ["container", "start", "--attach", name] if a.resume else ["container", "run", "--name", name,
        "--disable-progress-updates", "--cpus", "4", "--memory", "8g", "--volume", f"{a.sources}:/sources:ro",
        "--volume", f"{HERE}:/suite:ro", "--volume", f"{a.out}:/work", "--entrypoint", "python3", a.image,
        "/suite/reference/build-retrace.py"]
    run(command)
    run(["container", "delete", name])
    print("Built", a.out / "reference-tools", flush=True)


if __name__ == "__main__":
    main()
