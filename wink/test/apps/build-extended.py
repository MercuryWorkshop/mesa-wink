#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build pinned app tools in a fresh recording VM, independently of BuildKit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
from build_replay_support import HERE, MESA, run


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps")
    p.add_argument("--image", default="wink-reference-apps")
    p.add_argument("--source-lock", type=Path, help="reuse archives from a previously verified extended context")
    p.add_argument("--sources", type=Path, help="reuse a verified reference-context directory from another output root")
    p.add_argument("--resume", help="resume a failed wink-app-tools VM using its existing compilation cache")
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / "extended-tools/tools.lock.json").exists():
        raise RuntimeError("Refusing to overwrite locked app tools; use a fresh output root")
    if a.resume and not a.resume.startswith("wink-app-tools-"):
        p.error("--resume must name a VM created by this builder")
    if shutil.disk_usage(a.out).free < (1 if a.resume else 4 if a.source_lock else 12) * 1024**3:
        raise RuntimeError("Insufficient disk space for the fresh app build VM")
    pins = json.loads((HERE / "pins.json").read_text())
    stage = a.sources.resolve() if a.sources else a.out / "reference-context"
    if a.sources and not a.source_lock:
        p.error("--sources requires --source-lock")
    if a.source_lock:
        lock = json.loads(a.source_lock.read_text())
        if lock["pins"]["applications"] != pins["applications"]:
            raise RuntimeError("Staged application revisions differ from pins.json")
        bundle = stage / "app-inputs.tar.gz"
        expected = next(value for key, value in lock["inputs"].items() if key.endswith("app-inputs.tar.gz"))
        if digest(bundle) != expected:
            raise RuntimeError("Staged app bundle differs from verified image inputs")
        with tarfile.open(bundle) as archive:
            for name in ("filament", "ppsspp", "pspautotests"):
                with archive.extractfile(name + ".tar.gz") as source:
                    if hashlib.file_digest(source, "sha256").hexdigest() != digest(stage / (name + ".tar.gz")):
                        raise RuntimeError("Staged source archive changed: " + name)
    else:
        stage.mkdir(exist_ok=True)
        spec = importlib.util.spec_from_file_location("extended_apps", HERE / "extended-apps.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.stage_extended(a.out, stage, pins)
    identity = json.loads(subprocess.check_output(["container", "image", "inspect", a.image], text=True))
    if not identity:
        raise RuntimeError("Build the --apps reference image first")
    name = a.resume or "wink-app-tools-" + str(time.time_ns())
    library_query = "from pathlib import Path; import json; print(json.dumps(sorted({p.name for d in ('/lib/aarch64-linux-gnu','/usr/lib/aarch64-linux-gnu','/opt/reference/lib','/opt/tools/lib') for p in Path(d).glob('*') if p.is_file()})))"
    base_libraries = json.loads(subprocess.check_output(["container", "run", "--rm", "--disable-progress-updates",
        "--cpus", "1", "--memory", "256m", "--entrypoint", "python3", a.image, "-c", library_query], text=True))
    specification = {"schema": 1, "pins": pins, "image_identity": identity, "build_vm": name,
        "base_libraries": base_libraries,
        "builder_sha256": digest(HERE / "reference/build-apps.py"),
        "builder_source": (HERE / "reference/build-apps.py").read_text(),
        "sources": {name: digest(stage / (name + ".tar.gz")) for name in ("filament", "ppsspp", "pspautotests")}}
    previous = a.out / "extended-build.json"
    if a.resume and (not previous.exists() or json.loads(previous.read_text()).get("build_vm") != name):
        raise RuntimeError("Resume VM does not match this output root's build specification")
    if a.resume:
        old = json.loads(previous.read_text())
        if old["image_identity"] != identity or old["sources"] != specification["sources"]:
            raise RuntimeError("Resume image or source identity changed")
    (a.out / "extended-build.json").write_text(json.dumps(specification, indent=2) + "\n")
    print("Build VM:", name, "(retained on failure; resume with --resume)", flush=True)
    command = ["container", "start", "--attach", name] if a.resume else ["container", "run", "--name", name, "--disable-progress-updates", "--cpus", "4", "--memory", "8g",
         "--volume", f"{stage}:/sources:ro", "--volume", f"{HERE}:/suite:ro", "--volume", f"{a.out}:/work",
         "--entrypoint", "python3", a.image, "/suite/reference/build-apps.py"]
    run(command)
    run(["container", "delete", name])
    print("Built", a.out / "extended-tools", flush=True)


if __name__ == "__main__":
    main()
