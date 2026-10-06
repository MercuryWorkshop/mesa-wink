#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a pinned Filament Vulkan viewer with valid attachment lifetimes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from build_replay_support import HERE

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--sources", type=Path, required=True)
p.add_argument("--source-lock", type=Path, required=True, help="existing locked app tools manifest")
p.add_argument("--image", required=True)
a = p.parse_args()
a.out = a.out.resolve(); a.sources = a.sources.resolve()
a.out.mkdir(parents=True, exist_ok=True)
if (a.out / "filament-tools").exists():
    raise RuntimeError("Use a fresh output root for repaired tools")
lock = json.loads(a.source_lock.read_text())
pins = json.loads((HERE / "pins.json").read_text())
if lock["pins"]["applications"]["filament"] != pins["applications"]["filament"]:
    raise RuntimeError("Filament revision differs from pins.json")
identity = json.loads(subprocess.check_output(["container", "image", "inspect", a.image], text=True))
query = "from pathlib import Path; import json; print(json.dumps(sorted({p.name for d in ('/lib/aarch64-linux-gnu','/usr/lib/aarch64-linux-gnu','/opt/reference/lib','/opt/tools/lib') for p in Path(d).glob('*') if p.is_file()})))"
libraries = json.loads(subprocess.check_output(["container", "run", "--rm", "--disable-progress-updates",
    "--cpus", "1", "--memory", "256m", "--entrypoint", "python3", a.image, "-c", query], text=True))
patch = HERE / "reference/patch-filament.py"
builder = HERE / "reference/build-filament.py"
spec = {"schema": 1, "pins": pins, "image_identity": identity, "base_libraries": libraries,
    "source_sha256": lock["sources"]["filament"], "patch_source": patch.read_text(),
    "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(), "builder_source": builder.read_text(),
    "builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest()}
(a.out / "filament-build.json").write_text(json.dumps(spec, indent=2) + "\n")
subprocess.run(["container", "run", "--rm", "--disable-progress-updates", "--cpus", "4", "--memory", "8g",
    "--tmpfs", "/tmp", "--volume", f"{a.sources}:/sources:ro", "--volume", f"{HERE}:/suite:ro",
    "--volume", f"{a.out}:/work", "--entrypoint", "python3", a.image, "/suite/reference/build-filament.py"], check=True)
