#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build corrected Qt fixtures in the pinned local recording image."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from build_replay_support import HERE

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--image", required=True)
a = p.parse_args()
a.out = a.out.resolve()
a.out.mkdir(parents=True, exist_ok=True)
if (a.out / "qt-tools").exists():
    raise RuntimeError("Use a fresh output root for changed Qt tools")
builder = HERE / "reference/build-qt.py"
identity = json.loads(subprocess.check_output(["container", "image", "inspect", a.image], text=True))
(a.out / "qt-build.json").write_text(json.dumps({"schema": 1, "image_identity": identity,
    "builder_source": builder.read_text(), "builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest()}, indent=2) + "\n")
subprocess.run(["container", "run", "--rm", "--disable-progress-updates", "--cpus", "2", "--memory", "1g",
    "--volume", f"{HERE}:/suite:ro", "--volume", f"{a.out}:/work", "--entrypoint", "python3",
    a.image, "/suite/reference/build-qt.py"], check=True)
