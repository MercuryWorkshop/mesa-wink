#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the Qt recording fixture and lock its pre-recording adapters."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


source = Path("/suite/fixtures/qtquick")
build = Path("/tmp/qt-fixture")
lock = json.loads(Path("/work/qt-build.json").read_text())
if digest(Path(__file__)) != lock["builder_sha256"]:
    raise RuntimeError("Qt fixture builder changed")
subprocess.run(["apt-get", "update", "-qq"], check=True)
subprocess.run(["apt-get", "install", "-y", "--no-install-recommends", "qt6-base-private-dev"], check=True)
subprocess.run(["cmake", "-S", source, "-B", build, "-G", "Ninja"], check=True)
subprocess.run(["cmake", "--build", build, "-j2"], check=True)
out = Path("/work/qt-tools")
if out.exists():
    raise RuntimeError("Use a fresh output root for changed Qt tools")
out.mkdir()
for name in ("wink-qt-fixture",):
    shutil.copy2(build / name, out / name)
lock["packages"] = subprocess.check_output(["dpkg-query", "-W"], text=True)
lock["fixture_sources"] = {p.name: {"sha256": digest(p), "source": p.read_text()}
                           for p in source.iterdir() if p.is_file()}
lock["files"] = {p.name: digest(p) for p in out.iterdir() if p.is_file()}
(out / "tools.lock.json").write_text(json.dumps(lock, indent=2) + "\n")
