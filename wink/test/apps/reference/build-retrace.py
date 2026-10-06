#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the software retracers with the same WSI/context adapters as Wink."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()

def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)

spec = json.loads(Path("/work/reference-replay-build.json").read_text())
for name, expected in spec["builder_hashes"].items():
    if digest(Path("/suite") / name) != expected:
        raise RuntimeError("Builder changed: " + name)
archive = Path("/sources/apitrace.tar.gz")
if digest(archive) != spec["archive_sha256"]:
    raise RuntimeError("Source archive changed")
scratch = Path("/tmp/wink-reference-replay")
scratch.mkdir(exist_ok=True)
if not (scratch / "apitrace").exists():
    with tarfile.open(archive) as source:
        source.extractall(scratch, filter="data")
sys.path.insert(0, "/suite")
from glreplay import adapt_wsi
adapt_wsi(scratch / "apitrace")
run(["apt-get", "update"])
run(["apt-get", "install", "-y", "--no-install-recommends", "cmake", "ninja-build", "g++", "pkg-config",
     "libegl1-mesa-dev", "libgles2-mesa-dev", "libgl-dev", "libx11-dev", "libxext-dev", "libxrandr-dev",
     "libpng-dev", "zlib1g-dev", "libsnappy-dev", "libzstd-dev", "libbrotli-dev"])
build = scratch / "build"
run(["cmake", "-S", scratch / "apitrace", "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
     "-DENABLE_GUI=OFF", "-DBUILD_TESTING=OFF"])
run(["cmake", "--build", build, "--target", "glretrace", "eglretrace", "-j4"])
out = Path("/work/reference-tools")
temporary = out.with_name("reference-tools.partial")
if temporary.exists():
    shutil.rmtree(temporary)
(temporary / "bin").mkdir(parents=True)
(temporary / "lib").mkdir()
for name in ("glretrace", "eglretrace"):
    binary = next(p for p in (build / name, build / "bin" / name, build / "retrace" / name) if p.is_file())
    shutil.copy2(binary, temporary / "bin" / name)
    for path in re.findall(r"=> (/\S+)", subprocess.check_output(["ldd", binary], text=True)):
        source = Path(path)
        if source.name not in spec["base_libraries"]:
            shutil.copy2(source, temporary / "lib" / source.name)
spec["packages"] = subprocess.check_output(["dpkg-query", "-W"], text=True)
spec["files"] = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob("*") if p.is_file()}
(temporary / "tools.lock.json").write_text(json.dumps(spec, indent=2) + "\n")
if out.exists():
    raise RuntimeError("Reference tools are locked; choose a new output root")
temporary.rename(out)
