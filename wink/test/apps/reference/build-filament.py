#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the repaired Vulkan viewer with source/build files on VM tmpfs."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(command):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


spec = json.loads(Path("/work/filament-build.json").read_text())
if digest(Path(__file__)) != spec["builder_sha256"]:
    raise RuntimeError("Filament builder changed")
archive = Path("/sources/filament.tar.gz")
if digest(archive) != spec["source_sha256"]:
    raise RuntimeError("Pinned Filament archive changed")
scratch = Path("/tmp/filament-reference")
scratch.mkdir()
with tarfile.open(archive) as source:
    # The viewer has no WebGPU or DirectX backend. Avoid unpacking their
    # large vendored sources into the small recording machine's storage.
    members = (m for m in source if not any(
        m.name.startswith("filament/third_party/" + name + "/") for name in ("dawn", "dxc")))
    source.extractall(scratch, members=members, filter="data")
patch = Path("/suite/reference/patch-filament.py")
if digest(patch) != spec["patch_sha256"]:
    raise RuntimeError("Filament lifetime adapter changed")
run(["python3", patch, scratch / "filament"])
run(["apt-get", "update", "-qq"])
run(["apt-get", "install", "-y", "--no-install-recommends", "clang", "libc++-dev", "libc++abi-dev",
     "libsdl2-dev", "libvulkan-dev", "libgl-dev", "libglew-dev", "libpng-dev", "libzstd-dev", "pkg-config"])
os.environ.update(CC="clang", CXX="clang++")
sdl = scratch / "filament/third_party/libsdl2/tnt/CMakeLists.txt"
sdl.write_text('cmake_minimum_required(VERSION 3.19)\nproject(sdl2 C)\n'
               'find_package(SDL2 CONFIG REQUIRED)\nadd_library(sdl2 INTERFACE)\n'
               'target_link_libraries(sdl2 INTERFACE SDL2::SDL2)\n')
spec["system_sdl_adapter"] = sdl.read_text()
build = scratch / "build"
run(["cmake", "-S", scratch / "filament", "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
     "-DFILAMENT_BUILD_FILAMAT=ON", "-DFILAMENT_BUILD_GLTFIO=ON", "-DFILAMENT_SUPPORTS_VULKAN=ON",
     "-DFILAMENT_BUILD_JAVA=OFF", "-DFILAMENT_SUPPORTS_EGL_ON_LINUX=OFF"])
run(["cmake", "--build", build, "--target", "gltf_viewer", "-j4"])
out = Path("/work/filament-tools")
if out.exists():
    raise RuntimeError("Use a fresh output root for repaired Filament tools")
out.mkdir()
shutil.copy2(build / "samples/gltf_viewer", out / "gltf_viewer-vulkan")
shutil.copytree(build / "samples/assets/ibl", out / "filament-assets/ibl")
shutil.copytree(build / "samples/assets/fonts", out / "assets/fonts")
(out / "lib").mkdir()
libraries = subprocess.check_output(["ldd", out / "gltf_viewer-vulkan"], text=True)
for name in re.findall(r"=> (/\S+)", libraries):
    source = Path(name)
    if source.name not in spec["base_libraries"]:
        shutil.copy2(source, out / "lib" / source.name)
spec["packages"] = subprocess.check_output(["dpkg-query", "-W"], text=True)
spec["files"] = {str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()}
(out / "tools.lock.json").write_text(json.dumps(spec, indent=2) + "\n")
