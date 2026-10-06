#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compile app binaries in an ephemeral Linux VM and export only runtime tools."""
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


def run(command, env=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, env=env)


spec = json.loads(Path("/work/extended-build.json").read_text())
if digest(Path(__file__)) != spec["builder_sha256"]:
    raise RuntimeError("Builder script checksum changed")
scratch = Path("/tmp/wink-apps-build")
scratch.mkdir(exist_ok=True)
present_libraries = set(spec["base_libraries"])
for name, expected in spec["sources"].items():
    archive = Path("/sources") / (name + ".tar.gz")
    if digest(archive) != expected:
        raise RuntimeError("Source archive checksum changed: " + name)
    marker = scratch / (name + ".source-sha256")
    if marker.exists() and marker.read_text() != expected:
        raise RuntimeError("Resume source revision changed: " + name)
    if not marker.exists():
        with tarfile.open(archive) as source:
            source.extractall(scratch, filter="data")
        marker.write_text(expected)
run(["apt-get", "update"])
run(["apt-get", "install", "-y", "--no-install-recommends", "clang", "libc++-dev", "libc++abi-dev",
     "libegl1-mesa-dev", "libgles2-mesa-dev", "libsdl2-dev", "libvulkan-dev", "libgl-dev", "libglew-dev",
     "libpng-dev", "libsnappy-dev", "libzstd-dev", "libavcodec-dev", "libavformat-dev", "libavutil-dev",
     "libswresample-dev", "libswscale-dev", "libavdevice-dev", "libavfilter-dev",
     "libzip-dev", "zlib1g-dev", "pkg-config", "python3-dev"])
env = dict(os.environ, CC="clang", CXX="clang++")
# Filament's bundled SDL configuration assumes desktop Linux architectures.
# Use the image's packaged ARM64 SDL, preserving the same SDL application API.
sdl_adapter = '''cmake_minimum_required(VERSION 3.19)
project(sdl2 C)
find_package(SDL2 CONFIG REQUIRED)
add_library(sdl2 INTERFACE)
target_link_libraries(sdl2 INTERFACE SDL2::SDL2)
'''
(scratch / "filament/third_party/libsdl2/tnt/CMakeLists.txt").write_text(sdl_adapter)
spec["system_sdl_adapter"] = sdl_adapter
filament = scratch / "filament-build"
run(["cmake", "-S", scratch / "filament", "-B", filament, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
     "-DFILAMENT_BUILD_FILAMAT=ON", "-DFILAMENT_BUILD_GLTFIO=ON", "-DFILAMENT_SUPPORTS_VULKAN=ON",
     "-DFILAMENT_BUILD_JAVA=OFF", "-DFILAMENT_SUPPORTS_EGL_ON_LINUX=ON"], env)
run(["cmake", "--build", filament, "--target", "gltf_viewer", "-j4"])
ppsspp = scratch / "ppsspp-build"
run(["cmake", "-S", scratch / "ppsspp", "-B", ppsspp, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
     "-DUSING_GLES2=ON", "-DUSING_EGL=ON", "-DUSE_SYSTEM_FFMPEG=ON", "-DUNITTEST=OFF"])
run(["cmake", "--build", ppsspp, "--target", "PPSSPPSDL", "-j4"])
out = Path("/work/extended-tools")
temporary = out.with_name("extended-tools.partial")
if temporary.exists():
    shutil.rmtree(temporary)
temporary.mkdir()
shutil.copy2(filament / "samples/gltf_viewer", temporary)
shutil.copytree(filament / "samples/assets/ibl", temporary / "filament-assets/ibl")
shutil.copy2(ppsspp / "PPSSPPSDL", temporary)
shutil.copytree(scratch / "ppsspp/assets", temporary / "assets")
shutil.copytree(filament / "samples/assets/fonts", temporary / "assets/fonts", dirs_exist_ok=True)
shutil.copytree(scratch / "pspautotests/demos", temporary / "psp-demos")
(temporary / "lib").mkdir()
for binary in (temporary / "gltf_viewer", temporary / "PPSSPPSDL"):
    libraries = subprocess.check_output(["ldd", binary], text=True)
    for name in re.findall(r"=> (/\S+)", libraries):
        source = Path(name)
        if source.name not in present_libraries:
            shutil.copy2(source, temporary / "lib" / source.name)
spec["packages"] = subprocess.check_output(["dpkg-query", "-W"], text=True)
spec["files"] = {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob("*") if p.is_file()}
(temporary / "tools.lock.json").write_text(json.dumps(spec, indent=2) + "\n")
if out.exists():
    raise RuntimeError("Refusing to overwrite locked app tools; use a fresh output root")
temporary.rename(out)
print("WINK_FIXTURE extended tools built", flush=True)
