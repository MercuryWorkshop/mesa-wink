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
headers = scratch / "filament/filament/backend/src/opengl/gl_headers.h"
original = "#if defined(__ANDROID__) || defined(FILAMENT_USE_EXTERNAL_GLES3) || defined(__EMSCRIPTEN__)"
replacement = original + " || defined(WINK_REFERENCE_GLES)"
content = headers.read_text()
if replacement not in content:
    if content.count(original) != 1:
        raise RuntimeError("Filament GLES header selection changed upstream")
    headers.write_text(content.replace(original, replacement))
backend = scratch / "filament/filament/backend/CMakeLists.txt"
gles_adapter = "\nif(WINK_REFERENCE_GLES)\n  target_compile_definitions(backend PRIVATE WINK_REFERENCE_GLES)\n  target_link_libraries(backend PUBLIC GLESv2)\nendif()\n"
if gles_adapter not in backend.read_text():
    backend.write_text(backend.read_text() + gles_adapter)
spec["gles_header_adapter"] = replacement
spec["gles_link_adapter"] = gles_adapter
material = scratch / "filament/libs/gltfio/src/JitShaderProvider.cpp"
content = material.read_text()
# The desktop viewer's JIT defaults to desktop-only material packages.
# Include both shader models so the GLES engine selects the mobile variant.
if "builder.platform(MaterialBuilder::Platform::ALL);" not in content:
    material.write_text(content.replace("    builder.name(name)",
        "    builder.platform(MaterialBuilder::Platform::ALL);\n    builder.name(name)"))
platform = scratch / "filament/filament/backend/src/opengl/platforms/PlatformEGLHeadless.cpp"
content = platform.read_text()
if "#ifndef WINK_REFERENCE_GLES" not in content:
    content = content.replace("    int bindBlueGL = bluegl::bind();", "#ifndef WINK_REFERENCE_GLES\n    int bindBlueGL = bluegl::bind();")
    content = content.replace("    return PlatformEGL::createDriver(sharedContext, driverConfig);", "#endif\n    return PlatformEGL::createDriver(sharedContext, driverConfig);")
    platform.write_text(content)
filament = scratch / "filament-build"
run(["cmake", "-S", scratch / "filament", "-B", filament, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
     "-DFILAMENT_BUILD_FILAMAT=ON", "-DFILAMENT_BUILD_GLTFIO=ON", "-DFILAMENT_SUPPORTS_VULKAN=ON",
     "-DFILAMENT_BUILD_JAVA=OFF", "-DFILAMENT_SUPPORTS_EGL_ON_LINUX=ON", "-DFILAMENT_LINUX_IS_MOBILE=ON",
     "-DIS_HOST_PLATFORM=TRUE", "-DWINK_REFERENCE_GLES=ON"], env)
run(["cmake", "--build", filament, "--target", "gltf_viewer", "-j4"])
filament_vk = scratch / "filament-vulkan-build"
run(["cmake", "-S", scratch / "filament", "-B", filament_vk, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
     "-DFILAMENT_BUILD_FILAMAT=ON", "-DFILAMENT_BUILD_GLTFIO=ON", "-DFILAMENT_SUPPORTS_VULKAN=ON",
     "-DFILAMENT_BUILD_JAVA=OFF", "-DFILAMENT_SUPPORTS_EGL_ON_LINUX=OFF"], env)
run(["cmake", "--build", filament_vk, "--target", "gltf_viewer", "-j4"])
ppsspp = scratch / "ppsspp-build"
main = scratch / "ppsspp/SDL/SDLMain.cpp"
content = main.read_text()
main.write_text(content.replace("g_RefreshRate = displayMode.refresh_rate;",
    "g_RefreshRate = displayMode.refresh_rate > 0 ? displayMode.refresh_rate : 60;"))
loader = scratch / "ppsspp/Common/GPU/Vulkan/VulkanLoader.cpp"
content = loader.read_text()
if 'getenv("WINK_REFERENCE_CPU")' not in content:
    loader.write_text(content.replace("\t\tcase VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:",
        '\t\tcase VK_PHYSICAL_DEVICE_TYPE_CPU:\n\t\t\tif (getenv("WINK_REFERENCE_CPU")) anyGood = true;\n\t\t\tbreak;\n\t\tcase VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:'))
if "#include <cstdlib>" not in loader.read_text():
    loader.write_text(loader.read_text().replace("#include <cstring>", "#include <cstring>\n#include <cstdlib>"))
run(["cmake", "-S", scratch / "ppsspp", "-B", ppsspp, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
     "-DUSING_GLES2=ON", "-DUSING_EGL=OFF", "-DUSE_SYSTEM_FFMPEG=ON", "-DUNITTEST=OFF"])
run(["cmake", "--build", ppsspp, "--target", "PPSSPPSDL", "-j4"])
out = Path("/work/extended-tools")
temporary = out.with_name("extended-tools.partial")
if temporary.exists():
    shutil.rmtree(temporary)
temporary.mkdir()
shutil.copy2(filament / "samples/gltf_viewer", temporary / "gltf_viewer-gles")
shutil.copy2(filament_vk / "samples/gltf_viewer", temporary / "gltf_viewer-vulkan")
shutil.copytree(filament / "samples/assets/ibl", temporary / "filament-assets/ibl")
shutil.copy2(ppsspp / "PPSSPPSDL", temporary)
shutil.copytree(scratch / "ppsspp/assets", temporary / "assets")
shutil.copytree(filament / "samples/assets/fonts", temporary / "assets/fonts", dirs_exist_ok=True)
shutil.copytree(scratch / "pspautotests/demos", temporary / "psp-demos")
(temporary / "lib").mkdir()
for binary in (temporary / "gltf_viewer-gles", temporary / "gltf_viewer-vulkan", temporary / "PPSSPPSDL"):
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
