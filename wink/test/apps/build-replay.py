#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build pinned upstream API retracers with Wink platform adapters."""
import argparse
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
MESA = HERE.parents[2] if len(HERE.parents) > 2 else Path("/work")


def run(argv, **kw):
    print("+", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), check=True, **kw)


def checkout(root, name, pin, exclude_submodules=()):
    source = root / "deps" / name
    if not source.exists():
        source.mkdir(parents=True)
        run(["git", "-C", source, "init"])
        run(["git", "-C", source, "remote", "add", "origin", pin["url"]])
        run(["git", "-C", source, "fetch", "--depth=1", "origin", pin["commit"]])
        run(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
    actual = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if actual != pin["commit"]:
        raise RuntimeError(f"{source}: wrong revision {actual}; expected {pin['commit']}")
    command = ["git", "-C", source, "submodule", "update", "--init", "--depth=1", "--recursive"]
    if exclude_submodules:
        entries = subprocess.check_output(["git", "-C", source, "config", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"], text=True)
        paths = [line.split(maxsplit=1)[1] for line in entries.splitlines()]
        command += ["--", *(path for path in paths if path not in exclude_submodules)]
    run(command)
    return source


def patch(source, name, before, after):
    path = source / name
    text = path.read_text()
    if after in text:
        return
    if text.count(before) != 1:
        raise RuntimeError(f"patch anchor changed: {path}")
    path.write_text(text.replace(before, after))


def build_gl(a, source):
    from glreplay import adapt_wsi
    adapt_wsi(source)
    patch(source, "retrace/glstate_images.cpp", '#include "glstate.hpp"',
          '#include "glstate.hpp"\n#ifdef WINK_REPLAY\n#include "glretrace.hpp"\n#endif')
    patch(source, "retrace/glstate_images.cpp", "getDrawableBounds(GLint *width, GLint *height) {",
          "getDrawableBounds(GLint *width, GLint *height) {\n#ifdef WINK_REPLAY\n"
          "    auto *context = glretrace::getCurrentContext();\n"
          "    if (!context || !context->drawable) return false;\n"
          "    *width = context->drawable->width; *height = context->drawable->height;\n"
          "    return *width > 0 && *height > 0;\n#endif")
    path = source / "retrace/CMakeLists.txt"
    include = '\ninclude("${WINK_APPS}/replay/apitrace.cmake")\n'
    if include not in path.read_text():
        path.write_text(path.read_text() + include)
    # Cocoa surface queries apply to Apple's GL, not Wink's offscreen surfaces.
    for name in ("glretrace_main.cpp", "glstate.cpp", "glstate_images.cpp"):
        path = source / "retrace" / name
        content = path.read_text()
        content = content.replace("#if defined(__APPLE__)\n", "#if defined(__APPLE__) && !defined(WINK_REPLAY)\n")
        content = content.replace("#ifdef __APPLE__\n", "#if defined(__APPLE__) && !defined(WINK_REPLAY)\n")
        content = content.replace("#elif defined(__APPLE__)\n", "#elif defined(__APPLE__) && !defined(WINK_REPLAY)\n")
        path.write_text(content)
    build = a.out / ("gl-" + a.target)
    config = ["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
              "-DENABLE_GUI=OFF", "-DENABLE_X11=OFF", "-DENABLE_EGL=OFF", "-DBUILD_TESTING=OFF",
              "-DCMAKE_CXX_FLAGS=-DWINK_REPLAY=1", f"-DWINK_APPS={HERE}",
              f"-DWINK_INCLUDE={MESA / 'src/gallium/targets/wink-wasm'}"]
    if a.target == "native":
        config += [f"-DWINK_LIBRARY={a.library}", "-DCMAKE_PREFIX_PATH=/opt/homebrew"]
    else:
        variant = a.target
        package = MESA / "build/wink/out" / variant
        # Upstream disables exceptions by default; Wasm EH is required by Mesa.
        flags = "-DWINK_REPLAY=1 -fwasm-exceptions -sSUPPORT_LONGJMP=wasm" + (" -pthread" if variant == "mt" else "")
        path = source / "CMakeLists.txt"
        content = path.read_text().replace("-fno-exceptions", "-fexceptions")
        path.write_text(content)
        config = [a.emsdk / "upstream/emscripten/emcmake", *config]
        config += [f"-DWINK_PACKAGE={package}", f"-DCMAKE_C_FLAGS={flags}", f"-DCMAKE_CXX_FLAGS={flags}",
                   "-DCMAKE_DISABLE_FIND_PACKAGE_PkgConfig=TRUE", "-DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=TRUE",
                   "-DCMAKE_DISABLE_FIND_PACKAGE_PNG=TRUE"]
    run(config)
    run(["cmake", "--build", build, "--target", "wink-glretrace", "-j", a.jobs])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("api", choices=["gl", "vulkan"])
    p.add_argument("--target", choices=["native", "st", "mt"], default="native")
    p.add_argument("--out", type=Path, default=MESA / "build/wink-apps")
    p.add_argument("--library", type=Path, default=MESA / "build/wink-mac/src/gallium/targets/wink-wasm/libwink.dylib")
    p.add_argument("--emsdk", type=Path, default=Path.home() / ".cache/mesa-wink/emsdk")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    pins = json.loads((HERE / "pins.json").read_text())
    name = "apitrace" if a.api == "gl" else "gfxreconstruct"
    source = checkout(a.out, name + ("-native" if a.api == "vulkan" and a.target == "native" else ""), pins[name])
    if a.api == "gl":
        build_gl(a, source)
    else:
        from gfxreplay import build_vulkan
        build_vulkan(a, source)


if __name__ == "__main__":
    main()
