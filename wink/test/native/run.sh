#!/usr/bin/env bash
# libwink as SDL's GL on macOS: build sdl_gl.c against the wink fork of SDL
# and run it.
#
#   wink/test/native/run.sh LIBWINK SDL_BUILD [OUT]
#
# LIBWINK is libwink.dylib from a native build (-Dgex-dawn-native-dir), and
# SDL_BUILD the fork's CMake build directory, configured with
# -DSDL_COCOA_WINK_INCLUDE_DIR=<mesa>/src/gallium/targets/wink-wasm.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
wink=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
sdl=$(cd "$2" && pwd)
out=${3:-$(mktemp -d)}
mkdir -p "$out"
sdl_src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$sdl/CMakeCache.txt")

cc -O1 -g -o "$out/sdl_gl" "$here/sdl_gl.c" -I"$sdl_src/include" -I"$sdl/include" \
   -L"$sdl" -lSDL3 -Wl,-rpath,"$sdl"
SDL_VIDEO_COCOA_WINK="$wink" "$out/sdl_gl" 2> >(grep -v '^gex-wgpu: ' >&2)
