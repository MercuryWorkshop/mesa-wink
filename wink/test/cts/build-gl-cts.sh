#!/usr/bin/env bash
# Build the GL and GL ES conformance suites for macOS, against libwink: glcts
# (KHR-GL*, KHR-GLES*) and deqp-gles2/3/31, from VK-GL-CTS at Mesa CI's pins
# with Mesa CI's backports, plus a context factory for wink
# (deqp-osx-wink-context-factory.patch; --deqp-gl-context-type=wink).
#
#   WINK_GL_CTS_ROOT   where to clone and build (default ~/.cache/mesa-wink/gl-cts)
#
# opengl-cts-4.6.8.1 and opengl-es-cts-3.2.14.1 are one commit, so one build
# serves both. The backports are the union of build-deqp.sh's GL and GLES
# lists. Its "Don't build Vulkan utilities for GL builds" patch is left out:
# it removes the macOS platform's Vulkan sources but not their uses, and it
# only saves build time. run-gl-cts.sh runs the result.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MESA=$(cd "$HERE/../../.." && pwd)
ROOT="${WINK_GL_CTS_ROOT:-$HOME/.cache/mesa-wink/gl-cts}"
REF=opengl-cts-4.6.8.1
SRC="$ROOT/VK-GL-CTS"
BUILD="$ROOT/build"
JOBS=$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN)
BACKPORTS=(
  b5ed8718f19492781f8e9be3eb9d3346e961efa9
  6368ee8503dd9ca46eabfa2df293075d9034a214
  e39758bc9262f8593cf0fddf1364179c1fedf756
  fe92180156795a641b684722a199bdb64c69bf2a
  bf307df9e5c6c27499c2c799409d17554d39cafc
  5d392ccf4fa09d38d06ca66875a2aaac4fde2b11
)

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/KhronosGroup/VK-GL-CTS.git "$SRC"
  cd "$SRC"
  git config user.email "mesa-wink@example.com"
  git config user.name "mesa-wink"
  for c in "${BACKPORTS[@]}"; do
    curl -fsSL "https://github.com/KhronosGroup/VK-GL-CTS/commit/$c.patch" | git am -q -
  done
  git am -q "$HERE/deqp-osx-wink-context-factory.patch"
  python3 external/fetch_sources.py
fi

cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DDEQP_WINK_INCLUDE_DIR="$MESA/src/gallium/targets/wink-wasm" \
  -DSELECTED_BUILD_TARGETS="deqp-gles2;deqp-gles3;deqp-gles31;glcts"
ninja -C "$BUILD" -j "$JOBS" deqp-gles2 deqp-gles3 deqp-gles31 glcts

# The case lists Mesa CI runs: AOSP's for dEQP-GLES*, Khronos's for KHR-*.
M="$SRC/external/openglcts/data/gl_cts/data/mustpass"
mkdir -p "$BUILD/mustpass"
cp "$M"/gles/aosp_mustpass/main/*.txt "$M"/gles/khronos_mustpass/main/*-main.txt \
   "$M"/gl/khronos_mustpass/main/*-main.txt "$M"/gl/khronos_mustpass_single/main/*-single.txt \
   "$BUILD/mustpass/"
echo "built: $BUILD"
