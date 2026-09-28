#!/usr/bin/env bash
# Build the GL ES conformance suites -- dEQP-GLES2/3/31 and the Khronos
# KHR-GLES* sets in glcts -- the way Mesa CI does, for run-deqp.sh to point at
# the natively built gex_wgpu. Linux (the default dEQP target: X11 and EGL).
#
#   GEX_CTS_ROOT   where to clone and build (default ~/.cache/gex-wgpu-cts)
#   GEX_CTS_REF    VK-GL-CTS tag (default opengl-es-cts-3.2.14.1, Mesa CI's pin)
#
# deqp-runner is installed with cargo if it is not on PATH: 0.23.3, Mesa CI's.
#
# Not yet run: written on a machine without the native flow. Expect to adjust.
set -euo pipefail
ROOT="${GEX_CTS_ROOT:-$HOME/.cache/gex-wgpu-cts}"
REF="${GEX_CTS_REF:-opengl-es-cts-3.2.14.1}"
SRC="$ROOT/VK-GL-CTS"
BUILD="$ROOT/build-gles"
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/KhronosGroup/VK-GL-CTS.git "$SRC"
fi
( cd "$SRC" && python3 external/fetch_sources.py )

cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DDEQP_TARGET=default \
  -DSELECTED_BUILD_TARGETS="deqp-gles2;deqp-gles3;deqp-gles31;glcts"
ninja -C "$BUILD" -j "$JOBS" deqp-gles2 deqp-gles3 deqp-gles31 glcts

# The case lists Mesa CI runs: AOSP's for dEQP, Khronos's for KHR-GLES*.
M="$SRC/external/openglcts/data/gl_cts/data/mustpass/gles"
mkdir -p "$BUILD/mustpass"
cp "$M"/aosp_mustpass/main/*.txt "$BUILD/mustpass/"
cp "$M"/khronos_mustpass/main/*-main.txt "$BUILD/mustpass/"

if ! command -v deqp-runner >/dev/null 2>&1; then
  cargo install deqp-runner --version 0.23.3 --locked
fi
echo "built: $BUILD"
