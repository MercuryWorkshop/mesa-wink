#!/usr/bin/env bash
# Run the GL ES conformance suites against the natively built gex_wgpu with
# deqp-runner, comparing with the expectations in ../ci.
#
#   run-deqp.sh [native|cpu]     compression path: native ASTC/ETC2 (default),
#                                or hidden from the driver so st/mesa decodes
#   GEX_CTS_ROOT                 as build-cts.sh
#   GEX_DEQP_OUT                 results (default /tmp/gex-wgpu-deqp-<mode>)
#   GEX_DEQP_JOBS, GEX_DEQP_FRACTION   parallelism, and run 1 case in N
#
# dEQP's default Linux target opens its EGL display on X11, so DISPLAY must be
# set; a headless machine can use `xvfb-run -a run-deqp.sh`. The environment is
# run.sh's -- the software winsys, then GALLIUM_DRIVER=gex_wgpu -- and softpipe
# is one variable away as the reference: GALLIUM_DRIVER=softpipe run-deqp.sh.
#
# Not yet run: written on a machine without the native flow. Expect to adjust.
set -euo pipefail
HERE=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
MODE="${1:-native}"
BUILD="${GEX_CTS_ROOT:-$HOME/.cache/gex-wgpu-cts}/build-gles"
OUT="${GEX_DEQP_OUT:-/tmp/gex-wgpu-deqp-$MODE}"
JOBS="${GEX_DEQP_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
CI="$HERE/../ci"
[ -x "$BUILD/modules/gles3/deqp-gles3" ] || { echo "no CTS at $BUILD -- run build-cts.sh" >&2; exit 1; }
[ -n "${DISPLAY:-}" ] || { echo "DISPLAY is not set (try xvfb-run -a $0)" >&2; exit 1; }

case "$MODE" in
  native) ;;
  cpu) export GEX_WGPU_NO_ASTC=1 GEX_WGPU_NO_ETC2=1 ;;
  *) echo "usage: $0 [native|cpu]" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
ARGS='"--deqp-surface-width=256", "--deqp-surface-height=256", "--deqp-surface-type=pbuffer", "--deqp-gl-config-name=rgba8888d24s8ms0", "--deqp-visibility=hidden"'
cat > "$OUT/suite.toml" <<TOML
[[deqp]]
deqp = "$BUILD/modules/gles2/deqp-gles2"
caselists = ["$BUILD/mustpass/gles2-main.txt"]
deqp_args = [$ARGS]
version_check = "OpenGL ES 3.2"
renderer_check = "gex WebGPU"

[[deqp]]
deqp = "$BUILD/modules/gles3/deqp-gles3"
caselists = ["$BUILD/mustpass/gles3-main.txt"]
deqp_args = [$ARGS]

[[deqp]]
deqp = "$BUILD/modules/gles31/deqp-gles31"
caselists = ["$BUILD/mustpass/gles31-main.txt"]
deqp_args = [$ARGS]

[[deqp]]
deqp = "$BUILD/external/openglcts/modules/glcts"
caselists = [
  "$BUILD/mustpass/gles2-khr-main.txt",
  "$BUILD/mustpass/gles3-khr-main.txt",
  "$BUILD/mustpass/gles31-khr-main.txt",
  "$BUILD/mustpass/gles32-khr-main.txt",
]
deqp_args = [$ARGS]
TOML

exec "$HERE/run.sh" deqp-runner suite \
  --suite "$OUT/suite.toml" \
  --output "$OUT" \
  --baseline "$CI/gex_wgpu-fails.txt" \
  --flakes "$CI/gex_wgpu-flakes.txt" \
  --skips "$CI/gex_wgpu-skips.txt" \
  --jobs "$JOBS" \
  ${GEX_DEQP_FRACTION:+--fraction "$GEX_DEQP_FRACTION"}
