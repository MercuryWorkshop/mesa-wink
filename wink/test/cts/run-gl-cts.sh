#!/usr/bin/env bash
# Run the GL or GL ES conformance suites against libwink on macOS, with
# deqp-runner, the way Mesa CI runs llvmpipe's: a 256x256 RGBA8 D24S8
# offscreen surface.
#
#   run-gl-cts.sh gles|gl OUT [-- deqp-runner options, such as --jobs 4 or --fraction 50]
#
#   gles   dEQP-GLES2/3/31 and KHR-GLES2/3/31/32
#   gl     KHR-GL46 and KHR-Single-GL46
#
# WINK_GL_CTS_ROOT is build-gl-cts.sh's, and WINK_LIBRARY the libwink.dylib
# (default: build/wink-mac's). The expectations are
# ../../../src/gallium/drivers/gex_wgpu/ci/gex_wgpu-{fails,flakes,skips}.txt.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MESA=$(cd "$HERE/../../.." && pwd)
BUILD="${WINK_GL_CTS_ROOT:-$HOME/.cache/mesa-wink/gl-cts}/build"
WINK="${WINK_LIBRARY:-$MESA/build/wink-mac/src/gallium/targets/wink-wasm/libwink.dylib}"
CI="$MESA/src/gallium/drivers/gex_wgpu/ci"
SUITE=$1
OUT=$2
shift 2
[ "${1:-}" = "--" ] && shift
[ -x "$BUILD/external/openglcts/modules/glcts" ] || { echo "no CTS at $BUILD -- run build-gl-cts.sh" >&2; exit 1; }
[ -f "$WINK" ] || { echo "no libwink at $WINK" >&2; exit 1; }
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

ARGS='"--deqp-surface-width=256", "--deqp-surface-height=256", "--deqp-surface-type=pbuffer",
  "--deqp-gl-config-name=rgba8888d24s8ms0", "--deqp-visibility=hidden", "--deqp-gl-context-type=wink"'
M="$BUILD/mustpass"
case "$SUITE" in
gles)
  cat > "$OUT/suite.toml" <<TOML
[[deqp]]
deqp = "$BUILD/modules/gles2/deqp-gles2"
caselists = ["$M/gles2-main.txt"]
deqp_args = [$ARGS]
version_check = "OpenGL ES 3.2"
renderer_check = "gex WebGPU"

[[deqp]]
deqp = "$BUILD/modules/gles3/deqp-gles3"
caselists = ["$M/gles3-main.txt"]
deqp_args = [$ARGS]

[[deqp]]
deqp = "$BUILD/modules/gles31/deqp-gles31"
caselists = ["$M/gles31-main.txt"]
deqp_args = [$ARGS]

[[deqp]]
deqp = "$BUILD/external/openglcts/modules/glcts"
caselists = ["$M/gles2-khr-main.txt", "$M/gles3-khr-main.txt", "$M/gles31-khr-main.txt", "$M/gles32-khr-main.txt"]
deqp_args = [$ARGS]
TOML
  ;;
gl)
  # KHR-GL46 includes the earlier versions' tests, so it is the only list.
  cat > "$OUT/suite.toml" <<TOML
[[deqp]]
deqp = "$BUILD/external/openglcts/modules/glcts"
caselists = ["$M/gl46-main.txt", "$M/gl46-khr-single.txt"]
deqp_args = [$ARGS]
TOML
  ;;
*)
  echo "usage: $0 gles|gl OUT [-- deqp-runner options]" >&2
  exit 2
  ;;
esac

RUNNER=$(command -v deqp-runner || echo "$HOME/.cargo/bin/deqp-runner")
exec "$RUNNER" suite --suite "$OUT/suite.toml" --output "$OUT" \
  --baseline "$CI/gex_wgpu-fails.txt" --flakes "$CI/gex_wgpu-flakes.txt" --skips "$CI/gex_wgpu-skips.txt" \
  --env "DEQP_WINK_LIBRARY=$WINK" "$@"
