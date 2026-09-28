#!/usr/bin/env bash
# Run groups of dEQP-VK against the native wink Vulkan driver with
# deqp-runner (cargo install deqp-runner --version 0.23.3).
#
#   run-cts.sh OUT GROUP... [-- deqp-runner options]
#
# GROUP is a mustpass list without .txt (api, texture, ...) or a directory
# of them (pipeline, image).
# WVK_CTS_ROOT is build-cts.sh's (default ~/.cache/mesa-wink/vk-cts), and
# WVK_MESA_BUILD the Mesa build with the driver (default build/wvk-native).
# OUT gets deqp-runner's results.csv and failures.csv. WVK_CTS_TIMEOUT is
# the per-test timeout in seconds (default 60).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MESA=$(cd "$HERE/../../../.." && pwd)
ROOT="${WVK_CTS_ROOT:-$HOME/.cache/mesa-wink/vk-cts}"
BUILD="${WVK_MESA_BUILD:-$MESA/build/wvk-native}"
OUT=$1; shift
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

MUSTPASS="$ROOT/VK-GL-CTS/external/vulkancts/mustpass/main/vk-default"
: > "$OUT/caselist.txt"
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  # A group is a list, or a directory of them (pipeline, image, ...).
  if [ -f "$MUSTPASS/$1.txt" ]; then
    cat "$MUSTPASS/$1.txt" >> "$OUT/caselist.txt"
  elif [ -d "$MUSTPASS/$1" ]; then
    find "$MUSTPASS/$1" -name '*.txt' -exec cat {} + >> "$OUT/caselist.txt"
  else
    echo "no group $1; there are: $(cd "$MUSTPASS" && ls | sed 's/\.txt$//' | tr '\n' ' ')" >&2
    exit 1
  fi
  shift
done
[ "${1:-}" = "--" ] && shift

ICD=$(ls "$BUILD"/src/wink/vulkan/wink_devenv_icd.*.json | head -1)
DEQP="$ROOT/build/external/vulkancts/modules/vulkan"
RUNNER=$(command -v deqp-runner || echo "$HOME/.cargo/bin/deqp-runner")
ENVS=(--env "VK_DRIVER_FILES=$ICD" --env "MESA_LOG_LEVEL=${MESA_LOG_LEVEL:-error}")
[ "$(uname)" = Darwin ] && ENVS+=(--env "DYLD_LIBRARY_PATH=/opt/homebrew/lib")

cd "$DEQP"
exec "$RUNNER" run --deqp ./deqp-vk --caselist "$OUT/caselist.txt" --output "$OUT" \
  --skips "$HERE/../ci/wvk-skips.txt" "${ENVS[@]}" --timeout "${WVK_CTS_TIMEOUT:-60}" "$@" \
  -- --deqp-log-images=disable --deqp-log-shader-sources=disable
