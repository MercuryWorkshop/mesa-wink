#!/usr/bin/env bash
# Build dEQP-VK, the Vulkan conformance suite, for running against the
# native wink Vulkan driver: macOS or Linux, against the system's Vulkan
# loader.
#
#   WVK_CTS_ROOT   where to clone and build (default ~/.cache/mesa-wink/vk-cts)
#   WVK_CTS_REF    VK-GL-CTS tag (default vulkan-cts-1.4.6.2)
#
# Then, from the build's external/vulkancts/modules/vulkan directory:
#   VK_DRIVER_FILES=<mesa build>/src/wink/vulkan/wink_devenv_icd.<arch>.json \
#     ./deqp-vk --deqp-case='dEQP-VK.api.smoke.*'
set -euo pipefail
ROOT="${WVK_CTS_ROOT:-$HOME/.cache/mesa-wink/vk-cts}"
REF="${WVK_CTS_REF:-vulkan-cts-1.4.6.2}"
SRC="$ROOT/VK-GL-CTS"
BUILD="$ROOT/build"
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$REF" https://github.com/KhronosGroup/VK-GL-CTS.git "$SRC"
fi
( cd "$SRC" && python3 external/fetch_sources.py )

cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSELECTED_BUILD_TARGETS=deqp-vk
ninja -C "$BUILD" -j "$JOBS" deqp-vk
echo "built: $BUILD/external/vulkancts/modules/vulkan/deqp-vk"
