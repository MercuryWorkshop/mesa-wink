#!/bin/sh
# Builds the wink Vulkan driver's native tests into OUT (default: .):
# the shaders to SPIR-V headers with glslangValidator, then the programs
# against the Vulkan loader (and SDL3, for the window test).
#
#   src/wink/vulkan/test/build.sh [OUT]
#   VK_DRIVER_FILES=<build>/src/wink/vulkan/wink_devenv_icd.<arch>.json OUT/wvk_draw
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-.}
mkdir -p "$OUT/shaders"
for s in "$HERE"/shaders/*; do
  name=$(basename "$s" | tr . _)
  glslangValidator -V --target-env vulkan1.2 --quiet --vn "spv_$name" -o "$OUT/shaders/$name.h" "$s"
done
CFLAGS="-O1 -g -Wall $(pkg-config --cflags vulkan)"
LIBS=$(pkg-config --libs vulkan)
cc $CFLAGS -o "$OUT/wvk_smoke" "$HERE/wvk_smoke.c" $LIBS
cc $CFLAGS -I"$OUT/shaders" -o "$OUT/wvk_draw" "$HERE/wvk_draw.c" $LIBS
if pkg-config --exists sdl3; then
  cc $CFLAGS $(pkg-config --cflags sdl3) -o "$OUT/wvk_window" "$HERE/wvk_window.c" \
     $(pkg-config --libs sdl3) $LIBS $( [ "$(uname)" = Darwin ] && echo -lobjc )
fi
