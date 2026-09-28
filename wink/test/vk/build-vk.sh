#!/usr/bin/env bash
# Link the wink Vulkan driver's native tests (src/wink/vulkan/test) against
# a packaged variant, with a generated shim in place of the Vulkan loader,
# for run-browser.mjs:
#
#   build-vk.sh <variant> <packaged-dir> <emcc> <out-dir>
#   node wink/test/run-browser.mjs <out-dir> --page='vk.html?test=wvk_draw'
set -euo pipefail
variant=$1 pkg=$2 emcc=$3 out=$4
HERE=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
MESA=$(CDPATH='' cd -- "$HERE/../../.." && pwd -P)
TESTS="$MESA/src/wink/vulkan/test"
mkdir -p "$out/shaders"

python3 "$HERE/gen-shim.py" "$MESA/include/vulkan/vulkan_core.h" "$out/shim.c"
for s in "$TESTS"/shaders/*; do
  name=$(basename "$s" | tr . _)
  glslangValidator -V --target-env vulkan1.2 --quiet --vn "spv_$name" -o "$out/shaders/$name.h" "$s"
done

flags=()
while IFS= read -r line; do flags+=("$line"); done < "$pkg/wink.rsp"
extra=()
[ "$variant" = mt ] && extra+=(-sPTHREAD_POOL_SIZE=2)

for t in wvk_smoke wvk_draw; do
  "$emcc" -O2 -I"$pkg/include" -I"$MESA/include" -I"$out/shaders" -Dmain=wvk_test_main \
    "$TESTS/$t.c" "$HERE/main.c" "$HERE/publish.c" "$out/shim.c" \
    "$pkg/lib/libwink.a" "$pkg/lib/libwink_webgpu.a" "$pkg/lib/libwink_tint.a" \
    "${flags[@]}" --js-library="$HERE/library_vk_test.js" ${extra[@]+"${extra[@]}"} \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createVkTest \
    -sENVIRONMENT=web,worker --no-entry \
    -sEXPORTED_FUNCTIONS=_wvk_browser_main,_malloc,_free \
    -sINITIAL_MEMORY=128MB -sALLOW_MEMORY_GROWTH=1 -sASSERTIONS=1 --profiling-funcs \
    -o "$out/$t.js"
done
cp "$HERE/vk.html" "$out/vk.html"
echo "$out/vk.html"
