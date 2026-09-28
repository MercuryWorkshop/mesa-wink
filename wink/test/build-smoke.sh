#!/usr/bin/env bash
# Link wink/test/wink_smoke.c against a packaged variant exactly as a
# consumer would: plain emcc, the manifest's archives, required and
# recommended flags, and its --js-library list in order.
#
#   build-smoke.sh <variant> <packaged-dir> <emcc> <out-dir>
set -euo pipefail
variant=$1 pkg=$2 emcc=$3 out=$4
HERE=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
MESA=$(CDPATH='' cd -- "$HERE/../.." && pwd -P)
mkdir -p "$out"

flags=()
while IFS= read -r line; do flags+=("$line"); done < "$pkg/wink.rsp"

extra=()
if [ "$variant" = st ]; then
  # Mesa must not start a thread in the single-threaded build; the smoke
  # test's wrapper counts one as a failure instead of letting it fail quietly.
  extra+=(-Wl,--wrap=pthread_create)
fi

"$emcc" -O2 -I"$pkg/include" -I"$MESA/include" \
  "$HERE/wink_smoke.c" \
  "$pkg/lib/libwink.a" "$pkg/lib/libwink_webgpu.a" "$pkg/lib/libwink_tint.a" \
  "${flags[@]}" \
  --js-library="$HERE/library_wink_test.js" \
  ${extra[@]+"${extra[@]}"} \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createWinkSmoke \
  -sENVIRONMENT=web,worker --no-entry \
  -sEXPORTED_FUNCTIONS=_wink_smoke_main,_malloc,_free \
  -sINITIAL_MEMORY=64MB -sASSERTIONS=1 --profiling-funcs \
  -o "$out/wink-smoke.js"
cp "$HERE/smoke.html" "$out/index.html"
echo "$out/index.html"
