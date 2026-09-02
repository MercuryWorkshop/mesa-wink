#!/usr/bin/env bash
# Lay a Dawn build out the way this driver's meson expects.
#
# Step two of three; see README.md. Three things Dawn's cmake does not
# produce in the right shape:
#
#   libtint.a   one archive. Dawn builds Tint as several dozen small ones plus
#               Abseil and SPIRV-Tools; the wasm build already solved this by
#               concatenating them, and this is the same list with the native
#               ar.
#   include/    Tint's headers, rooted so "src/tint/..." and "absl/..."
#               resolve.
#   webgpu.h    Dawn's own generated header, installed where -Dgex-emdawn-dir
#               looks for it. The wasm build takes this from the emdawnwebgpu
#               package instead; the two were compared enum by enum at this
#               revision -- 443 shared values, none differing -- and Dawn's is
#               a superset, so a native build needs nothing from that package.
set -euo pipefail
ROOT="${GEX_DAWN_ROOT:-$HOME/.cache/gex-wgpu-dawn}"
SRC="$ROOT/dawn"
BUILD="$ROOT/build"
OUT="${GEX_DAWN_PKG:-$ROOT/pkg}"

[ -d "$BUILD" ] || { echo "no Dawn build at $BUILD -- run build-dawn.sh first" >&2; exit 1; }
mkdir -p "$OUT/include" "$OUT/lib"

mapfile -t ARCHIVES < <(
  find "$BUILD/src/tint" -name 'libtint_*.a'
  find "$BUILD" -name 'libabsl_*.a'
  find "$BUILD" -path '*spirv-tools*' -name 'libSPIRV-Tools*.a'
)
echo "[pkg] ${#ARCHIVES[@]} archives -> $OUT/lib/libtint.a"
rm -f "$OUT/lib/libtint.a"
{
  echo "create $OUT/lib/libtint.a"
  for a in "${ARCHIVES[@]}"; do echo "addlib $a"; done
  echo save
  echo end
} | ar -M
ar t "$OUT/lib/libtint.a" >/dev/null
# find_library(dirs: <pkg>) looks in the directory itself, not in lib/.
ln -sf lib/libtint.a "$OUT/libtint.a"

echo "[pkg] Tint and Abseil headers"
rm -rf "$OUT/include"; mkdir -p "$OUT/include"
( cd "$SRC" && find src \( -name '*.h' -o -name '*.inc' \) | tar -cf - -T - ) | tar -xf - -C "$OUT/include"
( cd "$SRC/third_party/abseil-cpp" && find absl \( -name '*.h' -o -name '*.inc' \) | tar -cf - -T - ) | tar -xf - -C "$OUT/include"

echo "[pkg] webgpu.h"
mkdir -p "$OUT/webgpu/include/webgpu" "$OUT/webgpu_cpp/include"
HEADER=$(find "$BUILD" -path '*include/dawn/webgpu.h' | head -1)
[ -n "$HEADER" ] || { echo "no generated webgpu.h under $BUILD" >&2; exit 1; }
cp "$HEADER" "$OUT/webgpu/include/webgpu/webgpu.h"
# meson's include_directories() insists the directory exists; the driver's C++
# never includes the C++ bindings, so an empty one is enough.
CPP=$(find "$BUILD" -path '*include/webgpu/webgpu_cpp.h' | head -1)
[ -n "$CPP" ] && { mkdir -p "$OUT/webgpu_cpp/include/webgpu"; cp "$CPP" "$OUT/webgpu_cpp/include/webgpu/"; }

echo "[pkg] libwebgpu_dawn"
for f in $(find "$BUILD" -name 'libwebgpu_dawn.so*'); do cp -a "$f" "$OUT/lib/" 2>/dev/null || true; done

echo "[pkg] done -> $OUT"
ls -la "$OUT/lib" | sed 's/^/  /'
