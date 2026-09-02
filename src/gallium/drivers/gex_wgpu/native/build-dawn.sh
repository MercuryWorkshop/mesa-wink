#!/usr/bin/env bash
# Build Dawn natively: libwebgpu_dawn + Tint, for the desktop gex_wgpu driver.
#
# Step one of three; see README.md in this directory. Nothing outside this
# Mesa checkout is needed.
#
#   GEX_DAWN_ROOT   where to clone and build Dawn (default ~/.cache/gex-wgpu-dawn)
#
# The Dawn revision is pinned so that the webgpu.h the driver was written
# against and the Tint it links are one tree.
set -euo pipefail
ROOT="${GEX_DAWN_ROOT:-$HOME/.cache/gex-wgpu-dawn}"
SRC="$ROOT/dawn"
BUILD="$ROOT/build"
REF=b465e23d19a5e27402d80191c0f8de1fb3b385b5

if [ ! -d "$SRC/.git" ]; then
  echo "[dawn] cloning"
  git clone --filter=blob:none https://dawn.googlesource.com/dawn "$SRC"
fi
cd "$SRC"
git rev-parse HEAD | grep -q "^$REF" || { echo "[dawn] checkout $REF"; git fetch --filter=blob:none origin "$REF" || git fetch origin; git checkout -q "$REF"; }
echo "[dawn] at $(git rev-parse --short HEAD)"

echo "[dawn] fetching dependencies"
python3 tools/fetch_dawn_dependencies.py --use-test-deps 2>/dev/null \
  || python3 tools/fetch_dawn_dependencies.py

echo "[dawn] configuring"
# Clang, not the system GCC. This Dawn revision does not compile with GCC --
# Tint's intrinsic tables trip -Winvalid-constexpr, which is a GCC diagnostic
# Clang does not raise -- and the wasm build never noticed because emcc is
# Clang underneath.
cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-Wno-c2y-extensions -Wno-deprecated-pragma -Wno-lifetime-safety-intra-tu-suggestions -Wno-unknown-warning-option" \
  -DCMAKE_C_FLAGS="-Wno-c2y-extensions -Wno-unknown-warning-option" \
  -DCMAKE_BUILD_TYPE=Release \
  -DDAWN_FETCH_DEPENDENCIES=ON \
  -DDAWN_ENABLE_VULKAN=ON \
  -DDAWN_ENABLE_OPENGLES=OFF -DDAWN_ENABLE_DESKTOP_GL=OFF \
  -DDAWN_ENABLE_NULL=ON \
  -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_TESTS=OFF \
  -DTINT_BUILD_TESTS=OFF -DTINT_BUILD_CMD_TOOLS=OFF \
  -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED \
  -DBUILD_SHARED_LIBS=OFF

echo "[dawn] building"
cmake --build "$BUILD" --target webgpu_dawn -j "$(nproc)"
echo "[dawn] done"
find "$BUILD" -name "libwebgpu_dawn*" -o -name "libtint*.a" | head -20
