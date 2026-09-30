#!/usr/bin/env bash
# Build Dawn natively: libwebgpu_dawn + Tint, for the desktop gex_wgpu driver.
#
# Step one of three; see README.md in this directory. Nothing outside this
# Mesa checkout is needed.
#
#   GEX_DAWN_ROOT   where to clone and build Dawn (default ~/.cache/gex-wgpu-dawn)
#   GEX_DAWN_SRC    an existing checkout at the pinned revision to build instead,
#                   such as the one mesa-wink's wink/build.sh fetched for Tint
#                   (~/.cache/mesa-wink/dawn/src-*): nothing is fetched into it
#                   that it already has
#
# The Dawn revision is pinned so that the webgpu.h the driver was written
# against and the Tint it links are one tree. Linux builds Dawn's Vulkan
# backend; macOS its Metal one.
set -euo pipefail
# Before any cd: BASH_SOURCE may be relative.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT="${GEX_DAWN_ROOT:-$HOME/.cache/gex-wgpu-dawn}"
SRC="${GEX_DAWN_SRC:-$ROOT/dawn}"
BUILD="$ROOT/build"
REF=cdda346c7ce9483630118c275e20ee128b736afb
JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if [ -n "${GEX_DAWN_SRC:-}" ]; then
  [ -f "$SRC/CMakeLists.txt" ] || { echo "[dawn] no Dawn checkout at $SRC" >&2; exit 1; }
  cd "$SRC"
  if git rev-parse HEAD >/dev/null 2>&1; then
    git rev-parse HEAD | grep -q "^$REF" || { echo "[dawn] $SRC is not at $REF" >&2; exit 1; }
  fi
else
  if [ ! -d "$SRC/.git" ]; then
    echo "[dawn] cloning"
    git clone --filter=blob:none https://dawn.googlesource.com/dawn "$SRC"
  fi
  cd "$SRC"
  git rev-parse HEAD | grep -q "^$REF" || { echo "[dawn] checkout $REF"; git fetch --filter=blob:none origin "$REF" || git fetch origin; git checkout -q "$REF"; }
fi
echo "[dawn] at $(git rev-parse --short HEAD 2>/dev/null || echo "$REF")"

if [ ! -f third_party/abseil-cpp/CMakeLists.txt ] || [ -z "${GEX_DAWN_SRC:-}" ]; then
  echo "[dawn] fetching dependencies"
  python3 tools/fetch_dawn_dependencies.py --use-test-deps 2>/dev/null \
    || python3 tools/fetch_dawn_dependencies.py
fi

# Metal's safe math still fuses a multiply and an add, which strict math,
# which the Vulkan driver asks for where a shader says not to, is not to.
# And a command buffer Metal fails (a GPU hang or fault) loses the device,
# where Dawn said nothing -- and the system ignored all the process's work
# after it. On Vulkan, which has no limit of the bytes a sample of the
# colour attachments takes, Dawn reported WebGPU's default of 32, too few
# for 8 RGBA8 attachments; and it zeroed a renderable 3D texture by
# rendering to a 2D view of it, which WebGPU has none of. Tint's SPIR-V
# reader takes the tier1 storage formats, as wink's wasm Tint does. And a
# framebuffer's clear values were an array too short for resolve targets,
# which wrote over its attachment count with more than four of them.
PATCHES="$HERE/../../../../../wink/patches"
for PATCH in "$PATCHES/dawn-metal-strict-math-contract.patch" "$PATCHES/dawn-metal-command-buffer-errors.patch" \
             "$PATCHES/dawn-vulkan-color-attachment-bytes.patch" \
             "$PATCHES/dawn-vulkan-clear-3d-texture.patch" \
             "$PATCHES/tint-spirv-reader-texel-formats.patch" \
             "$PATCHES/dawn-vulkan-framebuffer-clear-values.patch"; do
  if git -C "$SRC" apply --check "$PATCH" 2>/dev/null; then
    echo "[dawn] applying $(basename "$PATCH")"
    git -C "$SRC" apply "$PATCH"
  fi
done

case "$(uname -s)" in
  Darwin) BACKENDS=(-DDAWN_ENABLE_METAL=ON -DDAWN_ENABLE_VULKAN=OFF -DDAWN_USE_X11=OFF -DDAWN_USE_WAYLAND=OFF) ;;
  # Wayland surfaces as well as X11's, which Dawn leaves out by default.
  *)      BACKENDS=(-DDAWN_ENABLE_VULKAN=ON -DDAWN_USE_X11=ON -DDAWN_USE_WAYLAND=ON) ;;
esac

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
  -DDAWN_FETCH_DEPENDENCIES=OFF \
  "${BACKENDS[@]}" \
  -DDAWN_USE_GLFW=OFF \
  -DTINT_BUILD_SPV_READER=ON -DTINT_BUILD_WGSL_WRITER=ON \
  -DDAWN_ENABLE_OPENGLES=OFF -DDAWN_ENABLE_DESKTOP_GL=OFF \
  -DDAWN_ENABLE_NULL=ON \
  -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_TESTS=OFF \
  -DTINT_BUILD_TESTS=OFF -DTINT_BUILD_CMD_TOOLS=OFF \
  -DDAWN_BUILD_MONOLITHIC_LIBRARY=SHARED \
  -DBUILD_SHARED_LIBS=OFF

echo "[dawn] building"
cmake --build "$BUILD" --target webgpu_dawn -j "$JOBS"
echo "[dawn] done"
find "$BUILD" -name "libwebgpu_dawn*" | head -5
