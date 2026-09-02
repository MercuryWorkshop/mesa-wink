#!/usr/bin/env bash
# Build Mesa with the gex_wgpu driver for the desktop, against native Dawn.
#
# Step three of three; see README.md. Run from anywhere inside this checkout.
#
#   GEX_DAWN_PKG   the packaged Dawn (default $GEX_DAWN_ROOT/pkg)
#   GEX_MESA_OUT   the build directory (default /tmp/gex-mesa-native)
#
# softpipe is built alongside deliberately: it is the reference the same
# application can be run against by changing one environment variable, which
# is the only cheap way to tell "the driver is wrong" from "the application
# asked for something odd".
set -euo pipefail
HERE=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
MESA=$(CDPATH='' cd -- "$HERE/../../../../.." && pwd)
PKG="${GEX_DAWN_PKG:-${GEX_DAWN_ROOT:-$HOME/.cache/gex-wgpu-dawn}/pkg}"
OUT="${GEX_MESA_OUT:-/tmp/gex-mesa-native}"

[ -f "$MESA/meson.build" ] || { echo "$MESA does not look like a Mesa checkout" >&2; exit 1; }
[ -f "$PKG/lib/libtint.a" ] || {
    echo "no $PKG/lib/libtint.a -- run build-dawn.sh then package-dawn.sh" >&2
    exit 1
}

if [ ! -f "$OUT/build.ninja" ]; then
    meson setup "$OUT" "$MESA" \
        --buildtype=debugoptimized \
        -Dgallium-drivers=gex_wgpu,softpipe \
        -Dvulkan-drivers= \
        -Dglx=dri -Degl=enabled -Dgbm=disabled -Dplatforms=x11 \
        -Dllvm=disabled -Dshared-llvm=disabled \
        -Dgles1=disabled -Dgles2=enabled \
        -Dglvnd=disabled \
        -Dgex-emdawn-dir="$PKG" \
        -Dgex-tint-dir="$PKG" \
        -Dgex-dawn-native-dir="$PKG/lib" \
        -Dbuild-tests=false \
        --prefix="$OUT/inst"
fi
ninja -C "$OUT"
ninja -C "$OUT" install
echo
echo "built. run a GL program with:"
echo "  $HERE/run.sh <program>"
