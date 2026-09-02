#!/usr/bin/env bash
# Run a GL program against the natively-built gex_wgpu.
#
#   run.sh glxgears
#   GALLIUM_DRIVER=softpipe run.sh glxgears     # the reference, same Mesa
set -u
HERE=$(CDPATH='' cd -- "$(dirname "$0")" && pwd)
INST="${GEX_MESA_OUT:-/tmp/gex-mesa-native}/inst"
PKG="${GEX_DAWN_PKG:-${GEX_DAWN_ROOT:-$HOME/.cache/gex-wgpu-dawn}/pkg}"
[ -d "$INST/lib/dri" ] || { echo "no build at $INST -- run build-mesa.sh" >&2; exit 1; }
export LD_LIBRARY_PATH="$INST/lib:$PKG/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBGL_DRIVERS_PATH="$INST/lib/dri"
# LIBGL_ALWAYS_SOFTWARE picks the software winsys, which is how this driver is
# reached: it presents through Mesa's sw_winsys, not through DRI.
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-gex_wgpu}"
exec "$@"
