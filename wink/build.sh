#!/usr/bin/env bash
# Build Mesa's GL stack on the gex_wgpu WebGPU driver as static libraries for
# a .NET 10 browser-wasm app: Emscripten 3.1.56, mt (-pthread) and st.
#
# Runs on macOS (Homebrew) and Linux (Arch is the tested one). Everything the
# host would not already have -- emsdk, Tint, emdawnwebgpu, Python modules --
# is fetched into a cache; see README.md in this directory.
#
#   wink/build.sh [--variant=st|mt|both] [options]
#
# The output, per variant, is build/wink/out/<variant>/: libwink.a (one
# relocatable object: st/mesa, gallium, glsl, nir, gex_wgpu and the wink API),
# libwink_webgpu.a, libwink_tint.a, the --js-library files, headers, and a
# manifest naming the link flags a consumer must use.
set -euo pipefail

WINK_DIR=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd -P)
MESA_DIR=$(CDPATH='' cd -- "$WINK_DIR/.." && pwd -P)
[ -f "$MESA_DIR/meson.build" ] || { echo "$MESA_DIR is not a Mesa checkout" >&2; exit 1; }

. "$WINK_DIR/lib/common.sh"
. "$WINK_DIR/lib/deps.sh"
. "$WINK_DIR/lib/venv.sh"
. "$WINK_DIR/lib/emsdk.sh"
. "$WINK_DIR/lib/emdawn.sh"
. "$WINK_DIR/lib/tint.sh"
. "$WINK_DIR/lib/mesa.sh"
. "$WINK_DIR/lib/package.sh"

usage() {
  cat <<'EOF'
usage: wink/build.sh [options]

  --variant=st|mt|both     thread variant(s) to build (default: both)
  --buildtype=TYPE         meson buildtype: release (default) or debugoptimized
  --jobs=N                 parallel jobs (default: CPU count)
  --cache=DIR              downloads, emsdk, Dawn, Tint, venv
                           (default: ${XDG_CACHE_HOME:-~/.cache}/mesa-wink)
  --build-dir=DIR          meson build trees and tool wrappers (default: build/wink)
  --out=DIR                packaged output (default: <build-dir>/out)
  --emsdk=DIR              use this emsdk (upstream or .NET's layout) instead of
                           bootstrapping 3.1.56; $EMSDK works too
  --install-deps           install missing host tools (brew / pacman)
  --until=STAGE            stop after deps|venv|emsdk|emdawn|tint|mesa|package
  --wipe                   reconfigure the meson build trees from scratch
  --check-against=A:B      report symbol collisions with these consumer archives
  --test                   link wink/test/wink_smoke.c against each packaged variant
                           with plain emcc and run it in headless Chrome
  -v, --verbose            echo commands
  -h, --help

environment:
  EMSDK                    same as --emsdk
  WINK_ALLOW_ANY_EMSDK=1   accept an emcc that is not 3.1.56
  WINK_TINT_DIR_ST/_MT     prebuilt Tint (libtint.a + include/), e.g.
                           ikvm-wasm-build's tools/native-deps/tint-out-<variant>
  WINK_EMDAWN_DIR          an already patched emdawnwebgpu_pkg
  WINK_SPIRV_TOOLS_REV=    (empty) keep Dawn's own SPIRV-Tools in Tint
EOF
}

WINK_VARIANTS="st mt"
WINK_BUILDTYPE=release
WINK_JOBS=${JOBS:-$(ncpu)}
WINK_CACHE=${WINK_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/mesa-wink}
WINK_BUILD_ROOT="$MESA_DIR/build/wink"
WINK_OUT=
WINK_EMSDK=${EMSDK:-}
WINK_INSTALL_DEPS=0
WINK_UNTIL=
WINK_WIPE=0
WINK_CHECK_AGAINST=
WINK_TEST=0
WINK_VERBOSE=0

for arg in "$@"; do
  case $arg in
    --variant=both) WINK_VARIANTS="st mt" ;;
    --variant=st|--variant=mt) WINK_VARIANTS=${arg#*=} ;;
    --buildtype=*) WINK_BUILDTYPE=${arg#*=} ;;
    --jobs=*) WINK_JOBS=${arg#*=} ;;
    --cache=*) WINK_CACHE=${arg#*=} ;;
    --build-dir=*) WINK_BUILD_ROOT=${arg#*=} ;;
    --out=*) WINK_OUT=${arg#*=} ;;
    --emsdk=*) WINK_EMSDK=${arg#*=} ;;
    --install-deps) WINK_INSTALL_DEPS=1 ;;
    --until=*) WINK_UNTIL=${arg#*=} ;;
    --wipe) WINK_WIPE=1 ;;
    --check-against=*) WINK_CHECK_AGAINST=${arg#*=} ;;
    --test) WINK_TEST=1 ;;
    -v|--verbose) WINK_VERBOSE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $arg" ;;
  esac
done

mkdir -p "$WINK_CACHE" "$WINK_BUILD_ROOT"
WINK_CACHE=$(absdir "$WINK_CACHE")
WINK_BUILD_ROOT=$(absdir "$WINK_BUILD_ROOT")
WINK_OUT=${WINK_OUT:-$WINK_BUILD_ROOT/out}

stop_after() { [ "$WINK_UNTIL" = "$1" ] && { log "stopping after $1"; exit 0; }; return 0; }

detect_host
deps_check;   stop_after deps
venv_setup;   stop_after venv
emsdk_setup;  stop_after emsdk
emdawn_setup; stop_after emdawn

for v in $WINK_VARIANTS; do
  tint_setup "$v"
  [ "$WINK_UNTIL" = tint ] && continue
  mesa_build "$v"
  [ "$WINK_UNTIL" = mesa ] && continue
  package_variant "$v"
  if [ "$WINK_TEST" = 1 ]; then
    WINK_STAGE="test/$v"
    have node || die "--test needs node (22 or newer)"
    run bash "$WINK_DIR/test/build-smoke.sh" "$v" "$WINK_OUT/$v" "$WINK_EMBIN/emcc" "$WINK_BUILD_ROOT/test/$v" >/dev/null
    run node "$WINK_DIR/test/run-browser.mjs" "$WINK_BUILD_ROOT/test/$v" || die "smoke test failed"
    if [ "$v" = mt ]; then
      # And GL from a worker, executed on the main thread.
      run node "$WINK_DIR/test/run-browser.mjs" "$WINK_BUILD_ROOT/test/$v" --page=main-thread.html \
        || die "main-thread smoke test failed"
      # And the canvas, the device and GL all on a worker.
      run node "$WINK_DIR/test/run-browser.mjs" "$WINK_BUILD_ROOT/test/$v" --page=offscreen.html \
        || die "OffscreenCanvas smoke test failed"
    fi
    # The Vulkan driver: its native tests, with a shim for a loader.
    if have glslangValidator; then
      run bash "$WINK_DIR/test/vk/build-vk.sh" "$v" "$WINK_OUT/$v" "$WINK_EMBIN/emcc" "$WINK_BUILD_ROOT/test/vk-$v" >/dev/null
      for t in wvk_smoke wvk_draw; do
        run node "$WINK_DIR/test/run-browser.mjs" "$WINK_BUILD_ROOT/test/vk-$v" "--page=vk.html?test=$t" \
          || die "Vulkan test $t failed"
        if [ "$v" = mt ]; then
          run node "$WINK_DIR/test/run-browser.mjs" "$WINK_BUILD_ROOT/test/vk-$v" "--page=vk.html?test=$t&worker=1" \
            || die "Vulkan worker test $t failed"
        fi
      done
    else
      log "no glslangValidator: skipping the Vulkan tests"
    fi
  fi
done
WINK_STAGE=wink
case $WINK_UNTIL in
  tint|mesa) log "stopping after $WINK_UNTIL" ;;
  *) log "done: $WINK_OUT" ;;
esac
