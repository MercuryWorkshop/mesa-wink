# emsdk: the compiler .NET 11 links with. Sourced by wink/build.sh.
#
# Objects built by another emcc than the app's link risk a different
# libc/libc++ ABI, so the version is checked, not assumed.
#
# Layouts accepted for --emsdk / $EMSDK:
#   upstream emsdk   <root>/upstream/emscripten/emcc, config <root>/.emscripten
#   dotnet/runtime   <root>/emscripten/emcc,          config <root>/emscripten/.emscripten
#                    what the runtime provisions (.NET 11: its wasm tool cache)
#                    and fna-wasm-build's freeze-emsdk.sh zips -- ikvmcraft's
#                    statics/emsdk, which ikvm-wasm-build builds wink with;
#                    Linux binaries only
#   .NET workload    the Microsoft.NET.Runtime.Emscripten.<ver>.Sdk.<rid> pack,
#                    or a dotnet root to find it in, configured through
#                    DOTNET_EMSCRIPTEN_* variables and a frozen cache from the
#                    sibling Cache pack
# Without one, upstream $WINK_EMSDK_VERSION is bootstrapped. Under .NET 10
# (3.1.56) every .NET-built emsdk paired a clang that emitted the newer
# __wasm_setjmp helpers with a compiler-rt that had only saveSetjmp, so wink
# had to be compiled with upstream 3.1.56; .NET 11's (6.0.3) passes the
# packaging stage's symbol check against its own libc and compiler-rt
# (2026-11-10: 15870 defined, 2658 undefined), and is what wink is built with.

# .NET 11's emscripten reports 6.0.3 (commit 6ea9c28c38cd), although its tool cache and the runtime's
# emscripten-version.txt say 6.0.2; ikvm-wasm-build builds wink with that compiler (the frozen statics/emsdk).
WINK_EMSDK_VERSION=6.0.3
WINK_EMSDK_REPO=https://github.com/emscripten-core/emsdk.git

# The host's .NET runtime identifier, for finding the workload's pack.
emsdk_dotnet_rid() {
  local os arch
  case "$WINK_HOST" in macos) os=osx ;; *) os=linux ;; esac
  case "$WINK_HOST_ARCH" in arm64|aarch64) arch=arm64 ;; *) arch=x64 ;; esac
  echo "$os-$arch"
}

# The workload splits emsdk into sibling packs of one version: Sdk (emscripten,
# LLVM, binaryen), Cache (a frozen sysroot), Node and Python.
emsdk_locate_dotnet_pack() {
  local root=$1 packs name ver
  packs=$(dirname "$(dirname "$root")")
  name=$(basename "$(dirname "$root")")
  ver=$(basename "$root")
  WINK_EMSCRIPTEN="$root/tools/emscripten"
  WINK_EM_CONFIG="$root/tools/emscripten/.emscripten"
  WINK_EMSDK_ROOT=$root
  WINK_EM_ENV="DOTNET_EMSCRIPTEN_LLVM_ROOT='$root/tools/bin'
DOTNET_EMSCRIPTEN_BINARYEN_ROOT='$root/tools'
DOTNET_EMSCRIPTEN_NODE_JS='$packs/${name/.Sdk./.Node.}/$ver/tools/bin/node'
EM_CACHE='$packs/${name/.Sdk./.Cache.}/$ver/tools/emscripten/cache'
export DOTNET_EMSCRIPTEN_LLVM_ROOT DOTNET_EMSCRIPTEN_BINARYEN_ROOT DOTNET_EMSCRIPTEN_NODE_JS EM_CACHE"
  local py="$packs/${name/.Sdk./.Python.}/$ver/tools/bin/python3"
  [ -x "$py" ] && WINK_EMSDK_PY=$py
  return 0
}

emsdk_locate() {
  local root=$1 pack
  if [ -d "$root/src/mono/browser/emsdk" ]; then
    root="$root/src/mono/browser/emsdk"
  fi
  # A dotnet root: the workload's pack for this host, newest version.
  if [ -d "$root/packs" ] && [ ! -e "$root/tools/emscripten/emcc" ]; then
    pack=$(ls -d "$root/packs/Microsoft.NET.Runtime.Emscripten.$WINK_EMSDK_VERSION.Sdk.$(emsdk_dotnet_rid)"/*/ 2>/dev/null | sort | tail -1)
    [ -n "$pack" ] || die "no Emscripten $WINK_EMSDK_VERSION workload pack under $root (dotnet workload install wasm-tools)"
    root=${pack%/}
  fi
  if [ -x "$root/tools/emscripten/emcc" ]; then
    emsdk_locate_dotnet_pack "$root"
    return 0
  fi
  if [ -x "$root/upstream/emscripten/emcc" ]; then
    WINK_EMSCRIPTEN="$root/upstream/emscripten"
    WINK_EM_CONFIG="${EM_CONFIG:-$root/.emscripten}"
  elif [ -x "$root/emscripten/emcc" ]; then
    WINK_EMSCRIPTEN="$root/emscripten"
    WINK_EM_CONFIG="${EM_CONFIG:-$root/emscripten/.emscripten}"
  else
    return 1
  fi
  WINK_EMSDK_ROOT=$root
  [ -f "$WINK_EM_CONFIG" ] || die "no emscripten config at $WINK_EM_CONFIG (set EM_CONFIG)"
}

emsdk_bootstrap() {
  local root="$WINK_CACHE/emsdk"
  if [ ! -d "$root/.git" ]; then
    log "cloning emsdk"
    run git clone --depth 1 "$WINK_EMSDK_REPO" "$root"
  fi
  if ! stamp_ok "$root" "$WINK_EMSDK_VERSION"; then
    log "installing emscripten $WINK_EMSDK_VERSION"
    ( cd "$root" && EMSDK_PYTHON="$WINK_HOST_PY" run ./emsdk install "$WINK_EMSDK_VERSION" \
        && EMSDK_PYTHON="$WINK_HOST_PY" run ./emsdk activate "$WINK_EMSDK_VERSION" >/dev/null )
    stamp_write "$root" "$WINK_EMSDK_VERSION"
  fi
  emsdk_locate "$root" || die "emsdk install at $root is incomplete"
}

# Wrappers pin the environment into the tools themselves, so meson's cross
# file and a bare `ninja -C <build>` in a fresh shell both see this emsdk and
# nothing else.
emsdk_write_wrappers() {
  local bin="$WINK_BUILD_ROOT/bin" tool
  mkdir -p "$bin"
  for tool in emcc em++ emar emranlib emstrip emnm emcmake emconfigure embuilder; do
    [ -e "$WINK_EMSCRIPTEN/$tool" ] || continue
    cat > "$bin/$tool" <<EOF
#!/bin/sh
EMSDK='$WINK_EMSDK_ROOT'
EM_CONFIG='$WINK_EM_CONFIG'
EMSDK_PYTHON='${WINK_EMSDK_PY:-$WINK_HOST_PY}'
export EMSDK EM_CONFIG EMSDK_PYTHON
${EM_CACHE:+EM_CACHE='$EM_CACHE'; export EM_CACHE}
${WINK_EM_ENV:-}
exec '$WINK_EMSCRIPTEN/$tool' "\$@"
EOF
    chmod +x "$bin/$tool"
  done
  WINK_EMBIN=$bin
}

emsdk_setup() {
  WINK_STAGE=emsdk
  WINK_EM_ENV=
  WINK_EMSDK_PY=
  if [ -n "${WINK_EMSDK:-}" ]; then
    emsdk_locate "$(absdir "$WINK_EMSDK")" \
      || die "$WINK_EMSDK is not an upstream emsdk, dotnet/runtime's, or a .NET workload pack"
  else
    emsdk_bootstrap
  fi
  emsdk_write_wrappers
  # And into this process: emcmake's toolchain file has CMake run emcc
  # directly rather than through the wrappers, and the .NET pack's config
  # reads its paths from the environment.
  EMSDK=$WINK_EMSDK_ROOT
  EM_CONFIG=$WINK_EM_CONFIG
  EMSDK_PYTHON=${WINK_EMSDK_PY:-$WINK_HOST_PY}
  export EMSDK EM_CONFIG EMSDK_PYTHON
  [ -n "$WINK_EM_ENV" ] && eval "$WINK_EM_ENV"

  WINK_EMCC_VERSION_LINE=$("$WINK_EMBIN/emcc" --version | sed -n 1p)
  if ! printf '%s' "$WINK_EMCC_VERSION_LINE" | grep -q " $WINK_EMSDK_VERSION "; then
    if [ "${WINK_ALLOW_ANY_EMSDK:-0}" = 1 ]; then
      warn "emcc is not $WINK_EMSDK_VERSION: $WINK_EMCC_VERSION_LINE"
    else
      die "emcc is not $WINK_EMSDK_VERSION (.NET 11's): $WINK_EMCC_VERSION_LINE -- WINK_ALLOW_ANY_EMSDK=1 to override"
    fi
  fi
  WINK_CLANG_VERSION_LINE=$("$WINK_EMBIN/emcc" -v 2>&1 | grep -m1 'clang version' || true)
  log "$WINK_EMCC_VERSION_LINE"
  log "emsdk at $WINK_EMSDK_ROOT"
}
