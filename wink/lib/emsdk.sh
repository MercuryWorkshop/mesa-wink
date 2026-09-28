# emsdk: the compiler .NET 10 links with. Sourced by wink/build.sh.
#
# .NET 10's browser-wasm workload is pinned to Emscripten 3.1.56, and objects
# built by any other emcc risk a different libc/libc++ ABI in the final link, so
# the version is checked, not assumed.
#
# Layouts accepted for --emsdk / $EMSDK:
#   upstream emsdk   <root>/upstream/emscripten/emcc, config <root>/.emscripten
#   dotnet/runtime   <root>/emscripten/emcc,          config <root>/emscripten/.emscripten
#                    what the runtime repo provisions (src/mono/browser/emsdk) and
#                    fna-wasm-build's freeze-emsdk.sh zips; Linux binaries only
#   .NET workload    the Microsoft.NET.Runtime.Emscripten.3.1.56.Sdk.<rid> pack,
#                    or a dotnet root to find it in: the very compiler a .NET
#                    app's link uses, configured through DOTNET_EMSCRIPTEN_*
#                    variables and a frozen cache from the sibling Cache pack
# Without one, upstream 3.1.56 is bootstrapped, which is also what
# dotnet/runtime provisions. The workload pack is accepted but, as of 10.0.12,
# cannot build wink: its clang (dotnet-llvm-project 19.1.0) emits the newer
# __wasm_setjmp helpers while its own compiler-rt still has saveSetjmp, so
# anything with setjmp in it -- spirv_to_nir -- will not link. The packaging
# stage's symbol check says so rather than leaving it to the app's link.

WINK_EMSDK_VERSION=3.1.56
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
      die "emcc is not $WINK_EMSDK_VERSION (.NET 10's): $WINK_EMCC_VERSION_LINE -- WINK_ALLOW_ANY_EMSDK=1 to override"
    fi
  fi
  WINK_CLANG_VERSION_LINE=$("$WINK_EMBIN/emcc" -v 2>&1 | grep -m1 'clang version' || true)
  log "$WINK_EMCC_VERSION_LINE"
  log "emsdk at $WINK_EMSDK_ROOT"
}
