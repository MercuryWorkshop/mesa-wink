# emsdk: the compiler .NET 10 links with. Sourced by wink/build.sh.
#
# .NET 10's browser-wasm workload is pinned to Emscripten 3.1.56, and objects
# built by any other emcc risk a different libc/libc++ ABI in the final link, so
# the version is checked, not assumed.
#
# Two layouts are accepted for --emsdk / $EMSDK:
#   upstream emsdk   <root>/upstream/emscripten/emcc, config <root>/.emscripten
#   .NET's emsdk     <root>/emscripten/emcc,          config <root>/emscripten/.emscripten
# the second being what dotnet/runtime provisions (src/mono/browser/emsdk) and
# what fna-wasm-build's freeze-emsdk.sh zips. That one carries Linux binaries,
# so on macOS the upstream 3.1.56 is bootstrapped instead; the objects it
# produces are the same, the .NET patches touching no public header.

WINK_EMSDK_VERSION=3.1.56
WINK_EMSDK_REPO=https://github.com/emscripten-core/emsdk.git

emsdk_locate() {
  local root=$1
  if [ -d "$root/src/mono/browser/emsdk" ]; then
    root="$root/src/mono/browser/emsdk"
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
EMSDK_PYTHON='$WINK_HOST_PY'
export EMSDK EM_CONFIG EMSDK_PYTHON
${EM_CACHE:+EM_CACHE='$EM_CACHE'; export EM_CACHE}
exec '$WINK_EMSCRIPTEN/$tool' "\$@"
EOF
    chmod +x "$bin/$tool"
  done
  WINK_EMBIN=$bin
}

emsdk_setup() {
  WINK_STAGE=emsdk
  if [ -n "${WINK_EMSDK:-}" ]; then
    emsdk_locate "$(absdir "$WINK_EMSDK")" \
      || die "$WINK_EMSDK is neither an upstream emsdk nor .NET's emsdk layout"
  else
    emsdk_bootstrap
  fi
  emsdk_write_wrappers

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
