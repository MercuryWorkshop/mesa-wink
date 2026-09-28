# Configure and build Mesa for one variant. Sourced by wink/build.sh.

mesa_cross_file() {  # <variant> <out>
  local pt=""
  [ "$1" = mt ] && pt=", '-pthread'"
  sed -e "s|@WBIN@|$WINK_EMBIN|" -e "s|@PTHREAD@|$pt|g" \
    "$WINK_DIR/cross/wink-cross.ini.in" > "$2.tmp"
  mv "$2.tmp" "$2"
}

mesa_options() {  # <variant>
  local threads=disabled
  [ "$1" = mt ] && threads=enabled
  printf '%s\n' \
    -Dbuildtype="$WINK_BUILDTYPE" -Db_ndebug=true \
    -Db_staticpic=false -Db_lto=false \
    -Dplatforms=[] -Dgallium-drivers=gex_wgpu -Dvulkan-drivers=[] \
    -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dglvnd=disabled \
    -Dopengl=true -Dgles1=disabled -Dgles2=enabled -Dllvm=disabled \
    -Dgallium-va=disabled -Dgallium-rusticl=false -Dteflon=false \
    -Dvideo-codecs=[] -Dvalgrind=disabled -Dlibunwind=disabled -Dlmsensors=disabled \
    -Dzstd=disabled -Dzlib=disabled -Ddisplay-info=disabled -Dexpat=disabled \
    -Dxmlconfig=disabled -Dspirv-tools=disabled -Dshader-cache=disabled \
    -Dbuild-tests=false -Dtools=[] -Dperfetto=false -Dhtml-docs=disabled \
    -Dgex-emdawn-dir="$WINK_EMDAWN" -Dgex-tint-dir="$WINK_TINT" \
    -Dwink=enabled -Dgex-wasm-threads="$threads"
}

# The flags that make these objects linkable into .NET are easy to lose to a
# default somewhere in a subproject, so check what ninja actually ran.
mesa_check_flags() {  # <variant> <builddir>
  "$WINK_PY" - "$1" "$2/compile_commands.json" <<'EOF'
import json, shlex, sys
variant, path = sys.argv[1], sys.argv[2]
bad = []
for e in json.load(open(path)):
    args = e.get('arguments') or shlex.split(e['command'])
    name = e['file']
    if '-fPIC' in args or '-fpic' in args:
        bad.append('%s: -fPIC' % name)
    if not name.endswith(('.c', '.cpp', '.cc')):
        continue
    has_pt = '-pthread' in args
    if variant == 'mt' and not has_pt:
        bad.append('%s: missing -pthread' % name)
    if variant == 'st' and has_pt:
        bad.append('%s: -pthread in the st variant' % name)
    if '-fwasm-exceptions' not in args:
        bad.append('%s: missing -fwasm-exceptions' % name)
if bad:
    sys.stderr.write('\n'.join(bad[:20]) + '\n')
    sys.exit('%d compile commands with the wrong ABI flags' % len(bad))
EOF
}

mesa_build() {  # <variant>
  local variant=$1
  WINK_STAGE="mesa/$variant"
  local build="$WINK_BUILD_ROOT/$variant"
  local cross="$WINK_BUILD_ROOT/$variant.cross.ini"
  mesa_cross_file "$variant" "$cross"

  local opts=() line
  while IFS= read -r line; do opts+=("$line"); done <<EOF
$(mesa_options "$variant")
EOF
  local key
  # The compiler is part of the configuration: the wrappers keep their path
  # whichever emsdk they front, so meson would not notice a different one.
  key=$(hash_str "$(cat "$cross") ${opts[*]} $WINK_EMSDK_ROOT $WINK_EMCC_VERSION_LINE")
  local keyfile="$WINK_BUILD_ROOT/$variant.config"

  if [ ! -f "$build/build.ninja" ]; then
    log "configuring $build"
    run meson setup "$build" "$MESA_DIR" --cross-file "$cross" "${opts[@]}"
  elif [ "$WINK_WIPE" = 1 ] || [ "$(cat "$keyfile" 2>/dev/null)" != "$key" ]; then
    log "reconfiguring $build from scratch"
    run meson setup --wipe "$build" "$MESA_DIR" --cross-file "$cross" "${opts[@]}"
  fi
  printf '%s' "$key" > "$keyfile"

  log "building"
  run ninja -C "$build" -j "$WINK_JOBS" \
    src/gallium/targets/wink-wasm/wink.o \
    src/gallium/targets/wink-wasm/libwink_webgpu.a
  mesa_check_flags "$variant" "$build"
  WINK_MESA_BUILD=$build
}
