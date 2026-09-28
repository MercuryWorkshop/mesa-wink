# Tint for wasm: SPIR-V reader + WGSL writer as one libtint.a. Sourced by
# wink/build.sh.
#
# The same build as MobileGL's tools/build-tint.sh, which ikvm-wasm-build has
# been linking into .NET 10 on 3.1.56 in both thread variants: Dawn b465e23
# with every backend off, patches/dawn-tint-emscripten.patch, and SPIRV-Tools
# and SPIRV-Headers re-pinned to the revisions glslang's known_good names. The
# re-pin is what lets a single link carry Tint and glslang/shaderc with one
# copy of SPIRV-Tools; WINK_SPIRV_TOOLS_REV= (empty) keeps Dawn's own.
#
# Built once per variant: an mt archive must be -pthread like everything else
# in an mt link, and an st one must not be.

WINK_DAWN_REPO=${WINK_DAWN_REPO:-https://dawn.googlesource.com/dawn}
WINK_DAWN_REF=b465e23d19a5e27402d80191c0f8de1fb3b385b5
WINK_SPIRV_TOOLS_REV=${WINK_SPIRV_TOOLS_REV-33e02568181e3312f49a3cf33df470bf96ef293a}
WINK_SPIRV_HEADERS_REV=${WINK_SPIRV_HEADERS_REV-2a611a970fdbc41ac2e3e328802aed9985352dca}

tint_fetch_at() {  # <dir> <url> <rev>
  rm -rf "$1" && mkdir -p "$1"
  git -C "$1" init -q
  run git -C "$1" fetch -q --depth 1 "$2" "$3"
  git -C "$1" checkout -q FETCH_HEAD
}

tint_source() {
  local patch="$WINK_DIR/patches/dawn-tint-emscripten.patch"
  local key="dawn=$WINK_DAWN_REF patch=$(hash_file "$patch") spvtools=$WINK_SPIRV_TOOLS_REV spvheaders=$WINK_SPIRV_HEADERS_REV"
  WINK_DAWN_SRC="$WINK_CACHE/dawn/src-$(hash_str "$key")"
  stamp_ok "$WINK_DAWN_SRC" "$key" && return 0

  log "fetching Dawn $WINK_DAWN_REF"
  tint_fetch_at "$WINK_DAWN_SRC" "$WINK_DAWN_REPO" "$WINK_DAWN_REF"
  apply_patch_once "$WINK_DAWN_SRC" "$patch"
  log "fetching Dawn's dependencies"
  run "$WINK_PY" "$WINK_DAWN_SRC/tools/fetch_dawn_dependencies.py" -d "$WINK_DAWN_SRC" -s >/dev/null
  if [ -n "$WINK_SPIRV_TOOLS_REV" ]; then
    log "re-pinning SPIRV-Tools $WINK_SPIRV_TOOLS_REV, SPIRV-Headers $WINK_SPIRV_HEADERS_REV"
    tint_fetch_at "$WINK_DAWN_SRC/third_party/spirv-tools/src" https://github.com/KhronosGroup/SPIRV-Tools "$WINK_SPIRV_TOOLS_REV"
    tint_fetch_at "$WINK_DAWN_SRC/third_party/spirv-headers/src" https://github.com/KhronosGroup/SPIRV-Headers "$WINK_SPIRV_HEADERS_REV"
  fi
  stamp_write "$WINK_DAWN_SRC" "$key"
}

tint_setup() {  # <variant>
  local variant=$1
  WINK_STAGE="tint/$variant"

  local override
  case $variant in st) override=${WINK_TINT_DIR_ST:-} ;; mt) override=${WINK_TINT_DIR_MT:-} ;; esac
  if [ -n "$override" ]; then
    WINK_TINT=$(absdir "$override")
    [ -f "$WINK_TINT/libtint.a" ] && [ -d "$WINK_TINT/include/src/tint" ] \
      || die "$WINK_TINT has no libtint.a + include/src/tint"
    log "using $WINK_TINT"
    return 0
  fi

  local flags=""
  [ "$variant" = mt ] && flags="-pthread"
  local key="variant=$variant emcc=$WINK_EMCC_VERSION_LINE dawn=$WINK_DAWN_REF patch=$(hash_file "$WINK_DIR/patches/dawn-tint-emscripten.patch") spvtools=$WINK_SPIRV_TOOLS_REV flags=$flags"
  WINK_TINT="$WINK_CACHE/tint/$variant-em$WINK_EMSDK_VERSION-$(hash_str "$key")"
  stamp_ok "$WINK_TINT" "$key" && { log "cached at $WINK_TINT"; return 0; }

  tint_source
  local build="$WINK_TINT.build"
  log "configuring (flags: ${flags:-none})"
  run "$WINK_EMBIN/emcmake" cmake -S "$WINK_DAWN_SRC" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$flags" -DCMAKE_CXX_FLAGS="$flags" \
    -DDAWN_ENABLE_VULKAN=OFF -DDAWN_ENABLE_D3D11=OFF -DDAWN_ENABLE_D3D12=OFF \
    -DDAWN_ENABLE_METAL=OFF -DDAWN_ENABLE_OPENGLES=OFF -DDAWN_ENABLE_DESKTOP_GL=OFF \
    -DDAWN_ENABLE_NULL=OFF -DDAWN_ENABLE_SWIFTSHADER=OFF \
    -DDAWN_BUILD_SAMPLES=OFF -DDAWN_BUILD_TESTS=OFF -DDAWN_BUILD_BENCHMARKS=OFF \
    -DTINT_BUILD_TESTS=OFF -DTINT_BUILD_CMD_TOOLS=OFF \
    -DTINT_BUILD_SPV_READER=ON -DTINT_BUILD_WGSL_WRITER=ON -DTINT_BUILD_WGSL_READER=ON \
    -DTINT_BUILD_GLSL_WRITER=OFF -DTINT_BUILD_GLSL_VALIDATOR=OFF \
    -DTINT_BUILD_MSL_WRITER=OFF -DTINT_BUILD_HLSL_WRITER=OFF -DTINT_BUILD_SPV_WRITER=OFF \
    -DDAWN_USE_GLFW=OFF -DDAWN_USE_X11=OFF -DDAWN_USE_WAYLAND=OFF \
    -DDAWN_ENABLE_INSTALL=OFF -DTINT_ENABLE_INSTALL=OFF >/dev/null
  log "building tint_api"
  run cmake --build "$build" --target tint_api -j "$WINK_JOBS" >/dev/null

  # One archive, merged with an MRI script. emar is llvm-ar, which reads MRI on
  # every host, unlike the macOS system ar.
  rm -rf "$WINK_TINT.tmp" && mkdir -p "$WINK_TINT.tmp"
  local mri="$WINK_TINT.tmp/merge.mri" n=0 a
  echo "create $WINK_TINT.tmp/libtint.a" > "$mri"
  find "$build/src/tint" -name 'libtint_*.a' > "$WINK_TINT.tmp/archives"
  find "$build" -name 'libabsl_*.a' >> "$WINK_TINT.tmp/archives"
  find "$build/third_party/spirv-tools" -name 'libSPIRV-Tools*.a' >> "$WINK_TINT.tmp/archives"
  while IFS= read -r a; do echo "addlib $a" >> "$mri"; n=$((n + 1)); done < "$WINK_TINT.tmp/archives"
  printf 'save\nend\n' >> "$mri"
  [ "$n" -gt 0 ] || die "no archives under $build"
  run "$WINK_EMBIN/emar" -M < "$mri"
  "$WINK_EMBIN/emar" t "$WINK_TINT.tmp/libtint.a" >/dev/null
  rm -f "$mri" "$WINK_TINT.tmp/archives"
  log "merged $n archives"

  run "$WINK_PY" "$WINK_DIR/tools/copy-headers.py" "$WINK_DAWN_SRC" src "$WINK_TINT.tmp/include"
  run "$WINK_PY" "$WINK_DIR/tools/copy-headers.py" "$WINK_DAWN_SRC/third_party/abseil-cpp" absl "$WINK_TINT.tmp/include"

  rm -rf "$WINK_TINT" && mv "$WINK_TINT.tmp" "$WINK_TINT"
  stamp_write "$WINK_TINT" "$key"
  rm -rf "$build"
  log "ready at $WINK_TINT"
}
