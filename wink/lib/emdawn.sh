# emdawnwebgpu: webgpu.h and its JS implementation. Sourced by wink/build.sh.
#
# A recent release, built for Emscripten 4.0.10+ (its README), which .NET 11's
# emscripten 6.0.3 is. It is independent of Tint, which stays at Dawn cdda346
# for the AST SPIR-V reader (lib/tint.sh): the two share no code in the link.
# Until .NET 11 the pin was v20250822.104650 (Tint's revision, built against
# emsdk 4.0.8) with edits that made its JS load under 3.1.56; those are gone.
# tools/patch-emdawnwebgpu.py carries the fixes upstream still lacks.

WINK_EMDAWN_VERSION=v20261109.214834
WINK_EMDAWN_SHA512=ac0dfb47b7bc8f356778f711382a86d45528d53a8619f1df6e6325eeee0e0982a4f1bd7c53cec0640c65e8fd2d604a28fdced1003cd3ae9f76827d6b02bac9b6
WINK_EMDAWN_URL="https://github.com/google/dawn/releases/download/$WINK_EMDAWN_VERSION/emdawnwebgpu_pkg-$WINK_EMDAWN_VERSION.zip"

emdawn_setup() {
  WINK_STAGE=emdawn
  if [ -n "${WINK_EMDAWN_DIR:-}" ]; then
    WINK_EMDAWN=$(absdir "$WINK_EMDAWN_DIR")
    [ -f "$WINK_EMDAWN/webgpu/include/webgpu/webgpu.h" ] || die "$WINK_EMDAWN is not an emdawnwebgpu package"
    log "using $WINK_EMDAWN (not re-patched)"
    return 0
  fi

  local patcher="$WINK_DIR/tools/patch-emdawnwebgpu.py"
  local dir="$WINK_CACHE/emdawnwebgpu/$WINK_EMDAWN_VERSION-p$(hash_file "$patcher")"
  local zip="$WINK_CACHE/downloads/emdawnwebgpu_pkg-$WINK_EMDAWN_VERSION.zip"
  WINK_EMDAWN="$dir/emdawnwebgpu_pkg"
  stamp_ok "$dir" "$WINK_EMDAWN_SHA512" && { log "cached at $WINK_EMDAWN"; return 0; }

  mkdir -p "$WINK_CACHE/downloads"
  if [ ! -f "$zip" ] || [ "$(sha512_of "$zip")" != "$WINK_EMDAWN_SHA512" ]; then
    log "downloading $WINK_EMDAWN_VERSION"
    run curl -fL --retry 3 -o "$zip.part" "$WINK_EMDAWN_URL"
    [ "$(sha512_of "$zip.part")" = "$WINK_EMDAWN_SHA512" ] || die "sha512 mismatch for $WINK_EMDAWN_URL"
    mv "$zip.part" "$zip"
  fi
  rm -rf "$dir.tmp" && mkdir -p "$dir.tmp"
  run "$WINK_PY" "$WINK_DIR/tools/unzip.py" "$zip" "$dir.tmp"
  run "$WINK_PY" "$patcher" "$dir.tmp/emdawnwebgpu_pkg"
  rm -rf "$dir" && mv "$dir.tmp" "$dir"
  stamp_write "$dir" "$WINK_EMDAWN_SHA512"
  log "ready at $WINK_EMDAWN"
}
