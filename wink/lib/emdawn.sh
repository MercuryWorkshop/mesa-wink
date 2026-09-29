# emdawnwebgpu: webgpu.h and its JS implementation. Sourced by wink/build.sh.
#
# Pinned to the release whose Dawn revision is the one Tint is built from
# (cdda346): the last with Tint's AST SPIR-V reader, and the first release
# with texture-formats-tier1/tier2 and primitive-index that still has it. The
# package was built against emsdk 4.0.8; tools/patch-emdawnwebgpu.py carries
# the edits that make its JS load under 3.1.56 and behave in a heap over 2 GB.

WINK_EMDAWN_VERSION=v20250822.104650
WINK_EMDAWN_SHA512=7658dabf1bc593d23f57878edf48c8b1d3e5e55cab40cfb7523598ac90accddde6c4806e8ce9c53bd668f9116ad8c1bc7ed81d46b889feb8f7beb180b8eac0d3
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
