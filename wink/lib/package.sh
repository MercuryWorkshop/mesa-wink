# Lay out one variant for a consumer and check its symbols. Sourced by
# wink/build.sh.
#
# Link flags a consumer must, should, and must not use are written into
# manifest.json once, here, and the .rsp and .targets are generated from the
# same lists so the three cannot disagree.

WINK_JS_ORDER="library_webgpu_enum_tables.js library_webgpu_generated_struct_info.js library_webgpu_generated_sig_info.js library_webgpu.js library_wink.js"

# Meson makes uninstalled static libraries thin archives, which only name
# their members' paths: copied anywhere else they point at nothing. Rebuild a
# real archive from the members; `q` rather than `r`, so two members with the
# same basename are both kept.
package_archive() {  # <src.a> <dst.a>
  local src=$1 dst=$2 dir
  if [ "$(head -c 7 "$src")" != '!<thin>' ]; then
    cp "$src" "$dst"
    return 0
  fi
  dir=$(dirname "$src")
  rm -f "$dst"
  ( cd "$dir" && "$WINK_EMBIN/emar" t "$src" | while IFS= read -r m; do
      "$WINK_EMBIN/emar" qc "$dst" "$m"; done )
  "$WINK_EMBIN/emranlib" "$dst"
}

package_variant() {  # <variant>
  local v=$1
  WINK_STAGE="package/$v"
  local out="$WINK_OUT/$v" tmp="$WINK_OUT/$v.tmp"
  local target="$WINK_MESA_BUILD/src/gallium/targets/wink-wasm"
  local src="$MESA_DIR/src/gallium/targets/wink-wasm"
  rm -rf "$tmp" && mkdir -p "$tmp/lib" "$tmp/js" "$tmp/include/webgpu" "$tmp/symbols"

  # One member, so the consumer's link takes all of wink or none of it.
  ( cd "$target" && "$WINK_EMBIN/emar" rcs "$tmp/lib/libwink.a" wink.o )
  package_archive "$target/libwink_webgpu.a" "$tmp/lib/libwink_webgpu.a"
  cp "$WINK_TINT/libtint.a" "$tmp/lib/libwink_tint.a"

  local f
  for f in library_webgpu_enum_tables.js library_webgpu_generated_struct_info.js \
           library_webgpu_generated_sig_info.js library_webgpu.js webgpu-externs.js; do
    cp "$WINK_EMDAWN/webgpu/src/$f" "$tmp/js/"
  done
  cp "$src/library_wink.js" "$tmp/js/"
  cp "$src/wink.h" "$src/wink_emscripten_compat.h" "$tmp/include/"
  cp "$WINK_EMDAWN/webgpu/include/webgpu/webgpu.h" "$tmp/include/webgpu/"

  # The sysroot's libc, for the st pthread check. EM_CACHE when set (it always
  # is for .NET's pack, whose emcc cannot answer --cflags: that compiles a
  # test file the pack does not ship); otherwise ask emcc.
  local libc="" em_cache=${EM_CACHE:-}
  if [ -z "$em_cache" ]; then
    em_cache=$("$WINK_EMBIN/emcc" --cflags 2>/dev/null | tr ' ' '\n' \
                 | sed -n 's|^--sysroot=\(.*\)/sysroot$|\1|p' | head -1) || em_cache=
  fi
  local rt="" mt_suffix=""
  [ "$v" = mt ] && mt_suffix=-mt
  if [ -n "$em_cache" ]; then
    libc="$em_cache/sysroot/lib/wasm32-emscripten/libc.a"
    rt="$em_cache/sysroot/lib/wasm32-emscripten/libcompiler_rt-wasm-sjlj$mt_suffix.a"
  fi
  run "$WINK_PY" "$WINK_DIR/tools/wink-symbols.py" --emnm "$WINK_EMBIN/emnm" --variant "$v" \
    --object "$target/wink.o" --tint "$tmp/lib/libwink_tint.a" --webgpu "$tmp/lib/libwink_webgpu.a" \
    ${libc:+--libc "$libc"} ${rt:+--compiler-rt "$rt"} --check-against "$WINK_CHECK_AGAINST" \
    --collisions-out "$tmp/symbols/collisions.txt" --out "$tmp/symbols"


  WINK_V=$v WINK_TMP=$tmp WINK_JS_ORDER=$WINK_JS_ORDER \
  WINK_EMCC_VERSION_LINE=$WINK_EMCC_VERSION_LINE WINK_CLANG_VERSION_LINE=$WINK_CLANG_VERSION_LINE \
  WINK_DAWN_REF=$WINK_DAWN_REF WINK_EMDAWN_VERSION=$WINK_EMDAWN_VERSION \
  WINK_MESA_DESCRIBE=$(git -C "$MESA_DIR" describe --always --dirty 2>/dev/null || echo unknown) \
  WINK_BUILDTYPE=$WINK_BUILDTYPE WINK_TARGETS_IN="$WINK_DIR/msbuild/wink.targets.in" \
  "$WINK_PY" - <<'EOF'
import hashlib, json, os
e = os.environ
v, tmp = e['WINK_V'], e['WINK_TMP']
mt = v == 'mt'
js = e['WINK_JS_ORDER'].split()
required = ['-fwasm-exceptions', '-sSUPPORT_LONGJMP=wasm', '-sWASM_BIGINT=1'] + (['-pthread'] if mt else [])
# Tint's passes recurse deeply; a 16 MB stack needs an initial memory above it.
# Every pthread's stack is DEFAULT_PTHREAD_STACK_SIZE, which is STACK_SIZE
# unless set: 16 MB each of the program's threads. GL compiles its shaders
# in well under a megabyte, and Tint goes to a stack of its own where the
# thread's has less free than it may want (gex_wgpu_tint.cpp).
recommended = ['-sALLOW_MEMORY_GROWTH=1', '-sMAXIMUM_MEMORY=4GB', '-sSTACK_SIZE=16MB', '-sINITIAL_MEMORY>=32MB']
if mt:
    recommended.append('-sDEFAULT_PTHREAD_STACK_SIZE=2MB')
forbidden = ['-sUSE_WEBGPU=1', '-fexceptions (JS exceptions)', '-sDISABLE_EXCEPTION_CATCHING=0',
             '-sSUPPORT_LONGJMP=emscripten', '-fPIC objects in the same link']
files = {}
for root, _, names in os.walk(tmp):
    for n in names:
        p = os.path.join(root, n)
        files[os.path.relpath(p, tmp)] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
manifest = {
    'wink_abi': 1,
    'variant': v,
    'mesa': e['WINK_MESA_DESCRIBE'],
    'buildtype': e['WINK_BUILDTYPE'],
    'emcc': e['WINK_EMCC_VERSION_LINE'],
    'clang': e['WINK_CLANG_VERSION_LINE'],
    'dawn': e['WINK_DAWN_REF'],
    'emdawnwebgpu': e['WINK_EMDAWN_VERSION'],
    'link': {
        'archives': ['lib/libwink.a', 'lib/libwink_webgpu.a', 'lib/libwink_tint.a'],
        'js_libraries': ['js/' + j for j in js],
        'closure_externs': ['js/webgpu-externs.js'],
        'required': required,
        'recommended': recommended,
        'forbidden': forbidden,
    },
    'jspi': 'Calls that can wait (wink_init, and GL calls that read back) must run '
            'under a WebAssembly.promising-wrapped export on the thread that owns the device.',
    'files': files,
}
json.dump(manifest, open(os.path.join(tmp, 'manifest.json'), 'w'), indent=2, sort_keys=True)
flags = required + [f for f in recommended if '>=' not in f] + ['--js-library=%s' % os.path.join(tmp.rsplit('.tmp', 1)[0], 'js', j) for j in js]
open(os.path.join(tmp, 'wink.rsp'), 'w').write('\n'.join(flags) + '\n')
def jsflag(j):
    return '--js-library=&quot;$(WinkRoot)js/%s&quot;' % j
t = (open(e['WINK_TARGETS_IN']).read()
     .replace('@VARIANT@', v)
     .replace('@REQUIRED_FLAGS@', ' '.join(required))
     .replace('@WEBGPU_JS@', ' '.join(jsflag(j) for j in js if j != 'library_wink.js'))
     .replace('@WINK_JS@', jsflag('library_wink.js')))
open(os.path.join(tmp, 'wink.targets'), 'w').write(t)
EOF
  rm -rf "$out" && mv "$tmp" "$out"
  log "packaged $out"
}
