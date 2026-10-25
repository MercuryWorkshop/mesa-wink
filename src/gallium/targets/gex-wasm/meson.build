gex_mesa_smoke = executable(
  'gex-mesa-smoke',
  files('gex_mesa_smoke.c'),
  include_directories : [
    inc_include,
    inc_src,
    inc_mesa,
    inc_gallium,
    inc_gallium_aux,
    inc_gallium_winsys,
    inc_gallium_drivers,
  ],
  link_whole : [libglapi_bridge],
  link_with : [libglapi, libgallium, libmesa, libws_null],
  dependencies : [
    idep_mesautil,
    idep_nir,
    idep_libglsl,
    driver_softpipe,
    dep_thread,
    dep_clock,
  ],
  link_args : [
    '-sALLOW_MEMORY_GROWTH=1',
    '-sASSERTIONS=2',
    '--profiling-funcs',
  ],
  install : false,
)

# Compile gex's wire decoder against Mesa.  The source is owned by gex rather
# than vendored into Mesa, so the wire format and its decoder change together.
gex_root = join_paths(meson.project_source_root(), '..', '..')
gex_decoder = join_paths(gex_root, 'web', 'mesa', 'gexgl-mesa.cpp')
gex_jspi_js = join_paths(gex_root, 'web', 'gex-wgpu-jspi.js')
# The loader for the replay's cold part (gex: docs/mesa-wasm-size.md), which
# the build splits off with wasm-split after the link (SPLIT_MODULE below).
gex_split_js = join_paths(gex_root, 'web', 'gex-wgpu-split.js')
# Both live in gex, which holds this tree at third_party/mesa. Anywhere else
# the targets built from them are skipped rather than failing configuration.
have_gex_decoder = fs.exists(gex_decoder)
have_gex_jspi = fs.exists(gex_jspi_js)
if not have_gex_decoder
  message('gex-wasm: no gex checkout at @0@; skipping the replay targets'.format(gex_root))
endif
if have_gex_decoder
  gex_replay = executable(
    'gex-mesa-replay',
    gex_decoder,
    include_directories : [
      inc_include,
      inc_src,
      inc_mesa,
      inc_gallium,
      inc_gallium_aux,
      inc_gallium_winsys,
      inc_gallium_drivers,
    ],
    cpp_args : ['-DGEX_MESA_WASM=1'],
    link_whole : [libglapi_bridge],
    link_with : [libglapi, libgallium, libmesa, libws_null],
    dependencies : [
      idep_mesautil,
      idep_nir,
      idep_libglsl,
      driver_softpipe,
      dep_thread,
      dep_clock,
    ],
    link_args : [
      '--no-entry',
      '-sALLOW_MEMORY_GROWTH=1',
      '-sASSERTIONS=2',
      '-sMODULARIZE=1',
      '-sEXPORT_ES6=1',
      '-sEXPORT_NAME=createGexMesa',
      '-sENVIRONMENT=web,worker,node',
      '-sEXPORTED_FUNCTIONS=_gex_gl_init,_gex_gl_set_api,_gex_gl_setenv,_gex_gl_submit,_gex_gl_reply_len,_gex_gl_reply_data,_gex_gl_display,_gex_gl_presented,_gex_gl_produced,_malloc,_free',
      '-sEXPORTED_RUNTIME_METHODS=HEAPU8',
    ],
    install : false,
  )
endif

# Sanitizer shadow memory and metadata make the instrumented static image much
# larger.  Apply the larger heap only to these diagnostic targets; release
# browser artifacts retain their small initial allocation and grow on demand.
gex_corpus_initial_memory = ['-sINITIAL_MEMORY=32MB']
gex_wgpu_initial_memory = '-sINITIAL_MEMORY=64MB'
if get_option('gex-sanitize')
  gex_corpus_initial_memory = ['-sINITIAL_MEMORY=512MB']
  gex_wgpu_initial_memory = '-sINITIAL_MEMORY=512MB'
endif

# Compile the exact same Gallium semantic corpus used by the WebGPU browser
# target against softpipe.  Keeping one source corpus prevents backend-specific
# tests from silently encoding different state transitions or expected pixels.
gex_gallium_softpipe_conformance = executable(
  'gex-gallium-softpipe-conformance',
  files('gex_wgpu_smoke.c'),
  include_directories : [
    inc_include,
    inc_src,
    inc_gallium,
    inc_gallium_aux,
    inc_gallium_winsys,
    inc_gallium_drivers,
  ],
  c_args : ['-DGEX_CONFORMANCE_SOFTPIPE=1'],
  link_with : [libgallium, libws_null],
  dependencies : [
    idep_mesautil,
    idep_nir,
    driver_softpipe,
    dep_thread,
    dep_clock,
  ],
  link_args : gex_corpus_initial_memory + [
    '-sSTACK_SIZE=16MB',
    '-sALLOW_MEMORY_GROWTH=1',
    '-sASSERTIONS=2',
    '--profiling-funcs',
  ],
  install : false,
)

gex_emdawn_dir = get_option('gex-emdawn-dir')
if gex_emdawn_dir != ''
  gex_emdawn_js = [
    join_paths(gex_emdawn_dir, 'webgpu', 'src', 'library_webgpu_enum_tables.js'),
    join_paths(gex_emdawn_dir, 'webgpu', 'src', 'library_webgpu_generated_struct_info.js'),
    join_paths(gex_emdawn_dir, 'webgpu', 'src', 'library_webgpu_generated_sig_info.js'),
    join_paths(gex_emdawn_dir, 'webgpu', 'src', 'library_webgpu.js'),
  ]
  gex_emdawn_link_args = []
  foreach js_library : gex_emdawn_js
    gex_emdawn_link_args += '--js-library=' + js_library
  endforeach
  if have_gex_jspi
    gex_emdawn_link_args += '--js-library=' + gex_jspi_js
  endif

  # The production shape: gex's wire decoder, Mesa's frontend, and the WebGPU
  # Gallium driver.  Same source and same exported ABI as gex-mesa-replay, so
  # one recorded guest stream runs against softpipe and WebGPU unchanged and
  # the two can be compared directly.
  # The D3D device beside the GL one (gex: web/mesa/gexd3d-mesa.cpp), on
  # a dirk package. dirk is DXVK's front end and uses C++ exceptions, so the
  # link has them; Mesa's own objects are compiled as without it.
  gex_dirk_dir = get_option('gex-dirk-dir')
  gex_dirk_sources = []
  gex_dirk_args = []
  gex_dirk_link_args = []
  gex_dirk_include = []
  gex_dirk_depends = []
  if gex_dirk_dir != ''
    gex_dirk_sources = [join_paths(gex_root, 'web', 'mesa', 'gexd3d-mesa.cpp')]
    gex_dirk_args = ['-DGEX_MESA_DIRK=1']
    gex_dirk_link_args = [join_paths(gex_dirk_dir, 'lib', 'libdirk.a'), '-fwasm-exceptions']
    gex_dirk_include = include_directories(join_paths(gex_dirk_dir, 'include'))
    gex_dirk_depends = [join_paths(gex_dirk_dir, 'lib', 'libdirk.a')]
  endif

  if have_gex_decoder and have_gex_jspi
    gex_wgpu_replay = executable(
      'gex-wgpu-replay',
      gex_decoder,
      gex_dirk_sources,
      join_paths(gex_emdawn_dir, 'webgpu', 'src', 'webgpu.cpp'),
      include_directories : [
        inc_include,
        inc_src,
        inc_mesa,
        inc_gallium,
        inc_gallium_aux,
        inc_gallium_winsys,
        inc_gallium_drivers,
        gex_emdawn_include,
        gex_dirk_include,
      ],
      cpp_args : ['-DGEX_MESA_WASM=1', '-DGEX_MESA_WGPU=1'] + gex_dirk_args,
      link_whole : [libglapi_bridge],
      link_with : [libglapi, libgallium, libmesa],
      dependencies : [
        idep_mesautil,
        idep_nir,
        idep_libglsl,
        driver_gex_wgpu,
        dep_thread,
        dep_clock,
      ],
      link_depends : gex_emdawn_js + [gex_jspi_js, gex_split_js] + gex_dirk_depends,
      link_args : gex_emdawn_link_args + gex_dirk_link_args + [
        '--no-entry',
        '-sWASM_BIGINT=1',
        '-sSTACK_SIZE=16MB',
        gex_wgpu_initial_memory,
        '-sALLOW_MEMORY_GROWTH=1',
        # No ASSERTIONS: at 2 they put a stack-limit check on each of the
        # 15,695 stack-pointer updates (gex: docs/mesa-wasm-size.md). The
        # main stack goes first instead, so overflowing it wraps below
        # address 0 and traps; a pthread's 16 MB stack has no such guard.
        '-sASSERTIONS=0',
        '-Wl,--stack-first',
        '-sMODULARIZE=1',
        '-sEXPORT_ES6=1',
        '-sEXPORT_NAME=createGexWgpuReplay',
        '-sENVIRONMENT=web,worker',
        # Keep the name section: st/mesa calling a hook this driver leaves NULL
        # is the expected failure here, and an index is not a diagnosis. gex's
        # split step needs the names too; its dists strip them.
        '--profiling-funcs',
        # Emscripten's glue for a module split in two: the link also writes an
        # instrumented .wasm and the plain one as .wasm.orig, which gex's build
        # (web/mesa-split.py) splits into the module the page loads and the
        # cold part it loads only when a function there first runs.
        '-sSPLIT_MODULE=1',
        '--pre-js=' + gex_split_js,
        # Binaryen inlines a function with one caller whatever its size, which
        # folded gex's 2,520 generated per-command decoders (noinline in C++)
        # back into one 1.2 MB `one()`. Kept apart, each is its own function
        # the split can place, and the module compresses better.
        '-sBINARYEN_EXTRA_PASSES=--one-caller-inline-max-function-size=0',
        '-sEXPORTED_FUNCTIONS=_gex_gl_init,_gex_gl_set_api,_gex_gl_setenv,_gex_gl_submit,_gex_gl_reply_len,_gex_gl_reply_data,_gex_gl_display,_gex_gl_presented,_gex_gl_produced,_gex_wgsl_cache_load,_gex_wgsl_cache_dump,_gex_wgsl_cache_stats,_gex_wgsl_cache_stats2,_gex_wgsl_cache_budget,_malloc,_free',
        '-sEXPORTED_RUNTIME_METHODS=HEAPU8',
      ],
      # std::span, which emdawnwebgpu's webgpu.cpp uses since v20250822.
      override_options : ['cpp_std=c++20'],
      install : false,
    )
  endif

  if have_gex_jspi
    gex_wgpu_smoke = executable(
      'gex-wgpu-smoke',
      files('gex_wgpu_smoke.c'),
      join_paths(gex_emdawn_dir, 'webgpu', 'src', 'webgpu.cpp'),
      include_directories : [
        inc_include,
        inc_src,
        inc_gallium,
        inc_gallium_aux,
        inc_gallium_drivers,
        gex_emdawn_include,
      ],
      link_with : [libgallium],
      dependencies : [idep_mesautil, idep_nir, driver_gex_wgpu, dep_thread, dep_clock],
      link_depends : gex_emdawn_js + [gex_jspi_js],
      link_args : gex_emdawn_link_args + [
        '--no-entry',
        '-sWASM_BIGINT=1',
        '-sSTACK_SIZE=16MB',
        gex_wgpu_initial_memory,
        '-sALLOW_MEMORY_GROWTH=1',
        '-sASSERTIONS=2',
        '-sMODULARIZE=1',
        '-sEXPORT_ES6=1',
        '-sEXPORT_NAME=createGexWgpuSmoke',
        '-sENVIRONMENT=web,worker',
        '-sEXPORTED_FUNCTIONS=_gex_wgpu_smoke',
      ],
      # std::span, which emdawnwebgpu's webgpu.cpp uses since v20250822.
      override_options : ['cpp_std=c++20'],
      install : false,
    )
  endif
endif
