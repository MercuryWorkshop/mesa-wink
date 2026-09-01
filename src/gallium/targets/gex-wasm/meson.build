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

# Compile gex's existing wire decoder unchanged against Mesa.  The source is
# intentionally owned by gex rather than vendored into Mesa: this makes the
# MobileGL differential backend and the Mesa production backend consume the
# same packet implementation.
gex_root = join_paths(meson.project_source_root(), '..', '..')
gex_replay = executable(
  'gex-mesa-replay',
  join_paths(gex_root, 'web', 'mobilegl', 'gexgl-mobilegl.cpp'),
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
    '-sEXPORTED_FUNCTIONS=_gex_gl_init,_gex_gl_set_api,_gex_gl_submit,_gex_gl_reply_len,_gex_gl_reply_data,_gex_gl_display,_gex_gl_presented,_gex_gl_produced,_malloc,_free',
    '-sEXPORTED_RUNTIME_METHODS=HEAPU8',
  ],
  install : false,
)

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
  gex_emdawn_link_args += '--js-library=' + join_paths(gex_root, 'web', 'gex-wgpu-jspi.js')

  # The production shape: gex's wire decoder, Mesa's frontend, and the WebGPU
  # Gallium driver.  Same source and same exported ABI as gex-mesa-replay, so
  # one recorded guest stream runs against softpipe and WebGPU unchanged and
  # the two can be compared directly.
  gex_wgpu_replay = executable(
    'gex-wgpu-replay',
    join_paths(gex_root, 'web', 'mobilegl', 'gexgl-mobilegl.cpp'),
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
    ],
    cpp_args : ['-DGEX_MESA_WASM=1', '-DGEX_MESA_WGPU=1'],
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
    link_depends : gex_emdawn_js + [join_paths(gex_root, 'web', 'gex-wgpu-jspi.js')],
    link_args : gex_emdawn_link_args + [
      '--no-entry',
      '-sWASM_BIGINT=1',
      '-sSTACK_SIZE=16MB',
      gex_wgpu_initial_memory,
      '-sALLOW_MEMORY_GROWTH=1',
      '-sASSERTIONS=2',
      '-sMODULARIZE=1',
      '-sEXPORT_ES6=1',
      '-sEXPORT_NAME=createGexWgpuReplay',
      '-sENVIRONMENT=web,worker',
      # Keep the name section: st/mesa calling a hook this driver leaves NULL
      # is the expected failure here, and an index is not a diagnosis.
      '--profiling-funcs',
      '-sEXPORTED_FUNCTIONS=_gex_gl_init,_gex_gl_set_api,_gex_gl_submit,_gex_gl_reply_len,_gex_gl_reply_data,_gex_gl_display,_gex_gl_presented,_gex_gl_produced,_malloc,_free',
      '-sEXPORTED_RUNTIME_METHODS=HEAPU8',
    ],
    install : false,
  )

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
    link_depends : gex_emdawn_js + [join_paths(gex_root, 'web', 'gex-wgpu-jspi.js')],
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
    install : false,
  )
endif
