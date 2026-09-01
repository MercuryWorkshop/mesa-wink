gex_emdawn_dir = get_option('gex-emdawn-dir')
gex_tint_dir = get_option('gex-tint-dir')
if gex_emdawn_dir == ''
  error('gex_wgpu requires -Dgex-emdawn-dir=/path/to/emdawnwebgpu_pkg')
endif
if gex_tint_dir == ''
  error('gex_wgpu requires -Dgex-tint-dir=/path/to/tint')
endif

gex_emdawn_include = include_directories(
  join_paths(gex_emdawn_dir, 'webgpu', 'include'),
  join_paths(gex_emdawn_dir, 'webgpu_cpp', 'include'),
)
gex_tint_include = include_directories(join_paths(gex_tint_dir, 'include'))
gex_tint = cpp.find_library('tint', dirs : gex_tint_dir, static : true)

# On a desktop the WebGPU implementation is a library to link; in a browser it
# is the page. gex_wgpu_native.c is only built for the former: it is the device
# and the presentation path, and neither exists in the wasm build.
gex_dawn_dir = get_option('gex-dawn-native-dir')
gex_native_sources = []
gex_dawn_deps = []
if gex_dawn_dir != ''
  gex_native_sources += files('gex_wgpu_native.c')
  gex_dawn_deps += cc.find_library('webgpu_dawn', dirs : gex_dawn_dir)
endif

libgex_wgpu = static_library(
  'gex_wgpu',
  gex_native_sources + files(
    'gex_wgpu_screen.c',
    'gex_wgpu_tint.cpp',
    '../../../microsoft/compiler/dxil_nir_lower_int_cubemaps.c',
  ),
  include_directories : [
    gex_emdawn_include,
    inc_gallium,
    inc_gallium_aux,
    inc_include,
    inc_src,
    include_directories('../zink/nir_to_spirv'),
    gex_tint_include,
  ],
  dependencies : [idep_mesautil, idep_nir, idep_ntv, gex_tint],
  cpp_args : [
    '-std=c++20',
    '-fno-rtti',
    '-fno-exceptions',
    '-DTINT_BUILD_SPV_READER=1',
    '-DTINT_BUILD_WGSL_READER=1',
    '-DTINT_BUILD_WGSL_WRITER=1',
    '-DTINT_BUILD_SPV_WRITER=0',
    '-DTINT_BUILD_GLSL_WRITER=0',
    '-DTINT_BUILD_HLSL_WRITER=0',
    '-DTINT_BUILD_MSL_WRITER=0',
    '-DTINT_BUILD_IR_BINARY=0',
    '-DTINT_BUILD_IS_LINUX=1',
    '-DTINT_BUILD_IS_MAC=0',
    '-DTINT_BUILD_IS_WIN=0',
  ],
  gnu_symbol_visibility : 'hidden',
)

driver_gex_wgpu = declare_dependency(
  link_with : libgex_wgpu,
  dependencies : [gex_tint] + gex_dawn_deps,
  include_directories : [gex_emdawn_include, include_directories('.')],
  # What target-helpers/sw_helper.h keys GALLIUM_DRIVER=gex_wgpu off.
  compile_args : '-DGALLIUM_GEX_WGPU',
)
