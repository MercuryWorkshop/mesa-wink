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
# The shader compiler is src/wink/compiler's, shared with the Vulkan driver.
gex_tint = wink_tint

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
    '../../../microsoft/compiler/dxil_nir_lower_int_cubemaps.c',
  ),
  include_directories : [
    gex_emdawn_include,
    inc_gallium,
    inc_gallium_aux,
    inc_include,
    inc_src,
  ],
  # Not gex_tint: a static library's dependencies travel to everything that
  # links it, and wink ships Tint as its own archive so a consumer that
  # already links the same Tint (MobileGL's, from the same Dawn revision) can
  # leave one copy out. driver_gex_wgpu below still carries it.
  dependencies : [idep_mesautil, idep_nir, idep_wink_compiler],
  gnu_symbol_visibility : 'hidden',
)

driver_gex_wgpu = declare_dependency(
  link_with : libgex_wgpu,
  dependencies : [gex_tint] + gex_dawn_deps,
  include_directories : [gex_emdawn_include, include_directories('.')],
  # What target-helpers/sw_helper.h keys GALLIUM_DRIVER=gex_wgpu off.
  compile_args : '-DGALLIUM_GEX_WGPU',
)
