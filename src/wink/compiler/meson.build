# Copyright 2026 wink contributors
# SPDX-License-Identifier: MIT
#
# The shader compiler both wink drivers use: NIR to SPIR-V with zink's
# emitter, then SPIR-V to WGSL with Tint. Tint itself is not linked here:
# wink ships it as its own archive, so a consumer that already links the
# same Tint can leave one copy out, and each final link names it.

wink_tint_dir = get_option('gex-tint-dir')
if wink_tint_dir == ''
  error('the wink shader compiler needs -Dgex-tint-dir=/path/to/tint (libtint.a and include/)')
endif
wink_tint_include = include_directories(join_paths(wink_tint_dir, 'include'))
wink_tint = cpp.find_library('tint', dirs : wink_tint_dir, static : true)

wink_ntv_dir = '../../gallium/drivers/zink/nir_to_spirv'

libwink_compiler = static_library(
  'wink_compiler',
  files(
    'gex_wgpu_nir.c',
    'gex_wgpu_tint.cpp',
    join_paths(wink_ntv_dir, 'nir_to_spirv.c'),
    join_paths(wink_ntv_dir, 'spirv_builder.c'),
  ),
  include_directories : [
    inc_include,
    inc_src,
    inc_gallium,
    inc_gallium_aux,
    include_directories(wink_ntv_dir),
    wink_tint_include,
  ],
  dependencies : [idep_mesautil, idep_nir],
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
  build_by_default : false,
)

idep_wink_compiler = declare_dependency(
  link_with : libwink_compiler,
  include_directories : [include_directories('.'), include_directories(wink_ntv_dir)],
)
