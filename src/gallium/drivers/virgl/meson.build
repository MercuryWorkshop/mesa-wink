# Copyright © 2017 Intel Corporation
# SPDX-License-Identifier: MIT

# No libdrm on wasm: nothing in these sources includes one of its headers, and
# the winsys that did is not built there.
if host_machine.system() == 'emscripten'
  virgl_libdrm_dep = declare_dependency()
else
  virgl_libdrm_dep = dep_libdrm
endif

files_libvirgl = files(
  'virgl_buffer.c',
  'virgl_context.c',
  'virgl_encode.c',
  'virgl_query.c',
  'virgl_resource.c',
  'virgl_screen.c',
  'virgl_staging_mgr.c',
  'virgl_streamout.c',
  'virgl_transfer_queue.c',
  'virgl_texture.c',
  'virgl_tgsi.c',
  'virgl_video.c',
)

libvirgl = static_library(
  'virgl',
  [ files_libvirgl ],
  gnu_symbol_visibility : 'hidden',
  # `virgl_context.c` includes <libsync.h>, which lives in src/util and is
  # otherwise on the path only through a dependency wasm does not have.
  include_directories : [inc_include, inc_src, inc_gallium, inc_gallium_aux,
                         inc_virtio, include_directories('../../../util')],
  dependencies : [virgl_libdrm_dep, idep_mesautil, idep_xmlconfig, idep_nir],
)

virgl_deps = [libvirgl]
if not with_platform_windows
  if host_machine.system() != 'emscripten'
    virgl_deps += libvirgldrm
  endif
  virgl_deps += libvirglvtest
endif

driver_virgl = declare_dependency(
  compile_args : '-DGALLIUM_VIRGL',
  link_with : virgl_deps,
)

if with_tests
  subdir('tests')
endif
