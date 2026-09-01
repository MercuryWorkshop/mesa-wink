# Copyright © 2017 Intel Corporation
# SPDX-License-Identifier: MIT

# In a browser the socket this winsys opens is a call into the virglrenderer
# module next door: gex_vtest_os.h renames the five calls it makes, and is
# force-included so the winsys sources themselves stay untouched. The
# implementation is a separate library because it must be compiled *without*
# those renames.
gex_vtest_args = []
gex_vtest_link = []
if host_machine.system() == 'emscripten'
  gex_vtest_args += [
    '-DGEX_VTEST_OS_RENAME',
    '-include', meson.current_source_dir() / 'gex_vtest_os.h',
  ]
  gex_vtest_link += static_library(
    'gexvtestos',
    ['gex_vtest_os.c'],
    include_directories : [inc_include, inc_src],
    gnu_symbol_visibility : 'hidden',
  )
endif

libvirglvtest = static_library(
  'virglvtest',
  ['virgl_vtest_socket.c', 'virgl_vtest_winsys.c'],
  include_directories : [inc_include, inc_src, inc_gallium, inc_gallium_aux, inc_gallium_drivers, inc_virtio],
  dependencies : [dep_libvirglcommon, idep_mesautil],
  c_args : gex_vtest_args,
  link_with : gex_vtest_link,
  gnu_symbol_visibility : 'hidden',
)
