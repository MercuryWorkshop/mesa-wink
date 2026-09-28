# SPDX-License-Identifier: MIT
#
# wink: st/mesa on gex_wgpu for an application that links Mesa statically
# into its own Emscripten module -- a .NET browser-wasm app, typically -- with
# wink.h in place of EGL. wink/build.sh drives this and packages the result.
#
# The product is wink.o: one relocatable object holding the frontend,
# Gallium, the GLSL and NIR compilers, the driver and the wink API, with every
# object the API does not reach left out by the linker. One object rather
# than a pile of archives means the consumer's link takes all of wink or none
# of it, and the symbol checks in wink/build.sh see exactly what the app will.
#
# Deliberately not in it:
#   * libglapi_bridge, the public gl* entry points. GL is reached only through
#     wink_get_proc_address; a bare glFoo in the app's link would collide
#     with, or silently bind to, Emscripten's WebGL implementation.
#   * Tint, shipped as its own archive so a consumer that already links the
#     same Tint can link one copy.
#   * emdawnwebgpu's webgpu.cpp (libwink_webgpu.a), which must be linked
#     exactly once per module however many WebGPU users it has.

libwink_api = static_library(
  'wink_api',
  files('wink.c', 'wink_driconf.c', 'wink_emscripten_compat.c'),
  c_args : with_wink_vk ? ['-DWINK_HAVE_VULKAN'] : [],
  include_directories : [
    inc_include,
    inc_src,
    inc_mesa,
    inc_gallium,
    inc_gallium_aux,
    inc_gallium_drivers,
    include_directories('../../drivers/gex_wgpu'),
    gex_emdawn_include,
  ],
  dependencies : [idep_mesautil, idep_nir, idep_libglsl, idep_xmlconfig, dep_thread],
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
)

# The Vulkan driver, whole: its entrypoint tables reach the driver's and
# the runtime's functions only weakly, which pulls nothing from an archive.
# libwvk holds the runtime's objects already (it links the runtime whole).
wink_vk_whole = with_wink_vk ? [libwvk] : []
wink_vk_deps = with_wink_vk ? [idep_vulkan_util] : []

wink_reloc = executable(
  'wink',
  name_suffix : 'o',
  link_whole : [libwink_api] + wink_vk_whole,
  link_with : [libglapi, libgallium, libmesa, libgex_wgpu],
  dependencies : [idep_mesautil, idep_nir, idep_libglsl, idep_xmlconfig, dep_thread, dep_clock] + wink_vk_deps,
  link_args : ['-r'],
  install : false,
)

libwink_webgpu = static_library(
  'wink_webgpu',
  join_paths(gex_emdawn_dir, 'webgpu', 'src', 'webgpu.cpp'),
  include_directories : [gex_emdawn_include],
  install : false,
)
