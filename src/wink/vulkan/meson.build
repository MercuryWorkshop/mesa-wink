# Copyright 2026 wink contributors
# SPDX-License-Identifier: MIT
#
# wvk: Vulkan on WebGPU. Natively a Vulkan ICD over Dawn
# (-Dgex-dawn-native-dir); for the browser a static library for libwink,
# whose vk_icdGetInstanceProcAddr an application calls directly.

wvk_emdawn_dir = get_option('gex-emdawn-dir')
if wvk_emdawn_dir == ''
  error('the wink Vulkan driver needs -Dgex-emdawn-dir: a directory with webgpu/include/webgpu/webgpu.h')
endif
wvk_webgpu_include = include_directories(join_paths(wvk_emdawn_dir, 'webgpu', 'include'))
wvk_is_wasm = host_machine.system() == 'emscripten'

# The window systems a native build presents to: Metal on macOS, where
# Dawn always has it; elsewhere Mesa's own -Dplatforms.
wvk_c_args = []
wvk_wsi_deps = []
if host_machine.system() == 'darwin'
  wvk_c_args += '-DVK_USE_PLATFORM_METAL_EXT'
elif not wvk_is_wasm
  wvk_wsi_deps += idep_vulkan_wsi_defines
endif

wvk_dawn_deps = []
if not wvk_is_wasm
  wvk_dawn_dir = get_option('gex-dawn-native-dir')
  if wvk_dawn_dir == ''
    error('a native wink Vulkan driver needs -Dgex-dawn-native-dir: where libwebgpu_dawn is')
  endif
  wvk_dawn_deps += cc.find_library('webgpu_dawn', dirs : wvk_dawn_dir)
endif

wvk_dispatch_trampolines = custom_target(
  'wvk_dispatch_trampolines',
  input : [files('util/wvk_dispatch_trampolines_gen.py'), vk_api_xml],
  output : ['wvk_dispatch_trampolines.c', 'wvk_dispatch_trampolines.h'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@',
    '--out-c', '@OUTPUT0@', '--out-h', '@OUTPUT1@',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_dispatch_trampolines_gen_depend_files,
)

wvk_entrypoints = custom_target(
  'wvk_entrypoints',
  input : [vk_entrypoints_gen, vk_api_xml],
  output : ['wvk_entrypoints.h', 'wvk_entrypoints.c'],
  command : [
    prog_python, '@INPUT0@', '--xml', '@INPUT1@', '--proto', '--weak',
    '--out-h', '@OUTPUT0@', '--out-c', '@OUTPUT1@', '--prefix', 'wvk',
    '--beta', with_vulkan_beta.to_string()
  ],
  depend_files : vk_entrypoints_gen_depend_files,
)

wvk_files = files(
  'wvk_buffer.c',
  'wvk_cmd_buffer.c',
  'wvk_cmd_copy.c',
  'wvk_cmd_draw.c',
  'wvk_cmd_meta.c',
  'wvk_cmd_render.c',
  'wvk_device.c',
  'wvk_descriptor_set.c',
  'wvk_descriptor_set_layout.c',
  'wvk_device_memory.c',
  'wvk_format.c',
  'wvk_image.c',
  'wvk_instance.c',
  'wvk_physical_device.c',
  'wvk_query.c',
  'wvk_queue.c',
  'wvk_sampler.c',
  'wvk_shader.c',
  'wvk_sync.c',
  'wvk_wsi.c',
)
if wvk_is_wasm
  error('the browser half of the wink Vulkan driver is not written yet')
else
  wvk_files += files('wvk_platform_native.c')
endif

libwvk = static_library(
  'wvk',
  [wvk_files, wvk_entrypoints, wvk_dispatch_trampolines, sha1_h],
  include_directories : [inc_include, inc_src, wvk_webgpu_include],
  dependencies : [
    idep_mesautil,
    idep_vulkan_runtime,
    idep_vulkan_util,
    idep_nir,
    idep_wink_compiler,
    wvk_wsi_deps,
  ],
  c_args : [c_msvc_compat_args, no_override_init_args, wvk_c_args],
  gnu_symbol_visibility : 'hidden',
)

if not wvk_is_wasm
  wvk_link_args = [ld_args_build_id]
  if host_machine.system() == 'darwin'
    # The runtime's weak entrypoint tables leave symbols for the loader.
    wvk_link_args += ['-Wl,-undefined,dynamic_lookup', '-lobjc']
  else
    wvk_link_args += [ld_args_bsymbolic, ld_args_gc_sections]
  endif

  libvulkan_wink = shared_library(
    'vulkan_wink',
    link_whole : [libwvk],
    # What mesautil and the compiler use, which link_whole does not bring.
    dependencies : [wvk_dawn_deps, wink_tint, dep_zlib, dep_zstd, dep_spirv_tools],
    link_args : wvk_link_args,
    gnu_symbol_visibility : 'hidden',
    install : true,
  )

  wvk_icd_file_name = libname_prefix + 'vulkan_wink.' + libname_suffix

  custom_target(
    'wink_icd',
    input : [vk_icd_gen, vk_api_xml],
    output : 'wink_icd.' + vulkan_manifest_suffix,
    command : [
      prog_python, '@INPUT0@',
      '--api-version', '1.2', '--xml', '@INPUT1@',
      '--icd-lib-path', vulkan_icd_lib_path,
      '--icd-filename', wvk_icd_file_name,
      '--out', '@OUTPUT@',
    ],
    build_by_default : true,
    install_dir : with_vulkan_icd_dir,
    install_tag : 'runtime',
    install : true,
  )

  # For running from the build tree: VK_DRIVER_FILES=<this>.
  wvk_devenv_icd = custom_target(
    'wink_devenv_icd',
    input : [vk_icd_gen, vk_api_xml],
    output : 'wink_devenv_icd.@0@.json'.format(host_machine.cpu()),
    command : [
      prog_python, '@INPUT0@',
      '--api-version', '1.2', '--xml', '@INPUT1@',
      '--icd-lib-path', meson.current_build_dir(),
      '--icd-filename', wvk_icd_file_name,
      '--out', '@OUTPUT@',
    ],
    build_by_default : true,
  )
  devenv.append('VK_DRIVER_FILES', wvk_devenv_icd.full_path())
endif
