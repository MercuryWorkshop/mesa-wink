#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build Skia-pinned ANGLE for independent GLES references."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('skia_build', HERE / 'build.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def adapt_vulkan(source, loader):
    # Skia's GN tree has no Chromium build/ directory. Keep those imports
    # behind ANGLE's existing standalone-build switch.
    base.patch(source, 'third_party/externals/angle2/src/libANGLE/renderer/vulkan/BUILD.gn',
               '''import("//build/config/dcheck_always_on.gni")
import("//build_overrides/swiftshader.gni")
import("vulkan_backend.gni")''',
               '''import("vulkan_backend.gni")

if (angle_has_build) {
  import("//build/config/dcheck_always_on.gni")
  import("//build_overrides/swiftshader.gni")
}''')
    base.patch(source, 'third_party/externals/angle2/src/common/vulkan/BUILD.gn',
               '''import("//build_overrides/swiftshader.gni")
import("../../../gni/angle.gni")''',
               '''import("../../../gni/angle.gni")
if (angle_has_build) {
  import("//build_overrides/swiftshader.gni")
}''')
    base.patch(source, 'build_overrides/angle.gni',
               'angle_spirv_tools_dir = "//third_party/externals/spirv-tools"',
               '''angle_spirv_tools_dir = "//third_party/externals/spirv-tools"
angle_vulkan_memory_allocator_dir = "//third_party/externals/vulkanmemoryallocator"''')
    allocator = source / 'third_party/externals/vulkanmemoryallocator'
    (allocator / 'BUILD.gn').write_text('''source_set("vulkanmemoryallocator") {
  sources = [ "wink_vma.cpp" ]
  include_dirs = [ "include", "../vulkan-headers/include" ]
}
''')
    (allocator / 'wink_vma.cpp').write_text('''#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#define VMA_NULLABLE
#define VMA_NOT_NULL
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>
''')
    system = source / 'third_party/wink_system_vulkan_loader'
    system.mkdir(parents=True, exist_ok=True)
    (system / 'BUILD.gn').write_text('''config("system_vulkan") {
  libs = [ ''' + json.dumps(str(loader.resolve())) + ''' ]
  if (is_mac) {
    frameworks = [ "IOSurface.framework", "QuartzCore.framework" ]
  }
}
group("libvulkan") {
  public_configs = [ ":system_vulkan" ]
}
''')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=base.MESA / 'build/skia')
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--backend', choices=['metal', 'vulkan'], default='metal')
    p.add_argument('--vk-loader', type=Path,
                   default=Path('/opt/homebrew/opt/vulkan-loader/lib/libvulkan.dylib'))
    a = p.parse_args()
    source = a.out.resolve() / 'deps/skia'
    pin = json.loads((HERE / 'pins.json').read_text())['skia']['commit']
    if subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() != pin:
        p.error('Synchronize the pinned Skia checkout with build.py first')
    build = a.out.resolve() / ('angle' if a.backend == 'metal' else 'angle-vulkan')
    build.mkdir(parents=True, exist_ok=True)
    args = dict(is_debug=False, skia_use_angle=True, skia_gl_standard='gles',
                angle_enable_metal=a.backend == 'metal', angle_enable_gl=False,
                angle_enable_vulkan=a.backend == 'vulkan', angle_build_tests=False)
    if a.backend == 'vulkan':
        if not a.vk_loader.is_file():
            p.error(f'missing Vulkan loader: {a.vk_loader}')
        adapt_vulkan(source, a.vk_loader)
        args.update(angle_enable_swiftshader=False, angle_enable_vulkan_validation_layers=False,
                    angle_enable_vulkan_api_dump_layer=False, angle_use_custom_libvulkan=False,
                    angle_build_vulkan_system_info=True,
                    angle_vulkan_headers_dir='//third_party/externals/vulkan-headers',
                    angle_vulkan_loader_dir='//third_party/wink_system_vulkan_loader')
    (build / 'args.gn').write_text('\n'.join(f'{k} = {json.dumps(v)}' for k, v in args.items()) + '\n')
    base.run([source / 'bin/gn', 'gen', build], cwd=source)
    base.run(['ninja', '-C', build, '-j', a.jobs, 'libEGL', 'libGLESv2'], cwd=source)
    (build / 'wink-angle-build.json').write_text(json.dumps(dict(
        schema=1, backend=a.backend, gn_args=args, skia=pin,
        vulkan_loader=str(a.vk_loader.resolve()) if a.backend == 'vulkan' else None,
        vulkan_loader_sha256=hashlib.sha256(a.vk_loader.read_bytes()).hexdigest()
            if a.backend == 'vulkan' else None,
        angle=subprocess.check_output(['git', '-C', source / 'third_party/externals/angle2',
                                       'rev-parse', 'HEAD'], text=True).strip()), indent=2) + '\n')


if __name__ == '__main__':
    main()
