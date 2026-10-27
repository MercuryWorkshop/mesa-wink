#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build pinned Skia DM with test-only Wink platform adapters."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MESA = HERE.parents[2]


def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def patch(source, name, before, after):
    path = source / name
    content = path.read_text()
    if after in content:
        return
    if content.count(before) != 1:
        raise RuntimeError(f'{name}: patch anchor changed')
    path.write_text(content.replace(before, after))


def checkout(out):
    source = out / 'deps/skia'
    pin = json.loads((HERE / 'pins.json').read_text())['skia']
    if not source.exists():
        run(['git', 'clone', '--no-checkout', '--filter=blob:none', pin['url'], source])
        run(['git', '-C', source, 'fetch', '--depth=1', 'origin', pin['commit']])
        run(['git', '-C', source, 'checkout', '--detach', pin['commit']])
    actual = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin['commit']:
        raise RuntimeError(f'{source}: expected {pin["commit"]}, found {actual}')
    run([sys.executable, 'tools/git-sync-deps'], cwd=source)
    if not (source / 'bin/gn').exists():
        run([sys.executable, 'bin/fetch-gn'], cwd=source)
    return source


def adapt(source):
    # Only modify the disposable pinned dependency, never the user's Skia tree.
    target = source / 'tools/ganesh/gl/wink'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / 'GLTestContext_wink.cpp', target)
    patch(source, 'BUILD.gn',
          '"tools/ganesh/gl/mac/CreatePlatformGLTestContext_mac.cpp"',
          '"tools/ganesh/gl/wink/GLTestContext_wink.cpp"')
    patch(source, 'BUILD.gn', '"tools/ganesh/gl/none/CreatePlatformGLTestContext_none.cpp"',
          '"tools/ganesh/gl/wink/GLTestContext_wink.cpp"')
    # DM does not serialize unit-test successes in dm.json. Emit task lifecycle
    # records in dry runs too, so the runner can check inventory completeness.
    for event, signature in (
        ('DONE', 'static void done(const char* config, const char* src, const char* srcOptions, const char* name) {'),
        ('START', 'static void start(const char* config, const char* src, const char* srcOptions, const char* name) {'),
    ):
        patch(source, 'dm/DM.cpp', signature, signature +
              f'\n    fprintf(stderr, "WINK_SKIA_{event}\\t%s\\t%s\\t%s\\t%s\\n", config, src, srcOptions, name);')
    patch(source, 'dm/DM.cpp', '            if (result.isSkip()) {',
          '''            if (result.isSkip()) {
                fprintf(stderr, "WINK_SKIA_SKIP\\t%s\\t%s\\t%s\\t%s\\t%s\\n",
                        task.sink.tag.c_str(), task.src.tag.c_str(),
                        task.src.options.c_str(), name.c_str(), result.c_str());''')
    patch(source, 'tools/ganesh/GrContextFactory.cpp',
          '    // (shareIndex != 0) -> (shareContext != nullptr)',
          '''    if (const char* route = getenv("WINK_SKIA_ROUTE")) {
        if ((!strcmp(route, "gles") && type != ContextType::kGLES) ||
            (!strcmp(route, "vk") && type != ContextType::kVulkan)) return {};
    }
    // (shareIndex != 0) -> (shareContext != nullptr)''')
    patch(source, 'tools/ganesh/GrContextFactory.cpp',
          '    return this->getContextInfoInternal(type, overrides, nullptr, 0);',
          '''    auto result = this->getContextInfoInternal(type, overrides, nullptr, 0);
    if (result.directContext()) SkDebugf("WINK_SKIA_CONTEXT\\tganesh\\n");
    return result;''')
    patch(source, 'tools/graphite/ContextFactory.cpp',
          'ContextInfo ContextFactory::getContextInfo(skgpu::ContextType type) {',
          '''ContextInfo ContextFactory::getContextInfo(skgpu::ContextType type) {
    if (const char* route = getenv("WINK_SKIA_ROUTE")) {
        if ((!strcmp(route, "grvk") && type != skgpu::ContextType::kVulkan) ||
            (!strcmp(route, "grdawn_gles") && type != skgpu::ContextType::kDawn_OpenGLES)) return {};
    }''')
    patch(source, 'tools/graphite/ContextFactory.cpp',
          '    return ContextInfo{owned.fTestContext.get(), owned.fContext.get()};',
          '''    if (owned.fContext) SkDebugf("WINK_SKIA_CONTEXT\\tgraphite\\n");
    return ContextInfo{owned.fTestContext.get(), owned.fContext.get()};''')
    # Opt in to portability enumeration when the selected loader offers it.
    patch(source, 'tools/gpu/vk/VkTestUtils.cpp',
          '    // Set up VkInstanceCreateInfo',
          '''    bool winkPortability = false;
    for (const auto& ext : availableInstExts) {
        if (!strcmp(ext.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
            if (std::find_if(enabledInstExtNames.begin(), enabledInstExtNames.end(),
                    [](const char* n) { return !strcmp(n, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME); })
                    == enabledInstExtNames.end()) {
                enabledInstExtNames.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
            }
            winkPortability = true;
        }
    }
    // Set up VkInstanceCreateInfo''')
    patch(source, 'tools/gpu/vk/VkTestUtils.cpp', '    bool hasDebugExtension = false;',
          '''    if (winkPortability) createInstanceInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    bool hasDebugExtension = false;''')
    patch(source, 'tools/gpu/vk/VkTestUtils.cpp',
          '    grVkGetPhysicalDeviceProperties(physDev, &physDeviceProperties);',
          '''    grVkGetPhysicalDeviceProperties(physDev, &physDeviceProperties);
    SkDebugf("WINK_SKIA_BACKEND\\tvulkan\\t%s\\n", physDeviceProperties.deviceName);
    if (const char* expected = getenv("WINK_SKIA_EXPECT_VULKAN")) {
        if (!strstr(physDeviceProperties.deviceName, expected)) {
            SK_ABORT("Unexpected Vulkan device: %s", physDeviceProperties.deviceName);
        }
    }''')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=MESA / 'build/skia')
    p.add_argument('--jobs', type=int, default=6)
    p.add_argument('--library', type=Path, default=MESA / 'build/wink-mac/src/gallium/targets/wink-wasm/libwink.dylib')
    p.add_argument('--vk-loader', type=Path, default=Path('/opt/homebrew/opt/vulkan-loader/lib/libvulkan.dylib'))
    p.add_argument('--no-sync', action='store_true', help='Use an already synchronized pinned checkout')
    a = p.parse_args()
    a.out = a.out.resolve()
    source = a.out / 'deps/skia' if a.no_sync else checkout(a.out)
    pin = json.loads((HERE / 'pins.json').read_text())['skia']['commit']
    if subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() != pin:
        p.error('Skia revision does not match pins.json')
    if not a.library.is_file() or not a.vk_loader.is_file():
        p.error('Build native Wink and install a Vulkan loader first')
    adapt(source)
    args = {
        'is_debug': False, 'is_component_build': False,
        'skia_enable_ganesh': True, 'skia_enable_graphite': True,
        'skia_use_gl': True, 'skia_gl_standard': 'gles',
        'skia_use_vulkan': True, 'skia_use_dawn': False, 'skia_use_metal': False,
        'skia_use_partition_alloc': False,
        'skia_use_system_expat': False, 'skia_use_system_libjpeg_turbo': False,
        'skia_use_system_libpng': False, 'skia_use_system_libwebp': False,
        'skia_use_system_zlib': False, 'skia_use_system_icu': False,
        'skia_use_system_harfbuzz': False,
        'extra_cflags': [f'-I{MESA / "src/gallium/targets/wink-wasm"}',
                         f'-DSK_GPU_TOOLS_VK_LIBRARY_NAME={a.vk_loader.resolve()}'],
        'extra_ldflags': [str(a.library.resolve()), f'-Wl,-rpath,{a.library.resolve().parent}'],
    }
    build = a.out / 'native'
    build.mkdir(parents=True, exist_ok=True)
    gn_args = '\n'.join(f'{k} = {json.dumps(v)}' for k, v in args.items()) + '\n'
    (build / 'args.gn').write_text(gn_args)
    run([source / 'bin/gn', 'gen', build], cwd=source)
    run(['ninja', '-C', build, '-j', a.jobs, 'dm', 'skdiff'], cwd=source)
    (build / 'wink-build.json').write_text(json.dumps({
        'schema': 1, 'skia': pin, 'gn_args': args,
        'wink': subprocess.check_output(['git', '-C', MESA, 'rev-parse', 'HEAD'], text=True).strip(),
        'routes': ['gles', 'vk', 'grvk'],
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
