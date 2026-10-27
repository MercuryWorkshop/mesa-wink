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
CORRECTNESS_FIXES = ['graphite-vulkan-compressed-host-copy-row-texels-v1',
                     'freetype-wasm-svg-module-registration-v1',
                     'dm-image-extension-boundary-v1',
                     'ganesh-framebuffer-fetch-destination-swizzle-v1']


def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def patch(source, name, before, after):
    path = source / name
    content = path.read_text()
    # Another platform may already contain the same replacement. Its
    # presence does not mean this platform's old anchor was replaced too.
    if after in content and (before in after or before not in content):
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
    # Framebuffer fetch returns physical attachment channels. Convert back to
    # the logical surface color before shader blending, just as a destination
    # texture's sampler swizzle does. Alpha_8 backed by R8 writes a000 and must
    # read 000r; reading attachment alpha instead blends against constant one.
    patch(source, 'src/gpu/ganesh/GrXferProcessor.cpp',
          '        const char* dstColor = fragBuilder->dstColor();',
          '''        const char* dstColor = fragBuilder->dstColor();
        if (!args.fDstTextureSamplerHandle.isValid() &&
            args.fShaderCaps->fFBFetchSupport && args.fWriteSwizzle) {
            fragBuilder->codeAppendf("half4 _swizzledDstColor = %s.%s;", dstColor,
                                    args.fWriteSwizzle.invert().asString().c_str());
            dstColor = "_swizzledDstColor";
        }''')
    # SkOSFile::Iter matches suffixes literally: "bmp" also found ".wbmp",
    # causing duplicate task identities and overwriting image artifacts.
    patch(source, 'tools/flags/CommonFlagsImages.cpp',
          '                SkOSFile::Iter it(flag, ext);',
          '''                SkString suffix(".");
                suffix.append(ext);
                SkOSFile::Iter it(flag, suffix.c_str());''')
    # This Wasm module list omitted the SVG renderer despite compiling SVG
    # support. FreeType's SVG glyph metrics then dereferenced a missing module.
    patch(source, 'third_party/freetype2/include/freetype-no-type1/freetype/config/ftmodule.h',
          '//FT_USE_MODULE( FT_Renderer_Class, ft_svg_renderer_class )',
          '''#if defined(FT_CONFIG_OPTION_SVG)
FT_USE_MODULE( FT_Renderer_Class, ft_svg_renderer_class )
#endif''')
    # Unlike buffer uploads, this pinned host-copy path counts blocks rather
    # than texels. VK_EXT_host_image_copy uses the same texel row-length rule.
    patch(source, 'src/gpu/graphite/vk/VulkanTexture.cpp',
          '        copyRegion.memoryRowLength = levels[currentMipLevel].fRowBytes / bpp;',
          '''        copyRegion.memoryRowLength = (levels[currentMipLevel].fRowBytes / bpp) *
                CompressedDimensions(TextureFormatCompressionType(format), {1, 1}).width();''')
    target = source / 'tools/ganesh/gl/wink'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / 'GLTestContext_wink.cpp', target)
    # Reuse Skia's EGL context implementation for independent ANGLE references.
    # Keep it in a separate translation unit and behind an opt-in build define.
    egl = (source / 'tools/ganesh/gl/egl/CreatePlatformGLTestContext_egl.cpp').read_text()
    egl = egl.replace('CreatePlatformGLTestContext(', 'CreateWinkReferenceGLTestContext(')
    anchor = '        gl = GrGLInterfaces::MakeEGL();'
    if egl.count(anchor) != 1:
        raise RuntimeError('reference EGL factory anchor changed')
    egl = egl.replace(anchor, '''        gl = GrGLMakeAssembledInterface(nullptr, [](void*, const char* name) {
            return reinterpret_cast<GrGLFuncPtr>(eglGetProcAddress(name));
        });
        if (gl) SkDebugf("WINK_SKIA_BACKEND\\tgles\\t%s\\n", gl->fFunctions.fGetString(GR_GL_RENDERER));''')
    (target / 'GLTestContext_reference.cpp').write_text(
        '#ifdef WINK_SKIA_ANGLE_REFERENCE\n#include "include/gpu/ganesh/gl/GrGLAssembleInterface.h"\n' + egl + '\n#endif\n')
    patch(source, 'BUILD.gn', '      if (skia_use_gl && !is_canvaskit) {',
          '''      if (skia_use_gl && !is_canvaskit) {
        sources += [ "tools/ganesh/gl/wink/GLTestContext_reference.cpp" ]''')
    patch(source, 'gn/skia.gni', 'declare_args() {\n  is_canvaskit = false',
          'declare_args() {\n  wink_skia_angle_dir = ""\n  is_canvaskit = false')
    patch(source, 'BUILD.gn', '      defines = []\n      if (skia_enable_discrete_gpu)',
          '''      defines = []
      if (wink_skia_angle_dir != "") {
        defines += [ "WINK_SKIA_ANGLE_REFERENCE=1" ]
        include_dirs = [ "third_party/externals/angle2/include" ]
      }
      if (skia_enable_discrete_gpu)''')
    patch(source, 'BUILD.gn',
          '"tools/ganesh/gl/mac/CreatePlatformGLTestContext_mac.cpp"',
          '"tools/ganesh/gl/wink/GLTestContext_wink.cpp"')
    patch(source, 'BUILD.gn',
          '''        } else if (is_wasm) {
          # TODO: replace with webgl test context
          sources +=
              [ "tools/ganesh/gl/none/CreatePlatformGLTestContext_none.cpp" ]''',
          '''        } else if (is_wasm) {
          sources +=
              [ "tools/ganesh/gl/wink/GLTestContext_wink.cpp" ]''')
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
    # DM counts every source/sink pair in its progress total, then silently
    # vetoes incompatible pairs (most codec variants, for GPU sinks). Record
    # these separately so they cannot be mistaken for exercised GPU work.
    patch(source, 'dm/DM.cpp',
          '            if (src->veto(sink->flags()) ||',
          '''            const bool sourceVeto = src->veto(sink->flags());
            if (sourceVeto ||''')
    patch(source, 'dm/DM.cpp',
          '''                SkAutoSpinlock lock(gMutex);
                gPending--;
                continue;''',
          '''                fprintf(stderr, "WINK_SKIA_%s\\t%s\\t%s\\t%s\\t%s\\n",
                        sourceVeto ? "VETO" : "FILTER", sink.tag.c_str(), src.tag.c_str(),
                        src.options.c_str(), src->name().c_str());
                SkAutoSpinlock lock(gMutex);
                gPending--;
                continue;''')
    # Some GPU-labelled units explicitly use a mock to test CPU bookkeeping.
    # Permit it without attributing that work to the selected hardware route.
    # Upgrade earlier harness patches in the disposable dependency checkout.
    factory = source / 'tools/ganesh/GrContextFactory.cpp'
    content = factory.read_text()
    content = content.replace(
        '    if (const char* route = getenv("WINK_SKIA_ROUTE")) {',
        '    if (const char* route = type == ContextType::kMock ? nullptr : getenv("WINK_SKIA_ROUTE")) {')
    content = content.replace(
        '    if (result.directContext()) SkDebugf("WINK_SKIA_CONTEXT\\tganesh\\n");',
        '    if (result.directContext() && type != ContextType::kMock) SkDebugf("WINK_SKIA_CONTEXT\\tganesh\\n");')
    factory.write_text(content)
    patch(source, 'tools/ganesh/GrContextFactory.cpp',
          '    // (shareIndex != 0) -> (shareContext != nullptr)',
          '''    if (const char* route = type == ContextType::kMock ? nullptr : getenv("WINK_SKIA_ROUTE")) {
        if ((!strcmp(route, "gles") && type != ContextType::kGLES) ||
            (!strcmp(route, "vk") && type != ContextType::kVulkan)) return {};
    }
    // (shareIndex != 0) -> (shareContext != nullptr)''')
    patch(source, 'tools/ganesh/GrContextFactory.cpp',
          '    return this->getContextInfoInternal(type, overrides, nullptr, 0);',
          '''    auto result = this->getContextInfoInternal(type, overrides, nullptr, 0);
    if (result.directContext() && type != ContextType::kMock) SkDebugf("WINK_SKIA_CONTEXT\\tganesh\\n");
    return result;''')
    # Upstream DM deliberately skips GLES on desktop hosts. These are GLES
    # tests even on a Mac; selecting only a GLES sink does not change units.
    patch(source, 'dm/DMGpuTestProcs.cpp',
          '    static constexpr auto kNativeGLType = skgpu::ContextType::kGL;',
          '''    const auto kNativeGLType = getenv("WINK_SKIA_ROUTE") &&
            !strcmp(getenv("WINK_SKIA_ROUTE"), "gles")
            ? skgpu::ContextType::kGLES : skgpu::ContextType::kGL;''')
    patch(source, 'dm/DMGpuTestProcs.cpp',
          '''        GrContextFactory factory(options);
        ContextInfo ctxInfo = factory.getContextInfo(contextType);
        if (filter && !(*filter)(contextType)) {
            continue;
        }''',
          '''        if (filter && !(*filter)(contextType)) {
            continue;
        }
        GrContextFactory factory(options);
        ContextInfo ctxInfo = factory.getContextInfo(contextType);''')
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
    dawn_dir = source / 'tools/graphite/dawn/wink'
    dawn_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / 'DawnWinkEGL.cpp', dawn_dir)
    patch(source, 'BUILD.gn', '            "tools/graphite/dawn/GraphiteDawnTestContext.cpp",',
          '''            "tools/graphite/dawn/GraphiteDawnTestContext.cpp",
            "tools/graphite/dawn/wink/DawnWinkEGL.cpp",''')
    patch(source, 'BUILD.gn', '''      if (skia_use_dawn) {
        public_deps += [ "//third_party/dawn" ]''',
          '''      if (skia_use_dawn) {
        if (!defined(include_dirs)) { include_dirs = [] }
        include_dirs += [ "third_party/externals/angle2/include" ]
        public_deps += [ "//third_party/dawn" ]''')
    patch(source, 'tools/graphite/dawn/GraphiteDawnTestContext.cpp',
          '#include "dawn/dawn_proc.h"',
          '#include "dawn/dawn_proc.h"\n#include "dawn/native/OpenGLBackend.h"')
    patch(source, 'tools/graphite/dawn/GraphiteDawnTestContext.cpp',
          'namespace skiatest::graphite {',
          '''namespace skiatest::graphite {
void (*winkSkiaEGLGetProc(const char* name))();''')
    patch(source, 'tools/graphite/dawn/GraphiteDawnTestContext.cpp',
          '    options.nextInChain = &togglesDesc;',
          '''    options.nextInChain = &togglesDesc;
    dawn::native::opengl::RequestAdapterOptionsGetGLProc winkGL;
    if (backend == wgpu::BackendType::OpenGLES && getenv("WINK_SKIA_ROUTE") &&
        !strcmp(getenv("WINK_SKIA_ROUTE"), "grdawn_gles")) {
        winkGL.getProc = winkSkiaEGLGetProc;
        winkGL.nextInChain = options.nextInChain;
        options.nextInChain = &winkGL;
        options.backendType = wgpu::BackendType::OpenGLES;
    }''')
    patch(source, 'tools/graphite/dawn/GraphiteDawnTestContext.cpp',
          '    desc.nextInChain           = &togglesDesc;',
          '''    desc.nextInChain           = &togglesDesc;
    std::vector<const char*> winkDeviceToggles;
    if (backend == wgpu::BackendType::OpenGLES) {
        for (size_t i = 0; i < togglesDesc.enabledToggleCount; ++i) {
            winkDeviceToggles.push_back(togglesDesc.enabledToggles[i]);
        }
        // Graphite compiles pipelines on its executor. Dawn must release
        // the EGL context between calls before moving it to another thread.
        winkDeviceToggles.push_back("gl_allow_context_on_multi_threads");
        togglesDesc.enabledToggleCount = winkDeviceToggles.size();
        togglesDesc.enabledToggles = winkDeviceToggles.data();
    }''')
    # Skia's outer Dawn must not export weak Tint C++ symbols that could
    # coalesce with Wink's independently pinned inner Dawn/Tint library.
    patch(source, 'third_party/dawn/build_dawn.py',
          '      "-DDAWN_BUILD_MONOLITHIC_LIBRARY=OFF",',
          '''      "-DDAWN_BUILD_MONOLITHIC_LIBRARY=OFF",
      "-DCMAKE_C_VISIBILITY_PRESET=hidden",
      "-DCMAKE_CXX_VISIBILITY_PRESET=hidden",
      "-DCMAKE_VISIBILITY_INLINES_HIDDEN=ON",''')
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
    p.add_argument('--angle', type=Path, help='Optional ANGLE output directory for independent GLES references')
    p.add_argument('--dawn-gles', action='store_true', help='Build Graphite through an outer Dawn GLES backend into native-dawn/')
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
    if a.angle:
        angle = a.angle.resolve()
        for name in ('libEGL.dylib', 'libGLESv2.dylib'):
            if not (angle / name).is_file():
                p.error(f'missing {angle / name}; run build-angle.py first')
        args['wink_skia_angle_dir'] = str(angle)
        args['extra_ldflags'] += [str(angle / 'libEGL.dylib'), f'-Wl,-rpath,{angle}']
    if a.dawn_gles:
        args.update(skia_use_dawn=True, dawn_enable_opengles=True, dawn_enable_metal=False,
                    dawn_enable_vulkan=False, dawn_enable_d3d11=False, dawn_enable_d3d12=False)
    build = a.out / ('native-dawn' if a.dawn_gles else 'native')
    build.mkdir(parents=True, exist_ok=True)
    gn_args = '\n'.join(f'{k} = {json.dumps(v)}' for k, v in args.items()) + '\n'
    (build / 'args.gn').write_text(gn_args)
    run([source / 'bin/gn', 'gen', build], cwd=source)
    run(['ninja', '-C', build, '-j', a.jobs, 'dm', 'skdiff'], cwd=source)
    (build / 'wink-build.json').write_text(json.dumps({
        'schema': 1, 'skia': pin, 'gn_args': args, 'skia_fixes': CORRECTNESS_FIXES,
        'wink': subprocess.check_output(['git', '-C', MESA, 'rev-parse', 'HEAD'], text=True).strip(),
        'routes': ['gles', 'vk', 'grvk'] + (['grdawn_gles'] if a.dawn_gles else []),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
