#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build Skia DM against packaged Wink ST or MT, with JSPI entrypoints."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('skia_build', HERE / 'build.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
MESA = base.MESA


def adapt(source):
    base.adapt(source)
    target = source / 'tools/ganesh/gl/wink'
    shutil.copy2(HERE / 'browser_main.cpp', target)
    base.patch(source, 'BUILD.gn', '      test_app("dm") {',
               '      test_app("dm") {\n        if (is_wasm) { output_extension = "js" }')
    base.patch(source, 'BUILD.gn', '          "dm/DM.cpp",',
               '          "dm/DM.cpp",\n          "tools/ganesh/gl/wink/browser_main.cpp",')
    base.patch(source, 'dm/DM.cpp', '#if defined(SK_BUILD_FOR_IOS)\nextern "C" int skia_ios_main',
               '#if defined(__EMSCRIPTEN__)\nextern "C" int wink_skia_dm_main(int argc, char** argv) {\n#elif defined(SK_BUILD_FOR_IOS)\nextern "C" int skia_ios_main')
    # Browser apps resolve Vulkan directly, without dlopen or an ICD loader.
    signature = 'bool LoadVkLibraryAndGetProcAddrFuncs(PFN_vkGetInstanceProcAddr* instProc) {'
    base.patch(source, 'tools/gpu/vk/VkTestUtils.cpp', signature, signature + '''
#ifdef __EMSCRIPTEN__
    extern PFN_vkVoidFunction wink_skia_vk_proc(VkInstance, const char*);
    *instProc = wink_skia_vk_proc;
    return true;
#endif''')
    # Reuse the configured Emscripten, never activate Skia's different DEPS SDK.
    base.patch(source, 'gn/toolchain/BUILD.gn',
               '    if (!is_canvaskit) {\n      deps = [ "//gn/toolchain/emsdk:activate',
               '    if (false) {\n      deps = [ "//gn/toolchain/emsdk:activate')
    base.patch(source, 'gn/skia/BUILD.gn', '  extra_asmflags = []',
               '  wink_skia_variant = ""\n  extra_asmflags = []')
    base.patch(source, 'gn/skia/BUILD.gn', '    if (_supports_unsafe_buffers_mappings) {',
               '    if (_supports_unsafe_buffers_mappings && !is_wasm) {')
    base.patch(source, 'gn/skia/BUILD.gn', '    cflags_cc += [ "-fvisibility-inlines-hidden" ]',
               '    if (is_wasm) { cflags -= [ "-fPIC" ] }\n    cflags_cc += [ "-fvisibility-inlines-hidden" ]')
    base.patch(source, 'gn/skia/BUILD.gn', '        "-pthread",\n      ]\n      cflags = common',
               '      ]\n      if (wink_skia_variant == "mt") { common += [ "-pthread" ] }\n      cflags = common')
    base.patch(source, 'gn/skia/BUILD.gn', '        "-sSTACK_SIZE=1048576",  # 1 MB',
               '        "-sSTACK_SIZE=16777216",  # Wink/Tint JSPI stack')
    base.patch(source, 'gn/skia/BUILD.gn', '        "-sPROXY_TO_PTHREAD=1",\n        "-sEXIT_RUNTIME=1",',
               '        "--no-entry",\n        "-sEXIT_RUNTIME=0",')
    base.patch(source, 'gn/skia/BUILD.gn',
               '        "-sEXPORTED_RUNTIME_METHODS=[\'FS\', \'callMain\']",',
               '        "-sEXPORTED_RUNTIME_METHODS=[\'FS\', \'HEAPU8\', \'HEAPU32\', \'ENV\']",')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('target', choices=['st', 'mt'])
    p.add_argument('--out', type=Path, default=MESA / 'build/skia')
    p.add_argument('--emsdk', type=Path, default=Path.home() / '.cache/mesa-wink/emsdk')
    p.add_argument('--jobs', type=int, default=6)
    a = p.parse_args()
    source = a.out.resolve() / 'deps/skia'
    pin = json.loads((HERE / 'pins.json').read_text())['skia']['commit']
    if subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() != pin:
        p.error('Run build.py to synchronize the pinned Skia checkout first')
    package = MESA / 'build/wink/out' / a.target
    if not (package / 'manifest.json').is_file():
        p.error('Build and package Wink first')
    adapt(source)
    flags = ['-fwasm-exceptions', '-sSUPPORT_LONGJMP=wasm',
             f'-I{MESA / "src/gallium/targets/wink-wasm"}']
    if a.target == 'mt':
        flags += ['-pthread']
    exports = ['_wink_skia_main', '_malloc', '_free', '_wink_setenv']
    if a.target == 'mt':
        exports += ['_wink_skia_start', '_wink_skia_worker_body']
    link = [str(package / 'lib' / name) for name in ('libwink.a', 'libwink_webgpu.a', 'libwink_tint.a')]
    link += (package / 'wink.rsp').read_text().splitlines()
    link += [f'--js-library={HERE / "library_skia.js"}', f'--post-js={HERE / "publish.js"}',
             '-sEXPORTED_FUNCTIONS=' + json.dumps(exports),
             '-sEXPORTED_RUNTIME_METHODS=FS,HEAPU8,HEAPU32,ENV',
             '-sENVIRONMENT=web,worker', '-sINITIAL_MEMORY=256MB', '-sASSERTIONS=1']
    if a.target == 'mt':
        link += ['-sPTHREAD_POOL_SIZE=4']
    args = dict(target_cpu='wasm', skia_emsdk_dir=str(a.emsdk.resolve()),
                wink_skia_variant=a.target, is_debug=False, skia_use_gl=True,
                skia_gl_standard='gles', skia_enable_ganesh=True, skia_enable_graphite=True,
                skia_use_vulkan=True, skia_use_dawn=False, skia_use_partition_alloc=False,
                skia_use_freetype=True, skia_enable_fontmgr_custom_embedded=True,
                skia_use_system_freetype2=False,
                skia_use_system_libpng=False, skia_use_system_libjpeg_turbo=False,
                skia_use_system_libwebp=False, skia_use_system_zlib=False,
                skia_use_system_expat=False, skia_use_system_harfbuzz=False,
                skia_use_icu=True, skia_use_system_icu=False,
                extra_cflags=flags, extra_ldflags=link)
    build = a.out.resolve() / a.target
    build.mkdir(parents=True, exist_ok=True)
    (build / 'args.gn').write_text('\n'.join(f'{k} = {json.dumps(v)}' for k, v in args.items()) + '\n')
    base.run([source / 'bin/gn', 'gen', build], cwd=source)
    base.run(['ninja', '-C', build, '-j', a.jobs, 'dm'], cwd=source)
    shutil.copy2(HERE / 'browser.html', build / 'index.html')


if __name__ == '__main__':
    main()
