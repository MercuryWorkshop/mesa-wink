#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build pinned Skia DM with test-only Wink platform adapters."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MESA = HERE.parents[2]
CORRECTNESS_FIXES = ['graphite-vulkan-compressed-host-copy-row-texels-v1',
                     'freetype-wasm-svg-module-registration-v1',
                     'dm-image-extension-boundary-v1',
                     'ganesh-framebuffer-fetch-destination-swizzle-v1',
                     'graphite-automatic-image-tiling-v1',
                     'graphite-oversized-image-shader-paint-fallback-v1',
                     'graphite-precompile-random-local-matrix-v1',
                     'graphite-blur-precompile-coverage-v3',
                     'graphite-gradient-precompile-alpha-v1',
                     'graphite-yuv-precompile-alpha-v1',
                     'graphite-blend-precompile-child-context-v1',
                     'graphite-color-text-mask-precompile-v1',
                     'resource-cache-purge-clock-progress-v1',
                     'test-typeface-empty-native-fallback-v1',
                     'graphite-cache-budget-early-exit-v1',
                     'skvx-scalar-logical-not-mask-v1',
                     'gpu-write-pixels-guard-regions-v1',
                     'gradient-texture-direct-half-encoding-v1',
                     'paragraph-font-style-raw-width-test-v1']

# These pinned tests assert host-side layout, geometry, and packing. Keep
# running them with their original GPU-labelled registration, while recording
# their scope separately from hardware coverage.
HOST_UNIT_TESTS = {
    'tests/TestTest.cpp': ('DEF_GANESH_TEST', {'TestGpuFactory'}),
    'tests/GpuRectanizerTest.cpp': ('DEF_GANESH_TEST', {'GpuRectanizer'}),
    'tests/DrawOpAtlasTest.cpp': ('DEF_GANESH_TEST', {'GrDrawOpAtlasConfig_Basic'}),
    'tests/graphite/DrawAtlasTest.cpp': ('DEF_GRAPHITE_TEST', None),
    'tests/graphite/IntersectionTreeTest.cpp': ('DEF_GRAPHITE_TEST', None),
    'tests/graphite/RectTest.cpp': ('DEF_GRAPHITE_TEST', None),
    'tests/graphite/ShapeTest.cpp': ('DEF_GRAPHITE_TEST', None),
    'tests/graphite/TransformTest.cpp': ('DEF_GRAPHITE_TEST', None),
    'tests/graphite/UniformManagerTest.cpp': ('DEF_GRAPHITE_TEST', None),
    'tests/graphite/UniformOffsetCalculatorTest.cpp': ('DEF_GRAPHITE_TEST', None),
}


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


def annotate_host_units(source):
    for name, (macro, allowed) in HOST_UNIT_TESTS.items():
        content = (source / name).read_text()
        pattern = r'(' + macro + r'\(\s*(\w+)\s*,[^{};]*\)\s*\{)'
        matches = [m for m in re.finditer(pattern, content)
                   if allowed is None or m[2] in allowed]
        if not matches or (allowed is not None and {m[2] for m in matches} != allowed):
            raise RuntimeError(f'{name}: host-unit declarations changed')
        for match in matches:
            patch(source, name, match[1], match[1] +
                  f'\n    SkDebugf("WINK_SKIA_HOST_UNIT\\t{match[2]}\\n");')


def annotate_negative_operations(source):
    files = ['tests/graphite/' + name + '.cpp' for name in (
        'RecorderTest', 'RecordingSurfacesTest', 'ImageProviderTest',
        'GraphitePromiseImageTest', 'GraphiteYUVAPromiseImageTest', 'BackendTextureTest')]
    files.append('tests/VkProtectedContextTest.cpp')
    shutil.copy2(HERE / 'expected_diagnostic.h', source / 'tests/WinkExpectedDiagnostic.h')
    for name in files:
        patch(source, name, '#include "tests/Test.h"',
              '#include "tests/Test.h"\n#include "tests/WinkExpectedDiagnostic.h"')

    def expect(expression, operation, enabled='true'):
        return (f'wink_skia_expect_diagnostic({enabled}, "{operation}", '
                f'[&] {{ return {expression}; }})')

    patch(source, files[0], '!insert(o3.get())',
          '!' + expect('insert(o3.get())', 'recording-order'))
    patch(source, files[0],
          'InsertStatus status = context->insertRecording(info); // Should not crash on the null Recording!',
          'InsertStatus status = ' + expect('context->insertRecording(info)', 'null-recording') +
          '; // Should not crash on the null Recording!')

    # The test already checks that rejected replay leaves the target transparent.
    # Also assert its insertion status, and bracket only this insertion.
    patch(source, files[1],
          '    context->insertRecording({recording.get(), surface.get(), replayOffset, replayClip});',
          '''    const char* rejection = nullptr;
    if (canvasMipmapped != targetMipmapped) {
        rejection = "deferred-mipmaps";
    } else if (canvasMipmapped == skgpu::Mipmapped::kYes) {
        if (surfaceSize != recordingSize) rejection = "deferred-dimensions";
        else if (replayOffset != kNoOffset) rejection = "deferred-translation";
        else if (!replayClip.isEmpty()) rejection = "deferred-clip";
    }
    auto status = wink_skia_expect_diagnostic(rejection != nullptr, rejection, [&] {
        return context->insertRecording({recording.get(), surface.get(), replayOffset, replayClip});
    });
    REPORTER_ASSERT(reporter, static_cast<bool>(status) == (rejection == nullptr));''')

    patch(source, files[2],
          '        SkRect rectToDraw = SkRect::MakeWH(kSurfaceSize.width(), kSurfaceSize.height());',
          '''        SkRect rectToDraw = SkRect::MakeWH(kSurfaceSize.width(), kSurfaceSize.height());
        wink_skia_expect_diagnostic(expectedColor == kBackgroundColor, "image-provider", [&] {''')
    patch(source, files[2],
          '''            canvas->drawImageRect(imageToDraw, rectToDraw, sampling);
        }

        if (!surface->readPixels''',
          '''            canvas->drawImageRect(imageToDraw, rectToDraw, sampling);
        }
        });

        if (!surface->readPixels''')

    for name, yuva in ((files[3], False), (files[4], True)):
        for volatile, count in ((False, 4), (True, 6)):
            test = (('Volatile' if volatile else 'NonVolatile') + 'Graphite' +
                    ('YUVA' if yuva else '') + 'PromiseImageFulfillFailureTest')
            path = source / name
            content = path.read_text()
            declaration = re.search(r'\(\s*' + test + ',', content)
            if declaration is None:
                raise RuntimeError(f'{name}: {test} declaration changed')
            start = declaration.start()
            end = content.find('\nDEF_', start)
            if end < 0:
                end = len(content)
            body = content[start:end]
            old = '!context->insertRecording({ recording.get() })'
            new = '!' + expect('context->insertRecording({ recording.get() })',
                               'promise-volatile' if volatile else 'promise-nonvolatile')
            if new not in body:
                if body.count(old) != count:
                    raise RuntimeError(f'{name}: {test} negative insertion count changed')
                path.write_text(content[:start] + body.replace(old, new) + content[end:])

    expression = '''SkSurfaces::WrapBackendTexture(recorder.get(),
                                             texture,
                                             /*colorSpace=*/nullptr,
                                             /*props=*/nullptr)'''
    patch(source, files[5], 'surface = ' + expression + ';',
          'surface = ' + expect(expression, 'nonrenderable-surface') + ';')
    expression = '''helper->recorder()->createBackendTexture({ kSize, kSize },
                                                                         textureInfo)'''
    patch(source, files[6], 'BackendTexture backendTex = ' + expression + ';',
          'BackendTexture backendTex = ' + expect(expression, 'protected-texture',
              '!contextIsProtected && beTexIsProtected') + ';')


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
    annotate_negative_operations(source)
    # This test compares subpixel advances. Its default 0.01-pixel layout
    # rounding erased the valid 0.00390625-pixel difference in pinned Roboto.
    patch(source, 'modules/skparagraph/tests/SkParagraphTest.cpp',
          '''    auto widthOf = [&](const TextStyle& first, const TextStyle& second) {
        ParagraphStyle paragraphStyle;''',
          '''    auto widthOf = [&](const TextStyle& first, const TextStyle& second) {
        ParagraphStyle paragraphStyle;
        paragraphStyle.setApplyRoundingHack(false);''')
    patch(source, 'tools/fonts/FontToolUtils.cpp',
          '#if defined(SK_BUILD_FOR_ANDROID) && defined(SK_FONTMGR_ANDROID_AVAILABLE)\n'
          '#include "include/ports/SkFontMgr_android.h"',
          '#if defined(SK_FONTMGR_ANDROID_AVAILABLE)\n'
          '#include "include/ports/SkFontMgr_android.h"')
    patch(source, 'tools/fonts/FontToolUtils.cpp',
          '#elif defined(SK_FONTMGR_FREETYPE_DIRECTORY_AVAILABLE)',
          '''#elif defined(SK_FONTMGR_ANDROID_AVAILABLE) && defined(SK_TYPEFACE_FACTORY_FREETYPE)
            SkString base = GetResourcePath("");
            base.append("/");
            SkString config = GetResourcePath("wink-fonts.xml");
            SkFontMgr_Android_CustomFonts fonts{
                SkFontMgr_Android_CustomFonts::kOnlyCustom,
                base.c_str(), config.c_str(), nullptr, true};
            mgr = SkFontMgr_New_Android(&fonts, TestFontScanner());
#elif defined(SK_FONTMGR_FREETYPE_DIRECTORY_AVAILABLE)''')
    # Portable references must choose the same default bitmap emoji fixtures
    # as Wasm. Explicit SBIX/CBDT/COLR/SVG test cases keep their own selection.
    for anchor in (
        '#elif defined(SK_BUILD_FOR_MAC) || defined(SK_BUILD_FOR_IOS)\n'
        '        filename = "fonts/planetsbix.ttf";',
        '#elif defined(SK_BUILD_FOR_MAC) || defined(SK_BUILD_FOR_IOS)\n'
        '        sample = EmojiSample(EmojiFontFormat::Sbix);',
    ):
        patch(source, 'tools/fonts/FontToolUtils.cpp', anchor,
              anchor.replace('#elif defined(SK_BUILD_FOR_MAC) || defined(SK_BUILD_FOR_IOS)',
                             '#elif (defined(SK_BUILD_FOR_MAC) || defined(SK_BUILD_FOR_IOS)) && '
                             '!defined(WINK_SKIA_PORTABLE_TEXT)'))
    # Match the browser's text atlas sizing when generating native references.
    # Keep macOS's platform policy available for native application coverage.
    for anchor in ('#ifdef SK_BUILD_FOR_MAC\nstatic const int kExtraLargeDFFontLimit',
                   '#ifdef SK_BUILD_FOR_MAC\n    } else if (scaledTextSize <= kLargeDFFontLimit)'):
        patch(source, 'src/text/gpu/SubRunControl.cpp', anchor,
              anchor.replace('#ifdef SK_BUILD_FOR_MAC',
                             '#if defined(SK_BUILD_FOR_MAC) && !defined(WINK_SKIA_PORTABLE_TEXT)'))
    patch(source, 'include/gpu/ganesh/GrContextOptions.h',
          '#elif defined(SK_BUILD_FOR_MAC)\n    float fGlyphsAsPathsFontSize = 256;',
          '#elif defined(SK_BUILD_FOR_MAC) && !defined(WINK_SKIA_PORTABLE_TEXT)\n'
          '    float fGlyphsAsPathsFontSize = 256;')
    # Only modify the disposable pinned dependency, never the user's Skia tree.
    annotate_host_units(source)
    patch(source, 'src/gpu/ganesh/GrDirectContext.cpp',
          '''    direct->fGpu = GrMockGpu::Make(mockOptions, options, direct.get());
    if (!direct->init()) {
        return nullptr;
    }

    return direct;''',
          '''    direct->fGpu = GrMockGpu::Make(mockOptions, options, direct.get());
    if (!direct->init()) {
        return nullptr;
    }

    SkDebugf("WINK_SKIA_MOCK_CONTEXT\\tganesh\\n");
    return direct;''')
    # This bitmap contains numeric shader data. Bitmap::erase runs a color
    # pipeline whose scalar half stores truncate while native SIMD rounds.
    # Write premul colors and encoded offsets directly with the same RNE
    # conversion, without an unpremultiply/premultiply round trip.
    patch(source, 'src/gpu/GradientBitmap.cpp',
          '''        // TODO: there should be a way to directly set a premul pixel in a bitmap with
        // a premul color.
        SkColor4f unpremulColor = colors[i].unpremul();
        colorsAndOffsetsBitmap.erase(unpremulColor, SkIRect::MakeXYWH(i, 0, 1, 1));''',
          '''        auto* color = static_cast<SkHalf*>(colorsAndOffsetsBitmap.getAddr(i, 0));
        color[0] = SkFloatToHalf(colors[i].fR);
        color[1] = SkFloatToHalf(colors[i].fG);
        color[2] = SkFloatToHalf(colors[i].fB);
        color[3] = SkFloatToHalf(colors[i].fA);''')
    patch(source, 'src/gpu/GradientBitmap.cpp',
          '''        colorsAndOffsetsBitmap.erase(SkColor4f{mantissa, exponent, 0, 1},
                                     SkIRect::MakeXYWH(i, 1, 1, 1));''',
          '''        auto* encoded = static_cast<SkHalf*>(colorsAndOffsetsBitmap.getAddr(i, 1));
        encoded[0] = SkFloatToHalf(mantissa);
        encoded[1] = SkFloatToHalf(exponent);
        encoded[2] = 0;
        encoded[3] = SK_Half1;''')
    patch(source, 'tests/GradientTest.cpp',
          '#include "src/core/SkTLazy.h"',
          '''#include "src/core/SkHalf.h"
#include "src/core/SkTLazy.h"
#if defined(SK_GANESH) || defined(SK_GRAPHITE)
#include "src/gpu/GradientBitmap.h"
#endif''')
    patch(source, 'tests/GradientTest.cpp',
          'DEF_TEST(Gradient, reporter) {',
          '''#if defined(SK_GANESH) || defined(SK_GRAPHITE)
DEF_TEST(GradientTextureDataEncoding, reporter) {
    const SkPMColor4f colors[] = {{0.5003f, 0.25f, 0.75f, 1.f},
                                 {1.0006f, -0.125f, 0x1p-20f, 1.f},
                                 {0.1f, 0.2f, 0.3f, 0.5f}, {0, 0, 0, 0}};
    const float offsets[] = {0.f, 3.f / 199.f, 0.5f, 1.f};
    const SkHalf expected[2][4][4] = {
        {{0x3801, 0x3400, 0x3a00, 0x3c00}, {0x3c01, 0xb000, 0x0010, 0x3c00},
         {0x2e66, 0x3266, 0x34cd, 0x3800}, {0, 0, 0, 0}},
        {{0, 0, 0, 0x3c00}, {0x3bb8, 0xc600, 0, 0x3c00},
         {0x3800, 0, 0, 0x3c00}, {0x3800, 0x3c00, 0, 0x3c00}}
    };
    auto bitmap = skgpu::CreateGradientColorAndOffsetBitmap(4, colors, offsets);
    REPORTER_ASSERT(reporter, !bitmap.empty());
    if (bitmap.empty()) return;
    for (int row = 0; row < 2; ++row) {
        for (int x = 0; x < 4; ++x) {
            const auto* actual = static_cast<const SkHalf*>(bitmap.getAddr(x, row));
            for (int channel = 0; channel < 4; ++channel) {
                REPORTER_ASSERT(reporter, actual[channel] == expected[row][x][channel],
                                "row=%d x=%d channel=%d: %04x != %04x", row, x, channel,
                                actual[channel], expected[row][x][channel]);
            }
        }
    }
}
#endif

DEF_TEST(Gradient, reporter) {''')
    # Compare pre-write and post-write pixels outside the updated rectangle.
    # The pinned test compared its initial read with itself, and its first
    # two guard rectangles included the write or had reversed bounds.
    patch(source, 'tests/ReadWritePixelsGpuTest.cpp',
          '''        SkIRect borders[]{
                {               0,                 0, secondReadPM.width(), secondReadPM.height()},
                {checkRect.fRight,                 0,      checkRect.fLeft, secondReadPM.height()},
                { checkRect.fLeft,                 0,     checkRect.fRight,        checkRect.fTop},
                { checkRect.fLeft, checkRect.fBottom,     checkRect.fRight, secondReadPM.height()}
        };''',
          '''        SkIRect borders[]{
                {               0, checkRect.fTop,       checkRect.fLeft, checkRect.fBottom},
                {checkRect.fRight, checkRect.fTop, secondReadPM.width(), checkRect.fBottom},
                {               0,             0, secondReadPM.width(),    checkRect.fTop},
                {               0, checkRect.fBottom, secondReadPM.width(), secondReadPM.height()}
        };''')
    patch(source, 'tests/ReadWritePixelsGpuTest.cpp',
          'SkAssertResult(firstReadPM.extractSubset(&b, r));',
          'SkAssertResult(secondReadPM.extractSubset(&b, r));')
    # Vector selection consumes all-bit masks. The scalar fallback's logical
    # NOT returned 1 instead of all ones, corrupting clipped perspective quads
    # when Wasm builds disable SIMD.
    patch(source, 'src/core/SkVx.h',
          'SIT Vec<1,T> operator!(const Vec<1,T>& x) { return !x.val; }',
          '''SIT Vec<1,T> operator!(const Vec<1,T>& x) {
        return sk_bit_cast<T>(x.val ? M<T>(0) : M<T>(-1));
    }''')
    patch(source, 'tests/SkVxTest.cpp',
          'DEF_TEST(SkVx, r) {',
          '''DEF_TEST(SkVx, r) {
    // Logical masks must select complete values, including without SIMD.
    const int4 input{0, 1, -1, 0};
    REPORTER_ASSERT(r, all(!input == int4{-1, 0, 0, -1}));
    REPORTER_ASSERT(r, all(if_then_else(!input, float4(42), float4(47)) ==
                           float4{42, 47, 47, 42}));''')
    # Returned resources may put the cache over budget even when no resource
    # is old enough for an age purge, or its time limit has already elapsed.
    # Budget cleanup takes precedence, including these fast-return paths.
    patch(source, 'src/gpu/graphite/ResourceCache.cpp',
          '''        fPurgeableQueue.peek()->lastAccessTime() >= *purgeTime) {
        return;
    }
    if (!time_remains_before_stop_time()) {
        return;
    }''',
          '''        fPurgeableQueue.peek()->lastAccessTime() >= *purgeTime) {
        this->purgeAsNeeded();
        return;
    }
    if (!time_remains_before_stop_time()) {
        this->purgeAsNeeded();
        return;
    }''')
    patch(source, 'tests/graphite/GraphiteResourceCacheTest.cpp',
          '// Test call purgeResources on the ResourceCache and make sure all unlocked resources are getting',
          '''// Budget enforcement must still run when the age-purge deadline has expired.
DEF_GRAPHITE_TEST_FOR_ALL_CONTEXTS(GraphiteOverBudgetExpiredPurgeTest, reporter, context,
                                   CtsEnforcement::kNever) {
    auto recorder = context->makeRecorder();
    auto* provider = recorder->priv().resourceProvider();
    auto* cache = provider->resourceCache();
    cache->setMaxBudget(10);
    auto a = add_new_resource(reporter, provider->sharedContext(), cache, 9, Budgeted::kNo);
    auto b = add_new_resource(reporter, provider->sharedContext(), cache, 8, Budgeted::kNo);
    a.reset();
    b.reset();
    cache->forceProcessReturnedResources();
    REPORTER_ASSERT(reporter, cache->currentBudgetedBytes() == 17);
    cache->purgeResourcesNotUsedSince(skgpu::StdSteadyClock::time_point::max(),
                                     skgpu::StdSteadyClock::time_point::min());
    REPORTER_ASSERT(reporter, cache->getResourceCount() == 1);
    REPORTER_ASSERT(reporter, cache->currentBudgetedBytes() == 8 ||
                              cache->currentBudgetedBytes() == 9);
}

// Test call purgeResources on the ResourceCache and make sure all unlocked resources are getting''')
    # A host without installed fonts may return a non-null empty typeface.
    # Use the existing portable fallback for it as well as for null results,
    # while keeping the native manager's resource-font loading available.
    patch(source, 'tools/fonts/FontToolUtils.cpp',
          '''    if (face) {
        return face;
    }
    return CreatePortableTypeface(name, style);''',
          '''    if (face && face->countGlyphs() > 0) {
        return face;
    }
    return CreatePortableTypeface(name, style);''')
    # Browser steady clocks are quantized. A five-nanosecond sleep need not
    # advance their observable value; wait for actual ticks on both sides of
    # the purge threshold instead, preserving all cache-content assertions.
    patch(source, 'tests/ResourceCacheTest.cpp',
          '''    auto nowish = []() {
        // We sleep so that we ensure we get a value that is greater than the last call to
        // GrStdSteadyClock::now().
        std::this_thread::sleep_for(skgpu::StdSteadyClock::duration(5));
        auto result = skgpu::StdSteadyClock::now();
        // Also sleep afterwards so we don't get this value again.
        std::this_thread::sleep_for(skgpu::StdSteadyClock::duration(5));
        return result;
    };''',
          '''    auto nextTick = [](skgpu::StdSteadyClock::time_point previous) {
        auto now = skgpu::StdSteadyClock::now();
        while (now <= previous) {
            std::this_thread::yield();
            now = skgpu::StdSteadyClock::now();
        }
        return now;
    };
    auto nowish = [&]() {
        const auto result = nextTick(skgpu::StdSteadyClock::now());
        // The next resource insertion must also be strictly after result.
        nextTick(result);
        return result;
    };''')
    # Mask-filtered color glyphs can fall back to grayscale coverage. Preserve
    # the original shader before the ARGB-only branch strips it from options.
    patch(source, 'src/gpu/graphite/PublicPrecompile.cpp',
          '''                // For color emoji text, shaders don't affect the final color
                PaintOptions tmp = options;''',
          '''                if (!options.getMaskFilters().empty()) {
                    PaintOptions maskOptions = options;
                    maskOptions.setMaskFilters({});
                    PrecompileCombinations(sharedContext, maskOptions, keyContext,
                            static_cast<DrawTypeFlags>(DrawTypeFlags::kBitmapText_Mask |
                                                       (drawTypes & DrawTypeFlags::kAnalyticClip)),
                            false, Coverage::kSingleChannel, renderPassDesc);
                }
                // For color emoji text, shaders don't affect the final color
                PaintOptions tmp = options;''')
    # Runtime blend shaders give their children runtime-effect semantics,
    # including disabling paint-color colorization of alpha-only images.
    patch(source, 'src/gpu/graphite/precompile/PrecompileShader.cpp',
          '''        AddToKey<PrecompileShader>(keyContext, fSrcOptions, desiredSrcCombination);
        AddToKey<PrecompileShader>(keyContext, fDstOptions, desiredDstCombination);''',
          '''        const SkRuntimeEffect* runtimeBlend = blender->priv().asBlendMode() ? nullptr
                : GetKnownRuntimeEffect(SkKnownRuntimeEffects::StableKey::kBlend);
        KeyContext srcContext = runtimeBlend ? keyContext.forRuntimeEffect(runtimeBlend, 0)
                                             : keyContext;
        KeyContext dstContext = runtimeBlend ? keyContext.forRuntimeEffect(runtimeBlend, 1)
                                             : keyContext;
        AddToKey<PrecompileShader>(srcContext, fSrcOptions, desiredSrcCombination);
        AddToKey<PrecompileShader>(dstContext, fDstOptions, desiredDstCombination);''')
    patch(source, 'src/gpu/graphite/precompile/PrecompileShader.cpp',
          '            blender->priv().addToKey(keyContext, blenderCombination);',
          '''            blender->priv().addToKey(keyContext.forRuntimeEffect(runtimeBlend, 2),
                                       blenderCombination);''')
    # The real gradient path uses unpremultiplied interpolation for perceptual
    # color spaces and for explicit unpremultiplied interpolation. This is
    # visible in specialized keys inside working-color-space shaders.
    patch(source, 'src/gpu/graphite/precompile/PrecompileShader.cpp',
          '''        ColorSpaceTransformBlock::ColorSpaceTransformData csData(
                intermediateCS.get(), kPremul_SkAlphaType,
                dstCS, kPremul_SkAlphaType);''',
          '''        using ColorSpace = SkGradient::Interpolation::ColorSpace;
        bool inputPremul = static_cast<bool>(fInterpolation.fInPremul);
        switch (fInterpolation.fColorSpace) {
            case ColorSpace::kLab:
            case ColorSpace::kOKLab:
            case ColorSpace::kOKLabGamutMap:
            case ColorSpace::kLCH:
            case ColorSpace::kOKLCH:
            case ColorSpace::kOKLCHGamutMap:
            case ColorSpace::kHSL:
            case ColorSpace::kHWB:
                inputPremul = false;
                break;
            default:
                break;
        }
        ColorSpaceTransformBlock::ColorSpaceTransformData csData(
                intermediateCS.get(), inputPremul ? kPremul_SkAlphaType : kUnpremul_SkAlphaType,
                dstCS, keyContext.dstColorInfo().alphaType());''')
    # Coverage-layer blur consumes Alpha_8 images, and non-normal blur styles
    # combine the original and blurred masks. Precompile all those real paths,
    # including the actual coverage geometry and text's bitmap-mask fallback.
    patch(source, 'src/gpu/graphite/precompile/PrecompileImageFilter.cpp',
          '''    // For blur imagefilters we know we don't have alpha-only textures and don't need cubic
    // filtering.
    sk_sp<PrecompileShader> imageShader = PrecompileShaders::Image(
            ImageShaderFlags::kNoAlphaNoCubic);''',
          '''    // Blur inputs include alpha-only coverage masks. Cubic filtering is not used.
    sk_sp<PrecompileShader> imageShader = PrecompileShaders::Image(
            ImageShaderFlags::kExcludeCubic);''')
    patch(source, 'include/gpu/graphite/precompile/PrecompileMaskFilter.h',
          '''    virtual void createPipelines(const KeyContext&,
                                 const PaintOptions&,
                                 const RenderPassDesc&,''',
          '''    virtual void createPipelines(const KeyContext&,
                                 const PaintOptions&,
                                 DrawTypeFlags,
                                 const RenderPassDesc&,''')
    patch(source, 'src/gpu/graphite/precompile/PaintOptions.cpp',
          '            o->createPipelines(keyContext, *this, renderPassDesc, processCombination);',
          '            o->createPipelines(keyContext, *this, drawTypes, renderPassDesc, processCombination);')
    patch(source, 'src/gpu/graphite/precompile/PrecompileMaskFilter.cpp',
          '''            const PaintOptions& paintOptions,
            const RenderPassDesc& renderPassDescIn,''',
          '''            const PaintOptions& paintOptions,
            DrawTypeFlags drawTypes,
            const RenderPassDesc& renderPassDescIn,''')
    patch(source, 'src/gpu/graphite/precompile/PrecompileMaskFilter.cpp',
          '''                                                                   processCombination);

        // c.f. AutoLayerForImageFilter::addMaskFilterLayer.''',
          '''                                                                   processCombination);

        // Inner, solid and outer blur styles blend the original and blurred masks.
        const SkColorInfo maskInfo(kAlpha_8_SkColorType, kPremul_SkAlphaType, nullptr);
        auto maskShader = PrecompileShaders::Image(PrecompileShaders::ImageShaderFlags::kExcludeCubic,
                                                   SkSpan(&maskInfo, 1));
        for (SkBlendMode style : {SkBlendMode::kDstIn, SkBlendMode::kSrcOver,
                                 SkBlendMode::kDstOut}) {
            PaintOptions combineMasks;
            combineMasks.setShaders({{PrecompileShaders::Blend(SkSpan<const SkBlendMode>(&style, 1),
                                                   SkSpan(&maskShader, 1), SkSpan(&maskShader, 1))}});
            combineMasks.setBlendModes(SKSPAN_INIT_ONE(SkBlendMode::kSrc));
            for (Coverage coverage : {Coverage::kNone, Coverage::kSingleChannel}) {
                combineMasks.priv().buildCombinations(keyContext, DrawTypeFlags::kSimpleShape,
                        false, coverage, coverageRenderPassDesc, processCombination);
            }
        }

        // c.f. AutoLayerForImageFilter::addMaskFilterLayer.''')
    patch(source, 'src/gpu/graphite/precompile/PrecompileMaskFilter.cpp',
          '''            restoreOptions.setMaskFilters({});''',
          '''            restoreOptions.setMaskFilters({});
            // Text and complex paths with a mask filter can use a
            // grayscale atlas fallback instead of a filtered coverage layer.
            if (drawTypes & (DrawTypeFlags::kBitmapText_LCD | DrawTypeFlags::kBitmapText_Color |
                             DrawTypeFlags::kSDFText | DrawTypeFlags::kSDFText_LCD |
                             DrawTypeFlags::kNonSimpleShape)) {
                restoreOptions.priv().buildCombinations(keyContext,
                        static_cast<DrawTypeFlags>(DrawTypeFlags::kBitmapText_Mask |
                                                   (drawTypes & DrawTypeFlags::kAnalyticClip)),
                        false, Coverage::kSingleChannel,
                        renderPassDescIn, processCombination);
            }''')
    patch(source, 'src/gpu/graphite/precompile/PrecompileMaskFilter.cpp',
          '''                    renderPassDescIn,
                    processCombination);''',
          '''                    renderPassDescIn,
                    processCombination);
            // A complex path's coverage draw may require MSAA/stencil, but
            // restoring the filtered mask uses a single-sample depth pass.
            if (renderPassDescIn.fSampleCount != SampleCount::k1) {
                RenderPassDesc restorePass = renderPassDescIn;
                restorePass.fSampleCount = SampleCount::k1;
                restorePass.fColorAttachment.fSampleCount = SampleCount::k1;
                restorePass.fColorResolveAttachment = {};
                restorePass.fDepthStencilAttachment.fSampleCount = SampleCount::k1;
                restorePass.fDepthStencilAttachment.fFormat = caps->getDepthStencilFormat(
                        caps->avoidDepthMode() ? DepthStencilFlags::kNone
                                               : DepthStencilFlags::kDepth);
                restoreOptions.priv().buildCombinations(keyContext,
                        static_cast<DrawTypeFlags>(static_cast<int>(InternalDrawTypeFlags::kCoverageMask) |
                                                   (drawTypes & DrawTypeFlags::kAnalyticClip)),
                        false, Coverage::kSingleChannel, restorePass, processCombination);
                restoreOptions.priv().buildCombinations(keyContext,
                        static_cast<DrawTypeFlags>(DrawTypeFlags::kBitmapText_Mask |
                                                   (drawTypes & DrawTypeFlags::kAnalyticClip)),
                        false, Coverage::kSingleChannel,
                        restorePass, processCombination);
            }''')
    patch(source, 'src/gpu/graphite/precompile/PrecompileMaskFilter.cpp',
          '''                    static_cast<DrawTypeFlags>(InternalDrawTypeFlags::kCoverageMask),
                    /* withPrimitiveBlender= */ false,''',
          '''                    static_cast<DrawTypeFlags>(static_cast<int>(InternalDrawTypeFlags::kCoverageMask) |
                                               (drawTypes & DrawTypeFlags::kAnalyticClip)),
                    /* withPrimitiveBlender= */ false,''')
    patch(source, 'src/gpu/graphite/precompile/PrecompileMaskFilter.cpp',
          '''            coverageOptions.priv().buildCombinations(
                    keyContext,
                    DrawTypeFlags::kAnalyticRRect,
                    /* withPrimitiveBlender= */ false,
                    Coverage::kSingleChannel,
                    coverageRenderPassDesc,
                    processCombination);''',
          '''            for (Coverage coverage : {Coverage::kNone, Coverage::kSingleChannel}) {
                coverageOptions.priv().buildCombinations(
                        keyContext,
                        static_cast<DrawTypeFlags>(drawTypes | DrawTypeFlags::kSimpleShape |
                                                   DrawTypeFlags::kBitmapText_Mask),
                        /* withPrimitiveBlender= */ false,
                        coverage,
                        coverageRenderPassDesc,
                        processCombination);
                // Blurred text can fall back to paths requiring stencil and MSAA.
                RenderPassDesc pathPass = RenderPassDesc::Make(caps, info,
                        LoadOp::kClear, StoreOp::kStore, DepthStencilFlags::kDepthStencil,
                        {0.f, 0.f, 0.f, 0.f}, true, skgpu::Swizzle("a000"),
                        caps->getDstReadStrategy());
                coverageOptions.priv().buildCombinations(keyContext,
                        DrawTypeFlags::kNonSimpleShape, false, coverage, pathPass,
                        processCombination);
            }''')
    patch(source, 'src/gpu/graphite/precompile/PrecompileImageFilter.cpp',
          '''    blurPaintOptions.priv().buildCombinations(keyContext,
                                              DrawTypeFlags::kSimpleShape,
                                              /* withPrimitiveBlender= */ false,
                                              Coverage::kSingleChannel,
                                              renderPassDesc,
                                              processCombination);''',
          '''    for (Coverage coverage : {Coverage::kNone, Coverage::kSingleChannel}) {
        blurPaintOptions.priv().buildCombinations(keyContext,
                                                DrawTypeFlags::kSimpleShape,
                                                /* withPrimitiveBlender= */ false,
                                                coverage,
                                                renderPassDesc,
                                                processCombination);
    }''')
    # Replacing the perspective row of a translate(2,2) matrix with (0.5,0,1)
    # makes it singular. Compose an invertible perspective transform instead
    # so generated shader inputs remain valid for precompile comparisons.
    patch(source, 'tests/graphite/precompile/PaintParamsTestUtils.cpp',
          '        storage->setPerspX(0.5f);',
          '''        SkMatrix perspective = SkMatrix::I();
        perspective.setPerspX(0.5f);
        storage->postConcat(perspective);''')
    # YUVA sampling returns unpremultiplied color even when the image's
    # advertised alpha type is premultiplied; match add_yuv_image_to_key.
    patch(source, 'src/gpu/graphite/precompile/PrecompileShader.cpp',
          '''        colorXformData.fSteps = SkColorSpaceXformSteps(
                colorInfo.colorSpace(), colorInfo.alphaType(),
                dstColorSpace, colorInfo.alphaType());''',
          '''        const SkAlphaType srcAT = colorInfo.alphaType() == kPremul_SkAlphaType
                                                ? kUnpremul_SkAlphaType : colorInfo.alphaType();
        SkAlphaType dstAT = fUseDstColorSpace ? keyContext.dstColorInfo().alphaType()
                                            : colorInfo.alphaType();
        if (dstAT == kOpaque_SkAlphaType && srcAT == kUnpremul_SkAlphaType) {
            dstAT = kPremul_SkAlphaType;
        }
        colorXformData.fSteps = SkColorSpaceXformSteps(
                colorInfo.colorSpace(), srcAT, dstColorSpace, dstAT);''')
    patch(source, 'tests/graphite/precompile/PaintParamsKeyTest.cpp',
          '#include "tests/Test.h"', '#include "tests/Test.h"\n#include <cstdlib>')
    patch(source, 'tests/graphite/precompile/PaintParamsKeyTest.cpp',
          '            ++missingPipelines;',
          '''            ++missingPipelines;
            if (std::getenv("WINK_SKIA_TRACE_PRECOMPILE")) {
                GraphicsPipelineDesc actual;
                RenderPassDesc actualPass;
                UniqueKeyUtils::ExtractKeyDescs(precompileContext, k, &actual, &actualPass);
                auto* caps = precompileContext->priv().caps();
                auto* dict = precompileContext->priv().shaderCodeDictionary();
                SkDebugf("WINK_SKIA_MISSING_PIPELINE\\n%s\\n%s\\n",
                         actual.toString(caps, dict).c_str(), actualPass.toString().c_str());
                for (const auto& compiled : precompileKeys) {
                    GraphicsPipelineDesc candidate;
                    RenderPassDesc candidatePass;
                    UniqueKeyUtils::ExtractKeyDescs(precompileContext, compiled,
                                                   &candidate, &candidatePass);
                    if (candidate.renderStepID() == actual.renderStepID()) {
                        SkDebugf("WINK_SKIA_PRECOMPILED_PIPELINE\\n%s\\n%s\\n",
                                 candidate.toString(caps, dict).c_str(),
                                 candidatePass.toString().c_str());
                    }
                }
            }''')
    patch(source, 'tests/graphite/precompile/PaintParamsKeyTest.cpp',
          '    uint32_t seed = std::time(nullptr) % std::numeric_limits<uint32_t>::max();',
          '''    uint32_t seed = std::time(nullptr) % std::numeric_limits<uint32_t>::max();
    if (const char* overrideSeed = std::getenv("WINK_SKIA_PAINT_KEY_SEED")) {
        char* end = nullptr;
        unsigned long long parsed = std::strtoull(overrideSeed, &end, 10);
        if (end == overrideSeed || *end || parsed > std::numeric_limits<uint32_t>::max()) {
            SK_ABORT("Invalid WINK_SKIA_PAINT_KEY_SEED");
        }
        seed = static_cast<uint32_t>(parsed);
    }''')
    # Match Ganesh's existing hugebitmapshader contract: an image shader
    # beyond the texture limit draws using the paint color instead of dropping
    # the geometry. Ordinary image draws use tiling; raw shaders and failures
    # for images within the supported limit retain their existing error path.
    patch(source, 'src/gpu/graphite/KeyHelpers.cpp',
          '''static void add_to_key(const KeyContext& keyContext, const SkImageShader* shader) {
    SkASSERT(shader);''',
          '''static void add_to_key(const KeyContext& keyContext, const SkImageShader* shader) {
    SkASSERT(shader);
    const int maxTextureSize = keyContext.caps()->maxTextureSize();
    if (!shader->isRaw() && (shader->image()->width() > maxTextureSize ||
                            shader->image()->height() > maxTextureSize)) {
        RGBPaintColorBlock::AddBlock(keyContext);
        return;
    }''')
    # Graphite implements drawAsTiledImageRect but inherits the false opt-in
    # from SkDevice, so ordinary canvas draws bypass it and try to upload
    # images larger than maxTextureSize. Use the existing tiling path as
    # Ganesh does; manual SkTiledImageUtils draws already exercise this code.
    patch(source, 'src/gpu/graphite/Device.h',
          '    bool drawAsTiledImageRect(SkCanvas*,',
          '''    bool shouldDrawAsTiledImageRect() const override { return true; }
    bool drawAsTiledImageRect(SkCanvas*,''')
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
    patch(source, 'dm/DM.cpp', '        test.cpu(&reporter);',
          '        SkDebugf("WINK_SKIA_CPU_UNIT\\t%s\\n", test.fName);\n'
          '        test.cpu(&reporter);')
    for event, signature in (
        ('DONE', 'static void done(const char* config, const char* src, const char* srcOptions, const char* name) {'),
        ('START', 'static void start(const char* config, const char* src, const char* srcOptions, const char* name) {'),
    ):
        # A test may leave a partial diagnostic line on stderr. Start lifecycle
        # records on a fresh line so that completion cannot be swallowed.
        old = f'fprintf(stderr, "WINK_SKIA_{event}\\t'
        new = f'fprintf(stderr, "\\nWINK_SKIA_{event}\\t'
        if old in (source / 'dm/DM.cpp').read_text():
            patch(source, 'dm/DM.cpp', old, new)
        patch(source, 'dm/DM.cpp', signature, signature +
              f'\n    fprintf(stderr, "\\nWINK_SKIA_{event}\\t%s\\t%s\\t%s\\t%s\\n", config, src, srcOptions, name);')
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
    legacy_guard = '''    if (const char* route = type == ContextType::kMock ? nullptr : getenv("WINK_SKIA_ROUTE")) {
        if ((!strcmp(route, "gles") && type != ContextType::kGLES) ||
            (!strcmp(route, "vk") && type != ContextType::kVulkan)) return {};
    }
'''
    context_guard = '''    if (const char* route = getenv("WINK_SKIA_ROUTE")) {
        if (!strcmp(route, "gles") || !strcmp(route, "vk")) {
            const bool mock = type == ContextType::kMock;
            const bool selected = (!strcmp(route, "gles") && type == ContextType::kGLES) ||
                                  (!strcmp(route, "vk") && type == ContextType::kVulkan);
            SkDebugf("WINK_SKIA_CONTEXT_REQUEST\\tganesh\\t%s\\n",
                     mock ? "mock" : selected ? "selected" : "outside");
            if (!mock && !selected) return {};
        }
    }
'''
    content = content.replace(legacy_guard, context_guard).replace(
        legacy_guard.replace('type == ContextType::kMock ? nullptr : ', ''), context_guard)
    content = content.replace(
        '    if (result.directContext()) SkDebugf("WINK_SKIA_CONTEXT\\tganesh\\n");',
        '    if (result.directContext() && type != ContextType::kMock) SkDebugf("WINK_SKIA_CONTEXT\\tganesh\\n");')
    factory.write_text(content)
    patch(source, 'tools/ganesh/GrContextFactory.cpp',
          '    // (shareIndex != 0) -> (shareContext != nullptr)',
          context_guard + '    // (shareIndex != 0) -> (shareContext != nullptr)')
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
    legacy_graphite_guard = '''ContextInfo ContextFactory::getContextInfo(skgpu::ContextType type) {
    if (const char* route = getenv("WINK_SKIA_ROUTE")) {
        if ((!strcmp(route, "grvk") && type != skgpu::ContextType::kVulkan) ||
            (!strcmp(route, "grdawn_gles") && type != skgpu::ContextType::kDawn_OpenGLES)) return {};
    }'''
    graphite_guard = '''ContextInfo ContextFactory::getContextInfo(skgpu::ContextType type) {
    if (const char* route = getenv("WINK_SKIA_ROUTE")) {
        if (!strcmp(route, "grvk") || !strcmp(route, "grdawn_gles")) {
            const bool selected = (!strcmp(route, "grvk") && type == skgpu::ContextType::kVulkan) ||
                                  (!strcmp(route, "grdawn_gles") && type == skgpu::ContextType::kDawn_OpenGLES);
            SkDebugf("WINK_SKIA_CONTEXT_REQUEST\\tgraphite\\t%s\\n", selected ? "selected" : "outside");
            if (!selected) return {};
        }
    }'''
    graphite_factory = source / 'tools/graphite/ContextFactory.cpp'
    content = graphite_factory.read_text()
    if legacy_graphite_guard in content:
        graphite_factory.write_text(content.replace(legacy_graphite_guard, graphite_guard))
    patch(source, 'tools/graphite/ContextFactory.cpp',
          'ContextInfo ContextFactory::getContextInfo(skgpu::ContextType type) {', graphite_guard)
    # Some Vulkan-specific tests manage their backend context directly instead
    # of using either factory. Record that explicit API request too.
    patch(source, 'tools/gpu/vk/VkTestUtils.cpp',
          '''                            const CanPresentFn& canPresent,
                            bool isProtected) {''',
          '''                            const CanPresentFn& canPresent,
                            bool isProtected) {
    if (const char* route = getenv("WINK_SKIA_ROUTE")) {
        const bool selected = !strcmp(route, "vk") || !strcmp(route, "grvk");
        SkDebugf("WINK_SKIA_CONTEXT_REQUEST\\tvulkan\\t%s\\n", selected ? "selected" : "outside");
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
    legacy_dawn_dispatch = '''    options.nextInChain = &togglesDesc;
    dawn::native::opengl::RequestAdapterOptionsGetGLProc winkGL;
    if (backend == wgpu::BackendType::OpenGLES && getenv("WINK_SKIA_ROUTE") &&
        !strcmp(getenv("WINK_SKIA_ROUTE"), "grdawn_gles")) {
        winkGL.getProc = winkSkiaEGLGetProc;
        winkGL.nextInChain = options.nextInChain;
        options.nextInChain = &winkGL;
        options.backendType = wgpu::BackendType::OpenGLES;
    }'''
    dawn_dispatch = legacy_dawn_dispatch.replace(
        '!strcmp(getenv("WINK_SKIA_ROUTE"), "grdawn_gles")) {',
        '''!strcmp(getenv("WINK_SKIA_ROUTE"), "grdawn_gles") &&
        !getenv("WINK_SKIA_REFERENCE_GLES")) {''')
    dawn_context = source / 'tools/graphite/dawn/GraphiteDawnTestContext.cpp'
    content = dawn_context.read_text()
    if legacy_dawn_dispatch in content:
        dawn_context.write_text(content.replace(legacy_dawn_dispatch, dawn_dispatch))
    patch(source, 'tools/graphite/dawn/GraphiteDawnTestContext.cpp',
          '    options.nextInChain = &togglesDesc;', dawn_dispatch)
    patch(source, 'tools/graphite/dawn/GraphiteDawnTestContext.cpp',
          '''        if (backend == props.backendType) {
            matchedAdaptor = adapter;''',
          '''        if (backend == props.backendType) {
            if (backend == wgpu::BackendType::OpenGLES) {
                SkDebugf("WINK_SKIA_BACKEND\\tgles\\t%.*s\\n",
                         static_cast<int>(props.device.length), props.device.data);
            }
            matchedAdaptor = adapter;''')
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
    p.add_argument('--font-engine', choices=['platform', 'freetype'], default='platform',
                   help='Use FreeType for resource-font references to browser builds')
    p.add_argument('--text-policy', choices=['platform', 'portable'], default='platform',
                   help='Match browser distance-field atlas sizes and large-glyph thresholds')
    p.add_argument('--build-dir', type=Path, help='Override the native build directory')
    a = p.parse_args()
    a.out = a.out.resolve()
    source = a.out / 'deps/skia' if a.no_sync else checkout(a.out)
    pin = json.loads((HERE / 'pins.json').read_text())['skia']['commit']
    if subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() != pin:
        p.error('Skia revision does not match pins.json')
    if not a.library.is_file() or not a.vk_loader.is_file():
        p.error('Build native Wink and install a Vulkan loader first')
    adapt(source)
    if a.font_engine == 'freetype':
        from sync_fonts import ensure
        ensure(source, a.out / 'skparagraph-v4.zip')
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
    if a.font_engine == 'freetype':
        args.update(skia_use_freetype=True, skia_use_system_freetype2=False,
                    skia_use_fonthost_mac=False, skia_use_fontconfig=False,
                    skia_enable_fontmgr_custom_directory=False,
                    skia_enable_fontmgr_custom_empty=True,
                    skia_enable_fontmgr_custom_embedded=True)
    if a.text_policy == 'portable':
        args['extra_cflags'].append('-DWINK_SKIA_PORTABLE_TEXT')
    build = a.out / ('native-dawn' if a.dawn_gles else 'native')
    if a.font_engine == 'freetype':
        build = build.with_name(build.name + '-freetype')
    if a.text_policy == 'portable':
        build = build.with_name(build.name + '-portable')
    if a.build_dir:
        build = a.build_dir.resolve()
    build.mkdir(parents=True, exist_ok=True)
    gn_args = '\n'.join(f'{k} = {json.dumps(v)}' for k, v in args.items()) + '\n'
    (build / 'args.gn').write_text(gn_args)
    run([source / 'bin/gn', 'gen', build], cwd=source)
    run(['ninja', '-C', build, '-j', a.jobs, 'dm', 'skdiff'], cwd=source)
    (build / 'wink-build.json').write_text(json.dumps({
        'schema': 1, 'skia': pin, 'gn_args': args, 'skia_fixes': CORRECTNESS_FIXES,
        'font_engine': a.font_engine,
        'text_policy': a.text_policy,
        'font_selection': 'portable' if a.text_policy == 'portable' else 'platform',
        'font_manager': 'pinned-fixtures' if a.font_engine == 'freetype' else 'platform',
        'wink': subprocess.check_output(['git', '-C', MESA, 'rev-parse', 'HEAD'], text=True).strip(),
        'routes': ['gles', 'vk', 'grvk'] + (['grdawn_gles'] if a.dawn_gles else []),
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
