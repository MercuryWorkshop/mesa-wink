# Skia on Wink

This harness pins Skia and runs its own DM correctness tests through Wink.
The acceptance target is Ganesh GLES/Vulkan and Graphite Vulkan/Dawn GLES on
native macOS and browser ST/MT. A smoke pass is not full conformance.

## Native build

Build `libwink.dylib` and Wink's Vulkan ICD using the instructions in
`wink/README.md` and `src/wink/vulkan/README.md`, then:

```sh
python3 wink/test/skia/build.py
# Already synchronized dependencies:
python3 wink/test/skia/build.py --no-sync
```

Dependencies and outputs live in ignored `build/skia/`. Skia's DEPS file pins
its dependencies and `bin/fetch-gn` pins GN. `pins.json` pins Skia itself.
The adapter modifies only that disposable dependency checkout. Patch anchors
and revision checks fail if the upstream source no longer matches.

The pinned Skia integration also fixes its compressed Vulkan host-copy row
length: `memoryRowLength` counts texels, while the original code passed the
number of compression blocks. This fixes corrupted Graphite uploads on
drivers exposing host image copy. The `skia_fixes` build/run metadata records
this correction, and comparisons reject different fix sets. Keep unpatched
failure artifacts when investigating upstream behavior. No image tolerance is
added by this correction.

Framebuffer-fetch shader blending also reverses the render target's write
swizzle before reading destination color. An alpha-only surface backed by R8
writes alpha into red and must read it back from red; reading the attachment's
absent alpha instead produced black boxes around LCD text. The existing
`skbug_12212` GM now matches the independent ANGLE reference exactly. This
correction is recorded in `skia_fixes` for native and browser builds.

Graphite's existing image tiling implementation is enabled for ordinary canvas
image draws. The pinned version otherwise inherits a disabled tiling hook and
drops images larger than the texture limit. The large bitmap and picture GMs
now match their explicitly tiled variants exactly; this correction is also
recorded in `skia_fixes`.

For non-raw image shaders whose images exceed the texture limit, Graphite uses
the paint-color fallback required by Skia's existing `hugebitmapshader` GM,
matching Ganesh's behavior. This preserves the geometry when the shader image
cannot be uploaded. Raw shaders and conversion failures for images within the
texture limit keep their error path. This correction is recorded in
`skia_fixes`.

Graphite mask-filter precompilation includes alpha-only blur inputs, the
inner/solid/outer mask combinations, and the coverage geometry and render
passes used by rectangles and blurred text paths, including single-sample
restores after MSAA coverage and grayscale atlas fallbacks for text and paths,
retaining analytic clip modifiers on the restore draw.
Mask-filtered color text also precompiles its grayscale fallback using the
original paint shader, before the ARGB path removes that shader.
Gradient and YUVA precompilation also use the same intermediate alpha types
and destination alpha types as real draws, including specialized transforms
inside working-color-space shaders. The randomized precompile
test also composes its perspective transform instead of generating a singular
matrix. These corrections are recorded in `skia_fixes`; the original key and
pipeline coverage assertions remain enabled.

Runtime blend precompilation applies runtime-effect flags to each child, as
real runtime shaders do. In particular, alpha-only image children keep their
alpha data instead of being colorized with the paint color. This correction
is also recorded in `skia_fixes`.

The resource-cache purge test waits for observable steady-clock ticks instead
of assuming a five-nanosecond sleep advances the quantized browser clock. Its
purge and cache-content assertions remain enabled. This test correction is
recorded in `skia_fixes`.

Graphite's resource cache also enforces its memory budget before early returns
from age-limited or time-limited purges. Returned resources can put it over
budget even when none qualify for an age purge. This correction is recorded
in `skia_fixes`; the existing budget and purge assertions remain enabled.

The test typeface helper uses its portable fallback when a native font manager
returns an empty typeface, as can happen without installed system fonts in
Wasm. Resource fonts still load through FreeType. This correction is recorded
in `skia_fixes` and preserves the remote-glyph cache assertions.

SkVx's scalar logical NOT returns an all-bit mask, matching the SIMD path.
Returning the scalar boolean value `1` corrupts vector selection, including
Ganesh's clipping of perspective quads in Wasm. The original `crbug_224618`
GM exercises the missing geometry; the SkVx unit also checks mask selection.
This correction is recorded in `skia_fixes`.

The GPU pixel-write tests compare the before/after reads in four guard
rectangles covering every pixel outside the write. The pinned test compared
the before image with itself and had malformed guard rectangles. This
correction strengthens its preservation assertions and is recorded in
`skia_fixes`.

Gradient lookup textures encode premultiplied colors and numeric stop data
directly with round-to-nearest half conversion. Bitmap color erasure uses
different rounding on scalar and SIMD paths, shifting densely packed stops
in Wasm. The gradient encoding test checks exact half values, including
extended colors and subnormals. This correction is recorded in `skia_fixes`.

The integration registers FreeType's compiled SVG renderer in its Wasm module
list, fixing an SVG emoji metrics crash, and matches image extensions with a
leading dot so `.wbmp` is not also collected as `bmp`. These corrections are
included in `skia_fixes`. Source/sink vetoes are recorded separately as
`source_vetoes`; they are excluded by upstream compatibility checks and never
count toward completed GPU tasks. Explicit DM skip filters fail the run.

The default native build enables `gles`, `vk`, and `grvk`. To additionally
build Graphite through an outer Dawn GLES backend:

```sh
python3 wink/test/skia/build.py --no-sync --dawn-gles
python3 wink/test/skia/run.py run --dm build/skia/native-dawn/dm \
  --route grdawn_gles --sources tests --match SkSL --out build/skia/sksl-grdawn-gles
```

The test adapter supplies Dawn an explicit EGL/GL resolver backed by Wink.
The outer Dawn and its Tint symbols have hidden visibility to keep them
independent from Wink's inner Dawn. This route currently builds natively;
browser builds still require the outer Dawn integration. The adapter enables
Dawn's `gl_allow_context_on_multi_threads` setting so Graphite's pipeline
workers release the EGL context between calls instead of retaining stale
current contexts across tests.

An independent `grdawn_gles` reference needs GLES 3.1. ANGLE's Metal backend
on this host supplies GLES 3.0; build its Vulkan backend instead:

```sh
python3 wink/test/skia/build-angle.py --backend vulkan
python3 wink/test/skia/build.py --no-sync --dawn-gles --font-engine freetype \
  --angle build/skia/angle-vulkan
python3 wink/test/skia/run.py reference --route grdawn_gles --reference-angle \
  --dm build/skia/native-dawn-freetype/dm --icd /path/to/vk_swiftshader_icd.json \
  --sources gm --match '^aaclip$' --out build/skia/dawn-gles-reference
```

Use an independent Vulkan ICD that exposes GLES 3.1 through ANGLE; SwiftShader
does on this host. The reference bypasses Wink's EGL resolver and records
Dawn's actual GL renderer. The run also hashes the selected ICD, driver, and
ANGLE libraries. `--build-dir` can reuse an existing native build directory.

## Browser builds

Package both Wink variants first, then build DM with the same SDK:

```sh
wink/build.sh --variant=both
python3 wink/test/skia/build-browser.py st
python3 wink/test/skia/build-browser.py mt
python3 wink/test/skia/run.py run --target st --route gles --sources gm --match '^aaclip$' --out build/skia/browser-clip-st
python3 wink/test/skia/run.py run --target mt --route gles --sources gm --match '^aaclip$' --out build/skia/browser-clip-mt
```

Both variants also build the `vk` and `grvk` routes. Vulkan resolves through
Wink directly, without a loader. The ST body enters through JSPI on main; the
MT body enters through JSPI on a real pthread and routes GLES and Vulkan
through Wink's main-thread runner. Graphite's pipeline workers therefore use
the same WebGPU realm. The audit rejects an MT result with no worker marker.
ST disables DM's background GPU thread pool; MT uses two background threads.
Tests which explicitly create threads still require separate applicability
classification for ST.

The GL adapter bridges older Skia WebGL sync callbacks, which split a 64-bit
timeout into two words, to Wink's native Wasm C ABI. Both fence polling and
server waits retain the full timeout value.

Browser files stream individually into the run's `run-artifacts/` directory,
avoiding a single base64/CDP payload for a full GM corpus. The local browser
server only accepts uploads when `--artifact-dir` is provided, and rejects
paths outside it. The runner copies `dm.json` to the run root for comparison.

## Inventory and run

Use a fresh output directory for each invocation. Each run inventories tasks
before execution and writes commands, revisions, fixture hashes, backend
identities, logs, and task counts. `--threads 0` keeps DM's thread-safe work
on the calling thread; tests which explicitly create threads still do so.

```sh
python3 wink/test/skia/run.py list --route gles --out build/skia/list-gles
python3 wink/test/skia/run.py run --route gles --sources gm --match '^aaclip$' --out build/skia/clip-gles
python3 wink/test/skia/run.py run --route vk --out build/skia/all-vk
python3 wink/test/skia/run.py run --route grvk --out build/skia/all-grvk
python3 wink/test/skia/run.py run --route gles --sources tests --match SkSL --out build/skia/sksl-gles
```

The default sources are all compiled GPU unit tests for the selected
architecture, GMs, and the pinned image resources. `--sources` and `--match`
allow a reduced reproducer. A reduced run does not establish full coverage.
Additional resource corpora such as SKPs must be separately pinned before
being included in the acceptance suite.

Use `--route gles --config glesmsaa4` or `--route vk --config vkmsaa4` to run
Skia's four-sample rendering configurations on native or browser targets.
The route still controls the context factory and backend identity checks.
Comparisons require the same rendering configuration, including sample count.

Use `--paint-key-seed 1793113975` to replay `PaintParamsKeyTestReduced` with
an unsigned 32-bit seed, on native or browser routes. The equivalent environment
variable is `WINK_SKIA_PAINT_KEY_SEED`; the command-line option takes precedence.
The selected seed is recorded in run metadata and must match for comparisons.
Without an override, Skia keeps its time-derived seed. Set
`WINK_SKIA_TRACE_PRECOMPILE=1` on native or browser runs to print missing
pipeline descriptions alongside
precompiled candidates when diagnosing a coverage assertion.

GLES always uses Wink's context factory and proc-address API. Vulkan uses an
explicit ICD manifest (`--icd` overrides the default native build). GPU unit
context factories are restricted to the selected route, avoiding coverage
being attributed to a different backend. Unit tests which do not exercise an
observed GPU context are reported as unresolved, rather than passing silently.

## Independent reference and comparison

Browser builds use FreeType, while the default native Mac build uses CoreText.
Build native references with `build.py --no-sync --font-engine freetype`
(and `--angle build/skia/angle` for GLES), then select
`--dm build/skia/native-freetype/dm` when running those references. This keeps
resource-font loading and rasterization comparable. Comparisons reject
different font engines. Platform-specific GMs and default emoji selection
still require separate applicability checks.

For Vulkan, the same test binary can select an independent ICD. For example,
on a Mac with Homebrew MoltenVK:

```sh
python3 wink/test/skia/run.py reference --route vk --sources gm --match '^aaclip$' \
  --icd /opt/homebrew/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json \
  --out build/skia/clip-reference
python3 wink/test/skia/run.py compare build/skia/clip-reference build/skia/clip-vk \
  --out build/skia/clip-comparison.json
```

For an independent Ganesh GLES reference on macOS, build Skia's pinned ANGLE
Metal implementation and add it to the native test binary:

```sh
python3 wink/test/skia/build-angle.py
python3 wink/test/skia/build.py --no-sync --angle build/skia/angle
python3 wink/test/skia/run.py reference --route gles --reference-angle \
  --sources gm --match '^aaclip$' --out build/skia/clip-reference-gles
```

Only a reference invocation explicitly selects ANGLE; candidates always use
Wink. The runner checks the actual renderer and hashes both ANGLE libraries.
ANGLE Metal does not supply the GLES 3.1 required by Dawn, so this reference
configuration covers Ganesh GLES. Graphite Dawn GLES still needs an independent
GLES 3.1 reference.

Comparisons require matching Skia revisions, fixtures, route, sources, and
selection. They compare DM's raw output hashes, including alpha and output
format, rather than an RGB conversion or an eroded image mask. Missing,
extra, duplicate, or empty image results fail. Use Skia's `skdiff` on output
images to diagnose differences. No global image tolerance or failure allowlist
is applied.

## Gates

A native run fails on a nonzero exit, timeout, changed task inventory,
unstarted/unfinished task, skipped draw, absent or incorrect backend identity,
GPU validation/device error, or missing `dm.json`. A unit run alone checks
execution/assertions; rendered correctness additionally requires reference
comparison. Backend-inapplicable tests and upstream bugs remain visible until
their exclusions have independent evidence. Native runs also hash the loaded
Wink library and the selected ICD/library when present, so an uncommitted
driver build is distinguishable from another build at the same Git revision.

Run harness integrity checks with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s wink/test/skia -p 'test_*.py' -v
```

Full acceptance additionally requires complete browser ST/MT corpus runs,
browser outer Dawn GLES integration, independent Graphite Dawn GLES references, full CTS regression
comparisons, and repeated cold/warm-cache runs. Successful isolated browser
GMs or the native SkSL suites do not establish that acceptance.
