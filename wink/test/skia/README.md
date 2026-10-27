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

GLES always uses Wink's context factory and proc-address API. Vulkan uses an
explicit ICD manifest (`--icd` overrides the default native build). GPU unit
context factories are restricted to the selected route, avoiding coverage
being attributed to a different backend. Unit tests which do not exercise an
observed GPU context are reported as unresolved, rather than passing silently.

## Independent reference and comparison

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
