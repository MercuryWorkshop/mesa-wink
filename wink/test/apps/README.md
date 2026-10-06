# Application rendering correctness

This suite records real applications on this Mac in an Apple `container`
Linux ARM64 VM, using source-pinned llvmpipe and lavapipe. Native Wink and
browser WebGPU replay the **same API capture**. The browser runs upstream
apitrace or GFXReconstruct with Wink context, loader, surface and fence adapters;
application code is not ported to WebAssembly. Upstream GL handle/uniform
remapping and Vulkan object/memory remapping remain in use.

The full coverage gate contains 380 cases: three API proofs, nine app/API
combinations, and 23 shaderpacks × eight scenes × two presets. The fast gate
contains 11 cases. Missing assets, unsupported features, failed calls,
timeouts, unstable references and missing checkpoints fail the gate.

| Workload | Coverage |
| --- | --- |
| Iris/Fabric Minecraft 1.21.1 | 18 locked public packs and five publisher ZIPs; day, night, rain, underwater, Nether, End, motion, LabPBR; default and pack-specific stress settings |
| Godot 4.4.1 | Actual GLES compatibility, Vulkan Mobile and Forward+; materials, depth, shadows, text |
| Filament 1.60.0 | GLES and Vulkan; bundled DamagedHelmet, opaque/TAA/SSAO automation |
| PPSSPP 1.18.1 | GLES and Vulkan; public PSP autotest `3dstudio.prx`, bounded gameplay frames |
| Qt Quick 6 | GLES and Vulkan; gradients, blending, transforms, clipping, text and a one-pixel border |

## Prepare

Use Python 3.12+, Node 22+, Chrome with WebGPU/JSPI, CMake, Ninja, git and
Apple `container`. The source builders need network access. References use
four vCPUs, 8 GiB RAM and four llvmpipe threads. Allow at least 20 GiB free
for an initial extended build; a full shader recording campaign needs much
more storage. Recording stops admitting new cases below 4 GiB free.
The small `wink-reference` image contains only the CPU drivers and capture
tools; `--apps` adds Godot, Qt and Java; `--extended` adds Filament and PPSSPP.
The extended final image excludes their source trees and build tools added
for those two source builds.
`build-extended.py` alternatively builds their locked runtime tools in a
fresh VM, bypassing BuildKit. Failed build VMs retain compilation caches and
can be resumed with the printed `--resume` name. Successful VMs are deleted.
`--sources=DIR --source-lock=FILE` reuses checked source archives in a fresh
output root without cloning another copy.

Build Wink's native GL library and Vulkan ICD using the repository's
[native GL instructions](../../README.md) and
[Vulkan instructions](../../../src/wink/vulkan/README.md). Build the browser
packages with `wink/build.sh`. Defaults use the existing `build/wink-mac`,
`build/wvk-native` and `build/wink/out/{st,mt}` layouts; `--library`, `--icd`
and `--emsdk` override the relevant paths.

From the repository root on a Homebrew Mac:

```sh
/opt/homebrew/bin/python3 -m venv build/wink-apps/venv
source build/wink-apps/venv/bin/activate
python -m pip install -r wink/test/apps/requirements.txt
python wink/test/apps/bootstrap-reference.py --apps --image wink-reference-apps
python wink/test/apps/build-extended.py --image wink-reference-apps
python wink/test/apps/assets.py
python wink/test/apps/toolchains.py
python wink/test/apps/minecraft.py prepare --platform linux
python wink/test/apps/build-fixtures.py
python wink/test/apps/recipes.py --tools build/wink-apps/extended-tools
```

`pins.json` fixes the CPU Mesa revision, capture tools and app source
commits. Publisher downloads use checked SHA-512 hashes from the tracked
asset locks. Minecraft/Fabric libraries and assets use their published
checksums. `assets.py --refresh` explicitly resolves new releases into the
output lock; review that lock before replacing the tracked version.

SEUS Renewed, KUDA, projectLUMA, Chocapic13 and Sildur's Enhanced Default
require publisher ZIPs. The source links are in `corpus.json`. Register each
local ZIP and regenerate the recipes:

```sh
python wink/test/apps/assets.py --manual seus-renewed=/path/to/publisher.zip
python wink/test/apps/recipes.py --tools build/wink-apps/extended-tools
```

Stress settings come from actual options/profiles in each locked ZIP, with a
curated Potato preset. `--settings=DIR` supplies reviewed `PACK_ID.txt`
settings. The public SPBR LabPBR resource pack is locked and downloaded
automatically. Missing packs or stress settings stay in the case matrix
and fail coverage. No default-preset substitution occurs.

## Build the retracers

```sh
python wink/test/apps/build-replay.py gl --target native
python wink/test/apps/build-replay.py vulkan --target native
python wink/test/apps/build-replay.py gl --target st
python wink/test/apps/build-replay.py vulkan --target st
python wink/test/apps/build-replay.py gl --target mt
python wink/test/apps/build-replay.py vulkan --target mt
```

Cached source checkouts are owned by the suite. Adapters are reapplied
deterministically; native and Wasm Vulkan use separate source directories.
Single-thread GL replay restores the context and draw/read surfaces of each
recorded thread. EGL fence operations use corresponding GL sync operations;
client waits and destruction temporarily bind their producer context when
needed. Other fence types or attributes fail replay. These adapters are also
applied to the software retracers, retaining upstream object remapping.
An Emscripten 3.1.56 ThinLTO workaround resolves the mt HTML5 callback before
link-time optimization. The browser publishes raw Wasm `main` for JSPI and
returns PNGs and logs through the existing Chrome runner.

To update retracers against an existing reference image without rebuilding
it through BuildKit, build the checked CPU tools in a fresh VM:

```sh
python wink/test/apps/build-reference-replay.py --out build/wink-reference-tools --sources build/wink-apps/reference-context --source-lock build/wink-apps/reference.lock.json --image wink-reference-apps
```

Add `--reference-tools=build/wink-reference-tools/reference-tools` to the
`record` commands below. This verifies all exported files and the base image
identity, then locks the complete tool manifest into each accepted baseline.
The builder retains failed VMs for `--resume` and deletes successful VMs.

## Record and compare

First prove the capture/replay oracle itself. These references must match
live application pixel readback exactly and match three CPU replays:

```sh
python wink/test/apps/suite.py record --case probe-gl --case probe-gles --case probe-vulkan --image wink-reference-apps
python wink/test/apps/suite.py run --case probe-gl --case probe-gles --case probe-vulkan --target native --target st --target mt
```

Then record the app and shader cases. `list` shows the concrete matrix and
commands before running it:

```sh
python wink/test/apps/suite.py list --tier full --recipes build/wink-apps/recipes.json
python wink/test/apps/suite.py record --reuse-references --recipes build/wink-apps/recipes.json --image wink-reference-apps --tools build/wink-apps/extended-tools
python wink/test/apps/suite.py run --recipes build/wink-apps/recipes.json --target native --target st --target mt
```

Add `--tier full` to record/run the whole matrix, or repeat `--case ID` to
select cases. `--timeout` controls each recording/replay. A timed-out
recording stops its named VM. Each completed reference needs three identical
CPU replays and every requested checkpoint; Iris must log that its pack is
active and the fixture completed 96 frames. The fixture builds a seeded
material scene, freezes world rules, uses a fixed camera and moves it by
rendered frame for the motion case. Warmup/setup calls remain in the trace.

A locked baseline cannot be overwritten. `--reuse-references` validates its
case, trace and checkpoint hashes and records only missing cases. Refresh into a new `--out` root,
prepare its assets/fixtures/recipes there, and build its reference image to
write provenance. `--replayers=build/wink-apps` reuses existing retracers.
Partial failed captures are archived before retrying. Trace and checkpoint
hashes are checked before every candidate replay. Baselines also record the
image identity, case/input hashes, CPU identity and capture-runner hash.
External tools carry source/archive, compiler-package and runtime-file
checksums; recording verifies the complete tool tree and locks its manifest.
Recipes also lock Godot scene files, Filament batch settings, PPSSPP config
and launcher, and the Minecraft launcher. Regenerate recipes after changing
fixtures; changed inputs require a fresh baseline directory. Recording gives
`/dev/shm` a VM-backed tmpfs for emulated PSP memory.
Filament uses separate EGL/GLES and X11/Vulkan builds, with packaged ARM64
SDL and both material shader models. PPSSPP lets SDL create its GLES context,
defaults unknown display refresh rates to 60 Hz, and admits lavapipe's CPU
device only when `WINK_REFERENCE_CPU` is set. These source adapters are
included in the locked builder; Vulkan feature requests remain intact.
Large staging and browser files share APFS blocks while retaining independent
bytes. Image builds use content-derived input names and check transferred
checksums inside the VM before compiling, including on older container CLIs.

The default image gate uses the worst RGB channel, 8/255 tolerance, at most
0.5% raw differing pixels, one-pixel mask erosion and zero surviving
interior pixels. Qt text and clipping/border regions permit zero raw pixels
above tolerance. `policy.regions` adds further reviewed strict regions.
Erosion can hide thin defects; it is never the sole pass condition. CPU
repeatability and API-proof comparisons use zero tolerance and no erosion.

`results.json`, `junit.xml` and `report.html` describe the selected run. Each
report records its expected result count; an interrupted passing subset fails
the JSON and JUnit coverage gates. Each case retains its reference identity,
capture lock, logs and latest per-case
result; each successful candidate replay retains images, difference/mask
PNGs and `comparison.json`. Each candidate gets a fresh directory. Failed
feature/extension requests are reported without stripping them from the
capture. A partial selection passing does not imply the full matrix passed.

The live-readback oracle is enforced for the three API proofs. App baselines
require identical CPU replays and clean recording/replay logs. A software
retracer skipping an unavailable function rejects the reference even if its
images are repeatable. Stability alone does not prove that an extension
outside the tracer's wrapper coverage was fully captured.

Vulkan references also require `vulkan-lifetimes.json`, bound to the capture
checksum and the checker version. The reference runner streams the pinned
`gfxrecon-convert` JSON output through
[`reference/vulkan_lifetimes.py`](reference/vulkan_lifetimes.py), which checks
framebuffer and dynamic-rendering attachment lifetimes at recording time.
This focused check does not replace full Vulkan validation. Existing captures
can be audited without changing their traces, image baselines or locks:

```sh
container run --rm --disable-progress-updates --cpus 2 --memory 1g \
  --volume "$PWD/build/wink-apps-validated:/work" \
  --volume "$PWD/wink/test/apps/reference:/audit:ro" \
  --entrypoint python3 wink-reference-apps-v2 \
  /audit/vulkan_lifetimes.py /work/cases/filament-vulkan/capture.gfxr \
  --out /work/cases/filament-vulkan/vulkan-lifetimes.json
```

Browser replay preserves completed screenshots on a crash, while retaining
the failed result. Driver environment settings are installed before replay
starts for both API paths.

## Local validation

The 2026-10-06 run on this Mac has 13 admitted references: all three API
proofs, all three Godot backends, both Filament backends, PPSSPP GLES, both
Qt backends, and Complementary Reimagined/BSL day/default. Each matches three
software replays exactly. The proofs also match live pixel readback exactly.
Native, browser st and browser mt use identical captures and unchanged
comparison policies.

The original Qt GLES capture contained invalid `glPolygonMode` calls, and
Filament Vulkan used framebuffer attachments after destroying their images.
Those captures remain rejected and preserved under `superseded-references`.
Fresh captures use a Qt GLES profile that disables the desktop-only RHI
function, a Qt Vulkan device enabling only the required features, and a
source-pinned Filament adapter retaining framebuffer attachment owners.
Every adapter and rebuilt tool is hashed in its reference's tool manifest.
The accepted traces were recorded again after these application fixes.

| Workload | Observed candidate result |
| --- | --- |
| GL, GLES and Vulkan API proofs | All nine target checks passed |
| Godot GLES | Passed on all three targets |
| Godot Mobile / Forward+ | Device creation failed; `VK_KHR_buffer_device_address` unavailable |
| Filament GLES / PPSSPP GLES | All four image comparisons passed on native, browser st and browser mt |
| Filament Vulkan | Repaired reference passes attachment lifetime admission; all four comparisons passed on all three targets |
| Qt Quick GLES / Vulkan | Fresh valid API profiles; all four comparisons passed on all three targets |
| Iris Complementary Reimagined day/default | All four comparisons passed on all three targets after fixing raw loads from comparison-enabled depth textures |
| Iris BSL day/default | All four comparisons passed on all three targets |
| PPSSPP Vulkan | No accepted reference; capture attempts exceeded time or disk limits |

The aggregate has **33 passes, six unsupported results and 1,101 missing
references**, covering all 1,140 case/target rows. The full 380-case campaign
has **not passed**. Only two shaderpacks in one scene/preset have accepted
references. Other scenes/presets and packs remain unrecorded, and five packs
need publisher ZIPs. The local report is
`build/wink-apps-validated/{report.html,results.json,junit.xml}`.
Saved candidate results are consolidated with reference admission reapplied.
No comparison thresholds were relaxed and no accepted capture was edited.

Framebuffer fetch uses native Dawn attachment inputs when available. Chrome
uses ordered primitive replay with typed attachment snapshots, including
multiple targets and individual MSAA samples; this path can be expensive for
large draws. Vulkan precise occlusion counts passing samples using a private
additive attachment and a 64-bit GPU reduction on backends with boolean
queries. See the [query limits](../../../src/wink/vulkan/README.md) for its
single-draw overdraw and early-test limitations. Unaligned, suballocated
read-only Vulkan buffers are copied on the GPU into aligned bindings.

## Refresh an application reference

Build corrected tools in a fresh output root. Filament reuses the individually
hashed source archives from the extended tool build:

```sh
python wink/test/apps/build-qt.py --out build/wink-qt-fixed --image wink-reference-apps
python wink/test/apps/build-filament.py --out build/wink-filament-fixed --image wink-reference-apps --sources build/wink-apps/reference-context --source-lock build/wink-apps/extended-tools/tools.lock.json
```

Generate recipes in the fresh recording root with `--qt-tools` or
`--filament-tools` pointing to the builder's exported runtime tree. Record
with `suite.py record` and replay the new captures on every target before
installing them into an existing campaign. `--min-free-space-gib` adjusts the
recording disk reserve for small fixtures; its default remains four GiB.
For Qt, reuse the pinned base image provenance and checked CPU retracers:

```sh
cp build/wink-apps/reference.lock.json build/wink-qt-fixed/reference.lock.json
python wink/test/apps/recipes.py --out build/wink-qt-fixed --qt-tools build/wink-qt-fixed/qt-tools
python wink/test/apps/suite.py record --out build/wink-qt-fixed --recipes build/wink-qt-fixed/recipes.json --image wink-reference-apps --tools build/wink-qt-fixed/qt-tools --reference-tools build/wink-reference-tools/reference-tools --case qtquick-gles --case qtquick-vulkan
python wink/test/apps/suite.py run --out build/wink-qt-fixed --recipes build/wink-qt-fixed/recipes.json --replayers build/wink-apps --case qtquick-gles --case qtquick-vulkan --target native --target st --target mt
```

Reference replacement is explicit and requires a reason:

```sh
python wink/test/apps/refresh-reference.py --source build/wink-qt-fixed --out build/wink-apps-validated --case qtquick-gles --case qtquick-vulkan --reason 'Correct desktop-only GLES calls and the Vulkan device feature profile' --dry-run
python wink/test/apps/refresh-reference.py --source build/wink-qt-fixed --out build/wink-apps-validated --case qtquick-gles --case qtquick-vulkan --reason 'Correct desktop-only GLES calls and the Vulkan device feature profile'
```

The tool validates the whole selection before changing references, preserves
comparison policy and minimum checkpoint coverage, archives superseded case
directories, and records old/new capture hashes in `reference-refreshes.json`.
Immutable captures and CPU checkpoints can share disk space through hard
links; mutable candidate logs and results are copied independently.

## Harness checks

```sh
python -m unittest discover -s wink/test/apps -p 'test_*.py'
node --check wink/test/run-browser.mjs
```

The tests cover channel-specific errors, thin/raw and strict-region defects,
missing/incorrect images, validation errors, process failures, timeouts,
altered captures and missing coverage in JSON/JUnit reports. They supplement
real native/browser replay checks; they do not certify app coverage.
The current harness passes all 35 checks, including expired Vulkan
attachments and software retracers that skip calls.
