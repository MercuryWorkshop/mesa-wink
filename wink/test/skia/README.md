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

The current native build enables `gles`, `vk`, and `grvk`. It does not yet
build Graphite's outer Dawn GLES adapter. The runner recognizes
`grdawn_gles`, but an unavailable configuration fails inventory and is not
counted as tested.

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

A GLES reference needs a separately instrumented DM build using an independent
GLES implementation (`--dm` and `--source`). The Wink GLES adapter cannot
produce an independent reference; the runner rejects it as one.

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
their exclusions have independent evidence.

Run harness integrity checks with:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s wink/test/skia -p 'test_*.py' -v
```

Full acceptance additionally requires the browser ST/MT matrix, outer Dawn
GLES integration, independent GLES references, full CTS regression comparisons,
and repeated cold/warm-cache runs. These are not implied by the native runner.
