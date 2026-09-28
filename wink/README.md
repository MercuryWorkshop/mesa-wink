# wink: Mesa on WebGPU, for a static .NET wasm link

`wink/build.sh` builds Mesa's OpenGL and OpenGL ES on the `gex_wgpu` Gallium
driver as static libraries a .NET 10 browser-wasm app can link. The toolchain
is Emscripten 3.1.56, the one .NET 10 pins, and there are two builds:

| variant | for | built with |
| --- | --- | --- |
| `st` | .NET's default, single-threaded runtime | no `-pthread` |
| `mt` | `WasmEnableThreads=true` | `-pthread` everywhere, Tint included |

Both expose desktop GL 4.6 core and compatibility and GL ES 3.2, on the
adapters measured so far. The GL version depends on the device's WebGPU
limits: with 16384 textures and 1024 compute invocations it is 4.6; a
Chromebook-class 256 invocations drops desktop GL below 4.3, while ES 3.2
only needs 128.

## Building

```sh
wink/build.sh                 # both variants into build/wink/out/{st,mt}
wink/build.sh --variant=st    # one of them
wink/build.sh --test          # and run the browser smoke test on each
wink/build.sh --install-deps  # install missing host tools first
```

The host needs only what a Linux development machine normally has: git, curl,
cmake, ninja, bison ≥ 3, flex and Python ≥ 3.10. Node ≥ 22 and Chrome are
needed for `--test`. Everything else goes into a cache (default
`~/.cache/mesa-wink`, or `--cache=`):

* **emsdk 3.1.56**, bootstrapped. To use one you already have, pass
  `--emsdk=DIR` or set `$EMSDK`. That can be an upstream emsdk, or the layout
  dotnet/runtime provisions (`src/mono/browser/emsdk`, which is what
  fna-wasm-build's `freeze-emsdk.sh` zips). That frozen emsdk only has Linux
  binaries. emcc's version is checked, and `WINK_ALLOW_ANY_EMSDK=1` overrides
  the check.

  The .NET workload's own Emscripten pack (or a dotnet root, which is searched
  for it) is also accepted, but as of 10.0.12 it cannot build wink. Its clang
  (dotnet-llvm-project 19.1.0) emits the newer `__wasm_setjmp` helpers, while
  its own compiler-rt still defines only `saveSetjmp`, so code with setjmp in
  it, such as spirv_to_nir, will not link. The symbol check stops the build and
  says so. Upstream 3.1.56 is what dotnet/runtime provisions and what the
  runtime's own archives link against.
* **emdawnwebgpu v20250428.134257**, the `webgpu.h` implementation, verified
  by sha512 and patched for 3.1.56 (`tools/patch-emdawnwebgpu.py`).
* **Tint** (SPIR-V → WGSL), built for wasm from Dawn `b465e23`, once per
  variant. To reuse ikvm-wasm-build's `tint-out-<variant>`, set
  `WINK_TINT_DIR_ST` / `WINK_TINT_DIR_MT`.
* A Python venv with mako, pyyaml and packaging, plus meson if the host's is
  older than 1.4.

**macOS:** Homebrew supplies what is missing (`brew install bison flex meson
ninja cmake python@3.13 node`). bison and flex are keg-only, and the script
puts them ahead of Apple's bison 2.3. The scripts run under macOS's bash 3.2.

**Arch:** `sudo pacman -S --needed base-devel git curl python cmake ninja
bison flex nodejs` (and `chromium` for `--test`).

Every stage records what it was built from, so re-running is cheap, and a
stage that failed half way starts over. `--until=deps|venv|emsdk|emdawn|tint|mesa`
stops early; for example, `--until=tint` warms a CI cache.

## What you get

`build/wink/out/<variant>/`:

| path | what |
| --- | --- |
| `lib/libwink.a` | one relocatable object: st/mesa, Gallium, GLSL/NIR, gex_wgpu, the wink API |
| `lib/libwink_webgpu.a` | emdawnwebgpu's `webgpu.cpp`: link exactly once per module |
| `lib/libwink_tint.a` | Tint. Leave it out if the app already links the same Tint |
| `js/*.js` | the `--js-library` files, **in the order `manifest.json` lists them** |
| `include/` | `wink.h`, `wink_emscripten_compat.h`, `webgpu/webgpu.h` |
| `manifest.json` | versions, required, recommended and forbidden link flags, file hashes |
| `wink.rsp` | the emcc link flags, as a response file |
| `wink.targets` | MSBuild import for a .NET app |
| `symbols/` | what `libwink.a` defines and needs (see below) |

### Linking into .NET

```xml
<Import Project="path/to/build/wink/out/st/wink.targets" />
```

That adds the three archives as `NativeFileReference`s and the JS libraries
and required flags to `EmccExtraLDFlags`. It also refuses a variant that
doesn't match `WasmEnableThreads`. P/Invoke with `[DllImport("libwink")]`.

Import it after the app's own properties. These knobs are read at import:

| knob | when |
| --- | --- |
| `WinkLinkWebGPUGlue=false` | the app already links emdawnwebgpu's `webgpu.cpp` |
| `WinkLinkWebGPUJs=false` | the app passes its own four `library_webgpu*.js` (defaults to the glue knob). They must be the same emdawnwebgpu release, and wink's copies carry fixes a stock copy lacks |
| `WinkLinkTint=false` | the app already links the same Tint (MobileGL's, from the same Dawn revision) |

The stack is raised to 16 MB unless the app already chose one, through
`EmccStackSize` or a `STACK_SIZE` in `EmccExtraLDFlags`. GL runs on whatever
thread calls it, and that thread's stack is the one that matters; with
`WasmEnableThreads`, that is a pthread's.

Link flags, from the manifest:

* **Required:**
  * `-fwasm-exceptions -sSUPPORT_LONGJMP=wasm`. .NET links with wasm
    exception handling, and Mesa uses setjmp.
  * `-sWASM_BIGINT=1`.
  * `-pthread` for `mt`.
* **Recommended:** `-sSTACK_SIZE=16MB` (Tint recurses deeply), an initial
  memory above that, `-sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB`.
* **Forbidden:**
  * `-sUSE_WEBGPU=1`, a second and incompatible `webgpu.h` implementation.
  * JS-mode exceptions or longjmp.
  * `-fPIC` objects in the same link. Under PIC, `gl*` addresses resolve to
    null.

## Using it

There is no EGL, and **no `gl*` symbol is exported**. GL is reached only
through `wink_get_proc_address()`, because a static wasm link has one
namespace, and a bare `glFoo` would collide with, or silently bind to,
Emscripten's WebGL.

```c
wink_init(NULL);                                   /* may suspend */

wink_surface_attribs sa;
wink_surface_attribs_default(&sa);
sa.canvas = "#canvas";
sa.width = 1280; sa.height = 720;
wink_surface *surface = wink_surface_create(&sa, NULL);

wink_context_attribs ca;
wink_context_attribs_default(&ca);
ca.api = WINK_API_OPENGL_ES; ca.major = 3; ca.minor = 2;
wink_context *ctx = wink_context_create(&ca, NULL);

wink_make_current(ctx, surface, surface);
void (*glClear)(unsigned) = wink_get_proc_address("glClear");
/* ... */
wink_swap_buffers(surface);                        /* never suspends */
```

* **Contexts are not tied to a surface.** Like EGL_KHR_no_config_context, any
  context can be made current with any surface, or with none.
* **Versions are never silently lowered.** A version that can't be had is
  `WINK_ERROR_BAD_VERSION`; 0.0 asks for the highest there is.
* **Legacy seam.** `wink_emscripten_compat.h` is the WebGL-context-shaped seam
  ikvm-wasm-build's emscripten-glfw dispatches to (`wink_webgl_create_context`
  and so on), for use until the move to SDL3.

### Threads and JSPI

* **Owner thread.** WebGPU objects belong to the JavaScript realm that created
  them, so every wink call must come from the thread that called `wink_init`.
  Others get `WINK_ERROR_WRONG_THREAD`.
* **Waiting uses JSPI.** Anything that waits for the GPU (`wink_init` itself,
  and GL calls that read back, from `glReadPixels` to a query result) suspends
  through JSPI. The export that leads there must be entered through
  `WebAssembly.promising(<raw wasm export>)`, with no JS frames in between.
  `library_wink.js` wraps its imports in `WebAssembly.Suspending` itself.
  There is no `-sJSPI`, which is how ikvm-wasm-build's .NET runtime already
  paces frames.
* **Stock .NET runtimes.** They do not wrap exports in `promising`;
  fna-wasm-build's runtime fork does (`WasmEnableJSPI`). Without JSPI,
  `wink_init` returns `WINK_ERROR_NO_JSPI` rather than hanging later.
* **Frames.** A frame reaches the canvas when the owning thread returns to its
  event loop. Pacing (requestAnimationFrame) is the caller's job.

### Page-side knobs

All are optional properties of the Emscripten `Module`:

| property | effect |
| --- | --- |
| `winkDevice` / `preinitializedWebGPUDevice` | a `GPUDevice` to use instead of requesting one |
| `winkAdapterOptions` | passed to `navigator.gpu.requestAdapter` |
| `winkDeviceDescriptor(adapter, desc)` | amend or replace the device request |
| `winkCanvas` | a canvas, or `function(selector) → canvas`, for workers and embedders |
| `winkEnv` | `{NAME: value}` applied with setenv at `wink_init`, e.g. `GEX_WGPU_TIMING`, `MESA_DEBUG` |

Driver switches worth knowing:

* `GEX_WGPU_NO_ASTC`, `GEX_WGPU_NO_ETC2`, `GEX_WGPU_NO_BC` hide a compression
  feature, so st/mesa decodes on the CPU instead.
* `GEX_WGPU_TIMING=1` shows the share of time spent in draws and in present.

## Symbol checks

The packaging stage runs `tools/wink-symbols.py`, which **fails** on:

* any `gl*`/`egl*`/`glX*` symbol, defined or undefined;
* JS-mode exception or longjmp shims;
* WebGL references;
* strong duplicates between `libwink.a` and the Tint or webgpu archives;
* in `st`, pthread functions that the single-threaded libc lacks.

It warns about Mesa's C11 thread functions, which share names with libc's.
`--check-against=a.a:b.a` reports strong collisions with your own archives
(`symbols/collisions.txt`).

Checked against ikvm-wasm-build's current natives, the only collision is with
NG-GL4ES in `libglfw3.a`, which carries its own copy of Mesa's evaluator math.
The four symbols are `_math_init_eval`, `_math_horner_bezier_curve`,
`_math_horner_bezier_surf` and `_math_de_casteljau_surf`. They are renamed on
the NG-GL4ES side, with the same two-pass prefix `build-glfw.sh` already uses
for MobileGL, rather than in Mesa.

## Tests

`--test` links `test/wink_smoke.c` against each packaged variant with plain
emcc, exactly as the manifest says to, and runs it in headless Chrome
(`test/run-browser.mjs`; `CHROME=` picks the browser). For GL ES 3.2, core
4.6 and compatibility 4.6 it checks:

* clear and read back (a JSPI suspend);
* a draw in each API's shading language;
* swap, resize and read again;
* that too high a version is refused.

It also samples ASTC (4x4, and 6x6 blocks on a 7x5 texture) and ETC2 blocks.
Then it shuts down and repeats everything on a fresh device with ASTC and ETC2
hidden, to exercise st/mesa's CPU decode. In `st` it also proves Mesa never
starts a thread.

`test/dotnet/` links a variant into a real .NET 10 browser-wasm app through
`wink.targets`, then checks from managed code that the native side is there.
It only calls functions that never suspend, because stock .NET does not wrap
its exports in `promising`. Both variants link and pass with the stock .NET
10.0.401 SDK and `wasm-tools` workload.

```sh
dotnet publish wink/test/dotnet -c Release -p:WinkPackage=$PWD/build/wink/out/st -o /tmp/wink-dotnet
node wink/test/run-browser.mjs /tmp/wink-dotnet/wwwroot
```
