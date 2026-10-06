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
  for it) is also accepted. **Compile with upstream 3.1.56 anyway.** Every
  .NET-built emsdk measured has the same mismatch: the workload pack (10.0.12)
  and the dotnet/runtime-provisioned one that fna-wasm-build freezes. Their
  clang (dotnet-llvm-project 19.1.0) emits the newer `__wasm_setjmp` helpers,
  while their compiler-rt still defines only `saveSetjmp`, so code with setjmp
  in it, such as spirv_to_nir, will not link. ikvm-wasm-build compiles its
  natives with upstream 3.1.56 and lets .NET's emsdk link them for the same
  reason, and wink's default does the same. The symbol check stops the build,
  and says why, if the compiler and compiler-rt disagree.
* **emdawnwebgpu v20250822.104650**, the `webgpu.h` implementation, verified
  by sha512 and patched for 3.1.56 (`tools/patch-emdawnwebgpu.py`). It's the
  first release with `texture-formats-tier1`/`tier2`, which the drivers
  request where the browser has them (Chrome 154 does).
* **Tint** (SPIR-V → WGSL), built for wasm from Dawn `cdda346`, once per
  variant: the last Dawn with the AST SPIR-V reader wink uses, patched to
  read the tier1 storage formats (`patches/tint-spirv-reader-texel-formats.patch`).
  To reuse ikvm-wasm-build's `tint-out-<variant>`, set
  `WINK_TINT_DIR_ST` / `WINK_TINT_DIR_MT`. The native Dawn package
  (`src/gallium/drivers/gex_wgpu/native/build-dawn.sh`) is the same revision,
  built from the same patched source; natively, Dawn's Metal backend offers
  the tiers only with its `allow_unsafe_apis` toggle, which both drivers set.
  It also has `patches/dawn-metal-strict-math-contract.patch`: Metal's safe
  math mode still fuses a multiply and an add, and the Vulkan driver asks
  for strict math where a shader says not to. And
  `patches/dawn-metal-command-buffer-errors.patch`: a command buffer Metal
  fails -- a GPU hang or fault, after which the system ignores the
  process's work -- is logged and loses the device, which Dawn let pass.
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
  Others get `WINK_ERROR_WRONG_THREAD`. (See below for the exception.)
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
  event loop. By default pacing is the caller's job (requestAnimationFrame),
  and `wink_swap_buffers` never suspends.
  * A loop that never returns to the browser should call
    `wink_swap_interval(surface, n)`. Each swap then suspends for `n`
    animation frames, or for one turn of the event loop when `n` is 0.
  * `wink_yield_frames(n)` is the same wait without a swap, for a windowing
    layer's other paths.
  * Neither waits more than a second, because a hidden tab never runs
    requestAnimationFrame.

#### The canvas on a worker

This is the arrangement ikvmcraft uses for its .NET deputy thread:

1. Link with `-sOFFSCREENCANVAS_SUPPORT`.
2. Create the pthread with
   `emscripten_pthread_attr_settransferredcanvases(&attr, "#canvas")`. The
   worker receives the canvas as an OffscreenCanvas, and there `wink_init`,
   the device, the surface and GL all work in the ordinary mode.
3. On the worker, find the canvas. `wink_surface_create` looks it up as
   follows:
   * `Module.winkCanvas` wins;
   * then the canvas the worker owns under that DOM id (`#id`);
   * then `Module.canvas` for `#canvas` or no selector;
   * and, on a worker that owns exactly one canvas, that one, whatever the
     selector. ikvmcraft transfers `.canvas`, for one.
4. Enter wink through `WebAssembly.promising` on the worker too. A pthread's
   entry point isn't entered that way. The smoke test returns from it with
   the runtime kept alive, then calls its body through `promising` from a
   fresh task (`library_wink_test.js`). The .NET runtime wraps its mailbox
   and JSExports the same way.

`Module` on a worker is that worker's own. Emscripten doesn't copy the
page's `winkEnv` or `winkDevice` onto it, and a `GPUDevice` can't be
transferred anyway, so use `wink_setenv` there.

#### GL from a worker, run on the main thread

In `mt`, `WINK_INIT_GL_ON_MAIN_THREAD` in `wink_init_attribs.flags` keeps the
device, the canvas and the driver on the browser's main thread. The canvas
stays where it is, so there is no OffscreenCanvas transfer, and the worker
needs no JSPI. Any worker may then call wink and GL, but the main thread may
not, because it would wait for itself.

How it works:

* **Mesa's glthread, with main as its driver thread.** The worker's GL calls
  are marshalled into batches, as glthread always does. Instead of a Mesa
  thread, a JS runner on the main thread executes them
  (`util_queue_init_external`). Enqueueing a batch posts a message that
  schedules the runner.
* **Calls that need an answer wait for it.** `glGet*`, `glReadPixels`, query
  results and `glMapBuffer*` run on main while the worker blocks with
  `Atomics.wait`. Each GL function has a generated `_mesa_glthread_call_<F>`
  for this.
* **Client memory is not copied ahead.** Client-array draws and client-memory
  uploads run on main against the worker's memory. That is safe because both
  threads share one heap, and the worker waits until they finish.
* **JSPI only on main.** The runner enters wasm through
  `WebAssembly.promising` on a private 16 MB stack (`Module.winkRunnerStackSize`).
  It swaps the stack pointer around each suspension, so other main-thread
  code, such as the .NET runtime's, keeps its own stack. Main can therefore
  suspend for readbacks without the embedder wrapping anything.
* **Pacing.** `wink_swap_buffers` queues the present behind the frame's GL.
  It returns once the previous frame is on screen, so at most two frames are
  in flight. `wink_swap_interval` sets the number of animation frames per
  swap (default 1; 0 means just a turn of the event loop). Pacing therefore
  happens on main, and the worker never has to yield.

The main thread must keep returning to its event loop. Anything that blocks it
also blocks every GL call that waits for an answer. `st` builds refuse the flag
with `WINK_ERROR_BAD_ATTRIBUTE`. If `wink_init` itself is called on the main
thread, the flag is ignored.

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

Memory:

* Where the browser has no ETC2 (desktop Chrome on Windows and Linux),
  st/mesa decodes those textures to RGBA8, four to eight times their size,
  and keeps the compressed data too. `transcode_etc=true` (driconf's name,
  in `winkEnv`) stores them as BC1/BC3 where the device has BC, about the
  size they were: lossy -- the CTS's ETC cases fail at its threshold of 2
  in 255 -- and each upload is a BC compression on the CPU.
  (`transcode_astc`, st/mesa's compute pass for ASTC, does not work on this
  driver yet.)
* `GEX_WGSL_CACHE_BYTES` bounds the SPIR-V to WGSL translation cache
  (32 MiB); the page can set it with `gex_wgsl_cache_budget(bytes)`.
* Link with `-sMALLOC=dlmalloc` (the default) rather than mimalloc in `mt`:
  mimalloc keeps memory for each thread that allocated, and the wasm heap
  never shrinks, so what any thread held at its peak stays.

## Natively, on macOS

Against native Dawn (the package `src/gallium/drivers/gex_wgpu/native/`
builds), the same API is `libwink.dylib`, for a windowing layer to make its
GL. The wink fork of SDL does that in its Cocoa backend. A window's frames go
to a CAMetalLayer of its own, BGRA8, presented FIFO at every swap.

```sh
PKG=~/.cache/mesa-wink/dawn-native/pkg
PATH=/opt/homebrew/opt/bison/bin:/opt/homebrew/opt/flex/bin:$PATH \
meson setup build/wink-mac -Dvulkan-drivers= -Dgallium-drivers=gex_wgpu \
  -Dplatforms= -Dglx=disabled -Degl=disabled -Dopengl=true -Dgles1=disabled \
  -Dgles2=enabled -Dgbm=disabled -Dllvm=disabled -Dvideo-codecs= -Dtools= \
  -Dvulkan-layers= -Dgex-emdawn-dir=$PKG -Dgex-dawn-native-dir=$PKG/lib \
  -Dgex-tint-dir=$PKG -Dwink=enabled
ninja -C build/wink-mac src/gallium/targets/wink-wasm/libwink.dylib

cmake -S ../SDL -B build/sdl-mac -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSDL_COCOA_WINK_INCLUDE_DIR=$PWD/src/gallium/targets/wink-wasm \
  -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF
ninja -C build/sdl-mac
```

`SDL_VIDEO_COCOA_WINK=path/to/libwink.dylib` (a hint, or the environment)
makes SDL's GL wink's in place of NSOpenGL. Contexts follow the usual
`SDL_GL_SetAttribute`s, and the surface follows the window's pixel size.

Here the library **does** export `gl*`, unlike the wasm build. They are
linker aliases of the dispatch stubs that `wink_get_proc_address` returns
(`wink_gl_aliases.py`), because LWJGL finds GL by `dlsym` and Minecraft
checks the result against `SDL_GL_GetProcAddress`. Nothing else is exported.
Tint's archive has default visibility, and its weak symbols, if exported,
would be coalesced with libwebgpu_dawn's copy of Tint.

`wink/test/native/run.sh build/wink-mac/src/gallium/targets/wink-wasm/libwink.dylib build/sdl-mac`
opens a window and checks that:

* `dlsym` and SDL resolve `gl*` to the same functions;
* a clear and a GLSL draw read back, before and after a resize.

`WINK_CAPTURE=N:file.bmp` writes the back buffer of the Nth swap. To run
Minecraft's OpenGL backend on it, from a Prism Launcher instance:

```sh
src/wink/vulkan/test/run-minecraft.py 26.4-snapshot-1 --gl --world "New World" --capture 1500 --seconds 150
```

That sets `org.lwjgl.sdl.libname` to the fork, and `org.lwjgl.opengl.libname`
and `SDL_VIDEO_COCOA_WINK` to libwink. 26.3 and 26.4-snapshot-1 both render
their title screens and worlds.

### Conformance

The GL and GL ES conformance suites run on libwink here, through
`wink/test/cts`:

```sh
wink/test/cts/build-gl-cts.sh                  # into ~/.cache/mesa-wink/gl-cts
wink/test/cts/run-gl-cts.sh gles OUT           # dEQP-GLES2/3/31, KHR-GLES2/3/31/32
wink/test/cts/run-gl-cts.sh gl OUT             # KHR-GL46, KHR-Single-GL46
wink/test/cts/run-gl-cts.sh gl OUT -- --fraction 50 --jobs 4
```

**Build.** It uses VK-GL-CTS at Mesa CI's pins. `opengl-cts-4.6.8.1` and
`opengl-es-cts-3.2.14.1` are one commit, so one build serves both. It applies
Mesa CI's backports, but not its patch that stops building Vulkan utilities:
that patch breaks dEQP's macOS platform, and it only saves build time.

**Context factory.** dEQP's macOS platform only creates CGL (Apple GL)
contexts. `deqp-osx-wink-context-factory.patch` adds a "wink" context type,
chosen with `--deqp-gl-context-type=wink`. It loads the libwink.dylib that
`DEQP_WINK_LIBRARY` names, and its surfaces are offscreen ones, like the
pbuffers Mesa CI runs with.

**Runs.** The suites are the ones Mesa CI runs for llvmpipe, with the same
256x256 `rgba8888d24s8ms0` surface, plus KHR-GLES32 and the 4.6 lists. The
expectations are `src/gallium/drivers/gex_wgpu/ci/gex_wgpu-*.txt`, so a run's
`failures.csv` lists only what changed.

**Results (full runs, 2026-09-28, Apple M4).** Pass counts include
warnings and flakes.

| suite | cases | run | pass | fail, crash, timeout |
| --- | --- | --- | --- | --- |
| GL ES (dEQP + KHR) | 112,212 | 111,069 | 92.9% | 7,935 |
| GL 4.6 (KHR) | 31,062 | 20,298 | 98.6% | 295 |

Where GL ES fails most:

- stencil operations, on FBOs as well as the default framebuffer;
- blending;
- texture filtering;
- 32-bit `copy_image`;
- transform feedback arrays;
- fixed-point and unsigned fragment outputs.

Desktop GL's skips are extensions wink does not have: subgroups, sparse
textures, fragment shading rate and mesh shaders. Some of the fp64 builtin
tests take close to the 60-second limit, because their shaders are software
doubles.

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
starts a thread. In `mt` it runs the whole suite twice more, from pthreads:
once with `WINK_INIT_GL_ON_MAIN_THREAD` (`main-thread.html`), and once with
the canvas transferred to the worker and JSPI there (`offscreen.html`).

`test/dotnet/` links a variant into a real .NET 10 browser-wasm app through
`wink.targets`, then checks from managed code that the native side is there.
It only calls functions that never suspend, because stock .NET does not wrap
its exports in `promising`. Both variants link and pass with the stock .NET
10.0.401 SDK and `wasm-tools` workload.

```sh
dotnet publish wink/test/dotnet -c Release -p:WinkPackage=$PWD/build/wink/out/st -o /tmp/wink-dotnet
node wink/test/run-browser.mjs /tmp/wink-dotnet/wwwroot
```
