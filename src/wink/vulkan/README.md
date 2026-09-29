# wvk: Vulkan on WebGPU

The wink Vulkan driver: Vulkan 1.2 with `VK_KHR_portability_subset`, on a
WebGPU device. It is Mesa's Vulkan runtime on top, and underneath it is the
shader compiler the GL driver (`gex_wgpu`) uses, in `src/wink/compiler`.
Natively it is an ICD over Dawn, which is where Vulkan's tools, vkcube and
dEQP-VK run. In a browser it is part of libwink, over emdawnwebgpu.

## How it works

- **Commands are recorded, and replayed at submit.** Every `vkCmd*` goes
  into the command buffer's `vk_cmd_queue`. A submit replays its command
  buffers through the driver's own entrypoints into one `WGPUCommandEncoder`.
  WebGPU command buffers are single-use and Vulkan's are not. Replaying late
  also lets a pass know, when it begins, what it will need.
- **One in-order queue.** The runtime's emulated timelines defer a submit
  until what it waits for is pending. On one in-order WebGPU queue, nothing
  is then left to wait for on the GPU: barriers and events cost nothing, and
  acquiring a swapchain image never blocks. A second queue family, of
  compute and transfer, has a queue that submits to the same WebGPU queue.
  Two families let two threads submit at once, so native Dawn is asked for
  `ImplicitDeviceSynchronization`.
- **Host-visible memory is a copy on the host.** Ranges a submit reads are
  uploaded with `wgpuQueueWriteBuffer` before it. Ranges it writes are read
  back after it, and the submit counts as finished only then.
- **Pipelines are made at the draw.** A WebGPU pipeline holds much of what
  Vulkan keeps as dynamic state, and its bind group layouts hold what only
  the shaders say. So pipelines are built from the bound shaders and the
  runtime's dynamic graphics state, and cached by all of it. Bind groups are
  built from descriptor sets and kept until the set is written again.
- **Shaders:** SPIR-V → NIR (vtn) → descriptors remapped, combined image
  samplers split, push constants as a storage buffer → zink's `nir_to_spirv`
  → Tint → WGSL. Set N is `@group(N)`, and each element of a Vulkan binding
  is a binding of its own. The driver's bindings sit at the top of group 3.
- **Vertex attribute divisors** other than 1 are read by the vertex shader
  itself, from the vertex buffer bound as storage, since WebGPU steps an
  instance attribute every instance; so are attributes of strides not
  whole words, or at offsets their formats are not aligned to, which
  WebGPU cannot lay out.
- **Shaders are compiled for some of the state they're drawn with.** What
  they depend on:
  - the vertex input they pull;
  - the fragment shader's interpolation, which WebGPU wants a vertex
    shader's outputs to have;
  - alpha to one, and coverage from alpha where WebGPU's won't do (with
    one sample, or with the shader's own mask);
  - the samples positions are placed for;
  - whether the pass has a depth attachment to write to;
  - where each set's records are.

  Shaders compiled apart don't know that state: shader objects, a fast
  link's pipeline libraries, and anything with dynamic vertex input or
  multisampling. Each keeps its source for that: its NIR as compiled, and
  copies of its set layouts, serialized with it. A draw whose state
  differs from what a shader was compiled for uses a variant compiled
  from the source and cached (`wvk_shader_draw_variant`).
  `WINK_VK_VARIANT_LOG=1` says when one is compiled, and why.
  Monolithic pipelines' shaders are compiled for their draws and have no
  variants.
- **64-bit floats**, which WGSL has none of, are Mesa's float64.glsl in
  integer arithmetic (built by glslang, where there is one), inlined by
  nir_lower_doubles; and 64-bit integers are pairs of words
  (nir_lower_int64). Slow, but there.
- **8- and 16-bit values**, which WGSL has no integers of, are 32-bit
  words of their bits (`wvk_nir_small_types.c`): sign-extended where read
  as signed, kept to their bits where made. Half floats of float
  operations stay WGSL's f16; words of them, from memory or moves, are
  packHalf2x16's bits. Memory is words already: a store of less than a
  word is atomics on its bits, and interpolated half floats are floats.
- **What WebGPU's objects cannot hold** of descriptors, shaders do
  themselves: each set has a record for each of its bindings, which the
  shader reads at the draw (`wvk_nir_descriptors.c`), from the push binding
  after the push constants. Texel buffers are
  storage buffers whose format the shader decodes and encodes, views'
  swizzles are applied to what is read through them, and samplers' border
  colours, mirroring once, LOD bias and unnormalized coordinates are the
  shader's. Integer textures, which WebGPU does not sample, are fetched
  from, with the sampler's address modes and LOD applied; an integer cube
  is the 2D array of its faces, the face chosen as Vulkan chooses it.
  Comparisons WGSL cannot make -- with gradients, a bias, or at a LOD
  other than 0 -- fetch and compare each texel of the filter by hand, and
  textureQueryLod is worked out from the sampler's record.
- **Image atomics**, which WGSL does not have, are on a storage buffer of
  the image's texels. A 32-bit integer storage image has one besides its
  texture, and a shader that does atomics on it uses it for everything.
  Between barriers one or the other is the image: the texture is copied to
  the buffer before such a shader runs, and back at the next barrier.
- **A pass that reads what it renders to** -- an input attachment of its own
  subpass, a feedback loop -- reads a copy, which WebGPU wants: a texture is
  an attachment or bound in a pass, not both. The copy is made before the
  draw, with the pass begun again after, and kept until a barrier: a
  texture copy, or where WebGPU copies none a pass loading each texel --
  each sample of a multisampled image, depth into R32Float and stencil
  into R8Uint. Depth and stencil in read-only layouts are read-only to
  WebGPU, which lets shaders read them as they are. Passes store what
  they render even where Vulkan does not care, since one may be ended
  and begun again before Vulkan's is over.
- **Views in other formats.** WebGPU views a texture only as its format
  or its sRGB twin. A view of another format of the same texel size, or an
  uncompressed view of a compressed image (a texel a block), is of another
  texture of the image's, in that format. The image keeps, at each level,
  which of its textures are up to date, and brings one up to date with
  copies through a buffer when a draw, a pass or a transfer uses it.
  A compressed image whose size is not whole blocks is sampled from a
  shadow of its own size, decoded from it by a pass when sampled after a
  write.
- **Formats from `texture-formats-tier1` and `-tier2`**, where the device
  has them: the 16-bit normalized formats as textures (they're vertex
  formats regardless), rendering to 8-bit snorm, storage for most 8- and
  16-bit and 10-bit packed formats (so `shaderStorageImageExtendedFormats`),
  and read-write storage for R8, R16, RGBA8, RGBA16 and RGBA32 formats.
- **16-bit packed formats** (565, 4444, 5551, 1555, and the extension ones:
  `VK_EXT_4444_formats`, maintenance5's A1B5G5R5), which WebGPU does not
  have, are RGBA8 textures. Copies to and from buffers are a shader's,
  which packs and unpacks the texels; a format without alpha keeps its
  texture's at one. Blending is at 8 bits, not the format's.
- **Memory is one WGPUBuffer**, which the buffers bound to it share. WebGPU
  lets a buffer written as storage have no other use in a usage scope --
  a dispatch, or a whole render pass -- which suballocated memory has all
  the time: a uniform buffer and a storage buffer of one allocation. What
  a draw or dispatch reads of a buffer it writes, it reads from a copy
  made before it; and a draw at odds with what its pass used before
  begins the pass again.
- **Arrays of descriptors** are bindings of their own. A dynamic or
  non-uniform index chooses among them in the shader: reads from each,
  selected; writes only where the index is.
- **Triangle fans** are the lists of their triangles, each in the order
  Vulkan's fan has it: a list the host makes, or for a draw by index one a
  compute shader makes of the application's indices, restarting where
  they do. **Polygon modes** of lines and points are the same shader's
  lists of each triangle's edges or corners. Indirect draws of fans are drawn from the host's copy of their
  commands. **vkCmdDraw*IndirectCount** copies the draws and gives those
  past the count no instances, on the GPU.
- **Stencil masks and references of each face**: WebGPU has one of each.
  Where the faces' differ and both are drawn, a draw is drawn twice, each
  culling the other face.
- **Viewports**: WebGPU has one. A vertex shader that writes
  gl_ViewportIndex has each draw drawn once per viewport, with what is not
  that draw's out of the clip volume, as layers are.
- **Passes of several layers or views** are replayed once for each: a
  WebGPU pass renders one layer. A vertex shader's `gl_Layer` chooses by
  moving vertices of other layers out of the clip volume, and the view
  index is the pass's. Input attachments are fetched at the layer drawn.
- **Timestamps** are the beginnings of empty compute passes, WebGPU having
  them only where a pass begins or ends; a render pass is ended for one and
  begun again. An occlusion query a pass is ended in goes on in the next,
  its parts added, and clears are not counted.
- **Blits and `vkCmdClearAttachments`** come from the runtime's vk_meta.
  Its passes of several layers are begun again at each.
- **Present** copies the swapchain image into the surface's current
  texture. WebGPU hands out a new texture every frame, and Vulkan's images
  have to stay.

## Building and testing, natively

Dawn with Metal on macOS, or Vulkan on Linux, from
`src/gallium/drivers/gex_wgpu/native/build-dawn.sh` and `package-dawn.sh`.
Then:

```sh
PKG=~/.cache/mesa-wink/dawn-native/pkg
meson setup build/wvk-native -Dbuildtype=debugoptimized -Dvulkan-drivers=wink \
  -Dgallium-drivers= -Dplatforms= -Dglx=disabled -Degl=disabled -Dopengl=false \
  -Dgles1=disabled -Dgles2=disabled -Dllvm=disabled -Dvideo-codecs= -Dtools= \
  -Dvulkan-layers= -Dgex-emdawn-dir=$PKG -Dgex-dawn-native-dir=$PKG/lib \
  -Dgex-tint-dir=$PKG
ninja -C build/wvk-native
export VK_DRIVER_FILES=$PWD/build/wvk-native/src/wink/vulkan/wink_devenv_icd.$(uname -m | sed s/arm64/aarch64/).json
vulkaninfo --summary
```

`test/build.sh OUT` builds the tests; they need glslangValidator, and SDL3
for the window test:

- `wvk_smoke`: memory, copies, clears, sync, and a headless swapchain.
- `wvk_draw`: draws, push constants, a negative viewport, textures,
  `vkCmdClearAttachments` and blits, all read back.
- `wvk_window [--resize] [--hidpi]`: presents to an SDL3 window.

vkcube runs as well; brew's binary is
`vulkan-tools/*/cube/vkcube.app/Contents/MacOS/vkcube`.

dEQP-VK: `test/build-cts.sh` builds it, and `test/run-cts.sh OUT GROUP...`
runs mustpass groups (`api`, `draw`, `texture`, …) with deqp-runner.
Anything after `--` goes to deqp-runner, such as `--fraction 25`.

The expectations come from a full run: `ci/wvk-fails.txt` and
`ci/wvk-flakes.txt`. OUT's `failures.csv` lists only what changed.
The full run of 2026-09-28 took 1 h 50 min on an Apple M4, with 7 jobs:

- 2,443,884 cases; 313,069 ran, and 2,130,815 were unsupported;
- 98.5% of what ran passed;
- 4,799 failed, 9 crashed and 2 timed out.

The biggest failing families:

| failures | family |
| --- | --- |
| 1,371 | `descriptorset_random` (more than 10 storage buffers per stage) |
| 466 | `image.mutable` |
| 441 | `copy_and_blit` |
| 300 | `memory_model.message_passing` |
| 292 | `pipeline.monolithic`, mostly blending |
| 252 | `spirv_assembly` |

Minecraft: `test/run-minecraft.py INSTANCE [--world NAME] [--capture N]
[--seconds S]` runs a Prism Launcher instance offline, on this build's ICD
through brew's loader, with `--graphicsBackend vulkan`. It uses a scratch game
directory, with copies of the instance's options and the world. 26.3 and
26.4-snapshot-1 render their title screens and worlds. `--gl` runs the
OpenGL backend on wink's GL instead (`wink/README.md`).

## In a browser

`wink/build.sh` builds it into libwink. There is no loader, so an app
gets everything from `wink_vk_get_instance_proc_addr`, and a surface for
a canvas from `wink_vk_create_surface(instance, "#canvas", &surface)`.
The only instance extension it needs is `VK_KHR_surface`. The wink fork of
SDL does both when it is built with wink, so `SDL_Vulkan_*` just work.

Waiting suspends the wasm stack through JSPI. So the thread that calls
Vulkan has to be entered through `WebAssembly.promising`, the way .NET
enters its exports, on the main thread or on a worker that holds the
canvas as an OffscreenCanvas. The swapchain is the canvas's size, and
presenting ends the task: FIFO waits there for the next frame.

`wink/build.sh --test` runs `wvk_smoke` and `wvk_draw` in headless Chrome,
with a generated shim standing in for the loader (`wink/test/vk`). By
hand:

```sh
wink/test/vk/build-vk.sh st build/wink/out/st "$(command -v emcc)" OUT
node wink/test/run-browser.mjs OUT --page='vk.html?test=wvk_draw'
```

## Environment

| variable | effect |
| --- | --- |
| `WINK_VK_DUMP_WGSL=1` | print every shader's WGSL |
| `WINK_VK_CAPTURE=N:file.bmp` | write what the Nth present shows, for machines where nothing can capture the screen |
| `WINK_VK_ABORT_ON_ERROR=1` | abort on a WebGPU validation error, which is a driver bug |
| `WINK_VK_FALLBACK_ADAPTER=1` | use Dawn's software adapter |
| `WINK_VK_API_VERSION` | the Vulkan version reported |
| `WINK_VK_VARIANT_LOG=1` | say when a shader variant is compiled for a draw, and why |
| `GEX_WGPU_SPIRV_DIR=dir` | write every module to `dir/N.spv` before Tint reads it, for one Tint does not come back from |

## Not there yet

- Depth clamp clamps to [0, 1], not to the viewport's depth range: Dawn on
  Metal clamps at the near and far planes. A narrow or inverted range with
  depth clamp is off where geometry is past it.
- WebGPU has 10 storage buffers a stage in browsers, and texel buffers are
  storage buffers, beyond what Vulkan counts as such; so are storage images
  a shader does atomics on.
- Vulkan 1.3 is reported, but for `bufferDeviceAddress`: WebGPU has no
  pointers.
- Descriptor indexing's features, each of them, but not
  `VK_EXT_descriptor_indexing`: that asks for 500000 descriptors a stage,
  where WebGPU has a few dozen bindings.
- 64-bit floats as vertex inputs or varyings: WebGPU has neither.
- Scalar 8-bit division and remainder, and 8-bit specialization
  constants, fail a few CTS cases not yet understood.
- Device-scope memory ordering between workgroups: WGSL's barriers are
  the workgroup's, and none can be in non-uniform control flow, so a
  memory barrier there is dropped.
- An attribute larger than its binding's dynamic stride, which WebGPU
  cannot describe. Strips of 16-bit indices with primitive restart off
  still restart at 0xFFFF, as WebGPU's always do.
- Precise occlusion queries: WebGPU's count is exact on some backends
  only.
- Subgroups are one invocation each: Tint reads no subgroup operations from
  SPIR-V. Every operation but the quad ones is there, trivially.
- Blend states whose colour factors use both CONSTANT_COLOR and
  CONSTANT_ALPHA: WebGPU has one constant.
- Views of multisampled images in formats other than the image's, or its
  sRGB twin: other views are of a texture of the view's format, which
  copies through a buffer keep up to date, and WebGPU copies no
  multisampled texture.
- Gathers from integer cube maps take one texel four times: each face's
  2x2 footprint across its edges is not worked out.
- Lines and points of triangles are not culled: which way a triangle
  faces is the vertex shader's to say, after the lists are made.
- Multisampled array images are a texture for each layer, since WebGPU
  has no multisampled arrays: attachments, clears and resolves are right,
  but a shader sampling one sees its first layer.
- Feedback loops on multisampled depth or stencil that is written.
- Formats WebGPU does not multisample (RG32 and RGBA32 among them) have
  one sample, though Vulkan wants every colour attachment format to have
  framebufferColorSampleCounts.

Some of these are what WebGPU has no way to express. The rest are work still
to do.
