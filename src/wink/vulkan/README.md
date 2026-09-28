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
  acquiring a swapchain image never blocks.
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
  instance attribute every instance. It is the one piece of pipeline state
  compiled into shaders.
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
  draw, with the pass begun again after, and kept until a barrier. Depth
  and stencil in read-only layouts are read-only to WebGPU, which lets
  shaders read them as they are.
- **16-bit packed formats** (565, 4444, 5551, 1555), which WebGPU does not
  have, are RGBA8 textures. Copies to and from buffers are a shader's,
  which packs and unpacks the texels; a format without alpha keeps its
  texture's at one. Blending is at 8 bits, not the format's.
- **Arrays of descriptors** are bindings of their own. A dynamic or
  non-uniform index chooses among them in the shader: reads from each,
  selected; writes only where the index is.
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

## Not there yet

- WebGPU has 10 storage buffers a stage in browsers, and texel buffers are
  storage buffers, beyond what Vulkan counts as such; so are storage images
  a shader does atomics on.
- Vulkan 1.3 is reported, but for `bufferDeviceAddress`: WebGPU has no
  pointers.
- Descriptor indexing's features, each of them, but not
  `VK_EXT_descriptor_indexing`: that asks for 500000 descriptors a stage,
  where WebGPU has a few dozen bindings.
- Device-scope memory ordering between workgroups: WGSL's barriers are
  the workgroup's, and none can be in non-uniform control flow, so a
  memory barrier there is dropped.
- An attribute larger than its binding's dynamic stride, which WebGPU
  cannot describe.
- Precise occlusion queries: WebGPU's count is exact on some backends
  only. Alpha to coverage with a sample mask the shader writes.
- Subgroups are one invocation each: Tint reads no subgroup operations from
  SPIR-V. Every operation but the quad ones is there, trivially.
- Compressed images whose size is not a whole number of blocks: WebGPU's
  texture is the whole blocks, and sampling it covers the padding.
- Blend states whose colour factors use both CONSTANT_COLOR and
  CONSTANT_ALPHA: WebGPU has one constant.
- Multisampled array images are a texture for each layer, since WebGPU
  has no multisampled arrays: attachments, clears and resolves are right,
  but a shader sampling one sees its first layer.
- Feedback loops on multisampled images, and on depth or stencil that is
  written: WebGPU copies neither.
- Formats WebGPU does not multisample (RG32 and RGBA32 among them) have
  one sample, though Vulkan wants every colour attachment format to have
  framebufferColorSampleCounts.

Some of these are what WebGPU has no way to express. The rest are work still
to do.
