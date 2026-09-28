# wvk: Vulkan on WebGPU

The wink Vulkan driver: Vulkan 1.2 with `VK_KHR_portability_subset`, on a
WebGPU device. It is Mesa's Vulkan runtime on top, and underneath it is the
shader compiler the GL driver (`gex_wgpu`) uses, in `src/wink/compiler`.
Natively it is an ICD over Dawn, which is where Vulkan's tools, vkcube and
dEQP-VK run. The browser build, a static library for libwink, is still to
come.

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
- **Blits and `vkCmdClearAttachments`** come from the runtime's vk_meta.
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

## Environment

| variable | effect |
| --- | --- |
| `WINK_VK_DUMP_WGSL=1` | print every shader's WGSL |
| `WINK_VK_CAPTURE=N:file.bmp` | write what the Nth present shows, for machines where nothing can capture the screen |
| `WINK_VK_ABORT_ON_ERROR=1` | abort on a WebGPU validation error, which is a driver bug |
| `WINK_VK_FALLBACK_ADAPTER=1` | use Dawn's software adapter |
| `WINK_VK_API_VERSION` | the Vulkan version reported |

## Not there yet

- Texel buffers, in shaders.
- Dynamically indexed arrays of descriptors: the first element is used.
- Layered rendering: a pass is one layer. This affects blits of several
  layers and 3D blits, and multiview.
- Depth and stencil resolves.
- Copies into a depth32 or depth24 aspect.
- `shaderDrawParameters`.
- Precise occlusion queries, and timestamps.
- Subgroups.
- Viewports outside the attachments, which are clamped.
- Sampler LOD bias, border colours, and component swizzles.

Some of these are what WebGPU has no way to express. The rest are work still
to do.
