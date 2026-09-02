# Running gex_wgpu on a desktop

The driver takes a `WGPUDevice` and speaks `webgpu.h`; nothing in it needs a
browser. These three scripts give it a device from native Dawn and somewhere
to put a finished frame, so an ordinary Linux GL program runs on it with a
debugger attached and `softpipe` one environment variable away as a reference.

**This directory is self-contained.** Only this Mesa checkout is needed — the
gex repository is not.

## Prerequisites

`clang` (Dawn does not build with GCC at this revision), `cmake`, `ninja`,
`meson`, `python3`, `git`, and the usual X11 and libdrm development packages
Mesa wants. A Vulkan driver for your GPU, since that is what Dawn runs on.

## Three steps

```sh
cd src/gallium/drivers/gex_wgpu/native
./build-dawn.sh       # clone and build Dawn at the pinned revision (slow, once)
./package-dawn.sh     # lay it out the way this driver's meson expects
./build-mesa.sh       # configure and build Mesa with gex_wgpu and softpipe
```

Then:

```sh
./run.sh glxgears
./run.sh glxinfo -B
GALLIUM_DRIVER=softpipe ./run.sh glxgears     # the reference, same Mesa
```

Where things go, all overridable:

| variable | default | what |
| --- | --- | --- |
| `GEX_DAWN_ROOT` | `~/.cache/gex-wgpu-dawn` | Dawn clone and build |
| `GEX_DAWN_PKG` | `$GEX_DAWN_ROOT/pkg` | the packaged Dawn |
| `GEX_MESA_OUT` | `/tmp/gex-mesa-native` | the Mesa build and its install prefix |

## What the driver reads

| variable | what it does |
| --- | --- |
| `GEX_WGPU_TIMING=1` | frames and draws per second, and the share of wall clock in draws and in present |
| `GEX_WGPU_TIMING=2` | also the state-setting entry points, which costs about as much as it measures |
| `GEX_WGPU_NO_DRAW=1` | drop every draw. Useless output; the point is the frame rate left, which is the ceiling the application and st/mesa impose |

## Things worth knowing

`LIBGL_ALWAYS_SOFTWARE=1` is not asking for software rendering. It selects
Mesa's software winsys, which is how this driver is reached at all: it
presents through `sw_winsys` rather than through DRI. `run.sh` sets it.

Presentation reads the frame back to the CPU once a frame and hands it to
X11. That is the cost of using the software winsys, and it is per frame
rather than per draw, but it is why a trivial program can spend a third of
its time in present.

The driver reports **GL 3.3**. GL 4.0 is not reachable on WebGPU: it requires
`ARB_gpu_shader_fp64` and `ARB_tessellation_shader`, and WebGPU has neither.
Applications that insist on 4.x -- Blender wants 4.3 -- will refuse to start.
