/* SPDX-License-Identifier: MIT */
#ifndef GEX_WGPU_PUBLIC_H
#define GEX_WGPU_PUBLIC_H

#include <webgpu/webgpu.h>

struct pipe_screen;

#ifdef __cplusplus
extern "C" {
#endif

/* The browser owns device.  The returned screen retains only its queue and
 * must be destroyed before the JavaScript WebGPUDevice is released. */
struct pipe_screen *gex_wgpu_screen_create(WGPUDevice device);

/* The WebGPU texture behind a resource, for the winsys to present.  Returns
 * NULL for a resource this driver did not create a texture for (a buffer, or
 * one it declined).  Borrowed, not retained: it lives as long as the
 * resource. */
struct pipe_resource;
WGPUTexture gex_wgpu_resource_texture(struct pipe_resource *resource);

/* The device the screen was created with, so the winsys can encode its own
 * commands on the same one rather than being handed a second. */
WGPUDevice gex_wgpu_screen_device(struct pipe_screen *screen);

/* A screen for an ordinary desktop application: creates a device from native
 * Dawn and presents through Mesa's software winsys. Declared here rather than
 * in a winsys header because the driver is what implements it. */
struct sw_winsys;
struct pipe_screen *gex_wgpu_sw_screen_create(struct sw_winsys *winsys);

/* The process's native Dawn device, made on first use, and the instance it
 * came from -- for a frontend that presents to a surface of its own. Not
 * retained: they last as long as the process. */
WGPUDevice gex_wgpu_native_device(void);
WGPUInstance gex_wgpu_native_instance(void);

#ifdef __cplusplus
}
#endif

#endif
