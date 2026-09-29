/* SPDX-License-Identifier: MIT */
/*
 * wink on a desktop, against native Dawn: what library_wink.js is in a
 * browser. The device is gex_wgpu_native's, the one per process every
 * native frontend of the driver shares, and a surface is made of the layer
 * the windowing system has for the window (a CAMetalLayer on macOS).
 * Nothing suspends: native Dawn waits for itself.
 */
#include "wink_private.h"

#include <webgpu/webgpu.h>
#include <stddef.h>

#include "gex_wgpu_public.h"

int
wink_js_has_jspi(void)
{
   return 1;
}

void *
wink_js_acquire_device(void)
{
   WGPUDevice device = gex_wgpu_native_device();
   /* wink_shutdown releases what it was given; the process keeps this one. */
   if (device)
      wgpuDeviceAddRef(device);
   return device;
}

void *
wink_js_create_surface(const char *selector)
{
   (void)selector;
   return NULL;
}

/* The environment is the process's own. */
int
wink_js_get_env(char *buf, int len)
{
   if (buf && len > 0)
      buf[0] = '\0';
   return 0;
}

/* Only a remote (main-browser-thread) wink kicks; a native one is never. */
void
wink_js_kick(struct util_queue *queue)
{
   (void)queue;
}

/* Presentation paces a native swap (FIFO), so there is nothing to wait for. */
void
wink_js_yield_frame(int frames)
{
   (void)frames;
}

void *
wink_native_create_surface(void *layer)
{
   WGPUInstance instance = gex_wgpu_native_instance();
   if (!layer || !instance)
      return NULL;
#ifdef __APPLE__
   WGPUSurfaceSourceMetalLayer source = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
   source.layer = layer;
   WGPUSurfaceDescriptor desc = WGPU_SURFACE_DESCRIPTOR_INIT;
   desc.nextInChain = &source.chain;
   return wgpuInstanceCreateSurface(instance, &desc);
#else
   return NULL;
#endif
}
