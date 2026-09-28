/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * The browser: emdawnwebgpu delivers callbacks whenever JavaScript
 * resolves the promises behind them, so they are made spontaneous, and
 * waiting suspends the wasm stack through JSPI (library_wink.js) until one
 * of them signals, or a timer does. The caller then looks at its flags.
 */
#include "wvk_platform.h"
#include "wvk_instance.h"
#include "wvk_wsi.h"

#include <math.h>

/* In library_wink.js. wvk_js_wait is a WebAssembly.Suspending import: the
 * call that leads here has to be under WebAssembly.promising. */
extern void wvk_js_wait(double timeout_ms);
extern void wvk_js_signal(void);
extern void wink_js_yield_frame(int frames);
extern WGPUSurface wink_js_create_surface(const char *selector);
extern int wvk_js_surface_extent(WGPUSurface surface, uint32_t *width, uint32_t *height);

WGPUInstance
wvk_platform_create_instance(void)
{
   return wgpuCreateInstance(NULL);
}

WGPUCallbackMode
wvk_platform_callback_mode(void)
{
   return WGPUCallbackMode_AllowSpontaneous;
}

WGPUWaitStatus
wvk_platform_wait_any(WGPUInstance instance, WGPUFutureWaitInfo *waits, size_t count, uint64_t timeout_ns)
{
   /* The futures cannot be waited for as such; a callback of theirs, or
    * any other, signals. */
   (void)instance;
   (void)waits;
   (void)count;
   if (timeout_ns == 0)
      return WGPUWaitStatus_TimedOut;
   wvk_js_wait(timeout_ns == UINT64_MAX ? -1.0 : ceil(timeout_ns / 1e6));
   return WGPUWaitStatus_Success;
}

void
wvk_platform_signal(void)
{
   wvk_js_signal();
}

void
wvk_platform_sleep(uint64_t ns)
{
   wvk_js_wait(ceil(ns / 1e6));
}

void
wvk_platform_present_yield(bool fifo)
{
   /* The canvas presents when the task ends: FIFO waits for the next
    * frame, and the rest only let the event loop run. */
   wink_js_yield_frame(fifo ? 1 : 0);
}

/* A surface on a canvas, found as GL's surfaces are (library_wink.js):
 * what wink_vk_create_surface gives an application, since Vulkan has no
 * surface extension for a canvas. */
PUBLIC VkResult
wvk_create_canvas_surface(VkInstance _instance, const char *selector,
                          const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface)
{
   VK_FROM_HANDLE(wvk_instance, instance, _instance);
   if (!wvk_instance_wgpu(instance))
      return vk_error(instance, VK_ERROR_INITIALIZATION_FAILED);
   WGPUSurface wgpu = wink_js_create_surface(selector);
   if (!wgpu)
      return vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED, "no canvas for %s", selector ? selector : "(none)");
   VkResult result = wvk_surface_create(instance, VK_ICD_WSI_PLATFORM_HEADLESS, wgpu, pAllocator, pSurface);
   if (result == VK_SUCCESS)
      wvk_surface_from_handle(*pSurface)->canvas = true;
   return result;
}

bool
wvk_platform_surface_extent(WGPUSurface surface, VkExtent2D *extent)
{
   return wvk_js_surface_extent(surface, &extent->width, &extent->height) != 0;
}
