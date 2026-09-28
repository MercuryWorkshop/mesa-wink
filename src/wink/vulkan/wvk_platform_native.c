/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Native Dawn: its instance has a real event loop, which
 * wgpuInstanceWaitAny runs until a future completes.
 */
#include "wvk_platform.h"

#include "util/os_time.h"

WGPUInstance
wvk_platform_create_instance(void)
{
   WGPUInstanceDescriptor desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
   /* WaitAny with a timeout is how requests, fences and vkWaitSemaphores
    * are answered, and it is off unless asked for. */
   desc.capabilities.timedWaitAnyEnable = 1;
   desc.capabilities.timedWaitAnyMaxCount = 64;
   return wgpuCreateInstance(&desc);
}

WGPUCallbackMode
wvk_platform_callback_mode(void)
{
   return WGPUCallbackMode_WaitAnyOnly;
}

WGPUWaitStatus
wvk_platform_wait_any(WGPUInstance instance, WGPUFutureWaitInfo *waits, size_t count, uint64_t timeout_ns)
{
   return wgpuInstanceWaitAny(instance, count, waits, timeout_ns);
}

void
wvk_platform_signal(void)
{
}

void
wvk_platform_sleep(uint64_t ns)
{
   os_time_sleep(MAX2(ns / 1000, 1));
}

void
wvk_platform_surface_present(WGPUSurface surface)
{
   wgpuSurfacePresent(surface);
}

void
wvk_platform_present_yield(bool fifo)
{
   /* Dawn's surface paces FIFO itself, in getCurrentTexture. */
}

bool
wvk_platform_surface_extent(WGPUSurface surface, VkExtent2D *extent)
{
   return false;
}
