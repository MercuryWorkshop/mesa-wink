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
#ifdef WGPU_INSTANCE_LIMITS_INIT
   /* An instance feature and limit since mid-2025's webgpu.h. So is a device
    * per VkDevice from the one adapter: Dawn otherwise marks an adapter
    * consumed by its first device, as WebGPU says. */
   static const WGPUInstanceFeatureName features[] = {
      WGPUInstanceFeatureName_TimedWaitAny,
      WGPUInstanceFeatureName_MultipleDevicesPerAdapter,
   };
   WGPUInstanceLimits limits = WGPU_INSTANCE_LIMITS_INIT;
   limits.timedWaitAnyMaxCount = 64;
   desc.requiredFeatureCount = ARRAY_SIZE(features);
   desc.requiredFeatures = features;
   desc.requiredLimits = &limits;
#else
   desc.capabilities.timedWaitAnyEnable = 1;
   desc.capabilities.timedWaitAnyMaxCount = 64;
#endif
   /* Dawn's Metal backend has texture-formats-tier1 and -tier2 as
    * experimental, which an adapter only offers with this. They are
    * WebGPU's, and browsers have them; it enables nothing the driver does
    * not ask for. */
   static const char *const toggles[] = { "allow_unsafe_apis" };
   WGPUDawnTogglesDescriptor toggles_desc = WGPU_DAWN_TOGGLES_DESCRIPTOR_INIT;
   toggles_desc.enabledToggleCount = ARRAY_SIZE(toggles);
   toggles_desc.enabledToggles = toggles;
   desc.nextInChain = &toggles_desc.chain;
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
