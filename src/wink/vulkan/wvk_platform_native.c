/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Native Dawn: its instance has a real event loop, so the asynchronous
 * requests are made synchronous with wgpuInstanceWaitAny.
 */
#include "wvk_platform.h"

#include <stdio.h>
#include <stdlib.h>

WGPUInstance
wvk_platform_create_instance(void)
{
   WGPUInstanceDescriptor desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
   /* WaitAny with a timeout is how the requests below, fences and
    * vkWaitSemaphores are answered, and it is off unless asked for. */
   desc.capabilities.timedWaitAnyEnable = 1;
   desc.capabilities.timedWaitAnyMaxCount = 64;
   return wgpuCreateInstance(&desc);
}

struct request {
   WGPUAdapter adapter;
   WGPUDevice device;
};

static void
on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
           WGPUStringView message, void *userdata1, void *userdata2)
{
   struct request *r = userdata1;
   (void)userdata2;
   r->adapter = status == WGPURequestAdapterStatus_Success ? adapter : NULL;
   if (!r->adapter)
      mesa_logw("wvk: no WebGPU adapter: %.*s", wvk_sv_len(message), message.data);
}

WGPUAdapter
wvk_platform_request_adapter(WGPUInstance instance)
{
   struct request r = { 0 };
   WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
   options.powerPreference = WGPUPowerPreference_HighPerformance;
   /* Dawn's software adapter, for telling a bad command stream from a
    * hardware problem -- where Dawn was built with SwiftShader. */
   if (getenv("WINK_VK_FALLBACK_ADAPTER"))
      options.forceFallbackAdapter = true;
   WGPURequestAdapterCallbackInfo cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
   cb.mode = WGPUCallbackMode_WaitAnyOnly;
   cb.callback = on_adapter;
   cb.userdata1 = &r;
   if (!wvk_platform_wait(instance, wgpuInstanceRequestAdapter(instance, &options, cb), UINT64_MAX))
      return NULL;
   return r.adapter;
}

static void
on_device(WGPURequestDeviceStatus status, WGPUDevice device,
          WGPUStringView message, void *userdata1, void *userdata2)
{
   struct request *r = userdata1;
   (void)userdata2;
   r->device = status == WGPURequestDeviceStatus_Success ? device : NULL;
   if (!r->device)
      mesa_logw("wvk: no WebGPU device: %.*s", wvk_sv_len(message), message.data);
}

WGPUDevice
wvk_platform_request_device(WGPUInstance instance, WGPUAdapter adapter,
                            const WGPUDeviceDescriptor *desc)
{
   struct request r = { 0 };
   WGPURequestDeviceCallbackInfo cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
   cb.mode = WGPUCallbackMode_WaitAnyOnly;
   cb.callback = on_device;
   cb.userdata1 = &r;
   if (!wvk_platform_wait(instance, wgpuAdapterRequestDevice(adapter, desc, cb), UINT64_MAX))
      return NULL;
   return r.device;
}

bool
wvk_platform_wait(WGPUInstance instance, WGPUFuture future, uint64_t timeout_ns)
{
   WGPUFutureWaitInfo wait = { .future = future, .completed = false };
   const WGPUWaitStatus status = wgpuInstanceWaitAny(instance, 1, &wait, timeout_ns);
   return status == WGPUWaitStatus_Success && wait.completed;
}

void
wvk_platform_process_events(WGPUInstance instance)
{
   wgpuInstanceProcessEvents(instance);
}
