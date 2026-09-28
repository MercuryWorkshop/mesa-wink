/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * The adapter and the device, requested the same way everywhere: the
 * callback sets a flag, and the platform waits until it is set.
 */
#include "wvk_platform.h"

#include <stdlib.h>

struct request {
   bool done;
   WGPUAdapter adapter;
   WGPUDevice device;
};

static void
on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message,
           void *userdata1, void *userdata2)
{
   struct request *r = userdata1;
   (void)userdata2;
   r->adapter = status == WGPURequestAdapterStatus_Success ? adapter : NULL;
   if (!r->adapter)
      mesa_logw("wvk: no WebGPU adapter: %.*s", wvk_sv_len(message), message.data);
   r->done = true;
   wvk_platform_signal();
}

static void
on_device(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message,
          void *userdata1, void *userdata2)
{
   struct request *r = userdata1;
   (void)userdata2;
   r->device = status == WGPURequestDeviceStatus_Success ? device : NULL;
   if (!r->device)
      mesa_logw("wvk: no WebGPU device: %.*s", wvk_sv_len(message), message.data);
   r->done = true;
   wvk_platform_signal();
}

static void
wait_for(WGPUInstance instance, WGPUFuture future, const struct request *r)
{
   WGPUFutureWaitInfo wait = { .future = future };
   while (!r->done) {
      if (wvk_platform_wait_any(instance, &wait, 1, UINT64_MAX) == WGPUWaitStatus_Error)
         break;
   }
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
   cb.mode = wvk_platform_callback_mode();
   cb.callback = on_adapter;
   cb.userdata1 = &r;
   wait_for(instance, wgpuInstanceRequestAdapter(instance, &options, cb), &r);
   return r.adapter;
}

WGPUDevice
wvk_platform_request_device(WGPUInstance instance, WGPUAdapter adapter, const WGPUDeviceDescriptor *desc)
{
   struct request r = { 0 };
   WGPURequestDeviceCallbackInfo cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
   cb.mode = wvk_platform_callback_mode();
   cb.callback = on_device;
   cb.userdata1 = &r;
   wait_for(instance, wgpuAdapterRequestDevice(adapter, desc, cb), &r);
   return r.device;
}
