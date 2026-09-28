/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Where the WebGPU instance, adapter and device come from, and how to wait
 * for WebGPU: native Dawn blocks in wgpuInstanceWaitAny; the browser
 * suspends through JSPI. The rest of the driver never knows which.
 */
#ifndef WVK_PLATFORM_H
#define WVK_PLATFORM_H

#include "wvk_private.h"

/* NULL if WebGPU is unavailable. */
WGPUInstance wvk_platform_create_instance(void);

/* The adapter to expose, or NULL. */
WGPUAdapter wvk_platform_request_adapter(WGPUInstance instance);

WGPUDevice wvk_platform_request_device(WGPUInstance instance, WGPUAdapter adapter,
                                       const WGPUDeviceDescriptor *desc);

/* Wait for a future made with WGPUCallbackMode_WaitAnyOnly: true once it
 * completed, false on timeout or failure. timeout_ns UINT64_MAX waits for
 * good. */
bool wvk_platform_wait(WGPUInstance instance, WGPUFuture future, uint64_t timeout_ns);

/* Let WebGPU deliver callbacks that are due, without waiting. */
void wvk_platform_process_events(WGPUInstance instance);

#endif
