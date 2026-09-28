/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Where the WebGPU instance, adapter and device come from, and how to wait
 * for WebGPU. Natively, Dawn's callbacks are delivered from
 * wgpuInstanceWaitAny, which blocks. In a browser they come whenever
 * JavaScript resolves them, and waiting is suspending the wasm stack
 * through JSPI until one does. Callers therefore wait in a loop over their
 * own completion flags, and every callback calls wvk_platform_signal. The
 * rest of the driver never knows which platform it is on.
 */
#ifndef WVK_PLATFORM_H
#define WVK_PLATFORM_H

#include "wvk_private.h"

/* NULL if WebGPU is unavailable. */
WGPUInstance wvk_platform_create_instance(void);

/* The mode every callback of the driver is made with. */
WGPUCallbackMode wvk_platform_callback_mode(void);

/* Wait until one of the futures completes, or natively their callbacks
 * run, or in a browser any WebGPU callback runs, or the timeout passes
 * (UINT64_MAX: none). WGPUWaitStatus_Error if waiting is impossible. */
WGPUWaitStatus wvk_platform_wait_any(WGPUInstance instance, WGPUFutureWaitInfo *waits, size_t count,
                                     uint64_t timeout_ns);

/* Let time pass while polling for what another thread does: in a browser
 * that is the event loop's turn too. */
void wvk_platform_sleep(uint64_t ns);

/* A callback ran: whoever waits for one is to look again. */
void wvk_platform_signal(void);

/* The adapter to expose, or NULL. */
WGPUAdapter wvk_platform_request_adapter(WGPUInstance instance);

WGPUDevice wvk_platform_request_device(WGPUInstance instance, WGPUAdapter adapter,
                                       const WGPUDeviceDescriptor *desc);

/* How big a canvas's surface is, in a browser; natively false. */
bool wvk_platform_surface_extent(WGPUSurface surface, VkExtent2D *extent);

/* Let the frame go to the screen: in a browser, the event loop has to run
 * for the canvas to present, and FIFO waits for the next frame there. */
void wvk_platform_present_yield(bool fifo);

#endif
