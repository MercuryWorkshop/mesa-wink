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

#include <math.h>

/* In library_wink.js. wvk_js_wait is a WebAssembly.Suspending import: the
 * call that leads here has to be under WebAssembly.promising. */
extern void wvk_js_wait(double timeout_ms);
extern void wvk_js_signal(void);
extern void wink_js_yield_frame(int frames);

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
