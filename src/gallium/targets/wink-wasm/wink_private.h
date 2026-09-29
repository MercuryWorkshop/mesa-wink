/* SPDX-License-Identifier: MIT */
#ifndef WINK_PRIVATE_H
#define WINK_PRIVATE_H

#include <stdbool.h>
#include <stdint.h>

struct st_config_options;

/* wink_driconf.c: the defaults a DRI build would get from driconf, so the
 * browser and a native CTS run see the same st/mesa options. */
void wink_fill_st_options(struct st_config_options *options);

/* library_wink.js. Those that may suspend are wrapped in
 * WebAssembly.Suspending there. */
struct util_queue;
int wink_js_has_jspi(void);
void *wink_js_acquire_device(void);                 /* WGPUDevice; suspends */
void *wink_js_create_surface(const char *selector); /* WGPUSurface, or NULL */
int wink_js_get_env(char *buf, int len);
/* Have the main thread run `queue` (util_queue_run_external), from any thread. */
void wink_js_kick(struct util_queue *queue);
/* Let `frames` animation frames pass, or with 0 just give the event loop a
 * turn -- which is when a canvas is presented. Suspends: on the main thread's
 * runner, or on a thread entered through WebAssembly.promising. */
void wink_js_yield_frame(int frames);

/* wink_native.c: a surface of a native layer (a CAMetalLayer), or NULL. */
void *wink_native_create_surface(void *layer);                  /* WGPUSurface */

/* wink.c, called by library_wink.js: the main thread's runner, and the
 * device-lost notice. */
void wink_main_run_queue(struct util_queue *queue);
void wink_set_device_lost(void);

#endif
