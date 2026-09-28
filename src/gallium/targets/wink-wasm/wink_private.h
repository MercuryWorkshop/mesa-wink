/* SPDX-License-Identifier: MIT */
#ifndef WINK_PRIVATE_H
#define WINK_PRIVATE_H

#include <stdbool.h>
#include <stdint.h>

struct st_config_options;

/* wink_driconf.c: the defaults a DRI build would get from driconf, so the
 * browser and a native CTS run see the same st/mesa options. */
void wink_fill_st_options(struct st_config_options *options);

/* library_wink.js. The two that may suspend are wrapped in
 * WebAssembly.Suspending there. */
int wink_js_has_jspi(void);
void *wink_js_acquire_device(void);                 /* WGPUDevice; suspends */
void *wink_js_create_surface(const char *selector); /* WGPUSurface, or NULL */
int wink_js_get_env(char *buf, int len);
int wink_js_device_lost(void);

#endif
