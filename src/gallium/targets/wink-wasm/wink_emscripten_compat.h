/* SPDX-License-Identifier: MIT */
/*
 * The WebGL-context-shaped seam ikvm-wasm-build's emscripten-glfw patch
 * dispatches to (gl_backend.h there: <prefix>_webgl_create_context and
 * friends), so wink can be one of its runtime-selected backends before the
 * move to SDL3. New code should use wink.h directly.
 *
 * A "context handle" here is a wink surface and context pair on one canvas.
 * wink_get_proc_address comes from wink.h.
 */
#ifndef WINK_EMSCRIPTEN_COMPAT_H
#define WINK_EMSCRIPTEN_COMPAT_H

#include <emscripten/html5.h>
#include <emscripten/html5_webgl.h>

#include "wink.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The version and profile the next wink_webgl_create_context asks for --
 * GLFW's context hints, which EmscriptenWebGLContextAttributes cannot carry.
 * Defaults to the highest compatibility-profile version. */
WINK_EXPORT void wink_set_next_context(int api, int major, int minor, unsigned flags);

/* Whether the next wink_webgl_create_context's canvas composites its alpha
 * channel (GLFW_TRANSPARENT_FRAMEBUFFER). Defaults to 0: a window is opaque
 * whatever its alpha bits, as on a desktop, so an application that clears to
 * alpha 0 -- Minecraft does -- does not show the page through. */
WINK_EXPORT void wink_set_next_surface_transparent(int transparent);

/* wink_init_attribs.flags for the wink_init the next wink_webgl_create_context
 * makes -- WINK_INIT_GL_ON_MAIN_THREAD, for a windowing layer on a worker whose
 * canvas stayed on the main thread. Only the first context's call initializes
 * wink, so this matters before that one only. */
WINK_EXPORT void wink_set_next_init_flags(unsigned flags);

/* wink_swap_interval for the current context's surface. */
WINK_EXPORT void wink_set_swap_interval(int interval);

WINK_EXPORT EMSCRIPTEN_WEBGL_CONTEXT_HANDLE
wink_webgl_create_context(const char *target, const EmscriptenWebGLContextAttributes *attributes);
WINK_EXPORT EMSCRIPTEN_RESULT wink_webgl_make_context_current(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context);
WINK_EXPORT EMSCRIPTEN_RESULT wink_webgl_destroy_context(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE context);
/* Swap the current context's surface. */
WINK_EXPORT void wink_present(void);
/* Resize the current context's surface. */
WINK_EXPORT void wink_resize(int width, int height);
/* Nothing to do: kept for the seam's shape. */
WINK_EXPORT void wink_proc_init(void);

#ifdef __cplusplus
}
#endif

#endif
