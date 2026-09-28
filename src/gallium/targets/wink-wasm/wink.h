/* SPDX-License-Identifier: MIT */
/*
 * wink: Mesa's OpenGL and OpenGL ES on the gex_wgpu WebGPU driver, for an
 * application that links Mesa statically into an Emscripten module (a .NET
 * browser-wasm app, typically).
 *
 * This is the whole interface. There is no EGL: a windowing layer (an SDL or
 * GLFW backend) creates a surface for its canvas and a context, makes them
 * current, and looks GL up by name. No gl* symbol is exported -- a static
 * wasm link has one namespace, and the application may well link WebGL's GL
 * too -- so wink_get_proc_address() is the only way to GL.
 *
 * Threads and JSPI. WebGPU objects belong to the JavaScript realm that made
 * them. By default every call must come from the thread that called
 * wink_init() (anything else gets WINK_ERROR_WRONG_THREAD), and calls that
 * may wait for the GPU -- wink_init() itself, and any GL call that reads back,
 * from glReadPixels to a query result -- suspend through JSPI, so they must
 * run beneath an export the embedder wrapped in WebAssembly.promising. A frame
 * reaches the canvas when that thread returns to its event loop.
 *
 * WINK_INIT_GL_ON_MAIN_THREAD instead keeps all of that on the main thread and
 * lets any worker call wink and GL; see its definition.
 *
 * Every attribute struct begins with its own size, and a *_default()
 * initializer fills it; a newer wink reading an older caller's struct uses
 * defaults for the fields it did not know about.
 */
#ifndef WINK_H
#define WINK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WINK_ABI_VERSION 1

#ifdef WINK_BUILDING
#define WINK_EXPORT __attribute__((visibility("default")))
#else
#define WINK_EXPORT
#endif

typedef struct wink_surface wink_surface;
typedef struct wink_context wink_context;

enum wink_status {
   WINK_OK = 0,
   WINK_ERROR_NO_WEBGPU = -1,       /* no navigator.gpu, adapter or device */
   WINK_ERROR_NO_JSPI = -2,         /* WebAssembly.Suspending/promising absent */
   WINK_ERROR_WRONG_THREAD = -3,
   WINK_ERROR_NOT_INITIALIZED = -4,
   WINK_ERROR_BAD_ATTRIBUTE = -5,
   WINK_ERROR_BAD_VERSION = -6,     /* the requested version is not available */
   WINK_ERROR_BAD_PROFILE = -7,
   WINK_ERROR_NO_CANVAS = -8,
   WINK_ERROR_OUT_OF_MEMORY = -9,
   WINK_ERROR_DEVICE_LOST = -10,
   WINK_ERROR_BAD_SURFACE = -11,
   WINK_ERROR_BAD_CONTEXT = -12,
};

enum wink_api {
   WINK_API_OPENGL_COMPAT = 0,
   WINK_API_OPENGL_CORE = 1,
   WINK_API_OPENGL_ES = 2,
};

#define WINK_CONTEXT_DEBUG               (1u << 0)
#define WINK_CONTEXT_FORWARD_COMPATIBLE  (1u << 1)
#define WINK_CONTEXT_ROBUST_ACCESS       (1u << 2)
#define WINK_CONTEXT_LOSE_ON_RESET       (1u << 3)
#define WINK_CONTEXT_NO_ERROR            (1u << 4)
#define WINK_CONTEXT_RELEASE_NONE        (1u << 5)

/* wink_init_attribs.flags */
/* Threaded builds: the caller is a worker that makes GL calls, and the
 * device, the canvas and the driver live on the browser's main thread, where
 * the canvas already is -- no OffscreenCanvas transfer, no JSPI on the worker.
 * GL calls are recorded on the worker and run on main (Mesa's glthread with
 * the main thread as its driver thread); a call that needs an answer waits
 * for it. The main thread must keep returning to its event loop. Ignored when
 * wink_init is called on the main thread itself. */
#define WINK_INIT_GL_ON_MAIN_THREAD      (1u << 0)

typedef struct wink_init_attribs {
   uint32_t size;
   uint32_t flags;               /* WINK_INIT_* */
   /* A WGPUDevice to render with. NULL: use Module.winkDevice or
    * Module.preinitializedWebGPUDevice if the page set one, or else request an
    * adapter and a device with every feature and limit this driver can use. */
   void *webgpu_device;
} wink_init_attribs;

typedef struct wink_surface_attribs {
   uint32_t size;
   /* A CSS selector. NULL or "" uses Module.winkCanvas, then Module.canvas. */
   const char *canvas;
   int32_t width, height;        /* the default framebuffer, in pixels */
   uint8_t alpha_bits;           /* 0 or 8; 0 composites the canvas opaque */
   uint8_t depth_bits;           /* 0, 16, 24 or 32 */
   uint8_t stencil_bits;         /* 0 or 8 */
   uint8_t samples;              /* 0 or 1; multisampled default framebuffers are not yet supported */
   uint8_t srgb;                 /* an sRGB-capable default framebuffer */
   uint8_t premultiplied_alpha;  /* reserved: a browser composites a WebGPU canvas
                                  * premultiplied or not at all, so with alpha_bits
                                  * the content must be premultiplied */
   uint8_t manage_canvas_size;   /* set the canvas's width/height to the surface size */
   uint8_t offscreen;            /* no canvas at all, like a pbuffer */
} wink_surface_attribs;

typedef struct wink_context_attribs {
   uint32_t size;
   int32_t api;                  /* enum wink_api */
   int32_t major, minor;         /* 0.0: the highest version there is */
   uint32_t flags;               /* WINK_CONTEXT_* */
   wink_context *share;          /* share objects with this context, or NULL */
} wink_context_attribs;

WINK_EXPORT void wink_init_attribs_default(wink_init_attribs *attribs);
WINK_EXPORT void wink_surface_attribs_default(wink_surface_attribs *attribs);
WINK_EXPORT void wink_context_attribs_default(wink_context_attribs *attribs);

/* Acquire the device and create the screen. May suspend. Calling it again on
 * the same thread is a no-op returning WINK_OK. */
WINK_EXPORT int wink_init(const wink_init_attribs *attribs);
WINK_EXPORT void wink_shutdown(void);

/* setenv(), for the driver's GEX_WGPU_* and Mesa's MESA_* switches: a
 * browser's environment is empty. Module.winkEnv is read by wink_init() too. */
WINK_EXPORT int wink_setenv(const char *name, const char *value);

/* The highest version of each API, as major*10+minor, without creating a
 * context. 0 means unavailable. */
WINK_EXPORT int wink_query_versions(int *core, int *compat, int *es);

WINK_EXPORT wink_surface *wink_surface_create(const wink_surface_attribs *attribs, int *status);
WINK_EXPORT void wink_surface_destroy(wink_surface *surface);
WINK_EXPORT int wink_surface_resize(wink_surface *surface, int width, int height);
WINK_EXPORT void wink_surface_get_size(const wink_surface *surface, int *width, int *height);

/* A context is not tied to any surface: it may be made current with any of
 * them, or with none. Fails with WINK_ERROR_BAD_VERSION rather than creating
 * a lower version than asked for. */
WINK_EXPORT wink_context *wink_context_create(const wink_context_attribs *attribs, int *status);
WINK_EXPORT void wink_context_destroy(wink_context *context);
WINK_EXPORT void wink_context_get_version(const wink_context *context,
                                          int *api, int *major, int *minor);

/* NULL surfaces make the context current with no default framebuffer. */
WINK_EXPORT int wink_make_current(wink_context *context,
                                  wink_surface *draw, wink_surface *read);
WINK_EXPORT wink_context *wink_get_current_context(void);
WINK_EXPORT wink_surface *wink_get_current_surface(void);

/* Any GL or GL ES function this Mesa knows, extensions included; NULL for
 * anything else. Pointers are valid in every context. */
WINK_EXPORT void *wink_get_proc_address(const char *name);

/* Flush the current context and copy the surface's back buffer onto its
 * canvas, which shows it once the owning thread next yields to its event
 * loop. Then wait as wink_swap_interval says: by default not at all, so
 * this never suspends and pacing is the caller's (requestAnimationFrame).
 * With WINK_INIT_GL_ON_MAIN_THREAD the swap is queued behind the frame and
 * paced on the main thread instead, and this waits for the frame before it:
 * at most two in flight. */
WINK_EXPORT int wink_swap_buffers(wink_surface *surface);

/* How each swap of this surface waits: for `interval` animation frames, or
 * with 0 for one turn of the event loop -- enough for the frame to be shown,
 * and what a loop that never returns to the browser needs. On the owning
 * thread that suspends through JSPI, like a readback. WINK_SWAP_UNPACED
 * never waits; it is the default, except with WINK_INIT_GL_ON_MAIN_THREAD,
 * where the main thread paces with a default of 1 and UNPACED means 0. */
#define WINK_SWAP_UNPACED (-1)
WINK_EXPORT int wink_swap_interval(wink_surface *surface, int interval);

/* Let `frames` animation frames pass on the calling thread, or with 0 one
 * turn of its event loop, suspending through JSPI: the export that led here
 * must have been entered through WebAssembly.promising. requestAnimationFrame
 * never fires in a hidden tab, so no wait lasts longer than a second. Needs
 * no wink_init; it is for callers that pace or yield themselves, such as a
 * windowing layer whose WebGL fallback shares the loop. */
WINK_EXPORT void wink_yield_frames(int frames);

/* Non-zero once the WebGPU device is lost; every later call fails. */
WINK_EXPORT int wink_device_lost(void);
WINK_EXPORT const char *wink_status_string(int status);
/* The WGPUDevice in use, for an embedder that renders beside GL. */
WINK_EXPORT void *wink_webgpu_device(void);

#ifdef __cplusplus
}
#endif

#endif /* WINK_H */
