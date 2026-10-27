/* SPDX-License-Identifier: MIT */
/*
 * wink: st/mesa on gex_wgpu, driven directly -- no EGL, no DRI, no winsys.
 *
 * The shape is gex's wire decoder's (web/mesa/gexgl-mesa.cpp in the
 * gex repository), which has run this frontend on this driver in a browser
 * for a long time: a pipe_frontend_drawable whose attachments are plain
 * textures this file owns, st_api_create_context for the context, and a
 * texture-to-texture copy onto the canvas for presentation. What changes is
 * everything the wire decided for it: sizes, formats, versions and canvases
 * come from the caller, one surface and context each at a time becomes any
 * number, and a context no longer carries a surface's visual (see
 * wink_context_create).
 *
 * Two ways to run it:
 *
 *  - Here: everything on the thread that called wink_init. On a worker, the
 *    canvas must have been transferred to it (an OffscreenCanvas), and the
 *    driver's readbacks suspend that worker's stack through JSPI.
 *
 *  - On the main thread (WINK_INIT_GL_ON_MAIN_THREAD, threaded builds): the
 *    application calls GL from a worker, but the device, the canvas and the
 *    whole driver live on the browser's main thread, where the canvas already
 *    is. Contexts run Mesa's glthread with the main thread as its driver
 *    thread (_mesa_glthread_init_remote): GL calls are marshalled on the
 *    worker and executed on main, and a call that needs an answer waits for
 *    it. Everything else wink does -- surfaces, contexts, make-current -- is
 *    a call run there too (wink_rpc). The worker never touches WebGPU and
 *    never needs JSPI; the main thread's runner in library_wink.js is entered
 *    through WebAssembly.promising by wink itself, on a stack of its own.
 */
#define WINK_BUILDING 1
#include "wink.h"
#include "wink_private.h"

#include <webgpu/webgpu.h>

#include "frontend/api.h"
#include "mesa/glapi/glapi/glapi.h"
#include "mesa/main/glthread.h"
#include "mesa/main/mtypes.h"
#include "mesa/state_tracker/st_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_atomic.h"
#include "util/u_inlines.h"
#include "util/u_queue.h"
#include "util/os_time.h"
#include "gex_wgpu_public.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/threading.h>
#else
#define EMSCRIPTEN_KEEPALIVE
#endif
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct wink_surface {
   /* First: st/mesa hands the drawable back to the callbacks below. */
   struct pipe_frontend_drawable base;
   struct st_visual visual;
   struct pipe_resource *attachments[ST_ATTACHMENT_COUNT];
   unsigned width, height;
   bool offscreen;
   bool manage_canvas_size;
   bool opaque;
   WGPUSurface canvas;
   bool configured;
   /* On the main thread: how many frames a swap waits for, and the two swaps
    * that may be in flight. */
   int swap_interval;
   struct util_queue_fence swap_fence[2];
   unsigned swap_index;
};

struct wink_context {
   struct st_context *st;
   enum wink_api api;
   int major, minor;
};

static struct {
   bool initialized;
   /* WINK_INIT_GL_ON_MAIN_THREAD. */
   bool remote;
   /* Here: the one thread that may call wink. Remote: the main thread. */
   pthread_t owner;
   WGPUDevice device;
   struct pipe_frontend_screen fscreen;
   struct st_config_options options;
   uint32_t next_drawable_id;
   int lost;
   /* Remote: wink's own calls, run on the main thread. */
   struct util_queue control;
} wink;

/* What is current is per thread, as GL's is. */
static _Thread_local wink_context *tls_current;
static _Thread_local wink_surface *tls_draw, *tls_read;

/* ---------------------------------------------------------------------- */

static bool
calling_thread_ok(void)
{
#ifdef __EMSCRIPTEN_PTHREADS__
   /* Remote: any application thread, but not main -- it would wait for
    * itself. */
   if (wink.remote)
      return !emscripten_is_main_browser_thread();
   return pthread_equal(pthread_self(), wink.owner);
#else
   return true;
#endif
}

/* The guard every entry point after wink_init starts with. */
static int
check_ready(void)
{
   if (!wink.initialized)
      return WINK_ERROR_NOT_INITIALIZED;
   if (!calling_thread_ok())
      return WINK_ERROR_WRONG_THREAD;
   if (p_atomic_read(&wink.lost))
      return WINK_ERROR_DEVICE_LOST;
   return WINK_OK;
}

static void
set_status(int *status, int value)
{
   if (status)
      *status = value;
}

/* Copy a caller's struct over the defaults, as far as the caller's size says
 * it reaches: an older caller's shorter struct leaves the newer fields at
 * their defaults. */
static bool
take_attribs(void *dst, size_t dst_size, const void *src)
{
   if (!src)
      return true;
   uint32_t size;
   memcpy(&size, src, sizeof(size));
   if (size < sizeof(uint32_t))
      return false;
   memcpy((char *)dst + sizeof(uint32_t), (const char *)src + sizeof(uint32_t),
          (size < dst_size ? size : dst_size) - sizeof(uint32_t));
   return true;
}

/* ---------------------------------------------------------------------- */
/* The main thread's side of remote mode */

/* A job was queued for the main thread: have library_wink.js run it. */
static void
wink_kick(struct util_queue *queue)
{
   wink_js_kick(queue);
}

/* library_wink.js's runner, on the main thread, through
 * WebAssembly.promising: the jobs may suspend (every readback does). */
EMSCRIPTEN_KEEPALIVE void
wink_main_run_queue(struct util_queue *queue)
{
   util_queue_run_external(queue);
}

/* library_wink.js, when the device is lost: a flag any thread can read,
 * where a JS one would be visible only to the thread that set it. */
EMSCRIPTEN_KEEPALIVE void
wink_set_device_lost(void)
{
   p_atomic_set(&wink.lost, 1);
}

struct wink_rpc {
   void (*fn)(void *arg);
   void *arg;
};

static void
wink_rpc_execute(void *job, void *gdata, int thread_index)
{
   struct wink_rpc *rpc = job;
   rpc->fn(rpc->arg);
}

/* Run fn(arg) on the main thread and wait for it. */
static void
wink_rpc(void (*fn)(void *arg), void *arg)
{
#ifdef __EMSCRIPTEN__
   const bool here = !wink.remote || emscripten_is_main_browser_thread();
#else
   const bool here = true;
#endif
   if (here) {
      fn(arg);
      return;
   }
   struct wink_rpc rpc = { fn, arg };
   struct util_queue_fence fence;
   util_queue_fence_init(&fence);
   util_queue_add_job(&wink.control, &rpc, &fence, wink_rpc_execute, NULL, 0);
   util_queue_fence_wait(&fence);
   util_queue_fence_destroy(&fence);
}

/* ---------------------------------------------------------------------- */

void
wink_init_attribs_default(wink_init_attribs *a)
{
   memset(a, 0, sizeof(*a));
   a->size = sizeof(*a);
}

void
wink_surface_attribs_default(wink_surface_attribs *a)
{
   memset(a, 0, sizeof(*a));
   a->size = sizeof(*a);
   a->width = 1;
   a->height = 1;
   a->alpha_bits = 8;
   a->depth_bits = 24;
   a->stencil_bits = 8;
}

void
wink_context_attribs_default(wink_context_attribs *a)
{
   memset(a, 0, sizeof(*a));
   a->size = sizeof(*a);
   a->api = WINK_API_OPENGL_COMPAT;
}

const char *
wink_status_string(int status)
{
   switch (status) {
   case WINK_OK: return "ok";
   case WINK_ERROR_NO_WEBGPU: return "no WebGPU adapter or device";
   case WINK_ERROR_NO_JSPI: return "JSPI (WebAssembly.Suspending) is unavailable";
   case WINK_ERROR_WRONG_THREAD: return "called from the wrong thread";
   case WINK_ERROR_NOT_INITIALIZED: return "wink_init has not succeeded";
   case WINK_ERROR_BAD_ATTRIBUTE: return "invalid attribute";
   case WINK_ERROR_BAD_VERSION: return "requested version unavailable";
   case WINK_ERROR_BAD_PROFILE: return "invalid API or profile";
   case WINK_ERROR_NO_CANVAS: return "no canvas, or it already has another context";
   case WINK_ERROR_OUT_OF_MEMORY: return "out of memory";
   case WINK_ERROR_DEVICE_LOST: return "WebGPU device lost";
   case WINK_ERROR_BAD_SURFACE: return "invalid surface";
   case WINK_ERROR_BAD_CONTEXT: return "invalid context";
   default: return "unknown wink status";
   }
}

/* ---------------------------------------------------------------------- */
/* Initialization */

static int
get_frontend_param(struct pipe_frontend_screen *fscreen,
                   enum st_manager_param param)
{
   (void)fscreen;
   (void)param;
   return 0;
}

/* glthread tells the frontend about its driver thread; DRI uses it to show the
 * queue on the HUD, and there is no HUD here. */
static void
set_background_context(struct st_context *st,
                       struct util_queue_monitoring *queue_info)
{
   (void)st;
   (void)queue_info;
}

int
wink_setenv(const char *name, const char *value)
{
   if (!name || !*name || strchr(name, '='))
      return WINK_ERROR_BAD_ATTRIBUTE;
   int r = value ? setenv(name, value, 1) : unsetenv(name);
   return r == 0 ? WINK_OK : WINK_ERROR_OUT_OF_MEMORY;
}

/* Module.winkEnv, as "NAME=value" lines: the driver and Mesa read their
 * switches with getenv, and a browser's environment is otherwise empty. */
static void
apply_js_env(void)
{
   int len = wink_js_get_env(NULL, 0);
   if (len <= 1)
      return;
   char *buf = malloc(len);
   if (!buf)
      return;
   wink_js_get_env(buf, len);
   for (char *line = buf, *next; line && *line; line = next) {
      next = strchr(line, '\n');
      if (next)
         *next++ = '\0';
      char *eq = strchr(line, '=');
      if (eq && eq != line) {
         *eq = '\0';
         setenv(line, eq + 1, 1);
      }
   }
   free(buf);
}

static int
init_here(const wink_init_attribs *a)
{
   /* Without JSPI the driver's waits would spin on a promise that can never
    * settle, so refuse here instead of hanging on the first readback. */
   if (!wink_js_has_jspi())
      return WINK_ERROR_NO_JSPI;

   apply_js_env();

   WGPUDevice device = a->webgpu_device;
   if (device)
      wgpuDeviceAddRef(device);
   else
      device = wink_js_acquire_device();
   if (!device)
      return WINK_ERROR_NO_WEBGPU;

   memset(&wink.options, 0, sizeof(wink.options));
   wink_fill_st_options(&wink.options);

   /* The screen keeps the device's queue but not the device itself, so the
    * reference taken above is what keeps it alive until wink_shutdown. */
   wink.fscreen.screen = gex_wgpu_screen_create(device);
   if (!wink.fscreen.screen) {
      wgpuDeviceRelease(device);
      return WINK_ERROR_NO_WEBGPU;
   }
   wink.fscreen.get_param = get_frontend_param;
   wink.fscreen.set_background_context = set_background_context;
   wink.device = device;
   wink.owner = pthread_self();
   wink.lost = 0;
   return WINK_OK;
}

struct init_rpc {
   const wink_init_attribs *attribs;
   int status;
};

static void
init_rpc(void *p)
{
   struct init_rpc *rpc = p;
   rpc->status = init_here(rpc->attribs);
}

int
wink_init(const wink_init_attribs *attribs)
{
   if (wink.initialized)
      return calling_thread_ok() ? WINK_OK : WINK_ERROR_WRONG_THREAD;

   wink_init_attribs a;
   wink_init_attribs_default(&a);
   if (!take_attribs(&a, sizeof(a), attribs))
      return WINK_ERROR_BAD_ATTRIBUTE;

   bool remote = a.flags & WINK_INIT_GL_ON_MAIN_THREAD;
#ifdef __EMSCRIPTEN_PTHREADS__
   /* Already on the main thread: there is nothing to hand over. */
   if (remote && emscripten_is_main_browser_thread())
      remote = false;
#else
   if (remote)
      return WINK_ERROR_BAD_ATTRIBUTE;
#endif

   int status;
   if (remote) {
      if (!util_queue_init_external(&wink.control, "wink", 16, 0, NULL,
                                    wink_kick, NULL))
         return WINK_ERROR_OUT_OF_MEMORY;
      wink.remote = true;
      struct init_rpc rpc = { &a, WINK_OK };
      wink_rpc(init_rpc, &rpc);
      status = rpc.status;
      if (status != WINK_OK) {
         util_queue_destroy(&wink.control);
         wink.remote = false;
      }
   } else {
      status = init_here(&a);
   }
   wink.initialized = status == WINK_OK;
   return status;
}

static void
shutdown_rpc(void *p)
{
   (void)p;
   st_api_make_current(NULL, NULL, NULL);
   st_screen_destroy(&wink.fscreen);
   wink.fscreen.screen->destroy(wink.fscreen.screen);
   wgpuDeviceRelease(wink.device);
}

void
wink_shutdown(void)
{
   if (!wink.initialized || !calling_thread_ok())
      return;
   if (tls_current)
      wink_make_current(NULL, NULL, NULL);
   wink_rpc(shutdown_rpc, NULL);
   if (wink.remote)
      util_queue_destroy(&wink.control);
   memset(&wink, 0, sizeof(wink));
}

struct versions_rpc {
   int core, compat, es;
};

static void
versions_rpc(void *p)
{
   struct versions_rpc *v = p;
   int es1 = 0;
   st_api_query_versions(&wink.fscreen, &wink.options, &v->core, &v->compat,
                         &es1, &v->es);
}

int
wink_query_versions(int *core, int *compat, int *es)
{
   int status = check_ready();
   if (status != WINK_OK)
      return status;
   struct versions_rpc v = { 0, 0, 0 };
   wink_rpc(versions_rpc, &v);
   if (core) *core = v.core;
   if (compat) *compat = v.compat;
   if (es) *es = v.es;
   return WINK_OK;
}

int
wink_device_lost(void)
{
   return p_atomic_read(&wink.lost);
}

void *
wink_webgpu_device(void)
{
   return wink.device;
}

/* ---------------------------------------------------------------------- */
/* Surfaces: the default framebuffer */

static bool
surface_flush_front(struct st_context *st, struct pipe_frontend_drawable *drawable,
                    enum st_attachment_type statt)
{
   (void)st;
   (void)drawable;
   (void)statt;
   /* Nothing to copy to until wink_swap_buffers: the front buffer is just a
    * texture. Single-buffered rendering is shown by the next swap. */
   return true;
}

static bool
surface_flush_swapbuffers(struct st_context *st, struct pipe_frontend_drawable *drawable)
{
   (void)st;
   (void)drawable;
   return true;
}

static bool
surface_validate(struct st_context *st, struct pipe_frontend_drawable *drawable,
                 const enum st_attachment_type *statts, unsigned count,
                 struct pipe_resource **out, struct pipe_resource **resolve)
{
   (void)st;
   wink_surface *s = (wink_surface *)drawable;
   struct pipe_screen *screen = wink.fscreen.screen;

   for (unsigned i = 0; i < count; ++i) {
      const enum st_attachment_type statt = statts[i];
      out[i] = NULL;
      if (statt >= ST_ATTACHMENT_COUNT)
         return false;

      if (!s->attachments[statt]) {
         enum pipe_format format = PIPE_FORMAT_NONE;
         unsigned bind = 0;
         switch (statt) {
         case ST_ATTACHMENT_FRONT_LEFT:
         case ST_ATTACHMENT_BACK_LEFT:
            format = s->visual.color_format;
            /* Never PIPE_BIND_DISPLAY_TARGET: nothing is scanned out, the
             * texture is copied onto the canvas by wink_swap_buffers. */
            bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW;
            break;
         case ST_ATTACHMENT_DEPTH_STENCIL:
            format = s->visual.depth_stencil_format;
            bind = PIPE_BIND_DEPTH_STENCIL;
            break;
         default:
            break;
         }
         if (format == PIPE_FORMAT_NONE)
            continue;

         struct pipe_resource templ;
         memset(&templ, 0, sizeof(templ));
         templ.target = PIPE_TEXTURE_2D;
         templ.format = format;
         templ.width0 = s->width;
         templ.height0 = s->height;
         templ.depth0 = 1;
         templ.array_size = 1;
         templ.nr_samples = s->visual.samples;
         templ.nr_storage_samples = s->visual.samples;
         templ.bind = bind;
         s->attachments[statt] = screen->resource_create(screen, &templ);
         if (!s->attachments[statt])
            return false;
      }
      /* The caller owns what it is handed. */
      pipe_resource_reference(&out[i], s->attachments[statt]);
   }
   if (resolve)
      *resolve = NULL;
   return true;
}

static enum pipe_format
choose_depth_stencil(unsigned depth, unsigned stencil)
{
   struct pipe_screen *screen = wink.fscreen.screen;
   static const enum pipe_format d24s8[] = {
      PIPE_FORMAT_Z24_UNORM_S8_UINT, PIPE_FORMAT_S8_UINT_Z24_UNORM, PIPE_FORMAT_Z32_FLOAT_S8X24_UINT,
   };
   static const enum pipe_format d32s8[] = {
      PIPE_FORMAT_Z32_FLOAT_S8X24_UINT, PIPE_FORMAT_Z24_UNORM_S8_UINT,
   };
   static const enum pipe_format d32[] = {
      PIPE_FORMAT_Z32_FLOAT, PIPE_FORMAT_Z32_FLOAT_S8X24_UINT,
   };
   static const enum pipe_format d24[] = {
      PIPE_FORMAT_Z24X8_UNORM, PIPE_FORMAT_Z24_UNORM_S8_UINT, PIPE_FORMAT_Z32_FLOAT,
   };
   static const enum pipe_format d16[] = {
      PIPE_FORMAT_Z16_UNORM, PIPE_FORMAT_Z24X8_UNORM, PIPE_FORMAT_Z32_FLOAT,
   };
   const enum pipe_format *list;
   unsigned n;
   if (!depth && !stencil)
      return PIPE_FORMAT_NONE;
   if (stencil) {
      list = depth > 24 ? d32s8 : d24s8;
      n = depth > 24 ? ARRAY_SIZE(d32s8) : ARRAY_SIZE(d24s8);
   } else if (depth > 24) {
      list = d32; n = ARRAY_SIZE(d32);
   } else if (depth > 16) {
      list = d24; n = ARRAY_SIZE(d24);
   } else {
      list = d16; n = ARRAY_SIZE(d16);
   }
   for (unsigned i = 0; i < n; ++i) {
      if (screen->is_format_supported(screen, list[i], PIPE_TEXTURE_2D, 0, 0,
                                      PIPE_BIND_DEPTH_STENCIL))
         return list[i];
   }
   return PIPE_FORMAT_NONE;
}

static wink_surface *
surface_create_here(const wink_surface_attribs *a, int *status)
{
   wink_surface *s = calloc(1, sizeof(*s));
   if (!s) {
      set_status(status, WINK_ERROR_OUT_OF_MEMORY);
      return NULL;
   }
   s->width = a->width;
   s->height = a->height;
   s->offscreen = a->offscreen;
   s->manage_canvas_size = a->manage_canvas_size;
   s->opaque = a->alpha_bits == 0 || a->opaque;
   s->swap_interval = wink.remote ? 1 : WINK_SWAP_UNPACED;
   util_queue_fence_init(&s->swap_fence[0]);
   util_queue_fence_init(&s->swap_fence[1]);

   /* RGBA8 even without alpha: the canvas is configured RGBA8Unorm so the
    * present is a plain copy (WebGPU will not copy between RGBA and BGRA),
    * and an opaque composite mode is what hides the alpha channel. A native
    * surface is BGRA8, which is what Metal presents. */
   s->visual.buffer_mask = ST_ATTACHMENT_FRONT_LEFT_MASK | ST_ATTACHMENT_BACK_LEFT_MASK;
#ifdef __EMSCRIPTEN__
   s->visual.color_format = a->srgb ? PIPE_FORMAT_R8G8B8A8_SRGB : PIPE_FORMAT_R8G8B8A8_UNORM;
#else
   s->visual.color_format = a->srgb ? PIPE_FORMAT_B8G8R8A8_SRGB : PIPE_FORMAT_B8G8R8A8_UNORM;
#endif
   s->visual.depth_stencil_format = choose_depth_stencil(a->depth_bits, a->stencil_bits);
   if (s->visual.depth_stencil_format != PIPE_FORMAT_NONE)
      s->visual.buffer_mask |= ST_ATTACHMENT_DEPTH_STENCIL_MASK;
   else if (a->depth_bits || a->stencil_bits) {
      free(s);
      set_status(status, WINK_ERROR_BAD_ATTRIBUTE);
      return NULL;
   }
   s->visual.accum_format = PIPE_FORMAT_NONE;
   s->visual.samples = 0;

   s->base.visual = &s->visual;
   s->base.fscreen = &wink.fscreen;
   /* st/mesa keys its framebuffer cache on this; any unique non-zero value. */
   s->base.ID = ++wink.next_drawable_id;
   s->base.stamp = 1;
   s->base.flush_front = surface_flush_front;
   s->base.validate = surface_validate;
   s->base.flush_swapbuffers = surface_flush_swapbuffers;

   if (!s->offscreen) {
#ifdef __EMSCRIPTEN__
      s->canvas = wink_js_create_surface(a->canvas);
#else
      s->canvas = wink_native_create_surface(a->native_layer);
#endif
      if (!s->canvas) {
         free(s);
         set_status(status, WINK_ERROR_NO_CANVAS);
         return NULL;
      }
   }
   set_status(status, WINK_OK);
   return s;
}

struct surface_create_rpc {
   const wink_surface_attribs *attribs;
   int status;
   wink_surface *result;
};

static void
surface_create_rpc(void *p)
{
   struct surface_create_rpc *rpc = p;
   rpc->result = surface_create_here(rpc->attribs, &rpc->status);
}

wink_surface *
wink_surface_create(const wink_surface_attribs *attribs, int *status)
{
   int ready = check_ready();
   if (ready != WINK_OK) {
      set_status(status, ready);
      return NULL;
   }
   wink_surface_attribs a;
   wink_surface_attribs_default(&a);
   if (!take_attribs(&a, sizeof(a), attribs) ||
       a.width <= 0 || a.height <= 0 ||
       (a.alpha_bits != 0 && a.alpha_bits != 8) ||
       (a.depth_bits != 0 && a.depth_bits != 16 && a.depth_bits != 24 && a.depth_bits != 32) ||
       (a.stencil_bits != 0 && a.stencil_bits != 8) ||
       a.samples > 1) {
      set_status(status, WINK_ERROR_BAD_ATTRIBUTE);
      return NULL;
   }
   struct surface_create_rpc rpc = { &a, WINK_OK, NULL };
   wink_rpc(surface_create_rpc, &rpc);
   set_status(status, rpc.status);
   return rpc.result;
}

static void
drop_attachments(wink_surface *s)
{
   for (unsigned i = 0; i < ST_ATTACHMENT_COUNT; ++i)
      pipe_resource_reference(&s->attachments[i], NULL);
}

static void
surface_destroy_rpc(void *p)
{
   wink_surface *s = p;
   /* st/mesa caches a framebuffer per drawable; it must forget this one. */
   st_api_destroy_drawable(&s->base);
   drop_attachments(s);
   if (s->canvas) {
      if (s->configured)
         wgpuSurfaceUnconfigure(s->canvas);
      wgpuSurfaceRelease(s->canvas);
   }
   util_queue_fence_destroy(&s->swap_fence[0]);
   util_queue_fence_destroy(&s->swap_fence[1]);
   free(s);
}

void
wink_surface_destroy(wink_surface *s)
{
   if (!s || !wink.initialized || !calling_thread_ok())
      return;
   if (tls_draw == s || tls_read == s)
      wink_make_current(tls_current, tls_draw == s ? NULL : tls_draw,
                        tls_read == s ? NULL : tls_read);
   /* Remote: nothing may still be presenting it. */
   util_queue_fence_wait(&s->swap_fence[0]);
   util_queue_fence_wait(&s->swap_fence[1]);
   wink_rpc(surface_destroy_rpc, s);
}

struct resize_rpc {
   wink_surface *s;
   unsigned width, height;
};

static void
surface_resize_here(wink_surface *s, unsigned width, unsigned height)
{
   /* The attachments are the old size: drop them so validate builds new
    * ones, and bump the stamp, which is the only thing that makes st/mesa
    * ask again. The canvas is reconfigured by the next swap. */
   s->width = width;
   s->height = height;
   drop_attachments(s);
   p_atomic_inc(&s->base.stamp);
   s->configured = false;
   /* And the stamp is only looked at while framebuffer state is dirty, as
    * DRI's drawable invalidation knows: without this a context bound to the
    * surface keeps drawing into, and reading from, the old size. A context
    * bound later revalidates when it is made current. */
   struct st_context *st = st_api_get_current();
   if (st)
      st_context_invalidate_state(st, ST_INVALIDATE_FB_STATE);
}

static void
resize_rpc(void *p)
{
   struct resize_rpc *rpc = p;
   surface_resize_here(rpc->s, rpc->width, rpc->height);
}

#if MESA_GLTHREAD_REMOTE
static void
resize_glthread(struct gl_context *ctx, void *p)
{
   (void)ctx;
   resize_rpc(p);
}
#endif

int
wink_surface_resize(wink_surface *s, int width, int height)
{
   int status = check_ready();
   if (status != WINK_OK)
      return status;
   if (!s)
      return WINK_ERROR_BAD_SURFACE;
   if (width <= 0 || height <= 0)
      return WINK_ERROR_BAD_ATTRIBUTE;
   if ((unsigned)width == s->width && (unsigned)height == s->height)
      return WINK_OK;
   struct resize_rpc rpc = { s, (unsigned)width, (unsigned)height };
#if MESA_GLTHREAD_REMOTE
   /* After the frames already recorded against the old size, not before. */
   if (wink.remote && tls_current && (tls_draw == s || tls_read == s)) {
      _mesa_glthread_call(tls_current->st->ctx, resize_glthread, &rpc);
      return WINK_OK;
   }
#endif
   wink_rpc(resize_rpc, &rpc);
   return WINK_OK;
}

void
wink_surface_get_size(const wink_surface *s, int *width, int *height)
{
   if (width) *width = s ? (int)s->width : 0;
   if (height) *height = s ? (int)s->height : 0;
}

int
wink_swap_interval(wink_surface *s, int interval)
{
   if (!s)
      return WINK_ERROR_BAD_SURFACE;
   if (interval < WINK_SWAP_UNPACED)
      return WINK_ERROR_BAD_ATTRIBUTE;
   s->swap_interval = interval;
   return WINK_OK;
}

void
wink_yield_frames(int frames)
{
   wink_js_yield_frame(frames > 0 ? frames : 0);
}

/* ---------------------------------------------------------------------- */
/* Contexts */

static unsigned
st_flags_for(uint32_t flags)
{
   unsigned out = 0;
   if (flags & WINK_CONTEXT_DEBUG) out |= ST_CONTEXT_FLAG_DEBUG;
   if (flags & WINK_CONTEXT_FORWARD_COMPATIBLE) out |= ST_CONTEXT_FLAG_FORWARD_COMPATIBLE;
   if (flags & WINK_CONTEXT_NO_ERROR) out |= ST_CONTEXT_FLAG_NO_ERROR;
   if (flags & WINK_CONTEXT_RELEASE_NONE) out |= ST_CONTEXT_FLAG_RELEASE_NONE;
   return out;
}

static unsigned
pipe_flags_for(uint32_t flags)
{
   unsigned out = 0;
   if (flags & WINK_CONTEXT_ROBUST_ACCESS) out |= PIPE_CONTEXT_ROBUST_BUFFER_ACCESS;
   if (flags & WINK_CONTEXT_LOSE_ON_RESET) out |= PIPE_CONTEXT_LOSE_CONTEXT_ON_RESET;
   return out;
}

static struct st_context *
create_st(gl_api profile, const wink_context_attribs *a, enum st_context_error *error)
{
   struct st_context_attribs attribs;
   memset(&attribs, 0, sizeof(attribs));
   attribs.profile = profile;
   /* st/mesa computes the version it can support and only compares it with
    * what was asked for; asking for 1.0 skips the comparison and yields the
    * real ceiling, which wink_context_create then checks itself. */
   attribs.major = 1;
   attribs.minor = 0;
   attribs.flags = st_flags_for(a->flags);
   attribs.context_flags = pipe_flags_for(a->flags);
   /* A visual with no formats: _mesa_make_current only compares components
    * both sides have, so this context can be made current with any surface.
    * Double-buffered, since every wink surface is. */
   attribs.visual.buffer_mask = ST_ATTACHMENT_FRONT_LEFT_MASK | ST_ATTACHMENT_BACK_LEFT_MASK;
   attribs.visual.color_format = PIPE_FORMAT_NONE;
   attribs.visual.depth_stencil_format = PIPE_FORMAT_NONE;
   attribs.visual.accum_format = PIPE_FORMAT_NONE;
   attribs.options = wink.options;
   return st_api_create_context(&wink.fscreen, &attribs, error,
                                a->share ? a->share->st : NULL);
}

static wink_context *
context_create_here(const wink_context_attribs *a, int *status)
{
   const int want = a->major * 10 + a->minor;

   gl_api profile;
   switch (a->api) {
   case WINK_API_OPENGL_ES:
      if (want && want < 20) {
         /* ES 1.x is not built (gles1=disabled). */
         set_status(status, WINK_ERROR_BAD_VERSION);
         return NULL;
      }
      profile = API_OPENGLES2;
      break;
   case WINK_API_OPENGL_CORE:
      /* There is no core profile below 3.2; GLX and EGL give compatibility
       * for such a request, and so does this. */
      profile = want && want < 32 ? API_OPENGL_COMPAT : API_OPENGL_CORE;
      break;
   case WINK_API_OPENGL_COMPAT:
      profile = API_OPENGL_COMPAT;
      break;
   default:
      set_status(status, WINK_ERROR_BAD_PROFILE);
      return NULL;
   }

   enum st_context_error error = ST_CONTEXT_SUCCESS;
   struct st_context *st = create_st(profile, a, &error);
   if (!st) {
      set_status(status, error == ST_CONTEXT_ERROR_NO_MEMORY ? WINK_ERROR_OUT_OF_MEMORY
                                                             : WINK_ERROR_BAD_VERSION);
      return NULL;
   }
   const int have = st->ctx->Version;
   if (want && have < want) {
      st_destroy_context(st);
      set_status(status, WINK_ERROR_BAD_VERSION);
      return NULL;
   }

#if MESA_GLTHREAD_REMOTE
   /* The application's GL calls are recorded on its thread and run here. */
   if (wink.remote && !_mesa_glthread_init_remote(st->ctx, wink_kick, NULL)) {
      st_destroy_context(st);
      set_status(status, WINK_ERROR_OUT_OF_MEMORY);
      return NULL;
   }
#endif

   wink_context *c = calloc(1, sizeof(*c));
   if (!c) {
      st_destroy_context(st);
      set_status(status, WINK_ERROR_OUT_OF_MEMORY);
      return NULL;
   }
   c->st = st;
   c->api = profile == API_OPENGLES2 ? WINK_API_OPENGL_ES
          : profile == API_OPENGL_CORE ? WINK_API_OPENGL_CORE : WINK_API_OPENGL_COMPAT;
   c->major = have / 10;
   c->minor = have % 10;
   set_status(status, WINK_OK);
   return c;
}

struct context_create_rpc {
   const wink_context_attribs *attribs;
   int status;
   wink_context *result;
};

static void
context_create_rpc(void *p)
{
   struct context_create_rpc *rpc = p;
   rpc->result = context_create_here(rpc->attribs, &rpc->status);
}

wink_context *
wink_context_create(const wink_context_attribs *attribs, int *status)
{
   int ready = check_ready();
   if (ready != WINK_OK) {
      set_status(status, ready);
      return NULL;
   }
   wink_context_attribs a;
   wink_context_attribs_default(&a);
   if (!take_attribs(&a, sizeof(a), attribs) || a.major < 0 || a.minor < 0) {
      set_status(status, WINK_ERROR_BAD_ATTRIBUTE);
      return NULL;
   }
   struct context_create_rpc rpc = { &a, WINK_OK, NULL };
   wink_rpc(context_create_rpc, &rpc);
   set_status(status, rpc.status);
   return rpc.result;
}

static void
context_destroy_rpc(void *p)
{
   wink_context *c = p;
   st_destroy_context(c->st);
   free(c);
}

void
wink_context_destroy(wink_context *c)
{
   if (!c || !wink.initialized || !calling_thread_ok())
      return;
   if (tls_current == c)
      wink_make_current(NULL, NULL, NULL);
   wink_rpc(context_destroy_rpc, c);
}

void
wink_context_get_version(const wink_context *c, int *api, int *major, int *minor)
{
   if (api) *api = c ? (int)c->api : -1;
   if (major) *major = c ? c->major : 0;
   if (minor) *minor = c ? c->minor : 0;
}

struct make_current_rpc {
   wink_context *c;
   wink_surface *draw, *read;
   bool ok;
};

static void
make_current_rpc(void *p)
{
   struct make_current_rpc *rpc = p;
#if MESA_GLTHREAD_REMOTE
   /* What the application recorded before this must run against the old
    * binding; this is the driver thread, so this runs it. */
   if (wink.remote && rpc->c)
      _mesa_glthread_finish(rpc->c->st->ctx);
#endif
   rpc->ok = st_api_make_current(rpc->c ? rpc->c->st : NULL,
                                 rpc->draw ? &rpc->draw->base : NULL,
                                 rpc->read ? &rpc->read->base : NULL);
}

int
wink_make_current(wink_context *c, wink_surface *draw, wink_surface *read)
{
   int status = check_ready();
   if (status != WINK_OK)
      return status;
   if (!c && (draw || read))
      return WINK_ERROR_BAD_CONTEXT;

#if MESA_GLTHREAD_REMOTE
   /* Whatever this thread recorded for the old context goes first. */
   if (wink.remote && tls_current)
      _mesa_glthread_flush_batch(tls_current->st->ctx);
#endif

   struct make_current_rpc rpc = { c, draw, read, false };
   wink_rpc(make_current_rpc, &rpc);
   if (!rpc.ok)
      return WINK_ERROR_BAD_CONTEXT;

#if MESA_GLTHREAD_REMOTE
   /* This thread becomes the context's application thread: its GL calls go
    * through the marshal dispatch into the context's batches. */
   if (wink.remote) {
      struct gl_context *ctx = c ? c->st->ctx : NULL;
      _mesa_glapi_set_context(ctx);
      _mesa_glapi_set_dispatch(ctx ? ctx->MarshalExec : NULL);
   }
#endif
   tls_current = c;
   tls_draw = c ? draw : NULL;
   tls_read = c ? read : NULL;
   return WINK_OK;
}

wink_context *
wink_get_current_context(void)
{
   return tls_current;
}

wink_surface *
wink_get_current_surface(void)
{
   return tls_draw;
}

void *
wink_get_proc_address(const char *name)
{
   /* The dispatch stubs, which call through the calling thread's table --
    * the execution one here, the marshal one on a remote context's
    * application thread: valid in any context, and the only GL wink exposes. */
   if (!name || strncmp(name, "gl", 2) != 0)
      return NULL;
   return (void *)_mesa_glapi_get_proc_address(name);
}

/* ---------------------------------------------------------------------- */
/* Presentation */

static bool
configure_canvas(wink_surface *s)
{
   if (s->configured)
      return true;
   WGPUSurfaceConfiguration cfg;
   memset(&cfg, 0, sizeof(cfg));
   cfg.device = wink.device;
   cfg.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
#ifdef __EMSCRIPTEN__
   cfg.format = WGPUTextureFormat_RGBA8Unorm;
   /* emdawnwebgpu sets the canvas element's size from these unless they are
    * zero; zero leaves the page in charge of it. */
   cfg.width = s->manage_canvas_size ? s->width : 0;
   cfg.height = s->manage_canvas_size ? s->height : 0;
#else
   /* A native surface is the surface's size, which the window's resizes set. */
   cfg.format = WGPUTextureFormat_BGRA8Unorm;
   cfg.width = s->width;
   cfg.height = s->height;
#endif
   /* A browser composites a WebGPU canvas either opaque or premultiplied;
    * GPUCanvasAlphaMode has no unpremultiplied. */
   cfg.alphaMode = s->opaque ? WGPUCompositeAlphaMode_Opaque
                             : WGPUCompositeAlphaMode_Premultiplied;
   cfg.presentMode = WGPUPresentMode_Fifo;
   wgpuSurfaceConfigure(s->canvas, &cfg);
   s->configured = true;
   return true;
}

#ifndef __EMSCRIPTEN__
/* WINK_CAPTURE=N:FILE.bmp writes the back buffer of the Nth swap to a file,
 * for looking at what reached the screen where nothing can capture it. Read
 * back through the context, as glReadPixels would. */
static void
maybe_capture(struct st_context *st, struct pipe_resource *back)
{
   static int64_t frame = -1;
   static uint64_t swaps;
   static char path[512];
   if (frame < 0) {
      const char *env = getenv("WINK_CAPTURE");
      frame = 0;
      if (!env || sscanf(env, "%" SCNd64 ":%511s", &frame, path) != 2)
         frame = INT64_MAX;
   }
   if (++swaps != (uint64_t)frame || !st)
      return;

   const unsigned w = back->width0, h = back->height0;
   struct pipe_transfer *transfer = NULL;
   const uint8_t *data = pipe_texture_map(st->pipe, back, 0, 0, PIPE_MAP_READ, 0, 0, w, h, &transfer);
   FILE *f = data ? fopen(path, "wb") : NULL;
   if (f) {
      /* A bottom-up 32-bit BMP, which is BGRA: swap for RGBA buffers. */
      const bool rgba = back->format == PIPE_FORMAT_R8G8B8A8_UNORM ||
                        back->format == PIPE_FORMAT_R8G8B8A8_SRGB;
      const uint32_t bytes = w * h * 4;
      uint8_t header[54] = { 'B', 'M' };
      const uint32_t fields[] = { 54 + bytes, 54, 40, w, h };
      memcpy(header + 2, &fields[0], 4);
      memcpy(header + 10, &fields[1], 4);
      memcpy(header + 14, &fields[2], 4);
      memcpy(header + 18, &fields[3], 4);
      memcpy(header + 22, &fields[4], 4);
      header[26] = 1;
      header[28] = 32;
      fwrite(header, 1, sizeof(header), f);
      uint8_t *row = malloc(w * 4);
      for (unsigned y = h; row && y-- > 0;) {
         memcpy(row, data + (uint64_t)y * transfer->stride, w * 4);
         for (unsigned x = 0; rgba && x < w; ++x) {
            const uint8_t r = row[x * 4];
            row[x * 4] = row[x * 4 + 2];
            row[x * 4 + 2] = r;
         }
         fwrite(row, 1, w * 4, f);
      }
      free(row);
      fclose(f);
      fprintf(stderr, "wink: swap %" PRIu64 " written to %s\n", swaps, path);
   }
   if (data)
      pipe_texture_unmap(st->pipe, transfer);
}
#endif

/* Flush the current context and copy the back buffer onto the canvas. */
static void
swap_here(wink_surface *s)
{
   /* END_OF_FRAME submits the frame. Not glFinish: that also refreshes
    * persistent mappings, a readback every frame for nothing. */
   struct st_context *st = st_api_get_current();
   if (st)
      st_context_flush(st, ST_FLUSH_END_OF_FRAME, NULL, NULL, NULL);
   if (s->offscreen)
      return;

   struct pipe_resource *back = s->attachments[ST_ATTACHMENT_BACK_LEFT];
   WGPUTexture from = back ? gex_wgpu_resource_texture(back) : NULL;
   if (!from)
      return; /* nothing has been drawn to it yet */
#ifndef __EMSCRIPTEN__
   maybe_capture(st, back);
#endif

   configure_canvas(s);
   WGPUSurfaceTexture current;
   memset(&current, 0, sizeof(current));
   wgpuSurfaceGetCurrentTexture(s->canvas, &current);
   if (!current.texture ||
       (current.status != WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal &&
        current.status != WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal)) {
      if (current.texture)
         wgpuTextureRelease(current.texture);
      s->configured = false;
      return;
   }

   /* The canvas may be a different size from the surface -- the page resized
    * it, or the surface was resized and the page has not caught up -- and a
    * copy past either edge is a validation error that loses the frame. */
   WGPUExtent3D extent = {
      MIN2(s->width, wgpuTextureGetWidth(current.texture)),
      MIN2(s->height, wgpuTextureGetHeight(current.texture)),
      1,
   };
   WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(wink.device, NULL);
   if (encoder) {
      WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      src.texture = from;
      WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      dst.texture = current.texture;
      wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);
      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      if (commands) {
         WGPUQueue queue = wgpuDeviceGetQueue(wink.device);
         wgpuQueueSubmit(queue, 1, &commands);
         wgpuQueueRelease(queue);
         wgpuCommandBufferRelease(commands);
      }
      wgpuCommandEncoderRelease(encoder);
   }
   /* No wgpuSurfacePresent in a browser: the frame is presented when this
    * thread next yields to its event loop. Natively it is presented here,
    * FIFO, which is also the pacing. */
#ifndef __EMSCRIPTEN__
   wgpuSurfacePresent(s->canvas);
#endif
   wgpuTextureRelease(current.texture);
}

static void
swap_rpc(void *p)
{
   swap_here(p);
}

#if MESA_GLTHREAD_REMOTE
/* A remote swap, run on the main thread after the frame's batches. The
 * runner must give the event loop a turn for the frame to be presented, and
 * waiting for the next animation frame is also the pacing. */
static void
swap_glthread(struct gl_context *ctx, void *p)
{
   (void)ctx;
   wink_surface *s = p;
   swap_here(s);
   wink_js_yield_frame(s->swap_interval > 0 ? s->swap_interval : 0);
}
#endif

int
wink_swap_buffers(wink_surface *s)
{
   int status = check_ready();
   if (status != WINK_OK)
      return status;
   if (!s)
      return WINK_ERROR_BAD_SURFACE;

#if MESA_GLTHREAD_REMOTE
   if (wink.remote && tls_current) {
      /* At most two frames in flight: queue this one, then wait until the
       * one before it has been presented. */
      const unsigned i = s->swap_index;
      s->swap_index ^= 1;
      static uint64_t swap_wait_ns;
      util_queue_fence_wait(&s->swap_fence[i]);
      _mesa_glthread_call_async(tls_current->st->ctx, swap_glthread, s,
                                &s->swap_fence[i]);
      const int64_t t1 = os_time_get_nano();
      util_queue_fence_wait(&s->swap_fence[i ^ 1]);
      swap_wait_ns += (uint64_t)(os_time_get_nano() - t1);
      /* Every 240 frames: how often this thread waited for the main thread
       * between swaps, and for which GL calls -- each wait is a round trip
       * through the main thread's event loop. */
      static unsigned swaps;
      if (++swaps % 240 == 0) {
         char why[640];
         uint64_t waits, copies;
         _mesa_glthread_remote_census(why, sizeof(why), &waits, &copies);
         const uint64_t uploads = _mesa_glthread_remote_uploads;
         const uint64_t upload_kib = _mesa_glthread_remote_upload_bytes >> 10;
         _mesa_glthread_remote_uploads = 0;
         _mesa_glthread_remote_upload_bytes = 0;
         /* And how long it was held back without asking anything: by the
          * swap, which waits for the frame before this one to be presented,
          * and by glthread's queue, which holds a few batches at most. */
         static uint64_t last_full_ns, last_readbacks, last_kicks;
         extern uint64_t gex_wgpu_readbacks;
         const uint64_t full_ns = tls_current->st->ctx->GLThread.queue.full_wait_ns;
         const uint64_t readbacks = gex_wgpu_readbacks;
         const uint64_t kicks = tls_current->st->ctx->GLThread.queue.kicks;
         fprintf(stderr, "wink: the GL thread waited for the main thread %.1f "
                 "times a frame over 240 frames (%s); %.1f client-array draws "
                 "and %.1f big buffer uploads (%" PRIu64 " KiB in all) a frame "
                 "copied instead, %u read live; blocked %.2f ms a frame on the "
                 "swap, %.2f on a full queue of %d batches; %.2f GPU readbacks "
                 "and %.1f wakeups a frame on main\n", waits / 240.0,
                 why[0] ? why : "none", copies / 240.0, uploads / 240.0,
                 upload_kib, _mesa_glthread_remote_unmatched,
                 swap_wait_ns / 240.0 / 1e6, (full_ns - last_full_ns) / 240.0 / 1e6,
                 (int)tls_current->st->ctx->GLThread.queue.max_jobs + 2,
                 (readbacks - last_readbacks) / 240.0,
                 (kicks - last_kicks) / 240.0);
         swap_wait_ns = 0;
         last_full_ns = full_ns;
         last_readbacks = readbacks;
         last_kicks = kicks;
      }
      return WINK_OK;
   }
#endif
   wink_rpc(swap_rpc, s);
   /* Remote without a current context: nothing to pace, and this thread
    * need not be one that can suspend. */
   if (!wink.remote && s->swap_interval != WINK_SWAP_UNPACED)
      wink_js_yield_frame(s->swap_interval);
   return WINK_OK;
}

/* ---------------------------------------------------------------------- */
/* Vulkan: src/wink/vulkan's entry points, under wink's names. */

#ifdef WINK_HAVE_VULKAN
#include <vulkan/vulkan_core.h>

PFN_vkVoidFunction vk_icdGetInstanceProcAddr(VkInstance instance, const char *name);
VkResult wvk_create_canvas_surface(VkInstance instance, const char *selector,
                                   const VkAllocationCallbacks *allocator, VkSurfaceKHR *surface);

#ifdef __EMSCRIPTEN_PTHREADS__
#include "wink_vk_rpc.h"
#endif

WINK_EXPORT wink_vk_void_function
wink_vk_get_instance_proc_addr(void *instance, const char *name)
{
   PFN_vkVoidFunction actual = vk_icdGetInstanceProcAddr((VkInstance)instance, name);
#ifdef __EMSCRIPTEN_PTHREADS__
   if (wink.remote)
      actual = wink_vk_rpc_wrap(name, actual);
#endif
   return (wink_vk_void_function)actual;
}

struct wink_vk_surface_rpc {
   VkInstance instance;
   const char *selector;
   VkSurfaceKHR surface;
   VkResult result;
};

static void
wink_vk_surface_create_rpc(void *data)
{
   struct wink_vk_surface_rpc *a = data;
   a->result = wvk_create_canvas_surface(a->instance, a->selector, NULL, &a->surface);
}

WINK_EXPORT int
wink_vk_create_surface(void *instance, const char *selector, uint64_t *surface)
{
   struct wink_vk_surface_rpc a = { .instance = (VkInstance)instance, .selector = selector };
   wink_rpc(wink_vk_surface_create_rpc, &a);
   *surface = (uint64_t)a.surface;
   return a.result;
}
#else
/* Built without the Vulkan driver: there is none to find, so windowing
 * layers can ask without knowing how wink was built. */
WINK_EXPORT wink_vk_void_function
wink_vk_get_instance_proc_addr(void *instance, const char *name)
{
   return NULL;
}

WINK_EXPORT int
wink_vk_create_surface(void *instance, const char *selector, uint64_t *surface)
{
   *surface = 0;
   return -3; /* VK_ERROR_INITIALIZATION_FAILED */
}
#endif
