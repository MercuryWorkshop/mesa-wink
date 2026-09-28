/* SPDX-License-Identifier: MIT */
/*
 * wink: st/mesa on gex_wgpu, driven directly -- no EGL, no DRI, no winsys.
 *
 * The shape is gex's wire decoder's (web/mobilegl/gexgl-mobilegl.cpp in the
 * gex repository), which has run this frontend on this driver in a browser
 * for a long time: a pipe_frontend_drawable whose attachments are plain
 * textures this file owns, st_api_create_context for the context, and a
 * texture-to-texture copy onto the canvas for presentation. What changes is
 * everything the wire decided for it: sizes, formats, versions and canvases
 * come from the caller, one surface and context each at a time becomes any
 * number, and a context no longer carries a surface's visual (see
 * wink_context_create).
 */
#define WINK_BUILDING 1
#include "wink.h"
#include "wink_private.h"

#include <webgpu/webgpu.h>

#include "frontend/api.h"
#include "mesa/glapi/glapi/glapi.h"
#include "mesa/main/mtypes.h"
#include "mesa/state_tracker/st_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include "gex_wgpu_public.h"

#include <pthread.h>
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
};

struct wink_context {
   struct st_context *st;
   enum wink_api api;
   int major, minor;
};

static struct {
   bool initialized;
   pthread_t owner;
   WGPUDevice device;
   struct pipe_frontend_screen fscreen;
   struct st_config_options options;
   uint32_t next_drawable_id;
   wink_context *current;
   wink_surface *draw, *read;
} wink;

/* ---------------------------------------------------------------------- */

static bool
on_owner_thread(void)
{
#ifdef __EMSCRIPTEN_PTHREADS__
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
   if (!on_owner_thread())
      return WINK_ERROR_WRONG_THREAD;
   if (wink_js_device_lost())
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
   case WINK_ERROR_WRONG_THREAD: return "called from a thread other than wink_init's";
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

int
wink_init(const wink_init_attribs *attribs)
{
   if (wink.initialized)
      return on_owner_thread() ? WINK_OK : WINK_ERROR_WRONG_THREAD;

   wink_init_attribs a;
   wink_init_attribs_default(&a);
   if (!take_attribs(&a, sizeof(a), attribs))
      return WINK_ERROR_BAD_ATTRIBUTE;

   /* Without JSPI the driver's waits would spin on a promise that can never
    * settle, so refuse here instead of hanging on the first readback. */
   if (!wink_js_has_jspi())
      return WINK_ERROR_NO_JSPI;

   apply_js_env();

   WGPUDevice device = a.webgpu_device;
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
   wink.device = device;
   wink.owner = pthread_self();
   wink.initialized = true;
   return WINK_OK;
}

void
wink_shutdown(void)
{
   if (!wink.initialized || !on_owner_thread())
      return;
   st_api_make_current(NULL, NULL, NULL);
   wink.current = NULL;
   wink.draw = wink.read = NULL;
   st_screen_destroy(&wink.fscreen);
   wink.fscreen.screen->destroy(wink.fscreen.screen);
   wgpuDeviceRelease(wink.device);
   memset(&wink, 0, sizeof(wink));
}

int
wink_query_versions(int *core, int *compat, int *es)
{
   int status = check_ready();
   if (status != WINK_OK)
      return status;
   int c = 0, p = 0, es1 = 0, es2 = 0;
   st_api_query_versions(&wink.fscreen, &wink.options, &c, &p, &es1, &es2);
   if (core) *core = c;
   if (compat) *compat = p;
   if (es) *es = es2;
   return WINK_OK;
}

int
wink_device_lost(void)
{
   return wink_js_device_lost();
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

   wink_surface *s = calloc(1, sizeof(*s));
   if (!s) {
      set_status(status, WINK_ERROR_OUT_OF_MEMORY);
      return NULL;
   }
   s->width = a.width;
   s->height = a.height;
   s->offscreen = a.offscreen;
   s->manage_canvas_size = a.manage_canvas_size;
   s->opaque = a.alpha_bits == 0;

   /* RGBA8 even without alpha: the canvas is configured RGBA8Unorm so the
    * present is a plain copy (WebGPU will not copy between RGBA and BGRA),
    * and an opaque composite mode is what hides the alpha channel. */
   s->visual.buffer_mask = ST_ATTACHMENT_FRONT_LEFT_MASK | ST_ATTACHMENT_BACK_LEFT_MASK;
   s->visual.color_format = a.srgb ? PIPE_FORMAT_R8G8B8A8_SRGB : PIPE_FORMAT_R8G8B8A8_UNORM;
   s->visual.depth_stencil_format = choose_depth_stencil(a.depth_bits, a.stencil_bits);
   if (s->visual.depth_stencil_format != PIPE_FORMAT_NONE)
      s->visual.buffer_mask |= ST_ATTACHMENT_DEPTH_STENCIL_MASK;
   else if (a.depth_bits || a.stencil_bits) {
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
      s->canvas = wink_js_create_surface(a.canvas);
      if (!s->canvas) {
         free(s);
         set_status(status, WINK_ERROR_NO_CANVAS);
         return NULL;
      }
   }
   set_status(status, WINK_OK);
   return s;
}

static void
drop_attachments(wink_surface *s)
{
   for (unsigned i = 0; i < ST_ATTACHMENT_COUNT; ++i)
      pipe_resource_reference(&s->attachments[i], NULL);
}

void
wink_surface_destroy(wink_surface *s)
{
   if (!s || check_ready() == WINK_ERROR_WRONG_THREAD)
      return;
   if (wink.draw == s || wink.read == s) {
      wink_context *ctx = wink.current;
      wink_make_current(ctx, wink.draw == s ? NULL : wink.draw,
                        wink.read == s ? NULL : wink.read);
   }
   /* st/mesa caches a framebuffer per drawable; it must forget this one. */
   if (wink.initialized)
      st_api_destroy_drawable(&s->base);
   drop_attachments(s);
   if (s->canvas) {
      if (s->configured)
         wgpuSurfaceUnconfigure(s->canvas);
      wgpuSurfaceRelease(s->canvas);
   }
   free(s);
}

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
   /* The attachments are the old size: drop them so validate builds new
    * ones, and bump the stamp, which is the only thing that makes st/mesa
    * ask again. The canvas is reconfigured by the next swap. */
   s->width = width;
   s->height = height;
   drop_attachments(s);
   p_atomic_inc(&s->base.stamp);
   s->configured = false;
   return WINK_OK;
}

void
wink_surface_get_size(const wink_surface *s, int *width, int *height)
{
   if (width) *width = s ? (int)s->width : 0;
   if (height) *height = s ? (int)s->height : 0;
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
   const int want = a.major * 10 + a.minor;

   gl_api profile;
   switch (a.api) {
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
   struct st_context *st = create_st(profile, &a, &error);
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

void
wink_context_destroy(wink_context *c)
{
   if (!c || check_ready() == WINK_ERROR_WRONG_THREAD)
      return;
   if (wink.current == c) {
      st_api_make_current(NULL, NULL, NULL);
      wink.current = NULL;
      wink.draw = wink.read = NULL;
   }
   st_destroy_context(c->st);
   free(c);
}

void
wink_context_get_version(const wink_context *c, int *api, int *major, int *minor)
{
   if (api) *api = c ? (int)c->api : -1;
   if (major) *major = c ? c->major : 0;
   if (minor) *minor = c ? c->minor : 0;
}

int
wink_make_current(wink_context *c, wink_surface *draw, wink_surface *read)
{
   int status = check_ready();
   if (status != WINK_OK)
      return status;
   if (!c && (draw || read))
      return WINK_ERROR_BAD_CONTEXT;
   if (!st_api_make_current(c ? c->st : NULL, draw ? &draw->base : NULL,
                            read ? &read->base : NULL))
      return WINK_ERROR_BAD_CONTEXT;
   wink.current = c;
   wink.draw = c ? draw : NULL;
   wink.read = c ? read : NULL;
   return WINK_OK;
}

wink_context *
wink_get_current_context(void)
{
   return wink.current;
}

wink_surface *
wink_get_current_surface(void)
{
   return wink.draw;
}

void *
wink_get_proc_address(const char *name)
{
   /* The dispatch stubs, which call through the current context's table:
    * valid in any context, and the only GL wink exposes. */
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
   cfg.format = WGPUTextureFormat_RGBA8Unorm;
   cfg.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopyDst;
   /* emdawnwebgpu sets the canvas element's size from these unless they are
    * zero; zero leaves the page in charge of it. */
   cfg.width = s->manage_canvas_size ? s->width : 0;
   cfg.height = s->manage_canvas_size ? s->height : 0;
   /* A browser composites a WebGPU canvas either opaque or premultiplied;
    * GPUCanvasAlphaMode has no unpremultiplied. */
   cfg.alphaMode = s->opaque ? WGPUCompositeAlphaMode_Opaque
                             : WGPUCompositeAlphaMode_Premultiplied;
   cfg.presentMode = WGPUPresentMode_Fifo;
   wgpuSurfaceConfigure(s->canvas, &cfg);
   s->configured = true;
   return true;
}

int
wink_swap_buffers(wink_surface *s)
{
   int status = check_ready();
   if (status != WINK_OK)
      return status;
   if (!s)
      return WINK_ERROR_BAD_SURFACE;

   /* END_OF_FRAME submits the frame. Not glFinish: that also refreshes
    * persistent mappings, a readback every frame for nothing. */
   struct st_context *st = st_api_get_current();
   if (st)
      st_context_flush(st, ST_FLUSH_END_OF_FRAME, NULL, NULL, NULL);
   if (s->offscreen)
      return WINK_OK;

   struct pipe_resource *back = s->attachments[ST_ATTACHMENT_BACK_LEFT];
   WGPUTexture from = back ? gex_wgpu_resource_texture(back) : NULL;
   if (!from)
      return WINK_OK; /* nothing has been drawn to it yet */

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
      return WINK_OK;
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
   /* No wgpuSurfacePresent: in a browser the frame is presented when this
    * thread next yields to its event loop. */
   wgpuTextureRelease(current.texture);
   return WINK_OK;
}
