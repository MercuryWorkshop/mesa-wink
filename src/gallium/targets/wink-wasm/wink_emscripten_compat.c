/* SPDX-License-Identifier: MIT */
#define WINK_BUILDING 1
#include "wink_emscripten_compat.h"

#include <stdio.h>
#include <stdlib.h>

struct wink_webgl {
   wink_surface *surface;
   wink_context *context;
};

static wink_context_attribs next_context = {
   .size = sizeof(wink_context_attribs),
   .api = WINK_API_OPENGL_COMPAT,
};
static struct wink_webgl *current;
static int next_transparent;
static unsigned next_init_flags;

void
wink_set_next_init_flags(unsigned flags)
{
   next_init_flags = flags;
}

void
wink_set_swap_interval(int interval)
{
   if (current)
      wink_swap_interval(current->surface, interval);
}

void
wink_set_next_surface_transparent(int transparent)
{
   next_transparent = transparent != 0;
}

void
wink_set_next_context(int api, int major, int minor, unsigned flags)
{
   wink_context_attribs_default(&next_context);
   next_context.api = api;
   next_context.major = major;
   next_context.minor = minor;
   next_context.flags = flags;
}

EMSCRIPTEN_WEBGL_CONTEXT_HANDLE
wink_webgl_create_context(const char *target, const EmscriptenWebGLContextAttributes *attributes)
{
   wink_init_attribs ia;
   wink_init_attribs_default(&ia);
   ia.flags = next_init_flags;
   int status = wink_init(&ia);
   if (status != WINK_OK) {
      fprintf(stderr, "wink: %s\n", wink_status_string(status));
      return 0;
   }

   wink_surface_attribs sa;
   wink_surface_attribs_default(&sa);
   sa.canvas = target;
   int w = 0, h = 0;
   emscripten_get_canvas_element_size(target, &w, &h);
   sa.width = w > 0 ? w : 1;
   sa.height = h > 0 ? h : 1;
   if (attributes) {
      sa.alpha_bits = attributes->alpha ? 8 : 0;
      sa.depth_bits = attributes->depth ? 24 : 0;
      sa.stencil_bits = attributes->stencil ? 8 : 0;
      sa.premultiplied_alpha = attributes->premultipliedAlpha;
   }
   sa.opaque = !next_transparent;

   struct wink_webgl *pair = calloc(1, sizeof(*pair));
   if (!pair)
      return 0;
   pair->surface = wink_surface_create(&sa, &status);
   if (pair->surface)
      pair->context = wink_context_create(&next_context, &status);
   if (!pair->context) {
      fprintf(stderr, "wink: %s\n", wink_status_string(status));
      wink_surface_destroy(pair->surface);
      free(pair);
      return 0;
   }
   return (EMSCRIPTEN_WEBGL_CONTEXT_HANDLE)(uintptr_t)pair;
}

EMSCRIPTEN_RESULT
wink_webgl_make_context_current(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE handle)
{
   struct wink_webgl *h = (struct wink_webgl *)(uintptr_t)handle;
   int status = h ? wink_make_current(h->context, h->surface, h->surface)
                  : wink_make_current(NULL, NULL, NULL);
   if (status != WINK_OK)
      return EMSCRIPTEN_RESULT_FAILED;
   current = h;
   return EMSCRIPTEN_RESULT_SUCCESS;
}

EMSCRIPTEN_RESULT
wink_webgl_destroy_context(EMSCRIPTEN_WEBGL_CONTEXT_HANDLE handle)
{
   struct wink_webgl *h = (struct wink_webgl *)(uintptr_t)handle;
   if (!h)
      return EMSCRIPTEN_RESULT_INVALID_PARAM;
   if (current == h)
      current = NULL;
   wink_context_destroy(h->context);
   wink_surface_destroy(h->surface);
   free(h);
   return EMSCRIPTEN_RESULT_SUCCESS;
}

void
wink_present(void)
{
   if (current)
      wink_swap_buffers(current->surface);
}

void
wink_resize(int width, int height)
{
   if (current)
      wink_surface_resize(current->surface, width, height);
}

void
wink_proc_init(void)
{
}
