// SPDX-License-Identifier: MIT
// apitrace's handles, uniform remapping and call dispatcher remain upstream.
#include "glws.hpp"
#include "wink.h"
#include <cstdio>
#include <cstdlib>

namespace {
void check(int status) {
    if (status != WINK_OK) {
        std::fprintf(stderr, "wink replay error: %s\n", wink_status_string(status));
        std::exit(1);
    }
}
struct WinkDrawable : glws::Drawable {
    wink_surface *surface;
    WinkDrawable(const glws::Visual *v, int w, int h, bool pb) : Drawable(v, w, h, pb) {
        wink_surface_attribs a;
        wink_surface_attribs_default(&a);
        a.width = w; a.height = h;
        a.canvas = "#canvas";
        a.alpha_bits = 8;
        a.depth_bits = 24; a.stencil_bits = 8;
        a.manage_canvas_size = true;
#ifndef __EMSCRIPTEN__
        a.offscreen = true;
#else
        a.offscreen = pb;
#endif
        int status;
        surface = wink_surface_create(&a, &status);
        check(status);
    }
    ~WinkDrawable() override { wink_surface_destroy(surface); }
    void resize(int w, int h) override {
        check(wink_surface_resize(surface, w, h));
        Drawable::resize(w, h);
    }
    void swapBuffers() override { check(wink_swap_buffers(surface)); }
    void swapBuffersWithDamage(int *, int) override { swapBuffers(); }
};
struct WinkContext : glws::Context {
    wink_context *context;
    WinkContext(const glws::Visual *v, WinkContext *share, bool debug) : Context(v) {
        wink_context_attribs a;
        wink_context_attribs_default(&a);
        a.api = profile.es() ? WINK_API_OPENGL_ES :
                profile.core ? WINK_API_OPENGL_CORE : WINK_API_OPENGL_COMPAT;
        a.major = profile.major; a.minor = profile.minor;
        a.share = share ? share->context : nullptr;
        if (debug) a.flags |= WINK_CONTEXT_DEBUG;
        if (profile.forwardCompatible) a.flags |= WINK_CONTEXT_FORWARD_COMPATIBLE;
        int status;
        context = wink_context_create(&a, &status);
        check(status);
    }
    ~WinkContext() override { wink_context_destroy(context); }
};
}

namespace glws {
void init() { check(wink_init(nullptr)); }
void cleanup() { wink_shutdown(); }
Visual *createVisual(bool doubleBuffer, unsigned samples, Profile profile) {
    if (samples > 1) {
        std::fprintf(stderr, "wink replay unsupported: multisampled default framebuffer\n");
        std::exit(2);
    }
    auto *v = new Visual(profile);
    v->doubleBuffer = doubleBuffer;
    return v;
}
Drawable *createDrawable(const Visual *v, int w, int h, const pbuffer_info *pb) {
    return new WinkDrawable(v, w, h, pb != nullptr);
}
Context *createContext(const Visual *v, Context *share, bool debug) {
    return new WinkContext(v, static_cast<WinkContext *>(share), debug);
}
bool makeCurrentInternal(Drawable *draw, Drawable *read, Context *ctx) {
    int status = wink_make_current(ctx ? static_cast<WinkContext *>(ctx)->context : nullptr,
        draw ? static_cast<WinkDrawable *>(draw)->surface : nullptr,
        read ? static_cast<WinkDrawable *>(read)->surface : nullptr);
    check(status);
    return true;
}
bool processEvents() { return !wink_device_lost(); }
bool bindTexImage(Drawable *, int) { std::fprintf(stderr, "wink replay unsupported: bindTexImage\n"); return false; }
bool releaseTexImage(Drawable *, int) { std::fprintf(stderr, "wink replay unsupported: releaseTexImage\n"); return false; }
bool setPbufferAttrib(Drawable *, const int *) { std::fprintf(stderr, "wink replay unsupported: pbuffer attributes\n"); return false; }
}
