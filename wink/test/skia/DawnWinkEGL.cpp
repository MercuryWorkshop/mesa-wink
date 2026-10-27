// SPDX-License-Identifier: MIT
// Test-only EGL dispatch for Dawn's GLES backend. All GL comes from Wink.
// This is not an EGL library: Dawn receives this resolver explicitly.
#include "wink.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int display_token, config_token;
thread_local EGLint error = EGL_SUCCESS;
EGLDisplay display() { return &display_token; }
EGLConfig config() { return &config_token; }
EGLBoolean fail(EGLint code) { error = code; return EGL_FALSE; }
bool valid(EGLDisplay d) { return d == display(); }
template<typename T> T gl(const char* name) {
    auto p = wink_get_proc_address(name);
    if (!p) { std::fprintf(stderr, "missing Wink GL proc: %s\n", name); std::abort(); }
    return reinterpret_cast<T>(p);
}
EGLint wink_eglGetError() { EGLint e = error; error = EGL_SUCCESS; return e; }
EGLDisplay wink_eglGetDisplay(EGLNativeDisplayType) { return display(); }
EGLBoolean wink_eglInitialize(EGLDisplay d, EGLint* major, EGLint* minor) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    if (wink_init(nullptr) != WINK_OK) return fail(EGL_NOT_INITIALIZED);
    if (major) *major = 1;
    if (minor) *minor = 4;
    return EGL_TRUE;
}
EGLBoolean wink_eglTerminate(EGLDisplay d) { return valid(d) ? EGL_TRUE : fail(EGL_BAD_DISPLAY); }
const char* wink_eglQueryString(EGLDisplay d, EGLint name) {
    if (d == EGL_NO_DISPLAY && name == EGL_EXTENSIONS) return "EGL_EXT_client_extensions";
    if (!valid(d)) { fail(EGL_BAD_DISPLAY); return nullptr; }
    switch (name) {
    case EGL_VERSION: return "1.4 Wink Skia test adapter";
    case EGL_VENDOR: return "Wink";
    case EGL_CLIENT_APIS: return "OpenGL_ES";
    case EGL_EXTENSIONS: return "EGL_KHR_create_context EGL_EXT_create_context_robustness "
                               "EGL_KHR_no_config_context EGL_KHR_surfaceless_context EGL_KHR_fence_sync";
    default: fail(EGL_BAD_PARAMETER); return nullptr;
    }
}
EGLBoolean wink_eglBindAPI(EGLenum api) { return api == EGL_OPENGL_ES_API ? EGL_TRUE : fail(EGL_BAD_PARAMETER); }
EGLenum wink_eglQueryAPI() { return EGL_OPENGL_ES_API; }
EGLBoolean wink_eglGetConfigs(EGLDisplay d, EGLConfig* configs, EGLint size, EGLint* count) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    *count = 1;
    if (configs && size > 0) configs[0] = config();
    return EGL_TRUE;
}
EGLBoolean wink_eglChooseConfig(EGLDisplay d, const EGLint*, EGLConfig* configs, EGLint size, EGLint* count) {
    return wink_eglGetConfigs(d, configs, size, count);
}
EGLBoolean wink_eglGetConfigAttrib(EGLDisplay d, EGLConfig c, EGLint name, EGLint* value) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    if (c != config()) return fail(EGL_BAD_CONFIG);
    switch (name) {
    case EGL_RED_SIZE: case EGL_GREEN_SIZE: case EGL_BLUE_SIZE: case EGL_ALPHA_SIZE: *value = 8; break;
    case EGL_BUFFER_SIZE: *value = 32; break;
    case EGL_DEPTH_SIZE: *value = 24; break;
    case EGL_STENCIL_SIZE: *value = 8; break;
    case EGL_SAMPLES: case EGL_SAMPLE_BUFFERS: *value = 0; break;
    case EGL_SURFACE_TYPE: *value = EGL_PBUFFER_BIT; break;
    case EGL_RENDERABLE_TYPE: case EGL_CONFORMANT: *value = EGL_OPENGL_ES2_BIT | EGL_OPENGL_ES3_BIT; break;
    case EGL_CONFIG_ID: *value = 1; break;
    case EGL_COLOR_BUFFER_TYPE: *value = EGL_RGB_BUFFER; break;
    case EGL_CONFIG_CAVEAT: *value = EGL_NONE; break;
    default: return fail(EGL_BAD_ATTRIBUTE);
    }
    return EGL_TRUE;
}
EGLContext wink_eglCreateContext(EGLDisplay d, EGLConfig, EGLContext share, const EGLint* attributes) {
    if (!valid(d)) { fail(EGL_BAD_DISPLAY); return EGL_NO_CONTEXT; }
    wink_context_attribs a;
    wink_context_attribs_default(&a);
    a.api = WINK_API_OPENGL_ES; a.major = 3; a.minor = 1;
    a.share = static_cast<wink_context*>(share);
    for (auto p = attributes; p && p[0] != EGL_NONE; p += 2) {
        switch (p[0]) {
        case EGL_CONTEXT_CLIENT_VERSION: a.major = p[1]; break;
        case EGL_CONTEXT_MINOR_VERSION_KHR: a.minor = p[1]; break;
        case EGL_CONTEXT_OPENGL_ROBUST_ACCESS_EXT:
            if (p[1]) a.flags |= WINK_CONTEXT_ROBUST_ACCESS;
            break;
        case EGL_CONTEXT_FLAGS_KHR:
            if (p[1] & EGL_CONTEXT_OPENGL_ROBUST_ACCESS_BIT_KHR) a.flags |= WINK_CONTEXT_ROBUST_ACCESS;
            break;
        default: fail(EGL_BAD_ATTRIBUTE); return EGL_NO_CONTEXT;
        }
    }
    int status;
    auto ctx = wink_context_create(&a, &status);
    if (!ctx) { fail(EGL_BAD_MATCH); return EGL_NO_CONTEXT; }
    auto previous = wink_get_current_context();
    auto surface = wink_get_current_surface();
    if (wink_make_current(ctx, nullptr, nullptr) != WINK_OK) std::abort();
    auto renderer = gl<PFNGLGETSTRINGPROC>("glGetString")(GL_RENDERER);
    if (!renderer || !std::strstr(reinterpret_cast<const char*>(renderer), "gex")) std::abort();
    std::fprintf(stderr, "WINK_SKIA_BACKEND\tgles\t%s\n", renderer);
    if (wink_make_current(previous, surface, surface) != WINK_OK) std::abort();
    return ctx;
}
EGLBoolean wink_eglDestroyContext(EGLDisplay d, EGLContext ctx) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    wink_context_destroy(static_cast<wink_context*>(ctx));
    return EGL_TRUE;
}
EGLBoolean wink_eglMakeCurrent(EGLDisplay d, EGLSurface draw, EGLSurface read, EGLContext ctx) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    return wink_make_current(static_cast<wink_context*>(ctx), static_cast<wink_surface*>(draw),
                             static_cast<wink_surface*>(read)) == WINK_OK ? EGL_TRUE : fail(EGL_BAD_ACCESS);
}
EGLContext wink_eglGetCurrentContext() { return wink_get_current_context(); }
EGLDisplay wink_eglGetCurrentDisplay() { return wink_get_current_context() ? display() : EGL_NO_DISPLAY; }
EGLSurface wink_eglGetCurrentSurface(EGLint) { return wink_get_current_surface(); }
EGLBoolean wink_eglQueryContext(EGLDisplay d, EGLContext ctx, EGLint name, EGLint* value) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    int api, major, minor;
    wink_context_get_version(static_cast<wink_context*>(ctx), &api, &major, &minor);
    switch (name) {
    case EGL_CONTEXT_CLIENT_VERSION: *value = major; break;
    case EGL_CONTEXT_MINOR_VERSION_KHR: *value = minor; break;
    case EGL_CONTEXT_CLIENT_TYPE: *value = EGL_OPENGL_ES_API; break;
    case EGL_RENDER_BUFFER: *value = EGL_BACK_BUFFER; break;
    default: return fail(EGL_BAD_ATTRIBUTE);
    }
    return EGL_TRUE;
}
EGLSurface wink_eglCreatePbufferSurface(EGLDisplay d, EGLConfig, const EGLint* attributes) {
    if (!valid(d)) { fail(EGL_BAD_DISPLAY); return EGL_NO_SURFACE; }
    wink_surface_attribs a; wink_surface_attribs_default(&a); a.offscreen = true;
    for (auto p = attributes; p && p[0] != EGL_NONE; p += 2) {
        if (p[0] == EGL_WIDTH) a.width = p[1];
        else if (p[0] == EGL_HEIGHT) a.height = p[1];
        else { fail(EGL_BAD_ATTRIBUTE); return EGL_NO_SURFACE; }
    }
    int status;
    auto surface = wink_surface_create(&a, &status);
    if (!surface) fail(EGL_BAD_ALLOC);
    return surface;
}
EGLBoolean wink_eglDestroySurface(EGLDisplay d, EGLSurface s) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    wink_surface_destroy(static_cast<wink_surface*>(s)); return EGL_TRUE;
}
EGLBoolean wink_eglQuerySurface(EGLDisplay d, EGLSurface s, EGLint name, EGLint* value) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    if (!s) return fail(EGL_BAD_SURFACE);
    int w, h; wink_surface_get_size(static_cast<wink_surface*>(s), &w, &h);
    if (name == EGL_WIDTH) *value = w;
    else if (name == EGL_HEIGHT) *value = h;
    else return fail(EGL_BAD_ATTRIBUTE);
    return EGL_TRUE;
}
EGLSurface wink_eglCreatePixmapSurface(EGLDisplay, EGLConfig, EGLNativePixmapType, const EGLint*) {
    fail(EGL_BAD_NATIVE_PIXMAP); return EGL_NO_SURFACE;
}
EGLSurface wink_eglCreateWindowSurface(EGLDisplay, EGLConfig, EGLNativeWindowType, const EGLint*) {
    fail(EGL_BAD_NATIVE_WINDOW); return EGL_NO_SURFACE;
}
EGLSurface wink_eglCreatePbufferFromClientBuffer(EGLDisplay, EGLenum, EGLClientBuffer, EGLConfig, const EGLint*) {
    fail(EGL_BAD_MATCH); return EGL_NO_SURFACE;
}
EGLBoolean wink_eglCopyBuffers(EGLDisplay, EGLSurface, EGLNativePixmapType) { return fail(EGL_BAD_MATCH); }
EGLBoolean wink_eglBindTexImage(EGLDisplay, EGLSurface, EGLint) { return fail(EGL_BAD_MATCH); }
EGLBoolean wink_eglReleaseTexImage(EGLDisplay, EGLSurface, EGLint) { return fail(EGL_BAD_MATCH); }
EGLBoolean wink_eglSurfaceAttrib(EGLDisplay, EGLSurface, EGLint, EGLint) { return fail(EGL_BAD_ATTRIBUTE); }
EGLBoolean wink_eglSwapBuffers(EGLDisplay d, EGLSurface s) {
    if (!valid(d)) return fail(EGL_BAD_DISPLAY);
    return wink_swap_buffers(static_cast<wink_surface*>(s)) == WINK_OK ? EGL_TRUE : fail(EGL_BAD_SURFACE);
}
EGLBoolean wink_eglSwapInterval(EGLDisplay d, EGLint) { return valid(d) ? EGL_TRUE : fail(EGL_BAD_DISPLAY); }
EGLBoolean wink_eglWaitGL() { gl<PFNGLFINISHPROC>("glFinish")(); return EGL_TRUE; }
EGLBoolean wink_eglWaitClient() { return wink_eglWaitGL(); }
EGLBoolean wink_eglWaitNative(EGLint) { return fail(EGL_BAD_PARAMETER); }
EGLBoolean wink_eglReleaseThread() { return wink_eglMakeCurrent(display(), nullptr, nullptr, nullptr); }

struct Sync { GLsync sync; wink_context* context; };
struct CurrentScope {
    wink_context* previous = wink_get_current_context();
    wink_surface* surface = wink_get_current_surface();
    explicit CurrentScope(wink_context* c) { if (wink_make_current(c, nullptr, nullptr) != WINK_OK) std::abort(); }
    ~CurrentScope() { if (wink_make_current(previous, surface, surface) != WINK_OK) std::abort(); }
};
EGLSyncKHR wink_eglCreateSyncKHR(EGLDisplay d, EGLenum type, const EGLint*) {
    if (!valid(d) || type != EGL_SYNC_FENCE_KHR) { fail(EGL_BAD_ATTRIBUTE); return EGL_NO_SYNC_KHR; }
    auto owner = wink_get_current_context();
    if (!owner) { fail(EGL_BAD_MATCH); return EGL_NO_SYNC_KHR; }
    wink_context_attribs a; wink_context_attribs_default(&a);
    a.api = WINK_API_OPENGL_ES; a.major = 3; a.minor = 1; a.share = owner;
    int status;
    auto retained = wink_context_create(&a, &status);
    if (!retained) { fail(EGL_BAD_ALLOC); return EGL_NO_SYNC_KHR; }
    GLsync sync = gl<PFNGLFENCESYNCPROC>("glFenceSync")(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl<PFNGLFLUSHPROC>("glFlush")();
    if (!sync) { wink_context_destroy(retained); fail(EGL_BAD_ALLOC); return EGL_NO_SYNC_KHR; }
    return new Sync{sync, retained};
}
EGLBoolean wink_eglDestroySyncKHR(EGLDisplay d, EGLSyncKHR handle) {
    if (!valid(d) || !handle) return fail(EGL_BAD_PARAMETER);
    auto s = static_cast<Sync*>(handle);
    { CurrentScope scope(s->context); gl<PFNGLDELETESYNCPROC>("glDeleteSync")(s->sync); }
    wink_context_destroy(s->context); delete s; return EGL_TRUE;
}
EGLint wink_eglClientWaitSyncKHR(EGLDisplay d, EGLSyncKHR handle, EGLint flags, EGLTimeKHR timeout) {
    if (!valid(d) || !handle) return fail(EGL_BAD_PARAMETER);
    auto s = static_cast<Sync*>(handle); CurrentScope scope(s->context);
    const GLenum r = gl<PFNGLCLIENTWAITSYNCPROC>("glClientWaitSync")(
        s->sync, flags & EGL_SYNC_FLUSH_COMMANDS_BIT_KHR ? GL_SYNC_FLUSH_COMMANDS_BIT : 0, timeout);
    if (r == GL_ALREADY_SIGNALED || r == GL_CONDITION_SATISFIED) return EGL_CONDITION_SATISFIED_KHR;
    if (r == GL_TIMEOUT_EXPIRED) return EGL_TIMEOUT_EXPIRED_KHR;
    return fail(EGL_BAD_ACCESS);
}
EGLBoolean wink_eglGetSyncAttribKHR(EGLDisplay d, EGLSyncKHR handle, EGLint name, EGLint* value) {
    if (!valid(d) || !handle) return fail(EGL_BAD_PARAMETER);
    auto s = static_cast<Sync*>(handle); CurrentScope scope(s->context);
    if (name == EGL_SYNC_TYPE_KHR) *value = EGL_SYNC_FENCE_KHR;
    else if (name == EGL_SYNC_CONDITION_KHR) *value = EGL_SYNC_PRIOR_COMMANDS_COMPLETE_KHR;
    else if (name == EGL_SYNC_STATUS_KHR) {
        GLint status; gl<PFNGLGETSYNCIVPROC>("glGetSynciv")(s->sync, GL_SYNC_STATUS, 1, nullptr, &status);
        *value = status == GL_SIGNALED ? EGL_SIGNALED_KHR : EGL_UNSIGNALED_KHR;
    } else return fail(EGL_BAD_ATTRIBUTE);
    return EGL_TRUE;
}
using Proc = void (*)();
}
namespace skiatest::graphite {
Proc winkSkiaEGLGetProc(const char* name) {
#define ENTRY(n) if (!std::strcmp(name, "egl" #n)) return reinterpret_cast<Proc>(wink_egl##n)
    ENTRY(GetError); ENTRY(GetDisplay); ENTRY(Initialize); ENTRY(Terminate); ENTRY(QueryString);
    ENTRY(BindAPI); ENTRY(QueryAPI); ENTRY(GetConfigs); ENTRY(ChooseConfig); ENTRY(GetConfigAttrib);
    ENTRY(CreateContext); ENTRY(DestroyContext); ENTRY(MakeCurrent); ENTRY(GetCurrentContext);
    ENTRY(GetCurrentDisplay); ENTRY(GetCurrentSurface); ENTRY(QueryContext); ENTRY(CreatePbufferSurface);
    ENTRY(DestroySurface); ENTRY(QuerySurface); ENTRY(CreateWindowSurface); ENTRY(CreatePixmapSurface);
    ENTRY(CreatePbufferFromClientBuffer); ENTRY(CopyBuffers); ENTRY(BindTexImage); ENTRY(ReleaseTexImage);
    ENTRY(SurfaceAttrib); ENTRY(SwapBuffers); ENTRY(SwapInterval); ENTRY(WaitGL); ENTRY(WaitClient);
    ENTRY(WaitNative); ENTRY(ReleaseThread); ENTRY(CreateSyncKHR); ENTRY(DestroySyncKHR);
    ENTRY(ClientWaitSyncKHR); ENTRY(GetSyncAttribKHR);
#undef ENTRY
    if (!std::strcmp(name, "eglGetProcAddress")) return reinterpret_cast<Proc>(winkSkiaEGLGetProc);
    return reinterpret_cast<Proc>(wink_get_proc_address(name));
}
}
