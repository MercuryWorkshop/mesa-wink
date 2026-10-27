// SPDX-License-Identifier: MIT
// Skia's GL test factory backed exclusively by Wink, including shared contexts.
#include "tools/ganesh/gl/GLTestContext.h"
#include "include/gpu/ganesh/gl/GrGLAssembleInterface.h"
#include "wink.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <cstdlib>

#ifdef WINK_SKIA_ANGLE_REFERENCE
namespace sk_gpu_test {
GLTestContext* CreateWinkReferenceGLTestContext(GrGLStandard, GLTestContext*);
}
#endif

namespace {
void check(int status) {
    if (status != WINK_OK) SK_ABORT("Wink: %s", wink_status_string(status));
}
struct Lifetime {
    bool alive = true;
};
class WinkGLTestContext final : public sk_gpu_test::GLTestContext {
public:
    explicit WinkGLTestContext(WinkGLTestContext* share) {
        check(wink_init(nullptr));
        auto previous = restorer();
        SkScopeExit restore(std::move(previous));
        wink_surface_attribs sa;
        wink_surface_attribs_default(&sa);
        sa.width = sa.height = 256;
        sa.offscreen = true;
        int status;
        fSurface = wink_surface_create(&sa, &status);
        check(status);
        wink_context_attribs ca;
        wink_context_attribs_default(&ca);
        ca.api = WINK_API_OPENGL_ES;
        ca.major = 3;
        ca.minor = 0;
        ca.share = share ? share->fContext : nullptr;
        fContext = wink_context_create(&ca, &status);
        check(status);
        this->onPlatformMakeCurrent();
        auto assembled = GrGLMakeAssembledGLESInterface(nullptr, [](void*, const char* name) {
            return reinterpret_cast<GrGLFuncPtr>(wink_get_proc_address(name));
        });
        if (!assembled || !assembled->validate()) SK_ABORT("Wink GLES interface validation failed");
        auto gl = sk_make_sp<GrGLInterface>(*assembled);
        // Skia's EGLImageTest checks these platform capabilities before using
        // the virtual image methods below. Wink implements their texture-image
        // semantics directly, without an EGL display or EGL entry points.
        gl->fExtensions.add("EGL_KHR_image");
        gl->fExtensions.add("EGL_KHR_gl_texture_2D_image");
        const char* renderer = reinterpret_cast<const char*>(
                gl->fFunctions.fGetString(GR_GL_RENDERER));
        if (!renderer || !strstr(renderer, "gex")) SK_ABORT("Unexpected GLES renderer");
        std::fprintf(stderr, "WINK_SKIA_BACKEND\tgles\t%s\n", renderer);
        this->init(std::move(gl));
    }
    ~WinkGLTestContext() override {
        auto restore = this->onPlatformGetAutoContextRestore();
        this->onPlatformMakeCurrent();
        this->teardown();
        fLifetime->alive = false;
        wink_make_current(nullptr, nullptr, nullptr);
        wink_context_destroy(fContext);
        wink_surface_destroy(fSurface);
        if (restore) restore();
    }
    std::unique_ptr<GLTestContext> makeNew() const override {
        auto context = std::make_unique<WinkGLTestContext>(nullptr);
        context->makeCurrent();
        return context;
    }
    GrEGLImage texture2DToEGLImage(GrGLuint texture) const override {
        int status;
        return wink_image_create_texture_2d(texture, &status);
    }
    void destroyEGLImage(GrEGLImage image) const override {
        wink_image_destroy(static_cast<wink_image*>(image));
    }
    GrGLuint eglImageToExternalTexture(GrEGLImage image) const override {
        using Import = void (*)(GrGLenum, GrGLeglImage);
        auto import = reinterpret_cast<Import>(
                wink_get_proc_address("glEGLImageTargetTexture2DOES"));
        if (!image || !import) return 0;
        GrGLuint texture = 0;
        auto& gl = this->gl()->fFunctions;
        gl.fGenTextures(1, &texture);
        gl.fBindTexture(GR_GL_TEXTURE_EXTERNAL, texture);
        import(GR_GL_TEXTURE_EXTERNAL, image);
        if (gl.fGetError() != GR_GL_NO_ERROR) {
            gl.fDeleteTextures(1, &texture);
            return 0;
        }
        return texture;
    }
private:
    // Each thread's restorer weakly tracks the factory-owned context. Avoid
    // resurrecting a destroyed context when Skia abandons one while restoring.
    struct Current {
        wink_context* context = nullptr;
        wink_surface* surface = nullptr;
        std::weak_ptr<Lifetime> lifetime;
    };
    static thread_local Current current;
    static std::function<void()> restorer() {
        const Current saved = current;
        return [saved] {
            auto lifetime = saved.lifetime.lock();
            if (lifetime && lifetime->alive) {
                check(wink_make_current(saved.context, saved.surface, saved.surface));
                current = saved;
            } else {
                check(wink_make_current(nullptr, nullptr, nullptr));
                current = {};
            }
        };
    }
    void onPlatformMakeCurrent() const override {
        check(wink_make_current(fContext, fSurface, fSurface));
        current = {fContext, fSurface, fLifetime};
    }
    void onPlatformMakeNotCurrent() const override {
        check(wink_make_current(nullptr, nullptr, nullptr));
        current = {};
    }
    std::function<void()> onPlatformGetAutoContextRestore() const override {
        if (wink_get_current_context() == fContext) return nullptr;
        return restorer();
    }
    GrGLFuncPtr onPlatformGetProcAddress(const char* name) const override {
        return reinterpret_cast<GrGLFuncPtr>(wink_get_proc_address(name));
    }
    wink_context* fContext = nullptr;
    wink_surface* fSurface = nullptr;
    std::shared_ptr<Lifetime> fLifetime = std::make_shared<Lifetime>();
};
thread_local WinkGLTestContext::Current WinkGLTestContext::current;
}
namespace sk_gpu_test {
GLTestContext* CreatePlatformGLTestContext(GrGLStandard api, GLTestContext* share) {
#ifdef WINK_SKIA_ANGLE_REFERENCE
    if (std::getenv("WINK_SKIA_REFERENCE_GLES"))
        return CreateWinkReferenceGLTestContext(api, share);
#endif
    if (api == kGL_GrGLStandard) return nullptr;
    return new WinkGLTestContext(static_cast<WinkGLTestContext*>(share));
}
}
