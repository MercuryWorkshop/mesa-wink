// SPDX-License-Identifier: MIT
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/stack.h>
#include <emscripten/heap.h>
#include <emscripten/threading.h>
#include <emscripten/eventloop.h>
#include "wink.h"
#include <vulkan/vulkan_core.h>
#include <cstdio>
#include <cstdlib>
#ifdef __EMSCRIPTEN_PTHREADS__
#include <pthread.h>
#endif
extern "C" int wink_skia_dm_main(int, char**);
extern "C" void wink_skia_schedule(void);
namespace sk_gpu_test {
PFN_vkVoidFunction wink_skia_vk_proc(VkInstance instance, const char* name) {
    return reinterpret_cast<PFN_vkVoidFunction>(wink_vk_get_instance_proc_addr(instance, name));
}
}

extern "C" EMSCRIPTEN_KEEPALIVE int wink_skia_main(int argc, char** argv) {
#ifdef __EMSCRIPTEN_PTHREADS__
    // DM and its explicit test threads may call GLES; the Wink runner keeps
    // the GPU and its JSPI waits on main, including Vulkan calls issued by
    // Graphite's background pipeline workers.
    if (!emscripten_is_main_browser_thread()) {
        wink_init_attribs a;
        wink_init_attribs_default(&a);
        a.flags = WINK_INIT_GL_ON_MAIN_THREAD;
        const int status = wink_init(&a);
        if (status != WINK_OK) {
            std::fprintf(stderr, "WINK_SKIA_CONTEXT_FAILED\t%s\n", wink_status_string(status));
            return 1;
        }
    }
#endif
    return wink_skia_dm_main(argc, argv);
}
#ifdef __EMSCRIPTEN_PTHREADS__
static int worker_argc;
static char** worker_argv;
extern "C" EMSCRIPTEN_KEEPALIVE void wink_skia_worker_failed() {
    MAIN_THREAD_ASYNC_EM_ASM({ Module['winkSkiaDone'](-1); });
}
extern "C" EMSCRIPTEN_KEEPALIVE int wink_skia_worker_body() {
    std::fprintf(stderr, "WINK_SKIA_WORKER\t%d\n", !emscripten_is_main_browser_thread());
    const int rc = wink_skia_main(worker_argc, worker_argv);
    MAIN_THREAD_ASYNC_EM_ASM({ Module['winkSkiaDone']($0); }, rc);
    emscripten_runtime_keepalive_pop();
    return rc;
}
static void* worker(void*) {
    emscripten_runtime_keepalive_push();
    wink_skia_schedule();
    return nullptr;
}
extern "C" EMSCRIPTEN_KEEPALIVE int wink_skia_start(int argc, char** argv) {
    worker_argc = argc;
    worker_argv = argv;
    pthread_t thread;
    if (pthread_create(&thread, nullptr, worker, nullptr)) return 1;
    pthread_detach(thread);
    return 0;
}
#endif
#endif
