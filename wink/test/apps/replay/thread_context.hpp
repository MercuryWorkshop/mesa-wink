// SPDX-License-Identifier: MIT
#pragma once
#include "glimports.hpp"
#include "glretrace.hpp"
#include <map>
namespace retrace { extern bool singleThread; }

// Upstream's single-thread loop merges recorded threads without restoring
// their thread-local GL bindings. Keep object/uniform remapping upstream,
// while restoring each recorded thread's context and draw/read surfaces.
inline void winkRestoreTraceThread(trace::Call &call) {
    if (!retrace::singleThread) return;
    struct Binding {
        glretrace::Context *context = nullptr;
        glws::Drawable *drawable = nullptr;
        glws::Drawable *readable = nullptr;
    };
    static std::map<unsigned, Binding> bindings;
    static unsigned previous = 0;
    static bool initialized = false;
    if (initialized && previous != call.thread_id) {
        auto &saved = bindings[previous];
        auto *current = glretrace::getCurrentContext();
        if (saved.context != current) {
            if (current) current->aquire();
            if (saved.context) saved.context->release();
            saved.context = current;
        }
        saved.drawable = current ? current->drawable : nullptr;
        saved.readable = current ? current->readable : nullptr;
        const auto &next = bindings[call.thread_id];
        glretrace::makeCurrent(call, next.drawable, next.readable, next.context);
    }
    previous = call.thread_id;
    initialized = true;
}
