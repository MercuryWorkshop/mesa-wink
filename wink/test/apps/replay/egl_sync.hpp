// SPDX-License-Identifier: MIT
#pragma once
#include "glproc.hpp"
#include "glretrace.hpp"
#include "retrace.hpp"
#include <map>

// EGL_KHR_fence_sync on the OpenGL API has the same command-stream ordering
// as GL_ARB_sync. Translate only ordinary fences; unsupported types/attrs fail.
struct WinkEglFence { GLsync sync; glretrace::Context *producer; };
static std::map<unsigned long long, WinkEglFence> winkEglFences;

static WinkEglFence *winkEglFence(trace::Call &call) {
    auto found = winkEglFences.find(call.arg(1).toUIntPtr());
    if (found == winkEglFences.end()) {
        retrace::warning(call) << "error: unknown EGL fence\n";
        return nullptr;
    }
    return &found->second;
}

// EGL permits client waits and destruction without a current GL context.
// Borrow the producer for the GL translation, then restore the caller exactly.
struct WinkEglFenceContext {
    trace::Call &call;
    glretrace::Context *previous;
    glws::Drawable *drawable, *readable;
    WinkEglFenceContext(trace::Call &c, const WinkEglFence &f) : call(c),
            previous(glretrace::getCurrentContext()),
            drawable(previous ? previous->drawable : nullptr),
            readable(previous ? previous->readable : nullptr) {
        if (previous) previous->aquire();
        glretrace::makeCurrent(call, f.producer->drawable, f.producer->readable, f.producer);
    }
    ~WinkEglFenceContext() {
        glretrace::makeCurrent(call, drawable, readable, previous);
        if (previous) previous->release();
    }
};

static void winkEglCreateSync(trace::Call &call) {
    if (!call.ret->toUIntPtr()) return;
    const auto *attrs = call.arg(2).toArray();
    if (call.arg(1).toUInt() != 0x30F9 || (attrs && !attrs->values.empty() &&
            (attrs->values.size() != 1 || attrs->values[0]->toUInt() != 0x3038))) {
        retrace::warning(call) << "unsupported: EGL fence type or attributes\n";
        return;
    }
    GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!fence) retrace::warning(call) << "error: failed to create EGL fence\n";
    else {
        auto *producer = glretrace::getCurrentContext();
        producer->aquire();
        winkEglFences[call.ret->toUIntPtr()] = {fence, producer};
    }
}

static void winkEglDestroySync(trace::Call &call) {
    if (!call.ret->toBool()) return;
    if (auto *fence = winkEglFence(call)) {
        WinkEglFenceContext binding(call, *fence);
        glDeleteSync(fence->sync);
        fence->producer->release();
        winkEglFences.erase(call.arg(1).toUIntPtr());
    }
}

static void winkEglClientWaitSync(trace::Call &call) {
    auto *fence = winkEglFence(call);
    if (!fence) return;
    WinkEglFenceContext binding(call, *fence);
    unsigned flags = call.arg(2).toUInt();
    if (flags & ~1u) {
        retrace::warning(call) << "unsupported: EGL fence wait flags\n";
        return;
    }
    // Preserve a captured successful wait even if replay timing differs.
    if (call.ret->toUInt() == 0x30F6) {
        glretrace::blockOnFence(call, fence->sync, GL_SYNC_FLUSH_COMMANDS_BIT);
    } else {
        glClientWaitSync(fence->sync, flags ? GL_SYNC_FLUSH_COMMANDS_BIT : 0, call.arg(3).toUInt());
    }
}

static void winkEglWaitSync(trace::Call &call) {
    if (!call.ret->toBool()) return;
    if (call.arg(2).toUInt()) {
        retrace::warning(call) << "unsupported: EGL server wait flags\n";
        return;
    }
    if (auto *fence = winkEglFence(call)) glWaitSync(fence->sync, 0, GL_TIMEOUT_IGNORED);
}
