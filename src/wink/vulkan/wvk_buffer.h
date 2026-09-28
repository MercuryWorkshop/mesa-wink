/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_BUFFER_H
#define WVK_BUFFER_H

#include "wvk_private.h"

#include "vk_buffer.h"

struct wvk_device_memory;

/* A range of its memory's WGPUBuffer. */
struct wvk_buffer {
   struct vk_buffer vk;
   struct wvk_device_memory *mem;
   uint64_t offset;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER)

#endif
