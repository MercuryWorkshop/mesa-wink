/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_DEVICE_MEMORY_H
#define WVK_DEVICE_MEMORY_H

#include "wvk_private.h"

#include "vk_device_memory.h"

struct wvk_device;

/* Memory for buffers is one WGPUBuffer, made when a buffer is first bound
 * to it, so suballocating allocators such as VMA work: a VkBuffer is a
 * range of it. Memory for images is only accounting, since each VkImage is
 * a WGPUTexture of its own.
 *
 * Host-visible memory is also a copy on the host, which is what the
 * application maps. The ranges a submit reads are uploaded before it runs,
 * and the ranges it writes read back after, before the submit counts as
 * finished. */
struct wvk_device_memory {
   struct vk_device_memory vk;

   WGPUBuffer buffer;
   uint64_t buffer_size;

   void *host;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

/* The WGPUBuffer behind buffer memory, made on first use. */
VkResult wvk_device_memory_get_buffer(struct wvk_device *dev, struct wvk_device_memory *mem,
                                      WGPUBuffer *out);

#endif
