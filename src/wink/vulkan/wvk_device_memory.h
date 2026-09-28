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
   /* Unique on the device, for caches that name the buffer. */
   uint64_t serial;

   /* The application's reference, and descriptors': a descriptor may
    * outlive what it names, when nothing uses it, and bind groups still
    * have it. The last reference frees the memory, with the allocator
    * vkFreeMemory was given. */
   uint32_t refs;
   VkAllocationCallbacks free_alloc;
   bool free_with_alloc;

   void *host;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_device_memory, vk.base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)

void wvk_device_memory_ref(struct wvk_device_memory *mem);
void wvk_device_memory_unref(struct wvk_device_memory *mem);

/* What the WGPUBuffer, and the host copy, of memory of `size` bytes are:
 * whole words, and 256 bytes more. WebGPU wants a uniform binding to hold
 * all of the shader's block, where Vulkan lets a buffer be only as big as
 * what is read of it; the padding lets a binding reach past the end. */
static inline uint64_t
wvk_memory_buffer_size(uint64_t size)
{
   return align64(MAX2(size, 4), 4) + 256;
}

/* The WGPUBuffer behind buffer memory, made on first use. */
VkResult wvk_device_memory_get_buffer(struct wvk_device *dev, struct wvk_device_memory *mem,
                                      WGPUBuffer *out);

#endif
