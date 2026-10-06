/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_PHYSICAL_DEVICE_H
#define WVK_PHYSICAL_DEVICE_H

#include "wvk_private.h"

#include "vk_physical_device.h"
#include "compiler/nir/nir.h"
#include "vk_sync.h"
#include "vk_sync_timeline.h"

struct wvk_instance;

#ifdef __EMSCRIPTEN__
/* Dawn's framebuffer fetch, which no browser's WebGPU has: a feature no
 * adapter there has, so what it gates is off. */
#define WGPUFeatureName_FramebufferFetch ((WGPUFeatureName)0x7fff0f00)
#endif

/* The memory types, by role; see wvk_physical_device.c. */
enum wvk_memory_type {
   WVK_MEMORY_DEVICE,          /* buffers and linear images */
   WVK_MEMORY_UPLOAD,          /* host-visible buffers */
   WVK_MEMORY_READBACK,        /* host-visible, cached buffers */
   WVK_MEMORY_DEVICE_UPLOAD,   /* device-local and host-visible buffers */
   WVK_MEMORY_IMAGE,           /* optimal images, and only those */
   WVK_MEMORY_TYPE_COUNT,
};

/* The types buffers, and images that are their memory, may be bound to. */
#define WVK_MEMORY_BUFFER_TYPES (BITFIELD_MASK(WVK_MEMORY_TYPE_COUNT) & ~BITFIELD_BIT(WVK_MEMORY_IMAGE))

struct wvk_physical_device {
   struct vk_physical_device vk;

   WGPUAdapter adapter;
   WGPUAdapterInfo info;
   /* The adapter's limits, which is what a device is created with. */
   WGPULimits limits;
   uint32_t color_attachments;

   /* The adapter's features that the driver uses: what vkCreateDevice
    * requests. */
   WGPUFeatureName features[32];
   uint32_t feature_count;

   uint64_t heap_size;
   VkMemoryType memory_types[WVK_MEMORY_TYPE_COUNT];
   /* What each heap has allocated of it, every device's (memoryBudget). */
   uint64_t heap_used[2];

   /* Binary syncs of our own, and the runtime's timelines made of them;
    * so submits are deferred until their waits are pending, and one
    * in-order WebGPU queue then has nothing to wait for on the GPU. */
   struct vk_sync_timeline_type sync_timeline_type;
   const struct vk_sync_type *sync_types[3];

   nir_shader_compiler_options nir_options;
   /* Mesa's 64-bit floats in integer arithmetic, made when a shader first
    * has any (wvk_shader.c). */
   simple_mtx_t float64_lock;
   nir_shader *float64;
};

VK_DEFINE_HANDLE_CASTS(wvk_physical_device, vk.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)

static inline struct wvk_instance *
wvk_physical_device_instance(const struct wvk_physical_device *pdev)
{
   return (struct wvk_instance *)pdev->vk.instance;
}

bool wvk_physical_device_has_feature(const struct wvk_physical_device *pdev,
                                     WGPUFeatureName feature);

VkResult wvk_enumerate_physical_devices(struct vk_instance *instance);
void wvk_physical_device_destroy(struct vk_physical_device *pdev);

#endif
