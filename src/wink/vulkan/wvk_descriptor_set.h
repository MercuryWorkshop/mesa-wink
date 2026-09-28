/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_DESCRIPTOR_SET_H
#define WVK_DESCRIPTOR_SET_H

#include "wvk_private.h"
#include "wvk_descriptor_set_layout.h"

#include "vk_object.h"

#include "util/list.h"

struct wvk_device;
struct wvk_device_memory;
struct wvk_image_view;
struct wvk_sampler;
struct vk_descriptor_update_template;

/* A descriptor as the host keeps it; WebGPU bind groups are made of these
 * when a draw needs one. */
struct wvk_descriptor {
   /* Buffers: a range of the memory's WGPUBuffer, never VK_WHOLE_SIZE. */
   struct wvk_device_memory *mem;
   WGPUBuffer buffer;
   uint64_t offset;
   uint64_t size;
   /* Images, and samplers. */
   struct wvk_image_view *view;
   struct wvk_sampler *sampler;
};

/* Bind groups made from a set, for the layouts that asked. A layout is
 * the bind group layout of a pipeline: equal ones are one object, so a
 * set used by many pipelines alike needs one bind group. */
#define WVK_SET_BIND_GROUPS 4

struct wvk_set_bind_group {
   WGPUBindGroupLayout layout;
   uint64_t generation;
   /* What else went into it: the driver's bindings, for group 3. */
   uint64_t extra;
   WGPUBindGroup group;
};

/* Bind groups are immutable and descriptor sets are not, so a set's are
 * made again after a write: `generation` counts writes. */
struct wvk_descriptor_set {
   struct vk_object_base base;
   struct wvk_descriptor_pool *pool;
   struct list_head link;
   struct wvk_descriptor_set_layout *layout;

   uint64_t generation;
   struct wvk_set_bind_group bind_groups[WVK_SET_BIND_GROUPS];
   uint32_t bind_group_next;

   /* Inline uniform blocks: their bytes, and the buffer a bind group
    * shows them through. */
   uint8_t *inline_data;
   uint32_t inline_size;
   WGPUBuffer inline_buffer;
   uint64_t inline_uploaded;

   struct wvk_descriptor descriptors[];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)

struct wvk_descriptor_pool {
   struct vk_object_base base;
   struct list_head sets;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)

/* A set of its own for vkCmdPushDescriptorSet, owned by the caller. */
struct wvk_descriptor_set *wvk_descriptor_set_create(struct wvk_device *dev,
                                                     struct wvk_descriptor_set_layout *layout);
void wvk_descriptor_set_destroy(struct wvk_device *dev, struct wvk_descriptor_set *set);

void wvk_descriptor_set_write(struct wvk_device *dev, struct wvk_descriptor_set *set,
                              const VkWriteDescriptorSet *write);
void wvk_descriptor_set_write_template(struct wvk_device *dev, struct wvk_descriptor_set *set,
                                       const struct vk_descriptor_update_template *template,
                                       const void *data);

/* Drop the bind groups made from the set: it changed. */
void wvk_descriptor_set_invalidate(struct wvk_descriptor_set *set);

#endif
