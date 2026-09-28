/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_DESCRIPTOR_SET_LAYOUT_H
#define WVK_DESCRIPTOR_SET_LAYOUT_H

#include "wvk_private.h"

#include "vk_descriptor_set_layout.h"

struct wvk_sampler;

/* Set N is @group(N). WebGPU has no arrays of bindings, so every element
 * of a Vulkan binding is a WebGPU binding of its own, numbered in binding
 * order; a combined image sampler is a texture and a sampler, all of the
 * textures first. */
struct wvk_descriptor_set_binding_layout {
   VkDescriptorType type;
   uint32_t count;
   VkShaderStageFlags stages;
   /* This binding's first descriptor in the set's array of them. */
   uint32_t descriptor_index;
   /* Its first dynamic offset, among the set's, if it is dynamic. */
   uint32_t dynamic_index;
   /* Its first WebGPU binding; a combined image sampler's samplers follow
    * at slot + count. */
   uint32_t slot;
   /* count of them, or NULL. */
   struct wvk_sampler **immutable_samplers;
};

/* What a WebGPU binding of the set is: element `element` of Vulkan
 * binding `binding`, and for a combined image sampler which half. */
struct wvk_descriptor_slot {
   uint32_t binding;
   uint32_t element;
   bool sampler;
};

struct wvk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;

   uint32_t binding_count;
   uint32_t descriptor_count;
   uint32_t dynamic_count;
   uint32_t slot_count;
   struct wvk_descriptor_slot *slots;

   struct wvk_descriptor_set_binding_layout bindings[];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_descriptor_set_layout, vk.base, VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)

/* The driver's own bindings are at the top of the last group, which they
 * share with set 3 if a pipeline layout has one. */
#define WVK_DRIVER_GROUP 3
#define WVK_DRIVER_SLOT_BASE 992
#define WVK_DRIVER_SLOT_PUSH (WVK_DRIVER_SLOT_BASE + 0)
#define WVK_MAX_SET_SLOTS WVK_DRIVER_SLOT_BASE

/* The push binding: the application's push constants, then the driver's
 * own data for the draw -- the vertex shader's viewport transform. */
#define WVK_PUSH_SIZE 128
#define WVK_DRIVER_DATA_OFFSET WVK_PUSH_SIZE
#define WVK_PUSH_DATA_SIZE (WVK_PUSH_SIZE + 16)

static inline bool
wvk_descriptor_type_is_dynamic(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
          type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
}

/* How many WebGPU bindings one descriptor of this type takes. */
static inline uint32_t
wvk_descriptor_type_slots(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? 2 : 1;
}

#endif
