/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_SHADER_H
#define WVK_SHADER_H

#include "wvk_private.h"

#include "wvk_descriptor_set_layout.h"

#include "vk_shader.h"

struct wvk_device;

/* A resource a shader uses: what its bind group layout entry has to say. */
enum wvk_binding_kind {
   WVK_BINDING_UNIFORM,
   WVK_BINDING_STORAGE,
   WVK_BINDING_STORAGE_READ,
   WVK_BINDING_TEXTURE,
   WVK_BINDING_STORAGE_TEXTURE,
   WVK_BINDING_SAMPLER,
   WVK_BINDING_SAMPLER_COMPARE,
};

struct wvk_shader_binding {
   uint8_t group;
   uint8_t kind;             /* enum wvk_binding_kind */
   uint16_t binding;
   /* Textures and storage textures. */
   uint8_t view_dimension;   /* WGPUTextureViewDimension */
   uint8_t sample_type;      /* WGPUTextureSampleType */
   uint8_t multisampled;
   uint8_t access;           /* WGPUStorageTextureAccess */
   uint32_t format;          /* WGPUTextureFormat, of a storage texture */
   /* Buffers: the set layout said dynamic; and how much of a uniform
    * buffer the shader declares, which a binding has to hold. */
   bool dynamic;
   uint32_t min_size;
};

/* Of the serialized shaders, for the pipeline cache. */
#define WVK_SHADER_BINARY_VERSION 4

enum wvk_shader_variant {
   /* Vulkan's y points down, WebGPU's up: gl_Position.y negated. */
   WVK_SHADER_FLIPPED,
   /* For a viewport of negative height, which flips it back. */
   WVK_SHADER_UNFLIPPED,
   WVK_SHADER_VARIANTS,
};

struct wvk_shader {
   struct vk_shader vk;

   /* Unique for the device's life: what pipeline caches key on, where a
    * pointer could come back as another shader. */
   uint64_t id;

   /* WGSL, and the modules made of it. Only a vertex shader has the
    * unflipped variant. */
   char *wgsl[WVK_SHADER_VARIANTS];
   WGPUShaderModule module[WVK_SHADER_VARIANTS];

   uint32_t binding_count;
   struct wvk_shader_binding *bindings;

   /* Vertex: the attribute locations read. Fragment: the colour
    * attachments written. */
   uint32_t io_mask;
   /* Fragment: writes gl_FragDepth, discards. */
   bool writes_depth;
   /* The push constants read, as a storage buffer at the driver's
    * binding. */
   bool uses_push_constants;
   /* Vertex: BaseVertex, BaseInstance or DrawIndex read, which the draw
    * has to say. */
   bool uses_draw_parameters;
   /* Vertex: the Vulkan vertex bindings whose attributes the shader reads
    * itself, at WVK_DRIVER_SLOT_VERTEX + i. */
   uint8_t pulled_bindings[WVK_MAX_PULLED_BINDINGS];
   uint8_t pulled_count;
   /* Compute. */
   uint16_t workgroup_size[3];
};

static inline struct wvk_shader *
wvk_shader_from_vk(struct vk_shader *shader)
{
   return shader ? container_of(shader, struct wvk_shader, vk) : NULL;
}

extern const struct vk_device_shader_ops wvk_device_shader_ops;

/* The physical device's NIR options, which differ from the GL driver's in
 * one thing: Vulkan's InstanceIndex is WebGPU's instance_index. */
void wvk_init_nir_options(nir_shader_compiler_options *options);

#endif
