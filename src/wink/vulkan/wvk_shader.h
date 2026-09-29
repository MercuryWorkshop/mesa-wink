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
struct vk_dynamic_graphics_state;

/* A resource a shader uses: what its bind group layout entry has to say. */
enum wvk_binding_kind {
   WVK_BINDING_UNIFORM,
   WVK_BINDING_STORAGE,
   WVK_BINDING_STORAGE_READ,
   WVK_BINDING_TEXTURE,
   WVK_BINDING_STORAGE_TEXTURE,
   WVK_BINDING_SAMPLER,
   WVK_BINDING_SAMPLER_COMPARE,
   /* Only in bind group layouts: a sampler whose descriptor does not
    * filter (see binding_variant in wvk_cmd_draw.c). */
   WVK_BINDING_SAMPLER_NONFILTERING,
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
#define WVK_SHADER_BINARY_VERSION 11

/* A generic varying as the WGSL between the stages has it, by location:
 * how it is interpolated and what it is of, or 0 for none. WebGPU wants a
 * vertex shader's outputs to be interpolated as the fragment shader's
 * inputs are, where Vulkan goes by the fragment shader alone. */
#define WVK_MAX_VARYINGS 32
#define WVK_VARYING_PRESENT 0x80
#define WVK_VARYING_FLAT 0x01
#define WVK_VARYING_LINEAR 0x02
#define WVK_VARYING_UINT 0x04
#define WVK_VARYING_SINT 0x08

/* What of the state a shader may be drawn with it was compiled for. A draw
 * with other state -- a fragment shader it was not linked with, vertex
 * input it did not know, multisampling set at the draw -- draws a variant
 * compiled for that (wvk_shader_draw_variant). */
struct wvk_shader_key {
   /* Vertex: the fragment shader's inputs its outputs are made to match. */
   uint8_t fs_inputs[WVK_MAX_VARYINGS];
   /* Vertex: the attributes it reads itself (pull_divided_attributes),
    * hashed; 0 for none. */
   uint64_t pulled;
   /* Fragment: alpha to one, the coverage it computes from alpha itself,
    * and the samples it places positions or that coverage for (0: none). */
   uint8_t alpha_to_one, by_alpha, samples;
   /* Where each set's descriptor records are, and how many words there
    * are of them: of the layouts of both stages, which a library of
    * independent sets has only its own sets in. */
   uint32_t meta_base[4], meta_words;
};

struct wvk_shader_source;

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
   /* Vertex: gl_Layer written, so that a pass per layer draws it in every
    * one. */
   bool writes_layer;
   /* The vertex shader writes gl_ViewportIndex: a draw for each viewport. */
   bool writes_viewport;
   /* A fragment shader that covers samples as its first output's alpha
    * says itself, for alpha to one, or with its own sample mask: WebGPU's
    * alpha to coverage is off for it. */
   bool covers_by_alpha;
   /* Where each set's records start among the records in the push
    * binding, in words, and how many there are: the pipeline layout's. */
   uint32_t meta_base[4];
   uint32_t meta_words;
   /* Vertex: the Vulkan vertex bindings whose attributes the shader reads
    * itself, at WVK_DRIVER_SLOT_VERTEX + i. */
   uint8_t pulled_bindings[WVK_MAX_PULLED_BINDINGS];
   uint8_t pulled_count;
   /* Compute. */
   uint16_t workgroup_size[3];

   /* What it was compiled for, and from: the NIR and layouts its variants
    * are compiled from. A variant has no source of its own. */
   struct wvk_shader_key key;
   struct wvk_shader_source *source;
   /* Vertex: its outputs; fragment: its inputs (WVK_VARYING_*). */
   uint8_t varyings[WVK_MAX_VARYINGS];
   /* Vertex: the attribute locations it reads, pulled or not. */
   uint32_t attribs_read;
   /* The words of records of each set in its own layout, -1 for a set the
    * layout does not have. */
   int32_t set_words[4];
   /* Fragment: shades per sample, as its pipeline said; writes its own
    * sample mask; places sample positions for a number of samples. */
   bool per_sample, own_mask, uses_samples;
};

static inline struct wvk_shader *
wvk_shader_from_vk(struct vk_shader *shader)
{
   return shader ? container_of(shader, struct wvk_shader, vk) : NULL;
}

extern const struct vk_device_shader_ops wvk_device_shader_ops;

/* The shader, or its variant, for a draw of this state: a vertex shader
 * for this vertex input and fragment shader (which may be NULL), a
 * fragment shader for this multisampling at `samples`; either for the
 * records of both its and `other`'s layouts. The shader itself where it
 * was compiled for that, or a variant it could not compile. */
struct wvk_shader *wvk_shader_draw_variant(struct wvk_device *dev, struct wvk_shader *shader,
                                           const struct wvk_shader *other,
                                           const struct vk_dynamic_graphics_state *dyn, unsigned samples);

/* The physical device's NIR options, which differ from the GL driver's in
 * one thing: Vulkan's InstanceIndex is WebGPU's instance_index. */
void wvk_init_nir_options(nir_shader_compiler_options *options);

#endif
