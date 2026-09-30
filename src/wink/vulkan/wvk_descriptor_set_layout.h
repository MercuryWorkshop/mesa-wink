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
   /* The planes of a combined image sampler's views, of the YCbCr
    * conversions of its immutable samplers: each after the first is a
    * texture more, at slot + (1 + plane) * count, and a descriptor more,
    * at plane_descriptor_index + (plane - 1) * count, after the set's
    * others; and with them a sampler of the chroma filter, at
    * slot + (1 + planes) * count, the second plane's descriptor's. */
   uint32_t planes;
   uint32_t plane_descriptor_index;
   /* An inline uniform block: its bytes, and where they are in the set's
    * buffer of them. */
   uint32_t inline_offset;
   uint32_t inline_size;
};

/* What a WebGPU binding of the set is: element `element` of Vulkan
 * binding `binding`, and for a combined image sampler which half. */
struct wvk_descriptor_slot {
   uint32_t binding;
   uint32_t element;
   bool sampler;
   /* A texture of a plane after the first: which. */
   uint8_t plane;
};

/* The descriptor a slot is: its own, or its plane's. */
static inline uint32_t
wvk_slot_descriptor_index(const struct wvk_descriptor_set_binding_layout *bl, const struct wvk_descriptor_slot *s)
{
   return s->plane ? bl->plane_descriptor_index + (s->plane - 1) * bl->count + s->element
                   : bl->descriptor_index + s->element;
}

struct wvk_descriptor_set_layout {
   struct vk_descriptor_set_layout vk;

   uint32_t binding_count;
   uint32_t descriptor_count;
   uint32_t dynamic_count;
   uint32_t slot_count;
   struct wvk_descriptor_slot *slots;
   /* The bytes of the set's inline uniform blocks, each where a uniform
    * binding can start, and room after the last for WebGPU's minimum
    * binding size. */
   uint32_t inline_size;

   struct wvk_descriptor_set_binding_layout bindings[];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_descriptor_set_layout, vk.base, VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)

/* The driver's own bindings are at the top of the last group, which they
 * share with set 3 if a pipeline layout has one. */
#define WVK_DRIVER_GROUP 3
/* Below WebGPU's thousand bindings a group, with room for all of them. */
#define WVK_DRIVER_SLOT_BASE 970
#define WVK_DRIVER_SLOT_PUSH (WVK_DRIVER_SLOT_BASE + 0)
/* The draw's parameters (BaseVertex, BaseInstance): the push buffer again
 * for a direct draw, the indirect buffer for an indirect one. */
#define WVK_DRIVER_SLOT_DRAW (WVK_DRIVER_SLOT_BASE + 1)
/* Vertex buffers whose attributes the vertex shader reads itself: those
 * of bindings with a divisor other than 1, which WebGPU cannot step. */
#define WVK_DRIVER_SLOT_VERTEX (WVK_DRIVER_SLOT_BASE + 2)
#define WVK_MAX_PULLED_BINDINGS 6
/* A copy of each colour attachment a logic op reads, as it was before the
 * draw: WebGPU has no logic ops, so the fragment shader does them. */
#define WVK_DRIVER_SLOT_LOGIC_OP (WVK_DRIVER_SLOT_VERTEX + WVK_MAX_PULLED_BINDINGS)
#define WVK_MAX_LOGIC_OP_ATTACHMENTS 8
/* A copy of the depth attachment the depth bounds test reads, as it was
 * before the draw, in R32Float. */
#define WVK_DRIVER_SLOT_DEPTH_BOUNDS (WVK_DRIVER_SLOT_LOGIC_OP + WVK_MAX_LOGIC_OP_ATTACHMENTS)
/* The index buffer of an indexed draw of points drawn as quads, which the
 * vertex shader reads its point's index from (wvk_shader.c). */
#define WVK_DRIVER_SLOT_INDEX (WVK_DRIVER_SLOT_DEPTH_BOUNDS + 1)
/* What a vertex shader run as compute captures of the draw's vertices,
 * their clip positions, which a draw of wide lines reads (wvk_shader.c). */
#define WVK_DRIVER_SLOT_CAPTURE (WVK_DRIVER_SLOT_INDEX + 1)
/* A geometry shader run as compute: the vertices it emits, and the
 * indices of the primitives they make, which the draw after reads. */
#define WVK_DRIVER_SLOT_GS_VERTICES (WVK_DRIVER_SLOT_CAPTURE + 1)
#define WVK_DRIVER_SLOT_GS_INDICES (WVK_DRIVER_SLOT_CAPTURE + 2)
/* Tessellation run as compute: the control shader's records of its
 * vertices, and of its patches (levels, then patch varyings); the
 * tessellator's domain points and counts (wvk_tessellator.c). */
#define WVK_DRIVER_SLOT_TESS_VERTICES (WVK_DRIVER_SLOT_GS_INDICES + 1)
#define WVK_DRIVER_SLOT_TESS_PATCHES (WVK_DRIVER_SLOT_GS_INDICES + 2)
#define WVK_DRIVER_SLOT_TESS_POINTS (WVK_DRIVER_SLOT_GS_INDICES + 3)
#define WVK_DRIVER_SLOT_TESS_COUNTS (WVK_DRIVER_SLOT_GS_INDICES + 4)
/* The counters of the active pipeline statistics query that shaders add
 * to, a word a statistic by its bit (wvk_query.c). */
#define WVK_DRIVER_SLOT_STATS (WVK_DRIVER_SLOT_GS_INDICES + 5)
/* The device's locks of 64-bit atomics on buffers, a word each, which an
 * atomic takes the one of its address, and a word of zero after them
 * (wvk_nir_atomic64.c). */
#define WVK_DRIVER_SLOT_LOCKS (WVK_DRIVER_SLOT_GS_INDICES + 6)
#define WVK_ATOMIC_LOCKS 4096

#define WVK_MAX_SET_SLOTS WVK_DRIVER_SLOT_BASE

/* What WebGPU's objects cannot hold of a set's descriptors, and shaders
 * need -- a view's swizzle, a sampler's border colour, a texel buffer's
 * format: a record of WVK_META_WORDS words per WebGPU binding of the set.
 * The records of every bound set follow the driver's data in the push
 * binding, from WVK_META_OFFSET, set after set as the pipeline layout
 * has them: no binding of their own, of which WebGPU has few. See
 * wvk_nir_descriptors.c. */
#define WVK_META_WORDS 8

/* A texture's record: its view's swizzle, 0 for none, else 3 bits a
 * component of VkComponentSwizzle (ZERO to A); and which of R, G, B and A
 * its format has, a bit each, which a border colour is made into. */
#define WVK_META_TEX_SWIZZLE 0
#define WVK_META_TEX_CHANNELS 1
/* And the view's minimum LOD above its base level, a float, 0 for none
 * (VK_EXT_image_view_min_lod); and of a 2D view of a 3D image, the slice
 * (VK_EXT_image_2d_view_of_3d). */
#define WVK_META_TEX_MIN_LOD 2
#define WVK_META_TEX_SLICE 3

/* A sampler's record. */
#define WVK_META_SAMPLER_FLAGS 0
#define WVK_META_SAMPLER_BIAS 1      /* float */
#define WVK_META_SAMPLER_MIN_LOD 2   /* float */
#define WVK_META_SAMPLER_MAX_LOD 3   /* float */
#define WVK_META_SAMPLER_BORDER 4    /* 4 words, float or integer as the texture is */
/* Its flags: VkSamplerAddressMode 3 bits an axis, u, v, w, from bit 0. */
#define WVK_SAMPLER_UNNORMALIZED (1u << 9)
#define WVK_SAMPLER_MAG_LINEAR (1u << 10)
#define WVK_SAMPLER_MIN_LINEAR (1u << 11)
#define WVK_SAMPLER_MIP_LINEAR (1u << 12)
/* VkCompareOp, 3 bits, and whether it compares. */
#define WVK_SAMPLER_COMPARE_SHIFT 13
#define WVK_SAMPLER_COMPARE (1u << 16)
/* What a shader does itself when sampling a float texture: an axis
 * clamps to the border or mirrors once, or coordinates are texels. */
#define WVK_SAMPLER_EMULATE (1u << 17)
/* A min or max reduction (samplerFilterMinmax), which the shader does. */
#define WVK_SAMPLER_REDUCE_MIN (1u << 18)
#define WVK_SAMPLER_REDUCE_MAX (1u << 19)
/* Cubes sampled a face at a time (VK_EXT_non_seamless_cube_map), which a
 * shader's variant does (wvk_shader_draw_variant's views). */
#define WVK_SAMPLER_NON_SEAMLESS (1u << 20)
/* The largest LOD bias a sampler has, which maxSamplerLodBias reports. */
#define WVK_MAX_SAMPLER_LOD_BIAS 15.0f

/* A uniform or storage buffer's record: its range in bytes, 0 for a null
 * one -- for robustBufferAccess2's bounds, and nullDescriptor's. */
#define WVK_META_BUF_RANGE 0

/* A texel buffer's record. */
#define WVK_META_TEXEL_FORMAT 0    /* enum wvk_texel_type, and bytes a texel << 8 */
#define WVK_META_TEXEL_COUNT 1     /* elements */
#define WVK_META_TEXEL_START 2     /* where element 0 is in the binding, in bytes */
#define WVK_META_TEXEL_SIZES 3     /* 8 bits a channel, 0 past the last */
#define WVK_META_TEXEL_SHIFTS 4    /* 8 bits a channel: its first bit in the texel */
#define WVK_META_TEXEL_SWIZZLE 5   /* 4 bits for each of R, G, B, A: a channel, 4 for 0, 5 for 1 */

/* A storage image's record, when it is an image with atomics and so the
 * image's buffer of its texels (wvk_image.h) to a shader that does
 * atomics on it: where the view's texels are, in words. */
#define WVK_META_IMG_BASE 0        /* the view's first texel */
#define WVK_META_IMG_ROW 1         /* from one row to the next */
#define WVK_META_IMG_LAYER 2       /* from one layer or slice to the next */
#define WVK_META_IMG_WIDTH 3
#define WVK_META_IMG_HEIGHT 4
#define WVK_META_IMG_LAYERS 5      /* layers, or a 3D image's depth */
#define WVK_META_IMG_WORDS 6
/* Of every storage image: which of R, G, B and A its view's format has, a
 * bit each, none for a null one (as a texture's WVK_META_TEX_CHANNELS). */
#define WVK_META_IMG_CHANNELS 6
/* And of a view of a 3D image, the slices it has: the first, and how many
 * << 16 (VK_EXT_image_sliced_view_of_3d, and 2D views of one slice). */
#define WVK_META_IMG_SLICES 7

enum wvk_texel_type {
   WVK_TEXEL_UNORM,
   WVK_TEXEL_SNORM,
   WVK_TEXEL_UINT,
   WVK_TEXEL_SINT,
   WVK_TEXEL_FLOAT,
   WVK_TEXEL_UFLOAT_11_11_10,
};

/* The push binding: the application's push constants, then the driver's
 * own data for the draw:
 *   - the vertex shader's viewport transform, 4 floats;
 *   - the words of the draw binding BaseVertex and BaseInstance are, and
 *     DrawIndex, 4 uints;
 *   - a direct draw's BaseVertex and BaseInstance, which those words name
 *     then, 4 uints;
 *   - where each pulled vertex buffer's bound range starts in its binding,
 *     in bytes, 4 uints;
 *   - the viewport transform's depth part, 2 floats; and the layer and
 *     view a pass renders, 2 uints (a pass per layer or view, see
 *     wvk_cmd_buffer.c);
 *   - each pulled vertex buffer's stride, which may be dynamic, 4 uints;
 *   - a dispatch's base workgroup, 3 uints; and the viewport a draw is
 *     drawn in, a uint (a draw per viewport, see wvk_cmd_draw.c);
 *   - the depth bounds, 2 floats, which the fragment shader tests;
 *   - for points drawn as quads, 1 over the viewport's width and height,
 *     2 floats; and where an indexed draw's indices start in the index
 *     binding, in bytes, their size, 0 for a draw without, and whether
 *     its primitives restart, 3 uints;
 *   - for a vertex shader's storage writes, done by it as compute before
 *     the draw: the draw's vertices an instance, its vertices in all, its
 *     first vertex or index, and whether it restarts primitives, 4 uints;
 *   - for lines drawn as quads, their width, a float; whether they are a
 *     strip; how they are rasterized, 0 as parallelograms, 1 rectangles
 *     and 2 smooth ones; and their stipple, its factor << 16 | pattern,
 *     0 for none, 3 uints;
 *   - for a geometry shader run as compute: the draw's vertices an
 *     instance, its primitives an instance, the shader's invocations in
 *     all, and the draw's topology, 4 uints;
 *   - the pulled vertex buffers' starts and strides past the first four,
 *     2 uints each;
 *   - for tessellation run as compute: the draw's vertices an instance,
 *     its patches an instance, a patch's vertices, and its patches in
 *     all, 4 uints;
 *   - the blend constants, 4 floats, for blending the fragment shader does;
 *   - the discard rectangles (VK_EXT_discard_rectangles), which the
 *     fragment shader tests: how many, and whether they are exclusive, 4
 *     uints; then each's corners, x0, y0, x1, y1, 4 ints, 8 of them. */
#define WVK_PUSH_SIZE 128
#define WVK_DRIVER_DATA_OFFSET WVK_PUSH_SIZE
#define WVK_DRIVER_DRAW_OFFSET (WVK_DRIVER_DATA_OFFSET + 16)
#define WVK_DRIVER_DRAW_VALUES (WVK_DRIVER_DATA_OFFSET + 32)
#define WVK_DRIVER_PULL_OFFSET (WVK_DRIVER_DATA_OFFSET + 48)
#define WVK_DRIVER_DEPTH_OFFSET (WVK_DRIVER_DATA_OFFSET + 64)
#define WVK_DRIVER_LAYER_OFFSET (WVK_DRIVER_DEPTH_OFFSET + 8)
#define WVK_DRIVER_PULL_STRIDES (WVK_DRIVER_DATA_OFFSET + 80)
#define WVK_DRIVER_BASE_GROUP (WVK_DRIVER_DATA_OFFSET + 96)
#define WVK_DRIVER_VIEWPORT (WVK_DRIVER_BASE_GROUP + 12)
#define WVK_DRIVER_DEPTH_BOUNDS (WVK_DRIVER_DATA_OFFSET + 112)
#define WVK_DRIVER_POINT_SCALE (WVK_DRIVER_DATA_OFFSET + 120)
#define WVK_DRIVER_POINT_INDEX (WVK_DRIVER_DATA_OFFSET + 128)
#define WVK_DRIVER_VERTEX_RUN (WVK_DRIVER_DATA_OFFSET + 144)
#define WVK_DRIVER_LINE (WVK_DRIVER_DATA_OFFSET + 160)
#define WVK_DRIVER_GEOMETRY (WVK_DRIVER_DATA_OFFSET + 176)
#define WVK_DRIVER_PULL_MORE (WVK_DRIVER_DATA_OFFSET + 192)
#define WVK_DRIVER_TESS (WVK_DRIVER_DATA_OFFSET + 208)
#define WVK_DRIVER_BLEND (WVK_DRIVER_DATA_OFFSET + 224)
#define WVK_DRIVER_DISCARD (WVK_DRIVER_DATA_OFFSET + 240)
#define WVK_MAX_DISCARD_RECTANGLES 8
#define WVK_PUSH_DATA_SIZE (WVK_PUSH_SIZE + 240 + 16 + 16 * WVK_MAX_DISCARD_RECTANGLES)
#define WVK_META_OFFSET WVK_PUSH_DATA_SIZE
/* How much of the push buffer the push binding shows: its data, and as
 * many records as that leaves room for. */
#define WVK_PUSH_WINDOW (64 * 1024)

/* What a group-3 bind group's driver bindings are bound to: the push
 * buffer's generation, and memory serials (0: none) for the draw binding
 * (none: the push buffer), the pulled vertex buffers, the index buffer a
 * vertex shader reads, and the vertices it captures. */
struct wvk_driver_key {
   uint64_t push_generation;
   uint64_t draw;
   uint64_t vertex[WVK_MAX_PULLED_BINDINGS];
   uint64_t index;
   uint64_t capture;
   uint64_t gs_vertices, gs_indices;
   uint64_t tess[4];
   uint64_t stats;
};

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
