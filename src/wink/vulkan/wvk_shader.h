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
#define WVK_SHADER_BINARY_VERSION 21

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
/* What a draw's primitives are drawn as: themselves, or quads of points
 * or of wide lines (wvk_shader_draw_variant). */
enum wvk_draw_shape {
   WVK_DRAW_PRIMITIVES,
   WVK_DRAW_POINTS,
   WVK_DRAW_LINES,
};

/* The largest point drawn as a quad, in pixels (largePoints), and line. */
#define WVK_MAX_LINE_WIDTH 64.0f
#define WVK_MAX_POINT_SIZE 256.0f

/* Storage images without a format a shader is compiled for the views of. */
#define WVK_MAX_UNFORMATTED_IMAGES 8

struct wvk_shader_key {
   /* Vertex: the fragment shader's inputs its outputs are made to match. */
   uint8_t fs_inputs[WVK_MAX_VARYINGS];
   /* Vertex: the attributes it reads itself (pull_divided_attributes),
    * hashed; 0 for none. */
   uint64_t pulled;
   /* Fragment: alpha to one, the coverage it computes from alpha itself,
    * and the samples it places positions or that coverage for (0: none). */
   uint8_t alpha_to_one, by_alpha, samples;
   /* Fragment: its depth write dropped, for a pass without a depth
    * attachment, which WebGPU will not have it write to. */
   uint8_t no_depth;
   /* Points drawn as quads (largePoints): the vertex shader making six
    * vertices a point, the fragment shader reading gl_PointCoord from the
    * varying at location point_coord - 1 (0: none) and facing front, as it
    * does for wide lines too. */
   uint8_t points, point_coord;
   /* Vertex: the vertex shader as a compute shader, an invocation a vertex
    * of the draw, doing its storage writes, which WebGPU's vertex shaders
    * cannot (vertexPipelineStoresAndAtomics; wvk_cmd_draw.c runs it before
    * the draw, whose vertex shader has them stripped); 2 for one that
    * only captures the vertices' clip positions, which wide lines read; 3
    * for one that captures all of its outputs, a record a vertex, which a
    * geometry shader reads (WVK_RECORD_BUILTINS). */
   uint8_t runs;
   /* Vertex: wide lines drawn as quads, six vertices a segment (wideLines;
    * key.runs 2 capturing the positions they read). */
   uint8_t lines;
   /* Geometry: the shader as compute, an invocation a primitive of the
    * draw and invocation of the shader, reading the vertex shader's
    * records (key.runs 3) and writing its vertices' and their primitives'
    * (1); or the vertex shader of the draw after, reading those (2). And
    * the generic varyings the vertex shader's records have (vs_vars). */
   uint8_t geometry, vs_vars;
   /* Geometry: after tessellation, its inputs the evaluation shader's
    * records by the tessellator's lists, as many as it made. */
   uint8_t after_tess;
   /* Tessellation: the control or evaluation shader as compute (1), reading
    * the records of the stage before, of vs_vars generic varyings a vertex
    * and patch_vars patch varyings a patch; or the vertex shader of the draw
    * after the evaluation shader, reading its records (2). */
   uint8_t tess, patch_vars;
   /* Tessellation evaluation: the control shader's output patch size,
    * which gl_PatchVerticesIn is and its records go by. */
   uint8_t patch_vertices;
   /* Fragment, and the vertex shader after a geometry shader: gl_PrimitiveID
    * as the flat varying at location primitive_id - 1 (0: none), which the
    * geometry shader's records carry. */
   uint8_t primitive_id;
   /* Where each set's descriptor records are, and how many words there
    * are of them: of the layouts of both stages, which a library of
    * independent sets has only its own sets in. */
   uint32_t meta_base[4], meta_words;
   /* Fragment: the logic op, VkLogicOp + 1 (0: none), and the format of
    * each colour attachment it applies to (0: none), whose integer
    * representation it is done on. */
   uint32_t logic_op;
   uint32_t logic_format[8];
   /* Fragment: the depth bounds test, against a copy of the depth
    * attachment at WVK_DRIVER_SLOT_DEPTH_BOUNDS. */
   uint32_t depth_bounds;
   /* The format of each storage image the shader has none for, which
    * WGSL's must have: the VkFormat of the view bound there, in the order
    * of wvk_shader::unformatted. */
   uint32_t image_format[WVK_MAX_UNFORMATTED_IMAGES];
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
   /* Fragment: the colour attachments a logic op reads the copies of, at
    * WVK_DRIVER_SLOT_LOGIC_OP + i. */
   uint8_t logic_op_reads;
   /* Fragment: whether it reads the depth attachment's copy for the depth
    * bounds test. */
   bool reads_depth_bounds;
   /* Vertex: writes PointSize, and so draws points as quads. Fragment:
    * reads gl_PointCoord (1) or gl_FrontFacing (2), which a point's quad
    * has not; or gl_PrimitiveID (4), which the draw after a geometry
    * shader has not. */
   bool writes_point_size;
   uint8_t point_state;
   /* Vertex: writes storage, which a draw does as compute (key.runs). */
   bool writes_storage;
   /* The generic varyings of the records it writes: a vertex shader
    * capturing its outputs, a geometry shader its vertices. */
   uint8_t record_vars;
   /* Geometry: its input and output primitives (enum mesa_prim), the
    * vertices an invocation emits at most, and its invocations. */
   uint8_t gs_input, gs_output;
   uint16_t gs_max_vertices;
   uint8_t gs_invocations;
   /* Tessellation control: its patches' vertices, and the patch varyings
    * of its records. Evaluation: the domain (enum tess_primitive_mode),
    * spacing (enum gl_tess_spacing), winding and point mode. */
   uint8_t tess_vertices_out, patch_vars;
   uint8_t tess_domain, tess_spacing;
   bool tess_ccw, tess_point_mode;
   /* Its storage images without a format, as the set and slot of each
    * (set << 10 | slot), whose views' formats a draw or dispatch compiles
    * a variant for (key.image_format). */
   uint8_t unformatted_count;
   uint16_t unformatted[WVK_MAX_UNFORMATTED_IMAGES];
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
    * sample mask; places sample positions for a number of samples; writes
    * depth, before it was dropped. */
   bool per_sample, own_mask, uses_samples, source_writes_depth;
};

static inline struct wvk_shader *
wvk_shader_from_vk(struct vk_shader *shader)
{
   return shader ? container_of(shader, struct wvk_shader, vk) : NULL;
}

extern const struct vk_device_shader_ops wvk_device_shader_ops;

/* The shader, or its variant, for a draw of this state: a vertex shader
 * for this vertex input and fragment shader (which may be NULL), a
 * fragment shader for this multisampling at `samples`, a pass with or
 * without a depth attachment, and the logic op for these colour formats;
 * either for the records of both its and `other`'s layouts, and the
 * formats of the views bound where it has storage images without one
 * (`image_formats`, as for wvk_shader_dispatch_variant); and for points
 * or lines drawn as quads. Or with `runs`, the vertex shader's compute
 * shader doing its storage writes or capturing positions (key.runs). The
 * shader itself
 * where it was compiled for that, or a variant it could not compile. */
struct wvk_shader *wvk_shader_draw_variant(struct wvk_device *dev, struct wvk_shader *shader,
                                           const struct wvk_shader *other,
                                           const struct vk_dynamic_graphics_state *dyn, unsigned samples,
                                           bool has_depth, const VkFormat *color_formats,
                                           const VkFormat *image_formats, enum wvk_draw_shape shape,
                                           unsigned runs);

/* The shader, or its variant for the formats of the views bound where it
 * has storage images without one (wvk_shader::unformatted; 0 for one not
 * bound). */
struct wvk_shader *wvk_shader_dispatch_variant(struct wvk_device *dev, struct wvk_shader *shader,
                                               const VkFormat *image_formats);

/* A geometry shader's variant (key.geometry): as compute for a vertex
 * shader whose records have `vs_vars` generic varyings (1), or the vertex
 * shader of the draw after it, with `other` the fragment shader (2). */
struct wvk_shader *wvk_shader_geometry_variant(struct wvk_device *dev, struct wvk_shader *gs, unsigned kind,
                                               unsigned vs_vars, const struct wvk_shader *other,
                                               const struct vk_dynamic_graphics_state *dyn, bool after_tess);

/* A tessellation shader's variant (key.tess): as compute, for records of
 * the stage before with `vs_vars` generic varyings a vertex and
 * `patch_vars` patch varyings a patch (1); or the evaluation shader's draw
 * after it, with `other` the fragment shader (2). */
struct wvk_shader *wvk_shader_tess_variant(struct wvk_device *dev, struct wvk_shader *shader, unsigned kind,
                                           unsigned vs_vars, unsigned patch_vars, unsigned patch_vertices,
                                           const struct wvk_shader *other);

/* A patch's record: the outer and inner levels, a slot each, then its
 * patch varyings. */
static inline uint32_t
wvk_patch_stride(unsigned patch_vars)
{
   return 16 * (2 + patch_vars);
}

/* A record of a stage's outputs as the next reads them from a buffer: a
 * vec4 a slot, of the builtins (position, point size, clip and cull
 * distances, layer, viewport, primitive ID) and then the generic varyings,
 * to the last the stage has. */
#define WVK_RECORD_BUILTINS 9
static inline uint32_t
wvk_record_stride(unsigned vars)
{
   return 16 * (WVK_RECORD_BUILTINS + vars);
}

/* How many indices a geometry shader's invocation writes of its output
 * primitives, as lists: three a triangle of a strip, two a segment, one a
 * point. */
static inline uint32_t
wvk_geometry_indices(unsigned output, unsigned max_vertices)
{
   switch (output) {
   case MESA_PRIM_TRIANGLE_STRIP: return max_vertices > 2 ? 3 * (max_vertices - 2) : 0;
   case MESA_PRIM_LINE_STRIP: return max_vertices > 1 ? 2 * (max_vertices - 1) : 0;
   default: return max_vertices;
   }
}

/* Whether a logic op does anything to an attachment of `format`. */
bool wvk_format_takes_logic_op(VkFormat format);

/* The physical device's NIR options, which differ from the GL driver's in
 * one thing: Vulkan's InstanceIndex is WebGPU's instance_index. */
void wvk_init_nir_options(nir_shader_compiler_options *options);

#endif
