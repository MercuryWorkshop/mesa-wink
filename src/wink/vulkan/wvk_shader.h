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

/* The outputs transform feedback captures of a shader, at most. */
#define WVK_MAX_XFB_OUTPUTS 64

/* Of the serialized shaders, for the pipeline cache. */
#define WVK_SHADER_BINARY_VERSION 42

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
 * or of lines (wvk_shader_draw_variant) -- lines whose fragments know
 * where on the line they are too, stippled or smooth. */
enum wvk_draw_shape {
   WVK_DRAW_PRIMITIVES,
   WVK_DRAW_POINTS,
   WVK_DRAW_LINES,
   WVK_DRAW_LINES_PLACED,
};

/* The largest point drawn as a quad, in pixels (largePoints), and line. */
#define WVK_MAX_LINE_WIDTH 64.0f
#define WVK_MAX_POINT_SIZE 256.0f

/* Storage images without a format a shader is compiled for the views of. */
#define WVK_MAX_UNFORMATTED_IMAGES 8
/* Buffers of a stage merged into bindings of their WGPUBuffers, at most. */
#define WVK_MAX_MERGES 48
/* Cubes and the samplers a shader samples them with, which a non-seamless
 * sampler (VK_EXT_non_seamless_cube_map) has a variant sample a face at a
 * time. */
#define WVK_MAX_CUBE_SAMPLES 8
/* 2D textures and storage images a 2D view of a 3D image may be bound at
 * (VK_EXT_image_2d_view_of_3d), which a variant has as the 3D ones. */
#define WVK_MAX_FLAT_VIEWS 8

/* Descriptors of a shader's, as runs of slots: set << 10 | the first slot,
 * and how many -- an array's elements, which are slots of their own after
 * its split. Or always: of more runs than there is room for, or of what
 * has no slot of its own; whatever is bound. */
#define WVK_MAX_SLOT_RUNS 8
struct wvk_slot_runs {
   uint8_t count;
   bool always;
   uint16_t first[WVK_MAX_SLOT_RUNS];
   uint16_t length[WVK_MAX_SLOT_RUNS];
};

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
   /* The additive sample counter's output location plus one. */
   uint8_t occlusion;
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
   /* Fragment, where nothing gives gl_PrimitiveID otherwise, after a
    * vertex shader: WebGPU's primitive_index (1), or half of it, of a
    * point's or line's quad (2). */
   uint8_t primitive_index;
   /* Fragment, and the stage before it: the clip and cull distances the
    * fragment shader reads, which WebGPU's have no builtin of, as vec4
    * varyings from location clip_varying - 1 (0: none) on, clip_slots of
    * them, of the components as the clip distances' slots have them. */
   uint8_t clip_varying, clip_slots;
   /* Fragment: counting its invocations; geometry as compute: its
    * primitives -- for the active pipeline statistics query, in its
    * counters (WVK_DRIVER_SLOT_STATS) -- and, of a shader of more than one
    * vertex stream, the stream rasterized, whose clipping counts. */
   uint8_t stats, raster_stream;
   /* Geometry as compute: its primitives' lists beginning at their last
    * vertex, whose flat varyings they have (VK_EXT_provoking_vertex). */
   uint8_t provoking_last;
   /* Lines drawn as quads that are stippled or smooth: where on the line a
    * fragment is -- across it and along it, in pixels -- as the varying at
    * location line_coord - 1 (0: none), which the vertex shader writes and
    * the fragment shader stipples and covers by (expand_lines). */
   uint8_t line_coord;
   /* Fragment, and the vertex shader after a geometry shader: the
    * barycentric coordinates the fragment shader reads (bary_uses: 1
    * perspective ones, 2 linear), as vec3 varyings from location bary - 1
    * on (0: none); and its inputs of each vertex, the generic slots of
    * per_vertex_mask, as flat uvec4 varyings from location per_vertex - 1
    * on, the first vertex's, then the second's and the third's
    * (VK_KHR_fragment_shader_barycentric). The vertex shader pulls its
    * index from the geometry shader's lists then, to know its primitive's
    * others. */
   uint8_t bary, bary_uses, per_vertex;
   uint32_t per_vertex_mask;
   /* Mesh: as compute (1), a workgroup a mesh workgroup of the draw,
    * writing a record of each of its vertices and primitives -- of a task
    * shader's launches where after_task; or the vertex shader of the draw
    * after (2), a vertex a corner of a primitive, reading those. Task: as
    * compute (task 1), a workgroup a task workgroup, writing its payload
    * and the mesh workgroups it launches (wvk_cmd_draw.c's draw_mesh). */
   uint8_t mesh, after_task, task;
   /* Fragment: its colour outputs moved to the attachments whose
    * locations they are (dynamicRenderingLocalRead): attachment i's the
    * output at location color_map[i], none for MESA_VK_ATTACHMENT_UNUSED;
    * where color_remapped, else each at its own. */
   uint8_t color_remapped;
   uint8_t color_map[8];
   /* Which of wvk_shader::cube_samples a non-seamless sampler samples: a
    * bit each, their cubes the 2D arrays of their faces. */
   uint8_t non_seamless;
   /* Whether a sampler of wvk_shader::reduce_samplers bound reduces, which
    * min and max reductions are emulated for then (samplerFilterMinmax). */
   uint8_t reduces;
   /* Whether a view of wvk_shader::min_lod_views bound has a minimum LOD,
    * which is emulated then (VK_EXT_image_view_min_lod). */
   uint8_t min_lods;
   /* Which of wvk_shader::flat_views have a 2D view of a 3D image bound: a
    * bit each, the 3D texture or image the view is of, at its slice. */
   uint8_t views_3d;
   /* Storage and texel buffers merged into the bindings of the WGPUBuffers
    * they are of, where the stage has more of them than WebGPU binds
    * (WVK_DRIVER_SLOT_MERGED): each one's set << 10 | binding, and its
    * merged binding + 1, of merge_count of them. */
   uint8_t merge_count;
   uint8_t merge_slot[WVK_MAX_MERGES];
   uint16_t merge_binding[WVK_MAX_MERGES];
   /* Where each set's descriptor records are, and how many words there
    * are of them: of the layouts of both stages, which a library of
    * independent sets has only its own sets in. */
   uint32_t meta_base[4], meta_words;
   /* Fragment: the logic op, VkLogicOp + 1 (0: none), and the format of
    * each colour attachment it applies to (0: none), whose integer
    * representation it is done on. */
   uint32_t logic_op;
   uint32_t logic_format[8];
   /* Fragment: the colour attachments the shader blends itself, where
    * WebGPU's blending cannot be Vulkan's (a colour factor of
    * CONSTANT_ALPHA beside CONSTANT_COLOR), of a copy of the attachment as
    * a logic op is; each one's equation (wvk_blend_equation), and its
    * format in logic_format. */
   uint32_t blend_mask;
   uint32_t blend_equation[8];
   /* Fragment: the depth bounds test, against a copy of the depth
    * attachment at WVK_DRIVER_SLOT_DEPTH_BOUNDS. */
   uint32_t depth_bounds;
   /* Fragment: the discard rectangle test, of the rectangles the driver's
    * data has (WVK_DRIVER_DISCARD). */
   uint8_t discard_rects;
   /* Fragment: the colour attachments it reads as they are, by framebuffer
    * fetch (wvk_nir_fetch.c): each one's format (0: none), and the input
    * attachment index + 1 that reads it (0: its tile image reads alone). */
   uint32_t fetch_format[8];
   uint8_t fetch_input[8];
   /* Fragment: coarse shading (wvk_nir_fsr.c), of the pipeline's rate and
    * combiner ops in the driver's data (WVK_DRIVER_FSR); and the
    * attachment's texel size, the log2 of its width << 4 | of its height,
    * + 1 (0: no attachment). Fragment and vertex: the location + 1 of the
    * varying of the primitive's rate, which the vertex shader writes
    * gl_PrimitiveShadingRateEXT to (0: none). */
   uint8_t fsr, fsr_attachment, fsr_varying;
   /* The format of each storage image the shader has none for, which
    * WGSL's must have: the VkFormat of the view bound there, in the order
    * of wvk_shader::unformatted. */
   uint32_t image_format[WVK_MAX_UNFORMATTED_IMAGES];
};

struct wvk_shader_source;

/* The device's variants of any source (wvk_shader.c's variant_for). */
void wvk_variant_cache_init(struct wvk_device *dev);
void wvk_variant_cache_finish(struct wvk_device *dev);

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
   /* Fragment: the colour locations it reads of tile images, and the
    * input attachments it reads (at its own sample, where multisampled),
    * a bit each: what it may read by framebuffer fetch
    * (key.fetch_format). */
   uint8_t tile_reads, input_reads;
   /* Pre-rasterization: writes gl_PrimitiveShadingRateEXT. Fragment: has
    * what the device shades at 1x1 with (the sample mask, depth or stencil
    * written, sample shading: fragmentShadingRateWithShaderSampleMask and
    * the like). */
   bool writes_shading_rate, fsr_1x1;
   /* Geometry and mesh: the location + 1 of the generic varying they
    * record gl_PrimitiveShadingRateEXT as (0: none). */
   uint8_t shading_rate_location;
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
   /* Fragment: the clip and cull distances it reads, in all. */
   uint8_t clip_inputs;
   /* Vertex, tessellation evaluation, geometry: what transform feedback
    * captures of its outputs (nir_xfb_info), by the words of their slots
    * in its records -- the buffers it writes, their strides and vertex
    * streams, and each output's slot, first component, components, buffer
    * and offset. */
   uint8_t xfb_buffers;
   uint16_t xfb_strides[4];
   uint8_t xfb_streams[4];
   uint8_t xfb_output_count;
   struct wvk_xfb_output {
      uint8_t slot, component, count, buffer;
      uint16_t offset;
   } xfb_outputs[WVK_MAX_XFB_OUTPUTS];
   /* Vertex, tessellation evaluation, geometry: the cull distances it
    * writes, which WebGPU has none of. A draw culls by them in its
    * geometry shader, and one without has one that only passes its
    * primitives through (wvk_shader_cull_geometry), of each kind of
    * primitive, made the first time. */
   uint8_t cull_distances;
   struct wvk_shader *cull_gs[5];
   /* Geometry: its input and output primitives (enum mesa_prim), the
    * vertices an invocation emits at most, and its invocations; and the
    * vertex streams it emits to, to the last one, each of which has
    * vertices and lists of its own (lower_geometry). */
   uint8_t gs_input, gs_output;
   uint16_t gs_max_vertices;
   uint8_t gs_invocations;
   uint8_t gs_streams;
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
   /* Its samplings of float cubes, as the set and slot of the cube and of
    * the sampler (set << 10 | slot, the cube's in the low 16 bits), whose
    * samplers a draw or dispatch compiles a variant for (key.non_seamless),
    * where the device has non-seamless cubes. */
   uint8_t cube_sample_count;
   uint32_t cube_samples[WVK_MAX_CUBE_SAMPLES];
   /* The samplers its float samples are of: a variant of it emulates
    * reductions where one bound reduces (key.reduces). And the textures
    * it samples and fetches: a variant emulates views' minimum LODs where
    * one bound has one (key.min_lods). */
   struct wvk_slot_runs reduce_samplers;
   struct wvk_slot_runs min_lod_views;
   /* Its 2D textures and storage images, as the set and slot of each (set
    * << 10 | slot), where the device has 2D views of 3D images: whose views
    * a draw or dispatch compiles a variant for (key.views_3d). */
   uint8_t flat_view_count;
   uint16_t flat_views[WVK_MAX_FLAT_VIEWS];
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
   /* Vertex: the attribute locations it reads, pulled or not; and those it
    * reads as signed or unsigned integers. */
   uint32_t attribs_read;
   uint32_t attribs_sint, attribs_uint;
   /* The words of records of each set in its own layout, -1 for a set the
    * layout does not have. */
   int32_t set_words[4];
   /* Fragment: shades per sample, as its pipeline said; writes its own
    * sample mask; places sample positions for a number of samples; writes
    * depth, before it was dropped. */
   bool per_sample, own_mask, uses_samples, source_writes_depth;
   /* Of a float operation not to be contracted (NoContraction, or float
    * controls 2 without AllowContract): its modules are compiled with
    * Dawn's strict math, where Metal fuses a multiply and an add else. */
   bool strict_math;
   /* Fragment: the barycentric coordinates it reads, and its inputs of each
    * vertex (key.bary_uses and key.per_vertex_mask). */
   uint8_t bary_uses;
   uint32_t per_vertex_mask;
   /* Mesh: the vertices and primitives a workgroup may output, of which
    * type (enum mesa_prim), and the slots a primitive's record has, and a
    * vertex's (record_vars, of a mesh shader), packed (mesh_slots). Task
    * and mesh: the payload's bytes. */
   uint16_t mesh_vertices, mesh_primitives;
   uint8_t mesh_primitive, prim_vars;
   uint16_t payload_size;
};

static inline struct wvk_shader *
wvk_shader_from_vk(struct vk_shader *shader)
{
   return shader ? container_of(shader, struct wvk_shader, vk) : NULL;
}

extern const struct vk_device_shader_ops wvk_device_shader_ops;

/* What of a draw's or dispatch's descriptors a shader's variant is of: the
 * formats of the views bound where it has storage images without one
 * (wvk_shader::unformatted; 0 for one not bound), and which of its cube
 * samplings a non-seamless sampler does (wvk_shader::cube_samples). */
struct wvk_bound_views {
   VkFormat image_formats[WVK_MAX_UNFORMATTED_IMAGES];
   /* Of a stage of more storage and texel buffers than WebGPU binds: the
    * merged bindings of its buffers, as wvk_shader_key has them. */
   uint8_t merge_count;
   uint8_t merge_slot[WVK_MAX_MERGES];
   uint16_t merge_binding[WVK_MAX_MERGES];
   /* Of a fragment shader's draw: its pass's shading rate attachment's
    * texel size, as key.fsr_attachment. */
   uint8_t fsr_attachment;
   uint8_t non_seamless;
   /* And whether a sampler of its reduce_samplers reduces, and a view of
    * its min_lod_views has a minimum LOD. */
   bool reduces;
   bool min_lods;
   /* And which of its 2D ones have 2D views of 3D images bound
    * (wvk_shader::flat_views). */
   uint8_t views_3d;
};

/* The shader, or its variant, for a draw of this state: a vertex shader
 * for this vertex input and fragment shader (which may be NULL), a
 * fragment shader for this multisampling at `samples`, a pass with or
 * without a depth attachment, and the logic op for these colour formats;
 * either for the records of both its and `other`'s layouts, and the
 * descriptors bound (`views`, as for wvk_shader_dispatch_variant); and for points
 * or lines drawn as quads. Or with `runs`, the vertex shader's compute
 * shader doing its storage writes or capturing positions (key.runs). A
 * fragment shader's counts its invocations where `stats`, the active
 * pipeline statistics query's, has them. The shader itself
 * where it was compiled for that, or a variant it could not compile. */
struct wvk_shader *wvk_shader_draw_variant(struct wvk_device *dev, struct wvk_shader *shader,
                                           const struct wvk_shader *other,
                                           const struct vk_dynamic_graphics_state *dyn, unsigned samples,
                                           bool has_depth, const VkFormat *color_formats,
                                           const struct wvk_bound_views *views, enum wvk_draw_shape shape,
                                           unsigned runs, uint32_t stats);

/* The shader, or its variant for the descriptors bound (`views`). */
struct wvk_shader *wvk_shader_dispatch_variant(struct wvk_device *dev, struct wvk_shader *shader,
                                               const struct wvk_bound_views *views);

/* A geometry shader's variant (key.geometry): as compute for a vertex
 * shader whose records have `vs_vars` generic varyings (1), or the vertex
 * shader of the draw after it, with `other` the fragment shader, drawing
 * `points` as quads (2). The first counts its primitives where `stats`,
 * the active pipeline statistics query's, has them or clipping's. */
struct wvk_shader *wvk_shader_geometry_variant(struct wvk_device *dev, struct wvk_shader *gs, unsigned kind,
                                               unsigned vs_vars, const struct wvk_shader *other,
                                               const struct vk_dynamic_graphics_state *dyn, bool after_tess,
                                               bool points, uint32_t stats);

/* A mesh or task shader's variant: a mesh shader as compute, of a task
 * shader's launches where `after_task` (1), or the vertex shader of the
 * draw after it, with `other` the fragment shader (2); a task shader as
 * compute (1). Counting their invocations and primitives where `stats`,
 * the active queries', has them. */
struct wvk_shader *wvk_shader_mesh_variant(struct wvk_device *dev, struct wvk_shader *shader, unsigned kind,
                                           bool after_task, const struct wvk_shader *other,
                                           const struct vk_dynamic_graphics_state *dyn, uint32_t stats,
                                           const struct wvk_bound_views *views);

/* Whether a mesh shader's draw draws its points as quads (largePoints):
 * where it gives them a size, and they are rasterized (not `discard`). */
static inline bool
wvk_shader_mesh_quads(const struct wvk_shader *mesh, bool discard)
{
   return mesh->mesh_primitive == MESA_PRIM_POINTS && mesh->writes_point_size && !discard;
}

/* A tessellation shader's variant (key.tess): as compute, for records of
 * the stage before with `vs_vars` generic varyings a vertex and
 * `patch_vars` patch varyings a patch (1); or the evaluation shader's draw
 * after it, with `other` the fragment shader, drawing `points` as quads
 * (2). */
struct wvk_shader *wvk_shader_tess_variant(struct wvk_device *dev, struct wvk_shader *shader, unsigned kind,
                                           unsigned vs_vars, unsigned patch_vars, unsigned patch_vertices,
                                           const struct wvk_shader *other, bool points);

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

/* A mesh workgroup's primitives' block: the vertices and primitives it
 * output (SetMeshOutputsEXT), in 16 bytes, then a record a primitive of
 * `prim_slots` slots -- its indices at slot 0, whether it is culled at
 * slot 1, then the rest it writes, packed (wvk_shader.c's mesh_slots). */
static inline uint32_t
wvk_mesh_block(unsigned primitives, unsigned prim_slots)
{
   return 16 + primitives * 16 * prim_slots;
}

/* A task workgroup's payload, in the payloads' buffer. */
static inline uint32_t
wvk_payload_stride(unsigned size)
{
   return MAX2((size + 15) & ~15u, 16);
}

/* The words of a mesh draw's header (wvk_cmd_draw.c's draw_mesh): the
 * workgroups of its first stage, three; their total; their dispatch; the
 * first of the mesh workgroups of the round; the round's dispatch of
 * them, and their total; the round's draw; and of every round, the mesh
 * workgroups, and the first of the next round's. */
#define WVK_MESH_FIRST 0
#define WVK_MESH_FIRST_TOTAL 3
#define WVK_MESH_FIRST_DISPATCH 4
#define WVK_MESH_BASE 7
#define WVK_MESH_DISPATCH 8
#define WVK_MESH_TOTAL 11
#define WVK_MESH_DRAW 12
#define WVK_MESH_ALL 16
#define WVK_MESH_NEXT 17
#define WVK_MESH_HEADER_WORDS 20

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

/* A blend equation in a word: colour and alpha source and destination
 * factors (VkBlendFactor, 5 bits each) and operations (VkBlendOp, 3). */
static inline uint32_t
wvk_blend_equation(uint32_t src_c, uint32_t dst_c, uint32_t op_c, uint32_t src_a, uint32_t dst_a, uint32_t op_a)
{
   return src_c | dst_c << 5 | op_c << 10 | src_a << 13 | dst_a << 18 | op_a << 23;
}

/* Or an advanced one (VK_EXT_blend_operation_advanced): its mode, enum
 * pipe_advanced_blend_mode, whether the source and destination are
 * premultiplied, and the overlap, enum pipe_blend_overlap_mode. */
#define WVK_BLEND_ADVANCED (1u << 31)

static inline uint32_t
wvk_blend_advanced(uint32_t mode, bool src_premultiplied, bool dst_premultiplied, uint32_t overlap)
{
   return WVK_BLEND_ADVANCED | mode | (uint32_t)src_premultiplied << 6 | (uint32_t)dst_premultiplied << 7 |
          overlap << 8;
}

/* A geometry shader of `input` primitives that passes each through, for
 * the cull distances of `last`, a vertex or tessellation evaluation shader
 * drawn without one; NULL where there is none to be had. */
struct wvk_shader *wvk_shader_cull_geometry(struct wvk_device *dev, struct wvk_shader *last, unsigned input);

/* The colour attachments of those `written` whose blending the fragment
 * shader does, which WebGPU's cannot: see lower_blend. */
uint32_t wvk_shader_blend_mask(const struct vk_dynamic_graphics_state *dyn, uint32_t written);

/* Whether a logic op does anything to an attachment of `format`. */
bool wvk_format_takes_logic_op(VkFormat format);

/* The physical device's NIR options, which differ from the GL driver's in
 * one thing: Vulkan's InstanceIndex is WebGPU's instance_index. */
void wvk_init_nir_options(nir_shader_compiler_options *options);

#endif
