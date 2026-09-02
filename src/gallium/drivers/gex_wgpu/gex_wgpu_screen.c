/* SPDX-License-Identifier: MIT
 *
 * Initial Gallium-to-WebGPU resource/clear/readback implementation.  This is
 * intentionally not yet a GL-capable screen: all pipe caps remain zero until
 * their semantic families have tests.  The direct Gallium smoke test exercises
 * real WebGPU resources without allowing st/mesa to infer unsupported GL.
 */
#include "gex_wgpu_public.h"
#include "gex_wgpu_tint.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "compiler/nir/nir_xfb_info.h"
#include "microsoft/compiler/dxil_nir_lower_int_cubemaps.h"
#include "nir_to_spirv.h"
#include "util/os_time.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/format/u_format.h"
#include "util/half_float.h"
#include "util/u_inlines.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "util/simple_mtx.h"
#include "util/u_memory.h"
#include "util/u_helpers.h"
#include "util/u_prim.h"
#include "util/u_prim_restart.h"
#include "indices/u_indices.h"
#include "indices/u_primconvert.h"
#include "nir/tgsi_to_nir.h"
#include "util/u_blitter.h"
#include "util/u_framebuffer.h"
#include "util/u_draw.h"

#include "util/u_screen.h"
#include "util/u_upload_mgr.h"

#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void gex_wgpu_jspi_wait(void);
extern void gex_wgpu_jspi_signal(void);

/* st/mesa enables ARB_uniform_buffer_object only when a stage offers at least
 * 12 uniform blocks, which is 13 constant buffers counting GL's default block.
 * WebGPU's guaranteed maxUniformBuffersPerShaderStage is 12, so the page asks
 * requestDevice for more; a device that cannot give it will fail to create,
 * which is the honest outcome rather than advertising a limit and failing at
 * pipeline creation. */
#define GEX_WGPU_MAX_CONSTANT_BUFFERS 16
/* GL's minimum for GL_MAX_TRANSFORM_FEEDBACK_BUFFERS. */
#define GEX_WGPU_MAX_SO_BUFFERS 4
/* Capture buffers bind above the constant buffers so the two cannot collide. */
#define GEX_WGPU_SO_FIRST_BINDING GEX_WGPU_MAX_CONSTANT_BUFFERS
/* Two different limits, deliberately.
 *
 * GEX_WGPU_MAX_VERTEX_BUFFERS is a WebGPU limit: how many vertex buffers one
 * render pipeline may declare, and -- because the capture and geometry compute
 * variants read each one as a storage buffer -- how many of those bindings
 * they can spend. It cannot be raised past what the device offers on either
 * count.
 *
 * GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS is how many *Gallium* slots this driver
 * tracks, which is a different and much larger number: st/mesa binds one per
 * enabled vertex array whether or not the shader reads it. The draw path
 * numbers only the buffers that actually feed an attribute, so the WebGPU
 * limit applies to those rather than to everything bound. */
#define GEX_WGPU_MAX_VERTEX_BUFFERS 8
#define GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS PIPE_MAX_ATTRIBS
#define GEX_WGPU_MAX_VERTEX_ELEMENTS 16
/* Constant buffer slots past what WebGPU allows as uniform buffers are backed
 * by storage buffers instead, bound above the uniform ones in the same group. */
#define GEX_WGPU_STORAGE_UBO_FIRST_BINDING GEX_WGPU_MAX_CONSTANT_BUFFERS
/* Bindings used only by the transform-feedback compute variant, above the
 * render-stage ones so the two cannot collide. */
#define GEX_WGPU_VB_FIRST_BINDING \
   (GEX_WGPU_SO_FIRST_BINDING + GEX_WGPU_MAX_SO_BUFFERS)
#define GEX_WGPU_CAPTURE_PARAMS_BINDING \
   (GEX_WGPU_VB_FIRST_BINDING + GEX_WGPU_MAX_VERTEX_BUFFERS)
/* The draw's index buffer, read as words by the capture pass. A compute
 * shader has no fixed-function index fetch any more than it has a vertex
 * fetch, so an indexed transform feedback has to do the lookup itself. */
#define GEX_WGPU_CAPTURE_INDEX_BINDING (GEX_WGPU_CAPTURE_PARAMS_BINDING + 1)
/* Words of the capture parameter buffer. Byte offsets travel here rather than
 * in the binding, because a storage binding offset must be 256-byte aligned
 * and neither a GL vertex buffer nor a transform feedback range promises
 * more than four. */
#define GEX_WGPU_PARAM_FIRST_VERTEX 0
#define GEX_WGPU_PARAM_COUNT 1
#define GEX_WGPU_PARAM_SO_OFFSET 2
#define GEX_WGPU_PARAM_VB_OFFSET (GEX_WGPU_PARAM_SO_OFFSET + GEX_WGPU_MAX_SO_BUFFERS)
/* First index (in elements) and the base vertex added to every fetched index,
 * which are glDrawElements' `indices` offset and its basevertex. */
#define GEX_WGPU_PARAM_FIRST_INDEX \
   (GEX_WGPU_PARAM_VB_OFFSET + GEX_WGPU_MAX_VERTEX_BUFFERS)
#define GEX_WGPU_PARAM_INDEX_BIAS (GEX_WGPU_PARAM_FIRST_INDEX + 1)
#define GEX_WGPU_PARAM_WORDS (GEX_WGPU_PARAM_INDEX_BIAS + 1)
#define GEX_WGPU_CAPTURE_GROUP 64
/* Where a vertex shader run as compute writes its varyings when a geometry
 * shader is going to consume them. */
#define GEX_WGPU_VARYING_OUT_BINDING (GEX_WGPU_CAPTURE_PARAMS_BINDING + 1)

/* Bindings of the geometry stage's own compute pipeline, which is a separate
 * pipeline and so has its own group 0. */
#define GEX_WGPU_GS_PARAMS_BINDING 0
#define GEX_WGPU_GS_VSOUT_BINDING 1
#define GEX_WGPU_GS_VERTEX_BINDING 2
#define GEX_WGPU_GS_INDEX_BINDING 3
/* Words of the geometry pass parameter buffer. */
#define GEX_WGPU_GS_PARAM_PRIMITIVES 0
#define GEX_WGPU_GS_PARAM_WORDS 4
#define GEX_WGPU_GS_GROUP 32

/* Varyings carried between stages through a buffer, one vec4 each. Sixteen is
 * what the driver advertises as max_inputs/max_outputs. */
#define GEX_WGPU_MAX_VARYINGS 16

/* The primitive topologies WebGPU has. Everything else GL can draw -- triangle
 * fans, line loops, quads, polygons -- is rewritten into one of these by
 * u_primconvert before it reaches the pipeline.
 *
 * This has to be reported as a cap, not merely enforced: u_init_pipe_screen_caps
 * defaults supported_prim_modes to *every* mode, so a driver that quietly
 * refused the rest had st/mesa hand it triangle strips and then fail the
 * context. A triangle strip is the most common thing a GL program draws. */
#define GEX_WGPU_PRIM_MODES                                             \
   (BITFIELD_BIT(MESA_PRIM_POINTS) | BITFIELD_BIT(MESA_PRIM_LINES) |    \
    BITFIELD_BIT(MESA_PRIM_LINE_STRIP) |                                \
    BITFIELD_BIT(MESA_PRIM_TRIANGLES) |                                 \
    BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP))

static WGPUPrimitiveTopology
gex_wgpu_topology(enum mesa_prim mode)
{
   switch (mode) {
   case MESA_PRIM_POINTS: return WGPUPrimitiveTopology_PointList;
   case MESA_PRIM_LINES: return WGPUPrimitiveTopology_LineList;
   case MESA_PRIM_LINE_STRIP: return WGPUPrimitiveTopology_LineStrip;
   case MESA_PRIM_TRIANGLE_STRIP: return WGPUPrimitiveTopology_TriangleStrip;
   default: return WGPUPrimitiveTopology_TriangleList;
   }
}
/* Sixteen, not eight: GL 3.1 requires MAX_VERTEX_TEXTURE_IMAGE_UNITS >= 16,
 * and WebGPU's default maxSampledTexturesPerShaderStage and
 * maxSamplersPerShaderStage are both exactly that. The slot masks below are
 * sized to match; an eight-bit mask silently dropped slots 8 and up. */
#define GEX_WGPU_MAX_SAMPLERS 16
/* The widest bind group any of the four can have: the fragment texture group
 * carries a texture and a sampler per slot. */
#define GEX_WGPU_CACHE_MAX_ENTRIES (2 * GEX_WGPU_MAX_SAMPLERS)
#define GEX_WGPU_BGL_CACHE_SLOTS 64
#define GEX_WGPU_PL_CACHE_SLOTS 64
#define GEX_WGPU_BG_CACHE_SLOTS 512
#define GEX_WGPU_PIPELINE_CACHE_SLOTS 256
#define GEX_WGPU_PIPELINE_KEY_MAX 4096
/* Big enough that Minecraft, which uploads about two constant buffers per
 * draw, wraps every few dozen frames rather than every few draws. */
#define GEX_WGPU_CONSTANT_RING_BYTES (8u << 20)
#define GEX_WGPU_MAX_BATCHED_DRAWS 1024
#define GEX_WGPU_MAX_COLOR_TARGETS 8
/* Indexed by VARYING_SLOT_*, which is the numbering both stages agree on --
 * driver_location is assigned per stage and the vertex stage counts
 * gl_Position among its outputs, so the same varying has different
 * driver_locations on the two sides. */
#define GEX_WGPU_MAX_VARYING_LOCATIONS 64

struct gex_wgpu_screen {
   struct pipe_screen base;
   WGPUDevice device;
   WGPUQueue queue;
   /* Reading depth back.
    *
    * WebGPU will not copy the depth aspect of depth24plus or
    * depth24plus-stencil8 -- those formats have no defined memory layout, so
    * there is nothing for a buffer copy to mean. What it will do is *sample*
    * them, so depth comes back through a render pass instead: a fullscreen
    * draw that reads the depth texture and writes r32float, which is a format
    * a buffer copy does understand. Stencil needs none of this; its aspect is
    * plain stencil8 and copies directly.
    *
    * Built once and kept, because a readback that rebuilt a pipeline every
    * time would be slower than the copy it replaces. */
   WGPURenderPipeline depth_resolve;
   WGPUBindGroupLayout depth_resolve_layout;
   /* The mirror of the resolve: writing depth. A buffer copy into the depth
    * aspect is legal only for depth16unorm, so uploads go the other way --
    * the data lands in an r32float texture, and a fullscreen draw reads it
    * and writes @builtin(frag_depth). Stencil, again, needs none of this. */
   WGPURenderPipeline depth_upload[5];
   WGPUBindGroupLayout depth_upload_layout;
   /* The multisampled-depth resolve, one pipeline per depth format. Distinct
    * from `depth_resolve` above, which turns a packed depth texture into
    * something readable: this one turns 4x samples into one. WebGPU resolves
    * colour and only colour, so it does it the only way left -- a pass over
    * the destination that reads sample zero of the source through
    * `texture_depth_multisampled_2d` and writes `frag_depth`. */
   WGPURenderPipeline msaa_depth[5];
   WGPUBindGroupLayout msaa_depth_layout;
   bool texture_compression_bc;
   /* WebGPU's `dual-source-blending`. It gates ARB_blend_func_extended, which
    * is one of the two things standing between this driver and GL 3.3, and
    * without it the SRC1 blend factors have no meaning and a fragment shader
    * must not declare a second blend source. */
   bool dual_source_blending;
   bool timestamp_query;
   /* Optional WebGPU features the page may have been able to request.  Each
    * one gates a GL extension st/mesa derives from format support, so they
    * are tracked rather than assumed. */
   bool depth32_stencil8;
   bool depth_clip_control;
   bool rg11b10_renderable;
   /* WebGPU has no 16-bit normalized texture format, so GL's R16/RG16/RGBA16
    * and their snorm counterparts are backed by 32-bit float -- see
    * gex_wgpu_format_backing. That only works if the device will filter and
    * blend a float32 attachment, both of which are optional features. */
   bool float32_filterable;
   bool float32_blendable;
   /* WebGPU bounds the total bytes per sample across colour attachments, and
    * the default 32 is exactly eight RGBA8 targets -- so a framebuffer with
    * any wider format in it is a validation error rather than a wide
    * framebuffer. The page asks for more; this is what it got. */
   uint32_t max_colour_bytes_per_sample;
   /* Two vec4s of GL's default generic vertex attribute -- (0,0,0,1) as
    * floats at offset 0 and as integers at offset 16 -- for the locations a
    * shader reads that no vertex array supplies. Created on first need. */
   WGPUBuffer default_attributes;
   /* 1x1 stand-ins for a shader slot GL left without a texture, one per
    * (view dimension, sample type) the shaders actually ask for. See
    * gex_wgpu_default_texture_view. */
   struct {
      WGPUTextureViewDimension dimension;
      WGPUTextureSampleType sample_type;
      WGPUTexture texture;
      WGPUTextureView view;
   } default_textures[16];
   unsigned default_texture_count;
   /* One per WGPUSamplerBindingType: filtering, non-filtering, comparison. */
   WGPUSampler default_samplers[4];
   /* Bind group layouts, pipeline layouts and bind groups, kept by content.
    *
    * Every draw used to build all three from scratch and drop them again,
    * and wgpuDeviceCreateBindGroup alone measured 283 of a draw's 483 us --
    * more than the pipeline, the command buffer and the queue submit put
    * together. Nothing about them is per draw: they are a function of the
    * bound shaders and the bound resources, and consecutive draws in a scene
    * mostly share both.
    *
    * Direct-mapped and replace-on-collision, because a miss costs exactly
    * what the old code paid every time. Objects here are owned by the
    * screen; callers never release what a lookup returns. */
   /* The two layout caches are append-only, never evicting. There are only
    * as many distinct layouts as there are shader-and-binding shapes -- a
    * handful -- and eviction would be actively unsafe: the pipeline layout
    * cache keys on bind group layout *handles*, so freeing one leaves a
    * dangling key that a later allocation at the same address would match. */
   struct gex_wgpu_bgl_cache_slot {
      uint32_t hash;
      unsigned count;
      WGPUBindGroupLayoutEntry entries[GEX_WGPU_CACHE_MAX_ENTRIES];
      WGPUBindGroupLayout layout;
   } bgl_cache[GEX_WGPU_BGL_CACHE_SLOTS];
   unsigned bgl_cache_count;
   struct gex_wgpu_pl_cache_slot {
      uint32_t hash;
      unsigned count;
      WGPUBindGroupLayout layouts[4];
      WGPUPipelineLayout layout;
   } pl_cache[GEX_WGPU_PL_CACHE_SLOTS];
   unsigned pl_cache_count;
   /* Bind groups evicted this draw. Released at the start of the next one:
    * a draw builds up to four groups, two of them can land in the same slot,
    * and releasing the first while the second is being made would free a
    * group this very draw is about to bind. */
   WGPUBindGroup retired_groups[8];
   unsigned retired_count;
   struct gex_wgpu_bg_cache_slot {
      bool used;
      uint64_t epoch;
      uint32_t hash;
      WGPUBindGroupLayout layout;
      unsigned count;
      WGPUBindGroupEntry entries[GEX_WGPU_CACHE_MAX_ENTRIES];
      WGPUBindGroup group;
   } bg_cache[GEX_WGPU_BG_CACHE_SLOTS];
   uint64_t bg_cache_hits, bg_cache_misses;
   /* Bumped whenever a cached bind group or pipeline is evicted and released.
    *
    * The per-pass "already emitted" state compares raw handles, and a handle
    * that has been freed can be handed back for a different object at the
    * same address -- at which point a stale entry compares equal and the
    * bind is skipped, so the draw runs with whatever the pass had before.
    * That is invisible until it is not: SuperTuxKart went black while still
    * presenting sixty frames a second. Evictions are rare next to draws, so
    * throwing the whole emitted state away on one costs nothing. */
   uint64_t cache_epoch;
   uint64_t pass_serial;
   /* Everything above this line that a context may write.
    *
    * GL contexts are per thread and Mesa lets an application drive several at
    * once, but the caches, the retired list and the stand-in textures all
    * live on the screen, which is shared. Two threads racing one cache slot
    * free the object the other is about to bind, and the garbage reaches
    * Vulkan: `glthreads` segfaults inside radv, several frames in, with a
    * command buffer full of released handles. Nothing about it is visible
    * from the GL side. */
   simple_mtx_t cache_lock;
   /* One device and one queue serve every GL context, and GL contexts are per
    * thread. Recording into separate encoders concurrently is fine -- they are
    * separate objects -- but handing the finished buffers to one queue from
    * several threads at once is not, and lands in radv with a half-built
    * render pass. Separate from cache_lock so a submit does not wait on a
    * cache lookup, and so neither can be held while taking the other. */
   simple_mtx_t queue_lock;
   /* Held across anything that builds or submits Dawn commands.
    *
    * One device and one queue serve every GL context, and GL contexts are
    * per thread: `glthreads` drives several at once and lands in radv with a
    * half-recorded render pass. Recursive, because these paths call each
    * other -- a blit submits, a snapshot copies, a map flushes the batch.
    *
    * Only taken once a second context exists. Almost every application has
    * one, and an uncontended lock on every encoder and every map is still
    * hundreds of nanoseconds a draw that those applications need not pay. */
   mtx_t device_lock;
   bool device_lock_needed;
   unsigned context_count;
   /* Render pipelines, keyed on the bytes of their descriptor and everything
    * it points at. Serialising rather than enumerating the fields on purpose:
    * a field left out of an enumerated key returns the *wrong* pipeline,
    * which is silent corruption, whereas a byte that varies for no reason
    * only costs a miss. */
   struct gex_wgpu_pipeline_cache_slot {
      bool used;
      uint32_t hash;
      unsigned key_length;
      uint8_t key[GEX_WGPU_PIPELINE_KEY_MAX];
      WGPURenderPipeline pipeline;
   } pipeline_cache[GEX_WGPU_PIPELINE_CACHE_SLOTS];
   uint64_t pipeline_cache_hits, pipeline_cache_misses;
   /* Device limits, so caps report what this device can honour rather than
    * what the driver was compiled to hold. Advertising more than the device
    * allows does not fail here -- it fails later, at pipeline creation, on
    * whichever shader happens to use the extra slot. */
   uint32_t max_uniform_buffers;
   uint32_t max_storage_buffers_vertex;
};

struct gex_wgpu_resource {
   struct pipe_resource base;
   WGPUTexture texture;
   WGPUBuffer buffer;
   uint8_t *shadow;
   uint64_t allocation_size;
   /* Set once a compute pass has written this buffer on the GPU, which makes
    * the CPU shadow stale. glMapBuffer of a transform feedback target is
    * exactly that case, and the bind flags do not always say so: a buffer
    * created before it was ever bound as a feedback target carries no
    * PIPE_BIND_STREAM_OUTPUT. */
   bool gpu_written;
   /* Whether anything has ever put pixels in this texture -- an upload, a copy
    * into it, or its use as a render target. A texture that is sampled without
    * ever having been written reads as zeros, which is a black quad on screen
    * and no error anywhere. Tracking it is the only way to tell that apart
    * from a shader or blend problem. */
   bool texture_written;
   /* Touched by a draw that is recorded but not yet submitted. */
   bool in_batch;
   /* A snapshot of this texture, taken when a draw both renders to it and
    * samples it, and sampled in its place. Kept on the resource so it is
    * reused rather than reallocated every draw; rebuilt if the texture's
    * shape changes. See gex_wgpu_feedback_snapshot. */
   struct pipe_resource *feedback_copy;
   WGPUTextureView feedback_copy_view;
   /* Bytes ever written into this buffer from the CPU -- subdata and the
    * upload half of a write map. A constant buffer that is bound repeatedly
    * and reads as all zeros is either an application uploading zeros or an
    * upload that never arrived, and only this tells the two apart. */
   uint64_t bytes_written;
   /* Whether anything ever read this buffer's CPU shadow -- a CPU map, the
    * index scan in the draw path, or an unaligned subdata needing the bytes
    * around its range. A shadow nothing reads is pure wasm heap. */
   bool shadow_read;
   /* Whether the buffer was ever CPU-mapped at all, for any usage. A buffer
    * only ever written through buffer_subdata needs no persistent shadow --
    * an aligned subdata could upload straight from the caller's pointer --
    * whereas a mapped one hands the shadow out as the mapping. Which of the
    * two holds the memory decides how much a lazy shadow can recover. */
   bool ever_mapped;
};

struct gex_wgpu_query;

struct gex_wgpu_context {
   struct pipe_context base;
   bool failed;
   /* How deep draw_vbo is in itself. u_primconvert and the primitive-restart
    * split both re-enter it, and either can be handed a draw it converts to
    * something it converts again -- which is unbounded recursion on a 16 MiB
    * host stack, and a host stack overflow aborts the wasm module and costs
    * every case after it. Bounded, it costs the one draw and says so. */
   unsigned draw_depth;
   struct gex_wgpu_shader *vs;
   struct gex_wgpu_shader *fs;
   struct pipe_resource *colors[GEX_WGPU_MAX_COLOR_TARGETS];
   /* A colour target is a *surface*, and GL lets one view an sRGB texture as
    * linear or the other way round. The texture alone does not say which. */
   enum pipe_format color_formats[GEX_WGPU_MAX_COLOR_TARGETS];
   /* Which slice of the texture each target draws into. A framebuffer
    * attachment is a *view*, and GL routinely attaches a mip level other than
    * zero -- rendering a mip chain is how SuperTuxKart builds its lightmaps.
    * WebGPU expresses exactly this with baseMipLevel/baseArrayLayer, so the
    * level and layer have to survive from set_framebuffer_state to the render
    * pass rather than being asserted away as zero. */
   unsigned color_levels[GEX_WGPU_MAX_COLOR_TARGETS];
   unsigned color_layers[GEX_WGPU_MAX_COLOR_TARGETS];
   unsigned color_count;
   struct pipe_resource *depth;
   uint32_t fb_width;
   uint32_t fb_height;
   /* Samples per pixel of the bound framebuffer. WebGPU requires the render
    * pipeline's sample count to match its attachments exactly. */
   uint32_t fb_samples;
   bool viewport_set;
   struct pipe_viewport_state viewport;
   bool scissor_set;
   struct pipe_scissor_state scissor;
   struct gex_wgpu_rasterizer *rasterizer;
   struct pipe_depth_stencil_alpha_state *dsa;
   struct pipe_stencil_ref stencil_ref;
   struct pipe_blend_state *blend;
   struct pipe_blend_color blend_color;
   struct gex_wgpu_vertex_elements *vertex_elements;
   struct pipe_resource *vertex_buffers[GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS];
   uint32_t vertex_buffer_offsets[GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS];
   unsigned vertex_buffer_count;
   /* The same two pieces of state in the shapes u_blitter's save/restore
    * wants them. The driver's own copies above are decomposed for the paths
    * that use them; the blitter hands them straight back to set_*_state. */
   struct pipe_vertex_buffer
      vertex_buffer_state[GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS];
   struct pipe_framebuffer_state framebuffer;
   struct blitter_context *blitter;
   struct pipe_constant_buffer
      constant_buffers[MESA_SHADER_STAGES][GEX_WGPU_MAX_CONSTANT_BUFFERS];
   /* One ring for every constant buffer st/mesa hands over as a plain
    * pointer. Each upload takes the next aligned chunk and never reuses one,
    * so a write can never land on bytes an already-recorded draw still has
    * to read -- which is what lets draws share a command buffer.
    * See gex_wgpu_set_constant_buffer. */
   struct pipe_resource *constant_ring;
   unsigned constant_ring_offset;
   /* The ring's contents, staged on the CPU. Uniform uploads are the most
    * frequent call a frame makes -- two per draw, some 6800 a frame in
    * Minecraft -- and one wgpuQueueWriteBuffer each costs more than all the
    * draws put together. They are memcpy'd here instead and handed to the
    * queue as a single write covering the whole dirty range, just before the
    * batch that reads them is submitted. */
   uint8_t *constant_staging;
   unsigned constant_dirty_lo, constant_dirty_hi;
   /* Draws recorded into one command encoder, not yet submitted. */
   WGPUCommandEncoder pending_encoder;
   unsigned pending_draws;
   /* And the render pass they are being recorded into, held open across
    * consecutive draws that want the same attachments. One pass per draw is
    * what pinned every application to about 20000 draws a second whatever
    * its resolution -- the cost was the pass, not the pixels. */
   WGPURenderPassEncoder pending_pass;
   struct gex_wgpu_pass_key {
      struct pipe_resource *colors[GEX_WGPU_MAX_COLOR_TARGETS];
      enum pipe_format formats[GEX_WGPU_MAX_COLOR_TARGETS];
      unsigned levels[GEX_WGPU_MAX_COLOR_TARGETS];
      unsigned layers[GEX_WGPU_MAX_COLOR_TARGETS];
      unsigned color_count;
      struct pipe_resource *depth;
      bool depth_read_only;
      bool stencil_read_only;
   } pending_pass_key;
   /* What the open pass was last told. Consecutive draws in a scene mostly
    * differ in one uniform offset and nothing else, but the driver re-issued
    * the pipeline, every bind group, every vertex buffer, the index buffer
    * and the viewport for each one -- a Dawn encoder call is not free, it
    * validates. Re-emitting only what changed is what a Vulkan driver does
    * with dirty bits, and is most of the gap on a scene of many small draws.
    * All of it is cleared when a pass begins. */
   struct {
      WGPURenderPipeline pipeline;
      WGPUBindGroup groups[4];
      uint32_t dynamic_count[4];
      uint32_t dynamic[4][GEX_WGPU_MAX_CONSTANT_BUFFERS];
      WGPUBuffer vertex[GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS];
      uint64_t vertex_offset[GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS];
      unsigned vertex_count;
      WGPUBuffer index;
      WGPUIndexFormat index_format;
      uint64_t index_offset;
      float viewport[6];
      uint32_t scissor[4];
      bool scissor_set;
   } emitted;
   uint64_t emitted_epoch;
   /* Everything those draws read or write. A map or an upload has to push the
    * batch out only if it lands on one of these -- and almost none of them
    * do: an application streaming geometry maps a fresh buffer every time.
    * Without this, Minecraft broke its batch 806439 times, once every nine
    * draws, and batching cost more than it saved. */
   /* Referenced, not just noted. Dawn's per-pass usage tracking keeps raw
    * pointers to the buffers and textures a recorded draw uses, and resolves
    * them at submit; a resource freed in between leaves that list pointing at
    * nothing. With one submit per draw the window did not exist. With draws
    * batched it does, and Minecraft walked straight into it -- a crash inside
    * dawn::native::vulkan::Buffer::EnsureDataInitialized, a few seconds after
    * a world loaded. Holding a reference until the batch is submitted is both
    * the fix and exactly what the batch means. */
   struct pipe_resource *batch_touched[256];
   unsigned batch_touched_count;
   bool batch_touched_overflowed;
   /* Occlusion queries. WebGPU counts into a query set slot across a render
    * pass, so exactly one query can be in flight and it is resolved when its
    * result is asked for. */
   /* Transform feedback targets. WebGPU has no transform feedback, so these
    * are storage buffers the lowered vertex shader writes into. */
   struct pipe_stream_output_target *so_targets[GEX_WGPU_MAX_SO_BUFFERS];
   /* Where a client-array glDrawElements' indices were staged. Held on the
    * context rather than released on every path out of draw_vbo, which has
    * more than a dozen; the next indexed user draw replaces it and context
    * teardown drops it. */
   struct pipe_resource *uploaded_indices;
   unsigned so_target_count;
   struct gex_wgpu_compute_state *compute;
   /* The bound geometry shader, and the scratch the three-pass geometry path
    * runs through: the vertex shader's varyings, and the geometry shader's
    * output vertices and indices. Grown as draws need them. */
   struct gex_wgpu_gs *gs;
   /* Rewrites the topologies WebGPU does not have into the ones it does. */
   struct primconvert_context *primconvert;
   struct pipe_resource *gs_vs_out;
   struct pipe_resource *gs_vertices;
   struct pipe_resource *gs_indices;
   /* Shader storage buffers, which only the compute stage may write on
    * WebGPU. Indexed by slot, as Gallium binds them. */
   struct pipe_shader_buffer shader_buffers[GEX_WGPU_MAX_SO_BUFFERS];
   struct gex_wgpu_query *active_query;
   WGPUQuerySet occlusion_set;
   WGPUBuffer occlusion_resolve;
   WGPUBuffer occlusion_readback;
   struct gex_wgpu_sampler *samplers[GEX_WGPU_MAX_SAMPLERS];
   /* Views to bind instead of the sampler view's own, for this draw only.
    *
    * Set when a slot samples a texture that is also a colour attachment of
    * the same draw. WebGPU rejects the whole command buffer for that -- one
    * resource cannot be written and sampled in one synchronization scope --
    * and GL calls it a feedback loop and leaves it undefined only where the
    * texels actually overlap. An application that reads elsewhere in the
    * attachment is entitled to a correct answer, so the attachment is
    * snapshotted and the snapshot sampled. See gex_wgpu_feedback_snapshot. */
   WGPUTextureView feedback_view[GEX_WGPU_MAX_SAMPLERS];
   struct pipe_sampler_view *sampler_views[GEX_WGPU_MAX_SAMPLERS];
   /* The vertex stage's own bindings. Kept separate because GL's are
    * per-stage and the two stages' shaders number their bindings
    * independently; the vertex stage's textures live in bind group 3. */
   struct gex_wgpu_sampler *vertex_samplers[GEX_WGPU_MAX_SAMPLERS];
   struct pipe_sampler_view *vertex_sampler_views[GEX_WGPU_MAX_SAMPLERS];
};

struct gex_wgpu_shader {
   enum mesa_shader_stage stage;
   unsigned ubo_count;
   /* Bit per entry of ubo_slots: set where that constant buffer is backed by
    * a storage buffer rather than a uniform one. */
   uint32_t ubo_storage_mask;
   uint8_t ubo_slots[GEX_WGPU_MAX_CONSTANT_BUFFERS];
   uint32_t ubo_bindings[GEX_WGPU_MAX_CONSTANT_BUFFERS];
   unsigned texture_count;
   uint8_t texture_slots[GEX_WGPU_MAX_SAMPLERS];
   uint32_t texture_bindings[GEX_WGPU_MAX_SAMPLERS];
   WGPUTextureViewDimension texture_dimensions[GEX_WGPU_MAX_SAMPLERS];
   WGPUTextureSampleType texture_sample_types[GEX_WGPU_MAX_SAMPLERS];
   /* Which vertex attributes this shader actually fetches. A draw whose
    * vertex layout does not supply an attribute the shader reads is a WebGPU
    * validation error -- "Vertex attribute slot N used in ShaderModule" --
    * and that poisons the device for every later command, so it has to be
    * decided before the pipeline is built rather than discovered after. */
   uint64_t inputs_read;
   /* The @location numbers ntv actually emits for this stage's inputs, and
    * which of them are integer. inputs_read cannot answer this: it is a
    * VERT_ATTRIB_* mask whose generic attributes start at bit 16, while the
    * module's locations are driver locations counted from zero. */
   uint64_t vertex_locations;
   uint64_t vertex_locations_sint;
   uint64_t vertex_locations_uint;
   /* Which @location each Gallium vertex element feeds, or 0xff for an
    * element this shader has no input for. 0xff throughout means the scan
    * found no input variables, and the caller falls back to the element
    * index. */
   uint8_t vertex_element_location[GEX_WGPU_MAX_VERTEX_ELEMENTS];
   uint16_t integer_sampled_slots;
   /* Texture slots whose shader declaration is multisampled. */
   uint16_t multisampled_slots;
   /* Whether the shader reads gl_DrawID. Only then does a non-zero
    * drawid_offset change what it computes. */
   bool reads_draw_id;
   /* The base type this shader writes to colour output zero. WebGPU requires
    * it to match the target's; a mismatch is a validation error, and a
    * validation error poisons the device for every later command. */
   nir_alu_type color_output_type;
   unsigned sampler_count;
   uint8_t sampler_slots[GEX_WGPU_MAX_SAMPLERS];
   uint32_t sampler_bindings[GEX_WGPU_MAX_SAMPLERS];
   WGPUSamplerBindingType sampler_types[GEX_WGPU_MAX_SAMPLERS];
   WGPUShaderModule module;
   WGPUShaderModule module_y_flipped;
   nir_shader *original_nir;
   /* A clone of the vertex shader kept for building transform feedback
    * compute variants, and the variants built from it so far. */
   nir_shader *capture_nir;
   struct gex_wgpu_capture_variant *captures;
   /* A clone kept for the compute variants that feed a geometry shader, and
    * those variants. Separate from capture_nir because a vertex shader can
    * need one, the other, or both. */
   nir_shader *varying_nir;
   struct gex_wgpu_varying_variant *varyings;
   struct gex_wgpu_shader_variant *variants;
   /* The interpolation this stage declares per varying location: what a
    * fragment shader asks of its inputs, and what a vertex or geometry shader
    * declares on its outputs. WGSL requires the two to agree and rejects the
    * pipeline when they do not; GLSL 1.50 does not require it and takes the
    * fragment stage's qualifier as the meaning, so the vertex stage is
    * rewritten to match -- see gex_wgpu_interp_variant_get. */
   uint8_t varying_interp[GEX_WGPU_MAX_VARYING_LOCATIONS];
   uint64_t varying_interp_mask;
   struct gex_wgpu_interp_variant *interp_variants;
};

struct gex_wgpu_interp_variant {
   uint8_t interp[GEX_WGPU_MAX_VARYING_LOCATIONS];
   /* Sampler state this variant's texture lowering was built against, for a
    * vertex shader that samples an integer texture. The lowering reads
    * filters, wrap modes and view swizzles, so two draws with different
    * sampler state need different modules -- exactly as the fragment stage
    * keys its own variants. */
   uint16_t sampler_mask;
   struct pipe_sampler_state sampler_keys[GEX_WGPU_MAX_SAMPLERS];
   uint8_t swizzles[GEX_WGPU_MAX_SAMPLERS][4];
   /* The vertex-element unpack signature this module was built for. A vertex
    * shader compiled for one attribute layout cannot be reused for another:
    * the unpack is baked into the module. */
   uint8_t attrib_unpack[GEX_WGPU_MAX_VERTEX_ELEMENTS];
   uint64_t mask;
   /* Scale and translate folded into gl_Position for a viewport WebGPU
    * cannot set. All zero when the viewport fits and nothing is folded. */
   float viewport[4];
   WGPUShaderModule module;
   WGPUShaderModule module_y_flipped;
   struct gex_wgpu_interp_variant *next;
};

struct gex_wgpu_shader_variant {
   WGPUShaderModule module;
   /* Framebuffer height baked into this variant's gl_FragCoord correction,
    * or 0 when the draw is not flipped and none was applied. */
   /* The colour output base type this variant declares, which has to match
    * the bound target's. */
   nir_alu_type color_output_type;
   uint16_t sampler_mask;
   struct pipe_sampler_state sampler_keys[GEX_WGPU_MAX_SAMPLERS];
   uint8_t swizzles[GEX_WGPU_MAX_SAMPLERS][4];
   struct gex_wgpu_shader_variant *next;
};

struct gex_wgpu_rasterizer {
   bool scissor;
   bool front_ccw;
   unsigned cull_face;
   bool discard;
   bool unclipped_depth;
};

/* Which varyings travel between two stages through a buffer, and in what
 * order. One vec4 slot each, so the stride is count * 16 and slot i sits at
 * i * 16 -- the same shape the passthrough vertex shader's vertex elements
 * describe, because that buffer is read back as a vertex buffer. */
struct gex_wgpu_varying_layout {
   unsigned count;
   uint32_t location[GEX_WGPU_MAX_VARYINGS];
};

static unsigned
gex_wgpu_varying_stride(const struct gex_wgpu_varying_layout *layout)
{
   return layout->count * 16;
}

static int
gex_wgpu_varying_slot(const struct gex_wgpu_varying_layout *layout,
                      unsigned location)
{
   for (unsigned i = 0; i < layout->count; ++i)
      if (layout->location[i] == location)
         return (int)i;
   return -1;
}

/* The vertex layout a transform feedback compute variant was built for. The
 * variant reads the vertex buffers itself, so a different layout needs a
 * different shader; this is both the cache key and the pass's input. */
struct gex_wgpu_capture_layout {
   unsigned count;
   /* Whether the capture reads its vertex from a slot table rather than from
    * the invocation index. Part of the key because it changes the shader. */
   unsigned index_table;
   uint8_t buffer[PIPE_MAX_ATTRIBS];
   uint32_t offset[PIPE_MAX_ATTRIBS];
   uint32_t stride[PIPE_MAX_ATTRIBS];
   uint32_t format[PIPE_MAX_ATTRIBS];
};

/* A geometry shader, which runs as a compute pass.
 *
 * WebGPU has no geometry stage, so a draw with one bound becomes three passes:
 * the vertex shader runs as compute and writes its varyings to a buffer, this
 * shader runs as compute once per input primitive and writes output vertices
 * and indices, and an ordinary indexed draw rasterizes the result with a
 * generated passthrough vertex shader. That is the shape Asahi uses.
 *
 * The output is allocated worst-case -- max_vertices per input primitive --
 * rather than counted and prefix-summed. That costs memory and a zeroed index
 * buffer, and it buys away two whole passes and an indirect draw, which WebGPU
 * does not expose here anyway. Unwritten index slots stay zero, which is three
 * equal indices, which is a degenerate triangle that rasterizes nothing.
 */
struct gex_wgpu_gs {
   nir_shader *nir;
   /* What the geometry shader reads from the vertex shader, and what it
    * writes for the rasterization draw. */
   struct gex_wgpu_varying_layout inputs;
   struct gex_wgpu_varying_layout outputs;
   unsigned max_vertices;
   unsigned max_triangles;
   /* Output primitives per input primitive, and indices per output primitive.
    * A triangle strip of N vertices is N-2 triangles of 3 indices; a point
    * stream is N points of 1. Keeping both as fields is what lets the emitter
    * and the draw agree without either of them naming a topology. */
   unsigned max_output_prims;
   unsigned indices_per_prim;
   unsigned verts_per_input_prim;
   enum mesa_prim output_prim;
   unsigned ubo_count;
   uint8_t ubo_slots[GEX_WGPU_MAX_CONSTANT_BUFFERS];
   uint32_t ubo_bindings[GEX_WGPU_MAX_CONSTANT_BUFFERS];
   uint32_t ubo_storage_mask;
   WGPUShaderModule module;
   /* The rasterization half: a vertex shader that reads the geometry output
    * buffer as a vertex buffer and passes it straight through, and the vertex
    * element state describing that buffer. */
   struct gex_wgpu_shader *passthrough;
   void *passthrough_elements;
};

/* A vertex shader compiled to compute so that it writes its varyings to a
 * buffer, for a geometry shader to read. Cached per (vertex layout, varying
 * layout): the fetch depends on the first and the store on the second. */
struct gex_wgpu_varying_variant {
   struct gex_wgpu_capture_layout vertex_layout;
   struct gex_wgpu_varying_layout varyings;
   WGPUShaderModule module;
   unsigned vb_mask;
   struct gex_wgpu_varying_variant *next;
};

struct gex_wgpu_capture_variant {
   struct gex_wgpu_capture_layout layout;
   WGPUShaderModule module;
   unsigned so_mask;
   uint32_t so_bindings[GEX_WGPU_MAX_SO_BUFFERS];
   unsigned vb_mask;
   uint32_t vb_bindings[GEX_WGPU_MAX_VERTEX_BUFFERS];
   struct gex_wgpu_capture_variant *next;
};

/* WebGPU guarantees sixteen vertex attributes across eight buffers, and
 * shader_caps[MESA_SHADER_VERTEX].max_inputs already advertises sixteen --
 * refusing more than eight elements here contradicted that, and the geometry
 * passthrough (one element per geometry output) hit it as soon as the
 * geometry stage was allowed its full varying count. */

struct gex_wgpu_vertex_elements {
   unsigned count;
   unsigned buffer_count;
   struct pipe_vertex_element elements[GEX_WGPU_MAX_VERTEX_ELEMENTS];
   /* Per element: which shader-side unpack its format needs, or 0. Non-zero
    * anywhere means the vertex shader needs a variant -- see
    * gex_wgpu_vertex_unpack_kind. */
   uint8_t unpack[GEX_WGPU_MAX_VERTEX_ELEMENTS];
   bool any_unpack;
};

struct gex_wgpu_sampler {
   WGPUSampler sampler;
   struct pipe_sampler_state state;
};

struct gex_wgpu_sampler_view {
   struct pipe_sampler_view base;
   WGPUTextureView view;
   WGPUTextureView array_view;
};

/* One occlusion query.
 *
 * GL's occlusion queries count samples that passed the depth and stencil
 * tests, which is exactly what WebGPU's occlusion query set records.  Only one
 * slot is used: a second concurrent query would need slot allocation and a
 * per-query resolve, and GL allows only one query per target to be active
 * anyway. */
struct gex_wgpu_query {
   unsigned type;
   bool active;
   bool have_result;
   uint64_t result;
   /* TIME_ELAPSED only: CPU nanoseconds at begin_query. */
   uint64_t start;
};

struct gex_wgpu_transfer {
   struct pipe_transfer base;
   void *data;
   /* This write map is staged in `data` for the duration of the map instead of
    * being handed the resource's permanent shadow, and `data` is uploaded and
    * freed at unmap. See gex_wgpu_buffer_map. */
   bool staged_write;
};

struct gex_wgpu_map_state {
   WGPUMapAsyncStatus status;
   bool done;
   /* The status alone does not say why. `Aborted` in particular covers both
    * "unmapped before it resolved" and "destroyed before it resolved", and
    * those have nothing to do with each other. */
   char message[128];
};

struct gex_wgpu_ubo_lowering {
   bool failed;
   /* Whether `driver_location` on this shader's UBO variables is the Gallium
    * constant-buffer index. It is whenever the shader came through GLSL
    * linking -- see the long comment on gex_wgpu_lower_gallium_ubos -- and it
    * is not for a shader built by hand that only ever set `binding`. */
   bool trust_driver_location;
   /* Which of the six ways this pass can refuse a load, and the numbers behind
    * it. One shared message made a real application's shader link unreadable. */
   char why[160];
};

static void gex_wgpu_buffer_subdata(struct pipe_context *pipe,
                                    struct pipe_resource *resource,
                                    unsigned usage, unsigned offset,
                                    unsigned size, const void *data);

static bool
gex_wgpu_lower_ubo_load(nir_builder *b, nir_instr *instr, void *data)
{
   struct gex_wgpu_ubo_lowering *state = data;
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_load_ubo)
      return false;
   if (intr->def.bit_size != 32 || !nir_src_is_const(intr->src[0])) {
      state->failed = true;
      if (!state->why[0])
         snprintf(state->why, sizeof(state->why),
                  "load_ubo bit_size=%u block-index-const=%d",
                  intr->def.bit_size, (int)nir_src_is_const(intr->src[0]));
      return false;
   }
   const unsigned slot = nir_src_as_uint(intr->src[0]);
   const bool constant_offset = nir_src_is_const(intr->src[1]);
   const unsigned offset = constant_offset
      ? nir_src_as_uint(intr->src[1]) : nir_intrinsic_align_offset(intr);
   const unsigned alignment = constant_offset
      ? NIR_ALIGN_MUL_MAX : nir_intrinsic_align_mul(intr);
   const unsigned first_component = (offset & 15u) / 4;
   if ((offset & 3u) || alignment < 16 ||
       first_component + intr->def.num_components > 4) {
      state->failed = true;
      if (!state->why[0])
         snprintf(state->why, sizeof(state->why),
                  "load_ubo slot=%u offset=%u align=%u comps=%u first=%u%s",
                  slot, offset, alignment, intr->def.num_components,
                  first_component, constant_offset ? "" : " (dynamic)");
      return false;
   }
   /* `slot` is the Gallium constant-buffer index -- the block index st/mesa
    * put in the intrinsic and the index it will call set_constant_buffer with.
    * Find the variable that *is* that block.
    *
    * `driver_location` is the only field that carries it. `binding` is the GL
    * binding point (`layout(binding = N)`, or zero), which coincides with the
    * block index by accident and not otherwise, so matching on it picks an
    * unrelated block. That mismatch is what made SuperTuxKart decline roughly
    * two thirds of its draws: a load of block 1 bound itself to some variable
    * whose binding happened to be 1, and the block it really wanted went
    * unread. Only fall back to `binding` for a shader that never had
    * driver_location assigned at all -- one built by hand rather than linked. */
   nir_variable *block = NULL;
   nir_foreach_variable_with_modes(var, b->shader, nir_var_mem_ubo) {
      if (var->data.driver_location == slot) {
         block = var;
         break;
      }
   }
   if (!block && !state->trust_driver_location) {
      nir_foreach_variable_with_modes(var, b->shader, nir_var_mem_ubo) {
         if (var->data.binding == slot) {
            block = var;
            break;
         }
      }
   }
   if (!block || !block->interface_type ||
       !glsl_type_is_interface(block->interface_type) ||
       glsl_get_length(block->interface_type) != 1) {
      state->failed = true;
      if (!state->why[0])
         snprintf(state->why, sizeof(state->why),
                  "load_ubo slot=%u block=%s interface=%s fields=%d", slot,
                  block ? (block->name ? block->name : "(anon)") : "(none)",
                  block && block->interface_type ? "yes" : "no",
                  block && block->interface_type
                     ? (int)glsl_get_length(block->interface_type) : -1);
      return false;
   }
   const struct glsl_type *values =
      glsl_get_struct_field(block->interface_type, 0);
   if (!glsl_type_is_array(values) ||
       glsl_get_array_element(values) != glsl_vec4_type()) {
      state->failed = true;
      if (!state->why[0])
         snprintf(state->why, sizeof(state->why),
                  "load_ubo slot=%u member0 is %s, wanted array of vec4", slot,
                  glsl_get_type_name(values));
      return false;
   }
   block->type = block->interface_type;
   b->cursor = nir_before_instr(instr);
   nir_deref_instr *deref = nir_build_deref_var(b, block);
   deref = nir_build_deref_struct(b, deref, 0);
   nir_def *array_index = constant_offset
      ? nir_imm_int(b, offset / 16)
      : nir_ushr_imm(b, intr->src[1].ssa, 4);
   deref = nir_build_deref_array(b, deref, array_index);
   nir_def *loaded = nir_load_deref(b, deref);
   nir_def *replacement = nir_channels(
      b, loaded, BITFIELD_RANGE(first_component, intr->def.num_components));
   nir_def_rewrite_uses(&intr->def, replacement);
   nir_instr_remove(instr);
   return true;
}

/* `uniform_slots` is how many constant buffers WebGPU will take as uniform
 * buffers on this device. st/mesa reserves slot zero for the default uniform
 * block and reports MaxUniformBlocks as one less, so ARB_uniform_buffer_object
 * -- and with it GL 3.1 -- needs thirteen while WebGPU's guaranteed floor is
 * twelve. The slots past the limit become storage buffers, which have their
 * own budget; the block is an array of vec4, for which std140 and std430 agree,
 * so only the address space and the binding change. */
static bool
gex_wgpu_lower_gallium_ubos(nir_shader *nir,
                            enum mesa_shader_stage stage,
                            unsigned uniform_slots,
                            char *why, size_t why_len)
{
   struct gex_wgpu_ubo_lowering state = {0};
   NIR_PASS(_, nir, nir_opt_constant_folding);
   /* Give every uniform block the one shape the load lowering below can build
    * a deref for: an interface with a single `vec4[]` member.
    *
    * st/mesa already hands the *default* uniform block over in exactly that
    * form, which is why this driver only ever needed to accept it -- but a
    * real GLSL `layout(std140) uniform Foo { mat4 a; mat4 b; }` keeps its
    * declared members, and every one of SuperTuxKart's blocks is that. They
    * were rejected outright, which cost the application its whole modern
    * renderer.
    *
    * Reinterpreting a std140 block as an array of vec4 is sound rather than a
    * reinterpretation of convenience: std140 gives every member a 16-byte
    * aligned base except scalars and vec2, and a load that would straddle a
    * vec4 boundary is refused just below. So `offset / 16` picks the row and
    * `(offset & 15) / 4` the first component, which is what the lowering
    * already does. Only the block's *declared* type changes; its layout, its
    * binding and the bytes the application uploads do not. */
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo) {
      const struct glsl_type *iface = var->interface_type;
      if (!iface || !glsl_type_is_interface(iface))
         continue;
      if (glsl_get_length(iface) == 1) {
         const struct glsl_type *first = glsl_get_struct_field(iface, 0);
         if (glsl_type_is_array(first) &&
             glsl_get_array_element(first) == glsl_vec4_type())
            continue;
      }
      unsigned size = glsl_get_explicit_size(iface, false);
      if (!size)
         size = glsl_get_std140_size(iface, false);
      if (!size)
         continue;
      struct glsl_struct_field field = {0};
      field.type = glsl_array_type(glsl_vec4_type(), DIV_ROUND_UP(size, 16), 16);
      field.name = "values";
      field.location = -1;
      const struct glsl_type *flat =
         glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD140, false,
                             glsl_get_type_name(iface));
      var->interface_type = flat;
      var->type = flat;
   }
   /* Which field carries the constant-buffer index, decided once for the whole
    * shader rather than per variable.
    *
    * `gl_nir_lower_buffers` -- the GLSL linker's block lowering -- opens by
    * setting `driver_location = -1` on *every* UBO and SSBO variable, and then
    * assigns the program's block index to the ones it actually walks a deref
    * for. So a linked shader is in one of two states per variable: a real
    * block index, or -1 meaning "no load in this shader reads this block".
    *
    * Reading `binding` when `driver_location` is -1, which this pass used to
    * do, is therefore wrong twice over. It resurrects blocks nothing reads,
    * and `binding` is the GL binding point rather than the block index, so the
    * slot it produces is unrelated to the one st/mesa will bind. In
    * SuperTuxKart every unread block member -- and its shaders have many --
    * came back as a demand for constant buffer zero, which st had no reason to
    * bind, so `gex_wgpu_draw_vbo` declined the draw. Two thirds of the frame's
    * geometry went missing and the game presented an empty world at full rate.
    *
    * A shader built by hand never went through the linker and has whatever
    * driver_location it was given -- the corpus sets it alongside binding --
    * so "no variable has one" is the only case where `binding` is still the
    * better guess. */
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo) {
      if (var->data.driver_location != UINT_MAX) {
         state.trust_driver_location = true;
         break;
      }
   }
   nir_shader_instructions_pass(
      nir, gex_wgpu_lower_ubo_load, nir_metadata_control_flow, &state);
   if (state.failed) {
      if (why) snprintf(why, why_len, "%s", state.why);
      return false;
   }
   /* The loads now deref the blocks they read, so anything still unreferenced
    * is a block this shader does not use. Dropping it here is what keeps it
    * out of `ubo_slots`, and out of the bind group the draw path then insists
    * on having a buffer for.
    * UBOs only: a shader storage block is reached through `load_ssbo`, which
    * names its block by index and never derefs the variable, so sweeping
    * nir_var_mem_ssbo here would delete every SSBO in the shader. */
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_mem_ubo, NULL);
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo) {
      if (var->data.driver_location == UINT_MAX &&
          state.trust_driver_location) {
         /* Referenced -- it survived the sweep above -- but carrying no block
          * index, so there is no way to know which buffer it wants. Refusing
          * the link says so; guessing produced the silent empty frame. */
         if (why)
            snprintf(why, why_len,
                     "ubo block '%s' is read but has no block index",
                     var->name ? var->name : "(anon)");
         return false;
      }
      const unsigned slot = var->data.driver_location == UINT_MAX
         ? var->data.binding : var->data.driver_location;
      if (slot >= GEX_WGPU_MAX_CONSTANT_BUFFERS) {
         if (why)
            snprintf(why, why_len, "ubo block slot=%u above max=%u", slot,
                     (unsigned)GEX_WGPU_MAX_CONSTANT_BUFFERS);
         return false;
      }
      var->data.driver_location = slot;
      var->data.descriptor_set =
         stage == MESA_SHADER_VERTEX ? 1 : 2;
      var->data.binding = slot < uniform_slots
         ? slot
         : GEX_WGPU_STORAGE_UBO_FIRST_BINDING + (slot - uniform_slots);
      var->data.explicit_binding = 1;
   }
   bool moved = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo) {
      if (var->data.driver_location < uniform_slots)
         continue;
      var->data.mode = nir_var_mem_ssbo;
      moved = true;
   }
   if (moved)
      nir_fixup_deref_modes(nir);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return true;
}

static inline struct gex_wgpu_screen *
gex_screen(struct pipe_screen *screen)
{
   return (struct gex_wgpu_screen *)screen;
}

/* Marking the context failed.
 *
 * This driver refuses anything it cannot render correctly by latching a
 * failure flag, which every later entry point then short-circuits on.  That
 * makes the *first* refusal the only informative one -- everything after it is
 * a consequence -- so record where it happened.  Without this, an unsupported
 * operation surfaces as GL_OUT_OF_MEMORY from some unrelated allocation much
 * later.
 */
/* Running totals, reported from the present path: a declined draw is silent
 * and a frame still presents without it, so the ratio of declined to drawn is
 * the only thing that says whether a frame is empty by construction. */
unsigned gex_wgpu_draws_declined;
/* Host memory this driver is holding, so an out-of-memory can be attributed
 * rather than guessed at. Every buffer resource carries a CPU shadow of its
 * full size, which is the driver's largest claim on the wasm heap by far. */
uint64_t gex_wgpu_shadow_live;
uint64_t gex_wgpu_shadow_unread;
uint64_t gex_wgpu_shadow_mapped;
/* Bytes of shadow belonging to buffers mapped *right now*, as opposed to ever.
 * A transient per-map staging block can only recover what is not permanently
 * mapped, so this is the number that decides whether that redesign pays. */
uint64_t gex_wgpu_shadow_mapped_now;
unsigned gex_wgpu_maps_open;
/* Transfers handed out minus transfers freed. A leaked transfer keeps a
 * reference on its resource, so a leak here stops buffers ever being
 * destroyed -- which is exactly what unbounded shadow memory looks like. */
int gex_wgpu_transfers_live;
unsigned gex_wgpu_buffer_maps, gex_wgpu_buffer_unmaps;
/* Open buffer maps bucketed by their PIPE_MAP_* usage, so the leaking path can
 * be named rather than guessed at: a bucket that only ever grows is the one
 * whose unmap never arrives. */
#define GEX_WGPU_MAP_USAGE_BUCKETS 4096
static int gex_wgpu_map_usage_live[GEX_WGPU_MAP_USAGE_BUCKETS];
/* And by the resource's bind flags, which says who the buffer belongs to:
 * vertex, index and constant buffers come from different Mesa helpers. */
static int gex_wgpu_map_bind_live[GEX_WGPU_MAP_USAGE_BUCKETS];
/* Cumulative, not live: which map shapes the hot path is actually made of.
 * A redesign that makes the common map cheap has to know which one that is. */
static unsigned gex_wgpu_map_usage_total[GEX_WGPU_MAP_USAGE_BUCKETS];
static unsigned gex_wgpu_map_aligned, gex_wgpu_map_unaligned;
/* Why a permanent shadow got allocated, so the lazy path can be told what is
 * still forcing one. */
static unsigned gex_wgpu_maps_staged, gex_wgpu_maps_shadowed;
static unsigned gex_wgpu_shadow_cause_subdata, gex_wgpu_shadow_cause_read,
   gex_wgpu_shadow_cause_bind, gex_wgpu_shadow_cause_unaligned,
   gex_wgpu_shadow_cause_existing;
unsigned gex_wgpu_buffers_live;

static void
gex_wgpu_mark_mapped(struct gex_wgpu_resource *gex)
{
   if (gex->ever_mapped || !gex->shadow)
      return;
   gex->ever_mapped = true;
   gex_wgpu_shadow_mapped += gex->allocation_size;
}

/* The buffer's permanent CPU shadow, allocated the first time something needs
 * one rather than at creation.
 *
 * Most of what this driver allocates is never read back: a SuperTuxKart frame
 * held 1.93 GB of shadows of which 1.4 MB was ever read. Those buffers are
 * written through short-lived, 4-byte-aligned write maps -- 1,337,849 of
 * 1,337,859 maps in a run were aligned -- which need no permanent copy at all.
 * The ones that do need a shadow ask for it here. */
static uint8_t *
gex_wgpu_shadow_ensure(struct gex_wgpu_resource *gex)
{
   if (gex->shadow)
      return gex->shadow;
   gex->shadow = CALLOC(1, gex->allocation_size);
   if (gex->shadow) {
      gex_wgpu_shadow_live += gex->allocation_size;
      gex_wgpu_shadow_unread += gex->allocation_size;
      ++gex_wgpu_buffers_live;
   }
   return gex->shadow;
}

static void gex_wgpu_map_callback(WGPUMapAsyncStatus status,
                                  WGPUStringView message,
                                  void *userdata1, void *userdata2);

/* Read a 4-byte-aligned range of a buffer back from the GPU. Used both by a
 * read map of a GPU-written buffer and to fill a shadow that is being created
 * after the fact -- once the aligned-subdata path below stops creating one
 * eagerly, a later reader has to be able to recover the contents. */
/* Wall clock inside the driver's hot state-setting entry points, so that
 * "the rest of the frame" can be split into the driver's share and the
 * application's. Compiled to nothing unless GEX_WGPU_TIMING is on. */
static int64_t gex_wgpu_time_entry;
static bool gex_wgpu_timing_enabled(void);

struct gex_wgpu_entry_timer { int64_t start; };
static inline void
gex_wgpu_entry_timer_end(struct gex_wgpu_entry_timer *timer)
{
   if (timer->start)
      gex_wgpu_time_entry += os_time_get_nano() - timer->start;
}
/* GEX_WGPU_TIMING=2, not 1: two clock reads in a call this small cost about
 * as much as the call, so leaving it on halves the frame rate and makes the
 * share it reports an overestimate of itself. */
static bool gex_wgpu_entry_timing_enabled(void);
#define GEX_ENTRY_TIMER()                                                     \
   struct gex_wgpu_entry_timer gex_entry_timer                                \
      __attribute__((cleanup(gex_wgpu_entry_timer_end))) = {                  \
         gex_wgpu_entry_timing_enabled() ? os_time_get_nano() : 0 }

/* Every submission goes through here; see gex_wgpu_screen::queue_lock. */
static void
gex_wgpu_queue_submit(struct gex_wgpu_screen *screen, size_t count,
                      WGPUCommandBuffer const *buffers)
{
   simple_mtx_lock(&screen->queue_lock);
   wgpuQueueSubmit(screen->queue, count, buffers);
   simple_mtx_unlock(&screen->queue_lock);
}

/* Scoped device lock; see gex_wgpu_screen::device_lock. Nothing has to
 * unlock on the way out, which matters because these functions return from a
 * dozen places each. */
struct gex_wgpu_device_guard {
   struct gex_wgpu_screen *screen;
};

static inline void
gex_wgpu_device_unguard(struct gex_wgpu_device_guard *guard)
{
   if (guard->screen)
      mtx_unlock(&guard->screen->device_lock);
}

#define GEX_DEVICE_GUARD(scr)                                                 \
   struct gex_wgpu_device_guard gex_device_guard                              \
      __attribute__((cleanup(gex_wgpu_device_unguard))) = { NULL };            \
   do {                                                                       \
      struct gex_wgpu_screen *gex_guard_screen = (scr);                       \
      if (gex_guard_screen && gex_guard_screen->device_lock_needed) {          \
         mtx_lock(&gex_guard_screen->device_lock);                            \
         gex_device_guard.screen = gex_guard_screen;                          \
      }                                                                       \
   } while (0)

static void gex_wgpu_submit_pending_named(struct gex_wgpu_context *ctx,
                                          const char *why);
static void gex_wgpu_end_pass(struct gex_wgpu_context *ctx);
#define gex_wgpu_submit_pending(ctx) \
   gex_wgpu_submit_pending_named((ctx), __func__)

static bool
gex_wgpu_buffer_readback(struct pipe_context *pipe,
                         struct gex_wgpu_resource *gex,
                         uint32_t offset, uint32_t size, void *dest)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   /* Submits its own command buffer, so anything already recorded has to
    * reach the queue first or it would run after this. */
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   if (!size)
      return true;
   WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
   staging_desc.size = size;
   WGPUBuffer staging = wgpuDeviceCreateBuffer(screen->device, &staging_desc);
   WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   if (staging && encoder) {
      wgpuCommandEncoderCopyBufferToBuffer(encoder, gex->buffer, offset,
                                           staging, 0, size);
      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      if (commands) {
         gex_wgpu_queue_submit(screen, 1, &commands);
         wgpuCommandBufferRelease(commands);
      }
   }
   if (encoder)
      wgpuCommandEncoderRelease(encoder);
   struct gex_wgpu_map_state state = {WGPUMapAsyncStatus_Success, false};
   bool ok = false;
   if (staging) {
      WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
      callback.mode = WGPUCallbackMode_AllowSpontaneous;
      callback.callback = gex_wgpu_map_callback;
      callback.userdata1 = &state;
      wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, callback);
      while (!state.done)
         gex_wgpu_jspi_wait();
      if (state.status == WGPUMapAsyncStatus_Success) {
         const void *mapped =
            wgpuBufferGetConstMappedRange(staging, 0, size);
         if (mapped) {
            memcpy(dest, mapped, size);
            ok = true;
         }
      }
      wgpuBufferUnmap(staging);
      wgpuBufferRelease(staging);
   }
   return ok;
}

/* A shadow that something is about to read, with the buffer's current contents
 * in it. Recovering them costs one readback per buffer, not per map: the
 * shadow is then kept, and SuperTuxKart's 1.19 million read maps touch only
 * 1.4 MB of buffers between them. */
static uint8_t *
gex_wgpu_shadow_for_read(struct pipe_context *pipe,
                         struct gex_wgpu_resource *gex)
{
   if (gex->shadow)
      return gex->shadow;
   const bool needs_contents = gex->bytes_written != 0 || gex->gpu_written;
   if (!gex_wgpu_shadow_ensure(gex))
      return NULL;
   if (needs_contents &&
       !gex_wgpu_buffer_readback(pipe, gex, 0,
                                 (uint32_t)gex->allocation_size, gex->shadow))
      return NULL;
   return gex->shadow;
}

static void
gex_wgpu_mark_shadow_read(struct gex_wgpu_resource *gex)
{
   if (gex->shadow_read || !gex->shadow)
      return;
   gex->shadow_read = true;
   gex_wgpu_shadow_unread -= gex->allocation_size;
}

unsigned gex_wgpu_draws_issued;
/* Draws that actually reached the device, grouped by the size of the colour
 * target they went to.
 *
 * "The scene renders" and "the skybox renders" look identical in a single
 * screenshot, and the draw counters cannot tell them apart either: a frame
 * with a hundred thousand draws might be putting all of them into shadow maps
 * and off-screen targets while the presented buffer receives a handful. This
 * says where the work goes, which is the difference between a missing object
 * and a missing composite. */
#define GEX_WGPU_TARGET_BUCKETS 12
struct gex_wgpu_target_bucket {
   unsigned width, height, draws;
   /* The Gallium format, so a deferred renderer's targets can be told apart:
    * an HDR accumulation buffer and a final LDR one look identical as sizes
    * and completely different as formats, and which of them a pass writes is
    * the difference between "too bright" and "not tonemapped". */
   enum pipe_format format;
   /* Keyed on the texture too, not just its size: a deferred renderer has
    * several same-sized targets, and "which one gets the geometry" is a
    * different question from "what size is it". */
   WGPUTexture texture;
};
static struct gex_wgpu_target_bucket
   gex_wgpu_target_buckets[GEX_WGPU_TARGET_BUCKETS];

static void
gex_wgpu_count_draw_target(unsigned width, unsigned height, WGPUTexture texture,
                           enum pipe_format format)
{
   for (unsigned i = 0; i < GEX_WGPU_TARGET_BUCKETS; ++i) {
      struct gex_wgpu_target_bucket *b = &gex_wgpu_target_buckets[i];
      if (b->draws && b->texture != texture)
         continue;
      b->width = width;
      b->height = height;
      b->texture = texture;
      b->format = format;
      ++b->draws;
      return;
   }
}


/* The texture the present path last copied to the canvas, and how many draws
 * landed in it. A screen-sized colour target is not necessarily the one that
 * reaches the screen -- a deferred renderer has several -- so counting draws
 * by size cannot answer "did any of this reach the canvas". This can. */
WGPUTexture gex_wgpu_presented_texture;
unsigned gex_wgpu_draws_to_presented;

/* Draws per vertex shader. The shapes reported above are one line each, which
 * says nothing about which of them the frame is actually made of -- and
 * dumping the attributes of a shader responsible for twelve draws teaches
 * nothing about the ninety thousand that are invisible. */
#define GEX_WGPU_VS_BUCKETS 12
static struct { const void *vs; unsigned draws; }
   gex_wgpu_vs_buckets[GEX_WGPU_VS_BUCKETS];

static void
gex_wgpu_count_draw_vs(const void *vs)
{
   for (unsigned i = 0; i < GEX_WGPU_VS_BUCKETS; ++i) {
      if (gex_wgpu_vs_buckets[i].draws && gex_wgpu_vs_buckets[i].vs != vs)
         continue;
      gex_wgpu_vs_buckets[i].vs = vs;
      ++gex_wgpu_vs_buckets[i].draws;
      return;
   }
}

void gex_wgpu_report_draw_targets(void);
void
gex_wgpu_report_draw_targets(void)
{
   char line[256] = "";
   for (unsigned i = 0; i < GEX_WGPU_TARGET_BUCKETS; ++i) {
      const struct gex_wgpu_target_bucket *b = &gex_wgpu_target_buckets[i];
      if (!b->draws)
         break;
      snprintf(line + strlen(line), sizeof(line) - strlen(line),
               "%ux%u/%s:%u ", b->width, b->height,
               util_format_short_name(b->format), b->draws);
   }

   fprintf(stderr,
           "gex-wgpu: draws by colour target -- %s(presented buffer: %u)\n",
           line[0] ? line : "none", gex_wgpu_draws_to_presented);
   char vsline[256] = "";
   for (unsigned i = 0; i < GEX_WGPU_VS_BUCKETS; ++i) {
      if (!gex_wgpu_vs_buckets[i].draws)
         break;
      snprintf(vsline + strlen(vsline), sizeof(vsline) - strlen(vsline),
               "%p:%u ", gex_wgpu_vs_buckets[i].vs,
               gex_wgpu_vs_buckets[i].draws);
   }
   fprintf(stderr,
           "gex-wgpu: host memory -- %llu KiB of buffer shadows in %u buffers, "
           "%llu KiB never read, %llu KiB ever mapped, "
           "%llu KiB mapped now in %u maps\n",
           (unsigned long long)(gex_wgpu_shadow_live / 1024),
           gex_wgpu_buffers_live,
           (unsigned long long)(gex_wgpu_shadow_unread / 1024),
           (unsigned long long)(gex_wgpu_shadow_mapped / 1024),
           (unsigned long long)(gex_wgpu_shadow_mapped_now / 1024),
           gex_wgpu_maps_open);
   fprintf(stderr,
           "gex-wgpu: maps staged %u shadowed %u; shadow causes "
           "subdata %u read %u bind %u unaligned %u existing %u\n",
           gex_wgpu_maps_staged, gex_wgpu_maps_shadowed,
           gex_wgpu_shadow_cause_subdata, gex_wgpu_shadow_cause_read,
           gex_wgpu_shadow_cause_bind, gex_wgpu_shadow_cause_unaligned,
           gex_wgpu_shadow_cause_existing);
   fprintf(stderr,
           "gex-wgpu: live transfers %d, buffer maps %u unmaps %u\n",
           gex_wgpu_transfers_live, gex_wgpu_buffer_maps,
           gex_wgpu_buffer_unmaps);
   {
      char line[256] = "";
      for (unsigned u = 0; u < GEX_WGPU_MAP_USAGE_BUCKETS; ++u) {
         if (gex_wgpu_map_usage_live[u] == 0)
            continue;
         snprintf(line + strlen(line), sizeof(line) - strlen(line),
                  "0x%x:%d ", u, gex_wgpu_map_usage_live[u]);
      }
      fprintf(stderr, "gex-wgpu: open maps by usage -- %s\n",
              line[0] ? line : "none");
      line[0] = '\0';
      for (unsigned bnd = 0; bnd < GEX_WGPU_MAP_USAGE_BUCKETS; ++bnd) {
         if (gex_wgpu_map_bind_live[bnd] == 0)
            continue;
         snprintf(line + strlen(line), sizeof(line) - strlen(line),
                  "0x%x:%d ", bnd, gex_wgpu_map_bind_live[bnd]);
      }
      fprintf(stderr, "gex-wgpu: open maps by bind -- %s\n",
              line[0] ? line : "none");
      line[0] = '\0';
      for (unsigned u = 0; u < GEX_WGPU_MAP_USAGE_BUCKETS; ++u) {
         if (gex_wgpu_map_usage_total[u] == 0)
            continue;
         snprintf(line + strlen(line), sizeof(line) - strlen(line),
                  "0x%x:%u ", u, gex_wgpu_map_usage_total[u]);
      }
      fprintf(stderr,
              "gex-wgpu: all maps by usage -- %s(aligned %u, unaligned %u)\n",
              line[0] ? line : "none", gex_wgpu_map_aligned,
              gex_wgpu_map_unaligned);
   }
   fprintf(stderr, "gex-wgpu: draws by vertex shader -- %s\n",
           vsline[0] ? vsline : "none");
}
/* Why draws are being dropped, not just how many: with two thirds declined the
 * frame is missing most of its geometry, and the reason is the whole fix. */
unsigned gex_wgpu_declined_constant;
unsigned gex_wgpu_declined_texture;
unsigned gex_wgpu_declined_vertex;
const char *gex_wgpu_declined_last_why;

static void
gex_wgpu_mark_failed(struct gex_wgpu_context *ctx, const char *fn, int line)
{
   if (!ctx->failed)
      fprintf(stderr, "gex-wgpu: first refusal at %s:%d\n", fn, line);
   ctx->failed = true;
}

#define gex_fail(ctx) gex_wgpu_mark_failed((ctx), __func__, __LINE__)

static inline struct gex_wgpu_resource *
gex_resource(struct pipe_resource *resource)
{
   return (struct gex_wgpu_resource *)resource;
}

static void
gex_wgpu_set_compile_error(struct pipe_shader_state *state, const char *message)
{
   if (state->report_compile_error)
      state->error_message = strdup(message ? message : "shader compilation failed");
   /* Said out loud regardless of whether the caller asked to be told.
    * report_compile_error is set by the corpus and by st/mesa, but not by
    * Mesa's internal helpers -- u_blitter among them -- so a shader this
    * driver rejects for one of them used to vanish, and the only symptom was
    * a draw later refused for having no fragment shader. */
   static char last[256];
   const char *text = message ? message : "shader compilation failed";
   if (strncmp(last, text, sizeof(last)) != 0) {
      snprintf(last, sizeof(last), "%s", text);
      fprintf(stderr, "gex-wgpu: shader rejected: %s\n", text);
   }
}

static bool
gex_wgpu_lower_position_y(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   (void)data;
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_POS)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *pos = intr->src[1].ssa;
   nir_def *flipped = nir_vec4(b,
      nir_channel(b, pos, 0), nir_fneg(b, nir_channel(b, pos, 1)),
      nir_channel(b, pos, 2), nir_channel(b, pos, 3));
   nir_src_rewrite(&intr->src[1], flipped);
   return true;
}

/* Fold a GL viewport WebGPU cannot express into the vertex shader.
 *
 * WebGPU's setViewport requires the rectangle to lie inside the attachment;
 * GL allows one that hangs off any edge, and simply clips. The transform is
 * affine, so it can be done in clip space instead and the WebGPU viewport
 * left as the whole framebuffer: for window coordinate
 * w = (ndc + 1) * (size/2) + origin, matching a full-framebuffer viewport
 * means ndc' = ndc * (size/full) + (2*origin + size)/full - 1. Clip space
 * scales the translate by w.
 *
 * data points at four floats: x scale, y scale, x translate, y translate.
 */
static bool
gex_wgpu_lower_viewport_transform(nir_builder *b, nir_intrinsic_instr *intr,
                                  void *data)
{
   const float *xform = data;
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_POS)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *pos = intr->src[1].ssa;
   nir_def *w = nir_channel(b, pos, 3);
   nir_def *x = nir_ffma(b, nir_channel(b, pos, 0),
                         nir_imm_float(b, xform[0]),
                         nir_fmul_imm(b, w, xform[2]));
   nir_def *y = nir_ffma(b, nir_channel(b, pos, 1),
                         nir_imm_float(b, xform[1]),
                         nir_fmul_imm(b, w, xform[3]));
   nir_src_rewrite(&intr->src[1],
                   nir_vec4(b, x, y, nir_channel(b, pos, 2), w));
   return true;
}

struct gex_wgpu_border_lowering {
   bool s;
   bool t;
   bool linear;
   bool integer_sampled;
   bool integer_array;
   bool integer_1d;
   bool integer_3d;
   bool integer_cube;
   uint8_t integer_wrap_s;
   uint8_t integer_wrap_t;
   uint8_t integer_wrap_r;
   uint8_t integer_levels;
   uint8_t integer_mip_filter;
   float lod_bias;
   float integer_min_lod;
   float integer_max_lod;
   uint32_t integer_border[4];
   float color[4];
   uint8_t swizzle[4];
};

struct gex_wgpu_texture_lowering {
   bool failed;
   bool active[GEX_WGPU_MAX_SAMPLERS];
   struct gex_wgpu_border_lowering slots[GEX_WGPU_MAX_SAMPLERS];
};

static uint8_t
gex_wgpu_effective_view_swizzle(const struct pipe_sampler_view *view,
                                unsigned channel)
{
   const uint8_t requested = channel == 0 ? view->swizzle_r
                           : channel == 1 ? view->swizzle_g
                           : channel == 2 ? view->swizzle_b
                                          : view->swizzle_a;
   /* WebGPU exposes only RGBA BC1.  Gallium's RGB DXT1 formats decode the
    * three-color-mode selector as opaque black, so any logical alpha read
    * must be one even though the physical BC1 texel contains alpha zero. */
   if (requested == PIPE_SWIZZLE_W &&
       (view->format == PIPE_FORMAT_DXT1_RGB ||
        view->format == PIPE_FORMAT_DXT1_SRGB))
      return PIPE_SWIZZLE_1;
   return requested;
}

static bool
gex_wgpu_lower_sampler_lod_bias_instr(nir_builder *b, nir_instr *instr,
                                      void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (nir_alu_type_get_base_type(tex->dest_type) != nir_type_float)
      return false;
   const int sampler_src =
      nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (sampler_src < 0)
      return false;
   nir_variable *sampler_var = nir_deref_instr_get_variable(
      nir_src_as_deref(tex->src[sampler_src].src));
   const unsigned slot = sampler_var
      ? sampler_var->data.driver_location : GEX_WGPU_MAX_SAMPLERS;
   struct gex_wgpu_texture_lowering *table = data;
   if (slot >= GEX_WGPU_MAX_SAMPLERS || !table->active[slot]) {
      table->failed = true;
      return false;
   }
   const float bias = table->slots[slot].lod_bias;
   if (bias == 0.0f)
      return false;
   /* Gallium sampler bias does not affect textureGather. */
   if (tex->op == nir_texop_tg4)
      return true;
   if (tex->is_shadow) {
      table->failed = true;
      return false;
   }

   b->cursor = nir_before_instr(instr);
   if (tex->op == nir_texop_tex) {
      tex->op = nir_texop_txb;
      nir_tex_instr_add_src(
         tex, nir_tex_src_bias, nir_imm_float(b, bias));
   } else if (tex->op == nir_texop_txb || tex->op == nir_texop_txl) {
      const nir_tex_src_type source = tex->op == nir_texop_txl
         ? nir_tex_src_lod : nir_tex_src_bias;
      nir_def *original = nir_steal_tex_src(tex, source);
      if (!original) {
         table->failed = true;
         return false;
      }
      nir_tex_instr_add_src(
         tex, source, nir_fadd_imm(b, original, bias));
   } else if (tex->op == nir_texop_txd) {
      /* Gallium's explicit-gradient control path does not apply sampler
       * LOD bias.  Leave both derivatives unchanged. */
      return true;
   } else {
      table->failed = true;
      return false;
   }
   return true;
}

static bool
gex_wgpu_lower_integer_sample_instr(nir_builder *b, nir_instr *instr,
                                    void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   const nir_alu_type result_type =
      nir_alu_type_get_base_type(tex->dest_type);
   if (result_type != nir_type_uint && result_type != nir_type_int)
      return false;
   /* These are the legal integer-texture operations emitted by this pass. */
   if (tex->op == nir_texop_txf || tex->op == nir_texop_txs)
      return false;
   const int texture_src =
      nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   const int sampler_src =
      nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   const int coord_src = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int lod_src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   struct gex_wgpu_texture_lowering *table = data;
   const bool is_1d = tex->sampler_dim == GLSL_SAMPLER_DIM_1D;
   const bool is_3d = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
   const bool supported_dim =
      (is_1d && tex->coord_components == (tex->is_array ? 2 : 1)) ||
      (tex->sampler_dim == GLSL_SAMPLER_DIM_2D &&
       tex->coord_components == (tex->is_array ? 3 : 2)) ||
      (is_3d && !tex->is_array && tex->coord_components == 3);
   if ((tex->op != nir_texop_tex && tex->op != nir_texop_txl) ||
       !supported_dim || tex->is_shadow ||
       texture_src < 0 || sampler_src < 0 || coord_src < 0 ||
       (tex->op == nir_texop_txl && lod_src < 0)) {
      table->failed = true;
      return false;
   }
   nir_variable *sampler_var = nir_deref_instr_get_variable(
      nir_src_as_deref(tex->src[sampler_src].src));
   const unsigned slot = sampler_var
      ? sampler_var->data.driver_location : GEX_WGPU_MAX_SAMPLERS;
   if (slot >= GEX_WGPU_MAX_SAMPLERS || !table->active[slot] ||
       !table->slots[slot].integer_sampled) {
      table->failed = true;
      return false;
   }

   nir_def *texture_deref = tex->src[texture_src].src.ssa;
   nir_def *coord = tex->src[coord_src].src.ssa;
   b->cursor = nir_after_instr(instr);
   const struct gex_wgpu_border_lowering *lower = &table->slots[slot];
   if (lower->integer_cube && tex->op != nir_texop_txl) {
      table->failed = true;
      return false;
   }
   if (tex->is_array != lower->integer_array ||
       is_1d != lower->integer_1d ||
       is_3d != lower->integer_3d) {
      table->failed = true;
      return false;
   }
   nir_def *zero_lod = nir_imm_int(b, 0);
   nir_tex_instr *base_txs = nir_tex_instr_create(b->shader, 2);
   base_txs->op = nir_texop_txs;
   base_txs->sampler_dim = tex->sampler_dim;
   base_txs->is_array = tex->is_array;
   base_txs->dest_type = nir_type_int32;
   base_txs->texture_index = tex->texture_index;
   base_txs->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, texture_deref);
   base_txs->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, zero_lod);
   const unsigned spatial_components = is_1d ? 1 : is_3d ? 3 : 2;
   const nir_component_mask_t spatial_mask =
      (nir_component_mask_t)((1u << spatial_components) - 1u);
   const unsigned query_components =
      spatial_components + (tex->is_array ? 1 : 0);
   nir_def_init(&base_txs->instr, &base_txs->def, query_components, 32);
   nir_builder_instr_insert(b, &base_txs->instr);

   nir_def *selected_lod = zero_lod;
   if (lower->integer_mip_filter == PIPE_TEX_MIPFILTER_NEAREST) {
      nir_def *lod;
      if (tex->op == nir_texop_txl) {
         lod = tex->src[lod_src].src.ssa;
      } else {
         nir_def *size = nir_i2f32(
            b, nir_channels(b, &base_txs->def, spatial_mask));
         nir_def *sample_coord = tex->is_array
            ? nir_channels(b, coord, spatial_mask) : coord;
         nir_def *ddx = nir_fmul(b, nir_ddx(b, sample_coord), size);
         nir_def *ddy = nir_fmul(b, nir_ddy(b, sample_coord), size);
         nir_def *rho = nir_imm_float(b, 0.0f);
         for (unsigned axis = 0; axis < spatial_components; ++axis) {
            nir_def *axis_rho = nir_fmax(
               b, nir_fabs(b, nir_channel(b, ddx, axis)),
               nir_fabs(b, nir_channel(b, ddy, axis)));
            rho = nir_fmax(b, rho, axis_rho);
         }
         lod = nir_flog2(b, rho);
      }
      lod = nir_fadd_imm(b, lod, lower->lod_bias);
      lod = nir_fmax(b, lod, nir_imm_float(b, lower->integer_min_lod));
      lod = nir_fmin(b, lod, nir_imm_float(b, lower->integer_max_lod));
      lod = nir_fmax(b, lod, nir_imm_float(b, 0.0f));
      lod = nir_fmin(
         b, lod, nir_imm_float(b, (float)(lower->integer_levels - 1)));
      selected_lod = nir_f2i32(
         b, nir_ffloor(b, nir_fadd_imm(b, lod, 0.5f)));
   }
   nir_tex_instr *txs = base_txs;
   if (selected_lod != zero_lod) {
      txs = nir_tex_instr_create(b->shader, 2);
      txs->op = nir_texop_txs;
      txs->sampler_dim = tex->sampler_dim;
      txs->is_array = tex->is_array;
      txs->dest_type = nir_type_int32;
      txs->texture_index = tex->texture_index;
      txs->src[0] = nir_tex_src_for_ssa(
         nir_tex_src_texture_deref, texture_deref);
      txs->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, selected_lod);
      nir_def_init(&txs->instr, &txs->def, query_components, 32);
      nir_builder_instr_insert(b, &txs->instr);
   }
   nir_def *wrapped[3];
   for (unsigned axis = 0; axis < spatial_components; ++axis) {
      nir_def *component = nir_channel(b, coord, axis);
      const unsigned wrap = axis == 0 ? lower->integer_wrap_s
                            : axis == 1 ? lower->integer_wrap_t
                                        : lower->integer_wrap_r;
      if (wrap == PIPE_TEX_WRAP_REPEAT) {
         wrapped[axis] = nir_ffract(b, component);
      } else if (wrap == PIPE_TEX_WRAP_MIRROR_REPEAT) {
         nir_def *period = nir_fmul_imm(b, nir_ffract(
            b, nir_fmul_imm(b, component, 0.5f)), 2.0f);
         wrapped[axis] = nir_fsub(
            b, nir_imm_float(b, 1.0f),
            nir_fabs(b, nir_fadd_imm(b, period, -1.0f)));
      } else {
         wrapped[axis] = component;
      }
   }
   nir_def *sample_coord = is_1d ? wrapped[0]
      : is_3d ? nir_vec3(b, wrapped[0], wrapped[1], wrapped[2])
              : nir_vec2(b, wrapped[0], wrapped[1]);
   nir_def *scaled = nir_fmul(
      b, sample_coord,
      nir_i2f32(
         b, nir_channels(b, &txs->def, spatial_mask)));
   nir_def *texel = nir_f2i32(b, nir_ffloor(b, scaled));
   nir_def *zero_coord = is_1d ? nir_imm_int(b, 0)
      : is_3d ? nir_imm_ivec3(b, 0, 0, 0)
              : nir_imm_ivec2(b, 0, 0);
   texel = nir_imin(
      b, nir_imax(b, texel, zero_coord),
      nir_iadd_imm(
         b, nir_channels(
            b, &txs->def, spatial_mask), -1));
   nir_def *fetch_coord = texel;
   if (tex->is_array) {
      const unsigned layer_component = spatial_components;
      nir_def *layer = nir_f2i32(
         b, nir_ffloor(
            b, nir_fadd_imm(
               b, nir_channel(b, coord, layer_component), 0.5f)));
      layer = nir_imin(
         b, nir_imax(b, layer, nir_imm_int(b, 0)),
         nir_iadd_imm(
            b, nir_channel(b, &txs->def, layer_component), -1));
      fetch_coord = is_1d
         ? nir_vec2(b, nir_channel(b, texel, 0), layer)
         : nir_vec3(b, nir_channel(b, texel, 0),
                    nir_channel(b, texel, 1), layer);
   }
   nir_tex_instr *fetch = nir_tex_instr_create(b->shader, 3);
   fetch->op = nir_texop_txf;
   fetch->sampler_dim = tex->sampler_dim;
   fetch->is_array = tex->is_array;
   fetch->coord_components = spatial_components + (tex->is_array ? 1 : 0);
   fetch->dest_type = tex->dest_type;
   fetch->texture_index = tex->texture_index;
   fetch->sampler_index = tex->sampler_index;
   fetch->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, fetch_coord);
   fetch->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, texture_deref);
   fetch->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_lod, selected_lod);
   nir_def_init(&fetch->instr, &fetch->def, tex->def.num_components,
                tex->def.bit_size);
   nir_builder_instr_insert(b, &fetch->instr);
   nir_def *result = &fetch->def;
   if (lower->integer_wrap_s == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
       lower->integer_wrap_t == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
       lower->integer_wrap_r == PIPE_TEX_WRAP_CLAMP_TO_BORDER) {
      nir_def *outside = nir_imm_false(b);
      for (unsigned axis = 0; axis < spatial_components; ++axis) {
         const unsigned wrap = axis == 0 ? lower->integer_wrap_s
                               : axis == 1 ? lower->integer_wrap_t
                                           : lower->integer_wrap_r;
         if (wrap != PIPE_TEX_WRAP_CLAMP_TO_BORDER)
            continue;
         nir_def *component = nir_channel(b, coord, axis);
         outside = nir_ior(
            b, outside,
            nir_ior(b, nir_flt_imm(b, component, 0.0f),
                    nir_flt(b, nir_imm_float(b, 1.0f), component)));
      }
      nir_def *border = nir_imm_ivec4(
         b, (int32_t)lower->integer_border[0],
         (int32_t)lower->integer_border[1],
         (int32_t)lower->integer_border[2],
         (int32_t)lower->integer_border[3]);
      result = nir_bcsel(b, outside, border, result);
   }
   nir_def *swizzled[4];
   for (unsigned i = 0; i < 4; ++i) {
      const unsigned swizzle = lower->swizzle[i];
      if (swizzle <= PIPE_SWIZZLE_W) {
         swizzled[i] = nir_channel(b, result, swizzle);
      } else if (swizzle == PIPE_SWIZZLE_0) {
         swizzled[i] = nir_imm_int(b, 0);
      } else if (swizzle == PIPE_SWIZZLE_1) {
         swizzled[i] = nir_imm_int(b, 1);
      } else {
         table->failed = true;
         return false;
      }
   }
   result = nir_vec4(
      b, swizzled[0], swizzled[1], swizzled[2], swizzled[3]);
   nir_def_replace(&tex->def, result);
   return true;
}

static nir_def *
gex_wgpu_swizzle_channel(nir_builder *b, nir_def *value, unsigned swizzle)
{
   if (swizzle <= PIPE_SWIZZLE_W)
      return nir_channel(b, value, swizzle);
   if (swizzle == PIPE_SWIZZLE_0)
      return nir_imm_float(b, 0.0f);
   if (swizzle == PIPE_SWIZZLE_1)
      return nir_imm_float(b, 1.0f);
   return NULL;
}

static bool
gex_wgpu_lower_nearest_border_instr(nir_builder *b, nir_instr *instr,
                                    void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if ((tex->op != nir_texop_tex && tex->op != nir_texop_txl &&
        tex->op != nir_texop_txb && tex->op != nir_texop_txd) ||
       tex->sampler_dim != GLSL_SAMPLER_DIM_2D || tex->is_array ||
       tex->is_shadow || tex->dest_type != nir_type_float32)
      return false;
   const int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_index < 0 || tex->coord_components != 2)
      return false;

   struct gex_wgpu_texture_lowering *table = data;
   const int sampler_src =
      nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (sampler_src < 0) {
      table->failed = true;
      return false;
   }
   nir_variable *sampler_var = nir_deref_instr_get_variable(
      nir_src_as_deref(tex->src[sampler_src].src));
   const unsigned sampler_slot = sampler_var
      ? sampler_var->data.driver_location : GEX_WGPU_MAX_SAMPLERS;
   if (sampler_slot >= GEX_WGPU_MAX_SAMPLERS ||
       !table->active[sampler_slot]) {
      table->failed = true;
      return false;
   }
   const struct gex_wgpu_border_lowering *lower =
      &table->slots[sampler_slot];
   b->cursor = nir_after_instr(instr);
   nir_def *coord = tex->src[coord_index].src.ssa;
   nir_def *border = nir_imm_vec4(b, lower->color[0], lower->color[1],
                                  lower->color[2], lower->color[3]);
   nir_def *result = &tex->def;
   if ((lower->s || lower->t) && lower->linear) {
      const int texture_index =
         nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (texture_index < 0)
         return false;
      nir_tex_instr *txs = nir_tex_instr_create(b->shader, 1);
      txs->op = nir_texop_txs;
      txs->sampler_dim = GLSL_SAMPLER_DIM_2D;
      txs->dest_type = nir_type_int32;
      txs->src[0] = nir_tex_src_for_ssa(
         nir_tex_src_texture_deref, tex->src[texture_index].src.ssa);
      nir_def_init(&txs->instr, &txs->def, 2, 32);
      nir_builder_instr_insert(b, &txs->instr);
      nir_def *size = nir_i2f32(b, &txs->def);
      nir_def *inside_weight = nir_imm_float(b, 1.0f);
      for (unsigned axis = 0; axis < 2; ++axis) {
         if ((axis == 0 && !lower->s) || (axis == 1 && !lower->t))
            continue;
         nir_def *c = nir_channel(b, coord, axis);
         nir_def *dimension = nir_channel(b, size, axis);
         nir_def *left = nir_ffma(b, c, dimension,
                                  nir_imm_float(b, 0.5f));
         nir_def *right = nir_ffma(
            b, nir_fneg(b, c), dimension,
            nir_fadd(b, dimension, nir_imm_float(b, 0.5f)));
         nir_def *axis_weight = nir_fmin(
            b, nir_fmax(b, nir_fmin(b, left, right),
                        nir_imm_float(b, 0.0f)),
            nir_imm_float(b, 1.0f));
         inside_weight = nir_fmul(b, inside_weight, axis_weight);
      }
      result = nir_flrp(b, border, &tex->def, inside_weight);
   } else if (lower->s || lower->t) {
      nir_def *outside = nir_imm_false(b);
      if (lower->s) {
         nir_def *s = nir_channel(b, coord, 0);
         outside = nir_ior(b, outside,
                           nir_ior(b, nir_flt_imm(b, s, 0.0f),
                                   nir_flt(b, nir_imm_float(b, 1.0f), s)));
      }
      if (lower->t) {
         nir_def *t = nir_channel(b, coord, 1);
         outside = nir_ior(b, outside,
                           nir_ior(b, nir_flt_imm(b, t, 0.0f),
                                   nir_flt(b, nir_imm_float(b, 1.0f), t)));
      }
      result = nir_bcsel(b, outside, border, &tex->def);
   }
   nir_def *channels[4];
   for (unsigned i = 0; i < 4; ++i) {
      channels[i] = gex_wgpu_swizzle_channel(b, result, lower->swizzle[i]);
      if (!channels[i])
         return false;
   }
   result = nir_vec4(b, channels[0], channels[1], channels[2], channels[3]);
   nir_def_rewrite_uses_after(&tex->def, result);
   return true;
}

/* The view swizzle for the texture instructions the border and integer passes
 * do not reach.
 *
 * Those two apply the swizzle themselves, but each only over what it handles:
 * integer sampling of any dimension, and ordinary 2D float sampling. A
 * swizzled 1D, 3D, cube, array or shadow texture -- and a texelFetch of any of
 * them -- fell through both, so the swizzle was silently dropped and the pass
 * reported "no progress", which the caller could not tell from "cannot do it"
 * and latched the context on. st/mesa gives every depth texture a non-identity
 * swizzle, so that was every depth sample: KHR-GL32.texture_repeat_mode lost
 * 97 of its 162 cases to the resulting GL_OUT_OF_MEMORY, which dEQP treats as
 * fatal and which ends the entire run.
 */
static nir_def *
gex_wgpu_swizzle_constant(nir_builder *b, const nir_tex_instr *tex,
                          unsigned swizzle)
{
   const bool is_float =
      nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
   if (swizzle == PIPE_SWIZZLE_0)
      return is_float ? nir_imm_float(b, 0.0f) : nir_imm_int(b, 0);
   return is_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1);
}

/* textureQueryLod, computed rather than asked for.
 *
 * WGSL has no textureQueryLod and SPIR-V's OpImageQueryLod would only move
 * the failure into Tint, so the value is worked out from the coordinate the
 * way the GL specification defines it: scale the texture coordinate by the
 * texture's size, take its screen-space derivatives, and the larger of the
 * two lengths is the scale factor whose log2 is the level of detail.
 *
 * Returns what GL asks for -- x the level that would actually be accessed,
 * clamped into the mip chain, and y the unclamped computed value.
 *
 * Derivatives only exist in a fragment shader, which is where GL allows this
 * to be called; anywhere else it is left alone and refused later rather than
 * quietly given a wrong answer.
 */
static bool
gex_wgpu_lower_query_lod_instr(nir_builder *b, nir_instr *instr, void *data)
{
   (void)data;
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_lod)
      return false;
   if (b->shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   const int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_index < 0)
      return false;
   nir_def *coord = tex->src[coord_index].src.ssa;
   /* The array layer is not a coordinate for this purpose: it does not
    * participate in the scale factor. */
   const unsigned dims = glsl_get_sampler_dim_coordinate_components(
      (enum glsl_sampler_dim)tex->sampler_dim);
   if (coord->num_components < dims)
      return false;

   b->cursor = nir_before_instr(instr);
   coord = nir_trim_vector(b, coord, dims);

   /* Texture size at the base level, as floats, to scale the coordinate into
    * texels -- the derivative of a normalised coordinate says nothing on its
    * own. */
   nir_def *size = nir_i2f32(
      b, nir_build_texture_query(b, tex, nir_texop_txs, dims, nir_type_int32,
                                 false, true));
   nir_def *scaled = nir_fmul(b, coord, size);
   nir_def *ddx = nir_ddx(b, scaled);
   nir_def *ddy = nir_ddy(b, scaled);
   nir_def *rho = nir_fmax(b, nir_fast_length(b, ddx),
                           nir_fast_length(b, ddy));
   /* log2(0) is negative infinity, and a derivative of zero is ordinary --
    * a fragment where the coordinate does not move wants the base level. */
   nir_def *lambda = nir_flog2(b, nir_fmax(b, rho, nir_imm_float(b, 1e-20f)));

   nir_def *levels = nir_i2f32(
      b, nir_build_texture_query(b, tex, nir_texop_query_levels, 1,
                                 nir_type_int32, false, false));
   nir_def *top = nir_fadd_imm(b, levels, -1.0f);
   nir_def *clamped = nir_fclamp(b, lambda, nir_imm_float(b, 0.0f), top);

   nir_def_replace(&tex->def, nir_vec2(b, clamped, lambda));
   return true;
}

static bool
gex_wgpu_lower_view_swizzle_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct gex_wgpu_texture_lowering *table = data;
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   /* A size or level query returns no texel, and a new-style shadow compare
    * returns one scalar that GL does not swizzle. */
   if (nir_tex_instr_is_query(tex) ||
       (tex->is_shadow && tex->is_new_style_shadow))
      return false;
   /* The sampler deref names the slot for a sample; a texelFetch has none, so
    * the texture deref names it there. Both derive from the same combined
    * sampler and carry the same driver location. */
   int slot_src = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (slot_src < 0)
      slot_src = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (slot_src < 0)
      return false;
   nir_variable *slot_var = nir_deref_instr_get_variable(
      nir_src_as_deref(tex->src[slot_src].src));
   const unsigned slot = slot_var ? slot_var->data.driver_location
                                  : GEX_WGPU_MAX_SAMPLERS;
   if (slot >= GEX_WGPU_MAX_SAMPLERS || !table->active[slot])
      return false;
   const struct gex_wgpu_border_lowering *lower = &table->slots[slot];
   /* The integer pass owns every instruction on an integer-sampled slot. */
   if (lower->integer_sampled)
      return false;
   /* The nearest-border pass owns ordinary 2D float sampling; its guard is
    * repeated here rather than shared because it is the definition of what
    * this pass must not touch, and a divergence would double-swizzle. */
   if ((tex->op == nir_texop_tex || tex->op == nir_texop_txl ||
        tex->op == nir_texop_txb || tex->op == nir_texop_txd) &&
       tex->sampler_dim == GLSL_SAMPLER_DIM_2D && !tex->is_array &&
       !tex->is_shadow && tex->dest_type == nir_type_float32 &&
       tex->coord_components == 2 &&
       nir_tex_instr_src_index(tex, nir_tex_src_coord) >= 0)
      return false;
   if (lower->swizzle[0] == PIPE_SWIZZLE_X &&
       lower->swizzle[1] == PIPE_SWIZZLE_Y &&
       lower->swizzle[2] == PIPE_SWIZZLE_Z &&
       lower->swizzle[3] == PIPE_SWIZZLE_W)
      return false;
   for (unsigned i = 0; i < 4; ++i)
      if (lower->swizzle[i] > PIPE_SWIZZLE_1)
         return false;
   b->cursor = nir_after_instr(instr);
   /* textureGather selects one channel, so its swizzle is a component
    * remap rather than a shuffle of the result. */
   if (tex->op == nir_texop_tg4) {
      const unsigned swizzle = lower->swizzle[tex->component];
      if (swizzle <= PIPE_SWIZZLE_W) {
         tex->component = swizzle;
         return true;
      }
      nir_def *constant = gex_wgpu_swizzle_constant(b, tex, swizzle);
      nir_def_replace(&tex->def, nir_replicate(b, constant, 4));
      return true;
   }
   if (nir_tex_instr_dest_size(tex) != 4)
      return false;
   nir_def *channels[4];
   for (unsigned i = 0; i < 4; ++i)
      channels[i] = lower->swizzle[i] <= PIPE_SWIZZLE_W
         ? nir_channel(b, &tex->def, lower->swizzle[i])
         : gex_wgpu_swizzle_constant(b, tex, lower->swizzle[i]);
   /* rewrite_uses_after, not nir_def_replace.
    *
    * The channels above are extracted *from* tex->def, so replacing every use
    * of tex->def rewrote those extractions too: the new vec4's components
    * became channels of the vec4 itself. That self-reference is not a wrong
    * result, it is malformed NIR -- `nir_opt_copy_prop` never terminates on
    * it, so the shader compile hung, the GL device stopped answering, and the
    * guest sat in its ioctl until the runtime gave up thirty seconds later.
    *
    * The sibling pass above (nearest-border) already does it this way for the
    * same reason. The tg4 path a few lines up may keep nir_def_replace: its
    * replacement is an immediate and never reads tex->def.
    *
    * Reached only by the cases the nearest-border pass excludes -- texelFetch
    * most of all -- which is why an R8 texture read with texelFetch wedged
    * while ordinary sampling of the same texture was fine. */
   nir_def_rewrite_uses_after(&tex->def,
                              nir_vec4(b, channels[0], channels[1],
                                       channels[2], channels[3]));
   return true;
}

/* textureGrad on an integer texture.
 *
 * gex_wgpu_lower_integer_sample_instr emits only `tex` and `txl`; anything
 * else fails the whole lowering, so `textureGrad`/`textureGradOffset` on a
 * usampler/isampler took eight KHR-GL33.texture_swizzle smoke cases with it.
 * An integer texture is never filtered, so the gradient only ever chooses a
 * mip level -- exactly what nir_lower_tex's txd lowering computes. Float
 * textureGrad is left alone: the nearest-border pass already handles it, and
 * lowering it would change working shaders. */
static bool
gex_wgpu_lower_integer_txd(const nir_tex_instr *tex, const void *data)
{
   (void)data;
   const nir_alu_type base = nir_alu_type_get_base_type(tex->dest_type);
   return base == nir_type_uint || base == nir_type_int;
}

static bool
gex_wgpu_lower_texture_variant(nir_shader *nir,
                               const struct gex_wgpu_shader *shader,
                               struct gex_wgpu_sampler *const *samplers,
                               struct pipe_sampler_view *const *views)
{
   struct gex_wgpu_texture_lowering table = {0};
   const nir_lower_tex_options rect_options = {
      .lower_rect = true,
      .lower_txd_cb = gex_wgpu_lower_integer_txd,
   };
   bool progress = false;
   NIR_PASS(progress, nir, nir_lower_tex, &rect_options);
   bool has_integer_cube = false;
   for (unsigned i = 0; i < shader->texture_count; ++i) {
      const unsigned slot = shader->texture_slots[i];
      const struct pipe_sampler_view *view = views[slot];
      if (!samplers[slot]) {
         /* Read by texelFetch only. There is no sampler state to model, but
          * the view's swizzle still has to be applied. */
         if (view) {
            table.active[slot] = true;
            for (unsigned channel = 0; channel < 4; ++channel)
               table.slots[slot].swizzle[channel] =
                  gex_wgpu_effective_view_swizzle(view, channel);
         }
         continue;
      }
      const struct pipe_sampler_state *state = &samplers[slot]->state;
      const bool integer_sampled =
         shader->integer_sampled_slots & (1u << slot);
      has_integer_cube |= integer_sampled &&
         (view->target == PIPE_TEXTURE_CUBE ||
          view->target == PIPE_TEXTURE_CUBE_ARRAY);
      if (integer_sampled &&
          (state->min_img_filter != PIPE_TEX_FILTER_NEAREST ||
           state->mag_img_filter != PIPE_TEX_FILTER_NEAREST ||
           (state->min_mip_filter != PIPE_TEX_MIPFILTER_NONE &&
            state->min_mip_filter != PIPE_TEX_MIPFILTER_NEAREST) ||
           (state->wrap_s != PIPE_TEX_WRAP_CLAMP_TO_EDGE &&
            state->wrap_s != PIPE_TEX_WRAP_REPEAT &&
            state->wrap_s != PIPE_TEX_WRAP_MIRROR_REPEAT &&
            state->wrap_s != PIPE_TEX_WRAP_CLAMP_TO_BORDER) ||
           (state->wrap_t != PIPE_TEX_WRAP_CLAMP_TO_EDGE &&
            state->wrap_t != PIPE_TEX_WRAP_REPEAT &&
            state->wrap_t != PIPE_TEX_WRAP_MIRROR_REPEAT &&
            state->wrap_t != PIPE_TEX_WRAP_CLAMP_TO_BORDER) ||
           (state->wrap_r != PIPE_TEX_WRAP_CLAMP_TO_EDGE &&
            state->wrap_r != PIPE_TEX_WRAP_REPEAT &&
            state->wrap_r != PIPE_TEX_WRAP_MIRROR_REPEAT &&
            state->wrap_r != PIPE_TEX_WRAP_CLAMP_TO_BORDER) ||
           ((state->wrap_s == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
             state->wrap_t == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
             state->wrap_r == PIPE_TEX_WRAP_CLAMP_TO_BORDER) &&
            !state->border_color_is_integer) ||
           (view->target != PIPE_TEXTURE_2D &&
            view->target != PIPE_TEXTURE_1D &&
            view->target != PIPE_TEXTURE_1D_ARRAY &&
            view->target != PIPE_TEXTURE_RECT &&
            view->target != PIPE_TEXTURE_2D_ARRAY &&
            view->target != PIPE_TEXTURE_3D &&
            view->target != PIPE_TEXTURE_CUBE &&
            view->target != PIPE_TEXTURE_CUBE_ARRAY) ||
           view->swizzle_r > PIPE_SWIZZLE_1 ||
           view->swizzle_g > PIPE_SWIZZLE_1 ||
           view->swizzle_b > PIPE_SWIZZLE_1 ||
           view->swizzle_a > PIPE_SWIZZLE_1))
         return false;
      table.active[slot] = true;
      table.slots[slot] = (struct gex_wgpu_border_lowering) {
         .s = state->wrap_s == PIPE_TEX_WRAP_CLAMP_TO_BORDER,
         .t = state->wrap_t == PIPE_TEX_WRAP_CLAMP_TO_BORDER,
         .linear = state->min_img_filter == PIPE_TEX_FILTER_LINEAR,
         .integer_sampled = integer_sampled,
         .integer_array = view->target == PIPE_TEXTURE_1D_ARRAY ||
                          view->target == PIPE_TEXTURE_2D_ARRAY ||
                          view->target == PIPE_TEXTURE_CUBE ||
                          view->target == PIPE_TEXTURE_CUBE_ARRAY,
         .integer_1d = view->target == PIPE_TEXTURE_1D ||
                       view->target == PIPE_TEXTURE_1D_ARRAY,
         .integer_3d = view->target == PIPE_TEXTURE_3D,
         .integer_cube = view->target == PIPE_TEXTURE_CUBE ||
                         view->target == PIPE_TEXTURE_CUBE_ARRAY,
         .integer_wrap_s = state->wrap_s,
         .integer_wrap_t = state->wrap_t,
         .integer_wrap_r = state->wrap_r,
         .integer_levels =
            view->u.tex.last_level - view->u.tex.first_level + 1,
         .integer_mip_filter = state->min_mip_filter,
         .lod_bias = state->lod_bias,
         .integer_min_lod = state->min_lod,
         .integer_max_lod = state->max_lod,
         .integer_border = {state->border_color.ui[0],
                            state->border_color.ui[1],
                            state->border_color.ui[2],
                            state->border_color.ui[3]},
         .color = {state->border_color.f[0], state->border_color.f[1],
                   state->border_color.f[2], state->border_color.f[3]},
         .swizzle = {
            gex_wgpu_effective_view_swizzle(view, 0),
            gex_wgpu_effective_view_swizzle(view, 1),
            gex_wgpu_effective_view_swizzle(view, 2),
            gex_wgpu_effective_view_swizzle(view, 3),
         },
      };
   }
   progress |= has_integer_cube &&
      dxil_nir_lower_int_cubemaps(nir, true);
   progress |= nir_shader_instructions_pass(
      nir, gex_wgpu_lower_integer_sample_instr,
      nir_metadata_control_flow | nir_metadata_loop_analysis, &table);
   progress |= nir_shader_instructions_pass(
      nir, gex_wgpu_lower_sampler_lod_bias_instr,
      nir_metadata_control_flow | nir_metadata_loop_analysis, &table);
   progress |= nir_shader_instructions_pass(
      nir, gex_wgpu_lower_nearest_border_instr,
      nir_metadata_control_flow | nir_metadata_loop_analysis, &table);
   progress |= nir_shader_instructions_pass(
      nir, gex_wgpu_lower_view_swizzle_instr,
      nir_metadata_control_flow | nir_metadata_loop_analysis, &table);
   /* Only whether something went wrong. Returning `progress && !failed` made
    * a shader that needed no lowering indistinguishable from one that could
    * not be lowered, and the caller latches the context on the difference --
    * which, through st/mesa's GL_OUT_OF_MEMORY, ends the whole CTS run. A
    * variant that changed nothing is a correct variant. */
   (void)progress;
   return !table.failed;
}

/* gl_PointSize, which WGSL has no builtin for.
 *
 * Tint accepts a PointSize store only when it writes the constant 1.0 --
 * WebGPU draws every point one pixel wide -- and rejects the whole module
 * otherwise, which is a link failure the application sees as an empty info
 * log. GL programs set it routinely, so the store is dropped rather than the
 * program refused. What that gives up is point size, which this backend
 * cannot honour in any case.
 */
static bool
gex_wgpu_strip_point_size_instr(nir_builder *b, nir_intrinsic_instr *intr,
                                void *data)
{
   (void)b;
   (void)data;
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_PSIZ)
      return false;
   nir_instr_remove(&intr->instr);
   return true;
}

static void
gex_wgpu_strip_point_size(nir_shader *nir)
{
   if (!(nir->info.outputs_written & VARYING_BIT_PSIZ))
      return;
   nir_shader_intrinsics_pass(nir, gex_wgpu_strip_point_size_instr,
                              nir_metadata_control_flow, NULL);
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_shader_out) {
      if (var->data.location == VARYING_SLOT_PSIZ)
         exec_node_remove(&var->node);
   }
   nir->info.outputs_written &= ~VARYING_BIT_PSIZ;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

/* Four, matching zink. The reductions this exists for are lowered regardless
 * of the width; the width only decides whether wide ALU is split. */
static uint8_t
gex_wgpu_alu_target_width(const nir_instr *instr, const void *data)
{
   (void)instr;
   (void)data;
   return 4;
}

static WGPUShaderModule
gex_wgpu_compile_nir(struct pipe_context *pipe, nir_shader *nir,
                     enum mesa_shader_stage stage, bool flip_y,
                     char **error_out)
{
   *error_out = NULL;
   const nir_lower_tex_options texture_options = {
      .lower_1d = true,
      .lower_rect = true,
   };
   NIR_PASS(_, nir, nir_lower_tex, &texture_options);
   bool lowered_texture_declaration = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_texture(type) ||
          (glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_1D &&
           glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_RECT))
         continue;
      const bool is_array = glsl_sampler_type_is_array(type);
      var->type = glsl_texture_type(
         GLSL_SAMPLER_DIM_2D, is_array,
         glsl_get_sampler_result_type(type));
      lowered_texture_declaration = true;
   }
   if (lowered_texture_declaration)
      nir_fixup_deref_types(nir);
   if (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY)
      gex_wgpu_strip_point_size(nir);
   if (stage == MESA_SHADER_FRAGMENT) {
      /* GLSL ES 1.00 and desktop GLSL up to 1.20 write gl_FragColor, which
       * st/mesa keeps as FRAG_RESULT_COLOR: one output broadcast to every
       * draw buffer.  WebGPU has no such thing -- a colour target must have a
       * matching @location output -- so expand it into FRAG_RESULT_DATA0..n,
       * which is the only form the WGSL emission below produces.  Without
       * this the pipeline has a colour target and no fragment output, and
       * WebGPU rejects it.
       */
      NIR_PASS(_, nir, nir_lower_fragcolor,
               nir->info.fs.color_is_dual_source ? 1
                                                 : GEX_WGPU_MAX_COLOR_TARGETS);
      /* `discard` under a condition arrives as the `terminate_if` intrinsic,
       * and the SPIR-V emitter below implements only the unconditional
       * `terminate`. It did not refuse it -- it hit an `unreachable`, which is
       * a wasm trap: the shader compile died mid-call, and what surfaced was
       * st/mesa carrying on from the wreckage with a negative buffer-map width
       * and a 2.3 GB allocation. Turn it into ordinary control flow first,
       * which is exactly what zink does before the same emitter
       * (`zink_compiler.c:3057`). `demote_if` goes the same way. */
      NIR_PASS(_, nir, nir_lower_discard_if,
               nir_lower_demote_if_to_cf | nir_lower_terminate_if_to_cf);
   }
   if (stage == MESA_SHADER_VERTEX) {
      /* WGSL requires a vertex entry point to return @builtin(position), and
       * Tint enforces it: "a vertex shader must include the 'position'
       * builtin in its return type". GL does not -- a transform-feedback
       * program legitimately writes only its capture varyings and never
       * gl_Position, because it runs with the rasterizer discarded. Ten of
       * the transform_feedback cases failed to link for exactly that.
       *
       * Writing a constant is correct rather than merely sufficient: the
       * position of a shader that declares none is never consumed, since
       * nothing downstream rasterizes. */
      if (!(nir->info.outputs_written & VARYING_BIT_POS)) {
         nir_variable *pos = nir_variable_create(
            nir, nir_var_shader_out, glsl_vec4_type(), "gex_position");
         pos->data.location = VARYING_SLOT_POS;
         nir_function_impl *impl = nir_shader_get_entrypoint(nir);
         nir_builder b = nir_builder_at(nir_after_impl(impl));
         nir_store_var(&b, pos, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f), 0xf);
         nir->info.outputs_written |= VARYING_BIT_POS;
         nir_progress(true, impl, nir_metadata_control_flow);
      }
      NIR_PASS(_, nir, nir_lower_clip_halfz);
      if (flip_y)
         nir_shader_intrinsics_pass(
            nir, gex_wgpu_lower_position_y,
            nir_metadata_control_flow | nir_metadata_loop_analysis, NULL);
   }
   /* Vector comparison reductions, and anything wider than a vec4.
    *
    * `a.rgb == b.rgb` in GLSL is one `ball_fequal3` instruction, and zink's
    * SPIR-V emitter -- which this driver borrows whole -- has no case for
    * `ball_*` or `bany_*` at all: they fall through emit_alu's switch into
    * `unreachable`, which in wasm is a trap. The shader state creation died
    * mid-call, so glLinkProgram reported failure with an empty info log and
    * no driver diagnostic, and every KHR-GL33.pixelstoragemodes case failed
    * because its fragment shader compares `colour.rgb == refcolour.rgb`.
    *
    * nir_lower_alu_width expands those reductions into per-component feq plus
    * iand. The callback returns 4 for the same reason zink's does
    * (zink_compiler.c:5682): the reduction cases in that pass run ahead of the
    * width check, so this lowers what the emitter cannot take while leaving
    * ordinary vec4 ALU alone. */
   NIR_PASS(_, nir, nir_lower_alu_width, gex_wgpu_alu_target_width, NULL);
   /* textureQueryLod, which WGSL has no form of at all.
    *
    * This belongs here rather than in gex_wgpu_lower_texture_variant, where
    * it started: that pass only runs for shaders needing a sampler variant,
    * so a shader whose only unusual feature was the query went straight to
    * SPIR-V and Tint refused its OpImageQueryLod. Nothing in the lowering
    * depends on the bound view, so the base shader is the right place. */
   NIR_PASS(_, nir, nir_shader_instructions_pass,
            gex_wgpu_lower_query_lod_instr,
            nir_metadata_control_flow | nir_metadata_loop_analysis, NULL);
   ntv_shader_prepare(nir);
   /* No deref may outlive the block it is used in.
    *
    * A NIR deref is an SSA value and ntv translates one into a single
    * OpAccessChain, reused wherever the deref is used. That is faithful
    * SPIR-V and valid -- spirv-val accepts it -- but a pointer cannot be held
    * across basic blocks in WGSL, so Tint's reader has to hoist it into a
    * function-scope `var` of pointer type and then refuses its own output:
    * "function-scope 'var' must have a constructible type".
    *
    * It takes a loop to produce. `uniform mat4 M` multiplied by a value a
    * loop computed put the access chain for M inside the loop body and its
    * four uses after the loop; neither the matrix nor the loop fails alone.
    * glmark2's three `loop` scenes are exactly that shader, and this is why
    * they would not link.
    *
    * Re-creating each deref in the block that uses it removes the crossing
    * altogether, and costs nothing anywhere else: a deref already local to
    * its block is left as it is. */
   nir_foreach_function_impl(impl, nir)
      nir_rematerialize_derefs_in_use_blocks_impl(impl);
   const struct ntv_info info = {
      .is_native_vulkan = true,
      .omit_fragment_output_index_zero = true,
      .avoid_spv_khr_fma = true,
      .spirv_version = SPIRV_VERSION(1, 3),
      /* gex_wgpu has no bindless descriptor set.  Leaving this zero makes
       * nir_to_spirv treat every ordinary set-0 sampler as bindless and index
       * its two-entry bindless table with the sampler's driver location, which
       * writes out of bounds as soon as a shader declares a third sampler.
       * Use a set index no variable can carry.
       */
      .bindless_set_idx = UINT_MAX,
   };
   /* Timed, because "the shader path is slow" is not actionable and the three
    * stages have wildly different costs: NIR->SPIR-V is Mesa's, SPIR-V->WGSL
    * is Tint parsing and reprinting a whole program, and the module build is
    * the browser compiling that text again. */
   /* Whether this is one of the scene's vertex shaders, decided before the
    * NIR is freed. Nine or more inputs is what the geometry uses; the UI and
    * skybox shaders have one to four. */
   const bool dump_wgsl = nir->info.stage == MESA_SHADER_VERTEX &&
      util_bitcount64(nir->info.inputs_read) >= 9;
   const double t_start = os_time_get_nano() / 1000000.0;
   struct spirv_shader *spirv = nir_to_spirv(nir, &info);
   ralloc_free(nir);
   if (!spirv) {
      *error_out = strdup("Mesa NIR to SPIR-V failed");
      return NULL;
   }
   const double t_spirv = os_time_get_nano() / 1000000.0;

   char *wgsl = gex_wgpu_spirv_to_wgsl(spirv->words, spirv->num_words,
                                        error_out);
   const unsigned spirv_words = spirv->num_words;
   spirv_shader_delete(spirv);
   if (!wgsl)
      return NULL;
   const double t_wgsl = os_time_get_nano() / 1000000.0;

   WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
   source.code = (WGPUStringView){wgsl, WGPU_STRLEN};
   WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
   desc.nextInChain = &source.chain;
   WGPUShaderModule module = wgpuDeviceCreateShaderModule(
      gex_screen(pipe->screen)->device, &desc);
   const double t_module = os_time_get_nano() / 1000000.0;
   /* Capped, and then only when something is worth saying.
    *
    * One line per shader is fine for a demo and ruinous for a conformance
    * run: a texture_swizzle sweep compiles thousands of them, the harness
    * accumulates every console line, and a 45 MB log stalled the run outright
    * -- the browser stopped making progress long before the wall. The first
    * few compiles are what diagnose a slow shader path; after that only a
    * genuinely slow one is news. */
   {
      static unsigned said;
      const double total = t_module - t_start;
      if (said < 16 || total >= 100.0) {
         ++said;
         fprintf(stderr,
                 "gex-wgpu: shader compiled in %.0f ms (nir->spirv %.0f, "
                 "spirv->wgsl %.0f, module %.0f) spirv_words=%u "
                 "wgsl_bytes=%zu\n",
                 total, t_spirv - t_start, t_wgsl - t_spirv,
                 t_module - t_wgsl, spirv_words, strlen(wgsl));
      }
   }
   if (dump_wgsl) {
      /* In chunks: the harness truncates a console line, and the whole point
       * is to read how @builtin(position) is computed. Once per run. */
      static bool dumped;
      if (!dumped) {
         dumped = true;
         const size_t len = strlen(wgsl);
         fprintf(stderr, "gex-wgpu: scene vs WGSL (%zu bytes) follows\n", len);
         for (size_t at = 0; at < len; at += 160) {
            char chunk[161];
            const size_t n = MIN2((size_t)160, len - at);
            memcpy(chunk, wgsl + at, n);
            chunk[n] = '\0';
            for (size_t i = 0; i < n; ++i)
               if (chunk[i] == '\n') chunk[i] = ' ';
            fprintf(stderr, "gex-wgsl| %s\n", chunk);
         }
      }
   }
   FREE(wgsl);
   if (!module)
      *error_out = strdup("WebGPU shader module creation failed");
   return module;
}

/* Transform feedback, run as a compute pass.
 *
 * WGSL forbids a storage buffer with read_write access in the vertex stage,
 * so a vertex shader cannot write its own captures however large the
 * adapter's maxStorageBuffersInVertexStage happens to be.  Drivers on APIs
 * with the same restriction -- Metal, most obviously -- run the vertex shader
 * a second time as a compute shader whose invocations write the capture
 * buffers, and that is what happens here.
 *
 * A compute shader has no fixed-function vertex fetch, so attribute reads
 * become loads from the vertex buffers bound as storage buffers, and
 * gl_VertexID becomes the invocation index.  That makes the variant depend on
 * the vertex layout as well as on the shader, so it is cached per layout.
 *
 * Mesa's own nir_lower_xfb_to_stores is not usable: it rewrites store_output
 * *intrinsics*, and this driver keeps shader outputs as variables and derefs
 * all the way down because that is the form the SPIR-V emitter consumes.  The
 * capture arithmetic below is the one that pass documents,
 *
 *    (instance_id * num_vertices + vertex_id) * stride + offset
 *
 * with instanced capture refused at the draw, so the instance term is zero
 * and the invocation index alone selects the slot.
 */
struct gex_wgpu_capture {
   const struct gex_wgpu_capture_layout *layout;
   /* Set instead of the transform feedback buffers when this variant is
    * feeding a geometry shader: every varying in the layout is written to one
    * buffer, at a fixed vec4 slot. */
   const struct gex_wgpu_varying_layout *varyings;
   nir_variable *varying_out;
   nir_variable *params;
   nir_variable *index;
   nir_variable *so[GEX_WGPU_MAX_SO_BUFFERS];
   nir_variable *vb[GEX_WGPU_MAX_VERTEX_BUFFERS];
   unsigned so_mask;
   unsigned vb_mask;
   bool failed;
};

/* An unsized array of 32-bit words in an interface block.  Everything the
 * capture pass touches -- vertex data, captured output, the parameters -- is a
 * byte stream whose layout GL already fixed, so the shader only needs
 * somewhere to put words. */
static nir_variable *
gex_wgpu_word_block(nir_shader *nir, const char *name, unsigned binding,
                    unsigned driver_location)
{
   const struct glsl_type *words = glsl_array_type(glsl_uint_type(), 0, 4);
   struct glsl_struct_field field = {0};
   field.type = words;
   field.name = "data";
   const struct glsl_type *block =
      glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD430, false,
                          name);
   nir_variable *var = nir_variable_create(nir, nir_var_mem_ssbo, block, name);
   var->interface_type = block;
   var->data.binding = binding;
   var->data.driver_location = driver_location;
   var->data.explicit_binding = true;
   return var;
}

static nir_deref_instr *
gex_wgpu_block_words(nir_builder *b, nir_variable *var)
{
   return nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
}

static nir_def *
gex_wgpu_param(nir_builder *b, struct gex_wgpu_capture *state, unsigned word)
{
   nir_deref_instr *words = gex_wgpu_block_words(b, state->params);
   return nir_load_deref(
      b, nir_build_deref_array(b, words, nir_imm_int(b, word)));
}

/* Which vertex this invocation stands for.  The slot it captures into is the
 * invocation index; the vertex it reads is that plus the draw's first vertex,
 * because gl_VertexID counts from there. */
static nir_def *
gex_wgpu_capture_slot(nir_builder *b)
{
   return nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
}

static nir_def *
gex_wgpu_capture_vertex(nir_builder *b, struct gex_wgpu_capture *state)
{
   if (!state->layout || !state->layout->index_table || !state->index)
      return nir_iadd(b, gex_wgpu_capture_slot(b),
                      gex_wgpu_param(b, state, GEX_WGPU_PARAM_FIRST_VERTEX));

   /* One entry per captured vertex, already resolved through the index buffer
    * and through the primitive topology, so gl_VertexID is a single load. */
   nir_deref_instr *words = gex_wgpu_block_words(b, state->index);
   return nir_load_deref(
      b, nir_build_deref_array(b, words, gex_wgpu_capture_slot(b)));
}

static nir_variable *
gex_wgpu_capture_so_var(nir_shader *nir, struct gex_wgpu_capture *state,
                        unsigned buffer)
{
   if (buffer >= GEX_WGPU_MAX_SO_BUFFERS) {
      state->failed = true;
      return NULL;
   }
   if (!state->so[buffer]) {
      char name[32];
      snprintf(name, sizeof(name), "gex_xfb%u", buffer);
      state->so[buffer] =
         gex_wgpu_word_block(nir, name, GEX_WGPU_SO_FIRST_BINDING + buffer,
                             buffer);
      state->so_mask |= 1u << buffer;
   }
   return state->so[buffer];
}

static nir_variable *
gex_wgpu_capture_vb_var(nir_shader *nir, struct gex_wgpu_capture *state,
                        unsigned slot)
{
   if (slot >= GEX_WGPU_MAX_VERTEX_BUFFERS) {
      state->failed = true;
      return NULL;
   }
   if (!state->vb[slot]) {
      char name[32];
      snprintf(name, sizeof(name), "gex_vb%u", slot);
      state->vb[slot] =
         gex_wgpu_word_block(nir, name, GEX_WGPU_VB_FIRST_BINDING + slot,
                             slot);
      state->vb_mask |= 1u << slot;
   }
   return state->vb[slot];
}

/* Read one vertex attribute out of its vertex buffer, in place of the fetch a
 * compute shader does not get. */
static nir_def *
gex_wgpu_capture_fetch(nir_builder *b, struct gex_wgpu_capture *state,
                       unsigned element, unsigned components)
{
   const struct gex_wgpu_capture_layout *layout = state->layout;
   nir_def *zero = nir_imm_zero(b, components, 32);
   if (element >= layout->count) {
      state->failed = true;
      return zero;
    }

   unsigned words = 0;
   bool unorm8 = false;
   switch ((enum pipe_format)layout->format[element]) {
   case PIPE_FORMAT_R32_FLOAT: words = 1; break;
   case PIPE_FORMAT_R32G32_FLOAT: words = 2; break;
   case PIPE_FORMAT_R32G32B32_FLOAT: words = 3; break;
   case PIPE_FORMAT_R32G32B32A32_FLOAT: words = 4; break;
   case PIPE_FORMAT_R8G8B8A8_UNORM: words = 1; unorm8 = true; break;
   default:
      /* Every other format is translated by u_vbuf before it reaches the
       * driver, so reaching this is a driver bug rather than an app one. */
      state->failed = true;
      return zero;
   }

   const unsigned slot = layout->buffer[element];
   nir_variable *buffer = gex_wgpu_capture_vb_var(b->shader, state, slot);
   if (!buffer)
      return zero;

   /* Word-addressed, so the attribute has to start on a word boundary. GL
    * only requires alignment to the component type, so a byte-format
    * attribute at an odd offset is legal and refused rather than mis-read. */
   if ((layout->offset[element] & 3) || (layout->stride[element] & 3)) {
      state->failed = true;
      return zero;
   }

   nir_def *byte =
      nir_iadd(b, gex_wgpu_param(b, state, GEX_WGPU_PARAM_VB_OFFSET + slot),
               nir_iadd_imm(b,
                            nir_imul_imm(b, gex_wgpu_capture_vertex(b, state),
                                         layout->stride[element]),
                            layout->offset[element]));
   nir_def *word = nir_ushr_imm(b, byte, 2);
   nir_deref_instr *data = gex_wgpu_block_words(b, buffer);

   nir_def *raw[4];
   for (unsigned i = 0; i < words; ++i)
      raw[i] = nir_load_deref(
         b, nir_build_deref_array(b, data, nir_iadd_imm(b, word, i)));

   nir_def *comp[4];
   unsigned have = words;
   if (unorm8) {
      for (unsigned i = 0; i < 4; ++i)
         comp[i] = nir_fmul_imm(
            b, nir_u2f32(b, nir_iand_imm(b, nir_ushr_imm(b, raw[0], i * 8),
                                          0xff)),
            1.0 / 255.0);
      have = 4;
   } else {
      /* A NIR value is an untyped bit pattern, so the loaded words are
       * already the floats the shader expects. */
      for (unsigned i = 0; i < words; ++i)
         comp[i] = raw[i];
   }
   /* GL fills the components a format does not supply with (0, 0, 0, 1). */
   for (unsigned i = have; i < 4; ++i)
      comp[i] = nir_imm_float(b, i == 3 ? 1.0f : 0.0f);
   return nir_vec(b, comp, components);
}

static bool
gex_wgpu_capture_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gex_wgpu_capture *state = data;
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (!deref || deref->deref_type != nir_deref_type_var ||
       deref->var->data.mode != nir_var_shader_in)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = gex_wgpu_capture_fetch(
      b, state, deref->var->data.driver_location, intr->def.num_components);
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

/* The vertex-stage system values a compute invocation has to stand in for.
 * Instanced capture is refused at the draw, so the instance is always zero. */
static bool
gex_wgpu_capture_builtin(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gex_wgpu_capture *state = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_vertex_id:
      value = gex_wgpu_capture_vertex(b, state);
      break;
   case nir_intrinsic_load_vertex_id_zero_base:
      value = gex_wgpu_capture_slot(b);
      break;
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_base_vertex:
      value = gex_wgpu_param(b, state, GEX_WGPU_PARAM_FIRST_VERTEX);
      break;
   case nir_intrinsic_load_instance_id:
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_draw_id:
      value = nir_imm_int(b, 0);
      break;
   default:
      return false;
   }
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Emit the capture stores for one write to a captured output variable. */
static bool
gex_wgpu_capture_store(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gex_wgpu_capture *state = data;
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var || var->data.mode != nir_var_shader_out)
      return false;

   nir_def *value = intr->src[1].ssa;
   /* Which components of the varying slot this variable actually occupies.
    *
    * The linker packs several small varyings into one slot, so `a` and `b`
    * both arrive here as location 32 and are told apart only by
    * location_frac. Matching on the location alone made every store find the
    * first output written to that slot, so a capture split across two buffers
    * with gl_NextBuffer wrote one of them twice and the other not at all --
    * and because a single-varying capture has nothing to collide with, it
    * looked correct everywhere else. */
   const unsigned frac = var->data.location_frac;
   const unsigned var_mask =
      (unsigned)BITFIELD_MASK(value->num_components) << frac;

   const nir_xfb_info *xfb = b->shader->xfb_info;
   bool captured = false;
   for (unsigned o = 0; o < xfb->output_count; ++o) {
      const nir_xfb_output_info *out = &xfb->outputs[o];
      if (out->location != var->data.location)
         continue;
      if (!(out->component_mask & var_mask))
         continue;

      nir_variable *buffer =
         gex_wgpu_capture_so_var(b->shader, state, out->buffer);
      if (!buffer)
         return false;

      /* Bytes. nir_xfb_info's buffer stride is already in bytes -- it is
       * nir->info.xfb_stride, which is in dwords, times four
       * (nir_io_add_xfb_info.c:16 converts the other way) -- and so is
       * out->offset, which nir_gather_xfb_info advances by
       * bitcount(component_mask) * 4. Multiplying again made every captured
       * vertex land four strides apart, which is invisible in a one-vertex
       * capture and wrong in every other: the transform_feedback cases that
       * passed all draw a single point. */
      const unsigned stride = xfb->buffers[out->buffer].stride;
      b->cursor = nir_after_instr(&intr->instr);

      nir_def *byte = nir_iadd(
         b, gex_wgpu_param(b, state, GEX_WGPU_PARAM_SO_OFFSET + out->buffer),
         nir_iadd_imm(b, nir_imul_imm(b, gex_wgpu_capture_slot(b), stride),
                      out->offset));
      nir_def *word = nir_ushr_imm(b, byte, 2);
      nir_deref_instr *data_deref = gex_wgpu_block_words(b, buffer);

      /* One word per component: the captured stream is untyped bytes whose
       * layout GL already fixed, and a vector store would impose an alignment
       * it never promised.
       *
       * out->offset is the byte offset of the lowest component in the output's
       * mask, so a component's own offset is that plus its rank among the set
       * bits below it -- not its index within the variable, which is the same
       * number only when the variable starts at component zero and the output
       * covers all of it. */
      for (unsigned c = 0; c < value->num_components; ++c) {
         const unsigned component = frac + c;
         if (component > 3)
            break;
         if (!(out->component_mask & (1u << component)))
            continue;
         const unsigned rank =
            util_bitcount(out->component_mask & BITFIELD_MASK(component));
         nir_deref_instr *word_deref = nir_build_deref_array(
            b, data_deref, nir_iadd_imm(b, word, rank));
         nir_store_deref(b, word_deref, nir_channel(b, value, c), 0x1);
      }
      captured = true;
   }
   return captured;
}

/* A compute shader has no outputs; the captures above replaced the only ones
 * that mattered. */
static bool
gex_wgpu_capture_drop_output(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   (void)b;
   (void)data;
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var || var->data.mode != nir_var_shader_out)
      return false;
   nir_instr_remove(&intr->instr);
   return true;
}

/* Split each combined sampler into a texture and a bare sampler.
 *
 * tgsi_to_nir makes one glsl_sampler_type variable per sampler unit, and both
 * the texture and the sampler source of a tex instruction deref it. Vulkan
 * expresses that directly as a combined image-sampler, which is why zink
 * consumes the same NIR unchanged. WGSL has no such thing: a texture and a
 * sampler are separate resources and cannot share a binding, so the combined
 * form has to be taken apart before it reaches the emitter.
 *
 * The pair keeps one driver_location -- the driver's slot model expects a
 * texture and its sampler to agree on that -- and takes two bindings, which
 * is what makes them addressable separately in WGSL.
 */
static bool
gex_wgpu_split_combined_samplers(nir_shader *nir)
{
   nir_variable *split_from[GEX_WGPU_MAX_SAMPLERS] = {0};
   nir_variable *split_texture[GEX_WGPU_MAX_SAMPLERS] = {0};
   nir_variable *split_sampler[GEX_WGPU_MAX_SAMPLERS] = {0};
   unsigned count = 0;
   unsigned rewrites = 0, tex_seen = 0;
   unsigned split_slot[GEX_WGPU_MAX_SAMPLERS] = {0};

   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      /* Combined only: a bare sampler or a texture is already split. */
      if (!glsl_type_is_sampler(type) || glsl_type_is_bare_sampler(type))
         continue;
      if (var->type != type || count >= GEX_WGPU_MAX_SAMPLERS)
         return false;   /* arrays of samplers are not handled here */
      const unsigned slot = var->data.binding;
      if (slot >= GEX_WGPU_MAX_SAMPLERS)
         return false;

      char name[32];
      snprintf(name, sizeof(name), "gex_texture%u", slot);
      nir_variable *texture = nir_variable_create(
         nir, nir_var_uniform,
         glsl_texture_type(glsl_get_sampler_dim(type),
                           glsl_sampler_type_is_array(type),
                           glsl_get_sampler_result_type(type)),
         name);
      /* Set 3 for the vertex stage, 0 for the fragment stage. The two
       * shaders number their bindings independently, so they cannot share a
       * group; set 1 and set 2 are already the two stages' uniform blocks. */
      const unsigned texture_set =
         nir->info.stage == MESA_SHADER_VERTEX ? 3 : 0;
      texture->data.descriptor_set = texture_set;
      texture->data.driver_location = slot;
      texture->data.binding = 2 * slot;

      snprintf(name, sizeof(name), "gex_sampler%u", slot);
      nir_variable *sampler = nir_variable_create(
         nir, nir_var_uniform, glsl_bare_sampler_type(), name);
      sampler->data.descriptor_set = texture_set;
      sampler->data.driver_location = slot;
      sampler->data.binding = 2 * slot + 1;

      split_from[count] = var;
      split_slot[count] = slot;
      split_texture[count] = texture;
      split_sampler[count] = sampler;
      ++count;
   }
   if (!count)
      return true;

   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            ++tex_seen;
            /* tgsi_to_nir ends with nir_lower_samplers, which replaces the
             * derefs it built with plain indices. The emitter wants derefs
             * back, so they are rebuilt here against the split pair for the
             * unit the index names. */
            if (nir_tex_instr_src_index(tex, nir_tex_src_texture_deref) >= 0)
               continue;
            unsigned slot = tex->texture_index;
            unsigned k = 0;
            while (k < count && split_slot[k] != slot)
               ++k;
            if (k == count)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_tex_instr_add_src(
               tex, nir_tex_src_texture_deref,
               &nir_build_deref_var(&b, split_texture[k])->def);
            if (nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref) < 0 &&
                tex->op != nir_texop_txf && tex->op != nir_texop_txf_ms &&
                tex->op != nir_texop_txs && tex->op != nir_texop_query_levels)
               nir_tex_instr_add_src(
                  tex, nir_tex_src_sampler_deref,
                  &nir_build_deref_var(&b, split_sampler[k])->def);
            ++rewrites;
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }

   for (unsigned k = 0; k < count; ++k)
      exec_node_remove(&split_from[k]->node);
   /* A blit that does not filter -- a 1:1 copy is one -- fetches texels and
    * needs no sampler, so the sampler half of the pair goes unused. The
    * driver rejects a declared sampler nothing samples with, which is right:
    * it would be a binding the pipeline requires and no draw provides. */
   nir_remove_dead_variables(nir, nir_var_uniform, NULL);
   nir_fixup_deref_types(nir);
   fprintf(stderr,
           "gex-wgpu: split %u combined sampler(s), %u rebuilt, tex=%u\n",
           count, rewrites, tex_seen);
   return true;
}

/* Whether a varying slot reaches WGSL as an @location rather than a builtin.
 * ntv gives a location to everything its builtin switch does not claim, and
 * gl_Color / gl_TexCoord sit below VARYING_SLOT_VAR0 while still being
 * ordinary interpolated varyings -- excluding them left exactly the
 * mismatches this is meant to reconcile. */
static bool
gex_wgpu_is_location_varying(int location)
{
   switch (location) {
   case VARYING_SLOT_POS:
   case VARYING_SLOT_PSIZ:
   case VARYING_SLOT_CLIP_DIST0:
   case VARYING_SLOT_CLIP_DIST1:
   case VARYING_SLOT_CULL_DIST0:
   case VARYING_SLOT_CULL_DIST1:
   case VARYING_SLOT_LAYER:
   case VARYING_SLOT_VIEWPORT:
   case VARYING_SLOT_PRIMITIVE_ID:
   case VARYING_SLOT_EDGE:
   case VARYING_SLOT_FACE:
      return false;
   default:
      return location >= 0;
   }
}

static struct gex_wgpu_shader *
gex_wgpu_create_shader_state(struct pipe_context *pipe,
                             const struct pipe_shader_state *state,
                             enum mesa_shader_stage expected_stage)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   struct pipe_shader_state *mutable_state = (struct pipe_shader_state *)state;
   /* TGSI as well as NIR. Mesa's own helpers -- u_blitter's blit shaders come
    * through ureg, which hands the driver TGSI with no conversion of its own
    * -- are written that way regardless of what the driver prefers, so
    * refusing TGSI means refusing those helpers rather than refusing anything
    * an application wrote. */
   nir_shader *nir = state->type == PIPE_SHADER_IR_NIR ? state->ir.nir
      : state->type == PIPE_SHADER_IR_TGSI
      ? tgsi_to_nir(state->tokens, pipe->screen, false) : NULL;
   /* The two stages are compiled to separate WGSL modules, and WGSL requires
    * a varying's interpolation to match on both sides of the pipeline. TGSI
    * declares interpolation only on the fragment side, so a vertex output and
    * the fragment input it feeds can disagree -- u_blitter's texcoord arrives
    * noperspective in the fragment shader and flat out of the vertex shader,
    * which WebGPU rejects, and rejecting is a validation error rather than a
    * failed call.
    *
    * Smooth on both sides is the agreement to force. It is not a behaviour
    * change for the shaders this applies to: these blits draw a quad whose
    * vertices all have w = 1, where perspective-correct and linear
    * interpolation are the same thing. Integer varyings keep flat, which is
    * the only interpolation they may have. */
   if (nir && state->type == PIPE_SHADER_IR_TGSI) {
      const nir_variable_mode mode = nir->info.stage == MESA_SHADER_FRAGMENT
         ? nir_var_shader_in : nir_var_shader_out;
      nir_foreach_variable_with_modes(var, nir, mode) {
         const struct glsl_type *type = glsl_without_array(var->type);
         if (glsl_type_is_vector_or_scalar(type) &&
             glsl_get_base_type(type) == GLSL_TYPE_FLOAT)
            var->data.interpolation = INTERP_MODE_SMOOTH;
      }
   }
   /* Not TGSI-only. A GLSL `sampler2D` is itself a combined sampler, and
    * WebGPU has no such thing -- it wants a texture binding and a sampler
    * binding. Running this only for TGSI meant every shader st/mesa produced
    * from GLSL that samples a texture arrived with its derefs already lowered
    * away, so the texture metadata came out empty and the shader was rejected
    * as having "no dereferenced texture slots". That rejected all 162 cases
    * of texture_repeat_mode over an ordinary `texture(texture0, texCoord)`. */
   if (nir && !gex_wgpu_split_combined_samplers(nir)) {
      gex_wgpu_set_compile_error(
         mutable_state, "gex_wgpu could not split a combined sampler");
      ralloc_free(nir);
      return NULL;
   }
   if (!nir || nir->info.stage != expected_stage) {
      gex_wgpu_set_compile_error(mutable_state, "gex_wgpu requires matching NIR shader IR");
      if (nir)
         ralloc_free(nir);
      return NULL;
   }
   {
      char why[160] = "";
      if (!gex_wgpu_lower_gallium_ubos(nir, expected_stage,
                                       gex_screen(pipe->screen)
                                          ->max_uniform_buffers,
                                       why, sizeof(why))) {
         char detail[224];
         snprintf(detail, sizeof(detail),
                  "gex_wgpu requires vec4-backed, 16-byte-aligned Gallium UBO "
                  "loads: %s", why[0] ? why : "(no reason recorded)");
         gex_wgpu_set_compile_error(mutable_state, detail);
         ralloc_free(nir);
         return NULL;
      }
   }

   struct gex_wgpu_shader *shader = CALLOC_STRUCT(gex_wgpu_shader);
   if (!shader) {
      ralloc_free(nir);
      gex_wgpu_set_compile_error(mutable_state, "out of memory creating shader state");
      return NULL;
   }
   shader->stage = expected_stage;
   if (expected_stage == MESA_SHADER_VERTEX)
      shader->inputs_read = nir->info.inputs_read;

   /* What this stage says about each varying's interpolation, recorded by the
    * @location ntv will emit. The fragment stage's is the one GL means; the
    * vertex stage's only has to agree with it, which is what the variant
    * below arranges. */
   {
      const nir_variable_mode mode = expected_stage == MESA_SHADER_FRAGMENT
         ? nir_var_shader_in : nir_var_shader_out;
      nir_foreach_variable_with_modes(var, nir, mode) {
         const struct glsl_type *type = glsl_without_array(var->type);
         if (!gex_wgpu_is_location_varying(var->data.location) ||
             var->data.location >= GEX_WGPU_MAX_VARYING_LOCATIONS ||
             !glsl_type_is_vector_or_scalar(type))
            continue;
         shader->varying_interp[var->data.location] =
            (uint8_t)var->data.interpolation;
         shader->varying_interp_mask |= 1ull << var->data.location;
      }
   }

   /* A capturing vertex shader is kept as it is for rendering and cloned for
    * the compute variant that does the capturing, because WGSL will not let
    * the vertex stage write a storage buffer at all. The clone is taken now,
    * before the render path lowers anything into vertex-stage terms. */
   if (expected_stage == MESA_SHADER_VERTEX && nir->xfb_info &&
       nir->xfb_info->output_count)
      shader->capture_nir = nir_shader_clone(NULL, nir);
   /* And a clone for the compute variants that feed a geometry shader. Taken
    * unconditionally because whether one will be bound is not known here. */
   if (expected_stage == MESA_SHADER_VERTEX)
      shader->varying_nir = nir_shader_clone(NULL, nir);

   nir_foreach_variable_with_modes(var, nir,
                                   nir_var_mem_ubo | nir_var_mem_ssbo) {
      const unsigned expected_set =
         expected_stage == MESA_SHADER_VERTEX ? 1 : 2;
      const unsigned slot = var->data.driver_location;
      /* One block can appear as more than one NIR variable -- st/mesa emits a
       * variable per member for a block whose members were declared without an
       * instance name, which is how STK writes its UBOs. Both carry the same
       * slot and the same binding because they *are* the same buffer, so
       * recording it once and ignoring the repeats is what the binding code
       * already assumes. Only a genuine disagreement -- one slot claimed by
       * two bindings, or one binding by two slots -- is a conflict. */
      bool duplicate = false, same_buffer = false;
      unsigned clash = ~0u;
      for (unsigned i = 0; i < shader->ubo_count; ++i) {
         const bool slot_hit = shader->ubo_slots[i] == slot;
         const bool binding_hit = shader->ubo_bindings[i] == var->data.binding;
         if (slot_hit && binding_hit) {
            same_buffer = true;
            /* An SSBO and a UBO must never collapse onto one another: they are
             * bound as different resource types. */
            if (((shader->ubo_storage_mask >> i) & 1u) !=
                (var->data.mode == nir_var_mem_ssbo))
               same_buffer = false;
         } else if (slot_hit || binding_hit) {
            duplicate = true;
            if (clash == ~0u) clash = i;
         }
      }
      if (same_buffer && !duplicate)
         continue;
      if (shader->ubo_count >= GEX_WGPU_MAX_CONSTANT_BUFFERS ||
          slot >= GEX_WGPU_MAX_CONSTANT_BUFFERS ||
          duplicate || var->data.descriptor_set != expected_set) {
         /* Four different failures shared one message, which made a real
          * application's shader link an unreadable "invalid or duplicate".
          * Name which one, and the numbers behind it: the whole diagnosis of
          * a rejected link is which limit or which collision it hit. */
         char detail[192];
         snprintf(detail, sizeof(detail),
                  "gex_wgpu stage-group UBO rejected (%s): %s slot=%u "
                  "binding=%u set=%u want_set=%u count=%u max=%u",
                  shader->ubo_count >= GEX_WGPU_MAX_CONSTANT_BUFFERS
                     ? "too many for the stage group"
                     : slot >= GEX_WGPU_MAX_CONSTANT_BUFFERS
                        ? "slot above the stage-group maximum"
                        : duplicate ? "slot or binding already taken"
                                    : "wrong descriptor set",
                  var->data.mode == nir_var_mem_ssbo ? "ssbo" : "ubo",
                  slot, var->data.binding, var->data.descriptor_set,
                  expected_set, shader->ubo_count,
                  (unsigned)GEX_WGPU_MAX_CONSTANT_BUFFERS);
         if (duplicate)
            snprintf(detail + strlen(detail), sizeof(detail) - strlen(detail),
                     " name=%s clashes_with[%u] slot=%u binding=%u",
                     var->name ? var->name : "(anon)", clash,
                     shader->ubo_slots[clash], shader->ubo_bindings[clash]);
         ralloc_free(nir);
         FREE(shader);
         gex_wgpu_set_compile_error(mutable_state, detail);
         return NULL;
      }
      shader->ubo_slots[shader->ubo_count] = slot;
      shader->ubo_bindings[shader->ubo_count] = var->data.binding;
      if (var->data.mode == nir_var_mem_ssbo)
         shader->ubo_storage_mask |= 1u << shader->ubo_count;
      ++shader->ubo_count;
   }
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (glsl_type_is_texture(type)) {
         const unsigned slot = var->data.driver_location;
         const enum glsl_sampler_dim dim = glsl_get_sampler_dim(type);
         const bool is_array = glsl_sampler_type_is_array(type);
         const bool multisampled = dim == GLSL_SAMPLER_DIM_MS;
         const WGPUTextureViewDimension view_dimension =
            multisampled && !is_array
               ? WGPUTextureViewDimension_2D
            : dim == GLSL_SAMPLER_DIM_1D
               ? (is_array ? WGPUTextureViewDimension_2DArray
                           : WGPUTextureViewDimension_2D)
            : dim == GLSL_SAMPLER_DIM_RECT && !is_array
               ? WGPUTextureViewDimension_2D
            : dim == GLSL_SAMPLER_DIM_2D
               ? (is_array ? WGPUTextureViewDimension_2DArray
                           : WGPUTextureViewDimension_2D)
               : dim == GLSL_SAMPLER_DIM_3D && !is_array
               ? WGPUTextureViewDimension_3D
               : dim == GLSL_SAMPLER_DIM_CUBE
               ? (is_array ? WGPUTextureViewDimension_CubeArray
                           : WGPUTextureViewDimension_Cube)
               : WGPUTextureViewDimension_Undefined;
         bool duplicate = false;
         for (unsigned i = 0; i < shader->texture_count; ++i)
            duplicate |= shader->texture_slots[i] == slot ||
                         shader->texture_bindings[i] == var->data.binding;
         /* Set 0 for the fragment stage, 3 for the vertex stage -- the two
          * cannot share a group, since each shader numbers its own bindings
          * from zero. */
         const unsigned expected_texture_set =
            expected_stage == MESA_SHADER_VERTEX ? 3 : 0;
         if (shader->texture_count >= GEX_WGPU_MAX_SAMPLERS ||
             slot >= GEX_WGPU_MAX_SAMPLERS || duplicate ||
             view_dimension == WGPUTextureViewDimension_Undefined ||
             var->data.descriptor_set != expected_texture_set) {
            ralloc_free(nir);
            FREE(shader);
            gex_wgpu_set_compile_error(
               mutable_state, "gex_wgpu invalid or duplicate texture binding");
            return NULL;
         }
         shader->texture_slots[shader->texture_count] = slot;
         shader->texture_bindings[shader->texture_count] = var->data.binding;
         shader->texture_dimensions[shader->texture_count] = view_dimension;
         /* A multisampled texture is never filtered -- WGSL reaches it with
          * textureLoad and a sample index, not a sampler -- and WebGPU wants
          * the sample type to say so. */
         shader->texture_sample_types[shader->texture_count] = multisampled
            ? WGPUTextureSampleType_UnfilterableFloat
            : WGPUTextureSampleType_Float;
         if (multisampled)
            shader->multisampled_slots |= 1u << slot;
         ++shader->texture_count;
      } else if (glsl_type_is_bare_sampler(type)) {
         const unsigned slot = var->data.driver_location;
         bool duplicate = false;
         for (unsigned i = 0; i < shader->sampler_count; ++i)
            duplicate |= shader->sampler_slots[i] == slot ||
                         shader->sampler_bindings[i] == var->data.binding;
         if (shader->sampler_count >= GEX_WGPU_MAX_SAMPLERS ||
             slot >= GEX_WGPU_MAX_SAMPLERS || duplicate ||
             var->data.descriptor_set !=
                (expected_stage == MESA_SHADER_VERTEX ? 3u : 0u)) {
            ralloc_free(nir);
            FREE(shader);
            gex_wgpu_set_compile_error(
               mutable_state, "gex_wgpu invalid or duplicate sampler binding");
            return NULL;
         }
         shader->sampler_slots[shader->sampler_count] = slot;
         shader->sampler_bindings[shader->sampler_count] = var->data.binding;
         shader->sampler_types[shader->sampler_count] =
            WGPUSamplerBindingType_Filtering;
         ++shader->sampler_count;
      }
   }
   bool descriptors_valid = true;
   for (unsigned i = 0; descriptors_valid && i < shader->texture_count; ++i) {
      for (unsigned j = 0; j < shader->sampler_count; ++j)
         descriptors_valid &=
            shader->texture_bindings[i] != shader->sampler_bindings[j];
   }
   if (!descriptors_valid) {
      ralloc_free(nir);
      FREE(shader);
      gex_wgpu_set_compile_error(
         mutable_state,
         "gex_wgpu requires collision-free texture and sampler declarations");
      return NULL;
   }
   uint16_t texture_slots_used = 0, sampler_slots_used = 0;
   uint8_t shadow_slots = 0, float_slots = 0, uint_slots = 0, sint_slots = 0;
   nir_foreach_function(function, nir) {
      if (!function->impl)
         continue;
      nir_foreach_block(block, function->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            const int texture_src =
               nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
            if (texture_src < 0) {
               descriptors_valid = false;
               continue;
            }
            nir_variable *texture_var = nir_deref_instr_get_variable(
               nir_src_as_deref(tex->src[texture_src].src));
            const unsigned texture_slot = texture_var
               ? texture_var->data.driver_location : GEX_WGPU_MAX_SAMPLERS;
            if (texture_slot >= GEX_WGPU_MAX_SAMPLERS) {
               descriptors_valid = false;
               continue;
            }
            texture_slots_used |= 1u << texture_slot;
            const int sampler_src =
               nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
            if (sampler_src >= 0) {
               nir_variable *sampler_var = nir_deref_instr_get_variable(
                  nir_src_as_deref(tex->src[sampler_src].src));
               const unsigned sampler_slot = sampler_var
                  ? sampler_var->data.driver_location
                  : GEX_WGPU_MAX_SAMPLERS;
               if (sampler_slot != texture_slot) {
                  descriptors_valid = false;
                  continue;
               }
               sampler_slots_used |= 1u << sampler_slot;
            } else if (tex->op != nir_texop_txf &&
                       tex->op != nir_texop_txf_ms) {
               /* A plain fetch needs no sampler, and neither does a
                * multisampled one: WGSL reaches both with textureLoad. */
               descriptors_valid = false;
               continue;
            }
            if (tex->is_shadow)
               shadow_slots |= 1u << texture_slot;
            else if (nir_alu_type_get_base_type(tex->dest_type) ==
                     nir_type_float)
               float_slots |= 1u << texture_slot;
            else if (nir_alu_type_get_base_type(tex->dest_type) ==
                     nir_type_uint)
               uint_slots |= 1u << texture_slot;
            else if (nir_alu_type_get_base_type(tex->dest_type) ==
                     nir_type_int)
               sint_slots |= 1u << texture_slot;
            else
               descriptors_valid = false;
            if (sampler_src >= 0 &&
                (nir_alu_type_get_base_type(tex->dest_type) == nir_type_uint ||
                 nir_alu_type_get_base_type(tex->dest_type) == nir_type_int))
               shader->integer_sampled_slots |= 1u << texture_slot;
         }
      }
   }
   for (unsigned i = 0; i < shader->texture_count; ++i)
      descriptors_valid = descriptors_valid &&
         (texture_slots_used & (1u << shader->texture_slots[i]));
   for (unsigned i = 0; i < shader->sampler_count; ++i)
      descriptors_valid = descriptors_valid &&
         (sampler_slots_used & (1u << shader->sampler_slots[i]));
   for (unsigned slot = 0; slot < GEX_WGPU_MAX_SAMPLERS; ++slot) {
      if (!(sampler_slots_used & (1u << slot)))
         continue;
      bool declared = false;
      for (unsigned i = 0; i < shader->sampler_count; ++i)
         declared |= shader->sampler_slots[i] == slot;
      descriptors_valid &= declared;
   }
   descriptors_valid &= !(shadow_slots & (float_slots | uint_slots | sint_slots));
   descriptors_valid &= !(float_slots & (uint_slots | sint_slots));
   descriptors_valid &= !(uint_slots & sint_slots);
   for (unsigned i = 0; i < shader->texture_count; ++i) {
      const uint16_t slot_bit = 1u << shader->texture_slots[i];
      if (shadow_slots & slot_bit)
         shader->texture_sample_types[i] = WGPUTextureSampleType_Depth;
      else if (uint_slots & slot_bit)
         shader->texture_sample_types[i] = WGPUTextureSampleType_Uint;
      else if (sint_slots & slot_bit)
         shader->texture_sample_types[i] = WGPUTextureSampleType_Sint;
      /* WebGPU cube views cannot have integer sample types.  Integer cube
       * sampling is lowered to a 2D-array load in the state-specialized NIR
       * variant, over an alternate view of the same six-layer allocation. */
      if ((uint_slots | sint_slots) & slot_bit &&
          (shader->texture_dimensions[i] == WGPUTextureViewDimension_Cube ||
           shader->texture_dimensions[i] ==
              WGPUTextureViewDimension_CubeArray))
         shader->texture_dimensions[i] = WGPUTextureViewDimension_2DArray;
   }
   for (unsigned i = 0; i < shader->sampler_count; ++i) {
      const uint16_t slot_bit = 1u << shader->sampler_slots[i];
      if (shadow_slots & slot_bit)
         shader->sampler_types[i] = WGPUSamplerBindingType_Comparison;
      else if ((uint_slots | sint_slots) & slot_bit)
         shader->sampler_types[i] = WGPUSamplerBindingType_NonFiltering;
   }
   if (!descriptors_valid) {
      fprintf(stderr,
              "gex-wgpu: texture metadata invalid textures=%u samplers=%u "
              "texture-used=0x%02x sampler-used=0x%02x shadow=0x%02x "
              "float=0x%02x uint=0x%02x sint=0x%02x\n",
              shader->texture_count, shader->sampler_count,
              texture_slots_used, sampler_slots_used, shadow_slots,
              float_slots, uint_slots, sint_slots);
      ralloc_free(nir);
      FREE(shader);
      gex_wgpu_set_compile_error(
         mutable_state,
         "gex_wgpu requires dereferenced, consistently typed texture slots");
      return NULL;
   }
   if (expected_stage == MESA_SHADER_FRAGMENT) {
      shader->color_output_type = nir_type_float;
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
         if (var->data.location != FRAG_RESULT_DATA0 &&
             var->data.location != FRAG_RESULT_COLOR)
            continue;
         const struct glsl_type *type = glsl_without_array(var->type);
         shader->color_output_type =
            nir_get_nir_type_for_glsl_base_type(glsl_get_base_type(type));
         break;
      }
   }
   /* The @location numbers ntv will emit for this stage's inputs, taken from
    * the shader the modules are actually built from. Reading them earlier in
    * this function recorded a stale set: the lowering between there and here
    * can still change what the vertex stage declares, and a pipeline whose
    * vertex state is missing a location the module declares is a validation
    * error that poisons the device. */
   if (expected_stage == MESA_SHADER_VERTEX) {
      memset(shader->vertex_element_location, 0xff,
             sizeof(shader->vertex_element_location));
      nir_foreach_shader_in_variable(var, nir) {
         const struct glsl_type *bare = glsl_without_array(var->type);
         const unsigned slots = glsl_count_attribute_slots(var->type, true);
         const enum glsl_base_type base = glsl_get_base_type(bare);
         /* Two different numbers, and conflating them is the bug this
          * exists to avoid. Gallium's vertex elements are indexed by
          * st/mesa's driver_location; the @location the module declares is
          * assigned later by ntv_shader_prepare, which recomputes it from
          * var->data.location as "generic attribute number, or the
          * fixed-function slot for anything below GENERIC0". The two agree
          * only when a shader's generic attributes start at zero and run
          * contiguously -- so a shader binding attributes 1..3 declared
          * @location(1..3) while the pipeline offered 0..2, and WebGPU
          * rejected it as "vertex attribute slot 3 not present in the
          * VertexState". */
         const unsigned base_location =
            var->data.location >= VERT_ATTRIB_GENERIC0
               ? var->data.location - VERT_ATTRIB_GENERIC0 : var->data.location;
         for (unsigned slot = 0; slot < slots; ++slot) {
            const unsigned element = var->data.driver_location + slot;
            const unsigned location = base_location + slot;
            if (location >= 64)
               continue;
            shader->vertex_locations |= BITFIELD64_BIT(location);
            if (base == GLSL_TYPE_INT)
               shader->vertex_locations_sint |= BITFIELD64_BIT(location);
            else if (base == GLSL_TYPE_UINT)
               shader->vertex_locations_uint |= BITFIELD64_BIT(location);
            if (element < GEX_WGPU_MAX_VERTEX_ELEMENTS)
               shader->vertex_element_location[element] = (uint8_t)location;
         }
      }
   }
   /* Kept for the fragment stage's sampler and output variants, and for the
    * vertex stage's interpolation variant -- both rebuild the shader from
    * state that is only known once the other stage is bound. */
   shader->reads_draw_id =
      BITSET_TEST(nir->info.system_values_read, SYSTEM_VALUE_DRAW_ID);
   shader->original_nir =
      (expected_stage == MESA_SHADER_FRAGMENT ||
       expected_stage == MESA_SHADER_VERTEX)
      ? nir_shader_clone(NULL, nir) : NULL;
   if ((expected_stage == MESA_SHADER_FRAGMENT ||
        expected_stage == MESA_SHADER_VERTEX) && !shader->original_nir) {
      ralloc_free(nir);
      FREE(shader);
      gex_wgpu_set_compile_error(mutable_state,
                                 "out of memory cloning shader");
      return NULL;
   }
   nir_shader *flipped = (expected_stage == MESA_SHADER_VERTEX &&
                          !shader->integer_sampled_slots)
      ? nir_shader_clone(NULL, nir) : NULL;
   char *error = NULL;
   /* Integer sampling has to be lowered against the sampler state, which is
    * only known at draw time, so the base module is not built at all and the
    * variant path supplies it. The fragment stage has always done this; the
    * vertex stage used to compile eagerly and have Tint refuse the result --
    * "sampled image must have float component type" -- which reached the
    * application as a link error and failed 13 texture_swizzle cases. */
   const bool deferred_integer_sampling = shader->integer_sampled_slots != 0 &&
      (expected_stage == MESA_SHADER_FRAGMENT ||
       expected_stage == MESA_SHADER_VERTEX);
   if (deferred_integer_sampling)
      ralloc_free(nir);
   else
      shader->module = gex_wgpu_compile_nir(pipe, nir, expected_stage, false,
                                            &error);
   if (shader->module && expected_stage == MESA_SHADER_VERTEX) {
      if (flipped)
         shader->module_y_flipped = gex_wgpu_compile_nir(
            pipe, flipped, expected_stage, true, &error);
      else
         error = strdup("out of memory cloning vertex shader");
      flipped = NULL;
   }
   if ((!shader->module && !deferred_integer_sampling) ||
       (expected_stage == MESA_SHADER_VERTEX && !deferred_integer_sampling &&
        !shader->module_y_flipped)) {
      if (shader->module)
         wgpuShaderModuleRelease(shader->module);
      if (flipped && !shader->module_y_flipped)
         ralloc_free(flipped);
      if (shader->original_nir)
         ralloc_free(shader->original_nir);
      /* Name the stage. A rejection reaches the application as a link error
       * with no indication of which shader was at fault, and the two stages
       * fail for different reasons -- notably integer sampling, which the
       * fragment stage defers to a draw-time variant and the vertex stage
       * does not. */
      fprintf(stderr, "gex-wgpu: %s shader rejected (int_sampled=0x%x): %s\n",
              _mesa_shader_stage_to_string(expected_stage),
              shader->integer_sampled_slots,
              error ? error : "shader compilation failed");
      gex_wgpu_set_compile_error(mutable_state,
                                 error ? error : "shader compilation failed");
      FREE(error);
      FREE(shader);
      return NULL;
   }
   FREE(error);
   return shader;
}

static void *
gex_wgpu_create_vs_state(struct pipe_context *pipe,
                         const struct pipe_shader_state *state)
{
   return gex_wgpu_create_shader_state(pipe, state, MESA_SHADER_VERTEX);
}

static void *
gex_wgpu_create_fs_state(struct pipe_context *pipe,
                         const struct pipe_shader_state *state)
{
   return gex_wgpu_create_shader_state(pipe, state, MESA_SHADER_FRAGMENT);
}

static void
gex_wgpu_bind_vs_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   ((struct gex_wgpu_context *)pipe)->vs = state;
}

static void
gex_wgpu_bind_fs_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   ((struct gex_wgpu_context *)pipe)->fs = state;
}

static void
gex_wgpu_delete_shader_state(struct pipe_context *pipe, void *state)
{
   (void)pipe;
   struct gex_wgpu_shader *shader = state;
   if (!shader)
      return;
   if (shader->module)
      wgpuShaderModuleRelease(shader->module);
   if (shader->module_y_flipped)
      wgpuShaderModuleRelease(shader->module_y_flipped);
   struct gex_wgpu_shader_variant *variant = shader->variants;
   while (variant) {
      struct gex_wgpu_shader_variant *next = variant->next;
      if (variant->module)
         wgpuShaderModuleRelease(variant->module);
      FREE(variant);
      variant = next;
   }
   struct gex_wgpu_varying_variant *varying = shader->varyings;
   while (varying) {
      struct gex_wgpu_varying_variant *next = varying->next;
      if (varying->module)
         wgpuShaderModuleRelease(varying->module);
      FREE(varying);
      varying = next;
   }
   if (shader->varying_nir)
      ralloc_free(shader->varying_nir);
   struct gex_wgpu_capture_variant *capture = shader->captures;
   while (capture) {
      struct gex_wgpu_capture_variant *next = capture->next;
      if (capture->module)
         wgpuShaderModuleRelease(capture->module);
      FREE(capture);
      capture = next;
   }
   if (shader->capture_nir)
      ralloc_free(shader->capture_nir);
   if (shader->original_nir)
      ralloc_free(shader->original_nir);
   FREE(shader);
}

/* Declare the colour outputs with the base type the target expects.
 *
 * TGSI carries no type on a shader output, so a shader built from it declares
 * float whatever it actually writes -- u_blitter's integer texel fetch is the
 * case that matters. WebGPU rejects a pipeline whose fragment output type
 * disagrees with the colour target, and rejecting it is a validation error,
 * which poisons the device for every later command.
 *
 * This is a redeclaration, not a conversion: a NIR value is an untyped bit
 * pattern, so the same bits simply become correctly described. */
static void
gex_wgpu_retype_color_outputs(nir_shader *nir, nir_alu_type want)
{
   const enum glsl_base_type base =
      want == nir_type_uint ? GLSL_TYPE_UINT
      : want == nir_type_int ? GLSL_TYPE_INT : GLSL_TYPE_FLOAT;
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
      if (var->data.location != FRAG_RESULT_COLOR &&
          var->data.location < FRAG_RESULT_DATA0)
         continue;
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_vector_or_scalar(type))
         continue;
      var->type = glsl_vector_type(base, glsl_get_vector_elements(type));
   }
   nir_fixup_deref_types(nir);
}

static WGPUShaderModule
gex_wgpu_fragment_module(struct gex_wgpu_context *ctx)
{
   /* Declining, not latching. This returns NULL and draw_vbo checks it, so
    * the caller already learns of the failure -- latching on top ended the
    * context and every later map and upload in it, which is a cascade paid
    * for one shader this driver cannot build. */
#define GEX_FS_DECLINE(why)                                                   \
   do {                                                                       \
      static bool said;                                                        \
      if (!said) {                                                             \
         said = true;                                                          \
         fprintf(stderr, "gex-wgpu: fragment shader declined (%s)\n", why);    \
      }                                                                        \
   } while (0)

   struct gex_wgpu_shader *fs = ctx->fs;
   /* What the bound colour target needs the fragment output to be. */
   const nir_alu_type wanted_output =
      ctx->color_count == 0 ? nir_alu_type_get_base_type(fs->color_output_type)
      : util_format_is_pure_uint(ctx->color_formats[0]) ? nir_type_uint
      : util_format_is_pure_sint(ctx->color_formats[0]) ? nir_type_int
      : nir_type_float;
   const bool retype_outputs =
      wanted_output != nir_alu_type_get_base_type(fs->color_output_type);
   if (!fs->texture_count && !retype_outputs)
      return fs->module;
   uint16_t sampler_mask = 0;
   /* Kept apart on purpose: a retype needs a *variant*, but it does not need
    * the texture specialization. Folding the two together made every retyped
    * shader run sampler lowering it never asked for, which then refused. */
   bool needs_sampler_variant = false;
   for (unsigned i = 0; i < fs->texture_count; ++i) {
      const unsigned slot = fs->texture_slots[i];
      const struct gex_wgpu_sampler *sampler = ctx->samplers[slot];
      const struct pipe_sampler_view *view = ctx->sampler_views[slot];
      bool uses_sampler = false;
      for (unsigned j = 0; j < fs->sampler_count; ++j)
         uses_sampler |= fs->sampler_slots[j] == slot;
      /* An unbound slot is served by a 1x1 stand-in rather than refused; see
       * gex_wgpu_default_texture_view. It carries no swizzle, so the
       * specialization below has nothing to do for it. */
      (void)sampler;
      (void)uses_sampler;
      const bool swizzled = view &&
        (gex_wgpu_effective_view_swizzle(view, 0) != PIPE_SWIZZLE_X ||
         gex_wgpu_effective_view_swizzle(view, 1) != PIPE_SWIZZLE_Y ||
         gex_wgpu_effective_view_swizzle(view, 2) != PIPE_SWIZZLE_Z ||
         gex_wgpu_effective_view_swizzle(view, 3) != PIPE_SWIZZLE_W);
      /* A swizzled texture read only by texelFetch needs the swizzle too --
       * st/mesa gives one to every depth texture and to several sized
       * formats, and texture_size_promotion reads them with texelFetch.
       * gex_wgpu_lower_view_swizzle_instr owns that case; it takes the slot
       * from the texture deref when there is no sampler deref to take it
       * from. */
      if (!uses_sampler && swizzled)
         needs_sampler_variant = true;
      /* Every slot the lowering looks at belongs in the variant's key, not
       * just the ones with a sampler. This used to be set only inside the
       * `uses_sampler` branch below, so a texture read purely by texelFetch
       * left the mask at zero -- and a zero mask makes the match loop skip
       * every slot, which leaves `match` at its pre-loop value and hands back
       * the first cached variant for *any* later shader. One R8 texelFetch
       * with a reversed swizzle then answered for every texelFetch after it,
       * whatever its texture or swizzle: 225 of KHR-GL33.texture_swizzle's
       * functional cases came back with the first case's colours. */
      sampler_mask |= 1u << slot;
      /* The stand-in needs none of the checks below: they are all about
       * reconciling a real texture's target with a real sampler's state, and
       * an unbound slot has neither. */
      if (uses_sampler && view && sampler) {
         if ((view->target == PIPE_TEXTURE_RECT) !=
             sampler->state.unnormalized_coords) {
            GEX_FS_DECLINE("sampler state this driver cannot express");
            return NULL;
         }
         if (view->target == PIPE_TEXTURE_RECT &&
             (sampler->state.min_mip_filter != PIPE_TEX_MIPFILTER_NONE ||
              sampler->state.lod_bias != 0.0f ||
              (sampler->state.wrap_s != PIPE_TEX_WRAP_CLAMP_TO_EDGE &&
               sampler->state.wrap_s != PIPE_TEX_WRAP_CLAMP_TO_BORDER) ||
              (sampler->state.wrap_t != PIPE_TEX_WRAP_CLAMP_TO_EDGE &&
               sampler->state.wrap_t != PIPE_TEX_WRAP_CLAMP_TO_BORDER))) {
            GEX_FS_DECLINE("sampler state this driver cannot express");
            return NULL;
         }
         needs_sampler_variant |=
            (fs->integer_sampled_slots & (1u << slot)) ||
            sampler->state.wrap_s == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
            sampler->state.wrap_t == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
            sampler->state.wrap_r == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
            sampler->state.lod_bias != 0.0f ||
            swizzled;
      }
   }
   /* No gl_FragCoord correction here, deliberately.
    *
    * This driver sets `fs_coord_origin_upper_left` and clears
    * `fs_coord_origin_lower_left`, which tells st/mesa that fragment
    * coordinates arrive with WebGPU's top-down origin. st answers by running
    * nir_lower_wpos_ytransform itself, against a state variable it updates
    * with the bound framebuffer's orientation -- so the NIR handed to
    * create_fs_state already reads gl_FragCoord bottom-up on both
    * framebuffer 0 and an FBO, and so does every variant cloned from it.
    *
    * A second correction used to be applied here, `y' = H - y`, for the
    * window framebuffer only. Because a variant is built only for a shader
    * with a texture or a retyped output, that made gl_FragCoord depend on
    * whether the shader happened to sample anything: textureless shaders got
    * st's single transform and were right, textured ones got two and were
    * mirrored. In SuperTuxKart, whose scene composite addresses its render
    * target by gl_FragCoord and whose HUD does not, that showed up as the
    * whole 3D scene rendering upside down under a correctly-oriented HUD.
    *
    * tests/gl44-probe.sh fbo-orientation holds the line: it reads
    * gl_FragCoord.y back as a colour from two shaders that differ only in
    * whether they also sample a texture, and requires the same answer from
    * both. */
   const bool needs_variant = needs_sampler_variant || retype_outputs;
   if (!needs_variant)
      return fs->module;
   for (struct gex_wgpu_shader_variant *cached = fs->variants;
        cached; cached = cached->next) {
      bool match = cached->sampler_mask == sampler_mask &&
                   cached->color_output_type == wanted_output;
      for (unsigned slot = 0; match && slot < GEX_WGPU_MAX_SAMPLERS; ++slot) {
         if (!(sampler_mask & (1u << slot)))
            continue;
         const struct pipe_sampler_view *view = ctx->sampler_views[slot];
         const uint8_t swizzle[4] = {
            gex_wgpu_effective_view_swizzle(view, 0),
            gex_wgpu_effective_view_swizzle(view, 1),
            gex_wgpu_effective_view_swizzle(view, 2),
            gex_wgpu_effective_view_swizzle(view, 3),
         };
         /* `match &&`, not `match =`: assigning threw away the sampler_mask
          * and colour-output checks made before the loop, and every slot
          * compared before this one. */
         if (ctx->samplers[slot])
            match = match && memcmp(&cached->sampler_keys[slot],
                                    &ctx->samplers[slot]->state,
                                    sizeof(cached->sampler_keys[slot])) == 0;
         match = match && memcmp(cached->swizzles[slot], swizzle,
                                 sizeof(swizzle)) == 0;
      }
      if (match)
         return cached->module;
   }

   nir_shader *variant = fs->original_nir
      ? nir_shader_clone(NULL, fs->original_nir) : NULL;
   if (!variant) {
      gex_fail(ctx);
      return NULL;
   }
   if (retype_outputs)
      gex_wgpu_retype_color_outputs(variant, wanted_output);
   if (needs_sampler_variant && !gex_wgpu_lower_texture_variant(
          variant, fs, ctx->samplers, ctx->sampler_views)) {
      ralloc_free(variant);
      gex_fail(ctx);
      return NULL;
   }
   char *error = NULL;
   WGPUShaderModule module = gex_wgpu_compile_nir(
      &ctx->base, variant, MESA_SHADER_FRAGMENT, false, &error);
   if (!module) {
      fprintf(stderr, "gex-wgpu: fragment variant compile failed: %s\n",
              error ? error : "unknown");
      FREE(error);
      gex_fail(ctx);
      return NULL;
   }
   FREE(error);
   struct gex_wgpu_shader_variant *cached =
      CALLOC_STRUCT(gex_wgpu_shader_variant);
   if (!cached) {
      wgpuShaderModuleRelease(module);
      gex_fail(ctx);
      return NULL;
   }
   cached->module = module;
   cached->sampler_mask = sampler_mask;
   cached->color_output_type = wanted_output;
   for (unsigned slot = 0; slot < GEX_WGPU_MAX_SAMPLERS; ++slot) {
      if (!(sampler_mask & (1u << slot)))
         continue;
      const struct pipe_sampler_view *view = ctx->sampler_views[slot];
      if (ctx->samplers[slot])
         cached->sampler_keys[slot] = ctx->samplers[slot]->state;
      else
         memset(&cached->sampler_keys[slot], 0,
                sizeof(cached->sampler_keys[slot]));
      for (unsigned channel = 0; channel < 4; ++channel)
         cached->swizzles[slot][channel] =
            gex_wgpu_effective_view_swizzle(view, channel);
   }
   cached->next = fs->variants;
   fs->variants = cached;
   return module;
}

static void *
gex_wgpu_create_rasterizer_state(struct pipe_context *pipe,
                                 const struct pipe_rasterizer_state *state)
{
   GEX_ENTRY_TIMER();
   /* The fields this driver acts on, plus the ones it is safe to ignore.
    *
    * A refused rasterizer state is not a refused feature: cso binds it as
    * NULL and a NULL state reads as every default, so refusing one state
    * silently turns off scissoring, face culling, winding order and
    * rasterizer discard together. Every state st/mesa built was being refused
    * -- it sets point_quad_rasterization and sprite_coord_mode
    * unconditionally, and u_blitter sets flatshade -- so none of those four
    * had ever taken effect.
    *
    * The ignored group is point and line rasterization detail that WebGPU
    * fixes: a point is one pixel, so "rasterize points as quads" and the
    * sprite coordinate origin describe something that cannot happen, and the
    * line rules are not selectable. flatshade is fixed-function flat shading,
    * which st no longer asks for (caps->flatshade is false, so it lowers it
    * in NIR) and which only u_blitter still sets. multisample is a no-op on a
    * single-sample framebuffer, which is all this driver creates.
    *
    * Everything not listed -- polygon mode, polygon offset, smoothing,
    * stipple, clip planes, two-sided colour, colour clamping, the provoking
    * vertex -- would change pixels, and is still refused. */
   struct pipe_rasterizer_state supported = {0};
   supported.scissor = state->scissor;
   supported.front_ccw = state->front_ccw;
   supported.cull_face = state->cull_face;
   supported.rasterizer_discard = state->rasterizer_discard;
   supported.half_pixel_center = state->half_pixel_center;
   supported.depth_clip_near = state->depth_clip_near;
   supported.depth_clip_far = state->depth_clip_far;
   supported.flatshade = state->flatshade;
   supported.point_smooth = state->point_smooth;
   supported.sprite_coord_mode = state->sprite_coord_mode;
   supported.point_quad_rasterization = state->point_quad_rasterization;
   supported.point_line_tri_clip = state->point_line_tri_clip;
   supported.point_size_per_vertex = state->point_size_per_vertex;
   supported.multisample = state->multisample;
   supported.line_last_pixel = state->line_last_pixel;
   supported.line_rectangular = state->line_rectangular;
   supported.bottom_edge_rule = state->bottom_edge_rule;
   /* Line and point sizes, and the stipple pattern behind a disabled stipple.
    * WebGPU draws every line one pixel wide and every point one pixel, so
    * these describe something it does not offer rather than something this
    * driver is getting wrong -- and st/mesa sets line_stipple_pattern to
    * 0xffff whether or not stippling is on, which was refusing every state it
    * built. Stipple *enabled* is still refused, above. */
   supported.line_stipple_factor = state->line_stipple_factor;
   supported.line_stipple_pattern = state->line_stipple_pattern;
   supported.line_width = state->line_width;
   supported.point_size = state->point_size;
   supported.clip_halfz = state->clip_halfz;
   supported.depth_clamp = state->depth_clamp;
   supported.unclamped_fragment_depth_values =
      state->unclamped_fragment_depth_values;
   supported.rasterization_stream = state->rasterization_stream;
   supported.sprite_coord_enable = state->sprite_coord_enable;
   /* A refused rasterizer state binds as NULL, and a NULL one reads as every
    * default -- no scissor, no culling, and no rasterizer discard. That is
    * silent and wrong rather than loud and wrong, so say which field did it.
    * KHR-GL32.transform_feedback's discard cases rendered because of this. */
#define GEX_RASTERIZER_REFUSED(why)                                           \
   do {                                                                       \
      static bool said;                                                       \
      if (!said) {                                                            \
         said = true;                                                          \
         fprintf(stderr, "gex-wgpu: rasterizer state refused (%s)\n", why);   \
      }                                                                       \
      return NULL;                                                            \
   } while (0)
   if (state->cull_face & ~(PIPE_FACE_FRONT | PIPE_FACE_BACK))
      GEX_RASTERIZER_REFUSED("cull face");
   if (memcmp(state, &supported, sizeof(supported)) != 0) {
      const uint8_t *a = (const uint8_t *)state;
      const uint8_t *b = (const uint8_t *)&supported;
      size_t at = 0;
      while (at < sizeof(supported) && a[at] == b[at])
         ++at;
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: rasterizer state refused at byte %zu (0x%02x vs "
                 "0x%02x): fill=%u/%u offset=%u/%u/%u smooth=%u/%u "
                 "stipple=%u/%u twoside=%u clamp=%u/%u persample=%u "
                 "provoking_first=%u clip_planes=0x%x\n",
                 at, a[at], b[at], state->fill_front, state->fill_back,
                 state->offset_point, state->offset_line, state->offset_tri,
                 state->poly_smooth, state->line_smooth,
                 state->poly_stipple_enable, state->line_stipple_enable,
                 state->light_twoside, state->clamp_vertex_color,
                 state->clamp_fragment_color, state->force_persample_interp,
                 state->flatshade_first, state->clip_plane_enable);
      }
      return NULL;
   }
   /* GL clips near and far independently; WebGPU has one flag for both, so a
    * state that disables only one of them cannot be honoured. Nor can any of
    * it on a device without the feature. */
   if (state->depth_clip_near != state->depth_clip_far)
      GEX_RASTERIZER_REFUSED("depth clip near and far differ");
   if (!state->depth_clip_near && !gex_screen(pipe->screen)->depth_clip_control)
      GEX_RASTERIZER_REFUSED("depth clamp without depth-clip-control");
#undef GEX_RASTERIZER_REFUSED
   struct gex_wgpu_rasterizer *rasterizer =
      CALLOC_STRUCT(gex_wgpu_rasterizer);
   if (rasterizer) {
      rasterizer->scissor = state->scissor;
      rasterizer->front_ccw = state->front_ccw;
      rasterizer->cull_face = state->cull_face;
      rasterizer->discard = state->rasterizer_discard;
      rasterizer->unclipped_depth = !state->depth_clip_near;
   }
   return rasterizer;
}

static void
gex_wgpu_bind_rasterizer_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   ((struct gex_wgpu_context *)pipe)->rasterizer = state;
}

static void
gex_wgpu_delete_rasterizer_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   (void)pipe;
   FREE(state);
}

/* True when a stencil face's func and ops are ones WebGPU can name. */
static bool
gex_wgpu_stencil_face_ok(const struct pipe_stencil_state *face)
{
   return face->func >= PIPE_FUNC_NEVER && face->func <= PIPE_FUNC_ALWAYS &&
          face->fail_op <= PIPE_STENCIL_OP_INVERT &&
          face->zfail_op <= PIPE_STENCIL_OP_INVERT &&
          face->zpass_op <= PIPE_STENCIL_OP_INVERT;
}

static void *
gex_wgpu_create_dsa_state(struct pipe_context *pipe,
                          const struct pipe_depth_stencil_alpha_state *state)
{
   GEX_ENTRY_TIMER();
   (void)pipe;
   /* Checked field by field, not by memcmp against a rebuilt copy. The memcmp
    * made every *disabled* feature's leftover bytes part of the comparison,
    * and st/mesa fills the back face -- stencil[1] -- with the front face's
    * func, ops and masks whether or not two-sided stencil is on. So an
    * ordinary single-sided glStencilFunc never matched and the state was
    * refused, and nothing said so: cso stores the NULL, the draw runs with
    * ctx->dsa unset, and the pipeline is built with depth writes off and Keep
    * for every stencil op. The draw then quietly changes nothing at all --
    * which is how packed_depth_stencil lost eight cases whose only common
    * feature was that glEnable(GL_STENCIL_TEST) was on.
    *
    * Two-sided stencil is served now rather than refused: WebGPU has separate
    * stencilFront and stencilBack, so the only part of GL's two-sided state
    * it cannot express is a per-face mask, and that is what is checked here. */
   const bool two_sided = state->stencil[1].enabled;
   const char *why =
      state->alpha_enabled                              ? "alpha test"
      : state->depth_bounds_test                        ? "depth bounds test"
      : (state->depth_enabled &&
         (state->depth_func < PIPE_FUNC_NEVER ||
          state->depth_func > PIPE_FUNC_ALWAYS))        ? "depth compare"
      : (state->stencil[0].enabled &&
         !gex_wgpu_stencil_face_ok(&state->stencil[0])) ? "front stencil op"
      : (two_sided &&
         !gex_wgpu_stencil_face_ok(&state->stencil[1])) ? "back stencil op"
      /* One read mask and one write mask for both faces is all WebGPU has. */
      : (two_sided &&
         (state->stencil[0].valuemask != state->stencil[1].valuemask ||
          state->stencil[0].writemask != state->stencil[1].writemask))
                                                        ? "per-face stencil mask"
                                                        : NULL;
   if (why) {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: depth-stencil state refused (%s): depth=%d/%d "
                 "func=%u stencil=%d/%d funcs=%u/%u masks=%u/%u,%u/%u\n",
                 why, state->depth_enabled, state->depth_writemask,
                 state->depth_func, state->stencil[0].enabled,
                 state->stencil[1].enabled, state->stencil[0].func,
                 state->stencil[1].func, state->stencil[0].valuemask,
                 state->stencil[0].writemask, state->stencil[1].valuemask,
                 state->stencil[1].writemask);
      }
      return NULL;
   }
   struct pipe_depth_stencil_alpha_state *copy = MALLOC_STRUCT(
      pipe_depth_stencil_alpha_state);
   if (copy)
      *copy = *state;
   return copy;
}

static WGPUStencilOperation
gex_wgpu_stencil_operation(unsigned op)
{
   static const WGPUStencilOperation operations[] = {
      [PIPE_STENCIL_OP_KEEP] = WGPUStencilOperation_Keep,
      [PIPE_STENCIL_OP_ZERO] = WGPUStencilOperation_Zero,
      [PIPE_STENCIL_OP_REPLACE] = WGPUStencilOperation_Replace,
      [PIPE_STENCIL_OP_INCR] = WGPUStencilOperation_IncrementClamp,
      [PIPE_STENCIL_OP_DECR] = WGPUStencilOperation_DecrementClamp,
      [PIPE_STENCIL_OP_INCR_WRAP] = WGPUStencilOperation_IncrementWrap,
      [PIPE_STENCIL_OP_DECR_WRAP] = WGPUStencilOperation_DecrementWrap,
      [PIPE_STENCIL_OP_INVERT] = WGPUStencilOperation_Invert,
   };
   return op <= PIPE_STENCIL_OP_INVERT ? operations[op]
                                       : WGPUStencilOperation_Undefined;
}

static WGPUCompareFunction
gex_wgpu_compare(unsigned func)
{
   static const WGPUCompareFunction compare[] = {
      [PIPE_FUNC_NEVER] = WGPUCompareFunction_Never,
      [PIPE_FUNC_LESS] = WGPUCompareFunction_Less,
      [PIPE_FUNC_EQUAL] = WGPUCompareFunction_Equal,
      [PIPE_FUNC_LEQUAL] = WGPUCompareFunction_LessEqual,
      [PIPE_FUNC_GREATER] = WGPUCompareFunction_Greater,
      [PIPE_FUNC_NOTEQUAL] = WGPUCompareFunction_NotEqual,
      [PIPE_FUNC_GEQUAL] = WGPUCompareFunction_GreaterEqual,
      [PIPE_FUNC_ALWAYS] = WGPUCompareFunction_Always,
   };
   return func <= PIPE_FUNC_ALWAYS ? compare[func] : WGPUCompareFunction_Undefined;
}

static void
gex_wgpu_bind_dsa_state(struct pipe_context *pipe, void *state)
{
   ((struct gex_wgpu_context *)pipe)->dsa = state;
}

static void
gex_wgpu_set_stencil_ref(struct pipe_context *pipe,
                         const struct pipe_stencil_ref ref)
{
   GEX_ENTRY_TIMER();
   ((struct gex_wgpu_context *)pipe)->stencil_ref = ref;
}

static void
gex_wgpu_delete_dsa_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   (void)pipe;
   FREE(state);
}

static void *
gex_wgpu_create_blend_state(struct pipe_context *pipe,
                            const struct pipe_blend_state *state)
{
   GEX_ENTRY_TIMER();
   (void)pipe;
   struct pipe_blend_state supported = {0};
   (void)supported;
   /* Every GPUBlendFactor WebGPU has, less the dual-source four.
    *
    * This list used to stop after the two source-alpha factors and the four
    * constant ones, which refused blend states WebGPU can do perfectly well --
    * anything reading the destination, and the saturating factor. A refusal
    * here is not a declined draw: `create_blend_state` returning NULL leaves
    * st/mesa binding nothing, and a draw with no blend state writes its source
    * straight into the target. A transparent UI quad then arrives opaque.
    *
    * The dual-source factors stay out until the device is asked for the
    * `dual-source-blending` feature and the fragment shader declares
    * @blend_src, which is its own piece of work. */
   const unsigned factor_mask =
      BITFIELD_BIT(PIPE_BLENDFACTOR_ONE) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_ZERO) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_SRC_COLOR) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_INV_SRC_COLOR) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_SRC_ALPHA) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_INV_SRC_ALPHA) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_DST_COLOR) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_INV_DST_COLOR) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_DST_ALPHA) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_INV_DST_ALPHA) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_CONST_COLOR) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_INV_CONST_COLOR) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_CONST_ALPHA) |
      BITFIELD_BIT(PIPE_BLENDFACTOR_INV_CONST_ALPHA) |
      /* The dual-source four, only when the device granted the feature. A
       * pipeline naming them without it is a validation error, and a
       * validation error poisons the device rather than failing the call. */
      (gex_screen(pipe->screen)->dual_source_blending
          ? (BITFIELD_BIT(PIPE_BLENDFACTOR_SRC1_COLOR) |
             BITFIELD_BIT(PIPE_BLENDFACTOR_INV_SRC1_COLOR) |
             BITFIELD_BIT(PIPE_BLENDFACTOR_SRC1_ALPHA) |
             BITFIELD_BIT(PIPE_BLENDFACTOR_INV_SRC1_ALPHA))
          : 0u);
   supported.independent_blend_enable = state->independent_blend_enable;
   const unsigned state_count = state->independent_blend_enable
      ? GEX_WGPU_MAX_COLOR_TARGETS : 1;
   for (unsigned i = 0; i < state_count; ++i) {
      const struct pipe_rt_blend_state *rt = &state->rt[i];
      supported.rt[i] = *rt;
      const bool valid_func =
         rt->rgb_func == PIPE_BLEND_ADD ||
         rt->rgb_func == PIPE_BLEND_SUBTRACT ||
         rt->rgb_func == PIPE_BLEND_REVERSE_SUBTRACT ||
         rt->rgb_func == PIPE_BLEND_MIN || rt->rgb_func == PIPE_BLEND_MAX;
      const bool valid_alpha_func =
         rt->alpha_func == PIPE_BLEND_ADD ||
         rt->alpha_func == PIPE_BLEND_SUBTRACT ||
         rt->alpha_func == PIPE_BLEND_REVERSE_SUBTRACT ||
         rt->alpha_func == PIPE_BLEND_MIN || rt->alpha_func == PIPE_BLEND_MAX;
      const bool valid_factors =
         (factor_mask & BITFIELD_BIT(rt->rgb_src_factor)) &&
         (factor_mask & BITFIELD_BIT(rt->rgb_dst_factor)) &&
         (factor_mask & BITFIELD_BIT(rt->alpha_src_factor)) &&
         (factor_mask & BITFIELD_BIT(rt->alpha_dst_factor));
      if (rt->blend_enable &&
          (!valid_func || !valid_alpha_func || !valid_factors)) {
         /* Said out loud. This refusal costs the draw its blending and shows
          * up on screen as an opaque quad, with nothing anywhere reporting a
          * problem -- which is how it stayed hidden. */
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr,
                    "gex-wgpu: blend state refused (rt %u: %s) rgb=%u/%u/%u "
                    "alpha=%u/%u/%u\n", i,
                    !valid_func ? "rgb equation"
                       : !valid_alpha_func ? "alpha equation" : "factor",
                    rt->rgb_func, rt->rgb_src_factor, rt->rgb_dst_factor,
                    rt->alpha_func, rt->alpha_src_factor, rt->alpha_dst_factor);
         }
         return NULL;
      }
   }
   /* The state this driver does not model, named one field at a time.
    *
    * This used to be `memcmp(state, &supported, sizeof *state)`, which was
    * wrong twice. `pipe_blend_state` is a bitfield struct followed by an
    * array, so it has padding, and comparing a caller's struct against a
    * zero-initialised one compares that padding too -- a blend state could be
    * refused for bytes no field owns. And it caught `dither`, which is the
    * real damage: GL_DITHER is *enabled by default*, so st/mesa set it on
    * essentially every blend state SuperTuxKart created, every one of them was
    * refused, and the game rendered with no blending at all. Its transparent
    * HUD quads came out as opaque black rectangles.
    *
    * Dithering has no meaning here and none on any modern renderer -- WebGPU
    * has no control for it and Gallium drivers routinely ignore it -- so it is
    * ignored rather than refused. `max_rt` and the alpha-to-coverage dither
    * hint are likewise descriptive, not requests.
    *
    * `rt[]` past the first entry is deliberately not compared: with
    * independent blending off, Gallium says only `rt[0]` is meaningful, and
    * st/mesa does not promise the rest are zero. */
   const char *unmodelled =
      state->logicop_enable ? "logic op"
      : state->alpha_to_coverage ? "alpha to coverage"
      : state->alpha_to_one ? "alpha to one"
      : state->advanced_blend_func ? "advanced blend equation"
      : state->is_yuv ? "YUV target" : NULL;
   if (unmodelled) {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: blend state refused (%s) logicop=%u/%u a2c=%u "
                 "a2one=%u advanced=%u independent=%u max_rt=%u\n",
                 unmodelled, state->logicop_enable, state->logicop_func,
                 state->alpha_to_coverage, state->alpha_to_one,
                 state->advanced_blend_func,
                 state->independent_blend_enable, state->max_rt);
      }
      return NULL;
   }
   struct pipe_blend_state *copy = MALLOC_STRUCT(pipe_blend_state);
   if (copy)
      *copy = *state;
   return copy;
}

static WGPUBlendOperation
gex_wgpu_blend_operation(unsigned operation)
{
   switch (operation) {
   case PIPE_BLEND_ADD: return WGPUBlendOperation_Add;
   case PIPE_BLEND_SUBTRACT: return WGPUBlendOperation_Subtract;
   case PIPE_BLEND_REVERSE_SUBTRACT: return WGPUBlendOperation_ReverseSubtract;
   case PIPE_BLEND_MIN: return WGPUBlendOperation_Min;
   case PIPE_BLEND_MAX: return WGPUBlendOperation_Max;
   default: return WGPUBlendOperation_Undefined;
   }
}

static WGPUBlendFactor
gex_wgpu_blend_factor(unsigned factor)
{
   switch (factor) {
   case PIPE_BLENDFACTOR_ZERO: return WGPUBlendFactor_Zero;
   case PIPE_BLENDFACTOR_ONE: return WGPUBlendFactor_One;
   case PIPE_BLENDFACTOR_SRC_COLOR: return WGPUBlendFactor_Src;
   case PIPE_BLENDFACTOR_INV_SRC_COLOR: return WGPUBlendFactor_OneMinusSrc;
   case PIPE_BLENDFACTOR_SRC_ALPHA: return WGPUBlendFactor_SrcAlpha;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA:
      return WGPUBlendFactor_OneMinusSrcAlpha;
   case PIPE_BLENDFACTOR_DST_COLOR: return WGPUBlendFactor_Dst;
   case PIPE_BLENDFACTOR_INV_DST_COLOR: return WGPUBlendFactor_OneMinusDst;
   case PIPE_BLENDFACTOR_DST_ALPHA: return WGPUBlendFactor_DstAlpha;
   case PIPE_BLENDFACTOR_INV_DST_ALPHA:
      return WGPUBlendFactor_OneMinusDstAlpha;
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
      return WGPUBlendFactor_SrcAlphaSaturated;
   /* The second blend source, reachable only when `dual-source-blending` was
    * granted -- create_blend_state refuses these otherwise. */
   case PIPE_BLENDFACTOR_SRC1_COLOR: return WGPUBlendFactor_Src1;
   case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
      return WGPUBlendFactor_OneMinusSrc1;
   case PIPE_BLENDFACTOR_SRC1_ALPHA: return WGPUBlendFactor_Src1Alpha;
   case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
      return WGPUBlendFactor_OneMinusSrc1Alpha;
   /* GL's CONSTANT_ALPHA means "the constant's alpha in all three colour
    * channels", which WebGPU's `constant` -- the whole blend colour, per
    * channel -- does not express. It is right for the alpha equation and
    * wrong for the colour one; correcting that needs the blend constant
    * rewritten per draw, so it is left as it was and named here. */
   case PIPE_BLENDFACTOR_CONST_COLOR:
   case PIPE_BLENDFACTOR_CONST_ALPHA:
      return WGPUBlendFactor_Constant;
   case PIPE_BLENDFACTOR_INV_CONST_COLOR:
   case PIPE_BLENDFACTOR_INV_CONST_ALPHA:
      return WGPUBlendFactor_OneMinusConstant;
   default: return WGPUBlendFactor_Undefined;
   }
}

static void
gex_wgpu_set_blend_color(struct pipe_context *pipe,
                         const struct pipe_blend_color *color)
{
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (!color || !isfinite(color->color[0]) || !isfinite(color->color[1]) ||
       !isfinite(color->color[2]) || !isfinite(color->color[3])) {
      gex_fail(ctx);
      return;
   }
   ctx->blend_color = *color;
}

static void
gex_wgpu_bind_blend_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   ((struct gex_wgpu_context *)pipe)->blend = state;
}

static void
gex_wgpu_delete_blend_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   (void)pipe;
   FREE(state);
}

static void
gex_wgpu_set_viewport_states(struct pipe_context *pipe, unsigned start_slot,
                             unsigned num_viewports,
                             const struct pipe_viewport_state *states)
{
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   /* An all-zero viewport is what a caller restores when this context never
    * had one -- u_blitter saves the current viewport before its draw and puts
    * it back afterwards, whatever it was. That is not a drawable state and
    * not an error either; the draw path already refuses to draw without a
    * viewport, so record the absence rather than killing the context. */
   if (start_slot == 0 && num_viewports == 1 && states &&
       states[0].scale[0] == 0.0f && states[0].scale[1] == 0.0f &&
       states[0].scale[2] == 0.0f && states[0].translate[0] == 0.0f &&
       states[0].translate[1] == 0.0f && states[0].translate[2] == 0.0f) {
      ctx->viewport_set = false;
      return;
   }
   if (start_slot != 0 || num_viewports != 1 || !states ||
       !isfinite(states[0].scale[0]) || !isfinite(states[0].scale[1]) ||
       !isfinite(states[0].scale[2]) || !isfinite(states[0].translate[0]) ||
       !isfinite(states[0].translate[1]) || !isfinite(states[0].translate[2]) ||
       states[0].scale[0] <= 0.0f || states[0].scale[1] == 0.0f ||
       states[0].scale[2] < 0.0f ||
       states[0].swizzle_x != PIPE_VIEWPORT_SWIZZLE_POSITIVE_X ||
       states[0].swizzle_y != PIPE_VIEWPORT_SWIZZLE_POSITIVE_Y ||
       states[0].swizzle_z != PIPE_VIEWPORT_SWIZZLE_POSITIVE_Z ||
       states[0].swizzle_w != PIPE_VIEWPORT_SWIZZLE_POSITIVE_W) {
      if (!ctx->failed)
         fprintf(stderr,
                 "gex-wgpu: viewport refused slot=%u count=%u scale=%g,%g,%g "
                 "translate=%g,%g,%g\n", start_slot, num_viewports,
                 states ? states[0].scale[0] : 0.0f,
                 states ? states[0].scale[1] : 0.0f,
                 states ? states[0].scale[2] : 0.0f,
                 states ? states[0].translate[0] : 0.0f,
                 states ? states[0].translate[1] : 0.0f,
                 states ? states[0].translate[2] : 0.0f);
      gex_fail(ctx);
      return;
   }
   ctx->viewport = states[0];
   ctx->viewport_set = true;
}

static void
gex_wgpu_set_scissor_states(struct pipe_context *pipe, unsigned start_slot,
                            unsigned num_scissors,
                            const struct pipe_scissor_state *states)
{
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (start_slot != 0 || num_scissors != 1 || !states ||
       states[0].minx > states[0].maxx || states[0].miny > states[0].maxy) {
      gex_fail(ctx);
      return;
   }
   ctx->scissor = states[0];
   ctx->scissor_set = true;
}

static WGPUVertexFormat
gex_wgpu_vertex_format(enum pipe_format format, unsigned *size,
                       unsigned *alignment)
{
   switch (format) {
   case PIPE_FORMAT_R32_FLOAT:
      *size = 4; *alignment = 4; return WGPUVertexFormat_Float32;
   case PIPE_FORMAT_R32G32_FLOAT:
      *size = 8; *alignment = 4; return WGPUVertexFormat_Float32x2;
   case PIPE_FORMAT_R32G32B32_FLOAT:
      *size = 12; *alignment = 4; return WGPUVertexFormat_Float32x3;
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
      *size = 16; *alignment = 4; return WGPUVertexFormat_Float32x4;
   case PIPE_FORMAT_R32_UINT:
      *size = 4; *alignment = 4; return WGPUVertexFormat_Uint32;
   case PIPE_FORMAT_R32G32_UINT:
      *size = 8; *alignment = 4; return WGPUVertexFormat_Uint32x2;
   case PIPE_FORMAT_R32G32B32_UINT:
      *size = 12; *alignment = 4; return WGPUVertexFormat_Uint32x3;
   case PIPE_FORMAT_R32G32B32A32_UINT:
      *size = 16; *alignment = 4; return WGPUVertexFormat_Uint32x4;
   case PIPE_FORMAT_R32_SINT:
      *size = 4; *alignment = 4; return WGPUVertexFormat_Sint32;
   case PIPE_FORMAT_R32G32_SINT:
      *size = 8; *alignment = 4; return WGPUVertexFormat_Sint32x2;
   case PIPE_FORMAT_R32G32B32_SINT:
      *size = 12; *alignment = 4; return WGPUVertexFormat_Sint32x3;
   case PIPE_FORMAT_R32G32B32A32_SINT:
      *size = 16; *alignment = 4; return WGPUVertexFormat_Sint32x4;
   case PIPE_FORMAT_R16G16_FLOAT:
      *size = 4; *alignment = 2; return WGPUVertexFormat_Float16x2;
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
      *size = 8; *alignment = 2; return WGPUVertexFormat_Float16x4;
   case PIPE_FORMAT_R16G16_UNORM:
      *size = 4; *alignment = 2; return WGPUVertexFormat_Unorm16x2;
   case PIPE_FORMAT_R16G16B16A16_UNORM:
      *size = 8; *alignment = 2; return WGPUVertexFormat_Unorm16x4;
   case PIPE_FORMAT_R16G16_SNORM:
      *size = 4; *alignment = 2; return WGPUVertexFormat_Snorm16x2;
   case PIPE_FORMAT_R16G16B16A16_SNORM:
      *size = 8; *alignment = 2; return WGPUVertexFormat_Snorm16x4;
   case PIPE_FORMAT_R16G16_UINT:
      *size = 4; *alignment = 2; return WGPUVertexFormat_Uint16x2;
   case PIPE_FORMAT_R16G16B16A16_UINT:
      *size = 8; *alignment = 2; return WGPUVertexFormat_Uint16x4;
   case PIPE_FORMAT_R16G16_SINT:
      *size = 4; *alignment = 2; return WGPUVertexFormat_Sint16x2;
   case PIPE_FORMAT_R16G16B16A16_SINT:
      *size = 8; *alignment = 2; return WGPUVertexFormat_Sint16x4;
   case PIPE_FORMAT_R8G8_UNORM:
      *size = 2; *alignment = 1; return WGPUVertexFormat_Unorm8x2;
   case PIPE_FORMAT_R8G8B8A8_UNORM:
      *size = 4; *alignment = 1; return WGPUVertexFormat_Unorm8x4;
   /* GL_BGRA vertex arrays, which is EXT_vertex_array_bgra -- st/mesa gates
    * that extension on exactly this format being fetchable, and the extension
    * is the *only* thing this driver was missing between GL 3.1 and GL 3.2.
    * Being stuck at 3.1 costs more than a version number: the project's own
    * GL 4.4 probes are the only self-checking ground truth there is for this
    * path, and they refuse to compile against GLSL 1.40. WebGPU spells the
    * swizzle into the format rather than needing a shader variant. */
   case PIPE_FORMAT_B8G8R8A8_UNORM:
      *size = 4; *alignment = 1; return WGPUVertexFormat_Unorm8x4BGRA;
   case PIPE_FORMAT_R8G8_SNORM:
      *size = 2; *alignment = 1; return WGPUVertexFormat_Snorm8x2;
   case PIPE_FORMAT_R8G8B8A8_SNORM:
      *size = 4; *alignment = 1; return WGPUVertexFormat_Snorm8x4;
   case PIPE_FORMAT_R8G8_UINT:
      *size = 2; *alignment = 1; return WGPUVertexFormat_Uint8x2;
   case PIPE_FORMAT_R8G8B8A8_UINT:
      *size = 4; *alignment = 1; return WGPUVertexFormat_Uint8x4;
   case PIPE_FORMAT_R8G8_SINT:
      *size = 2; *alignment = 1; return WGPUVertexFormat_Sint8x2;
   case PIPE_FORMAT_R8G8B8A8_SINT:
      *size = 4; *alignment = 1; return WGPUVertexFormat_Sint8x4;
   case PIPE_FORMAT_R10G10B10A2_UNORM:
      *size = 4; *alignment = 4; return WGPUVertexFormat_Unorm10_10_10_2;
   /* The rest of ARB_vertex_type_2_10_10_10_rev. st/mesa wants all eight of
    * these before it will expose the extension, and WebGPU has exactly one of
    * them (the Unorm above). The other seven are fetched as a raw dword and
    * unpacked in the vertex shader -- see gex_wgpu_vertex_unpack_kind and
    * gex_wgpu_lower_vertex_attrib_unpack. v3d does the same thing.
    *
    * u_vbuf cannot cover this: these formats are absent from
    * vbuf_format_fallbacks, so format_translation leaves them alone and the
    * element is passed through as-is. And CPU translation would not help
    * anyway -- st_extensions.c asks is_format_supported directly, so a format
    * declined here is a format GL never offers. */
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_R10G10B10A2_SNORM:
   case PIPE_FORMAT_B10G10R10A2_SNORM:
   case PIPE_FORMAT_R10G10B10A2_USCALED:
   case PIPE_FORMAT_B10G10R10A2_USCALED:
   case PIPE_FORMAT_R10G10B10A2_SSCALED:
   case PIPE_FORMAT_B10G10R10A2_SSCALED:
      *size = 4; *alignment = 4; return WGPUVertexFormat_Uint32;
   default:
      *size = 0; *alignment = 0; return (WGPUVertexFormat)0;
   }
}

/* Which shader-side unpack a vertex format needs, or 0 for none. The low
 * bits name the conversion, GEX_WGPU_UNPACK_BGR flags the ones whose red and
 * blue channels are swapped relative to the packed order. */
enum {
   GEX_WGPU_UNPACK_NONE = 0,
   GEX_WGPU_UNPACK_UNORM,
   GEX_WGPU_UNPACK_SNORM,
   GEX_WGPU_UNPACK_USCALED,
   GEX_WGPU_UNPACK_SSCALED,
   GEX_WGPU_UNPACK_KIND_MASK = 0x0f,
   GEX_WGPU_UNPACK_BGR = 0x10,
};

static uint8_t
gex_wgpu_vertex_unpack_kind(enum pipe_format format)
{
   switch (format) {
   /* R10G10B10A2_UNORM is deliberately absent: WebGPU fetches it natively as
    * Unorm10_10_10_2, so it needs no shader work and no variant. */
   case PIPE_FORMAT_B10G10R10A2_UNORM:
      return GEX_WGPU_UNPACK_UNORM | GEX_WGPU_UNPACK_BGR;
   case PIPE_FORMAT_R10G10B10A2_SNORM:   return GEX_WGPU_UNPACK_SNORM;
   case PIPE_FORMAT_B10G10R10A2_SNORM:
      return GEX_WGPU_UNPACK_SNORM | GEX_WGPU_UNPACK_BGR;
   case PIPE_FORMAT_R10G10B10A2_USCALED: return GEX_WGPU_UNPACK_USCALED;
   case PIPE_FORMAT_B10G10R10A2_USCALED:
      return GEX_WGPU_UNPACK_USCALED | GEX_WGPU_UNPACK_BGR;
   case PIPE_FORMAT_R10G10B10A2_SSCALED: return GEX_WGPU_UNPACK_SSCALED;
   case PIPE_FORMAT_B10G10R10A2_SSCALED:
      return GEX_WGPU_UNPACK_SSCALED | GEX_WGPU_UNPACK_BGR;
   default:                              return GEX_WGPU_UNPACK_NONE;
   }
}

/* Rewrite the flagged vertex inputs from a fetched vec4 into the raw dword
 * WebGPU actually delivers, plus the arithmetic that reconstitutes the value.
 *
 * The inputs are still variables and derefs at this point, not load_input
 * intrinsics, so the variable type, the deref type and the load's component
 * count all have to be corrected together -- leaving any one of them at vec4
 * makes ntv emit a module whose declared @location type disagrees with the
 * pipeline's vertex format, which WebGPU rejects and which poisons the device
 * for every later command. */
static bool
gex_wgpu_lower_vertex_attrib_unpack(nir_shader *nir, const uint8_t *kinds)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl)
      return false;

   bool any = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in) {
      const unsigned element = var->data.driver_location;
      if (element < GEX_WGPU_MAX_VERTEX_ELEMENTS && kinds[element]) {
         var->type = glsl_uint_type();
         any = true;
      }
   }
   if (!any)
      return false;

   const unsigned bits[4] = {10, 10, 10, 2};
   nir_builder b = nir_builder_create(impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_deref) {
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type == nir_deref_type_var &&
                deref->var->data.mode == nir_var_shader_in) {
               const unsigned element = deref->var->data.driver_location;
               if (element < GEX_WGPU_MAX_VERTEX_ELEMENTS && kinds[element])
                  deref->type = deref->var->type;
            }
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *load = nir_instr_as_intrinsic(instr);
         if (load->intrinsic != nir_intrinsic_load_deref)
            continue;
         nir_deref_instr *deref = nir_src_as_deref(load->src[0]);
         if (!deref || deref->deref_type != nir_deref_type_var ||
             deref->var->data.mode != nir_var_shader_in)
            continue;
         const unsigned element = deref->var->data.driver_location;
         if (element >= GEX_WGPU_MAX_VERTEX_ELEMENTS || !kinds[element])
            continue;

         /* What the shader asked for, before the load is narrowed. */
         const unsigned wanted = load->def.num_components;
         const uint8_t kind = kinds[element] & GEX_WGPU_UNPACK_KIND_MASK;
         const bool bgr = (kinds[element] & GEX_WGPU_UNPACK_BGR) != 0;
         const bool is_signed = kind == GEX_WGPU_UNPACK_SNORM ||
                                kind == GEX_WGPU_UNPACK_SSCALED;

         load->num_components = 1;
         load->def.num_components = 1;
         load->def.bit_size = 32;

         b.cursor = nir_after_instr(instr);
         nir_def *packed = &load->def;
         nir_def *parts = is_signed
            ? nir_format_unpack_sint(&b, packed, bits, 4)
            : nir_format_unpack_uint(&b, packed, bits, 4);
         nir_def *value;
         switch (kind) {
         case GEX_WGPU_UNPACK_UNORM:
            value = nir_format_unorm_to_float(&b, parts, bits);
            break;
         case GEX_WGPU_UNPACK_SNORM:
            value = nir_format_snorm_to_float(&b, parts, bits);
            break;
         case GEX_WGPU_UNPACK_USCALED:
            value = nir_u2f32(&b, parts);
            break;
         default:
            value = nir_i2f32(&b, parts);
            break;
         }
         if (bgr) {
            /* Only the colour channels swap; alpha stays where it is. */
            const unsigned swizzle[4] = {2, 1, 0, 3};
            value = nir_swizzle(&b, value, swizzle, 4);
         }
         if (wanted != 4)
            value = nir_trim_vector(&b, value, wanted);
         nir_def_rewrite_uses_after(&load->def, value);
         progress = true;
      }
   }

   nir_progress(progress, impl, nir_metadata_control_flow);
   return progress;
}

static void *
gex_wgpu_create_vertex_elements_state(struct pipe_context *pipe,
                                      unsigned num_elements,
                                      const struct pipe_vertex_element *elements)
{
   GEX_ENTRY_TIMER();
   (void)pipe;
   if (num_elements > GEX_WGPU_MAX_VERTEX_ELEMENTS ||
       (num_elements && !elements)) {
      fprintf(stderr, "gex-wgpu: vertex elements refused: n=%u elements=%d\n",
              num_elements, elements ? 1 : 0);
      return NULL;
   }
   /* Zero elements is a legal state, not a failure: st/mesa creates one for a
    * program that reads no vertex attributes, and returning NULL for it makes
    * every later draw look like it has no vertex layout at all. */
   if (num_elements == 0) {
      struct gex_wgpu_vertex_elements *empty =
         CALLOC_STRUCT(gex_wgpu_vertex_elements);
      return empty;
   }
   bool buffers_used[8] = {0};
   unsigned highest_buffer = 0;
   for (unsigned i = 0; i < num_elements; ++i) {
      unsigned size, alignment;
      if (!gex_wgpu_vertex_format(elements[i].src_format, &size, &alignment)) {
         fprintf(stderr,
                 "gex-wgpu: vertex element %u format %s (%u) unsupported\n",
                 i, util_format_short_name(elements[i].src_format),
                 elements[i].src_format);
         return NULL;
      }
      /* A zero stride is a constant attribute -- every vertex reads the same
       * element -- and WebGPU expresses it directly: arrayStride 0 is legal,
       * and the offset is then bounded by maxVertexBufferArrayStride rather
       * than by the stride. Refusing it rejected a whole vertex-elements
       * state, which left nothing bound and turned an ordinary draw into a
       * latched context. */
      const bool packed = elements[i].src_stride == 0;
      if (elements[i].vertex_buffer_index >=
             GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS || elements[i].dual_slot ||
          elements[i].instance_divisor > 1 ||
          elements[i].src_stride % 4 ||
          elements[i].src_offset % alignment ||
          (!packed &&
           (uint64_t)elements[i].src_offset + size > elements[i].src_stride)) {
         fprintf(stderr,
                 "gex-wgpu: vertex element %u refused: vb=%u dual=%d "
                 "divisor=%u stride=%u offset=%u size=%u align=%u\n",
                 i, elements[i].vertex_buffer_index, (int)elements[i].dual_slot,
                 elements[i].instance_divisor, elements[i].src_stride,
                 elements[i].src_offset, size, alignment);
         return NULL;
      }
      const unsigned buffer = elements[i].vertex_buffer_index;
      buffers_used[buffer] = true;
      highest_buffer = MAX2(highest_buffer, buffer);
      for (unsigned previous = 0; previous < i; ++previous) {
         if (elements[previous].vertex_buffer_index != buffer)
            continue;
         /* Two elements on one buffer with *different step rates* are fine:
          * the draw path gives each step rate its own WebGPU vertex buffer
          * slot, bound to the same buffer. Refusing them was this driver's
          * most expensive bug.
          *
          * SuperTuxKart's object VAO is exactly that shape -- `i_position`,
          * `i_normal` and `i_color` per vertex, `i_origin`, `i_rotation`,
          * `i_scale`, `i_texture_trans` and `i_misc_data` per instance,
          * `glVertexAttribDivisor(8..12, 1)`. The whole vertex-elements state
          * was rejected, and because `create_vertex_elements_state` returning
          * NULL is not a refused draw, st/mesa simply left the previous state
          * bound. Every draw of the scene then ran with some earlier
          * non-instanced layout: the per-instance attributes were fetched per
          * vertex from the wrong offsets, `i_scale` came out zero, and since
          * the shader computes `i_position * i_scale + i_origin` every object
          * collapsed onto a point. Ninety thousand draws a frame rasterised
          * nothing, and the only thing on screen was the skybox and the HUD.
          *
          * It returned NULL without a word, which is why it took so long to
          * find. Both refusals below say what they refused. */
         if (elements[previous].instance_divisor !=
                elements[i].instance_divisor)
            continue;
         if (elements[previous].src_stride != elements[i].src_stride) {
            /* Same buffer, same step rate, different strides: one WebGPU
             * vertex buffer layout carries a single arrayStride, so this
             * genuinely cannot be expressed. */
            fprintf(stderr,
                    "gex-wgpu: vertex elements refused: elements %u and %u "
                    "share buffer %u at step rate %u but have strides %u and "
                    "%u\n", previous, i, buffer,
                    elements[i].instance_divisor,
                    elements[previous].src_stride, elements[i].src_stride);
            return NULL;
         }
      }
   }
   for (unsigned buffer = 0; buffer <= highest_buffer; ++buffer)
      if (!buffers_used[buffer]) {
         fprintf(stderr,
                 "gex-wgpu: vertex buffer slot %u unreferenced below "
                 "highest %u\n", buffer, highest_buffer);
         return NULL;
      }
   struct gex_wgpu_vertex_elements *state =
      CALLOC_STRUCT(gex_wgpu_vertex_elements);
   if (state) {
      state->count = num_elements;
      state->buffer_count = highest_buffer + 1;
      memcpy(state->elements, elements, num_elements * sizeof(*elements));
      for (unsigned i = 0; i < num_elements; ++i) {
         state->unpack[i] =
            gex_wgpu_vertex_unpack_kind(elements[i].src_format);
         state->any_unpack |= state->unpack[i] != GEX_WGPU_UNPACK_NONE;
      }
   }
   return state;
}

static void
gex_wgpu_bind_vertex_elements_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   ((struct gex_wgpu_context *)pipe)->vertex_elements = state;
}

static void
gex_wgpu_delete_vertex_elements_state(struct pipe_context *pipe, void *state)
{
   GEX_ENTRY_TIMER();
   (void)pipe;
   FREE(state);
}

static void
gex_wgpu_set_vertex_buffers(struct pipe_context *pipe, unsigned count,
                            const struct pipe_vertex_buffer *buffers)
{
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   /* Kept whole for u_blitter, which restores by handing this straight back. */
   for (unsigned i = 0; i < ARRAY_SIZE(ctx->vertex_buffer_state); ++i) {
      const struct pipe_vertex_buffer *from =
         buffers && i < count ? &buffers[i] : NULL;
      if (from)
         pipe_vertex_buffer_reference(&ctx->vertex_buffer_state[i], from);
      else
         pipe_vertex_buffer_unreference(&ctx->vertex_buffer_state[i]);
   }
   if (count == 0) {
      for (unsigned i = 0; i < ctx->vertex_buffer_count; ++i) {
         pipe_resource_reference(&ctx->vertex_buffers[i], NULL);
         ctx->vertex_buffer_offsets[i] = 0;
      }
      ctx->vertex_buffer_count = 0;
      return;
   }
   if (count > GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS || !buffers) {
      /* Latches the context, so it is the last thing that happens to this
       * application -- and the caps already tell st/mesa the limit is
       * GEX_WGPU_MAX_VERTEX_BUFFERS, so arriving here at all means something
       * bypassed u_vbuf. Say which of the two it was. */
      fprintf(stderr,
              "gex-wgpu: vertex buffers refused (%s) count=%u max=%u\n",
              !buffers ? "no buffer array" : "more buffers than the maximum",
              count, (unsigned)GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS);
      gex_fail(ctx);
      return;
   }
   for (unsigned i = 0; i < count; ++i) {
      /* An empty slot is Gallium's way of unbinding that slot, not an error.
       * Refusing it meant any caller that unbinds a subset -- u_blitter
       * restoring the slots it did not use is one -- killed the context. The
       * binding loop below already treats a null resource as an unbind. */
      if (!buffers[i].buffer.resource)
         continue;
      const char *why =
         buffers[i].is_user_buffer            ? "user buffer"
         : buffers[i].buffer.resource->target != PIPE_BUFFER ? "not a buffer"
         : !(buffers[i].buffer.resource->bind &
             (PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_VERTEX_STATE))
                                              ? "not bound as a vertex buffer"
         : buffers[i].buffer_offset > buffers[i].buffer.resource->width0
                                              ? "offset past the end" : NULL;
      if (why) {
         if (!ctx->failed)
            fprintf(stderr,
                    "gex-wgpu: vertex buffer %u refused (%s) offset=%u "
                    "bind=0x%x\n", i, why, buffers[i].buffer_offset,
                    buffers[i].buffer.resource
                       ? buffers[i].buffer.resource->bind : 0);
         gex_fail(ctx);
         return;
      }
   }
   for (unsigned i = 0; i < count; ++i) {
      pipe_resource_reference(&ctx->vertex_buffers[i],
                              buffers[i].buffer.resource);
      ctx->vertex_buffer_offsets[i] = buffers[i].buffer_offset;
   }
   for (unsigned i = count; i < ctx->vertex_buffer_count; ++i) {
      pipe_resource_reference(&ctx->vertex_buffers[i], NULL);
      ctx->vertex_buffer_offsets[i] = 0;
   }
   ctx->vertex_buffer_count = count;
}


/* The staged uniform bytes nobody has told the queue about yet. */
static void
gex_wgpu_flush_constant_ring(struct gex_wgpu_context *ctx)
{
   if (ctx->constant_dirty_hi <= ctx->constant_dirty_lo || !ctx->constant_ring)
      return;
   const unsigned lo = ctx->constant_dirty_lo & ~3u;
   const unsigned hi = ALIGN_POT(ctx->constant_dirty_hi, 4u);
   wgpuQueueWriteBuffer(gex_screen(ctx->base.screen)->queue,
                        gex_resource(ctx->constant_ring)->buffer, lo,
                        ctx->constant_staging + lo, hi - lo);
   ctx->constant_dirty_lo = 0;
   ctx->constant_dirty_hi = 0;
}

static void
gex_wgpu_batch_touch(struct gex_wgpu_context *ctx,
                     struct pipe_resource *resource)
{
   if (!resource)
      return;
   struct gex_wgpu_resource *gex = gex_resource(resource);
   if (gex->in_batch)
      return;
   if (ctx->batch_touched_count >= ARRAY_SIZE(ctx->batch_touched)) {
      /* Nowhere left to hold a reference, so the batch can no longer promise
       * this resource outlives it. The draw finishes and the caller submits.
       */
      ctx->batch_touched_overflowed = true;
      return;
   }
   gex->in_batch = true;
   struct pipe_resource **slot = &ctx->batch_touched[ctx->batch_touched_count];
   *slot = NULL;
   pipe_resource_reference(slot, resource);
   ++ctx->batch_touched_count;
}

/* Whether an operation on this resource has to wait for the recorded draws. */
static bool
gex_wgpu_batch_conflicts(struct gex_wgpu_context *ctx,
                         struct pipe_resource *resource)
{
   return ctx->pending_encoder &&
          (!resource || gex_resource(resource)->in_batch);
}

/* The same question for a transfer, which can answer it itself.
 *
 * PIPE_MAP_UNSYNCHRONIZED is the caller promising that the range it touches
 * is not one the GPU is still reading -- it is how u_upload_mgr appends to
 * the buffer every client-side vertex array is streamed through. Taking that
 * promise at its word is the difference between batching and not: Minecraft
 * draws from that one buffer constantly, so treating its uploads as
 * conflicts broke the batch 771623 times. */
static bool
gex_wgpu_batch_conflicts_usage(struct gex_wgpu_context *ctx,
                               struct pipe_resource *resource, unsigned usage)
{
   if (usage & PIPE_MAP_UNSYNCHRONIZED)
      return false;
   return gex_wgpu_batch_conflicts(ctx, resource);
}

static void
gex_wgpu_set_constant_buffer(struct pipe_context *pipe,
                             mesa_shader_stage shader, uint index,
                             const struct pipe_constant_buffer *buffer)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   /* An unbind has to succeed for every stage and every slot, including the
    * ones this driver does not implement: st/mesa and cso_context clear
    * constant buffers across all stages as ordinary state setup, and there is
    * nothing to get wrong about clearing a stage that is never drawn with.
    *
    * Same defect and same discovery as set_sampler_views: refusing it latched
    * the context for an application that never used those stages, so every
    * later draw was dropped and SuperTuxKart rendered nothing while reporting
    * no GL error. Binding real data outside vertex or fragment is still
    * unimplemented and still refused. */
   const bool unbinding = !buffer || (!buffer->buffer && !buffer->user_buffer);
   if (shader != MESA_SHADER_VERTEX && shader != MESA_SHADER_FRAGMENT) {
      if (!unbinding) {
         gex_fail(ctx);
         return;
      }
      if (shader < MESA_SHADER_STAGES && index < GEX_WGPU_MAX_CONSTANT_BUFFERS)
         util_copy_constant_buffer(&ctx->constant_buffers[shader][index], NULL);
      return;
   }
   if (index >= GEX_WGPU_MAX_CONSTANT_BUFFERS) {
      if (!unbinding)
         gex_fail(ctx);
      return;
   }
   /* `unbinding` -- not just a NULL pointer. st/mesa clears a slot both ways:
    * with no `pipe_constant_buffer` at all, and with one whose resource, user
    * pointer and size are all zero. The second shape reached the validation
    * below and latched the context, which is the same "refusing a clear is
    * refusing to do nothing" defect already fixed once here for the
    * non-vertex/fragment stages -- the predicate was computed above and then
    * only half used. SuperTuxKart clears vertex slots 2 and 3 during startup,
    * so every draw after that was dropped and it presented an empty frame at
    * full rate. */
   if (unbinding) {
      util_copy_constant_buffer(&ctx->constant_buffers[shader][index], NULL);
      return;
   }
   if (!buffer->buffer && buffer->user_buffer && buffer->buffer_size > 0 &&
       buffer->buffer_offset == 0) {
      /* st/mesa hands most constant buffers over as a plain pointer, and this
       * used to answer with a fresh WebGPU buffer every time -- created,
       * written, bound and dropped, once per call. That is the single most
       * frequent call a frame makes: Minecraft issues about two per draw, and
       * the allocation alone cost more than the draw did.
       *
       * One buffer per slot instead, grown when a larger upload arrives and
       * otherwise reused. Reuse is safe because a draw is submitted before
       * the next set_constant_buffer can run, and wgpuQueueWriteBuffer is
       * ordered against submits on the same queue -- so draw N sees the bytes
       * written for it, not draw N+1's. If draws are ever batched into one
       * submit this has to become a ring.
       *
       * It also makes the binding *stable*, which is what lets anything
       * downstream cache on it: a new buffer every draw is a new bind group
       * every draw. */
      if (buffer->buffer_size > GEX_WGPU_CONSTANT_RING_BYTES) {
         gex_fail(ctx);
         return;
      }
      if (!ctx->constant_ring) {
         struct pipe_resource templ = {0};
         templ.target = PIPE_BUFFER;
         templ.format = PIPE_FORMAT_R8_UNORM;
         templ.width0 = GEX_WGPU_CONSTANT_RING_BYTES;
         templ.height0 = 1;
         templ.depth0 = 1;
         templ.array_size = 1;
         templ.bind = PIPE_BIND_CONSTANT_BUFFER;
         ctx->constant_ring = pipe->screen->resource_create(pipe->screen,
                                                            &templ);
         if (!ctx->constant_ring) {
            gex_fail(ctx);
            return;
         }
      }
      /* WebGPU's guaranteed minUniformBufferOffsetAlignment, which is what a
       * dynamic offset has to be a multiple of. */
      unsigned offset = ALIGN_POT(ctx->constant_ring_offset, 256u);
      if (offset + buffer->buffer_size > GEX_WGPU_CONSTANT_RING_BYTES) {
         /* Wrapping reuses bytes earlier draws may still be waiting to read,
          * so everything recorded so far has to be on the queue before the
          * write that overwrites them. */
         gex_wgpu_submit_pending(ctx);
         gex_wgpu_flush_constant_ring(ctx);
         offset = 0;
      }
      ctx->constant_ring_offset = offset + buffer->buffer_size;
      if (!ctx->constant_staging) {
         ctx->constant_staging = MALLOC(GEX_WGPU_CONSTANT_RING_BYTES);
         if (!ctx->constant_staging) {
            gex_fail(ctx);
            return;
         }
      }
      memcpy(ctx->constant_staging + offset, buffer->user_buffer,
             buffer->buffer_size);
      if (ctx->constant_dirty_hi == 0 || offset < ctx->constant_dirty_lo)
         ctx->constant_dirty_lo = offset;
      if (offset + buffer->buffer_size > ctx->constant_dirty_hi)
         ctx->constant_dirty_hi = offset + buffer->buffer_size;
      if (!ctx->failed) {
         struct pipe_constant_buffer uploaded = {
            .buffer = ctx->constant_ring,
            .buffer_offset = offset,
            .buffer_size = buffer->buffer_size,
         };
         util_copy_constant_buffer(
            &ctx->constant_buffers[shader][index], &uploaded);
      }
      return;
   }
   if (!buffer->buffer || buffer->user_buffer || buffer->buffer_size == 0 ||
       buffer->buffer->target != PIPE_BUFFER ||
       !(buffer->buffer->bind & PIPE_BIND_CONSTANT_BUFFER) ||
       (buffer->buffer_offset & 255u) ||
       (uint64_t)buffer->buffer_offset + buffer->buffer_size >
          buffer->buffer->width0) {
      /* Seven conditions shared one latch, and a latched context turns every
       * later draw into a no-op -- the application keeps presenting, at full
       * rate, a frame nothing was drawn into. Name which one. */
      fprintf(stderr,
              "gex-wgpu: constant buffer refused (%s) shader=%u index=%u "
              "buffer=%p user=%p size=%u offset=%u target=%d bind=0x%x "
              "width0=%u\n",
              !buffer->buffer ? "no resource"
                 : buffer->user_buffer ? "user pointer with a resource"
                 : buffer->buffer_size == 0 ? "zero size"
                 : buffer->buffer->target != PIPE_BUFFER ? "not a buffer"
                 : !(buffer->buffer->bind & PIPE_BIND_CONSTANT_BUFFER)
                    ? "missing PIPE_BIND_CONSTANT_BUFFER"
                 : (buffer->buffer_offset & 255u) ? "offset not 256-aligned"
                 : "range past the end of the buffer",
              shader, index, (void *)buffer->buffer,
              (void *)buffer->user_buffer, buffer->buffer_size,
              buffer->buffer_offset,
              buffer->buffer ? (int)buffer->buffer->target : -1,
              buffer->buffer ? buffer->buffer->bind : 0u,
              buffer->buffer ? buffer->buffer->width0 : 0u);
      gex_fail(ctx);
      return;
   }
   util_copy_constant_buffer(&ctx->constant_buffers[shader][index], buffer);
   /* The fragment stage's 32-byte block, read back a few times.
    *
    * SuperTuxKart's foreground is uniformly light while its distance is
    * detailed, which is what a fog factor that saturates near the camera looks
    * like, and the UBO work above named this block: `u_fog_color` and
    * `u_fog_data`, two vec4s. Whether the numbers arriving here are the ones
    * the application set is the difference between a fog bug and a lighting
    * one, and nothing else in the pipeline will say. */
   /* The vertex stage's constant buffers, first sixteen floats of each.
    *
    * With the skybox draws removed the frame is black but for the HUD, so no
    * scene geometry reaches the screen at all -- and a vertex shader whose
    * transform arrives as zeros collapses every vertex, which looks exactly
    * like that. This is where a matrix would be. */
   if (shader == MESA_SHADER_VERTEX && buffer->buffer_size >= 64) {
      static unsigned told;
      if (told < 5) {
         ++told;
         const struct gex_wgpu_resource *vgex =
            gex_resource(ctx->constant_buffers[shader][index].buffer);
         const uint32_t voff =
            ctx->constant_buffers[shader][index].buffer_offset;
         if (vgex && vgex->shadow) {
            float m[16];
            memcpy(m, vgex->shadow + voff, sizeof(m));
            fprintf(stderr,
                    "gex-wgpu: vertex cb%u %uB written=%llu = "
                    "%.3f %.3f %.3f %.3f / %.3f %.3f %.3f %.3f / "
                    "%.3f %.3f %.3f %.3f / %.3f %.3f %.3f %.3f\n",
                    index, buffer->buffer_size,
                    (unsigned long long)vgex->bytes_written,
                    m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7],
                    m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
         }
      }
   }
   if (shader == MESA_SHADER_FRAGMENT && buffer->buffer_size == 32) {
      static unsigned told;
      if (told < 6) {
         ++told;
         const struct gex_wgpu_resource *gex =
            gex_resource(ctx->constant_buffers[shader][index].buffer);
         const uint32_t off =
            ctx->constant_buffers[shader][index].buffer_offset;
         if (gex && gex->shadow) {
            float v[8];
            memcpy(v, gex->shadow + off, sizeof(v));
            fprintf(stderr,
                    "gex-wgpu: fragment cb%u 32B = %.3f %.3f %.3f %.3f | "
                    "%.3f %.3f %.3f %.3f (offset=%u width0=%u written=%llu)\n",
                    index, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
                    off, ctx->constant_buffers[shader][index].buffer->width0,
                    (unsigned long long)gex->bytes_written);
         }
      }
   }
}

static const char *
gex_wgpu_get_name(struct pipe_screen *screen)
{
   (void)screen;
   return "gex WebGPU (bring-up)";
}

static const char *
gex_wgpu_get_vendor(struct pipe_screen *screen)
{
   (void)screen;
   return "Mesa";
}

static const char *
gex_wgpu_get_device_vendor(struct pipe_screen *screen)
{
   (void)screen;
   return "WebGPU";
}

/* Formats GL requires that WebGPU has no equivalent for, backed by a wider
 * format that holds every value of the original exactly.
 *
 * At GL 3.2 that is the 16-bit normalized set and nothing else. WebGPU has no
 * 16-bit unorm or snorm texture format at all, so st/mesa's fallback list for
 * GL_R16 runs off the end of DEFAULT_UNORM16_RGBA_FORMATS into
 * DEFAULT_RGBA_FORMATS and picks 8-bit RGBA -- which is why every r16, rg16
 * and rgba16 case in packed_pixels failed its gradient comparison. It was not
 * a readback defect; half the precision was gone before the readback.
 *
 * float32 holds them exactly: 65536 unorm levels and 65535 snorm ones all fit
 * in a 24-bit mantissa, so the round trip through util_format_translate is
 * lossless in both directions. The texture still filters, renders and blends
 * as an ordinary float texture -- provided the device granted
 * float32-filterable and float32-blendable, which is why is_format_supported
 * gates on them. The cost is twice the memory and a conversion on transfer.
 */
static enum pipe_format
gex_wgpu_format_backing(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
      return PIPE_FORMAT_R32_FLOAT;
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
      return PIPE_FORMAT_R32G32_FLOAT;
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
      return PIPE_FORMAT_R32G32B32A32_FLOAT;
   default:
      return PIPE_FORMAT_NONE;
   }
}

static WGPUTextureFormat
gex_wgpu_texture_format(enum pipe_format format)
{
   const enum pipe_format backing = gex_wgpu_format_backing(format);
   if (backing != PIPE_FORMAT_NONE)
      format = backing;
   switch (format) {
   case PIPE_FORMAT_R8_UNORM: return WGPUTextureFormat_R8Unorm;
   case PIPE_FORMAT_R8G8_UNORM: return WGPUTextureFormat_RG8Unorm;
   case PIPE_FORMAT_R8_UINT: return WGPUTextureFormat_R8Uint;
   case PIPE_FORMAT_R8G8_UINT: return WGPUTextureFormat_RG8Uint;
   case PIPE_FORMAT_R8G8B8A8_UINT: return WGPUTextureFormat_RGBA8Uint;
   case PIPE_FORMAT_R8_SINT: return WGPUTextureFormat_R8Sint;
   case PIPE_FORMAT_R8G8_SINT: return WGPUTextureFormat_RG8Sint;
   case PIPE_FORMAT_R8G8B8A8_SINT: return WGPUTextureFormat_RGBA8Sint;
   case PIPE_FORMAT_R16_FLOAT: return WGPUTextureFormat_R16Float;
   case PIPE_FORMAT_R16G16_FLOAT: return WGPUTextureFormat_RG16Float;
   case PIPE_FORMAT_R16G16B16A16_FLOAT: return WGPUTextureFormat_RGBA16Float;
   /* WebGPU core guarantees the 32-bit float colour formats as renderable and
    * sampleable; filtering them additionally needs the float32-filterable
    * feature, which is why texture_float_linear stays off. */
   case PIPE_FORMAT_R32_FLOAT: return WGPUTextureFormat_R32Float;
   case PIPE_FORMAT_R32G32_FLOAT: return WGPUTextureFormat_RG32Float;
   case PIPE_FORMAT_R32G32B32A32_FLOAT: return WGPUTextureFormat_RGBA32Float;
   /* 32-bit integer colour formats, core WebGPU and renderable there.
    * EXT_texture_integer -- required for desktop GL 3.0 -- asks for the
    * four-component pair specifically. */
   case PIPE_FORMAT_R32_UINT: return WGPUTextureFormat_R32Uint;
   case PIPE_FORMAT_R32_SINT: return WGPUTextureFormat_R32Sint;
   case PIPE_FORMAT_R32G32_UINT: return WGPUTextureFormat_RG32Uint;
   case PIPE_FORMAT_R32G32_SINT: return WGPUTextureFormat_RG32Sint;
   case PIPE_FORMAT_R32G32B32A32_UINT: return WGPUTextureFormat_RGBA32Uint;
   case PIPE_FORMAT_R32G32B32A32_SINT: return WGPUTextureFormat_RGBA32Sint;
   case PIPE_FORMAT_R16_UINT: return WGPUTextureFormat_R16Uint;
   case PIPE_FORMAT_R16_SINT: return WGPUTextureFormat_R16Sint;
   case PIPE_FORMAT_R16G16_UINT: return WGPUTextureFormat_RG16Uint;
   case PIPE_FORMAT_R16G16_SINT: return WGPUTextureFormat_RG16Sint;
   case PIPE_FORMAT_R16G16B16A16_UINT: return WGPUTextureFormat_RGBA16Uint;
   case PIPE_FORMAT_R16G16B16A16_SINT: return WGPUTextureFormat_RGBA16Sint;
   /* Sampleable in core WebGPU.  Only rgb10a2unorm is also renderable there;
    * rg11b10ufloat needs a feature and rgb9e5ufloat and the snorm formats are
    * sample-only, which is what gex_wgpu_color_renderable reflects. */
   case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT:
      return WGPUTextureFormat_Depth32FloatStencil8;
   case PIPE_FORMAT_R10G10B10A2_UNORM: return WGPUTextureFormat_RGB10A2Unorm;
   case PIPE_FORMAT_R10G10B10A2_UINT: return WGPUTextureFormat_RGB10A2Uint;
   case PIPE_FORMAT_R11G11B10_FLOAT: return WGPUTextureFormat_RG11B10Ufloat;
   case PIPE_FORMAT_R9G9B9E5_FLOAT: return WGPUTextureFormat_RGB9E5Ufloat;
   case PIPE_FORMAT_R8_SNORM: return WGPUTextureFormat_R8Snorm;
   case PIPE_FORMAT_R8G8_SNORM: return WGPUTextureFormat_RG8Snorm;
   case PIPE_FORMAT_R8G8B8A8_SNORM: return WGPUTextureFormat_RGBA8Snorm;
   case PIPE_FORMAT_R8G8B8A8_UNORM: return WGPUTextureFormat_RGBA8Unorm;
   case PIPE_FORMAT_R8G8B8A8_SRGB: return WGPUTextureFormat_RGBA8UnormSrgb;
   case PIPE_FORMAT_B8G8R8A8_UNORM: return WGPUTextureFormat_BGRA8Unorm;
   /* The sRGB counterpart of the one above. Without it, a BGRA texture had no
    * sRGB view format to declare, so the linear/sRGB reinterpretation that
    * GL_FRAMEBUFFER_SRGB and st/mesa's blits depend on was rejected at view
    * creation -- and st/mesa saw a format it could not make a complete
    * framebuffer from. Renderable below, in the same edit, because
    * advertising a format without telling the renderable check about it is
    * how the depth16 attempt turned a new capability into a latch. */
   case PIPE_FORMAT_B8G8R8A8_SRGB: return WGPUTextureFormat_BGRA8UnormSrgb;
   case PIPE_FORMAT_DXT1_RGB:
   case PIPE_FORMAT_DXT1_RGBA: return WGPUTextureFormat_BC1RGBAUnorm;
   case PIPE_FORMAT_DXT1_SRGB:
   case PIPE_FORMAT_DXT1_SRGBA: return WGPUTextureFormat_BC1RGBAUnormSrgb;
   case PIPE_FORMAT_DXT3_RGBA: return WGPUTextureFormat_BC2RGBAUnorm;
   case PIPE_FORMAT_DXT3_SRGBA: return WGPUTextureFormat_BC2RGBAUnormSrgb;
   case PIPE_FORMAT_DXT5_RGBA: return WGPUTextureFormat_BC3RGBAUnorm;
   case PIPE_FORMAT_DXT5_SRGBA: return WGPUTextureFormat_BC3RGBAUnormSrgb;
   case PIPE_FORMAT_RGTC1_UNORM: return WGPUTextureFormat_BC4RUnorm;
   case PIPE_FORMAT_RGTC1_SNORM: return WGPUTextureFormat_BC4RSnorm;
   case PIPE_FORMAT_RGTC2_UNORM: return WGPUTextureFormat_BC5RGUnorm;
   case PIPE_FORMAT_RGTC2_SNORM: return WGPUTextureFormat_BC5RGSnorm;
   case PIPE_FORMAT_BPTC_RGB_FLOAT: return WGPUTextureFormat_BC6HRGBFloat;
   case PIPE_FORMAT_BPTC_RGB_UFLOAT: return WGPUTextureFormat_BC6HRGBUfloat;
   case PIPE_FORMAT_BPTC_RGBA_UNORM: return WGPUTextureFormat_BC7RGBAUnorm;
   case PIPE_FORMAT_BPTC_SRGBA: return WGPUTextureFormat_BC7RGBAUnormSrgb;
   case PIPE_FORMAT_Z32_FLOAT: return WGPUTextureFormat_Depth32Float;
   /* Z16_UNORM was absent twice, both times because a depth-only framebuffer
    * had no render pass to draw into -- "draw refused (no colour target)" --
    * and the refusal latched the context, which cost far more than the format
    * was worth (packed_pixels.rectangle went 169 failures to 616).
    *
    * That pass exists now: a pipeline with no fragment state and no colour
    * attachment, which is what WebGPU calls a depth-only pass. Without the
    * format, GL_DEPTH_COMPONENT16 has no Gallium format st/mesa can allocate,
    * resource_create returns NULL, st/mesa reports GL_OUT_OF_MEMORY -- and
    * dEQP turns that into a tcu::ResourceError, which terminates the whole
    * test run. texture_repeat_mode died after 65 of its 162 cases exactly
    * there. A missing format is not a missing feature here; it is a missing
    * run. */
   case PIPE_FORMAT_Z16_UNORM: return WGPUTextureFormat_Depth16Unorm;
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
      return WGPUTextureFormat_Depth24PlusStencil8;
   default: return WGPUTextureFormat_Undefined;
   }
}

static bool
gex_wgpu_is_bc_format(enum pipe_format format)
{
   return format == PIPE_FORMAT_DXT1_RGB ||
          format == PIPE_FORMAT_DXT1_SRGB ||
          format == PIPE_FORMAT_DXT1_RGBA ||
          format == PIPE_FORMAT_DXT1_SRGBA ||
          format == PIPE_FORMAT_DXT3_RGBA ||
          format == PIPE_FORMAT_DXT3_SRGBA ||
          format == PIPE_FORMAT_DXT5_RGBA ||
          format == PIPE_FORMAT_DXT5_SRGBA ||
          format == PIPE_FORMAT_RGTC1_UNORM ||
          format == PIPE_FORMAT_RGTC1_SNORM ||
          format == PIPE_FORMAT_RGTC2_UNORM ||
          format == PIPE_FORMAT_RGTC2_SNORM ||
          format == PIPE_FORMAT_BPTC_RGB_FLOAT ||
          format == PIPE_FORMAT_BPTC_RGB_UFLOAT ||
          format == PIPE_FORMAT_BPTC_RGBA_UNORM ||
          format == PIPE_FORMAT_BPTC_SRGBA;
}

/* Bytes per pixel, from Mesa's own format table rather than a list here.
 *
 * A hand-written list drifts from the set of formats the driver accepts, and
 * the failure is not a wrong size but a zero one: texture_subdata refuses a
 * format it cannot size, and a refusal ends the context. Forty-one of the
 * fifty-five formats gex_wgpu_texture_format maps were unsized this way --
 * every 16- and 32-bit format, every snorm one, rgb10a2 -- so uploading any
 * of them killed the run.
 *
 * Block-compressed formats are deliberately not sized here: their callers
 * take the block path and ask util_format_get_blocksize directly, and a
 * bytes-per-pixel answer for them would be meaningless. */
static unsigned
gex_wgpu_format_bytes(enum pipe_format format)
{
   if (format == PIPE_FORMAT_NONE || util_format_is_compressed(format))
      return 0;
   return util_format_get_blocksize(format);
}

/* How a shader is allowed to sample this format, from Mesa's own description
 * of it rather than a list here.
 *
 * The same drift that made gex_wgpu_format_bytes return zero for
 * forty-one accepted formats applied here in a quieter way: an unlisted format
 * fell through to Float, so an integer texture read by an integer shader was
 * reported as a mismatch and the draw refused. Every 16- and 32-bit integer
 * format was unlisted. */
static WGPUTextureSampleType
gex_wgpu_format_sample_type(enum pipe_format format)
{
   if (util_format_is_depth_or_stencil(format))
      return WGPUTextureSampleType_Depth;
   if (util_format_is_pure_uint(format))
      return WGPUTextureSampleType_Uint;
   if (util_format_is_pure_sint(format))
      return WGPUTextureSampleType_Sint;
   return WGPUTextureSampleType_Float;
}

static WGPUTextureViewDimension
gex_wgpu_target_view_dimension(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_1D: return WGPUTextureViewDimension_2D;
   case PIPE_TEXTURE_1D_ARRAY: return WGPUTextureViewDimension_2DArray;
   case PIPE_TEXTURE_2D: return WGPUTextureViewDimension_2D;
   case PIPE_TEXTURE_RECT: return WGPUTextureViewDimension_2D;
   case PIPE_TEXTURE_2D_ARRAY: return WGPUTextureViewDimension_2DArray;
   case PIPE_TEXTURE_3D: return WGPUTextureViewDimension_3D;
   case PIPE_TEXTURE_CUBE: return WGPUTextureViewDimension_Cube;
   case PIPE_TEXTURE_CUBE_ARRAY: return WGPUTextureViewDimension_CubeArray;
   default: return WGPUTextureViewDimension_Undefined;
   }
}

/* Colour formats WebGPU guarantees are renderable.
 *
 * This is the core set every WebGPU implementation must support as a render
 * attachment, and each one already has a mapping in gex_wgpu_texture_format
 * above.  Reporting only RGBA8 and RGBA16F -- as this did -- also withholds
 * the GL extensions st/mesa derives from render-target support, notably
 * ARB_texture_rg, which OpenGL ES 3.0 requires.
 *
 * The integer formats are renderable but not blendable, in WebGPU and in GL
 * alike; that is enforced where the blend state is built, not here.
 */
static bool
gex_wgpu_color_renderable(struct pipe_screen *screen, enum pipe_format format)
{
   /* Renderable only when the page could request the optional feature. */
   if (format == PIPE_FORMAT_R11G11B10_FLOAT)
      return gex_screen(screen)->rg11b10_renderable;
   switch (format) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_SRGB:
   case PIPE_FORMAT_R8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   /* ARB_texture_rgb10_a2ui wants this renderable as well as sampleable,
    * which core WebGPU gives for both rgb10a2 formats. */
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_R32_UINT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_SINT:
      return true;
   /* Backed by float32, which is renderable in core WebGPU. GL needs these
    * renderable as well as sampleable: packed_pixels reads a texture back by
    * attaching it to a framebuffer, so an unrenderable R16 is an incomplete
    * framebuffer and not merely an unfilterable texture. */
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
      return gex_screen(screen)->float32_filterable &&
             gex_screen(screen)->float32_blendable;
   default:
      return false;
   }
}

/* Presentation reaches past the pipe interface deliberately: Gallium has no
 * concept of "the canvas", and this driver's drawable is a plain texture the
 * winsys owns. Handing out the texture is narrower than teaching the driver
 * about surfaces. */
WGPUTexture
gex_wgpu_resource_texture(struct pipe_resource *resource)
{
   return resource ? gex_resource(resource)->texture : NULL;
}

WGPUDevice
gex_wgpu_screen_device(struct pipe_screen *screen)
{
   return screen ? gex_screen(screen)->device : NULL;
}

static bool
gex_wgpu_is_format_supported(struct pipe_screen *screen,
                             enum pipe_format format,
                             enum pipe_texture_target target,
                             unsigned sample_count,
                             unsigned storage_sample_count,
                             unsigned bind)
{
   if (target == PIPE_BUFFER) {
      /* Answering yes here is a capability claim, not permission to
       * allocate: st/mesa reads it to decide whether a texture-buffer path
       * exists at all, and takes that path if told it does. Allocation is
       * separately permissive -- see resource_create -- because refusing an
       * allocation surfaces as an unrelated GL error, whereas claiming a
       * capability that does not work produces wrong pixels. */
      const unsigned allowed = PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_INDEX_BUFFER |
                               PIPE_BIND_CONSTANT_BUFFER |
                               PIPE_BIND_STREAM_OUTPUT |
                               PIPE_BIND_SHADER_BUFFER |
                               /* ARB_draw_indirect. The draw arguments are
                                * read back on the CPU rather than handed to
                                * WebGPU's drawIndirect, so this needs no
                                * WGPUBufferUsage_Indirect -- only that a
                                * buffer carrying the bind is not refused. */
                               PIPE_BIND_COMMAND_ARGS_BUFFER |
                               /* Display lists. Mesa's vbo_save tags every
                                * buffer it allocates with this, whether or
                                * not the driver supports draw_vertex_state
                                * (`bufferobj.c` reads the storage flag
                                * unconditionally). Refusing it made
                                * glNewList fail with GL_OUT_OF_MEMORY, which
                                * is what st reports when a driver latches --
                                * glxgears could not build its gears. Nothing
                                * is promised by accepting it: st only calls
                                * create_vertex_state when
                                * caps->draw_vertex_state says so, and this
                                * driver leaves that false, so the buffer is
                                * an ordinary vertex and index buffer. */
                               PIPE_BIND_VERTEX_STATE;
      if (sample_count > 1 || storage_sample_count > 1 || (bind & ~allowed))
         return false;
      /* A vertex format is the one buffer question with a real answer, and
       * this used to ignore `format` and say yes to all of them. That is what
       * u_vbuf reads to build its translation table (u_vbuf.c:301) and what
       * st/mesa reads for the vertex-format extensions, so claiming a format
       * the vertex layout below cannot express means u_vbuf never translates
       * it, it arrives here anyway, and `create_vertex_elements_state` refuses
       * the whole state -- no layout bound, and the next draw latches the
       * context. SuperTuxKart packs its normals as `r10g10b10a2_snorm`, which
       * WebGPU has no form of (it has only the unsigned one), and lost its
       * renderer to exactly that. Answering honestly hands the format to
       * u_vbuf, which converts it to something expressible. */
      if (bind == PIPE_BIND_VERTEX_BUFFER && format != PIPE_FORMAT_NONE) {
         unsigned vertex_size, vertex_alignment;
         if (!gex_wgpu_vertex_format(format, &vertex_size, &vertex_alignment))
            return false;
      }
      return true;
   }
   if (gex_wgpu_is_bc_format(format)) {
      const bool target_supported =
         target == PIPE_TEXTURE_2D || target == PIPE_TEXTURE_2D_ARRAY ||
         target == PIPE_TEXTURE_CUBE || target == PIPE_TEXTURE_CUBE_ARRAY;
      return gex_screen(screen)->texture_compression_bc && target_supported &&
             bind == PIPE_BIND_SAMPLER_VIEW &&
             MAX2(1, sample_count) == 1 &&
             MAX2(1, storage_sample_count) == 1;
   }
   if (format == PIPE_FORMAT_Z32_FLOAT ||
       format == PIPE_FORMAT_Z16_UNORM ||
       format == PIPE_FORMAT_Z24_UNORM_S8_UINT ||
       /* depth32float-stencil8 is an optional WebGPU feature; ARB_depth_buffer_float
        * needs it alongside plain depth32float. */
       (format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
        gex_screen(screen)->depth32_stencil8)) {
      const unsigned allowed =
         PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW;
      /* 4x multisampling, on the same terms as the colour formats below.
       *
       * This used to refuse any sample count above one, which is what made
       * `gl44-msaa-depth-blit-probe` fail against native Mesa: a 4x depth24
       * texture came back with `samples=0`, so the framebuffer was
       * GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT and the blit was
       * GL_INVALID_OPERATION. WebGPU multisamples a depth attachment exactly
       * as it does a colour one -- 4x is the guaranteed count, the attachment
       * is written by rendering, and WGSL reads it back through
       * texture_depth_multisampled_2d -- so there was never a reason for
       * depth to be the exception. */
      const unsigned depth_samples = MAX2(1, sample_count);
      return target == PIPE_TEXTURE_2D &&
             (depth_samples == 1 || depth_samples == 4) &&
             MAX2(1, storage_sample_count) == depth_samples &&
             bind && !(bind & ~allowed);
   }
   /* A format backed by float32 is only usable if the device will filter and
    * blend float32. Refusing it here is the whole point of the gate: st/mesa
    * would otherwise pick it for GL_R16, and the first sample or blend would
    * be a validation error -- which poisons the device for every later
    * command, where a "no" here only sends st/mesa to its next fallback. */
   if (gex_wgpu_format_backing(format) != PIPE_FORMAT_NONE &&
       !(gex_screen(screen)->float32_filterable &&
         gex_screen(screen)->float32_blendable))
      return false;
   const bool renderable = gex_wgpu_color_renderable(screen, format);
   /* A format is sampleable exactly when this driver can name it to WebGPU.
    * Deriving that from the format table rather than repeating the list keeps
    * the two from drifting: a format added to the table but forgotten here
    * would be silently unusable, and one removed would be advertised and then
    * fail at resource creation. */
   const bool sampled = gex_wgpu_texture_format(format) !=
                        WGPUTextureFormat_Undefined;
   /* The three window-system binds mean "a frontend will present this", not
    * anything WebGPU has an opinion about: the pixels still live in an
    * ordinary texture and reach the screen by being read back and handed to
    * the winsys. A browser build never sees them, because the page owns the
    * canvas; a desktop build sees them on every drawable, and refusing them
    * left drisw with no usable visual at all -- "failed to create drisw
    * screen" before a single GL call. */
   const unsigned present_binds = PIPE_BIND_DISPLAY_TARGET |
                                  PIPE_BIND_SCANOUT | PIPE_BIND_SHARED;
   const unsigned allowed = renderable && target == PIPE_TEXTURE_2D
      ? PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW | present_binds
      : PIPE_BIND_SAMPLER_VIEW | present_binds;

   /* WebGPU guarantees 4x multisampling for renderable formats, and only 4x:
    * 2, 8 and 16 are not part of the core feature set. A multisampled texture
    * is written by rendering and read by resolving -- it cannot be copied --
    * so the sampler-view bind is allowed alongside the render target, because
    * a resolve is performed by sampling it. */
   /* Renderable is not the same as multisample-capable. WebGPU excludes the
    * 32-bit-per-component formats and r32uint/r32sint from multisampling
    * entirely, so claiming 4x for them offers st/mesa a framebuffer that
    * cannot be created -- and the 16-bit normalized set inherits the
    * exclusion through its float32 backing. */
   const WGPUTextureFormat wgpu = gex_wgpu_texture_format(format);
   const bool multisampleable = renderable &&
      wgpu != WGPUTextureFormat_R32Float &&
      wgpu != WGPUTextureFormat_RG32Float &&
      wgpu != WGPUTextureFormat_RGBA32Float &&
      wgpu != WGPUTextureFormat_R32Uint &&
      wgpu != WGPUTextureFormat_R32Sint &&
      wgpu != WGPUTextureFormat_RG32Uint &&
      wgpu != WGPUTextureFormat_RG32Sint &&
      wgpu != WGPUTextureFormat_RGBA32Uint &&
      wgpu != WGPUTextureFormat_RGBA32Sint;
   const unsigned requested = MAX2(1, sample_count);
   if (requested > 1) {
      const unsigned multisample_binds =
         PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW;
      return multisampleable && requested == 4 &&
             MAX2(1, storage_sample_count) == requested &&
             target == PIPE_TEXTURE_2D &&
             bind && !(bind & ~multisample_binds);
   }

   /* Why a render-target query was refused. Framebuffer completeness is
    * decided entirely by st/mesa calling this, and a "no" leaves no trace
    * anywhere else -- the application sees GL_FRAMEBUFFER_UNSUPPORTED and the
    * driver sees nothing at all. Five guesses at the sRGB completeness
    * failures came from not having this line. */
   const bool target_ok =
      target == PIPE_TEXTURE_1D || target == PIPE_TEXTURE_1D_ARRAY ||
      target == PIPE_TEXTURE_2D ||
      target == PIPE_TEXTURE_RECT ||
      target == PIPE_TEXTURE_2D_ARRAY ||
      target == PIPE_TEXTURE_3D || target == PIPE_TEXTURE_CUBE ||
      target == PIPE_TEXTURE_CUBE_ARRAY;
   const bool bind_ok = bind && !(bind & ~allowed);
   /* Why a render-target query was refused, and which condition did it.
    * Framebuffer completeness is decided entirely by st/mesa calling this
    * hook, and a "no" leaves no trace anywhere else: the application sees
    * GL_FRAMEBUFFER_UNSUPPORTED and the driver sees nothing at all.
    *
    * Every failing condition is named, not just an unrenderable format --
    * an earlier version logged only `!renderable` and so stayed silent for a
    * format that is renderable but refused on its target or its binds, which
    * is exactly the case that was being hunted. Deduped per format, because
    * the first refusal of a run is st/mesa probing a list where any one
    * format will do. */
   if ((bind & PIPE_BIND_RENDER_TARGET) &&
       !(sampled && target_ok && bind_ok)) {
      static uint64_t said[(PIPE_FORMAT_COUNT + 63) / 64];
      const unsigned word = (unsigned)format / 64;
      const uint64_t bit = 1ull << ((unsigned)format % 64);
      if (format < PIPE_FORMAT_COUNT && !(said[word] & bit)) {
         said[word] |= bit;
         fprintf(stderr,
                 "gex-wgpu: render target refused %s target=%u samples=%u/%u "
                 "bind=0x%x allowed=0x%x (%s%s%s%s)\n",
                 util_format_short_name(format), target, sample_count,
                 storage_sample_count, bind, allowed,
                 sampled ? "" : "unmapped format ",
                 renderable ? "" : "not renderable ",
                 target_ok ? "" : "bad target ",
                 bind_ok ? "" : "bad binds");
      }
   }
   return sampled && target_ok &&
          bind &&
          MAX2(1, sample_count) == 1 &&
          MAX2(1, storage_sample_count) == 1 &&
          !(bind & ~allowed);
}

static struct pipe_resource *
gex_wgpu_resource_create(struct pipe_screen *screen,
                         const struct pipe_resource *templ)
{
   GEX_DEVICE_GUARD(gex_screen(screen));
   if (templ->target == PIPE_BUFFER) {
      /* Render-target and sampler-view binds on a *buffer* are st/mesa asking
       * for the option of a texture-buffer path it may never take. Refusing
       * them here is the wrong place to say no: nothing can render to or
       * sample a buffer, and the calls that would try refuse on their own,
       * whereas a null resource comes back as a GL error from whatever
       * unrelated operation happened to allocate it -- glGetTexImage with a
       * pixel buffer bound, for one. */
      const unsigned allowed = PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_INDEX_BUFFER |
                               PIPE_BIND_CONSTANT_BUFFER |
                               PIPE_BIND_STREAM_OUTPUT |
                               PIPE_BIND_SHADER_BUFFER |
                               PIPE_BIND_COMMAND_ARGS_BUFFER |
                               /* Display lists; see is_format_supported. */
                               PIPE_BIND_VERTEX_STATE |
                               PIPE_BIND_RENDER_TARGET |
                               PIPE_BIND_SAMPLER_VIEW;
      if (templ->width0 == 0 || templ->width0 > (1u << 30) ||
          templ->height0 != 1 || templ->depth0 != 1 || templ->array_size != 1 ||
          templ->last_level != 0 || templ->nr_samples > 1 ||
          templ->nr_storage_samples > 1 || !(templ->bind & allowed) ||
          (templ->bind & ~allowed)) {
         /* u_upload does not check for a NULL resource, so a refusal here
          * surfaces as an out-of-bounds access inside it. */
         fprintf(stderr,
                 "gex-wgpu: buffer resource refused: width0=%u bind=0x%x "
                 "unsupported=0x%x usage=%u\n",
                 templ->width0, templ->bind, templ->bind & ~allowed,
                 templ->usage);
         return NULL;
      }
      struct gex_wgpu_resource *resource = CALLOC_STRUCT(gex_wgpu_resource);
      if (!resource)
         return NULL;
      resource->allocation_size =
         ALIGN_POT((uint64_t)templ->width0,
                   (templ->bind & PIPE_BIND_CONSTANT_BUFFER) ? 16 : 4);
      /* No shadow yet -- gex_wgpu_shadow_ensure allocates one if anything
       * turns out to need it. */
      WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
      desc.size = resource->allocation_size;
      desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
      /* A vertex state *is* a vertex buffer with a layout attached, and
       * Mesa's display-list recorder tags its buffers with only that bind
       * even though it goes on to bind them through set_vertex_buffers when
       * the driver has no draw_vertex_state -- which this one does not. Left
       * out, the buffer had no Vertex usage and the bind was a WebGPU
       * validation error; glxgears drew nothing at all. */
      if (templ->bind & (PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_VERTEX_STATE))
         desc.usage |= WGPUBufferUsage_Vertex;
      if (templ->bind & PIPE_BIND_INDEX_BUFFER)
         desc.usage |= WGPUBufferUsage_Index;
      if (templ->bind & PIPE_BIND_CONSTANT_BUFFER) {
         desc.usage |= WGPUBufferUsage_Uniform;
         /* Slots past the device's uniform buffer limit are bound as storage
          * buffers instead; which slot a buffer lands in is not known here. */
         desc.usage |= WGPUBufferUsage_Storage;
      }
      if (templ->bind & (PIPE_BIND_STREAM_OUTPUT | PIPE_BIND_SHADER_BUFFER))
         desc.usage |= WGPUBufferUsage_Storage;
      /* A vertex buffer is also read as a storage buffer by the transform
       * feedback compute pass, which has no fixed-function fetch. */
      if (templ->bind & (PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_VERTEX_STATE))
         desc.usage |= WGPUBufferUsage_Storage;
      resource->buffer =
         wgpuDeviceCreateBuffer(gex_screen(screen)->device, &desc);
      if (!resource->buffer) {
         FREE(resource->shadow);
         FREE(resource);
         return NULL;
      }
      resource->base = *templ;
      resource->base.screen = screen;
      pipe_reference_init(&resource->base.reference, 1);
      return &resource->base;
   }
   /* Allocation is deliberately more permissive than the capability query.
    * is_format_supported answers "may st/mesa build a GL feature on this
    * combination", and st asks it with the binds it might one day want --
    * PIPE_BIND_RENDER_TARGET on every texture whose format is renderable,
    * whatever its target. Re-testing that here refused the allocation
    * instead, and a refused allocation is GL_OUT_OF_MEMORY from whatever call
    * happened to make it: glTexImage3D of an ordinary GL_RED texture failed
    * 4626 times in packed_pixels.varied_rectangle for exactly this reason,
    * because a 3D texture cannot be a WebGPU render attachment and so the
    * query said no.
    *
    * What has to hold here is narrower: a format this driver can name, a
    * shape WebGPU accepts, and a sample count it can honour. The binds the
    * texture cannot serve are simply left out of desc.usage below. */
   const unsigned requested_samples = MAX2(1, templ->nr_samples);
   if ((gex_wgpu_texture_format(templ->format) == WGPUTextureFormat_Undefined &&
        !gex_wgpu_is_bc_format(templ->format)) ||
       (requested_samples != 1 &&
        !gex_wgpu_is_format_supported(screen, templ->format, templ->target,
                                      templ->nr_samples,
                                      templ->nr_storage_samples,
                                      templ->bind)) ||
       templ->width0 == 0 || templ->height0 == 0 || templ->depth0 == 0 ||
       templ->width0 > 8192 || templ->height0 > 8192 ||
       templ->depth0 > 2048 ||
       (templ->target == PIPE_TEXTURE_1D &&
        (templ->height0 != 1 || templ->depth0 != 1 ||
         templ->array_size != 1)) ||
       (templ->target == PIPE_TEXTURE_1D_ARRAY &&
        (templ->height0 != 1 || templ->depth0 != 1 ||
         templ->array_size == 0 || templ->array_size > 2048)) ||
       (templ->target == PIPE_TEXTURE_2D && templ->array_size != 1) ||
       (templ->target == PIPE_TEXTURE_RECT &&
        (templ->array_size != 1 || templ->last_level != 0)) ||
       ((templ->target == PIPE_TEXTURE_1D ||
         templ->target == PIPE_TEXTURE_1D_ARRAY ||
         templ->target == PIPE_TEXTURE_2D ||
         templ->target == PIPE_TEXTURE_RECT ||
         templ->target == PIPE_TEXTURE_2D_ARRAY ||
         templ->target == PIPE_TEXTURE_CUBE ||
         templ->target == PIPE_TEXTURE_CUBE_ARRAY) && templ->depth0 != 1) ||
       (templ->target == PIPE_TEXTURE_2D_ARRAY &&
        (templ->array_size == 0 || templ->array_size > 2048)) ||
       (templ->target == PIPE_TEXTURE_3D && templ->array_size != 1) ||
       (templ->target == PIPE_TEXTURE_CUBE &&
        (templ->array_size != 6 || templ->width0 != templ->height0)) ||
       (templ->target == PIPE_TEXTURE_CUBE_ARRAY &&
        (templ->array_size < 6 || templ->array_size > 2046 ||
         templ->array_size % 6 != 0 || templ->width0 != templ->height0)) ||
       templ->last_level > util_logbase2(
          MAX3(templ->width0, templ->height0, templ->depth0)))
      return NULL;

   struct gex_wgpu_resource *resource = CALLOC_STRUCT(gex_wgpu_resource);
   if (!resource)
      return NULL;

   /* A texture with no WebGPU format has no resource, and st/mesa reports
    * that as an incomplete framebuffer with no other trace -- `if (!stObj ||
    * !stObj->pt) return GL_FALSE` in fbobject.c, which is indistinguishable
    * from a format refusal at the GL level. Say which format, once each. */
   if (gex_wgpu_texture_format(templ->format) == WGPUTextureFormat_Undefined &&
       !gex_wgpu_is_bc_format(templ->format)) {
      static uint64_t said[(PIPE_FORMAT_COUNT + 63) / 64];
      const unsigned word = (unsigned)templ->format / 64;
      const uint64_t bit = 1ull << ((unsigned)templ->format % 64);
      if (templ->format < PIPE_FORMAT_COUNT && !(said[word] & bit)) {
         said[word] |= bit;
         fprintf(stderr,
                 "gex-wgpu: texture has no WebGPU format: %s target=%u "
                 "bind=0x%x %ux%u\n",
                 util_format_short_name(templ->format), templ->target,
                 templ->bind, templ->width0, templ->height0);
      }
   }

   /* WebGPU sizes a compressed texture in whole blocks; GL does not, and a
    * 7x3 BC texture is ordinary there. Creating one is a validation error,
    * which poisons the device -- so it is declined instead, which costs the
    * one texture. Backing it with a rounded-up allocation would work, but the
    * texture would then be a different size from the one GL asked for, and
    * every coordinate reaching it would need adjusting; that is a real
    * feature, not a guard. */
   /* WebGPU forbids copies to or from a multisampled texture: it is written by
    * rendering and read by resolving, nothing else. */
   const unsigned samples = MAX2(1, templ->nr_samples);
   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.usage = samples > 1
      ? 0u : (WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst);
   if (templ->bind & PIPE_BIND_SAMPLER_VIEW)
      desc.usage |= WGPUTextureUsage_TextureBinding;
   if (templ->bind & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_DEPTH_STENCIL))
      desc.usage |= WGPUTextureUsage_RenderAttachment;
   /* A drawable's front and back buffers carry only the window-system bind on
    * the way in, and rendering into them is the entire point. */
   if (templ->bind & (PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT |
                      PIPE_BIND_SHARED))
      desc.usage |= WGPUTextureUsage_RenderAttachment |
                    WGPUTextureUsage_TextureBinding;
   /* A depth texture is written by rendering into it -- WebGPU takes no
    * buffer copy into the depth aspect of the packed formats -- so it has to
    * be attachable even when the caller only asked to sample it. */
   if (util_format_is_depth_or_stencil(templ->format))
      desc.usage |= WGPUTextureUsage_RenderAttachment;
   if (samples > 1) {
      desc.usage |= WGPUTextureUsage_RenderAttachment;
      /* Resolving reads the texture, so it must be bindable even though the
       * caller may only have asked for a render target. */
      desc.usage |= WGPUTextureUsage_TextureBinding;
   }
   /* GL lets a framebuffer attachment view an sRGB texture as linear, or the
    * reverse -- that is how GL_FRAMEBUFFER_SRGB is expressed, and st/mesa uses
    * it for ordinary blits too. WebGPU allows exactly that reinterpretation
    * and no other, but only if the alternative is declared now: a view whose
    * format is not in this list is rejected at creation. */
   WGPUTextureFormat view_formats[1];
   const enum pipe_format linear = util_format_linear(templ->format);
   const enum pipe_format counterpart = templ->format == linear
      ? util_format_srgb(templ->format) : linear;
   if (counterpart != PIPE_FORMAT_NONE && counterpart != templ->format) {
      const WGPUTextureFormat mapped = gex_wgpu_texture_format(counterpart);
      if (mapped != WGPUTextureFormat_Undefined) {
         view_formats[0] = mapped;
         desc.viewFormatCount = 1;
         desc.viewFormats = view_formats;
      }
   }
   desc.dimension = templ->target == PIPE_TEXTURE_3D
      ? WGPUTextureDimension_3D : WGPUTextureDimension_2D;
   /* WebGPU sizes a compressed texture in whole blocks; GL does not. A 7x3
    * BC texture is ordinary in GL and stores ceil(7/4) x ceil(3/4) = 2 blocks
    * -- which is exactly what an 8x4 WebGPU texture holds, so rounding the
    * allocation up to the block grid stores the same bytes in the same order.
    * Creating it at 7x3 is a validation error instead, and that poisons the
    * device.
    *
    * What rounding up does not fix is sampling: normalised coordinates then
    * address a texture larger than the one GL asked for. Every use in this
    * suite reads these back rather than sampling them, and having them at all
    * beats not having them, but a compressed texture of a non-block size is
    * sampled slightly wrong until the shader compensates. */
   const unsigned block_w = util_format_get_blockwidth(templ->format);
   const unsigned block_h = util_format_get_blockheight(templ->format);
   desc.size.width = block_w > 1 ? align(templ->width0, block_w) : templ->width0;
   desc.size.height = block_h > 1 ? align(templ->height0, block_h)
                                  : templ->height0;
   desc.size.depthOrArrayLayers = templ->target == PIPE_TEXTURE_3D
      ? templ->depth0 : templ->array_size;
   desc.format = gex_wgpu_texture_format(templ->format);
   desc.mipLevelCount = templ->last_level + 1;
   desc.sampleCount = samples;
   resource->texture = wgpuDeviceCreateTexture(gex_screen(screen)->device, &desc);
   if (!resource->texture) {
      FREE(resource);
      return NULL;
   }

   resource->base = *templ;
   resource->base.screen = screen;
   pipe_reference_init(&resource->base.reference, 1);
   return &resource->base;
}

static void
gex_wgpu_resource_destroy(struct pipe_screen *screen,
                          struct pipe_resource *resource)
{
   GEX_DEVICE_GUARD(gex_screen(screen));
   (void)screen;
   struct gex_wgpu_resource *gex = gex_resource(resource);
   if (gex->texture)
      wgpuTextureRelease(gex->texture);
   if (gex->buffer)
      wgpuBufferRelease(gex->buffer);
   if (gex->shadow) {
      gex_wgpu_shadow_live -= gex->allocation_size;
      --gex_wgpu_buffers_live;
      if (!gex->shadow_read)
         gex_wgpu_shadow_unread -= gex->allocation_size;
      if (gex->ever_mapped)
         gex_wgpu_shadow_mapped -= gex->allocation_size;
   }
   FREE(gex->shadow);
   FREE(gex);
}

static void
gex_wgpu_buffer_subdata(struct pipe_context *pipe,
                        struct pipe_resource *resource, unsigned usage,
                        unsigned offset, unsigned size, const void *data)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   /* A write reaches the queue where it is *called*, so anything already
    * recorded has to go first or it would read the new bytes. The constant
    * ring is the exception and the reason the ring exists: it hands out a
    * fresh chunk every time, so its writes never touch bytes a recorded draw
    * still has to read -- and there are two of them per draw, so breaking the
    * batch for them would leave nothing batched at all. */
   if (resource != ctx->constant_ring &&
       gex_wgpu_batch_conflicts_usage(ctx, resource, usage))
      gex_wgpu_submit_pending(ctx);
   struct gex_wgpu_resource *gex = gex_resource(resource);
   /* An aligned write to a buffer that has no shadow needs none: the bytes go
    * straight to the GPU and nothing here has to remember them. This is what
    * actually recovers the memory -- subdata was creating the shadow that
    * every later map then found and reused, so the staged map path never ran.
    *
    * Index and constant buffers are excluded because the draw path reads an
    * index buffer's shadow directly, and both are small. */
   const bool subdata_direct =
      size && !gex->shadow && data && !(usage & PIPE_MAP_READ) &&
      (offset & 3u) == 0 && (size & 3u) == 0 &&
      /* Index buffers only: the draw path scans an index buffer's shadow to
       * bound its vertex fetch. Constant buffers reach the GPU through
       * `gex->buffer` and nothing reads their shadow, and u_upload's constant
       * pages are megabyte-sized -- which is where the memory is. */
      !(resource->bind & PIPE_BIND_INDEX_BUFFER) &&
      resource->target == PIPE_BUFFER && gex->buffer && !ctx->failed &&
      (uint64_t)offset + size <= resource->width0;
   if (subdata_direct) {
      gex->bytes_written += size;
      wgpuQueueWriteBuffer(gex_screen(pipe->screen)->queue, gex->buffer,
                           offset, data, size);
      return;
   }
   /* Everything else keeps the shadow, and if the buffer already has contents
    * on the GPU they have to come back first -- an unaligned write reads the
    * bytes around its range out of it. */
   if (size && !gex->shadow) {
      ++gex_wgpu_shadow_cause_subdata;

      if (!gex_wgpu_shadow_for_read(pipe, gex)) { gex_fail(ctx); return; }
   }
   if (ctx->failed || resource->target != PIPE_BUFFER || !gex->buffer ||
       (size && !gex->shadow) || (!data && size) || (usage & PIPE_MAP_READ) ||
       (uint64_t)offset + size > resource->width0) {
      gex_fail(ctx);
      return;
   }
   gex->bytes_written += size;
   if (!size)
      return;
   memcpy(gex->shadow + offset, data, size);
   const uint32_t aligned_start = offset & ~3u;
   const uint32_t aligned_end = ALIGN_POT(offset + size, 4);
   /* Only an unaligned range needs bytes the caller did not supply. */
   if (aligned_start != offset || aligned_end != offset + size)
      gex_wgpu_mark_shadow_read(gex);
   wgpuQueueWriteBuffer(gex_screen(pipe->screen)->queue, gex->buffer,
                        aligned_start, gex->shadow + aligned_start,
                        aligned_end - aligned_start);
}

/* Depth uploads go through a render pass; the definition is with the readback
 * it mirrors. */
static bool
gex_wgpu_write_depth(struct gex_wgpu_context *ctx,
                     struct pipe_resource *resource, unsigned level,
                     const struct pipe_box *box, const uint8_t *src,
                     unsigned src_stride);

static void
gex_wgpu_texture_subdata(struct pipe_context *pipe,
                         struct pipe_resource *resource, unsigned level,
                         unsigned usage, const struct pipe_box *box,
                         const void *data, unsigned stride,
                         uintptr_t layer_stride)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   GEX_ENTRY_TIMER();
   if (gex_wgpu_batch_conflicts_usage((struct gex_wgpu_context *)pipe,
                                     resource, usage))
      gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   const bool one_d_array = resource &&
      resource->target == PIPE_TEXTURE_1D_ARRAY;
   const bool compressed = resource &&
      gex_wgpu_is_bc_format(resource->format);
   const unsigned bytes_per_pixel = resource
      ? gex_wgpu_format_bytes(resource->format) : 0;
   const unsigned block_width = compressed
      ? util_format_get_blockwidth(resource->format) : 1;
   const unsigned block_height = compressed
      ? util_format_get_blockheight(resource->format) : 1;
   const unsigned block_bytes = compressed
      ? util_format_get_blocksize(resource->format) : bytes_per_pixel;
   const unsigned block_rows = box && box->height > 0
      ? DIV_ROUND_UP((unsigned)box->height, block_height) : 0;
   const uint64_t row_bytes = box && box->width > 0
      ? (uint64_t)DIV_ROUND_UP((unsigned)box->width, block_width) * block_bytes
      : 0;
   const uint64_t required_layer = box && box->height > 0
      ? (uint64_t)stride * block_rows : 0;
   const uint64_t layer_pitch = layer_stride ? layer_stride : required_layer;
   const uint64_t required_data = box && box->height > 0
      ? layer_pitch * ((unsigned)box->depth - 1) +
        (uint64_t)stride * (block_rows - 1) + row_bytes : 0;
   const unsigned level_width = resource
      ? u_minify(resource->width0, level) : 0;
   const unsigned level_height = resource
      ? u_minify(resource->height0, level) : 0;
   if (ctx->failed ||
      (resource->target != PIPE_TEXTURE_1D &&
        resource->target != PIPE_TEXTURE_1D_ARRAY &&
        resource->target != PIPE_TEXTURE_2D &&
        resource->target != PIPE_TEXTURE_RECT &&
        resource->target != PIPE_TEXTURE_2D_ARRAY &&
        resource->target != PIPE_TEXTURE_3D &&
        resource->target != PIPE_TEXTURE_CUBE &&
        resource->target != PIPE_TEXTURE_CUBE_ARRAY) ||
       !gex_resource(resource)->texture || level > resource->last_level ||
       !box || !data || (!bytes_per_pixel && !compressed) ||
       (usage & PIPE_MAP_READ) || box->x < 0 || box->y < 0 || box->z < 0 ||
       box->width <= 0 || box->height <= 0 || box->depth <= 0 ||
       (uint64_t)(unsigned)box->x + box->width >
          u_minify(resource->width0, level) ||
       (uint64_t)(unsigned)box->y + box->height >
          (one_d_array ? resource->array_size
                       : u_minify(resource->height0, level)) ||
       (uint64_t)(unsigned)box->z + box->depth >
          (one_d_array ? 1 : resource->target == PIPE_TEXTURE_3D
             ? u_minify(resource->depth0, level) : resource->array_size) ||
       (compressed &&
        (((unsigned)box->x % block_width) ||
         ((unsigned)box->y % block_height) ||
         (((unsigned)box->width % block_width) &&
          (unsigned)box->x + (unsigned)box->width != level_width) ||
         (((unsigned)box->height % block_height) &&
          (unsigned)box->y + (unsigned)box->height != level_height))) ||
       stride < row_bytes || required_data > SIZE_MAX ||
       layer_pitch < required_layer || layer_pitch % stride != 0) {
      /* Named for the same reason texture_map's refusals are: a refusal ends
       * the run, so the next thing to implement has to be readable from the
       * log rather than deduced from a line number. */
      if (!ctx->failed)
         fprintf(stderr,
                 "gex-wgpu: texture_subdata refused level=%u/%u usage=0x%x "
                 "box=%d,%d,%d %dx%dx%d target=%u format=%u bpp=%u "
                 "compressed=%d texture=%d stride=%u row=%llu\n",
                 level, resource ? resource->last_level : 0, usage,
                 box ? box->x : -1, box ? box->y : -1, box ? box->z : -1,
                 box ? box->width : -1, box ? box->height : -1,
                 box ? box->depth : -1, resource ? resource->target : 0,
                 resource ? resource->format : 0, bytes_per_pixel,
                 (int)compressed,
                 resource && gex_resource(resource)->texture ? 1 : 0,
                 stride, (unsigned long long)row_bytes);
      gex_fail(ctx);
      return;
   }
   /* A depth-stencil texture is copied one aspect at a time, and only some
    * of those aspects can be copied at all: depth32float and
    * depth32float-stencil8 take a depth copy, depth24plus-stencil8 does not.
    * Encoding it anyway is a validation error, and a validation error poisons
    * the device for every later command rather than failing this one, so the
    * unsupported case returns instead.
    *
    * Returning here leaves the texture without the data, which is a real gap:
    * a program that uploads to a packed depth-stencil texture and reads it
    * back gets stale contents rather than an error. Latching instead was
    * measurably worse -- it ended the run and cost every later case -- and
    * the honest fix is a depth upload through a render pass, which this
    * driver does not have yet. */
   const WGPUTextureFormat wgpu_format =
      gex_wgpu_texture_format(resource->format);
   const bool depth_stencil =
      util_format_is_depth_or_stencil(resource->format);
   /* Only depth16unorm accepts a buffer copy into its depth aspect. WebGPU
    * refuses the rest -- depth32float included, which this checked for
    * depth24plus-stencil8 alone and so let through as a validation error.
    * That is the worst outcome available: it poisons the device for every
    * later command, where declining costs one upload.
    *
    * Reading depth back does not need this restriction and does not have it;
    * see gex_wgpu_read_depth, which resolves through a render pass. Writing
    * would need the mirror of that -- a depth-writing draw -- which is not
    * built. */
   if (depth_stencil && wgpu_format != WGPUTextureFormat_Depth16Unorm) {
      struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
      if (!gex_wgpu_write_depth(ctx, resource, level, box, data, stride)) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr,
                    "gex-wgpu: depth upload failed for %s\n",
                    util_format_short_name(resource->format));
         }
      }
      return;
   }

   WGPUTexelCopyTextureInfo destination = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
   destination.texture = gex_resource(resource)->texture;
   destination.mipLevel = level;
   destination.origin.x = box->x;
   destination.origin.y = one_d_array ? 0 : box->y;
   destination.origin.z = one_d_array ? box->y : box->z;
   destination.aspect = depth_stencil ? WGPUTextureAspect_DepthOnly
                                      : WGPUTextureAspect_All;
   /* A format with a wider backing is uploaded through a conversion: the
    * caller's bytes are laid out for the format GL asked for, and the texture
    * holds the backing one. util_format_translate does it from Mesa's own
    * description of both, so nothing here has to know that R16_UNORM is
    * sixteen bits of unsigned normalized -- and the round trip is exact. */
   const enum pipe_format backing = gex_wgpu_format_backing(resource->format);
   const void *upload = data;
   unsigned upload_stride = stride;
   uint64_t upload_layer = layer_pitch;
   size_t upload_size = (size_t)required_data;
   uint8_t *converted = NULL;
   if (backing != PIPE_FORMAT_NONE) {
      const unsigned dst_stride =
         (unsigned)box->width * util_format_get_blocksize(backing);
      const uint64_t dst_layer = (uint64_t)dst_stride * box->height;
      const uint64_t dst_size = dst_layer * (unsigned)box->depth;
      converted = dst_size <= SIZE_MAX ? MALLOC((size_t)dst_size) : NULL;
      if (!converted) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr, "gex-wgpu: no memory to convert %s upload\n",
                    util_format_short_name(resource->format));
         }
         return;
      }
      util_format_translate_3d(backing, converted, dst_stride, dst_layer,
                               0, 0, 0, resource->format, data, stride,
                               layer_pitch, 0, 0, 0,
                               (unsigned)box->width, (unsigned)box->height,
                               (unsigned)box->depth);
      upload = converted;
      upload_stride = dst_stride;
      upload_layer = dst_layer;
      upload_size = (size_t)dst_size;
   }
   WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
   layout.bytesPerRow = upload_stride;
   layout.rowsPerImage = one_d_array ? 1 : upload_layer / upload_stride;
   WGPUExtent3D extent = {
      compressed ? ALIGN_POT((uint32_t)box->width, block_width)
                 : (uint32_t)box->width,
      one_d_array ? 1u
                  : compressed
                  ? ALIGN_POT((uint32_t)box->height, block_height)
                  : (uint32_t)box->height,
      one_d_array ? (uint32_t)box->height : (uint32_t)box->depth,
   };
   gex_resource(resource)->texture_written = true;
   wgpuQueueWriteTexture(gex_screen(pipe->screen)->queue, &destination,
                         upload, upload_size, &layout, &extent);
   FREE(converted);
}

static struct pipe_sampler_view *
gex_wgpu_create_sampler_view(struct pipe_context *pipe,
                             struct pipe_resource *texture,
                             const struct pipe_sampler_view *templ)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   GEX_ENTRY_TIMER();
   /* A refused sampler view is returned as NULL, and not every caller checks:
    * u_blitter builds one for its blit source and dereferences it, which is a
    * null read at offset 4 -- the fault that takes down the whole module. So
    * say which view was refused and why, once per reason. */
#define GEX_SAMPLER_VIEW_REFUSED(why)                                         \
   do {                                                                       \
      static bool said;                                                        \
      if (!said) {                                                             \
         said = true;                                                          \
         fprintf(stderr,                                                       \
                 "gex-wgpu: sampler view refused (%s) format=%s texture=%s "   \
                 "target=%u levels=%u..%u layers=%u..%u\n",                    \
                 why, util_format_short_name(templ->format),                   \
                 util_format_short_name(texture->format), texture->target,     \
                 templ->u.tex.first_level, templ->u.tex.last_level,            \
                 templ->u.tex.first_layer, templ->u.tex.last_layer);           \
      }                                                                        \
   } while (0)

   if (!texture ||
      (texture->target != PIPE_TEXTURE_1D &&
        texture->target != PIPE_TEXTURE_1D_ARRAY &&
        texture->target != PIPE_TEXTURE_2D &&
        texture->target != PIPE_TEXTURE_RECT &&
        texture->target != PIPE_TEXTURE_2D_ARRAY &&
        texture->target != PIPE_TEXTURE_3D &&
        texture->target != PIPE_TEXTURE_CUBE &&
        texture->target != PIPE_TEXTURE_CUBE_ARRAY) ||
       (gex_wgpu_format_bytes(texture->format) == 0 &&
        !gex_wgpu_is_bc_format(texture->format) &&
        texture->format != PIPE_FORMAT_Z32_FLOAT &&
        texture->format != PIPE_FORMAT_Z24_UNORM_S8_UINT) ||
       templ->target != texture->target ||
       /* sRGB and its linear counterpart are the same bytes seen through a
        * different transfer function, and GL uses exactly that view to read
        * sRGB data undecoded. WebGPU allows it too, provided the alternative
        * was declared at creation -- which resource_create already does, in
        * viewFormats.
        *
        * Requiring exact equality refused it, and the caller that asks is
        * u_blitter, which does not null-check the view it gets back. The
        * dereference of that NULL is a read at offset 4 of address zero: it
        * faults the module, leaves Mesa's context-texture lock held, and
        * every later glReadPixels then deadlocks on it. One refused view,
        * and the rest of the run is gone. */
       util_format_linear(templ->format) !=
          util_format_linear(texture->format) ||
       templ->swizzle_r > PIPE_SWIZZLE_1 ||
       templ->swizzle_g > PIPE_SWIZZLE_1 ||
       templ->swizzle_b > PIPE_SWIZZLE_1 ||
       templ->swizzle_a > PIPE_SWIZZLE_1 ||
       /* A depth texture is swizzled like any other. GL hands depth to a
        * shader as (r, r, r, 1), so demanding identity here refused the
        * ordinary case -- and a refused view is a null one, which the
        * fragment path then declines, so the draw never happens. The
        * swizzle is applied in a shader variant, the same one every other
        * swizzled texture uses; nothing about depth needs it done
        * differently. */
       templ->u.tex.first_level > templ->u.tex.last_level ||
       templ->u.tex.last_level > texture->last_level ||
       templ->u.tex.first_layer > templ->u.tex.last_layer ||
       templ->u.tex.last_layer >=
          (texture->target == PIPE_TEXTURE_3D
             ? texture->depth0 : texture->array_size) ||
       ((texture->target == PIPE_TEXTURE_1D ||
         texture->target == PIPE_TEXTURE_2D ||
         texture->target == PIPE_TEXTURE_RECT) &&
        (templ->u.tex.first_layer != 0 || templ->u.tex.last_layer != 0)) ||
       (texture->target == PIPE_TEXTURE_3D &&
        (templ->u.tex.first_layer != 0 ||
         templ->u.tex.last_layer != texture->depth0 - 1)) ||
       (texture->target == PIPE_TEXTURE_CUBE &&
        (templ->u.tex.first_layer != 0 || templ->u.tex.last_layer != 5))) {
      GEX_SAMPLER_VIEW_REFUSED("shape or format");
      return NULL;
   }
   if (texture->target == PIPE_TEXTURE_CUBE_ARRAY &&
       (templ->u.tex.first_layer % 6 != 0 ||
        (templ->u.tex.last_layer + 1) % 6 != 0)) {
      GEX_SAMPLER_VIEW_REFUSED("cube-array layer range");
      return NULL;
   }
   struct gex_wgpu_sampler_view *view = CALLOC_STRUCT(gex_wgpu_sampler_view);
   if (!view)
      return NULL;
   WGPUTextureViewDescriptor desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   /* The view's format, not the texture's. GL's whole use of a differing view
    * format is to read sRGB data undecoded (or to write it encoded), and
    * naming the texture's format here gave the caller a decoding view of an
    * sRGB texture it had explicitly asked to see linearly -- the check above
    * already permits exactly that pair, and resource_create already declared
    * it in viewFormats, so the only thing missing was using it. */
   desc.format = gex_wgpu_texture_format(templ->format);
   desc.dimension = texture->target == PIPE_TEXTURE_1D_ARRAY ||
                    texture->target == PIPE_TEXTURE_2D_ARRAY
      ? WGPUTextureViewDimension_2DArray
      : texture->target == PIPE_TEXTURE_3D
      ? WGPUTextureViewDimension_3D
      : texture->target == PIPE_TEXTURE_CUBE
      ? WGPUTextureViewDimension_Cube
      : texture->target == PIPE_TEXTURE_CUBE_ARRAY
      ? WGPUTextureViewDimension_CubeArray
      : WGPUTextureViewDimension_2D;
   desc.baseMipLevel = templ->u.tex.first_level;
   desc.mipLevelCount = templ->u.tex.last_level - templ->u.tex.first_level + 1;
   if (texture->target != PIPE_TEXTURE_3D) {
      desc.baseArrayLayer = templ->u.tex.first_layer;
      desc.arrayLayerCount =
         templ->u.tex.last_layer - templ->u.tex.first_layer + 1;
   }
   /* Same rule as the depth resolve: a depth-only view of a combined format
    * must name the depth-only format, or WebGPU rejects the pair. */
   if (util_format_is_depth_or_stencil(texture->format)) {
      const WGPUTextureFormat packed = gex_wgpu_texture_format(texture->format);
      desc.aspect = WGPUTextureAspect_DepthOnly;
      desc.format = packed == WGPUTextureFormat_Depth24PlusStencil8
         ? WGPUTextureFormat_Depth24Plus
         : packed == WGPUTextureFormat_Depth32FloatStencil8
            ? WGPUTextureFormat_Depth32Float
            : packed;
   } else {
      desc.aspect = WGPUTextureAspect_All;
   }
   view->view = wgpuTextureCreateView(gex_resource(texture)->texture, &desc);
   if (!view->view) {
      FREE(view);
      return NULL;
   }
   if (texture->target == PIPE_TEXTURE_CUBE ||
       texture->target == PIPE_TEXTURE_CUBE_ARRAY) {
      desc.dimension = WGPUTextureViewDimension_2DArray;
      view->array_view =
         wgpuTextureCreateView(gex_resource(texture)->texture, &desc);
      if (!view->array_view) {
         wgpuTextureViewRelease(view->view);
         FREE(view);
         return NULL;
      }
   }
   view->base = *templ;
   view->base.context = pipe;
   view->base.texture = NULL;
   pipe_reference_init(&view->base.reference, 1);
   pipe_resource_reference(&view->base.texture, texture);
   return &view->base;
}

/* Release a sampler view whose context may already be going away.
 *
 * st/mesa keeps a "zombie" list of views whose context outlived their use and
 * drains it during teardown, calling this without checking that the driver
 * set it.  This driver's views hold nothing tied to a live context beyond the
 * WebGPU handles, so releasing is the same work as destroying.
 */
static void
gex_wgpu_sampler_view_release(struct pipe_context *pipe,
                              struct pipe_sampler_view *base);

static void
gex_wgpu_sampler_view_destroy(struct pipe_context *pipe,
                              struct pipe_sampler_view *base)
{
   /* The bind group cache keys on the entry bytes, and those name
    * this object by address -- which Dawn hands straight back for the
    * next one. Retire every cached binding rather than let a key match
    * a view that no longer exists. */
   ++gex_screen(pipe->screen)->cache_epoch;
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   GEX_ENTRY_TIMER();
   (void)pipe;
   struct gex_wgpu_sampler_view *view = (struct gex_wgpu_sampler_view *)base;
   pipe_resource_reference(&view->base.texture, NULL);
   if (view->view)
      wgpuTextureViewRelease(view->view);
   if (view->array_view)
      wgpuTextureViewRelease(view->array_view);
   FREE(view);
}

/* Drop a reference, which is not the same as destroying.
 *
 * This destroyed the view outright, so any view held more than once -- which
 * st/mesa does as soon as a context keeps its own reference -- was freed while
 * still referenced, and the next release of it decremented a reference count
 * through a dangling pointer. In wasm that surfaces as an unaligned atomic,
 * nowhere near the cause. u_default_sampler_view_release is the contract:
 * release means unreference, and destroy happens at zero. */
static void
gex_wgpu_sampler_view_release(struct pipe_context *pipe,
                              struct pipe_sampler_view *base)
{
   (void)pipe;
   pipe_sampler_view_reference(&base, NULL);
}

static void *
gex_wgpu_create_sampler_state(struct pipe_context *pipe,
                              const struct pipe_sampler_state *state)
{
   GEX_ENTRY_TIMER();
   const bool border = state->wrap_s == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
                       state->wrap_t == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
                       state->wrap_r == PIPE_TEX_WRAP_CLAMP_TO_BORDER;
   const bool comparison =
      state->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE;
   if ((state->min_img_filter != PIPE_TEX_FILTER_NEAREST &&
        state->min_img_filter != PIPE_TEX_FILTER_LINEAR) ||
       (state->mag_img_filter != PIPE_TEX_FILTER_NEAREST &&
        state->mag_img_filter != PIPE_TEX_FILTER_LINEAR) ||
       (state->wrap_s != PIPE_TEX_WRAP_CLAMP_TO_EDGE &&
        state->wrap_s != PIPE_TEX_WRAP_REPEAT &&
        state->wrap_s != PIPE_TEX_WRAP_MIRROR_REPEAT &&
        state->wrap_s != PIPE_TEX_WRAP_CLAMP_TO_BORDER) ||
       (state->wrap_t != PIPE_TEX_WRAP_CLAMP_TO_EDGE &&
        state->wrap_t != PIPE_TEX_WRAP_REPEAT &&
        state->wrap_t != PIPE_TEX_WRAP_MIRROR_REPEAT &&
        state->wrap_t != PIPE_TEX_WRAP_CLAMP_TO_BORDER) ||
       (state->wrap_r != PIPE_TEX_WRAP_CLAMP_TO_EDGE &&
        state->wrap_r != PIPE_TEX_WRAP_REPEAT &&
        state->wrap_r != PIPE_TEX_WRAP_MIRROR_REPEAT &&
        state->wrap_r != PIPE_TEX_WRAP_CLAMP_TO_BORDER) ||
       (state->min_mip_filter != PIPE_TEX_MIPFILTER_NONE &&
        state->min_mip_filter != PIPE_TEX_MIPFILTER_NEAREST &&
        state->min_mip_filter != PIPE_TEX_MIPFILTER_LINEAR) ||
       state->min_lod < 0.0f || state->max_lod < state->min_lod ||
       !isfinite(state->lod_bias) || state->lod_bias < -15.0f ||
       state->lod_bias > 15.0f)
      return NULL;
   if (state->compare_mode != PIPE_TEX_COMPARE_NONE && !comparison)
      return NULL;
   if (comparison && (state->compare_func < PIPE_FUNC_NEVER ||
                      state->compare_func > PIPE_FUNC_ALWAYS))
      return NULL;
   if (border && (comparison ||
                  state->min_img_filter != state->mag_img_filter ||
                  state->min_mip_filter != PIPE_TEX_MIPFILTER_NONE ||
                  state->min_lod != 0.0f || state->max_lod != 0.0f ||
                  (!state->border_color_is_integer &&
                   (!isfinite(state->border_color.f[0]) ||
                    !isfinite(state->border_color.f[1]) ||
                    !isfinite(state->border_color.f[2]) ||
                    !isfinite(state->border_color.f[3])))))
      return NULL;
   struct pipe_sampler_state supported = {0};
   supported.wrap_s = state->wrap_s;
   supported.wrap_t = state->wrap_t;
   supported.wrap_r = state->wrap_r;
   supported.min_img_filter = state->min_img_filter;
   supported.min_mip_filter = state->min_mip_filter;
   supported.mag_img_filter = state->mag_img_filter;
   supported.compare_mode = state->compare_mode;
   supported.compare_func = state->compare_func;
   supported.max_anisotropy = state->max_anisotropy;
   supported.min_lod = state->min_lod;
   supported.max_lod = state->max_lod;
   supported.lod_bias = state->lod_bias;
   supported.border_color = state->border_color;
   supported.border_color_is_integer = state->border_color_is_integer;
   supported.border_color_format = state->border_color_format;
   /* Rectangle coordinates are normalized in NIR before WebGPU emission, so
    * this Gallium bit participates in validation/variant identity but does
    * not change the physical WebGPU sampler descriptor. */
   supported.unnormalized_coords = state->unnormalized_coords;
   if (state->max_anisotropy > 1 ||
       memcmp(state, &supported, sizeof(supported)) != 0)
      return NULL;
   WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
   desc.addressModeU = state->wrap_s == PIPE_TEX_WRAP_REPEAT
      ? WGPUAddressMode_Repeat
      : state->wrap_s == PIPE_TEX_WRAP_MIRROR_REPEAT
      ? WGPUAddressMode_MirrorRepeat : WGPUAddressMode_ClampToEdge;
   desc.addressModeV = state->wrap_t == PIPE_TEX_WRAP_REPEAT
      ? WGPUAddressMode_Repeat
      : state->wrap_t == PIPE_TEX_WRAP_MIRROR_REPEAT
      ? WGPUAddressMode_MirrorRepeat : WGPUAddressMode_ClampToEdge;
   desc.addressModeW = state->wrap_r == PIPE_TEX_WRAP_REPEAT
      ? WGPUAddressMode_Repeat
      : state->wrap_r == PIPE_TEX_WRAP_MIRROR_REPEAT
      ? WGPUAddressMode_MirrorRepeat : WGPUAddressMode_ClampToEdge;
   desc.magFilter = state->mag_img_filter == PIPE_TEX_FILTER_LINEAR
      ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
   desc.minFilter = state->min_img_filter == PIPE_TEX_FILTER_LINEAR
      ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
   desc.mipmapFilter = state->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR
      ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
   desc.lodMinClamp = state->min_lod;
   desc.lodMaxClamp = state->max_lod;
   desc.maxAnisotropy = 1;
   if (comparison)
      desc.compare = gex_wgpu_compare(state->compare_func);
   struct gex_wgpu_sampler *sampler = CALLOC_STRUCT(gex_wgpu_sampler);
   if (!sampler)
      return NULL;
   sampler->sampler = wgpuDeviceCreateSampler(gex_screen(pipe->screen)->device,
                                               &desc);
   if (!sampler->sampler) {
      FREE(sampler);
      return NULL;
   }
   sampler->state = *state;
   return sampler;
}

static void
gex_wgpu_bind_sampler_states(struct pipe_context *pipe,
                             mesa_shader_stage shader, unsigned start_slot,
                             unsigned num_samplers, void **samplers)
{
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (start_slot >= GEX_WGPU_MAX_SAMPLERS ||
       num_samplers > GEX_WGPU_MAX_SAMPLERS - start_slot) {
      gex_fail(ctx);
      return;
   }

   if (shader != MESA_SHADER_FRAGMENT && shader != MESA_SHADER_VERTEX) {
      /* Clearing another stage's samplers still has to succeed: cso_context
       * unbinds every stage whenever a context is bound or torn down, and
       * refusing that would fail contexts that never sample outside the
       * stages implemented here. Binding a real sampler there is a different
       * matter. */
      for (unsigned i = 0; samplers && i < num_samplers; ++i) {
         if (samplers[i]) {
            gex_fail(ctx);
            return;
         }
      }
      return;
   }
   struct gex_wgpu_sampler **slots = shader == MESA_SHADER_VERTEX
      ? ctx->vertex_samplers : ctx->samplers;

   /* A null array means "unbind this range", which is how Gallium spells it. */
   for (unsigned i = 0; i < num_samplers; ++i)
      slots[start_slot + i] = samplers ? samplers[i] : NULL;
   if (!num_samplers && !samplers)
      slots[start_slot] = NULL;
}

static void
gex_wgpu_delete_sampler_state(struct pipe_context *pipe, void *state)
{
   /* The bind group cache keys on the entry bytes, and those name
    * this object by address -- which Dawn hands straight back for the
    * next one. Retire every cached binding rather than let a key match
    * a sampler that no longer exists. */
   ++gex_screen(pipe->screen)->cache_epoch;
   GEX_ENTRY_TIMER();
   (void)pipe;
   struct gex_wgpu_sampler *sampler = state;
   if (sampler && sampler->sampler)
      wgpuSamplerRelease(sampler->sampler);
   FREE(sampler);
}

static void
gex_wgpu_set_sampler_views(struct pipe_context *pipe,
                           mesa_shader_stage shader, unsigned start_slot,
                           unsigned num_views, unsigned unbind_num_trailing_slots,
                           struct pipe_sampler_view **views)
{
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (shader != MESA_SHADER_FRAGMENT && shader != MESA_SHADER_VERTEX) {
      /* Clearing another stage has to succeed, for the same reason
       * bind_sampler_states above says: cso_context unbinds every stage
       * whenever a context is bound or torn down, so refusing an unbind
       * fails contexts that never sample outside the fragment shader.
       *
       * Found by running a real application. SuperTuxKart creates its
       * context, st/mesa unbinds the vertex stage's views as part of ordinary
       * state setup, and this latched -- so every later draw was dropped and
       * the game rendered nothing at all while reporting no GL error. */
      for (unsigned i = 0; views && i < num_views; ++i) {
         if (views[i]) {
            gex_fail(ctx);
            return;
         }
      }
      return;
   }
   struct pipe_sampler_view **slots = shader == MESA_SHADER_VERTEX
      ? ctx->vertex_sampler_views : ctx->sampler_views;
   if (start_slot > GEX_WGPU_MAX_SAMPLERS ||
       num_views > GEX_WGPU_MAX_SAMPLERS - start_slot ||
       unbind_num_trailing_slots >
          GEX_WGPU_MAX_SAMPLERS - start_slot - num_views ||
       (num_views && !views)) {
      gex_fail(ctx);
      return;
   }
   for (unsigned i = 0; i < num_views; ++i)
      pipe_sampler_view_reference(&slots[start_slot + i], views[i]);
   for (unsigned i = 0; i < unbind_num_trailing_slots; ++i)
      pipe_sampler_view_reference(&slots[start_slot + num_views + i], NULL);
}

static void
gex_wgpu_map_callback(WGPUMapAsyncStatus status, WGPUStringView message,
                      void *userdata1, void *userdata2)
{
   (void)userdata2;
   struct gex_wgpu_map_state *state = userdata1;
   state->status = status;
   if (message.data && message.length) {
      const size_t n = MIN2(message.length, sizeof(state->message) - 1);
      memcpy(state->message, message.data, n);
      state->message[n] = '\0';
   }
   state->done = true;
   gex_wgpu_jspi_signal();
}

/* Which depth-upload pipeline a format needs: the depth-stencil state has to
 * name the attachment's exact format, so one pipeline per format this driver
 * can attach. */
static int
gex_wgpu_depth_upload_slot(WGPUTextureFormat format)
{
   /* Every depth format this driver can name, not just the three with
    * stencil. `Depth16Unorm` and `Depth24Plus` were missing, so a depth-only
    * framebuffer in either of them had no upload pipeline and no resolve
    * pipeline -- and the lookup failed silently, which is how
    * `gl44-msaa-depth-blit-probe` came back with a latched context and no
    * explanation. Its destination is `Depth16Unorm`. */
   switch (format) {
   case WGPUTextureFormat_Depth16Unorm: return 0;
   case WGPUTextureFormat_Depth24Plus: return 1;
   case WGPUTextureFormat_Depth24PlusStencil8: return 2;
   case WGPUTextureFormat_Depth32Float: return 3;
   case WGPUTextureFormat_Depth32FloatStencil8: return 4;
   default: return -1;
   }
}

static bool
gex_wgpu_depth_upload_pipeline(struct gex_wgpu_screen *screen,
                               WGPUTextureFormat format)
{
   const int slot = gex_wgpu_depth_upload_slot(format);
   if (slot < 0)
      return false;
   if (screen->depth_upload[slot])
      return true;

   static const char wgsl[] =
      "@group(0) @binding(0) var src: texture_2d<f32>;\n"
      "@vertex fn vs(@builtin(vertex_index) i: u32)"
      " -> @builtin(position) vec4<f32> {\n"
      "  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0),"
      " vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n"
      "  return vec4<f32>(p[i], 0.0, 1.0);\n"
      "}\n"
      "@fragment fn fs(@builtin(position) pos: vec4<f32>)"
      " -> @builtin(frag_depth) f32 {\n"
      "  return textureLoad(src, vec2<i32>(i32(pos.x), i32(pos.y)), 0).r;\n"
      "}\n";

   WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
   source.code = (WGPUStringView){wgsl, WGPU_STRLEN};
   WGPUShaderModuleDescriptor module_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
   module_desc.nextInChain = &source.chain;
   WGPUShaderModule module =
      wgpuDeviceCreateShaderModule(screen->device, &module_desc);
   if (!module)
      return false;

   if (!screen->depth_upload_layout) {
      WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      entry.binding = 0;
      entry.visibility = WGPUShaderStage_Fragment;
      entry.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
      entry.texture.viewDimension = WGPUTextureViewDimension_2D;
      WGPUBindGroupLayoutDescriptor layout_desc =
         WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
      layout_desc.entryCount = 1;
      layout_desc.entries = &entry;
      screen->depth_upload_layout =
         wgpuDeviceCreateBindGroupLayout(screen->device, &layout_desc);
   }

   WGPUPipelineLayoutDescriptor pipeline_layout_desc =
      WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
   pipeline_layout_desc.bindGroupLayoutCount = 1;
   pipeline_layout_desc.bindGroupLayouts = &screen->depth_upload_layout;
   WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(screen->device, &pipeline_layout_desc);

   WGPUDepthStencilState depth_state = WGPU_DEPTH_STENCIL_STATE_INIT;
   depth_state.format = format;
   depth_state.depthWriteEnabled = WGPUOptionalBool_True;
   depth_state.depthCompare = WGPUCompareFunction_Always;

   WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
   desc.layout = pipeline_layout;
   desc.vertex.module = module;
   desc.vertex.entryPoint = (WGPUStringView){"vs", WGPU_STRLEN};
   desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
   desc.multisample.count = 1;
   desc.multisample.mask = 0xffffffffu;
   desc.depthStencil = &depth_state;
   WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
   fragment.module = module;
   fragment.entryPoint = (WGPUStringView){"fs", WGPU_STRLEN};
   fragment.targetCount = 0;
   desc.fragment = &fragment;
   screen->depth_upload[slot] =
      wgpuDeviceCreateRenderPipeline(screen->device, &desc);

   if (pipeline_layout)
      wgpuPipelineLayoutRelease(pipeline_layout);
   wgpuShaderModuleRelease(module);
   return screen->depth_upload[slot] != NULL;
}

/* The same shape as the upload pipeline above, with a multisampled depth
 * source instead of a single-sample colour one.
 *
 * GL's depth resolve is not an average -- averaging depth is meaningless --
 * and Mesa's own resolve takes a single sample, so this takes sample zero. */
static bool
gex_wgpu_msaa_depth_pipeline(struct gex_wgpu_screen *screen,
                             WGPUTextureFormat format)
{
   const int slot = gex_wgpu_depth_upload_slot(format);
   if (slot < 0) {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: no depth resolve pipeline slot for WebGPU format "
                 "%d\n", (int)format);
      }
      return false;
   }
   if (screen->msaa_depth[slot])
      return true;

   static const char wgsl[] =
      "@group(0) @binding(0) var src: texture_depth_multisampled_2d;\n"
      "@vertex fn vs(@builtin(vertex_index) i: u32)"
      " -> @builtin(position) vec4<f32> {\n"
      "  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0),"
      " vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n"
      "  return vec4<f32>(p[i], 0.0, 1.0);\n"
      "}\n"
      "@fragment fn fs(@builtin(position) pos: vec4<f32>)"
      " -> @builtin(frag_depth) f32 {\n"
      "  return textureLoad(src, vec2<i32>(i32(pos.x), i32(pos.y)), 0);\n"
      "}\n";

   WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
   source.code = (WGPUStringView){wgsl, WGPU_STRLEN};
   WGPUShaderModuleDescriptor module_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
   module_desc.nextInChain = &source.chain;
   WGPUShaderModule module =
      wgpuDeviceCreateShaderModule(screen->device, &module_desc);
   if (!module)
      return false;

   if (!screen->msaa_depth_layout) {
      WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      entry.binding = 0;
      entry.visibility = WGPUShaderStage_Fragment;
      entry.texture.sampleType = WGPUTextureSampleType_Depth;
      entry.texture.viewDimension = WGPUTextureViewDimension_2D;
      entry.texture.multisampled = true;
      WGPUBindGroupLayoutDescriptor layout_desc =
         WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
      layout_desc.entryCount = 1;
      layout_desc.entries = &entry;
      screen->msaa_depth_layout =
         wgpuDeviceCreateBindGroupLayout(screen->device, &layout_desc);
   }

   WGPUPipelineLayoutDescriptor pipeline_layout_desc =
      WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
   pipeline_layout_desc.bindGroupLayoutCount = 1;
   pipeline_layout_desc.bindGroupLayouts = &screen->msaa_depth_layout;
   WGPUPipelineLayout resolve_layout =
      wgpuDeviceCreatePipelineLayout(screen->device, &pipeline_layout_desc);

   WGPUDepthStencilState depth_state = WGPU_DEPTH_STENCIL_STATE_INIT;
   depth_state.format = format;
   depth_state.depthWriteEnabled = WGPUOptionalBool_True;
   depth_state.depthCompare = WGPUCompareFunction_Always;

   WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
   desc.layout = resolve_layout;
   desc.vertex.module = module;
   desc.vertex.entryPoint = (WGPUStringView){"vs", WGPU_STRLEN};
   desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
   /* The destination is single-sample; only the source is multisampled. */
   desc.multisample.count = 1;
   desc.multisample.mask = 0xffffffffu;
   desc.depthStencil = &depth_state;
   WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
   fragment.module = module;
   fragment.entryPoint = (WGPUStringView){"fs", WGPU_STRLEN};
   fragment.targetCount = 0;
   desc.fragment = &fragment;
   screen->msaa_depth[slot] =
      wgpuDeviceCreateRenderPipeline(screen->device, &desc);
   if (!screen->msaa_depth[slot])
      fprintf(stderr,
              "gex-wgpu: multisampled depth resolve pipeline failed for "
              "format %d (slot %d)\n", (int)format, slot);

   if (resolve_layout)
      wgpuPipelineLayoutRelease(resolve_layout);
   wgpuShaderModuleRelease(module);
   return screen->msaa_depth[slot] != NULL;
}

/* Resolve a multisampled depth texture into a single-sample one. */
static bool
gex_wgpu_resolve_depth(struct pipe_context *pipe, struct pipe_resource *dst,
                       struct pipe_resource *src)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   /* Submits its own command buffer, so anything already recorded has to
    * reach the queue first or it would run after this. */
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   const WGPUTextureFormat wgpu = gex_wgpu_texture_format(dst->format);
   if (!gex_wgpu_msaa_depth_pipeline(screen, wgpu))
      return false;
   const int slot = gex_wgpu_depth_upload_slot(wgpu);

   WGPUTextureView msaa = wgpuTextureCreateView(gex_resource(src)->texture, NULL);
   WGPUTextureView single =
      wgpuTextureCreateView(gex_resource(dst)->texture, NULL);
   WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   WGPUBindGroup bind = NULL;
   if (msaa) {
      WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
      entry.binding = 0;
      entry.textureView = msaa;
      WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bind_desc.layout = screen->msaa_depth_layout;
      bind_desc.entryCount = 1;
      bind_desc.entries = &entry;
      bind = wgpuDeviceCreateBindGroup(screen->device, &bind_desc);
   }
   bool ok = msaa && single && encoder && bind;
   if (!ok)
      fprintf(stderr,
              "gex-wgpu: depth resolve setup incomplete (msaa=%d single=%d "
              "encoder=%d bind=%d)\n", (int)!!msaa, (int)!!single,
              (int)!!encoder, (int)!!bind);
   if (ok) {
      WGPURenderPassDepthStencilAttachment attachment =
         WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
      attachment.view = single;
      attachment.depthLoadOp = WGPULoadOp_Load;
      attachment.depthStoreOp = WGPUStoreOp_Store;
      /* Finite, for the reason the upload path records: the default is NaN and
       * the browser validates it whatever the load op says. */
      attachment.depthClearValue = 1.0f;
      if (util_format_has_stencil(util_format_description(dst->format))) {
         attachment.stencilLoadOp = WGPULoadOp_Load;
         attachment.stencilStoreOp = WGPUStoreOp_Store;
      }
      WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
      pass_desc.colorAttachmentCount = 0;
      pass_desc.depthStencilAttachment = &attachment;
      WGPURenderPassEncoder pass =
         wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
      ok = pass != NULL;
      if (!pass)
         fprintf(stderr, "gex-wgpu: depth resolve pass did not begin\n");
      if (pass) {
         wgpuRenderPassEncoderSetPipeline(pass, screen->msaa_depth[slot]);
         wgpuRenderPassEncoderSetBindGroup(pass, 0, bind, 0, NULL);
         wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass);
         wgpuRenderPassEncoderRelease(pass);
      }
   }
   if (ok) {
      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      ok = commands != NULL;
      if (!commands)
         fprintf(stderr, "gex-wgpu: depth resolve command buffer failed\n");
      if (commands) {
         gex_wgpu_queue_submit(screen, 1, &commands);
         wgpuCommandBufferRelease(commands);
      }
   }
   if (bind) wgpuBindGroupRelease(bind);
   if (encoder) wgpuCommandEncoderRelease(encoder);
   if (single) wgpuTextureViewRelease(single);
   if (msaa) wgpuTextureViewRelease(msaa);
   {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: depth resolve ran (%s %ux%u, %u samples -> 1) "
                 "ok=%d dst=%p tex=%p\n", util_format_short_name(src->format),
                 src->width0, src->height0,
                 MAX2(1u, (unsigned)src->nr_samples), (int)ok,
                 (void *)dst, (void *)gex_resource(dst)->texture);
      }
   }
   return ok;
}

/* Write depth (and stencil, if the format has it) into a texture.
 *
 * The inverse of gex_wgpu_read_depth: unpack Gallium's layout into floats,
 * put those in an r32float texture -- which a buffer copy can fill -- and let
 * a fullscreen draw move them into the depth attachment through
 * @builtin(frag_depth). Stencil is copied straight in; its aspect takes a
 * buffer copy.
 *
 * Returns false having changed nothing. */
static bool
gex_wgpu_write_depth(struct gex_wgpu_context *ctx,
                     struct pipe_resource *resource, unsigned level,
                     const struct pipe_box *box, const uint8_t *src,
                     unsigned src_stride)
{
   GEX_DEVICE_GUARD(gex_screen(ctx->base.screen));
   /* Submits its own command buffer, so anything already recorded has to
    * reach the queue first or it would run after this. */
   gex_wgpu_submit_pending(ctx);
   struct gex_wgpu_screen *screen = gex_screen(ctx->base.screen);
   const enum pipe_format format = resource->format;
   const WGPUTextureFormat wgpu = gex_wgpu_texture_format(format);
   if (!gex_wgpu_depth_upload_pipeline(screen, wgpu))
      return false;

   const int slot = gex_wgpu_depth_upload_slot(wgpu);
   const bool has_stencil =
      util_format_has_stencil(util_format_description(format));
   const unsigned w = box->width, h = box->height;

   float *depth = MALLOC((size_t)w * h * sizeof(float));
   uint8_t *stencil = has_stencil ? MALLOC((size_t)w * h) : NULL;
   if (!depth || (has_stencil && !stencil)) {
      FREE(depth);
      FREE(stencil);
      return false;
   }
   for (unsigned y = 0; y < h; ++y) {
      const uint8_t *row = src + (size_t)y * src_stride;
      for (unsigned x = 0; x < w; ++x) {
         switch (format) {
         case PIPE_FORMAT_Z24_UNORM_S8_UINT: {
            uint32_t packed;
            memcpy(&packed, row + (size_t)x * 4, 4);
            depth[y * w + x] = (float)(packed & 0xffffffu) / 16777215.0f;
            if (stencil) stencil[y * w + x] = (uint8_t)(packed >> 24);
            break;
         }
         case PIPE_FORMAT_Z32_FLOAT:
            memcpy(&depth[y * w + x], row + (size_t)x * 4, 4);
            break;
         case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT: {
            memcpy(&depth[y * w + x], row + (size_t)x * 8, 4);
            uint32_t st;
            memcpy(&st, row + (size_t)x * 8 + 4, 4);
            if (stencil) stencil[y * w + x] = (uint8_t)st;
            break;
         }
         default:
            FREE(depth);
            FREE(stencil);
            return false;
         }
      }
   }

   WGPUTextureDescriptor staging_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   staging_desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
   staging_desc.dimension = WGPUTextureDimension_2D;
   staging_desc.size = (WGPUExtent3D){w, h, 1};
   staging_desc.format = WGPUTextureFormat_R32Float;
   staging_desc.mipLevelCount = 1;
   staging_desc.sampleCount = 1;
   WGPUTexture staging = wgpuDeviceCreateTexture(screen->device, &staging_desc);

   bool ok = staging != NULL;
   if (ok) {
      WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      dst.texture = staging;
      WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
      layout.bytesPerRow = w * 4;
      layout.rowsPerImage = h;
      WGPUExtent3D extent = {w, h, 1};
      wgpuQueueWriteTexture(screen->queue, &dst, depth,
                            (size_t)w * h * sizeof(float), &layout, &extent);
   }

   WGPUTextureView staging_view = ok ? wgpuTextureCreateView(staging, NULL) : NULL;
   WGPUTextureViewDescriptor target_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   target_view_desc.dimension = WGPUTextureViewDimension_2D;
   target_view_desc.baseMipLevel = level;
   target_view_desc.mipLevelCount = 1;
   target_view_desc.baseArrayLayer = box->z;
   target_view_desc.arrayLayerCount = 1;
   WGPUTextureView target_view = ok
      ? wgpuTextureCreateView(gex_resource(resource)->texture, &target_view_desc)
      : NULL;

   WGPUBindGroup bind = NULL;
   if (staging_view) {
      WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
      entry.binding = 0;
      entry.textureView = staging_view;
      WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bind_desc.layout = screen->depth_upload_layout;
      bind_desc.entryCount = 1;
      bind_desc.entries = &entry;
      bind = wgpuDeviceCreateBindGroup(screen->device, &bind_desc);
   }
   ok = ok && staging_view && target_view && bind;

   if (ok) {
      WGPUCommandEncoder encoder =
         wgpuDeviceCreateCommandEncoder(screen->device, NULL);
      WGPURenderPassDepthStencilAttachment attachment =
         WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
      attachment.view = target_view;
      attachment.depthLoadOp = WGPULoadOp_Load;
      attachment.depthStoreOp = WGPUStoreOp_Store;
      /* Set even though the load op is Load and will not use them: the value
       * defaults to NaN, and the browser validates it as a float regardless,
       * rejecting the pass with "The provided float value is non-finite". */
      attachment.depthClearValue = 1.0f;
      attachment.stencilClearValue = 0;
      if (has_stencil) {
         attachment.stencilLoadOp = WGPULoadOp_Load;
         attachment.stencilStoreOp = WGPUStoreOp_Store;
      }
      WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
      pass_desc.colorAttachmentCount = 0;
      pass_desc.depthStencilAttachment = &attachment;
      WGPURenderPassEncoder pass =
         wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
      /* The draw covers the whole attachment; the viewport limits it to the
       * box the caller asked about, so the rest keeps its contents. */
      wgpuRenderPassEncoderSetViewport(pass, (float)box->x, (float)box->y,
                                       (float)w, (float)h, 0.0f, 1.0f);
      wgpuRenderPassEncoderSetPipeline(pass, screen->depth_upload[slot]);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, bind, 0, NULL);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass);
      wgpuRenderPassEncoderRelease(pass);

      if (has_stencil) {
         WGPUTexelCopyTextureInfo sdst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
         sdst.texture = gex_resource(resource)->texture;
         sdst.mipLevel = level;
         sdst.origin.x = box->x;
         sdst.origin.y = box->y;
         sdst.aspect = WGPUTextureAspect_StencilOnly;
         WGPUTexelCopyBufferLayout slayout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
         slayout.bytesPerRow = w;
         slayout.rowsPerImage = h;
         WGPUExtent3D sextent = {w, h, 1};
         wgpuQueueWriteTexture(screen->queue, &sdst, stencil, (size_t)w * h,
                               &slayout, &sextent);
      }

      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      if (commands) {
         gex_wgpu_queue_submit(screen, 1, &commands);
         wgpuCommandBufferRelease(commands);
      } else {
         ok = false;
      }
      wgpuCommandEncoderRelease(encoder);
   }

   if (bind) wgpuBindGroupRelease(bind);
   if (target_view) wgpuTextureViewRelease(target_view);
   if (staging_view) wgpuTextureViewRelease(staging_view);
   if (staging) wgpuTextureRelease(staging);
   FREE(depth);
   FREE(stencil);
   return ok;
}

/* Read a depth (or depth-stencil) texture back into Gallium's layout.
 *
 * Depth comes through the resolve pass above, into an r32float texture that a
 * buffer copy understands. Stencil comes straight out of its own aspect,
 * which WebGPU copies without complaint. The two are then packed the way the
 * pipe_format says, so st/mesa's readers see what they expect.
 *
 * Returns false having changed nothing, so the caller can still decline. */
static bool
gex_wgpu_read_depth(struct gex_wgpu_context *ctx,
                    struct pipe_resource *resource, unsigned level,
                    const struct pipe_box *box, uint8_t *out,
                    unsigned out_stride);

/* The fullscreen pass that turns a depth texture into r32float.
 *
 * Built on first use and kept on the screen. The shader is written directly
 * in WGSL rather than compiled from NIR: it is three lines, it never varies,
 * and routing it through the NIR path would mean building a Gallium shader
 * state for something no Gallium caller ever asks for. */
static bool
gex_wgpu_depth_resolve_pipeline(struct gex_wgpu_screen *screen)
{
   if (screen->depth_resolve)
      return true;

   static const char wgsl[] =
      "@group(0) @binding(0) var src: texture_depth_2d;\n"
      "@vertex fn vs(@builtin(vertex_index) i: u32)"
      " -> @builtin(position) vec4<f32> {\n"
      "  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0),"
      " vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n"
      "  return vec4<f32>(p[i], 0.0, 1.0);\n"
      "}\n"
      "@fragment fn fs(@builtin(position) pos: vec4<f32>)"
      " -> @location(0) f32 {\n"
      "  return textureLoad(src, vec2<i32>(i32(pos.x), i32(pos.y)), 0);\n"
      "}\n";

   WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
   source.code = (WGPUStringView){wgsl, WGPU_STRLEN};
   WGPUShaderModuleDescriptor module_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
   module_desc.nextInChain = &source.chain;
   WGPUShaderModule module =
      wgpuDeviceCreateShaderModule(screen->device, &module_desc);
   if (!module)
      return false;

   WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
   entry.binding = 0;
   entry.visibility = WGPUShaderStage_Fragment;
   entry.texture.sampleType = WGPUTextureSampleType_Depth;
   entry.texture.viewDimension = WGPUTextureViewDimension_2D;
   WGPUBindGroupLayoutDescriptor layout_desc =
      WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
   layout_desc.entryCount = 1;
   layout_desc.entries = &entry;
   screen->msaa_depth_layout =
      wgpuDeviceCreateBindGroupLayout(screen->device, &layout_desc);

   WGPUPipelineLayoutDescriptor pipeline_layout_desc =
      WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
   pipeline_layout_desc.bindGroupLayoutCount = 1;
   pipeline_layout_desc.bindGroupLayouts = &screen->msaa_depth_layout;
   WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(screen->device, &pipeline_layout_desc);

   WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
   target.format = WGPUTextureFormat_R32Float;
   target.writeMask = WGPUColorWriteMask_All;
   WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
   fragment.module = module;
   fragment.entryPoint = (WGPUStringView){"fs", WGPU_STRLEN};
   fragment.targetCount = 1;
   fragment.targets = &target;

   WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
   desc.layout = pipeline_layout;
   desc.vertex.module = module;
   desc.vertex.entryPoint = (WGPUStringView){"vs", WGPU_STRLEN};
   desc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
   desc.multisample.count = 1;
   desc.multisample.mask = 0xffffffffu;
   desc.fragment = &fragment;
   screen->depth_resolve = wgpuDeviceCreateRenderPipeline(screen->device, &desc);

   if (pipeline_layout)
      wgpuPipelineLayoutRelease(pipeline_layout);
   wgpuShaderModuleRelease(module);
   return screen->depth_resolve != NULL;
}

static bool
gex_wgpu_read_depth(struct gex_wgpu_context *ctx,
                    struct pipe_resource *resource, unsigned level,
                    const struct pipe_box *box, uint8_t *out,
                    unsigned out_stride)
{
   GEX_DEVICE_GUARD(gex_screen(ctx->base.screen));
   /* Submits its own command buffer, so anything already recorded has to
    * reach the queue first or it would run after this. */
   gex_wgpu_submit_pending(ctx);
   struct gex_wgpu_screen *screen = gex_screen(ctx->base.screen);
   if (!gex_wgpu_depth_resolve_pipeline(screen))
      return false;

   const enum pipe_format format = resource->format;
   const bool has_stencil = util_format_has_stencil(util_format_description(format));
   const unsigned width = u_minify(resource->width0, level);
   const unsigned height = u_minify(resource->height0, level);

   /* The resolve target is the whole level, and the box is taken out of it by
    * the copy. Rendering only the box would need the offset in the shader,
    * which means a uniform and a bind group for it; this costs a little
    * bandwidth and keeps the shader constant. */
   WGPUTextureDescriptor colour_desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   colour_desc.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
   colour_desc.dimension = WGPUTextureDimension_2D;
   colour_desc.size = (WGPUExtent3D){width, height, 1};
   colour_desc.format = WGPUTextureFormat_R32Float;
   colour_desc.mipLevelCount = 1;
   colour_desc.sampleCount = 1;
   WGPUTexture colour = wgpuDeviceCreateTexture(screen->device, &colour_desc);
   if (!colour)
      return false;

   /* A depth-only view of a combined format is not that format: WebGPU wants
    * the depth-only counterpart, and rejects the packed name against
    * TextureAspect::DepthOnly. Left at the default (the texture's own format)
    * this was a validation error, which poisons the device. */
   const WGPUTextureFormat texture_format = gex_wgpu_texture_format(format);
   WGPUTextureViewDescriptor src_view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   src_view_desc.format =
      texture_format == WGPUTextureFormat_Depth24PlusStencil8
         ? WGPUTextureFormat_Depth24Plus
      : texture_format == WGPUTextureFormat_Depth32FloatStencil8
         ? WGPUTextureFormat_Depth32Float
         : texture_format;
   src_view_desc.dimension = WGPUTextureViewDimension_2D;
   src_view_desc.baseMipLevel = level;
   src_view_desc.mipLevelCount = 1;
   src_view_desc.baseArrayLayer = 0;
   src_view_desc.arrayLayerCount = 1;
   src_view_desc.aspect = WGPUTextureAspect_DepthOnly;
   WGPUTextureView src_view =
      wgpuTextureCreateView(gex_resource(resource)->texture, &src_view_desc);
   WGPUTextureView colour_view = wgpuTextureCreateView(colour, NULL);

   WGPUBindGroupEntry bind_entry = WGPU_BIND_GROUP_ENTRY_INIT;
   bind_entry.binding = 0;
   bind_entry.textureView = src_view;
   WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bind_desc.layout = screen->msaa_depth_layout;
   bind_desc.entryCount = 1;
   bind_desc.entries = &bind_entry;
   WGPUBindGroup bind = wgpuDeviceCreateBindGroup(screen->device, &bind_desc);

   const uint32_t depth_row = ((uint32_t)box->width * 4u + 255u) & ~255u;
   const uint64_t depth_size = (uint64_t)depth_row * box->height;
   WGPUBufferDescriptor buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   buffer_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
   buffer_desc.size = depth_size;
   WGPUBuffer depth_buffer = wgpuDeviceCreateBuffer(screen->device, &buffer_desc);

   const uint32_t stencil_row = ((uint32_t)box->width + 255u) & ~255u;
   const uint64_t stencil_size = (uint64_t)stencil_row * box->height;
   WGPUBuffer stencil_buffer = NULL;
   if (has_stencil) {
      buffer_desc.size = stencil_size;
      stencil_buffer = wgpuDeviceCreateBuffer(screen->device, &buffer_desc);
   }

   bool ok = src_view && colour_view && bind && depth_buffer &&
             (!has_stencil || stencil_buffer);
   if (ok) {
      WGPUCommandEncoder encoder =
         wgpuDeviceCreateCommandEncoder(screen->device, NULL);
      WGPURenderPassColorAttachment attachment =
         WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
      attachment.view = colour_view;
      attachment.loadOp = WGPULoadOp_Clear;
      attachment.storeOp = WGPUStoreOp_Store;
      attachment.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
      WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
      pass_desc.colorAttachmentCount = 1;
      pass_desc.colorAttachments = &attachment;
      WGPURenderPassEncoder pass =
         wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
      wgpuRenderPassEncoderSetPipeline(pass, screen->depth_resolve);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, bind, 0, NULL);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass);
      wgpuRenderPassEncoderRelease(pass);

      WGPUTexelCopyTextureInfo from = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      from.texture = colour;
      from.origin.x = box->x;
      from.origin.y = box->y;
      WGPUTexelCopyBufferInfo to = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
      to.buffer = depth_buffer;
      to.layout.bytesPerRow = depth_row;
      to.layout.rowsPerImage = box->height;
      WGPUExtent3D extent = {box->width, box->height, 1};
      wgpuCommandEncoderCopyTextureToBuffer(encoder, &from, &to, &extent);

      if (has_stencil) {
         WGPUTexelCopyTextureInfo sfrom = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
         sfrom.texture = gex_resource(resource)->texture;
         sfrom.mipLevel = level;
         sfrom.origin.x = box->x;
         sfrom.origin.y = box->y;
         sfrom.aspect = WGPUTextureAspect_StencilOnly;
         WGPUTexelCopyBufferInfo sto = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
         sto.buffer = stencil_buffer;
         sto.layout.bytesPerRow = stencil_row;
         sto.layout.rowsPerImage = box->height;
         wgpuCommandEncoderCopyTextureToBuffer(encoder, &sfrom, &sto, &extent);
      }

      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      if (commands) {
         gex_wgpu_queue_submit(screen, 1, &commands);
         wgpuCommandBufferRelease(commands);
      } else {
         ok = false;
      }
      wgpuCommandEncoderRelease(encoder);
   }

   const float *depth_mapped = NULL;
   const uint8_t *stencil_mapped = NULL;
   if (ok) {
      struct gex_wgpu_map_state state = {WGPUMapAsyncStatus_Success, false};
      WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
      callback.mode = WGPUCallbackMode_AllowSpontaneous;
      callback.callback = gex_wgpu_map_callback;
      callback.userdata1 = &state;
      wgpuBufferMapAsync(depth_buffer, WGPUMapMode_Read, 0, depth_size, callback);
      while (!state.done)
         gex_wgpu_jspi_wait();
      depth_mapped = state.status == WGPUMapAsyncStatus_Success
         ? wgpuBufferGetConstMappedRange(depth_buffer, 0, depth_size) : NULL;
      ok = depth_mapped != NULL;
   }
   if (ok && has_stencil) {
      struct gex_wgpu_map_state state = {WGPUMapAsyncStatus_Success, false};
      WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
      callback.mode = WGPUCallbackMode_AllowSpontaneous;
      callback.callback = gex_wgpu_map_callback;
      callback.userdata1 = &state;
      wgpuBufferMapAsync(stencil_buffer, WGPUMapMode_Read, 0, stencil_size,
                         callback);
      while (!state.done)
         gex_wgpu_jspi_wait();
      stencil_mapped = state.status == WGPUMapAsyncStatus_Success
         ? wgpuBufferGetConstMappedRange(stencil_buffer, 0, stencil_size) : NULL;
      ok = stencil_mapped != NULL;
   }

   if (ok) {
      for (int y = 0; y < box->height; ++y) {
         const float *row =
            (const float *)((const uint8_t *)depth_mapped + (size_t)y * depth_row);
         const uint8_t *srow = stencil_mapped
            ? stencil_mapped + (size_t)y * stencil_row : NULL;
         uint8_t *dst = out + (size_t)y * out_stride;
         for (int x = 0; x < box->width; ++x) {
            const float d = row[x];
            const uint8_t st = srow ? srow[x] : 0;
            switch (format) {
            case PIPE_FORMAT_Z24_UNORM_S8_UINT: {
               /* Depth in the low 24 bits, stencil in the high 8: what
                * util_format calls Z24_UNORM_S8_UINT on a little-endian host. */
               /* Rounded in double, not in float. d * 16777215.0f is not
                * exactly representable in f32 across most of the range, so
                * adding 0.5f and truncating lands one below the correct
                * integer for a quarter of it -- and 1.0 rounds to 16777216,
                * which the mask below turns into depth 0. */
               const uint32_t z = (uint32_t)((double)CLAMP(d, 0.0f, 1.0f) *
                                             16777215.0 + 0.5);
               const uint32_t packed = (z & 0xffffffu) | ((uint32_t)st << 24);
               memcpy(dst + (size_t)x * 4, &packed, 4);
               break;
            }
            case PIPE_FORMAT_Z32_FLOAT:
               memcpy(dst + (size_t)x * 4, &d, 4);
               break;
            case PIPE_FORMAT_Z32_FLOAT_S8X24_UINT: {
               memcpy(dst + (size_t)x * 8, &d, 4);
               const uint32_t s = st;
               memcpy(dst + (size_t)x * 8 + 4, &s, 4);
               break;
            }
            default:
               ok = false;
               break;
            }
         }
      }
   }

   if (depth_mapped)
      wgpuBufferUnmap(depth_buffer);
   if (stencil_mapped)
      wgpuBufferUnmap(stencil_buffer);
   if (depth_buffer) wgpuBufferRelease(depth_buffer);
   if (stencil_buffer) wgpuBufferRelease(stencil_buffer);
   if (bind) wgpuBindGroupRelease(bind);
   if (colour_view) wgpuTextureViewRelease(colour_view);
   if (src_view) wgpuTextureViewRelease(src_view);
   wgpuTextureRelease(colour);
   return ok;
}


static void *
gex_wgpu_texture_map(struct pipe_context *pipe,
                     struct pipe_resource *resource,
                     unsigned level, unsigned usage,
                     const struct pipe_box *box,
                     struct pipe_transfer **out_transfer)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   if (gex_wgpu_batch_conflicts_usage((struct gex_wgpu_context *)pipe,
                                     resource, usage))
      gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   *out_transfer = NULL;
   const bool writing = (usage & PIPE_MAP_WRITE) != 0;
   const bool reading = (usage & PIPE_MAP_READ) != 0;

   /* A map that cannot be served returns NULL and says why; it does not
    * latch the context.
    *
    * The distinction is whether the caller learns of the failure. A draw that
    * is silently dropped makes every later result untrustworthy, which is why
    * gex_fail poisons the context. A map reports failure by returning NULL,
    * and st/mesa turns that into a GL error rather than into wrong pixels --
    * so latching here converts one unsupported readback into thousands of
    * dead cases and buys no correctness at all. */
#define GEX_MAP_REFUSE(why)                                                   \
   do {                                                                       \
      static bool said;                                                       \
      if (!said) {                                                            \
         said = true;                                                         \
         fprintf(stderr,                                                      \
                 "gex-wgpu: texture_map declined (%s) level=%u usage=0x%x "   \
                 "box=%d,%d,%d %dx%dx%d target=%u format=%u\n",               \
                 why, level, usage, box->x, box->y, box->z, box->width,       \
                 box->height, box->depth, resource->target, resource->format);\
      }                                                                       \
      return NULL;                                                            \
   } while (0)

   /* Reported apart from the shape checks: once the context has failed every
    * map declines, and attributing those to a bad box sends the reader after
    * the wrong thing. */
   if (ctx->failed)
      GEX_MAP_REFUSE("context already failed");
   if (box->x < 0 || box->y < 0 || box->z < 0 ||
       box->width <= 0 || box->height <= 0 || box->depth <= 0 ||
       (!reading && !writing))
      GEX_MAP_REFUSE("shape");

   /* A write map is how st/mesa uploads texture data whenever it does not go
    * through texture_subdata, which is most of glTexImage2D. It is staged in
    * ordinary memory and handed to the same upload path on unmap, so every
    * level, layer and block format texture_subdata handles works here too. */
   if (writing && !reading) {
      const bool compressed = gex_wgpu_is_bc_format(resource->format);
      const unsigned block_width = compressed
         ? util_format_get_blockwidth(resource->format) : 1;
      const unsigned block_height = compressed
         ? util_format_get_blockheight(resource->format) : 1;
      const unsigned block_bytes = compressed
         ? util_format_get_blocksize(resource->format)
         : gex_wgpu_format_bytes(resource->format);
      if (!block_bytes)
         GEX_MAP_REFUSE("format has no block size");
      const unsigned row = DIV_ROUND_UP((unsigned)box->width, block_width) *
                           block_bytes;
      const unsigned rows = DIV_ROUND_UP((unsigned)box->height, block_height);
      const uint64_t size = (uint64_t)row * rows * (unsigned)box->depth;
      if (!size || size > UINT32_MAX)
         GEX_MAP_REFUSE("staging size");

      struct gex_wgpu_transfer *transfer = CALLOC_STRUCT(gex_wgpu_transfer);
      if (transfer) ++gex_wgpu_transfers_live;
      if (transfer)
         transfer->data = CALLOC(1, (size_t)size);
      if (!transfer || !transfer->data) {
         if (transfer)
--gex_wgpu_transfers_live;
   --gex_wgpu_transfers_live;
         FREE(transfer);
         GEX_MAP_REFUSE("out of memory");
      }
      transfer->base.resource = NULL;
      pipe_resource_reference(&transfer->base.resource, resource);
      transfer->base.level = level;
      transfer->base.usage = usage;
      transfer->base.box = *box;
      transfer->base.stride = row;
      transfer->base.layer_stride = (uintptr_t)row * rows;
      *out_transfer = &transfer->base;
      return transfer->data;
   }

   /* The read path. A read-write map comes through here too: it reads back
    * now and uploads on unmap, which is what the caller means by asking for
    * both. */
   const unsigned pixel_bytes = gex_wgpu_format_bytes(resource->format);
   /* WebGPU copies one aspect of a depth-stencil texture at a time, and will
    * not copy the depth aspect of depth24plus at all, so a packed read has no
    * legal encoding. Declining is not just tidier than trying: the attempt is
    * a validation error, and a validation error poisons the device for every
    * later command rather than failing this one. */
   /* Not every depth format is uncopyable, only the packed ones. WebGPU will
    * copy the depth aspect of depth16unorm and depth32float; what it refuses
    * is depth24plus, whose depth aspect has no defined memory layout, and
    * depth24plus-stencil8 for the same reason. Refusing all four made
    * GL_DEPTH_COMPONENT16 unreadable for no reason WebGPU imposes. */
   /* A block-compressed texture has no bytes per pixel, only bytes per block,
    * so every size below is computed in blocks for one. glGetTexImage of a
    * compressed texture is how the CTS reads RGTC back, and refusing the map
    * made it "Error during glGetTexImage" 172 times in packed_pixels.rectangle
    * alone. WebGPU copies a compressed texture to a buffer the same way it
    * copies any other, in whole blocks. */
   const bool read_compressed = gex_wgpu_is_bc_format(resource->format);
   const unsigned read_block_w = util_format_get_blockwidth(resource->format);
   const unsigned read_block_h = util_format_get_blockheight(resource->format);
   const unsigned read_block_bytes =
      util_format_get_blocksize(resource->format);
   const unsigned read_row_blocks =
      DIV_ROUND_UP((unsigned)box->width, read_block_w);
   const unsigned read_col_blocks =
      DIV_ROUND_UP((unsigned)box->height, read_block_h);
   const bool depth = util_format_is_depth_or_stencil(resource->format);
   const WGPUTextureFormat depth_wgpu =
      depth ? gex_wgpu_texture_format(resource->format)
            : WGPUTextureFormat_Undefined;
   bool packed_depth = false;
   if (depth && depth_wgpu != WGPUTextureFormat_Depth16Unorm &&
       depth_wgpu != WGPUTextureFormat_Depth32Float)
      packed_depth = true;
   /* `box->z` is a cube face, an array layer or a depth slice, and the copy
    * below addresses all three the same way -- `origin.z`. Refusing a non-zero
    * one made glGetTexImage of any face but the first fail, and st/mesa does
    * not check: it carries on from a NULL map and asks for a buffer of a size
    * computed from nothing, which is how this surfaced -- a negative map width
    * and a 2.3 GB allocation several calls later. SuperTuxKart reads back cube
    * face 1 for its skybox. A multi-slice read is still refused: it would need
    * a staging buffer per slice, and nothing asks. */
   if (level > resource->last_level || box->depth != 1 ||
       /* The packed-depth branch below resolves through a render pass that
        * has no notion of a slice, so a non-zero one there would quietly read
        * layer zero. Refused rather than mis-served; the copy path underneath
        * is the one that gained slices. */
       (packed_depth && box->z != 0) ||
       (!pixel_bytes && !read_compressed) ||
       (read_compressed && (((unsigned)box->x % read_block_w) ||
                            ((unsigned)box->y % read_block_h))) ||
       (uint64_t)(unsigned)box->x + (unsigned)box->width >
          u_minify(resource->width0, level) ||
       (uint64_t)(unsigned)box->y + (unsigned)box->height >
          u_minify(resource->height0, level))
      GEX_MAP_REFUSE("read path supports one layer of a sized format, in bounds");
#undef GEX_MAP_REFUSE

   /* A packed depth format cannot be copied out, so it is resolved through a
    * render pass instead and packed here. Everything below is the ordinary
    * buffer-copy path, which the copyable formats still take. */
   if (packed_depth) {
      struct gex_wgpu_transfer *transfer = CALLOC_STRUCT(gex_wgpu_transfer);
      if (transfer) ++gex_wgpu_transfers_live;
      const unsigned row = (unsigned)box->width * pixel_bytes;
      if (transfer)
         transfer->data = CALLOC(1, (size_t)row * box->height);
      if (!transfer || !transfer->data ||
          !gex_wgpu_read_depth(ctx, resource, level, box, transfer->data, row)) {
         if (transfer) {
            FREE(transfer->data);
--gex_wgpu_transfers_live;
   --gex_wgpu_transfers_live;
         FREE(transfer);
         }
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr,
                    "gex-wgpu: depth resolve failed for %s\n",
                    util_format_short_name(resource->format));
         }
         return NULL;
      }
      transfer->base.resource = NULL;
      pipe_resource_reference(&transfer->base.resource, resource);
      transfer->base.level = level;
      transfer->base.usage = usage;
      transfer->base.box = *box;
      transfer->base.stride = row;
      transfer->base.layer_stride = (uint64_t)row * box->height;
      *out_transfer = &transfer->base;
      return transfer->data;
   }

   /* The texture holds the backing format's bytes, so that is what the copy
    * moves; the transfer the caller sees is still laid out in the format GL
    * asked for, and the two are reconciled below. */
   const enum pipe_format read_backing =
      gex_wgpu_format_backing(resource->format);
   const unsigned gpu_pixel_bytes = read_backing != PIPE_FORMAT_NONE
      ? util_format_get_blocksize(read_backing) : pixel_bytes;
   /* Two strides, because the texture and the transfer can be different
    * formats: the copy moves the backing format's bytes, and the caller is
    * handed the format GL asked for. They coincide for everything but the
    * 16-bit normalized set. */
   /* A copy that touches a depth or stencil aspect must cover the whole
    * subresource -- WebGPU says so, and Dawn enforces it:
    *
    *   Copy origin (x:4, y:4) and size (1x1) does not cover the entire
    *   subresource (origin: [0,0], size: 16x16)
    *
    * st/mesa reads depth a pixel at a time, so every one of those copies was
    * invalid, the command buffer was invalid with it, and the readback buffer
    * kept the zeros it was created with. That is why a depth resolve could
    * demonstrably write a constant into a texture and the very next read of
    * the same texture returned 0.000000.
    *
    * So a depth read copies the entire level and the requested box is taken
    * out of it below. It costs a full-level staging buffer for what may be one
    * pixel, which is what the rule leaves available. */
   const bool whole_subresource = depth;
   const uint32_t level_width = u_minify(resource->width0, level);
   const uint32_t level_height = u_minify(resource->height0, level);
   const uint32_t copy_x = whole_subresource ? 0u : (uint32_t)box->x;
   const uint32_t copy_y = whole_subresource ? 0u : (uint32_t)box->y;
   const uint32_t copy_width =
      whole_subresource ? level_width : (uint32_t)box->width;
   const uint32_t copy_height =
      whole_subresource ? level_height : (uint32_t)box->height;
   const uint32_t tight_row = read_compressed
      ? read_row_blocks * read_block_bytes
      : copy_width * gpu_pixel_bytes;
   const uint32_t transfer_row = read_compressed
      ? read_row_blocks * read_block_bytes
      : (uint32_t)box->width * pixel_bytes;
   const uint32_t copy_rows = read_compressed ? read_col_blocks : copy_height;
   const uint32_t bytes_per_row = (tight_row + 255u) & ~255u;
   const uint64_t readback_size = (uint64_t)bytes_per_row * copy_rows;
   WGPUBufferDescriptor buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   buffer_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
   buffer_desc.size = readback_size;
   WGPUBuffer readback = wgpuDeviceCreateBuffer(screen->device, &buffer_desc);
   WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   if (!readback || !encoder) {
      if (readback) wgpuBufferRelease(readback);
      if (encoder) wgpuCommandEncoderRelease(encoder);
      return NULL;
   }

   WGPUTexelCopyTextureInfo source = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
   source.texture = gex_resource(resource)->texture;
   source.mipLevel = level;
   source.origin.x = copy_x;
   source.origin.y = copy_y;
   source.origin.z = box->z;
   /* `all`, not `depth-only`, for a format that has only depth.
    *
    * WebGPU's aspect selector picks one aspect of a combined depth-stencil
    * texture; for `depth16unorm` and `depth32float` there is nothing to
    * select, and naming `depth-only` there produced a copy that moved no
    * bytes -- `gl44-msaa-depth-blit-probe` read zeros out of a texture a
    * resolve had demonstrably just written a constant into, from the same
    * pipe_resource and the same WGPUTexture. */
   source.aspect =
      depth && util_format_has_stencil(util_format_description(resource->format))
         ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
   WGPUTexelCopyBufferInfo destination = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
   destination.buffer = readback;
   destination.layout.bytesPerRow = bytes_per_row;
   destination.layout.rowsPerImage = copy_rows;
   /* A partial block at the right or bottom edge is still a whole block in
    * the texture, and WebGPU takes the copy extent in texels -- rounded up to
    * the block grid, which is exactly the allocation resource_create made. */
   WGPUExtent3D extent = {
      read_compressed ? read_row_blocks * read_block_w : copy_width,
      read_compressed ? read_col_blocks * read_block_h : copy_height,
      1,
   };
   wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &extent);
   WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
   wgpuCommandEncoderRelease(encoder);
   if (!command) {
      wgpuBufferRelease(readback);
      gex_fail(ctx);
      return NULL;
   }
   gex_wgpu_queue_submit(screen, 1, &command);
   wgpuCommandBufferRelease(command);

   struct gex_wgpu_map_state state = {0};
   WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
   callback.mode = WGPUCallbackMode_AllowSpontaneous;
   callback.callback = gex_wgpu_map_callback;
   callback.userdata1 = &state;
   wgpuBufferMapAsync(readback, WGPUMapMode_Read, 0, readback_size, callback);
   /* A loop, like the other two waits: a waiter can be woken by someone
    * else's completion now that the bridge wakes all of them, so the flag is
    * the condition and the wake is only a hint. */
   while (!state.done)
      gex_wgpu_jspi_wait();

   const uint8_t *mapped = state.status == WGPUMapAsyncStatus_Success
      ? wgpuBufferGetConstMappedRange(readback, 0, readback_size) : NULL;
   struct gex_wgpu_transfer *transfer = NULL;
   if (mapped && util_format_is_depth_or_stencil(resource->format)) {
      /* Which depth resource was actually read, and its first bytes. A resolve
       * that reports success and a readback that returns zeros are only
       * reconcilable if the two are looking at different textures. */
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: depth read %s res=%p tex=%p %dx%d bytes "
                 "%02x%02x %02x%02x %02x%02x %02x%02x\n",
                 util_format_short_name(resource->format), (void *)resource,
                 (void *)gex_resource(resource)->texture,
                 box->width, box->height,
                 mapped[0], mapped[1], mapped[2], mapped[3],
                 mapped[4], mapped[5], mapped[6], mapped[7]);
      }
   }
   if (mapped) {
      transfer = CALLOC_STRUCT(gex_wgpu_transfer);
      if (transfer) ++gex_wgpu_transfers_live;
      if (transfer) {
         transfer->base.resource = NULL;
         pipe_resource_reference(&transfer->base.resource, resource);
         transfer->base.level = level;
         transfer->base.usage = usage;
         transfer->base.box = *box;
         transfer->base.stride = transfer_row;
         transfer->base.layer_stride = (uint64_t)transfer_row * copy_rows;
         transfer->data = MALLOC(transfer->base.layer_stride);
         if (transfer->data && read_backing != PIPE_FORMAT_NONE)
            util_format_translate(resource->format, transfer->data,
                                  transfer->base.stride, 0, 0,
                                  read_backing, mapped, bytes_per_row, 0, 0,
                                  (unsigned)box->width,
                                  (unsigned)box->height);
         else if (transfer->data) {
            /* The staging buffer holds `copy_width x copy_height` starting at
             * `copy_x, copy_y`, which for a depth read is the whole level
             * rather than the requested box -- so take the box out of it. For
             * every other format the two coincide and the offsets are zero. */
            const uint64_t skip_x =
               (uint64_t)((uint32_t)box->x - copy_x) * gpu_pixel_bytes;
            const uint32_t skip_y = (uint32_t)box->y - copy_y;
            const uint32_t rows = read_compressed
               ? copy_rows : (uint32_t)box->height;
            for (unsigned row = 0; row < rows; ++row)
               memcpy((uint8_t *)transfer->data + row * transfer->base.stride,
                      mapped + (uint64_t)(row + skip_y) * bytes_per_row + skip_x,
                      transfer->base.stride);
         }
      }
   }
   wgpuBufferUnmap(readback);
   wgpuBufferRelease(readback);

   if (!transfer || !transfer->data) {
      if (transfer) {
         pipe_resource_reference(&transfer->base.resource, NULL);
--gex_wgpu_transfers_live;
         FREE(transfer);
      }
      /* This latches the context, so it is the last thing that happens to this
       * application -- say which of the three ways it got here. A failed
       * mapAsync in particular is a device-level symptom rather than anything
       * wrong with this call, and telling it apart from a plain allocation
       * failure is the whole first step. */
      fprintf(stderr,
              "gex-wgpu: texture readback failed (%s) status=%d size=%u "
              "level=%u usage=0x%x box=%d,%d,%d %dx%dx%d target=%d format=%d "
              "detail=\"%s\"\n",
              !mapped ? "map" : !transfer ? "transfer alloc" : "data alloc",
              (int)state.status, (unsigned)readback_size, level, usage,
              box->x, box->y, box->z, box->width, box->height, box->depth,
              (int)resource->target, (int)resource->format, state.message);
      gex_fail(ctx);
      return NULL;
   }
   *out_transfer = &transfer->base;
   return transfer->data;
}

/* Buffer mapping.
 *
 * Every buffer this driver creates carries a CPU shadow, which buffer_subdata
 * already treats as authoritative: nothing on the GPU writes a buffer yet (no
 * transform feedback, no shader storage, no compute), so the shadow and the
 * WGPUBuffer cannot disagree.  Mapping is therefore a pointer into the shadow
 * plus, on unmap, the same aligned upload buffer_subdata performs.
 *
 * When the GPU does gain the ability to write a buffer, a read map will have
 * to fall back to a readback the way texture_map does; until then that case
 * cannot arise and is asserted rather than silently mis-served.
 */
static void *
gex_wgpu_buffer_map(struct pipe_context *pipe, struct pipe_resource *resource,
                    unsigned level, unsigned usage,
                    const struct pipe_box *box,
                    struct pipe_transfer **out_transfer)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_resource *gex = gex_resource(resource);
   *out_transfer = NULL;
   if (ctx->failed || resource->target != PIPE_BUFFER || level != 0 ||
       !gex->buffer || box->x < 0 || box->width <= 0 ||
       (uint64_t)(unsigned)box->x + (unsigned)box->width > resource->width0) {
      /* Eight conditions, one latch. Say which, once: after the first the
       * context is failed and every later map reports the same thing. */
      if (!ctx->failed)
         fprintf(stderr,
                 "gex-wgpu: buffer map refused (%s) target=%d level=%u "
                 "usage=0x%x buffer=%p shadow=%p box=%d+%d width0=%u\n",
                 resource->target != PIPE_BUFFER ? "not a buffer"
                    : level != 0 ? "level != 0"
                    : !gex->buffer ? "no WGPUBuffer"
                    : box->x < 0 ? "negative offset"
                    : box->width <= 0 ? "width <= 0"
                                      : "range past the end",
                 (int)resource->target, level, usage, (void *)gex->buffer,
                 (void *)gex->shadow, box->x, box->width, resource->width0);
      gex_fail(ctx);
      return NULL;
   }

   struct gex_wgpu_transfer *transfer = CALLOC_STRUCT(gex_wgpu_transfer);
   if (transfer) ++gex_wgpu_transfers_live;
   if (!transfer) {
      gex_fail(ctx);
      return NULL;
   }
   transfer->base.resource = NULL;
   pipe_resource_reference(&transfer->base.resource, resource);
   transfer->base.level = level;
   transfer->base.usage = usage;
   transfer->base.box = *box;
   /* Gallium reports a buffer as one row of one layer. */
   transfer->base.stride = 0;
   transfer->base.layer_stride = 0;

   /* A buffer the GPU can write is no longer mirrored by the shadow: compute
    * shaders and transform feedback write the WebGPU buffer directly, so a
    * read has to come back from it. Everything else is still CPU-authored,
    * and the shadow is both correct and free. */
   const bool gpu_written = gex->gpu_written ||
      (resource->bind & (PIPE_BIND_STREAM_OUTPUT | PIPE_BIND_SHADER_BUFFER));
   /* Only a map that actually reads the GPU has to wait for the recorded
    * draws. A write map is served from the shadow and reaches the device at
    * unmap, so flushing here would break the batch for every streamed vertex
    * upload -- 802077 of them in one Minecraft run, once every nine draws. */
   if ((usage & PIPE_MAP_READ) && gpu_written &&
       gex_wgpu_batch_conflicts(ctx, resource))
      gex_wgpu_submit_pending(ctx);
   if ((usage & PIPE_MAP_READ) && gpu_written) {
      struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
      const uint32_t offset = (uint32_t)box->x & ~3u;
      const uint32_t end = ALIGN_POT((uint32_t)box->x + (uint32_t)box->width, 4);
      const uint32_t size = end - offset;

      WGPUBufferDescriptor staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
      staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
      staging_desc.size = size;
      WGPUBuffer staging = wgpuDeviceCreateBuffer(screen->device, &staging_desc);
      WGPUCommandEncoder encoder =
         wgpuDeviceCreateCommandEncoder(screen->device, NULL);
      if (staging && encoder) {
         wgpuCommandEncoderCopyBufferToBuffer(encoder, gex->buffer, offset,
                                              staging, 0, size);
         WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
         if (commands) {
            gex_wgpu_queue_submit(screen, 1, &commands);
            wgpuCommandBufferRelease(commands);
         }
      }
      if (encoder)
         wgpuCommandEncoderRelease(encoder);

      struct gex_wgpu_map_state state = {WGPUMapAsyncStatus_Success, false};
      const void *mapped = NULL;
      if (staging) {
         WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
         callback.mode = WGPUCallbackMode_AllowSpontaneous;
         callback.callback = gex_wgpu_map_callback;
         callback.userdata1 = &state;
         wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size, callback);
         while (!state.done)
            gex_wgpu_jspi_wait();
         if (state.status == WGPUMapAsyncStatus_Success)
            mapped = wgpuBufferGetConstMappedRange(staging, 0, size);
      }
      if (mapped) {
         transfer->data = MALLOC(box->width);
         if (transfer->data) {
            memcpy(transfer->data,
                   (const uint8_t *)mapped + ((uint32_t)box->x - offset),
                   box->width);
            /* Keep the shadow consistent so a later write map does not
             * resurrect stale bytes -- when there is one. */
            if (gex->shadow)
               memcpy(gex->shadow + box->x, transfer->data, box->width);
         }
      }
      if (staging) {
         wgpuBufferUnmap(staging);
         wgpuBufferRelease(staging);
      }
      if (!transfer->data) {
         pipe_resource_reference(&transfer->base.resource, NULL);
--gex_wgpu_transfers_live;
         FREE(transfer);
         gex_fail(ctx);
         return NULL;
      }
      *out_transfer = &transfer->base;
      return transfer->data;
   }

   /* A write-only map of a plain vertex buffer is staged for the duration of
    * the map rather than being handed a permanent shadow.
    *
    * `buffer_unmap` uploads the 4-byte-aligned range around what was mapped,
    * so an aligned map needs no bytes the caller did not write and no prior
    * contents at all. Index and constant buffers keep the shadow: the draw
    * path reads an index buffer's shadow directly to bound its vertex fetch,
    * and both are small. An unaligned map keeps it too -- there were ten of
    * those in 1.3 million. */
   const bool staged =
      !gex->shadow && !(usage & PIPE_MAP_READ) &&
      !(resource->bind & PIPE_BIND_INDEX_BUFFER) &&
      ((uint32_t)box->x & 3u) == 0 && ((uint32_t)box->width & 3u) == 0;
   if (staged) {
      transfer->data = CALLOC(1, (size_t)box->width);
      if (!transfer->data) {
         pipe_resource_reference(&transfer->base.resource, NULL);
         --gex_wgpu_transfers_live;
         FREE(transfer);
         gex_fail(ctx);
         return NULL;
      }
      transfer->staged_write = true;
      ++gex_wgpu_maps_staged;
      ++gex_wgpu_buffer_maps;
      ++gex_wgpu_map_usage_total[usage % GEX_WGPU_MAP_USAGE_BUCKETS];
      ++gex_wgpu_map_aligned;
      *out_transfer = &transfer->base;
      return transfer->data;
   }
   ++gex_wgpu_maps_shadowed;
   if (!gex->shadow && !gex_wgpu_shadow_for_read(pipe, gex)) {
      pipe_resource_reference(&transfer->base.resource, NULL);
      --gex_wgpu_transfers_live;
      FREE(transfer);
      gex_fail(ctx);
      return NULL;
   }
   if (!gex->shadow) {
      if (usage & PIPE_MAP_READ) ++gex_wgpu_shadow_cause_read;
      else if (resource->bind &
               (PIPE_BIND_INDEX_BUFFER | PIPE_BIND_CONSTANT_BUFFER))
         ++gex_wgpu_shadow_cause_bind;
      else ++gex_wgpu_shadow_cause_unaligned;
   } else {
      ++gex_wgpu_shadow_cause_existing;
   }
   if (!gex_wgpu_shadow_ensure(gex)) {
      pipe_resource_reference(&transfer->base.resource, NULL);
      --gex_wgpu_transfers_live;
      FREE(transfer);
      gex_fail(ctx);
      return NULL;
   }
   /* The shadow is the mapping; there is no staging copy to free. */
   gex_wgpu_mark_mapped(gex);
   gex_wgpu_shadow_mapped_now += gex->allocation_size;
   ++gex_wgpu_maps_open;
   ++gex_wgpu_buffer_maps;
   ++gex_wgpu_map_usage_live[usage % GEX_WGPU_MAP_USAGE_BUCKETS];
   ++gex_wgpu_map_bind_live[resource->bind % GEX_WGPU_MAP_USAGE_BUCKETS];
   ++gex_wgpu_map_usage_total[usage % GEX_WGPU_MAP_USAGE_BUCKETS];
   if ((box->x & 3) == 0 && (box->width & 3) == 0)
      ++gex_wgpu_map_aligned;
   else
      ++gex_wgpu_map_unaligned;
   transfer->data = NULL;
   if (usage & PIPE_MAP_READ)
      gex_wgpu_mark_shadow_read(gex);
   *out_transfer = &transfer->base;
   return gex->shadow + box->x;
}

/* An explicit flush of part of a write map.
 *
 * st/mesa's uploaders map with PIPE_MAP_FLUSH_EXPLICIT and then name the
 * sub-ranges they actually wrote, so this is where those reach the GPU;
 * buffer_unmap below deliberately does not upload again for such a map.
 */
static void
gex_wgpu_transfer_flush_region(struct pipe_context *pipe,
                               struct pipe_transfer *transfer,
                               const struct pipe_box *box)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   GEX_ENTRY_TIMER();
   if (transfer &&
       gex_wgpu_batch_conflicts_usage((struct gex_wgpu_context *)pipe,
                                      transfer->resource, transfer->usage))
      gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct pipe_resource *resource = transfer->resource;
   struct gex_wgpu_resource *gex = gex_resource(resource);
   struct gex_wgpu_transfer *gex_transfer =
      (struct gex_wgpu_transfer *)transfer;
   if (ctx->failed || resource->target != PIPE_BUFFER || !gex->buffer ||
       box->width <= 0)
      return;

   /* A staged write map has no shadow: the bytes live in the block handed to
    * the caller, and the flushed sub-range is uploaded straight out of it.
    * Widening to a 4-byte boundary stays inside the mapped range, whose start
    * and width are both aligned, so the extra bytes are the caller's own. */
   if (gex_transfer->staged_write) {
      const uint32_t map_start = (uint32_t)transfer->box.x;
      const uint32_t map_end = map_start + (uint32_t)transfer->box.width;
      uint32_t from = (uint32_t)(map_start + box->x) & ~3u;
      uint32_t to = ALIGN_POT((uint32_t)(map_start + box->x) +
                              (uint32_t)box->width, 4);
      if (from < map_start) from = map_start;
      if (to > map_end) to = map_end;
      if (to > from)
         wgpuQueueWriteBuffer(gex_screen(pipe->screen)->queue, gex->buffer,
                              from,
                              (const uint8_t *)gex_transfer->data +
                                 (from - map_start),
                              to - from);
      return;
   }
   if (!gex->shadow)
      return;

   /* The box is relative to the start of the mapped range, not the buffer. */
   const uint32_t start = (uint32_t)(transfer->box.x + box->x);
   const uint32_t aligned_start = start & ~3u;
   const uint32_t aligned_end = ALIGN_POT(start + (uint32_t)box->width, 4);
   if (aligned_end > gex->allocation_size) {
      gex_fail(ctx);
      return;
   }
   wgpuQueueWriteBuffer(gex_screen(pipe->screen)->queue, gex->buffer,
                        aligned_start, gex->shadow + aligned_start,
                        aligned_end - aligned_start);
}

static void
gex_wgpu_buffer_unmap(struct pipe_context *pipe,
                      struct pipe_transfer *transfer)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   GEX_ENTRY_TIMER();
   if (transfer &&
       gex_wgpu_batch_conflicts_usage((struct gex_wgpu_context *)pipe,
                                      transfer->resource, transfer->usage))
      gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_transfer *gex_transfer =
      (struct gex_wgpu_transfer *)transfer;
   struct pipe_resource *resource = transfer->resource;
   struct gex_wgpu_resource *gex = gex_resource(resource);
   ++gex_wgpu_buffer_unmaps;
   --gex_wgpu_map_usage_live[transfer->usage % GEX_WGPU_MAP_USAGE_BUCKETS];
   --gex_wgpu_map_bind_live[resource->bind % GEX_WGPU_MAP_USAGE_BUCKETS];

   /* A write map is only visible to the GPU once the range is uploaded.
    * With PIPE_MAP_FLUSH_EXPLICIT the frontend already named the dirty
    * sub-ranges and transfer_flush_region uploaded them; uploading the whole
    * range again here would be redundant, and for an unsynchronized map it
    * would also write back bytes another map may since have changed.
    */
   if (!ctx->failed && gex_transfer->staged_write &&
       !(transfer->usage & PIPE_MAP_FLUSH_EXPLICIT) && gex->buffer &&
       transfer->box.width > 0) {
      /* The staged block covers exactly the mapped range, and the range is
       * 4-aligned by construction, so it uploads as it stands. */
      wgpuQueueWriteBuffer(gex_screen(pipe->screen)->queue, gex->buffer,
                           (uint32_t)transfer->box.x, gex_transfer->data,
                           (size_t)transfer->box.width);
   } else if (!ctx->failed && (transfer->usage & PIPE_MAP_WRITE) &&
       !(transfer->usage & PIPE_MAP_FLUSH_EXPLICIT) && gex->buffer &&
       gex->shadow && transfer->box.width > 0) {
      const uint32_t start = (uint32_t)transfer->box.x;
      const uint32_t aligned_start = start & ~3u;
      const uint32_t aligned_end =
         ALIGN_POT(start + (uint32_t)transfer->box.width, 4);
      wgpuQueueWriteBuffer(gex_screen(pipe->screen)->queue, gex->buffer,
                           aligned_start, gex->shadow + aligned_start,
                           aligned_end - aligned_start);
   }

   if (!gex_transfer->staged_write && !gex_transfer->data && gex->shadow) {
      gex_wgpu_shadow_mapped_now -= gex->allocation_size;
      --gex_wgpu_maps_open;
   }
   /* A read map of a GPU-written buffer holds a staging copy. */
   FREE(gex_transfer->data);
   pipe_resource_reference(&gex_transfer->base.resource, NULL);
--gex_wgpu_transfers_live;
   FREE(gex_transfer);
}

static void
gex_wgpu_texture_unmap(struct pipe_context *pipe,
                       struct pipe_transfer *transfer)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_transfer *gex = (struct gex_wgpu_transfer *)transfer;
   /* A write map only reaches the texture here: what the caller wrote is
    * ordinary memory until it goes through the upload path. */
   if ((transfer->usage & PIPE_MAP_WRITE) && gex->data)
      gex_wgpu_texture_subdata(pipe, transfer->resource, transfer->level, 0,
                               &transfer->box, gex->data, transfer->stride,
                               transfer->layer_stride);
   pipe_resource_reference(&gex->base.resource, NULL);
   FREE(gex->data);
   --gex_wgpu_transfers_live;
   FREE(gex);
}

static void
gex_wgpu_clear_render_target(struct pipe_context *pipe,
                             struct pipe_surface *dst,
                             const union pipe_color_union *color,
                             unsigned dstx, unsigned dsty,
                             unsigned width, unsigned height,
                             bool render_condition_enabled)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct pipe_resource *resource = dst ? dst->texture : NULL;
   /* Whatever level the surface names, cleared at that level's own size --
    * matching set_framebuffer_state, which now renders into the attached
    * level. Requiring level zero here while allowing it there would refuse
    * exactly the clear that precedes drawing a mip chain. */
   const unsigned level = dst ? dst->level : 0;
   const unsigned level_width = resource ? u_minify(resource->width0, level) : 0;
   const unsigned level_height = resource ? u_minify(resource->height0, level) : 0;
   /* render_condition_enabled asks the clear to respect an active
    * conditional render; this driver implements no set_render_condition, so
    * none can be active, and st/mesa sets the flag regardless. Latching on it
    * ended runs. */
   (void)render_condition_enabled;
   if (ctx->failed || !resource ||
       level > resource->last_level ||
       dst->first_layer != dst->last_layer || dstx != 0 || dsty != 0 ||
       width != level_width || height != level_height) {
      gex_fail(ctx);
      return;
   }

   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   const bool clear_three_d = resource->target == PIPE_TEXTURE_3D;
   WGPUTextureViewDescriptor clear_view = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   clear_view.dimension = clear_three_d ? WGPUTextureViewDimension_3D
                                        : WGPUTextureViewDimension_2D;
   clear_view.baseMipLevel = level;
   clear_view.mipLevelCount = 1;
   if (!clear_three_d) {
      clear_view.baseArrayLayer = dst->first_layer;
      clear_view.arrayLayerCount = 1;
   }
   WGPUTextureView view =
      wgpuTextureCreateView(gex_resource(resource)->texture, &clear_view);
   WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   if (!view || !encoder) {
      if (view) wgpuTextureViewRelease(view);
      if (encoder) wgpuCommandEncoderRelease(encoder);
      gex_fail(ctx);
      return;
   }
   WGPURenderPassColorAttachment attachment = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
   attachment.view = view;
   if (clear_three_d)
      attachment.depthSlice = dst->first_layer;
   attachment.loadOp = WGPULoadOp_Clear;
   attachment.storeOp = WGPUStoreOp_Store;
   /* Which arm of the union carries the value depends on the target.
    *
    * pipe_color_union is a union, and for a pure-integer render target
    * st/mesa fills `i` or `ui` rather than `f`. Reading `f` unconditionally
    * reinterpreted the integer's bit pattern as a float: glClearBufferiv with
    * 1 became the denormal 1.4e-45, which rounds to zero, so every integer
    * clear produced 0,0,0,0. WGPUColor is doubles, and Dawn takes the integer
    * value in them for an integer attachment. */
   if (util_format_is_pure_sint(dst->format)) {
      attachment.clearValue.r = (double)color->i[0];
      attachment.clearValue.g = (double)color->i[1];
      attachment.clearValue.b = (double)color->i[2];
      attachment.clearValue.a = (double)color->i[3];
   } else if (util_format_is_pure_uint(dst->format)) {
      attachment.clearValue.r = (double)color->ui[0];
      attachment.clearValue.g = (double)color->ui[1];
      attachment.clearValue.b = (double)color->ui[2];
      attachment.clearValue.a = (double)color->ui[3];
   } else {
      attachment.clearValue.r = color->f[0];
      attachment.clearValue.g = color->f[1];
      attachment.clearValue.b = color->f[2];
      attachment.clearValue.a = color->f[3];
   }
   WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   pass_desc.colorAttachmentCount = 1;
   pass_desc.colorAttachments = &attachment;
   WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
   if (pass) {
      wgpuRenderPassEncoderEnd(pass);
      wgpuRenderPassEncoderRelease(pass);
   }
   WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
   wgpuCommandEncoderRelease(encoder);
   wgpuTextureViewRelease(view);
   if (!pass || !command) {
      if (command) wgpuCommandBufferRelease(command);
      gex_fail(ctx);
      return;
   }
   gex_wgpu_queue_submit(screen, 1, &command);
   wgpuCommandBufferRelease(command);
}

static void
gex_wgpu_clear_depth_stencil(struct pipe_context *pipe,
                             struct pipe_surface *dst, unsigned clear_flags,
                             double depth, unsigned stencil,
                             unsigned dstx, unsigned dsty,
                             unsigned width, unsigned height,
                             bool render_condition_enabled)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct pipe_resource *resource = dst ? dst->texture : NULL;
   const bool has_stencil = dst &&
      util_format_has_stencil(util_format_description(dst->format));
   /* Every depth format the driver can name, not a list that has to be kept
    * in step with one: this refusal latches, so a depth format added to the
    * table and forgotten here turns a new capability into a dead run. That is
    * how depth16 cost two attempts. */
   (void)render_condition_enabled;
   const bool clearable_depth = dst &&
      util_format_is_depth_or_stencil(dst->format) &&
      gex_wgpu_texture_format(dst->format) != WGPUTextureFormat_Undefined;
   const unsigned allowed_flags = has_stencil
      ? PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL : PIPE_CLEAR_DEPTH;
   if (ctx->failed || !resource || !clear_flags ||
       (clear_flags & ~allowed_flags) ||
       !clearable_depth ||
       dst->level != 0 || dst->first_layer != 0 || dst->last_layer != 0 ||
       dstx != 0 || dsty != 0 || width != resource->width0 ||
       height != resource->height0 ||
       /* `!(depth >= 0.0 && depth <= 1.0)`, not `depth < 0.0 || depth > 1.0`:
        * NaN compares false against both bounds, so the second form lets it
        * through to depthClearValue, where the browser rejects the pass. */
       ((clear_flags & PIPE_CLEAR_DEPTH) && !(depth >= 0.0 && depth <= 1.0))) {
      gex_fail(ctx);
      return;
   }
   {
      /* What the depth buffer is actually cleared to, once per value. A
       * buffer cleared to 0 rejects every fragment under a LESS test while
       * leaving anything that does not depth-test -- a skybox, a HUD --
       * perfectly visible. */
      static double told[4];
      static unsigned told_count;
      bool fresh = true;
      for (unsigned i = 0; i < told_count; ++i)
         if (told[i] == depth) { fresh = false; break; }
      if (fresh && told_count < 4) {
         told[told_count++] = depth;
         fprintf(stderr, "gex-wgpu: depth cleared to %.4f (flags=0x%x)\n",
                 depth, clear_flags);
      }
   }
   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   WGPUTextureView view =
      wgpuTextureCreateView(gex_resource(resource)->texture, NULL);
   WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   WGPURenderPassDepthStencilAttachment attachment =
      WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
   attachment.view = view;
   attachment.depthLoadOp = clear_flags & PIPE_CLEAR_DEPTH
      ? WGPULoadOp_Clear : WGPULoadOp_Load;
   attachment.depthStoreOp = WGPUStoreOp_Store;
   attachment.depthClearValue = depth;
   if (has_stencil) {
      attachment.stencilLoadOp = clear_flags & PIPE_CLEAR_STENCIL
         ? WGPULoadOp_Clear : WGPULoadOp_Load;
      attachment.stencilStoreOp = WGPUStoreOp_Store;
      attachment.stencilClearValue = stencil & 0xffu;
   }
   WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   desc.depthStencilAttachment = &attachment;
   WGPURenderPassEncoder pass = view && encoder
      ? wgpuCommandEncoderBeginRenderPass(encoder, &desc) : NULL;
   if (pass) {
      wgpuRenderPassEncoderEnd(pass);
      wgpuRenderPassEncoderRelease(pass);
   }
   WGPUCommandBuffer command = pass
      ? wgpuCommandEncoderFinish(encoder, NULL) : NULL;
   if (encoder) wgpuCommandEncoderRelease(encoder);
   if (view) wgpuTextureViewRelease(view);
   if (!pass || !command) {
      if (command) wgpuCommandBufferRelease(command);
      gex_fail(ctx);
      return;
   }
   gex_wgpu_queue_submit(screen, 1, &command);
   wgpuCommandBufferRelease(command);
}

/* The whole-framebuffer clear.
 *
 * st/mesa lowers glClear to this, not to clear_render_target: the latter is
 * for glClearBuffer-style single-target clears.  Both are expressed here as
 * the existing per-target clears over whatever the framebuffer currently
 * holds, so there is one implementation of the actual WebGPU work.
 *
 * A scissored clear is deliberately refused rather than approximated: the
 * per-target clears below only do whole surfaces, and quietly clearing more
 * than GL asked for would be a wrong picture instead of a missing feature.
 */
/* cso_context calls this unconditionally when it unbinds a context, unlike
 * set_min_samples next to it, which it null-checks.
 *
 * Every framebuffer this driver creates is single-sample, so sample zero is
 * the only sample there is: a mask with bit zero set covers it and any other
 * bits are about samples that do not exist.  A mask that clears bit zero means
 * "discard every fragment", which cannot be expressed here and is refused
 * rather than ignored.
 */
/* State st/mesa sets on every draw or at context init, which this backend
 * either cannot express or does not need.
 *
 * Each one accepts exactly the value that means "off" and refuses anything
 * else.  Accepting silently would render a picture GL did not ask for, which
 * is worse than an unsupported-feature failure: it looks like it worked.
 */
static void
gex_wgpu_set_polygon_stipple(struct pipe_context *pipe,
                             const struct pipe_poly_stipple *stipple)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (ctx->failed || !stipple)
      return;
   /* st/mesa calls this before every draw, with an all-ones pattern whenever
    * GL_POLYGON_STIPPLE is disabled -- which is the only case WebGPU can
    * render, as it has no stipple. */
   for (unsigned i = 0; i < ARRAY_SIZE(stipple->stipple); ++i) {
      if (stipple->stipple[i] != 0xffffffffu) {
         gex_fail(ctx);
         return;
      }
   }
}

static void
gex_wgpu_set_min_samples(struct pipe_context *pipe, unsigned min_samples)
{
   /* Nothing to do. WebGPU decides the shading rate from the shader, not from
    * the pipeline: a fragment shader that reads @builtin(sample_index) is
    * invoked per sample, and one that does not is invoked per fragment. So
    * the rate GL asks for here is either already in force or is being asked
    * for at a fraction, and GL explicitly permits rounding a fraction up.
    *
    * This used to latch the context for any rate above one, on the grounds
    * that framebuffers were single-sample. They have not been for a while --
    * 4x is supported and resolved -- so the refusal was outliving its
    * reason. */
   (void)pipe;
   (void)min_samples;
}

static void
gex_wgpu_set_clip_state(struct pipe_context *pipe,
                        const struct pipe_clip_state *clip)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   /* User clip planes are lowered into the shader by st/mesa when the
    * rasterizer asks for them; this driver enables none, so the state it is
    * handed is unused rather than ignored. */
   (void)clip;
   (void)ctx;
}

/* Occlusion queries.
 *
 * GL asks how many samples passed the depth and stencil tests; WebGPU's
 * occlusion query set answers exactly that, counted across a render pass.  The
 * pass has to be told which query set it writes to, so the draw path picks
 * this up from ctx->active_query.
 *
 * Only PIPE_QUERY_OCCLUSION_COUNTER and _PREDICATE are supported.  Timestamps
 * would need the timestamp-query feature and a different resolve; anything
 * else is refused rather than silently answering zero, which would read as
 * "nothing drawn".
 */
static struct pipe_query *
gex_wgpu_create_query(struct pipe_context *pipe, unsigned query_type,
                      unsigned index)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   const bool occlusion = query_type == PIPE_QUERY_OCCLUSION_COUNTER ||
                          query_type == PIPE_QUERY_OCCLUSION_PREDICATE ||
                          query_type == PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE;
   /* Counted on the CPU from the draws, not on the GPU: what these ask for is
    * how many primitives the vertex stage produced and how many transform
    * feedback wrote, and this driver decides both when it issues the draw.
    *
    * Returning NULL for them was much worse than an approximate count.
    * st/mesa turns a NULL query into GL_OUT_OF_MEMORY from glBeginQuery, the
    * CTS wraps that call in GLU_EXPECT_NO_ERROR, and dEQP treats
    * GL_OUT_OF_MEMORY as a fatal ResourceError -- so one unsupported query
    * type ended the entire run.
    *
    * The count is the vertex stage's. A geometry shader may emit a different
    * number and this does not model that; the emulated geometry pass would
    * have to report it. */
   const bool counted = query_type == PIPE_QUERY_PRIMITIVES_GENERATED ||
                        query_type == PIPE_QUERY_PRIMITIVES_EMITTED ||
                        query_type == PIPE_QUERY_SO_STATISTICS;
   /* Timer queries, measured on the CPU.
    *
    * This is what softpipe does -- caps.query_timestamp with
    * u_default_get_timestamp and os_time_get_nano() -- and it is the reason
    * ARB_timer_query can be advertised at all, which GL 3.3 requires.
    *
    * It measures when commands were *recorded*, not when the GPU ran them, so
    * it under-reports GPU work: everything here is queued on the WebGPU queue
    * and finishes later. WebGPU does expose a real timestamp-query feature
    * (screen->timestamp_query is already probed for it), and a GPU-accurate
    * implementation would attach timestampWrites to each pass and resolve the
    * query set the way the occlusion path below does. That is the right fix
    * if anything ever profiles against these numbers; until then this is an
    * honest ceiling-raiser rather than a measurement to trust. */
   const bool timer = query_type == PIPE_QUERY_TIME_ELAPSED ||
                      query_type == PIPE_QUERY_TIMESTAMP ||
                      query_type == PIPE_QUERY_TIMESTAMP_DISJOINT;
   if (index != 0 || (!occlusion && !counted && !timer))
      return NULL;

   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   if (counted || timer) {
      struct gex_wgpu_query *counter = CALLOC_STRUCT(gex_wgpu_query);
      if (counter)
         counter->type = query_type;
      return (struct pipe_query *)counter;
   }
   if (!ctx->occlusion_set) {
      WGPUQuerySetDescriptor set_desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
      set_desc.type = WGPUQueryType_Occlusion;
      set_desc.count = 1;
      ctx->occlusion_set = wgpuDeviceCreateQuerySet(screen->device, &set_desc);

      /* Resolve target, then a mappable copy: a QueryResolve buffer cannot
       * also be MapRead. */
      WGPUBufferDescriptor resolve_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
      resolve_desc.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
      resolve_desc.size = 8;
      ctx->occlusion_resolve = wgpuDeviceCreateBuffer(screen->device, &resolve_desc);

      WGPUBufferDescriptor read_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
      read_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
      read_desc.size = 8;
      ctx->occlusion_readback = wgpuDeviceCreateBuffer(screen->device, &read_desc);

      if (!ctx->occlusion_set || !ctx->occlusion_resolve ||
          !ctx->occlusion_readback)
         return NULL;
   }

   struct gex_wgpu_query *query = CALLOC_STRUCT(gex_wgpu_query);
   if (query)
      query->type = query_type;
   return (struct pipe_query *)query;
}

static void
gex_wgpu_destroy_query(struct pipe_context *pipe, struct pipe_query *q)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (ctx->active_query == (struct gex_wgpu_query *)q)
      ctx->active_query = NULL;
   FREE(q);
}

static bool
gex_wgpu_begin_query(struct pipe_context *pipe, struct pipe_query *q)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_query *query = (struct gex_wgpu_query *)q;
   if (!query)
      return false;
   /* A timer query does not occupy the occlusion slot: GL lets a TIME_ELAPSED
    * query run concurrently with an occlusion query, and claiming
    * ctx->active_query here would make each refuse the other. */
   if (query->type == PIPE_QUERY_TIME_ELAPSED ||
       query->type == PIPE_QUERY_TIMESTAMP ||
       query->type == PIPE_QUERY_TIMESTAMP_DISJOINT) {
      query->active = true;
      query->have_result = false;
      query->result = 0;
      query->start = os_time_get_nano();
      return true;
   }
   if (ctx->active_query)
      return false;
   query->active = true;
   query->have_result = false;
   query->result = 0;
   ctx->active_query = query;
   return true;
}

static bool
gex_wgpu_end_query(struct pipe_context *pipe, struct pipe_query *q)
{
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_query *query = (struct gex_wgpu_query *)q;
   if (!query)
      return false;
   /* glQueryCounter(GL_TIMESTAMP) reaches end_query without a begin_query, so
    * this must not assume `start` was ever set. */
   if (query->type == PIPE_QUERY_TIME_ELAPSED) {
      const uint64_t now = os_time_get_nano();
      query->result = query->active && now > query->start
                         ? now - query->start : 0;
      query->have_result = true;
   } else if (query->type == PIPE_QUERY_TIMESTAMP ||
              query->type == PIPE_QUERY_TIMESTAMP_DISJOINT) {
      query->result = os_time_get_nano();
      query->have_result = true;
   }
   query->active = false;
   if (ctx->active_query == query)
      ctx->active_query = NULL;
   return true;
}

static bool
gex_wgpu_get_query_result(struct pipe_context *pipe, struct pipe_query *q,
                          bool wait, union pipe_query_result *result)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   struct gex_wgpu_query *query = (struct gex_wgpu_query *)q;
   if (!query || query->active)
      return false;

   if (query->type == PIPE_QUERY_TIME_ELAPSED ||
       query->type == PIPE_QUERY_TIMESTAMP) {
      if (result)
         result->u64 = query->result;
      return true;
   }
   if (query->type == PIPE_QUERY_TIMESTAMP_DISJOINT) {
      if (result) {
         /* os_time_get_nano() is a nanosecond clock and never reports a gap. */
         result->timestamp_disjoint.frequency = UINT64_C(1000000000);
         result->timestamp_disjoint.disjoint = false;
      }
      return true;
   }

   if (query->type == PIPE_QUERY_PRIMITIVES_GENERATED ||
       query->type == PIPE_QUERY_PRIMITIVES_EMITTED ||
       query->type == PIPE_QUERY_SO_STATISTICS) {
      if (result) {
         if (query->type == PIPE_QUERY_SO_STATISTICS) {
            result->so_statistics.num_primitives_written = query->result;
            result->so_statistics.primitives_storage_needed = query->result;
         } else {
            result->u64 = query->result;
         }
      }
      return true;
   }

   if (!query->have_result) {
      if (!ctx->occlusion_set)
         return false;
      /* Everything this driver submits is already ordered on the queue, so
       * the counted passes are final by the time this runs. */
      WGPUCommandEncoder encoder =
         wgpuDeviceCreateCommandEncoder(screen->device, NULL);
      if (!encoder)
         return false;
      wgpuCommandEncoderResolveQuerySet(encoder, ctx->occlusion_set, 0, 1,
                                        ctx->occlusion_resolve, 0);
      wgpuCommandEncoderCopyBufferToBuffer(encoder, ctx->occlusion_resolve, 0,
                                           ctx->occlusion_readback, 0, 8);
      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      wgpuCommandEncoderRelease(encoder);
      if (!commands)
         return false;
      gex_wgpu_queue_submit(screen, 1, &commands);
      wgpuCommandBufferRelease(commands);

      struct gex_wgpu_map_state state = {WGPUMapAsyncStatus_Success, false};
      WGPUBufferMapCallbackInfo callback = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
      callback.mode = WGPUCallbackMode_AllowSpontaneous;
      callback.callback = gex_wgpu_map_callback;
      callback.userdata1 = &state;
      wgpuBufferMapAsync(ctx->occlusion_readback, WGPUMapMode_Read, 0, 8,
                         callback);
      while (!state.done)
         gex_wgpu_jspi_wait();
      const void *mapped = state.status == WGPUMapAsyncStatus_Success
         ? wgpuBufferGetConstMappedRange(ctx->occlusion_readback, 0, 8) : NULL;
      if (mapped)
         memcpy(&query->result, mapped, sizeof(query->result));
      wgpuBufferUnmap(ctx->occlusion_readback);
      if (!mapped)
         return false;
      query->have_result = true;
   }

   if (query->type == PIPE_QUERY_OCCLUSION_COUNTER)
      result->u64 = query->result;
   else
      result->b = query->result != 0;
   (void)wait;
   return true;
}

static void
gex_wgpu_set_active_query_state(struct pipe_context *pipe, bool enable)
{
   /* No queries are implemented, so there is nothing to pause or resume. */
   (void)pipe;
   (void)enable;
}

static void
gex_wgpu_set_debug_callback(struct pipe_context *pipe,
                            const struct util_debug_callback *cb)
{
   /* Driver-originated debug messages are not produced; dropping the callback
    * loses nothing. */
   (void)pipe;
   (void)cb;
}

/* Transform feedback targets.
 *
 * WebGPU has no transform feedback stage.  What it does have is storage
 * buffers writable from the vertex stage, which this device reports
 * maxStorageBuffersInVertexStage for; capture is therefore the vertex shader
 * storing its outputs itself, and these targets are the buffers it stores to.
 * The shader-side lowering is what makes the capability real, so the cap stays
 * off until that exists -- these hooks alone would let st/mesa bind targets
 * that nothing ever writes.
 */
struct gex_wgpu_so_target {
   struct pipe_stream_output_target base;
   /* Bytes already recorded into this target, counted from buffer_offset.
    *
    * GL's pause/resume and its explicit append offset are the same state seen
    * from two directions, and Gallium expresses both through the offsets
    * argument of set_stream_output_targets: a byte count starts the capture
    * there, and ~0 means carry on from wherever the last one stopped. Keeping
    * the running total on the target rather than on the context is what makes
    * pause work at all, because pausing unbinds every target and resuming
    * binds them back with ~0. */
   unsigned written;
   /* Bytes per captured vertex, learnt from the shader's xfb info the first
    * time this target records anything.
    *
    * DrawTransformFeedback needs a vertex count, and on real hardware that
    * count lives on the GPU because only the GPU knows how much fitted. Here
    * the driver decides the capture size itself, on the CPU, before the
    * dispatch runs -- so the count is simply `written / stride` and no
    * counter buffer or readback is involved. */
   unsigned stride;
};

static struct pipe_stream_output_target *
gex_wgpu_create_stream_output_target(struct pipe_context *pipe,
                                     struct pipe_resource *resource,
                                     unsigned buffer_offset,
                                     unsigned buffer_size)
{
   struct gex_wgpu_resource *gex = gex_resource(resource);
   if (!gex->buffer ||
       (uint64_t)buffer_offset + buffer_size > resource->width0)
      return NULL;

   struct gex_wgpu_so_target *target = CALLOC_STRUCT(gex_wgpu_so_target);
   if (!target)
      return NULL;
   pipe_reference_init(&target->base.reference, 1);
   target->base.buffer = NULL;
   pipe_resource_reference(&target->base.buffer, resource);
   target->base.context = pipe;
   target->base.buffer_offset = buffer_offset;
   target->base.buffer_size = buffer_size;
   return &target->base;
}

static void
gex_wgpu_stream_output_target_destroy(struct pipe_context *pipe,
                                      struct pipe_stream_output_target *target)
{
   (void)pipe;
   pipe_resource_reference(&target->buffer, NULL);
   FREE(target);
}

static void
gex_wgpu_set_stream_output_targets(struct pipe_context *pipe,
                                   unsigned num_targets,
                                   struct pipe_stream_output_target **targets,
                                   const unsigned *offsets,
                                   enum mesa_prim output_prim)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   (void)output_prim;
   if (num_targets > GEX_WGPU_MAX_SO_BUFFERS) {
      gex_fail(ctx);
      return;
   }
   for (unsigned i = 0; i < GEX_WGPU_MAX_SO_BUFFERS; ++i) {
      struct pipe_stream_output_target *target =
         i < num_targets && targets ? targets[i] : NULL;
      pipe_so_target_reference(&ctx->so_targets[i], target);
   }
   ctx->so_target_count = num_targets;
   /* ~0 is "append": leave the target's running total alone so the next
    * capture continues after the last one. Any other value is where GL wants
    * writing to start, which for glBeginTransformFeedback is zero. */
   for (unsigned i = 0; offsets && i < num_targets; ++i) {
      struct gex_wgpu_so_target *target =
         (struct gex_wgpu_so_target *)ctx->so_targets[i];
      if (target && offsets[i] != (unsigned)-1)
         target->written = offsets[i];
   }
}

/* Compute shaders.
 *
 * WebGPU compute is the only stage that may write a storage buffer, so this is
 * the foundation for transform feedback -- which needs the vertex shader re-run
 * as compute -- as well as for GL's own compute shaders later.
 *
 * The resource model here is deliberately narrow: storage buffers only, in one
 * bind group. That is what a capture pass needs, and anything else would be
 * advertised before it was written.
 */
struct gex_wgpu_compute_state {
   WGPUShaderModule module;
   unsigned so_mask;
   uint32_t so_bindings[GEX_WGPU_MAX_SO_BUFFERS];
   unsigned block[3];
};

static void *
gex_wgpu_create_compute_state(struct pipe_context *pipe,
                              const struct pipe_compute_state *state)
{
   if (state->ir_type != PIPE_SHADER_IR_NIR)
      return NULL;
   nir_shader *nir = (nir_shader *)state->prog;
   if (!nir || nir->info.stage != MESA_SHADER_COMPUTE) {
      if (nir)
         ralloc_free(nir);
      return NULL;
   }

   struct gex_wgpu_compute_state *compute =
      CALLOC_STRUCT(gex_wgpu_compute_state);
   if (!compute) {
      ralloc_free(nir);
      return NULL;
   }
   for (unsigned i = 0; i < 3; ++i)
      compute->block[i] = MAX2(1, nir->info.workgroup_size[i]);

   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ssbo) {
      const unsigned slot = var->data.driver_location;
      if (slot >= GEX_WGPU_MAX_SO_BUFFERS) {
         ralloc_free(nir);
         FREE(compute);
         return NULL;
      }
      compute->so_bindings[slot] = var->data.binding;
      compute->so_mask |= 1u << slot;
   }

   char *error = NULL;
   compute->module =
      gex_wgpu_compile_nir(pipe, nir, MESA_SHADER_COMPUTE, false, &error);
   if (!compute->module) {
      fprintf(stderr, "gex-wgpu: compute shader rejected: %s\n",
              error ? error : "unknown");
      free(error);
      FREE(compute);
      return NULL;
   }
   free(error);
   return compute;
}

static void
gex_wgpu_set_shader_buffers(struct pipe_context *pipe, mesa_shader_stage shader,
                            unsigned start_slot, unsigned count,
                            const struct pipe_shader_buffer *buffers,
                            unsigned writable_bitmask)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   (void)writable_bitmask;
   if (start_slot + count > GEX_WGPU_MAX_SO_BUFFERS) {
      gex_fail(ctx);
      return;
   }
   /* Only the compute stage can write one; a storage buffer bound to any
    * other stage would be silently unused, so refuse it rather than pretend. */
   if (shader != MESA_SHADER_COMPUTE) {
      for (unsigned i = 0; buffers && i < count; ++i) {
         if (buffers[i].buffer) {
            gex_fail(ctx);
            return;
         }
      }
      return;
   }
   for (unsigned i = 0; i < count; ++i) {
      struct pipe_shader_buffer *slot = &ctx->shader_buffers[start_slot + i];
      pipe_resource_reference(&slot->buffer, NULL);
      if (buffers && buffers[i].buffer) {
         pipe_resource_reference(&slot->buffer, buffers[i].buffer);
         slot->buffer_offset = buffers[i].buffer_offset;
         slot->buffer_size = buffers[i].buffer_size;
      }
   }
}

static void
gex_wgpu_bind_compute_state(struct pipe_context *pipe, void *state)
{
   ((struct gex_wgpu_context *)pipe)->compute = state;
}

static void
gex_wgpu_delete_compute_state(struct pipe_context *pipe, void *state)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_compute_state *compute = state;
   if (ctx->compute == compute)
      ctx->compute = NULL;
   if (compute && compute->module)
      wgpuShaderModuleRelease(compute->module);
   FREE(compute);
}

static void
gex_wgpu_launch_grid(struct pipe_context *pipe,
                     const struct pipe_grid_info *info)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   struct gex_wgpu_compute_state *compute = ctx->compute;
   if (ctx->failed || !compute || info->indirect) {
      gex_fail(ctx);
      return;
   }

   WGPUBindGroupLayoutEntry entries[GEX_WGPU_MAX_SO_BUFFERS];
   WGPUBindGroupEntry bindings[GEX_WGPU_MAX_SO_BUFFERS];
   unsigned count = 0;
   for (unsigned i = 0; i < GEX_WGPU_MAX_SO_BUFFERS; ++i) {
      if (!(compute->so_mask & (1u << i)))
         continue;
      const struct pipe_shader_buffer *slot = &ctx->shader_buffers[i];
      if (!slot->buffer) {
         gex_fail(ctx);
         return;
      }
      entries[count] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      entries[count].binding = compute->so_bindings[i];
      entries[count].visibility = WGPUShaderStage_Compute;
      entries[count].buffer.type = WGPUBufferBindingType_Storage;
      bindings[count] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
      bindings[count].binding = compute->so_bindings[i];
      bindings[count].buffer = gex_resource(slot->buffer)->buffer;
      bindings[count].offset = slot->buffer_offset;
      bindings[count].size = slot->buffer_size;
      ++count;
   }

   WGPUBindGroupLayoutDescriptor layout_desc =
      WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
   layout_desc.entryCount = count;
   layout_desc.entries = entries;
   WGPUBindGroupLayout layout =
      wgpuDeviceCreateBindGroupLayout(screen->device, &layout_desc);
   WGPUPipelineLayoutDescriptor pipeline_layout_desc =
      WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
   pipeline_layout_desc.bindGroupLayoutCount = count ? 1 : 0;
   pipeline_layout_desc.bindGroupLayouts = count ? &layout : NULL;
   WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(screen->device, &pipeline_layout_desc);

   WGPUComputePipelineDescriptor pipeline_desc =
      WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
   pipeline_desc.layout = pipeline_layout;
   pipeline_desc.compute.module = compute->module;
   pipeline_desc.compute.entryPoint = (WGPUStringView){"main", WGPU_STRLEN};
   WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(screen->device, &pipeline_desc);

   WGPUBindGroupDescriptor group_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   group_desc.layout = layout;
   group_desc.entryCount = count;
   group_desc.entries = bindings;
   WGPUBindGroup group =
      count ? wgpuDeviceCreateBindGroup(screen->device, &group_desc) : NULL;

   WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   if (pipeline && encoder && (!count || group)) {
      WGPUComputePassDescriptor pass_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
      WGPUComputePassEncoder pass =
         wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
      if (pass) {
         wgpuComputePassEncoderSetPipeline(pass, pipeline);
         if (group)
            wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
         wgpuComputePassEncoderDispatchWorkgroups(
            pass, info->grid[0], info->grid[1], info->grid[2]);
         wgpuComputePassEncoderEnd(pass);
         wgpuComputePassEncoderRelease(pass);
      }
      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      if (commands) {
         gex_wgpu_queue_submit(screen, 1, &commands);
         wgpuCommandBufferRelease(commands);
      }
   } else {
      gex_fail(ctx);
   }

   if (encoder) wgpuCommandEncoderRelease(encoder);
   if (group) wgpuBindGroupRelease(group);
   if (pipeline) wgpuComputePipelineRelease(pipeline);
   if (pipeline_layout) wgpuPipelineLayoutRelease(pipeline_layout);
   if (layout) wgpuBindGroupLayoutRelease(layout);
}

static void
gex_wgpu_flush_resource(struct pipe_context *pipe,
                        struct pipe_resource *resource)
{
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   /* st/mesa calls this before handing a resource to something outside GL.
    * Every command this driver records is submitted to the WebGPU queue as it
    * is produced, so a resource is already as flushed as it can be. */
   (void)pipe;
   (void)resource;
}

static void
gex_wgpu_memory_barrier(struct pipe_context *pipe, unsigned flags)
{
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   /* Everything this driver submits goes to one WebGPU queue in order, and
    * nothing is mapped while the GPU can still be writing it. */
   (void)pipe;
   (void)flags;
}

static void
gex_wgpu_texture_barrier(struct pipe_context *pipe, unsigned flags)
{
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   (void)pipe;
   (void)flags;
}

static void
gex_wgpu_render_condition(struct pipe_context *pipe, struct pipe_query *query,
                          bool condition, enum pipe_render_cond_flag mode)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   (void)condition;
   (void)mode;
   /* Conditional rendering needs queries, which do not exist here.  Clearing
    * the condition is the only supported call. */
   if (!ctx->failed && query)
      gex_fail(ctx);
}

/* Copy a rectangle of one texture into another.
 *
 * WebGPU has this natively as copyTextureToTexture, with the same restriction
 * GL's resource_copy_region carries: no scaling and no format conversion.  It
 * is also what st/mesa's blit reduces to for the cases below.
 */
static bool
gex_wgpu_copy_texture_region(struct pipe_context *pipe,
                             struct pipe_resource *dst, unsigned dst_level,
                             unsigned dstx, unsigned dsty, unsigned dstz,
                             struct pipe_resource *src, unsigned src_level,
                             const struct pipe_box *box, bool flip_y)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   /* Submits its own command buffer, so anything already recorded has to
    * reach the queue first or it would run after this. */
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   struct gex_wgpu_resource *gex_dst = gex_resource(dst);
   struct gex_wgpu_resource *gex_src = gex_resource(src);

   /* `z` is a slice: a cube face, an array layer, or a depth slice depending on
    * the target, and WebGPU spells all three the same way -- `origin.z` plus
    * `depthOrArrayLayers`. This used to refuse any non-zero z and any depth
    * other than one, which latched the context on something WebGPU expresses
    * directly. SuperTuxKart copies 512x512 into cube face 1 for its skybox,
    * and lost its whole renderer to it.
    *
    * The flipping path below is still single-slice: it is a per-row loop, and
    * a slice count would multiply the command count by it for no caller that
    * asks. */
   if (!gex_dst->texture || !gex_src->texture || dst->format != src->format ||
       box->width <= 0 || (flip_y && box->depth != 1)) {
      fprintf(stderr,
              "gex-wgpu: copy region refused (%s) dst_target=%d src_target=%d "
              "dst_format=%d src_format=%d box=%d,%d,%d %dx%dx%d dst=%u,%u,%u "
              "levels=%u<-%u\n",
              !gex_dst->texture
                 ? (dst->target == PIPE_BUFFER ? "destination is a buffer"
                                               : "destination has no texture")
                 : !gex_src->texture
                    ? (src->target == PIPE_BUFFER ? "source is a buffer"
                                                  : "source has no texture")
                 : dst->format != src->format ? "format mismatch"
                 : box->width <= 0 ? "width <= 0"
                                   : "flipping copy of more than one slice",
              (int)dst->target, (int)src->target, (int)dst->format,
              (int)src->format, box->x, box->y, box->z, box->width,
              box->height, box->depth, dstx, dsty, dstz, dst_level, src_level);
      return false;
   }

   WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   if (!encoder)
      return false;

   WGPUTexelCopyTextureInfo from = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
   from.texture = gex_src->texture;
   from.mipLevel = src_level;
   WGPUTexelCopyTextureInfo to = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
   to.texture = gex_dst->texture;
   to.mipLevel = dst_level;

   const uint32_t height = (uint32_t)abs(box->height);
   /* Before the branch: the flipping path below writes the destination just
    * as thoroughly, one row at a time, and marking only the straight copy
    * meant every y-flipped copy left its destination looking untouched.
    * glCopyTexSubImage2D from a window is always flipped -- GL's window
    * origin is the bottom -- so Minecraft's whole main menu was reported as
    * "samples a texture nothing ever wrote" while the copy was in fact
    * happening. */
   gex_resource(dst)->texture_written = true;
   if (!flip_y) {
      from.origin =
         (WGPUOrigin3D){(uint32_t)box->x, (uint32_t)box->y, (uint32_t)box->z};
      to.origin = (WGPUOrigin3D){dstx, dsty, dstz};
      WGPUExtent3D extent = {(uint32_t)box->width, height,
                             (uint32_t)box->depth};
      wgpuCommandEncoderCopyTextureToTexture(encoder, &from, &to, &extent);
   } else {
      /* WebGPU has no flipping copy, and GL asks for one whenever the window
       * origin disagrees with the texture's -- st/mesa's ReadPixels staging
       * blit is the common case.  One copy per row is O(height) commands
       * inside a single encoder: correct, and cheap enough for a readback
       * path, but a textured-quad blitter should replace it once one exists.
       *
       * A negative height means the box runs upward from box->y, so the rows
       * are box->y-1 down to box->y+height. */
      WGPUExtent3D row = {(uint32_t)box->width, 1, 1};
      for (uint32_t i = 0; i < height; ++i) {
         from.origin = (WGPUOrigin3D){(uint32_t)box->x,
                                      (uint32_t)box->y - 1 - i,
                                      (uint32_t)box->z};
         to.origin = (WGPUOrigin3D){dstx, dsty + i, dstz};
         wgpuCommandEncoderCopyTextureToTexture(encoder, &from, &to, &row);
      }
   }
   WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
   wgpuCommandEncoderRelease(encoder);
   if (!commands)
      return false;
   gex_wgpu_queue_submit(screen, 1, &commands);
   wgpuCommandBufferRelease(commands);
   (void)ctx;
   return true;
}

/* Snapshot a colour attachment so a draw that also samples it can read the
 * copy.
 *
 * WebGPU forbids one texture being a render attachment and a sampled binding
 * in the same pass outright -- "writable usage and another usage in the same
 * synchronization scope" -- and rejects the entire command buffer, taking
 * every draw batched with it. GL is narrower: a feedback loop is undefined
 * only where the texels read and written actually overlap, so an application
 * sampling a different region, or relying on the contents from before this
 * pass, is entitled to a correct answer.
 *
 * A snapshot gives exactly those semantics. It is the pre-pass contents, which
 * is what a non-overlapping read wants and what the overlapping case is not
 * allowed to depend on.
 *
 * This used to decline the draw instead, on the reasoning that such a binding
 * is "usually just stale". In SuperTuxKart it is not stale: seven draws a
 * frame sample an attachment deliberately, and declining them lost every draw
 * in the frame.
 *
 * Returns the view to bind, or NULL to leave the slot alone.
 */
static WGPUTextureView
gex_wgpu_feedback_snapshot(struct gex_wgpu_context *ctx,
                           struct pipe_resource *attachment)
{
   struct pipe_context *pipe = &ctx->base;
   struct pipe_screen *screen = pipe->screen;
   struct gex_wgpu_resource *source = gex_resource(attachment);

   /* Reuse the copy unless the shape moved under it. */
   if (source->feedback_copy &&
       (source->feedback_copy->width0 != attachment->width0 ||
        source->feedback_copy->height0 != attachment->height0 ||
        source->feedback_copy->format != attachment->format)) {
      if (source->feedback_copy_view)
         wgpuTextureViewRelease(source->feedback_copy_view);
      source->feedback_copy_view = NULL;
      pipe_resource_reference(&source->feedback_copy, NULL);
   }

   if (!source->feedback_copy) {
      struct pipe_resource templ = *attachment;
      templ.bind = PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_RENDER_TARGET;
      templ.target = PIPE_TEXTURE_2D;
      templ.depth0 = 1;
      templ.array_size = 1;
      templ.last_level = 0;
      templ.nr_samples = 0;
      templ.nr_storage_samples = 0;
      source->feedback_copy = screen->resource_create(screen, &templ);
      if (!source->feedback_copy) return NULL;
   }

   struct pipe_box box = {
      .x = 0, .y = 0, .z = 0,
      .width = (int)attachment->width0,
      .height = (int)attachment->height0,
      .depth = 1,
   };
   if (!gex_wgpu_copy_texture_region(pipe, source->feedback_copy, 0, 0, 0, 0,
                                     attachment, 0, &box, false))
      return NULL;

   if (!source->feedback_copy_view) {
      WGPUTexture texture = gex_resource(source->feedback_copy)->texture;
      if (!texture) return NULL;
      WGPUTextureViewDescriptor desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      desc.dimension = WGPUTextureViewDimension_2D;
      desc.mipLevelCount = 1;
      desc.arrayLayerCount = 1;
      source->feedback_copy_view = wgpuTextureCreateView(texture, &desc);
   }
   return source->feedback_copy_view;
}





static void
gex_wgpu_resource_copy_region(struct pipe_context *pipe,
                              struct pipe_resource *dst, unsigned dst_level,
                              unsigned dstx, unsigned dsty, unsigned dstz,
                              struct pipe_resource *src, unsigned src_level,
                              const struct pipe_box *box)
{
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (ctx->failed)
      return;
   if (!gex_wgpu_copy_texture_region(pipe, dst, dst_level, dstx, dsty, dstz,
                                     src, src_level, box, false))
      gex_fail(ctx);
}

/* st/mesa reaches for this to move pixels between surfaces -- notably its
 * read-pixels staging cache, which every ReadPixels-heavy program triggers.
 *
 * Only the degenerate case is implemented: same format, same size, no
 * scaling, filtering, scissor or blending.  That is exactly a texture copy,
 * which WebGPU can do; anything else needs a real blitter drawing a textured
 * quad, and pretending otherwise would silently produce the wrong pixels.
 */
/* Resolve a multisampled texture into a single-sample one.
 *
 * WebGPU has no resolve command: a resolve happens as a side effect of ending
 * a render pass whose colour attachment names a resolveTarget.  So this is a
 * render pass that loads the multisampled texture, draws nothing, and ends.
 *
 * That mechanism is colour-only -- there is no depth resolveTarget and no
 * `depthResolveMode` in the API -- so depth goes to gex_wgpu_resolve_depth
 * above, which reads sample zero in a shader and writes frag_depth. Letting
 * depth through here instead produced a pass that raised no error and wrote
 * nothing: `gl44-msaa-depth-blit-probe` read 0.000000 everywhere.
 */
static bool
gex_wgpu_resolve(struct pipe_context *pipe, struct pipe_resource *dst,
                 unsigned dst_level, struct pipe_resource *src,
                 unsigned src_level)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   /* Submits its own command buffer, so anything already recorded has to
    * reach the queue first or it would run after this. */
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   if (dst->format != src->format || dst_level != 0 || src_level != 0) {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: resolve refused: dst %s level %u, src %s level %u\n",
                 util_format_short_name(dst->format), dst_level,
                 util_format_short_name(src->format), src_level);
      }
      return false;
   }
   /* Depth has no resolve here, and saying so matters more than it looks.
    *
    * A WebGPU resolve is a side effect of ending a render pass whose *colour*
    * attachment names a resolveTarget; there is no depth equivalent, and no
    * `depthResolveMode` in the API. Handing a depth texture to the colour
    * attachment below produced no error and no pixels:
    * `gl44-msaa-depth-blit-probe` read 0.000000 for all four quadrants where
    * native Mesa reads 0.125, 0.375, 0.625 and 0.875. Silently wrong depth is
    * the worst answer available, so refuse and name it -- the caller turns a
    * false into a GL error, which is at least true.
    *
    * Doing this properly means a shader that reads
    * `texture_depth_multisampled_2d` and writes `frag_depth`, which is the
    * same shape as the depth-upload pipeline this driver already builds. */
   if (util_format_is_depth_or_stencil(src->format))
      return gex_wgpu_resolve_depth(pipe, dst, src);

   WGPUTextureView msaa = wgpuTextureCreateView(gex_resource(src)->texture, NULL);
   WGPUTextureView single = wgpuTextureCreateView(gex_resource(dst)->texture, NULL);
   WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   if (!msaa || !single || !encoder) {
      if (msaa) wgpuTextureViewRelease(msaa);
      if (single) wgpuTextureViewRelease(single);
      if (encoder) wgpuCommandEncoderRelease(encoder);
      return false;
   }

   WGPURenderPassColorAttachment attachment =
      WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
   attachment.view = msaa;
   attachment.resolveTarget = single;
   attachment.loadOp = WGPULoadOp_Load;
   /* Kept, not discarded. GL's resolve blit reads the multisample buffer and
    * leaves it alone, and resolving for display while continuing to render to
    * the same buffer is the ordinary way to use one -- discarding here handed
    * that program an undefined attachment. It is also what lets a shader
    * fetch the samples afterwards. */
   attachment.storeOp = WGPUStoreOp_Store;
   WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   pass_desc.colorAttachmentCount = 1;
   pass_desc.colorAttachments = &attachment;

   WGPURenderPassEncoder pass =
      wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
   if (pass) {
      wgpuRenderPassEncoderEnd(pass);
      wgpuRenderPassEncoderRelease(pass);
   }
   WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
   wgpuCommandEncoderRelease(encoder);
   wgpuTextureViewRelease(msaa);
   wgpuTextureViewRelease(single);
   if (!commands || !pass)
      return false;
   gex_wgpu_queue_submit(screen, 1, &commands);
   wgpuCommandBufferRelease(commands);
   return true;
}

/* A blit the copy path cannot serve -- a format conversion, a scale, a flip, a
 * partial write mask -- becomes a textured draw.
 *
 * That is what u_blitter is for, and using it rather than refusing matters far
 * beyond one operation: a refused blit latches the context, and one such
 * refusal early in a conformance run is what turned roughly 1845 packed_pixels
 * cases into failures that had nothing to do with the formats they name. */
static void
gex_wgpu_blit_with_blitter(struct gex_wgpu_context *ctx,
                           const struct pipe_blit_info *info)
{
   if (!ctx->blitter) {
      gex_fail(ctx);
      return;
   }
   util_blitter_save_blend(ctx->blitter, ctx->blend);
   util_blitter_save_depth_stencil_alpha(ctx->blitter, ctx->dsa);
   util_blitter_save_stencil_ref(ctx->blitter, &ctx->stencil_ref);
   util_blitter_save_rasterizer(ctx->blitter, ctx->rasterizer);
   util_blitter_save_fragment_shader(ctx->blitter, ctx->fs);
   util_blitter_save_vertex_shader(ctx->blitter, ctx->vs);
   util_blitter_save_geometry_shader(ctx->blitter, ctx->gs);
   util_blitter_save_vertex_elements(ctx->blitter, ctx->vertex_elements);
   util_blitter_save_vertex_buffers(ctx->blitter, ctx->vertex_buffer_state,
                                    ARRAY_SIZE(ctx->vertex_buffer_state));
   util_blitter_save_framebuffer(ctx->blitter, &ctx->framebuffer);
   util_blitter_save_viewport(ctx->blitter, &ctx->viewport);
   util_blitter_save_scissor(ctx->blitter, &ctx->scissor);
   util_blitter_save_fragment_sampler_states(
      ctx->blitter, GEX_WGPU_MAX_SAMPLERS, (void **)ctx->samplers);
   util_blitter_save_fragment_sampler_views(
      ctx->blitter, GEX_WGPU_MAX_SAMPLERS, ctx->sampler_views);
   util_blitter_save_so_targets(ctx->blitter, ctx->so_target_count,
                                ctx->so_targets, MESA_PRIM_TRIANGLES);
   util_blitter_blit(ctx->blitter, info, NULL);
}

/* A format-converting blit of the same region, done as a pixel transfer.
 *
 * st/mesa expresses several GL pixel operations as a blit between two
 * textures of different formats: an upload whose client type does not match
 * the texture's goes into a staging texture of the client's format and is
 * blitted in, and glReadPixels/glGetTexImage come back the same way. GL's
 * meaning there is a *conversion* -- an integer value too large for the
 * destination clamps, it does not wrap -- and u_blitter's textured quad
 * cannot express that: it samples and writes through the fragment stage, so
 * what reaches an 8-bit integer attachment is whatever the write conversion
 * does with a 16-bit value.
 *
 * Reading the source back and running util_format_translate is exactly the
 * conversion GL describes, from Mesa's own description of both formats. It
 * costs a round trip, which is the right price for an operation that was
 * silently producing wrong pixels: every packed_pixels case whose upload type
 * differed in width from its internal format failed on it.
 */
/* The one conversion util_format_translate refuses: a pure-signed source with
 * a pure-unsigned destination, or the reverse. GL asks for it whenever a
 * texture's internal format and the client's type differ in signedness --
 * GL_R8UI written from GL_INT data, say -- and defines it as a clamp into the
 * destination's range, not a reinterpretation.
 *
 * Done in two steps through a 32-bit intermediate so that Mesa's own packers
 * do the per-channel width clamping: translate into the source's signedness
 * at full width, clamp the sign away, then translate out of the
 * destination's signedness. Each half has matching signedness, which is
 * exactly what util_format_translate does allow.
 */
static bool
gex_wgpu_translate_mixed_sign(enum pipe_format dst_format, void *dst,
                              unsigned dst_stride, enum pipe_format src_format,
                              const void *src, unsigned src_stride,
                              unsigned width, unsigned height)
{
   const bool src_signed = util_format_is_pure_sint(src_format);
   const bool dst_signed = util_format_is_pure_sint(dst_format);
   if (!util_format_is_pure_integer(src_format) ||
       !util_format_is_pure_integer(dst_format) || src_signed == dst_signed ||
       !width || !height)
      return false;

   const unsigned wide_stride = width * 16;
   uint32_t *texels =
      MALLOC((size_t)wide_stride * height);
   if (!texels)
      return false;
   const enum pipe_format wide_src = src_signed
      ? PIPE_FORMAT_R32G32B32A32_SINT : PIPE_FORMAT_R32G32B32A32_UINT;
   const enum pipe_format wide_dst = dst_signed
      ? PIPE_FORMAT_R32G32B32A32_SINT : PIPE_FORMAT_R32G32B32A32_UINT;
   bool ok = util_format_translate(wide_src, texels, wide_stride, 0, 0,
                                   src_format, src, src_stride, 0, 0,
                                   width, height);
   if (ok) {
      for (size_t i = 0; i < (size_t)width * height * 4; ++i) {
         if (src_signed)
            texels[i] = (int32_t)texels[i] < 0 ? 0u : texels[i];
         else if (texels[i] > (uint32_t)INT32_MAX)
            texels[i] = (uint32_t)INT32_MAX;
      }
      ok = util_format_translate(dst_format, dst, dst_stride, 0, 0,
                                 wide_dst, texels, wide_stride, 0, 0,
                                 width, height);
   }
   FREE(texels);
   return ok;
}

/* util_format_translate takes an 8-bit-unorm route whenever *either* side
 * fits in 8-bit unorm, and that route needs the source's
 * unpack_rgba_8unorm. Mesa leaves that entry a stub for the signed RGTC
 * formats -- util_format_rgtc1_snorm_unpack_rgba_8unorm prints its own name
 * and writes nothing -- and translate still returns true, so a
 * glGetTexImage of a GL_COMPRESSED_SIGNED_RED_RGTC1 texture into
 * GL_UNSIGNED_BYTE silently produced zeros and reported success. The stub
 * announced itself in the log twice per run, once per failing case.
 *
 * A source that does not fit 8-bit unorm is signed, wider than 8 bits, or
 * floating, so float is a lossless intermediate for it, and the second step
 * clamps into the destination's range the way GL asks. This costs one
 * temporary and is only taken for the mixed pair. */
static bool
gex_wgpu_translate_via_float(enum pipe_format dst_format, void *dst,
                             unsigned dst_stride, enum pipe_format src_format,
                             const void *src, unsigned src_stride,
                             unsigned width, unsigned height)
{
   if (!width || !height || (uint64_t)width * 16 * height > UINT32_MAX)
      return false;
   const unsigned wide_stride = width * 16;
   float *texels = MALLOC((size_t)wide_stride * height);
   if (!texels)
      return false;
   bool ok = util_format_translate(PIPE_FORMAT_R32G32B32A32_FLOAT, texels,
                                   wide_stride, 0, 0, src_format, src,
                                   src_stride, 0, 0, width, height);
   if (ok)
      ok = util_format_translate(dst_format, dst, dst_stride, 0, 0,
                                 PIPE_FORMAT_R32G32B32A32_FLOAT, texels,
                                 wide_stride, 0, 0, width, height);
   FREE(texels);
   return ok;
}

/* Only the asymmetric pair. Two formats that both fit 8-bit unorm, or
 * neither, already take a route translate implements for real; depth and
 * stencil are handled before that check, and pure integers have their own
 * path through gex_wgpu_translate_mixed_sign. */
static bool
gex_wgpu_needs_float_intermediate(enum pipe_format src, enum pipe_format dst)
{
   if (util_format_is_depth_or_stencil(src) ||
       util_format_is_depth_or_stencil(dst) ||
       util_format_is_pure_integer(src) || util_format_is_pure_integer(dst))
      return false;
   return !util_format_fits_8unorm(util_format_description(src)) &&
          util_format_fits_8unorm(util_format_description(dst));
}

static bool
gex_wgpu_transfer_rect(enum pipe_format dst_format, void *dst,
                       unsigned dst_stride, enum pipe_format src_format,
                       const void *src, unsigned src_stride, unsigned width,
                       unsigned height)
{
   if (gex_wgpu_needs_float_intermediate(src_format, dst_format))
      return gex_wgpu_translate_via_float(dst_format, dst, dst_stride,
                                          src_format, src, src_stride, width,
                                          height);
   if (util_format_translate(dst_format, dst, dst_stride, 0, 0, src_format,
                             src, src_stride, 0, 0, width, height))
      return true;
   return gex_wgpu_translate_mixed_sign(dst_format, dst, dst_stride,
                                        src_format, src, src_stride, width,
                                        height);
}

static bool
gex_wgpu_blit_by_transfer(struct pipe_context *pipe,
                          const struct pipe_blit_info *info, bool flip_y)
{
   struct pipe_transfer *read = NULL, *write = NULL;
   struct pipe_box src_box = info->src.box;
   struct pipe_box dst_box = info->dst.box;
   const unsigned height = (unsigned)abs(src_box.height);
   src_box.height = (int)height;
   dst_box.height = (int)height;
   if (src_box.height < 0 || dst_box.height < 0)
      return false;
   void *src = gex_wgpu_texture_map(pipe, info->src.resource, info->src.level,
                                    PIPE_MAP_READ, &src_box, &read);
   if (!src)
      return false;
   void *dst = gex_wgpu_texture_map(pipe, info->dst.resource, info->dst.level,
                                    PIPE_MAP_WRITE, &dst_box, &write);
   if (!dst) {
      gex_wgpu_texture_unmap(pipe, read);
      return false;
   }
   bool ok;
   if (!flip_y) {
      /* Whole rect at once. A block-compressed side has to be translated this
       * way: its rows are block rows, so slicing the copy into texel rows
       * would cut a 4x4 block in four. */
      ok = gex_wgpu_transfer_rect(info->dst.format, dst, write->stride,
                                  info->src.format, src, read->stride,
                                  (unsigned)src_box.width, height);
   } else {
      ok = true;
      for (unsigned row = 0; ok && row < height; ++row) {
         const unsigned from = height - 1 - row;
         uint8_t *dst_row = (uint8_t *)dst + (size_t)row * write->stride;
         const uint8_t *src_row =
            (const uint8_t *)src + (size_t)from * read->stride;
         ok = gex_wgpu_transfer_rect(info->dst.format, dst_row, write->stride,
                                     info->src.format, src_row, read->stride,
                                     (unsigned)src_box.width, 1);
      }
   }
   gex_wgpu_texture_unmap(pipe, write);
   gex_wgpu_texture_unmap(pipe, read);
   return ok;
}

static void
gex_wgpu_blit(struct pipe_context *pipe, const struct pipe_blit_info *info)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (ctx->failed)
      return;

   /* A multisampled source is a resolve, not a copy. */
   if (info->src.resource && info->dst.resource &&
       MAX2(1, info->src.resource->nr_samples) > 1 &&
       MAX2(1, info->dst.resource->nr_samples) == 1) {
      /* The stencil half of a packed depth-stencil resolve is dropped.
       *
       * gex_wgpu_resolve_depth resolves depth by rendering it -- a pipeline
       * that samples texture_depth_multisampled_2d and writes frag_depth.
       * There is no equivalent for stencil: WebGPU has no stencil resolve,
       * and a shader cannot write an arbitrary stencil value at all, only
       * apply stencil ops against a reference, so reconstructing an 8-bit
       * pattern takes eight passes of bit tests. Saying so beats a silent
       * drop -- KHR-GL33.packed_depth_stencil.blit fails on exactly this,
       * having blitted DEPTH|STENCIL and then drawn with
       * glStencilFunc(GL_EQUAL, 1). */
      if (util_format_is_depth_or_stencil(info->src.format) &&
          util_format_has_stencil(util_format_description(info->src.format))) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr, "gex-wgpu: multisample resolve of %s keeps depth "
                            "and drops stencil\n",
                    util_format_short_name(info->src.format));
         }
      }
      if (!gex_wgpu_resolve(pipe, info->dst.resource, info->dst.level,
                            info->src.resource, info->src.level))
         gex_fail(ctx);
      return;
   }

   /* A negative height on one side is GL asking for a vertical flip, not a
    * different size. */
   const bool flip_y = (info->src.box.height < 0) != (info->dst.box.height < 0);
   const bool same_size =
      info->src.box.width == info->dst.box.width &&
      abs(info->src.box.height) == abs(info->dst.box.height) &&
      info->src.box.depth == info->dst.box.depth;
   /* sRGB and its linear counterpart are the same bytes, and WebGPU treats
    * them as copy-compatible -- a texture-to-texture copy between them is
    * legal and is a plain byte copy. Requiring exact equality sent every
    * sRGB GetTexImage through u_blitter instead, which is the path that
    * aborts:
    *
    *   glGetTexImage -> st_GetTexSubImage -> gex_wgpu_blit
    *     -> util_blitter_blit -> util_blitter_blit_generic -> abort
    *
    * st/mesa reaches framebuffer 0 and staging copies through a linear view
    * of an sRGB resource, so this is the ordinary case for those formats,
    * not a corner. */
   const bool srgb_compatible =
      info->src.resource && info->dst.resource &&
      util_format_linear(info->src.format) ==
         util_format_linear(info->dst.format) &&
      util_format_linear(info->dst.format) ==
         util_format_linear(info->dst.resource->format) &&
      util_format_linear(info->src.format) ==
         util_format_linear(info->src.resource->format);
   /* Whatever aspects the destination has. A depth-stencil blit arrives with
    * PIPE_MASK_ZS, and demanding PIPE_MASK_RGBA sent every same-format depth
    * copy down the u_blitter path instead of the texture-to-texture copy
    * directly below it. */
   const unsigned copy_mask =
      !util_format_is_depth_or_stencil(info->dst.format) ? PIPE_MASK_RGBA
      : util_format_has_stencil(util_format_description(info->dst.format))
         ? PIPE_MASK_ZS : PIPE_MASK_Z;
   /* render_condition_enable is not a refusal: it asks the blit to respect an
    * active conditional render, and this driver implements no
    * set_render_condition, so none can be active. st/mesa sets it on every
    * blit it issues, which was sending every same-format depth copy to
    * u_blitter instead of the texture-to-texture copy below. */
   if (!info->src.resource || !info->dst.resource ||
       !srgb_compatible ||
       !same_size || info->scissor_enable || info->swizzle_enable ||
       info->alpha_blend ||
       info->num_window_rectangles != 0 ||
       info->mask != copy_mask) {
      /* Declined rather than sent to u_blitter, which aborts the whole wasm
       * module here:
       *
       *   glGetTexImage -> st_GetTexSubImage -> gex_wgpu_blit
       *     -> util_blitter_blit -> util_blitter_blit_generic
       *     -> Aborted("pthread mutex deadlock detected")
       *
       * An aborted module is not one failed blit: every later call throws,
       * so the rest of the run is lost -- measured at 16426 dead submits and
       * every case after the first abort. Declining costs the one operation
       * that cannot be done, which is the trade this driver makes everywhere
       * else. Restore the blitter call once the deadlock is fixed; this is
       * containment, not a solution. */
      /* Declined rather than sent to u_blitter, which faults the whole wasm
       * module here:
       *
       *   glGetTexImage -> st_GetTexSubImage -> gex_wgpu_blit
       *     -> util_blitter_blit -> util_blitter_blit_generic
       *     -> "memory access out of bounds", then a mutex abort behind it
       *
       * The fault is inside util_blitter_blit_generic with no frame below it,
       * which is what a dereference of something this driver handed back
       * looks like: u_blitter builds a sampler view for the source and does
       * not check it, and create_sampler_view returns NULL for a format this
       * driver cannot name. The source format is printed above -- supporting
       * it is the real fix, and this decline is containment.
       *
       * A faulted module is not one failed blit: every later call throws, so
       * the rest of the run is lost. Measured: declining takes the run from
       * "aborts 611 cases in, reports 349" to "no aborts, reports 615", and
       * 615 is the honest number because the 349 was counted by a corpse. */
      /* One shape u_blitter gets wrong rather than merely cannot do: the
       * same region in a different format. That is a pixel transfer, and it
       * is done here as one. */
      /* Depth and stencil convert through this too. util_format_translate
       * knows both aspects, and texture_map already resolves a packed depth
       * texture on read and draws one back on write, so a
       * depth16 -> depth32f transfer is expressible here where u_blitter's
       * quad is not. The mask has to cover every aspect the destination has,
       * or the aspects it does not name would be overwritten. */
      const bool zs = util_format_is_depth_or_stencil(info->dst.format);
      const unsigned whole_mask = !zs ? PIPE_MASK_RGBA
         : util_format_has_stencil(util_format_description(info->dst.format))
            ? PIPE_MASK_ZS : PIPE_MASK_Z;
      const bool transferable =
         info->src.resource && info->dst.resource && same_size &&
         !info->scissor_enable && !info->swizzle_enable &&
         !info->alpha_blend &&
         info->num_window_rectangles == 0 &&
         info->mask == whole_mask &&
         zs == util_format_is_depth_or_stencil(info->src.format) &&
         info->src.box.depth == 1 && info->dst.box.depth == 1 &&
         info->src.box.z == 0 && info->dst.box.z == 0 &&
         MAX2(1, info->src.resource->nr_samples) == 1 &&
         MAX2(1, info->dst.resource->nr_samples) == 1 &&
         /* A block format has to be translated as a whole rect, which the
          * flipping path cannot do. Decompressing on readback is how
          * glGetTexImage of a compressed texture works, and declining it left
          * the caller's buffer untouched -- the three S3TC internal formats
          * read back as zeros. */
         (!flip_y || (!util_format_is_compressed(info->src.format) &&
                      !util_format_is_compressed(info->dst.format)));
      if (transferable && gex_wgpu_blit_by_transfer(pipe, info, flip_y))
         return;
      /* Reported here, after the transfer path has had its turn, so that a
       * blit this driver does serve is not announced as a decline. */
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: blit sent to u_blitter: %ux%u->%ux%u %s->%s "
                 "scissor=%d swizzle=%d blend=%d cond=%d wr=%u mask=0x%x\n",
                 info->src.box.width, info->src.box.height,
                 info->dst.box.width, info->dst.box.height,
                 util_format_short_name(info->src.format),
                 util_format_short_name(info->dst.format),
                 (int)info->scissor_enable, (int)info->swizzle_enable,
                 (int)info->alpha_blend, (int)info->render_condition_enable,
                 info->num_window_rectangles, info->mask);
      }
      gex_wgpu_blit_with_blitter(ctx, info);
      return;
   }

   if (!gex_wgpu_copy_texture_region(pipe, info->dst.resource,
                                     info->dst.level, info->dst.box.x,
                                     info->dst.box.y, info->dst.box.z,
                                     info->src.resource, info->src.level,
                                     &info->src.box, flip_y))
      gex_fail(ctx);
}

static void
gex_wgpu_set_sample_mask(struct pipe_context *pipe, unsigned sample_mask)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   if (ctx->failed)
      return;
   if (!(sample_mask & 1u))
      gex_fail(ctx);
}

static void
gex_wgpu_clear(struct pipe_context *pipe, unsigned buffers,
               uint32_t color_clear_mask, uint8_t stencil_clear_mask,
               const struct pipe_scissor_state *scissor_state,
               const union pipe_color_union *color, double depth,
               unsigned stencil)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   /* A partial channel or stencil mask needs a masked clear, which the
    * per-target clears below cannot express; refuse it rather than write
    * channels GL asked to be left alone. */
   if (ctx->failed || scissor_state ||
       ((buffers & PIPE_CLEAR_STENCIL) && stencil_clear_mask != 0xff)) {
      gex_fail(ctx);
      return;
   }

   if (buffers & PIPE_CLEAR_COLOR) {
      for (unsigned i = 0; i < ctx->color_count; ++i) {
         struct pipe_resource *resource = ctx->colors[i];
         if (!resource || !(buffers & (PIPE_CLEAR_COLOR0 << i)))
            continue;
         /* Four colormask bits per draw buffer; anything but all four is a
          * masked clear. */
         if (((color_clear_mask >> (4 * i)) & 0xf) != 0xf) {
            gex_fail(ctx);
            return;
         }
         /* set_framebuffer_state already required level zero, layer zero and
          * a surface format matching the texture, so the surface it validated
          * is reconstructible from the resource alone. */
         struct pipe_surface surface = {0};
         surface.texture = resource;
         surface.format = resource->format;
         surface.level = 0;
         surface.first_layer = 0;
         surface.last_layer = 0;
         gex_wgpu_clear_render_target(pipe, &surface, color, 0, 0,
                                      resource->width0, resource->height0,
                                      false);
      }
   }

   const unsigned depth_stencil = buffers & PIPE_CLEAR_DEPTHSTENCIL;
   if (depth_stencil && ctx->depth) {
      struct pipe_resource *resource = ctx->depth;
      struct pipe_surface surface = {0};
      surface.texture = resource;
      surface.format = resource->format;
      surface.level = 0;
      surface.first_layer = 0;
      surface.last_layer = 0;
      gex_wgpu_clear_depth_stencil(pipe, &surface, depth_stencil, depth,
                                   stencil, 0, 0, resource->width0,
                                   resource->height0, false);
   }
}

static void
gex_wgpu_set_framebuffer_state(struct pipe_context *pipe,
                               const struct pipe_framebuffer_state *state)
{
   GEX_ENTRY_TIMER();
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   /* Kept whole for u_blitter, as above. */
   util_copy_framebuffer_state(&ctx->framebuffer, state);
   struct pipe_resource *colors[GEX_WGPU_MAX_COLOR_TARGETS] = {0};
   enum pipe_format color_formats[GEX_WGPU_MAX_COLOR_TARGETS] = {0};
   unsigned color_levels[GEX_WGPU_MAX_COLOR_TARGETS] = {0};
   unsigned color_layers[GEX_WGPU_MAX_COLOR_TARGETS] = {0};
   unsigned color_count = 0;
   struct pipe_resource *depth = NULL;
   if (state && state->nr_cbufs) {
      if (state->nr_cbufs > GEX_WGPU_MAX_COLOR_TARGETS ||
          state->width == 0 || state->height == 0) {
         gex_fail(ctx);
      } else {
         color_count = state->nr_cbufs;
         for (unsigned i = 0; i < color_count; ++i) {
            const struct pipe_surface *surface = &state->cbufs[i];
            /* GL lets a framebuffer have a hole -- attachment 0 bound and
             * attachment 1 not -- and glDrawBuffers names GL_NONE for it.
             * WebGPU expresses exactly that as a null colour attachment with
             * an Undefined target format, so a hole is carried through rather
             * than refused. Refusing it latched the context, which through
             * st/mesa's GL_OUT_OF_MEMORY ends the whole CTS run. */
            if (!surface->texture) {
               colors[i] = NULL;
               color_formats[i] = PIPE_FORMAT_NONE;
               color_levels[i] = 0;
               color_layers[i] = 0;
               continue;
            }
            const char *why =
               state->width > surface->texture->width0 ||
                 state->height > surface->texture->height0 ? "larger than the texture"
               : (surface->format != surface->texture->format &&
                  util_format_linear(surface->format) !=
                     util_format_linear(surface->texture->format))
                                                    ? "surface reinterprets the format"
               : !gex_wgpu_color_renderable(pipe->screen, surface->format)
                                                    ? "format is not renderable"
               : surface->first_layer != surface->last_layer
                                                    ? "more than one layer" : NULL;
            if (why) {
               if (!ctx->failed)
                  fprintf(stderr,
                          "gex-wgpu: colour target %u refused (%s) fb=%ux%u "
                          "surface=%u texture=%u level=%u layers=%u..%u\n",
                          i, why, state->width, state->height,
                          surface->format,
                          surface->texture ? surface->texture->format : 0,
                          surface->level, surface->first_layer,
                          surface->last_layer);
               gex_fail(ctx);
               color_count = 0;
               break;
            }
            colors[i] = surface->texture;
            color_formats[i] = surface->format;
            color_levels[i] = surface->level;
            color_layers[i] = surface->first_layer;
            /* Attached as a render target, so from here on it has contents
             * whether or not anything uploaded to it. */
            gex_resource(surface->texture)->texture_written = true;
         }
         /* WebGPU bounds the attachments' combined bytes per sample. Exceeding
          * it is a validation error at pipeline creation, which poisons the
          * device; dropping the attachments past the budget leaves a
          * framebuffer st/mesa reports as unsupported, which is what GL says
          * when a driver cannot honour a set of draw buffers. */
         unsigned budget = gex_screen(pipe->screen)->max_colour_bytes_per_sample;
         for (unsigned i = 0; i < color_count; ++i) {
            const unsigned bytes = colors[i]
               ? util_format_get_blocksize(color_formats[i]) : 0;
            if (bytes <= budget) {
               budget -= bytes;
               continue;
            }
            static bool said;
            if (!said) {
               said = true;
               fprintf(stderr,
                       "gex-wgpu: colour target %u dropped: the attachments "
                       "exceed %u bytes per sample\n", i,
                       gex_screen(pipe->screen)->max_colour_bytes_per_sample);
            }
            colors[i] = NULL;
            color_formats[i] = PIPE_FORMAT_NONE;
         }
      }
   }
   /* Derived from the format table rather than a repeated list. Adding
    * Z16_UNORM to the table and to is_format_supported without updating a
    * hand-written list here let st/mesa create a depth16 texture, attach it,
    * and latch the context on the attachment -- which cost 189 cases in
    * packed_pixels.pbo_rectangle before the regression was traced back. A
    * depth format this driver can name to WebGPU is one it can attach. */
   const bool zs_attachable =
      state && state->zsbuf.texture &&
      util_format_is_depth_or_stencil(state->zsbuf.format) &&
      gex_wgpu_texture_format(state->zsbuf.format) !=
         WGPUTextureFormat_Undefined;
   if (zs_attachable &&
       state->zsbuf.level == 0 && state->zsbuf.first_layer == 0 &&
       state->zsbuf.last_layer == 0 &&
       state->width <= state->zsbuf.texture->width0 &&
       state->height <= state->zsbuf.texture->height0) {
      depth = state->zsbuf.texture;
      /* Same as the colour targets above: attached, so it has contents. A
       * shadow map is written only by being rendered into, and counting that
       * as "nothing ever wrote it" made the unwritten-texture warning report
       * every shadow lookup in the scene. */
      gex_resource(depth)->texture_written = true;
   } else if (state && state->zsbuf.texture) {
      gex_fail(ctx);
   }
   for (unsigned i = 0; i < GEX_WGPU_MAX_COLOR_TARGETS; ++i) {
      pipe_resource_reference(&ctx->colors[i],
                              i < color_count ? colors[i] : NULL);
      ctx->color_formats[i] = i < color_count ? color_formats[i]
                                             : PIPE_FORMAT_NONE;
      ctx->color_levels[i] = i < color_count ? color_levels[i] : 0;
      ctx->color_layers[i] = i < color_count ? color_layers[i] : 0;
   }
   ctx->color_count = color_count;
   pipe_resource_reference(&ctx->depth, depth);
   /* From the framebuffer, not from the colour count: a depth-only pass has
    * no colour attachment and still has a size, and taking zero here made
    * every viewport look out of bounds. */
   const bool has_attachment = color_count || depth;
   ctx->fb_width = has_attachment && state ? state->width : 0;
   ctx->fb_height = has_attachment && state ? state->height : 0;
   /* Every attachment must agree, which GL also requires of a complete
    * framebuffer -- but the count comes from whichever attachment there is,
    * not from colour attachment zero.
    *
    * Seeded at one, a depth-only framebuffer compared its 4x depth attachment
    * against a colour count that was never set and latched the context. GL
    * allows a framebuffer with depth and no colour, and a multisampled depth
    * blit is exactly that shape: `gl44-msaa-depth-blit-probe` reported
    * GL_OUT_OF_MEMORY -- this driver's latched context, seen from the
    * application -- on a framebuffer that was otherwise complete. */
   uint32_t samples = 0;
   for (unsigned i = 0; i < color_count; ++i) {
      const uint32_t attachment = MAX2(1, colors[i]->nr_samples);
      if (!samples)
         samples = attachment;
      else if (attachment != samples)
         gex_fail(ctx);
   }
   if (depth) {
      const uint32_t attachment = MAX2(1, depth->nr_samples);
      if (!samples)
         samples = attachment;
      else if (attachment != samples)
         gex_fail(ctx);
   }
   ctx->fb_samples = MAX2(1, samples);
}

/* Build, or find, the compute variant of this vertex shader for this vertex
 * layout. */
static struct gex_wgpu_capture_variant *
gex_wgpu_capture_variant_get(struct gex_wgpu_context *ctx,
                             struct gex_wgpu_shader *vs,
                             const struct gex_wgpu_capture_layout *layout)
{
   for (struct gex_wgpu_capture_variant *v = vs->captures; v; v = v->next)
      if (!memcmp(&v->layout, layout, sizeof(*layout)))
         return v;

   nir_shader *nir = nir_shader_clone(NULL, vs->capture_nir);
   if (!nir)
      return NULL;

   struct gex_wgpu_capture state = {0};
   state.layout = layout;
   state.params = gex_wgpu_word_block(nir, "gex_capture_params",
                                      GEX_WGPU_CAPTURE_PARAMS_BINDING, 0);
   if (layout->index_table)
      state.index = gex_wgpu_word_block(nir, "gex_capture_index",
                                        GEX_WGPU_CAPTURE_INDEX_BINDING, 0);

   /* Order matters. The attribute reads and the vertex-stage system values
    * have to become compute terms before anything else refers to them, the
    * capture stores are emitted from the output writes, and only then can
    * those writes go away. */
   nir_shader_intrinsics_pass(nir, gex_wgpu_capture_input,
                              nir_metadata_control_flow, &state);
   nir_shader_intrinsics_pass(nir, gex_wgpu_capture_builtin,
                              nir_metadata_control_flow, &state);
   nir_shader_intrinsics_pass(nir, gex_wgpu_capture_store,
                              nir_metadata_control_flow, &state);
   nir_shader_intrinsics_pass(nir, gex_wgpu_capture_drop_output,
                              nir_metadata_control_flow, &state);
   if (getenv("GEX_WGPU_XFB_DEBUG")) {
      const nir_xfb_info *dbg = nir->xfb_info;
      fprintf(stderr, "gex-xfb: so_mask=0x%x failed=%d outputs=%u written=0x%x\n",
              state.so_mask, (int)state.failed,
              dbg ? dbg->output_count : 0u,
              dbg ? dbg->buffers_written : 0u);
      for (unsigned o = 0; dbg && o < dbg->output_count; ++o)
         fprintf(stderr, "gex-xfb:   out[%u] loc=%u buffer=%u offset=%u mask=0x%x stride=%u\n",
                 o, dbg->outputs[o].location, dbg->outputs[o].buffer,
                 dbg->outputs[o].offset, dbg->outputs[o].component_mask,
                 dbg->buffers[dbg->outputs[o].buffer].stride);
   }
   if (state.failed || !state.so_mask) {
      ralloc_free(nir);
      return NULL;
   }

   /* Invocations past the vertex count must not write: the capture buffer is
    * usually longer than the draw, and WebGPU's bounds checking would not
    * catch a write that lands inside it but past the captured range. */
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_push_if(&b, nir_uge(&b, gex_wgpu_capture_slot(&b),
                           gex_wgpu_param(&b, &state, GEX_WGPU_PARAM_COUNT)));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_returns);

   nir->info.stage = MESA_SHADER_COMPUTE;
   nir->info.workgroup_size[0] = GEX_WGPU_CAPTURE_GROUP;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
   nir->info.num_ssbos = GEX_WGPU_CAPTURE_INDEX_BINDING + 1;
   nir->xfb_info = NULL;
   nir_remove_dead_variables(
      nir, nir_var_shader_in | nir_var_shader_out | nir_var_system_value, NULL);
   nir_fixup_deref_types(nir);
   NIR_PASS(_, nir, nir_opt_dce);

   struct gex_wgpu_capture_variant *variant =
      CALLOC_STRUCT(gex_wgpu_capture_variant);
   if (!variant) {
      ralloc_free(nir);
      return NULL;
   }
   variant->layout = *layout;
   variant->so_mask = state.so_mask;
   variant->vb_mask = state.vb_mask;
   for (unsigned i = 0; i < GEX_WGPU_MAX_SO_BUFFERS; ++i)
      variant->so_bindings[i] = GEX_WGPU_SO_FIRST_BINDING + i;
   for (unsigned i = 0; i < GEX_WGPU_MAX_VERTEX_BUFFERS; ++i)
      variant->vb_bindings[i] = GEX_WGPU_VB_FIRST_BINDING + i;

   char *error = NULL;
   variant->module = gex_wgpu_compile_nir(&ctx->base, nir,
                                          MESA_SHADER_COMPUTE, false, &error);
   if (!variant->module) {
      fprintf(stderr, "gex-wgpu: capture shader rejected: %s\n",
              error ? error : "unknown");
      free(error);
      FREE(variant);
      return NULL;
   }
   free(error);
   variant->next = vs->captures;
   vs->captures = variant;
   return variant;
}

/* Write one vec4 slot of an interstage buffer. */
static void
gex_wgpu_store_varying(nir_builder *b, nir_variable *buffer, nir_def *byte,
                       nir_def *value)
{
   nir_deref_instr *data = gex_wgpu_block_words(b, buffer);
   nir_def *word = nir_ushr_imm(b, byte, 2);
   for (unsigned c = 0; c < 4; ++c) {
      nir_deref_instr *slot =
         nir_build_deref_array(b, data, nir_iadd_imm(b, word, c));
      nir_def *component = c < value->num_components
         ? nir_channel(b, value, c) : nir_imm_float(b, 0.0f);
      nir_store_deref(b, slot, component, 0x1);
   }
}

/* Append the varying writes at the end of the shader.
 *
 * Reading the output variables once at the end, rather than rewriting each
 * store, is what makes this correct for a shader that writes an output
 * partially, more than once, or under control flow: the variable holds the
 * final value either way. */
static bool
gex_wgpu_emit_varying_stores(nir_shader *nir, struct gex_wgpu_capture *state,
                             nir_def *(*slot_of)(nir_builder *,
                                                 struct gex_wgpu_capture *))
{
   const struct gex_wgpu_varying_layout *layout = state->varyings;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   const unsigned stride = gex_wgpu_varying_stride(layout);
   nir_def *base = nir_imul_imm(&b, slot_of(&b, state), stride);

   for (unsigned i = 0; i < layout->count; ++i) {
      nir_variable *found = NULL;
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
         if (var->data.location == layout->location[i]) {
            found = var;
            break;
         }
      }
      /* A varying the consumer reads but this shader never declared reads as
       * zero, which is what GL says an unwritten input is worth. */
      nir_def *value = nir_imm_zero(&b, 4, 32);
      if (found) {
         if (!glsl_type_is_vector_or_scalar(found->type))
            return false;
         value = nir_load_var(&b, found);
      }
      gex_wgpu_store_varying(&b, state->varying_out,
                             nir_iadd_imm(&b, base, i * 16), value);
   }
   nir_progress(true, impl, nir_metadata_none);
   return true;
}

static nir_def *
gex_wgpu_capture_slot_of(nir_builder *b, struct gex_wgpu_capture *state)
{
   (void)state;
   return gex_wgpu_capture_slot(b);
}

/* Build, or find, the compute variant of this vertex shader that writes the
 * given varyings for a geometry shader to read. */
static struct gex_wgpu_varying_variant *
gex_wgpu_varying_variant_get(struct gex_wgpu_context *ctx,
                             struct gex_wgpu_shader *vs,
                             const struct gex_wgpu_capture_layout *vertex,
                             const struct gex_wgpu_varying_layout *varyings)
{
   for (struct gex_wgpu_varying_variant *v = vs->varyings; v; v = v->next) {
      if (!memcmp(&v->vertex_layout, vertex, sizeof(*vertex)) &&
          !memcmp(&v->varyings, varyings, sizeof(*varyings)))
         return v;
   }
   if (!vs->varying_nir)
      return NULL;

   nir_shader *nir = nir_shader_clone(NULL, vs->varying_nir);
   if (!nir)
      return NULL;

   struct gex_wgpu_capture state = {0};
   state.layout = vertex;
   state.varyings = varyings;
   state.params = gex_wgpu_word_block(nir, "gex_capture_params",
                                      GEX_WGPU_CAPTURE_PARAMS_BINDING, 0);
   state.varying_out = gex_wgpu_word_block(nir, "gex_varying_out",
                                           GEX_WGPU_VARYING_OUT_BINDING, 0);

   nir_shader_intrinsics_pass(nir, gex_wgpu_capture_input,
                              nir_metadata_control_flow, &state);
   nir_shader_intrinsics_pass(nir, gex_wgpu_capture_builtin,
                              nir_metadata_control_flow, &state);
   if (state.failed ||
       !gex_wgpu_emit_varying_stores(nir, &state, gex_wgpu_capture_slot_of)) {
      ralloc_free(nir);
      return NULL;
   }

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_push_if(&b, nir_uge(&b, gex_wgpu_capture_slot(&b),
                           gex_wgpu_param(&b, &state, GEX_WGPU_PARAM_COUNT)));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_returns);

   nir->info.stage = MESA_SHADER_COMPUTE;
   nir->info.workgroup_size[0] = GEX_WGPU_CAPTURE_GROUP;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
   nir->xfb_info = NULL;
   /* The outputs stay live -- the varying writes read them at the end -- so
    * they cannot simply be dropped, and a compute shader may not declare
    * them: Tint rejects a BuiltIn Position outside the stages that have one.
    * They become ordinary private variables instead. */
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_out)
      var->data.mode = nir_var_shader_temp;
   nir_fixup_deref_modes(nir);
   nir_remove_dead_variables(nir, nir_var_shader_in | nir_var_system_value,
                             NULL);
   nir_fixup_deref_types(nir);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_dce);

   struct gex_wgpu_varying_variant *variant =
      CALLOC_STRUCT(gex_wgpu_varying_variant);
   if (!variant) {
      ralloc_free(nir);
      return NULL;
   }
   variant->vertex_layout = *vertex;
   variant->varyings = *varyings;
   variant->vb_mask = state.vb_mask;

   char *error = NULL;
   variant->module = gex_wgpu_compile_nir(&ctx->base, nir,
                                          MESA_SHADER_COMPUTE, false, &error);
   if (!variant->module) {
      fprintf(stderr, "gex-wgpu: varying shader rejected: %s\n",
              error ? error : "unknown");
      free(error);
      FREE(variant);
      return NULL;
   }
   free(error);
   variant->next = vs->varyings;
   vs->varyings = variant;
   return variant;
}

/* ---- The geometry shader, lowered to compute ------------------------------
 *
 * One invocation per input primitive. It reads its input vertices out of the
 * buffer the vertex shader wrote, runs the shader body, and turns each
 * EmitVertex into a vertex written to the output buffer plus, once a strip has
 * three vertices, three indices written to the output index buffer.
 *
 * Nothing here introduces control flow, which matters because these run as
 * intrinsic passes over the existing blocks. Where a rule would want a branch
 * -- "only emit a triangle once the strip is long enough" -- it is expressed
 * as a select instead: an emit that completes no triangle writes three equal
 * indices, which is a degenerate triangle, into the slot the next real
 * triangle will overwrite anyway.
 */
struct gex_wgpu_gs_lowering {
   const struct gex_wgpu_gs *gs;
   nir_variable *params;
   nir_variable *vs_out;
   nir_variable *vertices;
   nir_variable *indices;
   nir_variable *verts_in_prim;
   nir_variable *triangles;
   bool failed;
};

static nir_def *
gex_wgpu_gs_invocation(nir_builder *b)
{
   return nir_channel(b, nir_load_global_invocation_id(b, 32), 0);
}

static nir_def *
gex_wgpu_load_varying(nir_builder *b, nir_variable *buffer, nir_def *byte,
                      unsigned components)
{
   nir_deref_instr *data = gex_wgpu_block_words(b, buffer);
   nir_def *word = nir_ushr_imm(b, byte, 2);
   nir_def *comp[4];
   for (unsigned c = 0; c < 4; ++c)
      comp[c] = nir_load_deref(
         b, nir_build_deref_array(b, data, nir_iadd_imm(b, word, c)));
   return nir_vec(b, comp, components);
}

/* A per-vertex geometry input becomes a read of the vertex shader's output
 * buffer, at the slot for (this primitive's base vertex + the vertex the
 * shader asked for). */
static bool
gex_wgpu_gs_input(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gex_wgpu_gs_lowering *state = data;
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (!deref || deref->deref_type != nir_deref_type_array)
      return false;
   nir_deref_instr *parent = nir_deref_instr_parent(deref);
   if (!parent || parent->deref_type != nir_deref_type_var ||
       parent->var->data.mode != nir_var_shader_in)
      return false;

   const int slot =
      gex_wgpu_varying_slot(&state->gs->inputs, parent->var->data.location);
   if (slot < 0) {
      state->failed = true;
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *vertex = deref->arr.index.ssa;
   nir_def *index =
      nir_iadd(b, nir_imul_imm(b, gex_wgpu_gs_invocation(b),
                               state->gs->verts_per_input_prim),
               vertex);
   nir_def *byte =
      nir_iadd_imm(b,
                   nir_imul_imm(b, index,
                                gex_wgpu_varying_stride(&state->gs->inputs)),
                   slot * 16);
   nir_def *value = gex_wgpu_load_varying(b, state->vs_out, byte,
                                          intr->def.num_components);
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

/* EmitVertex: write the output variables to the vertex buffer, and close a
 * triangle if the strip now has one. */
static bool
gex_wgpu_gs_emit(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gex_wgpu_gs_lowering *state = data;
   const struct gex_wgpu_gs *gs = state->gs;

   if (intr->intrinsic == nir_intrinsic_end_primitive_with_counter) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_store_var(b, state->verts_in_prim, nir_imm_int(b, 0), 0x1);
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_set_vertex_and_primitive_count) {
      /* Dropped. Note for anyone implementing transform feedback from this
       * stage: this intrinsic carries the emitted vertex and primitive counts
       * the shader actually reached, and those counts are the one piece the
       * buffers do not otherwise record -- an unemitted index slot holds a
       * degenerate triangle, which a legitimately degenerate emitted one is
       * indistinguishable from. Recording it per invocation is the first step
       * of that work; see the transform-feedback note in the handoff doc. */
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_emit_vertex_with_counter)
      return false;
   /* A stream other than zero has nowhere to go: this path rasterizes what it
    * writes, and transform feedback out of a geometry shader is separate. */
   if (nir_intrinsic_stream_id(intr) != 0) {
      state->failed = true;
      return false;
   }

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *invocation = gex_wgpu_gs_invocation(b);
   nir_def *emitted = intr->src[0].ssa;
   nir_def *base = nir_imul_imm(b, invocation, gs->max_vertices);
   nir_def *slot = nir_iadd(b, base, emitted);
   const unsigned stride = gex_wgpu_varying_stride(&gs->outputs);
   nir_def *vertex_byte = nir_imul_imm(b, slot, stride);

   for (unsigned i = 0; i < gs->outputs.count; ++i) {
      nir_variable *found = NULL;
      nir_foreach_variable_with_modes(var, b->shader, nir_var_shader_out) {
         if (var->data.location == gs->outputs.location[i]) {
            found = var;
            break;
         }
      }
      nir_def *value = found && glsl_type_is_vector_or_scalar(found->type)
         ? nir_load_var(b, found) : nir_imm_zero(b, 4, 32);
      gex_wgpu_store_varying(b, state->vertices,
                             nir_iadd_imm(b, vertex_byte, i * 16), value);
   }

   /* Points: every emitted vertex is a primitive of its own, so there is no
    * strip to close and no winding to alternate -- one index, written at the
    * running primitive counter. */
   if (gs->output_prim == MESA_PRIM_POINTS) {
      nir_def *point =
         nir_iadd(b, nir_imul_imm(b, invocation, gs->max_output_prims),
                  nir_load_var(b, state->triangles));
      nir_deref_instr *point_data = gex_wgpu_block_words(b, state->indices);
      nir_deref_instr *at = nir_build_deref_array(b, point_data, point);
      nir_store_deref(b, at, slot, 0x1);
      nir_store_var(b, state->triangles,
                    nir_iadd_imm(b, nir_load_var(b, state->triangles), 1), 0x1);
      nir_instr_remove(&intr->instr);
      return true;
   }

   /* The strip closes a triangle from this vertex and the previous two, with
    * the winding flipped on every other one. Before there are two previous
    * vertices there is no triangle, and the three indices collapse to one, so
    * the slot holds a degenerate triangle until a real one overwrites it. */
   nir_def *in_prim = nir_load_var(b, state->verts_in_prim);
   nir_def *closes = nir_uge_imm(b, in_prim, 2);
   nir_def *odd = nir_i2b(b, nir_iand_imm(b, in_prim, 1));
   nir_def *first = nir_iadd(b, base, nir_iadd_imm(b, emitted, -2));
   nir_def *second = nir_iadd(b, base, nir_iadd_imm(b, emitted, -1));
   nir_def *third = slot;
   nir_def *a = nir_bcsel(b, closes, nir_bcsel(b, odd, second, first), base);
   nir_def *c = nir_bcsel(b, closes, nir_bcsel(b, odd, first, second), base);
   nir_def *e = nir_bcsel(b, closes, third, base);

   nir_def *triangle =
      nir_iadd(b, nir_imul_imm(b, invocation, gs->max_triangles),
               nir_load_var(b, state->triangles));
   nir_deref_instr *index_data = gex_wgpu_block_words(b, state->indices);
   nir_def *index_word = nir_imul_imm(b, triangle, 3);
   nir_def *written[3] = {a, c, e};
   for (unsigned i = 0; i < 3; ++i) {
      nir_deref_instr *at = nir_build_deref_array(
         b, index_data, nir_iadd_imm(b, index_word, i));
      nir_store_deref(b, at, written[i], 0x1);
   }

   nir_store_var(b, state->triangles,
                 nir_iadd(b, nir_load_var(b, state->triangles),
                          nir_b2i32(b, closes)), 0x1);
   nir_store_var(b, state->verts_in_prim, nir_iadd_imm(b, in_prim, 1), 0x1);
   nir_instr_remove(&intr->instr);
   return true;
}

/* gl_PrimitiveIDIn is the input primitive, which is this invocation. */
static bool
gex_wgpu_gs_builtin(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct gex_wgpu_gs_lowering *state = data;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = NULL;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_primitive_id:
      value = gex_wgpu_gs_invocation(b);
      break;
   case nir_intrinsic_load_invocation_id:
      /* Geometry shader instancing is refused when the shader is created. */
      value = nir_imm_int(b, 0);
      break;
   default:
      return false;
   }
   (void)state;
   nir_def_rewrite_uses(&intr->def, value);
   nir_instr_remove(&intr->instr);
   return true;
}

/* Collect the varyings a stage declares, in location order, into the vec4-slot
 * layout the interstage buffers use. */
static bool
gex_wgpu_collect_varyings(nir_shader *nir, nir_variable_mode mode,
                          struct gex_wgpu_varying_layout *layout)
{
   memset(layout, 0, sizeof(*layout));
   nir_foreach_variable_with_modes(var, nir, mode) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_vector_or_scalar(type))
         return false;
      if (layout->count >= GEX_WGPU_MAX_VARYINGS)
         return false;
      /* Keep it sorted so producer and consumer agree without negotiating. */
      unsigned at = layout->count;
      while (at > 0 && layout->location[at - 1] > var->data.location) {
         layout->location[at] = layout->location[at - 1];
         --at;
      }
      layout->location[at] = var->data.location;
      ++layout->count;
   }
   return true;
}

/* The rasterization half: a vertex shader that reads the geometry shader's
 * output buffer as an ordinary vertex buffer and passes it straight through.
 *
 * Reading it as a vertex buffer rather than a storage buffer is deliberate.
 * WGSL only allows read-only storage in the vertex stage, and the SPIR-V
 * emitter declares every storage buffer read_write, so a vertex shader that
 * loaded the buffer directly would be rejected. Fixed-function vertex fetch
 * has no such restriction, and the buffer is already vec4 slots at a fixed
 * stride, which is exactly a vertex layout. */
static nir_shader *
gex_wgpu_gs_passthrough_nir(struct pipe_context *pipe,
                            const struct gex_wgpu_varying_layout *outputs)
{
   /* Built with this driver's own compiler options rather than none. A shader
    * created with NULL options has `nir->options` null, and NIR dereferences
    * it unconditionally -- nir_cleanup_functions reads
    * options->driver_functions -- so the passes this shader goes through were
    * reading through a null pointer. UBSan caught it; nothing had crashed,
    * which is exactly why it survived. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, pipe->screen->nir_options[MESA_SHADER_VERTEX],
      "gex-gs-passthrough");
   for (unsigned i = 0; i < outputs->count; ++i) {
      char name[24];
      snprintf(name, sizeof(name), "gex_gs_varying%u", i);
      nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in,
                                             glsl_vec4_type(), name);
      in->data.location = VERT_ATTRIB_GENERIC0 + i;
      in->data.driver_location = i;
      nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
                                              glsl_vec4_type(), name);
      out->data.location = outputs->location[i];
      out->data.driver_location = i;
      nir_store_var(&b, out, nir_load_var(&b, in), 0xf);
   }
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   return b.shader;
}

static WGPUShaderModule
gex_wgpu_gs_build_module(struct pipe_context *pipe, struct gex_wgpu_gs *gs)
{
   nir_shader *nir = nir_shader_clone(NULL, gs->nir);
   if (!nir)
      return NULL;

   struct gex_wgpu_gs_lowering state = {0};
   state.gs = gs;
   state.params = gex_wgpu_word_block(nir, "gex_gs_params",
                                      GEX_WGPU_GS_PARAMS_BINDING, 0);
   state.vs_out = gex_wgpu_word_block(nir, "gex_gs_vsout",
                                      GEX_WGPU_GS_VSOUT_BINDING, 1);
   state.vertices = gex_wgpu_word_block(nir, "gex_gs_vertices",
                                        GEX_WGPU_GS_VERTEX_BINDING, 2);
   state.indices = gex_wgpu_word_block(nir, "gex_gs_indices",
                                       GEX_WGPU_GS_INDEX_BINDING, 3);

   /* The shader's own constant buffers move to group one, so group zero is
    * this pass's buffers alone -- the same split the vertex compute pass
    * uses, so one dispatch shape serves both. */
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo)
      var->data.descriptor_set = 1;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   state.verts_in_prim =
      nir_local_variable_create(impl, glsl_uint_type(), "gex_verts_in_prim");
   state.triangles =
      nir_local_variable_create(impl, glsl_uint_type(), "gex_triangles");
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_store_var(&b, state.verts_in_prim, nir_imm_int(&b, 0), 0x1);
   nir_store_var(&b, state.triangles, nir_imm_int(&b, 0), 0x1);

   nir_shader_intrinsics_pass(nir, gex_wgpu_gs_input,
                              nir_metadata_control_flow, &state);
   nir_shader_intrinsics_pass(nir, gex_wgpu_gs_builtin,
                              nir_metadata_control_flow, &state);
   nir_shader_intrinsics_pass(nir, gex_wgpu_gs_emit,
                              nir_metadata_control_flow, &state);
   if (state.failed) {
      ralloc_free(nir);
      return NULL;
   }

   /* One invocation per input primitive; the rest of the workgroup has no
    * primitive to run and must not write. */
   b = nir_builder_at(nir_before_impl(impl));
   nir_deref_instr *params = gex_wgpu_block_words(&b, state.params);
   nir_def *count = nir_load_deref(
      &b, nir_build_deref_array(&b, params,
                                nir_imm_int(&b, GEX_WGPU_GS_PARAM_PRIMITIVES)));
   nir_push_if(&b, nir_uge(&b, gex_wgpu_gs_invocation(&b), count));
   nir_jump(&b, nir_jump_return);
   nir_pop_if(&b, NULL);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_returns);

   /* A compute shader has no shader inputs or outputs. The outputs are still
    * read at each emit, so they become ordinary private variables rather than
    * being deleted. */
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_out)
      var->data.mode = nir_var_shader_temp;
   nir_fixup_deref_modes(nir);
   nir_remove_dead_variables(nir, nir_var_shader_in | nir_var_system_value,
                             NULL);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_dce);

   nir->info.stage = MESA_SHADER_COMPUTE;
   nir->info.workgroup_size[0] = GEX_WGPU_GS_GROUP;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
   nir->xfb_info = NULL;

   char *error = NULL;
   WGPUShaderModule module =
      gex_wgpu_compile_nir(pipe, nir, MESA_SHADER_COMPUTE, false, &error);
   if (!module)
      fprintf(stderr, "gex-wgpu: geometry shader rejected: %s\n",
              error ? error : "unknown");
   free(error);
   return module;
}

static void *
gex_wgpu_create_gs_state(struct pipe_context *pipe,
                         const struct pipe_shader_state *state)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   struct pipe_shader_state *mutable_state = (struct pipe_shader_state *)state;
   nir_shader *nir = state->type == PIPE_SHADER_IR_NIR ? state->ir.nir : NULL;
   if (!nir || nir->info.stage != MESA_SHADER_GEOMETRY) {
      gex_wgpu_set_compile_error(mutable_state,
                                 "gex_wgpu requires geometry NIR");
      if (nir)
         ralloc_free(nir);
      return NULL;
   }

   /* What this path implements, stated once. Anything else is refused here
    * rather than mis-rendered later. */
   const char *refusal = NULL;
   const bool points_out = nir->info.gs.output_primitive == MESA_PRIM_POINTS;
   const bool points_in = nir->info.gs.input_primitive == MESA_PRIM_POINTS;
   if (!points_out && nir->info.gs.output_primitive != MESA_PRIM_TRIANGLE_STRIP)
      refusal = "gex_wgpu geometry shaders must output points or a triangle strip";
   else if (!points_in && nir->info.gs.input_primitive != MESA_PRIM_TRIANGLES)
      refusal = "gex_wgpu geometry shaders must take points or triangles";
   else if (points_in != points_out)
      refusal = "gex_wgpu geometry shaders cannot change between points and triangles";
   else if (nir->info.gs.invocations > 1)
      refusal = "gex_wgpu does not implement geometry shader instancing";
   else if (nir->info.gs.active_stream_mask & ~1u)
      refusal = "gex_wgpu geometry shaders may only write stream zero";
   /* A triangle strip needs three vertices before it closes anything; a point
    * stream needs one. Requiring three of both refused the smallest useful
    * geometry shader there is -- `max_vertices = 1` emitting a single point,
    * which is what the transform-feedback tests use. */
   else if (nir->info.gs.vertices_out < (points_out ? 1u : 3u))
      refusal = points_out
         ? "gex_wgpu geometry shaders must be able to emit a point"
         : "gex_wgpu geometry shaders must be able to emit a triangle";
   if (refusal) {
      gex_wgpu_set_compile_error(mutable_state, refusal);
      ralloc_free(nir);
      return NULL;
   }

   struct gex_wgpu_gs *gs = CALLOC_STRUCT(gex_wgpu_gs);
   if (!gs) {
      ralloc_free(nir);
      return NULL;
   }
   gs->max_vertices = nir->info.gs.vertices_out;
   gs->output_prim = nir->info.gs.output_primitive;
   if (gs->output_prim == MESA_PRIM_POINTS) {
      gs->max_output_prims = gs->max_vertices;
      gs->indices_per_prim = 1;
      gs->verts_per_input_prim = 1;
   } else {
      gs->max_output_prims = gs->max_vertices - 2;
      gs->indices_per_prim = 3;
      gs->verts_per_input_prim = 3;
   }
   gs->max_triangles = gs->max_output_prims;

   if (!gex_wgpu_lower_gallium_ubos(
          nir, MESA_SHADER_GEOMETRY,
          gex_screen(pipe->screen)->max_uniform_buffers, NULL, 0) ||
       !gex_wgpu_collect_varyings(nir, nir_var_shader_in, &gs->inputs) ||
       !gex_wgpu_collect_varyings(nir, nir_var_shader_out, &gs->outputs)) {
      gex_wgpu_set_compile_error(
         mutable_state, "gex_wgpu could not lay out geometry shader varyings");
      ralloc_free(nir);
      FREE(gs);
      return NULL;
   }
   nir_foreach_variable_with_modes(var, nir,
                                   nir_var_mem_ubo | nir_var_mem_ssbo) {
      if (var->data.descriptor_set != 2 ||
          gs->ubo_count >= GEX_WGPU_MAX_CONSTANT_BUFFERS)
         continue;
      gs->ubo_slots[gs->ubo_count] = var->data.driver_location;
      gs->ubo_bindings[gs->ubo_count] = var->data.binding;
      if (var->data.mode == nir_var_mem_ssbo)
         gs->ubo_storage_mask |= 1u << gs->ubo_count;
      ++gs->ubo_count;
   }

   NIR_PASS(_, nir, nir_lower_gs_intrinsics, 0);
   gs->nir = nir;
   gs->module = gex_wgpu_gs_build_module(pipe, gs);

   struct pipe_shader_state passthrough_state = {0};
   passthrough_state.type = PIPE_SHADER_IR_NIR;
   passthrough_state.ir.nir = gex_wgpu_gs_passthrough_nir(pipe, &gs->outputs);
   passthrough_state.report_compile_error = true;
   gs->passthrough = gex_wgpu_create_shader_state(pipe, &passthrough_state,
                                                  MESA_SHADER_VERTEX);
   free(passthrough_state.error_message);

   struct pipe_vertex_element elements[GEX_WGPU_MAX_VARYINGS];
   memset(elements, 0, sizeof(elements));
   for (unsigned i = 0; i < gs->outputs.count; ++i) {
      elements[i].src_offset = i * 16;
      elements[i].src_stride = gex_wgpu_varying_stride(&gs->outputs);
      elements[i].src_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      elements[i].vertex_buffer_index = 0;
   }
   gs->passthrough_elements = ctx->base.create_vertex_elements_state(
      &ctx->base, gs->outputs.count, elements);

   if (!gs->module || !gs->passthrough || !gs->passthrough_elements) {
      /* Named, because all three failures reach the application as an empty
       * link log and there is nothing else to go on. */
      gex_wgpu_set_compile_error(
         mutable_state,
         !gs->module
            ? "gex_wgpu could not compile the geometry compute module"
            : !gs->passthrough
            ? "gex_wgpu could not build the geometry passthrough vertex shader"
            : "gex_wgpu could not build the geometry passthrough vertex layout");
      if (gs->module)
         wgpuShaderModuleRelease(gs->module);
      if (gs->passthrough)
         gex_wgpu_delete_shader_state(pipe, gs->passthrough);
      ralloc_free(nir);
      FREE(gs);
      return NULL;
   }
   return gs;
}

static void
gex_wgpu_bind_gs_state(struct pipe_context *pipe, void *state)
{
   ((struct gex_wgpu_context *)pipe)->gs = state;
}

static void
gex_wgpu_delete_gs_state(struct pipe_context *pipe, void *state)
{
   struct gex_wgpu_gs *gs = state;
   if (!gs)
      return;
   if (gs->module)
      wgpuShaderModuleRelease(gs->module);
   if (gs->passthrough)
      gex_wgpu_delete_shader_state(pipe, gs->passthrough);
   if (gs->passthrough_elements)
      pipe->delete_vertex_elements_state(pipe, gs->passthrough_elements);
   if (gs->nir)
      ralloc_free(gs->nir);
   FREE(gs);
}

/* Run the capture. Returns false having already reported the refusal. */
/* Which vertex of the draw a decomposed primitive's Nth vertex comes from.
 *
 * Transform feedback captures assembled primitives, in primitive order: a
 * four-vertex triangle strip captures two triangles -- six vertices -- and a
 * line loop captures a closing line whose second vertex the draw never names
 * again. The compute pass that stands in for the vertex stage has no
 * fixed-function assembly, so the whole mapping is built on the CPU and read
 * as a table. UINT_MAX means a topology this path does not decompose.
 */
static unsigned
gex_wgpu_assembled_vertex(enum mesa_prim mode, unsigned prim, unsigned vertex,
                          unsigned count)
{
   switch (mode) {
   case MESA_PRIM_POINTS:
      return prim;
   case MESA_PRIM_LINES:
      return prim * 2 + vertex;
   case MESA_PRIM_LINE_STRIP:
      return prim + vertex;
   case MESA_PRIM_LINE_LOOP:
      return count ? (prim + vertex) % count : 0;
   case MESA_PRIM_TRIANGLES:
      return prim * 3 + vertex;
   case MESA_PRIM_TRIANGLE_STRIP:
      /* Odd strip triangles swap their last two vertices so the winding
       * stays consistent, and transform feedback records that order. */
      if (prim & 1)
         return prim + (vertex == 0 ? 0 : vertex == 1 ? 2 : 1);
      return prim + vertex;
   case MESA_PRIM_TRIANGLE_FAN:
      return vertex == 0 ? 0 : prim + vertex;
   default:
      return UINT_MAX;
   }
}

static bool
gex_wgpu_capture_dispatch(struct gex_wgpu_context *ctx,
                          const struct pipe_draw_info *info,
                          const struct pipe_draw_start_count_bias *draw)
{
   GEX_DEVICE_GUARD(gex_screen(ctx->base.screen));
   /* Submits its own command buffer, so anything already recorded has to
    * reach the queue first or it would run after this. */
   gex_wgpu_submit_pending(ctx);
   struct gex_wgpu_screen *screen = gex_screen(ctx->base.screen);

   /* An instanced capture still needs the invocation index to reach the
    * instance, which this variant does not model. */
   if (info->instance_count != 1) {
      gex_fail(ctx);
      return false;
   }

   /* The slot table: one entry per captured vertex, resolved through the
    * index buffer and through the primitive topology. */
   const unsigned verts_per_prim = mesa_vertices_per_prim(info->mode);
   const unsigned prims = u_prims_for_vertices(info->mode, draw->count);
   const uint64_t captured = (uint64_t)prims * verts_per_prim;
   if (!prims || !verts_per_prim || verts_per_prim > 3 ||
       captured > (1u << 22) ||
       gex_wgpu_assembled_vertex(info->mode, 0, 0, draw->count) == UINT_MAX) {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: capture declined (topology %u, %u vertices)\n",
                 info->mode, draw->count);
      }
      return false;
   }
   const uint8_t *indices = NULL;
   if (info->index_size) {
      indices = info->has_user_indices
         ? (const uint8_t *)info->index.user
         : (info->index.resource
               ? gex_resource(info->index.resource)->shadow : NULL);
      if (!indices ||
          (info->index_size != 1 && info->index_size != 2 &&
           info->index_size != 4)) {
         gex_fail(ctx);
         return false;
      }
      indices += (uint64_t)draw->start * info->index_size;
   }
   uint32_t *slot_table = MALLOC((size_t)captured * sizeof(uint32_t));
   if (!slot_table) {
      gex_fail(ctx);
      return false;
   }
   for (unsigned prim = 0; prim < prims; ++prim) {
      for (unsigned v = 0; v < verts_per_prim; ++v) {
         const unsigned at =
            gex_wgpu_assembled_vertex(info->mode, prim, v, draw->count);
         uint32_t vertex;
         if (!info->index_size) {
            vertex = draw->start + at;
         } else if (info->index_size == 1) {
            vertex = indices[at];
         } else if (info->index_size == 2) {
            uint16_t narrow;
            memcpy(&narrow, indices + (size_t)at * 2, 2);
            vertex = narrow;
         } else {
            memcpy(&vertex, indices + (size_t)at * 4, 4);
         }
         if (info->index_size)
            vertex += (uint32_t)draw->index_bias;
         slot_table[prim * verts_per_prim + v] = vertex;
      }
   }

   struct gex_wgpu_capture_layout layout = {0};
   layout.index_table = 1;
   if (ctx->vertex_elements) {
      layout.count = ctx->vertex_elements->count;
      for (unsigned i = 0; i < layout.count; ++i) {
         const struct pipe_vertex_element *ve =
            &ctx->vertex_elements->elements[i];
         layout.buffer[i] = ve->vertex_buffer_index;
         layout.offset[i] = ve->src_offset;
         layout.stride[i] = ve->src_stride;
         layout.format[i] = ve->src_format;
      }
   }

   struct gex_wgpu_capture_variant *variant =
      gex_wgpu_capture_variant_get(ctx, ctx->vs, &layout);
   /* The slot table, as a storage buffer. Created and released here rather
    * than cached: it is one draw's worth of assembly and it is written once. */
   WGPUBufferDescriptor table_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   table_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
   table_desc.size = (uint64_t)captured * sizeof(uint32_t);
   WGPUBuffer table_buffer = variant
      ? wgpuDeviceCreateBuffer(screen->device, &table_desc) : NULL;
   if (table_buffer)
      wgpuQueueWriteBuffer(screen->queue, table_buffer, 0, slot_table,
                           (size_t)table_desc.size);
   FREE(slot_table);
   if (!variant || !table_buffer) {
      if (table_buffer)
         wgpuBufferRelease(table_buffer);
      gex_fail(ctx);
      return false;
   }

   uint32_t params[GEX_WGPU_PARAM_WORDS] = {0};
   /* The table is indexed by the invocation, so there is no first vertex to
    * add, and the count is the number of assembled vertices rather than the
    * number the draw named. */
   params[GEX_WGPU_PARAM_FIRST_VERTEX] = 0u;
   /* Filled in after the targets are known: how many assembled vertices fit. */
   const nir_xfb_info *xfb =
      ctx->vs->capture_nir ? ctx->vs->capture_nir->xfb_info : NULL;
   unsigned writable_prims = prims;

   WGPUBindGroupLayoutEntry entries[GEX_WGPU_MAX_SO_BUFFERS +
                                   GEX_WGPU_MAX_VERTEX_BUFFERS + 2];
   WGPUBindGroupEntry bindings[GEX_WGPU_MAX_SO_BUFFERS +
                               GEX_WGPU_MAX_VERTEX_BUFFERS + 2];
   unsigned count = 0;

   for (unsigned i = 0; i < GEX_WGPU_MAX_SO_BUFFERS; ++i) {
      if (!(variant->so_mask & (1u << i)))
         continue;
      const struct pipe_stream_output_target *target = ctx->so_targets[i];
      /* A shader that captures needs its targets bound; without them the
       * dispatch would run and write nowhere. */
      if (!target || !target->buffer) {
         gex_fail(ctx);
         return false;
      }
      /* Bound whole and offset in the parameters, because a storage binding
       * offset must be 256-byte aligned and GL promises only four. */
      struct gex_wgpu_so_target *so = (struct gex_wgpu_so_target *)target;
      params[GEX_WGPU_PARAM_SO_OFFSET + i] = target->buffer_offset + so->written;
      /* The dispatch below writes it, so a later map has to read it back. */
      gex_resource(target->buffer)->gpu_written = true;
      /* GL stops capturing when a bound buffer would overflow, and it stops
       * on a whole primitive: a buffer with room for three vertices records
       * one triangle, not one and two thirds. The count the queries report is
       * what was written, not what was generated, and the CTS's query and
       * discard cases size their buffers to make the difference visible. */
      if (xfb && (xfb->buffers_written & (1u << i))) {
         const unsigned stride = xfb->buffers[i].stride;
         /* Room is what is left after an append, not the whole buffer. */
         const unsigned left = target->buffer_size > so->written
                                  ? (unsigned)(target->buffer_size - so->written)
                                  : 0u;
         const unsigned room = stride ? left / (stride * verts_per_prim) : 0u;
         writable_prims = MIN2(writable_prims, room);
      }
      entries[count] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      entries[count].binding = variant->so_bindings[i];
      entries[count].visibility = WGPUShaderStage_Compute;
      entries[count].buffer.type = WGPUBufferBindingType_Storage;
      bindings[count] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
      bindings[count].binding = variant->so_bindings[i];
      bindings[count].buffer = gex_resource(target->buffer)->buffer;
      bindings[count].offset = 0;
      bindings[count].size = gex_resource(target->buffer)->allocation_size;
      ++count;
   }

   for (unsigned i = 0; i < GEX_WGPU_MAX_VERTEX_BUFFERS; ++i) {
      if (!(variant->vb_mask & (1u << i)))
         continue;
      if (i >= ctx->vertex_buffer_count || !ctx->vertex_buffers[i]) {
         gex_fail(ctx);
         return false;
      }
      params[GEX_WGPU_PARAM_VB_OFFSET + i] = ctx->vertex_buffer_offsets[i];
      entries[count] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      entries[count].binding = variant->vb_bindings[i];
      entries[count].visibility = WGPUShaderStage_Compute;
      /* Writable, though the shader only reads it: the emitter declares every
       * storage buffer read_write and a bind group layout that says otherwise
       * is rejected as incompatible with the shader's own declaration. */
      entries[count].buffer.type = WGPUBufferBindingType_Storage;
      bindings[count] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
      bindings[count].binding = variant->vb_bindings[i];
      bindings[count].buffer = gex_resource(ctx->vertex_buffers[i])->buffer;
      bindings[count].offset = 0;
      bindings[count].size =
         gex_resource(ctx->vertex_buffers[i])->allocation_size;
      ++count;
   }

   const uint32_t written = writable_prims * verts_per_prim;
   params[GEX_WGPU_PARAM_COUNT] = written;
   /* Now that the count is known, move each target on by what this capture
    * will record, so a resumed or appended capture starts after it and
    * DrawTransformFeedback can read the total back. */
   for (unsigned i = 0; xfb && i < GEX_WGPU_MAX_SO_BUFFERS; ++i) {
      if (!(variant->so_mask & (1u << i)) ||
          !(xfb->buffers_written & (1u << i)))
         continue;
      struct gex_wgpu_so_target *so =
         (struct gex_wgpu_so_target *)ctx->so_targets[i];
      if (!so)
         continue;
      so->stride = xfb->buffers[i].stride;
      so->written += written * so->stride;
   }
   /* PRIMITIVES_GENERATED counts what the vertex stage produced;
    * PRIMITIVES_EMITTED counts what transform feedback recorded. */
   if (ctx->active_query) {
      switch (ctx->active_query->type) {
      case PIPE_QUERY_PRIMITIVES_EMITTED:
      case PIPE_QUERY_SO_STATISTICS:
         ctx->active_query->result += writable_prims;
         break;
      default:
         break;
      }
   }

   WGPUBufferDescriptor params_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   params_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
   params_desc.size = sizeof(params);
   WGPUBuffer params_buffer =
      wgpuDeviceCreateBuffer(screen->device, &params_desc);
   if (!params_buffer) {
      gex_fail(ctx);
      return false;
   }
   wgpuQueueWriteBuffer(screen->queue, params_buffer, 0, params,
                        sizeof(params));
   entries[count] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
   entries[count].binding = GEX_WGPU_CAPTURE_PARAMS_BINDING;
   entries[count].visibility = WGPUShaderStage_Compute;
   entries[count].buffer.type = WGPUBufferBindingType_Storage;
   bindings[count] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
   bindings[count].binding = GEX_WGPU_CAPTURE_PARAMS_BINDING;
   bindings[count].buffer = params_buffer;
   bindings[count].offset = 0;
   bindings[count].size = sizeof(params);
   ++count;

   entries[count] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
   entries[count].binding = GEX_WGPU_CAPTURE_INDEX_BINDING;
   entries[count].visibility = WGPUShaderStage_Compute;
   entries[count].buffer.type = WGPUBufferBindingType_Storage;
   bindings[count] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
   bindings[count].binding = GEX_WGPU_CAPTURE_INDEX_BINDING;
   bindings[count].buffer = table_buffer;
   bindings[count].offset = 0;
   bindings[count].size = table_desc.size;
   ++count;

   /* The variant is a clone of the vertex shader, so it kept that shader's
    * constant buffers, which live in their own descriptor set -- group 1 for
    * a vertex stage. Without them a capturing shader that reads any uniform
    * would reference a group the dispatch never bound. */
   WGPUBindGroupLayoutEntry ubo_entries[GEX_WGPU_MAX_CONSTANT_BUFFERS];
   WGPUBindGroupEntry ubo_bindings[GEX_WGPU_MAX_CONSTANT_BUFFERS];
   const unsigned ubo_count = ctx->vs->ubo_count;
   for (unsigned i = 0; i < ubo_count; ++i) {
      const unsigned slot = ctx->vs->ubo_slots[i];
      const struct pipe_constant_buffer *constant =
         &ctx->constant_buffers[MESA_SHADER_VERTEX][slot];
      if (!constant->buffer) {
         gex_fail(ctx);
         return false;
      }
      ubo_entries[i] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      ubo_entries[i].binding = ctx->vs->ubo_bindings[i];
      ubo_entries[i].visibility = WGPUShaderStage_Compute;
      ubo_entries[i].buffer.type = WGPUBufferBindingType_Uniform;
      ubo_entries[i].buffer.minBindingSize =
         ALIGN_POT(constant->buffer_size, 16);
      ubo_bindings[i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
      ubo_bindings[i].binding = ctx->vs->ubo_bindings[i];
      ubo_bindings[i].buffer = gex_resource(constant->buffer)->buffer;
      ubo_bindings[i].offset = constant->buffer_offset;
      ubo_bindings[i].size = ALIGN_POT(constant->buffer_size, 16);
   }

   WGPUBindGroupLayoutDescriptor layout_desc =
      WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
   layout_desc.entryCount = count;
   layout_desc.entries = entries;
   WGPUBindGroupLayout group_layout =
      wgpuDeviceCreateBindGroupLayout(screen->device, &layout_desc);
   WGPUBindGroupLayoutDescriptor ubo_layout_desc =
      WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
   ubo_layout_desc.entryCount = ubo_count;
   ubo_layout_desc.entries = ubo_entries;
   WGPUBindGroupLayout ubo_layout =
      wgpuDeviceCreateBindGroupLayout(screen->device, &ubo_layout_desc);
   WGPUBindGroupLayout group_layouts[2] = {group_layout, ubo_layout};
   WGPUPipelineLayoutDescriptor pipeline_layout_desc =
      WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
   pipeline_layout_desc.bindGroupLayoutCount = 2;
   pipeline_layout_desc.bindGroupLayouts = group_layouts;
   WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(screen->device, &pipeline_layout_desc);

   WGPUComputePipelineDescriptor pipeline_desc =
      WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
   pipeline_desc.layout = pipeline_layout;
   pipeline_desc.compute.module = variant->module;
   pipeline_desc.compute.entryPoint = (WGPUStringView){"main", WGPU_STRLEN};
   WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(screen->device, &pipeline_desc);

   WGPUBindGroupDescriptor group_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   group_desc.layout = group_layout;
   group_desc.entryCount = count;
   group_desc.entries = bindings;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(screen->device, &group_desc);
   WGPUBindGroupDescriptor ubo_group_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   ubo_group_desc.layout = ubo_layout;
   ubo_group_desc.entryCount = ubo_count;
   ubo_group_desc.entries = ubo_bindings;
   WGPUBindGroup ubo_group =
      wgpuDeviceCreateBindGroup(screen->device, &ubo_group_desc);

   WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   bool ok = pipeline && group && ubo_group && encoder;
   if (ok) {
      WGPUComputePassDescriptor pass_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
      WGPUComputePassEncoder pass =
         wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
      if (pass) {
         wgpuComputePassEncoderSetPipeline(pass, pipeline);
         wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
         wgpuComputePassEncoderSetBindGroup(pass, 1, ubo_group, 0, NULL);
         wgpuComputePassEncoderDispatchWorkgroups(
            pass, DIV_ROUND_UP(written, GEX_WGPU_CAPTURE_GROUP), 1, 1);
         wgpuComputePassEncoderEnd(pass);
         wgpuComputePassEncoderRelease(pass);
      } else {
         ok = false;
      }
      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      if (commands) {
         gex_wgpu_queue_submit(screen, 1, &commands);
         wgpuCommandBufferRelease(commands);
      } else {
         ok = false;
      }
   }
   if (encoder)
      wgpuCommandEncoderRelease(encoder);
   if (group)
      wgpuBindGroupRelease(group);
   if (ubo_group)
      wgpuBindGroupRelease(ubo_group);
   if (pipeline)
      wgpuComputePipelineRelease(pipeline);
   if (pipeline_layout)
      wgpuPipelineLayoutRelease(pipeline_layout);
   if (group_layout)
      wgpuBindGroupLayoutRelease(group_layout);
   if (ubo_layout)
      wgpuBindGroupLayoutRelease(ubo_layout);
   wgpuBufferRelease(params_buffer);
   wgpuBufferRelease(table_buffer);
   if (!ok)
      gex_fail(ctx);
   return ok;
}

static void
gex_wgpu_draw_vbo(struct pipe_context *pipe,
                  const struct pipe_draw_info *info,
                  unsigned drawid_offset,
                  const struct pipe_draw_indirect_info *indirect,
                  const struct pipe_draw_start_count_bias *draws,
                  unsigned num_draws);

/* One bind group's worth of buffers, in the two shapes these compute passes
 * use: group zero is the pass's own buffers, group one is the shader's
 * constant buffers. */
#define GEX_WGPU_MAX_GROUP_BINDINGS (GEX_WGPU_MAX_CONSTANT_BUFFERS + 8)
struct gex_wgpu_bindings {
   unsigned count;
   uint32_t storage_mask;
   uint32_t binding[GEX_WGPU_MAX_GROUP_BINDINGS];
   WGPUBuffer buffer[GEX_WGPU_MAX_GROUP_BINDINGS];
   uint64_t offset[GEX_WGPU_MAX_GROUP_BINDINGS];
   uint64_t size[GEX_WGPU_MAX_GROUP_BINDINGS];
};

static void
gex_wgpu_bind_storage(struct gex_wgpu_bindings *group, uint32_t binding,
                      struct pipe_resource *resource)
{
   if (group->count >= GEX_WGPU_MAX_GROUP_BINDINGS || !resource)
      return;
   const unsigned i = group->count++;
   group->binding[i] = binding;
   group->buffer[i] = gex_resource(resource)->buffer;
   group->offset[i] = 0;
   group->size[i] = gex_resource(resource)->allocation_size;
   group->storage_mask |= 1u << i;
}

static WGPUBindGroupLayout
gex_wgpu_group_layout(WGPUDevice device, const struct gex_wgpu_bindings *group)
{
   WGPUBindGroupLayoutEntry entries[GEX_WGPU_MAX_GROUP_BINDINGS];
   for (unsigned i = 0; i < group->count; ++i) {
      entries[i] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      entries[i].binding = group->binding[i];
      entries[i].visibility = WGPUShaderStage_Compute;
      /* Storage rather than ReadOnlyStorage even where the shader only reads:
       * the emitter declares every storage buffer read_write and WebGPU
       * rejects a layout that disagrees with the shader's declaration. */
      entries[i].buffer.type = (group->storage_mask & (1u << i))
         ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_Uniform;
   }
   WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
   desc.entryCount = group->count;
   desc.entries = entries;
   return wgpuDeviceCreateBindGroupLayout(device, &desc);
}

static WGPUBindGroup
gex_wgpu_group(WGPUDevice device, WGPUBindGroupLayout layout,
               const struct gex_wgpu_bindings *group)
{
   WGPUBindGroupEntry entries[GEX_WGPU_MAX_GROUP_BINDINGS];
   for (unsigned i = 0; i < group->count; ++i) {
      entries[i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
      entries[i].binding = group->binding[i];
      entries[i].buffer = group->buffer[i];
      entries[i].offset = group->offset[i];
      entries[i].size = group->size[i];
   }
   WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   desc.layout = layout;
   desc.entryCount = group->count;
   desc.entries = entries;
   return wgpuDeviceCreateBindGroup(device, &desc);
}

/* Run one compute pass over two bind groups, optionally zeroing a buffer
 * first so the same command encoder carries both. */
static bool
gex_wgpu_dispatch(struct gex_wgpu_context *ctx, WGPUShaderModule module,
                  const struct gex_wgpu_bindings *g0,
                  const struct gex_wgpu_bindings *g1,
                  struct pipe_resource *zero, unsigned workgroups)
{
   GEX_DEVICE_GUARD(gex_screen(ctx->base.screen));
   /* Submits its own command buffer, so anything already recorded has to
    * reach the queue first or it would run after this. */
   gex_wgpu_submit_pending(ctx);
   struct gex_wgpu_screen *screen = gex_screen(ctx->base.screen);
   WGPUBindGroupLayout layout0 = gex_wgpu_group_layout(screen->device, g0);
   WGPUBindGroupLayout layout1 = gex_wgpu_group_layout(screen->device, g1);
   WGPUBindGroupLayout layouts[2] = {layout0, layout1};
   WGPUPipelineLayoutDescriptor pipeline_layout_desc =
      WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
   pipeline_layout_desc.bindGroupLayoutCount = 2;
   pipeline_layout_desc.bindGroupLayouts = layouts;
   WGPUPipelineLayout pipeline_layout =
      wgpuDeviceCreatePipelineLayout(screen->device, &pipeline_layout_desc);

   WGPUComputePipelineDescriptor pipeline_desc =
      WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
   pipeline_desc.layout = pipeline_layout;
   pipeline_desc.compute.module = module;
   pipeline_desc.compute.entryPoint = (WGPUStringView){"main", WGPU_STRLEN};
   WGPUComputePipeline pipeline =
      wgpuDeviceCreateComputePipeline(screen->device, &pipeline_desc);

   WGPUBindGroup group0 = gex_wgpu_group(screen->device, layout0, g0);
   WGPUBindGroup group1 = gex_wgpu_group(screen->device, layout1, g1);
   WGPUCommandEncoder encoder =
      wgpuDeviceCreateCommandEncoder(screen->device, NULL);

   bool ok = pipeline && group0 && group1 && encoder;
   if (ok) {
      if (zero)
         wgpuCommandEncoderClearBuffer(encoder, gex_resource(zero)->buffer, 0,
                                       gex_resource(zero)->allocation_size);
      WGPUComputePassDescriptor pass_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
      WGPUComputePassEncoder pass =
         wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
      if (pass) {
         wgpuComputePassEncoderSetPipeline(pass, pipeline);
         wgpuComputePassEncoderSetBindGroup(pass, 0, group0, 0, NULL);
         wgpuComputePassEncoderSetBindGroup(pass, 1, group1, 0, NULL);
         wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);
         wgpuComputePassEncoderEnd(pass);
         wgpuComputePassEncoderRelease(pass);
      } else {
         ok = false;
      }
      WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
      if (commands) {
         gex_wgpu_queue_submit(screen, 1, &commands);
         wgpuCommandBufferRelease(commands);
      } else {
         ok = false;
      }
   }
   if (encoder)
      wgpuCommandEncoderRelease(encoder);
   if (group0)
      wgpuBindGroupRelease(group0);
   if (group1)
      wgpuBindGroupRelease(group1);
   if (pipeline)
      wgpuComputePipelineRelease(pipeline);
   if (pipeline_layout)
      wgpuPipelineLayoutRelease(pipeline_layout);
   if (layout0)
      wgpuBindGroupLayoutRelease(layout0);
   if (layout1)
      wgpuBindGroupLayoutRelease(layout1);
   return ok;
}

/* A small storage buffer holding the parameter words, valid for one dispatch. */
static struct pipe_resource *
gex_wgpu_params_buffer(struct gex_wgpu_context *ctx, const uint32_t *words,
                       unsigned count)
{
   struct pipe_resource templ;
   memset(&templ, 0, sizeof(templ));
   templ.target = PIPE_BUFFER;
   templ.format = PIPE_FORMAT_R8_UNORM;
   templ.width0 = count * 4;
   templ.height0 = 1;
   templ.depth0 = 1;
   templ.array_size = 1;
   templ.bind = PIPE_BIND_SHADER_BUFFER;
   struct pipe_resource *resource =
      ctx->base.screen->resource_create(ctx->base.screen, &templ);
   if (resource)
      wgpuQueueWriteBuffer(gex_screen(ctx->base.screen)->queue,
                           gex_resource(resource)->buffer, 0, words, count * 4);
   return resource;
}

static bool
gex_wgpu_ensure_buffer(struct gex_wgpu_context *ctx,
                       struct pipe_resource **slot, unsigned size,
                       unsigned bind)
{
   if (*slot && gex_resource(*slot)->allocation_size >= size)
      return true;
   pipe_resource_reference(slot, NULL);
   struct pipe_resource templ;
   memset(&templ, 0, sizeof(templ));
   templ.target = PIPE_BUFFER;
   templ.format = PIPE_FORMAT_R8_UNORM;
   templ.width0 = size;
   templ.height0 = 1;
   templ.depth0 = 1;
   templ.array_size = 1;
   templ.bind = bind;
   *slot = ctx->base.screen->resource_create(ctx->base.screen, &templ);
   return *slot != NULL;
}

/* Add a shader's constant buffers to the group-one bindings of a compute
 * pass, in whichever address space each slot uses. */
static bool
gex_wgpu_bind_constants(struct gex_wgpu_context *ctx,
                        struct gex_wgpu_bindings *group,
                        mesa_shader_stage stage, unsigned ubo_count,
                        const uint8_t *ubo_slots, const uint32_t *ubo_bindings,
                        uint32_t storage_mask)
{
   for (unsigned i = 0; i < ubo_count; ++i) {
      const struct pipe_constant_buffer *constant =
         &ctx->constant_buffers[stage][ubo_slots[i]];
      if (!constant->buffer || group->count >= GEX_WGPU_MAX_GROUP_BINDINGS)
         return false;
      const unsigned at = group->count++;
      group->binding[at] = ubo_bindings[i];
      group->buffer[at] = gex_resource(constant->buffer)->buffer;
      group->offset[at] = constant->buffer_offset;
      group->size[at] = constant->buffer_size;
      if (storage_mask & (1u << i))
         group->storage_mask |= 1u << at;
   }
   return true;
}

/* The three-pass geometry path: vertex shader to a buffer, geometry shader to
 * vertices and indices, then an ordinary indexed draw of the result. */
static bool
gex_wgpu_gs_draw(struct gex_wgpu_context *ctx,
                 const struct pipe_draw_info *info, unsigned drawid_offset,
                 const struct pipe_draw_start_count_bias *draw)
{
   struct gex_wgpu_gs *gs = ctx->gs;
   const enum mesa_prim wanted_mode =
      gs->verts_per_input_prim == 1 ? MESA_PRIM_POINTS : MESA_PRIM_TRIANGLES;
   /* Declined, not latched. What this path cannot do it cannot do for this
    * draw only; latching turns it into GL_OUT_OF_MEMORY from some later
    * allocation, which dEQP treats as fatal and which cost ten of the
    * eleven failing KHR-GL32.transform_feedback cases. */
#define GEX_GS_DECLINE(why)                                                   \
   do {                                                                       \
      static bool said;                                                       \
      if (!said) {                                                            \
         said = true;                                                          \
         fprintf(stderr, "gex-wgpu: geometry draw declined (%s)\n", why);      \
      }                                                                       \
      return false;                                                           \
   } while (0)
   if (info->index_size || info->instance_count != 1 ||
       info->mode != wanted_mode ||
       draw->count % gs->verts_per_input_prim != 0)
      GEX_GS_DECLINE("indexed, instanced, or the wrong input primitive");
   const unsigned prims = draw->count / gs->verts_per_input_prim;
   /* A geometry shader that reads no varying from the vertex stage is
    * ordinary -- the CTS's transform feedback shaders are exactly that -- so
    * a zero input stride is not a failure. The buffer still needs a size,
    * because a storage binding does. */
   const unsigned vs_stride =
      MAX2(gex_wgpu_varying_stride(&gs->inputs), 16u);
   const unsigned gs_stride = gex_wgpu_varying_stride(&gs->outputs);
   const unsigned index_count =
      prims * gs->max_output_prims * gs->indices_per_prim;
   if (!prims || !index_count || !gs_stride)
      GEX_GS_DECLINE("nothing to draw, or a geometry shader with no outputs");

   struct gex_wgpu_capture_layout layout;
   memset(&layout, 0, sizeof(layout));
   if (ctx->vertex_elements) {
      layout.count = ctx->vertex_elements->count;
      for (unsigned i = 0; i < layout.count; ++i) {
         const struct pipe_vertex_element *ve =
            &ctx->vertex_elements->elements[i];
         layout.buffer[i] = ve->vertex_buffer_index;
         layout.offset[i] = ve->src_offset;
         layout.stride[i] = ve->src_stride;
         layout.format[i] = ve->src_format;
      }
   }
   struct gex_wgpu_varying_variant *variant =
      gex_wgpu_varying_variant_get(ctx, ctx->vs, &layout, &gs->inputs);
   if (!variant ||
       !gex_wgpu_ensure_buffer(ctx, &ctx->gs_vs_out, draw->count * vs_stride,
                               PIPE_BIND_SHADER_BUFFER) ||
       !gex_wgpu_ensure_buffer(ctx, &ctx->gs_vertices,
                               prims * gs->max_vertices * gs_stride,
                               PIPE_BIND_VERTEX_BUFFER |
                               PIPE_BIND_SHADER_BUFFER) ||
       !gex_wgpu_ensure_buffer(ctx, &ctx->gs_indices, index_count * 4,
                               PIPE_BIND_INDEX_BUFFER |
                               PIPE_BIND_SHADER_BUFFER))
      GEX_GS_DECLINE("could not build the variant or its buffers");

   /* Pass one: the vertex shader, as compute, writing its varyings. */
   uint32_t params[GEX_WGPU_PARAM_WORDS] = {0};
   params[GEX_WGPU_PARAM_FIRST_VERTEX] = draw->start;
   params[GEX_WGPU_PARAM_COUNT] = draw->count;
   struct gex_wgpu_bindings vs_group0 = {0}, vs_group1 = {0};
   for (unsigned i = 0; i < GEX_WGPU_MAX_VERTEX_BUFFERS; ++i) {
      if (!(variant->vb_mask & (1u << i)))
         continue;
      if (i >= ctx->vertex_buffer_count || !ctx->vertex_buffers[i])
         GEX_GS_DECLINE("a vertex buffer the geometry pass reads is unbound");
      params[GEX_WGPU_PARAM_VB_OFFSET + i] = ctx->vertex_buffer_offsets[i];
      gex_wgpu_bind_storage(&vs_group0, GEX_WGPU_VB_FIRST_BINDING + i,
                            ctx->vertex_buffers[i]);
   }
   struct pipe_resource *vs_params =
      gex_wgpu_params_buffer(ctx, params, GEX_WGPU_PARAM_WORDS);
   gex_wgpu_bind_storage(&vs_group0, GEX_WGPU_VARYING_OUT_BINDING,
                         ctx->gs_vs_out);
   gex_wgpu_bind_storage(&vs_group0, GEX_WGPU_CAPTURE_PARAMS_BINDING,
                         vs_params);
   bool ok = vs_params &&
             gex_wgpu_bind_constants(ctx, &vs_group1, MESA_SHADER_VERTEX,
                                     ctx->vs->ubo_count, ctx->vs->ubo_slots,
                                     ctx->vs->ubo_bindings,
                                     ctx->vs->ubo_storage_mask);
   if (ok)
      ok = gex_wgpu_dispatch(ctx, variant->module, &vs_group0, &vs_group1,
                             NULL,
                             DIV_ROUND_UP(draw->count,
                                          GEX_WGPU_CAPTURE_GROUP));
   pipe_resource_reference(&vs_params, NULL);
   if (!ok) {
      gex_fail(ctx);
      return false;
   }

   /* Pass two: the geometry shader, as compute, one invocation per input
    * primitive. The index buffer is zeroed first, so a slot no invocation
    * writes holds three equal indices -- a degenerate triangle. */
   uint32_t gs_params[GEX_WGPU_GS_PARAM_WORDS] = {0};
   gs_params[GEX_WGPU_GS_PARAM_PRIMITIVES] = prims;
   struct gex_wgpu_bindings gs_group0 = {0}, gs_group1 = {0};
   struct pipe_resource *gs_param_buffer =
      gex_wgpu_params_buffer(ctx, gs_params, GEX_WGPU_GS_PARAM_WORDS);
   gex_wgpu_bind_storage(&gs_group0, GEX_WGPU_GS_PARAMS_BINDING,
                         gs_param_buffer);
   gex_wgpu_bind_storage(&gs_group0, GEX_WGPU_GS_VSOUT_BINDING, ctx->gs_vs_out);
   gex_wgpu_bind_storage(&gs_group0, GEX_WGPU_GS_VERTEX_BINDING,
                         ctx->gs_vertices);
   gex_wgpu_bind_storage(&gs_group0, GEX_WGPU_GS_INDEX_BINDING,
                         ctx->gs_indices);
   ok = gs_param_buffer &&
        gex_wgpu_bind_constants(ctx, &gs_group1, MESA_SHADER_GEOMETRY,
                                gs->ubo_count, gs->ubo_slots, gs->ubo_bindings,
                                gs->ubo_storage_mask);
   if (ok)
      ok = gex_wgpu_dispatch(ctx, gs->module, &gs_group0, &gs_group1,
                             ctx->gs_indices,
                             DIV_ROUND_UP(prims, GEX_WGPU_GS_GROUP));
   pipe_resource_reference(&gs_param_buffer, NULL);
   if (!ok) {
      gex_fail(ctx);
      return false;
   }

   /* Pass three: rasterize what the geometry shader wrote. The passthrough
    * vertex shader and its element state stand in for the application's for
    * the length of this draw. */
   struct gex_wgpu_shader *saved_vs = ctx->vs;
   struct gex_wgpu_vertex_elements *saved_elements = ctx->vertex_elements;
   struct pipe_resource *saved_buffer = ctx->vertex_buffers[0];
   const uint32_t saved_offset = ctx->vertex_buffer_offsets[0];
   const unsigned saved_count = ctx->vertex_buffer_count;

   ctx->vs = gs->passthrough;
   ctx->vertex_elements = gs->passthrough_elements;
   ctx->vertex_buffers[0] = ctx->gs_vertices;
   ctx->vertex_buffer_offsets[0] = 0;
   ctx->vertex_buffer_count = 1;
   ctx->gs = NULL;

   struct pipe_draw_info raster = *info;
   /* The topology is the geometry shader's output, not the application's
    * input: a points-out shader fed by a points draw still has to rasterize
    * as points, and inheriting info->mode would have rasterized whatever the
    * application happened to be drawing. */
   raster.mode = gs->output_prim == MESA_PRIM_POINTS ? MESA_PRIM_POINTS
                                                     : MESA_PRIM_TRIANGLES;
   raster.index_size = 4;
   raster.index.resource = ctx->gs_indices;
   raster.has_user_indices = false;
   raster.primitive_restart = false;
   raster.index_bounds_valid = false;
   struct pipe_draw_start_count_bias raster_draw;
   memset(&raster_draw, 0, sizeof(raster_draw));
   raster_draw.count = index_count;
   gex_wgpu_draw_vbo(&ctx->base, &raster, drawid_offset, NULL, &raster_draw, 1);

   ctx->gs = gs;
   ctx->vs = saved_vs;
   ctx->vertex_elements = saved_elements;
   ctx->vertex_buffers[0] = saved_buffer;
   ctx->vertex_buffer_offsets[0] = saved_offset;
   ctx->vertex_buffer_count = saved_count;
   return !ctx->failed;
}

/* A vertex module whose output interpolation matches the bound fragment
 * shader's inputs.
 *
 * WGSL requires a varying's interpolation to be declared identically on both
 * sides of the pipeline and rejects the module otherwise. GLSL 1.50 requires
 * no such thing: the qualifier is meaningful only on the fragment input, and
 * a vertex shader that says nothing about a varying the fragment shader
 * declares `flat` or `noperspective` is ordinary and correct GL. Two families
 * failed on exactly that -- glsl_noperspective and primitive_restart -- and
 * they failed as WebGPU validation errors, which poison the device rather
 * than failing one draw.
 *
 * The reconciliation has to happen here rather than at compile time because
 * the two shaders are compiled separately and either may be rebound. The
 * variant is keyed on the fragment stage's whole interpolation signature and
 * carries both y-flip forms, because the caller picks between those after.
 */
static WGPUShaderModule
gex_wgpu_interp_variant_get(struct gex_wgpu_context *ctx, bool flipped,
                            const float viewport[4])
{
   struct gex_wgpu_shader *vs = ctx->vs;
   struct gex_wgpu_shader *fs = ctx->fs;
   WGPUShaderModule plain = flipped ? vs->module_y_flipped : vs->module;
   const bool fold_viewport = viewport != NULL;
   /* A vertex format WebGPU cannot fetch is delivered as a raw dword and
    * unpacked in the shader, so the module depends on the bound vertex
    * elements as well as on the fragment stage. `plain` is not a legal answer
    * when this is set: it would read the dword as if it were the final
    * value. */
   static const uint8_t no_unpack[GEX_WGPU_MAX_VERTEX_ELEMENTS] = {0};
   const struct gex_wgpu_vertex_elements *ve = ctx->vertex_elements;
   const bool unpack_attribs = ve && ve->any_unpack;
   const uint8_t *want_unpack = unpack_attribs ? ve->unpack : no_unpack;
   /* A vertex shader that samples an integer texture has no base module: its
    * sampling is lowered here against the bound sampler state, the way the
    * fragment stage does it. `plain` is NULL for those, so it is never a
    * legal answer. */
   /* The same conditions the fragment stage uses, not just integer sampling.
    *
    * This began as `integer_sampled_slots != 0`, which is the case that
    * cannot compile at all -- but a float sampler2D with a non-identity view
    * swizzle compiles fine and comes out *wrong*, because nothing applies the
    * swizzle. KHR-GL33.texture_swizzle's functional matrix samples in the
    * vertex shader (`result = texelFetch(sampler, ivec2(0,0), 0).r` with the
    * position from gl_VertexID), so 225 of its cases read the source's red
    * channel unswizzled. Silently wrong is worse than refused, and it is the
    * reason every fragment-stage reduction of that failure passed. */
   uint16_t sampler_mask = 0;
   bool lower_textures = false;
   for (unsigned i = 0; i < vs->texture_count; ++i) {
      const unsigned slot = vs->texture_slots[i];
      const struct gex_wgpu_sampler *sampler = ctx->vertex_samplers[slot];
      const struct pipe_sampler_view *view = ctx->vertex_sampler_views[slot];
      bool uses_sampler = false;
      for (unsigned j = 0; j < vs->sampler_count; ++j)
         uses_sampler |= vs->sampler_slots[j] == slot;
      if (!view || (uses_sampler && !sampler)) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr, "gex-wgpu: vertex shader declined (view or "
                            "sampler missing for a sampled slot)\n");
         }
         return NULL;
      }
      sampler_mask |= 1u << slot;
      const bool swizzled =
         gex_wgpu_effective_view_swizzle(view, 0) != PIPE_SWIZZLE_X ||
         gex_wgpu_effective_view_swizzle(view, 1) != PIPE_SWIZZLE_Y ||
         gex_wgpu_effective_view_swizzle(view, 2) != PIPE_SWIZZLE_Z ||
         gex_wgpu_effective_view_swizzle(view, 3) != PIPE_SWIZZLE_W;
      lower_textures |= (vs->integer_sampled_slots & (1u << slot)) != 0 ||
                        swizzled;
      if (uses_sampler)
         lower_textures |=
            sampler->state.wrap_s == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
            sampler->state.wrap_t == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
            sampler->state.wrap_r == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
            sampler->state.lod_bias != 0.0f;
   }
   if (!lower_textures)
      sampler_mask = 0;
   if ((!fs && !fold_viewport && !unpack_attribs && !lower_textures) ||
       !vs->original_nir)
      return fold_viewport || unpack_attribs || lower_textures ? NULL : plain;

   const uint64_t fs_mask = fs ? fs->varying_interp_mask : 0;
   const uint64_t shared = vs->varying_interp_mask & fs_mask;
   bool differs = false;
   for (unsigned i = 0; i < GEX_WGPU_MAX_VARYING_LOCATIONS; ++i)
      if ((shared & (1ull << i)) &&
          vs->varying_interp[i] != fs->varying_interp[i])
         differs = true;
   if (!differs && !fold_viewport && !unpack_attribs && !lower_textures)
      return plain;

   const float none[4] = {0.0f, 0.0f, 0.0f, 0.0f};
   const float *want = fold_viewport ? viewport : none;
   const uint8_t *want_interp = fs ? fs->varying_interp : NULL;
   for (struct gex_wgpu_interp_variant *v = vs->interp_variants; v;
        v = v->next) {
      bool match = v->mask == fs_mask && v->sampler_mask == sampler_mask &&
                   memcmp(v->viewport, want, sizeof(v->viewport)) == 0 &&
                   memcmp(v->attrib_unpack, want_unpack,
                          sizeof(v->attrib_unpack)) == 0 &&
                   (!want_interp ||
                    memcmp(v->interp, want_interp, sizeof(v->interp)) == 0);
      for (unsigned slot = 0; match && slot < GEX_WGPU_MAX_SAMPLERS; ++slot) {
         if (!(sampler_mask & (1u << slot)))
            continue;
         const struct pipe_sampler_view *view = ctx->vertex_sampler_views[slot];
         const uint8_t swizzle[4] = {
            gex_wgpu_effective_view_swizzle(view, 0),
            gex_wgpu_effective_view_swizzle(view, 1),
            gex_wgpu_effective_view_swizzle(view, 2),
            gex_wgpu_effective_view_swizzle(view, 3),
         };
         /* `match &&`, as in the fragment path: assigning would discard the
          * interpolation, viewport and attribute-unpack checks made before
          * this loop, and every slot compared before this one. */
         if (ctx->vertex_samplers[slot])
            match = match && memcmp(&v->sampler_keys[slot],
                                    &ctx->vertex_samplers[slot]->state,
                                    sizeof(v->sampler_keys[slot])) == 0;
         match = match && memcmp(v->swizzles[slot], swizzle,
                                 sizeof(swizzle)) == 0;
      }
      if (match)
         return flipped ? v->module_y_flipped : v->module;
   }

   struct gex_wgpu_interp_variant *variant =
      CALLOC_STRUCT(gex_wgpu_interp_variant);
   if (!variant)
      return fold_viewport || unpack_attribs ? NULL : plain;
   variant->mask = fs_mask;
   memcpy(variant->viewport, want, sizeof(variant->viewport));
   memcpy(variant->attrib_unpack, want_unpack,
          sizeof(variant->attrib_unpack));
   variant->sampler_mask = sampler_mask;
   for (unsigned slot = 0; slot < GEX_WGPU_MAX_SAMPLERS; ++slot) {
      if (!(sampler_mask & (1u << slot)))
         continue;
      if (ctx->vertex_samplers[slot])
         variant->sampler_keys[slot] = ctx->vertex_samplers[slot]->state;
      for (unsigned channel = 0; channel < 4; ++channel)
         variant->swizzles[slot][channel] = gex_wgpu_effective_view_swizzle(
            ctx->vertex_sampler_views[slot], channel);
   }
   if (want_interp)
      memcpy(variant->interp, want_interp, sizeof(variant->interp));
   for (unsigned pass = 0; pass < 2; ++pass) {
      nir_shader *clone = nir_shader_clone(NULL, vs->original_nir);
      if (!clone)
         break;
      if (fs) {
         nir_foreach_variable_with_modes(var, clone, nir_var_shader_out) {
            if (!gex_wgpu_is_location_varying(var->data.location) ||
                var->data.location >= GEX_WGPU_MAX_VARYING_LOCATIONS ||
                !(fs->varying_interp_mask & (1ull << var->data.location)))
               continue;
            var->data.interpolation = fs->varying_interp[var->data.location];
         }
      }
      if (fold_viewport) {
         /* compile_nir flips y after this, so the y translate has to be
          * pre-negated for the flipped module: -(y*sy + ty*w) is
          * y*(-sy) - ty*w, and what is wanted is (-y)*sy + ty*w. */
         const float folded[4] = {viewport[0], viewport[1], viewport[2],
                                  pass == 1 ? -viewport[3] : viewport[3]};
         nir_shader_intrinsics_pass(
            clone, gex_wgpu_lower_viewport_transform,
            nir_metadata_control_flow | nir_metadata_loop_analysis,
            (void *)folded);
      }
      if (unpack_attribs)
         gex_wgpu_lower_vertex_attrib_unpack(clone, want_unpack);
      if (lower_textures &&
          !gex_wgpu_lower_texture_variant(clone, vs, ctx->vertex_samplers,
                                          ctx->vertex_sampler_views)) {
         ralloc_free(clone);
         break;
      }
      char *error = NULL;
      WGPUShaderModule module = gex_wgpu_compile_nir(
         &ctx->base, clone, MESA_SHADER_VERTEX, pass == 1, &error);
      FREE(error);
      if (pass == 0)
         variant->module = module;
      else
         variant->module_y_flipped = module;
   }
   if (!variant->module || !variant->module_y_flipped) {
      if (variant->module)
         wgpuShaderModuleRelease(variant->module);
      if (variant->module_y_flipped)
         wgpuShaderModuleRelease(variant->module_y_flipped);
      FREE(variant);
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr, "gex-wgpu: could not build a vertex shader variant "
                         "(interpolation, viewport, attribute unpack or "
                         "integer sampling)\n");
      }
      return fold_viewport || unpack_attribs || lower_textures ? NULL : plain;
   }
   variant->next = vs->interp_variants;
   vs->interp_variants = variant;
   return flipped ? variant->module_y_flipped : variant->module;
}

/* GL's default generic vertex attribute, (0, 0, 0, 1), in both a float and an
 * integer form.
 *
 * GL lets a shader read an attribute whose array is disabled: it gets the
 * current generic value. WebGPU has no such thing -- every @location a module
 * declares must be in the pipeline's vertex state, or the pipeline is a
 * validation error ("Vertex attribute slot N used in ShaderModule is not
 * present in the VertexState"), and a validation error poisons the device for
 * every later command. So the missing locations are supplied from here, on a
 * layout with arrayStride 0, which makes every vertex read the same value. */
/* A 1x1 texture for a sampler slot with no texture bound.
 *
 * GL allows this and defines it: sampling a unit whose texture is incomplete
 * -- or absent -- returns (0, 0, 0, 1), and applications rely on it. A GL 1.x
 * renderer leaves texture units enabled across state changes as a matter of
 * course, so st/mesa builds a fragment shader that samples a unit nothing is
 * bound to.
 *
 * WebGPU has no such allowance: every binding a module declares must be in
 * the bind group. The driver used to decline the whole draw, which is how
 * Minecraft lost the one draw that composites its framebuffer onto the
 * screen -- the entire game, every frame, leaving only whatever had last
 * been drawn straight to the window.
 *
 * Cached per (dimension, sample type) because the view has to match what the
 * module declared, or the bind group is a validation error.
 */
static WGPUTextureView
gex_wgpu_default_texture_view(struct gex_wgpu_screen *screen,
                              WGPUTextureViewDimension dimension,
                              WGPUTextureSampleType sample_type)
{
   simple_mtx_lock(&screen->cache_lock);
   for (unsigned i = 0; i < screen->default_texture_count; ++i)
      if (screen->default_textures[i].dimension == dimension &&
          screen->default_textures[i].sample_type == sample_type)
         { const WGPUTextureView gex_ret = screen->default_textures[i].view;
           simple_mtx_unlock(&screen->cache_lock);
           return gex_ret; }
   if (screen->default_texture_count >= ARRAY_SIZE(screen->default_textures))
      { const WGPUTextureView gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }

   WGPUTextureFormat format;
   switch (sample_type) {
   case WGPUTextureSampleType_Uint:  format = WGPUTextureFormat_RGBA8Uint; break;
   case WGPUTextureSampleType_Sint:  format = WGPUTextureFormat_RGBA8Sint; break;
   case WGPUTextureSampleType_Depth: format = WGPUTextureFormat_Depth16Unorm; break;
   default:                          format = WGPUTextureFormat_RGBA8Unorm; break;
   }
   const bool cube = dimension == WGPUTextureViewDimension_Cube ||
                     dimension == WGPUTextureViewDimension_CubeArray;
   const uint32_t layers = cube ? 6u : 1u;

   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.dimension = dimension == WGPUTextureViewDimension_3D
      ? WGPUTextureDimension_3D
      : dimension == WGPUTextureViewDimension_1D ? WGPUTextureDimension_1D
                                                 : WGPUTextureDimension_2D;
   desc.size = (WGPUExtent3D){1, dimension == WGPUTextureViewDimension_1D ? 1u : 1u,
                              layers};
   desc.format = format;
   desc.mipLevelCount = 1;
   desc.sampleCount = 1;
   desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
   /* A depth stand-in cannot be written by copy, only by rendering, so it is
    * left at WebGPU's zero -- which reads as the near plane. */
   if (sample_type == WGPUTextureSampleType_Depth)
      desc.usage = WGPUTextureUsage_TextureBinding |
                   WGPUTextureUsage_RenderAttachment;
   WGPUTexture texture = wgpuDeviceCreateTexture(screen->device, &desc);
   if (!texture)
      { const WGPUTextureView gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }

   if (sample_type != WGPUTextureSampleType_Depth) {
      /* Opaque black, which is what GL says an incomplete texture samples as.
       * WebGPU zeroes a new texture, so without this the alpha would be 0. */
      const uint8_t opaque_black[4] = {0, 0, 0, 255};
      const uint32_t opaque_black_int[4] = {0, 0, 0, 1};
      const bool integer = sample_type == WGPUTextureSampleType_Uint ||
                           sample_type == WGPUTextureSampleType_Sint;
      /* RGBA8Uint/Sint are still one byte per channel; the 1 goes in alpha. */
      const uint8_t texel[4] = {0, 0, 0, 1};
      (void)opaque_black_int;
      for (uint32_t layer = 0; layer < layers; ++layer) {
         WGPUTexelCopyTextureInfo to = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
         to.texture = texture;
         to.origin = (WGPUOrigin3D){0, 0, layer};
         WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
         layout.bytesPerRow = 4;
         layout.rowsPerImage = 1;
         WGPUExtent3D extent = {1, 1, 1};
         wgpuQueueWriteTexture(screen->queue, &to,
                               integer ? texel : opaque_black, 4, &layout,
                               &extent);
      }
   }

   WGPUTextureViewDescriptor view_desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   view_desc.format = format;
   view_desc.dimension = dimension;
   view_desc.mipLevelCount = 1;
   view_desc.arrayLayerCount = layers;
   view_desc.aspect = sample_type == WGPUTextureSampleType_Depth
      ? WGPUTextureAspect_DepthOnly : WGPUTextureAspect_All;
   WGPUTextureView view = wgpuTextureCreateView(texture, &view_desc);
   if (!view) {
      wgpuTextureRelease(texture);
      { const WGPUTextureView gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   }
   const unsigned i = screen->default_texture_count++;
   screen->default_textures[i].dimension = dimension;
   screen->default_textures[i].sample_type = sample_type;
   screen->default_textures[i].texture = texture;
   screen->default_textures[i].view = view;
   { const WGPUTextureView gex_ret = view;
     simple_mtx_unlock(&screen->cache_lock);
     return gex_ret; }
}

/* The sampler that goes with it, for a slot whose sampler state is also
 * unset. Nearest and clamped: nothing samples anything but the one texel. */
static WGPUSampler
gex_wgpu_default_sampler(struct gex_wgpu_screen *screen,
                         WGPUSamplerBindingType type)
{
   simple_mtx_lock(&screen->cache_lock);
   const unsigned index = type == WGPUSamplerBindingType_Comparison   ? 2u
                        : type == WGPUSamplerBindingType_NonFiltering ? 1u
                                                                      : 0u;
   if (screen->default_samplers[index])
      { const WGPUSampler gex_ret = screen->default_samplers[index];
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
   desc.addressModeU = WGPUAddressMode_ClampToEdge;
   desc.addressModeV = WGPUAddressMode_ClampToEdge;
   desc.addressModeW = WGPUAddressMode_ClampToEdge;
   desc.magFilter = WGPUFilterMode_Nearest;
   desc.minFilter = WGPUFilterMode_Nearest;
   desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
   desc.maxAnisotropy = 1;
   if (index == 2)
      desc.compare = WGPUCompareFunction_Always;
   screen->default_samplers[index] =
      wgpuDeviceCreateSampler(screen->device, &desc);
   { const WGPUSampler gex_ret = screen->default_samplers[index];
     simple_mtx_unlock(&screen->cache_lock);
     return gex_ret; }
}

static int64_t gex_wgpu_time_pipeline;
static int64_t gex_wgpu_time_submit;
static int64_t gex_wgpu_time_draw;
static int64_t gex_wgpu_time_bind;
static int64_t gex_wgpu_time_fsmod;
static int64_t gex_wgpu_time_bounds;
static int64_t gex_wgpu_time_record;
static int64_t gex_wgpu_time_views;
static int64_t gex_wgpu_time_finish;
static int64_t gex_wgpu_time_begin;
static int64_t gex_wgpu_time_end;
static int64_t gex_wgpu_time_bg;
static int64_t gex_wgpu_time_encoder;
uint64_t gex_wgpu_presents;
/* Nanoseconds inside flush_frontbuffer: the readback plus however long the
 * GPU still needed. The native target adds to it. */
uint64_t gex_wgpu_present_nanos;
static uint64_t gex_wgpu_batched_submits;
static uint64_t gex_wgpu_batched_draws;

static int64_t gex_wgpu_time_origin;
static uint64_t gex_wgpu_timed_draws;

static bool
gex_wgpu_entry_timing_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *value = getenv("GEX_WGPU_TIMING");
      enabled = value && *value == '2';
   }
   return enabled != 0;
}

static bool
gex_wgpu_timing_enabled(void)
{
   static int enabled = -1;
   if (enabled < 0) {
      const char *value = getenv("GEX_WGPU_TIMING");
      enabled = value && *value && *value != '0';
      if (enabled)
         gex_wgpu_time_origin = os_time_get_nano();
   }
   return enabled != 0;
}

/* --- cached bind group layouts, pipeline layouts and bind groups --------
 *
 * All three are keyed on the bytes of their descriptor's entry array. The
 * arrays are memset before being filled, so padding is zero and the byte
 * comparison is exact rather than merely likely.
 */
static uint32_t
gex_wgpu_cache_hash(const void *data, size_t bytes)
{
   const uint8_t *p = data;
   uint32_t h = 2166136261u;
   for (size_t i = 0; i < bytes; ++i) {
      h ^= p[i];
      h *= 16777619u;
   }
   return h ? h : 1u;
}

static WGPUBindGroupLayout
gex_wgpu_cached_bind_group_layout(struct gex_wgpu_screen *screen,
                                  const WGPUBindGroupLayoutEntry *entries,
                                  unsigned count)
{
   simple_mtx_lock(&screen->cache_lock);
   if (count > GEX_WGPU_CACHE_MAX_ENTRIES)
      { const WGPUBindGroupLayout gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   const size_t bytes = count * sizeof(*entries);
   const uint32_t hash =
      gex_wgpu_cache_hash(entries, bytes) ^ (count * 2654435761u);
   for (unsigned i = 0; i < screen->bgl_cache_count; ++i)
      if (screen->bgl_cache[i].hash == hash &&
          screen->bgl_cache[i].count == count &&
          memcmp(screen->bgl_cache[i].entries, entries, bytes) == 0)
         { const WGPUBindGroupLayout gex_ret = screen->bgl_cache[i].layout;
           simple_mtx_unlock(&screen->cache_lock);
           return gex_ret; }

   WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
   desc.entryCount = count;
   desc.entries = entries;
   WGPUBindGroupLayout layout =
      wgpuDeviceCreateBindGroupLayout(screen->device, &desc);
   if (!layout)
      { const WGPUBindGroupLayout gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   if (screen->bgl_cache_count < GEX_WGPU_BGL_CACHE_SLOTS) {
      struct gex_wgpu_bgl_cache_slot *slot =
         &screen->bgl_cache[screen->bgl_cache_count++];
      slot->hash = hash;
      slot->count = count;
      memcpy(slot->entries, entries, bytes);
      slot->layout = layout;
   } else {
      /* Full: the object is still correct, it is just never found again.
       * Leaked rather than freed, because a freed layout's address could be
       * reused and then match a pipeline layout key that named the old one. */
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr, "gex-wgpu: bind group layout cache full (%d); "
                         "layouts are no longer reused\n",
                 GEX_WGPU_BGL_CACHE_SLOTS);
      }
   }
   { const WGPUBindGroupLayout gex_ret = layout;
     simple_mtx_unlock(&screen->cache_lock);
     return gex_ret; }
}

static WGPUPipelineLayout
gex_wgpu_cached_pipeline_layout(struct gex_wgpu_screen *screen,
                                WGPUBindGroupLayout *layouts, unsigned count)
{
   simple_mtx_lock(&screen->cache_lock);
   if (count > 4)
      { const WGPUPipelineLayout gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   const size_t bytes = count * sizeof(*layouts);
   const uint32_t hash = gex_wgpu_cache_hash(layouts, bytes) ^ (count * 40503u);
   for (unsigned i = 0; i < screen->pl_cache_count; ++i)
      if (screen->pl_cache[i].hash == hash &&
          screen->pl_cache[i].count == count &&
          memcmp(screen->pl_cache[i].layouts, layouts, bytes) == 0)
         { const WGPUPipelineLayout gex_ret = screen->pl_cache[i].layout;
           simple_mtx_unlock(&screen->cache_lock);
           return gex_ret; }

   WGPUPipelineLayoutDescriptor desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
   desc.bindGroupLayoutCount = count;
   desc.bindGroupLayouts = layouts;
   WGPUPipelineLayout layout =
      wgpuDeviceCreatePipelineLayout(screen->device, &desc);
   if (!layout)
      { const WGPUPipelineLayout gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   if (screen->pl_cache_count < GEX_WGPU_PL_CACHE_SLOTS) {
      struct gex_wgpu_pl_cache_slot *slot =
         &screen->pl_cache[screen->pl_cache_count++];
      slot->hash = hash;
      slot->count = count;
      memcpy(slot->layouts, layouts, bytes);
      slot->layout = layout;
   }
   { const WGPUPipelineLayout gex_ret = layout;
     simple_mtx_unlock(&screen->cache_lock);
     return gex_ret; }
}

/* Draws recorded but not yet handed to the queue.
 *
 * A queue submit costs about 35 us whatever is in it, which was 70% of what
 * a draw cost once its pipeline and bind groups were being reused -- and
 * Minecraft issues some 780 draws a frame. They all go into one command
 * encoder now and reach the queue together.
 *
 * The ordering rule this rests on: wgpuQueueWriteBuffer and every other
 * submit take effect at the point they are *called*, not where the recorded
 * commands sit, so anything else that touches the device has to push these
 * out first. gex_wgpu_submit_pending is called at the top of every context
 * entry point that does, which is every one that is not a draw or a plain
 * state change. Uniform uploads are the deliberate exception: they are
 * sub-allocated from a ring and never rewrite bytes a recorded draw reads.
 */
/* Which entry point broke the batch, and how often. A batch that submits once
 * per draw costs more than no batching at all, and the only way to tell is to
 * name the caller. */
#define GEX_WGPU_SUBMIT_CAUSES 24
static const char *gex_wgpu_submit_cause[GEX_WGPU_SUBMIT_CAUSES];
static uint64_t gex_wgpu_submit_cause_count[GEX_WGPU_SUBMIT_CAUSES];

/* The periodic rolling report. Called from the draw path whether or not the
 * draw is issued, so GEX_WGPU_NO_DRAW still reports a frame rate. */
static void
gex_wgpu_report_timing(const struct gex_wgpu_screen *sc)
{
   if (++gex_wgpu_timed_draws % 20000 != 0)
      return;

         const int64_t wall = os_time_get_nano() - gex_wgpu_time_origin;
         const uint64_t n = gex_wgpu_timed_draws;
         /* Since the previous report, not since the start: an application
          * spends its first minute loading, and an average over that says
          * nothing about how fast it runs once it is going. */
         static int64_t last_at;
         static uint64_t last_draws, last_presents;
         static int64_t last_draw_nanos, last_entry_nanos;
         const double seconds = (wall - last_at) / 1e9;
         static uint64_t last_present_nanos;
         if (seconds > 0.0)
            fprintf(stderr,
                    "gex-wgpu: %.1f frames/s, %.0f draws/s, %.0f draws/frame"
                    ", %.0f%% of wall in draw_vbo, %.0f%% in state calls"
                    ", %.0f%% in present\n",
                    (gex_wgpu_presents - last_presents) / seconds,
                    (n - last_draws) / seconds,
                    gex_wgpu_presents > last_presents
                       ? (double)(n - last_draws) /
                         (double)(gex_wgpu_presents - last_presents) : 0.0,
                    100.0 * (gex_wgpu_time_draw - last_draw_nanos) /
                       (double)(wall - last_at),
                    100.0 * (gex_wgpu_time_entry - last_entry_nanos) /
                       (double)(wall - last_at),
                    100.0 * (gex_wgpu_present_nanos - last_present_nanos) /
                       (double)(wall - last_at));
         last_entry_nanos = gex_wgpu_time_entry;
         last_present_nanos = gex_wgpu_present_nanos;
         last_draw_nanos = gex_wgpu_time_draw;
         last_at = wall;
         last_draws = n;
         last_presents = gex_wgpu_presents;
         fprintf(stderr,
                 "gex-wgpu: %llu draws, %lld ms wall, %lld ms in draw_vbo "
                 "(%.0f us/draw: bounds %.0f, fs-module %.0f, layouts+bindgroups %.0f, "
                 "pipeline %.0f, views %.0f, encoder %.0f, bindgroups %.0f, begin %.0f, "
                 "setters %.0f, end %.0f, "
                 "finish %.0f, "
                 "submit %.0f, "
                 "rest %.0f)\n",
                 (unsigned long long)n, (long long)(wall / 1000000),
                 (long long)(gex_wgpu_time_draw / 1000000),
                 gex_wgpu_time_draw / 1000.0 / n,
                 gex_wgpu_time_bounds / 1000.0 / n,
                 gex_wgpu_time_fsmod / 1000.0 / n,
                 gex_wgpu_time_bind / 1000.0 / n,
                 gex_wgpu_time_pipeline / 1000.0 / n,
                 gex_wgpu_time_views / 1000.0 / n,
                 gex_wgpu_time_encoder / 1000.0 / n,
                 gex_wgpu_time_bg / 1000.0 / n,
                 gex_wgpu_time_begin / 1000.0 / n,
                 (gex_wgpu_time_record - gex_wgpu_time_views -
                  gex_wgpu_time_finish - gex_wgpu_time_begin -
                  gex_wgpu_time_end - gex_wgpu_time_bg -
                  gex_wgpu_time_encoder) / 1000.0 / n,
                 gex_wgpu_time_end / 1000.0 / n,
                 gex_wgpu_time_finish / 1000.0 / n,
                 gex_wgpu_time_submit / 1000.0 / n,
                 (gex_wgpu_time_draw - gex_wgpu_time_bounds -
                  gex_wgpu_time_fsmod -
                  gex_wgpu_time_bind - gex_wgpu_time_pipeline -
                  gex_wgpu_time_record -
                  gex_wgpu_time_submit) / 1000.0 / n);

         const uint64_t bg = sc->bg_cache_hits + sc->bg_cache_misses;
         const uint64_t pl = sc->pipeline_cache_hits + sc->pipeline_cache_misses;
         fprintf(stderr,
                 "gex-wgpu: cache hit rate -- bind groups %.1f%% (%llu), "
                 "pipelines %.1f%% (%llu)\n",
                 bg ? 100.0 * sc->bg_cache_hits / bg : 0.0,
                 (unsigned long long)sc->bg_cache_misses,
                 pl ? 100.0 * sc->pipeline_cache_hits / pl : 0.0,
                 (unsigned long long)sc->pipeline_cache_misses);
         fprintf(stderr, "gex-wgpu: %llu submits for %llu batched draws;"
                         " broken by:",
                 (unsigned long long)gex_wgpu_batched_submits,
                 (unsigned long long)gex_wgpu_batched_draws);
         for (unsigned i = 0; i < GEX_WGPU_SUBMIT_CAUSES; ++i)
            if (gex_wgpu_submit_cause[i])
               fprintf(stderr, " %s=%llu", gex_wgpu_submit_cause[i],
                       (unsigned long long)gex_wgpu_submit_cause_count[i]);
         fprintf(stderr, "\n");
      }

/* Closes the open render pass, leaving the encoder to carry on. Every path
 * that stops recording draws goes through here: a different framebuffer, a
 * clear, an occlusion query, or the submit itself. */
static void
gex_wgpu_end_pass(struct gex_wgpu_context *ctx)
{
   if (!ctx->pending_pass)
      return;
   wgpuRenderPassEncoderEnd(ctx->pending_pass);
   wgpuRenderPassEncoderRelease(ctx->pending_pass);
   ctx->pending_pass = NULL;
   memset(&ctx->pending_pass_key, 0, sizeof(ctx->pending_pass_key));
   memset(&ctx->emitted, 0, sizeof(ctx->emitted));
}

static void
gex_wgpu_submit_pending_named(struct gex_wgpu_context *ctx, const char *why)
{
   gex_wgpu_end_pass(ctx);
   if (!ctx->pending_encoder)
      return;
   for (unsigned i = 0; i < GEX_WGPU_SUBMIT_CAUSES; ++i) {
      if (!gex_wgpu_submit_cause[i]) gex_wgpu_submit_cause[i] = why;
      if (gex_wgpu_submit_cause[i] == why) {
         ++gex_wgpu_submit_cause_count[i];
         break;
      }
   }
   WGPUCommandEncoder encoder = ctx->pending_encoder;
   const unsigned draws = ctx->pending_draws;
   ctx->pending_encoder = NULL;
   ctx->pending_draws = 0;
   for (unsigned i = 0; i < ctx->batch_touched_count; ++i) {
      gex_resource(ctx->batch_touched[i])->in_batch = false;
      pipe_resource_reference(&ctx->batch_touched[i], NULL);
   }
   ctx->batch_touched_count = 0;
   ctx->batch_touched_overflowed = false;
   WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, NULL);
   wgpuCommandEncoderRelease(encoder);
   if (!command) {
      gex_fail(ctx);
      return;
   }
   /* Every uniform these draws read, in one write. It has to go on the queue
    * before the submit, which is exactly here. */
   gex_wgpu_flush_constant_ring(ctx);
   gex_wgpu_queue_submit(gex_screen(ctx->base.screen), 1, &command);
   wgpuCommandBufferRelease(command);
   gex_wgpu_batched_submits++;
   gex_wgpu_batched_draws += draws;
}

/* Dynamic offsets, which WebGPU takes as one array per bind group in
 * increasing binding order -- not in the order the entries were built. */
struct gex_wgpu_dynamic_offsets {
   unsigned count;
   uint32_t binding[GEX_WGPU_MAX_CONSTANT_BUFFERS];
   uint32_t offset[GEX_WGPU_MAX_CONSTANT_BUFFERS];
};

static void
gex_wgpu_add_dynamic_offset(struct gex_wgpu_dynamic_offsets *list,
                            uint32_t binding, uint32_t offset)
{
   if (list->count >= ARRAY_SIZE(list->binding))
      return;
   unsigned at = list->count++;
   while (at > 0 && list->binding[at - 1] > binding) {
      list->binding[at] = list->binding[at - 1];
      list->offset[at] = list->offset[at - 1];
      --at;
   }
   list->binding[at] = binding;
   list->offset[at] = offset;
}

/* Flatten a render pipeline descriptor, and everything it points at, into
 * one byte string usable as a cache key.
 *
 * The pointer fields are zeroed in the copies rather than hashed: they are
 * addresses of stack arrays in the caller and differ on every draw. Their
 * *contents* are appended instead. Anything this function does not know
 * about is still covered, because the structs go in whole. */
struct gex_wgpu_key_builder {
   uint8_t *at;
   size_t left;
   bool overflowed;
};

static void
gex_wgpu_key_add(struct gex_wgpu_key_builder *key, const void *data,
                 size_t bytes)
{
   if (bytes > key->left) {
      key->overflowed = true;
      return;
   }
   memcpy(key->at, data, bytes);
   key->at += bytes;
   key->left -= bytes;
}

static unsigned
gex_wgpu_pipeline_key(const WGPURenderPipelineDescriptor *desc,
                      uint8_t *out, size_t capacity)
{
   struct gex_wgpu_key_builder key = { out, capacity, false };

   WGPURenderPipelineDescriptor head = *desc;
   head.nextInChain = NULL;
   head.label = (WGPUStringView)WGPU_STRING_VIEW_INIT;
   head.vertex.buffers = NULL;
   head.vertex.constants = NULL;
   head.fragment = NULL;
   head.depthStencil = NULL;
   gex_wgpu_key_add(&key, &head, sizeof(head));

   for (size_t i = 0; i < desc->vertex.bufferCount; ++i) {
      WGPUVertexBufferLayout layout = desc->vertex.buffers[i];
      const WGPUVertexAttribute *attributes = layout.attributes;
      const size_t count = layout.attributeCount;
      layout.attributes = NULL;
      gex_wgpu_key_add(&key, &layout, sizeof(layout));
      gex_wgpu_key_add(&key, attributes, count * sizeof(*attributes));
   }

   if (desc->depthStencil) {
      WGPUDepthStencilState depth = *desc->depthStencil;
      depth.nextInChain = NULL;
      gex_wgpu_key_add(&key, &depth, sizeof(depth));
   }

   if (desc->fragment) {
      WGPUFragmentState fragment = *desc->fragment;
      const WGPUColorTargetState *targets = fragment.targets;
      const size_t count = fragment.targetCount;
      fragment.nextInChain = NULL;
      fragment.targets = NULL;
      fragment.constants = NULL;
      gex_wgpu_key_add(&key, &fragment, sizeof(fragment));
      for (size_t i = 0; i < count; ++i) {
         WGPUColorTargetState target = targets[i];
         const WGPUBlendState *blend = target.blend;
         target.nextInChain = NULL;
         target.blend = NULL;
         gex_wgpu_key_add(&key, &target, sizeof(target));
         if (blend)
            gex_wgpu_key_add(&key, blend, sizeof(*blend));
      }
   }
   return key.overflowed ? 0u : (unsigned)(capacity - key.left);
}

static WGPURenderPipeline
gex_wgpu_cached_pipeline(struct gex_wgpu_screen *screen,
                         const WGPURenderPipelineDescriptor *desc,
                         bool *cached)
{
   simple_mtx_lock(&screen->cache_lock);
   uint8_t key[GEX_WGPU_PIPELINE_KEY_MAX];
   const unsigned length = gex_wgpu_pipeline_key(desc, key, sizeof(key));
   *cached = false;
   if (!length)   /* too big to key: build it and let the caller own it */
      { const WGPURenderPipeline gex_ret = wgpuDeviceCreateRenderPipeline(screen->device, desc);
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }

   const uint32_t hash = gex_wgpu_cache_hash(key, length);
   struct gex_wgpu_pipeline_cache_slot *slot =
      &screen->pipeline_cache[hash % GEX_WGPU_PIPELINE_CACHE_SLOTS];
   *cached = true;
   if (slot->used && slot->hash == hash && slot->key_length == length &&
       memcmp(slot->key, key, length) == 0) {
      ++screen->pipeline_cache_hits;
      { const WGPURenderPipeline gex_ret = slot->pipeline;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   }
   ++screen->pipeline_cache_misses;

   WGPURenderPipeline pipeline =
      wgpuDeviceCreateRenderPipeline(screen->device, desc);
   if (!pipeline) {
      *cached = false;
      { const WGPURenderPipeline gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   }
   /* Safe to release the evicted one now: it is a different key from the one
    * this draw just took, and Dawn keeps its own reference for any work
    * already submitted with it. */
   if (slot->used) {
      /* No epoch bump. The pipeline cache does not consult the epoch at all,
       * so bumping it here only invalidated the bind group cache, which does
       * -- and that cache is shared with every other slot. */
      wgpuRenderPipelineRelease(slot->pipeline);
   }
   slot->used = true;
   slot->hash = hash;
   slot->key_length = length;
   memcpy(slot->key, key, length);
   slot->pipeline = pipeline;
   { const WGPURenderPipeline gex_ret = pipeline;
     simple_mtx_unlock(&screen->cache_lock);
     return gex_ret; }
}

/* Bind groups evicted while a draw was still assembling. Released here, one
 * draw later, by which point nothing holds them but Dawn's own reference for
 * whatever work was submitted. */
static void
gex_wgpu_release_retired_groups(struct gex_wgpu_screen *screen)
{
   simple_mtx_lock(&screen->cache_lock);
   for (unsigned i = 0; i < screen->retired_count; ++i)
      wgpuBindGroupRelease(screen->retired_groups[i]);
   screen->retired_count = 0;
   simple_mtx_unlock(&screen->cache_lock);
}

/* Keying a bind group on the *handles* in its entries is safe only because
 * the cache holds a reference to the group, which holds one to each buffer,
 * view and sampler in it -- so none of those handles can be freed and their
 * addresses reused while the key still names them. Without that reference a
 * recycled address would silently match the wrong entry. */
static WGPUBindGroup
gex_wgpu_cached_bind_group(struct gex_wgpu_screen *screen,
                           WGPUBindGroupLayout layout,
                           const WGPUBindGroupEntry *entries, unsigned count)
{
   simple_mtx_lock(&screen->cache_lock);
   if (count > GEX_WGPU_CACHE_MAX_ENTRIES)
      { const WGPUBindGroup gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   const size_t bytes = count * sizeof(*entries);
   uint32_t hash = gex_wgpu_cache_hash(entries, bytes);
   hash = gex_wgpu_cache_hash(&layout, sizeof(layout)) ^ hash ^ (count * 97u);

   struct gex_wgpu_bg_cache_slot *slot =
      &screen->bg_cache[hash % GEX_WGPU_BG_CACHE_SLOTS];
   if (slot->used && slot->epoch == screen->cache_epoch &&
       slot->hash == hash && slot->layout == layout &&
       slot->count == count && memcmp(slot->entries, entries, bytes) == 0) {
      ++screen->bg_cache_hits;
      { const WGPUBindGroup gex_ret = slot->group;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   }
   ++screen->bg_cache_misses;

   WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   desc.layout = layout;
   desc.entryCount = count;
   desc.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(screen->device, &desc);
   if (!group)
      { const WGPUBindGroup gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   if (slot->used) {
      /* Evicting one slot does not invalidate the others.
       *
       * The epoch exists because a key names its buffers, views and samplers
       * by address and Dawn hands a freed address straight back, so a stale
       * key could match the wrong object. What makes that impossible in the
       * ordinary case is that each slot holds a reference to its own group,
       * which holds one to everything the key names -- so nothing a live
       * slot names can be freed underneath it, whatever happens to its
       * neighbours. Only a driver-level sampler or view being destroyed can
       * outrun that, and those two paths still bump the epoch.
       *
       * Bumping it here instead invalidated all 512 slots on every eviction,
       * so a workload that evicted at all could never hit: Minecraft with a
       * crowd reported a 0.0% hit rate over 46,987,974 lookups, building a
       * fresh bind group for every draw. Removing it takes that to 99.6%.
       *
       * Worth measuring rather than assuming, though: the same scene goes
       * from ~55.5 to ~59.9 fps, about 8%. Forty-seven million redundant
       * bind group creations are worth removing, but Dawn builds them
       * cheaply enough that the cache was never the wall it looked like. */
      if (screen->retired_count < ARRAY_SIZE(screen->retired_groups))
         screen->retired_groups[screen->retired_count++] = slot->group;
      else
         wgpuBindGroupRelease(slot->group);
   }
   slot->used = true;
   slot->epoch = screen->cache_epoch;
   slot->hash = hash;
   slot->layout = layout;
   slot->count = count;
   memcpy(slot->entries, entries, bytes);
   slot->group = group;
   { const WGPUBindGroup gex_ret = group;
     simple_mtx_unlock(&screen->cache_lock);
     return gex_ret; }
}

static WGPUBuffer
gex_wgpu_default_attribute_buffer(struct gex_wgpu_screen *screen)
{
   simple_mtx_lock(&screen->cache_lock);
   if (screen->default_attributes)
      { const WGPUBuffer gex_ret = screen->default_attributes;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   desc.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
   desc.size = 32;
   WGPUBuffer buffer = wgpuDeviceCreateBuffer(screen->device, &desc);
   if (!buffer)
      { const WGPUBuffer gex_ret = NULL;
        simple_mtx_unlock(&screen->cache_lock);
        return gex_ret; }
   /* GL's *initial* current attribute value, not its current one.
    *
    * A disabled vertex array reads the value `glVertexAttrib*` last set for
    * that index, and this buffer only ever holds the default. st/mesa normally
    * hides that by uploading the current values itself and binding them, so
    * the gap only shows when it does not -- and it does not always: this
    * driver logs "vertex element N reads its constant value (buffer slot N is
    * not bound)" when st leaves the slot empty, and every location served from
    * here then reads (0, 0, 0, 1) whatever the application set.
    *
    * Anything positional fed that way lands in the wrong place. */
   const uint32_t contents[8] = {
      0, 0, 0, 0x3f800000u, /* 0.0, 0.0, 0.0, 1.0 as float32x4 */
      0, 0, 0, 1u,          /* 0, 0, 0, 1 as (s|u)int32x4 */
   };
   wgpuQueueWriteBuffer(screen->queue, buffer, 0, contents, sizeof(contents));
   screen->default_attributes = buffer;
   { const WGPUBuffer gex_ret = buffer;
     simple_mtx_unlock(&screen->cache_lock);
     return gex_ret; }
}

static void
gex_wgpu_draw_vbo_inner(struct pipe_context *pipe,
                        const struct pipe_draw_info *info,
                        unsigned drawid_offset,
                        const struct pipe_draw_indirect_info *indirect,
                        const struct pipe_draw_start_count_bias *draws,
                        unsigned num_draws);

/* True for the four topologies that carry adjacent vertices. */
static bool
gex_wgpu_is_adjacency_prim(enum mesa_prim mode)
{
   return mode == MESA_PRIM_LINES_ADJACENCY ||
          mode == MESA_PRIM_LINE_STRIP_ADJACENCY ||
          mode == MESA_PRIM_TRIANGLES_ADJACENCY ||
          mode == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY;
}

/* Render an adjacency primitive with no geometry shader bound.
 *
 * GL says the adjacent vertices are simply not there when nothing consumes
 * them: LINES_ADJACENCY draws a line from vertices 1 and 2 of every four,
 * TRIANGLES_ADJACENCY a triangle from 0, 2 and 4 of every six, and the strip
 * forms walk the same way one primitive at a time. No hardware driver
 * implements this, because the hardware does it, so u_index_translator maps
 * every adjacency mode to itself -- which is why handing one to
 * u_primconvert returned it unchanged and this function's absence showed up
 * as unbounded re-entry rather than as a wrong picture.
 *
 * The result is issued as an ordinary indexed draw with a fresh 32-bit index
 * list, so everything below this point sees a topology it already has.
 * Returns false having changed nothing. */
static bool
gex_wgpu_draw_adjacency(struct pipe_context *pipe,
                        const struct pipe_draw_info *info,
                        unsigned drawid_offset,
                        const struct pipe_draw_start_count_bias *draw)
{
   const unsigned count = draw->count;
   enum mesa_prim base;
   unsigned group, stride, per_prim, first_prim_vertices;
   switch (info->mode) {
   case MESA_PRIM_LINES_ADJACENCY:
      base = MESA_PRIM_LINES; group = 4; stride = 4; per_prim = 2;
      first_prim_vertices = 4;
      break;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      base = MESA_PRIM_LINES; group = 4; stride = 1; per_prim = 2;
      first_prim_vertices = 4;
      break;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      base = MESA_PRIM_TRIANGLES; group = 6; stride = 6; per_prim = 3;
      first_prim_vertices = 6;
      break;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      base = MESA_PRIM_TRIANGLES; group = 6; stride = 2; per_prim = 3;
      first_prim_vertices = 6;
      break;
   default:
      return false;
   }
   if (count < first_prim_vertices)
      return true; /* fewer vertices than one primitive: nothing is drawn */
   const unsigned prims = (count - first_prim_vertices) / stride + 1;
   const uint64_t out_count = (uint64_t)prims * per_prim;
   if (!prims || out_count > UINT32_MAX / 4)
      return false;

   /* The source indices, if any. A non-indexed draw numbers its vertices
    * from `start`, which the same rewrite can express as indices. */
   const void *src = NULL;
   struct pipe_transfer *transfer = NULL;
   if (info->index_size) {
      if (info->index_size != 1 && info->index_size != 2 &&
          info->index_size != 4)
         return false;
      if (info->has_user_indices) {
         if (!info->index.user)
            return false;
         src = (const uint8_t *)info->index.user +
               (size_t)draw->start * info->index_size;
      } else {
         src = pipe_buffer_map_range(pipe, info->index.resource,
                                     (unsigned)draw->start * info->index_size,
                                     count * info->index_size,
                                     PIPE_MAP_READ, &transfer);
         if (!src)
            return false;
      }
   }

   uint32_t *out = MALLOC((size_t)out_count * 4);
   if (!out) {
      if (transfer)
         pipe_buffer_unmap(pipe, transfer);
      return false;
   }
   unsigned at = 0;
   for (unsigned prim = 0; prim < prims; ++prim) {
      const unsigned origin = prim * stride;
      for (unsigned vertex = 0; vertex < per_prim; ++vertex) {
         /* Which vertex of the group this primitive's Nth corner is. The
          * strip forms are the same selection walked two vertices at a
          * time; a triangle strip alternates its winding, which matters
          * only when a face is being culled and is preserved here so that
          * it is right when one is. */
         unsigned pick;
         if (base == MESA_PRIM_LINES)
            pick = 1 + vertex;
         else if (info->mode == MESA_PRIM_TRIANGLE_STRIP_ADJACENCY &&
                  (prim & 1))
            pick = vertex == 0 ? 2 : vertex == 1 ? 0 : 4;
         else
            pick = vertex * 2;
         const unsigned position = origin + pick;
         assert(position < count);
         uint32_t index;
         switch (info->index_size) {
         case 0: index = (uint32_t)draw->start + position; break;
         case 1: index = ((const uint8_t *)src)[position]; break;
         case 2: index = ((const uint16_t *)src)[position]; break;
         default: index = ((const uint32_t *)src)[position]; break;
         }
         out[at++] = index;
      }
   }
   if (transfer)
      pipe_buffer_unmap(pipe, transfer);

   struct pipe_draw_info rewritten = *info;
   rewritten.mode = base;
   rewritten.index_size = 4;
   rewritten.has_user_indices = true;
   rewritten.index.user = out;
   rewritten.primitive_restart = false;
   rewritten.index_bounds_valid = false;
   rewritten.min_index = 0;
   rewritten.max_index = ~0u;
   struct pipe_draw_start_count_bias rewritten_draw = *draw;
   rewritten_draw.start = 0;
   rewritten_draw.count = (unsigned)out_count;
   gex_wgpu_draw_vbo(pipe, &rewritten, drawid_offset, NULL, &rewritten_draw, 1);
   FREE(out);
   return true;
}


static void
gex_wgpu_draw_vbo(struct pipe_context *pipe,
                  const struct pipe_draw_info *info,
                  unsigned drawid_offset,
                  const struct pipe_draw_indirect_info *indirect,
                  const struct pipe_draw_start_count_bias *draws,
                  unsigned num_draws)
{
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;

   /* See draw_depth: the two re-entrant paths below can loop, and the abort
    * that follows is not recoverable. Eight is far above the two levels a
    * legitimate conversion needs. */
   if (ctx->draw_depth >= 8) {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr,
                 "gex-wgpu: draw refused (draw_vbo re-entered %u deep) mode=%u "
                 "isz=%u restart=%d draws=%u\n",
                 ctx->draw_depth, info ? info->mode : 0,
                 info ? info->index_size : 0,
                 info ? (int)info->primitive_restart : 0, num_draws);
      }
      /* Declined, not latched. Latching becomes GL_OUT_OF_MEMORY in st/mesa,
       * which dEQP turns into a fatal ResourceError and loses every case
       * after this one; a lost draw is the smaller answer. */
      return;
   }
   ctx->draw_depth++;
   gex_wgpu_draw_vbo_inner(pipe, info, drawid_offset, indirect, draws,
                           num_draws);
   ctx->draw_depth--;
}




static void
gex_wgpu_draw_vbo_inner(struct pipe_context *pipe,
                        const struct pipe_draw_info *info,
                        unsigned drawid_offset,
                        const struct pipe_draw_indirect_info *indirect,
                        const struct pipe_draw_start_count_bias *draws,
                        unsigned num_draws)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   /* The whole call, so that the pipeline and submit figures below can be
    * read as fractions of the driver's own cost rather than of wall clock,
    * which also contains the application and the frame readback. */
   const bool gex_timing = gex_wgpu_timing_enabled();
   const int64_t t_draw_start = gex_timing ? os_time_get_nano() : 0;
   /* GEX_WGPU_NO_DRAW=1 drops every draw on the floor. Not useful output --
    * the point is the frame rate it leaves, which is the ceiling everything
    * above this driver imposes: the application's own CPU work and st/mesa's.
    * Comparing that with the real frame rate says whether there is anything
    * left to win down here. */
   static int gex_no_draw = -1;
   if (gex_no_draw < 0) {
      const char *value = getenv("GEX_WGPU_NO_DRAW");
      gex_no_draw = value && *value && *value != '0';
   }
   if (gex_no_draw) {
      gex_wgpu_report_timing(gex_screen(pipe->screen));
      return;
   }
   gex_wgpu_release_retired_groups(gex_screen(pipe->screen));

   /* A topology WebGPU does not have is rewritten into one it does, which
    * comes straight back here as a supported mode -- but only for the
    * topologies u_index_translator actually converts. */
   if (!ctx->failed && info && draws &&
       !(GEX_WGPU_PRIM_MODES & BITFIELD_BIT(info->mode))) {
      /* A geometry shader consumes the input topology itself; the adjacency
       * primitives exist for nothing else. Converting one first is both wrong
       * and impossible -- u_index_prim_type_convert maps LINES_ADJACENCY to
       * LINES_ADJACENCY -- so primconvert handed the draw straight back with
       * the same mode and this branch took it again. That is unbounded
       * re-entry: KHR-GL32.primitive_restart drew LINES_ADJACENCY through a
       * geometry shader and overflowed the 16 MiB host stack, which aborts
       * the wasm module and loses every case after it. */
      if (ctx->gs) {
         gex_wgpu_gs_draw(ctx, info, drawid_offset, &draws[0]);
         return;
      }
      /* With no geometry shader the adjacent vertices are dropped and the
       * primitive is drawn in its base form. Only once the restart split
       * below has run, so that this sees a plain index list. */
      if (gex_wgpu_is_adjacency_prim(info->mode) && num_draws == 1 &&
          !(info->primitive_restart && info->index_size)) {
         if (!gex_wgpu_draw_adjacency(pipe, info, drawid_offset, &draws[0]))
            gex_fail(ctx);
         return;
      }
      if (gex_wgpu_is_adjacency_prim(info->mode) &&
          (num_draws > 1 || (info->primitive_restart && info->index_size)))
         ; /* fall through to the multi-draw and restart splits below */
      else {
      const enum mesa_prim converted = u_index_prim_type_convert(
         GEX_WGPU_PRIM_MODES, info->mode, info->index_size != 0);
      if (!ctx->primconvert ||
          !(GEX_WGPU_PRIM_MODES & BITFIELD_BIT(converted))) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr,
                    "gex-wgpu: draw declined (topology %u converts to %u, "
                    "which this driver does not have either)\n",
                    info->mode, converted);
         }
         /* Declined, not latched: latching becomes GL_OUT_OF_MEMORY in
          * st/mesa, which dEQP treats as fatal. */
         return;
      }
      util_primconvert_draw_vbo(ctx->primconvert, info, drawid_offset, indirect,
                                draws, num_draws);
      return;
      }
   }

   /* One render pass per draw, so a multi-draw is the same work in a loop.
    * util_draw_vbo_without_prim_restart below re-issues a restarting draw as
    * however many restart-free ranges it found, so this is not a rare path. */
   if (info && draws && num_draws > 1) {
      for (unsigned i = 0; i < num_draws && !ctx->failed; ++i)
         gex_wgpu_draw_vbo(pipe, info, drawid_offset, indirect, &draws[i], 1);
      return;
   }

   /* ARB_draw_indirect, resolved on the CPU.
    *
    * The draw arguments live in a buffer; util_draw_indirect maps it, pulls
    * them out and re-issues an ordinary draw, which lands back here with
    * `indirect` NULL. That re-entry is what ctx->draw_depth bounds, and it is
    * one of the two levels the guard's comment allows for.
    *
    * WebGPU could take these natively -- drawIndirect and drawIndexedIndirect
    * expect exactly GL's DrawArraysIndirectCommand and
    * DrawElementsIndirectCommand layouts, field for field -- and that would
    * avoid the readback stall this costs. It also drags in the buffer needing
    * WGPUBufferUsage_Indirect and a non-zero firstInstance needing the
    * indirect-first-instance feature, so the CPU form is what goes in first
    * and the native one is a later optimisation.
    *
    * count_from_stream_output is a different feature (DrawTransformFeedback)
    * that util_draw_indirect explicitly asserts against, so it is handled
    * just below rather than being passed on. */
   if (indirect && info && indirect->count_from_stream_output && !ctx->failed) {
      /* glDrawTransformFeedback: draw as many vertices as a previous capture
       * recorded. The target carries that total in bytes, and it is exact
       * because this driver sized the capture on the CPU -- so unlike a
       * hardware driver there is no counter buffer to read back and no stall
       * here at all. */
      const struct gex_wgpu_so_target *so =
         (const struct gex_wgpu_so_target *)indirect->count_from_stream_output;
      struct pipe_draw_info counted = *info;
      struct pipe_draw_start_count_bias counted_draw = {0};
      counted.index_size = 0;
      counted.index_bounds_valid = false;
      counted_draw.start = 0;
      counted_draw.count = so->stride ? so->written / so->stride : 0u;
      if (counted_draw.count)
         gex_wgpu_draw_vbo(pipe, &counted, drawid_offset, NULL, &counted_draw,
                           1);
      return;
   }
   if (indirect && info && !ctx->failed) {
      util_draw_indirect(pipe, info, drawid_offset, indirect);
      return;
   }

   /* Primitive restart, split into restart-free draws. WebGPU has restart
    * only for strip topologies and only at a fixed index, so the general GL
    * form -- any index, any topology -- has to be resolved on the CPU, which
    * is what this helper does: it scans the index buffer and re-issues the
    * draw as the ranges between restarts. */
   if (info && draws && num_draws == 1 && info->primitive_restart &&
       info->index_size && !ctx->failed) {
      if (util_draw_vbo_without_prim_restart(pipe, info, drawid_offset,
                                             indirect, &draws[0]) != PIPE_OK)
         gex_fail(ctx);
      return;
   }

   /* GL_RASTERIZER_DISCARD: nothing reaches the framebuffer, so a draw with
    * no colour target and no fragment shader is legal -- and is exactly the
    * shape a transform feedback pass takes. */
   const bool discard = ctx->rasterizer && ctx->rasterizer->discard;
   const char *refusal =
      ctx->failed                                   ? "context already failed"
      : !ctx->vs                                    ? "no vertex shader"
      : (!discard && !ctx->fs)                      ? "no fragment shader"
      /* A depth-only pass is legal in both APIs: GL renders shadow maps that
       * way, and WebGPU takes zero colour attachments beside a depth one. The
       * pipeline omits its fragment stage for these, which is the form WebGPU
       * guarantees to validate. */
      : (!discard && !ctx->color_count && !ctx->depth) ? "no colour target"
      : (!info || !draws)                           ? "no draw"
      : indirect                                    ? "indirect count from stream output"
      : num_draws != 1                              ? "multi-draw"
      /* gl_DrawID. util_draw_indirect hands each sub-draw of a
       * glMultiDraw*Indirect its own index here, so refusing every non-zero
       * offset refused multi-draw indirect entirely -- and a refusal latches
       * the context. It only changes the result for a shader that actually
       * reads gl_DrawID, so that is what is refused; anything else does not
       * care what the offset was. */
      : (drawid_offset != 0 && ctx->vs && ctx->vs->reads_draw_id)
                                                    ? "draw id offset"
      : !(GEX_WGPU_PRIM_MODES & BITFIELD_BIT(info->mode)) ? "topology"
      : (info->index_size != 0 && info->index_size != 2 &&
         info->index_size != 4)                     ? "index size"
      : info->primitive_restart                     ? "primitive restart"
      /* Only an indexed draw has a base vertex. st/mesa leaves index_bias set
       * from a previous draw on a non-indexed one, where it means nothing --
       * and refusing on it latched the context, which cost draw_buffers its
       * case through GL_OUT_OF_MEMORY. */
      : (info->index_size && draws[0].index_bias != 0) ? "index bias"
      : NULL;

   /* A draw of no vertices, or of no instances, is legal in GL and draws
    * nothing. Both used to be refusals, and a refusal latches the context --
    * so one empty sub-draw poisoned everything after it. glMultiDrawArrays*
    * and glMultiDrawElements* routinely carry a zero-count entry, and
    * glMultiDrawArraysIndirect hit exactly that: the whole call came back as
    * GL_OUT_OF_MEMORY because of a sub-draw that was supposed to be a no-op.
    * Returning early is the entire correct behaviour. */
   if (!refusal && info && draws &&
       (draws[0].count == 0 || info->instance_count == 0))
      return;
   if (refusal) {
      /* Named, not just located. This is the busiest refusal in the driver,
       * and a line number costs a build cycle every time something new is
       * rejected here. */
      if (!ctx->failed)
         fprintf(stderr,
                 "gex-wgpu: draw refused (%s) mode=%u count=%u isz=%u "
                 "instances=%u colours=%u vs=%d fs=%d\n",
                 refusal, info ? info->mode : 0,
                 draws ? draws[0].count : 0, info ? info->index_size : 0,
                 info ? info->instance_count : 0, ctx->color_count,
                 ctx->vs ? 1 : 0, ctx->fs ? 1 : 0);
      gex_fail(ctx);
      return;
   }
   /* Gallium does not require the bound vertex-buffer count to match what the
    * element state references: st/mesa and u_vbuf both leave higher slots
    * bound from an earlier draw, and u_vbuf binds its own translated buffer
    * above the application's.  What must hold is that every slot the elements
    * actually read from is bound.
    */
   if (!ctx->vertex_elements) {
      /* An attributeless draw is legitimate -- a shader generating its own
       * vertices from gl_VertexID, which is how a fullscreen pass is usually
       * written -- and leftover bound buffers do not make it otherwise, since
       * the paragraph above establishes that stale bindings are routine.
       *
       * What does matter is whether the shader reads an attribute nothing
       * supplies. Keying this on the bound buffers instead refused ordinary
       * attributeless draws; keying it on nothing at all let a shader that
       * does read attributes reach the device and fail validation, which
       * poisoned it for every later command. The shader's own inputs are the
       * only thing that answers the question. */
      /* The locations the module declares are what matters, and those are
       * supplied from the default-attribute buffer below when nothing else
       * does -- which is what GL means by reading a disabled array. Only a
       * shader with more of them than there is room for is refused.
       *
       * inputs_read is not the test: it is a VERT_ATTRIB_* mask that is empty
       * for exactly this case, so keying on it let a module declaring
       * @location(3) reach the device with no vertex state and fail
       * validation -- which poisons it for every later command. */
      if (ctx->vs && util_bitcount64(ctx->vs->vertex_locations) >
                        GEX_WGPU_MAX_VERTEX_ELEMENTS) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr,
                    "gex-wgpu: draw refused (vertex shader declares %u inputs "
                    "at 0x%llx with no vertex elements bound)\n",
                    util_bitcount64(ctx->vs->vertex_locations),
                    (unsigned long long)ctx->vs->vertex_locations);
         }
         gex_fail(ctx);
         return;
      }
   } else {
      for (unsigned i = 0; i < ctx->vertex_elements->count; ++i) {
         const unsigned slot =
            ctx->vertex_elements->elements[i].vertex_buffer_index;
         /* An element whose buffer slot is not bound is GL's *disabled*
          * vertex array: the attribute reads the current constant value, not
          * memory. st/mesa expresses that by leaving the slot empty while the
          * element stays in the layout, so failing here killed the context on
          * something entirely ordinary -- and silently, which made it the
          * hardest refusal in this driver to find: the only symptom was a
          * canvas that stopped changing while the frame counter kept rising.
          * The location is left out of `supplied_locations` below instead, so
          * the default-attribute buffer serves it. */
         if (slot >= ctx->vertex_buffer_count || !ctx->vertex_buffers[slot]) {
            static bool said;
            if (!said) {
               said = true;
               fprintf(stderr,
                       "gex-wgpu: vertex element %u reads its constant value "
                       "(buffer slot %u is %s)\n", i, slot,
                       slot >= ctx->vertex_buffer_count ? "past the bound count"
                                                        : "not bound");
            }
         }
      }
   }

   /* Gallium's vertex buffer slots renumbered onto WebGPU's.
    *
    * `vertex_buffer_index` indexes the application's bound buffers, and
    * st/mesa binds one per enabled vertex array -- SuperTuxKart binds nine.
    * WebGPU allows eight on a pipeline, and this driver spent the Gallium
    * index directly as the WebGPU slot, so the ninth bound buffer latched the
    * context whether or not anything read it.
    *
    * Numbering only the buffers that feed an attribute fixes both halves of
    * that: a bound-but-unread slot costs nothing, and the WebGPU limit lands
    * on eight *used* buffers rather than eight *bound* ones. It also settles a
    * latent null dereference below, where the bind loop used to hand every
    * slot up to `vertex_buffer_count` to SetVertexBuffer -- including the
    * empty ones a disabled vertex array leaves behind.
    *
    * The map is keyed on the step mode as well as the buffer, because one
    * Gallium buffer can feed both per-vertex and per-instance elements and a
    * `GPUVertexBufferLayout` carries a single `stepMode` for all of its
    * attributes. Keying on the buffer alone collapsed them into one slot and
    * took the step mode from whichever element came first, so SuperTuxKart's
    * instanced objects -- which put `i_position`/`i_normal` (per vertex) and
    * `i_origin`/`i_rotation`/`i_scale` (per instance) in one buffer -- had
    * their instance attributes fetched with the vertex index. Their shader
    * computes `i_position * i_scale + i_origin`, so a zeroed scale collapses
    * every object to a point and none of the scene drew at all.
    *
    * Splitting them costs nothing: WebGPU is happy to see the same GPUBuffer
    * bound to two slots, which is exactly what this is. */
   int8_t vb_slot[GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS][2];
   uint8_t vb_buffer[GEX_WGPU_MAX_VERTEX_BUFFERS] = {0};
   bool vb_instanced[GEX_WGPU_MAX_VERTEX_BUFFERS] = {0};
   unsigned vb_slot_count = 0;
   memset(vb_slot, -1, sizeof(vb_slot));
   if (ctx->vertex_elements) {
      /* The same three tests the attribute loop applies further down, in the
       * same order: an element the shader has no location for, or whose slot
       * is empty, supplies nothing and must not claim a slot. */
      const bool by_element = !ctx->vs || ctx->vs->vertex_locations == 0;
      for (unsigned i = 0; i < ctx->vertex_elements->count; ++i) {
         if (!by_element &&
             (i >= GEX_WGPU_MAX_VERTEX_ELEMENTS ||
              ctx->vs->vertex_element_location[i] == 0xff))
            continue;
         const unsigned buffer =
            ctx->vertex_elements->elements[i].vertex_buffer_index;
         const unsigned step =
            ctx->vertex_elements->elements[i].instance_divisor ? 1u : 0u;
         if (buffer >= ctx->vertex_buffer_count ||
             buffer >= GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS ||
             !ctx->vertex_buffers[buffer] || vb_slot[buffer][step] >= 0)
            continue;
         if (vb_slot_count >= GEX_WGPU_MAX_VERTEX_BUFFERS) {
            /* Declined, not latched: this draw needs more vertex buffers than
             * WebGPU will take, and the next one probably does not. */
            ++gex_wgpu_draws_declined;
            ++gex_wgpu_declined_vertex;
            gex_wgpu_declined_last_why = "more vertex buffers than WebGPU takes";
            static bool said;
            if (!said) {
               said = true;
               fprintf(stderr,
                       "gex-wgpu: draw declined (more than %u vertex buffers "
                       "feed attributes; %u bound)\n",
                       (unsigned)GEX_WGPU_MAX_VERTEX_BUFFERS,
                       ctx->vertex_buffer_count);
            }
            return;
         }
         vb_buffer[vb_slot_count] = (uint8_t)buffer;
         vb_instanced[vb_slot_count] = step != 0;
         vb_slot[buffer][step] = (int8_t)vb_slot_count++;
      }
   }

   /* Primitive counting for the transform-feedback queries. Counted before
    * anything below can decline the draw, because GL counts what the vertex
    * stage produced whether or not this backend could render it. */
   if (ctx->active_query && draws) {
      const unsigned prims =
         u_prims_for_vertices(info->mode, draws[0].count) *
         MAX2(1u, info->instance_count);
      switch (ctx->active_query->type) {
      case PIPE_QUERY_PRIMITIVES_GENERATED:
         ctx->active_query->result += prims;
         break;
      default:
         /* PRIMITIVES_EMITTED and SO_STATISTICS are counted by the capture
          * pass, which is the only thing that knows how much of the draw
          * actually fit in the bound buffers. */
         break;
      }
   }

   /* A geometry shader replaces this draw with three passes of its own, the
    * last of which comes back through here with the passthrough shader
    * bound. */
   if (ctx->gs) {
      /* Transform feedback with a geometry shader captures nothing.
       *
       * GL captures the last pre-rasterization stage, which is the geometry
       * shader; this returns before the capture block below, and that block
       * would run the *vertex* shader anyway. So the feedback buffers keep
       * whatever they held -- silently, which is the part worth saying out
       * loud. It is what fails the five capture_geometry_* / query_geometry_*
       * / discard_geometry cases in KHR-GL33.transform_feedback.
       *
       * Rendering is left alone rather than refused: the draw itself is
       * correct, and an application that binds feedback buffers around a
       * geometry draw would otherwise lose its image as well as its capture.
       *
       * Doing it properly means capturing from the geometry pass, which
       * already runs as compute and could write the buffers directly -- it
       * needs the emitted-vertex count (dynamic, unlike the vertex stage's)
       * and a primitives-emitted count the driver does not model today. */
      if (ctx->so_target_count) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr, "gex-wgpu: transform feedback with a geometry "
                            "shader is not captured; the feedback buffers "
                            "are left unchanged\n");
         }
      }
      gex_wgpu_gs_draw(ctx, info, drawid_offset, &draws[0]);
      return;
   }

   /* Transform feedback is its own compute pass, run before the draw: WGSL
    * will not let the vertex stage write a storage buffer, so the vertex
    * shader is run a second time as compute to do the capturing. */
   if (ctx->so_target_count && ctx->vs->capture_nir &&
       !gex_wgpu_capture_dispatch(ctx, info, &draws[0]))
      return;
   if (discard)
      return;

   struct pipe_resource *index_resource = NULL;
   struct pipe_resource *uploaded_index_resource = NULL;
   unsigned index_upload_offset = 0;
   if (info->index_size) {
      if (info->has_user_indices) {
         /* glDrawElements from a client array. u_upload_mgr stages it into a
          * real buffer, which is the only form WebGPU can bind -- latching
          * here instead ended the run, through st/mesa's GL_OUT_OF_MEMORY. */
         pipe_resource_reference(&ctx->uploaded_indices, NULL);
         if (!util_upload_index_buffer(&ctx->base, info, &draws[0],
                                       &ctx->uploaded_indices,
                                       &index_upload_offset, 4)) {
            gex_fail(ctx);
            return;
         }
         uploaded_index_resource = ctx->uploaded_indices;
         index_resource = uploaded_index_resource;
      } else {
         index_resource = info->index.resource;
      }
      if (!index_resource || index_resource->target != PIPE_BUFFER ||
          !(index_resource->bind & PIPE_BIND_INDEX_BUFFER) ||
          (uint64_t)index_upload_offset +
             (uint64_t)(uploaded_index_resource ? 0 : draws[0].start) *
                info->index_size +
             (uint64_t)draws[0].count * info->index_size >
                index_resource->width0) {
         gex_fail(ctx);
         return;
      }
   }
   /* Where this draw's first index sits. An uploaded buffer already begins at
    * the draw's start, so the byte offset carries it and the index of the
    * first element is zero. */
   const unsigned first_index = uploaded_index_resource ? 0u : draws[0].start;
   const unsigned index_byte_offset =
      uploaded_index_resource ? index_upload_offset : 0u;
   const int64_t t_bounds_start = gex_timing ? os_time_get_nano() : 0;
   if (ctx->vertex_elements) {
      uint64_t last_vertex = (uint64_t)draws[0].start + draws[0].count - 1;
      if (info->index_size) {
         last_vertex = 0;
         gex_wgpu_mark_shadow_read(gex_resource(index_resource));
         const uint8_t *indices = gex_resource(index_resource)->shadow +
                                  index_byte_offset +
                                  (uint64_t)first_index * info->index_size;
         for (unsigned i = 0; i < draws[0].count; ++i) {
            uint32_t index;
            if (info->index_size == 2)
               memcpy(&index, indices + (uint64_t)i * 2, 2);
            else
               memcpy(&index, indices + (uint64_t)i * 4, 4);
            if (info->index_size == 2)
               index &= 0xffffu;
            last_vertex = MAX2(last_vertex, index);
         }
      }
      for (unsigned i = 0; i < ctx->vertex_elements->count; ++i) {
         const struct pipe_vertex_element *element =
            &ctx->vertex_elements->elements[i];
         const unsigned buffer = element->vertex_buffer_index;
         /* A disabled array has no buffer to overrun, and reading `width0`
          * through the empty slot is what this used to do: a garbage bound,
          * a failed range check, and a latched context roughly ten seconds
          * in -- after which every frame presents the clear colour and
          * nothing else, forever. The layout code above already routes these
          * to the default-attribute buffer. */
         if (buffer >= ctx->vertex_buffer_count || !ctx->vertex_buffers[buffer])
            continue;
         const uint64_t last_element = element->instance_divisor
            ? (uint64_t)info->start_instance + info->instance_count - 1
            : last_vertex;
         unsigned size, alignment;
         (void)gex_wgpu_vertex_format(element->src_format, &size, &alignment);
         const uint64_t end = (uint64_t)ctx->vertex_buffer_offsets[buffer] +
            element->src_offset + last_element * element->src_stride + size;
         if (end > ctx->vertex_buffers[buffer]->width0) {
            /* Counted, and reported every thousandth: whether this is one
             * stray draw or every draw of some object is the whole question,
             * and a one-shot message cannot tell the two apart. WebGPU returns
             * zeros for the overrun, so a position attribute read this way
             * collapses its vertices onto the origin -- which on screen is a
             * single triangle stretched across the frame. */
            static unsigned seen;
            if ((seen++ % 1000) == 0) {
               fprintf(stderr,
                       "gex-wgpu: vertex element reads past its buffer x%u "
                       "(element %u reads to %llu, "
                       "buffer slot %u is %u bytes) offset=%u src_offset=%u "
                       "stride=%u last=%llu size=%u divisor=%u bind=0x%x "
                       "draw=%u%s fmt=%d alloc=%llu gallium_off=%u user=%d "
                       "elements=%u bounds=%d/%u..%u bias=%d start=%u\n",
                       seen, i, (unsigned long long)end, buffer,
                       ctx->vertex_buffers[buffer]->width0,
                       ctx->vertex_buffer_offsets[buffer], element->src_offset,
                       element->src_stride,
                       (unsigned long long)last_element, size,
                       element->instance_divisor,
                       ctx->vertex_buffers[buffer]->bind,
                       draws[0].count, info->index_size ? " indexed" : "",
                       (int)element->src_format,
                       (unsigned long long)
                          gex_resource(ctx->vertex_buffers[buffer])
                             ->allocation_size,
                       ctx->vertex_buffer_state[buffer].buffer_offset,
                       (int)ctx->vertex_buffer_state[buffer].is_user_buffer,
                       ctx->vertex_elements->count,
                       /* Gallium's own claim about the index range, against
                        * this driver's scan of the index buffer's CPU shadow.
                        * If they disagree the shadow is stale and the whole
                        * report is a false alarm; if they agree the buffer
                        * really is too small for the draw. */
                       (int)info->index_bounds_valid, info->min_index,
                       info->max_index, draws[0].index_bias, draws[0].start);
            }
            /* Not fatal, and -- measured -- not visible either.
             *
             * GL calls an out-of-range fetch undefined but not an error, and
             * WebGPU is robust by construction: a vertex read past the end of
             * a bound buffer yields zeros rather than faulting. Refusing here
             * was stricter than either API and cost the whole context.
             *
             * It is also not the cause of any artifact, which was worth
             * establishing because the numbers look alarming: SuperTuxKart
             * trips this on about a quarter of its draws, always the same
             * shape -- eight elements describing one 44-byte vertex, a
             * 44-byte buffer, and an indexed draw reaching vertex nine.
             * Gallium agrees the draw needs ten (`index_bounds_valid` with
             * `max_index` 9), so the buffer really is one vertex short of what
             * is being drawn from it. Declining those draws as an experiment
             * changed nothing on screen except to delete the HUD, which they
             * were drawing correctly: the attribute that overruns is a packed
             * R16G16_SNORM normal at offset 36 that those shaders do not use.
             * So this is a real out-of-range read with no consequence, and
             * chasing it further is chasing the wrong thing. */
         }
      }
   }
   if (gex_timing) gex_wgpu_time_bounds += os_time_get_nano() - t_bounds_start;

   float viewport_x = 0.0f;
   float viewport_y = 0.0f;
   float viewport_width = ctx->fb_width;
   float viewport_height = ctx->fb_height;
   float min_depth = 0.0f;
   float max_depth = 1.0f;
   bool vertex_flipped = false;
   /* Filled in when the GL viewport cannot be given to WebGPU directly and is
    * folded into the vertex shader instead. */
   float viewport_fold[4] = {0.0f, 0.0f, 0.0f, 0.0f};
   bool fold_viewport = false;
   if (ctx->viewport_set) {
      const float sx = ctx->viewport.scale[0];
      const float sy = ctx->viewport.scale[1];
      const float sz = ctx->viewport.scale[2];
      viewport_x = ctx->viewport.translate[0] - sx;
      viewport_y = ctx->viewport.translate[1] - fabsf(sy);
      viewport_width = 2.0f * sx;
      viewport_height = 2.0f * fabsf(sy);
      min_depth = ctx->viewport.translate[2] - sz;
      max_depth = ctx->viewport.translate[2] + sz;
      if (sy > 0.0f)
         vertex_flipped = true;
      /* WebGPU takes a depth range inside [0, 1]. glDepthRange clamps to that
       * already, so a viewport outside it never came from an application:
       * Mesa's internal helpers set the conventional full-range transform
       * -- u_blitter uses scale 1, translate 0 -- because the range only
       * matters to a draw that writes depth, and theirs do not. Clamping is
       * what every other driver's viewport setup does with it. */
      if (min_depth < 0.0f || max_depth > 1.0f) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr,
                    "gex-wgpu: clamping viewport depth %.3f..%.3f into "
                    "[0, 1]\n", min_depth, max_depth);
         }
         min_depth = CLAMP(min_depth, 0.0f, 1.0f);
         max_depth = CLAMP(max_depth, 0.0f, 1.0f);
      }
      /* GL allows a viewport that reaches outside the framebuffer and simply
       * clips; WebGPU's setViewport requires the rectangle to be inside the
       * attachment, and passing one that is not is a validation error, which
       * poisons the device. The transform is affine, so it is folded into
       * gl_Position instead and WebGPU is given the whole framebuffer --
       * which is what a driver on a clipped-viewport API does. */
      if (viewport_x < 0.0f || viewport_y < 0.0f ||
          viewport_x + viewport_width > ctx->fb_width ||
          viewport_y + viewport_height > ctx->fb_height) {
         const float full_width = (float)ctx->fb_width;
         const float full_height = (float)ctx->fb_height;
         if (full_width <= 0.0f || full_height <= 0.0f ||
             viewport_width <= 0.0f || viewport_height <= 0.0f) {
            static bool said;
            if (!said) {
               said = true;
               fprintf(stderr,
                       "gex-wgpu: draw declined (degenerate viewport) "
                       "%.1f,%.1f %.1fx%.1f fb=%ux%u\n", viewport_x,
                       viewport_y, viewport_width, viewport_height,
                       ctx->fb_width, ctx->fb_height);
            }
            return;
         }
         viewport_fold[0] = viewport_width / full_width;
         viewport_fold[1] = viewport_height / full_height;
         viewport_fold[2] =
            (2.0f * viewport_x + viewport_width) / full_width - 1.0f;
         viewport_fold[3] =
            (2.0f * viewport_y + viewport_height) / full_height - 1.0f;
         fold_viewport = true;
         viewport_x = 0.0f;
         viewport_y = 0.0f;
         viewport_width = full_width;
         viewport_height = full_height;
      }
      if (min_depth > max_depth) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr,
                    "gex-wgpu: draw declined (inverted depth range "
                    "%.3f..%.3f)\n", min_depth, max_depth);
         }
         return;
      }
   }

   bool scissor_enabled = ctx->rasterizer && ctx->rasterizer->scissor;
   uint32_t scissor_x = 0, scissor_y = 0;
   uint32_t scissor_width = ctx->fb_width, scissor_height = ctx->fb_height;
   if (scissor_enabled) {
      if (!ctx->scissor_set) {
         if (!ctx->failed)
            fprintf(stderr, "gex-wgpu: draw refused (scissor enabled, "
                            "no scissor set)\n");
         gex_fail(ctx);
         return;
      }
      scissor_x = MIN2(ctx->scissor.minx, ctx->fb_width);
      scissor_y = MIN2(ctx->scissor.miny, ctx->fb_height);
      const uint32_t maxx = MIN2(ctx->scissor.maxx, ctx->fb_width);
      const uint32_t maxy = MIN2(ctx->scissor.maxy, ctx->fb_height);
      if (maxx <= scissor_x || maxy <= scissor_y)
         return;
      scissor_width = maxx - scissor_x;
      scissor_height = maxy - scissor_y;
   }

   struct gex_wgpu_screen *screen = gex_screen(pipe->screen);
   WGPUColorTargetState color_targets[GEX_WGPU_MAX_COLOR_TARGETS];
   WGPUBlendState blend_states[GEX_WGPU_MAX_COLOR_TARGETS];
   bool uses_blend_constant = false;
   for (unsigned i = 0; i < ctx->color_count; ++i) {
      color_targets[i] =
         (WGPUColorTargetState)WGPU_COLOR_TARGET_STATE_INIT;
      color_targets[i].format =
         gex_wgpu_texture_format(ctx->color_formats[i]);
      /* Undefined is how WebGPU names a null target, and a null target takes
       * neither a write mask nor a blend state. */
      if (color_targets[i].format == WGPUTextureFormat_Undefined)
         continue;
      if (!ctx->blend)
         continue;
      const struct pipe_rt_blend_state *rt =
         &ctx->blend->rt[ctx->blend->independent_blend_enable ? i : 0];
      color_targets[i].writeMask = 0;
      if (rt->colormask & PIPE_MASK_R)
         color_targets[i].writeMask |= WGPUColorWriteMask_Red;
      if (rt->colormask & PIPE_MASK_G)
         color_targets[i].writeMask |= WGPUColorWriteMask_Green;
      if (rt->colormask & PIPE_MASK_B)
         color_targets[i].writeMask |= WGPUColorWriteMask_Blue;
      if (rt->colormask & PIPE_MASK_A)
         color_targets[i].writeMask |= WGPUColorWriteMask_Alpha;
      if (!rt->blend_enable)
         continue;
      blend_states[i] = (WGPUBlendState)WGPU_BLEND_STATE_INIT;
      blend_states[i].color.operation = gex_wgpu_blend_operation(rt->rgb_func);
      blend_states[i].color.srcFactor =
         gex_wgpu_blend_factor(rt->rgb_src_factor);
      blend_states[i].color.dstFactor =
         gex_wgpu_blend_factor(rt->rgb_dst_factor);
      blend_states[i].alpha.operation =
         gex_wgpu_blend_operation(rt->alpha_func);
      blend_states[i].alpha.srcFactor =
         gex_wgpu_blend_factor(rt->alpha_src_factor);
      blend_states[i].alpha.dstFactor =
         gex_wgpu_blend_factor(rt->alpha_dst_factor);
      color_targets[i].blend = &blend_states[i];
      uses_blend_constant |=
         rt->rgb_src_factor == PIPE_BLENDFACTOR_CONST_COLOR ||
         rt->rgb_src_factor == PIPE_BLENDFACTOR_INV_CONST_COLOR ||
         rt->rgb_dst_factor == PIPE_BLENDFACTOR_CONST_COLOR ||
         rt->rgb_dst_factor == PIPE_BLENDFACTOR_INV_CONST_COLOR ||
         rt->alpha_src_factor == PIPE_BLENDFACTOR_CONST_ALPHA ||
         rt->alpha_src_factor == PIPE_BLENDFACTOR_INV_CONST_ALPHA ||
         rt->alpha_dst_factor == PIPE_BLENDFACTOR_CONST_ALPHA ||
         rt->alpha_dst_factor == PIPE_BLENDFACTOR_INV_CONST_ALPHA;
   }
   WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
   const int64_t t_fsmod_start = gex_timing ? os_time_get_nano() : 0;
   fragment.module = gex_wgpu_fragment_module(ctx);
   if (gex_timing) gex_wgpu_time_fsmod += os_time_get_nano() - t_fsmod_start;
   if (!fragment.module)
      return;
   fragment.entryPoint = (WGPUStringView){"main", WGPU_STRLEN};
   fragment.targetCount = ctx->color_count;
   fragment.targets = color_targets;
   WGPURenderPipelineDescriptor pipeline_desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
   WGPUShaderModule vertex_module = gex_wgpu_interp_variant_get(
      ctx, vertex_flipped, fold_viewport ? viewport_fold : NULL);
   if (!vertex_module)
      return;
   pipeline_desc.vertex.module = vertex_module;
   pipeline_desc.vertex.entryPoint = (WGPUStringView){"main", WGPU_STRLEN};
   WGPUVertexAttribute
      vertex_attributes[GEX_WGPU_MAX_VERTEX_BUFFERS][GEX_WGPU_MAX_VERTEX_ELEMENTS];
   WGPUVertexBufferLayout vertex_layouts[GEX_WGPU_MAX_VERTEX_BUFFERS];
   int default_attribute_slot = -1;
   uint64_t supplied_locations = 0;
   if (ctx->vertex_elements) {
      /* Indexed by the dense WebGPU slot `vb_slot` assigned, not by Gallium's
       * buffer index -- see the comment where that map is built. */
      unsigned attribute_counts[GEX_WGPU_MAX_VERTEX_BUFFERS] = {0};
      for (unsigned slot = 0; slot < vb_slot_count; ++slot)
         vertex_layouts[slot] =
            (WGPUVertexBufferLayout)WGPU_VERTEX_BUFFER_LAYOUT_INIT;
      /* The element index is only the @location when the shader's generic
       * attributes start at zero; the shader's own table is the answer, and
       * an element with no entry there feeds nothing and is left out. */
      const bool by_element = ctx->vs->vertex_locations == 0;
      for (unsigned i = 0; i < ctx->vertex_elements->count; ++i) {
         const struct pipe_vertex_element *element =
            &ctx->vertex_elements->elements[i];
         unsigned location = i;
         if (!by_element) {
            if (i >= GEX_WGPU_MAX_VERTEX_ELEMENTS ||
                ctx->vs->vertex_element_location[i] == 0xff)
               continue;
            location = ctx->vs->vertex_element_location[i];
         }
         const unsigned buffer = element->vertex_buffer_index;
         /* Disabled array: no buffer behind this element, so it supplies no
          * location and falls to the default-attribute buffer below. The slot
          * map already dropped those, so an unmapped buffer is exactly that
          * case and needs no second test. */
         const unsigned step = element->instance_divisor ? 1u : 0u;
         if (buffer >= GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS ||
             vb_slot[buffer][step] < 0)
            continue;
         const unsigned slot = (unsigned)vb_slot[buffer][step];
         const unsigned attribute = attribute_counts[slot]++;
         vertex_attributes[slot][attribute] =
            (WGPUVertexAttribute)WGPU_VERTEX_ATTRIBUTE_INIT;
         unsigned size, alignment;
         vertex_attributes[slot][attribute].format = gex_wgpu_vertex_format(
            element->src_format, &size, &alignment);
         vertex_attributes[slot][attribute].offset = element->src_offset;
         vertex_attributes[slot][attribute].shaderLocation = location;
         supplied_locations |= BITFIELD64_BIT(location);
      }
      for (unsigned slot = 0; slot < vb_slot_count; ++slot) {
         const unsigned buffer = vb_buffer[slot];
         const bool instanced = vb_instanced[slot];
         /* The stride has to come from an element of this slot's own step
          * mode: the per-vertex and per-instance halves of one buffer have
          * different strides, and taking the first element's regardless is how
          * the instance data came out wrong. */
         for (unsigned i = 0; i < ctx->vertex_elements->count; ++i) {
            const struct pipe_vertex_element *e =
               &ctx->vertex_elements->elements[i];
            if (e->vertex_buffer_index == buffer &&
                (e->instance_divisor != 0) == instanced) {
               vertex_layouts[slot].arrayStride = e->src_stride;
               break;
            }
         }
         vertex_layouts[slot].stepMode = instanced
            ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
         vertex_layouts[slot].attributeCount = attribute_counts[slot];
         vertex_layouts[slot].attributes = vertex_attributes[slot];
      }
      pipeline_desc.vertex.bufferCount = vb_slot_count;
      pipeline_desc.vertex.buffers = vertex_layouts;
   }
   /* Every @location the module declares that no element supplies. The element
    * index is the shaderLocation above, so the elements cover exactly
    * locations 0..count-1 -- and a draw with no element state at all supplies
    * none of them, which is the shape this reached first: st/mesa binds no
    * vertex elements for a shader whose arrays are all disabled, and the
    * pipeline was then built with no vertex state while the module still
    * declared its inputs. */
   {
      const uint64_t missing = ctx->vs->vertex_locations & ~supplied_locations;
      /* Which shader inputs a real vertex buffer feeds, and which fall through
       * to the default-attribute buffer's fixed (0, 0, 0, 1).
       *
       * If location 0 -- position, for every shader st builds -- is in
       * `missing`, every vertex of the draw collapses onto the same clip-space
       * point and the primitive has no area. That is indistinguishable on
       * screen from "the draw never happened", which is what most of this
       * application's draws look like. */
      {
         /* Once per distinct shader shape, not per draw: the first eight
          * draws of a frame are all the same two-attribute UI quad, and say
          * nothing about the geometry that matters. */
         static uint64_t seen_shapes[24];
         static unsigned seen_count;
         bool fresh = false;
         if (ctx->vertex_elements) {
            /* Keyed on the shader itself, not on its input mask and element
             * count: two different shaders share those often enough that the
             * one responsible for most of the frame -- 53,240 draws of 91,000
             * -- was hidden behind a lighter one with the same shape, and the
             * attributes dumped were of a shader that draws 1,971 times. */
            const uint64_t shape = (uint64_t)(uintptr_t)ctx->vs;
            fresh = true;
            for (unsigned i = 0; i < seen_count; ++i)
               if (seen_shapes[i] == shape) { fresh = false; break; }
            if (fresh && seen_count < 24)
               seen_shapes[seen_count++] = shape;
            else if (fresh)
               fresh = false;
         }
         if (fresh) {
            fprintf(stderr,
                    "gex-wgpu: vs %p inputs 0x%llx supplied 0x%llx missing 0x%llx "
                    "(elements=%u slots=%u) vp=%.0f,%.0f %.0fx%.0f depth=%.2f..%.2f "
                    "flip=%d fold=%d scissor=%d %d,%d %dx%d fb=%ux%u "
                    "depthtest=%d func=%d write=%d cfmt=%d srgb=%d\n",
                    (void *)ctx->vs,
                    (unsigned long long)ctx->vs->vertex_locations,
                    (unsigned long long)supplied_locations,
                    (unsigned long long)missing,
                    ctx->vertex_elements->count, vb_slot_count,
                    viewport_x, viewport_y, viewport_width, viewport_height,
                    min_depth, max_depth, (int)vertex_flipped,
                    (int)fold_viewport, (int)scissor_enabled,
                    scissor_x, scissor_y, scissor_width, scissor_height,
                    ctx->fb_width, ctx->fb_height,
                    ctx->dsa ? (int)ctx->dsa->depth_enabled : -1,
                    ctx->dsa ? (int)ctx->dsa->depth_func : -1,
                    ctx->dsa ? (int)ctx->dsa->depth_writemask : -1,
                    ctx->color_count ? (int)ctx->color_formats[0] : -1,
                    ctx->color_count
                       ? (int)util_format_is_srgb(ctx->color_formats[0]) : -1);
            /* The textures this draw samples, with their formats and sizes.
             * A washed-out frame is as likely to come from an albedo texture
             * whose sRGB decode is not happening -- sampled as linear, every
             * midtone reads brighter than it is -- as from the lighting, and
             * the format is the only thing that distinguishes them. A 1x1
             * texture here would mean a material fell back to a default. */
            for (unsigned t = 0; t < GEX_WGPU_MAX_SAMPLERS; ++t) {
               const struct pipe_sampler_view *sv = ctx->sampler_views[t];
               if (!sv || !sv->texture)
                  continue;
               fprintf(stderr,
                       "gex-wgpu:   tex%u %s %ux%u levels=%u srgb=%d "
                       "viewfmt=%s\n", t,
                       util_format_short_name(sv->texture->format),
                       sv->texture->width0, sv->texture->height0,
                       sv->texture->last_level + 1,
                       (int)util_format_is_srgb(sv->format),
                       util_format_short_name(sv->format));
            }
            /* The fragment stage's constants too. SuperTuxKart's ambient is a
             * set of spherical-harmonic coefficients in a uniform block, and a
             * blown-out frame is as likely to come from those as from a
             * colour-space mistake. */
            for (unsigned u = 0; ctx->fs && u < ctx->fs->ubo_count; ++u) {
               const unsigned cb = ctx->fs->ubo_slots[u];
               const struct pipe_constant_buffer *pcb =
                  &ctx->constant_buffers[MESA_SHADER_FRAGMENT][cb];
               if (!pcb->buffer || !gex_resource(pcb->buffer)->shadow) {
                  fprintf(stderr, "gex-wgpu:   fs cb%u unbound\n", cb);
                  continue;
               }
               const struct gex_wgpu_resource *r = gex_resource(pcb->buffer);
               const unsigned n = MIN2(pcb->buffer_size, 64u) / 4;
               char row[192] = "";
               for (unsigned f = 0; f < n; ++f) {
                  float x;
                  memcpy(&x, r->shadow + pcb->buffer_offset + f * 4, 4);
                  snprintf(row + strlen(row), sizeof(row) - strlen(row),
                           "%.3f ", x);
               }
               fprintf(stderr, "gex-wgpu:   fs cb%u %uB: %s\n", cb,
                       pcb->buffer_size, row);
            }
            /* The raw first vertex, as bytes and as f32. `i_scale` decoded to
             * zero as halves while its byte pattern looked like f32 1.0
             * straddling the field, so the question is where the real fields
             * begin -- and only the unstructured bytes answer that. */
            if (ctx->vertex_elements->count &&
                ctx->vertex_elements->elements[0].vertex_buffer_index <
                   GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS) {
               const unsigned b0 =
                  ctx->vertex_elements->elements[0].vertex_buffer_index;
               const unsigned stride0 =
                  ctx->vertex_elements->elements[0].src_stride;
               if (ctx->vertex_buffers[b0] &&
                   gex_resource(ctx->vertex_buffers[b0])->shadow &&
                   ctx->vertex_buffer_offsets[b0] + stride0 <=
                      ctx->vertex_buffers[b0]->width0) {
                  const uint8_t *v0 =
                     gex_resource(ctx->vertex_buffers[b0])->shadow +
                     ctx->vertex_buffer_offsets[b0];
                  for (unsigned at = 0; at + 16 <= stride0; at += 16) {
                     float f[4];
                     memcpy(f, v0 + at, sizeof(f));
                     fprintf(stderr,
                             "gex-wgpu:   v0[%2u] %02x%02x%02x%02x %02x%02x%02x%02x "
                             "%02x%02x%02x%02x %02x%02x%02x%02x = "
                             "%.3f %.3f %.3f %.3f\n", at,
                             v0[at+0], v0[at+1], v0[at+2], v0[at+3],
                             v0[at+4], v0[at+5], v0[at+6], v0[at+7],
                             v0[at+8], v0[at+9], v0[at+10], v0[at+11],
                             v0[at+12], v0[at+13], v0[at+14], v0[at+15],
                             f[0], f[1], f[2], f[3]);
                  }
               }
            }
            /* `i_scale` across the first few vertices, decoded as the halves
             * it actually is.
             *
             * The shader computes `i_position.xyz * i_scale.xyz + i_origin`,
             * so a zero scale collapses every vertex of an object onto its
             * origin and the primitives have no area -- which rasterises
             * nothing, with every input and the matrix still perfectly valid.
             * Printed as f32 this field looked like noise and was skipped. */
            for (unsigned e = 0; e < ctx->vertex_elements->count; ++e) {
               const struct pipe_vertex_element *el =
                  &ctx->vertex_elements->elements[e];
               if (el->src_format != PIPE_FORMAT_R16G16B16A16_FLOAT)
                  continue;
               const unsigned b = el->vertex_buffer_index;
               if (b >= GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS ||
                   !ctx->vertex_buffers[b] ||
                   !gex_resource(ctx->vertex_buffers[b])->shadow)
                  break;
               for (unsigned v = 0; v < 4; ++v) {
                  const uint64_t at = ctx->vertex_buffer_offsets[b] +
                     el->src_offset + (uint64_t)v * el->src_stride;
                  if (at + 8 > ctx->vertex_buffers[b]->width0) {
                     fprintf(stderr, "gex-wgpu:   el%u v%u past end\n", e, v);
                     break;
                  }
                  const uint8_t *p8 =
                     gex_resource(ctx->vertex_buffers[b])->shadow + at;
                  uint16_t h[4];
                  memcpy(h, p8, sizeof(h));
                  fprintf(stderr,
                          "gex-wgpu:   el%u(half4) v%u raw %04x %04x %04x %04x"
                          " = %.4f %.4f %.4f %.4f\n", e, v,
                          h[0], h[1], h[2], h[3],
                          _mesa_half_to_float(h[0]), _mesa_half_to_float(h[1]),
                          _mesa_half_to_float(h[2]), _mesa_half_to_float(h[3]));
               }
               break;
            }
            /* Each vertex element, with the shader location it feeds and the
             * bytes the first vertex would actually fetch. The scene shader
             * computes its position as `i_position * i_scale + i_origin`, so
             * whether those arrive as sensible numbers is the whole question,
             * and nothing upstream of the fetch can answer it. */
            for (unsigned e = 0; e < ctx->vertex_elements->count; ++e) {
               const struct pipe_vertex_element *el =
                  &ctx->vertex_elements->elements[e];
               const unsigned b = el->vertex_buffer_index;
               const int loc = (ctx->vs && e < GEX_WGPU_MAX_VERTEX_ELEMENTS)
                  ? (int)ctx->vs->vertex_element_location[e] : -1;
               char val[96] = "";
               if (b < GEX_WGPU_MAX_BOUND_VERTEX_BUFFERS &&
                   ctx->vertex_buffers[b] &&
                   gex_resource(ctx->vertex_buffers[b])->shadow) {
                  const uint8_t *base =
                     gex_resource(ctx->vertex_buffers[b])->shadow +
                     ctx->vertex_buffer_offsets[b] + el->src_offset;
                  const uint64_t end = ctx->vertex_buffer_offsets[b] +
                     el->src_offset + 16;
                  if (end <= ctx->vertex_buffers[b]->width0)
                     for (unsigned c = 0; c < 4; ++c) {
                        float f;
                        memcpy(&f, base + c * 4, 4);
                        snprintf(val + strlen(val), sizeof(val) - strlen(val),
                                 "%.3f ", f);
                     }
                  else
                     snprintf(val, sizeof(val), "(past end)");
               } else {
                  snprintf(val, sizeof(val), "(unbound)");
               }
               fprintf(stderr,
                       "gex-wgpu:   el%u -> @location(%d) buf%u off=%u "
                       "stride=%u fmt=%d div=%u v0: %s\n",
                       e, loc, b, el->src_offset, el->src_stride,
                       (int)el->src_format, el->instance_divisor, val);
            }
            /* Every constant buffer this vertex stage actually reads, not just
             * the first one seen at bind time. The buffers checked earlier
             * held a model matrix; a view-projection matrix lives somewhere
             * else, and if it arrives wrong every scene vertex leaves the
             * frustum while the skybox -- one attribute, no depth test, its
             * own matrix path -- still lands. */
            for (unsigned u = 0; u < ctx->vs->ubo_count; ++u) {
               const unsigned cb = ctx->vs->ubo_slots[u];
               const struct pipe_constant_buffer *pcb =
                  &ctx->constant_buffers[MESA_SHADER_VERTEX][cb];
               if (!pcb->buffer) {
                  fprintf(stderr, "gex-wgpu:   vs cb%u unbound\n", cb);
                  continue;
               }
               const struct gex_wgpu_resource *r = gex_resource(pcb->buffer);
               if (!r->shadow) {
                  fprintf(stderr, "gex-wgpu:   vs cb%u no shadow\n", cb);
                  continue;
               }
               /* Four vec4 rows at a time, so several matrices are legible
                * in sequence rather than one line of 146 numbers. */
               /* Out to byte 384: the shader reads its transform from
                * `field0[16..19]`, which is bytes 256..320, and a dump that
                * stopped at 256 showed four sane matrices while saying nothing
                * about the one actually used. */
               const unsigned n = MIN2(pcb->buffer_size, 384u) / 4;
               fprintf(stderr, "gex-wgpu:   vs cb%u %uB written=%llu\n", cb,
                       pcb->buffer_size, (unsigned long long)r->bytes_written);
               for (unsigned f = 0; f < n; f += 4) {
                  float x[4] = {0, 0, 0, 0};
                  for (unsigned c = 0; c < 4 && f + c < n; ++c)
                     memcpy(&x[c], r->shadow + pcb->buffer_offset + (f + c) * 4,
                            4);
                  fprintf(stderr,
                          "gex-wgpu:     [%3u] %10.3f %10.3f %10.3f %10.3f\n",
                          f * 4, x[0], x[1], x[2], x[3]);
               }
            }
         }
      }
      unsigned layout_count = vb_slot_count;
      if (missing && layout_count < GEX_WGPU_MAX_VERTEX_BUFFERS &&
          gex_wgpu_default_attribute_buffer(gex_screen(pipe->screen))) {
         const unsigned slot = layout_count;
         unsigned n = 0;
         vertex_layouts[slot] =
            (WGPUVertexBufferLayout)WGPU_VERTEX_BUFFER_LAYOUT_INIT;
         vertex_layouts[slot].arrayStride = 0;
         vertex_layouts[slot].stepMode = WGPUVertexStepMode_Vertex;
         u_foreach_bit64(location, missing) {
            if (n >= GEX_WGPU_MAX_VERTEX_ELEMENTS)
               break;
            const bool integer =
               (ctx->vs->vertex_locations_sint | ctx->vs->vertex_locations_uint)
               & BITFIELD64_BIT(location);
            vertex_attributes[slot][n] =
               (WGPUVertexAttribute)WGPU_VERTEX_ATTRIBUTE_INIT;
            vertex_attributes[slot][n].format =
               ctx->vs->vertex_locations_sint & BITFIELD64_BIT(location)
                  ? WGPUVertexFormat_Sint32x4
               : ctx->vs->vertex_locations_uint & BITFIELD64_BIT(location)
                  ? WGPUVertexFormat_Uint32x4 : WGPUVertexFormat_Float32x4;
            vertex_attributes[slot][n].offset = integer ? 16 : 0;
            vertex_attributes[slot][n].shaderLocation = location;
            ++n;
         }
         vertex_layouts[slot].attributeCount = n;
         vertex_layouts[slot].attributes = vertex_attributes[slot];
         layout_count = slot + 1;
         default_attribute_slot = (int)slot;
         pipeline_desc.vertex.bufferCount = layout_count;
         pipeline_desc.vertex.buffers = vertex_layouts;
      }
   }
   pipeline_desc.primitive.topology = gex_wgpu_topology(info->mode);
   /* WebGPU wants a strip pipeline to name its index format up front, because
    * that is what decides the primitive restart value. */
   if (info->index_size &&
       (info->mode == MESA_PRIM_LINE_STRIP ||
        info->mode == MESA_PRIM_TRIANGLE_STRIP))
      pipeline_desc.primitive.stripIndexFormat = info->index_size == 2
         ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
   /* GL_DEPTH_CLAMP: fragments outside the depth range are clamped into it
    * rather than clipped away. */
   pipeline_desc.primitive.unclippedDepth =
      ctx->rasterizer && ctx->rasterizer->unclipped_depth;
   /* The Y-lowered vertex variant makes WebGPU window order match Gallium's
    * window order, so front_ccw maps directly. */
   pipeline_desc.primitive.frontFace = ctx->rasterizer &&
                                       ctx->rasterizer->front_ccw
      ? WGPUFrontFace_CCW : WGPUFrontFace_CW;
   pipeline_desc.primitive.cullMode = !ctx->rasterizer
      ? WGPUCullMode_None
      : ctx->rasterizer->cull_face == PIPE_FACE_FRONT
      ? WGPUCullMode_Front
      : ctx->rasterizer->cull_face == PIPE_FACE_BACK
      ? WGPUCullMode_Back : WGPUCullMode_None;
   /* Must equal the attachments' sample count, or WebGPU rejects the pipeline
    * with "attachment state ... not compatible". This was hardcoded to one. */
   pipeline_desc.multisample.count = MAX2(1, ctx->fb_samples);
   pipeline_desc.multisample.mask = 0xffffffffu;
   /* No colour targets means no fragment stage. WebGPU's depth-only pipeline
    * omits it entirely, which is the form guaranteed to validate; a fragment
    * state with zero targets is the kind of thing that gets rejected, and a
    * rejected pipeline is a validation error, which poisons the device.
    *
    * What this gives up is a fragment shader's side effects on a depth-only
    * pass -- discard, most of all. That is a narrower loss than refusing the
    * draw outright, which is what happened before. */
   pipeline_desc.fragment = ctx->color_count ? &fragment : NULL;
   WGPUDepthStencilState depth_state = WGPU_DEPTH_STENCIL_STATE_INIT;
   if (ctx->depth) {
      const bool enabled = ctx->dsa && ctx->dsa->depth_enabled;
      /* From the format, not from a two-way guess. A pipeline whose depth
       * format differs from the attachment's is a validation error, and this
       * named only two of the four the driver can attach: depth16unorm read
       * as depth32float, and depth32float-stencil8 as depth32float without
       * its stencil state. Both poison the device. */
      const bool has_stencil =
         util_format_has_stencil(util_format_description(ctx->depth->format));
      depth_state.format = gex_wgpu_texture_format(ctx->depth->format);
      depth_state.depthWriteEnabled = enabled && ctx->dsa->depth_writemask
         ? WGPUOptionalBool_True : WGPUOptionalBool_False;
      depth_state.depthCompare = enabled
         ? gex_wgpu_compare(ctx->dsa->depth_func) : WGPUCompareFunction_Always;
      if (has_stencil) {
         const bool stencil_enabled = ctx->dsa && ctx->dsa->stencil[0].enabled;
         const struct pipe_stencil_state *stencil = stencil_enabled
            ? &ctx->dsa->stencil[0] : NULL;
         /* The back face is its own state when GL asked for one; otherwise
          * both faces take the front's, which is what single-sided means. */
         const struct pipe_stencil_state *back =
            stencil_enabled && ctx->dsa->stencil[1].enabled
               ? &ctx->dsa->stencil[1] : stencil;
         WGPUStencilFaceState face = WGPU_STENCIL_FACE_STATE_INIT;
         face.compare = stencil_enabled
            ? gex_wgpu_compare(stencil->func) : WGPUCompareFunction_Always;
         face.failOp = stencil_enabled
            ? gex_wgpu_stencil_operation(stencil->fail_op)
            : WGPUStencilOperation_Keep;
         face.depthFailOp = stencil_enabled
            ? gex_wgpu_stencil_operation(stencil->zfail_op)
            : WGPUStencilOperation_Keep;
         face.passOp = stencil_enabled
            ? gex_wgpu_stencil_operation(stencil->zpass_op)
            : WGPUStencilOperation_Keep;
         depth_state.stencilFront = face;
         WGPUStencilFaceState rear = face;
         if (back && back != stencil) {
            rear.compare = gex_wgpu_compare(back->func);
            rear.failOp = gex_wgpu_stencil_operation(back->fail_op);
            rear.depthFailOp = gex_wgpu_stencil_operation(back->zfail_op);
            rear.passOp = gex_wgpu_stencil_operation(back->zpass_op);
         }
         depth_state.stencilBack = rear;
         depth_state.stencilReadMask = stencil_enabled
            ? stencil->valuemask : 0xff;
         depth_state.stencilWriteMask = stencil_enabled
            ? stencil->writemask : 0;
      }
      pipeline_desc.depthStencil = &depth_state;
   }
   WGPUBindGroupLayout group_layouts[4] = {NULL, NULL, NULL, NULL};
   WGPUPipelineLayout explicit_pipeline_layout = NULL;
   /* The vertex group also carries the transform feedback capture buffers, so
    * a shader that captures needs it even with no uniform blocks. */
   const bool needs_vertex_constant =
      ctx->vs->ubo_count != 0;
   const bool needs_fragment_constant = ctx->fs->ubo_count != 0;
   const bool needs_texture = ctx->fs->texture_count != 0;
   bool missing_constant = false;
   static bool told_which;
   for (unsigned i = 0; i < ctx->vs->ubo_count; ++i) {
      const unsigned slot = ctx->vs->ubo_slots[i];
      if (!ctx->constant_buffers[MESA_SHADER_VERTEX][slot].buffer) {
         missing_constant = true;
         if (!told_which) {
            told_which = true;
            fprintf(stderr,
                    "gex-wgpu: vertex UBO slot %u empty (shader declares %u "
                    "block(s); size=%u user=%p)\n", slot, ctx->vs->ubo_count,
                    ctx->constant_buffers[MESA_SHADER_VERTEX][slot].buffer_size,
                    (void *)ctx->constant_buffers[MESA_SHADER_VERTEX][slot]
                       .user_buffer);
         }
      }
   }
   for (unsigned i = 0; i < ctx->fs->ubo_count; ++i) {
      const unsigned slot = ctx->fs->ubo_slots[i];
      if (!ctx->constant_buffers[MESA_SHADER_FRAGMENT][slot].buffer) {
         missing_constant = true;
         if (!told_which) {
            told_which = true;
            char asks[64] = "", has[128] = "";
            for (unsigned k = 0; k < ctx->fs->ubo_count &&
                                 k < GEX_WGPU_MAX_CONSTANT_BUFFERS; ++k)
               snprintf(asks + strlen(asks), sizeof(asks) - strlen(asks),
                        "%u ", ctx->fs->ubo_slots[k]);
            for (unsigned k = 0; k < GEX_WGPU_MAX_CONSTANT_BUFFERS; ++k)
               if (ctx->constant_buffers[MESA_SHADER_FRAGMENT][k].buffer)
                  snprintf(has + strlen(has), sizeof(has) - strlen(has),
                           "%u(%uB) ", k,
                           ctx->constant_buffers[MESA_SHADER_FRAGMENT][k]
                              .buffer_size);
            fprintf(stderr,
                    "gex-wgpu: fragment UBO slot %u empty -- shader asks for "
                    "[%s], bound are [%s]\n", slot, asks,
                    has[0] ? has : "none");
         }
      }
   }
   /* WebGPU checks the fragment output's base type against every colour
    * target's, and rejects the pipeline when they differ. Catching it here
    * costs one draw; letting it reach the device costs every command after it,
    * because a validation error poisons the device rather than failing the
    * call. TGSI carries no type on an output, so a shader built from it -- a
    * u_blitter integer fetch, for one -- arrives declaring float whatever it
    * actually writes. */
   const nir_alu_type wanted_output =
      ctx->color_count == 0 ? nir_type_invalid
      : util_format_is_pure_uint(ctx->color_formats[0]) ? nir_type_uint
      : util_format_is_pure_sint(ctx->color_formats[0]) ? nir_type_int
      : nir_type_float;
   /* A fragment output whose type disagrees with the colour target is
    * retyped, not refused. gex_wgpu_fragment_module builds a variant that
    * redeclares the output -- a NIR value is an untyped bit pattern, so the
    * same bits simply become correctly described -- and WebGPU then accepts
    * the pipeline.
    *
    * This was a refusal for a long time, on a measurement that said
    * permitting scored worse. That measurement was taken while the module was
    * crashing partway through every run, so what it compared was two
    * different amounts of dead context. With the crash fixed it is worth
    * exactly what it costs to retype a variable. */

   bool missing_texture = false;
   /* Which binding disagreed, and how. This check folds a dozen conditions
    * into one refusal, and a line number does not say which of them fired. */
   const char *binding_why = NULL;
   unsigned binding_slot = 0;
#define GEX_BINDING_BAD(cond, why)                                            \
   do {                                                                       \
      if (cond) {                                                             \
         missing_texture = true;                                              \
         if (!binding_why) { binding_why = (why); binding_slot = slot; }       \
      }                                                                       \
   } while (0)
   /* Slots holding a depth texture that the shader declared as an ordinary
    * float one. GL allows sampler2D on a depth texture -- that is how a depth
    * buffer is read back through a shader -- and NIR hands this driver
    * texture_2d<f32> for it, where WGSL would want texture_depth_2d. WebGPU
    * also allows a depth-format view to be bound as `unfilterable-float`,
    * which texture_2d<f32> accepts, provided the sampler is non-filtering --
    * and a depth texture is not filterable in WebGPU at all, so a filtering
    * sampler on one could not have been honoured either way. */
   /* Slots holding a depth texture that the shader declared as an ordinary
    * float one. GL allows sampler2D on a depth texture -- that is how a depth
    * buffer is read back through a shader -- and NIR hands this driver
    * texture_2d<f32> for it, where WGSL would want texture_depth_2d. WebGPU
    * also allows a depth-format view to be bound as `unfilterable-float`,
    * which texture_2d<f32> accepts, provided the sampler is non-filtering --
    * and a depth texture is not filterable in WebGPU at all, so a filtering
    * sampler on one could not have been honoured either way. */
/* Slots holding a depth texture that the shader declared as an ordinary
    * float one. GL allows sampler2D on a depth texture -- that is how a depth
    * buffer is read back through a shader -- and NIR hands this driver
    * texture_2d<f32> for it, where WGSL would want texture_depth_2d. WebGPU
    * also allows a depth-format view to be bound as `unfilterable-float`,
    * which texture_2d<f32> accepts, provided the sampler is non-filtering --
    * and a depth texture is not filterable in WebGPU at all, so a filtering
    * sampler on one could not have been honoured either way. */
   /* Slots holding a depth texture that the shader declared as an ordinary
    * float one. GL allows sampler2D on a depth texture -- that is how a depth
    * buffer is read back through a shader -- and NIR hands this driver
    * texture_2d<f32> for it, where WGSL would want texture_depth_2d. WebGPU
    * also allows a depth-format view to be bound as `unfilterable-float`,
    * which texture_2d<f32> accepts, provided the sampler is non-filtering --
    * and a depth texture is not filterable in WebGPU at all, so a filtering
    * sampler on one could not have been honoured either way. */
/* Slots holding a depth texture that the shader declared as an ordinary
    * float one. GL allows sampler2D on a depth texture -- that is how a depth
    * buffer is read back through a shader -- and NIR hands this driver
    * texture_2d<f32> for it, where WGSL would want texture_depth_2d. WebGPU
    * also allows a depth-format view to be bound as `unfilterable-float`,
    * which texture_2d<f32> accepts, provided the sampler is non-filtering --
    * and a depth texture is not filterable in WebGPU at all, so a filtering
    * sampler on one could not have been honoured either way. */
   /* Slots holding a depth texture that the shader declared as an ordinary
    * float one. GL allows sampler2D on a depth texture -- that is how a depth
    * buffer is read back through a shader -- and NIR hands this driver
    * texture_2d<f32> for it, where WGSL would want texture_depth_2d. WebGPU
    * also allows a depth-format view to be bound as `unfilterable-float`,
    * which texture_2d<f32> accepts, provided the sampler is non-filtering --
    * and a depth texture is not filterable in WebGPU at all, so a filtering
    * sampler on one could not have been honoured either way. */
/* Slots holding a depth texture that the shader declared as an ordinary
    * float one. GL allows sampler2D on a depth texture -- that is how a depth
    * buffer is read back through a shader -- and NIR hands this driver
    * texture_2d<f32> for it, where WGSL would want texture_depth_2d. WebGPU
    * also allows a depth-format view to be bound as `unfilterable-float`,
    * which texture_2d<f32> accepts, provided the sampler is non-filtering --
    * and a depth texture is not filterable in WebGPU at all, so a filtering
    * sampler on one could not have been honoured either way. */
   /* Slots holding a depth texture that the shader declared as an ordinary
    * float one. GL allows sampler2D on a depth texture -- that is how a depth
    * buffer is read back through a shader -- and NIR hands this driver
    * texture_2d<f32> for it, where WGSL would want texture_depth_2d. WebGPU
    * also allows a depth-format view to be bound as `unfilterable-float`,
    * which texture_2d<f32> accepts, provided the sampler is non-filtering --
    * and a depth texture is not filterable in WebGPU at all, so a filtering
    * sampler on one could not have been honoured either way. */

   uint32_t depth_as_float = 0;
   /* Per draw: a snapshot bound for one draw must not leak into the next. */
   memset(ctx->feedback_view, 0, sizeof(ctx->feedback_view));
   for (unsigned i = 0; i < ctx->fs->texture_count; ++i) {
      const unsigned slot = ctx->fs->texture_slots[i];
      const bool multisampled = (ctx->fs->multisampled_slots >> slot) & 1;
      /* Nothing bound: the bind group supplies a 1x1 stand-in that matches
       * the shader's declaration by construction, so every reconciliation
       * below -- sample type, dimension, multisampledness -- is about a
       * texture that is not there. GL defines this case rather than
       * forbidding it; see gex_wgpu_default_texture_view. */
      if (!ctx->sampler_views[slot])
         continue;
      /* A texture cannot be sampled while it is also a colour attachment of
       * this pass: WebGPU calls that "writable usage and another usage in the
       * same synchronization scope" and rejects the whole command buffer, and
       * one rejected buffer takes every draw batched with it. GL calls the
       * same thing a feedback loop and leaves the result undefined, so nothing
       * correct can depend on it -- and the binding that reaches here is
       * usually just stale, left over from a pass that really did sample it.
       * Declining this one draw is the narrow reading of "undefined". */
      if (ctx->sampler_views[slot]) {
         for (unsigned c = 0; c < ctx->color_count; ++c) {
            if (ctx->colors[c] &&
                ctx->sampler_views[slot]->texture == ctx->colors[c]) {
               WGPUTextureView snapshot =
                  gex_wgpu_feedback_snapshot(ctx, ctx->colors[c]);
               static bool said;
               if (!said) {
                  said = true;
                  fprintf(stderr,
                          "gex-wgpu: slot %u samples colour attachment %u -- "
                          "a GL feedback loop; sampling a snapshot of it%s\n",
                          slot, c, snapshot ? "" : " FAILED");
               }
               if (!snapshot)
                  GEX_BINDING_BAD(true, "samples a colour attachment");
               else if (slot < GEX_WGPU_MAX_SAMPLERS)
                  ctx->feedback_view[slot] = snapshot;
               break;
            }
         }
      }
      if (ctx->sampler_views[slot] && ctx->sampler_views[slot]->texture &&
          ctx->sampler_views[slot]->texture->target != PIPE_BUFFER &&
          !gex_resource(ctx->sampler_views[slot]->texture)->texture_written) {
         /* Sampled without ever having been written: no upload reached it, no
          * copy filled it and it was never a render target. WebGPU zeroes a
          * new texture, so this samples as transparent black and draws a black
          * quad -- with no error from anywhere, which is the hardest kind of
          * wrong frame to attribute. Not a refusal: GL allows sampling an
          * incomplete texture and defines the result, so this only says so. */
         static unsigned told;
         if (told < 4) {
            ++told;
            struct pipe_resource *t = ctx->sampler_views[slot]->texture;
            fprintf(stderr,
                    "gex-wgpu: slot %u samples a texture nothing ever wrote "
                    "(%ux%ux%u target=%d format=%d levels=%u)\n",
                    slot, t->width0, t->height0, t->depth0, (int)t->target,
                    (int)t->format, t->last_level + 1);
         }
      }
      if (ctx->sampler_views[slot]) {
         /* A multisampled texture is never filterable, whatever its format
          * would allow, and the shader's declaration has to match the texture
          * actually bound -- textureLoad with a sample index only exists for
          * one of them. */
         WGPUTextureSampleType actual_type = gex_wgpu_format_sample_type(
            ctx->sampler_views[slot]->format);
         if (multisampled && actual_type == WGPUTextureSampleType_Float)
            actual_type = WGPUTextureSampleType_UnfilterableFloat;
         const struct gex_wgpu_sampler *sampler = ctx->samplers[slot];
         if (actual_type == WGPUTextureSampleType_Depth &&
             ctx->fs->texture_sample_types[i] == WGPUTextureSampleType_Float &&
             sampler &&
             sampler->state.compare_mode != PIPE_TEX_COMPARE_R_TO_TEXTURE &&
             sampler->state.min_img_filter == PIPE_TEX_FILTER_NEAREST &&
             sampler->state.mag_img_filter == PIPE_TEX_FILTER_NEAREST &&
             sampler->state.min_mip_filter != PIPE_TEX_MIPFILTER_LINEAR) {
            actual_type = WGPUTextureSampleType_Float;
            depth_as_float |= 1u << slot;
         }
         GEX_BINDING_BAD(actual_type != ctx->fs->texture_sample_types[i],
                         "sample type differs from the shader's");
         GEX_BINDING_BAD(
            multisampled != (ctx->sampler_views[slot]->texture != NULL &&
                             MAX2(1, ctx->sampler_views[slot]
                                        ->texture->nr_samples) > 1),
            "multisampledness differs from the shader's");
         WGPUTextureViewDimension actual_dimension =
            gex_wgpu_target_view_dimension(
               ctx->sampler_views[slot]->target);
         if ((actual_dimension == WGPUTextureViewDimension_Cube ||
              actual_dimension == WGPUTextureViewDimension_CubeArray) &&
             ctx->fs->texture_dimensions[i] ==
                WGPUTextureViewDimension_2DArray &&
             (ctx->fs->texture_sample_types[i] == WGPUTextureSampleType_Uint ||
              ctx->fs->texture_sample_types[i] == WGPUTextureSampleType_Sint))
            actual_dimension = WGPUTextureViewDimension_2DArray;
         GEX_BINDING_BAD(actual_dimension != ctx->fs->texture_dimensions[i],
                         "view dimension differs from the shader's");
      }
   }
   for (unsigned i = 0; i < ctx->fs->sampler_count; ++i) {
      const unsigned slot = ctx->fs->sampler_slots[i];
      /* Likewise: an unbound sampler is served by the default one, which is
       * created to match the shader's binding type. */
      if (!ctx->samplers[slot])
         continue;
      {
         const bool expected_comparison =
            ctx->fs->sampler_types[i] ==
            WGPUSamplerBindingType_Comparison;
         const bool actual_comparison =
            ctx->samplers[slot]->state.compare_mode ==
            PIPE_TEX_COMPARE_R_TO_TEXTURE;
         GEX_BINDING_BAD(expected_comparison != actual_comparison,
                         "comparison sampler mismatch");
         if (ctx->fs->sampler_types[i] ==
             WGPUSamplerBindingType_NonFiltering) {
            GEX_BINDING_BAD(
               ctx->samplers[slot]->state.min_img_filter !=
                     PIPE_TEX_FILTER_NEAREST ||
               ctx->samplers[slot]->state.mag_img_filter !=
                     PIPE_TEX_FILTER_NEAREST ||
               ctx->samplers[slot]->state.min_mip_filter ==
                     PIPE_TEX_MIPFILTER_LINEAR,
               "filtering sampler on a non-filterable texture");
         }
      }
   }
#undef GEX_BINDING_BAD
   if (missing_constant || missing_texture) {
      /* Declined, not latched. Returning here already keeps a pipeline the
       * device would reject out of it, which is the whole job; latching on
       * top ended the context, and every later map and upload in it then
       * failed too. That cascade is what "Valid format used but glTexImage2D
       * failed" was counting -- 871 of them behind one refused draw.
       *
       * The refusal that reaches here most is a depth texture sampled by a
       * shader that declared an ordinary float one: WebGPU wants a depth
       * texture bound as `texture_depth_2d`, and NIR gives us
       * `texture_2d<f32>`. Fixing that needs a shader variant that
       * redeclares the texture, which is the real work; declining without
       * killing the context is what keeps the rest of the run honest until
       * then. */
      /* Counted, not just announced once. A declined draw is invisible: the
       * frame still presents, at full rate, missing whatever that draw was.
       * If it is most of them the canvas shows only the clear colour, which
       * is indistinguishable from an application that has not started. */
      ++gex_wgpu_draws_declined;
      if (missing_constant) ++gex_wgpu_declined_constant;
      else ++gex_wgpu_declined_texture;
      if (binding_why) gex_wgpu_declined_last_why = binding_why;
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr, "gex-wgpu: draw declined (%s) slot=%u\n",
                 missing_constant ? "a constant buffer is not bound"
                                  : binding_why ? binding_why : "binding",
                 binding_slot);
      }
      return;
   }
   WGPUBindGroupLayoutEntry
      texture_layout_entries[GEX_WGPU_MAX_SAMPLERS * 2];
   /* Sized by the actual maxima, not a literal: raising
    * GEX_WGPU_MAX_CONSTANT_BUFFERS past 8 would otherwise overflow these on a
    * shader that declares enough uniform blocks. */
   WGPUBindGroupLayoutEntry
      vertex_layout_entries[GEX_WGPU_MAX_CONSTANT_BUFFERS +
                            GEX_WGPU_MAX_SO_BUFFERS];
   WGPUBindGroupLayoutEntry
      fragment_layout_entries[GEX_WGPU_MAX_CONSTANT_BUFFERS];
   if (needs_texture) {
      for (unsigned i = 0; i < ctx->fs->texture_count; ++i) {
         texture_layout_entries[i] =
            (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
         texture_layout_entries[i].binding =
            ctx->fs->texture_bindings[i];
         texture_layout_entries[i].visibility = WGPUShaderStage_Fragment;
         /* Named as unfilterable-float where the shader asked for float and
          * the bound view is depth: that is the pair WebGPU accepts. */
         texture_layout_entries[i].texture.sampleType =
            (depth_as_float & (1u << ctx->fs->texture_slots[i]))
               ? WGPUTextureSampleType_UnfilterableFloat
               : ctx->fs->texture_sample_types[i];
         texture_layout_entries[i].texture.viewDimension =
            ctx->fs->texture_dimensions[i];
         texture_layout_entries[i].texture.multisampled =
            (ctx->fs->multisampled_slots >> ctx->fs->texture_slots[i]) & 1;
      }
      for (unsigned i = 0; i < ctx->fs->sampler_count; ++i) {
         const unsigned entry = ctx->fs->texture_count + i;
         texture_layout_entries[entry] =
            (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
         texture_layout_entries[entry].binding =
            ctx->fs->sampler_bindings[i];
         texture_layout_entries[entry].visibility =
            WGPUShaderStage_Fragment;
         texture_layout_entries[entry].sampler.type =
            (depth_as_float & (1u << ctx->fs->sampler_slots[i]))
               ? WGPUSamplerBindingType_NonFiltering
               : ctx->fs->sampler_types[i];
      }
   }
   for (unsigned i = 0; i < ctx->vs->ubo_count; ++i) {
      const unsigned slot = ctx->vs->ubo_slots[i];
      const bool storage = ctx->vs->ubo_storage_mask & (1u << i);
      vertex_layout_entries[i] =
         (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      vertex_layout_entries[i].binding = ctx->vs->ubo_bindings[i];
      vertex_layout_entries[i].visibility = WGPUShaderStage_Vertex;
      /* Storage, not ReadOnlyStorage, even though the shader only reads it:
       * the emitter declares every storage buffer read_write, and WebGPU
       * rejects a layout that disagrees with the shader's declaration. */
      vertex_layout_entries[i].buffer.type = storage
         ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_Uniform;
      /* The offset comes from the draw, not the bind group: uniforms are
       * sub-allocated from one ring, so the buffer and size repeat while the
       * offset does not. Baking the offset in would make a distinct bind
       * group per draw and there would be nothing left to cache. */
      vertex_layout_entries[i].buffer.hasDynamicOffset = true;
      vertex_layout_entries[i].buffer.minBindingSize =
         ALIGN_POT(ctx->constant_buffers[MESA_SHADER_VERTEX][slot].buffer_size, 16);
   }
   for (unsigned i = 0; i < ctx->fs->ubo_count; ++i) {
      const unsigned slot = ctx->fs->ubo_slots[i];
      const bool storage = ctx->fs->ubo_storage_mask & (1u << i);
      fragment_layout_entries[i] =
         (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      fragment_layout_entries[i].binding = ctx->fs->ubo_bindings[i];
      fragment_layout_entries[i].visibility = WGPUShaderStage_Fragment;
      /* Storage, not ReadOnlyStorage, even though the shader only reads it:
       * the emitter declares every storage buffer read_write, and WebGPU
       * rejects a layout that disagrees with the shader's declaration. */
      fragment_layout_entries[i].buffer.type = storage
         ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_Uniform;
      /* The offset comes from the draw, not the bind group: uniforms are
       * sub-allocated from one ring, so the buffer and size repeat while the
       * offset does not. Baking the offset in would make a distinct bind
       * group per draw and there would be nothing left to cache. */
      fragment_layout_entries[i].buffer.hasDynamicOffset = true;
      fragment_layout_entries[i].buffer.minBindingSize =
         ALIGN_POT(ctx->constant_buffers[MESA_SHADER_FRAGMENT][slot].buffer_size, 16);
   }
   /* The vertex stage's textures, in group 3. */
   WGPUBindGroupLayoutEntry
      vertex_texture_entries[2 * GEX_WGPU_MAX_SAMPLERS];
   const bool needs_vertex_texture = ctx->vs->texture_count != 0;
   if (needs_vertex_texture) {
      for (unsigned i = 0; i < ctx->vs->texture_count; ++i) {
         vertex_texture_entries[i] =
            (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
         vertex_texture_entries[i].binding = ctx->vs->texture_bindings[i];
         vertex_texture_entries[i].visibility = WGPUShaderStage_Vertex;
         vertex_texture_entries[i].texture.sampleType =
            ctx->vs->texture_sample_types[i];
         vertex_texture_entries[i].texture.viewDimension =
            ctx->vs->texture_dimensions[i];
         vertex_texture_entries[i].texture.multisampled =
            (ctx->vs->multisampled_slots >> ctx->vs->texture_slots[i]) & 1;
      }
      for (unsigned i = 0; i < ctx->vs->sampler_count; ++i) {
         const unsigned entry = ctx->vs->texture_count + i;
         vertex_texture_entries[entry] =
            (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
         vertex_texture_entries[entry].binding = ctx->vs->sampler_bindings[i];
         vertex_texture_entries[entry].visibility = WGPUShaderStage_Vertex;
         /* A vertex fetch is never filtered -- there are no derivatives -- so
          * WebGPU wants the sampler named non-filtering to match. */
         vertex_texture_entries[entry].sampler.type =
            WGPUSamplerBindingType_NonFiltering;
      }
   }
   const unsigned group_count = needs_vertex_texture ? 4
      : needs_fragment_constant ? 3
      : needs_vertex_constant ? 2 : needs_texture ? 1 : 0;
   const int64_t t_bind_start = gex_timing ? os_time_get_nano() : 0;
   for (unsigned group = 0; group < group_count; ++group) {
      WGPUBindGroupLayoutDescriptor desc =
         WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
      if (group == 0 && needs_texture) {
         desc.entryCount = ctx->fs->texture_count + ctx->fs->sampler_count;
         desc.entries = texture_layout_entries;
      } else if (group == 1 && needs_vertex_constant) {
         desc.entryCount = ctx->vs->ubo_count;
         desc.entries = vertex_layout_entries;
      } else if (group == 2 && needs_fragment_constant) {
         desc.entryCount = ctx->fs->ubo_count;
         desc.entries = fragment_layout_entries;
      } else if (group == 3 && needs_vertex_texture) {
         desc.entryCount = ctx->vs->texture_count + ctx->vs->sampler_count;
         desc.entries = vertex_texture_entries;
      }
      /* Owned by the screen's cache, so nothing here releases it. */
      group_layouts[group] =
         gex_wgpu_cached_bind_group_layout(screen, desc.entries,
                                           desc.entryCount);
      if (!group_layouts[group]) {
         gex_fail(ctx);
         return;
      }
   }
   if (group_count) {
      explicit_pipeline_layout =
         gex_wgpu_cached_pipeline_layout(screen, group_layouts, group_count);
      if (!explicit_pipeline_layout) {
         gex_fail(ctx);
         return;
      }
      pipeline_desc.layout = explicit_pipeline_layout;
   }
   /* GEX_WGPU_TIMING=1 splits a draw's wall clock into the three things it
    * does, because "the driver is slow" is not actionable and the three have
    * completely different fixes. Off by default; one clock read per draw when
    * on. */
   if (gex_timing) gex_wgpu_time_bind += os_time_get_nano() - t_bind_start;
   const int64_t t_pipeline_start = gex_timing ? os_time_get_nano() : 0;
   bool pipeline_cached = false;
   WGPURenderPipeline pipeline =
      gex_wgpu_cached_pipeline(screen, &pipeline_desc, &pipeline_cached);
   if (gex_timing)
      gex_wgpu_time_pipeline += os_time_get_nano() - t_pipeline_start;
   const int64_t t_record_start = gex_timing ? os_time_get_nano() : 0;
   /* Can this draw go into the pass that is already open?
    *
    * Everything the pass descriptor is built from goes in the key. An
    * occlusion query is the one thing never merged: the query index is per
    * pass, so two counting draws sharing one would both write slot zero. */
   struct gex_wgpu_pass_key wanted_pass;
   memset(&wanted_pass, 0, sizeof(wanted_pass));
   wanted_pass.color_count = ctx->color_count;
   for (unsigned i = 0; i < ctx->color_count; ++i) {
      wanted_pass.colors[i] = ctx->colors[i];
      wanted_pass.formats[i] = ctx->color_formats[i];
      wanted_pass.levels[i] = ctx->color_levels[i];
      wanted_pass.layers[i] = ctx->color_layers[i];
   }
   wanted_pass.depth = ctx->depth;
   /* Depth is read-only when this draw samples the texture it is attached to
    * and does not write it; computed here rather than with the attachment
    * because the answer is part of the key. */
   if (ctx->depth) {
      bool depth_sampled = false;
      for (unsigned i = 0; i < GEX_WGPU_MAX_SAMPLERS; ++i) {
         const struct pipe_sampler_view *view = ctx->sampler_views[i];
         if (view && view->texture == ctx->depth) {
            depth_sampled = true;
            break;
         }
      }
      wanted_pass.depth_read_only =
         depth_sampled && !(ctx->dsa && ctx->dsa->depth_writemask);
   }
   /* If the open pass already has exactly these attachments, none of the
    * views, descriptors or checks below are needed: they describe a pass that
    * is not going to be begun. Creating them anyway cost two to three
    * microseconds of every draw. */
   const bool counting_now = ctx->active_query && ctx->occlusion_set;
   const bool reuse = !counting_now && ctx->pending_pass &&
      memcmp(&ctx->pending_pass_key, &wanted_pass, sizeof(wanted_pass)) == 0;
   WGPUTextureView views[GEX_WGPU_MAX_COLOR_TARGETS] = {0};
   WGPUTextureView depth_view = NULL;
   bool missing_view = false;
   /* Only a pass that is about to begin needs attachment views. Making them
    * for a pass that is already open, and releasing them unused, was two to
    * three microseconds of every draw. */
   if (!reuse)
   for (unsigned i = 0; i < ctx->color_count; ++i) {
      /* Named explicitly: when the surface reinterprets an sRGB texture as
       * linear, or the reverse, the view is what carries that. */
      /* A hole has no texture and so no view; WebGPU reads that as a null
       * colour attachment, which is what GL_NONE in glDrawBuffers means. */
      if (!ctx->colors[i])
         continue;
      WGPUTextureViewDescriptor color_view = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      color_view.format = gex_wgpu_texture_format(ctx->color_formats[i]);
      /* A 3D texture has no 2D view: WebGPU attaches one whole, as a 3d view,
       * and picks the slice on the attachment with depthSlice. Asking for a
       * 2D view of it is a validation error, which is how attaching a
       * glFramebufferTexture3D target produced 768 of them. */
      const bool three_d = ctx->colors[i]->target == PIPE_TEXTURE_3D;
      color_view.dimension = three_d ? WGPUTextureViewDimension_3D
                                     : WGPUTextureViewDimension_2D;
      /* The attachment is one slice, and which slice is state the framebuffer
       * carried. Left at the defaults this always drew into level zero of
       * layer zero, so rendering a mip chain silently overwrote the base
       * level -- and the refusal that used to stand here hid that by killing
       * the draw instead. */
      color_view.baseMipLevel = ctx->color_levels[i];
      color_view.mipLevelCount = 1;
      if (!three_d) {
         color_view.baseArrayLayer = ctx->color_layers[i];
         color_view.arrayLayerCount = 1;
      }
      views[i] = wgpuTextureCreateView(
         gex_resource(ctx->colors[i])->texture, &color_view);
      missing_view |= !views[i];
   }
   if (!reuse && ctx->depth)
      depth_view = wgpuTextureCreateView(gex_resource(ctx->depth)->texture,
                                         NULL);
   if (gex_timing) gex_wgpu_time_views += os_time_get_nano() - t_record_start;
   const int64_t t_encoder_start = gex_timing ? os_time_get_nano() : 0;
   /* Joins whatever draws are already recorded rather than starting its own
    * command buffer; see gex_wgpu_submit_pending. */
   if (!ctx->pending_encoder)
      ctx->pending_encoder =
         wgpuDeviceCreateCommandEncoder(screen->device, NULL);
   WGPUCommandEncoder encoder = ctx->pending_encoder;
   const int64_t t_bg_start = gex_timing ? os_time_get_nano() : 0;
   if (gex_timing) gex_wgpu_time_encoder += t_bg_start - t_encoder_start;
   WGPUBindGroup bind_groups[4] = {NULL, NULL, NULL, NULL};
   struct gex_wgpu_dynamic_offsets dynamic[4] = {{0}, {0}, {0}, {0}};
   if (pipeline && needs_texture) {
      WGPUBindGroupEntry entries[GEX_WGPU_MAX_SAMPLERS * 2];
      for (unsigned i = 0; i < ctx->fs->texture_count; ++i) {
         const unsigned slot = ctx->fs->texture_slots[i];
         entries[i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
         entries[i].binding = ctx->fs->texture_bindings[i];
         struct gex_wgpu_sampler_view *view =
            (struct gex_wgpu_sampler_view *)ctx->sampler_views[slot];
         /* The snapshot, when this slot samples something the draw also
          * renders to; see gex_wgpu_feedback_snapshot. */
         entries[i].textureView =
            (slot < GEX_WGPU_MAX_SAMPLERS && ctx->feedback_view[slot])
               ? ctx->feedback_view[slot]
            /* GL leaves units enabled with nothing bound; a 1x1 opaque black
             * stand-in is what it says such a unit samples as. */
            : !view
               ? gex_wgpu_default_texture_view(
                    screen, ctx->fs->texture_dimensions[i],
                    ctx->fs->texture_sample_types[i])
            : ctx->fs->texture_dimensions[i] ==
                  WGPUTextureViewDimension_2DArray &&
               (ctx->sampler_views[slot]->target == PIPE_TEXTURE_CUBE ||
                ctx->sampler_views[slot]->target ==
                   PIPE_TEXTURE_CUBE_ARRAY)
               ? view->array_view : view->view;
         if (!entries[i].textureView) {
            static bool said;
            if (!said) {
               said = true;
               fprintf(stderr, "gex-wgpu: no texture for fragment slot %u "
                               "(dim=%d type=%d)\n", slot,
                       (int)ctx->fs->texture_dimensions[i],
                       (int)ctx->fs->texture_sample_types[i]);
            }
            gex_fail(ctx);
            return;
         }
      }
      for (unsigned i = 0; i < ctx->fs->sampler_count; ++i) {
         const unsigned slot = ctx->fs->sampler_slots[i];
         const unsigned entry = ctx->fs->texture_count + i;
         entries[entry] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
         entries[entry].binding = ctx->fs->sampler_bindings[i];
         entries[entry].sampler = ctx->samplers[slot]
            ? ctx->samplers[slot]->sampler
            : gex_wgpu_default_sampler(screen, ctx->fs->sampler_types[i]);
      }
      WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bind_desc.layout = group_layouts[0];
      bind_desc.entryCount = ctx->fs->texture_count + ctx->fs->sampler_count;
      bind_desc.entries = entries;
      bind_groups[0] = gex_wgpu_cached_bind_group(
         screen, bind_desc.layout, bind_desc.entries,
         bind_desc.entryCount);
   }
   if (pipeline && needs_vertex_constant) {
      /* Uniform blocks plus the transform feedback capture buffers, which
       * are written by the vertex stage and so live in this group. */
      WGPUBindGroupEntry
         entries[GEX_WGPU_MAX_CONSTANT_BUFFERS + GEX_WGPU_MAX_SO_BUFFERS];
      for (unsigned i = 0; i < ctx->vs->ubo_count; ++i) {
         const unsigned slot = ctx->vs->ubo_slots[i];
         const struct pipe_constant_buffer *constant =
            &ctx->constant_buffers[MESA_SHADER_VERTEX][slot];
         entries[i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
         entries[i].binding = ctx->vs->ubo_bindings[i];
         entries[i].buffer = gex_resource(constant->buffer)->buffer;
         entries[i].offset = 0;
         entries[i].size = ALIGN_POT(constant->buffer_size, 16);
         gex_wgpu_add_dynamic_offset(&dynamic[1],
                                     ctx->vs->ubo_bindings[i],
                                     constant->buffer_offset);
      }
      WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bind_desc.layout = group_layouts[1];
      bind_desc.entryCount = ctx->vs->ubo_count;
      bind_desc.entries = entries;
      bind_groups[1] = gex_wgpu_cached_bind_group(
         screen, bind_desc.layout, bind_desc.entries,
         bind_desc.entryCount);
   }
   if (pipeline && needs_fragment_constant) {
      WGPUBindGroupEntry entries[GEX_WGPU_MAX_CONSTANT_BUFFERS];
      for (unsigned i = 0; i < ctx->fs->ubo_count; ++i) {
         const unsigned slot = ctx->fs->ubo_slots[i];
         const struct pipe_constant_buffer *constant =
            &ctx->constant_buffers[MESA_SHADER_FRAGMENT][slot];
         entries[i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
         entries[i].binding = ctx->fs->ubo_bindings[i];
         entries[i].buffer = gex_resource(constant->buffer)->buffer;
         entries[i].offset = 0;
         entries[i].size = ALIGN_POT(constant->buffer_size, 16);
         gex_wgpu_add_dynamic_offset(&dynamic[2],
                                     ctx->fs->ubo_bindings[i],
                                     constant->buffer_offset);
      }
      WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bind_desc.layout = group_layouts[2];
      bind_desc.entryCount = ctx->fs->ubo_count;
      bind_desc.entries = entries;
      bind_groups[2] = gex_wgpu_cached_bind_group(
         screen, bind_desc.layout, bind_desc.entries,
         bind_desc.entryCount);
   }
   if (pipeline && needs_vertex_texture) {
      WGPUBindGroupEntry entries[2 * GEX_WGPU_MAX_SAMPLERS];
      for (unsigned i = 0; i < ctx->vs->texture_count; ++i) {
         const unsigned slot = ctx->vs->texture_slots[i];
         struct gex_wgpu_sampler_view *view =
            (struct gex_wgpu_sampler_view *)ctx->vertex_sampler_views[slot];
         entries[i] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
         entries[i].binding = ctx->vs->texture_bindings[i];
         entries[i].textureView = view ? view->view : NULL;
      }
      for (unsigned i = 0; i < ctx->vs->sampler_count; ++i) {
         const unsigned slot = ctx->vs->sampler_slots[i];
         const unsigned entry = ctx->vs->texture_count + i;
         entries[entry] = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
         entries[entry].binding = ctx->vs->sampler_bindings[i];
         entries[entry].sampler = ctx->vertex_samplers[slot]
            ? ctx->vertex_samplers[slot]->sampler : NULL;
      }
      WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bind_desc.layout = group_layouts[3];
      bind_desc.entryCount = ctx->vs->texture_count + ctx->vs->sampler_count;
      bind_desc.entries = entries;
      bind_groups[3] = gex_wgpu_cached_bind_group(
         screen, bind_desc.layout, bind_desc.entries,
         bind_desc.entryCount);
   }
   /* explicit_pipeline_layout, group_layouts[] and bind_groups[] all belong
    * to the screen's caches; releasing them here would free objects the next
    * draw expects to find. */
   const bool bind_failed =
      (needs_texture && !bind_groups[0]) ||
      (needs_vertex_constant && !bind_groups[1]) ||
      (needs_fragment_constant && !bind_groups[2]);
   if (!pipeline || (!reuse && (missing_view || (ctx->depth && !depth_view))) ||
       !encoder || bind_failed) {
      fprintf(stderr,
              "gex-wgpu: draw objects pipeline=%s view=%s encoder=%s bind=%s\n",
              pipeline ? "ok" : "failed", !missing_view ? "ok" : "failed",
              encoder ? "ok" : "failed",
              !bind_failed ? "ok" : "failed");
      if (pipeline && !pipeline_cached) wgpuRenderPipelineRelease(pipeline);
      for (unsigned i = 0; i < ctx->color_count; ++i)
         if (views[i]) wgpuTextureViewRelease(views[i]);
      if (depth_view) wgpuTextureViewRelease(depth_view);
      /* encoder is ctx->pending_encoder, which gex_fail tears down. */
      gex_fail(ctx);
      return;
   }

   WGPURenderPassColorAttachment
      attachments[GEX_WGPU_MAX_COLOR_TARGETS];
   for (unsigned i = 0; i < ctx->color_count; ++i) {
      attachments[i] =
         (WGPURenderPassColorAttachment)WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
      if (!views[i])
         continue;   /* a hole: null view, and no load or store op */
      attachments[i].view = views[i];
      if (ctx->colors[i] && ctx->colors[i]->target == PIPE_TEXTURE_3D)
         attachments[i].depthSlice = ctx->color_layers[i];
      attachments[i].loadOp = WGPULoadOp_Load;
      attachments[i].storeOp = WGPUStoreOp_Store;
   }
   WGPURenderPassDescriptor pass_desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   pass_desc.colorAttachmentCount = ctx->color_count;
   pass_desc.colorAttachments = attachments;
   WGPURenderPassDepthStencilAttachment depth_attachment =
      WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
   if (depth_view) {
      /* GL lets a depth texture be sampled while it is still attached, as long
       * as depth writes are off -- a depth pre-pass followed by shading that
       * reads it is the ordinary way to use one. WebGPU calls that "writable
       * usage and another usage in the same synchronization scope" and rejects
       * the whole command buffer unless the attachment is declared read-only.
       * So when this draw samples the very texture it is attached to, say so:
       * load, do not store, and no clear. */
      bool depth_sampled = false;
      for (unsigned i = 0; i < GEX_WGPU_MAX_SAMPLERS; ++i) {
         const struct pipe_sampler_view *view = ctx->sampler_views[i];
         if (view && view->texture == ctx->depth) {
            depth_sampled = true;
            break;
         }
      }
      const bool writes_depth = ctx->dsa && ctx->dsa->depth_writemask;
      const bool read_only = depth_sampled && !writes_depth;
      wanted_pass.depth_read_only = read_only;
      depth_attachment.view = depth_view;
      depth_attachment.depthReadOnly = read_only;
      /* Unconditionally, and outside the branch below. `..._INIT` leaves this
       * NaN, and the browser reads it off the dictionary and validates it as a
       * float whatever the load op says -- so a read-only depth attachment,
       * which sets no load op at all, still threw "The provided float value is
       * non-finite" out of beginRenderPass and took the entire pass with it.
       * Every draw in that pass was lost. Same defect, and the same one-line
       * fix, as the depth upload path above. */
      depth_attachment.depthClearValue = 1.0f;
      if (!read_only) {
         depth_attachment.depthLoadOp = WGPULoadOp_Load;
         depth_attachment.depthStoreOp = WGPUStoreOp_Store;
      }
      if (util_format_has_stencil(
             util_format_description(ctx->depth->format))) {
         depth_attachment.stencilReadOnly = read_only;
         if (!read_only) {
            depth_attachment.stencilLoadOp = WGPULoadOp_Load;
            depth_attachment.stencilStoreOp = WGPUStoreOp_Store;
         }
      }
      if (read_only) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr,
                    "gex-wgpu: depth attached read-only (it is also sampled "
                    "by this draw)\n");
         }
      }
      pass_desc.depthStencilAttachment = &depth_attachment;
   }
   /* A pass can only write occlusion results if it is told the query set up
    * front, so an active query has to be known before the pass begins. */
   const bool counting = ctx->active_query && ctx->occlusion_set;
   if (counting)
      pass_desc.occlusionQuerySet = ctx->occlusion_set;
   if (gex_timing) gex_wgpu_time_bg += os_time_get_nano() - t_bg_start;
   const int64_t t_begin_start = gex_timing ? os_time_get_nano() : 0;
   /* Reuse the open pass when this draw wants exactly the same attachments.
    * A counting draw never reuses one and never leaves one open, because the
    * occlusion query index belongs to the pass. */
   if (!reuse) {
      gex_wgpu_end_pass(ctx);
      ctx->pending_pass =
         wgpuCommandEncoderBeginRenderPass(encoder, &pass_desc);
      ++screen->pass_serial;
      memset(&ctx->emitted, 0, sizeof(ctx->emitted));
      if (!counting && ctx->pending_pass)
         ctx->pending_pass_key = wanted_pass;
   }
   /* An eviction since the last draw may have recycled a handle this state
    * still names; see gex_wgpu_screen::cache_epoch. */
   if (ctx->emitted_epoch != screen->cache_epoch) {
      memset(&ctx->emitted, 0, sizeof(ctx->emitted));
      ctx->emitted_epoch = screen->cache_epoch;
   }
   WGPURenderPassEncoder pass = ctx->pending_pass;
   if (gex_timing) gex_wgpu_time_begin += os_time_get_nano() - t_begin_start;
   if (pass) {
      if (counting)
         wgpuRenderPassEncoderBeginOcclusionQuery(pass, 0);
      if (ctx->emitted.pipeline != pipeline) {
         wgpuRenderPassEncoderSetPipeline(pass, pipeline);
         ctx->emitted.pipeline = pipeline;
      }
      if (ctx->depth &&
          util_format_has_stencil(
             util_format_description(ctx->depth->format)))
         wgpuRenderPassEncoderSetStencilReference(
            pass, ctx->stencil_ref.ref_value[0]);
      if (uses_blend_constant) {
         const WGPUColor blend_color = {
            ctx->blend_color.color[0], ctx->blend_color.color[1],
            ctx->blend_color.color[2], ctx->blend_color.color[3],
         };
         wgpuRenderPassEncoderSetBlendConstant(pass, &blend_color);
      }
      const float wanted_viewport[6] = { viewport_x, viewport_y,
                                         viewport_width, viewport_height,
                                         min_depth, max_depth };
      if (memcmp(ctx->emitted.viewport, wanted_viewport,
                 sizeof(wanted_viewport)) != 0) {
         wgpuRenderPassEncoderSetViewport(pass, viewport_x, viewport_y,
                                          viewport_width, viewport_height,
                                          min_depth, max_depth);
         memcpy(ctx->emitted.viewport, wanted_viewport,
                sizeof(wanted_viewport));
      }
      /* A draw with the scissor disabled wants the whole attachment, and has
       * to say so: the scissor belongs to the pass, and draws now share one,
       * so leaving it alone means inheriting whatever rect the last draw
       * asked for. SuperTuxKart's scrolling news ticker is clipped away from
       * the icon bar beside it, and without this the two were drawn on top of
       * each other -- the sort of thing that looks like a layout quirk until
       * the same frame from softpipe is put next to it. */
      uint32_t wanted_scissor[4];
      if (scissor_enabled) {
         wanted_scissor[0] = scissor_x;
         wanted_scissor[1] = scissor_y;
         wanted_scissor[2] = scissor_width;
         wanted_scissor[3] = scissor_height;
      } else {
         wanted_scissor[0] = 0;
         wanted_scissor[1] = 0;
         wanted_scissor[2] = ctx->fb_width;
         wanted_scissor[3] = ctx->fb_height;
      }
      if (!ctx->emitted.scissor_set ||
          memcmp(ctx->emitted.scissor, wanted_scissor,
                 sizeof(wanted_scissor)) != 0) {
         wgpuRenderPassEncoderSetScissorRect(
            pass, wanted_scissor[0], wanted_scissor[1],
            wanted_scissor[2], wanted_scissor[3]);
         memcpy(ctx->emitted.scissor, wanted_scissor, sizeof(wanted_scissor));
         ctx->emitted.scissor_set = true;
      }
      for (unsigned group = 0; group < 4; ++group) {
         if (!bind_groups[group])
            continue;
         const size_t offset_bytes = dynamic[group].count * sizeof(uint32_t);
         if (ctx->emitted.groups[group] == bind_groups[group] &&
             ctx->emitted.dynamic_count[group] == dynamic[group].count &&
             memcmp(ctx->emitted.dynamic[group], dynamic[group].offset,
                    offset_bytes) == 0)
            continue;
         wgpuRenderPassEncoderSetBindGroup(
            pass, group, bind_groups[group], dynamic[group].count,
            dynamic[group].count ? dynamic[group].offset : NULL);
         ctx->emitted.groups[group] = bind_groups[group];
         ctx->emitted.dynamic_count[group] = dynamic[group].count;
         memcpy(ctx->emitted.dynamic[group], dynamic[group].offset,
                offset_bytes);
      }
      /* By dense slot, matching the pipeline's layout array. Binding by
       * Gallium index instead both overran WebGPU's eight slots and walked
       * into the empty ones a disabled vertex array leaves behind. */
      for (unsigned slot = 0; slot < vb_slot_count; ++slot) {
         struct pipe_resource *buffer = ctx->vertex_buffers[vb_buffer[slot]];
         const uint32_t offset = ctx->vertex_buffer_offsets[vb_buffer[slot]];
         WGPUBuffer handle = gex_resource(buffer)->buffer;
         if (ctx->emitted.vertex[slot] == handle &&
             ctx->emitted.vertex_offset[slot] == offset)
            continue;
         wgpuRenderPassEncoderSetVertexBuffer(
            pass, slot, handle, offset, buffer->width0 - offset);
         ctx->emitted.vertex[slot] = handle;
         ctx->emitted.vertex_offset[slot] = offset;
      }
      /* After the application's, because the slot it goes in may still hold a
       * buffer bound for an earlier draw. */
      if (default_attribute_slot >= 0) {
         /* Tracked like any other slot: a later draw may put a real buffer
          * here, and it must not look unchanged because this one was never
          * recorded. */
         WGPUBuffer defaults = gex_screen(pipe->screen)->default_attributes;
         if (ctx->emitted.vertex[default_attribute_slot] != defaults ||
             ctx->emitted.vertex_offset[default_attribute_slot] != 0) {
            wgpuRenderPassEncoderSetVertexBuffer(
               pass, (uint32_t)default_attribute_slot, defaults, 0, 32);
            ctx->emitted.vertex[default_attribute_slot] = defaults;
            ctx->emitted.vertex_offset[default_attribute_slot] = 0;
         }
      }
      if (index_resource) {
         WGPUBuffer index_handle = gex_resource(index_resource)->buffer;
         const WGPUIndexFormat index_format = info->index_size == 2
            ? WGPUIndexFormat_Uint16 : WGPUIndexFormat_Uint32;
         if (ctx->emitted.index != index_handle ||
             ctx->emitted.index_format != index_format ||
             ctx->emitted.index_offset != index_byte_offset) {
            wgpuRenderPassEncoderSetIndexBuffer(
               pass, index_handle, index_format, index_byte_offset,
               index_resource->width0 - index_byte_offset);
            ctx->emitted.index = index_handle;
            ctx->emitted.index_format = index_format;
            ctx->emitted.index_offset = index_byte_offset;
         }
         ++gex_wgpu_draws_issued;
         gex_wgpu_count_draw_vs(ctx->vs);
         gex_wgpu_count_draw_target(
            ctx->fb_width, ctx->fb_height,
            ctx->color_count && ctx->colors[0]
               ? gex_resource(ctx->colors[0])->texture : NULL,
            ctx->color_count ? ctx->color_formats[0] : PIPE_FORMAT_NONE);
         if (ctx->color_count && ctx->colors[0] &&
             gex_resource(ctx->colors[0])->texture ==
                gex_wgpu_presented_texture)
            ++gex_wgpu_draws_to_presented;
         wgpuRenderPassEncoderDrawIndexed(
            pass, draws[0].count, info->instance_count, first_index,
            draws[0].index_bias, info->start_instance);
      } else {
         ++gex_wgpu_draws_issued;
         gex_wgpu_count_draw_vs(ctx->vs);
         gex_wgpu_count_draw_target(
            ctx->fb_width, ctx->fb_height,
            ctx->color_count && ctx->colors[0]
               ? gex_resource(ctx->colors[0])->texture : NULL,
            ctx->color_count ? ctx->color_formats[0] : PIPE_FORMAT_NONE);
         wgpuRenderPassEncoderDraw(pass, draws[0].count, info->instance_count,
                                   draws[0].start, info->start_instance);
      }
      if (counting)
         wgpuRenderPassEncoderEndOcclusionQuery(pass);
      const int64_t t_end_start = gex_timing ? os_time_get_nano() : 0;
      /* Left open for the next draw. Whatever stops recording -- a different
       * framebuffer, a clear, a map, the submit -- closes it. */
      if (counting)
         gex_wgpu_end_pass(ctx);
      if (gex_timing) gex_wgpu_time_end += os_time_get_nano() - t_end_start;
   }
   /* The views were recorded into the pass, which holds its own references,
    * so this draw is done with them even though nothing is submitted yet. */
   /* Null on the reuse path, where no views were made. */
   for (unsigned i = 0; i < ctx->color_count; ++i)
      if (views[i])
         wgpuTextureViewRelease(views[i]);
   if (depth_view)
      wgpuTextureViewRelease(depth_view);
   if (!pipeline_cached)
      wgpuRenderPipelineRelease(pipeline);
   /* bind_groups[] and, when cached, the pipeline belong to the screen. */
   if (!pass) {
      gex_fail(ctx);
      return;
   }
   const int64_t t_submit_start = gex_timing ? os_time_get_nano() : 0;
   if (gex_timing)
      gex_wgpu_time_record += t_submit_start - t_record_start;
   /* Everything this draw reads or writes, so a later map or upload knows
    * whether it has to wait for it. */
   for (unsigned i = 0; i < ctx->color_count; ++i)
      gex_wgpu_batch_touch(ctx, ctx->colors[i]);
   gex_wgpu_batch_touch(ctx, ctx->depth);
   gex_wgpu_batch_touch(ctx, index_resource);
   for (unsigned slot = 0; slot < vb_slot_count; ++slot)
      gex_wgpu_batch_touch(ctx, ctx->vertex_buffers[vb_buffer[slot]]);
   for (unsigned i = 0; i < ctx->vs->ubo_count; ++i)
      gex_wgpu_batch_touch(
         ctx,
         ctx->constant_buffers[MESA_SHADER_VERTEX][ctx->vs->ubo_slots[i]].buffer);
   for (unsigned i = 0; i < ctx->fs->ubo_count; ++i)
      gex_wgpu_batch_touch(
         ctx,
         ctx->constant_buffers[MESA_SHADER_FRAGMENT][ctx->fs->ubo_slots[i]].buffer);
   for (unsigned i = 0; i < ctx->fs->texture_count; ++i) {
      const unsigned slot = ctx->fs->texture_slots[i];
      if (ctx->sampler_views[slot])
         gex_wgpu_batch_touch(ctx, ctx->sampler_views[slot]->texture);
   }
   for (unsigned i = 0; i < ctx->vs->texture_count; ++i) {
      const unsigned slot = ctx->vs->texture_slots[i];
      if (ctx->vertex_sampler_views[slot])
         gex_wgpu_batch_touch(ctx, ctx->vertex_sampler_views[slot]->texture);
   }
   for (unsigned i = 0; i < ctx->so_target_count; ++i)
      if (ctx->so_targets[i])
         gex_wgpu_batch_touch(ctx, ctx->so_targets[i]->buffer);
   /* Bounded, so one command buffer cannot grow without limit and the GPU
    * always has something to be getting on with -- and submitted at once if
    * the batch ran out of room to keep this draw's resources alive. */
   if (++ctx->pending_draws >= GEX_WGPU_MAX_BATCHED_DRAWS ||
       ctx->batch_touched_overflowed)
      gex_wgpu_submit_pending(ctx);
   if (gex_timing) {
      gex_wgpu_time_submit += os_time_get_nano() - t_submit_start;
      gex_wgpu_time_draw += os_time_get_nano() - t_draw_start;
      gex_wgpu_report_timing(screen);
   }
}

static void
gex_wgpu_flush(struct pipe_context *pipe, struct pipe_fence_handle **fence,
               unsigned flags)
{
   GEX_DEVICE_GUARD(gex_screen(pipe->screen));
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   (void)pipe;
   (void)flags;
   if (fence)
      *fence = NULL;
}

static void
gex_wgpu_context_destroy(struct pipe_context *pipe)
{
   gex_wgpu_submit_pending((struct gex_wgpu_context *)pipe);
   struct gex_wgpu_context *ctx = (struct gex_wgpu_context *)pipe;
   for (unsigned i = 0; i < GEX_WGPU_MAX_SO_BUFFERS; ++i) {
      pipe_so_target_reference(&ctx->so_targets[i], NULL);
   pipe_resource_reference(&ctx->uploaded_indices, NULL);
      pipe_resource_reference(&ctx->shader_buffers[i].buffer, NULL);
   }
   if (ctx->blitter)
      util_blitter_destroy(ctx->blitter);
   if (ctx->primconvert)
      util_primconvert_destroy(ctx->primconvert);
   util_copy_framebuffer_state(&ctx->framebuffer, NULL);
   for (unsigned i = 0; i < ARRAY_SIZE(ctx->vertex_buffer_state); ++i)
      pipe_vertex_buffer_unreference(&ctx->vertex_buffer_state[i]);
   pipe_resource_reference(&ctx->constant_ring, NULL);
   FREE(ctx->constant_staging);
   /* The scratch the geometry path runs through. */
   pipe_resource_reference(&ctx->gs_vs_out, NULL);
   pipe_resource_reference(&ctx->gs_vertices, NULL);
   pipe_resource_reference(&ctx->gs_indices, NULL);
   if (ctx->occlusion_set) wgpuQuerySetRelease(ctx->occlusion_set);
   if (ctx->occlusion_resolve) wgpuBufferRelease(ctx->occlusion_resolve);
   if (ctx->occlusion_readback) wgpuBufferRelease(ctx->occlusion_readback);
   /* const_uploader aliases stream_uploader, so destroy it once. */
   if (ctx->base.stream_uploader) {
      u_upload_destroy(ctx->base.stream_uploader);
      ctx->base.stream_uploader = NULL;
      ctx->base.const_uploader = NULL;
   }
   for (unsigned i = 0; i < GEX_WGPU_MAX_COLOR_TARGETS; ++i)
      pipe_resource_reference(&ctx->colors[i], NULL);
   pipe_resource_reference(&ctx->depth, NULL);
   for (unsigned i = 0; i < ctx->vertex_buffer_count; ++i)
      pipe_resource_reference(&ctx->vertex_buffers[i], NULL);
   for (unsigned stage = 0; stage < MESA_SHADER_STAGES; ++stage) {
      for (unsigned slot = 0; slot < GEX_WGPU_MAX_CONSTANT_BUFFERS; ++slot)
         util_copy_constant_buffer(
            &ctx->constant_buffers[stage][slot], NULL);
   }
   for (unsigned slot = 0; slot < GEX_WGPU_MAX_SAMPLERS; ++slot)
      pipe_sampler_view_reference(&ctx->sampler_views[slot], NULL);
   p_atomic_dec(&pipe->screen->num_contexts);
   FREE(pipe);
}

static struct pipe_context *
gex_wgpu_context_create(struct pipe_screen *screen, void *priv, unsigned flags)
{
   /* The second context is the one that makes this a shared device; see
    * gex_wgpu_screen::device_lock. Never turned off again: a context that
    * went away may still have work in flight. */
   {
      struct gex_wgpu_screen *gex = gex_screen(screen);
      simple_mtx_lock(&gex->cache_lock);
      if (++gex->context_count > 1)
         gex->device_lock_needed = true;
      simple_mtx_unlock(&gex->cache_lock);
   }
   (void)flags;
   struct gex_wgpu_context *ctx = CALLOC_STRUCT(gex_wgpu_context);
   if (!ctx)
      return NULL;
   ctx->base.screen = screen;
   ctx->base.priv = priv;
   ctx->base.destroy = gex_wgpu_context_destroy;
   ctx->base.flush = gex_wgpu_flush;
   ctx->base.draw_vbo = gex_wgpu_draw_vbo;
   ctx->base.create_blend_state = gex_wgpu_create_blend_state;
   ctx->base.bind_blend_state = gex_wgpu_bind_blend_state;
   ctx->base.delete_blend_state = gex_wgpu_delete_blend_state;
   ctx->base.set_blend_color = gex_wgpu_set_blend_color;
   ctx->base.create_vs_state = gex_wgpu_create_vs_state;
   ctx->base.bind_vs_state = gex_wgpu_bind_vs_state;
   ctx->base.create_gs_state = gex_wgpu_create_gs_state;
   ctx->base.bind_gs_state = gex_wgpu_bind_gs_state;
   ctx->base.delete_gs_state = gex_wgpu_delete_gs_state;
   ctx->base.delete_vs_state = gex_wgpu_delete_shader_state;
   ctx->base.create_fs_state = gex_wgpu_create_fs_state;
   ctx->base.bind_fs_state = gex_wgpu_bind_fs_state;
   ctx->base.delete_fs_state = gex_wgpu_delete_shader_state;
   ctx->base.create_rasterizer_state = gex_wgpu_create_rasterizer_state;
   ctx->base.bind_rasterizer_state = gex_wgpu_bind_rasterizer_state;
   ctx->base.delete_rasterizer_state = gex_wgpu_delete_rasterizer_state;
   ctx->base.create_depth_stencil_alpha_state = gex_wgpu_create_dsa_state;
   ctx->base.bind_depth_stencil_alpha_state = gex_wgpu_bind_dsa_state;
   ctx->base.delete_depth_stencil_alpha_state = gex_wgpu_delete_dsa_state;
   ctx->base.set_stencil_ref = gex_wgpu_set_stencil_ref;
   ctx->base.create_sampler_state = gex_wgpu_create_sampler_state;
   ctx->base.bind_sampler_states = gex_wgpu_bind_sampler_states;
   ctx->base.delete_sampler_state = gex_wgpu_delete_sampler_state;
   ctx->base.create_sampler_view = gex_wgpu_create_sampler_view;
   ctx->base.sampler_view_destroy = gex_wgpu_sampler_view_destroy;
   ctx->base.sampler_view_release = gex_wgpu_sampler_view_release;
   ctx->base.set_sampler_views = gex_wgpu_set_sampler_views;
   ctx->base.create_vertex_elements_state = gex_wgpu_create_vertex_elements_state;
   ctx->base.bind_vertex_elements_state = gex_wgpu_bind_vertex_elements_state;
   ctx->base.delete_vertex_elements_state = gex_wgpu_delete_vertex_elements_state;
   ctx->base.set_framebuffer_state = gex_wgpu_set_framebuffer_state;
   ctx->base.set_viewport_states = gex_wgpu_set_viewport_states;
   ctx->base.set_scissor_states = gex_wgpu_set_scissor_states;
   ctx->base.set_vertex_buffers = gex_wgpu_set_vertex_buffers;
   ctx->base.set_constant_buffer = gex_wgpu_set_constant_buffer;
   /* Released rather than unreferenced by paths that allocate a one-off
    * resource -- u_primconvert's line-loop index buffer is one -- and left
    * null it is a call through a null function pointer, not a missing
    * feature. Nothing here needs more than the default. */
   ctx->base.resource_release = u_default_resource_release;
   ctx->base.buffer_subdata = gex_wgpu_buffer_subdata;
   ctx->base.buffer_map = gex_wgpu_buffer_map;
   ctx->base.buffer_unmap = gex_wgpu_buffer_unmap;
   ctx->base.transfer_flush_region = gex_wgpu_transfer_flush_region;
   ctx->base.clear = gex_wgpu_clear;
   ctx->base.set_sample_mask = gex_wgpu_set_sample_mask;
   ctx->base.resource_copy_region = gex_wgpu_resource_copy_region;
   ctx->base.blit = gex_wgpu_blit;
   ctx->base.set_polygon_stipple = gex_wgpu_set_polygon_stipple;
   ctx->base.set_min_samples = gex_wgpu_set_min_samples;
   ctx->base.set_clip_state = gex_wgpu_set_clip_state;
   ctx->base.set_active_query_state = gex_wgpu_set_active_query_state;
   ctx->base.create_query = gex_wgpu_create_query;
   ctx->base.destroy_query = gex_wgpu_destroy_query;
   ctx->base.begin_query = gex_wgpu_begin_query;
   ctx->base.end_query = gex_wgpu_end_query;
   ctx->base.get_query_result = gex_wgpu_get_query_result;
   ctx->base.set_debug_callback = gex_wgpu_set_debug_callback;
   ctx->base.create_stream_output_target = gex_wgpu_create_stream_output_target;
   ctx->base.stream_output_target_destroy = gex_wgpu_stream_output_target_destroy;
   ctx->base.set_stream_output_targets = gex_wgpu_set_stream_output_targets;
   ctx->base.set_shader_buffers = gex_wgpu_set_shader_buffers;
   ctx->base.create_compute_state = gex_wgpu_create_compute_state;
   ctx->base.bind_compute_state = gex_wgpu_bind_compute_state;
   ctx->base.delete_compute_state = gex_wgpu_delete_compute_state;
   ctx->base.launch_grid = gex_wgpu_launch_grid;
   ctx->base.flush_resource = gex_wgpu_flush_resource;
   ctx->base.memory_barrier = gex_wgpu_memory_barrier;
   ctx->base.texture_barrier = gex_wgpu_texture_barrier;
   ctx->base.render_condition = gex_wgpu_render_condition;
   ctx->base.texture_subdata = gex_wgpu_texture_subdata;
   ctx->base.clear_render_target = gex_wgpu_clear_render_target;
   ctx->base.clear_depth_stencil = gex_wgpu_clear_depth_stencil;
   ctx->base.texture_map = gex_wgpu_texture_map;
   ctx->base.texture_unmap = gex_wgpu_texture_unmap;

   /* Gallium's helpers allocate their staging buffers out of these, and they
    * are the driver's to create: u_vbuf reaches straight for
    * pipe_context::stream_uploader when it has to upload or translate vertex
    * data, without checking it, so leaving it null faults inside u_upload
    * rather than anywhere near this driver.
    */
   ctx->base.stream_uploader = u_upload_create_default(&ctx->base);
   if (!ctx->base.stream_uploader) {
      ctx->base.destroy(&ctx->base);
      return NULL;
   }
   /* One pool for both: this driver has no separate constant-buffer heap. */
   ctx->base.const_uploader = ctx->base.stream_uploader;

   ctx->primconvert = util_primconvert_create(&ctx->base, GEX_WGPU_PRIM_MODES);
   if (!ctx->primconvert) {
      ctx->base.destroy(&ctx->base);
      return NULL;
   }
   /* Created last: it builds shaders and state objects through this very
    * context, so every hook it uses has to be in place first. A failure here
    * is not fatal -- blits that need it will refuse, which is what they did
    * before it existed. */
   ctx->blitter = util_blitter_create(&ctx->base);

   p_atomic_inc(&screen->num_contexts);
   return &ctx->base;
}

static void
gex_wgpu_screen_destroy(struct pipe_screen *base)
{
   simple_mtx_destroy(&gex_screen(base)->cache_lock);
   simple_mtx_destroy(&gex_screen(base)->queue_lock);
   mtx_destroy(&gex_screen(base)->device_lock);
   struct gex_wgpu_screen *screen = gex_screen(base);
   if (screen->queue)
      wgpuQueueRelease(screen->queue);
   FREE(screen);
   glsl_type_singleton_decref();
}

/* What this driver can actually do.
 *
 * st/mesa derives the advertised GL version, every limit and every lowering
 * decision from these; a screen that leaves them zero -- as this one did --
 * cannot have a context created against it at all.
 *
 * The numbers below describe the WebGPU feature set this driver currently
 * implements, not WebGPU's own ceiling and not OpenGL 4.4.  Anything unproven
 * is left at zero on purpose: a capability advertised here is one st/mesa will
 * exercise, and the corresponding pipe_context hook has to exist and be
 * correct.  Raise these as families are proven, never ahead of them.
 */
/* Fences.
 *
 * gex_wgpu_flush never produces one -- every submission this driver makes is
 * already ordered on the WebGPU queue and readback waits on it explicitly --
 * so a fence handle is always NULL.  These still have to exist: st/mesa's
 * throttling holds a ring of fence slots and releases them at context
 * teardown through screen->fence_reference, without null-checking it.
 */
static void
gex_wgpu_fence_reference(struct pipe_screen *screen,
                         struct pipe_fence_handle **dst,
                         struct pipe_fence_handle *src)
{
   (void)screen;
   *dst = src;
}

static bool
gex_wgpu_fence_finish(struct pipe_screen *screen, struct pipe_context *ctx,
                      struct pipe_fence_handle *fence, uint64_t timeout)
{
   (void)screen;
   (void)ctx;
   (void)timeout;
   /* A NULL fence is already signalled by construction. */
   return fence == NULL;
}

static void
gex_wgpu_init_caps(struct gex_wgpu_screen *screen)
{
   struct pipe_caps *caps = (struct pipe_caps *)&screen->base.caps;

   /* Gallium's GL-minimum defaults first, then this driver's answers on top.
    * Every other driver does this, and skipping it left a long tail of caps at
    * zero that only surfaced one confusing failure at a time -- a zero
    * max_vertex_buffers made u_vbuf treat every vertex element as
    * incompatible, and a zero max_constant_buffer_size withheld
    * ARB_uniform_buffer_object. The argument is the number of vertex-buffer
    * slots that also accept a stream-output binding, which is none here. */
   u_init_pipe_screen_caps(&screen->base, 0);

   /* GLSL 1.30 is the floor for a core-profile context and for GLSL ES 3.00,
    * which is what the guest's GLES driver emits.  This is provisional: it
    * must end up derived from the families that actually pass rather than
    * chosen to make a context appear. */
   /* GLSL 1.50 is the version GL 3.2 compiles at. What it adds over 1.40 that
    * a driver has to supply is the geometry stage, which this driver runs as
    * a compute pass; the shapes it does not implement -- instancing, streams
    * past zero, inputs or outputs other than triangles and triangle strips --
    * are refused when the shader is created rather than mis-rendered. */
   /* 330 once the last GL 3.3 extension landed. What it adds over 1.50 is
    * almost entirely front-end -- explicit attribute and output locations,
    * the bit-encoding and pack/unpack builtins -- which the GLSL compiler and
    * NIR handle without driver support. The one part that needed the driver
    * is the 2_10_10_10 vertex type, and that is what
    * gex_wgpu_lower_vertex_attrib_unpack implements.
    *
    * Held to the standard the note above asks for: this is set because the
    * families that need it pass, not to make a context appear. The gl44
    * `multi-draw` and `uniform` probes refuse to compile below 330 and both
    * pass here; the corpus, the other probes and SuperTuxKart are unchanged
    * by the bump. */
   /* GL's alpha test is a fixed-function stage WebGPU does not have, and
    * gex_wgpu_create_dsa_state refuses any state that enables it. Left
    * advertised, st/mesa believes the driver implements it and sends the
    * state anyway: the refusal leaves ctx->dsa NULL and the draw silently
    * renders with the wrong depth and stencil configuration. Saying no here
    * makes st lower the test into the fragment shader as a discard, which is
    * both what WebGPU can run and exactly what GL specifies.
    *
    * Minecraft 1.8 is the case that found this -- alpha test is how a GL 1.x
    * renderer cuts out its font glyphs and foliage -- and nothing in the
    * browser had reached it, core profile having removed the feature. */
   caps->alpha_test = false;
   /* Doubles are deliberately not advertised, and the reason is circular.
    *
    * Mesa can lower fp64 in software -- float64.glsl and nir_lower_doubles,
    * which st runs by itself once lower_doubles_options asks for it. But st
    * only *builds* that shader when GLSL is 4.00 or better, and Mesa clamps
    * the GLSL version to the GL version, which stays at 3.3 until the whole
    * GL 4.0 extension set is present -- fp64 among it. So fp64 cannot be
    * delivered on its own; it arrives with the rung or not at all.
    *
    * Setting caps.doubles anyway is worse than leaving it: st advertises
    * ARB_gpu_shader_fp64, an application writes a double, and the draw dies
    * on `assert(softfp64 != NULL)` inside nir_lower_double_ops instead of
    * failing to compile. Measured, not guessed. */
   /* ARB_sample_shading. The rate is the shader's business in WebGPU (see
    * gex_wgpu_set_min_samples), gl_SampleID and gl_SampleMask reach WGSL as
    * @builtin(sample_index) and @builtin(sample_mask) through the emitter's
    * existing SpvBuiltInSampleId path, and 4x multisample targets already
    * work. */
   caps->sample_shading = true;
   /* ARB_texture_query_lod. WGSL has no textureQueryLod, so the value is
    * computed from the coordinate's derivatives instead; see
    * gex_wgpu_lower_query_lod_instr. */
   caps->texture_query_lod = true;
   caps->glsl_feature_level = 330;
   caps->glsl_feature_level_compatibility = 330;

   /* ARB_texture_gather, exposed below its GL 4.0 core version.
    *
    * WGSL has textureGather and zink's SPIR-V emitter already emits
    * nir_texop_tg4, so the only thing standing between this and working was
    * the cap st reads. The view-swizzle lowering already has its tg4 case --
    * a gather selects one channel, so its swizzle is a component remap and
    * not a shuffle of the result -- and the LOD-bias pass already leaves
    * gather alone, which Gallium requires.
    *
    * Four components: the 3-argument textureGather(sampler, coord, comp)
    * belongs to ARB_gpu_shader5, but the cap is what st reports through
    * GL_MAX_PROGRAM_TEXTURE_GATHER_COMPONENTS_ARB and 1 would claim red-only.
    * The offset range is WebGPU's own const-offset range for textureGather,
    * [-8, 7], which is exactly GL's required minimum. */
   /* ARB_texture_cube_map_array, likewise below its core version. WebGPU has
    * the `cube-array` view dimension and WGSL has texture_cube_array, and the
    * driver already routes PIPE_TEXTURE_CUBE_ARRAY through resource creation,
    * view creation (WGPUTextureViewDimension_CubeArray) and the sampler-view
    * path -- the cap was the only piece missing. It is how point-light shadow
    * maps and reflection probes are stored, so it is the cube-map feature
    * real renderers reach for. */
   caps->cube_map_array = true;

   caps->max_texture_gather_components = 4;
   caps->min_texture_gather_offset = -8;
   caps->max_texture_gather_offset = 7;

   caps->npot_textures = true;
   caps->mixed_framebuffer_sizes = true;
   caps->mixed_color_depth_bits = true;
   caps->texture_swizzle = true;
   caps->fragment_shader_texture_lod = true;
   /* Depth-comparison sampling, which WebGPU spells textureSampleCompare and
    * this driver already emits: the shadow-compare cases in the Gallium corpus
    * cover implicit, explicit and gradient lookups. ARB_shadow is the first
    * rung above desktop GL 1.3. */
   caps->texture_shadow_map = true;
   /* Occlusion queries are WebGPU's occlusion query set; conditional render
    * is the GL feature built on top of them. */
   caps->occlusion_query = true;
   /* ARB_draw_indirect. The draw path resolves these on the CPU with
    * util_draw_indirect -- see the note there. */
   caps->draw_indirect = true;
   /* ARB_timer_query, one of the four things GL 3.3 wants. CPU-timed -- see
    * the note in gex_wgpu_create_query. */
   caps->query_timestamp = true;
   caps->query_time_elapsed = true;
   caps->conditional_render = true;
   /* Geometry shaders, run as a compute pass. The output is allocated
    * worst-case per input primitive, so these are what bound that allocation
    * rather than a hardware limit. GL 3.2 asks for 256 and 1024. */
   /* ARB_depth_clamp, where the device has WebGPU's optional
    * depth-clip-control. Not the separate near/far form: WebGPU has one flag
    * for both, and create_rasterizer_state refuses a state that splits them. */
   caps->depth_clip_disable = screen->depth_clip_control;
   caps->depth_clip_disable_separate = false;
   /* flatshade and two_sided_color are left at u_screen's true. Turning them
    * off makes st lower both in NIR, and the lowering reaches an ALU opcode
    * zink's SPIR-V emitter does not implement -- which aborts the module.
    * Neither is reachable in a core profile anyway (glShadeModel and
    * two-sided lighting are compatibility-only), so st never sets them; only
    * u_blitter does, and create_rasterizer_state accepts and ignores its
    * flatshade rather than refusing the whole state. */
   /* WebGPU cube sampling is seamless and cannot be turned off, which is what
    * every Vulkan-backed driver reports too. The per-texture form stays
    * unadvertised precisely because it cannot be turned off. */
   /* ARB_texture_multisample: a shader can fetch a multisampled texture with
    * textureLoad and a sample index. WebGPU allows the TextureBinding usage on
    * a multisampled texture and WGSL has texture_multisampled_2d, so this is
    * the ordinary path rather than an emulation. */
   caps->texture_multisample = true;
   /* dest_surface_srgb_control is deliberately not set. st/mesa gates
    * EXT_framebuffer_sRGB on it, and 1909 "FBO is not complete ... with
    * sizedFormat: GL_SRGB8_ALPHA8" made it look like the missing piece --
    * a defect that is ours, since the MobileGL backend logs none of them.
    * Setting it changes the run not at all -- measured twice, and the second
    * time with the module healthy: 169 failures either way, zero throws
    * either way. (The completeness failures it was meant to fix turned out to
    * be crash fallout, not an sRGB problem at all.) Advertising it would be a
    * capability claim no passing test supports, which is the one rule this
    * file keeps. */
   caps->seamless_cube_map = true;
   caps->seamless_cube_map_per_texture = false;
   /* GL requires GL_MAX_TEXTURE_LOD_BIAS to be at least 2.0. Leaving it zero
    * reports a value the specification forbids, and the shader variant that
    * applies a sampler's lod_bias already handles a non-zero one. */
   caps->max_texture_lod_bias = 2.0f;
   caps->max_geometry_output_vertices = 256;
   caps->max_geometry_total_output_components = 1024;
   /* Primitive restart, which GL 3.1 needs as NV_primitive_restart. WebGPU has
    * it only for strip topologies and only at a fixed index, so draw_vbo
    * resolves the general form on the CPU with
    * util_draw_vbo_without_prim_restart -- the index buffer is scanned and the
    * draw re-issued as the ranges between restarts. */
   /* A mapped buffer is a CPU shadow that reaches the GPU only when a range
    * is flushed, so it is neither persistent nor coherent.
    *
    * u_init_pipe_screen_caps defaults this to true, which is right for a
    * driver whose mapped pointer *is* GPU memory and catastrophic here:
    * u_upload_mgr reads it, decides no flush is needed, and never calls
    * transfer_flush_region. Everything uploaded that way -- u_vbuf's
    * translated vertex buffers, u_primconvert's generated indices, st/mesa's
    * uploaded user buffers -- then reached the GPU as whatever the buffer
    * held before, with no error anywhere. */
   caps->buffer_map_persistent_coherent = false;
   caps->supported_prim_modes = GEX_WGPU_PRIM_MODES;
   /* Restart is resolved on the CPU by splitting the draw, which works for
    * every topology the driver takes. */
   caps->supported_prim_modes_with_restart = GEX_WGPU_PRIM_MODES;
   caps->primitive_restart = true;
   caps->primitive_restart_fixed_index = true;
   /* Transform feedback runs as a compute pass.
    *
    * The device reports maxStorageBuffersInVertexStage, which made a vertex
    * shader writing its own captures look possible, but that limit counts
    * *read-only* storage buffers. WGSL forbids the writable kind there, and
    * Tint says so exactly:
    *
    *   var with 'storage' address space and 'read_write' access mode
    *   cannot be used by vertex pipeline stage
    *
    * So the vertex shader is run a second time as a compute shader that
    * writes the captures -- the same shape Asahi uses for geometry stages --
    * with the vertex fetch a compute stage does not get emulated against the
    * vertex buffers. See gex_wgpu_capture_variant_get.
    *
    * Advertised because the Gallium corpus executes it: the
    * transform-feedback case builds a capturing vertex shader by hand and
    * checks the captured buffer component by component.
    */
   caps->max_stream_output_buffers = GEX_WGPU_MAX_SO_BUFFERS;
   /* ARB_transform_feedback2: pause/resume, append offsets and
    * DrawTransformFeedback, all of which ride on the per-target byte total
    * kept in gex_wgpu_so_target. */
   caps->stream_output_pause_resume = true;
   /* Both of these defaulted to zero, which st/mesa reports verbatim as
    * MAX_TRANSFORM_FEEDBACK_*_COMPONENTS -- so every program declaring a
    * feedback varying failed to link, however few it declared. GL 3.2 asks
    * for 64 interleaved and 4 separate, and the capture path carries exactly
    * GEX_WGPU_MAX_VARYINGS vec4 slots, which is 64 components: the floor the
    * spec sets and the ceiling this layout has happen to be the same number,
    * so this claims what the buffer can actually hold rather than a minimum
    * borrowed from the specification. */
   caps->max_stream_output_interleaved_components = GEX_WGPU_MAX_VARYINGS * 4;
   /* ARB_transform_feedback3: gl_NextBuffer and gl_SkipComponents, which let
    * one interleaved capture be split across several buffers. The linker
    * expresses that entirely in the xfb info the capture pass already walks
    * -- a buffer per entry, each with its own stride -- so it needs no new
    * machinery here, only the admission that it works. */
   caps->stream_output_interleave_buffers = true;
   caps->max_stream_output_separate_components = 4;
   /* rgba16float is filterable in core WebGPU; the 32-bit float formats are
    * only filterable with the float32-filterable feature, which this driver
    * does not request. */
   caps->texture_half_float_linear = true;
   caps->texture_float_linear = false;
   caps->fragment_shader_derivatives = true;
   caps->vertex_color_unclamped = true;
   caps->fragment_color_clamped = false;
   caps->vertex_color_clamped = false;

   /* WebGPU's guaranteed minimums, which is what a portable page may assume.
    * maxTextureDimension2D 8192, maxTextureArrayLayers 256. */
   caps->max_texture_2d_size = 8192;
   caps->max_texture_3d_levels = 12;    /* 2048 */
   caps->max_texture_cube_levels = 14;  /* 8192 */
   caps->max_texture_array_layers = 256;

   /* One WebGPU render pass can hold eight colour attachments; this driver
    * validates and binds up to its own maximum. */
   caps->max_render_targets = GEX_WGPU_MAX_COLOR_TARGETS;
   /* ARB_blend_func_extended, when the device granted the feature. st/mesa
    * turns this into a fragment shader with two blend sources on one colour
    * target -- `nir_lower_fragcolor` above already special-cases that shape --
    * and the SRC1 blend factors below become reachable. */
   caps->max_dual_source_render_targets = screen->dual_source_blending ? 1 : 0;
   caps->indep_blend_enable = true;
   caps->indep_blend_func = true;
   caps->blend_equation_separate = true;

   /* u_vbuf turns this into its allowed-buffer mask; left at zero it decides
    * every vertex element is incompatible and never creates a driver vertex
    * layout at all, so draws arrive with no layout bound. */
   /* Eight, matching what WebGPU binds.
    *
    * Sixteen would clear u_vbuf's fallback_always -- it sets that below
    * sixteen, and then runs its whole per-draw path for every draw of every
    * application -- so this looked like an easy win. Measured, it is a loss:
    * 110000 draws/s became 76000. u_vbuf's fallback is doing work this
    * driver needs done, and doing it more cheaply than the paths that
    * replace it. Left as it was, with the reason written down so the next
    * person does not try it twice. */
   /* Eight, matching what WebGPU binds.
    *
    * Sixteen looks like a free win: u_vbuf sets fallback_always below that
    * and then runs its whole per-draw path -- analysis, uploads, a CPU scan
    * of the index buffer -- for every draw of every application. Clearing it
    * needs this and every primitive mode advertised, and measured on a quiet
    * machine both together changed nothing: 108875 draws/s became 110235 in
    * the microbenchmark, and Minecraft stayed at 20 fps. Not worth claiming
    * modes and buffer counts the driver does not have. Written down so the
    * next person does not spend the afternoon on it. */
   caps->max_vertex_buffers = 8;
   caps->max_vertex_attrib_stride = 2048;
   /* Per-instance vertex attributes, which is ARB_instanced_arrays.
    *
    * Left false, st/mesa never exposed the extension, so every
    * `glVertexAttribDivisor` an application made was rejected with
    * GL_INVALID_OPERATION and quietly did nothing. SuperTuxKart calls it for
    * locations 8..12 and then draws with a shader that reads those as
    * per-instance data; with the divisors dropped they were fetched per vertex
    * from a buffer that holds no such data, `i_scale` came back zero, and
    * since the shader computes `i_position * i_scale + i_origin` every object
    * in the scene collapsed onto a point. Ninety thousand draws a frame
    * rasterised nothing and only the skybox and the HUD were ever visible.
    *
    * WebGPU expresses divisor 1 as `stepMode: instance`, and the draw path
    * gives each step rate its own vertex buffer slot bound to the same buffer,
    * so this is now supported. A divisor above 1 has no WebGPU equivalent and
    * is still refused at `create_vertex_elements_state` -- out loud. */
   caps->vertex_element_instance_divisor = true;
   caps->user_vertex_buffers = false;
   /* gl_InstanceID reaches WGSL as @builtin(instance_index), proven by the
    * vertex-gl-instance-id corpus case.  SPIR-V's InstanceIndex counts from
    * the draw's base instance while GL's gl_InstanceID counts from zero, so
    * these two must move together: advertising start_instance without
    * subtracting the base would make gl_InstanceID wrong. */
   caps->start_instance = false;
   caps->vs_instanceid = true;

   /* Uniforms reach the shader as uniform buffers; 256 is WebGPU's required
    * minUniformBufferOffsetAlignment. */
   caps->constant_buffer_offset_alignment = 256;
   /* WebGPU guarantees maxUniformBufferBindingSize of 64 KiB. st/mesa refuses
    * ARB_uniform_buffer_object below 16 KiB, and OpenGL ES 3.0 requires it. */
   caps->max_constant_buffer_size = 65536;
   caps->min_map_buffer_alignment = 64;

   caps->max_varyings = 16;
   caps->max_viewports = 1;
   caps->clip_halfz = true;
   caps->fs_coord_origin_upper_left = true;
   caps->fs_coord_pixel_center_half_integer = true;
   caps->fs_coord_origin_lower_left = false;
   caps->fs_coord_pixel_center_integer = false;

   caps->min_point_size = 1.0f;
   caps->max_point_size = 1.0f;
   caps->min_line_width = 1.0f;
   caps->max_line_width = 1.0f;
   caps->point_size_granularity = 0.1f;
   caps->line_width_granularity = 0.1f;

   caps->uma = false;
   caps->endianness = PIPE_ENDIAN_LITTLE;

   /* Deliberately absent, because the hooks behind them are: occlusion and
    * timer queries, transform feedback, shader storage and images, compute,
    * geometry and tessellation, texture buffers, conditional render,
    * multisample, seamless cube maps and primitive restart. */
}

static void
gex_wgpu_init_shader_caps(struct gex_wgpu_screen *screen)
{
   /* Vertex, fragment and geometry. WebGPU's pipeline has only the first two;
    * the geometry stage is emulated by running the shader as compute, so it
    * gets the same limits rather than a smaller set -- what constrains it is
    * the buffer the emulation writes, which the geometry caps below describe.
    * Tessellation still has no emulation and so is still absent. */
   const mesa_shader_stage stages[] = { MESA_SHADER_VERTEX,
                                        MESA_SHADER_FRAGMENT,
                                        MESA_SHADER_GEOMETRY };
   for (unsigned i = 0; i < ARRAY_SIZE(stages); ++i) {
      struct pipe_shader_caps *caps =
         (struct pipe_shader_caps *)&screen->base.shader_caps[stages[i]];

      /* NIR is compiled to SPIR-V and then to WGSL, so the instruction counts
       * are not a hardware limit; report the large values st/mesa expects from
       * a compiler-backed driver rather than inventing a small one. */
      caps->max_instructions = 16384;
      caps->max_alu_instructions = 16384;
      caps->max_tex_instructions = 16384;
      caps->max_tex_indirections = 16384;
      caps->max_control_flow_depth = 64;
      caps->max_inputs = stages[i] == MESA_SHADER_VERTEX ? 16 : 16;
      /* st/mesa reports MaxGeometryOutputComponents as four times this, and
       * GL 3.2 requires 128. Eight slots gave 32 -- a quarter of the floor --
       * and transform_feedback's geometry cases failed at link time with
       * "geometry shader uses too many output components (60 > 32)". The
       * geometry stage is emulated as a compute pass writing a buffer, so
       * what actually bounds it is that buffer, not a hardware slot count. */
      caps->max_outputs = stages[i] == MESA_SHADER_FRAGMENT
         ? 8 : GEX_WGPU_MAX_VARYINGS;
      caps->max_temps = 256;
      caps->max_const_buffer0_size = 16 * 1024;
      /* Never more than the device will bind. st/mesa needs 13 for
       * ARB_uniform_buffer_object (12 blocks plus GL's default block); a
       * device that offers only WebGPU's guaranteed 12 cannot have it, and
       * saying otherwise would fail at pipeline creation instead. */
      /* Uniform buffers first, then storage buffers for the slots past the
       * device's uniform limit. st/mesa reports MaxUniformBlocks as one less
       * than this, so twelve uniform buffers alone leave eleven blocks and
       * ARB_uniform_buffer_object -- GL 3.1 -- wants twelve. */
      caps->max_const_buffers =
         MIN2(GEX_WGPU_MAX_CONSTANT_BUFFERS,
              screen->max_uniform_buffers + screen->max_storage_buffers_vertex);
      caps->max_texture_samplers = GEX_WGPU_MAX_SAMPLERS;
      caps->max_sampler_views = GEX_WGPU_MAX_SAMPLERS;
      caps->max_shader_buffers = 0;
      caps->max_shader_images = 0;
      caps->max_hw_atomic_counters = 0;
      caps->max_hw_atomic_counter_buffers = 0;
      caps->supported_irs = 1 << PIPE_SHADER_IR_NIR;
      caps->cont_supported = true;
      caps->indirect_temp_addr = false;
      caps->indirect_const_addr = true;
      caps->integers = true;
      caps->fp16 = false;
      caps->int16 = false;
   }
}

/* NIR lowering this backend needs.
 *
 * These were previously a single all-zero struct shared by every stage, which
 * asks NIR to lower nothing.  The corpus survived that by hand-writing NIR
 * that needed no lowering; shaders st/mesa produces from GLSL do not.
 */
static void
gex_wgpu_init_nir_options(struct gex_wgpu_screen *screen)
{
   static const nir_shader_compiler_options options = {
      /* SPIR-V, and WGSL after it, have no native forms for these. */
      .lower_fdiv = true,
      .lower_flrp16 = true,
      .lower_flrp32 = true,
      .lower_flrp64 = true,
      .lower_fmod = true,
      .lower_bitfield_extract = true,
      .lower_bitfield_insert = true,
      .lower_uadd_carry = true,
      .lower_usub_borrow = true,
      .lower_fdph = true,
      /* The rest of what zink's emitter cannot spell. These are not
       * hypothetical: nir_to_spirv has no case for any of them, so an
       * unlowered one reaches emit_alu and aborts the process --
       * `seq/sne/slt/sge` did exactly that on the first native glxgears,
       * because Mesa's *fixed-function* vertex program is written in the
       * ARB_vertex_program instruction set and those four are its
       * comparisons. Nothing in the browser had reached them before, since
       * every application tested there brought its own GLSL.
       *
       * The list is zink's, because the emitter is zink's: an op it lowers
       * away is an op it cannot emit. The one deliberate difference is
       * lower_fsat, which nir_to_spirv does handle (as a clamp) and which is
       * better left for it than expanded into two more instructions here. */
      /* 64-bit anything, in software.
       *
       * WGSL has neither f64 nor i64, so every 64-bit operation has to become
       * a pair of 32-bit ones. Mesa already knows how: st runs
       * nir_lower_doubles against src/compiler/glsl/float64.glsl -- the same
       * software fp64 Intel ships -- and nir_lower_int64 for the integer
       * halves that lowering then produces. The driver's part is to ask for
       * it, which is these two masks and caps.doubles.
       *
       * ARB_gpu_shader_fp64 is one requirement of GL 4.0; the rung also wants
       * a tessellator, which nothing has written yet. */
      .lower_doubles_options = (nir_lower_doubles_options)(
         nir_lower_drcp | nir_lower_dsqrt | nir_lower_drsq | nir_lower_dtrunc |
         nir_lower_dfloor | nir_lower_dceil | nir_lower_dfract |
         nir_lower_dround_even | nir_lower_dmod | nir_lower_dsub |
         nir_lower_ddiv | nir_lower_dsign | nir_lower_dminmax |
         nir_lower_dsat | nir_lower_fp64_full_software),
      .lower_int64_options = (nir_lower_int64_options)(
         nir_lower_imul64 | nir_lower_isign64 | nir_lower_divmod64 |
         nir_lower_imul_high64 | nir_lower_bcsel64 | nir_lower_icmp64 |
         nir_lower_iadd64 | nir_lower_iabs64 | nir_lower_ineg64 |
         nir_lower_logic64 | nir_lower_minmax64 | nir_lower_shift64 |
         nir_lower_imul_2x32_64 | nir_lower_extract64 | nir_lower_ufind_msb64 |
         nir_lower_bit_count64 | nir_lower_find_lsb64 | nir_lower_conv64 |
         nir_lower_uadd_sat64 | nir_lower_usub_sat64 | nir_lower_iadd_sat64 |
         nir_lower_bitfield_reverse64 | nir_lower_bitfield_extract64),
      .lower_scmp = true,
      .lower_hadd = true,
      .lower_iadd_sat = true,
      .lower_uadd_sat = true,
      .lower_usub_sat = true,
      .lower_mul_high = true,
      .lower_fisnormal = true,
      .lower_fsat = false,
      .lower_extract_byte = true,
      .lower_extract_word = true,
      .lower_insert_byte = true,
      .lower_insert_word = true,
      .lower_pack_half_2x16 = true,
      .lower_pack_unorm_2x16 = true,
      .lower_pack_snorm_2x16 = true,
      .lower_pack_unorm_4x8 = true,
      .lower_pack_snorm_4x8 = true,
      .lower_unpack_half_2x16 = true,
      .lower_unpack_unorm_2x16 = true,
      .lower_unpack_snorm_2x16 = true,
      .lower_unpack_unorm_4x8 = true,
      .lower_unpack_snorm_4x8 = true,
      /* Uniforms reach the shader as a UBO, which is how this driver binds
       * constant buffers. */
      .lower_uniforms_to_ubo = true,
      /* Keep gl_ClipDistance as a float array. Without this the GLSL linker
       * packs it into a vec4[2] named gl_ClipDistanceMESA and zink's SPIR-V
       * emitter decorates that as BuiltIn ClipDistance -- which is a vec4
       * array where SPIR-V requires a scalar one, and Tint rejects the
       * module. clip_distance's two functional cases fail on exactly that. */
      .compact_arrays = true,
      /* Deliberately not vertex_id_zero_based or lower_base_vertex: the
       * SPIR-V emitter below implements load_vertex_id and load_base_vertex,
       * and asking NIR for the zero-based form instead produces
       * load_vertex_id_zero_base, which it does not. */
   };
   for (unsigned stage = 0; stage < MESA_SHADER_MESH_STAGES; ++stage)
      screen->base.nir_options[stage] = &options;
}

struct pipe_screen *
gex_wgpu_screen_create(WGPUDevice device)
{
   if (!device)
      return NULL;
   glsl_type_singleton_init_or_ref();
   struct gex_wgpu_screen *screen = CALLOC_STRUCT(gex_wgpu_screen);
   if (!screen) {
      glsl_type_singleton_decref();
      return NULL;
   }
   screen->device = device;
   WGPULimits limits = WGPU_LIMITS_INIT;
   screen->max_uniform_buffers = 12;      /* WebGPU's guaranteed minimum */
   screen->max_storage_buffers_vertex = 0;
   const bool got_limits =
      wgpuDeviceGetLimits(device, &limits) == WGPUStatus_Success;
   if (got_limits) {
      screen->max_uniform_buffers = limits.maxUniformBuffersPerShaderStage;
      /* The per-stage storage limits are newer than the single
       * maxStorageBuffersPerShaderStage, and an implementation that does not
       * report one leaves WGPU_LIMIT_U32_UNDEFINED there rather than a value.
       * Taken literally that is 4294967295, which turned a capability sum into
       * an overflow and silently *lowered* a cap. Fall back to the limit that
       * is always present. */
      screen->max_storage_buffers_vertex =
         limits.maxStorageBuffersInVertexStage == WGPU_LIMIT_U32_UNDEFINED
            ? limits.maxStorageBuffersPerShaderStage
            : limits.maxStorageBuffersInVertexStage;
      if (screen->max_storage_buffers_vertex == WGPU_LIMIT_U32_UNDEFINED)
         screen->max_storage_buffers_vertex = 0;
   }
   /* The caps are sized from these, so a limit the device did not grant is a
    * GL feature the frontend will not enable. Report them rather than leaving
    * a capability shortfall to be inferred from a missing extension. */
   fprintf(stderr,
           "gex-wgpu: device limits%s uniform/stage %u storage/vertex %u "
           "storage/stage %u\n", got_limits ? "" : " (query failed)",
           screen->max_uniform_buffers, screen->max_storage_buffers_vertex,
           limits.maxStorageBuffersPerShaderStage);
   screen->depth32_stencil8 =
      wgpuDeviceHasFeature(device, WGPUFeatureName_Depth32FloatStencil8);
   /* ARB_depth_clamp: WebGPU calls it unclippedDepth, and it is optional. */
   screen->depth_clip_control =
      wgpuDeviceHasFeature(device, WGPUFeatureName_DepthClipControl);
   screen->rg11b10_renderable =
      wgpuDeviceHasFeature(device, WGPUFeatureName_RG11B10UfloatRenderable);
   screen->texture_compression_bc =
      wgpuDeviceHasFeature(device, WGPUFeatureName_TextureCompressionBC);
   screen->max_colour_bytes_per_sample =
      got_limits && limits.maxColorAttachmentBytesPerSample
         ? limits.maxColorAttachmentBytesPerSample : 32u;
   /* Both gate the 16-bit normalized formats: they are backed by float32, and
    * a float32 attachment is only filterable and blendable when the device
    * granted these. Without them the formats stay unadvertised rather than
    * becoming a validation error the first time something samples or blends
    * one. */
   screen->float32_filterable =
      wgpuDeviceHasFeature(device, WGPUFeatureName_Float32Filterable);
   screen->float32_blendable =
      wgpuDeviceHasFeature(device, WGPUFeatureName_Float32Blendable);
   screen->dual_source_blending =
      wgpuDeviceHasFeature(device, WGPUFeatureName_DualSourceBlending);
   /* Real GPU timestamps, if the device has them. ARB_timer_query is one of
    * the two things GL 3.3 still wants, and the alternative -- answering from
    * the CPU clock -- measures the gap between commands being recorded rather
    * than the work being done, because submission here is asynchronous. With
    * this feature the question can be answered properly instead. */
   screen->timestamp_query =
      wgpuDeviceHasFeature(device, WGPUFeatureName_TimestampQuery);
   fprintf(stderr,
           "gex-wgpu: optional features bc=%d depth32s8=%d depthclip=%d "
           "rg11b10=%d f32filter=%d f32blend=%d dualsrc=%d timestamp=%d\n",
           (int)screen->texture_compression_bc, (int)screen->depth32_stencil8,
           (int)screen->depth_clip_control, (int)screen->rg11b10_renderable,
           (int)screen->float32_filterable, (int)screen->float32_blendable,
           (int)screen->dual_source_blending, (int)screen->timestamp_query);
   screen->queue = wgpuDeviceGetQueue(device);
   if (!screen->queue) {
      FREE(screen);
      glsl_type_singleton_decref();
      return NULL;
   }
   simple_mtx_init(&screen->cache_lock, mtx_plain);
   simple_mtx_init(&screen->queue_lock, mtx_plain);
   mtx_init(&screen->device_lock, mtx_plain | mtx_recursive);
   screen->base.destroy = gex_wgpu_screen_destroy;
   gex_wgpu_init_nir_options(screen);
   gex_wgpu_init_caps(screen);
   gex_wgpu_init_shader_caps(screen);
   screen->base.get_name = gex_wgpu_get_name;
   screen->base.get_vendor = gex_wgpu_get_vendor;
   screen->base.get_device_vendor = gex_wgpu_get_device_vendor;
   screen->base.is_format_supported = gex_wgpu_is_format_supported;
   /* Required alongside caps.query_timestamp: st/mesa calls this for
    * glGetInteger64v(GL_TIMESTAMP). Same CPU clock as the timer queries. */
   screen->base.get_timestamp = u_default_get_timestamp;
   screen->base.fence_reference = gex_wgpu_fence_reference;
   screen->base.fence_finish = gex_wgpu_fence_finish;
   screen->base.context_create = gex_wgpu_context_create;
   screen->base.resource_create = gex_wgpu_resource_create;
   screen->base.resource_destroy = gex_wgpu_resource_destroy;
   return &screen->base;
}
