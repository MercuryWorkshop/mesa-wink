/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_CMD_BUFFER_H
#define WVK_CMD_BUFFER_H

#include "wvk_private.h"

#include "vk_command_buffer.h"
#include "vk_graphics_state.h"

#include "util/u_dynarray.h"

#include "wvk_query.h"

struct wvk_cmd_buffer;
struct wvk_device;
struct wvk_device_memory;
struct wvk_image;
struct wvk_queue;

/* A range of host-visible memory a submit reads, so it is uploaded
 * before, or writes, so it is read back after. */
struct wvk_mem_range {
   struct wvk_device_memory *mem;
   uint64_t offset;
   uint64_t size;
};

/* What one submit becomes: one WGPUCommandEncoder, the host memory it
 * touches, and what to release once it has run. */
/* The slots of a replay's scratch buffers (wvk_replay::scratch): of a
 * mesh draw's, a tessellated draw's (and its geometry shader's), and a
 * geometry shader draw's. */
enum wvk_scratch_slot {
   WVK_SCRATCH_MESH = 0,
   WVK_SCRATCH_TESS = WVK_SCRATCH_MESH + 6,
   WVK_SCRATCH_GEOMETRY = WVK_SCRATCH_TESS + 9,
   WVK_SCRATCH_SLOTS = WVK_SCRATCH_GEOMETRY + 3,
};

struct wvk_scratch_clear {
   WGPUBuffer buffer;
   uint64_t size;
};

/* A texture of the submit's an attachment is copied into, for a draw that
 * reads it as it renders to it, and its view: of the size and formats it
 * was made for. */
struct wvk_attachment_copy {
   WGPUTexture texture;
   WGPUTextureView view;
   WGPUTextureFormat format, view_format;
   uint32_t width, height;
};

struct wvk_replay {
   WGPUCommandEncoder encoder;
   WGPURenderPassEncoder pass;
   WGPUComputePassEncoder compute;

   struct util_dynarray reads;      /* struct wvk_mem_range */
   struct util_dynarray writes;     /* struct wvk_mem_range */
   struct util_dynarray releases;   /* WGPUBuffer, destroyed after */
   struct util_dynarray textures;   /* WGPUTexture, destroyed after */
   struct util_dynarray views;      /* WGPUTextureView, released after */

   /* The attachments of the current pass: for vkCmdClearAttachments, and
    * what a pipeline for it has to match. */
   struct {
      uint32_t color_count;
      struct wvk_image_view *color[8];
      WGPUTextureFormat color_format[8];
      struct wvk_image_view *depth_stencil;
      WGPUTextureFormat depth_format;
      uint32_t samples;
      /* A pass without attachments: whether it is one, and the samples
       * its draws have, which its stand-in attachment has then. */
      bool empty;
      uint32_t empty_samples;
      VkExtent2D extent;
      VkRect2D area;
      uint32_t view_mask;
      /* Depth and stencil the pass does not write: WebGPU lets shaders
       * read those as it tests against them. */
      bool depth_read_only, stencil_read_only;
      /* Not attached at all: no test of them either. */
      bool depth_absent, stencil_absent;
      /* Counts passes, so that what is bound knows when it is gone. */
      uint32_t serial;
      /* The shading rate attachment's texels' size, 0 without one
       * (fsr_view). */
      VkExtent2D fsr_texel;
      /* Resolves WebGPU does not do -- integer formats, modes other than
       * averaging, depth and stencil -- done by vk_meta after the pass,
       * unless it only suspends. */
      struct {
         bool pending;
         bool suspending;
         VkRenderingInfo info;
         VkRenderingAttachmentInfo colors[8];
         VkRenderingAttachmentInfo depth, stencil;
         /* The colors' flags (VK_KHR_maintenance10), which they chain. */
         VkRenderingAttachmentFlagsInfoKHR flags[8];
      } resolve;
   } rendering;

   /* What is bound, as of the command buffer being replayed. */
   struct wvk_bound_state *state;

   /* A render pass of several layers or views is replayed once for each
    * (wvk_cmd_buffer.c): which one this is, and whether they are views. */
   bool pass_layered;
   bool pass_multiview;
   uint32_t pass_layer, pass_first_layer;
   /* A subpass whose views are not this one's: it neither clears nor
    * draws. */
   bool pass_idle;
   /* What the pass began with, for beginning it again at another layer:
    * vk_meta draws layer after layer in a pass of its own (see
    * wvk_meta_draw_volume). Valid while the pass is. */
   const VkRenderingInfo *pass_info;
   /* What pass_info points to: a copy of the pass as begun, which the
    * runtime's render passes have only on their stack. */
   struct {
      VkRenderingInfo info;
      VkRenderingAttachmentInfo colors[8];
      VkRenderingAttachmentInfo depth, stencil;
      /* Its shading rate attachment, the one thing of its chain kept. */
      VkRenderingFragmentShadingRateAttachmentInfoKHR fsr;
   } pass_copy;

   /* Push constants for the submit's draws, 256 bytes apart: written to
    * the device's push buffer just before it. */
   struct util_dynarray push_data;
   /* Push buffers the device grew out of during the submit. */
   struct util_dynarray push_buffers;   /* struct wvk_push_buffer */
   /* Descriptor sets made by vkCmdPushDescriptorSet, destroyed after. */
   struct util_dynarray push_sets;
   /* Memory for vk_meta's vertices, freed after. */
   struct util_dynarray meta_memory;   /* VkDeviceMemory */
   struct wvk_device *dev;
   /* The viewport this draw is drawn in: a draw for each, when the vertex
    * shader chooses (wvk_cmd_draw.c). */
   uint32_t draw_viewport;
   /* The faces this draw is drawn for: 0 both, 1 the front, 2 the back --
    * a draw for each where their stencil masks or references differ,
    * WebGPU having one of each (wvk_cmd_draw.c). */
   uint32_t draw_face;
   /* A draw of triangles drawn as a list the driver made of them: of the
    * triangles of a fan, or the edges or corners of each where the polygon
    * mode is lines or points (wvk_cmd_draw.c); else Undefined. */
   WGPUPrimitiveTopology list_topology;
   /* A draw of a list whose primitives restart, drawn of the list the
    * driver made of its whole ones (wvk_cmd_draw.c, LIST_COMPACT): that,
    * and where in it the indirect draw of them is; else NULL. */
   WGPUBuffer restart_list;
   uint64_t restart_args;
   /* A draw of points as quads, six vertices a point, which the vertex
    * shader moves to its corners (largePoints; wvk_shader.c). */
   bool draw_points;
   /* And of wide lines as quads, six vertices a segment (wideLines). */
   bool draw_lines;
   /* A draw of mesh shaders, whose fragment shader is after the mesh
    * shader (wvk_cmd_draw.c's draw_mesh). */
   bool draw_mesh;
   /* Images with atomics whose buffer of texels is the image now, until
    * the next barrier (wvk_replay_sync_atomic_images). */
   struct util_dynarray atomic_images; /* struct wvk_image * */
   /* Copies of images the pass reads as it renders to them, until the
    * pass ends or a barrier (wvk_cmd_draw.c). */
   struct util_dynarray snapshots;     /* struct wvk_snapshot */
   /* What a draw or dispatch reads of a buffer it also writes, which
    * WebGPU has in no usage scope with the write: copies made before it,
    * which its bind groups name instead. */
   struct util_dynarray buffer_swaps;  /* struct wvk_buffer_swap */
   /* And what a dispatch samples of a texture it writes as storage: views
    * of the levels it does not write, or of a copy (wvk_cmd_draw.c). */
   struct util_dynarray texture_swaps; /* struct wvk_texture_swap */
   /* The copies of colour attachments a draw's logic op reads, made
    * before it (wvk_cmd_draw.c), at WVK_DRIVER_SLOT_LOGIC_OP + i: of the
    * submit's textures for them, copied into again for each draw. */
   WGPUTextureView logic_op_views[8];
   struct wvk_attachment_copy logic_op_copies[8];
   /* And the copy of the depth attachment a depth bounds test reads: of
    * the level and layer the pass renders, into the submit's texture. */
   WGPUTextureView depth_bounds_view;
   struct wvk_attachment_copy depth_bounds_copy;
   /* The pass's shading rate attachment, a 2D view of its first layer
    * (WVK_DRIVER_SLOT_FSR). */
   WGPUTextureView fsr_view;
   VkImageView fsr_source;
   /* The buffers the render pass has used so far, and whether written: a
    * draw that would write what it read, or read what it wrote, begins it
    * again -- a render pass is one usage scope. */
   struct util_dynarray pass_buffers;  /* struct wvk_buffer_use */

   /* The serial this submit finishes at. */
   uint64_t serial;

   /* Occlusion queries: the submit's query set, which passes of command
    * buffers that have queries count in, and where the results go. */
   bool uses_queries;
   WGPUQuerySet query_set;
   WGPUTextureView precise_view;
   struct wvk_pass_query precise_query;
   uint32_t precise_slot;
   bool precise_dirty;
   WGPUBuffer query_resolve;
   uint32_t query_next;
   uint32_t pass_query_first;
   struct util_dynarray pass_queries;   /* struct wvk_pass_query */
   struct wvk_pass_query active_query;
   /* In a meta operation, whose passes the active query is not in. */
   bool query_held;
   /* Queries counted as the replay goes -- pipeline statistics, primitives
    * generated, transform feedback: what the host counts, the counters
    * shaders and the driver's passes add to on the GPU
    * (WVK_DRIVER_SLOT_STATS), each by WVK_COUNTER_* or statistic bit, and
    * the queries active, with the counters they count between them. */
   uint64_t host_counters[WVK_COUNTERS];
   WGPUBuffer counters;
   uint64_t counters_serial;
   /* The buffers of draws that run stages as compute -- mesh, tessellated
    * and geometry shader draws (wvk_cmd_draw.c's scratch) -- a slot each:
    * one of each for the submit's draws, which are one after another on
    * the GPU, made bigger for a draw that needs more. And those a draw has
    * again, to be zeroed where the next pass is ended, as a new buffer
    * starts. */
   struct {
      WGPUBuffer buffer;
      uint64_t size, serial;
   } scratch[WVK_SCRATCH_SLOTS];
   struct util_dynarray scratch_clears;
   uint32_t counter_mask;
   struct wvk_counter_query counter_queries[WVK_COUNTER_QUERIES];
   uint32_t counter_query_count;
   /* Transform feedback: the buffers bound for it, whether it is on (but
    * for vk_meta's draws), and where each buffer is written to next, in
    * bytes from its binding's offset, with whether it filled up -- on the
    * GPU, as primitives geometry shaders make decide (wvk_cmd_draw.c). */
   struct {
      struct wvk_device_memory *mem;
      WGPUBuffer buffer;
      uint64_t offset, size;
   } xfb_bindings[4];
   bool xfb_active;
   WGPUBuffer xfb_state;
   /* Conditional rendering (VK_EXT_conditional_rendering): the predicate's
    * buffer, whether it is on (but for vk_meta's commands, except those of
    * vkCmdClearAttachments) and whether it is inverted. */
   struct {
      struct wvk_device_memory *mem;
      WGPUBuffer buffer;
      uint64_t offset;
      bool active, inverted;
   } cond;
   /* Of a direct command under a predicate the GPU has: each draw the
    * driver makes of it is an indirect draw of arguments the predicate
    * zeroes (wvk_cmd_draw.c's predicated_draw). */
   bool cond_gpu;
   /* Timestamps: a query set of the submit's, and where each is resolved
    * to, 256 bytes apart as resolves want. */
   WGPUQuerySet timestamp_set;
   WGPUBuffer timestamp_resolve;
   uint32_t timestamp_next;
};

struct wvk_push_buffer {
   WGPUBuffer buffer;
   uint64_t size;
};

struct wvk_bound_state;

/* A fresh state for the next command buffer of a submit: nothing is bound
 * across command buffers. */
void wvk_replay_reset_state(struct wvk_replay *replay);
/* The layer and view the pass renders, in the driver's data. */
void wvk_replay_set_pass_layer(struct wvk_replay *replay);
/* The current pass again, at `layer`, with what it has drawn kept. */
void wvk_replay_restart_pass(struct wvk_cmd_buffer *cmd, uint32_t layer);
void wvk_replay_finish_state(struct wvk_device *dev, struct wvk_replay *replay);

/* What is bound, kept around a vk_meta operation. */
void *wvk_replay_save_state(struct wvk_replay *replay);
void wvk_replay_restore_state(struct wvk_replay *replay, void *saved);
/* Another copy of what wvk_replay_save_state saved, to restore twice. */
void *wvk_replay_copy_saved_state(const void *saved);
void wvk_replay_free_meta_memory(struct wvk_device *dev, struct wvk_replay *replay);

/* The resolves of a render pass instance WebGPU could not do itself. */
/* vkCmdResolveImage2 by vk_meta: what WebGPU's resolves cannot do. */
void wvk_cmd_meta_resolve_image(struct wvk_cmd_buffer *cmd, const VkResolveImageInfo2 *info);
void wvk_cmd_meta_clear_color_image(struct wvk_cmd_buffer *cmd, struct wvk_image *image, const VkClearColorValue *color,
                                    uint32_t range_count, const VkImageSubresourceRange *ranges);
void wvk_cmd_meta_resolve_rendering(struct wvk_cmd_buffer *cmd, const VkRenderingInfo *info);
/* A single-sampled view's `layers` from `layer` into every sample of a
 * multisampled one's (VK_EXT_multisampled_render_to_single_sampled). */
void wvk_cmd_meta_unresolve(struct wvk_cmd_buffer *cmd, struct wvk_image_view *from, struct wvk_image_view *to,
                            VkImageAspectFlags aspects, uint32_t layer, uint32_t layers);

/* A view of an R8 texture at least `extent`, for a pass without
 * attachments. The caller releases it. */
WGPUTextureView wvk_dummy_attachment(struct wvk_device *dev, VkExtent2D extent, uint32_t samples);

/* The submit's push constants into the push buffer: right before it. */
void wvk_replay_upload_push_constants(struct wvk_device *dev, struct wvk_replay *replay);

/* An application's command buffer is only its vk_cmd_queue: every vkCmd*
 * goes there (the exposed dispatch table), and submit replays it into an
 * internal command buffer, whose `replay` is what the driver's vkCmd*
 * implementations encode into. */
struct wvk_cmd_buffer {
   struct vk_command_buffer vk;
   struct wvk_replay *replay;

   /* Where the runtime keeps what of the dynamic state is too big to be
    * in it. */
   struct vk_vertex_input_state dynamic_vi;
   struct vk_sample_locations_state dynamic_sample_locations;
};

VK_DEFINE_HANDLE_CASTS(wvk_cmd_buffer, vk.base, VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops wvk_cmd_buffer_ops;

static inline struct wvk_device *
wvk_cmd_buffer_device(struct wvk_cmd_buffer *cmd)
{
   return (struct wvk_device *)cmd->vk.base.device;
}

/* Replay command buffers into WebGPU and submit them: one submit's worth,
 * finishing at `serial`. */
VkResult wvk_cmd_buffers_submit(struct wvk_device *dev, struct wvk_queue *queue,
                                struct vk_command_buffer *const *cmds, uint32_t count,
                                uint64_t serial);

/* A clear value as WebGPU wants it for a format: integers for integers. */
WGPUColor wvk_clear_color(VkFormat format, const VkClearColorValue *value);

/* Whether a pass's load op clears the format right: Metal's clear of
 * RGB10A2Uint keeps each channel to 8 bits, so it is cleared by a draw,
 * or by copies. */
static inline bool
wvk_format_clears_by_load(VkFormat format)
{
   return format != VK_FORMAT_A2B10G10R10_UINT_PACK32;
}

/* End whatever pass is open: copies and clears happen between passes. */
void wvk_replay_end_passes(struct wvk_replay *replay);

struct wvk_buffer_swap {
   uint32_t group, binding;
   WGPUBuffer copy;
   uint64_t size;
};

struct wvk_texture_swap {
   uint32_t group, binding;
   WGPUTextureView view;
};

struct wvk_buffer_use {
   WGPUBuffer buffer;
   bool write;
};

struct wvk_snapshot {
   const struct wvk_image *image;
   /* Which of the image's textures it is of. */
   uint32_t t;
   WGPUTexture texture;
};

/* Forget the copies of what the pass renders to: it may render more. And
 * the textures of the images it renders to are, again, all that is up to
 * date (wvk_image_written), after what was read of them through others. */
void wvk_replay_drop_snapshots(struct wvk_replay *replay);

/* Images with atomics, their buffers of texels copied back to their
 * textures: at a barrier, which is when anything else may see them, and
 * at the end of the submit. A render pass `cmd` is in goes on after. */
void wvk_replay_sync_atomic_images(struct wvk_cmd_buffer *cmd, struct wvk_replay *replay);

/* A submit reads or writes a range of a buffer's memory. Only host-visible
 * memory is tracked; the rest lives on the GPU alone. */
void wvk_replay_read(struct wvk_replay *replay, struct wvk_device_memory *mem,
                     uint64_t offset, uint64_t size);
void wvk_replay_write(struct wvk_replay *replay, struct wvk_device_memory *mem,
                      uint64_t offset, uint64_t size);
/* Conditional rendering's predicate, for a command it applies to: known
 * on the host to pass or to fail, or the GPU's (wvk_cmd_draw.c). */
enum wvk_condition { WVK_CONDITION_PASS, WVK_CONDITION_SKIP, WVK_CONDITION_GPU };
enum wvk_condition wvk_cmd_condition(struct wvk_cmd_buffer *cmd);

/* Whether the submit has written any of a range so far. */
bool wvk_replay_written(const struct wvk_replay *replay, const struct wvk_device_memory *mem,
                        uint64_t offset, uint64_t size);

/* A buffer made for this submit alone: filled from `data` if not NULL,
 * destroyed once the submit ran. */
WGPUBuffer wvk_replay_temp_buffer(struct wvk_device *dev, struct wvk_replay *replay,
                                  uint64_t size, WGPUBufferUsage usage, const void *data);
/* A texture made for this submit alone, the caller's reference to it the
 * submit's now: destroyed once the submit ran. Destroyed, not released:
 * in a browser, the last release only lets JavaScript collect the texture,
 * whenever it collects, and its memory is the GPU's until then. */
void wvk_replay_temp_texture(struct wvk_replay *replay, WGPUTexture texture);

#endif
