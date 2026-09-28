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
struct wvk_replay {
   WGPUCommandEncoder encoder;
   WGPURenderPassEncoder pass;
   WGPUComputePassEncoder compute;

   struct util_dynarray reads;      /* struct wvk_mem_range */
   struct util_dynarray writes;     /* struct wvk_mem_range */
   struct util_dynarray releases;   /* WGPUBuffer, destroyed after */
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
      VkExtent2D extent;
      VkRect2D area;
      uint32_t view_mask;
      /* Depth and stencil the pass does not write: WebGPU lets shaders
       * read those as it tests against them. */
      bool depth_read_only, stencil_read_only;
      /* Counts passes, so that what is bound knows when it is gone. */
      uint32_t serial;
      /* Resolves WebGPU does not do -- integer formats, modes other than
       * averaging, depth and stencil -- done by vk_meta after the pass,
       * unless it only suspends. */
      struct {
         bool pending;
         bool suspending;
         VkRenderingInfo info;
         VkRenderingAttachmentInfo colors[8];
         VkRenderingAttachmentInfo depth, stencil;
      } resolve;
   } rendering;

   /* What is bound, as of the command buffer being replayed. */
   struct wvk_bound_state *state;

   /* A render pass of several layers or views is replayed once for each
    * (wvk_cmd_buffer.c): which one this is, and whether they are views. */
   bool pass_layered;
   bool pass_multiview;
   uint32_t pass_layer;
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
   /* Images with atomics whose buffer of texels is the image now, until
    * the next barrier (wvk_replay_sync_atomic_images). */
   struct util_dynarray atomic_images; /* struct wvk_image * */
   /* Copies of images the pass reads as it renders to them, until the
    * pass ends or a barrier (wvk_cmd_draw.c). */
   struct util_dynarray snapshots;     /* struct wvk_snapshot */

   /* The serial this submit finishes at. */
   uint64_t serial;

   /* Occlusion queries: the submit's query set, which passes of command
    * buffers that have queries count in, and where the results go. */
   bool uses_queries;
   WGPUQuerySet query_set;
   WGPUBuffer query_resolve;
   uint32_t query_next;
   uint32_t pass_query_first;
   struct util_dynarray pass_queries;   /* struct wvk_pass_query */
   struct wvk_pass_query active_query;
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
void wvk_replay_free_meta_memory(struct wvk_device *dev, struct wvk_replay *replay);

/* The resolves of a render pass instance WebGPU could not do itself. */
/* vkCmdResolveImage2 by vk_meta: what WebGPU's resolves cannot do. */
void wvk_cmd_meta_resolve_image(struct wvk_cmd_buffer *cmd, const VkResolveImageInfo2 *info);
void wvk_cmd_meta_resolve_rendering(struct wvk_cmd_buffer *cmd, const VkRenderingInfo *info);

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

/* End whatever pass is open: copies and clears happen between passes. */
void wvk_replay_end_passes(struct wvk_replay *replay);

struct wvk_snapshot {
   const struct wvk_image *image;
   WGPUTexture texture;
};

/* Forget the copies of what the pass renders to: it may render more. */
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

/* A buffer made for this submit alone: filled from `data` if not NULL,
 * destroyed once the submit ran. */
WGPUBuffer wvk_replay_temp_buffer(struct wvk_device *dev, struct wvk_replay *replay,
                                  uint64_t size, WGPUBufferUsage usage, const void *data);

#endif
