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
      /* Counts passes, so that what is bound knows when it is gone. */
      uint32_t serial;
   } rendering;

   /* What is bound, as of the command buffer being replayed. */
   struct wvk_bound_state *state;

   /* Push constants for the submit's draws, 256 bytes apart: written to
    * the device's push buffer just before it. */
   struct util_dynarray push_data;
   /* Push buffers the device grew out of during the submit. */
   struct util_dynarray push_buffers;   /* struct wvk_push_buffer */
   /* Descriptor sets made by vkCmdPushDescriptorSet, destroyed after. */
   struct util_dynarray push_sets;
};

struct wvk_push_buffer {
   WGPUBuffer buffer;
   uint64_t size;
};

struct wvk_bound_state;

/* A fresh state for the next command buffer of a submit: nothing is bound
 * across command buffers. */
void wvk_replay_reset_state(struct wvk_replay *replay);
void wvk_replay_finish_state(struct wvk_device *dev, struct wvk_replay *replay);

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
