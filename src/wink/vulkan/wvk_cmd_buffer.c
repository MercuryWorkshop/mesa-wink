/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Recording and replay. An application's command buffer records every
 * vkCmd* into its vk_cmd_queue. A submit replays all of its command
 * buffers, through the driver's own entrypoints, into the device's replay
 * command buffer, whose vkCmd* implementations encode into one
 * WGPUCommandEncoder.
 *
 * Host-visible memory is a host copy the GPU never sees on its own: the
 * ranges a submit reads are uploaded with wgpuQueueWriteBuffer before it,
 * and the ranges it writes are copied into mappable buffers at its end and
 * copied back once it ran, before the submit counts as finished.
 */
#include "wvk_cmd_buffer.h"
#include "wvk_image.h"

#include "vk_framebuffer.h"
#include "vk_render_pass.h"

#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_queue.h"

#include "vk_cmd_queue.h"
#include "vk_command_pool.h"

#include <stdlib.h>

static VkResult
wvk_cmd_buffer_create(struct vk_command_pool *pool, VkCommandBufferLevel level,
                      struct vk_command_buffer **out)
{
   struct wvk_device *dev = container_of(pool->base.device, struct wvk_device, vk);
   struct wvk_cmd_buffer *cmd =
      vk_zalloc(&pool->alloc, sizeof(*cmd), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   VkResult result = vk_command_buffer_init_with_params(
      &cmd->vk, &(struct vk_command_buffer_init_params){
         .pool = pool,
         .ops = &wvk_cmd_buffer_ops,
         .level = level,
         /* Every command is recorded, to be replayed at each submit. */
         .needs_cmd_queue = true,
      });
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }
   cmd->vk.dynamic_graphics_state.vi = &cmd->dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations = &cmd->dynamic_sample_locations;
   *out = &cmd->vk;
   return VK_SUCCESS;
}

static void
wvk_cmd_buffer_reset(struct vk_command_buffer *vk_cmd, VkCommandBufferResetFlags flags)
{
   vk_command_buffer_reset(vk_cmd);
}

static void
wvk_cmd_buffer_destroy(struct vk_command_buffer *vk_cmd)
{
   struct wvk_cmd_buffer *cmd = container_of(vk_cmd, struct wvk_cmd_buffer, vk);
   vk_command_buffer_finish(&cmd->vk);
   vk_free(&cmd->vk.pool->alloc, cmd);
}

const struct vk_command_buffer_ops wvk_cmd_buffer_ops = {
   .create = wvk_cmd_buffer_create,
   .reset = wvk_cmd_buffer_reset,
   .destroy = wvk_cmd_buffer_destroy,
};

VKAPI_ATTR VkResult VKAPI_CALL
wvk_BeginCommandBuffer(VkCommandBuffer commandBuffer, const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   vk_command_buffer_reset(&cmd->vk);
   vk_command_buffer_begin(&cmd->vk, pBeginInfo);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   return vk_command_buffer_end(&cmd->vk);
}

/* ---------------------------------------------------------------------- */
/* The replay state */

void
wvk_replay_end_passes(struct wvk_replay *replay)
{
   if (replay->pass) {
      /* An occlusion query goes on in the next pass, if there is one. */
      wvk_replay_suspend_query(replay);
      wgpuRenderPassEncoderEnd(replay->pass);
      wgpuRenderPassEncoderRelease(replay->pass);
      replay->pass = NULL;
      if (replay->query_set)
         wvk_replay_resolve_queries(replay);
   }
   if (replay->compute) {
      wgpuComputePassEncoderEnd(replay->compute);
      wgpuComputePassEncoderRelease(replay->compute);
      replay->compute = NULL;
   }
}

static void
add_range(struct util_dynarray *ranges, struct wvk_device_memory *mem, uint64_t offset, uint64_t size)
{
   if (!mem || !mem->host || !size)
      return;
   /* WebGPU copies whole words. */
   const uint64_t start = offset & ~3ull;
   const uint64_t end = MIN2(align64(offset + size, 4), mem->buffer_size);
   if (end <= start)
      return;
   util_dynarray_append(ranges, ((struct wvk_mem_range){ mem, start, end - start }));
}

void
wvk_replay_read(struct wvk_replay *replay, struct wvk_device_memory *mem, uint64_t offset, uint64_t size)
{
   add_range(&replay->reads, mem, offset, size);
}

void
wvk_replay_write(struct wvk_replay *replay, struct wvk_device_memory *mem, uint64_t offset, uint64_t size)
{
   add_range(&replay->writes, mem, offset, size);
}

WGPUBuffer
wvk_replay_temp_buffer(struct wvk_device *dev, struct wvk_replay *replay, uint64_t size,
                       WGPUBufferUsage usage, const void *data)
{
   WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   desc.label = wvk_sv("wvk temporary");
   desc.usage = usage;
   desc.size = align64(MAX2(size, 4), 4);
   desc.mappedAtCreation = data != NULL;
   WGPUBuffer buffer = wgpuDeviceCreateBuffer(dev->wgpu, &desc);
   if (!buffer)
      return NULL;
   if (data) {
      void *p = wgpuBufferGetMappedRange(buffer, 0, desc.size);
      if (p) {
         memcpy(p, data, size);
         if (desc.size > size)
            memset((uint8_t *)p + size, 0, desc.size - size);
      }
      wgpuBufferUnmap(buffer);
   }
   util_dynarray_append(&replay->releases, buffer);
   return buffer;
}

static int
compare_ranges(const void *a, const void *b)
{
   const struct wvk_mem_range *x = a, *y = b;
   if (x->mem != y->mem)
      return (uintptr_t)x->mem < (uintptr_t)y->mem ? -1 : 1;
   return x->offset < y->offset ? -1 : x->offset > y->offset ? 1 : 0;
}

/* Sorted by memory and offset, with overlapping and touching ranges made
 * one. Returns the count. */
static uint32_t
coalesce(struct util_dynarray *ranges)
{
   const uint32_t count = util_dynarray_num_elements(ranges, struct wvk_mem_range);
   struct wvk_mem_range *r = util_dynarray_begin(ranges);
   if (count < 2)
      return count;
   qsort(r, count, sizeof(*r), compare_ranges);
   uint32_t out = 0;
   for (uint32_t i = 1; i < count; ++i) {
      if (r[i].mem == r[out].mem && r[i].offset <= r[out].offset + r[out].size) {
         const uint64_t end = MAX2(r[out].offset + r[out].size, r[i].offset + r[i].size);
         r[out].size = end - r[out].offset;
      } else {
         r[++out] = r[i];
      }
   }
   return out + 1;
}

/* ---------------------------------------------------------------------- */
/* Submit */

/* Run [first, end) of a command queue: the executor runs whole queues, so
 * the range is one for the while. */
static void
execute_range(struct list_head *first, struct list_head *end, VkCommandBuffer commandBuffer,
              const struct vk_device_dispatch_table *disp)
{
   if (first == end)
      return;
   struct vk_cmd_queue range;
   memset(&range, 0, sizeof(range));
   struct list_head *before = first->prev, *last = end->prev;
   before->next = end;
   end->prev = before;
   range.cmds.next = first;
   first->prev = &range.cmds;
   range.cmds.prev = last;
   last->next = &range.cmds;
   vk_cmd_queue_execute(&range, commandBuffer, disp);
   before->next = first;
   first->prev = before;
   last->next = end;
   end->prev = last;
}

/* The layers a render pass renders, a bit each, if more than one: its
 * views, or its layers. A pass that suspends or resumes, into another
 * command buffer, is one pass. A render pass object's are those of all
 * its subpasses, or its framebuffer's; `end` is what ends it. */
static uint32_t
pass_layers(const struct vk_cmd_queue_entry *cmd, bool *multiview, enum vk_cmd_type *end)
{
   *multiview = false;
   if (cmd->type == VK_CMD_BEGIN_RENDER_PASS2 || cmd->type == VK_CMD_BEGIN_RENDER_PASS) {
      const VkRenderPassBeginInfo *begin = cmd->type == VK_CMD_BEGIN_RENDER_PASS2
                                              ? cmd->u.begin_render_pass2.render_pass_begin
                                              : cmd->u.begin_render_pass.render_pass_begin;
      VK_FROM_HANDLE(vk_render_pass, pass, begin ? begin->renderPass : VK_NULL_HANDLE);
      VK_FROM_HANDLE(vk_framebuffer, framebuffer, begin ? begin->framebuffer : VK_NULL_HANDLE);
      *end = cmd->type == VK_CMD_BEGIN_RENDER_PASS2 ? VK_CMD_END_RENDER_PASS2 : VK_CMD_END_RENDER_PASS;
      if (!pass || !framebuffer)
         return 0;
      if (pass->is_multiview) {
         *multiview = true;
         return pass->view_mask;
      }
      return framebuffer->layers > 1 ? BITFIELD_MASK(MIN2(framebuffer->layers, 32)) : 0;
   }
   if (cmd->type != VK_CMD_BEGIN_RENDERING)
      return 0;
   *end = VK_CMD_END_RENDERING;
   const VkRenderingInfo *info = cmd->u.begin_rendering.rendering_info;
   if (!info || (info->flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT)))
      return 0;
   if (info->viewMask) {
      *multiview = true;
      return info->viewMask;
   }
   return info->layerCount > 1 ? BITFIELD_MASK(MIN2(info->layerCount, 32)) : 0;
}

/* A command queue, with each render pass of several layers or views run
 * once for each: WebGPU renders one layer a pass. */
static void
replay_queue(struct wvk_replay *replay, struct vk_cmd_queue *queue, VkCommandBuffer commandBuffer,
             const struct vk_device_dispatch_table *disp)
{
   struct list_head *head = &queue->cmds;
   struct list_head *run = head->next;
   for (struct list_head *it = head->next; it != head;) {
      const struct vk_cmd_queue_entry *cmd = list_entry(it, struct vk_cmd_queue_entry, cmd_link);
      bool multiview;
      enum vk_cmd_type end_type = VK_CMD_END_RENDERING;
      const uint32_t layers = pass_layers(cmd, &multiview, &end_type);
      struct list_head *end = it->next;
      while (layers && end != head && list_entry(end, struct vk_cmd_queue_entry, cmd_link)->type != end_type)
         end = end->next;
      if (!layers || end == head) {
         it = it->next;
         continue;
      }
      execute_range(run, it, commandBuffer, disp);
      struct list_head *after = end->next;
      u_foreach_bit(layer, layers) {
         replay->pass_layered = true;
         replay->pass_multiview = multiview;
         replay->pass_layer = layer;
         execute_range(it, after, commandBuffer, disp);
      }
      replay->pass_layered = false;
      replay->pass_multiview = false;
      replay->pass_layer = 0;
      run = it = after;
   }
   execute_range(run, head, commandBuffer, disp);
}

VkResult
wvk_cmd_buffers_submit(struct wvk_device *dev, struct wvk_queue *queue,
                       struct vk_command_buffer *const *cmds, uint32_t count, uint64_t serial)
{
   struct wvk_cmd_buffer *rcmd = dev->replay_cmd;
   struct wvk_replay replay;
   memset(&replay, 0, sizeof(replay));
   util_dynarray_init(&replay.reads, NULL);
   util_dynarray_init(&replay.writes, NULL);
   util_dynarray_init(&replay.releases, NULL);
   util_dynarray_init(&replay.views, NULL);
   util_dynarray_init(&replay.push_data, NULL);
   util_dynarray_init(&replay.push_buffers, NULL);
   util_dynarray_init(&replay.push_sets, NULL);
   util_dynarray_init(&replay.meta_memory, NULL);
   util_dynarray_init(&replay.atomic_images, NULL);
   util_dynarray_init(&replay.snapshots, NULL);
   util_dynarray_init(&replay.buffer_swaps, NULL);
   util_dynarray_init(&replay.texture_swaps, NULL);
   util_dynarray_init(&replay.pass_buffers, NULL);
   util_dynarray_init(&replay.pass_queries, NULL);
   replay.serial = serial;
   replay.dev = dev;

   WGPUCommandEncoderDescriptor edesc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
   replay.encoder = wgpuDeviceCreateCommandEncoder(dev->wgpu, &edesc);

   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   rcmd->replay = &replay;
   for (uint32_t i = 0; i < count; ++i) {
      /* Nothing is bound across command buffers: each starts afresh. */
      vk_command_buffer_reset(&rcmd->vk);
      vk_command_buffer_begin(&rcmd->vk, &begin);
      wvk_replay_reset_state(&replay);
      replay.uses_queries = wvk_cmd_queue_has_occlusion_queries(&cmds[i]->cmd_queue);
      replay_queue(&replay, &cmds[i]->cmd_queue, wvk_cmd_buffer_to_handle(rcmd), &dev->vk.dispatch_table);
   }
   wvk_replay_end_passes(&replay);
   wvk_replay_sync_atomic_images(NULL, &replay);
   util_dynarray_fini(&replay.atomic_images);
   wvk_replay_drop_snapshots(&replay);
   util_dynarray_fini(&replay.snapshots);
   util_dynarray_fini(&replay.buffer_swaps);
   util_dynarray_fini(&replay.texture_swaps);
   util_dynarray_fini(&replay.pass_buffers);
   rcmd->replay = NULL;
   /* What the replay itself recorded is of no further use. */
   vk_command_buffer_reset(&rcmd->vk);

   /* What the submit wrote to host-visible memory goes to mappable
    * buffers, at its very end. */
   const uint32_t write_count = coalesce(&replay.writes);
   struct wvk_readback *readbacks = write_count ? calloc(write_count, sizeof(*readbacks)) : NULL;
   uint32_t readback_count = 0;
   for (uint32_t i = 0; i < write_count; ++i) {
      const struct wvk_mem_range *w = util_dynarray_element(&replay.writes, struct wvk_mem_range, i);
      WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk readback");
      desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
      desc.size = w->size;
      WGPUBuffer staging = wgpuDeviceCreateBuffer(dev->wgpu, &desc);
      if (!staging || !readbacks)
         continue;
      wgpuCommandEncoderCopyBufferToBuffer(replay.encoder, w->mem->buffer, w->offset, staging, 0, w->size);
      readbacks[readback_count++] = (struct wvk_readback){
         .mem = w->mem, .offset = w->offset, .size = w->size, .staging = staging,
      };
   }

   WGPUCommandBufferDescriptor cdesc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
   WGPUCommandBuffer commands = wgpuCommandEncoderFinish(replay.encoder, &cdesc);
   wgpuCommandEncoderRelease(replay.encoder);

   /* What it reads of host-visible memory goes up first -- except where an
    * earlier submit is still to write the range back, since until then the
    * GPU's copy is the one that counts. */
   const uint32_t read_count = coalesce(&replay.reads);
   for (uint32_t i = 0; i < read_count; ++i) {
      const struct wvk_mem_range *r = util_dynarray_element(&replay.reads, struct wvk_mem_range, i);
      if (wvk_queue_range_pending(queue, r->mem, r->offset, r->size))
         continue;
      wgpuQueueWriteBuffer(queue->wgpu, r->mem->buffer, r->offset,
                           (const uint8_t *)r->mem->host + r->offset, r->size);
   }

   wvk_replay_upload_push_constants(dev, &replay);
   wgpuQueueSubmit(queue->wgpu, 1, &commands);
   wgpuCommandBufferRelease(commands);
   wvk_queue_commit(queue, serial, readbacks, readback_count);

   util_dynarray_foreach(&replay.releases, WGPUBuffer, b)
      wgpuBufferRelease(*b);
   util_dynarray_foreach(&replay.views, WGPUTextureView, v)
      wgpuTextureViewRelease(*v);
   util_dynarray_fini(&replay.reads);
   util_dynarray_fini(&replay.writes);
   util_dynarray_fini(&replay.releases);
   util_dynarray_fini(&replay.views);
   wvk_replay_finish_state(dev, &replay);
   wvk_replay_free_meta_memory(dev, &replay);
   wvk_replay_finish_queries(&replay);
   return VK_SUCCESS;
}

/* ---------------------------------------------------------------------- */
/* Synchronization: one in-order queue has nothing to wait for. */

void
wvk_replay_drop_snapshots(struct wvk_replay *replay)
{
   /* What was encoded has them still. */
   util_dynarray_foreach(&replay->snapshots, struct wvk_snapshot, s)
      wgpuTextureRelease(s->texture);
   util_dynarray_clear(&replay->snapshots);
   for (uint32_t i = 0; i < replay->rendering.color_count; ++i) {
      struct wvk_image_view *view = replay->rendering.color[i];
      if (view)
         wvk_image_written(wvk_image_view_image(view), view->texture, view->vk.base_mip_level, 1);
   }
}

void
wvk_replay_sync_atomic_images(struct wvk_cmd_buffer *cmd, struct wvk_replay *replay)
{
   if (!replay->atomic_images.size)
      return;
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   util_dynarray_foreach(&replay->atomic_images, struct wvk_image *, image) {
      wvk_image_copy_atomic(*image, replay->encoder, false);
      (*image)->atomic_in_buffer = false;
   }
   util_dynarray_clear(&replay->atomic_images);
   if (in_pass && cmd)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

/* But for images with atomics: what shaders did to them as buffers goes
 * back to their textures. */
VKAPI_ATTR void VKAPI_CALL
wvk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer, const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   wvk_replay_sync_atomic_images(cmd, cmd->replay);
   wvk_replay_drop_snapshots(cmd->replay);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount, const VkEvent *pEvents,
                   const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   wvk_replay_sync_atomic_images(cmd, cmd->replay);
   wvk_replay_drop_snapshots(cmd->replay);
}
