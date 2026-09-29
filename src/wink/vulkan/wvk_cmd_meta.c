/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * What the runtime's vk_meta does for us with draws: vkCmdClearAttachments
 * and vkCmdBlitImage. Its pipelines are NIR, compiled like any other, and
 * its commands go through the driver's own entrypoints, so they are part of
 * the replay like the application's. Around each, what the application had
 * bound is kept and put back.
 */
#include "wvk_cmd_buffer.h"

#include "wvk_buffer.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_image.h"
#include "wvk_physical_device.h"

#include "vk_format.h"
#include "vk_meta.h"

/* A buffer for vk_meta's rectangles: host-visible memory of its own, freed
 * after the submit, once its contents went up. */
static VkResult
wvk_cmd_bind_map_buffer(struct vk_command_buffer *vk_cmd, struct vk_meta_device *meta,
                        VkBuffer _buffer, void **map_out)
{
   struct wvk_cmd_buffer *cmd = container_of(vk_cmd, struct wvk_cmd_buffer, vk);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   VK_FROM_HANDLE(wvk_buffer, buffer, _buffer);
   VkDevice device = wvk_device_to_handle(dev);

   VkDeviceMemory memory;
   VkResult result = wvk_AllocateMemory(device, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = align64(buffer->vk.size, 16),
      .memoryTypeIndex = WVK_MEMORY_UPLOAD,
   }, NULL, &memory);
   if (result != VK_SUCCESS)
      return result;
   util_dynarray_append(&cmd->replay->meta_memory, memory);
   result = wvk_BindBufferMemory2(device, 1, &(VkBindBufferMemoryInfo){
      .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO, .buffer = _buffer, .memory = memory,
   });
   if (result != VK_SUCCESS)
      return result;
   *map_out = wvk_device_memory_from_handle(memory)->host;
   return VK_SUCCESS;
}

/* vk_meta draws a volume, or layers, in one pass of several layers, the
 * vertex shader choosing each rectangle's. That pass, begun during a
 * replay rather than from the command queue, is not replayed per layer:
 * so each layer is drawn in the pass begun again at that layer. */
static void
wvk_meta_draw_volume(struct vk_command_buffer *vk_cmd, struct vk_meta_device *meta,
                     const struct vk_meta_rect *rect, uint32_t layer_count)
{
   struct wvk_cmd_buffer *cmd = container_of(vk_cmd, struct wvk_cmd_buffer, vk);
   struct wvk_replay *replay = cmd->replay;
   /* A pass replayed per layer has the vertex shader's culling choose. */
   if (replay->pass_layered || (layer_count <= 1 && rect->layer == replay->pass_layer)) {
      vk_meta_draw_volume(vk_cmd, meta, rect, layer_count);
      return;
   }
   for (uint32_t i = 0; i < layer_count; ++i) {
      struct vk_meta_rect one = *rect;
      one.layer = rect->layer + i;
      if (one.layer != replay->pass_layer)
         wvk_replay_restart_pass(cmd, one.layer);
      vk_meta_draw_volume(vk_cmd, meta, &one, 1);
   }
}

VkResult
wvk_device_init_meta(struct wvk_device *dev)
{
   VkResult result = vk_meta_device_init(&dev->vk, &dev->meta);
   if (result != VK_SUCCESS)
      return result;
   /* A pass of several layers is replayed once for each (wvk_cmd_buffer.c),
    * or vk_meta's own drawn layer by layer, and vertex shaders choose
    * theirs. */
   dev->meta.use_layered_rendering = true;
   dev->meta.cmd_draw_volume = wvk_meta_draw_volume;
   dev->meta.use_gs_for_layer = false;
   dev->meta.use_stencil_export = false;
   /* vk_meta's own vertex shader for its rectangles, which it draws as two
    * triangles each: the rect-list topology maps to a triangle list. */
   dev->meta.use_rect_list_pipeline = true;
   dev->meta.cmd_bind_map_buffer = wvk_cmd_bind_map_buffer;
   dev->meta.max_bind_map_buffer_size_B = 64 * 1024;
   return VK_SUCCESS;
}

void
wvk_device_finish_meta(struct wvk_device *dev)
{
   vk_meta_device_finish(&dev->vk, &dev->meta);
}

void
wvk_replay_free_meta_memory(struct wvk_device *dev, struct wvk_replay *replay)
{
   util_dynarray_foreach(&replay->meta_memory, VkDeviceMemory, memory)
      wvk_FreeMemory(wvk_device_to_handle(dev), *memory, NULL);
   util_dynarray_fini(&replay->meta_memory);
}

/* What vk_meta changes and the application does not expect changed. */
struct wvk_meta_save {
   struct vk_dynamic_graphics_state dynamic;
   struct vk_vertex_input_state vi;
   struct vk_sample_locations_state sample_locations;
   void *state;
   bool xfb_active;
};

static void
meta_begin(struct wvk_cmd_buffer *cmd, struct wvk_meta_save *save)
{
   save->dynamic = cmd->vk.dynamic_graphics_state;
   save->vi = cmd->dynamic_vi;
   save->sample_locations = cmd->dynamic_sample_locations;
   save->state = wvk_replay_save_state(cmd->replay);
   /* vk_meta's draws are no application's to capture. */
   save->xfb_active = cmd->replay->xfb_active;
   cmd->replay->xfb_active = false;
}

static void
meta_end(struct wvk_cmd_buffer *cmd, struct wvk_meta_save *save)
{
   cmd->vk.dynamic_graphics_state = save->dynamic;
   cmd->dynamic_vi = save->vi;
   cmd->dynamic_sample_locations = save->sample_locations;
   cmd->vk.dynamic_graphics_state.vi = &cmd->dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations = &cmd->dynamic_sample_locations;
   /* Everything is set again at the next draw. */
   BITSET_SET_RANGE(cmd->vk.dynamic_graphics_state.dirty, 0, MESA_VK_DYNAMIC_GRAPHICS_STATE_ENUM_MAX - 1);
   wvk_replay_restore_state(cmd->replay, save->state);
   cmd->replay->xfb_active = save->xfb_active;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdClearAttachments(VkCommandBuffer commandBuffer, uint32_t attachmentCount,
                        const VkClearAttachment *pAttachments, uint32_t rectCount,
                        const VkClearRect *pRects)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   const struct wvk_replay *replay = cmd->replay;
   if (!replay->pass)
      return;

   /* The current pass, as vk_meta wants to know it. */
   struct vk_meta_rendering_info render = {
      .samples = replay->rendering.samples,
      .color_attachment_count = replay->rendering.color_count,
      .view_mask = replay->rendering.view_mask,
   };
   for (uint32_t i = 0; i < replay->rendering.color_count; ++i) {
      const struct wvk_image_view *view = replay->rendering.color[i];
      render.color_attachment_formats[i] = view ? view->vk.format : VK_FORMAT_UNDEFINED;
      render.color_attachment_write_masks[i] = 0xf;
   }
   if (replay->rendering.depth_stencil) {
      const VkFormat format = wvk_image_view_image(replay->rendering.depth_stencil)->vk.format;
      if (vk_format_has_depth(format))
         render.depth_attachment_format = format;
      if (vk_format_has_stencil(format))
         render.stencil_attachment_format = format;
   }

   struct wvk_meta_save save;
   /* Clears are not drawing, to an occlusion query. */
   wvk_replay_suspend_query(cmd->replay);
   meta_begin(cmd, &save);
   vk_meta_clear_attachments(&cmd->vk, &dev->meta, &render, attachmentCount, pAttachments,
                             rectCount, pRects);
   meta_end(cmd, &save);
   wvk_replay_resume_query(cmd->replay);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBlitImage2(VkCommandBuffer commandBuffer, const VkBlitImageInfo2 *pBlitImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_meta_save save;
   meta_begin(cmd, &save);
   vk_meta_blit_image2(&cmd->vk, &dev->meta, pBlitImageInfo);
   meta_end(cmd, &save);
}

void
wvk_cmd_meta_resolve_image(struct wvk_cmd_buffer *cmd, const VkResolveImageInfo2 *info)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_meta_save save;
   meta_begin(cmd, &save);
   vk_meta_resolve_image2(&cmd->vk, &dev->meta, info);
   meta_end(cmd, &save);
}

void
wvk_cmd_meta_clear_color_image(struct wvk_cmd_buffer *cmd, struct wvk_image *image, const VkClearColorValue *color,
                               uint32_t range_count, const VkImageSubresourceRange *ranges)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_meta_save save;
   meta_begin(cmd, &save);
   vk_meta_clear_color_image(&cmd->vk, &dev->meta, &image->vk, VK_IMAGE_LAYOUT_GENERAL, image->vk.format, color,
                             range_count, ranges);
   meta_end(cmd, &save);
}

void
wvk_cmd_meta_resolve_rendering(struct wvk_cmd_buffer *cmd, const VkRenderingInfo *info)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_meta_save save;
   /* A layer at a time, as a view mask has vk_meta do it: a multisampled
    * array is a texture for each layer, which is all a shader can read. */
   VkRenderingInfo layered = *info;
   if (!layered.viewMask && layered.layerCount > 1 && layered.layerCount <= 32)
      layered.viewMask = BITFIELD_MASK(layered.layerCount);
   meta_begin(cmd, &save);
   vk_meta_resolve_rendering(&cmd->vk, &dev->meta, &layered);
   meta_end(cmd, &save);
}

/* An attachment for a pass that has none: kept by size on the device. */
WGPUTextureView
wvk_dummy_attachment(struct wvk_device *dev, VkExtent2D extent, uint32_t samples)
{
   simple_mtx_lock(&dev->cache_lock);
   if (!dev->dummy_attachment || dev->dummy_extent.width < extent.width ||
       dev->dummy_extent.height < extent.height || dev->dummy_samples != samples) {
      if (dev->dummy_attachment)
         wgpuTextureRelease(dev->dummy_attachment);
      WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk attachment for a pass without any");
      desc.usage = WGPUTextureUsage_RenderAttachment;
      desc.size = (WGPUExtent3D){ MAX2(extent.width, dev->dummy_extent.width),
                                  MAX2(extent.height, dev->dummy_extent.height), 1 };
      desc.format = WGPUTextureFormat_R8Unorm;
      desc.sampleCount = samples;
      dev->dummy_attachment = wgpuDeviceCreateTexture(dev->wgpu, &desc);
      dev->dummy_extent = (VkExtent2D){ desc.size.width, desc.size.height };
      dev->dummy_samples = samples;
   }
   WGPUTextureView view = dev->dummy_attachment ? wgpuTextureCreateView(dev->dummy_attachment, NULL) : NULL;
   simple_mtx_unlock(&dev->cache_lock);
   return view;
}
