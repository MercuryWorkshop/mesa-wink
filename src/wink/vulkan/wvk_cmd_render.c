/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Dynamic rendering, which is also what the runtime turns render passes
 * into. A Vulkan render pass instance is a WebGPU render pass: load and
 * store ops map straight across, and a resolve is the attachment's
 * resolve target.
 */
#include "wvk_cmd_buffer.h"

#include "wvk_device.h"
#include "wvk_entrypoints.h"
#include "wvk_format.h"
#include "wvk_physical_device.h"
#include "wvk_image.h"

#include "vk_format.h"

#include <float.h>
#include <math.h>

/* WebGPU refuses a clear value that is NaN or infinite, and one refused
 * pass is a lost command buffer: NaN clears to 0, and infinity to the
 * largest float, which a 16-bit format takes as its infinity. Components a
 * format does not have are often left as garbage, too. */
static double
clear_finite(float f)
{
   if (isnan(f))
      return 0.0;
   if (isinf(f))
      return f > 0 ? FLT_MAX : -FLT_MAX;
   return f;
}

WGPUColor
wvk_clear_color(VkFormat format, const VkClearColorValue *value)
{
   /* WebGPU converts the clear value to the attachment's type, so integer
    * formats want the integers, not their bits as floats. */
   if (vk_format_is_uint(format))
      return (WGPUColor){ value->uint32[0], value->uint32[1], value->uint32[2], value->uint32[3] };
   if (vk_format_is_sint(format))
      return (WGPUColor){ value->int32[0], value->int32[1], value->int32[2], value->int32[3] };
   /* Alpha is one in a format without it, and a packed format's RGBA8
    * texture keeps it so. */
   const double alpha = util_format_has_alpha(vk_format_to_pipe_format(format)) ? clear_finite(value->float32[3]) : 1.0;
   return (WGPUColor){ clear_finite(value->float32[0]), clear_finite(value->float32[1]), clear_finite(value->float32[2]),
                       alpha };
}

static WGPULoadOp
load_op(VkAttachmentLoadOp op, bool resuming)
{
   if (resuming)
      return WGPULoadOp_Load;
   switch (op) {
   case VK_ATTACHMENT_LOAD_OP_CLEAR:
      return WGPULoadOp_Clear;
   case VK_ATTACHMENT_LOAD_OP_DONT_CARE:
      /* Whatever it holds may be dropped, and clearing is the cheaper way. */
      return WGPULoadOp_Clear;
   default:
      return WGPULoadOp_Load;
   }
}

/* WebGPU loads, clears or drops the whole of an attachment, Vulkan only
 * its render area. Where the area is less, the attachment is loaded, and
 * a clear is a draw over the area once the pass has begun. */
static bool
area_covers(const VkRect2D *area, const struct wvk_image_view *view)
{
   return area->offset.x <= 0 && area->offset.y <= 0 &&
          (int64_t)area->offset.x + area->extent.width >= view->vk.extent.width &&
          (int64_t)area->offset.y + area->extent.height >= view->vk.extent.height;
}

static WGPUStoreOp
store_op(VkAttachmentStoreOp op, bool suspending)
{
   if (suspending)
      return WGPUStoreOp_Store;
   return op == VK_ATTACHMENT_STORE_OP_DONT_CARE ? WGPUStoreOp_Discard : WGPUStoreOp_Store;
}

static WGPUTextureView
attachment_view(struct wvk_replay *replay, struct wvk_image_view *view, WGPUTextureAspect aspect)
{
   struct wvk_image *image = wvk_image_view_image(view);
   /* A pass of several layers is one for each: this one's. */
   WGPUTextureView v = wvk_image_attachment_view(image, view->format, aspect, view->vk.base_mip_level,
                                                 view->vk.base_array_layer + replay->pass_layer);
   util_dynarray_append(&replay->views, v);
   return v;
}

/* The pass's size and sample count, which its pipelines and viewports go
 * by: those of its attachments, which all agree. */
static void
note_attachment(struct wvk_replay *replay, struct wvk_image_view *view)
{
   const struct wvk_image *image = wvk_image_view_image(view);
   replay->rendering.samples = image->vk.samples;
   replay->rendering.extent = (VkExtent2D){
      u_minify(image->vk.extent.width, view->vk.base_mip_level),
      u_minify(image->vk.extent.height, view->vk.base_mip_level),
   };
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBeginRendering(VkCommandBuffer commandBuffer, const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_replay *replay = cmd->replay;
   const VkRenderingInfo *info = pRenderingInfo;
   replay->pass_idle = replay->pass_layered && replay->pass_multiview && info->viewMask &&
                       !(info->viewMask & BITFIELD_BIT(replay->pass_layer));
   /* An idle subpass keeps the layer as it was. */
   const bool resuming = (info->flags & VK_RENDERING_RESUMING_BIT) || replay->pass_idle;
   const bool suspending = info->flags & VK_RENDERING_SUSPENDING_BIT;

   wvk_replay_end_passes(replay);
   wvk_replay_set_pass_layer(replay);
   if (info != &replay->pass_copy.info) {
      /* Kept, to begin the pass again with (wvk_replay_restart_pass). */
      replay->pass_copy.info = *info;
      replay->pass_copy.info.pNext = NULL;
      const uint32_t n = MIN2(info->colorAttachmentCount, ARRAY_SIZE(replay->pass_copy.colors));
      for (uint32_t i = 0; i < n; ++i) {
         replay->pass_copy.colors[i] = info->pColorAttachments[i];
         replay->pass_copy.colors[i].pNext = NULL;
      }
      replay->pass_copy.info.colorAttachmentCount = n;
      replay->pass_copy.info.pColorAttachments = n ? replay->pass_copy.colors : NULL;
      if (info->pDepthAttachment) {
         replay->pass_copy.depth = *info->pDepthAttachment;
         replay->pass_copy.depth.pNext = NULL;
         replay->pass_copy.info.pDepthAttachment = &replay->pass_copy.depth;
      }
      if (info->pStencilAttachment) {
         replay->pass_copy.stencil = *info->pStencilAttachment;
         replay->pass_copy.stencil.pNext = NULL;
         replay->pass_copy.info.pStencilAttachment = &replay->pass_copy.stencil;
      }
   }
   replay->pass_info = &replay->pass_copy.info;

   const struct wvk_physical_device *pdev = wvk_device_physical(wvk_cmd_buffer_device(cmd));
   WGPURenderPassColorAttachment colors[8];
   /* Clears of attachments the render area does not cover. */
   VkClearAttachment clears[9];
   uint32_t clear_count = 0;
   const uint32_t color_count = MIN2(info->colorAttachmentCount, ARRAY_SIZE(colors));
   replay->rendering.color_count = color_count;
   /* Nothing to resolve by shader, until an attachment says otherwise. */
   memset(&replay->rendering.resolve, 0, sizeof(replay->rendering.resolve));
   replay->rendering.resolve.suspending = suspending;
   replay->rendering.resolve.info = *info;
   replay->rendering.resolve.info.pNext = NULL;
   replay->rendering.resolve.info.colorAttachmentCount = color_count;
   replay->rendering.area = info->renderArea;
   replay->rendering.view_mask = info->viewMask;
   replay->rendering.depth_format = WGPUTextureFormat_Undefined;
   replay->rendering.samples = 1;
   replay->rendering.extent = (VkExtent2D){ 0, 0 };
   ++replay->rendering.serial;
   for (uint32_t i = 0; i < color_count; ++i) {
      const VkRenderingAttachmentInfo *att = &info->pColorAttachments[i];
      VK_FROM_HANDLE(wvk_image_view, view, att->imageView);
      colors[i] = (WGPURenderPassColorAttachment)WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
      replay->rendering.color[i] = view;
      replay->rendering.color_format[i] = WGPUTextureFormat_Undefined;
      if (!view)
         continue;
      note_attachment(replay, view);
      replay->rendering.color_format[i] = view->format;
      colors[i].view = attachment_view(replay, view, WGPUTextureAspect_All);
      if (wvk_image_view_image(view)->dimension == WGPUTextureDimension_3D)
         colors[i].depthSlice = view->vk.base_array_layer + replay->pass_layer;
      colors[i].loadOp = load_op(att->loadOp, resuming);
      if (!resuming && !area_covers(&info->renderArea, view)) {
         colors[i].loadOp = WGPULoadOp_Load;
         if (att->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR)
            clears[clear_count++] = (VkClearAttachment){ VK_IMAGE_ASPECT_COLOR_BIT, i, att->clearValue };
      }
      colors[i].storeOp = store_op(att->storeOp, suspending);
      colors[i].clearValue = wvk_clear_color(view->vk.format, &att->clearValue.color);
      if (att->resolveMode != VK_RESOLVE_MODE_NONE && att->resolveImageView) {
         VK_FROM_HANDLE(wvk_image_view, resolve, att->resolveImageView);
         /* WebGPU averages, and only formats that can be. */
         if (att->resolveMode == VK_RESOLVE_MODE_AVERAGE_BIT &&
             (wvk_format_caps(pdev, resolve->vk.format) & WVK_CAP_RESOLVE)) {
            colors[i].resolveTarget = attachment_view(replay, resolve, WGPUTextureAspect_All);
         } else {
            replay->rendering.resolve.colors[i] = *att;
            replay->rendering.resolve.colors[i].pNext = NULL;
            replay->rendering.resolve.pending = true;
            /* What the shader resolves from has to outlive the pass. */
            colors[i].storeOp = WGPUStoreOp_Store;
         }
      }
   }

   /* WebGPU has one depth-stencil attachment: the depth and stencil
    * attachments must be views of one image. */
   WGPURenderPassDepthStencilAttachment ds = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
   const VkRenderingAttachmentInfo *depth = info->pDepthAttachment;
   const VkRenderingAttachmentInfo *stencil = info->pStencilAttachment;
   if (depth && !depth->imageView)
      depth = NULL;
   if (stencil && !stencil->imageView)
      stencil = NULL;
   replay->rendering.depth_stencil = NULL;
   if (depth || stencil) {
      VK_FROM_HANDLE(wvk_image_view, view, (depth ? depth : stencil)->imageView);
      replay->rendering.depth_stencil = view;
      note_attachment(replay, view);
      replay->rendering.depth_format = wvk_image_view_image(view)->format;
      ds.view = attachment_view(replay, view, WGPUTextureAspect_All);
      const VkFormat format = wvk_image_view_image(view)->vk.format;
      const bool whole = resuming || area_covers(&info->renderArea, view);
      VkClearAttachment ds_clear = { 0 };
      if (depth && depth->resolveMode != VK_RESOLVE_MODE_NONE && depth->resolveImageView) {
         replay->rendering.resolve.depth = *depth;
         replay->rendering.resolve.depth.pNext = NULL;
         replay->rendering.resolve.pending = true;
      }
      if (stencil && stencil->resolveMode != VK_RESOLVE_MODE_NONE && stencil->resolveImageView) {
         replay->rendering.resolve.stencil = *stencil;
         replay->rendering.resolve.stencil.pNext = NULL;
         replay->rendering.resolve.pending = true;
      }
      if (vk_format_has_depth(format)) {
         if (depth) {
            ds.depthLoadOp = whole ? load_op(depth->loadOp, resuming) : WGPULoadOp_Load;
            if (!whole && depth->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
               ds_clear.aspectMask |= VK_IMAGE_ASPECT_DEPTH_BIT;
               ds_clear.clearValue.depthStencil.depth = depth->clearValue.depthStencil.depth;
            }
            ds.depthStoreOp = replay->rendering.resolve.depth.resolveMode ? WGPUStoreOp_Store
                              : store_op(depth->storeOp, suspending);
            ds.depthClearValue = CLAMP(clear_finite(depth->clearValue.depthStencil.depth), 0.0, 1.0);
         } else {
            /* Not rendered to in this pass: kept as it is. */
            ds.depthReadOnly = true;
         }
      }
      if (vk_format_has_stencil(format)) {
         if (stencil) {
            ds.stencilLoadOp = whole ? load_op(stencil->loadOp, resuming) : WGPULoadOp_Load;
            if (!whole && stencil->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
               ds_clear.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
               ds_clear.clearValue.depthStencil.stencil = stencil->clearValue.depthStencil.stencil;
            }
            ds.stencilStoreOp = replay->rendering.resolve.stencil.resolveMode ? WGPUStoreOp_Store
                                : store_op(stencil->storeOp, suspending);
            ds.stencilClearValue = stencil->clearValue.depthStencil.stencil;
         } else {
            ds.stencilReadOnly = true;
         }
      }
      if (ds_clear.aspectMask)
         clears[clear_count++] = ds_clear;
   }

   /* WebGPU wants an attachment; Vulkan renders without any, for what the
    * shaders store or the queries count. An R8 one of the render area's
    * size, whose contents go. */
   uint32_t pass_color_count = color_count;
   bool any = depth || stencil;
   for (uint32_t i = 0; i < color_count; ++i)
      any |= colors[i].view != NULL;
   if (!any) {
      const VkExtent2D extent = {
         (uint32_t)(info->renderArea.offset.x + info->renderArea.extent.width),
         (uint32_t)(info->renderArea.offset.y + info->renderArea.extent.height),
      };
      const uint32_t samples = MAX2(replay->rendering.samples, 1);
      WGPUTextureView dummy = wvk_dummy_attachment(wvk_cmd_buffer_device(cmd), extent, samples);
      if (dummy) {
         util_dynarray_append(&replay->views, dummy);
         colors[0] = (WGPURenderPassColorAttachment)WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
         colors[0].view = dummy;
         colors[0].loadOp = WGPULoadOp_Clear;
         colors[0].storeOp = WGPUStoreOp_Discard;
         pass_color_count = MAX2(pass_color_count, 1);
         replay->rendering.color_count = pass_color_count;
         replay->rendering.color[0] = NULL;
         replay->rendering.color_format[0] = WGPUTextureFormat_R8Unorm;
         replay->rendering.extent = extent;
         replay->rendering.samples = samples;
      }
   }

   WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   desc.colorAttachmentCount = pass_color_count;
   desc.colorAttachments = colors;
   desc.depthStencilAttachment = (depth || stencil) ? &ds : NULL;
   desc.occlusionQuerySet = wvk_replay_query_set(wvk_cmd_buffer_device(cmd), replay);
   replay->pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &desc);
   if (clear_count) {
      const VkClearRect rect = { info->renderArea, 0, MAX2(info->layerCount, 1) };
      wvk_CmdClearAttachments(commandBuffer, clear_count, clears, 1, &rect);
   }
}

void
wvk_replay_restart_pass(struct wvk_cmd_buffer *cmd, uint32_t layer)
{
   struct wvk_replay *replay = cmd->replay;
   const VkRenderingInfo *begun = replay->pass_info;
   if (!begun)
      return;
   VkRenderingInfo info = *begun;
   info.flags |= VK_RENDERING_RESUMING_BIT;
   /* Resolves are the pass's end's, which has not come. */
   const bool pending = replay->rendering.resolve.pending;
   replay->pass_layer = layer;
   wvk_CmdBeginRendering(wvk_cmd_buffer_to_handle(cmd), &info);
   replay->rendering.resolve.pending |= pending;
   /* The pass is still the one begun, not this copy of it. */
   replay->pass_info = begun;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_replay *replay = cmd->replay;
   wvk_replay_end_passes(replay);
   replay->rendering.color_count = 0;
   replay->rendering.depth_stencil = NULL;
   replay->pass_info = NULL;
   /* A pass vk_meta drew layer by layer ends at the first again. */
   if (!replay->pass_layered)
      replay->pass_layer = 0;

   if (!replay->rendering.resolve.pending || replay->rendering.resolve.suspending)
      return;
   /* A copy: vk_meta's own passes begin rendering too. */
   VkRenderingAttachmentInfo colors[8], depth, stencil;
   memcpy(colors, replay->rendering.resolve.colors, sizeof(colors));
   depth = replay->rendering.resolve.depth;
   stencil = replay->rendering.resolve.stencil;
   VkRenderingInfo info = replay->rendering.resolve.info;
   info.pColorAttachments = colors;
   info.pDepthAttachment = depth.resolveMode ? &depth : NULL;
   info.pStencilAttachment = stencil.resolveMode ? &stencil : NULL;
   replay->rendering.resolve.pending = false;
   wvk_cmd_meta_resolve_rendering(cmd, &info);
}
