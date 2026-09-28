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
#include "wvk_image.h"

#include "vk_format.h"

#include <float.h>
#include <math.h>

/* WebGPU refuses a clear value that is NaN or infinite, and one refused
 * pass is a lost command buffer: NaN clears to 0, and infinity to the
 * largest float, which a 16-bit format takes as its infinity. Components a
 * format does not have are often left as garbage, too. */
static double
finite(float f)
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
   return (WGPUColor){ finite(value->float32[0]), finite(value->float32[1]), finite(value->float32[2]),
                       finite(value->float32[3]) };
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
   WGPUTextureView v = wvk_image_attachment_view(image, view->format, aspect,
                                                 view->vk.base_mip_level, view->vk.base_array_layer);
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
   const bool resuming = info->flags & VK_RENDERING_RESUMING_BIT;
   const bool suspending = info->flags & VK_RENDERING_SUSPENDING_BIT;

   wvk_replay_end_passes(replay);

   WGPURenderPassColorAttachment colors[8];
   const uint32_t color_count = MIN2(info->colorAttachmentCount, ARRAY_SIZE(colors));
   replay->rendering.color_count = color_count;
   replay->rendering.area = info->renderArea;
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
         colors[i].depthSlice = view->vk.base_array_layer;
      colors[i].loadOp = load_op(att->loadOp, resuming);
      colors[i].storeOp = store_op(att->storeOp, suspending);
      colors[i].clearValue = wvk_clear_color(view->vk.format, &att->clearValue.color);
      if (att->resolveMode != VK_RESOLVE_MODE_NONE && att->resolveImageView) {
         VK_FROM_HANDLE(wvk_image_view, resolve, att->resolveImageView);
         colors[i].resolveTarget = attachment_view(replay, resolve, WGPUTextureAspect_All);
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
      if (vk_format_has_depth(format)) {
         if (depth) {
            ds.depthLoadOp = load_op(depth->loadOp, resuming);
            ds.depthStoreOp = store_op(depth->storeOp, suspending);
            ds.depthClearValue = CLAMP(finite(depth->clearValue.depthStencil.depth), 0.0, 1.0);
         } else {
            /* Not rendered to in this pass: kept as it is. */
            ds.depthReadOnly = true;
         }
      }
      if (vk_format_has_stencil(format)) {
         if (stencil) {
            ds.stencilLoadOp = load_op(stencil->loadOp, resuming);
            ds.stencilStoreOp = store_op(stencil->storeOp, suspending);
            ds.stencilClearValue = stencil->clearValue.depthStencil.stencil;
         } else {
            ds.stencilReadOnly = true;
         }
      }
   }

   WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   desc.colorAttachmentCount = color_count;
   desc.colorAttachments = colors;
   desc.depthStencilAttachment = (depth || stencil) ? &ds : NULL;
   desc.occlusionQuerySet = wvk_replay_query_set(wvk_cmd_buffer_device(cmd), replay);
   replay->pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &desc);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdEndRendering(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   wvk_replay_end_passes(cmd->replay);
   cmd->replay->rendering.color_count = 0;
   cmd->replay->rendering.depth_stencil = NULL;
}
