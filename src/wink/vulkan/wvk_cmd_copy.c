/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Copies, fills and clears, between passes.
 *
 * WebGPU's buffer-texture copies want rows of a multiple of 256 bytes;
 * Vulkan takes any. An aligned copy is one WebGPU copy, and any other
 * goes a row at a time, for which WebGPU needs no row pitch at all.
 */
#include "wvk_cmd_buffer.h"

#include "wvk_buffer.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_image.h"

#include "vk_format.h"
#include "util/format/u_format.h"

static WGPUBuffer
buffer_wgpu(const struct wvk_buffer *buffer)
{
   return buffer->mem ? buffer->mem->buffer : NULL;
}

static uint64_t
buffer_offset(const struct wvk_buffer *buffer, uint64_t offset)
{
   return buffer->offset + offset;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *pCopyBufferInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_buffer, src, pCopyBufferInfo->srcBuffer);
   VK_FROM_HANDLE(wvk_buffer, dst, pCopyBufferInfo->dstBuffer);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pCopyBufferInfo->regionCount; ++i) {
      const VkBufferCopy2 *r = &pCopyBufferInfo->pRegions[i];
      const uint64_t so = buffer_offset(src, r->srcOffset), dof = buffer_offset(dst, r->dstOffset);
      wvk_replay_read(replay, src->mem, so, r->size);
      wvk_replay_write(replay, dst->mem, dof, r->size);
      if ((so | dof | r->size) & 3) {
         /* WebGPU copies whole words. */
         mesa_logw("wvk: vkCmdCopyBuffer of %" PRIu64 " bytes at %" PRIu64 " -> %" PRIu64
                   " is not word-aligned; copying the aligned part", r->size, so, dof);
         const uint64_t size = r->size & ~3ull;
         if (!size || ((so | dof) & 3))
            continue;
         wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, buffer_wgpu(src), so, buffer_wgpu(dst), dof, size);
         continue;
      }
      wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, buffer_wgpu(src), so, buffer_wgpu(dst), dof, r->size);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                    VkDeviceSize dataSize, const void *pData)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_buffer, dst, dstBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, dataSize, WGPUBufferUsage_CopySrc, pData);
   const uint64_t off = buffer_offset(dst, dstOffset);
   wvk_replay_write(replay, dst->mem, off, dataSize);
   wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, temp, 0, buffer_wgpu(dst), off, dataSize);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                  VkDeviceSize fillSize, uint32_t data)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_buffer, dst, dstBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;

   uint64_t size = vk_buffer_range(&dst->vk, dstOffset, fillSize) & ~3ull;
   if (!size)
      return;
   wvk_replay_end_passes(replay);
   const uint64_t off = buffer_offset(dst, dstOffset);
   wvk_replay_write(replay, dst->mem, off, size);
   if (data == 0) {
      wgpuCommandEncoderClearBuffer(replay->encoder, buffer_wgpu(dst), off, size);
      return;
   }
   /* A pattern needs a source to copy from. */
   uint32_t *pattern = malloc(size);
   if (!pattern)
      return;
   for (uint64_t i = 0; i < size / 4; ++i)
      pattern[i] = data;
   WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, size, WGPUBufferUsage_CopySrc, pattern);
   free(pattern);
   wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, temp, 0, buffer_wgpu(dst), off, size);
}

/* ---------------------------------------------------------------------- */
/* Buffers and images */

struct copy_layout {
   uint32_t block_w, block_h, block_bytes;
   uint32_t row_bytes;        /* one row of blocks, in the buffer */
   uint32_t rows_per_image;   /* in blocks */
};

static struct copy_layout
buffer_layout(VkFormat format, VkImageAspectFlags aspect, uint32_t row_length, uint32_t image_height,
              VkExtent3D extent)
{
   enum pipe_format pformat = vk_format_to_pipe_format(format);
   /* One aspect of a depth-stencil format is copied on its own. */
   if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT)
      pformat = PIPE_FORMAT_S8_UINT;
   else if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT)
      pformat = vk_format_to_pipe_format(vk_format_depth_only(format));
   struct copy_layout l = {
      .block_w = util_format_get_blockwidth(pformat),
      .block_h = util_format_get_blockheight(pformat),
      .block_bytes = util_format_get_blocksize(pformat),
   };
   const uint32_t width = row_length ? row_length : extent.width;
   const uint32_t height = image_height ? image_height : extent.height;
   l.row_bytes = DIV_ROUND_UP(width, l.block_w) * l.block_bytes;
   l.rows_per_image = DIV_ROUND_UP(height, l.block_h);
   return l;
}

static void
image_region(const struct wvk_image *image, const VkImageSubresourceLayers *sub, VkOffset3D offset,
             VkExtent3D extent, WGPUTexelCopyTextureInfo *tex, WGPUExtent3D *size)
{
   *tex = (WGPUTexelCopyTextureInfo)WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
   tex->texture = image->texture;
   tex->mipLevel = sub->mipLevel;
   tex->aspect = wvk_aspect(sub->aspectMask);
   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   tex->origin = (WGPUOrigin3D){
      (uint32_t)offset.x,
      image->vk.image_type == VK_IMAGE_TYPE_1D ? 0 : (uint32_t)offset.y,
      is_3d ? (uint32_t)offset.z : sub->baseArrayLayer,
   };
   const uint32_t layers = sub->layerCount == VK_REMAINING_ARRAY_LAYERS
                              ? image->vk.array_layers - sub->baseArrayLayer : sub->layerCount;
   *size = (WGPUExtent3D){
      extent.width,
      image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : extent.height,
      is_3d ? extent.depth : layers,
   };
}

/* One buffer-texture copy, split into rows if WebGPU wants a pitch the
 * buffer does not have. */
static void
copy_buffer_texture(struct wvk_replay *replay, bool to_texture, WGPUBuffer buffer, uint64_t offset,
                    const struct copy_layout *l, const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size)
{
   const uint32_t block_rows = DIV_ROUND_UP(size.height, l->block_h);
   WGPUTexelCopyBufferInfo buf = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
   buf.buffer = buffer;

   if (l->row_bytes % 256 == 0 || (block_rows <= 1 && size.depthOrArrayLayers <= 1)) {
      buf.layout.offset = offset;
      /* A pitch WebGPU is given must be aligned; one row needs none. */
      buf.layout.bytesPerRow = l->row_bytes % 256 == 0 ? l->row_bytes : WGPU_COPY_STRIDE_UNDEFINED;
      buf.layout.rowsPerImage = l->row_bytes % 256 == 0 ? l->rows_per_image : WGPU_COPY_STRIDE_UNDEFINED;
      if (to_texture)
         wgpuCommandEncoderCopyBufferToTexture(replay->encoder, &buf, tex, &size);
      else
         wgpuCommandEncoderCopyTextureToBuffer(replay->encoder, tex, &buf, &size);
      return;
   }

   for (uint32_t z = 0; z < size.depthOrArrayLayers; ++z) {
      for (uint32_t row = 0; row < block_rows; ++row) {
         WGPUTexelCopyTextureInfo t = *tex;
         t.origin.y += row * l->block_h;
         t.origin.z += z;
         const uint32_t h = MIN2(l->block_h, size.height - row * l->block_h);
         const WGPUExtent3D one = { size.width, h, 1 };
         buf.layout.offset = offset + (uint64_t)z * l->rows_per_image * l->row_bytes +
                             (uint64_t)row * l->row_bytes;
         buf.layout.bytesPerRow = WGPU_COPY_STRIDE_UNDEFINED;
         buf.layout.rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED;
         if (to_texture)
            wgpuCommandEncoderCopyBufferToTexture(replay->encoder, &buf, &t, &one);
         else
            wgpuCommandEncoderCopyTextureToBuffer(replay->encoder, &t, &buf, &one);
      }
   }
}

static uint64_t
region_bytes(const struct copy_layout *l, WGPUExtent3D size)
{
   const uint32_t block_rows = DIV_ROUND_UP(size.height, l->block_h);
   const uint64_t last_row = DIV_ROUND_UP(size.width, l->block_w) * l->block_bytes;
   return (uint64_t)(size.depthOrArrayLayers - 1) * l->rows_per_image * l->row_bytes +
          (uint64_t)(block_rows - 1) * l->row_bytes + last_row;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_buffer, src, pCopyBufferToImageInfo->srcBuffer);
   VK_FROM_HANDLE(wvk_image, dst, pCopyBufferToImageInfo->dstImage);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pCopyBufferToImageInfo->regionCount; ++i) {
      const VkBufferImageCopy2 *r = &pCopyBufferToImageInfo->pRegions[i];
      WGPUTexelCopyTextureInfo tex;
      WGPUExtent3D size;
      image_region(dst, &r->imageSubresource, r->imageOffset, r->imageExtent, &tex, &size);
      const struct copy_layout l = buffer_layout(dst->vk.format, r->imageSubresource.aspectMask,
                                                 r->bufferRowLength, r->bufferImageHeight, r->imageExtent);
      const uint64_t off = buffer_offset(src, r->bufferOffset);
      wvk_replay_read(replay, src->mem, off, region_bytes(&l, size));
      copy_buffer_texture(replay, true, buffer_wgpu(src), off, &l, &tex, size);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2 *pCopyImageToBufferInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pCopyImageToBufferInfo->srcImage);
   VK_FROM_HANDLE(wvk_buffer, dst, pCopyImageToBufferInfo->dstBuffer);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pCopyImageToBufferInfo->regionCount; ++i) {
      const VkBufferImageCopy2 *r = &pCopyImageToBufferInfo->pRegions[i];
      WGPUTexelCopyTextureInfo tex;
      WGPUExtent3D size;
      image_region(src, &r->imageSubresource, r->imageOffset, r->imageExtent, &tex, &size);
      const struct copy_layout l = buffer_layout(src->vk.format, r->imageSubresource.aspectMask,
                                                 r->bufferRowLength, r->bufferImageHeight, r->imageExtent);
      const uint64_t off = buffer_offset(dst, r->bufferOffset);
      wvk_replay_write(replay, dst->mem, off, region_bytes(&l, size));
      copy_buffer_texture(replay, false, buffer_wgpu(dst), off, &l, &tex, size);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyImage2(VkCommandBuffer commandBuffer, const VkCopyImageInfo2 *pCopyImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pCopyImageInfo->srcImage);
   VK_FROM_HANDLE(wvk_image, dst, pCopyImageInfo->dstImage);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   /* WebGPU copies between textures of one format, give or take sRGB;
    * Vulkan between any of one size, which goes through a buffer. */
   const bool same = src->format == dst->format ||
                     util_format_linear(vk_format_to_pipe_format(src->vk.format)) ==
                        util_format_linear(vk_format_to_pipe_format(dst->vk.format));
   for (uint32_t i = 0; i < pCopyImageInfo->regionCount; ++i) {
      const VkImageCopy2 *r = &pCopyImageInfo->pRegions[i];
      WGPUTexelCopyTextureInfo s, d;
      WGPUExtent3D size, dsize;
      image_region(src, &r->srcSubresource, r->srcOffset, r->extent, &s, &size);
      /* A compressed image's extent is in its own texels. */
      const enum pipe_format sf = vk_format_to_pipe_format(src->vk.format);
      const enum pipe_format df = vk_format_to_pipe_format(dst->vk.format);
      VkExtent3D dext = r->extent;
      dext.width = DIV_ROUND_UP(dext.width, util_format_get_blockwidth(sf)) * util_format_get_blockwidth(df);
      dext.height = DIV_ROUND_UP(dext.height, util_format_get_blockheight(sf)) * util_format_get_blockheight(df);
      image_region(dst, &r->dstSubresource, r->dstOffset, dext, &d, &dsize);
      if (same) {
         wgpuCommandEncoderCopyTextureToTexture(replay->encoder, &s, &d, &size);
         continue;
      }
      struct copy_layout l = buffer_layout(src->vk.format, r->srcSubresource.aspectMask, 0, 0, r->extent);
      l.row_bytes = align(l.row_bytes, 256);
      const uint64_t bytes = (uint64_t)size.depthOrArrayLayers * l.rows_per_image * l.row_bytes;
      WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, bytes,
                                               WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, NULL);
      copy_buffer_texture(replay, false, temp, 0, &l, &s, size);
      struct copy_layout dl = l;
      dl.block_w = util_format_get_blockwidth(df);
      dl.block_h = util_format_get_blockheight(df);
      copy_buffer_texture(replay, true, temp, 0, &dl, &d, dsize);
   }
}

/* ---------------------------------------------------------------------- */
/* Clears and resolves: render passes that only load or clear */

static void
clear_pass(struct wvk_replay *replay, struct wvk_image *image, uint32_t level, uint32_t layer,
           const VkClearColorValue *color, const VkClearDepthStencilValue *ds, VkImageAspectFlags aspects)
{
   WGPUTextureView view = wvk_image_attachment_view(image, image->format, WGPUTextureAspect_All, level, layer);
   util_dynarray_append(&replay->views, view);

   WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   WGPURenderPassColorAttachment c = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
   WGPURenderPassDepthStencilAttachment d = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
   if (color) {
      c.view = view;
      if (image->dimension == WGPUTextureDimension_3D)
         c.depthSlice = layer;
      c.loadOp = WGPULoadOp_Clear;
      c.storeOp = WGPUStoreOp_Store;
      c.clearValue = wvk_clear_color(image->vk.format, color);
      desc.colorAttachmentCount = 1;
      desc.colorAttachments = &c;
   } else {
      d.view = view;
      if (vk_format_has_depth(image->vk.format)) {
         d.depthLoadOp = (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? WGPULoadOp_Clear : WGPULoadOp_Load;
         d.depthStoreOp = WGPUStoreOp_Store;
         d.depthClearValue = ds->depth;
      }
      if (vk_format_has_stencil(image->vk.format)) {
         d.stencilLoadOp = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? WGPULoadOp_Clear : WGPULoadOp_Load;
         d.stencilStoreOp = WGPUStoreOp_Store;
         d.stencilClearValue = ds->stencil;
      }
      desc.depthStencilAttachment = &d;
   }
   WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &desc);
   wgpuRenderPassEncoderEnd(pass);
   wgpuRenderPassEncoderRelease(pass);
}

static void
clear_ranges(struct wvk_cmd_buffer *cmd, struct wvk_image *image, const VkClearColorValue *color,
             const VkClearDepthStencilValue *ds, uint32_t range_count, const VkImageSubresourceRange *ranges)
{
   struct wvk_replay *replay = cmd->replay;
   wvk_replay_end_passes(replay);
   if (!(image->usage & WGPUTextureUsage_RenderAttachment)) {
      mesa_logw("wvk: clearing a non-renderable format (%u) is not implemented", image->vk.format);
      return;
   }
   for (uint32_t r = 0; r < range_count; ++r) {
      const VkImageSubresourceRange *range = &ranges[r];
      const uint32_t levels = vk_image_subresource_level_count(&image->vk, range);
      for (uint32_t m = 0; m < levels; ++m) {
         const uint32_t level = range->baseMipLevel + m;
         const uint32_t layers = image->vk.image_type == VK_IMAGE_TYPE_3D
                                    ? u_minify(image->vk.extent.depth, level)
                                    : vk_image_subresource_layer_count(&image->vk, range);
         const uint32_t base = image->vk.image_type == VK_IMAGE_TYPE_3D ? 0 : range->baseArrayLayer;
         for (uint32_t l = 0; l < layers; ++l)
            clear_pass(replay, image, level, base + l, color, ds, range->aspectMask);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdClearColorImage(VkCommandBuffer commandBuffer, VkImage _image, VkImageLayout imageLayout,
                       const VkClearColorValue *pColor, uint32_t rangeCount,
                       const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, image, _image);
   clear_ranges(cmd, image, pColor, NULL, rangeCount, pRanges);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage _image, VkImageLayout imageLayout,
                              const VkClearDepthStencilValue *pDepthStencil, uint32_t rangeCount,
                              const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, image, _image);
   clear_ranges(cmd, image, NULL, pDepthStencil, rangeCount, pRanges);
}

/* A resolve is a render pass that loads the multisampled image and
 * resolves into the other -- of whole subresources, which is all WebGPU
 * resolves. */
VKAPI_ATTR void VKAPI_CALL
wvk_CmdResolveImage2(VkCommandBuffer commandBuffer, const VkResolveImageInfo2 *pResolveImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pResolveImageInfo->srcImage);
   VK_FROM_HANDLE(wvk_image, dst, pResolveImageInfo->dstImage);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pResolveImageInfo->regionCount; ++i) {
      const VkImageResolve2 *r = &pResolveImageInfo->pRegions[i];
      const uint32_t layers = r->srcSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS
                                 ? src->vk.array_layers - r->srcSubresource.baseArrayLayer
                                 : r->srcSubresource.layerCount;
      const VkExtent3D full = vk_image_mip_level_extent(&dst->vk, r->dstSubresource.mipLevel);
      if (r->srcOffset.x || r->srcOffset.y || r->dstOffset.x || r->dstOffset.y ||
          r->extent.width != full.width || r->extent.height != full.height)
         mesa_logw("wvk: resolving part of an image resolves all of it");
      for (uint32_t l = 0; l < layers; ++l) {
         WGPUTextureView s = wvk_image_attachment_view(src, src->format, WGPUTextureAspect_All,
                                                       r->srcSubresource.mipLevel,
                                                       r->srcSubresource.baseArrayLayer + l);
         WGPUTextureView d = wvk_image_attachment_view(dst, dst->format, WGPUTextureAspect_All,
                                                       r->dstSubresource.mipLevel,
                                                       r->dstSubresource.baseArrayLayer + l);
         util_dynarray_append(&replay->views, s);
         util_dynarray_append(&replay->views, d);
         WGPURenderPassColorAttachment c = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
         c.view = s;
         c.resolveTarget = d;
         c.loadOp = WGPULoadOp_Load;
         c.storeOp = WGPUStoreOp_Store;
         WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
         desc.colorAttachmentCount = 1;
         desc.colorAttachments = &c;
         WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &desc);
         wgpuRenderPassEncoderEnd(pass);
         wgpuRenderPassEncoderRelease(pass);
      }
   }
}
