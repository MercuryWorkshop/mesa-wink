/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */

/* VK_EXT_host_image_copy: the host's copies to, from and between images,
 * which WebGPU has no way to but its own transfers. Each is the transfer
 * commands of any command buffer -- of a buffer of host-visible memory the
 * host fills or reads, for host memory -- recorded into one of the
 * device's, submitted, and waited for; so every format copies as a buffer
 * copy of it does. The layout a memcpy copy's memory is in is a buffer
 * copy's of the whole subresource, rows and layers packed. */

#include "wvk_cmd_buffer.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_image.h"
#include "wvk_physical_device.h"
#include "wvk_queue.h"

#include "vk_command_pool.h"
#include "vk_format.h"
#include "vk_image.h"
#include "vk_util.h"
#include "util/os_time.h"

/* A buffer of host-visible memory, of the host's copy of its contents. */
struct staging {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *host;
};

static VkResult
staging_create(struct wvk_device *dev, uint64_t size, uint32_t memory_type, struct staging *s)
{
   VkDevice device = wvk_device_to_handle(dev);
   *s = (struct staging){ 0 };
   const VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = MAX2(size, 4),
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkResult result = wvk_CreateBuffer(device, &bci, NULL, &s->buffer);
   if (result != VK_SUCCESS)
      return result;
   result = wvk_device_memory_allocate_internal(dev, align64(MAX2(size, 4), 256), memory_type, &s->memory);
   if (result == VK_SUCCESS) {
      const VkBindBufferMemoryInfo bind = {
         .sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO,
         .buffer = s->buffer,
         .memory = s->memory,
      };
      result = wvk_BindBufferMemory2(device, 1, &bind);
   }
   if (result == VK_SUCCESS) {
      s->host = wvk_device_memory_from_handle(s->memory)->host;
      if (!s->host)
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (result != VK_SUCCESS) {
      wvk_DestroyBuffer(device, s->buffer, NULL);
      if (s->memory)
         wvk_FreeMemory(device, s->memory, NULL);
      *s = (struct staging){ 0 };
   }
   return result;
}

static void
staging_destroy(struct wvk_device *dev, struct staging *s)
{
   VkDevice device = wvk_device_to_handle(dev);
   wvk_DestroyBuffer(device, s->buffer, NULL);
   wvk_FreeMemory(device, s->memory, NULL);
}

/* A command buffer of the device's, begun; one at a time. */
static VkResult
host_cmd_begin(struct wvk_device *dev, VkCommandBuffer *out)
{
   simple_mtx_lock(&dev->host_copy_lock);
   if (!dev->host_copy_cmd) {
      struct vk_command_buffer *cmd;
      VkResult result = wvk_cmd_buffer_ops.create(&dev->host_copy_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &cmd);
      if (result != VK_SUCCESS) {
         simple_mtx_unlock(&dev->host_copy_lock);
         return result;
      }
      dev->host_copy_cmd = container_of(cmd, struct wvk_cmd_buffer, vk);
   }
   *out = wvk_cmd_buffer_to_handle(dev->host_copy_cmd);
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   VkResult result = dev->exposed_dispatch_table.BeginCommandBuffer(*out, &begin);
   if (result != VK_SUCCESS)
      simple_mtx_unlock(&dev->host_copy_lock);
   return result;
}

/* The command buffer ended, submitted as the queue's next submit, and
 * waited for: its readbacks are in the host's copies after. */
static VkResult
host_cmd_submit(struct wvk_device *dev, VkCommandBuffer handle)
{
   struct wvk_queue *queue = &dev->queue;
   VkResult result = dev->exposed_dispatch_table.EndCommandBuffer(handle);
   uint64_t serial = 0;
   if (result == VK_SUCCESS) {
      struct vk_command_buffer *cmd = &dev->host_copy_cmd->vk;
      simple_mtx_lock(&queue->submit_lock);
      result = wvk_queue_reserve(queue, &serial);
      if (result == VK_SUCCESS)
         result = wvk_cmd_buffers_submit(dev, queue, &cmd, 1, serial);
      simple_mtx_unlock(&queue->submit_lock);
   }
   while (result == VK_SUCCESS && wvk_queue_completed(queue) < serial) {
      const VkResult r = wvk_queue_wait_progress(queue, OS_TIMEOUT_INFINITE);
      if (r == VK_ERROR_DEVICE_LOST)
         result = r;
   }
   simple_mtx_unlock(&dev->host_copy_lock);
   return result;
}

/* Where a region of host memory's rows are, as a buffer copy has them:
 * its layout, its extent in blocks, its slices (layers or depth), and the
 * memory it spans. */
struct region {
   struct vk_image_buffer_layout layout;
   VkExtent3D blocks;
   uint32_t slices;
   uint64_t bytes;
};

static struct region
region_layout(const struct vk_image *image, const VkImageSubresourceLayers *subresource, VkExtent3D extent,
              uint32_t row_length, uint32_t image_height)
{
   const VkBufferImageCopy2 bic = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
      .bufferRowLength = row_length,
      .bufferImageHeight = image_height,
      .imageSubresource = *subresource,
      .imageExtent = extent,
   };
   struct region r;
   r.layout = vk_image_buffer_copy_layout(image, &bic);
   const VkFormat format = vk_format_get_aspect_format(image->format, subresource->aspectMask);
   const struct util_format_description *desc = vk_format_description(format);
   const VkExtent3D e = vk_image_sanitize_extent(image, extent);
   r.blocks = (VkExtent3D){
      DIV_ROUND_UP(e.width, desc->block.width),
      DIV_ROUND_UP(e.height, desc->block.height),
      DIV_ROUND_UP(e.depth, desc->block.depth),
   };
   r.slices = MAX2(r.blocks.depth, vk_image_subresource_layer_count(image, subresource));
   r.bytes = vk_image_buffer_range(image, &r.layout, &r.blocks, subresource);
   return r;
}

/* The rows a copy to host memory writes, and not what is between them. */
static void
copy_rows(void *dst, const void *src, const struct region *r)
{
   const uint64_t row = (uint64_t)r->blocks.width * r->layout.element_size_B;
   for (uint32_t z = 0; z < r->slices; ++z) {
      for (uint32_t y = 0; y < r->blocks.height; ++y) {
         const uint64_t at = z * r->layout.image_stride_B + (uint64_t)y * r->layout.row_stride_B;
         memcpy((uint8_t *)dst + at, (const uint8_t *)src + at, row);
      }
   }
}

/* A memcpy copy's region: the whole subresource, packed. */
static VkExtent3D
subresource_extent(const struct vk_image *image, const VkImageSubresourceLayers *subresource)
{
   return wvk_image_aspect_extent(image, subresource->aspectMask, subresource->mipLevel);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CopyMemoryToImage(VkDevice device, const VkCopyMemoryToImageInfo *info)
{
   VK_FROM_HANDLE(wvk_device, dev, device);
   VK_FROM_HANDLE(wvk_image, image, info->dstImage);
   const bool memcpy_layout = info->flags & VK_HOST_IMAGE_COPY_MEMCPY_BIT;
   VkResult result = VK_SUCCESS;
   for (uint32_t r = 0; result == VK_SUCCESS && r < info->regionCount; ++r) {
      const VkMemoryToImageCopy *region = &info->pRegions[r];
      const VkExtent3D extent = memcpy_layout ? subresource_extent(&image->vk, &region->imageSubresource)
                                              : region->imageExtent;
      const uint32_t row_length = memcpy_layout ? 0 : region->memoryRowLength;
      const uint32_t image_height = memcpy_layout ? 0 : region->memoryImageHeight;
      const struct region rl = region_layout(&image->vk, &region->imageSubresource, extent, row_length, image_height);
      struct staging s;
      result = staging_create(dev, rl.bytes, WVK_MEMORY_UPLOAD, &s);
      if (result != VK_SUCCESS)
         break;
      memcpy(s.host, region->pHostPointer, rl.bytes);
      VkCommandBuffer cmd;
      result = host_cmd_begin(dev, &cmd);
      if (result == VK_SUCCESS) {
         const VkBufferImageCopy2 copy = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferRowLength = row_length,
            .bufferImageHeight = image_height,
            .imageSubresource = region->imageSubresource,
            .imageOffset = memcpy_layout ? (VkOffset3D){ 0 } : region->imageOffset,
            .imageExtent = extent,
         };
         const VkCopyBufferToImageInfo2 cbi = {
            .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
            .srcBuffer = s.buffer,
            .dstImage = info->dstImage,
            .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .regionCount = 1,
            .pRegions = &copy,
         };
         dev->exposed_dispatch_table.CmdCopyBufferToImage2(cmd, &cbi);
         result = host_cmd_submit(dev, cmd);
      }
      staging_destroy(dev, &s);
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CopyImageToMemory(VkDevice device, const VkCopyImageToMemoryInfo *info)
{
   VK_FROM_HANDLE(wvk_device, dev, device);
   VK_FROM_HANDLE(wvk_image, image, info->srcImage);
   const bool memcpy_layout = info->flags & VK_HOST_IMAGE_COPY_MEMCPY_BIT;
   VkResult result = VK_SUCCESS;
   for (uint32_t r = 0; result == VK_SUCCESS && r < info->regionCount; ++r) {
      const VkImageToMemoryCopy *region = &info->pRegions[r];
      const VkExtent3D extent = memcpy_layout ? subresource_extent(&image->vk, &region->imageSubresource)
                                              : region->imageExtent;
      const uint32_t row_length = memcpy_layout ? 0 : region->memoryRowLength;
      const uint32_t image_height = memcpy_layout ? 0 : region->memoryImageHeight;
      const struct region rl = region_layout(&image->vk, &region->imageSubresource, extent, row_length, image_height);
      struct staging s;
      result = staging_create(dev, rl.bytes, WVK_MEMORY_READBACK, &s);
      if (result != VK_SUCCESS)
         break;
      VkCommandBuffer cmd;
      result = host_cmd_begin(dev, &cmd);
      if (result == VK_SUCCESS) {
         const VkBufferImageCopy2 copy = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
            .bufferRowLength = row_length,
            .bufferImageHeight = image_height,
            .imageSubresource = region->imageSubresource,
            .imageOffset = memcpy_layout ? (VkOffset3D){ 0 } : region->imageOffset,
            .imageExtent = extent,
         };
         const VkCopyImageToBufferInfo2 cib = {
            .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
            .srcImage = info->srcImage,
            .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .dstBuffer = s.buffer,
            .regionCount = 1,
            .pRegions = &copy,
         };
         dev->exposed_dispatch_table.CmdCopyImageToBuffer2(cmd, &cib);
         result = host_cmd_submit(dev, cmd);
      }
      if (result == VK_SUCCESS)
         copy_rows(region->pHostPointer, s.host, &rl);
      staging_destroy(dev, &s);
   }
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CopyImageToImage(VkDevice device, const VkCopyImageToImageInfo *info)
{
   VK_FROM_HANDLE(wvk_device, dev, device);
   VkCommandBuffer cmd;
   VkResult result = host_cmd_begin(dev, &cmd);
   if (result != VK_SUCCESS)
      return result;
   const VkCopyImageInfo2 ci = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
      .srcImage = info->srcImage,
      .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .dstImage = info->dstImage,
      .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = info->regionCount,
      .pRegions = info->pRegions,
   };
   dev->exposed_dispatch_table.CmdCopyImage2(cmd, &ci);
   return host_cmd_submit(dev, cmd);
}

/* Layouts are nothing to WebGPU. */
VKAPI_ATTR VkResult VKAPI_CALL
wvk_TransitionImageLayout(VkDevice device, uint32_t transitionCount,
                          const VkHostImageLayoutTransitionInfo *pTransitions)
{
   return VK_SUCCESS;
}

VkResult
wvk_device_init_host_copy(struct wvk_device *dev)
{
   simple_mtx_init(&dev->host_copy_lock, mtx_plain);
   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = WVK_QUEUE_FAMILY_ALL,
   };
   return vk_command_pool_init(&dev->vk, &dev->host_copy_pool, &pool_info, NULL);
}

void
wvk_device_finish_host_copy(struct wvk_device *dev)
{
   if (dev->host_copy_cmd)
      wvk_cmd_buffer_ops.destroy(&dev->host_copy_cmd->vk);
   vk_command_pool_finish(&dev->host_copy_pool);
   simple_mtx_destroy(&dev->host_copy_lock);
}
