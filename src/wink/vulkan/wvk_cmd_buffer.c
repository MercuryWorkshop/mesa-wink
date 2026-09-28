/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_cmd_buffer.h"

#include "wvk_device.h"
#include "wvk_entrypoints.h"
#include "wvk_queue.h"

#include "vk_command_pool.h"

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

VkResult
wvk_cmd_buffers_submit(struct wvk_device *dev, struct wvk_queue *queue,
                       struct vk_command_buffer *const *cmds, uint32_t count)
{
   /* M0: recording works, replay comes with memory and encoding. */
   if (count)
      return vk_errorf(dev, VK_ERROR_FEATURE_NOT_PRESENT, "command replay is not implemented yet");
   return VK_SUCCESS;
}
