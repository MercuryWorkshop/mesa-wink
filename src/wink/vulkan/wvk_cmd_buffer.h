/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_CMD_BUFFER_H
#define WVK_CMD_BUFFER_H

#include "wvk_private.h"

#include "vk_command_buffer.h"

struct wvk_device;
struct wvk_queue;

/* An application's command buffer is only its vk_cmd_queue: every vkCmd*
 * goes there (the exposed dispatch table), and submit replays it. The
 * same struct is the replay target, whose state is what the driver's
 * vkCmd* implementations encode with. */
struct wvk_cmd_buffer {
   struct vk_command_buffer vk;
};

VK_DEFINE_HANDLE_CASTS(wvk_cmd_buffer, vk.base, VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER)

extern const struct vk_command_buffer_ops wvk_cmd_buffer_ops;

/* Replay command buffers into WebGPU and submit them: one submit's worth. */
VkResult wvk_cmd_buffers_submit(struct wvk_device *dev, struct wvk_queue *queue,
                                struct vk_command_buffer *const *cmds, uint32_t count);

#endif
