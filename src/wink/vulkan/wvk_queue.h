/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_QUEUE_H
#define WVK_QUEUE_H

#include "wvk_private.h"

#include "vk_queue.h"

#include "util/u_atomic.h"

struct wvk_device;

#define WVK_MAX_PENDING_SUBMITS 64

/* The one queue. Every submit gets the next serial, and work-done
 * callbacks raise `completed` in order: WebGPU runs its queue in order, so
 * a finished serial means everything before it finished too. */
struct wvk_queue {
   struct vk_queue vk;
   WGPUQueue wgpu;

   uint64_t submitted;
   uint64_t completed;

   /* Work-done futures still outstanding, oldest first. */
   struct {
      uint64_t serial;
      WGPUFuture future;
   } pending[WVK_MAX_PENDING_SUBMITS];
   uint32_t pending_first;
   uint32_t pending_count;
};

static inline struct wvk_device *
wvk_queue_device(struct wvk_queue *queue)
{
   return (struct wvk_device *)queue->vk.base.device;
}

static inline uint64_t
wvk_queue_completed(struct wvk_queue *queue)
{
   return p_atomic_read(&queue->completed);
}

VkResult wvk_queue_init(struct wvk_device *dev, struct wvk_queue *queue,
                        const VkDeviceQueueCreateInfo *create_info, uint32_t index);
void wvk_queue_finish(struct wvk_device *dev, struct wvk_queue *queue);

/* Wait until another submit finishes: VK_SUCCESS once one did, VK_TIMEOUT
 * at abs_timeout_ns, VK_NOT_READY if nothing is outstanding. */
VkResult wvk_queue_wait_progress(struct wvk_queue *queue, uint64_t abs_timeout_ns);

#endif
