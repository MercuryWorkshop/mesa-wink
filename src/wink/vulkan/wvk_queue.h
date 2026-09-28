/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_QUEUE_H
#define WVK_QUEUE_H

#include "wvk_private.h"

#include "vk_queue.h"

#include "util/simple_mtx.h"
#include "util/u_atomic.h"

struct wvk_device;
struct wvk_device_memory;

#define WVK_MAX_PENDING_SUBMITS 64

/* Host-visible memory a submit wrote: copied to a mappable buffer at the
 * end of the submit, mapped once it ran, and copied into the host copy. */
struct wvk_readback {
   struct wvk_device_memory *mem;
   uint64_t offset;
   uint64_t size;
   WGPUBuffer staging;
   WGPUFuture future;
   bool mapped;
   bool failed;
};

struct wvk_pending_submit {
   uint64_t serial;
   WGPUFuture work_done;
   bool done;
   struct wvk_readback *readbacks;
   uint32_t readback_count;
};

/* The one queue. Every submit gets the next serial, and `completed` is
 * raised in order: WebGPU runs its queue in order, so a finished serial
 * means everything before it finished too. A submit is finished once its
 * work is done and what it wrote to host-visible memory is back. */
struct wvk_queue {
   struct vk_queue vk;
   WGPUQueue wgpu;

   /* Submits are the application's to serialize, but waits are not: any
    * thread may wait on a fence or a semaphore, and waiting retires. */
   simple_mtx_t lock;

   uint64_t submitted;
   uint64_t completed;

   struct wvk_pending_submit pending[WVK_MAX_PENDING_SUBMITS];
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

/* Whether a submit after `completed` may still write this range of
 * host-visible memory back: then the host copy is not the truth there,
 * and must not be uploaded over what the GPU has. */
bool wvk_queue_range_pending(struct wvk_queue *queue, struct wvk_device_memory *mem,
                             uint64_t offset, uint64_t size);

/* Reserve the next serial's slot, waiting if WebGPU is far behind. */
VkResult wvk_queue_reserve(struct wvk_queue *queue, uint64_t *serial);

/* The reserved submit went to WebGPU: its readbacks, which it now owns. */
void wvk_queue_commit(struct wvk_queue *queue, uint64_t serial,
                      struct wvk_readback *readbacks, uint32_t readback_count);

#endif
