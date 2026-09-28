/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_queue.h"

#include "wvk_cmd_buffer.h"
#include "wvk_device.h"
#include "wvk_instance.h"
#include "wvk_physical_device.h"
#include "wvk_platform.h"
#include "wvk_sync.h"

#include "util/os_time.h"

static struct wvk_instance *
queue_instance(struct wvk_queue *queue)
{
   return (struct wvk_instance *)wvk_queue_device(queue)->vk.physical->instance;
}

static void
on_work_done(WGPUQueueWorkDoneStatus status, void *userdata1, void *userdata2)
{
   struct wvk_queue *queue = userdata1;
   const uint64_t serial = (uint64_t)(uintptr_t)userdata2;
   if (status != WGPUQueueWorkDoneStatus_Success)
      vk_device_set_lost(&wvk_queue_device(queue)->vk, "queue work failed");
   /* In order, so this is a high-water mark. */
   if (serial > p_atomic_read(&queue->completed))
      p_atomic_set(&queue->completed, serial);
}

static void
retire_completed(struct wvk_queue *queue)
{
   const uint64_t completed = wvk_queue_completed(queue);
   while (queue->pending_count &&
          queue->pending[queue->pending_first].serial <= completed) {
      queue->pending_first = (queue->pending_first + 1) % WVK_MAX_PENDING_SUBMITS;
      --queue->pending_count;
   }
}

VkResult
wvk_queue_wait_progress(struct wvk_queue *queue, uint64_t abs_timeout_ns)
{
   retire_completed(queue);
   if (!queue->pending_count)
      return VK_NOT_READY;

   const uint64_t now = os_time_get_nano();
   const uint64_t timeout = abs_timeout_ns == OS_TIMEOUT_INFINITE ? UINT64_MAX
                            : abs_timeout_ns > now ? abs_timeout_ns - now : 0;
   const uint64_t before = wvk_queue_completed(queue);
   wvk_platform_wait(queue_instance(queue)->wgpu,
                     queue->pending[queue->pending_first].future, timeout);
   retire_completed(queue);
   if (vk_device_is_lost(&wvk_queue_device(queue)->vk))
      return VK_ERROR_DEVICE_LOST;
   return wvk_queue_completed(queue) > before ? VK_SUCCESS : VK_TIMEOUT;
}

/* Reserve a pending slot: the ring only fills when WebGPU is far behind,
 * so wait for the oldest submit then. */
static VkResult
reserve_pending(struct wvk_queue *queue)
{
   retire_completed(queue);
   while (queue->pending_count == WVK_MAX_PENDING_SUBMITS) {
      VkResult result = wvk_queue_wait_progress(queue, OS_TIMEOUT_INFINITE);
      if (result == VK_ERROR_DEVICE_LOST)
         return result;
   }
   return VK_SUCCESS;
}

static VkResult
wvk_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct wvk_queue *queue = container_of(vk_queue, struct wvk_queue, vk);
   struct wvk_device *dev = wvk_queue_device(queue);

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   /* Waits need nothing: submits are deferred until what they wait for is
    * pending, and pending means earlier on this in-order queue. */

   VkResult result = wvk_cmd_buffers_submit(dev, queue, submit->command_buffers,
                                            submit->command_buffer_count);
   if (result != VK_SUCCESS)
      return result;

   result = reserve_pending(queue);
   if (result != VK_SUCCESS)
      return result;

   const uint64_t serial = ++queue->submitted;
   WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
   cb.mode = WGPUCallbackMode_WaitAnyOnly;
   cb.callback = on_work_done;
   cb.userdata1 = queue;
   cb.userdata2 = (void *)(uintptr_t)serial;
   const WGPUFuture future = wgpuQueueOnSubmittedWorkDone(queue->wgpu, cb);
   const uint32_t slot = (queue->pending_first + queue->pending_count) % WVK_MAX_PENDING_SUBMITS;
   queue->pending[slot].serial = serial;
   queue->pending[slot].future = future;
   ++queue->pending_count;

   for (uint32_t i = 0; i < submit->signal_count; ++i) {
      struct vk_sync *sync = submit->signals[i].sync;
      if (wvk_sync_is_wvk(sync)) {
         wvk_sync_set_pending(sync, serial);
      } else {
         result = vk_sync_signal(&dev->vk, sync, submit->signals[i].signal_value);
         if (result != VK_SUCCESS)
            return result;
      }
   }
   return VK_SUCCESS;
}

VkResult
wvk_queue_init(struct wvk_device *dev, struct wvk_queue *queue,
               const VkDeviceQueueCreateInfo *create_info, uint32_t index)
{
   VkResult result = vk_queue_init(&queue->vk, &dev->vk, create_info, index);
   if (result != VK_SUCCESS)
      return result;

   queue->wgpu = wgpuDeviceGetQueue(dev->wgpu);
   queue->vk.driver_submit = wvk_queue_submit;
   return VK_SUCCESS;
}

void
wvk_queue_finish(struct wvk_device *dev, struct wvk_queue *queue)
{
   /* Let what was submitted finish before the device goes. */
   while (queue->pending_count &&
          wvk_queue_wait_progress(queue, OS_TIMEOUT_INFINITE) == VK_SUCCESS)
      ;
   if (queue->wgpu)
      wgpuQueueRelease(queue->wgpu);
   vk_queue_finish(&queue->vk);
}
