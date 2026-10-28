/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_sync.h"

#include "wvk_device.h"
#include "wvk_platform.h"
#include "wvk_queue.h"

#include "util/os_time.h"

static VkResult
wvk_sync_init(struct vk_device *device, struct vk_sync *vk_sync, uint64_t initial_value)
{
   struct wvk_sync *sync = container_of(vk_sync, struct wvk_sync, base);
   sync->signaled = initial_value != 0;
   sync->serial = 0;
   return VK_SUCCESS;
}

static void
wvk_sync_finish(struct vk_device *device, struct vk_sync *vk_sync)
{
}

static VkResult
wvk_sync_signal(struct vk_device *device, struct vk_sync *vk_sync, uint64_t value)
{
   struct wvk_sync *sync = container_of(vk_sync, struct wvk_sync, base);
   sync->signaled = true;
   sync->serial = 0;
   return VK_SUCCESS;
}

static VkResult
wvk_sync_reset(struct vk_device *device, struct vk_sync *vk_sync)
{
   struct wvk_sync *sync = container_of(vk_sync, struct wvk_sync, base);
   sync->signaled = false;
   sync->serial = 0;
   return VK_SUCCESS;
}

void
wvk_sync_set_pending(struct vk_sync *vk_sync, uint64_t serial)
{
   struct wvk_sync *sync = container_of(vk_sync, struct wvk_sync, base);
   assert(wvk_sync_is_wvk(vk_sync));
   sync->signaled = false;
   sync->serial = serial;
}

static bool
is_done(struct wvk_queue *queue, const struct wvk_sync *sync, bool pending_is_enough)
{
   if (sync->signaled)
      return true;
   if (!sync->serial)
      return false;
   return pending_is_enough || wvk_queue_completed(queue) >= sync->serial;
}

static VkResult
wvk_sync_wait_many(struct vk_device *vk_device, uint32_t wait_count,
                   const struct vk_sync_wait *waits, enum vk_sync_wait_flags flags,
                   uint64_t abs_timeout_ns)
{
   struct wvk_device *dev = container_of(vk_device, struct wvk_device, vk);
   const bool any = flags & VK_SYNC_WAIT_ANY;
   const bool pending = flags & VK_SYNC_WAIT_PENDING;

   /* GetFenceStatus and zero-timeout waits must advance completion too.
    * Native callbacks are WaitAnyOnly, and retiring a submit publishes its
    * readbacks. Testing the deadline before polling leaves a polling-only
    * client (Skia's async readback) observing NOT_READY forever. */
   if (!pending)
      wvk_queue_wait_progress(&dev->queue, 0);

   for (;;) {
      uint32_t done = 0;
      for (uint32_t i = 0; i < wait_count; ++i) {
         const struct wvk_sync *sync = container_of(waits[i].sync, struct wvk_sync, base);
         if (is_done(&dev->queue, sync, pending)) {
            if (any)
               return VK_SUCCESS;
            ++done;
         }
      }
      if (done == wait_count)
         return VK_SUCCESS;
      if (vk_device_is_lost(&dev->vk))
         return VK_ERROR_DEVICE_LOST;
      if (os_time_get_nano() >= abs_timeout_ns)
         return VK_TIMEOUT;

      /* Something on the queue finishing may be what is waited for. If
       * nothing is on it, only another thread can signal the host side;
       * poll for that. */
      VkResult result = wvk_queue_wait_progress(&dev->queue, abs_timeout_ns);
      if (result == VK_ERROR_DEVICE_LOST)
         return result;
      if (result == VK_NOT_READY)
         wvk_platform_sleep(100 * 1000);
   }
}

const struct vk_sync_type wvk_sync_type = {
   .size = sizeof(struct wvk_sync),
   .features = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_GPU_WAIT | VK_SYNC_FEATURE_GPU_MULTI_WAIT |
               VK_SYNC_FEATURE_CPU_WAIT |
               VK_SYNC_FEATURE_CPU_RESET | VK_SYNC_FEATURE_CPU_SIGNAL | VK_SYNC_FEATURE_WAIT_ANY |
               VK_SYNC_FEATURE_WAIT_PENDING,
   .init = wvk_sync_init,
   .finish = wvk_sync_finish,
   .signal = wvk_sync_signal,
   .reset = wvk_sync_reset,
   .wait_many = wvk_sync_wait_many,
};
