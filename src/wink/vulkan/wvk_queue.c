/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_queue.h"

#include "wvk_cmd_buffer.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_instance.h"
#include "wvk_physical_device.h"
#include "wvk_platform.h"
#include "wvk_sync.h"

#include "util/os_time.h"

#include <stdlib.h>

static struct wvk_instance *
queue_instance(struct wvk_queue *queue)
{
   return (struct wvk_instance *)wvk_queue_device(queue)->vk.physical->instance;
}

static struct wvk_pending_submit *
pending_at(struct wvk_queue *queue, uint32_t i)
{
   return &queue->pending[(queue->pending_first + i) % WVK_MAX_PENDING_SUBMITS];
}

/* webgpu.h gave the callback a message in mid-2025, in the same revision that
 * moved the per-stage storage limits to WGPUCompatibilityModeLimits. */
static void
#ifdef WGPU_COMPATIBILITY_MODE_LIMITS_INIT
on_work_done(WGPUQueueWorkDoneStatus status, WGPUStringView message, void *userdata1, void *userdata2)
#else
on_work_done(WGPUQueueWorkDoneStatus status, void *userdata1, void *userdata2)
#endif
{
   struct wvk_queue *queue = userdata1;
   struct wvk_pending_submit *p = userdata2;
   if (status != WGPUQueueWorkDoneStatus_Success)
      vk_device_set_lost(&wvk_queue_device(queue)->vk, "queue work failed");
   p->done = true;
   wvk_platform_signal();
}

static void
on_mapped(WGPUMapAsyncStatus status, WGPUStringView message, void *userdata1, void *userdata2)
{
   struct wvk_readback *rb = userdata1;
   (void)userdata2;
   rb->mapped = true;
   if (status != WGPUMapAsyncStatus_Success) {
      rb->failed = true;
      mesa_loge("wvk: reading host-visible memory back failed: %.*s", wvk_sv_len(message), message.data);
   }
   wvk_platform_signal();
}

/* The oldest submit is done if its work is, and every readback mapped:
 * then its memory goes back to the host copies, in order. */
static bool
retire_oldest(struct wvk_queue *queue)
{
   if (!queue->pending_count)
      return false;
   struct wvk_pending_submit *p = pending_at(queue, 0);
   if (!p->done)
      return false;
   for (uint32_t i = 0; i < p->readback_count; ++i) {
      if (!p->readbacks[i].mapped)
         return false;
   }
   for (uint32_t i = 0; i < p->readback_count; ++i) {
      struct wvk_readback *rb = &p->readbacks[i];
      /* Read into the host copy, not through a pointer to the mapping,
       * which in a browser is a copy of it in the wasm heap. */
      if (!rb->failed)
         wgpuBufferReadMappedRange(rb->staging, 0, (uint8_t *)rb->mem->host + rb->offset, rb->size);
      wgpuBufferUnmap(rb->staging);
      wgpuBufferDestroy(rb->staging);
      wgpuBufferRelease(rb->staging);
   }
   free(p->readbacks);
   p->readbacks = NULL;
   p->readback_count = 0;
   /* In order, so this is a high-water mark. */
   p_atomic_set(&queue->completed, p->serial);
   queue->pending_first = (queue->pending_first + 1) % WVK_MAX_PENDING_SUBMITS;
   --queue->pending_count;
   return true;
}

/* How long one waiter holds the queue: others waiting with a shorter
 * timeout get their turn in between. */
#define WVK_WAIT_SLICE_NS (2ull * 1000 * 1000)

VkResult
wvk_queue_wait_progress(struct wvk_queue *queue, uint64_t abs_timeout_ns)
{
   simple_mtx_lock(&queue->lock);
   if (retire_oldest(queue)) {
      while (retire_oldest(queue))
         ;
      simple_mtx_unlock(&queue->lock);
      return VK_SUCCESS;
   }
   if (!queue->pending_count) {
      simple_mtx_unlock(&queue->lock);
      return VK_NOT_READY;
   }

   /* Wait for whatever the oldest submit is still waiting for. */
   struct wvk_pending_submit *p = pending_at(queue, 0);
   WGPUFutureWaitInfo waits[64];
   uint32_t count = 0;
   if (!p->done)
      waits[count++] = (WGPUFutureWaitInfo){ .future = p->work_done };
   for (uint32_t i = 0; i < p->readback_count && count < ARRAY_SIZE(waits); ++i) {
      if (!p->readbacks[i].mapped)
         waits[count++] = (WGPUFutureWaitInfo){ .future = p->readbacks[i].future };
   }

   const uint64_t now = os_time_get_nano();
   const uint64_t timeout = abs_timeout_ns == OS_TIMEOUT_INFINITE ? UINT64_MAX
                            : abs_timeout_ns > now ? abs_timeout_ns - now : 0;
   const WGPUWaitStatus status = wvk_platform_wait_any(queue_instance(queue)->wgpu, waits, count,
                                                       MIN2(timeout, WVK_WAIT_SLICE_NS));

   bool progressed = false;
   while (retire_oldest(queue))
      progressed = true;
   simple_mtx_unlock(&queue->lock);
   if (status == WGPUWaitStatus_Error)
      return vk_device_set_lost(&wvk_queue_device(queue)->vk, "waiting for WebGPU failed");
   if (vk_device_is_lost(&wvk_queue_device(queue)->vk))
      return VK_ERROR_DEVICE_LOST;
   /* A readback mapping is progress too, if not yet a finished submit. */
   return progressed || status == WGPUWaitStatus_Success ? VK_SUCCESS : VK_TIMEOUT;
}

bool
wvk_queue_range_pending(struct wvk_queue *queue, struct wvk_device_memory *mem,
                        uint64_t offset, uint64_t size)
{
   simple_mtx_lock(&queue->lock);
   bool pending = false;
   for (uint32_t i = 0; i < queue->pending_count && !pending; ++i) {
      const struct wvk_pending_submit *p = pending_at(queue, i);
      for (uint32_t j = 0; j < p->readback_count; ++j) {
         const struct wvk_readback *rb = &p->readbacks[j];
         if (rb->mem == mem && rb->offset < offset + size && offset < rb->offset + rb->size)
            pending = true;
      }
   }
   simple_mtx_unlock(&queue->lock);
   return pending;
}

VkResult
wvk_queue_reserve(struct wvk_queue *queue, uint64_t *serial)
{
   simple_mtx_lock(&queue->lock);
   while (retire_oldest(queue))
      ;
   simple_mtx_unlock(&queue->lock);
   while (p_atomic_read(&queue->pending_count) == WVK_MAX_PENDING_SUBMITS) {
      VkResult result = wvk_queue_wait_progress(queue, OS_TIMEOUT_INFINITE);
      if (result == VK_ERROR_DEVICE_LOST)
         return result;
   }
   *serial = queue->submitted + 1;
   return VK_SUCCESS;
}

void
wvk_queue_commit(struct wvk_queue *queue, uint64_t serial,
                 struct wvk_readback *readbacks, uint32_t readback_count)
{
   simple_mtx_lock(&queue->lock);
   assert(serial == queue->submitted + 1);
   queue->submitted = serial;
   struct wvk_pending_submit *p = pending_at(queue, queue->pending_count);
   *p = (struct wvk_pending_submit){
      .serial = serial,
      .readbacks = readbacks,
      .readback_count = readback_count,
   };
   ++queue->pending_count;

   WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
   cb.mode = wvk_platform_callback_mode();
   cb.callback = on_work_done;
   cb.userdata1 = queue;
   cb.userdata2 = p;
   p->work_done = wgpuQueueOnSubmittedWorkDone(queue->wgpu, cb);

   for (uint32_t i = 0; i < readback_count; ++i) {
      WGPUBufferMapCallbackInfo map = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
      map.mode = wvk_platform_callback_mode();
      map.callback = on_mapped;
      map.userdata1 = &readbacks[i];
      readbacks[i].future = wgpuBufferMapAsync(readbacks[i].staging, WGPUMapMode_Read, 0,
                                               readbacks[i].size, map);
   }
   simple_mtx_unlock(&queue->lock);
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

   simple_mtx_lock(&queue->submit_lock);
   uint64_t serial;
   VkResult result = wvk_queue_reserve(queue, &serial);
   if (result == VK_SUCCESS)
      result = wvk_cmd_buffers_submit(dev, queue, submit->command_buffers,
                                      submit->command_buffer_count, serial);

   for (uint32_t i = 0; result == VK_SUCCESS && i < submit->signal_count; ++i) {
      struct vk_sync *sync = submit->signals[i].sync;
      if (wvk_sync_is_wvk(sync))
         wvk_sync_set_pending(sync, serial);
      else
         result = vk_sync_signal(&dev->vk, sync, submit->signals[i].signal_value);
   }
   simple_mtx_unlock(&queue->submit_lock);
   return result;
}

static VkResult
wvk_compute_queue_submit(struct vk_queue *vk_queue, struct vk_queue_submit *submit)
{
   struct wvk_device *dev = (struct wvk_device *)vk_queue->base.device;
   return wvk_queue_submit(&dev->queue.vk, submit);
}

VkResult
wvk_compute_queue_init(struct wvk_device *dev, struct vk_queue *queue,
                       const VkDeviceQueueCreateInfo *create_info)
{
   VkResult result = vk_queue_init(queue, &dev->vk, create_info, 0);
   if (result == VK_SUCCESS)
      queue->driver_submit = wvk_compute_queue_submit;
   return result;
}

VkResult
wvk_queue_init(struct wvk_device *dev, struct wvk_queue *queue,
               const VkDeviceQueueCreateInfo *create_info, uint32_t index)
{
   VkResult result = vk_queue_init(&queue->vk, &dev->vk, create_info, index);
   if (result != VK_SUCCESS)
      return result;

   simple_mtx_init(&queue->lock, mtx_plain);
   simple_mtx_init(&queue->submit_lock, mtx_plain);
   queue->wgpu = wgpuDeviceGetQueue(dev->wgpu);
   queue->vk.driver_submit = wvk_queue_submit;
   return VK_SUCCESS;
}

void
wvk_queue_finish(struct wvk_device *dev, struct wvk_queue *queue)
{
   /* Let what was submitted finish before the device goes. */
   while (queue->pending_count &&
          wvk_queue_wait_progress(queue, OS_TIMEOUT_INFINITE) != VK_ERROR_DEVICE_LOST)
      ;
   if (queue->wgpu)
      wgpuQueueRelease(queue->wgpu);
   simple_mtx_destroy(&queue->lock);
   simple_mtx_destroy(&queue->submit_lock);
   vk_queue_finish(&queue->vk);
}
