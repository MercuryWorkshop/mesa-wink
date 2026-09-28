/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Queries. WebGPU's occlusion queries belong to a render pass: its query
 * set is named when the pass begins, and the results are resolved into a
 * buffer after it ends. So a command buffer that has occlusion queries
 * gets a query set of the submit's on every pass, each Vulkan query the
 * next slot of it, and at the end of the pass the results go to the pool's
 * memory -- host-visible, so they come back like any other memory a
 * submit wrote. A query is available once the submit it ended in is done.
 *
 * Timestamps are zero: WebGPU writes them only at pass boundaries, and
 * timestampValidBits says there are none.
 */
#include "wvk_query.h"

#include "wvk_buffer.h"
#include "wvk_cmd_buffer.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_physical_device.h"
#include "wvk_platform.h"
#include "wvk_queue.h"

#include "vk_cmd_queue.h"

#include "util/os_time.h"

#include <stdlib.h>

#define WVK_QUERY_SET_SIZE 4096
/* resolveQuerySet writes to 256-byte aligned offsets: 32 queries. */
#define WVK_QUERY_ALIGN 32

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateQueryPool(VkDevice _device, const VkQueryPoolCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator, VkQueryPool *pQueryPool)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   struct wvk_query_pool *pool = vk_query_pool_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*pool));
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   if (pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS)
      mesa_logw_once("wvk: pipeline statistics queries are not supported; they read zero");

   pool->serial = calloc(pool->vk.query_count, sizeof(*pool->serial));
   VkResult result = pool->serial ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
   if (result == VK_SUCCESS)
      result = wvk_AllocateMemory(_device, &(VkMemoryAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = align64((uint64_t)pool->vk.query_count * 8, 256),
         .memoryTypeIndex = WVK_MEMORY_READBACK,
      }, NULL, &pool->memory);
   if (result == VK_SUCCESS) {
      pool->mem = wvk_device_memory_from_handle(pool->memory);
      result = wvk_device_memory_get_buffer(dev, pool->mem, &pool->buffer);
   }
   if (result != VK_SUCCESS) {
      if (pool->memory)
         wvk_FreeMemory(_device, pool->memory, NULL);
      free(pool->serial);
      vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
      return vk_error(dev, result);
   }
   *pQueryPool = wvk_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyQueryPool(VkDevice _device, VkQueryPool queryPool, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   if (!pool)
      return;
   wvk_FreeMemory(_device, pool->memory, NULL);
   free(pool->serial);
   vk_query_pool_destroy(&dev->vk, pAllocator, &pool->vk);
}

static void
reset_queries(struct wvk_query_pool *pool, uint32_t first, uint32_t count)
{
   for (uint32_t q = first; q < first + count && q < pool->vk.query_count; ++q) {
      pool->serial[q] = 0;
      ((uint64_t *)pool->mem->host)[q] = 0;
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_ResetQueryPool(VkDevice device, VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount)
{
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   reset_queries(pool, firstQuery, queryCount);
}

static bool
query_available(struct wvk_queue *queue, const struct wvk_query_pool *pool, uint32_t q)
{
   return pool->serial[q] && wvk_queue_completed(queue) >= pool->serial[q];
}

static void
write_result(void *dst, bool is64, uint64_t value)
{
   if (is64)
      *(uint64_t *)dst = value;
   else
      *(uint32_t *)dst = (uint32_t)MIN2(value, UINT32_MAX);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetQueryPoolResults(VkDevice _device, VkQueryPool queryPool, uint32_t firstQuery,
                        uint32_t queryCount, size_t dataSize, void *pData, VkDeviceSize stride,
                        VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   struct wvk_queue *queue = &dev->queue;
   const bool is64 = flags & VK_QUERY_RESULT_64_BIT;
   VkResult result = VK_SUCCESS;

   for (uint32_t i = 0; i < queryCount; ++i) {
      const uint32_t q = firstQuery + i;
      if (flags & VK_QUERY_RESULT_WAIT_BIT) {
         /* Ended on the queue, in a submit not yet done. */
         while (pool->serial[q] && !query_available(queue, pool, q)) {
            const VkResult r = wvk_queue_wait_progress(queue, OS_TIMEOUT_INFINITE);
            if (r == VK_ERROR_DEVICE_LOST)
               return r;
            if (r == VK_NOT_READY)
               wvk_platform_sleep(100 * 1000);
         }
      }
      const bool available = query_available(queue, pool, q);
      uint8_t *dst = (uint8_t *)pData + i * stride;
      if (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
         const uint64_t value = available ? ((const uint64_t *)pool->mem->host)[q] : 0;
         write_result(dst, is64, value);
      }
      if (!available)
         result = VK_NOT_READY;
      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
         write_result(dst + (is64 ? 8 : 4), is64, available);
   }
   return result;
}

/* ---------------------------------------------------------------------- */
/* Recording */

bool
wvk_cmd_queue_has_occlusion_queries(const struct vk_cmd_queue *queue)
{
   list_for_each_entry(struct vk_cmd_queue_entry, cmd, &queue->cmds, cmd_link) {
      VkQueryPool handle = VK_NULL_HANDLE;
      if (cmd->type == VK_CMD_BEGIN_QUERY)
         handle = cmd->u.begin_query.query_pool;
      else if (cmd->type == VK_CMD_BEGIN_QUERY_INDEXED_EXT)
         handle = cmd->u.begin_query_indexed_ext.query_pool;
      else if (cmd->type == VK_CMD_EXECUTE_COMMANDS)
         return true; /* A secondary might; the query set costs little. */
      if (handle && wvk_query_pool_from_handle(handle)->vk.query_type == VK_QUERY_TYPE_OCCLUSION)
         return true;
   }
   return false;
}

WGPUQuerySet
wvk_replay_query_set(struct wvk_device *dev, struct wvk_replay *replay)
{
   if (!replay->uses_queries)
      return NULL;
   if (!replay->query_set) {
      WGPUQuerySetDescriptor desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk occlusion queries");
      desc.type = WGPUQueryType_Occlusion;
      desc.count = WVK_QUERY_SET_SIZE;
      replay->query_set = wgpuDeviceCreateQuerySet(dev->wgpu, &desc);
      WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
      bd.label = wvk_sv("wvk query resolve");
      bd.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
      bd.size = WVK_QUERY_SET_SIZE * 8;
      replay->query_resolve = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
   }
   /* Each pass's queries start where a resolve can write to. */
   replay->query_next = align(replay->query_next, WVK_QUERY_ALIGN);
   replay->pass_query_first = replay->query_next;
   return replay->query_set;
}

void
wvk_replay_resolve_queries(struct wvk_replay *replay)
{
   const uint32_t first = replay->pass_query_first, count = replay->query_next - first;
   if (!count)
      return;
   wgpuCommandEncoderResolveQuerySet(replay->encoder, replay->query_set, first, count,
                                     replay->query_resolve, (uint64_t)first * 8);
   util_dynarray_foreach(&replay->pass_queries, struct wvk_pass_query, pq) {
      wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, replay->query_resolve, (uint64_t)pq->slot * 8,
                                           pq->pool->buffer, (uint64_t)pq->query * 8, 8);
      wvk_replay_write(replay, pq->pool->mem, (uint64_t)pq->query * 8, 8);
   }
   util_dynarray_clear(&replay->pass_queries);
   replay->pass_query_first = replay->query_next;
}

void
wvk_replay_finish_queries(struct wvk_replay *replay)
{
   if (replay->query_set)
      wgpuQuerySetRelease(replay->query_set);
   if (replay->query_resolve)
      wgpuBufferRelease(replay->query_resolve);
   util_dynarray_fini(&replay->pass_queries);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBeginQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query,
                            VkQueryControlFlags flags, uint32_t index)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   struct wvk_replay *replay = cmd->replay;
   if (pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION)
      return;
   if (!replay->pass || !replay->query_set) {
      mesa_logw_once("wvk: an occlusion query outside a render pass reads as visible");
      replay->active_query = (struct wvk_pass_query){ pool, query, UINT32_MAX };
      return;
   }
   if (replay->query_next >= WVK_QUERY_SET_SIZE) {
      mesa_logw_once("wvk: more than %u occlusion queries in a submit", WVK_QUERY_SET_SIZE);
      replay->active_query = (struct wvk_pass_query){ pool, query, UINT32_MAX };
      return;
   }
   replay->active_query = (struct wvk_pass_query){ pool, query, replay->query_next++ };
   wgpuRenderPassEncoderBeginOcclusionQuery(replay->pass, replay->active_query.slot);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdEndQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query,
                          uint32_t index)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   struct wvk_replay *replay = cmd->replay;
   if (query < pool->vk.query_count)
      pool->serial[query] = replay->serial;
   if (pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION)
      return;
   if (replay->active_query.pool == pool && replay->active_query.slot != UINT32_MAX && replay->pass) {
      wgpuRenderPassEncoderEndOcclusionQuery(replay->pass);
      util_dynarray_append(&replay->pass_queries, replay->active_query);
   } else {
      /* Not counted: say something was drawn, which is never wrong to
       * act on. */
      ((uint64_t *)pool->mem->host)[query] = 1;
   }
   replay->active_query = (struct wvk_pass_query){ 0 };
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdResetQueryPool(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery,
                      uint32_t queryCount)
{
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   /* In queue order: replay is, and the reset is before any end that
    * follows it. */
   reset_queries(pool, firstQuery, queryCount);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdWriteTimestamp2(VkCommandBuffer commandBuffer, VkPipelineStageFlags2 stage,
                       VkQueryPool queryPool, uint32_t query)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   if (query >= pool->vk.query_count)
      return;
   ((uint64_t *)pool->mem->host)[query] = 0;
   pool->serial[query] = cmd->replay->serial;
}

/* Results to a buffer on the GPU, in whatever layout the flags ask: a
 * compute shader, as WebGPU copies only bytes. Whether a query is
 * available there is known now, on the host: replay is in queue order. */
static const char copy_results_wgsl[] =
   "struct Params { first: u32, count: u32, stride: u32, flags: u32, dst_word: u32 }\n"
   "@group(0) @binding(0) var<storage, read> results: array<u32>;\n"
   "@group(0) @binding(1) var<storage, read> avail: array<u32>;\n"
   "@group(0) @binding(2) var<storage, read_write> dst: array<u32>;\n"
   "@group(0) @binding(3) var<uniform> p: Params;\n"
   "@compute @workgroup_size(64)\n"
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"
   "  let i = id.x;\n"
   "  if (i >= p.count) { return; }\n"
   "  let is64 = (p.flags & 1u) != 0u;\n"
   "  let with_avail = (p.flags & 4u) != 0u;\n"
   "  let partial = (p.flags & 8u) != 0u;\n"
   "  let a = avail[i] != 0u;\n"
   "  let w = p.dst_word + (i * p.stride) / 4u;\n"
   "  let q = p.first + i;\n"
   "  let lo = results[2u * q];\n"
   "  let hi = results[2u * q + 1u];\n"
   "  if (a || partial) {\n"
   "    if (is64) { dst[w] = select(0u, lo, a); dst[w + 1u] = select(0u, hi, a); }\n"
   "    else { dst[w] = select(0u, select(0xffffffffu, lo, hi == 0u), a); }\n"
   "  }\n"
   "  if (with_avail) {\n"
   "    if (is64) { dst[w + 2u] = u32(a); dst[w + 3u] = 0u; } else { dst[w + 1u] = u32(a); }\n"
   "  }\n"
   "}\n";

static WGPUComputePipeline
copy_results_pipeline(struct wvk_device *dev)
{
   simple_mtx_lock(&dev->cache_lock);
   if (!dev->copy_query_pipeline) {
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(copy_results_wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk copy query results");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk copy query results");
      desc.compute.module = module;
      desc.compute.entryPoint = wvk_sv("main");
      dev->copy_query_pipeline = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   simple_mtx_unlock(&dev->cache_lock);
   return dev->copy_query_pipeline;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyQueryPoolResults(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t firstQuery,
                            uint32_t queryCount, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                            VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   VK_FROM_HANDLE(wvk_buffer, dst, dstBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   if (!queryCount || !dst->mem)
      return;
   WGPUComputePipeline pipeline = copy_results_pipeline(dev);
   WGPUBuffer dst_buffer;
   if (!pipeline || wvk_device_memory_get_buffer(dev, dst->mem, &dst_buffer) != VK_SUCCESS)
      return;

   uint32_t *avail = calloc(queryCount, sizeof(*avail));
   if (!avail)
      return;
   for (uint32_t i = 0; i < queryCount; ++i)
      avail[i] = pool->serial[firstQuery + i] != 0;
   WGPUBuffer avail_buffer = wvk_replay_temp_buffer(dev, replay, queryCount * 4ull,
                                                    WGPUBufferUsage_Storage, avail);
   free(avail);

   /* Storage bindings start 256-byte aligned: the destination from below
    * its offset, and the shader told where in it to begin. */
   const uint64_t start = dst->offset + dstOffset;
   const uint64_t base = start & ~255ull;
   const uint32_t result_size = (flags & VK_QUERY_RESULT_64_BIT) ? 8 : 4;
   const uint64_t end = start + (queryCount - 1) * stride + result_size *
                        ((flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ? 2 : 1);
   const uint32_t params[8] = {
      firstQuery, queryCount, (uint32_t)stride, flags, (uint32_t)((start - base) / 4),
   };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params),
                                                     WGPUBufferUsage_Uniform, params);
   if (!avail_buffer || !params_buffer)
      return;

   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = pool->buffer, .offset = 0, .size = pool->mem->buffer_size },
      { .binding = 1, .buffer = avail_buffer, .offset = 0, .size = align64(queryCount * 4ull, 4) },
      { .binding = 2, .buffer = dst_buffer, .offset = base, .size = align64(end - base, 4) },
      { .binding = 3, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
   };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
   bg.entryCount = ARRAY_SIZE(entries);
   bg.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bg.layout);

   wvk_replay_end_passes(replay);
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderSetPipeline(pass, pipeline);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuComputePassEncoderDispatchWorkgroups(pass, DIV_ROUND_UP(queryCount, 64), 1, 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);

   wvk_replay_write(replay, dst->mem, start, end - start);
}

/* ---------------------------------------------------------------------- */
/* Events: host-side, set or reset when replayed -- which on one in-order
 * queue is before the commands after it and after those before it, if a
 * little before the GPU gets there. */

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateEvent(VkDevice _device, const VkEventCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   struct wvk_event *event = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*event), VK_OBJECT_TYPE_EVENT);
   if (!event)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   *pEvent = wvk_event_to_handle(event);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyEvent(VkDevice _device, VkEvent _event, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_event, event, _event);
   if (event)
      vk_object_free(&dev->vk, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetEventStatus(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(wvk_event, event, _event);
   return p_atomic_read(&event->set) ? VK_EVENT_SET : VK_EVENT_RESET;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_SetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(wvk_event, event, _event);
   p_atomic_set(&event->set, 1);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_ResetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(wvk_event, event, _event);
   p_atomic_set(&event->set, 0);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event, const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(wvk_event, event, _event);
   p_atomic_set(&event->set, 1);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event, VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(wvk_event, event, _event);
   p_atomic_set(&event->set, 0);
}

/* ---------------------------------------------------------------------- */
/* What Vulkan 1.2 has and this driver does not do. */

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateBufferView(VkDevice _device, const VkBufferViewCreateInfo *pCreateInfo,
                     const VkAllocationCallbacks *pAllocator, VkBufferView *pView)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   struct vk_buffer_view *view = vk_buffer_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   mesa_logw_once("wvk: texel buffers are not supported yet");
   *pView = vk_buffer_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyBufferView(VkDevice _device, VkBufferView bufferView, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(vk_buffer_view, view, bufferView);
   if (view)
      vk_buffer_view_destroy(&dev->vk, pAllocator, view);
}

VKAPI_ATTR uint64_t VKAPI_CALL
wvk_GetBufferOpaqueCaptureAddress(VkDevice device, const VkBufferDeviceAddressInfo *pInfo)
{
   return 0;
}

VKAPI_ATTR uint64_t VKAPI_CALL
wvk_GetDeviceMemoryOpaqueCaptureAddress(VkDevice device,
                                        const VkDeviceMemoryOpaqueCaptureAddressInfo *pInfo)
{
   return 0;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                         VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                         uint32_t stride)
{
   mesa_logw_once("wvk: vkCmdDrawIndirectCount is not supported (drawIndirectCount)");
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                VkBuffer countBuffer, VkDeviceSize countBufferOffset,
                                uint32_t maxDrawCount, uint32_t stride)
{
   mesa_logw_once("wvk: vkCmdDrawIndexedIndirectCount is not supported (drawIndirectCount)");
}
