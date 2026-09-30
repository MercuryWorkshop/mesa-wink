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
 * Timestamps: WebGPU writes them only where a pass begins or ends, so
 * each is an empty compute pass's beginning -- a render pass ended for it,
 * and begun again after -- resolved and copied to the pool's memory then,
 * in nanoseconds.
 *
 * Pipeline statistics, which WebGPU has none of, are the driver's counts:
 * what the host knows of each draw and dispatch (wvk_cmd_draw.c), and what
 * it cannot, added up on the GPU in a buffer of the query's -- fragment
 * shader invocations and a geometry shader's primitives by the shaders
 * themselves, what the tessellator made and indirect commands by passes of
 * the driver's. At the end, a pass adds the two into the pool's memory.
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
#define WVK_TIMESTAMP_SET_SIZE 1024
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
   pool->values = pool->vk.query_type == VK_QUERY_TYPE_PIPELINE_STATISTICS
                     ? MAX2(util_bitcount(pool->vk.pipeline_statistics), 1)
                  : pool->vk.query_type == VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT ? 2
                                                                                    : 1;

   pool->serial = calloc(pool->vk.query_count, sizeof(*pool->serial));
   VkResult result = pool->serial ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
   if (result == VK_SUCCESS)
      result = wvk_AllocateMemory(_device, &(VkMemoryAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = align64((uint64_t)pool->vk.query_count * pool->values * 8, 256),
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
      memset((uint64_t *)pool->mem->host + (uint64_t)q * pool->values, 0, pool->values * 8);
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
      const uint32_t size = is64 ? 8 : 4;
      for (uint32_t v = 0; v < pool->values && (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)); ++v) {
         const uint64_t value = available ? ((const uint64_t *)pool->mem->host)[(uint64_t)q * pool->values + v] : 0;
         write_result(dst + v * size, is64, value);
      }
      if (!available)
         result = VK_NOT_READY;
      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT)
         write_result(dst + pool->values * size, is64, available);
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
      bd.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc | WGPUBufferUsage_Storage;
      bd.size = WVK_QUERY_SET_SIZE * 8;
      replay->query_resolve = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
   }
   /* Each pass's queries start where a resolve can write to. */
   replay->query_next = align(replay->query_next, WVK_QUERY_ALIGN);
   replay->pass_query_first = replay->query_next;
   return replay->query_set;
}

/* A part of a query added to the parts before: 64-bit, which WGSL adds a
 * word at a time. */
static const char add_query_wgsl[] =
   "struct P { src: u32, dst: u32 }\n"
   "@group(0) @binding(0) var<storage, read> src: array<u32>;\n"
   "@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n"
   "@group(0) @binding(2) var<uniform> p: P;\n"
   "@compute @workgroup_size(1)\n"
   "fn main() {\n"
   "  let lo = dst[p.dst] + src[p.src];\n"
   "  let carry = select(0u, 1u, lo < src[p.src]);\n"
   "  dst[p.dst + 1u] = dst[p.dst + 1u] + src[p.src + 1u] + carry;\n"
   "  dst[p.dst] = lo;\n"
   "}\n";

static void
add_query_part(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_pass_query *pq)
{
   simple_mtx_lock(&dev->cache_lock);
   if (!dev->add_query_pipeline) {
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(add_query_wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk add query");
      desc.compute.module = module;
      desc.compute.entryPoint = wvk_sv("main");
      dev->add_query_pipeline = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   simple_mtx_unlock(&dev->cache_lock);
   const uint32_t params[4] = { pq->slot * 2, pq->query * 2 };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   if (!dev->add_query_pipeline || !params_buffer)
      return;
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = replay->query_resolve, .offset = 0, .size = WVK_QUERY_SET_SIZE * 8 },
      { .binding = 1, .buffer = pq->pool->buffer, .offset = 0, .size = pq->pool->mem->buffer_size },
      { .binding = 2, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
   };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = wgpuComputePipelineGetBindGroupLayout(dev->add_query_pipeline, 0);
   bg.entryCount = ARRAY_SIZE(entries);
   bg.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bg.layout);
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderSetPipeline(pass, dev->add_query_pipeline);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
}

void
wvk_replay_suspend_query(struct wvk_replay *replay)
{
   struct wvk_pass_query *q = &replay->active_query;
   if (!replay->pass || !q->pool || q->slot == UINT32_MAX || q->suspended)
      return;
   wgpuRenderPassEncoderEndOcclusionQuery(replay->pass);
   struct wvk_pass_query part = *q;
   part.accumulate = q->parts++ > 0;
   util_dynarray_append(&replay->pass_queries, part);
   q->suspended = true;
}

void
wvk_replay_resume_query(struct wvk_replay *replay)
{
   struct wvk_pass_query *q = &replay->active_query;
   if (!q->pool || !q->suspended || !replay->pass || !replay->query_set ||
       replay->query_next >= WVK_QUERY_SET_SIZE)
      return;
   q->slot = replay->query_next++;
   q->suspended = false;
   wgpuRenderPassEncoderBeginOcclusionQuery(replay->pass, q->slot);
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
      if (pq->accumulate)
         add_query_part(replay->dev, replay, pq);
      else
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
   if (replay->timestamp_set)
      wgpuQuerySetRelease(replay->timestamp_set);
   if (replay->timestamp_resolve)
      wgpuBufferRelease(replay->timestamp_resolve);
   util_dynarray_fini(&replay->pass_queries);
}

/* A compute pipeline of the device's, made the first time. */
static WGPUComputePipeline
device_pipeline(struct wvk_device *dev, WGPUComputePipeline *pipeline, const char *wgsl, const char *label)
{
   simple_mtx_lock(&dev->cache_lock);
   if (!*pipeline) {
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv(label);
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv(label);
      desc.compute.module = module;
      desc.compute.entryPoint = wvk_sv("main");
      *pipeline = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   simple_mtx_unlock(&dev->cache_lock);
   return *pipeline;
}

/* One invocation's pass of `pipeline`, of `entries` for its group. */
static void
one_pass(struct wvk_device *dev, struct wvk_replay *replay, WGPUComputePipeline pipeline,
         const WGPUBindGroupEntry *entries, uint32_t count)
{
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
   bg.entryCount = count;
   bg.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bg.layout);
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderSetPipeline(pass, pipeline);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
}

uint32_t
wvk_replay_stats(const struct wvk_replay *replay)
{
   return replay->counter_mask;
}

void
wvk_replay_count(struct wvk_replay *replay, uint32_t counter, uint64_t n)
{
   /* Once for a pass replayed a layer at a time: its draws are the same. */
   if (!(replay->counter_mask & BITFIELD_BIT(counter)) ||
       (replay->pass_layered && replay->pass_layer != replay->pass_first_layer))
      return;
   replay->host_counters[counter] += n;
}

void
wvk_replay_count_stat(struct wvk_replay *replay, VkQueryPipelineStatisticFlagBits bit, uint64_t n)
{
   wvk_replay_count(replay, util_logbase2(bit), n);
}

WGPUBuffer
wvk_replay_counters(struct wvk_replay *replay)
{
   if (!replay->counters) {
      static const uint32_t zero[WVK_COUNTERS] = { 0 };
      replay->counters = wvk_replay_temp_buffer(replay->dev, replay, sizeof(zero),
                                                WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc, zero);
      replay->counters_serial = ++replay->dev->temp_serial | (1ull << 62);
   }
   return replay->counters;
}

static const char add_stats_wgsl[] =
   "struct P { word: u32, divisor: u32, bits: u32, pad: u32 }\n"
   "@group(0) @binding(0) var<uniform> p: P;\n"
   "@group(0) @binding(1) var<storage, read> src: array<u32>;\n"
   "@group(0) @binding(2) var<storage, read_write> stats: array<atomic<u32>>;\n"
   "@compute @workgroup_size(1)\n"
   "fn main() {\n"
   "  let v = src[p.word] / max(p.divisor, 1u);\n"
   "  for (var i = 0u; i < 32u; i++) {\n"
   "    if ((p.bits & (1u << i)) != 0u) { atomicAdd(&stats[i], v); }\n"
   "  }\n"
   "}\n";

void
wvk_replay_add_stats(struct wvk_replay *replay, WGPUBuffer src, uint64_t src_size, uint32_t word,
                     uint32_t divisor, uint32_t bits)
{
   struct wvk_device *dev = replay->dev;
   bits &= replay->counter_mask;
   if (!bits || (replay->pass_layered && replay->pass_layer != replay->pass_first_layer))
      return;
   WGPUComputePipeline pipeline = device_pipeline(dev, &dev->add_stats_pipeline, add_stats_wgsl, "wvk add statistics");
   const uint32_t params[4] = { word, divisor, bits };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   WGPUBuffer counters = wvk_replay_counters(replay);
   if (!pipeline || !params_buffer || !counters)
      return;
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
      { .binding = 1, .buffer = src, .offset = 0, .size = align64(src_size, 4) },
      { .binding = 2, .buffer = counters, .offset = 0, .size = WVK_COUNTERS * 4 },
   };
   one_pass(dev, replay, pipeline, entries, ARRAY_SIZE(entries));
}

/* The counters a query has values of, in the order its values are: of
 * vertex stream `index`, for a query of one. */
static uint32_t
query_counter_mask(const struct wvk_query_pool *pool, uint32_t index)
{
   index = MIN2(index, 3);
   switch (pool->vk.query_type) {
   case VK_QUERY_TYPE_PIPELINE_STATISTICS:
      return pool->vk.pipeline_statistics;
   case VK_QUERY_TYPE_PRIMITIVES_GENERATED_EXT:
      return BITFIELD_BIT(WVK_COUNTER_PRIMITIVES(index));
   case VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT:
      return BITFIELD_BIT(WVK_COUNTER_MESH_PRIMITIVES);
   case VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT:
      return BITFIELD_BIT(WVK_COUNTER_XFB_WRITTEN(index)) | BITFIELD_BIT(WVK_COUNTER_XFB_NEEDED(index));
   default:
      return 0;
   }
}

static void
update_counter_mask(struct wvk_replay *replay)
{
   replay->counter_mask = 0;
   for (uint32_t i = 0; i < replay->counter_query_count; ++i)
      replay->counter_mask |= replay->counter_queries[i].mask;
}

/* A query begun: the counters where they are, the GPU's copied outside a
 * pass, the pass ended for it and begun again. */
static void
begin_counters(struct wvk_cmd_buffer *cmd, struct wvk_query_pool *pool, uint32_t query, uint32_t index)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_device *dev = replay->dev;
   if (replay->counter_query_count == WVK_COUNTER_QUERIES) {
      mesa_logw_once("wvk: more than %u counted queries at once", WVK_COUNTER_QUERIES);
      return;
   }
   WGPUBuffer counters = wvk_replay_counters(replay);
   WGPUBuffer begin = wvk_replay_temp_buffer(dev, replay, WVK_COUNTERS * 4,
                                             WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, NULL);
   if (!counters || !begin)
      return;
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, counters, 0, begin, 0, WVK_COUNTERS * 4);
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   struct wvk_counter_query *q = &replay->counter_queries[replay->counter_query_count++];
   *q = (struct wvk_counter_query){ .pool = pool, .query = query, .mask = query_counter_mask(pool, index),
                                   .begin = begin };
   memcpy(q->host_begin, replay->host_counters, sizeof(q->host_begin));
   update_counter_mask(replay);
}

/* A query's results: for each counter it has, in the order of their bits,
 * what the host counted since its begin and what the GPU did, to its
 * values in the pool. */
static const char finish_stats_wgsl[] =
   "struct P { words: array<vec4u, 17> }\n"
   "@group(0) @binding(0) var<uniform> p: P;\n"
   "@group(0) @binding(1) var<storage, read> now: array<u32>;\n"
   "@group(0) @binding(2) var<storage, read> begin: array<u32>;\n"
   "@group(0) @binding(3) var<storage, read_write> results: array<u32>;\n"
   "fn w(i: u32) -> u32 { return p.words[i / 4u][i % 4u]; }\n"
   "@compute @workgroup_size(1)\n"
   "fn main() {\n"
   "  let mask = w(64u);\n"
   "  var at = w(65u);\n"
   "  for (var i = 0u; i < 32u; i++) {\n"
   "    if ((mask & (1u << i)) == 0u) { continue; }\n"
   "    let gpu = now[i] - begin[i];\n"
   "    let lo = w(2u * i) + gpu;\n"
   "    results[at] = lo;\n"
   "    results[at + 1u] = w(2u * i + 1u) + select(0u, 1u, lo < gpu);\n"
   "    at += 2u;\n"
   "  }\n"
   "}\n";

static void
end_counters(struct wvk_cmd_buffer *cmd, struct wvk_query_pool *pool, uint32_t query)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_device *dev = replay->dev;
   uint32_t n = 0;
   while (n < replay->counter_query_count &&
          (replay->counter_queries[n].pool != pool || replay->counter_queries[n].query != query))
      ++n;
   if (n == replay->counter_query_count)
      return;
   const struct wvk_counter_query q = replay->counter_queries[n];
   replay->counter_queries[n] = replay->counter_queries[--replay->counter_query_count];
   update_counter_mask(replay);

   WGPUComputePipeline pipeline =
      device_pipeline(dev, &dev->finish_stats_pipeline, finish_stats_wgsl, "wvk finish statistics");
   uint32_t params[68] = { 0 };
   for (uint32_t i = 0; i < WVK_COUNTERS; ++i) {
      const uint64_t delta = replay->host_counters[i] - q.host_begin[i];
      params[2 * i] = (uint32_t)delta;
      params[2 * i + 1] = (uint32_t)(delta >> 32);
   }
   params[64] = q.mask;
   params[65] = query * pool->values * 2;
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   if (!pipeline || !params_buffer)
      return;
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
      { .binding = 1, .buffer = replay->counters, .offset = 0, .size = WVK_COUNTERS * 4 },
      { .binding = 2, .buffer = q.begin, .offset = 0, .size = WVK_COUNTERS * 4 },
      { .binding = 3, .buffer = pool->buffer, .offset = 0, .size = pool->mem->buffer_size },
   };
   one_pass(dev, replay, pipeline, entries, ARRAY_SIZE(entries));
   wvk_replay_write(replay, pool->mem, (uint64_t)query * pool->values * 8, pool->values * 8);
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBeginQueryIndexedEXT(VkCommandBuffer commandBuffer, VkQueryPool queryPool, uint32_t query,
                            VkQueryControlFlags flags, uint32_t index)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_query_pool, pool, queryPool);
   struct wvk_replay *replay = cmd->replay;
   if (query_counter_mask(pool, index)) {
      begin_counters(cmd, pool, query, index);
      return;
   }
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
   if (query_counter_mask(pool, index)) {
      end_counters(cmd, pool, query);
      return;
   }
   if (pool->vk.query_type != VK_QUERY_TYPE_OCCLUSION)
      return;
   if (replay->active_query.pool == pool && replay->active_query.slot != UINT32_MAX && replay->pass &&
       !replay->active_query.suspended) {
      wgpuRenderPassEncoderEndOcclusionQuery(replay->pass);
      struct wvk_pass_query part = replay->active_query;
      part.accumulate = part.parts > 0;
      util_dynarray_append(&replay->pass_queries, part);
   } else if (replay->active_query.pool == pool && replay->active_query.parts) {
      /* Counted, in the parts it had. */
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
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   if (query >= pool->vk.query_count)
      return;
   pool->serial[query] = replay->serial;
   ((uint64_t *)pool->mem->host)[query] = 0;
   if (!wvk_physical_device_has_feature(wvk_device_physical(dev), WGPUFeatureName_TimestampQuery))
      return;
   if (!replay->timestamp_set) {
      WGPUQuerySetDescriptor desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk timestamps");
      desc.type = WGPUQueryType_Timestamp;
      desc.count = WVK_TIMESTAMP_SET_SIZE;
      replay->timestamp_set = wgpuDeviceCreateQuerySet(dev->wgpu, &desc);
      WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
      bd.label = wvk_sv("wvk timestamp resolve");
      bd.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
      bd.size = WVK_TIMESTAMP_SET_SIZE * 256;
      replay->timestamp_resolve = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
   }
   if (replay->timestamp_next >= WVK_TIMESTAMP_SET_SIZE) {
      mesa_logw_once("wvk: more than %u timestamps in a submit", WVK_TIMESTAMP_SET_SIZE);
      return;
   }
   const uint32_t slot = replay->timestamp_next++;
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   WGPUPassTimestampWrites writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
   writes.querySet = replay->timestamp_set;
   writes.beginningOfPassWriteIndex = slot;
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   pd.timestampWrites = &writes;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuCommandEncoderResolveQuerySet(replay->encoder, replay->timestamp_set, slot, 1, replay->timestamp_resolve,
                                     (uint64_t)slot * 256);
   wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, replay->timestamp_resolve, (uint64_t)slot * 256,
                                        pool->buffer, (uint64_t)query * 8, 8);
   wvk_replay_write(replay, pool->mem, (uint64_t)query * 8, 8);
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

/* Results to a buffer on the GPU, in whatever layout the flags ask: a
 * compute shader, as WebGPU copies only bytes. Whether a query is
 * available there is known now, on the host: replay is in queue order. */
static const char copy_results_wgsl[] =
   "struct Params { first: u32, count: u32, stride: u32, flags: u32, dst_word: u32, values: u32 }\n"
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
   "  let size = select(1u, 2u, is64);\n"
   "  for (var v = 0u; v < p.values; v++) {\n"
   "    let lo = results[2u * (q * p.values + v)];\n"
   "    let hi = results[2u * (q * p.values + v) + 1u];\n"
   "    let at = w + v * size;\n"
   "    if (a || partial) {\n"
   "      if (is64) { dst[at] = select(0u, lo, a); dst[at + 1u] = select(0u, hi, a); }\n"
   "      else { dst[at] = select(0u, select(0xffffffffu, lo, hi == 0u), a); }\n"
   "    }\n"
   "  }\n"
   "  let av = w + p.values * size;\n"
   "  if (with_avail) {\n"
   "    if (is64) { dst[av] = u32(a); dst[av + 1u] = 0u; } else { dst[av] = u32(a); }\n"
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
                        (pool->values + ((flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ? 1 : 0));
   const uint32_t params[8] = {
      firstQuery, queryCount, (uint32_t)stride, flags, (uint32_t)((start - base) / 4), pool->values,
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

