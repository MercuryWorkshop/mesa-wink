/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Copies, fills and clears, between passes.
 *
 * WebGPU's buffer-texture copies want rows of a multiple of 256 bytes;
 * Vulkan takes any. An aligned copy is one WebGPU copy, and any other
 * goes a row at a time, for which WebGPU needs no row pitch at all --
 * except for depth and stencil, which are below.
 */
#include "wvk_cmd_buffer.h"

#include "wvk_buffer.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_format.h"
#include "wvk_image.h"
#include "wvk_physical_device.h"

#include "vk_format.h"
#include "util/format/u_format.h"
#include "util/ralloc.h"

static WGPUBuffer
buffer_wgpu(const struct wvk_buffer *buffer)
{
   return buffer->mem ? buffer->mem->buffer : NULL;
}

static uint64_t
buffer_offset(const struct wvk_buffer *buffer, uint64_t offset)
{
   return buffer->offset + offset;
}

static void repack(struct wvk_device *dev, struct wvk_replay *replay, WGPUBuffer src, uint64_t src_offset,
                   uint32_t src_pitch, uint32_t src_image, WGPUBuffer dst, uint64_t dst_offset, uint32_t dst_pitch,
                   uint32_t dst_image, uint32_t row_bytes, uint32_t rows, uint32_t images);

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyBuffer2(VkCommandBuffer commandBuffer, const VkCopyBufferInfo2 *pCopyBufferInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_buffer, src, pCopyBufferInfo->srcBuffer);
   VK_FROM_HANDLE(wvk_buffer, dst, pCopyBufferInfo->dstBuffer);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pCopyBufferInfo->regionCount; ++i) {
      const VkBufferCopy2 *r = &pCopyBufferInfo->pRegions[i];
      const uint64_t so = buffer_offset(src, r->srcOffset), dof = buffer_offset(dst, r->dstOffset);
      wvk_replay_read(replay, src->mem, so, r->size);
      wvk_replay_write(replay, dst->mem, dof, r->size);
      if ((so | dof | r->size) & 3) {
         /* WebGPU copies whole words: the rest is a shader's, byte by
          * byte, in rows of what a dispatch reaches. */
         const uint32_t row = 65535 * 64;
         for (uint64_t at = 0; at < r->size; at += row) {
            const uint32_t n = (uint32_t)MIN2(row, r->size - at);
            repack(wvk_cmd_buffer_device(cmd), replay, buffer_wgpu(src), so + at, n, n, buffer_wgpu(dst), dof + at,
                   n, n, n, 1, 1);
         }
         continue;
      }
      wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, buffer_wgpu(src), so, buffer_wgpu(dst), dof, r->size);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdUpdateBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                    VkDeviceSize dataSize, const void *pData)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_buffer, dst, dstBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, dataSize, WGPUBufferUsage_CopySrc, pData);
   const uint64_t off = buffer_offset(dst, dstOffset);
   wvk_replay_write(replay, dst->mem, off, dataSize);
   wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, temp, 0, buffer_wgpu(dst), off, dataSize);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdFillBuffer(VkCommandBuffer commandBuffer, VkBuffer dstBuffer, VkDeviceSize dstOffset,
                  VkDeviceSize fillSize, uint32_t data)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_buffer, dst, dstBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;

   uint64_t size = vk_buffer_range(&dst->vk, dstOffset, fillSize) & ~3ull;
   if (!size)
      return;
   wvk_replay_end_passes(replay);
   const uint64_t off = buffer_offset(dst, dstOffset);
   wvk_replay_write(replay, dst->mem, off, size);
   if (data == 0) {
      wgpuCommandEncoderClearBuffer(replay->encoder, buffer_wgpu(dst), off, size);
      return;
   }
   /* A pattern needs a source to copy from: of at most a megabyte, copied
    * as many times as the fill takes, not one as big as the fill -- which
    * was that much of the host's memory twice over, the wasm heap's for
    * good, and of the GPU's for the submit. */
   const uint64_t chunk = MIN2(size, 1024 * 1024);
   uint32_t *pattern = malloc(chunk);
   if (!pattern)
      return;
   for (uint64_t i = 0; i < chunk / 4; ++i)
      pattern[i] = data;
   WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, chunk, WGPUBufferUsage_CopySrc, pattern);
   free(pattern);
   if (!temp)
      return;
   for (uint64_t done = 0; done < size; done += chunk)
      wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, temp, 0, buffer_wgpu(dst), off + done,
                                           MIN2(chunk, size - done));
}

/* ---------------------------------------------------------------------- */
/* Buffers and images */

struct copy_layout {
   uint32_t block_w, block_h, block_bytes;
   uint32_t row_bytes;        /* one row of blocks, in the buffer */
   uint32_t rows_per_image;   /* in blocks */
};

static struct copy_layout
buffer_layout(VkFormat format, VkImageAspectFlags aspect, uint32_t row_length, uint32_t image_height,
              VkExtent3D extent)
{
   enum pipe_format pformat = vk_format_to_pipe_format(format);
   /* One aspect of a depth-stencil format is copied on its own. */
   if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT)
      pformat = PIPE_FORMAT_S8_UINT;
   else if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT)
      pformat = vk_format_to_pipe_format(vk_format_depth_only(format));
   struct copy_layout l = {
      .block_w = util_format_get_blockwidth(pformat),
      .block_h = util_format_get_blockheight(pformat),
      .block_bytes = util_format_get_blocksize(pformat),
   };
   const uint32_t width = row_length ? row_length : extent.width;
   const uint32_t height = image_height ? image_height : extent.height;
   l.row_bytes = DIV_ROUND_UP(width, l.block_w) * l.block_bytes;
   l.rows_per_image = DIV_ROUND_UP(height, l.block_h);
   return l;
}

static void
image_region(const struct wvk_image *image, const VkImageSubresourceLayers *sub, VkOffset3D offset,
             VkExtent3D extent, WGPUTexelCopyTextureInfo *tex, WGPUExtent3D *size)
{
   *tex = (WGPUTexelCopyTextureInfo)WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
   tex->texture = image->texture;
   tex->mipLevel = sub->mipLevel;
   tex->aspect = wvk_aspect(sub->aspectMask);
   const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
   tex->origin = (WGPUOrigin3D){
      (uint32_t)offset.x,
      image->vk.image_type == VK_IMAGE_TYPE_1D ? 0 : (uint32_t)offset.y,
      is_3d ? (uint32_t)offset.z : sub->baseArrayLayer,
   };
   const uint32_t layers = sub->layerCount == VK_REMAINING_ARRAY_LAYERS
                              ? image->vk.array_layers - sub->baseArrayLayer : sub->layerCount;
   *size = (WGPUExtent3D){
      extent.width,
      image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : extent.height,
      is_3d ? extent.depth : layers,
   };
   /* A region of a compressed image may end at the edge of the image, mid
    * block; WebGPU copies the whole block, which its texture has. */
   const enum pipe_format pformat = vk_format_to_pipe_format(image->vk.format);
   size->width = util_align_npot(size->width, util_format_get_blockwidth(pformat));
   size->height = util_align_npot(size->height, util_format_get_blockheight(pformat));
}

/* One buffer-texture copy, split into rows if WebGPU wants a pitch the
 * buffer does not have. */
static void
copy_buffer_texture(struct wvk_replay *replay, bool to_texture, WGPUBuffer buffer, uint64_t offset,
                    const struct copy_layout *l, const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size)
{
   const uint32_t block_rows = DIV_ROUND_UP(size.height, l->block_h);
   WGPUTexelCopyBufferInfo buf = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
   buf.buffer = buffer;

   if (l->row_bytes % 256 == 0 || (block_rows <= 1 && size.depthOrArrayLayers <= 1)) {
      buf.layout.offset = offset;
      /* A pitch WebGPU is given must be aligned; one row needs none. */
      buf.layout.bytesPerRow = l->row_bytes % 256 == 0 ? l->row_bytes : WGPU_COPY_STRIDE_UNDEFINED;
      buf.layout.rowsPerImage = l->row_bytes % 256 == 0 ? l->rows_per_image : WGPU_COPY_STRIDE_UNDEFINED;
      if (to_texture)
         wgpuCommandEncoderCopyBufferToTexture(replay->encoder, &buf, tex, &size);
      else
         wgpuCommandEncoderCopyTextureToBuffer(replay->encoder, tex, &buf, &size);
      return;
   }

   for (uint32_t z = 0; z < size.depthOrArrayLayers; ++z) {
      for (uint32_t row = 0; row < block_rows; ++row) {
         WGPUTexelCopyTextureInfo t = *tex;
         t.origin.y += row * l->block_h;
         t.origin.z += z;
         const uint32_t h = MIN2(l->block_h, size.height - row * l->block_h);
         const WGPUExtent3D one = { size.width, h, 1 };
         buf.layout.offset = offset + (uint64_t)z * l->rows_per_image * l->row_bytes +
                             (uint64_t)row * l->row_bytes;
         buf.layout.bytesPerRow = WGPU_COPY_STRIDE_UNDEFINED;
         buf.layout.rowsPerImage = WGPU_COPY_STRIDE_UNDEFINED;
         if (to_texture)
            wgpuCommandEncoderCopyBufferToTexture(replay->encoder, &buf, &t, &one);
         else
            wgpuCommandEncoderCopyTextureToBuffer(replay->encoder, &t, &buf, &one);
      }
   }
}

static uint64_t
region_bytes(const struct copy_layout *l, WGPUExtent3D size)
{
   const uint32_t block_rows = DIV_ROUND_UP(size.height, l->block_h);
   const uint64_t last_row = DIV_ROUND_UP(size.width, l->block_w) * l->block_bytes;
   return (uint64_t)(size.depthOrArrayLayers - 1) * l->rows_per_image * l->row_bytes +
          (uint64_t)(block_rows - 1) * l->row_bytes + last_row;
}

/* ---------------------------------------------------------------------- */
/* Depth and stencil
 *
 * WebGPU copies only whole subresources of these, stencil both ways and
 * depth16 both ways, depth32float only out, and depth24plus not at all.
 * What it does copy goes through a buffer of the whole subresource, which
 * a byte-wise shader repacks into or out of the application's layout; the
 * depth aspect is read out by a shader, whatever its format. */

/* Whether the region's bytes in the buffer have gaps the copy leaves be:
 * rows further apart than they are long, or images than their rows. */
static bool
region_has_gaps(const struct copy_layout *l, WGPUExtent3D size)
{
   const uint32_t row = DIV_ROUND_UP(size.width, l->block_w) * l->block_bytes;
   const uint32_t rows = DIV_ROUND_UP(size.height, l->block_h);
   return (rows > 1 && l->row_bytes > row) || (size.depthOrArrayLayers > 1 && l->rows_per_image > rows);
}

static const char repack_wgsl[] =
   "struct P { src_byte: u32, src_pitch: u32, src_image: u32, dst_byte: u32,\n"
   "           dst_pitch: u32, dst_image: u32, row_bytes: u32, rows: u32 }\n"
   "@group(0) @binding(0) var<storage, read> src: array<u32>;\n"
   "@group(0) @binding(1) var<storage, read_write> dst: array<atomic<u32>>;\n"
   "@group(0) @binding(2) var<uniform> p: P;\n"
   "@compute @workgroup_size(64)\n"
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"
   "  if (id.x >= p.row_bytes || id.y >= p.rows) { return; }\n"
   "  let s = p.src_byte + id.z * p.src_image + id.y * p.src_pitch + id.x;\n"
   "  let d = p.dst_byte + id.z * p.dst_image + id.y * p.dst_pitch + id.x;\n"
   "  let v = (src[s / 4u] >> ((s % 4u) * 8u)) & 0xffu;\n"
   "  let shift = (d % 4u) * 8u;\n"
   "  atomicAnd(&dst[d / 4u], ~(0xffu << shift));\n"
   "  atomicOr(&dst[d / 4u], v << shift);\n"
   "}\n";

static const char depth_out_wgsl[] =
   "struct P { ox: u32, oy: u32, layer: u32, level: u32, width: u32, height: u32,\n"
   "           dst_byte: u32, dst_pitch: u32, dst_image: u32, kind: u32 }\n"
   "@group(0) @binding(0) var src: texture_depth_2d_array;\n"
   "@group(0) @binding(1) var<storage, read_write> dst: array<atomic<u32>>;\n"
   "@group(0) @binding(2) var<uniform> p: P;\n"
   "@compute @workgroup_size(8, 8, 1)\n"
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"
   "  if (id.x >= p.width || id.y >= p.height) { return; }\n"
   "  let d = textureLoad(src, vec2i(i32(p.ox + id.x), i32(p.oy + id.y)), i32(p.layer + id.z), i32(p.level));\n"
   "  let at = p.dst_byte + id.z * p.dst_image + id.y * p.dst_pitch;\n"
   "  if (p.kind == 2u) {\n"
   "    let v = u32(round(clamp(d, 0.0, 1.0) * 65535.0));\n"
   "    let b = at + id.x * 2u;\n"
   "    let shift = (b % 4u) * 8u;\n"
   "    atomicAnd(&dst[b / 4u], ~(0xffffu << shift));\n"
   "    atomicOr(&dst[b / 4u], v << shift);\n"
   "  } else if (p.kind == 1u) {\n"
   /* Depth24 as the unorm depth_in made this float of, which rounding the
    * product can be one off from: the neighbour that makes it, if one. */
   "    let c = clamp(d, 0.0, 1.0);\n"
   "    var v = u32(round(c * 16777215.0));\n"
   "    if (f32(v) / 16777215.0 != c) {\n"
   "      if (v > 0u && f32(v - 1u) / 16777215.0 == c) { v = v - 1u; }\n"
   "      else if (v < 16777215u && f32(v + 1u) / 16777215.0 == c) { v = v + 1u; }\n"
   "    }\n"
   "    atomicStore(&dst[(at + id.x * 4u) / 4u], v);\n"
   "  } else {\n"
   "    atomicStore(&dst[(at + id.x * 4u) / 4u], bitcast<u32>(d));\n"
   "  }\n"
   "}\n";

/* A buffer into a depth24 or depth32 aspect, which WebGPU does not copy:
 * a triangle over the region writing each texel's frag_depth. Buffers hold
 * depth24 as the low 24 bits of 32, depth32 as a float. */
static const char depth_in_wgsl[] =
   "struct P { ox: u32, oy: u32, src_byte: u32, src_pitch: u32, kind: u32 }\n"
   "@group(0) @binding(0) var<storage, read> src: array<u32>;\n"
   "@group(0) @binding(1) var<uniform> p: P;\n"
   "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
   "  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));\n"
   "  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);\n"
   "}\n"
   "@fragment fn fs(@builtin(position) pos: vec4f) -> @builtin(frag_depth) f32 {\n"
   "  let x = u32(pos.x) - p.ox;\n"
   "  let y = u32(pos.y) - p.oy;\n"
   "  let w = src[(p.src_byte + y * p.src_pitch + x * 4u) / 4u];\n"
   "  return select(bitcast<f32>(w), f32(w & 0xffffffu) / 16777215.0, p.kind == 1u);\n"
   "}\n";

static WGPURenderPipeline
depth_in_pipeline(struct wvk_device *dev, WGPUTextureFormat format)
{
   WGPURenderPipeline pipeline = NULL;
   simple_mtx_lock(&dev->cache_lock);
   uint32_t i = 0;
   while (i < ARRAY_SIZE(dev->depth_in_pipelines) && dev->depth_in_pipelines[i].pipeline &&
          dev->depth_in_pipelines[i].format != format)
      ++i;
   /* Full: the last is made again, for this. Commands recorded with it
    * hold it still. */
   if (i == ARRAY_SIZE(dev->depth_in_pipelines)) {
      --i;
      wgpuRenderPipelineRelease(dev->depth_in_pipelines[i].pipeline);
      dev->depth_in_pipelines[i].pipeline = NULL;
   }
   if (!dev->depth_in_pipelines[i].pipeline) {
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(depth_in_wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk depth in");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      /* Depth only: the stencil stays as it is. */
      WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
      ds.format = format;
      ds.depthWriteEnabled = WGPUOptionalBool_True;
      ds.depthCompare = WGPUCompareFunction_Always;
      ds.stencilWriteMask = 0;
      WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
      fragment.module = module;
      fragment.entryPoint = wvk_sv("fs");
      WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk depth in");
      desc.vertex.module = module;
      desc.vertex.entryPoint = wvk_sv("vs");
      desc.depthStencil = &ds;
      desc.fragment = &fragment;
      dev->depth_in_pipelines[i].format = format;
      dev->depth_in_pipelines[i].pipeline = wgpuDeviceCreateRenderPipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   pipeline = dev->depth_in_pipelines[i].pipeline;
   simple_mtx_unlock(&dev->cache_lock);
   return pipeline;
}

static WGPUComputePipeline
copy_pipeline(struct wvk_device *dev, WGPUComputePipeline *slot, const char *wgsl, const char *label)
{
   simple_mtx_lock(&dev->cache_lock);
   if (!*slot) {
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
      *slot = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   simple_mtx_unlock(&dev->cache_lock);
   return *slot;
}

/* One compute dispatch, with its bind group and parameters, between
 * passes. */
static void
dispatch(struct wvk_device *dev, struct wvk_replay *replay, WGPUComputePipeline pipeline,
         const WGPUBindGroupEntry *entries, uint32_t entry_count, const void *params, size_t params_size,
         uint32_t x, uint32_t y, uint32_t z)
{
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, params_size, WGPUBufferUsage_Uniform, params);
   if (!params_buffer)
      return;
   WGPUBindGroupEntry all[4];
   memcpy(all, entries, entry_count * sizeof(*entries));
   all[entry_count] = (WGPUBindGroupEntry){
      .binding = entry_count, .buffer = params_buffer, .offset = 0, .size = align64(params_size, 16),
   };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
   bg.entryCount = entry_count + 1;
   bg.entries = all;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bg.layout);

   wvk_replay_end_passes(replay);
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderSetPipeline(pass, pipeline);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, z);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
}

/* A storage binding of [offset, offset + size) of a buffer: from the 256
 * bytes below, which is where bindings can start, and where in it that is. */
static WGPUBindGroupEntry
storage_range(uint32_t binding, WGPUBuffer buffer, uint64_t offset, uint64_t size, uint32_t *byte_out)
{
   const uint64_t base = offset & ~255ull;
   *byte_out = (uint32_t)(offset - base);
   return (WGPUBindGroupEntry){
      .binding = binding, .buffer = buffer, .offset = base, .size = align64(offset + size - base, 4),
   };
}

/* rows of row_bytes from one buffer layout to another, byte for byte. */
static void
repack(struct wvk_device *dev, struct wvk_replay *replay, WGPUBuffer src, uint64_t src_offset,
       uint32_t src_pitch, uint32_t src_image, WGPUBuffer dst, uint64_t dst_offset, uint32_t dst_pitch,
       uint32_t dst_image, uint32_t row_bytes, uint32_t rows, uint32_t images)
{
   WGPUComputePipeline pipeline = copy_pipeline(dev, &dev->repack_pipeline, repack_wgsl, "wvk repack");
   if (!pipeline || !row_bytes || !rows || !images)
      return;
   const uint64_t src_size = (uint64_t)(images - 1) * src_image + (uint64_t)(rows - 1) * src_pitch + row_bytes;
   const uint64_t dst_size = (uint64_t)(images - 1) * dst_image + (uint64_t)(rows - 1) * dst_pitch + row_bytes;
   uint32_t src_byte, dst_byte;
   const WGPUBindGroupEntry entries[] = {
      storage_range(0, src, src_offset, src_size, &src_byte),
      storage_range(1, dst, dst_offset, dst_size, &dst_byte),
   };
   const uint32_t params[8] = { src_byte, src_pitch, src_image, dst_byte, dst_pitch, dst_image, row_bytes, rows };
   dispatch(dev, replay, pipeline, entries, 2, params, sizeof(params), DIV_ROUND_UP(row_bytes, 64), rows, images);
}

/* ---------------------------------------------------------------------- */
/* 16-bit packed formats (WVK_CAP_PACKED16)
 *
 * RGBA8 textures, whose texels a shader unpacks out of buffers and packs
 * into them. Each channel of the texture is `size` bits of the texel from
 * bit `shift`, 8 bits of P's sizes and shifts each; a channel of none is
 * 0, and alpha 1. */

#define PACKED16_COMMON                                                                      \
   "struct P { byte: u32, pitch: u32, image: u32, width: u32, height: u32,\n"                  \
   "           origin_x: u32, origin_y: u32, origin_z: u32, sizes: u32, shifts: u32 }\n"       \
   "fn field(p: P, i: u32) -> vec2u {\n"                                                      \
   "  return vec2u((p.sizes >> (8u * i)) & 255u, (p.shifts >> (8u * i)) & 255u);\n"            \
   "}\n"

#define UNPACK16(texture, store)                                                             \
   PACKED16_COMMON                                                                           \
   "@group(0) @binding(0) var<storage, read> src: array<u32>;\n"                              \
   "@group(0) @binding(1) var dst: " texture ";\n"                                            \
   "@group(0) @binding(2) var<uniform> p: P;\n"                                               \
   "@compute @workgroup_size(8, 8, 1)\n"                                                      \
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"                                    \
   "  if (id.x >= p.width || id.y >= p.height) { return; }\n"                                 \
   "  let byte = p.byte + id.z * p.image + id.y * p.pitch + id.x * 2u;\n"                     \
   "  let v = (src[byte >> 2u] >> ((byte & 2u) * 8u)) & 0xffffu;\n"                            \
   "  var c = vec4f(0.0, 0.0, 0.0, 1.0);\n"                                                   \
   "  for (var i = 0u; i < 4u; i++) {\n"                                                      \
   "    let f = field(p, i);\n"                                                               \
   "    if (f.x != 0u) {\n"                                                                   \
   "      let m = (1u << f.x) - 1u;\n"                                                        \
   "      c[i] = f32((v >> f.y) & m) / f32(m);\n"                                             \
   "    }\n"                                                                                  \
   "  }\n"                                                                                    \
   "  " store "\n"                                                                            \
   "}\n"

#define PACK16(texture, load)                                                                \
   PACKED16_COMMON                                                                           \
   "@group(0) @binding(0) var src: " texture ";\n"                                            \
   "@group(0) @binding(1) var<storage, read_write> dst: array<atomic<u32>>;\n"                \
   "@group(0) @binding(2) var<uniform> p: P;\n"                                               \
   "@compute @workgroup_size(8, 8, 1)\n"                                                      \
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"                                    \
   "  if (id.x >= p.width || id.y >= p.height) { return; }\n"                                 \
   "  let c = " load ";\n"                                                                    \
   "  var v = 0u;\n"                                                                          \
   "  for (var i = 0u; i < 4u; i++) {\n"                                                      \
   "    let f = field(p, i);\n"                                                               \
   "    if (f.x != 0u) {\n"                                                                   \
   "      let m = (1u << f.x) - 1u;\n"                                                        \
   "      v |= (u32(round(clamp(c[i], 0.0, 1.0) * f32(m))) & m) << f.y;\n"                     \
   "    }\n"                                                                                  \
   "  }\n"                                                                                    \
   "  let byte = p.byte + id.z * p.image + id.y * p.pitch + id.x * 2u;\n"                     \
   "  let half = (byte & 2u) * 8u;\n"                                                         \
   "  atomicAnd(&dst[byte >> 2u], ~(0xffffu << half));\n"                                     \
   "  atomicOr(&dst[byte >> 2u], v << half);\n"                                               \
   "}\n"

static const char *const packed16_wgsl[4] = {
   UNPACK16("texture_storage_2d_array<rgba8unorm, write>",
            "textureStore(dst, vec2u(p.origin_x + id.x, p.origin_y + id.y), p.origin_z + id.z, c);"),
   UNPACK16("texture_storage_3d<rgba8unorm, write>",
            "textureStore(dst, vec3u(p.origin_x + id.x, p.origin_y + id.y, p.origin_z + id.z), c);"),
   PACK16("texture_2d_array<f32>",
          "textureLoad(src, vec2u(p.origin_x + id.x, p.origin_y + id.y), p.origin_z + id.z, 0)"),
   PACK16("texture_3d<f32>", "textureLoad(src, vec3u(p.origin_x + id.x, p.origin_y + id.y, p.origin_z + id.z), 0)"),
};

static bool
is_packed16(const struct wvk_image *image)
{
   const struct wvk_format *f = wvk_format_get(image->vk.format);
   return f && (f->caps & WVK_CAP_PACKED16);
}

/* A region of a packed image, as image_region has it, to or from a
 * buffer laid out as `l` says. */
static void
convert_packed16_texture(struct wvk_device *dev, struct wvk_replay *replay, WGPUTexture texture_of, VkFormat format,
                         bool is_3d, uint32_t layers, bool to_texture, WGPUBuffer buffer, uint64_t offset,
                         const struct copy_layout *l, const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size);

static void
convert_packed16(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image,
                 bool to_texture, WGPUBuffer buffer, uint64_t offset, const struct copy_layout *l,
                 const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size)
{
   convert_packed16_texture(dev, replay, image->texture, image->vk.format,
                            image->dimension == WGPUTextureDimension_3D, image->vk.array_layers, to_texture, buffer,
                            offset, l, tex, size);
}

/* The same of any texture of a packed format's, RGBA8. */
static void
convert_packed16_texture(struct wvk_device *dev, struct wvk_replay *replay, WGPUTexture texture_of, VkFormat format,
                         bool is_3d, uint32_t layers, bool to_texture, WGPUBuffer buffer, uint64_t offset,
                         const struct copy_layout *l, const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size)
{
   const unsigned which = (to_texture ? 0 : 2) + (is_3d ? 1 : 0);
   WGPUComputePipeline pipeline =
      copy_pipeline(dev, &dev->packed16_pipelines[which], packed16_wgsl[which], "wvk packed16");
   if (!pipeline || !size.width || !size.height || !size.depthOrArrayLayers)
      return;
   /* The texture's channels, where the format has them. */
   const struct util_format_description *desc = vk_format_description(format);
   uint32_t sizes = 0, shifts = 0;
   for (unsigned c = 0; c < 4; ++c) {
      const unsigned sw = desc->swizzle[c];
      if (sw > PIPE_SWIZZLE_W)
         continue;
      sizes |= (uint32_t)desc->channel[sw].size << (8 * c);
      shifts |= (uint32_t)desc->channel[sw].shift << (8 * c);
   }
   /* The level: every layer of it, or its depth. */
   WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   vd.dimension = is_3d ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2DArray;
   vd.baseMipLevel = tex->mipLevel;
   vd.mipLevelCount = 1;
   vd.arrayLayerCount = is_3d ? 1 : layers;
   WGPUTextureView view = wgpuTextureCreateView(texture_of, &vd);
   if (!view)
      return;
   util_dynarray_append(&replay->views, view);
   uint32_t byte;
   const WGPUBindGroupEntry range = storage_range(to_texture ? 0 : 1, buffer, offset, region_bytes(l, size), &byte);
   WGPUBindGroupEntry texture = WGPU_BIND_GROUP_ENTRY_INIT;
   texture.binding = to_texture ? 1 : 0;
   texture.textureView = view;
   const WGPUBindGroupEntry entries[2] = { to_texture ? range : texture, to_texture ? texture : range };
   const uint32_t params[12] = {
      byte, l->row_bytes, l->rows_per_image * l->row_bytes, size.width, size.height,
      tex->origin.x, tex->origin.y, tex->origin.z, sizes, shifts,
   };
   dispatch(dev, replay, pipeline, entries, 2, params, sizeof(params), DIV_ROUND_UP(size.width, 8),
            DIV_ROUND_UP(size.height, 8), size.depthOrArrayLayers);
}

/* A buffer-texture copy of a colour image: WebGPU's, or a shader's for a
 * packed format. */
static void
copy_color(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image, bool to_texture,
           WGPUBuffer buffer, uint64_t offset, const struct copy_layout *l, const WGPUTexelCopyTextureInfo *tex,
           WGPUExtent3D size)
{
   if (is_packed16(image))
      convert_packed16(dev, replay, image, to_texture, buffer, offset, l, tex, size);
   else
      copy_buffer_texture(replay, to_texture, buffer, offset, l, tex, size);
}

static bool
is_packed16_format(VkFormat format)
{
   const struct wvk_format *f = wvk_format_get(format);
   return f && (f->caps & WVK_CAP_PACKED16);
}

void
wvk_image_copy_packed(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image, uint32_t from,
                      uint32_t to, uint32_t level)
{
   uint32_t from_level, to_level;
   WGPUTexture src = wvk_image_texture(image, from, &from_level);
   WGPUTexture dst = wvk_image_texture(image, to, &to_level);
   const VkFormat src_format = wvk_image_texture_format(image, from), dst_format = wvk_image_texture_format(image, to);
   if (!src || !dst)
      return;
   const bool is_3d = image->dimension == WGPUTextureDimension_3D;
   const VkExtent3D e = vk_image_mip_level_extent(&image->vk, level);
   const WGPUExtent3D size = { e.width, image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : e.height,
                               is_3d ? e.depth : image->vk.array_layers };
   /* Their bits: two bytes a texel, as both formats have them. */
   const struct copy_layout l = { 1, 1, 2, align(size.width * 2, 256), size.height };
   WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, (uint64_t)l.row_bytes * size.height * size.depthOrArrayLayers,
                                            WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage,
                                            NULL);
   if (!temp)
      return;
   wvk_replay_end_passes(replay);
   WGPUTexelCopyTextureInfo tex = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
   tex.texture = src;
   tex.mipLevel = level - from_level;
   if (is_packed16_format(src_format))
      convert_packed16_texture(dev, replay, src, src_format, is_3d, image->vk.array_layers, false, temp, 0, &l, &tex,
                               size);
   else
      copy_buffer_texture(replay, false, temp, 0, &l, &tex, size);
   wvk_replay_end_passes(replay);
   tex.texture = dst;
   tex.mipLevel = level - to_level;
   if (is_packed16_format(dst_format))
      convert_packed16_texture(dev, replay, dst, dst_format, is_3d, image->vk.array_layers, true, temp, 0, &l, &tex,
                               size);
   else
      copy_buffer_texture(replay, true, temp, 0, &l, &tex, size);
   wvk_replay_end_passes(replay);
}

/* Where texel (x, y) of a linear image is in its memory. */
static uint64_t
linear_offset(const struct wvk_image *image, VkOffset3D o)
{
   const uint32_t bpp = util_format_get_blocksize(vk_format_to_pipe_format(image->vk.format));
   return image->offset + (uint64_t)o.y * image->pitch + (uint64_t)o.x * bpp;
}

static uint64_t
linear_bytes(const struct wvk_image *image, VkExtent3D e)
{
   const uint32_t bpp = util_format_get_blocksize(vk_format_to_pipe_format(image->vk.format));
   return (uint64_t)(e.height - 1) * image->pitch + (uint64_t)e.width * bpp;
}

static bool
is_ds_aspect(VkImageAspectFlags aspect)
{
   return aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
}

/* The whole of the subresources a region is in: what WebGPU copies. */
static void
whole_subresource(const struct wvk_image *image, const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size,
                  WGPUTexelCopyTextureInfo *whole, WGPUExtent3D *whole_size)
{
   *whole = *tex;
   whole->origin.x = 0;
   whole->origin.y = 0;
   *whole_size = (WGPUExtent3D){
      u_minify(image->vk.extent.width, tex->mipLevel),
      image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : u_minify(image->vk.extent.height, tex->mipLevel),
      size.depthOrArrayLayers,
   };
}

/* The stencil aspect, or depth16's, into a buffer of the whole
 * subresource, 256-byte rows: the temporary, and its pitch. */
static WGPUBuffer
stage_out(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image,
          const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size, uint32_t bytes, uint32_t *pitch)
{
   WGPUTexelCopyTextureInfo whole;
   WGPUExtent3D whole_size;
   whole_subresource(image, tex, size, &whole, &whole_size);
   *pitch = align(whole_size.width * bytes, 256);
   WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay,
                                            (uint64_t)*pitch * whole_size.height * whole_size.depthOrArrayLayers,
                                            WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
                                            WGPUBufferUsage_Storage, NULL);
   if (!temp)
      return NULL;
   WGPUTexelCopyBufferInfo buf = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
   buf.buffer = temp;
   buf.layout.bytesPerRow = *pitch;
   buf.layout.rowsPerImage = whole_size.height;
   wvk_replay_end_passes(replay);
   wgpuCommandEncoderCopyTextureToBuffer(replay->encoder, &whole, &buf, &whole_size);
   return temp;
}

static void
depth_out(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image,
          const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size, WGPUBuffer dst, uint64_t dst_offset,
          const struct copy_layout *l)
{
   WGPUComputePipeline pipeline = copy_pipeline(dev, &dev->depth_out_pipeline, depth_out_wgsl, "wvk depth out");
   if (!pipeline)
      return;
   WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   vd.dimension = WGPUTextureViewDimension_2DArray;
   vd.aspect = WGPUTextureAspect_DepthOnly;
   vd.baseMipLevel = tex->mipLevel;
   vd.mipLevelCount = 1;
   vd.baseArrayLayer = tex->origin.z;
   vd.arrayLayerCount = size.depthOrArrayLayers;
   WGPUTextureView view = wgpuTextureCreateView(image->texture, &vd);
   util_dynarray_append(&replay->views, view);

   const enum pipe_format depth = vk_format_to_pipe_format(vk_format_depth_only(image->vk.format));
   const uint32_t kind = depth == PIPE_FORMAT_Z16_UNORM ? 2 : depth == PIPE_FORMAT_Z32_FLOAT ? 0 : 1;
   const uint64_t bytes = (uint64_t)(size.depthOrArrayLayers - 1) * l->rows_per_image * l->row_bytes +
                          (uint64_t)(size.height - 1) * l->row_bytes + size.width * l->block_bytes;
   uint32_t dst_byte;
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .textureView = view },
      storage_range(1, dst, dst_offset, bytes, &dst_byte),
   };
   const uint32_t params[12] = {
      tex->origin.x, tex->origin.y, 0, 0, size.width, size.height,
      dst_byte, l->row_bytes, l->rows_per_image * l->row_bytes, kind,
   };
   dispatch(dev, replay, pipeline, entries, 2, params, sizeof(params), DIV_ROUND_UP(size.width, 8),
            DIV_ROUND_UP(size.height, 8), size.depthOrArrayLayers);
}

/* A pass a layer, drawing over the region only. */
static void
depth_in(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image,
         const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size, WGPUBuffer src, uint64_t src_offset,
         const struct copy_layout *l)
{
   WGPURenderPipeline pipeline = depth_in_pipeline(dev, image->format);
   if (!pipeline || !size.width || !size.height)
      return;
   const enum pipe_format depth = vk_format_to_pipe_format(vk_format_depth_only(image->vk.format));
   const bool stencil = vk_format_has_stencil(image->vk.format);
   const uint32_t image_bytes = l->rows_per_image * l->row_bytes;
   const uint64_t bytes = (uint64_t)(size.depthOrArrayLayers - 1) * image_bytes +
                          (uint64_t)(size.height - 1) * l->row_bytes + size.width * 4;
   uint32_t src_byte;
   const WGPUBindGroupEntry source = storage_range(0, src, src_offset, bytes, &src_byte);
   WGPUBindGroupLayout bgl = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
   const uint32_t width = u_minify(image->vk.extent.width, tex->mipLevel);
   const uint32_t height = u_minify(image->vk.extent.height, tex->mipLevel);

   wvk_replay_end_passes(replay);
   for (uint32_t z = 0; z < size.depthOrArrayLayers; ++z) {
      const uint32_t params[8] = {
         tex->origin.x, tex->origin.y, src_byte + z * image_bytes, l->row_bytes,
         depth == PIPE_FORMAT_Z32_FLOAT ? 0 : 1,
      };
      WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform,
                                                        params);
      if (!params_buffer)
         break;
      const WGPUBindGroupEntry entries[] = {
         source,
         { .binding = 1, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
      };
      WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bg.layout = bgl;
      bg.entryCount = ARRAY_SIZE(entries);
      bg.entries = entries;
      WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);

      WGPUTextureView view = wvk_image_attachment_view(image, WVK_IMAGE_TEXTURE, image->format, WGPUTextureAspect_All,
                                                       tex->mipLevel, tex->origin.z + z);
      util_dynarray_append(&replay->views, view);
      WGPURenderPassDepthStencilAttachment ds = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
      ds.view = view;
      ds.depthLoadOp = WGPULoadOp_Load;
      ds.depthStoreOp = WGPUStoreOp_Store;
      if (stencil) {
         ds.stencilLoadOp = WGPULoadOp_Load;
         ds.stencilStoreOp = WGPUStoreOp_Store;
      }
      WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
      pd.label = wvk_sv("wvk depth in");
      pd.depthStencilAttachment = &ds;
      WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &pd);
      wgpuRenderPassEncoderSetPipeline(pass, pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, NULL);
      wgpuRenderPassEncoderSetViewport(pass, 0, 0, (float)width, (float)height, 0.0f, 1.0f);
      wgpuRenderPassEncoderSetScissorRect(pass, tex->origin.x, tex->origin.y, size.width, size.height);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
      wgpuRenderPassEncoderEnd(pass);
      wgpuRenderPassEncoderRelease(pass);
      wgpuBindGroupRelease(group);
   }
   wgpuBindGroupLayoutRelease(bgl);
}

/* Compressed texels into a shadow of a size not whole blocks: a triangle
 * over a layer or slice, which the instance is, loading each texel. */
static const char shadow_wgsl[] =
   "struct V { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) z: u32 }\n"
   "@vertex fn vs(@builtin(vertex_index) i: u32, @builtin(instance_index) z: u32) -> V {\n"
   "  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));\n"
   "  return V(vec4f(uv * 2.0 - 1.0, 0.0, 1.0), z);\n"
   "}\n"
   "@fragment fn fs(v: V) -> @location(0) vec4f {\n"
   "  return LOAD;\n"
   "}\n";

static WGPURenderPipeline
shadow_pipeline(struct wvk_device *dev, WGPUTextureFormat format, bool is_3d)
{
   const uint32_t which = format == WGPUTextureFormat_RGBA8Unorm ? 0
                          : format == WGPUTextureFormat_RGBA8UnormSrgb ? 1 : 2;
   simple_mtx_lock(&dev->cache_lock);
   WGPURenderPipeline *slot = &dev->shadow_pipelines[which][is_3d];
   if (!*slot) {
      const char *decl = is_3d ? "@group(0) @binding(0) var src: texture_3d<f32>;\n"
                               : "@group(0) @binding(0) var src: texture_2d_array<f32>;\n";
      const char *load = is_3d ? "textureLoad(src, vec3i(vec2i(v.pos.xy), i32(v.z)), 0)"
                               : "textureLoad(src, vec2i(v.pos.xy), i32(v.z), 0)";
      char *body = strdup(shadow_wgsl);
      char *at = strstr(body, "LOAD");
      *at = '\0';
      char *code = ralloc_asprintf(NULL, "%s%s%s%s", decl, body, load, at + 4);
      free(body);
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(code);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk shadow");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      ralloc_free(code);
      WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
      target.format = format;
      WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
      fragment.module = module;
      fragment.entryPoint = wvk_sv("fs");
      fragment.targetCount = 1;
      fragment.targets = &target;
      WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk shadow");
      desc.vertex.module = module;
      desc.vertex.entryPoint = wvk_sv("vs");
      desc.fragment = &fragment;
      *slot = wgpuDeviceCreateRenderPipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   WGPURenderPipeline pipeline = *slot;
   simple_mtx_unlock(&dev->cache_lock);
   return pipeline;
}

void
wvk_image_update_shadow(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image, uint32_t level)
{
   const bool is_3d = image->dimension == WGPUTextureDimension_3D;
   WGPURenderPipeline pipeline = shadow_pipeline(dev, image->shadow_format, is_3d);
   if (!pipeline)
      return;
   const uint32_t base_layer = 0;
   const uint32_t layer_count = is_3d ? u_minify(image->vk.extent.depth, level) : image->vk.array_layers;

   WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   vd.dimension = is_3d ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2DArray;
   vd.baseMipLevel = level;
   vd.mipLevelCount = 1;
   vd.baseArrayLayer = 0;
   vd.arrayLayerCount = is_3d ? 1 : image->vk.array_layers;
   WGPUTextureView src = wgpuTextureCreateView(image->texture, &vd);
   util_dynarray_append(&replay->views, src);
   WGPUBindGroupLayout bgl = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
   const WGPUBindGroupEntry entry = { .binding = 0, .textureView = src };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = bgl;
   bg.entryCount = 1;
   bg.entries = &entry;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bgl);

   wvk_replay_end_passes(replay);
   for (uint32_t z = base_layer; z < base_layer + layer_count; ++z) {
      WGPUTextureViewDescriptor ad = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      ad.dimension = is_3d ? WGPUTextureViewDimension_3D : WGPUTextureViewDimension_2D;
      ad.baseMipLevel = level;
      ad.mipLevelCount = 1;
      ad.baseArrayLayer = is_3d ? 0 : z;
      ad.arrayLayerCount = 1;
      WGPUTextureView view = wgpuTextureCreateView(image->shadow, &ad);
      util_dynarray_append(&replay->views, view);
      WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
      color.view = view;
      color.depthSlice = is_3d ? z : WGPU_DEPTH_SLICE_UNDEFINED;
      color.loadOp = WGPULoadOp_Clear;
      color.storeOp = WGPUStoreOp_Store;
      WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
      pd.label = wvk_sv("wvk shadow");
      pd.colorAttachmentCount = 1;
      pd.colorAttachments = &color;
      WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &pd);
      wgpuRenderPassEncoderSetPipeline(pass, pipeline);
      wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, NULL);
      wgpuRenderPassEncoderDraw(pass, 3, 1, 0, z);
      wgpuRenderPassEncoderEnd(pass);
      wgpuRenderPassEncoderRelease(pass);
   }
   wgpuBindGroupRelease(group);
}

/* Each sample of a multisampled colour texture into another. */
static const char ms_copy_wgsl[] =
   "@group(0) @binding(0) var src: texture_multisampled_2d<T>;\n"
   "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
   "  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));\n"
   "  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);\n"
   "}\n"
   "@fragment fn fs(@builtin(position) pos: vec4f, @builtin(sample_index) s: u32) -> @location(0) vec4<T> {\n"
   "  return textureLoad(src, vec2i(pos.xy), i32(s));\n"
   "}\n";

static WGPURenderPipeline
ms_copy_pipeline(struct wvk_device *dev, WGPUTextureFormat format, VkFormat vk_format, uint32_t samples)
{
   WGPURenderPipeline pipeline = NULL;
   simple_mtx_lock(&dev->cache_lock);
   uint32_t i = 0;
   while (i < ARRAY_SIZE(dev->ms_copy_pipelines) && dev->ms_copy_pipelines[i].pipeline &&
          (dev->ms_copy_pipelines[i].format != format || dev->ms_copy_pipelines[i].samples != samples))
      ++i;
   /* Full: the last is made again, for this. Commands recorded with it
    * hold it still. */
   if (i == ARRAY_SIZE(dev->ms_copy_pipelines)) {
      --i;
      wgpuRenderPipelineRelease(dev->ms_copy_pipelines[i].pipeline);
      dev->ms_copy_pipelines[i].pipeline = NULL;
   }
   if (!dev->ms_copy_pipelines[i].pipeline) {
      const char *type = vk_format_is_sint(vk_format) ? "i32" : vk_format_is_int(vk_format) ? "u32" : "f32";
      char *code = ralloc_strdup(NULL, "");
      for (const char *c = ms_copy_wgsl; *c; ++c) {
         if (*c == 'T' && c[1] == '>')
            ralloc_strcat(&code, type);
         else
            ralloc_strncat(&code, c, 1);
      }
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(code);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk ms copy");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      ralloc_free(code);
      WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
      target.format = format;
      WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
      fragment.module = module;
      fragment.entryPoint = wvk_sv("fs");
      fragment.targetCount = 1;
      fragment.targets = &target;
      WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk ms copy");
      desc.vertex.module = module;
      desc.vertex.entryPoint = wvk_sv("vs");
      desc.fragment = &fragment;
      desc.multisample.count = samples;
      dev->ms_copy_pipelines[i].format = format;
      dev->ms_copy_pipelines[i].samples = samples;
      dev->ms_copy_pipelines[i].pipeline = wgpuDeviceCreateRenderPipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   pipeline = dev->ms_copy_pipelines[i].pipeline;
   simple_mtx_unlock(&dev->cache_lock);
   return pipeline;
}

WGPUTexture
wvk_image_snapshot_ms(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image)
{
   if (!image->texture || (image->vk.aspects & ~VK_IMAGE_ASPECT_COLOR_BIT))
      return NULL;
   WGPURenderPipeline pipeline = ms_copy_pipeline(dev, image->format, image->vk.format, image->vk.samples);
   if (!pipeline)
      return NULL;
   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("VkImage copy");
   desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment;
   desc.size = (WGPUExtent3D){ wgpuTextureGetWidth(image->texture), wgpuTextureGetHeight(image->texture), 1 };
   desc.format = image->format;
   desc.sampleCount = image->vk.samples;
   WGPUTexture copy = wgpuDeviceCreateTexture(dev->wgpu, &desc);
   if (!copy)
      return NULL;

   WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   vd.dimension = WGPUTextureViewDimension_2D;
   vd.baseArrayLayer = 0;
   vd.arrayLayerCount = 1;
   WGPUTextureView src = wgpuTextureCreateView(image->texture, &vd);
   util_dynarray_append(&replay->views, src);
   WGPUTextureView dst = wgpuTextureCreateView(copy, &vd);
   util_dynarray_append(&replay->views, dst);
   WGPUBindGroupLayout bgl = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
   const WGPUBindGroupEntry entry = { .binding = 0, .textureView = src };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = bgl;
   bg.entryCount = 1;
   bg.entries = &entry;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bgl);

   wvk_replay_end_passes(replay);
   WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
   color.view = dst;
   color.loadOp = WGPULoadOp_Clear;
   color.storeOp = WGPUStoreOp_Store;
   WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   pd.label = wvk_sv("wvk ms copy");
   pd.colorAttachmentCount = 1;
   pd.colorAttachments = &color;
   WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &pd);
   wgpuRenderPassEncoderSetPipeline(pass, pipeline);
   wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
   wgpuRenderPassEncoderEnd(pass);
   wgpuRenderPassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
   return copy;
}

/* A texel of a format as the words of its bits, and back: WGSL of the
 * value `v` a multisampled load gives, and of the words `b`. sRGB formats
 * are viewed as their linear twins, whose bits they are. */
static const struct ms_bits {
   WGPUTextureFormat format;
   char type; /* the sample type: f32, u32 or i32 */
   const char *to_bits, *from_bits;
} ms_bits[] = {
   { WGPUTextureFormat_R8Unorm, 'f', "vec4u(pack4x8unorm(v) & 0xffu, 0u, 0u, 0u)",
     "vec4f(unpack4x8unorm(b.x).x, 0.0, 0.0, 1.0)" },
   { WGPUTextureFormat_R8Uint, 'u', "vec4u(v.x & 0xffu, 0u, 0u, 0u)", "vec4u(b.x & 0xffu, 0u, 0u, 1u)" },
   { WGPUTextureFormat_R8Sint, 'i', "vec4u(bitcast<u32>(v.x) & 0xffu, 0u, 0u, 0u)",
     "vec4i(extractBits(bitcast<i32>(b.x), 0u, 8u), 0, 0, 1)" },
   { WGPUTextureFormat_RG8Unorm, 'f', "vec4u(pack4x8unorm(v) & 0xffffu, 0u, 0u, 0u)",
     "vec4f(unpack4x8unorm(b.x).xy, 0.0, 1.0)" },
   { WGPUTextureFormat_RG8Uint, 'u', "vec4u((v.x & 0xffu) | ((v.y & 0xffu) << 8u), 0u, 0u, 0u)",
     "vec4u(b.x & 0xffu, (b.x >> 8u) & 0xffu, 0u, 1u)" },
   { WGPUTextureFormat_RG8Sint, 'i', "vec4u((bitcast<u32>(v.x) & 0xffu) | ((bitcast<u32>(v.y) & 0xffu) << 8u), 0u, 0u, 0u)",
     "vec4i(extractBits(bitcast<i32>(b.x), 0u, 8u), extractBits(bitcast<i32>(b.x), 8u, 8u), 0, 1)" },
   { WGPUTextureFormat_R16Float, 'f', "vec4u(pack2x16float(v.xy) & 0xffffu, 0u, 0u, 0u)",
     "vec4f(unpack2x16float(b.x).x, 0.0, 0.0, 1.0)" },
   { WGPUTextureFormat_R16Uint, 'u', "vec4u(v.x & 0xffffu, 0u, 0u, 0u)", "vec4u(b.x & 0xffffu, 0u, 0u, 1u)" },
   { WGPUTextureFormat_R16Sint, 'i', "vec4u(bitcast<u32>(v.x) & 0xffffu, 0u, 0u, 0u)",
     "vec4i(extractBits(bitcast<i32>(b.x), 0u, 16u), 0, 0, 1)" },
   { WGPUTextureFormat_RGBA8Unorm, 'f', "vec4u(pack4x8unorm(v), 0u, 0u, 0u)", "unpack4x8unorm(b.x)" },
   { WGPUTextureFormat_BGRA8Unorm, 'f', "vec4u(pack4x8unorm(v.bgra), 0u, 0u, 0u)", "unpack4x8unorm(b.x).bgra" },
   { WGPUTextureFormat_RGBA8Snorm, 'f', "vec4u(pack4x8snorm(v), 0u, 0u, 0u)", "unpack4x8snorm(b.x)" },
   { WGPUTextureFormat_RGBA8Uint, 'u', "vec4u(dot(v & vec4u(0xffu), vec4u(1u, 0x100u, 0x10000u, 0x1000000u)), 0u, 0u, 0u)",
     "vec4u(b.x & 0xffu, (b.x >> 8u) & 0xffu, (b.x >> 16u) & 0xffu, b.x >> 24u)" },
   { WGPUTextureFormat_RGBA8Sint, 'i', "vec4u(dot(bitcast<vec4u>(v) & vec4u(0xffu), vec4u(1u, 0x100u, 0x10000u, 0x1000000u)), 0u, 0u, 0u)",
     "vec4i(extractBits(bitcast<i32>(b.x), 0u, 8u), extractBits(bitcast<i32>(b.x), 8u, 8u),"
     " extractBits(bitcast<i32>(b.x), 16u, 8u), extractBits(bitcast<i32>(b.x), 24u, 8u))" },
   { WGPUTextureFormat_RG16Float, 'f', "vec4u(pack2x16float(v.xy), 0u, 0u, 0u)",
     "vec4f(unpack2x16float(b.x), 0.0, 1.0)" },
   { WGPUTextureFormat_RG16Unorm, 'f', "vec4u(pack2x16unorm(v.xy), 0u, 0u, 0u)",
     "vec4f(unpack2x16unorm(b.x), 0.0, 1.0)" },
   { WGPUTextureFormat_RG16Snorm, 'f', "vec4u(pack2x16snorm(v.xy), 0u, 0u, 0u)",
     "vec4f(unpack2x16snorm(b.x), 0.0, 1.0)" },
   { WGPUTextureFormat_RG16Uint, 'u', "vec4u((v.x & 0xffffu) | (v.y << 16u), 0u, 0u, 0u)",
     "vec4u(b.x & 0xffffu, b.x >> 16u, 0u, 1u)" },
   { WGPUTextureFormat_RG16Sint, 'i', "vec4u((bitcast<u32>(v.x) & 0xffffu) | (bitcast<u32>(v.y) << 16u), 0u, 0u, 0u)",
     "vec4i(extractBits(bitcast<i32>(b.x), 0u, 16u), extractBits(bitcast<i32>(b.x), 16u, 16u), 0, 1)" },
   { WGPUTextureFormat_R32Float, 'f', "vec4u(bitcast<u32>(v.x), 0u, 0u, 0u)",
     "vec4f(bitcast<f32>(b.x), 0.0, 0.0, 1.0)" },
   { WGPUTextureFormat_R32Uint, 'u', "vec4u(v.x, 0u, 0u, 0u)", "vec4u(b.x, 0u, 0u, 1u)" },
   { WGPUTextureFormat_R32Sint, 'i', "vec4u(bitcast<u32>(v.x), 0u, 0u, 0u)", "vec4i(bitcast<i32>(b.x), 0, 0, 1)" },
   { WGPUTextureFormat_RGB10A2Unorm, 'f',
     "vec4u(u32(round(saturate(v.x) * 1023.0)) | (u32(round(saturate(v.y) * 1023.0)) << 10u) |"
     " (u32(round(saturate(v.z) * 1023.0)) << 20u) | (u32(round(saturate(v.w) * 3.0)) << 30u), 0u, 0u, 0u)",
     "vec4f(f32(b.x & 0x3ffu) / 1023.0, f32((b.x >> 10u) & 0x3ffu) / 1023.0,"
     " f32((b.x >> 20u) & 0x3ffu) / 1023.0, f32(b.x >> 30u) / 3.0)" },
   { WGPUTextureFormat_RGB10A2Uint, 'u',
     "vec4u((v.x & 0x3ffu) | ((v.y & 0x3ffu) << 10u) | ((v.z & 0x3ffu) << 20u) | (v.w << 30u), 0u, 0u, 0u)",
     "vec4u(b.x & 0x3ffu, (b.x >> 10u) & 0x3ffu, (b.x >> 20u) & 0x3ffu, b.x >> 30u)" },
   { WGPUTextureFormat_RGBA16Float, 'f', "vec4u(pack2x16float(v.xy), pack2x16float(v.zw), 0u, 0u)",
     "vec4f(unpack2x16float(b.x), unpack2x16float(b.y))" },
   { WGPUTextureFormat_RGBA16Unorm, 'f', "vec4u(pack2x16unorm(v.xy), pack2x16unorm(v.zw), 0u, 0u)",
     "vec4f(unpack2x16unorm(b.x), unpack2x16unorm(b.y))" },
   { WGPUTextureFormat_RGBA16Snorm, 'f', "vec4u(pack2x16snorm(v.xy), pack2x16snorm(v.zw), 0u, 0u)",
     "vec4f(unpack2x16snorm(b.x), unpack2x16snorm(b.y))" },
   { WGPUTextureFormat_RGBA16Uint, 'u',
     "vec4u((v.x & 0xffffu) | (v.y << 16u), (v.z & 0xffffu) | (v.w << 16u), 0u, 0u)",
     "vec4u(b.x & 0xffffu, b.x >> 16u, b.y & 0xffffu, b.y >> 16u)" },
   { WGPUTextureFormat_RGBA16Sint, 'i',
     "vec4u((bitcast<u32>(v.x) & 0xffffu) | (bitcast<u32>(v.y) << 16u),"
     " (bitcast<u32>(v.z) & 0xffffu) | (bitcast<u32>(v.w) << 16u), 0u, 0u)",
     "vec4i(extractBits(bitcast<i32>(b.x), 0u, 16u), extractBits(bitcast<i32>(b.x), 16u, 16u),"
     " extractBits(bitcast<i32>(b.y), 0u, 16u), extractBits(bitcast<i32>(b.y), 16u, 16u))" },
   { WGPUTextureFormat_RG32Float, 'f', "vec4u(bitcast<vec2u>(v.xy), 0u, 0u)",
     "vec4f(bitcast<vec2f>(b.xy), 0.0, 1.0)" },
   { WGPUTextureFormat_RG32Uint, 'u', "vec4u(v.xy, 0u, 0u)", "vec4u(b.xy, 0u, 1u)" },
   { WGPUTextureFormat_RG32Sint, 'i', "vec4u(bitcast<vec2u>(v.xy), 0u, 0u)", "vec4i(bitcast<vec2i>(b.xy), 0, 1)" },
   { WGPUTextureFormat_RGBA32Float, 'f', "bitcast<vec4u>(v)", "bitcast<vec4f>(b)" },
   { WGPUTextureFormat_RGBA32Uint, 'u', "v", "b" },
   { WGPUTextureFormat_RGBA32Sint, 'i', "bitcast<vec4u>(v)", "bitcast<vec4i>(b)" },
};

static WGPUTextureFormat
linear_format(WGPUTextureFormat format)
{
   return format == WGPUTextureFormat_RGBA8UnormSrgb   ? WGPUTextureFormat_RGBA8Unorm
          : format == WGPUTextureFormat_BGRA8UnormSrgb ? WGPUTextureFormat_BGRA8Unorm
                                                       : format;
}

static const struct ms_bits *
find_ms_bits(WGPUTextureFormat format)
{
   format = linear_format(format);
   for (uint32_t i = 0; i < ARRAY_SIZE(ms_bits); ++i) {
      if (ms_bits[i].format == format)
         return &ms_bits[i];
   }
   return NULL;
}

bool
wvk_ms_bits_format(WGPUTextureFormat format)
{
   return find_ms_bits(format) != NULL;
}

static const char *
ms_bits_type(char type)
{
   return type == 'u' ? "u32" : type == 'i' ? "i32" : "f32";
}

static WGPURenderPipeline
ms_bits_pipeline(struct wvk_device *dev, WGPUTextureFormat from, WGPUTextureFormat to, uint32_t samples)
{
   const struct ms_bits *a = find_ms_bits(from), *b = find_ms_bits(to);
   if (!a || !b)
      return NULL;
   WGPURenderPipeline pipeline = NULL;
   simple_mtx_lock(&dev->cache_lock);
   uint32_t i = 0;
   while (i < ARRAY_SIZE(dev->ms_alias_pipelines) && dev->ms_alias_pipelines[i].pipeline &&
          (dev->ms_alias_pipelines[i].from != a->format || dev->ms_alias_pipelines[i].to != b->format ||
           dev->ms_alias_pipelines[i].samples != samples))
      ++i;
   /* Full: the last is made again, for this. Commands recorded with it
    * hold it still. */
   if (i == ARRAY_SIZE(dev->ms_alias_pipelines)) {
      --i;
      wgpuRenderPipelineRelease(dev->ms_alias_pipelines[i].pipeline);
      dev->ms_alias_pipelines[i].pipeline = NULL;
   }
   if (!dev->ms_alias_pipelines[i].pipeline) {
      char *code = ralloc_asprintf(
         NULL,
         "@group(0) @binding(0) var src: texture_multisampled_2d<%s>;\n"
         "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
         "  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));\n"
         "  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);\n"
         "}\n"
         "fn to_bits(v: vec4<%s>) -> vec4u { return %s; }\n"
         "fn from_bits(b: vec4u) -> vec4<%s> { return %s; }\n"
         "@fragment fn fs(@builtin(position) pos: vec4f, @builtin(sample_index) s: u32) -> @location(0) vec4<%s> {\n"
         "  return from_bits(to_bits(textureLoad(src, vec2i(pos.xy), i32(s))));\n"
         "}\n",
         ms_bits_type(a->type), ms_bits_type(a->type), a->to_bits, ms_bits_type(b->type), b->from_bits,
         ms_bits_type(b->type));
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(code);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk ms bits");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      ralloc_free(code);
      /* Unfilterable: R32Float and the rest are loaded, not filtered. */
      WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
      entry.binding = 0;
      entry.visibility = WGPUShaderStage_Fragment;
      entry.texture.sampleType = a->type == 'u'   ? WGPUTextureSampleType_Uint
                                 : a->type == 'i' ? WGPUTextureSampleType_Sint
                                                  : WGPUTextureSampleType_UnfilterableFloat;
      entry.texture.viewDimension = WGPUTextureViewDimension_2D;
      entry.texture.multisampled = true;
      WGPUBindGroupLayoutDescriptor bgld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
      bgld.entryCount = 1;
      bgld.entries = &entry;
      WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(dev->wgpu, &bgld);
      WGPUPipelineLayoutDescriptor pld = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
      pld.bindGroupLayoutCount = 1;
      pld.bindGroupLayouts = &bgl;
      WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(dev->wgpu, &pld);
      wgpuBindGroupLayoutRelease(bgl);
      WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
      target.format = b->format;
      WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
      fragment.module = module;
      fragment.entryPoint = wvk_sv("fs");
      fragment.targetCount = 1;
      fragment.targets = &target;
      WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk ms bits");
      desc.layout = layout;
      desc.vertex.module = module;
      desc.vertex.entryPoint = wvk_sv("vs");
      desc.fragment = &fragment;
      desc.multisample.count = samples;
      dev->ms_alias_pipelines[i].from = a->format;
      dev->ms_alias_pipelines[i].to = b->format;
      dev->ms_alias_pipelines[i].samples = samples;
      dev->ms_alias_pipelines[i].pipeline = wgpuDeviceCreateRenderPipeline(dev->wgpu, &desc);
      wgpuPipelineLayoutRelease(layout);
      wgpuShaderModuleRelease(module);
   }
   pipeline = dev->ms_alias_pipelines[i].pipeline;
   simple_mtx_unlock(&dev->cache_lock);
   return pipeline;
}

static void ms_bits_pass(struct wvk_device *dev, struct wvk_replay *replay, WGPURenderPipeline pipeline,
                         WGPUTexture src, WGPUTextureFormat src_format, WGPUTexture dst,
                         WGPUTextureFormat dst_format);

void
wvk_image_copy_ms_bits(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image, uint32_t from,
                       uint32_t to)
{
   uint32_t first;
   WGPUTexture src = wvk_image_texture(image, from, &first);
   WGPUTexture dst = wvk_image_texture(image, to, &first);
   if (!src || !dst)
      return;
   const WGPUTextureFormat src_format = linear_format(wgpuTextureGetFormat(src));
   const WGPUTextureFormat dst_format = linear_format(wgpuTextureGetFormat(dst));
   WGPURenderPipeline pipeline = ms_bits_pipeline(dev, src_format, dst_format, image->vk.samples);
   if (!pipeline)
      return;
   wvk_replay_end_passes(replay);
   /* A multisampled array is a texture a layer. */
   for (uint32_t layer = 0; layer < image->vk.array_layers; ++layer) {
      uint32_t unused;
      src = wvk_image_texture_layer(image, from, layer, &unused);
      dst = wvk_image_texture_layer(image, to, layer, &unused);
      if (src && dst)
         ms_bits_pass(dev, replay, pipeline, src, src_format, dst, dst_format);
   }
}

static void
ms_bits_pass(struct wvk_device *dev, struct wvk_replay *replay, WGPURenderPipeline pipeline, WGPUTexture src,
             WGPUTextureFormat src_format, WGPUTexture dst, WGPUTextureFormat dst_format)
{

   WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   vd.dimension = WGPUTextureViewDimension_2D;
   vd.arrayLayerCount = 1;
   vd.mipLevelCount = 1;
   vd.format = src_format;
   WGPUTextureView src_view = wgpuTextureCreateView(src, &vd);
   util_dynarray_append(&replay->views, src_view);
   vd.format = dst_format;
   WGPUTextureView dst_view = wgpuTextureCreateView(dst, &vd);
   util_dynarray_append(&replay->views, dst_view);
   WGPUBindGroupLayout bgl = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
   const WGPUBindGroupEntry entry = { .binding = 0, .textureView = src_view };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = bgl;
   bg.entryCount = 1;
   bg.entries = &entry;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bgl);

   WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
   color.view = dst_view;
   color.loadOp = WGPULoadOp_Clear;
   color.storeOp = WGPUStoreOp_Store;
   WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   pd.label = wvk_sv("wvk ms bits");
   pd.colorAttachmentCount = 1;
   pd.colorAttachments = &color;
   WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &pd);
   wgpuRenderPassEncoderSetPipeline(pass, pipeline);
   wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
   wgpuRenderPassEncoderEnd(pass);
   wgpuRenderPassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
}

/* Depth into R32Float, or stencil into R8Uint: a pass a level and layer,
 * loading it. Only depth16 and depth32 copy out of a texture, and none
 * into one; stencil does, but not reliably out of combined formats. */
static const char ds_copy_wgsl[] =
   "struct V { @builtin(position) pos: vec4f, @location(0) @interpolate(flat) z: u32 }\n"
   "@vertex fn vs(@builtin(vertex_index) i: u32, @builtin(instance_index) z: u32) -> V {\n"
   "  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));\n"
   "  return V(vec4f(uv * 2.0 - 1.0, 0.0, 1.0), z);\n"
   "}\n";

static WGPURenderPipeline
ds_copy_pipeline(struct wvk_device *dev, bool stencil)
{
   simple_mtx_lock(&dev->cache_lock);
   WGPURenderPipeline *slot = &dev->ds_copy_pipelines[stencil];
   if (!*slot) {
      const char *fs = stencil
         ? "@group(0) @binding(0) var src: texture_2d_array<u32>;\n"
           "@fragment fn fs(v: V) -> @location(0) vec4u {\n"
           "  return textureLoad(src, vec2i(v.pos.xy), i32(v.z), 0);\n"
           "}\n"
         : "@group(0) @binding(0) var src: texture_depth_2d_array;\n"
           "@fragment fn fs(v: V) -> @location(0) vec4f {\n"
           "  return vec4f(textureLoad(src, vec2i(v.pos.xy), i32(v.z), 0), 0.0, 0.0, 1.0);\n"
           "}\n";
      char *code = ralloc_asprintf(NULL, "%s%s", ds_copy_wgsl, fs);
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(code);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk depth-stencil copy");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      ralloc_free(code);
      WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
      target.format = stencil ? WGPUTextureFormat_R8Uint : WGPUTextureFormat_R32Float;
      WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
      fragment.module = module;
      fragment.entryPoint = wvk_sv("fs");
      fragment.targetCount = 1;
      fragment.targets = &target;
      WGPURenderPipelineDescriptor pd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
      pd.label = wvk_sv("wvk depth-stencil copy");
      pd.vertex.module = module;
      pd.vertex.entryPoint = wvk_sv("vs");
      pd.fragment = &fragment;
      *slot = wgpuDeviceCreateRenderPipeline(dev->wgpu, &pd);
      wgpuShaderModuleRelease(module);
   }
   WGPURenderPipeline pipeline = *slot;
   simple_mtx_unlock(&dev->cache_lock);
   return pipeline;
}

/* Each sample of a 4-sample image's depth into R32Float, or stencil into
 * R8Uint: its first layer, which shaders see of a multisampled array. */
static WGPURenderPipeline
ds_copy_ms_pipeline(struct wvk_device *dev, bool stencil)
{
   simple_mtx_lock(&dev->cache_lock);
   WGPURenderPipeline *slot = &dev->ds_copy_ms_pipelines[stencil];
   if (!*slot) {
      const char *code = stencil
         ? "@group(0) @binding(0) var src: texture_multisampled_2d<u32>;\n"
           "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
           "  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));\n"
           "  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);\n"
           "}\n"
           "@fragment fn fs(@builtin(position) pos: vec4f, @builtin(sample_index) s: u32) -> @location(0) vec4u {\n"
           "  return textureLoad(src, vec2i(pos.xy), i32(s));\n"
           "}\n"
         : "@group(0) @binding(0) var src: texture_depth_multisampled_2d;\n"
           "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {\n"
           "  let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));\n"
           "  return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);\n"
           "}\n"
           "@fragment fn fs(@builtin(position) pos: vec4f, @builtin(sample_index) s: u32) -> @location(0) vec4f {\n"
           "  return vec4f(textureLoad(src, vec2i(pos.xy), i32(s)), 0.0, 0.0, 1.0);\n"
           "}\n";
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(code);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk depth-stencil ms copy");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
      target.format = stencil ? WGPUTextureFormat_R8Uint : WGPUTextureFormat_R32Float;
      WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
      fragment.module = module;
      fragment.entryPoint = wvk_sv("fs");
      fragment.targetCount = 1;
      fragment.targets = &target;
      WGPURenderPipelineDescriptor pd = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
      pd.label = wvk_sv("wvk depth-stencil ms copy");
      pd.vertex.module = module;
      pd.vertex.entryPoint = wvk_sv("vs");
      pd.fragment = &fragment;
      pd.multisample.count = 4;
      *slot = wgpuDeviceCreateRenderPipeline(dev->wgpu, &pd);
      wgpuShaderModuleRelease(module);
   }
   WGPURenderPipeline pipeline = *slot;
   simple_mtx_unlock(&dev->cache_lock);
   return pipeline;
}

static WGPUTexture
snapshot_ds_ms(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image, bool stencil)
{
   if (image->vk.samples != 4)
      return NULL;
   WGPURenderPipeline pipeline = ds_copy_ms_pipeline(dev, stencil);
   if (!pipeline)
      return NULL;
   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("VkImage copy");
   desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment;
   desc.size = (WGPUExtent3D){ wgpuTextureGetWidth(image->texture), wgpuTextureGetHeight(image->texture), 1 };
   desc.format = stencil ? WGPUTextureFormat_R8Uint : WGPUTextureFormat_R32Float;
   desc.sampleCount = 4;
   WGPUTexture copy = wgpuDeviceCreateTexture(dev->wgpu, &desc);
   if (!copy)
      return NULL;
   wvk_replay_end_passes(replay);
   WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   vd.dimension = WGPUTextureViewDimension_2D;
   vd.arrayLayerCount = 1;
   vd.mipLevelCount = 1;
   vd.aspect = stencil ? WGPUTextureAspect_StencilOnly : WGPUTextureAspect_DepthOnly;
   WGPUTextureView src = wgpuTextureCreateView(image->texture, &vd);
   util_dynarray_append(&replay->views, src);
   vd.aspect = WGPUTextureAspect_All;
   WGPUTextureView dst = wgpuTextureCreateView(copy, &vd);
   util_dynarray_append(&replay->views, dst);
   WGPUBindGroupLayout bgl = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
   const WGPUBindGroupEntry entry = { .binding = 0, .textureView = src };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = bgl;
   bg.entryCount = 1;
   bg.entries = &entry;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bgl);
   WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
   color.view = dst;
   color.loadOp = WGPULoadOp_Clear;
   color.storeOp = WGPUStoreOp_Store;
   WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   pd.label = wvk_sv("wvk depth-stencil ms copy");
   pd.colorAttachmentCount = 1;
   pd.colorAttachments = &color;
   WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &pd);
   wgpuRenderPassEncoderSetPipeline(pass, pipeline);
   wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
   wgpuRenderPassEncoderEnd(pass);
   wgpuRenderPassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
   return copy;
}

WGPUTexture
wvk_image_snapshot_ds(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image, bool stencil)
{
   if (image->texture && image->vk.samples > 1)
      return snapshot_ds_ms(dev, replay, image, stencil);
   if (!image->texture || image->dimension == WGPUTextureDimension_3D)
      return NULL;
   WGPURenderPipeline pipeline = ds_copy_pipeline(dev, stencil);
   if (!pipeline)
      return NULL;
   const uint32_t layers = image->vk.array_layers, levels = image->vk.mip_levels;
   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("VkImage copy");
   desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment;
   desc.size = (WGPUExtent3D){ wgpuTextureGetWidth(image->texture), wgpuTextureGetHeight(image->texture), layers };
   desc.format = stencil ? WGPUTextureFormat_R8Uint : WGPUTextureFormat_R32Float;
   desc.mipLevelCount = levels;
   WGPUTexture copy = wgpuDeviceCreateTexture(dev->wgpu, &desc);
   if (!copy)
      return NULL;
   wvk_replay_end_passes(replay);

   WGPUBindGroupLayout bgl = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
   for (uint32_t l = 0; l < levels; ++l) {
      WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      vd.dimension = WGPUTextureViewDimension_2DArray;
      vd.baseMipLevel = l;
      vd.mipLevelCount = 1;
      vd.baseArrayLayer = 0;
      vd.arrayLayerCount = layers;
      vd.aspect = stencil ? WGPUTextureAspect_StencilOnly : WGPUTextureAspect_DepthOnly;
      WGPUTextureView src = wgpuTextureCreateView(image->texture, &vd);
      util_dynarray_append(&replay->views, src);
      const WGPUBindGroupEntry entry = { .binding = 0, .textureView = src };
      WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bg.layout = bgl;
      bg.entryCount = 1;
      bg.entries = &entry;
      WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
      for (uint32_t z = 0; z < layers; ++z) {
         WGPUTextureViewDescriptor ad = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
         ad.dimension = WGPUTextureViewDimension_2D;
         ad.baseMipLevel = l;
         ad.mipLevelCount = 1;
         ad.baseArrayLayer = z;
         ad.arrayLayerCount = 1;
         WGPUTextureView view = wgpuTextureCreateView(copy, &ad);
         util_dynarray_append(&replay->views, view);
         WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
         color.view = view;
         color.loadOp = WGPULoadOp_Clear;
         color.storeOp = WGPUStoreOp_Store;
         WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
         pd.label = wvk_sv("wvk depth-stencil copy");
         pd.colorAttachmentCount = 1;
         pd.colorAttachments = &color;
         WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &pd);
         wgpuRenderPassEncoderSetPipeline(pass, pipeline);
         wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, NULL);
         wgpuRenderPassEncoderDraw(pass, 3, 1, 0, z);
         wgpuRenderPassEncoderEnd(pass);
         wgpuRenderPassEncoderRelease(pass);
      }
      wgpuBindGroupRelease(group);
   }
   wgpuBindGroupLayoutRelease(bgl);
   return copy;
}

bool
wvk_image_copy_depth_layer(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image,
                           uint32_t level, uint32_t layer, WGPUTextureView dst)
{
   if (!image->texture || image->vk.samples > 1 || image->dimension == WGPUTextureDimension_3D)
      return false;
   WGPURenderPipeline pipeline = ds_copy_pipeline(dev, false);
   if (!pipeline)
      return false;
   wvk_replay_end_passes(replay);
   WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   vd.dimension = WGPUTextureViewDimension_2DArray;
   vd.baseMipLevel = level;
   vd.mipLevelCount = 1;
   vd.baseArrayLayer = layer;
   vd.arrayLayerCount = 1;
   vd.aspect = WGPUTextureAspect_DepthOnly;
   WGPUTextureView src = wgpuTextureCreateView(image->texture, &vd);
   util_dynarray_append(&replay->views, src);
   WGPUBindGroupLayout bgl = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
   const WGPUBindGroupEntry entry = { .binding = 0, .textureView = src };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = bgl;
   bg.entryCount = 1;
   bg.entries = &entry;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bgl);
   WGPURenderPassColorAttachment color = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
   color.view = dst;
   color.loadOp = WGPULoadOp_Clear;
   color.storeOp = WGPUStoreOp_Store;
   WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   pd.label = wvk_sv("wvk depth copy");
   pd.colorAttachmentCount = 1;
   pd.colorAttachments = &color;
   WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &pd);
   wgpuRenderPassEncoderSetPipeline(pass, pipeline);
   wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
   wgpuRenderPassEncoderEnd(pass);
   wgpuRenderPassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
   return true;
}

/* A depth or stencil aspect to a buffer. */
static void
copy_ds_to_buffer(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image,
                  VkImageAspectFlags aspect, const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size,
                  WGPUBuffer dst, uint64_t dst_offset, const struct copy_layout *l)
{
   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      depth_out(dev, replay, image, tex, size, dst, dst_offset, l);
      return;
   }
   uint32_t pitch;
   WGPUBuffer temp = stage_out(dev, replay, image, tex, size, 1, &pitch);
   if (!temp)
      return;
   const uint32_t height = image->vk.image_type == VK_IMAGE_TYPE_1D ? 1
                           : u_minify(image->vk.extent.height, tex->mipLevel);
   repack(dev, replay, temp, (uint64_t)tex->origin.y * pitch + tex->origin.x, pitch, pitch * height, dst,
          dst_offset, l->row_bytes, l->rows_per_image * l->row_bytes, size.width, size.height,
          size.depthOrArrayLayers);
}

/* A buffer to a depth or stencil aspect: the whole subresource out, the
 * region patched in, and the whole back. */
static void
copy_buffer_to_ds(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image,
                  VkImageAspectFlags aspect, const WGPUTexelCopyTextureInfo *tex, WGPUExtent3D size,
                  WGPUBuffer src, uint64_t src_offset, const struct copy_layout *l)
{
   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT &&
       vk_format_to_pipe_format(vk_format_depth_only(image->vk.format)) != PIPE_FORMAT_Z16_UNORM) {
      depth_in(dev, replay, image, tex, size, src, src_offset, l);
      return;
   }
   const uint32_t bytes = aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? 2 : 1;
   uint32_t pitch;
   WGPUBuffer temp = stage_out(dev, replay, image, tex, size, bytes, &pitch);
   if (!temp)
      return;
   WGPUTexelCopyTextureInfo whole;
   WGPUExtent3D whole_size;
   whole_subresource(image, tex, size, &whole, &whole_size);
   repack(dev, replay, src, src_offset, l->row_bytes, l->rows_per_image * l->row_bytes, temp,
          (uint64_t)tex->origin.y * pitch + (uint64_t)tex->origin.x * bytes, pitch, pitch * whole_size.height,
          size.width * bytes, size.height, size.depthOrArrayLayers);
   WGPUTexelCopyBufferInfo buf = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
   buf.buffer = temp;
   buf.layout.bytesPerRow = pitch;
   buf.layout.rowsPerImage = whole_size.height;
   wgpuCommandEncoderCopyBufferToTexture(replay->encoder, &buf, &whole, &whole_size);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyBufferToImage2(VkCommandBuffer commandBuffer, const VkCopyBufferToImageInfo2 *pCopyBufferToImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_buffer, src, pCopyBufferToImageInfo->srcBuffer);
   VK_FROM_HANDLE(wvk_image, dst, pCopyBufferToImageInfo->dstImage);
   struct wvk_replay *replay = cmd->replay;

   /* A multi-planar image's planes, a region at a time. */
   if (dst->plane_count) {
      for (uint32_t i = 0; i < pCopyBufferToImageInfo->regionCount; ++i) {
         VkBufferImageCopy2 r = pCopyBufferToImageInfo->pRegions[i];
         VkCopyBufferToImageInfo2 one = *pCopyBufferToImageInfo;
         one.dstImage = wvk_image_to_handle(wvk_image_for_aspect(dst, &r.imageSubresource.aspectMask));
         one.regionCount = 1;
         one.pRegions = &r;
         wvk_CmdCopyBufferToImage2(commandBuffer, &one);
      }
      return;
   }
   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pCopyBufferToImageInfo->regionCount; ++i) {
      const VkBufferImageCopy2 *r = &pCopyBufferToImageInfo->pRegions[i];
      wvk_image_use(wvk_cmd_buffer_device(cmd), replay, dst, r->imageSubresource.mipLevel, 1, true);
      WGPUTexelCopyTextureInfo tex;
      WGPUExtent3D size;
      image_region(dst, &r->imageSubresource, r->imageOffset, r->imageExtent, &tex, &size);
      const struct copy_layout l = buffer_layout(dst->vk.format, r->imageSubresource.aspectMask,
                                                 r->bufferRowLength, r->bufferImageHeight, r->imageExtent);
      const uint64_t off = buffer_offset(src, r->bufferOffset);
      wvk_replay_read(replay, src->mem, off, region_bytes(&l, size));
      if (dst->linear) {
         const uint32_t bpp = util_format_get_blocksize(vk_format_to_pipe_format(dst->vk.format));
         const uint64_t at = linear_offset(dst, r->imageOffset);
         wvk_replay_read(replay, dst->mem, at, linear_bytes(dst, r->imageExtent));
         repack(wvk_cmd_buffer_device(cmd), replay, buffer_wgpu(src), off, l.row_bytes,
                l.rows_per_image * l.row_bytes, dst->mem->buffer, at, dst->pitch, 0,
                r->imageExtent.width * bpp, r->imageExtent.height, 1);
         wvk_replay_write(replay, dst->mem, at, linear_bytes(dst, r->imageExtent));
      } else if (is_ds_aspect(r->imageSubresource.aspectMask))
         copy_buffer_to_ds(wvk_cmd_buffer_device(cmd), replay, dst, r->imageSubresource.aspectMask, &tex,
                           size, buffer_wgpu(src), off, &l);
      else
         copy_color(wvk_cmd_buffer_device(cmd), replay, dst, true, buffer_wgpu(src), off, &l, &tex, size);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2 *pCopyImageToBufferInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pCopyImageToBufferInfo->srcImage);
   VK_FROM_HANDLE(wvk_buffer, dst, pCopyImageToBufferInfo->dstBuffer);
   struct wvk_replay *replay = cmd->replay;

   if (src->plane_count) {
      for (uint32_t i = 0; i < pCopyImageToBufferInfo->regionCount; ++i) {
         VkBufferImageCopy2 r = pCopyImageToBufferInfo->pRegions[i];
         VkCopyImageToBufferInfo2 one = *pCopyImageToBufferInfo;
         one.srcImage = wvk_image_to_handle(wvk_image_for_aspect(src, &r.imageSubresource.aspectMask));
         one.regionCount = 1;
         one.pRegions = &r;
         wvk_CmdCopyImageToBuffer2(commandBuffer, &one);
      }
      return;
   }
   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pCopyImageToBufferInfo->regionCount; ++i) {
      const VkBufferImageCopy2 *r = &pCopyImageToBufferInfo->pRegions[i];
      wvk_image_use(wvk_cmd_buffer_device(cmd), replay, src, r->imageSubresource.mipLevel, 1, false);
      WGPUTexelCopyTextureInfo tex;
      WGPUExtent3D size;
      image_region(src, &r->imageSubresource, r->imageOffset, r->imageExtent, &tex, &size);
      const struct copy_layout l = buffer_layout(src->vk.format, r->imageSubresource.aspectMask,
                                                 r->bufferRowLength, r->bufferImageHeight, r->imageExtent);
      const uint64_t off = buffer_offset(dst, r->bufferOffset);
      /* What is read back is the whole range: where the copy leaves gaps,
       * the buffer has to have what the host has there first. */
      if (region_has_gaps(&l, size))
         wvk_replay_read(replay, dst->mem, off, region_bytes(&l, size));
      wvk_replay_write(replay, dst->mem, off, region_bytes(&l, size));
      if (src->linear) {
         const uint32_t bpp = util_format_get_blocksize(vk_format_to_pipe_format(src->vk.format));
         const uint64_t at = linear_offset(src, r->imageOffset);
         wvk_replay_read(replay, src->mem, at, linear_bytes(src, r->imageExtent));
         repack(wvk_cmd_buffer_device(cmd), replay, src->mem->buffer, at, src->pitch, 0, buffer_wgpu(dst),
                off, l.row_bytes, l.rows_per_image * l.row_bytes, r->imageExtent.width * bpp,
                r->imageExtent.height, 1);
      } else if (is_ds_aspect(r->imageSubresource.aspectMask))
         copy_ds_to_buffer(wvk_cmd_buffer_device(cmd), replay, src, r->imageSubresource.aspectMask, &tex,
                           size, buffer_wgpu(dst), off, &l);
      else
         copy_color(wvk_cmd_buffer_device(cmd), replay, src, false, buffer_wgpu(dst), off, &l, &tex, size);
   }
}

/* A copy with a linear image on one side or both: a texture-buffer copy
 * with the linear image's pitch, which is aligned for it, or a repack
 * between two of them. */
static void
copy_image_linear(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *src,
                  struct wvk_image *dst, const VkImageCopy2 *r)
{
   const uint32_t bpp = util_format_get_blocksize(vk_format_to_pipe_format(src->vk.format));
   if (src->linear && dst->linear) {
      const uint64_t from = linear_offset(src, r->srcOffset), to = linear_offset(dst, r->dstOffset);
      wvk_replay_read(replay, src->mem, from, linear_bytes(src, r->extent));
      /* What is between the region's rows comes back too: as it is. */
      wvk_replay_read(replay, dst->mem, to, linear_bytes(dst, r->extent));
      repack(dev, replay, src->mem->buffer, from, src->pitch, 0, dst->mem->buffer, to, dst->pitch, 0,
             r->extent.width * bpp, r->extent.height, 1);
      wvk_replay_write(replay, dst->mem, to, linear_bytes(dst, r->extent));
      return;
   }
   struct wvk_image *linear = src->linear ? src : dst;
   struct wvk_image *tiled = src->linear ? dst : src;
   WGPUTexelCopyTextureInfo tex;
   WGPUExtent3D size;
   image_region(tiled, src->linear ? &r->dstSubresource : &r->srcSubresource,
                src->linear ? r->dstOffset : r->srcOffset, r->extent, &tex, &size);
   const VkOffset3D at = src->linear ? r->srcOffset : r->dstOffset;
   WGPUTexelCopyBufferInfo buf = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
   buf.buffer = linear->mem->buffer;
   buf.layout.offset = linear_offset(linear, at);
   buf.layout.bytesPerRow = linear->pitch;
   buf.layout.rowsPerImage = size.height;
   if (is_packed16(tiled)) {
      const struct copy_layout l = { 1, 1, 2, linear->pitch, size.height };
      if (src->linear)
         wvk_replay_read(replay, linear->mem, buf.layout.offset, linear_bytes(linear, r->extent));
      if (!src->linear)
         wvk_replay_read(replay, linear->mem, buf.layout.offset, linear_bytes(linear, r->extent));
      convert_packed16(dev, replay, tiled, src->linear, linear->mem->buffer, buf.layout.offset, &l, &tex, size);
      if (!src->linear)
         wvk_replay_write(replay, linear->mem, buf.layout.offset, linear_bytes(linear, r->extent));
      return;
   }
   /* Into it, what is between the region's rows comes back too, as it is:
    * the GPU's copy has to have it. */
   wvk_replay_read(replay, linear->mem, buf.layout.offset, linear_bytes(linear, r->extent));
   if (src->linear) {
      wgpuCommandEncoderCopyBufferToTexture(replay->encoder, &buf, &tex, &size);
   } else {
      wgpuCommandEncoderCopyTextureToBuffer(replay->encoder, &tex, &buf, &size);
      wvk_replay_write(replay, linear->mem, buf.layout.offset, linear_bytes(linear, r->extent));
   }
}

/* Whether WebGPU copies a depth-stencil region itself: only whole
 * subresources, every aspect of them, between textures of one format. */
static bool
whole_ds_copy(const struct wvk_image *src, const struct wvk_image *dst, const VkImageCopy2 *r)
{
   if (src->format != dst->format || r->srcSubresource.aspectMask != vk_format_aspects(src->vk.format))
      return false;
   const struct wvk_image *images[] = { src, dst };
   const VkOffset3D offsets[] = { r->srcOffset, r->dstOffset };
   const uint32_t levels[] = { r->srcSubresource.mipLevel, r->dstSubresource.mipLevel };
   for (uint32_t i = 0; i < 2; ++i) {
      if (offsets[i].x || offsets[i].y || r->extent.width != u_minify(images[i]->vk.extent.width, levels[i]) ||
          r->extent.height != u_minify(images[i]->vk.extent.height, levels[i]))
         return false;
   }
   return true;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyImage2(VkCommandBuffer commandBuffer, const VkCopyImageInfo2 *pCopyImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pCopyImageInfo->srcImage);
   VK_FROM_HANDLE(wvk_image, dst, pCopyImageInfo->dstImage);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;

   if (src->plane_count || dst->plane_count) {
      for (uint32_t i = 0; i < pCopyImageInfo->regionCount; ++i) {
         VkImageCopy2 r = pCopyImageInfo->pRegions[i];
         VkCopyImageInfo2 one = *pCopyImageInfo;
         one.srcImage = wvk_image_to_handle(wvk_image_for_aspect(src, &r.srcSubresource.aspectMask));
         one.dstImage = wvk_image_to_handle(wvk_image_for_aspect(dst, &r.dstSubresource.aspectMask));
         one.regionCount = 1;
         one.pRegions = &r;
         wvk_CmdCopyImage2(commandBuffer, &one);
      }
      return;
   }
   wvk_replay_end_passes(replay);
   /* WebGPU copies between textures of one format, give or take sRGB;
    * Vulkan between any of one size, which goes through a buffer. */
   const bool same = (src->format == dst->format ||
                      util_format_linear(vk_format_to_pipe_format(src->vk.format)) ==
                         util_format_linear(vk_format_to_pipe_format(dst->vk.format))) &&
                     /* Packed formats' textures are RGBA8 whatever their bits. */
                     ((!is_packed16(src) && !is_packed16(dst)) || src->vk.format == dst->vk.format);
   for (uint32_t i = 0; i < pCopyImageInfo->regionCount; ++i) {
      const VkImageCopy2 *r = &pCopyImageInfo->pRegions[i];
      wvk_image_use(dev, replay, src, r->srcSubresource.mipLevel, 1, false);
      wvk_image_use(dev, replay, dst, r->dstSubresource.mipLevel, 1, true);
      if (src->linear || dst->linear) {
         copy_image_linear(dev, replay, src, dst, r);
         continue;
      }
      WGPUTexelCopyTextureInfo s, d;
      WGPUExtent3D size, dsize;
      image_region(src, &r->srcSubresource, r->srcOffset, r->extent, &s, &size);
      /* A compressed image's extent is in its own texels. */
      const enum pipe_format sf = vk_format_to_pipe_format(src->vk.format);
      const enum pipe_format df = vk_format_to_pipe_format(dst->vk.format);
      VkExtent3D dext = r->extent;
      dext.width = DIV_ROUND_UP(dext.width, util_format_get_blockwidth(sf)) * util_format_get_blockwidth(df);
      dext.height = DIV_ROUND_UP(dext.height, util_format_get_blockheight(sf)) * util_format_get_blockheight(df);
      image_region(dst, &r->dstSubresource, r->dstOffset, dext, &d, &dsize);
      /* Between a depth or stencil aspect and a colour image of its texels'
       * size (VK_KHR_maintenance8): through a buffer as the aspect's copies
       * to and from buffers have it. */
      if (is_ds_aspect(r->srcSubresource.aspectMask) != is_ds_aspect(r->dstSubresource.aspectMask)) {
         const bool from_ds = is_ds_aspect(r->srcSubresource.aspectMask);
         const VkImageAspectFlags aspect = from_ds ? r->srcSubresource.aspectMask : r->dstSubresource.aspectMask;
         struct copy_layout l = buffer_layout((from_ds ? src : dst)->vk.format, aspect, 0, 0, r->extent);
         l.row_bytes = align(l.row_bytes, 256);
         const uint64_t bytes = (uint64_t)size.depthOrArrayLayers * l.rows_per_image * l.row_bytes;
         WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, align64(bytes, 4),
                                                  WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
                                                     WGPUBufferUsage_Storage,
                                                  NULL);
         if (!temp)
            continue;
         WGPUTexelCopyTextureInfo sa = s, da = d;
         if (from_ds) {
            sa.aspect = wvk_aspect(aspect);
            copy_ds_to_buffer(dev, replay, src, aspect, &sa, size, temp, 0, &l);
            copy_color(dev, replay, dst, true, temp, 0, &l, &d, dsize);
         } else {
            copy_color(dev, replay, src, false, temp, 0, &l, &s, size);
            da.aspect = wvk_aspect(aspect);
            copy_buffer_to_ds(dev, replay, dst, aspect, &da, dsize, temp, 0, &l);
         }
         continue;
      }
      /* Depth and stencil otherwise through a buffer, an aspect at a
       * time, as a buffer holds them. */
      if (is_ds_aspect(r->srcSubresource.aspectMask) && !whole_ds_copy(src, dst, r)) {
         u_foreach_bit(bit, r->srcSubresource.aspectMask) {
            const VkImageAspectFlags aspect = 1u << bit;
            const struct copy_layout l = buffer_layout(src->vk.format, aspect, 0, 0, r->extent);
            const uint64_t bytes = (uint64_t)size.depthOrArrayLayers * l.rows_per_image * l.row_bytes;
            WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, align64(bytes, 4),
                                                     WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
                                                     WGPUBufferUsage_Storage, NULL);
            if (!temp)
               break;
            WGPUTexelCopyTextureInfo sa = s, da = d;
            sa.aspect = da.aspect = wvk_aspect(aspect);
            copy_ds_to_buffer(dev, replay, src, aspect, &sa, size, temp, 0, &l);
            copy_buffer_to_ds(dev, replay, dst, aspect, &da, size, temp, 0, &l);
         }
         continue;
      }
      if (same && (src->layer_textures || dst->layer_textures)) {
         /* A multisampled array's layers are textures of their own. */
         for (uint32_t l = 0; l < size.depthOrArrayLayers; ++l) {
            WGPUTexelCopyTextureInfo sl = s, dl = d;
            sl.texture = wvk_image_layer_texture(src, s.origin.z + l, &sl.origin.z);
            dl.texture = wvk_image_layer_texture(dst, d.origin.z + l, &dl.origin.z);
            WGPUExtent3D one = size;
            one.depthOrArrayLayers = 1;
            wgpuCommandEncoderCopyTextureToTexture(replay->encoder, &sl, &dl, &one);
         }
         continue;
      }
      if (same) {
         wgpuCommandEncoderCopyTextureToTexture(replay->encoder, &s, &d, &size);
         continue;
      }
      struct copy_layout l = buffer_layout(src->vk.format, r->srcSubresource.aspectMask, 0, 0, r->extent);
      l.row_bytes = align(l.row_bytes, 256);
      const uint64_t bytes = (uint64_t)size.depthOrArrayLayers * l.rows_per_image * l.row_bytes;
      WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, bytes,
                                               WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst |
                                               WGPUBufferUsage_Storage, NULL);
      copy_color(dev, replay, src, false, temp, 0, &l, &s, size);
      struct copy_layout dl = l;
      dl.block_w = util_format_get_blockwidth(df);
      dl.block_h = util_format_get_blockheight(df);
      copy_color(dev, replay, dst, true, temp, 0, &dl, &d, dsize);
   }
}

/* ---------------------------------------------------------------------- */
/* Clears and resolves: render passes that only load or clear */

static void
clear_pass(struct wvk_replay *replay, struct wvk_image *image, uint32_t level, uint32_t layer,
           const VkClearColorValue *color, const VkClearDepthStencilValue *ds, VkImageAspectFlags aspects)
{
   WGPUTextureView view = wvk_image_attachment_view(image, WVK_IMAGE_TEXTURE, image->format, WGPUTextureAspect_All, level, layer);
   util_dynarray_append(&replay->views, view);

   WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
   WGPURenderPassColorAttachment c = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
   WGPURenderPassDepthStencilAttachment d = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
   if (color) {
      c.view = view;
      if (image->dimension == WGPUTextureDimension_3D)
         c.depthSlice = layer;
      c.loadOp = WGPULoadOp_Clear;
      c.storeOp = WGPUStoreOp_Store;
      c.clearValue = wvk_clear_color(image->vk.format, color);
      desc.colorAttachmentCount = 1;
      desc.colorAttachments = &c;
   } else {
      d.view = view;
      if (vk_format_has_depth(image->vk.format)) {
         d.depthLoadOp = (aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? WGPULoadOp_Clear : WGPULoadOp_Load;
         d.depthStoreOp = WGPUStoreOp_Store;
         d.depthClearValue = ds->depth;
      } else if (image->format == WGPUTextureFormat_Depth24PlusStencil8) {
         /* A stencil format's depth aspect, which is nobody's. */
         d.depthReadOnly = true;
      }
      if (vk_format_has_stencil(image->vk.format)) {
         d.stencilLoadOp = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? WGPULoadOp_Clear : WGPULoadOp_Load;
         d.stencilStoreOp = WGPUStoreOp_Store;
         d.stencilClearValue = ds->stencil;
      }
      desc.depthStencilAttachment = &d;
   }
   WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &desc);
   wgpuRenderPassEncoderEnd(pass);
   wgpuRenderPassEncoderRelease(pass);
}

/* A colour clear no pass can do: a linear image, which is its memory, or
 * a format WebGPU does not render to. The value packed as the format has
 * it, repeated over each subresource by copies. */
static void
clear_by_fill(struct wvk_cmd_buffer *cmd, struct wvk_image *image, const VkClearColorValue *color,
              uint32_t range_count, const VkImageSubresourceRange *ranges)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   const enum pipe_format format = vk_format_to_pipe_format(image->vk.format);
   const uint32_t bpp = util_format_get_blocksize(format);
   if (util_format_is_compressed(format) || bpp > 16) {
      mesa_logw_once("wvk: clearing %s is not supported", util_format_short_name(format));
      return;
   }
   uint8_t texel[16] = { 0 };
   util_format_pack_rgba(format, texel, color, 1);

   if (image->linear) {
      /* One level, one layer: a row of the value, into every row. */
      const uint32_t row = image->vk.extent.width * bpp;
      uint8_t *pattern = malloc(row);
      if (!pattern)
         return;
      for (uint32_t x = 0; x < image->vk.extent.width; ++x)
         memcpy(pattern + x * bpp, texel, bpp);
      WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, row, WGPUBufferUsage_Storage, pattern);
      free(pattern);
      if (!temp)
         return;
      repack(dev, replay, temp, 0, 0, 0, image->mem->buffer, image->offset, image->pitch, 0, row,
             image->vk.extent.height, 1);
      wvk_replay_write(replay, image->mem, image->offset, linear_bytes(image, image->vk.extent));
      return;
   }

   for (uint32_t r = 0; r < range_count; ++r) {
      const VkImageSubresourceRange *range = &ranges[r];
      const uint32_t levels = vk_image_subresource_level_count(&image->vk, range);
      for (uint32_t m = 0; m < levels; ++m) {
         const uint32_t level = range->baseMipLevel + m;
         const uint32_t w = u_minify(image->vk.extent.width, level);
         const uint32_t h = image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : u_minify(image->vk.extent.height, level);
         const uint32_t layers = image->vk.image_type == VK_IMAGE_TYPE_3D
                                    ? u_minify(image->vk.extent.depth, level)
                                    : vk_image_subresource_layer_count(&image->vk, range);
         const uint32_t base = image->vk.image_type == VK_IMAGE_TYPE_3D ? 0 : range->baseArrayLayer;
         const struct copy_layout l = {
            .block_w = 1, .block_h = 1, .block_bytes = bpp, .row_bytes = align(w * bpp, 256), .rows_per_image = h,
         };
         /* Rows of the value, a megabyte's at most, copied to each band
          * of rows in turn: not the whole level's, which was all of it on
          * the host twice over. */
         const uint32_t rows = MIN2(h, MAX2(1024 * 1024 / l.row_bytes, 1));
         uint8_t *pattern = calloc(rows, l.row_bytes);
         if (!pattern)
            return;
         for (uint32_t x = 0; x < w; ++x)
            memcpy(pattern + x * bpp, texel, bpp);
         for (uint32_t y = 1; y < rows; ++y)
            memcpy(pattern + (size_t)y * l.row_bytes, pattern, w * bpp);
         WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, (uint64_t)rows * l.row_bytes, WGPUBufferUsage_CopySrc,
                                                  pattern);
         free(pattern);
         if (!temp)
            return;
         for (uint32_t layer = 0; layer < layers; ++layer) {
            for (uint32_t y = 0; y < h; y += rows) {
               WGPUTexelCopyTextureInfo tex = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
               tex.texture = image->texture;
               tex.mipLevel = level;
               tex.origin.y = y;
               tex.origin.z = base + layer;
               copy_buffer_texture(replay, true, temp, 0, &l, &tex, (WGPUExtent3D){ w, MIN2(rows, h - y), 1 });
            }
         }
      }
   }
}

static void
clear_ranges(struct wvk_cmd_buffer *cmd, struct wvk_image *image, const VkClearColorValue *color,
             const VkClearDepthStencilValue *ds, uint32_t range_count, const VkImageSubresourceRange *ranges)
{
   struct wvk_replay *replay = cmd->replay;
   wvk_replay_end_passes(replay);
   for (uint32_t r = 0; r < range_count; ++r)
      wvk_image_use(wvk_cmd_buffer_device(cmd), replay, image, ranges[r].baseMipLevel,
                    vk_image_subresource_level_count(&image->vk, &ranges[r]), true);
   /* What a load op does not clear right: a draw where there are samples,
    * else copies. */
   if (color && !wvk_format_clears_by_load(image->vk.format) && image->vk.samples > 1) {
      wvk_cmd_meta_clear_color_image(cmd, image, color, range_count, ranges);
      return;
   }
   if (image->linear || !(image->usage & WGPUTextureUsage_RenderAttachment) ||
       (color && !wvk_format_clears_by_load(image->vk.format))) {
      if (color)
         clear_by_fill(cmd, image, color, range_count, ranges);
      else
         mesa_logw_once("wvk: clearing a depth-stencil format WebGPU does not render to");
      return;
   }
   for (uint32_t r = 0; r < range_count; ++r) {
      const VkImageSubresourceRange *range = &ranges[r];
      const uint32_t levels = vk_image_subresource_level_count(&image->vk, range);
      for (uint32_t m = 0; m < levels; ++m) {
         const uint32_t level = range->baseMipLevel + m;
         const uint32_t layers = image->vk.image_type == VK_IMAGE_TYPE_3D
                                    ? u_minify(image->vk.extent.depth, level)
                                    : vk_image_subresource_layer_count(&image->vk, range);
         const uint32_t base = image->vk.image_type == VK_IMAGE_TYPE_3D ? 0 : range->baseArrayLayer;
         for (uint32_t l = 0; l < layers; ++l)
            clear_pass(replay, image, level, base + l, color, ds, range->aspectMask);
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdClearColorImage(VkCommandBuffer commandBuffer, VkImage _image, VkImageLayout imageLayout,
                       const VkClearColorValue *pColor, uint32_t rangeCount,
                       const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, image, _image);
   clear_ranges(cmd, image, pColor, NULL, rangeCount, pRanges);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdClearDepthStencilImage(VkCommandBuffer commandBuffer, VkImage _image, VkImageLayout imageLayout,
                              const VkClearDepthStencilValue *pDepthStencil, uint32_t rangeCount,
                              const VkImageSubresourceRange *pRanges)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, image, _image);
   clear_ranges(cmd, image, NULL, pDepthStencil, rangeCount, pRanges);
}

/* A resolve is a render pass that loads the multisampled image and
 * resolves into the other -- of whole subresources of a format it can
 * average, which is all WebGPU resolves, and of sRGB ones their linear
 * values. The rest are vk_meta's draws, and those of other modes, or of
 * sRGB's encoded values (VkResolveImageModeInfoKHR). */
VKAPI_ATTR void VKAPI_CALL
wvk_CmdResolveImage2(VkCommandBuffer commandBuffer, const VkResolveImageInfo2 *pResolveImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pResolveImageInfo->srcImage);
   VK_FROM_HANDLE(wvk_image, dst, pResolveImageInfo->dstImage);
   struct wvk_replay *replay = cmd->replay;
   const VkResolveImageModeInfoKHR *mode = vk_find_struct_const(pResolveImageInfo->pNext, RESOLVE_IMAGE_MODE_INFO_KHR);
   const bool averaged = !mode || (mode->resolveMode == VK_RESOLVE_MODE_AVERAGE_BIT &&
                                   !((mode->flags & VK_RESOLVE_IMAGE_SKIP_TRANSFER_FUNCTION_BIT_KHR) &&
                                     vk_format_is_srgb(dst->vk.format)));

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pResolveImageInfo->regionCount; ++i) {
      const VkImageResolve2 *r = &pResolveImageInfo->pRegions[i];
      const uint32_t layers = r->srcSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS
                                 ? src->vk.array_layers - r->srcSubresource.baseArrayLayer
                                 : r->srcSubresource.layerCount;
      const VkExtent3D full = vk_image_mip_level_extent(&dst->vk, r->dstSubresource.mipLevel);
      const struct wvk_physical_device *pdev = wvk_device_physical(wvk_cmd_buffer_device(cmd));
      const VkExtent3D src_full = vk_image_mip_level_extent(&src->vk, r->srcSubresource.mipLevel);
      if (!averaged || dst->vk.image_type == VK_IMAGE_TYPE_3D || r->srcOffset.x || r->srcOffset.y ||
          r->dstOffset.x || r->dstOffset.y ||
          r->extent.width != full.width || r->extent.height != full.height || src->format != dst->format ||
          src_full.width != full.width || src_full.height != full.height ||
          !(wvk_format_caps(pdev, dst->vk.format) & WVK_CAP_RESOLVE)) {
         /* A layer at a time: a multisampled array is a texture for each
          * layer, which is all a shader can read of it. */
         for (uint32_t l = 0; l < layers; ++l) {
            VkImageResolve2 layer = *r;
            layer.srcSubresource.baseArrayLayer += l;
            layer.srcSubresource.layerCount = 1;
            layer.dstSubresource.baseArrayLayer += l;
            layer.dstSubresource.layerCount = 1;
            VkResolveImageInfo2 one = *pResolveImageInfo;
            one.regionCount = 1;
            one.pRegions = &layer;
            wvk_cmd_meta_resolve_image(cmd, &one);
         }
         continue;
      }
      wvk_image_use(wvk_cmd_buffer_device(cmd), replay, dst, r->dstSubresource.mipLevel, 1, true);
      for (uint32_t l = 0; l < layers; ++l) {
         WGPUTextureView s = wvk_image_attachment_view(src, WVK_IMAGE_TEXTURE, src->format, WGPUTextureAspect_All,
                                                       r->srcSubresource.mipLevel,
                                                       r->srcSubresource.baseArrayLayer + l);
         WGPUTextureView d = wvk_image_attachment_view(dst, WVK_IMAGE_TEXTURE, dst->format, WGPUTextureAspect_All,
                                                       r->dstSubresource.mipLevel,
                                                       r->dstSubresource.baseArrayLayer + l);
         util_dynarray_append(&replay->views, s);
         util_dynarray_append(&replay->views, d);
         WGPURenderPassColorAttachment c = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
         c.view = s;
         c.resolveTarget = d;
         c.loadOp = WGPULoadOp_Load;
         c.storeOp = WGPUStoreOp_Store;
         WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
         desc.colorAttachmentCount = 1;
         desc.colorAttachments = &c;
         WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(replay->encoder, &desc);
         wgpuRenderPassEncoderEnd(pass);
         wgpuRenderPassEncoderRelease(pass);
      }
   }
}
