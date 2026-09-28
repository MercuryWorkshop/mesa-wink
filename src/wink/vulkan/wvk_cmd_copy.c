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
#include "wvk_image.h"

#include "vk_format.h"
#include "util/format/u_format.h"

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
         /* WebGPU copies whole words. */
         mesa_logw("wvk: vkCmdCopyBuffer of %" PRIu64 " bytes at %" PRIu64 " -> %" PRIu64
                   " is not word-aligned; copying the aligned part", r->size, so, dof);
         const uint64_t size = r->size & ~3ull;
         if (!size || ((so | dof) & 3))
            continue;
         wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, buffer_wgpu(src), so, buffer_wgpu(dst), dof, size);
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
   /* A pattern needs a source to copy from. */
   uint32_t *pattern = malloc(size);
   if (!pattern)
      return;
   for (uint64_t i = 0; i < size / 4; ++i)
      pattern[i] = data;
   WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, size, WGPUBufferUsage_CopySrc, pattern);
   free(pattern);
   wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, temp, 0, buffer_wgpu(dst), off, size);
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
   "  } else {\n"
   "    let v = select(bitcast<u32>(d), u32(round(clamp(d, 0.0, 1.0) * 16777215.0)), p.kind == 1u);\n"
   "    atomicStore(&dst[(at + id.x * 4u) / 4u], v);\n"
   "  }\n"
   "}\n";

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
   const enum pipe_format depth = vk_format_to_pipe_format(vk_format_depth_only(image->vk.format));
   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT && depth != PIPE_FORMAT_Z16_UNORM) {
      /* TODO: a pass writing frag_depth. */
      mesa_logw_once("wvk: copies into a %s depth aspect are not supported yet",
                     util_format_short_name(depth));
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

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pCopyBufferToImageInfo->regionCount; ++i) {
      const VkBufferImageCopy2 *r = &pCopyBufferToImageInfo->pRegions[i];
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
         repack(wvk_cmd_buffer_device(cmd), replay, buffer_wgpu(src), off, l.row_bytes,
                l.rows_per_image * l.row_bytes, dst->mem->buffer, at, dst->pitch, 0,
                r->imageExtent.width * bpp, r->imageExtent.height, 1);
         wvk_replay_write(replay, dst->mem, at, linear_bytes(dst, r->imageExtent));
      } else if (is_ds_aspect(r->imageSubresource.aspectMask))
         copy_buffer_to_ds(wvk_cmd_buffer_device(cmd), replay, dst, r->imageSubresource.aspectMask, &tex,
                           size, buffer_wgpu(src), off, &l);
      else
         copy_buffer_texture(replay, true, buffer_wgpu(src), off, &l, &tex, size);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyImageToBuffer2(VkCommandBuffer commandBuffer, const VkCopyImageToBufferInfo2 *pCopyImageToBufferInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pCopyImageToBufferInfo->srcImage);
   VK_FROM_HANDLE(wvk_buffer, dst, pCopyImageToBufferInfo->dstBuffer);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pCopyImageToBufferInfo->regionCount; ++i) {
      const VkBufferImageCopy2 *r = &pCopyImageToBufferInfo->pRegions[i];
      WGPUTexelCopyTextureInfo tex;
      WGPUExtent3D size;
      image_region(src, &r->imageSubresource, r->imageOffset, r->imageExtent, &tex, &size);
      const struct copy_layout l = buffer_layout(src->vk.format, r->imageSubresource.aspectMask,
                                                 r->bufferRowLength, r->bufferImageHeight, r->imageExtent);
      const uint64_t off = buffer_offset(dst, r->bufferOffset);
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
         copy_buffer_texture(replay, false, buffer_wgpu(dst), off, &l, &tex, size);
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
   if (src->linear) {
      wvk_replay_read(replay, linear->mem, buf.layout.offset, linear_bytes(linear, r->extent));
      wgpuCommandEncoderCopyBufferToTexture(replay->encoder, &buf, &tex, &size);
   } else {
      wgpuCommandEncoderCopyTextureToBuffer(replay->encoder, &tex, &buf, &size);
      wvk_replay_write(replay, linear->mem, buf.layout.offset, linear_bytes(linear, r->extent));
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdCopyImage2(VkCommandBuffer commandBuffer, const VkCopyImageInfo2 *pCopyImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pCopyImageInfo->srcImage);
   VK_FROM_HANDLE(wvk_image, dst, pCopyImageInfo->dstImage);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   /* WebGPU copies between textures of one format, give or take sRGB;
    * Vulkan between any of one size, which goes through a buffer. */
   const bool same = src->format == dst->format ||
                     util_format_linear(vk_format_to_pipe_format(src->vk.format)) ==
                        util_format_linear(vk_format_to_pipe_format(dst->vk.format));
   for (uint32_t i = 0; i < pCopyImageInfo->regionCount; ++i) {
      const VkImageCopy2 *r = &pCopyImageInfo->pRegions[i];
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
      if (same) {
         wgpuCommandEncoderCopyTextureToTexture(replay->encoder, &s, &d, &size);
         continue;
      }
      struct copy_layout l = buffer_layout(src->vk.format, r->srcSubresource.aspectMask, 0, 0, r->extent);
      l.row_bytes = align(l.row_bytes, 256);
      const uint64_t bytes = (uint64_t)size.depthOrArrayLayers * l.rows_per_image * l.row_bytes;
      WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, bytes,
                                               WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, NULL);
      copy_buffer_texture(replay, false, temp, 0, &l, &s, size);
      struct copy_layout dl = l;
      dl.block_w = util_format_get_blockwidth(df);
      dl.block_h = util_format_get_blockheight(df);
      copy_buffer_texture(replay, true, temp, 0, &dl, &d, dsize);
   }
}

/* ---------------------------------------------------------------------- */
/* Clears and resolves: render passes that only load or clear */

static void
clear_pass(struct wvk_replay *replay, struct wvk_image *image, uint32_t level, uint32_t layer,
           const VkClearColorValue *color, const VkClearDepthStencilValue *ds, VkImageAspectFlags aspects)
{
   WGPUTextureView view = wvk_image_attachment_view(image, image->format, WGPUTextureAspect_All, level, layer);
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

static void
clear_ranges(struct wvk_cmd_buffer *cmd, struct wvk_image *image, const VkClearColorValue *color,
             const VkClearDepthStencilValue *ds, uint32_t range_count, const VkImageSubresourceRange *ranges)
{
   struct wvk_replay *replay = cmd->replay;
   wvk_replay_end_passes(replay);
   if (!(image->usage & WGPUTextureUsage_RenderAttachment)) {
      mesa_logw("wvk: clearing a non-renderable format (%u) is not implemented", image->vk.format);
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
 * resolves into the other -- of whole subresources, which is all WebGPU
 * resolves. */
VKAPI_ATTR void VKAPI_CALL
wvk_CmdResolveImage2(VkCommandBuffer commandBuffer, const VkResolveImageInfo2 *pResolveImageInfo)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(wvk_image, src, pResolveImageInfo->srcImage);
   VK_FROM_HANDLE(wvk_image, dst, pResolveImageInfo->dstImage);
   struct wvk_replay *replay = cmd->replay;

   wvk_replay_end_passes(replay);
   for (uint32_t i = 0; i < pResolveImageInfo->regionCount; ++i) {
      const VkImageResolve2 *r = &pResolveImageInfo->pRegions[i];
      const uint32_t layers = r->srcSubresource.layerCount == VK_REMAINING_ARRAY_LAYERS
                                 ? src->vk.array_layers - r->srcSubresource.baseArrayLayer
                                 : r->srcSubresource.layerCount;
      const VkExtent3D full = vk_image_mip_level_extent(&dst->vk, r->dstSubresource.mipLevel);
      if (r->srcOffset.x || r->srcOffset.y || r->dstOffset.x || r->dstOffset.y ||
          r->extent.width != full.width || r->extent.height != full.height)
         mesa_logw("wvk: resolving part of an image resolves all of it");
      for (uint32_t l = 0; l < layers; ++l) {
         WGPUTextureView s = wvk_image_attachment_view(src, src->format, WGPUTextureAspect_All,
                                                       r->srcSubresource.mipLevel,
                                                       r->srcSubresource.baseArrayLayer + l);
         WGPUTextureView d = wvk_image_attachment_view(dst, dst->format, WGPUTextureAspect_All,
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
