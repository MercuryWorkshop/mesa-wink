/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Draws and dispatches. A WebGPU pipeline holds much of what Vulkan keeps
 * as dynamic state, and its bind group layouts hold what Vulkan's shaders
 * say, so pipelines are made here, at a draw, from the bound shaders and
 * the runtime's dynamic graphics state -- into which the runtime puts a
 * Vulkan pipeline's static state too -- and cached by all of it. Bind
 * groups are made from descriptor sets at a draw as well, and cached in
 * the set until it is written again.
 */
#include "wvk_cmd_buffer.h"

#include "wvk_buffer.h"
#include "wvk_descriptor_set.h"
#include "wvk_descriptor_set_layout.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_format.h"
#include "wvk_image.h"
#include "wvk_physical_device.h"
#include "wvk_sampler.h"
#include "wvk_shader.h"

#include "vk_descriptor_update_template.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_pipeline_layout.h"

#include "util/hash_table.h"
#include "util/u_atomic.h"
#include "util/u_dynarray.h"

#include <stdlib.h>

#define WVK_PUSH_STRIDE 256
#define WVK_MAX_GROUPS 4
#define WVK_MAX_DYNAMIC 32

/* ---------------------------------------------------------------------- */
/* Bind group layouts and pipeline layouts */

struct wvk_bgl_entry {
   struct wvk_shader_binding b;
   uint32_t visibility;   /* WGPUShaderStage */
};

struct wvk_bgl {
   uint32_t hash;
   uint32_t count;
   WGPUBindGroupLayout layout;
   struct wvk_bgl_entry entries[];
};

struct wvk_layout {
   uint32_t hash;
   uint32_t group_count;
   struct wvk_bgl *groups[WVK_MAX_GROUPS];
   WGPUPipelineLayout layout;
};

static uint32_t
bgl_hash(const void *key)
{
   return ((const struct wvk_bgl *)key)->hash;
}

static bool
bgl_equal(const void *a, const void *b)
{
   const struct wvk_bgl *x = a, *y = b;
   return x->count == y->count && !memcmp(x->entries, y->entries, x->count * sizeof(x->entries[0]));
}

static uint32_t
layout_hash(const void *key)
{
   return ((const struct wvk_layout *)key)->hash;
}

static bool
layout_equal(const void *a, const void *b)
{
   const struct wvk_layout *x = a, *y = b;
   return x->group_count == y->group_count && !memcmp(x->groups, y->groups, sizeof(x->groups));
}

static WGPUBindGroupLayoutEntry
bgl_entry(const struct wvk_bgl_entry *e)
{
   WGPUBindGroupLayoutEntry out = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
   out.binding = e->b.binding;
   out.visibility = e->visibility;
   switch (e->b.kind) {
   case WVK_BINDING_UNIFORM:
      out.buffer.type = WGPUBufferBindingType_Uniform;
      out.buffer.hasDynamicOffset = e->b.dynamic;
      break;
   case WVK_BINDING_STORAGE:
      out.buffer.type = WGPUBufferBindingType_Storage;
      out.buffer.hasDynamicOffset = e->b.dynamic;
      break;
   case WVK_BINDING_STORAGE_READ:
      out.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
      out.buffer.hasDynamicOffset = e->b.dynamic;
      break;
   case WVK_BINDING_TEXTURE:
      out.texture.sampleType = e->b.sample_type;
      out.texture.viewDimension = e->b.view_dimension;
      out.texture.multisampled = e->b.multisampled;
      break;
   case WVK_BINDING_STORAGE_TEXTURE:
      out.storageTexture.access = e->b.access;
      out.storageTexture.format = e->b.format;
      out.storageTexture.viewDimension = e->b.view_dimension;
      break;
   case WVK_BINDING_SAMPLER:
      out.sampler.type = WGPUSamplerBindingType_Filtering;
      break;
   case WVK_BINDING_SAMPLER_COMPARE:
      out.sampler.type = WGPUSamplerBindingType_Comparison;
      break;
   case WVK_BINDING_SAMPLER_NONFILTERING:
      out.sampler.type = WGPUSamplerBindingType_NonFiltering;
      break;
   }
   return out;
}

/* Whether a shader binding is one the bound descriptors can change the
 * WebGPU type of: a sampled texture, and a sampler. */
static bool
is_variant_binding(const struct wvk_shader_binding *b)
{
   return (b->kind == WVK_BINDING_TEXTURE && b->sample_type == WGPUTextureSampleType_Float) ||
          b->kind == WVK_BINDING_SAMPLER;
}

/* Equal entries are one object: that is what lets a set's bind group be
 * used with every pipeline that uses the set alike. Called locked. */
static struct wvk_bgl *
get_bgl(struct wvk_device *dev, const struct wvk_bgl_entry *entries, uint32_t count)
{
   struct wvk_bgl *key = calloc(1, sizeof(*key) + count * sizeof(entries[0]));
   if (!key)
      return NULL;
   key->count = count;
   memcpy(key->entries, entries, count * sizeof(entries[0]));
   key->hash = _mesa_hash_data(key->entries, count * sizeof(entries[0])) ^ count;
   struct hash_entry *he = _mesa_hash_table_search(dev->bind_group_layouts, key);
   if (he) {
      free(key);
      return (struct wvk_bgl *)he->key;
   }

   WGPUBindGroupLayoutEntry *wentries = calloc(MAX2(count, 1), sizeof(*wentries));
   for (uint32_t i = 0; wentries && i < count; ++i)
      wentries[i] = bgl_entry(&entries[i]);
   WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
   desc.label = wvk_sv("wvk bind group layout");
   desc.entryCount = count;
   desc.entries = wentries;
   key->layout = wentries ? wgpuDeviceCreateBindGroupLayout(dev->wgpu, &desc) : NULL;
   free(wentries);
   if (!key->layout) {
      free(key);
      return NULL;
   }
   _mesa_hash_table_insert(dev->bind_group_layouts, key, key);
   return key;
}

static int
compare_entries(const void *a, const void *b)
{
   const struct wvk_bgl_entry *x = a, *y = b;
   return (int)x->b.binding - (int)y->b.binding;
}

static WGPUShaderStage
stage_bit(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX: return WGPUShaderStage_Vertex;
   case MESA_SHADER_FRAGMENT: return WGPUShaderStage_Fragment;
   default: return WGPUShaderStage_Compute;
   }
}

/* The pipeline layout of these shaders: every binding any of them uses,
 * visible to the stages that use it; with the bindings that `variant` has
 * a bit for, in the shaders' order, not filtering. */
static struct wvk_layout *
get_layout(struct wvk_device *dev, struct wvk_shader *const *shaders, uint32_t shader_count,
           uint64_t variant)
{
   uint32_t total = 0;
   for (uint32_t s = 0; s < shader_count; ++s)
      total += shaders[s] ? shaders[s]->binding_count : 0;
   struct wvk_bgl_entry *entries[WVK_MAX_GROUPS];
   uint32_t counts[WVK_MAX_GROUPS] = { 0 };
   for (uint32_t g = 0; g < WVK_MAX_GROUPS; ++g)
      entries[g] = calloc(MAX2(total, 1), sizeof(struct wvk_bgl_entry));

   uint32_t group_count = 0;
   unsigned variant_bit = 0;
   for (uint32_t s = 0; s < shader_count; ++s) {
      const struct wvk_shader *shader = shaders[s];
      if (!shader)
         continue;
      const WGPUShaderStage visibility = stage_bit(shader->vk.stage);
      for (uint32_t i = 0; i < shader->binding_count; ++i) {
         struct wvk_shader_binding unfiltered;
         const struct wvk_shader_binding *b = &shader->bindings[i];
         if (is_variant_binding(b) && variant_bit < 64 && (variant >> variant_bit++ & 1)) {
            unfiltered = *b;
            if (b->kind == WVK_BINDING_TEXTURE)
               unfiltered.sample_type = WGPUTextureSampleType_UnfilterableFloat;
            else
               unfiltered.kind = WVK_BINDING_SAMPLER_NONFILTERING;
            b = &unfiltered;
         }
         if (b->group >= WVK_MAX_GROUPS || !entries[b->group])
            continue;
         group_count = MAX2(group_count, b->group + 1u);
         struct wvk_bgl_entry *found = NULL;
         for (uint32_t j = 0; j < counts[b->group]; ++j) {
            if (entries[b->group][j].b.binding == b->binding)
               found = &entries[b->group][j];
         }
         if (found) {
            found->visibility |= visibility;
            /* One stage writes what another reads: storage for both. */
            if (found->b.kind == WVK_BINDING_STORAGE_READ && b->kind == WVK_BINDING_STORAGE)
               found->b.kind = WVK_BINDING_STORAGE;
            found->b.min_size = MAX2(found->b.min_size, b->min_size);
            /* Fetched in one, sampled in the other: sampled. */
            if (found->b.kind == WVK_BINDING_TEXTURE &&
                found->b.sample_type == WGPUTextureSampleType_UnfilterableFloat &&
                b->sample_type != WGPUTextureSampleType_UnfilterableFloat)
               found->b.sample_type = b->sample_type;
            continue;
         }
         entries[b->group][counts[b->group]++] = (struct wvk_bgl_entry){ .b = *b, .visibility = visibility };
      }
   }

   struct wvk_layout key = { .group_count = group_count };
   bool ok = true;
   for (uint32_t g = 0; g < group_count && ok; ++g) {
      qsort(entries[g], counts[g], sizeof(struct wvk_bgl_entry), compare_entries);
      key.groups[g] = counts[g] ? get_bgl(dev, entries[g], counts[g]) : dev->empty_bgl;
      ok = key.groups[g] != NULL;
   }
   for (uint32_t g = 0; g < WVK_MAX_GROUPS; ++g)
      free(entries[g]);
   if (!ok)
      return NULL;
   key.hash = _mesa_hash_data(key.groups, sizeof(key.groups)) ^ group_count;

   struct hash_entry *he = _mesa_hash_table_search(dev->pipeline_layouts, &key);
   if (he)
      return (struct wvk_layout *)he->key;

   struct wvk_layout *layout = malloc(sizeof(*layout));
   if (!layout)
      return NULL;
   *layout = key;
   WGPUBindGroupLayout bgls[WVK_MAX_GROUPS];
   for (uint32_t g = 0; g < group_count; ++g)
      bgls[g] = layout->groups[g]->layout;
   WGPUPipelineLayoutDescriptor desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
   desc.label = wvk_sv("wvk pipeline layout");
   desc.bindGroupLayoutCount = group_count;
   desc.bindGroupLayouts = bgls;
   layout->layout = wgpuDeviceCreatePipelineLayout(dev->wgpu, &desc);
   if (!layout->layout) {
      free(layout);
      return NULL;
   }
   _mesa_hash_table_insert(dev->pipeline_layouts, layout, layout);
   return layout;
}

/* ---------------------------------------------------------------------- */
/* Render and compute pipelines */

struct wvk_render_key {
   uint64_t vs, fs;
   uint32_t color_formats[8];
   uint32_t depth_format;
   uint32_t attr_formats[16];
   uint16_t attr_offsets[16];
   uint8_t attr_locations[16];
   uint8_t attr_bindings[16];
   uint16_t vb_strides[8];
   uint8_t vb_steps[8];
   /* The Vulkan binding of each WebGPU slot, and where in it the slot
    * starts: an attribute past its stride, which WebGPU refuses, is a slot
    * of its own that starts where the attribute is. */
   uint8_t vb_bindings[8];
   uint16_t vb_offsets[8];
   uint8_t attr_count, vb_count;
   uint8_t variant;
   uint8_t topology, strip_index_format, front_face, cull_mode, unclipped_depth;
   uint8_t samples, alpha_to_coverage;
   uint16_t sample_mask;
   uint8_t color_count;
   struct {
      uint8_t blend, write_mask, src_c, dst_c, op_c, src_a, dst_a, op_a;
   } color[8];
   uint8_t depth_write, depth_compare;
   struct {
      uint8_t fail, pass, depth_fail, compare;
   } front, back;
   uint8_t stencil_read_mask, stencil_write_mask;
   int32_t depth_bias;
   float depth_bias_slope, depth_bias_clamp;
   /* binding_variant of the shaders. */
   uint64_t bindings;
};

struct wvk_render_pipeline {
   struct wvk_render_key key;
   uint32_t hash;
   WGPURenderPipeline pipeline;
   struct wvk_layout *layout;
};

struct wvk_compute_pipeline {
   /* The key: the shader, and binding_variant of it. */
   uint64_t cs;
   uint64_t bindings;
   WGPUComputePipeline pipeline;
   struct wvk_layout *layout;
};

static uint32_t
render_hash(const void *key)
{
   return ((const struct wvk_render_pipeline *)key)->hash;
}

static bool
render_equal(const void *a, const void *b)
{
   return !memcmp(&((const struct wvk_render_pipeline *)a)->key,
                  &((const struct wvk_render_pipeline *)b)->key, sizeof(struct wvk_render_key));
}

static uint32_t
compute_hash(const void *key)
{
   return _mesa_hash_data(key, 2 * sizeof(uint64_t));
}

static bool
compute_equal(const void *a, const void *b)
{
   return !memcmp(a, b, 2 * sizeof(uint64_t));
}

VkResult
wvk_device_init_pipeline_caches(struct wvk_device *dev)
{
   simple_mtx_init(&dev->cache_lock, mtx_plain);
   dev->bind_group_layouts = _mesa_hash_table_create(NULL, bgl_hash, bgl_equal);
   dev->pipeline_layouts = _mesa_hash_table_create(NULL, layout_hash, layout_equal);
   dev->render_pipelines = _mesa_hash_table_create(NULL, render_hash, render_equal);
   dev->compute_pipelines = _mesa_hash_table_create(NULL, compute_hash, compute_equal);
   if (!dev->bind_group_layouts || !dev->pipeline_layouts || !dev->render_pipelines ||
       !dev->compute_pipelines)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   dev->empty_bgl = get_bgl(dev, NULL, 0);
   if (!dev->empty_bgl)
      return vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = dev->empty_bgl->layout;
   dev->empty_bind_group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);

   WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
   source.code = wvk_sv("@fragment fn main() {}\n");
   WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
   sm.nextInChain = &source.chain;
   sm.label = wvk_sv("wvk empty fragment");
   dev->empty_fragment = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);

   WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
   bd.label = wvk_sv("wvk null buffer");
   bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex |
              WGPUBufferUsage_Index;
   bd.size = 65536;
   dev->null_buffer = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
   WGPUSamplerDescriptor sd = WGPU_SAMPLER_DESCRIPTOR_INIT;
   dev->null_sampler = wgpuDeviceCreateSampler(dev->wgpu, &sd);
   return VK_SUCCESS;
}

void
wvk_device_finish_pipeline_caches(struct wvk_device *dev)
{
   if (dev->render_pipelines) {
      hash_table_foreach(dev->render_pipelines, he) {
         struct wvk_render_pipeline *p = (void *)he->key;
         if (p->pipeline)
            wgpuRenderPipelineRelease(p->pipeline);
         free(p);
      }
      _mesa_hash_table_destroy(dev->render_pipelines, NULL);
   }
   if (dev->compute_pipelines) {
      hash_table_foreach(dev->compute_pipelines, he) {
         struct wvk_compute_pipeline *p = (void *)he->key;
         if (p->pipeline)
            wgpuComputePipelineRelease(p->pipeline);
         free(p);
      }
      _mesa_hash_table_destroy(dev->compute_pipelines, NULL);
   }
   if (dev->pipeline_layouts) {
      hash_table_foreach(dev->pipeline_layouts, he) {
         struct wvk_layout *l = (void *)he->key;
         wgpuPipelineLayoutRelease(l->layout);
         free(l);
      }
      _mesa_hash_table_destroy(dev->pipeline_layouts, NULL);
   }
   if (dev->bind_group_layouts) {
      hash_table_foreach(dev->bind_group_layouts, he) {
         struct wvk_bgl *l = (void *)he->key;
         wgpuBindGroupLayoutRelease(l->layout);
         free(l);
      }
      _mesa_hash_table_destroy(dev->bind_group_layouts, NULL);
   }
   if (dev->empty_bind_group)
      wgpuBindGroupRelease(dev->empty_bind_group);
   if (dev->empty_fragment)
      wgpuShaderModuleRelease(dev->empty_fragment);
   if (dev->push_buffer)
      wgpuBufferRelease(dev->push_buffer);
   for (uint32_t i = 0; i < ARRAY_SIZE(dev->driver_groups); ++i) {
      if (dev->driver_groups[i].group)
         wgpuBindGroupRelease(dev->driver_groups[i].group);
   }
   if (dev->copy_query_pipeline)
      wgpuComputePipelineRelease(dev->copy_query_pipeline);
   if (dev->add_query_pipeline)
      wgpuComputePipelineRelease(dev->add_query_pipeline);
   if (dev->fan_pipeline)
      wgpuComputePipelineRelease(dev->fan_pipeline);
   if (dev->draw_count_pipeline)
      wgpuComputePipelineRelease(dev->draw_count_pipeline);
   if (dev->repack_pipeline)
      wgpuComputePipelineRelease(dev->repack_pipeline);
   for (uint32_t i = 0; i < ARRAY_SIZE(dev->packed16_pipelines); ++i) {
      if (dev->packed16_pipelines[i])
         wgpuComputePipelineRelease(dev->packed16_pipelines[i]);
   }
   if (dev->depth_out_pipeline)
      wgpuComputePipelineRelease(dev->depth_out_pipeline);
   for (uint32_t i = 0; i < ARRAY_SIZE(dev->depth_in_pipelines); ++i) {
      if (dev->depth_in_pipelines[i].pipeline)
         wgpuRenderPipelineRelease(dev->depth_in_pipelines[i].pipeline);
   }
   for (uint32_t i = 0; i < ARRAY_SIZE(dev->ds_copy_pipelines); ++i) {
      if (dev->ds_copy_pipelines[i])
         wgpuRenderPipelineRelease(dev->ds_copy_pipelines[i]);
   }
   for (uint32_t i = 0; i < ARRAY_SIZE(dev->ms_copy_pipelines); ++i) {
      if (dev->ms_copy_pipelines[i].pipeline)
         wgpuRenderPipelineRelease(dev->ms_copy_pipelines[i].pipeline);
   }
   for (uint32_t i = 0; i < ARRAY_SIZE(dev->shadow_pipelines); ++i) {
      for (uint32_t j = 0; j < 2; ++j) {
         if (dev->shadow_pipelines[i][j])
            wgpuRenderPipelineRelease(dev->shadow_pipelines[i][j]);
      }
   }
   if (dev->null_buffer)
      wgpuBufferRelease(dev->null_buffer);
   if (dev->null_sampler)
      wgpuSamplerRelease(dev->null_sampler);
   if (dev->dummy_attachment)
      wgpuTextureRelease(dev->dummy_attachment);
   for (uint32_t i = 0; i < dev->placeholder_count; ++i) {
      wgpuTextureViewRelease(dev->placeholders[i].view);
      wgpuTextureRelease(dev->placeholders[i].texture);
   }
   simple_mtx_destroy(&dev->cache_lock);
}

static WGPUPrimitiveTopology
topology(uint8_t vk)
{
   switch (vk) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return WGPUPrimitiveTopology_PointList;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return WGPUPrimitiveTopology_LineList;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return WGPUPrimitiveTopology_LineStrip;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: return WGPUPrimitiveTopology_TriangleStrip;
   default: return WGPUPrimitiveTopology_TriangleList;
   }
}

static WGPUBlendFactor
blend_factor(uint8_t vk)
{
   switch (vk) {
   case VK_BLEND_FACTOR_ZERO: return WGPUBlendFactor_Zero;
   case VK_BLEND_FACTOR_ONE: return WGPUBlendFactor_One;
   case VK_BLEND_FACTOR_SRC_COLOR: return WGPUBlendFactor_Src;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return WGPUBlendFactor_OneMinusSrc;
   case VK_BLEND_FACTOR_DST_COLOR: return WGPUBlendFactor_Dst;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return WGPUBlendFactor_OneMinusDst;
   case VK_BLEND_FACTOR_SRC_ALPHA: return WGPUBlendFactor_SrcAlpha;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return WGPUBlendFactor_OneMinusSrcAlpha;
   case VK_BLEND_FACTOR_DST_ALPHA: return WGPUBlendFactor_DstAlpha;
   case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return WGPUBlendFactor_OneMinusDstAlpha;
   /* WebGPU has one constant. As a colour factor, CONSTANT_ALPHA is it
    * with its alpha everywhere: see blend_constant. */
   case VK_BLEND_FACTOR_CONSTANT_COLOR:
   case VK_BLEND_FACTOR_CONSTANT_ALPHA: return WGPUBlendFactor_Constant;
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return WGPUBlendFactor_OneMinusConstant;
   case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE: return WGPUBlendFactor_SrcAlphaSaturated;
   case VK_BLEND_FACTOR_SRC1_COLOR: return WGPUBlendFactor_Src1;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return WGPUBlendFactor_OneMinusSrc1;
   case VK_BLEND_FACTOR_SRC1_ALPHA: return WGPUBlendFactor_Src1Alpha;
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return WGPUBlendFactor_OneMinusSrc1Alpha;
   default: return WGPUBlendFactor_One;
   }
}

static WGPUBlendOperation
blend_op(uint8_t vk)
{
   switch (vk) {
   case VK_BLEND_OP_SUBTRACT: return WGPUBlendOperation_Subtract;
   case VK_BLEND_OP_REVERSE_SUBTRACT: return WGPUBlendOperation_ReverseSubtract;
   case VK_BLEND_OP_MIN: return WGPUBlendOperation_Min;
   case VK_BLEND_OP_MAX: return WGPUBlendOperation_Max;
   default: return WGPUBlendOperation_Add;
   }
}

static WGPUStencilOperation
stencil_op(uint8_t vk)
{
   switch (vk) {
   case VK_STENCIL_OP_ZERO: return WGPUStencilOperation_Zero;
   case VK_STENCIL_OP_REPLACE: return WGPUStencilOperation_Replace;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP: return WGPUStencilOperation_IncrementClamp;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP: return WGPUStencilOperation_DecrementClamp;
   case VK_STENCIL_OP_INVERT: return WGPUStencilOperation_Invert;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP: return WGPUStencilOperation_IncrementWrap;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP: return WGPUStencilOperation_DecrementWrap;
   default: return WGPUStencilOperation_Keep;
   }
}

static WGPUIndexFormat
index_format(VkIndexType type)
{
   return type == VK_INDEX_TYPE_UINT32 ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16;
}

static WGPUStencilFaceState
stencil_face(uint8_t fail, uint8_t pass, uint8_t depth_fail, uint8_t compare)
{
   WGPUStencilFaceState f = WGPU_STENCIL_FACE_STATE_INIT;
   f.compare = wvk_compare_function(compare);
   f.failOp = stencil_op(fail);
   f.depthFailOp = stencil_op(depth_fail);
   f.passOp = stencil_op(pass);
   return f;
}

static WGPURenderPipeline
create_render_pipeline(struct wvk_device *dev, const struct wvk_render_key *k,
                       struct wvk_shader *vs, struct wvk_shader *fs, struct wvk_layout *layout)
{
   WGPUVertexAttribute attrs[16];
   WGPUVertexBufferLayout buffers[8];
   for (uint32_t b = 0; b < k->vb_count; ++b) {
      buffers[b] = (WGPUVertexBufferLayout)WGPU_VERTEX_BUFFER_LAYOUT_INIT;
      buffers[b].stepMode = k->vb_steps[b];
      buffers[b].arrayStride = k->vb_strides[b];
   }
   /* Attributes by binding, as WebGPU nests them. */
   uint32_t attr_count = 0;
   for (uint32_t b = 0; b < k->vb_count; ++b) {
      buffers[b].attributes = &attrs[attr_count];
      for (uint32_t a = 0; a < k->attr_count; ++a) {
         if (k->attr_bindings[a] != b)
            continue;
         attrs[attr_count++] = (WGPUVertexAttribute){
            .format = k->attr_formats[a],
            .offset = k->attr_offsets[a],
            .shaderLocation = k->attr_locations[a],
         };
         ++buffers[b].attributeCount;
      }
      if (!buffers[b].attributeCount)
         buffers[b].stepMode = WGPUVertexStepMode_Undefined;
   }

   WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("wvk render pipeline");
   desc.layout = layout->layout;
   desc.vertex.module = vs->module[k->variant] ? vs->module[k->variant] : vs->module[0];
   desc.vertex.entryPoint = wvk_sv("main");
   desc.vertex.bufferCount = k->vb_count;
   desc.vertex.buffers = buffers;

   desc.primitive.topology = k->topology;
   desc.primitive.stripIndexFormat = k->strip_index_format;
   desc.primitive.frontFace = k->front_face;
   desc.primitive.cullMode = k->cull_mode;
   desc.primitive.unclippedDepth = k->unclipped_depth;

   WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
   if (k->depth_format) {
      ds.format = k->depth_format;
      ds.depthWriteEnabled = k->depth_write ? WGPUOptionalBool_True : WGPUOptionalBool_False;
      ds.depthCompare = wvk_compare_function(k->depth_compare);
      ds.stencilFront = stencil_face(k->front.fail, k->front.pass, k->front.depth_fail, k->front.compare);
      ds.stencilBack = stencil_face(k->back.fail, k->back.pass, k->back.depth_fail, k->back.compare);
      ds.stencilReadMask = k->stencil_read_mask;
      ds.stencilWriteMask = k->stencil_write_mask;
      ds.depthBias = k->depth_bias;
      ds.depthBiasSlopeScale = k->depth_bias_slope;
      ds.depthBiasClamp = k->depth_bias_clamp;
      desc.depthStencil = &ds;
   }

   desc.multisample.count = k->samples;
   desc.multisample.mask = k->sample_mask;
   desc.multisample.alphaToCoverageEnabled = k->alpha_to_coverage;

   WGPUColorTargetState targets[8];
   WGPUBlendState blends[8];
   for (uint32_t i = 0; i < k->color_count; ++i) {
      targets[i] = (WGPUColorTargetState)WGPU_COLOR_TARGET_STATE_INIT;
      targets[i].format = k->color_formats[i];
      targets[i].writeMask = k->color[i].write_mask;
      if (k->color[i].blend) {
         blends[i] = (WGPUBlendState)WGPU_BLEND_STATE_INIT;
         blends[i].color.srcFactor = k->color[i].src_c;
         blends[i].color.dstFactor = k->color[i].dst_c;
         blends[i].color.operation = k->color[i].op_c;
         blends[i].alpha.srcFactor = k->color[i].src_a;
         blends[i].alpha.dstFactor = k->color[i].dst_a;
         blends[i].alpha.operation = k->color[i].op_a;
         targets[i].blend = &blends[i];
      }
   }
   WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
   if (fs || k->color_count) {
      fragment.module = fs ? fs->module[0] : dev->empty_fragment;
      fragment.entryPoint = wvk_sv("main");
      fragment.targetCount = k->color_count;
      fragment.targets = targets;
      desc.fragment = &fragment;
   }
   return wgpuDeviceCreateRenderPipeline(dev->wgpu, &desc);
}

/* ---------------------------------------------------------------------- */
/* What is bound */

struct wvk_bound_set {
   struct wvk_descriptor_set *set;
   uint32_t dynamic_offsets[WVK_MAX_DYNAMIC];
};

enum { WVK_BIND_GRAPHICS, WVK_BIND_COMPUTE, WVK_BIND_POINTS };

struct wvk_bound_buffer {
   struct wvk_device_memory *mem;
   WGPUBuffer buffer;
   uint64_t offset;
   uint64_t size;
   /* A buffer of the driver's own, which has no memory: its own serial,
    * for bind groups kept by what they name. */
   uint64_t serial;
};

/* A driver binding on an application's buffer: the draw binding on an
 * indirect buffer, or a pulled vertex buffer. */
struct wvk_driver_buffer {
   WGPUBuffer buffer;
   /* The memory's serial, or 0 for none. */
   uint64_t serial;
   /* The dynamic offset, and the bind group's size. */
   uint32_t offset;
   uint64_t size;
};

/* An indirect draw's binding: from its command rounded down to 256 bytes,
 * as dynamic offsets are, to the end of the command. The memory's padding
 * covers what that reaches past the buffer. */
#define WVK_DRAW_WINDOW (256 - 4 + sizeof(VkDrawIndexedIndirectCommand))

struct wvk_bound_state {
   /* The shaders a draw is of, which are the bound ones, or their variants
    * for its state (select_draw_shaders). */
   struct wvk_shader *vs, *fs, *cs;
   struct wvk_shader *bound_vs, *bound_fs;
   struct wvk_bound_set sets[WVK_BIND_POINTS][WVK_MAX_GROUPS];

   struct wvk_bound_buffer vb[MESA_VK_MAX_VERTEX_BINDINGS];
   struct wvk_bound_buffer ib;
   VkIndexType index_type;

   /* The push constants, and the driver's data after them. */
   uint8_t push[WVK_PUSH_DATA_SIZE];
   /* Where push[] is in the submit's push data, or UINT32_MAX if it
    * changed since; and whose records went with it. */
   uint32_t push_offset;
   uint32_t push_bind_point;
   const struct wvk_shader *push_shader;
   /* What the draw binding is: the push buffer, or (buffer set) an
    * indirect buffer from a 256-byte boundary. See set_draw_parameters. */
   struct wvk_driver_buffer draw;
   /* The vertex shader's pulled vertex buffers. See set_pulled_buffers. */
   struct wvk_driver_buffer pulled[WVK_MAX_PULLED_BINDINGS];
   /* The last draw's pipeline layout. */
   struct wvk_layout *graphics_layout;

   /* Of the current pass: what is set on it. */
   uint32_t pass_serial;
   WGPURenderPipeline pipeline;
   bool pipeline_flipped;
};

void
wvk_replay_reset_state(struct wvk_replay *replay)
{
   if (!replay->state)
      replay->state = calloc(1, sizeof(*replay->state));
   else
      memset(replay->state, 0, sizeof(*replay->state));
   if (replay->state) {
      replay->state->push_offset = UINT32_MAX;
      const float identity[4] = { 1, 0, 1, 0 };
      memcpy(replay->state->push + WVK_DRIVER_DATA_OFFSET, identity, sizeof(identity));
      const float depth[2] = { 1, 0 };
      memcpy(replay->state->push + WVK_DRIVER_DEPTH_OFFSET, depth, sizeof(depth));
   }
}

void
wvk_replay_set_pass_layer(struct wvk_replay *replay)
{
   struct wvk_bound_state *st = replay->state;
   const uint32_t words[2] = { replay->pass_layer, replay->pass_multiview ? replay->pass_layer : 0 };
   if (st && memcmp(st->push + WVK_DRIVER_LAYER_OFFSET, words, sizeof(words))) {
      memcpy(st->push + WVK_DRIVER_LAYER_OFFSET, words, sizeof(words));
      st->push_offset = UINT32_MAX;
   }
}

void *
wvk_replay_save_state(struct wvk_replay *replay)
{
   struct wvk_bound_state *copy = malloc(sizeof(*copy));
   if (copy)
      *copy = *replay->state;
   return copy;
}

void
wvk_replay_restore_state(struct wvk_replay *replay, void *saved)
{
   struct wvk_bound_state *st = replay->state;
   if (saved) {
      *st = *(struct wvk_bound_state *)saved;
      free(saved);
   }
   /* The pass saw other pipelines meanwhile. */
   st->pipeline = NULL;
}

void
wvk_replay_finish_state(struct wvk_device *dev, struct wvk_replay *replay)
{
   util_dynarray_foreach(&replay->push_sets, struct wvk_descriptor_set *, set)
      wvk_descriptor_set_destroy(dev, *set);
   util_dynarray_fini(&replay->push_sets);
   util_dynarray_fini(&replay->push_data);
   util_dynarray_foreach(&replay->push_buffers, struct wvk_push_buffer, old)
      wgpuBufferRelease(old->buffer);
   util_dynarray_fini(&replay->push_buffers);
   free(replay->state);
   replay->state = NULL;
}

void
wvk_replay_upload_push_constants(struct wvk_device *dev, struct wvk_replay *replay)
{
   const uint64_t size = replay->push_data.size;
   const uint8_t *data = util_dynarray_begin(&replay->push_data);
   if (size && dev->push_buffer)
      wgpuQueueWriteBuffer(dev->queue.wgpu, dev->push_buffer, 0, data, MIN2(size, dev->push_buffer_size));
   util_dynarray_foreach(&replay->push_buffers, struct wvk_push_buffer, old) {
      if (size)
         wgpuQueueWriteBuffer(dev->queue.wgpu, old->buffer, 0, data, MIN2(size, old->size));
      wgpuBufferRelease(old->buffer);
   }
   util_dynarray_clear(&replay->push_buffers);
}

/* The push buffer, big enough for `size` bytes of this submit's push
 * constants: grown when not. Draws before the growth have bind groups on
 * the old buffer, so the submit keeps it and writes its part there too. */
static bool
reserve_push_buffer(struct wvk_device *dev, struct wvk_replay *replay, uint64_t size)
{
   if (dev->push_buffer && dev->push_buffer_size >= size)
      return true;
   const uint64_t new_size = MAX2(util_next_power_of_two64(size), 64 * 1024);
   WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   desc.label = wvk_sv("wvk push constants");
   desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
   desc.size = new_size;
   WGPUBuffer buffer = wgpuDeviceCreateBuffer(dev->wgpu, &desc);
   if (!buffer)
      return false;
   if (dev->push_buffer) {
      struct wvk_push_buffer old = { dev->push_buffer, dev->push_buffer_size };
      util_dynarray_append(&replay->push_buffers, old);
   }
   dev->push_buffer = buffer;
   dev->push_buffer_size = new_size;
   ++dev->push_generation;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Commands that only change what is bound */

void wvk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                          const mesa_shader_stage *stages, struct vk_shader **const shaders);

void
wvk_cmd_bind_shaders(struct vk_command_buffer *vk_cmd, uint32_t stage_count,
                     const mesa_shader_stage *stages, struct vk_shader **const shaders)
{
   struct wvk_cmd_buffer *cmd = container_of(vk_cmd, struct wvk_cmd_buffer, vk);
   struct wvk_bound_state *st = cmd->replay->state;
   for (uint32_t i = 0; i < stage_count; ++i) {
      struct wvk_shader *shader = wvk_shader_from_vk(shaders[i]);
      switch (stages[i]) {
      case MESA_SHADER_VERTEX: st->vs = st->bound_vs = shader; break;
      case MESA_SHADER_FRAGMENT: st->fs = st->bound_fs = shader; break;
      case MESA_SHADER_COMPUTE: st->cs = shader; break;
      default:
         if (shader)
            mesa_logw_once("wvk: %s shaders are not supported", _mesa_shader_stage_to_string(stages[i]));
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBindDescriptorSets2(VkCommandBuffer commandBuffer, const VkBindDescriptorSetsInfo *info)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_bound_state *st = cmd->replay->state;
   uint32_t next_dynamic = 0;
   /* The push data carries the bound sets' records. */
   st->push_offset = UINT32_MAX;
   for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      VK_FROM_HANDLE(wvk_descriptor_set, set, info->pDescriptorSets[i]);
      const uint32_t index = info->firstSet + i;
      if (index >= WVK_MAX_GROUPS)
         break;
      for (uint32_t bp = 0; bp < WVK_BIND_POINTS; ++bp) {
         const VkShaderStageFlags want = bp == WVK_BIND_COMPUTE ? VK_SHADER_STAGE_COMPUTE_BIT
                                                                : VK_SHADER_STAGE_ALL_GRAPHICS;
         if (!(info->stageFlags & want))
            continue;
         struct wvk_bound_set *bound = &st->sets[bp][index];
         bound->set = set;
         const uint32_t dynamic = set ? MIN2(set->layout->dynamic_count, WVK_MAX_DYNAMIC) : 0;
         for (uint32_t d = 0; d < dynamic && next_dynamic + d < info->dynamicOffsetCount; ++d)
            bound->dynamic_offsets[d] = info->pDynamicOffsets[next_dynamic + d];
      }
      if (set)
         next_dynamic += set->layout->dynamic_count;
   }
}

static struct wvk_descriptor_set *
push_set(struct wvk_cmd_buffer *cmd, VkShaderStageFlags stages, VkPipelineLayout _layout, uint32_t index)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   VK_FROM_HANDLE(vk_pipeline_layout, layout, _layout);
   if (index >= WVK_MAX_GROUPS || index >= layout->set_count || !layout->set_layouts[index])
      return NULL;
   struct wvk_descriptor_set_layout *set_layout =
      container_of(layout->set_layouts[index], struct wvk_descriptor_set_layout, vk);
   struct wvk_descriptor_set *set = wvk_descriptor_set_create(dev, set_layout);
   if (!set)
      return NULL;
   util_dynarray_append(&cmd->replay->push_sets, set);
   cmd->replay->state->push_offset = UINT32_MAX;
   for (uint32_t bp = 0; bp < WVK_BIND_POINTS; ++bp) {
      const VkShaderStageFlags want = bp == WVK_BIND_COMPUTE ? VK_SHADER_STAGE_COMPUTE_BIT
                                                             : VK_SHADER_STAGE_ALL_GRAPHICS;
      if (stages & want)
         cmd->replay->state->sets[bp][index].set = set;
   }
   return set;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdPushDescriptorSet2(VkCommandBuffer commandBuffer, const VkPushDescriptorSetInfo *info)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_descriptor_set *set = push_set(cmd, info->stageFlags, info->layout, info->set);
   if (!set)
      return;
   for (uint32_t i = 0; i < info->descriptorWriteCount; ++i)
      wvk_descriptor_set_write(wvk_cmd_buffer_device(cmd), set, &info->pDescriptorWrites[i]);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdPushDescriptorSetWithTemplate2(VkCommandBuffer commandBuffer,
                                      const VkPushDescriptorSetWithTemplateInfo *info)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_descriptor_update_template, template, info->descriptorUpdateTemplate);
   struct wvk_descriptor_set *set =
      push_set(cmd, template->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE ? VK_SHADER_STAGE_COMPUTE_BIT
                                                                           : VK_SHADER_STAGE_ALL_GRAPHICS,
               info->layout, info->set);
   if (set)
      wvk_descriptor_set_write_template(wvk_cmd_buffer_device(cmd), set, template, info->pData);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdPushConstants2(VkCommandBuffer commandBuffer, const VkPushConstantsInfo *info)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_bound_state *st = cmd->replay->state;
   if (info->offset >= WVK_PUSH_SIZE)
      return;
   memcpy(st->push + info->offset, info->pValues, MIN2(info->size, WVK_PUSH_SIZE - info->offset));
   st->push_offset = UINT32_MAX;
}

static struct wvk_bound_buffer
bound_buffer(struct wvk_cmd_buffer *cmd, VkBuffer _buffer, VkDeviceSize offset, VkDeviceSize size)
{
   VK_FROM_HANDLE(wvk_buffer, buffer, _buffer);
   struct wvk_bound_buffer b = { 0 };
   if (!buffer || !buffer->mem)
      return b;
   if (wvk_device_memory_get_buffer(wvk_cmd_buffer_device(cmd), buffer->mem, &b.buffer) != VK_SUCCESS)
      return b;
   b.mem = buffer->mem;
   b.offset = buffer->offset + offset;
   b.size = vk_buffer_range(&buffer->vk, offset, size);
   return b;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBindVertexBuffers2(VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount,
                          const VkBuffer *pBuffers, const VkDeviceSize *pOffsets,
                          const VkDeviceSize *pSizes, const VkDeviceSize *pStrides)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_bound_state *st = cmd->replay->state;
   for (uint32_t i = 0; i < bindingCount && firstBinding + i < MESA_VK_MAX_VERTEX_BINDINGS; ++i)
      st->vb[firstBinding + i] = bound_buffer(cmd, pBuffers[i], pOffsets[i],
                                              pSizes ? pSizes[i] : VK_WHOLE_SIZE);
   if (pStrides)
      vk_cmd_set_vertex_binding_strides(&cmd->vk, firstBinding, bindingCount, pStrides);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBindIndexBuffer2(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                        VkDeviceSize size, VkIndexType indexType)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_bound_state *st = cmd->replay->state;
   st->ib = bound_buffer(cmd, buffer, offset, size);
   st->index_type = indexType;
   if (indexType == VK_INDEX_TYPE_UINT8)
      mesa_logw_once("wvk: 8-bit indices are not supported");
}

/* ---------------------------------------------------------------------- */
/* Bind groups */

/* The descriptor behind binding `slot` of a set, and for a combined image
 * sampler which half. */
static const struct wvk_descriptor *
slot_descriptor(const struct wvk_descriptor_set *set, uint32_t slot, bool *sampler,
                uint32_t *dynamic_index)
{
   if (!set || slot >= set->layout->slot_count)
      return NULL;
   const struct wvk_descriptor_slot *s = &set->layout->slots[slot];
   const struct wvk_descriptor_set_binding_layout *bl = &set->layout->bindings[s->binding];
   *sampler = s->sampler;
   *dynamic_index = bl->dynamic_index + s->element;
   return &set->descriptors[bl->descriptor_index + s->element];
}

/* What the bound descriptors say of their bindings that shaders cannot
 * know, and WebGPU's bind group layouts must: a texture whose view's
 * format does not filter (depth, 32-bit float where WebGPU will not) is
 * unfilterable-float, and a sampler that does not filter is
 * non-filtering. A bit for each binding is_variant_binding, in the
 * shaders' order. */
static uint64_t
binding_variant(struct wvk_device *dev, const struct wvk_bound_state *st, uint32_t bind_point,
                struct wvk_shader *const *shaders, uint32_t shader_count)
{
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   uint64_t variant = 0;
   unsigned bit = 0;
   for (uint32_t s = 0; s < shader_count; ++s) {
      const struct wvk_shader *shader = shaders[s];
      for (uint32_t i = 0; shader && i < shader->binding_count && bit < 64; ++i) {
         const struct wvk_shader_binding *b = &shader->bindings[i];
         if (!is_variant_binding(b))
            continue;
         const unsigned this_bit = bit++;
         if (b->group >= WVK_MAX_GROUPS)
            continue;
         bool sampler;
         uint32_t dynamic_index;
         const struct wvk_descriptor *d = slot_descriptor(st->sets[bind_point][b->group].set, b->binding,
                                                          &sampler, &dynamic_index);
         bool unfiltered = false;
         if (b->kind == WVK_BINDING_TEXTURE)
            unfiltered = d && d->view && !(wvk_format_caps(pdev, d->view_format) & WVK_CAP_FILTER);
         else
            unfiltered = d && d->sampler && d->sampler_type == WGPUSamplerBindingType_NonFiltering;
         if (unfiltered)
            variant |= 1ull << this_bit;
      }
   }
   return variant;
}

static void
note_buffer(struct wvk_replay *replay, const struct wvk_descriptor *d, bool written)
{
   if (!d || !d->mem)
      return;
   wvk_replay_read(replay, d->mem, d->offset, d->size);
   if (written)
      wvk_replay_write(replay, d->mem, d->offset, d->size);
}

/* The application's buffer behind a driver binding, or NULL for the push
 * binding. */
static const struct wvk_driver_buffer *
driver_binding_buffer(const struct wvk_bound_state *st, uint32_t binding)
{
   if (binding == WVK_DRIVER_SLOT_DRAW)
      return &st->draw;
   if (binding >= WVK_DRIVER_SLOT_VERTEX && binding < WVK_DRIVER_SLOT_VERTEX + WVK_MAX_PULLED_BINDINGS)
      return &st->pulled[binding - WVK_DRIVER_SLOT_VERTEX];
   return NULL;
}

/* A 1x1 texture of what the binding wants, for a descriptor never
 * written, which a partially bound set may leave. Its contents are not
 * the point: nothing reads it that the set uses. */
static WGPUTextureView
placeholder_view_locked(struct wvk_device *dev, const struct wvk_shader_binding *b);

static WGPUTextureView
placeholder_view(struct wvk_device *dev, const struct wvk_shader_binding *b)
{
   simple_mtx_lock(&dev->cache_lock);
   WGPUTextureView view = placeholder_view_locked(dev, b);
   simple_mtx_unlock(&dev->cache_lock);
   return view;
}

static WGPUTextureView
placeholder_view_locked(struct wvk_device *dev, const struct wvk_shader_binding *b)
{
   const bool storage = b->kind == WVK_BINDING_STORAGE_TEXTURE;
   const uint64_t key = (uint64_t)b->view_dimension | (uint64_t)b->sample_type << 8 |
                        (uint64_t)b->multisampled << 16 | (uint64_t)storage << 17 |
                        (uint64_t)(storage ? b->format : 0) << 24;
   for (uint32_t i = 0; i < dev->placeholder_count; ++i) {
      if (dev->placeholders[i].key == key)
         return dev->placeholders[i].view;
   }
   if (dev->placeholder_count == ARRAY_SIZE(dev->placeholders))
      return NULL;
   WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
   if (storage)
      format = b->format;
   else if (b->sample_type == WGPUTextureSampleType_Sint)
      format = WGPUTextureFormat_RGBA8Sint;
   else if (b->sample_type == WGPUTextureSampleType_Uint)
      format = WGPUTextureFormat_RGBA8Uint;
   else if (b->sample_type == WGPUTextureSampleType_Depth)
      format = WGPUTextureFormat_Depth32Float;
   const bool cube = b->view_dimension == WGPUTextureViewDimension_Cube ||
                     b->view_dimension == WGPUTextureViewDimension_CubeArray;
   WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
   td.label = wvk_sv("wvk placeholder");
   td.dimension = b->view_dimension == WGPUTextureViewDimension_3D ? WGPUTextureDimension_3D
                                                                   : WGPUTextureDimension_2D;
   td.size = (WGPUExtent3D){ 1, 1, cube ? 6 : 1 };
   td.format = format;
   td.sampleCount = b->multisampled ? 4 : 1;
   td.usage = storage ? WGPUTextureUsage_StorageBinding : WGPUTextureUsage_TextureBinding;
   if (b->multisampled)
      td.usage |= WGPUTextureUsage_RenderAttachment;
   WGPUTexture texture = wgpuDeviceCreateTexture(dev->wgpu, &td);
   if (!texture)
      return NULL;
   WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   vd.dimension = b->view_dimension;
   WGPUTextureView view = wgpuTextureCreateView(texture, &vd);
   dev->placeholders[dev->placeholder_count].key = key;
   dev->placeholders[dev->placeholder_count].texture = texture;
   dev->placeholders[dev->placeholder_count++].view = view;
   return view;
}

static const struct vk_stencil_test_face_state *draw_stencil_face(const struct wvk_replay *replay,
                                                                   const struct vk_dynamic_graphics_state *dyn);

/* The aspect a shader reads of a descriptor's view: of a view of both
 * depth and stencil, an input attachment's, the one its type says. */
static WGPUTextureAspect
descriptor_aspect(const struct wvk_descriptor *d, const struct wvk_shader_binding *b)
{
   if (d->view_desc.aspect != WGPUTextureAspect_All || !d->view_image ||
       d->view_image->vk.aspects != (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
      return d->view_desc.aspect;
   return b->sample_type == WGPUTextureSampleType_Uint || b->sample_type == WGPUTextureSampleType_Sint
             ? WGPUTextureAspect_StencilOnly : WGPUTextureAspect_DepthOnly;
}

/* What a snapshot of what the descriptor reads is of: the view's texture,
 * or of a depth-stencil image the aspect read. */
#define WVK_SNAPSHOT_DEPTH 29
#define WVK_SNAPSHOT_STENCIL 30

static uint32_t
snapshot_key(const struct wvk_descriptor *d, const struct wvk_shader_binding *b)
{
   if (!(d->view_image->vk.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)))
      return d->view_texture;
   return descriptor_aspect(d, b) == WGPUTextureAspect_StencilOnly ||
                (d->view_image->vk.aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
             ? WVK_SNAPSHOT_STENCIL : WVK_SNAPSHOT_DEPTH;
}

/* The copy of an image's texture the pass renders to, if it has one. */
static WGPUTexture
find_snapshot(const struct wvk_replay *replay, const struct wvk_image *image, uint32_t t)
{
   util_dynarray_foreach(&replay->snapshots, struct wvk_snapshot, s) {
      if (s->image == image && s->t == t)
         return s->texture;
   }
   return NULL;
}

/* How much of a buffer a binding binds. Storage bindings are whole words:
 * the last one taken whole, as shaders read and write words of what is
 * less. A uniform buffer binds as much of the buffer as there is: WebGPU
 * wants all of the shader's block bound, where Vulkan wants only what is
 * read. */
static uint64_t
binding_size(const struct wvk_physical_device *pdev, const struct wvk_shader_binding *b,
             const struct wvk_descriptor *d)
{
   const uint64_t max = b->kind == WVK_BINDING_UNIFORM ? pdev->limits.maxUniformBufferBindingSize
                                                       : pdev->limits.maxStorageBufferBindingSize;
   uint64_t size = d->size;
   if (b->kind == WVK_BINDING_UNIFORM && d->mem)
      size = MIN2(MAX2(size, b->min_size), d->mem->buffer_size - d->offset);
   else if (d->mem)
      size = MIN2(align64(size, 4), d->mem->buffer_size - d->offset);
   return MIN2(size, max) & ~3ull;
}

/* The copy a draw or dispatch reads instead of a binding, if it has one. */
static const struct wvk_buffer_swap *
find_buffer_swap(const struct wvk_replay *replay, uint32_t group, uint32_t binding)
{
   util_dynarray_foreach(&replay->buffer_swaps, struct wvk_buffer_swap, s) {
      if (s->group == group && s->binding == binding)
         return s;
   }
   return NULL;
}

/* A bind group of the set; with `replay`, what it reads of images the
 * pass renders to read their copies instead, and of buffers it writes. */
static WGPUBindGroup
make_bind_group(struct wvk_device *dev, const struct wvk_bgl *bgl, const struct wvk_descriptor_set *set,
                uint32_t group, const struct wvk_bound_state *st, struct wvk_replay *replay)
{
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   WGPUBindGroupEntry *entries = calloc(MAX2(bgl->count, 1), sizeof(*entries));
   if (!entries)
      return NULL;
   /* Views made for it, which it keeps. */
   struct util_dynarray own_views;
   util_dynarray_init(&own_views, NULL);
   for (uint32_t i = 0; i < bgl->count; ++i) {
      const struct wvk_shader_binding *b = &bgl->entries[i].b;
      WGPUBindGroupEntry *e = &entries[i];
      *e = (WGPUBindGroupEntry)WGPU_BIND_GROUP_ENTRY_INIT;
      e->binding = b->binding;
      if (group == WVK_DRIVER_GROUP && b->binding >= WVK_DRIVER_SLOT_BASE) {
         const struct wvk_driver_buffer *d = driver_binding_buffer(st, b->binding);
         e->offset = 0;
         if (d && d->buffer) {
            e->buffer = d->buffer;
            e->size = d->size;
         } else if (b->binding >= WVK_DRIVER_SLOT_VERTEX) {
            e->buffer = dev->null_buffer;
            e->size = 256;
         } else {
            /* The push data, the driver's, and the bound sets' records. */
            e->buffer = dev->push_buffer;
            e->size = b->binding == WVK_DRIVER_SLOT_PUSH ? WVK_PUSH_WINDOW : WVK_PUSH_DATA_SIZE;
         }
         continue;
      }
      bool sampler = false;
      uint32_t dynamic_index;
      const struct wvk_descriptor *d = slot_descriptor(set, b->binding, &sampler, &dynamic_index);
      switch (b->kind) {
      case WVK_BINDING_UNIFORM:
      case WVK_BINDING_STORAGE:
      case WVK_BINDING_STORAGE_READ: {
         if (!d || !d->buffer) {
            e->buffer = dev->null_buffer;
            e->size = 256;
            break;
         }
         const struct wvk_buffer_swap *swap = replay ? find_buffer_swap(replay, group, b->binding) : NULL;
         if (swap) {
            e->buffer = swap->copy;
            e->offset = 0;
            e->size = swap->size;
            break;
         }
         e->buffer = d->buffer;
         e->offset = d->offset;
         e->size = binding_size(pdev, b, d);
         break;
      }
      case WVK_BINDING_SAMPLER:
      case WVK_BINDING_SAMPLER_COMPARE:
      case WVK_BINDING_SAMPLER_NONFILTERING:
         e->sampler = d && d->sampler ? d->sampler : dev->null_sampler;
         break;
      case WVK_BINDING_TEXTURE:
      case WVK_BINDING_STORAGE_TEXTURE:
         e->textureView = d && d->view ? d->view : placeholder_view(dev, b);
         if (replay && d && d->view_image && b->kind == WVK_BINDING_TEXTURE) {
            const uint32_t key = snapshot_key(d, b);
            WGPUTexture copy = find_snapshot(replay, d->view_image, key);
            /* A copy of depth or stencil is of that alone, as a colour. */
            WGPUTextureViewDescriptor vd = d->view_desc;
            if (key == WVK_SNAPSHOT_DEPTH || key == WVK_SNAPSHOT_STENCIL) {
               vd.format = WGPUTextureFormat_Undefined;
               vd.aspect = WGPUTextureAspect_All;
            }
            WGPUTextureView v = copy ? wgpuTextureCreateView(copy, &vd) : NULL;
            if (v) {
               util_dynarray_append(&replay->views, v);
               e->textureView = v;
            }
         }
         if (d && d->view && d->view_image && descriptor_aspect(d, b) != d->view_desc.aspect &&
             e->textureView == d->view) {
            WGPUTextureViewDescriptor vd = d->view_desc;
            vd.aspect = descriptor_aspect(d, b);
            vd.format = WGPUTextureFormat_Undefined;
            uint32_t first_level;
            WGPUTextureView v = wgpuTextureCreateView(wvk_image_texture(d->view_image, d->view_texture, &first_level),
                                                      &vd);
            if (v) {
               util_dynarray_append(&own_views, v);
               e->textureView = v;
            }
         }
         break;
      }
   }
   WGPUBindGroup bg = NULL;
   for (uint32_t i = 0; i < bgl->count; ++i) {
      const uint8_t kind = bgl->entries[i].b.kind;
      if ((kind == WVK_BINDING_TEXTURE || kind == WVK_BINDING_STORAGE_TEXTURE) && !entries[i].textureView)
         goto out;
   }
   WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   desc.label = wvk_sv("wvk bind group");
   desc.layout = bgl->layout;
   desc.entryCount = bgl->count;
   desc.entries = entries;
   bg = wgpuDeviceCreateBindGroup(dev->wgpu, &desc);
out:
   util_dynarray_foreach(&own_views, WGPUTextureView, v)
      wgpuTextureViewRelease(*v);
   util_dynarray_fini(&own_views);
   free(entries);
   return bg;
}

/* What this layout's driver bindings are bound to now. */
static struct wvk_driver_key
driver_key(const struct wvk_device *dev, const struct wvk_bgl *bgl, uint32_t group,
           const struct wvk_bound_state *st)
{
   struct wvk_driver_key key = { 0 };
   if (group != WVK_DRIVER_GROUP)
      return key;
   key.push_generation = dev->push_generation;
   for (uint32_t i = 0; i < bgl->count; ++i) {
      const uint32_t binding = bgl->entries[i].b.binding;
      const struct wvk_driver_buffer *d = driver_binding_buffer(st, binding);
      if (!d)
         continue;
      if (binding == WVK_DRIVER_SLOT_DRAW)
         key.draw = d->serial;
      else
         key.vertex[binding - WVK_DRIVER_SLOT_VERTEX] = d->serial;
   }
   return key;
}

/* Images with atomics that the layout's shaders have as their buffers of
 * texels, copied there from their textures where those are the image:
 * outside any pass, and a render pass begun again after. */
static void
sync_atomic_buffers(struct wvk_cmd_buffer *cmd, const struct wvk_layout *layout, uint32_t bind_point)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const bool in_pass = replay->pass != NULL;
   bool ended = false;
   for (uint32_t g = 0; g < layout->group_count; ++g) {
      const struct wvk_bgl *bgl = layout->groups[g];
      const struct wvk_descriptor_set *set = st->sets[bind_point][g].set;
      for (uint32_t i = 0; set && i < bgl->count; ++i) {
         const struct wvk_shader_binding *b = &bgl->entries[i].b;
         if ((b->kind != WVK_BINDING_STORAGE && b->kind != WVK_BINDING_STORAGE_READ) ||
             (g == WVK_DRIVER_GROUP && b->binding >= WVK_DRIVER_SLOT_BASE))
            continue;
         bool sampler;
         uint32_t dynamic_index;
         const struct wvk_descriptor *d = slot_descriptor(set, b->binding, &sampler, &dynamic_index);
         if (!d || !d->atomic_image || d->atomic_image->atomic_in_buffer)
            continue;
         if (!ended) {
            wvk_replay_end_passes(replay);
            ended = true;
         }
         wvk_image_copy_atomic(d->atomic_image, replay->encoder, true);
         d->atomic_image->atomic_in_buffer = true;
         util_dynarray_append(&replay->atomic_images, d->atomic_image);
      }
   }
   if (ended && in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

/* Images the layout's shaders read or write through textures of theirs
 * behind the one last written -- aliases of other formats, shadows --
 * brought up to date: outside any pass, and a render pass begun again
 * after. Storage images are then what is up to date. */
static void
sync_image_textures(struct wvk_cmd_buffer *cmd, const struct wvk_layout *layout, uint32_t bind_point)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   if (!p_atomic_read(&dev->tracked_images))
      return;
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const bool in_pass = replay->pass != NULL;
   for (uint32_t g = 0; g < layout->group_count; ++g) {
      const struct wvk_bgl *bgl = layout->groups[g];
      const struct wvk_descriptor_set *set = st->sets[bind_point][g].set;
      for (uint32_t i = 0; set && i < bgl->count; ++i) {
         const struct wvk_shader_binding *b = &bgl->entries[i].b;
         if (b->kind != WVK_BINDING_TEXTURE && b->kind != WVK_BINDING_STORAGE_TEXTURE)
            continue;
         bool sampler;
         uint32_t dynamic_index;
         const struct wvk_descriptor *d = slot_descriptor(set, b->binding, &sampler, &dynamic_index);
         if (!d || !d->view_image || !d->view_image->tracked)
            continue;
         struct wvk_image *image = d->view_image;
         if (!wvk_image_current(image, d->view_texture, d->view_level, d->view_levels))
            wvk_image_make_current(dev, replay, image, d->view_texture, d->view_level, d->view_levels);
         if (b->kind == WVK_BINDING_STORAGE_TEXTURE)
            wvk_image_written(image, d->view_texture, d->view_level, d->view_levels);
      }
   }
   if (in_pass && !replay->pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

/* A set's inline uniform blocks as it has them now: written before the
 * submit, in the queue's order, after what used them before. */
static void
upload_inline_blocks(struct wvk_device *dev, struct wvk_descriptor_set *set)
{
   if (set->inline_buffer && set->inline_uploaded != set->generation) {
      wgpuQueueWriteBuffer(dev->queue.wgpu, set->inline_buffer, 0, set->inline_data, set->inline_size);
      set->inline_uploaded = set->generation;
   }
}

/* Whether the layout's shaders read, through the set, an image the pass
 * renders to and has a copy of. */
static bool
reads_snapshot(const struct wvk_replay *replay, const struct wvk_bgl *bgl, const struct wvk_descriptor_set *set)
{
   for (uint32_t i = 0; i < bgl->count; ++i) {
      const struct wvk_shader_binding *b = &bgl->entries[i].b;
      if (b->kind != WVK_BINDING_TEXTURE)
         continue;
      bool sampler;
      uint32_t dynamic_index;
      const struct wvk_descriptor *d = slot_descriptor(set, b->binding, &sampler, &dynamic_index);
      if (d && d->view_image && find_snapshot(replay, d->view_image, snapshot_key(d, b)))
         return true;
   }
   return false;
}

/* Whether the pass renders to what a descriptor reads of an image, which
 * WebGPU will not let it read as well: an aspect of depth or stencil it
 * may, read-only. Of another texture of the image's, what it reads is a
 * copy already (sync_image_textures). */
static bool
is_rendered_to(const struct wvk_replay *replay, const struct wvk_descriptor *d, WGPUTextureAspect aspect)
{
   for (uint32_t i = 0; i < replay->rendering.color_count; ++i) {
      const struct wvk_image_view *color = replay->rendering.color[i];
      if (color && wvk_image_view_image(color) == d->view_image && color->texture == d->view_texture)
         return true;
   }
   const struct wvk_image_view *ds = replay->rendering.depth_stencil;
   if (!ds || wvk_image_view_image(ds) != d->view_image)
      return false;
   switch (aspect) {
   case WGPUTextureAspect_DepthOnly:
      return !replay->rendering.depth_read_only;
   case WGPUTextureAspect_StencilOnly:
      return !replay->rendering.stencil_read_only;
   default:
      return !(replay->rendering.depth_read_only && replay->rendering.stencil_read_only);
   }
}

/* What a draw reads of images the pass renders to: copies, made before it
 * with the pass begun again after, and kept until a barrier or the pass
 * ends -- which is when Vulkan has what it rendered visible to reads. */
static void
snapshot_feedback(struct wvk_cmd_buffer *cmd, const struct wvk_layout *layout)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   bool ended = false;
   for (uint32_t g = 0; g < layout->group_count; ++g) {
      const struct wvk_bgl *bgl = layout->groups[g];
      const struct wvk_descriptor_set *set = st->sets[WVK_BIND_GRAPHICS][g].set;
      for (uint32_t i = 0; set && i < bgl->count; ++i) {
         const struct wvk_shader_binding *b = &bgl->entries[i].b;
         if (b->kind != WVK_BINDING_TEXTURE)
            continue;
         bool sampler;
         uint32_t dynamic_index;
         const struct wvk_descriptor *d = slot_descriptor(set, b->binding, &sampler, &dynamic_index);
         if (!d || !d->view_image || !is_rendered_to(replay, d, descriptor_aspect(d, b)))
            continue;
         const uint32_t key = snapshot_key(d, b);
         if (find_snapshot(replay, d->view_image, key))
            continue;
         if (!ended) {
            wvk_replay_end_passes(replay);
            ended = true;
         }
         WGPUTexture copy;
         if (key == WVK_SNAPSHOT_DEPTH || key == WVK_SNAPSHOT_STENCIL)
            copy = wvk_image_snapshot_ds(dev, replay, d->view_image, key == WVK_SNAPSHOT_STENCIL);
         else if (d->view_image->vk.samples > 1)
            copy = wvk_image_snapshot_ms(dev, replay, d->view_image);
         else
            copy = wvk_image_snapshot(dev->wgpu, d->view_image, d->view_texture, replay->encoder);
         if (!copy) {
            mesa_logw_once("wvk: a pass reads a multisampled depth image it renders to");
            continue;
         }
         const struct wvk_snapshot snapshot = { d->view_image, key, copy };
         util_dynarray_append(&replay->snapshots, snapshot);
      }
   }
   if (ended)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

/* A set's bind group for this layout: kept in the set until it changes. */
static WGPUBindGroup
get_bind_group(struct wvk_device *dev, struct wvk_bgl *bgl, struct wvk_descriptor_set *set, uint32_t group,
               const struct wvk_bound_state *st)
{
   const struct wvk_driver_key key = driver_key(dev, bgl, group, st);
   if (!set) {
      /* Only the driver's bindings, or nothing: kept on the device. */
      if (bgl == dev->empty_bgl)
         return dev->empty_bind_group;
      for (uint32_t i = 0; i < ARRAY_SIZE(dev->driver_groups); ++i) {
         if (dev->driver_groups[i].bgl == bgl && dev->driver_groups[i].group &&
             !memcmp(&dev->driver_groups[i].driver, &key, sizeof(key)))
            return dev->driver_groups[i].group;
      }
      WGPUBindGroup bg = make_bind_group(dev, bgl, NULL, group, st, NULL);
      if (!bg)
         return NULL;
      const uint32_t i = dev->driver_group_next;
      dev->driver_group_next = (i + 1) % ARRAY_SIZE(dev->driver_groups);
      if (dev->driver_groups[i].group)
         wgpuBindGroupRelease(dev->driver_groups[i].group);
      dev->driver_groups[i].bgl = bgl;
      dev->driver_groups[i].driver = key;
      dev->driver_groups[i].group = bg;
      return bg;
   }
   upload_inline_blocks(dev, set);
   for (uint32_t i = 0; i < WVK_SET_BIND_GROUPS; ++i) {
      struct wvk_set_bind_group *c = &set->bind_groups[i];
      if (c->group && c->layout == bgl->layout && c->generation == set->generation &&
          !memcmp(&c->driver, &key, sizeof(key)))
         return c->group;
   }
   WGPUBindGroup bg = make_bind_group(dev, bgl, set, group, st, NULL);
   if (!bg)
      return NULL;
   struct wvk_set_bind_group *c = &set->bind_groups[set->bind_group_next];
   set->bind_group_next = (set->bind_group_next + 1) % WVK_SET_BIND_GROUPS;
   if (c->group)
      wgpuBindGroupRelease(c->group);
   *c = (struct wvk_set_bind_group){
      .layout = bgl->layout, .generation = set->generation, .driver = key, .group = bg,
   };
   return bg;
}

/* The push constants' place in the submit's push data: them, the driver's
 * data, and the records of the sets bound at `bind_point`, as far as the
 * shaders' pipeline layout has them. */
static uint32_t
push_offset(struct wvk_device *dev, struct wvk_replay *replay, uint32_t bind_point)
{
   struct wvk_bound_state *st = replay->state;
   const struct wvk_shader *shader = bind_point == WVK_BIND_COMPUTE ? st->cs : st->vs;
   if (st->push_offset != UINT32_MAX && st->push_bind_point == bind_point && st->push_shader == shader)
      return st->push_offset;
   const uint32_t meta_words = shader ? MIN2(shader->meta_words, (WVK_PUSH_WINDOW - WVK_META_OFFSET) / 4) : 0;
   const uint32_t size = align(WVK_PUSH_DATA_SIZE + meta_words * 4, WVK_PUSH_STRIDE);
   const uint32_t offset = replay->push_data.size;
   uint8_t *slot = util_dynarray_grow_bytes(&replay->push_data, 1, size);
   if (!slot)
      return 0;
   memcpy(slot, st->push, WVK_PUSH_DATA_SIZE);
   memset(slot + WVK_PUSH_DATA_SIZE, 0, size - WVK_PUSH_DATA_SIZE);
   uint32_t *records = (uint32_t *)(slot + WVK_META_OFFSET);
   for (uint32_t g = 0; shader && g < WVK_MAX_GROUPS; ++g) {
      const struct wvk_descriptor_set *set = st->sets[bind_point][g].set;
      const uint32_t end = g + 1 < WVK_MAX_GROUPS ? shader->meta_base[g + 1] : shader->meta_words;
      if (set && shader->meta_base[g] < meta_words)
         wvk_descriptor_set_meta(set, records + shader->meta_base[g], MIN2(end, meta_words) - shader->meta_base[g]);
   }
   st->push_offset = offset;
   st->push_bind_point = bind_point;
   st->push_shader = shader;
   return offset;
}

/* What a draw or dispatch binds of buffers through its sets: WebGPU
 * lets none that it writes have another use in the same usage scope,
 * where suballocated memory, one WGPUBuffer, has many. */
struct buffer_use {
   uint32_t group, binding;
   WGPUBuffer buffer;
   uint64_t offset, size;
   bool write;
};

static uint32_t
gather_buffer_uses(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_layout *layout,
                   uint32_t bind_point, struct buffer_use *out, uint32_t max)
{
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   struct wvk_bound_state *st = replay->state;
   uint32_t n = 0;
   for (uint32_t g = 0; g < layout->group_count; ++g) {
      const struct wvk_bgl *bgl = layout->groups[g];
      const struct wvk_bound_set *bound = &st->sets[bind_point][g];
      for (uint32_t i = 0; bound->set && i < bgl->count && n < max; ++i) {
         const struct wvk_shader_binding *b = &bgl->entries[i].b;
         if ((g == WVK_DRIVER_GROUP && b->binding >= WVK_DRIVER_SLOT_BASE) ||
             (b->kind != WVK_BINDING_UNIFORM && b->kind != WVK_BINDING_STORAGE && b->kind != WVK_BINDING_STORAGE_READ))
            continue;
         bool sampler;
         uint32_t dynamic_index = 0;
         const struct wvk_descriptor *d = slot_descriptor(bound->set, b->binding, &sampler, &dynamic_index);
         if (!d || !d->buffer || !d->mem)
            continue;
         const uint32_t dynamic = b->dynamic && dynamic_index < WVK_MAX_DYNAMIC ? bound->dynamic_offsets[dynamic_index] : 0;
         out[n++] = (struct buffer_use){ g, b->binding, d->buffer, d->offset + dynamic, binding_size(pdev, b, d),
                                         b->kind == WVK_BINDING_STORAGE };
      }
   }
   return n;
}

static bool
writes_buffer(const struct buffer_use *uses, uint32_t n, WGPUBuffer buffer)
{
   for (uint32_t i = 0; i < n; ++i) {
      if (uses[i].write && uses[i].buffer == buffer)
         return true;
   }
   return false;
}

/* Copies of what the draw or dispatch reads of buffers it also writes,
 * made here, outside any pass -- which ends one if it has to. True if it
 * made any. A shader reading what it writes of the same range, which
 * Vulkan lets it, sees what was there before. */
static bool
swap_conflicting_reads(struct wvk_device *dev, struct wvk_replay *replay, const struct buffer_use *uses, uint32_t n)
{
   util_dynarray_clear(&replay->buffer_swaps);
   bool made = false;
   for (uint32_t i = 0; i < n; ++i) {
      if (uses[i].write || !writes_buffer(uses, n, uses[i].buffer) || !uses[i].size)
         continue;
      WGPUBuffer copy = wvk_replay_temp_buffer(dev, replay, uses[i].size,
                                               WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage |
                                               WGPUBufferUsage_CopyDst, NULL);
      if (!copy)
         continue;
      if (!made)
         wvk_replay_end_passes(replay);
      made = true;
      wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, uses[i].buffer, uses[i].offset, copy, 0, uses[i].size);
      const struct wvk_buffer_swap swap = { uses[i].group, uses[i].binding, copy, uses[i].size };
      util_dynarray_append(&replay->buffer_swaps, swap);
   }
   return made;
}

/* Whether the render pass has used a buffer as the draw would not: a
 * write of what it read, a read of what it wrote. */
static bool
pass_conflicts(const struct wvk_replay *replay, WGPUBuffer buffer, bool write)
{
   util_dynarray_foreach(&replay->pass_buffers, struct wvk_buffer_use, u) {
      if (u->buffer == buffer && u->write != write)
         return true;
   }
   return false;
}

static void
note_pass_use(struct wvk_replay *replay, WGPUBuffer buffer, bool write)
{
   util_dynarray_foreach(&replay->pass_buffers, struct wvk_buffer_use, u) {
      if (u->buffer == buffer && u->write == write)
         return;
   }
   const struct wvk_buffer_use use = { buffer, write };
   util_dynarray_append(&replay->pass_buffers, use);
}

/* Bind every group of the layout, or only the driver's. Returns false if
 * one could not be made, and the draw is to be skipped. */
static bool
bind_groups(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_layout *layout,
            uint32_t bind_point, WGPURenderPassEncoder pass, WGPUComputePassEncoder compute,
            bool driver_only)
{
   struct wvk_bound_state *st = replay->state;
   /* The push buffer has to exist and hold this submit's data before any
    * bind group names it. */
   bool uses_push = false;
   for (uint32_t g = driver_only ? WVK_DRIVER_GROUP : 0; g < layout->group_count; ++g) {
      const struct wvk_bgl *bgl = layout->groups[g];
      for (uint32_t i = 0; i < bgl->count; ++i)
         uses_push |= g == WVK_DRIVER_GROUP && bgl->entries[i].b.binding == WVK_DRIVER_SLOT_PUSH;
   }
   uint32_t push_at = 0;
   if (uses_push) {
      push_at = push_offset(dev, replay, bind_point);
      if (!reserve_push_buffer(dev, replay, replay->push_data.size + WVK_PUSH_WINDOW))
         return false;
   }

   for (uint32_t g = driver_only ? WVK_DRIVER_GROUP : 0; g < layout->group_count; ++g) {
      struct wvk_bgl *bgl = layout->groups[g];
      struct wvk_bound_set *bound = &st->sets[bind_point][g];
      /* One of its own, for this draw alone, where it reads copies. */
      bool swapped = false;
      util_dynarray_foreach(&replay->buffer_swaps, struct wvk_buffer_swap, s)
         swapped |= s->group == g;
      const bool reads_copies = swapped || (pass && replay->snapshots.size && bgl->count && bound->set &&
                                            reads_snapshot(replay, bgl, bound->set));
      WGPUBindGroup bg;
      if (reads_copies) {
         upload_inline_blocks(dev, bound->set);
         bg = make_bind_group(dev, bgl, bound->set, g, st, replay);
      } else {
         bg = get_bind_group(dev, bgl, bgl->count ? bound->set : NULL, g, st);
      }
      if (!bg)
         return false;

      /* Dynamic offsets in binding order, which is the entries' order;
       * and every buffer that a bind group reads, uploaded first. */
      uint32_t offsets[WVK_MAX_DYNAMIC];
      uint32_t offset_count = 0;
      for (uint32_t i = 0; i < bgl->count; ++i) {
         const struct wvk_shader_binding *b = &bgl->entries[i].b;
         if (g == WVK_DRIVER_GROUP && b->binding >= WVK_DRIVER_SLOT_BASE) {
            if (b->dynamic && offset_count < WVK_MAX_DYNAMIC) {
               /* The draw binding is the push buffer unless it is not. */
               const struct wvk_driver_buffer *d = driver_binding_buffer(st, b->binding);
               offsets[offset_count++] = d && (d->buffer || b->binding != WVK_DRIVER_SLOT_DRAW) ? d->offset
                                                                                                : push_at;
            }
            continue;
         }
         if (b->kind != WVK_BINDING_UNIFORM && b->kind != WVK_BINDING_STORAGE &&
             b->kind != WVK_BINDING_STORAGE_READ)
            continue;
         bool sampler;
         uint32_t dynamic_index = 0;
         const struct wvk_descriptor *d = slot_descriptor(bound->set, b->binding, &sampler, &dynamic_index);
         uint32_t dynamic = 0;
         if (b->dynamic && offset_count < WVK_MAX_DYNAMIC) {
            dynamic = dynamic_index < WVK_MAX_DYNAMIC ? bound->dynamic_offsets[dynamic_index] : 0;
            /* A copy starts where its range did. */
            offsets[offset_count++] = swapped && find_buffer_swap(replay, g, b->binding) ? 0 : dynamic;
         }
         if (d && d->mem) {
            struct wvk_descriptor at = *d;
            at.offset += dynamic;
            note_buffer(replay, &at, b->kind == WVK_BINDING_STORAGE);
         }
      }
      if (pass)
         wgpuRenderPassEncoderSetBindGroup(pass, g, bg, offset_count, offsets);
      else
         wgpuComputePassEncoderSetBindGroup(compute, g, bg, offset_count, offsets);
      if (reads_copies)
         wgpuBindGroupRelease(bg);
   }
   return true;
}

/* ---------------------------------------------------------------------- */
/* Draws */

static void
build_render_key(struct wvk_cmd_buffer *cmd, bool indexed, struct wvk_render_key *k)
{
   const struct wvk_replay *replay = cmd->replay;
   const struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   memset(k, 0, sizeof(*k));
   k->vs = st->vs->id;
   k->fs = st->fs ? st->fs->id : 0;
   k->variant = dyn->vp.viewport_count && dyn->vp.viewports[0].height < 0 ? WVK_SHADER_UNFLIPPED
                                                                        : WVK_SHADER_FLIPPED;

   /* Vertex input: only what the shader reads, and bindings as WebGPU
    * slots of the same number. */
   const struct vk_vertex_input_state *vi = dyn->vi;
   u_foreach_bit(loc, vi->attributes_valid & st->vs->io_mask) {
      if (k->attr_count == ARRAY_SIZE(k->attr_formats))
         break;
      const struct vk_vertex_attribute_state *a = &vi->attributes[loc];
      const struct wvk_format *f = wvk_format_get(a->format);
      if (!f || !f->vertex || a->binding >= MESA_VK_MAX_VERTEX_BINDINGS)
         continue;
      /* Bindings take slots in the order attributes name them. */
      const uint32_t stride = dyn->vi_binding_strides[a->binding];
      const uint32_t bytes = vk_format_get_blocksize(a->format);
      const uint16_t start = stride && a->offset + bytes > stride ? a->offset : 0;
      uint32_t slot = 0;
      while (slot < k->vb_count && (k->vb_bindings[slot] != a->binding || k->vb_offsets[slot] != start))
         ++slot;
      if (slot == k->vb_count) {
         if (slot == ARRAY_SIZE(k->vb_bindings)) {
            mesa_logw_once("wvk: a pipeline reads more vertex buffers than WebGPU has");
            continue;
         }
         k->vb_bindings[k->vb_count] = a->binding;
         k->vb_offsets[k->vb_count++] = start;
      }
      const uint32_t i = k->attr_count++;
      k->attr_formats[i] = f->vertex;
      k->attr_offsets[i] = a->offset - start;
      k->attr_locations[i] = loc;
      k->attr_bindings[i] = slot;
   }
   for (uint32_t slot = 0; slot < k->vb_count; ++slot) {
      const uint32_t b = k->vb_bindings[slot];
      k->vb_strides[slot] = dyn->vi_binding_strides[b];
      k->vb_steps[slot] = vi->bindings[b].input_rate == VK_VERTEX_INPUT_RATE_INSTANCE
                             ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
   }

   k->topology = replay->list_topology ? replay->list_topology : topology(dyn->ia.primitive_topology);
   if (indexed && (k->topology == WGPUPrimitiveTopology_LineStrip ||
                   k->topology == WGPUPrimitiveTopology_TriangleStrip))
      k->strip_index_format = index_format(st->index_type);
   k->front_face = dyn->rs.front_face == VK_FRONT_FACE_CLOCKWISE ? WGPUFrontFace_CW : WGPUFrontFace_CCW;
   k->cull_mode = dyn->rs.cull_mode == VK_CULL_MODE_FRONT_BIT  ? WGPUCullMode_Front
                  : dyn->rs.cull_mode == VK_CULL_MODE_BACK_BIT ? WGPUCullMode_Back
                                                               : WGPUCullMode_None;
   /* A draw for one face culls the other. */
   if (replay->draw_face)
      k->cull_mode = replay->draw_face == 1 ? WGPUCullMode_Back : WGPUCullMode_Front;
   k->unclipped_depth = dyn->rs.depth_clamp_enable;

   k->samples = MAX2(replay->rendering.samples, 1);
   k->sample_mask = dyn->ms.sample_mask;
   /* Not where the fragment shader covers samples itself. */
   k->alpha_to_coverage = dyn->ms.alpha_to_coverage_enable && k->samples > 1 && !(st->fs && st->fs->covers_by_alpha);

   k->color_count = replay->rendering.color_count;
   for (uint32_t i = 0; i < k->color_count; ++i) {
      k->color_formats[i] = replay->rendering.color_format[i];
      if (!k->color_formats[i])
         continue;
      const struct vk_color_blend_attachment_state *a = &dyn->cb.attachments[i];
      const bool written = st->fs && (st->fs->io_mask & BITFIELD_BIT(i)) &&
                           (dyn->cb.color_write_enables & BITFIELD_BIT(i));
      k->color[i].write_mask = written ? a->write_mask : 0;
      /* A packed format without alpha keeps its RGBA8 texture's at one,
       * which is what blending reads as its alpha. */
      const struct wvk_image_view *view = i < replay->rendering.color_count ? replay->rendering.color[i] : NULL;
      if (view) {
         const struct wvk_format *f = wvk_format_get(view->vk.view_format);
         if (f && (f->caps & WVK_CAP_PACKED16) &&
             !util_format_has_alpha(vk_format_to_pipe_format(view->vk.view_format)))
            k->color[i].write_mask &= ~VK_COLOR_COMPONENT_A_BIT;
      }
      k->color[i].blend = a->blend_enable && written;
      if (k->color[i].blend) {
         k->color[i].src_c = blend_factor(a->src_color_blend_factor);
         k->color[i].dst_c = blend_factor(a->dst_color_blend_factor);
         k->color[i].op_c = blend_op(a->color_blend_op);
         k->color[i].src_a = blend_factor(a->src_alpha_blend_factor);
         k->color[i].dst_a = blend_factor(a->dst_alpha_blend_factor);
         k->color[i].op_a = blend_op(a->alpha_blend_op);
         /* WebGPU's min and max take factors of one. */
         if (k->color[i].op_c == WGPUBlendOperation_Min || k->color[i].op_c == WGPUBlendOperation_Max)
            k->color[i].src_c = k->color[i].dst_c = WGPUBlendFactor_One;
         if (k->color[i].op_a == WGPUBlendOperation_Min || k->color[i].op_a == WGPUBlendOperation_Max)
            k->color[i].src_a = k->color[i].dst_a = WGPUBlendFactor_One;
      }
   }

   k->depth_format = replay->rendering.depth_format;
   const bool has_stencil = k->depth_format == WGPUTextureFormat_Depth24PlusStencil8 ||
                            k->depth_format == WGPUTextureFormat_Depth32FloatStencil8 ||
                            k->depth_format == WGPUTextureFormat_Stencil8;
   if (k->depth_format) {
      /* Without a depth aspect there is no depth test, whatever it says. */
      const struct wvk_image_view *ds_view = replay->rendering.depth_stencil;
      const bool has_depth = ds_view && vk_format_has_depth(wvk_image_view_image(ds_view)->vk.format);
      const bool depth_test = dyn->ds.depth.test_enable && has_depth && !replay->rendering.depth_absent;
      k->depth_compare = depth_test ? dyn->ds.depth.compare_op : VK_COMPARE_OP_ALWAYS;
      k->depth_write = depth_test && dyn->ds.depth.write_enable && !replay->rendering.depth_read_only;
      /* WebGPU wants the stencil state left alone without a stencil. */
      if (dyn->ds.stencil.test_enable && has_stencil && !replay->rendering.stencil_absent) {
         const struct vk_stencil_test_face_state *f = &dyn->ds.stencil.front, *b = &dyn->ds.stencil.back;
         k->front.fail = f->op.fail;
         k->front.pass = f->op.pass;
         k->front.depth_fail = f->op.depth_fail;
         k->front.compare = f->op.compare;
         k->back.fail = b->op.fail;
         k->back.pass = b->op.pass;
         k->back.depth_fail = b->op.depth_fail;
         k->back.compare = b->op.compare;
         /* WebGPU has one mask for both faces: the face's this draw is
          * for, or the one not culled. */
         const struct vk_stencil_test_face_state *m = draw_stencil_face(replay, dyn);
         k->stencil_read_mask = m->compare_mask;
         k->stencil_write_mask = m->write_mask;
      } else {
         k->front.compare = k->back.compare = VK_COMPARE_OP_ALWAYS;
         k->stencil_read_mask = k->stencil_write_mask = 0xff;
      }
      if (replay->rendering.stencil_read_only) {
         k->front.fail = k->front.pass = k->front.depth_fail = VK_STENCIL_OP_KEEP;
         k->back.fail = k->back.pass = k->back.depth_fail = VK_STENCIL_OP_KEEP;
         k->stencil_write_mask = 0;
      }
      /* Which WebGPU has of triangles alone. */
      if (dyn->rs.depth_bias.enable && k->topology != WGPUPrimitiveTopology_LineList &&
          k->topology != WGPUPrimitiveTopology_PointList && k->topology != WGPUPrimitiveTopology_LineStrip) {
         k->depth_bias = (int32_t)dyn->rs.depth_bias.constant_factor;
         k->depth_bias_slope = dyn->rs.depth_bias.slope_factor;
         k->depth_bias_clamp = dyn->rs.depth_bias.clamp;
      }
   }
}

static struct wvk_render_pipeline *
get_render_pipeline(struct wvk_device *dev, struct wvk_cmd_buffer *cmd, bool indexed)
{
   struct wvk_bound_state *st = cmd->replay->state;
   struct wvk_render_pipeline *key = calloc(1, sizeof(*key));
   if (!key)
      return NULL;
   build_render_key(cmd, indexed, &key->key);
   struct wvk_shader *const stages[] = { st->vs, st->fs };
   key->key.bindings = binding_variant(dev, st, WVK_BIND_GRAPHICS, stages, 2);
   key->hash = _mesa_hash_data(&key->key, sizeof(key->key));

   simple_mtx_lock(&dev->cache_lock);
   struct hash_entry *he = _mesa_hash_table_search(dev->render_pipelines, key);
   if (he) {
      simple_mtx_unlock(&dev->cache_lock);
      free(key);
      return (struct wvk_render_pipeline *)he->key;
   }
   struct wvk_shader *const shaders[] = { st->vs, st->fs };
   key->layout = get_layout(dev, shaders, 2, key->key.bindings);
   if (key->layout)
      key->pipeline = create_render_pipeline(dev, &key->key, st->vs, st->fs, key->layout);
   if (!key->pipeline) {
      simple_mtx_unlock(&dev->cache_lock);
      free(key);
      return NULL;
   }
   _mesa_hash_table_insert(dev->render_pipelines, key, key);
   simple_mtx_unlock(&dev->cache_lock);
   return key;
}

/* Viewport and scissor, as WebGPU has them: inside the attachments, and a
 * viewport of positive height, the flip being the shader's. */
/* The stencil state of the face a draw is for: the back where only the
 * back is drawn, else the front. */
static const struct vk_stencil_test_face_state *
draw_stencil_face(const struct wvk_replay *replay, const struct vk_dynamic_graphics_state *dyn)
{
   const bool back = replay->draw_face == 2 || (!replay->draw_face && dyn->rs.cull_mode == VK_CULL_MODE_FRONT_BIT);
   return back ? &dyn->ds.stencil.back : &dyn->ds.stencil.front;
}

static void
set_viewport_scissor(struct wvk_cmd_buffer *cmd, WGPURenderPassEncoder pass)
{
   struct wvk_bound_state *st = cmd->replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const VkExtent2D extent = cmd->replay->rendering.extent;
   /* The draw's viewport, of those a vertex shader chooses among. */
   const uint32_t index = cmd->replay->draw_viewport;
   if (memcmp(st->push + WVK_DRIVER_VIEWPORT, &index, 4)) {
      memcpy(st->push + WVK_DRIVER_VIEWPORT, &index, 4);
      st->push_offset = UINT32_MAX;
   }
   if (dyn->vp.viewport_count) {
      VkViewport v = dyn->vp.viewports[MIN2(index, dyn->vp.viewport_count - 1)];
      if (v.height < 0) {
         v.y += v.height;
         v.height = -v.height;
      }
      float x0 = CLAMP(v.x, 0.0f, (float)extent.width), y0 = CLAMP(v.y, 0.0f, (float)extent.height);
      float x1 = CLAMP(v.x + v.width, 0.0f, (float)extent.width);
      float y1 = CLAMP(v.y + v.height, 0.0f, (float)extent.height);
      /* WebGPU's viewport stays inside the attachments, and Vulkan's need
       * not: the vertex shader scales and moves the position to where the
       * clamped one puts it, in WebGPU's y-up space. */
      float transform[4] = { 1, 0, 1, 0 };
      if (x1 > x0 && y1 > y0 && (x0 != v.x || y0 != v.y || x1 != v.x + v.width || y1 != v.y + v.height)) {
         const float cw = x1 - x0, ch = y1 - y0;
         transform[0] = v.width / cw;
         transform[1] = (2 * (v.x - x0) + v.width - cw) / cw;
         transform[2] = v.height / ch;
         const float by = (ch - 2 * (v.y - y0) - v.height) / ch;
         /* The flipped shader negates y after the transform. */
         transform[3] = st->pipeline_flipped ? -by : by;
      }
      if (memcmp(st->push + WVK_DRIVER_DATA_OFFSET, transform, sizeof(transform))) {
         memcpy(st->push + WVK_DRIVER_DATA_OFFSET, transform, sizeof(transform));
         st->push_offset = UINT32_MAX;
      }
      /* An inverted depth range is the range the right way round, and the
       * shader's depth w - z: minDepth + z (maxDepth - minDepth) is
       * maxDepth + (1 - z) (minDepth - maxDepth). */
      const bool inverted = v.minDepth > v.maxDepth;
      const float min_depth = CLAMP(MIN2(v.minDepth, v.maxDepth), 0.0f, 1.0f);
      const float max_depth = CLAMP(MAX2(v.minDepth, v.maxDepth), 0.0f, 1.0f);
      const float depth[2] = { inverted ? -1.0f : 1.0f, inverted ? 1.0f : 0.0f };
      if (memcmp(st->push + WVK_DRIVER_DEPTH_OFFSET, depth, sizeof(depth))) {
         memcpy(st->push + WVK_DRIVER_DEPTH_OFFSET, depth, sizeof(depth));
         st->push_offset = UINT32_MAX;
      }
      wgpuRenderPassEncoderSetViewport(pass, x0, y0, x1 - x0, y1 - y0, min_depth, max_depth);
   }
   if (dyn->vp.scissor_count) {
      const VkRect2D s = dyn->vp.scissors[MIN2(index, dyn->vp.scissor_count - 1)];
      const int64_t x0 = CLAMP((int64_t)s.offset.x, 0, (int64_t)extent.width);
      const int64_t y0 = CLAMP((int64_t)s.offset.y, 0, (int64_t)extent.height);
      const int64_t x1 = CLAMP((int64_t)s.offset.x + s.extent.width, 0, (int64_t)extent.width);
      const int64_t y1 = CLAMP((int64_t)s.offset.y + s.extent.height, 0, (int64_t)extent.height);
      wgpuRenderPassEncoderSetScissorRect(pass, (uint32_t)x0, (uint32_t)y0, (uint32_t)(x1 - x0),
                                          (uint32_t)(y1 - y0));
   }
}

/* The vertex buffers whose attributes the vertex shader reads itself, as
 * storage: each memory's buffer from a 256-byte boundary, as dynamic
 * offsets are, as far as a binding can reach. The driver's data says
 * where in that the bound range starts. */
static void
set_pulled_buffers(struct wvk_device *dev, struct wvk_cmd_buffer *cmd, struct wvk_replay *replay)
{
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t starts[WVK_MAX_PULLED_BINDINGS] = { 0 }, strides[WVK_MAX_PULLED_BINDINGS] = { 0 };
   for (uint32_t i = 0; i < st->vs->pulled_count; ++i) {
      const struct wvk_bound_buffer *vb = &st->vb[st->vs->pulled_bindings[i]];
      strides[i] = dyn->vi_binding_strides[st->vs->pulled_bindings[i]];
      struct wvk_driver_buffer *d = &st->pulled[i];
      if (!vb->buffer || !vb->mem) {
         *d = (struct wvk_driver_buffer){ 0 };
         continue;
      }
      const uint64_t whole = vb->mem->buffer_size;
      const uint64_t size = MIN2(whole, pdev->limits.maxStorageBufferBindingSize) & ~3ull;
      const uint64_t offset = MIN2(vb->offset, whole - size) & ~255ull;
      if (vb->offset + vb->size > offset + size)
         mesa_logw_once("wvk: a vertex buffer the shader pulls reaches past what a binding can");
      *d = (struct wvk_driver_buffer){ vb->mem->buffer, vb->mem->serial, (uint32_t)offset, size };
      starts[i] = (uint32_t)(vb->offset - offset);
      wvk_replay_read(replay, vb->mem, vb->offset, vb->size);
   }
   if (memcmp(st->push + WVK_DRIVER_PULL_OFFSET, starts, sizeof(starts)) ||
       memcmp(st->push + WVK_DRIVER_PULL_STRIDES, strides, sizeof(strides))) {
      memcpy(st->push + WVK_DRIVER_PULL_OFFSET, starts, sizeof(starts));
      memcpy(st->push + WVK_DRIVER_PULL_STRIDES, strides, sizeof(strides));
      st->push_offset = UINT32_MAX;
   }
}

static bool
is_constant_alpha(uint8_t factor)
{
   return factor == VK_BLEND_FACTOR_CONSTANT_ALPHA || factor == VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
}

static bool
is_constant_color(uint8_t factor)
{
   return factor == VK_BLEND_FACTOR_CONSTANT_COLOR || factor == VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
}

/* The blend constant, as WebGPU's one constant has to be: where colour
 * factors use CONSTANT_ALPHA, its alpha in every channel. For alpha
 * factors both are the alpha anyway. Colour factors that use both
 * CONSTANT_COLOR and CONSTANT_ALPHA cannot be had. */
static WGPUColor
blend_constant(const struct vk_dynamic_graphics_state *dyn, uint32_t color_count)
{
   const float *bc = dyn->cb.blend_constants;
   bool alpha = false, color = false;
   for (uint32_t i = 0; i < MIN2(color_count, dyn->cb.attachment_count); ++i) {
      const struct vk_color_blend_attachment_state *a = &dyn->cb.attachments[i];
      if (!a->blend_enable)
         continue;
      alpha |= is_constant_alpha(a->src_color_blend_factor) || is_constant_alpha(a->dst_color_blend_factor);
      color |= is_constant_color(a->src_color_blend_factor) || is_constant_color(a->dst_color_blend_factor);
   }
   if (alpha && color)
      mesa_logw_once("wvk: colour blend factors with both CONSTANT_COLOR and CONSTANT_ALPHA");
   if (alpha && !color)
      return (WGPUColor){ bc[3], bc[3], bc[3], bc[3] };
   return (WGPUColor){ bc[0], bc[1], bc[2], bc[3] };
}

/* Everything a draw needs set on the pass. False skips the draw. */
static bool
flush_draw(struct wvk_cmd_buffer *cmd, bool indexed)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   WGPURenderPassEncoder pass = replay->pass;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;

   if (!pass) {
      mesa_logw_once("wvk: a draw outside a render pass");
      return false;
   }
   if (!st->vs) {
      mesa_logw_once("wvk: a draw without a vertex shader");
      return false;
   }
   /* A pass of several layers, but for its views, is one per layer: what
    * does not choose its layer is drawn in the first. */
   if (replay->pass_layered && !replay->pass_multiview && replay->pass_layer && !st->vs->writes_layer)
      return false;
   if (replay->pass_idle)
      return false;
   if (dyn->rs.rasterizer_discard_enable || dyn->rs.cull_mode == VK_CULL_MODE_FRONT_AND_BACK)
      return false;
   if (dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN && !replay->list_topology) {
      mesa_logw_once("wvk: an indirect draw of a triangle fan");
      return false;
   }

   /* A pass without attachments has the samples its draws have. */
   if (replay->rendering.empty && dyn->ms.rasterization_samples &&
       dyn->ms.rasterization_samples != replay->rendering.samples) {
      replay->rendering.empty_samples = dyn->ms.rasterization_samples;
      wvk_replay_restart_pass(cmd, replay->pass_layer);
      if (!replay->pass)
         return false;
   }

   struct wvk_render_pipeline *p = get_render_pipeline(dev, cmd, indexed);
   if (!p)
      return false;
   sync_image_textures(cmd, p->layout, WVK_BIND_GRAPHICS);
   sync_atomic_buffers(cmd, p->layout, WVK_BIND_GRAPHICS);
   snapshot_feedback(cmd, p->layout);

   /* A render pass is one usage scope: what the draw reads of buffers it
    * writes, copies; and a draw at odds with what the pass used before, a
    * pass of its own. */
   struct buffer_use uses[64];
   uint32_t n = gather_buffer_uses(dev, replay, p->layout, WVK_BIND_GRAPHICS, uses, ARRAY_SIZE(uses));
   if (swap_conflicting_reads(dev, replay, uses, n))
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   /* The rest of what it reads: vertex, index and pulled buffers. */
   for (uint32_t slot = 0; slot < p->key.vb_count && n < ARRAY_SIZE(uses); ++slot) {
      const struct wvk_bound_buffer *vb = &st->vb[p->key.vb_bindings[slot]];
      if (vb->buffer)
         uses[n++] = (struct buffer_use){ UINT32_MAX, 0, vb->buffer, vb->offset, vb->size, false };
   }
   if (indexed && st->ib.buffer && n < ARRAY_SIZE(uses))
      uses[n++] = (struct buffer_use){ UINT32_MAX, 0, st->ib.buffer, st->ib.offset, st->ib.size, false };
   for (uint32_t i = 0; i < st->vs->pulled_count && n < ARRAY_SIZE(uses); ++i) {
      const struct wvk_bound_buffer *vb = &st->vb[st->vs->pulled_bindings[i]];
      if (vb->buffer)
         uses[n++] = (struct buffer_use){ UINT32_MAX, 0, vb->buffer, vb->offset, vb->size, false };
   }
   if (st->draw.buffer && n < ARRAY_SIZE(uses))
      uses[n++] = (struct buffer_use){ UINT32_MAX, 0, st->draw.buffer, st->draw.offset, st->draw.size, false };
   bool restart = false;
   for (uint32_t i = 0; i < n; ++i) {
      const bool copied = uses[i].group != UINT32_MAX && find_buffer_swap(replay, uses[i].group, uses[i].binding);
      if (copied)
         continue;
      if (!uses[i].write && uses[i].group == UINT32_MAX && writes_buffer(uses, n, uses[i].buffer))
         mesa_logw_once("wvk: a draw writes a buffer it reads vertices, indices or its command from");
      restart |= pass_conflicts(replay, uses[i].buffer, uses[i].write);
   }
   if (restart)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   for (uint32_t i = 0; i < n; ++i) {
      if (uses[i].group == UINT32_MAX || !find_buffer_swap(replay, uses[i].group, uses[i].binding))
         note_pass_use(replay, uses[i].buffer, uses[i].write);
   }

   pass = replay->pass;
   if (!pass)
      return false;

   if (st->pass_serial != replay->rendering.serial) {
      st->pass_serial = replay->rendering.serial;
      st->pipeline = NULL;
   }
   if (st->pipeline != p->pipeline) {
      wgpuRenderPassEncoderSetPipeline(pass, p->pipeline);
      st->pipeline = p->pipeline;
   }
   st->pipeline_flipped = p->key.variant == WVK_SHADER_FLIPPED;
   /* Before the bind groups: it writes the driver's data among the push
    * constants. */
   set_viewport_scissor(cmd, pass);
   st->graphics_layout = p->layout;
   if (st->vs->pulled_count)
      set_pulled_buffers(dev, cmd, replay);
   const bool bound = bind_groups(dev, replay, p->layout, WVK_BIND_GRAPHICS, pass, NULL, false);
   util_dynarray_clear(&replay->buffer_swaps);
   if (!bound)
      return false;

   for (uint32_t slot = 0; slot < p->key.vb_count; ++slot) {
      const struct wvk_bound_buffer *vb = &st->vb[p->key.vb_bindings[slot]];
      const uint64_t start = p->key.vb_offsets[slot];
      if (!vb->buffer || start >= vb->size) {
         wgpuRenderPassEncoderSetVertexBuffer(pass, slot, dev->null_buffer, 0, WGPU_WHOLE_SIZE);
         continue;
      }
      wgpuRenderPassEncoderSetVertexBuffer(pass, slot, vb->buffer, vb->offset + start, vb->size - start);
      wvk_replay_read(replay, vb->mem, vb->offset, vb->size);
   }
   if (indexed) {
      const struct wvk_bound_buffer *ib = &st->ib;
      if (!ib->buffer)
         return false;
      wgpuRenderPassEncoderSetIndexBuffer(pass, ib->buffer, index_format(st->index_type),
                                          ib->offset, ib->size);
      wvk_replay_read(replay, ib->mem, ib->offset, ib->size);
   }

   const WGPUColor blend = blend_constant(dyn, replay->rendering.color_count);
   wgpuRenderPassEncoderSetBlendConstant(pass, &blend);
   wgpuRenderPassEncoderSetStencilReference(pass, draw_stencil_face(replay, dyn)->reference);
   return true;
}

/* Where the vertex shader finds BaseVertex and BaseInstance, and what
 * DrawIndex is. A direct draw's values are among the driver's data. An
 * indirect draw's are in its command, `at` in the buffer, which becomes
 * the draw binding. Only for shaders that read them, since the push data
 * changes with every draw. */
static void
set_draw_parameters(struct wvk_bound_state *st, const struct wvk_bound_buffer *indirect, uint64_t at,
                    bool indexed, int32_t base_vertex, uint32_t base_instance, uint32_t draw_id)
{
   if (!st->vs || !st->vs->uses_draw_parameters)
      return;
   uint32_t words[8] = { 0 };
   if (indirect) {
      const uint64_t start = at & ~(uint64_t)255;
      const uint32_t first = (uint32_t)(at - start) / 4;
      /* vertexOffset and firstInstance of a VkDrawIndexedIndirectCommand,
       * firstVertex and firstInstance of a VkDrawIndirectCommand. */
      words[0] = first + (indexed ? 3 : 2);
      words[1] = first + (indexed ? 4 : 3);
      st->draw = (struct wvk_driver_buffer){ indirect->buffer, indirect->mem ? indirect->mem->serial : indirect->serial,
                                             (uint32_t)start, WVK_DRAW_WINDOW };
   } else {
      words[0] = WVK_DRIVER_DRAW_VALUES / 4;
      words[1] = WVK_DRIVER_DRAW_VALUES / 4 + 1;
      words[4] = (uint32_t)base_vertex;
      words[5] = base_instance;
      st->draw = (struct wvk_driver_buffer){ 0 };
   }
   words[2] = draw_id;
   if (memcmp(st->push + WVK_DRIVER_DRAW_OFFSET, words, sizeof(words))) {
      memcpy(st->push + WVK_DRIVER_DRAW_OFFSET, words, sizeof(words));
      st->push_offset = UINT32_MAX;
   }
}

/* How many times a draw is drawn: once in each viewport, when the vertex
 * shader chooses among them; WebGPU has one. */
static uint32_t
draw_viewports(const struct wvk_cmd_buffer *cmd)
{
   const struct wvk_bound_state *st = cmd->replay->state;
   const uint32_t count = cmd->vk.dynamic_graphics_state.vp.viewport_count;
   return st->vs && st->vs->writes_viewport && count > 1 ? MIN2(count, WVK_MAX_VIEWPORTS) : 1;
}

/* And for each face, where their stencil masks or references differ and
 * both are drawn: WebGPU has one of each for both. */
static uint32_t
draw_faces(const struct wvk_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct wvk_replay *replay = cmd->replay;
   const struct vk_stencil_test_face_state *f = &dyn->ds.stencil.front, *b = &dyn->ds.stencil.back;
   const VkPrimitiveTopology t = dyn->ia.primitive_topology;
   const bool triangles = t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ||
                          t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN ||
                          t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY ||
                          t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
   const bool differ = f->compare_mask != b->compare_mask || f->write_mask != b->write_mask ||
                       f->reference != b->reference;
   return dyn->ds.stencil.test_enable && replay->rendering.depth_stencil && triangles && differ &&
                dyn->rs.cull_mode == VK_CULL_MODE_NONE
             ? 2 : 1;
}

/* The next of them, or the first again once all are drawn. */
static bool
next_viewport(struct wvk_cmd_buffer *cmd, uint32_t v, uint32_t n, uint32_t faces)
{
   cmd->replay->draw_viewport = v < n ? v / faces : 0;
   cmd->replay->draw_face = v < n && faces > 1 ? 1 + v % faces : 0;
   return v < n;
}

#define foreach_viewport(cmd)                                                                          \
   for (uint32_t _v = 0, _f = draw_faces(cmd), _n = draw_viewports(cmd) * _f; next_viewport(cmd, _v, _n, _f); \
        ++_v)

/* ---------------------------------------------------------------------- */
/* Lists the driver makes of a draw's triangles: of a fan, which WebGPU does
 * not have, its triangles, each (i + 1, i + 2, 0) as Vulkan's fan has it,
 * so that the provoking vertex is the same; and where the polygon mode is
 * lines or points, which WebGPU does not have either, each triangle's
 * edges, or its corners. A compute shader makes them of the application's
 * indices, or of the vertices in order. Culling is not done for lines and
 * points: it needs the vertex shader's positions. */

enum list_mode {
   LIST_TRIANGLES,
   LIST_EDGES,
   LIST_CORNERS,
};

static const char triangles_wgsl[] =
   "struct P { first: u32, wide: u32, count: u32, restart: u32,\n"
   "           topology: u32, mode: u32, indexed: u32, triangles: u32 }\n"
   "@group(0) @binding(0) var<storage, read> src: array<u32>;\n"
   "@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n"
   "@group(0) @binding(2) var<uniform> p: P;\n"
   "fn index(i: u32) -> u32 {\n"
   "  if (p.indexed == 0u) { return i; }\n"
   "  if (p.wide != 0u) { return src[p.first / 4u + i]; }\n"
   "  let byte = p.first + i * 2u;\n"
   "  return (src[byte / 4u] >> ((byte & 2u) * 8u)) & 0xffffu;\n"
   "}\n"
   /* A triangle's corners: a list's, a strip's -- {i, i+1+(i%2), i+2-(i%2)},
    * Vulkan's order -- or a fan's. */
   "fn corners(t: u32) -> vec3u {\n"
   "  if (p.topology == 0u) { return vec3u(t * 3u, t * 3u + 1u, t * 3u + 2u); }\n"
   "  if (p.topology == 1u) { return vec3u(t, t + 1u + (t & 1u), t + 2u - (t & 1u)); }\n"
   "  return vec3u(t + 1u, t + 2u, 0u);\n"
   "}\n"
   "fn emit(slot: u32, v: vec3u) {\n"
   "  if (p.mode == 1u) {\n"
   "    dst[slot * 6u] = v.x; dst[slot * 6u + 1u] = v.y;\n"
   "    dst[slot * 6u + 2u] = v.y; dst[slot * 6u + 3u] = v.z;\n"
   "    dst[slot * 6u + 4u] = v.z; dst[slot * 6u + 5u] = v.x;\n"
   "  } else {\n"
   "    dst[slot * 3u] = v.x; dst[slot * 3u + 1u] = v.y; dst[slot * 3u + 2u] = v.z;\n"
   "  }\n"
   "}\n"
   /* With primitive restart, one invocation goes through them, each strip
    * or fan begun again after the restart index; the rest of the list
    * degenerate, or for corners the last one again. */
   "fn restarting() {\n"
   "  var out = 0u;\n"
   "  var a = 0u;\n"
   "  var b = 0u;\n"
   "  var n = 0u;\n"
   "  var last = vec3u(0u);\n"
   "  for (var i = 0u; i < p.count; i++) {\n"
   "    let v = index(i);\n"
   "    if (v == p.restart) { n = 0u; continue; }\n"
   "    if (n >= 2u) {\n"
   "      var t = vec3u(b, v, a);\n"
   "      if (p.topology == 1u) { t = select(vec3u(a, b, v), vec3u(a, v, b), ((n - 2u) & 1u) != 0u); }\n"
   "      emit(out, t);\n"
   "      last = vec3u(t.z, t.z, t.z);\n"
   "      out++;\n"
   "    }\n"
   "    if (p.topology == 1u) { a = b; b = v; }\n"
   "    else { if (n == 0u) { a = v; } b = v; }\n"
   "    n++;\n"
   "  }\n"
   "  for (; out < p.triangles; out++) { emit(out, select(vec3u(0u), last, p.mode == 2u)); }\n"
   "}\n"
   "@compute @workgroup_size(64)\n"
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"
   "  let t = id.x + id.y * 65535u * 64u;\n"
   "  if (p.restart != 0u && p.topology != 0u) {\n"
   "    if (t == 0u) { restarting(); }\n"
   "    return;\n"
   "  }\n"
   "  if (t >= p.triangles) { return; }\n"
   "  let c = corners(t);\n"
   "  emit(t, vec3u(index(c.x), index(c.y), index(c.z)));\n"
   "}\n";

static uint32_t
list_triangles(VkPrimitiveTopology topology, uint32_t count)
{
   if (topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST)
      return count / 3;
   return count >= 3 ? count - 2 : 0;
}

/* A list of a draw's triangles, their edges or their corners, from the
 * bound index buffer's indices from `first` on, or of the vertices in
 * order: made before the pass goes on. */
static WGPUBuffer
triangles_on_gpu(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, enum list_mode mode,
                 uint32_t *index_count)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const VkPrimitiveTopology topology = dyn->ia.primitive_topology;
   const uint32_t triangles = list_triangles(topology, count);
   if (!triangles)
      return NULL;
   simple_mtx_lock(&dev->cache_lock);
   if (!dev->fan_pipeline) {
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(triangles_wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk triangle lists");
      desc.compute.module = module;
      desc.compute.entryPoint = wvk_sv("main");
      dev->fan_pipeline = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   simple_mtx_unlock(&dev->cache_lock);
   const struct wvk_bound_buffer *ib = &st->ib;
   if (!dev->fan_pipeline || (indexed && !ib->buffer))
      return NULL;
   const bool wide = st->index_type == VK_INDEX_TYPE_UINT32;
   uint64_t start = 0, base = 0, end = 4;
   if (indexed) {
      start = ib->offset + (uint64_t)first * (wide ? 4 : 2);
      base = start & ~255ull;
      end = start + (uint64_t)count * (wide ? 4 : 2);
   }
   const uint32_t per = mode == LIST_EDGES ? 6 : 3;
   const uint64_t out_bytes = (uint64_t)triangles * per * 4;
   WGPUBuffer out = wvk_replay_temp_buffer(dev, replay, out_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_Index, NULL);
   /* The restart index, or 0 for none: no index is 0 and restarts. */
   const uint32_t restart = indexed && dyn->ia.primitive_restart_enable ? (wide ? 0xffffffffu : 0xffffu) : 0;
   const uint32_t kind = topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ? 0
                         : topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP ? 1 : 2;
   const uint32_t params[8] = { (uint32_t)(start - base), wide, count, restart, kind, mode, indexed, triangles };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   if (!out || !params_buffer)
      return NULL;
   if (indexed)
      wvk_replay_read(replay, ib->mem, start, end - start);
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = indexed ? ib->buffer : dev->null_buffer, .offset = base,
        .size = indexed ? align64(end - base, 4) : 256 },
      { .binding = 1, .buffer = out, .offset = 0, .size = out_bytes },
      { .binding = 2, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
   };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = wgpuComputePipelineGetBindGroupLayout(dev->fan_pipeline, 0);
   bg.entryCount = ARRAY_SIZE(entries);
   bg.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bg.layout);
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderSetPipeline(pass, dev->fan_pipeline);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   const uint32_t groups = DIV_ROUND_UP(triangles, 64);
   wgpuComputePassEncoderDispatchWorkgroups(pass, MIN2(groups, 65535), DIV_ROUND_UP(groups, 65535), 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   *index_count = triangles * per;
   return out;
}

/* A draw of a list the driver made, `indices` its, as the bound index
 * buffer for the draw. */
static void
draw_list(struct wvk_cmd_buffer *cmd, WGPUBuffer indices, uint32_t index_count, WGPUPrimitiveTopology topology,
          uint32_t instances, int32_t base_vertex, uint32_t first_instance, bool indexed, uint32_t draw_id)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct wvk_bound_buffer ib = st->ib;
   const VkIndexType type = st->index_type;
   st->ib = (struct wvk_bound_buffer){ NULL, indices, 0, (uint64_t)index_count * 4 };
   st->index_type = VK_INDEX_TYPE_UINT32;
   replay->list_topology = topology;
   foreach_viewport(cmd) {
      set_draw_parameters(st, NULL, 0, indexed, base_vertex, first_instance, draw_id);
      if (!flush_draw(cmd, true))
         continue;
      wgpuRenderPassEncoderDrawIndexed(replay->pass, index_count, instances, 0, base_vertex, first_instance);
   }
   replay->list_topology = WGPUPrimitiveTopology_Undefined;
   st->ib = ib;
   st->index_type = type;
}

static void
draw_fan(struct wvk_cmd_buffer *cmd, WGPUBuffer indices, uint32_t triangles, uint32_t instances,
         int32_t base_vertex, uint32_t first_instance, bool indexed, uint32_t draw_id)
{
   draw_list(cmd, indices, triangles * 3, WGPUPrimitiveTopology_TriangleList, instances, base_vertex, first_instance,
             indexed, draw_id);
}

/* Of the polygon mode, lines or points; false where it is fill, or the
 * draw not of triangles. */
static bool
polygon_list(const struct wvk_cmd_buffer *cmd, enum list_mode *mode, WGPUPrimitiveTopology *topology)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const VkPrimitiveTopology t = dyn->ia.primitive_topology;
   if (dyn->rs.polygon_mode == VK_POLYGON_MODE_FILL ||
       (t != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST && t != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP &&
        t != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN))
      return false;
   *mode = dyn->rs.polygon_mode == VK_POLYGON_MODE_POINT ? LIST_CORNERS : LIST_EDGES;
   *topology = *mode == LIST_CORNERS ? WGPUPrimitiveTopology_PointList : WGPUPrimitiveTopology_LineList;
   return true;
}

static bool
draw_polygon_mode(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, uint32_t instances,
                  int32_t base_vertex, uint32_t first_instance, uint32_t draw_id)
{
   enum list_mode mode;
   WGPUPrimitiveTopology topology;
   if (!polygon_list(cmd, &mode, &topology))
      return false;
   uint32_t index_count = 0;
   WGPUBuffer indices = triangles_on_gpu(cmd, indexed, first, count, mode, &index_count);
   if (indices)
      draw_list(cmd, indices, index_count, topology, instances, base_vertex, first_instance, indexed, draw_id);
   return true;
}

static void
draw_fan_arrays(struct wvk_cmd_buffer *cmd, uint32_t vertex_count, uint32_t instances, uint32_t first_vertex,
                uint32_t first_instance, uint32_t draw_id)
{
   if (vertex_count < 3 || !instances)
      return;
   const uint32_t triangles = vertex_count - 2;
   uint32_t *list = malloc((size_t)triangles * 12);
   if (!list)
      return;
   for (uint32_t t = 0; t < triangles; ++t) {
      list[t * 3] = t + 1;
      list[t * 3 + 1] = t + 2;
      list[t * 3 + 2] = 0;
   }
   WGPUBuffer indices = wvk_replay_temp_buffer(wvk_cmd_buffer_device(cmd), cmd->replay, (uint64_t)triangles * 12,
                                               WGPUBufferUsage_Index, list);
   free(list);
   if (indices)
      draw_fan(cmd, indices, triangles, instances, (int32_t)first_vertex, first_instance, false, draw_id);
}

static void
draw_fan_elements(struct wvk_cmd_buffer *cmd, uint32_t first_index, uint32_t index_count, uint32_t instances,
                  int32_t vertex_offset, uint32_t first_instance, uint32_t draw_id)
{
   if (index_count < 3 || !instances)
      return;
   uint32_t count = 0;
   WGPUBuffer indices = triangles_on_gpu(cmd, true, first_index, index_count, LIST_TRIANGLES, &count);
   if (indices)
      draw_list(cmd, indices, count, WGPUPrimitiveTopology_TriangleList, instances, vertex_offset, first_instance,
                true, draw_id);
}

/* A draw, the `draw_id`th of a vkCmdDrawMulti*EXT. */
/* The bound shaders, or their variants for the state the draw has: the
 * fragment shader for its multisampling, the vertex shader for its vertex
 * input and for that fragment shader (wvk_shader_draw_variant). */
static void
select_draw_shaders(struct wvk_cmd_buffer *cmd)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   const unsigned samples = replay->rendering.empty && dyn->ms.rasterization_samples
                               ? dyn->ms.rasterization_samples
                               : MAX2(replay->rendering.samples, 1);
   st->fs = wvk_shader_draw_variant(dev, st->bound_fs, NULL, dyn, samples);
   st->vs = wvk_shader_draw_variant(dev, st->bound_vs, st->fs, dyn, samples);
}

static void
draw_arrays(struct wvk_cmd_buffer *cmd, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
            uint32_t firstInstance, uint32_t draw_id)
{
   if (!vertexCount || !instanceCount)
      return;
   select_draw_shaders(cmd);
   if (draw_polygon_mode(cmd, false, 0, vertexCount, instanceCount, (int32_t)firstVertex, firstInstance, draw_id))
      return;
   if (cmd->vk.dynamic_graphics_state.ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
      draw_fan_arrays(cmd, vertexCount, instanceCount, firstVertex, firstInstance, draw_id);
      return;
   }
   foreach_viewport(cmd) {
      set_draw_parameters(cmd->replay->state, NULL, 0, false, (int32_t)firstVertex, firstInstance, draw_id);
      if (!flush_draw(cmd, false))
         continue;
      wgpuRenderPassEncoderDraw(cmd->replay->pass, vertexCount, instanceCount, firstVertex, firstInstance);
   }
}

static void
draw_elements(struct wvk_cmd_buffer *cmd, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
              int32_t vertexOffset, uint32_t firstInstance, uint32_t draw_id)
{
   if (!indexCount || !instanceCount)
      return;
   select_draw_shaders(cmd);
   if (draw_polygon_mode(cmd, true, firstIndex, indexCount, instanceCount, vertexOffset, firstInstance, draw_id))
      return;
   if (cmd->vk.dynamic_graphics_state.ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
      draw_fan_elements(cmd, firstIndex, indexCount, instanceCount, vertexOffset, firstInstance, draw_id);
      return;
   }
   foreach_viewport(cmd) {
      set_draw_parameters(cmd->replay->state, NULL, 0, true, vertexOffset, firstInstance, draw_id);
      if (!flush_draw(cmd, true))
         continue;
      wgpuRenderPassEncoderDrawIndexed(cmd->replay->pass, indexCount, instanceCount, firstIndex,
                                       vertexOffset, firstInstance);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
            uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   draw_arrays(cmd, vertexCount, instanceCount, firstVertex, firstInstance, 0);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                   uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   draw_elements(cmd, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, 0);
}

/* VK_EXT_multi_draw: a draw for each, DrawIndex its number. */
VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount, const VkMultiDrawInfoEXT *pVertexInfo,
                    uint32_t instanceCount, uint32_t firstInstance, uint32_t stride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   for (uint32_t i = 0; i < drawCount; ++i) {
      const VkMultiDrawInfoEXT *d = (const VkMultiDrawInfoEXT *)((const uint8_t *)pVertexInfo + (size_t)i * stride);
      draw_arrays(cmd, d->vertexCount, instanceCount, d->firstVertex, firstInstance, i);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                           const VkMultiDrawIndexedInfoEXT *pIndexInfo, uint32_t instanceCount,
                           uint32_t firstInstance, uint32_t stride, const int32_t *pVertexOffset)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   for (uint32_t i = 0; i < drawCount; ++i) {
      const VkMultiDrawIndexedInfoEXT *d =
         (const VkMultiDrawIndexedInfoEXT *)((const uint8_t *)pIndexInfo + (size_t)i * stride);
      draw_elements(cmd, d->indexCount, instanceCount, d->firstIndex, pVertexOffset ? *pVertexOffset : d->vertexOffset,
                    firstInstance, i);
   }
}

static void
draw_indirect_bound(struct wvk_cmd_buffer *cmd, bool indexed, struct wvk_bound_buffer b, uint32_t drawCount,
                    uint32_t stride)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   if (!drawCount || !b.buffer)
      return;
   /* A fan's draws, and those of lines or points of triangles, as the
    * host has them, one by one as the lists the driver makes. */
   enum list_mode mode;
   WGPUPrimitiveTopology list;
   const bool polygon = polygon_list(cmd, &mode, &list);
   if (polygon || cmd->vk.dynamic_graphics_state.ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
      if (!b.mem || !b.mem->host) {
         mesa_logw_once("wvk: an indirect draw of a triangle fan, or not filled, in memory the host has no copy of");
         return;
      }
      for (uint32_t i = 0; i < drawCount; ++i) {
         const uint8_t *at = (const uint8_t *)b.mem->host + b.offset + (uint64_t)i * stride;
         if (indexed) {
            VkDrawIndexedIndirectCommand c;
            memcpy(&c, at, sizeof(c));
            if (!c.indexCount || !c.instanceCount)
               continue;
            if (!draw_polygon_mode(cmd, true, c.firstIndex, c.indexCount, c.instanceCount, c.vertexOffset,
                                   c.firstInstance, i))
               draw_fan_elements(cmd, c.firstIndex, c.indexCount, c.instanceCount, c.vertexOffset, c.firstInstance, i);
         } else {
            VkDrawIndirectCommand c;
            memcpy(&c, at, sizeof(c));
            if (!c.vertexCount || !c.instanceCount)
               continue;
            if (!draw_polygon_mode(cmd, false, 0, c.vertexCount, c.instanceCount, (int32_t)c.firstVertex,
                                   c.firstInstance, i))
               draw_fan_arrays(cmd, c.vertexCount, c.instanceCount, c.firstVertex, c.firstInstance, i);
         }
      }
      return;
   }
   foreach_viewport(cmd) {
   set_draw_parameters(st, &b, b.offset, indexed, 0, 0, 0);
   if (!flush_draw(cmd, indexed))
      continue;
   wvk_replay_read(replay, b.mem, b.offset, b.size);
   /* WebGPU draws one command at a time; where a shader reads the draw's
    * parameters, every one moves the driver's group to its own. */
   const bool per_draw = st->vs->uses_draw_parameters;
   for (uint32_t i = 0; i < drawCount; ++i) {
      const uint64_t at = b.offset + (uint64_t)i * stride;
      if (per_draw && i) {
         set_draw_parameters(st, &b, at, indexed, 0, 0, i);
         if (!bind_groups(wvk_cmd_buffer_device(cmd), replay, st->graphics_layout, WVK_BIND_GRAPHICS,
                          replay->pass, NULL, true))
            return;
      }
      if (indexed)
         wgpuRenderPassEncoderDrawIndexedIndirect(cmd->replay->pass, b.buffer, at);
      else
         wgpuRenderPassEncoderDrawIndirect(cmd->replay->pass, b.buffer, at);
   }
   }
}

static void
draw_indirect(struct wvk_cmd_buffer *cmd, bool indexed, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount,
              uint32_t stride)
{
   if (!drawCount)
      return;
   select_draw_shaders(cmd);
   const uint32_t size = indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
   draw_indirect_bound(cmd, indexed, bound_buffer(cmd, buffer, offset, (uint64_t)(drawCount - 1) * stride + size),
                       drawCount, stride);
}

/* vkCmdDraw*IndirectCount, which WebGPU does not have: the draws copied,
 * those past the count given no instances by a compute shader, and all
 * of them drawn. */
static const char draw_count_wgsl[] =
   "struct P { count_word: u32, stride_words: u32, max: u32 }\n"
   "@group(0) @binding(0) var<storage, read> count: array<u32>;\n"
   "@group(0) @binding(1) var<storage, read_write> draws: array<u32>;\n"
   "@group(0) @binding(2) var<uniform> p: P;\n"
   "@compute @workgroup_size(64)\n"
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"
   "  if (id.x >= p.max) { return; }\n"
   "  if (id.x >= count[p.count_word]) { draws[id.x * p.stride_words + 1u] = 0u; }\n"
   "}\n";

static void
draw_indirect_count(struct wvk_cmd_buffer *cmd, bool indexed, VkBuffer buffer, VkDeviceSize offset,
                    VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_draws, uint32_t stride)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   if (!max_draws)
      return;
   select_draw_shaders(cmd);
   const uint32_t size = indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
   const uint64_t bytes = (uint64_t)(max_draws - 1) * stride + size;
   const struct wvk_bound_buffer b = bound_buffer(cmd, buffer, offset, bytes);
   const struct wvk_bound_buffer c = bound_buffer(cmd, count_buffer, count_offset, 4);
   if (!b.buffer || !c.buffer)
      return;
   /* A fan's are drawn from the host's copy, the count too. */
   if (cmd->vk.dynamic_graphics_state.ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
      uint32_t count = max_draws;
      if (c.mem && c.mem->host)
         memcpy(&count, (const uint8_t *)c.mem->host + c.offset, 4);
      draw_indirect_bound(cmd, indexed, b, MIN2(count, max_draws), stride);
      return;
   }
   simple_mtx_lock(&dev->cache_lock);
   if (!dev->draw_count_pipeline) {
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(draw_count_wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk draw count");
      desc.compute.module = module;
      desc.compute.entryPoint = wvk_sv("main");
      dev->draw_count_pipeline = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
      wgpuShaderModuleRelease(module);
   }
   const uint64_t serial = ++dev->temp_serial | (1ull << 62);
   simple_mtx_unlock(&dev->cache_lock);
   /* Room after them for the window the draw binding shows of them. */
   WGPUBuffer draws = wvk_replay_temp_buffer(dev, replay, bytes + WVK_DRAW_WINDOW,
                                             WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect |
                                             WGPUBufferUsage_CopyDst, NULL);
   const uint64_t count_base = c.offset & ~255ull;
   const uint32_t params[4] = { (uint32_t)((c.offset - count_base) / 4), stride / 4, max_draws };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   if (!dev->draw_count_pipeline || !draws || !params_buffer)
      return;
   wvk_replay_read(replay, b.mem, b.offset, bytes);
   wvk_replay_read(replay, c.mem, c.offset, 4);
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, b.buffer, b.offset, draws, 0, align64(bytes, 4));
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = c.buffer, .offset = count_base, .size = align64(c.offset + 4 - count_base, 4) },
      { .binding = 1, .buffer = draws, .offset = 0, .size = align64(bytes, 4) },
      { .binding = 2, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
   };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = wgpuComputePipelineGetBindGroupLayout(dev->draw_count_pipeline, 0);
   bg.entryCount = ARRAY_SIZE(entries);
   bg.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bg.layout);
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderSetPipeline(pass, dev->draw_count_pipeline);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuComputePassEncoderDispatchWorkgroups(pass, DIV_ROUND_UP(max_draws, 64), 1, 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   const struct wvk_bound_buffer copy = { NULL, draws, 0, bytes, serial };
   draw_indirect_bound(cmd, indexed, copy, max_draws, stride);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                         VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                         uint32_t stride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   draw_indirect_count(cmd, false, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndexedIndirectCount(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                                VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                uint32_t stride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   draw_indirect_count(cmd, true, buffer, offset, countBuffer, countBufferOffset, maxDrawCount, stride);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                    uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   draw_indirect(cmd, false, buffer, offset, drawCount, stride);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndexedIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset,
                           uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   draw_indirect(cmd, true, buffer, offset, drawCount, stride);
}

/* ---------------------------------------------------------------------- */
/* Dispatches */

static struct wvk_compute_pipeline *
get_compute_pipeline(struct wvk_device *dev, const struct wvk_bound_state *st, struct wvk_shader *cs)
{
   const uint64_t key[2] = { cs->id, binding_variant(dev, st, WVK_BIND_COMPUTE, &cs, 1) };
   simple_mtx_lock(&dev->cache_lock);
   struct hash_entry *he = _mesa_hash_table_search(dev->compute_pipelines, key);
   if (he) {
      simple_mtx_unlock(&dev->cache_lock);
      return (struct wvk_compute_pipeline *)he->key;
   }
   struct wvk_compute_pipeline *p = calloc(1, sizeof(*p));
   if (!p) {
      simple_mtx_unlock(&dev->cache_lock);
      return NULL;
   }
   p->cs = key[0];
   p->bindings = key[1];
   p->layout = get_layout(dev, &cs, 1, p->bindings);
   if (p->layout) {
      WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
      desc.label = wvk_sv("wvk compute pipeline");
      desc.layout = p->layout->layout;
      desc.compute.module = cs->module[0];
      desc.compute.entryPoint = wvk_sv("main");
      p->pipeline = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
   }
   if (!p->pipeline) {
      simple_mtx_unlock(&dev->cache_lock);
      free(p);
      return NULL;
   }
   _mesa_hash_table_insert(dev->compute_pipelines, p, p);
   simple_mtx_unlock(&dev->cache_lock);
   return p;
}

/* A dispatch's base workgroup, which shaders add to their workgroup ids
 * (wvk_shader.c): WebGPU's start at zero. */
static void
set_base_group(struct wvk_replay *replay, uint32_t x, uint32_t y, uint32_t z)
{
   struct wvk_bound_state *st = replay->state;
   const uint32_t words[4] = { x, y, z, 0 };
   if (memcmp(st->push + WVK_DRIVER_BASE_GROUP, words, sizeof(words))) {
      memcpy(st->push + WVK_DRIVER_BASE_GROUP, words, sizeof(words));
      st->push_offset = UINT32_MAX;
   }
}

/* `indirect`, if given, is the dispatch's command, made a copy where the
 * dispatch writes its buffer. */
static WGPUComputePassEncoder
flush_dispatch(struct wvk_cmd_buffer *cmd, struct wvk_bound_buffer *indirect)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   if (!st->cs) {
      mesa_logw_once("wvk: a dispatch without a compute shader");
      return NULL;
   }
   if (replay->pass) {
      mesa_logw_once("wvk: a dispatch inside a render pass");
      return NULL;
   }
   struct wvk_compute_pipeline *p = get_compute_pipeline(dev, st, st->cs);
   if (!p)
      return NULL;
   sync_image_textures(cmd, p->layout, WVK_BIND_COMPUTE);
   sync_atomic_buffers(cmd, p->layout, WVK_BIND_COMPUTE);
   /* A dispatch is a usage scope: what it reads of buffers it writes,
    * copies. */
   struct buffer_use uses[64];
   const uint32_t n = gather_buffer_uses(dev, replay, p->layout, WVK_BIND_COMPUTE, uses, ARRAY_SIZE(uses));
   swap_conflicting_reads(dev, replay, uses, n);
   if (indirect && writes_buffer(uses, n, indirect->buffer)) {
      WGPUBuffer copy = wvk_replay_temp_buffer(dev, replay, 16, WGPUBufferUsage_Indirect | WGPUBufferUsage_CopyDst, NULL);
      if (copy) {
         wvk_replay_end_passes(replay);
         wvk_replay_read(replay, indirect->mem, indirect->offset, indirect->size);
         wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, indirect->buffer, indirect->offset, copy, 0, 12);
         *indirect = (struct wvk_bound_buffer){ NULL, copy, 0, 12 };
      }
   }
   if (!replay->compute) {
      WGPUComputePassDescriptor desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
      replay->compute = wgpuCommandEncoderBeginComputePass(replay->encoder, &desc);
   }
   wgpuComputePassEncoderSetPipeline(replay->compute, p->pipeline);
   const bool bound = bind_groups(dev, replay, p->layout, WVK_BIND_COMPUTE, NULL, replay->compute, false);
   util_dynarray_clear(&replay->buffer_swaps);
   return bound ? replay->compute : NULL;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY,
                    uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   if (!groupCountX || !groupCountY || !groupCountZ)
      return;
   set_base_group(cmd->replay, baseGroupX, baseGroupY, baseGroupZ);
   WGPUComputePassEncoder compute = flush_dispatch(cmd, NULL);
   if (compute)
      wgpuComputePassEncoderDispatchWorkgroups(compute, groupCountX, groupCountY, groupCountZ);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDispatch(VkCommandBuffer commandBuffer, uint32_t x, uint32_t y, uint32_t z)
{
   wvk_CmdDispatchBase(commandBuffer, 0, 0, 0, x, y, z);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDispatchIndirect(VkCommandBuffer commandBuffer, VkBuffer buffer, VkDeviceSize offset)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_bound_buffer b = bound_buffer(cmd, buffer, offset, sizeof(VkDispatchIndirectCommand));
   if (!b.buffer)
      return;
   set_base_group(cmd->replay, 0, 0, 0);
   WGPUComputePassEncoder compute = flush_dispatch(cmd, &b);
   if (!compute)
      return;
   if (b.mem)
      wvk_replay_read(cmd->replay, b.mem, b.offset, b.size);
   wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute, b.buffer, b.offset);
}
