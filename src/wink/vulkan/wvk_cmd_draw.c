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
#include "wvk_tessellator.h"
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
   /* An attribute read of a format neither WebGPU nor the shader can. */
   bool unfetchable;
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


/* The fragment shader of a draw that has none, with the discard rectangle
 * test on: the test (wvk_shader.c's lower_discard_rects), of the
 * rectangles the driver's data has. */
static struct wvk_shader *
create_discard_fs(struct wvk_device *dev)
{
   char wgsl[1200];
   snprintf(wgsl, sizeof(wgsl),
            "struct WvkPush { words: array<u32> }\n"
            "@group(%u) @binding(%u) var<storage, read> wvk_push: WvkPush;\n"
            "@fragment fn main(@builtin(position) pos: vec4f) {\n"
            "  let x = i32(floor(pos.x));\n"
            "  let y = i32(floor(pos.y));\n"
            "  let count = wvk_push.words[%uu];\n"
            "  var inside = false;\n"
            "  for (var i = 0u; i < min(count, %uu); i++) {\n"
            "    let at = %uu + 4u * i;\n"
            "    let r = vec4i(bitcast<i32>(wvk_push.words[at]), bitcast<i32>(wvk_push.words[at + 1u]),\n"
            "                  bitcast<i32>(wvk_push.words[at + 2u]), bitcast<i32>(wvk_push.words[at + 3u]));\n"
            "    inside = inside || (x >= r.x && y >= r.y && x < r.z && y < r.w);\n"
            "  }\n"
            "  if (inside == (wvk_push.words[%uu] != 0u)) { discard; }\n"
            "}\n",
            WVK_DRIVER_GROUP, WVK_DRIVER_SLOT_PUSH, WVK_DRIVER_DISCARD / 4, WVK_MAX_DISCARD_RECTANGLES,
            WVK_DRIVER_DISCARD / 4 + 4, WVK_DRIVER_DISCARD / 4 + 1);
   struct wvk_shader *fs = calloc(1, sizeof(*fs));
   struct wvk_shader_binding *bindings = calloc(1, sizeof(*bindings));
   if (!fs || !bindings) {
      free(fs);
      free(bindings);
      return NULL;
   }
   bindings[0] = (struct wvk_shader_binding){
      .group = WVK_DRIVER_GROUP,
      .kind = WVK_BINDING_STORAGE_READ,
      .binding = WVK_DRIVER_SLOT_PUSH,
      .dynamic = true,
   };
   fs->vk.stage = MESA_SHADER_FRAGMENT;
   fs->id = p_atomic_inc_return(&dev->next_shader_id);
   fs->bindings = bindings;
   fs->binding_count = 1;
   fs->key.discard_rects = 1;
   fs->uses_push_constants = true;
   WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
   source.code = wvk_sv(wgsl);
   WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
   sm.nextInChain = &source.chain;
   sm.label = wvk_sv("wvk discard rectangles fragment");
   fs->module[0] = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
   return fs;
}

/* The fragment shader of a draw that has none, with the depth bounds test
 * on: WebGPU draws depth without a fragment stage, and the test is the
 * fragment shader's (wvk_shader.c's lower_depth_bounds). Its bindings are
 * the two a lowered shader's are, so the layout, the copy and the bind
 * group are made for it as for any other. */
static struct wvk_shader *
create_depth_bounds_fs(struct wvk_device *dev)
{
   char wgsl[1024];
   snprintf(wgsl, sizeof(wgsl),
            "struct WvkPush { words: array<u32> }\n"
            "@group(%u) @binding(%u) var<storage, read> wvk_push: WvkPush;\n"
            "@group(%u) @binding(%u) var wvk_depth: texture_2d<f32>;\n"
            "@fragment fn main(@builtin(position) pos: vec4f) {\n"
            "  let held = textureLoad(wvk_depth, vec2i(pos.xy), 0).x;\n"
            "  let lo = bitcast<f32>(wvk_push.words[%uu]);\n"
            "  let hi = bitcast<f32>(wvk_push.words[%uu]);\n"
            "  if (held < lo || held > hi) { discard; }\n"
            "}\n",
            WVK_DRIVER_GROUP, WVK_DRIVER_SLOT_PUSH, WVK_DRIVER_GROUP, WVK_DRIVER_SLOT_DEPTH_BOUNDS,
            WVK_DRIVER_DEPTH_BOUNDS / 4, WVK_DRIVER_DEPTH_BOUNDS / 4 + 1);
   struct wvk_shader *fs = calloc(1, sizeof(*fs));
   struct wvk_shader_binding *bindings = calloc(2, sizeof(*bindings));
   if (!fs || !bindings) {
      free(fs);
      free(bindings);
      return NULL;
   }
   bindings[0] = (struct wvk_shader_binding){
      .group = WVK_DRIVER_GROUP,
      .kind = WVK_BINDING_STORAGE_READ,
      .binding = WVK_DRIVER_SLOT_PUSH,
      .dynamic = true,
   };
   bindings[1] = (struct wvk_shader_binding){
      .group = WVK_DRIVER_GROUP,
      .kind = WVK_BINDING_TEXTURE,
      .binding = WVK_DRIVER_SLOT_DEPTH_BOUNDS,
      .view_dimension = WGPUTextureViewDimension_2D,
      .sample_type = WGPUTextureSampleType_UnfilterableFloat,
   };
   fs->vk.stage = MESA_SHADER_FRAGMENT;
   fs->id = p_atomic_inc_return(&dev->next_shader_id);
   fs->bindings = bindings;
   fs->binding_count = 2;
   fs->reads_depth_bounds = true;
   fs->uses_push_constants = true;
   WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
   source.code = wvk_sv(wgsl);
   WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
   sm.nextInChain = &source.chain;
   sm.label = wvk_sv("wvk depth bounds fragment");
   fs->module[0] = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
   return fs;
}

VkResult
wvk_device_init_pipeline_caches(struct wvk_device *dev)
{
   simple_mtx_init(&dev->cache_lock, mtx_plain);
   util_dynarray_init(&dev->placeholders, NULL);
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
   dev->depth_bounds_fs = create_depth_bounds_fs(dev);
   dev->discard_fs = create_discard_fs(dev);

   WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
   bd.label = wvk_sv("wvk null buffer");
   bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_Vertex |
              WGPUBufferUsage_Index;
   bd.size = 65536;
   dev->null_buffer = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
   bd.label = wvk_sv("wvk null storage");
   bd.usage = WGPUBufferUsage_Storage;
   dev->null_storage = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
   bd.label = wvk_sv("wvk atomic locks");
   bd.size = WVK_ATOMIC_LOCKS * 4 + 4;
   dev->atomic_locks = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
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
   if (dev->discard_fs) {
      if (dev->discard_fs->module[0])
         wgpuShaderModuleRelease(dev->discard_fs->module[0]);
      free(dev->discard_fs->bindings);
      free(dev->discard_fs);
   }
   if (dev->depth_bounds_fs) {
      if (dev->depth_bounds_fs->module[0])
         wgpuShaderModuleRelease(dev->depth_bounds_fs->module[0]);
      free(dev->depth_bounds_fs->bindings);
      free(dev->depth_bounds_fs);
   }
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
   if (dev->add_stats_pipeline)
      wgpuComputePipelineRelease(dev->add_stats_pipeline);
   if (dev->finish_stats_pipeline)
      wgpuComputePipelineRelease(dev->finish_stats_pipeline);
   if (dev->fan_pipeline)
      wgpuComputePipelineRelease(dev->fan_pipeline);
   if (dev->uint8_index_pipeline)
      wgpuComputePipelineRelease(dev->uint8_index_pipeline);
   if (dev->xfb_pipeline)
      wgpuComputePipelineRelease(dev->xfb_pipeline);
   if (dev->xfb_count_pipeline)
      wgpuComputePipelineRelease(dev->xfb_count_pipeline);
   if (dev->draw_count_pipeline)
      wgpuComputePipelineRelease(dev->draw_count_pipeline);
   for (uint32_t i = 0; i < ARRAY_SIZE(dev->tess_pipelines); ++i) {
      if (dev->tess_pipelines[i])
         wgpuComputePipelineRelease(dev->tess_pipelines[i]);
   }
   if (dev->tess_bgl)
      wgpuBindGroupLayoutRelease(dev->tess_bgl);
   for (uint32_t i = 0; i < ARRAY_SIZE(dev->mesh_pipelines); ++i) {
      if (dev->mesh_pipelines[i])
         wgpuComputePipelineRelease(dev->mesh_pipelines[i]);
   }
   if (dev->mesh_bgl)
      wgpuBindGroupLayoutRelease(dev->mesh_bgl);
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
   if (dev->null_storage)
      wgpuBufferRelease(dev->null_storage);
   if (dev->atomic_locks)
      wgpuBufferRelease(dev->atomic_locks);
   if (dev->null_sampler)
      wgpuSamplerRelease(dev->null_sampler);
   if (dev->dummy_attachment)
      wgpuTextureRelease(dev->dummy_attachment);
   util_dynarray_foreach(&dev->placeholders, struct wvk_placeholder, p) {
      wgpuTextureViewRelease(p->view);
      wgpuTextureRelease(p->texture);
   }
   util_dynarray_fini(&dev->placeholders);
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
   struct wvk_shader *bound_vs, *bound_fs, *bound_cs;
   /* A geometry shader, and tessellation shaders, whose variants run as
    * compute (draw_geometry, draw_tessellated). */
   struct wvk_shader *bound_gs, *bound_tcs, *bound_tes;
   /* Task and mesh shaders, which run as compute too (draw_mesh). */
   struct wvk_shader *bound_task, *bound_mesh;
   struct wvk_bound_set sets[WVK_BIND_POINTS][WVK_MAX_GROUPS];

   struct wvk_bound_buffer vb[MESA_VK_MAX_VERTEX_BINDINGS];
   struct wvk_bound_buffer ib;
   VkIndexType index_type;
   /* 8-bit indices as 16-bit ones, which WebGPU has: made at the first
    * draw of them after the bind, as that draw restarts primitives or not
    * (widen_indices). */
   struct wvk_bound_buffer ib16;
   bool ib16_valid, ib16_restart;

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
   /* The index buffer a draw of points as quads reads. See draw_points. */
   struct wvk_driver_buffer index;
   /* The positions a draw of wide lines reads, or the records a geometry
    * shader does (draw_lines, draw_geometry); and what the latter writes. */
   struct wvk_driver_buffer capture;
   struct wvk_driver_buffer gs_vertices, gs_indices;
   /* Tessellation's: the control shader's records of vertices and patches,
    * the tessellator's points and counts (draw_tessellated). */
   struct wvk_driver_buffer tess[4];
   /* The active pipeline statistics query's counters, which shaders add
    * to (wvk_query.c). */
   struct wvk_driver_buffer stats;
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
      case MESA_SHADER_COMPUTE: st->cs = st->bound_cs = shader; break;
      case MESA_SHADER_GEOMETRY: st->bound_gs = shader; break;
      case MESA_SHADER_TESS_CTRL: st->bound_tcs = shader; break;
      case MESA_SHADER_TESS_EVAL: st->bound_tes = shader; break;
      case MESA_SHADER_TASK: st->bound_task = shader; break;
      case MESA_SHADER_MESH: st->bound_mesh = shader; break;
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
   /* What the push before of this set wrote, and this one does not, is
    * as it was. */
   for (uint32_t bp = 0; bp < WVK_BIND_POINTS; ++bp) {
      const struct wvk_descriptor_set *prev = cmd->replay->state->sets[bp][index].set;
      bool pushed = false;
      util_dynarray_foreach(&cmd->replay->push_sets, struct wvk_descriptor_set *, p)
         pushed |= *p == prev;
      if (pushed && prev->layout == set_layout) {
         wvk_descriptor_set_copy_all(set, prev);
         break;
      }
   }
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
   st->ib16_valid = false;
}

static const char widen_indices_wgsl[] =
   "struct P { start: u32, count: u32, restart: u32, pad: u32 }\n"
   "@group(0) @binding(0) var<storage, read> src: array<u32>;\n"
   "@group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n"
   "@group(0) @binding(2) var<uniform> p: P;\n"
   "@compute @workgroup_size(64)\n"
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"
   "  let i = id.x + id.y * 65535u * 64u;\n"
   "  if (i * 2u >= p.count) { return; }\n"
   "  var word = 0u;\n"
   "  for (var k = 0u; k < 2u; k++) {\n"
   "    let e = i * 2u + k;\n"
   "    if (e < p.count) {\n"
   "      let byte = p.start + e;\n"
   "      var v = (src[byte / 4u] >> ((byte % 4u) * 8u)) & 0xffu;\n"
   "      if (v == 0xffu && p.restart != 0u) { v = 0xffffu; }\n"
   "      word |= v << (k * 16u);\n"
   "    }\n"
   "  }\n"
   "  dst[i] = word;\n"
   "}\n";

/* The bound 8-bit indices as 16-bit ones for a draw (VK_KHR_index_type_uint8),
 * which WebGPU has no 8-bit indices of: a copy made by a pass of the
 * driver's -- the render pass ended for it and begun again -- where the
 * restart index is 0xFFFF if the draw restarts primitives, else 255 as it
 * is. Kept until the next bind, or a draw that restarts otherwise. True
 * where the bound indices are these for now, `saved` what to put back. */
static bool
widen_indices(struct wvk_cmd_buffer *cmd, struct wvk_bound_buffer *saved)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   if (st->index_type != VK_INDEX_TYPE_UINT8 || !st->ib.buffer)
      return false;
   const bool restart = cmd->vk.dynamic_graphics_state.ia.primitive_restart_enable;
   if (!st->ib16_valid || st->ib16_restart != restart) {
      simple_mtx_lock(&dev->cache_lock);
      if (!dev->uint8_index_pipeline) {
         WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
         source.code = wvk_sv(widen_indices_wgsl);
         WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
         sm.nextInChain = &source.chain;
         WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
         WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
         desc.label = wvk_sv("wvk 8-bit indices");
         desc.compute.module = module;
         desc.compute.entryPoint = wvk_sv("main");
         dev->uint8_index_pipeline = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
         wgpuShaderModuleRelease(module);
      }
      simple_mtx_unlock(&dev->cache_lock);
      const struct wvk_bound_buffer *ib = &st->ib;
      const uint64_t count = MIN2(ib->size, UINT32_MAX / 2);
      const uint64_t base = ib->offset & ~255ull;
      const uint64_t out_bytes = align64(MAX2(count * 2, 4), 4);
      WGPUBuffer out = wvk_replay_temp_buffer(dev, replay, out_bytes, WGPUBufferUsage_Storage | WGPUBufferUsage_Index,
                                              NULL);
      const uint32_t params[4] = { (uint32_t)(ib->offset - base), (uint32_t)count, restart };
      WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
      if (!dev->uint8_index_pipeline || !out || !params_buffer || !count)
         return false;
      wvk_replay_read(replay, ib->mem, ib->offset, ib->size);
      const WGPUBindGroupEntry entries[] = {
         { .binding = 0, .buffer = ib->buffer, .offset = base, .size = align64(ib->offset + count - base, 4) },
         { .binding = 1, .buffer = out, .offset = 0, .size = out_bytes },
         { .binding = 2, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
      };
      WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bg.layout = wgpuComputePipelineGetBindGroupLayout(dev->uint8_index_pipeline, 0);
      bg.entryCount = ARRAY_SIZE(entries);
      bg.entries = entries;
      WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
      wgpuBindGroupLayoutRelease(bg.layout);
      const bool in_pass = replay->pass != NULL;
      wvk_replay_end_passes(replay);
      WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
      WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
      wgpuComputePassEncoderSetPipeline(pass, dev->uint8_index_pipeline);
      wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
      const uint32_t groups = (uint32_t)DIV_ROUND_UP(DIV_ROUND_UP(count, 2), 64);
      wgpuComputePassEncoderDispatchWorkgroups(pass, MIN2(groups, 65535), DIV_ROUND_UP(groups, 65535), 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
      wgpuBindGroupRelease(group);
      if (in_pass)
         wvk_replay_restart_pass(cmd, replay->pass_layer);
      /* Whole words, as storage bindings of it want: the pad is no index
       * a draw of these reaches. */
      st->ib16 = (struct wvk_bound_buffer){ NULL, out, 0, out_bytes, ++dev->temp_serial | (1ull << 62) };
      st->ib16_valid = true;
      st->ib16_restart = restart;
   }
   *saved = st->ib;
   st->ib = st->ib16;
   st->index_type = VK_INDEX_TYPE_UINT16;
   return true;
}

static void
narrow_indices(struct wvk_cmd_buffer *cmd, const struct wvk_bound_buffer *saved)
{
   struct wvk_bound_state *st = cmd->replay->state;
   st->ib = *saved;
   st->index_type = VK_INDEX_TYPE_UINT8;
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
   return &set->descriptors[wvk_slot_descriptor_index(bl, s)];
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
   if (binding == WVK_DRIVER_SLOT_INDEX)
      return &st->index;
   if (binding == WVK_DRIVER_SLOT_CAPTURE)
      return &st->capture;
   if (binding == WVK_DRIVER_SLOT_GS_VERTICES)
      return &st->gs_vertices;
   if (binding == WVK_DRIVER_SLOT_GS_INDICES)
      return &st->gs_indices;
   if (binding >= WVK_DRIVER_SLOT_TESS_VERTICES && binding <= WVK_DRIVER_SLOT_TESS_COUNTS)
      return &st->tess[binding - WVK_DRIVER_SLOT_TESS_VERTICES];
   if (binding == WVK_DRIVER_SLOT_STATS)
      return &st->stats;
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
   util_dynarray_foreach(&dev->placeholders, struct wvk_placeholder, p) {
      if (p->key == key)
         return p->view;
   }
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
   const struct wvk_placeholder p = { .key = key, .texture = texture, .view = view };
   util_dynarray_append(&dev->placeholders, p);
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

/* The view a dispatch samples instead of a binding's, if it has one. */
static const struct wvk_texture_swap *
find_texture_swap(const struct wvk_replay *replay, uint32_t group, uint32_t binding)
{
   util_dynarray_foreach(&replay->texture_swaps, struct wvk_texture_swap, s) {
      if (s->group == group && s->binding == binding)
         return s;
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
/* A texture of its own for a writable storage binding of what an earlier
 * one of the group writes too: WebGPU will not have both, where Vulkan
 * leaves the application to keep their writes apart. What it writes is
 * lost, as one of the two writes of a texel would have been. */
static WGPUTextureView
scratch_storage_view(struct wvk_device *dev, const struct wvk_descriptor *d)
{
   const struct wvk_image *image = d->view_image;
   WGPUTextureViewDescriptor vd = d->view_desc;
   const VkExtent3D e = vk_image_mip_level_extent(&image->vk, d->view_level);
   const bool is_3d = vd.dimension == WGPUTextureViewDimension_3D;
   const uint32_t layers = vd.arrayLayerCount == WGPU_ARRAY_LAYER_COUNT_UNDEFINED
                              ? image->vk.array_layers - vd.baseArrayLayer
                              : vd.arrayLayerCount;
   WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
   td.label = wvk_sv("wvk scratch storage");
   td.usage = WGPUTextureUsage_StorageBinding;
   td.format = vd.format != WGPUTextureFormat_Undefined ? vd.format : image->format;
   td.dimension = is_3d ? WGPUTextureDimension_3D : WGPUTextureDimension_2D;
   td.size = (WGPUExtent3D){ e.width, e.height, is_3d ? e.depth : MAX2(layers, 1) };
   WGPUTexture texture = wgpuDeviceCreateTexture(dev->wgpu, &td);
   if (!texture)
      return NULL;
   vd.baseMipLevel = 0;
   vd.mipLevelCount = 1;
   vd.baseArrayLayer = 0;
   WGPUTextureView view = wgpuTextureCreateView(texture, &vd);
   wgpuTextureRelease(texture);
   return view;
}

/* Whether two writable storage views overlap: one of a texture's levels
 * and layers both, which WebGPU calls aliasing. */
static bool
storage_views_alias(const struct wvk_descriptor *a, const struct wvk_descriptor *b)
{
   if (!a->view_image || a->view_image != b->view_image || a->view_texture != b->view_texture)
      return false;
   const uint32_t a_levels = MAX2(a->view_levels, 1), b_levels = MAX2(b->view_levels, 1);
   if (a->view_level + a_levels <= b->view_level || b->view_level + b_levels <= a->view_level)
      return false;
   const uint32_t all = a->view_image->vk.array_layers;
   const uint32_t a0 = a->view_desc.baseArrayLayer, b0 = b->view_desc.baseArrayLayer;
   const uint32_t a1 = a->view_desc.arrayLayerCount == WGPU_ARRAY_LAYER_COUNT_UNDEFINED
                          ? all : a0 + a->view_desc.arrayLayerCount;
   const uint32_t b1 = b->view_desc.arrayLayerCount == WGPU_ARRAY_LAYER_COUNT_UNDEFINED
                          ? all : b0 + b->view_desc.arrayLayerCount;
   return a0 < b1 && b0 < a1;
}

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
      if (group == WVK_DRIVER_GROUP && b->binding >= WVK_DRIVER_SLOT_LOGIC_OP &&
          b->binding < WVK_DRIVER_SLOT_LOGIC_OP + WVK_MAX_LOGIC_OP_ATTACHMENTS) {
         /* The copy of the attachment a logic op reads (snapshot_logic_op). */
         const uint32_t rt = b->binding - WVK_DRIVER_SLOT_LOGIC_OP;
         e->textureView = replay && replay->logic_op_views[rt] ? replay->logic_op_views[rt] : placeholder_view(dev, b);
         continue;
      }
      if (group == WVK_DRIVER_GROUP && b->binding == WVK_DRIVER_SLOT_DEPTH_BOUNDS) {
         /* The copy of the depth attachment (prepare_depth_bounds). */
         e->textureView = replay && replay->depth_bounds_view ? replay->depth_bounds_view : placeholder_view(dev, b);
         continue;
      }
      if (group == WVK_DRIVER_GROUP && b->binding == WVK_DRIVER_SLOT_FSR) {
         e->textureView = replay && replay->fsr_view ? replay->fsr_view : placeholder_view(dev, b);
         continue;
      }
      if (group == WVK_DRIVER_GROUP && b->binding == WVK_DRIVER_SLOT_LOCKS) {
         e->buffer = dev->atomic_locks;
         e->size = WVK_ATOMIC_LOCKS * 4 + 4;
         continue;
      }
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
         /* None, or a null descriptor: the null buffer, as much of it as
          * a uniform block of the shader's wants. */
         if (!d || !d->buffer) {
            e->buffer = b->kind == WVK_BINDING_STORAGE ? dev->null_storage : dev->null_buffer;
            e->size = CLAMP(align64(b->min_size, 16), 256, 65536);
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
         /* A cube a variant samples a face at a time (key.non_seamless). */
         if (d && d->cube_faces && b->view_dimension == WGPUTextureViewDimension_2DArray)
            e->textureView = d->cube_faces;
         if (replay && b->kind == WVK_BINDING_TEXTURE) {
            const struct wvk_texture_swap *swap = find_texture_swap(replay, group, b->binding);
            if (swap) {
               e->textureView = swap->view;
               break;
            }
         }
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
   /* Writable storage bindings of one level of an image, which WebGPU
    * refuses: the later ones write to textures of their own. */
   for (uint32_t i = 0; set && i < bgl->count; ++i) {
      const struct wvk_shader_binding *b = &bgl->entries[i].b;
      if (b->kind != WVK_BINDING_STORAGE_TEXTURE || b->access == WGPUStorageTextureAccess_ReadOnly ||
          (group == WVK_DRIVER_GROUP && b->binding >= WVK_DRIVER_SLOT_BASE))
         continue;
      bool sampler;
      uint32_t dynamic_index;
      const struct wvk_descriptor *d = slot_descriptor(set, b->binding, &sampler, &dynamic_index);
      if (!d || !d->view || entries[i].textureView != d->view)
         continue;
      for (uint32_t j = 0; j < i; ++j) {
         const struct wvk_shader_binding *o = &bgl->entries[j].b;
         if (o->kind != WVK_BINDING_STORAGE_TEXTURE || o->access == WGPUStorageTextureAccess_ReadOnly)
            continue;
         const struct wvk_descriptor *od = slot_descriptor(set, o->binding, &sampler, &dynamic_index);
         if (!od || !od->view || !storage_views_alias(d, od))
            continue;
         WGPUTextureView v = scratch_storage_view(dev, d);
         if (v) {
            util_dynarray_append(&own_views, v);
            entries[i].textureView = v;
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
      else if (binding == WVK_DRIVER_SLOT_INDEX)
         key.index = d->serial;
      else if (binding == WVK_DRIVER_SLOT_CAPTURE)
         key.capture = d->serial;
      else if (binding == WVK_DRIVER_SLOT_GS_VERTICES)
         key.gs_vertices = d->serial;
      else if (binding == WVK_DRIVER_SLOT_GS_INDICES)
         key.gs_indices = d->serial;
      else if (binding >= WVK_DRIVER_SLOT_TESS_VERTICES && binding <= WVK_DRIVER_SLOT_TESS_COUNTS)
         key.tess[binding - WVK_DRIVER_SLOT_TESS_VERTICES] = d->serial;
      else if (binding == WVK_DRIVER_SLOT_STATS)
         key.stats = d->serial;
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
   /* The statistics query's counters, where one is active. */
   st->stats = replay->counters
                  ? (struct wvk_driver_buffer){ replay->counters, replay->counters_serial, 0, WVK_COUNTERS * 4 }
                  : (struct wvk_driver_buffer){ 0 };
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
      util_dynarray_foreach(&replay->texture_swaps, struct wvk_texture_swap, s)
         swapped |= s->group == g;
      /* And the driver's group of a draw whose logic op reads copies of
       * the attachments, which are new at every draw. */
      const bool logic_op = pass && g == WVK_DRIVER_GROUP && st->fs &&
                            (st->fs->logic_op_reads || st->fs->reads_depth_bounds || st->fs->key.fsr_attachment);
      const bool reads_copies = swapped || logic_op ||
                                (pass && replay->snapshots.size && bgl->count && bound->set &&
                                 reads_snapshot(replay, bgl, bound->set));
      WGPUBindGroup bg;
      if (reads_copies) {
         /* A group of the driver's bindings alone has no set. */
         if (bound->set)
            upload_inline_blocks(dev, bound->set);
         bg = make_bind_group(dev, bgl, bgl->count ? bound->set : NULL, g, st, replay);
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
               offsets[offset_count++] = b->binding == WVK_DRIVER_SLOT_LOCKS                       ? 0
                                         : d && (d->buffer || b->binding != WVK_DRIVER_SLOT_DRAW) ? d->offset
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
      /* One WebGPU has no format of, that the shader could not read
       * itself: the pipeline would be invalid, and with it the submit. */
      if (!f || !f->vertex) {
         k->unfetchable = true;
         continue;
      }
      if (a->binding >= MESA_VK_MAX_VERTEX_BINDINGS)
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
   /* A point or line is not culled, whichever way its quad's triangles face. */
   if (replay->draw_points || replay->draw_lines)
      k->cull_mode = WGPUCullMode_None;
   /* WebGPU clamps depth always, and clips it but where it is unclipped:
    * which Vulkan's depth clamp has, but where VK_EXT_depth_clip_enable
    * says otherwise. */
   k->unclipped_depth = !vk_rasterization_state_depth_clip_enable(&dyn->rs) &&
                        wvk_physical_device_has_feature(wvk_device_physical(wvk_cmd_buffer_device(cmd)),
                                                        WGPUFeatureName_DepthClipControl);

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
      /* A logic op is blending's place, for every attachment: Vulkan has
       * blending off while one is on, whether or not it applies. And none
       * where the fragment shader blends (lower_blend). */
      k->color[i].blend = a->blend_enable && written && !dyn->cb.logic_op_enable &&
                          !(st->fs->key.blend_mask & BITFIELD_BIT(i)) && a->color_blend_op < VK_BLEND_OP_ZERO_EXT;
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
      /* Which WebGPU has of triangles alone: and Vulkan not of points,
       * which the triangles of a point's quad are. */
      if (dyn->rs.depth_bias.enable && !replay->draw_points && !replay->draw_lines &&
          k->topology != WGPUPrimitiveTopology_LineList &&
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
   if (key->key.unfetchable) {
      mesa_logw_once("wvk: a draw of a vertex format neither WebGPU nor the vertex shader can read");
      free(key);
      return NULL;
   }
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
      /* A point's quad is its size in the viewport's pixels, down in
       * framebuffer y whatever the viewport's sign (expand_points). */
      if ((cmd->replay->draw_points || cmd->replay->draw_lines) && v.width && v.height) {
         const float scale[2] = { 1.0f / v.width, 1.0f / v.height };
         if (memcmp(st->push + WVK_DRIVER_POINT_SCALE, scale, sizeof(scale))) {
            memcpy(st->push + WVK_DRIVER_POINT_SCALE, scale, sizeof(scale));
            st->push_offset = UINT32_MAX;
         }
      }
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
      float depth[2] = { inverted ? -1.0f : 1.0f, inverted ? 1.0f : 0.0f };
      /* Clip space's z from -w to w (VK_EXT_depth_clip_control): (z + w) / 2
       * first, WebGPU's being from 0 to w. */
      if (dyn->vp.depth_clip_negative_one_to_one) {
         depth[1] += depth[0] * 0.5f;
         depth[0] *= 0.5f;
      }
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
set_pulled_buffers(struct wvk_device *dev, struct wvk_cmd_buffer *cmd, struct wvk_replay *replay,
                   const struct wvk_shader *vs)
{
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   uint32_t starts[WVK_MAX_PULLED_BINDINGS] = { 0 }, strides[WVK_MAX_PULLED_BINDINGS] = { 0 };
   for (uint32_t i = 0; i < vs->pulled_count; ++i) {
      const struct wvk_bound_buffer *vb = &st->vb[vs->pulled_bindings[i]];
      strides[i] = dyn->vi_binding_strides[vs->pulled_bindings[i]];
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
   /* The first four, and the rest past them. */
   const uint32_t more[4] = { starts[4], starts[5], strides[4], strides[5] };
   if (memcmp(st->push + WVK_DRIVER_PULL_OFFSET, starts, 16) ||
       memcmp(st->push + WVK_DRIVER_PULL_STRIDES, strides, 16) ||
       memcmp(st->push + WVK_DRIVER_PULL_MORE, more, sizeof(more))) {
      memcpy(st->push + WVK_DRIVER_PULL_OFFSET, starts, 16);
      memcpy(st->push + WVK_DRIVER_PULL_STRIDES, strides, 16);
      memcpy(st->push + WVK_DRIVER_PULL_MORE, more, sizeof(more));
      st->push_offset = UINT32_MAX;
   }
}

static bool
is_constant_alpha(uint8_t factor)
{
   return factor == VK_BLEND_FACTOR_CONSTANT_ALPHA || factor == VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
}

/* The blend constant, as WebGPU's one constant has to be: where colour
 * factors use CONSTANT_ALPHA, its alpha in every channel. For alpha
 * factors both are the alpha anyway. Where some use CONSTANT_COLOR too,
 * the fragment shader blends those that use CONSTANT_ALPHA itself
 * (lower_blend), with the constants the driver's data has. */
static WGPUColor
blend_constant(struct wvk_bound_state *st, const struct vk_dynamic_graphics_state *dyn, uint32_t color_count)
{
   const float *bc = dyn->cb.blend_constants;
   if (st->fs && st->fs->key.blend_mask) {
      if (memcmp(st->push + WVK_DRIVER_BLEND, bc, 16)) {
         memcpy(st->push + WVK_DRIVER_BLEND, bc, 16);
         st->push_offset = UINT32_MAX;
      }
      return (WGPUColor){ bc[0], bc[1], bc[2], bc[3] };
   }
   bool alpha = false;
   for (uint32_t i = 0; i < MIN2(color_count, dyn->cb.attachment_count); ++i) {
      const struct vk_color_blend_attachment_state *a = &dyn->cb.attachments[i];
      /* MIN and MAX take no factors. */
      if (a->blend_enable && a->color_blend_op != VK_BLEND_OP_MIN && a->color_blend_op != VK_BLEND_OP_MAX)
         alpha |= is_constant_alpha(a->src_color_blend_factor) || is_constant_alpha(a->dst_color_blend_factor);
   }
   const uint32_t written = st->fs ? st->fs->io_mask & dyn->cb.color_write_enables : 0;
   if (wvk_shader_blend_mask(dyn, written)) {
      mesa_logw_once("wvk: colour blend factors with both CONSTANT_COLOR and CONSTANT_ALPHA, not blended");
      return (WGPUColor){ bc[0], bc[1], bc[2], bc[3] };
   }
   if (alpha)
      return (WGPUColor){ bc[3], bc[3], bc[3], bc[3] };
   return (WGPUColor){ bc[0], bc[1], bc[2], bc[3] };
}

/* What a draw's logic op reads of the colour attachments: a copy of the
 * level and layer the pass renders to, made before the draw with the pass
 * begun again after -- for every such draw, since each has to see what the
 * one before it wrote. A 2D texture of the level's size, so its texels are
 * at the fragments' coordinates. */
static void
snapshot_logic_op(struct wvk_cmd_buffer *cmd)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   const struct wvk_shader *fs = replay->state->fs;
   if (!fs || !fs->logic_op_reads)
      return;
   wvk_replay_end_passes(replay);
   u_foreach_bit(i, fs->logic_op_reads) {
      replay->logic_op_views[i] = NULL;
      struct wvk_image_view *view = i < replay->rendering.color_count ? replay->rendering.color[i] : NULL;
      if (!view)
         continue;
      struct wvk_image *image = wvk_image_view_image(view);
      const uint32_t level = view->vk.base_mip_level;
      const uint32_t layer = view->vk.base_array_layer + replay->pass_layer;
      uint32_t texture_layer = layer, first_level = 0;
      WGPUTexture texture = view->texture != WVK_IMAGE_TEXTURE
                               ? wvk_image_texture(image, view->texture, &first_level)
                               : wvk_image_layer_texture(image, layer, &texture_layer);
      if (!texture)
         continue;
      const VkExtent3D e = vk_image_mip_level_extent(&image->vk, level);
      WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
      td.label = wvk_sv("wvk logic op destination");
      td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
      td.dimension = WGPUTextureDimension_2D;
      td.size = (WGPUExtent3D){ e.width, e.height, 1 };
      td.format = wgpuTextureGetFormat(texture);
      /* Seen as the pass sees it: an sRGB view of a texture that is not,
       * decoded, for blending. */
      if (view->format != td.format) {
         td.viewFormatCount = 1;
         td.viewFormats = &view->format;
      }
      WGPUTexture copy = wgpuDeviceCreateTexture(dev->wgpu, &td);
      if (!copy)
         continue;
      WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      src.texture = texture;
      src.mipLevel = level - first_level;
      src.origin.z = texture_layer;
      dst.texture = copy;
      wgpuCommandEncoderCopyTextureToTexture(replay->encoder, &src, &dst, &td.size);
      WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      vd.format = view->format;
      WGPUTextureView v = wgpuTextureCreateView(copy, td.viewFormatCount ? &vd : NULL);
      wgpuTextureRelease(copy);
      if (!v)
         continue;
      util_dynarray_append(&replay->views, v);
      replay->logic_op_views[i] = v;
   }
   wvk_replay_restart_pass(cmd, replay->pass_layer);
}

/* What a draw's discard rectangle test reads: the rectangles, as corners,
 * in the driver's data. */
static void
prepare_discard_rects(struct wvk_cmd_buffer *cmd)
{
   struct wvk_bound_state *st = cmd->replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!st->fs || (!st->fs->key.discard_rects && !st->fs->key.fsr))
      return;
   int32_t words[4 + 4 * WVK_MAX_DISCARD_RECTANGLES] = { 0 };
   const uint32_t count = MIN2(dyn->dr.rectangle_count, WVK_MAX_DISCARD_RECTANGLES);
   words[0] = (int32_t)count;
   words[1] = dyn->dr.mode == VK_DISCARD_RECTANGLE_MODE_EXCLUSIVE_EXT;
   /* And the pipeline's shading rate and its combiner ops, which coarse
    * shading combines at the fragment (wvk_nir_fsr.c). */
   if (st->fs->key.fsr) {
      const unsigned w = util_logbase2(CLAMP(dyn->fsr.fragment_size.width, 1, 4));
      const unsigned h = util_logbase2(CLAMP(dyn->fsr.fragment_size.height, 1, 4));
      words[2] = (int32_t)(((w << 2) | h) | dyn->fsr.combiner_ops[0] << 8 | dyn->fsr.combiner_ops[1] << 16);
   }
   for (uint32_t i = 0; i < count; ++i) {
      const VkRect2D *r = &dyn->dr.rectangles[i];
      words[4 + 4 * i] = r->offset.x;
      words[5 + 4 * i] = r->offset.y;
      words[6 + 4 * i] = (int32_t)MIN2((int64_t)r->offset.x + r->extent.width, INT32_MAX);
      words[7 + 4 * i] = (int32_t)MIN2((int64_t)r->offset.y + r->extent.height, INT32_MAX);
   }
   if (memcmp(st->push + WVK_DRIVER_DISCARD, words, sizeof(words))) {
      memcpy(st->push + WVK_DRIVER_DISCARD, words, sizeof(words));
      st->push_offset = UINT32_MAX;
   }
}

/* What a draw's depth bounds test reads: a copy of the depth the
 * attachment holds, in R32Float, made before the draw with the pass begun
 * again after; and the bounds, in the driver's data. */
static void
prepare_depth_bounds(struct wvk_cmd_buffer *cmd)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!st->fs || !st->fs->reads_depth_bounds)
      return;
   const float bounds[2] = { dyn->ds.depth.bounds_test.min, dyn->ds.depth.bounds_test.max };
   if (memcmp(st->push + WVK_DRIVER_DEPTH_BOUNDS, bounds, sizeof(bounds))) {
      memcpy(st->push + WVK_DRIVER_DEPTH_BOUNDS, bounds, sizeof(bounds));
      st->push_offset = UINT32_MAX;
   }
   replay->depth_bounds_view = NULL;
   const struct wvk_image_view *ds = replay->rendering.depth_stencil;
   if (!ds)
      return;
   WGPUTexture copy = wvk_image_snapshot_ds(dev, replay, wvk_image_view_image(ds), false);
   if (copy) {
      WGPUTextureViewDescriptor vd = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      vd.dimension = WGPUTextureViewDimension_2D;
      vd.baseMipLevel = ds->vk.base_mip_level;
      vd.mipLevelCount = 1;
      vd.baseArrayLayer = ds->vk.base_array_layer + replay->pass_layer;
      vd.arrayLayerCount = 1;
      WGPUTextureView v = wgpuTextureCreateView(copy, &vd);
      wgpuTextureRelease(copy);
      if (v) {
         util_dynarray_append(&replay->views, v);
         replay->depth_bounds_view = v;
      }
   }
   wvk_replay_restart_pass(cmd, replay->pass_layer);
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
   snapshot_logic_op(cmd);
   prepare_depth_bounds(cmd);
   prepare_discard_rects(cmd);

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
   /* Before the bind groups: they write the driver's data among the push
    * constants. */
   set_viewport_scissor(cmd, pass);
   const WGPUColor blend = blend_constant(st, dyn, replay->rendering.color_count);
   st->graphics_layout = p->layout;
   if (st->vs->pulled_count)
      set_pulled_buffers(dev, cmd, replay, st->vs);
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
   /* The lines of a topology with adjacency. */
   LIST_SEGMENTS,
   /* A list's whole primitives between restarts, as they are
    * (primitiveTopologyListRestart): WebGPU restarts strips only. */
   LIST_COMPACT,
};

static const char triangles_wgsl[] =
   "struct P { first: u32, wide: u32, count: u32, restart: u32,\n"
   "           topology: u32, mode: u32, indexed: u32, triangles: u32,\n"
   "           last: u32, counted: u32, instances: u32, base_vertex: u32,\n"
   "           first_instance: u32, pad0: u32, pad1: u32, pad2: u32 }\n"
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
   /* With adjacency, without the adjacent: a list's, and a strip's, of
    * Vulkan's table -- the same for its first and last primitives. */
   "  if (p.topology == 5u) { return vec3u(t * 6u, t * 6u + 2u, t * 6u + 4u); }\n"
   "  if (p.topology == 6u) {\n"
   "    if ((t & 1u) != 0u) { return vec3u(t * 2u + 2u, t * 2u, t * 2u + 4u); }\n"
   "    return vec3u(t * 2u, t * 2u + 2u, t * 2u + 4u);\n"
   "  }\n"
   "  return vec3u(t + 1u, t + 2u, 0u);\n"
   "}\n"
   /* Of the last vertex's flat varyings (VK_EXT_provoking_vertex): a
    * triangle's corners turned round to begin at its last -- a strip's
    * third, a fan's second, a list's last -- as WebGPU's are the first's. */
   "fn provoke(c: vec3u, t: u32) -> vec3u {\n"
   "  if (p.last == 0u) { return c; }\n"
   "  if (p.topology == 1u && (t & 1u) != 0u) { return vec3u(c.y, c.z, c.x); }\n"
   "  if (p.topology == 2u) { return vec3u(c.y, c.z, c.x); }\n"
   "  return vec3u(c.z, c.x, c.y);\n"
   "}\n"
   /* A line's ends, of a list and a strip with adjacency, and of a list
    * and a strip; the other way round of the last vertex's. */
   "fn ends(t: u32) -> vec2u {\n"
   "  var e = vec2u(t + 1u, t + 2u);\n"
   "  if (p.topology == 3u) { e = vec2u(t * 4u + 1u, t * 4u + 2u); }\n"
   "  if (p.topology == 7u) { e = vec2u(t * 2u, t * 2u + 1u); }\n"
   "  if (p.topology == 8u) { e = vec2u(t, t + 1u); }\n"
   "  return select(e, e.yx, p.last != 0u);\n"
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
   /* The new vertex is the last of a strip's and a fan's triangle. */
   "      if (p.last != 0u) {\n"
   "        t = select(vec3u(v, a, b), vec3u(v, b, a), p.topology == 1u && ((n - 2u) & 1u) != 0u);\n"
   "      }\n"
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
   /* And a strip with adjacency: each run between restarts a strip of
    * its own, whose primitives' vertices are the same of any. */
   "fn restarting_adjacent() {\n"
   "  var out = 0u;\n"
   "  var begin = 0u;\n"
   "  for (var i = 0u; i <= p.count; i++) {\n"
   "    if (i < p.count && index(i) != p.restart) { continue; }\n"
   "    let len = i - begin;\n"
   "    if (p.topology == 4u) {\n"
   "      for (var t = 0u; t + 3u < len; t++) {\n"
   "        let e = select(vec2u(t + 1u, t + 2u), vec2u(t + 2u, t + 1u), p.last != 0u);\n"
   "        dst[out * 2u] = index(begin + e.x); dst[out * 2u + 1u] = index(begin + e.y); out++;\n"
   "      }\n"
   "    } else if (len >= 6u) {\n"
   "      for (var t = 0u; t < (len - 4u) / 2u; t++) {\n"
   "        let c = provoke(corners(t), t);\n"
   "        emit(out, vec3u(index(begin + c.x), index(begin + c.y), index(begin + c.z))); out++;\n"
   "      }\n"
   "    }\n"
   "    begin = i + 1u;\n"
   "  }\n"
   "  for (; out < p.triangles; out++) {\n"
   "    if (p.mode == 3u) { dst[out * 2u] = 0u; dst[out * 2u + 1u] = 0u; } else { emit(out, vec3u(0u)); }\n"
   "  }\n"
   "}\n"
   /* A list's primitive's vertices: a point's, a line's, a triangle's, or
    * with adjacency a line's or a triangle's. And the indices the driver's
    * list has of each. */
   "fn vertices() -> u32 {\n"
   "  if (p.topology == 9u) { return 1u; }\n"
   "  if (p.topology == 7u) { return 2u; }\n"
   "  if (p.topology == 3u) { return 4u; }\n"
   "  if (p.topology == 5u) { return 6u; }\n"
   "  return 3u;\n"
   "}\n"
   "fn per() -> u32 {\n"
   "  if (p.mode == 1u) { return 6u; }\n"
   "  if (p.mode == 3u) { return 2u; }\n"
   "  return 3u;\n"
   "}\n"
   /* An indexed indirect draw of `n` of the list's indices, after its
    * `at` of them. */
   "fn draw(at: u32, n: u32) {\n"
   "  dst[at] = n; dst[at + 1u] = p.instances; dst[at + 2u] = 0u;\n"
   "  dst[at + 3u] = p.base_vertex; dst[at + 4u] = p.first_instance;\n"
   "}\n"
   /* A list that restarts: one invocation goes through it, a primitive
    * begun again after each restart index, the whole ones written. */
   "fn compacting() {\n"
   "  let k = vertices();\n"
   "  var v: array<u32, 6>;\n"
   "  var n = 0u;\n"
   "  var out = 0u;\n"
   "  for (var i = 0u; i < p.count; i++) {\n"
   "    let x = index(i);\n"
   "    if (x == p.restart) { n = 0u; continue; }\n"
   "    v[n] = x;\n"
   "    n++;\n"
   "    if (n == k) {\n"
   "      for (var j = 0u; j < k; j++) { dst[out + j] = v[j]; }\n"
   "      out += k;\n"
   "      n = 0u;\n"
   "    }\n"
   "  }\n"
   "  let total = p.triangles * k;\n"
   "  for (var j = out; j < total; j++) { dst[j] = 0xffffffffu; }\n"
   "  draw(total, out);\n"
   "}\n"
   "@compute @workgroup_size(64)\n"
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"
   "  let t = id.x + id.y * 65535u * 64u;\n"
   "  if (p.mode == 4u) {\n"
   "    if (t == 0u) { compacting(); }\n"
   "    return;\n"
   "  }\n"
   "  if (p.restart != 0u && (p.topology == 1u || p.topology == 2u)) {\n"
   "    if (t == 0u) { restarting(); }\n"
   "    return;\n"
   "  }\n"
   "  if (p.restart != 0u && (p.topology == 4u || p.topology == 6u)) {\n"
   "    if (t == 0u) { restarting_adjacent(); }\n"
   "    return;\n"
   "  }\n"
   /* Of a list made so: its whole primitives, which its draw has, and a
    * draw of as many of this one's. */
   "  var whole = p.triangles;\n"
   "  if (p.counted != 0u) {\n"
   "    whole = src[p.first / 4u + p.count] / vertices();\n"
   "    if (t == 0u) { draw(p.triangles * per(), whole * per()); }\n"
   "  }\n"
   "  if (t >= whole) { return; }\n"
   "  if (p.mode == 3u) {\n"
   "    let e = ends(t);\n"
   "    dst[t * 2u] = index(e.x); dst[t * 2u + 1u] = index(e.y);\n"
   "    return;\n"
   "  }\n"
   "  let c = provoke(corners(t), t);\n"
   "  emit(t, vec3u(index(c.x), index(c.y), index(c.z)));\n"
   "}\n";

/* The primitives a list is made of: triangles, or the lines of a
 * topology with adjacency. */
static uint32_t
list_triangles(VkPrimitiveTopology topology, uint32_t count)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return count;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return count / 3;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return count / 2;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return count >= 2 ? count - 1 : 0;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: return count / 4;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return count >= 4 ? count - 3 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY: return count / 6;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: return count >= 6 ? (count - 4) / 2 : 0;
   default: return count >= 3 ? count - 2 : 0;
   }
}

/* The vertices of a primitive of a list, for LIST_COMPACT. */
static uint32_t
list_vertices(VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return 1;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return 2;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: return 4;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY: return 6;
   default: return 3;
   }
}

/* A list of a draw's triangles, their edges or their corners, from the
 * bound index buffer's indices from `first` on, or of the vertices in
 * order: made before the pass goes on. Of a list the driver made of one
 * that restarts (LIST_COMPACT, or the bound index buffer one), an indexed
 * indirect draw of its `draw` -- instances, base vertex, first instance --
 * after it, at `*args`; else that is UINT64_MAX. */
static WGPUBuffer
triangles_on_gpu(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, enum list_mode mode,
                 bool last, const uint32_t draw[3], uint32_t *index_count, uint64_t *args)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const VkPrimitiveTopology topology = dyn->ia.primitive_topology;
   const uint32_t triangles = list_triangles(topology, count);
   *args = UINT64_MAX;
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
   /* Of the driver's list of a list that restarts: the draw after it too. */
   const bool counted = indexed && replay->restart_list && ib->buffer == replay->restart_list;
   uint64_t start = 0, base = 0, end = 4;
   if (indexed) {
      start = ib->offset + (uint64_t)first * (wide ? 4 : 2);
      base = start & ~255ull;
      end = start + (uint64_t)count * (wide ? 4 : 2) + (counted ? 20 : 0);
   }
   const uint32_t per = mode == LIST_EDGES      ? 6
                        : mode == LIST_SEGMENTS ? 2
                        : mode == LIST_COMPACT  ? list_vertices(topology)
                                                : 3;
   const uint64_t out_bytes = (uint64_t)triangles * per * 4;
   const bool drawn = counted || mode == LIST_COMPACT;
   WGPUBuffer out = wvk_replay_temp_buffer(dev, replay, out_bytes + (drawn ? 20 : 0),
                                           WGPUBufferUsage_Storage | WGPUBufferUsage_Index |
                                              (drawn ? WGPUBufferUsage_Indirect : 0),
                                           NULL);
   /* The restart index, or 0 for none: no index is 0 and restarts. */
   const uint32_t restart = indexed && dyn->ia.primitive_restart_enable ? (wide ? 0xffffffffu : 0xffffu) : 0;
   const uint32_t kind = topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST                    ? 0
                         : topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP                 ? 1
                         : topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY       ? 3
                         : topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY      ? 4
                         : topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY   ? 5
                         : topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY  ? 6
                         : topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST                      ? 7
                         : topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP                     ? 8
                         : topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST                     ? 9
                                                                                           : 2;
   const uint32_t params[16] = {
      (uint32_t)(start - base), wide, count, restart, kind, mode, indexed, triangles, last, counted,
      draw ? draw[0] : 0, draw ? draw[1] : 0, draw ? draw[2] : 0,
   };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   if (!out || !params_buffer)
      return NULL;
   if (indexed)
      wvk_replay_read(replay, ib->mem, start, end - start);
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = indexed ? ib->buffer : dev->null_buffer, .offset = base,
        .size = indexed ? align64(end - base, 4) : 256 },
      { .binding = 1, .buffer = out, .offset = 0, .size = out_bytes + (drawn ? 20 : 0) },
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
   const uint32_t groups = mode == LIST_COMPACT ? 1 : DIV_ROUND_UP(triangles, 64);
   wgpuComputePassEncoderDispatchWorkgroups(pass, MIN2(groups, 65535), DIV_ROUND_UP(groups, 65535), 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   *index_count = triangles * per;
   if (drawn)
      *args = out_bytes;
   return out;
}

static bool conditional_args(struct wvk_cmd_buffer *cmd, const uint32_t *host, const struct wvk_bound_buffer *src,
                             uint32_t records, uint32_t words, uint32_t stride, struct wvk_bound_buffer *out);

/* A draw the driver makes of a command under a predicate the GPU has
 * (wvk_cmd_condition): its arguments, zero where the predicate fails,
 * for an indirect draw of them -- made before its pass state is, as a
 * pass is ended for them. NULL for one that is drawn as it is. */
static WGPUBuffer
predicated_draw(struct wvk_cmd_buffer *cmd, const uint32_t *args, uint32_t words)
{
   struct wvk_bound_buffer b;
   if (!cmd->replay->cond_gpu || !conditional_args(cmd, args, NULL, 1, words, words * 4, &b))
      return NULL;
   return b.buffer;
}

/* A draw of a list the driver made, `indices` its, as the bound index
 * buffer for the draw: of `index_count` indices, or where `args` is not
 * UINT64_MAX, as many as the indirect draw there in `indices` has. */
static void
draw_list_at(struct wvk_cmd_buffer *cmd, WGPUBuffer indices, uint64_t offset, uint32_t index_count,
             WGPUPrimitiveTopology topology, uint32_t instances, int32_t base_vertex, uint32_t first_instance,
             bool indexed, uint32_t draw_id, uint64_t args)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct wvk_bound_buffer ib = st->ib;
   const VkIndexType type = st->index_type;
   st->ib = (struct wvk_bound_buffer){ NULL, indices, offset, (uint64_t)index_count * 4 };
   st->index_type = VK_INDEX_TYPE_UINT32;
   replay->list_topology = topology;
   WGPUBuffer predicated = NULL;
   if (args == UINT64_MAX) {
      predicated = predicated_draw(
         cmd, (const uint32_t[5]){ index_count, instances, 0, (uint32_t)base_vertex, first_instance }, 5);
   } else if (replay->cond_gpu) {
      const struct wvk_bound_buffer src = { NULL, indices, args, 20,
                                            ++wvk_cmd_buffer_device(cmd)->temp_serial | (1ull << 62) };
      struct wvk_bound_buffer b;
      if (conditional_args(cmd, NULL, &src, 1, 5, 20, &b))
         predicated = b.buffer;
   }
   foreach_viewport(cmd) {
      set_draw_parameters(st, NULL, 0, indexed, base_vertex, first_instance, draw_id);
      if (!flush_draw(cmd, true))
         continue;
      if (predicated)
         wgpuRenderPassEncoderDrawIndexedIndirect(replay->pass, predicated, 0);
      else if (args != UINT64_MAX)
         wgpuRenderPassEncoderDrawIndexedIndirect(replay->pass, indices, args);
      else
         wgpuRenderPassEncoderDrawIndexed(replay->pass, index_count, instances, 0, base_vertex, first_instance);
   }
   replay->list_topology = WGPUPrimitiveTopology_Undefined;
   st->ib = ib;
   st->index_type = type;
}

static void
draw_list(struct wvk_cmd_buffer *cmd, WGPUBuffer indices, uint32_t index_count, WGPUPrimitiveTopology topology,
          uint32_t instances, int32_t base_vertex, uint32_t first_instance, bool indexed, uint32_t draw_id,
          uint64_t args)
{
   draw_list_at(cmd, indices, 0, index_count, topology, instances, base_vertex, first_instance, indexed, draw_id,
                args);
}

static void
draw_fan(struct wvk_cmd_buffer *cmd, WGPUBuffer indices, uint32_t triangles, uint32_t instances,
         int32_t base_vertex, uint32_t first_instance, bool indexed, uint32_t draw_id)
{
   draw_list(cmd, indices, triangles * 3, WGPUPrimitiveTopology_TriangleList, instances, base_vertex, first_instance,
             indexed, draw_id, UINT64_MAX);
}

/* Whether a draw's flat varyings are its primitives' last vertices'
 * (VK_EXT_provoking_vertex), which WebGPU's are the first's: its lists
 * are made to begin at their last vertices (triangles_wgsl's provoke). Of
 * a fragment shader that has flat varyings only. */
static bool
provokes_last(const struct wvk_cmd_buffer *cmd)
{
   const struct wvk_bound_state *st = cmd->replay->state;
   if (cmd->vk.dynamic_graphics_state.rs.provoking_vertex != VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT || !st->fs)
      return false;
   for (unsigned i = 0; i < WVK_MAX_VARYINGS; ++i) {
      if (st->fs->varyings[i] & (WVK_VARYING_FLAT | WVK_VARYING_UINT | WVK_VARYING_SINT))
         return true;
   }
   return false;
}

/* Of the polygon mode, lines or points; false where it is fill, or the
 * draw not of triangles. And the lists of filled triangles and of lines
 * whose last vertices provoke, where `provoking`. */
static bool
polygon_list(const struct wvk_cmd_buffer *cmd, enum list_mode *mode, WGPUPrimitiveTopology *topology, bool provoking)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const VkPrimitiveTopology t = dyn->ia.primitive_topology;
   /* Topologies with adjacency, which WebGPU has not, drawn without it:
    * lines, and triangles as the polygon mode has them. */
   if (t == VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY || t == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY) {
      *mode = LIST_SEGMENTS;
      *topology = WGPUPrimitiveTopology_LineList;
      return true;
   }
   const bool adjacent = t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY ||
                         t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
   if (provoking && dyn->rs.polygon_mode == VK_POLYGON_MODE_FILL && provokes_last(cmd)) {
      /* Lines, but strips that restart, which the list makes none of. */
      if (t == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
          (t == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP && !dyn->ia.primitive_restart_enable)) {
         *mode = LIST_SEGMENTS;
         *topology = WGPUPrimitiveTopology_LineList;
         return true;
      }
      if (t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || t == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) {
         *mode = LIST_TRIANGLES;
         *topology = WGPUPrimitiveTopology_TriangleList;
         return true;
      }
   }
   if ((dyn->rs.polygon_mode == VK_POLYGON_MODE_FILL && !adjacent) ||
       (t != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST && t != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP &&
        t != VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN && !adjacent))
      return false;
   *mode = dyn->rs.polygon_mode == VK_POLYGON_MODE_POINT  ? LIST_CORNERS
           : dyn->rs.polygon_mode == VK_POLYGON_MODE_LINE ? LIST_EDGES
                                                          : LIST_TRIANGLES;
   *topology = *mode == LIST_CORNERS  ? WGPUPrimitiveTopology_PointList
               : *mode == LIST_EDGES ? WGPUPrimitiveTopology_LineList
                                     : WGPUPrimitiveTopology_TriangleList;
   return true;
}

static bool
draw_polygon_mode(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, uint32_t instances,
                  int32_t base_vertex, uint32_t first_instance, uint32_t draw_id)
{
   enum list_mode mode;
   WGPUPrimitiveTopology topology;
   if (!polygon_list(cmd, &mode, &topology, true))
      return false;
   uint32_t index_count = 0;
   uint64_t args;
   WGPUBuffer indices = triangles_on_gpu(cmd, indexed, first, count, mode, provokes_last(cmd),
                                         (const uint32_t[3]){ instances, (uint32_t)base_vertex, first_instance },
                                         &index_count, &args);
   if (indices)
      draw_list(cmd, indices, index_count, topology, instances, base_vertex, first_instance, indexed, draw_id, args);
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
   /* Of the last vertex's flat varyings: each triangle from its second
    * (triangles_wgsl's provoke). */
   const bool last = provokes_last(cmd);
   for (uint32_t t = 0; t < triangles; ++t) {
      list[t * 3] = last ? t + 2 : t + 1;
      list[t * 3 + 1] = last ? 0 : t + 2;
      list[t * 3 + 2] = last ? t + 1 : 0;
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
   uint64_t args;
   WGPUBuffer indices = triangles_on_gpu(cmd, true, first_index, index_count, LIST_TRIANGLES, provokes_last(cmd), NULL,
                                         &count, &args);
   if (indices)
      draw_list(cmd, indices, count, WGPUPrimitiveTopology_TriangleList, instances, vertex_offset, first_instance,
                true, draw_id, args);
}

/* A draw, the `draw_id`th of a vkCmdDrawMulti*EXT. */
/* The bound shaders, or their variants for the state the draw has: the
 * fragment shader for its multisampling, the vertex shader for its vertex
 * input and for that fragment shader (wvk_shader_draw_variant). */
/* What of the descriptors bound a shader's variant is of (struct
 * wvk_bound_views): the formats of the views where it has storage images
 * without one (wvk_shader::unformatted), 0 where none is; which of its
 * cube samplings are of non-seamless samplers (wvk_shader::cube_samples);
 * and which of its 2D textures and images are 2D views of 3D images. */
static const struct wvk_descriptor *
bound_descriptor(const struct wvk_bound_state *st, uint32_t bind_point, uint32_t at)
{
   const uint32_t set = at >> 10, slot = at & 0x3ff;
   bool sampler;
   uint32_t dynamic_index;
   return set < WVK_MAX_GROUPS ? slot_descriptor(st->sets[bind_point][set].set, slot, &sampler, &dynamic_index) : NULL;
}

static void
bound_views(const struct wvk_bound_state *st, uint32_t bind_point, const struct wvk_shader *shader,
            struct wvk_bound_views *views)
{
   memset(views, 0, sizeof(*views));
   if (!shader)
      return;
   for (uint32_t i = 0; i < shader->unformatted_count; ++i) {
      const struct wvk_descriptor *d = bound_descriptor(st, bind_point, shader->unformatted[i]);
      if (d && d->view)
         views->image_formats[i] = d->view_format;
   }
   for (uint32_t i = 0; i < shader->cube_sample_count; ++i) {
      const struct wvk_descriptor *d = bound_descriptor(st, bind_point, shader->cube_samples[i] >> 16);
      if (d && (d->sampler_meta[WVK_META_SAMPLER_FLAGS] & WVK_SAMPLER_NON_SEAMLESS))
         views->non_seamless |= BITFIELD_BIT(i);
   }
   for (uint32_t i = 0; i < shader->flat_view_count; ++i) {
      const struct wvk_descriptor *d = bound_descriptor(st, bind_point, shader->flat_views[i]);
      if (d && d->of_3d)
         views->views_3d |= BITFIELD_BIT(i);
   }
}

static bool lines_placed(const struct wvk_cmd_buffer *cmd);

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
   const bool has_depth = replay->rendering.depth_format != WGPUTextureFormat_Undefined;
   VkFormat color_formats[8] = { VK_FORMAT_UNDEFINED };
   for (uint32_t i = 0; i < MIN2(replay->rendering.color_count, 8); ++i) {
      const struct wvk_image_view *view = replay->rendering.color[i];
      color_formats[i] = view ? view->vk.view_format : VK_FORMAT_UNDEFINED;
   }
   struct wvk_bound_views fs_images, vs_images;
   bound_views(st, WVK_BIND_GRAPHICS, st->bound_fs, &fs_images);
   bound_views(st, WVK_BIND_GRAPHICS, st->bound_vs, &vs_images);
   /* And the pass's shading rate attachment's texel size. */
   const VkExtent2D texel = replay->rendering.fsr_texel;
   fs_images.fsr_attachment =
      replay->fsr_view && texel.width && texel.height
         ? ((util_logbase2(texel.width) << 4) | util_logbase2(texel.height)) + 1
         : 0;
   const enum wvk_draw_shape shape = replay->draw_points  ? WVK_DRAW_POINTS
                                     : replay->draw_lines ? (lines_placed(cmd) ? WVK_DRAW_LINES_PLACED : WVK_DRAW_LINES)
                                                          : WVK_DRAW_PRIMITIVES;
   /* After a geometry or tessellation evaluation shader, whose outputs the
    * fragment shader reads. */
   const struct wvk_shader *before = replay->draw_mesh ? st->bound_mesh
                                     : st->bound_gs  ? st->bound_gs
                                     : st->bound_tes ? st->bound_tes
                                                     : st->bound_vs;
   st->fs = wvk_shader_draw_variant(dev, st->bound_fs, before, dyn, samples, has_depth, color_formats, &fs_images,
                                    shape, 0, wvk_replay_stats(replay));
   st->vs = wvk_shader_draw_variant(dev, st->bound_vs, st->fs, dyn, samples, has_depth, color_formats, &vs_images,
                                    shape, 0, 0);
   /* No fragment shader to do the depth bounds test: the driver's. And the
    * discard rectangle test. */
   if (!st->fs && dyn->ds.depth.bounds_test.enable && has_depth && samples <= 1)
      st->fs = dev->depth_bounds_fs;
   else if (!st->fs && dyn->dr.enable)
      st->fs = dev->discard_fs;
}

/* Whether a draw's points are quads of the size the vertex shader gives
 * them (largePoints): where it gives them one, and can pull what it reads
 * (expand_points), which it can of so many vertex buffers. */
static bool run_vertices(struct wvk_cmd_buffer *cmd, unsigned runs, bool indexed, uint32_t first, uint32_t count,
                         uint32_t instances, int32_t base_vertex, uint32_t first_instance, uint32_t draw_id);
static void dispatch_graphics(struct wvk_cmd_buffer *cmd, struct wvk_shader *cs, uint64_t invocations);
static void dispatch_graphics_groups(struct wvk_cmd_buffer *cmd, struct wvk_shader *cs, uint64_t groups,
                                     WGPUBuffer indirect, uint64_t offset);

/* Whether the vertex shader can pull every attribute it reads itself: of
 * formats it can read, of no more vertex buffers than it can bind. */
static bool
pulls_every_attribute(const struct wvk_cmd_buffer *cmd, const char *what)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct wvk_bound_state *st = cmd->replay->state;
   uint32_t bindings = 0;
   u_foreach_bit(location, dyn->vi->attributes_valid & st->bound_vs->attribs_read) {
      const struct vk_vertex_attribute_state *a = &dyn->vi->attributes[location];
      if (!wvk_format_vertex_pullable(a->format)) {
         mesa_logw_once("wvk: %s of an attribute the shader cannot read itself, drawn a pixel wide", what);
         return false;
      }
      bindings |= BITFIELD_BIT(a->binding);
   }
   if (util_bitcount(bindings) > WVK_MAX_PULLED_BINDINGS) {
      mesa_logw_once("wvk: %s of more than %u vertex buffers, drawn a pixel wide", what, WVK_MAX_PULLED_BINDINGS);
      return false;
   }
   return true;
}

static struct wvk_shader *draw_gs(struct wvk_cmd_buffer *cmd);

static bool
draws_points(struct wvk_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct wvk_bound_state *st = cmd->replay->state;
   if (dyn->rs.rasterizer_discard_enable)
      return false;
   /* After a geometry or evaluation shader: its points, where it gives
    * them a size (shaderTessellationAndGeometryPointSize). */
   const struct wvk_shader *gs = draw_gs(cmd), *tcs = st->bound_tcs, *tes = st->bound_tes;
   if (gs)
      return gs->gs_output == MESA_PRIM_POINTS && gs->writes_point_size;
   if (tes)
      return (tes->tess_point_mode || (tcs && tcs->tess_point_mode)) && tes->writes_point_size;
   if (dyn->ia.primitive_topology != VK_PRIMITIVE_TOPOLOGY_POINT_LIST || !st->bound_vs ||
       !st->bound_vs->writes_point_size)
      return false;
   return pulls_every_attribute(cmd, "points");
}

/* Whether a draw's lines are quads of their width (wideLines): lines wider
 * than a pixel, whose vertex shader can pull what it reads (expand_lines). */
static bool
draws_lines(const struct wvk_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct wvk_bound_state *st = cmd->replay->state;
   const VkPrimitiveTopology t = dyn->ia.primitive_topology;
   /* Wide ones, and any that are rectangles or stippled
    * (VK_KHR_line_rasterization): WebGPU's are one pixel, and plain. */
   const bool quads = dyn->rs.line.width > 1.0f || dyn->rs.line.stipple.enable ||
                      dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR_KHR ||
                      dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_KHR;
   if ((t != VK_PRIMITIVE_TOPOLOGY_LINE_LIST && t != VK_PRIMITIVE_TOPOLOGY_LINE_STRIP) || !st->bound_vs || !quads ||
       dyn->rs.rasterizer_discard_enable)
      return false;
   return pulls_every_attribute(cmd, "wide lines");
}

/* Whether lines drawn as quads have fragments that know where on the line
 * they are: stippled, or smooth (lower_line_fs). */
static bool
lines_placed(const struct wvk_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   return dyn->rs.line.stipple.enable || dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_KHR;
}

/* A draw of wide lines as quads (draws_lines): the vertex shader run as
 * compute capturing every vertex's position (run_vertices), then a
 * triangle list of six vertices a segment, which reads the other end's. */
static void
draw_lines(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, uint32_t instances,
           int32_t base_vertex, uint32_t first_instance, uint32_t draw_id)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const bool strip = dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
   const uint32_t segments = strip ? (count > 1 ? count - 1 : 0) : count / 2;
   const uint64_t bytes = (uint64_t)count * instances * 16;
   if (!segments || !instances)
      return;
   if ((uint64_t)segments * 6 > UINT32_MAX || bytes > pdev->limits.maxStorageBufferBindingSize) {
      mesa_logw_once("wvk: a draw of more wide lines than their positions fit a binding");
      return;
   }
   WGPUBuffer capture = wvk_replay_temp_buffer(dev, replay, MAX2(bytes, 16), WGPUBufferUsage_Storage, NULL);
   if (!capture)
      return;
   st->capture = (struct wvk_driver_buffer){ capture, ++dev->temp_serial | (1ull << 62), 0, MAX2(bytes, 16) };
   if (!run_vertices(cmd, 2, indexed, first, count, instances, base_vertex, first_instance, draw_id)) {
      mesa_logw_once("wvk: wide lines whose vertex shader could not be made a compute shader");
      return;
   }
   const float width = dyn->rs.line.width;
   uint32_t line[4];
   memcpy(&line[0], &width, 4);
   line[1] = strip;
   line[2] = dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_KHR ? 2
             : dyn->rs.line.mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR_KHR      ? 1
                                                                                     : 0;
   line[3] = dyn->rs.line.stipple.enable ? dyn->rs.line.stipple.factor << 16 | dyn->rs.line.stipple.pattern : 0;
   if (memcmp(st->push + WVK_DRIVER_LINE, line, sizeof(line))) {
      memcpy(st->push + WVK_DRIVER_LINE, line, sizeof(line));
      st->push_offset = UINT32_MAX;
   }
   replay->list_topology = WGPUPrimitiveTopology_TriangleList;
   WGPUBuffer predicated = predicated_draw(cmd, (const uint32_t[4]){ segments * 6, instances, 0, first_instance }, 4);
   foreach_viewport(cmd) {
      set_draw_parameters(st, NULL, 0, indexed, indexed ? base_vertex : (int32_t)first, first_instance, draw_id);
      if (!flush_draw(cmd, false))
         continue;
      if (predicated)
         wgpuRenderPassEncoderDrawIndirect(replay->pass, predicated, 0);
      else
         wgpuRenderPassEncoderDraw(replay->pass, segments * 6, instances, 0, first_instance);
   }
   replay->list_topology = WGPUPrimitiveTopology_Undefined;
}

/* The index buffer as storage at WVK_DRIVER_SLOT_INDEX, for a vertex
 * shader that reads an indexed draw's indices itself (expand_points,
 * run_vertices), and where they start, their size and whether they
 * restart in the driver's data; none, of size 0, for a draw without.
 * False where there is none bound. */
static bool
bind_index_storage(struct wvk_cmd_buffer *cmd, bool indexed)
{
   const struct wvk_physical_device *pdev = wvk_device_physical(wvk_cmd_buffer_device(cmd));
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   uint32_t words[3] = { 0, 0, indexed && cmd->vk.dynamic_graphics_state.ia.primitive_restart_enable };
   st->index = (struct wvk_driver_buffer){ 0 };
   if (indexed) {
      const struct wvk_bound_buffer *ib = &st->ib;
      if (!ib->buffer)
         return false;
      /* A buffer of the driver's own, which has no memory: all of it. */
      if (!ib->mem) {
         st->index = (struct wvk_driver_buffer){ ib->buffer, ib->serial, 0, ib->offset + ib->size };
         words[0] = (uint32_t)ib->offset;
         words[1] = st->index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2;
         if (memcmp(st->push + WVK_DRIVER_POINT_INDEX, words, sizeof(words))) {
            memcpy(st->push + WVK_DRIVER_POINT_INDEX, words, sizeof(words));
            st->push_offset = UINT32_MAX;
         }
         return true;
      }
      const uint64_t whole = ib->mem->buffer_size;
      const uint64_t size = MIN2(whole, pdev->limits.maxStorageBufferBindingSize) & ~3ull;
      const uint64_t offset = MIN2(ib->offset, whole - size) & ~255ull;
      st->index = (struct wvk_driver_buffer){ ib->mem->buffer, ib->mem->serial, (uint32_t)offset, size };
      words[0] = (uint32_t)(ib->offset - offset);
      words[1] = st->index_type == VK_INDEX_TYPE_UINT32  ? 4
                 : st->index_type == VK_INDEX_TYPE_UINT8 ? 1
                                                         : 2;
      wvk_replay_read(replay, ib->mem, ib->offset, ib->size);
   }
   if (memcmp(st->push + WVK_DRIVER_POINT_INDEX, words, sizeof(words))) {
      memcpy(st->push + WVK_DRIVER_POINT_INDEX, words, sizeof(words));
      st->push_offset = UINT32_MAX;
   }
   return true;
}

static struct wvk_compute_pipeline *get_compute_pipeline(struct wvk_device *dev, const struct wvk_bound_state *st,
                                                         struct wvk_shader *cs, uint32_t bind_point);
static void swap_sampled_storage(struct wvk_cmd_buffer *cmd, const struct wvk_layout *layout, uint32_t bind_point);

/* A draw's vertex shader's storage writes, which WebGPU's vertex shaders
 * cannot do (vertexPipelineStoresAndAtomics): its compute shader, an
 * invocation a vertex of each instance (run_vertices), dispatched with the
 * draw's sets before the draw, the pass ended and begun again after. The
 * draw's own vertex shader has them stripped. Once a draw: not again in
 * the other layers of a pass drawn a layer at a time. */
static void
run_vertex_writes(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, uint32_t instances,
                  int32_t base_vertex, uint32_t first_instance, uint32_t draw_id)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   if (!st->bound_vs || !st->bound_vs->writes_storage || !count || !instances || replay->pass_idle ||
       (replay->pass_layered && !replay->pass_multiview && replay->pass_layer))
      return;
   const uint64_t total = (uint64_t)count * instances;
   if (total > UINT32_MAX) {
      mesa_logw_once("wvk: a vertex shader's storage writes for more vertices than can be numbered");
      return;
   }
   if (!run_vertices(cmd, 1, indexed, first, count, instances, base_vertex, first_instance, draw_id))
      mesa_logw_once("wvk: a vertex shader's storage writes, which could not be made a compute shader");
}

/* The draw's vertex shader run as compute (run_vertices in wvk_shader.c),
 * as `runs` says: for its storage writes, or capturing its positions. */
static bool
run_vertices(struct wvk_cmd_buffer *cmd, unsigned runs, bool indexed, uint32_t first, uint32_t count,
             uint32_t instances, int32_t base_vertex, uint32_t first_instance, uint32_t draw_id)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const uint64_t total = (uint64_t)count * instances;
   struct wvk_bound_views images;
   bound_views(st, WVK_BIND_GRAPHICS, st->bound_vs, &images);
   struct wvk_shader *cs =
      wvk_shader_draw_variant(dev, st->bound_vs, st->fs, dyn, 1, false, NULL, &images, WVK_DRAW_PRIMITIVES, runs, 0);
   if (!cs || cs->vk.stage != MESA_SHADER_COMPUTE)
      return false;
   if (!bind_index_storage(cmd, indexed))
      return true;
   const uint32_t run[4] = { count, (uint32_t)total, first, indexed && dyn->ia.primitive_restart_enable };
   if (memcmp(st->push + WVK_DRIVER_VERTEX_RUN, run, sizeof(run))) {
      memcpy(st->push + WVK_DRIVER_VERTEX_RUN, run, sizeof(run));
      st->push_offset = UINT32_MAX;
   }
   /* The draw's parameters as the compute shader reads them. */
   struct wvk_shader *draw_vs = st->vs;
   st->vs = cs;
   set_draw_parameters(st, NULL, 0, indexed, indexed ? base_vertex : (int32_t)first, first_instance, draw_id);
   st->vs = draw_vs;
   set_pulled_buffers(dev, cmd, replay, cs);
   dispatch_graphics(cmd, cs, total);
   return true;
}

/* A compute shader of the draw's -- a vertex or geometry shader run as one
 * -- of `invocations`, rows of 64 by as many workgroups as fit, with the
 * draw's sets and push data, whose records are its own: the render pass
 * ended, and begun again after. */
static void
dispatch_graphics(struct wvk_cmd_buffer *cmd, struct wvk_shader *cs, uint64_t invocations)
{
   dispatch_graphics_groups(cmd, cs, DIV_ROUND_UP(invocations, 64), NULL, 0);
}

/* The same of `groups` workgroups -- or as many as `indirect` says at
 * `offset`, which a pass before wrote. */
static void
dispatch_graphics_groups(struct wvk_cmd_buffer *cmd, struct wvk_shader *cs, uint64_t groups, WGPUBuffer indirect,
                         uint64_t offset)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   struct wvk_shader *vs = st->vs;
   st->vs = cs;
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   struct wvk_compute_pipeline *p = get_compute_pipeline(dev, st, cs, WVK_BIND_GRAPHICS);
   if (p) {
      sync_image_textures(cmd, p->layout, WVK_BIND_GRAPHICS);
      sync_atomic_buffers(cmd, p->layout, WVK_BIND_GRAPHICS);
      struct buffer_use uses[64];
      const uint32_t n = gather_buffer_uses(dev, replay, p->layout, WVK_BIND_GRAPHICS, uses, ARRAY_SIZE(uses));
      swap_conflicting_reads(dev, replay, uses, n);
      swap_sampled_storage(cmd, p->layout, WVK_BIND_GRAPHICS);
      if (!replay->compute) {
         WGPUComputePassDescriptor desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
         replay->compute = wgpuCommandEncoderBeginComputePass(replay->encoder, &desc);
      }
      wgpuComputePassEncoderSetPipeline(replay->compute, p->pipeline);
      if (bind_groups(dev, replay, p->layout, WVK_BIND_GRAPHICS, NULL, replay->compute, false)) {
         if (indirect) {
            wgpuComputePassEncoderDispatchWorkgroupsIndirect(replay->compute, indirect, offset);
         } else if (groups) {
            const uint32_t x = (uint32_t)MIN2(groups, 65535);
            wgpuComputePassEncoderDispatchWorkgroups(replay->compute, x, (uint32_t)DIV_ROUND_UP(groups, x), 1);
         }
      }
      util_dynarray_clear(&replay->buffer_swaps);
      util_dynarray_clear(&replay->texture_swaps);
   }
   wvk_replay_end_passes(replay);
   st->vs = vs;
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

/* A draw of points as quads (draws_points): a triangle list of six
 * vertices a point, whose vertex shader reads an indexed draw's indices
 * from the index buffer, bound as storage (expand_points). */
static void
draw_points(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, uint32_t instances,
            int32_t base_vertex, uint32_t first_instance, uint32_t draw_id)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   if (!count || !instances)
      return;
   if ((uint64_t)first * 6 + (uint64_t)count * 6 > UINT32_MAX) {
      mesa_logw_once("wvk: a draw of more points than six vertices each can number");
      return;
   }
   if (!bind_index_storage(cmd, indexed))
      return;
   replay->list_topology = WGPUPrimitiveTopology_TriangleList;
   WGPUBuffer predicated =
      predicated_draw(cmd, (const uint32_t[4]){ count * 6, instances, first * 6, first_instance }, 4);
   foreach_viewport(cmd) {
      set_draw_parameters(st, NULL, 0, indexed, indexed ? base_vertex : (int32_t)first, first_instance, draw_id);
      if (!flush_draw(cmd, false))
         continue;
      if (predicated)
         wgpuRenderPassEncoderDrawIndirect(replay->pass, predicated, 0);
      else
         wgpuRenderPassEncoderDraw(replay->pass, count * 6, instances, first * 6, first_instance);
   }
   replay->list_topology = WGPUPrimitiveTopology_Undefined;
}

/* A draw of the points of a list the driver made, as quads (draws_points):
 * six vertices an index, which the vertex shader reads from the list
 * bound as storage (expand_points); of `count` indices, or as many as the
 * indirect draw at `at` in `indirect` says. */
static void
draw_point_list(struct wvk_cmd_buffer *cmd, const struct wvk_driver_buffer *list, uint32_t start, uint32_t count,
                WGPUBuffer indirect, uint64_t at, uint32_t draw_id)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   if ((uint64_t)count * 6 > UINT32_MAX) {
      mesa_logw_once("wvk: a draw of more points than six vertices each can number");
      return;
   }
   st->index = *list;
   const uint32_t words[2] = { start, 4 };
   if (memcmp(st->push + WVK_DRIVER_POINT_INDEX, words, sizeof(words))) {
      memcpy(st->push + WVK_DRIVER_POINT_INDEX, words, sizeof(words));
      st->push_offset = UINT32_MAX;
   }
   replay->list_topology = WGPUPrimitiveTopology_TriangleList;
   /* Of a predicate the GPU has: the command's, zero where it fails. */
   if (replay->cond_gpu) {
      struct wvk_bound_buffer b;
      const struct wvk_bound_buffer src = { NULL, indirect, at, 16, ++wvk_cmd_buffer_device(cmd)->temp_serial };
      if (indirect ? conditional_args(cmd, NULL, &src, 1, 4, 16, &b)
                   : conditional_args(cmd, (const uint32_t[4]){ count * 6, 1, 0, 0 }, NULL, 1, 4, 16, &b)) {
         indirect = b.buffer;
         at = 0;
      }
   }
   foreach_viewport(cmd) {
      set_draw_parameters(st, NULL, 0, true, 0, 0, draw_id);
      if (!flush_draw(cmd, false))
         continue;
      if (indirect)
         wgpuRenderPassEncoderDrawIndirect(replay->pass, indirect, at);
      else
         wgpuRenderPassEncoderDraw(replay->pass, count * 6, 1, 0, 0);
   }
   replay->list_topology = WGPUPrimitiveTopology_Undefined;
}

/* A geometry shader's lists drawn a vertex each, which pulls its index
 * from them itself (geometry_vertex_shader), for the barycentric
 * coordinates of its corner and its primitive's other vertices: `count`
 * from `start`, or as the indexed draw at `at` of `indirect` has them --
 * its first four words, whose first index and base vertex are 0, a draw's. */
static void
draw_pulled_list(struct wvk_cmd_buffer *cmd, const struct wvk_driver_buffer *list, uint32_t start, uint32_t count,
                 WGPUPrimitiveTopology topology, WGPUBuffer indirect, uint64_t at, uint32_t draw_id)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   st->index = *list;
   const uint32_t words[2] = { start, 4 };
   if (memcmp(st->push + WVK_DRIVER_POINT_INDEX, words, sizeof(words))) {
      memcpy(st->push + WVK_DRIVER_POINT_INDEX, words, sizeof(words));
      st->push_offset = UINT32_MAX;
   }
   replay->list_topology = topology;
   if (replay->cond_gpu) {
      struct wvk_bound_buffer b;
      const struct wvk_bound_buffer src = { NULL, indirect, at, 16, ++wvk_cmd_buffer_device(cmd)->temp_serial };
      if (indirect ? conditional_args(cmd, NULL, &src, 1, 4, 16, &b)
                   : conditional_args(cmd, (const uint32_t[4]){ count, 1, 0, 0 }, NULL, 1, 4, 16, &b)) {
         indirect = b.buffer;
         at = 0;
      }
   }
   foreach_viewport(cmd) {
      set_draw_parameters(st, NULL, 0, true, 0, 0, draw_id);
      if (!flush_draw(cmd, false)) {
         continue;
      }
      if (indirect)
         wgpuRenderPassEncoderDrawIndirect(replay->pass, indirect, at);
      else
         wgpuRenderPassEncoderDraw(replay->pass, count, 1, 0, 0);
   }
   replay->list_topology = WGPUPrimitiveTopology_Undefined;
}

/* The primitives of `count` vertices of a topology, an instance, as a
 * geometry shader takes them; 0 for a topology it does not take them of. */
static uint32_t
geometry_primitives(VkPrimitiveTopology topology, enum mesa_prim input, uint32_t count)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
      return input == MESA_PRIM_POINTS ? count : 0;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
      return input == MESA_PRIM_LINES ? count / 2 : 0;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
      return input == MESA_PRIM_LINES && count > 1 ? count - 1 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
      return input == MESA_PRIM_TRIANGLES ? count / 3 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
      return input == MESA_PRIM_TRIANGLES && count > 2 ? count - 2 : 0;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
      return input == MESA_PRIM_LINES_ADJACENCY ? count / 4 : 0;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY:
      return input == MESA_PRIM_LINES_ADJACENCY && count > 3 ? count - 3 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      return input == MESA_PRIM_TRIANGLES_ADJACENCY ? count / 6 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY:
      return input == MESA_PRIM_TRIANGLES_ADJACENCY && count >= 6 ? (count - 4) / 2 : 0;
   default:
      return 0;
   }
}

/* The primitives of a topology, as a geometry shader takes them. */
static enum mesa_prim
topology_primitive(VkPrimitiveTopology topology)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return MESA_PRIM_POINTS;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return MESA_PRIM_LINES;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return MESA_PRIM_LINES_ADJACENCY;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: return MESA_PRIM_TRIANGLES_ADJACENCY;
   default: return MESA_PRIM_TRIANGLES;
   }
}

/* Whether a draw without a geometry shader, whose primitives are all of
 * stream 0, has another rasterized: then nothing is. */
static bool
rasterizes_other_stream(const struct wvk_cmd_buffer *cmd)
{
   return cmd->vk.dynamic_graphics_state.rs.rasterization_stream != 0;
}

/* The geometry shader a draw has: the pipeline's, or where the stage
 * before it writes cull distances, one that passes its primitives through
 * to cull them (wvk_shader_cull_geometry). */
static struct wvk_shader *
draw_gs(struct wvk_cmd_buffer *cmd)
{
   const struct wvk_bound_state *st = cmd->replay->state;
   if (st->bound_gs)
      return st->bound_gs;
   struct wvk_shader *last = st->bound_tes ? st->bound_tes : st->bound_vs;
   /* Or that transform feedback captures, where the geometry shader's lists
    * are what it writes (capture_xfb); or whose fragment shader reads
    * barycentric coordinates or inputs of each vertex, which the vertex
    * shader after one has of its lists (geometry_vertex_shader). */
   const bool bary = st->bound_fs && (st->bound_fs->bary_uses || st->bound_fs->per_vertex_mask);
   if (!last || (!last->cull_distances && !(cmd->replay->xfb_active && last->xfb_buffers) && !bary))
      return NULL;
   enum mesa_prim input = topology_primitive(cmd->vk.dynamic_graphics_state.ia.primitive_topology);
   if (st->bound_tes) {
      const struct wvk_shader *tcs = st->bound_tcs, *tes = st->bound_tes;
      const unsigned domain = tes->tess_domain ? tes->tess_domain : tcs ? tcs->tess_domain : 0;
      input = tes->tess_point_mode || (tcs && tcs->tess_point_mode) ? MESA_PRIM_POINTS
              : domain == TESS_PRIMITIVE_ISOLINES                     ? MESA_PRIM_LINES
                                                                      : MESA_PRIM_TRIANGLES;
   }
   return wvk_shader_cull_geometry(wvk_cmd_buffer_device(cmd), last, input);
}

/* Transform feedback's capture, of a vertex stream: the geometry shader's
 * lists of it walked in order -- invocation after invocation, each's
 * primitives as it made them -- and each primitive's vertices' outputs
 * written to the stream's buffers, at the offsets they are at, as long as
 * the whole primitive fits in every one; after the first that does not,
 * none of the stream's. One invocation: the order is the point. The
 * primitives written and those that would have been are the counters of
 * the stream's transform feedback queries. */
/* A macro's value, in WGSL. */
#define WVK_STR2(x) #x
#define WVK_STR(x) WVK_STR2(x)

static const char xfb_wgsl[] =
   "@group(0) @binding(0) var<storage, read> p: array<u32>;\n"
   "@group(0) @binding(1) var<storage, read> records: array<u32>;\n"
   "@group(0) @binding(2) var<storage, read> indices: array<u32>;\n"
   "@group(0) @binding(3) var<storage, read_write> state: array<u32>;\n"
   "@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;\n"
   "@group(0) @binding(5) var<storage, read> src: array<u32>;\n"
   "@group(0) @binding(6) var<storage, read_write> t0: array<u32>;\n"
   "@group(0) @binding(7) var<storage, read_write> t1: array<u32>;\n"
   "@group(0) @binding(8) var<storage, read_write> t2: array<u32>;\n"
   "@group(0) @binding(9) var<storage, read_write> t3: array<u32>;\n"
   "fn put(t: u32, i: u32, v: u32) {\n"
   "  switch t { case 0u: { t0[i] = v; } case 1u: { t1[i] = v; } case 2u: { t2[i] = v; } default: { t3[i] = v; } }\n"
   "}\n"
   "@compute @workgroup_size(1)\n"
   "fn main() {\n"
   "  if (p[25] != 0u && ((p[26] != 0u) == (p[27] != 0u))) { return; }\n"
   "  var invocations = p[0];\n"
   "  if (p[22] != 0xffffffffu) { invocations = src[p[22]]; }\n"
   "  let per = p[1]; let pv = p[2]; let stride = p[3]; let n = p[4]; let mask = p[5];\n"
   "  let base = p[23]; let stream = p[24];\n"
   "  var off = array<u32, 4>(state[0], state[1], state[2], state[3]);\n"
   "  var stopped = state[4u + stream] != 0u;\n"
   "  var written = 0u; var needed = 0u;\n"
   "  for (var g = 0u; g < invocations; g++) {\n"
   "    for (var k = 0u; k < per; k++) {\n"
   "      let at = base + (g * per + k) * pv;\n"
   "      if (indices[at] == 0u) { break; }\n"
   "      needed += 1u;\n"
   "      if (stopped) { continue; }\n"
   "      var fits = true;\n"
   "      for (var b = 0u; b < 4u; b++) {\n"
   "        if ((mask & (1u << b)) != 0u && off[b] + pv * p[6u + b] > p[10u + b]) { fits = false; }\n"
   "      }\n"
   "      if (!fits) { stopped = true; continue; }\n"
   "      for (var v = 0u; v < pv; v++) {\n"
   "        let rec = indices[at + v] * stride;\n"
   "        for (var o = 0u; o < n; o++) {\n"
   "          let w = p[28u + 2u * o];\n"
   "          let b = w >> 24u;\n"
   "          if ((mask & (1u << b)) == 0u) { continue; }\n"
   "          let first = (off[b] + v * p[6u + b] + p[29u + 2u * o]) / 4u + p[14u + b];\n"
   "          for (var c = 0u; c < ((w >> 16u) & 0xffu); c++) {\n"
   "            put(p[18u + b], first + c, records[rec + (w & 0xffu) * 4u + ((w >> 8u) & 0xffu) + c]);\n"
   "          }\n"
   "        }\n"
   "      }\n"
   "      for (var b = 0u; b < 4u; b++) {\n"
   "        if ((mask & (1u << b)) != 0u) { off[b] += pv * p[6u + b]; }\n"
   "      }\n"
   "      written += 1u;\n"
   "    }\n"
   "  }\n"
   "  for (var b = 0u; b < 4u; b++) { state[b] = off[b]; }\n"
   "  state[4u + stream] = select(0u, 1u, stopped);\n"
   "  atomicAdd(&counters[" WVK_STR(WVK_COUNTER_XFB_WRITTEN(0)) " + 2u * stream], written);\n"
   "  atomicAdd(&counters[" WVK_STR(WVK_COUNTER_XFB_NEEDED(0)) " + 2u * stream], needed);\n"
   "}\n";

static WGPUComputePipeline
xfb_pipeline(struct wvk_device *dev, WGPUComputePipeline *pipeline, const char *wgsl, const char *label)
{
   simple_mtx_lock(&dev->cache_lock);
   if (!*pipeline) {
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
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

/* The capture of a draw's primitives, a geometry shader's `gs` (its own or
 * the driver's): of its records and lists, as many invocations as
 * `invocations`, or as word `count_word` of `count` says, each stream's
 * lists after `capacity` invocations of the stream's before. Outside any
 * pass, the pass ended for it and begun again. */
static void
capture_xfb(struct wvk_cmd_buffer *cmd, const struct wvk_shader *gs, const struct wvk_driver_buffer *records,
            const struct wvk_driver_buffer *indices, uint32_t invocations, uint64_t capacity, WGPUBuffer count,
            uint64_t count_size, uint32_t count_word)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   if (!replay->xfb_active || !gs->xfb_buffers || !replay->xfb_state || replay->pass_idle ||
       (replay->pass_layered && replay->pass_layer != replay->pass_first_layer))
      return;
   WGPUComputePipeline pipeline = xfb_pipeline(dev, &dev->xfb_pipeline, xfb_wgsl, "wvk transform feedback");
   if (!pipeline)
      return;
   const uint32_t pv = gs->gs_output == MESA_PRIM_TRIANGLE_STRIP ? 3 : gs->gs_output == MESA_PRIM_LINE_STRIP ? 2 : 1;
   const uint32_t per = wvk_geometry_indices(gs->gs_output, gs->gs_max_vertices);
   bool in_pass = false;
   for (uint32_t stream = 0; stream < MAX2(gs->gs_streams, 1); ++stream) {
      uint32_t params[28 + 2 * WVK_MAX_XFB_OUTPUTS] = { 0 };
      params[0] = invocations;
      params[1] = per / pv;
      params[2] = pv;
      params[3] = wvk_record_stride(gs->record_vars) / 4;
      params[4] = gs->xfb_output_count;
      params[22] = count ? count_word : UINT32_MAX;
      params[23] = (uint32_t)(stream * capacity * per);
      params[24] = stream;
      /* Of a command under a predicate the GPU has: none where it fails,
       * the predicate copied to word 26. */
      if (replay->cond_gpu) {
         params[25] = 1;
         params[27] = replay->cond.inverted;
      }
      /* Each of the stream's buffers bound: one binding a WebGPU buffer,
       * over all the ranges of it, as writable bindings must not overlap. */
      WGPUBuffer slots[4] = { NULL };
      uint64_t slot_start[4] = { 0 }, slot_end[4] = { 0 };
      uint32_t slot_count = 0;
      for (uint32_t b = 0; b < 4; ++b) {
         if (!(gs->xfb_buffers & BITFIELD_BIT(b)) || gs->xfb_streams[b] != stream || !replay->xfb_bindings[b].buffer)
            continue;
         const uint64_t start = replay->xfb_bindings[b].offset, end = start + replay->xfb_bindings[b].size;
         uint32_t t = 0;
         while (t < slot_count && slots[t] != replay->xfb_bindings[b].buffer)
            ++t;
         if (t == slot_count) {
            slots[slot_count] = replay->xfb_bindings[b].buffer;
            slot_start[slot_count] = start & ~255ull;
            slot_end[slot_count++] = end;
         } else {
            slot_start[t] = MIN2(slot_start[t], start & ~255ull);
            slot_end[t] = MAX2(slot_end[t], end);
         }
         params[5] |= BITFIELD_BIT(b);
         params[6 + b] = gs->xfb_strides[b];
         params[10 + b] = (uint32_t)replay->xfb_bindings[b].size;
         params[18 + b] = t;
      }
      if (!params[5])
         continue;
      for (uint32_t b = 0; b < 4; ++b) {
         if (params[5] & BITFIELD_BIT(b))
            params[14 + b] = (uint32_t)((replay->xfb_bindings[b].offset - slot_start[params[18 + b]]) / 4);
      }
      for (uint32_t o = 0; o < gs->xfb_output_count; ++o) {
         const struct wvk_xfb_output *x = &gs->xfb_outputs[o];
         params[28 + 2 * o] = x->slot | x->component << 8 | x->count << 16 | (uint32_t)x->buffer << 24;
         params[29 + 2 * o] = x->offset;
      }
      WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params),
                                                        WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst, params);
      WGPUBuffer counters = wvk_replay_counters(replay);
      static const uint32_t one[4] = { 0 };
      WGPUBuffer none =
         count ? NULL : wvk_replay_temp_buffer(dev, replay, sizeof(one), WGPUBufferUsage_Storage, one);
      if (!params_buffer || !counters || (!count && !none))
         break;
      WGPUBindGroupEntry entries[10] = {
         { .binding = 0, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
         { .binding = 1, .buffer = records->buffer, .offset = 0, .size = records->size },
         { .binding = 2, .buffer = indices->buffer, .offset = 0, .size = indices->size },
         { .binding = 3, .buffer = replay->xfb_state, .offset = 0, .size = 32 },
         { .binding = 4, .buffer = counters, .offset = 0, .size = WVK_COUNTERS * 4 },
         { .binding = 5, .buffer = count ? count : none, .offset = 0, .size = count ? align64(count_size, 4) : 16 },
      };
      bool bound = true;
      for (uint32_t t = 0; t < 4; ++t) {
         WGPUBuffer buffer =
            t < slot_count ? slots[t] : wvk_replay_temp_buffer(dev, replay, 16, WGPUBufferUsage_Storage, NULL);
         if (!buffer) {
            bound = false;
            break;
         }
         entries[6 + t] = (WGPUBindGroupEntry){
            .binding = 6 + t, .buffer = buffer, .offset = t < slot_count ? slot_start[t] : 0,
            .size = t < slot_count ? align64(slot_end[t] - slot_start[t], 4) : 16,
         };
      }
      if (!bound)
         break;
      if (replay->cond_gpu)
         wvk_replay_read(replay, replay->cond.mem, replay->cond.offset, 4);
      /* What of the buffers is not written is as it was. */
      for (uint32_t b = 0; b < 4; ++b) {
         if (params[5] & BITFIELD_BIT(b))
            wvk_replay_read(replay, replay->xfb_bindings[b].mem, replay->xfb_bindings[b].offset,
                            replay->xfb_bindings[b].size);
      }
      WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
      bg.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
      bg.entryCount = ARRAY_SIZE(entries);
      bg.entries = entries;
      WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
      wgpuBindGroupLayoutRelease(bg.layout);
      in_pass |= replay->pass != NULL;
      wvk_replay_end_passes(replay);
      if (replay->cond_gpu)
         wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, replay->cond.buffer, replay->cond.offset, params_buffer,
                                              26 * 4, 4);
      WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
      WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
      wgpuComputePassEncoderSetPipeline(pass, pipeline);
      wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
      wgpuBindGroupRelease(group);
      for (uint32_t b = 0; b < 4; ++b) {
         if (params[5] & BITFIELD_BIT(b))
            wvk_replay_write(replay, replay->xfb_bindings[b].mem, replay->xfb_bindings[b].offset,
                             replay->xfb_bindings[b].size);
      }
   }
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

/* A draw with a geometry shader, which WebGPU has no stage of: the vertex
 * shader run as compute, a record a vertex of each instance (key.runs 3);
 * the geometry shader run as compute, an invocation a primitive of each
 * instance and invocation of its, reading those, and writing the vertices
 * it emits and the indices of the primitives they make, as a list
 * (key.geometry 1); then a draw of that list, whose vertex shader reads
 * the vertices (key.geometry 2). Primitives restarting in an indexed
 * strip are not: the strip goes on. */
static WGPUBuffer scratch(struct wvk_device *dev, struct wvk_replay *replay, unsigned i, uint64_t size,
                          WGPUBufferUsage usage, bool clear, struct wvk_driver_buffer *bound);

static void
draw_geometry(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, uint32_t instances,
              int32_t base_vertex, uint32_t first_instance, uint32_t draw_id)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct wvk_shader *gs = draw_gs(cmd);
   if (!gs || !st->bound_vs || !count || !instances || replay->pass_idle)
      return;
   const VkPrimitiveTopology topology = dyn->ia.primitive_topology;
   const uint32_t prims = geometry_primitives(topology, gs->gs_input, count);
   if (!prims) {
      if (!geometry_primitives(topology, gs->gs_input, 1024))
         mesa_logw_once("wvk: a draw of a topology its geometry shader does not take");
      return;
   }
   if (indexed && dyn->ia.primitive_restart_enable)
      mesa_logw_once("wvk: an indexed draw with a geometry shader, whose primitives do not restart");
   const uint64_t total = (uint64_t)prims * instances * gs->gs_invocations;
   const uint32_t per = wvk_geometry_indices(gs->gs_output, gs->gs_max_vertices);
   const uint64_t limit = pdev->limits.maxStorageBufferBindingSize;

   /* The vertex shader's records. */
   struct wvk_bound_views images;
   bound_views(st, WVK_BIND_GRAPHICS, st->bound_vs, &images);
   struct wvk_shader *vs =
      wvk_shader_draw_variant(dev, st->bound_vs, NULL, dyn, 1, false, NULL, &images, WVK_DRAW_PRIMITIVES, 3, 0);
   if (!vs || vs->vk.stage != MESA_SHADER_COMPUTE) {
      mesa_logw_once("wvk: a vertex shader before a geometry shader, which could not be made a compute shader");
      return;
   }
   const uint64_t records = (uint64_t)count * instances * wvk_record_stride(vs->record_vars);
   /* The geometry shader's, and the draw's. */
   const bool quads = replay->draw_points;
   struct wvk_shader *cs = wvk_shader_geometry_variant(dev, gs, 1, vs->record_vars, NULL, dyn, false, false,
                                                       gs == st->bound_gs ? wvk_replay_stats(replay) : 0);
   struct wvk_shader *passthrough = wvk_shader_geometry_variant(dev, gs, 2, 0, st->fs, dyn, false, quads, 0);
   if (!cs || !passthrough) {
      mesa_logw_once("wvk: a geometry shader that could not be made compute");
      return;
   }
   /* A pass drawn a layer at a time draws in its others only what chooses
    * its layer: again, each culling what is not its. */
   if (replay->pass_layered && !replay->pass_multiview && replay->pass_layer && !passthrough->writes_layer)
      return;
   /* Each vertex stream's vertices and lists, after the stream's before. */
   const uint32_t streams = MAX2(gs->gs_streams, 1);
   const uint64_t vertices = (1 + total * streams * gs->gs_max_vertices) * wvk_record_stride(gs->record_vars);
   const uint64_t indices = total * per;
   if (records > limit || vertices > limit || indices * streams * 4 > limit || total > UINT32_MAX) {
      mesa_logw_once("wvk: a draw with a geometry shader of more than its buffers can bind");
      return;
   }
   /* The submit's, draw after draw (scratch). */
   const uint64_t list_bytes = MAX2(indices * streams * 4, 16);
   WGPUBuffer capture = scratch(dev, replay, WVK_SCRATCH_GEOMETRY + 0, records, WGPUBufferUsage_Storage, true,
                                &st->capture);
   WGPUBuffer out = scratch(dev, replay, WVK_SCRATCH_GEOMETRY + 1, vertices, WGPUBufferUsage_Storage, true,
                            &st->gs_vertices);
   WGPUBuffer list = scratch(dev, replay, WVK_SCRATCH_GEOMETRY + 2, list_bytes,
                             WGPUBufferUsage_Storage | WGPUBufferUsage_Index, true, &st->gs_indices);
   if (!capture || !out || !list)
      return;

   if (!run_vertices(cmd, 3, indexed, first, count, instances, base_vertex, first_instance, draw_id))
      return;
   const uint32_t geometry[4] = { count, prims, (uint32_t)total, topology };
   if (memcmp(st->push + WVK_DRIVER_GEOMETRY, geometry, sizeof(geometry))) {
      memcpy(st->push + WVK_DRIVER_GEOMETRY, geometry, sizeof(geometry));
      st->push_offset = UINT32_MAX;
   }
   dispatch_graphics(cmd, cs, total);
   capture_xfb(cmd, gs, &st->gs_vertices, &st->gs_indices, (uint32_t)total, total, NULL, 0, 0);

   /* The rasterized stream's lists, of a stream it emits to. */
   const uint32_t raster = dyn->rs.rasterization_stream;
   if (raster >= streams)
      return;
   struct wvk_shader *draw_vs = st->vs;
   st->vs = passthrough;
   const WGPUPrimitiveTopology primitives = gs->gs_output == MESA_PRIM_TRIANGLE_STRIP ? WGPUPrimitiveTopology_TriangleList
                                            : gs->gs_output == MESA_PRIM_LINE_STRIP  ? WGPUPrimitiveTopology_LineList
                                                                                     : WGPUPrimitiveTopology_PointList;
   if (quads)
      draw_point_list(cmd, &st->gs_indices, (uint32_t)(raster * indices * 4), (uint32_t)indices, NULL, 0, draw_id);
   else if (passthrough->key.bary || passthrough->key.per_vertex)
      draw_pulled_list(cmd, &st->gs_indices, (uint32_t)(raster * indices * 4), (uint32_t)indices, primitives, NULL, 0,
                       draw_id);
   else
      draw_list_at(cmd, list, raster * indices * 4, (uint32_t)indices, primitives, 1, 0, 0, true, draw_id,
                   UINT64_MAX);
   st->vs = draw_vs;
}

/* A mesh draw's passes (draw_mesh). The setup: the draw's workgroups, of
 * its command or its indirect arguments -- none where a count or a
 * predicate says so -- as many as the buffers hold, and the dispatch of
 * them; and without a task shader, the mesh workgroups. The scan: after a
 * task shader, where each task workgroup's launch starts in the mesh
 * workgroups, and theirs. The round: the next of the mesh workgroups, as
 * many as the buffers hold, their dispatch and the draw of them. */
static const char mesh_wgsl[] =
   "struct P { g: vec3u, indirect: u32,\n"
   "           first_capacity: u32, mesh_capacity: u32, corners: u32, task: u32,\n"
   "           counted: u32, count_limit: u32, draw_index: u32, cond: u32,\n"
   "           args_word: u32, count_word: u32, cond_word: u32, all_capacity: u32 }\n"
   "@group(0) @binding(0) var<uniform> p: P;\n"
   "@group(0) @binding(1) var<storage, read> args: array<u32>;\n"
   "@group(0) @binding(2) var<storage, read_write> header: array<u32>;\n"
   "@group(0) @binding(3) var<storage, read_write> launches: array<u32>;\n"
   "@group(0) @binding(4) var<storage, read> count: array<u32>;\n"
   "@group(0) @binding(5) var<storage, read> predicate: array<u32>;\n"
   "\n"
   "fn split(total: u32, at: u32) {\n"
   "  let x = min(total, 65535u);\n"
   "  header[at] = x;\n"
   "  header[at + 1u] = select(0u, (total + x - 1u) / max(x, 1u), x != 0u);\n"
   "  header[at + 2u] = 1u;\n"
   "}\n"
   "fn mul_capped(a: u32, b: u32, cap: u32) -> u32 {\n"
   "  if (a == 0u || b == 0u) { return 0u; }\n"
   "  if (a > cap / b) { return cap; }\n"
   "  return min(a * b, cap);\n"
   "}\n"
   "fn mesh_totals(total: u32) {\n"
   "  header[" WVK_STR(WVK_MESH_ALL) "u] = min(total, p.all_capacity);\n"
   "  header[" WVK_STR(WVK_MESH_NEXT) "u] = 0u;\n"
   "}\n"
   "@compute @workgroup_size(1) fn setup() {\n"
   "  var g = p.g;\n"
   "  if (p.indirect != 0u) { g = vec3u(args[p.args_word], args[p.args_word + 1u], args[p.args_word + 2u]); }\n"
   "  if (p.counted != 0u && p.draw_index >= min(count[p.count_word], p.count_limit)) { g = vec3u(0u); }\n"
   "  if (p.cond != 0u && (predicate[p.cond_word] != 0u) == (p.cond == 2u)) { g = vec3u(0u); }\n"
   "  let total = mul_capped(mul_capped(g.x, g.y, p.first_capacity), g.z, p.first_capacity);\n"
   "  header[0] = g.x; header[1] = g.y; header[2] = g.z;\n"
   "  header[" WVK_STR(WVK_MESH_FIRST_TOTAL) "u] = total;\n"
   "  split(total, " WVK_STR(WVK_MESH_FIRST_DISPATCH) "u);\n"
   "  if (p.task == 0u) { mesh_totals(total); }\n"
   "}\n"
   "@compute @workgroup_size(1) fn scan() {\n"
   "  let tasks = header[" WVK_STR(WVK_MESH_FIRST_TOTAL) "u];\n"
   "  var sum = 0u;\n"
   "  for (var t = 0u; t < tasks; t++) {\n"
   "    let n = mul_capped(mul_capped(launches[4u * t], launches[4u * t + 1u], p.all_capacity),\n"
   "                       launches[4u * t + 2u], p.all_capacity);\n"
   "    launches[4u * t + 3u] = sum;\n"
   "    sum = min(sum + n, p.all_capacity);\n"
   "  }\n"
   "  mesh_totals(sum);\n"
   "}\n"
   "@compute @workgroup_size(1) fn round() {\n"
   "  let base = header[" WVK_STR(WVK_MESH_NEXT) "u];\n"
   "  let m = min(header[" WVK_STR(WVK_MESH_ALL) "u] - base, p.mesh_capacity);\n"
   "  header[" WVK_STR(WVK_MESH_BASE) "u] = base;\n"
   "  header[" WVK_STR(WVK_MESH_NEXT) "u] = base + m;\n"
   "  split(m, " WVK_STR(WVK_MESH_DISPATCH) "u);\n"
   "  header[" WVK_STR(WVK_MESH_TOTAL) "u] = m;\n"
   "  header[" WVK_STR(WVK_MESH_DRAW) "u] = m * p.corners;\n"
   "  header[" WVK_STR(WVK_MESH_DRAW) "u + 1u] = 1u;\n"
   "  header[" WVK_STR(WVK_MESH_DRAW) "u + 2u] = 0u;\n"
   "  header[" WVK_STR(WVK_MESH_DRAW) "u + 3u] = 0u;\n"
   "}\n";

/* The mesh passes, made the first time. */
static bool
mesh_pipelines(struct wvk_device *dev)
{
   simple_mtx_lock(&dev->cache_lock);
   if (!dev->mesh_bgl) {
      WGPUBindGroupLayoutEntry e[6];
      for (uint32_t i = 0; i < 6; ++i) {
         e[i] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
         e[i].binding = i;
         e[i].visibility = WGPUShaderStage_Compute;
         e[i].buffer.type = i == 0                  ? WGPUBufferBindingType_Uniform
                            : i == 2 || i == 3      ? WGPUBufferBindingType_Storage
                                                    : WGPUBufferBindingType_ReadOnlyStorage;
      }
      WGPUBindGroupLayoutDescriptor bd = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
      bd.label = wvk_sv("wvk mesh");
      bd.entryCount = 6;
      bd.entries = e;
      dev->mesh_bgl = wgpuDeviceCreateBindGroupLayout(dev->wgpu, &bd);
      WGPUPipelineLayoutDescriptor pd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
      pd.bindGroupLayoutCount = 1;
      pd.bindGroupLayouts = &dev->mesh_bgl;
      WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(dev->wgpu, &pd);
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(mesh_wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk mesh");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      static const char *const entries[3] = { "setup", "scan", "round" };
      for (uint32_t i = 0; i < 3; ++i) {
         WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
         desc.label = wvk_sv("wvk mesh");
         desc.layout = layout;
         desc.compute.module = module;
         desc.compute.entryPoint = wvk_sv(entries[i]);
         dev->mesh_pipelines[i] = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
      }
      wgpuShaderModuleRelease(module);
      wgpuPipelineLayoutRelease(layout);
   }
   simple_mtx_unlock(&dev->cache_lock);
   return dev->mesh_bgl && dev->mesh_pipelines[0] && dev->mesh_pipelines[1] && dev->mesh_pipelines[2];
}

/* How big each of a mesh draw's buffers may be: what a submit's mesh
 * draws share, which WebGPU keeps until the GPU is done with them. */
#define WVK_MESH_BUDGET (64ull << 20)

/* The rounds of a mesh draw whose mesh workgroups only the GPU knows the
 * number of -- after a task shader, or of indirect arguments -- each as
 * many as the buffers hold, at most as many as maxMeshWorkGroupTotalCount
 * takes: a round is a render pass ended and begun again, and the rounds of
 * fewer than there are, passes of nothing. And the most of any draw's. */
#define WVK_MESH_ROUNDS 8
#define WVK_MESH_MAX_ROUNDS 1024

/* A draw's buffer of slot `i` (wvk_replay::scratch), of at least `size`
 * bytes: the submit's, made bigger where it has to be -- and where it is
 * not, zeroed as a new one is (`clear`), before the next pass. Bound whole
 * where `bound` is. */
static WGPUBuffer
scratch(struct wvk_device *dev, struct wvk_replay *replay, unsigned i, uint64_t size, WGPUBufferUsage usage, bool clear,
        struct wvk_driver_buffer *bound)
{
   size = align64(MAX2(size, 16), 16);
   if (replay->scratch[i].size < size) {
      replay->scratch[i].buffer = wvk_replay_temp_buffer(dev, replay, size, usage | WGPUBufferUsage_CopyDst, NULL);
      replay->scratch[i].size = replay->scratch[i].buffer ? size : 0;
      replay->scratch[i].serial = ++dev->temp_serial | (1ull << 62);
   } else if (clear) {
      util_dynarray_append(&replay->scratch_clears, ((struct wvk_scratch_clear){ replay->scratch[i].buffer, size }));
   }
   if (bound)
      *bound = (struct wvk_driver_buffer){ replay->scratch[i].buffer, replay->scratch[i].serial, 0,
                                           replay->scratch[i].size };
   return replay->scratch[i].buffer;
}

/* A mesh draw (VK_EXT_mesh_shader), which WebGPU has no stages of: the
 * setup; a task shader, if there is one, as compute (key.task 1), a
 * workgroup a task workgroup, writing its payload and launch, and the scan
 * of the launches; the mesh shader as compute (key.mesh 1), a workgroup a
 * mesh workgroup, writing its vertices' and primitives' records; and a
 * draw of every primitive it may have made, a vertex a corner, whose
 * vertex shader reads those (key.mesh 2) -- every pass's dispatch the one
 * before's. Of `gx` * `gy` * `gz` workgroups, or of the arguments at `at`
 * of `indirect`, and for a draw of a count, of draw `draw_id` of those the
 * count at `count_at` of `count` says, at most `count_limit`. */
static void
draw_mesh(struct wvk_cmd_buffer *cmd, uint32_t gx, uint32_t gy, uint32_t gz, WGPUBuffer indirect, uint64_t at,
          WGPUBuffer count, uint64_t count_at, uint32_t count_limit, uint32_t draw_id)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct wvk_shader *mesh = st->bound_mesh, *task = st->bound_task;
   /* The shaders run without rasterization too, for what they store. */
   const bool discard = dyn->rs.rasterizer_discard_enable;
   if (!mesh || replay->pass_idle)
      return;
   if (!indirect && (!gx || !gy || !gz))
      return;
   /* Points of a size, as quads (expand_points): six vertices a point. */
   const bool quads = wvk_shader_mesh_quads(mesh, discard);
   replay->draw_mesh = true;
   replay->draw_points = quads;
   select_draw_shaders(cmd);
   replay->draw_mesh = replay->draw_points = false;
   const uint32_t stats = wvk_replay_stats(replay);
   struct wvk_shader *mesh_cs = wvk_shader_mesh_variant(dev, mesh, 1, task != NULL, NULL, dyn, stats);
   struct wvk_shader *mesh_vs = wvk_shader_mesh_variant(dev, mesh, 2, false, st->fs, dyn, 0);
   struct wvk_shader *task_cs = task ? wvk_shader_mesh_variant(dev, task, 1, false, NULL, dyn, stats) : NULL;
   if (!mesh_cs || !mesh_vs || (task && !task_cs) || !mesh_pipelines(dev)) {
      mesa_logw_once("wvk: mesh or task shaders that could not be made compute");
      return;
   }

   /* The buffers, of as many workgroups as the draw has, or as fit; the
    * mesh workgroups' of a round of them, as many rounds as they take. */
   const uint64_t limit = MIN2(pdev->limits.maxStorageBufferBindingSize, WVK_MESH_BUDGET);
   const uint64_t vertex_bytes = MAX2((uint64_t)mesh->mesh_vertices * 16 * mesh->record_vars, 16);
   const uint64_t prim_bytes = wvk_mesh_block(mesh->mesh_primitives, mesh->prim_vars);
   const uint32_t corners = quads                                    ? 6
                            : mesh->mesh_primitive == MESA_PRIM_POINTS ? 1
                            : mesh->mesh_primitive == MESA_PRIM_LINES  ? 2
                                                                       : 3;
   const uint64_t direct = (uint64_t)gx * gy * gz;
   uint64_t mesh_capacity = MIN3(limit / vertex_bytes, limit / prim_bytes,
                                 UINT32_MAX / MAX2((uint64_t)mesh->mesh_primitives * corners, 1));
   const uint32_t payload = wvk_payload_stride(MAX2(task ? task->payload_size : 0, mesh->payload_size));
   uint64_t task_capacity = 0;
   uint32_t rounds = (uint32_t)MIN2(DIV_ROUND_UP(1ull << 22, mesh_capacity), WVK_MESH_ROUNDS);
   if (task) {
      task_capacity = MIN3(indirect ? 65536 : direct, limit / payload, limit / 16);
      if (!indirect && direct > task_capacity)
         mesa_logw_once("wvk: a mesh draw of more task workgroups than its buffers hold, drawn as many as they do");
   } else if (!indirect) {
      mesh_capacity = MIN2(mesh_capacity, direct);
      rounds = DIV_ROUND_UP(direct, mesh_capacity);
      if (rounds > WVK_MESH_MAX_ROUNDS) {
         mesa_logw_once("wvk: a mesh draw of more workgroups than its rounds hold, drawn as many as they do");
         rounds = WVK_MESH_MAX_ROUNDS;
      }
   }
   const uint64_t all_capacity = MIN2(mesh_capacity * rounds, UINT32_MAX);
   const uint64_t first_capacity = task ? task_capacity : all_capacity;
   const WGPUBufferUsage storage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
   WGPUBuffer header = scratch(dev, replay, WVK_SCRATCH_MESH + 0, WVK_MESH_HEADER_WORDS * 4,
                               WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect, false, &st->tess[3]);
   WGPUBuffer launches = scratch(dev, replay, WVK_SCRATCH_MESH + 1, task_capacity * 16, storage, false,
                                 &st->tess[1]);
   WGPUBuffer payloads = scratch(dev, replay, WVK_SCRATCH_MESH + 2, MAX2(task_capacity, 1) * payload, storage,
                                 false, &st->capture);
   WGPUBuffer vertices = scratch(dev, replay, WVK_SCRATCH_MESH + 3, mesh_capacity * vertex_bytes, storage,
                                 false, &st->gs_vertices);
   WGPUBuffer prims = scratch(dev, replay, WVK_SCRATCH_MESH + 4, mesh_capacity * prim_bytes, storage, false,
                              &st->gs_indices);
   WGPUBuffer dummy = scratch(dev, replay, WVK_SCRATCH_MESH + 5, 16, WGPUBufferUsage_Storage, false, NULL);
   if (!header || !launches || !payloads || !vertices || !prims || !dummy)
      return;

   /* The setup: of what the command says. The launches start at none,
    * and each mesh workgroup's block of its primitives zeroes its own
    * counts and culls (lower_mesh). */
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   if (task)
      wgpuCommandEncoderClearBuffer(replay->encoder, launches, 0, align64(MAX2(task_capacity * 16, 16), 16));
   const bool cond = replay->cond_gpu && replay->cond.buffer;
   if (cond)
      wvk_replay_read(replay, replay->cond.mem, replay->cond.offset, 4);
   /* The indirect arguments from their word, the count's and the
    * predicate's too: bindings of theirs, on the 256-byte boundary below. */
   const uint64_t args_at = indirect ? at & ~255ull : 0, count_base = count ? count_at & ~255ull : 0;
   const uint64_t cond_base = cond ? replay->cond.offset & ~255ull : 0;
   const uint32_t params[16] = { gx, gy, gz, indirect != NULL,
                                 (uint32_t)MIN2(first_capacity, UINT32_MAX), (uint32_t)MIN2(mesh_capacity, UINT32_MAX),
                                 mesh->mesh_primitives * corners, task != NULL,
                                 count != NULL, count_limit, draw_id, cond ? (replay->cond.inverted ? 2 : 1) : 0,
                                 (uint32_t)((at - args_at) / 4), (uint32_t)((count_at - count_base) / 4),
                                 cond ? (uint32_t)((replay->cond.offset - cond_base) / 4) : 0,
                                 (uint32_t)all_capacity };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   if (!params_buffer)
      return;
   const WGPUBindGroupEntry entries[6] = {
      { .binding = 0, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
      { .binding = 1, .buffer = indirect ? indirect : dummy, .offset = args_at, .size = indirect ? (at - args_at) + 12 : 16 },
      { .binding = 2, .buffer = header, .offset = 0, .size = WVK_MESH_HEADER_WORDS * 4 },
      { .binding = 3, .buffer = launches, .offset = 0, .size = MAX2(task_capacity * 16, 16) },
      { .binding = 4, .buffer = count ? count : dummy, .offset = count_base, .size = count ? (count_at - count_base) + 4 : 16 },
      { .binding = 5, .buffer = cond ? replay->cond.buffer : dummy, .offset = cond_base,
        .size = cond ? (replay->cond.offset - cond_base) + 4 : 16 },
   };
   WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bgd.layout = dev->mesh_bgl;
   bgd.entryCount = 6;
   bgd.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bgd);
   if (!group)
      return;
   WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &cpd);
   wgpuComputePassEncoderSetPipeline(pass, dev->mesh_pipelines[0]);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);

   /* The task shader, and the scan of what it launched; DrawIndex in the
    * draw's words for both shaders, whatever vertex shader is bound. */
   const uint32_t draw_words[8] = { WVK_DRIVER_DRAW_VALUES / 4, WVK_DRIVER_DRAW_VALUES / 4 + 1, draw_id };
   if (memcmp(st->push + WVK_DRIVER_DRAW_OFFSET, draw_words, sizeof(draw_words))) {
      memcpy(st->push + WVK_DRIVER_DRAW_OFFSET, draw_words, sizeof(draw_words));
      st->push_offset = UINT32_MAX;
   }
   st->draw = (struct wvk_driver_buffer){ 0 };
   if (task) {
      dispatch_graphics_groups(cmd, task_cs, 0, header, WVK_MESH_FIRST_DISPATCH * 4);
      pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &cpd);
      wgpuComputePassEncoderSetPipeline(pass, dev->mesh_pipelines[1]);
      wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
   }

   for (uint32_t r = 0; r < rounds; ++r) {
      /* The round's mesh workgroups, and the mesh shader of them. */
      wvk_replay_end_passes(replay);
      pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &cpd);
      wgpuComputePassEncoderSetPipeline(pass, dev->mesh_pipelines[2]);
      wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
      wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
      wgpuComputePassEncoderEnd(pass);
      wgpuComputePassEncoderRelease(pass);
      dispatch_graphics_groups(cmd, mesh_cs, 0, header, WVK_MESH_DISPATCH * 4);

      /* The draw: every primitive each mesh workgroup may have made. */
      if (in_pass && !replay->pass)
         wvk_replay_restart_pass(cmd, replay->pass_layer);
      if (!replay->pass || discard)
         continue;
      struct wvk_shader *draw_vs = st->vs;
      st->vs = mesh_vs;
      replay->list_topology = corners == 1   ? WGPUPrimitiveTopology_PointList
                              : corners == 2 ? WGPUPrimitiveTopology_LineList
                                             : WGPUPrimitiveTopology_TriangleList;
      replay->draw_points = quads;
      if (!quads || bind_index_storage(cmd, false)) {
         foreach_viewport(cmd) {
            set_draw_parameters(st, NULL, 0, false, 0, 0, draw_id);
            if (!flush_draw(cmd, false))
               continue;
            wgpuRenderPassEncoderDrawIndirect(replay->pass, header, WVK_MESH_DRAW * 4);
         }
      }
      replay->draw_points = false;
      replay->list_topology = WGPUPrimitiveTopology_Undefined;
      st->vs = draw_vs;
   }
   wgpuBindGroupRelease(group);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawMeshTasksEXT(VkCommandBuffer commandBuffer, uint32_t groupCountX, uint32_t groupCountY,
                        uint32_t groupCountZ)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU;
   draw_mesh(cmd, groupCountX, groupCountY, groupCountZ, NULL, 0, NULL, 0, 0, 0);
   cmd->replay->cond_gpu = false;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawMeshTasksIndirectEXT(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset,
                                uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP || !drawCount)
      return;
   const struct wvk_bound_buffer b = bound_buffer(cmd, _buffer, offset, (uint64_t)(drawCount - 1) * stride + 12);
   if (!b.buffer)
      return;
   wvk_replay_read(cmd->replay, b.mem, b.offset, b.size);
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU;
   for (uint32_t i = 0; i < drawCount; ++i)
      draw_mesh(cmd, 0, 0, 0, b.buffer, b.offset + (uint64_t)i * stride, NULL, 0, 0, i);
   cmd->replay->cond_gpu = false;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawMeshTasksIndirectCountEXT(VkCommandBuffer commandBuffer, VkBuffer _buffer, VkDeviceSize offset,
                                     VkBuffer countBuffer, VkDeviceSize countBufferOffset, uint32_t maxDrawCount,
                                     uint32_t stride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP || !maxDrawCount)
      return;
   const struct wvk_bound_buffer b = bound_buffer(cmd, _buffer, offset, (uint64_t)(maxDrawCount - 1) * stride + 12);
   const struct wvk_bound_buffer c = bound_buffer(cmd, countBuffer, countBufferOffset, 4);
   if (!b.buffer || !c.buffer)
      return;
   wvk_replay_read(cmd->replay, b.mem, b.offset, b.size);
   wvk_replay_read(cmd->replay, c.mem, c.offset, 4);
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU;
   for (uint32_t i = 0; i < maxDrawCount; ++i)
      draw_mesh(cmd, 0, 0, 0, b.buffer, b.offset + (uint64_t)i * stride, c.buffer, c.offset, maxDrawCount, i);
   cmd->replay->cond_gpu = false;
}

/* The tessellator's passes, made the first time (wvk_tessellator.c): one
 * layout of its five bindings for all three. */
static bool
tessellator_pipelines(struct wvk_device *dev)
{
   simple_mtx_lock(&dev->cache_lock);
   if (!dev->tess_bgl) {
      WGPUBindGroupLayoutEntry e[5];
      for (uint32_t i = 0; i < 5; ++i) {
         e[i] = (WGPUBindGroupLayoutEntry)WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
         e[i].binding = i;
         e[i].visibility = WGPUShaderStage_Compute;
         e[i].buffer.type = i == 0 ? WGPUBufferBindingType_Uniform
                            : i == 1 ? WGPUBufferBindingType_ReadOnlyStorage
                                     : WGPUBufferBindingType_Storage;
      }
      WGPUBindGroupLayoutDescriptor bd = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
      bd.label = wvk_sv("wvk tessellator");
      bd.entryCount = 5;
      bd.entries = e;
      dev->tess_bgl = wgpuDeviceCreateBindGroupLayout(dev->wgpu, &bd);
      WGPUPipelineLayoutDescriptor pd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
      pd.bindGroupLayoutCount = 1;
      pd.bindGroupLayouts = &dev->tess_bgl;
      WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(dev->wgpu, &pd);
      WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
      source.code = wvk_sv(wvk_tessellator_wgsl);
      WGPUShaderModuleDescriptor sm = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
      sm.nextInChain = &source.chain;
      sm.label = wvk_sv("wvk tessellator");
      WGPUShaderModule module = wgpuDeviceCreateShaderModule(dev->wgpu, &sm);
      static const char *const entries[3] = { "count", "scan", "generate" };
      for (uint32_t i = 0; i < 3; ++i) {
         WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
         desc.label = wvk_sv("wvk tessellator");
         desc.layout = layout;
         desc.compute.module = module;
         desc.compute.entryPoint = wvk_sv(entries[i]);
         dev->tess_pipelines[i] = wgpuDeviceCreateComputePipeline(dev->wgpu, &desc);
      }
      wgpuShaderModuleRelease(module);
      wgpuPipelineLayoutRelease(layout);
   }
   simple_mtx_unlock(&dev->cache_lock);
   return dev->tess_bgl && dev->tess_pipelines[0] && dev->tess_pipelines[1] && dev->tess_pipelines[2];
}

/* How big each of a tessellated draw's buffers may be (draw_tessellated):
 * the submit's (scratch), each a worst case, which a binding's whole would
 * make gigabytes of. */
#define WVK_TESS_BUDGET (256ull << 20)

/* A tessellated draw, which WebGPU has no stages of: the vertex shader run
 * as compute, a record a vertex (key.runs 3); the control shader run as
 * compute, a workgroup a patch, writing records of its vertices and patches
 * (key.tess 1); the tessellator, counting each patch's domain points and
 * indices, placing them, and making them (wvk_tessellator.c); the
 * evaluation shader run as compute, an invocation a point, as many as were
 * made (key.tess 1); then an indexed draw of the tessellator's lists, whose
 * vertex shader reads the evaluation shader's records (key.tess 2) -- both
 * counts of the last two the tessellator's, indirect. Not with a geometry
 * shader after. */
static void
draw_tessellated(struct wvk_cmd_buffer *cmd, bool indexed, uint32_t first, uint32_t count, uint32_t instances,
                 int32_t base_vertex, uint32_t first_instance, uint32_t draw_id)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   struct wvk_shader *tcs = st->bound_tcs, *tes = st->bound_tes;
   if (!st->bound_vs || !tcs || !tes || !count || !instances || replay->pass_idle)
      return;
   const uint32_t vertices_in = dyn->ts.patch_control_points;
   const uint32_t per_instance = vertices_in ? count / vertices_in : 0;
   const uint64_t patches = (uint64_t)per_instance * instances;
   const uint32_t vertices_out = MAX2(tcs->tess_vertices_out, tes->tess_vertices_out);
   if (!patches || !vertices_out || patches > 1u << 20)
      return;
   /* The domain, spacing and winding either stage may say. */
   const unsigned domain = tes->tess_domain ? tes->tess_domain : tcs->tess_domain;
   const unsigned spacing = tes->tess_spacing ? tes->tess_spacing : tcs->tess_spacing;
   const bool point_mode = tes->tess_point_mode || tcs->tess_point_mode;
   bool ccw = tes->tess_domain ? tes->tess_ccw : tcs->tess_ccw;
   /* The tessellator's triangles wind as Vulkan's upper-left domain does,
    * in the clip space vertex shaders write; a lower-left one the other way. */
   if (dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT)
      ccw = !ccw;
   const uint32_t tess_domain = domain == TESS_PRIMITIVE_QUADS ? 1 : domain == TESS_PRIMITIVE_ISOLINES ? 2 : 0;
   const uint32_t output = point_mode ? 0 : tess_domain == 2 ? 1 : ccw ? 3 : 2;
   const uint32_t partitioning = spacing == TESS_SPACING_FRACTIONAL_ODD    ? 2
                                 : spacing == TESS_SPACING_FRACTIONAL_EVEN ? 3
                                                                           : 0;

   /* The stages' compute shaders, and the draw's. */
   struct wvk_bound_views images;
   bound_views(st, WVK_BIND_GRAPHICS, st->bound_vs, &images);
   struct wvk_shader *vs =
      wvk_shader_draw_variant(dev, st->bound_vs, NULL, dyn, 1, false, NULL, &images, WVK_DRAW_PRIMITIVES, 3, 0);
   if (!vs || vs->vk.stage != MESA_SHADER_COMPUTE) {
      mesa_logw_once("wvk: a vertex shader before tessellation, which could not be made a compute shader");
      return;
   }
   struct wvk_shader *control = wvk_shader_tess_variant(dev, tcs, 1, vs->record_vars, 0, 0, NULL, false);
   struct wvk_shader *evaluation =
      control ? wvk_shader_tess_variant(dev, tes, 1, control->record_vars, control->patch_vars, vertices_out, NULL, false)
              : NULL;
   const bool quads = replay->draw_points;
   struct wvk_shader *passthrough = wvk_shader_tess_variant(dev, tes, 2, 0, 0, 0, st->fs, quads && !draw_gs(cmd));
   if (!control || !evaluation || !passthrough || !tessellator_pipelines(dev)) {
      mesa_logw_once("wvk: tessellation shaders that could not be made compute");
      return;
   }
   if (replay->pass_layered && !replay->pass_multiview && replay->pass_layer && !passthrough->writes_layer)
      return;

   /* The buffers: what each stage writes, and the tessellator's, as much as
    * a binding can hold of them -- and of what the tessellator and a
    * geometry shader after it may make, sized for the most the levels
    * allow, as much as WVK_TESS_BUDGET of each: a draw of more is drawn as
    * far as they hold, which the tessellator stops at. */
   const uint64_t limit = pdev->limits.maxStorageBufferBindingSize;
   const uint64_t budget = MIN2(limit, WVK_TESS_BUDGET);
   const uint32_t out_stride = wvk_record_stride(evaluation->record_vars);
   const uint64_t records = (uint64_t)count * instances * wvk_record_stride(vs->record_vars);
   const uint64_t control_bytes = patches * vertices_out * wvk_record_stride(control->record_vars);
   const uint64_t patch_bytes = patches * wvk_patch_stride(control->patch_vars);
   const uint64_t points_capacity = MIN3(patches * WVK_TESS_MAX_POINTS, MIN2(budget / out_stride, budget / 16) - 1,
                                         1u << 22);
   const uint64_t indices_capacity = MIN3(patches * WVK_TESS_MAX_INDICES, budget / 4, 1u << 24);
   const uint64_t counts_bytes = (patches * 4 + 28) * 4;
   if (records > limit || control_bytes > limit || patch_bytes > limit) {
      mesa_logw_once("wvk: a tessellated draw of more than its buffers can bind");
      return;
   }
   /* The submit's, draw after draw: each a worst case, which new ones for
    * each would have the submit hold all of at once. */
   const WGPUBufferUsage storage = WGPUBufferUsage_Storage;
   const WGPUBufferUsage indexed_storage = storage | WGPUBufferUsage_Index;
   WGPUBuffer capture = scratch(dev, replay, WVK_SCRATCH_TESS + 0, records, storage, true, &st->capture);
   WGPUBuffer control_out = scratch(dev, replay, WVK_SCRATCH_TESS + 1, control_bytes, storage, true, &st->tess[0]);
   WGPUBuffer patch_out = scratch(dev, replay, WVK_SCRATCH_TESS + 2, patch_bytes, storage, true, &st->tess[1]);
   WGPUBuffer points = scratch(dev, replay, WVK_SCRATCH_TESS + 3, points_capacity * 16, storage, true, &st->tess[2]);
   WGPUBuffer counts = scratch(dev, replay, WVK_SCRATCH_TESS + 4, counts_bytes, storage | WGPUBufferUsage_Indirect,
                               true, &st->tess[3]);
   WGPUBuffer list = scratch(dev, replay, WVK_SCRATCH_TESS + 5, indices_capacity * 4, indexed_storage, true, NULL);
   WGPUBuffer out = scratch(dev, replay, WVK_SCRATCH_TESS + 6, (points_capacity + 1) * out_stride, storage, true,
                            &st->gs_vertices);
   if (!capture || !control_out || !patch_out || !points || !counts || !list || !out)
      return;

   /* A geometry shader after: of the tessellator's primitives, which its
    * input has to be. */
   struct wvk_shader *gs = draw_gs(cmd), *gs_cs = NULL, *gs_draw = NULL;
   const uint32_t prim_vertices = output == 0 ? 1 : output == 1 ? 2 : 3;
   uint64_t gs_capacity = 0;
   uint32_t gs_per = 0, gs_streams = 1;
   WGPUBuffer gs_out = NULL, gs_list = NULL;
   if (gs) {
      if (mesa_vertices_per_prim(gs->gs_input) != prim_vertices) {
         mesa_logw_once("wvk: a geometry shader of other primitives than tessellation makes");
         return;
      }
      gs_cs = wvk_shader_geometry_variant(dev, gs, 1, evaluation->record_vars, NULL, dyn, true, false,
                                          gs == st->bound_gs ? wvk_replay_stats(replay) : 0);
      gs_draw = wvk_shader_geometry_variant(dev, gs, 2, 0, st->fs, dyn, false, quads, 0);
      if (!gs_cs || !gs_draw) {
         mesa_logw_once("wvk: a geometry shader after tessellation that could not be made compute");
         return;
      }
      gs_per = wvk_geometry_indices(gs->gs_output, gs->gs_max_vertices);
      gs_streams = MAX2(gs->gs_streams, 1);
      const uint64_t gs_stride = wvk_record_stride(gs->record_vars);
      /* Each vertex stream's vertices and lists, after the stream's before
       * (lower_geometry, which has the capacity as the draw's total). */
      gs_capacity = MIN3(indices_capacity / prim_vertices * gs->gs_invocations,
                         budget / (gs_streams * MAX2(gs->gs_max_vertices, 1) * gs_stride) - 1,
                         budget / (gs_streams * MAX2(gs_per, 1) * 4));
      gs_out = scratch(dev, replay, WVK_SCRATCH_TESS + 7, (1 + gs_capacity * gs_streams * gs->gs_max_vertices) * gs_stride,
                       storage, true, NULL);
      gs_list = scratch(dev, replay, WVK_SCRATCH_TESS + 8, gs_capacity * gs_streams * gs_per * 4, indexed_storage, true,
                        NULL);
      if (!gs_out || !gs_list)
         return;
      const uint32_t geometry[4] = { 0, 1, (uint32_t)gs_capacity, 0 };
      if (memcmp(st->push + WVK_DRIVER_GEOMETRY, geometry, sizeof(geometry))) {
         memcpy(st->push + WVK_DRIVER_GEOMETRY, geometry, sizeof(geometry));
         st->push_offset = UINT32_MAX;
      }
   }

   /* The vertex and control shaders. */
   if (!run_vertices(cmd, 3, indexed, first, count, instances, base_vertex, first_instance, draw_id))
      return;
   const uint32_t tess[4] = { count, per_instance, vertices_in, (uint32_t)patches };
   if (memcmp(st->push + WVK_DRIVER_TESS, tess, sizeof(tess))) {
      memcpy(st->push + WVK_DRIVER_TESS, tess, sizeof(tess));
      st->push_offset = UINT32_MAX;
   }
   dispatch_graphics_groups(cmd, control, patches, NULL, 0);

   /* The tessellator. */
   const uint32_t params[12] = { (uint32_t)patches, tess_domain, partitioning, output,
                                 wvk_patch_stride(control->patch_vars) / 4, (uint32_t)points_capacity,
                                 (uint32_t)indices_capacity, 0,
                                 gs ? prim_vertices : 0, gs ? gs->gs_invocations : 0, (uint32_t)gs_capacity, gs_per };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   if (!params_buffer)
      return;
   const WGPUBindGroupEntry entries[5] = {
      { .binding = 0, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
      { .binding = 1, .buffer = patch_out, .offset = 0, .size = MAX2(patch_bytes, 16) },
      { .binding = 2, .buffer = counts, .offset = 0, .size = counts_bytes },
      { .binding = 3, .buffer = points, .offset = 0, .size = points_capacity * 16 },
      { .binding = 4, .buffer = list, .offset = 0, .size = MAX2(indices_capacity * 4, 16) },
   };
   WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bgd.layout = dev->tess_bgl;
   bgd.entryCount = 5;
   bgd.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bgd);
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   WGPUComputePassDescriptor cpd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &cpd);
   const uint32_t groups = (uint32_t)DIV_ROUND_UP(patches, 64);
   for (uint32_t i = 0; i < 3; ++i) {
      wgpuComputePassEncoderSetPipeline(pass, dev->tess_pipelines[i]);
      wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
      if (i == 1)
         wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
      else
         wgpuComputePassEncoderDispatchWorkgroups(pass, MIN2(groups, 65535), DIV_ROUND_UP(groups, 65535), 1);
   }
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   if (group)
      wgpuBindGroupRelease(group);
   /* For the pipeline statistics query: the evaluation shader's
    * invocations, a point each; and the primitives a geometry shader has,
    * or clipping where there is none. */
   const uint32_t t = (uint32_t)patches * 4;
   wvk_replay_add_stats(replay, counts, counts_bytes, t, 1,
                        VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_EVALUATION_SHADER_INVOCATIONS_BIT);
   /* The pipeline's own geometry shader counts; the driver's, for cull
    * distances or transform feedback, is none. */
   if (gs && gs == st->bound_gs)
      wvk_replay_add_stats(replay, counts, counts_bytes, t + 13, 1,
                           VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT);
   else
      wvk_replay_add_stats(replay, counts, counts_bytes, t + 1, prim_vertices,
                           VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                              VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                              BITFIELD_BIT(WVK_COUNTER_PRIMITIVES(0)));

   /* The evaluation shader, of as many points as were made, then the draw
    * of their lists -- or of what a geometry shader makes of them, run as
    * compute of as many as its buffers hold. */
   dispatch_graphics_groups(cmd, evaluation, 0, counts, (patches * 4 + 2) * 4);
   WGPUBuffer draw_list = list;
   uint64_t draw_bytes = MAX2(indices_capacity * 4, 16), draw_command = (patches * 4 + 5) * 4, draw_start = 0;
   WGPUPrimitiveTopology topology = output == 0   ? WGPUPrimitiveTopology_PointList
                                    : output == 1 ? WGPUPrimitiveTopology_LineList
                                                  : WGPUPrimitiveTopology_TriangleList;
   if (gs) {
      const uint64_t gs_stride = wvk_record_stride(gs->record_vars);
#define TEMP(b, size) ((struct wvk_driver_buffer){ b, ++dev->temp_serial | (1ull << 62), 0, size })
      st->capture = TEMP(out, (points_capacity + 1) * out_stride);
      st->index = TEMP(list, MAX2(indices_capacity * 4, 16));
      st->gs_vertices = TEMP(gs_out, (1 + gs_capacity * gs_streams * gs->gs_max_vertices) * gs_stride);
      st->gs_indices = TEMP(gs_list, MAX2(gs_capacity * gs_streams * gs_per * 4, 16));
#undef TEMP
      dispatch_graphics_groups(cmd, gs_cs, 0, counts, (patches * 4 + 10) * 4);
      capture_xfb(cmd, gs, &st->gs_vertices, &st->gs_indices, 0, gs_capacity, counts, counts_bytes,
                  (uint32_t)patches * 4 + 13);
      /* The rasterized stream's lists, of a stream it emits to. */
      const uint32_t raster = dyn->rs.rasterization_stream;
      if (raster >= gs_streams) {
         if (in_pass && !replay->pass)
            wvk_replay_restart_pass(cmd, replay->pass_layer);
         return;
      }
      passthrough = gs_draw;
      draw_list = gs_list;
      draw_start = raster * gs_capacity * gs_per * 4;
      draw_bytes = MAX2(gs_capacity * gs_per * 4, 16);
      draw_command = (patches * 4 + 14) * 4;
      topology = gs->gs_output == MESA_PRIM_TRIANGLE_STRIP ? WGPUPrimitiveTopology_TriangleList
                 : gs->gs_output == MESA_PRIM_LINE_STRIP  ? WGPUPrimitiveTopology_LineList
                                                          : WGPUPrimitiveTopology_PointList;
   }
   if (in_pass && !replay->pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   if (!gs && rasterizes_other_stream(cmd))
      return;
   struct wvk_shader *draw_vs = st->vs;
   if (quads) {
      /* Points as quads, six vertices an index, of the second command. */
      const struct wvk_driver_buffer l = { draw_list, ++dev->temp_serial | (1ull << 62), 0, draw_start + draw_bytes };
      st->vs = passthrough;
      draw_point_list(cmd, &l, (uint32_t)draw_start, 0, counts, (patches * 4 + (gs ? 23 : 19)) * 4, draw_id);
      st->vs = draw_vs;
      return;
   }
   if (passthrough->key.bary || passthrough->key.per_vertex) {
      const struct wvk_driver_buffer l = { draw_list, ++dev->temp_serial | (1ull << 62), 0, draw_start + draw_bytes };
      st->vs = passthrough;
      draw_pulled_list(cmd, &l, (uint32_t)draw_start, 0, topology, counts, draw_command, draw_id);
      st->vs = draw_vs;
      return;
   }
   const struct wvk_bound_buffer ib = st->ib;
   const VkIndexType type = st->index_type;
   st->vs = passthrough;
   st->ib = (struct wvk_bound_buffer){ NULL, draw_list, draw_start, draw_bytes };
   st->index_type = VK_INDEX_TYPE_UINT32;
   replay->list_topology = topology;
   WGPUBuffer draws = counts;
   if (replay->cond_gpu) {
      const struct wvk_bound_buffer src = { NULL, counts, draw_command, 20, ++dev->temp_serial };
      struct wvk_bound_buffer b;
      if (conditional_args(cmd, NULL, &src, 1, 5, 20, &b)) {
         draws = b.buffer;
         draw_command = 0;
      }
   }
   foreach_viewport(cmd) {
      set_draw_parameters(st, NULL, 0, true, 0, 0, draw_id);
      if (!flush_draw(cmd, true))
         continue;
      wgpuRenderPassEncoderDrawIndexedIndirect(replay->pass, draws, draw_command);
   }
   replay->list_topology = WGPUPrimitiveTopology_Undefined;
   st->ib = ib;
   st->index_type = type;
   st->vs = draw_vs;
}

/* The primitives of `count` vertices of a topology, for the pipeline
 * statistics query. */
static uint64_t
topology_primitive_count(VkPrimitiveTopology topology, uint32_t count, uint32_t patch_vertices)
{
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return count;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: return count / 2;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: return count > 1 ? count - 1 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: return count / 3;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN: return count > 2 ? count - 2 : 0;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY: return count / 4;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return count > 3 ? count - 3 : 0;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY: return count / 6;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: return count >= 6 ? (count - 4) / 2 : 0;
   case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: return patch_vertices ? count / patch_vertices : 0;
   default: return 0;
   }
}

/* What the host knows of a draw's pipeline statistics: its vertices and
 * primitives, and its vertex shader's invocations -- one a vertex, reused
 * or not -- then a geometry shader's invocations, the control shader's
 * patches, or what clipping has where there is neither. The GPU counts the
 * rest: fragment shader invocations, what geometry and evaluation shaders
 * make. */
static void
count_draw(struct wvk_cmd_buffer *cmd, uint32_t count, uint32_t instances)
{
   struct wvk_replay *replay = cmd->replay;
   if (!wvk_replay_stats(replay) || !count || !instances)
      return;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   const struct wvk_bound_state *st = replay->state;
   const uint64_t vertices = (uint64_t)count * instances;
   const uint64_t prims =
      topology_primitive_count(dyn->ia.primitive_topology, count, dyn->ts.patch_control_points) * instances;
   wvk_replay_count_stat(replay, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_VERTICES_BIT, vertices);
   wvk_replay_count_stat(replay, VK_QUERY_PIPELINE_STATISTIC_INPUT_ASSEMBLY_PRIMITIVES_BIT, prims);
   wvk_replay_count_stat(replay, VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT, vertices);
   if (st->bound_tes) {
      wvk_replay_count_stat(replay, VK_QUERY_PIPELINE_STATISTIC_TESSELLATION_CONTROL_SHADER_PATCHES_BIT, prims);
   } else if (st->bound_gs) {
      wvk_replay_count_stat(replay, VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_INVOCATIONS_BIT,
                            prims * st->bound_gs->gs_invocations);
   } else {
      wvk_replay_count_stat(replay, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT, prims);
      wvk_replay_count_stat(replay, VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT, prims);
      wvk_replay_count(replay, WVK_COUNTER_PRIMITIVES(0), prims);
   }
}

/* Of indirect draws, as the host has them. */
static void
count_indirect_draws(struct wvk_cmd_buffer *cmd, bool indexed, struct wvk_bound_buffer b, uint32_t draws,
                     uint32_t stride)
{
   if (!wvk_replay_stats(cmd->replay))
      return;
   if (!b.mem || !b.mem->host) {
      mesa_logw_once("wvk: an indirect draw in memory the host has no copy of, which pipeline statistics miss");
      return;
   }
   for (uint32_t i = 0; i < draws; ++i) {
      uint32_t words[2];
      memcpy(words, (const uint8_t *)b.mem->host + b.offset + (uint64_t)i * stride, sizeof(words));
      count_draw(cmd, words[0], words[1]);
   }
   (void)indexed;
}

static void
draw_arrays(struct wvk_cmd_buffer *cmd, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
            uint32_t firstInstance, uint32_t draw_id)
{
   if (!vertexCount || !instanceCount)
      return;
   count_draw(cmd, vertexCount, instanceCount);
   cmd->replay->draw_points = draws_points(cmd);
   cmd->replay->draw_lines = draws_lines(cmd);
   select_draw_shaders(cmd);
   if (cmd->replay->state->bound_tes || draw_gs(cmd)) {
      (cmd->replay->state->bound_tes ? draw_tessellated : draw_geometry)(cmd, false, firstVertex, vertexCount,
                                                                         instanceCount, 0, firstInstance, draw_id);
      cmd->replay->draw_points = cmd->replay->draw_lines = false;
      return;
   }
   run_vertex_writes(cmd, false, firstVertex, vertexCount, instanceCount, 0, firstInstance, draw_id);
   if (rasterizes_other_stream(cmd)) {
      cmd->replay->draw_points = cmd->replay->draw_lines = false;
      return;
   }
   if (cmd->replay->draw_points || cmd->replay->draw_lines) {
      if (cmd->replay->draw_points)
         draw_points(cmd, false, firstVertex, vertexCount, instanceCount, 0, firstInstance, draw_id);
      else
         draw_lines(cmd, false, firstVertex, vertexCount, instanceCount, 0, firstInstance, draw_id);
      cmd->replay->draw_points = cmd->replay->draw_lines = false;
      return;
   }
   if (draw_polygon_mode(cmd, false, 0, vertexCount, instanceCount, (int32_t)firstVertex, firstInstance, draw_id))
      return;
   if (cmd->vk.dynamic_graphics_state.ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
      draw_fan_arrays(cmd, vertexCount, instanceCount, firstVertex, firstInstance, draw_id);
      return;
   }
   WGPUBuffer predicated =
      predicated_draw(cmd, (const uint32_t[4]){ vertexCount, instanceCount, firstVertex, firstInstance }, 4);
   foreach_viewport(cmd) {
      set_draw_parameters(cmd->replay->state, NULL, 0, false, (int32_t)firstVertex, firstInstance, draw_id);
      if (!flush_draw(cmd, false))
         continue;
      if (predicated)
         wgpuRenderPassEncoderDrawIndirect(cmd->replay->pass, predicated, 0);
      else
         wgpuRenderPassEncoderDraw(cmd->replay->pass, vertexCount, instanceCount, firstVertex, firstInstance);
   }
}

static void draw_elements_widened(struct wvk_cmd_buffer *cmd, uint32_t indexCount, uint32_t instanceCount,
                                  uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance, uint32_t draw_id);

static void
draw_elements(struct wvk_cmd_buffer *cmd, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
              int32_t vertexOffset, uint32_t firstInstance, uint32_t draw_id)
{
   if (!indexCount || !instanceCount)
      return;
   struct wvk_bound_buffer saved;
   const bool widened = widen_indices(cmd, &saved);
   draw_elements_widened(cmd, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, draw_id);
   if (widened)
      narrow_indices(cmd, &saved);
}

/* Whether a draw's primitives restart, of a list (primitiveTopologyListRestart):
 * WebGPU restarts strips only. Not a patch list's, which tessellation
 * reads the indices of itself. */
static bool
restarts_list(const struct wvk_cmd_buffer *cmd)
{
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!dyn->ia.primitive_restart_enable)
      return false;
   switch (dyn->ia.primitive_topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST:
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY:
      return true;
   default:
      return false;
   }
}

static void draw_elements_listed(struct wvk_cmd_buffer *cmd, uint32_t indexCount, uint32_t instanceCount,
                                 uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance, uint32_t draw_id);
static void draw_list_restarts(struct wvk_cmd_buffer *cmd, uint32_t indexCount, uint32_t instanceCount,
                               uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance, uint32_t draw_id);

static void
draw_elements_widened(struct wvk_cmd_buffer *cmd, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
                      int32_t vertexOffset, uint32_t firstInstance, uint32_t draw_id)
{
   if (!indexCount || !instanceCount)
      return;
   count_draw(cmd, indexCount, instanceCount);
   cmd->replay->draw_points = draws_points(cmd);
   cmd->replay->draw_lines = draws_lines(cmd);
   select_draw_shaders(cmd);
   if (cmd->replay->state->bound_tes || draw_gs(cmd)) {
      (cmd->replay->state->bound_tes ? draw_tessellated : draw_geometry)(cmd, true, firstIndex, indexCount,
                                                                         instanceCount, vertexOffset, firstInstance,
                                                                         draw_id);
      cmd->replay->draw_points = cmd->replay->draw_lines = false;
      return;
   }
   if (restarts_list(cmd))
      draw_list_restarts(cmd, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, draw_id);
   else
      draw_elements_listed(cmd, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, draw_id);
}

/* A list that restarts: drawn of the driver's list of its whole
 * primitives, the restart index to its end, which the draw is of as many
 * of as there are, and the quads of points and wide lines and the
 * driver's lists of the rest leave out. */
static void
draw_list_restarts(struct wvk_cmd_buffer *cmd, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
                   int32_t vertexOffset, uint32_t firstInstance, uint32_t draw_id)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   uint32_t count = 0;
   uint64_t args;
   WGPUBuffer list = triangles_on_gpu(cmd, true, firstIndex, indexCount, LIST_COMPACT, false,
                                      (const uint32_t[3]){ instanceCount, (uint32_t)vertexOffset, firstInstance },
                                      &count, &args);
   if (!list) {
      replay->draw_points = replay->draw_lines = false;
      return;
   }
   const struct wvk_bound_buffer ib = st->ib;
   const VkIndexType type = st->index_type;
   st->ib = (struct wvk_bound_buffer){ NULL, list, 0, (uint64_t)count * 4,
                                       ++wvk_cmd_buffer_device(cmd)->temp_serial | (1ull << 62) };
   st->index_type = VK_INDEX_TYPE_UINT32;
   replay->restart_list = list;
   replay->restart_args = args;
   draw_elements_listed(cmd, count, instanceCount, 0, vertexOffset, firstInstance, draw_id);
   replay->restart_list = NULL;
   st->ib = ib;
   st->index_type = type;
}

/* The same, of the draw's indices as WebGPU's are, or the driver's list
 * of them: after its geometry and tessellation, which read indices of
 * their own. */
static void
draw_elements_listed(struct wvk_cmd_buffer *cmd, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
                     int32_t vertexOffset, uint32_t firstInstance, uint32_t draw_id)
{
   run_vertex_writes(cmd, true, firstIndex, indexCount, instanceCount, vertexOffset, firstInstance, draw_id);
   if (rasterizes_other_stream(cmd)) {
      cmd->replay->draw_points = cmd->replay->draw_lines = false;
      return;
   }
   if (cmd->replay->draw_points || cmd->replay->draw_lines) {
      if (cmd->replay->draw_points)
         draw_points(cmd, true, firstIndex, indexCount, instanceCount, vertexOffset, firstInstance, draw_id);
      else
         draw_lines(cmd, true, firstIndex, indexCount, instanceCount, vertexOffset, firstInstance, draw_id);
      cmd->replay->draw_points = cmd->replay->draw_lines = false;
      return;
   }
   if (draw_polygon_mode(cmd, true, firstIndex, indexCount, instanceCount, vertexOffset, firstInstance, draw_id))
      return;
   if (cmd->vk.dynamic_graphics_state.ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
      draw_fan_elements(cmd, firstIndex, indexCount, instanceCount, vertexOffset, firstInstance, draw_id);
      return;
   }
   if (cmd->replay->restart_list) {
      draw_list(cmd, cmd->replay->restart_list, indexCount, WGPUPrimitiveTopology_Undefined, instanceCount,
                vertexOffset, firstInstance, true, draw_id, cmd->replay->restart_args);
      return;
   }
   WGPUBuffer predicated = predicated_draw(
      cmd, (const uint32_t[5]){ indexCount, instanceCount, firstIndex, (uint32_t)vertexOffset, firstInstance }, 5);
   foreach_viewport(cmd) {
      set_draw_parameters(cmd->replay->state, NULL, 0, true, vertexOffset, firstInstance, draw_id);
      if (!flush_draw(cmd, true))
         continue;
      if (predicated)
         wgpuRenderPassEncoderDrawIndexedIndirect(cmd->replay->pass, predicated, 0);
      else
         wgpuRenderPassEncoderDrawIndexed(cmd->replay->pass, indexCount, instanceCount, firstIndex,
                                          vertexOffset, firstInstance);
   }
}

/* ---------------------------------------------------------------------- */
/* Conditional rendering (VK_EXT_conditional_rendering): the predicate as
 * the host has it -- in memory of its own, which nothing of the submit
 * or before it still to be read back writes -- or else the GPU's: the
 * command's arguments copied by a pass of the driver's, zero where the
 * predicate fails, and the command made an indirect one of them. */

enum wvk_condition
wvk_cmd_condition(struct wvk_cmd_buffer *cmd)
{
   struct wvk_replay *replay = cmd->replay;
   if (!replay->cond.active)
      return WVK_CONDITION_PASS;
   struct wvk_device_memory *mem = replay->cond.mem;
   const uint64_t offset = replay->cond.offset;
   if (mem && mem->host && !wvk_replay_written(replay, mem, offset, 4) &&
       !wvk_queue_range_pending(&wvk_cmd_buffer_device(cmd)->queue, mem, offset, 4)) {
      uint32_t value;
      memcpy(&value, (const uint8_t *)mem->host + offset, 4);
      return (value != 0) != replay->cond.inverted ? WVK_CONDITION_PASS : WVK_CONDITION_SKIP;
   }
   return WVK_CONDITION_GPU;
}

static const char cond_wgsl[] =
   "@group(0) @binding(0) var<storage, read> p: array<u32>;\n"
   "@group(0) @binding(1) var<storage, read> pred: array<u32>;\n"
   "@group(0) @binding(2) var<storage, read> src: array<u32>;\n"
   "@group(0) @binding(3) var<storage, read_write> dst: array<u32>;\n"
   "@compute @workgroup_size(64)\n"
   "fn main(@builtin(global_invocation_id) id: vec3u) {\n"
   "  let r = id.x;\n"
   "  if (r >= p[2]) { return; }\n"
   "  let on = (pred[p[0]] != 0u) != (p[1] != 0u);\n"
   "  for (var w = 0u; w < p[3]; w++) {\n"
   "    var v = 0u;\n"
   "    if (on) {\n"
   "      if (p[6] != 0u) { v = src[p[4] + r * p[5] + w]; } else { v = p[8u + r * p[3] + w]; }\n"
   "    }\n"
   "    dst[r * p[3] + w] = v;\n"
   "  }\n"
   "}\n";

/* The GPU's: `records` of `words` words, the host's `host` or `src`'s a
 * `stride` apart, to a buffer of the driver's of them packed -- outside
 * any pass, the pass ended for it and begun again. */
static bool
conditional_args(struct wvk_cmd_buffer *cmd, const uint32_t *host, const struct wvk_bound_buffer *src,
                 uint32_t records, uint32_t words, uint32_t stride, struct wvk_bound_buffer *out)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   if (!records || !replay->cond.buffer || (src && !src->buffer))
      return false;
   WGPUComputePipeline pipeline = xfb_pipeline(dev, &dev->cond_pipeline, cond_wgsl, "wvk conditional rendering");
   const uint32_t n = 8 + (host ? records * words : 0);
   uint32_t *params = calloc(n, sizeof(uint32_t));
   if (!pipeline || !params) {
      free(params);
      return false;
   }
   const uint64_t pred_base = replay->cond.offset & ~255ull;
   const uint64_t pred_end = replay->cond.mem ? MIN2(pred_base + 256, replay->cond.mem->buffer_size)
                                              : replay->cond.offset + 4;
   const uint64_t src_base = src ? src->offset & ~255ull : 0;
   const uint64_t src_bytes = src ? align64(src->offset - src_base + (uint64_t)(records - 1) * stride + words * 4, 4)
                                  : 16;
   params[0] = (uint32_t)((replay->cond.offset - pred_base) / 4);
   params[1] = replay->cond.inverted;
   params[2] = records;
   params[3] = words;
   params[4] = src ? (uint32_t)((src->offset - src_base) / 4) : 0;
   params[5] = stride / 4;
   params[6] = src != NULL;
   if (host)
      memcpy(params + 8, host, (size_t)records * words * 4);
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, n * 4, WGPUBufferUsage_Storage, params);
   free(params);
   const uint64_t bytes = (uint64_t)records * words * 4;
   WGPUBuffer dst = wvk_replay_temp_buffer(dev, replay, bytes + WVK_DRAW_WINDOW,
                                           WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect |
                                              WGPUBufferUsage_CopySrc, NULL);
   WGPUBuffer none = src ? NULL : wvk_replay_temp_buffer(dev, replay, 16, WGPUBufferUsage_Storage, NULL);
   if (!params_buffer || !dst || (!src && !none))
      return false;
   wvk_replay_read(replay, replay->cond.mem, replay->cond.offset, 4);
   if (src)
      wvk_replay_read(replay, src->mem, src->offset, src_bytes - (src->offset - src_base));
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = params_buffer, .offset = 0, .size = align64(n * 4, 4) },
      { .binding = 1, .buffer = replay->cond.buffer, .offset = pred_base, .size = align64(pred_end - pred_base, 4) },
      { .binding = 2, .buffer = src ? src->buffer : none, .offset = src_base, .size = src_bytes },
      { .binding = 3, .buffer = dst, .offset = 0, .size = align64(bytes, 4) },
   };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
   bg.entryCount = ARRAY_SIZE(entries);
   bg.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bg.layout);
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderSetPipeline(pass, pipeline);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuComputePassEncoderDispatchWorkgroups(pass, DIV_ROUND_UP(records, 64), 1, 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   *out = (struct wvk_bound_buffer){ NULL, dst, 0, bytes, ++dev->temp_serial | (1ull << 62) };
   return true;
}

static void draw_indirect_bound(struct wvk_cmd_buffer *cmd, bool indexed, struct wvk_bound_buffer b,
                                uint32_t drawCount, uint32_t stride);

/* Draws of the arguments conditional_args made. */
static void
draw_conditional(struct wvk_cmd_buffer *cmd, bool indexed, struct wvk_bound_buffer b, uint32_t draws)
{
   const uint32_t stride = indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
   struct wvk_bound_buffer saved;
   const bool widened = indexed && widen_indices(cmd, &saved);
   select_draw_shaders(cmd);
   draw_indirect_bound(cmd, indexed, b, draws, stride);
   if (widened)
      narrow_indices(cmd, &saved);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBeginConditionalRenderingEXT(VkCommandBuffer commandBuffer,
                                    const VkConditionalRenderingBeginInfoEXT *pConditionalRenderingBegin)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   const struct wvk_bound_buffer b =
      bound_buffer(cmd, pConditionalRenderingBegin->buffer, pConditionalRenderingBegin->offset, 4);
   cmd->replay->cond.mem = b.mem;
   cmd->replay->cond.buffer = b.buffer;
   cmd->replay->cond.offset = b.offset;
   cmd->replay->cond.inverted =
      pConditionalRenderingBegin->flags & VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;
   cmd->replay->cond.active = b.buffer != NULL;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   cmd->replay->cond.active = false;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDraw(VkCommandBuffer commandBuffer, uint32_t vertexCount, uint32_t instanceCount,
            uint32_t firstVertex, uint32_t firstInstance)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU;
   draw_arrays(cmd, vertexCount, instanceCount, firstVertex, firstInstance, 0);
   cmd->replay->cond_gpu = false;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndexed(VkCommandBuffer commandBuffer, uint32_t indexCount, uint32_t instanceCount,
                   uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU;
   draw_elements(cmd, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance, 0);
   cmd->replay->cond_gpu = false;
}

/* VK_EXT_multi_draw: a draw for each, DrawIndex its number. */
VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawMultiEXT(VkCommandBuffer commandBuffer, uint32_t drawCount, const VkMultiDrawInfoEXT *pVertexInfo,
                    uint32_t instanceCount, uint32_t firstInstance, uint32_t stride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU;
   for (uint32_t i = 0; i < drawCount; ++i) {
      const VkMultiDrawInfoEXT *d = (const VkMultiDrawInfoEXT *)((const uint8_t *)pVertexInfo + (size_t)i * stride);
      draw_arrays(cmd, d->vertexCount, instanceCount, d->firstVertex, firstInstance, i);
   }
   cmd->replay->cond_gpu = false;
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawMultiIndexedEXT(VkCommandBuffer commandBuffer, uint32_t drawCount,
                           const VkMultiDrawIndexedInfoEXT *pIndexInfo, uint32_t instanceCount,
                           uint32_t firstInstance, uint32_t stride, const int32_t *pVertexOffset)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU;
   for (uint32_t i = 0; i < drawCount; ++i) {
      const VkMultiDrawIndexedInfoEXT *d =
         (const VkMultiDrawIndexedInfoEXT *)((const uint8_t *)pIndexInfo + (size_t)i * stride);
      draw_elements(cmd, d->indexCount, instanceCount, d->firstIndex, pVertexOffset ? *pVertexOffset : d->vertexOffset,
                    firstInstance, i);
   }
   cmd->replay->cond_gpu = false;
}

static void
draw_indirect_bound(struct wvk_cmd_buffer *cmd, bool indexed, struct wvk_bound_buffer b, uint32_t drawCount,
                    uint32_t stride)
{
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   if (!drawCount || !b.buffer)
      return;
   count_indirect_draws(cmd, indexed, b, drawCount, stride);
   /* Tessellation and geometry shaders, whose passes need the draws'
    * counts: as the host has the draws. */
   if (st->bound_tes || draw_gs(cmd)) {
      if (!b.mem || !b.mem->host) {
         mesa_logw_once("wvk: an indirect draw with tessellation or a geometry shader, in memory the host has no "
                        "copy of");
         return;
      }
      for (uint32_t i = 0; i < drawCount; ++i) {
         const uint8_t *at = (const uint8_t *)b.mem->host + b.offset + (uint64_t)i * stride;
         if (indexed) {
            VkDrawIndexedIndirectCommand c;
            memcpy(&c, at, sizeof(c));
            (st->bound_tes ? draw_tessellated : draw_geometry)(cmd, true, c.firstIndex, c.indexCount, c.instanceCount,
                                                              c.vertexOffset, c.firstInstance, i);
         } else {
            VkDrawIndirectCommand c;
            memcpy(&c, at, sizeof(c));
            (st->bound_tes ? draw_tessellated : draw_geometry)(cmd, false, c.firstVertex, c.vertexCount,
                                                              c.instanceCount, 0, c.firstInstance, i);
         }
      }
      return;
   }
   /* Lists that restart, each draw as the host has it (draw_list_restarts). */
   if (indexed && restarts_list(cmd)) {
      if (!b.mem || !b.mem->host) {
         mesa_logw_once("wvk: an indirect draw of a list that restarts, in memory the host has no copy of");
         return;
      }
      const bool points = replay->draw_points, lines = replay->draw_lines;
      for (uint32_t i = 0; i < drawCount; ++i) {
         VkDrawIndexedIndirectCommand c;
         memcpy(&c, (const uint8_t *)b.mem->host + b.offset + (uint64_t)i * stride, sizeof(c));
         replay->draw_points = points;
         replay->draw_lines = lines;
         if (c.indexCount && c.instanceCount)
            draw_list_restarts(cmd, c.indexCount, c.instanceCount, c.firstIndex, c.vertexOffset, c.firstInstance, i);
      }
      return;
   }
   /* The vertex shader's storage writes, of each draw as the host has it. */
   if (st->bound_vs && st->bound_vs->writes_storage) {
      if (!b.mem || !b.mem->host) {
         mesa_logw_once("wvk: an indirect draw's vertex shader storage writes, in memory the host has no copy of");
      } else {
         for (uint32_t i = 0; i < drawCount; ++i) {
            const uint8_t *at = (const uint8_t *)b.mem->host + b.offset + (uint64_t)i * stride;
            if (indexed) {
               VkDrawIndexedIndirectCommand c;
               memcpy(&c, at, sizeof(c));
               run_vertex_writes(cmd, true, c.firstIndex, c.indexCount, c.instanceCount, c.vertexOffset,
                                 c.firstInstance, i);
            } else {
               VkDrawIndirectCommand c;
               memcpy(&c, at, sizeof(c));
               run_vertex_writes(cmd, false, c.firstVertex, c.vertexCount, c.instanceCount, 0, c.firstInstance, i);
            }
         }
      }
   }
   if (rasterizes_other_stream(cmd))
      return;
   /* Points or lines as quads, as the host has the draws (draw_indirect). */
   if (replay->draw_points || replay->draw_lines) {
      for (uint32_t i = 0; i < drawCount; ++i) {
         const uint8_t *at = (const uint8_t *)b.mem->host + b.offset + (uint64_t)i * stride;
         if (indexed) {
            VkDrawIndexedIndirectCommand c;
            memcpy(&c, at, sizeof(c));
            (replay->draw_points ? draw_points : draw_lines)(cmd, true, c.firstIndex, c.indexCount, c.instanceCount,
                                                            c.vertexOffset, c.firstInstance, i);
         } else {
            VkDrawIndirectCommand c;
            memcpy(&c, at, sizeof(c));
            (replay->draw_points ? draw_points : draw_lines)(cmd, false, c.firstVertex, c.vertexCount,
                                                            c.instanceCount, 0, c.firstInstance, i);
         }
      }
      return;
   }
   /* A fan's draws, and those of lines or points of triangles, as the
    * host has them, one by one as the lists the driver makes. */
   enum list_mode mode;
   WGPUPrimitiveTopology list;
   const bool polygon = polygon_list(cmd, &mode, &list, b.mem && b.mem->host);
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

static void draw_indirect_widened(struct wvk_cmd_buffer *cmd, bool indexed, VkBuffer buffer, VkDeviceSize offset,
                                  uint32_t drawCount, uint32_t stride);

/* Whether indirect draws are drawn from the host's copy of them: of
 * tessellation, a geometry shader, storage writes, points or lines as
 * quads, fans, polygon modes, or lists that restart. Under a predicate the GPU has, those
 * are each predicated as they are drawn (cond_gpu). */
static bool
indirect_on_host(struct wvk_cmd_buffer *cmd)
{
   const struct wvk_bound_state *st = cmd->replay->state;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   return st->bound_tes || draw_gs(cmd) || (st->bound_vs && st->bound_vs->writes_storage) || draws_points(cmd) ||
          draws_lines(cmd) || dyn->ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN ||
          dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL || restarts_list(cmd);
}

static void
draw_indirect(struct wvk_cmd_buffer *cmd, bool indexed, VkBuffer buffer, VkDeviceSize offset, uint32_t drawCount,
              uint32_t stride)
{
   struct wvk_bound_buffer saved;
   const bool widened = indexed && drawCount && widen_indices(cmd, &saved);
   draw_indirect_widened(cmd, indexed, buffer, offset, drawCount, stride);
   if (widened)
      narrow_indices(cmd, &saved);
}

static void
draw_indirect_widened(struct wvk_cmd_buffer *cmd, bool indexed, VkBuffer buffer, VkDeviceSize offset,
                      uint32_t drawCount, uint32_t stride)
{
   if (!drawCount)
      return;
   const uint32_t size = indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
   struct wvk_bound_buffer b = bound_buffer(cmd, buffer, offset, (uint64_t)(drawCount - 1) * stride + size);
   /* The draws, of none where the predicate fails. */
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   if (cond == WVK_CONDITION_GPU && !indirect_on_host(cmd)) {
      const struct wvk_bound_buffer src = b;
      if (!conditional_args(cmd, NULL, &src, drawCount, size / 4, stride, &b))
         return;
      stride = size;
   }
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU && b.mem;
   /* Points as quads need the draws' counts, which the host has where the
    * memory is its too. */
   cmd->replay->draw_points = draws_points(cmd) && b.mem && b.mem->host;
   cmd->replay->draw_lines = draws_lines(cmd) && b.mem && b.mem->host;
   select_draw_shaders(cmd);
   draw_indirect_bound(cmd, indexed, b, drawCount, stride);
   cmd->replay->draw_points = cmd->replay->draw_lines = false;
   cmd->replay->cond_gpu = false;
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

static void draw_indirect_count_widened(struct wvk_cmd_buffer *cmd, bool indexed, VkBuffer buffer,
                                        VkDeviceSize offset, VkBuffer count_buffer, VkDeviceSize count_offset,
                                        uint32_t max_draws, uint32_t stride);

static void
draw_indirect_count(struct wvk_cmd_buffer *cmd, bool indexed, VkBuffer buffer, VkDeviceSize offset,
                    VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_draws, uint32_t stride)
{
   struct wvk_bound_buffer saved;
   const bool widened = indexed && max_draws && widen_indices(cmd, &saved);
   draw_indirect_count_widened(cmd, indexed, buffer, offset, count_buffer, count_offset, max_draws, stride);
   if (widened)
      narrow_indices(cmd, &saved);
}

static void draw_indirect_count_bound(struct wvk_cmd_buffer *cmd, bool indexed, struct wvk_bound_buffer b,
                                      struct wvk_bound_buffer c, uint32_t max_draws, uint32_t stride);

static void
draw_indirect_count_widened(struct wvk_cmd_buffer *cmd, bool indexed, VkBuffer buffer, VkDeviceSize offset,
                            VkBuffer count_buffer, VkDeviceSize count_offset, uint32_t max_draws, uint32_t stride)
{
   if (!max_draws)
      return;
   const uint32_t size = indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
   const uint64_t bytes = (uint64_t)(max_draws - 1) * stride + size;
   struct wvk_bound_buffer b = bound_buffer(cmd, buffer, offset, bytes);
   const struct wvk_bound_buffer c = bound_buffer(cmd, count_buffer, count_offset, 4);
   if (!b.buffer || !c.buffer)
      return;
   /* The draws, of none where the predicate fails. */
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   if (cond == WVK_CONDITION_GPU && !indirect_on_host(cmd)) {
      const struct wvk_bound_buffer src = b;
      if (!conditional_args(cmd, NULL, &src, max_draws, size / 4, stride, &b))
         return;
      stride = size;
   }
   cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU && b.mem;
   draw_indirect_count_bound(cmd, indexed, b, c, max_draws, stride);
   cmd->replay->cond_gpu = false;
}

static void
draw_indirect_count_bound(struct wvk_cmd_buffer *cmd, bool indexed, struct wvk_bound_buffer b,
                          struct wvk_bound_buffer c, uint32_t max_draws, uint32_t stride)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   select_draw_shaders(cmd);
   const uint32_t size = indexed ? sizeof(VkDrawIndexedIndirectCommand) : sizeof(VkDrawIndirectCommand);
   const uint64_t bytes = (uint64_t)(max_draws - 1) * stride + size;
   /* A fan's are drawn from the host's copy, the count too; and those
    * whose vertex shader writes storage, which is done for each draw
    * (run_vertex_writes), and for no more than there are. */
   const struct wvk_bound_state *st = replay->state;
   if (cmd->vk.dynamic_graphics_state.ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN ||
       (st->bound_vs && st->bound_vs->writes_storage) || st->bound_tes || draw_gs(cmd)) {
      uint32_t count = max_draws;
      if (c.mem && c.mem->host)
         memcpy(&count, (const uint8_t *)c.mem->host + c.offset, 4);
      draw_indirect_bound(cmd, indexed, b, MIN2(count, max_draws), stride);
      return;
   }
   if (wvk_replay_stats(replay)) {
      uint32_t count = max_draws;
      if (c.mem && c.mem->host)
         memcpy(&count, (const uint8_t *)c.mem->host + c.offset, 4);
      count_indirect_draws(cmd, indexed, b, MIN2(count, max_draws), stride);
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

/* A compute pipeline of the shader, for the sets bound at `bind_point`:
 * the graphics ones for a vertex shader run as compute (run_vertices). */
static struct wvk_compute_pipeline *
get_compute_pipeline(struct wvk_device *dev, const struct wvk_bound_state *st, struct wvk_shader *cs,
                     uint32_t bind_point)
{
   const uint64_t key[2] = { cs->id, binding_variant(dev, st, bind_point, &cs, 1) };
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

/* What a dispatch samples of a texture it also writes as storage.
 *
 * WebGPU has the two in no usage scope together, per level and layer,
 * where Vulkan asks only that no texel be both read and written. A
 * mip chain built a level at a time does exactly that: EEVEE's samples
 * a view of the whole chain and writes the next level of it.
 *
 * Where every level written is above the sampled view's base, the view
 * stops short of them. It keeps its base, so a lod names the same level
 * as before, and the levels it loses are ones the dispatch writes, which
 * it could not have read in Vulkan either. Otherwise it samples a copy
 * made before the dispatch. */
static void
swap_sampled_storage(struct wvk_cmd_buffer *cmd, const struct wvk_layout *layout, uint32_t bind_point)
{
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   for (uint32_t g = 0; g < layout->group_count; ++g) {
      const struct wvk_bgl *bgl = layout->groups[g];
      const struct wvk_descriptor_set *set = st->sets[bind_point][g].set;
      for (uint32_t i = 0; set && i < bgl->count; ++i) {
         const struct wvk_shader_binding *b = &bgl->entries[i].b;
         if (b->kind != WVK_BINDING_TEXTURE || (g == WVK_DRIVER_GROUP && b->binding >= WVK_DRIVER_SLOT_BASE))
            continue;
         bool sampler;
         uint32_t dynamic_index;
         const struct wvk_descriptor *d = slot_descriptor(set, b->binding, &sampler, &dynamic_index);
         if (!d || !d->view || !d->view_image || (d->view_image->vk.aspects & ~VK_IMAGE_ASPECT_COLOR_BIT))
            continue;
         uint32_t lowest = UINT32_MAX;
         for (uint32_t og = 0; og < layout->group_count; ++og) {
            const struct wvk_bgl *obgl = layout->groups[og];
            const struct wvk_descriptor_set *oset = st->sets[bind_point][og].set;
            for (uint32_t j = 0; oset && j < obgl->count; ++j) {
               const struct wvk_shader_binding *o = &obgl->entries[j].b;
               if (o->kind != WVK_BINDING_STORAGE_TEXTURE || o->access == WGPUStorageTextureAccess_ReadOnly ||
                   (og == WVK_DRIVER_GROUP && o->binding >= WVK_DRIVER_SLOT_BASE))
                  continue;
               const struct wvk_descriptor *od = slot_descriptor(oset, o->binding, &sampler, &dynamic_index);
               if (od && od->view && storage_views_alias(d, od))
                  lowest = MIN2(lowest, od->view_level);
            }
         }
         if (lowest == UINT32_MAX)
            continue;
         WGPUTextureViewDescriptor vd = d->view_desc;
         WGPUTextureView v = NULL;
         if (lowest > d->view_level) {
            vd.mipLevelCount = lowest - d->view_level;
            uint32_t first_level;
            v = wgpuTextureCreateView(wvk_image_texture(d->view_image, d->view_texture, &first_level), &vd);
         } else {
            wvk_replay_end_passes(replay);
            WGPUTexture copy = wvk_image_snapshot(dev->wgpu, d->view_image, d->view_texture, replay->encoder);
            if (copy) {
               v = wgpuTextureCreateView(copy, &vd);
               wgpuTextureRelease(copy);
            }
         }
         if (!v)
            continue;
         util_dynarray_append(&replay->views, v);
         const struct wvk_texture_swap swap = { g, b->binding, v };
         util_dynarray_append(&replay->texture_swaps, swap);
      }
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
   /* Or its variant for the formats of storage images it has none for. */
   struct wvk_bound_views images;
   bound_views(st, WVK_BIND_COMPUTE, st->bound_cs, &images);
   st->cs = wvk_shader_dispatch_variant(dev, st->bound_cs, &images);
   if (!st->cs) {
      mesa_logw_once("wvk: a dispatch without a compute shader");
      return NULL;
   }
   if (replay->pass) {
      mesa_logw_once("wvk: a dispatch inside a render pass");
      return NULL;
   }
   struct wvk_compute_pipeline *p = get_compute_pipeline(dev, st, st->cs, WVK_BIND_COMPUTE);
   if (!p)
      return NULL;
   sync_image_textures(cmd, p->layout, WVK_BIND_COMPUTE);
   sync_atomic_buffers(cmd, p->layout, WVK_BIND_COMPUTE);
   /* A dispatch is a usage scope: what it reads of buffers it writes,
    * copies. */
   struct buffer_use uses[64];
   const uint32_t n = gather_buffer_uses(dev, replay, p->layout, WVK_BIND_COMPUTE, uses, ARRAY_SIZE(uses));
   swap_conflicting_reads(dev, replay, uses, n);
   swap_sampled_storage(cmd, p->layout, WVK_BIND_COMPUTE);
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
   util_dynarray_clear(&replay->texture_swaps);
   return bound ? replay->compute : NULL;
}

/* A dispatch's compute shader invocations, for the pipeline statistics
 * query. */
static void
count_dispatch(struct wvk_cmd_buffer *cmd, uint32_t x, uint32_t y, uint32_t z)
{
   const struct wvk_shader *cs = cmd->replay->state->cs;
   if (!cs || !wvk_replay_stats(cmd->replay))
      return;
   wvk_replay_count_stat(cmd->replay, VK_QUERY_PIPELINE_STATISTIC_COMPUTE_SHADER_INVOCATIONS_BIT,
                         (uint64_t)x * y * z * cs->workgroup_size[0] * cs->workgroup_size[1] * cs->workgroup_size[2]);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDispatchBase(VkCommandBuffer commandBuffer, uint32_t baseGroupX, uint32_t baseGroupY,
                    uint32_t baseGroupZ, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   if (!groupCountX || !groupCountY || !groupCountZ)
      return;
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   /* Under a predicate the GPU has: an indirect dispatch of none where
    * it fails. */
   struct wvk_bound_buffer groups;
   if (cond == WVK_CONDITION_GPU &&
       !conditional_args(cmd, (const uint32_t[3]){ groupCountX, groupCountY, groupCountZ }, NULL, 1, 3, 12, &groups))
      return;
   set_base_group(cmd->replay, baseGroupX, baseGroupY, baseGroupZ);
   WGPUComputePassEncoder compute = flush_dispatch(cmd, cond == WVK_CONDITION_GPU ? &groups : NULL);
   if (!compute)
      return;
   if (cond == WVK_CONDITION_GPU) {
      wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute, groups.buffer, groups.offset);
      return;
   }
   wgpuComputePassEncoderDispatchWorkgroups(compute, groupCountX, groupCountY, groupCountZ);
   count_dispatch(cmd, groupCountX, groupCountY, groupCountZ);
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
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   if (cond == WVK_CONDITION_GPU) {
      const struct wvk_bound_buffer src = b;
      if (!conditional_args(cmd, NULL, &src, 1, 3, 12, &b))
         return;
   }
   set_base_group(cmd->replay, 0, 0, 0);
   WGPUComputePassEncoder compute = flush_dispatch(cmd, &b);
   if (!compute)
      return;
   if (b.mem)
      wvk_replay_read(cmd->replay, b.mem, b.offset, b.size);
   wgpuComputePassEncoderDispatchWorkgroupsIndirect(compute, b.buffer, b.offset);
   if (wvk_replay_stats(cmd->replay) && cond == WVK_CONDITION_PASS) {
      VK_FROM_HANDLE(wvk_buffer, src, buffer);
      if (src->mem && src->mem->host) {
         uint32_t groups[3];
         memcpy(groups, (const uint8_t *)src->mem->host + src->offset + offset, sizeof(groups));
         count_dispatch(cmd, groups[0], groups[1], groups[2]);
      } else {
         mesa_logw_once("wvk: an indirect dispatch in memory the host has no copy of, which pipeline statistics "
                        "miss");
      }
   }
}

/* ---------------------------------------------------------------------- */
/* Transform feedback (VK_EXT_transform_feedback): captured by the pass the
 * geometry shader's lists go to (capture_xfb), which every draw has while
 * it is on -- one of the driver's where the pipeline has none. */

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBindTransformFeedbackBuffersEXT(VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount,
                                       const VkBuffer *pBuffers, const VkDeviceSize *pOffsets,
                                       const VkDeviceSize *pSizes)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_replay *replay = cmd->replay;
   for (uint32_t i = 0; i < bindingCount && firstBinding + i < 4; ++i) {
      const struct wvk_bound_buffer b =
         bound_buffer(cmd, pBuffers[i], pOffsets[i], pSizes ? pSizes[i] : VK_WHOLE_SIZE);
      replay->xfb_bindings[firstBinding + i].mem = b.mem;
      replay->xfb_bindings[firstBinding + i].buffer = b.buffer;
      replay->xfb_bindings[firstBinding + i].offset = b.offset;
      replay->xfb_bindings[firstBinding + i].size = b.size;
   }
}

/* Where each buffer's writes begin and end: a counter buffer's, bytes from
 * the binding's offset, into and out of the capture's state, copied
 * outside any pass. */
static void
copy_xfb_counters(struct wvk_cmd_buffer *cmd, uint32_t first, uint32_t count, const VkBuffer *buffers,
                  const VkDeviceSize *offsets, bool into_state)
{
   struct wvk_replay *replay = cmd->replay;
   const bool in_pass = replay->pass != NULL;
   bool ended = false;
   for (uint32_t i = 0; buffers && i < count && first + i < 4; ++i) {
      const struct wvk_bound_buffer c = bound_buffer(cmd, buffers[i], offsets ? offsets[i] : 0, 4);
      if (!c.buffer)
         continue;
      if (!ended) {
         wvk_replay_end_passes(replay);
         ended = true;
      }
      if (into_state) {
         wvk_replay_read(replay, c.mem, c.offset, 4);
         wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, c.buffer, c.offset, replay->xfb_state,
                                              (first + i) * 4, 4);
      } else {
         wgpuCommandEncoderCopyBufferToBuffer(replay->encoder, replay->xfb_state, (first + i) * 4, c.buffer,
                                              c.offset, 4);
         wvk_replay_write(replay, c.mem, c.offset, 4);
      }
   }
   if (ended && in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdBeginTransformFeedbackEXT(VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
                                 uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
                                 const VkDeviceSize *pCounterBufferOffsets)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   static const uint32_t zero[8] = { 0 };
   replay->xfb_state = wvk_replay_temp_buffer(dev, replay, sizeof(zero),
                                              WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc |
                                                 WGPUBufferUsage_CopyDst,
                                              zero);
   replay->xfb_active = replay->xfb_state != NULL;
   if (replay->xfb_state)
      copy_xfb_counters(cmd, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets, true);
}

VKAPI_ATTR void VKAPI_CALL
wvk_CmdEndTransformFeedbackEXT(VkCommandBuffer commandBuffer, uint32_t firstCounterBuffer,
                               uint32_t counterBufferCount, const VkBuffer *pCounterBuffers,
                               const VkDeviceSize *pCounterBufferOffsets)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_replay *replay = cmd->replay;
   if (replay->xfb_state)
      copy_xfb_counters(cmd, firstCounterBuffer, counterBufferCount, pCounterBuffers, pCounterBufferOffsets, false);
   replay->xfb_active = false;
}

/* A draw of as many vertices as a counter says were captured: its count
 * made into an indirect command on the GPU. */
static const char xfb_count_wgsl[] =
   "struct P { counter_offset: u32, stride: u32, instances: u32, first_instance: u32, word: u32 }\n"
   "@group(0) @binding(0) var<uniform> p: P;\n"
   "@group(0) @binding(1) var<storage, read> counter: array<u32>;\n"
   "@group(0) @binding(2) var<storage, read_write> command: array<u32>;\n"
   "@compute @workgroup_size(1)\n"
   "fn main() {\n"
   "  let bytes = counter[p.word];\n"
   "  command[0] = select(0u, (bytes - p.counter_offset) / max(p.stride, 1u), bytes > p.counter_offset);\n"
   "  command[1] = p.instances;\n"
   "  command[2] = 0u;\n"
   "  command[3] = p.first_instance;\n"
   "}\n";

VKAPI_ATTR void VKAPI_CALL
wvk_CmdDrawIndirectByteCountEXT(VkCommandBuffer commandBuffer, uint32_t instanceCount, uint32_t firstInstance,
                                VkBuffer counterBuffer, VkDeviceSize counterBufferOffset, uint32_t counterOffset,
                                uint32_t vertexStride)
{
   VK_FROM_HANDLE(wvk_cmd_buffer, cmd, commandBuffer);
   struct wvk_device *dev = wvk_cmd_buffer_device(cmd);
   struct wvk_replay *replay = cmd->replay;
   struct wvk_bound_state *st = replay->state;
   if (!instanceCount)
      return;
   const enum wvk_condition cond = wvk_cmd_condition(cmd);
   if (cond == WVK_CONDITION_SKIP)
      return;
   const struct wvk_bound_buffer c = bound_buffer(cmd, counterBuffer, counterBufferOffset & ~3ull, 4);
   /* What needs the count on the host has the host's copy of it: which a
    * counter written in this submit is not yet. */
   if (st->bound_tes || draw_gs(cmd) || draws_points(cmd) || draws_lines(cmd) ||
       (st->bound_vs && st->bound_vs->writes_storage) ||
       cmd->vk.dynamic_graphics_state.ia.primitive_topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN) {
      uint32_t bytes = 0;
      if (c.mem && c.mem->host)
         memcpy(&bytes, (const uint8_t *)c.mem->host + c.offset, 4);
      else
         mesa_logw_once("wvk: a draw by a counter's byte count in memory the host has no copy of");
      const uint32_t vertices = bytes > counterOffset ? (bytes - counterOffset) / MAX2(vertexStride, 1) : 0;
      cmd->replay->cond_gpu = cond == WVK_CONDITION_GPU;
      draw_arrays(cmd, vertices, instanceCount, 0, firstInstance, 0);
      cmd->replay->cond_gpu = false;
      return;
   }
   WGPUComputePipeline pipeline = xfb_pipeline(dev, &dev->xfb_count_pipeline, xfb_count_wgsl, "wvk counter draw");
   /* The counter, of a binding at an offset storage buffers can have. */
   const uint64_t binding = c.offset & ~255ull;
   const uint64_t binding_size = c.mem ? MIN2(256, c.mem->buffer_size - binding) & ~3ull : 256;
   const uint32_t params[8] = { counterOffset, vertexStride, instanceCount, firstInstance,
                                (uint32_t)(c.offset - binding) / 4 };
   WGPUBuffer params_buffer = wvk_replay_temp_buffer(dev, replay, sizeof(params), WGPUBufferUsage_Uniform, params);
   WGPUBuffer command = wvk_replay_temp_buffer(dev, replay, 16, WGPUBufferUsage_Storage | WGPUBufferUsage_Indirect,
                                               NULL);
   if (!pipeline || !params_buffer || !command || !c.buffer)
      return;
   wvk_replay_read(replay, c.mem, c.offset, 4);
   const WGPUBindGroupEntry entries[] = {
      { .binding = 0, .buffer = params_buffer, .offset = 0, .size = sizeof(params) },
      { .binding = 1, .buffer = c.buffer, .offset = binding, .size = binding_size },
      { .binding = 2, .buffer = command, .offset = 0, .size = 16 },
   };
   WGPUBindGroupDescriptor bg = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
   bg.layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
   bg.entryCount = ARRAY_SIZE(entries);
   bg.entries = entries;
   WGPUBindGroup group = wgpuDeviceCreateBindGroup(dev->wgpu, &bg);
   wgpuBindGroupLayoutRelease(bg.layout);
   const bool in_pass = replay->pass != NULL;
   wvk_replay_end_passes(replay);
   WGPUComputePassDescriptor pd = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
   WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(replay->encoder, &pd);
   wgpuComputePassEncoderSetPipeline(pass, pipeline);
   wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, NULL);
   wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
   wgpuComputePassEncoderEnd(pass);
   wgpuComputePassEncoderRelease(pass);
   wgpuBindGroupRelease(group);
   if (in_pass)
      wvk_replay_restart_pass(cmd, replay->pass_layer);
   /* The command, of no vertices where the predicate fails. */
   if (cond == WVK_CONDITION_GPU) {
      const struct wvk_bound_buffer src = { NULL, command, 0, 16, ++dev->temp_serial | (1ull << 62) };
      struct wvk_bound_buffer b;
      if (!conditional_args(cmd, NULL, &src, 1, 4, 16, &b))
         return;
      command = b.buffer;
   }
   select_draw_shaders(cmd);
   foreach_viewport(cmd) {
      set_draw_parameters(st, NULL, 0, false, 0, firstInstance, 0);
      if (!flush_draw(cmd, false))
         continue;
      wgpuRenderPassEncoderDrawIndirect(replay->pass, command, 0);
   }
}
