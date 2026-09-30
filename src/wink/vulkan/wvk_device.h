/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_DEVICE_H
#define WVK_DEVICE_H

#include "wvk_private.h"
#include "wvk_queue.h"
#include "wvk_descriptor_set_layout.h"

#include "vk_command_pool.h"
#include "vk_device.h"
#include "vk_meta.h"

#include "util/hash_table.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"

struct wvk_bgl;
struct wvk_cmd_buffer;
struct wvk_physical_device;

struct wvk_placeholder {
   uint64_t key;
   WGPUTexture texture;
   WGPUTextureView view;
};

struct wvk_device {
   struct vk_device vk;

   /* What applications reach: every vkCmd* records into the command
    * buffer's queue. vk.dispatch_table is the driver's own entrypoints,
    * which replay calls. */
   struct vk_device_dispatch_table exposed_dispatch_table;

   WGPUDevice wgpu;
   struct wvk_queue queue;
   /* Why WebGPU lost the device, for VK_EXT_device_fault. */
   char lost_message[VK_MAX_DESCRIPTION_SIZE];
   /* The compute family's, when one was asked for (wvk_queue.h). */
   struct vk_queue compute_queue;

   /* What submits replay into; submits to the one queue are serialized. */
   struct vk_command_pool replay_pool;
   struct wvk_cmd_buffer *replay_cmd;

   /* Blits and vkCmdClearAttachments. */
   struct vk_meta_device meta;

   /* VK_EXT_host_image_copy's copies, recorded one at a time into a
    * command buffer of this pool (wvk_host_copy.c). */
   simple_mtx_t host_copy_lock;
   struct vk_command_pool host_copy_pool;
   struct wvk_cmd_buffer *host_copy_cmd;

   /* WebGPU pipelines and layouts, made at draws and kept. */
   simple_mtx_t cache_lock;
   struct hash_table *bind_group_layouts;   /* struct wvk_bgl */
   struct hash_table *pipeline_layouts;     /* struct wvk_layout */
   struct hash_table *render_pipelines;     /* struct wvk_render_pipeline */
   struct hash_table *compute_pipelines;    /* struct wvk_compute_pipeline */
   uint64_t next_shader_id;
   uint64_t next_memory_serial;

   /* An empty group, for the gaps before the driver's. */
   struct wvk_bgl *empty_bgl;
   WGPUBindGroup empty_bind_group;
   /* A fragment shader with no outputs, for pipelines without one that
    * still have colour attachments to match. */
   WGPUShaderModule empty_fragment;
   /* The fragment shader of a draw without one whose depth bounds test is
    * on: the test alone (wvk_cmd_draw.c). */
   struct wvk_shader *depth_bounds_fs;
   /* Where a submit's push constants go; grown when a submit needs more. */
   WGPUBuffer push_buffer;
   uint64_t push_buffer_size;
   uint64_t push_generation;
   /* Bind groups of only the driver's bindings, by layout. */
   struct {
      struct wvk_bgl *bgl;
      struct wvk_driver_key driver;
      WGPUBindGroup group;
   } driver_groups[8];
   uint32_t driver_group_next;
   /* The shaders of copies WebGPU does not do: query results, bytes
    * between buffer layouts, and depth out of a texture. */
   WGPUComputePipeline copy_query_pipeline;
   WGPUComputePipeline add_query_pipeline;
   /* Pipeline statistics: counts added on the GPU, and a query's results
    * made of the host's and those at its end (wvk_query.c). */
   WGPUComputePipeline add_stats_pipeline;
   WGPUComputePipeline finish_stats_pipeline;
   WGPUComputePipeline fan_pipeline;
   WGPUComputePipeline uint8_index_pipeline;
   /* Transform feedback's: capture, and the vertex count of what was. */
   WGPUComputePipeline xfb_pipeline;
   WGPUComputePipeline xfb_count_pipeline;
   WGPUComputePipeline draw_count_pipeline;
   /* Conditional rendering's: a command's arguments, zero where its
    * predicate fails (wvk_cmd_draw.c). */
   WGPUComputePipeline cond_pipeline;
   /* The tessellator's passes: count, scan, generate (wvk_tessellator.c),
    * and their bind group layout. */
   WGPUComputePipeline tess_pipelines[3];
   WGPUBindGroupLayout tess_bgl;
   /* A mesh draw's passes: its setup, and the task workgroups' launches'
    * scan (wvk_cmd_draw.c's draw_mesh), and their bind group layout. */
   WGPUComputePipeline mesh_pipelines[2];
   WGPUBindGroupLayout mesh_bgl;
   /* The fragment shader of a draw that has none, for its discard
    * rectangle test (wvk_cmd_draw.c). */
   struct wvk_shader *discard_fs;
   /* Serials for the driver's own indirect buffers, above any memory's. */
   uint64_t temp_serial;
   WGPUComputePipeline repack_pipeline;
   /* 16-bit packed texels into and out of RGBA8 textures: 2D arrays and
    * 3D, each way (wvk_cmd_copy.c). */
   WGPUComputePipeline packed16_pipelines[4];
   WGPUComputePipeline depth_out_pipeline;
   /* And into depth: a pass writing frag_depth, for each depth format. */
   struct {
      WGPUTextureFormat format;
      WGPURenderPipeline pipeline;
   } depth_in_pipelines[4];
   /* Images made with aliases or a shadow (wvk_image.h), whose textures
    * draws bring up to date: none, and they need not look. */
   uint32_t tracked_images;
   /* Compressed texels into a shadow (wvk_image.h): for each format of
    * shadow, from a 2D array and from a 3D texture. */
   WGPURenderPipeline shadow_pipelines[3][2];
   /* Copies of multisampled colour textures, a pass writing each sample:
    * by format and sample count. */
   struct {
      WGPUTextureFormat format;
      uint32_t samples;
      WGPURenderPipeline pipeline;
   } ms_copy_pipelines[8];
   /* Depth into R32Float and stencil into R8Uint, for copies of them a
    * pass reads. */
   WGPURenderPipeline ds_copy_pipelines[2];
   /* What a binding with no descriptor written shows: 1x1 textures of
    * each kind a bind group layout can want (descriptorBindingPartiallyBound),
    * as many kinds as are asked for. */
   struct util_dynarray placeholders; /* struct wvk_placeholder */
   /* What a pass without attachments renders to. */
   WGPUTexture dummy_attachment;
   VkExtent2D dummy_extent;
   uint32_t dummy_samples;
   /* Bound where a descriptor was never written. */
   WGPUBuffer null_buffer;
   /* And of a writable storage binding: WebGPU lets no buffer be both
    * that and read-only in a dispatch, as the null buffer may be. The
    * shader writes none of it (robustBufferAccess2's bounds). */
   WGPUBuffer null_storage;
   WGPUSampler null_sampler;
};

/* The caches: made with the device, gone with it. */
VkResult wvk_device_init_pipeline_caches(struct wvk_device *dev);
void wvk_device_finish_pipeline_caches(struct wvk_device *dev);
VkResult wvk_device_init_meta(struct wvk_device *dev);
void wvk_device_finish_meta(struct wvk_device *dev);
VkResult wvk_device_init_host_copy(struct wvk_device *dev);
void wvk_device_finish_host_copy(struct wvk_device *dev);

VK_DEFINE_HANDLE_CASTS(wvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct wvk_physical_device *
wvk_device_physical(const struct wvk_device *dev)
{
   return (struct wvk_physical_device *)dev->vk.physical;
}

#endif
