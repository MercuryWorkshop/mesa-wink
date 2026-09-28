/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_DEVICE_H
#define WVK_DEVICE_H

#include "wvk_private.h"
#include "wvk_queue.h"

#include "vk_command_pool.h"
#include "vk_device.h"
#include "vk_meta.h"

#include "util/hash_table.h"
#include "util/simple_mtx.h"

struct wvk_bgl;
struct wvk_cmd_buffer;
struct wvk_physical_device;

struct wvk_device {
   struct vk_device vk;

   /* What applications reach: every vkCmd* records into the command
    * buffer's queue. vk.dispatch_table is the driver's own entrypoints,
    * which replay calls. */
   struct vk_device_dispatch_table exposed_dispatch_table;

   WGPUDevice wgpu;
   struct wvk_queue queue;

   /* What submits replay into; submits to the one queue are serialized. */
   struct vk_command_pool replay_pool;
   struct wvk_cmd_buffer *replay_cmd;

   /* Blits and vkCmdClearAttachments. */
   struct vk_meta_device meta;

   /* WebGPU pipelines and layouts, made at draws and kept. */
   simple_mtx_t cache_lock;
   struct hash_table *bind_group_layouts;   /* struct wvk_bgl */
   struct hash_table *pipeline_layouts;     /* struct wvk_layout */
   struct hash_table *render_pipelines;     /* struct wvk_render_pipeline */
   struct hash_table *compute_pipelines;    /* struct wvk_compute_pipeline */
   uint64_t next_shader_id;

   /* An empty group, for the gaps before the driver's. */
   struct wvk_bgl *empty_bgl;
   WGPUBindGroup empty_bind_group;
   /* A fragment shader with no outputs, for pipelines without one that
    * still have colour attachments to match. */
   WGPUShaderModule empty_fragment;
   /* Where a submit's push constants go; grown when a submit needs more. */
   WGPUBuffer push_buffer;
   uint64_t push_buffer_size;
   uint64_t push_generation;
   /* Bind groups of only the driver's bindings, by layout. */
   struct {
      struct wvk_bgl *bgl;
      uint64_t extra;
      WGPUBindGroup group;
   } driver_groups[4];
   uint32_t driver_group_next;
   /* Bound where a descriptor was never written. */
   WGPUBuffer null_buffer;
   WGPUSampler null_sampler;
};

/* The caches: made with the device, gone with it. */
VkResult wvk_device_init_pipeline_caches(struct wvk_device *dev);
void wvk_device_finish_pipeline_caches(struct wvk_device *dev);
VkResult wvk_device_init_meta(struct wvk_device *dev);
void wvk_device_finish_meta(struct wvk_device *dev);

VK_DEFINE_HANDLE_CASTS(wvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct wvk_physical_device *
wvk_device_physical(const struct wvk_device *dev)
{
   return (struct wvk_physical_device *)dev->vk.physical;
}

#endif
