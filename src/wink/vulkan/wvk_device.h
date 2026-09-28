/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_DEVICE_H
#define WVK_DEVICE_H

#include "wvk_private.h"
#include "wvk_queue.h"

#include "vk_device.h"

struct wvk_physical_device;

struct wvk_device {
   struct vk_device vk;

   /* What applications reach: every vkCmd* records into the command
    * buffer's queue. vk.dispatch_table is the driver's own entrypoints,
    * which replay calls. */
   struct vk_device_dispatch_table exposed_dispatch_table;

   WGPUDevice wgpu;
   struct wvk_queue queue;
};

VK_DEFINE_HANDLE_CASTS(wvk_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE)

static inline struct wvk_physical_device *
wvk_device_physical(const struct wvk_device *dev)
{
   return (struct wvk_physical_device *)dev->vk.physical;
}

#endif
