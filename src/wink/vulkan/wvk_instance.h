/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_INSTANCE_H
#define WVK_INSTANCE_H

#include "wvk_private.h"

#include "vk_instance.h"

struct wvk_instance {
   struct vk_instance vk;

   /* Made when first needed: to enumerate physical devices, or to make
    * a surface. See wvk_instance_wgpu. */
   WGPUInstance wgpu;

   /* Identifies this build: driverUUID, and the pipeline cache's. */
   uint8_t driver_uuid[VK_UUID_SIZE];
};

VK_DEFINE_HANDLE_CASTS(wvk_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)

/* The WebGPU instance, made if need be; NULL if there is no WebGPU. */
WGPUInstance wvk_instance_wgpu(struct wvk_instance *instance);

#endif
