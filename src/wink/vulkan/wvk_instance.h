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

   /* Made when physical devices are first enumerated. */
   WGPUInstance wgpu;

   /* Identifies this build: driverUUID, and the pipeline cache's. */
   uint8_t driver_uuid[VK_UUID_SIZE];
};

VK_DEFINE_HANDLE_CASTS(wvk_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)

#endif
