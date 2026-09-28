/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_device.h"

#include "wvk_cmd_buffer.h"
#include "wvk_entrypoints.h"
#include "wvk_instance.h"
#include "wvk_physical_device.h"
#include "wvk_platform.h"

#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_common_entrypoints.h"

#include <stdio.h>
#include <stdlib.h>

static void
on_uncaptured_error(WGPUDevice const *device, WGPUErrorType type, WGPUStringView message,
                    void *userdata1, void *userdata2)
{
   (void)device; (void)userdata1; (void)userdata2;
   /* A WebGPU validation error is a driver bug: Vulkan's own validation
    * is the application's, and happens before anything reaches WebGPU. */
   mesa_loge("wvk: WebGPU error (type %u): %.*s", (unsigned)type, wvk_sv_len(message), message.data);
   if (getenv("WINK_VK_ABORT_ON_ERROR"))
      abort();
}

static void
on_device_lost(WGPUDevice const *device, WGPUDeviceLostReason reason, WGPUStringView message,
               void *userdata1, void *userdata2)
{
   struct wvk_device *dev = userdata1;
   (void)device; (void)userdata2;
   /* Destroying the device reports it lost too; that is not news. */
   if (reason == WGPUDeviceLostReason_Destroyed || reason == WGPUDeviceLostReason_CallbackCancelled)
      return;
   vk_device_set_lost(&dev->vk, "WebGPU device lost (%u): %.*s", (unsigned)reason,
                      wvk_sv_len(message), message.data);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateDevice(VkPhysicalDevice physicalDevice, const VkDeviceCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkDevice *pDevice)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);
   struct wvk_instance *instance = wvk_physical_device_instance(pdev);
   VkResult result;

   struct wvk_device *dev = vk_zalloc2(&pdev->vk.instance->alloc, pAllocator, sizeof(*dev), 8,
                                       VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!dev)
      return vk_error(pdev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* The driver's own table: what replay, and the runtime's helpers, call. */
   struct vk_device_dispatch_table dispatch_table;
   vk_device_dispatch_table_from_entrypoints(&dispatch_table, &wvk_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&dispatch_table, &vk_common_device_entrypoints, false);

   result = vk_device_init(&dev->vk, &pdev->vk, &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(&dev->vk.alloc, dev);
      return result;
   }

   /* What the application reaches: recording for every vkCmd*. */
   vk_device_dispatch_table_from_entrypoints(&dev->exposed_dispatch_table,
                                             &vk_cmd_enqueue_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&dev->exposed_dispatch_table,
                                             &wvk_device_entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(&dev->exposed_dispatch_table,
                                             &vk_common_device_entrypoints, false);

   dev->vk.command_buffer_ops = &wvk_cmd_buffer_ops;
   /* Secondary command buffers replay through the same table. */
   dev->vk.command_dispatch_table = &dev->vk.dispatch_table;

   /* Everything the adapter offers: its limits, and the features the
    * driver has a use for. */
   WGPUDeviceDescriptor desc = WGPU_DEVICE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("wink Vulkan");
   desc.requiredFeatureCount = pdev->feature_count;
   desc.requiredFeatures = pdev->features;
   desc.requiredLimits = &pdev->limits;
   desc.uncapturedErrorCallbackInfo.callback = on_uncaptured_error;
   desc.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
   desc.deviceLostCallbackInfo.callback = on_device_lost;
   desc.deviceLostCallbackInfo.userdata1 = dev;
   dev->wgpu = wvk_platform_request_device(instance->wgpu, pdev->adapter, &desc);
   if (!dev->wgpu) {
      result = vk_errorf(pdev, VK_ERROR_INITIALIZATION_FAILED, "no WebGPU device");
      goto fail_init;
   }

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = 0,
   };
   result = vk_command_pool_init(&dev->vk, &dev->replay_pool, &pool_info, NULL);
   if (result != VK_SUCCESS)
      goto fail_device;
   struct vk_command_buffer *replay;
   result = wvk_cmd_buffer_ops.create(&dev->replay_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &replay);
   if (result != VK_SUCCESS)
      goto fail_pool;
   dev->replay_cmd = container_of(replay, struct wvk_cmd_buffer, vk);

   /* One queue family, with one queue. */
   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i) {
      const VkDeviceQueueCreateInfo *qci = &pCreateInfo->pQueueCreateInfos[i];
      assert(qci->queueFamilyIndex == 0 && qci->queueCount == 1);
      result = wvk_queue_init(dev, &dev->queue, qci, 0);
      if (result != VK_SUCCESS)
         goto fail_replay;
   }

   *pDevice = wvk_device_to_handle(dev);
   return VK_SUCCESS;

fail_replay:
   wvk_cmd_buffer_ops.destroy(&dev->replay_cmd->vk);
fail_pool:
   vk_command_pool_finish(&dev->replay_pool);
fail_device:
   wgpuDeviceRelease(dev->wgpu);
fail_init:
   vk_device_finish(&dev->vk);
   vk_free(&dev->vk.alloc, dev);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyDevice(VkDevice _device, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   if (!dev)
      return;

   if (dev->queue.vk.base.device)
      wvk_queue_finish(dev, &dev->queue);
   wvk_cmd_buffer_ops.destroy(&dev->replay_cmd->vk);
   vk_command_pool_finish(&dev->replay_pool);
   wgpuDeviceDestroy(dev->wgpu);
   wgpuDeviceRelease(dev->wgpu);
   vk_device_finish(&dev->vk);
   vk_free(&dev->vk.alloc, dev);
}

/* From the exposed table, so what an application is handed records. */
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
wvk_GetDeviceProcAddr(VkDevice _device, const char *pName)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   if (!dev || !pName)
      return NULL;
   const struct vk_instance *instance = dev->vk.physical->instance;
   return vk_device_dispatch_table_get_if_supported(&dev->exposed_dispatch_table, pName,
                                                    instance->app_info.api_version,
                                                    &instance->enabled_extensions,
                                                    &dev->vk.enabled_extensions);
}
