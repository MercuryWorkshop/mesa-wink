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
#include "wvk_shader.h"

#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_pipeline.h"
#include "util/mesa-blake3.h"
#include "vk_common_entrypoints.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#ifndef __EMSCRIPTEN__
/* Dawn's own messages: of its toggles' dump_shaders, the MSL. */
static void
on_dawn_log(WGPULoggingType type, WGPUStringView message, void *userdata1, void *userdata2)
{
   (void)userdata1; (void)userdata2;
   mesa_logi("wvk: Dawn (%u): %.*s", (unsigned)type, wvk_sv_len(message), message.data);
}
#endif

static void
on_device_lost(WGPUDevice const *device, WGPUDeviceLostReason reason, WGPUStringView message,
               void *userdata1, void *userdata2)
{
   struct wvk_device *dev = userdata1;
   (void)device; (void)userdata2;
   /* Destroying the device reports it lost too; that is not news. */
   if (reason == WGPUDeviceLostReason_Destroyed || reason == WGPUDeviceLostReason_CallbackCancelled)
      return;
   snprintf(dev->lost_message, sizeof(dev->lost_message), "WebGPU device lost (%u): %.*s", (unsigned)reason,
            wvk_sv_len(message), message.data);
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
   dev->vk.shader_ops = &wvk_device_shader_ops;
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
#ifndef __EMSCRIPTEN__
   /* WINK_VK_DAWN_TOGGLES: Dawn toggles to enable, comma-separated, for
    * debugging -- dump_shaders and the like. Dawn's own: emdawnwebgpu has
    * neither toggles nor its logging. */
   const char *toggles_env = getenv("WINK_VK_DAWN_TOGGLES");
   char *toggles_str = toggles_env ? strdup(toggles_env) : NULL;
   const char *toggles[16];
   WGPUDawnTogglesDescriptor toggles_desc = WGPU_DAWN_TOGGLES_DESCRIPTOR_INIT;
   if (toggles_str) {
      char *save = NULL;
      for (char *t = strtok_r(toggles_str, ",", &save); t && toggles_desc.enabledToggleCount < ARRAY_SIZE(toggles);
           t = strtok_r(NULL, ",", &save))
         toggles[toggles_desc.enabledToggleCount++] = t;
      toggles_desc.enabledToggles = toggles;
      desc.nextInChain = &toggles_desc.chain;
   }
   dev->wgpu = wvk_platform_request_device(instance->wgpu, pdev->adapter, &desc);
   free(toggles_str);
   if (dev->wgpu && toggles_env) {
      WGPULoggingCallbackInfo log = WGPU_LOGGING_CALLBACK_INFO_INIT;
      log.callback = on_dawn_log;
      wgpuDeviceSetLoggingCallback(dev->wgpu, log);
   }
#else
   dev->wgpu = wvk_platform_request_device(instance->wgpu, pdev->adapter, &desc);
#endif
   if (!dev->wgpu) {
      result = vk_errorf(pdev, VK_ERROR_INITIALIZATION_FAILED, "no WebGPU device");
      goto fail_init;
   }

   result = wvk_device_init_pipeline_caches(dev);
   if (result != VK_SUCCESS)
      goto fail_device;

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = 0,
   };
   result = vk_command_pool_init(&dev->vk, &dev->replay_pool, &pool_info, NULL);
   if (result != VK_SUCCESS)
      goto fail_caches;
   struct vk_command_buffer *replay;
   result = wvk_cmd_buffer_ops.create(&dev->replay_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, &replay);
   if (result != VK_SUCCESS)
      goto fail_pool;
   dev->replay_cmd = container_of(replay, struct wvk_cmd_buffer, vk);

   result = wvk_device_init_meta(dev);
   if (result != VK_SUCCESS)
      goto fail_replay;
   result = wvk_device_init_host_copy(dev);
   if (result != VK_SUCCESS)
      goto fail_host_copy;

   /* The one global priority there is (VK_KHR_global_priority): another
    * is not one of those the query has, or without it, a lower one is
    * this one and a higher one is not permitted. */
   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i) {
      const VkDeviceQueueGlobalPriorityCreateInfoKHR *prio =
         vk_find_struct_const(pCreateInfo->pQueueCreateInfos[i].pNext, DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR);
      if (!prio || prio->globalPriority == VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR)
         continue;
      if (dev->vk.enabled_features.globalPriorityQuery) {
         result = vk_error(dev, VK_ERROR_INITIALIZATION_FAILED);
         goto fail_host_copy_init;
      }
      if (prio->globalPriority > VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR) {
         result = vk_error(dev, VK_ERROR_NOT_PERMITTED_KHR);
         goto fail_host_copy_init;
      }
   }

   /* Two queue families of a queue each: everything, and compute and
    * transfer. The second's submits are the first's, which always exists
    * because it is where they go. */
   const VkDeviceQueueCreateInfo *family_info[WVK_QUEUE_FAMILIES] = { NULL };
   for (uint32_t i = 0; i < pCreateInfo->queueCreateInfoCount; ++i) {
      const VkDeviceQueueCreateInfo *qci = &pCreateInfo->pQueueCreateInfos[i];
      assert(qci->queueFamilyIndex < WVK_QUEUE_FAMILIES && qci->queueCount == 1);
      family_info[qci->queueFamilyIndex] = qci;
   }
   const VkDeviceQueueCreateInfo main_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = WVK_QUEUE_FAMILY_ALL,
      .queueCount = 1,
      .pQueuePriorities = (const float[]){ 1.0f },
   };
   result = wvk_queue_init(dev, &dev->queue, family_info[WVK_QUEUE_FAMILY_ALL] ? family_info[WVK_QUEUE_FAMILY_ALL] : &main_info, 0);
   if (result != VK_SUCCESS)
      goto fail_meta;
   if (family_info[WVK_QUEUE_FAMILY_COMPUTE]) {
      result = wvk_compute_queue_init(dev, &dev->compute_queue, family_info[WVK_QUEUE_FAMILY_COMPUTE]);
      if (result != VK_SUCCESS)
         goto fail_queue;
   }

   *pDevice = wvk_device_to_handle(dev);
   return VK_SUCCESS;

fail_queue:
   wvk_queue_finish(dev, &dev->queue);
fail_meta:
fail_host_copy_init:
   wvk_device_finish_host_copy(dev);
fail_host_copy:
   wvk_device_finish_meta(dev);
fail_replay:
   wvk_cmd_buffer_ops.destroy(&dev->replay_cmd->vk);
fail_pool:
   vk_command_pool_finish(&dev->replay_pool);
fail_caches:
   wvk_device_finish_pipeline_caches(dev);
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

   if (dev->compute_queue.base.device)
      vk_queue_finish(&dev->compute_queue);
   if (dev->queue.vk.base.device)
      wvk_queue_finish(dev, &dev->queue);
   wvk_device_finish_host_copy(dev);
   wvk_device_finish_meta(dev);
   wvk_cmd_buffer_ops.destroy(&dev->replay_cmd->vk);
   vk_command_pool_finish(&dev->replay_pool);
   wvk_device_finish_pipeline_caches(dev);
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

/* VK_EXT_device_fault: no addresses or vendor data, only what WebGPU said
 * when it lost the device. */
VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetDeviceFaultInfoEXT(VkDevice _device, VkDeviceFaultCountsEXT *pFaultCounts, VkDeviceFaultInfoEXT *pFaultInfo)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   pFaultCounts->addressInfoCount = 0;
   pFaultCounts->vendorInfoCount = 0;
   pFaultCounts->vendorBinarySize = 0;
   if (pFaultInfo)
      snprintf(pFaultInfo->description, sizeof(pFaultInfo->description), "%s",
               dev->lost_message[0] ? dev->lost_message : "No fault");
   return VK_SUCCESS;
}

/* A pipeline's identifier (VK_EXT_pipeline_properties): of its shaders'
 * cache keys, which are of everything they were compiled from. */
VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPipelinePropertiesEXT(VkDevice device, const VkPipelineInfoEXT *pPipelineInfo,
                             VkBaseOutStructure *pPipelineProperties)
{
   VK_FROM_HANDLE(vk_pipeline, pipeline, pPipelineInfo->pipeline);
   if (pPipelineProperties->sType != VK_STRUCTURE_TYPE_PIPELINE_PROPERTIES_IDENTIFIER_EXT)
      return VK_SUCCESS;
   VkPipelinePropertiesIdentifierEXT *id = (VkPipelinePropertiesIdentifierEXT *)pPipelineProperties;
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, &pipeline->bind_point, sizeof(pipeline->bind_point));
   const bool compute = pipeline->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE;
   for (uint32_t s = compute ? MESA_SHADER_COMPUTE : 0; s < (compute ? MESA_SHADER_COMPUTE + 1 : MESA_SHADER_COMPUTE);
        ++s) {
      const struct vk_shader *shader = pipeline->ops->get_shader(pipeline, s);
      if (shader)
         _mesa_blake3_update(&ctx, shader->pipeline.cache_key, sizeof(shader->pipeline.cache_key));
   }
   blake3_hash hash;
   _mesa_blake3_final(&ctx, hash);
   memcpy(id->pipelineIdentifier, hash, sizeof(id->pipelineIdentifier));
   return VK_SUCCESS;
}
