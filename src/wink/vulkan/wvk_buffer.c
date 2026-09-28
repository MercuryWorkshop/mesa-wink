/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_buffer.h"

#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_physical_device.h"

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateBuffer(VkDevice _device, const VkBufferCreateInfo *pCreateInfo,
                 const VkAllocationCallbacks *pAllocator, VkBuffer *pBuffer)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);

   if (pCreateInfo->size > pdev->limits.maxBufferSize)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   struct wvk_buffer *buffer = vk_buffer_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*buffer));
   if (!buffer)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   *pBuffer = wvk_buffer_to_handle(buffer);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyBuffer(VkDevice _device, VkBuffer _buffer, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_buffer, buffer, _buffer);
   if (buffer)
      vk_buffer_destroy(&dev->vk, pAllocator, &buffer->vk);
}

static void
get_requirements(const struct wvk_device *dev, VkDeviceSize size, VkBufferUsageFlags2 usage,
                 VkMemoryRequirements2 *pMemoryRequirements)
{
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   /* A range bound as a uniform or storage buffer must start where WebGPU
    * allows; anything else, on four bytes, which is what copies and vertex
    * and index buffers need. */
   uint64_t alignment = 16;
   if (usage & (VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_2_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_2_STORAGE_TEXEL_BUFFER_BIT |
                VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT))
      alignment = MAX3(alignment, pdev->limits.minUniformBufferOffsetAlignment,
                       pdev->limits.minStorageBufferOffsetAlignment);

   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = align64(size, 4),
      .alignment = alignment,
      .memoryTypeBits = BITFIELD_MASK(WVK_MEMORY_TYPE_COUNT),
   };

   vk_foreach_struct(stype, ext, pMemoryRequirements->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *dedicated = ext;
         /* A storage buffer written in a pass may not share a WGPUBuffer
          * with anything read in it, and a dedicated allocation cannot. */
         dedicated->prefersDedicatedAllocation = (usage & VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT) != 0;
         dedicated->requiresDedicatedAllocation = false;
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetBufferMemoryRequirements2(VkDevice _device, const VkBufferMemoryRequirementsInfo2 *pInfo,
                                 VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_buffer, buffer, pInfo->buffer);
   get_requirements(dev, buffer->vk.size, buffer->vk.usage, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetDeviceBufferMemoryRequirements(VkDevice _device, const VkDeviceBufferMemoryRequirements *pInfo,
                                      VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   get_requirements(dev, pInfo->pCreateInfo->size, vk_buffer_usage_flags(pInfo->pCreateInfo),
                    pMemoryRequirements);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_BindBufferMemory2(VkDevice _device, uint32_t bindInfoCount, const VkBindBufferMemoryInfo *pBindInfos)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VkResult first_error = VK_SUCCESS;

   for (uint32_t i = 0; i < bindInfoCount; ++i) {
      VK_FROM_HANDLE(wvk_buffer, buffer, pBindInfos[i].buffer);
      VK_FROM_HANDLE(wvk_device_memory, mem, pBindInfos[i].memory);
      WGPUBuffer wgpu;
      VkResult result = wvk_device_memory_get_buffer(dev, mem, &wgpu);
      if (result == VK_SUCCESS) {
         buffer->mem = mem;
         buffer->offset = pBindInfos[i].memoryOffset;
      } else if (first_error == VK_SUCCESS) {
         first_error = result;
      }
      const VkBindMemoryStatusKHR *status = vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
      if (status && status->pResult)
         *status->pResult = result;
   }
   return first_error;
}
