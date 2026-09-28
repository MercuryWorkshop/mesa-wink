/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_device_memory.h"

#include "wvk_device.h"
#include "wvk_entrypoints.h"
#include "wvk_physical_device.h"

#include "util/os_memory.h"
#include "util/u_atomic.h"

/* Every usage a buffer's range may be put to, since the memory does not
 * know which buffers will live in it. Map usages cannot be combined with
 * these, so the host copy is separate. */
#define WVK_BUFFER_USAGE                                                        \
   (WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index |  \
    WGPUBufferUsage_Vertex | WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | \
    WGPUBufferUsage_Indirect | WGPUBufferUsage_QueryResolve)

static bool
is_host_visible(const struct wvk_physical_device *pdev, uint32_t type)
{
   return pdev->memory_types[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_AllocateMemory(VkDevice _device, const VkMemoryAllocateInfo *pAllocateInfo,
                   const VkAllocationCallbacks *pAllocator, VkDeviceMemory *pMem)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   struct wvk_physical_device *pdev = wvk_device_physical(dev);

   if (pAllocateInfo->memoryTypeIndex >= WVK_MEMORY_TYPE_COUNT)
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);

   struct wvk_device_memory *mem =
      vk_device_memory_create(&dev->vk, pAllocateInfo, pAllocator, sizeof(*mem));
   if (!mem)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   if (is_host_visible(pdev, mem->vk.memory_type_index)) {
      /* Zeroed: an application may map and read before writing, and what
       * is uploaded should not be whatever the heap held. */
      const uint64_t host_size = wvk_memory_buffer_size(mem->vk.size);
      mem->host = os_malloc_aligned(host_size, 64);
      if (!mem->host) {
         vk_device_memory_destroy(&dev->vk, pAllocator, &mem->vk);
         return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
      }
      memset(mem->host, 0, host_size);
   }
   mem->refs = 1;

   *pMem = wvk_device_memory_to_handle(mem);
   return VK_SUCCESS;
}

void
wvk_device_memory_ref(struct wvk_device_memory *mem)
{
   p_atomic_inc(&mem->refs);
}

void
wvk_device_memory_unref(struct wvk_device_memory *mem)
{
   if (!p_atomic_dec_zero(&mem->refs))
      return;
   if (mem->buffer) {
      wgpuBufferDestroy(mem->buffer);
      wgpuBufferRelease(mem->buffer);
   }
   os_free_aligned(mem->host);
   struct vk_device *dev = mem->vk.base.device;
   vk_device_memory_destroy(dev, mem->free_with_alloc ? &mem->free_alloc : NULL, &mem->vk);
}

VKAPI_ATTR void VKAPI_CALL
wvk_FreeMemory(VkDevice _device, VkDeviceMemory _mem, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_device_memory, mem, _mem);
   if (!mem)
      return;
   (void)dev;
   if (pAllocator) {
      mem->free_alloc = *pAllocator;
      mem->free_with_alloc = true;
   }
   wvk_device_memory_unref(mem);
}

VkResult
wvk_device_memory_get_buffer(struct wvk_device *dev, struct wvk_device_memory *mem, WGPUBuffer *out)
{
   if (!mem->buffer) {
      const struct wvk_physical_device *pdev = wvk_device_physical(dev);
      const uint64_t size = wvk_memory_buffer_size(mem->vk.size);
      if (size > pdev->limits.maxBufferSize)
         return vk_errorf(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY,
                          "%" PRIu64 " bytes of buffer memory is more than WebGPU's %" PRIu64,
                          size, pdev->limits.maxBufferSize);
      WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
      desc.label = wvk_sv("VkDeviceMemory");
      desc.usage = WVK_BUFFER_USAGE;
      desc.size = size;
      mem->buffer = wgpuDeviceCreateBuffer(dev->wgpu, &desc);
      if (!mem->buffer)
         return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      mem->buffer_size = size;
      mem->serial = p_atomic_inc_return(&dev->next_memory_serial);
   }
   *out = mem->buffer;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_MapMemory2KHR(VkDevice _device, const VkMemoryMapInfoKHR *pMemoryMapInfo, void **ppData)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_device_memory, mem, pMemoryMapInfo->memory);

   if (!mem->host)
      return vk_error(dev, VK_ERROR_MEMORY_MAP_FAILED);
   *ppData = (uint8_t *)mem->host + pMemoryMapInfo->offset;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_UnmapMemory2KHR(VkDevice _device, const VkMemoryUnmapInfoKHR *pMemoryUnmapInfo)
{
   return VK_SUCCESS;
}

/* Every host-visible type is coherent. */
VKAPI_ATTR VkResult VKAPI_CALL
wvk_FlushMappedMemoryRanges(VkDevice _device, uint32_t memoryRangeCount,
                            const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_InvalidateMappedMemoryRanges(VkDevice _device, uint32_t memoryRangeCount,
                                 const VkMappedMemoryRange *pMemoryRanges)
{
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetDeviceMemoryCommitment(VkDevice device, VkDeviceMemory _mem, VkDeviceSize *pCommittedMemoryInBytes)
{
   VK_FROM_HANDLE(wvk_device_memory, mem, _mem);
   *pCommittedMemoryInBytes = mem->vk.size;
}
