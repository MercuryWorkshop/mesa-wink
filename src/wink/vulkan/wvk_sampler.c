/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * A VkSampler is a WGPUSampler, and what that cannot say -- a border
 * colour, mirroring once, a LOD bias, unnormalized coordinates, and all of
 * it for integer textures, which WebGPU does not sample -- is its record,
 * which shaders read and act on (wvk_nir_descriptors.c). For those axes the
 * WebGPU sampler clamps to the edge.
 */
#include "wvk_sampler.h"

#include "wvk_device.h"
#include "wvk_entrypoints.h"

static WGPUAddressMode
address_mode(VkSamplerAddressMode mode)
{
   switch (mode) {
   case VK_SAMPLER_ADDRESS_MODE_REPEAT:
      return WGPUAddressMode_Repeat;
   case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT:
      return WGPUAddressMode_MirrorRepeat;
   default:
      /* Edge, border, and mirror-once: WebGPU has only the edge. */
      return WGPUAddressMode_ClampToEdge;
   }
}

static WGPUFilterMode
filter_mode(VkFilter filter)
{
   return filter == VK_FILTER_NEAREST ? WGPUFilterMode_Nearest : WGPUFilterMode_Linear;
}

WGPUCompareFunction
wvk_compare_function(VkCompareOp op)
{
   switch (op) {
   case VK_COMPARE_OP_NEVER: return WGPUCompareFunction_Never;
   case VK_COMPARE_OP_LESS: return WGPUCompareFunction_Less;
   case VK_COMPARE_OP_EQUAL: return WGPUCompareFunction_Equal;
   case VK_COMPARE_OP_LESS_OR_EQUAL: return WGPUCompareFunction_LessEqual;
   case VK_COMPARE_OP_GREATER: return WGPUCompareFunction_Greater;
   case VK_COMPARE_OP_NOT_EQUAL: return WGPUCompareFunction_NotEqual;
   case VK_COMPARE_OP_GREATER_OR_EQUAL: return WGPUCompareFunction_GreaterEqual;
   default: return WGPUCompareFunction_Always;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateSampler(VkDevice _device, const VkSamplerCreateInfo *pCreateInfo,
                  const VkAllocationCallbacks *pAllocator, VkSampler *pSampler)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   const VkSamplerCreateInfo *info = pCreateInfo;

   struct wvk_sampler *sampler = vk_sampler_create(&dev->vk, info, pAllocator, sizeof(*sampler));
   if (!sampler)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
   desc.label = wvk_sv("VkSampler");
   desc.addressModeU = address_mode(info->addressModeU);
   desc.addressModeV = address_mode(info->addressModeV);
   desc.addressModeW = address_mode(info->addressModeW);
   desc.magFilter = filter_mode(info->magFilter);
   desc.minFilter = filter_mode(info->minFilter);
   desc.mipmapFilter = info->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR ? WGPUMipmapFilterMode_Linear
                                                                        : WGPUMipmapFilterMode_Nearest;
   /* WebGPU wants 0 <= min <= max; Vulkan allows max = VK_LOD_CLAMP_NONE. */
   desc.lodMinClamp = CLAMP(info->minLod, 0.0f, 32.0f);
   desc.lodMaxClamp = CLAMP(info->maxLod, desc.lodMinClamp, 32.0f);
   if (info->compareEnable)
      desc.compare = wvk_compare_function(info->compareOp);
   /* Anisotropy wants every filter linear in WebGPU. */
   if (info->anisotropyEnable && desc.magFilter == WGPUFilterMode_Linear &&
       desc.minFilter == WGPUFilterMode_Linear && desc.mipmapFilter == WGPUMipmapFilterMode_Linear)
      desc.maxAnisotropy = (uint16_t)CLAMP(info->maxAnisotropy, 1.0f, 16.0f);
   sampler->wgpu = wgpuDeviceCreateSampler(dev->wgpu, &desc);
   if (!sampler->wgpu) {
      vk_sampler_destroy(&dev->vk, pAllocator, &sampler->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   if (info->compareEnable)
      sampler->binding_type = WGPUSamplerBindingType_Comparison;
   else if (desc.magFilter == WGPUFilterMode_Linear || desc.minFilter == WGPUFilterMode_Linear ||
            desc.mipmapFilter == WGPUMipmapFilterMode_Linear)
      sampler->binding_type = WGPUSamplerBindingType_Filtering;
   else
      sampler->binding_type = WGPUSamplerBindingType_NonFiltering;

   const VkSamplerAddressMode modes[3] = { info->addressModeU, info->addressModeV, info->addressModeW };
   uint32_t flags = 0;
   for (uint32_t i = 0; i < 3; ++i) {
      flags |= ((uint32_t)modes[i] & 7) << (3 * i);
      if (modes[i] == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER ||
          modes[i] == VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE)
         flags |= WVK_SAMPLER_EMULATE;
   }
   if (info->unnormalizedCoordinates)
      flags |= WVK_SAMPLER_UNNORMALIZED | WVK_SAMPLER_EMULATE;
   if (info->magFilter == VK_FILTER_LINEAR)
      flags |= WVK_SAMPLER_MAG_LINEAR;
   if (info->minFilter == VK_FILTER_LINEAR)
      flags |= WVK_SAMPLER_MIN_LINEAR;
   if (info->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR)
      flags |= WVK_SAMPLER_MIP_LINEAR;
   if (info->compareEnable)
      flags |= WVK_SAMPLER_COMPARE | (((uint32_t)info->compareOp & 7) << WVK_SAMPLER_COMPARE_SHIFT);
   sampler->meta[WVK_META_SAMPLER_FLAGS] = flags;
   memcpy(&sampler->meta[WVK_META_SAMPLER_BIAS], &info->mipLodBias, 4);
   memcpy(&sampler->meta[WVK_META_SAMPLER_MIN_LOD], &info->minLod, 4);
   memcpy(&sampler->meta[WVK_META_SAMPLER_MAX_LOD], &info->maxLod, 4);
   memcpy(&sampler->meta[WVK_META_SAMPLER_BORDER], &sampler->vk.border_color_value, 16);

   *pSampler = wvk_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroySampler(VkDevice _device, VkSampler _sampler, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_sampler, sampler, _sampler);
   if (!sampler)
      return;
   wgpuSamplerRelease(sampler->wgpu);
   vk_sampler_destroy(&dev->vk, pAllocator, &sampler->vk);
}
