/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_image.h"

#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_format.h"
#include "wvk_physical_device.h"

#include "vk_format.h"
#include "util/format/u_format.h"

/* WebGPU lets a texture be viewed only as its sRGB or linear twin. */
static WGPUTextureFormat
srgb_twin(WGPUTextureFormat f)
{
   switch (f) {
#define TWIN(a, b) case WGPUTextureFormat_##a: return WGPUTextureFormat_##b; \
                   case WGPUTextureFormat_##b: return WGPUTextureFormat_##a
   TWIN(RGBA8Unorm, RGBA8UnormSrgb);
   TWIN(BGRA8Unorm, BGRA8UnormSrgb);
   TWIN(BC1RGBAUnorm, BC1RGBAUnormSrgb);
   TWIN(BC2RGBAUnorm, BC2RGBAUnormSrgb);
   TWIN(BC3RGBAUnorm, BC3RGBAUnormSrgb);
   TWIN(BC7RGBAUnorm, BC7RGBAUnormSrgb);
   TWIN(ETC2RGB8Unorm, ETC2RGB8UnormSrgb);
   TWIN(ETC2RGB8A1Unorm, ETC2RGB8A1UnormSrgb);
   TWIN(ETC2RGBA8Unorm, ETC2RGBA8UnormSrgb);
   TWIN(ASTC4x4Unorm, ASTC4x4UnormSrgb);
   TWIN(ASTC5x4Unorm, ASTC5x4UnormSrgb);
   TWIN(ASTC5x5Unorm, ASTC5x5UnormSrgb);
   TWIN(ASTC6x5Unorm, ASTC6x5UnormSrgb);
   TWIN(ASTC6x6Unorm, ASTC6x6UnormSrgb);
   TWIN(ASTC8x5Unorm, ASTC8x5UnormSrgb);
   TWIN(ASTC8x6Unorm, ASTC8x6UnormSrgb);
   TWIN(ASTC8x8Unorm, ASTC8x8UnormSrgb);
   TWIN(ASTC10x5Unorm, ASTC10x5UnormSrgb);
   TWIN(ASTC10x6Unorm, ASTC10x6UnormSrgb);
   TWIN(ASTC10x8Unorm, ASTC10x8UnormSrgb);
   TWIN(ASTC10x10Unorm, ASTC10x10UnormSrgb);
   TWIN(ASTC12x10Unorm, ASTC12x10UnormSrgb);
   TWIN(ASTC12x12Unorm, ASTC12x12UnormSrgb);
#undef TWIN
   default:
      return WGPUTextureFormat_Undefined;
   }
}

WGPUTextureAspect
wvk_aspect(VkImageAspectFlags aspects)
{
   if (aspects == VK_IMAGE_ASPECT_DEPTH_BIT)
      return WGPUTextureAspect_DepthOnly;
   if (aspects == VK_IMAGE_ASPECT_STENCIL_BIT)
      return WGPUTextureAspect_StencilOnly;
   return WGPUTextureAspect_All;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateImage(VkDevice _device, const VkImageCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);

   const struct wvk_format *format = wvk_format_get(pCreateInfo->format);
   const uint32_t caps = wvk_format_caps(pdev, pCreateInfo->format);
   if (!format || !caps || pCreateInfo->tiling != VK_IMAGE_TILING_OPTIMAL)
      return vk_error(dev, VK_ERROR_FORMAT_NOT_SUPPORTED);

   struct wvk_image *image = vk_image_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*image));
   if (!image)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   image->format = format->texture;
   image->dimension = pCreateInfo->imageType == VK_IMAGE_TYPE_3D ? WGPUTextureDimension_3D
                                                                  : WGPUTextureDimension_2D;

   /* Beyond what was asked: clears and blits are render passes, some copies
    * are shaders, and all of them need the texture bound one way or the
    * other. */
   const VkImageUsageFlags usage = image->vk.usage | image->vk.stencil_usage;
   WGPUTextureUsage wusage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
   if (caps & WVK_CAP_SAMPLE)
      wusage |= WGPUTextureUsage_TextureBinding;
   if (caps & (WVK_CAP_RENDER | WVK_CAP_DEPTH | WVK_CAP_STENCIL))
      wusage |= WGPUTextureUsage_RenderAttachment;
   if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && (caps & WVK_CAP_STORAGE) &&
       pCreateInfo->samples == VK_SAMPLE_COUNT_1_BIT)
      wusage |= WGPUTextureUsage_StorageBinding;
   image->usage = wusage;

   WGPUTextureFormat view_formats[1];
   size_t view_format_count = 0;
   if (pCreateInfo->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) {
      const WGPUTextureFormat twin = srgb_twin(format->texture);
      if (twin != WGPUTextureFormat_Undefined)
         view_formats[view_format_count++] = twin;
   }

   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("VkImage");
   desc.usage = wusage;
   desc.dimension = image->dimension;
   desc.size.width = image->vk.extent.width;
   desc.size.height = image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : image->vk.extent.height;
   desc.size.depthOrArrayLayers = image->vk.image_type == VK_IMAGE_TYPE_3D
                                     ? image->vk.extent.depth : image->vk.array_layers;
   desc.format = format->texture;
   desc.mipLevelCount = image->vk.mip_levels;
   desc.sampleCount = image->vk.samples;
   desc.viewFormatCount = view_format_count;
   desc.viewFormats = view_formats;
   image->texture = wgpuDeviceCreateTexture(dev->wgpu, &desc);
   if (!image->texture) {
      vk_image_destroy(&dev->vk, pAllocator, &image->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   *pImage = wvk_image_to_handle(image);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyImage(VkDevice _device, VkImage _image, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_image, image, _image);
   if (!image)
      return;
   /* Swapchain images belong to the surface. */
   if (image->texture) {
      wgpuTextureDestroy(image->texture);
      wgpuTextureRelease(image->texture);
   }
   vk_image_destroy(&dev->vk, pAllocator, &image->vk);
}

/* What the texture would take, for applications that suballocate. */
static uint64_t
image_size(VkFormat vk_format, VkImageType type, VkExtent3D extent, uint32_t levels,
           uint32_t layers, VkSampleCountFlagBits samples)
{
   const enum pipe_format format = vk_format_to_pipe_format(vk_format);
   const uint32_t bw = util_format_get_blockwidth(format);
   const uint32_t bh = util_format_get_blockheight(format);
   const uint32_t bytes = MAX2(util_format_get_blocksize(format), 1);
   uint64_t size = 0;
   for (uint32_t m = 0; m < levels; ++m) {
      const uint32_t w = u_minify(extent.width, m);
      const uint32_t h = type == VK_IMAGE_TYPE_1D ? 1 : u_minify(extent.height, m);
      const uint32_t d = type == VK_IMAGE_TYPE_3D ? u_minify(extent.depth, m) : 1;
      size += (uint64_t)DIV_ROUND_UP(w, bw) * DIV_ROUND_UP(h, bh) * bytes * d;
   }
   return size * layers * samples;
}

static void
get_image_requirements(VkFormat format, VkImageType type, VkExtent3D extent, uint32_t levels,
                       uint32_t layers, VkSampleCountFlagBits samples,
                       VkMemoryRequirements2 *pMemoryRequirements)
{
   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = align64(MAX2(image_size(format, type, extent, levels, layers, samples), 1), 256),
      .alignment = 256,
      /* There are no linear images to map. */
      .memoryTypeBits = BITFIELD_BIT(WVK_MEMORY_DEVICE),
   };
   vk_foreach_struct(stype, ext, pMemoryRequirements->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS: {
         VkMemoryDedicatedRequirements *dedicated = ext;
         dedicated->prefersDedicatedAllocation = false;
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
wvk_GetImageMemoryRequirements2(VkDevice _device, const VkImageMemoryRequirementsInfo2 *pInfo,
                                VkMemoryRequirements2 *pMemoryRequirements)
{
   VK_FROM_HANDLE(wvk_image, image, pInfo->image);
   get_image_requirements(image->vk.format, image->vk.image_type, image->vk.extent,
                          image->vk.mip_levels, image->vk.array_layers, image->vk.samples,
                          pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetDeviceImageMemoryRequirements(VkDevice _device, const VkDeviceImageMemoryRequirements *pInfo,
                                     VkMemoryRequirements2 *pMemoryRequirements)
{
   const VkImageCreateInfo *ci = pInfo->pCreateInfo;
   get_image_requirements(ci->format, ci->imageType, ci->extent, ci->mipLevels, ci->arrayLayers,
                          ci->samples, pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetImageSparseMemoryRequirements2(VkDevice device, const VkImageSparseMemoryRequirementsInfo2 *pInfo,
                                      uint32_t *pSparseMemoryRequirementCount,
                                      VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   *pSparseMemoryRequirementCount = 0;
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetDeviceImageSparseMemoryRequirements(VkDevice device, const VkDeviceImageMemoryRequirements *pInfo,
                                           uint32_t *pSparseMemoryRequirementCount,
                                           VkSparseImageMemoryRequirements2 *pSparseMemoryRequirements)
{
   *pSparseMemoryRequirementCount = 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_BindImageMemory2(VkDevice _device, uint32_t bindInfoCount, const VkBindImageMemoryInfo *pBindInfos)
{
   for (uint32_t i = 0; i < bindInfoCount; ++i) {
      const VkBindMemoryStatusKHR *status = vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
      if (status && status->pResult)
         *status->pResult = VK_SUCCESS;
   }
   return VK_SUCCESS;
}

/* Only linear images have a layout to report, and there are none. */
VKAPI_ATTR void VKAPI_CALL
wvk_GetImageSubresourceLayout2KHR(VkDevice device, VkImage image, const VkImageSubresource2KHR *pSubresource,
                                  VkSubresourceLayout2KHR *pLayout)
{
   pLayout->subresourceLayout = (VkSubresourceLayout){ 0 };
}

/* ---------------------------------------------------------------------- */
/* Views */

static WGPUTextureViewDimension
view_dimension(VkImageViewType type)
{
   switch (type) {
   case VK_IMAGE_VIEW_TYPE_1D:
   case VK_IMAGE_VIEW_TYPE_2D: return WGPUTextureViewDimension_2D;
   case VK_IMAGE_VIEW_TYPE_1D_ARRAY:
   case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return WGPUTextureViewDimension_2DArray;
   case VK_IMAGE_VIEW_TYPE_CUBE: return WGPUTextureViewDimension_Cube;
   case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return WGPUTextureViewDimension_CubeArray;
   case VK_IMAGE_VIEW_TYPE_3D: return WGPUTextureViewDimension_3D;
   default: return WGPUTextureViewDimension_Undefined;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateImageView(VkDevice _device, const VkImageViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_image, image, pCreateInfo->image);

   struct wvk_image_view *view = vk_image_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   const struct wvk_format *format = wvk_format_get(view->vk.format);
   view->format = format ? format->texture : image->format;
   view->aspect = wvk_aspect(view->vk.aspects);

   /* Views of an image without a texture -- a swapchain's, until acquired
    * -- are made when used. */
   if (image->texture && (image->usage & WGPUTextureUsage_TextureBinding)) {
      WGPUTextureViewDescriptor desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      /* One aspect of a depth-stencil texture is viewed as that aspect's
       * own format, which WebGPU works out itself. */
      desc.format = view->aspect == WGPUTextureAspect_All ? view->format : WGPUTextureFormat_Undefined;
      desc.dimension = view_dimension(view->vk.view_type);
      desc.baseMipLevel = view->vk.base_mip_level;
      desc.mipLevelCount = view->vk.level_count;
      desc.baseArrayLayer = image->dimension == WGPUTextureDimension_3D ? 0 : view->vk.base_array_layer;
      desc.arrayLayerCount = image->dimension == WGPUTextureDimension_3D ? 1 : view->vk.layer_count;
      desc.aspect = view->aspect;
      view->view = wgpuTextureCreateView(image->texture, &desc);
   }

   *pView = wvk_image_view_to_handle(view);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyImageView(VkDevice _device, VkImageView _view, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_image_view, view, _view);
   if (!view)
      return;
   if (view->view)
      wgpuTextureViewRelease(view->view);
   vk_image_view_destroy(&dev->vk, pAllocator, &view->vk);
}

WGPUTextureView
wvk_image_attachment_view(struct wvk_image *image, WGPUTextureFormat format, WGPUTextureAspect aspect,
                          uint32_t level, uint32_t layer)
{
   WGPUTextureViewDescriptor desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
   desc.format = aspect == WGPUTextureAspect_All ? format : WGPUTextureFormat_Undefined;
   desc.dimension = image->dimension == WGPUTextureDimension_3D ? WGPUTextureViewDimension_3D
                                                                 : WGPUTextureViewDimension_2D;
   desc.baseMipLevel = level;
   desc.mipLevelCount = 1;
   desc.baseArrayLayer = image->dimension == WGPUTextureDimension_3D ? 0 : layer;
   desc.arrayLayerCount = 1;
   desc.aspect = aspect;
   return wgpuTextureCreateView(image->texture, &desc);
}
