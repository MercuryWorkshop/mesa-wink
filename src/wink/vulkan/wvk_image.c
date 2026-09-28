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
   const bool linear = pCreateInfo->tiling == VK_IMAGE_TILING_LINEAR;
   if (!format || !caps || (pCreateInfo->tiling != VK_IMAGE_TILING_OPTIMAL && !linear))
      return vk_error(dev, VK_ERROR_FORMAT_NOT_SUPPORTED);
   if (linear && (!wvk_image_linear_supported(pdev, pCreateInfo->format, pCreateInfo->imageType,
                                              pCreateInfo->usage) ||
                  pCreateInfo->mipLevels != 1 || pCreateInfo->arrayLayers != 1 ||
                  pCreateInfo->samples != VK_SAMPLE_COUNT_1_BIT))
      return vk_error(dev, VK_ERROR_FORMAT_NOT_SUPPORTED);

   struct wvk_image *image = vk_image_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*image));
   if (!image)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   image->format = format->texture;
   if (linear) {
      image->linear = true;
      image->pitch = wvk_image_linear_pitch(pCreateInfo->format, pCreateInfo->extent.width);
      image->dimension = WGPUTextureDimension_2D;
      *pImage = wvk_image_to_handle(image);
      return VK_SUCCESS;
   }

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
   /* Copies into a packed format's texture are a shader's stores. */
   if ((caps & WVK_CAP_PACKED16) && pCreateInfo->samples == VK_SAMPLE_COUNT_1_BIT)
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
   /* WebGPU wants a block-compressed texture a whole number of blocks;
    * Vulkan takes any size. The texture is the next size up, and edge
    * copies go to its blocks; sampling is of a shadow of its own size. */
   const enum pipe_format pformat = vk_format_to_pipe_format(pCreateInfo->format);
   desc.size.width = util_align_npot(image->vk.extent.width, util_format_get_blockwidth(pformat));
   desc.size.height = image->vk.image_type == VK_IMAGE_TYPE_1D
                         ? 1 : util_align_npot(image->vk.extent.height, util_format_get_blockheight(pformat));
   desc.size.depthOrArrayLayers = image->vk.image_type == VK_IMAGE_TYPE_3D
                                     ? image->vk.extent.depth : image->vk.array_layers;
   desc.format = format->texture;
   desc.mipLevelCount = image->vk.mip_levels;
   desc.sampleCount = image->vk.samples;
   desc.viewFormatCount = view_format_count;
   desc.viewFormats = view_formats;
   if (image->vk.samples > 1 && image->vk.array_layers > 1) {
      /* Multisampled textures are one layer: one for each. */
      desc.size.depthOrArrayLayers = 1;
      image->layer_textures = calloc(image->vk.array_layers, sizeof(*image->layer_textures));
      for (uint32_t l = 0; image->layer_textures && l < image->vk.array_layers; ++l) {
         image->layer_textures[l] = wgpuDeviceCreateTexture(dev->wgpu, &desc);
         if (!image->layer_textures[l])
            break;
      }
      image->texture = image->layer_textures && image->layer_textures[image->vk.array_layers - 1]
                          ? image->layer_textures[0] : NULL;
   } else {
      image->texture = wgpuDeviceCreateTexture(dev->wgpu, &desc);
   }
   if (!image->texture) {
      if (image->layer_textures) {
         for (uint32_t l = 0; l < image->vk.array_layers && image->layer_textures[l]; ++l)
            wgpuTextureRelease(image->layer_textures[l]);
         free(image->layer_textures);
      }
      vk_image_destroy(&dev->vk, pAllocator, &image->vk);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   /* A compressed image of a size not whole blocks, sampled from a texture
    * of its size, uncompressed. */
   if ((caps & WVK_CAP_COMPRESSED) && (wusage & WGPUTextureUsage_TextureBinding) &&
       (desc.size.width != image->vk.extent.width ||
        (image->vk.image_type != VK_IMAGE_TYPE_1D && desc.size.height != image->vk.extent.height))) {
      const VkFormat f = pCreateInfo->format;
      const bool wide = f == VK_FORMAT_BC4_SNORM_BLOCK || f == VK_FORMAT_BC5_SNORM_BLOCK ||
                        f == VK_FORMAT_BC6H_UFLOAT_BLOCK || f == VK_FORMAT_BC6H_SFLOAT_BLOCK ||
                        (f >= VK_FORMAT_EAC_R11_UNORM_BLOCK && f <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK);
      image->shadow_format = wide ? WGPUTextureFormat_RGBA16Float
                             : vk_format_is_srgb(f) ? WGPUTextureFormat_RGBA8UnormSrgb
                                                    : WGPUTextureFormat_RGBA8Unorm;
      WGPUTextureFormat twin[1] = { srgb_twin(image->shadow_format) };
      WGPUTextureDescriptor sd = desc;
      sd.label = wvk_sv("VkImage decoded");
      sd.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
      sd.format = image->shadow_format;
      sd.size.width = image->vk.extent.width;
      sd.size.height = image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : image->vk.extent.height;
      sd.viewFormatCount = twin[0] != WGPUTextureFormat_Undefined && (pCreateInfo->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
      sd.viewFormats = twin;
      image->shadow = wgpuDeviceCreateTexture(dev->wgpu, &sd);
   }

   /* Texels atomics can be done on, as a buffer too. */
   if ((wusage & WGPUTextureUsage_StorageBinding) && wvk_format_has_atomics(pCreateInfo->format)) {
      uint64_t size = 0;
      for (uint32_t l = 0; l < image->vk.mip_levels && l < ARRAY_SIZE(image->atomic_offset); ++l) {
         const VkExtent3D e = vk_image_mip_level_extent(&image->vk, l);
         image->atomic_offset[l] = size;
         image->atomic_pitch[l] = align(e.width * 4, 256);
         size += (uint64_t)image->atomic_pitch[l] * e.height *
                 (image->vk.image_type == VK_IMAGE_TYPE_3D ? e.depth : image->vk.array_layers);
      }
      WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
      bd.label = wvk_sv("VkImage atomics");
      bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
      bd.size = size;
      image->atomic_size = size;
      image->atomic_buffer = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
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
   /* Swapchain images belong to the surface. Released, not destroyed:
    * views of it in descriptors, which may outlive it unused, still have
    * it, and it goes with the last of them. */
   if (image->layer_textures) {
      /* The first is `texture`. */
      for (uint32_t l = 1; l < image->vk.array_layers; ++l)
         wgpuTextureRelease(image->layer_textures[l]);
      free(image->layer_textures);
   }
   if (image->texture)
      wgpuTextureRelease(image->texture);
   if (image->atomic_buffer)
      wgpuBufferRelease(image->atomic_buffer);
   if (image->shadow)
      wgpuTextureRelease(image->shadow);
   vk_image_destroy(&dev->vk, pAllocator, &image->vk);
}

WGPUTexture
wvk_image_snapshot(WGPUDevice device, const struct wvk_image *image, WGPUCommandEncoder encoder)
{
   if (!image->texture || image->vk.samples > 1 || image->layer_textures ||
       (image->vk.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)))
      return NULL;
   WGPUTextureFormat view_formats[1];
   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("VkImage copy");
   desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
   desc.dimension = image->dimension;
   desc.size = (WGPUExtent3D){ wgpuTextureGetWidth(image->texture), wgpuTextureGetHeight(image->texture),
                               wgpuTextureGetDepthOrArrayLayers(image->texture) };
   desc.format = image->format;
   desc.mipLevelCount = wgpuTextureGetMipLevelCount(image->texture);
   if (image->vk.create_flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) {
      view_formats[0] = srgb_twin(image->format);
      desc.viewFormatCount = view_formats[0] != WGPUTextureFormat_Undefined;
      desc.viewFormats = view_formats;
   }
   WGPUTexture copy = wgpuDeviceCreateTexture(device, &desc);
   if (!copy)
      return NULL;
   for (uint32_t l = 0; l < desc.mipLevelCount; ++l) {
      WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      src.texture = image->texture;
      src.mipLevel = l;
      dst.texture = copy;
      dst.mipLevel = l;
      const WGPUExtent3D extent = {
         MAX2(desc.size.width >> l, 1), MAX2(desc.size.height >> l, 1),
         desc.dimension == WGPUTextureDimension_3D ? MAX2(desc.size.depthOrArrayLayers >> l, 1)
                                                    : desc.size.depthOrArrayLayers,
      };
      wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &extent);
   }
   return copy;
}

void
wvk_image_copy_atomic(struct wvk_image *image, WGPUCommandEncoder encoder, bool to_buffer)
{
   for (uint32_t l = 0; l < image->vk.mip_levels && l < ARRAY_SIZE(image->atomic_offset); ++l) {
      const VkExtent3D e = vk_image_mip_level_extent(&image->vk, l);
      WGPUTexelCopyTextureInfo texture = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      texture.texture = image->texture;
      texture.mipLevel = l;
      WGPUTexelCopyBufferInfo buffer = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
      buffer.buffer = image->atomic_buffer;
      buffer.layout.offset = image->atomic_offset[l];
      buffer.layout.bytesPerRow = image->atomic_pitch[l];
      buffer.layout.rowsPerImage = e.height;
      const WGPUExtent3D extent = {
         e.width, e.height, image->vk.image_type == VK_IMAGE_TYPE_3D ? e.depth : image->vk.array_layers,
      };
      if (to_buffer)
         wgpuCommandEncoderCopyTextureToBuffer(encoder, &texture, &buffer, &extent);
      else
         wgpuCommandEncoderCopyBufferToTexture(encoder, &buffer, &texture, &extent);
   }
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

bool
wvk_image_linear_supported(const struct wvk_physical_device *pdev, VkFormat format, VkImageType type,
                           VkImageUsageFlags usage)
{
   const uint32_t caps = wvk_format_caps(pdev, format);
   return type == VK_IMAGE_TYPE_2D && caps &&
          !(caps & (WVK_CAP_COMPRESSED | WVK_CAP_DEPTH | WVK_CAP_STENCIL)) &&
          !(usage & ~(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT));
}

uint32_t
wvk_image_linear_pitch(VkFormat format, uint32_t width)
{
   return align(width * util_format_get_blocksize(vk_format_to_pipe_format(format)), 256);
}

static void
get_image_requirements(VkFormat format, VkImageType type, VkImageTiling tiling, VkExtent3D extent,
                       uint32_t levels, uint32_t layers, VkSampleCountFlagBits samples,
                       VkMemoryRequirements2 *pMemoryRequirements)
{
   const bool linear = tiling == VK_IMAGE_TILING_LINEAR;
   const uint64_t size = linear ? (uint64_t)wvk_image_linear_pitch(format, extent.width) * extent.height
                                : image_size(format, type, extent, levels, layers, samples);
   pMemoryRequirements->memoryRequirements = (VkMemoryRequirements){
      .size = align64(MAX2(size, 1), 256),
      .alignment = 256,
      /* Linear images are memory, and may be mapped; the rest are
       * textures of their own. */
      .memoryTypeBits = linear ? BITFIELD_MASK(WVK_MEMORY_TYPE_COUNT) : BITFIELD_BIT(WVK_MEMORY_DEVICE),
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
   get_image_requirements(image->vk.format, image->vk.image_type, image->vk.tiling, image->vk.extent,
                          image->vk.mip_levels, image->vk.array_layers, image->vk.samples,
                          pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetDeviceImageMemoryRequirements(VkDevice _device, const VkDeviceImageMemoryRequirements *pInfo,
                                     VkMemoryRequirements2 *pMemoryRequirements)
{
   const VkImageCreateInfo *ci = pInfo->pCreateInfo;
   get_image_requirements(ci->format, ci->imageType, ci->tiling, ci->extent, ci->mipLevels,
                          ci->arrayLayers, ci->samples, pMemoryRequirements);
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
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < bindInfoCount; ++i) {
      VK_FROM_HANDLE(wvk_image, image, pBindInfos[i].image);
      VK_FROM_HANDLE(wvk_device_memory, mem, pBindInfos[i].memory);
      VkResult r = VK_SUCCESS;
      image->mem = mem;
      image->offset = pBindInfos[i].memoryOffset;
      /* A linear image is its memory's buffer. */
      if (image->linear && mem) {
         WGPUBuffer buffer;
         r = wvk_device_memory_get_buffer(dev, mem, &buffer);
      }
      const VkBindMemoryStatusKHR *status = vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
      if (status && status->pResult)
         *status->pResult = r;
      if (r != VK_SUCCESS)
         result = r;
   }
   return result;
}

/* Only linear images have a layout to report. */
VKAPI_ATTR void VKAPI_CALL
wvk_GetImageSubresourceLayout2KHR(VkDevice device, VkImage _image, const VkImageSubresource2KHR *pSubresource,
                                  VkSubresourceLayout2KHR *pLayout)
{
   VK_FROM_HANDLE(wvk_image, image, _image);
   pLayout->subresourceLayout = (VkSubresourceLayout){ 0 };
   if (!image->linear)
      return;
   const uint64_t size = (uint64_t)image->pitch * image->vk.extent.height;
   pLayout->subresourceLayout = (VkSubresourceLayout){
      .offset = 0,
      .size = size,
      .rowPitch = image->pitch,
      .arrayPitch = size,
      .depthPitch = size,
   };
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
   VkComponentMapping mapping = view->vk.swizzle;
   /* A packed format without alpha in an RGBA8 texture: whatever alpha it
    * was rendered with, the format's is one. */
   if (format && (format->caps & WVK_CAP_PACKED16) &&
       !util_format_has_alpha(vk_format_to_pipe_format(view->vk.view_format))) {
      VkComponentSwizzle *c[4] = { &mapping.r, &mapping.g, &mapping.b, &mapping.a };
      for (unsigned i = 0; i < 4; ++i) {
         if (*c[i] == VK_COMPONENT_SWIZZLE_A)
            *c[i] = VK_COMPONENT_SWIZZLE_ONE;
      }
   }
   const VkComponentMapping *m = &mapping;
   if (m->r != VK_COMPONENT_SWIZZLE_R || m->g != VK_COMPONENT_SWIZZLE_G || m->b != VK_COMPONENT_SWIZZLE_B ||
       m->a != VK_COMPONENT_SWIZZLE_A)
      view->swizzle = (uint32_t)m->r | (uint32_t)m->g << 3 | (uint32_t)m->b << 6 | (uint32_t)m->a << 9;
   /* A depth or stencil aspect is red; else what the format has. */
   if (view->vk.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
      view->channels = 1;
   } else {
      const struct util_format_description *desc =
         util_format_description(vk_format_to_pipe_format(view->vk.view_format));
      for (unsigned i = 0; desc && i < 4; ++i) {
         if (desc->swizzle[i] <= PIPE_SWIZZLE_W)
            view->channels |= 1u << i;
      }
   }

   /* Views of an image without a texture -- a swapchain's, until acquired
    * -- are made when used. */
   if (image->texture && (image->usage & WGPUTextureUsage_TextureBinding)) {
      WGPUTextureViewDescriptor desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      /* One aspect of a depth-stencil texture is viewed as that aspect's
       * own format, which WebGPU works out itself. */
      desc.format = view->aspect == WGPUTextureAspect_All ? view->format : WGPUTextureFormat_Undefined;
      desc.dimension = view_dimension(view->vk.view_type);
      /* A multisampled view is 2D in WebGPU, whatever array it says. */
      if (image->vk.samples > 1)
         desc.dimension = WGPUTextureViewDimension_2D;
      desc.baseMipLevel = view->vk.base_mip_level;
      desc.mipLevelCount = view->vk.level_count;
      uint32_t texture_layer;
      WGPUTexture texture = wvk_image_layer_texture(image, view->vk.base_array_layer, &texture_layer);
      /* A compressed image of a size not whole blocks is sampled from what
       * it decodes to. */
      if (image->shadow && vk_format_is_compressed(view->vk.view_format)) {
         texture = image->shadow;
         desc.format = vk_format_is_srgb(view->vk.view_format) ? WGPUTextureFormat_RGBA8UnormSrgb
                       : image->shadow_format == WGPUTextureFormat_RGBA8UnormSrgb ? WGPUTextureFormat_RGBA8Unorm
                                                                                   : image->shadow_format;
      }
      desc.baseArrayLayer = image->dimension == WGPUTextureDimension_3D ? 0 : texture_layer;
      desc.arrayLayerCount = image->dimension == WGPUTextureDimension_3D ? 1 : view->vk.layer_count;
      /* Of a multisampled array, the first layer: shaders have that. */
      if (image->vk.samples > 1)
         desc.arrayLayerCount = 1;
      desc.aspect = view->aspect;
      view->view = wgpuTextureCreateView(texture, &desc);
      view->view_desc = desc;
      /* Cubes as storage images, and input attachments, are 2D arrays. */
      if (desc.dimension == WGPUTextureViewDimension_Cube || desc.dimension == WGPUTextureViewDimension_CubeArray ||
          (desc.dimension == WGPUTextureViewDimension_2D && image->vk.samples == 1 &&
           (view->vk.usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))) {
         desc.dimension = WGPUTextureViewDimension_2DArray;
         view->array_view = wgpuTextureCreateView(texture, &desc);
         view->array_desc = desc;
      }
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
   if (view->array_view)
      wgpuTextureViewRelease(view->array_view);
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
   uint32_t texture_layer;
   WGPUTexture texture = wvk_image_layer_texture(image, layer, &texture_layer);
   desc.baseArrayLayer = image->dimension == WGPUTextureDimension_3D ? 0 : texture_layer;
   desc.arrayLayerCount = 1;
   desc.aspect = aspect;
   return wgpuTextureCreateView(texture, &desc);
}
