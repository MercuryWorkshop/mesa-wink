/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_image.h"

#include "wvk_cmd_buffer.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_format.h"
#include "wvk_physical_device.h"

#include "vk_format.h"
#include "util/format/u_format.h"
#include "util/u_atomic.h"

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

static void get_image_requirements(VkFormat format, VkImageType type, VkImageTiling tiling, VkExtent3D extent,
                                   uint32_t levels, uint32_t layers, VkSampleCountFlagBits samples,
                                   VkMemoryRequirements2 *pMemoryRequirements);

/* A plane of a multi-planar image as an image of its own: of the plane's
 * format and size, and none of the image's extensions. */
static VkImageCreateInfo
plane_create_info(const VkImageCreateInfo *info, uint32_t plane)
{
   const struct vk_format_ycbcr_info *ycbcr = vk_format_get_ycbcr_info(info->format);
   VkImageCreateInfo p = *info;
   p.pNext = NULL;
   p.format = ycbcr->planes[plane].format;
   p.extent.width = vk_format_get_plane_width(info->format, plane, info->extent.width);
   p.extent.height = vk_format_get_plane_height(info->format, plane, info->extent.height);
   p.flags &= ~VK_IMAGE_CREATE_DISJOINT_BIT;
   return p;
}

/* Where each plane of a multi-planar image is in memory it is bound to as
 * a whole, and how much memory that is. */
static uint64_t
plane_offsets(const VkImageCreateInfo *info, uint64_t offsets[3])
{
   uint64_t size = 0;
   for (uint32_t p = 0; p < vk_format_get_plane_count(info->format); ++p) {
      const VkImageCreateInfo pi = plane_create_info(info, p);
      VkMemoryRequirements2 req = { .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
      get_image_requirements(pi.format, pi.imageType, pi.tiling, pi.extent, pi.mipLevels, pi.arrayLayers,
                             pi.samples, &req);
      size = align64(size, req.memoryRequirements.alignment);
      if (offsets)
         offsets[p] = size;
      size += req.memoryRequirements.size;
   }
   return size;
}

static VkResult
create_planar_image(VkDevice _device, const VkImageCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                    VkImage *pImage)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   struct wvk_image *image = vk_image_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*image));
   if (!image)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   image->linear = pCreateInfo->tiling == VK_IMAGE_TILING_LINEAR;
   image->dimension = WGPUTextureDimension_2D;
   plane_offsets(pCreateInfo, image->plane_offset);
   for (uint32_t p = 0; p < vk_format_get_plane_count(pCreateInfo->format); ++p) {
      const VkImageCreateInfo pi = plane_create_info(pCreateInfo, p);
      VkImage plane;
      const VkResult result = wvk_CreateImage(_device, &pi, pAllocator, &plane);
      if (result != VK_SUCCESS) {
         wvk_DestroyImage(_device, wvk_image_to_handle(image), pAllocator);
         return result;
      }
      image->planes[p] = wvk_image_from_handle(plane);
      image->plane_count = p + 1;
   }
   *pImage = wvk_image_to_handle(image);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateImage(VkDevice _device, const VkImageCreateInfo *pCreateInfo,
                const VkAllocationCallbacks *pAllocator, VkImage *pImage)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);

   if (vk_format_get_plane_count(pCreateInfo->format) > 1)
      return wvk_format_caps(pdev, pCreateInfo->format) ? create_planar_image(_device, pCreateInfo, pAllocator, pImage)
                                                         : vk_error(dev, VK_ERROR_FORMAT_NOT_SUPPORTED);

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
   /* Until written, any texture of it is as good as another. */
   image->tracked = image->shadow != NULL;
   if (image->tracked)
      p_atomic_inc(&dev->tracked_images);
   for (uint32_t l = 0; l < ARRAY_SIZE(image->valid); ++l)
      image->valid[l] = ~0u;
   /* But of one made zero (zeroInitializeDeviceMemory): its texture's
    * blocks of zeros, which are no shadow's zeros until made into it. */
   if (pCreateInfo->initialLayout == VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT && image->shadow) {
      for (uint32_t l = 0; l < ARRAY_SIZE(image->valid); ++l)
         image->valid[l] = BITFIELD_BIT(WVK_IMAGE_TEXTURE);
   }

   /* Texels atomics can be done on, as a buffer too: of floats where the
    * device does atomics of them, or a view could be of integers. */
   const struct vk_features *features = &dev->vk.enabled_features;
   const bool float_atomics = features->shaderImageFloat32Atomics || features->shaderImageFloat32AtomicAdd ||
                              features->shaderImageFloat32AtomicMinMax ||
                              (pCreateInfo->flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT);
   if ((wusage & WGPUTextureUsage_StorageBinding) && wvk_format_has_atomics(pCreateInfo->format) &&
       (pCreateInfo->format != VK_FORMAT_R32_SFLOAT || float_atomics)) {
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
   for (uint32_t p = 0; p < image->plane_count; ++p)
      wvk_DestroyImage(_device, wvk_image_to_handle(image->planes[p]), pAllocator);
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
   for (uint32_t i = 0; i < image->alias_count; ++i)
      wgpuTextureRelease(image->aliases[i].texture);
   vk_image_destroy(&dev->vk, pAllocator, &image->vk);
}

WGPUTexture
wvk_image_snapshot(WGPUDevice device, const struct wvk_image *image, uint32_t t, WGPUCommandEncoder encoder)
{
   uint32_t first_level;
   WGPUTexture texture = wvk_image_texture(image, t, &first_level);
   if (!texture || image->vk.samples > 1 || image->layer_textures ||
       (image->vk.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)))
      return NULL;
   WGPUTextureFormat view_formats[1];
   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("VkImage copy");
   desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
   desc.dimension = image->dimension;
   desc.size = (WGPUExtent3D){ wgpuTextureGetWidth(texture), wgpuTextureGetHeight(texture),
                               wgpuTextureGetDepthOrArrayLayers(texture) };
   desc.format = wgpuTextureGetFormat(texture);
   desc.mipLevelCount = wgpuTextureGetMipLevelCount(texture);
   if ((image->vk.create_flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) || t != WVK_IMAGE_TEXTURE) {
      view_formats[0] = srgb_twin(desc.format);
      desc.viewFormatCount = view_formats[0] != WGPUTextureFormat_Undefined;
      desc.viewFormats = view_formats;
   }
   WGPUTexture copy = wgpuDeviceCreateTexture(device, &desc);
   if (!copy)
      return NULL;
   for (uint32_t l = 0; l < desc.mipLevelCount; ++l) {
      WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      src.texture = texture;
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
wvk_image_make_current(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image,
                       uint32_t t, uint32_t first, uint32_t count)
{
   const enum pipe_format pformat = vk_format_to_pipe_format(image->vk.format);
   const uint32_t bw = util_format_get_blockwidth(pformat), bh = util_format_get_blockheight(pformat);
   const uint32_t block_bytes = util_format_get_blocksize(pformat);
   for (uint32_t l = first; l < first + count && l < ARRAY_SIZE(image->valid); ++l) {
      if (image->valid[l] & BITFIELD_BIT(t))
         continue;
      /* The shadow is what the image's own texture decodes to. */
      if (t == WVK_IMAGE_SHADOW) {
         wvk_image_make_current(dev, replay, image, WVK_IMAGE_TEXTURE, l, 1);
         wvk_image_update_shadow(dev, replay, image, l);
         image->valid[l] |= BITFIELD_BIT(t);
         continue;
      }
      /* The others copy, through a buffer, what the one written last has:
       * texel for texel, or block for texel. */
      const uint32_t s = image->owner[l];
      uint32_t s_first, t_first;
      WGPUTexture from = wvk_image_texture(image, s, &s_first);
      WGPUTexture to = wvk_image_texture(image, t, &t_first);
      const uint32_t bx = DIV_ROUND_UP(u_minify(image->vk.extent.width, l), bw);
      const uint32_t by = image->vk.image_type == VK_IMAGE_TYPE_1D
                             ? 1 : DIV_ROUND_UP(u_minify(image->vk.extent.height, l), bh);
      const uint32_t bz = image->vk.image_type == VK_IMAGE_TYPE_3D ? u_minify(image->vk.extent.depth, l)
                                                                   : image->vk.array_layers;
      const uint32_t row = align(bx * block_bytes, 256);
      WGPUBuffer temp = wvk_replay_temp_buffer(dev, replay, (uint64_t)row * by * bz,
                                               WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst, NULL);
      if (!from || !to || !temp)
         continue;
      wvk_replay_end_passes(replay);
      WGPUTexelCopyBufferInfo buf = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
      buf.buffer = temp;
      buf.layout.bytesPerRow = row;
      buf.layout.rowsPerImage = by;
      WGPUTexelCopyTextureInfo a = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, b = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
      a.texture = from;
      a.mipLevel = l - s_first;
      b.texture = to;
      b.mipLevel = l - t_first;
      /* The image's own texture is in its format's texels. */
      const WGPUExtent3D own = { bx * bw, by * bh, bz }, blocks = { bx, by, bz };
      const WGPUExtent3D a_size = s == WVK_IMAGE_TEXTURE ? own : blocks;
      const WGPUExtent3D b_size = t == WVK_IMAGE_TEXTURE ? own : blocks;
      wgpuCommandEncoderCopyTextureToBuffer(replay->encoder, &a, &buf, &a_size);
      wgpuCommandEncoderCopyBufferToTexture(replay->encoder, &buf, &b, &b_size);
      image->valid[l] |= BITFIELD_BIT(t);
   }
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
          !(usage & ~(VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_HOST_TRANSFER_BIT));
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
   uint64_t size = linear ? (uint64_t)wvk_image_linear_pitch(format, extent.width) * extent.height
                          : image_size(format, type, extent, levels, layers, samples);
   /* A multi-planar image's planes, one after the other. */
   if (vk_format_get_plane_count(format) > 1) {
      const VkImageCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = type, .format = format, .extent = extent,
         .mipLevels = levels, .arrayLayers = layers, .samples = samples, .tiling = tiling,
      };
      size = plane_offsets(&info, NULL);
   }
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
   /* A plane of a disjoint image, as the plane is. */
   const VkImagePlaneMemoryRequirementsInfo *plane =
      vk_find_struct_const(pInfo->pNext, IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO);
   if (plane && image->plane_count)
      image = image->planes[wvk_image_aspect_plane(plane->planeAspect)];
   get_image_requirements(image->vk.format, image->vk.image_type, image->vk.tiling, image->vk.extent,
                          image->vk.mip_levels, image->vk.array_layers, image->vk.samples,
                          pMemoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetDeviceImageMemoryRequirements(VkDevice _device, const VkDeviceImageMemoryRequirements *pInfo,
                                     VkMemoryRequirements2 *pMemoryRequirements)
{
   const VkImageCreateInfo *ci = pInfo->pCreateInfo;
   if (vk_format_get_plane_count(ci->format) > 1 && (ci->flags & VK_IMAGE_CREATE_DISJOINT_BIT)) {
      const VkImageCreateInfo pi = plane_create_info(ci, wvk_image_aspect_plane(pInfo->planeAspect));
      get_image_requirements(pi.format, pi.imageType, pi.tiling, pi.extent, pi.mipLevels, pi.arrayLayers,
                             pi.samples, pMemoryRequirements);
      return;
   }
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
      /* A multi-planar image's planes: the one named, of a disjoint
       * image, or each where its part is. */
      if (image->plane_count) {
         const VkBindImagePlaneMemoryInfo *plane = vk_find_struct_const(pBindInfos[i].pNext, BIND_IMAGE_PLANE_MEMORY_INFO);
         for (uint32_t p = 0; p < image->plane_count && r == VK_SUCCESS; ++p) {
            if (plane && p != wvk_image_aspect_plane(plane->planeAspect))
               continue;
            const VkBindImageMemoryInfo one = {
               .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
               .image = wvk_image_to_handle(image->planes[p]),
               .memory = pBindInfos[i].memory,
               .memoryOffset = pBindInfos[i].memoryOffset + (plane ? 0 : image->plane_offset[p]),
            };
            r = wvk_BindImageMemory2(_device, 1, &one);
         }
         image->mem = mem;
         image->offset = pBindInfos[i].memoryOffset;
         const VkBindMemoryStatusKHR *status = vk_find_struct_const(pBindInfos[i].pNext, BIND_MEMORY_STATUS_KHR);
         if (status && status->pResult)
            *status->pResult = r;
         if (r != VK_SUCCESS)
            result = r;
         continue;
      }
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
   /* A host memcpy copy's memory (wvk_host_copy.c): the subresource's
    * rows, packed, as a buffer copy has them. */
   VkSubresourceHostMemcpySize *memcpy_size = vk_find_struct(pLayout->pNext, SUBRESOURCE_HOST_MEMCPY_SIZE);
   if (memcpy_size) {
      const VkImageSubresource *sub = &pSubresource->imageSubresource;
      const VkFormat format = vk_format_get_aspect_format(image->vk.format, sub->aspectMask);
      const struct util_format_description *desc = vk_format_description(format);
      const VkExtent3D e = wvk_image_aspect_extent(&image->vk, sub->aspectMask, sub->mipLevel);
      memcpy_size->size = (uint64_t)DIV_ROUND_UP(e.width, desc->block.width) *
                          DIV_ROUND_UP(e.height, desc->block.height) * DIV_ROUND_UP(e.depth, desc->block.depth) *
                          (desc->block.bits / 8);
   }
   if (!image->linear)
      return;
   /* A plane's, where it is in the image's memory. */
   uint64_t offset = 0;
   if (image->plane_count) {
      const uint32_t p = wvk_image_aspect_plane(pSubresource->imageSubresource.aspectMask);
      if (!(image->vk.create_flags & VK_IMAGE_CREATE_DISJOINT_BIT))
         offset = image->plane_offset[p];
      image = image->planes[p];
   }
   const uint64_t size = (uint64_t)image->pitch * image->vk.extent.height;
   pLayout->subresourceLayout = (VkSubresourceLayout){
      .offset = offset,
      .size = size,
      .rowPitch = image->pitch,
      .arrayPitch = size,
      .depthPitch = size,
   };
}

/* The layout an image of pInfo's would have (VK_KHR_maintenance5), which
 * is the one wvk_GetImageSubresourceLayout2KHR reports once it is made. */
VKAPI_ATTR void VKAPI_CALL
wvk_GetDeviceImageSubresourceLayoutKHR(VkDevice device, const VkDeviceImageSubresourceInfoKHR *pInfo,
                                       VkSubresourceLayout2KHR *pLayout)
{
   const VkImageCreateInfo *ci = pInfo->pCreateInfo;
   pLayout->subresourceLayout = (VkSubresourceLayout){ 0 };
   if (ci->tiling != VK_IMAGE_TILING_LINEAR)
      return;
   uint64_t offset = 0;
   VkImageCreateInfo plane = *ci;
   if (vk_format_get_plane_count(ci->format) > 1) {
      const uint32_t p = wvk_image_aspect_plane(pInfo->pSubresource->imageSubresource.aspectMask);
      uint64_t offsets[3];
      plane_offsets(ci, offsets);
      if (!(ci->flags & VK_IMAGE_CREATE_DISJOINT_BIT))
         offset = offsets[p];
      plane = plane_create_info(ci, p);
      ci = &plane;
   }
   const uint32_t pitch = wvk_image_linear_pitch(ci->format, ci->extent.width);
   const uint64_t size = (uint64_t)pitch * ci->extent.height;
   pLayout->subresourceLayout = (VkSubresourceLayout){
      .offset = offset,
      .size = size,
      .rowPitch = pitch,
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

/* The alias of the image a view of `format` is of, made if it is the
 * first: its texture (wvk_image_texture), or the image's own if there is
 * none to be had. */
static uint32_t
image_alias(struct wvk_device *dev, struct wvk_image *image, const struct wvk_image_view *view,
            WGPUTextureFormat format)
{
   const struct wvk_physical_device *pdev = wvk_device_physical(dev);
   const VkFormat vf = view->vk.view_format;
   const enum pipe_format pformat = vk_format_to_pipe_format(image->vk.format);
   /* A texel a block of a compressed image, at the view's one level. */
   const bool block = vk_format_is_compressed(image->vk.format) && !vk_format_is_compressed(vf);
   const uint32_t first = block ? view->vk.base_mip_level : 0;
   const WGPUTextureFormat twin = srgb_twin(format);
   uint32_t t = WVK_IMAGE_TEXTURE;
   simple_mtx_lock(&dev->cache_lock);
   for (uint32_t i = 0; i < image->alias_count; ++i) {
      if ((image->aliases[i].format == format || (twin && image->aliases[i].format == twin)) &&
          image->aliases[i].first_level == first) {
         t = i + 1;
         goto out;
      }
   }
   if (image->alias_count == ARRAY_SIZE(image->aliases))
      goto out;

   const uint32_t caps = wvk_format_caps(pdev, vf);
   WGPUTextureUsage usage = WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
   if (caps & WVK_CAP_SAMPLE)
      usage |= WGPUTextureUsage_TextureBinding;
   if (caps & WVK_CAP_RENDER)
      usage |= WGPUTextureUsage_RenderAttachment;
   if ((image->vk.usage & VK_IMAGE_USAGE_STORAGE_BIT) && (caps & WVK_CAP_STORAGE))
      usage |= WGPUTextureUsage_StorageBinding;
   WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
   desc.label = wvk_sv("VkImage alias");
   desc.usage = usage;
   desc.dimension = image->dimension;
   desc.format = format;
   desc.viewFormatCount = twin != WGPUTextureFormat_Undefined;
   desc.viewFormats = &twin;
   if (block) {
      desc.size.width = DIV_ROUND_UP(u_minify(image->vk.extent.width, first), util_format_get_blockwidth(pformat));
      desc.size.height = image->vk.image_type == VK_IMAGE_TYPE_1D ? 1
         : DIV_ROUND_UP(u_minify(image->vk.extent.height, first), util_format_get_blockheight(pformat));
      desc.size.depthOrArrayLayers = image->vk.image_type == VK_IMAGE_TYPE_3D
                                        ? u_minify(image->vk.extent.depth, first) : image->vk.array_layers;
      desc.mipLevelCount = 1;
   } else {
      desc.size.width = image->vk.extent.width;
      desc.size.height = image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : image->vk.extent.height;
      desc.size.depthOrArrayLayers = image->vk.image_type == VK_IMAGE_TYPE_3D ? image->vk.extent.depth
                                                                              : image->vk.array_layers;
      desc.mipLevelCount = image->vk.mip_levels;
   }
   WGPUTexture texture = wgpuDeviceCreateTexture(dev->wgpu, &desc);
   if (!texture)
      goto out;
   struct wvk_image_alias *alias = &image->aliases[image->alias_count];
   alias->format = format;
   alias->texture = texture;
   alias->first_level = first;
   alias->level_count = desc.mipLevelCount;
   if (!image->tracked)
      p_atomic_inc(&dev->tracked_images);
   image->tracked = true;
   t = ++image->alias_count;
out:
   simple_mtx_unlock(&dev->cache_lock);
   return t;
}

/* Whether a view of `format` needs a texture of the image's other than
 * its own, whose format and sRGB twin are all WebGPU views it as. */
static bool
needs_alias(const struct wvk_image *image, const struct wvk_image_view *view, WGPUTextureFormat format)
{
   if (!image->texture || image->layer_textures || image->vk.samples > 1 || view->aspect != WGPUTextureAspect_All ||
       (image->vk.aspects & ~VK_IMAGE_ASPECT_COLOR_BIT) || format == image->format ||
       format == srgb_twin(image->format))
      return false;
   /* Packed formats' texels are RGBA8, whose bytes are not theirs; and a
    * compressed view is its own format or its twin. */
   const struct wvk_format *a = wvk_format_get(image->vk.format), *b = wvk_format_get(view->vk.view_format);
   return a && b && !(a->caps & WVK_CAP_PACKED16) && !(b->caps & WVK_CAP_PACKED16) &&
          !vk_format_is_compressed(view->vk.view_format) &&
          util_format_get_blocksize(vk_format_to_pipe_format(image->vk.format)) ==
             util_format_get_blocksize(vk_format_to_pipe_format(view->vk.view_format));
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateImageView(VkDevice _device, const VkImageViewCreateInfo *pCreateInfo,
                    const VkAllocationCallbacks *pAllocator, VkImageView *pView)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_image, image, pCreateInfo->image);

   /* A plane of a multi-planar image: a view of the plane's image. */
   const VkImageAspectFlags planes = VK_IMAGE_ASPECT_PLANE_0_BIT | VK_IMAGE_ASPECT_PLANE_1_BIT |
                                     VK_IMAGE_ASPECT_PLANE_2_BIT;
   if (image->plane_count && (pCreateInfo->subresourceRange.aspectMask & planes)) {
      VkImageViewCreateInfo plane = *pCreateInfo;
      plane.subresourceRange.aspectMask = pCreateInfo->subresourceRange.aspectMask;
      plane.image = wvk_image_to_handle(wvk_image_for_aspect(image, &plane.subresourceRange.aspectMask));
      return wvk_CreateImageView(_device, &plane, pAllocator, pView);
   }

   struct wvk_image_view *view = vk_image_view_create(&dev->vk, pCreateInfo, pAllocator, sizeof(*view));
   if (!view)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* All of one, for a YCbCr conversion to sample: a view of each plane,
    * the first's this one's own. */
   if (image->plane_count) {
      const struct vk_format_ycbcr_info *ycbcr = vk_format_get_ycbcr_info(view->vk.format);
      for (uint32_t p = 0; ycbcr && p < MIN2(ycbcr->n_planes, image->plane_count); ++p) {
         const VkImageViewCreateInfo info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = wvk_image_to_handle(image->planes[p]),
            .viewType = pCreateInfo->viewType,
            .format = ycbcr->planes[p].format,
            .subresourceRange = {
               .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
               .baseMipLevel = view->vk.base_mip_level,
               .levelCount = view->vk.level_count,
               .baseArrayLayer = view->vk.base_array_layer,
               .layerCount = view->vk.layer_count,
            },
         };
         VkImageView plane;
         if (wvk_CreateImageView(_device, &info, pAllocator, &plane) != VK_SUCCESS) {
            wvk_DestroyImageView(_device, wvk_image_view_to_handle(view), pAllocator);
            return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
         }
         view->planes[p] = wvk_image_view_from_handle(plane);
         view->plane_count = p + 1;
      }
      if (view->plane_count) {
         view->view = view->planes[0]->view;
         view->view_desc = view->planes[0]->view_desc;
         view->format = view->planes[0]->format;
      }
      *pView = wvk_image_view_to_handle(view);
      return VK_SUCCESS;
   }

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
   /* Depth or stencil is (x, 0, 0, 1) to Vulkan; WebGPU leaves what else
    * a combined format's aspect reads as the backend has it. */
   if (view->vk.aspects & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
      VkComponentSwizzle *c[4] = { &mapping.r, &mapping.g, &mapping.b, &mapping.a };
      for (unsigned i = 0; i < 4; ++i) {
         if (*c[i] == VK_COMPONENT_SWIZZLE_G || *c[i] == VK_COMPONENT_SWIZZLE_B)
            *c[i] = VK_COMPONENT_SWIZZLE_ZERO;
         else if (*c[i] == VK_COMPONENT_SWIZZLE_A)
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

   /* A compressed image of a size not whole blocks is sampled from what
    * it decodes to; a view of a format WebGPU cannot view the texture as
    * is of a texture of that format. */
   if (needs_alias(image, view, view->format))
      view->texture = image_alias(dev, image, view, view->format);
   else if (image->shadow && vk_format_is_compressed(view->vk.view_format))
      view->texture = WVK_IMAGE_SHADOW;

   /* Views of an image without a texture -- a swapchain's, until acquired
    * -- are made when used. */
   if (image->texture && ((image->usage & WGPUTextureUsage_TextureBinding) || view->texture != WVK_IMAGE_TEXTURE)) {
      WGPUTextureViewDescriptor desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
      /* One aspect of a depth-stencil texture is viewed as that aspect's
       * own format, which WebGPU works out itself. */
      desc.format = view->aspect == WGPUTextureAspect_All ? view->format : WGPUTextureFormat_Undefined;
      desc.dimension = view_dimension(view->vk.view_type);
      /* A 2D view of a 3D image's slices (VK_EXT_image_2d_view_of_3d) is of
       * the level whole, as WebGPU views a 3D texture: shaders take the
       * slice, which the view's records say (storage.z_slice_offset). */
      if (image->vk.image_type == VK_IMAGE_TYPE_3D)
         desc.dimension = WGPUTextureViewDimension_3D;
      /* A multisampled view is 2D in WebGPU, whatever array it says. */
      if (image->vk.samples > 1)
         desc.dimension = WGPUTextureViewDimension_2D;
      desc.baseMipLevel = view->vk.base_mip_level;
      desc.mipLevelCount = view->vk.level_count;
      uint32_t texture_layer;
      WGPUTexture texture = wvk_image_layer_texture(image, view->vk.base_array_layer, &texture_layer);
      if (view->texture != WVK_IMAGE_TEXTURE && view->texture != WVK_IMAGE_SHADOW) {
         uint32_t first_level;
         texture = wvk_image_texture(image, view->texture, &first_level);
         desc.baseMipLevel -= first_level;
      } else if (view->texture == WVK_IMAGE_SHADOW) {
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
      if ((view->vk.usage & VK_IMAGE_USAGE_STORAGE_BIT) && desc.mipLevelCount > 1) {
         view->storage_desc = view->array_view ? view->array_desc : view->view_desc;
         view->storage_desc.mipLevelCount = 1;
         view->storage_view = wgpuTextureCreateView(texture, &view->storage_desc);
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
   /* The planes' views, whose first's `view` this one has. */
   if (view->plane_count) {
      for (uint32_t p = 0; p < view->plane_count; ++p)
         wvk_DestroyImageView(_device, wvk_image_view_to_handle(view->planes[p]), pAllocator);
      view->view = NULL;
   }
   if (view->view)
      wgpuTextureViewRelease(view->view);
   if (view->array_view)
      wgpuTextureViewRelease(view->array_view);
   if (view->storage_view)
      wgpuTextureViewRelease(view->storage_view);
   vk_image_view_destroy(&dev->vk, pAllocator, &view->vk);
}

WGPUTextureView
wvk_image_attachment_view(struct wvk_image *image, uint32_t t, WGPUTextureFormat format, WGPUTextureAspect aspect,
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
   if (t != WVK_IMAGE_TEXTURE) {
      uint32_t first_level;
      texture = wvk_image_texture(image, t, &first_level);
      desc.baseMipLevel -= first_level;
   }
   desc.baseArrayLayer = image->dimension == WGPUTextureDimension_3D ? 0 : texture_layer;
   desc.arrayLayerCount = 1;
   desc.aspect = aspect;
   return wgpuTextureCreateView(texture, &desc);
}
