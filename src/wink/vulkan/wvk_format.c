/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * VkFormat to WebGPU. WebGPU has no query for what a format can do, so the
 * capabilities are Dawn's table at the pinned revision
 * (src/dawn/native/Format.cpp, non-compatibility mode), and optional
 * features extend them in wvk_format_caps.
 *
 * A packed Vulkan format and a WebGPU format are the same when their bytes
 * are: A8B8G8R8_*_PACK32 is R8G8B8A8, A2B10G10R10_UNORM_PACK32 is
 * rgb10a2unorm, B10G11R11_UFLOAT_PACK32 is rg11b10ufloat.
 */
#include "wvk_format.h"

#include "wvk_descriptor_set_layout.h"
#include "wvk_physical_device.h"

#include "vk_format.h"
#include "util/format/u_format.h"

#define TF(x) WGPUTextureFormat_##x
#define VF(x) WGPUVertexFormat_##x

#define SAMPLE_F   (WVK_CAP_SAMPLE | WVK_CAP_FILTER | WVK_CAP_COPY_SRC | WVK_CAP_COPY_DST)
#define SAMPLE_UF  (WVK_CAP_SAMPLE | WVK_CAP_COPY_SRC | WVK_CAP_COPY_DST)   /* unfilterable float */
#define SAMPLE_I   (WVK_CAP_SAMPLE | WVK_CAP_INT | WVK_CAP_COPY_SRC | WVK_CAP_COPY_DST)
#define RT         WVK_CAP_RENDER
#define RT_FULL    (WVK_CAP_RENDER | WVK_CAP_BLEND | WVK_CAP_MSAA | WVK_CAP_RESOLVE)
#define MS         WVK_CAP_MSAA
#define ST         WVK_CAP_STORAGE
#define ST_RW      (WVK_CAP_STORAGE | WVK_CAP_STORAGE_RW)
#define COMP       (WVK_CAP_SAMPLE | WVK_CAP_FILTER | WVK_CAP_COPY_SRC | WVK_CAP_COPY_DST | WVK_CAP_COMPRESSED)

#define FMT(vk, tex, vtx, caps_) \
   [VK_FORMAT_##vk] = { .texture = TF(tex), .vertex = VF(vtx), .caps = (caps_) }
#define TEX(vk, tex, caps_) \
   [VK_FORMAT_##vk] = { .texture = TF(tex), .caps = (caps_) }
#define VTX(vk, vtx) \
   [VK_FORMAT_##vk] = { .texture = WGPUTextureFormat_Undefined, .vertex = VF(vtx), .caps = 0 }
#define FEAT(vk, tex, feat, caps_) \
   [VK_FORMAT_##vk] = { .texture = TF(tex), .feature = WGPUFeatureName_##feat, .caps = (caps_) }

/* Indexed by VkFormat; the core formats end at VK_FORMAT_ASTC_12x12_SRGB_BLOCK. */
static const struct wvk_format formats[VK_FORMAT_ASTC_12x12_SRGB_BLOCK + 1] = {
   /* 8-bit */
   FMT(R8_UNORM, R8Unorm, Unorm8, SAMPLE_F | RT_FULL),
   FMT(R8_SNORM, R8Snorm, Snorm8, SAMPLE_F),
   FMT(R8_UINT, R8Uint, Uint8, SAMPLE_I | RT | MS),
   FMT(R8_SINT, R8Sint, Sint8, SAMPLE_I | RT | MS),
   FMT(R8G8_UNORM, RG8Unorm, Unorm8x2, SAMPLE_F | RT_FULL),
   FMT(R8G8_SNORM, RG8Snorm, Snorm8x2, SAMPLE_F),
   FMT(R8G8_UINT, RG8Uint, Uint8x2, SAMPLE_I | RT | MS),
   FMT(R8G8_SINT, RG8Sint, Sint8x2, SAMPLE_I | RT | MS),
   FMT(R8G8B8A8_UNORM, RGBA8Unorm, Unorm8x4, SAMPLE_F | RT_FULL | ST),
   /* 16-bit packed: RGBA8, converted in copies. */
   TEX(R4G4B4A4_UNORM_PACK16, RGBA8Unorm, SAMPLE_F | RT_FULL | WVK_CAP_PACKED16),
   TEX(B4G4R4A4_UNORM_PACK16, RGBA8Unorm, SAMPLE_F | RT_FULL | WVK_CAP_PACKED16),
   TEX(R5G6B5_UNORM_PACK16, RGBA8Unorm, SAMPLE_F | RT_FULL | WVK_CAP_PACKED16),
   TEX(B5G6R5_UNORM_PACK16, RGBA8Unorm, SAMPLE_F | RT_FULL | WVK_CAP_PACKED16),
   TEX(R5G5B5A1_UNORM_PACK16, RGBA8Unorm, SAMPLE_F | RT_FULL | WVK_CAP_PACKED16),
   TEX(B5G5R5A1_UNORM_PACK16, RGBA8Unorm, SAMPLE_F | RT_FULL | WVK_CAP_PACKED16),
   TEX(A1R5G5B5_UNORM_PACK16, RGBA8Unorm, SAMPLE_F | RT_FULL | WVK_CAP_PACKED16),
   TEX(R8G8B8A8_SRGB, RGBA8UnormSrgb, SAMPLE_F | RT_FULL),
   FMT(R8G8B8A8_SNORM, RGBA8Snorm, Snorm8x4, SAMPLE_F | ST),
   FMT(R8G8B8A8_UINT, RGBA8Uint, Uint8x4, SAMPLE_I | RT | MS | ST),
   FMT(R8G8B8A8_SINT, RGBA8Sint, Sint8x4, SAMPLE_I | RT | MS | ST),
   FMT(A8B8G8R8_UNORM_PACK32, RGBA8Unorm, Unorm8x4, SAMPLE_F | RT_FULL | ST),
   TEX(A8B8G8R8_SRGB_PACK32, RGBA8UnormSrgb, SAMPLE_F | RT_FULL),
   FMT(A8B8G8R8_SNORM_PACK32, RGBA8Snorm, Snorm8x4, SAMPLE_F | ST),
   FMT(A8B8G8R8_UINT_PACK32, RGBA8Uint, Uint8x4, SAMPLE_I | RT | MS | ST),
   FMT(A8B8G8R8_SINT_PACK32, RGBA8Sint, Sint8x4, SAMPLE_I | RT | MS | ST),
   FMT(B8G8R8A8_UNORM, BGRA8Unorm, Unorm8x4BGRA, SAMPLE_F | RT_FULL),
   TEX(B8G8R8A8_SRGB, BGRA8UnormSrgb, SAMPLE_F | RT_FULL),

   /* 10-bit and packed float */
   FMT(A2B10G10R10_UNORM_PACK32, RGB10A2Unorm, Unorm10_10_10_2, SAMPLE_F | RT_FULL),
   TEX(A2B10G10R10_UINT_PACK32, RGB10A2Uint, SAMPLE_I | RT | MS),
   TEX(B10G11R11_UFLOAT_PACK32, RG11B10Ufloat, SAMPLE_F),
   TEX(E5B9G9R9_UFLOAT_PACK32, RGB9E5Ufloat, SAMPLE_F),

   /* 16-bit */
   FMT(R16_UINT, R16Uint, Uint16, SAMPLE_I | RT | MS),
   FMT(R16_SINT, R16Sint, Sint16, SAMPLE_I | RT | MS),
   FMT(R16_SFLOAT, R16Float, Float16, SAMPLE_F | RT_FULL),
   VTX(R16_UNORM, Unorm16),
   VTX(R16_SNORM, Snorm16),
   FMT(R16G16_UINT, RG16Uint, Uint16x2, SAMPLE_I | RT | MS),
   FMT(R16G16_SINT, RG16Sint, Sint16x2, SAMPLE_I | RT | MS),
   FMT(R16G16_SFLOAT, RG16Float, Float16x2, SAMPLE_F | RT_FULL),
   VTX(R16G16_UNORM, Unorm16x2),
   VTX(R16G16_SNORM, Snorm16x2),
   FMT(R16G16B16A16_UINT, RGBA16Uint, Uint16x4, SAMPLE_I | RT | MS | ST),
   FMT(R16G16B16A16_SINT, RGBA16Sint, Sint16x4, SAMPLE_I | RT | MS | ST),
   FMT(R16G16B16A16_SFLOAT, RGBA16Float, Float16x4, SAMPLE_F | RT_FULL | ST),
   VTX(R16G16B16A16_UNORM, Unorm16x4),
   VTX(R16G16B16A16_SNORM, Snorm16x4),

   /* 32-bit: floats are unfilterable and unblendable without features */
   FMT(R32_UINT, R32Uint, Uint32, SAMPLE_I | RT | ST_RW),
   FMT(R32_SINT, R32Sint, Sint32, SAMPLE_I | RT | ST_RW),
   FMT(R32_SFLOAT, R32Float, Float32, SAMPLE_UF | RT | MS | ST_RW),
   FMT(R32G32_UINT, RG32Uint, Uint32x2, SAMPLE_I | RT | ST),
   FMT(R32G32_SINT, RG32Sint, Sint32x2, SAMPLE_I | RT | ST),
   FMT(R32G32_SFLOAT, RG32Float, Float32x2, SAMPLE_UF | RT | ST),
   VTX(R32G32B32_UINT, Uint32x3),
   VTX(R32G32B32_SINT, Sint32x3),
   VTX(R32G32B32_SFLOAT, Float32x3),
   FMT(R32G32B32A32_UINT, RGBA32Uint, Uint32x4, SAMPLE_I | RT | ST),
   FMT(R32G32B32A32_SINT, RGBA32Sint, Sint32x4, SAMPLE_I | RT | ST),
   FMT(R32G32B32A32_SFLOAT, RGBA32Float, Float32x4, SAMPLE_UF | RT | ST),

   /* Depth and stencil. Depth24Plus has no copies at all in WebGPU, and
    * Depth32Float copies only out; wvk_cmd_copy works around both. */
   TEX(D16_UNORM, Depth16Unorm, WVK_CAP_SAMPLE | WVK_CAP_DEPTH | MS | WVK_CAP_COPY_SRC | WVK_CAP_COPY_DST),
   TEX(X8_D24_UNORM_PACK32, Depth24Plus, WVK_CAP_SAMPLE | WVK_CAP_DEPTH | MS),
   TEX(D32_SFLOAT, Depth32Float, WVK_CAP_SAMPLE | WVK_CAP_DEPTH | MS | WVK_CAP_COPY_SRC),
   /* Stencil alone with a depth aspect nothing uses: a fragment shader may
    * write depth with no depth attachment, and WebGPU refuses one that does
    * to Stencil8. */
   TEX(S8_UINT, Depth24PlusStencil8, WVK_CAP_SAMPLE | WVK_CAP_INT | WVK_CAP_STENCIL | MS | WVK_CAP_COPY_SRC | WVK_CAP_COPY_DST),
   TEX(D24_UNORM_S8_UINT, Depth24PlusStencil8, WVK_CAP_SAMPLE | WVK_CAP_DEPTH | WVK_CAP_STENCIL | MS),
   FEAT(D32_SFLOAT_S8_UINT, Depth32FloatStencil8, Depth32FloatStencil8,
        WVK_CAP_SAMPLE | WVK_CAP_DEPTH | WVK_CAP_STENCIL | MS | WVK_CAP_COPY_SRC),

   /* BC. Vulkan's RGB variants decode punch-through texels with alpha one;
    * WebGPU has only the RGBA ones, and the view makes up the difference. */
   FEAT(BC1_RGB_UNORM_BLOCK, BC1RGBAUnorm, TextureCompressionBC, COMP),
   FEAT(BC1_RGB_SRGB_BLOCK, BC1RGBAUnormSrgb, TextureCompressionBC, COMP),
   FEAT(BC1_RGBA_UNORM_BLOCK, BC1RGBAUnorm, TextureCompressionBC, COMP),
   FEAT(BC1_RGBA_SRGB_BLOCK, BC1RGBAUnormSrgb, TextureCompressionBC, COMP),
   FEAT(BC2_UNORM_BLOCK, BC2RGBAUnorm, TextureCompressionBC, COMP),
   FEAT(BC2_SRGB_BLOCK, BC2RGBAUnormSrgb, TextureCompressionBC, COMP),
   FEAT(BC3_UNORM_BLOCK, BC3RGBAUnorm, TextureCompressionBC, COMP),
   FEAT(BC3_SRGB_BLOCK, BC3RGBAUnormSrgb, TextureCompressionBC, COMP),
   FEAT(BC4_UNORM_BLOCK, BC4RUnorm, TextureCompressionBC, COMP),
   FEAT(BC4_SNORM_BLOCK, BC4RSnorm, TextureCompressionBC, COMP),
   FEAT(BC5_UNORM_BLOCK, BC5RGUnorm, TextureCompressionBC, COMP),
   FEAT(BC5_SNORM_BLOCK, BC5RGSnorm, TextureCompressionBC, COMP),
   FEAT(BC6H_UFLOAT_BLOCK, BC6HRGBUfloat, TextureCompressionBC, COMP),
   FEAT(BC6H_SFLOAT_BLOCK, BC6HRGBFloat, TextureCompressionBC, COMP),
   FEAT(BC7_UNORM_BLOCK, BC7RGBAUnorm, TextureCompressionBC, COMP),
   FEAT(BC7_SRGB_BLOCK, BC7RGBAUnormSrgb, TextureCompressionBC, COMP),

   /* ETC2 and EAC */
   FEAT(ETC2_R8G8B8_UNORM_BLOCK, ETC2RGB8Unorm, TextureCompressionETC2, COMP),
   FEAT(ETC2_R8G8B8_SRGB_BLOCK, ETC2RGB8UnormSrgb, TextureCompressionETC2, COMP),
   FEAT(ETC2_R8G8B8A1_UNORM_BLOCK, ETC2RGB8A1Unorm, TextureCompressionETC2, COMP),
   FEAT(ETC2_R8G8B8A1_SRGB_BLOCK, ETC2RGB8A1UnormSrgb, TextureCompressionETC2, COMP),
   FEAT(ETC2_R8G8B8A8_UNORM_BLOCK, ETC2RGBA8Unorm, TextureCompressionETC2, COMP),
   FEAT(ETC2_R8G8B8A8_SRGB_BLOCK, ETC2RGBA8UnormSrgb, TextureCompressionETC2, COMP),
   FEAT(EAC_R11_UNORM_BLOCK, EACR11Unorm, TextureCompressionETC2, COMP),
   FEAT(EAC_R11_SNORM_BLOCK, EACR11Snorm, TextureCompressionETC2, COMP),
   FEAT(EAC_R11G11_UNORM_BLOCK, EACRG11Unorm, TextureCompressionETC2, COMP),
   FEAT(EAC_R11G11_SNORM_BLOCK, EACRG11Snorm, TextureCompressionETC2, COMP),

   /* ASTC */
#define ASTC(w, h)                                                                  \
   FEAT(ASTC_##w##x##h##_UNORM_BLOCK, ASTC##w##x##h##Unorm, TextureCompressionASTC, COMP), \
   FEAT(ASTC_##w##x##h##_SRGB_BLOCK, ASTC##w##x##h##UnormSrgb, TextureCompressionASTC, COMP)
   ASTC(4, 4), ASTC(5, 4), ASTC(5, 5), ASTC(6, 5), ASTC(6, 6), ASTC(8, 5), ASTC(8, 6),
   ASTC(8, 8), ASTC(10, 5), ASTC(10, 6), ASTC(10, 8), ASTC(10, 10), ASTC(12, 10), ASTC(12, 12),
#undef ASTC
};

/* Extensions' formats, past the core ones: packed ones, as the core's are
 * (VK_EXT_4444_formats, VK_KHR_maintenance5). */
static const struct {
   VkFormat vk;
   struct wvk_format format;
} extension_formats[] = {
   { VK_FORMAT_A4R4G4B4_UNORM_PACK16,
     { .texture = TF(RGBA8Unorm), .caps = SAMPLE_F | RT_FULL | WVK_CAP_PACKED16 } },
   { VK_FORMAT_A4B4G4R4_UNORM_PACK16,
     { .texture = TF(RGBA8Unorm), .caps = SAMPLE_F | RT_FULL | WVK_CAP_PACKED16 } },
   { VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR,
     { .texture = TF(RGBA8Unorm), .caps = SAMPLE_F | RT_FULL | WVK_CAP_PACKED16 } },
};

const struct wvk_format *
wvk_format_get(VkFormat format)
{
   if ((unsigned)format >= ARRAY_SIZE(formats)) {
      for (unsigned i = 0; i < ARRAY_SIZE(extension_formats); ++i) {
         if (extension_formats[i].vk == format)
            return &extension_formats[i].format;
      }
      return NULL;
   }
   const struct wvk_format *f = &formats[format];
   if (f->texture == WGPUTextureFormat_Undefined && !f->vertex)
      return NULL;
   return f;
}

/* Whether a vertex shader can read an attribute of `format` itself, where
 * WebGPU has no vertex format of it (fetch_channel in wvk_shader.c): plain
 * channels of integers, normalized or scaled, of up to 32 bits and none
 * across two words, or of 16- and 32-bit floats. */
bool
wvk_format_vertex_pullable(VkFormat format)
{
   const struct util_format_description *d = vk_format_description(format);
   if (!d || d->layout != UTIL_FORMAT_LAYOUT_PLAIN || d->colorspace != UTIL_FORMAT_COLORSPACE_RGB ||
       d->block.bits > 128)
      return false;
   for (unsigned c = 0; c < d->nr_channels; ++c) {
      const struct util_format_channel_description *ch = &d->channel[c];
      if (ch->type == UTIL_FORMAT_TYPE_VOID)
         continue;
      if (ch->size == 0 || ch->size > 32 || (ch->shift % 32) + ch->size > 32 ||
          (ch->type != UTIL_FORMAT_TYPE_UNSIGNED && ch->type != UTIL_FORMAT_TYPE_SIGNED &&
           ch->type != UTIL_FORMAT_TYPE_FLOAT) ||
          (ch->type == UTIL_FORMAT_TYPE_FLOAT && ch->size != 16 && ch->size != 32))
         return false;
   }
   return true;
}

/* Whether WebGPU has a vertex format of it. */
bool
wvk_format_vertex_native(VkFormat format)
{
   const struct wvk_format *f = wvk_format_get(format);
   return f && f->vertex;
}

uint32_t
wvk_format_caps(const struct wvk_physical_device *pdev, VkFormat format)
{
   const struct wvk_format *f = wvk_format_get(format);
   if (!f || f->texture == WGPUTextureFormat_Undefined)
      return 0;
   if (f->feature && !wvk_physical_device_has_feature(pdev, f->feature))
      return 0;

   uint32_t caps = f->caps;
   switch (f->texture) {
   case WGPUTextureFormat_R32Float:
   case WGPUTextureFormat_RG32Float:
   case WGPUTextureFormat_RGBA32Float:
      if (wvk_physical_device_has_feature(pdev, WGPUFeatureName_Float32Filterable))
         caps |= WVK_CAP_FILTER;
      if (wvk_physical_device_has_feature(pdev, WGPUFeatureName_Float32Blendable))
         caps |= WVK_CAP_BLEND;
      break;
   case WGPUTextureFormat_RG11B10Ufloat:
      if (wvk_physical_device_has_feature(pdev, WGPUFeatureName_RG11B10UfloatRenderable))
         caps |= RT_FULL;
      break;
   case WGPUTextureFormat_BGRA8Unorm:
      if (wvk_physical_device_has_feature(pdev, WGPUFeatureName_BGRA8UnormStorage))
         caps |= ST;
      break;
   default:
      break;
   }
   return caps;
}

bool
wvk_texel_buffer_format(VkFormat format, uint32_t *record)
{
   const enum pipe_format pf = vk_format_to_pipe_format(format);
   const struct util_format_description *desc = util_format_description(pf);
   if (!desc || vk_format_is_depth_or_stencil(format))
      return false;
   const uint32_t bytes = desc->block.bits / 8;
   uint32_t type, sizes = 0, shifts = 0;
   if (pf == PIPE_FORMAT_R11G11B10_FLOAT) {
      type = WVK_TEXEL_UFLOAT_11_11_10;
      sizes = 32;
   } else {
      if (desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || desc->colorspace != UTIL_FORMAT_COLORSPACE_RGB ||
          desc->block.width != 1 || desc->block.height != 1 || bytes > 16 || !desc->nr_channels)
         return false;
      const struct util_format_channel_description *c0 = &desc->channel[0];
      if (c0->type == UTIL_FORMAT_TYPE_FLOAT)
         type = WVK_TEXEL_FLOAT;
      else if (c0->type == UTIL_FORMAT_TYPE_UNSIGNED && c0->normalized)
         type = WVK_TEXEL_UNORM;
      else if (c0->type == UTIL_FORMAT_TYPE_SIGNED && c0->normalized)
         type = WVK_TEXEL_SNORM;
      else if (c0->type == UTIL_FORMAT_TYPE_UNSIGNED && c0->pure_integer)
         type = WVK_TEXEL_UINT;
      else if (c0->type == UTIL_FORMAT_TYPE_SIGNED && c0->pure_integer)
         type = WVK_TEXEL_SINT;
      else
         return false;
      for (uint32_t c = 0; c < desc->nr_channels; ++c) {
         const struct util_format_channel_description *ch = &desc->channel[c];
         /* One type for every channel, and each of them whole in a word. */
         if (ch->type != c0->type || ch->normalized != c0->normalized || ch->pure_integer != c0->pure_integer ||
             ch->size > 32 || (type == WVK_TEXEL_FLOAT && ch->size != 16 && ch->size != 32) ||
             ch->shift / 32 != (ch->shift + ch->size - 1) / 32)
            return false;
         sizes |= (uint32_t)ch->size << (8 * c);
         shifts |= (uint32_t)ch->shift << (8 * c);
      }
   }
   uint32_t swizzle = 0;
   for (uint32_t i = 0; i < 4; ++i) {
      const unsigned sw = desc->swizzle[i];
      const uint32_t v = sw <= PIPE_SWIZZLE_W ? sw - PIPE_SWIZZLE_X : sw == PIPE_SWIZZLE_1 ? 5 : 4;
      swizzle |= v << (4 * i);
   }
   record[WVK_META_TEXEL_FORMAT] = type | bytes << 8;
   record[WVK_META_TEXEL_SIZES] = sizes;
   record[WVK_META_TEXEL_SHIFTS] = shifts;
   record[WVK_META_TEXEL_SWIZZLE] = swizzle;
   return true;
}
