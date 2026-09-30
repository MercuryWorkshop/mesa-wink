/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_FORMAT_H
#define WVK_FORMAT_H

#include "wvk_private.h"

struct wvk_physical_device;

/* What WebGPU can do with a texture format: Dawn's own table at the pinned
 * revision (src/dawn/native/Format.cpp), before optional features. */
enum wvk_format_cap {
   WVK_CAP_SAMPLE       = 1u << 0,
   WVK_CAP_FILTER       = 1u << 1,  /* float sample type, not unfilterable */
   WVK_CAP_RENDER       = 1u << 2,
   WVK_CAP_BLEND        = 1u << 3,
   WVK_CAP_MSAA         = 1u << 4,
   WVK_CAP_RESOLVE      = 1u << 5,
   WVK_CAP_STORAGE      = 1u << 6,  /* write-only and read-only storage */
   WVK_CAP_STORAGE_RW   = 1u << 7,
   WVK_CAP_DEPTH        = 1u << 8,
   WVK_CAP_STENCIL      = 1u << 9,
   WVK_CAP_COPY_SRC     = 1u << 10, /* texture to buffer or texture */
   WVK_CAP_COPY_DST     = 1u << 11, /* buffer or texture to texture */
   WVK_CAP_COMPRESSED   = 1u << 12,
   WVK_CAP_INT          = 1u << 13, /* uint or sint sample type */
   /* A 16-bit packed format WebGPU does not have: an RGBA8 texture, whose
    * copies to and from buffers convert (wvk_cmd_copy.c). */
   WVK_CAP_PACKED16     = 1u << 14,
   /* A multi-planar format: an image of it is one of each plane's format
    * (wvk_image.h), sampled through a YCbCr conversion. */
   WVK_CAP_PLANAR       = 1u << 15,
   /* A padded 16-bit format (R10X6 and the like): a 16-bit normalized
    * texture, of more bits than the format's own. */
   WVK_CAP_PADDED       = 1u << 16,
};

struct wvk_format {
   WGPUTextureFormat texture;  /* Undefined: none */
   WGPUVertexFormat vertex;    /* 0: none; WebGPU's formats start at 1 */
   /* An optional WebGPU feature the texture format needs, or 0. */
   WGPUFeatureName feature;
   uint32_t caps;
};

/* NULL for formats WebGPU has neither as a texture nor as a vertex format. */
const struct wvk_format *wvk_format_get(VkFormat format);

/* The capabilities on this physical device, optional features applied. */
uint32_t wvk_format_caps(const struct wvk_physical_device *pdev, VkFormat format);

/* Whether WebGPU has a vertex format of it; and, where it has not,
 * whether a vertex shader can read an attribute of it itself. */
bool wvk_format_vertex_native(VkFormat format);
bool wvk_format_vertex_pullable(VkFormat format);

/* A texel buffer's format as shaders decode and encode it: the words
 * WVK_META_TEXEL_FORMAT, SIZES, SHIFTS and SWIZZLE of its record. False
 * for formats that cannot be. */
bool wvk_texel_buffer_format(VkFormat format, uint32_t *record);

/* A storage image format shaders can do atomics on: of floats too, as
 * compare-exchange loops (shaderImageFloat32Atomics). */
static inline bool
wvk_format_has_atomics(VkFormat format)
{
   return format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT || format == VK_FORMAT_R32_SFLOAT;
}

/* A format a shading rate attachment can be of: an unsigned integer one
 * of WebGPU's own, whose first component fragment shaders fetch
 * (wvk_nir_fsr.c). */
static inline bool
wvk_format_shading_rate(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_R8_UINT:
   case VK_FORMAT_R8G8_UINT:
   case VK_FORMAT_R8G8B8A8_UINT:
   case VK_FORMAT_R16_UINT:
   case VK_FORMAT_R16G16_UINT:
   case VK_FORMAT_R16G16B16A16_UINT:
   case VK_FORMAT_R32_UINT:
   case VK_FORMAT_R32G32_UINT:
   case VK_FORMAT_R32G32B32A32_UINT:
      return true;
   default:
      return false;
   }
}

#endif
