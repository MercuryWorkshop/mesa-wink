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

#endif
