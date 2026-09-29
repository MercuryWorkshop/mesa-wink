/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_SAMPLER_H
#define WVK_SAMPLER_H

#include "wvk_private.h"

#include "wvk_descriptor_set_layout.h"

#include "vk_sampler.h"

struct wvk_sampler {
   struct vk_sampler vk;
   WGPUSampler wgpu;
   /* What a bind group layout entry for it has to say. */
   WGPUSamplerBindingType binding_type;
   /* What WebGPU's sampler cannot say, for shaders: its descriptor's
    * record (WVK_META_SAMPLER_*). */
   uint32_t meta[WVK_META_WORDS];
   /* Of a YCbCr conversion: the same, of its chroma filter, which the
    * chroma planes are sampled with. */
   WGPUSampler chroma;
   WGPUSamplerBindingType chroma_binding_type;
   uint32_t chroma_meta[WVK_META_WORDS];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_sampler, vk.base, VkSampler, VK_OBJECT_TYPE_SAMPLER)

WGPUCompareFunction wvk_compare_function(VkCompareOp op);

#endif
