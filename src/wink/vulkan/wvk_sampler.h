/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_SAMPLER_H
#define WVK_SAMPLER_H

#include "wvk_private.h"

#include "vk_sampler.h"

struct wvk_sampler {
   struct vk_sampler vk;
   WGPUSampler wgpu;
   /* What a bind group layout entry for it has to say. */
   WGPUSamplerBindingType binding_type;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_sampler, vk.base, VkSampler, VK_OBJECT_TYPE_SAMPLER)

WGPUCompareFunction wvk_compare_function(VkCompareOp op);

#endif
