/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_IMAGE_H
#define WVK_IMAGE_H

#include "wvk_private.h"

#include "vk_image.h"

/* A WGPUTexture of its own; its memory is accounting only. A 1D image is
 * a 2D texture one texel high, so that it can have mips and be rendered
 * to, which WebGPU's 1D textures cannot. */
struct wvk_image {
   struct vk_image vk;
   WGPUTexture texture;
   WGPUTextureFormat format;
   WGPUTextureDimension dimension;
   WGPUTextureUsage usage;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE)

struct wvk_image_view {
   struct vk_image_view vk;
   /* For binding: the whole subresource range, in the view's dimension. */
   WGPUTextureView view;
   WGPUTextureFormat format;
   WGPUTextureAspect aspect;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_image_view, vk.base, VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW)

static inline struct wvk_image *
wvk_image_view_image(const struct wvk_image_view *view)
{
   return container_of(view->vk.image, struct wvk_image, vk);
}

/* A single-level, single-layer 2D (or 3D slice) view for a render
 * attachment, which WebGPU insists on. The caller releases it. */
WGPUTextureView wvk_image_attachment_view(struct wvk_image *image, WGPUTextureFormat format,
                                          WGPUTextureAspect aspect, uint32_t level, uint32_t layer);

WGPUTextureAspect wvk_aspect(VkImageAspectFlags aspects);

#endif
