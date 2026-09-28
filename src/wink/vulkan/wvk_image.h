/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_IMAGE_H
#define WVK_IMAGE_H

#include "wvk_private.h"

#include "vk_image.h"

struct wvk_device_memory;
struct wvk_physical_device;

/* A WGPUTexture of its own; its memory is accounting only. A 1D image is
 * a 2D texture one texel high, so that it can have mips and be rendered
 * to, which WebGPU's 1D textures cannot. */
struct wvk_image {
   struct vk_image vk;
   WGPUTexture texture;
   WGPUTextureFormat format;
   WGPUTextureDimension dimension;
   WGPUTextureUsage usage;

   /* A linear image has no texture: it is its memory, rows `pitch` bytes
    * apart -- 256-aligned, so copies to and from textures are one WebGPU
    * copy each. Only transfers use one. */
   bool linear;
   uint32_t pitch;
   struct wvk_device_memory *mem;
   uint64_t offset;

   /* A 32-bit storage image, which shaders may do atomics on, and WGSL
    * has no atomics on textures: a storage buffer of its texels too, each
    * level's rows `atomic_pitch[level]` bytes apart, which shaders doing
    * atomics use for all they do with it. Between barriers one or the
    * other is the image (wvk_replay_sync_atomic_images). */
   WGPUBuffer atomic_buffer;
   uint64_t atomic_size;
   uint64_t atomic_offset[16];
   uint32_t atomic_pitch[16];
   bool atomic_in_buffer;
};

/* Copy an atomic image's texels from its texture to its buffer, or back,
 * outside any pass. */
void wvk_image_copy_atomic(struct wvk_image *image, WGPUCommandEncoder encoder, bool to_buffer);

/* Whether an image of this format, type and so on can be linear. */
bool wvk_image_linear_supported(const struct wvk_physical_device *pdev, VkFormat format, VkImageType type,
                                VkImageUsageFlags usage);
uint32_t wvk_image_linear_pitch(VkFormat format, uint32_t width);

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE)

struct wvk_image_view {
   struct vk_image_view vk;
   /* For binding: the whole subresource range, in the view's dimension;
    * and for a cube's, as a 2D array of its faces, which is how a storage
    * image of it binds, and how one is fetched from; and for an input
    * attachment's, as a 2D array, fetched at the layer drawn. */
   WGPUTextureView view;
   WGPUTextureView array_view;
   WGPUTextureFormat format;
   WGPUTextureAspect aspect;
   /* Its components, which WebGPU views have no way to remap, for shaders
    * to: WVK_META_TEX_SWIZZLE; and those its format has. */
   uint32_t swizzle;
   uint32_t channels;
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
