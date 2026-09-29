/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_IMAGE_H
#define WVK_IMAGE_H

#include "wvk_private.h"

#include "vk_image.h"

struct wvk_device;
struct wvk_device_memory;
struct wvk_physical_device;
struct wvk_replay;

/* A texture of the image's in another format, which WebGPU cannot view
 * the image's own as: of the same bytes a texel, or for an uncompressed
 * view of a compressed image a texel a block, at one level. */
struct wvk_image_alias {
   WGPUTextureFormat format;
   WGPUTexture texture;
   /* The image's levels it has, from its level 0. */
   uint32_t first_level;
   uint32_t level_count;
};

#define WVK_IMAGE_MAX_ALIASES 8
/* An image's textures: its own, alias i as i + 1, and its shadow. */
#define WVK_IMAGE_TEXTURE 0
#define WVK_IMAGE_SHADOW 31

/* A WGPUTexture of its own; its memory is accounting only. A 1D image is
 * a 2D texture one texel high, so that it can have mips and be rendered
 * to, which WebGPU's 1D textures cannot. */
struct wvk_image {
   struct vk_image vk;
   WGPUTexture texture;
   /* A multisampled array, which WebGPU has no textures of: a texture of
    * each layer, of which `texture` is the first. */
   WGPUTexture *layer_textures;
   /* A compressed image whose size is not whole blocks: its texture is the
    * next size up, which sampling would see. So it is sampled from this,
    * of its own size and uncompressed, which what is copied to the image
    * is decoded into (wvk_cmd_copy.c). */
   WGPUTexture shadow;
   WGPUTextureFormat shadow_format;
   /* Made as views need them, and kept with the image. */
   struct wvk_image_alias aliases[WVK_IMAGE_MAX_ALIASES];
   uint32_t alias_count;
   /* An image with aliases or a shadow: which of its textures have, at
    * each level, what was last written there, and which one it was
    * written to. Kept as submits are replayed, which is the queue's
    * order; copies between them bring one up to date when it is used
    * (wvk_image_make_current). */
   bool tracked;
   uint32_t valid[16];
   uint8_t owner[16];
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

/* Decode what the image's texture has at a level, every layer or 3D
 * slice, into its shadow (wvk_cmd_copy.c). */
void wvk_image_update_shadow(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image,
                             uint32_t level);

/* The image's texture `t`, and the image's level its level 0 is. */
static inline WGPUTexture
wvk_image_texture(const struct wvk_image *image, uint32_t t, uint32_t *first_level)
{
   *first_level = 0;
   if (t == WVK_IMAGE_SHADOW)
      return image->shadow;
   if (t == WVK_IMAGE_TEXTURE)
      return image->texture;
   *first_level = image->aliases[t - 1].first_level;
   return image->aliases[t - 1].texture;
}

/* Whether the image's texture `t` has what was last written to levels
 * [first, first + count). */
static inline bool
wvk_image_current(const struct wvk_image *image, uint32_t t, uint32_t first, uint32_t count)
{
   if (!image->tracked)
      return true;
   for (uint32_t l = first; l < first + count && l < ARRAY_SIZE(image->valid); ++l) {
      if (!(image->valid[l] & BITFIELD_BIT(t)))
         return false;
   }
   return true;
}

/* Bring it up to date, by copies from the texture written last: outside
 * any pass, whose end this is if one is going on. */
void wvk_image_make_current(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image,
                            uint32_t t, uint32_t first, uint32_t count);

/* After a write to texture `t` of the levels, the others are behind. */
static inline void
wvk_image_written(struct wvk_image *image, uint32_t t, uint32_t first, uint32_t count)
{
   if (!image->tracked)
      return;
   for (uint32_t l = first; l < first + count && l < ARRAY_SIZE(image->valid); ++l) {
      image->owner[l] = t;
      image->valid[l] = BITFIELD_BIT(t);
   }
}

/* A transfer's use of the image's own texture, which it first brings up
 * to date: outside any pass. */
static inline void
wvk_image_use(struct wvk_device *dev, struct wvk_replay *replay, struct wvk_image *image, uint32_t first,
              uint32_t count, bool write)
{
   if (!image->tracked)
      return;
   if (!wvk_image_current(image, WVK_IMAGE_TEXTURE, first, count))
      wvk_image_make_current(dev, replay, image, WVK_IMAGE_TEXTURE, first, count);
   if (write)
      wvk_image_written(image, WVK_IMAGE_TEXTURE, first, count);
}

/* The texture layer `layer` of the image is in, and which layer of it. */
static inline WGPUTexture
wvk_image_layer_texture(const struct wvk_image *image, uint32_t layer, uint32_t *texture_layer)
{
   if (image->layer_textures) {
      *texture_layer = 0;
      return image->layer_textures[MIN2(layer, image->vk.array_layers - 1)];
   }
   *texture_layer = layer;
   return image->texture;
}

/* A copy of the image's texture, every level and layer of it, for a pass
 * that reads the image as it renders to it. Colour images of one sample
 * only; NULL for the rest. */
WGPUTexture wvk_image_snapshot(WGPUDevice device, const struct wvk_image *image, uint32_t t,
                               WGPUCommandEncoder encoder);

/* The same of a multisampled colour image's first layer, which is what
 * shaders see of it: a pass copying each sample (wvk_cmd_copy.c). */
WGPUTexture wvk_image_snapshot_ms(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image);

/* The same of a depth-stencil image's depth, as R32Float, or its stencil,
 * as R8Uint; one sample only (wvk_cmd_copy.c). */
WGPUTexture wvk_image_snapshot_ds(struct wvk_device *dev, struct wvk_replay *replay, const struct wvk_image *image,
                                  bool stencil);

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
   /* And for a storage image of more than one level, its base level:
    * that is all a storage image reads and writes, and WebGPU binds no
    * more than one level as storage. */
   WGPUTextureView storage_view;
   /* How they were made, for a view alike of a copy of the image: a pass
    * that reads what it renders to reads a copy (wvk_cmd_draw.c). */
   WGPUTextureViewDescriptor view_desc, array_desc, storage_desc;
   WGPUTextureFormat format;
   WGPUTextureAspect aspect;
   /* Which of the image's textures it is of (wvk_image_texture). */
   uint32_t texture;
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
 * attachment, which WebGPU insists on, of the image's texture `t`. The
 * caller releases it. */
WGPUTextureView wvk_image_attachment_view(struct wvk_image *image, uint32_t t, WGPUTextureFormat format,
                                          WGPUTextureAspect aspect, uint32_t level, uint32_t layer);

WGPUTextureAspect wvk_aspect(VkImageAspectFlags aspects);

#endif
