/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Descriptor sets are host memory: the descriptors, as written. WebGPU's
 * bind groups are made from them at a draw, for the bind group layouts the
 * pipeline has, and kept until the set is written again. A pool is only a
 * list of its sets.
 */
#include "wvk_descriptor_set.h"

#include "wvk_buffer.h"
#include "wvk_device.h"
#include "wvk_device_memory.h"
#include "wvk_entrypoints.h"
#include "wvk_image.h"
#include "wvk_sampler.h"

#include "wvk_format.h"

#include "vk_alloc.h"
#include "vk_buffer_view.h"
#include "vk_descriptor_update_template.h"
#include "vk_util.h"
#include "vk_format.h"

void
wvk_descriptor_set_invalidate(struct wvk_descriptor_set *set)
{
   ++set->generation;
   for (uint32_t i = 0; i < WVK_SET_BIND_GROUPS; ++i) {
      if (set->bind_groups[i].group)
         wgpuBindGroupRelease(set->bind_groups[i].group);
      set->bind_groups[i] = (struct wvk_set_bind_group){ 0 };
   }
}

/* A YCbCr conversion's sampler of its chroma filter, in the descriptor of
 * the second plane. */
static void
set_chroma_sampler(struct wvk_descriptor *d, const struct wvk_sampler *sampler)
{
   if (d->sampler)
      wgpuSamplerRelease(d->sampler);
   d->sampler = sampler && sampler->chroma ? sampler->chroma : NULL;
   if (d->sampler) {
      wgpuSamplerAddRef(d->sampler);
      d->sampler_type = sampler->chroma_binding_type;
      memcpy(d->sampler_meta, sampler->chroma_meta, sizeof(d->sampler_meta));
   }
}

static void
set_sampler(struct wvk_descriptor *d, const struct wvk_sampler *sampler)
{
   if (d->sampler)
      wgpuSamplerRelease(d->sampler);
   d->sampler = NULL;
   memset(d->sampler_meta, 0, sizeof(d->sampler_meta));
   if (!sampler)
      return;
   d->sampler = sampler->wgpu;
   wgpuSamplerAddRef(d->sampler);
   d->sampler_type = sampler->binding_type;
   memcpy(d->sampler_meta, sampler->meta, sizeof(d->sampler_meta));
}

/* What the descriptor names, let go of; the sampler too unless it is
 * immutable, which stays. */
static void
clear_descriptor(struct wvk_descriptor *d, bool keep_sampler)
{
   if (d->mem)
      wvk_device_memory_unref(d->mem);
   if (d->buffer)
      wgpuBufferRelease(d->buffer);
   if (d->view)
      wgpuTextureViewRelease(d->view);
   if (d->cube_faces)
      wgpuTextureViewRelease(d->cube_faces);
   struct wvk_descriptor kept = { 0 };
   if (keep_sampler) {
      kept.sampler = d->sampler;
      kept.sampler_type = d->sampler_type;
      memcpy(kept.sampler_meta, d->sampler_meta, sizeof(kept.sampler_meta));
   } else if (d->sampler) {
      wgpuSamplerRelease(d->sampler);
   }
   *d = kept;
}

static void
copy_descriptor(struct wvk_descriptor *dst, const struct wvk_descriptor *src, bool keep_sampler)
{
   clear_descriptor(dst, keep_sampler);
   const struct wvk_descriptor kept = *dst;
   *dst = *src;
   if (dst->mem)
      wvk_device_memory_ref(dst->mem);
   if (dst->buffer)
      wgpuBufferAddRef(dst->buffer);
   if (dst->view)
      wgpuTextureViewAddRef(dst->view);
   if (dst->cube_faces)
      wgpuTextureViewAddRef(dst->cube_faces);
   if (keep_sampler) {
      dst->sampler = kept.sampler;
      dst->sampler_type = kept.sampler_type;
      memcpy(dst->sampler_meta, kept.sampler_meta, sizeof(dst->sampler_meta));
   } else if (dst->sampler) {
      wgpuSamplerAddRef(dst->sampler);
   }
}

void
wvk_descriptor_set_copy_all(struct wvk_descriptor_set *dst, const struct wvk_descriptor_set *src)
{
   if (dst->layout != src->layout)
      return;
   for (uint32_t i = 0; i < dst->layout->descriptor_count; ++i)
      copy_descriptor(&dst->descriptors[i], &src->descriptors[i], false);
   if (dst->inline_data && src->inline_data)
      memcpy(dst->inline_data, src->inline_data, dst->inline_size);
   wvk_descriptor_set_invalidate(dst);
}

struct wvk_descriptor_set *
wvk_descriptor_set_create(struct wvk_device *dev, struct wvk_descriptor_set_layout *layout)
{
   const size_t size = sizeof(struct wvk_descriptor_set) +
                       layout->descriptor_count * sizeof(struct wvk_descriptor);
   struct wvk_descriptor_set *set = vk_object_zalloc(&dev->vk, NULL, size, VK_OBJECT_TYPE_DESCRIPTOR_SET);
   if (!set)
      return NULL;
   vk_descriptor_set_layout_ref(&layout->vk);
   set->layout = layout;
   list_inithead(&set->link);

   /* Inline uniform blocks: bytes on the host, uploaded to their buffer
    * when a bind group is made after a write (wvk_cmd_draw.c). */
   if (layout->inline_size) {
      set->inline_size = layout->inline_size;
      set->inline_data = calloc(1, set->inline_size);
      WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
      bd.label = wvk_sv("wvk inline uniform blocks");
      bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
      bd.size = set->inline_size;
      set->inline_buffer = wgpuDeviceCreateBuffer(dev->wgpu, &bd);
      if (!set->inline_data || !set->inline_buffer) {
         wvk_descriptor_set_destroy(dev, set);
         return NULL;
      }
      /* Zeroes, as the buffer starts. */
      set->inline_uploaded = set->generation;
      for (uint32_t b = 0; b < layout->binding_count; ++b) {
         const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[b];
         if (bl->type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
            continue;
         struct wvk_descriptor *d = &set->descriptors[bl->descriptor_index];
         d->buffer = set->inline_buffer;
         wgpuBufferAddRef(d->buffer);
         d->offset = bl->inline_offset;
         d->size = set->inline_size - bl->inline_offset;
      }
   }

   /* Immutable samplers are there from the start. */
   for (uint32_t b = 0; b < layout->binding_count; ++b) {
      const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[b];
      if (!bl->immutable_samplers)
         continue;
      for (uint32_t e = 0; e < bl->count; ++e) {
         set_sampler(&set->descriptors[bl->descriptor_index + e], bl->immutable_samplers[e]);
         if (bl->planes > 1)
            set_chroma_sampler(&set->descriptors[bl->plane_descriptor_index + e], bl->immutable_samplers[e]);
      }
   }
   return set;
}

void
wvk_descriptor_set_destroy(struct wvk_device *dev, struct wvk_descriptor_set *set)
{
   wvk_descriptor_set_invalidate(set);
   for (uint32_t i = 0; i < set->layout->descriptor_count; ++i)
      clear_descriptor(&set->descriptors[i], false);
   if (set->inline_buffer)
      wgpuBufferRelease(set->inline_buffer);
   free(set->inline_data);
   list_del(&set->link);
   vk_descriptor_set_layout_unref(&dev->vk, &set->layout->vk);
   vk_object_free(&dev->vk, NULL, set);
}

static void
write_buffer(struct wvk_device *dev, struct wvk_descriptor *d, const VkDescriptorBufferInfo *info)
{
   VK_FROM_HANDLE(wvk_buffer, buffer, info->buffer);
   clear_descriptor(d, false);
   if (!buffer || !buffer->mem)
      return;
   WGPUBuffer wgpu = NULL;
   if (wvk_device_memory_get_buffer(dev, buffer->mem, &wgpu) != VK_SUCCESS)
      return;
   d->mem = buffer->mem;
   wvk_device_memory_ref(d->mem);
   d->buffer = wgpu;
   wgpuBufferAddRef(d->buffer);
   d->offset = buffer->offset + info->offset;
   d->size = vk_buffer_range(&buffer->vk, info->offset, info->range);
}

/* A texel buffer is its range, as a storage buffer: from the 256-byte
 * boundary below, where a WebGPU binding can start, and where the view
 * starts in that is its record's to say. The memory's padding covers the
 * rounding up. */
static void
write_texel_buffer(struct wvk_device *dev, struct wvk_descriptor *d, VkBufferView _view)
{
   VK_FROM_HANDLE(vk_buffer_view, view, _view);
   clear_descriptor(d, false);
   if (!view)
      return;
   struct wvk_buffer *buffer = container_of(view->buffer, struct wvk_buffer, vk);
   WGPUBuffer wgpu = NULL;
   if (!buffer->mem || wvk_device_memory_get_buffer(dev, buffer->mem, &wgpu) != VK_SUCCESS)
      return;
   const uint64_t at = buffer->offset + view->offset;
   const uint64_t start = at & ~255ull;
   d->mem = buffer->mem;
   wvk_device_memory_ref(d->mem);
   d->buffer = wgpu;
   wgpuBufferAddRef(d->buffer);
   d->offset = start;
   d->size = align64(at - start + view->range, 4);
   d->texel_format = view->format;
   d->texel_elements = (uint32_t)MIN2(view->elements, UINT32_MAX);
   d->texel_start = (uint32_t)(at - start);
}

static void
write_image(struct wvk_descriptor *d, const struct wvk_descriptor_set_binding_layout *bl,
            uint32_t element, const VkDescriptorImageInfo *info)
{
   const VkDescriptorType type = bl->type;
   const bool has_sampler = type == VK_DESCRIPTOR_TYPE_SAMPLER || type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   clear_descriptor(d, bl->immutable_samplers != NULL);
   if (type != VK_DESCRIPTOR_TYPE_SAMPLER) {
      const struct wvk_image_view *view = wvk_image_view_from_handle(info->imageView);
      /* A multi-planar view is its first plane's here, the others in the
       * binding's descriptors of them (write_one). */
      if (view && view->plane_count)
         view = view->planes[0];
      /* A storage image of a cube is the 2D array of its faces, and an
       * input attachment an array, fetched at the pass's layer. */
      WGPUTextureView wgpu = view ? view->view : NULL;
      const WGPUTextureViewDescriptor *desc = view ? &view->view_desc : NULL;
      /* And an integer cube, which shaders fetch the faces of. */
      const bool integer_cube = view && (view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE ||
                                         view->vk.view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) &&
                                vk_format_is_int(view->vk.view_format);
      if (view && view->array_view &&
          (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT || integer_cube)) {
         wgpu = view->array_view;
         desc = &view->array_desc;
      }
      if (view && view->storage_view && type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
         wgpu = view->storage_view;
         desc = &view->storage_desc;
      }
      if (wgpu) {
         d->view_image = wvk_image_view_image(view);
         d->view_texture = view->texture;
         d->view_level = view->vk.base_mip_level;
         d->view_levels = desc->mipLevelCount;
         d->view_desc = *desc;
         d->view_desc.label = (WGPUStringView)WGPU_STRING_VIEW_INIT;
         d->view = wgpu;
         wgpuTextureViewAddRef(d->view);
         if (wgpu != view->array_view && view->array_view &&
             (view->view_desc.dimension == WGPUTextureViewDimension_Cube ||
              view->view_desc.dimension == WGPUTextureViewDimension_CubeArray)) {
            d->cube_faces = view->array_view;
            wgpuTextureViewAddRef(d->cube_faces);
         }
         d->view_format = view->vk.view_format;
         d->swizzle = view->swizzle;
         d->channels = view->channels;
         d->min_lod = MAX2(view->vk.min_lod - (float)view->vk.base_mip_level, 0.0f);
         const struct wvk_image *of = wvk_image_view_image(view);
         if (of->vk.image_type == VK_IMAGE_TYPE_3D) {
            d->slices = view->vk.storage.z_slice_offset | view->vk.storage.z_slice_count << 16;
            d->of_3d = view->vk.view_type != VK_IMAGE_VIEW_TYPE_3D;
         }
      }
      struct wvk_image *image = view ? wvk_image_view_image(view) : NULL;
      if (type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE && image && image->atomic_buffer &&
          wvk_format_has_atomics(view->vk.view_format)) {
         /* The view's texels in the image's buffer of them. */
         const uint32_t level = view->vk.base_mip_level;
         const VkExtent3D e = vk_image_mip_level_extent(&image->vk, level);
         const bool is_3d = image->vk.image_type == VK_IMAGE_TYPE_3D;
         const uint32_t row = image->atomic_pitch[level] / 4;
         d->atomic_image = image;
         d->buffer = image->atomic_buffer;
         wgpuBufferAddRef(d->buffer);
         d->offset = 0;
         d->size = image->atomic_size;
         d->atomic_meta[WVK_META_IMG_BASE] =
            (uint32_t)(image->atomic_offset[level] / 4) + (is_3d ? 0 : view->vk.base_array_layer * row * e.height);
         d->atomic_meta[WVK_META_IMG_ROW] = row;
         d->atomic_meta[WVK_META_IMG_LAYER] = row * e.height;
         d->atomic_meta[WVK_META_IMG_WIDTH] = e.width;
         d->atomic_meta[WVK_META_IMG_HEIGHT] = image->vk.image_type == VK_IMAGE_TYPE_1D ? 1 : e.height;
         d->atomic_meta[WVK_META_IMG_LAYERS] = is_3d ? e.depth : view->vk.layer_count;
      }
   }
   if (has_sampler && !bl->immutable_samplers)
      set_sampler(d, wvk_sampler_from_handle(info->sampler));
}

static void
write_one(struct wvk_device *dev, struct wvk_descriptor_set *set, uint32_t binding,
          uint32_t element, const VkDescriptorImageInfo *image,
          const VkDescriptorBufferInfo *buffer, const VkBufferView *texel)
{
   const struct wvk_descriptor_set_binding_layout *bl = &set->layout->bindings[binding];
   struct wvk_descriptor *d = &set->descriptors[bl->descriptor_index + element];
   switch (bl->type) {
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      write_buffer(dev, d, buffer);
      break;
   case VK_DESCRIPTOR_TYPE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
      write_image(d, bl, element, image);
      /* A view's other planes, each a texture of its own. */
      for (uint32_t p = 1; p < bl->planes; ++p) {
         const struct wvk_image_view *view = image ? wvk_image_view_from_handle(image->imageView) : NULL;
         const VkDescriptorImageInfo plane = {
            .imageView = view && p < view->plane_count ? wvk_image_view_to_handle(view->planes[p]) : VK_NULL_HANDLE,
         };
         /* Its sampler, the second plane's of the chroma filter, stays. */
         struct wvk_descriptor_set_binding_layout pbl = *bl;
         pbl.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
         write_image(&set->descriptors[bl->plane_descriptor_index + (p - 1) * bl->count + element], &pbl, element,
                     &plane);
      }
      break;
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      write_texel_buffer(dev, d, texel ? *texel : VK_NULL_HANDLE);
      break;
   default:
      mesa_logw_once("wvk: %s descriptors are not supported yet",
                     vk_DescriptorType_to_str(bl->type));
      break;
   }
}

/* A write runs on into the following bindings when it has more
 * descriptors than its first binding. */
static void
next_element(const struct wvk_descriptor_set_layout *layout, uint32_t *binding, uint32_t *element)
{
   ++*element;
   while (*binding < layout->binding_count && *element >= layout->bindings[*binding].count) {
      *element = 0;
      ++*binding;
   }
}

/* Bytes of inline uniform blocks from `offset` in `binding`, running on
 * into the following blocks as a write of more than one does; to the set
 * if `to`, else from it. */
static void
inline_bytes(struct wvk_descriptor_set *set, uint32_t binding, uint32_t offset, void *data, uint32_t size,
             bool to)
{
   const struct wvk_descriptor_set_layout *layout = set->layout;
   uint8_t *bytes = data;
   for (; size && binding < layout->binding_count; ++binding, offset = 0) {
      const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[binding];
      if (bl->type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK || offset >= bl->inline_size)
         continue;
      const uint32_t n = MIN2(size, bl->inline_size - offset);
      uint8_t *at = set->inline_data + bl->inline_offset + offset;
      if (to)
         memcpy(at, bytes, n);
      else
         memcpy(bytes, at, n);
      bytes += n;
      size -= n;
   }
}

void
wvk_descriptor_set_write(struct wvk_device *dev, struct wvk_descriptor_set *set,
                         const VkWriteDescriptorSet *write)
{
   if (write->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
      /* dstArrayElement and descriptorCount are bytes. */
      const VkWriteDescriptorSetInlineUniformBlock *block =
         vk_find_struct_const(write->pNext, WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK);
      if (block && set->inline_data)
         inline_bytes(set, write->dstBinding, write->dstArrayElement, (void *)block->pData,
                      MIN2(block->dataSize, write->descriptorCount), true);
      wvk_descriptor_set_invalidate(set);
      return;
   }
   uint32_t binding = write->dstBinding, element = write->dstArrayElement;
   for (uint32_t i = 0; i < write->descriptorCount && binding < set->layout->binding_count; ++i) {
      write_one(dev, set, binding, element,
                write->pImageInfo ? &write->pImageInfo[i] : NULL,
                write->pBufferInfo ? &write->pBufferInfo[i] : NULL,
                write->pTexelBufferView ? &write->pTexelBufferView[i] : NULL);
      next_element(set->layout, &binding, &element);
   }
   wvk_descriptor_set_invalidate(set);
}

void
wvk_descriptor_set_write_template(struct wvk_device *dev, struct wvk_descriptor_set *set,
                                  const struct vk_descriptor_update_template *template,
                                  const void *data)
{
   for (uint32_t i = 0; i < template->entry_count; ++i) {
      const struct vk_descriptor_template_entry *entry = &template->entries[i];
      if (entry->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
         if (set->inline_data)
            inline_bytes(set, entry->binding, entry->array_element, (uint8_t *)data + entry->offset,
                         entry->array_count, true);
         continue;
      }
      uint32_t binding = entry->binding, element = entry->array_element;
      for (uint32_t j = 0; j < entry->array_count && binding < set->layout->binding_count; ++j) {
         const void *src = (const uint8_t *)data + entry->offset + j * entry->stride;
         write_one(dev, set, binding, element, src, src, src);
         next_element(set->layout, &binding, &element);
      }
   }
   wvk_descriptor_set_invalidate(set);
}

void
wvk_descriptor_set_meta(const struct wvk_descriptor_set *set, uint32_t *words, uint32_t max_words)
{
   const struct wvk_descriptor_set_layout *layout = set->layout;
   const uint32_t slots = MIN2(layout->slot_count, max_words / WVK_META_WORDS);
   memset(words, 0, slots * WVK_META_WORDS * 4);
   for (uint32_t s = 0; s < slots; ++s) {
      const struct wvk_descriptor_slot *slot = &layout->slots[s];
      const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[slot->binding];
      const struct wvk_descriptor *d = &set->descriptors[wvk_slot_descriptor_index(bl, slot)];
      uint32_t *record = words + s * WVK_META_WORDS;
      switch (bl->type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
         memcpy(record, d->sampler_meta, sizeof(d->sampler_meta));
         break;
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         if (slot->sampler) {
            memcpy(record, d->sampler_meta, sizeof(d->sampler_meta));
         } else {
            record[WVK_META_TEX_SWIZZLE] = d->swizzle;
            record[WVK_META_TEX_CHANNELS] = d->channels;
            memcpy(&record[WVK_META_TEX_MIN_LOD], &d->min_lod, 4);
            record[WVK_META_TEX_SLICE] = d->slices & 0xffff;
         }
         break;
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         record[WVK_META_TEX_SWIZZLE] = d->swizzle;
         record[WVK_META_TEX_CHANNELS] = d->channels;
         memcpy(&record[WVK_META_TEX_MIN_LOD], &d->min_lod, 4);
         record[WVK_META_TEX_SLICE] = d->slices & 0xffff;
         break;
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         if (d->atomic_image)
            memcpy(record, d->atomic_meta, sizeof(d->atomic_meta));
         record[WVK_META_IMG_CHANNELS] = d->channels;
         record[WVK_META_IMG_SLICES] = d->slices;
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
         record[WVK_META_BUF_RANGE] = d->buffer ? (uint32_t)MIN2(d->size, UINT32_MAX) : 0;
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         /* A format shaders cannot decode, or no view: nothing in it. */
         if (d->buffer && wvk_texel_buffer_format(d->texel_format, record)) {
            record[WVK_META_TEXEL_COUNT] = d->texel_elements;
            record[WVK_META_TEXEL_START] = d->texel_start;
         }
         break;
      default:
         break;
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_UpdateDescriptorSets(VkDevice _device, uint32_t descriptorWriteCount,
                         const VkWriteDescriptorSet *pDescriptorWrites,
                         uint32_t descriptorCopyCount, const VkCopyDescriptorSet *pDescriptorCopies)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   for (uint32_t i = 0; i < descriptorWriteCount; ++i) {
      VK_FROM_HANDLE(wvk_descriptor_set, set, pDescriptorWrites[i].dstSet);
      wvk_descriptor_set_write(dev, set, &pDescriptorWrites[i]);
   }
   for (uint32_t i = 0; i < descriptorCopyCount; ++i) {
      const VkCopyDescriptorSet *copy = &pDescriptorCopies[i];
      VK_FROM_HANDLE(wvk_descriptor_set, src, copy->srcSet);
      VK_FROM_HANDLE(wvk_descriptor_set, dst, copy->dstSet);
      uint32_t sb = copy->srcBinding, se = copy->srcArrayElement;
      uint32_t db = copy->dstBinding, de = copy->dstArrayElement;
      if (sb < src->layout->binding_count &&
          src->layout->bindings[sb].type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
         /* Bytes, through a copy: the two may be one set. */
         void *bytes = malloc(copy->descriptorCount);
         if (bytes && src->inline_data && dst->inline_data) {
            inline_bytes(src, sb, se, bytes, copy->descriptorCount, false);
            inline_bytes(dst, db, de, bytes, copy->descriptorCount, true);
         }
         free(bytes);
         wvk_descriptor_set_invalidate(dst);
         continue;
      }
      for (uint32_t j = 0; j < copy->descriptorCount; ++j) {
         if (sb >= src->layout->binding_count || db >= dst->layout->binding_count)
            break;
         const struct wvk_descriptor_set_binding_layout *sbl = &src->layout->bindings[sb];
         const struct wvk_descriptor_set_binding_layout *dbl = &dst->layout->bindings[db];
         const struct wvk_descriptor *s = &src->descriptors[sbl->descriptor_index + se];
         struct wvk_descriptor *d = &dst->descriptors[dbl->descriptor_index + de];
         copy_descriptor(d, s, dbl->immutable_samplers != NULL);
         /* And its planes'. */
         for (uint32_t p = 1; p < MIN2(sbl->planes, dbl->planes); ++p)
            copy_descriptor(&dst->descriptors[dbl->plane_descriptor_index + (p - 1) * dbl->count + de],
                            &src->descriptors[sbl->plane_descriptor_index + (p - 1) * sbl->count + se], true);
         next_element(src->layout, &sb, &se);
         next_element(dst->layout, &db, &de);
      }
      wvk_descriptor_set_invalidate(dst);
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_UpdateDescriptorSetWithTemplate(VkDevice _device, VkDescriptorSet descriptorSet,
                                    VkDescriptorUpdateTemplate descriptorUpdateTemplate,
                                    const void *pData)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_descriptor_set, set, descriptorSet);
   VK_FROM_HANDLE(vk_descriptor_update_template, template, descriptorUpdateTemplate);
   wvk_descriptor_set_write_template(dev, set, template, pData);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateDescriptorPool(VkDevice _device, const VkDescriptorPoolCreateInfo *pCreateInfo,
                         const VkAllocationCallbacks *pAllocator, VkDescriptorPool *pDescriptorPool)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   struct wvk_descriptor_pool *pool = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*pool),
                                                       VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   list_inithead(&pool->sets);
   *pDescriptorPool = wvk_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

static void
free_all(struct wvk_device *dev, struct wvk_descriptor_pool *pool)
{
   list_for_each_entry_safe(struct wvk_descriptor_set, set, &pool->sets, link)
      wvk_descriptor_set_destroy(dev, set);
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyDescriptorPool(VkDevice _device, VkDescriptorPool descriptorPool,
                          const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_descriptor_pool, pool, descriptorPool);
   if (!pool)
      return;
   free_all(dev, pool);
   vk_object_free(&dev->vk, pAllocator, pool);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_ResetDescriptorPool(VkDevice _device, VkDescriptorPool descriptorPool,
                        VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_descriptor_pool, pool, descriptorPool);
   free_all(dev, pool);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_AllocateDescriptorSets(VkDevice _device, const VkDescriptorSetAllocateInfo *pAllocateInfo,
                           VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_descriptor_pool, pool, pAllocateInfo->descriptorPool);
   for (uint32_t i = 0; i < pAllocateInfo->descriptorSetCount; ++i) {
      VK_FROM_HANDLE(wvk_descriptor_set_layout, layout, pAllocateInfo->pSetLayouts[i]);
      struct wvk_descriptor_set *set = wvk_descriptor_set_create(dev, layout);
      if (!set) {
         for (uint32_t j = 0; j < i; ++j) {
            wvk_descriptor_set_destroy(dev, wvk_descriptor_set_from_handle(pDescriptorSets[j]));
            pDescriptorSets[j] = VK_NULL_HANDLE;
         }
         return vk_error(dev, VK_ERROR_OUT_OF_POOL_MEMORY);
      }
      set->pool = pool;
      list_addtail(&set->link, &pool->sets);
      pDescriptorSets[i] = wvk_descriptor_set_to_handle(set);
   }
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_FreeDescriptorSets(VkDevice _device, VkDescriptorPool descriptorPool, uint32_t descriptorSetCount,
                       const VkDescriptorSet *pDescriptorSets)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   for (uint32_t i = 0; i < descriptorSetCount; ++i) {
      VK_FROM_HANDLE(wvk_descriptor_set, set, pDescriptorSets[i]);
      if (set)
         wvk_descriptor_set_destroy(dev, set);
   }
   return VK_SUCCESS;
}
