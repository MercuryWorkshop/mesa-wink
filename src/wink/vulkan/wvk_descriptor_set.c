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

   /* Immutable samplers are there from the start. */
   for (uint32_t b = 0; b < layout->binding_count; ++b) {
      const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[b];
      if (!bl->immutable_samplers)
         continue;
      for (uint32_t e = 0; e < bl->count; ++e)
         set->descriptors[bl->descriptor_index + e].sampler = bl->immutable_samplers[e];
   }
   return set;
}

void
wvk_descriptor_set_destroy(struct wvk_device *dev, struct wvk_descriptor_set *set)
{
   wvk_descriptor_set_invalidate(set);
   if (set->inline_buffer)
      wgpuBufferRelease(set->inline_buffer);
   if (set->meta_buffer)
      wgpuBufferRelease(set->meta_buffer);
   list_del(&set->link);
   vk_descriptor_set_layout_unref(&dev->vk, &set->layout->vk);
   vk_object_free(&dev->vk, NULL, set);
}

static void
write_buffer(struct wvk_device *dev, struct wvk_descriptor *d, const VkDescriptorBufferInfo *info)
{
   VK_FROM_HANDLE(wvk_buffer, buffer, info->buffer);
   *d = (struct wvk_descriptor){ 0 };
   if (!buffer || !buffer->mem)
      return;
   WGPUBuffer wgpu = NULL;
   if (wvk_device_memory_get_buffer(dev, buffer->mem, &wgpu) != VK_SUCCESS)
      return;
   d->mem = buffer->mem;
   d->buffer = wgpu;
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
   *d = (struct wvk_descriptor){ 0 };
   if (!view)
      return;
   struct wvk_buffer *buffer = container_of(view->buffer, struct wvk_buffer, vk);
   WGPUBuffer wgpu = NULL;
   if (!buffer->mem || wvk_device_memory_get_buffer(dev, buffer->mem, &wgpu) != VK_SUCCESS)
      return;
   const uint64_t at = buffer->offset + view->offset;
   const uint64_t start = at & ~255ull;
   d->mem = buffer->mem;
   d->buffer = wgpu;
   d->offset = start;
   d->size = align64(at - start + view->range, 4);
   d->buffer_view = view;
   d->texel_start = (uint32_t)(at - start);
}

static void
write_image(struct wvk_descriptor *d, const struct wvk_descriptor_set_binding_layout *bl,
            uint32_t element, const VkDescriptorImageInfo *info)
{
   const VkDescriptorType type = bl->type;
   if (type != VK_DESCRIPTOR_TYPE_SAMPLER)
      d->view = wvk_image_view_from_handle(info->imageView);
   if ((type == VK_DESCRIPTOR_TYPE_SAMPLER || type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) &&
       !bl->immutable_samplers)
      d->sampler = wvk_sampler_from_handle(info->sampler);
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

void
wvk_descriptor_set_write(struct wvk_device *dev, struct wvk_descriptor_set *set,
                         const VkWriteDescriptorSet *write)
{
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
wvk_descriptor_set_meta(const struct wvk_descriptor_set *set, uint32_t *words)
{
   const struct wvk_descriptor_set_layout *layout = set->layout;
   memset(words, 0, wvk_descriptor_set_meta_size(layout));
   for (uint32_t s = 0; s < layout->slot_count; ++s) {
      const struct wvk_descriptor_slot *slot = &layout->slots[s];
      const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[slot->binding];
      const struct wvk_descriptor *d = &set->descriptors[bl->descriptor_index + slot->element];
      uint32_t *record = words + s * WVK_META_WORDS;
      switch (bl->type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
         if (d->sampler)
            memcpy(record, d->sampler->meta, sizeof(d->sampler->meta));
         break;
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         if (slot->sampler && d->sampler)
            memcpy(record, d->sampler->meta, sizeof(d->sampler->meta));
         else if (!slot->sampler && d->view)
            record[WVK_META_TEX_SWIZZLE] = d->view->swizzle;
         break;
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         if (d->view)
            record[WVK_META_TEX_SWIZZLE] = d->view->swizzle;
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         /* A format shaders cannot decode, or no view: nothing in it. */
         if (d->buffer_view && wvk_texel_buffer_format(d->buffer_view->format, record)) {
            record[WVK_META_TEXEL_COUNT] = (uint32_t)MIN2(d->buffer_view->elements, UINT32_MAX);
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
      for (uint32_t j = 0; j < copy->descriptorCount; ++j) {
         if (sb >= src->layout->binding_count || db >= dst->layout->binding_count)
            break;
         const struct wvk_descriptor *s =
            &src->descriptors[src->layout->bindings[sb].descriptor_index + se];
         struct wvk_descriptor *d =
            &dst->descriptors[dst->layout->bindings[db].descriptor_index + de];
         struct wvk_sampler *immutable = dst->layout->bindings[db].immutable_samplers
                                            ? d->sampler : NULL;
         *d = *s;
         if (immutable)
            d->sampler = immutable;
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
