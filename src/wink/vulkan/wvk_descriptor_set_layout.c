/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#include "wvk_descriptor_set_layout.h"

#include "wvk_device.h"
#include "wvk_entrypoints.h"
#include "wvk_sampler.h"

#include "vk_alloc.h"
#include "vk_util.h"

#include "util/mesa-blake3.h"

static bool
binding_has_immutable_samplers(const VkDescriptorSetLayoutBinding *binding)
{
   return (binding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
           binding->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) &&
          binding->pImmutableSamplers;
}

static const VkDescriptorSetLayoutBinding *
find_binding(const VkDescriptorSetLayoutCreateInfo *info, uint32_t binding)
{
   for (uint32_t i = 0; i < info->bindingCount; ++i) {
      if (info->pBindings[i].binding == binding)
         return &info->pBindings[i];
   }
   return NULL;
}

static void
count_layout(const VkDescriptorSetLayoutCreateInfo *info, uint32_t *binding_count,
             uint32_t *slot_count, uint32_t *sampler_count)
{
   *binding_count = 0;
   *slot_count = 0;
   *sampler_count = 0;
   for (uint32_t i = 0; i < info->bindingCount; ++i) {
      const VkDescriptorSetLayoutBinding *b = &info->pBindings[i];
      *binding_count = MAX2(*binding_count, b->binding + 1);
      if (b->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)
         *slot_count += 1;
      else
         *slot_count += b->descriptorCount * wvk_descriptor_type_slots(b->descriptorType);
      if (binding_has_immutable_samplers(b))
         *sampler_count += b->descriptorCount;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateDescriptorSetLayout(VkDevice _device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                              const VkAllocationCallbacks *pAllocator,
                              VkDescriptorSetLayout *pSetLayout)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);

   uint32_t binding_count, slot_count, sampler_count;
   count_layout(pCreateInfo, &binding_count, &slot_count, &sampler_count);
   if (slot_count > WVK_MAX_SET_SLOTS)
      return vk_errorf(dev, VK_ERROR_OUT_OF_HOST_MEMORY,
                       "a descriptor set of %u WebGPU bindings is more than a bind group takes",
                       slot_count);

   VK_MULTIALLOC(ma);
   VK_MULTIALLOC_DECL_SIZE(&ma, struct wvk_descriptor_set_layout, layout,
                           sizeof(struct wvk_descriptor_set_layout) +
                              binding_count * sizeof(struct wvk_descriptor_set_binding_layout));
   VK_MULTIALLOC_DECL(&ma, struct wvk_descriptor_slot, slots, slot_count);
   VK_MULTIALLOC_DECL(&ma, struct wvk_sampler *, samplers, sampler_count);
   if (!vk_descriptor_set_layout_multizalloc(&dev->vk, &ma, pCreateInfo))
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   layout->binding_count = binding_count;
   layout->slot_count = slot_count;
   layout->slots = slots;

   /* In binding order, which is also the order of dynamic offsets. */
   uint32_t descriptor = 0, dynamic = 0, slot = 0, inline_end = 0;
   struct wvk_sampler **next_sampler = samplers;
   for (uint32_t b = 0; b < binding_count; ++b) {
      const VkDescriptorSetLayoutBinding *info = find_binding(pCreateInfo, b);
      struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[b];
      if (!info || !info->descriptorCount) {
         bl->type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
         continue;
      }
      bl->type = info->descriptorType;
      bl->stages = info->stageFlags;
      bl->descriptor_index = descriptor;
      bl->slot = slot;
      if (info->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
         /* descriptorCount is its size in bytes: one uniform buffer, a
          * range of the set's buffer for them. */
         bl->count = 1;
         bl->inline_offset = align(inline_end, 256);
         bl->inline_size = info->descriptorCount;
         inline_end = bl->inline_offset + align(info->descriptorCount, 16);
      } else {
         bl->count = info->descriptorCount;
      }
      descriptor += bl->count;
      if (wvk_descriptor_type_is_dynamic(bl->type)) {
         bl->dynamic_index = dynamic;
         dynamic += bl->count;
      }
      const uint32_t halves = wvk_descriptor_type_slots(bl->type);
      for (uint32_t h = 0; h < halves; ++h) {
         for (uint32_t e = 0; e < bl->count; ++e) {
            layout->slots[slot++] = (struct wvk_descriptor_slot){
               .binding = b,
               .element = e,
               .sampler = h == 1,
            };
         }
      }
      if (binding_has_immutable_samplers(info)) {
         bl->immutable_samplers = next_sampler;
         for (uint32_t e = 0; e < bl->count; ++e)
            bl->immutable_samplers[e] = wvk_sampler_from_handle(info->pImmutableSamplers[e]);
         next_sampler += bl->count;
      }
   }
   layout->descriptor_count = descriptor;
   layout->dynamic_count = dynamic;
   layout->inline_size = inline_end ? inline_end + 256 : 0;
   layout->vk.dynamic_descriptor_count = dynamic;

   /* What the runtime's pipeline cache keys shaders on: everything a
    * compile reads of the layout. */
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, &layout->binding_count, sizeof(layout->binding_count));
   for (uint32_t b = 0; b < binding_count; ++b) {
      const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[b];
      _mesa_blake3_update(&ctx, &bl->type, sizeof(bl->type));
      _mesa_blake3_update(&ctx, &bl->count, sizeof(bl->count));
      _mesa_blake3_update(&ctx, &bl->stages, sizeof(bl->stages));
      _mesa_blake3_update(&ctx, &bl->slot, sizeof(bl->slot));
      const bool immutable = bl->immutable_samplers != NULL;
      _mesa_blake3_update(&ctx, &immutable, sizeof(immutable));
   }
   _mesa_blake3_final(&ctx, layout->vk.blake3);

   *pSetLayout = wvk_descriptor_set_layout_to_handle(layout);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetDescriptorSetLayoutSupport(VkDevice _device, const VkDescriptorSetLayoutCreateInfo *pCreateInfo,
                                  VkDescriptorSetLayoutSupport *pSupport)
{
   uint32_t binding_count, slot_count, sampler_count;
   count_layout(pCreateInfo, &binding_count, &slot_count, &sampler_count);
   pSupport->supported = slot_count <= WVK_MAX_SET_SLOTS;
   vk_foreach_struct(stype, ext, pSupport->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_LAYOUT_SUPPORT: {
         /* The variable binding is the last: as many of it as the rest
          * leave room for. */
         VkDescriptorSetVariableDescriptorCountLayoutSupport *vs = ext;
         vs->maxVariableDescriptorCount = 0;
         const VkDescriptorSetLayoutBindingFlagsCreateInfo *flags =
            vk_find_struct_const(pCreateInfo->pNext, DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
         const VkDescriptorSetLayoutBinding *last = NULL;
         uint32_t last_index = 0;
         for (uint32_t i = 0; i < pCreateInfo->bindingCount; ++i) {
            if (!last || pCreateInfo->pBindings[i].binding > last->binding) {
               last = &pCreateInfo->pBindings[i];
               last_index = i;
            }
         }
         /* Only a binding of variable count has a maximum. */
         const bool variable = last && flags && last_index < flags->bindingCount &&
                               (flags->pBindingFlags[last_index] &
                                VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT);
         if (variable && pSupport->supported) {
            const uint32_t per = last->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK
                                    ? 0 : wvk_descriptor_type_slots(last->descriptorType);
            const uint32_t others = slot_count - (per ? last->descriptorCount * per : 1);
            /* An inline uniform block's count is its size in bytes. */
            vs->maxVariableDescriptorCount = per ? (WVK_MAX_SET_SLOTS - others) / per : 4096;
         }
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
}
