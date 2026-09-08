/*
 * Copyright 2018 Collabora Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef NIR_TO_SPIRV_H
#define NIR_TO_SPIRV_H

#include <stdlib.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

#include "compiler/glsl_types.h"
#include "compiler/shader_enums.h"
#include "pipe/p_state.h"
#include "util/hash_table.h"

#define SPIRV_VERSION(major, minor) (((major) << 16) | ((minor) << 8))
#define ZINK_WORKGROUP_SIZE_X 1
#define ZINK_WORKGROUP_SIZE_Y 2
#define ZINK_WORKGROUP_SIZE_Z 3
#define ZINK_VARIABLE_SHARED_MEM 4

struct spirv_shader {
   uint32_t *words;
   size_t num_words;
   uint32_t tcs_vertices_out_word;
};

struct ntv_info {
   bool have_vulkan_memory_model;
   bool have_workgroup_memory_explicit_layout;
   bool have_float_controls2;
   bool broken_arbitary_type_const;
   bool has_demote_to_helper;
   bool is_native_vulkan; //ignore zink-isms
   /* WebGPU treats any fragment Index decoration as dual-source blending.
    * Ordinary source zero must therefore omit the optional Vulkan decoration.
    */
   bool omit_fragment_output_index_zero;
   /* WebGPU's SPIR-V ingest rejects SPV_KHR_fma.  Use the equivalent core
    * GLSL.std.450 Fma extended instruction for nir_op_ffma instead. */
   bool avoid_spv_khr_fma;
   /* Never emit OpIsNan.  WGSL has no isNan builtin -- it was removed from
    * the language -- and Tint's reader turns the instruction into a call to
    * one anyway, so the module is rejected with "unresolved call target
    * 'isNan'".  The comparisons below that emit it are an optimisation, not a
    * semantic requirement: an *ordered* float comparison is already false
    * when either operand is NaN, so OpFOrdEqual(x, x) is exactly !isnan(x)
    * and OpFUnordNotEqual(x, x) is exactly isnan(x).  Taking the general path
    * costs one instruction and says the same thing. */
   bool avoid_spv_isnan;
   /* Store gl_SampleMask as a scalar rather than constructing the
    * single-element array Vulkan mandates.
    *
    * WGSL's @builtin(sample_mask) is a u32 *scalar*, so Tint flattens the
    * SPIR-V array declaration to one -- and then rejects the array-valued
    * store that Vulkan requires against it: "cannot assign 'array<i32, 1>'
    * to 'i32'". The declaration may stay an array; only the store has to be
    * scalar. Vulkan consumers must leave this false.
    */
   bool sample_mask_scalar;
   bool have_shader_viewport_index_layer;
   bool have_shader_output_layer;
   struct {
      uint8_t flush_denorms:3; // 16, 32, 64
      uint8_t preserve_denorms:3; // 16, 32, 64
      bool denorms_32_bit_independence:1;
      bool denorms_all_independence:1;
   } float_controls;
   uint32_t spirv_version;
   unsigned bindless_set_idx;
};

struct nir_shader;
struct pipe_stream_output_info;

struct spirv_shader *
nir_to_spirv(struct nir_shader *s, const struct ntv_info *sinfo);

void
spirv_shader_delete(struct spirv_shader *s);

void
ntv_shader_prepare(struct nir_shader *nir);

static inline bool
type_is_counter(const struct glsl_type *type)
{
   return glsl_get_base_type(glsl_without_array(type)) == GLSL_TYPE_ATOMIC_UINT;
}

static inline VkDescriptorType
zink_sampler_type(const struct glsl_type *type)
{
   assert(glsl_type_is_sampler(type));
   switch (glsl_get_sampler_dim(type)) {
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_MS:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
   case GLSL_SAMPLER_DIM_BUF:
      return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
   default:
      UNREACHABLE("unimplemented");
   }
   return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

static inline VkDescriptorType
zink_image_type(const struct glsl_type *type)
{
   assert(glsl_type_is_image(type));
   switch (glsl_get_sampler_dim(type)) {
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_MS:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
   case GLSL_SAMPLER_DIM_BUF:
      return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
   default:
      UNREACHABLE("unimplemented");
   }
   return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

bool
zink_nir_lower_b2b(struct nir_shader *shader);

#endif
