/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Colour attachments a fragment shader reads as they are, in rasterization
 * order: its tile image reads (VK_EXT_shader_tile_image), and its reads of
 * input attachments that are the pass's own colour attachments -- which
 * VK_EXT_rasterization_order_attachment_access orders, and which a copy
 * made before the draw (wvk_cmd_draw.c's snapshot_feedback) cannot be.
 * Both are Dawn's framebuffer fetch where the device has it: an input
 * named wink_color_N, which the WGSL is made @color(N) of
 * (gex_wgpu_tint.cpp), of the attachment's WebGPU format's components and
 * type, as Dawn wants.
 */
#include "wvk_nir.h"

#include <stdio.h>

#include "wvk_format.h"

#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
#include "vk_format.h"

static unsigned
wgpu_components(WGPUTextureFormat format)
{
   switch (format) {
   case WGPUTextureFormat_R8Unorm:
   case WGPUTextureFormat_R8Snorm:
   case WGPUTextureFormat_R8Uint:
   case WGPUTextureFormat_R8Sint:
   case WGPUTextureFormat_R16Unorm:
   case WGPUTextureFormat_R16Snorm:
   case WGPUTextureFormat_R16Uint:
   case WGPUTextureFormat_R16Sint:
   case WGPUTextureFormat_R16Float:
   case WGPUTextureFormat_R32Uint:
   case WGPUTextureFormat_R32Sint:
   case WGPUTextureFormat_R32Float:
      return 1;
   case WGPUTextureFormat_RG8Unorm:
   case WGPUTextureFormat_RG8Snorm:
   case WGPUTextureFormat_RG8Uint:
   case WGPUTextureFormat_RG8Sint:
   case WGPUTextureFormat_RG16Unorm:
   case WGPUTextureFormat_RG16Snorm:
   case WGPUTextureFormat_RG16Uint:
   case WGPUTextureFormat_RG16Sint:
   case WGPUTextureFormat_RG16Float:
   case WGPUTextureFormat_RG32Uint:
   case WGPUTextureFormat_RG32Sint:
   case WGPUTextureFormat_RG32Float:
      return 2;
   case WGPUTextureFormat_RG11B10Ufloat:
      return 3;
   default:
      return 4;
   }
}

unsigned
wvk_nir_fetch_components(VkFormat format)
{
   return wgpu_components(wvk_format_get(format)->texture);
}

bool
wvk_nir_fetchable(const struct wvk_physical_device *pdev, VkFormat format)
{
   const struct wvk_format *f = wvk_format_get(format);
   const uint32_t caps = wvk_format_caps(pdev, format);
   return f && f->texture != WGPUTextureFormat_Undefined && (caps & WVK_CAP_RENDER) &&
          !(caps & (WVK_CAP_DEPTH | WVK_CAP_STENCIL | WVK_CAP_PLANAR | WVK_CAP_COMPRESSED));
}

struct fetch_state {
   const uint32_t *formats;
   const uint8_t *inputs;
   const uint8_t *color_map;
   nir_variable *vars[8];
   uint32_t locations;
};

/* Attachment `rt` as it is, as `num_components` of `bit_size` bits. */
static nir_def *
fetch(nir_builder *b, struct fetch_state *s, unsigned rt, unsigned num_components, unsigned bit_size)
{
   const VkFormat format = s->formats[rt];
   const enum pipe_format pformat = vk_format_to_pipe_format(format);
   const bool uint = util_format_is_pure_uint(pformat);
   const bool sint = util_format_is_pure_sint(pformat);
   const enum glsl_base_type base = uint ? GLSL_TYPE_UINT : sint ? GLSL_TYPE_INT : GLSL_TYPE_FLOAT;
   const unsigned n = wvk_nir_fetch_components(format);
   if (!s->vars[rt]) {
      /* At a location no input has, from the top: Tint's reader wants one,
       * though @color takes its place. */
      unsigned location = 31;
      while (location > 0 && (s->locations & BITFIELD_BIT(location)))
         --location;
      s->locations |= BITFIELD_BIT(location);
      char name[16];
      snprintf(name, sizeof(name), "wink_color_%u", rt);
      nir_variable *var = nir_variable_create(b->shader, nir_var_shader_in, glsl_vector_type(base, n), name);
      var->data.location = VARYING_SLOT_VAR0 + location;
      var->data.interpolation = INTERP_MODE_FLAT;
      var->data.fb_fetch_output = true;
      s->vars[rt] = var;
   }
   nir_def *value = nir_load_var(b, s->vars[rt]);
   /* The channels the Vulkan format has not are its defaults: an RGBA8
    * texture of R5G6B5's has alpha. */
   const unsigned channels = util_format_get_nr_components(pformat);
   nir_def *c[4];
   for (unsigned i = 0; i < 4; ++i) {
      if (i < n && i < channels)
         c[i] = nir_channel(b, value, i);
      else if (i == 3)
         c[i] = base == GLSL_TYPE_FLOAT ? nir_imm_float(b, 1.0) : nir_imm_int(b, 1);
      else
         c[i] = nir_imm_int(b, 0);
   }
   value = nir_trim_vector(b, nir_vec(b, c, 4), num_components);
   if (bit_size != 32)
      value = base == GLSL_TYPE_FLOAT ? nir_f2fN(b, value, bit_size)
              : base == GLSL_TYPE_INT ? nir_i2iN(b, value, bit_size)
                                      : nir_u2uN(b, value, bit_size);
   return value;
}

/* The attachment at colour location `location`. */
static int
attachment_of(const struct fetch_state *s, unsigned location)
{
   if (!s->color_map)
      return location < 8 ? (int)location : -1;
   for (unsigned i = 0; i < 8; ++i) {
      if (s->color_map[i] == location)
         return i;
   }
   return -1;
}

/* An input attachment's variable, and which index of it a texel fetch
 * reads -- of a multisampled one, the invocation's own sample, which is
 * all a fetch is of; NULL for other fetches. */
static nir_variable *
input_attachment(nir_tex_instr *tex, int *index)
{
   const int at = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (at < 0)
      return NULL;
   if (tex->op == nir_texop_txf_ms && tex->sampler_dim == GLSL_SAMPLER_DIM_SUBPASS_MS) {
      const int sample = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
      if (sample < 0)
         return NULL;
      nir_scalar s = nir_scalar_chase_movs(nir_get_scalar(tex->src[sample].src.ssa, 0));
      if (!nir_scalar_is_intrinsic(s) || nir_scalar_intrinsic_op(s) != nir_intrinsic_load_sample_id)
         return NULL;
   } else if (tex->op != nir_texop_txf || tex->sampler_dim != GLSL_SAMPLER_DIM_SUBPASS) {
      return NULL;
   }
   nir_deref_instr *deref = nir_src_as_deref(tex->src[at].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (!var || var->data.index == NIR_VARIABLE_NO_INDEX)
      return NULL;
   *index = var->data.index;
   if (deref->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(deref->arr.index))
         return NULL;
      *index += nir_src_as_uint(deref->arr.index);
   }
   return var;
}

static int
tile_location(nir_intrinsic_instr *intr)
{
   const nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location < FRAG_RESULT_DATA0)
      return -1;
   int location = sem.location - FRAG_RESULT_DATA0;
   if (nir_src_is_const(intr->src[0]))
      location += nir_src_as_uint(intr->src[0]);
   return location;
}

static bool
subpass_var(nir_variable *var, void *data)
{
   const struct glsl_type *bare = glsl_without_array(var->type);
   return glsl_type_is_image(bare) && (glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_SUBPASS ||
                                       glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_SUBPASS_MS);
}

bool
wvk_nir_lower_framebuffer_fetch(nir_shader *nir, const uint32_t *formats, const uint8_t *inputs,
                                const uint8_t *color_map, uint8_t *tile_reads, uint8_t *input_reads)
{
   struct fetch_state s = { formats, inputs, color_map };
   nir_foreach_shader_in_variable(var, nir) {
      if (var->data.location >= VARYING_SLOT_VAR0) {
         const unsigned first = var->data.location - VARYING_SLOT_VAR0;
         const unsigned slots = glsl_count_attribute_slots(var->type, false);
         for (unsigned i = 0; i < slots && first + i < 32; ++i)
            s.locations |= BITFIELD_BIT(first + i);
      }
   }
   *tile_reads = 0;
   *input_reads = 0;
   bool progress = false;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_tile_image)
               continue;
            /* Of depth or stencil, or no attachment's: zero, which none
             * of shaderTileImageDepthReadAccess and the like reads. */
            const int location = tile_location(intr);
            if (location >= 0 && location < 8)
               *tile_reads |= nir_src_is_const(intr->src[0]) ? BITFIELD_BIT(location) : BITFIELD_RANGE(location, 8 - location);
            b.cursor = nir_before_instr(instr);
            const int rt = location >= 0 ? attachment_of(&s, location) : -1;
            nir_def *value = rt >= 0 && formats && formats[rt]
                                ? fetch(&b, &s, rt, intr->def.num_components, intr->def.bit_size)
                                : nir_imm_zero(&b, intr->def.num_components, intr->def.bit_size);
            nir_def_replace(&intr->def, value);
            progress = true;
         } else if (instr->type == nir_instr_type_tex) {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            int index;
            if (!input_attachment(tex, &index))
               continue;
            if (index < 8)
               *input_reads |= BITFIELD_BIT(index);
            int rt = -1;
            for (unsigned i = 0; inputs && i < 8; ++i) {
               if (inputs[i] == index + 1 && formats[i])
                  rt = i;
            }
            if (rt < 0)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def_replace(&tex->def, fetch(&b, &s, rt, tex->def.num_components, tex->def.bit_size));
            progress = true;
         }
      }
   }
   if (!nir_progress(progress, impl, nir_metadata_control_flow))
      return false;
   /* The tile images' variables, which nothing reads, and the input
    * attachments' that nothing reads now: no bindings of theirs. */
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   const nir_remove_dead_variables_options dead = { .can_remove_var = subpass_var };
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_image | nir_var_uniform, &dead);
   return true;
}
