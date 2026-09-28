/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * What shaders do themselves with descriptors, for what WebGPU's objects
 * cannot hold. Each set has a record for each of its WebGPU bindings, at
 * binding WVK_SET_META_SLOT of its group (wvk_descriptor_set_layout.h), and
 * the shader reads it at the draw:
 *
 *   - texel buffers, which WebGPU does not have, are storage buffers:
 *     fetches, stores, atomics and size queries become word accesses, the
 *     format decoded and encoded as the record describes it
 *   - a view's swizzle, which WebGPU views cannot have, is applied to what
 *     is read through it
 *
 * These run after the resource variables are split, when every texture,
 * image and sampler is a variable at its (group, binding), and before
 * buffers become words, which the accesses here then do too.
 */
#include "wvk_nir.h"

#include "wvk_descriptor_set_layout.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"

/* The index operand of accesses to storage buffer (group, binding). */
static nir_def *
buffer_index(nir_builder *b, uint32_t group, uint32_t binding)
{
   nir_def *index = nir_vulkan_resource_index(b, 2, 32, nir_imm_int(b, 0), .desc_set = group,
                                              .binding = binding,
                                              .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *desc = nir_load_vulkan_descriptor(b, 2, 32, index, .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   return nir_channel(b, desc, 0);
}

/* Word `word` of the record of binding `slot` of `group`. */
nir_def *
wvk_nir_load_meta(nir_builder *b, uint32_t group, nir_def *slot, uint32_t word)
{
   nir_def *at = nir_ishl_imm(b, nir_iadd_imm(b, nir_imul_imm(b, slot, WVK_META_WORDS), word), 2);
   return nir_load_ssbo(b, 1, 32, buffer_index(b, group, WVK_SET_META_SLOT), at, .align_mul = 4,
                        .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
load_word(nir_builder *b, nir_def *buffer, nir_def *word, bool writable)
{
   return nir_load_ssbo(b, 1, 32, buffer, nir_ishl_imm(b, word, 2), .align_mul = 4,
                        .access = writable ? 0 : ACCESS_NON_WRITEABLE);
}

static bool
deref_binding(nir_src src, uint32_t *group, uint32_t *slot)
{
   nir_deref_instr *deref = nir_src_as_deref(src);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var)
      return false;
   *group = var->data.descriptor_set;
   *slot = var->data.binding;
   return true;
}

/* ---------------------------------------------------------------------- */
/* Texel buffers */

struct texel_format {
   nir_def *type, *bytes, *count, *start, *sizes, *shifts, *swizzle;
};

static struct texel_format
texel_format(nir_builder *b, uint32_t group, uint32_t slot)
{
   nir_def *s = nir_imm_int(b, slot);
   nir_def *format = wvk_nir_load_meta(b, group, s, WVK_META_TEXEL_FORMAT);
   return (struct texel_format){
      .type = nir_iand_imm(b, format, 0xff),
      .bytes = nir_ubitfield_extract_imm(b, format, 8, 8),
      .count = wvk_nir_load_meta(b, group, s, WVK_META_TEXEL_COUNT),
      .start = wvk_nir_load_meta(b, group, s, WVK_META_TEXEL_START),
      .sizes = wvk_nir_load_meta(b, group, s, WVK_META_TEXEL_SIZES),
      .shifts = wvk_nir_load_meta(b, group, s, WVK_META_TEXEL_SHIFTS),
      .swizzle = wvk_nir_load_meta(b, group, s, WVK_META_TEXEL_SWIZZLE),
   };
}

static nir_def *
is_type(nir_builder *b, const struct texel_format *f, enum wvk_texel_type type)
{
   return nir_ieq_imm(b, f->type, type);
}

/* The largest value of an unsigned `size` bits, as a float. */
static nir_def *
unorm_max(nir_builder *b, nir_def *size)
{
   return nir_u2f32(b, nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), size), -1));
}

/* B10G11R11_UFLOAT's channels: halves without their sign and low mantissa
 * bits, which nir_format_unpack_11f11f10f takes through 16-bit integers
 * WGSL does not have. */
static nir_def *
unpack_11f11f10f(nir_builder *b, nir_def *packed)
{
   nir_def *halves[3] = {
      nir_mask_shift(b, packed, 0x000007ff, 4),
      nir_mask_shift(b, packed, 0x003ff800, -7),
      nir_mask_shift(b, packed, 0xffc00000, -17),
   };
   for (unsigned i = 0; i < 3; ++i)
      halves[i] = nir_channel(b, nir_unpack_half_2x16(b, halves[i]), 0);
   return nir_vec(b, halves, 3);
}

/* Texel `index`, as RGBA of the texture's type: the channels out of their
 * words, converted as the type is, put where the format's swizzle says.
 * Nothing past the end. */
static nir_def *
texel_fetch(nir_builder *b, uint32_t group, uint32_t slot, nir_def *index, nir_alu_type dest, bool writable)
{
   const struct texel_format f = texel_format(b, group, slot);
   nir_def *buffer = buffer_index(b, group, slot);
   const nir_alu_type base = nir_alu_type_get_base_type(dest);
   nir_def *zero = base == nir_type_float ? nir_imm_float(b, 0.0f) : nir_imm_int(b, 0);
   nir_def *one = base == nir_type_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1);
   /* Before the branch: a phi is first in its block. */
   nir_def *nothing = nir_imm_zero(b, 4, 32);

   nir_push_if(b, nir_ult(b, index, f.count));
   nir_def *first_bit = nir_ishl_imm(b, nir_iadd(b, f.start, nir_imul(b, index, f.bytes)), 3);
   nir_def *channels[4];
   nir_def *first_word = NULL;
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *size = nir_ubitfield_extract_imm(b, f.sizes, 8 * c, 8);
      nir_def *bit = nir_iadd(b, first_bit, nir_ubitfield_extract_imm(b, f.shifts, 8 * c, 8));
      nir_def *word = load_word(b, buffer, nir_ushr_imm(b, bit, 5), writable);
      if (!c)
         first_word = word;
      nir_def *offset = nir_iand_imm(b, bit, 31);
      nir_def *raw = nir_ubitfield_extract(b, word, offset, size);
      nir_def *value;
      if (base == nir_type_uint) {
         value = raw;
      } else if (base == nir_type_int) {
         value = nir_ibitfield_extract(b, word, offset, size);
      } else {
         nir_def *signed_raw = nir_ibitfield_extract(b, word, offset, size);
         nir_def *unorm = nir_fdiv(b, nir_u2f32(b, raw), unorm_max(b, size));
         nir_def *snorm = nir_fmax(b, nir_fdiv(b, nir_i2f32(b, signed_raw),
                                               unorm_max(b, nir_iadd_imm(b, size, -1))),
                                   nir_imm_float(b, -1.0f));
         nir_def *half = nir_channel(b, nir_unpack_half_2x16(b, raw), 0);
         nir_def *fl = nir_bcsel(b, nir_ieq_imm(b, size, 16), half, raw);
         value = nir_bcsel(b, is_type(b, &f, WVK_TEXEL_UNORM), unorm,
                           nir_bcsel(b, is_type(b, &f, WVK_TEXEL_SNORM), snorm, fl));
      }
      /* A channel the format does not have. */
      channels[c] = nir_bcsel(b, nir_ieq_imm(b, size, 0), zero, value);
   }
   if (base == nir_type_float) {
      nir_def *packed = unpack_11f11f10f(b, first_word);
      nir_def *eleven = is_type(b, &f, WVK_TEXEL_UFLOAT_11_11_10);
      for (unsigned c = 0; c < 3; ++c)
         channels[c] = nir_bcsel(b, eleven, nir_channel(b, packed, c), channels[c]);
   }
   nir_def *out[4];
   for (unsigned i = 0; i < 4; ++i) {
      nir_def *from = nir_ubitfield_extract_imm(b, f.swizzle, 4 * i, 4);
      out[i] = nir_bcsel(b, nir_ieq_imm(b, from, 0), channels[0],
               nir_bcsel(b, nir_ieq_imm(b, from, 1), channels[1],
               nir_bcsel(b, nir_ieq_imm(b, from, 2), channels[2],
               nir_bcsel(b, nir_ieq_imm(b, from, 3), channels[3],
               nir_bcsel(b, nir_ieq_imm(b, from, 5), one, zero)))));
   }
   nir_def *texel = nir_vec(b, out, 4);
   nir_pop_if(b, NULL);
   return nir_if_phi(b, texel, nothing);
}

/* Channel `c` of a value to store: the component of RGBA the format's
 * swizzle puts it at, converted to its bits. */
static nir_def *
encode_channel(nir_builder *b, const struct texel_format *f, nir_def *value, nir_alu_type src, unsigned c,
               nir_def *size)
{
   nir_def *component = nir_imm_int(b, 0);
   bool first = true;
   for (unsigned i = 4; i-- > 0;) {
      nir_def *from = nir_ubitfield_extract_imm(b, f->swizzle, 4 * i, 4);
      nir_def *v = nir_channel(b, value, i);
      component = first ? v : nir_bcsel(b, nir_ieq_imm(b, from, c), v, component);
      first = false;
   }
   const nir_alu_type base = nir_alu_type_get_base_type(src);
   if (base != nir_type_float)
      return component;
   nir_def *max = unorm_max(b, size);
   nir_def *unorm = nir_f2u32(b, nir_fround_even(b, nir_fmul(b, nir_fsat(b, component), max)));
   nir_def *smax = unorm_max(b, nir_iadd_imm(b, size, -1));
   nir_def *snorm = nir_f2i32(b, nir_fround_even(b, nir_fmul(b, nir_fmin(b, nir_fmax(b, component,
                                                                                     nir_imm_float(b, -1.0f)),
                                                                        nir_imm_float(b, 1.0f)),
                                                            smax)));
   nir_def *half = nir_pack_half_2x16_split(b, component, nir_imm_float(b, 0.0f));
   nir_def *fl = nir_bcsel(b, nir_ieq_imm(b, size, 16), half, component);
   return nir_bcsel(b, is_type(b, f, WVK_TEXEL_UNORM), unorm,
                    nir_bcsel(b, is_type(b, f, WVK_TEXEL_SNORM), snorm, fl));
}

/* Texel `index` set to `value`: each channel into its word, whole words
 * stored, parts of one with atomics, so that neighbours written at the
 * same time stay. Nothing past the end. */
static void
texel_store(nir_builder *b, uint32_t group, uint32_t slot, nir_def *index, nir_def *value, nir_alu_type src)
{
   const struct texel_format f = texel_format(b, group, slot);
   nir_def *buffer = buffer_index(b, group, slot);
   value = nir_pad_vector_imm_int(b, value, 0, 4);

   nir_push_if(b, nir_ult(b, index, f.count));
   nir_def *first_bit = nir_ishl_imm(b, nir_iadd(b, f.start, nir_imul(b, index, f.bytes)), 3);
   nir_def *eleven = NULL;
   if (nir_alu_type_get_base_type(src) == nir_type_float)
      eleven = nir_format_pack_11f11f10f(b, nir_trim_vector(b, value, 3));
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *size = nir_ubitfield_extract_imm(b, f.sizes, 8 * c, 8);
      nir_push_if(b, nir_ine_imm(b, size, 0));
      nir_def *bit = nir_iadd(b, first_bit, nir_ubitfield_extract_imm(b, f.shifts, 8 * c, 8));
      nir_def *word = nir_ushr_imm(b, bit, 5);
      nir_def *offset = nir_iand_imm(b, bit, 31);
      nir_def *raw = encode_channel(b, &f, value, src, c, size);
      if (eleven && c == 0)
         raw = nir_bcsel(b, is_type(b, &f, WVK_TEXEL_UFLOAT_11_11_10), eleven, raw);
      nir_push_if(b, nir_ieq_imm(b, size, 32));
      nir_store_ssbo(b, raw, buffer, nir_ishl_imm(b, word, 2), .align_mul = 4);
      nir_push_else(b, NULL);
      nir_def *mask = nir_ishl(b, nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), size), -1), offset);
      nir_def *at = nir_ishl_imm(b, word, 2);
      nir_ssbo_atomic(b, 32, buffer, at, nir_inot(b, mask), .atomic_op = nir_atomic_op_iand);
      nir_ssbo_atomic(b, 32, buffer, at, nir_iand(b, nir_ishl(b, raw, offset), mask),
                      .atomic_op = nir_atomic_op_ior);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* An atomic on texel `index` of a 32-bit integer texel buffer. */
static nir_def *
texel_atomic(nir_builder *b, uint32_t group, uint32_t slot, nir_intrinsic_instr *intr, nir_def *index)
{
   const struct texel_format f = texel_format(b, group, slot);
   nir_def *buffer = buffer_index(b, group, slot);
   nir_def *nothing = nir_imm_int(b, 0);
   nir_push_if(b, nir_ult(b, index, f.count));
   nir_def *at = nir_iadd(b, f.start, nir_ishl_imm(b, index, 2));
   nir_def *result;
   if (intr->intrinsic == nir_intrinsic_image_deref_atomic_swap)
      result = nir_ssbo_atomic_swap(b, 32, buffer, at, intr->src[3].ssa, intr->src[4].ssa,
                                    .atomic_op = nir_intrinsic_atomic_op(intr));
   else
      result = nir_ssbo_atomic(b, 32, buffer, at, intr->src[3].ssa, .atomic_op = nir_intrinsic_atomic_op(intr));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, result, nothing);
}

static nir_def *
texel_count(nir_builder *b, uint32_t group, uint32_t slot, unsigned components)
{
   nir_def *count = wvk_nir_load_meta(b, group, nir_imm_int(b, slot), WVK_META_TEXEL_COUNT);
   return nir_pad_vector_imm_int(b, count, 1, components);
}

static bool
lower_texel_buffer_instr(nir_builder *b, nir_instr *instr, void *data)
{
   uint32_t group, slot;
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (tex->sampler_dim != GLSL_SAMPLER_DIM_BUF || texture < 0 ||
          !deref_binding(tex->src[texture].src, &group, &slot))
         return false;
      b->cursor = nir_before_instr(instr);
      nir_def *result;
      if (tex->op == nir_texop_txs) {
         result = texel_count(b, group, slot, tex->def.num_components);
      } else {
         const int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
         nir_def *index = coord >= 0 ? nir_channel(b, tex->src[coord].src.ssa, 0) : nir_imm_int(b, 0);
         result = nir_trim_vector(b, texel_fetch(b, group, slot, index, tex->dest_type, false),
                                  tex->def.num_components);
      }
      nir_def_replace(&tex->def, result);
      return true;
   }
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
      break;
   default:
      return false;
   }
   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_BUF || !deref_binding(intr->src[0], &group, &slot))
      return false;
   b->cursor = nir_before_instr(instr);
   if (intr->intrinsic == nir_intrinsic_image_deref_size) {
      nir_def_replace(&intr->def, texel_count(b, group, slot, intr->def.num_components));
      return true;
   }
   nir_def *index = nir_channel(b, intr->src[1].ssa, 0);
   if (intr->intrinsic == nir_intrinsic_image_deref_store) {
      texel_store(b, group, slot, index, intr->src[3].ssa, nir_intrinsic_src_type(intr));
      nir_instr_remove(instr);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_image_deref_load) {
      nir_def *texel = texel_fetch(b, group, slot, index, nir_intrinsic_dest_type(intr), true);
      nir_def_replace(&intr->def, nir_trim_vector(b, texel, intr->def.num_components));
      return true;
   }
   nir_def_replace(&intr->def, texel_atomic(b, group, slot, intr, index));
   return true;
}

bool
wvk_nir_lower_texel_buffers(nir_shader *nir)
{
   return nir_shader_instructions_pass(nir, lower_texel_buffer_instr, nir_metadata_none, NULL);
}

/* ---------------------------------------------------------------------- */
/* Swizzles */

/* One component of a swizzled texel: VkComponentSwizzle `code` of it. */
static nir_def *
swizzle_component(nir_builder *b, nir_def *texel, nir_def *code, nir_def *zero, nir_def *one)
{
   return nir_bcsel(b, nir_ieq_imm(b, code, VK_COMPONENT_SWIZZLE_R), nir_channel(b, texel, 0),
          nir_bcsel(b, nir_ieq_imm(b, code, VK_COMPONENT_SWIZZLE_G), nir_channel(b, texel, 1),
          nir_bcsel(b, nir_ieq_imm(b, code, VK_COMPONENT_SWIZZLE_B), nir_channel(b, texel, 2),
          nir_bcsel(b, nir_ieq_imm(b, code, VK_COMPONENT_SWIZZLE_A), nir_channel(b, texel, 3),
          nir_bcsel(b, nir_ieq_imm(b, code, VK_COMPONENT_SWIZZLE_ONE), one, zero)))));
}

static bool
reads_texels(const nir_tex_instr *tex)
{
   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
   case nir_texop_txf:
   case nir_texop_txf_ms:
   case nir_texop_tg4:
      return !tex->is_shadow && tex->def.num_components == 4 && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF;
   default:
      return false;
   }
}

static bool
lower_swizzle_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   uint32_t group, slot;
   if (!reads_texels(tex) || texture < 0 || !deref_binding(tex->src[texture].src, &group, &slot))
      return false;

   /* What used the texel uses the swizzled one, when there is one: the
    * branch below is in blocks of its own. */
   b->cursor = nir_after_instr(instr);
   nir_def *placeholder = nir_undef(b, tex->def.num_components, tex->def.bit_size);
   nir_def_rewrite_uses(&tex->def, placeholder);
   b->cursor = nir_after_instr(instr);
   const bool is_float = nir_alu_type_get_base_type(tex->dest_type) == nir_type_float;
   nir_def *zero = is_float ? nir_imm_float(b, 0.0f) : nir_imm_int(b, 0);
   nir_def *one = is_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1);
   nir_def *code = wvk_nir_load_meta(b, group, nir_imm_int(b, slot), WVK_META_TEX_SWIZZLE);
   nir_push_if(b, nir_ine_imm(b, code, 0));
   nir_def *swizzled;
   if (tex->op == nir_texop_tg4) {
      /* A gather is of one component: the one the swizzle takes that
       * component from, which is a gather of its own. */
      nir_def *gathered[4];
      for (unsigned c = 0; c < 4; ++c) {
         if (c == tex->component) {
            gathered[c] = &tex->def;
            continue;
         }
         nir_tex_instr *other = nir_instr_as_tex(nir_instr_clone(b->shader, instr));
         other->component = c;
         nir_builder_instr_insert(b, &other->instr);
         gathered[c] = &other->def;
      }
      nir_def *c = nir_ubitfield_extract_imm(b, code, 3 * tex->component, 3);
      nir_def *zeros = nir_imm_zero(b, 4, 32);
      nir_def *ones = nir_vec4(b, one, one, one, one);
      swizzled = nir_bcsel(b, nir_ieq_imm(b, c, VK_COMPONENT_SWIZZLE_R), gathered[0],
                 nir_bcsel(b, nir_ieq_imm(b, c, VK_COMPONENT_SWIZZLE_G), gathered[1],
                 nir_bcsel(b, nir_ieq_imm(b, c, VK_COMPONENT_SWIZZLE_B), gathered[2],
                 nir_bcsel(b, nir_ieq_imm(b, c, VK_COMPONENT_SWIZZLE_A), gathered[3],
                 nir_bcsel(b, nir_ieq_imm(b, c, VK_COMPONENT_SWIZZLE_ONE), ones, zeros)))));
   } else {
      nir_def *out[4];
      for (unsigned i = 0; i < 4; ++i)
         out[i] = swizzle_component(b, &tex->def, nir_ubitfield_extract_imm(b, code, 3 * i, 3), zero, one);
      swizzled = nir_vec(b, out, 4);
   }
   nir_pop_if(b, NULL);
   nir_def *result = nir_if_phi(b, swizzled, &tex->def);
   nir_def_rewrite_uses(placeholder, result);
   nir_instr_remove(nir_def_instr(placeholder));
   return true;
}

bool
wvk_nir_lower_view_swizzles(nir_shader *nir)
{
   return nir_shader_instructions_pass(nir, lower_swizzle_instr, nir_metadata_none, NULL);
}
