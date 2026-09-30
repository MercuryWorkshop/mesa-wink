/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * What shaders do themselves with descriptors, for what WebGPU's objects
 * cannot hold. Each set has a record for each of its WebGPU bindings
 * (wvk_descriptor_set_layout.h), and the shader reads it at the draw:
 *
 *   - texel buffers, which WebGPU does not have, are storage buffers:
 *     fetches, stores, atomics and size queries become word accesses, the
 *     format decoded and encoded as the record describes it
 *   - a view's swizzle, which WebGPU views cannot have, is applied to what
 *     is read through it
 *
 * These run after the resource variables are split, when every texture,
 * image and sampler is a variable at its (group, binding), and before
 * buffers become words, which the accesses here then do too. The records
 * are in the push binding (WVK_META_OFFSET), each group's where `meta_base`
 * says: the pipeline layout's order. The buffer accesses made here are
 * bounded here, by the records, and ACCESS_IN_BOUNDS: not bounded again
 * as buffers of their own are (robustBufferAccess2).
 */
#include "wvk_nir.h"

#include "wvk_descriptor_set_layout.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
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

/* Word `word` of the record of binding `slot` of `group`: in the push
 * binding, after the driver's data, where the pipeline layout puts the
 * group's records. */
nir_def *
wvk_nir_load_meta(nir_builder *b, const uint32_t *meta, uint32_t group, nir_def *slot, uint32_t word)
{
   const uint32_t first = WVK_META_OFFSET / 4 + meta[MIN2(group, 3)] + word;
   nir_def *at = nir_ishl_imm(b, nir_iadd_imm(b, nir_imul_imm(b, slot, WVK_META_WORDS), first), 2);
   return nir_load_ssbo(b, 1, 32, buffer_index(b, WVK_DRIVER_GROUP, WVK_DRIVER_SLOT_PUSH), at, .align_mul = 4,
                        .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

static nir_def *
load_word(nir_builder *b, nir_def *buffer, nir_def *word, bool writable)
{
   return nir_load_ssbo(b, 1, 32, buffer, nir_ishl_imm(b, word, 2), .align_mul = 4,
                        .access = ACCESS_IN_BOUNDS | (writable ? 0 : ACCESS_NON_WRITEABLE));
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
   /* The driver's own textures (the copies logic ops and the depth bounds
    * test read) are no descriptors, and have no records. */
   return *group != WVK_DRIVER_GROUP || *slot < WVK_DRIVER_SLOT_BASE;
}

/* ---------------------------------------------------------------------- */
/* Texel buffers */

struct texel_format {
   nir_def *type, *bytes, *count, *start, *sizes, *shifts, *swizzle;
};

static struct texel_format
texel_format(nir_builder *b, const uint32_t *meta, uint32_t group, uint32_t slot)
{
   nir_def *s = nir_imm_int(b, slot);
   nir_def *format = wvk_nir_load_meta(b, meta, group, s, WVK_META_TEXEL_FORMAT);
   return (struct texel_format){
      .type = nir_iand_imm(b, format, 0xff),
      .bytes = nir_ubitfield_extract_imm(b, format, 8, 8),
      .count = wvk_nir_load_meta(b, meta, group, s, WVK_META_TEXEL_COUNT),
      .start = wvk_nir_load_meta(b, meta, group, s, WVK_META_TEXEL_START),
      .sizes = wvk_nir_load_meta(b, meta, group, s, WVK_META_TEXEL_SIZES),
      .shifts = wvk_nir_load_meta(b, meta, group, s, WVK_META_TEXEL_SHIFTS),
      .swizzle = wvk_nir_load_meta(b, meta, group, s, WVK_META_TEXEL_SWIZZLE),
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
texel_fetch(nir_builder *b, const uint32_t *meta, uint32_t group, uint32_t slot, nir_def *index, nir_alu_type dest, bool writable)
{
   const struct texel_format f = texel_format(b, meta, group, slot);
   nir_def *buffer = buffer_index(b, group, slot);
   const nir_alu_type base = nir_alu_type_get_base_type(dest);
   nir_def *zero = base == nir_type_float ? nir_imm_float(b, 0.0f) : nir_imm_int(b, 0);
   nir_def *one = base == nir_type_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1);
   /* Before the branch, as a phi is first in its block: past the end,
    * zero, but of a channel the format has not, what the swizzle says
    * (robustBufferAccess2's (0, 0, 1) for G, B and A). */
   nir_def *none[4];
   for (unsigned i = 0; i < 4; ++i)
      none[i] = nir_bcsel(b, nir_ieq_imm(b, nir_ubitfield_extract_imm(b, f.swizzle, 4 * i, 4), 5), one, zero);
   nir_def *nothing = nir_vec(b, none, 4);

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
texel_store(nir_builder *b, const uint32_t *meta, uint32_t group, uint32_t slot, nir_def *index, nir_def *value, nir_alu_type src)
{
   const struct texel_format f = texel_format(b, meta, group, slot);
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
      nir_store_ssbo(b, raw, buffer, nir_ishl_imm(b, word, 2), .align_mul = 4, .access = ACCESS_IN_BOUNDS);
      nir_push_else(b, NULL);
      nir_def *mask = nir_ishl(b, nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), size), -1), offset);
      nir_def *at = nir_ishl_imm(b, word, 2);
      nir_ssbo_atomic(b, 32, buffer, at, nir_inot(b, mask), .access = ACCESS_IN_BOUNDS,
                      .atomic_op = nir_atomic_op_iand);
      nir_ssbo_atomic(b, 32, buffer, at, nir_iand(b, nir_ishl(b, raw, offset), mask), .access = ACCESS_IN_BOUNDS,
                      .atomic_op = nir_atomic_op_ior);
      nir_pop_if(b, NULL);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* An atomic on texel `index` of a 32-bit integer texel buffer. */
static nir_def *
texel_atomic(nir_builder *b, const uint32_t *meta, uint32_t group, uint32_t slot, nir_intrinsic_instr *intr, nir_def *index)
{
   const struct texel_format f = texel_format(b, meta, group, slot);
   nir_def *buffer = buffer_index(b, group, slot);
   nir_def *nothing = nir_imm_int(b, 0);
   nir_push_if(b, nir_ult(b, index, f.count));
   nir_def *at = nir_iadd(b, f.start, nir_ishl_imm(b, index, 2));
   nir_def *result;
   if (intr->intrinsic == nir_intrinsic_image_deref_atomic_swap)
      result = nir_ssbo_atomic_swap(b, 32, buffer, at, intr->src[3].ssa, intr->src[4].ssa, .access = ACCESS_IN_BOUNDS,
                                    .atomic_op = nir_intrinsic_atomic_op(intr));
   else
      result = nir_ssbo_atomic(b, 32, buffer, at, intr->src[3].ssa, .access = ACCESS_IN_BOUNDS,
                               .atomic_op = nir_intrinsic_atomic_op(intr));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, result, nothing);
}

static nir_def *
texel_count(nir_builder *b, const uint32_t *meta, uint32_t group, uint32_t slot, unsigned components)
{
   nir_def *count = wvk_nir_load_meta(b, meta, group, nir_imm_int(b, slot), WVK_META_TEXEL_COUNT);
   return nir_pad_vector_imm_int(b, count, 1, components);
}

static bool
lower_texel_buffer_instr(nir_builder *b, nir_instr *instr, void *data)
{
   const uint32_t *meta = data;
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
         result = texel_count(b, meta, group, slot, tex->def.num_components);
      } else {
         const int coord = nir_tex_instr_src_index(tex, nir_tex_src_coord);
         nir_def *index = coord >= 0 ? nir_channel(b, tex->src[coord].src.ssa, 0) : nir_imm_int(b, 0);
         result = nir_trim_vector(b, texel_fetch(b, meta, group, slot, index, tex->dest_type, false),
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
      nir_def_replace(&intr->def, texel_count(b, meta, group, slot, intr->def.num_components));
      return true;
   }
   nir_def *index = nir_channel(b, intr->src[1].ssa, 0);
   if (intr->intrinsic == nir_intrinsic_image_deref_store) {
      texel_store(b, meta, group, slot, index, intr->src[3].ssa, nir_intrinsic_src_type(intr));
      nir_instr_remove(instr);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_image_deref_load) {
      nir_def *texel = texel_fetch(b, meta, group, slot, index, nir_intrinsic_dest_type(intr), true);
      nir_def_replace(&intr->def, nir_trim_vector(b, texel, intr->def.num_components));
      return true;
   }
   nir_def_replace(&intr->def, texel_atomic(b, meta, group, slot, intr, index));
   return true;
}

bool
wvk_nir_lower_texel_buffers(nir_shader *nir, const uint32_t *meta_base)
{
   return nir_shader_instructions_pass(nir, lower_texel_buffer_instr, nir_metadata_none, (void *)meta_base);
}

/* ---------------------------------------------------------------------- */
/* Images with atomics */

/* A storage image a shader does atomics on is its image's buffer of texels
 * (wvk_image.h): loads, stores, atomics and size queries are words of it,
 * where the record says the view's texels are. Out of bounds, loads are
 * zero and the rest nothing, as robustImageAccess has it. */
static nir_variable *
atomic_image_var(nir_intrinsic_instr *intr)
{
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (!var || nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF ||
       nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_MS)
      return NULL;
   return var;
}

static bool
find_atomic_images(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   /* Atomics, and the memory model's atomic loads and stores. */
   const bool atomic_access = (intr->intrinsic == nir_intrinsic_image_deref_load ||
                               intr->intrinsic == nir_intrinsic_image_deref_store) &&
                              (nir_intrinsic_access(intr) & ACCESS_ATOMIC);
   if (intr->intrinsic != nir_intrinsic_image_deref_atomic && intr->intrinsic != nir_intrinsic_image_deref_atomic_swap &&
       !atomic_access)
      return false;
   nir_variable *var = atomic_image_var(intr);
   if (var)
      _mesa_set_add(data, var);
   return false;
}

struct atomic_images {
   const uint32_t *meta;
   struct set *vars;
};

static bool
lower_atomic_image_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct atomic_images *state = data;
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
      break;
   default:
      return false;
   }
   nir_variable *var = atomic_image_var(intr);
   if (!var || !_mesa_set_search(state->vars, var))
      return false;
   const uint32_t group = var->data.descriptor_set, slot = var->data.binding;
   const uint32_t *meta = state->meta;
   b->cursor = nir_before_instr(&intr->instr);
   if (intr->intrinsic == nir_intrinsic_image_deref_samples) {
      nir_def_replace(&intr->def, nir_imm_int(b, 1));
      return true;
   }

   nir_def *s = nir_imm_int(b, slot);
   nir_def *width = wvk_nir_load_meta(b, meta, group, s, WVK_META_IMG_WIDTH);
   nir_def *height = wvk_nir_load_meta(b, meta, group, s, WVK_META_IMG_HEIGHT);
   nir_def *layers = wvk_nir_load_meta(b, meta, group, s, WVK_META_IMG_LAYERS);
   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool array = nir_intrinsic_image_array(intr);
   if (intr->intrinsic == nir_intrinsic_image_deref_size) {
      nir_def *size[3] = { width, height, layers };
      if (dim == GLSL_SAMPLER_DIM_1D)
         size[1] = layers;
      else if (dim == GLSL_SAMPLER_DIM_CUBE)
         size[2] = nir_udiv_imm(b, layers, 6);
      nir_def_replace(&intr->def, nir_vec(b, size, intr->def.num_components));
      return true;
   }

   /* The texel, and whether it is in the view. */
   nir_def *coord = intr->src[1].ssa;
   nir_def *x = nir_channel(b, coord, 0);
   nir_def *y = dim == GLSL_SAMPLER_DIM_1D ? nir_imm_int(b, 0) : nir_channel(b, coord, 1);
   nir_def *z = nir_imm_int(b, 0);
   if (dim == GLSL_SAMPLER_DIM_1D && array)
      z = nir_channel(b, coord, 1);
   else if (array || dim == GLSL_SAMPLER_DIM_3D || dim == GLSL_SAMPLER_DIM_CUBE)
      z = nir_channel(b, coord, 2);
   nir_def *inside = nir_iand(b, nir_iand(b, nir_ult(b, x, width), nir_ult(b, y, height)), nir_ult(b, z, layers));
   nir_def *word = nir_iadd(b, wvk_nir_load_meta(b, meta, group, s, WVK_META_IMG_BASE),
                            nir_iadd(b, x, nir_iadd(b, nir_imul(b, y, wvk_nir_load_meta(b, meta, group, s, WVK_META_IMG_ROW)),
                                                    nir_imul(b, z, wvk_nir_load_meta(b, meta, group, s, WVK_META_IMG_LAYER)))));
   nir_def *at = nir_ishl_imm(b, word, 2);
   nir_def *buffer = buffer_index(b, group, slot);
   nir_def *zero = nir_imm_int(b, 0);

   if (intr->intrinsic == nir_intrinsic_image_deref_store) {
      nir_push_if(b, inside);
      nir_store_ssbo(b, nir_channel(b, intr->src[3].ssa, 0), buffer, at, .align_mul = 4, .access = ACCESS_IN_BOUNDS);
      nir_pop_if(b, NULL);
      nir_instr_remove(&intr->instr);
      return true;
   }

   nir_push_if(b, inside);
   nir_def *value;
   if (intr->intrinsic == nir_intrinsic_image_deref_load)
      value = nir_load_ssbo(b, 1, 32, buffer, at, .align_mul = 4, .access = ACCESS_IN_BOUNDS);
   else if (intr->intrinsic == nir_intrinsic_image_deref_atomic_swap)
      value = nir_ssbo_atomic_swap(b, 32, buffer, at, intr->src[3].ssa, intr->src[4].ssa, .access = ACCESS_IN_BOUNDS,
                                   .atomic_op = nir_intrinsic_atomic_op(intr));
   else
      value = nir_ssbo_atomic(b, 32, buffer, at, intr->src[3].ssa, .access = ACCESS_IN_BOUNDS,
                              .atomic_op = nir_intrinsic_atomic_op(intr));
   nir_pop_if(b, NULL);
   value = nir_if_phi(b, value, zero);
   if (intr->intrinsic == nir_intrinsic_image_deref_load) {
      /* One channel: the rest are 0, and alpha 1. */
      const bool is_float = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) == nir_type_float;
      nir_def *texel[4] = { value, zero, zero, is_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1) };
      value = nir_vec(b, texel, intr->def.num_components);
   }
   nir_def_replace(&intr->def, value);
   return true;
}

bool
wvk_nir_lower_image_atomics(nir_shader *nir, const uint32_t *meta_base)
{
   struct atomic_images state = { .meta = meta_base, .vars = _mesa_pointer_set_create(NULL) };
   nir_shader_intrinsics_pass(nir, find_atomic_images, nir_metadata_all, state.vars);
   bool progress = false;
   if (state.vars->entries)
      progress = nir_shader_intrinsics_pass(nir, lower_atomic_image_instr, nir_metadata_none, &state);
   _mesa_set_destroy(state.vars, NULL);
   return progress;
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
   const uint32_t *meta = data;
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
   nir_def *code = wvk_nir_load_meta(b, meta, group, nir_imm_int(b, slot), WVK_META_TEX_SWIZZLE);
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
wvk_nir_lower_view_swizzles(nir_shader *nir, const uint32_t *meta_base)
{
   return nir_shader_instructions_pass(nir, lower_swizzle_instr, nir_metadata_none, (void *)meta_base);
}

/* ---------------------------------------------------------------------- */
/* Samplers */

static nir_def *
sampler_word(nir_builder *b, const uint32_t *meta, uint32_t group, uint32_t slot, unsigned word)
{
   return wvk_nir_load_meta(b, meta, group, nir_imm_int(b, slot), word);
}

static nir_def *
address_mode(nir_builder *b, nir_def *flags, unsigned axis)
{
   return nir_ubitfield_extract_imm(b, flags, 3 * axis, 3);
}

static nir_def *
flag(nir_builder *b, nir_def *flags, uint32_t bit)
{
   return nir_test_mask(b, flags, bit);
}

static nir_tex_instr *
texture_query(nir_builder *b, nir_tex_instr *tex, nir_texop op, nir_def *lod)
{
   const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   nir_tex_instr *q = nir_tex_instr_create(b->shader, lod ? 2 : 1);
   q->op = op;
   /* As the texture is: an integer cube's is a 2D array of its faces. */
   const struct glsl_type *type = glsl_without_array(nir_src_as_deref(tex->src[texture].src)->type);
   q->sampler_dim = glsl_get_sampler_dim(type);
   q->is_array = glsl_sampler_type_is_array(type);
   q->dest_type = nir_type_int32;
   q->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, tex->src[texture].src.ssa);
   if (lod)
      q->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_def_init(&q->instr, &q->def, op == nir_texop_txs ? nir_tex_instr_dest_size(q) : 1, 32);
   nir_builder_instr_insert(b, &q->instr);
   return q;
}

/* Vulkan's depth comparison of `ref` with `value`: VkCompareOp `op`. */
static nir_def *
depth_compare(nir_builder *b, nir_def *op, nir_def *ref, nir_def *value)
{
   nir_def *lt = nir_flt(b, ref, value), *eq = nir_feq(b, ref, value), *gt = nir_flt(b, value, ref);
   nir_def *pass = nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_LESS), lt,
                   nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_EQUAL), eq,
                   nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_LESS_OR_EQUAL), nir_ior(b, lt, eq),
                   nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_GREATER), gt,
                   nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_NOT_EQUAL), nir_inot(b, eq),
                   nir_bcsel(b, nir_ieq_imm(b, op, VK_COMPARE_OP_GREATER_OR_EQUAL), nir_ior(b, gt, eq),
                   nir_ieq_imm(b, op, VK_COMPARE_OP_ALWAYS)))))));
   return nir_b2f32(b, pass);
}

/* The sampler's LOD bias, which WebGPU's samplers have none of: in the
 * sample's own bias or LOD, or its gradients' scale. A comparison has no
 * way to take one. */
static void
apply_lod_bias(nir_builder *b, nir_tex_instr *tex, nir_def *bias)
{
   if (tex->is_shadow)
      return;
   int idx;
   switch (tex->op) {
   case nir_texop_tex:
      tex->op = nir_texop_txb;
      nir_tex_instr_add_src(tex, nir_tex_src_bias, bias);
      break;
   case nir_texop_txb:
      idx = nir_tex_instr_src_index(tex, nir_tex_src_bias);
      nir_src_rewrite(&tex->src[idx].src, nir_fadd(b, tex->src[idx].src.ssa, bias));
      break;
   case nir_texop_txl:
      idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_src_rewrite(&tex->src[idx].src, nir_fadd(b, tex->src[idx].src.ssa, bias));
      break;
   case nir_texop_txd: {
      nir_def *scale = nir_fexp2(b, bias);
      idx = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
      nir_src_rewrite(&tex->src[idx].src, nir_fmul(b, tex->src[idx].src.ssa, scale));
      idx = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
      nir_src_rewrite(&tex->src[idx].src, nir_fmul(b, tex->src[idx].src.ssa, scale));
      break;
   }
   default:
      break;
   }
}

/* The border colour as a texel of the view: channels its format lacks are
 * 0, alpha 1 ("Conversion to RGBA"). */
static void
border_texel(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot, bool is_float, nir_def *out[4])
{
   const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   uint32_t tex_group, tex_slot;
   nir_def *channels = nir_imm_int(b, 0xf);
   if (texture >= 0 && deref_binding(tex->src[texture].src, &tex_group, &tex_slot))
      channels = wvk_nir_load_meta(b, meta, tex_group, nir_imm_int(b, tex_slot), WVK_META_TEX_CHANNELS);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *missing = c == 3 ? (is_float ? nir_imm_float(b, 1.0f) : nir_imm_int(b, 1)) : nir_imm_int(b, 0);
      out[c] = nir_bcsel(b, nir_test_mask(b, channels, 1u << c),
                         sampler_word(b, meta, group, slot, WVK_META_SAMPLER_BORDER + c), missing);
   }
}

/* The LOD a sample selects: given, or from the derivatives or gradients
 * of the coordinates in texels of the base level; with the sampler's bias
 * and LOD range. Before the bias went into the instruction. */
static nir_def *
sample_lambda_raw(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot,
                  nir_def *coord, unsigned spatial)
{
   nir_def *lambda;
   const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (tex->op == nir_texop_txl && lod_idx >= 0) {
      lambda = tex->src[lod_idx].src.ssa;
   } else if (tex->op == nir_texop_txd || (b->shader->info.stage == MESA_SHADER_FRAGMENT &&
                                           (tex->op == nir_texop_tex || tex->op == nir_texop_txb ||
                                            tex->op == nir_texop_lod))) {
      nir_def *base = &texture_query(b, tex, nir_texop_txs, nir_imm_int(b, 0))->def;
      nir_def *dx, *dy;
      if (tex->op == nir_texop_txd) {
         dx = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddx)].src.ssa;
         dy = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddy)].src.ssa;
      } else {
         dx = nir_ddx(b, nir_trim_vector(b, coord, spatial));
         dy = nir_ddy(b, nir_trim_vector(b, coord, spatial));
      }
      nir_def *rho;
      if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         /* A face's coordinates are the direction over its major axis,
          * from -1 to 1 across the face's size. */
         nir_def *dir = nir_trim_vector(b, coord, 3);
         nir_def *major = nir_fmax(b, nir_fmax(b, nir_fabs(b, nir_channel(b, dir, 0)), nir_fabs(b, nir_channel(b, dir, 1))),
                                   nir_fabs(b, nir_channel(b, dir, 2)));
         nir_def *scale = nir_fdiv(b, nir_fmul_imm(b, nir_i2f32(b, nir_channel(b, base, 0)), 0.5),
                                   nir_fmax(b, major, nir_imm_float(b, 1e-20f)));
         rho = nir_fmul(b, nir_fmax(b, nir_fast_length(b, dx), nir_fast_length(b, dy)), scale);
      } else {
         nir_def *scale = nir_i2f32(b, nir_trim_vector(b, base, spatial));
         rho = nir_fmax(b, nir_fast_length(b, nir_fmul(b, dx, scale)), nir_fast_length(b, nir_fmul(b, dy, scale)));
      }
      lambda = nir_flog2(b, nir_fmax(b, rho, nir_imm_float(b, 1e-20f)));
      const int bias_idx = nir_tex_instr_src_index(tex, nir_tex_src_bias);
      if (tex->op == nir_texop_txb && bias_idx >= 0)
         lambda = nir_fadd(b, lambda, tex->src[bias_idx].src.ssa);
   } else {
      lambda = nir_imm_float(b, 0.0f);
   }
   return nir_fadd(b, lambda, sampler_word(b, meta, group, slot, WVK_META_SAMPLER_BIAS));
}

static nir_def *
sample_lambda(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot,
              nir_def *coord, unsigned spatial)
{
   nir_def *lambda = sample_lambda_raw(b, meta, tex, group, slot, coord, spatial);
   return nir_fclamp(b, lambda, sampler_word(b, meta, group, slot, WVK_META_SAMPLER_MIN_LOD),
                     sampler_word(b, meta, group, slot, WVK_META_SAMPLER_MAX_LOD));
}

/* Sampling a float texture: the bias, and where the sampler clamps to the
 * border, mirrors once or takes texels, the coordinates first and the
 * result after. The WebGPU sampler clamps to the edge on those axes. A
 * border axis weighs the border in by how much of the filter's footprint
 * is outside, at the level's size: exact for nearest and for bilinear on
 * one level. */
/* textureQueryLod, which WGSL has not: the LOD with the sampler's bias,
 * and the level that would be sampled -- within the sampler's range and
 * the view's levels, and the nearest level unless mipmaps are linear. */
/* A sample's least LOD (shaderResourceMinLod), which WGSL has no way to
 * give: an explicit LOD raised to it, or gradients scaled up to it --
 * the sampler's bias, which apply_lod_bias adds after, left out. */
static void
lower_min_lod(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot)
{
   const int idx = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   if (idx < 0)
      return;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *min_lod = tex->src[idx].src.ssa;
   nir_tex_instr_remove_src(tex, idx);
   if (tex->op == nir_texop_tg4)
      return;
   nir_def *sampler_bias = sampler_word(b, meta, group, slot, WVK_META_SAMPLER_BIAS);
   const int bias_idx = nir_tex_instr_src_index(tex, nir_tex_src_bias);
   nir_def *bias = tex->op == nir_texop_txb && bias_idx >= 0 ? tex->src[bias_idx].src.ssa : nir_imm_float(b, 0.0f);

   const bool implicit = tex->op == nir_texop_tex || tex->op == nir_texop_txb;
   if (tex->op == nir_texop_txl || (implicit && b->shader->info.stage != MESA_SHADER_FRAGMENT)) {
      /* No derivatives outside fragment shaders: the LOD given, or 0. */
      const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *lod = lod_idx >= 0 ? tex->src[lod_idx].src.ssa : bias;
      nir_def *raised = nir_fsub(b, nir_fmax(b, nir_fadd(b, lod, sampler_bias), min_lod), sampler_bias);
      if (lod_idx >= 0) {
         nir_src_rewrite(&tex->src[lod_idx].src, raised);
      } else {
         if (bias_idx >= 0)
            nir_tex_instr_remove_src(tex, bias_idx);
         tex->op = nir_texop_txl;
         nir_tex_instr_add_src(tex, nir_tex_src_lod, raised);
      }
      return;
   }
   if (!implicit && tex->op != nir_texop_txd)
      return;

   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   const unsigned spatial = tex->coord_components - (tex->is_array ? 1 : 0);
   nir_def *lambda = sample_lambda_raw(b, meta, tex, group, slot, coord, spatial);
   nir_def *scale = nir_fexp2(b, nir_fadd(b, bias, nir_fmax(b, nir_fsub(b, min_lod, lambda), nir_imm_float(b, 0.0f))));
   if (tex->op == nir_texop_txd) {
      const int dx = nir_tex_instr_src_index(tex, nir_tex_src_ddx);
      nir_src_rewrite(&tex->src[dx].src, nir_fmul(b, tex->src[dx].src.ssa, scale));
      const int dy = nir_tex_instr_src_index(tex, nir_tex_src_ddy);
      nir_src_rewrite(&tex->src[dy].src, nir_fmul(b, tex->src[dy].src.ssa, scale));
      return;
   }
   nir_def *spatial_coord = nir_trim_vector(b, coord, spatial);
   nir_def *dx = nir_fmul(b, nir_ddx(b, spatial_coord), scale);
   nir_def *dy = nir_fmul(b, nir_ddy(b, spatial_coord), scale);
   if (bias_idx >= 0)
      nir_tex_instr_remove_src(tex, bias_idx);
   tex->op = nir_texop_txd;
   nir_tex_instr_add_src(tex, nir_tex_src_ddx, dx);
   nir_tex_instr_add_src(tex, nir_tex_src_ddy, dy);
}

static bool
lower_query_lod(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot)
{
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (b->shader->info.stage != MESA_SHADER_FRAGMENT || coord_idx < 0)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   const unsigned spatial = glsl_get_sampler_dim_coordinate_components(tex->sampler_dim);
   nir_def *lambda = sample_lambda_raw(b, meta, tex, group, slot, tex->src[coord_idx].src.ssa, spatial);
   nir_def *flags = sampler_word(b, meta, group, slot, WVK_META_SAMPLER_FLAGS);
   nir_def *d = nir_fclamp(b, lambda, sampler_word(b, meta, group, slot, WVK_META_SAMPLER_MIN_LOD),
                           sampler_word(b, meta, group, slot, WVK_META_SAMPLER_MAX_LOD));
   d = nir_bcsel(b, flag(b, flags, WVK_SAMPLER_MIP_LINEAR), d, nir_ffloor(b, nir_fadd_imm(b, d, 0.5)));
   nir_def *levels = nir_i2f32(b, &texture_query(b, tex, nir_texop_query_levels, NULL)->def);
   nir_def *level = nir_fclamp(b, d, nir_imm_float(b, 0.0f), nir_fadd_imm(b, levels, -1.0f));
   nir_def_replace(&tex->def, nir_trim_vector(b, nir_vec2(b, level, lambda), tex->def.num_components));
   return true;
}

static nir_def *wrap_texel(nir_builder *b, nir_def *t, nir_def *extent, nir_def *mode, nir_def **outside);

/* A sample of a sampler that reduces by the minimum or maximum
 * (samplerFilterMinmax), which WebGPU's samplers do not: the texels of the
 * filter's footprint of weight not zero -- at the level the LOD selects,
 * or the two, of linear mipmaps -- fetched, each wrapped as the sampler
 * says or the border, and their least or greatest, component by
 * component. */
static nir_def *
reduce_sample(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot, nir_def *flags)
{
   const int texture_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   const unsigned spatial = tex->coord_components - (tex->is_array ? 1 : 0);
   nir_def *lambda = sample_lambda(b, meta, tex, group, slot, coord, spatial);
   nir_def *is_max = flag(b, flags, WVK_SAMPLER_REDUCE_MAX);
   nir_def *zero = nir_imm_float(b, 0.0f);
   nir_def *minified = nir_flt(b, zero, lambda);
   nir_def *linear = nir_bcsel(b, minified, flag(b, flags, WVK_SAMPLER_MIN_LINEAR),
                               flag(b, flags, WVK_SAMPLER_MAG_LINEAR));
   nir_def *mip_linear = nir_iand(b, minified, flag(b, flags, WVK_SAMPLER_MIP_LINEAR));
   nir_def *last = nir_iadd_imm(b, &texture_query(b, tex, nir_texop_query_levels, NULL)->def, -1);
   nir_def *l = nir_fmax(b, lambda, zero);
   nir_def *first = nir_f2i32(b, nir_bcsel(b, mip_linear, nir_ffloor(b, l), nir_fround_even(b, l)));
   first = nir_bcsel(b, minified, nir_imin(b, first, last), nir_imm_int(b, 0));
   nir_def *levels[2] = { first, nir_imin(b, nir_iadd_imm(b, first, 1), last) };
   nir_def *counts[2] = { nir_imm_true(b), nir_iand(b, mip_linear, nir_fneu_imm(b, nir_ffract(b, l), 0.0)) };

   nir_def *border[4];
   border_texel(b, meta, tex, group, slot, true, border);
   nir_def *border_value = nir_vec(b, border, 4);
   nir_def *inf = nir_imm_float(b, INFINITY);
   nir_def *result = nir_bcsel(b, is_max, nir_fneg(b, nir_vec4(b, inf, inf, inf, inf)), nir_vec4(b, inf, inf, inf, inf));
   for (unsigned k = 0; k < 2; ++k) {
      nir_def *size = &texture_query(b, tex, nir_texop_txs, levels[k])->def;
      /* Each axis's first texel, and its second where the filter is
       * linear and it weighs anything. */
      nir_def *texel[3][2], *out[3][2], *second[3];
      for (unsigned i = 0; i < spatial; ++i) {
         nir_def *extent = nir_channel(b, size, i);
         nir_def *c = nir_channel(b, coord, i);
         nir_def *u = nir_bcsel(b, flag(b, flags, WVK_SAMPLER_UNNORMALIZED), c, nir_fmul(b, c, nir_i2f32(b, extent)));
         nir_def *shifted = nir_fadd_imm(b, u, -0.5);
         nir_def *t0 = nir_f2i32(b, nir_ffloor(b, nir_bcsel(b, linear, shifted, u)));
         second[i] = nir_iand(b, linear, nir_fneu_imm(b, nir_ffract(b, shifted), 0.0));
         if (offset_idx >= 0 && i < tex->src[offset_idx].src.ssa->num_components)
            t0 = nir_iadd(b, t0, nir_channel(b, tex->src[offset_idx].src.ssa, i));
         for (unsigned j = 0; j < 2; ++j) {
            out[i][j] = nir_imm_false(b);
            texel[i][j] = wrap_texel(b, nir_iadd_imm(b, t0, j), extent, address_mode(b, flags, i), &out[i][j]);
         }
      }
      nir_def *layer = NULL;
      if (tex->is_array)
         layer = nir_imin(b, nir_imax(b, nir_f2i32(b, nir_fround_even(b, nir_channel(b, coord, spatial))),
                                      nir_imm_int(b, 0)),
                          nir_iadd_imm(b, nir_channel(b, size, spatial), -1));
      for (unsigned q = 0; q < (1u << spatial); ++q) {
         nir_def *comps[4];
         nir_def *counted = counts[k], *outside = nir_imm_false(b);
         for (unsigned i = 0; i < spatial; ++i) {
            const unsigned j = (q >> i) & 1;
            comps[i] = texel[i][j];
            outside = nir_ior(b, outside, out[i][j]);
            if (j)
               counted = nir_iand(b, counted, second[i]);
         }
         if (layer)
            comps[spatial] = layer;
         nir_tex_instr *txf = nir_tex_instr_create(b->shader, 3);
         txf->op = nir_texop_txf;
         txf->sampler_dim = tex->sampler_dim;
         txf->is_array = tex->is_array;
         txf->dest_type = tex->dest_type;
         txf->coord_components = tex->coord_components;
         txf->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, tex->src[texture_idx].src.ssa);
         txf->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_vec(b, comps, tex->coord_components));
         txf->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, levels[k]);
         nir_def_init(&txf->instr, &txf->def, 4, 32);
         nir_builder_instr_insert(b, &txf->instr);
         nir_def *value = nir_bcsel(b, outside, border_value, &txf->def);
         nir_def *reduced = nir_bcsel(b, is_max, nir_fmax(b, result, value), nir_fmin(b, result, value));
         result = nir_bcsel(b, counted, reduced, result);
      }
   }
   return nir_trim_vector(b, result, tex->def.num_components);
}

/* A cube's reduction: linear, the 2x2 texels its base level's gathers
 * have, across edges as the hardware finds them -- which is the footprint
 * of a sample of level 0; nearest, the texel WebGPU's nearest sampler has
 * already. */
static nir_def *
reduce_cube_sample(nir_builder *b, nir_tex_instr *tex, nir_def *flags)
{
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   nir_def *is_max = flag(b, flags, WVK_SAMPLER_REDUCE_MAX);
   nir_def *comps[4];
   for (unsigned c = 0; c < 4; ++c) {
      nir_tex_instr *tg4 = nir_tex_instr_create(b->shader, 3);
      tg4->op = nir_texop_tg4;
      tg4->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
      tg4->is_array = tex->is_array;
      tg4->dest_type = tex->dest_type;
      tg4->coord_components = tex->coord_components;
      tg4->component = c;
      tg4->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref,
                                        tex->src[nir_tex_instr_src_index(tex, nir_tex_src_texture_deref)].src.ssa);
      tg4->src[1] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref,
                                        tex->src[nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref)].src.ssa);
      tg4->src[2] = nir_tex_src_for_ssa(nir_tex_src_coord, tex->src[coord_idx].src.ssa);
      nir_def_init(&tg4->instr, &tg4->def, 4, 32);
      nir_builder_instr_insert(b, &tg4->instr);
      nir_def *v = &tg4->def;
      nir_def *lo = nir_fmin(b, nir_fmin(b, nir_channel(b, v, 0), nir_channel(b, v, 1)),
                             nir_fmin(b, nir_channel(b, v, 2), nir_channel(b, v, 3)));
      nir_def *hi = nir_fmax(b, nir_fmax(b, nir_channel(b, v, 0), nir_channel(b, v, 1)),
                             nir_fmax(b, nir_channel(b, v, 2), nir_channel(b, v, 3)));
      comps[c] = nir_bcsel(b, is_max, hi, lo);
   }
   return nir_trim_vector(b, nir_vec(b, comps, 4), tex->def.num_components);
}

/* Around a float sample, where its sampler reduces: the reduction's result
 * instead of the average WebGPU filters to. */
static void
lower_reduction(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot)
{
   if (tex->is_shadow || tex->def.bit_size != 32 || nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref) < 0 ||
       nir_tex_instr_src_index(tex, nir_tex_src_texture_deref) < 0 || nir_tex_instr_src_index(tex, nir_tex_src_coord) < 0 ||
       (tex->op != nir_texop_tex && tex->op != nir_texop_txb && tex->op != nir_texop_txl && tex->op != nir_texop_txd))
      return;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *flags = sampler_word(b, meta, group, slot, WVK_META_SAMPLER_FLAGS);
   nir_def *reduces = nir_test_mask(b, flags, WVK_SAMPLER_REDUCE_MIN | WVK_SAMPLER_REDUCE_MAX);
   const bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   /* A cube's nearest texel is the sample's own. */
   if (cube) {
      const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      nir_def *lambda = sample_lambda(b, meta, tex, group, slot, tex->src[coord_idx].src.ssa, 3);
      nir_def *linear = nir_bcsel(b, nir_flt(b, nir_imm_float(b, 0.0f), lambda),
                                  flag(b, flags, WVK_SAMPLER_MIN_LINEAR), flag(b, flags, WVK_SAMPLER_MAG_LINEAR));
      reduces = nir_iand(b, reduces, linear);
   }
   nir_push_if(b, reduces);
   nir_def *reduced = cube ? reduce_cube_sample(b, tex, flags)
                           : reduce_sample(b, meta, tex, group, slot, flags);
   nir_pop_if(b, NULL);
   reduced = nir_if_phi(b, reduced, nir_undef(b, reduced->num_components, 32));
   b->cursor = nir_after_instr(&tex->instr);
   nir_def *placeholder = nir_undef(b, tex->def.num_components, tex->def.bit_size);
   nir_def_rewrite_uses(&tex->def, placeholder);
   nir_def *value = nir_bcsel(b, reduces, reduced, &tex->def);
   nir_def_rewrite_uses(placeholder, value);
   nir_instr_remove(nir_def_instr(placeholder));
}

static bool
lower_float_sample(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot)
{
   lower_reduction(b, meta, tex, group, slot);
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *flags = sampler_word(b, meta, group, slot, WVK_META_SAMPLER_FLAGS);
   /* What the emulation below needs of the LOD, before the bias is in. */
   const int coord_at = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const bool emulable = tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE && tex->op != nir_texop_lod && coord_at >= 0 &&
                         tex->op != nir_texop_tg4;
   nir_def *lambda = NULL;
   if (emulable) {
      nir_def *none = nir_imm_float(b, 0.0f);
      nir_push_if(b, flag(b, flags, WVK_SAMPLER_EMULATE));
      nir_def *l = sample_lambda(b, meta, tex, group, slot, tex->src[coord_at].src.ssa,
                                 tex->coord_components - (tex->is_array ? 1 : 0));
      nir_pop_if(b, NULL);
      lambda = nir_if_phi(b, l, none);
   }
   apply_lod_bias(b, tex, nir_fclamp(b, sampler_word(b, meta, group, slot, WVK_META_SAMPLER_BIAS),
                                     nir_imm_float(b, -WVK_MAX_SAMPLER_LOD_BIAS),
                                     nir_imm_float(b, WVK_MAX_SAMPLER_LOD_BIAS)));
   /* A cube's address modes are the cube's. */
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE || tex->op == nir_texop_lod || coord_idx < 0)
      return true;

   nir_def *coord = tex->src[coord_idx].src.ssa;
   const unsigned spatial = tex->coord_components - (tex->is_array ? 1 : 0);
   nir_def *one = nir_imm_float(b, 1.0f);
   nir_push_if(b, flag(b, flags, WVK_SAMPLER_EMULATE));
   /* Minified, the minification filter at the level the LOD selects;
    * magnified, the magnification filter at the base. */
   nir_def *minified = lambda ? nir_flt(b, nir_imm_float(b, 0.0f), lambda) : nir_imm_false(b);
   nir_def *level = nir_imm_int(b, 0);
   if (lambda) {
      nir_def *levels = &texture_query(b, tex, nir_texop_query_levels, NULL)->def;
      nir_def *chosen = nir_bcsel(b, flag(b, flags, WVK_SAMPLER_MIP_LINEAR), nir_ffloor(b, lambda),
                                  nir_fround_even(b, lambda));
      level = nir_imin(b, nir_imax(b, nir_f2i32(b, chosen), nir_imm_int(b, 0)), nir_iadd_imm(b, levels, -1));
      level = nir_bcsel(b, minified, level, nir_imm_int(b, 0));
   }
   nir_def *size = &texture_query(b, tex, nir_texop_txs, level)->def;
   nir_def *comps[4];
   for (unsigned i = 0; i < tex->coord_components; ++i)
      comps[i] = nir_channel(b, coord, i);
   /* An offset is the sampler's to wrap too: in the coordinates, and out
    * again after, where the sample adds it. */
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   nir_def *weight = one;
   for (unsigned i = 0; i < spatial; ++i) {
      nir_def *extent = nir_i2f32(b, nir_channel(b, size, i));
      nir_def *mode = address_mode(b, flags, i);
      nir_def *c = nir_bcsel(b, flag(b, flags, WVK_SAMPLER_UNNORMALIZED), nir_fdiv(b, comps[i], extent), comps[i]);
      nir_def *offset = offset_idx >= 0 && i < tex->src[offset_idx].src.ssa->num_components
                           ? nir_fdiv(b, nir_i2f32(b, nir_channel(b, tex->src[offset_idx].src.ssa, i)), extent)
                           : NULL;
      if (offset)
         c = nir_fadd(b, c, offset);
      c = nir_bcsel(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE), nir_fabs(b, c), c);
      nir_def *border = nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);
      nir_def *linear = nir_bcsel(b, minified, flag(b, flags, WVK_SAMPLER_MIN_LINEAR),
                                  flag(b, flags, WVK_SAMPLER_MAG_LINEAR));
      nir_def *texel = nir_fmul(b, c, extent);
      nir_def *w_linear = nir_fsat(b, nir_fadd_imm(b, nir_fmin(b, texel, nir_fsub(b, extent, texel)), 0.5));
      nir_def *w_nearest = nir_b2f32(b, nir_iand(b, nir_fge_imm(b, c, 0.0), nir_flt_imm(b, c, 1.0)));
      weight = nir_fmul(b, weight, nir_bcsel(b, border, nir_bcsel(b, linear, w_linear, w_nearest), one));
      nir_def *half = nir_fdiv(b, nir_imm_float(b, 0.5f), extent);
      nir_def *inset = nir_fmin(b, nir_fmax(b, c, half), nir_fsub(b, one, half));
      comps[i] = nir_bcsel(b, nir_iand(b, border, linear), inset, c);
      if (offset)
         comps[i] = nir_fsub(b, comps[i], offset);
   }
   nir_def *adjusted = nir_vec(b, comps, tex->coord_components);
   nir_pop_if(b, NULL);
   nir_def *new_coord = nir_if_phi(b, adjusted, coord);
   weight = nir_if_phi(b, weight, one);
   nir_src_rewrite(&tex->src[coord_idx].src, new_coord);

   /* After: the border, where the footprint was out. */
   b->cursor = nir_after_instr(&tex->instr);
   nir_def *placeholder = nir_undef(b, tex->def.num_components, tex->def.bit_size);
   nir_def_rewrite_uses(&tex->def, placeholder);
   b->cursor = nir_after_instr(&tex->instr);
   nir_push_if(b, nir_fneu_imm(b, weight, 1.0));
   nir_def *border[4];
   border_texel(b, meta, tex, group, slot, true, border);
   nir_def *value;
   if (tex->is_shadow) {
      const int ref = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
      value = depth_compare(b, nir_ubitfield_extract_imm(b, flags, WVK_SAMPLER_COMPARE_SHIFT, 3), tex->src[ref].src.ssa,
                            nir_fsat(b, border[0]));
      value = nir_pad_vector(b, value, tex->def.num_components);
   } else if (tex->op == nir_texop_tg4) {
      value = nir_vec4(b, border[tex->component], border[tex->component], border[tex->component],
                       border[tex->component]);
   } else {
      value = nir_vec(b, border, 4);
   }
   value = nir_trim_vector(b, value, tex->def.num_components);
   nir_def *mixed = nir_flrp(b, value, &tex->def, weight);
   nir_pop_if(b, NULL);
   nir_def_rewrite_uses(placeholder, nir_if_phi(b, mixed, &tex->def));
   nir_instr_remove(nir_def_instr(placeholder));
   return true;
}

/* `t` modulo a positive `m`, never negative: irem is WGSL's %, which
 * takes the sign of `t`, and so does what imod becomes on the way. */
static nir_def *
positive_mod(nir_builder *b, nir_def *t, nir_def *m)
{
   nir_def *r = nir_irem(b, t, m);
   return nir_bcsel(b, nir_ilt_imm(b, r, 0), nir_iadd(b, r, m), r);
}

/* One texel coordinate of an integer texture, wrapped as the axis's
 * address mode says; `outside` is set where the border is. */
static nir_def *
wrap_texel(nir_builder *b, nir_def *t, nir_def *extent, nir_def *mode, nir_def **outside)
{
   nir_def *last = nir_iadd_imm(b, extent, -1);
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *clamped = nir_imin(b, nir_imax(b, t, zero), last);
   nir_def *repeat = positive_mod(b, t, extent);
   nir_def *period = positive_mod(b, t, nir_ishl_imm(b, extent, 1));
   nir_def *mirrored = nir_bcsel(b, nir_ilt(b, period, extent), period,
                                 nir_isub(b, nir_iadd_imm(b, nir_ishl_imm(b, extent, 1), -1), period));
   nir_def *once = nir_imin(b, nir_bcsel(b, nir_ilt(b, t, zero), nir_isub(b, nir_imm_int(b, -1), t), t), last);
   nir_def *out = nir_ior(b, nir_ilt(b, t, zero), nir_ige(b, t, extent));
   *outside = nir_ior(b, *outside, nir_iand(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER), out));
   return nir_bcsel(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_REPEAT), repeat,
          nir_bcsel(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT), mirrored,
          nir_bcsel(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE), once, clamped)));
}

/* The face of a cube a direction selects, and where in it, as Vulkan's
 * cube map face selection has it. */
static void
cube_face(nir_builder *b, nir_def *dir, nir_def **face, nir_def **s, nir_def **t)
{
   nir_def *x = nir_channel(b, dir, 0), *y = nir_channel(b, dir, 1), *z = nir_channel(b, dir, 2);
   nir_def *ax = nir_fabs(b, x), *ay = nir_fabs(b, y), *az = nir_fabs(b, z);
   nir_def *zero = nir_imm_float(b, 0.0f);
   nir_def *is_z = nir_iand(b, nir_fge(b, az, ax), nir_fge(b, az, ay));
   nir_def *is_y = nir_iand(b, nir_inot(b, is_z), nir_fge(b, ay, ax));
   nir_def *ma = nir_bcsel(b, is_z, az, nir_bcsel(b, is_y, ay, ax));
   nir_def *sc = nir_bcsel(b, is_z, nir_bcsel(b, nir_fge(b, z, zero), x, nir_fneg(b, x)),
                           nir_bcsel(b, is_y, x, nir_bcsel(b, nir_fge(b, x, zero), nir_fneg(b, z), z)));
   nir_def *tc = nir_bcsel(b, is_y, nir_bcsel(b, nir_fge(b, y, zero), z, nir_fneg(b, z)), nir_fneg(b, y));
   *face = nir_bcsel(b, is_z, nir_bcsel(b, nir_fge(b, z, zero), nir_imm_int(b, 4), nir_imm_int(b, 5)),
                     nir_bcsel(b, is_y, nir_bcsel(b, nir_fge(b, y, zero), nir_imm_int(b, 2), nir_imm_int(b, 3)),
                               nir_bcsel(b, nir_fge(b, x, zero), nir_imm_int(b, 0), nir_imm_int(b, 1))));
   nir_def *inv = nir_frcp(b, nir_fmax(b, ma, nir_imm_float(b, 1e-20f)));
   *s = nir_fmul_imm(b, nir_fadd_imm(b, nir_fmul(b, sc, inv), 1.0), 0.5);
   *t = nir_fmul_imm(b, nir_fadd_imm(b, nir_fmul(b, tc, inv), 1.0), 0.5);
}

/* Sampling an integer cube, which WebGPU does not: a fetch from the 2D
 * array of its faces, at the texel nearest where the direction meets the
 * face, of the level the LOD selects. */
static bool
lower_integer_cube_sample(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot)
{
   const int texture_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (texture_idx < 0 || coord_idx < 0 ||
       !glsl_sampler_type_is_array(glsl_without_array(nir_src_as_deref(tex->src[texture_idx].src)->type)))
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   nir_def *lambda = sample_lambda(b, meta, tex, group, slot, coord, 3);
   nir_def *levels = &texture_query(b, tex, nir_texop_query_levels, NULL)->def;
   nir_def *level = nir_f2i32(b, nir_fround_even(b, nir_fmax(b, lambda, nir_imm_float(b, 0.0f))));
   level = nir_imin(b, level, nir_iadd_imm(b, levels, -1));
   nir_def *size = &texture_query(b, tex, nir_texop_txs, level)->def;
   nir_def *face, *s, *t;
   cube_face(b, coord, &face, &s, &t);
   nir_def *w = nir_channel(b, size, 0), *h = nir_channel(b, size, 1);
   nir_def *x = nir_imin(b, nir_imax(b, nir_f2i32(b, nir_ffloor(b, nir_fmul(b, s, nir_i2f32(b, w)))), nir_imm_int(b, 0)),
                         nir_iadd_imm(b, w, -1));
   nir_def *y = nir_imin(b, nir_imax(b, nir_f2i32(b, nir_ffloor(b, nir_fmul(b, t, nir_i2f32(b, h)))), nir_imm_int(b, 0)),
                         nir_iadd_imm(b, h, -1));
   nir_def *layer = face;
   if (tex->is_array) {
      nir_def *cubes = nir_udiv_imm(b, nir_channel(b, size, 2), 6);
      nir_def *index = nir_imin(b, nir_imax(b, nir_f2i32(b, nir_fround_even(b, nir_channel(b, coord, 3))), nir_imm_int(b, 0)),
                                nir_iadd_imm(b, cubes, -1));
      layer = nir_iadd(b, face, nir_imul_imm(b, index, 6));
   }
   nir_tex_instr *txf = nir_tex_instr_create(b->shader, 3);
   txf->op = nir_texop_txf;
   txf->sampler_dim = GLSL_SAMPLER_DIM_2D;
   txf->is_array = true;
   txf->dest_type = tex->dest_type;
   txf->coord_components = 3;
   txf->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, tex->src[texture_idx].src.ssa);
   txf->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_vec3(b, x, y, layer));
   txf->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, level);
   nir_def_init(&txf->instr, &txf->def, 4, tex->def.bit_size);
   nir_builder_instr_insert(b, &txf->instr);
   if (tex->op == nir_texop_tg4) {
      nir_def *c = nir_channel(b, &txf->def, tex->component);
      nir_def_replace(&tex->def, nir_vec4(b, c, c, c, c));
      return true;
   }
   nir_def_replace(&tex->def, nir_trim_vector(b, &txf->def, tex->def.num_components));
   return true;
}

/* Gathering from an integer texture, which WebGPU samples not at all:
 * the 2x2 texels about the coordinates in the base level, each wrapped as
 * the sampler says or the border, in gather's order. */
static nir_def *
integer_gather(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot,
               nir_def *flags)
{
   const int texture_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   const unsigned n = tex->coord_components;
   const unsigned spatial = n - (tex->is_array ? 1 : 0);
   nir_def *level = nir_imm_int(b, 0);
   nir_def *size = &texture_query(b, tex, nir_texop_txs, level)->def;

   /* Each axis's two texels, and whether each is outside. */
   nir_def *texel[2][2], *out[2][2];
   for (unsigned i = 0; i < 2; ++i) {
      if (i >= spatial) {
         texel[i][0] = texel[i][1] = nir_imm_int(b, 0);
         out[i][0] = out[i][1] = nir_imm_false(b);
         continue;
      }
      nir_def *c = nir_channel(b, coord, i);
      nir_def *extent = nir_channel(b, size, i);
      nir_def *u = nir_bcsel(b, flag(b, flags, WVK_SAMPLER_UNNORMALIZED), c, nir_fmul(b, c, nir_i2f32(b, extent)));
      nir_def *t0 = nir_f2i32(b, nir_ffloor(b, nir_fadd_imm(b, u, -0.5)));
      if (offset_idx >= 0)
         t0 = nir_iadd(b, t0, nir_channel(b, tex->src[offset_idx].src.ssa, i));
      for (unsigned k = 0; k < 2; ++k) {
         out[i][k] = nir_imm_false(b);
         texel[i][k] = wrap_texel(b, nir_iadd_imm(b, t0, k), extent, address_mode(b, flags, i), &out[i][k]);
      }
   }
   nir_def *layer = NULL;
   if (tex->is_array)
      layer = nir_imin(b, nir_imax(b, nir_f2i32(b, nir_fround_even(b, nir_channel(b, coord, spatial))),
                                   nir_imm_int(b, 0)),
                       nir_iadd_imm(b, nir_channel(b, size, spatial), -1));

   nir_def *border[4];
   border_texel(b, meta, tex, group, slot, false, border);
   /* (i0, j1), (i1, j1), (i1, j0), (i0, j0). */
   static const unsigned order[4][2] = { { 0, 1 }, { 1, 1 }, { 1, 0 }, { 0, 0 } };
   nir_def *result[4];
   for (unsigned q = 0; q < 4; ++q) {
      const unsigned xi = order[q][0], yi = order[q][1];
      nir_def *comps[3] = { texel[0][xi], texel[1][yi], layer };
      nir_tex_instr *txf = nir_tex_instr_create(b->shader, 3);
      txf->op = nir_texop_txf;
      txf->sampler_dim = tex->sampler_dim;
      txf->is_array = tex->is_array;
      txf->dest_type = tex->dest_type;
      txf->coord_components = n;
      txf->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, tex->src[texture_idx].src.ssa);
      txf->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_vec(b, comps, n));
      txf->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, level);
      nir_def_init(&txf->instr, &txf->def, 4, tex->def.bit_size);
      nir_builder_instr_insert(b, &txf->instr);
      result[q] = nir_bcsel(b, nir_ior(b, out[0][xi], out[1][yi]), border[tex->component],
                            nir_channel(b, &txf->def, tex->component));
   }
   return nir_vec(b, result, 4);
}

/* Sampling an integer texture, which WebGPU does not: a fetch, of the
 * level the LOD selects (nearest; integer formats do not filter), at the
 * texel nearest the coordinates, wrapped as the sampler says, or the
 * border. */
static bool
lower_integer_sample(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot)
{
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE || tex->is_shadow)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   const int texture_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (texture_idx < 0 || coord_idx < 0)
      return false;
   nir_def *flags = sampler_word(b, meta, group, slot, WVK_META_SAMPLER_FLAGS);
   nir_def *bias = sampler_word(b, meta, group, slot, WVK_META_SAMPLER_BIAS);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   const unsigned n = tex->coord_components;
   const unsigned spatial = n - (tex->is_array ? 1 : 0);

   /* The LOD: given, or from the derivatives of the coordinates in texels
    * of the base level. */
   nir_def *base = &texture_query(b, tex, nir_texop_txs, nir_imm_int(b, 0))->def;
   nir_def *lambda;
   const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   if (tex->op == nir_texop_txl && lod_idx >= 0) {
      lambda = tex->src[lod_idx].src.ssa;
   } else if (tex->op == nir_texop_txd || b->shader->info.stage == MESA_SHADER_FRAGMENT) {
      nir_def *dx, *dy;
      if (tex->op == nir_texop_txd) {
         dx = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddx)].src.ssa;
         dy = tex->src[nir_tex_instr_src_index(tex, nir_tex_src_ddy)].src.ssa;
      } else {
         dx = nir_ddx(b, nir_trim_vector(b, coord, spatial));
         dy = nir_ddy(b, nir_trim_vector(b, coord, spatial));
      }
      nir_def *scale = nir_i2f32(b, nir_trim_vector(b, base, spatial));
      nir_def *rho = nir_fmax(b, nir_fast_length(b, nir_fmul(b, dx, scale)), nir_fast_length(b, nir_fmul(b, dy, scale)));
      lambda = nir_flog2(b, nir_fmax(b, rho, nir_imm_float(b, 1e-20f)));
      if (tex->op == nir_texop_txb)
         lambda = nir_fadd(b, lambda, tex->src[nir_tex_instr_src_index(tex, nir_tex_src_bias)].src.ssa);
   } else {
      lambda = nir_imm_float(b, 0.0f);
   }
   lambda = nir_fadd(b, lambda, bias);
   lambda = nir_fclamp(b, lambda, sampler_word(b, meta, group, slot, WVK_META_SAMPLER_MIN_LOD),
                       sampler_word(b, meta, group, slot, WVK_META_SAMPLER_MAX_LOD));
   nir_def *levels = &texture_query(b, tex, nir_texop_query_levels, NULL)->def;
   nir_def *level = nir_f2i32(b, nir_fround_even(b, nir_fmax(b, lambda, nir_imm_float(b, 0.0f))));
   level = nir_imin(b, level, nir_iadd_imm(b, levels, -1));
   level = nir_bcsel(b, flag(b, flags, WVK_SAMPLER_UNNORMALIZED), nir_imm_int(b, 0), level);

   if (tex->op == nir_texop_tg4) {
      nir_def_replace(&tex->def, integer_gather(b, meta, tex, group, slot, flags));
      return true;
   }

   nir_def *size = &texture_query(b, tex, nir_texop_txs, level)->def;
   nir_def *outside = nir_imm_false(b);
   nir_def *comps[4];
   for (unsigned i = 0; i < n; ++i) {
      nir_def *c = nir_channel(b, coord, i);
      nir_def *extent = nir_channel(b, size, i);
      if (i >= spatial) {
         /* The layer: nearest, and within the array. */
         comps[i] = nir_imin(b, nir_imax(b, nir_f2i32(b, nir_fround_even(b, c)), nir_imm_int(b, 0)),
                             nir_iadd_imm(b, extent, -1));
         continue;
      }
      nir_def *texel = nir_bcsel(b, flag(b, flags, WVK_SAMPLER_UNNORMALIZED), nir_ffloor(b, c),
                                 nir_ffloor(b, nir_fmul(b, c, nir_i2f32(b, extent))));
      nir_def *t = nir_f2i32(b, texel);
      if (offset_idx >= 0)
         t = nir_iadd(b, t, nir_channel(b, tex->src[offset_idx].src.ssa, i));
      comps[i] = wrap_texel(b, t, extent, address_mode(b, flags, i), &outside);
   }

   nir_tex_instr *txf = nir_tex_instr_create(b->shader, 3);
   txf->op = nir_texop_txf;
   txf->sampler_dim = tex->sampler_dim;
   txf->is_array = tex->is_array;
   txf->dest_type = tex->dest_type;
   txf->coord_components = n;
   txf->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, tex->src[texture_idx].src.ssa);
   txf->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_vec(b, comps, n));
   txf->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, level);
   nir_def_init(&txf->instr, &txf->def, tex->def.num_components, tex->def.bit_size);
   nir_builder_instr_insert(b, &txf->instr);

   nir_def *border[4];
   border_texel(b, meta, tex, group, slot, false, border);
   nir_def *value = nir_trim_vector(b, nir_vec(b, border, 4), tex->def.num_components);
   nir_def_replace(&tex->def, nir_bcsel(b, outside, value, &txf->def));
   return true;
}

/* One level's filtered comparison: the texels the filter takes, each
 * wrapped as the sampler's address modes say or the border's depth, each
 * compared with `ref`, weighed. */
static nir_def *
compare_level(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot,
              nir_def *flags, nir_def *level, nir_def *linear, nir_def *ref, nir_def *border_depth)
{
   const int texture_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   const unsigned n = tex->coord_components;
   const unsigned spatial = n - (tex->is_array ? 1 : 0);
   nir_def *op = nir_ubitfield_extract_imm(b, flags, WVK_SAMPLER_COMPARE_SHIFT, 3);
   nir_def *size = &texture_query(b, tex, nir_texop_txs, level)->def;

   nir_def *first[3], *frac[3];
   for (unsigned i = 0; i < spatial; ++i) {
      nir_def *u = nir_fmul(b, nir_channel(b, coord, i), nir_i2f32(b, nir_channel(b, size, i)));
      nir_def *lin = nir_fadd_imm(b, u, -0.5);
      nir_def *base = nir_bcsel(b, linear, nir_ffloor(b, lin), nir_ffloor(b, u));
      frac[i] = nir_bcsel(b, linear, nir_fsub(b, lin, nir_ffloor(b, lin)), nir_imm_float(b, 0.0f));
      first[i] = nir_f2i32(b, base);
      if (offset_idx >= 0)
         first[i] = nir_iadd(b, first[i], nir_channel(b, tex->src[offset_idx].src.ssa, i));
   }
   nir_def *layer = NULL;
   if (tex->is_array)
      layer = nir_imin(b, nir_imax(b, nir_f2i32(b, nir_fround_even(b, nir_channel(b, coord, spatial))),
                                   nir_imm_int(b, 0)),
                       nir_iadd_imm(b, nir_channel(b, size, spatial), -1));

   nir_def *sum = nir_imm_float(b, 0.0f);
   for (unsigned tap = 0; tap < (1u << spatial); ++tap) {
      nir_def *comps[4];
      nir_def *outside = nir_imm_false(b);
      nir_def *weight = nir_imm_float(b, 1.0f);
      for (unsigned i = 0; i < spatial; ++i) {
         const bool next = tap & (1u << i);
         nir_def *t = next ? nir_iadd_imm(b, first[i], 1) : first[i];
         comps[i] = wrap_texel(b, t, nir_channel(b, size, i), address_mode(b, flags, i), &outside);
         weight = nir_fmul(b, weight, next ? frac[i] : nir_fsub(b, nir_imm_float(b, 1.0f), frac[i]));
      }
      if (layer)
         comps[spatial] = layer;
      nir_tex_instr *txf = nir_tex_instr_create(b->shader, 3);
      txf->op = nir_texop_txf;
      txf->sampler_dim = tex->sampler_dim;
      txf->is_array = tex->is_array;
      txf->dest_type = nir_type_float32;
      txf->coord_components = n;
      txf->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, tex->src[texture_idx].src.ssa);
      txf->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_vec(b, comps, n));
      txf->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, level);
      nir_def_init(&txf->instr, &txf->def, 4, 32);
      nir_builder_instr_insert(b, &txf->instr);
      nir_def *depth = nir_bcsel(b, outside, border_depth, nir_channel(b, &txf->def, 0));
      sum = nir_fadd(b, sum, nir_fmul(b, weight, depth_compare(b, op, ref, depth)));
   }
   return sum;
}

/* A comparison with gradients, or at a LOD other than 0, which WGSL has
 * none of: by hand, at the levels the LOD selects, linear between them
 * where mipmaps are. */
static bool
lower_shadow_explicit(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot)
{
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int ref_idx = nir_tex_instr_src_index(tex, nir_tex_src_comparator);
   if (coord_idx < 0 || ref_idx < 0 || tex->coord_components - (tex->is_array ? 1 : 0) > 2)
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *flags = sampler_word(b, meta, group, slot, WVK_META_SAMPLER_FLAGS);
   const unsigned spatial = tex->coord_components - (tex->is_array ? 1 : 0);
   nir_def *lambda = sample_lambda(b, meta, tex, group, slot, tex->src[coord_idx].src.ssa, spatial);
   nir_def *levels = &texture_query(b, tex, nir_texop_query_levels, NULL)->def;
   nir_def *minified = nir_flt(b, nir_imm_float(b, 0.0f), lambda);
   nir_def *linear = nir_bcsel(b, minified, flag(b, flags, WVK_SAMPLER_MIN_LINEAR), flag(b, flags, WVK_SAMPLER_MAG_LINEAR));
   nir_def *mip_linear = nir_iand(b, minified, flag(b, flags, WVK_SAMPLER_MIP_LINEAR));
   nir_def *lam = nir_fmax(b, lambda, nir_imm_float(b, 0.0f));
   nir_def *top = nir_iadd_imm(b, levels, -1);
   nir_def *l0 = nir_f2i32(b, nir_bcsel(b, mip_linear, nir_ffloor(b, lam), nir_ffloor(b, nir_fadd_imm(b, lam, 0.5))));
   l0 = nir_imin(b, l0, top);
   nir_def *l1 = nir_imin(b, nir_iadd_imm(b, l0, 1), top);
   nir_def *t = nir_bcsel(b, mip_linear, nir_ffract(b, lam), nir_imm_float(b, 0.0f));
   nir_def *border[4];
   border_texel(b, meta, tex, group, slot, true, border);
   nir_def *ref = nir_channel(b, tex->src[ref_idx].src.ssa, 0);
   nir_def *r0 = compare_level(b, meta, tex, group, slot, flags, l0, linear, ref, border[0]);
   nir_def *r1 = compare_level(b, meta, tex, group, slot, flags, l1, linear, ref, border[0]);
   nir_def *result = nir_flrp(b, r0, r1, t);
   nir_def_replace(&tex->def, nir_pad_vector_imm_int(b, result, 0, tex->def.num_components));
   return true;
}

/* A cube a non-seamless sampler samples (VK_EXT_non_seamless_cube_map),
 * which the variant has as the 2D array of its faces (wvk_shader.c's
 * face_cubes): the face the direction selects on its own, at the LOD the
 * cube's derivatives give -- without the sampler's bias, which what
 * follows adds -- and from there on a 2D array's sample, wrapped as the
 * sampler says. */
static void
face_sample(nir_builder *b, const uint32_t *meta, nir_tex_instr *tex, uint32_t group, uint32_t slot)
{
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_idx < 0)
      return;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   nir_def *lod = NULL;
   if (tex->op != nir_texop_tg4)
      lod = nir_fsub(b, sample_lambda_raw(b, meta, tex, group, slot, coord, 3),
                     sampler_word(b, meta, group, slot, WVK_META_SAMPLER_BIAS));
   nir_def *face, *s, *t;
   cube_face(b, coord, &face, &s, &t);
   nir_def *layer = face;
   if (tex->is_array) {
      /* The cube, of those there are, and its face. */
      nir_def *cubes = nir_udiv_imm(b, nir_channel(b, &texture_query(b, tex, nir_texop_txs, nir_imm_int(b, 0))->def, 2), 6);
      nir_def *index = nir_imin(b, nir_imax(b, nir_f2i32(b, nir_fround_even(b, nir_channel(b, coord, 3))), nir_imm_int(b, 0)),
                                nir_iadd_imm(b, cubes, -1));
      layer = nir_iadd(b, face, nir_imul_imm(b, index, 6));
   }
   layer = nir_i2f32(b, layer);
   nir_src_rewrite(&tex->src[coord_idx].src, nir_vec3(b, s, t, layer));
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   tex->coord_components = 3;
   if (!lod)
      return;
   const nir_tex_src_type gone[] = { nir_tex_src_bias, nir_tex_src_ddx, nir_tex_src_ddy, nir_tex_src_lod };
   for (unsigned i = 0; i < ARRAY_SIZE(gone); ++i) {
      const int idx = nir_tex_instr_src_index(tex, gone[i]);
      if (idx >= 0)
         nir_tex_instr_remove_src(tex, idx);
   }
   tex->op = nir_texop_txl;
   nir_tex_instr_add_src(tex, nir_tex_src_lod, lod);
}

static bool
lower_sampler_instr(nir_builder *b, nir_instr *instr, void *data)
{
   const uint32_t *meta = data;
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
   case nir_texop_txl:
   case nir_texop_txd:
   case nir_texop_tg4:
   case nir_texop_lod:
      break;
   default:
      return false;
   }
   const int sampler = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   uint32_t group, slot;
   if (sampler < 0 || !deref_binding(tex->src[sampler].src, &group, &slot))
      return false;
   if (tex->op == nir_texop_lod)
      return lower_query_lod(b, meta, tex, group, slot);
   const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE && texture >= 0 &&
       (tex->is_shadow || nir_alu_type_get_base_type(tex->dest_type) == nir_type_float) &&
       glsl_get_sampler_dim(glsl_without_array(nir_src_as_deref(tex->src[texture].src)->type)) == GLSL_SAMPLER_DIM_2D)
      face_sample(b, meta, tex, group, slot);
   lower_min_lod(b, meta, tex, group, slot);
   /* WGSL compares at level 0 or with implicit derivatives, and without a
    * bias: the rest by hand. */
   if (tex->is_shadow) {
      const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      const bool level_zero = tex->op == nir_texop_txl && lod_idx >= 0 && nir_src_is_const(tex->src[lod_idx].src) &&
                              nir_src_as_float(tex->src[lod_idx].src) == 0.0;
      const bool explicit = tex->op == nir_texop_txd || tex->op == nir_texop_txb ||
                            (tex->op == nir_texop_txl && !level_zero);
      if (explicit && tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE && lower_shadow_explicit(b, meta, tex, group, slot))
         return true;
      if (explicit && tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         /* A cube's faces are not fetched by hand: at level 0, which is
          * all a shadow cube map mostly has. */
         mesa_logw_once("wvk: a shadow cube sampled with a bias, LOD or gradients is sampled at level 0");
         b->cursor = nir_before_instr(&tex->instr);
         const nir_tex_src_type gone[] = { nir_tex_src_bias, nir_tex_src_ddx, nir_tex_src_ddy, nir_tex_src_lod };
         for (unsigned i = 0; i < ARRAY_SIZE(gone); ++i) {
            const int idx = nir_tex_instr_src_index(tex, gone[i]);
            if (idx >= 0)
               nir_tex_instr_remove_src(tex, idx);
         }
         tex->op = nir_texop_txl;
         nir_tex_instr_add_src(tex, nir_tex_src_lod, nir_imm_float(b, 0.0f));
      }
   }
   if (nir_alu_type_get_base_type(tex->dest_type) != nir_type_float && !tex->is_shadow)
      return tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE ? lower_integer_cube_sample(b, meta, tex, group, slot)
                                                       : lower_integer_sample(b, meta, tex, group, slot);
   return lower_float_sample(b, meta, tex, group, slot);
}

bool
wvk_nir_lower_samplers(nir_shader *nir, const uint32_t *meta_base)
{
   return nir_shader_instructions_pass(nir, lower_sampler_instr, nir_metadata_none, (void *)meta_base);
}

/* ---------------------------------------------------------------------- */
/* robustImageAccess2, and null image descriptors (nullDescriptor): an
 * access past an image view's bounds, or of a null view, reads zero -- of
 * alpha one where the view's format has none -- and writes nothing; and a
 * null view's queries are zero. What the format has is the view's record's
 * channels, which a null view has none of. After the variables are split
 * and before the views' swizzles, which take a texel so replaced as they
 * take any. */

struct robust_images {
   const uint32_t *meta;
   bool bounds, null;
};

/* The channels of the view a deref is of, or NULL for no descriptor. */
static nir_def *
view_channels(nir_builder *b, const uint32_t *meta, nir_src src, bool storage)
{
   uint32_t group, slot;
   if (!deref_binding(src, &group, &slot))
      return NULL;
   return wvk_nir_load_meta(b, meta, group, nir_imm_int(b, slot),
                            storage ? WVK_META_IMG_CHANNELS : WVK_META_TEX_CHANNELS);
}

/* Zero, of alpha one where the format has channels and not alpha. */
static nir_def *
replaced_texel(nir_builder *b, nir_def *channels, unsigned components, unsigned bit_size, nir_alu_type type)
{
   nir_def *zero = nir_imm_zero(b, 1, bit_size);
   nir_def *one = nir_alu_type_get_base_type(type) == nir_type_float ? nir_imm_floatN_t(b, 1.0, bit_size)
                                                                     : nir_imm_intN_t(b, 1, bit_size);
   nir_def *alpha = nir_bcsel(b, nir_iand(b, nir_ine_imm(b, channels, 0), nir_ieq_imm(b, nir_iand_imm(b, channels, 8), 0)),
                              one, zero);
   nir_def *c[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < components; ++i)
      c[i] = i == 3 ? alpha : zero;
   return nir_vec(b, c, components);
}

/* The instruction where `valid`, else `otherwise` for its result. */
static void
wrap_valid(nir_builder *b, nir_instr *instr, nir_def *valid, nir_def *otherwise)
{
   nir_def *def = nir_instr_def(instr);
   nir_push_if(b, valid);
   nir_instr *clone = nir_instr_clone(b->shader, instr);
   nir_builder_instr_insert(b, clone);
   nir_pop_if(b, NULL);
   if (def)
      nir_def_rewrite_uses(def, nir_if_phi(b, nir_instr_def(clone), otherwise));
   nir_instr_remove(instr);
}

/* A texture query of `tex`'s texture, at `lod` for a size. */
static nir_def *
view_query(nir_builder *b, nir_tex_instr *tex, nir_texop op, nir_def *lod)
{
   const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   nir_tex_instr *q = nir_tex_instr_create(b->shader, lod ? 2 : 1);
   q->op = op;
   q->sampler_dim = tex->sampler_dim;
   q->is_array = tex->is_array;
   q->dest_type = nir_type_int32;
   q->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, tex->src[texture].src.ssa);
   if (lod)
      q->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_def_init(&q->instr, &q->def, nir_tex_instr_dest_size(q), 32);
   nir_builder_instr_insert(b, &q->instr);
   return &q->def;
}

/* Whether a texel fetch's coordinates are in the view: of its level, in
 * its layers, of its samples. */
static nir_def *
fetch_in_bounds(nir_builder *b, nir_tex_instr *tex)
{
   const int coord_src = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int lod_src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int offset_src = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   const int ms_src = nir_tex_instr_src_index(tex, nir_tex_src_ms_index);
   if (coord_src < 0)
      return nir_imm_true(b);
   nir_def *coord = tex->src[coord_src].src.ssa;
   nir_def *lod = lod_src >= 0 ? tex->src[lod_src].src.ssa : nir_imm_int(b, 0);
   if (offset_src >= 0) {
      nir_def *offset = nir_pad_vector_imm_int(b, tex->src[offset_src].src.ssa, 0, coord->num_components);
      coord = nir_iadd(b, coord, offset);
   }
   /* Of a multisampled texture, which has no levels: its size. */
   nir_def *size = view_query(b, tex, nir_texop_txs, tex->op == nir_texop_txf_ms ? NULL : lod);
   nir_def *valid = nir_ball(b, nir_ult(b, coord, nir_trim_vector(b, size, coord->num_components)));
   if (tex->op == nir_texop_txf) {
      nir_def *levels = view_query(b, tex, nir_texop_query_levels, NULL);
      valid = nir_iand(b, valid, nir_ult(b, lod, levels));
   }
   if (ms_src >= 0) {
      nir_def *samples = view_query(b, tex, nir_texop_texture_samples, NULL);
      valid = nir_iand(b, valid, nir_ult(b, tex->src[ms_src].src.ssa, samples));
   }
   return valid;
}

/* Whether an image access's coordinates are in the view (as
 * nir_lower_robust_access has it). */
static nir_def *
image_in_bounds(nir_builder *b, nir_intrinsic_instr *intr)
{
   const enum glsl_sampler_dim dim = nir_intrinsic_image_dim(intr);
   const bool array = nir_intrinsic_image_array(intr);
   const unsigned coords = nir_image_intrinsic_coord_components(intr);
   nir_def *coord = nir_trim_vector(b, intr->src[1].ssa, coords);
   const unsigned size_components = dim == GLSL_SAMPLER_DIM_CUBE && !array ? coords - 1 : coords;
   nir_def *size = nir_image_deref_size(b, size_components, 32, intr->src[0].ssa, nir_imm_int(b, 0),
                                        .image_dim = dim, .image_array = array);
   if (dim == GLSL_SAMPLER_DIM_CUBE) {
      nir_def *z = array ? nir_imul_imm(b, nir_channel(b, size, 2), 6) : nir_imm_int(b, 6);
      size = nir_vec3(b, nir_channel(b, size, 0), nir_channel(b, size, 1), z);
   }
   nir_def *valid = nir_ball(b, nir_ult(b, coord, size));
   if (dim == GLSL_SAMPLER_DIM_MS) {
      nir_def *samples = nir_image_deref_samples(b, 32, intr->src[0].ssa, .image_dim = dim, .image_array = array);
      valid = nir_iand(b, valid, nir_ult(b, intr->src[2].ssa, samples));
   }
   return valid;
}

static bool
robust_image_instr(nir_builder *b, nir_instr *instr, const struct robust_images *r)
{
   b->cursor = nir_before_instr(instr);
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
      if (texture < 0 || tex->sampler_dim == GLSL_SAMPLER_DIM_BUF)
         return false;
      const bool fetch = tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms;
      if (!r->null && !(r->bounds && fetch))
         return false;
      nir_def *channels = view_channels(b, r->meta, tex->src[texture].src, false);
      if (!channels)
         return false;
      nir_def *valid = r->bounds && fetch ? fetch_in_bounds(b, tex) : nir_imm_true(b);
      if (r->null)
         valid = nir_iand(b, valid, nir_ine_imm(b, channels, 0));
      /* Of a texel: replaced. Of a comparison or a query: zero. */
      const bool texel = !tex->is_shadow && tex->def.num_components == 4 && tex->op != nir_texop_txs &&
                         tex->op != nir_texop_query_levels && tex->op != nir_texop_texture_samples &&
                         tex->op != nir_texop_lod;
      nir_def *otherwise = texel ? replaced_texel(b, channels, 4, tex->def.bit_size, tex->dest_type)
                                 : nir_imm_zero(b, tex->def.num_components, tex->def.bit_size);
      wrap_valid(b, instr, valid, otherwise);
      return true;
   }
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   bool access = false;
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
      access = true;
      break;
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_levels:
      if (!r->null)
         return false;
      break;
   default:
      return false;
   }
   if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF)
      return false;
   nir_def *channels = view_channels(b, r->meta, intr->src[0], true);
   if (!channels)
      return false;
   nir_def *valid = access && r->bounds ? image_in_bounds(b, intr) : nir_imm_true(b);
   if (r->null)
      valid = nir_iand(b, valid, nir_ine_imm(b, channels, 0));
   nir_def *otherwise = NULL;
   if (nir_intrinsic_infos[intr->intrinsic].has_dest) {
      const bool texel = intr->intrinsic == nir_intrinsic_image_deref_load && intr->def.num_components == 4;
      otherwise = texel ? replaced_texel(b, channels, 4, intr->def.bit_size, nir_intrinsic_dest_type(intr))
                        : nir_imm_zero(b, intr->def.num_components, intr->def.bit_size);
   }
   wrap_valid(b, instr, valid, otherwise);
   return true;
}

bool
wvk_nir_lower_robust_images(nir_shader *nir, const uint32_t *meta_base, bool bounds, bool null)
{
   if (!bounds && !null)
      return false;
   const struct robust_images r = { .meta = meta_base, .bounds = bounds, .null = null };
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      /* The instructions first: the ones made here are not to be made
       * again. */
      struct util_dynarray instrs;
      util_dynarray_init(&instrs, NULL);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block)
            util_dynarray_append(&instrs, instr);
      }
      nir_builder b = nir_builder_create(impl);
      bool changed = false;
      util_dynarray_foreach(&instrs, nir_instr *, i)
         changed |= robust_image_instr(&b, *i, &r);
      util_dynarray_fini(&instrs);
      progress |= nir_progress(changed, impl, nir_metadata_none);
   }
   return progress;
}

/* ---------------------------------------------------------------------- */
/* A view's minimum LOD (VK_EXT_image_view_min_lod): of a view that has
 * one, a sampling's LOD is at least it -- the sample's MinLod raised to it,
 * which the samplers' lowering then does (lower_min_lod) -- and a fetch
 * below its level reads zero. A view without one samples as before, which
 * the record says at the draw. Before the samplers' lowering. */

static bool
view_min_lod_instr(nir_builder *b, nir_tex_instr *tex, const uint32_t *meta)
{
   const bool fetch = tex->op == nir_texop_txf;
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb && tex->op != nir_texop_txl &&
       tex->op != nir_texop_txd && !fetch)
      return false;
   const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   uint32_t group, slot;
   if (texture < 0 || tex->sampler_dim == GLSL_SAMPLER_DIM_BUF || !deref_binding(tex->src[texture].src, &group, &slot))
      return false;
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *min = wvk_nir_load_meta(b, meta, group, nir_imm_int(b, slot), WVK_META_TEX_MIN_LOD);
   if (fetch) {
      const int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
      nir_def *level = lod >= 0 ? tex->src[lod].src.ssa : nir_imm_int(b, 0);
      wrap_valid(b, &tex->instr, nir_ige(b, level, nir_f2i32(b, nir_ffloor(b, min))),
                 nir_imm_zero(b, tex->def.num_components, tex->def.bit_size));
      return true;
   }
   nir_push_if(b, nir_flt_imm(b, nir_fneg(b, min), 0.0));
   /* With its own MinLod, the greater. */
   const int own = nir_tex_instr_src_index(tex, nir_tex_src_min_lod);
   nir_def *floor = own >= 0 ? nir_fmax(b, tex->src[own].src.ssa, min) : min;
   nir_tex_instr *raised = nir_instr_as_tex(nir_instr_clone(b->shader, &tex->instr));
   nir_builder_instr_insert(b, &raised->instr);
   if (own >= 0)
      nir_src_rewrite(&raised->src[own].src, floor);
   else
      nir_tex_instr_add_src(raised, nir_tex_src_min_lod, floor);
   nir_push_else(b, NULL);
   nir_instr *plain = nir_instr_clone(b->shader, &tex->instr);
   nir_builder_instr_insert(b, plain);
   nir_pop_if(b, NULL);
   nir_def_rewrite_uses(&tex->def, nir_if_phi(b, &raised->def, nir_instr_def(plain)));
   nir_instr_remove(&tex->instr);
   return true;
}

bool
wvk_nir_lower_view_min_lod(nir_shader *nir, const uint32_t *meta_base)
{
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      /* The instructions first: the ones made here are not to be made
       * again. */
      struct util_dynarray instrs;
      util_dynarray_init(&instrs, NULL);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex)
               util_dynarray_append(&instrs, instr);
         }
      }
      nir_builder b = nir_builder_create(impl);
      bool changed = false;
      util_dynarray_foreach(&instrs, nir_instr *, i)
         changed |= view_min_lod_instr(&b, nir_instr_as_tex(*i), meta_base);
      util_dynarray_fini(&instrs);
      progress |= nir_progress(changed, impl, nir_metadata_none);
   }
   return progress;
}

/* ---------------------------------------------------------------------- */
/* The slices of a 3D image a storage view has (VK_EXT_image_sliced_view_of_3d,
 * and a 2D view's one of them): WebGPU views the level whole, so z is the
 * view's first slice on, and the size's depth the view's slices. After the
 * robust images' bounds, which are of the view's own coordinates. */

static bool
image_slices_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const uint32_t *meta = data;
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
      break;
   default:
      return false;
   }
   uint32_t group, slot;
   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_3D || !deref_binding(intr->src[0], &group, &slot))
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *slices = wvk_nir_load_meta(b, meta, group, nir_imm_int(b, slot), WVK_META_IMG_SLICES);
   if (intr->intrinsic == nir_intrinsic_image_deref_size) {
      if (intr->def.num_components < 3)
         return false;
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *count = nir_ushr_imm(b, slices, 16);
      nir_def *depth = nir_bcsel(b, nir_ine_imm(b, count, 0), count, nir_channel(b, &intr->def, 2));
      nir_def *size = nir_vector_insert_imm(b, &intr->def, depth, 2);
      nir_def_rewrite_uses_after(&intr->def, size);
      return true;
   }
   nir_def *coord = intr->src[1].ssa;
   nir_def *z = nir_iadd(b, nir_channel(b, coord, 2), nir_iand_imm(b, slices, 0xffff));
   nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(b, coord, z, 2));
   return true;
}

bool
wvk_nir_lower_image_slices(nir_shader *nir, const uint32_t *meta_base)
{
   return nir_shader_intrinsics_pass(nir, image_slices_instr, nir_metadata_control_flow, (void *)meta_base);
}
