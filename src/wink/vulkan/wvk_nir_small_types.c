/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * 8- and 16-bit values, which WGSL has no integers of, and Tint reads no
 * SPIR-V of: every one a 32-bit word of its bits, zero-extended -- but
 * for half floats a float operation makes, which WGSL's f16 has. What
 * reads a word as a signed integer sign-extends it; what makes one keeps
 * it to its bits; a half float to or from a word is packHalf2x16's and
 * unpackHalf2x16's. Memory is words by now (nir_lower_mem_access_bit_sizes),
 * and variables of such types are words, or 32-bit floats where they are
 * interpolated.
 */
#include "wvk_nir.h"

#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"

struct widen {
   /* The bit size each def had where it was 8 or 16 and is a word now. */
   struct hash_table *old_size;
   /* The phis retyped, whose sources are made words last. */
   struct util_dynarray phis;
   bool progress;
};

static bool
is_small(unsigned bit_size)
{
   return bit_size == 8 || bit_size == 16;
}

static uint32_t
mask_of(unsigned bits)
{
   return bits >= 32 ? ~0u : (1u << bits) - 1;
}

/* A word as its low `bits`, sign-extended: extractBits', which a shift
 * left of a signed integer, undefined past its sign to Metal, is not. */
static nir_def *
sext(nir_builder *b, nir_def *word, unsigned bits)
{
   return nir_ibitfield_extract(b, word, nir_imm_int(b, 0), nir_imm_int(b, bits));
}

/* A half float as a word of its bits, and back: packHalf2x16's. */
static nir_def *
half_to_word(nir_builder *b, nir_def *f32)
{
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < f32->num_components; ++i)
      comps[i] = nir_pack_half_2x16(b, nir_vec2(b, nir_channel(b, f32, i), nir_imm_float(b, 0.0f)));
   return nir_vec(b, comps, f32->num_components);
}

/* The same, rounded toward zero: one less in magnitude where rounding to
 * the nearest went past the value. */
static nir_def *
half_to_word_rtz(nir_builder *b, nir_def *f32)
{
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < f32->num_components; ++i) {
      nir_def *x = nir_channel(b, f32, i);
      nir_def *h = nir_pack_half_2x16(b, nir_vec2(b, x, nir_imm_float(b, 0.0f)));
      nir_def *back = nir_channel(b, nir_unpack_half_2x16(b, h), 0);
      nir_def *over = nir_iand(b, nir_flt(b, nir_fabs(b, x), nir_fabs(b, back)), nir_feq(b, x, x));
      comps[i] = nir_bcsel(b, over, nir_iadd_imm(b, h, -1), h);
   }
   return nir_vec(b, comps, f32->num_components);
}

static nir_def *
word_to_half_f32(nir_builder *b, nir_def *word)
{
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < word->num_components; ++i)
      comps[i] = nir_channel(b, nir_unpack_half_2x16(b, nir_channel(b, word, i)), 0);
   return nir_vec(b, comps, word->num_components);
}

static unsigned
old_size(const struct widen *w, const nir_def *def)
{
   struct hash_entry *he = _mesa_hash_table_search(w->old_size, def);
   return he ? (unsigned)(uintptr_t)he->data : 0;
}

static void
set_old_size(struct widen *w, nir_def *def, unsigned bits)
{
   _mesa_hash_table_insert(w->old_size, def, (void *)(uintptr_t)bits);
}

/* A source as a word, if it was small: a half float made one. */
static nir_def *
as_word(nir_builder *b, const struct widen *w, nir_def *def)
{
   if (def->bit_size == 16)
      return half_to_word(b, nir_f2f32(b, def));
   return def;
}

/* A source as a 32-bit float of its value: a word's half float, a half
 * float's, or the float it is. */
static nir_def *
as_f32(nir_builder *b, const struct widen *w, nir_def *def)
{
   if (def->bit_size == 16)
      return nir_f2f32(b, def);
   if (old_size(w, def) == 16)
      return word_to_half_f32(b, def);
   return def;
}

/* A source as an integer of 32 bits, or 64 if it is: a word sign- or
 * zero-extended from its old size. */
static nir_def *
as_int(nir_builder *b, const struct widen *w, nir_def *def, bool is_signed)
{
   def = as_word(b, w, def);
   const unsigned bits = old_size(w, def);
   if (!bits)
      return def;
   return is_signed ? sext(b, def, bits) : def;
}

static nir_def *
keep_bits(nir_builder *b, nir_def *word, unsigned bits)
{
   return bits < 32 ? nir_iand_imm(b, word, mask_of(bits)) : word;
}

/* The source of an ALU instruction as it is read: its swizzle applied,
 * a word of the size it was. */
static nir_def *
alu_src(nir_builder *b, struct widen *w, nir_alu_instr *alu, unsigned i)
{
   nir_def *s = nir_mov_alu(b, alu->src[i], nir_ssa_alu_instr_src_components(alu, i));
   const unsigned bits = old_size(w, alu->src[i].src.ssa);
   if (bits && s != alu->src[i].src.ssa)
      set_old_size(w, s, bits);
   return s;
}

static nir_def *
lower_conversion(nir_builder *b, struct widen *w, nir_alu_instr *alu, unsigned dst_bits)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   const nir_alu_type src_base = nir_alu_type_get_base_type(info->input_types[0]);
   const nir_alu_type dst_base = nir_alu_type_get_base_type(info->output_type);
   nir_def *src = alu_src(b, w, alu, 0);

   /* The source as a value of 32 bits (64 if it has them). */
   nir_def *value;
   switch (src_base) {
   case nir_type_float:
      value = src->bit_size == 64 ? src : as_f32(b, w, src);
      break;
   case nir_type_int:
      value = as_int(b, w, src, true);
      break;
   case nir_type_uint:
      value = as_int(b, w, src, false);
      break;
   default: /* bool */
      value = src;
      break;
   }

   const unsigned out = MAX2(dst_bits, 32);
   nir_def *result;
   switch (dst_base) {
   case nir_type_float:
      if (src_base == nir_type_float)
         result = value->bit_size == out ? value : nir_type_convert(b, value, nir_type_float, nir_type_float | out, nir_rounding_mode_undef);
      else if (src_base == nir_type_bool)
         result = nir_b2fN(b, value, out);
      else
         result = nir_type_convert(b, value, src_base, nir_type_float | out, nir_rounding_mode_undef);
      if (dst_bits == 16) {
         nir_def *f = result->bit_size == 32 ? result : nir_f2f32(b, result);
         result = alu->op == nir_op_f2f16_rtz ? half_to_word_rtz(b, f) : half_to_word(b, f);
      }
      return result;
   case nir_type_int:
   case nir_type_uint:
      if (src_base == nir_type_bool)
         result = nir_b2iN(b, value, out);
      else if (src_base == nir_type_float)
         result = nir_type_convert(b, value, nir_type_float, dst_base | out, nir_rounding_mode_undef);
      else
         result = value->bit_size == out ? value : nir_type_convert(b, value, src_base, dst_base | out, nir_rounding_mode_undef);
      return keep_bits(b, result, dst_bits);
   default:
      return NULL;
   }
}

/* A small integer or untyped ALU instruction, at 32 bits. */
static nir_def *
lower_int_alu(nir_builder *b, struct widen *w, nir_alu_instr *alu, unsigned dst_bits)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_def *srcs[NIR_MAX_VEC_COMPONENTS] = { NULL };
   unsigned small_src_bits = 0;
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_def *s = alu_src(b, w, alu, i);
      const nir_alu_type t = nir_alu_type_get_base_type(info->input_types[i]);
      small_src_bits = MAX2(small_src_bits, old_size(w, s) ? old_size(w, s) : (s->bit_size == 16 ? 16 : 0));
      if (t == nir_type_float)
         srcs[i] = as_f32(b, w, s);
      else if (t == nir_type_int)
         srcs[i] = as_int(b, w, s, true);
      else
         srcs[i] = as_word(b, w, s);
   }
   const unsigned bits = is_small(dst_bits) ? dst_bits : small_src_bits;

   nir_def *r;
   switch (alu->op) {
   case nir_op_ishl:
      r = nir_ishl(b, srcs[0], nir_iand_imm(b, srcs[1], bits - 1));
      break;
   case nir_op_ishr:
      r = nir_ishr(b, srcs[0], nir_iand_imm(b, srcs[1], bits - 1));
      break;
   case nir_op_ushr:
      r = nir_ushr(b, srcs[0], nir_iand_imm(b, srcs[1], bits - 1));
      break;
   case nir_op_urol:
   case nir_op_uror: {
      nir_def *s = nir_iand_imm(b, srcs[1], bits - 1);
      nir_def *back = nir_isub_imm(b, bits, s);
      r = alu->op == nir_op_urol ? nir_ior(b, nir_ishl(b, srcs[0], s), nir_ushr(b, srcs[0], back))
                                 : nir_ior(b, nir_ushr(b, srcs[0], s), nir_ishl(b, srcs[0], back));
      break;
   }
   case nir_op_bitfield_reverse:
      r = nir_ushr_imm(b, nir_bitfield_reverse(b, srcs[0]), 32 - bits);
      break;
   case nir_op_uclz:
      r = nir_iadd_imm(b, nir_uclz(b, srcs[0]), -(int)(32 - bits));
      break;
   case nir_op_uadd_carry:
      r = nir_ushr_imm(b, nir_iadd(b, srcs[0], srcs[1]), bits);
      break;
   case nir_op_umul_high:
      r = nir_ushr_imm(b, nir_imul(b, srcs[0], srcs[1]), bits);
      break;
   case nir_op_imul_high:
      r = nir_ishr_imm(b, nir_imul(b, srcs[0], srcs[1]), bits);
      break;
   case nir_op_uadd_sat:
      r = nir_umin(b, nir_iadd(b, srcs[0], srcs[1]), nir_imm_int(b, mask_of(bits)));
      break;
   case nir_op_iadd_sat:
   case nir_op_isub_sat: {
      nir_def *v = alu->op == nir_op_iadd_sat ? nir_iadd(b, srcs[0], srcs[1]) : nir_isub(b, srcs[0], srcs[1]);
      r = nir_imin(b, nir_imax(b, v, nir_imm_int(b, -(1 << (bits - 1)))), nir_imm_int(b, (1 << (bits - 1)) - 1));
      break;
   }
   case nir_op_pack_32_2x16_split:
      r = nir_ior(b, nir_ishl_imm(b, srcs[1], 16), srcs[0]);
      return r;
   case nir_op_pack_32_2x16: {
      nir_def *v = srcs[0];
      return nir_ior(b, nir_ishl_imm(b, nir_channel(b, v, 1), 16), nir_channel(b, v, 0));
   }
   case nir_op_pack_32_4x8_split:
      return nir_ior(b, nir_ior(b, srcs[0], nir_ishl_imm(b, srcs[1], 8)),
                     nir_ior(b, nir_ishl_imm(b, srcs[2], 16), nir_ishl_imm(b, srcs[3], 24)));
   case nir_op_pack_32_4x8: {
      nir_def *v = srcs[0];
      return nir_ior(b, nir_ior(b, nir_channel(b, v, 0), nir_ishl_imm(b, nir_channel(b, v, 1), 8)),
                     nir_ior(b, nir_ishl_imm(b, nir_channel(b, v, 2), 16), nir_ishl_imm(b, nir_channel(b, v, 3), 24)));
   }
   case nir_op_pack_64_4x16: {
      nir_def *v = srcs[0];
      nir_def *lo = nir_ior(b, nir_ishl_imm(b, nir_channel(b, v, 1), 16), nir_channel(b, v, 0));
      nir_def *hi = nir_ior(b, nir_ishl_imm(b, nir_channel(b, v, 3), 16), nir_channel(b, v, 2));
      return nir_pack_64_2x32_split(b, lo, hi);
   }
   case nir_op_unpack_32_2x16_split_x:
      return nir_iand_imm(b, srcs[0], 0xffff);
   case nir_op_unpack_32_2x16_split_y:
      return nir_ushr_imm(b, srcs[0], 16);
   case nir_op_unpack_32_2x16:
      return nir_vec2(b, nir_iand_imm(b, srcs[0], 0xffff), nir_ushr_imm(b, srcs[0], 16));
   case nir_op_unpack_32_4x8:
      return nir_vec4(b, nir_iand_imm(b, srcs[0], 0xff), nir_iand_imm(b, nir_ushr_imm(b, srcs[0], 8), 0xff),
                      nir_iand_imm(b, nir_ushr_imm(b, srcs[0], 16), 0xff), nir_ushr_imm(b, srcs[0], 24));
   case nir_op_unpack_64_4x16: {
      nir_def *lo = nir_unpack_64_2x32_split_x(b, srcs[0]), *hi = nir_unpack_64_2x32_split_y(b, srcs[0]);
      return nir_vec4(b, nir_iand_imm(b, lo, 0xffff), nir_ushr_imm(b, lo, 16), nir_iand_imm(b, hi, 0xffff),
                      nir_ushr_imm(b, hi, 16));
   }
   default: {
      /* The operation itself, which is the same on wider operands. */
      nir_alu_instr *nalu = nir_alu_instr_create(b->shader, alu->op);
      nalu->fp_math_ctrl = alu->fp_math_ctrl;
      nalu->no_signed_wrap = false;
      nalu->no_unsigned_wrap = false;
      for (unsigned i = 0; i < info->num_inputs; ++i)
         nalu->src[i].src = nir_src_for_ssa(srcs[i]);
      unsigned out_bits = is_small(dst_bits) ? 32 : dst_bits;
      if (nir_alu_type_get_base_type(info->output_type) == nir_type_float && dst_bits == 16)
         out_bits = 32;
      nir_def_init(&nalu->instr, &nalu->def, alu->def.num_components, out_bits);
      nir_builder_instr_insert(b, &nalu->instr);
      r = &nalu->def;
      break;
   }
   }
   const nir_alu_type out = nir_alu_type_get_base_type(info->output_type);
   if (out == nir_type_bool || !is_small(dst_bits))
      return r;
   if (out == nir_type_float)
      return half_to_word(b, r);
   /* Moves and selects of words keep them as they are. */
   if (alu->op == nir_op_mov || nir_op_is_vec(alu->op) || alu->op == nir_op_bcsel)
      return r;
   return keep_bits(b, r, dst_bits);
}

/* A half-float operation, which WGSL has: its half-float sources that
 * are words made half floats again. */
static bool
lower_half_alu(nir_builder *b, struct widen *w, nir_alu_instr *alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   bool progress = false;
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_def *s = alu->src[i].src.ssa;
      if (old_size(w, s) != 16 ||
          nir_alu_type_get_base_type(info->input_types[i]) != nir_type_float)
         continue;
      b->cursor = nir_before_instr(&alu->instr);
      nir_src_rewrite(&alu->src[i].src, nir_f2f16(b, word_to_half_f32(b, s)));
      progress = true;
   }
   return progress;
}

static bool
widen_alu(nir_builder *b, struct widen *w, nir_alu_instr *alu)
{
   const nir_op_info *info = &nir_op_infos[alu->op];
   const unsigned dst_bits = alu->def.bit_size;
   bool small_src = false;
   for (unsigned i = 0; i < info->num_inputs; ++i)
      small_src |= old_size(w, alu->src[i].src.ssa) || alu->src[i].src.ssa->bit_size == 16 ||
                   alu->src[i].src.ssa->bit_size == 8;
   if (!is_small(dst_bits) && !small_src)
      return false;

   const nir_alu_type out = nir_alu_type_get_base_type(info->output_type);
   /* Half floats a float operation makes stay half floats, but for
    * conversions, and moves and selects, which are of words. */
   if (dst_bits == 16 && out == nir_type_float && !info->is_conversion)
      return lower_half_alu(b, w, alu);
   /* A float operation of half floats, into anything else: of their
    * values as 32-bit floats. */

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *r = info->is_conversion ? lower_conversion(b, w, alu, dst_bits) : lower_int_alu(b, w, alu, dst_bits);
   if (!r)
      return false;
   /* A word that was small, as a result that is not: a def of its own,
    * which is 32 bits to what reads it. */
   if (!is_small(dst_bits) && old_size(w, r))
      r = nir_mov(b, r);
   if (is_small(dst_bits))
      set_old_size(w, r, dst_bits);
   nir_def_rewrite_uses(&alu->def, r);
   nir_instr_remove(&alu->instr);
   return true;
}

/* A type of the same shape, of 32-bit words, or floats where it is
 * interpolated. */
static const struct glsl_type *
widen_type(const struct glsl_type *type, bool floats)
{
   if (glsl_type_is_array(type)) {
      const struct glsl_type *e = widen_type(glsl_get_array_element(type), floats);
      return e ? glsl_array_type(e, glsl_get_length(type), 0) : NULL;
   }
   if (glsl_type_is_struct_or_ifc(type)) {
      const unsigned n = glsl_get_length(type);
      struct glsl_struct_field *fields = calloc(n, sizeof(*fields));
      bool changed = false;
      for (unsigned i = 0; i < n; ++i) {
         fields[i] = *glsl_get_struct_field_data(type, i);
         const struct glsl_type *f = widen_type(fields[i].type, floats);
         if (f) {
            fields[i].type = f;
            changed = true;
         }
      }
      const struct glsl_type *t = changed ? glsl_struct_type(fields, n, glsl_get_type_name(type), false) : NULL;
      free(fields);
      return t;
   }
   /* A matrix of half floats: an array of its columns, widened -- which
    * its derefs, a column at a time, are of too. */
   if (glsl_type_is_matrix(type)) {
      const struct glsl_type *column = widen_type(glsl_get_column_type(type), floats);
      return column ? glsl_array_type(column, glsl_get_matrix_columns(type), 0) : NULL;
   }
   if (!glsl_type_is_vector_or_scalar(type) || !is_small(glsl_get_bit_size(type)))
      return NULL;
   const enum glsl_base_type base = glsl_get_base_type(type);
   if (base == GLSL_TYPE_FLOAT16 && floats)
      return glsl_vector_type(GLSL_TYPE_FLOAT, glsl_get_vector_elements(type));
   return glsl_vector_type(base == GLSL_TYPE_INT16 || base == GLSL_TYPE_INT8 ? GLSL_TYPE_INT : GLSL_TYPE_UINT,
                           glsl_get_vector_elements(type));
}

static bool
retype_variables(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_variable_in_shader(var, nir) {
      if (!(var->data.mode & (nir_var_shader_in | nir_var_shader_out | nir_var_shader_temp)))
         continue;
      /* Interpolated half floats as floats; flat ones, and integers, as
       * the words they are. */
      const bool floats = var->data.mode & (nir_var_shader_in | nir_var_shader_out);
      const struct glsl_type *t = widen_type(var->type, floats);
      if (t) {
         var->type = t;
         progress = true;
      }
   }
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_function_temp_variable(var, impl) {
         const struct glsl_type *t = widen_type(var->type, false);
         if (t) {
            var->type = t;
            progress = true;
         }
      }
   }
   if (progress)
      nir_fixup_deref_types(nir);
   return progress;
}

/* Loads and stores of variables retyped: floats made words, words kept to
 * their bits. */
static bool
widen_deref_access(nir_builder *b, struct widen *w, nir_intrinsic_instr *intr)
{
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (!glsl_type_is_vector_or_scalar(deref->type))
      return false;
   const enum glsl_base_type base = glsl_get_base_type(deref->type);
   if (intr->intrinsic == nir_intrinsic_load_deref) {
      if (!is_small(intr->def.bit_size))
         return false;
      const unsigned bits = intr->def.bit_size;
      intr->def.bit_size = 32;
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *word = base == GLSL_TYPE_FLOAT ? half_to_word(b, &intr->def) : keep_bits(b, &intr->def, bits);
      if (word != &intr->def)
         nir_def_rewrite_uses_after(&intr->def, word);
      set_old_size(w, word, bits);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_store_deref) {
      nir_def *value = intr->src[1].ssa;
      const unsigned bits = old_size(w, value) ? old_size(w, value) : (is_small(value->bit_size) ? value->bit_size : 0);
      if (!bits)
         return false;
      b->cursor = nir_before_instr(&intr->instr);
      /* A signed integer, sign-extended: what it goes to may be of fewer
       * bits, and clamp. */
      nir_def *v = base == GLSL_TYPE_FLOAT ? as_f32(b, w, value)
                   : base == GLSL_TYPE_INT ? sext(b, as_word(b, w, value), bits)
                                           : as_word(b, w, value);
      nir_src_rewrite(&intr->src[1], v);
      return true;
   }
   return false;
}

static bool
widen_impl(nir_function_impl *impl, nir_shader *nir)
{
   struct widen w = { .old_size = _mesa_pointer_hash_table_create(NULL) };
   util_dynarray_init(&w.phis, NULL);
   nir_builder b = nir_builder_create(impl);
   bool progress = false;

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_load_const: {
            nir_load_const_instr *lc = nir_instr_as_load_const(instr);
            if (!is_small(lc->def.bit_size))
               break;
            const unsigned bits = lc->def.bit_size;
            for (unsigned i = 0; i < lc->def.num_components; ++i) {
               const uint32_t v = bits == 8 ? lc->value[i].u8 : lc->value[i].u16;
               lc->value[i] = nir_const_value_for_uint(v, 32);
            }
            lc->def.bit_size = 32;
            set_old_size(&w, &lc->def, bits);
            progress = true;
            break;
         }
         case nir_instr_type_undef: {
            nir_undef_instr *u = nir_instr_as_undef(instr);
            if (!is_small(u->def.bit_size))
               break;
            set_old_size(&w, &u->def, u->def.bit_size);
            u->def.bit_size = 32;
            progress = true;
            break;
         }
         case nir_instr_type_phi: {
            nir_phi_instr *phi = nir_instr_as_phi(instr);
            if (!is_small(phi->def.bit_size))
               break;
            set_old_size(&w, &phi->def, phi->def.bit_size);
            phi->def.bit_size = 32;
            util_dynarray_append(&w.phis, phi);
            progress = true;
            break;
         }
         case nir_instr_type_alu:
            progress |= widen_alu(&b, &w, nir_instr_as_alu(instr));
            break;
         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_deref || intr->intrinsic == nir_intrinsic_store_deref) {
               progress |= widen_deref_access(&b, &w, intr);
               break;
            }
            if (nir_intrinsic_infos[intr->intrinsic].has_dest && is_small(intr->def.bit_size)) {
               mesa_loge("wvk: a %u-bit %s", intr->def.bit_size, nir_intrinsic_infos[intr->intrinsic].name);
               break;
            }
            /* Other sources that were small are words now, as intrinsics
             * other than those take. */
            for (unsigned i = 0; i < nir_intrinsic_infos[intr->intrinsic].num_srcs; ++i) {
               nir_def *s = intr->src[i].ssa;
               if (s->bit_size == 16 && !old_size(&w, s)) {
                  b.cursor = nir_before_instr(instr);
                  nir_src_rewrite(&intr->src[i], as_word(&b, &w, s));
                  progress = true;
               }
            }
            break;
         }
         default:
            break;
         }
      }
   }

   /* Phis' sources that are half floats: words, made where they come from. */
   util_dynarray_foreach(&w.phis, nir_phi_instr *, p) {
      nir_foreach_phi_src(src, *p) {
         if (src->src.ssa->bit_size != 16)
            continue;
         b.cursor = nir_after_block_before_jump(src->pred);
         nir_src_rewrite(&src->src, as_word(&b, &w, src->src.ssa));
      }
   }
   util_dynarray_fini(&w.phis);
   _mesa_hash_table_destroy(w.old_size, NULL);
   return nir_progress(progress, impl, nir_metadata_none);
}

bool
wvk_nir_widen_small_types(nir_shader *nir)
{
   bool progress = retype_variables(nir);
   nir_foreach_function_impl(impl, nir)
      progress |= widen_impl(impl, nir);
   return progress;
}
