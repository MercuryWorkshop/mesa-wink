/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Cooperative matrices (VK_KHR_cooperative_matrix), of subgroups of one
 * invocation (wvk_nir_subgroups.c): a matrix is all the invocation's, an
 * array of its elements row by row, and each operation is of each element
 * in turn, unrolled: of as many as the matrices' sizes have, which
 * wvk_GetPhysicalDeviceCooperativeMatrixPropertiesKHR keeps small. (Loops
 * of them over arrays filled element by element come out wrong on Metal,
 * but for 32-bit floats.)
 */
#include "wvk_nir.h"

#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"
#include "util/u_dynarray.h"

static unsigned
elements(const struct glsl_cmat_description *d)
{
   return d->rows * d->cols;
}

/* A matrix's type as an array of its elements, and an array's of them. */
static const struct glsl_type *
remap_type(struct hash_table *map, const struct glsl_type *type)
{
   struct hash_entry *entry = _mesa_hash_table_search(map, type);
   if (entry)
      return entry->data;
   const struct glsl_type *out = type;
   if (glsl_type_is_cmat(type)) {
      const struct glsl_cmat_description *d = glsl_get_cmat_description(type);
      out = glsl_array_type(glsl_scalar_type(d->element_type), elements(d), 0);
   } else if (glsl_type_is_array(type)) {
      const struct glsl_type *element = glsl_get_array_element(type);
      const struct glsl_type *remapped = remap_type(map, element);
      if (remapped != element)
         out = glsl_array_type(remapped, glsl_get_length(type), glsl_get_explicit_stride(type));
   }
   _mesa_hash_table_insert(map, type, (void *)out);
   return out;
}

static struct glsl_cmat_description
desc_of(nir_src src)
{
   return *glsl_get_cmat_description(nir_src_as_deref(src)->type);
}

/* A matrix's deref as the array it is to become. */
static nir_deref_instr *
as_array(nir_builder *b, struct hash_table *map, nir_src src)
{
   nir_deref_instr *deref = nir_src_as_deref(src);
   return nir_build_deref_cast(b, &deref->def, deref->modes, remap_type(map, deref->type), 0);
}

static nir_def *
load_element(nir_builder *b, nir_deref_instr *array, nir_def *i)
{
   return nir_load_deref(b, nir_build_deref_array(b, array, i));
}

static void
store_element(nir_builder *b, nir_deref_instr *array, nir_def *i, nir_def *value)
{
   nir_store_deref(b, nir_build_deref_array(b, array, i), value, 0x1);
}

/* A value of `from` as `to`, of the signedness given of each. */
static nir_def *
convert(nir_builder *b, nir_def *value, enum glsl_base_type from, bool from_signed, enum glsl_base_type to,
        bool to_signed)
{
   from = glsl_apply_signedness_to_base_type(from, from_signed);
   to = glsl_apply_signedness_to_base_type(to, to_signed);
   const nir_alu_type src = nir_get_nir_type_for_glsl_base_type(from);
   const nir_alu_type dst = nir_get_nir_type_for_glsl_base_type(to);
   if (src == dst)
      return value;
   const nir_op op = nir_type_conversion_op(src, dst, nir_rounding_mode_undef);
   return op == nir_op_mov ? value : nir_build_alu1(b, op, value);
}

static void
lower_load_store(nir_builder *b, struct hash_table *map, nir_intrinsic_instr *intr)
{
   const bool load = intr->intrinsic == nir_intrinsic_cmat_load;
   const nir_src mat_src = intr->src[load ? 0 : 1];
   const struct glsl_cmat_description d = desc_of(mat_src);
   nir_deref_instr *ptr = nir_src_as_deref(intr->src[load ? 1 : 0]);
   nir_def *stride = intr->src[2].ssa;
   const bool col_major = nir_intrinsic_matrix_layout(intr) == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR;
   const unsigned element_bytes = glsl_base_type_bit_size(d.element_type) / 8;
   /* The stride is of what the pointer is to. */
   const uint32_t ptr_stride = glsl_get_bit_size(ptr->type) / 8 * glsl_get_vector_elements(ptr->type);
   ptr = nir_build_deref_cast(b, &ptr->def, ptr->modes, ptr->type, ptr_stride);
   nir_deref_instr *array = as_array(b, map, mat_src);
   const unsigned index_bits = ptr->def.bit_size;

   for (unsigned e = 0; e < elements(&d); ++e) {
      nir_def *i = nir_imm_int(b, e);
      const unsigned row = e / d.cols, col = e % d.cols;
      const unsigned major = col_major ? col : row, minor = col_major ? row : col;
      nir_deref_instr *at =
         nir_build_deref_ptr_as_array(b, ptr, nir_u2uN(b, nir_imul_imm(b, stride, major), index_bits));
      at = nir_build_deref_cast(b, &at->def, ptr->modes, glsl_scalar_type(d.element_type), element_bytes);
      at = nir_build_deref_ptr_as_array(b, at, nir_imm_intN_t(b, minor, index_bits));
      if (load)
         store_element(b, array, i, nir_load_deref(b, at));
      else
         nir_store_deref(b, at, load_element(b, array, i), 0x1);
   }
   nir_instr_remove(&intr->instr);
}

/* Each element of the result of `op` of each of the sources' -- or the
 * scalar, for a scalar operation. */
static void
lower_elementwise(nir_builder *b, struct hash_table *map, nir_intrinsic_instr *intr)
{
   const struct glsl_cmat_description d = desc_of(intr->src[0]);
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   nir_deref_instr *dst = as_array(b, map, intr->src[0]);
   nir_deref_instr *a = as_array(b, map, intr->src[1]);
   nir_deref_instr *other = intr->intrinsic == nir_intrinsic_cmat_binary_op ? as_array(b, map, intr->src[2]) : NULL;
   const nir_op op = nir_intrinsic_alu_op(intr);

   for (unsigned e = 0; e < elements(&d); ++e) {
      nir_def *i = nir_imm_int(b, e);
      nir_def *x = load_element(b, a, i);
      nir_def *value;
      switch (intr->intrinsic) {
      case nir_intrinsic_cmat_unary_op:
         value = nir_build_alu1(b, op, x);
         break;
      case nir_intrinsic_cmat_binary_op:
         value = nir_build_alu2(b, op, x, load_element(b, other, i));
         break;
      default:
         value = nir_build_alu2(b, op, x, intr->src[2].ssa);
         break;
      }
      store_element(b, dst, i, value);
   }
   b->fp_math_ctrl = nir_fp_fast_math;
   nir_instr_remove(&intr->instr);
}

/* Of another element type, or use -- which is only a name here -- or
 * transposed. */
static void
lower_convert(nir_builder *b, struct hash_table *map, nir_intrinsic_instr *intr)
{
   const bool transpose = intr->intrinsic == nir_intrinsic_cmat_transpose;
   const struct glsl_cmat_description dd = desc_of(intr->src[0]), sd = desc_of(intr->src[1]);
   const nir_cmat_signed sign = nir_intrinsic_cmat_signed_mask(intr);
   b->fp_math_ctrl = nir_intrinsic_fp_math_ctrl(intr);
   nir_deref_instr *dst = as_array(b, map, intr->src[0]);
   nir_deref_instr *src = as_array(b, map, intr->src[1]);

   for (unsigned e = 0; e < elements(&dd); ++e) {
      const unsigned from = transpose ? (e % dd.cols) * sd.cols + e / dd.cols : e;
      nir_def *value = convert(b, load_element(b, src, nir_imm_int(b, from)), sd.element_type,
                               sign & NIR_CMAT_A_SIGNED, dd.element_type, sign & NIR_CMAT_RESULT_SIGNED);
      store_element(b, dst, nir_imm_int(b, e), value);
   }
   b->fp_math_ctrl = nir_fp_fast_math;
   nir_instr_remove(&intr->instr);
}

/* The same bits, of another type. */
static void
lower_bitcast(nir_builder *b, struct hash_table *map, nir_intrinsic_instr *intr)
{
   const struct glsl_cmat_description d = desc_of(intr->src[0]);
   nir_deref_instr *dst = as_array(b, map, intr->src[0]);
   nir_deref_instr *src = as_array(b, map, intr->src[1]);
   for (unsigned e = 0; e < elements(&d); ++e)
      store_element(b, dst, nir_imm_int(b, e), load_element(b, src, nir_imm_int(b, e)));
   nir_instr_remove(&intr->instr);
}

/* x + y, of `bits`, held at the type's limits: of the sign each has. */
static nir_def *
add_saturated(nir_builder *b, nir_def *x, nir_def *y, bool is_signed)
{
   nir_def *sum = nir_iadd(b, x, y);
   if (!is_signed)
      return nir_bcsel(b, nir_ult(b, sum, x), nir_imm_intN_t(b, -1, x->bit_size), sum);
   const unsigned bits = x->bit_size;
   nir_def *over = nir_ilt_imm(b, nir_iand(b, nir_ixor(b, x, sum), nir_ixor(b, y, sum)), 0);
   nir_def *limit = nir_bcsel(b, nir_ilt_imm(b, x, 0), nir_imm_intN_t(b, u_intN_min(bits), bits),
                              nir_imm_intN_t(b, u_intN_max(bits), bits));
   return nir_bcsel(b, over, limit, sum);
}

/* D = A * B + C: each of D's elements the dot product of A's row and B's
 * column, in the accumulator's type, and C's -- added saturating, where
 * the operation says (saturatingAccumulation), to the sum as it wraps. */
static void
lower_muladd(nir_builder *b, struct hash_table *map, nir_intrinsic_instr *intr)
{
   const struct glsl_cmat_description ad = desc_of(intr->src[1]), bd = desc_of(intr->src[2]);
   const struct glsl_cmat_description cd = desc_of(intr->src[3]), rd = desc_of(intr->src[0]);
   const nir_cmat_signed sign = nir_intrinsic_cmat_signed_mask(intr);
   nir_deref_instr *dst = as_array(b, map, intr->src[0]);
   nir_deref_instr *a = as_array(b, map, intr->src[1]);
   nir_deref_instr *bm = as_array(b, map, intr->src[2]);
   nir_deref_instr *c = as_array(b, map, intr->src[3]);
   const bool integer = glsl_base_type_is_integer(rd.element_type);
   const bool result_signed = sign & NIR_CMAT_RESULT_SIGNED;
   const bool saturate = integer && nir_intrinsic_saturate(intr);
   for (unsigned e = 0; e < elements(&rd); ++e) {
      const unsigned row = e / rd.cols, col = e % rd.cols;
      nir_def *addend = convert(b, load_element(b, c, nir_imm_int(b, e)), cd.element_type, sign & NIR_CMAT_C_SIGNED,
                                rd.element_type, result_signed);
      nir_def *acc = saturate ? nir_imm_intN_t(b, 0, addend->bit_size) : addend;
      for (unsigned k = 0; k < ad.cols; ++k) {
         nir_def *x = load_element(b, a, nir_imm_int(b, row * ad.cols + k));
         nir_def *y = load_element(b, bm, nir_imm_int(b, k * bd.cols + col));
         x = convert(b, x, ad.element_type, sign & NIR_CMAT_A_SIGNED, rd.element_type, result_signed);
         y = convert(b, y, bd.element_type, sign & NIR_CMAT_B_SIGNED, rd.element_type, result_signed);
         acc = integer ? nir_iadd(b, acc, nir_imul(b, x, y)) : nir_ffma(b, x, y, acc);
      }
      if (saturate)
         acc = add_saturated(b, acc, addend, result_signed);
      store_element(b, dst, nir_imm_int(b, e), acc);
   }
   nir_instr_remove(&intr->instr);
}

static bool
is_cmat_intrinsic(const nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_length:
   case nir_intrinsic_cmat_construct:
   case nir_intrinsic_cmat_extract:
   case nir_intrinsic_cmat_insert:
   case nir_intrinsic_cmat_copy:
   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store:
   case nir_intrinsic_cmat_unary_op:
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op:
   case nir_intrinsic_cmat_convert:
   case nir_intrinsic_cmat_transpose:
   case nir_intrinsic_cmat_bitcast:
   case nir_intrinsic_cmat_muladd:
      return true;
   default:
      return false;
   }
}

static void
lower_intrinsic(nir_builder *b, struct hash_table *map, nir_intrinsic_instr *intr)
{
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_cmat_length: {
      const struct glsl_cmat_description d = nir_intrinsic_cmat_desc(intr);
      nir_def_replace(&intr->def, nir_imm_int(b, elements(&d)));
      break;
   }
   case nir_intrinsic_cmat_construct: {
      const struct glsl_cmat_description d = desc_of(intr->src[0]);
      nir_deref_instr *dst = as_array(b, map, intr->src[0]);
      for (unsigned e = 0; e < elements(&d); ++e)
         store_element(b, dst, nir_imm_int(b, e), intr->src[1].ssa);
      nir_instr_remove(&intr->instr);
      break;
   }
   case nir_intrinsic_cmat_extract:
      nir_def_replace(&intr->def, load_element(b, as_array(b, map, intr->src[0]), intr->src[1].ssa));
      break;
   case nir_intrinsic_cmat_insert: {
      /* The source's elements, and the one. */
      const struct glsl_cmat_description d = desc_of(intr->src[0]);
      nir_deref_instr *dst = as_array(b, map, intr->src[0]);
      nir_deref_instr *src = as_array(b, map, intr->src[2]);
      for (unsigned e = 0; e < elements(&d); ++e) {
         nir_def *i = nir_imm_int(b, e);
         nir_def *value = nir_bcsel(b, nir_ieq(b, i, intr->src[3].ssa), intr->src[1].ssa, load_element(b, src, i));
         store_element(b, dst, i, value);
      }
      nir_instr_remove(&intr->instr);
      break;
   }
   case nir_intrinsic_cmat_copy:
      nir_copy_deref(b, as_array(b, map, intr->src[0]), as_array(b, map, intr->src[1]));
      nir_instr_remove(&intr->instr);
      break;
   case nir_intrinsic_cmat_load:
   case nir_intrinsic_cmat_store:
      lower_load_store(b, map, intr);
      break;
   case nir_intrinsic_cmat_unary_op:
   case nir_intrinsic_cmat_binary_op:
   case nir_intrinsic_cmat_scalar_op:
      lower_elementwise(b, map, intr);
      break;
   case nir_intrinsic_cmat_convert:
   case nir_intrinsic_cmat_transpose:
      lower_convert(b, map, intr);
      break;
   case nir_intrinsic_cmat_bitcast:
      lower_bitcast(b, map, intr);
      break;
   case nir_intrinsic_cmat_muladd:
      lower_muladd(b, map, intr);
      break;
   default:
      UNREACHABLE("not a cooperative matrix intrinsic");
   }
}

/* The operations first, while their derefs have the matrices' types; then
 * the variables and derefs, arrays. */
static bool
lower_impl(nir_function_impl *impl, struct hash_table *map)
{
   struct util_dynarray ops;
   util_dynarray_init(&ops, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic && is_cmat_intrinsic(nir_instr_as_intrinsic(instr)))
            util_dynarray_append(&ops, nir_instr_as_intrinsic(instr));
      }
   }
   bool progress = ops.size != 0;
   nir_builder b = nir_builder_create(impl);
   util_dynarray_foreach(&ops, nir_intrinsic_instr *, intr)
      lower_intrinsic(&b, map, *intr);
   util_dynarray_fini(&ops);

   nir_foreach_function_temp_variable(var, impl) {
      const struct glsl_type *type = remap_type(map, var->type);
      progress |= type != var->type;
      var->type = type;
   }
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         const struct glsl_type *type = remap_type(map, deref->type);
         progress |= type != deref->type;
         deref->type = type;
      }
   }
   return nir_progress(progress, impl, nir_metadata_none);
}

bool
wvk_nir_lower_cooperative_matrix(nir_shader *nir)
{
   if (!mesa_shader_stage_is_compute(nir->info.stage) || !nir->info.cs.has_cooperative_matrix)
      return false;
   struct hash_table *map = _mesa_pointer_hash_table_create(NULL);
   bool progress = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_temp) {
      const struct glsl_type *type = remap_type(map, var->type);
      progress |= type != var->type;
      var->type = type;
   }
   nir_foreach_function_impl(impl, nir)
      progress |= lower_impl(impl, map);
   _mesa_hash_table_destroy(map, NULL);
   return progress;
}
