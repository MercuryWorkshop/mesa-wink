/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * bfloat16 (VK_KHR_shader_bfloat16) and 8-bit floats (VK_EXT_shader_float8,
 * E4M3 and E5M2), which WGSL has none of: their conversions to and from
 * 32-bit floats, which is all a shader does with them outside cooperative
 * matrices (wvk_nir_cmat.c, which converts through these), as integer
 * arithmetic on their bits -- rounding to nearest even, as
 * util/bfloat.h's and util/float8.c's do.
 */
#include "wvk_nir.h"

#include "compiler/nir/nir_builder.h"

/* bfloat16's bits: a float's upper half. */
static nir_def *
bf2f(nir_builder *b, nir_def *x)
{
   return nir_ishl_imm(b, nir_u2u32(b, x), 16);
}

/* The upper half rounded by the lower, a NaN made quiet. */
static nir_def *
f2bf(nir_builder *b, nir_def *f)
{
   nir_def *bits = f;
   nir_def *upper = nir_ushr_imm(b, bits, 16);
   nir_def *rounded = nir_ushr_imm(b, nir_iadd(b, bits, nir_iadd_imm(b, nir_iand_imm(b, upper, 1), 0x7fff)), 16);
   nir_def *nan = nir_fneu(b, f, f);
   return nir_u2u16(b, nir_bcsel(b, nan, nir_ior_imm(b, upper, 1 << 6), rounded));
}

/* A float of an 8-bit one's sign, `exp_bits` of exponent and `man_bits` of
 * mantissa: its bits, but of the smallest exponent, which are a multiple
 * of the smallest denormal. */
static nir_def *
small_to_float(nir_builder *b, nir_def *x, unsigned exp_bits, unsigned man_bits)
{
   const unsigned bias = (1u << (exp_bits - 1)) - 1;
   x = nir_u2u32(b, x);
   nir_def *sign = nir_ishl_imm(b, nir_iand_imm(b, x, 0x80), 24);
   nir_def *e = nir_iand_imm(b, nir_ushr_imm(b, x, man_bits), BITFIELD_MASK(exp_bits));
   nir_def *m = nir_iand_imm(b, x, BITFIELD_MASK(man_bits));
   nir_def *normal =
      nir_ior(b, sign, nir_ior(b, nir_ishl_imm(b, nir_iadd_imm(b, e, 127 - bias), 23), nir_ishl_imm(b, m, 23 - man_bits)));
   nir_def *denormal = nir_fmul_imm(b, nir_u2f32(b, m), ldexp(1.0, 1 - (int)bias - (int)man_bits));
   denormal = nir_ior(b, denormal, sign);
   return nir_bcsel(b, nir_ieq_imm(b, e, 0), denormal, normal);
}

static nir_def *
e4m3fn2f(nir_builder *b, nir_def *x)
{
   /* No infinities; all ones is NaN. */
   nir_def *f = small_to_float(b, x, 4, 3);
   return nir_bcsel(b, nir_ieq_imm(b, nir_iand_imm(b, nir_u2u32(b, x), 0x7f), 0x7f), nir_imm_int(b, 0xffc00000), f);
}

static nir_def *
e5m22f(nir_builder *b, nir_def *x)
{
   nir_def *f = small_to_float(b, x, 5, 2);
   nir_def *bits = nir_u2u32(b, x);
   nir_def *top = nir_ieq_imm(b, nir_iand_imm(b, bits, 0x7c), 0x7c);
   nir_def *inf = nir_ior_imm(b, nir_ishl_imm(b, nir_iand_imm(b, bits, 0x80), 24), 0x7f800000);
   nir_def *nan = nir_ine_imm(b, nir_iand_imm(b, bits, 0x3), 0);
   return nir_bcsel(b, top, nir_bcsel(b, nan, nir_imm_int(b, 0xffc00000), inf), f);
}

/* A float as an 8-bit one's bits, rounded to nearest even: of the
 * magnitude, its sign after. Of more than `overflow`, `too_big`. */
static nir_def *
float_to_small(nir_builder *b, nir_def *f, unsigned exp_bits, unsigned man_bits, double overflow, nir_def *too_big)
{
   const unsigned bias = (1u << (exp_bits - 1)) - 1;
   const unsigned drop = 23 - man_bits;
   nir_def *a = nir_fabs(b, f);
   nir_def *sign = nir_iand_imm(b, nir_ushr_imm(b, f, 24), 0x80);
   /* Normal: the mantissa rounded by what is dropped of it. */
   nir_def *rounded = nir_iadd(b, a, nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, a, drop), 1),
                                                  (1u << (drop - 1)) - 1));
   nir_def *e = nir_iadd_imm(b, nir_ushr_imm(b, rounded, 23), (int)bias - 127);
   nir_def *normal = nir_ior(b, nir_ishl_imm(b, e, man_bits), nir_iand_imm(b, nir_ushr_imm(b, rounded, drop),
                                                                             BITFIELD_MASK(man_bits)));
   /* Denormal: of the smallest denormal, rounded -- one of the smallest
    * normal's is its bits too. */
   nir_def *denormal = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, a, ldexp(1.0, (int)bias - 1 + (int)man_bits))));
   nir_def *value = nir_bcsel(b, nir_flt_imm(b, a, ldexp(1.0, 1 - (int)bias)), denormal, normal);
   value = nir_ior(b, value, sign);
   return nir_bcsel(b, nir_fgt_imm(b, a, overflow), too_big, value);
}

static nir_def *
f2e4m3fn(nir_builder *b, nir_def *f, nir_op op)
{
   /* Finite only: too big is NaN. */
   nir_def *v = float_to_small(b, f, 4, 3, 464.0, nir_imm_int(b, 0xff));
   v = nir_bcsel(b, nir_fneu(b, f, f), nir_imm_int(b, 0xff), v);
   if (op != nir_op_f2e4m3fn) {
      v = nir_bcsel(b, nir_fgt_imm(b, f, 448.0), nir_imm_int(b, 0x7e), v);
      v = nir_bcsel(b, nir_flt_imm(b, f, -448.0), nir_imm_int(b, 0xfe), v);
   }
   if (op == nir_op_f2e4m3fn_satfn)
      v = nir_bcsel(b, nir_feq_imm(b, nir_fabs(b, f), INFINITY), nir_imm_int(b, 0x7f), v);
   return nir_u2u8(b, v);
}

static nir_def *
f2e5m2(nir_builder *b, nir_def *f, nir_op op)
{
   /* At 61440 and past, infinity. */
   nir_def *sign = nir_iand_imm(b, nir_ushr_imm(b, f, 24), 0x80);
   nir_def *v = float_to_small(b, f, 5, 2, 61440.0, nir_ior_imm(b, sign, 0x7c));
   v = nir_bcsel(b, nir_feq_imm(b, nir_fabs(b, f), 61440.0), nir_ior_imm(b, sign, 0x7c), v);
   v = nir_bcsel(b, nir_fneu(b, f, f), nir_imm_int(b, 0xfe), v);
   if (op == nir_op_f2e5m2_sat) {
      v = nir_bcsel(b, nir_fgt_imm(b, f, 57344.0), nir_imm_int(b, 0x7b), v);
      v = nir_bcsel(b, nir_flt_imm(b, f, -57344.0), nir_imm_int(b, 0xfb), v);
   }
   return nir_u2u8(b, v);
}

static bool
lower_instr(nir_builder *b, nir_alu_instr *alu, void *data)
{
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *src = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *value;
   switch (alu->op) {
   case nir_op_bf2f:
      value = bf2f(b, src);
      break;
   case nir_op_f2bf:
      value = f2bf(b, src);
      break;
   case nir_op_e4m3fn2f:
      value = e4m3fn2f(b, src);
      break;
   case nir_op_e5m22f:
      value = e5m22f(b, src);
      break;
   case nir_op_f2e4m3fn:
   case nir_op_f2e4m3fn_sat:
   case nir_op_f2e4m3fn_satfn:
      value = f2e4m3fn(b, src, alu->op);
      break;
   case nir_op_f2e5m2:
   case nir_op_f2e5m2_sat:
      value = f2e5m2(b, src, alu->op);
      break;
   default:
      return false;
   }
   nir_def_replace(&alu->def, value);
   return true;
}

bool
wvk_nir_lower_minifloats(nir_shader *nir)
{
   return nir_shader_alu_pass(nir, lower_instr, nir_metadata_control_flow, NULL);
}
