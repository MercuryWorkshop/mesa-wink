/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Coarse shading (VK_KHR_fragment_shading_rate), which WebGPU has none of.
 * A coarse fragment is at most 2x2 here, the pixels of a quad, or a row or
 * column of one: each of them shades as the fragment would, at its centre
 * -- its position and interpolated inputs are the fragment's, so that it
 * computes what the fragment does, and writes that to its own pixel.
 *
 * Its derivatives are across fragments, not pixels: each derivative's
 * operand computed again of the pixel's own position and inputs, whose
 * fine derivative is the pixel's, scaled by the fragment's size; and an
 * implicit LOD is of those. What the fragment does that is seen besides
 * its outputs -- storage writes, atomics -- its first covered pixel does
 * alone, and the others read an atomic's result of it across the quad,
 * through derivatives, which are differences of the quad's pixels.
 */
#include "wvk_nir.h"

#include "wvk_descriptor_set_layout.h"

#include "compiler/nir/nir_builder.h"
#include "util/hash_table.h"

struct fsr_state {
   nir_function_impl *impl;
   /* The fragment's size, and as floats. */
   nir_def *size, *sizef;
   /* Whether the pixel is the right, or lower, one of its quad's. */
   nir_def *odd_x, *odd_y;
   /* Whether it is the fragment's first covered pixel, and where that is
    * from it: across in x, in y. */
   nir_def *leader, *lead_x, *lead_y;
   /* What each input and position of the fragment's is of the pixel's. */
   struct hash_table *own;
};

static const struct glsl_type *
bare(const nir_variable *var)
{
   return glsl_without_array(var->type);
}

/* A 32-bit value of the pixel across the quad from this one, in x or y:
 * the fine derivative is the difference of the two, exact of 16-bit
 * halves as floats. */
static nir_def *
across32(nir_builder *b, struct fsr_state *s, nir_def *v, bool y)
{
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < v->num_components; ++c) {
      nir_def *u = nir_channel(b, v, c);
      nir_def *halves[2] = { nir_iand_imm(b, u, 0xffff), nir_ushr_imm(b, u, 16) };
      for (unsigned h = 0; h < 2; ++h) {
         nir_def *f = nir_u2f32(b, halves[h]);
         nir_def *d = y ? nir_ddy_fine(b, f) : nir_ddx_fine(b, f);
         nir_def *other = nir_bcsel(b, y ? s->odd_y : s->odd_x, nir_fsub(b, f, d), nir_fadd(b, f, d));
         halves[h] = nir_f2u32(b, nir_fround_even(b, other));
      }
      comps[c] = nir_ior(b, halves[0], nir_ishl_imm(b, halves[1], 16));
   }
   return nir_vec(b, comps, v->num_components);
}

static nir_def *
across(nir_builder *b, struct fsr_state *s, nir_def *v, bool y)
{
   switch (v->bit_size) {
   case 1:
      return nir_i2b(b, across32(b, s, nir_b2i32(b, v), y));
   case 64: {
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned c = 0; c < v->num_components; ++c)
         comps[c] = nir_pack_64_2x32(b, across32(b, s, nir_unpack_64_2x32(b, nir_channel(b, v, c)), y));
      return nir_vec(b, comps, v->num_components);
   }
   case 32:
      return across32(b, s, v, y);
   default:
      return nir_u2uN(b, across32(b, s, nir_u2u32(b, v), y), v->bit_size);
   }
}

/* The leader's value of `v`, which only it computed. */
static nir_def *
from_leader(nir_builder *b, struct fsr_state *s, nir_def *v)
{
   nir_def *x = across(b, s, v, false);
   nir_def *y = across(b, s, v, true);
   nir_def *xy = across(b, s, x, true);
   return nir_bcsel(b, s->lead_x, nir_bcsel(b, s->lead_y, xy, x), nir_bcsel(b, s->lead_y, y, v));
}

/* `def` computed again of the pixel's own position and inputs, before
 * the builder's cursor: ALU of what is, what else there is as it is. */
static nir_def *
own_value(nir_builder *b, struct hash_table *map, nir_def *def)
{
   struct hash_entry *he = _mesa_hash_table_search(map, def);
   if (he)
      return he->data;
   nir_def *result = def;
   nir_instr *instr = nir_def_instr(def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      const unsigned n = nir_op_infos[alu->op].num_inputs;
      nir_def *srcs[NIR_ALU_MAX_INPUTS];
      bool changed = false;
      for (unsigned i = 0; i < n; ++i) {
         srcs[i] = own_value(b, map, alu->src[i].src.ssa);
         changed |= srcs[i] != alu->src[i].src.ssa;
      }
      if (changed) {
         nir_alu_instr *copy = nir_alu_instr_create(b->shader, alu->op);
         copy->fp_math_ctrl = alu->fp_math_ctrl;
         copy->no_signed_wrap = alu->no_signed_wrap;
         copy->no_unsigned_wrap = alu->no_unsigned_wrap;
         for (unsigned i = 0; i < n; ++i) {
            copy->src[i].src = nir_src_for_ssa(srcs[i]);
            memcpy(copy->src[i].swizzle, alu->src[i].swizzle, sizeof(alu->src[i].swizzle));
         }
         nir_def_init(&copy->instr, &copy->def, alu->def.num_components, alu->def.bit_size);
         nir_builder_instr_insert(b, &copy->instr);
         result = &copy->def;
      }
   }
   _mesa_hash_table_insert(map, def, result);
   return result;
}

static bool
is_derivative(nir_intrinsic_op op, bool *y)
{
   switch (op) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddx_coarse:
      *y = false;
      return true;
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddy_coarse:
      *y = true;
      return true;
   default:
      return false;
   }
}

/* Whether an intrinsic does what is seen besides the fragment's outputs:
 * a storage write or an atomic. */
static bool
is_side_effect(nir_intrinsic_instr *intr)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_store_deref:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
      return nir_deref_mode_may_be(nir_src_as_deref(intr->src[0]),
                                   nir_var_mem_ssbo | nir_var_mem_global | nir_var_image);
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_store_global:
   case nir_intrinsic_global_atomic:
   case nir_intrinsic_global_atomic_swap:
      return true;
   default:
      return false;
   }
}

/* A shading rate's log2 of width and height (Vulkan's encoding: of
 * width in bits 2-3, of height in bits 0-1). */
static void
rate_sizes(nir_builder *b, nir_def *rate, nir_def **w, nir_def **h)
{
   *w = nir_iand_imm(b, nir_ushr_imm(b, rate, 2), 3);
   *h = nir_iand_imm(b, rate, 3);
}

/* The rates combined by a combiner op (VkFragmentShadingRateCombinerOpKHR),
 * of the driver's data: of log2s, a product is a sum. */
static nir_def *
combined(nir_builder *b, nir_def *op, nir_def *a, nir_def *other)
{
   nir_def *v = a;
   v = nir_bcsel(b, nir_ieq_imm(b, op, VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR), other, v);
   v = nir_bcsel(b, nir_ieq_imm(b, op, VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MIN_KHR), nir_umin(b, a, other), v);
   v = nir_bcsel(b, nir_ieq_imm(b, op, VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_KHR), nir_umax(b, a, other), v);
   return nir_bcsel(b, nir_ieq_imm(b, op, VK_FRAGMENT_SHADING_RATE_COMBINER_OP_MUL_KHR), nir_iadd(b, a, other), v);
}

static void
combine(nir_builder *b, nir_def *op, nir_def **w, nir_def **h, nir_def *bw, nir_def *bh)
{
   *w = combined(b, op, *w, bw);
   *h = combined(b, op, *h, bh);
}

bool
wvk_nir_lower_fragment_shading_rate(nir_shader *nir, const struct wvk_fsr_options *o)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   /* No coarse shading: gl_ShadingRateEXT is 1x1's. */
   if (!o || !o->enabled) {
      bool progress = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_load_frag_shading_rate)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def_replace(&nir_instr_as_intrinsic(instr)->def, nir_imm_int(&b, 0));
            progress = true;
         }
      }
      return nir_progress(progress, impl, nir_metadata_control_flow);
   }

   /* What there is to do again, gathered first: the shader's own
    * derivatives, implicit LODs and side effects, not those made here. */
   struct util_dynarray derivatives, samples, effects, positions, inputs;
   util_dynarray_init(&derivatives, NULL);
   util_dynarray_init(&samples, NULL);
   util_dynarray_init(&effects, NULL);
   util_dynarray_init(&positions, NULL);
   util_dynarray_init(&inputs, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_tex) {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if ((tex->op == nir_texop_tex || tex->op == nir_texop_txb) && !tex->is_shadow &&
                tex->sampler_dim != GLSL_SAMPLER_DIM_MS && tex->sampler_dim != GLSL_SAMPLER_DIM_BUF &&
                tex->sampler_dim != GLSL_SAMPLER_DIM_SUBPASS && tex->sampler_dim != GLSL_SAMPLER_DIM_SUBPASS_MS &&
                nir_tex_instr_src_index(tex, nir_tex_src_coord) >= 0)
               util_dynarray_append(&samples, tex);
            continue;
         }
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         bool y;
         if (is_derivative(intr->intrinsic, &y)) {
            util_dynarray_append(&derivatives, intr);
         } else if (is_side_effect(intr)) {
            util_dynarray_append(&effects, intr);
         } else if (intr->intrinsic == nir_intrinsic_load_frag_coord) {
            util_dynarray_append(&positions, intr);
         } else if (intr->intrinsic == nir_intrinsic_load_deref) {
            nir_variable *var = nir_intrinsic_get_var(intr, 0);
            if (!var || var->data.mode != nir_var_shader_in)
               continue;
            if (var->data.location == VARYING_SLOT_POS)
               util_dynarray_append(&positions, intr);
            else if (var->data.location >= VARYING_SLOT_VAR0 && var->data.interpolation != INTERP_MODE_FLAT &&
                     !var->data.fb_fetch_output && glsl_type_is_float_16_32(glsl_without_array_or_matrix(bare(var))))
               util_dynarray_append(&inputs, intr);
         }
      }
   }

   struct fsr_state s = { .impl = impl, .own = _mesa_pointer_hash_table_create(NULL) };

   /* The rate: the pipeline's, the primitive's and the attachment's,
    * combined, and at most 2x2. */
   nir_def *index = nir_vulkan_resource_index(&b, 2, 32, nir_imm_int(&b, 0), .desc_set = WVK_DRIVER_GROUP,
                                              .binding = WVK_DRIVER_SLOT_PUSH,
                                              .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *desc = nir_load_vulkan_descriptor(&b, 2, 32, index, .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *pipeline = nir_load_ssbo(&b, 1, 32, nir_channel(&b, desc, 0), nir_imm_int(&b, WVK_DRIVER_FSR),
                                     .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
   nir_def *op0 = nir_iand_imm(&b, nir_ushr_imm(&b, pipeline, 8), 0xff);
   nir_def *op1 = nir_iand_imm(&b, nir_ushr_imm(&b, pipeline, 16), 0xff);
   nir_def *w, *h, *pw, *ph;
   rate_sizes(&b, pipeline, &w, &h);
   nir_def *primitive = nir_imm_int(&b, 0);
   if (o->varying) {
      nir_variable *var = nir_variable_create(nir, nir_var_shader_in, glsl_int_type(), "wvk_shading_rate");
      var->data.location = VARYING_SLOT_VAR0 + o->varying - 1;
      var->data.interpolation = INTERP_MODE_FLAT;
      primitive = nir_load_var(&b, var);
   }
   rate_sizes(&b, primitive, &pw, &ph);
   combine(&b, op0, &w, &h, pw, ph);
   nir_def *aw = nir_imm_int(&b, 0), *ah = nir_imm_int(&b, 0);
   if (o->attachment)
      rate_sizes(&b, o->attachment(&b, o->data), &aw, &ah);
   combine(&b, op1, &w, &h, aw, ah);
   w = nir_umin(&b, w, nir_imm_int(&b, 1));
   h = nir_umin(&b, h, nir_imm_int(&b, 1));
   nir_def *rate = nir_ior(&b, nir_ishl_imm(&b, w, 2), h);
   s.size = nir_vec2(&b, nir_ishl(&b, nir_imm_int(&b, 1), w), nir_ishl(&b, nir_imm_int(&b, 1), h));
   s.sizef = nir_i2f32(&b, s.size);

   /* The fragment's centre, and where the pixel is in its quad. */
   nir_def *coord = nir_load_frag_coord(&b);
   nir_def *pixel = nir_f2i32(&b, nir_ffloor(&b, nir_trim_vector(&b, coord, 2)));
   nir_def *origin = nir_iand(&b, pixel, nir_inot(&b, nir_iadd_imm(&b, s.size, -1)));
   nir_def *centre = nir_fadd(&b, nir_i2f32(&b, origin), nir_fmul_imm(&b, s.sizef, 0.5));
   nir_def *offset = nir_fsub(&b, centre, nir_trim_vector(&b, coord, 2));
   nir_def *position = nir_vec4(&b, nir_channel(&b, centre, 0), nir_channel(&b, centre, 1), nir_channel(&b, coord, 2),
                                nir_channel(&b, coord, 3));
   _mesa_hash_table_insert(s.own, position, coord);
   s.odd_x = nir_i2b(&b, nir_iand_imm(&b, nir_channel(&b, pixel, 0), 1));
   s.odd_y = nir_i2b(&b, nir_iand_imm(&b, nir_channel(&b, pixel, 1), 1));

   /* Its first covered pixel, of the quad's in order, those of the
    * fragment. */
   nir_def *covered = nir_b2i32(&b, nir_ine_imm(&b, nir_load_sample_mask_in(&b), 0));
   nir_def *covered_x = across(&b, &s, covered, false);
   nir_def *covered_y = across(&b, &s, covered, true);
   nir_def *covered_xy = across(&b, &s, covered_x, true);
   nir_def *wide = nir_ieq_imm(&b, w, 1), *tall = nir_ieq_imm(&b, h, 1);
   nir_def *found = nir_imm_false(&b);
   s.lead_x = nir_imm_false(&b);
   s.lead_y = nir_imm_false(&b);
   for (unsigned p = 0; p < 4; ++p) {
      nir_def *dx = (p & 1) ? nir_inot(&b, s.odd_x) : s.odd_x;
      nir_def *dy = (p & 2) ? nir_inot(&b, s.odd_y) : s.odd_y;
      nir_def *in = nir_iand(&b, nir_ior(&b, nir_inot(&b, dx), wide), nir_ior(&b, nir_inot(&b, dy), tall));
      nir_def *cov = nir_bcsel(&b, dx, nir_bcsel(&b, dy, covered_xy, covered_x), nir_bcsel(&b, dy, covered_y, covered));
      nir_def *take = nir_iand(&b, nir_iand(&b, in, nir_i2b(&b, cov)), nir_inot(&b, found));
      s.lead_x = nir_bcsel(&b, take, dx, s.lead_x);
      s.lead_y = nir_bcsel(&b, take, dy, s.lead_y);
      found = nir_ior(&b, found, take);
   }
   s.leader = nir_iand(&b, found, nir_inot(&b, nir_ior(&b, s.lead_x, s.lead_y)));

   /* Interpolated inputs at the centre: of the pixel's, by their
    * derivatives -- of those read whole at the start, where every pixel
    * of the quad is. */
   struct hash_table *read = _mesa_pointer_hash_table_create(NULL);
   util_dynarray_foreach(&inputs, nir_intrinsic_instr *, at) {
      nir_intrinsic_instr *load = *at;
      nir_deref_instr *deref = nir_src_as_deref(load->src[0]);
      const bool whole = deref->deref_type == nir_deref_type_var;
      nir_def *own = NULL, *moved = NULL;
      struct hash_entry *he = whole ? _mesa_hash_table_search(read, deref->var) : NULL;
      if (he) {
         nir_def_replace(&load->def, he->data);
         continue;
      }
      b.cursor = whole ? nir_after_instr(nir_def_instr(offset)) : nir_after_instr(&load->instr);
      own = whole ? nir_load_var(&b, deref->var) : &load->def;
      if (own->num_components != load->def.num_components || own->bit_size != load->def.bit_size) {
         if (whole)
            nir_instr_remove(nir_def_instr(own));
         b.cursor = nir_after_instr(&load->instr);
         own = &load->def;
      }
      nir_def *off = own->bit_size == 32 ? offset : nir_f2fN(&b, offset, own->bit_size);
      moved = nir_fadd(&b, own, nir_fmul(&b, nir_ddx_fine(&b, own), nir_channel(&b, off, 0)));
      moved = nir_fadd(&b, moved, nir_fmul(&b, nir_ddy_fine(&b, own), nir_channel(&b, off, 1)));
      _mesa_hash_table_insert(s.own, moved, own);
      if (own == &load->def) {
         nir_def_rewrite_uses_after(&load->def, moved);
      } else {
         _mesa_hash_table_insert(read, deref->var, moved);
         nir_def_replace(&load->def, moved);
      }
   }
   _mesa_hash_table_destroy(read, NULL);

   /* The position the fragment's. */
   util_dynarray_foreach(&positions, nir_intrinsic_instr *, at) {
      nir_intrinsic_instr *load = *at;
      b.cursor = nir_before_instr(&load->instr);
      nir_def_replace(&load->def, nir_trim_vector(&b, position, load->def.num_components));
   }

   /* Derivatives across fragments. */
   util_dynarray_foreach(&derivatives, nir_intrinsic_instr *, at) {
      nir_intrinsic_instr *intr = *at;
      bool y;
      is_derivative(intr->intrinsic, &y);
      b.cursor = nir_before_instr(&intr->instr);
      struct hash_table *map = _mesa_hash_table_clone(s.own, NULL);
      nir_def *v = own_value(&b, map, intr->src[0].ssa);
      _mesa_hash_table_destroy(map, NULL);
      nir_def *d = y ? nir_ddy_fine(&b, v) : nir_ddx_fine(&b, v);
      nir_def *scale = nir_channel(&b, s.sizef, y);
      nir_def_replace(&intr->def, nir_fmul(&b, d, d->bit_size == 32 ? scale : nir_f2fN(&b, scale, d->bit_size)));
   }

   /* Implicit LODs, of those. */
   util_dynarray_foreach(&samples, nir_tex_instr *, at) {
      nir_tex_instr *tex = *at;
      const int coord_at = nir_tex_instr_src_index(tex, nir_tex_src_coord);
      const unsigned spatial = tex->coord_components - (tex->is_array ? 1 : 0);
      b.cursor = nir_before_instr(&tex->instr);
      struct hash_table *map = _mesa_hash_table_clone(s.own, NULL);
      nir_def *c = nir_trim_vector(&b, own_value(&b, map, tex->src[coord_at].src.ssa), spatial);
      _mesa_hash_table_destroy(map, NULL);
      nir_def *dx = nir_fmul(&b, nir_ddx_fine(&b, c), nir_channel(&b, s.sizef, 0));
      nir_def *dy = nir_fmul(&b, nir_ddy_fine(&b, c), nir_channel(&b, s.sizef, 1));
      const int bias_at = nir_tex_instr_src_index(tex, nir_tex_src_bias);
      if (bias_at >= 0) {
         nir_def *scale = nir_fexp2(&b, tex->src[bias_at].src.ssa);
         dx = nir_fmul(&b, dx, scale);
         dy = nir_fmul(&b, dy, scale);
         nir_tex_instr_remove_src(tex, bias_at);
      }
      tex->op = nir_texop_txd;
      nir_tex_instr_add_src(tex, nir_tex_src_ddx, dx);
      nir_tex_instr_add_src(tex, nir_tex_src_ddy, dy);
   }

   /* Side effects the leader's, and an atomic's result of it. */
   util_dynarray_foreach(&effects, nir_intrinsic_instr *, at) {
      nir_intrinsic_instr *intr = *at;
      const bool has_dest = nir_intrinsic_infos[intr->intrinsic].has_dest;
      b.cursor = nir_before_instr(&intr->instr);
      nir_def *none = has_dest ? nir_imm_zero(&b, intr->def.num_components, intr->def.bit_size) : NULL;
      nir_push_if(&b, s.leader);
      nir_instr_remove(&intr->instr);
      nir_builder_instr_insert(&b, &intr->instr);
      nir_pop_if(&b, NULL);
      if (has_dest) {
         nir_def *result = nir_if_phi(&b, &intr->def, none);
         nir_def *value = from_leader(&b, &s, result);
         nir_foreach_use_including_if_safe(use, &intr->def) {
            if (nir_src_is_if(use) || nir_src_use_instr(use) != nir_def_instr(result))
               nir_src_rewrite(use, value);
         }
      }
   }

   /* gl_ShadingRateEXT. */
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_load_frag_shading_rate)
            continue;
         b.cursor = nir_before_instr(instr);
         nir_def_replace(&nir_instr_as_intrinsic(instr)->def, rate);
      }
   }

   _mesa_hash_table_destroy(s.own, NULL);
   util_dynarray_fini(&derivatives);
   util_dynarray_fini(&samples);
   util_dynarray_fini(&effects);
   util_dynarray_fini(&positions);
   util_dynarray_fini(&inputs);
   return nir_progress(true, impl, nir_metadata_none);
}
