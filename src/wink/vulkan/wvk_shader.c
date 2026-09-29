/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Shaders: the runtime's vk_shader, compiled SPIR-V -> NIR -> WGSL with
 * the compiler the GL driver uses (src/wink/compiler). What is Vulkan's
 * here is getting vtn's NIR into the shape that compiler takes:
 *
 *   - every descriptor at its WebGPU binding: set N is @group(N), and each
 *     element of a Vulkan binding a binding of its own (see
 *     wvk_descriptor_set_layout.h)
 *   - combined image samplers taken apart into a texture and a sampler,
 *     which is all WGSL has
 *   - push constants read from a storage buffer at the driver's binding
 *
 * and saying what the shader uses, which is what the bind group layouts
 * of a pipeline are made of: a Vulkan set layout does not say whether a
 * texture is 2D or a cube, float or integer, and WebGPU insists.
 *
 * Of a pipeline's state, only vertex attributes WebGPU cannot step are
 * compiled in (pull_divided_attributes). WebGPU pipelines are made at the
 * draw, from the modules here and whatever the state is then.
 */
#include "wvk_shader.h"

#include "wvk_descriptor_set_layout.h"
#include "wvk_device.h"
#include "wvk_format.h"
#include "wvk_nir.h"
#include "wvk_physical_device.h"
#include "wvk_query.h"
#include "wvk_sampler.h"

#include "gex_wgpu_nir.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_deref.h"
#include "compiler/nir/nir_serialize.h"
#include "compiler/nir/nir_xfb_info.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/blob.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/u_atomic.h"
#include "util/mesa-blake3.h"
#include "vk_format.h"
#include "vk_nir.h"
#include "vk_nir_convert_ycbcr.h"
#include "vk_ycbcr_conversion.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"

#include <stdlib.h>

#ifdef WVK_HAVE_FLOAT64
#include "wvk_float64_spv.h"
#endif


void
wvk_init_nir_options(nir_shader_compiler_options *options)
{
   *options = *gex_wgpu_nir_options();
   /* InstanceIndex, the builtin, is instance_index, the builtin: both count
    * from firstInstance. GL's gl_InstanceID does not, which is why the GL
    * driver leaves this off. */
   options->instance_id_includes_base_index = true;
   /* Half floats in and out of words, for texel buffers and attributes:
    * the SPIR-V emitter has them, and lowered they would be 16-bit
    * integers, which WGSL does not have. */
   options->lower_pack_half_2x16 = false;
   options->lower_unpack_half_2x16 = false;
   /* Saturating arithmetic, which integer dot products have: SPIR-V's is
    * Intel's alone. */
   options->lower_uadd_sat = true;
   /* OpUMulExtended and friends, which the emitter has not. */
   options->lower_mul_high = true;
   options->lower_usub_sat = true;
   options->lower_iadd_sat = true;
}

static const nir_shader_compiler_options *
wvk_get_nir_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                    const struct vk_pipeline_robustness_state *rs)
{
   struct wvk_physical_device *pdev = container_of(vk_pdev, struct wvk_physical_device, vk);
   return &pdev->nir_options;
}

static struct spirv_to_nir_options
wvk_get_spirv_options(struct vk_physical_device *vk_pdev, mesa_shader_stage stage,
                      const struct vk_pipeline_robustness_state *rs)
{
   /* Buffers through vulkan_resource_index, which zink's emitter turns
    * back into the variables; push constants as a variable, which becomes
    * a storage buffer. */
   return (struct spirv_to_nir_options){
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_logical,
      .shared_addr_format = nir_address_format_32bit_offset,
      .constant_addr_format = nir_address_format_32bit_index_offset,
   };
}

static void
optimize(nir_shader *nir)
{
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_deref);
      NIR_PASS(progress, nir, nir_opt_copy_prop_vars);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_if, 0);
      NIR_PASS(progress, nir, nir_opt_undef);
   } while (progress);
}

static nir_def *driver_buffer(nir_builder *b, uint32_t binding);

/* vkCmdDispatchBase's base workgroup, from the driver's data: WebGPU's
 * workgroup ids start at zero. */
static bool
lower_base_workgroup_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_workgroup_id &&
       intr->intrinsic != nir_intrinsic_load_global_invocation_id &&
       intr->intrinsic != nir_intrinsic_load_base_workgroup_id)
      return false;
   if (intr->def.bit_size != 32 || intr->def.num_components != 3)
      return false;
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *base = nir_load_ssbo(b, 3, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH), nir_imm_int(b, WVK_DRIVER_BASE_GROUP),
                                 .align_mul = 16, .access = ACCESS_NON_WRITEABLE);
   if (intr->intrinsic == nir_intrinsic_load_base_workgroup_id) {
      nir_def_replace(&intr->def, base);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_global_invocation_id) {
      const uint16_t *size = b->shader->info.workgroup_size;
      base = nir_imul(b, base, nir_imm_ivec3(b, size[0], size[1], size[2]));
   }
   nir_def *sum = nir_iadd(b, &intr->def, base);
   nir_def_rewrite_uses_after(&intr->def, sum);
   return true;
}

/* gl_DeviceIndex, of the one device there is, which NIR has no intrinsic
 * for. */
static bool
lower_device_index_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_system_value || var->data.location != SYSTEM_VALUE_DEVICE_INDEX)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_def_replace(&intr->def, nir_imm_intN_t(b, 0, intr->def.bit_size));
   /* And its deref, of a variable that goes. */
   nir_deref_instr_remove_if_unused(deref);
   return true;
}

/* OpQuantizeToF16, through WGSL's pack2x16float, which leaves what
 * f16 cannot hold undefined: those, and denormals, which flush, by hand. */
static bool
lower_quantize_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->op != nir_op_fquantize2f16)
      return false;
   b->cursor = nir_before_instr(instr);
   nir_def *a = nir_ssa_for_alu_src(b, alu, 0);
   nir_def *comps[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < a->num_components; ++i) {
      nir_def *x = nir_channel(b, a, i);
      nir_def *mag = nir_fabs(b, x);
      nir_def *sign = nir_iand_imm(b, x, 0x80000000u);
      nir_def *half = nir_channel(b, nir_unpack_half_2x16(b, nir_pack_half_2x16(b, nir_vec2(b, x, nir_imm_float(b, 0)))), 0);
      /* What rounds past the largest half, 65504, is infinite. */
      nir_def *inf = nir_ior_imm(b, sign, 0x7f800000u);
      nir_def *v = nir_bcsel(b, nir_fge_imm(b, mag, 65520.0f), inf, half);
      v = nir_bcsel(b, nir_flt_imm(b, mag, 6.103515625e-05f /* 2^-14 */), sign, v);
      comps[i] = nir_bcsel(b, nir_fneu(b, x, x), x, v);
   }
   nir_def_replace(&alu->def, nir_vec(b, comps, a->num_components));
   return true;
}

/* Helper invocations, which WGSL has no builtin for: those outside the
 * primitive cover no samples, and the shader knows which it demoted. */
/* A sampling at an explicit LOD at an offset not constant: the offset,
 * in texels of the level the LOD is of, moved into the coordinates --
 * which nir_lower_tex would scale by level 0's. */
static bool
lower_lod_offset_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int offset_src = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   const int lod_src = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int coord_src = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (tex->op != nir_texop_txl || offset_src < 0 || lod_src < 0 || coord_src < 0 ||
       nir_src_is_const(tex->src[offset_src].src) || tex->sampler_dim == GLSL_SAMPLER_DIM_RECT ||
       tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)
      return false;
   b->cursor = nir_before_instr(instr);
   const unsigned dims = tex->coord_components - tex->is_array;
   nir_def *level = nir_imax(b, nir_f2i32(b, nir_ffloor(b, tex->src[lod_src].src.ssa)), nir_imm_int(b, 0));
   nir_def *size = nir_build_texture_query(b, tex, nir_texop_txs, tex->coord_components, nir_type_int32, false, false);
   nir_tex_instr_add_src(nir_instr_as_tex(nir_def_instr(size)), nir_tex_src_lod, level);
   nir_def *offset = nir_i2f32(b, nir_trim_vector(b, tex->src[offset_src].src.ssa, dims));
   nir_def *moved = nir_fmul(b, offset, nir_frcp(b, nir_i2f32(b, nir_trim_vector(b, size, dims))));
   nir_def *coord = tex->src[coord_src].src.ssa;
   nir_def *c[4];
   for (unsigned i = 0; i < coord->num_components; ++i)
      c[i] = i < dims ? nir_fadd(b, nir_channel(b, coord, i), nir_channel(b, moved, i)) : nir_channel(b, coord, i);
   nir_src_rewrite(&tex->src[coord_src].src, nir_vec(b, c, coord->num_components));
   nir_tex_instr_remove_src(tex, offset_src);
   return true;
}

/* An offset that is not a constant, which WGSL's are: a gather's, and
 * since VK_KHR_maintenance8 any sampling's or fetch's. */
static bool
dynamic_offset(const nir_instr *instr, const void *data)
{
   const nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int i = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   return i >= 0 && !nir_src_is_const(tex->src[i].src);
}

static bool
lower_helper_invocations(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   bool uses = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             (nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_load_helper_invocation ||
              nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_is_helper_invocation))
            uses = true;
      }
   }
   if (!uses)
      return false;

   nir_variable *demoted = nir_local_variable_create(impl, glsl_bool_type(), "demoted");
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_store_var(&b, demoted, nir_imm_false(&b), 0x1);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_demote:
            nir_store_var(&b, demoted, nir_imm_true(&b), 0x1);
            break;
         case nir_intrinsic_demote_if:
            nir_store_var(&b, demoted, nir_ior(&b, nir_load_var(&b, demoted), intr->src[0].ssa), 0x1);
            break;
         case nir_intrinsic_load_helper_invocation:
         case nir_intrinsic_is_helper_invocation: {
            nir_def *uncovered = nir_ieq_imm(&b, nir_load_sample_mask_in(&b), 0);
            nir_def_replace(&intr->def, nir_ior(&b, uncovered, nir_load_var(&b, demoted)));
            break;
         }
         default:
            break;
         }
      }
   }
   return nir_progress(true, impl, nir_metadata_none);
}

/* 64-bit floats, which WGSL has none of: Mesa's library of them in
 * integer arithmetic (float64.glsl), made once. Its 64-bit integers are
 * pairs of words later, as the shader's are. */
static const nir_shader *
float64_library(struct wvk_physical_device *pdev)
{
#ifdef WVK_HAVE_FLOAT64
   simple_mtx_lock(&pdev->float64_lock);
   if (!pdev->float64) {
      const struct spirv_capabilities caps = {
         .Addresses = true, .Float64 = true, .Int8 = true, .Int16 = true, .Int64 = true, .Shader = true,
      };
      const struct spirv_to_nir_options options = {
         .capabilities = &caps,
         .environment = NIR_SPIRV_VULKAN,
         .create_library = true,
      };
      nir_shader *nir = spirv_to_nir(wvk_float64_spv_source, sizeof(wvk_float64_spv_source) / 4, NULL,
                                     MESA_SHADER_VERTEX, "main", &options, &pdev->nir_options);
      if (nir) {
         NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
         NIR_PASS(_, nir, nir_lower_returns);
         NIR_PASS(_, nir, nir_inline_functions);
         nir_sweep(nir);
      }
      pdev->float64 = nir;
   }
   simple_mtx_unlock(&pdev->float64_lock);
   return pdev->float64;
#else
   (void)pdev;
   return NULL;
#endif
}

/* Operations of 64-bit floats, a component at a time, which the library's
 * functions are of. */
static bool
is_float64_alu(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   const nir_op_info *info = &nir_op_infos[alu->op];
   if (alu->def.bit_size == 64 && nir_alu_type_get_base_type(info->output_type) == nir_type_float)
      return true;
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      if (alu->src[i].src.ssa->bit_size == 64 &&
          nir_alu_type_get_base_type(info->input_types[i]) == nir_type_float)
         return true;
   }
   return false;
}

/* The library's parameters are 64-bit integers, which the calls' double
 * variables are read as through casts, and vars_to_ssa leaves a variable
 * that is cast alone: the variables of the integers they are read as. */
static bool
retype_cast_params(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *cast = nir_instr_as_deref(instr);
            if (cast->deref_type != nir_deref_type_cast || !nir_deref_mode_is(cast, nir_var_function_temp))
               continue;
            nir_deref_instr *parent = nir_deref_instr_parent(cast);
            if (!parent || parent->deref_type != nir_deref_type_var ||
                glsl_get_bit_size(parent->type) != glsl_get_bit_size(cast->type) ||
                glsl_get_components(parent->type) != glsl_get_components(cast->type) ||
                !glsl_type_is_vector_or_scalar(cast->type))
               continue;
            parent->var->type = cast->type;
            progress = true;
         }
      }
   }
   if (!progress)
      return false;
   nir_fixup_deref_types(nir);
   /* The casts are of the type they cast to now. */
   NIR_PASS(_, nir, nir_opt_deref);
   return true;
}

/* Conversions between half and double floats, by way of floats: the
 * library's are of floats and doubles alone. */
static bool
half_double_conversion_instr(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if (!nir_op_infos[alu->op].is_conversion ||
       nir_alu_type_get_base_type(nir_op_infos[alu->op].input_types[0]) != nir_type_float ||
       nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type) != nir_type_float)
      return false;
   const unsigned from = alu->src[0].src.ssa->bit_size, to = alu->def.bit_size;
   if (!((from == 16 && to == 64) || (from == 64 && to == 16)))
      return false;
   b->cursor = nir_before_instr(&alu->instr);
   nir_def *x = nir_mov_alu(b, alu->src[0], alu->def.num_components);
   if (alu->op == nir_op_f2f16_rtz) {
      /* Toward zero, which by way of a float would round twice: the
       * double's bits truncated. */
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned i = 0; i < x->num_components; ++i) {
         nir_def *hi = nir_unpack_64_2x32_split_y(b, nir_channel(b, x, i));
         nir_def *lo = nir_unpack_64_2x32_split_x(b, nir_channel(b, x, i));
         nir_def *sign = nir_iand_imm(b, nir_ushr_imm(b, hi, 16), 0x8000);
         nir_def *exp = nir_iand_imm(b, nir_ushr_imm(b, hi, 20), 0x7ff);
         nir_def *mant = nir_iand_imm(b, hi, 0xfffff);
         nir_def *e = nir_iadd_imm(b, exp, 15 - 1023);
         nir_def *normal = nir_ior(b, nir_ishl_imm(b, e, 10), nir_ushr_imm(b, mant, 10));
         /* Below the least normal: the mantissa, its 1 with it, shifted. */
         nir_def *shift = nir_isub_imm(b, 1, e);
         nir_def *sub = nir_bcsel(b, nir_ilt_imm(b, shift, 12),
                                  nir_ushr(b, nir_ior_imm(b, nir_ushr_imm(b, mant, 10), 0x400), shift), nir_imm_int(b, 0));
         nir_def *nan = nir_ine_imm(b, nir_ior(b, mant, lo), 0);
         nir_def *h = nir_bcsel(b, nir_ieq_imm(b, exp, 0x7ff), nir_bcsel(b, nan, nir_imm_int(b, 0x7e00), nir_imm_int(b, 0x7c00)),
                      nir_bcsel(b, nir_ige_imm(b, e, 31), nir_imm_int(b, 0x7bff),
                      nir_bcsel(b, nir_ige_imm(b, e, 1), normal, sub)));
         comps[i] = nir_u2u16(b, nir_ior(b, sign, h));
      }
      nir_def_replace(&alu->def, nir_vec(b, comps, x->num_components));
      return true;
   }
   nir_def *f = nir_f2f32(b, x);
   nir_def_replace(&alu->def, to == 64 ? nir_f2f64(b, f) : nir_f2f16(b, f));
   return true;
}

static void
lower_float64(struct wvk_physical_device *pdev, nir_shader *nir)
{
   /* Variables of them the shader no longer uses, which the emitter would
    * still declare. */
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp | nir_var_shader_temp, NULL);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   if (!(nir->info.bit_sizes_float & 64))
      return;
   const nir_shader *library = float64_library(pdev);
   if (!library) {
      mesa_loge("wvk: a shader of 64-bit floats, without Mesa's library of them");
      return;
   }
   /* Variables of them indexed dynamically, which vars_to_ssa leaves:
    * values, chosen among. */
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_function_temp, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_shader_alu_pass, half_double_conversion_instr, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_lower_flrp, 64, false);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, is_float64_alu, NULL);
   /* Every operation: those the library has no function of, in terms of
    * those it has. */
   const nir_lower_doubles_options all = (nir_lower_doubles_options)(
      nir_lower_drcp | nir_lower_dsqrt | nir_lower_drsq | nir_lower_dtrunc | nir_lower_dfloor | nir_lower_dceil |
      nir_lower_dfract | nir_lower_dround_even | nir_lower_dmod | nir_lower_dsub | nir_lower_ddiv |
      nir_lower_dsign | nir_lower_dminmax | nir_lower_dsat | nir_lower_fp64_full_software);
   NIR_PASS(_, nir, nir_lower_doubles, library, all);
   bool conversions = false;
   NIR_PASS(conversions, nir, nir_lower_int64_float_conversions);
   if (conversions) {
      NIR_PASS(_, nir, nir_opt_algebraic);
      NIR_PASS(_, nir, nir_lower_doubles, library, all);
   }
   /* The calls' parameters: values, and their variables gone, which the
    * emitter would declare of types WGSL has not. */
   NIR_PASS(_, nir, retype_cast_params);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp | nir_var_shader_temp, NULL);
}

static void
wvk_preprocess_nir(struct vk_physical_device *vk_pdev, nir_shader *nir,
                   const struct vk_pipeline_robustness_state *rs)
{
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* WGSL's discard is a demote: a terminate is one, and a return. */
      NIR_PASS(_, nir, nir_lower_terminate_to_demote);
      NIR_PASS(_, nir, nir_lower_halt_to_return);
      NIR_PASS(_, nir, nir_lower_returns);
   }
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~nir_var_function_temp);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   lower_float64(container_of(vk_pdev, struct wvk_physical_device, vk), nir);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_device_index_instr, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, wvk_nir_lower_subgroups);
   /* ALU the SPIR-V emitter has no instruction for. */
   NIR_PASS(_, nir, nir_lower_frexp);
   NIR_PASS(_, nir, nir_shader_instructions_pass, lower_quantize_instr, nir_metadata_control_flow, NULL);
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, lower_helper_invocations);
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   }
   /* Input attachments are textures, fetched at the fragment: the
    * runtime's subpasses are passes of their own. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      const nir_input_attachment_options ia = { 0 };
      NIR_PASS(_, nir, nir_lower_input_attachments, &ia);
   }
   optimize(nir);
}

/* The vertex attributes the shader reads itself: those of instance-rate
 * bindings with a divisor other than 1, since WebGPU steps an instance
 * attribute every instance; and those WebGPU cannot lay out, of a stride
 * not whole words or at an offset its format is not aligned to. */
static bool
is_pulled(const struct vk_vertex_input_state *vi, int location)
{
   if (!vi || location < 0 || location >= MESA_VK_MAX_VERTEX_ATTRIBUTES ||
       !(vi->attributes_valid & BITFIELD_BIT(location)))
      return false;
   const struct vk_vertex_attribute_state *a = &vi->attributes[location];
   const struct vk_vertex_binding_state *vb = &vi->bindings[a->binding];
   const uint32_t align = MIN2(vk_format_get_blocksize(a->format), 4);
   /* And those WebGPU has no format of, where the shader can read them. */
   if (!wvk_format_vertex_native(a->format))
      return wvk_format_vertex_pullable(a->format);
   return (vb->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE && vb->divisor != 1) ||
          (vb->stride % 4) || (align && a->offset % align);
}

/* Whether a vertex shader reads the attribute at `location` itself: where
 * WebGPU cannot, or for points drawn as quads (`all`, expand_points) every
 * one the shader can. */
static bool
pulls(const struct vk_vertex_input_state *vi, int location, bool all)
{
   if (!all)
      return is_pulled(vi, location);
   return vi && location >= 0 && location < MESA_VK_MAX_VERTEX_ATTRIBUTES &&
          (vi->attributes_valid & BITFIELD_BIT(location)) &&
          wvk_format_vertex_pullable(vi->attributes[location].format);
}

/* The attributes of `read` a vertex shader reads itself, as `vi` has them,
 * hashed: 0 for none (wvk_shader_key.pulled). A stride is only what makes
 * an attribute pulled, the draw giving its value. */
static uint64_t
pull_key(const struct vk_vertex_input_state *vi, uint32_t read, bool all)
{
   if (!vi)
      return 0;
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   _mesa_blake3_update(&ctx, &all, sizeof(all));
   bool any = false;
   /* And those it reads that the draw has none of (vertexAttributeRobustness). */
   const uint32_t missing = read & ~vi->attributes_valid;
   if (missing) {
      _mesa_blake3_update(&ctx, &missing, sizeof(missing));
      any = true;
   }
   u_foreach_bit(location, vi->attributes_valid & read) {
      if (!pulls(vi, location, all))
         continue;
      const struct vk_vertex_attribute_state *a = &vi->attributes[location];
      const struct vk_vertex_binding_state *vb = &vi->bindings[a->binding];
      const uint32_t words[] = { location, a->binding, a->format, a->offset, vb->divisor, vb->input_rate,
                                 vb->stride % 4 };
      _mesa_blake3_update(&ctx, words, sizeof(words));
      any = true;
   }
   if (!any)
      return 0;
   blake3_hash hash;
   _mesa_blake3_final(&ctx, hash);
   uint64_t key;
   memcpy(&key, hash, sizeof(key));
   return key | 1;
}

static bool wants_sample_shading(const struct vk_graphics_pipeline_state *state);

static void
wvk_hash_state(struct vk_physical_device *device, const struct vk_graphics_pipeline_state *state,
               const struct vk_features *enabled_features, VkShaderStageFlags stages,
               blake3_hash blake3_out)
{
   /* Only the attributes a vertex shader reads itself are compiled in. */
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
   const struct vk_vertex_input_state *vi = state ? state->vi : NULL;
   if ((stages & VK_SHADER_STAGE_VERTEX_BIT) && vi) {
      u_foreach_bit(location, vi->attributes_valid) {
         if (!is_pulled(vi, location))
            continue;
         const struct vk_vertex_attribute_state *a = &vi->attributes[location];
         const struct vk_vertex_binding_state *vb = &vi->bindings[a->binding];
         const uint32_t words[] = { location, a->binding, a->format, a->offset, vb->divisor, vb->stride,
                                    vb->input_rate };
         _mesa_blake3_update(&ctx, words, sizeof(words));
      }
   }
   /* And whether fragments are shaded per sample. */
   if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
      const uint8_t per_sample = wants_sample_shading(state);
      _mesa_blake3_update(&ctx, &per_sample, 1);
      /* And alpha to one and to coverage, which it may do itself. */
      const uint8_t alpha[3] = {
         state && state->ms && state->ms->alpha_to_one_enable,
         state && state->ms && state->ms->alpha_to_coverage_enable,
         state && state->ms ? state->ms->rasterization_samples : 1,
      };
      _mesa_blake3_update(&ctx, alpha, sizeof(alpha));
   }
   /* And whether images may be null, which shaders check themselves. */
   const uint8_t null_images = enabled_features->nullDescriptor;
   _mesa_blake3_update(&ctx, &null_images, 1);
   _mesa_blake3_final(&ctx, blake3_out);
}

/* The colour outputs at the locations of the attachments they are for
 * (dynamicRenderingLocalRead's colorAttachmentLocations), WebGPU's targets
 * being the pass's attachments in order: attachment i's is the output at
 * location map[i]. An output no attachment is at is gone. */
static bool
remap_color_outputs(nir_shader *nir, const uint8_t map[8])
{
   NIR_PASS(_, nir, nir_lower_io_array_vars_to_elements_no_indirects, true);
   bool gone = false;
   nir_foreach_shader_out_variable(var, nir) {
      if (var->data.location < FRAG_RESULT_DATA0)
         continue;
      const unsigned location = var->data.location - FRAG_RESULT_DATA0;
      int attachment = -1;
      for (unsigned i = 0; i < 8 && attachment < 0; ++i) {
         if (map[i] == location)
            attachment = i;
      }
      if (attachment < 0) {
         var->data.mode = nir_var_shader_temp;
         gone = true;
      } else {
         var->data.location = FRAG_RESULT_DATA0 + attachment;
      }
   }
   if (gone) {
      nir_fixup_deref_modes(nir);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   }
   return nir_progress(true, nir_shader_get_entrypoint(nir), nir_metadata_all);
}

/* Alpha to one, which WebGPU has not: a fragment shader's colour outputs
 * with an alpha of one. And alpha to coverage where it is on too, or the
 * shader writes its own sample mask, which WebGPU's will not go with:
 * the shader's, the samples covered as its first output's alpha says,
 * that many of them. */
static bool
lower_alpha_to_one(nir_shader *nir, bool to_one, bool to_coverage, unsigned samples)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_variable *mask_var = NULL;
   nir_foreach_shader_out_variable(var, nir) {
      if (var->data.location == FRAG_RESULT_SAMPLE_MASK)
         mask_var = var;
   }
   bool progress = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_deref)
            continue;
         nir_variable *var = nir_intrinsic_get_var(intr, 0);
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         if (!var || var->data.mode != nir_var_shader_out || var->data.location < FRAG_RESULT_DATA0 ||
             !glsl_type_is_vector_or_scalar(deref->type) || glsl_get_base_type(deref->type) != GLSL_TYPE_FLOAT ||
             glsl_get_vector_elements(deref->type) < 4 || !(nir_intrinsic_write_mask(intr) & 0x8))
            continue;
         b.cursor = nir_before_instr(instr);
         nir_def *value = intr->src[1].ssa;
         if (to_coverage && var->data.location == FRAG_RESULT_DATA0 && var->data.index == 0) {
            nir_def *alpha = nir_fsat(&b, nir_channel(&b, value, 3));
            nir_def *n = nir_f2u32(&b, nir_fround_even(&b, nir_fmul_imm(&b, alpha, samples)));
            nir_def *covered = nir_bcsel(&b, nir_uge_imm(&b, n, 32), nir_imm_int(&b, -1),
                                         nir_iadd_imm(&b, nir_ishl(&b, nir_imm_int(&b, 1), n), -1));
            if (!mask_var) {
               /* An array, as the SPIR-V's is: not what
                * nir_create_variable_with_location will make. */
               mask_var = nir_variable_create(nir, nir_var_shader_out, glsl_array_type(glsl_int_type(), 1, 0),
                                              "gl_SampleMask");
               mask_var->data.location = FRAG_RESULT_SAMPLE_MASK;
               mask_var->data.driver_location = nir->num_outputs++;
               /* Of every sample, until the shader says. */
               nir_builder top = nir_builder_at(nir_before_impl(impl));
               nir_store_deref(&top, nir_build_deref_array_imm(&top, nir_build_deref_var(&top, mask_var), 0),
                               nir_imm_int(&top, -1), 0x1);
            }
            nir_deref_instr *m = nir_build_deref_array_imm(&b, nir_build_deref_var(&b, mask_var), 0);
            nir_store_deref(&b, m, nir_iand(&b, nir_load_deref(&b, m), covered), 0x1);
         }
         if (to_one) {
            nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(&b, value, nir_imm_floatN_t(&b, 1.0, value->bit_size), 3));
         }
         progress = true;
      }
   }
   return nir_progress(progress, impl, nir_metadata_control_flow);
}

/* Vulkan's logic ops, which WebGPU has no colour state for.
 *
 * The fragment shader does them. A logic op is defined on an attachment's
 * integer representation -- XOR of 0.5 and 0.25 is not a float operation --
 * so each output is taken to the attachment format's own integers, combined
 * with the destination's, and brought back; float and sRGB attachments are
 * not logic-op'd, and their outputs pass as they are. The destination is a
 * copy of the attachment taken before the draw (wvk_cmd_draw.c), read at
 * the fragment: WebGPU will not have a pass read what it renders to. So
 * primitives of one draw that overlap see the attachment as it was before
 * the draw rather than each other. */
enum logic_kind {
   LOGIC_NONE,
   LOGIC_UNORM,
   LOGIC_SNORM,
   LOGIC_UINT,
   LOGIC_SINT,
};

/* A component's representation in the format: its kind and its bits. */
static enum logic_kind
logic_channel(VkFormat format, unsigned component, unsigned *bits)
{
   const struct util_format_description *d = vk_format_description(format);
   *bits = 0;
   if (!d || d->colorspace != UTIL_FORMAT_COLORSPACE_RGB || d->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       d->swizzle[component] > PIPE_SWIZZLE_W)
      return LOGIC_NONE;
   const struct util_format_channel_description *c = &d->channel[d->swizzle[component]];
   *bits = c->size;
   if (!c->size || c->size > 32)
      return LOGIC_NONE;
   if (c->type == UTIL_FORMAT_TYPE_UNSIGNED)
      return c->normalized ? LOGIC_UNORM : LOGIC_UINT;
   if (c->type == UTIL_FORMAT_TYPE_SIGNED)
      return c->normalized ? LOGIC_SNORM : LOGIC_SINT;
   return LOGIC_NONE;
}

/* Whether a logic op does anything to an attachment of `format`. */
bool
wvk_format_takes_logic_op(VkFormat format)
{
   for (unsigned c = 0; c < 4; ++c) {
      unsigned bits;
      if (logic_channel(format, c, &bits) != LOGIC_NONE)
         return true;
   }
   return false;
}

static nir_def *
logic_apply(nir_builder *b, VkLogicOp op, nir_def *s, nir_def *d)
{
   switch (op) {
   case VK_LOGIC_OP_CLEAR: return nir_imm_int(b, 0);
   case VK_LOGIC_OP_AND: return nir_iand(b, s, d);
   case VK_LOGIC_OP_AND_REVERSE: return nir_iand(b, s, nir_inot(b, d));
   case VK_LOGIC_OP_COPY: return s;
   case VK_LOGIC_OP_AND_INVERTED: return nir_iand(b, nir_inot(b, s), d);
   case VK_LOGIC_OP_NO_OP: return d;
   case VK_LOGIC_OP_XOR: return nir_ixor(b, s, d);
   case VK_LOGIC_OP_OR: return nir_ior(b, s, d);
   case VK_LOGIC_OP_NOR: return nir_inot(b, nir_ior(b, s, d));
   case VK_LOGIC_OP_EQUIVALENT: return nir_inot(b, nir_ixor(b, s, d));
   case VK_LOGIC_OP_INVERT: return nir_inot(b, d);
   case VK_LOGIC_OP_OR_REVERSE: return nir_ior(b, s, nir_inot(b, d));
   case VK_LOGIC_OP_COPY_INVERTED: return nir_inot(b, s);
   case VK_LOGIC_OP_OR_INVERTED: return nir_ior(b, nir_inot(b, s), d);
   case VK_LOGIC_OP_NAND: return nir_inot(b, nir_iand(b, s, d));
   case VK_LOGIC_OP_SET: default: return nir_imm_int(b, -1);
   }
}

/* One component: the source and the destination as the format's integers,
 * combined, and the result as the output's value again. */
static nir_def *
logic_component(nir_builder *b, VkLogicOp op, enum logic_kind kind, unsigned bits, nir_def *src,
                nir_def *dst)
{
   const uint32_t mask = bits >= 32 ? ~0u : (1u << bits) - 1u;
   nir_def *s, *d;
   switch (kind) {
   case LOGIC_UNORM:
      s = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fsat(b, src), mask)));
      d = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fsat(b, dst), mask)));
      break;
   case LOGIC_SNORM: {
      const double max = (double)((1u << (bits - 1)) - 1u);
      s = nir_f2i32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fmax(b, nir_fmin(b, src, nir_imm_float(b, 1.0f)),
                                                                       nir_imm_float(b, -1.0f)), max)));
      d = nir_f2i32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fmax(b, nir_fmin(b, dst, nir_imm_float(b, 1.0f)),
                                                                       nir_imm_float(b, -1.0f)), max)));
      break;
   }
   default:
      s = src;
      d = dst;
      break;
   }
   nir_def *r = nir_iand_imm(b, logic_apply(b, op, nir_iand_imm(b, s, mask), nir_iand_imm(b, d, mask)), mask);
   /* Signed kinds take their sign from the top bit they have. */
   if ((kind == LOGIC_SNORM || kind == LOGIC_SINT) && bits < 32)
      r = nir_ishr_imm(b, nir_ishl_imm(b, r, 32 - bits), 32 - bits);
   switch (kind) {
   case LOGIC_UNORM:
      return nir_fmul_imm(b, nir_u2f32(b, r), 1.0 / (double)mask);
   case LOGIC_SNORM:
      return nir_fmax(b, nir_fmul_imm(b, nir_i2f32(b, r), 1.0 / (double)((1u << (bits - 1)) - 1u)),
                      nir_imm_float(b, -1.0f));
   default:
      return r;
   }
}

struct logic_op_state {
   const struct wvk_shader_key *key;
   nir_variable *dst[8];
   uint8_t reads;
};

static bool
lower_logic_op_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct logic_op_state *state = data;
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_out || var->data.location < FRAG_RESULT_DATA0 ||
       var->data.index != 0 || !glsl_type_is_vector_or_scalar(deref->type))
      return false;
   unsigned rt = var->data.location - FRAG_RESULT_DATA0;
   if (deref->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(deref->arr.index))
         return false;
      rt += nir_src_as_uint(deref->arr.index);
   } else if (deref->deref_type != nir_deref_type_var) {
      return false;
   }
   if (rt >= 8 || !state->key->logic_format[rt])
      return false;
   const VkFormat format = (VkFormat)state->key->logic_format[rt];
   const VkLogicOp op = (VkLogicOp)(state->key->logic_op - 1);

   /* The copy of the attachment, as the format samples. */
   if (!state->dst[rt]) {
      const enum glsl_base_type base = vk_format_is_uint(format)   ? GLSL_TYPE_UINT
                                       : vk_format_is_sint(format) ? GLSL_TYPE_INT
                                                                   : GLSL_TYPE_FLOAT;
      nir_variable *tex = nir_variable_create(b->shader, nir_var_uniform,
                                              glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, base), "wvk_logic_op_dst");
      tex->data.descriptor_set = WVK_DRIVER_GROUP;
      tex->data.binding = WVK_DRIVER_SLOT_LOGIC_OP + rt;
      state->dst[rt] = tex;
   }
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *coord = nir_f2i32(b, nir_trim_vector(b, nir_load_frag_coord(b), 2));
   nir_def *dst = nir_txf(b, coord, .texture_deref = nir_build_deref_var(b, state->dst[rt]), .lod = nir_imm_int(b, 0),
                          .dest_type = vk_format_is_uint(format)   ? nir_type_uint32
                                       : vk_format_is_sint(format) ? nir_type_int32
                                                                   : nir_type_float32);
   nir_def *src = intr->src[1].ssa;
   nir_def *out[4];
   for (unsigned c = 0; c < src->num_components; ++c) {
      unsigned bits;
      const enum logic_kind kind = logic_channel(format, c, &bits);
      out[c] = kind == LOGIC_NONE ? nir_channel(b, src, c)
                                  : logic_component(b, op, kind, bits, nir_channel(b, src, c), nir_channel(b, dst, c));
   }
   nir_src_rewrite(&intr->src[1], nir_vec(b, out, src->num_components));
   state->reads |= BITFIELD_BIT(rt);
   return true;
}

static bool
lower_logic_op(nir_shader *nir, const struct wvk_shader_key *key, uint8_t *reads)
{
   struct logic_op_state state = { .key = key };
   const bool progress = nir_shader_intrinsics_pass(nir, lower_logic_op_instr, nir_metadata_control_flow, &state);
   *reads = state.reads;
   return progress;
}

/* Blending WebGPU cannot do, done by the fragment shader: where colour
 * factors use both CONSTANT_COLOR and CONSTANT_ALPHA, which WebGPU's one
 * blend constant cannot both be (wvk_shader_blend_mask). What the shader
 * writes to such an attachment goes to a local, and at the end it is
 * blended with what the attachment holds -- a copy of it, as a logic op
 * reads -- by the attachment's equation, with the blend constants of the
 * driver's data, and written. Its second output, a dual-source blend's,
 * is then only a local. */
static nir_def *
blend_factor_value(nir_builder *b, VkBlendFactor f, unsigned c, nir_def *s, nir_def *s1, nir_def *d, nir_def *k)
{
   nir_def *one = nir_imm_float(b, 1.0f);
   switch (f) {
   case VK_BLEND_FACTOR_ZERO: return nir_imm_float(b, 0.0f);
   case VK_BLEND_FACTOR_SRC_COLOR: return nir_channel(b, s, c);
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR: return nir_fsub(b, one, nir_channel(b, s, c));
   case VK_BLEND_FACTOR_DST_COLOR: return nir_channel(b, d, c);
   case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR: return nir_fsub(b, one, nir_channel(b, d, c));
   case VK_BLEND_FACTOR_SRC_ALPHA: return nir_channel(b, s, 3);
   case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA: return nir_fsub(b, one, nir_channel(b, s, 3));
   case VK_BLEND_FACTOR_DST_ALPHA: return nir_channel(b, d, 3);
   case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA: return nir_fsub(b, one, nir_channel(b, d, 3));
   case VK_BLEND_FACTOR_CONSTANT_COLOR: return nir_channel(b, k, c);
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR: return nir_fsub(b, one, nir_channel(b, k, c));
   case VK_BLEND_FACTOR_CONSTANT_ALPHA: return nir_channel(b, k, 3);
   case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return nir_fsub(b, one, nir_channel(b, k, 3));
   case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE:
      return c == 3 ? one : nir_fmin(b, nir_channel(b, s, 3), nir_fsub(b, one, nir_channel(b, d, 3)));
   case VK_BLEND_FACTOR_SRC1_COLOR: return nir_channel(b, s1, c);
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR: return nir_fsub(b, one, nir_channel(b, s1, c));
   case VK_BLEND_FACTOR_SRC1_ALPHA: return nir_channel(b, s1, 3);
   case VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA: return nir_fsub(b, one, nir_channel(b, s1, 3));
   case VK_BLEND_FACTOR_ONE:
   default: return one;
   }
}

/* Each component clamped as a normalized attachment's are, before
 * blending: its values and its factors. */
static nir_def *
blend_clamp(nir_builder *b, nir_def *v, const struct util_format_description *d)
{
   if (d && d->is_unorm)
      return nir_fsat(b, v);
   if (d && d->is_snorm)
      return nir_fmax(b, nir_fmin(b, v, nir_imm_float(b, 1.0f)), nir_imm_float(b, -1.0f));
   return v;
}

static nir_def *
blend_apply(nir_builder *b, uint32_t eq, VkFormat format, nir_def *s, nir_def *s1, nir_def *d, nir_def *k)
{
   const struct util_format_description *desc = vk_format_description(format);
   s = blend_clamp(b, s, desc);
   s1 = blend_clamp(b, s1, desc);
   d = blend_clamp(b, d, desc);
   k = blend_clamp(b, k, desc);
   nir_def *out[4];
   for (unsigned c = 0; c < 4; ++c) {
      const uint32_t e = c == 3 ? eq >> 13 : eq;
      const VkBlendOp op = (VkBlendOp)((e >> 10) & 7);
      nir_def *sc = nir_channel(b, s, c), *dc = nir_channel(b, d, c);
      if (op == VK_BLEND_OP_MIN) {
         out[c] = nir_fmin(b, sc, dc);
         continue;
      }
      if (op == VK_BLEND_OP_MAX) {
         out[c] = nir_fmax(b, sc, dc);
         continue;
      }
      nir_def *sf = blend_clamp(b, blend_factor_value(b, (VkBlendFactor)(e & 31), c, s, s1, d, k), desc);
      nir_def *df = blend_clamp(b, blend_factor_value(b, (VkBlendFactor)((e >> 5) & 31), c, s, s1, d, k), desc);
      nir_def *st = nir_fmul(b, sc, sf), *dt = nir_fmul(b, dc, df);
      out[c] = op == VK_BLEND_OP_SUBTRACT           ? nir_fsub(b, st, dt)
               : op == VK_BLEND_OP_REVERSE_SUBTRACT ? nir_fsub(b, dt, st)
                                                    : nir_fadd(b, st, dt);
   }
   return nir_vec(b, out, 4);
}

/* The attachment an output's store is to, and the component it starts
 * at, or -1. */
static int
blend_store_target(nir_intrinsic_instr *intr, unsigned *index)
{
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return -1;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_out || var->data.location < FRAG_RESULT_DATA0 ||
       var->data.location >= FRAG_RESULT_DATA0 + 8 || !glsl_type_is_vector_or_scalar(deref->type) ||
       !glsl_type_is_float_16_32(deref->type))
      return -1;
   int rt = var->data.location - FRAG_RESULT_DATA0;
   if (deref->deref_type == nir_deref_type_array) {
      if (!nir_src_is_const(deref->arr.index))
         return -1;
      rt += nir_src_as_uint(deref->arr.index);
   } else if (deref->deref_type != nir_deref_type_var) {
      return -1;
   }
   *index = var->data.index;
   return rt < 8 ? rt : -1;
}

static bool
lower_blend(nir_shader *nir, const struct wvk_shader_key *key, uint8_t *reads)
{
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_out, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_variable *src[8] = { 0 }, *src1 = NULL;
   u_foreach_bit(rt, key->blend_mask)
      src[rt] = nir_local_variable_create(impl, glsl_vec4_type(), "wvk_blend_src");
   /* Every store to them, to the locals instead. */
   bool dual = false;
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         unsigned index;
         const int rt = blend_store_target(intr, &index);
         if (rt < 0 || !(key->blend_mask & BITFIELD_BIT(rt)) || (index && rt))
            continue;
         nir_variable *var = nir_intrinsic_get_var(intr, 0);
         if (index && !src1)
            src1 = nir_local_variable_create(impl, glsl_vec4_type(), "wvk_blend_src1");
         dual |= index != 0;
         b.cursor = nir_before_instr(instr);
         const unsigned frac = var->data.location_frac;
         nir_def *v = nir_f2f32(&b, intr->src[1].ssa);
         nir_def *comps[4];
         for (unsigned c = 0; c < 4; ++c)
            comps[c] = c >= frac && c - frac < v->num_components ? nir_channel(&b, v, c - frac)
                                                                  : nir_undef(&b, 1, 32);
         nir_store_var(&b, index ? src1 : src[rt], nir_vec(&b, comps, 4),
                       (nir_intrinsic_write_mask(intr) << frac) & 0xf);
         nir_instr_remove(instr);
      }
   }
   /* At the end, blended and written. */
   b.cursor = nir_after_impl(impl);
   nir_def *coord = nir_f2i32(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2));
   nir_def *k = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH), nir_imm_int(&b, WVK_DRIVER_BLEND),
                              .align_mul = 16, .access = ACCESS_NON_WRITEABLE);
   u_foreach_bit(rt, key->blend_mask) {
      nir_variable *tex = nir_variable_create(nir, nir_var_uniform,
                                              glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
                                              "wvk_blend_dst");
      tex->data.descriptor_set = WVK_DRIVER_GROUP;
      tex->data.binding = WVK_DRIVER_SLOT_LOGIC_OP + rt;
      nir_def *d = nir_txf(&b, coord, .texture_deref = nir_build_deref_var(&b, tex), .lod = nir_imm_int(&b, 0),
                           .dest_type = nir_type_float32);
      nir_def *s1 = rt == 0 && src1 ? nir_load_var(&b, src1) : nir_imm_vec4(&b, 0, 0, 0, 0);
      nir_def *r = blend_apply(&b, key->blend_equation[rt], (VkFormat)key->logic_format[rt],
                               nir_load_var(&b, src[rt]), s1, d, k);
      nir_foreach_shader_out_variable(var, nir) {
         if (var->data.index != 0 || var->data.location < FRAG_RESULT_DATA0)
            continue;
         const unsigned loc = var->data.location - FRAG_RESULT_DATA0;
         const struct glsl_type *t = var->type;
         nir_deref_instr *deref = nir_build_deref_var(&b, var);
         if (glsl_type_is_array(t)) {
            if (rt < loc || rt >= loc + glsl_get_length(t))
               continue;
            deref = nir_build_deref_array_imm(&b, deref, rt - loc);
            t = glsl_get_array_element(t);
         } else if (rt != loc) {
            continue;
         }
         if (!glsl_type_is_vector_or_scalar(t) || !glsl_type_is_float_16_32(t))
            continue;
         const unsigned n = glsl_get_vector_elements(t), frac = var->data.location_frac;
         nir_def *v = nir_channels(&b, r, BITFIELD_RANGE(frac, n));
         nir_store_deref(&b, deref, glsl_get_bit_size(t) == 16 ? nir_f2f16(&b, v) : v, BITFIELD_MASK(n));
      }
      *reads |= BITFIELD_BIT(rt);
   }
   /* The second output, a local now. */
   if (dual) {
      nir_remove_dead_derefs_impl(impl);
      nir_foreach_shader_out_variable_safe(var, nir) {
         if (var->data.index != 0 && var->data.location == FRAG_RESULT_DATA0)
            exec_node_remove(&var->node);
      }
   }
   return nir_progress(true, impl, nir_metadata_none);
}

static bool
is_constant_alpha_factor(uint8_t f)
{
   return f == VK_BLEND_FACTOR_CONSTANT_ALPHA || f == VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
}

static bool
is_constant_color_factor(uint8_t f)
{
   return f == VK_BLEND_FACTOR_CONSTANT_COLOR || f == VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
}

uint32_t
wvk_shader_blend_mask(const struct vk_dynamic_graphics_state *dyn, uint32_t written)
{
   uint32_t alpha = 0, color = 0;
   for (uint32_t i = 0; i < MIN2(dyn->cb.attachment_count, 8); ++i) {
      const struct vk_color_blend_attachment_state *a = &dyn->cb.attachments[i];
      if (!a->blend_enable || !(written & BITFIELD_BIT(i)) || !a->write_mask)
         continue;
      if (is_constant_alpha_factor(a->src_color_blend_factor) || is_constant_alpha_factor(a->dst_color_blend_factor))
         alpha |= BITFIELD_BIT(i);
      if (is_constant_color_factor(a->src_color_blend_factor) || is_constant_color_factor(a->dst_color_blend_factor))
         color |= BITFIELD_BIT(i);
   }
   return alpha && color ? alpha : 0;
}

/* The vertex a point drawn as a quad is of (expand_points), or a vertex
 * shader run as compute (run_vertices), in a local of this name: where its
 * attributes are pulled from; and the instance, of the latter. */
#define WVK_POINT_VERTEX "wvk_vertex"
#define WVK_RUN_INSTANCE "wvk_instance"

static nir_def *load_index(nir_builder *b, nir_def *start, nir_def *size, nir_def *at, nir_def **restarts);

/* Points drawn as quads (largePoints), which WebGPU has none of: six
 * vertices a point, which wvk_cmd_draw.c draws as a triangle list. The
 * point is vertex_index / 6 and the corner vertex_index % 6; the point's
 * vertex is that, or for an indexed draw its index in the index buffer
 * plus BaseVertex, and every attribute is pulled at it (the local
 * WVK_POINT_VERTEX, pull_divided_attributes) -- none where the index
 * restarts primitives, the quad out of the clip volume. At the end the position
 * moves to the corner by half of PointSize, in pixels of the viewport,
 * whose inverse size the driver's data has: in Vulkan's clip space, before
 * the viewport transform, where +y is down whatever the viewport's sign.
 * And gl_PointCoord goes to the fragment shader at `point_coord` - 1. */
static void
expand_points(nir_shader *nir, unsigned point_coord)
{
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   /* The point, its corner, and its vertex. */
   nir_def *raw = nir_load_vertex_id(&b);
   nir_def *point = nir_udiv_imm(&b, raw, 6);
   nir_def *corner = nir_umod_imm(&b, raw, 6);
   nir_def *index = nir_load_ssbo(&b, 3, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                  nir_imm_int(&b, WVK_DRIVER_POINT_INDEX), .align_mul = 8,
                                  .access = ACCESS_NON_WRITEABLE);
   nir_def *start = nir_channel(&b, index, 0), *size = nir_channel(&b, index, 1);
   nir_def *no = nir_imm_false(&b);
   nir_push_if(&b, nir_ine_imm(&b, size, 0));
   nir_def *restarts;
   nir_def *indexed = nir_iadd(&b, load_index(&b, start, size, point, &restarts), nir_load_base_vertex(&b));
   restarts = nir_iand(&b, restarts, nir_ine_imm(&b, nir_channel(&b, index, 2), 0));
   nir_push_else(&b, NULL);
   nir_pop_if(&b, NULL);
   nir_def *vertex = nir_if_phi(&b, indexed, point);
   nir_def *gone = nir_if_phi(&b, restarts, no);
   nir_variable *element = nir_local_variable_create(impl, glsl_uint_type(), WVK_POINT_VERTEX);
   nir_store_var(&b, element, vertex, 0x1);

   /* What the shader says of its vertex is the point's; and its position
    * and point size go to locals, until the end. */
   nir_variable *position_out = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_POS);
   nir_variable *size_out = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_PSIZ);
   if (!position_out) {
      position_out = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
      position_out->data.location = VARYING_SLOT_POS;
   }
   nir_variable *position = nir_local_variable_create(impl, glsl_vec4_type(), "wvk_point_position");
   nir_variable *point_size = nir_local_variable_create(impl, glsl_float_type(), "wvk_point_size");
   nir_store_var(&b, position, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f), 0xf);
   nir_store_var(&b, point_size, nir_imm_float(&b, 1.0f), 0x1);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_vertex_id && &intr->def != raw)
               nir_def_replace(&intr->def, vertex);
            continue;
         }
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var || (deref->var != position_out && deref->var != size_out))
            continue;
         deref->var = deref->var == position_out ? position : point_size;
         deref->modes = nir_var_function_temp;
      }
   }
   nir_fixup_deref_modes(nir);

   b.cursor = nir_after_impl(impl);
   nir_def *pos = nir_load_var(&b, position);
   nir_def *diameter = nir_fmin(&b, nir_fmax(&b, nir_load_var(&b, point_size), nir_imm_float(&b, 1.0f)),
                                nir_imm_float(&b, WVK_MAX_POINT_SIZE));
   nir_def *scale = nir_load_ssbo(&b, 2, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                  nir_imm_int(&b, WVK_DRIVER_POINT_SCALE), .align_mul = 8,
                                  .access = ACCESS_NON_WRITEABLE);
   /* Corners 0 1 2 and 2 1 3 of the square: -1 left or up, +1 right or down. */
   nir_def *right = nir_ior(&b, nir_ieq_imm(&b, corner, 1), nir_uge_imm(&b, corner, 4));
   nir_def *down = nir_ior(&b, nir_ieq_imm(&b, corner, 2), nir_ior(&b, nir_ieq_imm(&b, corner, 3),
                                                                  nir_ieq_imm(&b, corner, 5)));
   nir_def *dx = nir_bcsel(&b, right, nir_imm_float(&b, 1.0f), nir_imm_float(&b, -1.0f));
   nir_def *dy = nir_bcsel(&b, down, nir_imm_float(&b, 1.0f), nir_imm_float(&b, -1.0f));
   nir_def *w = nir_channel(&b, pos, 3);
   nir_def *reach = nir_fmul(&b, diameter, w);
   nir_def *x = nir_ffma(&b, nir_fmul(&b, dx, nir_channel(&b, scale, 0)), reach, nir_channel(&b, pos, 0));
   nir_def *y = nir_ffma(&b, nir_fmul(&b, dy, nir_channel(&b, scale, 1)), reach, nir_channel(&b, pos, 1));
   nir_store_var(&b, position_out,
                 nir_bcsel(&b, gone, nir_imm_vec4(&b, 2.0f, 2.0f, 2.0f, 1.0f), nir_vec4(&b, x, y, nir_channel(&b, pos, 2), w)),
                 0xf);
   if (point_coord) {
      nir_variable *coord = nir_find_variable_with_location(nir, nir_var_shader_out,
                                                            VARYING_SLOT_VAR0 + point_coord - 1);
      if (!coord) {
         coord = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "wvk_point_coord");
         coord->data.location = VARYING_SLOT_VAR0 + point_coord - 1;
         coord->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
      }
      nir_def *s = nir_fmul_imm(&b, nir_fadd_imm(&b, dx, 1.0f), 0.5f);
      nir_def *t = nir_fmul_imm(&b, nir_fadd_imm(&b, dy, 1.0f), 0.5f);
      nir_store_var(&b, coord, nir_vec4(&b, s, t, nir_imm_float(&b, 0.0f), nir_imm_float(&b, 1.0f)), 0xf);
   }
   /* PointSize is WebGPU's to refuse, and no longer written. */
   if (size_out)
      exec_node_remove(&size_out->node);
   nir_progress(true, impl, nir_metadata_none);
}

/* An index of the draw's index buffer at `at`, as the driver's data says
 * it is (WVK_DRIVER_POINT_INDEX: where it starts, and its size), and
 * whether it is the one that restarts primitives. */
static nir_def *
load_index(nir_builder *b, nir_def *start, nir_def *size, nir_def *at, nir_def **restarts)
{
   nir_def *byte = nir_iadd(b, start, nir_imul(b, at, size));
   nir_def *word = nir_load_ssbo(b, 1, 32, driver_buffer(b, WVK_DRIVER_SLOT_INDEX), nir_iand_imm(b, byte, ~3),
                                 .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
   nir_def *bits = nir_ishl_imm(b, size, 3);
   nir_def *part = nir_ubitfield_extract(b, word, nir_ishl_imm(b, nir_iand_imm(b, byte, 3), 3), bits);
   nir_def *four = nir_ieq_imm(b, size, 4);
   nir_def *value = nir_bcsel(b, four, word, part);
   *restarts = nir_ieq(b, value, nir_bcsel(b, four, nir_imm_int(b, -1),
                                           nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), bits), -1)));
   return value;
}

/* Wide lines as quads (wideLines), which WebGPU has none of: six vertices
 * a segment, a triangle list of two (wvk_cmd_draw.c). The segment is
 * vertex_index / 6 -- its endpoints 2s and 2s + 1 of a list, s and s + 1
 * of a strip, from the draw's first vertex or index -- and the corner
 * vertex_index % 6, which is of one endpoint: the vertex shader runs for
 * that one, with its attributes pulled there. At its end the position
 * moves across the line by half its width: up or down where the line is
 * more across than up, else sideways, as Vulkan's non-strict and Bresenham
 * lines are; or along its normal, for rectangular lines, and by a pixel
 * more for smooth ones -- the other endpoint's position read from what the
 * draw's vertex shader, run as compute, captured (run_vertices). A segment
 * across a restart goes. With `line_coord`, the varying there says where
 * on the line the vertex is, in pixels: across it, and along it from the
 * start of the list's line or the strip -- a rectangle's length, or a
 * parallelogram's along its major axis -- for the stipple and the coverage
 * (lower_line_fs). */
static void
expand_lines(nir_shader *nir, unsigned line_coord)
{
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   nir_def *raw = nir_load_vertex_id(&b);
   nir_def *segment = nir_udiv_imm(&b, raw, 6);
   nir_def *corner = nir_umod_imm(&b, raw, 6);
   nir_def *line = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                 nir_imm_int(&b, WVK_DRIVER_LINE), .align_mul = 16, .access = ACCESS_NON_WRITEABLE);
   nir_def *run = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                nir_imm_int(&b, WVK_DRIVER_VERTEX_RUN), .align_mul = 16,
                                .access = ACCESS_NON_WRITEABLE);
   nir_def *index = nir_load_ssbo(&b, 2, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                  nir_imm_int(&b, WVK_DRIVER_POINT_INDEX), .align_mul = 8,
                                  .access = ACCESS_NON_WRITEABLE);
   nir_def *first_end = nir_bcsel(&b, nir_ine_imm(&b, nir_channel(&b, line, 1), 0), segment,
                                  nir_ishl_imm(&b, segment, 1));
   /* Corners 0 1 2 and 3 4 5: A- B- A+ and A+ B- B+. */
   nir_def *at_b = nir_ior(&b, nir_ieq_imm(&b, corner, 1), nir_uge_imm(&b, corner, 4));
   nir_def *up = nir_ior(&b, nir_ieq_imm(&b, corner, 2), nir_ior(&b, nir_ieq_imm(&b, corner, 3),
                                                                nir_ieq_imm(&b, corner, 5)));
   nir_def *own_at = nir_iadd(&b, first_end, nir_b2i32(&b, at_b));
   nir_def *other_at = nir_iadd(&b, first_end, nir_b2i32(&b, nir_inot(&b, at_b)));
   nir_def *first = nir_channel(&b, run, 2);
   nir_def *start = nir_channel(&b, index, 0), *size = nir_channel(&b, index, 1);
   nir_def *no = nir_imm_false(&b);
   nir_def *plain = nir_iadd(&b, first, own_at);
   nir_push_if(&b, nir_ine_imm(&b, size, 0));
   nir_def *own_restarts, *other_restarts;
   nir_def *own = load_index(&b, start, size, nir_iadd(&b, first, own_at), &own_restarts);
   load_index(&b, start, size, nir_iadd(&b, first, other_at), &other_restarts);
   nir_def *indexed = nir_iadd(&b, own, nir_load_base_vertex(&b));
   nir_def *restarts = nir_iand(&b, nir_ine_imm(&b, nir_channel(&b, run, 3), 0),
                                nir_ior(&b, own_restarts, other_restarts));
   nir_push_else(&b, NULL);
   nir_pop_if(&b, NULL);
   nir_def *vertex = nir_if_phi(&b, indexed, plain);
   nir_def *gone = nir_if_phi(&b, restarts, no);
   nir_variable *element = nir_local_variable_create(impl, glsl_uint_type(), WVK_POINT_VERTEX);
   nir_store_var(&b, element, vertex, 0x1);

   nir_variable *position_out = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_POS);
   if (!position_out) {
      position_out = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
      position_out->data.location = VARYING_SLOT_POS;
   }
   nir_variable *position = nir_local_variable_create(impl, glsl_vec4_type(), "wvk_line_position");
   nir_store_var(&b, position, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f), 0xf);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_vertex_id && &intr->def != raw)
               nir_def_replace(&intr->def, vertex);
            continue;
         }
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var || deref->var != position_out)
            continue;
         deref->var = position;
         deref->modes = nir_var_function_temp;
      }
   }
   nir_fixup_deref_modes(nir);

   b.cursor = nir_after_impl(impl);
   nir_def *pos = nir_load_var(&b, position);
   nir_def *instance = nir_isub(&b, nir_load_instance_id(&b), nir_load_base_instance(&b));
   nir_def *other_slot = nir_iadd(&b, nir_imul(&b, instance, nir_channel(&b, run, 0)), other_at);
   nir_def *other = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_CAPTURE),
                                  nir_ishl_imm(&b, other_slot, 4), .align_mul = 16,
                                  .access = ACCESS_NON_WRITEABLE);
   nir_def *scale = nir_load_ssbo(&b, 2, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                  nir_imm_int(&b, WVK_DRIVER_POINT_SCALE), .align_mul = 8,
                                  .access = ACCESS_NON_WRITEABLE);
   /* Which way the line goes, in pixels. */
   nir_def *own_xy = nir_fdiv(&b, nir_trim_vector(&b, pos, 2), nir_channel(&b, pos, 3));
   nir_def *other_xy = nir_fdiv(&b, nir_trim_vector(&b, other, 2), nir_channel(&b, other, 3));
   nir_def *d = nir_fdiv(&b, nir_fsub(&b, other_xy, own_xy), scale);
   nir_def *across = nir_fge(&b, nir_fabs(&b, nir_channel(&b, d, 0)), nir_fabs(&b, nir_channel(&b, d, 1)));
   nir_def *side = nir_bcsel(&b, up, nir_imm_float(&b, 1.0f), nir_imm_float(&b, -1.0f));
   nir_def *width = nir_fmin(&b, nir_fmax(&b, nir_channel(&b, line, 0), nir_imm_float(&b, 1.0f)),
                             nir_imm_float(&b, WVK_MAX_LINE_WIDTH));
   nir_def *reach = nir_fmul(&b, nir_fmul(&b, side, width), nir_channel(&b, pos, 3));
   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *dx = nir_bcsel(&b, across, zero, nir_fmul(&b, reach, nir_channel(&b, scale, 0)));
   nir_def *dy = nir_bcsel(&b, across, nir_fmul(&b, reach, nir_channel(&b, scale, 1)), zero);
   /* A rectangle's: along the normal of A to B -- d is twice the pixels,
    * from the vertex to the other end -- by half the width, and a pixel
    * more where it is smooth, the coverage falling off there. */
   nir_def *mode = nir_channel(&b, line, 2);
   nir_def *rectangle = nir_ine_imm(&b, mode, 0);
   nir_def *ab = nir_fmul_imm(&b, nir_bcsel(&b, at_b, nir_fneg(&b, d), d), 0.5f);
   nir_def *length = nir_fsqrt(&b, nir_fdot(&b, ab, ab));
   nir_def *normal = nir_bcsel(&b, nir_flt_imm(&b, length, 1e-6f), nir_imm_vec2(&b, 0.0f, 1.0f),
                               nir_fdiv(&b, nir_vec2(&b, nir_fneg(&b, nir_channel(&b, ab, 1)), nir_channel(&b, ab, 0)),
                                        length));
   nir_def *half = nir_fadd(&b, nir_fmul_imm(&b, width, 0.5f),
                            nir_bcsel(&b, nir_ieq_imm(&b, mode, 2), nir_imm_float(&b, 1.0f), zero));
   nir_def *out = nir_fmul(&b, nir_fmul_imm(&b, nir_fmul(&b, side, half), 2.0f), nir_channel(&b, pos, 3));
   dx = nir_bcsel(&b, rectangle, nir_fmul(&b, out, nir_fmul(&b, nir_channel(&b, normal, 0), nir_channel(&b, scale, 0))),
                  dx);
   dy = nir_bcsel(&b, rectangle, nir_fmul(&b, out, nir_fmul(&b, nir_channel(&b, normal, 1), nir_channel(&b, scale, 1))),
                  dy);
   nir_def *moved = nir_vec4(&b, nir_fadd(&b, nir_channel(&b, pos, 0), dx), nir_fadd(&b, nir_channel(&b, pos, 1), dy),
                             nir_channel(&b, pos, 2), nir_channel(&b, pos, 3));
   nir_store_var(&b, position_out, nir_bcsel(&b, gone, nir_imm_zero(&b, 4, 32), moved), 0xf);

   if (line_coord) {
      /* How long a segment is, of pixels: a rectangle's length, a
       * parallelogram's along its major axis. */
#define SEGMENT_LENGTH(v)                                                                                        \
      nir_bcsel(&b, rectangle, nir_fsqrt(&b, nir_fdot(&b, v, v)),                                             \
                nir_fmax(&b, nir_fabs(&b, nir_channel(&b, v, 0)), nir_fabs(&b, nir_channel(&b, v, 1))))
      /* Of a stippled strip, the segments before this one, back to the
       * strip's start or its last restart: from their ends' positions. */
      nir_variable *before = nir_local_variable_create(impl, glsl_float_type(), "wvk_line_before");
      nir_variable *at = nir_local_variable_create(impl, glsl_uint_type(), "wvk_line_segment");
      nir_store_var(&b, before, zero, 0x1);
      nir_store_var(&b, at, segment, 0x1);
      nir_push_if(&b, nir_iand(&b, nir_ine_imm(&b, nir_channel(&b, line, 1), 0),
                              nir_ine_imm(&b, nir_channel(&b, line, 3), 0)));
      nir_loop *loop = nir_push_loop(&b);
      {
         nir_def *j = nir_load_var(&b, at);
         nir_break_if(&b, nir_ieq_imm(&b, j, 0));
         nir_def *prev = nir_iadd_imm(&b, j, -1);
         nir_def *no_restart = nir_imm_false(&b);
         nir_push_if(&b, nir_iand(&b, nir_ine_imm(&b, size, 0), nir_ine_imm(&b, nir_channel(&b, run, 3), 0)));
         nir_def *r0, *r1;
         load_index(&b, start, size, nir_iadd(&b, first, prev), &r0);
         load_index(&b, start, size, nir_iadd(&b, first, j), &r1);
         nir_def *restarted = nir_ior(&b, r0, r1);
         nir_pop_if(&b, NULL);
         nir_break_if(&b, nir_if_phi(&b, restarted, no_restart));
         nir_def *slot = nir_iadd(&b, nir_imul(&b, instance, nir_channel(&b, run, 0)), prev);
         nir_def *p = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_CAPTURE), nir_ishl_imm(&b, slot, 4),
                                    .align_mul = 16, .access = ACCESS_NON_WRITEABLE);
         nir_def *q = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_CAPTURE),
                                    nir_ishl_imm(&b, nir_iadd_imm(&b, slot, 1), 4), .align_mul = 16,
                                    .access = ACCESS_NON_WRITEABLE);
         nir_def *pq = nir_fmul_imm(&b, nir_fdiv(&b, nir_fsub(&b, nir_fdiv(&b, nir_trim_vector(&b, q, 2), nir_channel(&b, q, 3)),
                                                            nir_fdiv(&b, nir_trim_vector(&b, p, 2), nir_channel(&b, p, 3))),
                                                 scale),
                                    0.5f);
         nir_store_var(&b, before, nir_fadd(&b, nir_load_var(&b, before), SEGMENT_LENGTH(pq)), 0x1);
         nir_store_var(&b, at, prev, 0x1);
      }
      nir_pop_loop(&b, loop);
      nir_pop_if(&b, NULL);
      nir_def *along = nir_fadd(&b, nir_load_var(&b, before), nir_bcsel(&b, at_b, SEGMENT_LENGTH(ab), zero));
#undef SEGMENT_LENGTH
      nir_def *across_px = nir_fmul(&b, side, nir_bcsel(&b, rectangle, half, nir_fmul_imm(&b, width, 0.5f)));
      /* The output the fragment shader's input there was given, if any
       * (match_fragment_inputs), as expand_points has it. */
      nir_variable *coord = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_VAR0 + line_coord - 1);
      if (!coord) {
         coord = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "wvk_line_coord");
         coord->data.location = VARYING_SLOT_VAR0 + line_coord - 1;
      }
      coord->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
      nir_store_var(&b, coord, nir_vec4(&b, across_px, along, zero, zero), 0xf);
   }
   nir_progress(true, impl, nir_metadata_none);
}

/* The variable of the cull distances among `mode`'s: after the clip
 * distances, in their slots (nir_merge_clip_cull_distance_vars). */
static nir_variable *
cull_distance_var(nir_shader *nir, nir_variable_mode mode)
{
   if (!nir->info.cull_distance_array_size)
      return NULL;
   const unsigned first = nir->info.clip_distance_array_size;
   nir_foreach_variable_with_modes(var, nir, mode) {
      if (var->data.compact && var->data.location >= VARYING_SLOT_CLIP_DIST0 &&
          var->data.location <= VARYING_SLOT_CLIP_DIST1 &&
          (var->data.location - VARYING_SLOT_CLIP_DIST0) * 4 + var->data.location_frac == first)
         return var;
   }
   return NULL;
}

static int record_slot(unsigned location);

/* What transform feedback captures of a stage's outputs, from its
 * nir_xfb_info: by record slot, where a record has the output. */
static void
gather_xfb(struct wvk_shader *shader, const nir_shader *nir)
{
   const nir_xfb_info *xfb = nir->xfb_info;
   shader->xfb_buffers = 0;
   shader->xfb_output_count = 0;
   if (!xfb)
      return;
   for (unsigned b = 0; b < 4; ++b) {
      shader->xfb_strides[b] = xfb->buffers[b].stride;
      shader->xfb_streams[b] = MIN2(xfb->buffer_to_stream[b], 3);
      if (xfb->buffers_written & BITFIELD_BIT(b))
         shader->xfb_buffers |= BITFIELD_BIT(b);
   }
   for (unsigned i = 0; i < xfb->output_count && shader->xfb_output_count < WVK_MAX_XFB_OUTPUTS; ++i) {
      const nir_xfb_output_info *o = &xfb->outputs[i];
      const int slot = record_slot(o->location);
      if (slot < 0 || !o->component_mask || !(shader->xfb_buffers & BITFIELD_BIT(o->buffer)))
         continue;
      shader->xfb_outputs[shader->xfb_output_count++] = (struct wvk_xfb_output){
         .slot = slot, .component = ffs(o->component_mask) - 1, .count = util_bitcount(o->component_mask),
         .buffer = o->buffer, .offset = o->offset,
      };
   }
}

/* A varying's slot in a record (WVK_RECORD_BUILTINS), or -1 for one it
 * has not. */
static int
record_slot(unsigned location)
{
   switch (location) {
   case VARYING_SLOT_POS: return 0;
   case VARYING_SLOT_PSIZ: return 1;
   case VARYING_SLOT_CLIP_DIST0: return 2;
   case VARYING_SLOT_CLIP_DIST1: return 3;
   case VARYING_SLOT_CULL_DIST0: return 4;
   case VARYING_SLOT_CULL_DIST1: return 5;
   case VARYING_SLOT_LAYER: return 6;
   case VARYING_SLOT_VIEWPORT: return 7;
   case VARYING_SLOT_PRIMITIVE_ID: return 8;
   default:
      return location >= VARYING_SLOT_VAR0 && location < VARYING_SLOT_VAR0 + 32
                ? WVK_RECORD_BUILTINS + (int)(location - VARYING_SLOT_VAR0)
                : -1;
   }
}

/* How many generic varyings a record of these variables has: to the last
 * slot of theirs. */
static unsigned
record_vars(nir_shader *nir, nir_variable_mode mode)
{
   unsigned vars = 0;
   nir_foreach_variable_with_modes(var, nir, mode) {
      if (var->data.location < VARYING_SLOT_VAR0 || var->data.location >= VARYING_SLOT_VAR0 + 32 || var->data.patch)
         continue;
      const struct glsl_type *type = nir_is_arrayed_io(var, nir->info.stage) ? glsl_get_array_element(var->type)
                                                                             : var->type;
      vars = MAX2(vars, var->data.location - VARYING_SLOT_VAR0 + glsl_count_vec4_slots(type, false, true));
   }
   return MIN2(vars, 32);
}

/* A patch varying's slot in a patch's record (wvk_patch_stride), or -1. */
static int
patch_slot(unsigned location)
{
   if (location == VARYING_SLOT_TESS_LEVEL_OUTER)
      return 0;
   if (location == VARYING_SLOT_TESS_LEVEL_INNER)
      return 1;
   return location >= VARYING_SLOT_PATCH0 && location < VARYING_SLOT_PATCH0 + 30
             ? 2 + (int)(location - VARYING_SLOT_PATCH0)
             : -1;
}

/* How many patch varyings a patch's record has: to the last of theirs. */
static unsigned
patch_vars(nir_shader *nir, nir_variable_mode mode)
{
   unsigned vars = 0;
   nir_foreach_variable_with_modes(var, nir, mode) {
      if (!var->data.patch || var->data.location < VARYING_SLOT_PATCH0)
         continue;
      vars = MAX2(vars, var->data.location - VARYING_SLOT_PATCH0 + glsl_count_vec4_slots(var->type, false, true));
   }
   return MIN2(vars, 30);
}

static unsigned
type_size_vec4(const struct glsl_type *type, bool bindless)
{
   return glsl_count_vec4_slots(type, false, bindless);
}

/* A lowered output's store or load made one of a record in buffer
 * `binding` from byte `base`: words of 32 bits, a vec4 a slot. What is not
 * in a record is dropped, or reads as 0. */
static void
record_access(nir_builder *b, nir_intrinsic_instr *intr, uint32_t binding, nir_def *base, bool patch)
{
   const bool store = intr->intrinsic == nir_intrinsic_store_output ||
                      intr->intrinsic == nir_intrinsic_store_per_vertex_output;
   const unsigned location = nir_intrinsic_io_semantics(intr).location;
   const int slot = patch ? patch_slot(location) : record_slot(location);
   nir_def *offset = nir_get_io_offset_src(intr)->ssa;
   const unsigned component = nir_intrinsic_component(intr);
   if (slot < 0) {
      if (store)
         nir_instr_remove(&intr->instr);
      else
         nir_def_replace(&intr->def, nir_imm_zero(b, intr->def.num_components, intr->def.bit_size));
      return;
   }
   nir_def *byte = nir_iadd(b, base, nir_iadd_imm(b, nir_ishl_imm(b, offset, 4), slot * 16 + component * 4));
   if (store) {
      nir_def *value = intr->src[0].ssa;
      const nir_alu_type type = nir_intrinsic_src_type(intr);
      unsigned mask = nir_intrinsic_write_mask(intr) & BITFIELD_MASK(value->num_components);
      if (value->bit_size == 16) {
         value = nir_alu_type_get_base_type(type) == nir_type_float ? nir_f2f32(b, value)
                 : nir_alu_type_get_base_type(type) == nir_type_int ? nir_i2i32(b, value)
                                                                      : nir_u2u32(b, value);
      } else if (value->bit_size == 64) {
         /* Two words a component, as its slot has them. */
         for (unsigned c = 0; c < value->num_components; ++c) {
            if (mask & BITFIELD_BIT(c))
               nir_store_ssbo(b, nir_unpack_64_2x32(b, nir_channel(b, value, c)), driver_buffer(b, binding),
                              nir_iadd_imm(b, byte, c * 8), .write_mask = 0x3, .align_mul = 4);
         }
         nir_instr_remove(&intr->instr);
         return;
      }
      nir_store_ssbo(b, value, driver_buffer(b, binding), byte, .write_mask = mask, .align_mul = 4);
      nir_instr_remove(&intr->instr);
   } else {
      nir_def *value;
      if (intr->def.bit_size == 64) {
         nir_def *out[4];
         for (unsigned c = 0; c < intr->def.num_components; ++c)
            out[c] = nir_pack_64_2x32(b, nir_load_ssbo(b, 2, 32, driver_buffer(b, binding), nir_iadd_imm(b, byte, c * 8),
                                                       .align_mul = 4));
         value = nir_vec(b, out, intr->def.num_components);
      } else {
         value = nir_load_ssbo(b, intr->def.num_components, 32, driver_buffer(b, binding), byte, .align_mul = 4);
      }
      if (intr->def.bit_size == 16) {
         value = nir_alu_type_get_base_type(nir_intrinsic_dest_type(intr)) == nir_type_float ? nir_f2f16(b, value)
                                                                                           : nir_u2u16(b, value);
      }
      nir_def_replace(&intr->def, value);
   }
}

/* The draw's vertex, 0 its first, that is vertex `j` of the geometry
 * shader's input primitive `prim`, for the draw's topology (a
 * VkPrimitiveTopology), as Vulkan's primitive assembly has it. */
static nir_def *
primitive_vertex(nir_builder *b, nir_def *topology, nir_def *prim, nir_def *j, nir_def *prims)
{
   nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, prim, 1), 0);
   /* A strip's triangle: i, i + 1, i + 2, or i, i + 2, i + 1 where odd. */
   nir_def *strip = nir_iadd(b, prim, nir_bcsel(b, nir_iand(b, odd, nir_ine_imm(b, j, 0)), nir_isub_imm(b, 3, j), j));
   /* A fan's: i + 1, i + 2, 0. */
   nir_def *fan = nir_bcsel(b, nir_ieq_imm(b, j, 2), nir_imm_int(b, 0), nir_iadd(b, prim, nir_iadd_imm(b, j, 1)));
   /* A strip's with adjacency: Vulkan's table of first, middle and last
    * primitives, even and odd, and of one alone. */
   nir_def *i2 = nir_ishl_imm(b, prim, 1);
   nir_def *first = nir_ieq_imm(b, prim, 0), *last = nir_ieq(b, prim, nir_iadd_imm(b, prims, -1));
   static const int8_t alone[6] = { 0, 1, 2, 5, 4, 3 }, head[6] = { 0, 1, 2, 6, 4, 3 };
   static const int8_t mid_odd[6] = { 2, -2, 0, 3, 4, 6 }, mid_even[6] = { 0, -2, 2, 6, 4, 3 };
   static const int8_t last_odd[6] = { 2, -2, 0, 3, 4, 5 }, last_even[6] = { 0, -2, 2, 5, 4, 3 };
   nir_def *adj = nir_imm_int(b, 0);
   for (int k = 5; k >= 0; --k) {
      nir_def *mid = nir_bcsel(b, odd, nir_imm_int(b, mid_odd[k]), nir_imm_int(b, mid_even[k]));
      nir_def *end = nir_bcsel(b, odd, nir_imm_int(b, last_odd[k]), nir_imm_int(b, last_even[k]));
      nir_def *rel = nir_bcsel(b, last, end, mid);
      nir_def *v = nir_bcsel(b, first, nir_imm_int(b, head[k]), nir_iadd(b, i2, rel));
      v = nir_bcsel(b, nir_iand(b, first, last), nir_imm_int(b, alone[k]), v);
      adj = nir_bcsel(b, nir_ieq_imm(b, j, k), v, adj);
   }
   nir_def *per = nir_bcsel(b, nir_ieq_imm(b, topology, VK_PRIMITIVE_TOPOLOGY_LINE_LIST), nir_imm_int(b, 2),
                  nir_bcsel(b, nir_ieq_imm(b, topology, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST), nir_imm_int(b, 3),
                  nir_bcsel(b, nir_ieq_imm(b, topology, VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY), nir_imm_int(b, 4),
                  nir_bcsel(b, nir_ieq_imm(b, topology, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY),
                            nir_imm_int(b, 6), nir_imm_int(b, 0)))));
   nir_def *list = nir_iadd(b, nir_imul(b, prim, per), j);
   nir_def *v = nir_bcsel(b, nir_ine_imm(b, per, 0), list, nir_iadd(b, prim, j));
   v = nir_bcsel(b, nir_ieq_imm(b, topology, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP), strip, v);
   v = nir_bcsel(b, nir_ieq_imm(b, topology, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN), fan, v);
   v = nir_bcsel(b, nir_ieq_imm(b, topology, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY), adj, v);
   return v;
}

/* The vertex streams a geometry shader emits to, to the last one: of its
 * emits, however much of its info has been gathered. */
static unsigned
wvk_geometry_streams(const nir_shader *nir)
{
   unsigned streams = 1;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_emit_vertex || intr->intrinsic == nir_intrinsic_end_primitive)
               streams = MAX2(streams, MIN2(nir_intrinsic_stream_id(intr), 3) + 1);
         }
      }
   }
   return streams;
}

/* A geometry shader as compute (key.geometry 1), which WebGPU has no stage
 * of: an invocation a primitive of the draw, instance by instance, and a
 * shader invocation of each (the driver's data, WVK_DRIVER_GEOMETRY). Its
 * inputs are the vertex shader's records (key.runs 3) of the primitive's
 * vertices; its outputs a record it keeps, which EmitVertex writes out at
 * the invocation's next vertex -- slot 0 is none, of position 0, which
 * unwritten indices name -- with the indices of what the vertex completes
 * of a strip, as a list: which the draw after draws (key.geometry 2).
 * Each vertex stream's vertices and lists are after those of all
 * invocations of the streams before (as many as the driver's data says
 * there is room for), and its primitives are counted apart, clipping's of
 * the rasterized stream's. Compute's stage after its buffers are words
 * (compile_one). */
static void
lower_geometry(nir_shader *nir, struct wvk_shader *shader, unsigned vs_vars, bool after_tess, bool stats,
               unsigned raster_stream, bool provoking_last)
{
   const unsigned vertices_in = mesa_vertices_per_prim(nir->info.gs.input_primitive);
   const unsigned streams = wvk_geometry_streams(nir);
   const unsigned max_vertices = nir->info.gs.vertices_out;
   const unsigned invocations = MAX2(nir->info.gs.invocations, 1);
   const unsigned output = nir->info.gs.output_primitive;
   const unsigned indices = wvk_geometry_indices(output, max_vertices);
   const uint32_t in_stride = wvk_record_stride(vs_vars);
   shader->record_vars = record_vars(nir, nir_var_shader_out);
   const uint32_t out_stride = wvk_record_stride(shader->record_vars);
   const unsigned slots = WVK_RECORD_BUILTINS + shader->record_vars;
   /* Its cull distances, after its clip distances. */
   const unsigned culls = cull_distance_var(nir, nir_var_shader_out) ? MIN2(nir->info.cull_distance_array_size, 8) : 0;
   const unsigned first_cull = nir->info.clip_distance_array_size;

   NIR_PASS(_, nir, nir_lower_gs_intrinsics,
            nir_lower_gs_intrinsics_per_stream | nir_lower_gs_intrinsics_count_vertices_per_primitive);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out)
      var->data.driver_location = var->data.location;
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out, type_size_vec4,
            nir_lower_io_lower_64bit_to_32);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   /* Which invocation of which primitive of which instance. */
   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_def *groups = nir_load_num_workgroups(&b);
   nir_def *g = nir_iadd(&b, nir_channel(&b, id, 0),
                         nir_imul(&b, nir_channel(&b, id, 1), nir_imul_imm(&b, nir_channel(&b, groups, 0), 64)));
   nir_def *geo = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                nir_imm_int(&b, WVK_DRIVER_GEOMETRY), .align_mul = 16,
                                .access = ACCESS_NON_WRITEABLE);
   nir_def *count = nir_channel(&b, geo, 0), *prims = nir_umax(&b, nir_channel(&b, geo, 1), nir_imm_int(&b, 1));
   nir_def *topology = nir_channel(&b, geo, 3);
   nir_def *total = nir_channel(&b, geo, 2);
   /* The invocations each stream has room for: all of them, or after
    * tessellation as many as its buffers were made for. */
   nir_def *capacity = total;
   nir_def *invocation = nir_umod_imm(&b, g, invocations);
   nir_def *primitive = nir_udiv_imm(&b, g, invocations);
   nir_def *prim = nir_umod(&b, primitive, prims);
   nir_def *instance = nir_udiv(&b, primitive, prims);
   nir_def *vertex_base = nir_imul(&b, instance, count);
   /* After tessellation, as many as the tessellator made, of its lists
    * (wvk_tessellator.c's scan): every primitive of every instance. */
   if (after_tess) {
      nir_def *tess = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                    nir_imm_int(&b, WVK_DRIVER_TESS), .align_mul = 16,
                                    .access = ACCESS_NON_WRITEABLE);
      total = nir_load_ssbo(&b, 1, 32, driver_buffer(&b, WVK_DRIVER_SLOT_TESS_COUNTS),
                            nir_iadd_imm(&b, nir_ishl_imm(&b, nir_channel(&b, tess, 3), 4), 13 * 4),
                            .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
      prim = primitive;
   }
   /* The vertex it writes next is the first of its own; a stream's, of
    * the stream's, after all invocations' of the streams before. */
   nir_def *first_out[4], *first_index[4];
   nir_variable *made[4];
   for (unsigned st = 0; st < streams; ++st) {
      nir_def *slot = st ? nir_iadd(&b, nir_imul_imm(&b, capacity, st), g) : g;
      first_out[st] = nir_iadd_imm(&b, nir_imul_imm(&b, slot, max_vertices), 1);
      first_index[st] = nir_imul_imm(&b, slot, indices);
      made[st] = nir_local_variable_create(impl, glsl_uint_type(), "wvk_gs_primitives");
      nir_store_var(&b, made[st], nir_imm_int(&b, 0), 0x1);
   }
   nir_variable *record = nir_local_variable_create(impl, glsl_array_type(glsl_uvec4_type(), slots, 16),
                                                    "wvk_gs_record");
   for (unsigned i = 0; i < slots; ++i)
      nir_store_deref(&b, nir_build_deref_array_imm(&b, nir_build_deref_var(&b, record), i), nir_imm_zero(&b, 4, 32),
                      0xf);
   /* The vertex unwritten indices name: outside the clip volume, which a
    * point or line of it is not drawn in either. */
   nir_push_if(&b, nir_ieq_imm(&b, g, 0));
   nir_store_ssbo(&b, nir_imm_vec4(&b, 2.0f, 2.0f, 2.0f, 1.0f), driver_buffer(&b, WVK_DRIVER_SLOT_GS_VERTICES),
                  nir_imm_int(&b, 0), .write_mask = 0xf, .align_mul = 16);
   nir_pop_if(&b, NULL);
   nir_if *run_if = nir_push_if(&b, nir_ult(&b, g, total));
   nir_cf_reinsert(&body, b.cursor);
   nir_pop_if(&b, run_if);

   /* Emits, collected first: each is a branch, which splits its block. */
   struct util_dynarray emits;
   util_dynarray_init(&emits, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_per_vertex_input: {
            nir_def *slot;
            if (after_tess) {
               /* The vertex the tessellator's list names, a record of the
                * evaluation shader's. */
               nir_def *at = nir_iadd(&b, nir_imul_imm(&b, prim, vertices_in), intr->src[0].ssa);
               slot = nir_load_ssbo(&b, 1, 32, driver_buffer(&b, WVK_DRIVER_SLOT_INDEX), nir_ishl_imm(&b, at, 2),
                                    .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
            } else {
               slot = nir_iadd(&b, vertex_base, primitive_vertex(&b, topology, prim, intr->src[0].ssa, prims));
            }
            record_access(&b, intr, WVK_DRIVER_SLOT_CAPTURE, nir_imul_imm(&b, slot, in_stride), false);
            break;
         }
         case nir_intrinsic_load_input:
            nir_def_replace(&intr->def, nir_imm_zero(&b, intr->def.num_components, intr->def.bit_size));
            break;
         case nir_intrinsic_load_primitive_id:
            nir_def_replace(&intr->def, prim);
            break;
         case nir_intrinsic_load_invocation_id:
            nir_def_replace(&intr->def, invocation);
            break;
         case nir_intrinsic_store_output:
         case nir_intrinsic_load_output: {
            /* The record it keeps, as words. */
            const int slot = record_slot(nir_intrinsic_io_semantics(intr).location);
            const bool store = intr->intrinsic == nir_intrinsic_store_output;
            if (slot < 0) {
               if (store)
                  nir_instr_remove(instr);
               else
                  nir_def_replace(&intr->def, nir_imm_zero(&b, intr->def.num_components, intr->def.bit_size));
               break;
            }
            nir_deref_instr *at = nir_build_deref_array(
               &b, nir_build_deref_var(&b, record), nir_iadd_imm(&b, nir_get_io_offset_src(intr)->ssa, slot));
            const unsigned component = nir_intrinsic_component(intr);
            if (store) {
               nir_def *value = intr->src[0].ssa;
               if (value->bit_size == 16)
                  value = nir_alu_type_get_base_type(nir_intrinsic_src_type(intr)) == nir_type_float
                             ? nir_f2f32(&b, value) : nir_u2u32(&b, value);
               nir_def *words[4];
               nir_def *old = nir_load_deref(&b, at);
               const unsigned mask = nir_intrinsic_write_mask(intr);
               for (unsigned c = 0; c < 4; ++c) {
                  const int k = (int)c - (int)component;
                  words[c] = k >= 0 && k < (int)value->num_components && (mask & BITFIELD_BIT(k))
                                ? nir_channel(&b, value, k) : nir_channel(&b, old, c);
               }
               nir_store_deref(&b, at, nir_vec(&b, words, 4), 0xf);
               nir_instr_remove(instr);
            } else {
               nir_def *v = nir_channels(&b, nir_load_deref(&b, at),
                                         BITFIELD_RANGE(component, intr->def.num_components));
               if (intr->def.bit_size == 16)
                  v = nir_u2u16(&b, v);
               nir_def_replace(&intr->def, v);
            }
            break;
         }
         case nir_intrinsic_emit_vertex_with_counter:
            if (nir_intrinsic_stream_id(intr) < streams)
               util_dynarray_append(&emits, intr);
            else
               nir_instr_remove(instr);
            break;
         case nir_intrinsic_end_primitive_with_counter:
         case nir_intrinsic_set_vertex_and_primitive_count:
            nir_instr_remove(instr);
            break;
         default:
            break;
         }
      }
   }

   util_dynarray_foreach(&emits, nir_intrinsic_instr *, e) {
      nir_intrinsic_instr *intr = *e;
      const unsigned stream = nir_intrinsic_stream_id(intr);
      b.cursor = nir_before_instr(&intr->instr);
      nir_def *emitted = intr->src[0].ssa, *in_primitive = intr->src[1].ssa;
      nir_push_if(&b, nir_ult_imm(&b, emitted, max_vertices));
      /* The vertex: its record. */
      nir_def *out = nir_iadd(&b, first_out[stream], emitted);
      nir_def *base = nir_imul_imm(&b, out, out_stride);
      for (unsigned i = 0; i < slots; ++i) {
         nir_def *words = nir_load_deref(&b, nir_build_deref_array_imm(&b, nir_build_deref_var(&b, record), i));
         nir_store_ssbo(&b, words, driver_buffer(&b, WVK_DRIVER_SLOT_GS_VERTICES), nir_iadd_imm(&b, base, i * 16),
                        .write_mask = 0xf, .align_mul = 16);
      }
      /* And what it completes: a triangle of a strip, as the strip has it
       * -- i, i + 1, i + 2, or where odd i, i + 2, i + 1 -- or a segment, or
       * the point. */
      const unsigned per = output == MESA_PRIM_TRIANGLE_STRIP ? 3 : output == MESA_PRIM_LINE_STRIP ? 2 : 1;
      nir_def *n = nir_load_var(&b, made[stream]);
      nir_push_if(&b, nir_uge_imm(&b, in_primitive, per - 1));
      nir_def *at = nir_ishl_imm(&b, nir_iadd(&b, first_index[stream], nir_imul_imm(&b, n, per)), 2);
      nir_def *ids[3];
      if (per == 3) {
         nir_def *odd = nir_ine_imm(&b, nir_iand_imm(&b, in_primitive, 1), 0);
         nir_def *a = nir_iadd_imm(&b, out, -2), *m = nir_iadd_imm(&b, out, -1);
         ids[0] = a;
         ids[1] = nir_bcsel(&b, odd, out, m);
         ids[2] = nir_bcsel(&b, odd, m, out);
         /* Of the last vertex's flat varyings: from the vertex just
          * emitted, the triangle's last, round the same way. */
         if (provoking_last) {
            ids[0] = out;
            ids[1] = nir_bcsel(&b, odd, m, a);
            ids[2] = nir_bcsel(&b, odd, a, m);
         }
      } else if (per == 2) {
         ids[0] = nir_iadd_imm(&b, out, -1);
         ids[1] = out;
         if (provoking_last) {
            ids[0] = out;
            ids[1] = nir_iadd_imm(&b, out, -1);
         }
      } else {
         ids[0] = out;
      }
      /* Not where its every vertex has a cull distance of one plane
       * negative: culled, as WebGPU has no cull distances. */
      nir_def *culled = nir_imm_false(&b);
      for (unsigned c = 0; c < culls; ++c) {
         nir_def *all = nir_imm_true(&b);
         for (unsigned v = 0; v < per; ++v) {
            nir_def *d = nir_load_ssbo(&b, 1, 32, driver_buffer(&b, WVK_DRIVER_SLOT_GS_VERTICES),
                                       nir_iadd_imm(&b, nir_imul_imm(&b, ids[v], out_stride),
                                                    (record_slot(VARYING_SLOT_CLIP_DIST0) + (first_cull + c) / 4) * 16 +
                                                       ((first_cull + c) % 4) * 4),
                                       .align_mul = 4);
            all = nir_iand(&b, all, nir_flt_imm(&b, d, 0.0));
         }
         culled = nir_ior(&b, culled, all);
      }
      nir_push_if(&b, nir_inot(&b, culled));
      nir_store_ssbo(&b, nir_vec(&b, ids, per), driver_buffer(&b, WVK_DRIVER_SLOT_GS_INDICES), at,
                     .write_mask = BITFIELD_MASK(per), .align_mul = 4);
      nir_store_var(&b, made[stream], nir_iadd_imm(&b, n, 1), 0x1);
      nir_pop_if(&b, NULL);
      nir_pop_if(&b, NULL);
      nir_pop_if(&b, NULL);
      nir_instr_remove(&intr->instr);
   }
   util_dynarray_fini(&emits);
   /* Its primitives, for the pipeline statistics query: what it made, and
    * what clipping has of them, the rasterized stream's; and for the
    * primitives generated queries, each stream's. */
   if (stats) {
      b.cursor = nir_after_impl(impl);
      nir_def *all = nir_imm_int(&b, 0);
      for (unsigned st = 0; st < streams; ++st) {
         nir_def *n = nir_load_var(&b, made[st]);
         all = nir_iadd(&b, all, n);
         unsigned counters[3], count = 0;
         counters[count++] = WVK_COUNTER_PRIMITIVES(st);
         if (st == raster_stream) {
            counters[count++] = util_logbase2(VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT);
            counters[count++] = util_logbase2(VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT);
         }
         nir_push_if(&b, nir_ine_imm(&b, n, 0));
         for (unsigned i = 0; i < count; ++i)
            nir_ssbo_atomic(&b, 32, driver_buffer(&b, WVK_DRIVER_SLOT_STATS), nir_imm_int(&b, counters[i] * 4), n,
                            .atomic_op = nir_atomic_op_iadd);
         nir_pop_if(&b, NULL);
      }
      nir_push_if(&b, nir_ine_imm(&b, all, 0));
      nir_ssbo_atomic(&b, 32, driver_buffer(&b, WVK_DRIVER_SLOT_STATS),
                      nir_imm_int(&b, util_logbase2(VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT) * 4),
                      all, .atomic_op = nir_atomic_op_iadd);
      nir_pop_if(&b, NULL);
   }
   nir_progress(true, impl, nir_metadata_none);
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_shader_in | nir_var_shader_out)
      exec_node_remove(&var->node);
   NIR_PASS(_, nir, nir_opt_dce);
}

/* The index of an invocation of compute run as a stage, rows of `width`
 * by as many workgroups as fit (wvk_cmd_draw.c's dispatch_graphics). */
static nir_def *
compute_slot(nir_builder *b, unsigned width)
{
   nir_def *id = nir_load_global_invocation_id(b, 32);
   nir_def *groups = nir_load_num_workgroups(b);
   return nir_iadd(b, nir_channel(b, id, 0),
                   nir_imul(b, nir_channel(b, id, 1), nir_imul_imm(b, nir_channel(b, groups, 0), width)));
}

/* A tessellation control shader as compute (key.tess 1): a workgroup a
 * patch of each instance, an invocation a vertex of its output patch, so
 * that barrier() is the workgroup's and gl_InvocationID the invocation's
 * index in it. Its inputs are the vertex shader's records (key.runs 3) of
 * the patch's vertices; its outputs a record a vertex and a record a patch
 * (levels, then patch varyings), in buffers that barriers make the
 * workgroup's, which the tessellator and evaluation shader read after. */
static void
lower_tcs(nir_shader *nir, struct wvk_shader *shader, unsigned vs_vars)
{
   const unsigned vertices_out = MAX2(nir->info.tess.tcs_vertices_out, 1);
   shader->record_vars = record_vars(nir, nir_var_shader_out);
   shader->patch_vars = patch_vars(nir, nir_var_shader_out);
   const uint32_t in_stride = wvk_record_stride(vs_vars);
   const uint32_t out_stride = wvk_record_stride(shader->record_vars);
   const uint32_t patch_stride = wvk_patch_stride(shader->patch_vars);

   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out)
      var->data.driver_location = var->data.location;
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out, type_size_vec4,
            nir_lower_io_lower_64bit_to_32);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *wg = nir_load_workgroup_id(&b);
   nir_def *groups = nir_load_num_workgroups(&b);
   nir_def *patch_global = nir_iadd(&b, nir_channel(&b, wg, 0), nir_imul(&b, nir_channel(&b, wg, 1),
                                                                          nir_channel(&b, groups, 0)));
   nir_def *invocation = nir_load_local_invocation_index(&b);
   nir_def *tess = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH), nir_imm_int(&b, WVK_DRIVER_TESS),
                                 .align_mul = 16, .access = ACCESS_NON_WRITEABLE);
   nir_def *count = nir_channel(&b, tess, 0);
   nir_def *per_instance = nir_umax(&b, nir_channel(&b, tess, 1), nir_imm_int(&b, 1));
   nir_def *vertices_in = nir_channel(&b, tess, 2);
   nir_def *instance = nir_udiv(&b, patch_global, per_instance);
   nir_def *patch = nir_umod(&b, patch_global, per_instance);
   nir_def *first_in = nir_iadd(&b, nir_imul(&b, instance, count), nir_imul(&b, patch, vertices_in));
   nir_def *patch_base = nir_imul_imm(&b, patch_global, patch_stride);
   nir_if *run_if = nir_push_if(&b, nir_ult(&b, patch_global, nir_channel(&b, tess, 3)));
   nir_cf_reinsert(&body, b.cursor);
   nir_pop_if(&b, run_if);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_per_vertex_input: {
            nir_def *base = nir_imul_imm(&b, nir_iadd(&b, first_in, intr->src[0].ssa), in_stride);
            record_access(&b, intr, WVK_DRIVER_SLOT_CAPTURE, base, false);
            break;
         }
         case nir_intrinsic_load_per_vertex_output:
         case nir_intrinsic_store_per_vertex_output: {
            nir_def *vertex = intr->src[intr->intrinsic == nir_intrinsic_store_per_vertex_output ? 1 : 0].ssa;
            nir_def *base = nir_imul_imm(&b, nir_iadd(&b, nir_imul_imm(&b, patch_global, vertices_out), vertex),
                                         out_stride);
            record_access(&b, intr, WVK_DRIVER_SLOT_TESS_VERTICES, base, false);
            break;
         }
         case nir_intrinsic_load_output:
         case nir_intrinsic_store_output:
            record_access(&b, intr, WVK_DRIVER_SLOT_TESS_PATCHES, patch_base, true);
            break;
         case nir_intrinsic_load_input:
            nir_def_replace(&intr->def, nir_imm_zero(&b, intr->def.num_components, intr->def.bit_size));
            break;
         case nir_intrinsic_load_patch_vertices_in:
            nir_def_replace(&intr->def, vertices_in);
            break;
         case nir_intrinsic_load_primitive_id:
            nir_def_replace(&intr->def, patch);
            break;
         case nir_intrinsic_load_invocation_id:
            nir_def_replace(&intr->def, invocation);
            break;
         case nir_intrinsic_barrier:
            /* What it orders of outputs is storage now. */
            if (nir_intrinsic_memory_modes(intr) & (nir_var_shader_out | nir_var_shader_in))
               nir_intrinsic_set_memory_modes(intr, (nir_intrinsic_memory_modes(intr) &
                                                     ~(nir_var_shader_out | nir_var_shader_in)) |
                                                       nir_var_mem_ssbo);
            break;
         default:
            break;
         }
      }
   }
   nir_progress(true, impl, nir_metadata_none);
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_shader_in | nir_var_shader_out)
      exec_node_remove(&var->node);
   NIR_PASS(_, nir, nir_opt_dce);
}

/* A tessellation evaluation shader as compute (key.tess 1): an invocation a
 * domain point the tessellator made, as many as it counted (an indirect
 * dispatch). The point says its patch; its inputs are the control shader's
 * records of the patch's vertices and of the patch; its outputs a record,
 * at the point's vertex of the draw after, slot 0 being the one unwritten
 * indices name. */
static void
lower_tes(nir_shader *nir, struct wvk_shader *shader, unsigned tcs_vars, unsigned tcs_patch_vars,
          unsigned patch_vertices)
{
   const unsigned domain = nir->info.tess._primitive_mode;
   shader->record_vars = record_vars(nir, nir_var_shader_out);
   const uint32_t in_stride = wvk_record_stride(tcs_vars);
   const uint32_t patch_stride = wvk_patch_stride(tcs_patch_vars);
   const uint32_t out_stride = wvk_record_stride(shader->record_vars);

   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_in | nir_var_shader_out, UINT32_MAX);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out)
      var->data.driver_location = var->data.location;
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out, type_size_vec4,
            nir_lower_io_lower_64bit_to_32);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *k = compute_slot(&b, 64);
   nir_def *tess = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH), nir_imm_int(&b, WVK_DRIVER_TESS),
                                 .align_mul = 16, .access = ACCESS_NON_WRITEABLE);
   nir_def *per_instance = nir_umax(&b, nir_channel(&b, tess, 1), nir_imm_int(&b, 1));
   nir_def *vertices = nir_imm_int(&b, MAX2(patch_vertices, 1));
   /* The points the tessellator made, after each patch's counts and
    * where they go. */
   nir_def *total = nir_load_ssbo(&b, 1, 32, driver_buffer(&b, WVK_DRIVER_SLOT_TESS_COUNTS),
                                  nir_ishl_imm(&b, nir_ishl_imm(&b, nir_channel(&b, tess, 3), 2), 2), .align_mul = 4,
                                  .access = ACCESS_NON_WRITEABLE);
   /* The vertex unwritten indices name: out of sight. */
   nir_push_if(&b, nir_ieq_imm(&b, k, 0));
   nir_store_ssbo(&b, nir_imm_vec4(&b, 2.0f, 2.0f, 2.0f, 1.0f), driver_buffer(&b, WVK_DRIVER_SLOT_GS_VERTICES),
                  nir_imm_int(&b, 0), .write_mask = 0xf, .align_mul = 16);
   nir_pop_if(&b, NULL);
   nir_def *point = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_TESS_POINTS), nir_ishl_imm(&b, k, 4),
                                  .align_mul = 16, .access = ACCESS_NON_WRITEABLE);
   nir_def *patch_global = nir_channel(&b, point, 3);
   nir_def *u = nir_channel(&b, point, 0), *v = nir_channel(&b, point, 1);
   nir_def *w = domain == TESS_PRIMITIVE_TRIANGLES ? nir_fsub(&b, nir_fsub_imm(&b, 1.0f, u), v)
                                                   : nir_imm_float(&b, 0.0f);
   nir_def *coord = nir_vec3(&b, u, v, w);
   nir_def *first_in = nir_imul(&b, patch_global, vertices);
   nir_def *patch_base = nir_imul_imm(&b, patch_global, patch_stride);
   nir_def *out_base = nir_imul_imm(&b, nir_iadd_imm(&b, k, 1), out_stride);
   nir_if *run_if = nir_push_if(&b, nir_ult(&b, k, total));
   nir_cf_reinsert(&body, b.cursor);
   nir_pop_if(&b, run_if);

   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_per_vertex_input: {
            nir_def *base = nir_imul_imm(&b, nir_iadd(&b, first_in, intr->src[0].ssa), in_stride);
            record_access(&b, intr, WVK_DRIVER_SLOT_TESS_VERTICES, base, false);
            break;
         }
         case nir_intrinsic_load_input:
            record_access(&b, intr, WVK_DRIVER_SLOT_TESS_PATCHES, patch_base, true);
            break;
         case nir_intrinsic_load_output:
         case nir_intrinsic_store_output:
            record_access(&b, intr, WVK_DRIVER_SLOT_GS_VERTICES, out_base, false);
            break;
         case nir_intrinsic_load_tess_coord:
            nir_def_replace(&intr->def, nir_trim_vector(&b, coord, intr->def.num_components));
            break;
         case nir_intrinsic_load_tess_coord_xy:
            nir_def_replace(&intr->def, nir_vec2(&b, u, v));
            break;
         case nir_intrinsic_load_tess_level_outer:
         case nir_intrinsic_load_tess_level_inner: {
            const unsigned slot = intr->intrinsic == nir_intrinsic_load_tess_level_outer ? 0 : 1;
            nir_def *levels = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_TESS_PATCHES),
                                            nir_iadd_imm(&b, patch_base, slot * 16), .align_mul = 16,
                                            .access = ACCESS_NON_WRITEABLE);
            nir_def_replace(&intr->def, nir_trim_vector(&b, levels, intr->def.num_components));
            break;
         }
         case nir_intrinsic_load_patch_vertices_in:
            nir_def_replace(&intr->def, vertices);
            break;
         case nir_intrinsic_load_primitive_id:
            nir_def_replace(&intr->def, nir_umod(&b, patch_global, per_instance));
            break;
         default:
            break;
         }
      }
   }
   nir_progress(true, impl, nir_metadata_none);
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_shader_in | nir_var_shader_out)
      exec_node_remove(&var->node);
   NIR_PASS(_, nir, nir_opt_dce);
}

/* A record's slots of a variable, from binding `binding` at `base`, into
 * the variable (geometry_vertex_shader): as nir_lower_io lays it out. */
static void
load_record_into(nir_builder *b, nir_deref_instr *deref, const struct glsl_type *type, unsigned location,
                 unsigned frac, bool compact, uint32_t binding, nir_def *base)
{
   if (compact) {
      for (unsigned e = 0; e < glsl_get_length(type); ++e) {
         const unsigned total = frac + e;
         const int slot = record_slot(location + total / 4);
         if (slot < 0)
            continue;
         nir_def *word = nir_load_ssbo(b, 1, 32, driver_buffer(b, binding),
                                       nir_iadd_imm(b, base, slot * 16 + (total % 4) * 4), .align_mul = 4,
                                       .access = ACCESS_NON_WRITEABLE);
         nir_store_deref(b, nir_build_deref_array_imm(b, deref, e), word, 0x1);
      }
      return;
   }
   if (glsl_type_is_struct_or_ifc(type)) {
      unsigned at = location;
      for (unsigned f = 0; f < glsl_get_length(type); ++f) {
         const struct glsl_type *field = glsl_get_struct_field(type, f);
         load_record_into(b, nir_build_deref_struct(b, deref, f), field, at, 0, false, binding, base);
         at += glsl_count_vec4_slots(field, false, true);
      }
      return;
   }
   if (glsl_type_is_array(type) || glsl_type_is_matrix(type)) {
      const struct glsl_type *element = glsl_type_is_array(type) ? glsl_get_array_element(type)
                                                                 : glsl_get_column_type(type);
      const unsigned n = glsl_type_is_array(type) ? glsl_get_length(type) : glsl_get_matrix_columns(type);
      const unsigned per = glsl_count_vec4_slots(element, false, true);
      for (unsigned e = 0; e < n; ++e)
         load_record_into(b, nir_build_deref_array_imm(b, deref, e), element, location + e * per, frac, false,
                          binding, base);
      return;
   }
   if (glsl_type_is_vector_or_scalar(type) && glsl_get_bit_size(type) == 64) {
      /* Two words a component, two components a slot. */
      nir_def *out[4];
      const unsigned n = glsl_get_vector_elements(type);
      for (unsigned c = 0; c < n; ++c) {
         const unsigned at = frac / 2 + c;
         const int slot = record_slot(location + at / 2);
         out[c] = slot < 0 ? nir_imm_int64(b, 0)
                           : nir_pack_64_2x32(b, nir_load_ssbo(b, 2, 32, driver_buffer(b, binding),
                                                               nir_iadd_imm(b, base, slot * 16 + (at % 2) * 8),
                                                               .align_mul = 4, .access = ACCESS_NON_WRITEABLE));
      }
      nir_store_deref(b, deref, nir_vec(b, out, n), BITFIELD_MASK(n));
      return;
   }
   const int slot = record_slot(location);
   if (slot < 0 || !glsl_type_is_vector_or_scalar(type))
      return;
   const unsigned n = glsl_get_vector_elements(type);
   nir_def *value = nir_load_ssbo(b, n, 32, driver_buffer(b, binding), nir_iadd_imm(b, base, slot * 16 + frac * 4),
                                  .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
   if (glsl_get_bit_size(type) == 16)
      value = glsl_get_base_type(type) == GLSL_TYPE_FLOAT16 ? nir_f2f16(b, value) : nir_u2u16(b, value);
   nir_store_deref(b, deref, value, BITFIELD_MASK(n));
}

static bool split_io_structs(nir_shader *nir);

/* The vertex shader of the draw after a geometry shader run as compute
 * (key.geometry 2), made of the geometry shader's source: its outputs,
 * each of the record lower_geometry wrote of the vertex vertex_index names. */
static void
geometry_vertex_shader(nir_shader *nir, unsigned primitive_id, unsigned raster_stream)
{
   NIR_PASS(_, nir, split_io_structs);
   /* Of the rasterized stream's varyings, where the streams' share
    * locations: the others' are transform feedback's alone. */
   nir_foreach_shader_out_variable_safe(var, nir) {
      if (var->data.location >= VARYING_SLOT_VAR0 && !(var->data.stream & NIR_STREAM_PACKED) &&
          var->data.stream != raster_stream)
         exec_node_remove(&var->node);
   }
   const uint32_t stride = wvk_record_stride(record_vars(nir, nir_var_shader_out));
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_shader_in | nir_var_system_value)
      exec_node_remove(&var->node);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list list;
   nir_cf_extract(&list, nir_before_impl(impl), nir_after_impl(impl));
   nir_cf_delete(&list);
   exec_list_make_empty(&impl->locals);
   memset(&nir->info.gs, 0, sizeof(nir->info.gs));
   memset(&nir->info.tess, 0, sizeof(nir->info.tess));
   nir->info.stage = MESA_SHADER_VERTEX;
   /* A position, which the record has even of a shader that writes none:
    * the vertex unwritten indices name is out of sight. */
   if (!nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_POS)) {
      nir_variable *pos = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
      pos->data.location = VARYING_SLOT_POS;
   }
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *base = nir_imul_imm(&b, nir_load_vertex_id(&b), stride);
   nir_foreach_shader_out_variable(var, nir)
      load_record_into(&b, nir_build_deref_var(&b, var), var->type, var->data.location, var->data.location_frac,
                       var->data.compact, WVK_DRIVER_SLOT_GS_VERTICES, base);
   /* gl_PrimitiveID, which a vertex shader has no builtin of: a flat
    * varying, where the fragment shader reads it (lower_fs_primitive_id). */
   nir_variable *id = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_PRIMITIVE_ID);
   if (id && primitive_id) {
      id->data.location = VARYING_SLOT_VAR0 + primitive_id - 1;
      id->data.interpolation = INTERP_MODE_FLAT;
   } else if (id) {
      id->data.mode = nir_var_shader_temp;
      nir_fixup_deref_modes(nir);
   }
   nir_progress(true, impl, nir_metadata_none);
}

/* Whether an instruction writes storage: a buffer, or an image. */
static bool
writes_storage_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   bool *writes = data;
   switch (intr->intrinsic) {
   case nir_intrinsic_store_deref:
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
      *writes |= nir_deref_mode_is(nir_src_as_deref(intr->src[0]), nir_var_mem_ssbo);
      break;
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
      *writes = true;
      break;
   default:
      break;
   }
   return false;
}

/* A vertex shader's storage writes gone, for the draw: WebGPU's vertex
 * shaders write no storage, and its compute shader has done them before
 * (run_vertices). An atomic gives the value as it is. Those to buffers
 * first, and to images (`data` non-NULL) once the images the shader
 * writes have their formats, which they then have as its compute
 * shader's do (format_storage_images). */
static bool
strip_storage_writes_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const bool images = data != NULL;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_store_deref:
      if (images || !nir_deref_mode_is(nir_src_as_deref(intr->src[0]), nir_var_mem_ssbo))
         return false;
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap:
      if (images || !nir_deref_mode_is(nir_src_as_deref(intr->src[0]), nir_var_mem_ssbo))
         return false;
      nir_def_replace(&intr->def, nir_load_deref(b, nir_src_as_deref(intr->src[0])));
      return true;
   case nir_intrinsic_image_deref_store:
      if (!images)
         return false;
      nir_instr_remove(&intr->instr);
      return true;
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
      if (!images)
         return false;
      nir_def_replace(&intr->def, nir_imm_zero(b, intr->def.num_components, intr->def.bit_size));
      return true;
   default:
      return false;
   }
}

/* A vertex shader as the compute shader doing its storage writes (key.runs;
 * vertexPipelineStoresAndAtomics): an invocation a vertex of each instance
 * of the draw the driver's data has (WVK_DRIVER_VERTEX_RUN) -- the vertex
 * an index of the index buffer for an indexed draw, but where it restarts
 * primitives -- with its attributes all pulled there, and its outputs gone;
 * or with `capture`, their clip positions kept, 16 bytes a vertex at
 * WVK_DRIVER_SLOT_CAPTURE. Compute's stage after its buffers are words
 * (compile_one). */
static void
run_vertices(nir_shader *nir, struct wvk_shader *shader, unsigned runs)
{
   const bool capture = runs == 2, records = runs == 3;
   NIR_PASS(_, nir, nir_lower_returns);
   nir_variable *position = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_POS);
   if (records) {
      /* Its outputs a record a vertex: stores of slots (write_records). */
      shader->record_vars = record_vars(nir, nir_var_shader_out);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_out, UINT32_MAX);
      nir_foreach_shader_out_variable(var, nir)
         var->data.driver_location = var->data.location;
      NIR_PASS(_, nir, nir_lower_io, nir_var_shader_out, type_size_vec4, nir_lower_io_lower_64bit_to_32);
   } else {
      nir_foreach_shader_out_variable(var, nir)
         var->data.mode = nir_var_shader_temp;
      nir_fixup_deref_modes(nir);
   }
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder b = nir_builder_at(nir_before_impl(impl));

   /* Which vertex of which instance: rows of 64 * groups wide. */
   nir_def *id = nir_load_global_invocation_id(&b, 32);
   nir_def *groups = nir_load_num_workgroups(&b);
   nir_def *slot = nir_iadd(&b, nir_channel(&b, id, 0),
                            nir_imul(&b, nir_channel(&b, id, 1), nir_imul_imm(&b, nir_channel(&b, groups, 0), 64)));
   nir_def *run = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                nir_imm_int(&b, WVK_DRIVER_VERTEX_RUN), .align_mul = 16,
                                .access = ACCESS_NON_WRITEABLE);
   nir_def *count = nir_umax(&b, nir_channel(&b, run, 0), nir_imm_int(&b, 1));
   nir_def *at = nir_iadd(&b, nir_channel(&b, run, 2), nir_umod(&b, slot, count));
   nir_def *instance = nir_iadd(&b, nir_load_base_instance(&b), nir_udiv(&b, slot, count));
   nir_def *index = nir_load_ssbo(&b, 2, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                  nir_imm_int(&b, WVK_DRIVER_POINT_INDEX), .align_mul = 8,
                                  .access = ACCESS_NON_WRITEABLE);
   nir_def *start = nir_channel(&b, index, 0), *size = nir_channel(&b, index, 1);
   nir_def *no = nir_imm_false(&b);
   nir_push_if(&b, nir_ine_imm(&b, size, 0));
   nir_def *byte = nir_iadd(&b, start, nir_imul(&b, at, size));
   nir_def *word = nir_load_ssbo(&b, 1, 32, driver_buffer(&b, WVK_DRIVER_SLOT_INDEX), nir_iand_imm(&b, byte, ~3),
                                 .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
   nir_def *bits = nir_ishl_imm(&b, size, 3);
   nir_def *part = nir_ubitfield_extract(&b, word, nir_ishl_imm(&b, nir_iand_imm(&b, byte, 3), 3), bits);
   nir_def *value = nir_bcsel(&b, nir_ieq_imm(&b, size, 4), word, part);
   nir_def *all_ones = nir_bcsel(&b, nir_ieq_imm(&b, size, 4), nir_imm_int(&b, -1),
                                 nir_iadd_imm(&b, nir_ishl(&b, nir_imm_int(&b, 1), bits), -1));
   nir_def *restarts = nir_iand(&b, nir_ine_imm(&b, nir_channel(&b, run, 3), 0), nir_ieq(&b, value, all_ones));
   nir_def *indexed = nir_iadd(&b, value, nir_load_base_vertex(&b));
   nir_push_else(&b, NULL);
   nir_pop_if(&b, NULL);
   nir_def *vertex = nir_if_phi(&b, indexed, at);
   nir_def *restart = nir_if_phi(&b, restarts, no);
   nir_def *skip = nir_ior(&b, nir_uge(&b, slot, nir_channel(&b, run, 1)), restart);
   nir_variable *vertex_var = nir_local_variable_create(impl, glsl_uint_type(), WVK_POINT_VERTEX);
   nir_variable *instance_var = nir_local_variable_create(impl, glsl_uint_type(), WVK_RUN_INSTANCE);
   nir_store_var(&b, vertex_var, vertex, 0x1);
   nir_store_var(&b, instance_var, instance, 0x1);
   nir_if *run_if = nir_push_if(&b, nir_inot(&b, skip));
   nir_cf_reinsert(&body, b.cursor);
   /* Or only where it is, for wide lines to read (expand_lines). */
   if (capture) {
      b.cursor = nir_after_cf_list(&run_if->then_list);
      nir_def *pos = position ? nir_load_var(&b, position) : nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f);
      nir_store_ssbo(&b, pos, driver_buffer(&b, WVK_DRIVER_SLOT_CAPTURE), nir_ishl_imm(&b, slot, 4),
                     .write_mask = 0xf, .align_mul = 16);
   }
   nir_pop_if(&b, run_if);

   /* What the shader says of its vertex and instance is these; and its
    * outputs, of a record at `slot`. */
   const uint32_t stride = wvk_record_stride(shader->record_vars);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_load_vertex_id) {
            nir_def_replace(&intr->def, vertex);
         } else if (intr->intrinsic == nir_intrinsic_load_instance_id) {
            nir_def_replace(&intr->def, instance);
         } else if (records && (intr->intrinsic == nir_intrinsic_store_output ||
                                intr->intrinsic == nir_intrinsic_load_output)) {
            b.cursor = nir_before_instr(instr);
            record_access(&b, intr, WVK_DRIVER_SLOT_CAPTURE, nir_imul_imm(&b, slot, stride), false);
         }
      }
   }
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_temp, NULL);
}

/* What of a fragment shader a point's quad does not give it: gl_PointCoord,
 * which the vertex shader sends (expand_points), and facing front. */
static bool
lower_point_fs_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_variable *coord = data;
   b->cursor = nir_before_instr(&intr->instr);
   switch (intr->intrinsic) {
   case nir_intrinsic_load_front_face:
      nir_def_replace(&intr->def, nir_imm_true(b));
      return true;
   case nir_intrinsic_load_point_coord:
   case nir_intrinsic_load_point_coord_maybe_flipped:
      nir_def_replace(&intr->def, coord ? nir_trim_vector(b, nir_load_var(b, coord), 2)
                                        : nir_imm_vec2(b, 0.5f, 0.5f));
      return true;
   default:
      return false;
   }
}

static bool
lower_point_fs(nir_shader *nir, unsigned point_coord)
{
   nir_variable *coord = NULL;
   if (point_coord) {
      coord = nir_variable_create(nir, nir_var_shader_in, glsl_vec4_type(), "wvk_point_coord");
      coord->data.location = VARYING_SLOT_VAR0 + point_coord - 1;
      coord->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
   }
   return nir_shader_intrinsics_pass(nir, lower_point_fs_instr, nir_metadata_control_flow, coord);
}

/* A line's quad's fragments where they are on the line (expand_lines):
 * gone where its stipple's bit for where along the line they are is clear
 * -- the bit of that in pixels over the factor -- and of a smooth line, of
 * alpha by how much of their pixel is within half the width of it. */
static bool
lower_line_fs(nir_shader *nir, unsigned line_coord)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_variable *coord = nir_variable_create(nir, nir_var_shader_in, glsl_vec4_type(), "wvk_line_coord");
   coord->data.location = VARYING_SLOT_VAR0 + line_coord - 1;
   coord->data.interpolation = INTERP_MODE_NOPERSPECTIVE;
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *at = nir_load_var(&b, coord);
   nir_def *line = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                 nir_imm_int(&b, WVK_DRIVER_LINE), .align_mul = 16, .access = ACCESS_NON_WRITEABLE);
   nir_def *stipple = nir_channel(&b, line, 3);
   nir_def *factor = nir_umax(&b, nir_ushr_imm(&b, stipple, 16), nir_imm_int(&b, 1));
   nir_def *place = nir_f2u32(&b, nir_fmax(&b, nir_channel(&b, at, 1), nir_imm_float(&b, 0.0f)));
   nir_def *bit = nir_iand_imm(&b, nir_udiv(&b, place, factor), 15);
   nir_def *clear = nir_ieq_imm(&b, nir_iand_imm(&b, nir_ushr(&b, stipple, bit), 1), 0);
   nir_terminate_if(&b, nir_iand(&b, nir_ine_imm(&b, stipple, 0), clear));
   nir_def *half = nir_fmul_imm(&b, nir_channel(&b, line, 0), 0.5f);
   nir_def *covered = nir_fsat(&b, nir_fsub(&b, nir_fadd_imm(&b, half, 0.5f), nir_fabs(&b, nir_channel(&b, at, 0))));
   nir_def *coverage = nir_bcsel(&b, nir_ieq_imm(&b, nir_channel(&b, line, 2), 2), covered, nir_imm_float(&b, 1.0f));
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_deref)
            continue;
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         nir_variable *var = nir_deref_instr_get_variable(deref);
         if (!var || var->data.mode != nir_var_shader_out || var->data.location < FRAG_RESULT_DATA0 ||
             !glsl_type_is_vector(deref->type) || glsl_get_base_type(deref->type) != GLSL_TYPE_FLOAT ||
             glsl_get_vector_elements(deref->type) < 4 || !(nir_intrinsic_write_mask(intr) & 0x8))
            continue;
         b.cursor = nir_before_instr(instr);
         nir_def *value = intr->src[1].ssa;
         nir_def *alpha = nir_fmul(&b, nir_channel(&b, value, 3), coverage);
         nir_src_rewrite(&intr->src[1], nir_vector_insert_imm(&b, value, alpha, 3));
      }
   }
   return nir_progress(true, impl, nir_metadata_none);
}

/* What of a point a fragment shader reads: 1 gl_PointCoord, 2 gl_FrontFacing. */
static bool
point_state_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   uint8_t *state = data;
   if (intr->intrinsic == nir_intrinsic_load_point_coord ||
       intr->intrinsic == nir_intrinsic_load_point_coord_maybe_flipped)
      *state |= 1;
   else if (intr->intrinsic == nir_intrinsic_load_front_face)
      *state |= 2;
   else if (intr->intrinsic == nir_intrinsic_load_primitive_id)
      *state |= 4;
   return false;
}

/* gl_PrimitiveID as 0, where no geometry shader gives it. */
static bool
zero_fs_primitive_id(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_variable *var = nir_find_variable_with_location(nir, nir_var_shader_in, VARYING_SLOT_PRIMITIVE_ID);
   if (var) {
      var->data.mode = nir_var_shader_temp;
      nir_fixup_deref_modes(nir);
      nir_builder b = nir_builder_at(nir_before_impl(impl));
      nir_store_var(&b, var, nir_imm_zero(&b, glsl_get_vector_elements(var->type), 32), 0x1);
   }
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_load_primitive_id)
            continue;
         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_def_replace(&nir_instr_as_intrinsic(instr)->def, nir_imm_int(&b, 0));
      }
   }
   return nir_progress(true, impl, nir_metadata_none);
}

/* gl_PrimitiveID of a fragment shader after a geometry shader: the flat
 * varying at `location`, which the draw's vertex shader has of the
 * geometry shader's records (geometry_vertex_shader). */
static bool
lower_fs_primitive_id(nir_shader *nir, unsigned location)
{
   nir_variable *var = nir_find_variable_with_location(nir, nir_var_shader_in, VARYING_SLOT_PRIMITIVE_ID);
   if (!var) {
      var = nir_variable_create(nir, nir_var_shader_in, glsl_int_type(), "wvk_primitive_id");
      var->data.location = VARYING_SLOT_PRIMITIVE_ID;
   }
   var->data.location = VARYING_SLOT_VAR0 + location;
   var->data.interpolation = INTERP_MODE_FLAT;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_primitive_id)
            continue;
         nir_builder b = nir_builder_at(nir_before_instr(instr));
         nir_def_replace(&intr->def, nir_load_var(&b, var));
      }
   }
   return nir_progress(true, impl, nir_metadata_control_flow);
}

/* Sample shading, which WebGPU does where a fragment shader reads its
 * sample: at every sample, which is at least as many as any minimum. */
static bool
wants_sample_shading(const struct vk_graphics_pipeline_state *state)
{
   return state && state->ms && state->ms->sample_shading_enable && state->ms->min_sample_shading > 0.0f;
}

/* Where a sample is in its pixel: WebGPU's multisampling is of 4 samples,
 * at Vulkan's standard locations for 4, or of 1 at the centre. */
static nir_def *
sample_position(nir_builder *b, nir_def *id, unsigned samples)
{
   if (samples <= 1)
      return nir_imm_vec2(b, 0.5f, 0.5f);
   static const float x[4] = { 0.375f, 0.875f, 0.125f, 0.625f }, y[4] = { 0.125f, 0.375f, 0.625f, 0.875f };
   nir_def *px = nir_imm_float(b, x[3]), *py = nir_imm_float(b, y[3]);
   for (int i = 2; i >= 0; --i) {
      px = nir_bcsel(b, nir_ieq_imm(b, id, i), nir_imm_float(b, x[i]), px);
      py = nir_bcsel(b, nir_ieq_imm(b, id, i), nir_imm_float(b, y[i]), py);
   }
   return nir_vec2(b, px, py);
}

struct sample_state {
   unsigned samples;
   bool frag_coord;
};

/* gl_SamplePosition, which WGSL has not; and with sample shading
 * gl_FragCoord at the sample, which WGSL's is at the pixel's centre. */
static bool
lower_sample_pos_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct sample_state *s = data;
   if (intr->intrinsic == nir_intrinsic_load_sample_pos || intr->intrinsic == nir_intrinsic_load_sample_pos_or_center) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, sample_position(b, nir_load_sample_id(b), s->samples));
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_load_frag_coord && s->frag_coord && s->samples > 1) {
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *pos = sample_position(b, nir_load_sample_id(b), s->samples);
      nir_def *xy = nir_fadd(b, nir_ffloor(b, nir_trim_vector(b, &intr->def, 2)), pos);
      nir_def *fc = nir_vec4(b, nir_channel(b, xy, 0), nir_channel(b, xy, 1), nir_channel(b, &intr->def, 2),
                             nir_channel(b, &intr->def, 3));
      nir_def_rewrite_uses_after(&intr->def, fc);
      return true;
   }
   return false;
}

/* interpolateAtCentroid, AtSample and AtOffset, which WGSL has none of:
 * the input at the centre, moved by its derivatives to the offset --
 * exact for what interpolates linearly across the pixel, which is all
 * that does; the centroid is the centre. */
static bool
lower_interp_at_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   const struct sample_state *s = data;
   if (intr->intrinsic != nir_intrinsic_interp_deref_at_centroid &&
       intr->intrinsic != nir_intrinsic_interp_deref_at_sample &&
       intr->intrinsic != nir_intrinsic_interp_deref_at_offset)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *value = nir_load_deref(b, nir_src_as_deref(intr->src[0]));
   nir_def *offset = NULL;
   if (intr->intrinsic == nir_intrinsic_interp_deref_at_offset)
      offset = intr->src[1].ssa;
   else if (intr->intrinsic == nir_intrinsic_interp_deref_at_sample)
      offset = nir_fadd_imm(b, sample_position(b, intr->src[1].ssa, s->samples), -0.5);
   if (offset) {
      value = nir_fadd(b, value, nir_fmul(b, nir_ddx(b, value), nir_channel(b, offset, 0)));
      value = nir_fadd(b, value, nir_fmul(b, nir_ddy(b, value), nir_channel(b, offset, 1)));
   }
   nir_def_replace(&intr->def, value);
   return true;
}

/* Inputs to interpolate at each sample, or at the centroid: WebGPU wants
 * the vertex output and the fragment input to interpolate alike, and the
 * vertex shader's is at the centre. A sample's is its value moved there,
 * which shades per sample as Vulkan's Sample does; a centroid's the
 * centre. */
static bool
lower_input_sampling(nir_shader *nir, const struct sample_state *s)
{
   struct set *sampled = _mesa_pointer_set_create(NULL);
   nir_foreach_shader_in_variable(var, nir) {
      if (var->data.location < VARYING_SLOT_VAR0)
         continue;
      if (var->data.sample)
         _mesa_set_add(sampled, var);
      var->data.sample = false;
      var->data.centroid = false;
   }
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            nir_variable *var = intr->intrinsic == nir_intrinsic_load_deref ? nir_intrinsic_get_var(intr, 0) : NULL;
            if (!var || !_mesa_set_search(sampled, var))
               continue;
            b.cursor = nir_after_instr(instr);
            nir_def *offset = nir_fadd_imm(&b, sample_position(&b, nir_load_sample_id(&b), s->samples), -0.5);
            nir_def *value = nir_fadd(&b, &intr->def, nir_fmul(&b, nir_ddx(&b, &intr->def), nir_channel(&b, offset, 0)));
            value = nir_fadd(&b, value, nir_fmul(&b, nir_ddy(&b, &intr->def), nir_channel(&b, offset, 1)));
            nir_def_rewrite_uses_after(&intr->def, value);
            progress = true;
         }
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   _mesa_set_destroy(sampled, NULL);
   return progress;
}

/* The fragment shader made to run for each sample: it reads the sample,
 * in a test that never passes and that nothing can tell never does -- the
 * draw's viewport, from the driver's data, is under 16. */
static bool
shade_per_sample(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_before_impl(impl));
   nir_def *viewport = nir_load_ssbo(&b, 1, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                     nir_imm_int(&b, WVK_DRIVER_VIEWPORT), .align_mul = 4,
                                     .access = ACCESS_NON_WRITEABLE);
   nir_demote_if(&b, nir_ieq(&b, nir_load_sample_id(&b), nir_iadd_imm(&b, viewport, 0x10000)));
   BITSET_SET(nir->info.system_values_read, SYSTEM_VALUE_SAMPLE_ID);
   nir->info.fs.uses_sample_shading = true;
   /* Its inputs stay at the pixel's centre, not each sample's: WebGPU has
    * a vertex output and the fragment input it meets interpolate alike, and
    * the vertex shader is made without knowing. */
   return nir_progress(true, impl, nir_metadata_none);
}

/* The layout of Vulkan binding (set, binding), or NULL. */
static const struct wvk_descriptor_set_binding_layout *
binding_layout(const struct vk_shader_compile_info *info, uint32_t set, uint32_t binding)
{
   if (set >= info->set_layout_count || !info->set_layouts[set])
      return NULL;
   const struct wvk_descriptor_set_layout *layout =
      container_of(info->set_layouts[set], struct wvk_descriptor_set_layout, vk);
   if (binding >= layout->binding_count || layout->bindings[binding].type == VK_DESCRIPTOR_TYPE_MAX_ENUM)
      return NULL;
   return &layout->bindings[binding];
}

/*
 * Textures, samplers and storage images: one variable per element of the
 * binding, at its WebGPU binding, and a combined image sampler as two.
 */

struct split_var {
   uint32_t count;
   nir_variable **texture;   /* or the image, or the sampler of a bare one */
   nir_variable **sampler;   /* combined image samplers only */
   /* A combined image sampler's planes after the first, of a YCbCr
    * conversion (nir_vk_lower_ycbcr_tex): textures of their own, and a
    * sampler of the chroma filter for them. */
   nir_variable **planes[2];
   nir_variable **chroma;
};

struct lower_state {
   const struct vk_shader_compile_info *info;
   struct hash_table *split;   /* nir_variable * -> struct split_var * */
   bool failed;
};

static nir_variable *
new_var(nir_shader *nir, nir_variable_mode mode, const struct glsl_type *type,
        const nir_variable *from, uint32_t binding, const char *suffix, uint32_t element)
{
   char name[64];
   snprintf(name, sizeof(name), "%s%s%u", from->name ? from->name : "res", suffix, element);
   nir_variable *var = nir_variable_create(nir, mode, type, name);
   var->data.descriptor_set = from->data.descriptor_set;
   var->data.binding = binding;
   var->data.access = from->data.access;
   var->data.image.format = from->data.image.format;
   var->data.how_declared = from->data.how_declared;
   return var;
}

/* An integer cube texture as the 2D array of its faces, which is what is
 * bound for one (wvk_descriptor_set.c): WebGPU samples no integer
 * textures, and a cube's faces are fetched from by hand
 * (wvk_nir_descriptors.c). */
static const struct glsl_type *
integer_cube_as_array(const struct glsl_type *type)
{
   if (!glsl_type_is_texture(type) || glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_CUBE ||
       glsl_get_sampler_result_type(type) == GLSL_TYPE_FLOAT)
      return type;
   return glsl_texture_type(GLSL_SAMPLER_DIM_2D, true, glsl_get_sampler_result_type(type));
}

static void
split_resource_vars(nir_shader *nir, struct lower_state *state)
{
   /* The variables to split, first: the new ones join the same list. */
   struct util_dynarray originals;
   util_dynarray_init(&originals, state->split);
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform | nir_var_image)
      util_dynarray_append(&originals, var);

   util_dynarray_foreach(&originals, nir_variable *, v) {
      nir_variable *var = *v;
      const struct glsl_type *bare = glsl_without_array(var->type);
      if (!glsl_type_is_sampler(bare) && !glsl_type_is_texture(bare) && !glsl_type_is_image(bare))
         continue;
      /* The driver's own are where they go: a logic op's copies. */
      if (var->data.descriptor_set == WVK_DRIVER_GROUP && var->data.binding >= WVK_DRIVER_SLOT_BASE)
         continue;
      const struct wvk_descriptor_set_binding_layout *bl =
         binding_layout(state->info, var->data.descriptor_set, var->data.binding);
      if (!bl) {
         mesa_loge("wvk: a shader uses set %u binding %u, which its pipeline layout does not have",
                   var->data.descriptor_set, var->data.binding);
         state->failed = true;
         continue;
      }
      struct split_var *s = rzalloc(state->split, struct split_var);
      /* The layout's count: an array without a size is as big as that. */
      s->count = glsl_type_is_array(var->type) ? MAX2(bl->count, 1) : 1;
      s->texture = rzalloc_array(s, nir_variable *, s->count);
      const bool combined = glsl_type_is_sampler(bare) && !glsl_type_is_bare_sampler(bare);
      if (combined)
         s->sampler = rzalloc_array(s, nir_variable *, s->count);
      for (uint32_t e = 0; e < s->count; ++e) {
         if (combined) {
            const struct glsl_type *texture = integer_cube_as_array(
               glsl_texture_type(glsl_get_sampler_dim(bare), glsl_sampler_type_is_array(bare),
                                 glsl_get_sampler_result_type(bare)));
            s->texture[e] = new_var(nir, nir_var_uniform, texture, var, bl->slot + e, "_tex", e);
            s->sampler[e] = new_var(nir, nir_var_uniform,
                                    glsl_sampler_type_is_shadow(bare) ? glsl_bare_shadow_sampler_type()
                                                                      : glsl_bare_sampler_type(),
                                    var, bl->slot + bl->count + e, "_smp", e);
            for (uint32_t p = 1; p < MIN2(bl->planes, 3); ++p) {
               if (!s->planes[p - 1])
                  s->planes[p - 1] = rzalloc_array(s, nir_variable *, s->count);
               s->planes[p - 1][e] = new_var(nir, nir_var_uniform, texture, var, bl->slot + (1 + p) * bl->count + e,
                                             "_plane", e);
            }
            if (bl->planes > 1) {
               if (!s->chroma)
                  s->chroma = rzalloc_array(s, nir_variable *, s->count);
               s->chroma[e] = new_var(nir, nir_var_uniform, glsl_bare_sampler_type(), var,
                                      bl->slot + (1 + bl->planes) * bl->count + e, "_chroma", e);
            }
         } else if (glsl_type_is_image(bare) && (glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_SUBPASS ||
                                                 glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_SUBPASS_MS)) {
            /* An input attachment: fetched, as a texture, at the layer
             * or view drawn. A multisampled one is one layer. */
            const bool ms = glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_SUBPASS_MS;
            const struct glsl_type *texture =
               glsl_texture_type(ms ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D, !ms,
                                 glsl_get_sampler_result_type(bare));
            s->texture[e] = new_var(nir, nir_var_uniform, texture, var, bl->slot + e, "_ia", e);
         } else if (glsl_type_is_image(bare) && glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_CUBE) {
            /* A cube image is a 2D array of its faces, as Vulkan addresses
             * it, and as WebGPU can bind it: no storage cube textures. */
            const struct glsl_type *array =
               glsl_image_type(GLSL_SAMPLER_DIM_2D, true, glsl_get_sampler_result_type(bare));
            s->texture[e] = new_var(nir, var->data.mode, array, var, bl->slot + e, "_", e);
         } else {
            s->texture[e] = new_var(nir, var->data.mode, integer_cube_as_array(bare), var, bl->slot + e, "_", e);
         }
      }
      _mesa_hash_table_insert(state->split, var, s);
   }
}

/* Where a deref of a split resource points: its elements, the constant
 * part of which one, and the rest, NULL if the index is constant. Arrays
 * of arrays are flattened, as the layout's elements are. */
struct element_ref {
   struct split_var *s;
   uint32_t base;
   nir_def *index;
   /* A texture's plane, of a YCbCr conversion's sampling. */
   uint32_t plane;
};

static bool
resolve_element(struct lower_state *state, nir_builder *b, nir_deref_instr *deref, struct element_ref *out)
{
   *out = (struct element_ref){ 0 };
   for (nir_deref_instr *d = deref; d; d = nir_deref_instr_parent(d)) {
      if (d->deref_type == nir_deref_type_var) {
         struct hash_entry *he = _mesa_hash_table_search(state->split, d->var);
         if (!he)
            return false;
         out->s = he->data;
         out->base = MIN2(out->base, out->s->count - 1);
         return true;
      }
      if (d->deref_type != nir_deref_type_array)
         continue;
      const uint32_t stride = glsl_type_is_array(d->type) ? glsl_get_aoa_size(d->type) : 1;
      if (nir_src_is_const(d->arr.index)) {
         out->base += nir_src_as_uint(d->arr.index) * stride;
      } else {
         b->cursor = nir_before_instr(&deref->instr);
         nir_def *i = nir_imul_imm(b, d->arr.index.ssa, stride);
         out->index = out->index ? nir_iadd(b, out->index, i) : i;
      }
   }
   return false;
}

static nir_variable *
ref_var(const struct element_ref *r, uint32_t element, bool sampler)
{
   element = MIN2(element, r->s->count - 1);
   if (!sampler && r->plane && r->plane <= 2 && r->s->planes[r->plane - 1])
      return r->s->planes[r->plane - 1][element];
   if (sampler && r->plane && r->s->chroma)
      return r->s->chroma[element];
   return sampler && r->s->sampler ? r->s->sampler[element] : r->s->texture[element];
}

/* How many elements a dynamic index chooses among, from the constant
 * part on. */
static uint32_t
ref_choices(const struct element_ref *r)
{
   return r->index ? r->s->count - r->base : 1;
}

/* An instruction on dynamically indexed arrays of resources: one on each
 * element they can be, and what the index says of their results. Reads
 * are all done, and selected from, which keeps them in uniform control
 * flow for derivatives; stores and atomics only where the index is.
 * `srcs` are the indices of the instruction's resource sources, `refs`
 * where they point: sources with the same index choose together. */
static nir_src *
instr_src(nir_instr *instr, unsigned index)
{
   if (instr->type == nir_instr_type_tex)
      return &nir_instr_as_tex(instr)->src[index].src;
   return &nir_instr_as_intrinsic(instr)->src[index];
}

static void
rewrite_dynamic(nir_builder *b, nir_instr *instr, const unsigned *srcs, const bool *samplers,
                struct element_ref *refs, unsigned count, bool writes)
{
   /* The indices, each once. */
   nir_def *indices[2] = { NULL, NULL };
   uint32_t choices[2] = { 1, 1 };
   unsigned which[2];
   unsigned n_indices = 0;
   for (unsigned i = 0; i < count; ++i) {
      which[i] = 0;
      if (!refs[i].index)
         continue;
      unsigned k = 0;
      while (k < n_indices && indices[k] != refs[i].index)
         ++k;
      if (k == n_indices) {
         indices[n_indices] = refs[i].index;
         choices[n_indices++] = ref_choices(&refs[i]);
      } else {
         choices[k] = MIN2(choices[k], ref_choices(&refs[i]));
      }
      which[i] = k;
   }
   b->cursor = nir_before_instr(instr);
   nir_def *def = nir_instr_def(instr);
   nir_def *result = NULL;
   for (uint32_t c0 = 0; c0 < choices[0]; ++c0) {
      for (uint32_t c1 = 0; c1 < choices[1]; ++c1) {
         const uint32_t picked[2] = { c0, c1 };
         nir_def *cond = nir_imm_true(b);
         for (unsigned k = 0; k < n_indices; ++k)
            cond = nir_iand(b, cond, nir_ieq_imm(b, indices[k], picked[k]));
         if (writes)
            nir_push_if(b, cond);
         /* The clone's resource sources, at the same places, made before
          * it. */
         nir_def *derefs[2];
         for (unsigned i = 0; i < count; ++i) {
            const uint32_t element = refs[i].base + (refs[i].index ? picked[which[i]] : 0);
            derefs[i] = &nir_build_deref_var(b, ref_var(&refs[i], element, samplers[i]))->def;
         }
         nir_instr *clone = nir_instr_clone(b->shader, instr);
         nir_builder_instr_insert(b, clone);
         for (unsigned i = 0; i < count; ++i)
            nir_src_rewrite(instr_src(clone, srcs[i]), derefs[i]);
         nir_def *value = nir_instr_def(clone);
         if (writes) {
            nir_pop_if(b, NULL);
            if (value)
               value = nir_if_phi(b, value, result ? result : nir_undef(b, value->num_components, value->bit_size));
            result = value;
         } else if (value) {
            result = result ? nir_bcsel(b, cond, value, result) : value;
         }
      }
   }
   if (def)
      nir_def_rewrite_uses(def, result);
   nir_instr_remove(instr);
}

/* An access to a cube image as one to the 2D array of its faces, which
 * the variable now is: the coordinates are the same, and a size has the
 * faces as layers, which are six a cube. */
static void
cube_image_as_array(nir_builder *b, nir_intrinsic_instr *intr)
{
   const bool arrayed = nir_intrinsic_image_array(intr);
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
   nir_intrinsic_set_image_array(intr, true);
   if (intr->intrinsic != nir_intrinsic_image_deref_size)
      return;
   const unsigned n = intr->def.num_components;
   intr->num_components = 3;
   intr->def.num_components = 3;
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *placeholder = nir_undef(b, n, intr->def.bit_size);
   nir_def_rewrite_uses(&intr->def, placeholder);
   b->cursor = nir_after_instr(&intr->instr);
   nir_def *size = arrayed ? nir_vec3(b, nir_channel(b, &intr->def, 0), nir_channel(b, &intr->def, 1),
                                      nir_udiv_imm(b, nir_channel(b, &intr->def, 2), 6))
                           : nir_trim_vector(b, &intr->def, 2);
   nir_def_rewrite_uses(placeholder, nir_trim_vector(b, size, n));
   nir_instr_remove(nir_def_instr(placeholder));
}

/* Whether an image variable is written: stored to, or its texels atomics'. */
static bool
image_written_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_image_deref_store && intr->intrinsic != nir_intrinsic_image_deref_atomic &&
       intr->intrinsic != nir_intrinsic_image_deref_atomic_swap)
      return false;
   nir_variable *var = nir_deref_instr_get_variable(nir_src_as_deref(intr->src[0]));
   if (var)
      _mesa_set_add(data, var);
   return false;
}

/* Storage images without a format (shaderStorageImageWriteWithoutFormat
 * and ReadWithoutFormat), which WGSL's have to have: the format of the
 * view a draw or dispatch binds there, which its variant is compiled for
 * (`records`); until one is, the 32-bit one-channel format of the image's
 * type, which WebGPU reads and writes, and atomics want. One only read is
 * a texture, which needs none. After split_resource_vars, where each
 * variable is one slot. */
static void
format_storage_images(nir_shader *nir, struct wvk_shader *shader, const struct wvk_shader_key *records,
                      struct util_dynarray *bgra)
{
   struct set *written = _mesa_pointer_set_create(NULL);
   nir_shader_intrinsics_pass(nir, image_written_instr, nir_metadata_all, written);
   nir_foreach_variable_with_modes(var, nir, nir_var_image) {
      if (var->data.image.format != PIPE_FORMAT_NONE || !_mesa_set_search(written, var))
         continue;
      const enum glsl_base_type base = glsl_get_sampler_result_type(glsl_without_array(var->type));
      VkFormat format = base == GLSL_TYPE_INT    ? VK_FORMAT_R32_SINT
                        : base == GLSL_TYPE_UINT ? VK_FORMAT_R32_UINT
                                                 : VK_FORMAT_R32_SFLOAT;
      if (shader->unformatted_count < WVK_MAX_UNFORMATTED_IMAGES) {
         const unsigned i = shader->unformatted_count++;
         shader->unformatted[i] = (uint16_t)(var->data.descriptor_set << 10 | var->data.binding);
         if (records && records->image_format[i])
            format = (VkFormat)records->image_format[i];
         shader->key.image_format[i] = format;
      } else {
         mesa_logw_once("wvk: more storage images without a format than %u", WVK_MAX_UNFORMATTED_IMAGES);
      }
      /* SPIR-V has no BGRA8 format, which WGSL's storage textures have
       * (bgra8unorm-storage), and which is the same to the shader: RGBA8
       * as ntv declares it, BGRA8 in the WGSL and the layout after. */
      if (format == VK_FORMAT_B8G8R8A8_UNORM) {
         format = VK_FORMAT_R8G8B8A8_UNORM;
         const uint32_t at = var->data.descriptor_set << 16 | var->data.binding;
         util_dynarray_append(bgra, at);
      }
      var->data.image.format = vk_format_to_pipe_format(format);
   }
   _mesa_set_destroy(written, NULL);
}

/* A storage texture ntv declared rgba8unorm that is bgra8unorm (see
 * format_storage_images): its declaration in the WGSL, in place. */
static void
declare_bgra_storage(char *wgsl, uint32_t group, uint32_t binding)
{
   char decl[48];
   snprintf(decl, sizeof(decl), "@group(%u) @binding(%u) var ", group, binding);
   char *at = wgsl ? strstr(wgsl, decl) : NULL;
   if (!at)
      return;
   const char *end = strchr(at, ';');
   char *format = strstr(at, "<rgba8unorm,");
   if (format && end && format < end)
      memcpy(format + 1, "bgra8unorm", strlen("bgra8unorm"));
}

/* Image loads and stores typed as the image is: SignExtend and ZeroExtend
 * let the value's signedness differ, and WGSL does not. The bits are the
 * same. */
static bool
match_image_type_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_image_deref_load && intr->intrinsic != nir_intrinsic_image_deref_store)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   enum glsl_base_type base = glsl_get_sampler_result_type(glsl_without_array(deref->type));
   /* A storage texture's channels are its format's: the variable too. */
   if (var && var->data.image.format != PIPE_FORMAT_NONE) {
      const enum pipe_format f = var->data.image.format;
      const enum glsl_base_type fbase = util_format_is_pure_sint(f)   ? GLSL_TYPE_INT
                                        : util_format_is_pure_uint(f) ? GLSL_TYPE_UINT
                                                                      : GLSL_TYPE_FLOAT;
      const struct glsl_type *bare = glsl_without_array(var->type);
      if (fbase != glsl_get_sampler_result_type(bare)) {
         var->type = glsl_type_wrap_in_arrays(
            glsl_image_type(glsl_get_sampler_dim(bare), glsl_sampler_type_is_array(bare), fbase), var->type);
         *(bool *)data = true;
      }
      base = fbase;
   }
   nir_alu_type want = base == GLSL_TYPE_INT ? nir_type_int : base == GLSL_TYPE_UINT ? nir_type_uint : nir_type_float;
   if (intr->intrinsic == nir_intrinsic_image_deref_load) {
      const nir_alu_type have = nir_intrinsic_dest_type(intr);
      if (nir_alu_type_get_base_type(have) == want || want == nir_type_float)
         return false;
      nir_intrinsic_set_dest_type(intr, want | nir_alu_type_get_type_size(have));
   } else {
      const nir_alu_type have = nir_intrinsic_src_type(intr);
      if (nir_alu_type_get_base_type(have) == want || want == nir_type_float)
         return false;
      nir_intrinsic_set_src_type(intr, want | nir_alu_type_get_type_size(have));
   }
   return true;
}

/* Queries of an integer cube, which is the 2D array of its faces
 * (integer_cube_as_array): sampling it is the samplers' lowering's. */
static bool
integer_cube_query_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int texture = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_CUBE || texture < 0 ||
       (tex->op != nir_texop_txs && tex->op != nir_texop_query_levels))
      return false;
   const struct glsl_type *type = glsl_without_array(nir_src_as_deref(tex->src[texture].src)->type);
   if (glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_2D)
      return false;
   const bool arrayed = tex->is_array;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   if (tex->op != nir_texop_txs)
      return true;
   /* The faces' size, and cubes of six layers each. */
   const unsigned n = tex->def.num_components;
   tex->def.num_components = 3;
   b->cursor = nir_after_instr(instr);
   nir_def *placeholder = nir_undef(b, n, tex->def.bit_size);
   nir_def_rewrite_uses(&tex->def, placeholder);
   b->cursor = nir_after_instr(instr);
   nir_def *size = arrayed ? nir_vec3(b, nir_channel(b, &tex->def, 0), nir_channel(b, &tex->def, 1),
                                      nir_udiv_imm(b, nir_channel(b, &tex->def, 2), 6))
                           : nir_trim_vector(b, &tex->def, 2);
   nir_def_rewrite_uses(placeholder, nir_trim_vector(b, size, n));
   nir_instr_remove(nir_def_instr(placeholder));
   return true;
}

static bool
rewrite_resource_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_state *state = data;
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      /* A combined sampler has only its texture deref for fetches and
       * queries, and both for sampling; either way each half goes to its
       * own variable. */
      unsigned srcs[2];
      bool samplers[2];
      struct element_ref refs[2];
      unsigned count = 0;
      bool dynamic = false;
      /* A plane of a YCbCr conversion's sampling: its own texture. */
      uint32_t plane = 0;
      const int plane_src = nir_tex_instr_src_index(tex, nir_tex_src_plane);
      if (plane_src >= 0) {
         plane = nir_src_as_uint(tex->src[plane_src].src);
         nir_tex_instr_remove_src(tex, plane_src);
      }
      for (unsigned i = 0; i < tex->num_srcs && count < 2; ++i) {
         const bool is_sampler = tex->src[i].src_type == nir_tex_src_sampler_deref;
         if (tex->src[i].src_type != nir_tex_src_texture_deref && !is_sampler)
            continue;
         if (!resolve_element(state, b, nir_src_as_deref(tex->src[i].src), &refs[count]))
            continue;
         refs[count].plane = plane;
         srcs[count] = i;
         samplers[count] = is_sampler;
         dynamic |= refs[count].index != NULL;
         ++count;
      }
      if (!count)
         return false;
      if (dynamic) {
         rewrite_dynamic(b, instr, srcs, samplers, refs, count, false);
         return true;
      }
      b->cursor = nir_before_instr(instr);
      for (unsigned i = 0; i < count; ++i)
         nir_src_rewrite(&tex->src[srcs[i]].src,
                         &nir_build_deref_var(b, ref_var(&refs[i], refs[i].base, samplers[i]))->def);
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
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_levels: {
      struct element_ref ref;
      if (!resolve_element(state, b, nir_src_as_deref(intr->src[0]), &ref))
         return false;
      if (nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_CUBE)
         cube_image_as_array(b, intr);
      if (ref.index) {
         const unsigned src = 0;
         const bool sampler = false;
         const bool writes = intr->intrinsic == nir_intrinsic_image_deref_store ||
                             intr->intrinsic == nir_intrinsic_image_deref_atomic ||
                             intr->intrinsic == nir_intrinsic_image_deref_atomic_swap;
         rewrite_dynamic(b, instr, &src, &sampler, &ref, 1, writes);
         return true;
      }
      b->cursor = nir_before_instr(instr);
      nir_src_rewrite(&intr->src[0], &nir_build_deref_var(b, ref_var(&ref, ref.base, false))->def);
      return true;
   }
   case nir_intrinsic_vulkan_resource_index: {
      /* Buffers: each element its own binding, found by (set, binding) in
       * zink's emitter. A dynamic index stays, from the binding's first
       * element: lower_to_words chooses among them. */
      const uint32_t set = nir_intrinsic_desc_set(intr);
      const uint32_t binding = nir_intrinsic_binding(intr);
      /* The driver's own are where they go. */
      if (set == WVK_DRIVER_GROUP && binding >= WVK_DRIVER_SLOT_BASE)
         return false;
      const struct wvk_descriptor_set_binding_layout *bl = binding_layout(state->info, set, binding);
      if (!bl) {
         mesa_loge("wvk: a shader uses set %u binding %u, which its pipeline layout does not have",
                   set, binding);
         state->failed = true;
         return false;
      }
      nir_intrinsic_set_binding(intr, bl->slot);
      if (nir_src_is_const(intr->src[0])) {
         nir_intrinsic_set_binding(intr, bl->slot + MIN2(nir_src_as_uint(intr->src[0]), bl->count - 1));
         b->cursor = nir_before_instr(instr);
         nir_src_rewrite(&intr->src[0], nir_imm_int(b, 0));
      }
      return true;
   }
   default:
      return false;
   }
}

/* A buffer variable per element too, at its binding, so that zink's
 * emitter finds each by (set, binding). */
static void
split_buffer_vars(nir_shader *nir, struct lower_state *state)
{
   /* The originals first: the clones join the same list. */
   struct util_dynarray originals;
   util_dynarray_init(&originals, state->split);
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo | nir_var_mem_ssbo)
      util_dynarray_append(&originals, var);

   util_dynarray_foreach(&originals, nir_variable *, v) {
      nir_variable *var = *v;
      /* The driver's own, such as the push constants, are where they go. */
      if (var->data.descriptor_set == WVK_DRIVER_GROUP && var->data.binding >= WVK_DRIVER_SLOT_BASE)
         continue;
      const struct wvk_descriptor_set_binding_layout *bl =
         binding_layout(state->info, var->data.descriptor_set, var->data.binding);
      if (!bl) {
         /* Declared and never used is allowed to be missing. */
         exec_node_remove(&var->node);
         continue;
      }
      /* vtn declares an array of blocks as one block: how many there are
       * is the layout's to say. */
      const uint32_t count = bl->count;
      const struct glsl_type *bare = glsl_without_array(var->type);
      for (uint32_t e = 0; e < count; ++e) {
         nir_variable *v = e == 0 ? var : nir_variable_clone(var, nir);
         v->type = bare;
         v->data.binding = bl->slot + e;
         if (e)
            nir_shader_add_variable(nir, v);
      }
   }
}

/* Push constants: a read-only storage buffer at the driver's binding, as
 * WGSL has none. Storage, not uniform, since push constant blocks have
 * std430's 4-byte array strides, which a uniform buffer cannot. */
static bool
lower_push_constants(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_push_const) {
      var->data.mode = nir_var_mem_ssbo;
      var->data.descriptor_set = WVK_DRIVER_GROUP;
      var->data.binding = WVK_DRIVER_SLOT_PUSH;
      var->data.access |= ACCESS_NON_WRITEABLE | ACCESS_RESTRICT;
      progress = true;
   }
   if (progress)
      nir_fixup_deref_modes(nir);
   return progress;
}

static nir_def *driver_buffer(nir_builder *b, uint32_t binding);

/* The layer or view the pass renders: the driver's data (a pass a layer). */
static nir_def *
driver_layer(nir_builder *b, unsigned which)
{
   nir_def *words = nir_load_ssbo(b, 2, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                                  nir_imm_int(b, WVK_DRIVER_LAYER_OFFSET), .align_mul = 8,
                                  .access = ACCESS_NON_WRITEABLE);
   return nir_channel(b, words, which);
}

/* gl_Layer and gl_ViewportIndex from a vertex shader. WebGPU renders one
 * layer and one viewport: a pass of several layers is one per layer, and
 * a vertex of another layer than the pass's goes out of the clip volume
 * -- x = 2w is outside whatever the viewport transform does -- so its
 * primitive is not drawn there. */
static bool
lower_layer_outputs(nir_shader *nir, struct wvk_shader *shader)
{
   if (nir->info.stage != MESA_SHADER_VERTEX)
      return false;
   bool progress = false;
   nir_variable *layer = NULL, *viewport = NULL, *position = NULL;
   nir_foreach_shader_out_variable(var, nir) {
      if (var->data.location == VARYING_SLOT_POS)
         position = var;
      if (var->data.location == VARYING_SLOT_LAYER || var->data.location == VARYING_SLOT_VIEWPORT) {
         if (var->data.location == VARYING_SLOT_LAYER)
            layer = var;
         else
            viewport = var;
         var->data.mode = nir_var_shader_temp;
         progress = true;
      }
   }
   if (layer && !glsl_type_is_scalar(layer->type))
      layer = NULL;
   if (viewport && !glsl_type_is_scalar(viewport->type))
      viewport = NULL;
   /* A viewport is a draw of its own, as a layer is a pass: what is not
    * the draw's goes out of the clip volume. */
   if ((layer || viewport) && position) {
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_builder b = nir_builder_at(nir_before_impl(impl));
      if (layer)
         nir_store_var(&b, layer, nir_imm_int(&b, 0), 0x1);
      if (viewport)
         nir_store_var(&b, viewport, nir_imm_int(&b, 0), 0x1);
      b.cursor = nir_after_impl(impl);
      nir_def *elsewhere = nir_imm_false(&b);
      if (layer)
         elsewhere = nir_ior(&b, elsewhere, nir_ine(&b, nir_load_var(&b, layer), driver_layer(&b, 0)));
      if (viewport) {
         nir_def *drawn = nir_load_ssbo(&b, 1, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                        nir_imm_int(&b, WVK_DRIVER_VIEWPORT), .align_mul = 4,
                                        .access = ACCESS_NON_WRITEABLE);
         elsewhere = nir_ior(&b, elsewhere, nir_ine(&b, nir_load_var(&b, viewport), drawn));
      }
      nir_push_if(&b, elsewhere);
      nir_store_var(&b, position, nir_imm_vec4(&b, 2.0f, 2.0f, 0.0f, 1.0f), 0xf);
      nir_pop_if(&b, NULL);
      nir_progress(true, impl, nir_metadata_none);
      shader->writes_layer = layer != NULL;
      shader->writes_viewport = viewport != NULL;
   }
   if (progress) {
      nir_fixup_deref_modes(nir);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp | nir_var_shader_temp, NULL);
   }
   return progress;
}

/* gl_Layer in a fragment shader, and the view index: the pass's, from the
 * driver's data. */
static bool
lower_layer_id_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_deref && b->shader->info.stage == MESA_SHADER_FRAGMENT &&
       nir_intrinsic_get_var(intr, 0) && nir_intrinsic_get_var(intr, 0)->data.mode == nir_var_shader_in &&
       nir_intrinsic_get_var(intr, 0)->data.location == VARYING_SLOT_VIEWPORT) {
      /* The draw's, of the draw per viewport. */
      b->cursor = nir_before_instr(&intr->instr);
      nir_def_replace(&intr->def, nir_load_ssbo(b, 1, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                                                nir_imm_int(b, WVK_DRIVER_VIEWPORT), .align_mul = 4,
                                                .access = ACCESS_NON_WRITEABLE));
      return true;
   }
   if (intr->intrinsic != nir_intrinsic_load_layer_id && intr->intrinsic != nir_intrinsic_load_view_index)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, driver_layer(b, intr->intrinsic == nir_intrinsic_load_view_index));
   return true;
}

/* Sampling an integer texture: WGSL has only fetches of those. Vulkan
 * samples integer formats nearest, so the fetch is of the texel the
 * coordinate falls in, clamped to the edge -- which is what vk_meta's and
 * nearly every application's samplers for these do. */

/*
 * Buffers as arrays of words. WGSL has layout rules of its own, which
 * std430 and row-major matrices break, and Tint then refuses the module.
 * So every access is lowered to an offset (nir_lower_explicit_io), and
 * each buffer becomes an array of 32-bit words -- of vec4s for a uniform
 * buffer, whose arrays WGSL wants 16 bytes apart -- which has no layout to
 * get wrong. This is what the GL driver does with its buffers too.
 */

/* Push constants through a descriptor, as every other buffer is, so that
 * the lowering below sees them the same way. */
static bool
push_constants_to_descriptor(nir_shader *nir)
{
   nir_variable *push = NULL;
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ssbo) {
      if (var->data.descriptor_set == WVK_DRIVER_GROUP && var->data.binding == WVK_DRIVER_SLOT_PUSH)
         push = var;
   }
   if (!push)
      return false;
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type != nir_deref_type_var || deref->var != push)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def *index = nir_vulkan_resource_index(&b, 2, 32, nir_imm_int(&b, 0),
                                                       .desc_set = WVK_DRIVER_GROUP,
                                                       .binding = WVK_DRIVER_SLOT_PUSH,
                                                       .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            nir_def *desc = nir_load_vulkan_descriptor(&b, 2, 32, index,
                                                       .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            nir_deref_instr *cast = nir_build_deref_cast(&b, desc, nir_var_mem_ssbo, push->type, 0);
            nir_def_replace(&deref->def, &cast->def);
         }
      }
      /* What hangs off the cast is an address of its size now. */
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_deref)
               continue;
            nir_deref_instr *deref = nir_instr_as_deref(instr);
            if (deref->deref_type != nir_deref_type_var && deref->deref_type != nir_deref_type_cast &&
                nir_deref_mode_is(deref, nir_var_mem_ssbo))
               deref->def.num_components = nir_src_num_components(deref->parent);
         }
      }
      nir_progress(true, impl, nir_metadata_none);
   }
   return true;
}

static nir_mem_access_size_align
word_access(nir_intrinsic_op intrin, uint8_t bytes, uint8_t bit_size, uint32_t align_mul,
            uint32_t align_offset, bool offset_is_const, enum gl_access_qualifier access,
            const void *cb_data)
{
   /* Whole words, a vector of them at most. A store of less than a word,
    * or not on one, is a word's atomics on its part; of more, the whole
    * words it has first. */
   const bool store = intrin == nir_intrinsic_store_ssbo || intrin == nir_intrinsic_store_shared;
   const uint32_t align = nir_combined_align(align_mul, align_offset);
   unsigned words = DIV_ROUND_UP(bytes, 4);
   if (store)
      words = align < 4 || bytes < 4 ? 1 : bytes / 4;
   return (nir_mem_access_size_align){
      .num_components = MIN2(words, 4),
      .bit_size = 32,
      .align = 4,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

/* Shared memory: 64-, 16- and 8-bit accesses as words, which WGSL has no
 * types for; the rest as they are, which ntv's shared block takes. */
static nir_mem_access_size_align
shared_access(nir_intrinsic_op intrin, uint8_t bytes, uint8_t bit_size, uint32_t align_mul, uint32_t align_offset,
              bool offset_is_const, enum gl_access_qualifier access, const void *cb_data)
{
   if (bit_size != 32)
      return word_access(intrin, bytes, bit_size, align_mul, align_offset, offset_is_const, access, cb_data);
   return (nir_mem_access_size_align){
      .num_components = MIN2(bytes / (bit_size / 8), 4),
      .bit_size = bit_size,
      .align = bit_size / 8,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

struct word_buffers {
   /* (set << 16 | binding) + 1 -> nir_variable * of words. */
   struct hash_table *vars;
   const struct vk_shader_compile_info *info;
   /* Uniform buffers: how big each block is, from the variables vtn made. */
   struct hash_table *ubo_sizes;
   /* Accesses whose binding a variable pointer chooses, which only the
    * index they are given at run time tells (lower_variable_buffers). */
   bool untraced;
   /* Bounds checked against each descriptor's range (its record's
    * WVK_META_BUF_RANGE): robustBufferAccess2, or null descriptors. */
   const uint32_t *meta;
   bool robust_ubo, robust_ssbo;
};

/* The binding an index operand of load_ssbo and friends came from:
 * vulkan_resource_index, through load_vulkan_descriptor; and for a
 * dynamically indexed array of buffers, the index, else NULL. */
static bool
index_binding(nir_src src, uint32_t *set, uint32_t *binding, nir_def **dynamic)
{
   nir_scalar s = nir_scalar_resolved(src.ssa, 0);
   if (dynamic)
      *dynamic = NULL;
   for (unsigned depth = 0; depth < 8; ++depth) {
      if (!nir_scalar_is_intrinsic(s))
         return false;
      nir_intrinsic_instr *intr = nir_def_as_intrinsic(s.def);
      if (intr->intrinsic == nir_intrinsic_vulkan_resource_index) {
         *set = nir_intrinsic_desc_set(intr);
         *binding = nir_intrinsic_binding(intr);
         if (dynamic && !nir_src_is_const(intr->src[0]))
            *dynamic = intr->src[0].ssa;
         return true;
      }
      if (intr->intrinsic != nir_intrinsic_load_vulkan_descriptor &&
          intr->intrinsic != nir_intrinsic_vulkan_resource_reindex)
         return false;
      s = nir_scalar_resolved(intr->src[0].ssa, 0);
   }
   return false;
}

/* How many elements the binding starting at WebGPU binding `slot` of
 * `set` has. */
static uint32_t
slot_elements(const struct vk_shader_compile_info *info, uint32_t set, uint32_t slot)
{
   if (set >= info->set_layout_count || !info->set_layouts[set])
      return 1;
   const struct wvk_descriptor_set_layout *layout =
      container_of(info->set_layouts[set], struct wvk_descriptor_set_layout, vk);
   for (uint32_t i = 0; i < layout->binding_count; ++i) {
      const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[i];
      if (bl->type != VK_DESCRIPTOR_TYPE_MAX_ENUM && bl->slot == slot)
         return MAX2(bl->count, 1);
   }
   return 1;
}

static nir_variable *
word_var(nir_shader *nir, struct word_buffers *wb, bool ubo, uint32_t set, uint32_t binding)
{
   void *k = (void *)(uintptr_t)(((uintptr_t)set << 16 | binding) + 1);
   struct hash_entry *he = _mesa_hash_table_search(wb->vars, k);
   if (he)
      return he->data;
   struct glsl_struct_field field = { .name = "words", .offset = 0 };
   if (ubo) {
      struct hash_entry *size = _mesa_hash_table_search(wb->ubo_sizes, k);
      const uint32_t bytes = size ? (uint32_t)(uintptr_t)size->data : 16;
      field.type = glsl_array_type(glsl_uvec4_type(), MAX2(DIV_ROUND_UP(bytes, 16), 1), 16);
   } else {
      field.type = glsl_array_type(glsl_uint_type(), 0, 4);
   }
   const struct glsl_type *type =
      glsl_interface_type(&field, 1, ubo ? GLSL_INTERFACE_PACKING_STD140 : GLSL_INTERFACE_PACKING_STD430,
                          false, ubo ? "wvk_ubo" : "wvk_ssbo");
   char name[32];
   snprintf(name, sizeof(name), "%s_%u_%u", ubo ? "ubo" : "ssbo", set, binding);
   nir_variable *var = nir_variable_create(nir, ubo ? nir_var_mem_ubo : nir_var_mem_ssbo, type, name);
   var->data.descriptor_set = set;
   var->data.binding = binding;
   var->interface_type = type;
   _mesa_hash_table_insert(wb->vars, k, var);
   return var;
}

/* The word at a byte offset: a deref of it, and for a uniform buffer the
 * component of the vec4 it is in. */
static nir_def *
load_word(nir_builder *b, nir_variable *var, bool ubo, nir_def *word)
{
   nir_deref_instr *words = nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
   if (!ubo)
      return nir_load_deref(b, nir_build_deref_array(b, words, word));
   nir_def *vec = nir_load_deref(b, nir_build_deref_array(b, words, nir_ushr_imm(b, word, 2)));
   return nir_vector_extract(b, vec, nir_iand_imm(b, word, 3));
}

static nir_def *lower_word_atomic(nir_builder *b, nir_variable *var, nir_intrinsic_instr *intr, nir_def *first);

/* The access, on the words of buffer (set, binding): its value, or NULL
 * for a store. */
static nir_def *
lower_word_access(nir_builder *b, struct word_buffers *wb, nir_intrinsic_instr *intr, bool ubo, uint32_t set,
                  uint32_t binding)
{
   nir_variable *var = word_var(b->shader, wb, ubo, set, binding);
   /* The descriptor's range, where accesses are bounded by it: loads past
    * it are zero, and stores and atomics none. */
   nir_def *range = NULL;
   const bool bounded = nir_intrinsic_has_access(intr) && (nir_intrinsic_access(intr) & ACCESS_IN_BOUNDS);
   if ((ubo ? wb->robust_ubo : wb->robust_ssbo) && wb->meta && !bounded &&
       (set != WVK_DRIVER_GROUP || binding < WVK_DRIVER_SLOT_BASE)) {
      /* From the push binding's words: this pass makes the words of any
       * load it could make of it. */
      nir_variable *push = word_var(b->shader, wb, false, WVK_DRIVER_GROUP, WVK_DRIVER_SLOT_PUSH);
      const uint32_t at = WVK_META_OFFSET / 4 + wb->meta[MIN2(set, 3)] + binding * WVK_META_WORDS + WVK_META_BUF_RANGE;
      range = load_word(b, push, false, nir_imm_int(b, at));
   }
   if (intr->intrinsic == nir_intrinsic_get_ssbo_size) {
      if (range)
         return range;
      nir_deref_instr *words = nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
      nir_def *length = nir_deref_buffer_array_length(b, 32, &words->def, .access = ACCESS_NON_WRITEABLE);
      return nir_imul_imm(b, length, 4);
   }

   const int offset_src = intr->intrinsic == nir_intrinsic_store_ssbo ? 2 : 1;
   nir_def *first = nir_ushr_imm(b, intr->src[offset_src].ssa, 2);

   if (intr->intrinsic == nir_intrinsic_load_ubo || intr->intrinsic == nir_intrinsic_load_ssbo) {
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned c = 0; c < intr->def.num_components; ++c) {
         nir_def *at = nir_iadd_imm(b, first, c);
         comps[c] = load_word(b, var, ubo, at);
         if (range)
            comps[c] = nir_bcsel(b, nir_ult(b, nir_ishl_imm(b, at, 2), range), comps[c], nir_imm_int(b, 0));
      }
      return nir_vec(b, comps, intr->def.num_components);
   }

   nir_deref_instr *words = nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
   if (intr->intrinsic == nir_intrinsic_store_ssbo) {
      nir_def *value = intr->src[0].ssa;
      u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
         nir_def *at = nir_iadd_imm(b, first, c);
         if (range)
            nir_push_if(b, nir_ult(b, nir_ishl_imm(b, at, 2), range));
         nir_store_deref(b, nir_build_deref_array(b, words, at), nir_channel(b, value, c), 0x1);
         if (range)
            nir_pop_if(b, NULL);
      }
      return NULL;
   }

   /* Atomics, on one word; none past the range, which returns zero. */
   if (range) {
      nir_def *zero = nir_imm_int(b, 0);
      nir_push_if(b, nir_ult(b, nir_ishl_imm(b, first, 2), range));
      nir_def *result = lower_word_atomic(b, var, intr, first);
      nir_pop_if(b, NULL);
      return nir_if_phi(b, result, zero);
   }
   return lower_word_atomic(b, var, intr, first);
}

/* An atomic on word `first` of a buffer's words. */
static nir_def *
lower_word_atomic(nir_builder *b, nir_variable *var, nir_intrinsic_instr *intr, nir_def *first)
{
   nir_deref_instr *words = nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
   nir_deref_instr *word = nir_build_deref_array(b, words, first);
   const nir_atomic_op op = nir_intrinsic_atomic_op(intr);
   if (op == nir_atomic_op_imin || op == nir_atomic_op_imax || op == nir_atomic_op_fadd || op == nir_atomic_op_fmin ||
       op == nir_atomic_op_fmax) {
      /* The words are unsigned, and so are WGSL's atomics on them: a
       * signed minimum or maximum is swapped in until nothing came
       * between; and so is a float sum, minimum or maximum, WGSL having
       * no atomics of floats (shaderBufferFloat32AtomicAdd and MinMax,
       * and images' through their buffers of texels). */
      nir_variable *old = nir_local_variable_create(b->impl, glsl_uint_type(), "atomic_old");
      nir_store_var(b, old, nir_deref_atomic(b, 32, &word->def, nir_imm_int(b, 0), .atomic_op = nir_atomic_op_iadd),
                    0x1);
      nir_loop *loop = nir_push_loop(b);
      {
         nir_def *cur = nir_load_var(b, old);
         nir_def *data = intr->src[2].ssa;
         nir_def *want = op == nir_atomic_op_imin   ? nir_imin(b, cur, data)
                         : op == nir_atomic_op_imax ? nir_imax(b, cur, data)
                         : op == nir_atomic_op_fadd ? nir_fadd(b, cur, data)
                         : op == nir_atomic_op_fmin ? nir_fmin(b, cur, data)
                                                    : nir_fmax(b, cur, data);
         nir_def *prev = nir_deref_atomic_swap(b, 32, &word->def, cur, want, .atomic_op = nir_atomic_op_cmpxchg);
         nir_store_var(b, old, prev, 0x1);
         nir_break_if(b, nir_ieq(b, prev, cur));
      }
      nir_pop_loop(b, loop);
      return nir_load_var(b, old);
   }
   if (intr->intrinsic == nir_intrinsic_ssbo_atomic_swap)
      return nir_deref_atomic_swap(b, 32, &word->def, intr->src[2].ssa, intr->src[3].ssa,
                                   .atomic_op = nir_intrinsic_atomic_op(intr));
   return nir_deref_atomic(b, 32, &word->def, intr->src[2].ssa, .atomic_op = nir_intrinsic_atomic_op(intr));
}

static bool
lower_to_words_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct word_buffers *wb = data;
   bool ubo = false, writes = false;
   int index_src;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
      ubo = true;
      FALLTHROUGH;
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_get_ssbo_size:
      index_src = 0;
      break;
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      index_src = 0;
      writes = true;
      break;
   case nir_intrinsic_store_ssbo:
      index_src = 1;
      writes = true;
      break;
   default:
      return false;
   }
   uint32_t set, binding;
   nir_def *dynamic;
   if (!index_binding(intr->src[index_src], &set, &binding, &dynamic)) {
      /* A variable pointer's: after the rest. */
      wb->untraced = true;
      return false;
   }
   if ((intr->intrinsic == nir_intrinsic_load_ubo || intr->intrinsic == nir_intrinsic_load_ssbo) &&
       intr->def.bit_size != 32) {
      mesa_loge("wvk: a %u-bit buffer load", intr->def.bit_size);
      return false;
   }
   if (intr->intrinsic == nir_intrinsic_store_ssbo && intr->src[0].ssa->bit_size != 32) {
      mesa_loge("wvk: a %u-bit buffer store", intr->src[0].ssa->bit_size);
      return false;
   }
   b->cursor = nir_before_instr(&intr->instr);

   if (!dynamic) {
      nir_def *result = lower_word_access(b, wb, intr, ubo, set, binding);
      if (result)
         nir_def_replace(&intr->def, result);
      else
         nir_instr_remove(&intr->instr);
      return true;
   }

   /* An array of buffers, dynamically indexed: each element that can be,
    * reads selected from, writes where the index is. */
   const uint32_t count = slot_elements(wb->info, set, binding);
   nir_def *result = NULL;
   for (uint32_t e = 0; e < count; ++e) {
      nir_def *here = nir_ieq_imm(b, dynamic, e);
      if (writes)
         nir_push_if(b, here);
      nir_def *value = lower_word_access(b, wb, intr, ubo, set, binding + e);
      if (writes) {
         nir_pop_if(b, NULL);
         if (value)
            value = nir_if_phi(b, value, result ? result : nir_undef(b, value->num_components, value->bit_size));
         result = value;
      } else {
         result = result ? nir_bcsel(b, here, value, result) : value;
      }
   }
   if (result)
      nir_def_replace(&intr->def, result);
   else
      nir_instr_remove(&intr->instr);
   return true;
}

/* A buffer a variable pointer may choose: its set, WebGPU binding, and
 * whether uniform. */
struct buffer_candidate {
   uint32_t set, binding;
   bool ubo;
};

static uint32_t
buffer_id(uint32_t set, uint32_t binding)
{
   return set << 16 | binding;
}

/* Variable pointers (variablePointersStorageBuffer): a pointer's buffer
 * chosen by a phi or select, which no access can be told the binding of
 * from where its index came. So every buffer's index says at run time
 * which it is -- set << 16 | binding, the element added to the binding
 * -- and each such access is one on each buffer of its kind the shader
 * has, reads selected from and writes where the index is. */
static bool
lower_variable_buffers(nir_shader *nir, struct word_buffers *wb)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   struct util_dynarray candidates;
   util_dynarray_init(&candidates, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         b.cursor = nir_before_instr(instr);
         if (intr->intrinsic == nir_intrinsic_vulkan_resource_index) {
            const uint32_t set = nir_intrinsic_desc_set(intr), binding = nir_intrinsic_binding(intr);
            const VkDescriptorType type = (VkDescriptorType)nir_intrinsic_desc_type(intr);
            const bool ubo = type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
                             type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                             type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
            /* Not the driver's own, which no pointer of the shader's is. */
            const bool driver = set == WVK_DRIVER_GROUP && binding >= WVK_DRIVER_SLOT_BASE;
            const uint32_t count = driver ? 0 : slot_elements(wb->info, set, binding);
            for (uint32_t e = 0; e < count; ++e) {
               bool seen = false;
               util_dynarray_foreach(&candidates, struct buffer_candidate, c)
                  seen |= c->set == set && c->binding == binding + e && c->ubo == ubo;
               if (!seen)
                  util_dynarray_append(&candidates, ((struct buffer_candidate){ set, binding + e, ubo }));
            }
            nir_def *id = nir_iadd_imm(&b, intr->src[0].ssa, buffer_id(set, binding));
            nir_def_replace(&intr->def, nir_vec2(&b, id, nir_imm_int(&b, 0)));
         } else if (intr->intrinsic == nir_intrinsic_vulkan_resource_reindex) {
            nir_def *index = intr->src[0].ssa;
            nir_def_replace(&intr->def, nir_vec2(&b, nir_iadd(&b, nir_channel(&b, index, 0), intr->src[1].ssa),
                                                 nir_channel(&b, index, 1)));
         } else if (intr->intrinsic == nir_intrinsic_load_vulkan_descriptor) {
            nir_def_replace(&intr->def, intr->src[0].ssa);
         }
      }
   }
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         bool ubo = false, writes = false;
         int index_src = 0;
         switch (intr->intrinsic) {
         case nir_intrinsic_load_ubo: ubo = true; break;
         case nir_intrinsic_load_ssbo:
         case nir_intrinsic_get_ssbo_size: break;
         case nir_intrinsic_ssbo_atomic:
         case nir_intrinsic_ssbo_atomic_swap: writes = true; break;
         case nir_intrinsic_store_ssbo: index_src = 1; writes = true; break;
         default: continue;
         }
         b.cursor = nir_before_instr(instr);
         nir_def *id = nir_channel(&b, intr->src[index_src].ssa, 0);
         nir_def *result = NULL;
         util_dynarray_foreach(&candidates, struct buffer_candidate, c) {
            if (c->ubo != ubo)
               continue;
            nir_def *here = nir_ieq_imm(&b, id, buffer_id(c->set, c->binding));
            if (writes)
               nir_push_if(&b, here);
            nir_def *value = lower_word_access(&b, wb, intr, ubo, c->set, c->binding);
            if (writes) {
               nir_pop_if(&b, NULL);
               if (value)
                  value = nir_if_phi(&b, value, result ? result : nir_undef(&b, value->num_components, value->bit_size));
               result = value;
            } else {
               result = result ? nir_bcsel(&b, here, value, result) : value;
            }
         }
         if (nir_intrinsic_infos[intr->intrinsic].has_dest && !result)
            result = nir_undef(&b, intr->def.num_components, intr->def.bit_size);
         if (result)
            nir_def_replace(&intr->def, result);
         else
            nir_instr_remove(instr);
      }
   }
   util_dynarray_fini(&candidates);
   return nir_progress(true, impl, nir_metadata_none);
}

static bool
zero_descriptors_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_vulkan_descriptor &&
       intr->intrinsic != nir_intrinsic_vulkan_resource_index &&
       intr->intrinsic != nir_intrinsic_vulkan_resource_reindex)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_zero(b, intr->def.num_components, intr->def.bit_size));
   return true;
}

/* The index operand of loads from one of the driver's buffers. */
static nir_def *
driver_buffer(nir_builder *b, uint32_t binding)
{
   nir_def *index = nir_vulkan_resource_index(b, 2, 32, nir_imm_int(b, 0), .desc_set = WVK_DRIVER_GROUP,
                                              .binding = binding,
                                              .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   nir_def *desc = nir_load_vulkan_descriptor(b, 2, 32, index, .desc_type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
   return nir_channel(b, desc, 0);
}

/* The depth bounds test, which WebGPU has no depth-stencil state for.
 *
 * It tests the depth the attachment *holds*, not the fragment's, so the
 * fragment shader reads it from a copy of the attachment taken before the
 * draw (wvk_cmd_draw.c) and leaves where it is outside the bounds, which
 * the driver's data has. At the end, as Vulkan's late test is, so that a
 * killed fragment's side effects still happen; at the start for a shader
 * that asks for early fragment tests, which would not have them. */
static bool
lower_depth_bounds(nir_shader *nir)
{
   const bool early = nir->info.fs.early_fragment_tests;
   /* One way out, so that the end is where every invocation gets to. */
   if (!early)
      NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(early ? nir_before_impl(impl) : nir_after_impl(impl));
   nir_variable *tex = nir_variable_create(nir, nir_var_uniform,
                                           glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
                                           "wvk_depth_bounds_dst");
   tex->data.descriptor_set = WVK_DRIVER_GROUP;
   tex->data.binding = WVK_DRIVER_SLOT_DEPTH_BOUNDS;
   nir_def *coord = nir_f2i32(&b, nir_trim_vector(&b, nir_load_frag_coord(&b), 2));
   nir_def *held = nir_channel(&b, nir_txf(&b, coord, .texture_deref = nir_build_deref_var(&b, tex),
                                           .lod = nir_imm_int(&b, 0), .dest_type = nir_type_float32), 0);
   nir_def *bounds = nir_load_ssbo(&b, 2, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                   nir_imm_int(&b, WVK_DRIVER_DEPTH_BOUNDS), .align_mul = 8,
                                   .access = ACCESS_NON_WRITEABLE);
   nir_def *outside = nir_ior(&b, nir_flt(&b, held, nir_channel(&b, bounds, 0)),
                              nir_flt(&b, nir_channel(&b, bounds, 1), held));
   nir_terminate_if(&b, outside);
   return nir_progress(true, impl, nir_metadata_none);
}

/* BaseVertex, BaseInstance and DrawIndex. The driver's data says which
 * words of the draw binding the first two are -- a direct draw's own
 * values in the push buffer, or the fields of an indirect draw's command
 * where it is -- and says DrawIndex itself. See set_draw_parameters. */
static bool
lower_draw_parameters_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   unsigned c;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_base_vertex:
      c = 0;
      break;
   case nir_intrinsic_load_base_instance:
      c = 1;
      break;
   case nir_intrinsic_load_draw_id:
      c = 2;
      break;
   default:
      return false;
   }
   /* DrawIndex alone too: the draw's words say it. */
   struct wvk_shader *shader = data;
   shader->uses_draw_parameters = true;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *where = nir_load_ssbo(b, 4, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                                  nir_imm_int(b, WVK_DRIVER_DRAW_OFFSET), .align_mul = 16,
                                  .access = ACCESS_NON_WRITEABLE);
   nir_def *value = nir_channel(b, where, c);
   if (c < 2) {
      value = nir_load_ssbo(b, 1, 32, driver_buffer(b, WVK_DRIVER_SLOT_DRAW), nir_ishl_imm(b, value, 2),
                            .align_mul = 4, .access = ACCESS_NON_WRITEABLE);
   }
   nir_def_replace(&intr->def, value);
   return true;
}

/* One channel of a vertex attribute, as its format has it, from the
 * attribute's first byte in the pulled binding. Vulkan aligns attributes
 * to their components, or to the whole of a packed format, so no channel
 * straddles two words. */
static nir_def *
fetch_channel(nir_builder *b, nir_def *address, const struct util_format_description *desc, unsigned c,
              uint32_t slot)
{
   const struct util_format_channel_description *ch = &desc->channel[c];
   nir_def *bit = nir_iadd_imm(b, nir_ishl_imm(b, address, 3), ch->shift);
   nir_def *word = nir_load_ssbo(b, 1, 32, driver_buffer(b, WVK_DRIVER_SLOT_VERTEX + slot),
                                 nir_ishl_imm(b, nir_ushr_imm(b, bit, 5), 2), .align_mul = 4,
                                 .access = ACCESS_NON_WRITEABLE);
   nir_def *shift = nir_iand_imm(b, bit, 31);
   const bool is_signed = ch->type == UTIL_FORMAT_TYPE_SIGNED;
   nir_def *bits = ch->size == 32 ? word
                   : is_signed    ? nir_ibitfield_extract(b, word, shift, nir_imm_int(b, ch->size))
                                  : nir_ubitfield_extract(b, word, shift, nir_imm_int(b, ch->size));
   switch (ch->type) {
   case UTIL_FORMAT_TYPE_FLOAT:
      return ch->size == 16 ? nir_channel(b, nir_unpack_half_2x16(b, bits), 0) : bits;
   case UTIL_FORMAT_TYPE_UNSIGNED:
      if (ch->normalized)
         return nir_fdiv_imm(b, nir_u2f32(b, bits), (double)u_uintN_max(ch->size));
      return ch->pure_integer ? bits : nir_u2f32(b, bits);
   case UTIL_FORMAT_TYPE_SIGNED:
      if (ch->normalized)
         return nir_fmax(b, nir_fdiv_imm(b, nir_i2f32(b, bits), (double)u_intN_max(ch->size)),
                         nir_imm_float(b, -1.0f));
      return ch->pure_integer ? bits : nir_i2f32(b, bits);
   default:
      return nir_imm_int(b, 0);
   }
}

/* The vertex a vertex-rate attribute is fetched for, and the instance an
 * instance-rate one. */
static nir_def *
vertex_element(nir_builder *b)
{
   nir_foreach_function_temp_variable(var, b->impl) {
      if (var->name && !strcmp(var->name, WVK_POINT_VERTEX))
         return nir_load_var(b, var);
   }
   return nir_load_vertex_id(b);
}

static nir_def *
instance_element(nir_builder *b)
{
   nir_foreach_function_temp_variable(var, b->impl) {
      if (var->name && !strcmp(var->name, WVK_RUN_INSTANCE))
         return nir_load_var(b, var);
   }
   return nir_load_instance_id(b);
}

/* An attribute's value: Vulkan's element firstInstance + (InstanceIndex -
 * firstInstance) / divisor, or firstInstance for a divisor of 0, from
 * where the draw's range of the buffer starts in the binding. */
static nir_def *
fetch_attribute(nir_builder *b, const struct vk_vertex_input_state *vi, int location, uint32_t slot)
{
   const struct vk_vertex_attribute_state *a = &vi->attributes[location];
   const struct vk_vertex_binding_state *vb = &vi->bindings[a->binding];
   nir_def *element;
   if (vb->input_rate == VK_VERTEX_INPUT_RATE_VERTEX) {
      element = vertex_element(b);
   } else {
      nir_def *first = nir_load_base_instance(b);
      element = first;
      if (vb->divisor)
         element = nir_iadd(b, first, nir_udiv_imm(b, nir_isub(b, instance_element(b), first), vb->divisor));
   }
   /* Where the binding starts, and its stride as the draw has it, which
    * may be dynamic: of the first four, and past them. */
   nir_def *start, *stride;
   if (slot < 4) {
      start = nir_load_ssbo(b, 1, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                            nir_imm_int(b, WVK_DRIVER_PULL_OFFSET + slot * 4), .align_mul = 4,
                            .access = ACCESS_NON_WRITEABLE);
      stride = nir_load_ssbo(b, 1, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                             nir_imm_int(b, WVK_DRIVER_PULL_STRIDES + slot * 4), .align_mul = 4,
                             .access = ACCESS_NON_WRITEABLE);
   } else {
      start = nir_load_ssbo(b, 1, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                            nir_imm_int(b, WVK_DRIVER_PULL_MORE + (slot - 4) * 4), .align_mul = 4,
                            .access = ACCESS_NON_WRITEABLE);
      stride = nir_load_ssbo(b, 1, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                             nir_imm_int(b, WVK_DRIVER_PULL_MORE + 8 + (slot - 4) * 4), .align_mul = 4,
                             .access = ACCESS_NON_WRITEABLE);
   }
   nir_def *address = nir_iadd(b, start, nir_iadd_imm(b, nir_imul(b, element, stride), a->offset));

   const struct util_format_description *desc = vk_format_description(a->format);
   nir_def *channels[4];
   for (unsigned c = 0; c < desc->nr_channels; ++c)
      channels[c] = fetch_channel(b, address, desc, c, slot);
   const bool integer = util_format_is_pure_integer(desc->format);
   nir_def *out[4];
   for (unsigned i = 0; i < 4; ++i) {
      switch (desc->swizzle[i]) {
      case PIPE_SWIZZLE_X:
      case PIPE_SWIZZLE_Y:
      case PIPE_SWIZZLE_Z:
      case PIPE_SWIZZLE_W:
         out[i] = channels[desc->swizzle[i] - PIPE_SWIZZLE_X];
         break;
      case PIPE_SWIZZLE_1:
         out[i] = integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0f);
         break;
      default:
         out[i] = nir_imm_int(b, 0);
         break;
      }
   }
   return nir_vec(b, out, 4);
}

static bool
pull_divided_attributes(nir_shader *nir, const struct vk_graphics_pipeline_state *state,
                        struct wvk_shader *shader)
{
   const struct vk_vertex_input_state *vi = state ? state->vi : NULL;
   if (nir->info.stage != MESA_SHADER_VERTEX || !vi)
      return false;
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_deref)
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            if (!nir_deref_mode_is(deref, nir_var_shader_in))
               continue;
            /* A location of a matrix or an array is an element of it. */
            nir_variable *var = nir_deref_instr_get_variable(deref);
            int location = var->data.location - VERT_ATTRIB_GENERIC0;
            if (deref->deref_type == nir_deref_type_array && nir_src_is_const(deref->arr.index))
               location += nir_src_as_uint(deref->arr.index);
            else if (deref->deref_type != nir_deref_type_var)
               continue;
            /* One the draw has no attribute at (vertexAttributeRobustness):
             * (0, 0, 0, 1), as a format's missing components are. */
            if (location >= 0 && location < MESA_VK_MAX_VERTEX_ATTRIBUTES &&
                !(vi->attributes_valid & BITFIELD_BIT(location))) {
               b.cursor = nir_before_instr(instr);
               const unsigned bits = intr->def.bit_size;
               const bool is_float = glsl_base_type_is_float(glsl_get_base_type(glsl_without_array(deref->type)));
               nir_def *zero = nir_imm_zero(&b, 1, bits);
               nir_def *one = is_float ? nir_imm_floatN_t(&b, 1.0, bits) : nir_imm_intN_t(&b, 1, bits);
               nir_def *c[4] = { zero, zero, zero, one };
               nir_def_replace(&intr->def, nir_vec(&b, c, intr->def.num_components));
               progress = true;
               continue;
            }
            if (!pulls(vi, location, shader->key.points || shader->key.runs || shader->key.lines))
               continue;
            if (glsl_get_bit_size(deref->type) == 64) {
               mesa_logw_once("wvk: a 64-bit attribute the shader pulls");
               continue;
            }
            const uint32_t binding = vi->attributes[location].binding;
            uint32_t slot = 0;
            while (slot < shader->pulled_count && shader->pulled_bindings[slot] != binding)
               ++slot;
            if (slot == WVK_MAX_PULLED_BINDINGS) {
               mesa_logw_once("wvk: more than %u vertex bindings with divisors", WVK_MAX_PULLED_BINDINGS);
               continue;
            }
            if (slot == shader->pulled_count)
               shader->pulled_bindings[shader->pulled_count++] = binding;
            b.cursor = nir_before_instr(instr);
            nir_def *value = nir_trim_vector(&b, fetch_attribute(&b, vi, location, slot), intr->def.num_components);
            /* A 16-bit input, of the 32-bit value. */
            if (intr->def.bit_size != 32) {
               const enum glsl_base_type base = glsl_get_base_type(deref->type);
               value = base == GLSL_TYPE_FLOAT16 ? nir_f2fN(&b, value, intr->def.bit_size)
                                                 : nir_u2uN(&b, value, intr->def.bit_size);
            }
            nir_def_replace(&intr->def, value);
            progress = true;
         }
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   if (progress) {
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_in, NULL);
   }
   return progress;
}

/* The viewport transform from the driver's data: a vertex shader's
 * position is scaled and moved to where Vulkan's viewport would have put
 * it, WebGPU's being kept inside the attachments. See set_viewport_scissor. */
static bool
lower_viewport_transform(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_VERTEX)
      return false;
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_variable *var = nir_intrinsic_get_var(intr, 0);
            if (!var || var->data.mode != nir_var_shader_out || var->data.location != VARYING_SLOT_POS)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_def *t = nir_load_ssbo(&b, 4, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                       nir_imm_int(&b, WVK_DRIVER_DATA_OFFSET), .align_mul = 16,
                                       .access = ACCESS_NON_WRITEABLE);
            nir_def *pos = intr->src[1].ssa;
            nir_def *w = nir_channel(&b, pos, 3);
            nir_def *x = nir_ffma(&b, nir_channel(&b, pos, 0), nir_channel(&b, t, 0),
                                  nir_fmul(&b, w, nir_channel(&b, t, 1)));
            nir_def *y = nir_ffma(&b, nir_channel(&b, pos, 1), nir_channel(&b, t, 2),
                                  nir_fmul(&b, w, nir_channel(&b, t, 3)));
            /* Depth: w - z where Vulkan's range is inverted, which WebGPU's
             * cannot be. */
            nir_def *d = nir_load_ssbo(&b, 2, 32, driver_buffer(&b, WVK_DRIVER_SLOT_PUSH),
                                       nir_imm_int(&b, WVK_DRIVER_DEPTH_OFFSET), .align_mul = 16,
                                       .access = ACCESS_NON_WRITEABLE);
            nir_def *z = nir_ffma(&b, nir_channel(&b, pos, 2), nir_channel(&b, d, 0),
                                  nir_fmul(&b, w, nir_channel(&b, d, 1)));
            nir_src_rewrite(&intr->src[1], nir_vec4(&b, x, y, z, w));
            progress = true;
         }
      }
      nir_progress(progress, impl, nir_metadata_control_flow);
   }
   return progress;
}

static bool
lower_buffers_to_words(nir_shader *nir, const struct vk_shader_compile_info *info,
                       const struct vk_graphics_pipeline_state *state, struct wvk_shader *shader,
                       void *mem_ctx)
{
   const struct vk_pipeline_robustness_state *rs = info->robustness;
   struct word_buffers wb = {
      .vars = _mesa_pointer_hash_table_create(mem_ctx),
      .info = info,
      .ubo_sizes = _mesa_pointer_hash_table_create(mem_ctx),
      .meta = shader->meta_base,
      .robust_ubo = rs && (rs->uniform_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2 ||
                           rs->null_uniform_buffer_descriptor),
      .robust_ssbo = rs && (rs->storage_buffers == VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS_2 ||
                            rs->null_storage_buffer_descriptor),
   };
   NIR_PASS(_, nir, push_constants_to_descriptor);
   /* How big each uniform block is, before its variable goes. */
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo) {
      void *k = (void *)(uintptr_t)(((uintptr_t)var->data.descriptor_set << 16 | var->data.binding) + 1);
      const uint32_t bytes = glsl_get_explicit_size(glsl_without_array(var->type), false);
      _mesa_hash_table_insert(wb.ubo_sizes, k, (void *)(uintptr_t)bytes);
   }
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ubo | nir_var_mem_ssbo,
            nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, lower_viewport_transform);
   /* Before the draw parameters: pulling reads BaseInstance. */
   NIR_PASS(_, nir, pull_divided_attributes, state, shader);
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_draw_parameters_instr, nir_metadata_control_flow,
               shader);
   }
   /* Stores of less than a word, atomics on the bits of it: others may
    * store to the rest of the word. */
   const nir_lower_mem_access_bit_sizes_options sizes = {
      .callback = word_access,
      .modes = nir_var_mem_ubo | nir_var_mem_ssbo,
      .may_lower_unaligned_stores_to_atomics = true,
   };
   NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &sizes);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   /* vtn's variables go; the words take their place. */
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_mem_ubo | nir_var_mem_ssbo)
      exec_node_remove(&var->node);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_to_words_instr, nir_metadata_none, &wb);
   if (wb.untraced)
      NIR_PASS(_, nir, lower_variable_buffers, &wb);
   /* Every access has its binding now; what is left of the descriptors is
    * their offset part, which is zero. */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, zero_descriptors_instr, nir_metadata_control_flow, NULL);
   optimize(nir);
   return true;
}

/* 1D images are 2D textures one texel high, as 1D images are here: the
 * variable, the intrinsics' dimension, a y of 0 in coordinates, and sizes
 * without the height again. */
static bool
lower_1d_image_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   switch (intr->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_image_deref_levels:
      break;
   default:
      return false;
   }
   if (nir_intrinsic_image_dim(intr) != GLSL_SAMPLER_DIM_1D)
      return false;
   const bool array = nir_intrinsic_image_array(intr);
   nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);

   if (intr->intrinsic == nir_intrinsic_image_deref_size) {
      b->cursor = nir_after_instr(&intr->instr);
      intr->def.num_components = array ? 3 : 2;
      intr->num_components = intr->def.num_components;
      nir_def *size = array ? nir_vec2(b, nir_channel(b, &intr->def, 0), nir_channel(b, &intr->def, 2))
                            : nir_channel(b, &intr->def, 0);
      nir_def_rewrite_uses_after(&intr->def, size);
      return true;
   }
   if (intr->intrinsic == nir_intrinsic_image_deref_samples ||
       intr->intrinsic == nir_intrinsic_image_deref_levels)
      return true;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *coord = intr->src[1].ssa;
   nir_def *zero = nir_imm_zero(b, 1, coord->bit_size);
   nir_def *xy[4] = {
      nir_channel(b, coord, 0), zero, array ? nir_channel(b, coord, 1) : zero, zero,
   };
   nir_src_rewrite(&intr->src[1], nir_vec(b, xy, coord->num_components));
   return true;
}

/* WGSL has no multisampled arrays, and multisampled images are one layer
 * here: the array goes from the texture, and the layer from its fetches. */
static bool
lower_ms_array_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->sampler_dim != GLSL_SAMPLER_DIM_MS || !tex->is_array)
      return false;
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   b->cursor = nir_before_instr(instr);
   if (coord_idx >= 0 && tex->op != nir_texop_txs && tex->op != nir_texop_texture_samples)
      nir_src_rewrite(&tex->src[coord_idx].src, nir_trim_vector(b, tex->src[coord_idx].src.ssa, 2));
   tex->is_array = false;
   if (tex->op == nir_texop_txs) {
      /* The size: the layers are one. */
      b->cursor = nir_after_instr(instr);
      tex->def.num_components = 2;
      nir_def *size = nir_vec3(b, nir_channel(b, &tex->def, 0), nir_channel(b, &tex->def, 1), nir_imm_int(b, 1));
      nir_def_rewrite_uses_after(&tex->def, size);
   } else {
      tex->coord_components = 2;
   }
   return true;
}

static bool
lower_ms_arrays(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      const struct glsl_type *bare = glsl_without_array(var->type);
      if (!glsl_type_is_texture(bare) || glsl_get_sampler_dim(bare) != GLSL_SAMPLER_DIM_MS ||
          !glsl_sampler_type_is_array(bare))
         continue;
      var->type = glsl_type_wrap_in_arrays(
         glsl_texture_type(GLSL_SAMPLER_DIM_MS, false, glsl_get_sampler_result_type(bare)), var->type);
      progress = true;
   }
   if (!progress)
      return false;
   nir_fixup_deref_types(nir);
   return nir_shader_instructions_pass(nir, lower_ms_array_instr, nir_metadata_control_flow, NULL);
}

/* Demote is WGSL's discard already; Tint reads only OpKill for it. */
static bool
demote_to_terminate_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_demote)
      intr->intrinsic = nir_intrinsic_terminate;
   else if (intr->intrinsic == nir_intrinsic_demote_if)
      intr->intrinsic = nir_intrinsic_terminate_if;
   else
      return false;
   return true;
}

/* Structs among a stage's varyings -- GLSL's interface blocks, and arrays
 * of them -- as a variable for each member of each element: WGSL has no
 * structs there, and the SPIR-V emitter gives them offsets Tint refuses.
 * Each leaf takes the location and interpolation its member has. */
struct io_leaf {
   nir_variable *var;
   /* Which element of each array, and which member of each struct, from
    * the variable down to the leaf. */
   uint8_t path[8];
   uint8_t depth;
};

struct io_split {
   nir_variable *var;
   struct util_dynarray leaves;
   /* Indexed by a variable somewhere, and so left whole. */
   bool failed;
};

/* The leaves of `type`, at `location`: `offset` is how many locations
 * the arrays around it have moved it, for the members of the variable's
 * own struct, whose locations vtn resolved (nir_variable::members). */
static void
flatten_io_type(nir_shader *nir, struct io_split *split, const struct glsl_type *type, int location,
                unsigned offset, const struct nir_variable_data *data, bool top, uint8_t *path, uint8_t depth)
{
   const bool struct_below = glsl_type_is_struct_or_ifc(glsl_without_array(type));
   if (!struct_below || depth == ARRAY_SIZE(((struct io_leaf *)0)->path)) {
      nir_variable *var = nir_variable_clone(split->var, nir);
      var->type = type;
      var->members = NULL;
      var->num_members = 0;
      var->data.location = location;
      var->data.location_frac = 0;
      if (data) {
         var->data.location_frac = data->location_frac;
         var->data.interpolation = data->interpolation;
         var->data.centroid = data->centroid;
         var->data.sample = data->sample;
      }
      var->name = ralloc_asprintf(nir, "%s_%u", split->var->name ? split->var->name : "io",
                                  (unsigned)util_dynarray_num_elements(&split->leaves, struct io_leaf));
      nir_shader_add_variable(nir, var);
      struct io_leaf leaf = { .var = var, .depth = depth };
      memcpy(leaf.path, path, depth);
      util_dynarray_append(&split->leaves, leaf);
      return;
   }
   if (glsl_type_is_array(type)) {
      const struct glsl_type *elem = glsl_get_array_element(type);
      const unsigned slots = glsl_count_vec4_slots(elem, false, false);
      for (unsigned e = 0; e < glsl_get_length(type); ++e) {
         path[depth] = e;
         flatten_io_type(nir, split, elem, location + e * slots, offset + e * slots, data, top, path, depth + 1);
      }
      return;
   }
   int at = location;
   for (unsigned m = 0; m < glsl_get_length(type); ++m) {
      const struct glsl_type *member = glsl_get_struct_field(type, m);
      const struct nir_variable_data *member_data = data;
      if (top && split->var->members) {
         /* A member of the block is where vtn says, in each element. */
         member_data = &split->var->members[m];
         at = split->var->members[m].location + offset;
      }
      path[depth] = m;
      flatten_io_type(nir, split, member, at, offset, member_data, false, path, depth + 1);
      at += glsl_count_vec4_slots(member, false, false);
   }
}

static bool
split_io_structs(nir_shader *nir)
{
   nir_variable_mode mode;
   if (nir->info.stage == MESA_SHADER_VERTEX)
      mode = nir_var_shader_out;
   else if (nir->info.stage == MESA_SHADER_FRAGMENT)
      mode = nir_var_shader_in;
   else
      return false;

   struct util_dynarray splits = UTIL_DYNARRAY_INIT;
   nir_foreach_variable_with_modes(var, nir, mode) {
      if (!glsl_type_is_struct_or_ifc(glsl_without_array(var->type)))
         continue;
      const int location = var->members ? var->members[0].location : var->data.location;
      if (location < VARYING_SLOT_VAR0)
         continue;
      struct io_split split = { .var = var, .leaves = UTIL_DYNARRAY_INIT };
      util_dynarray_append(&splits, split);
   }
   if (!splits.size)
      return false;
   util_dynarray_foreach(&splits, struct io_split, split) {
      uint8_t path[8];
      flatten_io_type(nir, split, split->var->type, split->var->data.location, 0, &split->var->data, true, path, 0);
   }

   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (!nir_intrinsic_infos[intr->intrinsic].num_srcs || !nir_src_as_deref(intr->src[0]))
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            nir_variable *root = nir_deref_instr_get_variable(deref);
            struct io_split *split = NULL;
            util_dynarray_foreach(&splits, struct io_split, s) {
               if (s->var == root)
                  split = s;
            }
            if (!split)
               continue;
            nir_deref_path p;
            nir_deref_path_init(&p, deref, NULL);
            /* The leaf: struct members and array elements down to it, which
             * have to be constant. */
            uint8_t path[8];
            unsigned depth = 0, i = 1;
            for (; p.path[i]; ++i) {
               const struct glsl_type *parent = p.path[i - 1]->type;
               if (!glsl_type_is_struct_or_ifc(glsl_without_array(parent)) || depth == ARRAY_SIZE(path))
                  break;
               nir_deref_instr *d = p.path[i];
               if (d->deref_type == nir_deref_type_struct) {
                  path[depth++] = d->strct.index;
               } else if (d->deref_type == nir_deref_type_array && nir_src_is_const(d->arr.index)) {
                  path[depth++] = nir_src_as_uint(d->arr.index);
               } else {
                  split->failed = true;
                  break;
               }
            }
            struct io_leaf *leaf = NULL;
            util_dynarray_foreach(&split->leaves, struct io_leaf, l) {
               if (l->depth == depth && !memcmp(l->path, path, depth))
                  leaf = l;
            }
            if (!leaf) {
               nir_deref_path_finish(&p);
               split->failed = true;
               continue;
            }
            /* The rest of the chain, on the leaf. */
            b.cursor = nir_before_instr(instr);
            nir_deref_instr *d = nir_build_deref_var(&b, leaf->var);
            for (; p.path[i]; ++i)
               d = nir_build_deref_follower(&b, d, p.path[i]);
            nir_deref_path_finish(&p);
            nir_src_rewrite(&intr->src[0], &d->def);
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   util_dynarray_foreach(&splits, struct io_split, split) {
      if (split->failed)
         mesa_logw_once("wvk: a varying struct indexed by a variable");
      else
         exec_node_remove(&split->var->node);
      util_dynarray_fini(&split->leaves);
   }
   util_dynarray_fini(&splits);
   NIR_PASS(_, nir, nir_opt_dce);
   return true;
}

/* Vectors of fewer than four components among a stage's outputs or
 * inputs, as vec4s. A fragment output of fewer than its attachment has,
 * which Vulkan leaves undefined, WebGPU refuses; and so does a vertex
 * output of more components than the fragment input at its location,
 * which Vulkan allows: every user varying is a vec4 on either side. A
 * location shared among components of several variables stays. */
static bool
pad_io_var(nir_shader *nir, nir_variable *var)
{
   const struct glsl_type *old = glsl_without_array(var->type);
   if (!glsl_type_is_vector_or_scalar(old) || glsl_get_vector_elements(old) == 4 ||
       glsl_get_bit_size(old) != 32 || var->data.location_frac)
      return false;
   nir_foreach_variable_with_modes(other, nir, var->data.mode) {
      if (other != var && other->data.location == var->data.location)
         return false;
   }
   /* Of every array level it has. */
   var->type = glsl_type_wrap_in_arrays(glsl_vector_type(glsl_get_base_type(old), 4), var->type);
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            const bool load = intr->intrinsic == nir_intrinsic_load_deref ||
                              intr->intrinsic == nir_intrinsic_interp_deref_at_centroid ||
                              intr->intrinsic == nir_intrinsic_interp_deref_at_sample ||
                              intr->intrinsic == nir_intrinsic_interp_deref_at_offset;
            if ((intr->intrinsic != nir_intrinsic_store_deref && !load) ||
                nir_intrinsic_get_var(intr, 0) != var ||
                !glsl_type_is_vector_or_scalar(nir_src_as_deref(intr->src[0])->type))
               continue;
            if (load) {
               /* What was read: the components it had. */
               const unsigned n = intr->def.num_components;
               b.cursor = nir_after_instr(instr);
               intr->num_components = 4;
               intr->def.num_components = 4;
               nir_def *trimmed = nir_trim_vector(&b, &intr->def, n);
               nir_def_rewrite_uses_after(&intr->def, trimmed);
               continue;
            }
            /* What is written: the components it wrote, the rest zero. */
            b.cursor = nir_before_instr(instr);
            nir_def *value = intr->src[1].ssa;
            nir_def *comps[4];
            for (unsigned c = 0; c < 4; ++c)
               comps[c] = c < value->num_components ? nir_channel(&b, value, c)
                                                    : nir_imm_zero(&b, 1, value->bit_size);
            nir_src_rewrite(&intr->src[1], nir_vec(&b, comps, 4));
            intr->num_components = 4;
            nir_intrinsic_set_write_mask(intr, nir_intrinsic_write_mask(intr) |
                                                  (0xf & ~BITFIELD_MASK(value->num_components)));
         }
      }
   }
   return true;
}

static bool
pad_io_vectors(nir_shader *nir)
{
   bool progress = false;
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      nir_foreach_shader_out_variable(var, nir) {
         if (var->data.location >= FRAG_RESULT_DATA0)
            progress |= pad_io_var(nir, var);
      }
   }
   if (nir->info.stage == MESA_SHADER_VERTEX || nir->info.stage == MESA_SHADER_FRAGMENT) {
      const nir_variable_mode mode = nir->info.stage == MESA_SHADER_VERTEX ? nir_var_shader_out : nir_var_shader_in;
      nir_foreach_variable_with_modes(var, nir, mode) {
         if (var->data.location >= VARYING_SLOT_VAR0 && var->data.location <= VARYING_SLOT_VAR31)
            progress |= pad_io_var(nir, var);
      }
   }
   if (progress)
      nir_fixup_deref_types(nir);
   return progress;
}

static bool
lower_1d_images(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_image | nir_var_uniform) {
      const struct glsl_type *bare = glsl_without_array(var->type);
      if (!glsl_type_is_image(bare) || glsl_get_sampler_dim(bare) != GLSL_SAMPLER_DIM_1D)
         continue;
      var->type = glsl_type_wrap_in_arrays(
         glsl_image_type(GLSL_SAMPLER_DIM_2D, glsl_sampler_type_is_array(bare),
                         glsl_get_sampler_result_type(bare)), var->type);
      progress = true;
   }
   if (!progress)
      return false;
   nir_fixup_deref_types(nir);
   return nir_shader_intrinsics_pass(nir, lower_1d_image_instr, nir_metadata_control_flow, NULL);
}

/* The (set, binding) a buffer deref chain starts from: a variable, or
 * vtn's cast of a loaded descriptor. */
static bool
deref_buffer_binding(nir_deref_instr *deref, uint32_t *set, uint32_t *binding)
{
   while (deref) {
      if (deref->deref_type == nir_deref_type_var) {
         *set = deref->var->data.descriptor_set;
         *binding = deref->var->data.binding;
         return true;
      }
      if (deref->deref_type == nir_deref_type_cast) {
         nir_intrinsic_instr *intr = nir_src_as_intrinsic(deref->parent);
         while (intr && intr->intrinsic != nir_intrinsic_vulkan_resource_index)
            intr = nir_intrinsic_infos[intr->intrinsic].num_srcs ? nir_src_as_intrinsic(intr->src[0]) : NULL;
         if (!intr)
            return false;
         *set = nir_intrinsic_desc_set(intr);
         *binding = nir_intrinsic_binding(intr);
         return true;
      }
      deref = nir_deref_instr_parent(deref);
   }
   return false;
}

struct usage {
   /* (group << 16 | binding) of buffers used, and written. */
   struct set *used;
   struct set *written;
   /* Textures sampled (not only fetched or queried), and compared;
    * samplers used for comparisons. */
   struct set *sampled;
   struct set *compared;
   /* Images loaded from, or with atomics: those only stored to are
    * write-only, which more formats are storage of than read-write. */
   struct set *loaded;
};

static void *
key(uint32_t set, uint32_t binding)
{
   return (void *)(uintptr_t)(((uintptr_t)set << 16 | binding) + 1);
}

static void
note_var_deref(struct set *s, nir_src *src)
{
   nir_deref_instr *deref = nir_src_as_deref(*src);
   nir_variable *var = deref ? nir_deref_instr_get_variable(deref) : NULL;
   if (var)
      _mesa_set_add(s, key(var->data.descriptor_set, var->data.binding));
}

static bool
gather_usage_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct usage *u = data;
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      const bool samples = tex->op == nir_texop_tex || tex->op == nir_texop_txb ||
                           tex->op == nir_texop_txl || tex->op == nir_texop_txd ||
                           tex->op == nir_texop_tg4 || tex->op == nir_texop_lod;
      for (unsigned i = 0; i < tex->num_srcs; ++i) {
         if (tex->src[i].src_type != nir_tex_src_texture_deref &&
             tex->src[i].src_type != nir_tex_src_sampler_deref)
            continue;
         if (samples)
            note_var_deref(u->sampled, &tex->src[i].src);
         if (tex->is_shadow)
            note_var_deref(u->compared, &tex->src[i].src);
      }
      return false;
   }
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      if (intr->intrinsic == nir_intrinsic_vulkan_resource_index) {
         _mesa_set_add(u->used, key(nir_intrinsic_desc_set(intr), nir_intrinsic_binding(intr)));
         return false;
      }
      bool writes = false;
      switch (intr->intrinsic) {
      case nir_intrinsic_store_deref:
      case nir_intrinsic_deref_atomic:
      case nir_intrinsic_deref_atomic_swap:
         writes = true;
         break;
      case nir_intrinsic_image_deref_store:
      case nir_intrinsic_image_deref_atomic:
      case nir_intrinsic_image_deref_atomic_swap:
         /* An image never written is read-only, as Tint takes it. */
         note_var_deref(u->written, &intr->src[0]);
         if (intr->intrinsic != nir_intrinsic_image_deref_store)
            note_var_deref(u->loaded, &intr->src[0]);
         return false;
      case nir_intrinsic_image_deref_load:
      case nir_intrinsic_image_deref_sparse_load:
         note_var_deref(u->loaded, &intr->src[0]);
         return false;
      default:
         break;
      }
      if (writes) {
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         uint32_t set, binding;
         if (deref && nir_deref_mode_is(deref, nir_var_mem_ssbo) &&
             deref_buffer_binding(deref, &set, &binding))
            _mesa_set_add(u->written, key(set, binding));
      }
      return false;
   }
   if (instr->type == nir_instr_type_deref) {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      if (deref->deref_type == nir_deref_type_var &&
          (deref->var->data.mode & (nir_var_mem_ubo | nir_var_mem_ssbo)))
         _mesa_set_add(u->used, key(deref->var->data.descriptor_set, deref->var->data.binding));
   }
   return false;
}

static WGPUTextureViewDimension
view_dimension(const struct glsl_type *type)
{
   const bool array = glsl_sampler_type_is_array(type);
   switch (glsl_get_sampler_dim(type)) {
   case GLSL_SAMPLER_DIM_3D:
      return WGPUTextureViewDimension_3D;
   case GLSL_SAMPLER_DIM_CUBE:
      return array ? WGPUTextureViewDimension_CubeArray : WGPUTextureViewDimension_Cube;
   default:
      /* 1D is 2D here, as its images are. */
      return array ? WGPUTextureViewDimension_2DArray : WGPUTextureViewDimension_2D;
   }
}

static WGPUTextureFormat
storage_format(enum pipe_format format)
{
   const struct wvk_format *f = wvk_format_get(vk_format_from_pipe_format(format));
   return f ? f->texture : WGPUTextureFormat_Undefined;
}

/* The bindings the shader uses, for its bind group layouts. Variables it
 * does not use go, so that nothing has to be bound for them. */
static bool
reflect_bindings(nir_shader *nir, const struct vk_shader_compile_info *info,
                 struct wvk_shader *shader, void *mem_ctx)
{
   struct usage u = {
      .used = _mesa_pointer_set_create(mem_ctx),
      .written = _mesa_pointer_set_create(mem_ctx),
      .loaded = _mesa_pointer_set_create(mem_ctx),
      .sampled = _mesa_pointer_set_create(mem_ctx),
      .compared = _mesa_pointer_set_create(mem_ctx),
   };
   nir_shader_instructions_pass(nir, gather_usage_instr, nir_metadata_all, &u);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_uniform | nir_var_image, NULL);

   uint32_t count = 0;
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_uniform | nir_var_image)
      ++count;
   struct wvk_shader_binding *bindings = calloc(MAX2(count, 1), sizeof(*bindings));
   if (!bindings)
      return false;

   count = 0;
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_mem_ubo | nir_var_mem_ssbo | nir_var_uniform | nir_var_image) {
      void *k = key(var->data.descriptor_set, var->data.binding);
      struct wvk_shader_binding *sb = &bindings[count];
      *sb = (struct wvk_shader_binding){
         .group = var->data.descriptor_set,
         .binding = var->data.binding,
      };
      const struct glsl_type *type = glsl_without_array(var->type);
      if (var->data.mode & (nir_var_mem_ubo | nir_var_mem_ssbo)) {
         if (!_mesa_set_search(u.used, k)) {
            exec_node_remove(&var->node);
            continue;
         }
         if (var->data.mode == nir_var_mem_ubo) {
            sb->kind = WVK_BINDING_UNIFORM;
            sb->min_size = glsl_get_explicit_size(type, false);
         } else if (_mesa_set_search(u.written, k)) {
            sb->kind = WVK_BINDING_STORAGE;
         } else {
            /* Read-only, which a vertex shader's storage has to be. */
            var->data.access |= ACCESS_NON_WRITEABLE;
            sb->kind = WVK_BINDING_STORAGE_READ;
         }
         if (var->data.descriptor_set == WVK_DRIVER_GROUP && var->data.binding >= WVK_DRIVER_SLOT_BASE) {
            sb->dynamic = true;
            shader->uses_push_constants |= var->data.binding == WVK_DRIVER_SLOT_PUSH;
            shader->uses_draw_parameters |= var->data.binding == WVK_DRIVER_SLOT_DRAW;
         } else {
            /* The set layout's binding this slot came from. */
            const struct wvk_descriptor_set_layout *layout =
               var->data.descriptor_set < info->set_layout_count && info->set_layouts[var->data.descriptor_set]
                  ? container_of(info->set_layouts[var->data.descriptor_set],
                                 struct wvk_descriptor_set_layout, vk)
                  : NULL;
            if (layout && var->data.binding < layout->slot_count) {
               const struct wvk_descriptor_slot *slot = &layout->slots[var->data.binding];
               sb->dynamic = wvk_descriptor_type_is_dynamic(layout->bindings[slot->binding].type);
            }
         }
      } else if (glsl_type_is_bare_sampler(type)) {
         sb->kind = glsl_sampler_type_is_shadow(type) || _mesa_set_search(u.compared, k)
                       ? WVK_BINDING_SAMPLER_COMPARE : WVK_BINDING_SAMPLER;
      } else if (glsl_type_is_image(type) &&
                 ((var->data.access & ACCESS_NON_WRITEABLE) || !_mesa_set_search(u.written, k))) {
         /* Read-only, even if not read either: Tint takes no image that
          * is neither. */
         var->data.access |= ACCESS_NON_WRITEABLE;
         var->data.access &= ~ACCESS_NON_READABLE;
         /* Tint makes a read-only storage image a texture it loads from. */
         sb->kind = WVK_BINDING_TEXTURE;
         sb->view_dimension = view_dimension(type);
         sb->multisampled = glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_MS;
         switch (glsl_get_sampler_result_type(type)) {
         case GLSL_TYPE_INT: sb->sample_type = WGPUTextureSampleType_Sint; break;
         case GLSL_TYPE_UINT: sb->sample_type = WGPUTextureSampleType_Uint; break;
         default: sb->sample_type = WGPUTextureSampleType_UnfilterableFloat; break;
         }
      } else if (glsl_type_is_image(type)) {
         sb->kind = WVK_BINDING_STORAGE_TEXTURE;
         sb->view_dimension = view_dimension(type);
         sb->format = storage_format(var->data.image.format);
         if (sb->format == WGPUTextureFormat_Undefined) {
            mesa_loge("wvk: a storage image without a format, which WebGPU needs");
            free(bindings);
            return false;
         }
         if (!_mesa_set_search(u.loaded, k))
            var->data.access |= ACCESS_NON_READABLE;
         if (var->data.access & ACCESS_NON_WRITEABLE)
            sb->access = WGPUStorageTextureAccess_ReadOnly;
         else if (var->data.access & ACCESS_NON_READABLE)
            sb->access = WGPUStorageTextureAccess_WriteOnly;
         else
            sb->access = WGPUStorageTextureAccess_ReadWrite;
      } else {
         sb->kind = WVK_BINDING_TEXTURE;
         sb->view_dimension = view_dimension(type);
         sb->multisampled = glsl_get_sampler_dim(type) == GLSL_SAMPLER_DIM_MS;
         switch (glsl_get_sampler_result_type(type)) {
         case GLSL_TYPE_INT:
            sb->sample_type = WGPUTextureSampleType_Sint;
            break;
         case GLSL_TYPE_UINT:
            sb->sample_type = WGPUTextureSampleType_Uint;
            break;
         default:
            if (_mesa_set_search(u.compared, k))
               sb->sample_type = WGPUTextureSampleType_Depth;
            else if (_mesa_set_search(u.sampled, k) && !sb->multisampled)
               sb->sample_type = WGPUTextureSampleType_Float;
            else
               /* Only fetched: the most any float texture binds as. */
               sb->sample_type = WGPUTextureSampleType_UnfilterableFloat;
            break;
         }
      }
      ++count;
   }
   shader->bindings = bindings;
   shader->binding_count = count;
   return true;
}

static VkResult
make_module(struct wvk_device *dev, const char *wgsl, const char *label, WGPUShaderModule *out)
{
   WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
   source.code = wvk_sv(wgsl);
   WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
   desc.nextInChain = &source.chain;
   desc.label = wvk_sv(label);
   *out = wgpuDeviceCreateShaderModule(dev->wgpu, &desc);
   return *out ? VK_SUCCESS : vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
}

static const struct vk_shader_ops wvk_shader_ops;

/* What a vertex or fragment shader is compiled from, kept to compile its
 * variants from (wvk_shader_draw_variant): its NIR as compile got it, and
 * copies of the set layouts the compile reads -- of their own, for a
 * shader outlives its layouts, and a cached one comes without them. */
struct wvk_shader_source {
   struct blob nir;
   VkShaderCreateFlagsEXT flags;
   VkShaderStageFlags next_stage_mask;
   /* The pipeline's robustness, which its variants are compiled with. */
   struct vk_pipeline_robustness_state robustness;
   uint32_t set_layout_count;
   struct wvk_descriptor_set_layout *layouts[MESA_VK_MAX_DESCRIPTOR_SETS];
   struct vk_descriptor_set_layout *vk_layouts[MESA_VK_MAX_DESCRIPTOR_SETS];

   simple_mtx_t lock;
   /* struct wvk_shader_key * -> struct wvk_shader * */
   struct hash_table *variants;
};

static uint32_t
key_hash(const void *key)
{
   return _mesa_hash_data(key, sizeof(struct wvk_shader_key));
}

static bool
key_equal(const void *a, const void *b)
{
   return !memcmp(a, b, sizeof(struct wvk_shader_key));
}

/* A layout as the compile reads it: without the samplers, which it does not. */
static struct wvk_descriptor_set_layout *
layout_alloc(uint32_t binding_count, uint32_t slot_count)
{
   struct wvk_descriptor_set_layout *l =
      calloc(1, sizeof(*l) + binding_count * sizeof(l->bindings[0]));
   if (!l)
      return NULL;
   l->binding_count = binding_count;
   l->slot_count = slot_count;
   l->slots = calloc(MAX2(slot_count, 1), sizeof(*l->slots));
   if (!l->slots) {
      free(l);
      return NULL;
   }
   return l;
}

static struct wvk_descriptor_set_layout *
layout_copy(const struct wvk_descriptor_set_layout *from)
{
   struct wvk_descriptor_set_layout *l = layout_alloc(from->binding_count, from->slot_count);
   if (!l)
      return NULL;
   l->descriptor_count = from->descriptor_count;
   l->dynamic_count = from->dynamic_count;
   l->inline_size = from->inline_size;
   memcpy(l->bindings, from->bindings, from->binding_count * sizeof(l->bindings[0]));
   for (uint32_t i = 0; i < from->binding_count; ++i)
      l->bindings[i].immutable_samplers = NULL;
   memcpy(l->slots, from->slots, from->slot_count * sizeof(l->slots[0]));
   return l;
}

static void
source_destroy(struct wvk_device *dev, struct wvk_shader_source *src)
{
   if (src->variants) {
      hash_table_foreach(src->variants, he) {
         struct wvk_shader *variant = he->data;
         if (variant)
            variant->vk.ops->destroy(&dev->vk, &variant->vk, NULL);
      }
      _mesa_hash_table_destroy(src->variants, NULL);
   }
   for (uint32_t g = 0; g < ARRAY_SIZE(src->layouts); ++g) {
      if (src->layouts[g]) {
         free(src->layouts[g]->slots);
         free(src->layouts[g]);
      }
   }
   blob_finish(&src->nir);
   simple_mtx_destroy(&src->lock);
   free(src);
}

static struct wvk_shader_source *
source_alloc(void)
{
   struct wvk_shader_source *src = calloc(1, sizeof(*src));
   if (!src)
      return NULL;
   blob_init(&src->nir);
   simple_mtx_init(&src->lock, mtx_plain);
   src->variants = _mesa_hash_table_create(NULL, key_hash, key_equal);
   if (!src->variants) {
      blob_finish(&src->nir);
      simple_mtx_destroy(&src->lock);
      free(src);
      return NULL;
   }
   return src;
}

/* The source of what compile is about to compile, before it changes it. */
static struct wvk_shader_source *
source_create(const struct vk_shader_compile_info *info)
{
   struct wvk_shader_source *src = source_alloc();
   if (!src)
      return NULL;
   nir_serialize(&src->nir, info->nir, false);
   src->flags = info->flags;
   src->next_stage_mask = info->next_stage_mask;
   if (info->robustness)
      src->robustness = *info->robustness;
   src->set_layout_count = MIN2(info->set_layout_count, ARRAY_SIZE(src->layouts));
   bool ok = !src->nir.out_of_memory;
   for (uint32_t g = 0; ok && g < src->set_layout_count; ++g) {
      if (!info->set_layouts[g])
         continue;
      src->layouts[g] = layout_copy(container_of(info->set_layouts[g], struct wvk_descriptor_set_layout, vk));
      ok = src->layouts[g] != NULL;
      if (ok)
         src->vk_layouts[g] = &src->layouts[g]->vk;
   }
   if (!ok) {
      src->set_layout_count = 0;
      source_destroy(NULL, src);
      return NULL;
   }
   return src;
}

static void
source_serialize(const struct wvk_shader_source *src, struct blob *blob)
{
   blob_write_uint32(blob, src->nir.size);
   blob_write_bytes(blob, src->nir.data, src->nir.size);
   blob_write_uint32(blob, src->flags);
   blob_write_uint32(blob, src->next_stage_mask);
   blob_write_bytes(blob, &src->robustness, sizeof(src->robustness));
   blob_write_uint32(blob, src->set_layout_count);
   for (uint32_t g = 0; g < src->set_layout_count; ++g) {
      const struct wvk_descriptor_set_layout *l = src->layouts[g];
      blob_write_uint8(blob, l != NULL);
      if (!l)
         continue;
      blob_write_uint32(blob, l->binding_count);
      blob_write_uint32(blob, l->slot_count);
      blob_write_uint32(blob, l->descriptor_count);
      blob_write_uint32(blob, l->dynamic_count);
      blob_write_uint32(blob, l->inline_size);
      for (uint32_t i = 0; i < l->binding_count; ++i) {
         const struct wvk_descriptor_set_binding_layout *b = &l->bindings[i];
         const uint32_t words[] = { b->type, b->count, b->stages, b->descriptor_index, b->dynamic_index,
                                    b->slot, b->inline_offset, b->inline_size, b->planes,
                                    b->plane_descriptor_index };
         blob_write_bytes(blob, words, sizeof(words));
      }
      for (uint32_t i = 0; i < l->slot_count; ++i) {
         blob_write_uint32(blob, l->slots[i].binding);
         blob_write_uint32(blob, l->slots[i].element);
         blob_write_uint8(blob, l->slots[i].sampler);
         blob_write_uint8(blob, l->slots[i].plane);
      }
   }
}

static struct wvk_shader_source *
source_deserialize(struct blob_reader *blob)
{
   struct wvk_shader_source *src = source_alloc();
   if (!src)
      return NULL;
   const uint32_t nir_size = blob_read_uint32(blob);
   const void *nir = blob_read_bytes(blob, nir_size);
   if (nir)
      blob_write_bytes(&src->nir, nir, nir_size);
   src->flags = blob_read_uint32(blob);
   src->next_stage_mask = blob_read_uint32(blob);
   blob_copy_bytes(blob, &src->robustness, sizeof(src->robustness));
   /* Read once: MIN2 would read it twice. */
   const uint32_t set_layout_count = blob_read_uint32(blob);
   src->set_layout_count = MIN2(set_layout_count, ARRAY_SIZE(src->layouts));
   for (uint32_t g = 0; !blob->overrun && g < src->set_layout_count; ++g) {
      if (!blob_read_uint8(blob))
         continue;
      const uint32_t binding_count = blob_read_uint32(blob);
      const uint32_t slot_count = blob_read_uint32(blob);
         if (blob->overrun || binding_count > WVK_MAX_SET_SLOTS || slot_count > WVK_MAX_SET_SLOTS)
         break;
      struct wvk_descriptor_set_layout *l = layout_alloc(binding_count, slot_count);
      if (!l)
         break;
      src->layouts[g] = l;
      src->vk_layouts[g] = &l->vk;
      l->descriptor_count = blob_read_uint32(blob);
      l->dynamic_count = blob_read_uint32(blob);
      l->inline_size = blob_read_uint32(blob);
      for (uint32_t i = 0; i < binding_count; ++i) {
         uint32_t words[10];
         blob_copy_bytes(blob, words, sizeof(words));
         l->bindings[i] = (struct wvk_descriptor_set_binding_layout){
            .type = words[0], .count = words[1], .stages = words[2], .descriptor_index = words[3],
            .dynamic_index = words[4], .slot = words[5], .inline_offset = words[6], .inline_size = words[7],
            .planes = words[8], .plane_descriptor_index = words[9],
         };
      }
      for (uint32_t i = 0; i < slot_count; ++i) {
         l->slots[i].binding = blob_read_uint32(blob);
         l->slots[i].element = blob_read_uint32(blob);
         l->slots[i].sampler = blob_read_uint8(blob);
         l->slots[i].plane = blob_read_uint8(blob);
      }
   }
   if (blob->overrun || !nir || src->nir.out_of_memory) {
      source_destroy(NULL, src);
      return NULL;
   }
   return src;
}

static void
wvk_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                   const VkAllocationCallbacks *pAllocator)
{
   struct wvk_shader *shader = wvk_shader_from_vk(vk_shader);
   for (uint32_t i = 0; i < ARRAY_SIZE(shader->cull_gs); ++i) {
      if (shader->cull_gs[i])
         wvk_shader_destroy(vk_dev, &shader->cull_gs[i]->vk, NULL);
   }
   if (shader->source)
      source_destroy(container_of(vk_dev, struct wvk_device, vk), shader->source);
   for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i) {
      if (shader->module[i])
         wgpuShaderModuleRelease(shader->module[i]);
      free(shader->wgsl[i]);
   }
   free(shader->bindings);
   vk_shader_free(vk_dev, pAllocator, vk_shader);
}

/* A varying as WVK_VARYING_* has it: integers are flat, as WGSL insists. */
static uint8_t
varying_code(const nir_variable *var)
{
   const enum glsl_base_type base = glsl_get_base_type(glsl_without_array(var->type));
   if (glsl_base_type_is_integer(base) || base == GLSL_TYPE_BOOL) {
      const bool is_signed = base == GLSL_TYPE_INT || base == GLSL_TYPE_INT16 || base == GLSL_TYPE_INT8 ||
                             base == GLSL_TYPE_INT64;
      return WVK_VARYING_PRESENT | WVK_VARYING_FLAT | (is_signed ? WVK_VARYING_SINT : WVK_VARYING_UINT);
   }
   switch (var->data.interpolation) {
   case INTERP_MODE_FLAT: return WVK_VARYING_PRESENT | WVK_VARYING_FLAT;
   case INTERP_MODE_NOPERSPECTIVE: return WVK_VARYING_PRESENT | WVK_VARYING_LINEAR;
   default: return WVK_VARYING_PRESENT;
   }
}

static void
record_varyings(nir_shader *nir, nir_variable_mode mode, uint8_t out[WVK_MAX_VARYINGS])
{
   memset(out, 0, WVK_MAX_VARYINGS);
   nir_foreach_variable_with_modes(var, nir, mode) {
      if (var->data.location < VARYING_SLOT_VAR0)
         continue;
      const unsigned first = var->data.location - VARYING_SLOT_VAR0;
      const unsigned slots = glsl_count_attribute_slots(var->type, false);
      for (unsigned i = 0; i < slots && first + i < WVK_MAX_VARYINGS; ++i)
         out[first + i] = varying_code(var);
   }
}

/* Whether a vertex shader's outputs are what the fragment shader reads:
 * there, and interpolated as it has them. */
static bool
varyings_match(const struct wvk_shader *vs, const struct wvk_shader *fs)
{
   if (!fs)
      return true;
   const uint8_t how = WVK_VARYING_FLAT | WVK_VARYING_LINEAR;
   for (unsigned i = 0; i < WVK_MAX_VARYINGS; ++i) {
      if (fs->varyings[i] && (!vs->varyings[i] || (vs->varyings[i] & how) != (fs->varyings[i] & how)))
         return false;
   }
   return true;
}

/* The clip and cull distances among a stage's variables of `mode`, and
 * where each starts in the components of their slots. */
static unsigned
clip_distance_vars(nir_shader *nir, nir_variable_mode mode, nir_variable **vars, unsigned *starts)
{
   unsigned n = 0;
   nir_foreach_variable_with_modes(var, nir, mode) {
      if (n < 2 && var->data.compact && var->data.location >= VARYING_SLOT_CLIP_DIST0 &&
          var->data.location <= VARYING_SLOT_CLIP_DIST1) {
         vars[n] = var;
         starts[n++] = (var->data.location - VARYING_SLOT_CLIP_DIST0) * 4 + var->data.location_frac;
      }
   }
   return n;
}

/* A vertex shader's clip and cull distances also as the fragment shader
 * reads them (key.clip_varying): at the end, their components as vec4s at
 * `location` on, where whatever it had there is no output any more. */
static void
write_clip_varyings(nir_shader *nir, unsigned location, unsigned slots)
{
   nir_variable *vars[2];
   unsigned starts[2];
   const unsigned n = clip_distance_vars(nir, nir_var_shader_out, vars, starts);
   nir_foreach_shader_out_variable(var, nir) {
      if (var->data.location < VARYING_SLOT_VAR0)
         continue;
      const unsigned first = var->data.location - VARYING_SLOT_VAR0;
      if (first < location + slots && location < first + glsl_count_attribute_slots(var->type, false))
         var->data.mode = nir_var_shader_temp;
   }
   nir_fixup_deref_modes(nir);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   for (unsigned j = 0; j < slots; ++j) {
      nir_def *comps[4];
      for (unsigned c = 0; c < 4; ++c) {
         const unsigned k = j * 4 + c;
         comps[c] = nir_imm_float(&b, 0.0f);
         for (unsigned v = 0; v < n; ++v) {
            if (k >= starts[v] && k - starts[v] < glsl_get_length(vars[v]->type))
               comps[c] = nir_load_deref(&b, nir_build_deref_array_imm(&b, nir_build_deref_var(&b, vars[v]),
                                                                        k - starts[v]));
         }
      }
      nir_variable *out = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "wvk_clip_distances");
      out->data.location = VARYING_SLOT_VAR0 + location + j;
      nir_store_var(&b, out, nir_vec(&b, comps, 4), 0xf);
   }
   nir_progress(true, impl, nir_metadata_none);
}

/* A fragment shader's clip and cull distances read from the vec4s at
 * `location` on (key.clip_varying), or none there are 0. */
static void
read_clip_varyings(nir_shader *nir, bool from, unsigned location, unsigned slots)
{
   nir_variable *vars[2];
   unsigned starts[2];
   const unsigned n = clip_distance_vars(nir, nir_var_shader_in, vars, starts);
   if (!n)
      return;
   struct set *lower = _mesa_pointer_set_create(NULL);
   for (unsigned v = 0; v < n; ++v)
      _mesa_set_add(lower, vars[v]);
   NIR_PASS(_, nir, nir_lower_indirect_var_derefs_to_if_else_trees, lower);
   _mesa_set_destroy(lower, NULL);
   nir_variable *ins[2] = { NULL };
   for (unsigned j = 0; from && j < slots; ++j) {
      ins[j] = nir_variable_create(nir, nir_var_shader_in, glsl_vec4_type(), "wvk_clip_distances");
      ins[j]->data.location = VARYING_SLOT_VAR0 + location + j;
   }
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder b = nir_builder_create(impl);
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_deref)
            continue;
         nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
         nir_variable *var = nir_deref_instr_get_variable(deref);
         int v = var == vars[0] ? 0 : n > 1 && var == vars[1] ? 1 : -1;
         if (v < 0 || deref->deref_type != nir_deref_type_array || !nir_src_is_const(deref->arr.index))
            continue;
         const unsigned k = starts[v] + nir_src_as_uint(deref->arr.index);
         b.cursor = nir_before_instr(instr);
         nir_def *value = k / 4 < slots && ins[k / 4] ? nir_channel(&b, nir_load_var(&b, ins[k / 4]), k % 4)
                                                       : nir_imm_float(&b, 0.0f);
         nir_def_replace(&intr->def, value);
      }
   }
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   for (unsigned v = 0; v < n; ++v)
      exec_node_remove(&vars[v]->node);
}

/* A vertex shader's outputs made a fragment shader's inputs: interpolated
 * as it reads them, and of zero where it reads what is not written. */
static void
match_fragment_inputs(nir_shader *vs, const uint8_t inputs[WVK_MAX_VARYINGS])
{
   uint32_t written = 0;
   nir_foreach_shader_out_variable(var, vs) {
      if (var->data.location < VARYING_SLOT_VAR0)
         continue;
      const unsigned first = var->data.location - VARYING_SLOT_VAR0;
      const unsigned slots = glsl_count_attribute_slots(var->type, false);
      for (unsigned i = 0; i < slots && first + i < WVK_MAX_VARYINGS; ++i) {
         written |= BITFIELD_BIT(first + i);
         const uint8_t in = inputs[first + i];
         if (!in || (varying_code(var) & (WVK_VARYING_UINT | WVK_VARYING_SINT)))
            continue;
         var->data.interpolation = in & WVK_VARYING_FLAT     ? INTERP_MODE_FLAT
                                   : in & WVK_VARYING_LINEAR ? INTERP_MODE_NOPERSPECTIVE
                                                             : INTERP_MODE_SMOOTH;
      }
   }
   nir_function_impl *impl = nir_shader_get_entrypoint(vs);
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   bool added = false;
   for (unsigned i = 0; i < WVK_MAX_VARYINGS; ++i) {
      const uint8_t in = inputs[i];
      if (!in || (written & BITFIELD_BIT(i)))
         continue;
      const struct glsl_type *type = in & WVK_VARYING_UINT ? glsl_uvec4_type()
                                     : in & WVK_VARYING_SINT ? glsl_ivec4_type()
                                                             : glsl_vec4_type();
      nir_variable *var = nir_variable_create(vs, nir_var_shader_out, type, "wvk_unwritten");
      var->data.location = VARYING_SLOT_VAR0 + i;
      var->data.interpolation = in & WVK_VARYING_FLAT     ? INTERP_MODE_FLAT
                                : in & WVK_VARYING_LINEAR ? INTERP_MODE_NOPERSPECTIVE
                                                          : INTERP_MODE_SMOOTH;
      nir_store_var(&b, var, nir_imm_zero(&b, 4, 32), 0xf);
      added = true;
   }
   nir_progress(added, impl, nir_metadata_control_flow);
}

/* A fragment shader's depth written nowhere: for a pass without depth. */
static bool
drop_depth_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_out || var->data.location != FRAG_RESULT_DEPTH)
      return false;
   nir_instr_remove(&intr->instr);
   return true;
}

/* A float's conversion to an integer where it is past the integer's range:
 * the integer's largest, as GPUs do. WGSL's saturates at the largest float
 * below it, 2^31 - 128 for i32. */
static bool
saturate_f2i_instr(nir_builder *b, nir_alu_instr *alu, void *data)
{
   if ((alu->op != nir_op_f2i32 && alu->op != nir_op_f2u32) || nir_src_bit_size(alu->src[0].src) != 32)
      return false;
   b->cursor = nir_after_instr(&alu->instr);
   nir_def *x = nir_ssa_for_alu_src(b, alu, 0);
   const bool is_signed = alu->op == nir_op_f2i32;
   nir_def *over = nir_fge_imm(b, x, is_signed ? 2147483648.0 : 4294967296.0);
   nir_def *top = nir_imm_int(b, is_signed ? INT32_MAX : (int32_t)UINT32_MAX);
   nir_def *out = nir_bcsel(b, over, top, &alu->def);
   nir_def_rewrite_uses_after(&alu->def, out);
   return true;
}

/* 64-bit varyings, which WGSL has no types of: each an array of 32-bit
 * words at the same location -- a component a word, two a 64-bit one, and
 * a vector of more than two in two locations, as Vulkan counts them --
 * which its accesses pack and unpack. A vertex shader's inputs are
 * attributes, which it reads its own way. */
struct io64 {
   unsigned comps;     /* 64-bit components of the vector */
   unsigned slots;     /* locations a vector takes */
   unsigned elements;  /* vectors, of arrays and matrix columns */
   bool arrayed;
};

static const struct glsl_type *
io64_word_type(const struct io64 *io)
{
   const unsigned words = io->comps * 2;
   /* One vector in one location: its words; else a vec4 a location. */
   const struct glsl_type *t = glsl_vector_type(GLSL_TYPE_UINT, io->slots == 1 ? words : 4);
   if (io->elements * io->slots > 1)
      t = glsl_array_type(t, io->elements * io->slots, 0);
   return t;
}

/* The words at element `e` of a lowered variable, and its location after
 * where the vector's second half is. */
static nir_deref_instr *
io64_word_deref(nir_builder *b, nir_deref_instr *base, const struct io64 *io, nir_def *e, unsigned half)
{
   if (io->elements * io->slots == 1)
      return base;
   nir_def *at = nir_iadd_imm(b, nir_imul_imm(b, e, io->slots), half);
   return nir_build_deref_array(b, base, at);
}

/* Whether a varying is one lower_64bit_io lowers. */
static bool
io64_candidate(const nir_shader *nir, const nir_variable *var)
{
   if (var->data.location < VARYING_SLOT_VAR0)
      return false;
   const struct glsl_type *t = nir_is_arrayed_io(var, nir->info.stage) ? glsl_get_array_element(var->type)
                                                                       : var->type;
   return glsl_type_is_64bit(glsl_without_array_or_matrix(t)) && !glsl_type_is_struct_or_ifc(glsl_without_array(t));
}

static bool
lower_64bit_io(nir_shader *nir)
{
   const nir_variable_mode modes =
      nir_var_shader_out | (nir->info.stage == MESA_SHADER_VERTEX ? 0 : nir_var_shader_in);
   bool any = false;
   nir_foreach_variable_with_modes(var, nir, modes)
      any |= io64_candidate(nir, var);
   if (!any)
      return false;
   /* Accesses of whole vectors, of the types as they are. */
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_array_deref_of_vec, modes, NULL, nir_lower_direct_array_deref_of_vec_load |
            nir_lower_indirect_array_deref_of_vec_load | nir_lower_direct_array_deref_of_vec_store |
            nir_lower_indirect_array_deref_of_vec_store);

   struct hash_table *vars = _mesa_pointer_hash_table_create(NULL);
   nir_foreach_variable_with_modes(var, nir, modes) {
      if (!io64_candidate(nir, var))
         continue;
      const bool arrayed = nir_is_arrayed_io(var, nir->info.stage);
      const struct glsl_type *t = arrayed ? glsl_get_array_element(var->type) : var->type;
      struct io64 *io = ralloc(vars, struct io64);
      io->arrayed = arrayed;
      io->elements = 1;
      const struct glsl_type *e = t;
      while (glsl_type_is_array(e)) {
         io->elements *= glsl_get_length(e);
         e = glsl_get_array_element(e);
      }
      if (glsl_type_is_matrix(e)) {
         io->elements *= glsl_get_matrix_columns(e);
         e = glsl_get_column_type(e);
      }
      io->comps = glsl_get_vector_elements(e);
      io->slots = var->data.location_frac + io->comps * 2 > 4 ? 2 : 1;
      const struct glsl_type *words = io64_word_type(io);
      var->type = arrayed ? glsl_array_type(words, glsl_get_length(var->type), 0) : words;
      /* As 64-bit ones are, and WGSL's integers must say. */
      var->data.interpolation = INTERP_MODE_FLAT;
      _mesa_hash_table_insert(vars, var, io);
   }

   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_deref && intr->intrinsic != nir_intrinsic_store_deref)
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            nir_variable *var = nir_deref_instr_get_variable(deref);
            struct hash_entry *he = var ? _mesa_hash_table_search(vars, var) : NULL;
            if (!he)
               continue;
            const struct io64 *io = he->data;
            b.cursor = nir_before_instr(instr);
            /* The deref's path: the vertex, then the element, of arrays and
             * a matrix's columns, innermost last. */
            nir_deref_path path;
            nir_deref_path_init(&path, deref, NULL);
            nir_deref_instr *base = nir_build_deref_var(&b, var);
            unsigned i = 1;
            if (io->arrayed) {
               base = nir_build_deref_array(&b, base, path.path[i]->arr.index.ssa);
               ++i;
            }
            nir_def *e = nir_imm_int(&b, 0);
            for (; path.path[i]; ++i) {
               nir_deref_instr *d = path.path[i];
               const struct glsl_type *parent = path.path[i - 1]->type;
               unsigned inner = 1;
               const struct glsl_type *t = glsl_get_array_element(parent);
               while (glsl_type_is_array(t)) {
                  inner *= glsl_get_length(t);
                  t = glsl_get_array_element(t);
               }
               if (glsl_type_is_matrix(t))
                  inner *= glsl_get_matrix_columns(t);
               if (glsl_type_is_matrix(parent))
                  inner = 1;
               e = nir_iadd(&b, e, nir_imul_imm(&b, d->arr.index.ssa, inner));
            }
            nir_deref_path_finish(&path);

            if (intr->intrinsic == nir_intrinsic_load_deref) {
               nir_def *words[8];
               for (unsigned h = 0; h < io->slots; ++h) {
                  nir_def *v = nir_load_deref(&b, io64_word_deref(&b, base, io, e, h));
                  for (unsigned c = 0; c < v->num_components && h * 4 + c < 8; ++c)
                     words[h * 4 + c] = nir_channel(&b, v, c);
               }
               nir_def *comps[4];
               for (unsigned c = 0; c < intr->def.num_components; ++c)
                  comps[c] = nir_pack_64_2x32_split(&b, words[2 * c], words[2 * c + 1]);
               nir_def_replace(&intr->def, nir_vec(&b, comps, intr->def.num_components));
            } else {
               nir_def *value = intr->src[1].ssa;
               const unsigned mask = nir_intrinsic_write_mask(intr);
               nir_def *words[8];
               nir_def *undef = nir_undef(&b, 1, 32);
               for (unsigned w = 0; w < 8; ++w)
                  words[w] = undef;
               for (unsigned c = 0; c < value->num_components; ++c) {
                  words[2 * c] = nir_unpack_64_2x32_split_x(&b, nir_channel(&b, value, c));
                  words[2 * c + 1] = nir_unpack_64_2x32_split_y(&b, nir_channel(&b, value, c));
               }
               for (unsigned h = 0; h < io->slots; ++h) {
                  const unsigned n = io->slots == 1 ? io->comps * 2 : 4;
                  unsigned wmask = 0;
                  for (unsigned c = 0; c < n; ++c) {
                     if (mask & BITFIELD_BIT((h * 4 + c) / 2))
                        wmask |= BITFIELD_BIT(c);
                  }
                  if (wmask)
                     nir_store_deref(&b, io64_word_deref(&b, base, io, e, h), nir_vec(&b, words + h * 4, n), wmask);
               }
               nir_instr_remove(instr);
            }
         }
      }
      nir_progress(true, impl, nir_metadata_control_flow);
   }
   _mesa_hash_table_destroy(vars, NULL);
   NIR_PASS(_, nir, nir_remove_dead_derefs);
   return true;
}

static VkResult
compile_one(struct wvk_device *dev, struct vk_shader_compile_info *info,
            const struct vk_graphics_pipeline_state *pipeline_state, const struct wvk_shader_key *records,
            const VkAllocationCallbacks *pAllocator, struct vk_shader **shader_out)
{
   nir_shader *nir = info->nir;
   /* Compute's, for a vertex shader run as one (run_vertices). */
   mesa_shader_stage stage = nir->info.stage;
   VkResult result = VK_SUCCESS;

   struct wvk_shader *shader = vk_shader_zalloc(&dev->vk, &wvk_shader_ops, stage, pAllocator,
                                                sizeof(*shader));
   if (!shader) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   shader->id = p_atomic_inc_return(&dev->next_shader_id);

   void *mem_ctx = ralloc_context(NULL);
   struct lower_state state = {
      .info = info,
      .split = _mesa_pointer_hash_table_create(mem_ctx),
   };
   /* The records of the pipeline layout's sets, set after set; or where
    * a draw has them, of both its stages' layouts. */
   uint32_t meta_words = 0;
   for (uint32_t g = 0; g < 4; ++g) {
      shader->meta_base[g] = meta_words;
      shader->set_words[g] = -1;
      if (g < info->set_layout_count && info->set_layouts[g]) {
         const struct wvk_descriptor_set_layout *layout =
            container_of(info->set_layouts[g], struct wvk_descriptor_set_layout, vk);
         shader->set_words[g] = layout->slot_count * WVK_META_WORDS;
         meta_words += layout->slot_count * WVK_META_WORDS;
      }
   }
   shader->meta_words = meta_words;
   if (records) {
      memcpy(shader->meta_base, records->meta_base, sizeof(shader->meta_base));
      shader->meta_words = meta_words = records->meta_words;
   }
   memcpy(shader->key.meta_base, shader->meta_base, sizeof(shader->meta_base));
   shader->key.meta_words = shader->meta_words;
   if (WVK_META_OFFSET + meta_words * 4 > WVK_PUSH_WINDOW)
      mesa_logw_once("wvk: a pipeline layout of more descriptors than their records have room for");

   /* Varyings of a location each, and none 64-bit. */
   if (stage != MESA_SHADER_COMPUTE) {
      NIR_PASS(_, nir, split_io_structs);
      NIR_PASS(_, nir, lower_64bit_io);
   }

   /* What transform feedback captures, which the driver's pass does
    * (capture_xfb): no shader's, nor any compute shader's to have. */
   if (nir->info.stage == MESA_SHADER_VERTEX)
      gather_xfb(shader, nir);
   nir->xfb_info = NULL;
   /* And the outputs are no transform feedback's, so that those packed in
    * a location may be one vector of it (nir_opt_vectorize_io_vars). */
   nir_foreach_shader_out_variable(var, nir) {
      var->data.explicit_xfb_buffer = false;
      var->data.explicit_xfb_stride = false;
      var->data.explicit_offset = false;
   }
   NIR_PASS(_, nir, lower_push_constants);
   /* Tessellation's stages as compute, for the records of the stage before. */
   unsigned workgroup = 64;
   if (nir->info.stage == MESA_SHADER_TESS_CTRL && records && records->tess == 1) {
      shader->key.tess = 1;
      shader->key.vs_vars = records->vs_vars;
      workgroup = MAX2(nir->info.tess.tcs_vertices_out, 1);
      lower_tcs(nir, shader, records->vs_vars);
   } else if (nir->info.stage == MESA_SHADER_TESS_EVAL && records && records->tess == 1) {
      shader->key.tess = 1;
      shader->key.vs_vars = records->vs_vars;
      shader->key.patch_vars = records->patch_vars;
      shader->key.patch_vertices = records->patch_vertices;
      lower_tes(nir, shader, records->vs_vars, records->patch_vars, records->patch_vertices);
   } else if (records && records->tess == 2) {
      shader->key.tess = 2;
   }
   /* A geometry shader as compute, for its draw's vertex shader's records. */
   if (nir->info.stage == MESA_SHADER_GEOMETRY && records && records->geometry == 1) {
      shader->key.geometry = 1;
      shader->key.vs_vars = records->vs_vars;
      shader->key.after_tess = records->after_tess;
      shader->key.stats = records->stats;
      shader->key.raster_stream = records->raster_stream;
      shader->key.provoking_last = records->provoking_last;
      lower_geometry(nir, shader, records->vs_vars, records->after_tess, records->stats, records->raster_stream,
                     records->provoking_last);
   } else if (records && records->geometry == 2) {
      shader->key.geometry = 2;
   }
   /* Where varyings are sampled is the fragment shader's alone: at the
    * centre, as far as WebGPU's matching of the stages goes. */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_foreach_shader_out_variable(var, nir) {
         var->data.sample = false;
         var->data.centroid = false;
      }
      /* The attributes read, before pulling makes some of them reads of
       * buffers: what a draw's vertex input is looked at for. */
      nir_foreach_shader_in_variable(var, nir) {
         const int first = var->data.location - VERT_ATTRIB_GENERIC0;
         const unsigned slots = glsl_count_attribute_slots(var->type, true);
         for (unsigned i = 0; first >= 0 && i < slots && first + i < 32; ++i)
            shader->attribs_read |= BITFIELD_BIT(first + i);
      }
      /* A draw of points as quads (expand_points) pulls every attribute. */
      shader->writes_point_size = nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_PSIZ);
      if (records && records->points) {
         shader->key.points = 1;
         shader->key.point_coord = records->point_coord;
         expand_points(nir, records->point_coord);
      }
      /* Storage writes, which its compute shader does (run_vertices), and
       * the draw's vertex shader has not. */
      nir_shader_intrinsics_pass(nir, writes_storage_instr, nir_metadata_all, &shader->writes_storage);
      if (records && records->lines) {
         shader->key.lines = 1;
         shader->key.line_coord = records->line_coord;
         expand_lines(nir, records->line_coord);
      }
      /* WebGPU has no cull distances: a draw with them culls in its
       * geometry shader (lower_geometry), which is the records'. */
      nir_variable *cull = cull_distance_var(nir, nir_var_shader_out);
      if (cull) {
         shader->cull_distances = MIN2(nir->info.cull_distance_array_size, 8);
         if (!records || !records->runs) {
            cull->data.mode = nir_var_shader_temp;
            nir_fixup_deref_modes(nir);
            NIR_PASS(_, nir, nir_lower_global_vars_to_local);
         }
      }
      if (records && records->runs) {
         shader->key.runs = records->runs;
         /* Capturing positions only: the writes are the other's. */
         if (records->runs == 2 && shader->writes_storage)
            NIR_PASS(_, nir, nir_shader_intrinsics_pass, strip_storage_writes_instr, nir_metadata_control_flow,
                     NULL);
         run_vertices(nir, shader, records->runs);
      } else if (shader->writes_storage) {
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, strip_storage_writes_instr, nir_metadata_control_flow, NULL);
      }
      shader->key.pulled = pull_key(pipeline_state ? pipeline_state->vi : NULL, shader->attribs_read,
                                    shader->key.points || shader->key.runs || shader->key.lines);
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT && records && records->color_remapped) {
      NIR_PASS(_, nir, remap_color_outputs, records->color_map);
      shader->key.color_remapped = 1;
      memcpy(shader->key.color_map, records->color_map, sizeof(records->color_map));
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      const struct vk_multisample_state *ms = pipeline_state ? pipeline_state->ms : NULL;
      bool own_mask = false;
      nir_foreach_shader_out_variable(var, nir)
         own_mask |= var->data.location == FRAG_RESULT_SAMPLE_MASK;
      /* WebGPU has no alpha to coverage of one sample either. */
      const bool to_one = ms && ms->alpha_to_one_enable;
      const bool by_alpha = ms && ms->alpha_to_coverage_enable &&
                            (to_one || own_mask || ms->rasterization_samples <= 1);
      if (to_one || by_alpha)
         NIR_PASS(_, nir, lower_alpha_to_one, to_one, by_alpha, ms->rasterization_samples);
      shader->covers_by_alpha = by_alpha;
      shader->own_mask = own_mask;
      shader->key.alpha_to_one = to_one;
      shader->key.by_alpha = by_alpha;
      /* What of a point it reads, which a point drawn as a quad does not
       * give it: then from the vertex shader, for a draw's variant. */
      nir_shader_intrinsics_pass(nir, point_state_instr, nir_metadata_all, &shader->point_state);
      if (nir_find_variable_with_location(nir, nir_var_shader_in, VARYING_SLOT_PRIMITIVE_ID))
         shader->point_state |= 4;
      /* Its invocations, for the pipeline statistics query, first thing:
       * a helper's add is none. */
      if (records && records->stats) {
         nir_function_impl *impl = nir_shader_get_entrypoint(nir);
         nir_builder b = nir_builder_at(nir_before_impl(impl));
         nir_ssbo_atomic(&b, 32, driver_buffer(&b, WVK_DRIVER_SLOT_STATS),
                         nir_imm_int(&b, util_logbase2(VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT) * 4),
                         nir_imm_int(&b, 1), .atomic_op = nir_atomic_op_iadd);
         nir_progress(true, impl, nir_metadata_none);
         shader->key.stats = 1;
      }
      /* Its clip and cull distances: 0, but where a draw has them as
       * varyings (a variant). */
      shader->clip_inputs = MIN2(nir->info.clip_distance_array_size + nir->info.cull_distance_array_size, 8);
      if (shader->clip_inputs) {
         const bool from = records && records->clip_varying;
         read_clip_varyings(nir, from, from ? records->clip_varying - 1 : 0, from ? records->clip_slots : 0);
         if (from) {
            shader->key.clip_varying = records->clip_varying;
            shader->key.clip_slots = records->clip_slots;
         }
      }
      if (records && records->primitive_id) {
         NIR_PASS(_, nir, lower_fs_primitive_id, records->primitive_id - 1);
         shader->key.primitive_id = records->primitive_id;
      } else if (shader->point_state & 4) {
         /* WebGPU's fragment shaders have no primitive ID: 0, but after a
          * geometry shader, which gives it (a variant). */
         NIR_PASS(_, nir, zero_fs_primitive_id);
      }
      if (records && records->points) {
         NIR_PASS(_, nir, lower_point_fs, records->point_coord);
         shader->key.points = 1;
         shader->key.point_coord = records->point_coord;
      }
      if (records && records->line_coord) {
         NIR_PASS(_, nir, lower_line_fs, records->line_coord);
         shader->key.line_coord = records->line_coord;
      }
      /* A draw's depth bounds test and logic op, for its variant. */
      if (records && records->depth_bounds) {
         NIR_PASS(_, nir, lower_depth_bounds);
         shader->reads_depth_bounds = true;
         shader->key.depth_bounds = 1;
      }
      if (records && records->logic_op) {
         NIR_PASS(_, nir, lower_logic_op, records, &shader->logic_op_reads);
         shader->key.logic_op = records->logic_op;
         memcpy(shader->key.logic_format, records->logic_format, sizeof(records->logic_format));
      }
      if (records && records->blend_mask) {
         NIR_PASS(_, nir, lower_blend, records, &shader->logic_op_reads);
         shader->key.blend_mask = records->blend_mask;
         memcpy(shader->key.blend_equation, records->blend_equation, sizeof(records->blend_equation));
         memcpy(shader->key.logic_format, records->logic_format, sizeof(records->logic_format));
      }
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* Depth dropped where the pass has none: the pipeline's, or the
       * draw's for a variant. */
      nir_foreach_shader_out_variable(var, nir)
         shader->source_writes_depth |= var->data.location == FRAG_RESULT_DEPTH;
      /* A render pass without depth is one that says what it has: a
       * fragment shader library's knows nothing of its attachments. */
      const struct vk_render_pass_state *rp = pipeline_state ? pipeline_state->rp : NULL;
      const bool no_depth = records ? records->no_depth
                                    : rp && vk_render_pass_state_has_attachment_info(rp) &&
                                         rp->depth_attachment_format == VK_FORMAT_UNDEFINED &&
                                         rp->stencil_attachment_format == VK_FORMAT_UNDEFINED;
      if (no_depth && shader->source_writes_depth) {
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, drop_depth_instr, nir_metadata_control_flow, NULL);
         /* The derefs of it go with the stores, and then the variable. */
         NIR_PASS(_, nir, nir_opt_dce);
         nir_foreach_shader_out_variable_safe(var, nir) {
            if (var->data.location == FRAG_RESULT_DEPTH)
               exec_node_remove(&var->node);
         }
         shader->key.no_depth = true;
      }
   }
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      const bool per_sample = wants_sample_shading(pipeline_state);
      if (per_sample)
         NIR_PASS(_, nir, shade_per_sample);
      struct sample_state s = {
         .samples = pipeline_state && pipeline_state->ms ? pipeline_state->ms->rasterization_samples : 4,
         .frag_coord = per_sample,
      };
      bool placed = false;
      NIR_PASS(placed, nir, nir_shader_intrinsics_pass, lower_sample_pos_instr, nir_metadata_control_flow, &s);
      NIR_PASS(placed, nir, nir_shader_intrinsics_pass, lower_interp_at_instr, nir_metadata_control_flow, &s);
      NIR_PASS(placed, nir, lower_input_sampling, &s);
      shader->per_sample = per_sample;
      shader->uses_samples = placed;
      /* The coverage from alpha is of the samples too. */
      shader->key.samples = placed || shader->key.by_alpha ? s.samples : 0;
   }
   /* Shared memory as words, as ntv's is, and accessed a word at a time:
    * a 64-bit or 16-bit value is words' worth, which WGSL has no types of. */
   if (nir->info.stage == MESA_SHADER_COMPUTE) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
      /* In words, which the size is made a whole number of: ntv's block is
       * of them, and a block of one 16-bit value (an explicit layout's,
       * workgroupMemoryExplicitLayout16BitAccess) would be none. */
      nir->info.shared_size = align(nir->info.shared_size, 4);
      /* A null initializer (shaderZeroInitializeWorkgroupMemory): stores of
       * zero, and the variable gone, which ntv would lay out again. */
      bool initialized = false;
      nir_foreach_variable_with_modes(var, nir, nir_var_mem_shared) {
         initialized |= var->constant_initializer != NULL || var->pointer_initializer != NULL;
         var->constant_initializer = NULL;
         var->pointer_initializer = NULL;
      }
      if (initialized)
         NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size, 4);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_mem_shared, NULL);
      /* Stores of less than a word, atomics on the bits of it. */
      const nir_lower_mem_access_bit_sizes_options shared = {
         .callback = shared_access,
         .modes = nir_var_mem_shared,
         .may_lower_unaligned_stores_to_atomics = true,
      };
      NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &shared);
   }
   /* Varyings WGSL can have: no structs, and a location each; and flat
    * integers, which a vertex shader's need not say. */
   NIR_PASS(_, nir, split_io_structs);
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_foreach_shader_out_variable(var, nir) {
         if (var->data.location >= VARYING_SLOT_VAR0 &&
             glsl_base_type_is_integer(glsl_get_base_type(glsl_without_array_or_matrix(var->type))))
            var->data.interpolation = INTERP_MODE_FLAT;
      }
   }
   NIR_PASS(_, nir, nir_opt_vectorize_io_vars,
            nir->info.stage == MESA_SHADER_VERTEX ? nir_var_shader_out
            : nir->info.stage == MESA_SHADER_FRAGMENT ? nir_var_shader_in : 0);
   NIR_PASS(_, nir, lower_layer_outputs, shader);
   bool viewport_read = false;
   NIR_PASS(viewport_read, nir, nir_shader_intrinsics_pass, lower_layer_id_instr, nir_metadata_control_flow, NULL);
   /* gl_ViewportIndex, read from the driver's data: no input of its own. */
   if (viewport_read && nir->info.stage == MESA_SHADER_FRAGMENT) {
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_in, NULL);
   }
   if (nir->info.stage == MESA_SHADER_COMPUTE)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_base_workgroup_instr, nir_metadata_control_flow, NULL);
   split_resource_vars(nir, &state);
   split_buffer_vars(nir, &state);
   /* Writes by a dynamic index are ladders of ifs. */
   NIR_PASS(_, nir, nir_shader_instructions_pass, rewrite_resource_instr, nir_metadata_none, &state);
   NIR_PASS(_, nir, nir_shader_instructions_pass, integer_cube_query_instr, nir_metadata_control_flow, NULL);
   struct util_dynarray bgra;
   util_dynarray_init(&bgra, mem_ctx);
   format_storage_images(nir, shader, records, &bgra);
   if (nir->info.stage == MESA_SHADER_VERTEX && shader->writes_storage && shader->key.runs != 1)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, strip_storage_writes_instr, nir_metadata_control_flow,
               (void *)1);
   bool retyped = false;
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, match_image_type_instr, nir_metadata_all, &retyped);
   if (retyped)
      nir_fixup_deref_types(nir);
   /* Out of bounds, and of null views: a texel replaced before the
    * swizzle takes it apart, as a fetched one is. */
   const struct vk_pipeline_robustness_state *robustness = info->robustness;
   NIR_PASS(_, nir, wvk_nir_lower_robust_images, shader->meta_base,
            robustness && robustness->images == VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS_2,
            dev->vk.enabled_features.nullDescriptor);
   /* Swizzles first: a gather's is a gather of another component, which
    * then samples as its own; and the border replaces a texel before the
    * swizzle takes it apart. */
   NIR_PASS(_, nir, wvk_nir_lower_view_swizzles, shader->meta_base);
   /* Projection divided out first: the samplers' lowering takes the
    * coordinates as they are. And a gather at offsets WGSL has none of --
    * four, or one not constant -- is gathers each at one, or at the
    * coordinates moved; as is any sampling or fetch at an offset not
    * constant (VK_KHR_maintenance8). */
   const nir_lower_tex_options txp = { .lower_txp = ~0u };
   NIR_PASS(_, nir, nir_lower_tex, &txp);
   NIR_PASS(_, nir, nir_shader_instructions_pass, lower_lod_offset_instr, nir_metadata_control_flow, NULL);
   const nir_lower_tex_options offsets = {
      .lower_tg4_offsets = true,
      .lower_offset_filter = dynamic_offset,
   };
   NIR_PASS(_, nir, nir_lower_tex, &offsets);
   NIR_PASS(_, nir, wvk_nir_lower_samplers, shader->meta_base);
   NIR_PASS(_, nir, lower_1d_images);
   NIR_PASS(_, nir, lower_ms_arrays);
   NIR_PASS(_, nir, pad_io_vectors);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, demote_to_terminate_instr, nir_metadata_all, NULL);
   NIR_PASS(_, nir, wvk_nir_lower_image_atomics, shader->meta_base);
   NIR_PASS(_, nir, wvk_nir_lower_texel_buffers, shader->meta_base);
   lower_buffers_to_words(nir, info, pipeline_state, shader, mem_ctx);
   /* A vertex, geometry or tessellation shader run as compute is compute's
    * from here. */
   if (shader->key.runs || shader->key.geometry == 1 || shader->key.tess == 1) {
      nir->info.stage = stage = MESA_SHADER_COMPUTE;
      nir->info.workgroup_size[0] = workgroup;
      nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
      nir->info.workgroup_size_variable = false;
      shader->vk.stage = MESA_SHADER_COMPUTE;
      /* Nothing is left of its interface, but what it could not pull. */
      NIR_PASS(_, nir, nir_opt_dce);
      NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_shader_in | nir_var_shader_out, NULL);
      if (!exec_list_is_empty(&nir->variables)) {
         nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out) {
            result = vk_errorf(dev, VK_ERROR_UNKNOWN, "a shader run as compute reads attributes it cannot pull");
            goto fail;
         }
      }
   }
   /* The signed atomics' loops' variables. */
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   optimize(nir);
   /* What is left of 8 and 16 bits, words: memory is by now. */
   bool widened = false;
   NIR_PASS(widened, nir, wvk_nir_widen_small_types);
   if (widened)
      optimize(nir);
   /* And what the optimizations made of 64-bit floats since: a subtraction
    * an addition of the negation, say. */
   lower_float64(wvk_device_physical(dev), nir);
   NIR_PASS(_, nir, nir_shader_alu_pass, saturate_f2i_instr, nir_metadata_control_flow, NULL);
   /* The combined and arrayed originals, now without a use. */
   hash_table_foreach(state.split, he)
      exec_node_remove(&((nir_variable *)he->key)->node);
   if (state.failed) {
      result = vk_errorf(dev, VK_ERROR_UNKNOWN, "a shader's descriptors do not fit its layout");
      goto fail;
   }
   if (!reflect_bindings(nir, info, shader, mem_ctx)) {
      result = vk_errorf(dev, VK_ERROR_UNKNOWN, "a shader uses what WebGPU cannot bind");
      goto fail;
   }
   util_dynarray_foreach(&bgra, uint32_t, at) {
      for (uint32_t i = 0; i < shader->binding_count; ++i) {
         struct wvk_shader_binding *b = &shader->bindings[i];
         if (b->kind == WVK_BINDING_STORAGE_TEXTURE && b->group == *at >> 16 && b->binding == (*at & 0xffff))
            b->format = WGPUTextureFormat_BGRA8Unorm;
      }
   }

   /* The compiler keeps a vertex input's driver_location as its WGSL
    * @location, which for Vulkan is the attribute location. */
   if (stage == MESA_SHADER_VERTEX) {
      nir_foreach_shader_in_variable(var, nir) {
         if (var->data.location >= VERT_ATTRIB_GENERIC0)
            var->data.driver_location = var->data.location - VERT_ATTRIB_GENERIC0;
      }
   }

   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   if (stage == MESA_SHADER_VERTEX)
      shader->io_mask = (uint32_t)(nir->info.inputs_read >> VERT_ATTRIB_GENERIC0);
   if (stage == MESA_SHADER_FRAGMENT) {
      shader->io_mask = (uint32_t)(nir->info.outputs_written >> FRAG_RESULT_DATA0);
      shader->writes_depth = nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH);
   }
   if (stage == MESA_SHADER_COMPUTE) {
      for (unsigned i = 0; i < 3; ++i)
         shader->workgroup_size[i] = nir->info.workgroup_size[i];
   }

   if (stage == MESA_SHADER_VERTEX)
      record_varyings(nir, nir_var_shader_out, shader->varyings);
   else if (stage == MESA_SHADER_FRAGMENT)
      record_varyings(nir, nir_var_shader_in, shader->varyings);

   if (getenv("WINK_VK_DUMP_NIR"))
      nir_print_shader(nir, stderr);

   /* A vertex shader twice: as Vulkan's y needs, and flipped back for a
    * viewport of negative height. */
   nir_shader *unflipped = stage == MESA_SHADER_VERTEX ? nir_shader_clone(NULL, nir) : NULL;
   struct gex_wgpu_wgsl_info wi;
   char *error = NULL;
   shader->wgsl[WVK_SHADER_FLIPPED] =
      gex_wgpu_nir_to_wgsl(nir, stage, stage == MESA_SHADER_VERTEX, true, &wi, &error);
   nir = NULL;
   if (!shader->wgsl[WVK_SHADER_FLIPPED]) {
      ralloc_free(unflipped);
      mesa_loge("wvk: %s shader compile failed: %s", _mesa_shader_stage_to_string(stage),
                error ? error : "?");
      result = vk_errorf(dev, VK_ERROR_UNKNOWN, "shader compile failed: %s", error ? error : "?");
      free(error);
      goto fail;
   }
   if (unflipped) {
      struct gex_wgpu_wgsl_info wi2;
      shader->wgsl[WVK_SHADER_UNFLIPPED] =
         gex_wgpu_nir_to_wgsl(unflipped, stage, false, true, &wi2, &error);
      if (!shader->wgsl[WVK_SHADER_UNFLIPPED]) {
         result = vk_errorf(dev, VK_ERROR_UNKNOWN, "shader compile failed: %s", error ? error : "?");
         free(error);
         goto fail;
      }
   }
   util_dynarray_foreach(&bgra, uint32_t, at) {
      for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
         declare_bgra_storage(shader->wgsl[i], *at >> 16, *at & 0xffff);
   }
   if (getenv("WINK_VK_DUMP_WGSL"))
      fprintf(stderr, "wvk: %s WGSL:\n%s\n", wi.label, shader->wgsl[WVK_SHADER_FLIPPED]);

   for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i) {
      if (shader->wgsl[i] && (result = make_module(dev, shader->wgsl[i], wi.label, &shader->module[i])))
         goto fail;
   }

   ralloc_free(mem_ctx);
   *shader_out = &shader->vk;
   return VK_SUCCESS;

fail:
   ralloc_free(nir);
   ralloc_free(mem_ctx);
   wvk_shader_destroy(&dev->vk, &shader->vk, pAllocator);
   return result;
}

/* WebGPU wants a vertex output interpolated and sampled as the fragment
 * input it feeds is, where Vulkan goes by the fragment shader alone. A
 * pipeline's stages are compiled together, and hashed together, so the
 * vertex shader takes the fragment shader's. */
static void
link_interpolation(nir_shader *producer, nir_shader *fs)
{
   nir_foreach_shader_out_variable(out, producer) {
      if (out->data.location < VARYING_SLOT_VAR0)
         continue;
      const unsigned out_end = out->data.location + glsl_count_attribute_slots(out->type, false);
      nir_foreach_shader_in_variable(in, fs) {
         const unsigned in_end = in->data.location + glsl_count_attribute_slots(in->type, false);
         if (in->data.location >= out_end || out->data.location >= in_end)
            continue;
         out->data.interpolation = in->data.interpolation;
         out->data.centroid = in->data.centroid;
         out->data.sample = in->data.sample;
      }
   }
}

/* A geometry or tessellation shader, which is its variants alone
 * (key.geometry, key.tess): what they are made of, its source, and what the
 * draw needs to know of it. */
static VkResult
geometry_shell(struct wvk_device *dev, struct vk_shader_compile_info *info, const VkAllocationCallbacks *pAllocator,
               struct vk_shader **shader_out)
{
   nir_shader *nir = info->nir;
   struct wvk_shader *shader = vk_shader_zalloc(&dev->vk, &wvk_shader_ops, nir->info.stage, pAllocator,
                                                sizeof(*shader));
   if (!shader) {
      ralloc_free(nir);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   shader->id = p_atomic_inc_return(&dev->next_shader_id);
   uint32_t meta_words = 0;
   for (uint32_t g = 0; g < 4; ++g) {
      shader->meta_base[g] = meta_words;
      shader->set_words[g] = -1;
      if (g < info->set_layout_count && info->set_layouts[g]) {
         const struct wvk_descriptor_set_layout *layout =
            container_of(info->set_layouts[g], struct wvk_descriptor_set_layout, vk);
         shader->set_words[g] = layout->slot_count * WVK_META_WORDS;
         meta_words += layout->slot_count * WVK_META_WORDS;
      }
   }
   shader->meta_words = meta_words;
   memcpy(shader->key.meta_base, shader->meta_base, sizeof(shader->meta_base));
   shader->key.meta_words = meta_words;
   if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      shader->gs_input = nir->info.gs.input_primitive;
      shader->gs_output = nir->info.gs.output_primitive;
      shader->gs_max_vertices = nir->info.gs.vertices_out;
      shader->gs_invocations = MAX2(nir->info.gs.invocations, 1);
      shader->gs_streams = wvk_geometry_streams(nir);
   } else {
      shader->tess_vertices_out = nir->info.tess.tcs_vertices_out;
      shader->tess_domain = nir->info.tess._primitive_mode;
      shader->tess_spacing = nir->info.tess.spacing;
      shader->tess_ccw = nir->info.tess.ccw;
      shader->tess_point_mode = nir->info.tess.point_mode;
      shader->patch_vars = patch_vars(nir, nir_var_shader_out);
   }
   shader->record_vars = record_vars(nir, nir_var_shader_out);
   if (nir->info.stage != MESA_SHADER_TESS_CTRL)
      gather_xfb(shader, nir);
   if (nir->info.stage != MESA_SHADER_TESS_CTRL && cull_distance_var(nir, nir_var_shader_out))
      shader->cull_distances = MIN2(nir->info.cull_distance_array_size, 8);
   /* Points it gives a size, which its draw's are quads of. */
   shader->writes_point_size = nir->info.stage != MESA_SHADER_TESS_CTRL &&
                               nir_find_variable_with_location(nir, nir_var_shader_out, VARYING_SLOT_PSIZ);
   ralloc_free(nir);
   *shader_out = &shader->vk;
   return VK_SUCCESS;
}

/* A varying copied, down to its vectors. */
static void
copy_varying(nir_builder *b, nir_deref_instr *dst, nir_deref_instr *src)
{
   const struct glsl_type *type = dst->type;
   if (glsl_type_is_vector_or_scalar(type)) {
      nir_store_deref(b, dst, nir_load_deref(b, src), ~0);
      return;
   }
   for (unsigned i = 0; i < glsl_get_length(type); ++i) {
      if (glsl_type_is_struct_or_ifc(type))
         copy_varying(b, nir_build_deref_struct(b, dst, i), nir_build_deref_struct(b, src, i));
      else
         copy_varying(b, nir_build_deref_array_imm(b, dst, i), nir_build_deref_array_imm(b, src, i));
   }
}

/* A geometry shader that emits the primitive it is given, for the cull
 * distances of the stage before: of its outputs, in and out. */
static nir_shader *
cull_geometry_nir(const nir_shader *last, enum mesa_prim input)
{
   static const unsigned lists[5][3] = { { 0 }, { 0, 1 }, { 0, 1, 2 }, { 1, 2 }, { 0, 2, 4 } };
   unsigned k, emits;
   enum mesa_prim output;
   switch (input) {
   case MESA_PRIM_POINTS: k = 0, emits = 1, output = MESA_PRIM_POINTS; break;
   case MESA_PRIM_LINES: k = 1, emits = 2, output = MESA_PRIM_LINE_STRIP; break;
   case MESA_PRIM_TRIANGLES: k = 2, emits = 3, output = MESA_PRIM_TRIANGLE_STRIP; break;
   case MESA_PRIM_LINES_ADJACENCY: k = 3, emits = 2, output = MESA_PRIM_LINE_STRIP; break;
   case MESA_PRIM_TRIANGLES_ADJACENCY: k = 4, emits = 3, output = MESA_PRIM_TRIANGLE_STRIP; break;
   default: return NULL;
   }
   const unsigned n = mesa_vertices_per_prim(input);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY, last->options, "wvk cull distances");
   nir_shader *gs = b.shader;
   gs->info.gs.input_primitive = input;
   gs->info.gs.output_primitive = output;
   gs->info.gs.vertices_in = n;
   gs->info.gs.vertices_out = emits;
   gs->info.gs.invocations = 1;
   gs->info.gs.active_stream_mask = 1;
   struct util_dynarray pairs;
   util_dynarray_init(&pairs, NULL);
   nir_foreach_shader_out_variable(var, (nir_shader *)last) {
      nir_variable *out = nir_variable_clone(var, gs);
      nir_shader_add_variable(gs, out);
      nir_variable *in = nir_variable_clone(var, gs);
      in->data.mode = nir_var_shader_in;
      in->type = glsl_array_type(var->type, n, 0);
      nir_shader_add_variable(gs, in);
      util_dynarray_append(&pairs, in);
      util_dynarray_append(&pairs, out);
   }
   nir_variable **vars = pairs.data;
   const unsigned count = util_dynarray_num_elements(&pairs, nir_variable *) / 2;
   for (unsigned v = 0; v < emits; ++v) {
      for (unsigned i = 0; i < count; ++i)
         copy_varying(&b, nir_build_deref_var(&b, vars[2 * i + 1]),
                      nir_build_deref_array_imm(&b, nir_build_deref_var(&b, vars[2 * i]), lists[k][v]));
      nir_emit_vertex(&b, 0);
   }
   nir_end_primitive(&b, 0);
   util_dynarray_fini(&pairs);
   nir_shader_gather_info(gs, nir_shader_get_entrypoint(gs));
   /* What transform feedback captures is the stage before's. */
   if (last->xfb_info) {
      const size_t size = nir_xfb_info_size(last->xfb_info->output_count);
      gs->xfb_info = ralloc_size(gs, size);
      memcpy(gs->xfb_info, last->xfb_info, size);
   }
   gs->info.clip_distance_array_size = last->info.clip_distance_array_size;
   gs->info.cull_distance_array_size = last->info.cull_distance_array_size;
   return gs;
}

struct wvk_shader *
wvk_shader_cull_geometry(struct wvk_device *dev, struct wvk_shader *last, unsigned input)
{
   const int k = input == MESA_PRIM_POINTS            ? 0
                 : input == MESA_PRIM_LINES           ? 1
                 : input == MESA_PRIM_TRIANGLES       ? 2
                 : input == MESA_PRIM_LINES_ADJACENCY ? 3
                 : input == MESA_PRIM_TRIANGLES_ADJACENCY ? 4
                                                         : -1;
   if (k < 0 || !last || !last->source)
      return NULL;
   struct wvk_shader_source *src = last->source;
   simple_mtx_lock(&src->lock);
   if (!last->cull_gs[k]) {
      struct blob_reader reader;
      blob_reader_init(&reader, src->nir.data, src->nir.size);
      nir_shader *nir = nir_deserialize(NULL, &wvk_device_physical(dev)->nir_options, &reader);
      nir_shader *gs = nir ? cull_geometry_nir(nir, (enum mesa_prim)input) : NULL;
      ralloc_free(nir);
      struct vk_shader_compile_info info = {
         .stage = MESA_SHADER_GEOMETRY,
         .nir = gs,
      };
      struct wvk_shader_source *gs_src = gs ? source_create(&info) : NULL;
      struct vk_shader *shell = NULL;
      if (gs_src && geometry_shell(dev, &info, NULL, &shell) == VK_SUCCESS) {
         wvk_shader_from_vk(shell)->source = gs_src;
         last->cull_gs[k] = wvk_shader_from_vk(shell);
      } else {
         if (gs_src)
            source_destroy(dev, gs_src);
         else
            ralloc_free(gs);
         mesa_logw_once("wvk: cull distances, whose geometry shader could not be made");
      }
   }
   simple_mtx_unlock(&src->lock);
   return last->cull_gs[k];
}

/* Whether a shader may write a storage image without a format, which its
 * draws and dispatches compile variants for (format_storage_images). */
static bool
has_unformatted_images(const nir_shader *nir)
{
   nir_foreach_variable_with_modes(var, nir, nir_var_image) {
      if (var->data.image.format == PIPE_FORMAT_NONE && !(var->data.access & ACCESS_NON_WRITEABLE))
         return true;
   }
   return false;
}

/* The YCbCr conversion of an immutable sampler of the layout's, which
 * sampling through is a sample of each plane (nir_vk_lower_ycbcr_tex). */
static const struct vk_ycbcr_conversion_state *
ycbcr_conversion_lookup(const void *data, uint32_t set, uint32_t binding, uint32_t array_index)
{
   const struct vk_shader_compile_info *info = data;
   if (set >= info->set_layout_count || !info->set_layouts[set])
      return NULL;
   const struct wvk_descriptor_set_layout *layout =
      container_of(info->set_layouts[set], struct wvk_descriptor_set_layout, vk);
   if (binding >= layout->binding_count)
      return NULL;
   const struct wvk_descriptor_set_binding_layout *bl = &layout->bindings[binding];
   if (!bl->immutable_samplers || array_index >= bl->count || !bl->immutable_samplers[array_index])
      return NULL;
   const struct vk_ycbcr_conversion *conversion = bl->immutable_samplers[array_index]->vk.ycbcr_conversion;
   return conversion ? &conversion->state : NULL;
}

static VkResult
wvk_compile(struct vk_device *vk_dev, uint32_t shader_count, struct vk_shader_compile_info *infos,
            const struct vk_graphics_pipeline_state *state, const struct vk_features *enabled_features,
            const VkAllocationCallbacks *pAllocator, struct vk_shader **shaders_out)
{
   struct wvk_device *dev = container_of(vk_dev, struct wvk_device, vk);
   nir_shader *vs = NULL, *fs = NULL;
   /* The sources first, as the application's shaders are: variants are
    * of those, linked to whatever they are drawn with. */
   struct wvk_shader_source *sources[MESA_VK_MAX_GRAPHICS_PIPELINE_STAGES] = { NULL };
   for (uint32_t i = 0; i < shader_count; ++i) {
      /* Sampling through YCbCr conversions, a plane at a time, before the
       * sources keep it: their layouts have no samplers. */
      NIR_PASS(_, infos[i].nir, nir_vk_lower_ycbcr_tex, ycbcr_conversion_lookup, &infos[i]);
      if (infos[i].stage == MESA_SHADER_VERTEX)
         vs = infos[i].nir;
      else if (infos[i].stage == MESA_SHADER_FRAGMENT)
         fs = infos[i].nir;
      /* And a compute shader's, where its variants are of the formats of
       * storage images it has none for. */
      if ((infos[i].stage == MESA_SHADER_VERTEX || infos[i].stage == MESA_SHADER_FRAGMENT ||
           infos[i].stage == MESA_SHADER_GEOMETRY || infos[i].stage == MESA_SHADER_TESS_CTRL ||
           infos[i].stage == MESA_SHADER_TESS_EVAL || has_unformatted_images(infos[i].nir)) &&
          i < ARRAY_SIZE(sources))
         sources[i] = source_create(&infos[i]);
   }
   if (vs && fs)
      link_interpolation(vs, fs);
   for (uint32_t i = 0; i < shader_count; ++i) {
      VkResult result = infos[i].stage == MESA_SHADER_GEOMETRY || infos[i].stage == MESA_SHADER_TESS_CTRL ||
                                 infos[i].stage == MESA_SHADER_TESS_EVAL
                           ? geometry_shell(dev, &infos[i], pAllocator, &shaders_out[i])
                           : compile_one(dev, &infos[i], state, NULL, pAllocator, &shaders_out[i]);
      if (result != VK_SUCCESS) {
         /* compile consumes every NIR, whatever happens. */
         for (uint32_t j = i + 1; j < shader_count; ++j)
            ralloc_free(infos[j].nir);
         for (uint32_t j = 0; j < i; ++j) {
            wvk_shader_destroy(vk_dev, shaders_out[j], pAllocator);
            shaders_out[j] = NULL;
         }
         for (uint32_t j = i; j < shader_count && j < ARRAY_SIZE(sources); ++j) {
            if (sources[j])
               source_destroy(dev, sources[j]);
         }
         return result;
      }
      if (i < ARRAY_SIZE(sources))
         wvk_shader_from_vk(shaders_out[i])->source = sources[i];
   }
   return VK_SUCCESS;
}

/* A variant of `base` for `key`, compiled from its source. */
static struct wvk_shader *
compile_variant(struct wvk_device *dev, struct wvk_shader *base, const struct wvk_shader_key *key,
                const struct vk_vertex_input_state *vi)
{
   const struct wvk_shader_source *src = base->source;
   const mesa_shader_stage stage = base->vk.stage;
   struct blob_reader reader;
   blob_reader_init(&reader, src->nir.data, src->nir.size);
   nir_shader *nir = nir_deserialize(NULL, &wvk_device_physical(dev)->nir_options, &reader);
   if (!nir)
      return NULL;

   struct vk_multisample_state ms = { 0 };
   struct vk_graphics_pipeline_state state = { 0 };
   /* A geometry shader's draw after it: a vertex shader of its outputs. */
   if ((stage == MESA_SHADER_GEOMETRY && key->geometry == 2) || (stage == MESA_SHADER_TESS_EVAL && key->tess == 2)) {
      geometry_vertex_shader(nir, key->primitive_id, key->raster_stream);
      if (key->clip_varying)
         write_clip_varyings(nir, key->clip_varying - 1, key->clip_slots);
      match_fragment_inputs(nir, key->fs_inputs);
   }
   if (stage == MESA_SHADER_VERTEX) {
      if (key->clip_varying && !key->runs)
         write_clip_varyings(nir, key->clip_varying - 1, key->clip_slots);
      match_fragment_inputs(nir, key->fs_inputs);
      state.vi = vi;
   } else if (stage == MESA_SHADER_FRAGMENT) {
      ms.rasterization_samples = key->samples ? key->samples : 4;
      ms.alpha_to_one_enable = key->alpha_to_one;
      ms.alpha_to_coverage_enable = key->by_alpha;
      ms.sample_shading_enable = base->per_sample;
      ms.min_sample_shading = base->per_sample ? 1.0f : 0.0f;
      state.ms = &ms;
   }
   struct vk_shader_compile_info info = {
      .stage = nir->info.stage,
      .flags = src->flags,
      .next_stage_mask = src->next_stage_mask,
      .nir = nir,
      .robustness = &src->robustness,
      .set_layout_count = src->set_layout_count,
      .set_layouts = src->vk_layouts,
   };
   struct vk_shader *out;
   if (compile_one(dev, &info, &state, key, NULL, &out) != VK_SUCCESS)
      return NULL;
   struct wvk_shader *variant = wvk_shader_from_vk(out);
   if (stage == MESA_SHADER_VERTEX || key->geometry == 2 || key->tess == 2)
      memcpy(variant->key.fs_inputs, key->fs_inputs, sizeof(key->fs_inputs));
   return variant;
}

static struct wvk_shader *variant_for(struct wvk_device *dev, struct wvk_shader *shader,
                                      const struct wvk_shader_key *key, const struct vk_vertex_input_state *vi);

struct wvk_shader *
wvk_shader_geometry_variant(struct wvk_device *dev, struct wvk_shader *gs, unsigned kind, unsigned vs_vars,
                            const struct wvk_shader *other, const struct vk_dynamic_graphics_state *dyn,
                            bool after_tess, bool points, uint32_t stats)
{
   (void)dyn;
   if (!gs || !gs->source)
      return NULL;
   struct wvk_shader_key key;
   memset(&key, 0, sizeof(key));
   for (uint32_t g = 0; g < 4; ++g) {
      key.meta_base[g] = key.meta_words;
      const int32_t words = MAX2(gs->set_words[g], other ? other->set_words[g] : -1);
      key.meta_words += MAX2(words, 0);
   }
   key.geometry = kind;
   if (kind == 1) {
      key.vs_vars = vs_vars;
      key.after_tess = after_tess;
      key.stats = (stats & (VK_QUERY_PIPELINE_STATISTIC_GEOMETRY_SHADER_PRIMITIVES_BIT |
                            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_INVOCATIONS_BIT |
                            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
                            BITFIELD_RANGE(WVK_COUNTER_PRIMITIVES(0), 4))) != 0;
      if (key.stats && gs->gs_streams > 1)
         key.raster_stream = MIN2(dyn->rs.rasterization_stream, 3);
      key.provoking_last = dyn->rs.provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;
   } else if (other) {
      memcpy(key.fs_inputs, other->varyings, sizeof(key.fs_inputs));
      key.primitive_id = other->key.primitive_id;
      key.clip_varying = other->key.clip_varying;
      key.clip_slots = other->key.clip_slots;
   }
   /* The draw's varyings, of the stream it rasterizes. */
   if (kind == 2 && gs->gs_streams > 1)
      key.raster_stream = MIN2(dyn->rs.rasterization_stream, 3);
   /* Points as quads of the size it gives them (expand_points). */
   if (kind == 2 && points) {
      key.points = 1;
      key.point_coord = other ? other->key.point_coord : 0;
   }
   struct wvk_shader *variant = variant_for(dev, gs, &key, NULL);
   return variant != gs ? variant : NULL;
}

struct wvk_shader *
wvk_shader_tess_variant(struct wvk_device *dev, struct wvk_shader *shader, unsigned kind, unsigned vs_vars,
                        unsigned patch_vars, unsigned patch_vertices, const struct wvk_shader *other, bool points)
{
   if (!shader || !shader->source)
      return NULL;
   struct wvk_shader_key key;
   memset(&key, 0, sizeof(key));
   for (uint32_t g = 0; g < 4; ++g) {
      key.meta_base[g] = key.meta_words;
      const int32_t words = MAX2(shader->set_words[g], other ? other->set_words[g] : -1);
      key.meta_words += MAX2(words, 0);
   }
   key.tess = kind;
   if (kind == 1) {
      key.vs_vars = vs_vars;
      key.patch_vars = patch_vars;
      key.patch_vertices = patch_vertices;
   } else if (other) {
      memcpy(key.fs_inputs, other->varyings, sizeof(key.fs_inputs));
      key.primitive_id = other->key.primitive_id;
      key.clip_varying = other->key.clip_varying;
      key.clip_slots = other->key.clip_slots;
   }
   if (kind == 2 && points) {
      key.points = 1;
      key.point_coord = other ? other->key.point_coord : 0;
   }
   struct wvk_shader *variant = variant_for(dev, shader, &key, NULL);
   return variant != shader ? variant : NULL;
}


/* The formats of the views bound where the shader has storage images
 * without one into `key`, the shader's own where none is bound. Whether
 * they are the shader's. */
static bool
key_image_formats(const struct wvk_shader *shader, const VkFormat *image_formats, struct wvk_shader_key *key)
{
   bool same = true;
   for (uint32_t i = 0; i < shader->unformatted_count; ++i) {
      const uint32_t f = image_formats && image_formats[i] ? image_formats[i] : shader->key.image_format[i];
      key->image_format[i] = f;
      same &= f == shader->key.image_format[i];
   }
   return same;
}

struct wvk_shader *
wvk_shader_draw_variant(struct wvk_device *dev, struct wvk_shader *shader, const struct wvk_shader *other,
                        const struct vk_dynamic_graphics_state *dyn, unsigned samples, bool has_depth,
                        const VkFormat *color_formats, const VkFormat *image_formats, enum wvk_draw_shape shape,
                        unsigned runs, uint32_t stats)
{
   bool points = shape == WVK_DRAW_POINTS;
   if (!shader || !shader->source)
      return shader;

   struct wvk_shader_key key;
   memset(&key, 0, sizeof(key));
   const bool images = key_image_formats(shader, image_formats, &key);
   /* The records: every set either stage's layout has, in order. */
   for (uint32_t g = 0; g < 4; ++g) {
      key.meta_base[g] = key.meta_words;
      const int32_t words = MAX2(shader->set_words[g], other ? other->set_words[g] : -1);
      key.meta_words += MAX2(words, 0);
   }
   const bool records = !memcmp(key.meta_base, shader->key.meta_base, sizeof(key.meta_base)) &&
                        key.meta_words == shader->key.meta_words;
   struct vk_vertex_input_state vi;
   if (shader->vk.stage == MESA_SHADER_VERTEX) {
      /* The strides the draw has, which may be dynamic. */
      vi = *dyn->vi;
      u_foreach_bit(b, vi.bindings_valid)
         vi.bindings[b].stride = dyn->vi_binding_strides[b];
      /* Its compute shader doing its storage writes, which reads every
       * attribute itself (run_vertices). */
      if (runs) {
         key.runs = runs;
         key.pulled = pull_key(&vi, shader->attribs_read, true);
         return variant_for(dev, shader, &key, &vi);
      }
      /* Wide lines as quads (expand_lines). */
      if (other) {
         key.clip_varying = other->key.clip_varying;
         key.clip_slots = other->key.clip_slots;
      }
      if (shape == WVK_DRAW_LINES || shape == WVK_DRAW_LINES_PLACED) {
         key.lines = 1;
         key.line_coord = other ? other->key.line_coord : 0;
         if (other)
            memcpy(key.fs_inputs, other->varyings, sizeof(key.fs_inputs));
         key.pulled = pull_key(&vi, shader->attribs_read, true);
         return variant_for(dev, shader, &key, &vi);
      }
      /* Points as quads where it gives them a size, and gl_PointCoord where
       * the fragment shader reads it (expand_points). */
      points &= shader->writes_point_size;
      if (points) {
         key.points = 1;
         key.point_coord = other ? other->key.point_coord : 0;
      }
      key.pulled = pull_key(&vi, shader->attribs_read, points);
      if (records && images && key.pulled == shader->key.pulled && !points && !shader->key.points &&
          !key.clip_varying && varyings_match(shader, other))
         return shader;
      if (other)
         memcpy(key.fs_inputs, other->varyings, sizeof(key.fs_inputs));
   } else if (shader->vk.stage == MESA_SHADER_FRAGMENT) {
      key.alpha_to_one = dyn->ms.alpha_to_one_enable;
      key.by_alpha = dyn->ms.alpha_to_coverage_enable && (key.alpha_to_one || shader->own_mask || samples <= 1);
      key.samples = shader->uses_samples || key.by_alpha ? samples : 0;
      key.no_depth = shader->source_writes_depth && !has_depth;
      key.stats = (stats & VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT) != 0;
      /* Its colour outputs at their attachments' locations
       * (dynamicRenderingLocalRead), of the attachments the pass has, and
       * which of them it writes. */
      uint32_t outputs = shader->io_mask;
      bool identity = true;
      for (uint32_t i = 0; i < 8; ++i) {
         const bool attached = color_formats && color_formats[i];
         key.color_map[i] = attached ? dyn->cal.color_map[i] : MESA_VK_ATTACHMENT_UNUSED;
         identity &= !attached || key.color_map[i] == i;
      }
      if (identity) {
         memset(key.color_map, 0, sizeof(key.color_map));
      } else {
         key.color_remapped = 1;
         outputs = 0;
         for (uint32_t i = 0; i < 8; ++i) {
            if (key.color_map[i] != MESA_VK_ATTACHMENT_UNUSED && (shader->io_mask & BITFIELD_BIT(key.color_map[i])))
               outputs |= BITFIELD_BIT(i);
         }
      }
      /* After a geometry shader: gl_PrimitiveID a varying, at the first
       * location it has no input at. */
      if (other && other->vk.stage == MESA_SHADER_GEOMETRY && (shader->point_state & 4)) {
         for (uint32_t i = 0; i < WVK_MAX_VARYINGS && !key.primitive_id; ++i) {
            if (!shader->varyings[i])
               key.primitive_id = i + 1;
         }
      }
      /* Its clip and cull distances: vec4 varyings at the first
       * locations it has no input at, in a row. */
      if (shader->clip_inputs) {
         const unsigned slots = DIV_ROUND_UP(shader->clip_inputs, 4);
         for (uint32_t i = 0; i + slots <= WVK_MAX_VARYINGS && !key.clip_varying; ++i) {
            bool free = true;
            for (uint32_t j = 0; j < slots; ++j)
               free &= !shader->varyings[i + j] && i + j + 1 != key.primitive_id;
            if (free) {
               key.clip_varying = i + 1;
               key.clip_slots = slots;
            }
         }
      }
      /* A point's quad: gl_PointCoord at the first location it has no
       * input at, and facing front, as a line's does. */
      if (shape != WVK_DRAW_PRIMITIVES && shader->point_state) {
         key.points = 1;
         for (uint32_t i = 0; points && (shader->point_state & 1) && i < WVK_MAX_VARYINGS && !key.point_coord; ++i) {
            if (!shader->varyings[i] && i + 1 != key.primitive_id &&
                (!key.clip_varying || i + 1 < key.clip_varying || i + 1 >= key.clip_varying + key.clip_slots))
               key.point_coord = i + 1;
         }
      }
      /* A line's that is stippled or smooth: where on the line fragments
       * are, at the first location free after those. */
      for (uint32_t i = 0; shape == WVK_DRAW_LINES_PLACED && i < WVK_MAX_VARYINGS && !key.line_coord; ++i) {
         if (!shader->varyings[i] && i + 1 != key.primitive_id && i + 1 != key.point_coord &&
             (!key.clip_varying || i + 1 < key.clip_varying || i + 1 >= key.clip_varying + key.clip_slots))
            key.line_coord = i + 1;
      }
      /* The logic op, where it changes anything: COPY is the output as it
       * is. On a single-sampled pass, whose attachments are copied texel
       * for texel; a multisampled one would need each sample's. */
      if (dyn->cb.logic_op_enable && dyn->cb.logic_op != VK_LOGIC_OP_COPY && color_formats) {
         if (samples > 1) {
            mesa_logw_once("wvk: a logic op on a multisampled pass, which is not done");
         } else {
            for (uint32_t i = 0; i < 8; ++i) {
               if (color_formats[i] && (outputs & BITFIELD_BIT(i)) &&
                   wvk_format_takes_logic_op(color_formats[i])) {
                  key.logic_format[i] = color_formats[i];
                  key.logic_op = dyn->cb.logic_op + 1u;
               }
            }
         }
      }
      /* Blending WebGPU's cannot be, on a single-sampled pass as a logic
       * op: every attachment whose colour factors use CONSTANT_ALPHA
       * where another's (or its own) use CONSTANT_COLOR, which WebGPU's
       * one blend constant cannot both be. */
      const uint32_t blends = color_formats && !dyn->cb.logic_op_enable
                                 ? wvk_shader_blend_mask(dyn, outputs & dyn->cb.color_write_enables)
                                 : 0;
      if (blends && samples > 1) {
         mesa_logw_once("wvk: colour blend factors with both CONSTANT_COLOR and CONSTANT_ALPHA on a multisampled "
                        "pass, which is not done");
      } else {
         u_foreach_bit(i, blends) {
            const struct vk_color_blend_attachment_state *a = &dyn->cb.attachments[i];
            if (!color_formats[i])
               continue;
            key.blend_mask |= BITFIELD_BIT(i);
            key.blend_equation[i] = wvk_blend_equation(a->src_color_blend_factor, a->dst_color_blend_factor,
                                                       a->color_blend_op, a->src_alpha_blend_factor,
                                                       a->dst_alpha_blend_factor, a->alpha_blend_op);
            key.logic_format[i] = color_formats[i];
         }
      }
      /* The depth bounds test, where there is depth to test: against a
       * copy of a single-sampled attachment, texel for texel. */
      if (dyn->ds.depth.bounds_test.enable && has_depth) {
         if (samples > 1)
            mesa_logw_once("wvk: a depth bounds test on a multisampled pass, which is not done");
         else
            key.depth_bounds = 1;
      }
      if (key_equal(&key, &shader->key))
         return shader;
   } else {
      return shader;
   }
   return variant_for(dev, shader, &key, &vi);
}

/* The source's variant for `key`: made the first time. */
static struct wvk_shader *
variant_for(struct wvk_device *dev, struct wvk_shader *shader, const struct wvk_shader_key *k_in,
            const struct vk_vertex_input_state *vi)
{
   const struct wvk_shader_key key = *k_in;
   struct wvk_shader_source *src = shader->source;
   simple_mtx_lock(&src->lock);
   struct hash_entry *he = _mesa_hash_table_search(src->variants, &key);
   struct wvk_shader *variant = he ? he->data : NULL;
   if (!he) {
      /* A variant that failed is kept as none, not tried at every draw. */
      if (getenv("WINK_VK_VARIANT_LOG")) {
         const struct wvk_shader_key *own = &shader->key;
         fprintf(stderr, "wvk: %s shader %" PRIu64 " variant:%s%s%s%s%s%s%s%s%s%s%s%s\n",
                 _mesa_shader_stage_to_string(shader->vk.stage), shader->id,
                 key.pulled != own->pulled ? " vertex input" : "",
                 memcmp(key.fs_inputs, own->fs_inputs, sizeof(key.fs_inputs)) ? " varyings" : "",
                 key.alpha_to_one != own->alpha_to_one || key.by_alpha != own->by_alpha ? " alpha" : "",
                 key.samples != own->samples ? " samples" : "",
                 memcmp(key.meta_base, own->meta_base, sizeof(key.meta_base)) || key.meta_words != own->meta_words
                    ? " records" : "",
                 key.no_depth != own->no_depth ? " depth" : "",
                 key.logic_op != own->logic_op || memcmp(key.logic_format, own->logic_format, sizeof(key.logic_format))
                    ? " logic op" : "",
                 key.depth_bounds != own->depth_bounds ? " depth bounds" : "",
                 key.blend_mask != own->blend_mask ||
                       memcmp(key.blend_equation, own->blend_equation, sizeof(key.blend_equation))
                    ? " blending" : "",
                 memcmp(key.image_format, own->image_format, sizeof(key.image_format)) ? " image formats" : "",
                 key.line_coord != own->line_coord ? " line fragments" : "",
                 key.color_remapped != own->color_remapped ||
                       memcmp(key.color_map, own->color_map, sizeof(key.color_map))
                    ? " attachment locations" : "");
      }
      variant = compile_variant(dev, shader, &key, vi);
      if (!variant)
         mesa_logw_once("wvk: a shader's variant for a draw or dispatch failed to compile");
      struct wvk_shader_key *k = ralloc(src->variants, struct wvk_shader_key);
      if (k) {
         *k = key;
         _mesa_hash_table_insert(src->variants, k, variant);
      } else if (variant) {
         variant->vk.ops->destroy(&dev->vk, &variant->vk, NULL);
         variant = NULL;
      }
   }
   simple_mtx_unlock(&src->lock);
   return variant ? variant : shader;
}

struct wvk_shader *
wvk_shader_dispatch_variant(struct wvk_device *dev, struct wvk_shader *shader, const VkFormat *image_formats)
{
   if (!shader || !shader->source || !shader->unformatted_count)
      return shader;
   struct wvk_shader_key key = shader->key;
   if (key_image_formats(shader, image_formats, &key))
      return shader;
   return variant_for(dev, shader, &key, NULL);
}

static void
write_string(struct blob *blob, const char *s)
{
   blob_write_uint32(blob, s ? (uint32_t)strlen(s) + 1 : 0);
   if (s)
      blob_write_bytes(blob, s, strlen(s) + 1);
}

static char *
read_string(struct blob_reader *blob)
{
   const uint32_t size = blob_read_uint32(blob);
   if (!size)
      return NULL;
   const char *s = blob_read_bytes(blob, size);
   if (blob->overrun || s[size - 1])
      return NULL;
   return strdup(s);
}

static bool
wvk_shader_serialize(struct vk_device *vk_dev, const struct vk_shader *vk_shader, struct blob *blob)
{
   const struct wvk_shader *shader = container_of(vk_shader, struct wvk_shader, vk);
   blob_write_uint8(blob, shader->vk.stage);
   for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
      write_string(blob, shader->wgsl[i]);
   blob_write_uint32(blob, shader->binding_count);
   blob_write_bytes(blob, shader->bindings, shader->binding_count * sizeof(*shader->bindings));
   blob_write_uint32(blob, shader->io_mask);
   blob_write_uint8(blob, shader->logic_op_reads);
   blob_write_uint8(blob, shader->reads_depth_bounds);
   blob_write_uint8(blob, shader->writes_point_size);
   blob_write_uint8(blob, shader->cull_distances);
   blob_write_uint8(blob, shader->clip_inputs);
   blob_write_uint8(blob, shader->xfb_buffers);
   blob_write_bytes(blob, shader->xfb_strides, sizeof(shader->xfb_strides));
   blob_write_bytes(blob, shader->xfb_streams, sizeof(shader->xfb_streams));
   blob_write_uint8(blob, shader->xfb_output_count);
   blob_write_bytes(blob, shader->xfb_outputs, shader->xfb_output_count * sizeof(shader->xfb_outputs[0]));
   blob_write_uint8(blob, shader->writes_storage);
   blob_write_uint8(blob, shader->record_vars);
   blob_write_uint8(blob, shader->gs_input);
   blob_write_uint8(blob, shader->gs_output);
   blob_write_uint16(blob, shader->gs_max_vertices);
   blob_write_uint8(blob, shader->gs_invocations);
   blob_write_uint8(blob, shader->gs_streams);
   blob_write_uint8(blob, shader->tess_vertices_out);
   blob_write_uint8(blob, shader->patch_vars);
   blob_write_uint8(blob, shader->tess_domain);
   blob_write_uint8(blob, shader->tess_spacing);
   blob_write_uint8(blob, shader->tess_ccw);
   blob_write_uint8(blob, shader->tess_point_mode);
   blob_write_uint8(blob, shader->point_state);
   blob_write_uint8(blob, shader->unformatted_count);
   blob_write_bytes(blob, shader->unformatted, sizeof(shader->unformatted));
   blob_write_uint8(blob, shader->writes_depth);
   blob_write_uint8(blob, shader->uses_push_constants);
   blob_write_uint8(blob, shader->uses_draw_parameters);
   blob_write_uint8(blob, shader->writes_layer);
   blob_write_uint8(blob, shader->writes_viewport);
   blob_write_uint8(blob, shader->covers_by_alpha);
   blob_write_bytes(blob, shader->meta_base, sizeof(shader->meta_base));
   blob_write_uint32(blob, shader->meta_words);
   blob_write_uint8(blob, shader->pulled_count);
   blob_write_bytes(blob, shader->pulled_bindings, sizeof(shader->pulled_bindings));
   blob_write_bytes(blob, shader->workgroup_size, sizeof(shader->workgroup_size));
   blob_write_bytes(blob, &shader->key, sizeof(shader->key));
   blob_write_bytes(blob, shader->varyings, sizeof(shader->varyings));
   blob_write_uint32(blob, shader->attribs_read);
   blob_write_bytes(blob, shader->set_words, sizeof(shader->set_words));
   blob_write_uint8(blob, shader->per_sample);
   blob_write_uint8(blob, shader->own_mask);
   blob_write_uint8(blob, shader->uses_samples);
   blob_write_uint8(blob, shader->source_writes_depth);
   blob_write_uint8(blob, shader->source != NULL);
   if (shader->source)
      source_serialize(shader->source, blob);
   return !blob->out_of_memory;
}

static VkResult
wvk_shader_deserialize(struct vk_device *vk_dev, struct blob_reader *blob, uint32_t binary_version,
                       const VkAllocationCallbacks *pAllocator, struct vk_shader **shader_out)
{
   struct wvk_device *dev = container_of(vk_dev, struct wvk_device, vk);
   if (binary_version != WVK_SHADER_BINARY_VERSION)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   const mesa_shader_stage stage = blob_read_uint8(blob);
   char *wgsl[WVK_SHADER_VARIANTS];
   for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
      wgsl[i] = read_string(blob);
   const uint32_t binding_count = blob_read_uint32(blob);
   const void *bindings = blob_read_bytes(blob, binding_count * sizeof(struct wvk_shader_binding));
   const uint32_t io_mask = blob_read_uint32(blob);
   const uint8_t logic_op_reads = blob_read_uint8(blob);
   const bool reads_depth_bounds = blob_read_uint8(blob);
   const bool writes_point_size = blob_read_uint8(blob);
   const uint8_t cull_distances = blob_read_uint8(blob);
   const uint8_t clip_inputs = blob_read_uint8(blob);
   const uint8_t xfb_buffers = blob_read_uint8(blob);
   uint16_t xfb_strides[4];
   blob_copy_bytes(blob, xfb_strides, sizeof(xfb_strides));
   uint8_t xfb_streams[4];
   blob_copy_bytes(blob, xfb_streams, sizeof(xfb_streams));
   /* Read once: MIN2 would read it twice. */
   const uint8_t xfb_outputs_read = blob_read_uint8(blob);
   const uint8_t xfb_output_count = MIN2(xfb_outputs_read, WVK_MAX_XFB_OUTPUTS);
   struct wvk_xfb_output xfb_outputs[WVK_MAX_XFB_OUTPUTS];
   blob_copy_bytes(blob, xfb_outputs, xfb_output_count * sizeof(xfb_outputs[0]));
   const bool writes_storage = blob_read_uint8(blob);
   const uint8_t record_vars = blob_read_uint8(blob);
   const uint8_t gs_input = blob_read_uint8(blob);
   const uint8_t gs_output = blob_read_uint8(blob);
   const uint16_t gs_max_vertices = blob_read_uint16(blob);
   const uint8_t gs_invocations = blob_read_uint8(blob);
   const uint8_t gs_streams = blob_read_uint8(blob);
   const uint8_t tess_vertices_out = blob_read_uint8(blob);
   const uint8_t tess_patch_vars = blob_read_uint8(blob);
   const uint8_t tess_domain = blob_read_uint8(blob);
   const uint8_t tess_spacing = blob_read_uint8(blob);
   const bool tess_ccw = blob_read_uint8(blob);
   const bool tess_point_mode = blob_read_uint8(blob);
   const uint8_t point_state = blob_read_uint8(blob);
   const uint8_t unformatted_count = blob_read_uint8(blob);
   uint16_t unformatted[WVK_MAX_UNFORMATTED_IMAGES];
   blob_copy_bytes(blob, unformatted, sizeof(unformatted));
   const bool writes_depth = blob_read_uint8(blob);
   const bool uses_push_constants = blob_read_uint8(blob);
   const bool uses_draw_parameters = blob_read_uint8(blob);
   const bool writes_layer = blob_read_uint8(blob);
   const bool writes_viewport = blob_read_uint8(blob);
   const bool covers_by_alpha = blob_read_uint8(blob);
   uint32_t meta_base[4];
   blob_copy_bytes(blob, meta_base, sizeof(meta_base));
   const uint32_t meta_words = blob_read_uint32(blob);
   const uint8_t pulled_count = blob_read_uint8(blob);
   uint8_t pulled_bindings[WVK_MAX_PULLED_BINDINGS];
   blob_copy_bytes(blob, pulled_bindings, sizeof(pulled_bindings));
   uint16_t workgroup_size[3];
   blob_copy_bytes(blob, workgroup_size, sizeof(workgroup_size));
   struct wvk_shader_key key;
   blob_copy_bytes(blob, &key, sizeof(key));
   uint8_t varyings[WVK_MAX_VARYINGS];
   blob_copy_bytes(blob, varyings, sizeof(varyings));
   const uint32_t attribs_read = blob_read_uint32(blob);
   int32_t set_words[4];
   blob_copy_bytes(blob, set_words, sizeof(set_words));
   const bool per_sample = blob_read_uint8(blob);
   const bool own_mask = blob_read_uint8(blob);
   const bool uses_samples = blob_read_uint8(blob);
   const bool source_writes_depth = blob_read_uint8(blob);
   struct wvk_shader_source *source = blob_read_uint8(blob) ? source_deserialize(blob) : NULL;
   /* A geometry or tessellation shader is its source alone (geometry_shell). */
   const bool shell = stage == MESA_SHADER_GEOMETRY || stage == MESA_SHADER_TESS_CTRL ||
                      stage == MESA_SHADER_TESS_EVAL;
   if (blob->overrun || (!wgsl[0] && !shell) || (shell && !source)) {
      for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
         free(wgsl[i]);
      if (source)
         source_destroy(dev, source);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }

   struct wvk_shader *shader = vk_shader_zalloc(&dev->vk, &wvk_shader_ops, stage, pAllocator,
                                                sizeof(*shader));
   if (!shader) {
      for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
         free(wgsl[i]);
      if (source)
         source_destroy(dev, source);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   shader->id = p_atomic_inc_return(&dev->next_shader_id);
   memcpy(shader->wgsl, wgsl, sizeof(wgsl));
   shader->binding_count = binding_count;
   shader->bindings = calloc(MAX2(binding_count, 1), sizeof(*shader->bindings));
   if (shader->bindings)
      memcpy(shader->bindings, bindings, binding_count * sizeof(*shader->bindings));
   shader->io_mask = io_mask;
   shader->logic_op_reads = logic_op_reads;
   shader->reads_depth_bounds = reads_depth_bounds;
   shader->writes_point_size = writes_point_size;
   shader->cull_distances = cull_distances;
   shader->clip_inputs = clip_inputs;
   shader->xfb_buffers = xfb_buffers;
   memcpy(shader->xfb_strides, xfb_strides, sizeof(xfb_strides));
   memcpy(shader->xfb_streams, xfb_streams, sizeof(xfb_streams));
   shader->xfb_output_count = xfb_output_count;
   memcpy(shader->xfb_outputs, xfb_outputs, xfb_output_count * sizeof(xfb_outputs[0]));
   shader->writes_storage = writes_storage;
   shader->record_vars = record_vars;
   shader->gs_input = gs_input;
   shader->gs_output = gs_output;
   shader->gs_max_vertices = gs_max_vertices;
   shader->gs_invocations = gs_invocations;
   shader->gs_streams = gs_streams;
   shader->tess_vertices_out = tess_vertices_out;
   shader->patch_vars = tess_patch_vars;
   shader->tess_domain = tess_domain;
   shader->tess_spacing = tess_spacing;
   shader->tess_ccw = tess_ccw;
   shader->tess_point_mode = tess_point_mode;
   shader->point_state = point_state;
   shader->unformatted_count = MIN2(unformatted_count, WVK_MAX_UNFORMATTED_IMAGES);
   memcpy(shader->unformatted, unformatted, sizeof(unformatted));
   shader->writes_depth = writes_depth;
   shader->uses_push_constants = uses_push_constants;
   shader->uses_draw_parameters = uses_draw_parameters;
   shader->writes_layer = writes_layer;
   shader->writes_viewport = writes_viewport;
   shader->covers_by_alpha = covers_by_alpha;
   memcpy(shader->meta_base, meta_base, sizeof(meta_base));
   shader->meta_words = meta_words;
   shader->pulled_count = MIN2(pulled_count, WVK_MAX_PULLED_BINDINGS);
   memcpy(shader->pulled_bindings, pulled_bindings, sizeof(pulled_bindings));
   memcpy(shader->workgroup_size, workgroup_size, sizeof(workgroup_size));
   shader->key = key;
   memcpy(shader->varyings, varyings, sizeof(varyings));
   shader->attribs_read = attribs_read;
   memcpy(shader->set_words, set_words, sizeof(set_words));
   shader->per_sample = per_sample;
   shader->own_mask = own_mask;
   shader->uses_samples = uses_samples;
   shader->source_writes_depth = source_writes_depth;
   shader->source = source;
   VkResult result = shader->bindings ? VK_SUCCESS : vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   for (uint32_t i = 0; result == VK_SUCCESS && i < WVK_SHADER_VARIANTS; ++i) {
      if (shader->wgsl[i])
         result = make_module(dev, shader->wgsl[i], "cached", &shader->module[i]);
   }
   if (result != VK_SUCCESS) {
      wvk_shader_destroy(vk_dev, &shader->vk, pAllocator);
      return result;
   }
   *shader_out = &shader->vk;
   return VK_SUCCESS;
}

/* VK_KHR_pipeline_executable_properties: a shader is one executable, its
 * WGSL, which is what there is to show of it -- the WebGPU implementation
 * compiles it on. */
#define WRITE_STR(field, ...)                                   \
   do {                                                         \
      memset(field, 0, sizeof(field));                          \
      snprintf(field, sizeof(field), __VA_ARGS__);              \
   } while (0)

static VkResult
wvk_shader_get_executable_properties(struct vk_device *device, const struct vk_shader *vk_shader,
                                     uint32_t *executable_count,
                                     VkPipelineExecutablePropertiesKHR *properties)
{
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutablePropertiesKHR, out, properties, executable_count);
   vk_outarray_append_typed(VkPipelineExecutablePropertiesKHR, &out, props) {
      props->stages = mesa_to_vk_shader_stage(vk_shader->stage);
      props->subgroupSize = 1;
      WRITE_STR(props->name, "%s", _mesa_shader_stage_to_string(vk_shader->stage));
      WRITE_STR(props->description, "%s shader, as WGSL", _mesa_shader_stage_to_string(vk_shader->stage));
   }
   return vk_outarray_status(&out);
}

static VkResult
wvk_shader_get_executable_statistics(struct vk_device *device, const struct vk_shader *vk_shader,
                                     uint32_t executable_index, uint32_t *statistic_count,
                                     VkPipelineExecutableStatisticKHR *statistics)
{
   const struct wvk_shader *shader = container_of(vk_shader, struct wvk_shader, vk);
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableStatisticKHR, out, statistics, statistic_count);
   vk_outarray_append_typed(VkPipelineExecutableStatisticKHR, &out, stat) {
      WRITE_STR(stat->name, "WGSL bytes");
      WRITE_STR(stat->description, "How long the shader's WGSL is");
      stat->format = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR;
      stat->value.u64 = shader->wgsl[WVK_SHADER_FLIPPED] ? strlen(shader->wgsl[WVK_SHADER_FLIPPED]) : 0;
   }
   vk_outarray_append_typed(VkPipelineExecutableStatisticKHR, &out, stat) {
      WRITE_STR(stat->name, "Bindings");
      WRITE_STR(stat->description, "The WebGPU bindings the shader has, the driver's among them");
      stat->format = VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR;
      stat->value.u64 = shader->binding_count;
   }
   return vk_outarray_status(&out);
}

static VkResult
wvk_shader_get_executable_internal_representations(
   struct vk_device *device, const struct vk_shader *vk_shader, uint32_t executable_index,
   uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   const struct wvk_shader *shader = container_of(vk_shader, struct wvk_shader, vk);
   VK_OUTARRAY_MAKE_TYPED(VkPipelineExecutableInternalRepresentationKHR, out, internal_representations,
                          internal_representation_count);
   bool incomplete = false;
   for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i) {
      const char *text = shader->wgsl[i];
      if (!text)
         continue;
      vk_outarray_append_typed(VkPipelineExecutableInternalRepresentationKHR, &out, ir) {
         WRITE_STR(ir->name, "%s", i == WVK_SHADER_UNFLIPPED ? "WGSL, unflipped" : "WGSL");
         WRITE_STR(ir->description, "%s", i == WVK_SHADER_UNFLIPPED
                                             ? "The shader as WebGPU is given it, for a pass not flipped in y"
                                             : "The shader as WebGPU is given it");
         ir->isText = VK_TRUE;
         const size_t size = strlen(text) + 1;
         if (!ir->pData) {
            ir->dataSize = size;
         } else {
            memcpy(ir->pData, text, MIN2(ir->dataSize, size));
            if (ir->dataSize < size) {
               ((char *)ir->pData)[ir->dataSize - 1] = 0;
               incomplete = true;
            }
            ir->dataSize = MIN2(ir->dataSize, size);
         }
      }
   }
   return incomplete ? VK_INCOMPLETE : vk_outarray_status(&out);
}
#undef WRITE_STR

static const struct vk_shader_ops wvk_shader_ops = {
   .destroy = wvk_shader_destroy,
   .serialize = wvk_shader_serialize,
   .get_executable_properties = wvk_shader_get_executable_properties,
   .get_executable_statistics = wvk_shader_get_executable_statistics,
   .get_executable_internal_representations = wvk_shader_get_executable_internal_representations,
};

/* In wvk_cmd_draw.c. */
void wvk_cmd_bind_shaders(struct vk_command_buffer *cmd, uint32_t stage_count,
                          const mesa_shader_stage *stages, struct vk_shader **const shaders);

const struct vk_device_shader_ops wvk_device_shader_ops = {
   .get_nir_options = wvk_get_nir_options,
   .get_spirv_options = wvk_get_spirv_options,
   .preprocess_nir = wvk_preprocess_nir,
   .hash_state = wvk_hash_state,
   .compile = wvk_compile,
   .deserialize = wvk_shader_deserialize,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
   .cmd_bind_shaders = wvk_cmd_bind_shaders,
};
