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

#include "gex_wgpu_nir.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_deref.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/blob.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"
#include "util/u_atomic.h"
#include "util/mesa-blake3.h"
#include "vk_format.h"
#include "vk_nir.h"
#include "vk_graphics_state.h"
#include "vk_pipeline.h"

#include <stdlib.h>


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
   nir_def_replace(&intr->def, nir_imm_intN_t(b, 0, intr->def.bit_size));
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
/* A gather's offset that is not a constant, which WGSL's are. */
static bool
dynamic_gather_offset(const nir_instr *instr, const void *data)
{
   const nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int i = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   return tex->op == nir_texop_tg4 && i >= 0 && !nir_src_is_const(tex->src[i].src);
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
   return (vb->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE && vb->divisor != 1) ||
          (vb->stride % 4) || (align && a->offset % align);
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
   }
   _mesa_blake3_final(&ctx, blake3_out);
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
      for (unsigned i = 0; i < tex->num_srcs && count < 2; ++i) {
         const bool is_sampler = tex->src[i].src_type == nir_tex_src_sampler_deref;
         if (tex->src[i].src_type != nir_tex_src_texture_deref && !is_sampler)
            continue;
         if (!resolve_element(state, b, nir_src_as_deref(tex->src[i].src), &refs[count]))
            continue;
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

/* The access, on the words of buffer (set, binding): its value, or NULL
 * for a store. */
static nir_def *
lower_word_access(nir_builder *b, struct word_buffers *wb, nir_intrinsic_instr *intr, bool ubo, uint32_t set,
                  uint32_t binding)
{
   nir_variable *var = word_var(b->shader, wb, ubo, set, binding);
   if (intr->intrinsic == nir_intrinsic_get_ssbo_size) {
      nir_deref_instr *words = nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
      nir_def *length = nir_deref_buffer_array_length(b, 32, &words->def, .access = ACCESS_NON_WRITEABLE);
      return nir_imul_imm(b, length, 4);
   }

   const int offset_src = intr->intrinsic == nir_intrinsic_store_ssbo ? 2 : 1;
   nir_def *first = nir_ushr_imm(b, intr->src[offset_src].ssa, 2);

   if (intr->intrinsic == nir_intrinsic_load_ubo || intr->intrinsic == nir_intrinsic_load_ssbo) {
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned c = 0; c < intr->def.num_components; ++c)
         comps[c] = load_word(b, var, ubo, nir_iadd_imm(b, first, c));
      return nir_vec(b, comps, intr->def.num_components);
   }

   nir_deref_instr *words = nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
   if (intr->intrinsic == nir_intrinsic_store_ssbo) {
      nir_def *value = intr->src[0].ssa;
      u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
         nir_store_deref(b, nir_build_deref_array(b, words, nir_iadd_imm(b, first, c)),
                         nir_channel(b, value, c), 0x1);
      }
      return NULL;
   }

   /* Atomics, on one word. */
   nir_deref_instr *word = nir_build_deref_array(b, words, first);
   const nir_atomic_op op = nir_intrinsic_atomic_op(intr);
   if (op == nir_atomic_op_imin || op == nir_atomic_op_imax) {
      /* The words are unsigned, and so are WGSL's atomics on them: a
       * signed minimum or maximum is swapped in until nothing came
       * between. */
      nir_variable *old = nir_local_variable_create(b->impl, glsl_uint_type(), "atomic_old");
      nir_store_var(b, old, nir_deref_atomic(b, 32, &word->def, nir_imm_int(b, 0), .atomic_op = nir_atomic_op_iadd),
                    0x1);
      nir_loop *loop = nir_push_loop(b);
      {
         nir_def *cur = nir_load_var(b, old);
         nir_def *data = intr->src[2].ssa;
         nir_def *want = op == nir_atomic_op_imin ? nir_imin(b, cur, data) : nir_imax(b, cur, data);
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
      mesa_loge("wvk: a buffer access whose binding cannot be told");
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
      element = nir_load_vertex_id(b);
   } else {
      nir_def *first = nir_load_base_instance(b);
      element = first;
      if (vb->divisor)
         element = nir_iadd(b, first, nir_udiv_imm(b, nir_isub(b, nir_load_instance_id(b), first), vb->divisor));
   }
   nir_def *starts = nir_load_ssbo(b, 4, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                                   nir_imm_int(b, WVK_DRIVER_PULL_OFFSET), .align_mul = 16,
                                   .access = ACCESS_NON_WRITEABLE);
   /* The stride as the draw has it: it may be dynamic. */
   nir_def *strides = nir_load_ssbo(b, 4, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                                    nir_imm_int(b, WVK_DRIVER_PULL_STRIDES), .align_mul = 16,
                                    .access = ACCESS_NON_WRITEABLE);
   nir_def *address = nir_iadd(b, nir_channel(b, starts, slot),
                               nir_iadd_imm(b, nir_imul(b, element, nir_channel(b, strides, slot)), a->offset));

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
            if (!is_pulled(vi, location))
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
   struct word_buffers wb = {
      .vars = _mesa_pointer_hash_table_create(mem_ctx),
      .info = info,
      .ubo_sizes = _mesa_pointer_hash_table_create(mem_ctx),
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
               NULL);
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
   const struct glsl_type *vec4 = glsl_vector_type(glsl_get_base_type(old), 4);
   var->type = glsl_type_is_array(var->type) ? glsl_array_type(vec4, glsl_get_length(var->type), 0) : vec4;
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

static void
wvk_shader_destroy(struct vk_device *vk_dev, struct vk_shader *vk_shader,
                   const VkAllocationCallbacks *pAllocator)
{
   struct wvk_shader *shader = wvk_shader_from_vk(vk_shader);
   for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i) {
      if (shader->module[i])
         wgpuShaderModuleRelease(shader->module[i]);
      free(shader->wgsl[i]);
   }
   free(shader->bindings);
   vk_shader_free(vk_dev, pAllocator, vk_shader);
}

static VkResult
compile_one(struct wvk_device *dev, struct vk_shader_compile_info *info,
            const struct vk_graphics_pipeline_state *pipeline_state, const VkAllocationCallbacks *pAllocator,
            struct vk_shader **shader_out)
{
   nir_shader *nir = info->nir;
   const mesa_shader_stage stage = nir->info.stage;
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
   /* The records of the pipeline layout's sets, set after set. */
   uint32_t meta_words = 0;
   for (uint32_t g = 0; g < 4; ++g) {
      shader->meta_base[g] = meta_words;
      if (g < info->set_layout_count && info->set_layouts[g]) {
         const struct wvk_descriptor_set_layout *layout =
            container_of(info->set_layouts[g], struct wvk_descriptor_set_layout, vk);
         meta_words += layout->slot_count * WVK_META_WORDS;
      }
   }
   shader->meta_words = meta_words;
   if (WVK_META_OFFSET + meta_words * 4 > WVK_PUSH_WINDOW)
      mesa_logw_once("wvk: a pipeline layout of more descriptors than their records have room for");

   NIR_PASS(_, nir, lower_push_constants);
   /* Where varyings are sampled is the fragment shader's alone: at the
    * centre, as far as WebGPU's matching of the stages goes. */
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_foreach_shader_out_variable(var, nir) {
         var->data.sample = false;
         var->data.centroid = false;
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
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_sample_pos_instr, nir_metadata_control_flow, &s);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_interp_at_instr, nir_metadata_control_flow, &s);
      NIR_PASS(_, nir, lower_input_sampling, &s);
   }
   /* Shared memory as words, as ntv's is, and accessed a word at a time:
    * a 64-bit or 16-bit value is words' worth, which WGSL has no types of. */
   if (nir->info.stage == MESA_SHADER_COMPUTE) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
      /* A null initializer (shaderZeroInitializeWorkgroupMemory): stores of
       * zero, and the variable gone, which ntv would lay out again. */
      bool initialized = false;
      nir_foreach_variable_with_modes(var, nir, nir_var_mem_shared) {
         initialized |= var->constant_initializer != NULL || var->pointer_initializer != NULL;
         var->constant_initializer = NULL;
         var->pointer_initializer = NULL;
      }
      if (initialized) {
         /* In words, which the size is made a whole number of. */
         nir->info.shared_size = align(nir->info.shared_size, 4);
         NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size, 4);
      }
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
   /* Varyings WGSL can have: no structs, and a location each. */
   NIR_PASS(_, nir, split_io_structs);
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
   bool retyped = false;
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, match_image_type_instr, nir_metadata_all, &retyped);
   if (retyped)
      nir_fixup_deref_types(nir);
   /* Swizzles first: a gather's is a gather of another component, which
    * then samples as its own; and the border replaces a texel before the
    * swizzle takes it apart. */
   NIR_PASS(_, nir, wvk_nir_lower_view_swizzles, shader->meta_base);
   /* Projection divided out first: the samplers' lowering takes the
    * coordinates as they are. And a gather at offsets WGSL has none of --
    * four, or one not constant -- is gathers each at one, or at the
    * coordinates moved. */
   const nir_lower_tex_options txp = {
      .lower_txp = ~0u,
      .lower_tg4_offsets = true,
      .lower_offset_filter = dynamic_gather_offset,
   };
   NIR_PASS(_, nir, nir_lower_tex, &txp);
   NIR_PASS(_, nir, wvk_nir_lower_samplers, shader->meta_base);
   NIR_PASS(_, nir, lower_1d_images);
   NIR_PASS(_, nir, lower_ms_arrays);
   NIR_PASS(_, nir, pad_io_vectors);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, demote_to_terminate_instr, nir_metadata_all, NULL);
   NIR_PASS(_, nir, wvk_nir_lower_image_atomics, shader->meta_base);
   NIR_PASS(_, nir, wvk_nir_lower_texel_buffers, shader->meta_base);
   lower_buffers_to_words(nir, info, pipeline_state, shader, mem_ctx);
   /* The signed atomics' loops' variables. */
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   optimize(nir);
   /* What is left of 8 and 16 bits, words: memory is by now. */
   bool widened = false;
   NIR_PASS(widened, nir, wvk_nir_widen_small_types);
   if (widened)
      optimize(nir);
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

static VkResult
wvk_compile(struct vk_device *vk_dev, uint32_t shader_count, struct vk_shader_compile_info *infos,
            const struct vk_graphics_pipeline_state *state, const struct vk_features *enabled_features,
            const VkAllocationCallbacks *pAllocator, struct vk_shader **shaders_out)
{
   struct wvk_device *dev = container_of(vk_dev, struct wvk_device, vk);
   nir_shader *vs = NULL, *fs = NULL;
   for (uint32_t i = 0; i < shader_count; ++i) {
      if (infos[i].stage == MESA_SHADER_VERTEX)
         vs = infos[i].nir;
      else if (infos[i].stage == MESA_SHADER_FRAGMENT)
         fs = infos[i].nir;
   }
   if (vs && fs)
      link_interpolation(vs, fs);
   for (uint32_t i = 0; i < shader_count; ++i) {
      VkResult result = compile_one(dev, &infos[i], state, pAllocator, &shaders_out[i]);
      if (result != VK_SUCCESS) {
         /* compile consumes every NIR, whatever happens. */
         for (uint32_t j = i + 1; j < shader_count; ++j)
            ralloc_free(infos[j].nir);
         for (uint32_t j = 0; j < i; ++j) {
            wvk_shader_destroy(vk_dev, shaders_out[j], pAllocator);
            shaders_out[j] = NULL;
         }
         return result;
      }
   }
   return VK_SUCCESS;
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
   for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
      write_string(blob, shader->wgsl[i]);
   blob_write_uint32(blob, shader->binding_count);
   blob_write_bytes(blob, shader->bindings, shader->binding_count * sizeof(*shader->bindings));
   blob_write_uint32(blob, shader->io_mask);
   blob_write_uint8(blob, shader->writes_depth);
   blob_write_uint8(blob, shader->uses_push_constants);
   blob_write_uint8(blob, shader->uses_draw_parameters);
   blob_write_uint8(blob, shader->writes_layer);
   blob_write_uint8(blob, shader->writes_viewport);
   blob_write_bytes(blob, shader->meta_base, sizeof(shader->meta_base));
   blob_write_uint32(blob, shader->meta_words);
   blob_write_uint8(blob, shader->pulled_count);
   blob_write_bytes(blob, shader->pulled_bindings, sizeof(shader->pulled_bindings));
   blob_write_bytes(blob, shader->workgroup_size, sizeof(shader->workgroup_size));
   return !blob->out_of_memory;
}

static VkResult
wvk_shader_deserialize(struct vk_device *vk_dev, struct blob_reader *blob, uint32_t binary_version,
                       const VkAllocationCallbacks *pAllocator, struct vk_shader **shader_out)
{
   struct wvk_device *dev = container_of(vk_dev, struct wvk_device, vk);
   if (binary_version != WVK_SHADER_BINARY_VERSION)
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);

   /* The stage is not in the blob: the first WGSL says it. */
   char *wgsl[WVK_SHADER_VARIANTS];
   for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
      wgsl[i] = read_string(blob);
   const uint32_t binding_count = blob_read_uint32(blob);
   const void *bindings = blob_read_bytes(blob, binding_count * sizeof(struct wvk_shader_binding));
   const uint32_t io_mask = blob_read_uint32(blob);
   const bool writes_depth = blob_read_uint8(blob);
   const bool uses_push_constants = blob_read_uint8(blob);
   const bool uses_draw_parameters = blob_read_uint8(blob);
   const bool writes_layer = blob_read_uint8(blob);
   const bool writes_viewport = blob_read_uint8(blob);
   uint32_t meta_base[4];
   blob_copy_bytes(blob, meta_base, sizeof(meta_base));
   const uint32_t meta_words = blob_read_uint32(blob);
   const uint8_t pulled_count = blob_read_uint8(blob);
   uint8_t pulled_bindings[WVK_MAX_PULLED_BINDINGS];
   blob_copy_bytes(blob, pulled_bindings, sizeof(pulled_bindings));
   uint16_t workgroup_size[3];
   blob_copy_bytes(blob, workgroup_size, sizeof(workgroup_size));
   if (blob->overrun || !wgsl[0]) {
      for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
         free(wgsl[i]);
      return vk_error(dev, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   const mesa_shader_stage stage = strstr(wgsl[0], "@vertex")     ? MESA_SHADER_VERTEX
                                   : strstr(wgsl[0], "@fragment") ? MESA_SHADER_FRAGMENT
                                                                  : MESA_SHADER_COMPUTE;

   struct wvk_shader *shader = vk_shader_zalloc(&dev->vk, &wvk_shader_ops, stage, pAllocator,
                                                sizeof(*shader));
   if (!shader) {
      for (uint32_t i = 0; i < WVK_SHADER_VARIANTS; ++i)
         free(wgsl[i]);
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   shader->id = p_atomic_inc_return(&dev->next_shader_id);
   memcpy(shader->wgsl, wgsl, sizeof(wgsl));
   shader->binding_count = binding_count;
   shader->bindings = calloc(MAX2(binding_count, 1), sizeof(*shader->bindings));
   if (shader->bindings)
      memcpy(shader->bindings, bindings, binding_count * sizeof(*shader->bindings));
   shader->io_mask = io_mask;
   shader->writes_depth = writes_depth;
   shader->uses_push_constants = uses_push_constants;
   shader->uses_draw_parameters = uses_draw_parameters;
   shader->writes_layer = writes_layer;
   shader->writes_viewport = writes_viewport;
   memcpy(shader->meta_base, meta_base, sizeof(meta_base));
   shader->meta_words = meta_words;
   shader->pulled_count = MIN2(pulled_count, WVK_MAX_PULLED_BINDINGS);
   memcpy(shader->pulled_bindings, pulled_bindings, sizeof(pulled_bindings));
   memcpy(shader->workgroup_size, workgroup_size, sizeof(workgroup_size));
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

static VkResult
wvk_shader_get_executable_properties(struct vk_device *device, const struct vk_shader *shader,
                                     uint32_t *executable_count,
                                     VkPipelineExecutablePropertiesKHR *properties)
{
   *executable_count = 0;
   return VK_SUCCESS;
}

static VkResult
wvk_shader_get_executable_statistics(struct vk_device *device, const struct vk_shader *shader,
                                     uint32_t executable_index, uint32_t *statistic_count,
                                     VkPipelineExecutableStatisticKHR *statistics)
{
   *statistic_count = 0;
   return VK_SUCCESS;
}

static VkResult
wvk_shader_get_executable_internal_representations(
   struct vk_device *device, const struct vk_shader *shader, uint32_t executable_index,
   uint32_t *internal_representation_count,
   VkPipelineExecutableInternalRepresentationKHR *internal_representations)
{
   *internal_representation_count = 0;
   return VK_SUCCESS;
}

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
