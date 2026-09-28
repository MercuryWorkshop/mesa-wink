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
#include "wvk_physical_device.h"

#include "gex_wgpu_nir.h"

#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/blob.h"
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

static void
wvk_preprocess_nir(struct vk_physical_device *vk_pdev, nir_shader *nir,
                   const struct vk_pipeline_robustness_state *rs)
{
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   nir_remove_non_entrypoints(nir);
   NIR_PASS(_, nir, nir_lower_variable_initializers, ~nir_var_function_temp);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_lower_system_values);
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
 * attribute every instance. */
static bool
is_pulled(const struct vk_vertex_input_state *vi, int location)
{
   if (!vi || location < 0 || location >= MESA_VK_MAX_VERTEX_ATTRIBUTES ||
       !(vi->attributes_valid & BITFIELD_BIT(location)))
      return false;
   const struct vk_vertex_binding_state *vb = &vi->bindings[vi->attributes[location].binding];
   return vb->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE && vb->divisor != 1;
}

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
         const uint32_t words[] = { location, a->binding, a->format, a->offset, vb->stride, vb->divisor };
         _mesa_blake3_update(&ctx, words, sizeof(words));
      }
   }
   _mesa_blake3_final(&ctx, blake3_out);
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
      s->count = glsl_type_is_array(var->type) ? glsl_get_aoa_size(var->type) : 1;
      s->texture = rzalloc_array(s, nir_variable *, s->count);
      const bool combined = glsl_type_is_sampler(bare) && !glsl_type_is_bare_sampler(bare);
      if (combined)
         s->sampler = rzalloc_array(s, nir_variable *, s->count);
      for (uint32_t e = 0; e < s->count; ++e) {
         if (combined) {
            const struct glsl_type *texture =
               glsl_texture_type(glsl_get_sampler_dim(bare), glsl_sampler_type_is_array(bare),
                                 glsl_get_sampler_result_type(bare));
            s->texture[e] = new_var(nir, nir_var_uniform, texture, var, bl->slot + e, "_tex", e);
            s->sampler[e] = new_var(nir, nir_var_uniform,
                                    glsl_sampler_type_is_shadow(bare) ? glsl_bare_shadow_sampler_type()
                                                                      : glsl_bare_sampler_type(),
                                    var, bl->slot + bl->count + e, "_smp", e);
         } else if (glsl_type_is_image(bare) && (glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_SUBPASS ||
                                                 glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_SUBPASS_MS)) {
            /* An input attachment: fetched, as a texture. */
            const bool ms = glsl_get_sampler_dim(bare) == GLSL_SAMPLER_DIM_SUBPASS_MS;
            const struct glsl_type *texture =
               glsl_texture_type(ms ? GLSL_SAMPLER_DIM_MS : GLSL_SAMPLER_DIM_2D, false,
                                 glsl_get_sampler_result_type(bare));
            s->texture[e] = new_var(nir, nir_var_uniform, texture, var, bl->slot + e, "_ia", e);
         } else {
            s->texture[e] = new_var(nir, var->data.mode, bare, var, bl->slot + e, "_", e);
         }
      }
      _mesa_hash_table_insert(state->split, var, s);
   }
}

/* Which element of its binding a deref is of: arrays of textures are
 * separate bindings, so the index has to be known. */
static nir_variable *
element_var(struct lower_state *state, nir_deref_instr *deref, bool sampler)
{
   uint32_t element = 0;
   nir_deref_instr *d = deref;
   while (d->deref_type != nir_deref_type_var) {
      if (d->deref_type == nir_deref_type_array) {
         if (!nir_src_is_const(d->arr.index)) {
            mesa_logw_once("wvk: a dynamically indexed array of textures is not supported yet; "
                           "the first element is used");
         } else {
            element += nir_src_as_uint(d->arr.index);
         }
      }
      d = nir_deref_instr_parent(d);
      if (!d)
         return NULL;
   }
   struct hash_entry *he = _mesa_hash_table_search(state->split, d->var);
   if (!he)
      return NULL;
   struct split_var *s = he->data;
   element = MIN2(element, s->count - 1);
   if (sampler && s->sampler)
      return s->sampler[element];
   return s->texture[element];
}

static bool
rewrite_resource_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct lower_state *state = data;
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      bool progress = false;
      /* A combined sampler has only its texture deref for fetches and
       * queries, and both for sampling; either way each half goes to its
       * own variable. */
      for (unsigned i = 0; i < tex->num_srcs; ++i) {
         const bool is_sampler = tex->src[i].src_type == nir_tex_src_sampler_deref;
         if (tex->src[i].src_type != nir_tex_src_texture_deref && !is_sampler)
            continue;
         nir_variable *var = element_var(state, nir_src_as_deref(tex->src[i].src), is_sampler);
         if (!var)
            continue;
         b->cursor = nir_before_instr(instr);
         nir_src_rewrite(&tex->src[i].src, &nir_build_deref_var(b, var)->def);
         progress = true;
      }
      return progress;
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
      nir_variable *var = element_var(state, nir_src_as_deref(intr->src[0]), false);
      if (!var)
         return false;
      b->cursor = nir_before_instr(instr);
      nir_src_rewrite(&intr->src[0], &nir_build_deref_var(b, var)->def);
      return true;
   }
   case nir_intrinsic_vulkan_resource_index: {
      /* Buffers: each element its own binding, found by (set, binding) in
       * zink's emitter. */
      const uint32_t set = nir_intrinsic_desc_set(intr);
      const uint32_t binding = nir_intrinsic_binding(intr);
      const struct wvk_descriptor_set_binding_layout *bl = binding_layout(state->info, set, binding);
      if (!bl) {
         mesa_loge("wvk: a shader uses set %u binding %u, which its pipeline layout does not have",
                   set, binding);
         state->failed = true;
         return false;
      }
      uint32_t element = 0;
      if (nir_src_is_const(intr->src[0]))
         element = MIN2(nir_src_as_uint(intr->src[0]), bl->count - 1);
      else
         mesa_logw_once("wvk: a dynamically indexed array of buffers is not supported yet; "
                        "the first element is used");
      nir_intrinsic_set_binding(intr, bl->slot + element);
      b->cursor = nir_before_instr(instr);
      nir_src_rewrite(&intr->src[0], nir_imm_int(b, 0));
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

/* gl_Layer and gl_ViewportIndex from a vertex shader: WebGPU renders one
 * layer and one viewport, and so does everything here, one pass at a time
 * -- vk_meta's shaders write the layer anyway. */
static bool
drop_layer_outputs(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_VERTEX)
      return false;
   bool progress = false;
   nir_foreach_shader_out_variable(var, nir) {
      if (var->data.location == VARYING_SLOT_LAYER || var->data.location == VARYING_SLOT_VIEWPORT) {
         var->data.mode = nir_var_shader_temp;
         progress = true;
      }
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

/* gl_Layer in a fragment shader, and the view index: always the first,
 * as a pass is one layer. TODO: a pass per layer for layered blits. */
static bool
lower_layer_id_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_layer_id && intr->intrinsic != nir_intrinsic_load_view_index)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def_replace(&intr->def, nir_imm_zero(b, intr->def.num_components, intr->def.bit_size));
   return true;
}

/* Sampling an integer texture: WGSL has only fetches of those. Vulkan
 * samples integer formats nearest, so the fetch is of the texel the
 * coordinate falls in, clamped to the edge -- which is what vk_meta's and
 * nearly every application's samplers for these do. */
static bool
lower_integer_sample_instr(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (nir_alu_type_get_base_type(tex->dest_type) == nir_type_float || tex->is_shadow)
      return false;
   if (tex->op != nir_texop_tex && tex->op != nir_texop_txb && tex->op != nir_texop_txl &&
       tex->op != nir_texop_txd)
      return false;
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE)
      return false;

   b->cursor = nir_before_instr(instr);
   const int texture_idx = nir_tex_instr_src_index(tex, nir_tex_src_texture_deref);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   const int lod_idx = nir_tex_instr_src_index(tex, nir_tex_src_lod);
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   if (texture_idx < 0 || coord_idx < 0)
      return false;
   nir_def *texture = tex->src[texture_idx].src.ssa;
   nir_def *coord = tex->src[coord_idx].src.ssa;
   nir_def *lod = tex->op == nir_texop_txl && lod_idx >= 0
                     ? nir_f2i32(b, nir_fround_even(b, tex->src[lod_idx].src.ssa))
                     : nir_imm_int(b, 0);

   nir_tex_instr *txs = nir_tex_instr_create(b->shader, 2);
   txs->op = nir_texop_txs;
   txs->sampler_dim = tex->sampler_dim;
   txs->is_array = tex->is_array;
   txs->dest_type = nir_type_int32;
   txs->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, texture);
   txs->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_def_init(&txs->instr, &txs->def, nir_tex_instr_dest_size(txs), 32);
   nir_builder_instr_insert(b, &txs->instr);
   nir_def *size = &txs->def;

   const unsigned n = tex->coord_components;
   const unsigned spatial = n - (tex->is_array ? 1 : 0);
   nir_def *comps[4];
   for (unsigned i = 0; i < n; ++i) {
      nir_def *c = nir_channel(b, coord, i);
      nir_def *extent = nir_channel(b, size, i);
      nir_def *texel = i < spatial ? nir_f2i32(b, nir_ffloor(b, nir_fmul(b, c, nir_i2f32(b, extent))))
                                   : nir_f2i32(b, nir_fround_even(b, c));
      if (offset_idx >= 0 && i < spatial)
         texel = nir_iadd(b, texel, nir_channel(b, tex->src[offset_idx].src.ssa, i));
      comps[i] = nir_imax(b, nir_imin(b, texel, nir_iadd_imm(b, extent, -1)), nir_imm_int(b, 0));
   }

   nir_tex_instr *txf = nir_tex_instr_create(b->shader, 3);
   txf->op = nir_texop_txf;
   txf->sampler_dim = tex->sampler_dim;
   txf->is_array = tex->is_array;
   txf->dest_type = tex->dest_type;
   txf->coord_components = n;
   txf->src[0] = nir_tex_src_for_ssa(nir_tex_src_texture_deref, texture);
   txf->src[1] = nir_tex_src_for_ssa(nir_tex_src_coord, nir_vec(b, comps, n));
   txf->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, lod);
   nir_def_init(&txf->instr, &txf->def, tex->def.num_components, tex->def.bit_size);
   nir_builder_instr_insert(b, &txf->instr);
   nir_def_replace(&tex->def, &txf->def);
   return true;
}

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
   /* Whole words, a vector of them at most. */
   return (nir_mem_access_size_align){
      .num_components = MIN2(DIV_ROUND_UP(bytes, 4), 4),
      .bit_size = 32,
      .align = 4,
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
 * vulkan_resource_index, through load_vulkan_descriptor. */
static bool
index_binding(nir_src src, uint32_t *set, uint32_t *binding)
{
   nir_scalar s = nir_scalar_resolved(src.ssa, 0);
   for (unsigned depth = 0; depth < 8; ++depth) {
      if (!nir_scalar_is_intrinsic(s))
         return false;
      nir_intrinsic_instr *intr = nir_def_as_intrinsic(s.def);
      if (intr->intrinsic == nir_intrinsic_vulkan_resource_index) {
         *set = nir_intrinsic_desc_set(intr);
         *binding = nir_intrinsic_binding(intr);
         return true;
      }
      if (intr->intrinsic != nir_intrinsic_load_vulkan_descriptor &&
          intr->intrinsic != nir_intrinsic_vulkan_resource_reindex)
         return false;
      s = nir_scalar_resolved(intr->src[0].ssa, 0);
   }
   return false;
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

static bool
lower_to_words_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct word_buffers *wb = data;
   bool ubo = false;
   int index_src;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_ubo:
      ubo = true;
      FALLTHROUGH;
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_get_ssbo_size:
      index_src = 0;
      break;
   case nir_intrinsic_store_ssbo:
      index_src = 1;
      break;
   default:
      return false;
   }
   uint32_t set, binding;
   if (!index_binding(intr->src[index_src], &set, &binding)) {
      mesa_loge("wvk: a buffer access whose binding cannot be told");
      return false;
   }
   nir_variable *var = word_var(b->shader, wb, ubo, set, binding);
   b->cursor = nir_before_instr(&intr->instr);

   if (intr->intrinsic == nir_intrinsic_get_ssbo_size) {
      nir_deref_instr *words = nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
      nir_def *length = nir_deref_buffer_array_length(b, 32, &words->def, .access = ACCESS_NON_WRITEABLE);
      nir_def_replace(&intr->def, nir_imul_imm(b, length, 4));
      return true;
   }

   const int offset_src = intr->intrinsic == nir_intrinsic_store_ssbo ? 2 : 1;
   nir_def *first = nir_ushr_imm(b, intr->src[offset_src].ssa, 2);

   if (intr->intrinsic == nir_intrinsic_load_ubo || intr->intrinsic == nir_intrinsic_load_ssbo) {
      if (intr->def.bit_size != 32) {
         mesa_loge("wvk: a %u-bit buffer load", intr->def.bit_size);
         return false;
      }
      nir_def *comps[NIR_MAX_VEC_COMPONENTS];
      for (unsigned c = 0; c < intr->def.num_components; ++c)
         comps[c] = load_word(b, var, ubo, nir_iadd_imm(b, first, c));
      nir_def_replace(&intr->def, nir_vec(b, comps, intr->def.num_components));
      return true;
   }

   nir_deref_instr *words = nir_build_deref_struct(b, nir_build_deref_var(b, var), 0);
   if (intr->intrinsic == nir_intrinsic_store_ssbo) {
      nir_def *value = intr->src[0].ssa;
      if (value->bit_size != 32) {
         mesa_loge("wvk: a %u-bit buffer store", value->bit_size);
         return false;
      }
      u_foreach_bit(c, nir_intrinsic_write_mask(intr)) {
         nir_store_deref(b, nir_build_deref_array(b, words, nir_iadd_imm(b, first, c)),
                         nir_channel(b, value, c), 0x1);
      }
      nir_instr_remove(&intr->instr);
      return true;
   }

   /* Atomics, on one word. */
   nir_deref_instr *word = nir_build_deref_array(b, words, first);
   nir_def *result;
   if (intr->intrinsic == nir_intrinsic_ssbo_atomic_swap)
      result = nir_deref_atomic_swap(b, 32, &word->def, intr->src[2].ssa, intr->src[3].ssa,
                                     .atomic_op = nir_intrinsic_atomic_op(intr));
   else
      result = nir_deref_atomic(b, 32, &word->def, intr->src[2].ssa,
                                .atomic_op = nir_intrinsic_atomic_op(intr));
   nir_def_replace(&intr->def, result);
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
   nir_def *first = nir_load_base_instance(b);
   nir_def *element = first;
   if (vb->divisor)
      element = nir_iadd(b, first, nir_udiv_imm(b, nir_isub(b, nir_load_instance_id(b), first), vb->divisor));
   nir_def *starts = nir_load_ssbo(b, 4, 32, driver_buffer(b, WVK_DRIVER_SLOT_PUSH),
                                   nir_imm_int(b, WVK_DRIVER_PULL_OFFSET), .align_mul = 16,
                                   .access = ACCESS_NON_WRITEABLE);
   nir_def *address = nir_iadd(b, nir_channel(b, starts, slot),
                               nir_iadd_imm(b, nir_imul_imm(b, element, vb->stride), a->offset));

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
            if (glsl_get_bit_size(deref->type) != 32) {
               mesa_logw_once("wvk: a 64-bit attribute with a divisor");
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
            nir_def *value = fetch_attribute(&b, vi, location, slot);
            nir_def_replace(&intr->def, nir_trim_vector(&b, value, intr->def.num_components));
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
            nir_src_rewrite(&intr->src[1], nir_vec4(&b, x, y, nir_channel(&b, pos, 2), w));
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
   const nir_lower_mem_access_bit_sizes_options sizes = {
      .callback = word_access,
      .modes = nir_var_mem_ubo | nir_var_mem_ssbo,
   };
   NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &sizes);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   /* vtn's variables go; the words take their place. */
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_mem_ubo | nir_var_mem_ssbo)
      exec_node_remove(&var->node);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_to_words_instr, nir_metadata_control_flow, &wb);
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

/* A fragment output of fewer components than its attachment has, which
 * Vulkan leaves undefined and WebGPU refuses: every colour output is a
 * vec4, the rest zero. */
static bool
pad_fragment_outputs(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      return false;
   bool progress = false;
   nir_foreach_shader_out_variable(var, nir) {
      if (var->data.location < FRAG_RESULT_DATA0 || glsl_type_is_array(var->type))
         continue;
      const unsigned n = glsl_get_vector_elements(var->type);
      if (n == 4 || !glsl_type_is_vector_or_scalar(var->type))
         continue;
      const struct glsl_type *old = var->type;
      var->type = glsl_vector_type(glsl_get_base_type(old), 4);
      nir_foreach_function_impl(impl, nir) {
         nir_builder b = nir_builder_create(impl);
         nir_foreach_block(block, impl) {
            nir_foreach_instr_safe(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_store_deref ||
                   nir_intrinsic_get_var(intr, 0) != var)
                  continue;
               b.cursor = nir_before_instr(instr);
               nir_def *value = intr->src[1].ssa;
               nir_def *comps[4];
               for (unsigned c = 0; c < 4; ++c)
                  comps[c] = c < value->num_components ? nir_channel(&b, value, c)
                                                       : nir_imm_zero(&b, 1, value->bit_size);
               nir_src_rewrite(&intr->src[1], nir_vec(&b, comps, 4));
               nir_intrinsic_set_write_mask(intr, 0xf);
            }
         }
      }
      progress = true;
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
      } else if (glsl_type_is_image(type) && (var->data.access & ACCESS_NON_WRITEABLE)) {
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
   NIR_PASS(_, nir, lower_push_constants);
   NIR_PASS(_, nir, drop_layer_outputs);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_layer_id_instr, nir_metadata_control_flow, NULL);
   split_resource_vars(nir, &state);
   split_buffer_vars(nir, &state);
   NIR_PASS(_, nir, nir_shader_instructions_pass, rewrite_resource_instr, nir_metadata_control_flow,
            &state);
   NIR_PASS(_, nir, nir_shader_instructions_pass, lower_integer_sample_instr, nir_metadata_control_flow,
            NULL);
   NIR_PASS(_, nir, lower_1d_images);
   NIR_PASS(_, nir, lower_ms_arrays);
   NIR_PASS(_, nir, pad_fragment_outputs);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, demote_to_terminate_instr, nir_metadata_all, NULL);
   lower_buffers_to_words(nir, info, pipeline_state, shader, mem_ctx);
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

static VkResult
wvk_compile(struct vk_device *vk_dev, uint32_t shader_count, struct vk_shader_compile_info *infos,
            const struct vk_graphics_pipeline_state *state, const struct vk_features *enabled_features,
            const VkAllocationCallbacks *pAllocator, struct vk_shader **shaders_out)
{
   struct wvk_device *dev = container_of(vk_dev, struct wvk_device, vk);
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
