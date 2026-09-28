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
 * Nothing of a pipeline's state is compiled in: WebGPU pipelines are made
 * at the draw, from the modules here and whatever the state is then.
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

#define WVK_SHADER_BLOB_VERSION 1

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
   optimize(nir);
}

static void
wvk_hash_state(struct vk_physical_device *device, const struct vk_graphics_pipeline_state *state,
               const struct vk_features *enabled_features, VkShaderStageFlags stages,
               blake3_hash blake3_out)
{
   /* No state is compiled in. */
   struct mesa_blake3 ctx;
   _mesa_blake3_init(&ctx);
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
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_mem_ubo | nir_var_mem_ssbo) {
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
      const uint32_t count = glsl_type_is_array(var->type) ? glsl_get_aoa_size(var->type) : 1;
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
            const VkAllocationCallbacks *pAllocator, struct vk_shader **shader_out)
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
      VkResult result = compile_one(dev, &infos[i], pAllocator, &shaders_out[i]);
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
   blob_write_bytes(blob, shader->workgroup_size, sizeof(shader->workgroup_size));
   return !blob->out_of_memory;
}

static VkResult
wvk_shader_deserialize(struct vk_device *vk_dev, struct blob_reader *blob, uint32_t binary_version,
                       const VkAllocationCallbacks *pAllocator, struct vk_shader **shader_out)
{
   struct wvk_device *dev = container_of(vk_dev, struct wvk_device, vk);
   if (binary_version != WVK_SHADER_BLOB_VERSION)
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
