/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * NIR to WGSL; see gex_wgpu_nir.h. Moved out of gex_wgpu_screen.c as it
 * stood, so that the Vulkan driver compiles with exactly what GL does.
 */
#include "gex_wgpu_nir.h"
#include "gex_wgpu_tint.h"

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "nir_to_spirv.h"
#include "util/os_time.h"
#include "util/ralloc.h"
#include "util/u_math.h"
#include "util/u_memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool
gex_wgpu_lower_position_y(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   (void)data;
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_POS)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *pos = intr->src[1].ssa;
   nir_def *flipped = nir_vec4(b,
      nir_channel(b, pos, 0), nir_fneg(b, nir_channel(b, pos, 1)),
      nir_channel(b, pos, 2), nir_channel(b, pos, 3));
   nir_src_rewrite(&intr->src[1], flipped);
   return true;
}

/* textureQueryLod, computed rather than asked for.
 *
 * WGSL has no textureQueryLod and SPIR-V's OpImageQueryLod would only move
 * the failure into Tint, so the value is worked out from the coordinate the
 * way the GL specification defines it: scale the texture coordinate by the
 * texture's size, take its screen-space derivatives, and the larger of the
 * two lengths is the scale factor whose log2 is the level of detail.
 *
 * Returns what GL asks for -- x the level that would actually be accessed,
 * clamped into the mip chain, and y the unclamped computed value.
 *
 * Derivatives only exist in a fragment shader, which is where GL allows this
 * to be called; anywhere else it is left alone and refused later rather than
 * quietly given a wrong answer.
 */
static bool
gex_wgpu_lower_query_lod_instr(nir_builder *b, nir_instr *instr, void *data)
{
   (void)data;
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_lod)
      return false;
   if (b->shader->info.stage != MESA_SHADER_FRAGMENT)
      return false;

   const int coord_index = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (coord_index < 0)
      return false;
   nir_def *coord = tex->src[coord_index].src.ssa;
   /* The array layer is not a coordinate for this purpose: it does not
    * participate in the scale factor. */
   const unsigned dims = glsl_get_sampler_dim_coordinate_components(
      (enum glsl_sampler_dim)tex->sampler_dim);
   if (coord->num_components < dims)
      return false;

   b->cursor = nir_before_instr(instr);
   coord = nir_trim_vector(b, coord, dims);

   /* Texture size at the base level, as floats, to scale the coordinate into
    * texels -- the derivative of a normalised coordinate says nothing on its
    * own. */
   nir_def *size = nir_i2f32(
      b, nir_build_texture_query(b, tex, nir_texop_txs, dims, nir_type_int32,
                                 false, true));
   nir_def *scaled = nir_fmul(b, coord, size);
   nir_def *ddx = nir_ddx(b, scaled);
   nir_def *ddy = nir_ddy(b, scaled);
   nir_def *rho = nir_fmax(b, nir_fast_length(b, ddx),
                           nir_fast_length(b, ddy));
   /* log2(0) is negative infinity, and a derivative of zero is ordinary --
    * a fragment where the coordinate does not move wants the base level. */
   nir_def *lambda = nir_flog2(b, nir_fmax(b, rho, nir_imm_float(b, 1e-20f)));

   nir_def *levels = nir_i2f32(
      b, nir_build_texture_query(b, tex, nir_texop_query_levels, 1,
                                 nir_type_int32, false, false));
   nir_def *top = nir_fadd_imm(b, levels, -1.0f);
   nir_def *clamped = nir_fclamp(b, lambda, nir_imm_float(b, 0.0f), top);

   nir_def_replace(&tex->def, nir_vec2(b, clamped, lambda));
   return true;
}

/* Give a lone index-1 fragment output the index-0 partner WGSL insists on.
 *
 * WGSL allows `@blend_src` only on a *pair*, so an output at index 1 with no
 * index-0 output at the same location cannot be expressed and Tint rejects the
 * module -- a failed link, which is the whole program rather than one draw.
 * GL allows the shape and says index 0's value is undefined when nothing
 * writes it, so a synthesized output writing zeros satisfies both: any value
 * is conformant for GL, and the pair is what WGSL wants.
 *
 * Only the missing halves are added, and only for a fragment shader. A shader
 * that already declares both -- EEVEE's eevee_light_culling_debug, the case
 * the decoration rule beside this was written for -- is untouched. */
static void
gex_wgpu_add_missing_index_zero_output(nir_shader *nir)
{
   if (nir->info.stage != MESA_SHADER_FRAGMENT)
      return;
   uint32_t have_zero = 0, have_one = 0;
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
      if (var->data.location < FRAG_RESULT_DATA0)
         continue;
      const unsigned at = var->data.location - FRAG_RESULT_DATA0;
      if (at >= 32)
         continue;
      if (var->data.index == 0)
         have_zero |= 1u << at;
      else if (var->data.index == 1)
         have_one |= 1u << at;
   }
   const uint32_t missing = have_one & ~have_zero;
   if (!missing)
      return;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl)
      return;
   nir_builder b = nir_builder_at(nir_after_impl(impl));
   uint32_t left = missing;
   while (left) {
      const unsigned at = u_bit_scan(&left);
      char name[40];
      snprintf(name, sizeof(name), "gex_blend_src0_%u", at);
      nir_variable *out =
         nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), name);
      out->data.location = FRAG_RESULT_DATA0 + at;
      out->data.index = 0;
      out->data.driver_location = at;
      out->data.interpolation = INTERP_MODE_NONE;
      nir_store_var(&b, out, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 0.0f), 0xf);
   }
   nir_progress(true, impl, nir_metadata_none);
}

/* GL's old-style shadow sampling, which returns a vec4.
 *
 * `shadow2D` and `shadow2DProj` are GLSL 1.20 functions and they return a
 * vec4, not the scalar `texture(sampler2DShadow, ...)` returns. NIR carries
 * the difference as `is_new_style_shadow`, and SPIR-V has only the new form:
 * `OpImageSample*Dref*` produces a scalar, so emitting one with a vec4 result
 * type is invalid and Tint says so --
 *
 *   error: line:73: Expected Result Type to be int or float scalar type
 *
 * -- which rejected `shadowtex`'s whole fragment shader. zink hits this too
 * and answers it the same way (zink_compiler.c, lower_zs_swizzle_tex): sample
 * as new-style and expand the scalar back out in the shader.
 *
 * The expansion is (r, r, r, 1), which is what GL_DEPTH_TEXTURE_MODE's
 * default of GL_LUMINANCE asks for and what st/mesa's own view swizzle for a
 * depth texture says. GL_INTENSITY would want alpha to be r as well; that
 * needs the bound view, so it would have to move into the swizzle variant,
 * and no installed program asks for it. */
static bool
gex_wgpu_lower_old_style_shadow(nir_builder *b, nir_instr *instr, void *data)
{
   (void)data;
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   /* Queries are excluded, and the exclusion is not cosmetic. A size query
    * derived from a shadow lookup keeps its is_shadow flag -- nir_lower_tex's
    * rect lowering builds one through nir_get_texture_size, which copies it
    * -- so a txs on a 2D shadow texture matched everything below and had its
    * two components collapsed to one and splatted. `sampler2DRectShadow`
    * then divided its coordinate by vec2(width, width), and Tint refused the
    * module with "Result Type has 1 components, but 2 expected". */
   if (!tex->is_shadow || tex->is_new_style_shadow ||
       nir_tex_instr_is_query(tex) || tex->def.num_components < 2 ||
       tex->op == nir_texop_tg4 || tex->is_sparse)
      return false;
   const unsigned components = tex->def.num_components;
   tex->def.num_components = 1;
   tex->is_new_style_shadow = true;
   b->cursor = nir_after_instr(instr);
   nir_def *channels[4];
   for (unsigned i = 0; i < components; ++i)
      channels[i] = i == 3 ? nir_imm_float(b, 1.0f) : &tex->def;
   nir_def *expanded = nir_vec(b, channels, components);
   nir_def_rewrite_uses_after(&tex->def, expanded);
   return true;
}

/* gl_PointSize, which WGSL has no builtin for.
 *
 * Tint accepts a PointSize store only when it writes the constant 1.0 --
 * WebGPU draws every point one pixel wide -- and rejects the whole module
 * otherwise, which is a link failure the application sees as an empty info
 * log. GL programs set it routinely, so the store is dropped rather than the
 * program refused. What that gives up is point size, which this backend
 * cannot honour in any case.
 */
static bool
gex_wgpu_strip_point_size_instr(nir_builder *b, nir_intrinsic_instr *intr,
                                void *data)
{
   (void)b;
   (void)data;
   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   if (!var || var->data.mode != nir_var_shader_out ||
       var->data.location != VARYING_SLOT_PSIZ)
      return false;
   nir_instr_remove(&intr->instr);
   return true;
}

/* A barrier's memory modes, narrowed to the address spaces WebGPU has.
 *
 * zink's emitter turns nir_var_mem_global into
 * SpvMemorySemanticsCrossWorkgroupMemoryMask, and Tint's SPIR-V reader
 * refuses it:
 *
 *   error: unsupported control barrier semantics: 512
 *
 * 512 is that bit. WGSL has no cross-workgroup address space at all -- a
 * storage buffer is the widest thing a barrier can order here -- so global
 * is folded into ssbo, which is what the memory actually is.
 *
 * The no-modes case has to be filled in rather than left alone: the emitter
 * reads an empty set as "everything", and its everything includes the same
 * bit. Naming the three spaces that do exist keeps the barrier as strong as
 * the shader asked for without the one Tint cannot read. Blender's EEVEE has
 * such a barrier in a compute shader. */
static bool
gex_wgpu_narrow_barrier_instr(nir_builder *b, nir_intrinsic_instr *intr,
                              void *data)
{
   (void)b;
   (void)data;
   if (intr->intrinsic != nir_intrinsic_barrier)
      return false;
   if (nir_intrinsic_memory_scope(intr) == SCOPE_NONE)
      return false;
   const nir_variable_mode modes = nir_intrinsic_memory_modes(intr);
   const nir_variable_mode wanted =
      !modes ? (nir_var_mem_shared | nir_var_mem_ssbo | nir_var_image)
             : ((modes & ~nir_var_mem_global) |
                ((modes & nir_var_mem_global) ? nir_var_mem_ssbo : 0));
   bool progress = false;
   if (wanted != modes) {
      nir_intrinsic_set_memory_modes(intr, wanted);
      progress = true;
   }
   /* A barrier with no *execution* scope is GLSL's memoryBarrier() family,
    * which zink emits as OpMemoryBarrier -- and Tint's SPIR-V reader does not
    * implement that instruction at all:
    *
    *   error: unhandled instruction with opcode 225: OpMemoryBarrier
    *
    * WGSL has no standalone memory barrier; storageBarrier() and
    * workgroupBarrier() are control barriers. Giving the barrier the workgroup
    * execution scope makes zink emit OpControlBarrier instead, which Tint
    * takes -- and a control barrier is *stronger* than the memory barrier it
    * replaces, so the ordering the shader asked for still holds.
    *
    * Only in compute: WGSL has no barrier of any kind in the other stages, so
    * one there stays a named refusal rather than becoming an illegal module.
    * Three of KHR-GL44.compute_shader's shaders are rejected for this.
    *
    * **But a control barrier is not stronger in every respect.** It carries
    * one requirement a memory barrier does not: it must be reached in uniform
    * control flow. GLSL allows memoryBarrier*() anywhere, and shaders put one
    * inside a branch on gl_LocalInvocationID -- KHR-GL44's
    * compute_shader.shared-struct has memoryBarrierBuffer() in one arm and
    * memoryBarrierShared() in the other, with a comment saying the calls are
    * not needed and are there to check the compiler accepts them. Substituting
    * a control barrier there produced a module Tint rejects outright:
    *
    *   error: 'workgroupBarrier' must only be called from uniform control flow
    *
    * -- and a rejected module loses the whole shader, which is worse than
    * anything the fence was buying. So the substitution only happens where
    * uniformity is guaranteed, and elsewhere the pure memory barrier is
    * dropped.
    *
    * Dropping it is a real weakening and worth being explicit about: it
    * removes a fence the shader asked for. What makes it defensible is that a
    * standalone memory barrier orders nothing observable on its own -- without
    * a control barrier there is no execution ordering for the fence to be
    * relative to -- and any shader that does pair one with `barrier()` has
    * that barrier in uniform control flow, where it carries the memory
    * semantics itself. This is the trade [[crashes-hide-behind-close-enough]]
    * describes: the nearest available construct is not the same construct,
    * and the difference has to be checked rather than assumed.
    *
    * "Uniform" is judged conservatively as "not nested in any if or loop".
    * That is stricter than the rule -- a branch on a uniform is uniform too --
    * but it needs no analysis and errs towards dropping a fence rather than
    * emitting a module that cannot be read. */
   /* Outside compute there is no barrier of any kind in WGSL -- not even the
    * control barrier the compute path substitutes -- so a pure memory barrier
    * there has nowhere to go and was left as OpMemoryBarrier, which Tint
    * cannot read:
    *
    *   error: unhandled instruction with opcode 225: OpMemoryBarrier
    *
    * losing the shader. It is dropped instead, on the same reasoning as the
    * nested case below: a standalone memory barrier orders nothing observable
    * on its own, and in these stages nothing can pair it with a control
    * barrier, because there is none to pair it with. A barrier that *does*
    * carry an execution scope outside compute is a different matter and stays
    * a named refusal -- GL only allows one in a tessellation control shader,
    * and this driver runs those as compute, so it would mean something has
    * gone wrong rather than that a feature is missing.
    * KHR-GL44's shader_storage_buffer_object.advanced-matrix is this. */
   if (b->shader->info.stage != MESA_SHADER_COMPUTE &&
       nir_intrinsic_execution_scope(intr) == SCOPE_NONE) {
      static bool said;
      if (!said) {
         said = true;
         fprintf(stderr, "gex-wgpu: a memoryBarrier outside a compute shader "
                         "is dropped; WGSL has no barrier in that stage\n");
      }
      nir_instr_remove(&intr->instr);
      return true;
   }
   if (b->shader->info.stage == MESA_SHADER_COMPUTE &&
       nir_intrinsic_execution_scope(intr) == SCOPE_NONE) {
      bool nested = false;
      for (nir_cf_node *node = intr->instr.block->cf_node.parent;
           node && !nested; node = node->parent)
         nested = node->type == nir_cf_node_if ||
                  node->type == nir_cf_node_loop;
      if (nested) {
         static bool said;
         if (!said) {
            said = true;
            fprintf(stderr, "gex-wgpu: a memoryBarrier inside control flow "
                            "is dropped; WGSL has no standalone memory "
                            "barrier and a control barrier would need "
                            "uniform control flow\n");
         }
         nir_instr_remove(&intr->instr);
         return true;
      }
      nir_intrinsic_set_execution_scope(intr, SCOPE_WORKGROUP);
      progress = true;
   }
   /* And the *memory* scope has to be the workgroup as well:
    *
    *   error: storageBarrier requires workgroup memory scope
    *
    * GLSL's memoryBarrierBuffer() asks for device scope, and WGSL has no
    * barrier of any scope wider than a workgroup -- storageBarrier() and
    * workgroupBarrier() are both workgroup-scoped. So the widest thing
    * available is taken. That is weaker than what the shader wrote, and it is
    * the same ceiling as the cross-workgroup semantics folded away above:
    * nothing here can order writes between workgroups, because WebGPU has no
    * way to express it. Within a workgroup -- which is what a compute shader's
    * shared memory and its own storage writes need -- it is exact. */
   if (b->shader->info.stage == MESA_SHADER_COMPUTE &&
       nir_intrinsic_memory_scope(intr) != SCOPE_NONE &&
       nir_intrinsic_memory_scope(intr) != SCOPE_WORKGROUP &&
       nir_intrinsic_memory_scope(intr) != SCOPE_SUBGROUP &&
       nir_intrinsic_memory_scope(intr) != SCOPE_INVOCATION) {
      nir_intrinsic_set_memory_scope(intr, SCOPE_WORKGROUP);
      progress = true;
   }
   return progress;
}

static bool
gex_wgpu_lower_point_coord_instr(nir_builder *b, nir_intrinsic_instr *intr,
                                 void *data)
{
   (void)data;
   if (intr->intrinsic != nir_intrinsic_load_point_coord &&
       intr->intrinsic != nir_intrinsic_load_point_coord_maybe_flipped)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *centre = nir_imm_vec2(b, 0.5f, 0.5f);
   nir_def_replace(&intr->def, centre);
   return true;
}

static void
gex_wgpu_strip_point_size(nir_shader *nir)
{
   if (!(nir->info.outputs_written & VARYING_BIT_PSIZ))
      return;
   nir_shader_intrinsics_pass(nir, gex_wgpu_strip_point_size_instr,
                              nir_metadata_control_flow, NULL);
   /* Dropping the store leaves the deref that addressed it, and unlinking the
    * variable out from under a live deref is a dangling reference:
    *
    *   %3 = deref_var &gl_PointSize (shader_out highp float)
    *        error: entry (nir_validate.c:323)
    *
    * which aborts the process rather than merely mis-compiling, so any shader
    * that set gl_PointSize took the driver down with it. DCE first, so the
    * deref goes with the store and the variable is unused by the time it is
    * removed. */
   NIR_PASS(_, nir, nir_opt_dce);
   nir_foreach_variable_with_modes_safe(var, nir, nir_var_shader_out) {
      if (var->data.location == VARYING_SLOT_PSIZ)
         exec_node_remove(&var->node);
   }
   nir->info.outputs_written &= ~VARYING_BIT_PSIZ;
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

/* Four, matching zink. The reductions this exists for are lowered regardless
 * of the width; the width only decides whether wide ALU is split. */
static uint8_t
gex_wgpu_alu_target_width(const nir_instr *instr, const void *data)
{
   (void)instr;
   (void)data;
   return 4;
}

static void
gex_wgpu_tex_log(nir_shader *nir, const char *where)
{
   if (!getenv("GEX_WGPU_TEX_LOG"))
      return;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *t = nir_instr_as_tex(instr);
            fprintf(stderr, "gex-wgpu: tex[%s] op=%d dim=%d array=%d "
                    "shadow=%d new=%d comps=%u want=%u\n", where, (int)t->op,
                    (int)t->sampler_dim, (int)t->is_array, (int)t->is_shadow,
                    (int)t->is_new_style_shadow, t->def.num_components,
                    nir_tex_instr_result_size(t));
         }
      }
   }
}

/* Whether a texture instruction carries an offset that is not a constant.
 *
 * Tint's SPIR-V reader accepts the `ConstOffset` image operand and rejects the
 * dynamic `Offset` one:
 *
 *   error: unsupported image operands (16): %34 = OpImageGather ...
 *
 * -- 16 being SpvImageOperandsOffsetMask. WGSL's offset parameter must be a
 * const-expression, so there is nothing to translate it to. GL 4.0 does allow
 * a non-constant offset for textureGatherOffset (ARB_gpu_shader5 relaxed the
 * constant requirement), so this is a real GL feature and not a shader bug,
 * and every one of KHR-GL44.texture_gather's fifteen offset-gather cases is
 * it -- including the float ones, which have nothing else wrong with them.
 *
 * nir_lower_tex folds the offset into the coordinate instead, which is what
 * the offset means: texels for a rect sampler, offset/textureSize for a
 * normalised one, and never applied to an array layer. That is Mesa's own
 * lowering rather than a local approximation, so it agrees with the
 * reference.
 *
 * A *constant* offset must not be lowered: `ConstOffset` is exact, while the
 * lowering divides by the texture size and lands on a texel boundary, which
 * is precisely where a gather's footprint is ambiguous. */
static bool
gex_wgpu_tex_offset_is_dynamic(const nir_instr *instr, const void *data)
{
   (void)data;
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex((nir_instr *)instr);
   const int at = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   return at >= 0 && !nir_src_as_const_value(tex->src[at].src);
}

/* ntv indexes its shared and scratch blocks by *word*, not by byte.
 *
 * Those blocks are `array<uintN, size/(N/8)>` in a wrapper struct, and
 * emit_load_special / emit_store_special use the intrinsic's offset directly
 * as the access-chain index -- adding only the component number to it. NIR's
 * load_shared and store_shared offsets are in **bytes**, so they have to be
 * divided first. zink does this in rewrite_bo_access_instr; this driver reuses
 * ntv without zink's pipeline around it, and never did.
 *
 * Atomics are deliberately not touched: emit_shared_atomic_intrinsic divides
 * by the element size itself, so scaling them here would divide twice. The two
 * halves of ntv disagree about the units and each caller has to match.
 *
 * The symptom was not a refusal. A byte offset used as a word index writes
 * four times too far, and WGSL clamps an out-of-range constant-size array
 * index rather than trapping -- so `shared uint lane[64]` kept only the
 * sixteen entries whose scaled index still landed inside it, and read back
 * numbers that looked like arithmetic rather than corruption:
 * `shared uint lane[64]` summed to 3144 instead of 2016, and a shared struct
 * array to 120, which is exactly sum(0..15). Every KHR-GL44 compute_shader
 * shared-memory case failed on values with nothing in any log. */
static bool
gex_wgpu_scale_shared_offset(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   (void)data;
   unsigned bit_size;
   unsigned at;
   switch (intr->intrinsic) {
   case nir_intrinsic_load_shared:
   case nir_intrinsic_load_scratch:
      bit_size = intr->def.bit_size;
      at = 0;
      break;
   case nir_intrinsic_store_shared:
   case nir_intrinsic_store_scratch:
      bit_size = nir_src_bit_size(intr->src[0]);
      at = 1;
      break;
   default:
      return false;
   }
   if (bit_size < 8)
      return false;
   b->cursor = nir_before_instr(&intr->instr);
   nir_src_rewrite(&intr->src[at],
                   nir_udiv_imm(b, intr->src[at].ssa, bit_size / 8));
   return true;
}

/* Rebuild a deref chain at the builder's cursor, so a second user of it gets
 * an access chain of its own rather than sharing one. */
static nir_deref_instr *
gex_wgpu_clone_deref(nir_builder *b, nir_deref_instr *deref)
{
   switch (deref->deref_type) {
   case nir_deref_type_var:
      return nir_build_deref_var(b, deref->var);
   case nir_deref_type_struct: {
      nir_deref_instr *parent =
         gex_wgpu_clone_deref(b, nir_deref_instr_parent(deref));
      return parent ? nir_build_deref_struct(b, parent, deref->strct.index)
                    : NULL;
   }
   case nir_deref_type_array: {
      nir_deref_instr *parent =
         gex_wgpu_clone_deref(b, nir_deref_instr_parent(deref));
      return parent ? nir_build_deref_array(b, parent, deref->arr.index.ssa)
                    : NULL;
   }
   default:
      return NULL;
   }
}

/* No atomic may share its deref with another instruction.
 *
 * This is the same class of defect as the block-crossing deref above, and it
 * has the same cause: ntv turns one NIR deref into one OpAccessChain and
 * reuses it. Where the users are two *atomics*, Tint's SPIR-V reader retypes
 * the pointed-at element as `atomic<u32>` and then cannot express a shared
 * pointer to it:
 *
 *   cannot take the address of value of type 'ptr<storage, atomic<u32>,
 *   read_write>'
 *
 * A sub-word shader image store is exactly two atomics on one word -- clear
 * this texel's field, then set it -- so it produced this every time, while
 * either atomic alone compiled. That asymmetry is what identified it; the
 * message names neither the image nor the atomics.
 *
 * After nir_rematerialize_derefs_in_use_blocks_impl, which is per *block* and
 * so leaves two atomics in one block sharing, and after every optimisation --
 * common subexpression elimination would simply merge the copies again. */
static void
gex_wgpu_split_atomic_derefs(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_deref_atomic &&
                intr->intrinsic != nir_intrinsic_deref_atomic_swap)
               continue;
            nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
            if (!deref)
               continue;
            b.cursor = nir_before_instr(instr);
            nir_deref_instr *copy = gex_wgpu_clone_deref(&b, deref);
            if (copy)
               nir_src_rewrite(&intr->src[0], &copy->def);
         }
      }
   }
   /* The originals are unreferenced now where every user was an atomic. */
   NIR_PASS(_, nir, nir_opt_dce);
}

/* Whether anything 64-bit is left in the shader at all.
 *
 * The sweep below is only worth its passes -- one of which flattens indirect
 * indexing and can grow a big shader -- when there is something for them to
 * find. Almost every shader has nothing, and pays one walk. */
bool
gex_wgpu_has_64_bit(nir_shader *nir)
{
   nir_foreach_variable_in_shader(var, nir)
      if (glsl_type_contains_64bit(var->type))
         return true;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               const nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->def.bit_size == 64)
                  return true;
               for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
                  if (alu->src[i].src.ssa->bit_size == 64)
                     return true;
            } else if (instr->type == nir_instr_type_intrinsic) {
               const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (nir_intrinsic_infos[intr->intrinsic].has_dest &&
                   intr->def.bit_size == 64)
                  return true;
            } else if (instr->type == nir_instr_type_load_const) {
               if (nir_instr_as_load_const(instr)->def.bit_size == 64)
                  return true;
            }
         }
      }
   }
   return false;
}

/* Which ALU operations the late 64-bit sweep scalarises: the 64-bit ones, as
 * nir_lower_doubles and nir_lower_int64 both want them. */
static bool
gex_wgpu_is_64_bit_alu(const nir_instr *instr, const void *data)
{
   (void)data;
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (alu->def.bit_size == 64)
      return true;
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i)
      if (alu->src[i].src.ssa->bit_size == 64)
         return true;
   return false;
}

/* An undefined value is the fourth construct with a bit size that no ALU
 * lowering reaches. nir_lower_int64 rewrites operations; an undef is not one,
 * so a 64-bit one walks straight through and reaches the emitter as
 * OpUndef %ulong -- and Tint refuses the module for the type, exactly as it
 * would for a real 64-bit operation. It is not a rare shape either: the fp64
 * builtin tests reach one whenever a value is conditionally assigned, and
 * unpack_64_2x32_split reads it, which is what kept it alive through DCE.
 *
 * Rebuilding it as a pack of two 32-bit undefs says the same thing in a type
 * the emitter can spell, and the fold that follows cancels the pack against
 * those unpacks. */
static bool
gex_wgpu_split_64bit_undef(nir_shader *nir)
{
   bool any = false;
   nir_foreach_function_impl(impl, nir) {
      bool progress = false;
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_undef)
               continue;
            nir_undef_instr *undef = nir_instr_as_undef(instr);
            if (undef->def.bit_size != 64)
               continue;
            b.cursor = nir_after_instr(instr);
            nir_def *parts[NIR_MAX_VEC_COMPONENTS];
            for (unsigned c = 0; c < undef->def.num_components; ++c)
               parts[c] = nir_pack_64_2x32_split(&b, nir_undef(&b, 1, 32),
                                                 nir_undef(&b, 1, 32));
            nir_def_replace(&undef->def,
                            nir_vec(&b, parts, undef->def.num_components));
            progress = true;
         }
      }
      any |= nir_progress(progress, impl, nir_metadata_control_flow);
   }
   return any;
}

/* Interpolation sampling, added to a pre-fragment stage's output struct.
 *
 * WebGPU requires the two stages to declare the same interpolation
 * *sampling*, and rejects the pipeline otherwise. GL puts `centroid` and
 * `sample` on the fragment input alone, so the vertex side never carries one
 * and every such varying mismatched. SPIR-V cannot express the agreement:
 * SpvDecorationCentroid and SpvDecorationSample are defined for fragment
 * inputs, an output of a pre-fragment stage carrying them is not valid, and
 * Tint drops them -- which is why ntv deliberately does not emit them there.
 *
 * So the agreement is written into the WGSL text, which is the last point at
 * which the two sides are both visible and still editable.
 *
 * Two things make the edit narrow enough to trust. The location Tint prints
 * is `var->data.driver_location`, the same number on both sides -- that is
 * what makes @location match at all today. And the edit has to stay inside
 * the *output struct*: a vertex shader's own attributes are @location too, on
 * the entry point's parameters, and `@location(0) pos_param` collides
 * textually with the `@location(0)` of the first varying. The output struct is
 * the one containing @builtin(position), which is what this searches for.
 *
 * `sampling` is indexed by location and holds the varying_aux bits: 1 for
 * centroid, 2 for sample, 0 to leave alone. Returns a fresh malloc'd string,
 * or the original pointer when there is nothing to do. */
static char *
gex_wgpu_wgsl_add_output_sampling(char *wgsl, const uint8_t *sampling,
                                  unsigned count)
{
   unsigned wanted = 0;
   for (unsigned i = 0; i < count; ++i)
      if (sampling[i])
         ++wanted;
   if (!wanted)
      return wgsl;

   /* The output struct: find the `struct ... {` whose body carries
    * @builtin(position). */
   const char *pos = strstr(wgsl, "@builtin(position)");
   if (!pos)
      return wgsl;
   const char *open = NULL;
   for (const char *p = pos; p > wgsl; --p) {
      if (*p == '{') { open = p; break; }
   }
   if (!open)
      return wgsl;
   const char *close = strchr(open, '}');
   if (!close)
      return wgsl;

   const size_t len = strlen(wgsl);
   /* Each insertion adds at most `@interpolate(perspective, centroid) `. */
   char *out = malloc(len + (size_t)wanted * 48 + 1);
   if (!out)
      return wgsl;
   size_t w = 0;
   const char *p = wgsl;
   while (*p) {
      if (p >= open && p < close && !strncmp(p, "@location(", 10)) {
         const char *digits = p + 10;
         char *end = NULL;
         const unsigned long loc = strtoul(digits, &end, 10);
         if (end && *end == ')' && loc < count && sampling[loc]) {
            const char *how = (sampling[loc] & 1u) ? "centroid" : "sample";
            /* Whatever ntv said about the *mode* is already here for a
             * non-default one -- `@interpolate(linear)` for noperspective.
             * Keep that word and add the sampling to it; a mode this rewrote
             * would disagree with the fragment side just as loudly as a
             * missing sampling does. */
            const char *after = end + 1;
            while (*after == ' ' || *after == '\n' || *after == '\t')
               ++after;
            char mode[16] = "perspective";
            const char *resume = end + 1;
            if (!strncmp(after, "@interpolate(", 13)) {
               const char *arg = after + 13;
               size_t n = 0;
               while (n < sizeof(mode) - 1 && arg[n] && arg[n] != ',' &&
                      arg[n] != ')')
                  ++n;
               /* `@interpolate(flat)` has no sampling to speak of and WGSL
                * does not accept one, so a flat varying is left as it is. */
               if (n && strncmp(arg, "flat", 4)) {
                  memcpy(mode, arg, n);
                  mode[n] = '\0';
                  const char *shut = strchr(arg, ')');
                  if (shut)
                     resume = shut + 1;
               } else {
                  resume = NULL;
               }
            }
            if (resume) {
               w += (size_t)snprintf(out + w, 48 + 24,
                                     "@location(%lu) @interpolate(%s, %s)",
                                     loc, mode, how);
               p = resume;
               continue;
            }
         }
      }
      out[w++] = *p++;
   }
   out[w] = '\0';
   free(wgsl);
   return out;
}

char *
gex_wgpu_nir_to_wgsl(nir_shader *nir, mesa_shader_stage stage, bool flip_y,
                     bool clip_halfz, struct gex_wgpu_wgsl_info *out,
                     char **error_out)
{
   *error_out = NULL;
   const nir_lower_tex_options texture_options = {
      .lower_1d = true,
      .lower_rect = true,
      /* Projective lookups, divided out in NIR rather than left to SPIR-V.
       *
       * SPIR-V's OpImageSample*Proj*Dref* does not project the depth
       * reference -- the coordinate carries the divisor and D_ref is a
       * separate operand that is used as it stands -- while GL's
       * shadow2DProj compares against coord.p / coord.q. zink's emitter
       * passes the reference straight through, so every projective shadow
       * comparison used an undivided reference: `shadow2DProj` answered zero
       * everywhere, which for a shadow map means fully shadowed and looks
       * like a depth range or bias problem rather than a missing divide.
       *
       * nir_lower_tex's project_src divides the coordinate *and* the
       * comparator, so lowering here removes the whole question. Non-shadow
       * projection is lowered with it: WGSL has no projective sampling
       * either, so Tint was undoing it again, and one frcp plus a multiply is
       * what that costs. */
      .lower_txp = ~0u,
      /* texelFetchOffset. SPIR-V allows ConstOffset on OpImageFetch and Tint
       * does not:
       *
       *   error: ConstOffset is only permitted for sampling, gather, or
       *   depth-reference gather operations:
       *     %41 = OpImageFetch %38 %40 %18 Lod|ConstOffset %16 %20
       *
       * and a rejected module means the program does not link at all -- the
       * integer-texture probe died at glLinkProgram, not at a wrong texel.
       * Adding the offset to the coordinate is what the offset means for a
       * fetch, since a fetch takes texels and never wraps or filters. */
      .lower_txf_offset = true,
      /* textureGatherOffset with a non-constant offset; see the filter. */
      .lower_offset_filter = gex_wgpu_tex_offset_is_dynamic,
   };
   gex_wgpu_tex_log(nir, "entry");
   NIR_PASS(_, nir, nir_lower_tex, &texture_options);

   /* ARB_shader_group_vote, with the invocation group defined as one
    * invocation.
    *
    * The three ballots reach NIR as vote_any, vote_all and vote_ieq, and the
    * chain that would carry them to the device is complete but for one link:
    * zink's emitter has SpvOpGroupNonUniformAny/All/AllEqual, the device
    * grants WGPUFeatureName_Subgroups, and WGSL has subgroupAny and
    * subgroupAll -- but Tint's SPIR-V *reader* has never heard of the opcodes
    * (they appear in its writer and nowhere else), so a shader using them is
    * refused outright and the program fails to link.
    *
    * GL's group is implementation-defined and may be a single invocation, so
    * the ballots are answerable without any subgroup operation at all:
    * anyInvocationARB and allInvocationsARB are then the value itself, and
    * allInvocationsEqualARB is true. That is what nir_lower_subgroups does
    * with lower_vote_trivial, and it is what zink itself does for a stage the
    * device does not support subgroups in -- while still claiming the
    * extension. Nothing OpGroupNonUniform is emitted, so Tint's reader is
    * never asked.
    *
    * This is a real behavioural difference from a driver that votes in wider
    * groups, not merely an implementation detail: llvmpipe answers
    * allInvocationsARB of a condition only one invocation sees with false,
    * and this answers true. gl44-group-vote-probe's `all_mixed` case exists
    * to say so out loud. It is claimed only because
    * KHR-GL45.shader_group_vote's five cases accept it; see the note beside
    * caps->shader_group_vote. */
   {
      const nir_lower_subgroups_options vote_options = {
         .subgroup_size = 1,
         .ballot_bit_size = 32,
         .ballot_components = 1,
         .lower_vote_trivial = true,
      };
      NIR_PASS(_, nir, nir_lower_subgroups, &vote_options);
   }
   gex_wgpu_tex_log(nir, "post-lower-tex");
   bool lowered_texture_declaration = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_uniform) {
      const struct glsl_type *type = glsl_without_array(var->type);
      if (!glsl_type_is_texture(type) ||
          (glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_1D &&
           glsl_get_sampler_dim(type) != GLSL_SAMPLER_DIM_RECT))
         continue;
      const bool is_array = glsl_sampler_type_is_array(type);
      var->type = glsl_texture_type(
         GLSL_SAMPLER_DIM_2D, is_array,
         glsl_get_sampler_result_type(type));
      lowered_texture_declaration = true;
   }
   if (lowered_texture_declaration)
      nir_fixup_deref_types(nir);
   if (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_GEOMETRY)
      gex_wgpu_strip_point_size(nir);
   if (stage == MESA_SHADER_FRAGMENT) {
      /* GLSL ES 1.00 and desktop GLSL up to 1.20 write gl_FragColor, which
       * st/mesa keeps as FRAG_RESULT_COLOR: one output broadcast to every
       * draw buffer.  WebGPU has no such thing -- a colour target must have a
       * matching @location output -- so expand it into FRAG_RESULT_DATA0..n,
       * which is the only form the WGSL emission below produces.  Without
       * this the pipeline has a colour target and no fragment output, and
       * WebGPU rejects it.
       */
      NIR_PASS(_, nir, nir_lower_fragcolor,
               nir->info.fs.color_is_dual_source ? 1
                                                 : GEX_WGPU_MAX_COLOR_TARGETS);
      /* `discard` under a condition arrives as the `terminate_if` intrinsic,
       * and the SPIR-V emitter below implements only the unconditional
       * `terminate`. It did not refuse it -- it hit an `unreachable`, which is
       * a wasm trap: the shader compile died mid-call, and what surfaced was
       * st/mesa carrying on from the wreckage with a negative buffer-map width
       * and a 2.3 GB allocation. Turn it into ordinary control flow first,
       * which is exactly what zink does before the same emitter
       * (`zink_compiler.c:3057`). `demote_if` goes the same way. */
      NIR_PASS(_, nir, nir_lower_discard_if,
               nir_lower_demote_if_to_cf | nir_lower_terminate_if_to_cf);
   }
   if (stage == MESA_SHADER_VERTEX) {
      /* WGSL requires a vertex entry point to return @builtin(position), and
       * Tint enforces it: "a vertex shader must include the 'position'
       * builtin in its return type". GL does not -- a transform-feedback
       * program legitimately writes only its capture varyings and never
       * gl_Position, because it runs with the rasterizer discarded. Ten of
       * the transform_feedback cases failed to link for exactly that.
       *
       * Writing a constant is correct rather than merely sufficient: the
       * position of a shader that declares none is never consumed, since
       * nothing downstream rasterizes. */
      if (!(nir->info.outputs_written & VARYING_BIT_POS)) {
         nir_variable *pos = nir_variable_create(
            nir, nir_var_shader_out, glsl_vec4_type(), "gex_position");
         pos->data.location = VARYING_SLOT_POS;
         nir_function_impl *impl = nir_shader_get_entrypoint(nir);
         nir_builder b = nir_builder_at(nir_after_impl(impl));
         nir_store_var(&b, pos, nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 1.0f), 0xf);
         nir->info.outputs_written |= VARYING_BIT_POS;
         nir_progress(true, impl, nir_metadata_control_flow);
      }
      /* GL's default clip space runs z from -1 to 1 and WebGPU's runs 0 to 1,
       * so the default needs converting -- but `glClipControl(...,
       * GL_ZERO_TO_ONE)` says the shader already writes z in [0, 1] and the
       * conversion must **not** happen. Running it unconditionally
       * transformed such a shader twice: z = 0.5 landed at 0.75, z = 0.25 at
       * 0.625, which is (z + 1) / 2 applied again.
       *
       * That is silent -- no error and no refusal, every fragment simply at
       * the wrong depth -- and GL_ZERO_TO_ONE is half of **reverse-Z**, which
       * is how a modern renderer gets usable depth precision at distance. The
       * driver advertised caps->clip_halfz and *accepted* the rasterizer
       * state while nothing read it. */
      if (!clip_halfz)
         NIR_PASS(_, nir, nir_lower_clip_halfz);
      if (flip_y)
         nir_shader_intrinsics_pass(
            nir, gex_wgpu_lower_position_y,
            nir_metadata_control_flow | nir_metadata_loop_analysis, NULL);
   }
   /* Vector comparison reductions, and anything wider than a vec4.
    *
    * `a.rgb == b.rgb` in GLSL is one `ball_fequal3` instruction, and zink's
    * SPIR-V emitter -- which this driver borrows whole -- has no case for
    * `ball_*` or `bany_*` at all: they fall through emit_alu's switch into
    * `unreachable`, which in wasm is a trap. The shader state creation died
    * mid-call, so glLinkProgram reported failure with an empty info log and
    * no driver diagnostic, and every KHR-GL33.pixelstoragemodes case failed
    * because its fragment shader compares `colour.rgb == refcolour.rgb`.
    *
    * nir_lower_alu_width expands those reductions into per-component feq plus
    * iand. The callback returns 4 for the same reason zink's does
    * (zink_compiler.c:5682): the reduction cases in that pass run ahead of the
    * width check, so this lowers what the emitter cannot take while leaving
    * ordinary vec4 ALU alone. */
   NIR_PASS(_, nir, nir_lower_alu_width, gex_wgpu_alu_target_width, NULL);
   /* Old-style (vec4) shadow sampling, which SPIR-V has no instruction for.
    * Before ntv, and in the base shader rather than a variant: nothing about
    * it depends on the bound view. */
   NIR_PASS(_, nir, nir_shader_instructions_pass,
            gex_wgpu_lower_old_style_shadow,
            nir_metadata_control_flow | nir_metadata_loop_analysis, NULL);
   /* textureQueryLod, which WGSL has no form of at all.
    *
    * This belongs here rather than in gex_wgpu_lower_texture_variant, where
    * it started: that pass only runs for shaders needing a sampler variant,
    * so a shader whose only unusual feature was the query went straight to
    * SPIR-V and Tint refused its OpImageQueryLod. Nothing in the lowering
    * depends on the bound view, so the base shader is the right place. */
   NIR_PASS(_, nir, nir_shader_instructions_pass,
            gex_wgpu_lower_query_lod_instr,
            nir_metadata_control_flow | nir_metadata_loop_analysis, NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            gex_wgpu_lower_point_coord_instr,
            nir_metadata_control_flow | nir_metadata_loop_analysis, NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass,
            gex_wgpu_narrow_barrier_instr,
            nir_metadata_all, NULL);
   /* Keep st/mesa's vertex input numbering across ntv_shader_prepare.
    *
    * That pass rewrites every vertex input's driver_location -- which becomes
    * the module's @location -- as
    *
    *   location >= VERT_ATTRIB_GENERIC0 ? location - VERT_ATTRIB_GENERIC0
    *                                    : location
    *
    * and those two ranges overlap. VERT_ATTRIB_POS is 0 and
    * VERT_ATTRIB_GENERIC0 is 16, so any shader using a fixed-function
    * attribute together with a generic one declares two inputs at
    * @location(0), and the SPIR-V is invalid before Tint even sees it:
    *
    *   [VUID-StandaloneSpirv-OpEntryPoint-08721] Entry-point has conflicting
    *   input location assignment at location 0, component 0
    *
    * st/mesa builds exactly that shape for its own fixed-function work --
    * VERT_ATTRIB_POS, VERT_ATTRIB_COLOR0 and VERT_ATTRIB_GENERIC0 together --
    * so glDrawPixels, glCopyPixels and the FBO clear paths all produced a
    * rejected vertex shader, which latched the context, after which every
    * texture_map returned NULL and the application segfaulted on it. That was
    * 17 of the 27 crashes in a sweep of the 336 installed GL programs, and
    * seven more refusals besides.
    *
    * st/mesa's own numbering does not collide: it is dense from zero, and it
    * is what Gallium's vertex elements are indexed by -- so it is also the
    * numbering the pipeline's shaderLocation uses. Keeping it makes the two
    * sides agree by construction rather than by a formula that has to be
    * mirrored. Keyed by variable pointer, because these passes may reorder
    * the variable list. */
   unsigned saved_locations[GEX_WGPU_MAX_VERTEX_ELEMENTS * 2];
   const nir_variable *saved_vars[GEX_WGPU_MAX_VERTEX_ELEMENTS * 2];
   unsigned saved_count = 0;
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_foreach_shader_in_variable(var, nir) {
         if (saved_count >= ARRAY_SIZE(saved_vars))
            break;
         saved_vars[saved_count] = var;
         saved_locations[saved_count] = var->data.driver_location;
         ++saved_count;
      }
   }
   gex_wgpu_tex_log(nir, "pre-ntv");
   ntv_shader_prepare(nir);
   if (nir->info.stage == MESA_SHADER_VERTEX) {
      nir_foreach_shader_in_variable(var, nir) {
         for (unsigned i = 0; i < saved_count; ++i) {
            if (saved_vars[i] == var) {
               var->data.driver_location = saved_locations[i];
               break;
            }
         }
      }
   }
   /* No deref may outlive the block it is used in.
    *
    * A NIR deref is an SSA value and ntv translates one into a single
    * OpAccessChain, reused wherever the deref is used. That is faithful
    * SPIR-V and valid -- spirv-val accepts it -- but a pointer cannot be held
    * across basic blocks in WGSL, so Tint's reader has to hoist it into a
    * function-scope `var` of pointer type and then refuses its own output:
    * "function-scope 'var' must have a constructible type".
    *
    * It takes a loop to produce. `uniform mat4 M` multiplied by a value a
    * loop computed put the access chain for M inside the loop body and its
    * four uses after the loop; neither the matrix nor the loop fails alone.
    * glmark2's three `loop` scenes are exactly that shader, and this is why
    * they would not link.
    *
    * Re-creating each deref in the block that uses it removes the crossing
    * altogether, and costs nothing anywhere else: a deref already local to
    * its block is left as it is. */
   nir_foreach_function_impl(impl, nir)
      nir_rematerialize_derefs_in_use_blocks_impl(impl);
   gex_wgpu_split_atomic_derefs(nir);
   /* A dual-source fragment shader must decorate *both* its outputs with an
    * Index, and an ordinary one must not decorate its only output at all.
    *
    * ntv omits the Index on index 0 because Tint turns one into
    * `@blend_src(0)`, and WGSL allows @blend_src only on a pair -- a lone
    * output carrying it is rejected. But omitting it on a shader that *does*
    * have a second output at index 1 is the mirror image of the same rule:
    *
    *   '@blend_src' and '@location' are used on one member while another
    *   member with '@location' doesn't use '@blend_src'
    *
    * which is what failed EEVEE's eevee_light_culling_debug -- two outputs at
    * location 0, one of them index 1, exactly the dual-source form GL asks
    * for. Detected from the variables rather than from
    * info.fs.color_is_dual_source, because this runs after passes that
    * rewrite the outputs and the flag is only as fresh as the last
    * nir_shader_gather_info. */
   /* A lone index-*1* output is the other half of the same rule, and it needs
    * more than a decoration: WGSL wants `@blend_src(0)` to exist, so if the
    * shader never declared an index-0 output there is nothing to decorate and
    * Tint says
    *
    *   '@blend_src(0)' is missing when '@blend_src' is used
    *
    * which failed the *link*. GL permits this outright --
    * `glBindFragDataLocationIndexed(program, 0, 1, "color")` with no index-0
    * output at all, which is KHR-GL45.program_interface_query's
    * output-location-index -- and leaves index 0's value **undefined** when
    * nothing writes it. So synthesizing one that writes zeros is exact by
    * specification rather than a guess: any value satisfies GL, and the pair
    * satisfies WGSL. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      gex_wgpu_add_missing_index_zero_output(nir);
   bool dual_source_output = false;
   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_out)
         dual_source_output |= var->data.index != 0;
   }
   const struct ntv_info info = {
      .is_native_vulkan = true,
      .omit_fragment_output_index_zero = !dual_source_output,
      /* All eight KHR-GL45.shaders.struct.local.{basic,nested}_{not_,}equal
       * cases failed on this and nothing else. Comparing two structs whose
       * members hold the same values leaves CSE with one SSA value on both
       * sides of the member comparison, ntv recognises `x != x` as a NaN test
       * and emits OpIsNan -- which WGSL has no builtin for. */
      .avoid_spv_isnan = true,
      .avoid_spv_khr_fma = true,
      /* WGSL's sample_mask builtin is a scalar, so the single-element array
       * Vulkan wants stored into gl_SampleMask cannot be assigned to it. */
      .sample_mask_scalar = true,
      .spirv_version = SPIRV_VERSION(1, 3),
      /* gex_wgpu has no bindless descriptor set.  Leaving this zero makes
       * nir_to_spirv treat every ordinary set-0 sampler as bindless and index
       * its two-entry bindless table with the sampler's driver location, which
       * writes out of bounds as soon as a shader declares a third sampler.
       * Use a set index no variable can carry.
       */
      .bindless_set_idx = UINT_MAX,
   };
   /* Timed, because "the shader path is slow" is not actionable and the three
    * stages have wildly different costs: NIR->SPIR-V is Mesa's, SPIR-V->WGSL
    * is Tint parsing and reprinting a whole program, and the module build is
    * the browser compiling that text again. */
   /* Whether this is one of the scene's vertex shaders, decided before the
    * NIR is freed. Nine or more inputs is what the geometry uses; the UI and
    * skybox shaders have one to four. */
   /* GEX_WGPU_DUMP_VS dumps every vertex and fragment module rather than the
    * one the heuristic guesses is the scene shader. Reading what @interpolate
    * each side declares is how the interpolation-sampling mismatch was
    * finally understood. */
   const bool dump_wgsl =
      (nir->info.stage == MESA_SHADER_VERTEX &&
       util_bitcount64(nir->info.inputs_read) >= 9) ||
      (getenv("GEX_WGPU_DUMP_VS") &&
       (nir->info.stage == MESA_SHADER_VERTEX ||
        nir->info.stage == MESA_SHADER_FRAGMENT));
   const double t_start = os_time_get_nano() / 1000000.0;
   /* A tex instruction with no texture deref aborts the emitter --
    *
    *   nir_to_spirv.c: emit_tex: Assertion `var' failed.
    *
    * -- which takes the application with it, and an abort is never an
    * acceptable answer to a shader this driver cannot translate. st/mesa
    * lowers the derefs away and gex_wgpu_split_combined_samplers rebuilds
    * them, so reaching here without one means that pass did not run for this
    * shader: it was the compute path once and the geometry path after that.
    * Refuse by name instead, and say which stage, so the next one is a
    * message rather than a core dump. */
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            if (nir_tex_instr_src_index(nir_instr_as_tex(instr),
                                        nir_tex_src_texture_deref) >= 0)
               continue;
            char *detail = NULL;
            asprintf(&detail,
                     "a texture instruction in the %s stage has no texture "
                     "deref, so the combined-sampler split did not run for it",
                     mesa_shader_stage_name(nir->info.stage));
            *error_out = detail;
            ralloc_free(nir);
            return NULL;
         }
      }
   }
   if (getenv("GEX_WGPU_IMAGE_TRACE")) {
      nir_foreach_variable_with_modes(var, nir, nir_var_image) {
         const struct glsl_type *t = glsl_without_array(var->type);
         fprintf(stderr, "gex-imgvar: stage=%u '%s' fmt=%s dim=%u array=%d "
                 "ms=%d access=%s%s\n", (unsigned)nir->info.stage,
                 var->name ? var->name : "(anon)",
                 util_format_short_name(var->data.image.format),
                 (unsigned)glsl_get_sampler_dim(t),
                 (int)glsl_sampler_type_is_array(t),
                 (int)(glsl_get_sampler_dim(t) == GLSL_SAMPLER_DIM_MS),
                 (var->data.access & ACCESS_NON_READABLE) ? "W" : "",
                 (var->data.access & ACCESS_NON_WRITEABLE) ? "R" : "");
      }
   }
   /* A last sweep for 64-bit integers, after everything else has run.
    *
    * WGSL has no i64, and Tint refuses a module that contains one outright --
    * "unhandled integer width: 64" -- so a single survivor costs the whole
    * shader. st already lowers them: nir_lower_doubles turns fp64 into the
    * software implementation and nir_lower_int64 expands the 64-bit integer
    * halves it produces. But st runs gl_nir_opts *after* both, and an
    * optimisation can put one back: an if/else assigning a double becomes a
    * `bcsel` on the packed 64-bit value, which no earlier pass is left to see.
    *
    * That is what KHR-GL45.shading_language_420pack's implicit_conversions
    * cases hit, and the disassembly showed it plainly -- a pack_64_2x32, an
    * OpSelect on the packed ulong, an unpack straight after:
    *
    *     %94 = OpBitcast %ulong %93
    *     %95 = OpSelect %ulong %39 %94 %ulong_0
    *    %103 = OpBitcast %v2uint %95
    *
    * Running the lowering once more here is cheap for every shader that has
    * no 64-bit anything -- the pass walks and finds nothing -- and is the only
    * place late enough to catch what the optimiser reintroduces. The
    * scalarisation is what nir_lower_doubles wants of vector 64-bit ops, and
    * st does the same before its own call. */
   if (nir->options->lower_int64_options && gex_wgpu_has_64_bit(nir)) {
      /* A 64-bit *local variable* is not an operation and no ALU lowering
       * touches it: it reaches the emitter as OpTypePointer Function %ulong
       * and Tint refuses the module for the type alone. It survives SSA
       * conversion when it is an array the shader indexes dynamically, which
       * is what the fp64 builtin tests do for a dvec3 and not for a dvec2.
       * Flattening the indexing and re-running vars-to-ssa dissolves it into
       * values the lowering below can see. */
      /* And a 64-bit *register* is not an operation either. The emitter gives
       * every register an OpVariable in the Function storage class, so a
       * 64-bit one reaches Tint as OpTypePointer Function %ulong -- the same
       * refusal a local variable earns, from a construct the ALU lowering
       * below cannot reach, because nir_lower_int64 rewrites SSA defs and a
       * register is not one. That is what survived here: seven
       * pack_64_2x32_split results, every one of them consumed by a
       * store_reg. Putting the shader back into SSA first turns those stores
       * into values, and then the pack/unpack pairs fold. */
      NIR_PASS(_, nir, nir_lower_reg_intrinsics_to_ssa);
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees,
               nir_var_function_temp, UINT32_MAX);
      NIR_PASS(_, nir, nir_lower_vars_to_ssa);
      NIR_PASS(_, nir, nir_lower_alu_to_scalar, gex_wgpu_is_64_bit_alu, NULL);
      NIR_PASS(_, nir, nir_lower_int64);
      NIR_PASS(_, nir, gex_wgpu_split_64bit_undef);
      /* The lowering *represents* its results with pack_64_2x32_split and
       * unpack_64_2x32_split, so it leaves 64-bit values behind by
       * construction and expects an optimiser to fold the pairs away. Without
       * this loop the type survives to the emitter and Tint refuses the
       * module for it -- the lowering having done its work and undone none of
       * the harm. */
      bool progress;
      do {
         progress = false;
         /* A phi is the third construct the ALU lowering cannot reach: it
          * merges two 64-bit values and stays 64-bit, and the emitter types
          * it accordingly. This splits each into a pair of 32-bit phis, whose
          * sources are unpacks of the packs above -- which is why it belongs
          * inside the loop rather than before it, so the fold that follows
          * can cancel the two against each other. */
         NIR_PASS(progress, nir, nir_lower_64bit_phis);
         NIR_PASS(progress, nir, nir_opt_copy_prop);
         NIR_PASS(progress, nir, nir_opt_algebraic);
         NIR_PASS(progress, nir, nir_opt_constant_folding);
         NIR_PASS(progress, nir, nir_opt_dce);
      } while (progress);
      /* And hand the emitter back the form it requires. nir_to_spirv has no
       * case for a phi -- it asserts outright -- so a shader reaches it out
       * of SSA, with phi webs held in registers. That is why the 64-bit
       * values were in registers to begin with, and taking them out of that
       * form above is only borrowed for the length of the lowering. The
       * registers rebuilt here are 32-bit, because the phis they come from
       * are. */
      NIR_PASS(_, nir, nir_convert_from_ssa, true, false);
      if (getenv("GEX_WGPU_INT64_LOG")) {
         nir_foreach_variable_in_shader(var, nir)
            if (glsl_type_contains_64bit(var->type))
               fprintf(stderr, "gex-int64: variable '%s' mode=0x%x type=%s\n",
                       var->name ? var->name : "(anon)",
                       (unsigned)var->data.mode, glsl_get_type_name(var->type));
         nir_foreach_function_impl(impl, nir) {
            nir_foreach_block(block, impl)
               nir_foreach_instr(instr, block) {
                  if (instr->type == nir_instr_type_alu) {
                     nir_alu_instr *x = nir_instr_as_alu(instr);
                     if (x->def.bit_size == 64)
                        fprintf(stderr, "gex-int64: alu %s def64\n",
                                nir_op_infos[x->op].name);
                     for (unsigned i = 0;
                          i < nir_op_infos[x->op].num_inputs; ++i)
                        if (x->src[i].src.ssa->bit_size == 64)
                           fprintf(stderr, "gex-int64: alu %s src%u 64\n",
                                   nir_op_infos[x->op].name, i);
                  } else if (instr->type == nir_instr_type_intrinsic) {
                     nir_intrinsic_instr *x = nir_instr_as_intrinsic(instr);
                     const char *nm =
                        nir_intrinsic_infos[x->intrinsic].name;
                     if (nir_intrinsic_infos[x->intrinsic].has_dest &&
                         x->def.bit_size == 64)
                        fprintf(stderr, "gex-int64: intr %s def64\n", nm);
                     for (unsigned i = 0;
                          i < nir_intrinsic_infos[x->intrinsic].num_srcs; ++i)
                        if (x->src[i].ssa->bit_size == 64)
                           fprintf(stderr, "gex-int64: intr %s src%u 64\n",
                                   nm, i);
                     if (x->intrinsic == nir_intrinsic_decl_reg &&
                         nir_intrinsic_bit_size(x) == 64)
                        fprintf(stderr, "gex-int64: decl_reg 64\n");
                  } else if (instr->type == nir_instr_type_load_const) {
                     if (nir_instr_as_load_const(instr)->def.bit_size == 64)
                        fprintf(stderr, "gex-int64: load_const 64\n");
                  } else if (instr->type == nir_instr_type_phi) {
                     if (nir_instr_as_phi(instr)->def.bit_size == 64)
                        fprintf(stderr, "gex-int64: phi 64\n");
                  } else if (instr->type == nir_instr_type_undef) {
                     if (nir_instr_as_undef(instr)->def.bit_size == 64)
                        fprintf(stderr, "gex-int64: undef 64\n");
                  }
               }
         }
         fprintf(stderr, "gex-int64: end of stage %u\n",
                 (unsigned)nir->info.stage);
      }
   }
   /* Word indices, not byte offsets; see the note above. Last, so anything
    * this file added earlier is scaled too. */
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, gex_wgpu_scale_shared_offset,
            nir_metadata_control_flow, NULL);
   struct spirv_shader *spirv = nir_to_spirv(nir, &info);
   /* Kept before the free: the dump below wants it, and reading it back off
    * the freed shader is what printed "stage=4294967264". Harmless in effect
    * -- it is a diagnostic line -- but it is a read of freed memory on a path
    * anyone debugging a shader turns on. */
   const unsigned dumped_stage = (unsigned)nir->info.stage;
   /* What this stage's outputs say about interpolation sampling, read off the
    * NIR while it still exists and applied to the WGSL below. Only a
    * pre-fragment stage has any: the fragment side's own centroid/sample
    * reaches SPIR-V and needs no help. */
   uint8_t out_sampling[GEX_WGPU_MAX_VARYING_LOCATIONS] = {0};
   if (nir->info.stage < MESA_SHADER_FRAGMENT) {
      nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
         if (!var->data.centroid && !var->data.sample)
            continue;
         if (var->data.driver_location >= GEX_WGPU_MAX_VARYING_LOCATIONS)
            continue;
         out_sampling[var->data.driver_location] =
            (uint8_t)(var->data.centroid ? 1u : 2u);
      }
   }
   /* The label's name is taken before the free, and that is the whole of
    * it: reading nir->info.name after ralloc_free is a use-after-free, and
    * it segfaulted on the third module of every run -- late enough to look
    * like a driver defect in whatever was being tested rather than in the
    * diagnostic added to find it. The buffer is a local because
    * WGPUStringView does not own its bytes and
    * wgpuDeviceCreateShaderModule copies during the call, so it has to
    * outlive that call and nothing more. */
   snprintf(out->label, sizeof(out->label), "%s:%s",
            _mesa_shader_stage_to_abbrev(stage),
            (nir->info.name && nir->info.name[0]) ? nir->info.name : "app");
   ralloc_free(nir);
   if (!spirv) {
      *error_out = strdup("Mesa NIR to SPIR-V failed");
      return NULL;
   }
   const double t_spirv = os_time_get_nano() / 1000000.0;

   char *wgsl = gex_wgpu_spirv_to_wgsl(spirv->words, spirv->num_words,
                                        error_out);
   const unsigned spirv_words = spirv->num_words;
   spirv_shader_delete(spirv);
   if (!wgsl)
      return NULL;
   wgsl = gex_wgpu_wgsl_add_output_sampling(wgsl, out_sampling,
                                            GEX_WGPU_MAX_VARYING_LOCATIONS);
   const double t_wgsl = os_time_get_nano() / 1000000.0;
   out->stage = dumped_stage;
   out->dump = dump_wgsl;
   out->spirv_words = spirv_words;
   out->t_start = t_start;
   out->t_spirv = t_spirv;
   out->t_wgsl = t_wgsl;
   return wgsl;
}

/* What NIR has to look like for zink's emitter and Tint, for every stage of
 * both drivers. */
const nir_shader_compiler_options *
gex_wgpu_nir_options(void)
{
   static const nir_shader_compiler_options options = {
      /* SPIR-V, and WGSL after it, have no native forms for these. */
      .lower_fdiv = true,
      .lower_flrp16 = true,
      .lower_flrp32 = true,
      .lower_flrp64 = true,
      .lower_fmod = true,
      .lower_bitfield_extract = true,
      .lower_bitfield_insert = true,
      .lower_uadd_carry = true,
      .lower_usub_borrow = true,
      .lower_fdph = true,
      /* The rest of what zink's emitter cannot spell. These are not
       * hypothetical: nir_to_spirv has no case for any of them, so an
       * unlowered one reaches emit_alu and aborts the process --
       * `seq/sne/slt/sge` did exactly that on the first native glxgears,
       * because Mesa's *fixed-function* vertex program is written in the
       * ARB_vertex_program instruction set and those four are its
       * comparisons. Nothing in the browser had reached them before, since
       * every application tested there brought its own GLSL.
       *
       * The list is zink's, because the emitter is zink's: an op it lowers
       * away is an op it cannot emit. The one deliberate difference is
       * lower_fsat, which nir_to_spirv does handle (as a clamp) and which is
       * better left for it than expanded into two more instructions here. */
      /* 64-bit anything, in software.
       *
       * WGSL has neither f64 nor i64, so every 64-bit operation has to become
       * a pair of 32-bit ones. Mesa already knows how: st runs
       * nir_lower_doubles against src/compiler/glsl/float64.glsl -- the same
       * software fp64 Intel ships -- and nir_lower_int64 for the integer
       * halves that lowering then produces. The driver's part is to ask for
       * it, which is these two masks and caps.doubles.
       *
       * ARB_gpu_shader_fp64 is one requirement of GL 4.0; the rung also wants
       * a tessellator, which nothing has written yet. */
      .lower_doubles_options = (nir_lower_doubles_options)(
         nir_lower_drcp | nir_lower_dsqrt | nir_lower_drsq | nir_lower_dtrunc |
         nir_lower_dfloor | nir_lower_dceil | nir_lower_dfract |
         nir_lower_dround_even | nir_lower_dmod | nir_lower_dsub |
         nir_lower_ddiv | nir_lower_dsign | nir_lower_dminmax |
         nir_lower_dsat | nir_lower_fp64_full_software),
      .lower_int64_options = (nir_lower_int64_options)(
         nir_lower_imul64 | nir_lower_isign64 | nir_lower_divmod64 |
         nir_lower_imul_high64 | nir_lower_bcsel64 | nir_lower_icmp64 |
         nir_lower_iadd64 | nir_lower_iabs64 | nir_lower_ineg64 |
         nir_lower_logic64 | nir_lower_minmax64 | nir_lower_shift64 |
         nir_lower_imul_2x32_64 | nir_lower_extract64 | nir_lower_ufind_msb64 |
         nir_lower_bit_count64 | nir_lower_find_lsb64 | nir_lower_conv64 |
         nir_lower_uadd_sat64 | nir_lower_usub_sat64 | nir_lower_iadd_sat64 |
         nir_lower_bitfield_reverse64 | nir_lower_bitfield_extract64 |
         /* The rest of the enum, because "any 64-bit operation" has to mean
          * all of them: one that survives reaches Tint as a 64-bit integer
          * and is refused there ("unhandled integer width: 64"), which is a
          * whole shader lost for one instruction. The subgroup and scan
          * entries cannot arise while this driver advertises no subgroup
          * extension, and are set for the same reason: the list should not
          * need revisiting when one is added. */
         nir_lower_iadd3_64 | nir_lower_scan_reduce_bitwise64 |
         nir_lower_scan_reduce_iadd64 | nir_lower_subgroup_shuffle64 |
         nir_lower_vote_ieq64),
      .lower_scmp = true,
      .lower_hadd = true,
      .lower_iadd_sat = true,
      .lower_uadd_sat = true,
      .lower_usub_sat = true,
      .lower_mul_high = true,
      .lower_fisnormal = true,
      .lower_fsat = false,
      .lower_extract_byte = true,
      .lower_extract_word = true,
      .lower_insert_byte = true,
      .lower_insert_word = true,
      .lower_pack_half_2x16 = true,
      .lower_pack_unorm_2x16 = true,
      .lower_pack_snorm_2x16 = true,
      .lower_pack_unorm_4x8 = true,
      .lower_pack_snorm_4x8 = true,
      .lower_unpack_half_2x16 = true,
      .lower_unpack_unorm_2x16 = true,
      .lower_unpack_snorm_2x16 = true,
      .lower_unpack_unorm_4x8 = true,
      .lower_unpack_snorm_4x8 = true,
      /* Uniforms reach the shader as a UBO, which is how this driver binds
       * constant buffers. */
      .lower_uniforms_to_ubo = true,
      /* Keep gl_ClipDistance as a float array. Without this the GLSL linker
       * packs it into a vec4[2] named gl_ClipDistanceMESA and zink's SPIR-V
       * emitter decorates that as BuiltIn ClipDistance -- which is a vec4
       * array where SPIR-V requires a scalar one, and Tint rejects the
       * module. clip_distance's two functional cases fail on exactly that. */
      .compact_arrays = true,
      /* Deliberately not vertex_id_zero_based or lower_base_vertex: the
       * SPIR-V emitter below implements load_vertex_id and load_base_vertex,
       * and asking NIR for the zero-based form instead produces
       * load_vertex_id_zero_base, which it does not. */
   };
   return &options;
}
