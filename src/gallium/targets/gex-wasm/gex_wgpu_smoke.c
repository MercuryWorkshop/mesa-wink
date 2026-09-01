/* SPDX-License-Identifier: MIT
 * Real-browser Gallium/WebGPU resource, clear, synchronization and readback.
 */
#include <emscripten.h>
#ifndef GEX_CONFORMANCE_SOFTPIPE
#include <webgpu/webgpu.h>

#include "gex_wgpu_public.h"
#else
#include "frontend/sw_winsys.h"
#include "softpipe/sp_public.h"
#include "sw/null/null_sw_winsys.h"
#endif
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/format/u_format.h"
#include "util/format/u_format_bptc.h"
#include "util/u_inlines.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GEX_CONFORMANCE_SOFTPIPE
extern WGPUDevice emscripten_webgpu_get_device(void);
int gex_wgpu_smoke(void);
#endif

static struct pipe_screen *test_screen;

/* ---------------------------------------------------------------------------
 * Bound-state tracking
 *
 * Gallium drivers keep raw pointers to bound CSO objects.  Deleting one while
 * it is still bound leaves the driver holding a dangling pointer; softpipe
 * asserts on this for shaders and (in this tree) for the remaining state
 * objects too.  st/mesa never does it because it unbinds first, so the corpus
 * has to behave the same way.
 *
 * Rather than hand-auditing every one of the corpus' delete sites, every bind
 * and delete goes through these helpers.  They mirror the driver's own view of
 * what is bound and unbind an object before deleting it.
 * ------------------------------------------------------------------------ */

struct gex_bound_state {
   struct pipe_context *ctx;
   void *fs;
   void *vs;
   void *gs;
   void *blend;
   void *dsa;
   void *rast;
   void *velems;
   void *samplers[MESA_SHADER_STAGES][PIPE_MAX_SAMPLERS];
   unsigned num_samplers[MESA_SHADER_STAGES];
};

static struct gex_bound_state gex_bound_states[8];

static struct gex_bound_state *
gex_bound(struct pipe_context *ctx)
{
   struct gex_bound_state *slot = NULL;

   for (unsigned i = 0; i < ARRAY_SIZE(gex_bound_states); i++) {
      if (gex_bound_states[i].ctx == ctx)
         return &gex_bound_states[i];
      if (!slot && gex_bound_states[i].ctx == NULL)
         slot = &gex_bound_states[i];
   }

   /* Running out of tracking slots would silently stop unbinding, so make it
    * loud instead.
    */
   assert(slot);
   if (!slot)
      return &gex_bound_states[0];

   memset(slot, 0, sizeof(*slot));
   slot->ctx = ctx;
   return slot;
}

static void
gex_forget_context(struct pipe_context *ctx)
{
   for (unsigned i = 0; i < ARRAY_SIZE(gex_bound_states); i++) {
      if (gex_bound_states[i].ctx == ctx)
         memset(&gex_bound_states[i], 0, sizeof(gex_bound_states[i]));
   }
}

static void
gex_destroy_context(struct pipe_context *ctx)
{
   gex_forget_context(ctx);
   ctx->destroy(ctx);
}

static void
gex_bind_fs(struct pipe_context *ctx, void *fs)
{
   gex_bound(ctx)->fs = fs;
   ctx->bind_fs_state(ctx, fs);
}

static void
gex_bind_vs(struct pipe_context *ctx, void *vs)
{
   gex_bound(ctx)->vs = vs;
   ctx->bind_vs_state(ctx, vs);
}

static void
gex_bind_blend(struct pipe_context *ctx, void *blend)
{
   gex_bound(ctx)->blend = blend;
   ctx->bind_blend_state(ctx, blend);
}

static void
gex_bind_dsa(struct pipe_context *ctx, void *dsa)
{
   gex_bound(ctx)->dsa = dsa;
   ctx->bind_depth_stencil_alpha_state(ctx, dsa);
}

static void
gex_bind_rast(struct pipe_context *ctx, void *rast)
{
   gex_bound(ctx)->rast = rast;
   ctx->bind_rasterizer_state(ctx, rast);
}

static void
gex_bind_velems(struct pipe_context *ctx, void *velems)
{
   gex_bound(ctx)->velems = velems;
   ctx->bind_vertex_elements_state(ctx, velems);
}

static void
gex_bind_samplers(struct pipe_context *ctx, mesa_shader_stage shader,
                  unsigned start, unsigned num, void **samplers)
{
   struct gex_bound_state *state = gex_bound(ctx);

   assert(shader < MESA_SHADER_STAGES);
   assert(start + num <= PIPE_MAX_SAMPLERS);

   for (unsigned i = 0; i < num; i++)
      state->samplers[shader][start + i] = samplers ? samplers[i] : NULL;

   /* Same "highest non-null entry" rule the driver uses. */
   unsigned j = MAX2(state->num_samplers[shader], start + num);
   while (j > 0 && state->samplers[shader][j - 1] == NULL)
      j--;
   state->num_samplers[shader] = j;

   ctx->bind_sampler_states(ctx, shader, start, num, samplers);
}

static void
gex_delete_fs(struct pipe_context *ctx, void *fs)
{
   struct gex_bound_state *state = gex_bound(ctx);

   if (fs && state->fs == fs)
      gex_bind_fs(ctx, NULL);
   ctx->delete_fs_state(ctx, fs);
}

static void
gex_delete_vs(struct pipe_context *ctx, void *vs)
{
   struct gex_bound_state *state = gex_bound(ctx);

   if (vs && state->vs == vs)
      gex_bind_vs(ctx, NULL);
   ctx->delete_vs_state(ctx, vs);
}

static void
gex_delete_blend(struct pipe_context *ctx, void *blend)
{
   struct gex_bound_state *state = gex_bound(ctx);

   if (blend && state->blend == blend)
      gex_bind_blend(ctx, NULL);
   ctx->delete_blend_state(ctx, blend);
}

static void
gex_delete_dsa(struct pipe_context *ctx, void *dsa)
{
   struct gex_bound_state *state = gex_bound(ctx);

   if (dsa && state->dsa == dsa)
      gex_bind_dsa(ctx, NULL);
   ctx->delete_depth_stencil_alpha_state(ctx, dsa);
}

static void
gex_delete_rast(struct pipe_context *ctx, void *rast)
{
   struct gex_bound_state *state = gex_bound(ctx);

   if (rast && state->rast == rast)
      gex_bind_rast(ctx, NULL);
   ctx->delete_rasterizer_state(ctx, rast);
}

static void
gex_delete_velems(struct pipe_context *ctx, void *velems)
{
   struct gex_bound_state *state = gex_bound(ctx);

   if (velems && state->velems == velems)
      gex_bind_velems(ctx, NULL);
   ctx->delete_vertex_elements_state(ctx, velems);
}

static void
gex_delete_sampler(struct pipe_context *ctx, void *sampler)
{
   struct gex_bound_state *state = gex_bound(ctx);

   if (!sampler) {
      ctx->delete_sampler_state(ctx, sampler);
      return;
   }

   for (unsigned shader = 0; shader < MESA_SHADER_STAGES; shader++) {
      unsigned num = state->num_samplers[shader];
      bool bound = false;

      for (unsigned i = 0; i < num; i++)
         bound |= state->samplers[shader][i] == sampler;
      if (!bound)
         continue;

      /* draw's polygon-stipple wrapper asserts that sampler updates start at
       * slot zero, so rebind the whole range with this object removed.
       */
      void *replacement[PIPE_MAX_SAMPLERS];
      for (unsigned i = 0; i < num; i++) {
         replacement[i] = state->samplers[shader][i] == sampler
            ? NULL : state->samplers[shader][i];
      }
      gex_bind_samplers(ctx, (mesa_shader_stage)shader, 0, num, replacement);
   }

   ctx->delete_sampler_state(ctx, sampler);
}

static const uint8_t *
readback_pixel(const uint8_t *pixels, unsigned stride,
               unsigned x, unsigned y)
{
   return pixels + y * stride + x * 4;
}

static nir_shader *
finish_test_shader(nir_builder *b)
{
   /* NIR compiler options are part of the Gallium screen contract.  Leaving
    * this NULL happened to read zero-filled low Wasm memory in release builds,
    * but is an actual null dereference in NIR-to-TGSI and can change lowering
    * decisions.  Every shader in the shared corpus uses the backend's options.
    */
   b->shader->options = test_screen->nir_options[b->shader->info.stage];
   nir_shader_gather_info(b->shader, b->impl);
   nir_assign_io_var_locations(b->shader, nir_var_shader_in);
   nir_assign_io_var_locations(b->shader, nir_var_shader_out);
#ifndef NDEBUG
   fprintf(stderr, "GEX corpus shader: %s\n", b->shader->info.name);
   nir_validate_shader(b->shader, "after constructing GEX corpus shader");
#endif
   return b->shader;
}

static nir_shader *
make_vs(float clip_z, bool fullscreen)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "gex-test-vs");
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   color->data.location = VARYING_SLOT_VAR0;
   nir_def *vertex_id = nir_load_vertex_id(&b);
   nir_def *p0 = nir_imm_vec4(&b, -1.0f, -1.0f, clip_z, 1.0f);
   nir_def *p1 = nir_imm_vec4(&b, fullscreen ? 3.0f : 0.0f,
                              -1.0f, clip_z, 1.0f);
   nir_def *p2 = nir_imm_vec4(&b, -1.0f,
                              fullscreen ? 3.0f : 1.0f, clip_z, 1.0f);
   nir_def *pos = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), p0,
                            nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                      p1, p2));
   nir_store_var(&b, position, pos, 0xf);
   nir_store_var(&b, color, nir_imm_vec4(&b, 0.0f, 1.0f, 0.0f, 1.0f), 0xf);
   return finish_test_shader(&b);
}

/* A vertex shader whose colour is decided by gl_InstanceID.
 *
 * This is what backs the vs_instanceid capability.  It matters that the value
 * is the GL one: SPIR-V's InstanceIndex, which the NIR-to-SPIR-V translation
 * emits, counts from the draw's base instance, while GL's gl_InstanceID always
 * counts from zero.  The two agree only while base instance is zero, which is
 * why the driver leaves start_instance off while advertising this.
 */
static nir_shader *
make_instance_id_vs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "gex-instance-id-vs");
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   color->data.location = VARYING_SLOT_VAR0;

   nir_def *vertex_id = nir_load_vertex_id(&b);
   /* The same half-surface triangle the other draw cases use, so the
    * shared coverage check can tell drawn pixels from the clear colour. */
   nir_def *p0 = nir_imm_vec4(&b, -1.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p1 = nir_imm_vec4(&b, 0.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p2 = nir_imm_vec4(&b, -1.0f, 1.0f, 0.0f, 1.0f);
   nir_def *pos = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), p0,
                            nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                      p1, p2));
   nir_store_var(&b, position, pos, 0xf);

   /* Both instances cover the surface, so the last one drawn decides the
    * colour: green proves instance one ran and saw its own index, red proves
    * the index never advanced past zero. */
   nir_def *instance = nir_load_instance_id(&b);
   nir_def *second = nir_ieq_imm(&b, instance, 1);
   nir_store_var(&b, color,
                 nir_bcsel(&b, second,
                           nir_imm_vec4(&b, 0.0f, 1.0f, 0.0f, 1.0f),
                           nir_imm_vec4(&b, 1.0f, 0.0f, 0.0f, 1.0f)),
                 0xf);
   return finish_test_shader(&b);
}

/* A compute shader that fills a storage buffer with a known function of the
 * workgroup index.
 *
 * On WebGPU, compute is the only stage permitted to write a storage buffer,
 * which makes this the foundation for transform feedback: capturing vertex
 * outputs there means re-running the vertex shader as compute. This case
 * proves the NIR -> SPIR-V -> WGSL path for a compute shader that stores.
 */
static nir_shader *
make_compute_fill_cs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, NULL,
                                                   "gex-compute-fill-cs");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   const struct glsl_type *words = glsl_array_type(glsl_uint_type(), 0, 4);
   struct glsl_struct_field field = {0};
   field.type = words;
   field.name = "data";
   const struct glsl_type *block =
      glsl_interface_type(&field, 1, GLSL_INTERFACE_PACKING_STD430, false,
                          "gex_compute_block");
   nir_variable *buffer =
      nir_variable_create(b.shader, nir_var_mem_ssbo, block, "gex_out");
   buffer->interface_type = block;
   buffer->data.binding = 16;
   buffer->data.driver_location = 0;
   buffer->data.explicit_binding = true;

   /* Drivers size their storage-buffer state from this; leaving it zero makes
    * the binding invisible and the stores land nowhere. */
   b.shader->info.num_ssbos = 1;

   nir_def *index = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_deref_instr *array =
      nir_build_deref_struct(&b, nir_build_deref_var(&b, buffer), 0);
   nir_deref_instr *slot = nir_build_deref_array(&b, array, index);
   nir_store_deref(&b, slot,
                   nir_iadd_imm(&b, nir_imul_imm(&b, index, 7), 3), 0x1);
   return finish_test_shader(&b);
}

static nir_shader *finish_constant_test_shader(nir_builder *b);
static nir_def *make_constant_load(nir_builder *b, unsigned binding);

/* A vertex shader whose captured varying is a known function of the vertex
 * *and* of a uniform, so what lands in the transform feedback buffer can be
 * checked exactly.
 *
 * The uniform is not decoration. Where a backend runs the capture in a
 * separate pass -- gex_wgpu re-runs this shader as compute, because WGSL will
 * not let the vertex stage write a storage buffer -- that pass has to bind
 * the shader's constant buffers too, and almost every real capturing shader
 * reads one. A capture shader that only ever used vertex inputs would not
 * exercise it. */
static nir_shader *
make_xfb_vs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "gex-xfb-vs");
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *captured = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "captured");
   captured->data.location = VARYING_SLOT_VAR0;

   nir_def *vertex_id = nir_load_vertex_id(&b);
   nir_def *p0 = nir_imm_vec4(&b, -1.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p1 = nir_imm_vec4(&b, 0.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p2 = nir_imm_vec4(&b, -1.0f, 1.0f, 0.0f, 1.0f);
   nir_store_var(&b, position,
                 nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), p0,
                           nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1), p1, p2)),
                 0xf);

   /* (vertex_id, 100 + vertex_id, 200, 300) as floats, plus the uniform: the
    * first two vary per vertex, so a capture that wrote the same vertex
    * repeatedly, or wrote them in the wrong order, would not match, and the
    * uniform is (1, 0, 0, 1) so two components move and two do not. */
   nir_def *id = nir_u2f32(&b, vertex_id);
   nir_def *base = nir_vec4(&b, id, nir_fadd_imm(&b, id, 100.0f),
                            nir_imm_float(&b, 200.0f),
                            nir_imm_float(&b, 300.0f));
   nir_store_var(&b, captured,
                 nir_fadd(&b, base, make_constant_load(&b, 0)), 0xf);
   return finish_constant_test_shader(&b);
}

/* A geometry shader that takes each triangle and emits it again, with the
 * colour replaced.
 *
 * Passing the position through unchanged means the triangle has to land where
 * the vertex shader put it, which only happens if the geometry stage read the
 * vertex shader's output correctly. Replacing the colour means the result is
 * distinguishable from the vertex shader's own output, so a backend that
 * quietly skipped the geometry stage fails rather than passes. */
static nir_shader *
make_passthrough_gs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_GEOMETRY, NULL,
                                                  "gex-passthrough-gs");
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 3;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;

   nir_variable *in_position = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_array_type(glsl_vec4_type(), 3, 0),
      "gl_PositionIn");
   in_position->data.location = VARYING_SLOT_POS;
   nir_variable *out_position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   out_position->data.location = VARYING_SLOT_POS;
   nir_variable *out_color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   out_color->data.location = VARYING_SLOT_VAR0;

   for (unsigned i = 0; i < 3; ++i) {
      nir_deref_instr *at = nir_build_deref_array(
         &b, nir_build_deref_var(&b, in_position), nir_imm_int(&b, i));
      nir_store_var(&b, out_position, nir_load_deref(&b, at), 0xf);
      nir_store_var(&b, out_color,
                    nir_imm_vec4(&b, 1.0f, 0.0f, 1.0f, 1.0f), 0xf);
      nir_emit_vertex(&b);
   }
   nir_end_primitive(&b);
   return finish_test_shader(&b);
}

/* A vertex shader that puts one corner of the viewport at each of the first
 * four vertex indices, in the order a given topology needs to cover it: a
 * strip wants them zig-zag, a fan wants them in a ring. Anything past the
 * fourth index is pulled inside, for the point and line cases. */
static nir_shader *
make_topology_vs(bool fan_order, float scale)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, NULL, "gex-topology-vs");
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   color->data.location = VARYING_SLOT_VAR0;

   static const float strip[4][2] = {
      {-1.0f, -1.0f}, {1.0f, -1.0f}, {-1.0f, 1.0f}, {1.0f, 1.0f},
   };
   static const float fan[4][2] = {
      {-1.0f, -1.0f}, {1.0f, -1.0f}, {1.0f, 1.0f}, {-1.0f, 1.0f},
   };
   const float (*corner)[2] = fan_order ? fan : strip;

   nir_def *id = nir_load_vertex_id(&b);
   nir_def *chosen = nir_imm_vec4(&b, corner[3][0] * scale,
                                  corner[3][1] * scale, 0.0f, 1.0f);
   for (int i = 2; i >= 0; --i)
      chosen = nir_bcsel(&b, nir_ieq_imm(&b, id, i),
                         nir_imm_vec4(&b, corner[i][0] * scale,
                                      corner[i][1] * scale, 0.0f, 1.0f),
                         chosen);
   nir_store_var(&b, position, chosen, 0xf);
   nir_store_var(&b, color, nir_imm_vec4(&b, 1.0f, 0.0f, 0.0f, 1.0f), 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_varying_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
                                                   "gex-varying-fs");
   nir_variable *input = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "vertex_color");
   input->data.location = VARYING_SLOT_VAR0;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, nir_load_var(&b, input), 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_color_fs(float r, float g, float blue, float a)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
                                                   "gex-color-fs");
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, nir_imm_vec4(&b, r, g, blue, a), 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_mrt_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
                                                   "gex-mrt-fs");
   const float values[2][4] = {
      {1.0f, 0.0f, 0.0f, 1.0f},
      {0.0f, 1.0f, 1.0f, 1.0f},
   };
   for (unsigned target = 0; target < 2; ++target) {
      char name[16];
      snprintf(name, sizeof(name), "out_color%u", target);
      nir_variable *color = nir_variable_create(
         b.shader, nir_var_shader_out, glsl_vec4_type(), name);
      color->data.location = FRAG_RESULT_DATA0 + target;
      color->data.index = 0;
      nir_store_var(&b, color,
                    nir_imm_vec4(&b, values[target][0], values[target][1],
                                 values[target][2], values[target][3]),
                    0xf);
   }
   return finish_test_shader(&b);
}

static nir_shader *
finish_constant_test_shader(nir_builder *b)
{
   nir_shader *shader = finish_test_shader(b);
   nir_lower_uniforms_to_ubo(shader, false, false);
   return shader;
}

static nir_def *
make_constant_load(nir_builder *b, unsigned binding)
{
   (void)binding;
   b->shader->num_uniforms = 1;
   return nir_load_uniform(b, 4, 32, nir_imm_int(b, 0), .range = 16,
                           .dest_type = nir_type_float32);
}

static nir_shader *
make_constant_buffer_fs(unsigned binding)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
                                                   "gex-constant-buffer-fs");
   nir_def *value = make_constant_load(&b, binding);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, value, 0xf);
   return finish_constant_test_shader(&b);
}

static nir_shader *
make_constant_buffer_second_vec4_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-constant-buffer-second-vec4-fs");
   b.shader->num_uniforms = 2;
   nir_def *value = nir_load_uniform(
      &b, 4, 32, nir_imm_int(&b, 1), .range = 16,
      .dest_type = nir_type_float32);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, value, 0xf);
   return finish_constant_test_shader(&b);
}

static nir_shader *
make_dynamic_constant_buffer_vs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, NULL, "gex-dynamic-constant-buffer-vs");
   b.shader->num_uniforms = 3;
   nir_def *position_value = nir_load_uniform(
      &b, 4, 32, nir_load_vertex_id(&b), .range = 48,
      .dest_type = nir_type_float32);
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   color->data.location = VARYING_SLOT_VAR0;
   nir_store_var(&b, position, position_value, 0xf);
   nir_store_var(&b, color, nir_imm_vec4(&b, 0.0f, 1.0f, 0.0f, 1.0f), 0xf);
   return finish_constant_test_shader(&b);
}

/* A fragment shader whose only constant buffer sits at `slot`.
 *
 * A high slot is not just a bigger number. WebGPU guarantees only twelve
 * uniform buffers per stage, so gex_wgpu backs the slots past its device's
 * limit with storage buffers instead -- a different address space, a different
 * binding, and a different bind group layout entry. This is what walks that
 * path; every other constant buffer case here uses slot zero or one. */
static nir_shader *
make_slot_constant_buffer_fs(unsigned slot)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-slot-constant-buffer-fs");
   const struct glsl_type *values = glsl_array_type(glsl_vec4_type(), 1, 16);
   char name[24];
   snprintf(name, sizeof(name), "constant_buffer_%u", slot);
   struct glsl_struct_field field = {
      .type = values,
      .name = "data",
      .location = -1,
   };
   nir_variable *block =
      nir_variable_create(b.shader, nir_var_mem_ubo, values, name);
   block->interface_type = glsl_interface_type(
      &field, 1, GLSL_INTERFACE_PACKING_STD430, false, name);
   block->data.binding = slot;
   block->data.driver_location = slot;
   block->data.explicit_binding = 1;

   nir_def *value = nir_load_ubo(
      &b, 4, 32, nir_imm_int(&b, slot), nir_imm_int(&b, 0),
      .align_mul = 16, .range = 16);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, value, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_two_constant_buffer_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-two-constant-buffer-fs");
   const struct glsl_type *values =
      glsl_array_type(glsl_vec4_type(), 1, 16);
   struct glsl_struct_field field = {
      .type = values,
      .name = "data",
      .location = -1,
   };
   for (unsigned slot = 0; slot < 2; ++slot) {
      char name[24];
      snprintf(name, sizeof(name), "constant_buffer_%u", slot);
      nir_variable *block = nir_variable_create(
         b.shader, nir_var_mem_ubo, values, name);
      block->interface_type = glsl_interface_type(
         &field, 1, GLSL_INTERFACE_PACKING_STD430, false, name);
      block->data.binding = slot;
      block->data.driver_location = slot;
      block->data.explicit_binding = 1;
   }
   nir_def *first = nir_load_ubo(
      &b, 4, 32, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
      .align_mul = 16, .range = 16);
   nir_def *second = nir_load_ubo(
      &b, 4, 32, nir_imm_int(&b, 1), nir_imm_int(&b, 0),
      .align_mul = 16, .range = 16);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, nir_fadd(&b, first, second), 0xf);
   b.shader->info.num_ubos = 2;
   b.shader->info.first_ubo_is_default_ubo = true;
   return finish_test_shader(&b);
}

static nir_shader *
make_constant_buffer_vs(unsigned binding)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "gex-constant-buffer-vs");
   nir_def *value = make_constant_load(&b, binding);
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   color->data.location = VARYING_SLOT_VAR0;
   nir_def *vertex_id = nir_load_vertex_id(&b);
   nir_def *p0 = nir_imm_vec4(&b, -1.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p1 = nir_imm_vec4(&b, 0.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p2 = nir_imm_vec4(&b, -1.0f, 1.0f, 0.0f, 1.0f);
   nir_def *pos = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), p0,
                            nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                      p1, p2));
   nir_store_var(&b, position, pos, 0xf);
   nir_store_var(&b, color, value, 0xf);
   return finish_constant_test_shader(&b);
}

static nir_shader *
make_vertex_fetch_vs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "gex-vertex-fetch-vs");
   nir_variable *input = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "position");
   input->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   color->data.location = VARYING_SLOT_VAR0;
   nir_store_var(&b, position, nir_load_var(&b, input), 0xf);
   nir_store_var(&b, color, nir_imm_vec4(&b, 0.0f, 1.0f, 0.0f, 1.0f), 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_interleaved_vertex_color_vs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "gex-interleaved-vs");
   nir_variable *position_in = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "position");
   position_in->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *color_in = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "color");
   color_in->data.location = VERT_ATTRIB_GENERIC1;
   nir_variable *position_out = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position_out->data.location = VARYING_SLOT_POS;
   nir_variable *color_out = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   color_out->data.location = VARYING_SLOT_VAR0;
   nir_store_var(&b, position_out, nir_load_var(&b, position_in), 0xf);
   nir_store_var(&b, color_out, nir_load_var(&b, color_in), 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_compact_position_vs(unsigned components, float r, float g, float blue)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "gex-compact-position-vs");
   nir_variable *position_in = nir_variable_create(
      b.shader, nir_var_shader_in,
      glsl_vector_type(GLSL_TYPE_FLOAT, components), "position");
   position_in->data.location = VERT_ATTRIB_GENERIC0;
   nir_variable *position_out = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position_out->data.location = VARYING_SLOT_POS;
   nir_variable *color_out = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "vertex_color");
   color_out->data.location = VARYING_SLOT_VAR0;
   nir_def *input = nir_load_var(&b, position_in);
   nir_def *z = components == 3 ? nir_channel(&b, input, 2)
                                : nir_imm_float(&b, 0.0f);
   nir_store_var(&b, position_out,
                 nir_vec4(&b, nir_channel(&b, input, 0),
                          nir_channel(&b, input, 1), z,
                          nir_imm_float(&b, 1.0f)), 0xf);
   nir_store_var(&b, color_out, nir_imm_vec4(&b, r, g, blue, 1.0f), 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_texture_coord_vs(float u, float v)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, NULL,
                                                   "gex-texture-coord-vs");
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *coord = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec2_type(), "texture_coord");
   coord->data.location = VARYING_SLOT_VAR0;
   nir_def *vertex_id = nir_load_vertex_id(&b);
   nir_def *p0 = nir_imm_vec4(&b, -1.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p1 = nir_imm_vec4(&b,  0.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p2 = nir_imm_vec4(&b, -1.0f,  1.0f, 0.0f, 1.0f);
   nir_def *pos = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), p0,
                            nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                      p1, p2));
   nir_store_var(&b, position, pos, 0xf);
   nir_store_var(&b, coord, nir_imm_vec2(&b, u, v), 0x3);
   return finish_test_shader(&b);
}

static nir_shader *
make_gradient_texture_vs(float extent)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, NULL, "gex-gradient-texture-vs");
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *coord = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec2_type(), "texture_coord");
   coord->data.location = VARYING_SLOT_VAR0;
   nir_def *vertex_id = nir_load_vertex_id(&b);
   nir_def *p0 = nir_imm_vec4(&b, -1.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p1 = nir_imm_vec4(&b, 3.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p2 = nir_imm_vec4(&b, -1.0f, 3.0f, 0.0f, 1.0f);
   nir_def *t0 = nir_imm_vec2(&b, 0.0f, 0.0f);
   nir_def *t1 = nir_imm_vec2(&b, extent, 0.0f);
   nir_def *t2 = nir_imm_vec2(&b, 0.0f, extent);
   nir_def *pos = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), p0,
                            nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                      p1, p2));
   nir_def *uv = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), t0,
                           nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                     t1, t2));
   nir_store_var(&b, position, pos, 0xf);
   nir_store_var(&b, coord, uv, 0x3);
   return finish_test_shader(&b);
}

static nir_shader *
make_gradient_texture_3d_vs(float r_extent)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, NULL, "gex-gradient-texture-3d-vs");
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *coord = nir_variable_create(
      b.shader, nir_var_shader_out,
      glsl_vector_type(GLSL_TYPE_FLOAT, 3), "texture_coord");
   coord->data.location = VARYING_SLOT_VAR0;
   nir_def *vertex_id = nir_load_vertex_id(&b);
   nir_def *p0 = nir_imm_vec4(&b, -1.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p1 = nir_imm_vec4(&b, 3.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p2 = nir_imm_vec4(&b, -1.0f, 3.0f, 0.0f, 1.0f);
   nir_def *t0 = nir_imm_vec3(&b, 0.5f, 0.5f, 0.0f);
   nir_def *t1 = nir_imm_vec3(&b, 0.5f, 0.5f, r_extent);
   nir_def *t2 = nir_imm_vec3(&b, 0.5f, 0.5f, 0.0f);
   nir_def *pos = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), p0,
                            nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                      p1, p2));
   nir_def *uvw = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), t0,
                            nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                      t1, t2));
   nir_store_var(&b, position, pos, 0xf);
   nir_store_var(&b, coord, uvw, 0x7);
   return finish_test_shader(&b);
}

static nir_shader *
make_texture_fs(bool varying_coord, float u, float v, float explicit_lod)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
                                                   "gex-texture-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
      "texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_deref_instr *texture_deref = nir_build_deref_var(&b, texture);
   nir_deref_instr *sampler_deref = nir_build_deref_var(&b, sampler);
   nir_def *coord;
   if (varying_coord) {
      nir_variable *input = nir_variable_create(
         b.shader, nir_var_shader_in, glsl_vec2_type(), "texture_coord");
      input->data.location = VARYING_SLOT_VAR0;
      coord = nir_load_var(&b, input);
   } else {
      coord = nir_imm_vec2(&b, u, v);
   }
   const bool use_lod = explicit_lod >= 0.0f;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, use_lod ? 4 : 3);
   tex->op = use_lod ? nir_texop_txl : nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_deref,
                                      &texture_deref->def);
   tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_deref,
                                      &sampler_deref->def);
   if (use_lod)
      tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_lod,
                                        nir_imm_float(&b, explicit_lod));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, &tex->def, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_texture_transform_fs(float scale, float bias)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-texture-transform-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
      "texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec2(&b, 0.5f, 0.5f));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *value = nir_fadd_imm(
      &b, nir_fmul_imm(&b, &tex->def, scale), bias);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, value, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_two_texture_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
                                                   "gex-two-texture-fs");
   nir_def *samples[2];
   for (unsigned slot = 0; slot < 2; ++slot) {
      char texture_name[16], sampler_name[16];
      snprintf(texture_name, sizeof(texture_name), "texture%u", slot);
      snprintf(sampler_name, sizeof(sampler_name), "sampler%u", slot);
      nir_variable *texture = nir_variable_create(
         b.shader, nir_var_uniform,
         glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
         texture_name);
      texture->data.descriptor_set = 0;
      texture->data.binding = slot;
      texture->data.driver_location = slot;
      nir_variable *sampler = nir_variable_create(
         b.shader, nir_var_uniform, glsl_bare_sampler_type(), sampler_name);
      sampler->data.descriptor_set = 0;
      sampler->data.binding = slot + 2;
      sampler->data.driver_location = slot;
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
      tex->op = nir_texop_tex;
      tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
      tex->coord_components = 2;
      tex->dest_type = nir_type_float32;
      tex->texture_index = slot;
      tex->sampler_index = slot;
      tex->src[0] = nir_tex_src_for_ssa(
         nir_tex_src_coord, nir_imm_vec2(&b, 0.5f, 0.5f));
      tex->src[1] = nir_tex_src_for_ssa(
         nir_tex_src_texture_deref,
         &nir_build_deref_var(&b, texture)->def);
      tex->src[2] = nir_tex_src_for_ssa(
         nir_tex_src_sampler_deref,
         &nir_build_deref_var(&b, sampler)->def);
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      samples[slot] = &tex->def;
   }
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, nir_fadd(&b, samples[0], samples[1]), 0xf);
   return finish_test_shader(&b);
}

/* Fetch one sample of a multisampled texture.
 *
 * There is no sampler: WGSL reaches a texture_multisampled_2d with
 * textureLoad and a sample index, which is exactly what texelFetch on a
 * sampler2DMS means, so this is the ordinary path rather than an emulation. */
static nir_shader *
make_multisample_texel_fetch_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-ms-texel-fetch-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_MS, false, GLSL_TYPE_FLOAT),
      "ms_texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf_ms;
   tex->sampler_dim = GLSL_SAMPLER_DIM_MS;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_ivec2(&b, 0, 0));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_ms_index, nir_imm_int(&b, 0));
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, &tex->def, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_integer_texel_fetch_fs(bool is_signed)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, is_signed
         ? "gex-sint-texel-fetch-fs" : "gex-uint-texel-fetch-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, false,
                        is_signed ? GLSL_TYPE_INT : GLSL_TYPE_UINT),
      is_signed ? "sint_texture0" : "uint_texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = is_signed ? nir_type_int32 : nir_type_uint32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_ivec2(&b, 0, 0));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_lod, nir_imm_int(&b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *normalized = is_signed
      ? nir_fmul_imm(&b, nir_fadd_imm(&b, nir_i2f32(&b, &tex->def), 128.0f),
                     1.0f / 255.0f)
      : nir_fmul_imm(&b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f);
   if (!is_signed)
      normalized = nir_vec4(&b, nir_channel(&b, normalized, 0),
                            nir_channel(&b, normalized, 1),
                            nir_channel(&b, normalized, 2),
                            nir_imm_float(&b, 1.0f));
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, normalized, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_integer_texture_lod_fs(bool is_signed, bool implicit_lod,
                            bool varying_coord,
                            float u, float v, float lod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, is_signed
         ? "gex-sint-texture-lod-fs" : "gex-uint-texture-lod-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, false,
                        is_signed ? GLSL_TYPE_INT : GLSL_TYPE_UINT),
      is_signed ? "sint_texture0" : "uint_texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_def *coord;
   if (varying_coord) {
      nir_variable *input = nir_variable_create(
         b.shader, nir_var_shader_in, glsl_vec2_type(), "texture_coord");
      input->data.location = VARYING_SLOT_VAR0;
      coord = nir_load_var(&b, input);
   } else {
      coord = nir_imm_vec2(&b, u, v);
   }
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, implicit_lod ? 3 : 4);
   tex->op = implicit_lod ? nir_texop_tex : nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = is_signed ? nir_type_int32 : nir_type_uint32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   if (!implicit_lod)
      tex->src[3] = nir_tex_src_for_ssa(
         nir_tex_src_lod, nir_imm_float(&b, lod));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *normalized = is_signed
      ? nir_fmul_imm(&b, nir_fadd_imm(&b, nir_i2f32(&b, &tex->def), 128.0f),
                     1.0f / 255.0f)
      : nir_fmul_imm(&b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, normalized, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_uint_array_texture_lod_fs(float layer)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-uint-array-texture-lod-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, true, GLSL_TYPE_UINT),
      "uint_array_texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   tex->coord_components = 3;
   tex->dest_type = nir_type_uint32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec3(&b, 0.25f, 0.25f, layer));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *normalized = nir_fmul_imm(
      &b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, normalized, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_1d_texture_fs(bool integer, bool implicit, float coord, float lod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL,
      integer ? "gex-uint-1d-texture-fs" : "gex-float-1d-texture-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_1D, false,
                        integer ? GLSL_TYPE_UINT : GLSL_TYPE_FLOAT),
      "texture_1d0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_def *sample_coord;
   if (implicit) {
      nir_variable *input = nir_variable_create(
         b.shader, nir_var_shader_in, glsl_float_type(), "texture_coord");
      input->data.location = VARYING_SLOT_VAR0;
      sample_coord = nir_load_var(&b, input);
   } else {
      sample_coord = nir_imm_float(&b, coord);
   }
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, implicit ? 3 : 4);
   tex->op = implicit ? nir_texop_tex : nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_1D;
   tex->coord_components = 1;
   tex->dest_type = integer ? nir_type_uint32 : nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, sample_coord);
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   if (!implicit)
      tex->src[3] = nir_tex_src_for_ssa(
         nir_tex_src_lod, nir_imm_float(&b, lod));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *result = integer
      ? nir_fmul_imm(&b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f)
      : &tex->def;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, result, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_1d_array_texture_fs(bool integer, float coord, float layer)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, integer
         ? "gex-uint-1d-array-texture-fs"
         : "gex-float-1d-array-texture-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_1D, true,
                        integer ? GLSL_TYPE_UINT : GLSL_TYPE_FLOAT),
      "texture_1d_array0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_1D;
   tex->is_array = true;
   tex->coord_components = 2;
   tex->dest_type = integer ? nir_type_uint32 : nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec2(&b, coord, layer));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *result = integer
      ? nir_fmul_imm(&b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f)
      : &tex->def;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, result, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_rect_texture_fs(bool integer, float x, float y)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, integer
         ? "gex-uint-rect-texture-fs" : "gex-float-rect-texture-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_RECT, false,
                        integer ? GLSL_TYPE_UINT : GLSL_TYPE_FLOAT),
      "texture_rect0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_RECT;
   tex->coord_components = 2;
   tex->dest_type = integer ? nir_type_uint32 : nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec2(&b, x, y));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *result = integer
      ? nir_fmul_imm(&b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f)
      : &tex->def;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, result, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_gradient_texture_1d_vs(float extent)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, NULL, "gex-gradient-texture-1d-vs");
   nir_variable *position = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "gl_Position");
   position->data.location = VARYING_SLOT_POS;
   nir_variable *coord = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_float_type(), "texture_coord");
   coord->data.location = VARYING_SLOT_VAR0;
   nir_def *vertex_id = nir_load_vertex_id(&b);
   nir_def *p0 = nir_imm_vec4(&b, -1.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p1 = nir_imm_vec4(&b, 3.0f, -1.0f, 0.0f, 1.0f);
   nir_def *p2 = nir_imm_vec4(&b, -1.0f, 3.0f, 0.0f, 1.0f);
   nir_def *pos = nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 0), p0,
                            nir_bcsel(&b, nir_ieq_imm(&b, vertex_id, 1),
                                      p1, p2));
   nir_def *s = nir_bcsel(
      &b, nir_ieq_imm(&b, vertex_id, 1), nir_imm_float(&b, extent),
      nir_imm_float(&b, 0.0f));
   nir_store_var(&b, position, pos, 0xf);
   nir_store_var(&b, coord, s, 0x1);
   return finish_test_shader(&b);
}

static nir_shader *
make_uint_3d_texel_fetch_fs(int z)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-uint-3d-texel-fetch-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_3D, false, GLSL_TYPE_UINT),
      "uint_3d_texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_txf;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->coord_components = 3;
   tex->dest_type = nir_type_uint32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_ivec3(&b, 0, 0, z));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_lod, nir_imm_int(&b, 0));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *normalized = nir_fmul_imm(
      &b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, normalized, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_uint_3d_texture_lod_fs(float r, float lod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-uint-3d-texture-lod-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_3D, false, GLSL_TYPE_UINT),
      "uint_3d_texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->coord_components = 3;
   tex->dest_type = nir_type_uint32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec3(&b, 0.5f, 0.5f, r));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_lod, nir_imm_float(&b, lod));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *normalized = nir_fmul_imm(
      &b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, normalized, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_uint_3d_texture_implicit_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-uint-3d-texture-implicit-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_3D, false, GLSL_TYPE_UINT),
      "uint_3d_texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_variable *input = nir_variable_create(
      b.shader, nir_var_shader_in,
      glsl_vector_type(GLSL_TYPE_FLOAT, 3), "texture_coord");
   input->data.location = VARYING_SLOT_VAR0;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_3D;
   tex->coord_components = 3;
   tex->dest_type = nir_type_uint32;
   tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                      nir_load_var(&b, input));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *normalized = nir_fmul_imm(
      &b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, normalized, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_fmad_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-fmad-fs");
   nir_variable *input = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec4_type(), "vertex_color");
   input->data.location = VARYING_SLOT_VAR0;
   nir_def *red = nir_fmad(&b, nir_channel(&b, nir_load_var(&b, input), 1),
                           nir_imm_float(&b, 1.0f / 16.0f),
                           nir_imm_float(&b, 0.25f));
   nir_def *result = nir_vec4(&b, red, nir_imm_float(&b, 0.0f),
                              nir_imm_float(&b, 0.0f),
                              nir_imm_float(&b, 1.0f));
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, result, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_frag_coord_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-frag-coord-fs");
   nir_def *coord = nir_load_frag_coord(&b);
   coord = nir_fmul_imm(&b, coord, 1.0f / 8.0f);
   nir_def *result = nir_vec4(&b, nir_channel(&b, coord, 0),
                              nir_channel(&b, coord, 1),
                              nir_imm_float(&b, 0.0f),
                              nir_imm_float(&b, 1.0f));
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, result, 0xf);
   return finish_test_shader(&b);
}


static nir_shader *
make_texture_array_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-texture-array-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, true, GLSL_TYPE_FLOAT),
      "texture_array0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_array = true;
   tex->coord_components = 3;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec3(&b, 0.5f, 0.5f, 1.0f));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, &tex->def, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_texture_cube_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-texture-cube-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_CUBE, false, GLSL_TYPE_FLOAT),
      "texture_cube0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->coord_components = 3;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec3(&b, 0.0f, 0.0f, -1.0f));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, &tex->def, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_uint_texture_cube_faces_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-uint-texture-cube-lod-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_CUBE, false, GLSL_TYPE_UINT),
      "uint_texture_cube0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   static const float directions[12][3] = {
      { 1.0f,  0.0f,  0.0f}, { 1.0f,  0.6f,  0.6f},
      {-1.0f,  0.0f,  0.0f}, {-1.0f,  0.6f, -0.6f},
      { 0.0f,  1.0f,  0.0f}, {-0.6f,  1.0f, -0.6f},
      { 0.0f, -1.0f,  0.0f}, {-0.6f, -1.0f,  0.6f},
      { 0.0f,  0.0f,  1.0f}, {-0.6f,  0.6f,  1.0f},
      { 0.0f,  0.0f, -1.0f}, { 0.6f,  0.6f, -1.0f},
   };
   static const uint8_t expected[12][4] = {
      { 4, 0, 0, 255}, { 1, 0, 0, 255},
      { 8, 0, 0, 255}, { 5, 0, 0, 255},
      {12, 0, 0, 255}, { 9, 0, 0, 255},
      {16, 0, 0, 255}, {13, 0, 0, 255},
      {20, 0, 0, 255}, {17, 0, 0, 255},
      {24, 0, 0, 255}, {21, 0, 0, 255},
   };
   nir_def *all_match = nir_imm_true(&b);
   for (unsigned sample = 0; sample < 12; ++sample) {
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
      tex->op = nir_texop_txl;
      tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
      tex->coord_components = 3;
      tex->dest_type = nir_type_uint32;
      tex->src[0] = nir_tex_src_for_ssa(
         nir_tex_src_coord,
         nir_imm_vec3(&b, directions[sample][0], directions[sample][1],
                      directions[sample][2]));
      tex->src[1] = nir_tex_src_for_ssa(
         nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
      tex->src[2] = nir_tex_src_for_ssa(
         nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
      tex->src[3] = nir_tex_src_for_ssa(
         nir_tex_src_lod, nir_imm_float(&b, 0.0f));
      nir_def_init(&tex->instr, &tex->def, 4, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      for (unsigned channel = 0; channel < 4; ++channel)
         all_match = nir_iand(
            &b, all_match,
            nir_ieq_imm(&b, nir_channel(&b, &tex->def, channel),
                        expected[sample][channel]));
   }
   nir_def *result = nir_bcsel(
      &b, all_match, nir_imm_vec4(&b, 0.0f, 1.0f, 0.0f, 1.0f),
      nir_imm_vec4(&b, 1.0f, 0.0f, 0.0f, 1.0f));
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, result, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_texture_cube_array_fs(bool integer, float cube_index)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, integer
         ? "gex-uint-texture-cube-array-fs"
         : "gex-float-texture-cube-array-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_CUBE, true,
                        integer ? GLSL_TYPE_UINT : GLSL_TYPE_FLOAT),
      integer ? "uint_texture_cube_array0" : "texture_cube_array0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txl;
   tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
   tex->is_array = true;
   tex->coord_components = 4;
   tex->dest_type = integer ? nir_type_uint32 : nir_type_float32;
   /* Top-left of +Z in the selected cube.  This simultaneously validates the
    * cube projection, face order, and six-layer cube-index stride. */
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord,
      nir_imm_vec4(&b, -0.6f, 0.6f, 1.0f, cube_index));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_lod, nir_imm_float(&b, 0.0f));
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *result = integer
      ? nir_fmul_imm(&b, nir_u2f32(&b, &tex->def), 1.0f / 255.0f)
      : &tex->def;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, result, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_shadow_fs(float reference)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, NULL,
                                                   "gex-shadow-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
      "shadow_texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "shadow_sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->is_shadow = true;
   /* GLSL's sampler2DShadow lookups return a scalar, which is what both
    * TGSI's shadow TEX (replicated) and SPIR-V's OpImageSampleDref produce.
    * Without this the instruction's result size is 4 and NIR validation
    * rejects the single-component destination below.
    */
   tex->is_new_style_shadow = true;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec2(&b, 0.5f, 0.5f));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_comparator, nir_imm_float(&b, reference));
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 1, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *sample = &tex->def;
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color,
                 nir_vec4(&b, sample, sample, sample,
                          nir_imm_float(&b, 1.0f)),
                 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_texture_gradient_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-texture-gradient-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
      "texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
   tex->op = nir_texop_txd;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec2(&b, 0.5f, 0.5f));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_ddx, nir_imm_vec2(&b, 0.5f, 0.0f));
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_ddy, nir_imm_vec2(&b, 0.0f, 0.5f));
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[4] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, &tex->def, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_texture_bias_fs(float bias)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-texture-bias-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
      "texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_variable *input = nir_variable_create(
      b.shader, nir_var_shader_in, glsl_vec2_type(), "texture_coord");
   input->data.location = VARYING_SLOT_VAR0;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
   tex->op = nir_texop_txb;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_load_var(&b, input));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_bias, nir_imm_float(&b, bias));
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref, &nir_build_deref_var(&b, texture)->def);
   tex->src[3] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref, &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, &tex->def, 0xf);
   return finish_test_shader(&b);
}

static nir_shader *
make_texture_constant_fs(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, NULL, "gex-texture-constant-fs");
   nir_variable *texture = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_texture_type(GLSL_SAMPLER_DIM_2D, false, GLSL_TYPE_FLOAT),
      "texture0");
   texture->data.descriptor_set = 0;
   texture->data.binding = 0;
   nir_variable *sampler = nir_variable_create(
      b.shader, nir_var_uniform, glsl_bare_sampler_type(), "sampler0");
   sampler->data.descriptor_set = 0;
   sampler->data.binding = 1;
   nir_tex_instr *tex = nir_tex_instr_create(b.shader, 3);
   tex->op = nir_texop_tex;
   tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
   tex->coord_components = 2;
   tex->dest_type = nir_type_float32;
   tex->src[0] = nir_tex_src_for_ssa(
      nir_tex_src_coord, nir_imm_vec2(&b, 0.75f, 0.25f));
   tex->src[1] = nir_tex_src_for_ssa(
      nir_tex_src_texture_deref,
      &nir_build_deref_var(&b, texture)->def);
   tex->src[2] = nir_tex_src_for_ssa(
      nir_tex_src_sampler_deref,
      &nir_build_deref_var(&b, sampler)->def);
   nir_def_init(&tex->instr, &tex->def, 4, 32);
   nir_builder_instr_insert(&b, &tex->instr);
   nir_def *constant = make_constant_load(&b, 2);
   nir_variable *color = nir_variable_create(
      b.shader, nir_var_shader_out, glsl_vec4_type(), "out_color");
   color->data.location = FRAG_RESULT_DATA0;
   color->data.index = 0;
   nir_store_var(&b, color, nir_fadd(&b, &tex->def, constant), 0xf);
   return finish_constant_test_shader(&b);
}

static bool
check_uniform_tolerance(struct pipe_context *ctx, struct pipe_resource *resource,
                        const uint8_t expected[4], unsigned tolerance,
                        const char *label)
{
   struct pipe_box box = {.x = 0, .y = 0, .z = 0,
                          .width = 8, .height = 8, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   const uint8_t *pixels = ctx->texture_map(ctx, resource, 0, PIPE_MAP_READ,
                                            &box, &transfer);
   if (!pixels || !transfer) {
      fprintf(stderr, "FAIL gex-wgpu %s map\n", label);
      return false;
   }
   for (unsigned y = 0; y < 8; ++y) {
      for (unsigned x = 0; x < 8; ++x) {
         const uint8_t *pixel = readback_pixel(pixels, transfer->stride, x, y);
         bool matches = true;
         for (unsigned channel = 0; channel < 4; ++channel) {
            int delta = (int)pixel[channel] - (int)expected[channel];
            if (delta < 0) delta = -delta;
            matches &= (unsigned)delta <= tolerance;
         }
         if (!matches) {
            fprintf(stderr, "FAIL gex-wgpu %s pixel %u,%u=%u,%u,%u,%u\n",
                    label, x, y, pixel[0], pixel[1], pixel[2], pixel[3]);
            ctx->texture_unmap(ctx, transfer);
            return false;
         }
      }
   }
   ctx->texture_unmap(ctx, transfer);
   return true;
}

/* How many pixels differ from the clear colour. Points and lines are asserted
 * on this rather than on a pixel pattern: which pixels a line covers is a
 * rasterization rule this corpus does not pin down, but whether anything was
 * drawn at all is exactly what a rejected pipeline gets wrong. */
static int
count_covered(struct pipe_context *ctx, struct pipe_resource *resource,
              const char *label)
{
   const uint8_t blue[4] = {64, 128, 191, 255};
   struct pipe_box box = {.x = 0, .y = 0, .z = 0,
                          .width = 8, .height = 8, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   const uint8_t *pixels = ctx->texture_map(ctx, resource, 0, PIPE_MAP_READ,
                                            &box, &transfer);
   if (!pixels || !transfer) {
      fprintf(stderr, "FAIL gex-wgpu %s map\n", label);
      return -1;
   }
   int covered = 0;
   for (unsigned y = 0; y < 8; ++y) {
      for (unsigned x = 0; x < 8; ++x) {
         if (memcmp(readback_pixel(pixels, transfer->stride, x, y), blue, 4))
            ++covered;
      }
   }
   ctx->texture_unmap(ctx, transfer);
   return covered;
}

static bool
check_uniform(struct pipe_context *ctx, struct pipe_resource *resource,
              const uint8_t expected[4], const char *label)
{
   return check_uniform_tolerance(ctx, resource, expected, 0, label);
}

static bool
check_asymmetric_color_tolerance(struct pipe_context *ctx,
                                 struct pipe_resource *resource,
                                 bool positive_y, bool scissored,
                                 const uint8_t foreground[4],
                                 unsigned tolerance, const char *label)
{
   const uint8_t blue[4] = {64, 128, 191, 255};
   struct pipe_box box = {.x = 0, .y = 0, .z = 0,
                          .width = 8, .height = 8, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   const uint8_t *pixels = ctx->texture_map(ctx, resource, 0, PIPE_MAP_READ,
                                            &box, &transfer);
   if (!pixels || !transfer) {
      fprintf(stderr, "FAIL gex-wgpu %s map\n", label);
      return false;
   }
   for (unsigned y = 0; y < 8; ++y) {
      for (unsigned x = 0; x < 8; ++x) {
         bool covered = positive_y ? 2 * x + y <= 6 : y >= 2 * x + 1;
         if (scissored)
            covered &= x >= 1 && x < 3 && y >= 1 && y < 6;
         const uint8_t *expected = covered ? foreground : blue;
         const uint8_t *pixel = readback_pixel(pixels, transfer->stride, x, y);
         bool matches = true;
         for (unsigned channel = 0; channel < 4; ++channel) {
            int delta = (int)pixel[channel] - (int)expected[channel];
            if (delta < 0) delta = -delta;
            matches &= (unsigned)delta <= tolerance;
         }
         if (!matches) {
            fprintf(stderr, "FAIL gex-wgpu %s pixel %u,%u=%u,%u,%u,%u\n",
                    label, x, y, pixel[0], pixel[1], pixel[2], pixel[3]);
            ctx->texture_unmap(ctx, transfer);
            return false;
         }
      }
   }
   ctx->texture_unmap(ctx, transfer);
   return true;
}

static bool
check_asymmetric_color(struct pipe_context *ctx, struct pipe_resource *resource,
                       bool positive_y, bool scissored,
                       const uint8_t foreground[4], const char *label)
{
   return check_asymmetric_color_tolerance(
      ctx, resource, positive_y, scissored, foreground, 0, label);
}

static bool
check_asymmetric(struct pipe_context *ctx, struct pipe_resource *resource,
                 bool positive_y, bool scissored, const char *label)
{
   const uint8_t green[4] = {0, 255, 0, 255};
   return check_asymmetric_color(ctx, resource, positive_y, scissored, green,
                                 label);
}

static bool
check_frag_coord(struct pipe_context *ctx, struct pipe_resource *resource)
{
   const uint8_t blue[4] = {64, 128, 191, 255};
   struct pipe_box box = {.x = 0, .y = 0, .z = 0,
                          .width = 8, .height = 8, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   const uint8_t *pixels = ctx->texture_map(ctx, resource, 0, PIPE_MAP_READ,
                                            &box, &transfer);
   if (!pixels || !transfer) {
      fprintf(stderr, "FAIL gex-wgpu frag-coord map\n");
      return false;
   }
   for (unsigned y = 0; y < 8; ++y) {
      for (unsigned x = 0; x < 8; ++x) {
         const uint8_t foreground[4] = {
            ((2 * x + 1) * 255 + 8) / 16,
            ((2 * y + 1) * 255 + 8) / 16,
            0, 255,
         };
         const uint8_t *expected = 2 * x + y <= 6 ? foreground : blue;
         const uint8_t *pixel = readback_pixel(pixels, transfer->stride, x, y);
         if (memcmp(pixel, expected, 4) != 0) {
            fprintf(stderr,
                    "FAIL gex-wgpu frag-coord pixel %u,%u=%u,%u,%u,%u\n",
                    x, y, pixel[0], pixel[1], pixel[2], pixel[3]);
            ctx->texture_unmap(ctx, transfer);
            return false;
         }
      }
   }
   ctx->texture_unmap(ctx, transfer);
   return true;
}

static int
run_gallium_conformance(struct pipe_screen *screen, bool webgpu_boundary,
                        const char *backend)
{
   if (!screen) {
      fprintf(stderr, "FAIL gex-gallium %s screen\n", backend);
      return 2;
   }
   test_screen = screen;
   const mesa_shader_stage required_nir_stages[] = {
      MESA_SHADER_VERTEX, MESA_SHADER_FRAGMENT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(required_nir_stages); ++i) {
      const mesa_shader_stage stage = required_nir_stages[i];
      if (!screen->nir_options[stage]) {
         fprintf(stderr, "FAIL gex-gallium %s NIR options stage=%u\n",
                 backend, stage);
         screen->destroy(screen);
         return 2;
      }
   }
   if (!screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_UNORM,
                                    PIPE_TEXTURE_2D, 1, 1,
                                    PIPE_BIND_RENDER_TARGET) ||
       (webgpu_boundary &&
        (!screen->is_format_supported(screen, PIPE_FORMAT_R8_UNORM,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8G8_UNORM,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_B8G8R8A8_UNORM,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_UINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8_UINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8G8_UINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8_SINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8G8_SINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_SINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_SAMPLER_VIEW) ||
         /* These are in WebGPU's core renderable set, and the driver now
          * reports them.  They were once pinned as unsupported here, which
          * also withheld the GL extensions st/mesa derives from render-target
          * support -- ARB_texture_rg among them, which OpenGL ES 3.0
          * requires. */
         !screen->is_format_supported(screen, PIPE_FORMAT_R8_UNORM,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_RENDER_TARGET) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_B8G8R8A8_UNORM,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_RENDER_TARGET) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_UINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_RENDER_TARGET) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8_UINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_RENDER_TARGET) ||
         !screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_SINT,
                                      PIPE_TEXTURE_2D, 1, 1,
                                      PIPE_BIND_RENDER_TARGET) ||
         /* 4x multisampled render targets, proven by multisample-4x-resolve. */
         !screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_UNORM,
                                      PIPE_TEXTURE_2D, 4, 4,
                                      PIPE_BIND_RENDER_TARGET) ||
         /* Two samples is not part of WebGPU's core set, and a
          * block-compressed format is never a render target. */
         screen->is_format_supported(screen, PIPE_FORMAT_R8G8B8A8_UNORM,
                                     PIPE_TEXTURE_2D, 2, 2,
                                     PIPE_BIND_RENDER_TARGET) ||
         screen->is_format_supported(screen, PIPE_FORMAT_DXT1_RGBA,
                                     PIPE_TEXTURE_2D, 1, 1,
                                     PIPE_BIND_RENDER_TARGET)))) {
      fprintf(stderr, "FAIL gex-gallium %s capability boundary\n", backend);
      screen->destroy(screen);
      return 3;
   }

   struct pipe_resource templ;
   memset(&templ, 0, sizeof(templ));
   templ.target = PIPE_TEXTURE_2D;
   templ.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   templ.width0 = 8;
   templ.height0 = 8;
   templ.depth0 = 1;
   templ.array_size = 1;
   templ.nr_samples = 1;
   templ.nr_storage_samples = 1;
   templ.bind = PIPE_BIND_RENDER_TARGET;
   struct pipe_resource *resource = screen->resource_create(screen, &templ);
   struct pipe_context *ctx = screen->context_create(screen, NULL, 0);
   if (!resource || !ctx) {
      fprintf(stderr, "FAIL gex-wgpu resource/context\n");
      if (resource) pipe_resource_reference(&resource, NULL);
      if (ctx) gex_destroy_context(ctx);
      screen->destroy(screen);
      return 4;
   }

   struct pipe_surface surface;
   memset(&surface, 0, sizeof(surface));
   pipe_reference_init(&surface.reference, 1);
   surface.format = templ.format;
   surface.texture = resource;
   union pipe_color_union color = {.f = {0.25f, 0.5f, 0.75f, 1.0f}};
   ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);

   struct pipe_shader_state vs_state;
   struct pipe_shader_state fs_state;
   memset(&vs_state, 0, sizeof(vs_state));
   memset(&fs_state, 0, sizeof(fs_state));
   vs_state.type = PIPE_SHADER_IR_NIR;
   vs_state.ir.nir = make_vs(0.0f, false);
   vs_state.report_compile_error = true;
   fs_state.type = PIPE_SHADER_IR_NIR;
   fs_state.ir.nir = make_varying_fs();
   fs_state.report_compile_error = true;
   void *vs = ctx->create_vs_state(ctx, &vs_state);
   void *fs = ctx->create_fs_state(ctx, &fs_state);
   if (!vs || !fs) {
      fprintf(stderr, "FAIL gex-wgpu shader vs=%s fs=%s\n",
              vs_state.error_message ? vs_state.error_message : "unknown",
              fs_state.error_message ? fs_state.error_message : "unknown");
      free(vs_state.error_message);
      free(fs_state.error_message);
      if (vs) gex_delete_vs(ctx, vs);
      if (fs) gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 5;
   }
   free(vs_state.error_message);
   free(fs_state.error_message);
   gex_bind_vs(ctx, vs);
   gex_bind_fs(ctx, fs);
   struct pipe_framebuffer_state framebuffer;
   memset(&framebuffer, 0, sizeof(framebuffer));
   framebuffer.width = 8;
   framebuffer.height = 8;
   framebuffer.samples = 1;
   framebuffer.nr_cbufs = 1;
   framebuffer.cbufs[0] = surface;
   ctx->set_framebuffer_state(ctx, &framebuffer);
   struct pipe_viewport_state viewport;
   memset(&viewport, 0, sizeof(viewport));
   viewport.scale[0] = 4.0f;
   viewport.scale[1] = 4.0f;
   viewport.scale[2] = 0.5f;
   viewport.translate[0] = 4.0f;
   viewport.translate[1] = 4.0f;
   viewport.translate[2] = 0.5f;
   viewport.swizzle_x = PIPE_VIEWPORT_SWIZZLE_POSITIVE_X;
   viewport.swizzle_y = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Y;
   viewport.swizzle_z = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Z;
   viewport.swizzle_w = PIPE_VIEWPORT_SWIZZLE_POSITIVE_W;
   ctx->set_viewport_states(ctx, 0, 1, &viewport);
   struct pipe_rasterizer_state base_rasterizer_state;
   memset(&base_rasterizer_state, 0, sizeof(base_rasterizer_state));
   base_rasterizer_state.half_pixel_center = 1;
   base_rasterizer_state.depth_clip_near = 1;
   base_rasterizer_state.depth_clip_far = 1;
   void *base_rasterizer =
      ctx->create_rasterizer_state(ctx, &base_rasterizer_state);
   if (!base_rasterizer) {
      fprintf(stderr, "FAIL gex-gallium %s base rasterizer\n", backend);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 7;
   }
   gex_bind_rast(ctx, base_rasterizer);
   struct pipe_blend_state base_blend_state;
   memset(&base_blend_state, 0, sizeof(base_blend_state));
   base_blend_state.rt[0].colormask = PIPE_MASK_RGBA;
   void *base_blend = ctx->create_blend_state(ctx, &base_blend_state);
   struct pipe_depth_stencil_alpha_state base_dsa_state;
   memset(&base_dsa_state, 0, sizeof(base_dsa_state));
   void *base_dsa =
      ctx->create_depth_stencil_alpha_state(ctx, &base_dsa_state);
   if (!base_blend || !base_dsa) {
      fprintf(stderr, "FAIL gex-gallium %s base fragment state\n", backend);
      if (base_blend) gex_delete_blend(ctx, base_blend);
      if (base_dsa) gex_delete_dsa(ctx, base_dsa);
      gex_delete_rast(ctx, base_rasterizer);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 7;
   }
   gex_bind_blend(ctx, base_blend);
   gex_bind_dsa(ctx, base_dsa);
   void *empty_vertex_elements = NULL;
#ifdef GEX_CONFORMANCE_SOFTPIPE
   empty_vertex_elements =
      ctx->create_vertex_elements_state(ctx, 0, NULL);
   if (!empty_vertex_elements) {
      fprintf(stderr, "FAIL gex-gallium softpipe empty vertex state\n");
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 7;
   }
   gex_bind_velems(ctx, empty_vertex_elements);
   ctx->set_sample_mask(ctx, ~0u);
#endif
   struct pipe_draw_info draw_info;
   memset(&draw_info, 0, sizeof(draw_info));
   draw_info.mode = MESA_PRIM_TRIANGLES;
   draw_info.instance_count = 1;
   draw_info.index_bounds_valid = true;
   draw_info.min_index = 0;
   draw_info.max_index = 2;
   struct pipe_draw_start_count_bias draw = {.start = 0, .count = 3};
   ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
   const uint8_t green[4] = {0, 255, 0, 255};
   const uint8_t blue[4] = {64, 128, 191, 255};
   if (!check_asymmetric(ctx, resource, true, false,
                         "explicit-positive-viewport")) {
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 7;
   }
   /* Geometry shaders, which WebGPU has no stage for: gex_wgpu runs the
    * vertex shader as compute into a buffer, the geometry shader as compute
    * into vertices and indices, and rasterizes the result. Skipped where the
    * backend does not offer the stage at all. */
   if (screen->shader_caps[MESA_SHADER_GEOMETRY].max_instructions) {
      struct pipe_shader_state gs_state = {0};
      gs_state.type = PIPE_SHADER_IR_NIR;
      gs_state.ir.nir = make_passthrough_gs();
      gs_state.report_compile_error = true;
      void *gs = ctx->create_gs_state(ctx, &gs_state);
      if (!gs)
         fprintf(stderr, "FAIL gex geometry shader=%s\n",
                 gs_state.error_message ? gs_state.error_message : "unknown");
      bool geometry_ok = gs != NULL;
      if (geometry_ok) {
         const uint8_t magenta[4] = {255, 0, 255, 255};
         ctx->bind_gs_state(ctx, gs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         geometry_ok = check_asymmetric_color(ctx, resource, true, false,
                                              magenta, "geometry-passthrough");
         ctx->bind_gs_state(ctx, NULL);
         ctx->delete_gs_state(ctx, gs);
      }
      free(gs_state.error_message);
      if (!geometry_ok) {
         gex_bind_vs(ctx, NULL);
         gex_bind_fs(ctx, NULL);
         gex_delete_vs(ctx, vs);
         gex_delete_fs(ctx, fs);
         gex_destroy_context(ctx);
         pipe_resource_reference(&resource, NULL);
         screen->destroy(screen);
         return 7;
      }
      /* Put the surface back to what the next case expects. */
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
   }

   /* Every primitive topology GL can draw. The ones WebGPU has go straight to
    * the pipeline; the rest -- fans, loops, quads -- are rewritten by
    * u_primconvert. This is the case that would have caught the driver
    * accepting only independent triangles while st/mesa, reading the default
    * supported_prim_modes, believed it accepted everything. */
   {
      struct pipe_shader_state quad_state = {0}, fan_state = {0};
      struct pipe_shader_state sparse_state = {0};
      quad_state.type = PIPE_SHADER_IR_NIR;
      quad_state.ir.nir = make_topology_vs(false, 1.0f);
      quad_state.report_compile_error = true;
      fan_state.type = PIPE_SHADER_IR_NIR;
      fan_state.ir.nir = make_topology_vs(true, 1.0f);
      fan_state.report_compile_error = true;
      sparse_state.type = PIPE_SHADER_IR_NIR;
      sparse_state.ir.nir = make_topology_vs(false, 0.5f);
      sparse_state.report_compile_error = true;
      void *quad_vs = ctx->create_vs_state(ctx, &quad_state);
      void *fan_vs = ctx->create_vs_state(ctx, &fan_state);
      void *sparse_vs = ctx->create_vs_state(ctx, &sparse_state);
      bool topology_ok = quad_vs && fan_vs && sparse_vs;
      if (!topology_ok)
         fprintf(stderr, "FAIL gex topology shaders\n");

      struct pipe_draw_info topology_info = draw_info;
      topology_info.index_bounds_valid = false;
      struct pipe_draw_start_count_bias four = {.start = 0, .count = 4};
      const uint8_t red[4] = {255, 0, 0, 255};

      /* Each of these covers the whole viewport, so the result is exact. */
      const struct {
         enum mesa_prim mode;
         void **shader;
         const char *name;
      } covering[] = {
         {MESA_PRIM_TRIANGLE_STRIP, &quad_vs, "topology-triangle-strip"},
         {MESA_PRIM_TRIANGLE_FAN, &fan_vs, "topology-triangle-fan"},
         {MESA_PRIM_QUADS, &fan_vs, "topology-quads"},
         {MESA_PRIM_POLYGON, &fan_vs, "topology-polygon"},
      };
      for (unsigned i = 0; topology_ok && i < ARRAY_SIZE(covering); ++i) {
         gex_bind_vs(ctx, *covering[i].shader);
         topology_info.mode = covering[i].mode;
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &topology_info, 0, NULL, &four, 1);
         topology_ok = check_uniform(ctx, resource, red, covering[i].name);
      }

      /* Points and lines cover too little to name pixel by pixel, so the
       * assertion is how much was covered. Four points at distinct pixels is
       * exact; a line's endpoints are a rasterization rule this corpus does
       * not pin down, so those only have to have drawn something. */
      const struct {
         enum mesa_prim mode;
         int least;
         int most;
         const char *name;
      } sparse[] = {
         {MESA_PRIM_POINTS, 4, 4, "topology-points"},
         {MESA_PRIM_LINES, 4, 64, "topology-lines"},
         {MESA_PRIM_LINE_STRIP, 4, 64, "topology-line-strip"},
         {MESA_PRIM_LINE_LOOP, 4, 64, "topology-line-loop"},
      };
      for (unsigned i = 0; topology_ok && i < ARRAY_SIZE(sparse); ++i) {
         gex_bind_vs(ctx, sparse_vs);
         topology_info.mode = sparse[i].mode;
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &topology_info, 0, NULL, &four, 1);
         const int covered = count_covered(ctx, resource, sparse[i].name);
         topology_ok = covered >= sparse[i].least && covered <= sparse[i].most;
         if (!topology_ok)
            fprintf(stderr, "FAIL gex-wgpu %s covered %d, wanted %d..%d\n",
                    sparse[i].name, covered, sparse[i].least, sparse[i].most);
      }

      gex_bind_vs(ctx, vs);
      if (quad_vs) gex_delete_vs(ctx, quad_vs);
      if (fan_vs) gex_delete_vs(ctx, fan_vs);
      if (sparse_vs) gex_delete_vs(ctx, sparse_vs);
      free(quad_state.error_message);
      free(fan_state.error_message);
      free(sparse_state.error_message);
      if (!topology_ok) {
         gex_bind_fs(ctx, NULL);
         gex_delete_vs(ctx, vs);
         gex_delete_fs(ctx, fs);
         gex_destroy_context(ctx);
         pipe_resource_reference(&resource, NULL);
         screen->destroy(screen);
         return 7;
      }
      /* Put the surface back to what the next case expects. */
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
   }

   struct pipe_shader_state fmad_state = {0};
   fmad_state.type = PIPE_SHADER_IR_NIR;
   fmad_state.ir.nir = make_fmad_fs();
   fmad_state.report_compile_error = true;
   void *fmad_fs = ctx->create_fs_state(ctx, &fmad_state);
   bool fmad_ok = fmad_fs != NULL;
   if (fmad_ok) {
      const uint8_t fmad_red[4] = {80, 0, 0, 255};
      gex_bind_fs(ctx, fmad_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      fmad_ok = check_asymmetric_color(
         ctx, resource, true, false, fmad_red, "fmad-varying");
   }
   if (!fmad_ok)
      fprintf(stderr, "FAIL gex-wgpu fmad shader=%s\n",
              fmad_state.error_message ? fmad_state.error_message : "unknown");
   free(fmad_state.error_message);
   gex_bind_fs(ctx, fs);
   if (fmad_fs)
      gex_delete_fs(ctx, fmad_fs);
   if (!fmad_ok) {
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 7;
   }
   if (webgpu_boundary) {
      struct pipe_shader_state frag_coord_state = {0};
      frag_coord_state.type = PIPE_SHADER_IR_NIR;
      frag_coord_state.ir.nir = make_frag_coord_fs();
      frag_coord_state.report_compile_error = true;
      void *frag_coord_fs = ctx->create_fs_state(ctx, &frag_coord_state);
      bool frag_coord_ok = frag_coord_fs != NULL;
      if (frag_coord_ok) {
         gex_bind_fs(ctx, frag_coord_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         frag_coord_ok = check_frag_coord(ctx, resource);
      }
      if (!frag_coord_ok)
         fprintf(stderr, "FAIL gex-wgpu half-integer frag-coord shader=%s\n",
                 frag_coord_state.error_message
                    ? frag_coord_state.error_message : "unknown");
      free(frag_coord_state.error_message);
      gex_bind_fs(ctx, fs);
      if (frag_coord_fs)
         gex_delete_fs(ctx, frag_coord_fs);
      if (!frag_coord_ok) {
         gex_bind_vs(ctx, NULL);
         gex_bind_fs(ctx, NULL);
         gex_delete_vs(ctx, vs);
         gex_delete_fs(ctx, fs);
         gex_destroy_context(ctx);
         pipe_resource_reference(&resource, NULL);
         screen->destroy(screen);
         return 7;
      }
   }

   struct pipe_resource *mrt_resource = screen->resource_create(screen, &templ);
   struct pipe_surface mrt_surface = {0};
   pipe_reference_init(&mrt_surface.reference, 1);
   mrt_surface.format = templ.format;
   mrt_surface.texture = mrt_resource;
   struct pipe_shader_state mrt_state = {0};
   mrt_state.type = PIPE_SHADER_IR_NIR;
   mrt_state.ir.nir = make_mrt_fs();
   mrt_state.report_compile_error = true;
   void *mrt_fs = ctx->create_fs_state(ctx, &mrt_state);
   bool mrt_ok = mrt_resource && mrt_fs;
   if (mrt_ok) {
      const uint8_t red[4] = {255, 0, 0, 255};
      const uint8_t cyan[4] = {0, 255, 255, 255};
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->clear_render_target(ctx, &mrt_surface, &color, 0, 0, 8, 8, false);
      framebuffer.nr_cbufs = 2;
      framebuffer.cbufs[1] = mrt_surface;
      ctx->set_framebuffer_state(ctx, &framebuffer);
      gex_bind_fs(ctx, mrt_fs);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      mrt_ok = check_asymmetric_color(
         ctx, resource, true, false, red, "mrt-color-zero");
      if (mrt_ok)
         mrt_ok = check_asymmetric_color(
            ctx, mrt_resource, true, false, cyan, "mrt-color-one");
      struct pipe_blend_state independent_blend_state = {0};
      independent_blend_state.independent_blend_enable = 1;
      independent_blend_state.rt[0].colormask = PIPE_MASK_R;
      independent_blend_state.rt[1].colormask = PIPE_MASK_G | PIPE_MASK_B;
      independent_blend_state.rt[1].blend_enable = 1;
      independent_blend_state.rt[1].rgb_func = PIPE_BLEND_ADD;
      independent_blend_state.rt[1].rgb_src_factor = PIPE_BLENDFACTOR_ONE;
      independent_blend_state.rt[1].rgb_dst_factor = PIPE_BLENDFACTOR_ONE;
      independent_blend_state.rt[1].alpha_func = PIPE_BLEND_ADD;
      independent_blend_state.rt[1].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
      independent_blend_state.rt[1].alpha_dst_factor = PIPE_BLENDFACTOR_ONE;
      void *independent_blend = ctx->create_blend_state(
         ctx, &independent_blend_state);
      if (mrt_ok)
         mrt_ok = independent_blend != NULL;
      if (mrt_ok) {
         const uint8_t masked_red[4] = {255, 128, 191, 255};
         const uint8_t additive_cyan[4] = {64, 255, 255, 255};
         gex_bind_blend(ctx, independent_blend);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->clear_render_target(ctx, &mrt_surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         mrt_ok = check_asymmetric_color(
            ctx, resource, true, false, masked_red,
            "mrt-independent-color-mask");
         if (mrt_ok)
            mrt_ok = check_asymmetric_color(
               ctx, mrt_resource, true, false, additive_cyan,
               "mrt-independent-additive-blend");
      }
      gex_bind_blend(ctx, base_blend);
      if (independent_blend)
         gex_delete_blend(ctx, independent_blend);
   }
   framebuffer.nr_cbufs = 1;
   memset(&framebuffer.cbufs[1], 0, sizeof(framebuffer.cbufs[1]));
   ctx->set_framebuffer_state(ctx, &framebuffer);
   gex_bind_fs(ctx, fs);
   if (mrt_fs)
      gex_delete_fs(ctx, mrt_fs);
   free(mrt_state.error_message);
   if (mrt_resource)
      pipe_resource_reference(&mrt_resource, NULL);
   if (!mrt_ok) {
      fprintf(stderr, "FAIL gex-gallium %s multiple render targets\n", backend);
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 7;
   }

   struct pipe_rasterizer_state rasterizer_state;
   memset(&rasterizer_state, 0, sizeof(rasterizer_state));
   rasterizer_state.scissor = 1;
   rasterizer_state.half_pixel_center = 1;
   rasterizer_state.depth_clip_near = 1;
   rasterizer_state.depth_clip_far = 1;
   void *rasterizer = ctx->create_rasterizer_state(ctx, &rasterizer_state);
   struct pipe_scissor_state scissor = {
      .minx = 1, .miny = 1, .maxx = 3, .maxy = 6,
   };
   if (!rasterizer) {
      fprintf(stderr, "FAIL gex-wgpu rasterizer state\n");
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 8;
   }
   gex_bind_rast(ctx, rasterizer);
   ctx->set_scissor_states(ctx, 0, 1, &scissor);
   ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
   ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);

   struct pipe_box box = {.x = 0, .y = 0, .z = 0,
                          .width = 8, .height = 8, .depth = 1};
   struct pipe_transfer *transfer = NULL;
   const uint8_t *pixels = ctx->texture_map(ctx, resource, 0, PIPE_MAP_READ,
                                            &box, &transfer);
   if (!pixels || !transfer) {
      fprintf(stderr, "FAIL gex-wgpu viewport/scissor map\n");
      gex_bind_rast(ctx, NULL);
      gex_delete_rast(ctx, rasterizer);
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 9;
   }
   for (unsigned y = 0; y < 8; ++y) {
      for (unsigned x = 0; x < 8; ++x) {
         const uint8_t *pixel = readback_pixel(pixels, transfer->stride, x, y);
         const bool covered = 2 * x + y <= 6 &&
                              x >= 1 && x < 3 && y >= 1 && y < 6;
         const uint8_t *expected = covered ? green : blue;
         if (memcmp(pixel, expected, sizeof(green)) != 0) {
            fprintf(stderr,
                    "FAIL gex-wgpu viewport/scissor pixel %u,%u=%u,%u,%u,%u\n",
                    x, y, pixel[0], pixel[1], pixel[2], pixel[3]);
            ctx->texture_unmap(ctx, transfer);
            gex_bind_rast(ctx, NULL);
            gex_delete_rast(ctx, rasterizer);
            gex_bind_vs(ctx, NULL);
            gex_bind_fs(ctx, NULL);
            gex_delete_vs(ctx, vs);
            gex_delete_fs(ctx, fs);
            gex_destroy_context(ctx);
            pipe_resource_reference(&resource, NULL);
            screen->destroy(screen);
            return 10;
         }
      }
   }
   ctx->texture_unmap(ctx, transfer);
   gex_bind_rast(ctx, base_rasterizer);
   gex_delete_rast(ctx, rasterizer);

   struct pipe_shader_state inside_state;
   struct pipe_shader_state outside_state;
   memset(&inside_state, 0, sizeof(inside_state));
   memset(&outside_state, 0, sizeof(outside_state));
   inside_state.type = PIPE_SHADER_IR_NIR;
   inside_state.ir.nir = make_vs(-0.5f, true);
   inside_state.report_compile_error = true;
   outside_state.type = PIPE_SHADER_IR_NIR;
   outside_state.ir.nir = make_vs(-1.25f, true);
   outside_state.report_compile_error = true;
   void *inside_vs = ctx->create_vs_state(ctx, &inside_state);
   void *outside_vs = ctx->create_vs_state(ctx, &outside_state);
   bool clip_ok = inside_vs && outside_vs;
   if (!clip_ok) {
      fprintf(stderr, "FAIL gex-wgpu clip shaders inside=%s outside=%s\n",
              inside_state.error_message ? inside_state.error_message : "unknown",
              outside_state.error_message ? outside_state.error_message : "unknown");
   } else {
      gex_bind_vs(ctx, inside_vs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      clip_ok = check_uniform(ctx, resource, green, "clip-inside-negative-z");
      if (clip_ok) {
         gex_bind_vs(ctx, outside_vs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         clip_ok = check_uniform(ctx, resource, blue, "clip-outside-negative-z");
      }
      /* GL_DEPTH_CLAMP: the same triangle, behind the near plane, must now be
       * clamped into the depth range and drawn rather than clipped away. That
       * it is the *same* shader as the case above is the point -- only the
       * rasterizer state differs, so a backend that ignored the state would
       * still report blue here. */
      if (clip_ok && screen->caps.depth_clip_disable) {
         struct pipe_rasterizer_state clamp_state = base_rasterizer_state;
         clamp_state.depth_clip_near = 0;
         clamp_state.depth_clip_far = 0;
         void *clamp = ctx->create_rasterizer_state(ctx, &clamp_state);
         clip_ok = clamp != NULL;
         if (!clip_ok)
            fprintf(stderr, "FAIL gex depth-clamp rasterizer state\n");
         if (clip_ok) {
            gex_bind_rast(ctx, clamp);
            gex_bind_vs(ctx, outside_vs);
            ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            clip_ok = check_uniform(ctx, resource, green,
                                    "depth-clamp-outside-near");
            gex_bind_rast(ctx, base_rasterizer);
            gex_delete_rast(ctx, clamp);
         }
      }
   }
   free(inside_state.error_message);
   free(outside_state.error_message);
   gex_bind_vs(ctx, NULL);
   if (inside_vs) gex_delete_vs(ctx, inside_vs);
   if (outside_vs) gex_delete_vs(ctx, outside_vs);
   if (!clip_ok) {
      gex_bind_fs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 11;
   }

   struct pipe_resource constant_templ;
   memset(&constant_templ, 0, sizeof(constant_templ));
   constant_templ.target = PIPE_BUFFER;
   constant_templ.format = PIPE_FORMAT_R8_UNORM;
   constant_templ.width0 = 512;
   constant_templ.height0 = 1;
   constant_templ.depth0 = 1;
   constant_templ.array_size = 1;
   constant_templ.bind = PIPE_BIND_CONSTANT_BUFFER;
   struct pipe_resource *constant_resource =
      screen->resource_create(screen, &constant_templ);
   const float constant_red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
   const float constant_cyan[4] = {0.0f, 1.0f, 1.0f, 1.0f};
   if (constant_resource) {
      ctx->buffer_subdata(ctx, constant_resource, PIPE_MAP_WRITE, 0,
                          sizeof(constant_red), constant_red);
      ctx->buffer_subdata(ctx, constant_resource, PIPE_MAP_WRITE, 256,
                          sizeof(constant_cyan), constant_cyan);
   }
   struct pipe_shader_state constant_fs_state = {0};
   constant_fs_state.type = PIPE_SHADER_IR_NIR;
   constant_fs_state.ir.nir = make_constant_buffer_fs(0);
   constant_fs_state.report_compile_error = true;
   void *constant_fs = ctx->create_fs_state(ctx, &constant_fs_state);
   bool constant_ok = constant_resource && constant_fs;
   struct pipe_constant_buffer constant = {
      .buffer = constant_resource,
      .buffer_offset = 0,
      .buffer_size = sizeof(constant_red),
   };
   if (constant_ok) {
      const uint8_t red[4] = {255, 0, 0, 255};
      const uint8_t cyan[4] = {0, 255, 255, 255};
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, constant_fs);
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, &constant);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      constant_ok = check_asymmetric_color(
         ctx, resource, true, false, red, "constant-buffer-offset-zero");
      if (constant_ok) {
         constant.buffer_offset = 256;
         ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, &constant);
         pipe_resource_reference(&constant_resource, NULL);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         constant_ok = check_asymmetric_color(
            ctx, resource, true, false, cyan,
            "constant-buffer-subrange-rebind-lifetime");
      }
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, NULL);
   }
   if (constant_ok) {
      const float user_white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
      const uint8_t white[4] = {255, 255, 255, 255};
      struct pipe_constant_buffer user_constant = {
         .buffer_size = sizeof(user_white),
         .user_buffer = user_white,
      };
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, constant_fs);
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, &user_constant);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      constant_ok = check_asymmetric_color(
         ctx, resource, true, false, white, "constant-user-buffer-upload");
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, NULL);
   }
   struct pipe_shader_state second_vec4_state = {0};
   second_vec4_state.type = PIPE_SHADER_IR_NIR;
   second_vec4_state.ir.nir = make_constant_buffer_second_vec4_fs();
   second_vec4_state.report_compile_error = true;
   void *second_vec4_fs = ctx->create_fs_state(ctx, &second_vec4_state);
   struct pipe_resource *second_vec4_resource =
      screen->resource_create(screen, &constant_templ);
   const float two_vec4s[8] = {
      1.0f, 0.0f, 0.0f, 1.0f,
      0.0f, 1.0f, 1.0f, 1.0f,
   };
   if (second_vec4_resource)
      ctx->buffer_subdata(ctx, second_vec4_resource, PIPE_MAP_WRITE, 0,
                          sizeof(two_vec4s), two_vec4s);
   if (constant_ok)
      constant_ok = second_vec4_fs && second_vec4_resource;
   if (constant_ok) {
      const uint8_t cyan[4] = {0, 255, 255, 255};
      struct pipe_constant_buffer second_vec4_constant = {
         .buffer = second_vec4_resource,
         .buffer_size = sizeof(two_vec4s),
      };
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, second_vec4_fs);
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0,
                               &second_vec4_constant);
      pipe_resource_reference(&second_vec4_resource, NULL);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      constant_ok = check_asymmetric_color(
         ctx, resource, true, false, cyan,
         "constant-buffer-second-vec4-offset");
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, NULL);
   }
   if (second_vec4_resource)
      pipe_resource_reference(&second_vec4_resource, NULL);
   gex_bind_fs(ctx, fs);
   if (second_vec4_fs)
      gex_delete_fs(ctx, second_vec4_fs);
   free(second_vec4_state.error_message);
   struct pipe_shader_state dynamic_constant_state = {0};
   dynamic_constant_state.type = PIPE_SHADER_IR_NIR;
   dynamic_constant_state.ir.nir = make_dynamic_constant_buffer_vs();
   dynamic_constant_state.report_compile_error = true;
   void *dynamic_constant_vs =
      ctx->create_vs_state(ctx, &dynamic_constant_state);
   struct pipe_resource *dynamic_constant_resource =
      screen->resource_create(screen, &constant_templ);
   const float dynamic_positions[12] = {
      -1.0f, -1.0f, 0.0f, 1.0f,
       0.0f, -1.0f, 0.0f, 1.0f,
      -1.0f,  1.0f, 0.0f, 1.0f,
   };
   if (dynamic_constant_resource)
      ctx->buffer_subdata(ctx, dynamic_constant_resource, PIPE_MAP_WRITE, 0,
                          sizeof(dynamic_positions), dynamic_positions);
   if (constant_ok)
      constant_ok = dynamic_constant_vs && dynamic_constant_resource;
   if (constant_ok) {
      struct pipe_constant_buffer dynamic_constant = {
         .buffer = dynamic_constant_resource,
         .buffer_size = sizeof(dynamic_positions),
      };
      gex_bind_vs(ctx, dynamic_constant_vs);
      gex_bind_fs(ctx, fs);
      ctx->set_constant_buffer(ctx, MESA_SHADER_VERTEX, 0,
                               &dynamic_constant);
      pipe_resource_reference(&dynamic_constant_resource, NULL);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      constant_ok = check_asymmetric(
         ctx, resource, true, false, "constant-buffer-dynamic-vec4-index");
      ctx->set_constant_buffer(ctx, MESA_SHADER_VERTEX, 0, NULL);
   }
   if (dynamic_constant_resource)
      pipe_resource_reference(&dynamic_constant_resource, NULL);
   gex_bind_vs(ctx, NULL);
   if (dynamic_constant_vs)
      gex_delete_vs(ctx, dynamic_constant_vs);
   free(dynamic_constant_state.error_message);
   struct pipe_shader_state two_constant_state = {0};
   two_constant_state.type = PIPE_SHADER_IR_NIR;
   two_constant_state.ir.nir = make_two_constant_buffer_fs();
   two_constant_state.report_compile_error = true;
   void *two_constant_fs = ctx->create_fs_state(ctx, &two_constant_state);
   struct pipe_resource *two_constant_resources[2] = {
      screen->resource_create(screen, &constant_templ),
      screen->resource_create(screen, &constant_templ),
   };
   const float two_constant_values[2][4] = {
      {1.0f, 0.0f, 0.0f, 0.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
   };
   for (unsigned i = 0; i < 2; ++i) {
      if (two_constant_resources[i])
         ctx->buffer_subdata(ctx, two_constant_resources[i], PIPE_MAP_WRITE,
                             0, sizeof(two_constant_values[i]),
                             two_constant_values[i]);
   }
   if (constant_ok)
      constant_ok = two_constant_fs && two_constant_resources[0] &&
                    two_constant_resources[1];
   if (constant_ok) {
      const uint8_t yellow[4] = {255, 255, 0, 255};
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, two_constant_fs);
      for (unsigned i = 0; i < 2; ++i) {
         struct pipe_constant_buffer two_constant = {
            .buffer = two_constant_resources[i],
            .buffer_size = sizeof(two_constant_values[i]),
         };
         ctx->set_constant_buffer(
            ctx, MESA_SHADER_FRAGMENT, i, &two_constant);
         pipe_resource_reference(&two_constant_resources[i], NULL);
      }
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      constant_ok = check_asymmetric_color(
         ctx, resource, true, false, yellow,
         "two-fragment-constant-buffers-lifetime");
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, NULL);
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 1, NULL);
   }
   for (unsigned i = 0; i < 2; ++i) {
      if (two_constant_resources[i])
         pipe_resource_reference(&two_constant_resources[i], NULL);
   }
   gex_bind_fs(ctx, fs);
   if (two_constant_fs)
      gex_delete_fs(ctx, two_constant_fs);
   free(two_constant_state.error_message);

   /* The highest constant buffer slot the driver claims. On a device whose
    * uniform buffer limit is lower than that -- WebGPU guarantees only
    * twelve -- this slot is backed by a storage buffer, which is the only way
    * the driver can offer the thirteen slots ARB_uniform_buffer_object needs.
    */
   const unsigned high_slot =
      screen->shader_caps[MESA_SHADER_FRAGMENT].max_const_buffers - 1;
   if (constant_ok && high_slot > 0) {
      struct pipe_shader_state slot_state = {0};
      slot_state.type = PIPE_SHADER_IR_NIR;
      slot_state.ir.nir = make_slot_constant_buffer_fs(high_slot);
      slot_state.report_compile_error = true;
      void *slot_fs = ctx->create_fs_state(ctx, &slot_state);
      if (!slot_fs && slot_state.error_message)
         fprintf(stderr, "FAIL gex constant-buffer-high-slot shader=%s\n",
                 slot_state.error_message);
      struct pipe_resource *slot_resource =
         screen->resource_create(screen, &constant_templ);
      const float magenta[4] = {1.0f, 0.0f, 1.0f, 1.0f};
      if (slot_resource)
         ctx->buffer_subdata(ctx, slot_resource, PIPE_MAP_WRITE, 0,
                             sizeof(magenta), magenta);
      constant_ok = slot_fs && slot_resource;
      if (constant_ok) {
         const uint8_t expected[4] = {255, 0, 255, 255};
         const struct pipe_constant_buffer slot_constant = {
            .buffer = slot_resource,
            .buffer_size = sizeof(magenta),
         };
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, slot_fs);
         ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, high_slot,
                                  &slot_constant);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         constant_ok = check_asymmetric_color(
            ctx, resource, true, false, expected, "constant-buffer-high-slot");
         ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, high_slot, NULL);
      }
      gex_bind_fs(ctx, fs);
      if (slot_fs)
         gex_delete_fs(ctx, slot_fs);
      if (slot_resource)
         pipe_resource_reference(&slot_resource, NULL);
      free(slot_state.error_message);
   }

   struct pipe_shader_state constant_vs_state = {0};
   constant_vs_state.type = PIPE_SHADER_IR_NIR;
   constant_vs_state.ir.nir = make_constant_buffer_vs(2);
   constant_vs_state.report_compile_error = true;
   void *constant_vs = ctx->create_vs_state(ctx, &constant_vs_state);
   struct pipe_resource *vertex_constant_resource =
      screen->resource_create(screen, &constant_templ);
   const float constant_magenta[4] = {1.0f, 0.0f, 1.0f, 1.0f};
   if (vertex_constant_resource)
      ctx->buffer_subdata(ctx, vertex_constant_resource, PIPE_MAP_WRITE, 0,
                          sizeof(constant_magenta), constant_magenta);
   if (constant_ok)
      constant_ok = constant_vs && vertex_constant_resource;
   if (constant_ok) {
      const uint8_t magenta[4] = {255, 0, 255, 255};
      struct pipe_constant_buffer vertex_constant = {
         .buffer = vertex_constant_resource,
         .buffer_size = sizeof(constant_magenta),
      };
      gex_bind_vs(ctx, constant_vs);
      gex_bind_fs(ctx, fs);
      ctx->set_constant_buffer(ctx, MESA_SHADER_VERTEX, 0, &vertex_constant);
      pipe_resource_reference(&vertex_constant_resource, NULL);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      constant_ok = check_asymmetric_color(
         ctx, resource, true, false, magenta,
         "vertex-constant-buffer-lifetime");
      ctx->set_constant_buffer(ctx, MESA_SHADER_VERTEX, 0, NULL);
   }
   if (vertex_constant_resource)
      pipe_resource_reference(&vertex_constant_resource, NULL);
   struct pipe_resource *shared_constant_resource =
      screen->resource_create(screen, &constant_templ);
   const float constant_yellow[4] = {1.0f, 1.0f, 0.0f, 1.0f};
   if (shared_constant_resource)
      ctx->buffer_subdata(ctx, shared_constant_resource, PIPE_MAP_WRITE, 0,
                          sizeof(constant_yellow), constant_yellow);
   if (constant_ok)
      constant_ok = shared_constant_resource != NULL;
   if (constant_ok) {
      const uint8_t yellow[4] = {255, 255, 0, 255};
      struct pipe_constant_buffer shared_constant = {
         .buffer = shared_constant_resource,
         .buffer_size = sizeof(constant_yellow),
      };
      gex_bind_vs(ctx, constant_vs);
      gex_bind_fs(ctx, constant_fs);
      ctx->set_constant_buffer(ctx, MESA_SHADER_VERTEX, 0, &shared_constant);
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, &shared_constant);
      pipe_resource_reference(&shared_constant_resource, NULL);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      constant_ok = check_asymmetric_color(
         ctx, resource, true, false, yellow,
         "vertex-fragment-constant-buffer-coexistence-lifetime");
      ctx->set_constant_buffer(ctx, MESA_SHADER_VERTEX, 0, NULL);
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, NULL);
   }
   if (shared_constant_resource)
      pipe_resource_reference(&shared_constant_resource, NULL);
   if (constant_resource)
      pipe_resource_reference(&constant_resource, NULL);
   gex_bind_fs(ctx, fs);
   gex_bind_vs(ctx, NULL);
   if (constant_fs)
      gex_delete_fs(ctx, constant_fs);
   if (constant_vs)
      gex_delete_vs(ctx, constant_vs);
   if (!constant_ok) {
      fprintf(stderr, "FAIL gex-wgpu constant buffer shader=%s\n",
              constant_fs_state.error_message
                 ? constant_fs_state.error_message : "unknown");
      free(constant_fs_state.error_message);
      free(constant_vs_state.error_message);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 12;
   }
   free(constant_fs_state.error_message);
   free(constant_vs_state.error_message);

   struct pipe_resource vertex_templ;
   struct pipe_resource index_templ;
   memset(&vertex_templ, 0, sizeof(vertex_templ));
   vertex_templ.target = PIPE_BUFFER;
   vertex_templ.format = PIPE_FORMAT_R8_UNORM;
   vertex_templ.width0 = 64;
   vertex_templ.height0 = 1;
   vertex_templ.depth0 = 1;
   vertex_templ.array_size = 1;
   vertex_templ.bind = PIPE_BIND_VERTEX_BUFFER;
   index_templ = vertex_templ;
   index_templ.width0 = 6;
   index_templ.bind = PIPE_BIND_INDEX_BUFFER;
   struct pipe_resource *vertex_resource =
      screen->resource_create(screen, &vertex_templ);
   struct pipe_resource *index_resource =
      screen->resource_create(screen, &index_templ);

   const float vertices[16] = {
       9.0f,  9.0f, 9.0f, 9.0f,
      -1.0f, -1.0f, 0.0f, 1.0f,
       0.0f, -1.0f, 0.0f, 1.0f,
      -1.0f,  1.0f, 0.0f, 1.0f,
   };
   const uint16_t indices[3] = {0, 1, 2};
   if (vertex_resource && index_resource) {
      ctx->buffer_subdata(ctx, vertex_resource, PIPE_MAP_WRITE, 0,
                          sizeof(vertices), vertices);
      ctx->buffer_subdata(ctx, index_resource, PIPE_MAP_WRITE, 0,
                          sizeof(indices), indices);
   }
   struct pipe_vertex_element element;
   memset(&element, 0, sizeof(element));
   element.src_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
   element.src_stride = 16;
   void *vertex_elements = ctx->create_vertex_elements_state(ctx, 1, &element);
   struct pipe_vertex_buffer vertex_buffer;
   memset(&vertex_buffer, 0, sizeof(vertex_buffer));
   vertex_buffer.buffer.resource = vertex_resource;
   vertex_buffer.buffer_offset = 16;
   if (vertex_elements && vertex_resource)
      gex_bind_velems(ctx, vertex_elements);
   if (vertex_resource)
      ctx->set_vertex_buffers(ctx, 1, &vertex_buffer);

   struct pipe_shader_state fetch_state;
   memset(&fetch_state, 0, sizeof(fetch_state));
   fetch_state.type = PIPE_SHADER_IR_NIR;
   fetch_state.ir.nir = make_vertex_fetch_vs();
   fetch_state.report_compile_error = true;
   void *fetch_vs = ctx->create_vs_state(ctx, &fetch_state);
   void *interleaved_elements = NULL;
   void *interleaved_vs = NULL;
   void *compact2_elements = NULL, *compact3_elements = NULL;
   void *compact2_vs = NULL, *compact3_vs = NULL;
   void *separate_elements = NULL;
   struct pipe_resource *separate_color_resource = NULL;
   void *instance_elements = NULL;
   struct pipe_resource *instance_color_resource = NULL;
   bool fetch_ok = vertex_resource && index_resource && vertex_elements && fetch_vs;
   if (!fetch_ok) {
      fprintf(stderr, "FAIL gex-wgpu vertex/index setup shader=%s\n",
              fetch_state.error_message ? fetch_state.error_message : "unknown");
   } else {
      viewport.scale[1] = -4.0f;
      ctx->set_viewport_states(ctx, 0, 1, &viewport);
      gex_bind_vs(ctx, fetch_vs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      struct pipe_draw_info indexed_info;
      memset(&indexed_info, 0, sizeof(indexed_info));
      indexed_info.mode = MESA_PRIM_TRIANGLES;
      indexed_info.index_size = 2;
      indexed_info.instance_count = 1;
      indexed_info.index.resource = index_resource;
      ctx->draw_vbo(ctx, &indexed_info, 0, NULL, &draw, 1);
      fetch_ok = check_asymmetric(ctx, resource, false, false,
                                  "vertex-index-fetch");
      if (fetch_ok) {
         /* Primitive restart, which desktop GL 3.1 requires as
          * NV_primitive_restart. WebGPU has restart only for strip topologies
          * and only at a fixed index, so the driver resolves the general form
          * on the CPU and re-issues the draw as the ranges between restarts.
          *
          * The indices are the triangle, a restart, and then two more indices
          * that cannot finish a triangle. GL drops that incomplete primitive,
          * so this has to draw exactly what the unrestarted [0, 1, 2] above
          * drew -- while a backend that ignored the restart would assemble
          * (0xffff, 0, 1), whose out-of-range vertex moves the triangle. */
         struct pipe_resource restart_templ = index_templ;
         restart_templ.width0 = 6 * sizeof(uint16_t);
         struct pipe_resource *restart_index =
            screen->resource_create(screen, &restart_templ);
         const uint16_t restart_indices[6] = {0, 1, 2, 0xffff, 0, 1};
         if (restart_index)
            ctx->buffer_subdata(ctx, restart_index, PIPE_MAP_WRITE, 0,
                                sizeof(restart_indices), restart_indices);
         fetch_ok = restart_index != NULL;
         if (fetch_ok) {
            struct pipe_draw_info restart_info;
            memset(&restart_info, 0, sizeof(restart_info));
            restart_info.mode = MESA_PRIM_TRIANGLES;
            restart_info.index_size = 2;
            restart_info.instance_count = 1;
            restart_info.primitive_restart = true;
            restart_info.restart_index = 0xffff;
            restart_info.index.resource = restart_index;
            struct pipe_draw_start_count_bias restart_draw;
            memset(&restart_draw, 0, sizeof(restart_draw));
            restart_draw.count = 6;
            ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &restart_info, 0, NULL, &restart_draw, 1);
            fetch_ok = check_asymmetric(ctx, resource, false, false,
                                        "primitive-restart");
         }
         if (restart_index)
            pipe_resource_reference(&restart_index, NULL);
      }
      if (fetch_ok) {
         struct interleaved_vertex {
            float position[4];
            uint8_t color[4];
         } interleaved_vertices[3] = {
            {{-1.0f, -1.0f, 0.0f, 1.0f}, {255, 0, 255, 255}},
            {{ 0.0f, -1.0f, 0.0f, 1.0f}, {255, 0, 255, 255}},
            {{-1.0f,  1.0f, 0.0f, 1.0f}, {255, 0, 255, 255}},
         };
         ctx->buffer_subdata(ctx, vertex_resource, PIPE_MAP_WRITE, 0,
                             sizeof(interleaved_vertices),
                             interleaved_vertices);
         struct pipe_vertex_element interleaved[2];
         memset(interleaved, 0, sizeof(interleaved));
         interleaved[0].src_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         interleaved[0].src_stride = sizeof(struct interleaved_vertex);
         interleaved[1].src_format = PIPE_FORMAT_R8G8B8A8_UNORM;
         interleaved[1].src_offset = 16;
         interleaved[1].src_stride = sizeof(struct interleaved_vertex);
         interleaved_elements =
            ctx->create_vertex_elements_state(ctx, 2, interleaved);
         struct pipe_shader_state interleaved_state = {0};
         interleaved_state.type = PIPE_SHADER_IR_NIR;
         interleaved_state.ir.nir = make_interleaved_vertex_color_vs();
         interleaved_state.report_compile_error = true;
         interleaved_vs = ctx->create_vs_state(ctx, &interleaved_state);
         vertex_buffer.buffer_offset = 0;
         if (interleaved_elements)
            gex_bind_velems(ctx, interleaved_elements);
         ctx->set_vertex_buffers(ctx, 1, &vertex_buffer);
         fetch_ok = interleaved_elements && interleaved_vs;
         if (fetch_ok) {
            const uint8_t magenta[4] = {255, 0, 255, 255};
            gex_bind_vs(ctx, interleaved_vs);
            gex_bind_fs(ctx, fs);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            fetch_ok = check_asymmetric_color(
               ctx, resource, false, false, magenta,
               "vertex-interleaved-float4-unorm8x4");
         }
         free(interleaved_state.error_message);
      }
      if (fetch_ok) {
         const float positions2[6] = {
            -1.0f, -1.0f, 0.0f, -1.0f, -1.0f, 1.0f,
         };
         ctx->buffer_subdata(ctx, vertex_resource, PIPE_MAP_WRITE, 0,
                             sizeof(positions2), positions2);
         struct pipe_vertex_element compact2 = {0};
         compact2.src_format = PIPE_FORMAT_R32G32_FLOAT;
         compact2.src_stride = 8;
         compact2_elements =
            ctx->create_vertex_elements_state(ctx, 1, &compact2);
         struct pipe_shader_state compact2_state = {0};
         compact2_state.type = PIPE_SHADER_IR_NIR;
         compact2_state.ir.nir = make_compact_position_vs(2, 0.0f, 1.0f, 1.0f);
         compact2_state.report_compile_error = true;
         compact2_vs = ctx->create_vs_state(ctx, &compact2_state);
         fetch_ok = compact2_elements && compact2_vs;
         if (fetch_ok) {
            gex_bind_velems(ctx, compact2_elements);
            gex_bind_vs(ctx, compact2_vs);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            const uint8_t cyan[4] = {0, 255, 255, 255};
            fetch_ok = check_asymmetric_color(
               ctx, resource, false, false, cyan, "vertex-float2-expansion");
         }
         free(compact2_state.error_message);
      }
      if (fetch_ok) {
         const float positions3[9] = {
            -1.0f, -1.0f, 0.0f,
             0.0f, -1.0f, 0.0f,
            -1.0f,  1.0f, 0.0f,
         };
         ctx->buffer_subdata(ctx, vertex_resource, PIPE_MAP_WRITE, 0,
                             sizeof(positions3), positions3);
         struct pipe_vertex_element compact3 = {0};
         compact3.src_format = PIPE_FORMAT_R32G32B32_FLOAT;
         compact3.src_stride = 12;
         compact3_elements =
            ctx->create_vertex_elements_state(ctx, 1, &compact3);
         struct pipe_shader_state compact3_state = {0};
         compact3_state.type = PIPE_SHADER_IR_NIR;
         compact3_state.ir.nir = make_compact_position_vs(3, 1.0f, 1.0f, 0.0f);
         compact3_state.report_compile_error = true;
         compact3_vs = ctx->create_vs_state(ctx, &compact3_state);
         fetch_ok = compact3_elements && compact3_vs;
         if (fetch_ok) {
            gex_bind_velems(ctx, compact3_elements);
            gex_bind_vs(ctx, compact3_vs);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            const uint8_t yellow[4] = {255, 255, 0, 255};
            fetch_ok = check_asymmetric_color(
               ctx, resource, false, false, yellow,
               "vertex-float3-expansion");
         }
         free(compact3_state.error_message);
      }
      if (fetch_ok) {
         const float separate_positions[12] = {
            -1.0f, -1.0f, 0.0f, 1.0f,
             0.0f, -1.0f, 0.0f, 1.0f,
            -1.0f,  1.0f, 0.0f, 1.0f,
         };
         const uint8_t separate_colors[12] = {
            0, 128, 255, 255, 0, 128, 255, 255, 0, 128, 255, 255,
         };
         struct pipe_resource color_buffer_templ = vertex_templ;
         color_buffer_templ.width0 = sizeof(separate_colors);
         separate_color_resource =
            screen->resource_create(screen, &color_buffer_templ);
         ctx->buffer_subdata(ctx, vertex_resource, PIPE_MAP_WRITE, 0,
                             sizeof(separate_positions), separate_positions);
         if (separate_color_resource)
            ctx->buffer_subdata(ctx, separate_color_resource, PIPE_MAP_WRITE, 0,
                                sizeof(separate_colors), separate_colors);
         struct pipe_vertex_element separate[2] = {0};
         separate[0].src_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         separate[0].src_stride = 16;
         separate[0].vertex_buffer_index = 0;
         separate[1].src_format = PIPE_FORMAT_R8G8B8A8_UNORM;
         separate[1].src_stride = 4;
         separate[1].vertex_buffer_index = 1;
         separate_elements =
            ctx->create_vertex_elements_state(ctx, 2, separate);
         struct pipe_vertex_buffer separate_buffers[2] = {0};
         separate_buffers[0].buffer.resource = vertex_resource;
         separate_buffers[1].buffer.resource = separate_color_resource;
         fetch_ok = separate_color_resource && separate_elements;
         if (fetch_ok) {
            gex_bind_velems(ctx, separate_elements);
            ctx->set_vertex_buffers(ctx, 2, separate_buffers);
            pipe_resource_reference(&separate_color_resource, NULL);
            gex_bind_vs(ctx, interleaved_vs);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            const uint8_t azure[4] = {0, 128, 255, 255};
            fetch_ok = check_asymmetric_color(
               ctx, resource, false, false, azure,
               "vertex-separate-float4-unorm8x4-streams");
         }
      }
      if (fetch_ok) {
         const uint8_t instance_colors[8] = {
            255, 0, 0, 255, 255, 128, 0, 255,
         };
         struct pipe_resource instance_buffer_templ = vertex_templ;
         instance_buffer_templ.width0 = sizeof(instance_colors);
         instance_color_resource =
            screen->resource_create(screen, &instance_buffer_templ);
         if (instance_color_resource)
            ctx->buffer_subdata(ctx, instance_color_resource, PIPE_MAP_WRITE, 0,
                                sizeof(instance_colors), instance_colors);
         struct pipe_vertex_element instanced[2] = {0};
         instanced[0].src_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
         instanced[0].src_stride = 16;
         instanced[0].vertex_buffer_index = 0;
         instanced[1].src_format = PIPE_FORMAT_R8G8B8A8_UNORM;
         instanced[1].src_stride = 4;
         instanced[1].vertex_buffer_index = 1;
         instanced[1].instance_divisor = 1;
         instance_elements =
            ctx->create_vertex_elements_state(ctx, 2, instanced);
         struct pipe_vertex_buffer instance_buffers[2] = {0};
         instance_buffers[0].buffer.resource = vertex_resource;
         instance_buffers[1].buffer.resource = instance_color_resource;
         fetch_ok = instance_color_resource && instance_elements;
         if (fetch_ok) {
            gex_bind_velems(ctx, instance_elements);
            ctx->set_vertex_buffers(ctx, 2, instance_buffers);
            pipe_resource_reference(&instance_color_resource, NULL);
            gex_bind_vs(ctx, interleaved_vs);
            struct pipe_draw_info instance_info = draw_info;
            instance_info.instance_count = 1;
            instance_info.start_instance = 0;
            const uint8_t red[4] = {255, 0, 0, 255};
            const uint8_t orange[4] = {255, 128, 0, 255};
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &instance_info, 0, NULL, &draw, 1);
            fetch_ok = check_asymmetric_color(
               ctx, resource, false, false, red,
               "vertex-instance-divisor-one-first");
            if (fetch_ok) {
               instance_info.start_instance = 1;
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &instance_info, 0, NULL, &draw, 1);
               fetch_ok = check_asymmetric_color(
                  ctx, resource, false, false, orange,
                  "vertex-instance-base-instance-one");
            }
            if (fetch_ok) {
               instance_info.start_instance = 0;
               instance_info.instance_count = 2;
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &instance_info, 0, NULL, &draw, 1);
               fetch_ok = check_asymmetric_color(
                  ctx, resource, false, false, orange,
                  "vertex-two-instance-order");
            }
         }
      }
   }
   free(fetch_state.error_message);
   gex_bind_vs(ctx, NULL);
   gex_bind_fs(ctx, NULL);
   if (fetch_vs) gex_delete_vs(ctx, fetch_vs);
   if (interleaved_vs) gex_delete_vs(ctx, interleaved_vs);
   if (compact2_vs) gex_delete_vs(ctx, compact2_vs);
   if (compact3_vs) gex_delete_vs(ctx, compact3_vs);
   ctx->set_vertex_buffers(ctx, 0, NULL);
   gex_bind_velems(ctx, empty_vertex_elements);
   if (vertex_elements)
      gex_delete_velems(ctx, vertex_elements);
   if (interleaved_elements)
      gex_delete_velems(ctx, interleaved_elements);
   if (compact2_elements)
      gex_delete_velems(ctx, compact2_elements);
   if (compact3_elements)
      gex_delete_velems(ctx, compact3_elements);
   if (separate_elements)
      gex_delete_velems(ctx, separate_elements);
   if (instance_elements)
      gex_delete_velems(ctx, instance_elements);
   if (separate_color_resource)
      pipe_resource_reference(&separate_color_resource, NULL);
   if (instance_color_resource)
      pipe_resource_reference(&instance_color_resource, NULL);
   if (vertex_resource) pipe_resource_reference(&vertex_resource, NULL);
   if (index_resource) pipe_resource_reference(&index_resource, NULL);

   if (fetch_ok && webgpu_boundary && ctx->launch_grid &&
       ctx->set_shader_buffers) {
      /* Compute writing a storage buffer -- what transform feedback needs,
       * since WebGPU forbids that from the vertex stage.
       *
       * WebGPU only: this shader keeps its storage buffer as a variable and
       * derefs, which is the form the SPIR-V emitter consumes. Backends that
       * go through TGSI want the lowered store_ssbo form instead, so the same
       * NIR cannot serve both and softpipe is not a reference here. */
      struct pipe_resource compute_templ;
      memset(&compute_templ, 0, sizeof(compute_templ));
      compute_templ.target = PIPE_BUFFER;
      compute_templ.format = PIPE_FORMAT_R8_UNORM;
      compute_templ.width0 = 4 * sizeof(uint32_t);
      compute_templ.height0 = 1;
      compute_templ.depth0 = 1;
      compute_templ.array_size = 1;
      compute_templ.bind = PIPE_BIND_SHADER_BUFFER;
      struct pipe_resource *compute_buffer =
         screen->resource_create(screen, &compute_templ);

      struct pipe_compute_state compute_state = {0};
      compute_state.ir_type = PIPE_SHADER_IR_NIR;
      compute_state.prog = make_compute_fill_cs();
      void *cs = compute_buffer
         ? ctx->create_compute_state(ctx, &compute_state) : NULL;
      fetch_ok = compute_buffer && cs;
      if (fetch_ok) {
         struct pipe_shader_buffer binding = {0};
         binding.buffer = compute_buffer;
         binding.buffer_offset = 0;
         binding.buffer_size = compute_templ.width0;
         ctx->set_shader_buffers(ctx, MESA_SHADER_COMPUTE, 0, 1, &binding, 1);
         ctx->bind_compute_state(ctx, cs);

         struct pipe_grid_info grid = {0};
         grid.block[0] = 1; grid.block[1] = 1; grid.block[2] = 1;
         grid.grid[0] = 4; grid.grid[1] = 1; grid.grid[2] = 1;
         ctx->launch_grid(ctx, &grid);

         struct pipe_box compute_box = {
            .x = 0, .y = 0, .z = 0,
            .width = (int)compute_templ.width0, .height = 1, .depth = 1,
         };
         struct pipe_transfer *compute_transfer = NULL;
         const uint32_t *filled = ctx->buffer_map(
            ctx, compute_buffer, 0, PIPE_MAP_READ, &compute_box,
            &compute_transfer);
         if (!filled) {
            fprintf(stderr, "FAIL gex-wgpu compute-storage-write map\n");
            fetch_ok = false;
         } else {
            for (unsigned i = 0; i < 4; ++i) {
               if (filled[i] == i * 7 + 3)
                  continue;
               fprintf(stderr,
                       "FAIL gex-wgpu compute-storage-write word %u = %u, "
                       "expected %u\n", i, filled[i], i * 7 + 3);
               fetch_ok = false;
               break;
            }
            ctx->buffer_unmap(ctx, compute_transfer);
         }
         ctx->bind_compute_state(ctx, NULL);
         ctx->set_shader_buffers(ctx, MESA_SHADER_COMPUTE, 0, 1, NULL, 0);
      }
      if (cs)
         ctx->delete_compute_state(ctx, cs);
      if (compute_buffer)
         pipe_resource_reference(&compute_buffer, NULL);
   }

   if (fetch_ok && screen->caps.max_stream_output_buffers > 0) {
      /* Transform feedback: the vertex shader's captured varying, written to a
       * buffer and read back.
       *
       * WebGPU has no transform feedback stage, so gex_wgpu lowers the capture
       * into storage-buffer writes from the vertex shader. That lowering is
       * the whole feature, and this is the only place it can be exercised
       * before the advertised GL version is high enough to expose transform
       * feedback to a program. */
      struct pipe_resource capture_templ;
      memset(&capture_templ, 0, sizeof(capture_templ));
      capture_templ.target = PIPE_BUFFER;
      capture_templ.format = PIPE_FORMAT_R8_UNORM;
      capture_templ.width0 = 3 * 4 * sizeof(float);
      capture_templ.height0 = 1;
      capture_templ.depth0 = 1;
      capture_templ.array_size = 1;
      capture_templ.bind = PIPE_BIND_STREAM_OUTPUT;
      struct pipe_resource *capture =
         screen->resource_create(screen, &capture_templ);

      /* The uniform the capturing shader reads. A backend that runs the
       * capture as its own pass has to bind this for that pass too. */
      struct pipe_resource xfb_constant_templ;
      memset(&xfb_constant_templ, 0, sizeof(xfb_constant_templ));
      xfb_constant_templ.target = PIPE_BUFFER;
      xfb_constant_templ.format = PIPE_FORMAT_R8_UNORM;
      xfb_constant_templ.width0 = 256;
      xfb_constant_templ.height0 = 1;
      xfb_constant_templ.depth0 = 1;
      xfb_constant_templ.array_size = 1;
      xfb_constant_templ.bind = PIPE_BIND_CONSTANT_BUFFER;
      struct pipe_resource *xfb_constant =
         screen->resource_create(screen, &xfb_constant_templ);
      const float xfb_uniform[4] = {1.0f, 0.0f, 0.0f, 1.0f};
      if (xfb_constant)
         ctx->buffer_subdata(ctx, xfb_constant, PIPE_MAP_WRITE, 0,
                             sizeof(xfb_uniform), xfb_uniform);

      struct pipe_shader_state xfb_state = {0};
      xfb_state.type = PIPE_SHADER_IR_NIR;
      xfb_state.ir.nir = make_xfb_vs();
      xfb_state.report_compile_error = true;
      /* One buffer, one four-component varying, tightly packed. */
      xfb_state.stream_output.num_outputs = 1;
      xfb_state.stream_output.stride[0] = 4;
      /* The second output: gl_Position is written first, so the captured
       * varying is register one. Capturing register zero silently records
       * positions instead, which is what makes the expected values below
       * worth checking component by component. */
      xfb_state.stream_output.output[0].register_index = 1;
      xfb_state.stream_output.output[0].start_component = 0;
      xfb_state.stream_output.output[0].num_components = 4;
      xfb_state.stream_output.output[0].output_buffer = 0;
      xfb_state.stream_output.output[0].dst_offset = 0;

      /* The same description in the form the lowering reads. */
      nir_xfb_info *xfb =
         rzalloc_size(xfb_state.ir.nir,
                      sizeof(nir_xfb_info) + sizeof(nir_xfb_output_info));
      xfb->buffers_written = 1;
      xfb->streams_written = 1;
      /* Bytes, not dwords. `nir_gather_xfb_info` fills this in as
       * `nir->info.xfb_stride[i] * 4`, so the four dwords named above are
       * sixteen bytes here. Setting four made all three captured vertices
       * land one float apart instead of four, overlapping each other: the
       * buffer came back as `1 100 200 301 301 301 0 ...` because the last
       * write to each slot won, and the case has failed for that reason and
       * not because of anything in the driver. */
      xfb->buffers[0].stride = 16;
      xfb->buffers[0].varying_count = 1;
      xfb->output_count = 1;
      xfb->outputs[0].buffer = 0;
      xfb->outputs[0].offset = 0;
      xfb->outputs[0].location = VARYING_SLOT_VAR0;
      xfb->outputs[0].component_mask = 0xf;
      xfb_state.ir.nir->xfb_info = xfb;
      xfb_state.ir.nir->info.has_transform_feedback_varyings = true;
      xfb_state.ir.nir->info.xfb_stride[0] = 4;

      void *xfb_vs = ctx->create_vs_state(ctx, &xfb_state);
      if (!xfb_vs && xfb_state.error_message)
         fprintf(stderr, "FAIL gex xfb shader=%s\n", xfb_state.error_message);
      free(xfb_state.error_message);

      struct pipe_stream_output_target *so_target =
         capture && xfb_vs
            ? ctx->create_stream_output_target(ctx, capture, 0,
                                               capture_templ.width0) : NULL;
      fetch_ok = capture && xfb_vs && so_target && xfb_constant;
      if (fetch_ok) {
         const struct pipe_constant_buffer xfb_uniform_buffer = {
            .buffer = xfb_constant,
            .buffer_offset = 0,
            .buffer_size = sizeof(xfb_uniform),
         };
         ctx->set_constant_buffer(ctx, MESA_SHADER_VERTEX, 0,
                                  &xfb_uniform_buffer);
         const unsigned offsets[1] = {0};
         ctx->set_stream_output_targets(ctx, 1, &so_target, offsets,
                                        MESA_PRIM_TRIANGLES);
         gex_bind_vs(ctx, xfb_vs);
         gex_bind_fs(ctx, fs);
         /* Depth is disabled by this state, not by unbinding it: a NULL
          * depth-stencil-alpha state means "no state object", which a driver
          * is entitled to dereference. Whether a depth attachment exists is
          * the framebuffer's business. */
         gex_bind_dsa(ctx, base_dsa);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         ctx->set_stream_output_targets(ctx, 0, NULL, NULL, MESA_PRIM_TRIANGLES);

         struct pipe_box capture_box = {
            .x = 0, .y = 0, .z = 0,
            .width = (int)capture_templ.width0, .height = 1, .depth = 1,
         };
         struct pipe_transfer *capture_transfer = NULL;
         const float *captured = ctx->buffer_map(
            ctx, capture, 0, PIPE_MAP_READ, &capture_box, &capture_transfer);
         if (!captured) {
            fprintf(stderr, "FAIL gex-wgpu transform-feedback map\n");
            fetch_ok = false;
         } else {
            for (unsigned v = 0; v < 3 && fetch_ok; ++v) {
               /* The shader adds the bound uniform, (1, 0, 0, 1). */
               const float expected[4] = {
                  (float)v + 1.0f, 100.0f + (float)v, 200.0f, 301.0f,
               };
               for (unsigned c = 0; c < 4; ++c) {
                  if (captured[v * 4 + c] == expected[c])
                     continue;
                  fprintf(stderr,
                          "FAIL gex-wgpu transform-feedback vertex %u "
                          "component %u = %f, expected %f\n",
                          v, c, captured[v * 4 + c], expected[c]);
                  fprintf(stderr, "gex-wgpu: captured buffer =");
                  for (unsigned d = 0; d < 12; ++d)
                     fprintf(stderr, " %.1f", captured[d]);
                  fprintf(stderr, " (want 1 100 200 301 | 2 101 200 301 | "
                                  "3 102 200 301)\n");
                  fetch_ok = false;
                  break;
               }
            }
            ctx->buffer_unmap(ctx, capture_transfer);
         }
         gex_bind_vs(ctx, NULL);
      }
      ctx->set_constant_buffer(ctx, MESA_SHADER_VERTEX, 0, NULL);
      if (so_target)
         ctx->stream_output_target_destroy(ctx, so_target);
      if (xfb_vs)
         gex_delete_vs(ctx, xfb_vs);
      if (capture)
         pipe_resource_reference(&capture, NULL);
      if (xfb_constant)
         pipe_resource_reference(&xfb_constant, NULL);
   }

   if (fetch_ok && !screen->caps.fake_sw_msaa) {
      /* 4x multisampling with a resolve, which desktop GL 3.0 requires and
       * WebGPU provides directly.
       *
       * Skipped on a backend that fakes multisampling: it reports four samples
       * and stores one, so a resolve there proves nothing.
       *
       * A multisampled texture cannot be copied or mapped, so the only way to
       * see what was drawn is to resolve it into a single-sample one -- which
       * makes this a test of the resolve path, not just of pipeline creation.
       */
      struct pipe_resource msaa_templ = templ;
      msaa_templ.nr_samples = 4;
      msaa_templ.nr_storage_samples = 4;
      /* Sampler view as well as render target: a resolve is performed by
       * sampling the multisampled texture. */
      msaa_templ.bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW;
      struct pipe_resource *msaa = screen->resource_create(screen, &msaa_templ);
      fetch_ok = msaa != NULL;
      if (fetch_ok) {
         struct pipe_surface msaa_surface;
         memset(&msaa_surface, 0, sizeof(msaa_surface));
         pipe_reference_init(&msaa_surface.reference, 1);
         msaa_surface.format = msaa_templ.format;
         msaa_surface.texture = msaa;

         struct pipe_framebuffer_state msaa_fb;
         memset(&msaa_fb, 0, sizeof(msaa_fb));
         msaa_fb.width = 8;
         msaa_fb.height = 8;
         msaa_fb.nr_cbufs = 1;
         msaa_fb.cbufs[0] = msaa_surface;
         ctx->set_framebuffer_state(ctx, &msaa_fb);

         /* A fullscreen triangle, deliberately: with real multisampling the
          * pixels along a triangle edge resolve to a blend of the two sides,
          * so an exact-match check only means anything where no edge crosses
          * the surface. */
         struct pipe_shader_state msaa_vs_state = {0};
         msaa_vs_state.type = PIPE_SHADER_IR_NIR;
         msaa_vs_state.ir.nir = make_vs(0.0f, true);
         msaa_vs_state.report_compile_error = true;
         void *msaa_vs = ctx->create_vs_state(ctx, &msaa_vs_state);
         free(msaa_vs_state.error_message);
         fetch_ok = msaa_vs != NULL;
         if (!fetch_ok) {
            pipe_resource_reference(&msaa, NULL);
            goto msaa_done;
         }
         gex_bind_vs(ctx, msaa_vs);
         gex_bind_fs(ctx, fs);
         /* This framebuffer has no depth attachment, so the state bound here
          * disables depth -- which is not the same as binding none. A NULL
          * depth-stencil-alpha state is "no state object", and softpipe
          * dereferences it unconditionally when it builds its quad pipeline. */
         gex_bind_dsa(ctx, base_dsa);
         ctx->clear_render_target(ctx, &msaa_surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);

         /* Unbind before resolving: the multisampled texture is the blit
          * source, and reading a bound render target is not defined. */
         struct pipe_framebuffer_state restore;
         memset(&restore, 0, sizeof(restore));
         restore.width = 8;
         restore.height = 8;
         restore.nr_cbufs = 1;
         restore.cbufs[0] = surface;
         ctx->set_framebuffer_state(ctx, &restore);

         struct pipe_blit_info resolve;
         memset(&resolve, 0, sizeof(resolve));
         resolve.src.resource = msaa;
         resolve.src.format = msaa_templ.format;
         resolve.src.box.width = 8;
         resolve.src.box.height = 8;
         resolve.src.box.depth = 1;
         resolve.dst.resource = resource;
         resolve.dst.format = templ.format;
         resolve.dst.box.width = 8;
         resolve.dst.box.height = 8;
         resolve.dst.box.depth = 1;
         resolve.mask = PIPE_MASK_RGBA;
         resolve.filter = PIPE_TEX_FILTER_NEAREST;
         ctx->blit(ctx, &resolve);

         const uint8_t green[4] = {0, 255, 0, 255};
         fetch_ok = check_uniform(ctx, resource, green,
                                  "multisample-4x-resolve");

         /* ARB_texture_multisample: read the same texture from a shader
          * instead of resolving it. The resolve above already proved the
          * contents, so a different answer here is the fetch path's fault. */
         if (fetch_ok && screen->caps.texture_multisample) {
            struct pipe_shader_state ms_state = {0};
            ms_state.type = PIPE_SHADER_IR_NIR;
            ms_state.ir.nir = make_multisample_texel_fetch_fs();
            ms_state.report_compile_error = true;
            void *ms_fs = ctx->create_fs_state(ctx, &ms_state);
            if (!ms_fs)
               fprintf(stderr, "FAIL gex multisample-texel-fetch shader=%s\n",
                       ms_state.error_message ? ms_state.error_message
                                              : "unknown");
            struct pipe_sampler_view ms_view_templ;
            memset(&ms_view_templ, 0, sizeof(ms_view_templ));
            ms_view_templ.target = PIPE_TEXTURE_2D;
            ms_view_templ.format = msaa_templ.format;
            ms_view_templ.swizzle_r = PIPE_SWIZZLE_X;
            ms_view_templ.swizzle_g = PIPE_SWIZZLE_Y;
            ms_view_templ.swizzle_b = PIPE_SWIZZLE_Z;
            ms_view_templ.swizzle_a = PIPE_SWIZZLE_W;
            struct pipe_sampler_view *ms_view =
               ctx->create_sampler_view(ctx, msaa, &ms_view_templ);
            fetch_ok = ms_fs && ms_view;
            if (fetch_ok) {
               ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0,
                                      &ms_view);
               gex_bind_fs(ctx, ms_fs);
               ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8,
                                        false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               fetch_ok = check_uniform(ctx, resource, green,
                                        "multisample-texel-fetch");
               ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
               gex_bind_fs(ctx, fs);
            }
            if (ms_view)
               pipe_sampler_view_reference(&ms_view, NULL);
            if (ms_fs)
               gex_delete_fs(ctx, ms_fs);
            free(ms_state.error_message);
         }
         gex_bind_vs(ctx, NULL);
         gex_delete_vs(ctx, msaa_vs);
      }
      if (msaa)
         pipe_resource_reference(&msaa, NULL);
   }
msaa_done:

   if (fetch_ok) {
      /* gl_InstanceID, with no vertex buffers bound: the shader builds its own
       * fullscreen triangle, so this isolates the instance index from the
       * per-instance attribute path tested above. */
      struct pipe_shader_state instance_id_state = {0};
      instance_id_state.type = PIPE_SHADER_IR_NIR;
      instance_id_state.ir.nir = make_instance_id_vs();
      instance_id_state.report_compile_error = true;
      void *instance_id_vs = ctx->create_vs_state(ctx, &instance_id_state);
      fetch_ok = instance_id_vs != NULL;
      if (fetch_ok) {
         gex_bind_vs(ctx, instance_id_vs);
         gex_bind_fs(ctx, fs); /* the varying pass-through */
         struct pipe_draw_info instance_id_info = draw_info;
         instance_id_info.instance_count = 2;
         instance_id_info.start_instance = 0;
         struct pipe_draw_start_count_bias instance_id_draw = {
            .start = 0, .count = 3, .index_bias = 0,
         };
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &instance_id_info, 0, NULL, &instance_id_draw, 1);
         const uint8_t green[4] = {0, 255, 0, 255};
         fetch_ok = check_asymmetric_color(ctx, resource, false, false, green,
                                           "vertex-gl-instance-id");
      }
      if (!fetch_ok && instance_id_state.error_message)
         fprintf(stderr, "FAIL gex instance-id shader=%s\n",
                 instance_id_state.error_message);
      free(instance_id_state.error_message);
      gex_bind_vs(ctx, NULL);
      if (instance_id_vs)
         gex_delete_vs(ctx, instance_id_vs);
   }

   if (!fetch_ok) {
      gex_bind_fs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 12;
   }

   /* Uploading a texture by mapping it for write, which is how st/mesa moves
    * most of glTexImage2D. The corpus reached textures only through
    * texture_subdata before, so this path -- staging in memory and handing it
    * to the upload on unmap -- had no coverage at all. Every texel differs, so
    * a stride, origin or row-order mistake shows up rather than cancelling. */
   {
      struct pipe_resource map_templ;
      memset(&map_templ, 0, sizeof(map_templ));
      map_templ.target = PIPE_TEXTURE_2D;
      map_templ.format = PIPE_FORMAT_R8G8B8A8_UNORM;
      map_templ.width0 = 4;
      map_templ.height0 = 4;
      map_templ.depth0 = 1;
      map_templ.array_size = 1;
      map_templ.nr_samples = 1;
      map_templ.nr_storage_samples = 1;
      map_templ.bind = PIPE_BIND_SAMPLER_VIEW;
      struct pipe_resource *mapped = screen->resource_create(screen, &map_templ);
      struct pipe_box map_box = {
         .x = 0, .y = 0, .z = 0, .width = 4, .height = 4, .depth = 1,
      };
      bool map_ok = mapped != NULL;

      if (map_ok) {
         struct pipe_transfer *write = NULL;
         uint8_t *dst = ctx->texture_map(ctx, mapped, 0, PIPE_MAP_WRITE,
                                         &map_box, &write);
         if (!dst || !write) {
            fprintf(stderr, "FAIL gex-wgpu texture-write-map write map\n");
            map_ok = false;
         } else {
            for (unsigned y = 0; y < 4; ++y) {
               for (unsigned x = 0; x < 4; ++x) {
                  uint8_t *texel = dst + y * write->stride + x * 4;
                  texel[0] = (uint8_t)(16 * x + 1);
                  texel[1] = (uint8_t)(16 * y + 2);
                  texel[2] = (uint8_t)(4 * y + x);
                  texel[3] = 255;
               }
            }
            ctx->texture_unmap(ctx, write);
         }
      }

      if (map_ok) {
         struct pipe_transfer *read = NULL;
         const uint8_t *src = ctx->texture_map(ctx, mapped, 0, PIPE_MAP_READ,
                                               &map_box, &read);
         if (!src || !read) {
            fprintf(stderr, "FAIL gex-wgpu texture-write-map read back\n");
            map_ok = false;
         } else {
            for (unsigned y = 0; y < 4 && map_ok; ++y) {
               for (unsigned x = 0; x < 4 && map_ok; ++x) {
                  const uint8_t *texel = src + y * read->stride + x * 4;
                  const uint8_t want[4] = {
                     (uint8_t)(16 * x + 1), (uint8_t)(16 * y + 2),
                     (uint8_t)(4 * y + x), 255,
                  };
                  if (memcmp(texel, want, 4) == 0)
                     continue;
                  fprintf(stderr,
                          "FAIL gex-wgpu texture-write-map texel %u,%u = "
                          "%u,%u,%u,%u expected %u,%u,%u,%u\n", x, y,
                          texel[0], texel[1], texel[2], texel[3],
                          want[0], want[1], want[2], want[3]);
                  map_ok = false;
               }
            }
            ctx->texture_unmap(ctx, read);
         }
      }

      if (mapped)
         pipe_resource_reference(&mapped, NULL);
      if (!map_ok) {
         gex_bind_vs(ctx, NULL);
         gex_bind_fs(ctx, NULL);
         gex_delete_vs(ctx, vs);
         gex_delete_fs(ctx, fs);
         gex_destroy_context(ctx);
         pipe_resource_reference(&resource, NULL);
         screen->destroy(screen);
         return 12;
      }
   }

   struct pipe_resource sample_templ;
   memset(&sample_templ, 0, sizeof(sample_templ));
   sample_templ.target = PIPE_TEXTURE_2D;
   sample_templ.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   sample_templ.width0 = 2;
   sample_templ.height0 = 2;
   sample_templ.depth0 = 1;
   sample_templ.array_size = 1;
   sample_templ.nr_samples = 1;
   sample_templ.nr_storage_samples = 1;
   sample_templ.last_level = 1;
   sample_templ.bind = PIPE_BIND_SAMPLER_VIEW;
   struct pipe_resource *sample_texture =
      screen->resource_create(screen, &sample_templ);
   const uint8_t texels[16] = {
      255,   0,   0, 255, 255, 255,   0, 255,
        0,   0, 255, 255, 255, 255, 255, 255,
   };
   struct pipe_box sample_box = {
      .x = 0, .y = 0, .z = 0, .width = 2, .height = 2, .depth = 1,
   };
   if (sample_texture)
      ctx->texture_subdata(ctx, sample_texture, 0, PIPE_MAP_WRITE,
                           &sample_box, texels, 8, 16);
   const uint8_t green_texel[4] = {0, 255, 0, 255};
   struct pipe_box patch_box = {
      .x = 1, .y = 0, .z = 0, .width = 1, .height = 1, .depth = 1,
   };
   if (sample_texture)
      ctx->texture_subdata(ctx, sample_texture, 0, PIPE_MAP_WRITE,
                           &patch_box, green_texel, 4, 4);
   const uint8_t mip_texel[4] = {255, 0, 255, 255};
   struct pipe_box mip_box = {
      .x = 0, .y = 0, .z = 0, .width = 1, .height = 1, .depth = 1,
   };
   if (sample_texture)
      ctx->texture_subdata(ctx, sample_texture, 1, PIPE_MAP_WRITE,
                           &mip_box, mip_texel, 4, 4);
   struct pipe_sampler_view view_templ;
   memset(&view_templ, 0, sizeof(view_templ));
   view_templ.target = PIPE_TEXTURE_2D;
   view_templ.format = PIPE_FORMAT_R8G8B8A8_UNORM;
   view_templ.swizzle_r = PIPE_SWIZZLE_X;
   view_templ.swizzle_g = PIPE_SWIZZLE_Y;
   view_templ.swizzle_b = PIPE_SWIZZLE_Z;
   view_templ.swizzle_a = PIPE_SWIZZLE_W;
   view_templ.u.tex.last_level = 1;
   struct pipe_sampler_view *sample_view = sample_texture
      ? ctx->create_sampler_view(ctx, sample_texture, &view_templ) : NULL;
   struct pipe_sampler_state sampler_state;
   memset(&sampler_state, 0, sizeof(sampler_state));
   sampler_state.wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   sampler_state.wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   sampler_state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   sampler_state.min_img_filter = PIPE_TEX_FILTER_NEAREST;
   sampler_state.min_mip_filter = PIPE_TEX_MIPFILTER_NONE;
   sampler_state.mag_img_filter = PIPE_TEX_FILTER_NEAREST;
   void *sample_sampler = ctx->create_sampler_state(ctx, &sampler_state);
   if (sample_sampler) {
      void *samplers[1] = {sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
   if (sample_view) {
      struct pipe_sampler_view *views[1] = {sample_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
   }
   struct pipe_shader_state texture_state;
   memset(&texture_state, 0, sizeof(texture_state));
   texture_state.type = PIPE_SHADER_IR_NIR;
   texture_state.ir.nir = make_texture_fs(false, 0.75f, 0.25f, -1.0f);
   texture_state.report_compile_error = true;
   void *texture_fs = ctx->create_fs_state(ctx, &texture_state);
   bool texture_ok = sample_texture && sample_view && sample_sampler && texture_fs;
   if (!texture_ok) {
      fprintf(stderr, "FAIL gex-wgpu texture setup shader=%s\n",
              texture_state.error_message ? texture_state.error_message : "unknown");
   } else {
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, texture_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric(ctx, resource, false, false,
                                    "texture-nearest-green");
   }

   const enum pipe_format bc_formats[] = {
      PIPE_FORMAT_DXT1_RGB,
      PIPE_FORMAT_DXT1_SRGB,
      PIPE_FORMAT_DXT1_RGBA,
      PIPE_FORMAT_DXT1_SRGBA,
      PIPE_FORMAT_DXT3_RGBA,
      PIPE_FORMAT_DXT3_SRGBA,
      PIPE_FORMAT_DXT5_RGBA,
      PIPE_FORMAT_DXT5_SRGBA,
   };
   const char *bc_labels[] = {
      "texture-bc1-rgb-opaque-three-color-mode",
      "texture-bc1-srgb-opaque-three-color-mode",
      "texture-bc1-rgba-decoded-block",
      "texture-bc1-srgba-decoded-block",
      "texture-bc2-rgba-decoded-alpha-block",
      "texture-bc2-srgba-decoded-alpha-block",
      "texture-bc3-rgba-decoded-block",
      "texture-bc3-srgba-decoded-block",
   };
   struct pipe_resource *bc_textures[ARRAY_SIZE(bc_formats)] = {0};
   struct pipe_sampler_view *bc_views[ARRAY_SIZE(bc_formats)] = {0};
   /* RGB565 0x8410 expands to (132, 130, 132).  The same encoded endpoint
    * therefore distinguishes linear views from sRGB views after sampling. */
   const uint8_t bc1_gray[8] = {
      0x10, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   /* The BC1 three-color selector physically decodes transparent black.
    * DXT1_RGB requires the same RGB value with logical alpha forced to one. */
   const uint8_t bc1_transparent_black[8] = {
      0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
   };
   const uint8_t bc2_gray[16] = {
      0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
      0x10, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const uint8_t bc3_gray[16] = {
      0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x10, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   for (unsigned i = 0; texture_ok && i < ARRAY_SIZE(bc_formats); ++i) {
      const bool rgb_only = bc_formats[i] == PIPE_FORMAT_DXT1_RGB ||
                            bc_formats[i] == PIPE_FORMAT_DXT1_SRGB;
      const bool bc1 = rgb_only ||
                       bc_formats[i] == PIPE_FORMAT_DXT1_RGBA ||
                       bc_formats[i] == PIPE_FORMAT_DXT1_SRGBA;
      const bool bc2 = bc_formats[i] == PIPE_FORMAT_DXT3_RGBA ||
                       bc_formats[i] == PIPE_FORMAT_DXT3_SRGBA;
      const bool srgb = bc_formats[i] == PIPE_FORMAT_DXT1_SRGB ||
                        bc_formats[i] == PIPE_FORMAT_DXT1_SRGBA ||
                        bc_formats[i] == PIPE_FORMAT_DXT3_SRGBA ||
                        bc_formats[i] == PIPE_FORMAT_DXT5_SRGBA;
      const uint8_t *blocks = rgb_only ? bc1_transparent_black
                              : bc1 ? bc1_gray
                              : bc2 ? bc2_gray : bc3_gray;
      const unsigned block_bytes = bc1 ? 8 : 16;
      const uint8_t opaque_black[4] = {0, 0, 0, 255};
      const uint8_t linear_gray[4] = {132, 130, 132, 255};
      const uint8_t srgb_gray[4] = {59, 57, 59, 255};
      const uint8_t linear_gray_bc2[4] = {132, 130, 132, 136};
      const uint8_t srgb_gray_bc2[4] = {59, 57, 59, 136};
      const uint8_t *expected = rgb_only ? opaque_black
                                : bc2 && srgb ? srgb_gray_bc2
                                : bc2 ? linear_gray_bc2
                                : srgb ? srgb_gray : linear_gray;
      if (!webgpu_boundary)
         continue;
      struct pipe_resource bc_templ = sample_templ;
      bc_templ.format = bc_formats[i];
      bc_templ.width0 = 4;
      bc_templ.height0 = 4;
      bc_templ.last_level = 0;
      struct pipe_resource *bc_texture =
         screen->resource_create(screen, &bc_templ);
      bc_textures[i] = bc_texture;
      const struct pipe_box bc_box = {
         .x = 0, .y = 0, .z = 0, .width = 4, .height = 4, .depth = 1,
      };
      if (bc_texture)
         ctx->texture_subdata(ctx, bc_texture, 0, PIPE_MAP_WRITE,
                              &bc_box, blocks, block_bytes, block_bytes);
      struct pipe_sampler_view bc_view_templ = view_templ;
      bc_view_templ.format = bc_formats[i];
      bc_view_templ.u.tex.last_level = 0;
      struct pipe_sampler_view *bc_view = bc_texture
         ? ctx->create_sampler_view(ctx, bc_texture, &bc_view_templ) : NULL;
      bc_views[i] = bc_view;
      texture_ok = bc_texture && bc_view;
      if (texture_ok) {
         struct pipe_sampler_view *views[1] = {bc_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
         gex_bind_fs(ctx, texture_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, expected, bc_labels[i]);
      } else {
         fprintf(stderr, "FAIL gex-wgpu BC format %u setup\n", bc_formats[i]);
      }
   }
   const enum pipe_format rgtc_formats[] = {
      PIPE_FORMAT_RGTC1_UNORM, PIPE_FORMAT_RGTC1_SNORM,
      PIPE_FORMAT_RGTC2_UNORM, PIPE_FORMAT_RGTC2_SNORM,
   };
   const uint8_t bc4_unorm_one[8] = {
      0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const uint8_t bc4_snorm_one[8] = {
      0x7f, 0x81, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const uint8_t bc4_snorm_negative_one[8] = {
      0x81, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   uint8_t bc5_unorm_one[16], bc5_snorm_opposite[16];
   memcpy(bc5_unorm_one, bc4_unorm_one, 8);
   memcpy(bc5_unorm_one + 8, bc4_unorm_one, 8);
   memcpy(bc5_snorm_opposite, bc4_snorm_one, 8);
   memcpy(bc5_snorm_opposite + 8, bc4_snorm_negative_one, 8);
   struct pipe_resource *rgtc_textures[ARRAY_SIZE(rgtc_formats)] = {0};
   struct pipe_sampler_view *rgtc_views[ARRAY_SIZE(rgtc_formats)] = {0};
   for (unsigned i = 0;
        texture_ok && webgpu_boundary && i < ARRAY_SIZE(rgtc_formats); ++i) {
      const bool two_channel = i >= 2;
      const uint8_t *blocks = i == 0 ? bc4_unorm_one
                              : i == 1 ? bc4_snorm_one
                              : i == 2 ? bc5_unorm_one
                                       : bc5_snorm_opposite;
      const unsigned block_bytes = two_channel ? 16 : 8;
      struct pipe_resource templ = sample_templ;
      templ.format = rgtc_formats[i];
      templ.width0 = 4;
      templ.height0 = 4;
      templ.last_level = 0;
      rgtc_textures[i] = screen->resource_create(screen, &templ);
      const struct pipe_box box = {
         .x = 0, .y = 0, .z = 0, .width = 4, .height = 4, .depth = 1,
      };
      if (rgtc_textures[i])
         ctx->texture_subdata(ctx, rgtc_textures[i], 0, PIPE_MAP_WRITE,
                              &box, blocks, block_bytes, block_bytes);
      struct pipe_sampler_view rgtc_view_templ = view_templ;
      rgtc_view_templ.format = rgtc_formats[i];
      rgtc_view_templ.u.tex.last_level = 0;
      rgtc_views[i] = rgtc_textures[i]
         ? ctx->create_sampler_view(ctx, rgtc_textures[i], &rgtc_view_templ)
         : NULL;
      texture_ok = rgtc_textures[i] && rgtc_views[i];
      if (texture_ok) {
         const uint8_t red[4] = {255, 0, 0, 255};
         const uint8_t yellow[4] = {255, 255, 0, 255};
         const char *labels[] = {
            "texture-bc4-unorm-red-channel",
            "texture-bc4-snorm-positive-endpoint",
            "texture-bc5-unorm-two-channels",
            "texture-bc5-snorm-opposite-endpoints",
         };
         struct pipe_sampler_view *views[1] = {rgtc_views[i]};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
         gex_bind_fs(ctx, texture_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, i == 2 ? yellow : red, labels[i]);
      } else {
         fprintf(stderr, "FAIL gex-wgpu RGTC format %u setup\n",
                 rgtc_formats[i]);
      }
   }
   if (texture_ok && webgpu_boundary) {
      const enum pipe_format bptc_formats[] = {
         PIPE_FORMAT_BPTC_RGB_FLOAT,
         PIPE_FORMAT_BPTC_RGB_UFLOAT,
         PIPE_FORMAT_BPTC_RGBA_UNORM,
         PIPE_FORMAT_BPTC_SRGBA,
      };
      const char *labels[] = {
         "texture-bc6h-sfloat-negative-transform",
         "texture-bc6h-ufloat-hdr-decode",
         "texture-bc7-unorm-decode",
         "texture-bc7-srgb-decode",
      };
      uint8_t blocks[4][16];
      float signed_source[4 * 4 * 4];
      float unsigned_source[4 * 4 * 4];
      uint8_t unorm_source[4 * 4 * 4];
      for (unsigned pixel = 0; pixel < 16; ++pixel) {
         const float signed_value[4] = {-0.5f, 0.25f, 0.75f, 1.0f};
         const float unsigned_value[4] = {0.25f, 0.5f, 0.75f, 1.0f};
         const uint8_t unorm_value[4] = {132, 130, 132, 200};
         memcpy(signed_source + pixel * 4, signed_value,
                sizeof(signed_value));
         memcpy(unsigned_source + pixel * 4, unsigned_value,
                sizeof(unsigned_value));
         memcpy(unorm_source + pixel * 4, unorm_value,
                sizeof(unorm_value));
      }
      util_format_bptc_rgb_float_pack_rgba_float(
         blocks[0], 16, signed_source, 4 * 4 * sizeof(float), 4, 4);
      util_format_bptc_rgb_ufloat_pack_rgba_float(
         blocks[1], 16, unsigned_source, 4 * 4 * sizeof(float), 4, 4);
      util_format_bptc_rgba_unorm_pack_rgba_8unorm(
         blocks[2], 16, unorm_source, 4 * 4, 4, 4);
      util_format_bptc_srgba_pack_rgba_8unorm(
         blocks[3], 16, unorm_source, 4 * 4, 4, 4);
      struct pipe_shader_state signed_state = {0};
      signed_state.type = PIPE_SHADER_IR_NIR;
      signed_state.ir.nir = make_texture_transform_fs(0.5f, 0.5f);
      signed_state.report_compile_error = true;
      void *signed_fs = ctx->create_fs_state(ctx, &signed_state);
      struct pipe_shader_state plain_state = {0};
      plain_state.type = PIPE_SHADER_IR_NIR;
      plain_state.ir.nir = make_texture_fs(false, 0.5f, 0.5f, -1.0f);
      plain_state.report_compile_error = true;
      void *plain_fs = ctx->create_fs_state(ctx, &plain_state);
      struct pipe_resource *textures[4] = {0};
      struct pipe_sampler_view *views[4] = {0};
      texture_ok = signed_fs && plain_fs;
      for (unsigned i = 0; texture_ok && i < 4; ++i) {
         float decoded[4 * 4 * 4];
         if (i == 0)
            util_format_bptc_rgb_float_unpack_rgba_float(
               decoded, 4 * 4 * sizeof(float), blocks[i], 16, 4, 4);
         else if (i == 1)
            util_format_bptc_rgb_ufloat_unpack_rgba_float(
               decoded, 4 * 4 * sizeof(float), blocks[i], 16, 4, 4);
         else if (i == 2)
            util_format_bptc_rgba_unorm_unpack_rgba_float(
               decoded, 4 * 4 * sizeof(float), blocks[i], 16, 4, 4);
         else
            util_format_bptc_srgba_unpack_rgba_float(
               decoded, 4 * 4 * sizeof(float), blocks[i], 16, 4, 4);
         const float *sample = decoded + (2 * 4 + 2) * 4;
         uint8_t expected[4];
         for (unsigned channel = 0; channel < 4; ++channel) {
            float value = sample[channel];
            if (i == 0)
               value = value * 0.5f + 0.5f;
            value = CLAMP(value, 0.0f, 1.0f);
            expected[channel] = (uint8_t)lroundf(value * 255.0f);
         }
         struct pipe_resource templ = sample_templ;
         templ.format = bptc_formats[i];
         templ.width0 = 4;
         templ.height0 = 4;
         templ.last_level = 0;
         textures[i] = screen->resource_create(screen, &templ);
         const struct pipe_box box = {
            .x = 0, .y = 0, .z = 0, .width = 4, .height = 4, .depth = 1,
         };
         if (textures[i])
            ctx->texture_subdata(ctx, textures[i], 0, PIPE_MAP_WRITE,
                                 &box, blocks[i], 16, 16);
         struct pipe_sampler_view view = view_templ;
         view.format = bptc_formats[i];
         view.u.tex.last_level = 0;
         views[i] = textures[i]
            ? ctx->create_sampler_view(ctx, textures[i], &view) : NULL;
         texture_ok = textures[i] && views[i];
         if (texture_ok) {
            struct pipe_sampler_view *bound[1] = {views[i]};
            ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0,
                                   bound);
            gex_bind_fs(ctx, i == 0 ? signed_fs : plain_fs);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            texture_ok = check_asymmetric_color_tolerance(
               ctx, resource, false, false, expected, i < 2 ? 2 : 1,
               labels[i]);
            if (!texture_ok)
               fprintf(stderr,
                       "FAIL gex-wgpu BPTC reference format=%u expected="
                       "%u,%u,%u,%u decoded=%f,%f,%f,%f\n",
                       bptc_formats[i], expected[0], expected[1], expected[2],
                       expected[3], sample[0], sample[1], sample[2], sample[3]);
         } else {
            fprintf(stderr, "FAIL gex-wgpu BPTC format %u setup\n",
                    bptc_formats[i]);
         }
      }
      gex_bind_fs(ctx, texture_fs);
      if (sample_view) {
         struct pipe_sampler_view *bound[1] = {sample_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, bound);
      }
      free(signed_state.error_message);
      free(plain_state.error_message);
      if (signed_fs)
         gex_delete_fs(ctx, signed_fs);
      if (plain_fs)
         gex_delete_fs(ctx, plain_fs);
      for (unsigned i = 0; i < 4; ++i) {
         if (views[i])
            pipe_sampler_view_reference(&views[i], NULL);
         if (textures[i])
            pipe_resource_reference(&textures[i], NULL);
      }
   }
   struct pipe_shader_state bc_mip_state = {0};
   bc_mip_state.type = PIPE_SHADER_IR_NIR;
   bc_mip_state.ir.nir = make_texture_fs(false, 0.5f, 0.5f, 1.0f);
   bc_mip_state.report_compile_error = true;
   void *bc_mip_fs = ctx->create_fs_state(ctx, &bc_mip_state);
   struct pipe_resource bc_mip_templ = sample_templ;
   bc_mip_templ.format = PIPE_FORMAT_DXT1_RGBA;
   bc_mip_templ.width0 = 4;
   bc_mip_templ.height0 = 4;
   bc_mip_templ.last_level = 1;
   struct pipe_resource *bc_mip_texture = webgpu_boundary
      ? screen->resource_create(screen, &bc_mip_templ) : NULL;
   const uint8_t bc1_green[8] = {
      0xe0, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const struct pipe_box bc_level0_box = {
      .x = 0, .y = 0, .z = 0, .width = 4, .height = 4, .depth = 1,
   };
   const struct pipe_box bc_level1_edge_box = {
      .x = 0, .y = 0, .z = 0, .width = 2, .height = 2, .depth = 1,
   };
   if (bc_mip_texture) {
      ctx->texture_subdata(ctx, bc_mip_texture, 0, PIPE_MAP_WRITE,
                           &bc_level0_box, bc1_gray, 8, 8);
      ctx->texture_subdata(ctx, bc_mip_texture, 1, PIPE_MAP_WRITE,
                           &bc_level1_edge_box, bc1_green, 8, 8);
   }
   struct pipe_sampler_view bc_mip_view_templ = view_templ;
   bc_mip_view_templ.format = PIPE_FORMAT_DXT1_RGBA;
   bc_mip_view_templ.u.tex.last_level = 1;
   struct pipe_sampler_view *bc_mip_view = bc_mip_texture
      ? ctx->create_sampler_view(ctx, bc_mip_texture, &bc_mip_view_templ)
      : NULL;
   struct pipe_sampler_state bc_mip_sampler_state = sampler_state;
   bc_mip_sampler_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
   bc_mip_sampler_state.max_lod = 1.0f;
   void *bc_mip_sampler =
      ctx->create_sampler_state(ctx, &bc_mip_sampler_state);
   /* wasm softpipe's TGSI interpreter has no valid dispatch entry for
    * compressed explicit-LOD sampling and traps in exec_tex.  The four
    * base-level decode cases above remain differential; keep this mip-edge
    * copy and sample as a real-WebGPU boundary test. */
   if (texture_ok && webgpu_boundary)
      texture_ok = bc_mip_texture && bc_mip_view && bc_mip_fs && bc_mip_sampler;
   if (texture_ok && webgpu_boundary) {
      const uint8_t green[4] = {0, 255, 0, 255};
      struct pipe_sampler_view *views[1] = {bc_mip_view};
      void *samplers[1] = {bc_mip_sampler};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      pipe_resource_reference(&bc_mip_texture, NULL);
      gex_bind_fs(ctx, bc_mip_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, green,
         "texture-bc1-edge-block-explicit-mip-one");
   } else if (webgpu_boundary && bc_mip_state.error_message) {
      fprintf(stderr, "FAIL gex-wgpu BC mip shader=%s\n",
              bc_mip_state.error_message);
   }
   free(bc_mip_state.error_message);
   if (bc_mip_view)
      pipe_sampler_view_reference(&bc_mip_view, NULL);
   if (bc_mip_texture)
      pipe_resource_reference(&bc_mip_texture, NULL);
   gex_bind_fs(ctx, texture_fs);
   if (sample_sampler) {
      void *samplers[1] = {sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
   if (bc_mip_sampler)
      gex_delete_sampler(ctx, bc_mip_sampler);
   if (bc_mip_fs)
      gex_delete_fs(ctx, bc_mip_fs);
   if (sample_view) {
      struct pipe_sampler_view *views[1] = {sample_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(bc_formats); ++i) {
      if (bc_views[i])
         pipe_sampler_view_reference(&bc_views[i], NULL);
      if (bc_textures[i])
         pipe_resource_reference(&bc_textures[i], NULL);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(rgtc_formats); ++i) {
      if (rgtc_views[i])
         pipe_sampler_view_reference(&rgtc_views[i], NULL);
      if (rgtc_textures[i])
         pipe_resource_reference(&rgtc_textures[i], NULL);
   }
   if (texture_ok && webgpu_boundary) {
      const enum pipe_texture_target targets[3] = {
         PIPE_TEXTURE_2D_ARRAY, PIPE_TEXTURE_CUBE, PIPE_TEXTURE_CUBE_ARRAY,
      };
      const unsigned layers[3] = {2, 6, 12};
      const unsigned green_layers[3] = {1, 5, 10};
      const unsigned first_view_layers[3] = {0, 0, 6};
      const unsigned last_view_layers[3] = {1, 5, 11};
      const char *labels[3] = {
         "texture-bc1-2d-array-layer-one",
         "texture-bc1-cube-negative-z-face",
         "texture-bc1-cube-array-rebased-second-cube-positive-z-face",
      };
      const uint8_t bc1_red[8] = {
         0x00, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      };
      struct pipe_resource *layered_textures[3] = {0};
      struct pipe_sampler_view *layered_views[3] = {0};
      void *layered_fs[3] = {0};
      struct pipe_shader_state layered_states[3] = {{0}};
      for (unsigned test = 0; texture_ok && test < 3; ++test) {
         struct pipe_resource templ = sample_templ;
         templ.target = targets[test];
         templ.format = PIPE_FORMAT_DXT1_RGBA;
         templ.width0 = 4;
         templ.height0 = 4;
         templ.array_size = layers[test];
         templ.last_level = 0;
         layered_textures[test] = screen->resource_create(screen, &templ);
         uint8_t blocks[12 * 8];
         for (unsigned layer = 0; layer < layers[test]; ++layer)
            memcpy(blocks + layer * 8,
                   layer == green_layers[test] ? bc1_green : bc1_red, 8);
         const struct pipe_box box = {
            .x = 0, .y = 0, .z = 0, .width = 4, .height = 4,
            .depth = (int)layers[test],
         };
         if (layered_textures[test])
            ctx->texture_subdata(ctx, layered_textures[test], 0,
                                 PIPE_MAP_WRITE, &box, blocks, 8, 8);
         struct pipe_sampler_view layered_view_templ = view_templ;
         layered_view_templ.target = targets[test];
         layered_view_templ.format = PIPE_FORMAT_DXT1_RGBA;
         layered_view_templ.u.tex.last_level = 0;
         layered_view_templ.u.tex.first_layer = first_view_layers[test];
         layered_view_templ.u.tex.last_layer = last_view_layers[test];
         layered_views[test] = layered_textures[test]
            ? ctx->create_sampler_view(ctx, layered_textures[test],
                                       &layered_view_templ) : NULL;
         layered_states[test].type = PIPE_SHADER_IR_NIR;
         layered_states[test].ir.nir = test == 0
            ? make_texture_array_fs()
            : test == 1 ? make_texture_cube_fs()
                        : make_texture_cube_array_fs(false, 0.0f);
         layered_states[test].report_compile_error = true;
         layered_fs[test] =
            ctx->create_fs_state(ctx, &layered_states[test]);
         texture_ok = layered_textures[test] && layered_views[test] &&
                      layered_fs[test];
         if (texture_ok) {
            const uint8_t green[4] = {0, 255, 0, 255};
            struct pipe_sampler_view *views[1] = {layered_views[test]};
            ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
            gex_bind_fs(ctx, layered_fs[test]);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            texture_ok = check_asymmetric_color(
               ctx, resource, false, false, green, labels[test]);
         } else {
            fprintf(stderr,
                    "FAIL gex-wgpu compressed layered target %u setup: %s\n",
                    targets[test], layered_states[test].error_message
                       ? layered_states[test].error_message : "unknown");
         }
      }
      gex_bind_fs(ctx, texture_fs);
      if (sample_view) {
         struct pipe_sampler_view *views[1] = {sample_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      }
      for (unsigned test = 0; test < 3; ++test) {
         free(layered_states[test].error_message);
         if (layered_fs[test])
            gex_delete_fs(ctx, layered_fs[test]);
         if (layered_views[test])
            pipe_sampler_view_reference(&layered_views[test], NULL);
         if (layered_textures[test])
            pipe_resource_reference(&layered_textures[test], NULL);
      }
   }
   struct pipe_resource *texture_constant_resource =
      screen->resource_create(screen, &constant_templ);
   const float texture_constant_red[4] = {1.0f, 0.0f, 0.0f, 0.0f};
   if (texture_constant_resource)
      ctx->buffer_subdata(ctx, texture_constant_resource, PIPE_MAP_WRITE, 0,
                          sizeof(texture_constant_red), texture_constant_red);
   struct pipe_shader_state texture_constant_state = {0};
   texture_constant_state.type = PIPE_SHADER_IR_NIR;
   texture_constant_state.ir.nir = make_texture_constant_fs();
   texture_constant_state.report_compile_error = true;
   void *texture_constant_fs =
      ctx->create_fs_state(ctx, &texture_constant_state);
   if (texture_ok)
      texture_ok = texture_constant_resource && texture_constant_fs;
   if (texture_ok) {
      const uint8_t yellow[4] = {255, 255, 0, 255};
      struct pipe_constant_buffer texture_constant = {
         .buffer = texture_constant_resource,
         .buffer_size = sizeof(texture_constant_red),
      };
      gex_bind_fs(ctx, texture_constant_fs);
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0,
                               &texture_constant);
      pipe_resource_reference(&texture_constant_resource, NULL);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, yellow,
         "texture-fragment-constant-descriptor-coexistence-lifetime");
      ctx->set_constant_buffer(ctx, MESA_SHADER_FRAGMENT, 0, NULL);
   }
   if (texture_constant_resource)
      pipe_resource_reference(&texture_constant_resource, NULL);
   if (!texture_ok && texture_constant_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu texture/constant shader=%s\n",
              texture_constant_state.error_message);
   free(texture_constant_state.error_message);
   if (texture_constant_fs) {
      gex_bind_fs(ctx, texture_fs);
      gex_delete_fs(ctx, texture_constant_fs);
   }

   struct pipe_resource two_texture_templ = sample_templ;
   two_texture_templ.width0 = 1;
   two_texture_templ.height0 = 1;
   two_texture_templ.last_level = 0;
   struct pipe_resource *two_textures[2] = {
      screen->resource_create(screen, &two_texture_templ),
      screen->resource_create(screen, &two_texture_templ),
   };
   const uint8_t two_texels[2][4] = {
      {255, 0, 0, 0},
      {0, 255, 0, 255},
   };
   struct pipe_sampler_view *two_views[2] = {NULL, NULL};
   struct pipe_sampler_view two_view_templ = view_templ;
   two_view_templ.u.tex.last_level = 0;
   for (unsigned i = 0; i < 2; ++i) {
      if (two_textures[i]) {
         ctx->texture_subdata(ctx, two_textures[i], 0, PIPE_MAP_WRITE,
                              &mip_box, two_texels[i], 4, 4);
         two_views[i] = ctx->create_sampler_view(
            ctx, two_textures[i], &two_view_templ);
      }
   }
   struct pipe_sampler_view *two_swizzle_view = NULL;
   if (two_textures[0]) {
      struct pipe_sampler_view swizzle_templ = two_view_templ;
      swizzle_templ.swizzle_r = PIPE_SWIZZLE_Z;
      swizzle_templ.swizzle_b = PIPE_SWIZZLE_X;
      two_swizzle_view = ctx->create_sampler_view(
         ctx, two_textures[0], &swizzle_templ);
   }
   struct pipe_shader_state two_texture_state = {0};
   two_texture_state.type = PIPE_SHADER_IR_NIR;
   two_texture_state.ir.nir = make_two_texture_fs();
   two_texture_state.report_compile_error = true;
   void *two_texture_fs = ctx->create_fs_state(ctx, &two_texture_state);
   if (texture_ok)
      texture_ok = two_textures[0] && two_textures[1] &&
                   two_views[0] && two_views[1] && two_texture_fs;
   if (texture_ok) {
      const uint8_t yellow[4] = {255, 255, 0, 255};
      void *two_samplers[2] = {sample_sampler, sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 2,
                               two_samplers);
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 2, 0, two_views);
      for (unsigned i = 0; i < 2; ++i)
         pipe_resource_reference(&two_textures[i], NULL);
      gex_bind_fs(ctx, two_texture_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, yellow,
         "two-texture-sampler-descriptor-lifetime");
      if (texture_ok && two_swizzle_view) {
         const uint8_t cyan[4] = {0, 255, 255, 255};
         struct pipe_sampler_view *swizzled[1] = {two_swizzle_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0,
                                swizzled);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, cyan,
            "two-texture-slot-specific-swizzle-lifetime");
      } else if (texture_ok) {
         texture_ok = false;
      }
   }
   if (!texture_ok && two_texture_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu two-texture shader=%s\n",
              two_texture_state.error_message);
   free(two_texture_state.error_message);
   gex_bind_fs(ctx, NULL);
   ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 1, 0, 1, NULL);
   if (sample_view) {
      struct pipe_sampler_view *restored_view[1] = {sample_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0,
                             restored_view);
   }
   /* softpipe's polygon-stipple wrapper requires sampler updates to begin at
    * slot zero.  Restore slot zero while explicitly clearing slot one. */
   void *restored_samplers[2] = {sample_sampler, NULL};
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 2,
                            restored_samplers);
   for (unsigned i = 0; i < 2; ++i) {
      if (two_views[i])
         pipe_sampler_view_reference(&two_views[i], NULL);
      if (two_textures[i])
         pipe_resource_reference(&two_textures[i], NULL);
   }
   if (two_swizzle_view)
      pipe_sampler_view_reference(&two_swizzle_view, NULL);
   if (two_texture_fs)
      gex_delete_fs(ctx, two_texture_fs);

   const enum pipe_format channel_formats[6] = {
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8G8_UNORM,
      PIPE_FORMAT_B8G8R8A8_UNORM,
      PIPE_FORMAT_R16_FLOAT,
      PIPE_FORMAT_R16G16_FLOAT,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
   };
   const unsigned channel_strides[6] = {1, 2, 4, 2, 4, 8};
   const uint8_t channel_texels[6][8] = {
      {128},
      {64, 192},
      {10, 20, 30, 255},
      {0x00, 0x38},
      {0x00, 0x34, 0x00, 0x3a},
      {0x00, 0x34, 0x00, 0x38, 0x00, 0x3a, 0x00, 0x3c},
   };
   const uint8_t channel_expected[6][4] = {
      {128, 0, 0, 255},
      {64, 192, 0, 255},
      {30, 20, 10, 255},
      {128, 0, 0, 255},
      {64, 191, 0, 255},
      {64, 128, 191, 255},
   };
   const char *channel_labels[6] = {
      "texture-r8-channel-defaults-lifetime",
      "texture-rg8-channel-defaults-lifetime",
      "texture-bgra8-byte-order-lifetime",
      "texture-r16f-channel-defaults-lifetime",
      "texture-rg16f-channel-defaults-lifetime",
      "texture-rgba16f-sampling-lifetime",
   };
   for (unsigned i = 0; texture_ok && i < 6; ++i) {
      struct pipe_resource channel_templ = two_texture_templ;
      channel_templ.format = channel_formats[i];
      struct pipe_resource *channel_texture =
         screen->resource_create(screen, &channel_templ);
      struct pipe_sampler_view channel_view_templ = two_view_templ;
      channel_view_templ.format = channel_formats[i];
      struct pipe_sampler_view *channel_view = channel_texture
         ? ctx->create_sampler_view(ctx, channel_texture, &channel_view_templ)
         : NULL;
      texture_ok = channel_texture && channel_view;
      if (texture_ok) {
         ctx->texture_subdata(ctx, channel_texture, 0, PIPE_MAP_WRITE,
                              &mip_box, channel_texels[i], channel_strides[i],
                              channel_strides[i]);
         struct pipe_sampler_view *views[1] = {channel_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
         pipe_resource_reference(&channel_texture, NULL);
         gex_bind_fs(ctx, texture_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, channel_expected[i],
            channel_labels[i]);
      }
      if (channel_view)
         pipe_sampler_view_reference(&channel_view, NULL);
      if (channel_texture)
         pipe_resource_reference(&channel_texture, NULL);
   }
   struct pipe_shader_state uint_state = {0};
   uint_state.type = PIPE_SHADER_IR_NIR;
   uint_state.ir.nir = make_integer_texel_fetch_fs(false);
   uint_state.report_compile_error = true;
   void *uint_fs = ctx->create_fs_state(ctx, &uint_state);
   if (texture_ok)
      texture_ok = uint_fs != NULL;
   const enum pipe_format uint_formats[3] = {
      PIPE_FORMAT_R8_UINT,
      PIPE_FORMAT_R8G8_UINT,
      PIPE_FORMAT_R8G8B8A8_UINT,
   };
   const unsigned uint_strides[3] = {1, 2, 4};
   const uint8_t uint_texels[3][4] = {
      {64},
      {64, 128},
      {64, 128, 192, 255},
   };
   const uint8_t uint_expected[3][4] = {
      {64, 0, 0, 255},
      {64, 128, 0, 255},
      {64, 128, 192, 255},
   };
   const char *uint_labels[3] = {
      "texture-r8ui-texel-fetch-defaults-lifetime",
      "texture-rg8ui-texel-fetch-defaults-lifetime",
      "texture-rgba8ui-texel-fetch-exact-lifetime",
   };
   for (unsigned i = 0; texture_ok && i < 3; ++i) {
      struct pipe_resource uint_templ = two_texture_templ;
      uint_templ.format = uint_formats[i];
      struct pipe_resource *uint_texture =
         screen->resource_create(screen, &uint_templ);
      struct pipe_sampler_view uint_view_templ = two_view_templ;
      uint_view_templ.format = uint_formats[i];
      struct pipe_sampler_view *uint_view = uint_texture
         ? ctx->create_sampler_view(ctx, uint_texture, &uint_view_templ) : NULL;
      texture_ok = uint_texture && uint_view;
      if (texture_ok) {
         struct pipe_sampler_view *views[1] = {uint_view};
         ctx->texture_subdata(ctx, uint_texture, 0, PIPE_MAP_WRITE,
                              &mip_box, uint_texels[i], uint_strides[i],
                              uint_strides[i]);
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
         pipe_resource_reference(&uint_texture, NULL);
         gex_bind_fs(ctx, uint_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, uint_expected[i], uint_labels[i]);
      }
      if (uint_view)
         pipe_sampler_view_reference(&uint_view, NULL);
      if (uint_texture)
         pipe_resource_reference(&uint_texture, NULL);
   }
   if (!texture_ok && uint_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu uint-texel-fetch shader=%s\n",
              uint_state.error_message);
   free(uint_state.error_message);
   if (uint_fs) {
      gex_bind_fs(ctx, texture_fs);
      gex_delete_fs(ctx, uint_fs);
   }
   struct pipe_shader_state sint_state = {0};
   sint_state.type = PIPE_SHADER_IR_NIR;
   sint_state.ir.nir = make_integer_texel_fetch_fs(true);
   sint_state.report_compile_error = true;
   void *sint_fs = ctx->create_fs_state(ctx, &sint_state);
   if (texture_ok)
      texture_ok = sint_fs != NULL;
   const enum pipe_format sint_formats[3] = {
      PIPE_FORMAT_R8_SINT,
      PIPE_FORMAT_R8G8_SINT,
      PIPE_FORMAT_R8G8B8A8_SINT,
   };
   const unsigned sint_strides[3] = {1, 2, 4};
   const int8_t sint_texels[3][4] = {
      {-64},
      {-64, 64},
      {-64, 0, 64, 127},
   };
   const uint8_t sint_expected[3][4] = {
      {64, 128, 128, 129},
      {64, 192, 128, 129},
      {64, 128, 192, 255},
   };
   const char *sint_labels[3] = {
      "texture-r8i-texel-fetch-defaults-lifetime",
      "texture-rg8i-texel-fetch-defaults-lifetime",
      "texture-rgba8i-texel-fetch-exact-lifetime",
   };
   for (unsigned i = 0; texture_ok && i < 3; ++i) {
      struct pipe_resource sint_templ = two_texture_templ;
      sint_templ.format = sint_formats[i];
      struct pipe_resource *sint_texture =
         screen->resource_create(screen, &sint_templ);
      struct pipe_sampler_view sint_view_templ = two_view_templ;
      sint_view_templ.format = sint_formats[i];
      struct pipe_sampler_view *sint_view = sint_texture
         ? ctx->create_sampler_view(ctx, sint_texture, &sint_view_templ) : NULL;
      texture_ok = sint_texture && sint_view;
      if (texture_ok) {
         struct pipe_sampler_view *views[1] = {sint_view};
         ctx->texture_subdata(ctx, sint_texture, 0, PIPE_MAP_WRITE,
                              &mip_box, sint_texels[i], sint_strides[i],
                              sint_strides[i]);
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
         pipe_resource_reference(&sint_texture, NULL);
         gex_bind_fs(ctx, sint_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, sint_expected[i], sint_labels[i]);
      }
      if (sint_view)
         pipe_sampler_view_reference(&sint_view, NULL);
      if (sint_texture)
         pipe_resource_reference(&sint_texture, NULL);
   }
   if (!texture_ok && sint_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu sint-texel-fetch shader=%s\n",
              sint_state.error_message);
   free(sint_state.error_message);
   if (sint_fs) {
      gex_bind_fs(ctx, texture_fs);
      gex_delete_fs(ctx, sint_fs);
   }
   struct pipe_resource uint_sample_templ = sample_templ;
   uint_sample_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   uint_sample_templ.last_level = 1;
   struct pipe_resource *uint_sample_texture =
      screen->resource_create(screen, &uint_sample_templ);
   const uint8_t uint_sample_texels[16] = {
      255, 0, 0, 255, 0, 255, 0, 255,
      0, 0, 255, 255, 255, 255, 255, 255,
   };
   struct pipe_sampler_view uint_sample_view_templ = view_templ;
   uint_sample_view_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   uint_sample_view_templ.u.tex.last_level = 0;
   struct pipe_sampler_view *uint_sample_view = uint_sample_texture
      ? ctx->create_sampler_view(ctx, uint_sample_texture,
                                 &uint_sample_view_templ) : NULL;
   struct pipe_sampler_view uint_sample_base_view_templ =
      uint_sample_view_templ;
   uint_sample_base_view_templ.u.tex.first_level = 1;
   uint_sample_base_view_templ.u.tex.last_level = 1;
   struct pipe_sampler_view *uint_sample_base_view = uint_sample_texture
      ? ctx->create_sampler_view(ctx, uint_sample_texture,
                                 &uint_sample_base_view_templ) : NULL;
   struct pipe_sampler_view uint_sample_mip_view_templ =
      uint_sample_view_templ;
   uint_sample_mip_view_templ.u.tex.last_level = 1;
   struct pipe_sampler_view *uint_sample_mip_view = uint_sample_texture
      ? ctx->create_sampler_view(ctx, uint_sample_texture,
                                 &uint_sample_mip_view_templ) : NULL;
   struct pipe_sampler_view uint_sample_swizzle_view_templ =
      uint_sample_view_templ;
   uint_sample_swizzle_view_templ.swizzle_r = PIPE_SWIZZLE_Z;
   uint_sample_swizzle_view_templ.swizzle_g = PIPE_SWIZZLE_Y;
   uint_sample_swizzle_view_templ.swizzle_b = PIPE_SWIZZLE_X;
   uint_sample_swizzle_view_templ.swizzle_a = PIPE_SWIZZLE_1;
   struct pipe_sampler_view *uint_sample_swizzle_view = uint_sample_texture
      ? ctx->create_sampler_view(ctx, uint_sample_texture,
                                 &uint_sample_swizzle_view_templ) : NULL;
   struct pipe_shader_state uint_sample_states[9] = {{0}};
   uint_sample_states[0].type = PIPE_SHADER_IR_NIR;
   uint_sample_states[0].ir.nir =
      make_integer_texture_lod_fs(false, false, false, 0.75f, 0.25f, 0.0f);
   uint_sample_states[0].report_compile_error = true;
   uint_sample_states[1].type = PIPE_SHADER_IR_NIR;
   uint_sample_states[1].ir.nir =
      make_integer_texture_lod_fs(false, false, false, -0.25f, 0.25f, 0.0f);
   uint_sample_states[1].report_compile_error = true;
   uint_sample_states[2].type = PIPE_SHADER_IR_NIR;
   uint_sample_states[2].ir.nir =
      make_integer_texture_lod_fs(false, false, false, 0.25f, 1.0f, 0.0f);
   uint_sample_states[2].report_compile_error = true;
   uint_sample_states[3].type = PIPE_SHADER_IR_NIR;
   uint_sample_states[3].ir.nir =
      make_integer_texture_lod_fs(false, true, false, 0.75f, 0.25f, 0.0f);
   uint_sample_states[3].report_compile_error = true;
   uint_sample_states[4].type = PIPE_SHADER_IR_NIR;
   uint_sample_states[4].ir.nir =
      make_integer_texture_lod_fs(false, false, false, -0.75f, 0.25f, 0.0f);
   uint_sample_states[4].report_compile_error = true;
   const float mip_lods[3] = {0.49f, 0.51f, 5.0f};
   for (unsigned i = 0; i < 3; ++i) {
      uint_sample_states[5 + i].type = PIPE_SHADER_IR_NIR;
      uint_sample_states[5 + i].ir.nir =
         make_integer_texture_lod_fs(
            false, false, false, 0.25f, 0.25f, mip_lods[i]);
      uint_sample_states[5 + i].report_compile_error = true;
   }
   uint_sample_states[8].type = PIPE_SHADER_IR_NIR;
   uint_sample_states[8].ir.nir =
      make_integer_texture_lod_fs(false, true, true, 0.0f, 0.0f, 0.0f);
   uint_sample_states[8].report_compile_error = true;
   void *uint_sample_fs[9] = {
      ctx->create_fs_state(ctx, &uint_sample_states[0]),
      ctx->create_fs_state(ctx, &uint_sample_states[1]),
      ctx->create_fs_state(ctx, &uint_sample_states[2]),
      ctx->create_fs_state(ctx, &uint_sample_states[3]),
      ctx->create_fs_state(ctx, &uint_sample_states[4]),
      ctx->create_fs_state(ctx, &uint_sample_states[5]),
      ctx->create_fs_state(ctx, &uint_sample_states[6]),
      ctx->create_fs_state(ctx, &uint_sample_states[7]),
      ctx->create_fs_state(ctx, &uint_sample_states[8]),
   };
   struct pipe_shader_state uint_gradient_vs_state = {0};
   uint_gradient_vs_state.type = PIPE_SHADER_IR_NIR;
   uint_gradient_vs_state.ir.nir = make_gradient_texture_vs(16.0f);
   uint_gradient_vs_state.report_compile_error = true;
   void *uint_gradient_vs =
      ctx->create_vs_state(ctx, &uint_gradient_vs_state);
   if (texture_ok)
      texture_ok = uint_sample_texture && uint_sample_view &&
                   uint_sample_base_view && uint_sample_mip_view &&
                   uint_sample_swizzle_view &&
                   uint_sample_fs[0] && uint_sample_fs[1] &&
                   uint_sample_fs[2] && uint_sample_fs[3] &&
                   uint_sample_fs[4] && uint_sample_fs[5] &&
                   uint_sample_fs[6] && uint_sample_fs[7] &&
                   uint_sample_fs[8] && uint_gradient_vs;
   if (texture_ok) {
      const uint8_t expected[4][4] = {
         {0, 255, 0, 255},
         {255, 0, 0, 255},
         {0, 0, 255, 255},
         {0, 255, 0, 255},
      };
      const char *labels[4] = {
         "texture-rgba8ui-normalized-lod-nearest",
         "texture-rgba8ui-normalized-lod-clamp-negative",
         "texture-rgba8ui-normalized-lod-clamp-one",
         "texture-rgba8ui-normalized-implicit-single-level",
      };
      ctx->texture_subdata(ctx, uint_sample_texture, 0, PIPE_MAP_WRITE,
                           &sample_box, uint_sample_texels, 8, 16);
      const uint8_t base_level_texel[4] = {0, 255, 255, 255};
      ctx->texture_subdata(ctx, uint_sample_texture, 1, PIPE_MAP_WRITE,
                           &mip_box, base_level_texel, 4, 4);
      struct pipe_sampler_view *views[1] = {uint_sample_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&uint_sample_texture, NULL);
      for (unsigned i = 0; texture_ok && i < 4; ++i) {
         gex_bind_fs(ctx, uint_sample_fs[i]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, expected[i], labels[i]);
      }
      if (texture_ok) {
         const uint8_t cyan[4] = {0, 255, 255, 255};
         struct pipe_sampler_view *base_views[1] = {uint_sample_base_view};
         ctx->set_sampler_views(
            ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, base_views);
         gex_bind_fs(ctx, uint_sample_fs[0]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, cyan,
            "texture-rgba8ui-normalized-nonzero-base-level");
      }
      if (texture_ok) {
         struct pipe_sampler_state uint_mip_state = sampler_state;
         uint_mip_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
         uint_mip_state.max_lod = 1.0f;
         void *uint_mip_sampler =
            ctx->create_sampler_state(ctx, &uint_mip_state);
         texture_ok = uint_mip_sampler != NULL;
         if (texture_ok) {
            const uint8_t mip_expected[3][4] = {
               {255, 0, 0, 255},
               {0, 255, 255, 255},
               {0, 255, 255, 255},
            };
            const char *mip_labels[3] = {
               "texture-rgba8ui-explicit-lod-nearest-round-down",
               "texture-rgba8ui-explicit-lod-nearest-round-up",
               "texture-rgba8ui-explicit-lod-view-clamp",
            };
            void *mip_samplers[1] = {uint_mip_sampler};
            struct pipe_sampler_view *mip_views[1] = {uint_sample_mip_view};
            gex_bind_samplers(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, mip_samplers);
            ctx->set_sampler_views(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, mip_views);
            for (unsigned i = 0; texture_ok && i < 3; ++i) {
               gex_bind_fs(ctx, uint_sample_fs[5 + i]);
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               texture_ok = check_asymmetric_color(
                  ctx, resource, false, false, mip_expected[i], mip_labels[i]);
            }
         }
         if (uint_mip_sampler)
            gex_delete_sampler(ctx, uint_mip_sampler);
         if (texture_ok) {
            struct pipe_sampler_state bias_lod_state = sampler_state;
            bias_lod_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
            bias_lod_state.max_lod = 1.0f;
            bias_lod_state.lod_bias = 1.0f;
            void *bias_lod_sampler =
               ctx->create_sampler_state(ctx, &bias_lod_state);
            texture_ok = bias_lod_sampler != NULL;
            if (texture_ok) {
               void *bias_lod_samplers[1] = {bias_lod_sampler};
               struct pipe_sampler_view *mip_views[1] = {
                  uint_sample_mip_view,
               };
               gex_bind_samplers(
                  ctx, MESA_SHADER_FRAGMENT, 0, 1, bias_lod_samplers);
               ctx->set_sampler_views(
                  ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, mip_views);
               gex_bind_fs(ctx, uint_sample_fs[0]);
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               const uint8_t cyan[4] = {0, 255, 255, 255};
               texture_ok = check_asymmetric_color(
                  ctx, resource, false, false, cyan,
                  "texture-rgba8ui-explicit-lod-positive-sampler-bias");
            }
            if (bias_lod_sampler)
               gex_delete_sampler(ctx, bias_lod_sampler);
         }
         if (texture_ok) {
            struct pipe_sampler_state min_lod_state = sampler_state;
            min_lod_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
            min_lod_state.min_lod = 0.6f;
            min_lod_state.max_lod = 1.0f;
            void *min_lod_sampler =
               ctx->create_sampler_state(ctx, &min_lod_state);
            texture_ok = min_lod_sampler != NULL;
            if (texture_ok) {
               void *min_lod_samplers[1] = {min_lod_sampler};
               struct pipe_sampler_view *mip_views[1] = {uint_sample_mip_view};
               gex_bind_samplers(
                  ctx, MESA_SHADER_FRAGMENT, 0, 1, min_lod_samplers);
               ctx->set_sampler_views(
                  ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, mip_views);
               gex_bind_fs(ctx, uint_sample_fs[0]);
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               const uint8_t cyan[4] = {0, 255, 255, 255};
               texture_ok = check_asymmetric_color(
                  ctx, resource, false, false, cyan,
                  "texture-rgba8ui-explicit-lod-min-clamp");
            }
            if (min_lod_sampler)
               gex_delete_sampler(ctx, min_lod_sampler);
         }
         if (texture_ok) {
            struct pipe_sampler_state max_lod_state = sampler_state;
            max_lod_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
            max_lod_state.max_lod = 0.4f;
            void *max_lod_sampler =
               ctx->create_sampler_state(ctx, &max_lod_state);
            texture_ok = max_lod_sampler != NULL;
            if (texture_ok) {
               void *max_lod_samplers[1] = {max_lod_sampler};
               struct pipe_sampler_view *mip_views[1] = {uint_sample_mip_view};
               gex_bind_samplers(
                  ctx, MESA_SHADER_FRAGMENT, 0, 1, max_lod_samplers);
               ctx->set_sampler_views(
                  ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, mip_views);
               gex_bind_fs(ctx, uint_sample_fs[7]);
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               const uint8_t red[4] = {255, 0, 0, 255};
               texture_ok = check_asymmetric_color(
                  ctx, resource, false, false, red,
                  "texture-rgba8ui-explicit-lod-max-clamp");
            }
            if (max_lod_sampler)
               gex_delete_sampler(ctx, max_lod_sampler);
         }
         if (texture_ok) {
            struct pipe_sampler_state implicit_lod_state = sampler_state;
            implicit_lod_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
            implicit_lod_state.max_lod = 1.0f;
            void *implicit_lod_sampler =
               ctx->create_sampler_state(ctx, &implicit_lod_state);
            texture_ok = implicit_lod_sampler != NULL;
            if (texture_ok) {
               void *implicit_samplers[1] = {implicit_lod_sampler};
               struct pipe_sampler_view *mip_views[1] = {uint_sample_mip_view};
               gex_bind_samplers(
                  ctx, MESA_SHADER_FRAGMENT, 0, 1, implicit_samplers);
               ctx->set_sampler_views(
                  ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, mip_views);
               gex_bind_vs(ctx, uint_gradient_vs);
               gex_bind_fs(ctx, uint_sample_fs[8]);
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               const uint8_t cyan[4] = {0, 255, 255, 255};
               texture_ok = check_uniform(
                  ctx, resource, cyan,
                  "texture-rgba8ui-implicit-lod-derivative-level-one");
               gex_bind_vs(ctx, vs);
            }
            if (implicit_lod_sampler)
               gex_delete_sampler(ctx, implicit_lod_sampler);
         }
         if (sample_sampler) {
            void *default_samplers[1] = {sample_sampler};
            gex_bind_samplers(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, default_samplers);
         }
      }
      if (texture_ok) {
         struct pipe_sampler_state uint_repeat_state = sampler_state;
         uint_repeat_state.wrap_s = PIPE_TEX_WRAP_REPEAT;
         uint_repeat_state.wrap_t = PIPE_TEX_WRAP_REPEAT;
         void *uint_repeat_sampler =
            ctx->create_sampler_state(ctx, &uint_repeat_state);
         texture_ok = uint_repeat_sampler != NULL;
         if (texture_ok) {
            void *repeat_samplers[1] = {uint_repeat_sampler};
            struct pipe_sampler_view *level_zero_views[1] = {uint_sample_view};
            gex_bind_samplers(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, repeat_samplers);
            ctx->set_sampler_views(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, level_zero_views);
            gex_bind_fs(ctx, uint_sample_fs[1]);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            const uint8_t green[4] = {0, 255, 0, 255};
            texture_ok = check_asymmetric_color(
               ctx, resource, false, false, green,
               "texture-rgba8ui-normalized-repeat-negative");
         }
         if (uint_repeat_sampler)
            gex_delete_sampler(ctx, uint_repeat_sampler);
         if (sample_sampler) {
            void *default_samplers[1] = {sample_sampler};
            gex_bind_samplers(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, default_samplers);
         }
      }
      if (texture_ok) {
         struct pipe_sampler_view *swizzle_views[1] = {
            uint_sample_swizzle_view,
         };
         ctx->set_sampler_views(
            ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, swizzle_views);
         gex_bind_fs(ctx, uint_sample_fs[5]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t swizzled[4] = {0, 0, 255, 1};
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, swizzled,
            "texture-rgba8ui-normalized-swizzle-bgr-one");
      }
      if (texture_ok) {
         struct pipe_sampler_state uint_mirror_state = sampler_state;
         uint_mirror_state.wrap_s = PIPE_TEX_WRAP_MIRROR_REPEAT;
         uint_mirror_state.wrap_t = PIPE_TEX_WRAP_MIRROR_REPEAT;
         void *uint_mirror_sampler =
            ctx->create_sampler_state(ctx, &uint_mirror_state);
         texture_ok = uint_mirror_sampler != NULL;
         if (texture_ok) {
            void *mirror_samplers[1] = {uint_mirror_sampler};
            struct pipe_sampler_view *level_zero_views[1] = {uint_sample_view};
            gex_bind_samplers(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, mirror_samplers);
            ctx->set_sampler_views(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, level_zero_views);
            gex_bind_fs(ctx, uint_sample_fs[4]);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            const uint8_t green[4] = {0, 255, 0, 255};
            texture_ok = check_asymmetric_color(
               ctx, resource, false, false, green,
               "texture-rgba8ui-normalized-mirror-negative");
         }
         if (uint_mirror_sampler)
            gex_delete_sampler(ctx, uint_mirror_sampler);
         if (sample_sampler) {
            void *default_samplers[1] = {sample_sampler};
            gex_bind_samplers(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, default_samplers);
         }
      }
      if (texture_ok) {
         struct pipe_sampler_state uint_border_state = sampler_state;
         uint_border_state.wrap_s = PIPE_TEX_WRAP_CLAMP_TO_BORDER;
         uint_border_state.wrap_t = PIPE_TEX_WRAP_CLAMP_TO_BORDER;
         uint_border_state.border_color_is_integer = true;
         uint_border_state.border_color.ui[0] = 17;
         uint_border_state.border_color.ui[1] = 33;
         uint_border_state.border_color.ui[2] = 65;
         uint_border_state.border_color.ui[3] = 129;
         void *uint_border_sampler =
            ctx->create_sampler_state(ctx, &uint_border_state);
         texture_ok = uint_border_sampler != NULL;
         if (texture_ok) {
            void *border_samplers[1] = {uint_border_sampler};
            struct pipe_sampler_view *level_zero_views[1] = {uint_sample_view};
            gex_bind_samplers(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, border_samplers);
            ctx->set_sampler_views(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, level_zero_views);
            gex_bind_fs(ctx, uint_sample_fs[1]);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            const uint8_t border[4] = {17, 33, 65, 129};
            texture_ok = check_asymmetric_color(
               ctx, resource, false, false, border,
               "texture-rgba8ui-normalized-integer-border");
         }
         if (uint_border_sampler)
            gex_delete_sampler(ctx, uint_border_sampler);
         if (sample_sampler) {
            void *default_samplers[1] = {sample_sampler};
            gex_bind_samplers(
               ctx, MESA_SHADER_FRAGMENT, 0, 1, default_samplers);
         }
      }
   }
   for (unsigned i = 0; i < 9; ++i) {
      if (!texture_ok && uint_sample_states[i].error_message)
         fprintf(stderr, "FAIL gex-wgpu uint-sample shader %u=%s\n",
                 i, uint_sample_states[i].error_message);
      free(uint_sample_states[i].error_message);
      if (uint_sample_fs[i])
         gex_delete_fs(ctx, uint_sample_fs[i]);
   }
   if (!texture_ok && uint_gradient_vs_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu uint-gradient shader=%s\n",
              uint_gradient_vs_state.error_message);
   free(uint_gradient_vs_state.error_message);
   if (uint_gradient_vs)
      gex_delete_vs(ctx, uint_gradient_vs);
   if (uint_sample_view)
      pipe_sampler_view_reference(&uint_sample_view, NULL);
   if (uint_sample_base_view)
      pipe_sampler_view_reference(&uint_sample_base_view, NULL);
   if (uint_sample_mip_view)
      pipe_sampler_view_reference(&uint_sample_mip_view, NULL);
   if (uint_sample_swizzle_view)
      pipe_sampler_view_reference(&uint_sample_swizzle_view, NULL);
   if (uint_sample_texture)
      pipe_resource_reference(&uint_sample_texture, NULL);
   struct pipe_resource sint_sample_templ = sample_templ;
   sint_sample_templ.format = PIPE_FORMAT_R8G8B8A8_SINT;
   sint_sample_templ.last_level = 0;
   struct pipe_resource *sint_sample_texture =
      screen->resource_create(screen, &sint_sample_templ);
   const int8_t sint_sample_texels[16] = {
      -128, -64, 0, 127, 127, 0, -64, -128,
      0, 0, 0, 0, 0, 0, 0, 0,
   };
   struct pipe_sampler_view sint_sample_view_templ = view_templ;
   sint_sample_view_templ.format = PIPE_FORMAT_R8G8B8A8_SINT;
   sint_sample_view_templ.u.tex.last_level = 0;
   struct pipe_sampler_view *sint_sample_view = sint_sample_texture
      ? ctx->create_sampler_view(ctx, sint_sample_texture,
                                 &sint_sample_view_templ) : NULL;
   struct pipe_shader_state sint_sample_states[2] = {{0}};
   sint_sample_states[0].type = PIPE_SHADER_IR_NIR;
   sint_sample_states[0].ir.nir =
      make_integer_texture_lod_fs(true, false, false, 0.75f, 0.25f, 0.0f);
   sint_sample_states[0].report_compile_error = true;
   sint_sample_states[1].type = PIPE_SHADER_IR_NIR;
   sint_sample_states[1].ir.nir =
      make_integer_texture_lod_fs(true, false, false, -0.25f, 0.25f, 0.0f);
   sint_sample_states[1].report_compile_error = true;
   void *sint_sample_fs[2] = {
      ctx->create_fs_state(ctx, &sint_sample_states[0]),
      ctx->create_fs_state(ctx, &sint_sample_states[1]),
   };
   if (texture_ok)
      texture_ok = sint_sample_texture && sint_sample_view &&
                   sint_sample_fs[0] && sint_sample_fs[1];
   if (texture_ok) {
      const uint8_t expected[4] = {255, 128, 64, 0};
      ctx->texture_subdata(ctx, sint_sample_texture, 0, PIPE_MAP_WRITE,
                           &sample_box, sint_sample_texels, 8, 16);
      struct pipe_sampler_view *views[1] = {sint_sample_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&sint_sample_texture, NULL);
      gex_bind_fs(ctx, sint_sample_fs[0]);
      ctx->clear_render_target(ctx, &surface, &color,
                               0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, expected,
         "texture-rgba8i-normalized-lod-nearest");
   }
   if (texture_ok) {
      struct pipe_sampler_state sint_border_state = sampler_state;
      sint_border_state.wrap_s = PIPE_TEX_WRAP_CLAMP_TO_BORDER;
      sint_border_state.wrap_t = PIPE_TEX_WRAP_CLAMP_TO_BORDER;
      sint_border_state.border_color_is_integer = true;
      sint_border_state.border_color.i[0] = -128;
      sint_border_state.border_color.i[1] = -64;
      sint_border_state.border_color.i[2] = 0;
      sint_border_state.border_color.i[3] = 127;
      void *sint_border_sampler =
         ctx->create_sampler_state(ctx, &sint_border_state);
      texture_ok = sint_border_sampler != NULL;
      if (texture_ok) {
         void *border_samplers[1] = {sint_border_sampler};
         gex_bind_samplers(
            ctx, MESA_SHADER_FRAGMENT, 0, 1, border_samplers);
         gex_bind_fs(ctx, sint_sample_fs[1]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t border[4] = {0, 64, 128, 255};
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, border,
            "texture-rgba8i-normalized-integer-border");
      }
      if (sint_border_sampler)
         gex_delete_sampler(ctx, sint_border_sampler);
      if (sample_sampler) {
         void *default_samplers[1] = {sample_sampler};
         gex_bind_samplers(
            ctx, MESA_SHADER_FRAGMENT, 0, 1, default_samplers);
      }
   }
   for (unsigned i = 0; i < 2; ++i) {
      if (!texture_ok && sint_sample_states[i].error_message)
         fprintf(stderr, "FAIL gex-wgpu sint-sample shader %u=%s\n",
                 i, sint_sample_states[i].error_message);
      free(sint_sample_states[i].error_message);
      if (sint_sample_fs[i])
         gex_delete_fs(ctx, sint_sample_fs[i]);
   }
   if (sint_sample_view)
      pipe_sampler_view_reference(&sint_sample_view, NULL);
   if (sint_sample_texture)
      pipe_resource_reference(&sint_sample_texture, NULL);
   struct pipe_resource uint_array_templ = sample_templ;
   uint_array_templ.target = PIPE_TEXTURE_2D_ARRAY;
   uint_array_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   uint_array_templ.array_size = 2;
   uint_array_templ.last_level = 0;
   struct pipe_resource *uint_array_texture =
      screen->resource_create(screen, &uint_array_templ);
   const uint8_t uint_array_texels[32] = {
      255, 0, 0, 255, 255, 0, 0, 255,
      255, 0, 0, 255, 255, 0, 0, 255,
      0, 255, 0, 255, 0, 255, 0, 255,
      0, 255, 0, 255, 0, 255, 0, 255,
   };
   struct pipe_sampler_view uint_array_view_templ = view_templ;
   uint_array_view_templ.target = PIPE_TEXTURE_2D_ARRAY;
   uint_array_view_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   uint_array_view_templ.u.tex.first_layer = 0;
   uint_array_view_templ.u.tex.last_layer = 1;
   uint_array_view_templ.u.tex.last_level = 0;
   struct pipe_sampler_view *uint_array_view = uint_array_texture
      ? ctx->create_sampler_view(ctx, uint_array_texture,
                                 &uint_array_view_templ) : NULL;
   struct pipe_shader_state uint_array_states[3] = {{0}};
   const float array_layers[3] = {0.49f, 0.51f, 5.0f};
   void *uint_array_fs[3] = {0};
   for (unsigned i = 0; i < 3; ++i) {
      uint_array_states[i].type = PIPE_SHADER_IR_NIR;
      uint_array_states[i].ir.nir =
         make_uint_array_texture_lod_fs(array_layers[i]);
      uint_array_states[i].report_compile_error = true;
      uint_array_fs[i] = ctx->create_fs_state(ctx, &uint_array_states[i]);
   }
   if (texture_ok)
      texture_ok = uint_array_texture && uint_array_view &&
                   uint_array_fs[0] && uint_array_fs[1] && uint_array_fs[2];
   if (texture_ok) {
      const struct pipe_box array_box = {
         .x = 0, .y = 0, .z = 0, .width = 2, .height = 2, .depth = 2,
      };
      ctx->texture_subdata(ctx, uint_array_texture, 0, PIPE_MAP_WRITE,
                           &array_box, uint_array_texels, 8, 16);
      struct pipe_sampler_view *views[1] = {uint_array_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&uint_array_texture, NULL);
      const uint8_t expected[3][4] = {
         {255, 0, 0, 255},
         {0, 255, 0, 255},
         {0, 255, 0, 255},
      };
      const char *labels[3] = {
         "texture-rgba8ui-array-layer-round-down-lifetime",
         "texture-rgba8ui-array-layer-round-up-lifetime",
         "texture-rgba8ui-array-layer-clamp-lifetime",
      };
      gex_bind_vs(ctx, vs);
      for (unsigned i = 0; texture_ok && i < 3; ++i) {
         gex_bind_fs(ctx, uint_array_fs[i]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, expected[i], labels[i]);
      }
   }
   for (unsigned i = 0; i < 3; ++i) {
      if (!texture_ok && uint_array_states[i].error_message)
         fprintf(stderr, "FAIL gex-wgpu uint-array shader %u=%s\n",
                 i, uint_array_states[i].error_message);
      free(uint_array_states[i].error_message);
      if (uint_array_fs[i])
         gex_delete_fs(ctx, uint_array_fs[i]);
   }
   if (uint_array_view)
      pipe_sampler_view_reference(&uint_array_view, NULL);
   if (uint_array_texture)
      pipe_resource_reference(&uint_array_texture, NULL);
   struct pipe_resource one_d_templ = sample_templ;
   one_d_templ.target = PIPE_TEXTURE_1D;
   one_d_templ.width0 = 2;
   one_d_templ.height0 = 1;
   one_d_templ.depth0 = 1;
   one_d_templ.array_size = 1;
   one_d_templ.last_level = 1;
   struct pipe_resource *one_d_texture =
      screen->resource_create(screen, &one_d_templ);
   const uint8_t one_d_texels[8] = {
      255, 0, 0, 255, 0, 255, 0, 255,
   };
   const uint8_t one_d_mip[4] = {0, 0, 255, 255};
   struct pipe_sampler_view one_d_view_templ = view_templ;
   one_d_view_templ.target = PIPE_TEXTURE_1D;
   one_d_view_templ.u.tex.last_level = 1;
   struct pipe_sampler_view *one_d_view = one_d_texture
      ? ctx->create_sampler_view(ctx, one_d_texture, &one_d_view_templ) : NULL;
   struct pipe_shader_state one_d_states[3] = {{0}};
   const float one_d_coords[3] = {0.75f, -0.25f, 0.75f};
   const float one_d_lods[3] = {0.0f, 0.0f, 1.0f};
   void *one_d_fs[3] = {0};
   for (unsigned i = 0; i < 3; ++i) {
      one_d_states[i].type = PIPE_SHADER_IR_NIR;
      one_d_states[i].ir.nir =
         make_1d_texture_fs(false, false, one_d_coords[i], one_d_lods[i]);
      one_d_states[i].report_compile_error = true;
      one_d_fs[i] = ctx->create_fs_state(ctx, &one_d_states[i]);
   }
   if (texture_ok)
      texture_ok = one_d_texture && one_d_view && one_d_fs[0] &&
                   one_d_fs[1] && one_d_fs[2];
   if (texture_ok) {
      const struct pipe_box line_box = {
         .x = 0, .y = 0, .z = 0, .width = 2, .height = 1, .depth = 1,
      };
      ctx->texture_subdata(ctx, one_d_texture, 0, PIPE_MAP_WRITE,
                           &line_box, one_d_texels, 8, 8);
      ctx->texture_subdata(ctx, one_d_texture, 1, PIPE_MAP_WRITE,
                           &mip_box, one_d_mip, 4, 4);
      struct pipe_sampler_view *views[1] = {one_d_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&one_d_texture, NULL);
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, one_d_fs[0]);
      ctx->clear_render_target(ctx, &surface, &color,
                               0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      const uint8_t green[4] = {0, 255, 0, 255};
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, green,
         "texture-1d-normalized-lod-zero-lifetime");
   }
   if (texture_ok) {
      struct pipe_sampler_state repeat_1d_state = sampler_state;
      repeat_1d_state.wrap_s = PIPE_TEX_WRAP_REPEAT;
      void *repeat_1d_sampler =
         ctx->create_sampler_state(ctx, &repeat_1d_state);
      texture_ok = repeat_1d_sampler != NULL;
      if (texture_ok) {
         void *samplers[1] = {repeat_1d_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_fs(ctx, one_d_fs[1]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         /* REPEAT folds -0.25 back to 0.75, which lands on the second
          * (green) texel; CLAMP_TO_EDGE would give the first (red) one.
          */
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, green,
            "texture-1d-repeat-negative");
      }
      if (repeat_1d_sampler)
         gex_delete_sampler(ctx, repeat_1d_sampler);
   }
   if (texture_ok) {
      struct pipe_sampler_state mip_1d_state = sampler_state;
      mip_1d_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
      mip_1d_state.max_lod = 1.0f;
      void *mip_1d_sampler = ctx->create_sampler_state(ctx, &mip_1d_state);
      texture_ok = mip_1d_sampler != NULL;
      if (texture_ok) {
         void *samplers[1] = {mip_1d_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_fs(ctx, one_d_fs[2]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t blue[4] = {0, 0, 255, 255};
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, blue,
            "texture-1d-explicit-lod-one");
      }
      if (mip_1d_sampler)
         gex_delete_sampler(ctx, mip_1d_sampler);
   }
   for (unsigned i = 0; i < 3; ++i) {
      if (!texture_ok && one_d_states[i].error_message)
         fprintf(stderr, "FAIL gex-wgpu 1d shader %u=%s\n",
                 i, one_d_states[i].error_message);
      free(one_d_states[i].error_message);
      if (one_d_fs[i])
         gex_delete_fs(ctx, one_d_fs[i]);
   }
   if (one_d_view)
      pipe_sampler_view_reference(&one_d_view, NULL);
   if (one_d_texture)
      pipe_resource_reference(&one_d_texture, NULL);
   if (sample_sampler) {
      void *samplers[1] = {sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }

   struct pipe_resource uint_1d_templ = one_d_templ;
   uint_1d_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   struct pipe_resource *uint_1d_texture =
      screen->resource_create(screen, &uint_1d_templ);
   const uint8_t uint_1d_mip[4] = {0, 255, 255, 255};
   struct pipe_sampler_view uint_1d_view_templ = one_d_view_templ;
   uint_1d_view_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   struct pipe_sampler_view *uint_1d_view = uint_1d_texture
      ? ctx->create_sampler_view(ctx, uint_1d_texture,
                                 &uint_1d_view_templ) : NULL;
   struct pipe_shader_state uint_1d_states[4] = {{0}};
   uint_1d_states[0].ir.nir = make_1d_texture_fs(
      true, false, 0.75f, 0.0f);
   uint_1d_states[1].ir.nir = make_1d_texture_fs(
      true, false, -0.25f, 0.0f);
   uint_1d_states[2].ir.nir = make_1d_texture_fs(
      true, false, 0.75f, 0.51f);
   uint_1d_states[3].ir.nir = make_1d_texture_fs(
      true, true, 0.0f, 0.0f);
   void *uint_1d_fs[4] = {0};
   for (unsigned i = 0; i < 4; ++i) {
      uint_1d_states[i].type = PIPE_SHADER_IR_NIR;
      uint_1d_states[i].report_compile_error = true;
      uint_1d_fs[i] = ctx->create_fs_state(ctx, &uint_1d_states[i]);
   }
   struct pipe_shader_state uint_1d_vs_state = {0};
   uint_1d_vs_state.type = PIPE_SHADER_IR_NIR;
   uint_1d_vs_state.ir.nir = make_gradient_texture_1d_vs(16.0f);
   uint_1d_vs_state.report_compile_error = true;
   void *uint_1d_vs = ctx->create_vs_state(ctx, &uint_1d_vs_state);
   if (texture_ok)
      texture_ok = uint_1d_texture && uint_1d_view && uint_1d_fs[0] &&
                   uint_1d_fs[1] && uint_1d_fs[2] && uint_1d_fs[3] &&
                   uint_1d_vs;
   if (texture_ok) {
      const struct pipe_box line_box = {
         .x = 0, .y = 0, .z = 0, .width = 2, .height = 1, .depth = 1,
      };
      ctx->texture_subdata(ctx, uint_1d_texture, 0, PIPE_MAP_WRITE,
                           &line_box, one_d_texels, 8, 8);
      ctx->texture_subdata(ctx, uint_1d_texture, 1, PIPE_MAP_WRITE,
                           &mip_box, uint_1d_mip, 4, 4);
      struct pipe_sampler_view *views[1] = {uint_1d_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&uint_1d_texture, NULL);
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, uint_1d_fs[0]);
      ctx->clear_render_target(ctx, &surface, &color,
                               0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      const uint8_t green[4] = {0, 255, 0, 255};
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, green,
         "texture-rgba8ui-1d-normalized-lod-zero-lifetime");
   }
   if (texture_ok) {
      struct pipe_sampler_state repeat_1d_state = sampler_state;
      repeat_1d_state.wrap_s = PIPE_TEX_WRAP_REPEAT;
      void *repeat_1d_sampler =
         ctx->create_sampler_state(ctx, &repeat_1d_state);
      texture_ok = repeat_1d_sampler != NULL;
      if (texture_ok) {
         void *samplers[1] = {repeat_1d_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_fs(ctx, uint_1d_fs[1]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t green[4] = {0, 255, 0, 255};
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, green,
            "texture-rgba8ui-1d-normalized-repeat-negative");
      }
      if (repeat_1d_sampler)
         gex_delete_sampler(ctx, repeat_1d_sampler);
   }
   if (texture_ok) {
      struct pipe_sampler_state mip_1d_state = sampler_state;
      mip_1d_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
      mip_1d_state.max_lod = 1.0f;
      void *mip_1d_sampler = ctx->create_sampler_state(ctx, &mip_1d_state);
      texture_ok = mip_1d_sampler != NULL;
      if (texture_ok) {
         void *samplers[1] = {mip_1d_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_fs(ctx, uint_1d_fs[2]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t cyan[4] = {0, 255, 255, 255};
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, cyan,
            "texture-rgba8ui-1d-explicit-lod-round-up");
         if (texture_ok) {
            gex_bind_vs(ctx, uint_1d_vs);
            gex_bind_fs(ctx, uint_1d_fs[3]);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            texture_ok = check_uniform(
               ctx, resource, cyan,
               "texture-rgba8ui-1d-implicit-lod-derivative-level-one");
            gex_bind_vs(ctx, vs);
         }
      }
      if (mip_1d_sampler)
         gex_delete_sampler(ctx, mip_1d_sampler);
   }
   for (unsigned i = 0; i < 4; ++i) {
      if (!texture_ok && uint_1d_states[i].error_message)
         fprintf(stderr, "FAIL gex-wgpu uint-1d shader %u=%s\n",
                 i, uint_1d_states[i].error_message);
      free(uint_1d_states[i].error_message);
      if (uint_1d_fs[i])
         gex_delete_fs(ctx, uint_1d_fs[i]);
   }
   if (!texture_ok && uint_1d_vs_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu uint-1d vertex shader=%s\n",
              uint_1d_vs_state.error_message);
   free(uint_1d_vs_state.error_message);
   if (uint_1d_vs)
      gex_delete_vs(ctx, uint_1d_vs);
   if (uint_1d_view)
      pipe_sampler_view_reference(&uint_1d_view, NULL);
   if (uint_1d_texture)
      pipe_resource_reference(&uint_1d_texture, NULL);
   if (sample_sampler) {
      void *samplers[1] = {sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
   const uint8_t one_d_array_texels[16] = {
      255, 0, 0, 255, 255, 0, 0, 255,
      0, 255, 0, 255, 0, 255, 0, 255,
   };
   for (unsigned integer = 0; texture_ok && integer < 2; ++integer) {
      struct pipe_resource one_d_array_templ = one_d_templ;
      one_d_array_templ.target = PIPE_TEXTURE_1D_ARRAY;
      one_d_array_templ.format = integer
         ? PIPE_FORMAT_R8G8B8A8_UINT : PIPE_FORMAT_R8G8B8A8_UNORM;
      one_d_array_templ.array_size = 2;
      one_d_array_templ.last_level = 0;
      struct pipe_resource *one_d_array_texture =
         screen->resource_create(screen, &one_d_array_templ);
      struct pipe_box one_d_array_box = {
         .x = 0, .y = 0, .z = 0, .width = 2, .height = 2, .depth = 1,
      };
      if (one_d_array_texture)
         ctx->texture_subdata(ctx, one_d_array_texture, 0, PIPE_MAP_WRITE,
                              &one_d_array_box, one_d_array_texels, 8, 16);
      struct pipe_sampler_view one_d_array_view_templ = one_d_view_templ;
      one_d_array_view_templ.target = PIPE_TEXTURE_1D_ARRAY;
      one_d_array_view_templ.format = one_d_array_templ.format;
      one_d_array_view_templ.u.tex.last_level = 0;
      one_d_array_view_templ.u.tex.last_layer = 1;
      struct pipe_sampler_view *one_d_array_view = one_d_array_texture
         ? ctx->create_sampler_view(ctx, one_d_array_texture,
                                    &one_d_array_view_templ) : NULL;
      struct pipe_shader_state state = {0};
      state.type = PIPE_SHADER_IR_NIR;
      state.ir.nir = make_1d_array_texture_fs(integer, 0.75f, 1.0f);
      state.report_compile_error = true;
      void *fs = ctx->create_fs_state(ctx, &state);
      texture_ok = one_d_array_texture && one_d_array_view && fs;
      if (texture_ok) {
         const uint8_t green[4] = {0, 255, 0, 255};
         struct pipe_sampler_view *views[1] = {one_d_array_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
         pipe_resource_reference(&one_d_array_texture, NULL);
         gex_bind_fs(ctx, fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, green, integer
               ? "texture-rgba8ui-1d-array-layer-one-lifetime"
               : "texture-float-1d-array-layer-one-lifetime");
      } else if (state.error_message) {
         fprintf(stderr, "FAIL gex-wgpu 1d-array shader=%s\n",
                 state.error_message);
      }
      free(state.error_message);
      if (fs)
         gex_delete_fs(ctx, fs);
      if (one_d_array_view)
         pipe_sampler_view_reference(&one_d_array_view, NULL);
      if (one_d_array_texture)
         pipe_resource_reference(&one_d_array_texture, NULL);
   }
   struct pipe_sampler_state rect_sampler_state = sampler_state;
   rect_sampler_state.unnormalized_coords = 1;
   void *rect_sampler = ctx->create_sampler_state(ctx, &rect_sampler_state);
   if (texture_ok)
      texture_ok = rect_sampler != NULL;
   if (!rect_sampler)
      fprintf(stderr, "FAIL gex-wgpu rect sampler state\n");
   if (texture_ok) {
      void *samplers[1] = {rect_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
#ifdef GEX_CONFORMANCE_SOFTPIPE
   /* The wasm softpipe TGSI interpreter has no valid exec_tex dispatch for
    * integer RECT sampling and traps in exec_tex.  Keep the logical float
    * RECT oracle here; the exact integer RECT case still runs on WebGPU. */
   const unsigned rect_case_count = 1;
#else
   const unsigned rect_case_count = 2;
#endif
   void *rect_fs[2] = {0};
   for (unsigned integer = 0;
        texture_ok && integer < rect_case_count; ++integer) {
      struct pipe_resource rect_templ = sample_templ;
      rect_templ.target = PIPE_TEXTURE_RECT;
      rect_templ.format = integer
         ? PIPE_FORMAT_R8G8B8A8_UINT : PIPE_FORMAT_R8G8B8A8_UNORM;
      rect_templ.last_level = 0;
      struct pipe_resource *rect_texture =
         screen->resource_create(screen, &rect_templ);
      if (rect_texture)
         ctx->texture_subdata(ctx, rect_texture, 0, PIPE_MAP_WRITE,
                              &sample_box, texels, 8, 16);
      struct pipe_sampler_view rect_view_templ = view_templ;
      rect_view_templ.target = PIPE_TEXTURE_RECT;
      rect_view_templ.format = rect_templ.format;
      rect_view_templ.u.tex.last_level = 0;
      struct pipe_sampler_view *rect_view = rect_texture
         ? ctx->create_sampler_view(ctx, rect_texture, &rect_view_templ) : NULL;
      struct pipe_shader_state state = {0};
      state.type = PIPE_SHADER_IR_NIR;
      state.ir.nir = make_rect_texture_fs(integer, 1.5f, 0.5f);
      state.report_compile_error = true;
      rect_fs[integer] = ctx->create_fs_state(ctx, &state);
      void *fs = rect_fs[integer];
      texture_ok = rect_texture && rect_view && fs;
      if (!texture_ok)
         fprintf(stderr,
                 "FAIL gex-wgpu rect setup integer=%u resource=%p view=%p "
                 "shader=%p error=%s\n",
                 integer, (void *)rect_texture, (void *)rect_view, fs,
                 state.error_message ? state.error_message : "none");
      if (texture_ok) {
         const uint8_t yellow[4] = {255, 255, 0, 255};
         struct pipe_sampler_view *views[1] = {rect_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
         pipe_resource_reference(&rect_texture, NULL);
         gex_bind_fs(ctx, fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, yellow, integer
               ? "texture-rgba8ui-rect-unnormalized-coordinate-lifetime"
               : "texture-float-rect-unnormalized-coordinate-lifetime");
      } else if (state.error_message) {
         fprintf(stderr, "FAIL gex-wgpu rect shader=%s\n",
                 state.error_message);
      }
      free(state.error_message);
      if (rect_view)
         pipe_sampler_view_reference(&rect_view, NULL);
      if (rect_texture)
         pipe_resource_reference(&rect_texture, NULL);
   }
   if (sample_sampler) {
      void *samplers[1] = {sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
   struct pipe_resource uint_3d_templ = sample_templ;
   uint_3d_templ.target = PIPE_TEXTURE_3D;
   uint_3d_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   uint_3d_templ.width0 = 2;
   uint_3d_templ.height0 = 2;
   uint_3d_templ.depth0 = 2;
   uint_3d_templ.array_size = 1;
   uint_3d_templ.last_level = 1;
   struct pipe_resource *uint_3d_texture =
      screen->resource_create(screen, &uint_3d_templ);
   const uint8_t uint_3d_texels[32] = {
      255, 0, 0, 255, 255, 0, 0, 255,
      255, 0, 0, 255, 255, 0, 0, 255,
      0, 255, 0, 255, 0, 255, 0, 255,
      0, 255, 0, 255, 0, 255, 0, 255,
   };
   const uint8_t uint_3d_mip_texel[4] = {0, 255, 255, 255};
   struct pipe_sampler_view uint_3d_view_templ = view_templ;
   uint_3d_view_templ.target = PIPE_TEXTURE_3D;
   uint_3d_view_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   uint_3d_view_templ.u.tex.first_layer = 0;
   uint_3d_view_templ.u.tex.last_layer = 1;
   uint_3d_view_templ.u.tex.last_level = 1;
   struct pipe_sampler_view *uint_3d_view = uint_3d_texture
      ? ctx->create_sampler_view(ctx, uint_3d_texture,
                                 &uint_3d_view_templ) : NULL;
   struct pipe_shader_state uint_3d_state = {0};
   uint_3d_state.type = PIPE_SHADER_IR_NIR;
   uint_3d_state.ir.nir = make_uint_3d_texel_fetch_fs(1);
   uint_3d_state.report_compile_error = true;
   void *uint_3d_fs = ctx->create_fs_state(ctx, &uint_3d_state);
   struct pipe_shader_state uint_3d_sample_states[5] = {{0}};
   const float sample_r[5] = {0.75f, -0.25f, -0.75f, 0.75f, 0.75f};
   const float sample_lod[5] = {0.0f, 0.0f, 0.0f, 0.49f, 0.51f};
   void *uint_3d_sample_fs[5] = {0};
   for (unsigned i = 0; i < 5; ++i) {
      uint_3d_sample_states[i].type = PIPE_SHADER_IR_NIR;
      uint_3d_sample_states[i].ir.nir =
         make_uint_3d_texture_lod_fs(sample_r[i], sample_lod[i]);
      uint_3d_sample_states[i].report_compile_error = true;
      uint_3d_sample_fs[i] =
         ctx->create_fs_state(ctx, &uint_3d_sample_states[i]);
   }
   struct pipe_shader_state uint_3d_implicit_state = {0};
   uint_3d_implicit_state.type = PIPE_SHADER_IR_NIR;
   uint_3d_implicit_state.ir.nir = make_uint_3d_texture_implicit_fs();
   uint_3d_implicit_state.report_compile_error = true;
   void *uint_3d_implicit_fs =
      ctx->create_fs_state(ctx, &uint_3d_implicit_state);
   struct pipe_shader_state uint_3d_gradient_vs_state = {0};
   uint_3d_gradient_vs_state.type = PIPE_SHADER_IR_NIR;
   uint_3d_gradient_vs_state.ir.nir = make_gradient_texture_3d_vs(16.0f);
   uint_3d_gradient_vs_state.report_compile_error = true;
   void *uint_3d_gradient_vs =
      ctx->create_vs_state(ctx, &uint_3d_gradient_vs_state);
   if (texture_ok)
      texture_ok = uint_3d_texture && uint_3d_view && uint_3d_fs &&
                   uint_3d_sample_fs[0] && uint_3d_sample_fs[1] &&
                   uint_3d_sample_fs[2] && uint_3d_sample_fs[3] &&
                   uint_3d_sample_fs[4] && uint_3d_implicit_fs &&
                   uint_3d_gradient_vs;
   if (texture_ok) {
      const struct pipe_box volume_box = {
         .x = 0, .y = 0, .z = 0, .width = 2, .height = 2, .depth = 2,
      };
      ctx->texture_subdata(ctx, uint_3d_texture, 0, PIPE_MAP_WRITE,
                           &volume_box, uint_3d_texels, 8, 16);
      const struct pipe_box mip_volume_box = {
         .x = 0, .y = 0, .z = 0, .width = 1, .height = 1, .depth = 1,
      };
      ctx->texture_subdata(ctx, uint_3d_texture, 1, PIPE_MAP_WRITE,
                           &mip_volume_box, uint_3d_mip_texel, 4, 4);
      struct pipe_sampler_view *views[1] = {uint_3d_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&uint_3d_texture, NULL);
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, uint_3d_fs);
      ctx->clear_render_target(ctx, &surface, &color,
                               0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      const uint8_t green[4] = {0, 255, 0, 255};
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, green,
         "texture-rgba8ui-3d-texel-fetch-z-one-lifetime");
   }
   if (texture_ok) {
      const uint8_t expected[2][4] = {
         {0, 255, 0, 255},
         {255, 0, 0, 255},
      };
      const char *labels[2] = {
         "texture-rgba8ui-3d-normalized-r-upper",
         "texture-rgba8ui-3d-normalized-r-clamp-negative",
      };
      for (unsigned i = 0; texture_ok && i < 2; ++i) {
         gex_bind_fs(ctx, uint_3d_sample_fs[i]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, expected[i], labels[i]);
      }
   }
   if (texture_ok) {
      struct pipe_sampler_state mip_3d_state = sampler_state;
      mip_3d_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
      mip_3d_state.max_lod = 1.0f;
      void *mip_3d_sampler = ctx->create_sampler_state(ctx, &mip_3d_state);
      texture_ok = mip_3d_sampler != NULL;
      if (texture_ok) {
         void *samplers[1] = {mip_3d_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         const uint8_t expected[2][4] = {
            {0, 255, 0, 255},
            {0, 255, 255, 255},
         };
         const char *labels[2] = {
            "texture-rgba8ui-3d-explicit-lod-nearest-round-down",
            "texture-rgba8ui-3d-explicit-lod-nearest-round-up",
         };
         for (unsigned i = 0; texture_ok && i < 2; ++i) {
            gex_bind_fs(ctx, uint_3d_sample_fs[3 + i]);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            texture_ok = check_asymmetric_color(
               ctx, resource, false, false, expected[i], labels[i]);
         }
         if (texture_ok) {
            gex_bind_vs(ctx, uint_3d_gradient_vs);
            gex_bind_fs(ctx, uint_3d_implicit_fs);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            const uint8_t cyan[4] = {0, 255, 255, 255};
            texture_ok = check_uniform(
               ctx, resource, cyan,
               "texture-rgba8ui-3d-implicit-lod-r-derivative-level-one");
            gex_bind_vs(ctx, vs);
         }
      }
      if (mip_3d_sampler)
         gex_delete_sampler(ctx, mip_3d_sampler);
   }
   if (texture_ok) {
      struct pipe_sampler_state repeat_r_state = sampler_state;
      repeat_r_state.wrap_r = PIPE_TEX_WRAP_REPEAT;
      void *repeat_r_sampler =
         ctx->create_sampler_state(ctx, &repeat_r_state);
      texture_ok = repeat_r_sampler != NULL;
      if (texture_ok) {
         void *samplers[1] = {repeat_r_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_fs(ctx, uint_3d_sample_fs[1]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t green[4] = {0, 255, 0, 255};
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, green,
            "texture-rgba8ui-3d-normalized-r-repeat-negative");
      }
      if (repeat_r_sampler)
         gex_delete_sampler(ctx, repeat_r_sampler);
   }
   if (texture_ok) {
      struct pipe_sampler_state mirror_r_state = sampler_state;
      mirror_r_state.wrap_r = PIPE_TEX_WRAP_MIRROR_REPEAT;
      void *mirror_r_sampler =
         ctx->create_sampler_state(ctx, &mirror_r_state);
      texture_ok = mirror_r_sampler != NULL;
      if (texture_ok) {
         void *samplers[1] = {mirror_r_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_fs(ctx, uint_3d_sample_fs[2]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t green[4] = {0, 255, 0, 255};
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, green,
            "texture-rgba8ui-3d-normalized-r-mirror-negative");
      }
      if (mirror_r_sampler)
         gex_delete_sampler(ctx, mirror_r_sampler);
   }
   if (texture_ok) {
      struct pipe_sampler_state border_r_state = sampler_state;
      border_r_state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_BORDER;
      border_r_state.border_color_is_integer = true;
      border_r_state.border_color.ui[0] = 17;
      border_r_state.border_color.ui[1] = 33;
      border_r_state.border_color.ui[2] = 65;
      border_r_state.border_color.ui[3] = 129;
      void *border_r_sampler =
         ctx->create_sampler_state(ctx, &border_r_state);
      texture_ok = border_r_sampler != NULL;
      if (texture_ok) {
         void *samplers[1] = {border_r_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_fs(ctx, uint_3d_sample_fs[1]);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t border[4] = {17, 33, 65, 129};
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, border,
            "texture-rgba8ui-3d-normalized-r-integer-border");
      }
      if (border_r_sampler)
         gex_delete_sampler(ctx, border_r_sampler);
   }
   if (sample_sampler) {
      void *samplers[1] = {sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
   if (!texture_ok && uint_3d_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu uint-3d shader=%s\n",
              uint_3d_state.error_message);
   free(uint_3d_state.error_message);
   if (uint_3d_fs)
      gex_delete_fs(ctx, uint_3d_fs);
   if (!texture_ok && uint_3d_implicit_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu uint-3d implicit shader=%s\n",
              uint_3d_implicit_state.error_message);
   free(uint_3d_implicit_state.error_message);
   if (uint_3d_implicit_fs)
      gex_delete_fs(ctx, uint_3d_implicit_fs);
   if (!texture_ok && uint_3d_gradient_vs_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu uint-3d gradient vertex shader=%s\n",
              uint_3d_gradient_vs_state.error_message);
   free(uint_3d_gradient_vs_state.error_message);
   if (uint_3d_gradient_vs)
      gex_delete_vs(ctx, uint_3d_gradient_vs);
   for (unsigned i = 0; i < 5; ++i) {
      if (!texture_ok && uint_3d_sample_states[i].error_message)
         fprintf(stderr, "FAIL gex-wgpu uint-3d sample shader %u=%s\n",
                 i, uint_3d_sample_states[i].error_message);
      free(uint_3d_sample_states[i].error_message);
      if (uint_3d_sample_fs[i])
         gex_delete_fs(ctx, uint_3d_sample_fs[i]);
   }
   if (uint_3d_view)
      pipe_sampler_view_reference(&uint_3d_view, NULL);
   if (uint_3d_texture)
      pipe_resource_reference(&uint_3d_texture, NULL);
   struct pipe_resource half_target_templ = templ;
   half_target_templ.format = PIPE_FORMAT_R16G16B16A16_FLOAT;
   half_target_templ.bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW;
   struct pipe_resource *half_target =
      screen->resource_create(screen, &half_target_templ);
   struct pipe_surface half_surface = {0};
   pipe_reference_init(&half_surface.reference, 1);
   half_surface.format = half_target_templ.format;
   half_surface.texture = half_target;
   struct pipe_shader_state half_vs_state = {0}, half_fs_state = {0};
   half_vs_state.type = PIPE_SHADER_IR_NIR;
   half_vs_state.ir.nir = make_vs(0.0f, true);
   half_vs_state.report_compile_error = true;
   half_fs_state.type = PIPE_SHADER_IR_NIR;
   half_fs_state.ir.nir = make_color_fs(0.25f, 0.5f, 0.75f, 1.0f);
   half_fs_state.report_compile_error = true;
   void *half_vs = ctx->create_vs_state(ctx, &half_vs_state);
   void *half_fs = ctx->create_fs_state(ctx, &half_fs_state);
   struct pipe_sampler_view half_view_templ = view_templ;
   half_view_templ.format = half_target_templ.format;
   half_view_templ.u.tex.last_level = 0;
   struct pipe_sampler_view *half_view = NULL;
   if (texture_ok)
      texture_ok = half_target && half_vs && half_fs;
   if (texture_ok) {
      framebuffer.cbufs[0] = half_surface;
      ctx->set_framebuffer_state(ctx, &framebuffer);
      gex_bind_vs(ctx, half_vs);
      gex_bind_fs(ctx, half_fs);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      framebuffer.cbufs[0] = surface;
      ctx->set_framebuffer_state(ctx, &framebuffer);
      half_view = ctx->create_sampler_view(
         ctx, half_target, &half_view_templ);
      texture_ok = half_view != NULL;
   }
   if (texture_ok) {
      const uint8_t half_expected[4] = {64, 128, 191, 255};
      struct pipe_sampler_view *views[1] = {half_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&half_target, NULL);
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, texture_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, half_expected,
         "texture-rgba16f-render-then-sample-lifetime");
   }
   framebuffer.cbufs[0] = surface;
   ctx->set_framebuffer_state(ctx, &framebuffer);
   free(half_vs_state.error_message);
   free(half_fs_state.error_message);
   if (half_vs)
      gex_delete_vs(ctx, half_vs);
   if (half_fs)
      gex_delete_fs(ctx, half_fs);
   if (half_view)
      pipe_sampler_view_reference(&half_view, NULL);
   if (half_target)
      pipe_resource_reference(&half_target, NULL);
   struct pipe_resource array_templ = two_texture_templ;
   array_templ.target = PIPE_TEXTURE_2D_ARRAY;
   array_templ.array_size = 2;
   struct pipe_resource *array_texture =
      screen->resource_create(screen, &array_templ);
   const uint8_t array_texels[8] = {
      255, 0, 0, 255,
      0, 255, 0, 255,
   };
   struct pipe_box array_box = {
      .x = 0, .y = 0, .z = 0, .width = 1, .height = 1, .depth = 2,
   };
   if (array_texture)
      ctx->texture_subdata(ctx, array_texture, 0, PIPE_MAP_WRITE,
                           &array_box, array_texels, 4, 4);
   struct pipe_sampler_view array_view_templ = two_view_templ;
   array_view_templ.target = PIPE_TEXTURE_2D_ARRAY;
   array_view_templ.u.tex.last_layer = 1;
   struct pipe_sampler_view *array_view = array_texture
      ? ctx->create_sampler_view(ctx, array_texture, &array_view_templ) : NULL;
   struct pipe_shader_state array_state = {0};
   array_state.type = PIPE_SHADER_IR_NIR;
   array_state.ir.nir = make_texture_array_fs();
   array_state.report_compile_error = true;
   void *array_fs = ctx->create_fs_state(ctx, &array_state);
   if (texture_ok)
      texture_ok = array_texture && array_view && array_fs;
   if (texture_ok) {
      struct pipe_sampler_view *views[1] = {array_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&array_texture, NULL);
      gex_bind_fs(ctx, array_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric(
         ctx, resource, false, false,
         "texture-2d-array-layer-one-lifetime");
   }
   if (!texture_ok && array_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu texture-array shader=%s\n",
              array_state.error_message);
   free(array_state.error_message);
   if (array_fs)
      gex_delete_fs(ctx, array_fs);
   if (array_view)
      pipe_sampler_view_reference(&array_view, NULL);
   if (array_texture)
      pipe_resource_reference(&array_texture, NULL);
   struct pipe_resource cube_templ = two_texture_templ;
   cube_templ.target = PIPE_TEXTURE_CUBE;
   cube_templ.array_size = 6;
   struct pipe_resource *cube_texture =
      screen->resource_create(screen, &cube_templ);
   const uint8_t cube_texels[24] = {
      255, 0, 0, 255, 255, 0, 0, 255,
      255, 0, 0, 255, 255, 0, 0, 255,
      255, 0, 0, 255,   0, 255, 0, 255,
   };
   struct pipe_box cube_box = {
      .x = 0, .y = 0, .z = 0, .width = 1, .height = 1, .depth = 6,
   };
   if (cube_texture)
      ctx->texture_subdata(ctx, cube_texture, 0, PIPE_MAP_WRITE,
                           &cube_box, cube_texels, 4, 4);
   struct pipe_sampler_view cube_view_templ = two_view_templ;
   cube_view_templ.target = PIPE_TEXTURE_CUBE;
   cube_view_templ.u.tex.last_layer = 5;
   struct pipe_sampler_view *cube_view = cube_texture
      ? ctx->create_sampler_view(ctx, cube_texture, &cube_view_templ) : NULL;
   struct pipe_shader_state cube_state = {0};
   cube_state.type = PIPE_SHADER_IR_NIR;
   cube_state.ir.nir = make_texture_cube_fs();
   cube_state.report_compile_error = true;
   void *cube_fs = ctx->create_fs_state(ctx, &cube_state);
   if (texture_ok)
      texture_ok = cube_texture && cube_view && cube_fs;
   if (texture_ok) {
      struct pipe_sampler_view *views[1] = {cube_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&cube_texture, NULL);
      gex_bind_fs(ctx, cube_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric(
         ctx, resource, false, false,
         "texture-cube-negative-z-face-lifetime");
   }
   if (!texture_ok && cube_state.error_message)
      fprintf(stderr, "FAIL gex-wgpu texture-cube shader=%s\n",
              cube_state.error_message);
   free(cube_state.error_message);
   if (cube_fs)
      gex_delete_fs(ctx, cube_fs);
   if (cube_view)
      pipe_sampler_view_reference(&cube_view, NULL);
   if (cube_texture)
      pipe_resource_reference(&cube_texture, NULL);
   struct pipe_resource uint_cube_templ = two_texture_templ;
   uint_cube_templ.target = PIPE_TEXTURE_CUBE;
   uint_cube_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   uint_cube_templ.width0 = 2;
   uint_cube_templ.height0 = 2;
   uint_cube_templ.array_size = 6;
   struct pipe_resource *uint_cube_texture =
      screen->resource_create(screen, &uint_cube_templ);
   const uint8_t uint_cube_texels[6][4][4] = {
      {{ 1, 0, 0, 255}, { 2, 0, 0, 255},
       { 3, 0, 0, 255}, { 4, 0, 0, 255}},
      {{ 5, 0, 0, 255}, { 6, 0, 0, 255},
       { 7, 0, 0, 255}, { 8, 0, 0, 255}},
      {{ 9, 0, 0, 255}, {10, 0, 0, 255},
       {11, 0, 0, 255}, {12, 0, 0, 255}},
      {{13, 0, 0, 255}, {14, 0, 0, 255},
       {15, 0, 0, 255}, {16, 0, 0, 255}},
      {{17, 0, 0, 255}, {18, 0, 0, 255},
       {19, 0, 0, 255}, {20, 0, 0, 255}},
      {{21, 0, 0, 255}, {22, 0, 0, 255},
       {23, 0, 0, 255}, {24, 0, 0, 255}},
   };
   struct pipe_box uint_cube_box = cube_box;
   uint_cube_box.width = 2;
   uint_cube_box.height = 2;
   if (uint_cube_texture)
      ctx->texture_subdata(ctx, uint_cube_texture, 0, PIPE_MAP_WRITE,
                           &uint_cube_box, uint_cube_texels, 8, 16);
   struct pipe_sampler_view uint_cube_view_templ = cube_view_templ;
   uint_cube_view_templ.format = PIPE_FORMAT_R8G8B8A8_UINT;
   struct pipe_sampler_view *uint_cube_view = uint_cube_texture
      ? ctx->create_sampler_view(ctx, uint_cube_texture,
                                 &uint_cube_view_templ) : NULL;
   if (texture_ok)
      texture_ok = uint_cube_texture && uint_cube_view;
   if (texture_ok) {
      struct pipe_sampler_view *views[1] = {uint_cube_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&uint_cube_texture, NULL);
      struct pipe_shader_state state = {0};
      state.type = PIPE_SHADER_IR_NIR;
      state.ir.nir = make_uint_texture_cube_faces_fs();
      state.report_compile_error = true;
      void *fs = ctx->create_fs_state(ctx, &state);
      texture_ok = fs != NULL;
      if (texture_ok) {
         const uint8_t green[4] = {0, 255, 0, 255};
         gex_bind_fs(ctx, fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, green,
            "texture-uint-cube-six-face-orientation-lifetime");
      } else if (state.error_message) {
         fprintf(stderr, "FAIL gex-wgpu texture-uint-cube shader=%s\n",
                 state.error_message);
      }
      free(state.error_message);
      if (fs)
         gex_delete_fs(ctx, fs);
   }
   if (uint_cube_view)
      pipe_sampler_view_reference(&uint_cube_view, NULL);
   if (uint_cube_texture)
      pipe_resource_reference(&uint_cube_texture, NULL);
   const uint8_t cube_array_texels[2][12][4][4] = {
      {[10][0] = {0, 255, 0, 255}},
      {[10][0] = {255, 0, 255, 255}},
   };
   const uint8_t cube_array_expected[2][4] = {
      {0, 255, 0, 255}, {255, 0, 255, 255},
   };
   const char *cube_array_labels[2] = {
      "texture-float-cube-array-base-six-index-zero-orientation-lifetime",
      "texture-uint-cube-array-index-one-positive-z-orientation-lifetime",
   };
   if (texture_ok && webgpu_boundary) {
      struct pipe_resource bad_cube_array_templ = uint_cube_templ;
      bad_cube_array_templ.target = PIPE_TEXTURE_CUBE_ARRAY;
      bad_cube_array_templ.array_size = 7;
      struct pipe_resource *bad_cube_array =
         screen->resource_create(screen, &bad_cube_array_templ);
      texture_ok = bad_cube_array == NULL;
      if (bad_cube_array)
         pipe_resource_reference(&bad_cube_array, NULL);
   }
   for (unsigned integer = 0; texture_ok && integer < 2; ++integer) {
      struct pipe_resource cube_array_templ = uint_cube_templ;
      cube_array_templ.target = PIPE_TEXTURE_CUBE_ARRAY;
      cube_array_templ.format = integer
         ? PIPE_FORMAT_R8G8B8A8_UINT : PIPE_FORMAT_R8G8B8A8_UNORM;
      cube_array_templ.array_size = 12;
      struct pipe_resource *cube_array_texture =
         screen->resource_create(screen, &cube_array_templ);
      struct pipe_box cube_array_box = {
         .x = 0, .y = 0, .z = 0, .width = 2, .height = 2, .depth = 12,
      };
      if (cube_array_texture)
         ctx->texture_subdata(ctx, cube_array_texture, 0, PIPE_MAP_WRITE,
                              &cube_array_box, cube_array_texels[integer],
                              8, 16);
      struct pipe_sampler_view cube_array_view_templ = cube_view_templ;
      cube_array_view_templ.target = PIPE_TEXTURE_CUBE_ARRAY;
      cube_array_view_templ.format = cube_array_templ.format;
      cube_array_view_templ.u.tex.first_layer = integer ? 0 : 6;
      cube_array_view_templ.u.tex.last_layer = 11;
      struct pipe_sampler_view bad_view_templ = cube_array_view_templ;
      bad_view_templ.u.tex.first_layer = 1;
      struct pipe_sampler_view *bad_view = cube_array_texture
         ? ctx->create_sampler_view(ctx, cube_array_texture, &bad_view_templ)
         : NULL;
      if (webgpu_boundary)
         texture_ok = bad_view == NULL;
      if (bad_view)
         pipe_sampler_view_reference(&bad_view, NULL);
      struct pipe_sampler_view *cube_array_view = cube_array_texture
         ? ctx->create_sampler_view(ctx, cube_array_texture,
                                    &cube_array_view_templ) : NULL;
      struct pipe_shader_state state = {0};
      state.type = PIPE_SHADER_IR_NIR;
      state.ir.nir = make_texture_cube_array_fs(integer, integer ? 1.0f : 0.0f);
      state.report_compile_error = true;
      void *fs = ctx->create_fs_state(ctx, &state);
      texture_ok = cube_array_texture && cube_array_view && fs;
      if (texture_ok) {
         struct pipe_sampler_view *views[1] = {cube_array_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
         pipe_resource_reference(&cube_array_texture, NULL);
         gex_bind_fs(ctx, fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, cube_array_expected[integer],
            cube_array_labels[integer]);
      } else if (state.error_message) {
         fprintf(stderr, "FAIL gex-wgpu %s shader=%s\n",
                 cube_array_labels[integer], state.error_message);
      }
      free(state.error_message);
      if (fs)
         gex_delete_fs(ctx, fs);
      if (cube_array_view)
         pipe_sampler_view_reference(&cube_array_view, NULL);
      if (cube_array_texture)
         pipe_resource_reference(&cube_array_texture, NULL);
   }
   struct pipe_resource shadow_templ = templ;
   shadow_templ.format = PIPE_FORMAT_Z32_FLOAT;
   shadow_templ.bind = PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW;
   struct pipe_resource *shadow_texture =
      screen->resource_create(screen, &shadow_templ);
   struct pipe_surface shadow_surface = {0};
   pipe_reference_init(&shadow_surface.reference, 1);
   shadow_surface.format = shadow_templ.format;
   shadow_surface.texture = shadow_texture;
   if (shadow_texture)
      ctx->clear_depth_stencil(ctx, &shadow_surface, PIPE_CLEAR_DEPTH,
                               0.5, 0, 0, 0, 8, 8, false);
   struct pipe_sampler_view shadow_view_templ = view_templ;
   shadow_view_templ.format = PIPE_FORMAT_Z32_FLOAT;
   shadow_view_templ.u.tex.last_level = 0;
   struct pipe_sampler_view *shadow_view = shadow_texture
      ? ctx->create_sampler_view(ctx, shadow_texture, &shadow_view_templ) : NULL;
   struct pipe_sampler_state shadow_sampler_state = sampler_state;
   shadow_sampler_state.compare_mode = PIPE_TEX_COMPARE_R_TO_TEXTURE;
   shadow_sampler_state.compare_func = PIPE_FUNC_LEQUAL;
   void *shadow_sampler =
      ctx->create_sampler_state(ctx, &shadow_sampler_state);
   struct pipe_shader_state shadow_pass_state = {0}, shadow_fail_state = {0};
   shadow_pass_state.type = PIPE_SHADER_IR_NIR;
   shadow_pass_state.ir.nir = make_shadow_fs(0.25f);
   shadow_pass_state.report_compile_error = true;
   shadow_fail_state.type = PIPE_SHADER_IR_NIR;
   shadow_fail_state.ir.nir = make_shadow_fs(0.75f);
   shadow_fail_state.report_compile_error = true;
   void *shadow_pass_fs = ctx->create_fs_state(ctx, &shadow_pass_state);
   void *shadow_fail_fs = ctx->create_fs_state(ctx, &shadow_fail_state);
   if (texture_ok)
      texture_ok = shadow_texture && shadow_view && shadow_sampler &&
                   shadow_pass_fs && shadow_fail_fs;
   if (texture_ok) {
      const uint8_t white[4] = {255, 255, 255, 255};
      const uint8_t black[4] = {0, 0, 0, 255};
      void *samplers[1] = {shadow_sampler};
      struct pipe_sampler_view *views[1] = {shadow_view};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&shadow_texture, NULL);
      gex_bind_vs(ctx, vs);
      gex_bind_fs(ctx, shadow_pass_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      texture_ok = check_asymmetric_color(
         ctx, resource, false, false, white,
         "texture-shadow-lequal-accepted-lifetime");
      if (texture_ok) {
         gex_bind_fs(ctx, shadow_fail_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric_color(
            ctx, resource, false, false, black,
            "texture-shadow-lequal-rejected-lifetime");
      }
   }
   if (!texture_ok &&
       (shadow_pass_state.error_message || shadow_fail_state.error_message))
      fprintf(stderr, "FAIL gex-wgpu shadow shaders pass=%s fail=%s\n",
              shadow_pass_state.error_message
                 ? shadow_pass_state.error_message : "unknown",
              shadow_fail_state.error_message
                 ? shadow_fail_state.error_message : "unknown");
   free(shadow_pass_state.error_message);
   free(shadow_fail_state.error_message);
   if (shadow_pass_fs)
      gex_delete_fs(ctx, shadow_pass_fs);
   if (shadow_fail_fs)
      gex_delete_fs(ctx, shadow_fail_fs);
   if (shadow_view)
      pipe_sampler_view_reference(&shadow_view, NULL);
   if (shadow_texture)
      pipe_resource_reference(&shadow_texture, NULL);
   if (sample_sampler) {
      void *samplers[1] = {sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
   if (shadow_sampler)
      gex_delete_sampler(ctx, shadow_sampler);
   struct pipe_resource gradient_templ = sample_templ;
   gradient_templ.width0 = 4;
   gradient_templ.height0 = 4;
   gradient_templ.last_level = 2;
   struct pipe_resource *gradient_texture =
      screen->resource_create(screen, &gradient_templ);
   uint8_t gradient_mip0[64], gradient_mip1[16];
   const uint8_t gradient_mip2[4] = {0, 0, 255, 255};
   for (unsigned i = 0; i < 16; ++i) {
      gradient_mip0[4 * i + 0] = 255;
      gradient_mip0[4 * i + 1] = 0;
      gradient_mip0[4 * i + 2] = 0;
      gradient_mip0[4 * i + 3] = 255;
   }
   for (unsigned i = 0; i < 4; ++i) {
      gradient_mip1[4 * i + 0] = 0;
      gradient_mip1[4 * i + 1] = 255;
      gradient_mip1[4 * i + 2] = 0;
      gradient_mip1[4 * i + 3] = 255;
   }
   const struct pipe_box gradient_boxes[3] = {
      {.x = 0, .y = 0, .z = 0, .width = 4, .height = 4, .depth = 1},
      {.x = 0, .y = 0, .z = 0, .width = 2, .height = 2, .depth = 1},
      {.x = 0, .y = 0, .z = 0, .width = 1, .height = 1, .depth = 1},
   };
   if (gradient_texture) {
      ctx->texture_subdata(ctx, gradient_texture, 0, PIPE_MAP_WRITE,
                           &gradient_boxes[0], gradient_mip0, 16, 64);
      ctx->texture_subdata(ctx, gradient_texture, 1, PIPE_MAP_WRITE,
                           &gradient_boxes[1], gradient_mip1, 8, 16);
      ctx->texture_subdata(ctx, gradient_texture, 2, PIPE_MAP_WRITE,
                           &gradient_boxes[2], gradient_mip2, 4, 4);
   }
   struct pipe_sampler_view gradient_view_templ = view_templ;
   gradient_view_templ.u.tex.last_level = 2;
   struct pipe_sampler_view *gradient_view = gradient_texture
      ? ctx->create_sampler_view(ctx, gradient_texture, &gradient_view_templ)
      : NULL;
   struct pipe_sampler_state gradient_sampler_state = sampler_state;
   gradient_sampler_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
   gradient_sampler_state.max_lod = 2.0f;
   void *gradient_sampler =
      ctx->create_sampler_state(ctx, &gradient_sampler_state);
   struct pipe_sampler_state positive_bias_state = gradient_sampler_state;
   positive_bias_state.lod_bias = 1.0f;
   void *positive_bias_sampler =
      ctx->create_sampler_state(ctx, &positive_bias_state);
   struct pipe_sampler_state negative_bias_state = gradient_sampler_state;
   negative_bias_state.lod_bias = -1.0f;
   void *negative_bias_sampler =
      ctx->create_sampler_state(ctx, &negative_bias_state);
   struct pipe_shader_state gradient_vs_state = {0}, gradient_fs_state = {0},
      gradient_explicit_state = {0}, gradient_lod_state = {0},
      gradient_bias_state = {0};
   gradient_vs_state.type = PIPE_SHADER_IR_NIR;
   gradient_vs_state.ir.nir = make_gradient_texture_vs(8.0f);
   gradient_vs_state.report_compile_error = true;
   gradient_fs_state.type = PIPE_SHADER_IR_NIR;
   gradient_fs_state.ir.nir = make_texture_fs(true, 0.0f, 0.0f, -1.0f);
   gradient_fs_state.report_compile_error = true;
   gradient_explicit_state.type = PIPE_SHADER_IR_NIR;
   gradient_explicit_state.ir.nir = make_texture_gradient_fs();
   gradient_explicit_state.report_compile_error = true;
   gradient_lod_state.type = PIPE_SHADER_IR_NIR;
   gradient_lod_state.ir.nir =
      make_texture_fs(false, 0.5f, 0.5f, 0.0f);
   gradient_lod_state.report_compile_error = true;
   gradient_bias_state.type = PIPE_SHADER_IR_NIR;
   gradient_bias_state.ir.nir = make_texture_bias_fs(-1.0f);
   gradient_bias_state.report_compile_error = true;
   void *gradient_vs = ctx->create_vs_state(ctx, &gradient_vs_state);
   void *gradient_fs = ctx->create_fs_state(ctx, &gradient_fs_state);
   void *gradient_explicit_fs =
      ctx->create_fs_state(ctx, &gradient_explicit_state);
   void *gradient_lod_fs =
      ctx->create_fs_state(ctx, &gradient_lod_state);
   void *gradient_bias_fs =
      ctx->create_fs_state(ctx, &gradient_bias_state);
   if (texture_ok)
      texture_ok = gradient_texture && gradient_view && gradient_sampler &&
                   positive_bias_sampler && negative_bias_sampler &&
                   gradient_vs && gradient_fs && gradient_explicit_fs &&
                   gradient_lod_fs && gradient_bias_fs;
   if (texture_ok) {
      void *samplers[1] = {gradient_sampler};
      struct pipe_sampler_view *views[1] = {gradient_view};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      pipe_resource_reference(&gradient_texture, NULL);
      gex_bind_vs(ctx, gradient_vs);
      gex_bind_fs(ctx, gradient_fs);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      const uint8_t implicit_green[4] = {0, 255, 0, 255};
      texture_ok = check_uniform(
         ctx, resource, implicit_green,
         "texture-implicit-gradient-lod-one-lifetime");
      if (texture_ok) {
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, gradient_explicit_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         texture_ok = check_asymmetric(
            ctx, resource, false, false,
            "texture-explicit-gradient-lod-one-lifetime");
      }
      if (texture_ok) {
         gex_bind_vs(ctx, gradient_vs);
         gex_bind_fs(ctx, gradient_bias_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t red[4] = {255, 0, 0, 255};
         texture_ok = check_uniform(
            ctx, resource, red, "texture-shader-bias-only");
      }
      if (texture_ok) {
         void *samplers[1] = {positive_bias_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_vs(ctx, gradient_vs);
         gex_bind_fs(ctx, gradient_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t blue[4] = {0, 0, 255, 255};
         texture_ok = check_uniform(
            ctx, resource, blue,
            "texture-implicit-gradient-positive-sampler-bias");
      }
      if (texture_ok) {
         gex_bind_vs(ctx, gradient_vs);
         gex_bind_fs(ctx, gradient_bias_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t green[4] = {0, 255, 0, 255};
         texture_ok = check_uniform(
            ctx, resource, green,
            "texture-shader-plus-sampler-bias-composition");
      }
      if (texture_ok) {
         void *samplers[1] = {negative_bias_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_vs(ctx, gradient_vs);
         gex_bind_fs(ctx, gradient_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t red[4] = {255, 0, 0, 255};
         texture_ok = check_uniform(
            ctx, resource, red,
            "texture-implicit-gradient-negative-sampler-bias");
      }
      if (texture_ok) {
         void *samplers[1] = {positive_bias_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_vs(ctx, gradient_vs);
         gex_bind_fs(ctx, gradient_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t blue[4] = {0, 0, 255, 255};
         texture_ok = check_uniform(
            ctx, resource, blue,
            "texture-sampler-bias-variant-rebind");
      }
      if (texture_ok) {
         void *samplers[1] = {positive_bias_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
         gex_bind_vs(ctx, gradient_vs);
         gex_bind_fs(ctx, gradient_explicit_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t green[4] = {0, 255, 0, 255};
         texture_ok = check_uniform(
            ctx, resource, green,
            "texture-explicit-gradient-ignores-sampler-bias");
      }
      if (texture_ok) {
         gex_bind_fs(ctx, gradient_lod_fs);
         ctx->clear_render_target(ctx, &surface, &color,
                                  0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         const uint8_t green[4] = {0, 255, 0, 255};
         texture_ok = check_uniform(
            ctx, resource, green,
            "texture-explicit-lod-positive-sampler-bias");
      }
      if (gradient_sampler) {
         void *samplers[1] = {gradient_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
   }
   free(gradient_vs_state.error_message);
   free(gradient_fs_state.error_message);
   free(gradient_explicit_state.error_message);
   free(gradient_lod_state.error_message);
   free(gradient_bias_state.error_message);
   if (gradient_vs)
      gex_delete_vs(ctx, gradient_vs);
   if (gradient_fs)
      gex_delete_fs(ctx, gradient_fs);
   if (gradient_explicit_fs)
      gex_delete_fs(ctx, gradient_explicit_fs);
   if (gradient_lod_fs)
      gex_delete_fs(ctx, gradient_lod_fs);
   if (gradient_bias_fs)
      gex_delete_fs(ctx, gradient_bias_fs);
   if (gradient_view)
      pipe_sampler_view_reference(&gradient_view, NULL);
   if (gradient_texture)
      pipe_resource_reference(&gradient_texture, NULL);
   if (sample_sampler) {
      void *samplers[1] = {sample_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
   if (gradient_sampler)
      gex_delete_sampler(ctx, gradient_sampler);
   if (positive_bias_sampler)
      gex_delete_sampler(ctx, positive_bias_sampler);
   if (negative_bias_sampler)
      gex_delete_sampler(ctx, negative_bias_sampler);
   if (sample_view) {
      struct pipe_sampler_view *restored_view[1] = {sample_view};
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0,
                             restored_view);
   }

   free(texture_state.error_message);
   gex_bind_fs(ctx, NULL);
   for (unsigned i = 0; i < 2; ++i) {
      if (rect_fs[i])
         gex_delete_fs(ctx, rect_fs[i]);
   }
   if (texture_fs) gex_delete_fs(ctx, texture_fs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (rect_sampler) gex_delete_sampler(ctx, rect_sampler);
   if (sample_sampler) gex_delete_sampler(ctx, sample_sampler);
   if (!texture_ok) {
      ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
      if (sample_view) pipe_sampler_view_reference(&sample_view, NULL);
      if (sample_texture) pipe_resource_reference(&sample_texture, NULL);
      gex_bind_vs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return 13;
   }

   /* The center of the four texels must be their bilinear average.  This is a
    * separate shader and sampler object so pipeline-layout and state rebinding
    * bugs cannot hide behind the preceding nearest-filter draw. */
   sampler_state.min_img_filter = PIPE_TEX_FILTER_LINEAR;
   sampler_state.mag_img_filter = PIPE_TEX_FILTER_LINEAR;
   void *linear_sampler = ctx->create_sampler_state(ctx, &sampler_state);
   if (linear_sampler) {
      void *samplers[1] = {linear_sampler};
      gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
   }
   struct pipe_shader_state linear_state;
   memset(&linear_state, 0, sizeof(linear_state));
   linear_state.type = PIPE_SHADER_IR_NIR;
   linear_state.ir.nir = make_texture_fs(true, 0.0f, 0.0f, -1.0f);
   linear_state.report_compile_error = true;
   void *linear_fs = ctx->create_fs_state(ctx, &linear_state);
   struct pipe_shader_state texture_vs_state;
   memset(&texture_vs_state, 0, sizeof(texture_vs_state));
   texture_vs_state.type = PIPE_SHADER_IR_NIR;
   texture_vs_state.ir.nir = make_texture_coord_vs(0.5f, 0.5f);
   texture_vs_state.report_compile_error = true;
   void *texture_vs = ctx->create_vs_state(ctx, &texture_vs_state);
   bool linear_ok = linear_sampler && linear_fs && texture_vs;
   if (!linear_ok) {
      fprintf(stderr, "FAIL gex-wgpu texture linear setup fs=%s vs=%s\n",
              linear_state.error_message ? linear_state.error_message : "unknown",
              texture_vs_state.error_message ? texture_vs_state.error_message : "unknown");
   } else {
      const uint8_t average[4] = {128, 128, 128, 255};
      /* The sampler view owns the texture.  Drop the application's resource
       * reference before drawing to prove bound-view lifetime semantics. */
      pipe_resource_reference(&sample_texture, NULL);
      gex_bind_vs(ctx, texture_vs);
      gex_bind_fs(ctx, linear_fs);
      ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
      ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
      linear_ok = check_asymmetric_color(ctx, resource, false, false, average,
                                         "texture-linear-center");
   }
   free(linear_state.error_message);
   free(texture_vs_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (linear_fs) gex_delete_fs(ctx, linear_fs);
   if (texture_vs) gex_delete_vs(ctx, texture_vs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (linear_sampler) gex_delete_sampler(ctx, linear_sampler);

   bool repeat_ok = linear_ok;
   void *repeat_sampler = NULL;
   void *mirror_sampler = NULL;
   void *repeat_fs = NULL;
   struct pipe_shader_state repeat_state;
   memset(&repeat_state, 0, sizeof(repeat_state));
   if (repeat_ok) {
      sampler_state.wrap_s = PIPE_TEX_WRAP_REPEAT;
      sampler_state.wrap_t = PIPE_TEX_WRAP_REPEAT;
      sampler_state.wrap_r = PIPE_TEX_WRAP_REPEAT;
      sampler_state.min_img_filter = PIPE_TEX_FILTER_NEAREST;
      sampler_state.mag_img_filter = PIPE_TEX_FILTER_NEAREST;
      repeat_sampler = ctx->create_sampler_state(ctx, &sampler_state);
      if (repeat_sampler) {
         void *samplers[1] = {repeat_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
      repeat_state.type = PIPE_SHADER_IR_NIR;
      repeat_state.ir.nir = make_texture_fs(false, 1.25f, 0.25f, -1.0f);
      repeat_state.report_compile_error = true;
      repeat_fs = ctx->create_fs_state(ctx, &repeat_state);
      repeat_ok = repeat_sampler && repeat_fs;
      if (!repeat_ok) {
         fprintf(stderr, "FAIL gex-wgpu texture repeat setup shader=%s\n",
                 repeat_state.error_message ? repeat_state.error_message : "unknown");
      } else {
         const uint8_t red[4] = {255, 0, 0, 255};
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, repeat_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         repeat_ok = check_asymmetric_color(ctx, resource, false, false, red,
                                            "texture-repeat-nearest");
         if (repeat_ok) {
            sampler_state.wrap_s = PIPE_TEX_WRAP_MIRROR_REPEAT;
            sampler_state.wrap_t = PIPE_TEX_WRAP_MIRROR_REPEAT;
            sampler_state.wrap_r = PIPE_TEX_WRAP_MIRROR_REPEAT;
            mirror_sampler = ctx->create_sampler_state(ctx, &sampler_state);
            if (mirror_sampler) {
               void *samplers[1] = {mirror_sampler};
               gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1,
                                        samplers);
               const uint8_t green[4] = {0, 255, 0, 255};
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               repeat_ok = check_asymmetric_color(
                  ctx, resource, false, false, green,
                  "texture-mirror-repeat-nearest");
            } else {
               repeat_ok = false;
            }
         }
      }
   }
   free(repeat_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (repeat_fs) gex_delete_fs(ctx, repeat_fs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (mirror_sampler) gex_delete_sampler(ctx, mirror_sampler);
   if (repeat_sampler) gex_delete_sampler(ctx, repeat_sampler);

   bool mip_ok = repeat_ok;
   void *mip_sampler = NULL;
   void *mip_fs = NULL;
   struct pipe_shader_state mip_state;
   memset(&mip_state, 0, sizeof(mip_state));
   if (mip_ok) {
      sampler_state.wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST;
      sampler_state.max_lod = 1.0f;
      mip_sampler = ctx->create_sampler_state(ctx, &sampler_state);
      if (mip_sampler) {
         void *samplers[1] = {mip_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
      mip_state.type = PIPE_SHADER_IR_NIR;
      mip_state.ir.nir = make_texture_fs(false, 0.5f, 0.5f, 1.0f);
      mip_state.report_compile_error = true;
      mip_fs = ctx->create_fs_state(ctx, &mip_state);
      mip_ok = mip_sampler && mip_fs;
      if (!mip_ok) {
         fprintf(stderr, "FAIL gex-wgpu texture mip setup shader=%s\n",
                 mip_state.error_message ? mip_state.error_message : "unknown");
      } else {
         const uint8_t magenta[4] = {255, 0, 255, 255};
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, mip_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         mip_ok = check_asymmetric_color(ctx, resource, false, false, magenta,
                                         "texture-explicit-mip-one");
      }
   }
   free(mip_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (mip_fs) gex_delete_fs(ctx, mip_fs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (mip_sampler) gex_delete_sampler(ctx, mip_sampler);

   bool trilinear_ok = mip_ok;
   void *trilinear_sampler = NULL;
   void *trilinear_fs = NULL;
   struct pipe_shader_state trilinear_state;
   memset(&trilinear_state, 0, sizeof(trilinear_state));
   if (trilinear_ok) {
      sampler_state.min_img_filter = PIPE_TEX_FILTER_LINEAR;
      sampler_state.mag_img_filter = PIPE_TEX_FILTER_LINEAR;
      sampler_state.min_mip_filter = PIPE_TEX_MIPFILTER_LINEAR;
      sampler_state.max_lod = 1.0f;
      trilinear_sampler = ctx->create_sampler_state(ctx, &sampler_state);
      if (trilinear_sampler) {
         void *samplers[1] = {trilinear_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
      trilinear_state.type = PIPE_SHADER_IR_NIR;
      trilinear_state.ir.nir =
         make_texture_fs(false, 0.25f, 0.25f, 0.5f);
      trilinear_state.report_compile_error = true;
      trilinear_fs = ctx->create_fs_state(ctx, &trilinear_state);
      trilinear_ok = trilinear_sampler && trilinear_fs;
      if (!trilinear_ok) {
         fprintf(stderr, "FAIL gex-wgpu trilinear setup shader=%s\n",
                 trilinear_state.error_message
                    ? trilinear_state.error_message : "unknown");
      } else {
         const uint8_t red_magenta_half[4] = {255, 0, 128, 255};
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, trilinear_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         trilinear_ok = check_asymmetric_color(
            ctx, resource, false, false, red_magenta_half,
            "texture-trilinear-half-lod");
      }
   }
   free(trilinear_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (trilinear_fs) gex_delete_fs(ctx, trilinear_fs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (trilinear_sampler)
      gex_delete_sampler(ctx, trilinear_sampler);

   bool border_ok = trilinear_ok;
   void *border_sampler = NULL;
   void *border_sampler_yellow = NULL;
   void *border_fs = NULL;
   void *border_vs = NULL;
   void *dead_border_fs = NULL;
   struct pipe_shader_state border_state;
   struct pipe_shader_state border_vs_state;
   struct pipe_shader_state dead_border_state;
   memset(&border_state, 0, sizeof(border_state));
   memset(&border_vs_state, 0, sizeof(border_vs_state));
   memset(&dead_border_state, 0, sizeof(dead_border_state));
   if (border_ok) {
      sampler_state.wrap_s = PIPE_TEX_WRAP_CLAMP_TO_BORDER;
      sampler_state.wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.min_mip_filter = PIPE_TEX_MIPFILTER_NONE;
      sampler_state.max_lod = 0.0f;
      sampler_state.border_color.f[0] = 0.0f;
      sampler_state.border_color.f[1] = 1.0f;
      sampler_state.border_color.f[2] = 1.0f;
      sampler_state.border_color.f[3] = 1.0f;
      border_sampler = ctx->create_sampler_state(ctx, &sampler_state);
      if (border_sampler) {
         void *samplers[1] = {border_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
      border_state.type = PIPE_SHADER_IR_NIR;
      border_state.ir.nir = make_texture_fs(true, 0.0f, 0.0f, -1.0f);
      border_state.report_compile_error = true;
      border_fs = ctx->create_fs_state(ctx, &border_state);
      border_vs_state.type = PIPE_SHADER_IR_NIR;
      border_vs_state.ir.nir = make_texture_coord_vs(-0.25f, 0.25f);
      border_vs_state.report_compile_error = true;
      border_vs = ctx->create_vs_state(ctx, &border_vs_state);
      border_ok = border_sampler && border_fs && border_vs;
      if (!border_ok) {
         fprintf(stderr, "FAIL gex-wgpu texture border setup fs=%s vs=%s\n",
                 border_state.error_message ? border_state.error_message : "unknown",
                 border_vs_state.error_message ? border_vs_state.error_message : "unknown");
      } else {
         const uint8_t cyan[4] = {0, 255, 255, 255};
         gex_bind_vs(ctx, border_vs);
         gex_bind_fs(ctx, border_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         border_ok = check_asymmetric_color(ctx, resource, false, false, cyan,
                                            "texture-nearest-border-cyan");
         if (border_ok) {
            sampler_state.border_color.f[0] = 1.0f;
            sampler_state.border_color.f[1] = 1.0f;
            sampler_state.border_color.f[2] = 0.0f;
            border_sampler_yellow =
               ctx->create_sampler_state(ctx, &sampler_state);
            if (border_sampler_yellow) {
               void *samplers[1] = {border_sampler_yellow};
               gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1,
                                        samplers);
               const uint8_t yellow[4] = {255, 255, 0, 255};
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               border_ok = check_asymmetric_color(
                  ctx, resource, false, false, yellow,
                  "texture-nearest-border-rebind-yellow");
               if (border_ok) {
                  void *samplers[1] = {border_sampler};
                  gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1,
                                           samplers);
                  const uint8_t cyan[4] = {0, 255, 255, 255};
                  ctx->clear_render_target(ctx, &surface, &color,
                                           0, 0, 8, 8, false);
                  ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
                  border_ok = check_asymmetric_color(
                     ctx, resource, false, false, cyan,
                     "texture-nearest-border-cache-reuse-cyan");
               }
               if (border_ok) {
                  void *samplers[1] = {border_sampler_yellow};
                  gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1,
                                           samplers);
                  dead_border_state.type = PIPE_SHADER_IR_NIR;
                  dead_border_state.ir.nir =
                     make_texture_fs(false, -0.25f, 0.25f, -1.0f);
                  dead_border_state.report_compile_error = true;
                  dead_border_fs =
                     ctx->create_fs_state(ctx, &dead_border_state);
                  border_ok = dead_border_fs != NULL;
                  if (border_ok) {
                     gex_bind_vs(ctx, vs);
                     gex_bind_fs(ctx, dead_border_fs);
                     ctx->clear_render_target(ctx, &surface, &color,
                                              0, 0, 8, 8, false);
                     ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
                     border_ok = check_asymmetric_color(
                        ctx, resource, false, false, yellow,
                        "texture-dead-sample-explicit-layout");
                  }
               }
            } else {
               border_ok = false;
            }
         }
      }
   }
   free(border_state.error_message);
   free(border_vs_state.error_message);
   free(dead_border_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (border_fs) gex_delete_fs(ctx, border_fs);
   if (dead_border_fs) gex_delete_fs(ctx, dead_border_fs);
   if (border_vs) gex_delete_vs(ctx, border_vs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (border_sampler_yellow)
      gex_delete_sampler(ctx, border_sampler_yellow);
   if (border_sampler) gex_delete_sampler(ctx, border_sampler);

   bool linear_border_ok = border_ok;
   void *linear_border_sampler = NULL;
   void *linear_border_fs = NULL;
   void *linear_border_vs = NULL;
   struct pipe_shader_state linear_border_fs_state;
   struct pipe_shader_state linear_border_vs_state;
   memset(&linear_border_fs_state, 0, sizeof(linear_border_fs_state));
   memset(&linear_border_vs_state, 0, sizeof(linear_border_vs_state));
   if (linear_border_ok) {
      sampler_state.wrap_s = PIPE_TEX_WRAP_CLAMP_TO_BORDER;
      sampler_state.wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.min_img_filter = PIPE_TEX_FILTER_LINEAR;
      sampler_state.mag_img_filter = PIPE_TEX_FILTER_LINEAR;
      sampler_state.border_color.f[0] = 1.0f;
      sampler_state.border_color.f[1] = 0.0f;
      sampler_state.border_color.f[2] = 1.0f;
      sampler_state.border_color.f[3] = 1.0f;
      linear_border_sampler =
         ctx->create_sampler_state(ctx, &sampler_state);
      if (linear_border_sampler) {
         void *samplers[1] = {linear_border_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
      linear_border_fs_state.type = PIPE_SHADER_IR_NIR;
      linear_border_fs_state.ir.nir =
         make_texture_fs(true, 0.0f, 0.0f, -1.0f);
      linear_border_fs_state.report_compile_error = true;
      linear_border_fs =
         ctx->create_fs_state(ctx, &linear_border_fs_state);
      linear_border_vs_state.type = PIPE_SHADER_IR_NIR;
      linear_border_vs_state.ir.nir = make_texture_coord_vs(0.0f, 0.25f);
      linear_border_vs_state.report_compile_error = true;
      linear_border_vs =
         ctx->create_vs_state(ctx, &linear_border_vs_state);
      linear_border_ok =
         linear_border_sampler && linear_border_fs && linear_border_vs;
      if (linear_border_ok) {
         const uint8_t red_magenta_half[4] = {255, 0, 128, 255};
         gex_bind_vs(ctx, linear_border_vs);
         gex_bind_fs(ctx, linear_border_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         linear_border_ok = check_asymmetric_color(
            ctx, resource, false, false, red_magenta_half,
            "texture-linear-border-half-weight");
      }
   }
   free(linear_border_fs_state.error_message);
   free(linear_border_vs_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (linear_border_fs)
      gex_delete_fs(ctx, linear_border_fs);
   if (linear_border_vs)
      gex_delete_vs(ctx, linear_border_vs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (linear_border_sampler)
      gex_delete_sampler(ctx, linear_border_sampler);

   bool swizzle_ok = linear_border_ok;
   struct pipe_sampler_view *swizzle_view = NULL;
   void *swizzle_sampler = NULL;
   void *swizzle_fs = NULL;
   struct pipe_shader_state swizzle_state;
   memset(&swizzle_state, 0, sizeof(swizzle_state));
   if (swizzle_ok) {
      struct pipe_sampler_view swizzle_templ = view_templ;
      swizzle_templ.swizzle_r = PIPE_SWIZZLE_Z;
      swizzle_templ.swizzle_g = PIPE_SWIZZLE_Y;
      swizzle_templ.swizzle_b = PIPE_SWIZZLE_X;
      swizzle_view = ctx->create_sampler_view(
         ctx, sample_view->texture, &swizzle_templ);
      if (swizzle_view) {
         struct pipe_sampler_view *views[1] = {swizzle_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      }
      memset(&sampler_state.border_color, 0,
             sizeof(sampler_state.border_color));
      sampler_state.wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      sampler_state.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
      swizzle_sampler = ctx->create_sampler_state(ctx, &sampler_state);
      if (swizzle_sampler) {
         void *samplers[1] = {swizzle_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
      swizzle_state.type = PIPE_SHADER_IR_NIR;
      swizzle_state.ir.nir =
         make_texture_fs(false, 0.25f, 0.25f, -1.0f);
      swizzle_state.report_compile_error = true;
      swizzle_fs = ctx->create_fs_state(ctx, &swizzle_state);
      swizzle_ok = swizzle_view && swizzle_sampler && swizzle_fs;
      if (!swizzle_ok) {
         fprintf(stderr, "FAIL gex-wgpu texture swizzle setup shader=%s\n",
                 swizzle_state.error_message
                    ? swizzle_state.error_message : "unknown");
      } else {
         const uint8_t blue_texel[4] = {0, 0, 255, 255};
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, swizzle_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         swizzle_ok = check_asymmetric_color(ctx, resource, false, false,
                                             blue_texel,
                                             "texture-view-bgra-swizzle");
      }
   }
   free(swizzle_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (swizzle_fs) gex_delete_fs(ctx, swizzle_fs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (swizzle_sampler) gex_delete_sampler(ctx, swizzle_sampler);

   bool base_level_ok = swizzle_ok;
   struct pipe_sampler_view *base_level_view = NULL;
   void *base_level_sampler = NULL;
   void *base_level_fs = NULL;
   struct pipe_shader_state base_level_state;
   memset(&base_level_state, 0, sizeof(base_level_state));
   if (base_level_ok) {
      struct pipe_sampler_view base_level_templ = view_templ;
      base_level_templ.u.tex.first_level = 1;
      base_level_templ.u.tex.last_level = 1;
      base_level_view = ctx->create_sampler_view(
         ctx, sample_view->texture, &base_level_templ);
      if (base_level_view) {
         struct pipe_sampler_view *views[1] = {base_level_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      }
      sampler_state.min_img_filter = PIPE_TEX_FILTER_NEAREST;
      sampler_state.mag_img_filter = PIPE_TEX_FILTER_NEAREST;
      sampler_state.min_mip_filter = PIPE_TEX_MIPFILTER_NONE;
      sampler_state.max_lod = 0.0f;
      base_level_sampler = ctx->create_sampler_state(ctx, &sampler_state);
      if (base_level_sampler) {
         void *samplers[1] = {base_level_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
      base_level_state.type = PIPE_SHADER_IR_NIR;
      base_level_state.ir.nir =
         make_texture_fs(false, 0.5f, 0.5f, 0.0f);
      base_level_state.report_compile_error = true;
      base_level_fs = ctx->create_fs_state(ctx, &base_level_state);
      base_level_ok = base_level_view && base_level_sampler && base_level_fs;
      if (base_level_ok) {
         const uint8_t magenta[4] = {255, 0, 255, 255};
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, base_level_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         base_level_ok = check_asymmetric_color(
            ctx, resource, false, false, magenta,
            "texture-view-nonzero-base-level");
      }
   }
   free(base_level_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (base_level_fs) gex_delete_fs(ctx, base_level_fs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (base_level_sampler)
      gex_delete_sampler(ctx, base_level_sampler);

   bool srgb_ok = base_level_ok;
   struct pipe_resource *srgb_texture = NULL;
   struct pipe_sampler_view *srgb_view = NULL;
   void *srgb_sampler = NULL;
   void *srgb_fs = NULL;
   struct pipe_shader_state srgb_state;
   memset(&srgb_state, 0, sizeof(srgb_state));
   if (srgb_ok) {
      struct pipe_resource srgb_templ = sample_templ;
      srgb_templ.format = PIPE_FORMAT_R8G8B8A8_SRGB;
      srgb_templ.width0 = 1;
      srgb_templ.height0 = 1;
      srgb_templ.last_level = 0;
      srgb_texture = screen->resource_create(screen, &srgb_templ);
      const uint8_t encoded_gray[4] = {128, 128, 128, 255};
      if (srgb_texture)
         ctx->texture_subdata(ctx, srgb_texture, 0, PIPE_MAP_WRITE,
                              &mip_box, encoded_gray, 4, 4);
      struct pipe_sampler_view srgb_view_templ = view_templ;
      srgb_view_templ.format = PIPE_FORMAT_R8G8B8A8_SRGB;
      srgb_view_templ.u.tex.last_level = 0;
      srgb_view = srgb_texture
         ? ctx->create_sampler_view(ctx, srgb_texture, &srgb_view_templ) : NULL;
      if (srgb_view) {
         struct pipe_sampler_view *views[1] = {srgb_view};
         ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 1, 0, views);
      }
      srgb_sampler = ctx->create_sampler_state(ctx, &sampler_state);
      if (srgb_sampler) {
         void *samplers[1] = {srgb_sampler};
         gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 1, samplers);
      }
      srgb_state.type = PIPE_SHADER_IR_NIR;
      srgb_state.ir.nir = make_texture_fs(false, 0.5f, 0.5f, -1.0f);
      srgb_state.report_compile_error = true;
      srgb_fs = ctx->create_fs_state(ctx, &srgb_state);
      srgb_ok = srgb_texture && srgb_view && srgb_sampler && srgb_fs;
      if (srgb_ok) {
         const uint8_t decoded_gray[4] = {55, 55, 55, 255};
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, srgb_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         srgb_ok = check_asymmetric_color(ctx, resource, false, false,
                                          decoded_gray,
                                          "texture-srgb-decode");
      }
   }
   free(srgb_state.error_message);
   gex_bind_fs(ctx, NULL);
   gex_bind_vs(ctx, NULL);
   if (srgb_fs) gex_delete_fs(ctx, srgb_fs);
   gex_bind_samplers(ctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   if (srgb_sampler) gex_delete_sampler(ctx, srgb_sampler);
   ctx->set_sampler_views(ctx, MESA_SHADER_FRAGMENT, 0, 0, 1, NULL);
   if (srgb_view) pipe_sampler_view_reference(&srgb_view, NULL);
   if (srgb_texture) pipe_resource_reference(&srgb_texture, NULL);
   if (base_level_view)
      pipe_sampler_view_reference(&base_level_view, NULL);
   if (swizzle_view) pipe_sampler_view_reference(&swizzle_view, NULL);
   if (sample_view) pipe_sampler_view_reference(&sample_view, NULL);
   if (sample_texture) pipe_resource_reference(&sample_texture, NULL);

   bool depth_ok = srgb_ok;
   struct pipe_resource *depth_texture = NULL;
   void *dsa = NULL, *dsa_no_write = NULL, *dsa_greater = NULL;
   void *depth_vs = NULL, *far_vs = NULL, *near_vs = NULL;
   void *red_fs = NULL;
   if (depth_ok) {
      struct pipe_resource depth_templ = templ;
      depth_templ.format = PIPE_FORMAT_Z32_FLOAT;
      depth_templ.bind = PIPE_BIND_DEPTH_STENCIL;
      depth_texture = screen->resource_create(screen, &depth_templ);
      struct pipe_surface depth_surface;
      memset(&depth_surface, 0, sizeof(depth_surface));
      pipe_reference_init(&depth_surface.reference, 1);
      depth_surface.format = PIPE_FORMAT_Z32_FLOAT;
      depth_surface.texture = depth_texture;
      framebuffer.zsbuf = depth_surface;
      if (depth_texture)
         ctx->set_framebuffer_state(ctx, &framebuffer);
      struct pipe_depth_stencil_alpha_state dsa_state;
      memset(&dsa_state, 0, sizeof(dsa_state));
      dsa_state.depth_enabled = 1;
      dsa_state.depth_writemask = 1;
      dsa_state.depth_func = PIPE_FUNC_LESS;
      dsa = ctx->create_depth_stencil_alpha_state(ctx, &dsa_state);
      dsa_state.depth_writemask = 0;
      dsa_no_write = ctx->create_depth_stencil_alpha_state(ctx, &dsa_state);
      dsa_state.depth_writemask = 1;
      dsa_state.depth_func = PIPE_FUNC_GREATER;
      dsa_greater = ctx->create_depth_stencil_alpha_state(ctx, &dsa_state);
      if (dsa)
         gex_bind_dsa(ctx, dsa);
      struct pipe_shader_state depth_vs_state = {0};
      struct pipe_shader_state far_vs_state = {0};
      struct pipe_shader_state near_vs_state = {0};
      struct pipe_shader_state red_fs_state = {0};
      depth_vs_state.type = far_vs_state.type = near_vs_state.type =
         red_fs_state.type = PIPE_SHADER_IR_NIR;
      depth_vs_state.ir.nir = make_vs(0.0f, true);
      far_vs_state.ir.nir = make_vs(0.5f, true);
      near_vs_state.ir.nir = make_vs(-0.5f, true);
      red_fs_state.ir.nir = make_color_fs(1.0f, 0.0f, 0.0f, 1.0f);
      depth_vs = ctx->create_vs_state(ctx, &depth_vs_state);
      far_vs = ctx->create_vs_state(ctx, &far_vs_state);
      near_vs = ctx->create_vs_state(ctx, &near_vs_state);
      red_fs = ctx->create_fs_state(ctx, &red_fs_state);
      depth_ok = depth_texture && dsa && dsa_no_write && dsa_greater &&
                 depth_vs && far_vs && near_vs && red_fs;
      if (depth_ok) {
         const uint8_t red[4] = {255, 0, 0, 255};
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->clear_depth_stencil(ctx, &depth_surface, PIPE_CLEAR_DEPTH,
                                  1.0, 0, 0, 0, 8, 8, false);
         gex_bind_vs(ctx, depth_vs);
         gex_bind_fs(ctx, fs);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         gex_bind_vs(ctx, far_vs);
         gex_bind_fs(ctx, red_fs);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         depth_ok = check_uniform(ctx, resource, green,
                                  "depth-far-fragment-rejected");
         if (depth_ok) {
            gex_bind_vs(ctx, near_vs);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            depth_ok = check_uniform(ctx, resource, red,
                                     "depth-near-fragment-accepted");
         }
         if (depth_ok) {
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->clear_depth_stencil(ctx, &depth_surface, PIPE_CLEAR_DEPTH,
                                     1.0, 0, 0, 0, 8, 8, false);
            gex_bind_dsa(ctx, dsa_no_write);
            gex_bind_vs(ctx, depth_vs);
            gex_bind_fs(ctx, fs);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            gex_bind_dsa(ctx, dsa);
            gex_bind_vs(ctx, far_vs);
            gex_bind_fs(ctx, red_fs);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            depth_ok = check_uniform(ctx, resource, red,
                                     "depth-write-mask-disabled");
         }
         if (depth_ok) {
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->clear_depth_stencil(ctx, &depth_surface, PIPE_CLEAR_DEPTH,
                                     0.0, 0, 0, 0, 8, 8, false);
            gex_bind_dsa(ctx, dsa_greater);
            gex_bind_vs(ctx, depth_vs);
            gex_bind_fs(ctx, fs);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            gex_bind_vs(ctx, far_vs);
            gex_bind_fs(ctx, red_fs);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            depth_ok = check_uniform(ctx, resource, red,
                                     "depth-greater-comparison");
         }
      }
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      gex_bind_dsa(ctx, base_dsa);
      if (depth_vs) gex_delete_vs(ctx, depth_vs);
      if (far_vs) gex_delete_vs(ctx, far_vs);
      if (near_vs) gex_delete_vs(ctx, near_vs);
      if (red_fs) gex_delete_fs(ctx, red_fs);
      if (dsa) gex_delete_dsa(ctx, dsa);
      if (dsa_no_write)
         gex_delete_dsa(ctx, dsa_no_write);
      if (dsa_greater)
         gex_delete_dsa(ctx, dsa_greater);
      framebuffer.zsbuf.texture = NULL;
      ctx->set_framebuffer_state(ctx, &framebuffer);
      if (depth_texture) pipe_resource_reference(&depth_texture, NULL);
   }

   bool stencil_ok = depth_ok;
   struct pipe_resource *stencil_texture = NULL;
   void *stencil_write = NULL, *stencil_equal = NULL;
   void *stencil_vs = NULL, *stencil_green_fs = NULL, *stencil_red_fs = NULL;
   if (stencil_ok) {
      struct pipe_resource stencil_templ = templ;
      stencil_templ.format = PIPE_FORMAT_Z24_UNORM_S8_UINT;
      stencil_templ.bind = PIPE_BIND_DEPTH_STENCIL;
      stencil_texture = screen->resource_create(screen, &stencil_templ);
      struct pipe_surface stencil_surface;
      memset(&stencil_surface, 0, sizeof(stencil_surface));
      pipe_reference_init(&stencil_surface.reference, 1);
      stencil_surface.format = PIPE_FORMAT_Z24_UNORM_S8_UINT;
      stencil_surface.texture = stencil_texture;
      framebuffer.zsbuf = stencil_surface;
      if (stencil_texture)
         ctx->set_framebuffer_state(ctx, &framebuffer);

      struct pipe_depth_stencil_alpha_state stencil_state;
      memset(&stencil_state, 0, sizeof(stencil_state));
      stencil_state.stencil[0].enabled = 1;
      stencil_state.stencil[0].func = PIPE_FUNC_ALWAYS;
      stencil_state.stencil[0].fail_op = PIPE_STENCIL_OP_KEEP;
      stencil_state.stencil[0].zfail_op = PIPE_STENCIL_OP_KEEP;
      stencil_state.stencil[0].zpass_op = PIPE_STENCIL_OP_REPLACE;
      stencil_state.stencil[0].valuemask = 0xff;
      stencil_state.stencil[0].writemask = 0x0f;
      stencil_write = ctx->create_depth_stencil_alpha_state(ctx,
                                                            &stencil_state);
      stencil_state.stencil[0].func = PIPE_FUNC_EQUAL;
      stencil_state.stencil[0].zpass_op = PIPE_STENCIL_OP_KEEP;
      stencil_state.stencil[0].writemask = 0;
      stencil_equal = ctx->create_depth_stencil_alpha_state(ctx,
                                                            &stencil_state);

      struct pipe_shader_state stencil_vs_state = {0};
      struct pipe_shader_state stencil_green_fs_state = {0};
      struct pipe_shader_state stencil_red_fs_state = {0};
      stencil_vs_state.type = stencil_green_fs_state.type =
         stencil_red_fs_state.type = PIPE_SHADER_IR_NIR;
      stencil_vs_state.ir.nir = make_vs(0.0f, true);
      stencil_green_fs_state.ir.nir = make_color_fs(0.0f, 1.0f, 0.0f, 1.0f);
      stencil_red_fs_state.ir.nir = make_color_fs(1.0f, 0.0f, 0.0f, 1.0f);
      stencil_vs = ctx->create_vs_state(ctx, &stencil_vs_state);
      stencil_green_fs = ctx->create_fs_state(ctx, &stencil_green_fs_state);
      stencil_red_fs = ctx->create_fs_state(ctx, &stencil_red_fs_state);
      stencil_ok = stencil_texture && stencil_write && stencil_equal &&
                   stencil_vs && stencil_green_fs && stencil_red_fs;
      if (stencil_ok) {
         const uint8_t red[4] = {255, 0, 0, 255};
         struct pipe_stencil_ref ref = {{0x05, 0}};
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->clear_depth_stencil(ctx, &stencil_surface,
                                  PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL,
                                  1.0, 0xa0, 0, 0, 8, 8, false);
         ctx->set_stencil_ref(ctx, ref);
         gex_bind_dsa(ctx, stencil_write);
         gex_bind_vs(ctx, stencil_vs);
         gex_bind_fs(ctx, stencil_green_fs);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);

         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ref.ref_value[0] = 0xa5;
         ctx->set_stencil_ref(ctx, ref);
         gex_bind_dsa(ctx, stencil_equal);
         gex_bind_fs(ctx, stencil_red_fs);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         stencil_ok = check_uniform(ctx, resource, red,
                                    "stencil-masked-replace-equal");
         if (stencil_ok) {
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ref.ref_value[0] = 0x05;
            ctx->set_stencil_ref(ctx, ref);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            stencil_ok = check_uniform(ctx, resource, blue,
                                       "stencil-full-byte-mismatch-rejected");
         }
      }
      gex_bind_dsa(ctx, base_dsa);
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      if (stencil_vs) gex_delete_vs(ctx, stencil_vs);
      if (stencil_green_fs) gex_delete_fs(ctx, stencil_green_fs);
      if (stencil_red_fs) gex_delete_fs(ctx, stencil_red_fs);
      if (stencil_write)
         gex_delete_dsa(ctx, stencil_write);
      if (stencil_equal)
         gex_delete_dsa(ctx, stencil_equal);
      framebuffer.zsbuf.texture = NULL;
      ctx->set_framebuffer_state(ctx, &framebuffer);
      if (stencil_texture) pipe_resource_reference(&stencil_texture, NULL);
   }

   bool blend_ok = stencil_ok;
   void *blend = NULL, *color_mask = NULL, *additive = NULL, *reverse = NULL;
   void *constant_blend = NULL, *blend_vs = NULL, *blend_fs = NULL;
   void *gray_fs = NULL, *quarter_fs = NULL, *white_fs = NULL;
   if (blend_ok) {
      struct pipe_blend_state blend_state;
      memset(&blend_state, 0, sizeof(blend_state));
      blend_state.rt[0].blend_enable = 1;
      blend_state.rt[0].rgb_func = PIPE_BLEND_ADD;
      blend_state.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_SRC_ALPHA;
      blend_state.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_INV_SRC_ALPHA;
      blend_state.rt[0].alpha_func = PIPE_BLEND_ADD;
      blend_state.rt[0].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
      blend_state.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_INV_SRC_ALPHA;
      blend_state.rt[0].colormask = PIPE_MASK_RGBA;
      blend = ctx->create_blend_state(ctx, &blend_state);
      struct pipe_shader_state blend_vs_state = {0};
      struct pipe_shader_state blend_fs_state = {0};
      blend_vs_state.type = blend_fs_state.type = PIPE_SHADER_IR_NIR;
      blend_vs_state.ir.nir = make_vs(0.0f, true);
      blend_fs_state.ir.nir = make_color_fs(1.0f, 0.0f, 0.0f, 0.5f);
      struct pipe_shader_state gray_fs_state = {0};
      gray_fs_state.type = PIPE_SHADER_IR_NIR;
      gray_fs_state.ir.nir = make_color_fs(0.3f, 0.3f, 0.3f, 0.3f);
      struct pipe_shader_state quarter_fs_state = {0};
      quarter_fs_state.type = PIPE_SHADER_IR_NIR;
      quarter_fs_state.ir.nir = make_color_fs(0.25f, 0.25f, 0.25f, 0.25f);
      struct pipe_shader_state white_fs_state = {0};
      white_fs_state.type = PIPE_SHADER_IR_NIR;
      white_fs_state.ir.nir = make_color_fs(1.0f, 1.0f, 1.0f, 1.0f);
      blend_vs = ctx->create_vs_state(ctx, &blend_vs_state);
      blend_fs = ctx->create_fs_state(ctx, &blend_fs_state);
      gray_fs = ctx->create_fs_state(ctx, &gray_fs_state);
      quarter_fs = ctx->create_fs_state(ctx, &quarter_fs_state);
      white_fs = ctx->create_fs_state(ctx, &white_fs_state);
      blend_ok = blend && blend_vs && blend_fs && gray_fs && quarter_fs && white_fs;
      if (blend_ok) {
         /* src-alpha-over onto the {64,128,191} clear lands exactly on an
          * 8-bit rounding tie in two channels:
          *   R = 0.5 + (64/255)*0.5  -> 159.5
          *   B =       (191/255)*0.5 ->  95.5
          * In float32 the two ties fall on opposite sides, and which way a
          * tie goes is a property of the GPU, not of this driver: SwiftShader
          * returns {160,64,96}, the RX 6500 XT returns {159,64,96}. Both are
          * correct to within the 1 ULP that blending is specified to. Only
          * this case is affected -- every other corpus expectation is an
          * exact value, and they stay exact. */
         const uint8_t blended[4] = {160, 64, 96, 255};
         gex_bind_blend(ctx, blend);
         gex_bind_vs(ctx, blend_vs);
         gex_bind_fs(ctx, blend_fs);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         blend_ok = check_uniform_tolerance(ctx, resource, blended, 1,
                                            "blend-src-alpha-over");
         if (blend_ok) {
            struct pipe_blend_state mask_state;
            memset(&mask_state, 0, sizeof(mask_state));
            mask_state.rt[0].colormask = PIPE_MASK_R;
            color_mask = ctx->create_blend_state(ctx, &mask_state);
            blend_ok = color_mask != NULL;
            if (blend_ok) {
               const uint8_t red_only[4] = {255, 128, 191, 255};
               gex_bind_blend(ctx, color_mask);
               ctx->clear_render_target(ctx, &surface, &color,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               blend_ok = check_uniform(ctx, resource, red_only,
                                        "color-write-mask-red");
            }
         }
         if (blend_ok) {
            struct pipe_blend_state arithmetic_state;
            memset(&arithmetic_state, 0, sizeof(arithmetic_state));
            arithmetic_state.rt[0].blend_enable = 1;
            arithmetic_state.rt[0].rgb_func = PIPE_BLEND_ADD;
            arithmetic_state.rt[0].alpha_func = PIPE_BLEND_ADD;
            arithmetic_state.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_ONE;
            arithmetic_state.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_ONE;
            arithmetic_state.rt[0].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
            arithmetic_state.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_ONE;
            arithmetic_state.rt[0].colormask = PIPE_MASK_RGBA;
            additive = ctx->create_blend_state(ctx, &arithmetic_state);
            arithmetic_state.rt[0].rgb_func = PIPE_BLEND_REVERSE_SUBTRACT;
            arithmetic_state.rt[0].alpha_func = PIPE_BLEND_REVERSE_SUBTRACT;
            reverse = ctx->create_blend_state(ctx, &arithmetic_state);
            blend_ok = additive && reverse;
            if (blend_ok) {
               const union pipe_color_union gray_clear =
                  {.f = {0.2f, 0.2f, 0.2f, 0.2f}};
               /* 0.2 clear stores as 51; 51/255 + 0.3 is exactly 0.5, so
                * every channel lands on the 127.5 tie -- the same 1 ULP
                * hardware/software split as blend-src-alpha-over above.
                * blend-reverse-subtract and blend-constant-color below have
                * always allowed 1 for this reason; these two were the
                * outliers. */
               const uint8_t half[4] = {128, 128, 128, 128};
               gex_bind_fs(ctx, gray_fs);
               gex_bind_blend(ctx, additive);
               ctx->clear_render_target(ctx, &surface, &gray_clear,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               blend_ok = check_uniform_tolerance(ctx, resource, half, 1,
                                                  "blend-additive-one-one");
               if (blend_ok) {
                  const union pipe_color_union high_clear =
                     {.f = {0.75f, 0.75f, 0.75f, 0.75f}};
                  gex_bind_blend(ctx, reverse);
                  gex_bind_fs(ctx, quarter_fs);
                  ctx->clear_render_target(ctx, &surface, &high_clear,
                                           0, 0, 8, 8, false);
                  ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
                  blend_ok = check_uniform_tolerance(
                     ctx, resource, half, 1, "blend-reverse-subtract");
               }
            }
         }
         if (blend_ok) {
            struct pipe_blend_state constant_state;
            memset(&constant_state, 0, sizeof(constant_state));
            constant_state.rt[0].blend_enable = 1;
            constant_state.rt[0].rgb_func = PIPE_BLEND_ADD;
            constant_state.rt[0].alpha_func = PIPE_BLEND_ADD;
            constant_state.rt[0].rgb_src_factor =
               PIPE_BLENDFACTOR_CONST_COLOR;
            constant_state.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_ZERO;
            constant_state.rt[0].alpha_src_factor =
               PIPE_BLENDFACTOR_CONST_COLOR;
            constant_state.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_ZERO;
            constant_state.rt[0].colormask = PIPE_MASK_RGBA;
            constant_blend = ctx->create_blend_state(ctx, &constant_state);
            blend_ok = constant_blend != NULL;
            if (blend_ok) {
               const struct pipe_blend_color constant =
                  {.color = {0.25f, 0.5f, 0.75f, 0.5f}};
               const union pipe_color_union black = {.f = {0, 0, 0, 0}};
               const uint8_t expected[4] = {64, 128, 191, 128};
               ctx->set_blend_color(ctx, &constant);
               gex_bind_blend(ctx, constant_blend);
               gex_bind_fs(ctx, white_fs);
               ctx->clear_render_target(ctx, &surface, &black,
                                        0, 0, 8, 8, false);
               ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
               blend_ok = check_uniform_tolerance(
                  ctx, resource, expected, 1, "blend-constant-color");
            }
         }
      }
      gex_bind_blend(ctx, base_blend);
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      if (blend_vs) gex_delete_vs(ctx, blend_vs);
      if (blend_fs) gex_delete_fs(ctx, blend_fs);
      if (gray_fs) gex_delete_fs(ctx, gray_fs);
      if (quarter_fs) gex_delete_fs(ctx, quarter_fs);
      if (white_fs) gex_delete_fs(ctx, white_fs);
      if (color_mask) gex_delete_blend(ctx, color_mask);
      if (additive) gex_delete_blend(ctx, additive);
      if (reverse) gex_delete_blend(ctx, reverse);
      if (constant_blend) gex_delete_blend(ctx, constant_blend);
      if (blend) gex_delete_blend(ctx, blend);
   }

   bool cull_ok = blend_ok;
   void *cw_front = NULL, *ccw_front = NULL, *ccw_cull_front = NULL;
   if (cull_ok) {
      struct pipe_rasterizer_state cull_state;
      memset(&cull_state, 0, sizeof(cull_state));
      cull_state.half_pixel_center = 1;
      cull_state.depth_clip_near = 1;
      cull_state.depth_clip_far = 1;
      cull_state.cull_face = PIPE_FACE_BACK;
      cw_front = ctx->create_rasterizer_state(ctx, &cull_state);
      cull_state.cull_face = PIPE_FACE_FRONT;
      cull_state.front_ccw = 1;
      ccw_cull_front = ctx->create_rasterizer_state(ctx, &cull_state);
      cull_state.cull_face = PIPE_FACE_BACK;
      ccw_front = ctx->create_rasterizer_state(ctx, &cull_state);
      cull_ok = cw_front && ccw_front && ccw_cull_front;
      if (cull_ok) {
         viewport.scale[1] = -4.0f;
         ctx->set_viewport_states(ctx, 0, 1, &viewport);
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, fs);
         gex_bind_rast(ctx, ccw_front);
         ctx->clear_render_target(ctx, &surface, &color, 0, 0, 8, 8, false);
         ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
         cull_ok = check_asymmetric(ctx, resource, false, false,
                                    "cull-negative-y-ccw-kept");
         if (cull_ok) {
            gex_bind_rast(ctx, cw_front);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            cull_ok = check_uniform(ctx, resource, blue,
                                    "cull-negative-y-cw-rejected");
         }
         if (cull_ok) {
            viewport.scale[1] = -4.0f;
            ctx->set_viewport_states(ctx, 0, 1, &viewport);
            gex_bind_rast(ctx, ccw_cull_front);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            cull_ok = check_uniform(ctx, resource, blue,
                                    "cull-front-rejected");
         }
         if (cull_ok) {
            viewport.scale[1] = 4.0f;
            ctx->set_viewport_states(ctx, 0, 1, &viewport);
            gex_bind_rast(ctx, cw_front);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            cull_ok = check_asymmetric(ctx, resource, true, false,
                                       "cull-positive-y-cw-kept");
         }
         if (cull_ok) {
            gex_bind_rast(ctx, ccw_front);
            ctx->clear_render_target(ctx, &surface, &color,
                                     0, 0, 8, 8, false);
            ctx->draw_vbo(ctx, &draw_info, 0, NULL, &draw, 1);
            cull_ok = check_uniform(ctx, resource, blue,
                                    "cull-positive-y-ccw-rejected");
         }
      }
      gex_bind_rast(ctx, base_rasterizer);
      gex_bind_vs(ctx, NULL);
      gex_bind_fs(ctx, NULL);
      if (cw_front) gex_delete_rast(ctx, cw_front);
      if (ccw_front) gex_delete_rast(ctx, ccw_front);
      if (ccw_cull_front)
         gex_delete_rast(ctx, ccw_cull_front);
   }
   if (cull_ok && webgpu_boundary) {
      struct pipe_resource bad_vertex_templ = vertex_templ;
      struct pipe_resource bad_index_templ = index_templ;
      bad_vertex_templ.width0 = 16;
      bad_index_templ.width0 = 2;
      struct pipe_resource *bad_vertex =
         screen->resource_create(screen, &bad_vertex_templ);
      struct pipe_resource *bad_index =
         screen->resource_create(screen, &bad_index_templ);
      const float one_vertex[4] = {0.0f, 0.0f, 0.0f, 1.0f};
      const uint16_t out_of_range_index = 4;
      if (bad_vertex && bad_index) {
         ctx->buffer_subdata(ctx, bad_vertex, PIPE_MAP_WRITE, 0,
                             sizeof(one_vertex), one_vertex);
         ctx->buffer_subdata(ctx, bad_index, PIPE_MAP_WRITE, 0,
                             sizeof(out_of_range_index), &out_of_range_index);
      }
      struct pipe_vertex_element bad_element = {0};
      bad_element.src_format = PIPE_FORMAT_R32G32B32A32_FLOAT;
      bad_element.src_stride = 16;
      void *bad_elements =
         ctx->create_vertex_elements_state(ctx, 1, &bad_element);
      struct pipe_vertex_buffer bad_vertex_buffer = {0};
      bad_vertex_buffer.buffer.resource = bad_vertex;
      struct pipe_draw_info bad_indexed_info = {0};
      bad_indexed_info.mode = MESA_PRIM_TRIANGLES;
      bad_indexed_info.index_size = 2;
      bad_indexed_info.instance_count = 1;
      bad_indexed_info.index.resource = bad_index;
      struct pipe_draw_start_count_bias bad_draw = {.start = 0, .count = 1};
      if (bad_vertex && bad_index && bad_elements) {
         gex_bind_velems(ctx, bad_elements);
         ctx->set_vertex_buffers(ctx, 1, &bad_vertex_buffer);
         gex_bind_vs(ctx, vs);
         gex_bind_fs(ctx, fs);
         ctx->draw_vbo(ctx, &bad_indexed_info, 0, NULL, &bad_draw, 1);
      }
      struct pipe_transfer *bad_bounds_transfer = NULL;
      void *bad_bounds_map = bad_vertex && bad_index && bad_elements
         ? ctx->texture_map(ctx, resource, 0, PIPE_MAP_READ,
                            &(struct pipe_box){0, 0, 0, 1, 1, 1},
                            &bad_bounds_transfer) : (void *)1;
      cull_ok = !bad_bounds_map && !bad_bounds_transfer;
      if (!cull_ok)
         fprintf(stderr, "FAIL gex-wgpu indexed vertex bounds did not fail closed\n");
      if (bad_bounds_transfer)
         ctx->texture_unmap(ctx, bad_bounds_transfer);
      ctx->set_vertex_buffers(ctx, 0, NULL);
      gex_bind_velems(ctx, NULL);
      if (bad_elements)
         gex_delete_velems(ctx, bad_elements);
      if (bad_vertex) pipe_resource_reference(&bad_vertex, NULL);
      if (bad_index) pipe_resource_reference(&bad_index, NULL);
   }
   if (!cull_ok) {
      gex_bind_vs(ctx, NULL);
      gex_delete_vs(ctx, vs);
      gex_delete_fs(ctx, fs);
      gex_destroy_context(ctx);
      pipe_resource_reference(&resource, NULL);
      screen->destroy(screen);
      return !linear_ok ? 14
         : (!repeat_ok ? 15
            : (!mip_ok ? 16
               : (!trilinear_ok ? 17
                  : (!border_ok ? 18
                     : (!linear_border_ok ? 19
                        : (!swizzle_ok ? 20
                           : (!base_level_ok ? 21
                              : (!srgb_ok ? 22
                                 : (!depth_ok ? 23
                                    : (!stencil_ok ? 24
                                       : (!blend_ok ? 25 : 26)))))))))));
   }

   gex_bind_vs(ctx, NULL);
   gex_bind_fs(ctx, NULL);
   gex_bind_rast(ctx, NULL);
   gex_delete_rast(ctx, base_rasterizer);
   gex_bind_blend(ctx, NULL);
   gex_delete_blend(ctx, base_blend);
   gex_bind_dsa(ctx, NULL);
   gex_delete_dsa(ctx, base_dsa);
#ifdef GEX_CONFORMANCE_SOFTPIPE
   gex_bind_velems(ctx, NULL);
   gex_delete_velems(ctx, empty_vertex_elements);
#endif
   gex_delete_vs(ctx, vs);
   gex_delete_fs(ctx, fs);
   gex_destroy_context(ctx);
   pipe_resource_reference(&resource, NULL);

   if (!webgpu_boundary) {
      printf("PASS gex-gallium-%s varying viewport scissor cull-winding clip depth-compare-write-mask stencil-masked-replace-equal blend-src-alpha-add-reverse-constant-color-mask mrt-independent-mask-blend constant-buffer-user-subrange-static-dynamic-offset-rebind-lifetime-vertex-fragment-coexistence-multi-buffer texture-constant-multi-descriptor-coexistence-lifetime-slot-swizzle-r8-rg8-bgra-r16f-rg16f-rgba16f-r8ui-rg8ui-rgba8ui-r8i-rg8i-rgba8i-s3tc-rgtc-linear-srgb-snorm-mesa-decoder-dxt1-rgb-opaque texel-fetch-normalized-uint-sint-explicit-implicit-multilevel-lod-nearest-clamps-clamp-repeat-mirror-integer-border-boundaries-base-level-typed-swizzle-array-layers-1d-emulated-2d-float-uint-repeat-mip-implicit-1d-array-float-uint-layer-upload-lifetime-rect-float-unnormalized-coordinate-3d-texel-fetch-normalized-r-wrap-mip-implicit-r-derivative-integer-border-fmad-render-sample-array-cube-integer-cube-six-face-orientation-cube-array-float-uint-base-index-orientation-shadow-compare-implicit-explicit-gradient vertex-index-float2-float3-interleaved-unorm8-multistream-lifetime-instance-divisor-base-order sampler-lod-bias-positive-negative-explicit-compose-rebind texture-nearest-linear-repeat-mirror-mip-trilinear-border-linear-border-swizzle-base-level-srgb-subrect-varying-lifetime\n",
             backend);
      screen->destroy(screen);
      return 0;
   }

   struct pipe_context *bad_ctx = screen->context_create(screen, NULL, 0);
   struct pipe_resource *bad_texture =
      screen->resource_create(screen, &sample_templ);
   const uint8_t tiny_source[4] = {1, 2, 3, 4};
   if (!bad_ctx || !bad_texture) {
      fprintf(stderr, "FAIL gex-wgpu malformed-upload setup\n");
      if (bad_ctx) bad_ctx->destroy(bad_ctx);
      if (bad_texture) pipe_resource_reference(&bad_texture, NULL);
      screen->destroy(screen);
      return 27;
   }
   bad_ctx->texture_subdata(bad_ctx, bad_texture, 0, PIPE_MAP_WRITE,
                            &sample_box, tiny_source, UINT32_MAX, UINTPTR_MAX);
   struct pipe_transfer *bad_transfer = NULL;
   void *bad_map = bad_ctx->texture_map(bad_ctx, bad_texture, 0, PIPE_MAP_READ,
                                        &sample_box, &bad_transfer);
   if (bad_map || bad_transfer) {
      fprintf(stderr, "FAIL gex-wgpu malformed-upload did not fail closed\n");
      if (bad_transfer) bad_ctx->texture_unmap(bad_ctx, bad_transfer);
      bad_ctx->destroy(bad_ctx);
      pipe_resource_reference(&bad_texture, NULL);
      screen->destroy(screen);
      return 28;
   }
   bad_ctx->destroy(bad_ctx);
   pipe_resource_reference(&bad_texture, NULL);
   struct pipe_context *bad_layer_ctx =
      screen->context_create(screen, NULL, 0);
   struct pipe_resource *bad_layer_texture =
      screen->resource_create(screen, &sample_templ);
   struct pipe_box bad_layer_box = sample_box;
   bad_layer_box.depth = 2;
   if (!bad_layer_ctx || !bad_layer_texture) {
      fprintf(stderr, "FAIL gex-wgpu malformed-layer-upload setup\n");
      if (bad_layer_ctx) bad_layer_ctx->destroy(bad_layer_ctx);
      if (bad_layer_texture)
         pipe_resource_reference(&bad_layer_texture, NULL);
      screen->destroy(screen);
      return 29;
   }
   bad_layer_ctx->texture_subdata(
      bad_layer_ctx, bad_layer_texture, 0, PIPE_MAP_WRITE,
      &bad_layer_box, tiny_source, 4, 4);
   bad_transfer = NULL;
   bad_map = bad_layer_ctx->texture_map(
      bad_layer_ctx, bad_layer_texture, 0, PIPE_MAP_READ,
      &sample_box, &bad_transfer);
   if (bad_map || bad_transfer) {
      fprintf(stderr,
              "FAIL gex-wgpu malformed-layer-upload did not fail closed\n");
      if (bad_transfer)
         bad_layer_ctx->texture_unmap(bad_layer_ctx, bad_transfer);
      bad_layer_ctx->destroy(bad_layer_ctx);
      pipe_resource_reference(&bad_layer_texture, NULL);
      screen->destroy(screen);
      return 30;
   }
   bad_layer_ctx->destroy(bad_layer_ctx);
   pipe_resource_reference(&bad_layer_texture, NULL);
   struct pipe_context *bad_bc_ctx =
      screen->context_create(screen, NULL, 0);
   struct pipe_resource bad_bc_templ = sample_templ;
   bad_bc_templ.format = PIPE_FORMAT_DXT1_RGBA;
   bad_bc_templ.width0 = 8;
   bad_bc_templ.height0 = 4;
   bad_bc_templ.last_level = 0;
   struct pipe_resource *bad_bc_texture = bad_bc_ctx
      ? screen->resource_create(screen, &bad_bc_templ) : NULL;
   struct pipe_resource *bad_bc_probe = bad_bc_ctx
      ? screen->resource_create(screen, &sample_templ) : NULL;
   const struct pipe_box unaligned_bc_box = {
      .x = 2, .y = 0, .z = 0, .width = 4, .height = 4, .depth = 1,
   };
   if (!bad_bc_ctx || !bad_bc_texture || !bad_bc_probe) {
      fprintf(stderr, "FAIL gex-wgpu malformed-BC-upload setup\n");
      if (bad_bc_ctx) bad_bc_ctx->destroy(bad_bc_ctx);
      if (bad_bc_texture) pipe_resource_reference(&bad_bc_texture, NULL);
      if (bad_bc_probe) pipe_resource_reference(&bad_bc_probe, NULL);
      screen->destroy(screen);
      return 31;
   }
   bad_bc_ctx->texture_subdata(bad_bc_ctx, bad_bc_texture, 0, PIPE_MAP_WRITE,
                               &unaligned_bc_box, bc1_gray, 8, 8);
   bad_transfer = NULL;
   bad_map = bad_bc_ctx->texture_map(bad_bc_ctx, bad_bc_probe, 0,
                                     PIPE_MAP_READ, &sample_box,
                                     &bad_transfer);
   if (bad_map || bad_transfer) {
      fprintf(stderr,
              "FAIL gex-wgpu malformed-BC-upload did not fail closed\n");
      if (bad_transfer) bad_bc_ctx->texture_unmap(bad_bc_ctx, bad_transfer);
      bad_bc_ctx->destroy(bad_bc_ctx);
      pipe_resource_reference(&bad_bc_texture, NULL);
      pipe_resource_reference(&bad_bc_probe, NULL);
      screen->destroy(screen);
      return 32;
   }
   bad_bc_ctx->destroy(bad_bc_ctx);
   pipe_resource_reference(&bad_bc_texture, NULL);
   pipe_resource_reference(&bad_bc_probe, NULL);
   printf("PASS gex-wgpu-gallium varying viewport scissor cull-winding clip frag-coord-half-integer depth-compare-write-mask stencil-masked-replace-equal blend-src-alpha-add-reverse-constant-color-mask mrt-independent-mask-blend constant-buffer-user-subrange-static-dynamic-offset-rebind-lifetime-vertex-fragment-coexistence-multi-buffer texture-constant-multi-descriptor-coexistence-lifetime-slot-swizzle-r8-rg8-bgra-r16f-rg16f-rgba16f-r8ui-rg8ui-rgba8ui-r8i-rg8i-rgba8i-bc1-bc2-bc3-bc4-bc5-bc6h-bc7-linear-srgb-snorm-block-decode-dxt1-rgb-opaque-edge-mip-2d-array-cube-cube-array texel-fetch-normalized-uint-sint-explicit-implicit-multilevel-lod-nearest-clamps-clamp-repeat-mirror-integer-border-boundaries-base-level-typed-swizzle-array-layers-1d-emulated-2d-float-uint-repeat-mip-implicit-1d-array-float-uint-layer-upload-lifetime-rect-float-uint-unnormalized-coordinate-3d-texel-fetch-normalized-r-wrap-mip-implicit-r-derivative-integer-border-fmad-render-sample-array-cube-integer-cube-six-face-orientation-cube-array-float-uint-base-index-orientation-shadow-compare-implicit-explicit-gradient vertex-index-float2-float3-interleaved-unorm8-multistream-lifetime-instance-divisor-base-order-indexed-bounds sampler-lod-bias-positive-negative-explicit-compose-rebind texture-nearest-linear-repeat-mirror-mip-trilinear-border-linear-border-swizzle-base-level-srgb-subrect-varying-lifetime malformed-upload-layer-bounds-compressed-block-alignment-cube-array-shape-view-alignment\n");
   screen->destroy(screen);
   return 0;
}

#ifdef GEX_CONFORMANCE_SOFTPIPE
static bool
check_bc_cpu_decoder(void)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_DXT1_RGB, PIPE_FORMAT_DXT1_SRGB,
      PIPE_FORMAT_DXT1_RGBA, PIPE_FORMAT_DXT1_SRGBA,
      PIPE_FORMAT_DXT3_RGBA, PIPE_FORMAT_DXT3_SRGBA,
      PIPE_FORMAT_DXT5_RGBA, PIPE_FORMAT_DXT5_SRGBA,
   };
   const uint8_t bc1_transparent_black[8] = {
      0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
   };
   const uint8_t bc1_gray[8] = {
      0x10, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const uint8_t bc2_gray[16] = {
      0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88, 0x88,
      0x10, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const uint8_t bc3_gray[16] = {
      0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x10, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   const uint8_t expected[8][4] = {
      {0, 0, 0, 255}, {0, 0, 0, 255},
      {132, 130, 132, 255}, {59, 57, 59, 255},
      {132, 130, 132, 136}, {59, 57, 59, 136},
      {132, 130, 132, 255}, {59, 57, 59, 255},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(formats); ++i) {
      const bool rgb = i < 2;
      const bool bc1 = i < 4;
      const bool bc2 = i == 4 || i == 5;
      const uint8_t *blocks = rgb ? bc1_transparent_black
                              : bc1 ? bc1_gray
                              : bc2 ? bc2_gray : bc3_gray;
      const unsigned block_bytes = bc1 ? 8 : 16;
      uint8_t decoded[4 * 4 * 4];
      util_format_unpack_rgba_8unorm_rect(
         formats[i], decoded, 4 * 4, blocks, block_bytes, 4, 4);
      for (unsigned pixel = 0; pixel < 16; ++pixel) {
         if (memcmp(decoded + pixel * 4, expected[i], 4) != 0) {
            fprintf(stderr,
                    "FAIL gex-wgpu Mesa S3TC decoder format=%u pixel=%u "
                    "value=%u,%u,%u,%u\n",
                    formats[i], pixel, decoded[pixel * 4 + 0],
                    decoded[pixel * 4 + 1], decoded[pixel * 4 + 2],
                    decoded[pixel * 4 + 3]);
            return false;
         }
      }
   }
   const enum pipe_format rgtc_formats[] = {
      PIPE_FORMAT_RGTC1_UNORM, PIPE_FORMAT_RGTC1_SNORM,
      PIPE_FORMAT_RGTC2_UNORM, PIPE_FORMAT_RGTC2_SNORM,
   };
   const uint8_t rgtc_blocks[4][16] = {
      {0xff, 0x00, 0, 0, 0, 0, 0, 0},
      {0x7f, 0x81, 0, 0, 0, 0, 0, 0},
      {0xff, 0x00, 0, 0, 0, 0, 0, 0,
       0xff, 0x00, 0, 0, 0, 0, 0, 0},
      {0x7f, 0x81, 0, 0, 0, 0, 0, 0,
       0x81, 0x7f, 0, 0, 0, 0, 0, 0},
   };
   const float rgtc_expected[4][4] = {
      {1.0f, 0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f},
      {1.0f, 1.0f, 0.0f, 1.0f}, {1.0f, -1.0f, 0.0f, 1.0f},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(rgtc_formats); ++i) {
      const unsigned block_bytes = i < 2 ? 8 : 16;
      float decoded[4 * 4 * 4];
      util_format_unpack_rgba_rect(
         rgtc_formats[i], decoded, 4 * 4 * sizeof(float), rgtc_blocks[i],
         block_bytes, 4, 4);
      for (unsigned pixel = 0; pixel < 16; ++pixel) {
         for (unsigned channel = 0; channel < 4; ++channel) {
            if (fabsf(decoded[pixel * 4 + channel] -
                      rgtc_expected[i][channel]) > 0.00001f) {
               fprintf(stderr,
                       "FAIL gex-wgpu Mesa RGTC decoder format=%u pixel=%u "
                       "channel=%u value=%f\n",
                       rgtc_formats[i], pixel, channel,
                       decoded[pixel * 4 + channel]);
               return false;
            }
         }
      }
   }
   return true;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   struct sw_winsys *winsys = null_sw_create();
   if (!winsys) {
      fprintf(stderr, "FAIL gex-gallium softpipe winsys\n");
      glsl_type_singleton_decref();
      return 1;
   }
   struct pipe_screen *screen = softpipe_create_screen(winsys);
   int result = run_gallium_conformance(screen, false, "softpipe");
   if (!result && !check_bc_cpu_decoder())
      result = 33;
   winsys->destroy(winsys);
   glsl_type_singleton_decref();
   return result;
}
#else
EMSCRIPTEN_KEEPALIVE int
gex_wgpu_smoke(void)
{
   glsl_type_singleton_init_or_ref();
   WGPUDevice device = emscripten_webgpu_get_device();
   if (!device) {
      fprintf(stderr, "FAIL gex-wgpu device\n");
      glsl_type_singleton_decref();
      return 1;
   }
   const int result = run_gallium_conformance(
      gex_wgpu_screen_create(device), true, "webgpu");
   glsl_type_singleton_decref();
   return result;
}
#endif
