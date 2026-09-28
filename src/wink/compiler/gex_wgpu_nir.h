/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * NIR to WGSL, for both wink drivers: the GL one (gex_wgpu) and the Vulkan
 * one (wvk). The shader goes to SPIR-V through zink's emitter and to WGSL
 * through Tint's SPIR-V reader, with whatever lowering the two of them and
 * WebGPU need on the way. What reaches gex_wgpu_nir_to_wgsl has its
 * resources at their final @group and @binding already: descriptor_set and
 * binding of each variable are what the WGSL says, and textures and samplers
 * are separate variables.
 */
#ifndef GEX_WGPU_NIR_H
#define GEX_WGPU_NIR_H

#include "compiler/nir/nir.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Raised from sixteen with the vertex-attribute limit: the passthrough
 * rasterisation of an emulated stage builds one element per varying, so its
 * capacity has to reach whatever the device granted. Array capacity only --
 * what GL reports as MAX_VERTEX_ATTRIBS is a separate claim. */
#define GEX_WGPU_MAX_VERTEX_ELEMENTS 32
#define GEX_WGPU_MAX_COLOR_TARGETS 8
/* Indexed by VARYING_SLOT_*, which is the numbering both stages agree on --
 * driver_location is assigned per stage and the vertex stage counts
 * gl_Position among its outputs, so the same varying has different
 * driver_locations on the two sides. */
#define GEX_WGPU_MAX_VARYING_LOCATIONS 64

/* What a compile says of itself besides the WGSL. */
struct gex_wgpu_wgsl_info {
   /* "stage:name", for the module's label. */
   char label[80];
   unsigned stage;
   /* The shader is one worth dumping (GEX_WGPU_DUMP_VS). */
   bool dump;
   unsigned spirv_words;
   /* os_time_get_nano, in milliseconds: at the start, after SPIR-V, and
    * after WGSL. */
   double t_start, t_spirv, t_wgsl;
};

/* Consumes `nir`, whether it succeeds or not. Returns malloc'd WGSL, or NULL
 * with a malloc'd reason in *error_out. flip_y negates the vertex stage's
 * gl_Position.y; clip_halfz says the depth is already in WebGPU's 0..1,
 * where otherwise GL's -1..1 is mapped to it. */
char *gex_wgpu_nir_to_wgsl(nir_shader *nir, mesa_shader_stage stage, bool flip_y,
                           bool clip_halfz, struct gex_wgpu_wgsl_info *out,
                           char **error_out);

/* The compiler options every stage is built with. */
const nir_shader_compiler_options *gex_wgpu_nir_options(void);

/* Whether anything in the shader is 64 bits wide. */
bool gex_wgpu_has_64_bit(nir_shader *nir);

#ifdef __cplusplus
}
#endif

#endif
