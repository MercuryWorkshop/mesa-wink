/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * The driver's own NIR passes, besides wvk_shader.c's.
 */
#ifndef WVK_NIR_H
#define WVK_NIR_H

#include "compiler/nir/nir.h"

struct nir_builder;

/* Word `word` of the record of binding `slot` of `group` (see
 * wvk_descriptor_set_layout.h, WVK_META_OFFSET); `meta` is where each
 * group's records start, in words. */
nir_def *wvk_nir_load_meta(struct nir_builder *b, const uint32_t *meta, uint32_t group, nir_def *slot,
                           uint32_t word);

/* Cooperative matrices as arrays of the one invocation that is a subgroup
 * (wvk_nir_cmat.c): before variable copies are lowered. */
bool wvk_nir_lower_cooperative_matrix(nir_shader *nir);

/* Texel buffers as storage buffers, their format as their records say. */
bool wvk_nir_lower_texel_buffers(nir_shader *nir, const uint32_t *meta_base);

/* Views' swizzles, as their records say. */
bool wvk_nir_lower_view_swizzles(nir_shader *nir, const uint32_t *meta_base);

/* What WebGPU's samplers cannot do, as samplers' records say: border
 * colours, mirroring once, LOD bias, unnormalized coordinates, and
 * sampling integer textures at all. */
bool wvk_nir_lower_samplers(nir_shader *nir, const uint32_t *meta_base);

/* Image accesses bounded by their views (robustImageAccess2), and null
 * views reading and querying zero (nullDescriptor), as records say. */
bool wvk_nir_lower_robust_images(nir_shader *nir, const uint32_t *meta_base, bool bounds, bool null);
/* A view's minimum LOD, where the device has them (VK_EXT_image_view_min_lod):
 * before the samplers' lowering. */
bool wvk_nir_lower_view_min_lod(nir_shader *nir, const uint32_t *meta_base);
/* The slices of 3D images storage views have (VK_EXT_image_sliced_view_of_3d,
 * and 2D views of 3D images): after the robust images' bounds. */
bool wvk_nir_lower_image_slices(nir_shader *nir, const uint32_t *meta_base);

/* Storage images shaders do atomics on, as their image's buffer of
 * texels (wvk_image.h), where their records say. */
bool wvk_nir_lower_image_atomics(nir_shader *nir, const uint32_t *meta_base);

/* Subgroups of one invocation (wvk_nir_subgroups.c). */
bool wvk_nir_lower_subgroups(nir_shader *nir);

/* 8- and 16-bit values as 32-bit words, but half floats of float
 * operations (wvk_nir_small_types.c). */
bool wvk_nir_widen_small_types(nir_shader *nir);


#endif
