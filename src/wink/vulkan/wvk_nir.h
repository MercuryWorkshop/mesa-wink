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
 * wvk_descriptor_set_layout.h, WVK_SET_META_SLOT). */
nir_def *wvk_nir_load_meta(struct nir_builder *b, uint32_t group, nir_def *slot, uint32_t word);

/* Texel buffers as storage buffers, their format as their records say. */
bool wvk_nir_lower_texel_buffers(nir_shader *nir);

/* Views' swizzles, as their records say. */
bool wvk_nir_lower_view_swizzles(nir_shader *nir);

#endif
