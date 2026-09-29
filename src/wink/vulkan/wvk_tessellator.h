/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_TESSELLATOR_H
#define WVK_TESSELLATOR_H

/* The tessellator's WGSL: entry points count, scan and generate. */
extern const char wvk_tessellator_wgsl[];

/* The most points and indices a patch makes (at levels of 64). */
#define WVK_TESS_MAX_POINTS (65 * 65)
#define WVK_TESS_MAX_INDICES (64 * 64 * 2 * 3)

#endif
