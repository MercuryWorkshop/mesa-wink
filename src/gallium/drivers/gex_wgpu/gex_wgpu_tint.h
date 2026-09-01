/* SPDX-License-Identifier: MIT */
#ifndef GEX_WGPU_TINT_H
#define GEX_WGPU_TINT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The returned strings use malloc and are released with free by the caller. */
char *gex_wgpu_spirv_to_wgsl(const uint32_t *words, size_t word_count,
                             char **error_out);

#ifdef __cplusplus
}
#endif

#endif
