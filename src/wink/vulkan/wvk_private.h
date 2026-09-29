/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * wvk: Vulkan on WebGPU.
 *
 * A Vulkan physical device is a WebGPU adapter and a Vulkan device a
 * WGPUDevice. Every command is recorded into the command buffer's
 * vk_cmd_queue and replayed at submit, into one WGPUCommandEncoder per
 * submit: WebGPU command buffers are single-use, and Vulkan's are not.
 * The same source builds natively against Dawn, which is where Vulkan's
 * tools and conformance tests run, and for the browser inside libwink.
 */
#ifndef WVK_PRIVATE_H
#define WVK_PRIVATE_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <webgpu/webgpu.h>

#include "vk_alloc.h"
#include "vk_log.h"
#include "vk_util.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_math.h"

/* What Vulkan version this driver reports; WINK_VK_API_VERSION=1.3 raises
 * it for applications that check, with 1.3's extra features reported as
 * absent. See wvk_physical_device.c. */
/* Viewports a vertex shader chooses among, each a draw of its own. */
#define WVK_MAX_VIEWPORTS 16

/* Vulkan 1.3, but for bufferDeviceAddress: WebGPU has no pointers. */
#define WVK_API_VERSION VK_MAKE_API_VERSION(0, 1, 3, VK_HEADER_VERSION)

/* Queue families: everything, and compute and transfer (wvk_queue.h). */
enum {
   WVK_QUEUE_FAMILY_ALL,
   WVK_QUEUE_FAMILY_COMPUTE,
   WVK_QUEUE_FAMILIES,
};
#define WVK_MAX_MEMORY_TYPES 4
#define WVK_MAX_MEMORY_HEAPS 2

/* WebGPU's own sample counts. */
#define WVK_SAMPLE_COUNTS (VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT)

static inline WGPUStringView
wvk_sv(const char *s)
{
   return (WGPUStringView){ s, s ? WGPU_STRLEN : 0 };
}

/* A WGPUStringView's length as printf's "%.*s" wants it. */
static inline int
wvk_sv_len(WGPUStringView sv)
{
   if (!sv.data)
      return 0;
   return sv.length == WGPU_STRLEN ? (int)strlen(sv.data) : (int)sv.length;
}

#define wvk_debug(...) mesa_logd("wvk: " __VA_ARGS__)

#endif
