/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_SYNC_H
#define WVK_SYNC_H

#include "wvk_private.h"

#include "vk_sync.h"

struct wvk_queue;

/* A binary sync: signaled by the host, or by the submit whose serial it
 * holds once the queue has finished it. The runtime builds timeline
 * semaphores out of these. */
struct wvk_sync {
   struct vk_sync base;
   bool signaled;
   /* The queue serial that signals it, or 0. */
   uint64_t serial;
};

extern const struct vk_sync_type wvk_sync_type;

static inline bool
wvk_sync_is_wvk(const struct vk_sync *sync)
{
   return sync->type == &wvk_sync_type;
}

/* The queue's signal operation for serial. */
void wvk_sync_set_pending(struct vk_sync *sync, uint64_t serial);

#endif
