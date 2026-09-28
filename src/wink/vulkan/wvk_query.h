/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_QUERY_H
#define WVK_QUERY_H

#include "wvk_private.h"

#include "vk_buffer_view.h"
#include "vk_object.h"
#include "vk_query_pool.h"

struct vk_cmd_queue;
struct wvk_device;
struct wvk_device_memory;
struct wvk_replay;

struct wvk_query_pool {
   struct vk_query_pool vk;
   /* The results, 8 bytes a query: host-visible, so a submit that wrote
    * them reads them back. */
   VkDeviceMemory memory;
   struct wvk_device_memory *mem;
   WGPUBuffer buffer;
   /* The submit a query ended in, or 0: available once it is done. */
   uint64_t *serial;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_query_pool, vk.base, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL)

struct wvk_event {
   struct vk_object_base base;
   uint32_t set;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)

/* An occlusion query of the current pass, and where in the submit's query
 * set it is counted. */
struct wvk_pass_query {
   struct wvk_query_pool *pool;
   uint32_t query;
   uint32_t slot;
   /* A part of a query a pass was ended in, or a clear came in: added to
    * what the parts before counted, but for the first. */
   bool accumulate;
   /* The active query: how many parts it has so far, and whether it is
    * between them. */
   uint32_t parts;
   bool suspended;
};

/* The active occlusion query ended where the pass is going to, or a clear
 * is not to be counted; and begun again, in a slot of its own, after. */
void wvk_replay_suspend_query(struct wvk_replay *replay);
void wvk_replay_resume_query(struct wvk_replay *replay);

/* Whether a command buffer has occlusion queries, and so its passes need
 * the submit's query set. */
bool wvk_cmd_queue_has_occlusion_queries(const struct vk_cmd_queue *queue);

/* For a pass about to begin: the query set it counts in, or NULL. */
WGPUQuerySet wvk_replay_query_set(struct wvk_device *dev, struct wvk_replay *replay);
/* After a pass ended: its results to the pools. */
void wvk_replay_resolve_queries(struct wvk_replay *replay);
void wvk_replay_finish_queries(struct wvk_replay *replay);

#endif
