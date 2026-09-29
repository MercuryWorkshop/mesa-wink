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

/* The statistics of a pipeline statistics query, VkQueryPipelineStatisticFlagBits'
 * bits; and after them the counters of each vertex stream's transform
 * feedback query, the primitives written and those that would have been,
 * and of its primitives generated query. */
#define WVK_STATS 11
#define WVK_COUNTER_XFB_WRITTEN(stream) (11 + 2 * (stream))
#define WVK_COUNTER_XFB_NEEDED(stream) (12 + 2 * (stream))
#define WVK_COUNTER_PRIMITIVES(stream) (19 + (stream))
#define WVK_COUNTERS 32

/* A query counted as the replay goes (wvk_query.c): its pool and query,
 * the counters it has values of, and where they were at its begin -- the
 * host's, and a copy of the GPU's. */
struct wvk_counter_query {
   struct wvk_query_pool *pool;
   uint32_t query;
   uint32_t mask;
   uint64_t host_begin[WVK_COUNTERS];
   WGPUBuffer begin;
};
#define WVK_COUNTER_QUERIES 4

struct wvk_query_pool {
   struct vk_query_pool vk;
   /* How many values a query has: one, but a statistics query's, one for
    * each statistic it counts. */
   uint32_t values;
   /* The results, 8 bytes a value: host-visible, so a submit that wrote
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

/* Counts of the active pipeline statistics query: `n` of the statistic
 * `bit`, which the host knows; or, on the GPU and outside a pass, what
 * word `word` of `src` says, divided by `divisor`, of each of `bits`. */
void wvk_replay_count_stat(struct wvk_replay *replay, VkQueryPipelineStatisticFlagBits bit, uint64_t n);
/* Or of counter `counter` (WVK_COUNTERS). */
void wvk_replay_count(struct wvk_replay *replay, uint32_t counter, uint64_t n);
/* The GPU's counters (WVK_DRIVER_SLOT_STATS), made the first time. */
WGPUBuffer wvk_replay_counters(struct wvk_replay *replay);
void wvk_replay_add_stats(struct wvk_replay *replay, WGPUBuffer src, uint64_t src_size, uint32_t word,
                          uint32_t divisor, uint32_t bits);
/* The counters the active queries count, 0 for none. */
uint32_t wvk_replay_stats(const struct wvk_replay *replay);

/* For a pass about to begin: the query set it counts in, or NULL. */
WGPUQuerySet wvk_replay_query_set(struct wvk_device *dev, struct wvk_replay *replay);
/* After a pass ended: its results to the pools. */
void wvk_replay_resolve_queries(struct wvk_replay *replay);
void wvk_replay_finish_queries(struct wvk_replay *replay);

#endif
