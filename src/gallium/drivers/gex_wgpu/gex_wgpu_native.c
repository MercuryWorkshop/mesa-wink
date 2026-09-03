/* SPDX-License-Identifier: MIT */
/* gex_wgpu on a desktop, against native Dawn.
 *
 * The driver itself never depended on a browser: it takes a WGPUDevice and
 * speaks webgpu.h, and everything emscripten-shaped lives in the page glue.
 * So the only things missing to run an ordinary Linux GL application through
 * it are a device and a way to put a finished frame on the screen, which is
 * what this file is.
 *
 * The device comes from Dawn's native implementation of the same C API the
 * browser provides. Presentation goes through Mesa's software winsys -- the
 * one softpipe and llvmpipe use -- because that is what every frontend
 * already knows how to drive, from GLX to EGL to Wayland. Softpipe can hand
 * the winsys its own resource directly, since it rendered into CPU memory;
 * this driver's pixels are on the GPU, so a frame is read back and copied.
 * That readback is the cost of the arrangement and it is per frame, not per
 * draw.
 */
#include <stdbool.h>
#include <stdint.h>

#include "gex_wgpu_public.h"

/* sw_winsys.h uses bool and pipe types without including anything. */
#include "pipe/p_state.h"
#include "frontend/sw_winsys.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "util/u_inlines.h"
#include "util/os_time.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/format/u_format.h"
#include "util/box.h"

#include <stdio.h>
#include <string.h>
#include <sched.h>
#include <time.h>

struct gex_wgpu_sw_screen {
   struct sw_winsys *winsys;
   /* The scanout resource and the displaytarget it is copied into. One is
    * enough: a frontend presents one drawable at a time and reuses its back
    * buffer, so this is a cache with the only entry that is ever hot. */
   struct pipe_resource *presented;
   struct sw_displaytarget *target;
   unsigned target_width;
   unsigned target_height;
   unsigned target_stride;
   enum pipe_format target_format;
   void (*wrapped_destroy)(struct pipe_screen *);
};

/* Per screen, not one global.
 *
 * Mesa builds a screen per X display connection and a toolkit on more than
 * one thread opens several, so this used to be whichever screen was created
 * last -- every other screen then presented through that one's winsys and
 * displaytarget, at that one's size. The device is shared (see create_device)
 * but the drawable is not. */
#define GEX_WGPU_SW_SCREENS 8
static struct {
   struct pipe_screen *screen;
   struct gex_wgpu_sw_screen *state;
} sw_screens[GEX_WGPU_SW_SCREENS];

static struct gex_wgpu_sw_screen *
sw_state_for(struct pipe_screen *screen)
{
   for (unsigned i = 0; i < GEX_WGPU_SW_SCREENS; ++i)
      if (sw_screens[i].screen == screen)
         return sw_screens[i].state;
   return NULL;
}

/* The instance the device came from. Kept because the driver's blocking
 * readbacks need something to pump; see gex_wgpu_jspi_wait below. */
static WGPUInstance native_instance;

/* --- device ------------------------------------------------------------- */

struct request_result {
   WGPUAdapter adapter;
   WGPUDevice device;
   bool done;
};

static void
on_adapter(WGPURequestAdapterStatus status, WGPUAdapter adapter,
           WGPUStringView message, void *userdata1, void *userdata2)
{
   (void)userdata2;
   struct request_result *out = userdata1;
   out->adapter = status == WGPURequestAdapterStatus_Success ? adapter : NULL;
   out->done = true;
   if (!out->adapter)
      fprintf(stderr, "gex-wgpu: no adapter: %.*s\n",
              (int)(message.length == WGPU_STRLEN ? 0 : message.length),
              message.data ? message.data : "");
}

static void
on_device(WGPURequestDeviceStatus status, WGPUDevice device,
          WGPUStringView message, void *userdata1, void *userdata2)
{
   (void)userdata2;
   struct request_result *out = userdata1;
   out->device = status == WGPURequestDeviceStatus_Success ? device : NULL;
   out->done = true;
   if (!out->device)
      fprintf(stderr, "gex-wgpu: no device: %.*s\n",
              (int)(message.length == WGPU_STRLEN ? 0 : message.length),
              message.data ? message.data : "");
}

static void
on_uncaptured_error(WGPUDevice const *device, WGPUErrorType type,
                    WGPUStringView message, void *userdata1, void *userdata2)
{
   (void)device; (void)userdata1; (void)userdata2;
   /* The browser prints these to the console and the page's error capture
    * turns them into a failed run. On a desktop nothing else is listening, so
    * this is the only place a validation error is ever seen. */
   fprintf(stderr, "gex-wgpu: WebGPU error (type %u): %.*s\n", (unsigned)type,
           (int)(message.length == WGPU_STRLEN ? 0 : message.length),
           message.data ? message.data : "");
}

/* Block until a future completes. Native Dawn has a real event loop, so this
 * is the supported way to make an asynchronous API synchronous during
 * start-up; nothing else is running yet. */
static bool
wait_for(WGPUInstance instance, WGPUFuture future)
{
   WGPUFutureWaitInfo wait = { .future = future, .completed = false };
   const WGPUWaitStatus status =
      wgpuInstanceWaitAny(instance, 1, &wait, UINT64_MAX);
   return status == WGPUWaitStatus_Success && wait.completed;
}

static WGPUDevice
create_device(void)
{
   /* One device for the whole process, however many screens ask.
    *
    * Mesa builds a screen per X display connection, and a toolkit that
    * renders on more than one thread opens more than one: Krita reaches five.
    * Each was given a WebGPU device of its own, which costs seconds of
    * adapter and device creation each -- and, worse, made resource sharing
    * across them impossible. Qt shares GL contexts (QOpenGLWidget requires
    * it, and composites the whole window through the shared one), and a
    * texture created on one WebGPU device does not exist on another. Nothing
    * reports that: the sharing simply yields nothing, and Krita's window was
    * black while every draw it did issue succeeded.
    *
    * Nothing releases the device, so there is no reference counting to do
    * here; the process keeps it until it exits. */
   static WGPUDevice cached;
   if (cached)
      return cached;
   WGPUInstanceDescriptor instance_desc = WGPU_INSTANCE_DESCRIPTOR_INIT;
   /* WaitAny is what the two requests below are answered through, and it is
    * off unless asked for. */
   instance_desc.capabilities.timedWaitAnyEnable = 1;
   instance_desc.capabilities.timedWaitAnyMaxCount = 4;
   WGPUInstance instance = wgpuCreateInstance(&instance_desc);
   native_instance = instance;
   if (!instance) {
      fprintf(stderr, "gex-wgpu: wgpuCreateInstance failed\n");
      return NULL;
   }

   struct request_result adapter_result = { 0 };
   WGPURequestAdapterOptions adapter_options =
      WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
   adapter_options.powerPreference = WGPUPowerPreference_HighPerformance;
   WGPURequestAdapterCallbackInfo adapter_cb =
      WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
   adapter_cb.mode = WGPUCallbackMode_WaitAnyOnly;
   adapter_cb.callback = on_adapter;
   adapter_cb.userdata1 = &adapter_result;
   if (!wait_for(instance, wgpuInstanceRequestAdapter(instance,
                                                      &adapter_options,
                                                      adapter_cb)) ||
       !adapter_result.adapter)
      return NULL;

   /* A device gets none of the optional features and WebGPU's *default*
    * limits unless it asks, and the driver derives GL capability from what
    * the device actually has -- so a bare request is not a conservative
    * choice, it is a smaller GL. Left unasked, this reported GL 2.1 rather
    * than 3.3: ARB_depth_buffer_float and EXT_packed_float are GL 3.0
    * requirements and each is one optional WebGPU feature.
    *
    * The same list the browser side asks for, for the same reasons, so that
    * the two targets present the same driver. Anything the adapter does not
    * have is dropped rather than requested -- requesting a missing feature
    * fails the whole device. */
   static const WGPUFeatureName optional[] = {
      WGPUFeatureName_TextureCompressionBC,
      WGPUFeatureName_Depth32FloatStencil8,
      WGPUFeatureName_RG11B10UfloatRenderable,
      WGPUFeatureName_DepthClipControl,
      WGPUFeatureName_Float32Filterable,
      WGPUFeatureName_Float32Blendable,
      WGPUFeatureName_ClipDistances,
      WGPUFeatureName_DualSourceBlending,
      WGPUFeatureName_TimestampQuery,
   };
   WGPUFeatureName granted[sizeof(optional) / sizeof(optional[0])];
   size_t granted_count = 0;
   for (size_t i = 0; i < sizeof(optional) / sizeof(optional[0]); ++i)
      if (wgpuAdapterHasFeature(adapter_result.adapter, optional[i]))
         granted[granted_count++] = optional[i];

   /* Limits, likewise clamped to the adapter: st/mesa needs 13 constant
    * buffers per stage for ARB_uniform_buffer_object and WebGPU's default is
    * 12, and storage buffers in the vertex stage default to none at all. */
   WGPULimits adapter_limits = WGPU_LIMITS_INIT;
   WGPULimits limits = WGPU_LIMITS_INIT;
   const bool have_limits =
      wgpuAdapterGetLimits(adapter_result.adapter, &adapter_limits) ==
      WGPUStatus_Success;
   if (have_limits) {
#define GEX_WANT(field, value)                                              \
      if (adapter_limits.field != WGPU_LIMIT_U32_UNDEFINED &&               \
          adapter_limits.field > 0)                                         \
         limits.field = MIN2(adapter_limits.field, (uint32_t)(value))
      GEX_WANT(maxUniformBuffersPerShaderStage, 16);
      GEX_WANT(maxStorageBuffersPerShaderStage, 16);
      GEX_WANT(maxStorageBuffersInVertexStage, 16);
      GEX_WANT(maxStorageBuffersInFragmentStage, 16);
      /* Eight RGBA8 draw buffers is exactly WebGPU's default 32 bytes per
       * sample, so GL 3.2's eight draw buffers of any wider format fails on
       * it. */
      GEX_WANT(maxColorAttachmentBytesPerSample, 64);
      /* Compute, because a *granted* limit of 256 is what st/mesa reads and
       * ARB_compute_shader wants 1024 -- GL 4.3 requires
       * MAX_COMPUTE_WORK_GROUP_INVOCATIONS to be at least that. WebGPU's
       * default is 256, but requestDevice may ask for more and the adapter
       * decides; asking is the difference between "the device cannot" and
       * "nobody asked it to". */
      GEX_WANT(maxComputeInvocationsPerWorkgroup, 1024);
      GEX_WANT(maxComputeWorkgroupSizeX, 1024);
      GEX_WANT(maxComputeWorkgroupSizeY, 1024);
      GEX_WANT(maxComputeWorkgroupSizeZ, 64);
      GEX_WANT(maxComputeWorkgroupStorageSize, 32768);
      /* Textures, because the *GL version* depends on them and nothing else
       * does: the 4.1 rung of _mesa_compute_version requires a 16384 texture
       * and 2048 array layers where WebGPU guarantees 8192 and 256. A device
       * that can do more will say so; one that cannot leaves the caps at the
       * floor and the driver reports GL 4.0. */
      GEX_WANT(maxTextureDimension1D, 16384);
      GEX_WANT(maxTextureDimension2D, 16384);
      GEX_WANT(maxTextureDimension3D, 2048);
      GEX_WANT(maxTextureArrayLayers, 2048);
#undef GEX_WANT
      fprintf(stderr,
              "gex-wgpu: adapter compute limits invocations/workgroup %u "
              "workgroup %ux%ux%u shared %uB (asking for %u)\n",
              adapter_limits.maxComputeInvocationsPerWorkgroup,
              adapter_limits.maxComputeWorkgroupSizeX,
              adapter_limits.maxComputeWorkgroupSizeY,
              adapter_limits.maxComputeWorkgroupSizeZ,
              adapter_limits.maxComputeWorkgroupStorageSize,
              limits.maxComputeInvocationsPerWorkgroup);
   }

   struct request_result device_result = { 0 };
   WGPUDeviceDescriptor device_desc = WGPU_DEVICE_DESCRIPTOR_INIT;
   device_desc.requiredFeatureCount = granted_count;
   device_desc.requiredFeatures = granted;
   if (have_limits)
      device_desc.requiredLimits = &limits;
   device_desc.uncapturedErrorCallbackInfo.callback = on_uncaptured_error;
   fprintf(stderr, "gex-wgpu: requesting %zu of %zu optional features\n",
           granted_count, sizeof(optional) / sizeof(optional[0]));
   WGPURequestDeviceCallbackInfo device_cb =
      WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
   device_cb.mode = WGPUCallbackMode_WaitAnyOnly;
   device_cb.callback = on_device;
   device_cb.userdata1 = &device_result;
   if (!wait_for(instance, wgpuAdapterRequestDevice(adapter_result.adapter,
                                                    &device_desc, device_cb)))
      return NULL;
   cached = device_result.device;
   return cached;
}

/* --- blocking readback ---------------------------------------------------
 *
 * WebGPU has no synchronous map, and Gallium's transfer interface is entirely
 * synchronous, so the driver bridges the two by spinning on its own `done`
 * flag around gex_wgpu_jspi_wait(). In the browser that call suspends the
 * whole wasm stack through JSPI and the page's event loop runs the callback.
 * Natively there is no stack to suspend and no page: the callback runs when
 * Dawn is asked to run it, which is what this does.
 *
 * The signal half exists only for JSPI's benefit -- something has to resolve
 * the promise -- and has nothing to do here, because by the time control
 * returns from ProcessEvents the callbacks have already run.
 */
void gex_wgpu_jspi_wait(void);
void gex_wgpu_jspi_signal(void);

void
gex_wgpu_jspi_wait(void)
{
   if (!native_instance) return;
   wgpuInstanceProcessEvents(native_instance);
   /* Yield rather than sleep, for a while.
    *
    * A readback finishes in about a millisecond, and nanosleep cannot be
    * asked for less than the scheduler's granularity -- ask for 20 us and
    * get 60 -- so a fixed sleep per poll spent most of the wait asleep past
    * the point the data had arrived. That showed up as 43% of Minecraft's
    * frame inside flush_frontbuffer. sched_yield returns immediately when
    * nothing else wants the core, which makes this a polite spin.
    *
    * The sleep is kept for the tail: if something really is not coming back,
    * burning a core on it helps nobody. */
   static unsigned spins;
   if (++spins < 4096) {
      sched_yield();
      return;
   }
   const struct timespec tick = { .tv_sec = 0, .tv_nsec = 50000 };
   nanosleep(&tick, NULL);
}

/* Reset by the caller's success, so a long wait does not leave the next one
 * starting in its slow tail. */
void
gex_wgpu_jspi_signal(void)
{
}

/* --- presentation ------------------------------------------------------- */

static void
gex_wgpu_sw_flush_frontbuffer(struct pipe_screen *screen,
                              struct pipe_context *pipe,
                              struct pipe_resource *resource,
                              unsigned level, unsigned layer,
                              void *context_private, unsigned nboxes,
                              struct pipe_box *sub_box)
{
   struct gex_wgpu_sw_screen *state = sw_state_for(screen);
   if (!state || !state->winsys || !resource || !pipe) return;
   extern uint64_t gex_wgpu_present_nanos;
   const int64_t present_start = os_time_get_nano();
   struct sw_winsys *winsys = state->winsys;

   if (state->target &&
       (state->target_width != resource->width0 ||
        state->target_height != resource->height0 ||
        state->target_format != resource->format)) {
      winsys->displaytarget_destroy(winsys, state->target);
      state->target = NULL;
   }
   if (!state->target) {
      unsigned stride = 0;
      state->target = winsys->displaytarget_create(
         winsys, PIPE_BIND_DISPLAY_TARGET, resource->format,
         resource->width0, resource->height0, 64, NULL, &stride);
      if (!state->target) return;
      /* The winsys picks this, and it is not width * bytes-per-pixel: the
       * X11 backend pads every row out to its own alignment. Computing it
       * here instead of keeping what displaytarget_create returned shifted
       * each row a little further than the last, which reads as a shear --
       * glxgears' gears came out as diagonal streaks with, tellingly, exactly
       * the right pixels in them. */
      state->target_stride = stride;
      state->target_width = resource->width0;
      state->target_height = resource->height0;
      state->target_format = resource->format;
   }

   /* GPU to CPU, once a frame. The driver's own transfer path is what
    * glReadPixels uses, so this needs nothing of its internals. */
   struct pipe_transfer *transfer = NULL;
   struct pipe_box box;
   u_box_2d(0, 0, resource->width0, resource->height0, &box);
   uint8_t *from = pipe->texture_map(pipe, resource, level, PIPE_MAP_READ,
                                     &box, &transfer);
   if (!from) return;

   /* What actually reached the front buffer, as opposed to what the driver
    * believes it drew. Minecraft 26.2 rendered 148 draws a frame at 30 fps
    * into a window that screenshotted pure black, and no counter in the
    * driver could tell "the draws went somewhere else" from "the copy to X
    * is broken". GEX_WGPU_PRESENT_LOG=1 prints the presented pixels; set it
    * to a path prefix to also write the first frame and every 300th as a
    * PPM. */
   {
      static int present_log = -1;
      static const char *prefix;
      if (present_log < 0) {
         const char *value = getenv("GEX_WGPU_PRESENT_LOG");
         present_log = value && *value ? 1 : 0;
         prefix = value && *value != '1' ? value : NULL;
      }
      if (present_log) {
         static uint64_t seen;
         const unsigned bpp = util_format_get_blocksize(resource->format);
         uint64_t sum = 0, nonzero = 0;
         for (unsigned y = 0; y < resource->height0; ++y) {
            const uint8_t *row = from + (size_t)y * transfer->stride;
            for (unsigned x = 0; x < resource->width0 * bpp; ++x) {
               sum += row[x];
               nonzero += row[x] != 0;
            }
         }
         const double bytes =
            (double)resource->width0 * resource->height0 * bpp;
         if (seen % 30 == 0) {
            /* Which colour targets the frame's draws actually went to. A
             * black present with a busy driver is either "the draws landed
             * elsewhere" or "the copy to X is broken", and the histogram is
             * what separates them. */
            extern void gex_wgpu_report_draw_targets(void);
            gex_wgpu_report_draw_targets();
         }
         if (seen % 30 == 0)
            fprintf(stderr,
                    "gex-wgpu: present %llu %ux%u fmt=%d stride=%u "
                    "mean=%.1f nonzero=%.1f%%\n",
                    (unsigned long long)seen, resource->width0,
                    resource->height0, (int)resource->format,
                    transfer->stride, sum / bytes, 100.0 * nonzero / bytes);
         {
            extern const char *gex_wgpu_presented_trace_take(void);
            const char *trace = gex_wgpu_presented_trace_take();
            if (seen % 30 == 0)
               fprintf(stderr, "gex-wgpu: on the window this frame: %s\n",
                       trace[0] ? trace : "nothing");
         }
         if (prefix && (seen == 0 || seen % 300 == 0) && bpp == 4) {
            char path[512];
            snprintf(path, sizeof(path), "%s-%llu.ppm", prefix,
                     (unsigned long long)seen);
            FILE *out = fopen(path, "wb");
            if (out) {
               fprintf(out, "P6\n%u %u\n255\n", resource->width0,
                       resource->height0);
               for (unsigned y = 0; y < resource->height0; ++y) {
                  const uint8_t *row = from + (size_t)y * transfer->stride;
                  for (unsigned x = 0; x < resource->width0; ++x)
                     fwrite(row + (size_t)x * 4, 1, 3, out);
               }
               fclose(out);
            }
         }
         ++seen;
      }
   }

   uint8_t *to = winsys->displaytarget_map(winsys, state->target,
                                           PIPE_MAP_WRITE);
   if (to) {
      const unsigned bytes =
         util_format_get_stride(resource->format, resource->width0);
      const unsigned rows =
         util_format_get_nblocksy(resource->format, resource->height0);
      /* Never wider than either side's row. */
      const unsigned copy = MIN3(bytes, state->target_stride,
                                 transfer->stride);
      for (unsigned y = 0; y < rows; ++y)
         memcpy(to + (size_t)y * state->target_stride,
                from + (size_t)y * transfer->stride, copy);
      winsys->displaytarget_unmap(winsys, state->target);
   }
   pipe->texture_unmap(pipe, transfer);
   (void)layer;

   /* Name the texture the window is actually made of, so the per-draw
    * counter that has always been reported as "presented buffer: N" means
    * something. It was never assigned, so it read 0 whatever happened --
    * which is indistinguishable from a frame whose draws all went elsewhere,
    * the exact question Minecraft 26.2 raised. */
   {
      extern WGPUTexture gex_wgpu_presented_texture;
      extern WGPUTexture gex_wgpu_resource_texture(struct pipe_resource *);
      gex_wgpu_presented_texture = gex_wgpu_resource_texture(resource);
   }

   extern uint64_t gex_wgpu_presents;
   ++gex_wgpu_presents;
   /* The frame boundary the draw bisection counts from. */
   extern uint64_t gex_wgpu_frame_draw;
   gex_wgpu_frame_draw = 0;
   gex_wgpu_present_nanos += os_time_get_nano() - present_start;
   winsys->displaytarget_display(winsys, state->target, context_private,
                                 nboxes, sub_box);
   (void)screen;
}

static void
gex_wgpu_sw_destroy(struct pipe_screen *screen)
{
   struct gex_wgpu_sw_screen *state = sw_state_for(screen);
   if (state) {
      if (state->target && state->winsys)
         state->winsys->displaytarget_destroy(state->winsys, state->target);
      void (*wrapped)(struct pipe_screen *) = state->wrapped_destroy;
      for (unsigned i = 0; i < GEX_WGPU_SW_SCREENS; ++i)
         if (sw_screens[i].screen == screen) {
            sw_screens[i].screen = NULL;
            sw_screens[i].state = NULL;
         }
      FREE(state);
      if (wrapped) wrapped(screen);
   }
}

struct pipe_screen *
gex_wgpu_sw_screen_create(struct sw_winsys *winsys)
{
   WGPUDevice device = create_device();
   if (!device) {
      fprintf(stderr, "gex-wgpu: no native WebGPU device; not creating a "
                      "screen\n");
      return NULL;
   }
   struct pipe_screen *screen = gex_wgpu_screen_create(device);
   if (!screen) return NULL;

   struct gex_wgpu_sw_screen *state = CALLOC_STRUCT(gex_wgpu_sw_screen);
   if (!state) return screen;
   state->winsys = winsys;
   state->wrapped_destroy = screen->destroy;
   unsigned slot = GEX_WGPU_SW_SCREENS;
   for (unsigned i = 0; i < GEX_WGPU_SW_SCREENS; ++i)
      if (!sw_screens[i].screen) { slot = i; break; }
   if (slot == GEX_WGPU_SW_SCREENS) {
      /* Eight display connections is already more than any toolkit seen
       * here opens; a ninth would present through a stale drawable, so say
       * so rather than do that. */
      fprintf(stderr, "gex-wgpu: more than %u screens; not presenting for "
                      "this one\n", (unsigned)GEX_WGPU_SW_SCREENS);
      FREE(state);
      return screen;
   }
   sw_screens[slot].screen = screen;
   sw_screens[slot].state = state;
   screen->flush_frontbuffer = gex_wgpu_sw_flush_frontbuffer;
   screen->destroy = gex_wgpu_sw_destroy;
   fprintf(stderr, "gex-wgpu: native screen on Dawn\n");
   return screen;
}
