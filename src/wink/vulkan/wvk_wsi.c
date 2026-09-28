/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * Surfaces, swapchains and present. A surface is a WGPUSurface, and a
 * swapchain's images are ordinary images of the driver's: WebGPU hands out
 * a new texture every frame, where Vulkan wants the images an application
 * made framebuffers of to stay. Present copies the image into the
 * surface's texture of the moment and presents that. The one in-order
 * queue makes this safe without waiting for anything: the copy is queued
 * after what rendered the image and before whatever renders into it next,
 * so acquiring never waits either. What paces an application is the
 * surface: Dawn's getCurrentTexture blocks in FIFO mode until a texture
 * is free again.
 */
#include "wvk_wsi.h"

#include "wvk_device.h"
#include "wvk_entrypoints.h"
#include "wvk_image.h"
#include "wvk_instance.h"
#include "wvk_physical_device.h"

#include "vk_enum_to_str.h"
#include "vk_fence.h"
#include "vk_semaphore.h"
#include "vk_sync.h"

#include "util/os_time.h"
#include "wvk_platform.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/stack_array.h"

#ifdef VK_USE_PLATFORM_METAL_EXT
#include <objc/message.h>
#include <objc/runtime.h>
#endif

VkResult
wvk_surface_create(struct wvk_instance *instance, VkIcdWsiPlatform platform, WGPUSurface wgpu,
                   const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface)
{
   struct wvk_surface *surface = vk_zalloc2(&instance->vk.alloc, pAllocator, sizeof(*surface), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!surface) {
      if (wgpu)
         wgpuSurfaceRelease(wgpu);
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   surface->icd.base.platform = platform;
   surface->wgpu = wgpu;
   *pSurface = wvk_surface_to_handle(surface);
   return VK_SUCCESS;
}

/* A surface over a native window: what Dawn makes of it. */
static VkResult
create_native_surface(VkInstance _instance, const void *icd, size_t icd_size,
                      WGPUChainedStruct *source, const VkAllocationCallbacks *pAllocator,
                      VkSurfaceKHR *pSurface)
{
   VK_FROM_HANDLE(wvk_instance, instance, _instance);
   WGPUInstance wgpu_instance = wvk_instance_wgpu(instance);
   if (!wgpu_instance)
      return vk_error(instance, VK_ERROR_INITIALIZATION_FAILED);

   WGPUSurfaceDescriptor desc = WGPU_SURFACE_DESCRIPTOR_INIT;
   desc.nextInChain = source;
   desc.label = wvk_sv("VkSurfaceKHR");
   WGPUSurface wgpu = wgpuInstanceCreateSurface(wgpu_instance, &desc);
   if (!wgpu)
      return vk_errorf(instance, VK_ERROR_INITIALIZATION_FAILED, "WebGPU made no surface of the window");

   const VkIcdSurfaceBase *base = icd;
   VkResult result = wvk_surface_create(instance, base->platform, wgpu, pAllocator, pSurface);
   if (result == VK_SUCCESS)
      memcpy(&wvk_surface_from_handle(*pSurface)->icd, icd, icd_size);
   return result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateHeadlessSurfaceEXT(VkInstance _instance, const VkHeadlessSurfaceCreateInfoEXT *pCreateInfo,
                             const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface)
{
   VK_FROM_HANDLE(wvk_instance, instance, _instance);
   return wvk_surface_create(instance, VK_ICD_WSI_PLATFORM_HEADLESS, NULL, pAllocator, pSurface);
}

#ifdef VK_USE_PLATFORM_METAL_EXT
VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateMetalSurfaceEXT(VkInstance _instance, const VkMetalSurfaceCreateInfoEXT *pCreateInfo,
                          const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface)
{
   const VkIcdSurfaceMetal icd = {
      .base.platform = VK_ICD_WSI_PLATFORM_METAL,
      .pLayer = pCreateInfo->pLayer,
   };
   WGPUSurfaceSourceMetalLayer source = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
   source.layer = (void *)pCreateInfo->pLayer;
   return create_native_surface(_instance, &icd, sizeof(icd), &source.chain, pAllocator, pSurface);
}

/* CGRect, without CoreGraphics: CGFloat is a double on 64 bits. */
struct wvk_cg_rect {
   double x, y, width, height;
};

/* What MoltenVK calls the layer's natural drawable size: its bounds in
 * pixels. The drawable size itself is whatever the swapchain set. */
static VkExtent2D
metal_layer_extent(const CAMetalLayer *layer)
{
   id obj = (id)layer;
   struct wvk_cg_rect bounds;
   double scale;
#if defined(__x86_64__)
   ((void (*)(struct wvk_cg_rect *, id, SEL))objc_msgSend_stret)(&bounds, obj, sel_registerName("bounds"));
   scale = ((double (*)(id, SEL))objc_msgSend_fpret)(obj, sel_registerName("contentsScale"));
#else
   bounds = ((struct wvk_cg_rect (*)(id, SEL))objc_msgSend)(obj, sel_registerName("bounds"));
   scale = ((double (*)(id, SEL))objc_msgSend)(obj, sel_registerName("contentsScale"));
#endif
   return (VkExtent2D){
      .width = (uint32_t)MAX2(bounds.width * scale + 0.5, 0.0),
      .height = (uint32_t)MAX2(bounds.height * scale + 0.5, 0.0),
   };
}
#endif

#ifdef VK_USE_PLATFORM_XCB_KHR
VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateXcbSurfaceKHR(VkInstance _instance, const VkXcbSurfaceCreateInfoKHR *pCreateInfo,
                        const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface)
{
   const VkIcdSurfaceXcb icd = {
      .base.platform = VK_ICD_WSI_PLATFORM_XCB,
      .connection = pCreateInfo->connection,
      .window = pCreateInfo->window,
   };
   WGPUSurfaceSourceXCBWindow source = WGPU_SURFACE_SOURCE_XCB_WINDOW_INIT;
   source.connection = pCreateInfo->connection;
   source.window = pCreateInfo->window;
   return create_native_surface(_instance, &icd, sizeof(icd), &source.chain, pAllocator, pSurface);
}
#endif

#ifdef VK_USE_PLATFORM_XLIB_KHR
VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateXlibSurfaceKHR(VkInstance _instance, const VkXlibSurfaceCreateInfoKHR *pCreateInfo,
                         const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface)
{
   const VkIcdSurfaceXlib icd = {
      .base.platform = VK_ICD_WSI_PLATFORM_XLIB,
      .dpy = pCreateInfo->dpy,
      .window = pCreateInfo->window,
   };
   WGPUSurfaceSourceXlibWindow source = WGPU_SURFACE_SOURCE_XLIB_WINDOW_INIT;
   source.display = pCreateInfo->dpy;
   source.window = pCreateInfo->window;
   return create_native_surface(_instance, &icd, sizeof(icd), &source.chain, pAllocator, pSurface);
}
#endif

#ifdef VK_USE_PLATFORM_WAYLAND_KHR
VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateWaylandSurfaceKHR(VkInstance _instance, const VkWaylandSurfaceCreateInfoKHR *pCreateInfo,
                            const VkAllocationCallbacks *pAllocator, VkSurfaceKHR *pSurface)
{
   const VkIcdSurfaceWayland icd = {
      .base.platform = VK_ICD_WSI_PLATFORM_WAYLAND,
      .display = pCreateInfo->display,
      .surface = pCreateInfo->surface,
   };
   WGPUSurfaceSourceWaylandSurface source = WGPU_SURFACE_SOURCE_WAYLAND_SURFACE_INIT;
   source.display = pCreateInfo->display;
   source.surface = pCreateInfo->surface;
   return create_native_surface(_instance, &icd, sizeof(icd), &source.chain, pAllocator, pSurface);
}
#endif

VKAPI_ATTR void VKAPI_CALL
wvk_DestroySurfaceKHR(VkInstance _instance, VkSurfaceKHR _surface,
                      const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_instance, instance, _instance);
   struct wvk_surface *surface = wvk_surface_from_handle(_surface);
   if (!surface)
      return;
   if (surface->wgpu)
      wgpuSurfaceRelease(surface->wgpu);
   vk_free2(&instance->vk.alloc, pAllocator, surface);
}

/* The size of the window, or UINT32_MAX where the swapchain decides it:
 * on a headless surface, and where the window cannot be asked. */
static VkExtent2D
surface_current_extent(const struct wvk_surface *surface)
{
#ifdef VK_USE_PLATFORM_METAL_EXT
   if (surface->icd.base.platform == VK_ICD_WSI_PLATFORM_METAL)
      return metal_layer_extent(surface->icd.metal.pLayer);
#endif
   return (VkExtent2D){ UINT32_MAX, UINT32_MAX };
}

/* Whether the window no longer has the swapchain's size: the application
 * should make a new one, though presenting still works, scaled. */
static bool
swapchain_suboptimal(const struct wvk_swapchain *sc)
{
   const VkExtent2D extent = surface_current_extent(sc->surface);
   return extent.width != UINT32_MAX &&
          (extent.width != sc->extent.width || extent.height != sc->extent.height);
}

/* What WebGPU says of the surface on this adapter. The caller frees the
 * members; false for a headless surface, or failure. */
static bool
get_capabilities(const struct wvk_physical_device *pdev, const struct wvk_surface *surface,
                 WGPUSurfaceCapabilities *caps)
{
   *caps = (WGPUSurfaceCapabilities)WGPU_SURFACE_CAPABILITIES_INIT;
   if (!surface->wgpu)
      return false;
   return wgpuSurfaceGetCapabilities(surface->wgpu, pdev->adapter, caps) == WGPUStatus_Success;
}

static bool
has_format(const WGPUSurfaceCapabilities *caps, WGPUTextureFormat format)
{
   for (size_t i = 0; i < caps->formatCount; ++i) {
      if (caps->formats[i] == format)
         return true;
   }
   return false;
}

/* The formats a swapchain can have, and what the surface is configured
 * with for each: sRGB or not is only how the image is viewed, and a copy
 * between the two twins converts nothing, so either twin of what the
 * surface takes will do. sRGB first, as Mesa's own WSI orders them. */
static const struct {
   VkFormat vk;
   WGPUTextureFormat linear;
   WGPUTextureFormat srgb;
} surface_formats[] = {
   { VK_FORMAT_B8G8R8A8_SRGB, WGPUTextureFormat_BGRA8Unorm, WGPUTextureFormat_BGRA8UnormSrgb },
   { VK_FORMAT_B8G8R8A8_UNORM, WGPUTextureFormat_BGRA8Unorm, WGPUTextureFormat_BGRA8UnormSrgb },
   { VK_FORMAT_R8G8B8A8_SRGB, WGPUTextureFormat_RGBA8Unorm, WGPUTextureFormat_RGBA8UnormSrgb },
   { VK_FORMAT_R8G8B8A8_UNORM, WGPUTextureFormat_RGBA8Unorm, WGPUTextureFormat_RGBA8UnormSrgb },
   { VK_FORMAT_A2B10G10R10_UNORM_PACK32, WGPUTextureFormat_RGB10A2Unorm, WGPUTextureFormat_RGB10A2Unorm },
};

/* The format to configure the surface with for images of `format`, or
 * Undefined if it cannot take them. A headless surface takes the 8-bit
 * ones. */
static WGPUTextureFormat
configure_format(const WGPUSurfaceCapabilities *caps, bool headless, VkFormat format)
{
   for (size_t i = 0; i < ARRAY_SIZE(surface_formats); ++i) {
      if (surface_formats[i].vk != format)
         continue;
      if (headless)
         return surface_formats[i].linear == WGPUTextureFormat_RGB10A2Unorm
                   ? WGPUTextureFormat_Undefined : surface_formats[i].linear;
      if (has_format(caps, surface_formats[i].linear))
         return surface_formats[i].linear;
      if (has_format(caps, surface_formats[i].srgb))
         return surface_formats[i].srgb;
   }
   return WGPUTextureFormat_Undefined;
}

static VkPresentModeKHR
vk_present_mode(WGPUPresentMode mode)
{
   switch (mode) {
   case WGPUPresentMode_Immediate: return VK_PRESENT_MODE_IMMEDIATE_KHR;
   case WGPUPresentMode_Mailbox: return VK_PRESENT_MODE_MAILBOX_KHR;
   case WGPUPresentMode_FifoRelaxed: return VK_PRESENT_MODE_FIFO_RELAXED_KHR;
   default: return VK_PRESENT_MODE_FIFO_KHR;
   }
}

static WGPUPresentMode
wgpu_present_mode(VkPresentModeKHR mode)
{
   switch (mode) {
   case VK_PRESENT_MODE_IMMEDIATE_KHR: return WGPUPresentMode_Immediate;
   case VK_PRESENT_MODE_MAILBOX_KHR: return WGPUPresentMode_Mailbox;
   case VK_PRESENT_MODE_FIFO_RELAXED_KHR: return WGPUPresentMode_FifoRelaxed;
   default: return WGPUPresentMode_Fifo;
   }
}

static VkCompositeAlphaFlagsKHR
vk_alpha_modes(const WGPUSurfaceCapabilities *caps)
{
   VkCompositeAlphaFlagsKHR flags = 0;
   for (size_t i = 0; i < caps->alphaModeCount; ++i) {
      switch (caps->alphaModes[i]) {
      case WGPUCompositeAlphaMode_Opaque: flags |= VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR; break;
      case WGPUCompositeAlphaMode_Premultiplied: flags |= VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR; break;
      case WGPUCompositeAlphaMode_Unpremultiplied: flags |= VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR; break;
      case WGPUCompositeAlphaMode_Inherit: flags |= VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR; break;
      default: break;
      }
   }
   return flags ? flags : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
}

static WGPUCompositeAlphaMode
wgpu_alpha_mode(VkCompositeAlphaFlagBitsKHR mode)
{
   switch (mode) {
   case VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR: return WGPUCompositeAlphaMode_Premultiplied;
   case VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR: return WGPUCompositeAlphaMode_Unpremultiplied;
   case VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR: return WGPUCompositeAlphaMode_Inherit;
   default: return WGPUCompositeAlphaMode_Opaque;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceSurfaceSupportKHR(VkPhysicalDevice physicalDevice, uint32_t queueFamilyIndex,
                                       VkSurfaceKHR _surface, VkBool32 *pSupported)
{
   /* The one queue family presents to anything. */
   *pSupported = queueFamilyIndex == 0;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceSurfaceCapabilitiesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR _surface,
                                            VkSurfaceCapabilitiesKHR *pSurfaceCapabilities)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);
   struct wvk_surface *surface = wvk_surface_from_handle(_surface);

   WGPUSurfaceCapabilities caps;
   const bool have_caps = get_capabilities(pdev, surface, &caps);
   if (surface->wgpu && !have_caps)
      return VK_ERROR_SURFACE_LOST_KHR;
   if (have_caps && !(caps.usages & WGPUTextureUsage_CopyDst))
      mesa_logw("wvk: the surface cannot be copied to, which presenting needs");

   const uint32_t max_dim = pdev->limits.maxTextureDimension2D;
   *pSurfaceCapabilities = (VkSurfaceCapabilitiesKHR){
      .minImageCount = 2,
      .maxImageCount = ARRAY_SIZE(((struct wvk_swapchain *)NULL)->images),
      .currentExtent = surface_current_extent(surface),
      .minImageExtent = { 1, 1 },
      .maxImageExtent = { max_dim, max_dim },
      .maxImageArrayLayers = 1,
      .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .supportedCompositeAlpha = have_caps ? vk_alpha_modes(&caps) : VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      /* The images are the driver's own, and only ever copied from. */
      .supportedUsageFlags = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                             VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
   };
   /* An empty window cannot have a swapchain; it has to be at least 1x1. */
   VkExtent2D *current = &pSurfaceCapabilities->currentExtent;
   if (current->width != UINT32_MAX) {
      current->width = CLAMP(current->width, 1, max_dim);
      current->height = CLAMP(current->height, 1, max_dim);
   }
   if (have_caps)
      wgpuSurfaceCapabilitiesFreeMembers(caps);
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceSurfaceCapabilities2KHR(VkPhysicalDevice physicalDevice,
                                             const VkPhysicalDeviceSurfaceInfo2KHR *pSurfaceInfo,
                                             VkSurfaceCapabilities2KHR *pSurfaceCapabilities)
{
   vk_foreach_struct(stype, ext, pSurfaceCapabilities->pNext)
      vk_debug_ignored_stype(stype);
   return wvk_GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, pSurfaceInfo->surface,
                                                      &pSurfaceCapabilities->surfaceCapabilities);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceSurfaceFormatsKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR _surface,
                                       uint32_t *pSurfaceFormatCount,
                                       VkSurfaceFormatKHR *pSurfaceFormats)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);
   struct wvk_surface *surface = wvk_surface_from_handle(_surface);
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, pSurfaceFormats, pSurfaceFormatCount);

   WGPUSurfaceCapabilities caps;
   const bool have_caps = get_capabilities(pdev, surface, &caps);
   if (surface->wgpu && !have_caps)
      return VK_ERROR_SURFACE_LOST_KHR;
   for (size_t i = 0; i < ARRAY_SIZE(surface_formats); ++i) {
      if (configure_format(&caps, !surface->wgpu, surface_formats[i].vk) == WGPUTextureFormat_Undefined)
         continue;
      vk_outarray_append_typed(VkSurfaceFormatKHR, &out, f) {
         f->format = surface_formats[i].vk;
         f->colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
      }
   }
   if (have_caps)
      wgpuSurfaceCapabilitiesFreeMembers(caps);
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceSurfaceFormats2KHR(VkPhysicalDevice physicalDevice,
                                        const VkPhysicalDeviceSurfaceInfo2KHR *pSurfaceInfo,
                                        uint32_t *pSurfaceFormatCount,
                                        VkSurfaceFormat2KHR *pSurfaceFormats)
{
   uint32_t count;
   VkResult result = wvk_GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, pSurfaceInfo->surface,
                                                            &count, NULL);
   if (result != VK_SUCCESS)
      return result;
   STACK_ARRAY(VkSurfaceFormatKHR, formats, count);
   result = wvk_GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, pSurfaceInfo->surface,
                                                   &count, formats);
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormat2KHR, out, pSurfaceFormats, pSurfaceFormatCount);
   for (uint32_t i = 0; result == VK_SUCCESS && i < count; ++i) {
      vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, f) {
         f->surfaceFormat = formats[i];
      }
   }
   STACK_ARRAY_FINISH(formats);
   return result == VK_SUCCESS ? vk_outarray_status(&out) : result;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceSurfacePresentModesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR _surface,
                                            uint32_t *pPresentModeCount,
                                            VkPresentModeKHR *pPresentModes)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);
   struct wvk_surface *surface = wvk_surface_from_handle(_surface);
   VK_OUTARRAY_MAKE_TYPED(VkPresentModeKHR, out, pPresentModes, pPresentModeCount);

   WGPUSurfaceCapabilities caps;
   if (!get_capabilities(pdev, surface, &caps)) {
      if (surface->wgpu)
         return VK_ERROR_SURFACE_LOST_KHR;
      /* Presenting to nothing never waits. */
      static const VkPresentModeKHR headless[] = {
         VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_MAILBOX_KHR,
      };
      for (size_t i = 0; i < ARRAY_SIZE(headless); ++i)
         vk_outarray_append_typed(VkPresentModeKHR, &out, m) *m = headless[i];
      return vk_outarray_status(&out);
   }
   for (size_t i = 0; i < caps.presentModeCount; ++i)
      vk_outarray_append_typed(VkPresentModeKHR, &out, m) *m = vk_present_mode(caps.presentModes[i]);
   wgpuSurfaceCapabilitiesFreeMembers(caps);
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDevicePresentRectanglesKHR(VkPhysicalDevice physicalDevice, VkSurfaceKHR _surface,
                                          uint32_t *pRectCount, VkRect2D *pRects)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);
   struct wvk_surface *surface = wvk_surface_from_handle(_surface);
   VK_OUTARRAY_MAKE_TYPED(VkRect2D, out, pRects, pRectCount);
   VkExtent2D extent = surface_current_extent(surface);
   if (extent.width == UINT32_MAX)
      extent = (VkExtent2D){ pdev->limits.maxTextureDimension2D, pdev->limits.maxTextureDimension2D };
   vk_outarray_append_typed(VkRect2D, &out, rect) {
      *rect = (VkRect2D){ .extent = extent };
   }
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetDeviceGroupPresentCapabilitiesKHR(VkDevice device,
                                         VkDeviceGroupPresentCapabilitiesKHR *pCapabilities)
{
   memset(pCapabilities->presentMask, 0, sizeof(pCapabilities->presentMask));
   pCapabilities->presentMask[0] = 0x1;
   pCapabilities->modes = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetDeviceGroupSurfacePresentModesKHR(VkDevice device, VkSurfaceKHR surface,
                                         VkDeviceGroupPresentModeFlagsKHR *pModes)
{
   *pModes = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR;
   return VK_SUCCESS;
}

static void
destroy_images(struct wvk_device *dev, struct wvk_swapchain *sc)
{
   for (uint32_t i = 0; i < sc->image_count; ++i)
      wvk_DestroyImage(wvk_device_to_handle(dev), sc->images[i], NULL);
   sc->image_count = 0;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateSwapchainKHR(VkDevice _device, const VkSwapchainCreateInfoKHR *pCreateInfo,
                       const VkAllocationCallbacks *pAllocator, VkSwapchainKHR *pSwapchain)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_swapchain, old, pCreateInfo->oldSwapchain);
   struct wvk_physical_device *pdev = wvk_device_physical(dev);
   struct wvk_surface *surface = wvk_surface_from_handle(pCreateInfo->surface);
   VkResult result;

   WGPUSurfaceCapabilities caps;
   const bool have_caps = get_capabilities(pdev, surface, &caps);
   if (surface->wgpu && !have_caps)
      return vk_error(dev, VK_ERROR_SURFACE_LOST_KHR);
   const WGPUTextureFormat surface_format = configure_format(&caps, !surface->wgpu,
                                                             pCreateInfo->imageFormat);
   const bool copyable = !have_caps || (caps.usages & WGPUTextureUsage_CopyDst);
   if (have_caps)
      wgpuSurfaceCapabilitiesFreeMembers(caps);
   if (surface_format == WGPUTextureFormat_Undefined)
      return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED, "the surface does not take %s",
                       vk_Format_to_str(pCreateInfo->imageFormat));
   /* TODO: a draw instead of the copy, for surfaces that are only render
    * attachments. */
   if (!copyable)
      return vk_errorf(dev, VK_ERROR_INITIALIZATION_FAILED, "the surface cannot be copied to");

   struct wvk_swapchain *sc = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*sc),
                                               VK_OBJECT_TYPE_SWAPCHAIN_KHR);
   if (!sc)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);
   sc->surface = surface;
   sc->extent = pCreateInfo->imageExtent;
   sc->format = pCreateInfo->imageFormat;
   sc->surface_format = surface_format;

   VkImageFormatListCreateInfo format_list = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
   };
   if (pCreateInfo->flags & VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR) {
      const VkImageFormatListCreateInfo *list =
         vk_find_struct_const(pCreateInfo->pNext, IMAGE_FORMAT_LIST_CREATE_INFO);
      if (list)
         format_list = *list;
      format_list.pNext = NULL;
   }
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = format_list.viewFormatCount ? &format_list : NULL,
      .flags = (pCreateInfo->flags & VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR)
                  ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT : 0,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = pCreateInfo->imageFormat,
      .extent = { pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height, 1 },
      .mipLevels = 1,
      .arrayLayers = pCreateInfo->imageArrayLayers,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = pCreateInfo->imageUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   const uint32_t count = CLAMP(pCreateInfo->minImageCount, 2, ARRAY_SIZE(sc->images));
   for (uint32_t i = 0; i < count; ++i) {
      result = wvk_CreateImage(_device, &image_info, NULL, &sc->images[i]);
      if (result != VK_SUCCESS)
         goto fail;
      sc->image_count = i + 1;
   }

   if (surface->wgpu) {
      WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
      config.device = dev->wgpu;
      config.format = surface_format;
      config.usage = WGPUTextureUsage_CopyDst;
      config.width = sc->extent.width;
      config.height = sc->extent.height;
      config.alphaMode = wgpu_alpha_mode(pCreateInfo->compositeAlpha);
      config.presentMode = wgpu_present_mode(pCreateInfo->presentMode);
      wgpuSurfaceConfigure(surface->wgpu, &config);
   }

   /* What the old swapchain has acquired can still be presented; it just
    * cannot acquire any more. */
   if (old)
      old->retired = true;
   if (surface->swapchain && surface->swapchain != old)
      surface->swapchain->retired = true;
   surface->swapchain = sc;

   *pSwapchain = wvk_swapchain_to_handle(sc);
   return VK_SUCCESS;

fail:
   destroy_images(dev, sc);
   vk_object_free(&dev->vk, pAllocator, sc);
   return result;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroySwapchainKHR(VkDevice _device, VkSwapchainKHR swapchain,
                        const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_swapchain, sc, swapchain);
   if (!sc)
      return;
   if (sc->surface->swapchain == sc) {
      if (sc->surface->wgpu)
         wgpuSurfaceUnconfigure(sc->surface->wgpu);
      sc->surface->swapchain = NULL;
   }
   destroy_images(dev, sc);
   vk_object_free(&dev->vk, pAllocator, sc);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetSwapchainImagesKHR(VkDevice device, VkSwapchainKHR swapchain,
                          uint32_t *pSwapchainImageCount, VkImage *pSwapchainImages)
{
   VK_FROM_HANDLE(wvk_swapchain, sc, swapchain);
   VK_OUTARRAY_MAKE_TYPED(VkImage, out, pSwapchainImages, pSwapchainImageCount);
   for (uint32_t i = 0; i < sc->image_count; ++i)
      vk_outarray_append_typed(VkImage, &out, image) *image = sc->images[i];
   return vk_outarray_status(&out);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_AcquireNextImage2KHR(VkDevice _device, const VkAcquireNextImageInfoKHR *pAcquireInfo,
                         uint32_t *pImageIndex)
{
   VK_FROM_HANDLE(wvk_device, dev, _device);
   VK_FROM_HANDLE(wvk_swapchain, sc, pAcquireInfo->swapchain);
   VK_FROM_HANDLE(vk_semaphore, semaphore, pAcquireInfo->semaphore);
   VK_FROM_HANDLE(vk_fence, fence, pAcquireInfo->fence);
   VkResult result;

   if (vk_device_is_lost(&dev->vk))
      return VK_ERROR_DEVICE_LOST;
   if (sc->retired)
      return VK_ERROR_OUT_OF_DATE_KHR;

   /* Round robin, over what is not acquired: an image is ready as soon as
    * it was presented, as the copy out of it is queued before anything
    * else that uses it. */
   uint32_t index = UINT32_MAX;
   for (uint32_t i = 0; i < sc->image_count; ++i) {
      const uint32_t candidate = (sc->next + i) % sc->image_count;
      if (!sc->acquired[candidate]) {
         index = candidate;
         break;
      }
   }
   if (index == UINT32_MAX)
      return pAcquireInfo->timeout ? VK_TIMEOUT : VK_NOT_READY;
   sc->acquired[index] = true;
   sc->next = (index + 1) % sc->image_count;

   if (semaphore) {
      result = vk_sync_signal(&dev->vk, vk_semaphore_get_active_sync(semaphore), 0);
      if (result != VK_SUCCESS)
         return result;
   }
   if (fence) {
      result = vk_sync_signal(&dev->vk, vk_fence_get_active_sync(fence), 0);
      if (result != VK_SUCCESS)
         return result;
   }

   *pImageIndex = index;
   return swapchain_suboptimal(sc) ? VK_SUBOPTIMAL_KHR : VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_AcquireNextImageKHR(VkDevice device, VkSwapchainKHR swapchain, uint64_t timeout,
                        VkSemaphore semaphore, VkFence fence, uint32_t *pImageIndex)
{
   const VkAcquireNextImageInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_ACQUIRE_NEXT_IMAGE_INFO_KHR,
      .swapchain = swapchain,
      .timeout = timeout,
      .semaphore = semaphore,
      .fence = fence,
      .deviceMask = 0x1,
   };
   return wvk_AcquireNextImage2KHR(device, &info, pImageIndex);
}

/* What a present comes after: every submit the application made, in
 * WebGPU. Deferred submits wait for timeline points to become pending;
 * the application signalling those from another thread is what makes them
 * go, so poll for that. */
static VkResult
flush_submits(struct wvk_device *dev, struct wvk_queue *queue)
{
   for (;;) {
      VkResult result = vk_device_flush(&dev->vk);
      if (result != VK_SUCCESS)
         return result;
      if (vk_queue_is_empty(&queue->vk))
         return VK_SUCCESS;
      if (vk_device_is_lost(&dev->vk))
         return VK_ERROR_DEVICE_LOST;
      os_time_sleep(100);
   }
}

/* Worst first: an error, then out of date, then suboptimal. */
static VkResult
worse(VkResult a, VkResult b)
{
   if (a < 0)
      return a;
   if (b < 0)
      return b;
   return a == VK_SUBOPTIMAL_KHR ? a : b;
}

/* WINK_VK_CAPTURE=N:FILE.bmp writes what the Nth present shows to a file,
 * for looking at what reached the screen where nothing can capture it. */
static void
on_capture_mapped(WGPUMapAsyncStatus status, WGPUStringView message, void *userdata1, void *userdata2)
{
   *(bool *)userdata1 = status == WGPUMapAsyncStatus_Success;
}

static void
maybe_capture(struct wvk_device *dev, struct wvk_swapchain *sc, struct wvk_image *image)
{
   static int64_t frame = -1;
   static uint64_t presents;
   static char path[512];
   if (frame < 0) {
      const char *env = getenv("WINK_VK_CAPTURE");
      frame = 0;
      if (!env || sscanf(env, "%" SCNu64 ":%511s", &frame, path) != 2)
         frame = INT64_MAX;
   }
   if (++presents != (uint64_t)frame)
      return;

   const uint32_t w = sc->extent.width, h = sc->extent.height;
   const uint32_t pitch = align(w * 4, 256);
   WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
   desc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
   desc.size = (uint64_t)pitch * h;
   WGPUBuffer buffer = wgpuDeviceCreateBuffer(dev->wgpu, &desc);
   WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(dev->wgpu, NULL);
   const WGPUTexelCopyTextureInfo src = { .texture = image->texture };
   const WGPUTexelCopyBufferInfo dst = { .layout = { .bytesPerRow = pitch, .rowsPerImage = h }, .buffer = buffer };
   const WGPUExtent3D size = { w, h, 1 };
   wgpuCommandEncoderCopyTextureToBuffer(encoder, &src, &dst, &size);
   WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
   wgpuQueueSubmit(dev->queue.wgpu, 1, &commands);
   wgpuCommandBufferRelease(commands);
   wgpuCommandEncoderRelease(encoder);

   bool mapped = false;
   WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
   cb.mode = WGPUCallbackMode_WaitAnyOnly;
   cb.callback = on_capture_mapped;
   cb.userdata1 = &mapped;
   struct wvk_instance *instance = (struct wvk_instance *)dev->vk.physical->instance;
   wvk_platform_wait(instance->wgpu, wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, desc.size, cb), UINT64_MAX);
   const uint8_t *data = mapped ? wgpuBufferGetConstMappedRange(buffer, 0, desc.size) : NULL;
   FILE *f = data ? fopen(path, "wb") : NULL;
   if (f) {
      /* A bottom-up 32-bit BMP, which is BGRA: swap for RGBA images. */
      const bool rgba = image->format == WGPUTextureFormat_RGBA8Unorm ||
                        image->format == WGPUTextureFormat_RGBA8UnormSrgb;
      const uint32_t bytes = w * h * 4;
      uint8_t header[54] = { 'B', 'M' };
      const uint32_t fields[] = { 54 + bytes, 0, 54, 40, w, h };
      memcpy(header + 2, &fields[0], 4);
      memcpy(header + 10, &fields[2], 4);
      memcpy(header + 14, &fields[3], 4);
      memcpy(header + 18, &fields[4], 4);
      memcpy(header + 22, &fields[5], 4);
      header[26] = 1;
      header[28] = 32;
      fwrite(header, 1, sizeof(header), f);
      uint8_t *row = malloc(w * 4);
      for (uint32_t y = h; row && y-- > 0;) {
         memcpy(row, data + (uint64_t)y * pitch, w * 4);
         for (uint32_t x = 0; rgba && x < w; ++x) {
            const uint8_t r = row[x * 4];
            row[x * 4] = row[x * 4 + 2];
            row[x * 4 + 2] = r;
         }
         fwrite(row, 1, w * 4, f);
      }
      free(row);
      fclose(f);
      mesa_logi("wvk: present %" PRIu64 " written to %s", presents, path);
   }
   if (mapped)
      wgpuBufferUnmap(buffer);
   wgpuBufferRelease(buffer);
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_QueuePresentKHR(VkQueue _queue, const VkPresentInfoKHR *pPresentInfo)
{
   VK_FROM_HANDLE(vk_queue, vk_queue, _queue);
   struct wvk_queue *queue = container_of(vk_queue, struct wvk_queue, vk);
   struct wvk_device *dev = wvk_queue_device(queue);
   const uint32_t count = pPresentInfo->swapchainCount;

   if (vk_queue_is_lost(&queue->vk))
      return VK_ERROR_DEVICE_LOST;

   /* The binary semaphores waited on are pending once what signals them
    * is submitted; in order on the one queue, that is all it takes. */
   VkResult result = flush_submits(dev, queue);
   if (result != VK_SUCCESS)
      return result;

   STACK_ARRAY(WGPUSurfaceTexture, textures, count);
   STACK_ARRAY(VkResult, results, count);
   WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(dev->wgpu, NULL);
   for (uint32_t i = 0; i < count; ++i) {
      VK_FROM_HANDLE(wvk_swapchain, sc, pPresentInfo->pSwapchains[i]);
      const uint32_t index = pPresentInfo->pImageIndices[i];
      VK_FROM_HANDLE(wvk_image, image, sc->images[index]);
      struct wvk_surface *surface = sc->surface;

      sc->acquired[index] = false;
      maybe_capture(dev, sc, image);
      textures[i] = (WGPUSurfaceTexture)WGPU_SURFACE_TEXTURE_INIT;
      results[i] = swapchain_suboptimal(sc) ? VK_SUBOPTIMAL_KHR : VK_SUCCESS;
      /* An old swapchain's images, and a headless surface's, go nowhere. */
      if (!surface->wgpu || surface->swapchain != sc)
         continue;

      wgpuSurfaceGetCurrentTexture(surface->wgpu, &textures[i]);
      switch (textures[i].status) {
      case WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal:
         break;
      case WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal:
         results[i] = VK_SUBOPTIMAL_KHR;
         break;
      case WGPUSurfaceGetCurrentTextureStatus_Timeout:
         /* No texture to be had in time: this frame is dropped. */
         break;
      case WGPUSurfaceGetCurrentTextureStatus_Outdated:
         results[i] = VK_ERROR_OUT_OF_DATE_KHR;
         break;
      case WGPUSurfaceGetCurrentTextureStatus_Lost:
         results[i] = VK_ERROR_SURFACE_LOST_KHR;
         break;
      default:
         results[i] = VK_ERROR_DEVICE_LOST;
         break;
      }
      if (!textures[i].texture)
         continue;
      if (results[i] < 0) {
         wgpuTextureRelease(textures[i].texture);
         textures[i].texture = NULL;
         continue;
      }

      /* The surface is the swapchain's size; the image may be larger than
       * a surface that failed to follow, never smaller. */
      const WGPUTexelCopyTextureInfo src = { .texture = image->texture };
      const WGPUTexelCopyTextureInfo dst = { .texture = textures[i].texture };
      const WGPUExtent3D size = {
         MIN2(sc->extent.width, wgpuTextureGetWidth(textures[i].texture)),
         MIN2(sc->extent.height, wgpuTextureGetHeight(textures[i].texture)),
         1,
      };
      wgpuCommandEncoderCopyTextureToTexture(encoder, &src, &dst, &size);
   }
   WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
   wgpuQueueSubmit(queue->wgpu, 1, &commands);
   wgpuCommandBufferRelease(commands);
   wgpuCommandEncoderRelease(encoder);

   result = VK_SUCCESS;
   for (uint32_t i = 0; i < count; ++i) {
      VK_FROM_HANDLE(wvk_swapchain, sc, pPresentInfo->pSwapchains[i]);
      if (textures[i].texture) {
         wgpuSurfacePresent(sc->surface->wgpu);
         wgpuTextureRelease(textures[i].texture);
      }
      if (pPresentInfo->pResults)
         pPresentInfo->pResults[i] = results[i];
      result = worse(result, results[i]);
   }
   STACK_ARRAY_FINISH(results);
   STACK_ARRAY_FINISH(textures);
   return result;
}
