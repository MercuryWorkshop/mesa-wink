/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef WVK_WSI_H
#define WVK_WSI_H

#include "wvk_private.h"

#include <vulkan/vk_icd.h>
#include "vk_object.h"

struct wvk_instance;
struct wvk_swapchain;

/* A surface the driver made: the loader's layout first, as since loader
 * interface v3 surfaces are the ICD's, then the WGPUSurface. */
struct wvk_surface {
   union {
      VkIcdSurfaceBase base;
      VkIcdSurfaceHeadless headless;
#ifdef VK_USE_PLATFORM_METAL_EXT
      VkIcdSurfaceMetal metal;
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
      VkIcdSurfaceXcb xcb;
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
      VkIcdSurfaceXlib xlib;
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
      VkIcdSurfaceWayland wayland;
#endif
   } icd;

   /* NULL for a headless surface: presenting to it shows nothing. */
   WGPUSurface wgpu;

   /* The swapchain the surface is configured for: the newest one. */
   struct wvk_swapchain *swapchain;
};

static inline struct wvk_surface *
wvk_surface_from_handle(VkSurfaceKHR handle)
{
   return (struct wvk_surface *)(uintptr_t)handle;
}

static inline VkSurfaceKHR
wvk_surface_to_handle(struct wvk_surface *surface)
{
   return (VkSurfaceKHR)(uintptr_t)surface;
}

/* Wrap a WGPUSurface, which the new surface owns, as a VkSurfaceKHR; for
 * the platforms Vulkan has no surface extension for, such as a canvas. */
VkResult wvk_surface_create(struct wvk_instance *instance, VkIcdWsiPlatform platform,
                            WGPUSurface wgpu, const VkAllocationCallbacks *pAllocator,
                            VkSurfaceKHR *pSurface);

struct wvk_swapchain {
   struct vk_object_base base;

   struct wvk_surface *surface;
   /* A newer swapchain took the surface over: acquiring is out of date. */
   bool retired;

   VkExtent2D extent;
   VkFormat format;
   /* What the surface is configured with: the images' format, or its
    * sRGB twin, which a copy converts nothing between. */
   WGPUTextureFormat surface_format;

   uint32_t image_count;
   uint32_t next;
   VkImage images[16];
   bool acquired[16];
};

VK_DEFINE_NONDISP_HANDLE_CASTS(wvk_swapchain, base, VkSwapchainKHR, VK_OBJECT_TYPE_SWAPCHAIN_KHR)

#endif
