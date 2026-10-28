/*
 * Copyright 2026 wink contributors
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * SPDX-License-Identifier: MIT
 *
 * From kosmickrisp's kk_instance.c: the instance, and the proc-address
 * lookups that hand out wvk's trampolines -- the ones that reach a
 * device's exposed, recording dispatch table.
 */
#include "wvk_instance.h"

#include "wvk_dispatch_trampolines.h"
#include "wvk_entrypoints.h"
#include "wvk_physical_device.h"
#include "wvk_platform.h"

#include "git_sha1.h"
#include "util/mesa-blake3.h"

#include <stdlib.h>

VKAPI_ATTR VkResult VKAPI_CALL
wvk_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   const uint32_t override = vk_get_version_override();
   *pApiVersion = override ? override : WVK_API_VERSION;
   return VK_SUCCESS;
}

static const struct vk_instance_extension_table instance_extensions = {
   .KHR_device_group_creation = true,
   .KHR_external_fence_capabilities = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_physical_device_properties2 = true,
   .KHR_get_surface_capabilities2 = true,
   .KHR_portability_enumeration = true,
   .KHR_surface = true,
   .KHR_surface_maintenance1 = true,
   .EXT_surface_maintenance1 = true,
   .EXT_debug_report = true,
   .EXT_debug_utils = true,
   .EXT_headless_surface = true,
#ifdef VK_USE_PLATFORM_METAL_EXT
   .EXT_metal_surface = true,
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
   .KHR_wayland_surface = true,
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
   .KHR_xcb_surface = true,
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
   .KHR_xlib_surface = true,
#endif
};

VKAPI_ATTR VkResult VKAPI_CALL
wvk_EnumerateInstanceExtensionProperties(const char *pLayerName,
                                         uint32_t *pPropertyCount,
                                         VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);

   return vk_enumerate_instance_extension_properties(&instance_extensions,
                                                     pPropertyCount, pProperties);
}

/* A browser links the ICD directly, with no Vulkan loader to provide these
 * core queries. There are no layers in that configuration. Native loaders
 * enumerate their own layers independently of the ICD's empty list. */
VKAPI_ATTR VkResult VKAPI_CALL
wvk_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount, VkLayerProperties *pProperties)
{
   *pPropertyCount = 0;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_EnumerateDeviceLayerProperties(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                   VkLayerProperties *pProperties)
{
   *pPropertyCount = 0;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_CreateInstance(const VkInstanceCreateInfo *pCreateInfo,
                   const VkAllocationCallbacks *pAllocator,
                   VkInstance *pInstance)
{
   if (pAllocator == NULL)
      pAllocator = vk_default_allocator();

   struct wvk_instance *instance =
      vk_zalloc(pAllocator, sizeof(*instance), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_instance_dispatch_table dispatch_table;
   vk_instance_dispatch_table_from_entrypoints(&dispatch_table, &wvk_instance_entrypoints, true);

   VkResult result = vk_instance_init(&instance->vk, &instance_extensions,
                                      &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, instance);
      return result;
   }

   instance->vk.physical_devices.enumerate = wvk_enumerate_physical_devices;
   instance->vk.physical_devices.destroy = wvk_physical_device_destroy;

   /* This build, as far as caches are concerned: the version and commit.
    * There is no build-id to go by in a static wasm link. */
   static const char build[] = "wink-vulkan " PACKAGE_VERSION MESA_GIT_SHA1;
   blake3_hash hash;
   _mesa_blake3_compute(build, sizeof(build) - 1, hash);
   memcpy(instance->driver_uuid, hash, VK_UUID_SIZE);

   *pInstance = wvk_instance_to_handle(instance);
   return VK_SUCCESS;
}

WGPUInstance
wvk_instance_wgpu(struct wvk_instance *instance)
{
   if (!instance->wgpu)
      instance->wgpu = wvk_platform_create_instance();
   return instance->wgpu;
}

VKAPI_ATTR void VKAPI_CALL
wvk_DestroyInstance(VkInstance _instance, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(wvk_instance, instance, _instance);
   if (!instance)
      return;

   /* The physical devices hold adapters from this instance. */
   vk_instance_finish(&instance->vk);
   if (instance->wgpu)
      wgpuInstanceRelease(instance->wgpu);
   vk_free(&instance->vk.alloc, instance);
}

static PFN_vkVoidFunction
wvk_instance_get_proc_addr(const struct vk_instance *instance,
                           const struct vk_instance_entrypoint_table *entrypoints,
                           const char *name)
{
   PFN_vkVoidFunction func;

   /* The Vulkan 1.0 spec for vkGetInstanceProcAddr has a table of exactly
    * when we have to return valid function pointers, NULL, or it's left
    * undefined. */
   if (name == NULL)
      return NULL;

#define LOOKUP_VK_ENTRYPOINT(entrypoint)                                       \
   if (strcmp(name, "vk" #entrypoint) == 0)                                    \
      return (PFN_vkVoidFunction)entrypoints->entrypoint

   LOOKUP_VK_ENTRYPOINT(EnumerateInstanceExtensionProperties);
   LOOKUP_VK_ENTRYPOINT(EnumerateInstanceLayerProperties);
   LOOKUP_VK_ENTRYPOINT(EnumerateInstanceVersion);
   LOOKUP_VK_ENTRYPOINT(CreateInstance);
   /* GetInstanceProcAddr() can also be called with a NULL instance. */
   LOOKUP_VK_ENTRYPOINT(GetInstanceProcAddr);
#undef LOOKUP_VK_ENTRYPOINT

   /* Since ICD interface v7 these can also come through
    * vk_icdGetInstanceProcAddr. */
   if (strcmp(name, "vk_icdNegotiateLoaderICDInterfaceVersion") == 0)
      return (PFN_vkVoidFunction)vk_icdNegotiateLoaderICDInterfaceVersion;
   if (strcmp(name, "vk_icdGetPhysicalDeviceProcAddr") == 0)
      return (PFN_vkVoidFunction)vk_icdGetPhysicalDeviceProcAddr;

   if (instance == NULL)
      return NULL;

   func = vk_instance_dispatch_table_get_if_supported(
      &instance->dispatch_table, name, instance->app_info.api_version,
      &instance->enabled_extensions);
   if (func != NULL)
      return func;

   func = vk_physical_device_dispatch_table_get_if_supported(
      &wvk_physical_device_trampolines, name, instance->app_info.api_version,
      &instance->enabled_extensions);
   if (func != NULL)
      return func;

   return vk_device_dispatch_table_get_if_supported(
      &wvk_device_trampolines, name, instance->app_info.api_version,
      &instance->enabled_extensions, NULL);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
wvk_GetInstanceProcAddr(VkInstance _instance, const char *pName)
{
   VK_FROM_HANDLE(wvk_instance, instance, _instance);
   return wvk_instance_get_proc_addr(instance ? &instance->vk : NULL,
                                     &wvk_instance_entrypoints, pName);
}

/* The loader's entry point into the driver, and -- in a static link, where
 * there is no loader -- the application's. */
PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return wvk_GetInstanceProcAddr(instance, pName);
}
