// SPDX-License-Identifier: MIT
// Browser WSI aliases preserve Vulkan replay semantics while mapping the
// captured platform/headless surface to Wink's canvas surface.
#include "wink.h"
#include <vulkan/vulkan.h>
#include <cstring>
#include <vector>

static VkResult create_instance(const VkInstanceCreateInfo *original,
                                const VkAllocationCallbacks *allocator, VkInstance *instance) {
    VkInstanceCreateInfo info = *original;
    std::vector<const char*> extensions;
    for (uint32_t i = 0; i < info.enabledExtensionCount; ++i) {
        const char *name = info.ppEnabledExtensionNames[i];
        if (!std::strcmp(name, "VK_EXT_headless_surface") ||
            !std::strcmp(name, "VK_KHR_xlib_surface") ||
            !std::strcmp(name, "VK_KHR_xcb_surface") ||
            !std::strcmp(name, "VK_KHR_wayland_surface")) continue;
        extensions.push_back(name);
    }
    info.enabledExtensionCount = extensions.size();
    info.ppEnabledExtensionNames = extensions.data();
    auto call = reinterpret_cast<PFN_vkCreateInstance>(wink_vk_get_instance_proc_addr(nullptr, "vkCreateInstance"));
    return call(&info, allocator, instance);
}

static VkResult create_headless(VkInstance instance, const void*,
                                const VkAllocationCallbacks*, VkSurfaceKHR *surface) {
    return static_cast<VkResult>(wink_vk_create_surface(instance, "#canvas", reinterpret_cast<uint64_t*>(surface)));
}

extern "C" void *wink_replay_vk_get_instance_proc_addr(void *instance, const char *name) {
    if (!std::strcmp(name, "vkCreateInstance")) return reinterpret_cast<void*>(create_instance);
    if (!std::strcmp(name, "vkCreateHeadlessSurfaceEXT") ||
        !std::strcmp(name, "vkCreateXlibSurfaceKHR") ||
        !std::strcmp(name, "vkCreateXcbSurfaceKHR") ||
        !std::strcmp(name, "vkCreateWaylandSurfaceKHR")) return reinterpret_cast<void*>(create_headless);
    return reinterpret_cast<void*>(wink_vk_get_instance_proc_addr(instance, name));
}
