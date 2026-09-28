/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * The wink Vulkan driver presenting to a window: an SDL3 window's surface,
 * cleared to a colour that changes every frame, in FIFO. It reports the
 * frame time, which the surface paces, and remakes the swapchain when the
 * window changes size. With --screenshot FILE it holds one colour for the
 * last frames and captures the window (macOS screencapture), to check
 * what reached the screen.
 *
 * Build: cc -o wvk_window wvk_window.c $(pkg-config --cflags --libs sdl3 vulkan)
 * Run:   VK_DRIVER_FILES=<build>/src/wink/vulkan/wink_devenv_icd.<arch>.json \
 *          ./wvk_window [--frames N] [--screenshot FILE] [--resize] [--hidpi]
 */
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <objc/message.h>
#include <objc/runtime.h>
#endif

#define VK(call) do {                                                       \
      VkResult r_ = (call);                                                 \
      if (r_ != VK_SUCCESS) { printf("FAIL %s = %d\n", #call, r_); exit(1); } \
   } while (0)

static VkInstance instance;
static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkSurfaceKHR surface;
static VkSwapchainKHR swapchain;
static VkFormat format;
static VkExtent2D extent;
static VkImage images[16];
static uint32_t image_count;

static void
make_swapchain(SDL_Window *window)
{
   VkSurfaceCapabilitiesKHR caps;
   VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pdev, surface, &caps));
   if (caps.currentExtent.width != UINT32_MAX) {
      extent = caps.currentExtent;
   } else {
      int w, h;
      SDL_GetWindowSizeInPixels(window, &w, &h);
      extent = (VkExtent2D){ (uint32_t)w, (uint32_t)h };
   }
   VkSurfaceFormatKHR formats[64];
   uint32_t count = 64;
   const VkResult r = vkGetPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &count, formats);
   if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
      printf("FAIL vkGetPhysicalDeviceSurfaceFormatsKHR = %d\n", r);
      exit(1);
   }
   format = formats[0].format;
   for (uint32_t i = 0; i < count; ++i) {
      if (formats[i].format == VK_FORMAT_B8G8R8A8_UNORM)
         format = formats[i].format;
   }

   VkSwapchainKHR old = swapchain;
   VK(vkCreateSwapchainKHR(dev, &(VkSwapchainCreateInfoKHR){
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = surface,
      .minImageCount = caps.minImageCount + 1, .imageFormat = format,
      .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, .imageExtent = extent,
      .imageArrayLayers = 1, .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .preTransform = caps.currentTransform, .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE, .oldSwapchain = old,
   }, NULL, &swapchain));
   if (old) {
      VK(vkDeviceWaitIdle(dev));
      vkDestroySwapchainKHR(dev, old, NULL);
   }
   image_count = 16;
   VK(vkGetSwapchainImagesKHR(dev, swapchain, &image_count, images));
   int pw, ph;
   SDL_GetWindowSizeInPixels(window, &pw, &ph);
   printf("swapchain: %ux%u (window %dx%d pixels), format %d, %u images\n", extent.width, extent.height,
          pw, ph, format, image_count);
}

int
main(int argc, char **argv)
{
   int frames = 180;
   const char *screenshot = NULL;
   bool resize = false;
   SDL_WindowFlags flags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE;
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--frames") && i + 1 < argc)
         frames = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--screenshot") && i + 1 < argc)
         screenshot = argv[++i];
      else if (!strcmp(argv[i], "--resize"))
         resize = true;
      else if (!strcmp(argv[i], "--hidpi"))
         flags |= SDL_WINDOW_HIGH_PIXEL_DENSITY;
   }

   if (!SDL_Init(SDL_INIT_VIDEO)) {
      printf("FAIL SDL_Init: %s\n", SDL_GetError());
      return 1;
   }
   /* The loader, not MoltenVK: it is what VK_DRIVER_FILES talks to. */
   if (!SDL_Vulkan_LoadLibrary(getenv("WVK_VULKAN_LIBRARY") ? getenv("WVK_VULKAN_LIBRARY")
                                                             : "/opt/homebrew/lib/libvulkan.1.dylib")) {
      printf("FAIL SDL_Vulkan_LoadLibrary: %s\n", SDL_GetError());
      return 1;
   }
   SDL_Window *window = SDL_CreateWindow("wink Vulkan", 640, 360, flags);
   if (!window) {
      printf("FAIL SDL_CreateWindow: %s\n", SDL_GetError());
      return 1;
   }

   Uint32 sdl_count = 0;
   const char *const *sdl_exts = SDL_Vulkan_GetInstanceExtensions(&sdl_count);
   const char *exts[16];
   uint32_t ext_count = 0;
   for (Uint32 i = 0; i < sdl_count && ext_count < 15; ++i)
      exts[ext_count++] = sdl_exts[i];
   /* So that portability drivers -- MoltenVK, to compare with -- are seen. */
   VkInstanceCreateFlags instance_flags = 0;
   VkExtensionProperties available[64];
   uint32_t available_count = 64;
   vkEnumerateInstanceExtensionProperties(NULL, &available_count, available);
   for (uint32_t i = 0; i < available_count; ++i) {
      if (!strcmp(available[i].extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
         exts[ext_count++] = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
         instance_flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
      }
   }
   VK(vkCreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .flags = instance_flags,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2,
      },
      .enabledExtensionCount = ext_count, .ppEnabledExtensionNames = exts,
   }, NULL, &instance));
   if (!SDL_Vulkan_CreateSurface(window, instance, NULL, &surface)) {
      printf("FAIL SDL_Vulkan_CreateSurface: %s\n", SDL_GetError());
      return 1;
   }
   uint32_t count = 1;
   vkEnumeratePhysicalDevices(instance, &count, &pdev);
   if (!count) {
      printf("FAIL no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   printf("%s\n", props.deviceName);

   const char *dev_exts[] = { "VK_KHR_portability_subset", "VK_KHR_swapchain" };
   VK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
         .pQueuePriorities = &(float){ 1.0f },
      },
      .enabledExtensionCount = 2, .ppEnabledExtensionNames = dev_exts,
   }, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   make_swapchain(window);

   VkCommandPool pool;
   VK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   }, NULL, &pool));
   /* Two frames in flight. */
   VkCommandBuffer cmds[2];
   VK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 2,
   }, cmds));
   VkSemaphore acquired[2], rendered[2];
   VkFence fences[2];
   for (int i = 0; i < 2; ++i) {
      VK(vkCreateSemaphore(dev, &(VkSemaphoreCreateInfo){ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }, NULL, &acquired[i]));
      VK(vkCreateSemaphore(dev, &(VkSemaphoreCreateInfo){ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }, NULL, &rendered[i]));
      VK(vkCreateFence(dev, &(VkFenceCreateInfo){
         .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT,
      }, NULL, &fences[i]));
   }

   const Uint64 start = SDL_GetTicksNS();
   Uint64 last = start, worst = 0;
   int remade = 0;
   bool quit = false;
   for (int frame = 0; frame < frames && !quit; ++frame) {
      SDL_Event event;
      while (SDL_PollEvent(&event)) {
         if (event.type == SDL_EVENT_QUIT)
            quit = true;
      }
      if (resize && frame == frames / 2)
         SDL_SetWindowSize(window, 800, 450);

      const int f = frame % 2;
      VK(vkWaitForFences(dev, 1, &fences[f], VK_TRUE, UINT64_MAX));
      uint32_t index;
      VkResult result = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, acquired[f], VK_NULL_HANDLE, &index);
      if (result == VK_ERROR_OUT_OF_DATE_KHR) {
         make_swapchain(window);
         ++remade;
         continue;
      }
      if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
         printf("FAIL acquire = %d\n", result);
         return 1;
      }
      VK(vkResetFences(dev, 1, &fences[f]));

      /* Cycles through hues; the last frames hold one, for a screenshot. */
      const bool hold = screenshot && frame >= frames - 30;
      const float t = hold ? 0.0f : frame / 60.0f;
      const VkClearColorValue color = { .float32 = {
         hold ? 1.0f : 0.5f + 0.5f * SDL_sinf(t),
         hold ? 0.5f : 0.5f + 0.5f * SDL_sinf(t + 2.1f),
         hold ? 0.0f : 0.5f + 0.5f * SDL_sinf(t + 4.2f), 1.0f,
      } };
      VkCommandBuffer cmd = cmds[f];
      VK(vkResetCommandBuffer(cmd, 0));
      VK(vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
         .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
      }));
      const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                           0, NULL, 0, NULL, 1, &(VkImageMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .image = images[index], .subresourceRange = range,
      });
      vkCmdClearColorImage(cmd, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                           0, NULL, 0, NULL, 1, &(VkImageMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
         .image = images[index], .subresourceRange = range,
      });
      VK(vkEndCommandBuffer(cmd));
      const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
      VK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .waitSemaphoreCount = 1, .pWaitSemaphores = &acquired[f], .pWaitDstStageMask = &stage,
         .commandBufferCount = 1, .pCommandBuffers = &cmd,
         .signalSemaphoreCount = 1, .pSignalSemaphores = &rendered[f],
      }, fences[f]));
      result = vkQueuePresentKHR(queue, &(VkPresentInfoKHR){
         .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
         .waitSemaphoreCount = 1, .pWaitSemaphores = &rendered[f],
         .swapchainCount = 1, .pSwapchains = &swapchain, .pImageIndices = &index,
      });
      if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
         make_swapchain(window);
         ++remade;
      } else if (result != VK_SUCCESS) {
         printf("FAIL present = %d\n", result);
         return 1;
      }

      const Uint64 now = SDL_GetTicksNS();
      if (frame > 2 && now - last > worst)
         worst = now - last;
      last = now;
   }
   VK(vkDeviceWaitIdle(dev));
   const double seconds = (SDL_GetTicksNS() - start) / 1e9;
   printf("%d frames in %.2f s: %.1f fps, worst frame %.1f ms, %d swapchains remade\n", frames, seconds,
          frames / seconds, worst / 1e6, remade);

   if (screenshot) {
      /* Presenting has stopped on the held colour; capture what shows. */
      long number = 0;
#ifdef __APPLE__
      id nswindow = (id)SDL_GetPointerProperty(SDL_GetWindowProperties(window),
                                               SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, NULL);
      if (nswindow)
         number = ((long (*)(id, SEL))objc_msgSend)(nswindow, sel_registerName("windowNumber"));
#endif
      char command[512];
      snprintf(command, sizeof(command), "screencapture -x -o -l%ld '%s'", number, screenshot);
      printf("%s: %d\n", command, system(command));
   }

   for (int i = 0; i < 2; ++i) {
      vkDestroySemaphore(dev, acquired[i], NULL);
      vkDestroySemaphore(dev, rendered[i], NULL);
      vkDestroyFence(dev, fences[i], NULL);
   }
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroySwapchainKHR(dev, swapchain, NULL);
   vkDestroyDevice(dev, NULL);
   SDL_Vulkan_DestroySurface(instance, surface, NULL);
   vkDestroyInstance(instance, NULL);
   SDL_DestroyWindow(window);
   SDL_Quit();
   printf("PASS\n");
   return 0;
}
