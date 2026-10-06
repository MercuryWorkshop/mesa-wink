/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * What the wink Vulkan driver's tests share: a device on the wink
 * physical device, buffers and images, and one-shot submits.
 */
#ifndef WVK_TEST_H
#define WVK_TEST_H

#include <vulkan/vulkan.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(cond, ...) do {                                               \
      if (!(cond)) { ++failures; printf("FAIL " __VA_ARGS__); printf("\n"); } \
   } while (0)
#define VK(call) do {                                                       \
      VkResult r_ = (call);                                                 \
      if (r_ != VK_SUCCESS) { printf("FAIL %s = %d\n", #call, r_); exit(1); } \
   } while (0)

static VkInstance instance;
static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties mem_props;

static void
test_init(uint32_t extension_count, const char *const *extensions, const void *features)
{
   VK(vkCreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2,
      },
   }, NULL, &instance));
   uint32_t count = 8;
   VkPhysicalDevice pdevs[8];
   VK(vkEnumeratePhysicalDevices(instance, &count, pdevs));
   for (uint32_t i = 0; i < count && !pdev; ++i) {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(pdevs[i], &props);
      if (strncmp(props.deviceName, "wink", 4) == 0) {
         pdev = pdevs[i];
         printf("%s\n", props.deviceName);
      }
   }
   if (!pdev) {
      printf("FAIL no wink device\n");
      exit(1);
   }
   vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
   VK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
         .pQueuePriorities = &(float){ 1.0f },
      },
      .enabledExtensionCount = extension_count, .ppEnabledExtensionNames = extensions,
   }, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   }, NULL, &pool));
}

static int
test_finish(void)
{
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf(failures ? "FAILED %d\n" : "PASS\n", failures);
   return failures ? 1 : 0;
}

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & want) == want)
         return i;
   }
   printf("FAIL no memory type 0x%x in 0x%x\n", want, bits);
   exit(1);
}

struct buffer {
   VkBuffer buffer;
   VkDeviceMemory memory;
   void *map;
};

#define HOST (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)

static struct buffer
make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props)
{
   struct buffer b = { 0 };
   VK(vkCreateBuffer(dev, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
   }, NULL, &b.buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, b.buffer, &req);
   VK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, props),
   }, NULL, &b.memory));
   VK(vkBindBufferMemory(dev, b.buffer, b.memory, 0));
   if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      VK(vkMapMemory(dev, b.memory, 0, VK_WHOLE_SIZE, 0, &b.map));
   return b;
}

static void
free_buffer(struct buffer *b)
{
   vkDestroyBuffer(dev, b->buffer, NULL);
   vkFreeMemory(dev, b->memory, NULL);
}

struct image {
   VkImage image;
   VkDeviceMemory memory;
   VkImageView view;
};

static struct image
make_image_samples(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage, VkSampleCountFlagBits samples)
{
   struct image im = { 0 };
   VK(vkCreateImage(dev, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = format, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   }, NULL, &im.image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, im.image, &req);
   VK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   }, NULL, &im.memory));
   VK(vkBindImageMemory(dev, im.image, im.memory, 0));
   const bool depth = format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM ||
                      format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
   VK(vkCreateImageView(dev, &(VkImageViewCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = { depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   }, NULL, &im.view));
   return im;
}

static struct image
make_image(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage)
{
   return make_image_samples(w, h, format, usage, VK_SAMPLE_COUNT_1_BIT);
}

static void
free_image(struct image *im)
{
   vkDestroyImageView(dev, im->view, NULL);
   vkDestroyImage(dev, im->image, NULL);
   vkFreeMemory(dev, im->memory, NULL);
}

static VkCommandBuffer
begin(void)
{
   VkCommandBuffer cmd;
   VK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   }, &cmd));
   VK(vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   }));
   return cmd;
}

static void
submit_and_wait(VkCommandBuffer cmd)
{
   VK(vkEndCommandBuffer(cmd));
   VkFence fence;
   VK(vkCreateFence(dev, &(VkFenceCreateInfo){ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }, NULL, &fence));
   VK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   }, fence));
   VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
   vkDestroyFence(dev, fence, NULL);
   vkFreeCommandBuffers(dev, pool, 1, &cmd);
}

static VkShaderModule
make_module(const uint32_t *code, size_t size)
{
   VkShaderModule module;
   VK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code,
   }, NULL, &module));
   return module;
}

static bool
near(int a, int b)
{
   return abs(a - b) <= 3;
}

#endif
