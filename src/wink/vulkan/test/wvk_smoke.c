/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * The wink Vulkan driver without shaders: what a plain Vulkan application
 * does before it draws, checked by reading the results back.
 *
 *   - buffer copies, fills and updates through device-local memory
 *   - a render pass that clears, and the image read back -- with a row
 *     pitch WebGPU would not take directly, too
 *   - vkCmdClearColorImage, and a render pass made with vkCreateRenderPass
 *   - one command buffer submitted twice
 *   - fences, timeline semaphores, and a submit waiting for a value the
 *     host signals later
 *   - a swapchain on a headless surface: acquire, render, present, and a
 *     swapchain replacing it
 *
 * Build: cc -o wvk_smoke wvk_smoke.c $(pkg-config --cflags --libs vulkan)
 * Run:   VK_DRIVER_FILES=<build>/src/wink/vulkan/wink_devenv_icd.<arch>.json ./wvk_smoke
 */
#include <vulkan/vulkan.h>

#include <stdarg.h>
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
static PFN_vkCmdBeginRenderingKHR begin_rendering;
static PFN_vkCmdEndRenderingKHR end_rendering;

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
make_image(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage)
{
   struct image im = { 0 };
   VK(vkCreateImage(dev, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = format, .extent = { w, h, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   }, NULL, &im.image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, im.image, &req);
   VK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   }, NULL, &im.memory));
   VK(vkBindImageMemory(dev, im.image, im.memory, 0));
   VK(vkCreateImageView(dev, &(VkImageViewCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   }, NULL, &im.view));
   return im;
}

static void
free_image(struct image *im)
{
   vkDestroyImageView(dev, im->view, NULL);
   vkDestroyImage(dev, im->image, NULL);
   vkFreeMemory(dev, im->memory, NULL);
}

static VkCommandBuffer
begin(VkCommandBufferUsageFlags flags)
{
   VkCommandBuffer cmd;
   VK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
   }, &cmd));
   VK(vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = flags,
   }));
   return cmd;
}

static void
submit_and_wait(VkCommandBuffer cmd)
{
   VkFence fence;
   VK(vkCreateFence(dev, &(VkFenceCreateInfo){ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }, NULL, &fence));
   VK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd,
   }, fence));
   VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
   vkDestroyFence(dev, fence, NULL);
}

static void
test_buffers(void)
{
   struct buffer a = make_buffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   struct buffer b = make_buffer(256, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   struct buffer c = make_buffer(256, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   for (int i = 0; i < 256; ++i)
      ((uint8_t *)a.map)[i] = (uint8_t)i;

   VkCommandBuffer cmd = begin(0);
   vkCmdCopyBuffer(cmd, a.buffer, b.buffer, 1, &(VkBufferCopy){ 0, 0, 256 });
   vkCmdFillBuffer(cmd, b.buffer, 128, 64, 0xdeadbeef);
   const uint32_t update[4] = { 1, 2, 3, 4 };
   vkCmdUpdateBuffer(cmd, b.buffer, 16, sizeof(update), update);
   vkCmdCopyBuffer(cmd, b.buffer, c.buffer, 1, &(VkBufferCopy){ 0, 0, 256 });
   VK(vkEndCommandBuffer(cmd));
   submit_and_wait(cmd);

   const uint8_t *out = c.map;
   const uint32_t *words = c.map;
   CHECK(out[0] == 0 && out[15] == 15 && out[32] == 32 && out[127] == 127 && out[255] == 255,
         "buffer copy: %u %u %u %u %u", out[0], out[15], out[32], out[127], out[255]);
   CHECK(words[4] == 1 && words[7] == 4, "update buffer: %u %u", words[4], words[7]);
   CHECK(words[32] == 0xdeadbeef && words[47] == 0xdeadbeef && words[48] == 0xc3c2c1c0,
         "fill buffer: %08x %08x %08x", words[32], words[47], words[48]);

   /* The same command buffer again, after the source changed: a replay,
    * not the first submit's results. */
   for (int i = 0; i < 256; ++i)
      ((uint8_t *)a.map)[i] = (uint8_t)(255 - i);
   memset(c.map, 0, 256);
   submit_and_wait(cmd);
   CHECK(out[0] == 255 && out[255] == 0 && words[4] == 1 && words[32] == 0xdeadbeef,
         "resubmitted: %u %u %u %08x", out[0], out[255], words[4], words[32]);

   /* Disjoint ranges in a single allocation still share a WebGPU buffer.
    * Exercise both one VkBuffer and a second VkBuffer bound to that memory. */
   VkBuffer alias;
   VK(vkCreateBuffer(dev, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 256,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   }, NULL, &alias));
   VK(vkBindBufferMemory(dev, alias, b.memory, 0));
   VkCommandBuffer same = begin(0);
   vkCmdCopyBuffer(same, a.buffer, b.buffer, 1, &(VkBufferCopy){ 0, 0, 256 });
   vkCmdCopyBuffer(same, b.buffer, b.buffer, 1, &(VkBufferCopy){ 0, 64, 32 });
   vkCmdCopyBuffer(same, b.buffer, alias, 1, &(VkBufferCopy){ 32, 128, 32 });
   vkCmdCopyBuffer(same, b.buffer, alias, 1, &(VkBufferCopy){ 2, 193, 31 });
   vkCmdCopyBuffer(same, b.buffer, c.buffer, 1, &(VkBufferCopy){ 0, 0, 256 });
   VK(vkEndCommandBuffer(same));
   submit_and_wait(same);
   CHECK(memcmp(out + 64, a.map, 32) == 0 && memcmp(out + 128, (uint8_t *)a.map + 32, 32) == 0,
         "same-allocation copies changed bytes");
   CHECK(memcmp(out + 193, (uint8_t *)a.map + 2, 31) == 0 && out[192] == 63 && out[224] == 31,
         "unaligned same-allocation copy changed bytes outside its range");
   vkDestroyBuffer(dev, alias, NULL);

   free_buffer(&a);
   free_buffer(&b);
   free_buffer(&c);
   printf("buffers: done\n");
}

static bool
near(uint8_t a, uint8_t b)
{
   return abs((int)a - (int)b) <= 1;
}

static void
test_clears(void)
{
   const uint32_t w = 64, h = 48;
   struct image im = make_image(w, h, VK_FORMAT_R8G8B8A8_UNORM,
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   /* Rows of 256 bytes, which WebGPU copies as they are; and of 65
    * texels, which it will not. */
   struct buffer aligned = make_buffer(w * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   struct buffer pitched = make_buffer(65 * h * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

   VkCommandBuffer cmd = begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
   begin_rendering(cmd, &(VkRenderingInfo){
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { { 0, 0 }, { w, h } }, .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &(VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = im.view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue.color.float32 = { 0.25f, 0.5f, 0.75f, 1.0f },
      },
   });
   end_rendering(cmd);
   const VkImageSubresourceLayers sub = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
   vkCmdCopyImageToBuffer(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, aligned.buffer, 1,
                          &(VkBufferImageCopy){ .imageSubresource = sub, .imageExtent = { w, h, 1 } });
   vkCmdCopyImageToBuffer(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pitched.buffer, 1,
                          &(VkBufferImageCopy){ .bufferRowLength = 65, .imageSubresource = sub,
                                                .imageExtent = { w, h, 1 } });
   VK(vkEndCommandBuffer(cmd));
   submit_and_wait(cmd);

   const uint8_t *p = aligned.map;
   CHECK(near(p[0], 64) && near(p[1], 128) && near(p[2], 191) && p[3] == 255,
         "rendering clear: %u %u %u %u", p[0], p[1], p[2], p[3]);
   p = (const uint8_t *)pitched.map + (65 * 47 + 63) * 4;
   CHECK(near(p[0], 64) && near(p[1], 128) && near(p[2], 191) && p[3] == 255,
         "pitched readback: %u %u %u %u", p[0], p[1], p[2], p[3]);

   /* vkCmdClearColorImage, then a render pass from vkCreateRenderPass that
    * loads it and clears nothing. */
   VkRenderPass rp;
   VK(vkCreateRenderPass(dev, &(VkRenderPassCreateInfo){
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
      .pAttachments = &(VkAttachmentDescription){
         .format = VK_FORMAT_R8G8B8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
      .subpassCount = 1,
      .pSubpasses = &(VkSubpassDescription){
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
         .pColorAttachments = &(VkAttachmentReference){ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
      },
   }, NULL, &rp));
   VkFramebuffer fb;
   VK(vkCreateFramebuffer(dev, &(VkFramebufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp, .attachmentCount = 1,
      .pAttachments = &im.view, .width = w, .height = h, .layers = 1,
   }, NULL, &fb));

   cmd = begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
   vkCmdClearColorImage(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        &(VkClearColorValue){ .float32 = { 1.0f, 0.0f, 0.0f, 1.0f } }, 1,
                        &(VkImageSubresourceRange){ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 });
   vkCmdBeginRenderPass(cmd, &(VkRenderPassBeginInfo){
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp, .framebuffer = fb,
      .renderArea = { { 0, 0 }, { w, h } },
   }, VK_SUBPASS_CONTENTS_INLINE);
   vkCmdEndRenderPass(cmd);
   vkCmdCopyImageToBuffer(cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, aligned.buffer, 1,
                          &(VkBufferImageCopy){ .imageSubresource = sub, .imageExtent = { w, h, 1 } });
   VK(vkEndCommandBuffer(cmd));
   submit_and_wait(cmd);
   p = aligned.map;
   CHECK(p[0] == 255 && p[1] == 0 && p[2] == 0 && p[3] == 255, "clear color image: %u %u %u %u",
         p[0], p[1], p[2], p[3]);

   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   free_buffer(&aligned);
   free_buffer(&pitched);
   free_image(&im);
   printf("clears: done\n");
}

static void
test_sync(void)
{
   VkSemaphore timeline;
   VK(vkCreateSemaphore(dev, &(VkSemaphoreCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &(VkSemaphoreTypeCreateInfo){
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
         .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0,
      },
   }, NULL, &timeline));

   struct buffer c = make_buffer(16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkCommandBuffer cmd = begin(0);
   vkCmdFillBuffer(cmd, c.buffer, 0, 16, 0x12345678);
   VK(vkEndCommandBuffer(cmd));

   /* Signals 1 on the queue. */
   const uint64_t one = 1;
   VK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &(VkTimelineSemaphoreSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
         .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &one,
      },
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 1, .pSignalSemaphores = &timeline,
   }, VK_NULL_HANDLE));
   VK(vkWaitSemaphores(dev, &(VkSemaphoreWaitInfo){
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
      .pSemaphores = &timeline, .pValues = &one,
   }, UINT64_MAX));
   CHECK(((uint32_t *)c.map)[0] == 0x12345678, "timeline wait: %08x", ((uint32_t *)c.map)[0]);

   /* Waits for 2, which the host signals afterwards, and signals 3. */
   memset(c.map, 0, 16);
   const uint64_t two = 2, three = 3;
   const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   VK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .pNext = &(VkTimelineSemaphoreSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
         .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &two,
         .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &three,
      },
      .waitSemaphoreCount = 1, .pWaitSemaphores = &timeline, .pWaitDstStageMask = &stage,
      .commandBufferCount = 1, .pCommandBuffers = &cmd,
      .signalSemaphoreCount = 1, .pSignalSemaphores = &timeline,
   }, VK_NULL_HANDLE));
   uint64_t value = 0;
   VK(vkGetSemaphoreCounterValue(dev, timeline, &value));
   CHECK(value == 1, "timeline ran ahead of its wait: %llu", (unsigned long long)value);
   VK(vkSignalSemaphore(dev, &(VkSemaphoreSignalInfo){
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = timeline, .value = 2,
   }));
   VK(vkWaitSemaphores(dev, &(VkSemaphoreWaitInfo){
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
      .pSemaphores = &timeline, .pValues = &three,
   }, UINT64_MAX));
   CHECK(((uint32_t *)c.map)[3] == 0x12345678, "wait-before-signal: %08x", ((uint32_t *)c.map)[3]);

   VK(vkQueueWaitIdle(queue));
   VK(vkDeviceWaitIdle(dev));
   free_buffer(&c);
   vkDestroySemaphore(dev, timeline, NULL);
   printf("sync: done\n");
}

static void
test_timestamps(void)
{
   VkQueueFamilyProperties properties;
   uint32_t count = 1;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &count, &properties);
   if (!properties.timestampValidBits) {
      printf("timestamps: unavailable\n");
      return;
   }
   VkQueryPool queries;
   VK(vkCreateQueryPool(dev, &(VkQueryPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2,
   }, NULL, &queries));
   struct buffer b = make_buffer(4096, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   VkCommandBuffer cmd = begin(0);
   vkCmdResetQueryPool(cmd, queries, 0, 2);
   vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
   vkCmdFillBuffer(cmd, b.buffer, 0, VK_WHOLE_SIZE, 0x12345678);
   vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 1);
   VK(vkEndCommandBuffer(cmd));
   submit_and_wait(cmd);
   struct { uint64_t value, available; } results[2] = {{0}};
   VK(vkGetQueryPoolResults(dev, queries, 0, 2, sizeof(results), results,
                           sizeof(results[0]), VK_QUERY_RESULT_64_BIT |
                           VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
   CHECK(results[0].available && results[1].available, "timestamps not available");
   CHECK(results[0].value > 0 && results[1].value >= results[0].value,
         "timestamps not ordered: %llu, %llu",
         (unsigned long long)results[0].value, (unsigned long long)results[1].value);
   free_buffer(&b);
   vkDestroyQueryPool(dev, queries, NULL);
   printf("timestamps: done\n");
}

static void
test_present(void)
{
   PFN_vkCreateHeadlessSurfaceEXT create_headless =
      (PFN_vkCreateHeadlessSurfaceEXT)vkGetInstanceProcAddr(instance, "vkCreateHeadlessSurfaceEXT");
   VkSurfaceKHR surface;
   VK(create_headless(instance, &(VkHeadlessSurfaceCreateInfoEXT){
      .sType = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT,
   }, NULL, &surface));
   VkBool32 supported = VK_FALSE;
   VK(vkGetPhysicalDeviceSurfaceSupportKHR(pdev, 0, surface, &supported));
   CHECK(supported, "the queue cannot present");
   VkSurfaceCapabilitiesKHR caps;
   VK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pdev, surface, &caps));
   CHECK(caps.currentExtent.width == UINT32_MAX && caps.minImageCount <= 3 && caps.maxImageCount >= 3,
         "headless caps: %u, %u..%u", caps.currentExtent.width, caps.minImageCount, caps.maxImageCount);
   VkSurfaceFormatKHR formats[16];
   uint32_t format_count = 16;
   VK(vkGetPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, formats));
   bool have_bgra = false;
   for (uint32_t i = 0; i < format_count; ++i)
      have_bgra |= formats[i].format == VK_FORMAT_B8G8R8A8_UNORM;
   CHECK(have_bgra, "no B8G8R8A8_UNORM surface format");

   enum { W = 32, H = 16, FRAMES = 5 };
   VkSwapchainCreateInfoKHR info = {
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = surface,
      .minImageCount = 3, .imageFormat = VK_FORMAT_B8G8R8A8_UNORM,
      .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR, .imageExtent = { W, H },
      .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE,
   };
   VkSwapchainKHR sc;
   VK(vkCreateSwapchainKHR(dev, &info, NULL, &sc));
   uint32_t image_count = 0;
   VK(vkGetSwapchainImagesKHR(dev, sc, &image_count, NULL));
   VkImage images[16];
   image_count = image_count > 16 ? 16 : image_count;
   VK(vkGetSwapchainImagesKHR(dev, sc, &image_count, images));
   CHECK(image_count >= 3, "%u swapchain images", image_count);

   VkSemaphore acquired, rendered;
   VK(vkCreateSemaphore(dev, &(VkSemaphoreCreateInfo){ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }, NULL, &acquired));
   VK(vkCreateSemaphore(dev, &(VkSemaphoreCreateInfo){ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }, NULL, &rendered));
   VkFence fence;
   VK(vkCreateFence(dev, &(VkFenceCreateInfo){ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }, NULL, &fence));
   struct buffer out = make_buffer(W * H * 4 * FRAMES, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

   uint32_t last = UINT32_MAX;
   for (uint32_t frame = 0; frame < FRAMES; ++frame) {
      uint32_t index;
      VK(vkAcquireNextImageKHR(dev, sc, UINT64_MAX, acquired, VK_NULL_HANDLE, &index));
      CHECK(index < image_count && index != last, "acquired image %u after %u", index, last);
      last = index;

      VkCommandBuffer cmd = begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
      const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                           0, NULL, 0, NULL, 1, &(VkImageMemoryBarrier){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .image = images[index], .subresourceRange = range,
      });
      const VkClearColorValue color = { .float32 = { frame / 4.0f, 0.5f, 1.0f - frame / 4.0f, 1.0f } };
      vkCmdClearColorImage(cmd, images[index], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);
      vkCmdCopyImageToBuffer(cmd, images[index], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1,
                             &(VkBufferImageCopy){
         .bufferOffset = frame * W * H * 4,
         .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { W, H, 1 },
      });
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
         .waitSemaphoreCount = 1, .pWaitSemaphores = &acquired, .pWaitDstStageMask = &stage,
         .commandBufferCount = 1, .pCommandBuffers = &cmd,
         .signalSemaphoreCount = 1, .pSignalSemaphores = &rendered,
      }, fence));
      VkResult result = VK_ERROR_UNKNOWN;
      VK(vkQueuePresentKHR(queue, &(VkPresentInfoKHR){
         .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
         .waitSemaphoreCount = 1, .pWaitSemaphores = &rendered,
         .swapchainCount = 1, .pSwapchains = &sc, .pImageIndices = &index, .pResults = &result,
      }));
      CHECK(result == VK_SUCCESS, "present result %d", result);
      VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
      VK(vkResetFences(dev, 1, &fence));
      vkFreeCommandBuffers(dev, pool, 1, &cmd);
   }
   for (uint32_t frame = 0; frame < FRAMES; ++frame) {
      const uint8_t *p = (const uint8_t *)out.map + frame * W * H * 4 + (W * (H - 1) + W - 1) * 4;
      const float r = frame / 4.0f;
      /* B8G8R8A8: blue first. */
      CHECK(near(p[0], (1.0f - r) * 255) && near(p[1], 128) && near(p[2], r * 255) && p[3] == 255,
            "frame %u: %u %u %u %u", frame, p[0], p[1], p[2], p[3]);
   }

   /* A new swapchain takes the surface over; the old one is out of date. */
   info.oldSwapchain = sc;
   info.imageExtent = (VkExtent2D){ W * 2, H * 2 };
   VkSwapchainKHR sc2;
   VK(vkCreateSwapchainKHR(dev, &info, NULL, &sc2));
   uint32_t index;
   VkResult result = vkAcquireNextImageKHR(dev, sc, 0, VK_NULL_HANDLE, fence, &index);
   CHECK(result == VK_ERROR_OUT_OF_DATE_KHR, "acquiring from a replaced swapchain: %d", result);
   vkDestroySwapchainKHR(dev, sc, NULL);
   VK(vkAcquireNextImageKHR(dev, sc2, 0, VK_NULL_HANDLE, fence, &index));
   VK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 0));
   VK(vkQueuePresentKHR(queue, &(VkPresentInfoKHR){
      .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
      .swapchainCount = 1, .pSwapchains = &sc2, .pImageIndices = &index,
   }));

   VK(vkDeviceWaitIdle(dev));
   vkDestroySwapchainKHR(dev, sc2, NULL);
   free_buffer(&out);
   vkDestroyFence(dev, fence, NULL);
   vkDestroySemaphore(dev, acquired, NULL);
   vkDestroySemaphore(dev, rendered, NULL);
   vkDestroySurfaceKHR(instance, surface, NULL);
   printf("present: done\n");
}

int
main(void)
{
   const char *instance_extensions[] = { "VK_KHR_surface", "VK_EXT_headless_surface" };
   VK(vkCreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_2,
      },
      .enabledExtensionCount = 2, .ppEnabledExtensionNames = instance_extensions,
   }, NULL, &instance));
   uint32_t count = 0;
   VK(vkEnumeratePhysicalDevices(instance, &count, NULL));
   VkPhysicalDevice pdevs[8];
   count = count > 8 ? 8 : count;
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
      return 1;
   }
   vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);

   const char *extensions[] = { "VK_KHR_portability_subset", "VK_KHR_dynamic_rendering",
                                "VK_KHR_swapchain" };
   VK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &(VkPhysicalDeviceDynamicRenderingFeatures){
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
         .pNext = &(VkPhysicalDeviceVulkan12Features){
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, .timelineSemaphore = VK_TRUE,
         },
         .dynamicRendering = VK_TRUE,
      },
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = 0, .queueCount = 1,
         .pQueuePriorities = &(float){ 1.0f },
      },
      .enabledExtensionCount = 3, .ppEnabledExtensionNames = extensions,
   }, NULL, &dev));
   begin_rendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(dev, "vkCmdBeginRenderingKHR");
   end_rendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(dev, "vkCmdEndRenderingKHR");
   if (!begin_rendering || !end_rendering) {
      printf("FAIL no vkCmdBeginRenderingKHR\n");
      return 1;
   }
   vkGetDeviceQueue(dev, 0, 0, &queue);
   VK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   }, NULL, &pool));

   test_buffers();
   test_clears();
   test_sync();
   test_timestamps();
   test_present();

   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf(failures ? "FAILED %d\n" : "PASS\n", failures);
   return failures ? 1 : 0;
}
