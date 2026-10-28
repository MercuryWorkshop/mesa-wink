/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * The wink Vulkan driver drawing: shaders, pipelines and descriptors,
 * checked by reading the image back.
 *
 *   - a quad from a vertex buffer, a colour at each corner: which corner
 *     lands where says whether y is right
 *   - push constants that change between two draws of one pass
 *   - a negative viewport height (VK_KHR_maintenance1) flipping it back
 *   - a textured quad: a uniform buffer and a combined image sampler
 *   - vkCmdClearAttachments inside a pass, and vkCmdBlitImage scaling up
 *   - BaseVertex, BaseInstance and DrawIndex, in direct and indirect
 *     draws, and an attribute with a divisor of 2
 *
 * Build: src/wink/vulkan/test/build.sh OUT
 */
#include "wvk_test.h"

#include "quad_vert.h"
#include "dead_input_vert.h"
#include "color_frag.h"
#include "solid_frag.h"
#include "discard_frag.h"
#include "texture_frag.h"
#include "instanced_vert.h"
#include "norm16_sample_comp.h"
#include "ms_input_frag.h"
#include <math.h>

enum { W = 64, H = 64 };

struct push {
   float tint[4];
   float offset[2];
   float pad[2];
};

static const float quad[] = {
   /* pos       colour */
   -1, -1,      1, 0, 0,   /* top left: red */
    1, -1,      0, 1, 0,   /* top right: green */
   -1,  1,      0, 0, 1,   /* bottom left: blue */
    1,  1,      1, 1, 1,   /* bottom right: white */
};

/* The pipeline for the quad, or with `divisor` for instanced.vert: one
 * instance-rate attribute of that divisor, and strips of four indices. */
static VkPipeline
make_pipeline_samples_state(VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs, uint32_t divisor,
                            VkSampleCountFlagBits samples, VkSampleMask mask, VkFormat stencil_format,
                            const VkPipelineColorBlendAttachmentState *blend)
{
   const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkPipelineShaderStageCreateInfo stages[] = {
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main" },
      { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main" },
   };
   const VkVertexInputAttributeDescription attrs[] = {
      { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = 0 },
      { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 8 },
   };
   const VkVertexInputAttributeDescription instance_attr = {
      .location = 0, .binding = 0, .format = VK_FORMAT_R8G8B8A8_UNORM, .offset = 0,
   };
   const VkVertexInputBindingDivisorDescriptionKHR binding_divisor = { .binding = 0, .divisor = divisor };
   const VkDynamicState dynamic[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                      VK_DYNAMIC_STATE_BLEND_CONSTANTS };
   VkPipeline pipeline;
   VK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &(VkPipelineRenderingCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
         .colorAttachmentCount = 1, .pColorAttachmentFormats = &format,
         .stencilAttachmentFormat = stencil_format,
      },
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = divisor ? &(VkPipelineVertexInputStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
         .pNext = &(VkPipelineVertexInputDivisorStateCreateInfoKHR){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_KHR,
            .vertexBindingDivisorCount = 1, .pVertexBindingDivisors = &binding_divisor,
         },
         .vertexBindingDescriptionCount = 1,
         .pVertexBindingDescriptions = &(VkVertexInputBindingDescription){
            .binding = 0, .stride = 4, .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE,
         },
         .vertexAttributeDescriptionCount = 1, .pVertexAttributeDescriptions = &instance_attr,
      } : &(VkPipelineVertexInputStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
         .vertexBindingDescriptionCount = 1,
         .pVertexBindingDescriptions = &(VkVertexInputBindingDescription){
            .binding = 0, .stride = 5 * sizeof(float), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
         },
         .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attrs,
      },
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
      },
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
         .viewportCount = 1, .scissorCount = 1,
      },
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
         .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
      },
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = samples, .pSampleMask = &mask,
      },
      .pDepthStencilState = stencil_format != VK_FORMAT_UNDEFINED ? &(VkPipelineDepthStencilStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
         .stencilTestEnable = VK_TRUE,
         .front = { .compareOp = VK_COMPARE_OP_EQUAL, .compareMask = 0xff, .reference = 7 },
         .back = { .compareOp = VK_COMPARE_OP_EQUAL, .compareMask = 0xff, .reference = 7 },
      } : NULL,
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
         .pNext = &(VkPipelineColorBlendAdvancedStateCreateInfoEXT){
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_ADVANCED_STATE_CREATE_INFO_EXT,
            .srcPremultiplied = VK_TRUE, .dstPremultiplied = VK_TRUE,
            .blendOverlap = VK_BLEND_OVERLAP_UNCORRELATED_EXT,
         },
         .attachmentCount = 1,
         .pAttachments = blend ? blend : &(VkPipelineColorBlendAttachmentState){
            .colorWriteMask = 0xf,
         },
      },
      .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
         .dynamicStateCount = 3, .pDynamicStates = dynamic,
      },
      .layout = layout,
   }, NULL, &pipeline));
   return pipeline;
}

static VkPipeline
make_pipeline_samples(VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs, uint32_t divisor,
                      VkSampleCountFlagBits samples, VkSampleMask mask)
{
   return make_pipeline_samples_state(layout, vs, fs, divisor, samples, mask, VK_FORMAT_UNDEFINED, NULL);
}

static VkPipeline
make_pipeline_with(VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs, uint32_t divisor)
{
   return make_pipeline_samples(layout, vs, fs, divisor, VK_SAMPLE_COUNT_1_BIT, ~0u);
}

static VkPipeline
make_pipeline(VkPipelineLayout layout, VkShaderModule vs, VkShaderModule fs)
{
   return make_pipeline_with(layout, vs, fs, 0);
}

static PFN_vkCmdBeginRenderingKHR begin_rendering;
static PFN_vkCmdEndRenderingKHR end_rendering;

static void
begin_pass(VkCommandBuffer cmd, struct image *target, bool negative_viewport)
{
   const VkImageSubresourceRange range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1,
                        &(VkImageMemoryBarrier){
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .image = target->image, .subresourceRange = range,
   });
   begin_rendering(cmd, &(VkRenderingInfo){
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &(VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = target->view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      },
   });
   const VkViewport viewport = negative_viewport ? (VkViewport){ 0, H, W, -(float)H, 0, 1 }
                                                 : (VkViewport){ 0, 0, W, H, 0, 1 };
   vkCmdSetViewport(cmd, 0, 1, &viewport);
   vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){ { 0, 0 }, { W, H } });
}

static void
end_pass_and_read(VkCommandBuffer cmd, struct image *target, struct buffer *out)
{
   end_rendering(cmd);
   vkCmdCopyImageToBuffer(cmd, target->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out->buffer, 1,
                          &(VkBufferImageCopy){
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { W, H, 1 },
   });
}

static const uint8_t *
pixel(const struct buffer *b, int x, int y)
{
   return (const uint8_t *)b->map + (y * W + x) * 4;
}

/* The colour that dominates a pixel: which corner of the quad it is. */
static bool
is(const uint8_t *p, int r, int g, int b)
{
   return near(p[0], r) && near(p[1], g) && near(p[2], b) && p[3] == 255;
}

static bool
mostly(const uint8_t *p, bool r, bool g, bool b)
{
   return (p[0] > 200) == r && (p[1] > 200) == g && (p[2] > 200) == b && p[3] == 255;
}

static void
test_colors(VkBuffer vertices, const uint32_t *vertex_code, size_t vertex_bytes)
{
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct push),
      },
   }, NULL, &layout));
   VkShaderModule vs = make_module(vertex_code, vertex_bytes);
   VkShaderModule fs = make_module(spv_color_frag, sizeof(spv_color_frag));
   VkPipeline pipeline = make_pipeline(layout, vs, fs);

   struct image target = make_image(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   struct buffer out = make_buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
   const VkDeviceSize zero = 0;

   /* The whole quad, then a red-only copy of it moved right by half. */
   VkCommandBuffer cmd = begin();
   begin_pass(cmd, &target, false);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &zero);
   struct push push = { { 1, 1, 1, 1 }, { 0, 0 } };
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(push), &push);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   push = (struct push){ { 1, 0, 0, 1 }, { 1, 0 } };
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(push), &push);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   end_pass_and_read(cmd, &target, &out);
   submit_and_wait(cmd);

   const uint8_t *tl = pixel(&out, 0, 0), *bl = pixel(&out, 0, H - 1);
   const uint8_t *tr = pixel(&out, W - 1, 0), *br = pixel(&out, W - 1, H - 1);
   CHECK(mostly(tl, true, false, false), "top left is not red: %u %u %u %u", tl[0], tl[1], tl[2], tl[3]);
   CHECK(mostly(bl, false, false, true), "bottom left is not blue: %u %u %u %u", bl[0], bl[1], bl[2], bl[3]);
   /* The right half is the second draw's: red only. */
   CHECK(tr[1] == 0 && tr[2] == 0 && tr[3] == 255, "top right is not red only: %u %u %u %u",
         tr[0], tr[1], tr[2], tr[3]);
   CHECK(br[1] == 0 && br[2] == 0 && br[3] == 255, "bottom right is not red only: %u %u %u %u",
         br[0], br[1], br[2], br[3]);
   const uint8_t *mid_left = pixel(&out, W / 4, H / 2);
   CHECK(mid_left[1] || mid_left[2], "the first draw's left half is gone: %u %u %u", mid_left[0],
         mid_left[1], mid_left[2]);

   /* Flipped by the viewport: blue at the top. */
   cmd = begin();
   begin_pass(cmd, &target, true);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &zero);
   push = (struct push){ { 1, 1, 1, 1 }, { 0, 0 } };
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(push), &push);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   end_pass_and_read(cmd, &target, &out);
   submit_and_wait(cmd);
   tl = pixel(&out, 0, 0);
   bl = pixel(&out, 0, H - 1);
   CHECK(mostly(tl, false, false, true), "flipped: top left is not blue: %u %u %u %u", tl[0], tl[1], tl[2], tl[3]);
   CHECK(mostly(bl, true, false, false), "flipped: bottom left is not red: %u %u %u %u", bl[0], bl[1], bl[2], bl[3]);

   free_buffer(&out);
   free_image(&target);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   printf("colors: done\n");
}

/* Shader-emulated blend equations must preserve each destination sample,
 * including a second draw reusing the copy and a partial sample mask. */
static void
test_shader_blend(VkBuffer vertices, unsigned samples, VkSampleMask mask, bool advanced)
{
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct push),
      },
   }, NULL, &layout));
   VkShaderModule vs = make_module(spv_quad_vert, sizeof(spv_quad_vert));
   VkShaderModule fs = make_module(spv_solid_frag, sizeof(spv_solid_frag));
   const VkPipelineColorBlendAttachmentState blend = {
      .blendEnable = VK_TRUE, .colorWriteMask = 0xf,
      .srcColorBlendFactor = advanced ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_CONSTANT_COLOR,
      .dstColorBlendFactor = advanced ? VK_BLEND_FACTOR_ONE : VK_BLEND_FACTOR_CONSTANT_ALPHA,
      .colorBlendOp = advanced ? VK_BLEND_OP_MULTIPLY_EXT : VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = advanced ? VK_BLEND_OP_MULTIPLY_EXT : VK_BLEND_OP_ADD,
   };
   VkPipeline pipeline = make_pipeline_samples_state(layout, vs, fs, 0, samples, mask,
                                                     VK_FORMAT_UNDEFINED, &blend);
   struct image target = make_image_samples(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                            samples);
   struct image resolved = make_image(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   struct buffer out = make_buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
   const float initial[] = { .2f, .4f, .6f, advanced ? .75f : 1.f };
   const float constants[] = { .25f, .5f, .75f, .125f };
   const struct push push = { { .4f, .2f, .1f, .5f }, { 0, 0 } };
   VkCommandBuffer cmd = begin();
   const VkRenderingAttachmentInfo color = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = target.view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.color.float32 = { initial[0], initial[1], initial[2], initial[3] },
      .resolveMode = samples > 1 ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
      .resolveImageView = samples > 1 ? resolved.view : VK_NULL_HANDLE,
      .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   begin_rendering(cmd, &(VkRenderingInfo){
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = { { 0, 0 }, { W, H } }, .layerCount = 1,
      .colorAttachmentCount = 1, .pColorAttachments = &color,
   });
   vkCmdSetViewport(cmd, 0, 1, &(VkViewport){ 0, 0, W, H, 0, 1 });
   vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){ { 0, 0 }, { W, H } });
   vkCmdSetBlendConstants(cmd, constants);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   const VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &zero);
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(push), &push);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   end_rendering(cmd);
   vkCmdCopyImageToBuffer(cmd, samples > 1 ? resolved.image : target.image,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1, &(VkBufferImageCopy){
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { W, H, 1 },
   });
   submit_and_wait(cmd);
   float expected[4];
   memcpy(expected, initial, sizeof(expected));
   for (unsigned draw = 0; draw < 2; ++draw) {
      const float dst_alpha = expected[3];
      for (unsigned c = 0; c < 4; ++c) {
         const float src = push.tint[c], dst = expected[c];
         expected[c] = advanced ? (c == 3 ? src + dst - src * dst
                                             : src * (1 - dst_alpha) + dst * (1 - push.tint[3]) + src * dst)
                                : (c == 3 ? src : src * constants[c] + dst * constants[3]);
      }
   }
   const uint8_t *p = pixel(&out, W / 4, H / 4);
   for (unsigned c = 0; c < 4; ++c) {
      const float value = mask == 5 ? (expected[c] + initial[c]) * .5f : expected[c];
      const unsigned want = (unsigned)(value * 255 + .5f);
      CHECK(near(p[c], want), "shader blend advanced=%u samples=%u mask=%x channel=%u: %u want %u",
            advanced, samples, mask, c, p[c], want);
   }
   free_buffer(&out);
   free_image(&target);
   free_image(&resolved);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   printf("shader blend advanced=%u samples=%u mask=%x: done\n", advanced, samples, mask);
}

/* A pooled stencil image is larger than two differently-sized colour targets.
 * The clear determines whether the quad passes stencil; reusing it at another
 * extent checks auxiliary-view replacement as well as the first allocation. */
static void
test_transient_stencil(VkBuffer vertices)
{
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct push),
      },
   }, NULL, &layout));
   VkShaderModule vs = make_module(spv_quad_vert, sizeof(spv_quad_vert));
   VkShaderModule fs = make_module(spv_color_frag, sizeof(spv_color_frag));
   VkPipeline pipeline = make_pipeline_samples_state(layout, vs, fs, 0, VK_SAMPLE_COUNT_1_BIT,
                                                     ~0u, VK_FORMAT_S8_UINT, NULL);
   struct image stencil = make_image(W * 2, H * 2, VK_FORMAT_S8_UINT,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
   struct buffer out = make_buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
   const VkDeviceSize zero = 0;
   const struct push push = { { 1, 1, 1, 1 }, { 0, 0 } };
   for (unsigned size = 0; size < 2; ++size) {
      const uint32_t width = W - size * 8, height = H - size * 16;
      struct image target = make_image(width, height, VK_FORMAT_R8G8B8A8_UNORM,
                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
      for (unsigned clear = 0; clear < 2; ++clear) {
         VkCommandBuffer cmd = begin();
         const VkRenderingAttachmentInfo color = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = target.view, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue.color.float32 = { 0, 0, 0, 1 },
         };
         const VkRenderingAttachmentInfo ds = {
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = stencil.view, .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .clearValue.depthStencil.stencil = clear ? 7 : 3,
         };
         begin_rendering(cmd, &(VkRenderingInfo){
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { { 0, 0 }, { width, height } }, .layerCount = 1,
            .colorAttachmentCount = 1, .pColorAttachments = &color, .pStencilAttachment = &ds,
         });
         vkCmdSetViewport(cmd, 0, 1, &(VkViewport){ 0, 0, width, height, 0, 1 });
         vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){ { 0, 0 }, { width, height } });
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
         vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &zero);
         vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                            0, sizeof(push), &push);
         vkCmdDraw(cmd, 4, 1, 0, 0);
         end_rendering(cmd);
         vkCmdCopyImageToBuffer(cmd, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1,
                                &(VkBufferImageCopy){
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { width, height, 1 },
         });
         submit_and_wait(cmd);
         const uint8_t *p = (const uint8_t *)out.map + (width + 1) * 4;
         CHECK(clear ? mostly(p, true, false, false) : is(p, 0, 0, 0),
               "transient stencil %ux%u clear %u: %u,%u,%u,%u", width, height, clear,
               p[0], p[1], p[2], p[3]);
      }
      free_image(&target);
   }
   free_buffer(&out);
   free_image(&stencil);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   printf("transient stencil extents: done\n");
}

static void
test_precise_queries(VkBuffer vertices, VkSampleCountFlagBits samples, VkSampleMask mask, unsigned covered)
{
   VkPhysicalDeviceFeatures features;
   vkGetPhysicalDeviceFeatures(pdev, &features);
   CHECK(features.occlusionQueryPrecise, "precise occlusion queries unavailable");
   if (!features.occlusionQueryPrecise)
      return;
   VkQueryPool queries;
   VK(vkCreateQueryPool(dev, &(VkQueryPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_OCCLUSION, .queryCount = 5,
   }, NULL, &queries));
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct push),
      },
   }, NULL, &layout));
   VkShaderModule vs = make_module(spv_quad_vert, sizeof(spv_quad_vert));
   VkShaderModule fs = make_module(spv_color_frag, sizeof(spv_color_frag));
   VkPipeline pipeline = make_pipeline_samples(layout, vs, fs, 0, samples, mask);
   VkShaderModule discard = make_module(spv_discard_frag, sizeof(spv_discard_frag));
   VkPipeline discard_pipeline = make_pipeline_samples(layout, vs, discard, 0, samples, mask);
   struct image target = make_image_samples(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, samples);
   VkCommandBuffer cmd = begin();
   vkCmdResetQueryPool(cmd, queries, 0, 5);
   begin_pass(cmd, &target, false);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   const VkDeviceSize zero = 0;
   vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &zero);
   const struct push push = { { 1, 1, 1, 1 }, { 0, 0 } };
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                      0, sizeof(push), &push);
   vkCmdBeginQuery(cmd, queries, 0, VK_QUERY_CONTROL_PRECISE_BIT);
   vkCmdDraw(cmd, 4, 3, 0, 0);
   /* Clears and a split of the underlying WebGPU pass must not change the
    * count, and the second draw's overlapping samples count again. */
   vkCmdClearAttachments(cmd, 1, &(VkClearAttachment){
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0,
   }, 1, &(VkClearRect){ .rect = {{0,0},{W,H}}, .layerCount = 1 });
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                        0, 0, NULL, 0, NULL, 0, NULL);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdEndQuery(cmd, queries, 0);
   vkCmdBeginQuery(cmd, queries, 1, VK_QUERY_CONTROL_PRECISE_BIT);
   vkCmdEndQuery(cmd, queries, 1);
   vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){ {0,0}, {W/2,H/2} });
   vkCmdBeginQuery(cmd, queries, 2, VK_QUERY_CONTROL_PRECISE_BIT);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdEndQuery(cmd, queries, 2);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, discard_pipeline);
   vkCmdBeginQuery(cmd, queries, 3, VK_QUERY_CONTROL_PRECISE_BIT);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdEndQuery(cmd, queries, 3);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBeginQuery(cmd, queries, 4, VK_QUERY_CONTROL_PRECISE_BIT);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdEndQuery(cmd, queries, 4);
   end_rendering(cmd);
   submit_and_wait(cmd);
   uint64_t result[5][2] = {{0}};
   VK(vkGetQueryPoolResults(dev, queries, 0, 5, sizeof(result), result, sizeof(result[0]),
      VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT));
   const uint64_t expected[] = { 4 * W * H * covered, 0, W * H * covered / 4,
                                 W * H * covered / 8, W * H * covered / 4 };
   for (unsigned i = 0; i < 5; ++i)
      CHECK(result[i][1] && result[i][0] == expected[i], "precise query %u: %llu available %llu, expected %llu",
            i, (unsigned long long)result[i][0], (unsigned long long)result[i][1], (unsigned long long)expected[i]);
   vkDestroyQueryPool(dev, queries, NULL);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyPipeline(dev, discard_pipeline, NULL);
   vkDestroyShaderModule(dev, discard, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   free_image(&target);
   printf("precise queries (%u samples, mask %x): done\n", samples, mask);
}

static struct image
make_linear_texture(void **map, VkSubresourceLayout *layout)
{
   const VkFormat formats[] = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM };
   for (unsigned i = 0; i < 2; ++i) {
      VkSamplerYcbcrConversionImageFormatProperties ycbcr = {
         .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
      };
      VkImageFormatProperties2 props = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, .pNext = &ycbcr,
      };
      VkPhysicalDeviceImageFormatInfo2 info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
         .format = formats[i], .type = VK_IMAGE_TYPE_2D, .tiling = VK_IMAGE_TILING_LINEAR,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      };
      VK(vkGetPhysicalDeviceImageFormatProperties2(pdev, &info, &props));
      CHECK(ycbcr.combinedImageSamplerDescriptorCount == i + 1,
            "linear plane descriptor count: %u", ycbcr.combinedImageSamplerDescriptorCount);
      CHECK(props.imageFormatProperties.maxMipLevels == 1 && props.imageFormatProperties.maxArrayLayers == 1,
            "linear subresource limits");
      info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      CHECK(vkGetPhysicalDeviceImageFormatProperties2(pdev, &info, &props) == VK_ERROR_FORMAT_NOT_SUPPORTED,
            "linear storage images must remain unsupported");
   }
   struct image im = { 0 };
   VK(vkCreateImage(dev, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = { 2, 2, 1 }, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
   }, NULL, &im.image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, im.image, &req);
   VK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size + req.alignment,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, HOST),
   }, NULL, &im.memory));
   VK(vkBindImageMemory(dev, im.image, im.memory, req.alignment));
   void *allocation;
   VK(vkMapMemory(dev, im.memory, 0, VK_WHOLE_SIZE, 0, &allocation));
   *map = (uint8_t *)allocation + req.alignment;
   vkGetImageSubresourceLayout(dev, im.image, &(VkImageSubresource){
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
   }, layout);
   VK(vkCreateImageView(dev, &(VkImageViewCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
   }, NULL, &im.view));
   return im;
}

static void
test_texture(VkBuffer vertices, bool linear, VkFormat format)
{
   /* A 2x2 texture, a texel per corner, sampled nearest. */
   const bool norm16 = format != VK_FORMAT_R8G8B8A8_UNORM;
   void *linear_map = NULL;
   VkSubresourceLayout linear_layout;
   struct image tex = linear ? make_linear_texture(&linear_map, &linear_layout)
                            : make_image(2, 2, format,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   struct buffer staging = make_buffer(norm16 ? 32 : 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HOST);
   const uint8_t texels[16] = { 255, 0, 0, 255,   0, 255, 0, 255,
                                0, 0, 255, 255,   255, 255, 255, 255 };
   memcpy(staging.map, texels, sizeof(texels));
   struct buffer ubo = make_buffer_at(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST, 16);
   const float scale[4] = { 1.0f, 0.5f, 1.0f, 1.0f };
   memset(ubo.map, 0, sizeof(scale));

   VkSampler sampler;
   VK(vkCreateSampler(dev, &(VkSamplerCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = norm16 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
      .minFilter = norm16 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f,
   }, NULL, &sampler));

   const VkDescriptorSetLayoutBinding bindings[] = {
      { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
   };
   VkDescriptorSetLayout set_layout;
   VK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2,
      .pBindings = bindings,
   }, NULL, &set_layout));
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
      .pSetLayouts = &set_layout, .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(struct push) },
   }, NULL, &layout));
   const VkDescriptorPoolSize sizes[] = {
      { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
   };
   VkDescriptorPool dpool;
   VK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 2,
      .pPoolSizes = sizes,
   }, NULL, &dpool));
   VkDescriptorSet set;
   VK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool,
      .descriptorSetCount = 1, .pSetLayouts = &set_layout,
   }, &set));
   const VkWriteDescriptorSet writes[] = {
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &(VkDescriptorBufferInfo){ ubo.buffer, 0, VK_WHOLE_SIZE } },
      { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
        .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &(VkDescriptorImageInfo){ sampler, tex.view,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL } },
   };
   vkUpdateDescriptorSets(dev, 2, writes, 0, NULL);

   VkShaderModule vs = make_module(spv_quad_vert, sizeof(spv_quad_vert));
   VkShaderModule fs = make_module(spv_texture_frag, sizeof(spv_texture_frag));
   VkPipeline pipeline = make_pipeline(layout, vs, fs);

   struct image target = make_image(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   struct buffer out = make_buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
   const VkDeviceSize zero = 0;

   for (unsigned round = 0; round < ((linear || norm16) ? 3u : 1u); ++round) {
      if (norm16) {
         const unsigned max = format == VK_FORMAT_R16G16B16A16_SNORM ? 32767 : 65535;
         for (unsigned n = 0; n < 16; ++n) {
            const unsigned c = n % 4;
            const unsigned from = round == 1 && (c == 0 || c == 2) ? n ^ 2u : n;
            ((uint16_t *)staging.map)[n] = texels[from] ? max : 0;
         }
      }
      /* Refresh after a host rewrite, then after a GPU buffer-to-image copy.
       * The sampled shadow must never overwrite authoritative linear memory. */
      if (linear && round < 2) {
         for (unsigned y = 0; y < 2; ++y) {
            uint8_t *row = (uint8_t *)linear_map + linear_layout.offset + y * linear_layout.rowPitch;
            memcpy(row, texels + y * 8, 8);
            if (round == 1)
               for (unsigned x = 0; x < 2; ++x) {
               const uint8_t r = row[x * 4];
               row[x * 4] = row[x * 4 + 2];
               row[x * 4 + 2] = r;
               }
         }
      }
      VkCommandBuffer cmd = begin();
      /* The aligned binding must copy the GPU's current bytes, rather than
       * reading the zeroes from the mapped allocation. */
      vkCmdUpdateBuffer(cmd, ubo.buffer, 0, sizeof(scale), scale);
      if (!linear || round == 2) {
         vkCmdCopyBufferToImage(cmd, staging.buffer, tex.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                &(VkBufferImageCopy){
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { 2, 2, 1 },
         });
      }
      begin_pass(cmd, &target, false);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &zero);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
      const struct push push = { { 1, 1, 1, 1 }, { 0, 0 } };
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
      vkCmdDraw(cmd, 4, 1, 0, 0);
      end_pass_and_read(cmd, &target, &out);
      submit_and_wait(cmd);

      const uint8_t *tl = pixel(&out, 0, 0), *tr = pixel(&out, W - 1, 0);
      const uint8_t *bl = pixel(&out, 0, H - 1), *br = pixel(&out, W - 1, H - 1);
      CHECK(is(tl, round == 1 ? 0 : 255, 0, round == 1 ? 255 : 0),
            "texture top left round %u: %u %u %u %u", round, tl[0], tl[1], tl[2], tl[3]);
      CHECK(is(tr, 0, 128, 0), "texture top right: %u %u %u %u", tr[0], tr[1], tr[2], tr[3]);
      CHECK(is(bl, round == 1 ? 255 : 0, 0, round == 1 ? 0 : 255),
            "texture bottom left round %u: %u %u %u %u", round, bl[0], bl[1], bl[2], bl[3]);
      CHECK(is(br, 255, 128, 255), "texture bottom right: %u %u %u %u", br[0], br[1], br[2], br[3]);
      if (norm16) {
         const uint8_t *middle = pixel(&out, W / 2, H / 2);
         CHECK(is(middle, round == 1 ? 131 : 128, 66, round == 1 ? 128 : 131),
               "norm16 bilinear center format %u round %u: %u %u %u %u",
               format, round, middle[0], middle[1], middle[2], middle[3]);
      }
   }

   free_buffer(&out);
   free_image(&target);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   vkDestroySampler(dev, sampler, NULL);
   free_buffer(&ubo);
   free_buffer(&staging);
   if (linear_map)
      vkUnmapMemory(dev, tex.memory);
   free_image(&tex);
   printf("%stexture format %u: done\n", linear ? "linear sampled " : "", format);
}

/* A subpassInputMS of an MSRTSS image must read individual samples of
 * the emulated attachment, not the application's resolved image. */
static void
test_msrtss_input(VkBuffer vertices)
{
   VkDescriptorSetLayout set_layout;
   VK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
      .pBindings = &(VkDescriptorSetLayoutBinding){ 0, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1,
                                                  VK_SHADER_STAGE_FRAGMENT_BIT, NULL },
   }, NULL, &set_layout));
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set_layout, .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct push) },
   }, NULL, &layout));
   VkShaderModule vs = make_module(spv_quad_vert, sizeof(spv_quad_vert));
   VkShaderModule solid = make_module(spv_solid_frag, sizeof(spv_solid_frag));
   VkShaderModule input = make_module(spv_ms_input_frag, sizeof(spv_ms_input_frag));
   VkPipeline write = make_pipeline_samples(layout, vs, solid, 0, VK_SAMPLE_COUNT_4_BIT, 1);
   VkPipeline read = make_pipeline_samples(layout, vs, input, 0, VK_SAMPLE_COUNT_4_BIT, ~0u);
   struct image target = make_image_samples_flags(W, H, VK_FORMAT_R8G8B8A8_UNORM,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_BIT_EXT);
   struct buffer out = make_buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
   VkDescriptorPool desc_pool;
   VK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1,
      .pPoolSizes = &(VkDescriptorPoolSize){ VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1 },
   }, NULL, &desc_pool));
   VkDescriptorSet set;
   VK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = desc_pool,
      .descriptorSetCount = 1, .pSetLayouts = &set_layout,
   }, &set));
   vkUpdateDescriptorSets(dev, 1, &(VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
      .pImageInfo = &(VkDescriptorImageInfo){ VK_NULL_HANDLE, target.view, VK_IMAGE_LAYOUT_GENERAL },
   }, 0, NULL);
   VkCommandBuffer cmd = begin();
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 1,
      &(VkImageMemoryBarrier){ .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
         .image = target.image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1} });
   const uint32_t index = 0;
   const VkRenderingInputAttachmentIndexInfoKHR indices = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INPUT_ATTACHMENT_INDEX_INFO_KHR,
      .colorAttachmentCount = 1, .pColorAttachmentInputIndices = &index,
   };
   const VkMultisampledRenderToSingleSampledInfoEXT ms = {
      .sType = VK_STRUCTURE_TYPE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_INFO_EXT,
      .pNext = &indices, .multisampledRenderToSingleSampledEnable = VK_TRUE,
      .rasterizationSamples = VK_SAMPLE_COUNT_4_BIT,
   };
   begin_rendering(cmd, &(VkRenderingInfo){
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .pNext = &ms,
      .renderArea = {{0, 0}, {W, H}}, .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &(VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = target.view,
         .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      },
   });
   vkCmdSetViewport(cmd, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
   vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){{0, 0}, {W, H}});
   const VkDeviceSize zero = 0;
   const struct push push = { .tint = {1, 0, 0, 1} };
   vkCmdBindVertexBuffers(cmd, 0, 1, &vertices, &zero);
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(push), &push);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, write);
   vkCmdDraw(cmd, 4, 1, 0, 0);
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1,
      &(VkMemoryBarrier){ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT }, 0, NULL, 0, NULL);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, read);
   vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
   /* Three reads swap red/green back and forth, invalidating snapshots
    * and restarting the pass between them. The final color is green. */
   for (unsigned draw = 0; draw < 3; ++draw) {
      if (draw)
         vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_DEPENDENCY_BY_REGION_BIT, 1,
            &(VkMemoryBarrier){ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
               .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
               .dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT }, 0, NULL, 0, NULL);
      vkCmdDraw(cmd, 4, 1, 0, 0);
   }
   end_pass_and_read(cmd, &target, &out);
   submit_and_wait(cmd);
   for (unsigned y = 0; y < H; ++y) {
      for (unsigned x = 0; x < W; ++x) {
         const uint8_t *p = pixel(&out, x, y);
         CHECK(is(p, 0, 255, 0), "MSRTSS sample-zero read at %u,%u: %u %u %u %u",
               x, y, p[0], p[1], p[2], p[3]);
      }
   }
   vkDestroyDescriptorPool(dev, desc_pool, NULL);
   free_buffer(&out);
   free_image(&target);
   vkDestroyPipeline(dev, read, NULL);
   vkDestroyPipeline(dev, write, NULL);
   vkDestroyShaderModule(dev, input, NULL);
   vkDestroyShaderModule(dev, solid, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   printf("MSRTSS sample-zero input attachment: done\n");
}

static void
test_norm16_precision(void)
{
   const VkDescriptorSetLayoutBinding bindings[] = {
      { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL },
   };
   VkDescriptorSetLayout sl;
   VK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bindings,
   }, NULL, &sl));
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &sl,
   }, NULL, &layout));
   VkShaderModule shader = make_module(spv_norm16_sample_comp, sizeof(spv_norm16_sample_comp));
   VkPipeline pipeline;
   VK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
      .stage = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                 .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main" },
   }, NULL, &pipeline));
   VkSampler sampler;
   VK(vkCreateSampler(dev, &(VkSamplerCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .minFilter = VK_FILTER_LINEAR,
      .magFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   }, NULL, &sampler));
   const VkDescriptorPoolSize sizes[] = {
      { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
   };
   VkDescriptorPool pool;
   VK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
      .poolSizeCount = 2, .pPoolSizes = sizes,
   }, NULL, &pool));
   VkDescriptorSet set;
   VK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool,
      .descriptorSetCount = 1, .pSetLayouts = &sl,
   }, &set));
   struct buffer source = make_buffer(32, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HOST);
   struct buffer result = make_buffer(32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST);
   const VkFormat formats[] = { VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM,
      VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM,
      VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM };
   for (unsigned f = 0; f < 6; ++f) {
      const unsigned channels = 1u << (f / 2);
      const bool sign = f & 1;
      struct image image = make_image(2, 2, formats[f], VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
      const VkWriteDescriptorSet writes[] = {
         { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
           .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           .pImageInfo = &(VkDescriptorImageInfo){ sampler, image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL } },
         { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
           .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           .pBufferInfo = &(VkDescriptorBufferInfo){ result.buffer, 0, 32 } },
      };
      vkUpdateDescriptorSets(dev, 2, writes, 0, NULL);
      for (unsigned round = 0; round < 2; ++round) {
         float expected[8] = {0, 0, 0, 1, 0, 0, 0, 1};
         for (unsigned n = 0; n < 4 * channels; ++n) {
            const int value = 12345 + n * 719 + round * 1337;
            const int encoded = sign && (n & 1) ? -value : value;
            ((uint16_t *)source.map)[n] = encoded;
            const float decoded = (float)encoded / (sign ? 32767.0f : 65535.0f);
            const unsigned c = n % channels;
            if (n < channels) expected[c] = decoded;
            if (n < channels) expected[4 + c] = 0;
            expected[4 + c] += decoded * 0.25f;
         }
         memset(result.map, 0, 32);
         VkCommandBuffer cmd = begin();
         vkCmdCopyBufferToImage(cmd, source.buffer, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &(VkBufferImageCopy){ .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                                    .imageExtent = {2, 2, 1} });
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
         vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
         vkCmdDispatch(cmd, 1, 1, 1);
         submit_and_wait(cmd);
         /* Native normalized filtering may round to the source format's
          * precision. The texel-center check still requires full float32
          * conversion, catching an intermediate float16 sampling copy. */
         for (unsigned n = 0; n < 8; ++n)
            CHECK(fabsf(((float *)result.map)[n] - expected[n]) <
                     (n < 4 ? 0.0000002f : 1.0f / (sign ? 32767.0f : 65535.0f)),
                  "norm16 precision format %u round %u channel %u: %.9f expected %.9f",
                  formats[f], round, n, ((float *)result.map)[n], expected[n]);
      }
      free_image(&image);
   }
   free_buffer(&source);
   free_buffer(&result);
   vkDestroyDescriptorPool(dev, pool, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyShaderModule(dev, shader, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorSetLayout(dev, sl, NULL);
   printf("norm16 full precision and rewrites: done\n");
}

static void
test_clear_attachments(void)
{
   struct image target = make_image(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   struct buffer out = make_buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
   VkCommandBuffer cmd = begin();
   begin_pass(cmd, &target, false);
   vkCmdClearAttachments(cmd, 1, &(VkClearAttachment){
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0,
      .clearValue.color.float32 = { 0, 1, 0, 1 },
   }, 1, &(VkClearRect){ .rect = { { 16, 8 }, { 32, 16 } }, .baseArrayLayer = 0, .layerCount = 1 });
   end_pass_and_read(cmd, &target, &out);
   submit_and_wait(cmd);
   const uint8_t *in = pixel(&out, 20, 10), *outside = pixel(&out, 20, 30), *corner = pixel(&out, 0, 0);
   CHECK(is(in, 0, 255, 0), "cleared rect: %u %u %u %u", in[0], in[1], in[2], in[3]);
   CHECK(outside[1] == 0 && corner[1] == 0, "cleared outside the rect: %u %u", outside[1], corner[1]);
   free_buffer(&out);
   free_image(&target);
   printf("clear attachments: done\n");
}

static void
test_blit(void)
{
   /* 2x2 up to W x H, nearest: four quadrants. */
   struct image src = make_image(2, 2, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   struct image dst = make_image(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_DST_BIT);
   struct buffer staging = make_buffer(16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HOST);
   const uint8_t texels[16] = { 255, 0, 0, 255,   0, 255, 0, 255,
                                0, 0, 255, 255,   255, 255, 255, 255 };
   memcpy(staging.map, texels, sizeof(texels));
   struct buffer out = make_buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);

   VkCommandBuffer cmd = begin();
   vkCmdCopyBufferToImage(cmd, staging.buffer, src.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                          &(VkBufferImageCopy){
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { 2, 2, 1 },
   });
   vkCmdBlitImage(cmd, src.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst.image,
                  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &(VkImageBlit){
      .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .srcOffsets = { { 0, 0, 0 }, { 2, 2, 1 } },
      .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
      .dstOffsets = { { 0, 0, 0 }, { W, H, 1 } },
   }, VK_FILTER_NEAREST);
   vkCmdCopyImageToBuffer(cmd, dst.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out.buffer, 1,
                          &(VkBufferImageCopy){
      .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 }, .imageExtent = { W, H, 1 },
   });
   submit_and_wait(cmd);
   const uint8_t *tl = pixel(&out, 1, 1), *tr = pixel(&out, W - 2, 1);
   const uint8_t *bl = pixel(&out, 1, H - 2), *br = pixel(&out, W - 2, H - 2);
   CHECK(is(tl, 255, 0, 0), "blit top left: %u %u %u %u", tl[0], tl[1], tl[2], tl[3]);
   CHECK(is(tr, 0, 255, 0), "blit top right: %u %u %u %u", tr[0], tr[1], tr[2], tr[3]);
   CHECK(is(bl, 0, 0, 255), "blit bottom left: %u %u %u %u", bl[0], bl[1], bl[2], bl[3]);
   CHECK(is(br, 255, 255, 255), "blit bottom right: %u %u %u %u", br[0], br[1], br[2], br[3]);
   free_buffer(&out);
   free_buffer(&staging);
   free_image(&dst);
   free_image(&src);
   printf("blit: done\n");
}

/* A 4x2 grid of cells: the cell of pixel (x, y). */
static const uint8_t *
cell(const struct buffer *b, int column, int row)
{
   return pixel(b, column * (W / 4) + W / 8, row * (H / 2) + H / 4);
}

static void
test_instances(void)
{
   VkPipelineLayout layout;
   VK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
         VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(struct push),
      },
   }, NULL, &layout));
   VkShaderModule vs = make_module(spv_instanced_vert, sizeof(spv_instanced_vert));
   VkShaderModule fs = make_module(spv_color_frag, sizeof(spv_color_frag));
   VkPipeline pipeline = make_pipeline_with(layout, vs, fs, 2);

   /* Per instance, a divisor of 2 apart: element firstInstance +
    * (InstanceIndex - firstInstance) / 2. */
   static const uint8_t colors[][4] = {
      { 0, 0, 0, 255 }, { 255, 0, 0, 255 }, { 0, 255, 0, 255 }, { 0, 0, 255, 255 }, { 255, 255, 255, 255 },
   };
   struct buffer instances = make_buffer(sizeof(colors), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, HOST);
   memcpy(instances.map, colors, sizeof(colors));
   static const uint16_t strip[] = { 0, 1, 2, 3 };
   struct buffer indices = make_buffer(sizeof(strip), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, HOST);
   memcpy(indices.map, strip, sizeof(strip));
   /* Past something else, as buffers of commands are. */
   const VkDrawIndexedIndirectCommand commands[] = {
      { .indexCount = 4, .instanceCount = 4, .vertexOffset = 10, .firstInstance = 1 },
      { .indexCount = 4, .instanceCount = 4, .vertexOffset = 20, .firstInstance = 3 },
   };
   const VkDeviceSize command_offset = 100;
   struct buffer indirect = make_buffer(command_offset + sizeof(commands), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, HOST);
   memcpy((uint8_t *)indirect.map + command_offset, commands, sizeof(commands));

   struct image target = make_image(W, H, VK_FORMAT_R8G8B8A8_UNORM,
                                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
   struct buffer out = make_buffer(W * H * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
   const VkDeviceSize zero = 0;
   const struct push push = { { 1, 1, 1, 1 }, { 0, 0 } };

   /* Two indirect draws, one a row: row 0 instances 1-4, row 1 3-6. */
   VkCommandBuffer cmd = begin();
   begin_pass(cmd, &target, false);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindVertexBuffers(cmd, 0, 1, &instances.buffer, &zero);
   vkCmdBindIndexBuffer(cmd, indices.buffer, 0, VK_INDEX_TYPE_UINT16);
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(push), &push);
   vkCmdDrawIndexedIndirect(cmd, indirect.buffer, command_offset, 2, sizeof(VkDrawIndexedIndirectCommand));
   end_pass_and_read(cmd, &target, &out);
   submit_and_wait(cmd);
   static const int expect[2][4] = { { 1, 1, 2, 2 }, { 3, 3, 4, 4 } };
   for (int row = 0; row < 2; ++row) {
      for (int column = 0; column < 4; ++column) {
         const uint8_t *p = cell(&out, column, row), *c = colors[expect[row][column]];
         CHECK(is(p, c[0], c[1], c[2]), "indirect: cell %d,%d is %u %u %u %u, not colour %d", column, row,
               p[0], p[1], p[2], p[3], expect[row][column]);
      }
   }

   /* A direct draw: row 0 again, instances 2-5. */
   cmd = begin();
   begin_pass(cmd, &target, false);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   vkCmdBindVertexBuffers(cmd, 0, 1, &instances.buffer, &zero);
   vkCmdBindIndexBuffer(cmd, indices.buffer, 0, VK_INDEX_TYPE_UINT16);
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                      sizeof(push), &push);
   vkCmdDrawIndexed(cmd, 4, 4, 0, 30, 2);
   end_pass_and_read(cmd, &target, &out);
   submit_and_wait(cmd);
   static const int direct[4] = { 2, 2, 3, 3 };
   for (int column = 0; column < 4; ++column) {
      const uint8_t *p = cell(&out, column, 0), *c = colors[direct[column]];
      CHECK(is(p, c[0], c[1], c[2]), "direct: cell %d is %u %u %u %u, not colour %d", column, p[0], p[1],
            p[2], p[3], direct[column]);
   }
   const uint8_t *empty = cell(&out, 0, 1);
   CHECK(empty[0] == 0 && empty[1] == 0 && empty[2] == 0, "direct: row 1 was drawn: %u %u %u", empty[0],
         empty[1], empty[2]);

   free_buffer(&out);
   free_image(&target);
   free_buffer(&indirect);
   free_buffer(&indices);
   free_buffer(&instances);
   vkDestroyPipeline(dev, pipeline, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   printf("draw parameters and divisors: done\n");
}

int
main(void)
{
   const char *extensions[] = { "VK_KHR_portability_subset", "VK_KHR_dynamic_rendering",
                                "VK_KHR_vertex_attribute_divisor", "VK_EXT_blend_operation_advanced",
                                "VK_EXT_multisampled_render_to_single_sampled",
                                "VK_KHR_dynamic_rendering_local_read" };
   VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR local_read = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR,
      .dynamicRenderingLocalRead = VK_TRUE,
   };
   VkPhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT msrtss = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES_EXT,
      .pNext = &local_read, .multisampledRenderToSingleSampled = VK_TRUE,
   };
   VkPhysicalDeviceVertexAttributeDivisorFeaturesKHR divisor = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_KHR,
      .pNext = &msrtss, .vertexAttributeInstanceRateDivisor = VK_TRUE,
   };
   VkPhysicalDeviceDynamicRenderingFeatures rendering = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES, .pNext = &divisor,
      .dynamicRendering = VK_TRUE,
   };
   VkPhysicalDeviceVulkan11Features vk11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, .pNext = &rendering,
      .shaderDrawParameters = VK_TRUE,
   };
   test_init(6, extensions, &(VkPhysicalDeviceFeatures2){
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &vk11,
      .features = { .multiDrawIndirect = VK_TRUE, .occlusionQueryPrecise = VK_TRUE },
   });
   begin_rendering = (PFN_vkCmdBeginRenderingKHR)vkGetDeviceProcAddr(dev, "vkCmdBeginRenderingKHR");
   end_rendering = (PFN_vkCmdEndRenderingKHR)vkGetDeviceProcAddr(dev, "vkCmdEndRenderingKHR");

   struct buffer vertices = make_buffer(sizeof(quad), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, HOST);
   memcpy(vertices.map, quad, sizeof(quad));

   test_colors(vertices.buffer, spv_quad_vert, sizeof(spv_quad_vert));
   test_colors(vertices.buffer, spv_dead_input_vert, sizeof(spv_dead_input_vert));
   for (unsigned advanced = 0; advanced < 2; ++advanced) {
      test_shader_blend(vertices.buffer, 1, ~0u, advanced);
      test_shader_blend(vertices.buffer, 4, ~0u, advanced);
      test_shader_blend(vertices.buffer, 4, 5, advanced);
   }
   test_transient_stencil(vertices.buffer);
   test_precise_queries(vertices.buffer, VK_SAMPLE_COUNT_1_BIT, ~0u, 1);
   test_precise_queries(vertices.buffer, VK_SAMPLE_COUNT_4_BIT, ~0u, 4);
   test_precise_queries(vertices.buffer, VK_SAMPLE_COUNT_4_BIT, 5u, 2);
   test_texture(vertices.buffer, false, VK_FORMAT_R8G8B8A8_UNORM);
   test_texture(vertices.buffer, true, VK_FORMAT_R8G8B8A8_UNORM);
   test_texture(vertices.buffer, false, VK_FORMAT_R16G16B16A16_UNORM);
   test_texture(vertices.buffer, false, VK_FORMAT_R16G16B16A16_SNORM);
   test_norm16_precision();
   test_msrtss_input(vertices.buffer);
   test_clear_attachments();
   test_blit();
   test_instances();

   free_buffer(&vertices);
   return test_finish();
}
