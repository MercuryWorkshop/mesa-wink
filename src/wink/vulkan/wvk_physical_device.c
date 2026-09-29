/*
 * Copyright 2026 wink contributors
 * SPDX-License-Identifier: MIT
 *
 * The physical device: one WebGPU adapter, described as Vulkan.
 *
 * Limits are the adapter's, since vkCreateDevice asks for all of them.
 * Features are what WebGPU can do and the driver implements; where
 * Vulkan requires more than WebGPU has, the smaller truth is reported
 * rather than the requirement -- 8 vertex bindings, not 16 -- which is
 * knowingly non-conformant and why VK_KHR_portability_subset is exposed.
 */
#include "wvk_physical_device.h"

#include "wvk_entrypoints.h"
#include "wvk_descriptor_set_layout.h"
#include "wvk_format.h"
#include "wvk_image.h"
#include "wvk_shader.h"
#include "wvk_instance.h"
#include "wvk_platform.h"
#include "wvk_sync.h"

#include "vk_enum_defines.h"
#include "vk_format.h"

#include "git_sha1.h"
#include "util/u_debug.h"

#include <stdio.h>
#include <stdlib.h>

/* The WebGPU features the driver has a use for; a device asks for those
 * the adapter has. Deliberately not the 16-bit normalized formats: the
 * browser has none, and the native path should not be able to do what the
 * browser cannot. */
static const WGPUFeatureName wanted_features[] = {
   WGPUFeatureName_DepthClipControl,
   WGPUFeatureName_Depth32FloatStencil8,
   WGPUFeatureName_TimestampQuery,
   WGPUFeatureName_TextureCompressionBC,
   WGPUFeatureName_TextureCompressionBCSliced3D,
   WGPUFeatureName_TextureCompressionETC2,
   WGPUFeatureName_TextureCompressionASTC,
   WGPUFeatureName_TextureCompressionASTCSliced3D,
   WGPUFeatureName_IndirectFirstInstance,
   WGPUFeatureName_ShaderF16,
   WGPUFeatureName_RG11B10UfloatRenderable,
   WGPUFeatureName_BGRA8UnormStorage,
   WGPUFeatureName_Float32Filterable,
   WGPUFeatureName_Float32Blendable,
   WGPUFeatureName_ClipDistances,
   WGPUFeatureName_DualSourceBlending,
   /* More storage formats, and the 16-bit normalized ones (wvk_format.c). */
   WGPUFeatureName_TextureFormatsTier1,
   WGPUFeatureName_TextureFormatsTier2,
#ifndef __EMSCRIPTEN__
   /* Native Dawn is only safe to call from several threads with this, and
    * Vulkan lets them all: queues of two families, and creating while
    * submitting. A browser's device is its thread's alone. */
   WGPUFeatureName_ImplicitDeviceSynchronization,
#endif
};

bool
wvk_physical_device_has_feature(const struct wvk_physical_device *pdev,
                                WGPUFeatureName feature)
{
   for (uint32_t i = 0; i < pdev->feature_count; ++i) {
      if (pdev->features[i] == feature)
         return true;
   }
   return false;
}

static uint32_t
api_version(void)
{
   const uint32_t override = vk_get_version_override();
   if (override)
      return override;
   const char *v = getenv("WINK_VK_API_VERSION");
   if (v && strcmp(v, "1.2") == 0)
      return VK_MAKE_API_VERSION(0, 1, 2, VK_HEADER_VERSION);
   return WVK_API_VERSION;
}

static void
get_device_extensions(const struct wvk_physical_device *pdev,
                      struct vk_device_extension_table *ext)
{
   *ext = (struct vk_device_extension_table){
      .KHR_bind_memory2 = true,
      .KHR_copy_commands2 = true,
      .KHR_create_renderpass2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_draw_indirect_count = true,
      .KHR_depth_stencil_resolve = true,
      .KHR_descriptor_update_template = true,
      .KHR_device_group = true,
      .KHR_driver_properties = true,
      .KHR_dynamic_rendering = true,
      .KHR_format_feature_flags2 = true,
      .KHR_get_memory_requirements2 = true,
      .KHR_image_format_list = true,
      .KHR_imageless_framebuffer = true,
      .KHR_maintenance1 = true,
      .KHR_maintenance2 = true,
      .KHR_maintenance3 = true,
      .KHR_maintenance4 = true,
      .KHR_maintenance5 = true,
      .KHR_maintenance6 = true,
      .KHR_multiview = true,
      .KHR_portability_subset = true,
      .KHR_push_descriptor = true,
      .KHR_relaxed_block_layout = true,
      .KHR_sampler_mirror_clamp_to_edge = true,
      .KHR_sampler_ycbcr_conversion = true,
      .EXT_sampler_filter_minmax = true,
      .EXT_index_type_uint8 = true,
      .EXT_nested_command_buffer = true,
      .EXT_primitives_generated_query = true,
      .EXT_shader_replicated_composites = true,
      .EXT_transform_feedback = true,
      .EXT_ycbcr_2plane_444_formats = true,
      .KHR_index_type_uint8 = true,
      .KHR_load_store_op_none = true,
      .KHR_shader_expect_assume = true,
      .KHR_shader_maximal_reconvergence = true,
      .KHR_shader_subgroup_uniform_control_flow = true,
      .KHR_separate_depth_stencil_layouts = true,
      .KHR_shader_draw_parameters = true,
      .KHR_16bit_storage = true,
      .EXT_scalar_block_layout = true,
      .EXT_multi_draw = true,
      .KHR_8bit_storage = true,
      .KHR_shader_float16_int8 = true,
      .KHR_shader_float_controls = true,
      .KHR_shader_float_controls2 = true,
      .KHR_shader_integer_dot_product = true,
      .KHR_shader_non_semantic_info = true,
      .KHR_shader_terminate_invocation = true,
      .KHR_shader_subgroup_extended_types = true,
      .KHR_shader_subgroup_rotate = true,
      .KHR_spirv_1_4 = true,
      .KHR_storage_buffer_storage_class = true,
      .KHR_swapchain = true,
      .KHR_synchronization2 = true,
      .KHR_timeline_semaphore = true,
      .KHR_uniform_buffer_standard_layout = true,
      .KHR_vulkan_memory_model = true,
      .KHR_zero_initialize_workgroup_memory = true,
      .EXT_4444_formats = true,
      .EXT_border_color_swizzle = true,
      .EXT_custom_border_color = true,
      .EXT_extended_dynamic_state = true,
      .EXT_extended_dynamic_state2 = true,
      .EXT_extended_dynamic_state3 = true,
      .EXT_graphics_pipeline_library = true,
      .EXT_shader_object = true,
      .EXT_color_write_enable = true,
      .KHR_pipeline_library = true,
      .EXT_host_query_reset = true,
      .EXT_image_robustness = true,
      .EXT_pipeline_robustness = true,
      .EXT_depth_clip_control = true,
      .EXT_host_image_copy = true,
      .EXT_depth_clip_enable = wvk_physical_device_has_feature(pdev, WGPUFeatureName_DepthClipControl),
      .KHR_global_priority = true,
      .EXT_global_priority = true,
      .EXT_global_priority_query = true,
      .EXT_inline_uniform_block = true,
      .EXT_pipeline_creation_cache_control = true,
      .EXT_pipeline_creation_feedback = true,
      .EXT_private_data = true,
      .EXT_separate_stencil_usage = true,
      .EXT_shader_viewport_index_layer = true,
      .EXT_shader_demote_to_helper_invocation = true,
      .EXT_subgroup_size_control = true,
      .EXT_texel_buffer_alignment = true,
      .EXT_tooling_info = true,
      /* WebGPU steps instances one at a time: other divisors are read by
       * the shader (wvk_shader.c, pull_divided_attributes). */
      .EXT_vertex_attribute_divisor = true,
      .KHR_vertex_attribute_divisor = true,
      .EXT_vertex_input_dynamic_state = true,
   };
   (void)pdev;
}

static void
get_device_features(const struct wvk_physical_device *pdev, struct vk_features *f)
{
   const bool bc = wvk_physical_device_has_feature(pdev, WGPUFeatureName_TextureCompressionBC);
   const bool etc2 = wvk_physical_device_has_feature(pdev, WGPUFeatureName_TextureCompressionETC2);
   const bool astc = wvk_physical_device_has_feature(pdev, WGPUFeatureName_TextureCompressionASTC);
   const bool clip = wvk_physical_device_has_feature(pdev, WGPUFeatureName_ClipDistances);

   *f = (struct vk_features){
      /* Vulkan 1.0. WebGPU bounds-checks every access. */
      .robustBufferAccess = true,
      .fullDrawIndexUint32 = true,
      .imageCubeArray = true,
      .independentBlend = true,
      .dualSrcBlend = wvk_physical_device_has_feature(pdev, WGPUFeatureName_DualSourceBlending),
      /* A CPU loop over indirect draws. */
      .multiDrawIndirect = true,
      /* VK_EXT_multi_draw: a draw for each. */
      .multiDraw = true,
      .drawIndirectFirstInstance = wvk_physical_device_has_feature(pdev, WGPUFeatureName_IndirectFirstInstance),
      /* WebGPU clamps depth always; unclippedDepth turns off the clipping. */
      .depthClamp = wvk_physical_device_has_feature(pdev, WGPUFeatureName_DepthClipControl),
      .depthBiasClamp = true,
      /* The fragment shader's, against a copy of the attachment taken
       * before the draw (wvk_shader.c, wvk_cmd_draw.c). */
      .logicOp = true,
      /* The fragment shader's too, against a copy of the depth attachment. */
      .depthBounds = true,
      .samplerAnisotropy = true,
      .textureCompressionETC2 = etc2,
      .textureCompressionASTC_LDR = astc,
      .textureCompressionBC = bc,
      .fragmentStoresAndAtomics = true,
      /* A vertex shader's storage writes are its compute shader's, run
       * before the draw (run_vertex_writes in wvk_cmd_draw.c). */
      .vertexPipelineStoresAndAtomics = true,
      /* Points of the size the vertex shader gives them: quads, a variant
       * of the shader moving six vertices to a point's corners. */
      .largePoints = true,
      /* And those a geometry or evaluation shader gives a size: its draw's
       * vertex shader, of its records, draws them so. */
      .shaderTessellationAndGeometryPointSize = true,
      /* The driver's counts, the host's and the shaders' (wvk_query.c). */
      .pipelineStatisticsQuery = true,
      /* As compute, before a draw of what it emits (draw_geometry). */
      .geometryShader = true,
      /* As compute, around a tessellator of the same (draw_tessellated). */
      .tessellationShader = true,
      /* Lines of the width the draw has: quads the same way, the vertex
       * shader run as compute giving the ends' positions. */
      .wideLines = true,
      /* Gathers at several offsets, or one not constant (wvk_shader.c). */
      .shaderImageGatherExtended = true,
      /* Explicit LODs raised, gradients scaled (wvk_nir_descriptors.c). */
      .shaderResourceMinLod = true,
      /* Secondary command buffers are replayed into the primary's pass,
       * whose queries they are in. */
      .inheritedQueries = true,
      /* A pass without attachments is begun again with as many samples
       * as a draw's (wvk_cmd_draw.c). */
      .variableMultisampleRate = true,
      /* Every sample, the fragment shader made to read its sample. */
      .sampleRateShading = true,
      /* The fragment shader's alpha, and its coverage by it (wvk_shader.c). */
      .alphaToOne = true,
      /* Lines and points: lists of the triangles' edges and corners
       * (wvk_cmd_draw.c), without culling. */
      .fillModeNonSolid = true,
      /* Every format it names is texture-formats-tier1's storage. */
      .shaderStorageImageExtendedFormats = wvk_physical_device_has_feature(pdev, WGPUFeatureName_TextureFormatsTier1),
      /* A storage image without a format has the bound view's: a variant
       * of the shader for it, made at the draw or dispatch (wvk_shader.c). */
      .shaderStorageImageReadWithoutFormat = true,
      .shaderStorageImageWriteWithoutFormat = true,
      /* 64-bit integers are pairs of words (nir_lower_int64). */
      .shaderInt64 = true,
#ifdef WVK_HAVE_FLOAT64
      /* In integers, of Mesa's float64.glsl (wvk_shader.c). */
      .shaderFloat64 = true,
#endif
      /* 8- and 16-bit integers are words, and so are those in memory,
       * stored by atomics on their bits (wvk_nir_small_types.c). */
      .shaderInt16 = true,
      .shaderClipDistance = clip,
      /* Culled in a geometry shader run as compute, the draw's or one of
       * the driver's (wvk_shader_cull_geometry). */
      .shaderCullDistance = clip,
      /* Arrays of descriptors are bindings of their own, among which a
       * dynamic index chooses in the shader (wvk_shader.c). */
      .shaderUniformBufferArrayDynamicIndexing = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,

      /* Vulkan 1.1. A pass of several views is one for each. */
      .shaderDrawParameters = true,
      .storageBuffer16BitAccess = true,
      .uniformAndStorageBuffer16BitAccess = true,
      .storagePushConstant16 = true,
      .storageInputOutput16 = true,
      .multiview = true,
      /* Multi-planar images, a texture a plane (wvk_image.h), sampled a
       * plane a binding through the conversion (wvk_shader.c). */
      .samplerYcbcrConversion = true,
      /* A buffer a pointer's phi or select chooses: each access one on
       * every buffer it may be, by the index (wvk_shader.c). */
      .variablePointersStorageBuffer = true,
      .variablePointers = true,
      /* Min and max reductions, the shader's (wvk_nir_descriptors.c). */
      .samplerFilterMinmax = true,
      /* What vtn reads and the rest ignores, or subgroups of one have
       * already. */
      .shaderExpectAssume = true,
      .shaderReplicatedComposites = true,
      .shaderSubgroupUniformControlFlow = true,
      .shaderMaximalReconvergence = true,
      .ycbcr2plane444Formats = true,
      /* 8-bit indices widened to 16 by a pass of the driver's
       * (wvk_cmd_draw.c). */
      .indexTypeUint8 = true,
      /* Of the geometry shader's lists, the driver's where the pipeline
       * has none, walked by a pass of its (wvk_cmd_draw.c), each vertex
       * stream's; which the geometry shader has a list of each of. */
      .transformFeedback = true,
      .geometryStreams = true,
      /* Each stream's primitives, counted as pipeline statistics are
       * (wvk_query.c). */
      .primitivesGeneratedQuery = true,
      .primitivesGeneratedQueryWithRasterizerDiscard = true,
      .primitivesGeneratedQueryWithNonZeroStreams = true,
      /* Secondaries are commands of the primary's queue (vk_cmd_queue),
       * however deep. */
      .nestedCommandBuffer = true,
      .nestedCommandBufferRendering = true,
      .nestedCommandBufferSimultaneousUse = true,
      /* Their stages run as compute within each view's replay of the pass,
       * the view index the driver's data. */
      .multiviewGeometryShader = true,
      .multiviewTessellationShader = true,
      /* A draw in each viewport, the vertex shader's gl_ViewportIndex
       * choosing which (wvk_shader.c). */
      .multiViewport = true,

      /* Vulkan 1.2 */
      .samplerMirrorClampToEdge = true,
      /* The draws past the count given no instances (wvk_cmd_draw.c). */
      .drawIndirectCount = true,
      /* A vertex shader's gl_Layer: vertices of other layers leave the
       * clip volume in each layer's pass. */
      .shaderOutputLayer = true,
      .shaderOutputViewportIndex = true,
      /* Descriptor indexing: bind groups are made when the submit is
       * replayed, so updates after binding are what it sees; a binding
       * never written shows a placeholder. */
      /* Each feature of descriptor indexing, but not the extension's set
       * of them: that also asks for 500000 descriptors a stage, and
       * WebGPU has a few dozen bindings. */
      .descriptorIndexing = false,
      .shaderInputAttachmentArrayDynamicIndexing = true,
      .shaderUniformTexelBufferArrayDynamicIndexing = true,
      .shaderStorageTexelBufferArrayDynamicIndexing = true,
      .shaderUniformBufferArrayNonUniformIndexing = true,
      .shaderSampledImageArrayNonUniformIndexing = true,
      .shaderStorageBufferArrayNonUniformIndexing = true,
      .shaderStorageImageArrayNonUniformIndexing = true,
      .shaderInputAttachmentArrayNonUniformIndexing = true,
      .shaderUniformTexelBufferArrayNonUniformIndexing = true,
      .shaderStorageTexelBufferArrayNonUniformIndexing = true,
      .descriptorBindingUniformBufferUpdateAfterBind = true,
      .descriptorBindingSampledImageUpdateAfterBind = true,
      .descriptorBindingStorageImageUpdateAfterBind = true,
      .descriptorBindingStorageBufferUpdateAfterBind = true,
      .descriptorBindingUniformTexelBufferUpdateAfterBind = true,
      .descriptorBindingStorageTexelBufferUpdateAfterBind = true,
      /* Inline uniform blocks: a uniform buffer of the set's, written
       * from the host's copy when the set's bind group is made. */
      .inlineUniformBlock = true,
      .descriptorBindingInlineUniformBlockUpdateAfterBind = true,
      .descriptorBindingUpdateUnusedWhilePending = true,
      .descriptorBindingPartiallyBound = true,
      .descriptorBindingVariableDescriptorCount = true,
      .runtimeDescriptorArray = true,
      .shaderFloat16 = wvk_physical_device_has_feature(pdev, WGPUFeatureName_ShaderF16),
      .shaderInt8 = true,
      .storageBuffer8BitAccess = true,
      .uniformAndStorageBuffer8BitAccess = true,
      .storagePushConstant8 = true,
      .shaderSubgroupExtendedTypes = true,
      /* A subgroup of one reads its own value, whichever id. */
      .subgroupBroadcastDynamicId = true,
      .imagelessFramebuffer = true,
      .uniformBufferStandardLayout = true,
      /* Buffers are words at the offsets SPIR-V gives. */
      .scalarBlockLayout = true,
      .separateDepthStencilLayouts = true,
      .hostQueryReset = true,
      .timelineSemaphore = true,

      /* Vulkan 1.3, or their extensions */
      .dynamicRendering = true,
      .synchronization2 = true,
      .shaderDemoteToHelperInvocation = true,
      /* A demote, and a return (wvk_shader.c). */
      .shaderTerminateInvocation = true,
      .shaderZeroInitializeWorkgroupMemory = true,
      .robustImageAccess = true,
      /* WebGPU is robust whatever a pipeline asks for: its accesses are
       * WebGPU's, which keeps them in bounds. */
      .pipelineRobustness = true,
      /* The SPIR-V's float modes, which NIR has, and WGSL mostly not. */
      .shaderFloatControls2 = true,
      /* The one priority of the one queue. */
      .globalPriorityQuery = true,
      /* Unclipped depth, as depth clamp has (wvk_cmd_draw.c). */
      .depthClipEnable = wvk_physical_device_has_feature(pdev, WGPUFeatureName_DepthClipControl),
      /* Clip space's z made WebGPU's by the viewport transform's depth. */
      .depthClipControl = true,
      /* The transfers of a command buffer of the device's, waited for
       * (wvk_host_copy.c). */
      .hostImageCopy = true,
      .pipelineCreationCacheControl = true,
      .privateData = true,
      /* Subgroups are one invocation, and so are full, and of the size
       * any shader asks for that is possible. */
      .subgroupSizeControl = true,
      .computeFullSubgroups = true,
      .shaderSubgroupRotate = true,
      .shaderSubgroupRotateClustered = true,
      .maintenance4 = true,
      /* Dot products of integers are multiplies and adds. */
      .shaderIntegerDotProduct = true,
      /* WGSL's atomics and barriers, which are what the memory model
       * asks for within a workgroup; wider scopes are the workgroup's. */
      .vulkanMemoryModel = true,
      .vulkanMemoryModelDeviceScope = true,
      .texelBufferAlignment = true,

      /* Pipelines are made at the draw from the dynamic state
       * (wvk_cmd_draw.c): any of it can be dynamic, what shaders are
       * compiled for included -- a draw of other state draws a variant
       * (wvk_shader_draw_variant). WebGPU has no tessellation, and only
       * the state of what is supported is. */
      .extendedDynamicState = true,
      .extendedDynamicState2 = true,
      /* Logic ops are the fragment shader's (wvk_shader.c), and so as
       * dynamic as any state a variant is compiled for. */
      .extendedDynamicState2LogicOp = true,
      /* Tessellation reads both at the draw (wvk_cmd_draw.c). */
      .extendedDynamicState2PatchControlPoints = true,
      .extendedDynamicState3TessellationDomainOrigin = true,
      /* The geometry shader's draw is of the rasterized stream's lists,
       * chosen at the draw. */
      .extendedDynamicState3RasterizationStream = true,
      .vertexInputDynamicState = true,
      .extendedDynamicState3DepthClampEnable = wvk_physical_device_has_feature(pdev, WGPUFeatureName_DepthClipControl),
      .extendedDynamicState3DepthClipEnable = wvk_physical_device_has_feature(pdev, WGPUFeatureName_DepthClipControl),
      /* The depth factors of the viewport transform (wvk_cmd_draw.c). */
      .extendedDynamicState3DepthClipNegativeOneToOne = true,
      .extendedDynamicState3PolygonMode = true,
      .extendedDynamicState3RasterizationSamples = true,
      .extendedDynamicState3SampleMask = true,
      .extendedDynamicState3AlphaToCoverageEnable = true,
      .extendedDynamicState3AlphaToOneEnable = true,
      .extendedDynamicState3ColorBlendEnable = true,
      .extendedDynamicState3ColorBlendEquation = true,
      .extendedDynamicState3ColorWriteMask = true,
      .extendedDynamicState3LogicOpEnable = true,

      /* Samplers' records do what WebGPU's samplers cannot
       * (wvk_nir_descriptors.c); a view's swizzle applies to the border. */
      .customBorderColors = true,
      .customBorderColorWithoutFormat = true,
      .borderColorSwizzle = true,
      .borderColorSwizzleFromImage = true,

      /* VK_KHR_vertex_attribute_divisor */
      .vertexAttributeInstanceRateDivisor = true,
      .vertexAttributeInstanceRateZeroDivisor = true,

      .maintenance5 = true,
      .maintenance6 = true,
      /* The runtime's (vk_pipeline.c): a fast link's stages are compiled
       * apart, which draws link (wvk_shader_draw_variant). */
      .graphicsPipelineLibrary = true,
      /* The runtime's too (vk_shader.c): shaders compiled alone, which a
       * draw links, of state that is all dynamic. */
      .shaderObject = true,
      /* Of the pipeline's colour write masks (wvk_cmd_draw.c). */
      .colorWriteEnable = true,
      /* VK_EXT_4444_formats: packed, as the core's are (wvk_format.c). */
      .formatA4R4G4B4 = true,
      .formatA4B4G4R4 = true,

      /* VK_KHR_portability_subset: what WebGPU leaves out */
      .constantAlphaColorBlendFactors = true,
      .events = true,
      /* Of another texture of the image's, kept in step (wvk_image.h). */
      .imageViewFormatReinterpretation = true,
      .imageViewFormatSwizzle = true, /* by the shader: wvk_nir_descriptors.c */
      .imageView2DOn3DImage = false,
      .multisampleArrayImage = false,
      .mutableComparisonSamplers = true,
      /* The corners of the triangles, as points (wvk_cmd_draw.c). */
      .pointPolygons = true,
      .samplerMipLodBias = true,
      .separateStencilMaskRef = true, /* a draw for each face (wvk_cmd_draw.c) */
      .shaderSampleRateInterpolationFunctions = true, /* by derivatives (wvk_shader.c) */
      /* The tessellator makes both (wvk_tessellator.c). */
      .tessellationIsolines = true,
      .tessellationPointMode = true,
      .triangleFans = true, /* as triangle lists (wvk_cmd_draw.c) */
      .vertexAttributeAccessBeyondStride = true,
   };
}

static void
get_device_properties(const struct wvk_physical_device *pdev,
                      const struct wvk_instance *instance,
                      struct vk_properties *p)
{
   const WGPULimits *l = &pdev->limits;
   const uint32_t max_2d = l->maxTextureDimension2D;
   const bool clip = wvk_physical_device_has_feature(pdev, WGPUFeatureName_ClipDistances);

   VkPhysicalDeviceType type;
   switch (pdev->info.adapterType) {
   case WGPUAdapterType_DiscreteGPU: type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU; break;
   case WGPUAdapterType_IntegratedGPU: type = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU; break;
   case WGPUAdapterType_CPU: type = VK_PHYSICAL_DEVICE_TYPE_CPU; break;
   default: type = VK_PHYSICAL_DEVICE_TYPE_OTHER; break;
   }

   /* The driver keeps two storage buffers per stage for itself: the push
    * binding, with the push constants, its own data and the descriptors'
    * records, and the draw's parameters. Texel buffers are storage
    * buffers too, beyond what Vulkan counts as such. */
   const uint32_t ubos = l->maxUniformBuffersPerShaderStage;
   const uint32_t ssbos = l->maxStorageBuffersPerShaderStage - 2;

   *p = (struct vk_properties){
      .apiVersion = api_version(),
      .driverVersion = vk_get_driver_version(),
      .vendorID = pdev->info.vendorID,
      .deviceID = pdev->info.deviceID,
      .deviceType = type,

      .maxImageDimension1D = l->maxTextureDimension1D,
      .maxImageDimension2D = max_2d,
      .maxImageDimension3D = l->maxTextureDimension3D,
      .maxImageDimensionCube = max_2d,
      .maxImageArrayLayers = l->maxTextureArrayLayers,
      /* Texel buffers are storage buffers with the format decoded in the
       * shader. */
      .maxTexelBufferElements = (uint32_t)MIN2(l->maxStorageBufferBindingSize / 16, 1u << 27),
      .maxUniformBufferRange = (uint32_t)MIN2(l->maxUniformBufferBindingSize, UINT32_MAX),
      .maxStorageBufferRange = (uint32_t)MIN2(l->maxStorageBufferBindingSize, UINT32_MAX),
      .maxPushConstantsSize = 128,
      .maxMemoryAllocationCount = 4096,
      .maxSamplerAllocationCount = 4000,
      /* Buffers and images are separate WebGPU objects. */
      .bufferImageGranularity = 1,
      .sparseAddressSpaceSize = 0,
      .maxBoundDescriptorSets = l->maxBindGroups,
      .maxPerStageDescriptorSamplers = l->maxSamplersPerShaderStage,
      .maxPerStageDescriptorUniformBuffers = ubos,
      .maxPerStageDescriptorStorageBuffers = ssbos,
      .maxPerStageDescriptorSampledImages = l->maxSampledTexturesPerShaderStage,
      .maxPerStageDescriptorStorageImages = l->maxStorageTexturesPerShaderStage,
      .maxPerStageDescriptorInputAttachments = MIN2(l->maxSampledTexturesPerShaderStage, 8),
      .maxPerStageResources = l->maxSamplersPerShaderStage + ubos + ssbos +
                              l->maxSampledTexturesPerShaderStage +
                              l->maxStorageTexturesPerShaderStage + l->maxColorAttachments,
      .maxDescriptorSetSamplers = 6 * l->maxSamplersPerShaderStage,
      .maxDescriptorSetUniformBuffers = 6 * ubos,
      .maxDescriptorSetUniformBuffersDynamic = l->maxDynamicUniformBuffersPerPipelineLayout,
      .maxDescriptorSetStorageBuffers = 6 * ssbos,
      .maxDescriptorSetStorageBuffersDynamic = l->maxDynamicStorageBuffersPerPipelineLayout,
      .maxDescriptorSetSampledImages = 6 * l->maxSampledTexturesPerShaderStage,
      .maxDescriptorSetStorageImages = 6 * l->maxStorageTexturesPerShaderStage,
      .maxDescriptorSetInputAttachments = MIN2(l->maxSampledTexturesPerShaderStage, 8),

      /* Descriptor indexing: WebGPU's limits are the limits after binding
       * too, which is less than Vulkan asks of them. */
      .maxUpdateAfterBindDescriptorsInAllPools = 1u << 20,
      .shaderUniformBufferArrayNonUniformIndexingNative = false,
      .shaderSampledImageArrayNonUniformIndexingNative = false,
      .shaderStorageBufferArrayNonUniformIndexingNative = false,
      .shaderStorageImageArrayNonUniformIndexingNative = false,
      .shaderInputAttachmentArrayNonUniformIndexingNative = false,
      .robustBufferAccessUpdateAfterBind = true,
      .quadDivergentImplicitLod = false,
      .maxPerStageDescriptorUpdateAfterBindSamplers = l->maxSamplersPerShaderStage,
      .maxPerStageDescriptorUpdateAfterBindUniformBuffers = ubos,
      .maxPerStageDescriptorUpdateAfterBindStorageBuffers = ssbos,
      .maxPerStageDescriptorUpdateAfterBindSampledImages = l->maxSampledTexturesPerShaderStage,
      .maxPerStageDescriptorUpdateAfterBindStorageImages = l->maxStorageTexturesPerShaderStage,
      .maxPerStageDescriptorUpdateAfterBindInputAttachments = MIN2(l->maxSampledTexturesPerShaderStage, 8),
      .maxPerStageUpdateAfterBindResources = l->maxSamplersPerShaderStage + ubos + ssbos +
                                             l->maxSampledTexturesPerShaderStage +
                                             l->maxStorageTexturesPerShaderStage + l->maxColorAttachments,
      .maxDescriptorSetUpdateAfterBindSamplers = 6 * l->maxSamplersPerShaderStage,
      .maxDescriptorSetUpdateAfterBindUniformBuffers = 6 * ubos,
      .maxDescriptorSetUpdateAfterBindUniformBuffersDynamic = l->maxDynamicUniformBuffersPerPipelineLayout,
      .maxDescriptorSetUpdateAfterBindStorageBuffers = 6 * ssbos,
      .maxDescriptorSetUpdateAfterBindStorageBuffersDynamic = l->maxDynamicStorageBuffersPerPipelineLayout,
      .maxDescriptorSetUpdateAfterBindSampledImages = 6 * l->maxSampledTexturesPerShaderStage,
      .maxDescriptorSetUpdateAfterBindStorageImages = 6 * l->maxStorageTexturesPerShaderStage,
      .maxDescriptorSetUpdateAfterBindInputAttachments = MIN2(l->maxSampledTexturesPerShaderStage, 8),
      .maxVertexInputAttributes = l->maxVertexAttributes,
      /* Binding numbers up to Vulkan's minimum, onto WebGPU's slots: a
       * pipeline can use as many bindings as WebGPU has buffers. */
      .maxVertexInputBindings = MAX2(l->maxVertexBuffers, 16),
      .maxVertexInputAttributeOffset = l->maxVertexBufferArrayStride - 1,
      .maxVertexInputBindingStride = l->maxVertexBufferArrayStride,
      .maxVertexOutputComponents = l->maxInterStageShaderVariables * 4,
      /* A geometry shader is compute, its records a buffer's (wvk_shader.c);
       * its outputs are the draw's vertex shader's after it. */
      .maxGeometryShaderInvocations = 127,
      .maxGeometryInputComponents = 128,
      .maxGeometryOutputComponents = l->maxInterStageShaderVariables * 4,
      .maxGeometryOutputVertices = 256,
      .maxGeometryTotalOutputComponents = 1024,
      /* Tessellation's stages are compute too, and the tessellator the
       * D3D11 reference's (wvk_tessellator.c). */
      .maxTessellationGenerationLevel = 64,
      .maxTessellationPatchSize = 32,
      .maxTessellationControlPerVertexInputComponents = 128,
      .maxTessellationControlPerVertexOutputComponents = 128,
      .maxTessellationControlPerPatchOutputComponents = 120,
      .maxTessellationControlTotalOutputComponents = 4096,
      .maxTessellationEvaluationInputComponents = 128,
      .maxTessellationEvaluationOutputComponents = l->maxInterStageShaderVariables * 4,
      .maxFragmentInputComponents = l->maxInterStageShaderVariables * 4,
      .maxFragmentOutputAttachments = l->maxColorAttachments,
      .maxFragmentDualSrcAttachments =
         wvk_physical_device_has_feature(pdev, WGPUFeatureName_DualSourceBlending) ? 1 : 0,
      .maxFragmentCombinedOutputResources = l->maxColorAttachments + ssbos +
                                            l->maxStorageTexturesPerShaderStage,
      .maxComputeSharedMemorySize = l->maxComputeWorkgroupStorageSize,
      .maxComputeWorkGroupCount = { l->maxComputeWorkgroupsPerDimension,
                                    l->maxComputeWorkgroupsPerDimension,
                                    l->maxComputeWorkgroupsPerDimension },
      .maxComputeWorkGroupInvocations = l->maxComputeInvocationsPerWorkgroup,
      .maxComputeWorkGroupSize = { l->maxComputeWorkgroupSizeX, l->maxComputeWorkgroupSizeY,
                                   l->maxComputeWorkgroupSizeZ },
      .subPixelPrecisionBits = 4,
      .subTexelPrecisionBits = 4,
      .mipmapPrecisionBits = 4,
      .maxDrawIndexedIndexValue = UINT32_MAX,
      .maxDrawIndirectCount = UINT32_MAX,
      /* VK_EXT_multi_draw */
      .maxMultiDrawCount = 2048,
      /* WebGPU samplers have no LOD bias. */
      /* Vulkan's minimum. WebGPU samplers have no bias: it is ignored. */
      .maxSamplerLodBias = WVK_MAX_SAMPLER_LOD_BIAS,
      /* VK_EXT_transform_feedback: NIR's streams and buffers. A vertex's
       * outputs are written at their offsets, however far apart, to the
       * stride vertex input has. */
      .maxTransformFeedbackStreams = 4,
      .maxTransformFeedbackBuffers = 4,
      .maxTransformFeedbackBufferSize = UINT32_MAX,
      .maxTransformFeedbackStreamDataSize = 2048,
      .maxTransformFeedbackBufferDataSize = 2048,
      .maxTransformFeedbackBufferDataStride = 2048,
      .transformFeedbackQueries = true,
      .transformFeedbackStreamsLinesTriangles = true,
      .transformFeedbackRasterizationStreamSelect = true,
      .transformFeedbackDraw = true,
      /* VK_EXT_pipeline_robustness: WebGPU's own, of buffers in bounds
       * and images of VK_EXT_image_robustness. */
      .defaultRobustnessStorageBuffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
      .defaultRobustnessUniformBuffers = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
      .defaultRobustnessVertexInputs = VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_ROBUST_BUFFER_ACCESS,
      .defaultRobustnessImages = VK_PIPELINE_ROBUSTNESS_IMAGE_BEHAVIOR_ROBUST_IMAGE_ACCESS,
      /* VK_EXT_nested_command_buffer */
      .maxCommandBufferNestingLevel = UINT32_MAX,
      .maxSamplerAnisotropy = 16.0f,
      .maxViewports = WVK_MAX_VIEWPORTS,
      .maxViewportDimensions = { max_2d, max_2d },
      .viewportBoundsRange = { -2.0f * max_2d, 2.0f * max_2d - 1.0f },
      .viewportSubPixelBits = 0,
      .minMemoryMapAlignment = 64,
      .minTexelBufferOffsetAlignment = l->minStorageBufferOffsetAlignment,
      .minUniformBufferOffsetAlignment = l->minUniformBufferOffsetAlignment,
      .minStorageBufferOffsetAlignment = l->minStorageBufferOffsetAlignment,
      /* WGSL's constant texel offsets. */
      .minTexelOffset = -8,
      .maxTexelOffset = 7,
      .minTexelGatherOffset = -8,
      .maxTexelGatherOffset = 7,
      .minInterpolationOffset = -0.5f,
      .maxInterpolationOffset = 0.4375f,
      .subPixelInterpolationOffsetBits = 4,
      .maxFramebufferWidth = max_2d,
      .maxFramebufferHeight = max_2d,
      /* A pass of several layers is one for each (wvk_cmd_buffer.c). */
      .maxFramebufferLayers = l->maxTextureArrayLayers,
      .framebufferColorSampleCounts = WVK_SAMPLE_COUNTS,
      .framebufferDepthSampleCounts = WVK_SAMPLE_COUNTS,
      .framebufferStencilSampleCounts = WVK_SAMPLE_COUNTS,
      .framebufferNoAttachmentsSampleCounts = WVK_SAMPLE_COUNTS,
      .maxColorAttachments = l->maxColorAttachments,
      .sampledImageColorSampleCounts = WVK_SAMPLE_COUNTS,
      /* WebGPU multisamples few integer formats. */
      .sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .sampledImageDepthSampleCounts = WVK_SAMPLE_COUNTS,
      .sampledImageStencilSampleCounts = WVK_SAMPLE_COUNTS,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .maxSampleMaskWords = 1,
      /* An empty compute pass's beginning, anywhere (wvk_query.c). */
      .timestampComputeAndGraphics = wvk_physical_device_has_feature(pdev, WGPUFeatureName_TimestampQuery),
      .timestampPeriod = 1.0f,
      .maxClipDistances = clip ? 8 : 0,
      .maxCullDistances = clip ? 8 : 0,
      .maxCombinedClipAndCullDistances = clip ? 8 : 0,
      .discreteQueuePriorities = 2,
      /* Points are quads of any size (expand_points in wvk_shader.c). */
      .pointSizeRange = { 1.0f, WVK_MAX_POINT_SIZE },
      /* And lines quads of any width, a parallelogram across the minor
       * axis (expand_lines). */
      .lineWidthRange = { 1.0f, WVK_MAX_LINE_WIDTH },
      .pointSizeGranularity = 0.125f,
      .lineWidthGranularity = 0.125f,
      .strictLines = false,
      .standardSampleLocations = true,
      /* WebGPU's own copy rules: rows of 256 bytes. */
      .optimalBufferCopyOffsetAlignment = 256,
      .optimalBufferCopyRowPitchAlignment = 256,
      .nonCoherentAtomSize = 64,

      /* Vulkan 1.1 */
      /* Every invocation is a subgroup of its own (wvk_nir_subgroups.c):
       * Tint reads no subgroup operations. Quads would need four. */
      .subgroupSize = 1,
      .subgroupSupportedStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT |
                                 VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                                 VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
      .subgroupSupportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
                                     VK_SUBGROUP_FEATURE_ARITHMETIC_BIT | VK_SUBGROUP_FEATURE_BALLOT_BIT |
                                     VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT |
                                     VK_SUBGROUP_FEATURE_CLUSTERED_BIT | VK_SUBGROUP_FEATURE_ROTATE_BIT |
                                     VK_SUBGROUP_FEATURE_ROTATE_CLUSTERED_BIT,
      .subgroupQuadOperationsInAllStages = false,
      .pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_ALL_CLIP_PLANES,
      .maxMultiviewViewCount = 8,
      .maxMultiviewInstanceIndex = UINT32_MAX / 8,
      .protectedNoFault = false,
      .maxPerSetDescriptors = 1024,
      .maxMemoryAllocationSize = l->maxBufferSize,

      /* Vulkan 1.2 */
      .driverID = VK_DRIVER_ID_MESA_VENUS,
      .conformanceVersion = { 0, 0, 0, 0 },
      .denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
      .roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_NONE,
      .shaderSignedZeroInfNanPreserveFloat32 = false,
      .maxUpdateAfterBindDescriptorsInAllPools = 0,
      .supportedDepthResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .independentResolveNone = true,
      .independentResolve = true,
      .filterMinmaxSingleComponentFormats = true,
      .filterMinmaxImageComponentMapping = true,
      .maxTimelineSemaphoreValueDifference = UINT64_MAX,
      .framebufferIntegerColorSampleCounts = VK_SAMPLE_COUNT_1_BIT,

      /* Vulkan 1.3. Inline uniform blocks are uniform buffers, of those
       * WebGPU has for each stage. */
      .maxInlineUniformBlockSize = 4096,
      .maxPerStageDescriptorInlineUniformBlocks = 4,
      .maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = 4,
      .maxDescriptorSetInlineUniformBlocks = 4,
      .maxDescriptorSetUpdateAfterBindInlineUniformBlocks = 4,
      .maxInlineUniformTotalSize = 4 * 4096,
      .minSubgroupSize = 1,
      .maxSubgroupSize = 1,
      .maxComputeWorkgroupSubgroups = l->maxComputeInvocationsPerWorkgroup,
      .requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT,
      .maxBufferSize = l->maxBufferSize,
      /* A texel buffer starts anywhere: its record says where. */
      .storageTexelBufferOffsetAlignmentBytes = 16,
      .storageTexelBufferOffsetSingleTexelAlignment = true,
      .uniformTexelBufferOffsetAlignmentBytes = 16,
      .uniformTexelBufferOffsetSingleTexelAlignment = true,

      /* VK_KHR_push_descriptor */
      .maxPushDescriptors = 32,

      /* VK_EXT_custom_border_color */
      .maxCustomBorderColorSamplers = 4000,

      /* VK_KHR_vertex_attribute_divisor */
      .maxVertexAttribDivisor = UINT32_MAX,
      .supportsNonZeroFirstInstance = true,

      /* VK_KHR_portability_subset */
      .minVertexInputBindingStrideAlignment = 4,

      /* VK_KHR_maintenance5. A polygon's corners in point mode are a
       * pixel, whatever PointSize says; a depth or stencil view's missing
       * channels are 0, and its alpha 1 (wvk_image.c). */
      .earlyFragmentMultisampleCoverageAfterSampleCounting = true,
      .earlyFragmentSampleMaskTestBeforeSampleCounting = true,
      .depthStencilSwizzleOneSupport = true,
      .polygonModePointSize = false,
      .nonStrictSinglePixelWideLinesUseParallelogram = false,
      .nonStrictWideLinesUseParallelogram = true,

      /* VK_EXT_extended_dynamic_state3: a pipeline is of its draw's
       * topology, whichever it is. */
      .dynamicPrimitiveTopologyUnrestricted = true,

      /* VK_EXT_graphics_pipeline_library: a vertex shader of other
       * interpolation than its fragment shader's is a variant of it. */
      .graphicsPipelineLibraryFastLinking = true,
      .graphicsPipelineLibraryIndependentInterpolationDecoration = true,

      /* VK_KHR_maintenance6 */
      .blockTexelViewCompatibleMultipleLayers = false,
      .maxCombinedImageSamplerDescriptorCount = 3,
      .fragmentShadingRateClampCombinerInputs = false,
   };

   /* "wink on Apple M2 (Metal)": what the device is, and what it is on. */
   const char *backend;
   switch (pdev->info.backendType) {
   case WGPUBackendType_Metal: backend = "Metal"; break;
   case WGPUBackendType_Vulkan: backend = "Vulkan"; break;
   case WGPUBackendType_D3D12: backend = "D3D12"; break;
   case WGPUBackendType_D3D11: backend = "D3D11"; break;
   case WGPUBackendType_OpenGL: backend = "OpenGL"; break;
   case WGPUBackendType_OpenGLES: backend = "OpenGL ES"; break;
   case WGPUBackendType_WebGPU: backend = "browser"; break;
   default: backend = "WebGPU"; break;
   }
   /* A browser keeps the device's name to itself, and says at most the
    * vendor and the architecture. */
   const WGPUAdapterInfo *info = &pdev->info;
   char what[128];
   if (wvk_sv_len(info->device) || wvk_sv_len(info->description)) {
      const WGPUStringView name = wvk_sv_len(info->device) ? info->device : info->description;
      snprintf(what, sizeof(what), "%.*s", wvk_sv_len(name), name.data);
   } else if (wvk_sv_len(info->vendor)) {
      snprintf(what, sizeof(what), "%.*s%s%.*s", wvk_sv_len(info->vendor), info->vendor.data,
               wvk_sv_len(info->architecture) ? " " : "", wvk_sv_len(info->architecture),
               info->architecture.data ? info->architecture.data : "");
   } else {
      snprintf(what, sizeof(what), "WebGPU");
   }
   snprintf(p->deviceName, sizeof(p->deviceName), "wink on %s (%s)", what, backend);

   memcpy(p->pipelineCacheUUID, instance->driver_uuid, VK_UUID_SIZE);
   memcpy(p->driverUUID, instance->driver_uuid, VK_UUID_SIZE);
   /* Shaders are cached as WGSL, which is this build's. */
   memcpy(p->shaderBinaryUUID, instance->driver_uuid, VK_UUID_SIZE);
   p->shaderBinaryVersion = WVK_SHADER_BINARY_VERSION;
   /* The device: the adapter's IDs, and which backend it is reached by. */
   uint32_t uuid[4] = { pdev->info.vendorID, pdev->info.deviceID, (uint32_t)pdev->info.backendType,
                        0x77696e6b /* "wink" */ };
   memcpy(p->deviceUUID, uuid, VK_UUID_SIZE);
   /* VK_EXT_host_image_copy: any layout, which WebGPU has none of; a
    * memcpy copy's memory is a buffer copy's, which is this build's. */
   static const VkImageLayout host_copy_layouts[] = {
      VK_IMAGE_LAYOUT_GENERAL,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      VK_IMAGE_LAYOUT_PREINITIALIZED,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL,
      VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
      VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
   };
   p->pCopySrcLayouts = (VkImageLayout *)host_copy_layouts;
   p->copySrcLayoutCount = ARRAY_SIZE(host_copy_layouts);
   p->pCopyDstLayouts = (VkImageLayout *)host_copy_layouts;
   p->copyDstLayoutCount = ARRAY_SIZE(host_copy_layouts);
   memcpy(p->optimalTilingLayoutUUID, instance->driver_uuid, VK_UUID_SIZE);
   p->identicalMemoryTypeRequirements = true;
   snprintf(p->driverName, VK_MAX_DRIVER_NAME_SIZE, "wink");
   snprintf(p->driverInfo, VK_MAX_DRIVER_INFO_SIZE, "Mesa " PACKAGE_VERSION MESA_GIT_SHA1
            ", Vulkan on WebGPU (%s)", backend);
}

static void
init_memory(struct wvk_physical_device *pdev)
{
   /* WebGPU says nothing about memory size. In a browser, host-visible
    * memory is a copy in the wasm heap, which a 32-bit address space
    * bounds; natively, report what a mid-range GPU has. */
#ifdef __EMSCRIPTEN__
   pdev->heap_size = 1ull << 30;
#else
   pdev->heap_size = 4ull << 30;
#endif
   /* Ordered as Vulkan requires: a type before any whose flags are a
    * superset of its own. */
   pdev->memory_types[WVK_MEMORY_DEVICE] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
   };
   pdev->memory_types[WVK_MEMORY_UPLOAD] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
      .heapIndex = 1,
   };
   pdev->memory_types[WVK_MEMORY_READBACK] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                       VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
      .heapIndex = 1,
   };
   pdev->memory_types[WVK_MEMORY_DEVICE_UPLOAD] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
   };
}

VkResult
wvk_enumerate_physical_devices(struct vk_instance *_instance)
{
   struct wvk_instance *instance = (struct wvk_instance *)_instance;

   if (!wvk_instance_wgpu(instance))
      return VK_SUCCESS; /* no WebGPU: no devices, not an error */
   WGPUAdapter adapter = wvk_platform_request_adapter(instance->wgpu);
   if (!adapter)
      return VK_SUCCESS;

   struct wvk_physical_device *pdev =
      vk_zalloc(&instance->vk.alloc, sizeof(*pdev), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!pdev) {
      wgpuAdapterRelease(adapter);
      return vk_error(instance, VK_ERROR_OUT_OF_HOST_MEMORY);
   }
   pdev->adapter = adapter;
   pdev->info = (WGPUAdapterInfo)WGPU_ADAPTER_INFO_INIT;
   wgpuAdapterGetInfo(adapter, &pdev->info);
   pdev->limits = (WGPULimits)WGPU_LIMITS_INIT;
   if (wgpuAdapterGetLimits(adapter, &pdev->limits) != WGPUStatus_Success) {
      mesa_logw("wvk: the adapter reports no limits");
      pdev->limits = (WGPULimits)WGPU_LIMITS_INIT;
   }
   for (size_t i = 0; i < ARRAY_SIZE(wanted_features); ++i) {
      if (wgpuAdapterHasFeature(adapter, wanted_features[i]) &&
          pdev->feature_count < ARRAY_SIZE(pdev->features))
         pdev->features[pdev->feature_count++] = wanted_features[i];
   }

   struct vk_physical_device_dispatch_table dispatch_table;
   vk_physical_device_dispatch_table_from_entrypoints(&dispatch_table,
                                                      &wvk_physical_device_entrypoints, true);

   struct vk_device_extension_table extensions;
   get_device_extensions(pdev, &extensions);
   struct vk_features features;
   get_device_features(pdev, &features);
   struct vk_properties properties;
   get_device_properties(pdev, instance, &properties);
   wvk_init_nir_options(&pdev->nir_options);
   simple_mtx_init(&pdev->float64_lock, mtx_plain);

   VkResult result = vk_physical_device_init(&pdev->vk, &instance->vk, &extensions,
                                             &features, &properties, &dispatch_table);
   if (result != VK_SUCCESS) {
      wgpuAdapterInfoFreeMembers(pdev->info);
      wgpuAdapterRelease(adapter);
      vk_free(&instance->vk.alloc, pdev);
      return result;
   }

   init_memory(pdev);

   pdev->sync_timeline_type = vk_sync_timeline_get_type(&wvk_sync_type);
   pdev->sync_types[0] = &wvk_sync_type;
   pdev->sync_types[1] = &pdev->sync_timeline_type.sync;
   pdev->sync_types[2] = NULL;
   pdev->vk.supported_sync_types = pdev->sync_types;

   list_addtail(&pdev->vk.link, &instance->vk.physical_devices.list);
   return VK_SUCCESS;
}

void
wvk_physical_device_destroy(struct vk_physical_device *vk_pdev)
{
   struct wvk_physical_device *pdev = container_of(vk_pdev, struct wvk_physical_device, vk);
   ralloc_free(pdev->float64);
   simple_mtx_destroy(&pdev->float64_lock);
   vk_physical_device_finish(&pdev->vk);
   wgpuAdapterInfoFreeMembers(pdev->info);
   wgpuAdapterRelease(pdev->adapter);
   vk_free(&pdev->vk.instance->alloc, pdev);
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physicalDevice,
                                            uint32_t *pQueueFamilyPropertyCount,
                                            VkQueueFamilyProperties2 *pQueueFamilyProperties)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, pQueueFamilyProperties,
                          pQueueFamilyPropertyCount);

   /* One WebGPU queue, which does everything, in order; and a family of
    * compute and transfer, whose queue is that one too (wvk_queue.h). */
   static const VkQueueFlags flags[WVK_QUEUE_FAMILIES] = {
      [WVK_QUEUE_FAMILY_ALL] = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
      [WVK_QUEUE_FAMILY_COMPUTE] = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT,
   };
   for (uint32_t f = 0; f < WVK_QUEUE_FAMILIES; ++f) {
      vk_outarray_append_typed(VkQueueFamilyProperties2, &out, p) {
         p->queueFamilyProperties = (VkQueueFamilyProperties){
            .queueFlags = flags[f],
            .queueCount = 1,
            .timestampValidBits = wvk_physical_device_has_feature(pdev, WGPUFeatureName_TimestampQuery) ? 64 : 0,
            .minImageTransferGranularity = { 1, 1, 1 },
         };
         vk_foreach_struct(stype, ext, p->pNext) {
            switch (stype) {
            case VK_STRUCTURE_TYPE_QUEUE_FAMILY_GLOBAL_PRIORITY_PROPERTIES_KHR: {
               VkQueueFamilyGlobalPriorityPropertiesKHR *prio = ext;
               prio->priorityCount = 1;
               prio->priorities[0] = VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
               break;
            }
            default:
               vk_debug_ignored_stype(stype);
               break;
            }
         }
      }
   }
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
                                       VkPhysicalDeviceMemoryProperties2 *pMemoryProperties)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);
   VkPhysicalDeviceMemoryProperties *props = &pMemoryProperties->memoryProperties;

   /* Device-local types in a device-local heap, as Vulkan insists; the
    * host-visible ones that are not, in a heap of their own. */
   props->memoryHeapCount = 2;
   props->memoryHeaps[0] = (VkMemoryHeap){
      .size = pdev->heap_size,
      .flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT,
   };
   props->memoryHeaps[1] = (VkMemoryHeap){
      .size = pdev->heap_size / 2,
   };
   props->memoryTypeCount = WVK_MEMORY_TYPE_COUNT;
   for (uint32_t i = 0; i < WVK_MEMORY_TYPE_COUNT; ++i)
      props->memoryTypes[i] = pdev->memory_types[i];

   vk_foreach_struct(stype, ext, pMemoryProperties->pNext) {
      vk_debug_ignored_stype(stype);
   }
}

/* ---------------------------------------------------------------------- */
/* Formats */

static VkFormatFeatureFlags2
optimal_features(const struct wvk_physical_device *pdev, VkFormat format)
{
   const uint32_t caps = wvk_format_caps(pdev, format);
   VkFormatFeatureFlags2 f = 0;
   if (!caps)
      return 0;
   /* Multi-planar: its planes copied, and sampled through a conversion
    * (nir_vk_lower_ycbcr_tex), a texture a plane. */
   if (caps & WVK_CAP_PLANAR)
      return VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
             VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
             VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT | VK_FORMAT_FEATURE_2_MIDPOINT_CHROMA_SAMPLES_BIT | VK_FORMAT_FEATURE_2_COSITED_CHROMA_SAMPLES_BIT |
             VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT |
             VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_YCBCR_CONVERSION_SEPARATE_RECONSTRUCTION_FILTER_BIT |
             VK_FORMAT_FEATURE_2_DISJOINT_BIT;

   if (caps & WVK_CAP_SAMPLE) {
      f |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_2_BLIT_SRC_BIT;
      if (caps & WVK_CAP_FILTER)
         f |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
      /* Min and max, of float and depth formats: fetched and compared. */
      if (!(caps & WVK_CAP_INT))
         f |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT;
      if (caps & WVK_CAP_DEPTH)
         f |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
   }
   if (caps & WVK_CAP_RENDER) {
      f |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_BLIT_DST_BIT;
      if (caps & WVK_CAP_BLEND)
         f |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT;
   }
   if (caps & (WVK_CAP_DEPTH | WVK_CAP_STENCIL))
      f |= VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
   if (caps & WVK_CAP_STORAGE) {
      f |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
      /* Without a format too: a variant of the shader has the view's
       * (wvk_shader.c), which reads it where WebGPU reads and writes the
       * format -- and where the shader only reads, it is a texture. */
      f |= VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
      if (caps & WVK_CAP_STORAGE_RW)
         f |= VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT;
      /* Through the image's buffer of its texels (wvk_image.h). */
      if (wvk_format_has_atomics(format))
         f |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT;
   }
   /* Transfers work on every format: the ones WebGPU will not copy --
    * Depth24Plus at all, Depth32Float into -- go through a shader. And so
    * do the host's, which are those transfers (wvk_host_copy.c). */
   f |= VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
        VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT;
   return f;
}

static VkFormatFeatureFlags2
buffer_features(const struct wvk_physical_device *pdev, VkFormat format)
{
   const struct wvk_format *f = wvk_format_get(format);
   (void)pdev;
   VkFormatFeatureFlags2 features = 0;
   /* WebGPU's vertex formats, and those the vertex shader reads itself. */
   if ((f && f->vertex) || wvk_format_vertex_pullable(format))
      features |= VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT;
   /* Texel buffers are storage buffers, and any format the shader can
    * take apart and put together will do. */
   uint32_t record[WVK_META_WORDS];
   if (wvk_texel_buffer_format(format, record)) {
      features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT | VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
      if (format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT)
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   }
   return features;
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice, VkFormat format,
                                       VkFormatProperties2 *pFormatProperties)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);

   const VkFormatFeatureFlags2 optimal = optimal_features(pdev, format);
   const VkFormatFeatureFlags2 buffer = buffer_features(pdev, format);
   /* WebGPU has no linear textures: a linear image is memory, and only
    * copied to and from. */
   VkFormatFeatureFlags2 linear =
      wvk_image_linear_supported(pdev, format, VK_IMAGE_TYPE_2D, 0)
         ? VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
              VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT
         : 0;
   if (linear && (wvk_format_caps(pdev, format) & WVK_CAP_PLANAR))
      linear |= VK_FORMAT_FEATURE_2_DISJOINT_BIT;

   pFormatProperties->formatProperties = (VkFormatProperties){
      .linearTilingFeatures = vk_format_features2_to_features(linear),
      .optimalTilingFeatures = vk_format_features2_to_features(optimal),
      .bufferFeatures = vk_format_features2_to_features(buffer),
   };

   vk_foreach_struct(stype, ext, pFormatProperties->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3: {
         VkFormatProperties3 *p = ext;
         p->linearTilingFeatures = linear;
         p->optimalTilingFeatures = optimal;
         p->bufferFeatures = buffer;
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice physicalDevice,
                                            const VkPhysicalDeviceImageFormatInfo2 *info,
                                            VkImageFormatProperties2 *pImageFormatProperties)
{
   VK_FROM_HANDLE(wvk_physical_device, pdev, physicalDevice);
   const WGPULimits *l = &pdev->limits;

   pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){ 0 };

   const uint32_t caps = wvk_format_caps(pdev, info->format);
   if (!caps)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if (info->tiling == VK_IMAGE_TILING_LINEAR) {
      /* A linear image is memory copied to and from, and never has a view,
       * so being allowed views of other formats changes nothing about it. */
      if (!wvk_image_linear_supported(pdev, info->format, info->type, info->usage) ||
          (info->flags & ~(VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT |
                           ((caps & WVK_CAP_PLANAR) ? VK_IMAGE_CREATE_DISJOINT_BIT : 0))))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
         .maxExtent = { l->maxTextureDimension2D, l->maxTextureDimension2D, 1 },
         .maxMipLevels = 1,
         .maxArrayLayers = 1,
         .sampleCounts = VK_SAMPLE_COUNT_1_BIT,
         .maxResourceSize = 1ull << 31,
      };
      return VK_SUCCESS;
   }
   if (info->tiling != VK_IMAGE_TILING_OPTIMAL)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   VkImageUsageFlags usage = info->usage;
   const VkImageStencilUsageCreateInfo *stencil_usage =
      vk_find_struct_const(info->pNext, IMAGE_STENCIL_USAGE_CREATE_INFO);
   if (stencil_usage)
      usage |= stencil_usage->stencilUsage;

   if ((usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) &&
       !(caps & WVK_CAP_SAMPLE))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   /* An input attachment is an attachment too. */
   if ((usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) &&
       !(caps & (WVK_CAP_RENDER | WVK_CAP_DEPTH | WVK_CAP_STENCIL)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   /* A multi-planar image's planes may be storage, viewed as formats
    * that are, where it says so. */
   const bool plane_storage = (caps & WVK_CAP_PLANAR) && (info->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
   if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !(caps & WVK_CAP_STORAGE) && !plane_storage)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && !(caps & WVK_CAP_RENDER))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if ((usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
       !(caps & (WVK_CAP_DEPTH | WVK_CAP_STENCIL)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   if (info->flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT |
                      VK_IMAGE_CREATE_SPARSE_ALIASED_BIT))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   VkExtent3D extent;
   uint32_t layers;
   switch (info->type) {
   case VK_IMAGE_TYPE_1D:
      /* A 1D image is a 2D texture one texel high, so it can have mips. */
      extent = (VkExtent3D){ l->maxTextureDimension1D, 1, 1 };
      layers = l->maxTextureArrayLayers;
      if (caps & (WVK_CAP_DEPTH | WVK_CAP_STENCIL | WVK_CAP_COMPRESSED))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      break;
   case VK_IMAGE_TYPE_2D:
      extent = (VkExtent3D){ l->maxTextureDimension2D, l->maxTextureDimension2D, 1 };
      layers = l->maxTextureArrayLayers;
      break;
   case VK_IMAGE_TYPE_3D:
      extent = (VkExtent3D){ l->maxTextureDimension3D, l->maxTextureDimension3D,
                             l->maxTextureDimension3D };
      layers = 1;
      if (caps & (WVK_CAP_DEPTH | WVK_CAP_STENCIL))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      /* Compressed 3D needs the sliced-3D features. */
      if (caps & WVK_CAP_COMPRESSED) {
         const struct wvk_format *f = wvk_format_get(info->format);
         const WGPUFeatureName sliced =
            f->feature == WGPUFeatureName_TextureCompressionBC ? WGPUFeatureName_TextureCompressionBCSliced3D :
            f->feature == WGPUFeatureName_TextureCompressionASTC ? WGPUFeatureName_TextureCompressionASTCSliced3D :
            WGPUFeatureName_Force32;
         if (!wvk_physical_device_has_feature(pdev, sliced))
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }
      break;
   default:
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }

   /* A multi-planar image: 2D, of one level and layer, not disjoint but
    * where the format says it may be. */
   if (caps & WVK_CAP_PLANAR) {
      if (info->type != VK_IMAGE_TYPE_2D ||
          (usage & ~(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_HOST_TRANSFER_BIT |
                     (plane_storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0))))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
         .maxExtent = extent,
         .maxMipLevels = 1,
         .maxArrayLayers = 1,
         .sampleCounts = VK_SAMPLE_COUNT_1_BIT,
         .maxResourceSize = UINT32_MAX,
      };
   } else if (info->flags & VK_IMAGE_CREATE_DISJOINT_BIT) {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }
   VkSampleCountFlags samples = VK_SAMPLE_COUNT_1_BIT;
   if (info->type == VK_IMAGE_TYPE_2D && (caps & WVK_CAP_MSAA) &&
       !(info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
       !(usage & VK_IMAGE_USAGE_STORAGE_BIT))
      samples = WVK_SAMPLE_COUNTS;

   if (!(caps & WVK_CAP_PLANAR)) {
      pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
         .maxExtent = extent,
         .maxMipLevels = util_logbase2(MAX3(extent.width, extent.height, extent.depth)) + 1,
         .maxArrayLayers = layers,
         .sampleCounts = samples,
         .maxResourceSize = UINT32_MAX,
      };
   }

   vk_foreach_struct(stype, ext, pImageFormatProperties->pNext) {
      switch (stype) {
      case VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES: {
         /* A texture a plane, in the bindings a descriptor of it takes. */
         VkSamplerYcbcrConversionImageFormatProperties *p = ext;
         p->combinedImageSamplerDescriptorCount = vk_format_get_plane_count(info->format);
         break;
      }
      case VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY: {
         /* The texture is the same with host transfers or without. */
         VkHostImageCopyDevicePerformanceQuery *p = ext;
         p->optimalDeviceAccess = true;
         p->identicalMemoryLayout = true;
         break;
      }
      case VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES: {
         VkExternalImageFormatProperties *p = ext;
         p->externalMemoryProperties = (VkExternalMemoryProperties){ 0 };
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }

   const VkPhysicalDeviceExternalImageFormatInfo *external =
      vk_find_struct_const(info->pNext, PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
   if (external && external->handleType)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetPhysicalDeviceSparseImageFormatProperties2(VkPhysicalDevice physicalDevice,
                                                  const VkPhysicalDeviceSparseImageFormatInfo2 *pFormatInfo,
                                                  uint32_t *pPropertyCount,
                                                  VkSparseImageFormatProperties2 *pProperties)
{
   *pPropertyCount = 0;
}

VKAPI_ATTR void VKAPI_CALL
wvk_GetPhysicalDeviceExternalBufferProperties(VkPhysicalDevice physicalDevice,
                                              const VkPhysicalDeviceExternalBufferInfo *pExternalBufferInfo,
                                              VkExternalBufferProperties *pExternalBufferProperties)
{
   pExternalBufferProperties->externalMemoryProperties = (VkExternalMemoryProperties){ 0 };
}
