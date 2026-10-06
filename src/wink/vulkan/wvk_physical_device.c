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
#include "wvk_device_memory.h"

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
#include "vk_shader_module.h"

#include "git_sha1.h"
#include "util/u_atomic.h"
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
   /* gl_PrimitiveID of a fragment shader without a geometry shader. */
   WGPUFeatureName_PrimitiveIndex,
#ifndef __EMSCRIPTEN__
   /* Attachments read as they are: tile images, and input attachments
    * in rasterization order (wvk_nir_fetch.c). Dawn's; no browser has it. */
   WGPUFeatureName_FramebufferFetch,
#endif
   /* More storage formats, and the 16-bit normalized ones (wvk_format.c). */
   WGPUFeatureName_TextureFormatsTier1,
   WGPUFeatureName_TextureFormatsTier2,
#ifndef __EMSCRIPTEN__
   /* Strict math, for what is not to be contracted (wvk_shader.c). */
   WGPUFeatureName_ShaderModuleCompilationOptions,
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

/* Whether the device can run mesh and task shaders as compute: of
 * VK_EXT_mesh_shader's least workgroups and shared memory. */
static bool
wvk_physical_device_meshes(const struct wvk_physical_device *pdev)
{
   const WGPULimits *l = &pdev->limits;
   return l->maxComputeWorkgroupStorageSize >= 32768 && l->maxComputeInvocationsPerWorkgroup >= 128 &&
          l->maxComputeWorkgroupSizeX >= 128 && l->maxComputeWorkgroupSizeY >= 128 && l->maxComputeWorkgroupSizeZ >= 128;
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
      .KHR_maintenance7 = true,
      .KHR_maintenance8 = true,
      .KHR_maintenance9 = true,
      .KHR_maintenance10 = true,
      .KHR_cooperative_matrix = true,
      .KHR_shader_bfloat16 = true,
      .EXT_shader_float8 = true,
      .KHR_pipeline_executable_properties = true,
      .EXT_depth_clamp_zero_one = true,
      .KHR_depth_clamp_zero_one = true,
      .KHR_workgroup_memory_explicit_layout = true,
      .EXT_vertex_attribute_robustness = true,
      .EXT_shader_module_identifier = true,
      .EXT_image_view_min_lod = true,
      .EXT_non_seamless_cube_map = true,
      .EXT_image_2d_view_of_3d = true,
      .EXT_image_sliced_view_of_3d = true,
      .EXT_blend_operation_advanced = true,
      .KHR_variable_pointers = true,
      .KHR_external_memory = true,
      .KHR_external_fence = true,
      .KHR_external_semaphore = true,
      .EXT_queue_family_foreign = true,
      .EXT_shader_subgroup_ballot = true,
      .EXT_shader_subgroup_vote = true,
      .EXT_memory_priority = true,
      .EXT_pageable_device_local_memory = true,
      .EXT_device_fault = true,
      .EXT_pipeline_properties = true,
      .EXT_image_compression_control = true,
      .EXT_legacy_dithering = true,
      .EXT_debug_marker = true,
      .KHR_shader_fma = true,
      .KHR_shader_untyped_pointers = true,
      .KHR_fragment_shader_barycentric = true,
      .EXT_mutable_descriptor_type = true,
      .VALVE_mutable_descriptor_type = true,
      .EXT_mesh_shader = wvk_physical_device_meshes(pdev),
      .EXT_discard_rectangles = true,
      .EXT_ycbcr_image_arrays = true,
      /* R10X6G10X6B10X6A10X6's texture is RGBA16Unorm's, which samples
       * as it is (wvk_format.c). */
      .EXT_rgba10x6_formats = true,
      .EXT_multisampled_render_to_single_sampled = true,
      .EXT_legacy_vertex_attributes = true,
      .KHR_multiview = true,
      /* What WebGPU leaves out; hidden for testing what the CTS tests only
       * of implementations without it (WINK_VK_NO_PORTABILITY). */
      .KHR_portability_subset = !getenv("WINK_VK_NO_PORTABILITY"),
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
      .KHR_swapchain_maintenance1 = true,
      .EXT_swapchain_maintenance1 = true,
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
      .EXT_robustness2 = true,
      .KHR_robustness2 = true,
      .EXT_depth_clip_control = true,
      .EXT_host_image_copy = true,
      .EXT_conditional_rendering = true,
      .EXT_provoking_vertex = true,
      .KHR_map_memory2 = true,
      .EXT_memory_budget = true,
      .KHR_unified_image_layouts = true,
      .EXT_zero_initialize_device_memory = true,
      .KHR_shader_relaxed_extended_instruction = true,
      .EXT_load_store_op_none = true,
      .EXT_dynamic_rendering_unused_attachments = true,
      .EXT_frame_boundary = true,
      .EXT_depth_bias_control = true,
      .EXT_attachment_feedback_loop_layout = true,
      .EXT_attachment_feedback_loop_dynamic_state = true,
      .EXT_pipeline_protected_access = true,
      .KHR_shader_atomic_int64 = true,
      .KHR_fragment_shading_rate = true,
      .EXT_shader_atomic_float = true,
      .EXT_shader_atomic_float2 = true,
      .EXT_primitive_topology_list_restart = true,
      .KHR_line_rasterization = true,
      .KHR_dynamic_rendering_local_read = true,
      .EXT_shader_tile_image = wvk_physical_device_has_feature(pdev, WGPUFeatureName_FramebufferFetch),
      .EXT_rasterization_order_attachment_access =
         wvk_physical_device_has_feature(pdev, WGPUFeatureName_FramebufferFetch),
      .ARM_rasterization_order_attachment_access =
         wvk_physical_device_has_feature(pdev, WGPUFeatureName_FramebufferFetch),
      .EXT_line_rasterization = true,
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
   const bool fetch = wvk_physical_device_has_feature(pdev, WGPUFeatureName_FramebufferFetch);

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
      /* Native Vulkan counts in hardware. Metal and browsers add passing
       * samples to a private float attachment, then reduce it into u64. */
      .occlusionQueryPrecise = pdev->info.backendType == WGPUBackendType_Vulkan ||
                              wvk_physical_device_has_feature(pdev, WGPUFeatureName_Float32Blendable),
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
      /* VK_EXT_ycbcr_image_arrays */
      .ycbcrImageArrays = true,
      /* VK_EXT_rgba10x6_formats */
      .formatRgba10x6WithoutYCbCrSampler = true,
      /* To multisampled images of the attachments', resolved into them
       * (wvk_cmd_render.c). */
      .multisampledRenderToSingleSampled = true,
      /* Unaligned attributes, and of other types than the shader's: which
       * the vertex shader reads itself (wvk_shader.c's mismatched). */
      .legacyVertexAttributes = true,
      /* Present fences, modes of each present, and images released
       * (wvk_wsi.c); the images made with the swapchain, deferred or not. */
      .swapchainMaintenance1 = true,
      /* Of a subgroup of one invocation, which holds all of a matrix
       * (wvk_nir_cmat.c); its loads are the buffers' own, robust or not. */
      .cooperativeMatrix = true,
      .cooperativeMatrixRobustBufferAccess = true,
      /* bfloat16 and 8-bit floats, as their bits, converted through
       * 32-bit floats (wvk_nir_minifloats.c); in cooperative matrices
       * too. */
      .shaderBFloat16Type = true,
      .shaderBFloat16CooperativeMatrix = true,
      .shaderFloat8 = true,
      .shaderFloat8CooperativeMatrix = true,
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
      /* Words under a lock (wvk_nir_atomic64.c). */
      .shaderBufferInt64Atomics = true,
      .shaderSharedInt64Atomics = true,
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
      /* Buffers bounded by their descriptors' ranges, in the shader, and
       * null ones of no range (wvk_shader.c's lower_word_access); texel
       * buffers by their elements (wvk_nir_descriptors.c). */
      .robustBufferAccess2 = true,
      /* Images bounded by their views, and null ones reading zero, by
       * the shader too (wvk_nir_descriptors.c). */
      .robustImageAccess2 = true,
      .nullDescriptor = true,
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
      /* The predicate the host's where it has it, else indirect commands
       * of the arguments a pass of the driver's zeroes (wvk_cmd_draw.c). */
      .conditionalRendering = true,
      .inheritedConditionalRendering = true,
      /* Lists of the draw's primitives, each beginning at its last vertex,
       * where the fragment shader has flat varyings (wvk_cmd_draw.c); and
       * a geometry shader's. */
      .provokingVertexLast = true,
      .transformFeedbackPreservesProvokingVertex = false,
      /* Layouts are nothing to WebGPU. */
      .unifiedImageLayouts = true,
      /* WebGPU's buffers and textures begin zeroed, and so does the host's
       * copy of memory. */
      .zeroInitializeDeviceMemory = true,
      .shaderRelaxedExtendedInstruction = true,
      /* Render pipelines are of the pass's attachments, whatever the
       * pipeline's rendering info said. */
      .dynamicRenderingUnusedAttachments = true,
      .frameBoundary = true,
      /* The default representation only, which is WebGPU's. */
      .depthBiasControl = true,
      /* What a draw samples of an attachment of its pass is a copy, made
       * before it and kept until a barrier (snapshot_feedback). */
      .attachmentFeedbackLoopLayout = true,
      .attachmentFeedbackLoopDynamicState = true,
      /* There is no protected memory: the pipeline flags ask nothing. */
      .pipelineProtectedAccess = true,
      /* Atomics of 32-bit floats: exchanges are of their bits, and sums,
       * minimums and maximums compare-exchange loops on words -- of
       * buffers, shared memory, and images' buffers of texels
       * (wvk_shader.c, gex_wgpu_nir.c). WGSL has no 64-bit atomics, and
       * 16-bit floats would be parts of words. */
      .shaderBufferFloat32Atomics = true,
      .shaderBufferFloat32AtomicAdd = true,
      .shaderSharedFloat32Atomics = true,
      .shaderSharedFloat32AtomicAdd = true,
      .shaderImageFloat32Atomics = true,
      .shaderImageFloat32AtomicAdd = true,
      .shaderBufferFloat32AtomicMinMax = true,
      .shaderSharedFloat32AtomicMinMax = true,
      .shaderImageFloat32AtomicMinMax = true,
      /* Lists that restart are drawn of the driver's lists of their whole
       * primitives (wvk_cmd_draw.c, LIST_COMPACT); patches of them
       * tessellation would have to read so too. */
      .primitiveTopologyListRestart = true,
      .primitiveTopologyPatchListRestart = false,
      /* Lines of each mode but the default one-pixel ones are quads, as
       * wide ones are, rectangles or parallelograms (Bresenham's), and a
       * fragment shader's stipple and coverage for them (wvk_shader.c's
       * expand_lines and lower_line_fs). */
      /* What a draw reads of its pass's attachments is a copy, made before
       * it and kept until a barrier (snapshot_feedback); and a fragment
       * shader's outputs are moved to the attachments' locations (a
       * variant's remap_color_outputs). */
      .dynamicRenderingLocalRead = true,
      .rectangularLines = true,
      .bresenhamLines = true,
      .smoothLines = true,
      .stippledRectangularLines = true,
      .stippledBresenhamLines = true,
      .stippledSmoothLines = true,
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
      .extendedDynamicState3ProvokingVertexMode = true,
      .extendedDynamicState3LineRasterizationMode = true,
      .extendedDynamicState3ColorBlendAdvanced = true,
      .extendedDynamicState3LineStippleEnable = true,
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
      /* Secondaries and commands of the pass's own in one pass: all are
       * the primary's commands, replayed in order (vk_cmd_queue). */
      .maintenance7 = true,
      /* Offsets not constant are the coordinates moved (wvk_shader.c);
       * copies between depth or stencil and colour go through the
       * texture copies' shaders where WebGPU's own will not; SPIR-V's
       * remainders are NIR's already; and barriers are in order. */
      .maintenance8 = true,
      /* Events and barriers of one in-order queue; and a device of no
       * queues has its own, which nothing submits to. */
      .maintenance9 = true,
      /* vk_meta's resolves of the modes and transfer functions it is
       * told (VkResolveImageModeInfoKHR, VkRenderingAttachmentFlagsInfoKHR),
       * and depth and stencil copied on either queue family: they are one
       * WebGPU queue. */
      .maintenance10 = true,
      /* A shader's WGSL (wvk_shader.c). */
      .pipelineExecutableInfo = true,
      /* Dawn clamps to [0, 1] whatever the viewport's range. */
      .depthClampZeroOne = true,
      /* Shared memory is words already, where any block may be at 0. */
      .workgroupMemoryExplicitLayout = true,
      .workgroupMemoryExplicitLayoutScalarBlockLayout = true,
      .workgroupMemoryExplicitLayout8BitAccess = true,
      .workgroupMemoryExplicitLayout16BitAccess = true,
      .vertexAttributeRobustness = true,
      /* The runtime's, pipelines of them found in its caches. */
      .shaderModuleIdentifier = true,
      /* The view's record's, which a sampling's LOD is raised to and a
       * fetch below reads zero (wvk_nir_descriptors.c). */
      .minLod = true,
      /* A variant samples such a cube as the 2D array of its faces, a
       * face at a time (wvk_shader.c's face_cubes). */
      .nonSeamlessCubeMap = true,
      /* A 2D view of a 3D image is the 3D texture's view, a variant taking
       * the slice its record says (wvk_shader.c's flat_views); and a
       * storage view of some slices moves z to them (wvk_nir_descriptors.c). */
      .image2DViewOf3D = true,
      .sampler2DViewOf3D = true,
      .imageSlicedViewOf3D = true,
      /* The fragment shader's, against a copy of the attachment made before
       * each draw (lower_blend): within a draw, not coherent. */
      .advancedBlendCoherentOperations = false,
      /* Hints, which WebGPU has nowhere to take. */
      .memoryPriority = true,
      .pageableDeviceLocalMemory = true,
      /* What WebGPU said when it lost the device. */
      .deviceFault = true,
      .pipelinePropertiesIdentifier = true,
      /* Every image uncompressed, for all WebGPU says. */
      .imageCompressionControl = true,
      /* Nothing is dithered, which legacy dithering allows. */
      .legacyDithering = true,
      /* WGSL's fma, which Metal's is: fused. Not of 64-bit floats, whose
       * software multiplies and then adds. */
      .shaderFmaFloat16 = wvk_physical_device_has_feature(pdev, WGPUFeatureName_ShaderF16),
      .shaderFmaFloat32 = true,
      /* Buffers and workgroup memory are words anyway (wvk_shader.c). */
      .shaderUntypedPointers = true,
      /* Of the geometry shader's lists, which a draw whose fragment shader
       * reads them goes through (wvk_shader.c's geometry_vertex_shader). */
      .fragmentShaderBarycentric = true,
      /* Colour attachments by framebuffer fetch, in rasterization order
       * on a tiler (wvk_nir_fetch.c); not depth or stencil, which Metal
       * fetches none of. */
      .shaderTileImageColorReadAccess = fetch,
      /* Coarse fragments, each pixel of one shading as it does
       * (wvk_nir_fsr.c): the pipeline's rate, and a vertex shader's of
       * its primitives. */
      .pipelineFragmentShadingRate = true,
      .primitiveFragmentShadingRate = true,
      .attachmentFragmentShadingRate = true,
      .rasterizationOrderColorAttachmentAccess = fetch,
      /* Every descriptor is one struct, whose binding's WebGPU binding the
       * shader's use of it types (wvk_descriptor_set.c). */
      .mutableDescriptorType = true,
      /* As compute, and the draw of what they made (wvk_cmd_draw.c's
       * draw_mesh). */
      .taskShader = wvk_physical_device_meshes(pdev),
      .meshShader = wvk_physical_device_meshes(pdev),
      .meshShaderQueries = wvk_physical_device_meshes(pdev),
      /* Recorded with each primitive, as its other outputs of it are. */
      .primitiveFragmentShadingRateMeshShader = wvk_physical_device_meshes(pdev),
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
      .imageView2DOn3DImage = true,
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
      .maxFragmentOutputAttachments = pdev->color_attachments,
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
      /* VK_EXT_provoking_vertex: the lists are the draw's. */
      .provokingVertexModePerPipeline = true,
      .transformFeedbackPreservesTriangleFanProvokingVertex = false,
      /* VK_EXT_blend_operation_advanced: the fragment shader's, of every
       * attachment, NIR's equations. */
      .advancedBlendMaxColorAttachments = 8,
      .advancedBlendIndependentBlend = true,
      .advancedBlendNonPremultipliedSrcColor = true,
      .advancedBlendNonPremultipliedDstColor = true,
      .advancedBlendCorrelatedOverlap = true,
      .advancedBlendAllOperations = true,
      /* VK_EXT_discard_rectangles: the fragment shader's test. */
      .maxDiscardRectangles = WVK_MAX_DISCARD_RECTANGLES,
      /* VK_EXT_mesh_shader: the least of each. Payloads and outputs are
       * in buffers; shared memory is workgroup memory, of which WebGPU has
       * to have the 32 KiB a task shader may. */
      .maxTaskWorkGroupTotalCount = 1u << 22,
      .maxTaskWorkGroupCount = { 65535, 65535, 65535 },
      .maxTaskWorkGroupInvocations = 128,
      .maxTaskWorkGroupSize = { 128, 128, 128 },
      .maxTaskPayloadSize = 16384,
      .maxTaskSharedMemorySize = 32768,
      .maxTaskPayloadAndSharedMemorySize = 32768,
      .maxMeshWorkGroupTotalCount = 1u << 22,
      .maxMeshWorkGroupCount = { 65535, 65535, 65535 },
      .maxMeshWorkGroupInvocations = 128,
      .maxMeshWorkGroupSize = { 128, 128, 128 },
      .maxMeshSharedMemorySize = 28672,
      .maxMeshPayloadAndSharedMemorySize = 28672,
      .maxMeshOutputMemorySize = 32768,
      .maxMeshPayloadAndOutputMemorySize = 48128,
      .maxMeshOutputComponents = 128,
      .maxMeshOutputVertices = 256,
      .maxMeshOutputPrimitives = 256,
      .maxMeshOutputLayers = 8,
      .maxMeshMultiviewViewCount = 1,
      .meshOutputPerVertexGranularity = 1,
      .meshOutputPerPrimitiveGranularity = 1,
      .maxPreferredTaskWorkGroupInvocations = 64,
      .maxPreferredMeshWorkGroupInvocations = 64,
      /* VK_KHR_fragment_shader_barycentric: a strip's triangles are the
       * geometry shader's, whose order is its provoking vertex's. */
      .triStripVertexOrderIndependentOfProvokingVertex = false,
      /* VK_EXT_shader_tile_image: Metal's programmable blending, of the
       * invocation's own sample. */
      /* VK_KHR_fragment_shading_rate: a quad's at most (wvk_nir_fsr.c);
       * 1x1 with what is the pixel's or sample's own. */
      .minFragmentShadingRateAttachmentTexelSize = { 8, 8 },
      .maxFragmentShadingRateAttachmentTexelSize = { 32, 32 },
      .maxFragmentShadingRateAttachmentTexelSizeAspectRatio = 4,
      .primitiveFragmentShadingRateWithMultipleViewports = false,
      .layeredShadingRateAttachments = false,
      .fragmentShadingRateNonTrivialCombinerOps = true,
      .maxFragmentSize = { 2, 2 },
      .maxFragmentSizeAspectRatio = 2,
      .maxFragmentShadingRateCoverageSamples = 16,
      .maxFragmentShadingRateRasterizationSamples = VK_SAMPLE_COUNT_4_BIT,
      .fragmentShadingRateWithShaderDepthStencilWrites = false,
      .fragmentShadingRateWithSampleMask = false,
      .fragmentShadingRateWithShaderSampleMask = false,
      .fragmentShadingRateWithConservativeRasterization = false,
      .fragmentShadingRateWithFragmentShaderInterlock = false,
      .fragmentShadingRateWithCustomSampleLocations = false,
      .fragmentShadingRateStrictMultiplyCombiner = true,
      .shaderTileImageCoherentReadAccelerated = true,
      .shaderTileImageReadSampleFromPixelRateInvocation = false,
      .shaderTileImageReadFromHelperInvocation = false,
      /* VK_KHR_line_rasterization: the quads' corners are the vertex
       * shader's positions moved, as precise as the rest. */
      .lineSubPixelPrecisionBits = 4,
      /* VK_EXT_robustness2: a word's bounds. */
      .robustStorageBufferAccessSizeAlignment = 4,
      .robustUniformBufferAccessSizeAlignment = 4,
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
      .maxColorAttachments = pdev->color_attachments,
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
      /* Less the padding memory's buffer has (wvk_device_memory_padding). */
      .maxMemoryAllocationSize = (l->maxBufferSize - WVK_MEMORY_ROBUST_PADDING) & ~255ull,

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
      .maxBufferSize = (l->maxBufferSize - WVK_MEMORY_ROBUST_PADDING) & ~255ull,
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
      /* VK_KHR_maintenance7: no fragment shading rate; a pass writes the
       * depth/stencil attachment whole. */
      .robustFragmentShadingRateAttachmentAccess = false,
      .separateDepthStencilAttachmentAccess = false,
      /* VK_KHR_maintenance9: no sparse images; and WebGPU fills in what a
       * vertex format has not with 0, and alpha 1. */
      .image2DViewOf3DSparse = false,
      .defaultVertexAttributeValue = VK_DEFAULT_VERTEX_ATTRIBUTE_VALUE_ZERO_ZERO_ZERO_ONE_KHR,
      /* VK_KHR_maintenance10: the 4-bit formats' opaque black is their
       * own; Metal's resolves of sRGB average linear values, and vk_meta's
       * the encoded ones where told to. */
      .rgba4OpaqueBlackSwizzled = false,
      /* VK_KHR_cooperative_matrix */
      .cooperativeMatrixSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT,
      /* VK_EXT_legacy_vertex_attributes: those the shader reads itself. */
      .nativeUnalignedPerformance = false,
      .resolveSrgbFormatAppliesTransferFunction = true,
      .resolveSrgbFormatSupportsTransferFunctionControl = true,
      .maxDescriptorSetTotalUniformBuffersDynamic = l->maxDynamicUniformBuffersPerPipelineLayout,
      .maxDescriptorSetTotalStorageBuffersDynamic = l->maxDynamicStorageBuffersPerPipelineLayout,
      .maxDescriptorSetTotalBuffersDynamic =
         l->maxDynamicUniformBuffersPerPipelineLayout + l->maxDynamicStorageBuffersPerPipelineLayout,
      .maxDescriptorSetUpdateAfterBindTotalUniformBuffersDynamic = l->maxDynamicUniformBuffersPerPipelineLayout,
      .maxDescriptorSetUpdateAfterBindTotalStorageBuffersDynamic = l->maxDynamicStorageBuffersPerPipelineLayout,
      .maxDescriptorSetUpdateAfterBindTotalBuffersDynamic =
         l->maxDynamicUniformBuffersPerPipelineLayout + l->maxDynamicStorageBuffersPerPipelineLayout,
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
   /* The runtime's identifiers of modules (VK_EXT_shader_module_identifier). */
   STATIC_ASSERT(sizeof(vk_shaderModuleIdentifierAlgorithmUUID) == sizeof(p->shaderModuleIdentifierAlgorithmUUID));
   memcpy(p->shaderModuleIdentifierAlgorithmUUID, vk_shaderModuleIdentifierAlgorithmUUID,
          sizeof(p->shaderModuleIdentifierAlgorithmUUID));
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
   /* Optimal images' alone. Their memory is accounting, each a texture of
    * its own, but buffer memory is a WGPUBuffer of all of it: an allocator
    * that put both in one block made a buffer of the images' bytes too,
    * which nothing used. Images of a type buffers cannot have keep the
    * two apart. */
   pdev->memory_types[WVK_MEMORY_IMAGE] = (VkMemoryType){
      .propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
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
   /* Vulkan has no limit of colour attachments' bytes per sample, so as
    * many attachments as it says are to be had of any format: WebGPU
    * counts an RGBA8 attachment 8 bytes, and 16 at most of the formats
    * it renders to but a few 32-bit four-channel ones. Not below Vulkan's
    * minimum of 4, which a device of 32 bytes (WebGPU's default, and what
    * Dawn reports on Vulkan) has only of the smaller formats. */
   pdev->color_attachments =
      MIN2(pdev->limits.maxColorAttachments,
           MAX2(pdev->limits.maxColorAttachmentBytesPerSample / 8, 4));
   for (size_t i = 0; i < ARRAY_SIZE(wanted_features); ++i) {
      if (wgpuAdapterHasFeature(adapter, wanted_features[i]) &&
          pdev->feature_count < ARRAY_SIZE(pdev->features))
         pdev->features[pdev->feature_count++] = wanted_features[i];
   }
   /* Reserve an attachment (and four bytes per sample) for exact occlusion
    * counts on backends whose WebGPU query returns only visibility. */
   if (pdev->info.backendType != WGPUBackendType_Vulkan &&
       wvk_physical_device_has_feature(pdev, WGPUFeatureName_Float32Blendable))
      pdev->color_attachments = MIN2(pdev->color_attachments,
         MIN2(7, (pdev->limits.maxColorAttachmentBytesPerSample - 4) / 8));

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
            case VK_STRUCTURE_TYPE_QUEUE_FAMILY_OWNERSHIP_TRANSFER_PROPERTIES_KHR: {
               /* Both families' submits are one queue's: an image is the
                * same to either (VK_KHR_maintenance9). */
               VkQueueFamilyOwnershipTransferPropertiesKHR *own = ext;
               own->optimalImageTransferToQueueFamilies = BITFIELD_MASK(WVK_QUEUE_FAMILIES);
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
      switch (stype) {
      case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT: {
         /* All of each heap, as far as the driver knows: WebGPU says
          * nothing of what others use. */
         VkPhysicalDeviceMemoryBudgetPropertiesEXT *budget = ext;
         for (uint32_t i = 0; i < VK_MAX_MEMORY_HEAPS; ++i) {
            budget->heapBudget[i] = i < props->memoryHeapCount ? props->memoryHeaps[i].size : 0;
            budget->heapUsage[i] = i < props->memoryHeapCount ? p_atomic_read(&pdev->heap_used[i]) : 0;
         }
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
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
      /* A shading rate attachment, which fragment shaders fetch the
       * first component of (wvk_nir_fsr.c). */
      if (wvk_format_shading_rate(format))
         f |= VK_FORMAT_FEATURE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
   }
   /* Transfers work on every format: the ones WebGPU will not copy --
    * Depth24Plus at all, Depth32Float into -- go through a shader. And so
    * do the host's, which are those transfers (wvk_host_copy.c). */
   f |= VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT |
        VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT;
   /* On any queue family: they are all one WebGPU queue, whose render
    * passes copy what WebGPU will not (VK_KHR_maintenance10). */
   if (caps & WVK_CAP_DEPTH)
      f |= VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_COMPUTE_QUEUE_BIT_KHR | VK_FORMAT_FEATURE_2_DEPTH_COPY_ON_TRANSFER_QUEUE_BIT_KHR;
   if (caps & WVK_CAP_STENCIL)
      f |= VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_COMPUTE_QUEUE_BIT_KHR |
           VK_FORMAT_FEATURE_2_STENCIL_COPY_ON_TRANSFER_QUEUE_BIT_KHR;
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
      case VK_STRUCTURE_TYPE_SUBPASS_RESOLVE_PERFORMANCE_QUERY_EXT: {
         /* A pass rendering it multisampled has an image of its own, and
          * a resolve at its end, and a draw to fill it at its start. */
         VkSubpassResolvePerformanceQueryEXT *q = ext;
         q->optimal = VK_FALSE;
         break;
      }
      default:
         vk_debug_ignored_stype(stype);
         break;
      }
   }
}

/* Whether an optimal image of `format` can have these uses. */
static bool
format_usage_supported(const struct wvk_physical_device *pdev, VkFormat format, VkImageType type,
                       VkImageUsageFlags usage, bool plane_storage)
{
   const uint32_t caps = wvk_format_caps(pdev, format);
   if (!caps)
      return false;
   if ((usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)) && !(caps & WVK_CAP_SAMPLE))
      return false;
   /* An input attachment is an attachment too. */
   if ((usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT) && !(caps & (WVK_CAP_RENDER | WVK_CAP_DEPTH | WVK_CAP_STENCIL)))
      return false;
   if ((usage & VK_IMAGE_USAGE_STORAGE_BIT) && !(caps & WVK_CAP_STORAGE) && !plane_storage)
      return false;
   if ((usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) && !(caps & WVK_CAP_RENDER))
      return false;
   if ((usage & VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR) &&
       (!wvk_format_shading_rate(format) || type != VK_IMAGE_TYPE_2D))
      return false;
   if ((usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) && !(caps & (WVK_CAP_DEPTH | WVK_CAP_STENCIL)))
      return false;
   return true;
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

   /* A multi-planar image's planes may be storage, viewed as formats
    * that are, where it says so. */
   const bool plane_storage = (caps & WVK_CAP_PLANAR) && (info->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT);
   if (info->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) {
      /* Each use of the image's own format or of a format it may be
       * viewed as: of a texel of as many bytes, compressed alike. */
      const enum pipe_format pf = vk_format_to_pipe_format(info->format);
      u_foreach_bit(bit, usage) {
         const VkImageUsageFlags one = BITFIELD_BIT(bit);
         bool any = format_usage_supported(pdev, info->format, info->type, one, plane_storage);
         for (VkFormat f = VK_FORMAT_R4G4_UNORM_PACK8; !any && f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
            const enum pipe_format of = vk_format_to_pipe_format(f);
            if (of == PIPE_FORMAT_NONE || util_format_get_blocksize(of) != util_format_get_blocksize(pf) ||
                util_format_is_compressed(of) != util_format_is_compressed(pf) ||
                util_format_get_blockwidth(of) != util_format_get_blockwidth(pf) ||
                util_format_get_blockheight(of) != util_format_get_blockheight(pf) ||
                vk_format_is_depth_or_stencil(f) != vk_format_is_depth_or_stencil(info->format))
               continue;
            any = format_usage_supported(pdev, f, info->type, one, false);
         }
         if (!any)
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }
   } else if (!format_usage_supported(pdev, info->format, info->type, usage, plane_storage)) {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }
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

   /* A multi-planar image: 2D, of one level, of layers as any other
    * (VK_EXT_ycbcr_image_arrays), not disjoint but where the format says it
    * may be. */
   if (caps & WVK_CAP_PLANAR) {
      if (info->type != VK_IMAGE_TYPE_2D ||
          (usage & ~(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_HOST_TRANSFER_BIT |
                     (plane_storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0))))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      pImageFormatProperties->imageFormatProperties = (VkImageFormatProperties){
         .maxExtent = extent,
         .maxMipLevels = 1,
         .maxArrayLayers = layers,
         .sampleCounts = VK_SAMPLE_COUNT_1_BIT,
         .maxResourceSize = UINT32_MAX,
      };
   } else if (info->flags & VK_IMAGE_CREATE_DISJOINT_BIT) {
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }
   /* Rendered multisampled, to an image of its own of the format: which
    * WebGPU has to be able to multisample. */
   if ((info->flags & VK_IMAGE_CREATE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_BIT_EXT) &&
       (info->type != VK_IMAGE_TYPE_2D || !(caps & WVK_CAP_MSAA)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
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
      case VK_STRUCTURE_TYPE_IMAGE_COMPRESSION_PROPERTIES_EXT: {
         /* WebGPU says nothing of how textures are stored. */
         VkImageCompressionPropertiesEXT *p = ext;
         p->imageCompressionFlags = VK_IMAGE_COMPRESSION_DISABLED_EXT;
         p->imageCompressionFixedRateFlags = VK_IMAGE_COMPRESSION_FIXED_RATE_NONE_EXT;
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

/* A quad's fragment sizes (wvk_nir_fsr.c), largest first: of the sample
 * counts WebGPU has, and 1x1 of any. */
VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceFragmentShadingRatesKHR(VkPhysicalDevice physicalDevice, uint32_t *pFragmentShadingRateCount,
                                             VkPhysicalDeviceFragmentShadingRateKHR *pFragmentShadingRates)
{
   static const VkExtent2D sizes[] = { { 2, 2 }, { 2, 1 }, { 1, 2 }, { 1, 1 } };
   VK_OUTARRAY_MAKE_TYPED(VkPhysicalDeviceFragmentShadingRateKHR, out, pFragmentShadingRates,
                          pFragmentShadingRateCount);
   for (uint32_t i = 0; i < ARRAY_SIZE(sizes); ++i) {
      vk_outarray_append_typed(VkPhysicalDeviceFragmentShadingRateKHR, &out, r) {
         r->fragmentSize = sizes[i];
         r->sampleCounts = i == ARRAY_SIZE(sizes) - 1 ? ~0u : WVK_SAMPLE_COUNTS;
      }
   }
   return vk_outarray_status(&out);
}

/* The matrices a subgroup of one invocation multiplies (wvk_nir_cmat.c),
 * 8x8x8: of 16- and 32-bit floats, into either; of 8-, 16- and 32-bit
 * integers, of either signedness, into 32 bits -- or into their own size,
 * unsigned: the CTS has signed ones of 8 and 16 bits wrong -- and the 8-bit
 * ones saturating too; and of bfloat16 and 8-bit floats into 32-bit
 * floats, or into their own kind or 16-bit ones. */
VKAPI_ATTR VkResult VKAPI_CALL
wvk_GetPhysicalDeviceCooperativeMatrixPropertiesKHR(VkPhysicalDevice physicalDevice, uint32_t *pPropertyCount,
                                                    VkCooperativeMatrixPropertiesKHR *pProperties)
{
#define F16 VK_COMPONENT_TYPE_FLOAT16_KHR
#define F32 VK_COMPONENT_TYPE_FLOAT32_KHR
#define U8 VK_COMPONENT_TYPE_UINT8_KHR
#define S8 VK_COMPONENT_TYPE_SINT8_KHR
#define BF16 VK_COMPONENT_TYPE_BFLOAT16_KHR
#define E4M3 VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT
#define E5M2 VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT
#define U16 VK_COMPONENT_TYPE_UINT16_KHR
#define S16 VK_COMPONENT_TYPE_SINT16_KHR
#define U32 VK_COMPONENT_TYPE_UINT32_KHR
#define S32 VK_COMPONENT_TYPE_SINT32_KHR
   static const VkComponentTypeKHR types[][5] = {
      { F16, F16, F16, F16, 0 }, { F16, F16, F32, F32, 0 }, { F32, F32, F32, F32, 0 },
      { U8, U8, U32, U32, 0 },   { S8, S8, S32, S32, 0 },   { U8, U8, S32, S32, 0 },
      { S8, S8, U32, U32, 0 },   { U8, U8, U8, U8, 0 },     { U16, U16, U16, U16, 0 },
      { U16, U16, U32, U32, 0 },
      { S16, S16, S32, S32, 0 }, { U32, U32, U32, U32, 0 }, { S32, S32, S32, S32, 0 },
      { U8, U8, U32, U32, 1 },   { S8, S8, S32, S32, 1 },
      { BF16, BF16, F32, F32, 0 }, { BF16, BF16, BF16, BF16, 0 }, { E4M3, E4M3, F32, F32, 0 },
      { E5M2, E5M2, F32, F32, 0 }, { E4M3, E4M3, F16, F16, 0 },   { E5M2, E5M2, F16, F16, 0 },
   };
#undef F16
#undef F32
#undef U8
#undef S8
#undef BF16
#undef E4M3
#undef E5M2
#undef U16
#undef S16
#undef U32
#undef S32
   VK_OUTARRAY_MAKE_TYPED(VkCooperativeMatrixPropertiesKHR, out, pProperties, pPropertyCount);
   for (uint32_t i = 0; i < ARRAY_SIZE(types); ++i) {
      vk_outarray_append_typed(VkCooperativeMatrixPropertiesKHR, &out, p) {
         *p = (VkCooperativeMatrixPropertiesKHR){
            .sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR,
            .MSize = 8,
            .NSize = 8,
            .KSize = 8,
            .AType = types[i][0],
            .BType = types[i][1],
            .CType = types[i][2],
            .ResultType = types[i][3],
            .saturatingAccumulation = types[i][4],
            .scope = VK_SCOPE_SUBGROUP_KHR,
         };
      }
   }
   return vk_outarray_status(&out);
}
