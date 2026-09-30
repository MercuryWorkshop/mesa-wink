# wvk: Vulkan on WebGPU

The wink Vulkan driver: Vulkan 1.2 with `VK_KHR_portability_subset`, on a
WebGPU device. It is Mesa's Vulkan runtime on top, and underneath it is the
shader compiler the GL driver (`gex_wgpu`) uses, in `src/wink/compiler`.
Natively it is an ICD over Dawn, which is where Vulkan's tools, vkcube and
dEQP-VK run. In a browser it is part of libwink, over emdawnwebgpu.

## How it works

- **Commands are recorded, and replayed at submit.** Every `vkCmd*` goes
  into the command buffer's `vk_cmd_queue`. A submit replays its command
  buffers through the driver's own entrypoints into one `WGPUCommandEncoder`.
  WebGPU command buffers are single-use and Vulkan's are not. Replaying late
  also lets a pass know, when it begins, what it will need.
- **One in-order queue.** The runtime's emulated timelines defer a submit
  until what it waits for is pending. On one in-order WebGPU queue, nothing
  is then left to wait for on the GPU: barriers and events cost nothing, and
  acquiring a swapchain image never blocks. A second queue family, of
  compute and transfer, has a queue that submits to the same WebGPU queue.
  Two families let two threads submit at once, so native Dawn is asked for
  `ImplicitDeviceSynchronization`.
- **Host-visible memory is a copy on the host.** Ranges a submit reads are
  uploaded with `wgpuQueueWriteBuffer` before it. Ranges it writes are read
  back after it, and the submit counts as finished only then.
- **Pipelines are made at the draw.** A WebGPU pipeline holds much of what
  Vulkan keeps as dynamic state, and its bind group layouts hold what only
  the shaders say. So pipelines are built from the bound shaders and the
  runtime's dynamic graphics state, and cached by all of it. Bind groups are
  built from descriptor sets and kept until the set is written again.
- **Shaders:** SPIR-V → NIR (vtn) → descriptors remapped, combined image
  samplers split, push constants as a storage buffer → zink's `nir_to_spirv`
  → Tint → WGSL. Set N is `@group(N)`, and each element of a Vulkan binding
  is a binding of its own. The driver's bindings sit at the top of group 3.
- **Vertex attribute divisors** other than 1 are read by the vertex shader
  itself, from the vertex buffer bound as storage, since WebGPU steps an
  instance attribute every instance; so are attributes of strides not
  whole words, or at offsets their formats are not aligned to, which
  WebGPU cannot lay out.
- **Shaders are compiled for some of the state they're drawn with.** What
  they depend on:
  - the vertex input they pull;
  - the fragment shader's interpolation, which WebGPU wants a vertex
    shader's outputs to have;
  - alpha to one, and coverage from alpha where WebGPU's won't do (with
    one sample, or with the shader's own mask);
  - the samples positions are placed for;
  - whether the pass has a depth attachment to write to;
  - where each set's records are.

  Shaders compiled apart don't know that state: shader objects, a fast
  link's pipeline libraries, and anything with dynamic vertex input or
  multisampling. Each keeps its source for that: its NIR as compiled, and
  copies of its set layouts, serialized with it. A draw whose state
  differs from what a shader was compiled for uses a variant compiled
  from the source and cached (`wvk_shader_draw_variant`) -- by the
  device too, serialized, of a digest of the source and what it is a
  variant for, so that the variants of another shader alike, of another
  pipeline, are made of that and have its render pipelines. A pipeline's
  own shaders are kept so too, by a digest of all their compile reads --
  the source, the NIR as linked, push constants, immutable samplers'
  conversions and the pipeline state wvk_hash_state names -- not by the
  runtime's cache of weak references, whose hits were a matter of timing.
  `WINK_VK_VARIANT_LOG=1` says when one is made, why, and how long it
  took.
  Monolithic pipelines' shaders are compiled for their draws and have no
  variants.
- **64-bit floats**, which WGSL has none of, are Mesa's float64.glsl in
  integer arithmetic (built by glslang, where there is one), inlined by
  nir_lower_doubles; and 64-bit integers are pairs of words
  (nir_lower_int64). Slow, but there. A 64-bit varying is its words, flat,
  at the same locations, which its accesses pack and unpack.
- **8- and 16-bit values**, which WGSL has no integers of, are 32-bit
  words of their bits (`wvk_nir_small_types.c`): sign-extended where read
  as signed, kept to their bits where made. Half floats of float
  operations stay WGSL's f16; words of them, from memory or moves, are
  packHalf2x16's bits. Memory is words already: a store of less than a
  word is atomics on its bits, and interpolated half floats are floats.
- **Min and max reductions** (samplerFilterMinmax), which WebGPU's
  samplers do not do: a sample of a sampler that reduces fetches the
  filter's footprint -- the texels of weight not zero, at the level the LOD
  selects or the two of linear mipmaps, each wrapped as the sampler says
  or the border -- and takes their least or greatest, component by
  component. The WebGPU sampler of one filters nothing, so it takes depth
  textures too. A cube's, whose texels are across faces, is the base
  level's gathers, which the hardware finds the edges of: not the level a
  minified sample selects. It is a variant's, keyed on whether a sampler
  bound reduces -- each element of an array of them looked at -- since a
  shader of many samples, with it in all of them, was enough more for
  Apple's GPU to hang on.
- **8-bit indices** (indexTypeUint8) are widened to 16 bits by a pass of
  the driver's at the first draw after a bind, the restart index 0xFFFF
  where the draw restarts primitives.
- **Primitives generated queries** count each vertex stream's primitives,
  as the pipeline statistics queries count theirs. Counted queries share
  counters the replay keeps, host and GPU, which each query takes the
  difference of between its begin and end, so any of them may be active
  at once.
- **Transform feedback** (VK_EXT_transform_feedback, four streams): while
  it is on, every draw takes the geometry shader path -- one of the
  driver's that passes primitives through where the pipeline has none --
  and a pass after the geometry shader walks each stream's lists in order,
  writing each whole primitive's captured outputs from the records to the
  stream's buffers, until one is full. Where each buffer is at is the
  GPU's, copied from and to the counter buffers at begin and end; the
  primitives written and needed are counters of each stream's transform
  feedback queries. A draw by a counter's byte count is an indirect draw
  made on the GPU, but where the draw needs its count on the host (a
  geometry or tessellation shader's, points' and lines', and fans'),
  which a counter written in the same submit it does not have yet.
- **Conditional rendering** (VK_EXT_conditional_rendering): the predicate
  as the host has it, where the memory is the host's and nothing before
  in the submit writes it; else each draw and dispatch is an indirect one
  of arguments a pass of the driver's copies, zero where the predicate
  fails -- each draw the driver makes of a direct command, too -- and
  transform feedback's capture reads it. A render pass is ended for each.
- **Variable pointers** (variablePointers, variablePointersStorageBuffer):
  a buffer access whose binding a phi or select chooses, which cannot be
  traced back to one, is an access of each buffer of its kind the shader
  has. Every buffer index then says at run time which binding it is (set
  and WebGPU binding), and reads are selected from, writes done where it
  matches.
- **What WebGPU's objects cannot hold** of descriptors, shaders do
  themselves: each set has a record for each of its bindings, which the
  shader reads at the draw (`wvk_nir_descriptors.c`), from the push binding
  after the push constants. Texel buffers are
  storage buffers whose format the shader decodes and encodes, views'
  swizzles are applied to what is read through them, and samplers' border
  colours, mirroring once, LOD bias and unnormalized coordinates are the
  shader's. Integer textures, which WebGPU does not sample, are fetched
  from, with the sampler's address modes and LOD applied; an integer cube
  is the 2D array of its faces, the face chosen as Vulkan chooses it.
  Comparisons WGSL cannot make -- with gradients, a bias, or at a LOD
  other than 0 -- fetch and compare each texel of the filter by hand, and
  textureQueryLod is worked out from the sampler's record. A view's
  minimum LOD (minLod) is its record's: where it has one, a sample's LOD
  is raised to it as a MinLod operand's is, and a fetch below its level
  reads zero -- in a variant keyed on whether a view bound has one, as
  it makes two of every sample. A cube a non-seamless sampler samples (nonSeamlessCubeMap)
  is, in a variant keyed on the samplers bound, the 2D array of its faces:
  the face the direction selects, at the LOD the cube's derivatives give,
  wrapped as the sampler says. A 2D view of a 3D image's slice
  (VK_EXT_image_2d_view_of_3d) is WebGPU's 3D view of the level, which a
  variant keyed on the views bound samples at the middle of the slice its
  record says, and loads and stores at it; and a storage view of some of
  a 3D image's slices (VK_EXT_image_sliced_view_of_3d) moves z to its
  first and says its count for the depth.
- **Image atomics**, which WGSL does not have, are on a storage buffer of
  the image's texels. A 32-bit integer storage image has one besides its
  texture, and a shader that does atomics on it uses it for everything.
  Between barriers one or the other is the image: the texture is copied to
  the buffer before such a shader runs, and back at the next barrier. An
  R32 float one has one where the device does float atomics, or a view of
  it could be of integers.
- **Float atomics** (VK_EXT_shader_atomic_float and float2, 32-bit): WGSL
  has atomics of integers only. An exchange, load or store is of the
  float's bits; a sum, minimum or maximum a loop of compare-exchanges on
  the word until nothing came between -- of buffers, shared memory, and
  images' buffers of texels.
- **64-bit integer atomics** (shaderBufferInt64Atomics,
  shaderSharedInt64Atomics): each takes a lock of a table by its address
  -- the device's for buffers, a few after the workgroup's memory for it
  -- reads and writes the two words by 32-bit atomics, and gives it back,
  in one turn of a loop, so that invocations of one SIMD group do not
  wait on each other. WGSL's atomics are relaxed, and Metal reordered an
  exchange's words past the lock, so each access depends on the one
  before by a zero read from memory, which no compiler can fold.
- **A pass that reads what it renders to** -- an input attachment of its own
  subpass, a feedback loop -- reads a copy, which WebGPU wants: a texture is
  an attachment or bound in a pass, not both. The copy is made before the
  draw, with the pass begun again after, and kept until a barrier: a
  texture copy, or where WebGPU copies none a pass loading each texel --
  each sample of a multisampled image, depth into R32Float and stencil
  into R8Uint. Depth and stencil in read-only layouts are read-only to
  WebGPU, which lets shaders read them as they are. Passes store what
  they render even where Vulkan does not care, since one may be ended
  and begun again before Vulkan's is over. Dynamic rendering's local
  reads (dynamicRenderingLocalRead) are the same copies; attachments at
  other locations than their own have a fragment shader variant whose
  outputs are moved to them, WebGPU's targets being the pass's
  attachments in order, and an output no attachment is at is gone.
- **Colour attachments read as they are** -- tile images
  (VK_EXT_shader_tile_image, colour only), and input attachments that
  are colour attachments of their own subpass or rendering, by the input
  attachment map, in rasterization order
  (VK_EXT_rasterization_order_attachment_access) -- are Dawn's framebuffer
  fetch, where the device has it (a tiler's Metal): no copy. The fragment
  shader's variant for the attachments' formats has an input for each,
  named `wink_color_N`, which the WGSL makes `@color(N)`, of as many
  components as the attachment's WebGPU format and its output to that
  attachment of as many, since Metal wants the two of one type. A
  multisampled one is read at the invocation's own sample, which is all
  a fetch has. Not depth or stencil, which Metal fetches none of.
- **Fragment shading rates** (VK_KHR_fragment_shading_rate: the
  pipeline's, a vertex shader's of its primitives, an attachment's), which
  WebGPU has none of, are coarse fragments of at most 2x2 -- a quad's
  pixels, or a row or column of them -- every pixel of which shades as the
  fragment would (wvk_nir_fsr.c): at its centre, its interpolated inputs
  moved there by their derivatives, so that each computes what the
  fragment does and writes it to its own pixel. Derivatives are the
  fragment's: of their operand computed again of the pixel's own inputs,
  scaled by the fragment's size, and implicit LODs of those. Storage
  writes and atomics the fragment's first covered pixel does alone, and
  the others read an atomic's result of it across the quad, through
  derivatives. The fragment shader works the rate out: the pipeline's and
  the combiner ops of the driver's data, the primitive's a flat varying
  (a geometry or mesh shader's recorded as its other outputs are), the
  attachment's a fetch from a 2D view of its first layer. At 1x1 with
  sample shading, a sample mask, depth or stencil written.
- **Logic ops** are the fragment shader's: WebGPU has no colour state
  for them. A draw with one on gets a variant that takes each output to
  its attachment format's integers (unorm, snorm, uint and sint, at the
  format's own bits), combines them with the destination's and brings the
  result back; float and sRGB attachments pass unchanged, and blending is
  off, as Vulkan has it. The destination is a copy of the attachment's
  level and layer, made before each such draw with the pass begun again
  after, so primitives of one draw that overlap see the attachment as it
  was before the draw. Not on a multisampled pass, which says so.
- **Blend states WebGPU cannot express**, colour factors that use both
  CONSTANT_COLOR and CONSTANT_ALPHA where WebGPU has one constant, are the
  fragment shader's as well. The attachments whose colour factors use
  CONSTANT_ALPHA blend in a variant: it keeps what the shader writes to
  them, and at the end blends it with a copy of the attachment, made as a
  logic op's is, by the attachment's equation, clamped as normalized
  formats are. Dual-source blending works the same way. WebGPU blends the
  others with the real constant. Not on a multisampled pass either.
- **Advanced blending** (VK_EXT_blend_operation_advanced) is blended the
  same way, by NIR's equations (`nir_color_blend_advanced`), all of the
  operations, premultiplied or not, with each overlap, on up to 8
  attachments; not on a multisampled pass. Not coherent: primitives of one draw that overlap see the
  attachment as it was before the draw, as the extension allows without
  advancedBlendCoherentOperations.
- **The depth bounds test** is the fragment shader's too. It tests the
  depth the attachment holds, so a draw with it on gets a variant that
  reads a copy of the depth attachment (R32Float, made before the draw)
  at its fragment and leaves where the depth is outside the bounds, which
  the driver's push data has. At the end of the shader, as Vulkan's late
  test is, or at the start where it asks for early tests; a draw without a
  fragment shader gets the driver's own, which is the test alone. Not on a
  multisampled pass, which says so.
- **Discard rectangles** (VK_EXT_discard_rectangles), eight of them, are
  the fragment shader's first thing: a draw with them on gets a variant
  that leaves a fragment outside every rectangle, or inside one of
  exclusive ones, of those the driver's push data has; a draw without a
  fragment shader gets the driver's own, which is the test alone.
- **Points of any size** (largePoints) are quads: WebGPU's points are a
  pixel. A draw of a point list whose vertex shader writes PointSize is a
  triangle list of six vertices a point, drawn with a variant of the
  vertex shader that takes the point from vertex_index / 6 -- an indexed
  draw's index from the index buffer, bound as storage -- pulls every
  attribute itself at it, and moves the position to the corner, by half
  of PointSize in the viewport's pixels; gl_PointCoord goes to the
  fragment shader as a varying, and it faces front. Not for attributes of
  more vertex buffers than the pulled ones (4), nor for indirect draws in
  memory the host has no copy of: those are a pixel each, and say so.
- **Wide lines** (wideLines) are quads too, a parallelogram across the
  minor axis, as Vulkan's non-strict lines may be: the vertex shader run
  as compute captures every vertex's clip position first, then a triangle
  list of six vertices a segment runs the vertex shader for each corner's
  end and moves it by half the width, up or down where the line runs more
  across than up, else sideways, which the other end's captured position
  says. Segments across a restart go. The same limits as for points.
- **Line rasterization modes** (VK_KHR_line_rasterization): Bresenham's
  lines are those parallelograms, and WebGPU's own at one pixel;
  rectangular lines the same quads moved along the line's normal instead,
  at any width, and smooth ones a pixel wider, the fragment shader's alpha
  multiplied by how much of the pixel is within half the width. A
  stippled line's quads carry where along the line each fragment is --
  from the start of a list's line, or of a strip back to its last
  restart -- and the fragment shader discards those whose bit of the
  pattern is clear.
- **Geometry shaders** (geometryShader) run as compute, which is all a
  WebGPU device has besides vertex and fragment stages: the vertex shader
  run as compute writes a record of its outputs a vertex of each instance
  -- a vec4 a slot, builtins then generic varyings -- which the geometry
  shader, run as compute an invocation a primitive of each instance and
  invocation of its, reads by the draw's topology's primitive assembly;
  EmitVertex writes its record out, at the invocation's next vertex, with
  the indices of what that vertex completes of a strip, as a list. Then a
  draw of that list, whose vertex shader is made of the geometry shader's
  outputs and reads the records. gl_PrimitiveID reaches the fragment
  shader as a flat varying; gl_Layer draws in each layer of a pass drawn a
  layer at a time. Each vertex stream has vertices and lists of its own,
  after all invocations' of the streams before; the draw is of the
  rasterized stream's. Indexed strips do not restart their primitives
  for it. Without one, topologies with adjacency
  are drawn without the adjacent vertices, as lists the driver makes; and
  gl_PrimitiveID of a fragment shader after a vertex shader is WebGPU's
  primitive_index, where the device has it -- half of it, of a point's or
  line's quad -- which Tint's IR reader has for SPIR-V's PrimitiveId, and
  its AST reader not. After tessellation, it is 0.
- **Tessellation** (tessellationShader) is compute too, around a
  tessellator of the same: the D3D11 reference Mesa's tessellator is, in
  WGSL (wvk_tessellator.c), an invocation a patch. The vertex shader run
  as compute writes its records; the control shader, a workgroup a patch
  and an invocation a vertex of its (barrier() the workgroup's), records
  of its vertices and patches; the tessellator counts each patch's domain
  points and indices, adds them up and writes them; the evaluation shader
  runs an invocation a point, as many as were made (an indirect dispatch),
  writing records the draw after reads, an indexed indirect draw of the
  tessellator's lists. Buffers are as big as a patch at level 64 would
  need, to 256 MB each; patches past that are not drawn. A geometry
  shader after reads the evaluation shader's records by the tessellator's
  lists, its invocations the tessellator's count too. The buffers of
  these draws, and of mesh and geometry shader draws, are the submit's,
  each draw's again after the one before -- zeroed, as new ones would be
  -- so a submit holds one draw's worth, not all of theirs.
- **Points a geometry or evaluation shader gives a size**
  (shaderTessellationAndGeometryPointSize) are quads, as the vertex
  shader's are: the draw after it, of six vertices an index, reads its
  records by the list, bound as storage; after tessellation, an indirect
  draw the tessellator writes the count of.
- **Pipeline statistics queries** (pipelineStatisticsQuery) are the
  driver's counts. The host knows most of each direct draw's from its
  counts and topology: its vertices and primitives, vertex shader
  invocations, a geometry shader's invocations, the control shader's
  patches, what clipping has without either, and a dispatch's invocations.
  An indirect command's come from the host's copy of it. The GPU counts
  the rest into a buffer of the query's: fragment shader invocations, by
  an atomic at each invocation's start; a geometry shader's primitives, as
  it makes its lists; and what the tessellator made, by a pass of the
  driver's. At the end of the query, a pass adds the two into the pool's
  memory. A pass replayed a layer at a time counts once.
- **Cull distances** (shaderCullDistance) are culled where primitives are
  made into lists: a geometry shader run as compute writes no indices for
  a primitive whose every vertex has one plane's distance negative. A
  draw without a geometry shader, whose vertex or evaluation shader writes
  cull distances, gets one of the driver's that passes each primitive
  through, made of that shader's outputs. Its draw's vertex shader writes
  none. The clip and cull distances a fragment shader reads, which
  WebGPU's have no builtin of, are vec4 varyings at locations it has no
  inputs at, written by the stage before as well.
- **A vertex shader's storage writes** (vertexPipelineStoresAndAtomics)
  are a compute shader's: WebGPU's vertex shaders write no storage. A
  draw whose vertex shader writes buffers or images first dispatches the
  shader made compute -- an invocation a vertex of each instance, the
  vertex an index of the index buffer where the draw is indexed and it
  does not restart, every attribute pulled -- with the draw's sets and the
  pass ended and begun again after; then draws with the vertex shader
  without its writes, whose atomics give the value as it is. So a vertex
  shader's outputs do not see what its own atomics return. An indirect
  draw's writes are of the commands as the host has them.
- **Storage images without a format** have the format of the view bound
  there: WGSL's storage textures declare theirs, so a draw or dispatch
  compiles a variant of the shader for the views' formats, as it does for
  the rest of its state (compute shaders with such images keep their
  source for it too). Until one is bound, the 32-bit one-channel format
  of the image's type. BGRA8, which SPIR-V has no format for, is declared
  RGBA8 to ntv and renamed in the WGSL. An image only read is a texture,
  and needs none.
- **A dispatch that samples what it writes as storage** -- a mip chain
  built a level at a time, sampling the whole chain -- samples a view
  that stops below the lowest level written where it can, which keeps
  every lod where it was, and a copy made before the dispatch where it
  cannot. WebGPU has a sampled and a written use of one level in no usage
  scope together; Vulkan asks only that no texel be both. A storage image
  of a view of several levels binds its base level, which is all Vulkan
  reads or writes of it and all WebGPU will bind as storage.
- **Barriers WGSL cannot prove uniform.** WGSL refuses a barrier under a
  branch its uniformity analysis cannot show the whole workgroup takes,
  and the analysis is conservative: a load from a read-write storage
  buffer is never uniform to it, nor any component of a vector that
  `gl_LocalInvocationID` went into. For a module Tint refuses for that
  alone, the condition of each `if` around a barrier, and of each exit
  from a loop that has one, is broadcast through a workgroup variable
  and `workgroupUniformLoad`. That is uniform for WGSL, and in a valid
  Vulkan program the condition was uniform already
  (`src/wink/compiler/gex_wgpu_tint.cpp`). `GEX_WGPU_UNIFORM_LOG=1` says
  why a module stayed refused.
- **Views in other formats.** WebGPU views a texture only as its format
  or its sRGB twin. A view of another format of the same texel size, or an
  uncompressed view of a compressed image (a texel a block), is of another
  texture of the image's, in that format. The image keeps, at each level,
  which of its textures are up to date, and brings one up to date with
  copies through a buffer when a draw, a pass or a transfer uses it.
  A compressed image whose size is not whole blocks is sampled from a
  shadow of its own size, decoded from it by a pass when sampled after a
  write.
- **Formats from `texture-formats-tier1` and `-tier2`**, where the device
  has them: the 16-bit normalized formats as textures (they're vertex
  formats regardless), rendering to 8-bit snorm, storage for most 8- and
  16-bit and 10-bit packed formats (so `shaderStorageImageExtendedFormats`),
  and read-write storage for R8, R16, RGBA8, RGBA16 and RGBA32 formats.
- **16-bit packed formats** (565, 4444, 5551, 1555, and the extension ones:
  `VK_EXT_4444_formats`, maintenance5's A1B5G5R5), which WebGPU does not
  have, are RGBA8 textures. Copies to and from buffers are a shader's,
  which packs and unpacks the texels; a format without alpha keeps its
  texture's at one. Blending is at 8 bits, not the format's.
- **Multi-planar formats and YCbCr conversions** (samplerYcbcrConversion):
  the 2- and 3-plane 4:2:0, 4:2:2 and 4:4:4 formats of 8, 10, 12 and 16
  bits (the last three with texture-formats-tier1). An image of one is an
  image of each plane, of the plane's format and size (R10X6 and the like
  are 16-bit normalized textures), bound to its part of the memory or,
  disjoint, to its own. Copies of a plane are the plane image's, and so is
  a view of one. A view of the whole has a view of each plane. A combined
  image sampler whose immutable sampler has a conversion takes a texture
  binding for each plane after the first, and a sampler of the chroma
  filter for them. The runtime's nir_vk_lower_ycbcr_tex samples each plane
  and converts, before the shader's source is kept. Of optimal tiling,
  they have layers as any other image does (ycbcrImageArrays).
- **Memory is one WGPUBuffer**, which the buffers bound to it share. WebGPU
  lets a buffer written as storage have no other use in a usage scope --
  a dispatch, or a whole render pass -- which suballocated memory has all
  the time: a uniform buffer and a storage buffer of one allocation. What
  a draw or dispatch reads of a buffer it writes, it reads from a copy
  made before it; and a draw at odds with what its pass used before
  begins the pass again.
- **Arrays of descriptors** are bindings of their own. A dynamic or
  non-uniform index chooses among them in the shader: reads from each,
  selected; writes only where the index is.
- **Mutable descriptors** (mutableDescriptorType): every descriptor is one
  struct already, and a binding's WebGPU binding is of the kind the
  shader's use of it says. A mutable one keeps the type written or copied
  there last, whose record it has. Not aliased: one WebGPU binding an
  element, which two variables of different kinds cannot share -- which
  only descriptor indexing's arrays, which WebGPU has too few bindings
  for, need.
- **Triangle fans** are the lists of their triangles, each in the order
  Vulkan's fan has it: a list the host makes, or for a draw by index one a
  compute shader makes of the application's indices, restarting where
  they do. **Polygon modes** of lines and points are the same shader's
  lists of each triangle's edges or corners. Indirect draws of fans are drawn from the host's copy of their
  commands. **vkCmdDraw*IndirectCount** copies the draws and gives those
  past the count no instances, on the GPU.
- **Lists that restart** (primitiveTopologyListRestart), which WebGPU's
  do not: the same shader makes a list of their whole primitives, the
  restart index to its end, and after it an indirect draw of the whole
  ones; points and wide lines drawn as quads leave out those of the
  restart index. Patch lists do not restart.
- **Stencil masks and references of each face**: WebGPU has one of each.
  Where the faces' differ and both are drawn, a draw is drawn twice, each
  culling the other face.
- **Viewports**: WebGPU has one. A vertex shader that writes
  gl_ViewportIndex has each draw drawn once per viewport, with what is not
  that draw's out of the clip volume, as layers are.
- **Passes of several layers or views** are replayed once for each: a
  WebGPU pass renders one layer. A vertex shader's `gl_Layer` chooses by
  moving vertices of other layers out of the clip volume, and the view
  index is the pass's. Input attachments are fetched at the layer drawn.
- **Timestamps** are the beginnings of empty compute passes, WebGPU having
  them only where a pass begins or ends; a render pass is ended for one and
  begun again. An occlusion query a pass is ended in goes on in the next,
  its parts added, and clears are not counted.
- **Blits and `vkCmdClearAttachments`** come from the runtime's vk_meta.
  Its passes of several layers are begun again at each.
- **Host image copies** (hostImageCopy) are the transfer commands of a
  command buffer of the device's, submitted and waited for: host memory
  goes through a buffer of host-visible memory, and a memcpy copy's layout
  is a buffer copy's of the whole subresource, packed.
- **Depth clip enable and control**: WebGPU clamps depth always, and clips
  it but where it is unclipped (depthClipEnable); clip space from -w to w
  is the viewport transform's depth, (z + w) / 2 first.
- **Pipeline robustness**: robustness 2's bounds (below) where a
  pipeline's robustness, or the device's features, ask for them, and
  WebGPU's own otherwise; a queue's global priority is medium, the one
  there is.
- **Buffers bounded by their descriptors** (robustBufferAccess2,
  nullDescriptor): where a pipeline's robustness says so, each word a
  shader loads, stores or does an atomic on of a uniform or storage
  buffer is checked against the range its descriptor's record has --
  zero for a null one -- and one past it is zero, or not written; texel
  buffers were bounded by their elements already, and read (0, 0, 1) for
  channels their format has not. Memory has a uniform binding's padding
  then, as the shader's block may be bigger than what is bound. Null
  descriptors are the null buffer's, a writable one's of its own. The CTS
  tests these only of implementations without VK_KHR_portability_subset
  (WINK_VK_NO_PORTABILITY hides it).
- **Images bounded by their views** (robustImageAccess2, and null image
  descriptors): a fetch, load, store or atomic of a texel out of the view,
  or of a null view, is checked in the shader -- a load reading zero, and
  alpha one for a format without it; a store or atomic nothing -- as is
  one of a channel the view's format has not (the record's channels).
  Storage images only stored to are write-only, which WebGPU has of more
  formats than read-write. What stays wrong: RG32 storage images both
  read and written, which WebGPU cannot have.
- **The provoking vertex** (provokingVertexLast): WebGPU's flat varyings
  are the first vertex's, so a draw whose provoking vertex is the last is
  drawn with each primitive's vertices rotated, the last first -- as its
  own lists of them, the host's or a compute shader's for strips, fans and
  indexed draws.
- **VK_KHR_maintenance8 and 9**: an offset not constant, which WGSL's
  are, is the coordinates moved -- by the texels of the level of an
  explicit LOD; OpSMod is the remainder made of the divisor's sign;
  copies between a depth or stencil aspect and a colour image go through
  a buffer as the aspect's buffer copies do, and depth24's unorm comes
  back from its float as the one that made it. Events, barriers and
  queue family ownership are nothing to one in-order queue.
- **Rendering single-sampled attachments multisampled**
  (VK_EXT_multisampled_render_to_single_sampled): a pass that does has
  each such attachment's view a multisampled image of its own, of the
  view's level and layers, which it renders to instead, and which is
  resolved into the view at its end -- averaged, or as the attachment
  says -- and filled from the view first, every sample alike (vk_meta's
  blit, which the runtime lets write a multisampled image), where the pass
  loads or its render area is less than all of it: WebGPU's resolves are
  of all of an attachment. Of 4 samples, which is all WebGPU has, and of
  formats WebGPU multisamples.
- **VK_KHR_maintenance10**: a resolve of another mode, or of an sRGB
  format's encoded values -- WebGPU's average the linear ones -- is
  vk_meta's, of vkCmdResolveImage2 or of a pass whose attachment says so;
  depth and stencil copy on the compute family, which is the one WebGPU
  queue too.
- **Smaller ones**: a pipeline's executables (pipelineExecutableInfo)
  are its shaders' WGSL, which is what there is to show; depth clamps to
  [0, 1] as Dawn does anyway (depthClampZeroOne); workgroup memory of an
  explicit layout is the words shared memory already is, each block at
  0; and an attribute the shader reads that the draw has none of reads
  (0, 0, 0, 1) (vertexAttributeRobustness), in a variant.
- **What WebGPU has no word for**: memory priorities (memoryPriority,
  pageableDeviceLocalMemory) are hints with nowhere to go; no image is
  compressed that WebGPU says (imageCompressionControl reports
  DISABLED); nothing is dithered, which legacyDithering allows; a device
  fault (deviceFault) is what WebGPU said when it lost the device; and a
  pipeline's identifier (pipelinePropertiesIdentifier) is of its shaders'
  cache keys. The subgroup ballot and vote of the EXT extensions are the
  KHR ones, of subgroups of one.
- **bfloat16 and 8-bit floats** (VK_KHR_shader_bfloat16,
  VK_EXT_shader_float8): their bits, which a shader only converts to and
  from 32-bit floats outside cooperative matrices -- integer arithmetic
  (wvk_nir_minifloats.c), rounding to nearest even as Mesa's own do --
  and cooperative matrices of them do their arithmetic in 32-bit floats,
  rounding once. Not dot products of bfloat16.
- **Cooperative matrices** (VK_KHR_cooperative_matrix), of a subgroup
  that is one invocation: all of a matrix is that invocation's, an array
  of its elements row by row (wvk_nir_cmat.c), and each operation is of
  each element in turn, unrolled -- a multiply-add each element's dot
  product, and C added saturating where the operation says. 8x8x8, of
  16- and 32-bit floats and 8-, 16- and 32-bit integers.
  Loops of the operations over the arrays come out wrong on Metal but for
  32-bit floats, which is why they are unrolled, and why the matrices are
  small.
- **FMA and contraction** (VK_KHR_shader_fma): OpFmaKHR is WGSL's fma,
  which Metal's is, fused -- for 16- and 32-bit floats; 64-bit ones are
  software that multiplies and then adds. A shader with a float operation
  not to be contracted (NoContraction, or float controls 2 without
  AllowContract) is compiled with Dawn's strict math, natively, where
  Metal would fuse its multiply and add otherwise
  (`wink/patches/dawn-metal-strict-math-contract.patch`).
- **Untyped pointers** (shaderUntypedPointers): buffers and workgroup
  memory are words already; copies of memory become loads and stores
  while they are still derefs. Pointers only compared (OpPtrEqual) are of
  the buffer each is of, as variable ones are.
- **Mesh and task shaders** (meshShader, taskShader, meshShaderQueries),
  which WebGPU has no stages of, are compute, in passes of a draw: a
  setup, of the draw's workgroups -- its command's or its indirect
  arguments', none where a count or the predicate says so; a task shader,
  a workgroup a task workgroup, writing its payload and launch to buffers;
  a scan of the launches, which each mesh workgroup finds its task
  workgroup and grid in; the mesh shader, a workgroup a mesh workgroup,
  writing a record of each vertex and of each primitive, its indices and
  whether it is culled, its outputs' counts first -- of the slots it
  writes, packed, the first invocation's counts and launches the
  workgroup's; and a draw of every primitive each may have made, a vertex
  a corner, whose vertex shader reads those -- those not made, or culled,
  or culled by their cull distances, out of sight; points of a size are
  quads, as a vertex shader's are. Each pass's dispatch the one before's.
  The buffers are the submit's, 64 MB each at most: the mesh workgroups are
  in rounds of as many as they hold, a round's mesh shader and draw after
  the one before's -- as many rounds as a draw's workgroups take, or
  eight, where only the GPU knows how many it has. Not multiview, nor of
  shading rates; a line list whose lines meet on a pixel's edge colors it
  as Metal does.
- **Barycentric coordinates and inputs of each vertex**
  (fragmentShaderBarycentric), which WGSL has neither of, are the geometry
  shader's: a draw whose fragment shader reads them goes through one --
  the driver's pass-through where the pipeline has none, whose primitives'
  vertices are in the order they are rasterized in. The vertex shader
  after it draws its lists a vertex each, pulling its index from them:
  its corner of the primitive is its coordinate, a vec3 varying of each
  kind read, and the primitive's vertices' records are flat uvec4
  varyings of each slot read, which the fragment shader makes its inputs
  of each vertex of. Not of mesh shaders, which there are none of; and a
  line list's segments of the last provoking vertex are drawn from their
  end, whose flat varyings they have, which may cover an end's pixel
  otherwise.
- **Present** copies the swapchain image into the surface's current
  texture. WebGPU hands out a new texture every frame, and Vulkan's images
  have to stay.
  A present may change the swapchain's present mode, which reconfigures
  the surface, and signal a fence once its image may be used again, which
  is an empty submit after the copy's; images acquired and not presented
  may be released (VK_KHR_swapchain_maintenance1). Every mode is
  compatible with every other; nothing is scaled.
- **Legacy vertex attributes** (legacyVertexAttributes): attributes at
  offsets or of strides their components are not aligned to, or of
  another numeric type than the shader reads, are the vertex shader's to
  read itself, as those WebGPU cannot lay out are -- a channel that
  straddles two words of both, and the bits of the value its format makes
  as the type the shader reads.

## Building and testing, natively

Dawn with Metal on macOS, or Vulkan on Linux, from
`src/gallium/drivers/gex_wgpu/native/build-dawn.sh` and `package-dawn.sh`.
Then:

```sh
PKG=~/.cache/mesa-wink/dawn-native/pkg
meson setup build/wvk-native -Dbuildtype=debugoptimized -Dvulkan-drivers=wink \
  -Dgallium-drivers= -Dplatforms= -Dglx=disabled -Degl=disabled -Dopengl=false \
  -Dgles1=disabled -Dgles2=disabled -Dllvm=disabled -Dvideo-codecs= -Dtools= \
  -Dvulkan-layers= -Dgex-emdawn-dir=$PKG -Dgex-dawn-native-dir=$PKG/lib \
  -Dgex-tint-dir=$PKG
ninja -C build/wvk-native
export VK_DRIVER_FILES=$PWD/build/wvk-native/src/wink/vulkan/wink_devenv_icd.$(uname -m | sed s/arm64/aarch64/).json
vulkaninfo --summary
```

`test/build.sh OUT` builds the tests; they need glslangValidator, and SDL3
for the window test:

- `wvk_smoke`: memory, copies, clears, sync, and a headless swapchain.
- `wvk_draw`: draws, push constants, a negative viewport, textures,
  `vkCmdClearAttachments` and blits, all read back.
- `wvk_window [--resize] [--hidpi]`: presents to an SDL3 window.

vkcube runs as well; brew's binary is
`vulkan-tools/*/cube/vkcube.app/Contents/MacOS/vkcube`.

dEQP-VK: `test/build-cts.sh` builds it, and `test/run-cts.sh OUT GROUP...`
runs mustpass groups (`api`, `draw`, `texture`, …) with deqp-runner.
Anything after `--` goes to deqp-runner, such as `--fraction 25`.

The expectations come from a full run: `ci/wvk-fails.txt` and
`ci/wvk-flakes.txt`. OUT's `failures.csv` lists only what changed.
The full run of 2026-09-28 took 1 h 50 min on an Apple M4, with 7 jobs:

- 2,443,884 cases; 313,069 ran, and 2,130,815 were unsupported;
- 98.5% of what ran passed;
- 4,799 failed, 9 crashed and 2 timed out.

The biggest failing families:

| failures | family |
| --- | --- |
| 1,371 | `descriptorset_random` (more than 10 storage buffers per stage) |
| 466 | `image.mutable` |
| 441 | `copy_and_blit` |
| 300 | `memory_model.message_passing` |
| 292 | `pipeline.monolithic`, mostly blending |
| 252 | `spirv_assembly` |

Minecraft: `test/run-minecraft.py INSTANCE [--world NAME] [--capture N]
[--seconds S]` runs a Prism Launcher instance offline, on this build's ICD
through brew's loader, with `--graphicsBackend vulkan`. It uses a scratch game
directory, with copies of the instance's options and the world. 26.3 and
26.4-snapshot-1 render their title screens and worlds. `--gl` runs the
OpenGL backend on wink's GL instead (`wink/README.md`).

## In a browser

`wink/build.sh` builds it into libwink. There is no loader, so an app
gets everything from `wink_vk_get_instance_proc_addr`, and a surface for
a canvas from `wink_vk_create_surface(instance, "#canvas", &surface)`.
The only instance extension it needs is `VK_KHR_surface`. The wink fork of
SDL does both when it is built with wink, so `SDL_Vulkan_*` just work.

Waiting suspends the wasm stack through JSPI. So the thread that calls
Vulkan has to be entered through `WebAssembly.promising`, the way .NET
enters its exports, on the main thread or on a worker that holds the
canvas as an OffscreenCanvas. The swapchain is the canvas's size, and
presenting ends the task: FIFO waits there for the next frame.

`wink/build.sh --test` runs `wvk_smoke` and `wvk_draw` in headless Chrome,
with a generated shim standing in for the loader (`wink/test/vk`). By
hand:

```sh
wink/test/vk/build-vk.sh st build/wink/out/st "$(command -v emcc)" OUT
node wink/test/run-browser.mjs OUT --page='vk.html?test=wvk_draw'
```

## Environment

| variable | effect |
| --- | --- |
| `WINK_VK_DUMP_WGSL=1` | print every shader's WGSL |
| `WINK_VK_CAPTURE=N:file.bmp` | write what the Nth present shows, for machines where nothing can capture the screen |
| `WINK_VK_ABORT_ON_ERROR=1` | abort on a WebGPU validation error, which is a driver bug |
| `WINK_VK_FALLBACK_ADAPTER=1` | use Dawn's software adapter |
| `WINK_VK_API_VERSION` | the Vulkan version reported |
| `WINK_VK_NO_PORTABILITY=1` | hide VK_KHR_portability_subset, for CTS groups that skip on implementations of it (robustness2): testing only |
| `WINK_VK_VARIANT_LOG=1` | say when a shader variant is compiled for a draw, and why |
| `WINK_VK_DAWN_TOGGLES=a,b` | enable Dawn toggles, and print Dawn's log (at `MESA_LOG_LEVEL=info`): `dump_shaders` for the MSL |
| `GEX_WGPU_SPIRV_DIR=dir` | write every module to `dir/N.spv` before Tint reads it, for one Tint does not come back from |
| `GEX_WGPU_UNIFORM_LOG=1` | say why a module refused for a barrier in non-uniform control flow could not be made uniform |

## Not there yet

- Depth clamp clamps to [0, 1], not to the viewport's depth range: Dawn on
  Metal clamps at the near and far planes. A narrow or inverted range with
  depth clamp is off where geometry is past it.
- WebGPU has 10 storage buffers a stage in browsers, and texel buffers are
  storage buffers, beyond what Vulkan counts as such; so are storage images
  a shader does atomics on.
- Vulkan 1.3 is reported, but for `bufferDeviceAddress`: WebGPU has no
  pointers.
- Descriptor indexing's features, each of them, but not
  `VK_EXT_descriptor_indexing`: that asks for 500000 descriptors a stage,
  where WebGPU has a few dozen bindings.
- 64-bit floats as vertex inputs: WebGPU has none.
- Line rasterization modes, stipples and widths of lines with adjacency,
  which are WebGPU's one-pixel lines; and Bresenham's lines on a
  multisampled pass, whose quads cover samples, not whole pixels.
- RG32 storage images both read and written: WebGPU's read-write storage
  textures are of one-channel 32-bit formats and tier2's, which RG32 is
  not. And atomics of 64-bit floats, of 16-bit floats, and of 64-bit
  images: WGSL's are of 32-bit words, and WebGPU has no 64-bit formats.
- Scalar 8-bit division and remainder, and 8-bit specialization
  constants, fail a few CTS cases not yet understood.
- A return before a barrier, taken by some invocations: WGSL refuses the
  barrier, and broadcasting the return's condition would send the rest
  of the workgroup home too.
- Blender's EEVEE renders, but about 0.6% of pixels have no G-buffer (the
  same pixels in the normal, colour and AO passes, in a pattern that
  repeats every 32 pixels; depth matches Metal), and the world lights
  the scene about 20% more brightly than on Metal.
- Device-scope memory ordering between workgroups: WGSL's barriers are
  the workgroup's, and none can be in non-uniform control flow, so a
  memory barrier there is dropped.
- An attribute larger than its binding's dynamic stride, which WebGPU
  cannot describe. Strips of 16-bit indices with primitive restart off
  still restart at 0xFFFF, as WebGPU's always do.
- Precise occlusion queries: WebGPU's occlusion queries are boolean on
  Dawn's Metal and D3D12 backends, and not precise on Vulkan.
- Pipeline statistics of indirect draws and dispatches in memory the host
  has no copy of, which are not counted.
- Subgroups are one invocation each: Tint reads no subgroup operations from
  SPIR-V. Every operation but the quad ones is there, trivially.
- Blend states whose colour factors use both CONSTANT_COLOR and
  CONSTANT_ALPHA, on a multisampled pass: WebGPU has one constant, and
  the fragment shader's blending reads single-sampled copies.
- Views of multisampled images in formats other than the image's, or its
  sRGB twin: other views are of a texture of the view's format, which
  copies through a buffer keep up to date, and WebGPU copies no
  multisampled texture.
- Gathers from integer cube maps take one texel four times: each face's
  2x2 footprint across its edges is not worked out.
- Lines and points of triangles are not culled: which way a triangle
  faces is the vertex shader's to say, after the lists are made.
- The single-plane 4:2:2 formats (G8B8G8R8_422 and the like), which WebGPU
  has nothing to sample as; and storage of a whole multi-planar image,
  rather than of its planes' views.
- Images bound to the same memory share nothing but it: an optimal image's
  texels are its texture's, so aliasing images see each other's writes only
  through copies (YCbCr's memory_alias cases among them).
- Multisampled array images are a texture for each layer, since WebGPU
  has no multisampled arrays: attachments, clears, copies and resolves
  (vk_meta's, for formats WebGPU does not resolve, a layer at a time) are
  right, but a shader sampling one sees its first layer.
- Copies of part of a multisampled depth or stencil image, or of one of
  its aspects: WebGPU copies multisampled textures whole, and a buffer
  holds one sample.
- Feedback loops on multisampled depth or stencil that is written.
- Formats WebGPU does not multisample (RG32 and RGBA32 among them) have
  one sample, though Vulkan wants every colour attachment format to have
  framebufferColorSampleCounts.

Some of these are what WebGPU has no way to express. The rest are work still
to do.
