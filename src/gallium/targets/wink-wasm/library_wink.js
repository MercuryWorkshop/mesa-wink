// SPDX-License-Identifier: MIT
//
// wink's JavaScript half, for --js-library. Written against Emscripten
// 3.1.56's library format (mergeInto), which .NET 10 links with.
//
// JSPI is applied by hand, not by -sJSPI/-sASYNCIFY: the imports that wait
// are wrapped in WebAssembly.Suspending here, and the embedder wraps the
// export that leads to them in WebAssembly.promising. That is the arrangement
// ikvm-wasm-build's .NET runtime already uses for its frame pacing, and it
// needs nothing from the link.
//
// Every page-facing knob is a Module property, bracketed so closure keeps it:
//   Module.winkDevice          a GPUDevice to use (or preinitializedWebGPUDevice)
//   Module.winkAdapterOptions  passed to navigator.gpu.requestAdapter
//   Module.winkDeviceDescriptor(adapter, desc)  may amend or replace desc
//   Module.winkCanvas          a canvas, or function(selector) -> canvas
//   Module.winkEnv             {NAME: value} or ["NAME=value"], applied as setenv

var LibraryWink = {
  $WINK: {
    waiting: [],
    lost: false,
    errors: 0,

    wakeAll: function () {
      var waiting = WINK.waiting;
      if (!waiting.length) return;
      WINK.waiting = [];
      for (var i = 0; i < waiting.length; ++i) waiting[i]();
    },

    // What the driver can use, requested only when the adapter has it; the
    // driver asks the device what it got rather than assuming. The list and
    // the reasons for each limit below are gex's (web/src/mesa-gl.js).
    features: [
      'texture-compression-bc', 'texture-compression-bc-sliced-3d',
      'texture-compression-etc2',
      'texture-compression-astc', 'texture-compression-astc-sliced-3d',
      'depth32float-stencil8', 'rg11b10ufloat-renderable', 'depth-clip-control',
      'float32-filterable', 'float32-blendable', 'clip-distances',
      'dual-source-blending', 'timestamp-query', 'indirect-first-instance',
    ],
    // Raised to this, clamped to the adapter: WebGPU's defaults are far below
    // what adapters offer and the driver sizes its GL caps from what it was
    // granted -- maxTextureDimension2D alone decides between GL 4.0 and 4.3.
    limitsUp: {
      'maxUniformBuffersPerShaderStage': 16,
      'maxInterStageShaderVariables': 32,
      'maxVertexAttributes': 32,
      'maxStorageBuffersPerShaderStage': 24,
      'maxStorageBuffersInVertexStage': 24,
      'maxStorageBuffersInFragmentStage': 24,
      'maxStorageTexturesPerShaderStage': 8,
      'maxStorageTexturesInVertexStage': 8,
      'maxStorageTexturesInFragmentStage': 8,
      'maxDynamicUniformBuffersPerPipelineLayout': 32,
      'maxDynamicStorageBuffersPerPipelineLayout': 16,
      'maxSampledTexturesPerShaderStage': 16,
      'maxColorAttachmentBytesPerSample': 128,
      'maxComputeInvocationsPerWorkgroup': 1024,
      'maxComputeWorkgroupSizeX': 1024,
      'maxComputeWorkgroupSizeY': 1024,
      'maxComputeWorkgroupSizeZ': 64,
      'maxComputeWorkgroupStorageSize': 32768,
      'maxTextureDimension1D': 16384,
      'maxTextureDimension2D': 16384,
      'maxTextureDimension3D': 2048,
      'maxTextureArrayLayers': 2048,
    },
    // Alignments run the other way: lower is better, and 32 is WebGPU's floor.
    limitsDown: {
      'minUniformBufferOffsetAlignment': 32,
      'minStorageBufferOffsetAlignment': 32,
    },

    descriptor: function (adapter) {
      var requiredFeatures = WINK.features.filter(function (f) { return adapter['features']['has'](f); });
      var requiredLimits = {};
      var have = adapter['limits'];
      var short = [];
      for (var up in WINK.limitsUp) {
        if (typeof have[up] != 'number') continue;
        requiredLimits[up] = Math.min(have[up], WINK.limitsUp[up]);
        if (have[up] < WINK.limitsUp[up]) short.push(up + ' ' + have[up]);
      }
      for (var down in WINK.limitsDown) {
        if (typeof have[down] != 'number') continue;
        requiredLimits[down] = Math.max(have[down], WINK.limitsDown[down]);
      }
      if (short.length) err('wink: adapter limits below what the driver can use: ' + short.join(', '));
      return {'requiredFeatures': requiredFeatures, 'requiredLimits': requiredLimits};
    },

    adopt: function (device) {
      device['lost']['then'](function (info) {
        WINK.lost = true;
        err('wink: WebGPU device lost: ' + info['reason'] + ': ' + info['message']);
        // A waiter whose map will now never complete is woken to see the loss.
        WINK.wakeAll();
      });
      device['addEventListener']('uncapturederror', function (e) {
        if (WINK.errors++ < 32) err('wink: WebGPU error: ' + e['error']['message']);
      });
      return WebGPU.importJsDevice(device);
    },
  },

  // The driver's only wait: every readback maps a staging buffer and loops
  // on this until the map callback has run. A list of waiters, not one: a
  // callback that starts another readback suspends again inside the first,
  // and every waiter re-checks its own flag on waking.
  gex_wgpu_jspi_wait__deps: ['$WINK'],
  gex_wgpu_jspi_wait: function () {
    if (WINK.lost) return;
    return new Promise(function (resolve) { WINK.waiting.push(resolve); });
  },
  gex_wgpu_jspi_wait__postset:
    "if (typeof WebAssembly.Suspending == 'function') _gex_wgpu_jspi_wait = new WebAssembly.Suspending(_gex_wgpu_jspi_wait);",

  gex_wgpu_jspi_signal__deps: ['$WINK'],
  gex_wgpu_jspi_signal: function () {
    WINK.wakeAll();
  },

  wink_js_has_jspi: function () {
    return (typeof WebAssembly.Suspending == 'function' &&
            typeof WebAssembly.promising == 'function') ? 1 : 0;
  },

  wink_js_acquire_device__deps: ['$WINK', '$WebGPU', 'emwgpuCreateDevice', 'emwgpuCreateQueue'],
  wink_js_acquire_device: function () {
    var given = Module['winkDevice'] || Module['preinitializedWebGPUDevice'];
    if (given) return WINK.adopt(given);
    var gpu = typeof navigator != 'undefined' ? navigator['gpu'] : undefined;
    if (!gpu) {
      err('wink: navigator.gpu is unavailable');
      return 0;
    }
    return (async function () {
      try {
        var adapter = await gpu['requestAdapter'](Module['winkAdapterOptions'] || {'powerPreference': 'high-performance'});
        if (!adapter) {
          err('wink: no WebGPU adapter');
          return 0;
        }
        var desc = WINK.descriptor(adapter);
        if (Module['winkDeviceDescriptor']) desc = Module['winkDeviceDescriptor'](adapter, desc) || desc;
        return WINK.adopt(await adapter['requestDevice'](desc));
      } catch (e) {
        err('wink: requesting a WebGPU device failed: ' + e);
        return 0;
      }
    })();
  },
  wink_js_acquire_device__postset:
    "if (typeof WebAssembly.Suspending == 'function') _wink_js_acquire_device = new WebAssembly.Suspending(_wink_js_acquire_device);",

  // wgpuInstanceCreateSurface's work, with a wider idea of where the canvas
  // is: a worker has no document to query, and an embedder may already hold
  // the canvas (or its OffscreenCanvas) rather than a selector for it.
  wink_js_create_surface__deps: ['$WebGPU', 'emwgpuCreateSurface', '$UTF8ToString'],
  wink_js_create_surface: function (selectorPtr) {
    var selector = selectorPtr ? UTF8ToString(selectorPtr) : '';
    var provided = Module['winkCanvas'];
    var canvas = null;
    if (typeof provided == 'function') canvas = provided(selector);
    else if (provided && !selector) canvas = provided;
    if (!canvas && selector) {
      var id = selector.charAt(0) == '#' ? selector.substr(1) : selector;
      if (typeof GL != 'undefined' && GL.offscreenCanvases && GL.offscreenCanvases[id])
        canvas = GL.offscreenCanvases[id].offscreenCanvas;
      else if (typeof document != 'undefined')
        canvas = document.querySelector(selector);
    }
    if (!canvas && !selector) canvas = Module['canvas'];
    if (!canvas) {
      err('wink: no canvas' + (selector ? ' for ' + selector : ''));
      return 0;
    }
    var context = canvas.getContext('webgpu');
    if (!context) {
      // Most often the canvas already has a WebGL or 2D context.
      err('wink: the canvas will not give a webgpu context');
      return 0;
    }
    var ptr = _emwgpuCreateSurface(0);
    WebGPU.Internals.jsObjectInsert(ptr, context);
    return ptr;
  },

  wink_js_get_env__deps: ['$stringToUTF8', '$lengthBytesUTF8'],
  wink_js_get_env: function (buf, len) {
    var env = Module['winkEnv'];
    if (!env) return 0;
    var text = '';
    if (Array.isArray(env)) {
      for (var i = 0; i < env.length; ++i) text += env[i] + '\n';
    } else {
      for (var name in env) text += name + '=' + env[name] + '\n';
    }
    var need = lengthBytesUTF8(text) + 1;
    if (buf && len >= need) stringToUTF8(text, buf, len);
    return need;
  },

  wink_js_device_lost__deps: ['$WINK'],
  wink_js_device_lost: function () {
    return WINK.lost ? 1 : 0;
  },
};

mergeInto(LibraryManager.library, LibraryWink);
