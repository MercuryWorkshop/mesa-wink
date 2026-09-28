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
// With WINK_INIT_GL_ON_MAIN_THREAD the driver runs on the main thread, in
// jobs a worker queues and wink_js_kick announces. The main thread's runner
// below enters them through WebAssembly.promising itself, so the embedder has
// nothing to wrap there; and it runs them on a shadow stack of its own. That
// matters because a suspended job's frames sit on the C stack while the event
// loop runs other wasm -- the embedder's own suspending calls among them --
// and two suspended stacks interleaved on one shadow stack corrupt it. So
// while a job is suspended the stack pointer is the main stack's, and while it
// runs it is the runner's.
//
// Every page-facing knob is a Module property, bracketed so closure keeps it:
//   Module.winkDevice          a GPUDevice to use (or preinitializedWebGPUDevice)
//   Module.winkAdapterOptions  passed to navigator.gpu.requestAdapter
//   Module.winkDeviceDescriptor(adapter, desc)  may amend or replace desc
//   Module.winkCanvas          a canvas, or function(selector) -> canvas; on a
//                              worker, the worker's own Module
//   Module.winkEnv             {NAME: value} or ["NAME=value"], applied as setenv
//   Module.winkRunnerStackSize the main-thread runner's stack (default 16MB)

var LibraryWink = {
  $WINK__deps: ['stackSave', 'stackRestore', 'malloc', 'wink_main_run_queue', 'wink_set_device_lost'],
  $WINK: {
    waiting: [],
    lost: false,
    errors: 0,

    // The main thread's runner. `pending` is made on first use: a library
    // object is written out as a literal, and a Set does not survive that.
    pending: null,
    running: false,
    scheduled: false,
    inRunner: false,
    runnerTop: 0,
    outerSP: 0,
    promisingRun: null,

    schedule: function () {
      if (WINK.running || WINK.scheduled) return;
      WINK.scheduled = true;
      // A fresh task-free stack: no wasm frames under the promising call.
      queueMicrotask(WINK.drain);
    },

    drain: async function () {
      WINK.scheduled = false;
      WINK.running = true;
      try {
        while (WINK.pending && WINK.pending.size) {
          var queues = Array.from(WINK.pending);
          WINK.pending.clear();
          for (var i = 0; i < queues.length; ++i) await WINK.runQueue(queues[i]);
        }
      } catch (e) {
        err('wink: main-thread runner failed: ' + (e && e.stack || e));
      } finally {
        WINK.running = false;
      }
    },

    runQueue: async function (queue) {
      if (!WINK.promisingRun) {
        var size = Module['winkRunnerStackSize'] || 16 * 1024 * 1024;
        var base = _malloc(size);
        if (!base) throw new Error('no memory for the runner stack');
        WINK.runnerTop = (base + size) & ~15;
        WINK.promisingRun = WebAssembly.promising(wasmExports['wink_main_run_queue']);
      }
      WINK.outerSP = stackSave();
      stackRestore(WINK.runnerTop);
      WINK.inRunner = true;
      try {
        await WINK.promisingRun(queue);
      } finally {
        WINK.inRunner = false;
        stackRestore(WINK.outerSP);
      }
    },

    // What every suspending import returns. Inside the runner the stack
    // pointer is handed back to the main stack for the duration, and taken
    // back just before the job resumes.
    suspend: function (promise) {
      if (!WINK.inRunner) return promise;
      var mine = stackSave();
      stackRestore(WINK.outerSP);
      return promise.then(function (value) {
        WINK.outerSP = stackSave();
        stackRestore(mine);
        return value;
      });
    },

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
        _wink_set_device_lost();
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
    return WINK.suspend(new Promise(function (resolve) { WINK.waiting.push(resolve); }));
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
    return WINK.suspend((async function () {
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
    })());
  },
  wink_js_acquire_device__postset:
    "if (typeof WebAssembly.Suspending == 'function') _wink_js_acquire_device = new WebAssembly.Suspending(_wink_js_acquire_device);",

  // wgpuInstanceCreateSurface's work, with a wider idea of where the canvas
  // is: a worker has no document to query, and an embedder may already hold
  // the canvas (or its OffscreenCanvas) rather than a selector for it.
  //
  // A worker's canvas is one pthread_create transferred to it
  // (-sOFFSCREENCANVAS_SUPPORT, emscripten_pthread_attr_settransferredcanvases).
  // Emscripten files those under GL.offscreenCanvases by the element's DOM id,
  // whatever selector named them, and drops them unless $GL is linked in --
  // hence the dependency. A worker that owns exactly one takes it for any
  // selector: ikvmcraft transfers ".canvas", for one.
  wink_js_create_surface__deps: ['$WebGPU', 'emwgpuCreateSurface', '$UTF8ToString',
#if OFFSCREENCANVAS_SUPPORT
    '$GL',
#endif
  ],
  wink_js_create_surface: function (selectorPtr) {
    var selector = selectorPtr ? UTF8ToString(selectorPtr) : '';
    var provided = Module['winkCanvas'];
    var canvas = null;
    if (typeof provided == 'function') canvas = provided(selector);
    else if (provided && !selector) canvas = provided;
    var owned = typeof GL != 'undefined' && GL.offscreenCanvases || {};
    if (!canvas && selector) {
      var id = selector.charAt(0) == '#' ? selector.substr(1) : selector;
      if (owned[id]) canvas = owned[id].offscreenCanvas;
    }
    if (!canvas && (!selector || selector == '#canvas')) canvas = Module['canvas'];
    if (!canvas && selector && typeof document != 'undefined') canvas = document.querySelector(selector);
    if (!canvas && typeof document == 'undefined') {
      var ids = Object.keys(owned).filter(function (k) { return owned[k]; });
      if (ids.length == 1) canvas = owned[ids[0]].offscreenCanvas;
    }
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

  // From any thread: the runner lives on main.
  wink_js_kick__deps: ['$WINK'],
  wink_js_kick__proxy: 'async',
  wink_js_kick__sig: 'vp',
  wink_js_kick: function (queue) {
    if (!WINK.pending) WINK.pending = new Set();
    WINK.pending.add(queue);
    WINK.schedule();
  },

  // A presented frame needs the event loop; pacing is waiting for more of it.
  // The setTimeout fallback is for hidden tabs, where requestAnimationFrame
  // never fires and a job waiting on it would park the runner for good.
  wink_js_yield_frame__deps: ['$WINK'],
  wink_js_yield_frame: function (frames) {
    return WINK.suspend(new Promise(function (resolve) {
      var left = frames;
      var settled = false;
      var done = function () { if (!settled) { settled = true; resolve(); } };
      var tick = function () { if (--left <= 0) done(); else requestAnimationFrame(tick); };
      if (left > 0 && typeof requestAnimationFrame == 'function') requestAnimationFrame(tick);
      else setTimeout(done, 0);
      setTimeout(done, 1000);
    }));
  },
  wink_js_yield_frame__postset:
    "if (typeof WebAssembly.Suspending == 'function') _wink_js_yield_frame = new WebAssembly.Suspending(_wink_js_yield_frame);",
};

mergeInto(LibraryManager.library, LibraryWink);
