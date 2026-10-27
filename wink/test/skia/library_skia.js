// SPDX-License-Identifier: MIT
mergeInto(LibraryManager.library, {
  wink_skia_schedule__deps: ['wink_skia_worker_failed'],
  wink_skia_schedule: function () {
    var exports = typeof wasmExports !== 'undefined' ? wasmExports : Module['asm'];
    setTimeout(function () {
      WebAssembly.promising(exports['wink_skia_worker_body'])().catch(function (e) {
        err('WINK_SKIA_CONTEXT_FAILED: ' + e.stack);
        _wink_skia_worker_failed();
      });
    }, 0);
  },
});
