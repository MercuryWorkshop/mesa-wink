// SPDX-License-Identifier: MIT
addOnPostRun(function () {
  var exports = typeof wasmExports !== 'undefined' ? wasmExports : Module['asm'];
  Module['winkSkiaRaw'] = exports['wink_skia_main'];
});
