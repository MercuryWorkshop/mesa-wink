// SPDX-License-Identifier: MIT
//
// Test-only: publish the raw wasm export, which the page wraps in
// WebAssembly.promising (it takes a wasm function, not Emscripten's JS
// wrapper around one).
var LibraryVkTest = {
  wvk_test_publish__postset:
    "addOnPostRun(function () {" +
    "  var exports = typeof wasmExports != 'undefined' ? wasmExports : Module['asm'];" +
    "  Module['wvkTestRaw'] = exports['wvk_browser_main'];" +
    "});",
  wvk_test_publish: function () {},
};
mergeInto(LibraryManager.library, LibraryVkTest);
