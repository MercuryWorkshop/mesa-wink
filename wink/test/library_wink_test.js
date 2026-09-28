// SPDX-License-Identifier: MIT
//
// Test-only: publish the raw wasm export of wink_smoke_main. The page wraps
// it in WebAssembly.promising, which accepts a wasm function and not the JS
// wrapper Module._wink_smoke_main is -- exactly what an embedder like the
// .NET runtime does for its own exports.
var LibraryWinkTest = {
  wink_test_publish__postset:
    "addOnPostRun(function () {" +
    "  var exports = typeof wasmExports != 'undefined' ? wasmExports : Module['asm'];" +
    "  Module['winkSmokeRaw'] = exports['wink_smoke_main'];" +
    "});",
  wink_test_publish: function () {},

  // A fresh task, so no JS frame sits between promising and the wasm body.
  wink_test_run_promising__deps: ['$UTF8ToString'],
  wink_test_run_promising: function (namePtr) {
    var name = UTF8ToString(namePtr);
    var exports = typeof wasmExports != 'undefined' ? wasmExports : Module['asm'];
    var run = WebAssembly.promising(exports[name]);
    setTimeout(function () {
      run()['catch'](function (e) { err('wink test: ' + name + ' failed: ' + (e && e.stack || e)); });
    }, 0);
  },
};
mergeInto(LibraryManager.library, LibraryWinkTest);
