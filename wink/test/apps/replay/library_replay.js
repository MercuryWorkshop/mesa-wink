// SPDX-License-Identifier: MIT
var LibraryReplay = {
  wink_replay_publish__postset:
    "addOnPostRun(function () {" +
    " var exports = typeof wasmExports != 'undefined' ? wasmExports : Module['asm'];" +
    " Module['winkReplayRaw'] = exports['__main_argc_argv'] || exports['main'];" +
    "});",
  wink_replay_publish: function () {},
};
mergeInto(LibraryManager.library, LibraryReplay);
