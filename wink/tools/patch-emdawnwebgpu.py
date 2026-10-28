#!/usr/bin/env python3
"""Make the pinned emdawnwebgpu package work under Emscripten 3.1.56.

The package (Dawn cdda346, release v20250822.104650) was built against emsdk
4.0.8. Each edit below is applied independently and idempotently: text already
in its new form is left alone, text in its old form is replaced, and text in
neither form is an error -- a silently unpatched library is the failure this
exists to prevent.

The compatibility edits originated in MobileGL's
cmake/FetchEmdawnwebgpu.cmake; the API corrections are exercised by Wink.
"""
import os
import sys

EDITS = {
    'webgpu/src/library_webgpu.js': [
        # 3.1.56 spells the stack helpers without the '$' library prefix.
        ("errorCallback__deps: ['$stackSave', '$stackRestore', '$stringToUTF8OnStack']",
         "errorCallback__deps: ['stackSave', 'stackRestore', '$stringToUTF8OnStack']"),
        # The non-Asyncify stub must take the arguments its signature declares,
        # or 3.1.56's library processing rejects the mismatch.
        ('emwgpuWaitAny: () => {',
         'emwgpuWaitAny: (futurePtr, futureCount, timeoutMSPtr) => {'),
        # Objects are keyed by pointer, but _emwgpuCreate* returns reach JS as
        # signed i32: in a heap over 2 GB -- the .NET host's is about 4 GB --
        # inserts land under negative keys while lookups are >>> 0. Upstream
        # made jsObjectInsert and getJsObject unsigned; the import and delete
        # paths still are not.
        ('WebGPU.Internals.jsObjects[ptr] = obj;',
         'WebGPU.Internals.jsObjects[ptr >>> 0] = obj;'),
        ('delete WebGPU.Internals.jsObjects[ptr];',
         'delete WebGPU.Internals.jsObjects[ptr >>> 0];'),
        # The C API uses UINT32_MAX for an omitted pass timestamp. The JS
        # API needs undefined; forwarding the sentinel fails validation.
        ('makePassTimestampWrites: (ptr) => {',
         'optionalQuerySetIndex: (index) => index === 0xffffffff ? undefined : index,\n\n'
         '    makePassTimestampWrites: (ptr) => {'),
        ('"beginningOfPassWriteIndex": {{{ gpu.makeGetU32(\'ptr\', C_STRUCTS.WGPUPassTimestampWrites.beginningOfPassWriteIndex) }}}',
         '"beginningOfPassWriteIndex": WebGPU.optionalQuerySetIndex({{{ gpu.makeGetU32(\'ptr\', C_STRUCTS.WGPUPassTimestampWrites.beginningOfPassWriteIndex) }}})'),
        ('"endOfPassWriteIndex": {{{ gpu.makeGetU32(\'ptr\', C_STRUCTS.WGPUPassTimestampWrites.endOfPassWriteIndex) }}}',
         '"endOfPassWriteIndex": WebGPU.optionalQuerySetIndex({{{ gpu.makeGetU32(\'ptr\', C_STRUCTS.WGPUPassTimestampWrites.endOfPassWriteIndex) }}})'),
        # WGPU_DEPTH_CLEAR_VALUE_UNDEFINED is NaN. The WebIDL float conversion
        # rejects NaN even when depth is read-only or the format is stencil
        # only. Omit the JS dictionary value as the C sentinel requests.
        ('"depthClearValue": {{{ makeGetValue(\'dsaPtr\', C_STRUCTS.WGPURenderPassDepthStencilAttachment.depthClearValue, \'float\') }}}',
         '"depthClearValue": ((value) => Number.isNaN(value) ? undefined : value)({{{ makeGetValue(\'dsaPtr\', C_STRUCTS.WGPURenderPassDepthStencilAttachment.depthClearValue, \'float\') }}})'),
    ],
    'webgpu/src/library_webgpu_enum_tables.js': [
        # Upstream typo: the WebGPU blend factor is 'src1-alpha'. The wrong
        # string makes the browser reject every dual-source-blend pipeline.
        ("'one-minus-src1alpha'", "'one-minus-src1-alpha'"),
        ("'src1alpha'", "'src1-alpha'"),
    ],
}

# Names library_wink.js depends on; their absence means a different package.
REQUIRED = {
    'webgpu/src/library_webgpu.js': ['importJsDevice', 'emwgpuCreateSurface'],
}


def main(pkg):
    for rel, edits in EDITS.items():
        path = os.path.join(pkg, rel)
        with open(path, encoding='utf-8') as f:
            text = f.read()
        orig = text
        for old, new in edits:
            if new in text:
                continue
            if old not in text:
                sys.exit('%s: neither the old nor the new form of %r is present' % (rel, old))
            text = text.replace(old, new)
        if text != orig:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(text)
            print('patched %s' % rel)
    for rel, names in REQUIRED.items():
        with open(os.path.join(pkg, rel), encoding='utf-8') as f:
            text = f.read()
        for name in names:
            if name not in text:
                sys.exit('%s: expected %s, which library_wink.js uses' % (rel, name))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('usage: patch-emdawnwebgpu.py <emdawnwebgpu_pkg dir>')
    main(sys.argv[1])
