#!/usr/bin/env python3
"""Fix the pinned emdawnwebgpu package (v20261109.214834, Dawn 7435fd8) for wink.

Each edit below is applied independently and idempotently: text already in its
new form is left alone, text in its old form is replaced, and text in neither
form is an error -- a silently unpatched library is the failure this exists to
prevent. Upstream has since fixed the dual-source blend factor spelling; the
Emscripten 3.1.56 compatibility edits (stack helper deps, the emwgpuWaitAny
stub's arity) went with the move to .NET 11's emscripten 6.0.3.
"""

import os
import sys

EDITS = {
    'webgpu/src/library_webgpu.js': [
        # Objects are keyed by pointer, but the _emwgpuCreate* result importJs*
        # inserts reaches JS as a signed i32: in a heap over 2 GB -- the .NET
        # host's is about 4 GB -- it lands under a negative key while every
        # lookup (getJsObject, and jsObjectInsert's ensurePointerUnsigned) is
        # unsigned. emwgpuDelete's is a 'p' parameter, which emscripten makes
        # unsigned once the heap can pass 2 GB; kept explicit anyway.
        ('WebGPU.Internals.jsObjects[ptr] = obj;',
         'WebGPU.Internals.jsObjects[ptr >>> 0] = obj;'),
        ('delete WebGPU.Internals.jsObjects[ptr];',
         'delete WebGPU.Internals.jsObjects[ptr >>> 0];'),
        # The C API uses UINT32_MAX for an omitted pass timestamp. The JS
        # API needs undefined; forwarding the sentinel fails validation.
        ('makePassTimestampWrites: (ptr) => {',
         'optionalQuerySetIndex: (index) => index === 0xffffffff ? undefined : index,\n\n'
         '    makePassTimestampWrites: (ptr) => {'),
        ('"beginningOfPassWriteIndex": {{{ makeGetValue(\'ptr\', C_STRUCTS.WGPUPassTimestampWrites.beginningOfPassWriteIndex, \'u32\') }}}',
         '"beginningOfPassWriteIndex": WebGPU.optionalQuerySetIndex({{{ makeGetValue(\'ptr\', C_STRUCTS.WGPUPassTimestampWrites.beginningOfPassWriteIndex, \'u32\') }}})'),
        ('"endOfPassWriteIndex": {{{ makeGetValue(\'ptr\', C_STRUCTS.WGPUPassTimestampWrites.endOfPassWriteIndex, \'u32\') }}}',
         '"endOfPassWriteIndex": WebGPU.optionalQuerySetIndex({{{ makeGetValue(\'ptr\', C_STRUCTS.WGPUPassTimestampWrites.endOfPassWriteIndex, \'u32\') }}})'),
        # WGPU_DEPTH_CLEAR_VALUE_UNDEFINED is NaN. The WebIDL float conversion
        # rejects NaN even when depth is read-only or the format is stencil
        # only. Omit the JS dictionary value as the C sentinel requests.
        ('"depthClearValue": {{{ makeGetValue(\'dsaPtr\', C_STRUCTS.WGPURenderPassDepthStencilAttachment.depthClearValue, \'float\') }}}',
         '"depthClearValue": ((value) => Number.isNaN(value) ? undefined : value)({{{ makeGetValue(\'dsaPtr\', C_STRUCTS.WGPURenderPassDepthStencilAttachment.depthClearValue, \'float\') }}})'),
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
