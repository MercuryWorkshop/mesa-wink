# Copyright 2026 wink contributors
# SPDX-License-Identifier: MIT
#
# An ld64 alias list naming each GL function after the dispatch stub that
# wink_get_proc_address returns for it, from shared-glapi's generated header:
#
#   wink_gl_aliases.py shared_glapi_mapi_tmp.h OUT
#
# A dylib's gl* symbols, for a loader that finds GL by dlsym (LWJGL's on
# macOS) and then checks those against SDL_GL_GetProcAddress's.
import re
import sys

text = open(sys.argv[1]).read()


def section(start, end='};'):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


# Its names, NUL-separated, but for the last.
pool = section('static const char public_string_pool[] =', ';')
names = {}
offset = 0
for name in re.findall(r'"(\w+)(?:\\0)?"', pool):
    names[offset] = name
    offset += len(name) + 1

entries = re.findall(r'\(_glapi_proc\) _dispatch_stub_(\w+)', section('static const _glapi_proc public_entries[] ='))
stubs = re.findall(r'\{ (\d+), (\d+) \}', section('static const struct mapi_stub public_stubs[] ='))

with open(sys.argv[2], 'w') as out:
    for name_offset, slot in stubs:
        out.write(f'__dispatch_stub_{entries[int(slot)]} _gl{names[int(name_offset)]}\n')
