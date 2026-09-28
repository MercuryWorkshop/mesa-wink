#!/usr/bin/env python3
"""Copy every .h/.inc under <root>/<subdir> to <dest>/<subdir>, keeping paths.

Replaces `find | tar -T - | tar -x`, whose -T spelling differs between GNU and
BSD tar.
"""
import os
import shutil
import sys

def main(root, subdir, dest):
    top = os.path.join(root, subdir)
    n = 0
    for dirpath, _, files in os.walk(top):
        for name in files:
            if not name.endswith(('.h', '.inc')):
                continue
            src = os.path.join(dirpath, name)
            dst = os.path.join(dest, os.path.relpath(src, root))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    if n == 0:
        sys.exit('no headers under %s' % top)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('usage: copy-headers.py <root> <subdir> <dest>')
    main(*sys.argv[1:])
