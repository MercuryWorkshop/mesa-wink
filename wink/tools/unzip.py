#!/usr/bin/env python3
"""Extract a zip, keeping file modes. `unzip` is not on every minimal host."""
import os
import sys
import zipfile

def main(archive, dest):
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = z.extract(info, dest)
            mode = (info.external_attr >> 16) & 0o777
            if mode and not info.is_dir():
                os.chmod(path, mode)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('usage: unzip.py <archive.zip> <dest>')
    main(sys.argv[1], sys.argv[2])
