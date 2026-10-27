#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the Skia-pinned ANGLE Metal backend for independent GLES references."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('skia_build', HERE / 'build.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=base.MESA / 'build/skia')
    p.add_argument('--jobs', type=int, default=6)
    a = p.parse_args()
    source = a.out.resolve() / 'deps/skia'
    build = a.out.resolve() / 'angle'
    build.mkdir(parents=True, exist_ok=True)
    args = dict(is_debug=False, skia_use_angle=True, skia_gl_standard='gles',
                angle_enable_metal=True, angle_enable_gl=False,
                angle_enable_vulkan=False, angle_build_tests=False)
    (build / 'args.gn').write_text('\n'.join(f'{k} = {json.dumps(v)}' for k, v in args.items()) + '\n')
    base.run([source / 'bin/gn', 'gen', build], cwd=source)
    base.run(['ninja', '-C', build, '-j', a.jobs, 'libEGL', 'libGLESv2'], cwd=source)


if __name__ == '__main__':
    main()
