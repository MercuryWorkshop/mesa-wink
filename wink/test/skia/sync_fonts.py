#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Install the paragraph fonts pinned by Skia's own bot asset version."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent


def ensure(source, archive):
    pins = json.loads((HERE / 'pins.json').read_text())
    pin = pins['paragraph_fonts']
    version = (source / 'infra/bots/assets/skparagraph/VERSION').read_text().strip()
    if version != pin['version']:
        raise ValueError('Skia paragraph-font asset version differs from pins.json')
    if not archive.is_file():
        archive.parent.mkdir(parents=True, exist_ok=True)
        temporary = archive.with_suffix('.download')
        subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                        '--max-time', '180', pin['url'], '--output', str(temporary)], check=True)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != pin['sha256']:
            raise ValueError('paragraph-font archive hash differs from pins.json')
        temporary.replace(archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != pin['sha256']:
        raise ValueError('paragraph-font archive hash differs from pins.json')
    target = source / 'resources/extra_fonts'
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for name in package.namelist():
            if name.startswith('.cipdpkg/'):
                continue
            # This pinned asset consists of flat font files only.
            if Path(name).name != name or name in ('.', '..'):
                raise ValueError('invalid paragraph-font archive path: ' + name)
            content = package.read(name)
            path = target / name
            if path.exists() and path.read_bytes() != content:
                raise ValueError('paragraph-font fixture was modified: ' + str(path))
            if not path.exists():
                path.write_bytes(content)
    system = pins['system_fonts']
    target = source / 'resources/wink_fonts'
    target.mkdir(exist_ok=True)
    for name, checksum in system['files'].items():
        path = target / name
        if not path.is_file():
            cached = archive.parent / 'roboto-system-fonts' / name
            if cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest() == checksum:
                shutil.copy2(cached, path)
            else:
                url = system['url'] + ('LICENSE' if name == 'LICENSE' else 'src/hinted/' + name)
                temporary = path.with_suffix('.download')
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                '--max-time', '60', url, '--output', str(temporary)], check=True)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                    raise ValueError('system font hash differs from pins.json: ' + name)
                temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError('system font fixture was modified: ' + str(path))
    shutil.copy2(HERE / 'fonts.xml', source / 'resources/wink-fonts.xml')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=HERE.parents[2] / 'build/skia')
    a = p.parse_args()
    ensure(a.out.resolve() / 'deps/skia', a.out.resolve() / 'skparagraph-v4.zip')


if __name__ == '__main__':
    main()
