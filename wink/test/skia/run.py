#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Inventory, run, and compare Skia DM without treating missing work as success."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
MESA = HERE.parents[2]
ROUTES = {'gles': ('gles', False), 'vk': ('vulkan', False), 'grvk': ('vulkan', True),
          'grdawn_gles': ('gles', True)}
CONFIG_ROUTES = {**{route: route for route in ROUTES},
                 'glesmsaa4': 'gles', 'vkmsaa4': 'vk'}


def selected_config(a):
    config = a.config or a.route
    if CONFIG_ROUTES.get(config) != a.route:
        raise ValueError(f'config {config} does not use route {a.route}')
    return config


def parse_seed(value):
    seed = int(value, 10)
    if not 0 <= seed <= 0xffffffff:
        raise ValueError('seed must be an unsigned 32-bit integer')
    return seed


def selected_seed(a):
    if a.paint_key_seed is not None:
        return a.paint_key_seed
    value = os.environ.get('WINK_SKIA_PAINT_KEY_SEED')
    return parse_seed(value) if value is not None else None


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def events(log, kind):
    return [tuple(line.split('\t', 4)[1:]) for line in log.splitlines()
            if line.startswith('WINK_SKIA_' + kind + '\t')]


def inspect_run(expected, log, returncode, route, reference=False):
    completed = events(log, 'DONE')
    started = events(log, 'START')
    problems = []
    if returncode != 0:
        problems.append(f'DM exited with {returncode}')
    if not expected:
        problems.append('empty inventory')
    if Counter(expected) != Counter(completed):
        problems.append('completed task inventory differs from dry run')
    if Counter(started) != Counter(completed):
        problems.append('tasks completed without starting or remained unfinished')
    skipped = events(log, 'SKIP')
    if skipped:
        problems.append(f'{len(skipped)} tasks skipped')
    filtered = events(log, 'FILTER')
    if filtered:
        problems.append(f'{len(filtered)} tasks excluded by DM skip filters')
    active_unit = None
    unit_context = False
    no_context = []
    api = ROUTES[route][0]
    for line in log.splitlines():
        if line.startswith('WINK_SKIA_START\tunit\ttest\t'):
            active_unit = line.split('\t', 4)[4]
            unit_context = False
        elif active_unit and (line.startswith('WINK_SKIA_CONTEXT\t') or
                              line.startswith('WINK_SKIA_BACKEND\t' + api + '\t')):
            unit_context = True
        elif line.startswith('WINK_SKIA_DONE\tunit\ttest\t') and active_unit:
            if not unit_context:
                no_context.append(active_unit)
            active_unit = None
    if no_context:
        problems.append(f'{len(no_context)} GPU unit tests completed without an observed GPU context')
    identities = events(log, 'BACKEND')
    matching = [r for r in identities if r[0] == api]
    if not matching:
        problems.append('no backend identity recorded')
    if reference:
        # Ganesh keeps a sentinel GL context alive for Vulkan/TSAN. It does
        # not render this configuration; check the selected API's identity.
        if any('gex' in ' '.join(r).lower() or 'wink on' in ' '.join(r).lower() for r in matching):
            problems.append('reference unexpectedly uses Wink')
    elif api == 'gles':
        if not matching or any('gex' not in ' '.join(r).lower() for r in matching):
            problems.append('GLES backend is not Wink')
    elif not matching or any('wink on' not in ' '.join(r).lower() for r in matching):
        problems.append('Vulkan backend is not Wink')
    # Capture errors even where the graphics API leaves DM's exit status zero.
    if re.search(r'validation error|webgpu error|uncaptured error|device (?:lost|error)|WINK_SKIA_CONTEXT_FAILED', log, re.I):
        problems.append('graphics validation, device, or context error')
    if re.search(r'draw dropped!|\*\* ERROR \*\*', log, re.I):
        problems.append('Skia reported an error or dropped a draw')
    return {'pass': not problems, 'problems': problems,
            'discovered': len(expected), 'started': len(started),
            'completed': len(completed), 'skipped': skipped, 'backends': identities,
            'source_vetoes': events(log, 'VETO'), 'filtered': filtered,
            'units_without_context': no_context,
            'missing': list((Counter(expected) - Counter(completed)).elements()),
            'unexpected': list((Counter(completed) - Counter(expected)).elements())}


def results(root):
    data = json.loads((Path(root) / 'dm.json').read_text())
    rows = {}
    for row in data['results']:
        key = row['key']
        # Configs are checked at the run level; equivalent rendering outputs
        # are matched by source identity and output format, never file order.
        identity = (key['source_type'], key.get('source_options', ''), key['name'],
                    json.dumps(row['options'], sort_keys=True))
        if identity in rows:
            raise ValueError(f'duplicate result: {identity}')
        rows[identity] = row
    return rows


def compare(reference, candidate):
    reference, candidate = Path(reference), Path(candidate)
    ref_run = json.loads((reference / 'run.json').read_text())
    cand_run = json.loads((candidate / 'run.json').read_text())
    problems = []
    for key in ('skia', 'skia_fixes', 'route', 'sources', 'match', 'fixtures', 'paint_key_seed'):
        if ref_run.get(key) != cand_run.get(key):
            problems.append(f'{key} differs')
    if ref_run.get('config', ref_run.get('route')) != cand_run.get('config', cand_run.get('route')):
        problems.append('config differs')
    if ref_run.get('role') != 'reference' or cand_run.get('role') != 'candidate':
        problems.append('expected independent reference and candidate roles')
    if not ref_run.get('pass') or not cand_run.get('pass'):
        problems.append('reference or candidate execution did not pass')
    a, b = results(reference), results(candidate)
    if not a or not b:
        problems.append('empty rendering results')
    missing = sorted(a.keys() - b.keys())
    unexpected = sorted(b.keys() - a.keys())
    different = [dict(source=list(key), reference=a[key]['md5'], candidate=b[key]['md5'])
                 for key in sorted(a.keys() & b.keys()) if a[key]['md5'] != b[key]['md5']]
    if missing or unexpected or different:
        problems.append('rendering output differs')
    return dict(pass_=not problems, problems=problems, missing=missing,
                unexpected=unexpected, differences=different, matched=len(a.keys() & b.keys()))


def fixtures(source):
    root = source / 'resources'
    if not root.is_dir():
        raise ValueError(f'missing resources: {root}')
    # Include path names and bytes: adding/removing resources changes coverage.
    h = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_file():
            h.update(str(path.relative_to(root)).encode() + b'\0')
            h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def driver_provenance(dm, environment):
    paths = []
    manifest = dm.parent / 'wink-build.json'
    if manifest.is_file():
        build = json.loads(manifest.read_text())
        paths.extend(Path(flag) for flag in build['gn_args']['extra_ldflags']
                     if Path(flag).is_absolute() and Path(flag).is_file())
        angle = build['gn_args'].get('wink_skia_angle_dir')
        if angle:
            paths.extend(Path(angle) / name for name in ('libEGL.dylib', 'libGLESv2.dylib'))
    for name in environment.get('VK_DRIVER_FILES', '').split(os.pathsep):
        if not name:
            continue
        path = Path(name)
        paths.append(path)
        if path.is_file():
            library = json.loads(path.read_text())['ICD']['library_path']
            library = Path(library)
            paths.append(library if library.is_absolute() else path.parent / library)
    return {str(path.resolve()): digest(path) for path in paths if path.is_file()}


def execute(a):
    if a.out.exists():
        raise ValueError(f'output already exists: {a.out}; use a fresh directory')
    a.out.mkdir(parents=True)
    source = a.source.resolve()
    revision = subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip()
    expected_pin = json.loads((HERE / 'pins.json').read_text())['skia']['commit']
    if revision != expected_pin:
        raise ValueError('Skia revision differs from pins.json')
    env = os.environ.copy()
    seed = selected_seed(a)
    if seed is not None:
        env['WINK_SKIA_PAINT_KEY_SEED'] = str(seed)
    env['WINK_SKIA_ROUTE'] = a.route
    env['WINK_VK_ABORT_ON_ERROR'] = '1'
    if a.icd:
        env['VK_DRIVER_FILES'] = str(a.icd.resolve())
    if a.role == 'candidate':
        if not a.icd:
            env['VK_DRIVER_FILES'] = str(MESA / 'build/wvk-native/src/wink/vulkan/wink_devenv_icd.aarch64.json')
        env['WINK_SKIA_EXPECT_VULKAN'] = 'wink on'
    else:
        env.pop('WINK_SKIA_EXPECT_VULKAN', None)
    env.pop('WINK_SKIA_REFERENCE_GLES', None)
    if a.role == 'reference' and a.reference_angle:
        env['WINK_SKIA_REFERENCE_GLES'] = '1'
    config, graphite = selected_config(a), ROUTES[a.route][1]
    args = [str(a.dm.resolve()), '--config', config, '--src', *a.sources,
            '--resourcePath', str(source / 'resources'), '--images', str(source / 'resources/images'),
            '--threads', '0', '--verbose', '--nocpu']
    args += ['--nogpu', '--graphite'] if graphite else ['--nographite']
    if a.match:
        args += ['--match', *a.match]
    manifest = a.dm.parent / 'wink-build.json'
    skia_fixes = json.loads(manifest.read_text()).get('skia_fixes', []) if manifest.is_file() else []
    metadata = dict(schema=1, role=a.role, route=a.route, config=config, target='native', skia=revision,
                    skia_fixes=skia_fixes,
                    wink=subprocess.check_output(['git', '-C', MESA, 'rev-parse', 'HEAD'], text=True).strip(),
                    dm_sha256=digest(a.dm), host=platform.platform(), sources=a.sources,
                    match=a.match, fixtures=fixtures(source), command=args,
                    drivers=driver_provenance(a.dm, env), paint_key_seed=seed,
                    environment={k: env[k] for k in ('VK_DRIVER_FILES', 'WINK_SKIA_EXPECT_VULKAN',
                                 'WINK_SKIA_ROUTE', 'WINK_VK_ABORT_ON_ERROR',
                                 'WINK_SKIA_PAINT_KEY_SEED',
                                 'WINK_SKIA_TRACE_PRECOMPILE') if k in env})
    write_json(a.out / 'reproduce.json', metadata)
    write_json(a.out / 'run.json', dict(metadata, **{'pass': False, 'problems': ['inventory has not completed']}))
    inventory_command = [*args, '--dryRun']
    try:
        with (a.out / 'inventory.log').open('w') as log:
            inventory = subprocess.run(inventory_command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=a.timeout)
    except subprocess.TimeoutExpired:
        write_json(a.out / 'run.json', dict(metadata, **{'pass': False, 'problems': ['inventory timed out']}))
        return 1
    text = (a.out / 'inventory.log').read_text(errors='replace')
    expected = events(text, 'DONE')
    write_json(a.out / 'inventory.json', [list(task) for task in expected])
    if inventory.returncode or not expected:
        metadata.update({'pass': False, 'problems': ['inventory failed or was empty']})
        write_json(a.out / 'run.json', metadata)
        return 1
    if a.command == 'list':
        metadata.update({'pass': False, 'problems': ['inventory only; tests have not run'],
                         'discovered': len(expected)})
        write_json(a.out / 'run.json', metadata)
        print(f'{len(expected)} tasks inventoried in {a.out}')
        return 0
    command = [*args, '--writePath', str(a.out.resolve())]
    start = time.monotonic()
    try:
        with (a.out / 'dm.log').open('w') as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=a.timeout)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        returncode = 124
    text = (a.out / 'dm.log').read_text(errors='replace')
    metadata.update(inspect_run(expected, text, returncode, a.route, a.role == 'reference'))
    metadata['elapsed_seconds'] = time.monotonic() - start
    if not (a.out / 'dm.json').is_file():
        metadata['pass'] = False
        metadata['problems'].append('missing dm.json')
    write_json(a.out / 'run.json', metadata)
    print(json.dumps({k: metadata[k] for k in ('pass', 'problems', 'discovered', 'completed')}))
    return 0 if metadata['pass'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('list', 'run', 'reference'):
        s = sub.add_parser(name)
        s.add_argument('--dm', type=Path, default=MESA / 'build/skia/native/dm')
        s.add_argument('--target', choices=['native', 'st', 'mt'], default='native')
        s.add_argument('--browser-dir', type=Path, help='Directory containing a browser DM build')
        s.add_argument('--source', type=Path, default=MESA / 'build/skia/deps/skia')
        s.add_argument('--route', choices=ROUTES, required=True)
        s.add_argument('--config', choices=CONFIG_ROUTES,
                       help='Skia rendering configuration; defaults to the selected route')
        s.add_argument('--out', type=Path, required=True)
        s.add_argument('--sources', nargs='+', default=['tests', 'gm', 'image'])
        s.add_argument('--match', nargs='+', default=[])
        s.add_argument('--timeout', type=int, default=7200)
        s.add_argument('--paint-key-seed', type=parse_seed,
                       help='Reproduce PaintParamsKeyTestReduced with an unsigned 32-bit seed')
        s.add_argument('--icd', type=Path, help='Explicit ICD manifest; required for Vulkan references')
        s.add_argument('--reference-angle', action='store_true', help='Select the independent ANGLE factory in a reference build')
        s.set_defaults(role='reference' if name == 'reference' else 'candidate')
    s = sub.add_parser('compare')
    s.add_argument('reference', type=Path)
    s.add_argument('candidate', type=Path)
    s.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    try:
        if a.command == 'compare':
            report = compare(a.reference, a.candidate)
            report['pass'] = report.pop('pass_')
            a.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(a.out, report)
            return 0 if report['pass'] else 1
        selected_config(a)
        if a.target != 'native':
            from run_browser import execute as execute_browser
            return execute_browser(a)
        return execute(a)
    except (ValueError, OSError, subprocess.SubprocessError) as e:
        p.exit(1, f'{e}\n')


if __name__ == '__main__':
    sys.exit(main())
