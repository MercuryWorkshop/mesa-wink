# SPDX-License-Identifier: MIT
"""Browser execution adapter for run.py; artifacts stream to the local runner."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import urlencode

from run import HERE, MESA, ROUTES, digest, events, fixtures, inspect_run, selected_config, selected_seed, write_json


def execute(a):
    if a.role != 'candidate':
        raise ValueError('Generate independent references natively; browser runs are Wink candidates')
    if a.out.exists():
        raise ValueError(f'output already exists: {a.out}')
    build = (a.browser_dir or MESA / 'build/skia' / a.target).resolve()
    for name in ('dm.js', 'dm.wasm', 'index.html'):
        if not (build / name).is_file():
            raise ValueError(f'missing {build / name}; run build-browser.py {a.target}')
    revision = subprocess.check_output(['git', '-C', a.source, 'rev-parse', 'HEAD'], text=True).strip()
    pin = json.loads((HERE / 'pins.json').read_text())['skia']['commit']
    if revision != pin:
        raise ValueError('Skia revision differs from pins.json')
    a.out.mkdir(parents=True)
    graphite = ROUTES[a.route][1]
    config = selected_config(a)
    seed = selected_seed(a)
    trace = os.environ.get('WINK_SKIA_TRACE_PRECOMPILE')
    args = ['--config', config, '--src', *a.sources, '--resourcePath', '/resources',
            '--images', '/resources/images', '--threads', '0',
            '--gpuThreads', '0' if a.target == 'st' else '2', '--verbose']
    if not a.include_cpu_tests:
        args.append('--nocpu')
    args += ['--nogpu', '--graphite'] if graphite else ['--nographite']
    if a.match:
        args += ['--match', *a.match]
    manifest = build / 'wink-build.json'
    skia_fixes = json.loads(manifest.read_text()).get('skia_fixes', []) if manifest.is_file() else []
    metadata = dict(schema=1, role='candidate', route=a.route, config=config, target=a.target,
                    font_engine='freetype',
                    include_cpu_tests=a.include_cpu_tests,
                    text_policy='portable',
                    font_selection='portable',
                    skia_fixes=skia_fixes,
                    skia=pin, sources=a.sources, match=a.match, fixtures=fixtures(a.source),
                    paint_key_seed=seed,
                    environment={'WINK_SKIA_TRACE_PRECOMPILE': trace} if trace is not None else {},
                    wasm_sha256=digest(build / 'dm.wasm'), js_sha256=digest(build / 'dm.js'),
                    wink=subprocess.check_output(['git', '-C', MESA, 'rev-parse', 'HEAD'], text=True).strip(),
                    **{'pass': False, 'problems': ['execution has not completed']})
    write_json(a.out / 'run.json', metadata)
    expected = []
    for phase in (['inventory'] if a.command == 'list' else ['inventory', 'run']):
        result_path = (a.out / f'{phase}-browser.json').resolve()
        artifact_dir = (a.out / f'{phase}-artifacts').resolve()
        phase_args = args + (['--dryRun'] if phase == 'inventory' else ['--writePath', '/out'])
        options = {'args': json.dumps(phase_args), 'route': a.route, 'target': a.target}
        if seed is not None:
            options['paint-key-seed'] = seed
        if trace is not None:
            options['trace-precompile'] = trace
        query = urlencode(options)
        command = ['node', str(HERE.parent / 'run-browser.mjs'), str(build),
                   '--page=index.html?' + query, '--timeout=' + str(a.timeout),
                   '--result=' + str(result_path), '--artifact-dir=' + str(artifact_dir)]
        write_json(a.out / f'{phase}-command.json', command)
        with (a.out / f'{phase}-launcher.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout + 30)
        result = json.loads(result_path.read_text())
        log = '\n'.join(result.get('log', []))
        (a.out / ('inventory.log' if phase == 'inventory' else 'dm.log')).write_text(log)
        if phase == 'inventory':
            expected = events(log, 'DONE')
            write_json(a.out / 'inventory.json', expected)
            if result.get('rc') != 0 or not expected:
                metadata['problems'] = ['browser inventory failed or was empty']
                write_json(a.out / 'run.json', metadata)
                return 1
        else:
            metadata.update(inspect_run(expected, log, result.get('rc', -1), a.route))
            metadata['browser'] = result.get('browser')
            if a.target == 'mt' and not events(log, 'WORKER') == [('1',)]:
                metadata['pass'] = False
                metadata['problems'].append('MT body did not execute on a worker')
            if (artifact_dir / 'dm.json').is_file():
                shutil.copy2(artifact_dir / 'dm.json', a.out / 'dm.json')
            else:
                metadata['pass'] = False
                metadata['problems'].append('missing browser dm.json artifact')
    if a.command == 'list':
        metadata['problems'] = ['inventory only; tests have not run']
        metadata['discovered'] = len(expected)
    write_json(a.out / 'run.json', metadata)
    print(json.dumps({k: metadata.get(k) for k in ('pass', 'problems', 'discovered', 'completed')}))
    return 0 if a.command == 'list' or metadata['pass'] else 1
