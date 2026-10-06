#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Explicitly install fresh admitted captures, preserving superseded evidence."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from suite import cases, sha256, validate_reference


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def copy_reference(path, target):
    path = Path(path)
    # Only immutable captures and CPU checkpoints share disk space.
    # Candidate results and logs are rewritten by subsequent runs.
    if path.name in ('capture.trace', 'capture.gfxr') or path.parent.name.startswith('reference-'):
        try:
            os.link(path, target)
            return target
        except OSError:
            pass
    return shutil.copy2(path, target)


def refresh(source, out, case_ids, reason, dry_run=False):
    source, out = source.resolve(), out.resolve()
    if source == out or source.is_relative_to(out) or out.is_relative_to(source) or not reason.strip():
        raise ValueError('Use separate output roots and a nonempty refresh reason')
    selected = cases(SimpleNamespace(out=source, recipes=source / 'recipes.json', tier='full', case=case_ids))
    if {c['id'] for c in selected} != set(case_ids):
        raise ValueError('Unknown refresh case')
    recipes_path = out / 'recipes.json'
    recipes = json.loads(recipes_path.read_text())
    journal_path = out / 'reference-refreshes.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else []
    prepared = []
    # Preflight the whole selection before changing any admitted reference.
    for case in selected:
        original = source / 'cases' / case['id']
        admitted = validate_reference(case, original)
        if not admitted['pass']:
            raise RuntimeError('Fresh reference failed admission: ' + json.dumps(admitted))
        destination = out / 'cases' / case['id']
        previous_lock = destination / 'capture.lock.json'
        previous = json.loads(previous_lock.read_text()) if previous_lock.exists() else None
        if previous:
            for field in ('policy', 'min_frames'):
                if previous['case'].get(field) != case.get(field):
                    raise RuntimeError('A refresh must preserve comparison policy and coverage: ' + field)
            if sha256(previous_lock) == sha256(original / 'capture.lock.json'):
                if dry_run:
                    print('Admitted reference already installed:', case['id'])
                    continue
                raise RuntimeError('Reference already installed: ' + case['id'])
        prepared.append((case, original, destination, previous, admitted))
    for case, original, destination, previous, admitted in prepared:
        if dry_run:
            print('Admitted reference ready:', case['id'])
            continue
        stamp = str(time.time_ns())
        temporary = destination.with_name('.refresh-' + case['id'] + '-' + stamp)
        retired = None
        try:
            shutil.copytree(original, temporary, copy_function=copy_reference)
            if not validate_reference(case, temporary)['pass']:
                raise RuntimeError('Copied reference failed admission')
            if destination.exists():
                retired = out / 'superseded-references' / (case['id'] + '-' + stamp)
                retired.parent.mkdir(exist_ok=True)
                destination.rename(retired)
            try:
                temporary.rename(destination)
            except OSError:
                if retired:
                    retired.rename(destination)
                raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        recipes[case['id']] = case
        journal.append({'case': case['id'], 'reason': reason, 'source': str(original),
                        'superseded': str(retired) if retired else None,
                        'previous_trace_sha256': previous['trace_sha256'] if previous else None,
                        'trace_sha256': admitted['lock']['trace_sha256'],
                        'capture_lock_sha256': sha256(destination / 'capture.lock.json')})
        write_json(journal_path, journal)
        write_json(recipes_path, recipes)
        print('Installed admitted reference:', case['id'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--case', action='append', required=True)
    p.add_argument('--reason', required=True)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    refresh(a.source, a.out, a.case, a.reason, a.dry_run)


if __name__ == '__main__':
    main()
