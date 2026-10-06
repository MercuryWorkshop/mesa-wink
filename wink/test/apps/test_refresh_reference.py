# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import suite

spec = importlib.util.spec_from_file_location('refresh_reference', Path(__file__).with_name('refresh-reference.py'))
refresh_reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh_reference)


class RefreshTests(unittest.TestCase):
    def prepare(self, root, value, tolerance=2, ids=('probe-gles',)):
        root.mkdir()
        recipes = {name: {'id': name, 'api': name.removeprefix('probe-'), 'tier': 'fast',
                          'probe': True, 'min_frames': 2, 'policy': {'tolerance': tolerance}}
                   for name in ids}
        (root / 'recipes.json').write_text(json.dumps(recipes))
        for case in recipes.values():
            directory = root / 'cases' / case['id']
            images = directory / 'reference-0'
            images.mkdir(parents=True)
            trace = directory / 'capture.trace'
            trace.write_bytes(value)
            for index in (1, 2):
                Image.new('RGB', (2, 2), 'red').save(images / f'frame-{index}.png')
            (directory / 'capture.lock.json').write_text(json.dumps({
                'case': case, 'case_sha256': suite.case_hash(case),
                'trace_sha256': suite.sha256(trace),
                'checkpoints': {p.name: suite.sha256(p) for p in images.glob('*.png')}}))
            (directory / 'record.log').write_text('recorded successfully\n')
        return root

    def test_refresh_preserves_previous_capture_and_isolates_mutable_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.prepare(root / 'source', b'fresh capture')
            out = self.prepare(root / 'out', b'previous capture')
            refresh_reference.refresh(source, out, ['probe-gles'], 'Fix invalid API calls')
            journal = json.loads((out / 'reference-refreshes.json').read_text())
            self.assertEqual(len(journal), 1)
            retired = Path(journal[0]['superseded'])
            self.assertEqual((retired / 'capture.trace').read_bytes(), b'previous capture')
            self.assertEqual((out / 'cases/probe-gles/capture.trace').read_bytes(), b'fresh capture')
            (source / 'cases/probe-gles/record.log').write_text('new source diagnostic')
            self.assertEqual((out / 'cases/probe-gles/record.log').read_text(), 'recorded successfully\n')
            self.assertEqual(journal[0]['reason'], 'Fix invalid API calls')

    def test_changed_comparison_policy_cannot_replace_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self.prepare(root / 'source', b'fresh', tolerance=10)
            out = self.prepare(root / 'out', b'previous', tolerance=2)
            with self.assertRaisesRegex(RuntimeError, 'preserve comparison policy'):
                refresh_reference.refresh(source, out, ['probe-gles'], 'Refresh')
            self.assertEqual((out / 'cases/probe-gles/capture.trace').read_bytes(), b'previous')
            self.assertFalse((out / 'reference-refreshes.json').exists())

    def test_rejected_second_case_changes_no_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ids = ('probe-gl', 'probe-gles')
            source = self.prepare(root / 'source', b'fresh', ids=ids)
            out = self.prepare(root / 'out', b'previous', ids=ids)
            (source / 'cases/probe-gles/record.log').write_text('warning: ignoring call to unavailable function glPolygonMode')
            with self.assertRaisesRegex(RuntimeError, 'failed admission'):
                refresh_reference.refresh(source, out, list(ids), 'Refresh')
            for name in ids:
                self.assertEqual((out / 'cases' / name / 'capture.trace').read_bytes(), b'previous')
            self.assertFalse((out / 'superseded-references').exists())


if __name__ == '__main__':
    unittest.main()
