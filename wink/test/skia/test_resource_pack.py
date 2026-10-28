# SPDX-License-Identifier: MIT
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_browser


class ResourcePackIntegrity(unittest.TestCase):
    def test_stale_or_unrecorded_resource_pack_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / 'st'
            build.mkdir()
            for name in ('dm.js', 'dm.wasm', 'dm.data', 'index.html'):
                (build / name).write_text('unused')
            pin = json.loads((run_browser.HERE / 'pins.json').read_text())['skia']['commit']
            args = SimpleNamespace(role='candidate', out=root / 'result', browser_dir=build,
                                   target='st', source=root / 'source')
            for manifest in ({}, {'fixtures': 'older'}):
                (build / 'wink-build.json').write_text(json.dumps(manifest))
                with patch.object(run_browser.subprocess, 'check_output', return_value=pin), \
                     patch.object(run_browser, 'fixtures', return_value='current'):
                    with self.assertRaisesRegex(ValueError, 'resource pack differs'):
                        run_browser.execute(args)
                self.assertFalse(args.out.exists())
