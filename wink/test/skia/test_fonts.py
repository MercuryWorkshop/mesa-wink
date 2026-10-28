# SPDX-License-Identifier: MIT
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import sync_fonts


class PinnedFonts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'skia'
        (self.source / 'resources').mkdir(parents=True)
        version = self.source / 'infra/bots/assets/skparagraph/VERSION'
        version.parent.mkdir(parents=True)
        version.write_text('4\n')
        (self.root / 'fonts.xml').write_text('<familyset version="21"/>')
        self.archive = self.root / 'fonts.zip'
        self.patcher = patch.object(sync_fonts, 'HERE', self.root)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def package(self, name='Example.ttf'):
        with zipfile.ZipFile(self.archive, 'w') as package:
            package.writestr(name, b'font bytes')
            package.writestr('.cipdpkg/manifest.json', '{}')
        (self.root / 'pins.json').write_text(json.dumps(dict(
            paragraph_fonts=dict(version='4', url='unused',
                sha256=hashlib.sha256(self.archive.read_bytes()).hexdigest()),
            system_fonts=dict(url='unused', files={}))))

    def test_pinned_install_is_idempotent_and_detects_modified_fonts(self):
        self.package()
        with patch.object(sync_fonts.subprocess, 'run') as download:
            sync_fonts.ensure(self.source, self.archive)
            sync_fonts.ensure(self.source, self.archive)
            download.assert_not_called()
        installed = self.source / 'resources/extra_fonts/Example.ttf'
        self.assertEqual(installed.read_bytes(), b'font bytes')
        installed.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'fixture was modified'):
            sync_fonts.ensure(self.source, self.archive)

    def test_corrupt_archive_is_rejected(self):
        self.package()
        self.archive.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'archive hash differs'):
            sync_fonts.ensure(self.source, self.archive)

    def test_archive_paths_cannot_escape_fixture_directory(self):
        self.package('../outside.ttf')
        with self.assertRaisesRegex(ValueError, 'invalid paragraph-font archive path'):
            sync_fonts.ensure(self.source, self.archive)
        self.assertFalse((self.source / 'resources/outside.ttf').exists())
