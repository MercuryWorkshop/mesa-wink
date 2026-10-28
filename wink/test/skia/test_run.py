# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('skia_run', Path(__file__).with_name('run.py'))
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
build_spec = importlib.util.spec_from_file_location('skia_build', Path(__file__).with_name('build.py'))
builder = importlib.util.module_from_spec(build_spec)
build_spec.loader.exec_module(builder)


class DependencyAdaptation(unittest.TestCase):
    def test_platform_anchor_is_not_skipped_when_replacement_exists_elsewhere(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'BUILD.gn'
            path.write_text('mac: wink_factory\nwasm: null_factory\n')
            builder.patch(root, 'BUILD.gn', 'null_factory', 'wink_factory')
            self.assertEqual(path.read_text(), 'mac: wink_factory\nwasm: wink_factory\n')
            builder.patch(root, 'BUILD.gn', 'null_factory', 'wink_factory')
            self.assertEqual(path.read_text().count('wink_factory'), 2)

    def test_instrumentation_anchor_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'DM.cpp'
            path.write_text('void done() {}')
            for _ in range(2):
                builder.patch(root, 'DM.cpp', 'void done() {', 'void done() { record();')
            self.assertEqual(path.read_text(), 'void done() { record();}')


class ExecutionIntegrity(unittest.TestCase):
    task = ('gles', 'gm', '', 'clip')
    log = ('WINK_SKIA_BACKEND\tgles\tgex WebGPU\n'
           'WINK_SKIA_START\tgles\tgm\t\tclip\n'
           'WINK_SKIA_DONE\tgles\tgm\t\tclip\n')

    def check_log(self, log, expected=None, code=0):
        return suite.inspect_run(expected if expected is not None else [self.task], log, code, 'gles')

    def test_complete_execution(self):
        self.assertTrue(self.check_log(self.log)['pass'])

    def test_missing_duplicate_and_unstarted_tasks_fail(self):
        for log in (self.log.replace('WINK_SKIA_DONE', 'unused'),
                    self.log + 'WINK_SKIA_DONE\tgles\tgm\t\tclip\n',
                    self.log.replace('WINK_SKIA_START', 'unused')):
            self.assertFalse(self.check_log(log)['pass'])

    def test_empty_inventory_and_exit_errors_fail(self):
        self.assertFalse(self.check_log(self.log, expected=[])['pass'])
        for code in (1, -11, 124):
            self.assertFalse(self.check_log(self.log, code=code)['pass'])

    def test_source_veto_is_reported_without_counting_as_gpu_work(self):
        veto = 'WINK_SKIA_VETO\tgles\timage\tcodec\tCPUOnly.png\n'
        result = self.check_log(self.log + veto)
        self.assertTrue(result['pass'])
        self.assertEqual(result['completed'], 1)
        self.assertEqual(result['source_vetoes'], [('gles', 'image', 'codec', 'CPUOnly.png')])
        self.assertFalse(self.check_log(self.log + veto.replace('VETO', 'FILTER'))['pass'])

    def test_skips_and_validation_errors_fail_even_with_exit_zero(self):
        for extra in ('WINK_SKIA_SKIP\tgles\tgm\t\tclip\tunsupported\n',
                      'WebGPU validation error\n', 'Device lost\n',
                      'gex-wgpu: WebGPU error (type 2): invalid attachment\n',
                      '[skia] WARNING - Key context creation failed, draw dropped!\n',
                      '[skia] ** ERROR ** unexpected failure\n'):
            self.assertFalse(self.check_log(self.log + extra)['pass'])

    def test_wrong_backend_and_wink_reference_fail(self):
        self.assertFalse(self.check_log(self.log.replace('gex WebGPU', 'Apple GL'))['pass'])
        self.assertFalse(suite.inspect_run([self.task], self.log, 0, 'gles', reference=True)['pass'])

    def test_gpu_unit_with_no_context_is_not_a_pass(self):
        log = self.log + 'WINK_SKIA_START\tunit\ttest\t\tGPUOnly\nWINK_SKIA_DONE\tunit\ttest\t\tGPUOnly\n'
        expected = [self.task, ('unit', 'test', '', 'GPUOnly')]
        result = self.check_log(log, expected)
        self.assertFalse(result['pass'])
        self.assertEqual(result['units_without_context'], ['GPUOnly'])
        log = log.replace('WINK_SKIA_DONE\tunit', 'WINK_SKIA_CONTEXT\tganesh\nWINK_SKIA_DONE\tunit')
        self.assertTrue(self.check_log(log, expected)['pass'])

    def test_vulkan_unit_cannot_count_sentinel_gles_as_gpu_work(self):
        task = ('unit', 'test', '', 'VKOnly')
        log = ('WINK_SKIA_BACKEND\tvulkan\twink on Metal\n'
               'WINK_SKIA_START\tunit\ttest\t\tVKOnly\n'
               'WINK_SKIA_BACKEND\tgles\tgex WebGPU\n'
               'WINK_SKIA_DONE\tunit\ttest\t\tVKOnly\n')
        result = suite.inspect_run([task], log, 0, 'vk')
        self.assertFalse(result['pass'])
        self.assertEqual(result['units_without_context'], ['VKOnly'])

    def test_host_mock_and_other_api_work_are_not_hardware_coverage(self):
        task = ('unit', 'test', '', 'ScopedUnit')
        for scope, evidence in (
                ('host', 'WINK_SKIA_HOST_UNIT\tScopedUnit\n'),
                ('mock', 'WINK_SKIA_MOCK_CONTEXT\tganesh\n'),
                ('other_api', 'WINK_SKIA_CONTEXT_REQUEST\tganesh\toutside\n')):
            log = (self.log + 'WINK_SKIA_START\tunit\ttest\t\tScopedUnit\n' + evidence +
                   'WINK_SKIA_DONE\tunit\ttest\t\tScopedUnit\n')
            result = self.check_log(log, [self.task, task])
            self.assertTrue(result['pass'])
            self.assertEqual(result['unit_coverage'][scope], ['ScopedUnit'])
            self.assertEqual(result['unit_coverage']['gpu'], [])
            self.assertEqual(result['units_without_context'], ['ScopedUnit'])
            self.assertFalse(self.check_log(log, [self.task, task], code=1)['pass'])
            self.assertFalse(self.check_log(log + '[skia] ** ERROR ** unexpected\n',
                                            [self.task, task])['pass'])

    def test_failed_selected_or_mock_context_stays_unresolved(self):
        task = ('unit', 'test', '', 'FailedContext')
        for evidence in (
                'WINK_SKIA_CONTEXT_REQUEST\tganesh\tmock\n',
                'WINK_SKIA_HOST_UNIT\tAnotherUnit\n',
                'WINK_SKIA_CONTEXT_REQUEST\tganesh\tselected\n'
                'WINK_SKIA_CONTEXT_REQUEST\tganesh\toutside\n'
                'WINK_SKIA_MOCK_CONTEXT\tganesh\n'
                'WINK_SKIA_HOST_UNIT\tFailedContext\n'):
            log = (self.log + 'WINK_SKIA_START\tunit\ttest\t\tFailedContext\n' + evidence +
                   'WINK_SKIA_DONE\tunit\ttest\t\tFailedContext\n')
            result = self.check_log(log, [self.task, task])
            self.assertFalse(result['pass'])
            self.assertEqual(result['unit_coverage']['unresolved'], ['FailedContext'])

    def test_scoped_host_work_alone_cannot_establish_a_gpu_backend(self):
        task = ('unit', 'test', '', 'HostOnly')
        log = ('WINK_SKIA_START\tunit\ttest\t\tHostOnly\n'
               'WINK_SKIA_HOST_UNIT\tHostOnly\n'
               'WINK_SKIA_DONE\tunit\ttest\t\tHostOnly\n')
        result = self.check_log(log, [task])
        self.assertFalse(result['pass'])
        self.assertIn('no backend identity recorded', result['problems'])


    def test_expected_rejection_does_not_excuse_other_failures(self):
        task = ('unit', 'test', '', 'Negative')
        log = (self.log + 'WINK_SKIA_START\tunit\ttest\t\tNegative\n'
               'WINK_SKIA_CONTEXT\tganesh\n'
               'WINK_SKIA_EXPECT_BEGIN\tnull-recording\n'
               '[skia] ** ERROR ** Cannot insert null Recording\n'
               'WINK_SKIA_EXPECT_END\tnull-recording\n'
               'WINK_SKIA_DONE\tunit\ttest\t\tNegative\n')
        result = self.check_log(log, [self.task, task])
        self.assertTrue(result['pass'])
        self.assertEqual(len(result['expected_diagnostics']), 1)
        self.assertFalse(self.check_log(log, [self.task, task], code=1)['pass'])
        self.assertFalse(self.check_log(log + 'WebGPU validation error\n',
                                        [self.task, task])['pass'])
        self.assertFalse(self.check_log(log.replace('WINK_SKIA_CONTEXT\tganesh\n', ''),
                                        [self.task, task])['pass'])

    def test_cpu_registration_is_separate_from_hardware_coverage(self):
        task = ('unit', 'test', '', 'CPUUnit')
        log = (self.log + 'WINK_SKIA_START\tunit\ttest\t\tCPUUnit\n'
               'WINK_SKIA_CPU_UNIT\tCPUUnit\n'
               'WINK_SKIA_DONE\tunit\ttest\t\tCPUUnit\n')
        result = self.check_log(log, [self.task, task])
        self.assertTrue(result['pass'])
        self.assertEqual(result['unit_coverage']['cpu'], ['CPUUnit'])
        self.assertEqual(result['unit_coverage']['gpu'], [])
        self.assertFalse(self.check_log(log.replace('WINK_SKIA_CPU_UNIT\tCPUUnit',
                                                   'WINK_SKIA_CPU_UNIT\tWrongUnit'),
                                        [self.task, task])['pass'])


class ConfigurationIntegrity(unittest.TestCase):
    def test_configs_keep_the_selected_backend(self):
        for config, route in suite.CONFIG_ROUTES.items():
            self.assertEqual(suite.selected_config(SimpleNamespace(route=route, config=config)), config)
            self.assertEqual(suite.selected_config(SimpleNamespace(route=route, config=None)), route)
        for route, config in [('gles', 'vkmsaa4'), ('vk', 'glesmsaa4'), ('grvk', 'vkmsaa4')]:
            with self.assertRaises(ValueError):
                suite.selected_config(SimpleNamespace(route=route, config=config))


class ReferenceIntegrity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref = Path(self.tmp.name) / 'ref'
        self.candidate = Path(self.tmp.name) / 'candidate'
        for root, role in ((self.ref, 'reference'), (self.candidate, 'candidate')):
            root.mkdir()
            suite.write_json(root / 'run.json', dict(role=role, route='gles', skia='abc',
                             sources=['gm'], match=[], fixtures='same', **{'pass': True}))
            suite.write_json(root / 'dm.json', {'results': [self.row('clip', 'abc')]})

    @staticmethod
    def row(name, md5):
        return dict(key=dict(source_type='gm', name=name, config='gles'),
                    options=dict(color_depth='8', alpha_type='premul'), md5=md5)

    def test_exact_pixels_pass(self):
        self.assertTrue(suite.compare(self.ref, self.candidate)['pass_'])

    def test_different_sample_configuration_fails_with_identical_pixels(self):
        path = self.candidate / 'run.json'
        run = json.loads(path.read_text())
        run['config'] = 'glesmsaa4'
        suite.write_json(path, run)
        self.assertIn('config differs', suite.compare(self.ref, self.candidate)['problems'])

    def test_different_font_engine_fails_with_identical_pixels(self):
        path = self.candidate / 'run.json'
        run = json.loads(path.read_text())
        run['font_engine'] = 'freetype'
        suite.write_json(path, run)
        self.assertIn('font_engine differs', suite.compare(self.ref, self.candidate)['problems'])

    def test_different_cpu_unit_selection_fails_with_identical_pixels(self):
        path = self.candidate / 'run.json'
        run = json.loads(path.read_text())
        run['include_cpu_tests'] = True
        suite.write_json(path, run)
        self.assertIn('include_cpu_tests differs', suite.compare(self.ref, self.candidate)['problems'])

    def test_font_manager_mismatch_fails_with_identical_pixels(self):
        path = self.candidate / 'run.json'
        run = json.loads(path.read_text())
        run['font_manager'] = 'pinned-fixtures'
        suite.write_json(path, run)
        self.assertIn('font_manager differs', suite.compare(self.ref, self.candidate)['problems'])

    def test_text_policy_mismatch_and_legacy_browser_policy(self):
        path = self.candidate / 'run.json'
        run = json.loads(path.read_text())
        run['target'] = 'mt'
        suite.write_json(path, run)
        self.assertIn('text_policy differs', suite.compare(self.ref, self.candidate)['problems'])
        path = self.ref / 'run.json'
        run = json.loads(path.read_text())
        run['text_policy'] = 'portable'
        run['font_selection'] = 'portable'
        suite.write_json(path, run)
        self.assertTrue(suite.compare(self.ref, self.candidate)['pass_'])

    def test_default_font_selection_mismatch_fails_with_identical_pixels(self):
        path = self.candidate / 'run.json'
        run = json.loads(path.read_text())
        run['font_selection'] = 'portable'
        suite.write_json(path, run)
        self.assertIn('font_selection differs', suite.compare(self.ref, self.candidate)['problems'])

    def test_different_random_seed_fails_even_with_identical_pixels(self):
        path = self.candidate / 'run.json'
        run = json.loads(path.read_text())
        run['paint_key_seed'] = 1793113975
        suite.write_json(path, run)
        result = suite.compare(self.ref, self.candidate)
        self.assertFalse(result['pass_'])
        self.assertIn('paint_key_seed differs', result['problems'])

    def test_different_skia_correctness_fixes_fail(self):
        path = self.candidate / 'run.json'
        run = json.loads(path.read_text())
        run['skia_fixes'] = ['compressed-host-copy-fix']
        suite.write_json(path, run)
        self.assertFalse(suite.compare(self.ref, self.candidate)['pass_'])

    def test_alpha_or_pixel_change_fails(self):
        suite.write_json(self.candidate / 'dm.json', {'results': [self.row('clip', 'changed-alpha')]})
        self.assertFalse(suite.compare(self.ref, self.candidate)['pass_'])

    def test_missing_and_empty_results_fail(self):
        suite.write_json(self.candidate / 'dm.json', {'results': []})
        self.assertFalse(suite.compare(self.ref, self.candidate)['pass_'])

    def test_duplicate_results_rejected(self):
        row = self.row('clip', 'abc')
        suite.write_json(self.candidate / 'dm.json', {'results': [row, row]})
        with self.assertRaises(ValueError):
            suite.compare(self.ref, self.candidate)

    def test_failed_reference_and_different_fixtures_fail(self):
        metadata = json.loads((self.ref / 'run.json').read_text())
        metadata.update({'pass': False, 'fixtures': 'different'})
        suite.write_json(self.ref / 'run.json', metadata)
        self.assertFalse(suite.compare(self.ref, self.candidate)['pass_'])


if __name__ == '__main__':
    unittest.main()
