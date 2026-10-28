# SPDX-License-Identifier: MIT
import unittest

from diagnostics import EXPECTED, inspect


class NegativeOperations(unittest.TestCase):
    def log(self, body):
        return '\n'.join(['WINK_SKIA_START\tunit\ttest\t\tnegative', *body,
                          'WINK_SKIA_DONE\tunit\ttest\t\tnegative'])

    def test_exact_operation(self):
        result = inspect(self.log(['WINK_SKIA_EXPECT_BEGIN\tnull-recording',
                                   *EXPECTED['null-recording'],
                                   'WINK_SKIA_EXPECT_END\tnull-recording']))
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['expected_diagnostics'][0]['unit'], 'negative')

    def test_errors_outside_operation_or_extra_errors_fail(self):
        for extra in [EXPECTED['null-recording'][0], '[skia] ** ERROR ** unrelated']:
            for inside in [True, False]:
                body = ['WINK_SKIA_EXPECT_BEGIN\tnull-recording',
                        *EXPECTED['null-recording']]
                if inside:
                    body.append(extra)
                body.append('WINK_SKIA_EXPECT_END\tnull-recording')
                if not inside:
                    body.append(extra)
                self.assertTrue(inspect(self.log(body))['problems'])

    def test_missing_unknown_unbalanced_and_cross_task_scopes_fail(self):
        for body in [
            ['WINK_SKIA_EXPECT_BEGIN\tnull-recording', 'WINK_SKIA_EXPECT_END\tnull-recording'],
            ['WINK_SKIA_EXPECT_BEGIN\tunknown'],
            ['WINK_SKIA_EXPECT_END\tnull-recording'],
            ['WINK_SKIA_EXPECT_BEGIN\tnull-recording', *EXPECTED['null-recording']],
            ['WINK_SKIA_EXPECT_BEGIN\tnull-recording', 'WINK_SKIA_EXPECT_BEGIN\timage-provider'],
        ]:
            self.assertTrue(inspect(self.log(body))['problems'])


if __name__ == '__main__':
    unittest.main()
