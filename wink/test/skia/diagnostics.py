# SPDX-License-Identifier: MIT
"""Operation-scoped diagnostics from Skia's deliberate negative tests."""
import re


# Each scope surrounds one rejected operation, with its original result,
# callback, and pixel assertions intact. Every listed message is required
# exactly once; other Skia errors remain failures even inside the scope.
EXPECTED = {
    'null-recording': ('[skia] ** ERROR ** Cannot insert null Recording',),
    'recording-order': ('[skia] ** ERROR ** Recordings are expected to be replayed in order',),
    'promise-volatile': ('[skia] ** ERROR ** Volitile PromiseImage instantiation has failed',),
    'promise-nonvolatile': ('[skia] ** ERROR ** Non-volatile PromiseImage instantiation has failed',),
    'image-provider': ('[skia] WARNING - Key context creation failed in Device::drawGeometry, draw dropped!',),
    'protected-texture': ('[skia] ** ERROR ** Tried to create protected VkImage in unprotected Context.',),
    'nonrenderable-surface': (re.compile(
        r'\[skia\] \*\* ERROR \*\* validate_backend_texture failed: backendTex.info = '
        r'(?:Vulkan|Dawn)\(viewFormat=RGBA8,[^\n]*\); colorType = 4'),),
}
for suffix, message in {
    'mipmaps': "Deferred canvas mipmap settings don't match instantiating target's.",
    'dimensions': "Deferred canvas dimensions don't match instantiating target's dimensions.",
    'translation': 'Replay translation is not allowed when replaying draws to a mipmapped deferred canvas.',
    'clip': 'Replay clip is not allowed when replaying draws to a mipmapped deferred canvas.',
}.items():
    EXPECTED['deferred-' + suffix] = (
        '[skia] ** ERROR ** ' + message,
        '[skia] ** ERROR ** Failed to set up deferred replay target',
    )


def inspect(log):
    problems, accepted, unexpected = [], [], []
    unit = None
    scope = None
    remaining = []
    for line in log.splitlines():
        if line.startswith('WINK_SKIA_START\tunit\ttest\t'):
            unit = line.split('\t', 4)[4]
        elif line.startswith('WINK_SKIA_DONE\tunit\ttest\t'):
            if scope:
                problems.append('expected diagnostic scope crossed a task boundary')
            unit = None
        elif line.startswith('WINK_SKIA_EXPECT_BEGIN\t'):
            name = line.split('\t', 1)[1]
            if scope or not unit or name not in EXPECTED:
                problems.append('invalid expected diagnostic scope: ' + name)
                continue
            scope = name
            remaining = list(EXPECTED[name])
        elif line.startswith('WINK_SKIA_EXPECT_END\t'):
            name = line.split('\t', 1)[1]
            if name != scope or not scope:
                problems.append('unbalanced expected diagnostic scope: ' + name)
            elif remaining:
                problems.append('missing expected diagnostic in ' + name)
            scope = None
            remaining = []
        elif re.search(r'draw dropped!|\*\* ERROR \*\*', line, re.I):
            match = next((message for message in remaining
                          if (message.fullmatch(line) if isinstance(message, re.Pattern)
                              else message == line)), None) if scope else None
            if match is not None:
                remaining.remove(match)
                accepted.append({'unit': unit, 'operation': scope, 'message': line})
            else:
                unexpected.append(line)
    if scope:
        problems.append('unfinished expected diagnostic scope: ' + scope)
    if unexpected:
        problems.append('Skia reported an error or dropped a draw')
    return dict(problems=problems, expected_diagnostics=accepted,
                unexpected_diagnostics=unexpected)
