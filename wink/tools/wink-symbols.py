#!/usr/bin/env python3
"""Check libwink's symbols against what a .NET static wasm link will do with them.

A static wasm link has no symbol namespaces and hidden visibility does not
hide anything from it, so every global wink defines can collide with something
else in the app, and every undefined one must be satisfied by exactly the right
thing. Failures:

  * any gl*/egl*/glX* symbol, defined or undefined. wink exposes GL only
    through wink_get_proc_address(); a bare undefined glFoo would be silently
    bound by emscripten's library_webgl.js, a defined one collides with every
    other GL in the app.
  * the JS exception / setjmp shims (invoke_*, emscripten_longjmp, ...), or
    setjmp helpers without __wasm_longjmp: an object built without
    -fwasm-exceptions, which .NET's wasm-EH link rejects.
  * emscripten_webgl_*: WebGL reached from the WebGPU build.
  * a strong symbol defined both in wink.o and in the Tint or webgpu archive.
  * (st) a pthread_* reference the single-threaded libc does not define.

With --check-against, strong-vs-strong collisions with consumer archives are
reported (not failed: the consumer decides).
"""
import argparse
import os
import re
import subprocess
import sys

GL_RE = re.compile(r'^(gl|egl|glX)[A-Z]')
# The JS-mode exception and longjmp machinery. saveSetjmp, testSetjmp and
# getTempRet0 are not on this list: 3.1.56's LLVM uses those compiler-rt
# helpers in both modes, and what tells them apart is the longjmp itself --
# __wasm_longjmp in wasm mode, emscripten_longjmp plus invoke_* in JS mode.
SHIM_RE = re.compile(r'^(invoke_|emscripten_longjmp$|setThrew$|'
                     r'__cxa_find_matching_catch|__resumeException$)')
SJLJ_HELPERS = ('saveSetjmp', 'testSetjmp')
WEBGL_RE = re.compile(r'^emscripten_webgl_')
JS_RE = re.compile(r'^(wink_js_|gex_wgpu_jspi_|wgpu[A-Z]|emwgpu|emscripten_|_emscripten_|__syscall_|_tzset_js|_mktime_js|_localtime_js|_gmtime_js|_abort_js|_emval|__call_sighandler)')


def nm(emnm, path):
    """(defined_strong, defined_weak, undefined) global symbol sets."""
    out = subprocess.run([emnm, '--extern-only', path], check=True,
                         capture_output=True, text=True).stdout
    strong, weak, undef = set(), set(), set()
    for line in out.splitlines():
        parts = line.split()
        if not parts or line.endswith(':'):
            continue
        if len(parts) == 2:
            kind, name = parts
        elif len(parts) == 3:
            _, kind, name = parts
        else:
            continue
        if kind in ('U', 'w'):
            undef.add(name)
        elif kind in ('W', 'V'):
            weak.add(name)
        else:
            strong.add(name)
    return strong, weak, undef - strong - weak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--emnm', required=True)
    ap.add_argument('--variant', choices=('st', 'mt'), required=True)
    ap.add_argument('--object', required=True, help='wink.o')
    ap.add_argument('--tint', required=True)
    ap.add_argument('--webgpu', required=True)
    ap.add_argument('--libc', help="the variant's emscripten libc.a, for the st pthread check")
    ap.add_argument('--compiler-rt', help="the variant's libcompiler_rt-wasm-sjlj*.a, for the setjmp ABI check")
    ap.add_argument('--check-against', default='')
    ap.add_argument('--collisions-out', help='write the colliding symbol names here, one per line')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    w_strong, w_weak, w_undef = nm(a.emnm, a.object)
    t_strong, t_weak, _ = nm(a.emnm, a.tint)
    g_strong, g_weak, _ = nm(a.emnm, a.webgpu)

    fails, warns, notes = [], [], []

    gl = sorted(s for s in w_strong | w_weak | w_undef if GL_RE.match(s))
    if gl:
        fails.append('GL entry points in wink.o (use wink_get_proc_address): ' + ' '.join(gl[:30]))
    shims = sorted(s for s in w_undef if SHIM_RE.match(s))
    if shims:
        fails.append('JS exception/setjmp shims referenced (missing -fwasm-exceptions?): ' + ' '.join(shims[:30]))
    if any(h in w_undef for h in SJLJ_HELPERS) and '__wasm_longjmp' not in w_undef:
        fails.append('setjmp is used but not in wasm mode: no __wasm_longjmp reference')
    # The setjmp helpers are the compiler's ABI with compiler-rt, and they were
    # renamed (saveSetjmp -> __wasm_setjmp) between LLVM builds that both call
    # themselves 19 and ship as Emscripten 3.1.56. .NET's workload pack pairs a
    # clang that emits the new names with a compiler-rt that has the old ones.
    if a.compiler_rt and os.path.exists(a.compiler_rt):
        rt_strong, rt_weak, _ = nm(a.emnm, a.compiler_rt)
        wanted = sorted(s for s in w_undef
                        if s in SJLJ_HELPERS + ('__wasm_setjmp', '__wasm_setjmp_test', '__wasm_longjmp'))
        missing = [s for s in wanted if s not in rt_strong | rt_weak]
        if missing:
            fails.append('setjmp helpers %s are not in %s: the compiler that built wink and this '
                         "emsdk's compiler-rt disagree" % (' '.join(missing), os.path.basename(a.compiler_rt)))
    webgl = sorted(s for s in w_undef if WEBGL_RE.match(s))
    if webgl:
        fails.append('WebGL referenced: ' + ' '.join(webgl))
    for name, strong in (('libwink_tint.a', t_strong), ('libwink_webgpu.a', g_strong)):
        dup = sorted(w_strong & strong)
        if dup:
            fails.append('defined in both wink.o and %s: %s' % (name, ' '.join(dup[:30])))

    if a.variant == 'st' and a.libc and os.path.exists(a.libc):
        c_strong, c_weak, _ = nm(a.emnm, a.libc)
        missing = sorted(s for s in w_undef if s.startswith('pthread_') and s not in c_strong | c_weak)
        if missing:
            fails.append('pthread functions the single-threaded libc lacks: ' + ' '.join(missing))
    elif a.variant == 'st':
        notes.append('st pthread check skipped: no libc.a given')

    thr = sorted(s for s in w_strong if re.match(r'^(thrd|mtx|cnd|tss|call_once)_?', s))
    if thr:
        warns.append('C11 threads defined by Mesa (libc has these names too): ' + ' '.join(thr))

    resolved = t_strong | t_weak | g_strong | g_weak
    unresolved = sorted(s for s in w_undef if s not in resolved and not JS_RE.match(s))

    collisions = set()
    for arch in filter(None, a.check_against.split(':')):
        a_strong, _, _ = nm(a.emnm, arch)
        clash = sorted(w_strong & a_strong)
        if clash:
            warns.append('%d collisions with %s: %s' % (len(clash), arch, ' '.join(clash[:40])))
            collisions.update(clash)
    if a.collisions_out:
        with open(a.collisions_out, 'w') as f:
            f.write(''.join(n + '\n' for n in sorted(collisions)))

    with open(os.path.join(a.out, 'defined.txt'), 'w') as f:
        f.write('\n'.join(sorted(w_strong | w_weak)) + '\n')
    with open(os.path.join(a.out, 'undefined.txt'), 'w') as f:
        f.write('\n'.join(sorted(w_undef)) + '\n')
    with open(os.path.join(a.out, 'external.txt'), 'w') as f:
        f.write('# undefined in wink.o and not provided by the Tint or webgpu archives or a\n'
                '# JS library: expected to come from libc/libc++/compiler-rt at the final link\n')
        f.write('\n'.join(unresolved) + '\n')
    with open(os.path.join(a.out, 'report.txt'), 'w') as f:
        for tag, lines in (('FAIL', fails), ('WARN', warns), ('NOTE', notes)):
            for l in lines:
                f.write('%s %s\n' % (tag, l))
        f.write('defined %d, undefined %d (%d external)\n' % (len(w_strong | w_weak), len(w_undef), len(unresolved)))

    for l in warns:
        print('warning: ' + l, file=sys.stderr)
    for l in notes:
        print('note: ' + l, file=sys.stderr)
    if fails:
        for l in fails:
            print('error: ' + l, file=sys.stderr)
        sys.exit(1)
    print('symbols ok: %d defined, %d undefined, %d external' % (len(w_strong | w_weak), len(w_undef), len(unresolved)))


if __name__ == '__main__':
    main()
