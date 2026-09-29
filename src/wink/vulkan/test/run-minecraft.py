#!/usr/bin/env python3
# Run a Prism Launcher instance of Minecraft on the wink Vulkan driver, or
# with --gl on wink's GL (libwink, through the SDL fork's Cocoa backend).
#
#   run-minecraft.py INSTANCE [--gl] [--world NAME] [--capture N] [--seconds S] [-- game args]
#
# Offline, in a scratch game directory under $TMPDIR (or --game-dir): the
# instance's options.txt, and with --world a copy of that save, so nothing of
# the instance is written to. The Khronos loader (brew's) is what LWJGL and
# SDL load, with VK_DRIVER_FILES naming this build's ICD. With --gl, LWJGL
# loads --sdl (the fork, built with SDL_COCOA_WINK_INCLUDE_DIR) and --wink
# as its GL, which SDL_VIDEO_COCOA_WINK makes SDL's too. --capture N writes the Nth present to
# capture.bmp there (WINK_VK_CAPTURE, or WINK_CAPTURE). The game is stopped
# after --seconds, if given.
import argparse, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MESA = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
PRISM = os.path.expanduser('~/Library/Application Support/PrismLauncher')

ap = argparse.ArgumentParser()
ap.add_argument('instance')
ap.add_argument('--world')
ap.add_argument('--capture', type=int)
ap.add_argument('--seconds', type=float)
ap.add_argument('--game-dir')
ap.add_argument('--loader', default='/opt/homebrew/lib/libvulkan.1.dylib')
ap.add_argument('--icd', default=os.path.join(MESA, 'build/wvk-native/src/wink/vulkan/wink_devenv_icd.aarch64.json'))
ap.add_argument('--gl', action='store_true')
ap.add_argument('--sdl', default=os.path.join(MESA, 'build/sdl-mac/libSDL3.0.dylib'))
ap.add_argument('--wink', default=os.path.join(MESA, 'build/wink-mac/src/gallium/targets/wink-wasm/libwink.dylib'))
argv = sys.argv[1:]
rest = argv[argv.index('--') + 1:] if '--' in argv else []
a = ap.parse_args(argv[:argv.index('--')] if '--' in argv else argv)

inst = os.path.join(PRISM, 'instances', a.instance)
pack = json.load(open(os.path.join(inst, 'mmc-pack.json')))
version = [c for c in pack['components'] if c['uid'] == 'net.minecraft'][0]['version']
lwjgl = [c for c in pack['components'] if c['uid'] == 'org.lwjgl3'][0]['version']

def lib_path(name):
    parts = name.split(':')
    g, art, v = parts[:3]
    fn = f'{art}-{v}' + (f'-{parts[3]}' if len(parts) > 3 else '') + '.jar'
    return os.path.join(PRISM, 'libraries', *g.split('.'), art, v, fn)

def allowed(lib):
    ok = not lib.get('rules')
    for r in lib.get('rules', []):
        o = r.get('os', {})
        if not o or (o.get('name') in ('osx', 'osx-arm64') and o.get('arch', 'arm64') in ('arm64', 'aarch64')):
            ok = r['action'] == 'allow'
    return ok

mc = json.load(open(os.path.join(PRISM, 'meta/net.minecraft', version + '.json')))
lw = json.load(open(os.path.join(PRISM, 'meta/org.lwjgl3', lwjgl + '.json')))
cp = [p for p in (lib_path(l['name']) for l in lw['libraries'] + mc['libraries'] if allowed(l)) if os.path.exists(p)]
cp.append(os.path.join(PRISM, 'libraries/com/mojang/minecraft', version, f'minecraft-{version}-client.jar'))

game = a.game_dir or tempfile.mkdtemp(prefix=f'wink-mc-{a.instance}-')
os.makedirs(os.path.join(game, 'saves'), exist_ok=True)
if os.path.exists(os.path.join(inst, 'minecraft/options.txt')):
    shutil.copy(os.path.join(inst, 'minecraft/options.txt'), game)
extra = []
if a.world:
    dst = os.path.join(game, 'saves', a.world)
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(os.path.join(inst, 'minecraft/saves', a.world), dst,
                    ignore=shutil.ignore_patterns('session.lock'))
    extra = ['--quickPlaySingleplayer', a.world]

cfg = open(os.path.join(inst, 'instance.cfg')).read()
java = cfg.split('JavaPath=')[1].split('\n')[0] if 'JavaPath=' in cfg else 'java'
args = [java, '-XstartOnFirstThread', *mc.get('+jvmArgs', []), '-cp', ':'.join(cp), mc['mainClass'],
        '--username', 'wink', '--version', version, '--gameDir', game,
        '--assetsDir', os.path.join(PRISM, 'assets'), '--assetIndex', mc['assetIndex']['id'],
        '--uuid', '0' * 32, '--accessToken', '0',
        '--graphicsBackend', 'opengl' if a.gl else 'vulkan', *extra, *rest]

if a.gl:
    env = dict(os.environ,
               JAVA_TOOL_OPTIONS=f'-Dorg.lwjgl.sdl.libname={a.sdl} -Dorg.lwjgl.opengl.libname={a.wink}',
               SDL_VIDEO_COCOA_WINK=a.wink)
else:
    env = dict(os.environ,
               JAVA_TOOL_OPTIONS=f'-Dorg.lwjgl.vulkan.libname={a.loader}',
               SDL_VULKAN_LIBRARY=a.loader,
               VK_DRIVER_FILES=a.icd)
if a.capture:
    env['WINK_CAPTURE' if a.gl else 'WINK_VK_CAPTURE'] = f'{a.capture}:{os.path.join(game, "capture.bmp")}'
print('game directory:', game, file=sys.stderr)
p = subprocess.Popen(args, env=env, cwd=game)
try:
    p.wait(timeout=a.seconds)
except subprocess.TimeoutExpired:
    p.terminate()
    try:
        p.wait(10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
sys.exit(0 if p.returncode in (0, -9, -15, 137, 143) else p.returncode)
