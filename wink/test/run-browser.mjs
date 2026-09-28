#!/usr/bin/env node
// SPDX-License-Identifier: MIT
//
// Serve a directory, open its index.html in headless Chrome, relay the
// console, and wait for the page to set globalThis.__winkResult.
//
//   node run-browser.mjs <dir> [--page=index.html] [--timeout=SECONDS]
//
// The server always sends COOP/COEP: the mt build needs SharedArrayBuffer,
// and the st build does not mind them. CHROME overrides the browser;
// WINK_WEBGPU_SOFTWARE=1 asks Linux Chrome for SwiftShader's WebGPU.
// Needs Node 22+ (the global WebSocket).
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from 'node:fs';
import { tmpdir, platform } from 'node:os';
import { extname, join, normalize } from 'node:path';

const dir = process.argv[2];
const timeout = Number((process.argv.find((a) => a.startsWith('--timeout=')) || '=120').split('=')[1]) * 1000;
const pageName = (process.argv.find((a) => a.startsWith('--page=')) || '=index.html').split('=')[1];
if (!dir || !existsSync(join(dir, pageName))) {
  console.error('usage: run-browser.mjs <dir> [--page=index.html] [--timeout=SECONDS]');
  process.exit(2);
}

const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
                '.wasm': 'application/wasm', '.json': 'application/json' };
const server = createServer((req, res) => {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
  const file = join(dir, path === '/' ? 'index.html' : path);
  if (!existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404);
    res.end();
    return;
  }
  res.writeHead(200, {
    'Content-Type': TYPES[extname(file)] || 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cache-Control': 'no-store',
  });
  res.end(readFileSync(file));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const url = `http://127.0.0.1:${server.address().port}/${pageName}`;

function findChrome() {
  if (process.env.CHROME) return process.env.CHROME;
  const candidates = platform() === 'darwin'
    ? ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
       '/Applications/Chromium.app/Contents/MacOS/Chromium']
    : ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome-stable', '/usr/bin/google-chrome'];
  return candidates.find((c) => existsSync(c));
}
const chrome = findChrome();
if (!chrome) {
  console.error('no Chrome found; set CHROME=/path/to/chrome');
  process.exit(2);
}

const profile = mkdtempSync(join(tmpdir(), 'wink-chrome-'));
const args = [
  '--headless=new', '--no-first-run', '--no-default-browser-check',
  '--enable-unsafe-webgpu', '--disable-hang-monitor', '--password-store=basic',
  '--remote-debugging-port=0', '--remote-allow-origins=*', `--user-data-dir=${profile}`,
];
if (platform() === 'linux') {
  args.push(...(process.env.WINK_WEBGPU_SOFTWARE === '1'
    ? ['--use-angle=swiftshader-webgl', '--use-webgpu-adapter=swiftshader']
    : ['--use-angle=vulkan', '--enable-features=Vulkan']));
  args.push('--ozone-platform=headless');
}
args.push('about:blank');
const browser = spawn(chrome, args, { stdio: ['ignore', 'ignore', 'pipe'] });

// Chrome prints its DevTools endpoint once it is listening.
const wsBrowser = await new Promise((resolve, reject) => {
  let buf = '';
  browser.stderr.on('data', (d) => {
    buf += d;
    const m = buf.match(/DevTools listening on (ws:\/\/\S+)/);
    if (m) resolve(m[1]);
  });
  browser.on('exit', (c) => reject(new Error(`chrome exited (${c}) before DevTools was up`)));
});
const port = new URL(wsBrowser).port;
const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const page = targets.find((t) => t.type === 'page');

// The smallest CDP client that does: id-correlated calls plus events.
const ws = new WebSocket(page.webSocketDebuggerUrl);
let nextId = 0;
const pending = new Map();
ws.addEventListener('message', (e) => {
  const msg = JSON.parse(e.data);
  if (msg.id !== undefined && pending.has(msg.id)) {
    pending.get(msg.id)(msg);
    pending.delete(msg.id);
  } else if (msg.method === 'Runtime.consoleAPICalled') {
    console.log('[page]', msg.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
  } else if (msg.method === 'Runtime.exceptionThrown') {
    console.log('[page exception]', msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
  }
});
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
const call = (method, params = {}) => new Promise((resolve) => {
  const id = ++nextId;
  pending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});

let result = null;
let code = 1;
try {
  await call('Runtime.enable');
  await call('Page.navigate', { url });
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const r = await call('Runtime.evaluate', { expression: 'JSON.stringify(globalThis.__winkResult ?? null)', returnByValue: true });
    const v = r.result?.result?.value;
    if (v && v !== 'null') {
      result = JSON.parse(v);
      break;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  if (!result) console.log(`timed out after ${timeout / 1000}s`);
  else code = result.pass ? 0 : 1;
  console.log(result ? (result.pass ? 'PASS' : `FAIL (rc ${result.rc})`) : 'FAIL (no result)');
} finally {
  ws.close();
  browser.kill();
  server.close();
  await new Promise((r) => setTimeout(r, 200));
  rmSync(profile, { recursive: true, force: true });
}
process.exit(code);
