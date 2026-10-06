#!/usr/bin/env node
// SPDX-License-Identifier: MIT
//
// Serve a directory, open its index.html in headless Chrome, relay the
// console, and wait for the page to set globalThis.__winkResult.
//
//   node run-browser.mjs <dir> [--page=index.html] [--timeout=SECONDS]
//                               [--pause-after=SECONDS]  (print where the page is)
//
// The server always sends COOP/COEP: the mt build needs SharedArrayBuffer,
// and the st build does not mind them. CHROME overrides the browser;
// WINK_WEBGPU_SOFTWARE=1 asks Linux Chrome for SwiftShader's WebGPU.
// Needs Node 22+ (the global WebSocket).
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir, platform } from 'node:os';
import { basename, dirname, extname, join, normalize } from 'node:path';

const dir = process.argv[2];
const timeout = Number((process.argv.find((a) => a.startsWith('--timeout=')) || '=120').split('=')[1]) * 1000;
const pageArg = process.argv.find((a) => a.startsWith('--page=')) || '--page=index.html';
const pageName = pageArg.slice(pageArg.indexOf('=') + 1);
const resultArg = process.argv.find(a => a.startsWith('--result='));
const resultPath = resultArg?.slice('--result='.length);
// A page may take a query (vk.html?test=...): the file is what precedes it.
if (!dir || !existsSync(join(dir, pageName.split('?')[0]))) {
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
  } else if (msg.method === 'Debugger.paused') {
    // --pause-after: where the page was, innermost first.
    console.log('[paused]\n  ' + msg.params.callFrames.map((f) => f.functionName || '(anonymous)').join('\n  '));
  } else if (msg.method === 'Runtime.exceptionThrown') {
    console.log('[page exception]', msg.params.exceptionDetails.exception?.description || msg.params.exceptionDetails.text);
  }
});
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
// A page busy in wasm answers nothing: every call gives up after 10s, so
// the overall timeout still ends the run.
const call = (method, params = {}) => new Promise((resolve) => {
  const id = ++nextId;
  const timer = setTimeout(() => { pending.delete(id); resolve({}); }, 10000);
  pending.set(id, (msg) => { clearTimeout(timer); resolve(msg); });
  ws.send(JSON.stringify({ id, method, params }));
});
const pauseAfter = Number((process.argv.find((a) => a.startsWith('--pause-after=')) || '=0').split('=')[1]);

let result = null;
let code = 1;
try {
  await call('Runtime.enable');
  await call('Page.navigate', { url });
  const deadline = Date.now() + timeout;
  if (pauseAfter) {
    await call('Debugger.enable');
    setTimeout(() => { call('Debugger.pause'); }, pauseAfter * 1000);
  }
  while (Date.now() < deadline) {
    const r = await call('Runtime.evaluate', { expression: 'JSON.stringify(globalThis.__winkResult ?? null)', returnByValue: true });
    const v = r.result?.result?.value;
    if (v && v !== 'null') {
      result = JSON.parse(v);
      break;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
  if (!result) {
    console.log(`timed out after ${timeout / 1000}s`);
    if (resultPath) {
      mkdirSync(dirname(resultPath), { recursive: true });
      writeFileSync(resultPath, JSON.stringify({ pass: false, rc: -1, error: 'timeout', artifacts: [] }) + '\n');
    }
  }
  else {
    code = result.pass ? 0 : 1;
    if (resultPath) {
      mkdirSync(dirname(resultPath), { recursive: true });
      const artifacts = [];
      for (const item of result.artifacts || []) {
        if (item.name !== basename(item.name) || !/^[A-Za-z0-9_.-]+\.png$/.test(item.name) || item.encoding !== 'base64') {
          throw new Error('invalid browser replay artifact');
        }
        const bytes = Buffer.from(item.data, 'base64');
        writeFileSync(join(dirname(resultPath), item.name), bytes);
        artifacts.push({ name: item.name, bytes: bytes.length });
      }
      writeFileSync(resultPath, JSON.stringify({ ...result, artifacts }, null, 2) + '\n');
    }
  }
  console.log(result ? (result.pass ? 'PASS' : `FAIL (rc ${result.rc})`) : 'FAIL (no result)');
} finally {
  ws.close();
  browser.kill();
  server.close();
  await new Promise((r) => setTimeout(r, 200));
  rmSync(profile, { recursive: true, force: true });
}
process.exit(code);
