// SPDX-License-Identifier: MIT
import { dotnet } from './_framework/dotnet.js';

const log = [];
globalThis.__winkResult = null;
try {
  const { runMain } = await dotnet
    .withConfig({ environmentVariables: {} })
    .withModuleConfig({
      print: (s) => { log.push(s); console.log(s); },
      printErr: (s) => { log.push(s); console.warn(s); },
    })
    .create();
  const rc = await runMain();
  globalThis.__winkResult = { pass: rc === 0, rc, log };
} catch (e) {
  console.error('exception: ' + (e && e.stack || e));
  globalThis.__winkResult = { pass: false, rc: -1, log };
}
