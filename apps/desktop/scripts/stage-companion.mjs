#!/usr/bin/env node
import { mkdirSync, copyFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve, join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { stageRemoteLifecycle } from './stage-remote-lifecycle.mjs';

const desktop = fileURLToPath(new URL('..', import.meta.url));
const source = resolve(desktop, '../companion-bridge');
const staged = join(desktop, 'src-tauri/resources/companion-bridge');
mkdirSync(staged, { recursive: true });
for (const name of ['bridge.mjs', 'index.mjs', 'qr.swift', 'package.json']) copyFileSync(join(source, name), join(staged, name));
if (process.platform === 'darwin') {
  const build = spawnSync('/usr/bin/swiftc', ['-O', join(source, 'qr.swift'), '-o', join(staged, 'qr-helper')], { stdio: 'inherit' });
  if (build.status !== 0) process.exit(build.status ?? 1);
  const sign = spawnSync('/usr/bin/codesign', ['--force', '--sign', '-', join(staged, 'qr-helper')], { stdio: 'inherit' });
  if (sign.status !== 0) process.exit(sign.status ?? 1);
}
console.log('Staged the optional iPhone pairing service.');
stageRemoteLifecycle();
console.log('Staged the optional headless lifecycle source.');
