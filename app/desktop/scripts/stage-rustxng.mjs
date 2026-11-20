#!/usr/bin/env node
/** Pin official RustXNG bytes and ship their notices plus the exact corresponding source. */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { RUSTXNG_VERSION, RUSTXNG_SHA256 } from '../../host/server/rustxng-runtime.mjs';

const commit = 'a8ba33d8c9ceed91236f3abf029254d0f84995bb';
const root = resolve(import.meta.dirname, '../../..');
const destination = join(root, 'apps/desktop/src-tauri/resources/harness/rustxng');
const release = `https://github.com/sahenjp/RustXNG/releases/download/v${RUSTXNG_VERSION}/`;
const inputs = [
  { name: 'rustxng.exe', url: release + `rustxng-v${RUSTXNG_VERSION}-x86_64-pc-windows-msvc.exe`, sha256: RUSTXNG_SHA256 },
  { name: 'LICENSE', url: release + 'LICENSE', sha256: '57c8ff33c9c0cfc3ef00e650a1cc910d7ee479a8bc509f6c9209a7c2a11399d6' },
  { name: 'NOTICE', url: release + 'NOTICE', sha256: '0740830f4a7b153429270002eee52506fbc1fefcd55e1aad27d76ef3262853b6' },
  { name: 'corresponding-source.tar.gz', url: `https://codeload.github.com/sahenjp/RustXNG/tar.gz/${commit}`, sha256: '927abe258fe2257a90b8a7467cc8b547ecb816d352ce493a0d646fc95960ba61' },
];
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
if (!process.argv.includes('--execute')) {
  console.log(JSON.stringify({ execute: false, destination, profile: 'full', commit, inputs }, null, 2));
} else {
  mkdirSync(destination, { recursive: true });
  const files = {};
  for (const input of inputs) {
    const path = join(destination, input.name);
    let bytes;
    if (existsSync(path)) bytes = readFileSync(path);
    else {
      const response = await fetch(input.url, { signal: AbortSignal.timeout(120_000) });
      if (!response.ok) throw new Error(`Download failed for ${input.name}: ${response.status}`);
      bytes = Buffer.from(await response.arrayBuffer());
      if (bytes.length > 64 * 1024 * 1024 || bytes.length < 100) throw new Error('Unexpected RustXNG download size.');
    }
    const digest = sha(bytes);
    if (input.sha256 && input.sha256 !== digest) throw new Error(`RustXNG checksum mismatch: ${input.name}`);
    if (!existsSync(path)) writeFileSync(path, bytes, { flag: 'wx' });
    files[input.name] = { url: input.url, sha256: digest, bytes: bytes.length };
  }
  const record = { version: RUSTXNG_VERSION, profile: 'full', sourceCommit: commit, license: 'AGPL-3.0-or-later', unmodifiedOfficialBinary: true, files };
  const manifest = join(destination, 'SOURCE.json');
  if (existsSync(manifest)) {
    if (JSON.stringify(JSON.parse(readFileSync(manifest, 'utf8'))) !== JSON.stringify(record)) throw new Error('Existing RustXNG provenance differs; preserve and review it first.');
  } else writeFileSync(manifest, JSON.stringify(record, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify(record, null, 2));
}
