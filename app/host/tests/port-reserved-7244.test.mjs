// #7244: on a Windows PC with WSL mirrored networking the usual port 47831 sits in a block the system keeps back (netstat does not list it, a bind is
// answered like a port in use), so the app took a new random port at every start and said another program used 47831. Now: the app asks for 47831
// first, remembers the port it took (shell-fallback-port.json, written by the shell only) and uses it again, and the notice says the port is
// reserved by the system when nothing listens on it. The Rust side is tested in src-tauri/src/shell.rs (cargo test); this file pins the host and the page.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { parsePortNotice } from '../server/shell-prefs.mjs';

test('the host reads the reserved notice and keeps the older ones as they were', () => {
  assert.deepEqual(parsePortNotice('reserved:47831:30159'), { wanted: 47831, used: 30159, reserved: true });
  assert.deepEqual(parsePortNotice('busy:47831:51234'), { wanted: 47831, used: 51234 });
  assert.deepEqual(parsePortNotice('busy:8080:51234:terminal'), { wanted: 8080, used: 51234, terminal: true });
  assert.equal(parsePortNotice('reserved:47831:30159:terminal'), null, 'a Terminal model server is never "reserved"');
  assert.equal(parsePortNotice('reserved:x:1'), null);
  assert.equal(parsePortNotice('reserved:47831'), null);
  assert.equal(parsePortNotice('taken:47831:30159'), null);
});

test('the page says the port is reserved, in every language, with both port numbers and the way out', async () => {
  const store = await readFile(new URL('../src/lib/store.ts', import.meta.url), 'utf8');
  const en = JSON.parse(await readFile(new URL('../src/locales/en.json', import.meta.url), 'utf8'));
  assert.match(store, /shell\.portNotice\?\.reserved\)pushToast\('warn',tr\('Port \{0\} is reserved by the system',\[shell\.portNotice\.wanted\]\),tr\('/);
  assert.ok(store.includes(`tr('${en.portReservedBody}',[shell.portNotice.wanted,shell.portNotice.used])`), 'the toast uses the catalog sentence word for word');
  assert.match(en.portReservedBody, /WSL/);
  assert.match(en.portReservedBody, /Settings and set App port/);
  for (const lang of ['en', 'zh', 'ja', 'de', 'es', 'ru', 'pt', 'ko', 'fr', 'he']) {
    const catalog = JSON.parse(await readFile(new URL(`../src/locales/${lang}.json`, import.meta.url), 'utf8'));
    assert.ok(catalog.portReservedTitle?.includes('{0}'), `${lang}: title`);
    assert.ok(catalog.portReservedBody?.includes('{0}') && catalog.portReservedBody.includes('{1}'), `${lang}: body names both ports`);
    assert.ok(!/\bport 47831\b/.test(catalog.portReservedBody), `${lang}: no number written into the sentence`);
  }
});

test('the shell asks for the wanted port first, then the remembered one, and writes the port it picked', async () => {
  const main = await readFile(new URL('../../desktop/src-tauri/src/main.rs', import.meta.url), 'utf8');
  const wanted = main.indexOf('TcpListener::bind(("127.0.0.1", wanted))'), remembered = main.indexOf('shell::read_fallback(wanted)'), write = main.indexOf('shell::write_fallback(wanted, port)');
  assert.ok(wanted > 0 && remembered > wanted && write > remembered, 'wanted port, then the remembered one, then a new one that is saved');
  assert.match(main, /const PREFERRED_PORT: u16 = 47831;/, 'the default port stays');
  assert.match(main, /shell::holder_of\(wanted\) == shell::Holder::Nobody/);
  assert.match(main, /shell::reserved_notice\(wanted, used\)/);
  const shell = await readFile(new URL('../../desktop/src-tauri/src/shell.rs', import.meta.url), 'utf8');
  assert.match(shell, /"shell-fallback-port\.json"/);
});
