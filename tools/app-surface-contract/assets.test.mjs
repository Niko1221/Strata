import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp, mkdir, readFile, rm, symlink, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {basename, dirname, join, resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {exportCandidate, inventory, verifyCandidate} from './assets.mjs';

const fingerprint = 'a'.repeat(64);
async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'strata-app-assets-'));
  t.after(async () => {
    if (dirname(resolve(root)) !== resolve(tmpdir()) || !basename(root).startsWith('strata-app-assets-'))
      throw Error('unsafe_fixture_cleanup');
    await rm(root, {recursive:true, force:true});
  });
  const hostDir = join(root, 'host');
  await mkdir(join(hostDir, 'dist/assets'), {recursive:true});
  await writeFile(join(hostDir, 'dist/index.html'), '<div id="root"></div><script src="/assets/app.js"></script>');
  await writeFile(join(hostDir, 'dist/assets/app.js'), '/* Synthetic official source bytes. */');
  const inspected = {hostDir, fingerprint, appVersion:'0.2.1', controlVersion:'0.1.46.3'};
  const snapshot = {schema:1, state:'compatible', app:{version:'0.2.1', controlVersion:'0.1.46.3', fingerprint},
    contexts:[8192,32768,65536,131072,204800,262144,393216,524288]};
  const output = join(root, 'candidate'), options = {reinspect:async () => inspected};
  const make = () => exportCandidate(inspected, output, snapshot, options);
  return {root, hostDir, inspected, snapshot, output, options, make};
}

test('export preserves source bytes and verifies only a pinned matching App identity', async t => {
  const f = await fixture(t), before = await inventory(join(f.hostDir, 'dist'));
  const receipt = await f.make();
  assert.equal(receipt.state, 'verified');
  assert.equal(receipt.activated, false);
  assert.equal(receipt.fileCount, 2);
  assert.deepEqual(await inventory(join(f.hostDir, 'dist')), before);
  assert.deepEqual(await inventory(join(f.output, 'dist')), before);
  await verifyCandidate(f.output, {manifestSha256:receipt.manifestSha256, current:f.snapshot});
  await assert.rejects(verifyCandidate(f.output, {manifestSha256:receipt.manifestSha256,
    current:{...f.snapshot, app:{...f.snapshot.app, fingerprint:'b'.repeat(64)}}}), /app_source_changed/);
  await assert.rejects(verifyCandidate(f.output, {manifestSha256:receipt.manifestSha256}), /needs_validation/);
});

test('changed asset bytes, extra files and changed manifests fail independently', async t => {
  const f = await fixture(t), receipt = await f.make();
  const verify = () => verifyCandidate(f.output, {manifestSha256:receipt.manifestSha256, current:f.snapshot});
  const js = join(f.output, 'dist/assets/app.js'), original = await readFile(js);
  await writeFile(js, 'modified');
  await assert.rejects(verify(), /candidate_assets_changed/);
  await writeFile(js, original);
  const extra = join(f.output, 'unexpected.txt');
  await writeFile(extra, 'extra');
  await assert.rejects(verify(), /candidate_inventory_changed/);
  await rm(extra);
  const manifest = join(f.output, 'manifest.json');
  await writeFile(manifest, (await readFile(manifest, 'utf8')) + ' ');
  await assert.rejects(verify(), /candidate_manifest_changed/);
});

test('export never overwrites a candidate or writes into the installed App', async t => {
  const f = await fixture(t);
  const receipt = await f.make();
  const before = await readFile(join(f.output, 'manifest.json'));
  await assert.rejects(f.make(), {code:'EEXIST'});
  assert.deepEqual(await readFile(join(f.output, 'manifest.json')), before);
  await assert.rejects(exportCandidate(f.inspected, join(f.hostDir, 'candidate'), f.snapshot, f.options), /outside_app/);
  assert.equal(receipt.activated, false);
});

test('an App changing during export leaves no valid candidate manifest', async t => {
  const f = await fixture(t);
  await assert.rejects(exportCandidate(f.inspected, f.output, f.snapshot,
    {reinspect:async () => ({...f.inspected, fingerprint:'b'.repeat(64)})}), /app_source_changed/);
  await assert.rejects(readFile(join(f.output, 'manifest.json')), {code:'ENOENT'});
});

test('directory links in official assets are rejected before copying', async t => {
  const f = await fixture(t);
  const outside = join(f.root, 'outside');
  await mkdir(outside);
  await writeFile(join(outside, 'private.txt'), 'not an asset');
  await symlink(outside, join(f.hostDir, 'dist/linked'), process.platform === 'win32' ? 'junction' : 'dir');
  await assert.rejects(f.make(), /linked_asset_path/);
});

test('an explicitly repinned malformed inventory still cannot traverse out of the candidate', async t => {
  const f = await fixture(t);
  await f.make();
  const path = join(f.output, 'manifest.json'), manifest = JSON.parse(await readFile(path));
  manifest.files[0].path = '../outside.txt';
  const raw = Buffer.from(JSON.stringify(manifest));
  await writeFile(path, raw);
  await assert.rejects(verifyCandidate(f.output, {manifestSha256:createHash('sha256').update(raw).digest('hex'), current:f.snapshot}), /invalid_asset_name/);
});
