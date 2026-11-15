/** Read-only release-input checks. No runtime/model or network requests. */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {pathToFileURL} from 'node:url';

const [mode, rootArg, inputArg, commit, receiptArg] = process.argv.slice(2);
const root = resolve(rootArg), input = resolve(inputArg);
const mod = async name => import(pathToFileURL(join(root, name)).href);
const {sha256, nativeSources, ggmlIdentity, verifyEngine, verifyNativeBuild,
  verifyPublicAssets} = await mod('apps/host/scripts/release-integrity.mjs');
const json = path => JSON.parse(readFileSync(path, 'utf8').replace(/^\uFEFF/, ''));
if (mode === 'dependencies') {
  const harness = join(input, 'harness'), bundle = json(join(harness, 'BUNDLE.json'));
  assert.equal(bundle.commit, '571902028a75c64c1d767f4ae5fd099485f0ddf0');
  const {HARNESS_BUDGET_PATCH: patch} = await mod('apps/desktop/scripts/harness-budget.mjs');
  const actual = bundle.patches?.find(row => row.id === patch.id);
  assert.equal(actual?.beforeSha256, patch.beforeSha256);
  assert.equal(actual?.afterSha256, sha256(join(harness, patch.file)));
  const {missingHarnessImports} = await mod('apps/desktop/scripts/harness-prune.mjs');
  assert.deepEqual(missingHarnessImports(harness), []);
  const search = join(harness, 'rustxng'), source = json(join(search, 'SOURCE.json'));
  const {RUSTXNG_VERSION, RUSTXNG_SHA256} = await mod('apps/host/server/rustxng-runtime.mjs');
  assert.equal(source.version, RUSTXNG_VERSION);
  assert.equal(source.profile, 'full');
  assert.equal(source.sourceCommit, 'a8ba33d8c9ceed91236f3abf029254d0f84995bb');
  assert.equal(sha256(join(search, 'rustxng.exe')), RUSTXNG_SHA256);
  for (const name of ['LICENSE', 'NOTICE', 'corresponding-source.tar.gz'])
    assert.equal(sha256(join(search, name)), source.files[name].sha256);
  const {LOCAL_BACKENDS, findLocalBackend, verifyLocalBackend} = await mod('apps/host/server/local-backends.mjs');
  for (const choice of Object.keys(LOCAL_BACKENDS)) {
    const manifest = await verifyLocalBackend(findLocalBackend(choice, {installRoot: input}), choice);
    assert.equal(manifest.version, LOCAL_BACKENDS[choice].packageVersion);
  }
  verifyPublicAssets('H:/Strata/releases/hakua033-public-assets', json('H:/Strata/releases/hakua033-public-assets.receipt.json'));
} else if (mode === 'native') {
  assert.match(commit, /^[a-f0-9]{40}$/);
  assert.equal(execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], {encoding:'utf8'}).trim(), commit);
  const record = json(receiptArg), manifest = verifyEngine(input, '0.6.0');
  assert.equal(record.sourceCommit, commit);
  assert.equal(resolve(record.sourceRoot), root);
  assert.equal(record.releaseQualified, false);
  assert.equal(record.nativeBuildSHA256, sha256(join(input, 'native-build.json')));
  assert.equal(manifest.build_id.replace(/-hakua-runtimes-v1$/, ''), commit);
  for (const binary of ['strata.exe', 'strata-q36.exe'])
    assert.equal(record.binaries[binary], sha256(join(input, binary)));
  const build = json(join(input, 'native-build.json'));
  assert.deepEqual(build.sources, nativeSources(root));
  assert.deepEqual(build.recipe.ggml, ggmlIdentity(root));
  assert.equal(build.recipe.ggml.mode, 'vendored');
  verifyNativeBuild(input, root, manifest, {sourceRoot:root, sourceCommit:commit,
    manifestSHA256:sha256(join(input, 'engine-component.json'))});
} else throw new Error('Unknown verification mode');
console.log(JSON.stringify({ok:true, mode, releaseQualified:false}));
