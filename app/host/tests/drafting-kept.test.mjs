// A Drafting choice made on the Models screen is kept for the next start (0.2.15 release test T6-app, p1004, C2 app): the Load saved
// "drafting": "off" into the model's run config (prepareDrafter / rememberSetting), and the write at ready then put back the copy of the
// run config the Load had read before that, so the key was gone: after an app restart the page showed Auto and the next Load ran DFlash2.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { nativeFixture, finishJob } from './helpers/native-fixture.mjs';

const source = readFileSync(fileURLToPath(new URL('../server/native-control.mjs', import.meta.url)), 'utf8').replace(/\r\n/g, '\n');

test('the Load saves the Drafting choice before the engine starts', () => {
  const load = source.slice(source.indexOf('async function loadEngine'), source.indexOf('const fingerprint=createHash'));
  assert.match(load, /await prepareDrafter\(job,args,cfg,configFile,drafts,\{noSplit\}\)/);
  const prepare = source.slice(source.indexOf('async function prepareDrafter'), source.indexOf('function rememberSetting'));
  assert.match(prepare, /rememberSetting\(configFile,'drafting',asked===null\|\|asked==='auto'\?null:asked\)/);
});

test('the run-config write at ready starts from the file as it is now, so the saved choice is kept', () => {
  const at = source.indexOf("instance.status='ready';instance.phase='idle';await noteDrafting(job,instance,key);");
  assert.ok(at > 0, 'the ready branch of loadEngine');
  const ready = source.slice(at, source.indexOf('emit();ensurePoll();', at));
  assert.match(ready, /const current=readJson\(path\);[\s\S]*const selected=[^;]*persistOrdinaryKv\(current,config,args\.kv,policyFor\(args\.model,config\)\)/,
    'the write at ready reads the run config again (it holds the Drafting choice the Load saved)');
  assert.match(ready, /writeModelConfig\(path,\{\.\.\.selected,/);
  assert.doesNotMatch(ready, /writeModelConfig\(record\(args\.model\)\.path,\{\.\.\.config,/, 'not the copy read before the choice was saved');
  assert.match(ready, /instance\.configSave=\{status:'failed'/, 'save failure is reported without undoing READY');
});

test('ordinary load keeps the selected Drafting choice when READY persists resource settings', async () => {
  const f = nativeFixture();
  try {
    await f.control.refresh(true);
    const done = await finishJob(f.control, f.control.operation('load', { model: 'iq2_xs', kv: 'fp16', values: { drafting: 'off' } }).id);
    assert.equal(done.state, 'done', done.error);
    assert.equal(f.control.state().instances[0].configSave.status, 'saved');
    const saved = JSON.parse(readFileSync(join(f.repo, 'strata-iq2_xs.json'), 'utf8'));
    assert.equal(saved.drafting, 'off');
  } finally { await f.cleanup(); }
});

test('after an app restart the Models screen shows the saved Drafting choice: the page follows the saved value when it arrives', () => {
  const screen = readFileSync(fileURLToPath(new URL('../src/components/NativeModels.tsx', import.meta.url)), 'utf8').replace(/\r\n/g, '\n');
  const at = screen.indexOf("setDrafting(installed.drafting ?? 'auto');");
  assert.ok(at > 0, 'the page opens a model with its saved choice');
  const deps = screen.slice(screen.indexOf('}, [', at), screen.indexOf(']);', at));
  assert.match(deps, /installed\?\.drafting/, 'a saved choice that arrives after the page opened is shown (otherwise the next Load sends Auto)');
});
