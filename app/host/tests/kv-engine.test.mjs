import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { kvMemoryFactor, requireModelKvCompatible } from '../server/kv-settings.mjs';

test('engine validation returns issues rather than throwing on invalid K/V', async t => {
  const prior = process.env.STRATA_HOME;
  const root = mkdtempSync(join(tmpdir(), 'strata-kv-validation-'));
  process.env.STRATA_HOME = root;
  t.after(() => { if(prior===undefined) delete process.env.STRATA_HOME; else process.env.STRATA_HOME=prior; rmSync(root,{recursive:true,force:true}); });
  const { validate } = await import('../server/engine.mjs');
  const { VARIANTS } = await import('../server/catalog.mjs');
  assert.ok(VARIANTS.length > 0);
  const result = await validate({variantId: VARIANTS[0].id, expert:[{id:'ctx.kType',value:'invalid-type'}]});
  assert.equal(result.ok,false);
  assert.equal(result.predicted,null);
  assert.ok(result.issues.some(x=>x.key==='ctx.kType' && x.level==='error'));
});
test('mixed precision is reflected in nominal memory planning', () => {
  assert.equal(kvMemoryFactor([{id:'ctx.kType',value:'FP16'},{id:'ctx.vType',value:'Q4_0'}]),1.25);
});
test('narrow GGUF KV width requires explicit FP16 for both cache tensors', () => {
  const model={local:{path:'fixture.gguf'},kvModes:['FP16']};
  assert.throws(()=>requireModelKvCompatible(model,[]),{code:'KV_MODEL_WIDTH'});
  assert.throws(()=>requireModelKvCompatible(model,[{id:'ctx.kvMode',value:'FP16'},{id:'ctx.vType',value:'Q8_0'}]),{code:'KV_MODEL_WIDTH'});
  assert.doesNotThrow(()=>requireModelKvCompatible(model,[{id:'ctx.kvMode',value:'FP16'}]));
});
