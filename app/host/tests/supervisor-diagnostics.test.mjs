import test from 'node:test';
import assert from 'node:assert/strict';
import * as supervisor from '../server/supervisor.mjs';

const collector=secret=>{assert.equal(typeof supervisor.createStartupDiagnostics,'function');return supervisor.createStartupDiagnostics(secret);};
const root="0.02.912.349 E llama_model_load: error loading model: check_tensor_dims: tensor 'blk.32.attn_norm.weight' not found";
test('first actionable loader cause survives generic failure cascade and the 400-line ring limit',()=>{
 const d=collector();d.consume('err','warning: fit failed, retrying\n'+root+'\n');
 for(let i=0;i<500;i++)d.consume('err',`progress ${i}\n`);
 for(const line of ['common_init_from_params: failed to load model','srv load_model: failed to load model','main: exiting due to model loading error'])d.consume('err',line+'\n');
 const text=d.summary();assert.match(text,/blk\.32\.attn_norm\.weight.*not found/);
 assert.ok(text.length<2000);assert.match(text,/exiting due to model loading error/);
});
test('chunk boundaries, final unterminated line, ANSI and secrets are handled before retaining errors',()=>{
 const d=collector('secret-fixture');
 d.consume('err',Buffer.from('\x1b[31mllama_model_load: error loading model: check_tensor_'));
 d.consume('err',Buffer.from("dims: tensor 'blk.32.attn_norm.weight' not found secret-"));
 d.consume('err',Buffer.from('fixture\x1b[0m\nfailed to load model'));
 d.flush();const text=d.summary();
 assert.match(text,/check_tensor_dims/);assert.match(text,/\[redacted\]/);assert.match(text,/failed to load model/);
 assert.ok(!text.includes('secret-fixture'));assert.ok(!text.includes('\x1b'));
});
test('stdout does not masquerade as an engine failure and collectors do not leak between runs',()=>{
 const d=collector();d.consume('out',root+'\n');assert.equal(d.summary(),'');
 d.consume('err','bad option: -m\n');assert.match(d.summary(),/bad option: -m/);
 assert.equal(collector().summary(),'');
});
test('unbounded unterminated diagnostics stay bounded and a later actual error is retained',()=>{
 const d=collector();for(let i=0;i<100;i++)d.consume('err','x'.repeat(8192));
 d.consume('err','\n'+root+'\n');assert.match(d.summary(),/blk\.32/);assert.ok(d.summary().length<2000);
});
