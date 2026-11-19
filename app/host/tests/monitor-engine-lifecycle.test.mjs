import test from 'node:test';
import assert from 'node:assert/strict';
import * as monitor from '../server/monitor.mjs';

test('engine failure clears loading, context and model identity; unrelated engines do not',async()=>{
 await monitor.beginLoading('local-fixture');
 monitor.setContext({messages:500});
 monitor.engineStopped({modelId:'another',status:'error',error:'fixture'});
 assert.equal(monitor.snapshot().phase,'loading');
 monitor.engineStopped({modelId:'local-fixture',status:'error',error:'fixture'});
 const stopped=monitor.snapshot();
 assert.equal(stopped.phase,'idle');
 assert.equal(stopped.variantId,null);
 assert.equal(stopped.context.usedTokens,0);
 assert.equal(stopped.phaseLabel,'Model stopped after an error');
});
