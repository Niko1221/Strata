// #6772: the model's settings file is also edited by hand (Expert settings > Open settings file: "env": {"STRATA_HC_Q8_TC": "0", ...}). The settings panel loads the values when it
// opens and saves the WHOLE form back, so an entry added to the file while the panel was open was written over by the old values. The panel now sends what it loaded as `base`;
// only what the person changed in the panel is written.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {nativeFixture} from './helpers/native-fixture.mjs';
import {mergeEnvChanges} from '../server/native-settings.mjs';

const handEdit=(path,env,extra={})=>{const c=JSON.parse(readFileSync(path,'utf8'));writeFileSync(path,JSON.stringify({...c,env:{...(c.env??{}),...env},...extra}));};

test('the pure merge: the changes of the form go onto the file as it is now',()=>{
  const out=mergeEnvChanges({STRATA_A:'1',STRATA_B:'2'},{STRATA_A:'1'},{STRATA_A:'1',STRATA_HC_Q8_TC:'0',PATH:'/x'});
  assert.deepEqual(out,{STRATA_A:'1',STRATA_B:'2',STRATA_HC_Q8_TC:'0'},'B is added by the form, the hand-added flag stays, PATH is not a STRATA_ name (restoreNativeEnv keeps it from the file)');
  assert.deepEqual(mergeEnvChanges({STRATA_A:'1'},{STRATA_A:'1',STRATA_B:'2'},{STRATA_A:'1',STRATA_B:'2',STRATA_C:'3'}),{STRATA_A:'1',STRATA_C:'3'},'a flag the form removed is removed');
});

test('an env flag added to the file while the panel was open survives a save from the stale panel',async()=>{
  const f=nativeFixture();try{
    const path=join(f.repo,'strata-iq2_xs.json');await f.control.refresh();
    const loaded=f.control.settings('iq2_xs').values;                       // the panel opens
    handEdit(path,{STRATA_HC_Q8_TC:'0',STRATA_HC_Q8_BF16TC:'0'},{my_own_key:'kept'});   // Open settings file: the person adds the workaround
    f.control.save('iq2_xs',{values:{...loaded,parallel:2},base:loaded}); // the stale panel saves one changed option
    const saved=JSON.parse(readFileSync(path,'utf8'));
    assert.equal(saved.parallel,2);
    assert.equal(saved.env.STRATA_HC_Q8_TC,'0');
    assert.equal(saved.env.STRATA_HC_Q8_BF16TC,'0');
    assert.equal(saved.my_own_key,'kept');
  }finally{await f.cleanup();}
});

test('without a base (an older page) the whole form is written, as before; with one, a changed env flag and a removed one still work',async()=>{
  const f=nativeFixture();try{
    const path=join(f.repo,'strata-iq2_xs.json');handEdit(path,{STRATA_ONE:'1',STRATA_TWO:'2'});await f.control.refresh();
    const loaded=f.control.settings('iq2_xs').values;
    handEdit(path,{STRATA_HAND:'h'});
    f.control.save('iq2_xs',{values:{...loaded,env:{...loaded.env,STRATA_ONE:'9',STRATA_NEW:'n'}}});     // no base: the form wins
    assert.equal(JSON.parse(readFileSync(path,'utf8')).env.STRATA_HAND,undefined);
  }finally{await f.cleanup();}
  const g=nativeFixture();try{
    const path=join(g.repo,'strata-iq2_xs.json');handEdit(path,{STRATA_ONE:'1',STRATA_TWO:'2'});await g.control.refresh();
    const loaded=g.control.settings('iq2_xs').values;
    handEdit(path,{STRATA_HAND:'h'});
    const form={...loaded.env,STRATA_ONE:'9',STRATA_NEW:'n'};delete form.STRATA_TWO;
    g.control.save('iq2_xs',{values:{...loaded,env:form},base:loaded});
    const env=JSON.parse(readFileSync(path,'utf8')).env;
    assert.equal(env.STRATA_ONE,'9');assert.equal(env.STRATA_NEW,'n');assert.equal(env.STRATA_TWO,undefined);assert.equal(env.STRATA_HAND,'h');
  }finally{await g.cleanup();}
});
