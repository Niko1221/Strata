import test from 'node:test';
import assert from 'node:assert/strict';
import {isolatedTestEnvironment} from '../scripts/test-environment.mjs';

test('suite children use the scratch profile and explicit Harness without personal connection or authentication settings',()=>{
 const base={PATH:'system-tools',STRATA_API_KEY:'synthetic-owner-key',STRATA_BASE_URL:'http://owner.invalid',STRATA_LLAMA_SERVER:'owner-runtime',STRATA_MEMORY_BOBNYAN_KEY:'synthetic-memory-key',STRATA_MEMORY_PRINCIPAL_KEY:'synthetic-principal-key',STRATA_PROFILE_DIR:'owner-profiles',STRATA_RUNTIME_DIR:'owner-runtime-state',DSH_HOME:'owner-harness',strata_other_setting:'owner-setting',dsh_permission_mode:'owner-permissions',STRATA_HARNESS_DIR:'qualified-harness',STRATA_TEST_CONCURRENCY:'1'};
 const profile={HOME:'scratch-home',USERPROFILE:'scratch-home',STRATA_HOME:'scratch-home',STRATA_COMPONENTS_DIR:'scratch-components',STRATA_TEST_SCRATCH:'scratch'};
 const env=isolatedTestEnvironment(base,profile);
 assert.deepEqual(Object.keys(env).filter(name=>/^(STRATA_|DSH_)/i.test(name)).sort(),['STRATA_COMPONENTS_DIR','STRATA_HARNESS_DIR','STRATA_HOME','STRATA_TEST_SCRATCH']);
 assert.equal(env.STRATA_HARNESS_DIR,'qualified-harness');assert.equal(env.PATH,'system-tools');assert.equal(env.USERPROFILE,'scratch-home');
 assert.equal(base.STRATA_API_KEY,'synthetic-owner-key','the caller environment is never mutated');
 assert.deepEqual(isolatedTestEnvironment({strata_harness_dir:'qualified-harness'},{}),{STRATA_HARNESS_DIR:'qualified-harness'});
 assert.deepEqual(isolatedTestEnvironment({DSH_HOME:'owner'},{}),{});
});
