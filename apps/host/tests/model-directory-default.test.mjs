import { fileURLToPath } from 'node:url';
import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,writeFileSync,realpathSync,rmSync,symlinkSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {defaultModelDirectory} from '../server/model-directory-default.mjs';
test('macOS and Linux default to Documents/Strata/Models',()=>{
 for(const platform of ['darwin','linux'])assert.equal(defaultModelDirectory({platform,home:'/home/me',env:{},read:()=>{throw new Error('no configuration');}}),'/home/me/Documents/Strata/Models');
});
test('Windows uses the known Documents folder including OneDrive redirection',()=>{
 assert.equal(defaultModelDirectory({platform:'win32',home:'C:\\Users\\Me',env:{},windowsDocuments:()=> 'C:\\Users\\Me\\OneDrive\\Documents'}),'C:\\Users\\Me\\OneDrive\\Documents\\Strata\\Models');
 assert.equal(defaultModelDirectory({platform:'win32',home:'C:\\Users\\Me',env:{},windowsDocuments:()=>{throw new Error('unavailable');}}),'C:\\Users\\Me\\Documents\\Strata\\Models');
});
test('Linux respects localized XDG Documents and rejects shell expressions',()=>{
 assert.equal(defaultModelDirectory({platform:'linux',home:'/home/me',env:{XDG_CONFIG_HOME:'/custom'},read:path=>{assert.equal(path,'/custom/user-dirs.dirs');return 'XDG_DOCUMENTS_DIR="$HOME/Dokumente"';}}),'/home/me/Dokumente/Strata/Models');
 assert.equal(defaultModelDirectory({platform:'linux',home:'/home/me',env:{XDG_DOCUMENTS_DIR:'${HOME}/Work'},read:()=>''}),'/home/me/Work/Strata/Models');
 assert.equal(defaultModelDirectory({platform:'linux',home:'/home/me',env:{XDG_DOCUMENTS_DIR:'$(echo /private)'},read:()=>''}),'/home/me/Documents/Strata/Models');
});
test('new hosts create the default; custom folders and legacy weights remain in place',t=>{
 const root=realpathSync(mkdtempSync(join(tmpdir(),'strata-documents-')));t.after(()=>rmSync(root,{recursive:true,force:true}));
 function boot(name,state,weights=false){const home=join(root,name),documents=join(root,name+'-documents');mkdirSync(join(home,'data'),{recursive:true});mkdirSync(join(home,'models'),{recursive:true});if(state)writeFileSync(join(home,'data','state.json'),JSON.stringify(state));if(weights){mkdirSync(join(home,'models','native','qwen'),{recursive:true});writeFileSync(join(home,'models','native','qwen','model.gguf'),'GGUFfixture');}const env={...process.env,STRATA_HOME:home,STRATA_DOCUMENTS_DIR:documents,STRATA_TEST_NATIVE:'1'};delete env.STRATA_MODELS_DIR;const result=spawnSync(process.execPath,['--input-type=module','-e',"const store=await import('./server/store.mjs');if(process.env.STRATA_TEST_NATIVE==='1'&&store.MODELS_DIR.endsWith('/Strata/Models')){const {statSync}=await import('node:fs');const {join}=await import('node:path');const native=store.nativeDataDirectory(store.MODELS_DIR);const a=statSync(join(native,'models')),b=statSync(store.MODELS_DIR);if(a.ino!==b.ino||a.dev!==b.dev)throw new Error('Native weights miss requested model folder');}console.log(store.MODELS_DIR);"],{cwd:fileURLToPath(new URL('..',import.meta.url)),env,encoding:'utf8'});assert.equal(result.status,0,result.stderr);return {path:result.stdout.trim(),home,documents};}
 const fresh=boot('fresh');assert.equal(fresh.path,join(fresh.documents,'Strata','Models'));
 const target=join(root,'redirected-target');mkdirSync(target);symlinkSync(target,join(root,'redirected-documents'));assert.equal(boot('redirected').path,join(target,'Strata','Models'));
 const custom=join(root,'custom');mkdirSync(custom);assert.equal(boot('custom-host',{modelsDir:custom}).path,custom);
 const legacy=boot('legacy',{},true);assert.equal(legacy.path,join(legacy.home,'models'));
});
