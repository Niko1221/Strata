/** Bounded, serial intake checks using the project's isolated profile and fs guard. */
import {mkdtempSync,mkdirSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {spawn} from 'node:child_process';
import {isolatedTestEnvironment} from '../app/host/scripts/test-environment.mjs';
const root=resolve(import.meta.dirname,'../apps/host');
const names=process.argv.slice(2);
if(!names.length||names.some(name=>!/^[-a-z0-9]+\.test\.mjs$/.test(name)))throw new Error('Pass explicit test basenames.');
const scratch=mkdtempSync(join(process.env.STRATA_TEST_SCRATCH_ROOT??'C:/Users/downl/AppData/Local/Temp/opencode','strata-hybrid-')),temp=join(scratch,'tmp');mkdirSync(temp);
const env=isolatedTestEnvironment(process.env,{STRATA_TEST_SCRATCH:scratch,TEMP:temp,TMP:temp,TMPDIR:temp,
 STRATA_TEST_WINDOWS_PROFILE:process.env.USERPROFILE,
 HOME:join(scratch,'home'),USERPROFILE:join(scratch,'home'),STRATA_HOME:join(scratch,'home'),
 STRATA_COMPONENTS_DIR:join(scratch,'components'),STRATA_HARNESS_DIR:join(scratch,'no-harness'),
 LOCALAPPDATA:join(scratch,'Local'),APPDATA:join(scratch,'Roaming'),XDG_DATA_HOME:join(scratch,'xdg-data'),
 XDG_STATE_HOME:join(scratch,'xdg-state'),XDG_CONFIG_HOME:join(scratch,'xdg-config')});
const guard=pathToFileURL(join(root,'scripts/test-fs-guard.mjs')).href;
for(const key of ['HOME','LOCALAPPDATA','APPDATA','XDG_DATA_HOME','XDG_STATE_HOME','XDG_CONFIG_HOME'])mkdirSync(env[key],{recursive:true});
env.NODE_OPTIONS=`--import=${guard}`;
console.log(`Scratch: ${scratch}`);
const child=spawn(process.execPath,['--test','--test-concurrency=1','--test-timeout=90000',...names.map(name=>join(root,'tests',name))],{cwd:root,env,stdio:'inherit',windowsHide:true});
child.on('error',error=>{console.error(error);process.exitCode=1;});
child.on('exit',(code,signal)=>{process.exitCode=code??(signal?1:0);});
