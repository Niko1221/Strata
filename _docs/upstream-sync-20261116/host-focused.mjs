/** Run named Host tests with the same private filesystem environment as scripts/test.mjs. */
import {mkdtempSync,mkdirSync,realpathSync,existsSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {tmpdir} from 'node:os';
import {resolve,join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {isolatedTestEnvironment} from '../../apps/host/scripts/test-environment.mjs';
const root=resolve(import.meta.dirname,'../../apps/host');
const names=process.argv.slice(2);
if(!names.length||names.some(name=>!/^[a-z0-9-]+$/.test(name)))throw new Error('Explicit test basenames required');
const files=names.map(name=>join(root,'tests',`${name}.test.mjs`));
if(files.some(path=>!existsSync(path)))throw new Error('Requested test file missing');
const scratch=realpathSync(mkdtempSync(join(tmpdir(),'host-focus-'))),testTemp=join(scratch,'tmp');
for(const name of ['tmp','home','Local','Roaming','components','xdg-data','xdg-state','xdg-config'])mkdirSync(join(scratch,name));
const env=isolatedTestEnvironment(process.env,{STRATA_TEST_SCRATCH:scratch,TEMP:testTemp,TMP:testTemp,TMPDIR:testTemp,HOME:join(scratch,'home'),USERPROFILE:join(scratch,'home'),STRATA_HOME:join(scratch,'home'),STRATA_SETTINGS_FILE:join(scratch,'app-settings.json'),STRATA_COMPONENTS_DIR:join(scratch,'components'),LOCALAPPDATA:join(scratch,'Local'),APPDATA:join(scratch,'Roaming'),XDG_DATA_HOME:join(scratch,'xdg-data'),XDG_STATE_HOME:join(scratch,'xdg-state'),XDG_CONFIG_HOME:join(scratch,'xdg-config')});
if(process.platform==='win32')env.STRATA_TEST_WINDOWS_PROFILE=process.env.USERPROFILE;
for(const name of ['STRATA_CONTROL_DIR','STRATA_SETTINGS_FILE','STRATA_NO_ACQUIRE','STRATA_PACKAGED'])delete env[name];
if(process.platform==='win32'&&env.PSModulePath){const ps7=p=>{const l=p.toLowerCase().replace(/\//g,'\\');return l.includes('\\powershell\\')&&!l.includes('\\windowspowershell\\');};env.PSModulePath=env.PSModulePath.split(';').filter(p=>p&&!ps7(p)).join(';');}
console.log(JSON.stringify({node:process.version,tests:names,filesystemGuard:true}));
const guard=pathToFileURL(join(root,'scripts/test-fs-guard.mjs')).href;
const child=spawn(process.execPath,['--import',guard,'--test','--test-concurrency=1','--test-timeout=120000',...files],{cwd:root,stdio:'inherit',env});
child.on('error',error=>{console.error(error.message);process.exitCode=1;});
child.on('exit',(code,signal)=>{process.exitCode=code??(signal?1:0);});
