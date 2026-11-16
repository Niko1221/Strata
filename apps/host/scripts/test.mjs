#!/usr/bin/env node
/** Enumerate test files explicitly so npm test works without shell glob expansion. */
import {readdirSync,realpathSync} from 'node:fs';
import {spawn} from 'node:child_process';
import {resolve} from 'node:path';
import {isolatedTestEnvironment} from './test-environment.mjs';
const root=resolve(import.meta.dirname,'..');
const files=readdirSync(resolve(root,'tests')).filter(name=>name.endsWith('.test.mjs')).sort().map(name=>resolve(root,'tests',name));
/* Every test file runs against scratch copies of the per-user places the host writes to (the owner's real profile is never touched). */
import {mkdtempSync,mkdirSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
// Chrome's POSIX singleton socket has a small path limit. CI's nested TMPDIR
// leaves too little space for even a fresh browser profile; keep our private
// suite root short while retaining the same filesystem isolation guard.
const scratch=realpathSync(mkdtempSync(join(process.platform==='win32'?tmpdir():'/tmp','sh-')));
const testTemp=join(scratch,'tmp');mkdirSync(testTemp);
// Default command cwd is os.homedir(); make the private profile before any
// parallel test starts rather than relying on another fixture to create it.
for(const name of ['home','Local','Roaming','components','xdg-data','xdg-state','xdg-config'])mkdirSync(join(scratch,name));
const guard=pathToFileURL(join(root,'scripts/test-fs-guard.mjs')).href;
const env=isolatedTestEnvironment(process.env,{STRATA_TEST_SCRATCH:scratch,TEMP:testTemp,TMP:testTemp,TMPDIR:testTemp,HOME:join(scratch,'home'),USERPROFILE:join(scratch,'home'),STRATA_HOME:join(scratch,'home'),STRATA_SETTINGS_FILE:join(scratch,'app-settings.json'),STRATA_COMPONENTS_DIR:join(scratch,'components'),LOCALAPPDATA:join(scratch,'Local'),APPDATA:join(scratch,'Roaming'),XDG_DATA_HOME:join(scratch,'xdg-data'),XDG_STATE_HOME:join(scratch,'xdg-state'),XDG_CONFIG_HOME:join(scratch,'xdg-config')});
// Chromium resolves the Windows account token through its registered profile.
// Browser data still use each fixture's explicit disposable user-data directory.
if(process.platform==='win32')env.STRATA_TEST_WINDOWS_PROFILE=process.env.USERPROFILE;
// Each fixture selects its own Control and settings through its scratch HOME.
// A caller's packaged Control/settings override would silently switch test routes
// and share state between otherwise independent fixture homes.
delete env.STRATA_CONTROL_DIR;
delete env.STRATA_SETTINGS_FILE;
// Acquisition policy is selected by each fixture; a packaged caller would mask mocked downloads.
delete env.STRATA_NO_ACQUIRE;
delete env.STRATA_PACKAGED;
/* Run from PowerShell 7 (GitHub's Windows runners), PSModulePath names PowerShell 7's module folders first and a Windows PowerShell 5.1 child
   that a test starts cannot load Get-Acl / Get-FileHash from them: keep only the folders that are not PowerShell 7's. */
if(process.platform==='win32'&&env.PSModulePath){const ps7=p=>{const l=p.toLowerCase().replace(/\//g,'\\');return l.includes('\\powershell\\')&&!l.includes('\\windowspowershell\\');};env.PSModulePath=env.PSModulePath.split(';').filter(p=>p&&!ps7(p)).join(';');}
const requested=process.env.STRATA_TEST_CONCURRENCY;
const concurrency=requested===undefined?null:Number(requested);
if(concurrency!==null&&(!Number.isSafeInteger(concurrency)||concurrency<1||concurrency>16))throw new Error('STRATA_TEST_CONCURRENCY must be an integer from 1 to 16');
const requestedTimeout=process.env.STRATA_TEST_TIMEOUT_MS;
const timeout=requestedTimeout===undefined?600000:Number(requestedTimeout);
if(timeout!==null&&(!Number.isSafeInteger(timeout)||timeout<1000||timeout>3600000))throw new Error('STRATA_TEST_TIMEOUT_MS must be an integer from 1000 to 3600000');
const child=spawn(process.execPath,['--import',guard,'--test',...(concurrency===null?[]:[`--test-concurrency=${concurrency}`]),...(timeout===null?[]:[`--test-timeout=${timeout}`]),...process.argv.slice(2),...files],{cwd:root,stdio:'inherit',env});
child.on('error',error=>{console.error(error.message);process.exitCode=1;});
child.on('exit',(code,signal)=>{process.exitCode=code??(signal?1:0);});
