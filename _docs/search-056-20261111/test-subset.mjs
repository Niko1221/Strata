/** Reproduce the failing test files against an isolated baseline without touching user state. */
import { spawn } from 'node:child_process';
import { readFileSync, mkdtempSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { isolatedTestEnvironment } from '../../app/host/scripts/test-environment.mjs';
const root=resolve(process.argv[2]);
const receipt='H:/Strata/releases/search-056-20261111-codex/receipts/host-suite.txt';
const files=[...new Set([...readFileSync(receipt,'utf8').matchAll(/^test at tests[\\/]([^:]+):/gm)].map(match=>join(root,'tests',match[1])))];
if(!files.length)throw new Error('No failing test files found.');
const scratch=mkdtempSync('H:/Strata/releases/search-056-20261111-codex/temp/baseline-');
const temp=join(scratch,'tmp');mkdirSync(temp);
const env=isolatedTestEnvironment(process.env,{STRATA_TEST_SCRATCH:scratch,TEMP:temp,TMP:temp,TMPDIR:temp,HOME:join(scratch,'home'),USERPROFILE:join(scratch,'home'),STRATA_HOME:join(scratch,'home'),STRATA_COMPONENTS_DIR:join(scratch,'components'),LOCALAPPDATA:join(scratch,'Local'),APPDATA:join(scratch,'Roaming'),XDG_DATA_HOME:join(scratch,'xdg-data'),XDG_STATE_HOME:join(scratch,'xdg-state'),XDG_CONFIG_HOME:join(scratch,'xdg-config')});
for(const name of ['STRATA_CONTROL_DIR','STRATA_SETTINGS_FILE','STRATA_NO_ACQUIRE','STRATA_PACKAGED'])delete env[name];
const child=spawn(process.execPath,['--import',pathToFileURL(join(root,'scripts/test-fs-guard.mjs')).href,'--test','--test-concurrency=2',...files],{cwd:root,env,stdio:'inherit',windowsHide:true});
child.once('error',error=>{console.error(error);process.exitCode=1;});
child.once('exit',code=>{process.exitCode=code??1;});
