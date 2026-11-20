/** Explicit CPU-only Python checks with task-local state and no auto-loaded pytest plugins. */
import {mkdtempSync,mkdirSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {spawn} from 'node:child_process';
import {isolatedTestEnvironment} from '../app/host/scripts/test-environment.mjs';

const files=process.argv.slice(2);
if(!files.length||files.some(f=>!/^serve\/test_[a-z0-9_]+\.py$/.test(f)))throw new Error('Pass explicit serve/test_*.py paths.');
const scratch=mkdtempSync('C:/Users/downl/AppData/Local/Temp/opencode/strata-gui-python-');
const dirs=Object.fromEntries(['home','local','roaming','tmp','components','cache'].map(n=>[n,join(scratch,n)]));
for(const dir of Object.values(dirs))mkdirSync(dir);
const env=isolatedTestEnvironment(process.env,{
 HOME:dirs.home,USERPROFILE:dirs.home,APPDATA:dirs.roaming,LOCALAPPDATA:dirs.local,
 TEMP:dirs.tmp,TMP:dirs.tmp,TMPDIR:dirs.tmp,XDG_CONFIG_HOME:dirs.home,XDG_DATA_HOME:dirs.home,XDG_STATE_HOME:dirs.home,
 STRATA_HOME:dirs.home,STRATA_COMPONENTS_DIR:dirs.components,STRATA_HARNESS_DIR:join(scratch,'no-harness'),
 STRATA_SETTINGS_FILE:join(dirs.home,'settings.json'),STRATA_NO_ACQUIRE:'1',STRATA_AGENT:'off',
 STRATA_TEST_NODE_EXE:process.execPath,CUDA_VISIBLE_DEVICES:'',HIP_VISIBLE_DEVICES:'',
 UV_CACHE_DIR:dirs.cache,UV_PYTHON_DOWNLOADS:'never',PYTHONDONTWRITEBYTECODE:'1',PYTHONNOUSERSITE:'1',
 PYTEST_DISABLE_PLUGIN_AUTOLOAD:'1',
});
delete env.PYTEST_ADDOPTS;
const args=['run','--no-project','--offline','--python','C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe',
 'python','-B','-m','pytest','-q','--confcutdir=serve','-p','no:cacheprovider',...files];
console.log(`Scratch: ${scratch}`);
const child=spawn('C:/Users/downl/.local/bin/uv.exe',args,{cwd:resolve(import.meta.dirname,'..'),env,stdio:'inherit',windowsHide:true});
child.on('error',error=>{console.error(error);process.exitCode=1;});
child.on('exit',(code,signal)=>{process.exitCode=code??(signal?1:0);});
