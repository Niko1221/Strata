/** Verify current packaged Control source bytes while retaining its actual build commit. */
import {execFileSync} from 'node:child_process';
import {existsSync,readFileSync} from 'node:fs';
import {join,relative,resolve} from 'node:path';
import {files,sha256,verifyTree,verifyControlData} from '../../app/host/scripts/release-integrity.mjs';
const root=resolve(import.meta.dirname,'../..'), control=join(root,'apps/desktop/src-tauri/resources/control');
const commit=readFileSync(join(control,'COMMIT'),'utf8').trim();
if(!/^[a-f0-9]{40}$/.test(commit)||readFileSync(join(control,'VERSION'),'utf8').trim()!=='0.5.9')throw Error('Wrong Control build identity');
execFileSync('git',['merge-base','--is-ancestor',commit,'HEAD'],{cwd:root,windowsHide:true});
let count=0;
for(const name of ['strata_cli','strata_control','serve']) {verifyTree(join(root,name),join(control,name));count+=files(join(control,name)).length;}
for(const name of ['strata.py','requirements.txt','requirements-monitor.txt','setup.py']) {
  if(sha256(join(root,name))!==sha256(join(control,name)))throw Error('Control source changed: '+name);
  count++;
}
for(const path of files(join(control,'tools')).filter(path=>path.endsWith('.py'))) {
  const name=relative(control,path);
  if(!existsSync(join(root,name))||sha256(join(root,name))!==sha256(path))throw Error('Control tool source changed: '+name);
  count++;
}
verifyControlData(root,control);
console.log(JSON.stringify({controlSourceCommit:commit,identicalPackagedSourceFiles:count,version:'0.5.9'}));
