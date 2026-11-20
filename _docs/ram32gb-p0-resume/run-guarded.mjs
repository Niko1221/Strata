/** The existing application RAM policy around one owned diagnostic/controller. */
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {openSync, writeSync, closeSync, mkdirSync, writeFileSync, readFileSync} from 'node:fs';
import {freemem} from 'node:os';
import {fileURLToPath} from 'node:url';
import {createLocalMemoryGuard} from '../../app/host/server/local-memory-guard.mjs';

const root=fileURLToPath(new URL('../../',import.meta.url));
const base='H:/Strata/evidence/ram32gb-p0-resume-20261115';
const [kind,name]=process.argv.slice(2);
if(!['probe','baseline'].includes(kind)||!/^[a-zA-Z0-9_-]{1,80}$/.test(name??''))throw new Error('Explicit kind and fresh name required');
mkdirSync(base,{recursive:true});
const fd=openSync(`${base}/${name}-host-memory.jsonl`,'wx');
const driver=kind==='probe'?`${root}/_docs/ram32gb-p0-resume/native-probe.py`:`${root}/tools/qwen35/q36_bounded_run.py`;
const args=kind==='probe'?[driver,name]:[driver,'--spec',`${base}/${name}.spec.json`];
const started=Date.now();
const child=spawn('C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe',args,
  {cwd:root,env:{...process.env,STRATA_Q36_STARTUP_DETAIL:'1'},windowsHide:true,stdio:['ignore','inherit','inherit']});
const guard=createLocalMemoryGuard({sample:()=>{
  const available=freemem();
  writeSync(fd,JSON.stringify({atMs:Date.now(),availableBytes:available,scope:'system_available_physical_ram'})+'\n');
  return available;
},onPressure:()=>{if(!child.kill())throw new Error('Owned controller stop was not accepted');}});
guard.start();
child.once('error',()=>{guard.dispose();closeSync(fd);process.exitCode=1;});
child.once('exit',(code,signal)=>{
  guard.dispose();closeSync(fd);
  const sha=path=>createHash('sha256').update(readFileSync(path)).digest('hex');
  writeFileSync(`${base}/${name}-host-guard.json`,JSON.stringify({kind,name,startedAt:new Date(started).toISOString(),
    finishedAt:new Date().toISOString(),exitCode:code,signal,guard:guard.snapshot(),
    hostGuardSHA256:sha(`${root}/apps/host/server/local-memory-guard.mjs`),adapterSHA256:sha(fileURLToPath(import.meta.url)),
    ownership:'retained_controller_handle_and_inherited_job_descendants',qualification:'NOT_RUN'},null,2),{flag:'wx'});
  process.exitCode=code===0&&!guard.snapshot().pressure?0:1;
});
