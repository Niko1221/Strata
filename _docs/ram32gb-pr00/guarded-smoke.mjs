/** Run the private benchmark controller with the application's existing RAM stop policy. */
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {createLocalMemoryGuard, RAM_FLOOR_BYTES} from '../../app/host/server/local-memory-guard.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const output = 'H:/Strata/evidence/ram32gb-pr00-20261115';
const attempt = process.argv[2] ?? 'r4';
if (!/^r[4-9]$/.test(attempt)) throw new Error('Use a fresh bounded attempt r4..r9');
const spec = JSON.parse(readFileSync(`${output}/smoke-inputs/spec-r3.json`, 'utf8'));
spec.run_id = `q36-short-observation-${attempt}`;
spec.output_dir = `${output}/${spec.run_id}`;
spec.min_available_ram_bytes = RAM_FLOOR_BYTES;
const specPath = `${output}/smoke-inputs/spec-${attempt}.json`;
writeFileSync(specPath, JSON.stringify(spec, null, 2), {flag:'wx'});
const started = Date.now();
const child = spawn('C:/Users/downl/AppData/Local/Programs/Python/Python312/python.exe',
  [`${root}/tools/qwen35/q36_bounded_run.py`, '--spec', specPath],
  {cwd:root,env:process.env,windowsHide:true,stdio:['ignore','inherit','inherit']});
const guard = createLocalMemoryGuard({onPressure:() => {
  // Only this retained controller handle is stopped. Its nested kill-on-close
  // Job owns all native descendants; no names, ports or unrelated PIDs are used.
  if (!child.kill()) throw new Error('owned_controller_stop_not_accepted');
}});
guard.start();
child.once('error', () => { guard.dispose(); process.exitCode=1; });
child.once('exit', (code, signal) => {
  guard.dispose();
  const sha = path => createHash('sha256').update(readFileSync(path)).digest('hex');
  const receipt = attempt === 'r4' ? 'dynamic-host-guard' : `dynamic-host-guard-${attempt}`;
  writeFileSync(`${output}/${receipt}.json`, JSON.stringify({
    startedAt: new Date(started).toISOString(), endedAt:new Date().toISOString(),
    exitCode:code, signal, guard:guard.snapshot(),
    controllerScope:'owned_benchmark_controller_and_inherited_job_descendants',
    hostGuardSHA256:sha(`${root}/apps/host/server/local-memory-guard.mjs`),
    adapterSHA256:sha(fileURLToPath(import.meta.url)), qualification:'NOT_RUN',
  },null,2),{flag:'wx'});
  process.exitCode = code === 0 && !guard.snapshot().pressure ? 0 : 1;
});
