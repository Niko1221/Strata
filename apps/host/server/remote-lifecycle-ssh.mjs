/** Fixed SSH bootstrap for a private, content-verified copy of this build's source. */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { sshManagementConfig } from './remote-model-ssh.mjs';

const fail = (status, message, code) => Object.assign(new Error(message), { status, code });
const bootstrap = `import base64,gzip,hashlib,io,json,os,pathlib,secrets,shutil,stat,subprocess,sys
try:
 if sys.version_info<(3,10): raise RuntimeError("python")
 raw=sys.stdin.buffer.read(12582913)
 if len(raw)>12582912: raise ValueError("input")
 payload=json.loads(raw); digest=payload["sha256"]
 if not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest): raise ValueError("digest")
 folder=pathlib.Path.home()/".local"/"share"/"strata"/"remote-lifecycle-bundles"
 folder.mkdir(parents=True,exist_ok=True,mode=0o700)
 if folder.is_symlink() or folder.stat().st_uid!=os.getuid(): raise ValueError("private folder")
 folder.chmod(0o700); cached=folder/("archive-"+digest+".json.gz")
 if "bundle" in payload: archive=base64.b64decode(payload["bundle"],validate=True)
 else:
  try:
   with os.fdopen(os.open(cached,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)),"rb") as saved:
    details=os.fstat(saved.fileno())
    if details.st_uid!=os.getuid() or details.st_mode&0o077: raise ValueError("archive owner")
    archive=saved.read(8388609)
  except FileNotFoundError:
   print(json.dumps({"status":428,"body":{"code":"lifecycle_bundle_missing","error":"The private source cache needs to be restored."}}));sys.exit(0)
 if len(archive)>8388608 or hashlib.sha256(archive).hexdigest()!=digest: raise ValueError("archive")
 with gzip.GzipFile(fileobj=io.BytesIO(archive)) as stream: raw=stream.read(41943041)
 if len(raw)>41943040: raise ValueError("source size")
 manifest=json.loads(raw); files=manifest["files"]
 if manifest.get("schema")!=1 or not isinstance(files,list) or not 1<=len(files)<=4096: raise ValueError("manifest")
 allowed={"serve","strata_control","strata_cli","tools","data"}; names=set(); total=0; decoded=[]
 for entry in files:
  name=entry["path"]; parts=pathlib.PurePosixPath(name).parts
  if not isinstance(name,str) or not parts or name.startswith("/") or chr(92) in name or any(p in (".","..","__pycache__") for p in parts) or name in names: raise ValueError("path")
  if parts[0] not in allowed and name not in {"setup.py","strata.py","VERSION","requirements.txt","requirements-build.txt","requirements-monitor.txt","remote_model_manager.py","remote_model_lifecycle.py"}: raise ValueError("root")
  content=base64.b64decode(entry["data"],validate=True); total+=len(content)
  if len(content)>8388608 or total>25165824 or hashlib.sha256(content).hexdigest()!=entry["sha256"]: raise ValueError("file")
  names.add(name); decoded.append((name,content,entry["sha256"]))
 if not {"serve/server.py","strata_control/runtime.py","remote_model_manager.py","remote_model_lifecycle.py"}<=names: raise ValueError("required")
 target=folder/("source-"+digest)
 if not target.exists():
  temporary=folder/(".source-"+secrets.token_hex(16)); temporary.mkdir(mode=0o700)
  try:
   for name,content,expected in decoded:
    path=temporary/name; parent=temporary
    for part in pathlib.PurePosixPath(name).parts[:-1]: parent=parent/part; parent.mkdir(exist_ok=True,mode=0o700)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
    with os.fdopen(fd,"wb") as output: output.write(content)
   try: os.rename(temporary,target)
   except OSError:
    if not target.is_dir(): raise
  finally:
   if temporary.exists(): shutil.rmtree(temporary)
 if target.is_symlink() or target.stat().st_uid!=os.getuid(): raise ValueError("private source")
 for name,content,expected in decoded:
  path=target/name
  for parent in (path,*path.parents):
   if parent==folder: break
   if parent.is_symlink() or parent.stat().st_uid!=os.getuid() or parent.stat().st_mode&0o077: raise ValueError("source owner")
  if not stat.S_ISREG(path.lstat().st_mode): raise ValueError("source file")
  with os.fdopen(os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)),"rb") as source:
   if hashlib.sha256(source.read(8388609)).hexdigest()!=expected: raise ValueError("source changed")
 if "bundle" in payload:
  temporary=folder/(".archive-"+secrets.token_hex(16))
  try:
   fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
   with os.fdopen(fd,"wb") as output: output.write(archive)
   os.replace(temporary,cached)
  finally:
   try: temporary.unlink()
   except FileNotFoundError: pass
 request=payload["request"]; request["sourcePath"]=str(target)
 env=dict(os.environ,PYTHONPATH=str(target),PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1"); env.pop("PYTHONHOME",None)
 result=subprocess.run([sys.executable,str(target/"remote_model_lifecycle.py")],input=json.dumps(request).encode(),stdout=sys.stdout.buffer,stderr=sys.stderr.buffer,cwd=target,env=env,check=False)
 sys.exit(result.returncode)
except RuntimeError:
 print(json.dumps({"status":503,"body":{"error":"Python 3.10 or newer is required on this SSH computer."}}))
except Exception:
 print(json.dumps({"status":503,"body":{"error":"Could not validate the private lifecycle source. Rebuild this Strata app or check SSH account permissions."}}))
`;
export const SSH_LIFECYCLE_COMMAND = `python3 -c '${bootstrap.replace(/'/g, "'\\''")}'`;

function sshFailure(stderr) {
  if (/host key verification failed|remote host identification has changed|no .*host key is known/i.test(stderr)) return fail(503, 'Verify this SSH computer’s host key in Terminal before trying again.', 'ssh_host_key');
  if (/permission denied|authentication failed|too many authentication failures/i.test(stderr)) return fail(401, 'SSH authentication failed. Configure a working SSH key or agent in Terminal.', 'ssh_auth');
  if (/python3[^\n]*(?:not found|no such file|not recognized)/i.test(stderr)) return fail(503, 'Python 3.10 or newer is required on this SSH computer.', 'ssh_python');
  return fail(503, 'Could not reach this SSH computer. Check its saved SSH settings and Terminal connection.', 'ssh_connection');
}

export function createRemoteLifecycleSSH({ spawnProcess = spawn, bundlePath = new URL('../python/remote-lifecycle-source.json.gz', import.meta.url), sshPath = '/usr/bin/ssh', timeoutMs = 150_000, maxOutputBytes = 2 * 1024 * 1024, maxErrorBytes = 64 * 1024 } = {}) {
  return async function call(connection, action, body, { namespace, destination, signal, upload = false } = {}) {
    const config = sshManagementConfig(connection);
    if (!['snapshot', 'prepare-plan', 'prepare', 'serve-plan', 'serve', 'connect'].includes(action) && !/^jobs\/[a-zA-Z0-9_-]{1,160}\/cancel$/.test(action)) throw fail(400, 'Choose a supported lifecycle action.', 'ssh_request');
    if (!/^[a-f0-9]{64}$/.test(namespace) || !/^[a-f0-9]{64}$/.test(destination)) throw fail(400, 'Invalid lifecycle destination.', 'ssh_request');
    signal?.throwIfAborted();
    let archive;
    try { if (statSync(bundlePath).size > 8 * 1024 * 1024) throw Error(); archive = readFileSync(bundlePath); }
    catch { throw fail(503, 'The lifecycle source bundle is missing. Rebuild or reinstall this Strata build.', 'ssh_bundle'); }
    const request = { owner: 'desktop', namespace, destination, ...(config.sourcePath ? { catalogSourcePath: config.sourcePath } : {}), action, ...(body === undefined ? {} : { body }) };
    if (Buffer.byteLength(JSON.stringify(request)) > 16384) throw fail(413, 'This lifecycle request is too large.', 'ssh_request_limit');
    const digest = createHash('sha256').update(archive).digest('hex'), included = upload === true;
    const payload = JSON.stringify({ ...(included ? { bundle: archive.toString('base64') } : {}), sha256: digest, request });
    const args = ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12', '-o', 'ConnectionAttempts=1', '-o', 'ClearAllForwardings=yes', '-o', 'ForwardAgent=no', '-o', 'PermitLocalCommand=no',
      ...(config.port ? ['-p', String(config.port)] : []), ...(config.user ? ['-l', config.user] : []), '--', config.host, SSH_LIFECYCLE_COMMAND];
    const response = await new Promise((resolve, reject) => {
      let child, timer, forced, done = false, stdout = '', stderr = '', outputSize = 0, errorSize = 0;
      const finish = (error, value) => { if (done) return; done = true; clearTimeout(timer); signal?.removeEventListener('abort', aborted); error ? reject(error) : resolve(value); };
      const stop = error => { if (done) return; forced = setTimeout(() => { try { child?.kill('SIGKILL'); } catch {} }, 1000); forced.unref(); try { child?.kill('SIGTERM'); } catch {} finish(error); };
      const aborted = () => stop(fail(499, 'This SSH request was cancelled. Accepted lifecycle jobs keep running on that computer; refresh their status.', 'ssh_cancelled'));
      try { child = spawnProcess(sshPath, args, { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true, shell: false }); }
      catch { finish(fail(503, 'The system SSH client could not start.', 'ssh_unavailable')); return; }
      child.on('error', () => stop(fail(503, 'The system SSH client could not start.', 'ssh_unavailable')));
      child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
      child.stdout.on('data', chunk => { if (done) return; outputSize += Buffer.byteLength(chunk); if (outputSize > maxOutputBytes) return stop(fail(502, 'SSH lifecycle returned too much data.', 'ssh_output_limit')); stdout += chunk; });
      child.stderr.on('data', chunk => { if (done) return; errorSize += Buffer.byteLength(chunk); if (errorSize > maxErrorBytes) return stop(fail(502, 'SSH lifecycle returned too much diagnostic data.', 'ssh_error_limit')); stderr += chunk; });
      child.on('close', code => {
        clearTimeout(forced); if (done) return;
        if (code !== 0) { finish(sshFailure(stderr)); return; }
        let response; try { response = JSON.parse(stdout); } catch { finish(fail(502, 'SSH lifecycle returned an unreadable response.', 'ssh_protocol')); return; }
        if (!response || !Number.isInteger(response.status) || response.status < 200 || response.status > 599 || !response.body || Array.isArray(response.body) || typeof response.body !== 'object') { finish(fail(502, 'SSH lifecycle returned an unreadable response.', 'ssh_protocol')); return; }
        finish(null, response);
      });
      child.stdin.on('error', () => {});
      timer = setTimeout(() => stop(fail(504, 'SSH lifecycle timed out. Refresh to check jobs already accepted on that computer.', 'ssh_timeout')), timeoutMs);
      signal?.addEventListener('abort', aborted, { once: true }); if (signal?.aborted) aborted(); else child.stdin.end(payload);
    });
    if (!included && response.status === 428 && response.body.code === 'lifecycle_bundle_missing') {
      return call(connection, action, body, { namespace, destination, signal, upload: true });
    }
    return response;
  };
}
