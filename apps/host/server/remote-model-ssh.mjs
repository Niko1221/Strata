/** The system SSH client owns authentication and host trust; requests never enter a shell command. */
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync, statSync } from 'node:fs';
import { normalizeModelManagement } from './model-providers.mjs';

const fail = (status, message, code) => Object.assign(new Error(message), { status, code });
const actions = /^(?:|\/plan|\/downloads(?:\/[a-zA-Z0-9_-]{1,160}\/cancel)?|\/imports(?:\/(?:browse|inspect))?)$/;
const bootstrap = `import base64,hashlib,json,os,pathlib,secrets,subprocess,sys
try:
 if sys.version_info < (3,10):
  print(json.dumps({"status":503,"body":{"error":"Python 3.10 or newer is required on this SSH computer."}}));sys.exit(0)
 payload=json.loads(sys.stdin.buffer.read(1500000))
 source=base64.b64decode(payload["helper"],validate=True)
 digest=hashlib.sha256(source).hexdigest()
 if len(source)>524288 or digest!=payload["sha256"]: raise ValueError("helper")
 folder=pathlib.Path.home()/".local"/"share"/"strata"/"remote-model-helpers"
 folder.mkdir(parents=True,exist_ok=True,mode=0o700)
 if folder.is_symlink() or folder.stat().st_uid!=os.getuid(): raise ValueError("private folder")
 folder.chmod(0o700)
 target=folder/("manager-"+digest+".py")
 flags=os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)
 valid=False
 try:
  with os.fdopen(os.open(target,flags),"rb") as existing:
   valid=hashlib.sha256(existing.read(524289)).hexdigest()==digest
 except FileNotFoundError: pass
 if not valid:
  temporary=folder/(".manager-"+secrets.token_hex(16)+".tmp")
  try:
   fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0),0o600)
   with os.fdopen(fd,"wb") as output: output.write(source)
   os.replace(temporary,target)
  finally:
   try: temporary.unlink()
   except FileNotFoundError: pass
 target.chmod(0o600)
 request=json.dumps(payload["request"]).encode()
 result=subprocess.run([sys.executable,str(target)],input=request,stdout=sys.stdout.buffer,stderr=sys.stderr.buffer,check=False)
 sys.exit(result.returncode)
except Exception:
 print(json.dumps({"status":503,"body":{"error":"Could not prepare remote model management. Check the SSH account permissions."}}))
 sys.exit(0)
`;
// This command is identical for every connection. SSH's remote shell sees no user-supplied text.
export const SSH_MODEL_COMMAND = `python3 -c '${bootstrap.replace(/'/g, "'\\''")}'`;

export function sshManagementConfig(connection) {
  try {
    const value = normalizeModelManagement(connection?.modelManagement);
    if (!value) throw new Error();
    return value;
  } catch { throw fail(400, 'Enter valid SSH model-management settings for this connection.', 'ssh_settings'); }
}

export function sshManagementDestination(connection) {
  return createHash('sha256').update(JSON.stringify({ id: connection.id, ...sshManagementConfig(connection) })).digest('hex');
}

function sshFailure(stderr) {
  if (/host key verification failed|remote host identification has changed|no .*host key is known/i.test(stderr)) return fail(503, 'The SSH host key is unknown or has changed. Verify this computer from Terminal, then try again.', 'ssh_host_key');
  if (/permission denied|authentication failed|too many authentication failures/i.test(stderr)) return fail(401, 'SSH authentication failed. Configure a working SSH key or agent in Terminal for this connection.', 'ssh_auth');
  if (/python3[^\n]*(?:not found|no such file|not recognized)/i.test(stderr)) return fail(503, 'Python 3.10 or newer is required on this SSH computer.', 'ssh_python');
  if (/could not resolve hostname|name or service not known|nodename nor servname/i.test(stderr)) return fail(503, 'The SSH host could not be found. Check this connection’s SSH host.', 'ssh_host');
  return fail(503, 'Could not reach SSH model management. Check the SSH host, port and Terminal connection.', 'ssh_connection');
}

export function createRemoteModelSSH({ spawnProcess = spawn, helperPath = new URL('../python/remote_model_manager.py', import.meta.url), sshPath = '/usr/bin/ssh', timeoutMs = 150_000, maxOutputBytes = 2 * 1024 * 1024, maxErrorBytes = 64 * 1024 } = {}) {
  return async function call(connection, action, method, body, owner, { signal, requestId } = {}) {
    const config = sshManagementConfig(connection);
    if (!actions.test(action) || method !== (action ? 'POST' : 'GET') || typeof owner !== 'string' || !/^[a-zA-Z0-9:_-]{1,200}$/.test(owner)) throw fail(400, 'Choose a supported SSH model action.', 'ssh_request');
    signal?.throwIfAborted();
    let source;
    try {
      if (statSync(helperPath).size > 524288) throw new Error();
      source = readFileSync(helperPath);
    } catch { throw fail(503, 'The remote model helper is missing from this Strata build. Reinstall this build and try again.', 'ssh_helper'); }
    const request = { owner, ...(config.sourcePath ? { sourcePath: config.sourcePath } : {}), action, method, ...(body !== undefined ? { body } : {}), ...(requestId ? { requestId } : {}) };
    const requestJSON = JSON.stringify(request);
    if (Buffer.byteLength(requestJSON) > 16384) throw fail(413, 'This model management request is too large.', 'ssh_request_limit');
    const payload = JSON.stringify({ helper: source.toString('base64'), sha256: createHash('sha256').update(source).digest('hex'), request });
    const args = ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=12', '-o', 'ConnectionAttempts=1', '-o', 'ClearAllForwardings=yes', '-o', 'ForwardAgent=no', '-o', 'PermitLocalCommand=no',
      ...(config.port ? ['-p', String(config.port)] : []), ...(config.user ? ['-l', config.user] : []), '--', config.host, SSH_MODEL_COMMAND];
    return new Promise((resolve, reject) => {
      let child, timer, forced, done = false, stdout = '', stderr = '', outputSize = 0, errorSize = 0;
      const finish = (error, value) => {
        if (done) return;
        done = true; clearTimeout(timer); signal?.removeEventListener('abort', aborted);
        error ? reject(error) : resolve(value);
      };
      const stop = error => {
        if (done) return;
        forced = setTimeout(() => { try { child?.kill('SIGKILL'); } catch {} }, 1000); forced.unref();
        try { child?.kill('SIGTERM'); } catch {}
        finish(error);
      };
      const aborted = () => stop(fail(499, 'The SSH model-management request was cancelled. Downloads already accepted on the computer continue there.', 'ssh_cancelled'));
      try { child = spawnProcess(sshPath, args, { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true, shell: false }); }
      catch { finish(fail(503, 'The system SSH client could not start. Check this Mac’s SSH installation.', 'ssh_unavailable')); return; }
      child.on('error', () => stop(fail(503, 'The system SSH client could not start. Check this Mac’s SSH installation.', 'ssh_unavailable')));
      child.stdout.setEncoding('utf8'); child.stderr.setEncoding('utf8');
      child.stdout.on('data', chunk => {
        if (done) return;
        outputSize += Buffer.byteLength(chunk);
        if (outputSize > maxOutputBytes) return stop(fail(502, 'SSH model management returned too much data.', 'ssh_output_limit'));
        stdout += chunk.toString();
      });
      child.stderr.on('data', chunk => {
        if (done) return;
        errorSize += Buffer.byteLength(chunk);
        if (errorSize > maxErrorBytes) return stop(fail(502, 'SSH model management returned too much diagnostic data.', 'ssh_error_limit'));
        stderr += chunk.toString();
      });
      child.on('close', code => {
        clearTimeout(forced);
        if (done) return;
        if (code !== 0) { finish(sshFailure(stderr)); return; }
        let response;
        try { response = JSON.parse(stdout); } catch { finish(fail(502, 'The SSH computer returned an unreadable model response.', 'ssh_protocol')); return; }
        if (!response || !Number.isInteger(response.status) || response.status < 200 || response.status > 599 || !response.body || Array.isArray(response.body) || typeof response.body !== 'object') {
          finish(fail(502, 'The SSH computer returned an unreadable model response.', 'ssh_protocol')); return;
        }
        finish(null, response);
      });
      child.stdin.on('error', () => {});
      timer = setTimeout(() => stop(fail(504, 'SSH model management timed out. Refresh the model list to check downloads already accepted on that computer.', 'ssh_timeout')), timeoutMs);
      signal?.addEventListener('abort', aborted, { once: true });
      if (signal?.aborted) aborted();
      else child.stdin.end(payload);
    });
  };
}
