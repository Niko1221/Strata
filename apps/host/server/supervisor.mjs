/**
 * Engine supervisor.
 *
 * Owns the inference process: resolves a binary, translates the tuning
 * configuration into real flags, spawns it, waits for health, keeps a log ring
 * buffer, and polls the engine's own Prometheus metrics so the Monitor shows
 * measured numbers rather than guesses.
 *
 * Supported backends, in preference order:
 *   1. llama.cpp `llama-server` (bundled under $STRATA_HOME/engine, or on PATH)
 *   2. any already-running OpenAI-compatible server on localhost
 */
import { CONTEXT_TOKENS } from './catalog.mjs';
import { randomBytes } from 'node:crypto';
import { privateLlamaArgs } from './llama-memory-policy.mjs';
import { spawn } from 'node:child_process';
import { existsSync, readFileSync, realpathSync, readdirSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { MODELS_DIR } from './store.mjs';
import { componentsDirectory } from './control-runtime.mjs';
import { which } from './util.mjs';
import { buildSpin8Args, probeBackendCapabilities } from './spin8-backend.mjs';
import { resolveKvSettings, explicitKvEnvironment } from './kv-settings.mjs';
import {localBackendChoice,findLocalBackend,verifyLocalBackend,requireRuntimeKv} from './local-backends.mjs';
import {createRunObservation,requestedFromArgs,observationEnabled} from './lifecycle-observation.mjs';
import {createStartupDiagnostics} from './startup-diagnostics.mjs';
import {readGGUF,qwen35MissingMtp} from './gguf.mjs';
import { freemem } from 'node:os';
import { admitLocalMemory, createLocalMemoryGuard } from './local-memory-guard.mjs';
export {createStartupDiagnostics} from './startup-diagnostics.mjs';

const LOG_LIMIT = 400;
const emptyMetrics = () => ({source:'none',promptTokens:0,predictedTokens:0,promptTps:0,outputTps:0,
  kvUsage:0,kvTokens:0,requestsProcessing:0,requestsDeferred:0,sampledAt:null});

const state = {
  status: 'stopped', // stopped | starting | running | stopping | error
  binary: null,
  binarySource: null,
  requestedBackend: 'Automatic',
  runtimeVersion: null,
  pid: null,
  port: 8080,
  modelId: null,
  modelPath: null,
  args: [],
  command: '',
  startedAt: null,
  error: null,
  logs: [],
  metrics: emptyMetrics(),
};

let child = null;
let contextOwner = null, contextCapacity = null;
let launchSequence = 0;
let ramGuard = null;
let observation = null;
export function runObservation(){return observation?.snapshot()??null;}
let lastStart = null, privateKey = '', memoryMode = false;
const stopListeners=new Set();
export function onBeforeStop(listener){stopListeners.add(listener);return()=>stopListeners.delete(listener);}
export function authenticationKey(){return privateKey;}
export function memoryTarget(){
  if(!isRunning()||!memoryMode||!privateKey)throw new Error('Open Hakua memory chat for a ready local model first.');
  return {pid:state.pid,port:state.port,model:state.modelId,args:[...state.args],server_key:privateKey};
}
export async function enableMemory(){
  if(!isRunning()||!lastStart)throw new Error('Load a PrismML or TurboQuant model first.');
  if(!['PrismML','TurboQuant'].includes(state.requestedBackend))throw new Error('Hakua memory requires a verified PrismML or TurboQuant runtime.');
  if(!memoryMode){
    if(state.metrics.requestsProcessing||state.metrics.requestsDeferred)throw new Error('Finish the current model requests before opening memory chat.');
    await start({...lastStart,privateMemory:true});
  }
  return memoryTarget();
}
let pollTimer = null;
let hubRef = null;
let mcpAt=0,mcpStatus=null,mcpPending=false;

export function init(hub) {
  hubRef = hub;
}

/* ------------------------------------------------------------------ binary --- */

export function findBinary({ platform = process.platform, env = process.env, modelsDir = MODELS_DIR, componentsDir = componentsDirectory({ env, platform }), backend='Automatic', installRoot } = {}) {
  if(backend!=='Automatic')return findLocalBackend(backend,{componentsDir,platform,installRoot});
  const binaryNames = platform === 'win32' ? ['llama-server.exe', 'llama-server'] : ['llama-server'];
  const candidates = [];

  if (env.STRATA_LLAMA_SERVER) candidates.push({ path: env.STRATA_LLAMA_SERVER, source: 'STRATA_LLAMA_SERVER' });

  // The app's generic GGUF component is self-contained: its executable and DLLs stay together.
  // A staged update only becomes active after its pointer is written, as with the Control component.
  const genericRoot = join(componentsDir, 'generic-gguf');
  try {
    const pointer = JSON.parse(readFileSync(join(genericRoot, 'active.json'), 'utf8').replace(/^\uFEFF/, ''));
    if (typeof pointer?.folder === 'string' && /^[\w.+-]+$/.test(pointer.folder) && !['.', '..'].includes(pointer.folder)) {
      const activeDir = resolve(genericRoot, pointer.folder);
      if (dirname(activeDir) === resolve(genericRoot) && dirname(realpathSync(activeDir)) === realpathSync(genericRoot)) {
        for (const name of binaryNames) candidates.push({ path: join(activeDir, name), source: 'generic-gguf' });
      }
    }
  } catch { /* no activated generic backend */ }
  for (const name of binaryNames) candidates.push({ path: join(genericRoot, name), source: 'generic-gguf' });

  // A build downloaded into the app's own folder wins over anything on PATH:
  // it is the version Strata was tested against.
  try {
    const engineDir = join(modelsDir, '..', 'engine');
    if (existsSync(engineDir)) {
      if (platform === 'win32') {
        for (const name of binaryNames) candidates.push({ path: join(engineDir, name), source: 'bundled' });
      }
      for (const entry of readdirSync(engineDir)) {
        const dir = join(engineDir, entry);
        if (!statSync(dir).isDirectory()) continue;
        for (const name of binaryNames) {
          const full = join(dir, name);
          if (existsSync(full)) candidates.push({ path: full, source: 'bundled' });
          if (platform === 'win32') {
            candidates.push({ path: join(dir, 'bin', 'Release', name), source: 'bundled' });
            candidates.push({ path: join(dir, 'bin', name), source: 'bundled' });
          }
        }
      }
    }
  } catch {
    /* ignore unreadable engine dir */
  }

  for (const name of binaryNames) {
    candidates.push({ path: `/opt/homebrew/bin/${name}`, source: 'homebrew' });
    candidates.push({ path: `/usr/local/bin/${name}`, source: 'homebrew' });
  }

  return candidates.find((c) => existsSync(c.path)) ?? null;
}

export async function resolveBinary({ updateState = true, backend='Automatic' } = {}) {
  const found = findBinary({backend});
  if (found) {
    if (updateState) {
      state.binary = found.path;
      state.binarySource = found.source;
    }
    return found;
  }
  if(backend!=='Automatic')return null;
  const onPath = await which(process.platform === 'win32' ? 'llama-server.exe' : 'llama-server');
  if (onPath) {
    if (updateState) {
      state.binary = onPath;
      state.binarySource = 'PATH';
    }
    return { path: onPath, source: 'PATH' };
  }
  return null;
}

export async function probeCapabilities({ manifestPath = process.env.STRATA_BONSAI_MANIFEST ?? null } = {}) {
  const binary = await resolveBinary({ updateState: false });
  return probeBackendCapabilities(binary?.path, { manifestPath });
}

/* --------------------------------------------------------------- arguments --- */


/**
 * Turn the tuning configuration into a real llama-server command line.
 * Exported so the UI can show exactly what will run.
 */
export function buildArgs({ modelPath, config, hardware, port = state.port }) {
  // Check before lifecycle changes: an unsupported request must not stop a server.
  const spin8Args = buildSpin8Args(config);
  const beginner = config.beginner ?? {};
  const params = new Map((config.expert ?? []).map((p) => [p.id, p.value]));
  const num = (id, fallback) => {
    const v = Number(params.get(id));
    return Number.isFinite(v) ? v : fallback;
  };
  const ctxLabel = beginner.contextLength === 'recommended' ? '128K' : String(beginner.contextLength).toUpperCase();
  const ctx = CONTEXT_TOKENS[ctxLabel.toLowerCase()] ?? 32768;

  const args = [
    '-m', modelPath,
    '--host', '127.0.0.1',
    '--port', String(port),
    '-c', String(ctx),
    '--metrics',
    '--slots',
    '--jinja',
    '--no-webui',
  ];

  // GPU offload — on unified memory "all layers" is the right default.
  const split = String(params.get('gpu.layers') ?? 'Automatic');
  const ngl = split === 'CPU only' ? 0 : split === 'Quarter' ? Math.max(1, Math.round(Number(config.layerCount ?? 48) / 4)) : split === 'Half' ? Math.max(1, Math.round(Number(config.layerCount ?? 48) / 2)) : 999;
  args.push('-ngl', String(ngl));
  const cpuMoeLayers = params.get('gpu.cpuMoeLayers') ?? 0;
  if (!Number.isInteger(cpuMoeLayers) || cpuMoeLayers < 0 || cpuMoeLayers > 256) throw new Error('MoE layers in CPU memory must be a whole number from 0 to 256.');
  if (cpuMoeLayers > 0) args.push('--n-cpu-moe', String(cpuMoeLayers));

  const threads = num('perf.threads', Math.max(4, (hardware?.cpu?.cores ?? 8) - 2));
  args.push('-t', String(threads));

  const kv = resolveKvSettings(config.expert);
  args.push('-ctk', kv.k, '-ctv', kv.v);

  const batch = num('perf.batch', 2048);
  args.push('-b', String(batch), '-ub', String(Math.max(128, Math.min(512, Math.round(batch / 4)))));

  if (params.get('perf.flashAttn') === false) args.push('-fa', 'off');
  else args.push('-fa', 'on');

  if (params.get('mem.mmap') === false) args.push('--no-mmap');
  if (params.get('mem.lock') === true) args.push('--mlock');

  const parallel = Math.max(1, num('server.parallel', 1));
  args.push('-np', String(parallel));

  // Sampling defaults applied to every request by the server.
  args.push(
    '--temp', String(params.get('sample.temp') ?? 0.6),
    '--top-p', String(params.get('sample.topP') ?? 0.95),
    '--top-k', String(params.get('sample.topK') ?? 40),
    '--repeat-penalty', String(params.get('sample.repeat') ?? 1.05),
  );
  const seed = Number(params.get('sample.seed') ?? -1);
  if (seed >= 0) args.push('--seed', String(seed));

  if (config.serving?.apiKey) args.push('--api-key', config.serving.apiKey);

  return [...args, ...spin8Args];
}

function publicArgs() {
  return state.args.map((value,index)=>state.args[index-1]==='--api-key'?'[redacted]':value);
}
export function describeCommand() {
  return [state.binary, ...publicArgs()].filter(Boolean).join(' ');
}

/* ---------------------------------------------------------------- lifecycle --- */

function log(stream, line) {
  const entry = { at: new Date().toISOString(), stream, line };
  state.logs.push(entry);
  if (state.logs.length > LOG_LIMIT) state.logs.shift();
  hubRef?.send('engine-log', entry);
}

export function logs(limit = 200) {
  return state.logs.slice(-limit);
}

function earlyExitError(proc, spawnError, diagnostics) {
  if (spawnError) return new Error(`Engine could not start: ${spawnError.message}`);
  if (proc.exitCode === null && proc.signalCode === null) return null;
  const reason = proc.signalCode ? `signal ${proc.signalCode}` : `code ${proc.exitCode}`;
  diagnostics.flush();
  const diagnostic = diagnostics.summary();
  return new Error(`Engine exited early with ${reason}${diagnostic ? `: ${diagnostic}` : '. Check engine logs for the cause.'}`);
}

async function waitForHealth(port, timeoutMs = 90_000, proc, getSpawnError = () => null, diagnostics) {
  const deadline = Date.now() + timeoutMs;
  let attempts = 0, lastProbe = 'no response', reported = '';
  while (Date.now() < deadline) {
    const stopped = earlyExitError(proc, getSpawnError(), diagnostics);
    if (stopped) throw stopped;
    let healthy = false;
    try {
      attempts++;
      const res = await fetch(`http://127.0.0.1:${port}/health`, { signal: AbortSignal.timeout(1500) });
      const body = await res.json().catch(() => ({}));
      healthy = res.ok && body.status === 'ok';
      // Whitelist phase labels, not arbitrary backend response/error text.
      const loading = body.status === 'loading' || body.error?.message === 'Loading model';
      lastProbe = `HTTP ${res.status} ${healthy ? 'ready' : loading ? 'loading model' : 'not ready'}`;
    } catch (error) {
      lastProbe = error?.name === 'TimeoutError' ? 'health probe timed out' : 'connection unavailable';
    }
    const stoppedAfterProbe = earlyExitError(proc, getSpawnError(), diagnostics);
    if (stoppedAfterProbe) throw stoppedAfterProbe;
    if (healthy) return true;
    if (lastProbe !== reported) { log('supervisor', `engine health: ${lastProbe}`); reported = lastProbe; }
    await new Promise((r) => setTimeout(r, 700));
  }
  throw new Error(`Engine did not become healthy in time (budget ${timeoutMs} ms; ${attempts} probes; last: ${lastProbe})`);
}

/** Read the server's effective per-slot window once, never GGUF training metadata.
 * llama.cpp /props default_generation_settings.n_ctx is already per slot (and
 * can be capped below the requested -c). Do not divide it by total_slots again.
 */
async function readContextCapacity(port, modelPath, key) {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/props`, {
      signal: AbortSignal.timeout(1500), redirect: 'error',
      headers: key ? { authorization: `Bearer ${key}` } : undefined,
    });
    if (!response.ok) return null;
    const props = await response.json();
    const capacity = props?.default_generation_settings?.n_ctx;
    if (!Number.isSafeInteger(capacity) || capacity <= 0) return null;
    // Bind the response to the launched model as well as the owned child.
    if (typeof props.model_path !== 'string' || resolve(props.model_path) !== resolve(modelPath)) return null;
    return capacity;
  } catch { return null; } // Capability failure leaves plain chat available.
}

/** Validate a selected runtime before an already loaded model is stopped. */
export async function preflight({config,modelPath}) {
  const backend=localBackendChoice(config?.expert);
  const binary=await resolveBinary({backend,updateState:false});
  if(!binary)throw new Error(`No ${backend==='Automatic'?'llama-server':backend} runtime found. Install the matching local runtime first.`);
  if(!existsSync(modelPath))throw new Error(`Model file missing: ${modelPath}`);
  const kv=resolveKvSettings(config?.expert);
  const runtime=backend==='Automatic'?null:await verifyLocalBackend(binary,backend);
  if(runtime)requireRuntimeKv(runtime,kv);
  else if([kv.k,kv.v].some(type=>type.startsWith('turbo')))throw new Error('Choose the verified TurboQuant llama-server backend before using a TurboQuant K/V cache.');
  return {binary,backend,runtime};
}

export async function start({ modelId, modelPath, port = 8080, config, hardware, timeoutMs = 120_000, privateMemory = false, beforeStart, runObservation: captureRun = observationEnabled(), observationProbe }) {
  const ordinaryArgs = buildArgs({ modelPath, config, hardware, port });
  const args = privateMemory ? privateLlamaArgs(ordinaryArgs) : ordinaryArgs;
  const {binary,backend,runtime}=await preflight({config,modelPath});
  if (state.status === 'running' && state.modelId === modelId && state.port === port && state.binary===binary.path && state.requestedBackend===backend && JSON.stringify(state.args)===JSON.stringify(args)) {
    return status();
  }
  if(privateMemory && !['PrismML','TurboQuant'].includes(backend))throw new Error('A verified local memory runtime is required.');
  // This guard covers this Windows process's physical RAM, not a model-window
  // promise. Reject before stopping another runtime or changing user settings.
  if (process.platform === 'win32') {
    const modelBytes = statSync(modelPath).size;
    const fullyOffloaded = args[args.indexOf('-ngl') + 1] === '999' && hardware?.gpus?.some(gpu => !gpu.unified && gpu.vramGb * 1024 ** 3 > modelBytes + 1024 ** 3);
    admitLocalMemory(freemem(), { fullyOffloaded: !!fullyOffloaded, modelBytes, mapped: !args.includes('--no-mmap') });
  }
  await beforeStart?.();
  if (child) await stop();
  const capture=captureRun?createRunObservation({backend:backend==='Automatic'?'unknown':backend,requested:requestedFromArgs(args),gpuProbe:observationProbe}):null;
  observation=capture;
  if(capture){void capture.identify({executable:binary.path,version:runtime?.version,modelPath});void capture.sampleGpu();}
  lastStart={modelId,modelPath,port,config:structuredClone(config),hardware:structuredClone(hardware),timeoutMs};
  memoryMode=privateMemory;
  privateKey=privateMemory?randomBytes(32).toString('hex'):'';
  state.binary=binary.path;
  state.binarySource=binary.source;
  state.requestedBackend=backend;
  state.runtimeVersion=runtime?`${backend} ${runtime.version} · ${runtime.source.commit.slice(0,8)}`:null;

  const sequence = ++launchSequence;
  state.status = 'starting';
  contextOwner = null; contextCapacity = null;
  state.port = port;
  state.modelId = modelId;
  state.modelPath = modelPath;
  state.error = null;
  state.logs = [];
  state.metrics = emptyMetrics();
  mcpAt=0;mcpStatus=null;
  state.args = args;
  state.command = describeCommand();
  hubRef?.send('engine-status', status());

  log('supervisor', `$ ${state.command}`);

  child = spawn(binary.path, state.args, {
    cwd: join(MODELS_DIR, '..'),
    windowsHide: true,
    env: { ...explicitKvEnvironment(), LLAMA_CACHE: join(MODELS_DIR, '..', 'cache'), ...(privateKey?{LLAMA_API_KEY:privateKey}:{}) },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const launched = child;
  let spawnError = null;
  let memoryFailure = null;
  const diagnostics=createStartupDiagnostics(privateKey || config.serving?.apiKey || '');
  launched.on('error', error => { spawnError = error; });

  state.pid = child.pid;
  state.startedAt = new Date().toISOString();

  const onData = (stream) => (chunk) => {
    diagnostics.consume(stream,chunk);
    capture?.consume(stream,chunk);
    for (const line of chunk.toString().split('\n')) {
      if (line.trim()) log(stream, line.replace(/\s+$/, ''));
    }
  };
  child.stdout.on('data', onData('out'));
  child.stderr.on('data', onData('err'));

  child.on('exit', (code, signal) => {
    capture?.finish(state.status==='stopping'?'stopped':state.status==='starting'?'startup_exit':code===0?'stopped':'error',code);
    if(capture)void capture.sampleGpu();
    if(child!==launched)return; // A late exit from the previous process cannot stop its replacement.
    contextOwner = null; contextCapacity = null;
    ramGuard?.dispose();
    log('supervisor', `engine exited (${signal ?? code})`);
    child = null;
    state.pid = null;
    if (state.status !== 'stopping') {
      state.status = code === 0 ? 'stopped' : 'error';
      if (code !== 0 && code !== null) state.error = `Engine exited with code ${code}`;
    } else {
      state.status = 'stopped';
    }
    stopPolling();
    hubRef?.send('engine-status', status());
  });

  if (process.platform === 'win32') {
    ramGuard = createLocalMemoryGuard({ onPressure: async pressure => {
      if (child !== launched) return;
      memoryFailure = new Error('RAM protection stopped the local model before exhausting system memory. Close other applications, reduce the batch size or disable memory mapping, then reload.');
      log('supervisor', `RAM protection: available=${pressure.availableBytes ?? 'unavailable'} threshold=${Math.round(pressure.thresholdBytes)} floor=${pressure.floorBytes}`);
      try { await stop(); }
      finally {
        // A late result cannot overwrite the status of a replacement engine.
        if (!child || child === launched) { state.status = 'error'; state.error = memoryFailure.message; hubRef?.send('engine-status', status()); }
      }
    } });
    ramGuard.start();
  }

  try {
    await waitForHealth(port, timeoutMs, launched, () => memoryFailure ?? spawnError, diagnostics);
    if (memoryFailure) throw memoryFailure;
    if (child !== launched) throw new Error('The engine was stopped while loading.');
  } catch (err) {
    // Exit can clear child before this continuation runs. A generation also
    // distinguishes that legitimate startup failure from a stop/replacement.
    if (launchSequence !== sequence) throw err;
    // Only offer the predictor diagnosis after proving every trunk tensor. A
    // filename or one missing layer alone cannot distinguish a broken download.
    if(/tensor 'blk\.32\.attn_norm\.weight' not found/.test(err.message)){
      try{
        const compatibility=qwen35MissingMtp(readGGUF(modelPath,{includeTensors:true}));
        if(compatibility)err.message+=`. ${compatibility.message}`;
      }catch{ /* Preserve the original loader error if structural proof is unavailable. */ }
    }
    capture?.finish(launched.exitCode!==null?'startup_exit':'error',launched.exitCode);
    const cleanupSequence = launchSequence + 1;
    await stop();
    if (launchSequence !== cleanupSequence) throw err;
    state.status = 'error';
    state.error = err.message;
    hubRef?.send('engine-status', status());
    throw err;
  }

  const capacity = await readContextCapacity(port, modelPath, privateKey || config.serving?.apiKey || '');
  // A stop/replacement during the probe must not resurrect this process or
  // publish its capacity for a new child, even if the model id is unchanged.
  if (child !== launched || state.status !== 'starting' || launched.exitCode !== null || launched.signalCode !== null) {
    throw new Error('The engine was stopped while checking its context size.');
  }
  contextOwner = launched;
  contextCapacity = capacity;
  if (capacity === null) log('supervisor', 'Agent context size unavailable; plain chat remains available.');
  state.status = 'running';
  capture?.ready();
  if(capture)void capture.sampleGpu({ownedPids:[launched.pid]});
  state.startedAt = new Date().toISOString();
  startPolling();
  hubRef?.send('engine-status', status());
  log('supervisor', 'engine ready');
  return status();
}

export async function stop() {
  launchSequence++;
  contextOwner = null; contextCapacity = null;
  ramGuard?.dispose();
  if(state.status==='starting'&&observation?.snapshot().outcome==='starting')observation.finish('cancelled');
  for(const listener of stopListeners)listener();
  state.metrics=emptyMetrics();
  if (!child) {
    state.status = 'stopped';
    state.pid = null;
    stopPolling();
    hubRef?.send('engine-status', status());
    return status();
  }
  state.status = 'stopping';
  hubRef?.send('engine-status', status());
  const proc = child;
  let exited = waitForExit(proc, 4000);
  proc.kill('SIGTERM');
  if (!await exited) {
    log('supervisor', 'engine ignored SIGTERM — killing');
    // Requesting a forced kill does not establish that the child has exited.
    // Keep its ownership/PID until the exit event, including the failure path.
    exited = waitForExit(proc, 4000);
    proc.kill('SIGKILL');
    if (!await exited) {
      state.status = 'error';
      state.error = 'The engine did not exit after stopping. Retry unload before loading another model.';
      hubRef?.send('engine-status', status());
      throw new Error(state.error);
    }
  }
  if (child && child !== proc) return status();
  child = null;
  state.status = 'stopped';
  state.pid = null;
  state.startedAt = null;
  stopPolling();
  hubRef?.send('engine-status', status());
  return status();
}

function waitForExit(proc, timeoutMs) {
  if (proc.exitCode !== null || proc.signalCode !== null) return Promise.resolve(true);
  return new Promise(resolve => {
    const finished = () => { clearTimeout(timer); proc.removeListener('exit', finished); resolve(true); };
    const timer = setTimeout(() => { proc.removeListener('exit', finished); resolve(false); }, timeoutMs);
    proc.once('exit', finished);
  });
}

export async function restart(opts) {
  // start validates the complete runtime before stopping either engine. Hash
  // every launch, once: repeated verification here made one load do triple I/O.
  return start(opts);
}

/* ------------------------------------------------------------------ metrics --- */

function startPolling() {
  stopPolling();
  pollTimer = setInterval(() => void pollMetrics(), 1000);
  pollTimer.unref?.();
  void pollMetrics();
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

function parsePrometheus(text) {
  const out = {};
  for (const line of text.split('\n')) {
    if (!line || line.startsWith('#')) continue;
    const match = /^([a-zA-Z0-9_:]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$/.exec(line.trim());
    if (!match) continue;
    out[match[1]] = Number(match[3]);
  }
  return out;
}

/** Original Strata serves JSON; llama.cpp serves Prometheus. Preserve the original dashboard facts. */
export function parseEngineMetrics(text) {
  if(text.trim().startsWith('{')) {
    const dashboard=JSON.parse(text);
    if(!dashboard.engine||!dashboard.live)throw new Error('Unrecognized engine metrics');
    const live=dashboard.live;
    return {source:'engine',dashboard,promptTokens:live.prompt_tokens??0,predictedTokens:live.generated??0,promptTps:live.prefill_tok_s_mean??0,outputTps:live.tok_s??0,kvUsage:0,kvTokens:0,requestsProcessing:live.running??(['reading','generating'].includes(live.state)?1:0),requestsDeferred:live.queued??0};
  }
  const m=parsePrometheus(text);
  if(!Object.keys(m).some(key=>/prompt_tokens|tokens_predicted|predicted_tokens|kv_cache|requests_processing/.test(key)))throw new Error('Unrecognized engine metrics');
  const p=name=>m[`llamacpp:${name}`]??m[name]??0;
  return {source:'engine',promptTokens:p('prompt_tokens_total'),predictedTokens:p('tokens_predicted_total'),promptTps:p('prompt_tokens_seconds'),outputTps:p('predicted_tokens_seconds'),kvUsage:p('kv_cache_usage_ratio'),kvTokens:p('kv_cache_tokens'),requestsProcessing:p('requests_processing'),requestsDeferred:p('requests_deferred')};
}

export async function pollMetrics() {
  if (state.status !== 'running') return state.metrics;
  const owner=child;
  try {
    const res = await fetch(`http://127.0.0.1:${state.port}/metrics${state.metrics.dashboard?'?requests=all':''}`, {
      signal: AbortSignal.timeout(1500),
      headers: privateKey || state.apiKey ? { authorization: `Bearer ${privateKey || state.apiKey}` } : undefined,
    });
    if (!res.ok) throw new Error(String(res.status));
    const metrics=parseEngineMetrics(await res.text());
    if(child!==owner||state.status!=='running')return state.metrics;
    state.metrics = { ...metrics, sampledAt: new Date().toISOString() };
    if(state.metrics.dashboard) {
      state.metrics.dashboard.mcp=mcpStatus;
      if(!mcpPending&&Date.now()-mcpAt>10000) {
        mcpPending=true;mcpAt=Date.now();
        void fetch(`http://127.0.0.1:${state.port}/mcp`,{signal:AbortSignal.timeout(1500),headers:privateKey||state.apiKey?{authorization:`Bearer ${privateKey||state.apiKey}`}:undefined})
          .then(async response=>{mcpStatus=response.ok?await response.json():null;})
          .catch(()=>{mcpStatus=null;}).finally(()=>{mcpPending=false;});
      }
    }
  } catch {
    if(child!==owner||state.status!=='running')return state.metrics;
    state.metrics = { ...state.metrics, source: 'unavailable', sampledAt: new Date().toISOString() };
  }
  return state.metrics;
}

/* ------------------------------------------------------------------- status --- */

export function status() {
  const binary = state.binary ? { path: state.binary, source: state.binarySource, ...(state.runtimeVersion?{version:state.runtimeVersion}:{}) } : null;
  return {
    status: state.status,
    running: state.status === 'running',
    requestedBackend: state.requestedBackend,
    binary,
    pid: state.pid,
    port: state.port,
    modelId: state.modelId,
    modelPath: state.modelPath,
    contextCapacity: state.status === 'running' && child && contextOwner === child ? contextCapacity : null,
    startedAt: state.startedAt,
    error: state.error,
    args: publicArgs(),
    command: state.command,
    metrics: state.metrics,
    logCount: state.logs.length,
    ...(ramGuard ? { memoryGuard: ramGuard.snapshot() } : {}),
  };
}

export function currentPort() {
  return state.port;
}

export function isRunning() {
  return state.status === 'running';
}
