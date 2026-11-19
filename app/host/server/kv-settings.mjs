/** Independent cache precision. Unknown types never silently become q8_0. */
const TYPES = Object.freeze({ INT8: 'q8_0', FP16: 'f16', BF16: 'bf16', Q8_0: 'q8_0', Q4_0: 'q4_0', TURBO2:'turbo2', TURBO3:'turbo3', TURBO4:'turbo4' });
/** Launches use explicit -ctk/-ctv. A different fork's ambient type can fail
 * during environment parsing before CLI overrides are considered. Scope the
 * policy to the child and keep automatic/layer-adaptive selection disabled
 * until a separately qualified explicit auto profile exists. */
export function explicitKvEnvironment(base = process.env) {
  const env = { ...base };
  for (const key of Object.keys(env)) {
    if (['LLAMA_ARG_CACHE_TYPE_K', 'LLAMA_ARG_CACHE_TYPE_V',
      'TURBO_AUTO_ASYMMETRIC', 'TURBO_LAYER_ADAPTIVE'].includes(key.toUpperCase())) delete env[key];
  }
  env.TURBO_AUTO_ASYMMETRIC = '0';
  env.TURBO_LAYER_ADAPTIVE = '0';
  return env;
}
export function resolveKvSettings(expert = []) {
  const values = new Map(expert.map(p => [p.id, p.value]));
  const legacy = values.get('ctx.kvMode') ?? 'INT8';
  const convert = value => {
    const result = TYPES[String(value).toUpperCase()];
    if (!result) {
      const error = new Error(`Unsupported cache type: ${String(value)}`);
      error.code = 'KV_UNSUPPORTED_TYPE';
      throw error;
    }
    return result;
  };
  const select = key => {
    const value = values.get(key);
    return convert(value === undefined || value === 'Inherit' ? legacy : value);
  };
  const k = select('ctx.kType'), v = select('ctx.vType');
  if (values.get('perf.flashAttn') === false && !['f16', 'bf16'].includes(v)) {
    const error = new Error('A quantized V cache requires flash attention.');
    error.code = 'KV_FLASH_ATTENTION_REQUIRED';
    throw error;
  }
  return { k, v };
}

export function requireModelKvCompatible(model, expert = []) {
  const { k, v } = resolveKvSettings(expert);
  if (!model?.local) return;
  for(const [type,width] of [[k,model.local.keyLength],[v,model.local.valueLength]]) {
    if(type.startsWith('turbo')) {
      // The pinned fork pads rotation groups to 128; the actual head need not be 128 wide.
      if(!Number.isInteger(width)||width<1) {
        const error=new Error('TurboQuant needs verified K/V attention geometry. Re-import the GGUF to refresh its metadata, or choose a supported cache precision.');
        error.code='KV_TURBO_MODEL_WIDTH';throw error;
      }
      continue;
    }
    if(['f16','bf16'].includes(type)||model.kvModes?.includes('INT8'))continue;
    const error = new Error('This GGUF needs an FP16 K/V cache. Set KV mode to FP16 and K/V cache precision to Inherit or FP16, then load it again.');
    error.code = 'KV_MODEL_WIDTH';throw error;
  }
  return;
}

/** Retains the host's nominal memory estimate; actual allocation is measured
 * by the engine. Assumes equal K/V widths, as the previous combined selector did. */
export function kvMemoryFactor(expert = []) {
  const { k, v } = resolveKvSettings(expert);
  const factors = { f16: 2, bf16: 2, q8_0: 1, q4_0: .5, turbo2:34/128, turbo3:50/128, turbo4:68/128 };
  return (factors[k] + factors[v]) / 2;
}
