/** Local availability of a dedicated DeepSeek executable; this is not inference qualification. */
import {statSync} from 'node:fs';
import {basename,dirname,resolve} from 'node:path';
import {homedir} from 'node:os';
import {nativeServingPolicy} from './native-serving-policy.mjs';

export const DEEPSEEK_RUNTIME_UNAVAILABLE='The dedicated DeepSeek engine for this model is not available. Its files are kept. Install a compatible strata-dsv41 engine before loading it; the Qwen engine cannot run this model.';

/** Architecture is authoritative. A Qwen GGUF with DeepSeek in its name is never treated as DeepSeek. */
export function deepseekRuntime(config,configPath,{family=null,missing=[]}={}){
 if(nativeServingPolicy(config,family).id!=='deepseek41')return null;
 const unavailable={available:false,reason:DEEPSEEK_RUNTIME_UNAVAILABLE,executionCompatibility:'unverified'};
 const raw=config?.exe;
 if(typeof raw!=='string'||!raw||raw.startsWith('component:')||/[\0\r\n]/.test(raw))return unavailable;
 const expanded=raw==='~'?homedir():/^~[\\/]/.test(raw)?resolve(homedir(),raw.slice(2)):raw;
 const exe=resolve(typeof config.cwd==='string'&&config.cwd?config.cwd:dirname(configPath),expanded);
 // A generic Qwen engine and a directory with this name are not dedicated runtimes.
 if(!/^strata-dsv41(?:\.exe)?$/i.test(basename(exe))||missing.some(p=>p?.part==='engine'&&['missing','invalid'].includes(p.status)))return unavailable;
 try{if(!statSync(exe).isFile())return unavailable;}catch{return unavailable;}
 return {available:true,reason:null,executionCompatibility:'unverified',exe};
}
