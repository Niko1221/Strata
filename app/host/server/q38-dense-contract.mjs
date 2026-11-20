/** Product settings for the owned scalar dense runner; no memory-fit estimates. */
import {isAbsolute} from 'node:path';
import {DENSE_KV} from './native-serving-policy.mjs';
import {capture,flagOf,groups,merge} from './native-user-args.mjs';
export const denseConfig = cfg => cfg?.arch === 'qwen38_dense';
export function denseRunConfig(config, args={}) {
 const cfg={...config,...(args.values??{})};
 for(const key of ['cache_type_k','cache_type_v','q36_memory','q36_run','spin8_mode','spin8_native_config','spin8_native_sha256','layer_split','replicas','split_fallback','slot_save_path','gpu_layers','refusal_direction_removal','ds_experts'])if(cfg[key]!=null&&cfg[key]!==''&&cfg[key]!==false)throw new Error(`Dense native does not support ${key}.`);
 if(args.vision&&args.vision!=='no'||cfg.vision||cfg.parallel!=null&&cfg.parallel!==1||cfg.mtp===true||cfg.drafting!=null&&cfg.drafting!=='off')throw new Error('Dense native supports one text session with MTP/drafting off.');
 // Generated args are fallback data. Explicit user_args and captured raw edits
 // retain authority over scalar defaults, including a disabled language profile.
 const owned=Array.isArray(cfg.args_written)?capture(cfg,new Set()):merge(cfg.args??[],cfg.user_args??[]);
 const explicit=new Set(groups(owned).map(flagOf));
 const raw=merge(cfg.args??[],owned),values=new Map();
 const flags=['--model','--ctx-size','--kv','--output-language','--spin8-view','--spin8-profile'];
 if(raw.length%2)throw new Error('Dense native options require flag/value pairs.');
 for(let i=0;i<raw.length;i+=2){if(!flags.includes(raw[i])||values.has(raw[i])||typeof raw[i+1]!=='string'||!raw[i+1]||/[\x00-\x1f\x7f]/.test(raw[i+1]))throw new Error('Unsupported, repeated or malformed dense native option.');values.set(raw[i],raw[i+1]);}
 if(!isAbsolute(values.get('--model')??''))throw new Error('Dense native needs an absolute original GGUF path.');
 if(explicit.has('--ctx-size')&&!/^[0-9]+$/.test(values.get('--ctx-size')))throw new Error('Invalid dense native context argument.');
 const context=explicit.has('--ctx-size')?Number(values.get('--ctx-size')):args.context??cfg.max_context??Number(values.get('--ctx-size')??4096);
 const kv=explicit.has('--kv')?values.get('--kv'):args.kv??cfg.dense_kv??values.get('--kv')??'auto';
 const language=explicit.has('--output-language')?values.get('--output-language'):cfg.output_language??values.get('--output-language')??'ja';
 if(!Number.isInteger(context)||context<1||context>262144||!DENSE_KV.includes(kv)||!['ja','model-default'].includes(language))throw new Error('Invalid dense native context, KV or output language.');
 if(values.has('--spin8-view')&&!['vector','plus','minus'].includes(values.get('--spin8-view')))throw new Error('Invalid dense native Spin8 view.');
 values.set('--ctx-size',String(context));values.set('--kv',kv);values.set('--output-language',language);
 const argv=[...values].flat();
 return {...cfg,args:argv,args_written:[...argv],user_args:owned,max_context:context,dense_kv:kv,output_language:language,parallel:1,drafting:'off',mtp:false,split_engine:'off'};
}
