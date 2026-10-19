/** Editable native keys. Unknown config keys and program paths stay intact. */
export const NATIVE_FIELDS=[
 {id:'reasoning_budget_tokens',label:'Thinking token budget',group:'Responses',type:'number',min:0,max:1048576,hint:'Leave blank for the model default.'},
 {id:'fit_max_tokens',label:'Fit output to remaining context',group:'Responses',type:'boolean',hint:'Shortens the requested output cap when the prompt leaves less room.'},
 {id:'anthropic_thinking',label:'Anthropic thinking',group:'Responses',type:'enum',choices:['model','on_request'],hint:'Follow the model default or think only when requested.'},
 {id:'effort_position',label:'Reasoning effort position',group:'Responses',type:'enum',choices:['start','end']},
 {id:'aliases',label:'API model aliases',group:'Serving',type:'array',hint:'Comma-separated names clients may use for this model.'},
 {id:'idle_unload_s',label:'Unload after idle seconds',group:'Serving',type:'number',min:0,max:86400,hint:'Zero keeps the model loaded.'},
 {id:'lazy_load',label:'Load on first request',group:'Serving',type:'boolean'},
 {id:'batchMtp',label:'MTP per request',group:'Serving',type:'boolean',hint:'Each concurrent request uses the draft layer. Left alone, Strata turns it on for one graphics card with 24 GB or more and several requests at once, and off otherwise; choose on or off here to decide yourself.'},
 {id:'parallel',label:'Concurrent requests',group:'Serving',type:'number',min:1,max:8,hint:'More slots need more memory and can slow each response.'},
 {id:'engine_silence_s',label:'Engine stall timeout (seconds)',group:'Recovery',type:'number',min:10,max:3600},
 {id:'api_monitor',label:'Keep API request metadata',group:'Recovery',type:'boolean'},
 {id:'layer_split',label:'Layer split',group:'GPU & memory',type:'string',hint:'auto, or increasing split points such as 18 or 16,32.'},
 {id:'gpu_order',label:'GPU order',group:'GPU & memory',type:'enum',choices:['auto','as_given']},
 {id:'split_skip_if_fits',label:'Use one GPU when the model fits',group:'GPU & memory',type:'boolean'},
 {id:'vram_elastic',label:'Elastic GPU expert cache',group:'GPU & memory',type:'boolean'},
 {id:'vram_reserve_mib',label:'GPU memory for other apps (MiB)',group:'GPU & memory',type:'number',min:0,max:131072,hint:'Applied as --vram-reserve-mib when loading.'},
 {id:'vision_tokens',label:'Maximum image tokens',group:'Image reading',type:'number',min:1,max:16384},
 {id:'vision_min_tokens',label:'Minimum image tokens',group:'Image reading',type:'number',min:0,max:16384},
 {id:'mcp_timeout_s',label:'Tool timeout (seconds)',group:'Tools',type:'number',min:1,max:3600},
 {id:'mcp_max_rounds',label:'Maximum tool rounds',group:'Tools',type:'number',min:1,max:100},
 {id:'mcp_max_result_chars',label:'Maximum tool result characters',group:'Tools',type:'number',min:100,max:1000000},
];
export function validateNativeSettings(values){
 if(!values||typeof values!=='object'||Array.isArray(values))throw new Error('Native settings must be an object.');
 const result={};
 for(const [key,value] of Object.entries(values)){

  if(key==='args'){if(!Array.isArray(value)||value.length>200||value.some(v=>typeof v!=='string'||v.length>4000||/[\n\r\0]/.test(v))||value.some(v=>['--host','--port','--api-key'].includes(v)))throw new Error('Engine arguments must be text values. Network access stays managed by Strata.');result.args=[...value];continue;}
  if(key==='env'){if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Engine environment must be an object.');const env={};for(const [name,text] of Object.entries(value)){if(!/^STRATA_[A-Z0-9_]+$/.test(name)||typeof text!=='string'||text.length>4000||text.includes('\0'))throw new Error('Use STRATA_ environment flags with text values.');env[name]=text;}result.env=env;continue;}
  if(key==='sampling'){if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('Sampling must be an object.');const limits={temperature:[0,10],top_p:[Number.MIN_VALUE,1],top_k:[1,64],min_p:[0,1],repetition_penalty:[Number.MIN_VALUE,10],presence_penalty:[0,10],frequency_penalty:[0,10],penalty_last_n:[0,1048576],seed:[1,2147483647]};for(const [name,n] of Object.entries(value)){if(name==='experimental_speed_projection'){if(typeof n!=='boolean')throw new Error('Speed projection must be on or off.');continue;}if(!limits[name]||!Number.isFinite(n)||(['top_k','seed','penalty_last_n'].includes(name)&&!Number.isInteger(n))||n<limits[name][0]||n>limits[name][1])throw new Error(`Choose a valid sampling ${name}.`);}result.sampling={...value};continue;}
  const field=NATIVE_FIELDS.find(f=>f.id===key);if(!field)throw new Error(`Unknown native setting: ${key}`);
  if(value===null||value===''){result[key]=null;continue;}
  if(field.type==='number'&&(!Number.isFinite(value)||value<field.min||value>field.max||!Number.isInteger(value)))throw new Error(`${field.label}: choose a whole number from ${field.min} to ${field.max}.`);
  if(field.type==='boolean'&&typeof value!=='boolean')throw new Error(`${field.label}: choose on or off.`);
  if(field.type==='string'&&(typeof value!=='string'||value.length>256))throw new Error(`${field.label}: enter at most 256 characters.`);
  if(field.type==='enum'&&!field.choices.includes(value))throw new Error(`${field.label}: choose a supported option.`);
  if(field.type==='array'&&(!Array.isArray(value)||value.length>24||value.some(v=>typeof v!=='string'||!v.trim()||v.length>80)||new Set(value).size!==value.length))throw new Error('Choose distinct model aliases, up to 80 characters each.');
  if(key==='layer_split'&&value!=='auto'){const points=value.split(',').map(Number);if(!points.length||points.some((n,i)=>!Number.isInteger(n)||n<1||n>=48||(i>0&&n<=points[i-1])))throw new Error('Layer split points must increase from 1 to 47, for example 16,32.');}
  result[key]=value;
 }
 return result;
}
const MASK='••••••••';
export function publicMcpServers(servers){return Object.fromEntries(Object.entries(servers).map(([name,s])=>[name,{...s,...(s.env?{env:Object.fromEntries(Object.keys(s.env).map(k=>[k,MASK]))}:{}),...(s.headers?{headers:Object.fromEntries(Object.keys(s.headers).map(k=>[k,MASK]))}:{})}]));}
export function validateMcpServers(servers,previous={}){
 if(!servers||typeof servers!=='object'||Array.isArray(servers)||Object.keys(servers).length>30)throw new Error('MCP servers must be an object with at most 30 entries.');
 const out={};for(const [name,s] of Object.entries(servers)){
  if(!/^[a-z0-9_-]{1,64}$/i.test(name)||!s||typeof s!=='object'||Array.isArray(s))throw new Error('Give each tool server a simple, unique name.');
  if(!!s.command===!!s.url)throw new Error(`${name}: choose a command or HTTP URL.`);
  for(const key of Object.keys(s))if(!['command','args','env','cwd','url','headers','disabled'].includes(key))throw new Error(`${name}: unsupported property ${key}.`);
  if(s.command&&(typeof s.command!=='string'||s.command.length>1000||/[\n\r\0]/.test(s.command)))throw new Error(`${name}: choose a valid program.`);
  if(s.url){let url;try{url=new URL(s.url);}catch{throw new Error(`${name}: enter a valid HTTP URL.`);}if(!['http:','https:'].includes(url.protocol)||url.username||url.password)throw new Error(`${name}: use an HTTP URL without embedded credentials.`);}
  if(s.args&&(!Array.isArray(s.args)||s.args.length>100||s.args.some(v=>typeof v!=='string'||v.length>4000||v.includes('\0'))))throw new Error(`${name}: arguments must be a list of strings.`);
  if(s.cwd!==undefined&&(typeof s.cwd!=='string'||s.cwd.length>4000||/[\r\n\0]/.test(s.cwd)))throw new Error(`${name}: choose a valid working directory.`);
  if(s.disabled!==undefined&&typeof s.disabled!=='boolean')throw new Error(`${name}: disabled must be true or false.`);
  out[name]={...s};for(const key of ['env','headers'])if(s[key]){if(typeof s[key]!=='object'||Array.isArray(s[key]))throw new Error(`${name}: ${key} must be an object.`);out[name][key]={};for(const [k,v] of Object.entries(s[key])){if(typeof v!=='string'||v.length>16000||v.includes('\0')||(key==='headers'&&(/[\r\n]/.test(v)||!/^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/.test(k))))throw new Error(`${name}: ${key} values must be text.`);if(v===MASK){if(previous[name]?.[key]?.[k]===undefined)throw new Error(`${name}: enter the new ${k} value.`);out[name][key][k]=previous[name][key][k];}else out[name][key][k]=v;}}
 }
 return out;
}

const sensitive=name=>/KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL/i.test(name);
export function publicNativeEnv(env){return Object.fromEntries(Object.entries(env).filter(([name])=>/^STRATA_[A-Z0-9_]+$/.test(name)).map(([name,value])=>[name,sensitive(name)?MASK:value]));}
export function restoreNativeEnv(env,previous={}){const result=Object.fromEntries(Object.entries(previous).filter(([name])=>!/^STRATA_[A-Z0-9_]+$/.test(name)));for(const [name,value] of Object.entries(env)){if(value===MASK){if(previous[name]===undefined)throw new Error(`Enter a new value for ${name}.`);result[name]=previous[name];}else result[name]=value;}return result;}
