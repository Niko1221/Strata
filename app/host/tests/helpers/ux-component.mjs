/** Exercise the shipped React handlers with isolated hooks/API stubs; no browser/GUI claim. */
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

export async function component(file,api={}){
  const key='ux'+Math.random().toString(36).slice(2),ctx={states:[],cursor:0,effects:[],api,toasts:[],store:{workspaces:[{id:'root',name:'Root'}]},loads:0};
  globalThis[key]=ctx;
  const stubs={
    react:`const c=globalThis[${JSON.stringify(key)}];export function useState(initial){const i=c.cursor++;if(!(i in c.states))c.states[i]=initial;return [c.states[i],value=>{c.states[i]=typeof value==='function'?value(c.states[i]):value;}];}export function useRef(initial){return useState({current:initial})[0];}export function useEffect(fn,deps){const i=c.cursor++,before=c.states[i];if(!before||deps.some((v,n)=>v!==before.deps[n])){before?.cleanup?.();c.states[i]={deps};c.effects.push(()=>c.states[i].cleanup=fn());}}`,
    jsx:`export const Fragment='Fragment';export const jsx=(type,props)=>({type,props});export const jsxs=jsx;`,
    ui:`export const Btn='Btn',Dialog='Dialog',ErrorNotice='ErrorNotice',Field='Field',Icon='Icon',Select='Select';`,
    client:`export const api=globalThis[${JSON.stringify(key)}].api;`,
    store:`const c=globalThis[${JSON.stringify(key)}];export const pushToast=(...args)=>c.toasts.push(args);export const useStore=fn=>fn(c.store);export const setSkills=async skills=>{c.store.skills=skills;};export const loadSkills=async()=>{c.loads++;};export const setSettingsSection=()=>{};`,
    router:`export const navigate=()=>{};`,
    i18n:`export const tr=(s,values)=>values?values.reduce((v,x,i)=>v.replace('{'+i+'}',String(x)),s):s;`,
  };
  const compiled=await build({entryPoints:[fileURLToPath(new URL('../../src/components/'+file,import.meta.url))],bundle:true,write:false,format:'esm',platform:'node',jsx:'automatic',logLevel:'silent',plugins:[{name:'isolated-ui',setup(b){b.onResolve({filter:/^(react(?:\/jsx-runtime)?|\.\.?\/(?:lib\/)?(?:ui|client|store|router|i18n))$/},args=>({path:args.path==='react/jsx-runtime'?'jsx':args.path.split('/').at(-1),namespace:'stub'}));b.onLoad({filter:/.*/,namespace:'stub'},args=>({contents:stubs[args.path],loader:'js'}));}}]});
  const module=await import('data:text/javascript;base64,'+Buffer.from(compiled.outputFiles[0].text).toString('base64'));
  return {...ctx,ctx,module,render(name,props){ctx.cursor=0;const tree=module[name](props);for(const effect of ctx.effects.splice(0))effect();return tree;}};
}
const text=node=>typeof node==='string'?node:Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):'';
function nodes(node,predicate){if(!node||typeof node!=='object')return [];if(Array.isArray(node))return node.flatMap(n=>nodes(n,predicate));return [...(predicate(node)?[node]:[]),...Object.values(node.props??{}).flatMap(v=>nodes(v,predicate))];}
export const button=(tree,label)=>nodes(tree,n=>n.type==='Btn'&&text(n)===label)[0];
export const flush=()=>new Promise(resolve=>setImmediate(resolve));


export {nodes};
