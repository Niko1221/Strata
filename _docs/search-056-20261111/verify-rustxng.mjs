import { mkdirSync, writeFileSync } from 'node:fs';
import { createRustxngRuntime } from '../../app/host/server/rustxng-runtime.mjs';
import { createRustxngSearch } from '../../app/host/server/web-search-rustxng.mjs';

const out = 'H:/Strata/releases/search-056-20261111-codex';
const runtime = createRustxngRuntime();
try {
  const [a,b] = await Promise.all([runtime.start(),runtime.start()]);
  if (a!==b) throw new Error('Concurrent starts created different endpoints.');
  const config = await (await fetch(new URL('config',a))).json();
  const health = await fetch(new URL('healthz',a));
  if (!health.ok) throw new Error('Health check failed.');
  const search = createRustxngSearch({endpoint:a});
  const searches=[];
  for (const query of ['tokio runtime documentation','Tauri NSIS Windows installer documentation']) {
    const started=Date.now(), result=await search.search({query,maxResults:5});
    if (!result.sources.length) throw new Error('No search results.');
    searches.push({query,milliseconds:Date.now()-started,sources:result.sources});
  }
  mkdirSync(out,{recursive:true});
  const result={version:config.version,profile:'full',endpoint:a,health:true,config:true,concurrentStartShared:true,searches};
  writeFileSync(out+'/rustxng-live.json',JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify(result,null,2));
} finally { await runtime.close(); }
try { await runtime.start(); throw new Error('Closed runtime restarted.'); } catch(error) { if(!/closed/.test(error.message))throw error; }
console.log('RustXNG lifecycle check passed.');
