/** Real isolated browser/MCP search and coordinate-click smoke; no model is mocked as autonomous. */
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {createServer} from 'node:http';
import {existsSync,mkdirSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {findBrowser,findPlaywrightMcp} from '../../app/host/server/agent/browser-probe.mjs';
import {buildProfile} from '../../app/host/server/agent/profile.mjs';

const attempt=process.argv[2];assert.match(attempt??'',/^[a-z0-9-]+$/);
const work=join('H:/Strata/releases/repair-055-20261110-opencode','browser-search-'+attempt);
assert.equal(existsSync(work),false);mkdirSync(work);
const browser=findBrowser(),cli=findPlaywrightMcp('C:/Users/downl/Documents/ChatGPT/Strata/apps/desktop/src-tauri/resources/harness');
assert.ok(browser.found&&cli,'Staged browser component and installed browser required');
process.env.STRATA_BROWSER_HEADLESS='1';
const profile=buildProfile({local:[{id:'text-fixture',baseUrl:'http://127.0.0.1:1/v1',modelId:'text-only',vision:false}],
  tools:{off:new Set(),search:{provider:'builtin',key:''},browserEnabled:true,browserPath:browser.path,browserCli:cli,servers:{}}});
const rows=JSON.parse(profile.patch.split('\n').slice(1).join('\n'));
const config=rows.flatMap(r=>r.insert??[]).find(r=>r.id==='mcp-browser').config;
const result={passed:false,actualBrowser:true,actualMcp:true,isolated:true,scriptedToolCalls:true,
  autonomousModelSelection:false,publicSearchRequested:process.argv.includes('--public')||process.argv.includes('--public-ddg'),checks:[]};
let child,lines,server,exit,serial=0,total=0;
const pending=new Map();
function rpc(method,params,seconds=60){
  const id=++serial;
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>{pending.delete(id);reject(Error('MCP deadline: '+method));},seconds*1000);
    pending.set(id,{resolve:value=>{clearTimeout(timer);resolve(value);},reject:error=>{clearTimeout(timer);reject(error);}});
    child.stdin.write(JSON.stringify({jsonrpc:'2.0',id,method,params})+'\n');
  });
}
const content=value=>(value.content??[]).filter(c=>c.type==='text').map(c=>c.text).join('\n');
async function tool(name,args={}){
  const value=await rpc('tools/call',{name,arguments:args},90);
  assert.notEqual(value.isError,true,'Browser tool failed: '+content(value));
  return content(value);
}
try{
  const env=Object.fromEntries(Object.entries(process.env).filter(([key])=>!/(API_KEY|TOKEN|SECRET|PASSWORD)$/i.test(key)&&!/^STRATA_/.test(key)));
  child=spawn(config.command,[...config.args,'--output-dir',work],{cwd:work,env,windowsHide:true,stdio:['pipe','pipe','pipe']});
  exit=new Promise(resolve=>child.once('exit',(code,signal)=>resolve({code,signal})));
  let stderr='';child.stderr.on('data',bytes=>{assert.ok(stderr.length+bytes.length<1024*1024,'Bounded browser stderr');stderr+=bytes.toString();});
  lines=createInterface({input:child.stdout});
  lines.on('line',line=>{total+=Buffer.byteLength(line);assert.ok(total<8*1024*1024,'Bounded MCP output');const value=JSON.parse(line);if(value.id){const call=pending.get(value.id);if(call){pending.delete(value.id);if(value.error)call.reject(Error(value.error.message));else call.resolve(value.result);}}});
  await rpc('initialize',{protocolVersion:'2024-11-05',capabilities:{},clientInfo:{name:'strata-browser-acceptance',version:'1'}});
  child.stdin.write(JSON.stringify({jsonrpc:'2.0',method:'notifications/initialized'})+'\n');
  const tools=(await rpc('tools/list',{})).tools;
  result.offeredBrowserTools=tools.map(t=>t.name);
  result.inputSchemas=Object.fromEntries(tools.filter(t=>['browser_type','browser_mouse_click_xy'].includes(t.name)).map(t=>[t.name,t.inputSchema]));
  assert.ok(result.offeredBrowserTools.includes('browser_snapshot'));
  assert.ok(result.offeredBrowserTools.includes('browser_mouse_click_xy'),'Computer-use coordinate tool missing');
  result.checks.push('production-profile-offers-browser-coordinate-tools');
  server=createServer((req,res)=>{
    const url=new URL(req.url,'http://127.0.0.1');res.setHeader('content-type','text/html; charset=utf-8');
    if(url.pathname==='/source')res.end('<html><title>Fixture source</title><h1>Source result verified</h1><p>Public fictional result.</p></html>');
    else if(url.searchParams.get('q')==='Strata')res.end('<html><title>Search results</title><h1>Results for Strata</h1><a href="/source">Source result</a></html>');
    else res.end('<html><title>Fixture search</title><form><label>Search <input name="q"></label><button>Search</button></form></html>');
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const base='http://127.0.0.1:'+server.address().port;
  await tool('browser_navigate',{url:base});
  const first=await tool('browser_snapshot');writeFileSync(join(work,'search-page.txt'),first);
  const ref=/textbox "Search"[^\n]*\[ref=([^\]]+)\]/.exec(first)?.[1];assert.ok(ref,'Search input missing from real snapshot');
  await tool('browser_type',{target:ref,text:'Strata',submit:true});
  const snapshot=await tool('browser_snapshot');
  assert.match(snapshot,/Results for Strata/);result.checks.push('real-browser-query-and-submit');
  const box=/link "Source result"[^\n]*\[box=([^\]]+)\]/.exec(snapshot)?.[1];assert.ok(box,'Result coordinate box missing');
  const [x,y,width,height]=box.split(',').map(Number);assert.ok([x,y,width,height].every(Number.isFinite));
  const clickSchema=tools.find(t=>t.name==='browser_mouse_click_xy').inputSchema;
  result.coordinateInputSchema=clickSchema;
  await tool('browser_mouse_click_xy',{x:x+width/2,y:y+height/2});
  assert.match(await tool('browser_snapshot'),/Source result verified/);result.checks.push('real-coordinate-click-opens-result');
  if(result.publicSearchRequested){
    result.publicProvider=process.argv.includes('--public-ddg')?'DuckDuckGo':'Bing';
    await tool('browser_navigate',{url:result.publicProvider==='DuckDuckGo'?'https://duckduckgo.com/?q=Microsoft+Playwright+GitHub&ia=web':'https://www.bing.com/search?q=Microsoft+Playwright+GitHub'});
    await tool('browser_wait_for',{time:2});
    const publicPage=await tool('browser_snapshot');
    writeFileSync(join(work,'public-search.txt'),publicPage);
    assert.doesNotMatch(publicPage,/verify you are human|captcha|unusual traffic/i,'Search site refused browser access');
    assert.match(publicPage,/(?:link|heading) "[^"\n]*(?:microsoft\/playwright|GitHub[^"\n]*Playwright|Playwright[^"\n]*GitHub)/i,'Actual result link missing; query echo is not evidence');
    result.checks.push('public-browser-search-results-observed');
  }
  await tool('browser_close');result.passed=true;
}catch(error){result.error=error.message;process.exitCode=1;}
finally{
  lines?.close();child?.stdin.end();
  if(child&&child.exitCode===null){await Promise.race([exit,new Promise(resolve=>setTimeout(resolve,5000))]);if(child.exitCode===null){child.kill();await exit;}}
  for(const call of pending.values())call.reject(Error('Owned MCP closed'));pending.clear();
  if(server)await new Promise(resolve=>server.close(resolve));
  writeFileSync(join(work,'result.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result));
}
