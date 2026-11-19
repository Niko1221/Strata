/** Production profile capabilities and local-model search guidance; no network calls. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {buildProfile} from '../server/agent/profile.mjs';
import {manualDecision,planDecision} from '../server/agent/plugins/strata-gate.mjs';

const local=[{id:'local',baseUrl:'http://127.0.0.1:1/v1',modelId:'fictional',context:32768}];
const tools=()=>({off:new Set(),search:{provider:'builtin',key:''},servers:{},browserEnabled:false,
  browserPath:null,browserCli:null,webWithoutAsking:false});
const entries=p=>JSON.parse(p.patch.split('\n').slice(1).join('\n'));
const prompt=p=>entries(p).find(r=>r.id==='preset-standard').config.plugins.find(r=>r.id==='persona').config.prefix;

test('local model is instructed to verify current or uncertain facts using an available search tool',()=>{
  const p=buildProfile({local,tools:tools()});
  assert.match(prompt(p),/web_search.*before answering.*current|current.*before answering.*web_search/s);
  assert.match(prompt(p),/official|primary/);assert.match(prompt(p),/cite|source links/);
  assert.match(prompt(p),/greetings|rewriting/);assert.match(prompt(p),/untrusted/);
  assert.equal(entries(p).find(r=>r.id==='preset-standard').config.plugins.find(r=>r.id==='tool-web').config.search,true);
});

test('turning search off removes search guidance and keeps Manual permission intact',()=>{
  const settings=tools();settings.off.add('search');
  const p=buildProfile({local,tools:settings});
  assert.doesNotMatch(prompt(p),/Use web_search before answering/);
  const gate=entries(p).flatMap(r=>r.insert??[]).find(r=>r.id==='strata-gate');
  assert.equal(gate.config.web.skipAsk,false);
  assert.equal(entries(p).find(r=>r.id==='permission').config.defaultPreset,'strata-manual');
});

test('browser computer-use has isolated coordinate tools and text snapshots for a text-only local model',()=>{
  const settings={...tools(),browserEnabled:true,browserPath:'C:/fixture/browser.exe',browserCli:'C:/fixture/cli.js'};
  const p=buildProfile({local,tools:settings});
  const browser=entries(p).flatMap(r=>r.insert??[]).find(r=>r.id==='mcp-browser');
  assert.ok(browser.config.args.includes('--isolated'));
  assert.equal(browser.config.args[browser.config.args.indexOf('--caps')+1],'vision');
  assert.ok(browser.config.args.includes('--snapshot-boxes'));
  assert.equal(browser.config.args[browser.config.args.indexOf('--image-responses')+1],'omit');
  assert.match(prompt(p),/browser.*search|search.*browser/);
  assert.match(prompt(p),/snapshot|accessibility/);
  for(const argument of ['--extension','--cdp-endpoint','--user-data-dir','--no-sandbox'])assert.equal(browser.config.args.includes(argument),false);
  const off=buildProfile({local,tools:tools()});
  assert.equal(entries(off).flatMap(r=>r.insert??[]).some(r=>r.id==='mcp-browser'),false);
  assert.doesNotMatch(prompt(off),/For browser search/);
});

test('vision-capable browser route may return images; API-only profiles retain the existing persona',()=>{
  const settings={...tools(),browserEnabled:true,browserPath:'C:/fixture/browser.exe',browserCli:'C:/fixture/cli.js'};
  const p=buildProfile({local:[{...local[0],vision:true}],tools:settings});
  const browser=entries(p).flatMap(r=>r.insert??[]).find(r=>r.id==='mcp-browser');
  assert.equal(browser.config.args[browser.config.args.indexOf('--image-responses')+1],'allow');
  assert.doesNotMatch(prompt(buildProfile({local:[],tools:tools()})),/Use web_search before answering/);
});

test('computer-use coordinate actions still ask in Manual and are refused in Plan',()=>{
  for(const name of ['browser_mouse_click_xy','browser_mouse_drag_xy','browser_mouse_wheel','browser_mouse_down','browser_mouse_up']){
    assert.equal(manualDecision({name:'mcp__browser__'+name,arguments:{x:10,y:20}}).kind,'ask');
    assert.equal(planDecision({name:'mcp__browser__'+name,arguments:{x:10,y:20}}).kind,'deny');
  }
});
