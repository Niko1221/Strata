import test from 'node:test';
import assert from 'node:assert/strict';
import {component,nodes,button,flush} from './helpers/ux-component.mjs';
test('chat search starts blank, sends query only after Search; close/late response never attaches',async()=>{
  const posts=[];let finish;let used=0;
  const c=await component('WebSearchDialog.tsx',{get:async()=>({available:true,destination:'DuckDuckGo / Bing'}),post:(...args)=>{posts.push(args);return new Promise(r=>finish=r);}});
  const props={open:true,sessionId:'a',onClose:()=>{},onUse:()=>used++};
  c.render('WebSearchDialog',props);await flush();let tree=c.render('WebSearchDialog',props);
  assert.equal(nodes(tree,n=>n.type==='input')[0].props.value,'');assert.equal(posts.length,0);
  nodes(tree,n=>n.type==='input')[0].props.onChange({target:{value:'user query'}});tree=c.render('WebSearchDialog',props);
  const pending=button(tree,'Search').props.onClick();assert.deepEqual(posts[0][1],{sessionId:'a',query:'user query',confirmed:true});
  tree=c.render('WebSearchDialog',props);button(tree,'Cancel').props.onClick();assert.equal(posts[0][2].aborted,true);
  finish({resultId:'late',expiresAt:Date.now()+1000,sources:[{url:'https://example.com',title:'late',snippet:'late'}]});await pending;await flush();
  tree=c.render('WebSearchDialog',props);assert.equal(button(tree,'Use results in question'),undefined);assert.equal(used,0);
});
test('results attach only after Use; empty and offline states stay visible and cannot attach',async()=>{
  for(const outcome of ['sources','empty','offline']){
    let selected;
    const result={resultId:'record',expiresAt:Date.now()+60000,sources:outcome==='sources'?[{url:'https://example.com/source',title:'Title',snippet:'Snippet'}]:[]};
    const c=await component('WebSearchDialog.tsx',{get:async()=>({available:true}),post:async()=>{if(outcome==='offline')throw new Error('Offline');return result;}});
    const props={open:true,sessionId:'a',onClose:()=>{},onUse:value=>selected=value};
    c.render('WebSearchDialog',props);await flush();let tree=c.render('WebSearchDialog',props);
    nodes(tree,n=>n.type==='input')[0].props.onChange({target:{value:'query'}});tree=c.render('WebSearchDialog',props);button(tree,'Search').props.onClick();await flush();tree=c.render('WebSearchDialog',props);
    assert.equal(selected,undefined);
    if(outcome==='sources'){button(tree,'Use results in question').props.onClick();assert.equal(selected,result);}
    else{assert.equal(button(tree,'Use results in question'),undefined);assert.ok(outcome==='offline'?nodes(tree,n=>n.type==='ErrorNotice').length:nodes(tree,n=>n.type==='p'&&n.props.children==='No results. Try another search query.').length);}
  }
});
