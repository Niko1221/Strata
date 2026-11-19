import test from 'node:test';
import assert from 'node:assert/strict';
import {component,nodes,button,flush} from './helpers/ux-component.mjs';
test('search key success waits for PUT; failed save keeps key and error without success toast',async()=>{
  let reject,resolve;const state={groups:[],search:{provider:'exa',keySet:false},browser:{found:false},custom:[]};
  const c=await component('SettingsTools.tsx',{get:async()=>state,put:()=>new Promise((a,b)=>{resolve=a;reject=b;})});
  c.render('SettingsTools',{});await flush();let tree=c.render('SettingsTools',{});
  nodes(tree,n=>n.type==='input'&&n.props.type==='password')[0].props.onChange({target:{value:'fake-key'}});
  tree=c.render('SettingsTools',{});let saving=button(tree,'Save key').props.onClick();
  assert.equal(c.ctx.toasts.length,0);reject(new Error('Save failed'));await saving;
  tree=c.render('SettingsTools',{});assert.equal(nodes(tree,n=>n.type==='input'&&n.props.type==='password')[0].props.value,'fake-key');
  assert.equal(nodes(tree,n=>n.type==='ErrorNotice')[0].props.message,'Save failed');assert.equal(c.ctx.toasts.length,0);
  saving=button(tree,'Save key').props.onClick();resolve({...state,search:{provider:'exa',keySet:true}});await saving;
  tree=c.render('SettingsTools',{});assert.equal(nodes(tree,n=>n.type==='input'&&n.props.type==='password')[0].props.value,'');assert.equal(c.ctx.toasts.length,1);
});
