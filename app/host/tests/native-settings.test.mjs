import test from 'node:test';
import assert from 'node:assert/strict';
import {validateNativeSettings,validateMcpServers,publicMcpServers,publicNativeEnv,restoreNativeEnv} from '../server/native-settings.mjs';
import {checkRunConfig} from '../server/run-config-check.mjs';
import {validateChatImages} from '../server/chat-images.mjs';
test('native settings reject invalid ranges and unmanaged network arguments',()=>{
 assert.deepEqual(validateNativeSettings({parallel:2,layer_split:'16,32',reasoning_budget_tokens:null,sampling:{temperature:.7}}),{parallel:2,layer_split:'16,32',reasoning_budget_tokens:null,sampling:{temperature:.7}});
 for(const value of [{parallel:0},{parallel:1.2},{layer_split:'32,16'},{layer_split:'48'},{exe:'/tmp/program'},{args:['--host','0.0.0.0']},{env:{HOME:'/tmp'}},{sampling:{top_p:2}},{anthropic_thinking:'always'}])assert.throws(()=>validateNativeSettings(value));
});
test('MCP secrets round trip masked values and reject malformed transport settings',()=>{
 const previous={files:{command:'node',args:['server.mjs'],env:{API_KEY:'private-key'},disabled:false},remote:{url:'https://example.test/mcp',headers:{Authorization:'Bearer private'}}};
 const masked=publicMcpServers(previous);assert.equal(masked.files.env.API_KEY,'••••••••');assert.equal(JSON.stringify(masked).includes('private'),false);assert.deepEqual(validateMcpServers(masked,previous),previous);
 for(const value of [{x:{command:'node',url:'https://example.test'}},{x:{url:'file:///tmp'}},{x:{command:'node',cwd:123}},{x:{url:'https://example.test',headers:{Authorization:'secret\r\nX: bad'}}},{x:{command:'node',env:{KEY:'••••••••'}}}])assert.throws(()=>validateMcpServers(value));
});
test('environment masks preserve credentials and unrelated existing configuration',()=>{
 const previous={PATH:'/native/bin',STRATA_API_KEY:'private',STRATA_PREFILL:'2'};
 assert.deepEqual(publicNativeEnv(previous),{STRATA_API_KEY:'••••••••',STRATA_PREFILL:'2'});
 assert.deepEqual(restoreNativeEnv({STRATA_API_KEY:'••••••••',STRATA_PREFILL:'3'},previous),{...previous,STRATA_PREFILL:'3'});
 assert.throws(()=>restoreNativeEnv({STRATA_TOKEN:'••••••••'}));
});
test('chat accepts bounded attached images and rejects remote or oversized image payloads',()=>{
 const photo={name:'photo.png',url:'data:image/png;base64,aGVsbG8='};assert.deepEqual(validateChatImages([photo]),[photo]);
 assert.throws(()=>validateChatImages([{url:'https://private.test/image.png'}]));assert.throws(()=>validateChatImages(Array(5).fill(photo)));assert.throws(()=>validateChatImages([{url:'data:image/png;base64,'+Buffer.alloc(6*1024*1024+1).toString('base64')}]));
});
test('native KV cache types and Spin(8) capsule settings validate enums, digests and pairing',()=>{
 const digest='a'.repeat(64);
 assert.deepEqual(
  validateNativeSettings({cache_type_k:'turbo4',cache_type_v:'q8_0',spin8_mode:'consensus',spin8_native_config:' /models/spin8.json ',spin8_native_sha256:digest}),
  {cache_type_k:'turbo4',cache_type_v:'q8_0',spin8_mode:'consensus',spin8_native_config:'/models/spin8.json',spin8_native_sha256:digest}
 );
 for(const bad of [
  {cache_type_k:'turbo99'},
  {cache_type_v:'fp8'},
  {spin8_mode:'spinor-plus'},
  {spin8_native_config:'/models/spin8.json'},
  {spin8_native_sha256:digest},
  {spin8_native_config:'/models/spin8.json',spin8_native_sha256:'A'.repeat(64)},
  {spin8_mode:'off',spin8_native_config:'/models/spin8.json',spin8_native_sha256:digest},
 ])assert.throws(()=>validateNativeSettings(bad));
 assert.equal(checkRunConfig(JSON.stringify({cache_type_k:'turbo3',cache_type_v:'turbo2',spin8_mode:'vector',spin8_native_config:'/models/spin8.json',spin8_native_sha256:digest})).ok,true);
 assert.equal(checkRunConfig(JSON.stringify({spin8_native_config:'/models/spin8.json'})).ok,false);
 assert.equal(checkRunConfig(JSON.stringify({spin8_mode:'off',spin8_native_config:'/models/spin8.json',spin8_native_sha256:digest})).ok,false);
});
