// No model/server required: execute the real renderer against a synthetic DOM that
// resets nested scroll positions when innerHTML replaces the corresponding nodes.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(process.env.STRATA_TEST_APP_JS || path.join(__dirname, '../serve/web/app.js'), 'utf8');
const renderer = source.slice(source.indexOf('function updateAssistant('), source.indexOf('function renderChat('));
function fixture() {
  let nodes = [{scrollTop:260, scrollLeft:85}, {scrollTop:140, scrollLeft:20}], writes = 0;
  const bubble = {scrollTop:90, scrollLeft:10, classList:{add(){},remove(){}},
    querySelectorAll(){return nodes;},
    set innerHTML(value){writes++; nodes=nodes.map(()=>({scrollTop:0,scrollLeft:0})); this.scrollTop=0;this.scrollLeft=0;}};
  const thinking = {scrollTop:180,scrollLeft:5,dataset:{},_text:'old thoughts',
    get textContent(){return this._text;},set textContent(v){this._text=v;this.scrollTop=0;this.scrollLeft=0;}};
  const det = {open:true,dataset:{touched:'1'}};
  const parts = {'details.think':det,'.thinking':thinking,'.think-title':{},'.st-bubble':bubble,'.meta-text':{},'[data-msg-copy]':{}};
  const el = {querySelector(s){return parts[s];}};
  const context = vm.createContext({settings:{show:true},fmt:String,answerHtml:m=>m.text});
  vm.runInContext(renderer+'; this.render=updateAssistant;',context);
  const message = {text:'long code and tool output',reasoning:'long thoughts'};
  return {bubble,thinking,message,get nodes(){return nodes;},get writes(){return writes;},
    render(streaming=true){context.render(el,message,streaming);}};
}
test('stream repaint preserves vertical and horizontal code/tool scroll positions',()=>{
  const f=fixture(); f.render();
  assert.deepEqual(f.nodes,[{scrollTop:260,scrollLeft:85},{scrollTop:140,scrollLeft:20}]);
});
test('unchanged content does not replace the answer DOM',()=>{
  const f=fixture(); f.render(); f.render(); assert.equal(f.writes,1);
});
test('manual reading position survives subsequent tokens and final repaint',()=>{
  const f=fixture(); f.render(); f.nodes[0].scrollTop=410; f.nodes[1].scrollLeft=70;
  f.message.text+=' more tokens'; f.render(); f.render(false);
  assert.equal(f.nodes[0].scrollTop,410); assert.equal(f.nodes[1].scrollLeft,70);
});
test('answer container and Thinking retain their independent reading positions',()=>{
  const f=fixture(); f.render();
  assert.equal(f.bubble.scrollTop,90); assert.equal(f.bubble.scrollLeft,10);
  assert.equal(f.thinking.scrollTop,180); assert.equal(f.thinking.scrollLeft,5);
});
// Exercise the streaming paint callback itself, including the large-delta case
// where a follower would no longer pass nearBottom() after content is appended.
const paintCode=source.match(/const paint = \(\) => \{[\s\S]*?\};/)[0];
for (const following of [false,true]) test(`outer chat ${following?'follows new content':'keeps manual position'} across large deltas`,()=>{
  const scroll={scrollTop:following?900:250,scrollHeight:1000,clientHeight:100};
  const context=vm.createContext({frame:1,el:{},m:{},$:()=>scroll,
    nearBottom:()=>scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<120,
    scrollDown(force){if(force || scroll.scrollHeight-scroll.scrollTop-scroll.clientHeight<120) scroll.scrollTop=scroll.scrollHeight;},
    updateAssistant(){scroll.scrollHeight+=1000;}});
  vm.runInContext(paintCode+'; paint();',context);
  assert.equal(scroll.scrollTop,following?2000:250);
});
