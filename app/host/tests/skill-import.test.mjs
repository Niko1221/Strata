import test from 'node:test';
import assert from 'node:assert/strict';
import { previewSkillMarkdown } from '../server/skill-import.mjs';
const markdown='---\nname: careful-review\ndescription: Review a change.\n---\n\nFollow these instructions exactly.\nSecond line.\n';
test('instruction-only preview uses existing parser, validates size/fields and does not write',()=>{
  const preview=previewSkillMarkdown(markdown);
  assert.equal(preview.name,'careful-review');assert.equal(preview.description,'Review a change.');
  assert.equal(preview.body,'Follow these instructions exactly.\nSecond line.');
  assert.deepEqual(preview.triggers,['careful review']);assert.equal(preview.enabled,false);
  for(const input of [undefined,'instructions without metadata',markdown.replace('careful-review',''),markdown.replace('Review a change.','x'.repeat(401)),markdown+'x'.repeat(40001),'あ'.repeat(23000)]) assert.throws(()=>previewSkillMarkdown(input));
});
test('import rejects duplicates across bundled, created and workspace names',()=>{
  for(const source of ['bundled','created','workspace'])assert.throws(()=>previewSkillMarkdown(markdown,[{name:'Careful Review',source}]),/already/);
});
