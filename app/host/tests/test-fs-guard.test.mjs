import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,rmdirSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,parse} from 'node:path';
import {assertScratchPath} from '../scripts/test-fs-guard.mjs';
test('recursive cleanup accepts only a child of its dedicated scratch tree',()=>{
 const root=mkdtempSync(join(tmpdir(),'strata-delete-guard-'));
 const child=join(root,'owned');mkdirSync(child);
 assert.equal(assertScratchPath(child,root),child);
 for(const path of [root,parse(root).root,join(parse(root).root,'Strata'),root+'-sibling'])assert.throws(()=>assertScratchPath(path,root),/outside scratch/);
 rmdirSync(child);rmdirSync(root);
});
