/** Distinguish the exact existing locale gaps from new sidebar regressions. */
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
const root=fileURLToPath(new URL('../../',import.meta.url));
const baseline='c1ceee69e4d5d9e65f9a62b0a8ff10f8b93b3d4c';
const before=name=>JSON.parse(execFileSync('git',['show',`${baseline}:apps/host/src/locales/${name}.json`],{cwd:root,encoding:'utf8'}));
const after=name=>JSON.parse(readFileSync(new URL(`../../app/host/src/locales/${name}.json`,import.meta.url),'utf8'));
const oldEnglish=before('en'),english=after('en'),keys=Object.keys(english).filter(key=>key.startsWith('wc55'));
assert.equal(keys.length,29);const rows=[];
for(const language of ['en','ja','zh','de','fr','es','pt','ru','ko']){
 const old=before(language),current=after(language);
 const oldMissing=Object.keys(oldEnglish).filter(key=>!(key in old)),missing=Object.keys(english).filter(key=>!(key in current));
 assert.ok(missing.every(key=>oldMissing.includes(key)),`${language}: new missing key`);
 for(const key of keys){assert.ok(typeof current[key]==='string'&&current[key].trim());assert.deepEqual((current[key].match(/\{\d+\}/g)??[]).sort(),(english[key].match(/\{\d+\}/g)??[]).sort(),`${language}:${key}`);}
 rows.push({language,baselineMissing:oldMissing.length,currentMissing:missing.length,newKeysTranslated:keys.length});
}
console.log(JSON.stringify({noNewMissingKeys:true,baseline,rows},null,2));
