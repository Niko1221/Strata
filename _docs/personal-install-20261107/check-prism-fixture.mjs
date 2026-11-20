/** Diagnose fixture serialization against the immutable public provenance metadata only. */
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
const artifact=readFileSync('H:/Strata/validation/qwen35-mtp-load-20261107/runtime-prism-0.4.1-1965755c/runtime.json');
const fixture=readFileSync(new URL('../../app/host/tests/fixtures/prism-0.4.1-runtime.json',import.meta.url),'utf8');
assert.deepEqual(JSON.parse(artifact),JSON.parse(fixture));
const hash=value=>createHash('sha256').update(value).digest('hex'),lf=fixture.replaceAll('\r\n','\n');
console.log(JSON.stringify({semanticMatch:true,artifactSHA256:hash(artifact),lfSHA256:hash(lf),crlfSHA256:hash(lf.replaceAll('\n','\r\n')),crlfNoTrailingSHA256:hash(lf.trimEnd().replaceAll('\n','\r\n')),artifactCRLFCount:(artifact.toString().match(/\r\n/g)??[]).length,artifactTrailingNewline:artifact.at(-1)===10}));
