/** Actual store request ownership with controlled promises; no Host, network, filesystem or browser. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const here = path => fileURLToPath(new URL(path, import.meta.url));
let sequence = 0;
const turn = () => new Promise(resolve => setImmediate(resolve));
async function fixture() {
    globalThis.window = globalThis;
    globalThis.document = { addEventListener() {}, visibilityState: 'visible', documentElement: { dataset: {}, style: { setProperty() {} }, classList: { toggle() {}, add() {}, remove() {} } } };
    globalThis.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
    globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
    globalThis.sessionStorage = { getItem: () => null, setItem() {}, removeItem() {} };
    globalThis.__imagePatchGates = [];
    globalThis.__hostSelected = 'q6';
    globalThis.__hostApplied = null;
    globalThis.__imageApplies = 0;
    const output = await build({ entryPoints: [here('../src/lib/store.ts')], bundle: true, write: false, format: 'esm', platform: 'node', alias: { react: here('./helpers/react-stub.mjs') }, plugins: [{ name: 'controlled-client', setup(b) {
        b.onResolve({ filter: /\/client$/ }, () => ({ path: 'client', namespace: 'controlled' }));
        b.onLoad({ filter: /.*/, namespace: 'controlled' }, () => ({ contents: `
            export class ApiError extends Error {}
            const forbidden=async()=>{throw Error('unexpected non-selection API')};
            const library=()=>({selected:globalThis.__hostSelected,applied:globalThis.__hostApplied??'off',variants:[],generationAvailable:false});
            export const api={get:forbidden,post:async path=>{if(path!=='/api/image-models/apply')return forbidden();globalThis.__imageApplies++;globalThis.__hostApplied=globalThis.__hostSelected;return library()},put:forbidden,del:forbidden,delete:forbidden,patch:(path,body)=>new Promise((resolve,reject)=>globalThis.__imagePatchGates.push({path,body,resolve,reject,commit:()=>{globalThis.__hostSelected=body.selected;resolve(library())}}))};
            export const onEvent=()=>()=>{},streamPost=forbidden;` }));
    } }] });
    const store = await import(`data:text/javascript;base64,${Buffer.from(output.outputFiles[0].text+'\n// fixture '+(++sequence)).toString('base64')}`);
    const library = selected => ({ selected, applied: 'off', variants: [], generationAvailable: false });
    const initial = store.selectImageModel('q6');
    await turn();
    __imagePatchGates[0].resolve(library('q6')); await initial;
    return { store, gates: __imagePatchGates, library };
}

test('actual store synchronously publishes intended selection and an older reply cannot clear current pending state', async () => {
    const { store, gates, library } = await fixture();
    const old = store.selectImageModel('off');
    await turn();
    const current = store.selectImageModel('q6');
    await turn();
    assert.equal(store.getState().imageModels.selected, 'q6');
    assert.equal(store.getState().busy['image:select'], true);
    assert.deepEqual(gates.slice(1).map(g=>[g.path,g.body.selected]), [['/api/image-models','off']]);
    assert.equal(store.imageSelectionReady('q6'),false);
    await assert.rejects(store.applyImageModel('q6'),/Confirm/);
    gates[1].resolve(library('off')); await old;
    await turn();
    assert.equal(store.getState().imageModels.selected, 'q6');
    assert.equal(store.getState().busy['image:select'], true);
    gates[2].resolve(library('q6')); await current;
    assert.equal(store.getState().busy['image:select'], false);
    assert.equal(store.imageSelectionReady('q6'),true);
});

test('persisted Host selection cannot be superseded by a late older write before apply', async () => {
    const { store, gates } = await fixture();
    const old = store.selectImageModel('off');
    await turn();
    const current = store.selectImageModel('q6');
    await turn();
    if (gates[2]) {
        // Independent writes reproduce the original late-older Host mutation.
        gates[2].commit(); await current;
        gates[1].commit(); await old;
    } else {
        gates[1].commit(); await old;
        await turn();
        gates[2].commit(); await current;
    }
    await store.applyImageModel('q6');
    assert.equal(globalThis.__hostSelected, 'q6');
    assert.equal(globalThis.__hostApplied, 'q6');
});

test('failed latest off intent cannot apply previous Host q6 and explicit recovery confirms off', async () => {
    const { store, gates } = await fixture();
    const reason = new Error('selection write failed before reaching Host');
    const failed = store.selectImageModel('off');
    const rejected = assert.rejects(failed, error => error === reason);
    await turn(); gates[1].reject(reason); await rejected;
    assert.equal(globalThis.__hostSelected, 'q6');
    await assert.rejects(store.applyImageModel('off'), error => error === reason || error.message === reason.message);
    assert.equal(globalThis.__imageApplies, 0);
    const recovery = store.selectImageModel('off');
    await turn(); gates[2].commit(); await recovery;
    await store.applyImageModel('off');
    assert.equal(globalThis.__hostApplied, 'off');
    assert.equal(globalThis.__imageApplies, 1);
});

test('an explicit apply queued before a superseding selection cannot apply a different intent', async () => {
    const {store,gates}=await fixture();
    const application=store.applyImageModel('q6');
    const rejected=assert.rejects(application,/Confirm/);
    const next=store.selectImageModel('off');
    await rejected;
    await turn();
    assert.equal(globalThis.__imageApplies,0);
    gates[1].commit();await next;
    await store.applyImageModel('off');
    assert.equal(globalThis.__hostApplied,'off');
});

test('ordered writes keep old rejection observable without preventing the latest selection', async () => {
    const { store, gates, library } = await fixture();
    const old = store.selectImageModel('off');
    await turn();
    const current = store.selectImageModel('q6');
    gates[1].resolve(library('off')); await old;
    await turn();
    gates[2].resolve(library('q6')); await current;
    assert.equal(store.getState().imageModels.selected, 'q6');
    assert.equal(store.getState().busy['image:select'], false);
    const failing = store.selectImageModel('q4');
    const rejection = assert.rejects(failing, /stale selection/);
    await turn();
    const latest = store.selectImageModel('off');
    gates[3].reject(new Error('stale selection')); await rejection;
    await turn();
    assert.equal(store.getState().imageModels.selected, 'off');
    assert.equal(store.getState().busy['image:select'], true);
    gates[4].resolve(library('off')); await latest;
    assert.equal(store.getState().imageModels.selected, 'off');
    assert.equal(store.getState().busy['image:select'], false);
});
