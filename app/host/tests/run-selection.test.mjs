import test from 'node:test';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';

const built = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/run-selection.ts', import.meta.url))], bundle: true, write: false, format: 'esm', platform: 'node' });
const { resolveLocalRunSelection: resolve, runModelSettings, readLastLocalModel, rememberRunSelection, LAST_LOCAL_MODEL_KEY, RUN_VARIANT_KEY } = await import(`data:text/javascript;base64,${Buffer.from(built.outputFiles[0].text).toString('base64')}`);
const q36 = { id: 'q36', family: 'Local models', quant: 'Q4_K_M', local: {}, contextMax: 131072 };
const hikari = { id: 'hikari', family: 'Local models', quant: 'PQ2_0', local: {}, contextMax: 32768 };
const native = { id: 'native', family: 'Qwen', quant: 'Q2_0', contextMax: 131072 };
const variants = [q36, hikari, native];
const engine = { expert: [{ id: 'runtime.backend', recommended: 'Automatic' }, { id: 'gpu.cpuMoeLayers', recommended: 0 }, { id: 'ctx.length', recommended: '128K' }] };
const hardware = { cpu: { cores: 6 } };
const config = { schema: 'strata.engine/1', beginner: { contextLength: '128k' }, expert: { 'ctx.length': '128K', 'gpu.cpuMoeLayers': 28, 'runtime.backend': 'Automatic' } };
const profiles = { schema: 1, defaults: { 'pc\nq36': 'daily' }, profiles: [{ id: 'daily', variantId: 'q36', hardware: 'pc', config, imageModel: 'off' }] };

test('registry readiness precedes selection; explicit links beat the last viewed model', () => {
    assert.deepEqual(resolve(null, 'q36', 'hikari'), { status: 'waiting' });
    assert.deepEqual(resolve(variants, 'q36', 'hikari'), { status: 'selected', id: 'q36' });
    assert.deepEqual(resolve(variants, 'hikari', 'q36'), { status: 'selected', id: 'hikari' });
    assert.deepEqual(resolve(variants, null, 'q36'), { status: 'selected', id: 'q36' });
    assert.notEqual(LAST_LOCAL_MODEL_KEY, RUN_VARIANT_KEY, 'restart choice is distinct from session draft ownership');
});

test('unknown, empty, removed and nonlocal IDs never silently select another model', () => {
    for (const id of ['removed', '', 'native']) assert.deepEqual(resolve(variants, id, 'q36'), { status: 'missing', id });
    for (const last of [null, 'removed', 'native']) assert.deepEqual(resolve(variants, null, last), { status: 'none' });
    assert.deepEqual(resolve([], 'q36', 'hikari'), { status: 'missing', id: 'q36' });
});

test('the last local selection survives a browser session restart independently of engine/current model', () => {
    const storage = () => {
        const data = new Map();
        return { getItem: key => data.get(key) ?? null, setItem: (key, value) => data.set(key, value) };
    };
    const local = storage(), session = storage();
    session.setItem(RUN_VARIANT_KEY, 'hikari');
    assert.equal(readLastLocalModel(local, session), 'hikari', 'migrate an existing session choice');
    rememberRunSelection(q36, local, session);
    rememberRunSelection(native, local, session);
    assert.equal(session.getItem(RUN_VARIANT_KEY), 'native');
    assert.equal(local.getItem(LAST_LOCAL_MODEL_KEY), 'q36');
    const restarted = storage();
    assert.deepEqual(resolve(variants, null, readLastLocalModel(local, restarted)), { status: 'selected', id: 'q36' });
    assert.deepEqual(resolve(variants, 'hikari', readLastLocalModel(local, restarted)), { status: 'selected', id: 'hikari' });
});

test('restored route replacement notifies routing without creating a Back/Forward loop', async t => {
    const source = await build({ entryPoints: [fileURLToPath(new URL('../src/lib/router.ts', import.meta.url))], bundle: true, write: false, platform: 'node', format: 'esm', alias: { react: fileURLToPath(new URL('./helpers/react-stub.mjs', import.meta.url)) } });
    const { replaceRoute, parseLocation } = await import(`data:text/javascript;base64,${Buffer.from(source.outputFiles[0].text).toString('base64')}`);
    const old = globalThis.window;
    t.after(() => { globalThis.window = old; });
    let replacements = 0, notifications = 0;
    globalThis.window = {
        location: { hash: '#/models/local' },
        history: { replaceState(_state, _title, hash) { replacements++; window.location.hash = hash; } },
        dispatchEvent(event) { assert.equal(event.type, 'hashchange'); notifications++; },
    };
    replaceRoute('models', 'local', { variant: 'q36' });
    assert.equal(parseLocation(window.location.hash).query.get('variant'), 'q36');
    replaceRoute('models', 'local', { variant: 'q36' });
    assert.equal(replacements, 1);
    assert.equal(notifications, 1);
});

test('Q36 / Hikari profile switching retains exact identity, backend and saved precision', () => {
    const q = runModelSettings(q36, engine, hardware, profiles, 'pc');
    const h = runModelSettings(hikari, engine, hardware, profiles, 'pc');
    assert.equal(q.profile, 'daily');
    assert.deepEqual(JSON.parse(q.draft), config);
    assert.equal(JSON.parse(h.draft).expert['runtime.backend'], 'PrismML');
    assert.equal(JSON.parse(h.draft).beginner.contextLength, '32k');
    assert.equal(JSON.parse(h.draft).expert['gpu.cpuMoeLayers'], 0);
    assert.equal(q36.quant, 'Q4_K_M');
    assert.deepEqual(profiles.profiles[0].config, config, 'switching never rewrites a saved profile');
});

test('back/forward uses per-model unsaved drafts even within the Local models family', () => {
    const drafts = new Map();
    const q = runModelSettings(q36, engine, hardware, profiles, 'pc');
    const edited = JSON.parse(q.draft); edited.expert['gpu.cpuMoeLayers'] = 17;
    drafts.set(q36.id, { ...q, draft: JSON.stringify(edited) });
    const h = runModelSettings(hikari, engine, hardware, profiles, 'pc');
    drafts.set(hikari.id, { ...h, draft: '{unfinished', image: 'q6' });
    for (const id of ['q36', 'hikari', 'q36']) {
        const model = variants.find(v => v.id === resolve(variants, id, null).id);
        const restored = runModelSettings(model, engine, hardware, profiles, 'pc', drafts.get(id));
        assert.equal(restored.profile, id === 'q36' ? 'daily' : 'current');
        if (id === 'q36') assert.equal(JSON.parse(restored.draft).expert['gpu.cpuMoeLayers'], 17);
        else assert.deepEqual(restored, drafts.get(id));
    }
    assert.equal(profiles.profiles[0].config.expert['gpu.cpuMoeLayers'], 28);
});

test('a default pointing at another identity or computer cannot supply its settings', () => {
    for (const model of [hikari, native]) {
        const corrupt = { ...profiles, defaults: { [`pc\n${model.id}`]: 'daily' } };
        assert.equal(runModelSettings(model, engine, hardware, corrupt, 'pc').profile, 'current');
    }
    assert.equal(runModelSettings(q36, engine, hardware, { ...profiles, defaults: { 'other\nq36': 'daily' } }, 'other').profile, 'current');
});
