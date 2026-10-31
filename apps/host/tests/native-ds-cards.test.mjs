/** The app must not keep the one card a DeepSeek config of setup 0.2.8 and earlier names after setup's own pick (strata_control/gpu_pick.py). */
import test from 'node:test';
import assert from 'node:assert/strict';
import { setupPickedOneCard } from '../server/native-control.mjs';

const amd = (index, extra = {}) => ({ index, vendor: 'amd', vram_gb: 32, usable: true, ...extra });
const four = { gpus: [0, 1, 2, 3].map(i => amd(i)) };
const old = { arch: 'deepseek41', backend: 'hip', gpu: 0, gpus_asked: true };
const linux = process.platform !== 'win32';

test('setup\'s own one-card pick is not a choice (Linux, several cards)', { skip: !linux }, () => {
  assert.equal(setupPickedOneCard(old, four), true);
});
test('another card, a marker, roles, a list or another model are the person\'s', () => {
  assert.equal(setupPickedOneCard({ ...old, gpu: 2 }, four), false);
  assert.equal(setupPickedOneCard({ ...old, gpus_by: 'person' }, four), false);
  assert.equal(setupPickedOneCard({ ...old, gpu_roles: { 0: 'main' } }, four), false);
  assert.equal(setupPickedOneCard({ ...old, gpu: [0, 1] }, four), false);
  assert.equal(setupPickedOneCard({ ...old, arch: 'qwen4exp' }, four), false);
  assert.equal(setupPickedOneCard({ ...old, backend: 'cuda' }, four), false);
});
test('one usable card, or an APU beside one discrete card, leaves nothing to widen', () => {
  assert.equal(setupPickedOneCard(old, { gpus: [amd(0)] }), false);
  assert.equal(setupPickedOneCard({ ...old, gpu: 1 }, { gpus: [amd(0, { uma: true, vram_gb: 96 }), amd(1)] }), false);
});
