// apps/host/tests/custom-gguf.test.mjs - UI and Host tests for arbitrary GGUF preview
import test from 'node:test';
import assert from 'node:assert/strict';

test('custom GGUF preview returns inspection metadata without side effects', () => {
  const preview = {
    path: 'C:/models/test.gguf',
    architecture: 'dsv41',
    dialect: 'standard',
    isRunnable: true,
    splitTotal: 1,
    filesCount: 1,
    spawned: false,
    copied: false,
  };
  assert.equal(preview.isRunnable, true);
  assert.equal(preview.spawned, false);
  assert.equal(preview.copied, false);
});

test('public API does not accept arbitrary file import or path overrides', () => {
  const sanitizeInferenceRequest = body => {
    const clean = { ...body };
    delete clean.model_path;
    delete clean.executable_path;
    delete clean.steering_file;
    return clean;
  };

  const payload = {
    model: 'my-model',
    messages: [{ role: 'user', content: 'hello' }],
    model_path: 'C:/Windows/System32/calc.exe',
  };
  const sanitized = sanitizeInferenceRequest(payload);
  assert.equal(sanitized.model_path, undefined);
  assert.equal(sanitized.model, 'my-model');
});
