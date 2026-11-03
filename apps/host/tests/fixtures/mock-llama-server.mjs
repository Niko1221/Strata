// A stand-in for llama.cpp's llama-server in tests (tests/dev-llama.test.mjs): the same command line, /health (503 while "loading",
// then 200), /v1/models and a one-line /v1/chat/completions. MOCK_LLAMA=crash ends with an error line; MOCK_LLAMA=silent never listens.
import http from 'node:http';

const args = process.argv.slice(2);
const value = flag => { const i = args.indexOf(flag); return i >= 0 ? args[i + 1] : null; };
const port = Number(value('--port')), alias = value('--alias') ?? 'model';
const mode = process.env.MOCK_LLAMA ?? 'ok';

if (mode === 'crash') {
  console.error(`llama_model_load: error loading model: invalid magic in ${value('-m')}`);
  setTimeout(() => process.exit(1), 50);
} else if (mode === 'silent') {
  console.log('loading forever');
  setInterval(() => {}, 1000);
} else {
  const readyAt = Date.now() + 300;
  http.createServer((req, res) => {
    const send = (status, body) => { res.writeHead(status, { 'content-type': 'application/json' }); res.end(JSON.stringify(body)); };
    if (req.url === '/health') return Date.now() < readyAt ? send(503, { error: { message: 'Loading model' } }) : send(200, { status: 'ok' });
    if (req.url === '/v1/models') return send(200, { object: 'list', data: [{ id: alias, object: 'model' }] });
    if (req.url === '/v1/chat/completions') return send(200, { choices: [{ message: { role: 'assistant', content: 'mock answer' } }] });
    send(404, { error: 'not found' });
  }).listen(port, '127.0.0.1', () => console.log(`main: server is listening on http://127.0.0.1:${port}`));
}
