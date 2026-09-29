// Bounded real Pi A -> B -> A coding workflow, used by agent_cache_benchmark.py.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';

const [packageDir, cwd, agentDir, output] = process.argv.slice(2);
const {createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager} =
  await import(pathToFileURL(path.join(packageDir, 'dist/index.js')));
const runtime = await ModelRuntime.create({authPath: path.join(agentDir, 'auth.json'),
  modelsPath: path.join(agentDir, 'models.json'), allowModelNetwork: false});
const model = runtime.getModel('strata-benchmark', 'qwen3.8-flash-next-iq3_s');
if (!model) throw new Error('Benchmark model unavailable');
const events = fs.createWriteStream(output + '.events.jsonl');
const sessions = [];
const phases = [];
const budgets = new WeakMap();
const system = 'You are implementing a small Python standard-library project. Work only in the current directory. '
  + 'Do not access the network, install dependencies, or read parent directories. Use at most 12 tool calls per task. '
  + 'Keep tool outputs below 16000 characters and shell commands below 15 seconds. Implement the requested files, '
  + 'run focused checks, then give a short result. Do not build unrelated features.';
async function makeSession() {
  const budget = {calls: 0, limited: false};
  const settingsManager = SettingsManager.inMemory({compaction: {enabled: false}, retry: {enabled: false}});
  const loader = new DefaultResourceLoader({cwd, agentDir, settingsManager, noExtensions: true,
    noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true,
    extensionFactories: [pi => {
      pi.on('before_provider_request', event => {
        const payload = {...event.payload, temperature: 0, max_tokens: 2048,
          chat_template_kwargs: {enable_thinking: false}};
        delete payload.max_completion_tokens;
        return payload;
      });
      pi.on('tool_call', event => {
        if (++budget.calls > 12) {
          budget.limited = true;
          return {block: true, terminate: true, reason: 'Benchmark tool-call limit'};
        }
        if (event.toolName === 'bash') event.input.timeout = Math.min(event.input.timeout || 15, 15);
      });
      pi.on('tool_result', event => {
        let left = 16000;
        return {content: event.content.map(part => {
          if (part.type !== 'text') return part;
          const marker = '\n[benchmark output limit]';
          const text = part.text.length <= left ? part.text
            : left >= marker.length ? part.text.slice(0, left - marker.length) + marker : '';
          left -= text.length;
          return {...part, text};
        })};
      });
    }],
    systemPromptOverride: () => system});
  await loader.reload();
  const {session} = await createAgentSession({cwd, agentDir, model, modelRuntime: runtime,
    thinkingLevel: 'off', tools: ['read', 'write', 'edit', 'bash'], resourceLoader: loader,
    sessionManager: SessionManager.inMemory(), settingsManager});
  sessions.push(session);
  budgets.set(session, budget);
  return session;
}
async function phase(session, name, prompt) {
  const budget = budgets.get(session);
  budget.calls = 0; budget.limited = false;
  const record = {name, tool_calls: 0, tool_ms: 0, turns: 0, limited: false};
  const activeTools = new Map();
  const started = performance.now();
  const unsubscribe = session.subscribe(event => {
    events.write(JSON.stringify({phase: name, elapsed_ms: performance.now() - started, event}) + '\n');
    if (event.type === 'turn_start') record.turns++;
    if (event.type === 'tool_execution_start') {
      record.tool_calls++;
      activeTools.set(event.toolCallId, performance.now());
    }
    if (event.type === 'tool_execution_end') {
      record.tool_ms += performance.now() - (activeTools.get(event.toolCallId) ?? performance.now());
    }
    if (record.tool_calls > 12 || record.turns > 16) {
      record.limited = true;
      void session.abort();
    }
  });
  const timeout = setTimeout(() => { record.limited = true; void session.abort(); }, 240000);
  try {
    await session.prompt(prompt);
    const last = session.messages.filter(m => m.role === 'assistant').at(-1);
    if (record.limited || budget.limited || !last || ['error', 'aborted'].includes(last.stopReason)) {
      throw new Error(`Phase ${name} failed: ${last?.errorMessage ?? last?.stopReason ?? 'limit'}`);
    }
    record.answer = last.content;
  } finally {
    clearTimeout(timeout); unsubscribe(); record.wall_ms = performance.now() - started;
    phases.push(record);
    fs.writeFileSync(output, JSON.stringify({phases}, null, 2) + '\n');
  }
}
try {
  const controller = await makeSession();
  const worker = await makeSession();
  await phase(controller, 'controller-implement',
    'Implement report.py summarize(text) using SPEC.md. Read the complete sample.csv (a realistic input corpus) '
    + 'before implementing. Implement only the library for now; another developer will implement cli.py. '
    + 'You may add small tests. Summarize what you implemented.');
  await phase(worker, 'worker-cli',
    'You are responsible for cli.py. Read SPEC.md and report.py, implement the specified CLI, and verify both '
    + 'file and stdin inputs. Do not modify report.py. Keep your answer brief.');
  await phase(controller, 'controller-integrate',
    'The CLI developer has finished. Read cli.py and verify the integrated project against SPEC.md, especially '
    + 'quoted commas, timezone-to-UTC day conversion, last-valid duplicate IDs, invalid rows, empty input, '
    + 'Decimal totals, stdin, and missing-file exit status. Fix any problems in report.py or cli.py. '
    + 'Run focused checks and report the result.');
} finally {
  for (const session of sessions) session.dispose();
  events.end();
}
