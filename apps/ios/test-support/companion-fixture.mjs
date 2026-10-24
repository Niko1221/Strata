import http from 'node:http';
import { spawn } from 'node:child_process';
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

// An explicit test host is always supplied. This fixture cannot discover or call
// a person's running Strata app, even when one is open on the same Mac.
const moduleURL = process.env.STRATA_TEST_BRIDGE_MODULE
  ? pathToFileURL(resolve(process.env.STRATA_TEST_BRIDGE_MODULE))
  : new URL('../../companion-bridge/bridge.mjs', import.meta.url);
const { createBridge } = await import(moduleURL.href);
const root = mkdtempSync(join(tmpdir(), 'strata-ios-ui-'));
chmodSync(root, 0o700);

const calls = [];
const intro = 'Earlier Mac history is available on the phone.';
const follow = '\n\nNow I am checking the workspace.';
const reasoning = 'Inspect the existing conversation.\nI am reviewing the Mac workspace.';
const progressBlocks = (toolIds = []) => [
  { kind: 'text', text: intro },
  { kind: 'group', id: 'fixture-group-earlier', line: 'Ran 3 commands', closed: true, toolIds: [] },
  { kind: 'text', text: follow },
  { kind: 'group', id: 'fixture-group-live', line: 'Ran 3 commands, edited a file', closed: false, added: 8, removed: 2, toolIds },
];
const assistant = {
  id: 'fixture-assistant', role: 'assistant', content: intro + follow,
  reasoning, reasoningMs: 2400, streaming: false, blocks: progressBlocks(), tools: [],
};
const conversation = {
  id: 'fixture-mac-research', title: 'Mac browser research', updatedAt: new Date().toISOString(),
  workspace: join(root, 'Strata UI fixture'), modelName: 'Fixture model', mode: 'manual', active: false, status: 'Ready',
  messages: [{ id: 'fixture-user', role: 'user', content: 'Research this topic on the Mac.' }, assistant],
};
const pending = [];
const remoteTargetId = 'remote-aaaaaaaaaaaaaaaaaaaaaaaa';
const downloadJobs = new Map([['local', []], [remoteTargetId, []]]);
const downloadPlans = new Map();
let nextPlan = 0;
let nextDownload = 0;
let newestDownload;
let managementUnsupported = false;
const targetName = target => target === 'local' ? 'Fixture Mac' : 'Remote Strata';
const catalogModels = target => [{ id: 'catalog-native', name: 'Synthetic native model', sizeGb: 12,
  state: downloadJobs.get(target).some(job => job.modelId === 'catalog-native' && job.state === 'done') ? 'installed' : 'available',
  readPictures: true, detail: 'Synthetic catalog item. No model files are downloaded.' }];
const latestDownload = () => newestDownload;
const detail = () => structuredClone({ conversation, pending });
const tool = (id, state) => ({
  id, name: 'mcp__browser__browser_navigate', state, verb: 'Opened a page',
  target: 'https://example.com/research', preview: 'Read the example research page',
  effect: 'Use the browser on the Mac', scope: 'connected',
  result: 'Page title: Example research. Synthetic browser activity for UI verification.', live: true,
  ask: { why: 'Read this page in the Mac browser', detail: 'Synthetic approval request for a test fixture.' },
});
const host = {
  async json(path, options = {}) {
    calls.push({ path, method: options.method ?? 'GET', body: options.body ?? null });
    if (path === '/api/models') return { variants: [], loaded: null };
    if (path === '/api/native') return { instances: [], catalog: { families: [] } };
    if (path === '/api/monitor') return { phase: 'idle', outputTps: 12, hardware: { ramUsedGb: 8, ramTotalGb: 48 } };
    if (path === '/api/serving') return { requests: [] };
    if (path === '/api/model-providers') return { connections: [{ id: 'fixture-provider', name: 'Fixture Mac', models: [{ id: 'fixture-model', name: 'Fixture model' }] }] };
    if (path === '/api/model-management/targets') {
      if (managementUnsupported) throw Object.assign(new Error('Unsupported management endpoint'), { status: 404 });
      return { targets: [{ id: 'local', name: 'Fixture Mac', kind: 'local' }, { id: remoteTargetId, name: 'Remote Strata', kind: 'remote' }] };
    }
    const management = /^\/api\/model-management\/targets\/(local|remote-[a-f0-9]{24})(?:\/(plan|downloads)(?:\/([a-zA-Z0-9_-]+)\/cancel)?)?$/.exec(path);
    if (management) {
      const [, target, action, jobId] = management;
      if (!downloadJobs.has(target)) throw new Error('Unknown synthetic target');
      if (!action) return structuredClone({ supported: true, targetName: targetName(target), capabilities: { downloads: true, imports: false }, models: catalogModels(target), jobs: downloadJobs.get(target) });
      if (action === 'plan') {
        if (!catalogModels(target).some(model => model.id === options.body?.modelId) || typeof options.body?.readPictures !== 'boolean') throw new Error('Invalid typed synthetic plan request');
        const plan = { planId: `fixture-plan-${++nextPlan}`, targetName: targetName(target), modelName: 'Synthetic native model', downloadGb: 12, diskFreeGb: 100, diskNeededGb: 14,
          readPictures: options.body.readPictures, expiresAt: Date.now() + 60_000,
          warnings: ['Running native models will be unloaded before preparing files.', 'The engine may be downloaded or updated.', 'The model will not be loaded automatically.'] };
        downloadPlans.set(plan.planId, { ...plan, target, modelId: options.body.modelId });
        return plan;
      }
      if (jobId) {
        const job = downloadJobs.get(target).find(job => job.id === jobId);
        if (!job || !job.canCancel) throw new Error('Unknown or unowned synthetic job');
        job.state = 'cancelled'; job.summary = 'Download stopped. Completed files are kept.'; job.canCancel = false;
        return structuredClone({ job });
      }
      const plan = downloadPlans.get(options.body?.planId);
      if (!plan || plan.target !== target || plan.expiresAt <= Date.now()) throw new Error('Missing or expired synthetic plan');
      downloadPlans.delete(plan.planId);
      const job = { id: `fixture-download-${++nextDownload}`, modelId: plan.modelId, name: 'Synthetic native model', kind: 'download', state: 'running', summary: 'Downloading synthetic model files', percent: 10, doneGb: 1.2, totalGb: 12, canCancel: true };
      downloadJobs.get(target).unshift(job); newestDownload = job;
      return structuredClone({ job });
    }
    if (path === '/api/companion/conversations') return options.method === 'POST' ? detail() : { conversations: [conversation], activeChatId: conversation.id };
    if (path === `/api/companion/conversations/${conversation.id}`) return detail();
    if (path === `/api/companion/conversations/${conversation.id}/messages`) {
      conversation.messages.push(
        { id: 'fixture-continued-user', role: 'user', content: options.body.message },
        { id: 'fixture-continued-assistant', role: 'assistant', content: 'SAME_MAC_SESSION_CONTINUED', streaming: true, tools: [] },
      );
      conversation.active = true; conversation.status = 'Working';
      return detail();
    }
    if (path === `/api/companion/conversations/${conversation.id}/approval`) {
      const current = pending.find(action => action.id === options.body.toolId);
      if (!current) throw new Error('Stale fixture approval');
      current.state = options.body.decision === 'approve' ? 'done' : 'denied';
      current.approved = options.body.decision === 'approve';
      pending.splice(pending.indexOf(current), 1);
      return detail();
    }
    if (path === `/api/companion/conversations/${conversation.id}/stop`) {
      conversation.active = false; conversation.status = 'Stopped';
      for (const message of conversation.messages) message.streaming = false;
      return detail();
    }
    throw new Error('Unexpected synthetic host path: ' + path);
  },
};

let bridge;
let control;
let testProcess;
let closing;
async function close() {
  closing ??= (async () => {
    try {
      if (testProcess && testProcess.exitCode === null) testProcess.kill('SIGTERM');
      if (control?.listening) await new Promise(resolve => control.close(resolve));
      await bridge?.close();
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
  })();
  return closing;
}
for (const [signal, status] of [['SIGINT', 130], ['SIGTERM', 143]]) {
  process.once(signal, async () => { await close(); process.exit(status); });
}

try {
  bridge = await createBridge({ directory: join(root, 'bridge'), host, listen: '127.0.0.1', port: 0, adminPort: 0, advertiseHost: '127.0.0.1', qr: false });
  control = http.createServer((request, response) => {
    try {
      const action = request.url;
      if (request.method === 'POST') {
        switch (action) {
        case '/state':
          break;
        case '/download-progress': {
          const job = latestDownload();
          if (!job) throw new Error('Start a synthetic download first');
          job.percent = 65; job.doneGb = 7.8; job.summary = 'Downloading picture reader';
          break;
        }
        case '/download-error': {
          const job = latestDownload();
          if (!job) throw new Error('Start a synthetic download first');
          job.state = 'error'; job.error = 'Synthetic network interruption. Completed files can be reused.'; job.canCancel = false;
          break;
        }
        case '/download-complete': {
          const job = latestDownload();
          if (!job) throw new Error('Start a synthetic download first');
          job.state = 'done'; job.summary = 'Model files are ready on the computer.'; job.percent = 100; job.doneGb = 12; job.canCancel = false; delete job.error;
          break;
        }
        case '/management-unsupported':
          managementUnsupported = true;
          break;
        case '/thinking':
          conversation.active = true; conversation.status = 'Thinking';
          assistant.content = ''; assistant.blocks = []; assistant.streaming = true; delete assistant.reasoningMs;
          break;
        case '/progress':
          conversation.active = true; conversation.status = 'Working';
          assistant.content = intro + follow; assistant.blocks = progressBlocks(); assistant.reasoningMs = 2400; assistant.streaming = true;
          break;
        case '/browser':
          conversation.active = true; conversation.status = 'Browsing the web';
          assistant.streaming = true; assistant.tools = [tool('browser-live', 'running')]; assistant.blocks = progressBlocks(['browser-live']);
          break;
        case '/allow':
        case '/deny': {
          const approval = tool(action === '/allow' ? 'helper-session:approve-action' : 'helper-session:deny-action', 'proposed');
          assistant.tools.push(approval); pending.push(approval);
          break;
        }
        case '/complete':
          conversation.active = false; conversation.status = 'Ready';
          if (assistant.tools[0]) assistant.tools[0].state = 'done';
          assistant.streaming = false; assistant.reasoningMs = 6500;
          assistant.blocks.find(block => block.id === 'fixture-group-live').closed = true;
          break;
        default: response.writeHead(404).end(); return;
        }
      } else if (request.method !== 'GET' || action !== '/state') {
        response.writeHead(404).end(); return;
      }
      response.setHeader('Content-Type', 'application/json');
      response.end(JSON.stringify({ conversation, pending, calls, downloadJobs: Object.fromEntries(downloadJobs), planCount: nextPlan, downloadCount: nextDownload }));
    } catch (error) {
      response.writeHead(500, { 'Content-Type': 'application/json' });
      response.end(JSON.stringify({ error: error.message }));
    }
  });
  await new Promise((resolve, reject) => { control.once('error', reject); control.listen(0, '127.0.0.1', resolve); });
  const connection = { invitation: bridge.invitation(), control: `http://127.0.0.1:${control.address().port}` };
  if (['--test', '--test-downloads'].includes(process.argv[2])) {
    const destination = process.argv[3];
    if (!destination) throw new Error('Supply an installed simulator destination after --test.');
    const platforms = destination.split(',').map(value => value.trim()).filter(value => value.startsWith('platform='));
    if (platforms.length !== 1 || platforms[0] !== 'platform=iOS Simulator') {
      throw new Error('The UI fixture requires an explicit platform=iOS Simulator destination. Physical devices are not supported.');
    }
    const project = fileURLToPath(new URL('../StrataMobile.xcodeproj', import.meta.url));
    const derivedData = process.env.STRATA_TEST_DERIVED_DATA ?? fileURLToPath(new URL('../build', import.meta.url));
    const status = await new Promise((resolve, reject) => {
      testProcess = spawn('xcodebuild', [
        '-project', project, '-scheme', 'StrataMobile', '-destination', destination,
        '-derivedDataPath', derivedData,
        '-only-testing:StrataMobileUITests/CompanionSmokeTests/' + (process.argv[2] === '--test-downloads' ? 'testModelDownloadReviewProgressCancelRetryAndTargetSelection' : 'testSharedMacHistoryLiveBrowserApprovalAndContinuation'),
        '-parallel-testing-enabled', 'NO', 'test',
      ], { stdio: 'inherit', env: { ...process.env,
        TEST_RUNNER_STRATA_TEST_INVITATION: connection.invitation,
        TEST_RUNNER_STRATA_SHARED_FIXTURE_CONTROL: connection.control,
        TEST_RUNNER_STRATA_MODEL_MANAGEMENT_FIXTURE_CONTROL: connection.control,
      } });
      testProcess.once('error', reject);
      testProcess.once('exit', code => resolve(code ?? 1));
    });
    process.exitCode = status;
    await close();
  } else {
    if (process.argv.length > 2) throw new Error('Usage: companion-fixture.mjs [--test|--test-downloads <simulator destination>]');
    const connectionFile = join(root, 'connection.json');
    writeFileSync(connectionFile, JSON.stringify(connection), { mode: 0o600 });
    console.log(JSON.stringify({ connectionFile }));
  }
} catch (error) {
  await close();
  console.error(error.message);
  process.exitCode = 1;
}
